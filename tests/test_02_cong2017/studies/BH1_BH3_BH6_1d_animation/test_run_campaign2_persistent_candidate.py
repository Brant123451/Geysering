from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import inspect
import math

import numpy as np
import pytest

import run_campaign2_persistent_candidate as candidate_module
from campaign2_shared_contract import APPARATUS, SHARED_CLOSURE
from campaign2_tee_riemann import (
    GasTrace,
    LiquidBranchTrace,
    solve_gas_tee,
    solve_liquid_tee,
    solve_liquid_tee_with_blocked_riser,
)
from campaign2_vertical_twofluid_kernel import (
    VerticalTwoFluidState,
    atmospheric_empty_state,
    isothermal_common_pressure_faces,
    lower_material_front_geometric_timestep_limit,
    lower_material_front_star_state,
    mixture_hydrostatic_pressure_faces,
)
from case1_persistent_coupling import (
    TeeTransaction,
    transaction_from_tee_solutions,
)
from run_campaign2_persistent_candidate import (
    CandidateNumerics,
    CoupledTimestepLimitExceeded,
    PersistentCampaign2Candidate,
    SMOKE_END_LIMIT_S,
)


SELECTED_DIAMETERS_M = (0.016, 0.026, 0.041)


def _freeze_mutable_value(value):
    """Return an exact, equality-safe representation of nested solver state."""

    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            contiguous.dtype.str,
            contiguous.shape,
            contiguous.tobytes(),
        )
    if isinstance(value, np.generic):
        return _freeze_mutable_value(value.item())
    if isinstance(value, float):
        return ("float", value.hex())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _freeze_mutable_value(getattr(value, field.name)),
                )
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _freeze_mutable_value(key),
                    _freeze_mutable_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_mutable_value(item) for item in value)
    return value


def _candidate_mutable_fingerprint(
    candidate: PersistentCampaign2Candidate,
):
    """Capture every step-mutable field without relying on object identity."""

    horizontal = candidate.horizontal_owner
    boundary = horizontal.reservoir_boundary

    def mutable_attributes(obj, structural):
        return {
            name: value
            for name, value in vars(obj).items()
            if name not in structural and not callable(value)
        }

    return _freeze_mutable_value(
        {
            "candidate": mutable_attributes(
                candidate,
                {
                    "riser_diameter_m",
                    "numerics",
                    "horizontal_owner",
                    "vertical_kernel",
                    "global_budget",
                },
            ),
            "horizontal_owner": mutable_attributes(
                horizontal,
                {"solver", "reservoir_boundary"},
            ),
            "reservoir_boundary": (
                None
                if boundary is None
                else mutable_attributes(boundary, {"solver"})
            ),
            "global_budget": mutable_attributes(
                candidate.global_budget,
                set(),
            ),
            "derived_time_s": candidate.time_s,
        }
    )


def _without_diameter(contract: dict[str, object]) -> dict[str, object]:
    result = dict(contract)
    result.pop("riser_diameter_m")
    vertical = dict(result["vertical_parameters"])
    vertical.pop("diameter_m")
    result["vertical_parameters"] = vertical
    return result


def _install_thin_upper_cut_cell(
    candidate: PersistentCampaign2Candidate,
    *,
    liquid_area_m2: float,
) -> None:
    """Replace the first atmospheric cell by one conservative cut cell."""

    state = candidate.vertical_state
    parameters = candidate.parameters
    area = parameters.full_area_m2
    liquid_area = float(liquid_area_m2)
    assert parameters.area_tolerance_m2 < liquid_area < area
    interface_cell = next(
        cell for cell, value in enumerate(state.Al) if value == 0.0
    )
    al = list(state.Al)
    ql = list(state.Ql)
    mg = list(state.Mg)
    jg = list(state.Jg)
    al[interface_cell] = liquid_area
    ql[interface_cell] = 0.0
    mg[interface_cell] *= (area - liquid_area) / area
    jg[interface_cell] = 0.0
    candidate.vertical_state = replace(
        state,
        Al=tuple(al),
        Ql=tuple(ql),
        Mg=tuple(mg),
        Jg=tuple(jg),
    )


def _install_finite_bottom_gas_pocket(
    candidate: PersistentCampaign2Candidate,
    *,
    liquid_fraction: float = 0.75,
    common_velocity_m_s: float = 0.0,
) -> None:
    """Create one restart-valid lower cut from conserved inventory only."""

    state = candidate.vertical_state
    parameters = candidate.parameters
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    fraction = float(liquid_fraction)
    assert 0.0 < fraction < 1.0
    pressure = candidate._vertical_pressure_faces()[0]
    rho_g = pressure / (
        parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    )
    al = list(state.Al)
    ql = list(state.Ql)
    mg = list(state.Mg)
    jg = list(state.Jg)
    old_liquid = math.fsum(state.Al) * dz
    old_gas = math.fsum(state.Mg)
    al[0] = fraction * area
    ql[0] = al[0] * common_velocity_m_s
    gas_volume = (area - al[0]) * dz
    mg[0] = rho_g * gas_volume
    jg[0] = mg[0] * common_velocity_m_s
    candidate.vertical_state = replace(
        state,
        Al=tuple(al),
        Ql=tuple(ql),
        Mg=tuple(mg),
        Jg=tuple(jg),
        lower_material_front_cell=0,
        lower_material_front_orientation="gas_below_liquid_above",
    )
    candidate.global_budget.initial_liquid_volume_m3 += (
        math.fsum(candidate.vertical_state.Al) * dz - old_liquid
    )
    candidate.global_budget.initial_gas_mass_kg += (
        math.fsum(candidate.vertical_state.Mg) - old_gas
    )


def _replace_preview_bottom_liquid_flow(
    candidate: PersistentCampaign2Candidate,
    bottom_liquid_flow_m3_s: float,
):
    """Return an uncommitted T preview carrying a prescribed net liquid flow."""

    preview = candidate._solve_current_tee_uncommitted()
    return replace(
        preview,
        transaction=replace(
            preview.transaction,
            west_liquid_flow_m3_s=float(bottom_liquid_flow_m3_s),
            east_liquid_flow_m3_s=0.0,
        ),
    )


def _rebase_vertical_checkpoint(
    candidate: PersistentCampaign2Candidate,
    state: VerticalTwoFluidState,
) -> None:
    """Install a declared constructed vertical checkpoint conservatively."""

    parameters = candidate.parameters
    dz = parameters.cell_length_m
    old_liquid = math.fsum(candidate.vertical_state.Al) * dz
    old_gas = math.fsum(candidate.vertical_state.Mg)
    candidate.vertical_state = state
    candidate.global_budget.initial_liquid_volume_m3 += (
        math.fsum(state.Al) * dz - old_liquid
    )
    candidate.global_budget.initial_gas_mass_kg += (
        math.fsum(state.Mg) - old_gas
    )


def _install_uniform_lower_front_checkpoint(
    candidate: PersistentCampaign2Candidate,
    *,
    front_cell: int,
    front_liquid_fraction: float,
    common_velocity_m_s: float,
    lower_gas_density_kg_m3: float | None = None,
    time_s: float | None = None,
) -> None:
    """Build one explicit stress checkpoint from conserved phase fields.

    This helper is test-only: it is not a trajectory generator and does not
    inspect a measured outcome.  Every resolved phase is assigned the same
    material velocity so the public driver can exercise its exact lower-front
    events without a hidden source or gas seed.
    """

    state = candidate.vertical_state
    parameters = candidate.parameters
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    fraction = float(front_liquid_fraction)
    assert 0.0 < fraction <= 1.0
    top_start = next(
        cell for cell, liquid_area in enumerate(state.Al) if liquid_area == 0.0
    )
    assert 0 <= front_cell < top_start
    if lower_gas_density_kg_m3 is None:
        bottom_pressure = candidate._vertical_pressure_faces(state)[0]
        lower_density = bottom_pressure / (
            parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
        )
    else:
        lower_density = float(lower_gas_density_kg_m3)
    assert math.isfinite(lower_density) and lower_density > 0.0

    liquid_area: list[float] = []
    liquid_flow: list[float] = []
    gas_mass: list[float] = []
    gas_momentum: list[float] = []
    velocity = float(common_velocity_m_s)
    for cell in range(parameters.cell_count):
        if cell < front_cell or cell >= top_start:
            al = 0.0
        elif cell == front_cell:
            al = fraction * area
        else:
            al = area
        if cell <= front_cell:
            mg = lower_density * (area - al) * dz
        elif cell >= top_start:
            mg = float(state.Mg[cell])
        else:
            mg = 0.0
        liquid_area.append(al)
        liquid_flow.append(al * velocity if al > 0.0 else 0.0)
        gas_mass.append(mg)
        gas_momentum.append(mg * velocity if mg > 0.0 else 0.0)

    checkpoint = replace(
        state,
        Al=tuple(liquid_area),
        Ql=tuple(liquid_flow),
        Mg=tuple(gas_mass),
        Jg=tuple(gas_momentum),
        time_s=state.time_s if time_s is None else float(time_s),
        lower_material_front_cell=front_cell,
        lower_material_front_orientation="gas_below_liquid_above",
    )
    _rebase_vertical_checkpoint(candidate, checkpoint)


def _install_horizontal_contact_checkpoint(
    candidate: PersistentCampaign2Candidate,
    *,
    time_s: float,
    pressure_abs_Pa: float,
) -> None:
    """Put the conserved Case-1 pocket nose exactly at the physical T face."""

    owner = candidate.horizontal_owner
    solver = owner.solver
    state = owner.state
    old_liquid = float(np.sum(state.area) * solver.dx)
    old_gas = float(state.gas.mass)
    interface_x = solver.physical_length - solver.physical_junction_face_x
    area = np.where(
        solver.x < interface_x,
        0.0,
        solver.section.full_area,
    )
    gas_volume = solver.section.full_area * interface_x
    pressure = float(pressure_abs_Pa)
    gas = replace(
        state.gas,
        reference_volume=gas_volume,
        reference_pressure_abs=pressure,
        volume=gas_volume,
        mass=(
            pressure
            * gas_volume
            / (solver.config.gas_constant * solver.config.temperature)
        ),
    )
    owner.state = replace(
        state,
        time=float(time_s),
        area=area,
        discharge=np.zeros_like(area),
        gas=gas,
        air_pressure_abs=pressure,
        interface_x=interface_x,
        interface_speed=0.0,
        wetting_front_x=interface_x,
    )
    candidate.global_budget.initial_liquid_volume_m3 += (
        float(np.sum(area) * solver.dx) - old_liquid
    )
    candidate.global_budget.initial_gas_mass_kg += gas.mass - old_gas


def _assert_record_budgets_close(record) -> None:
    assert abs(record.horizontal_tee_liquid_residual_m3) <= 2.0e-18
    assert abs(record.vertical_tee_liquid_residual_m3) <= 2.0e-18
    assert abs(record.horizontal_tee_gas_residual_kg) <= 2.0e-18
    assert abs(record.vertical_tee_gas_residual_kg) <= 2.0e-18
    assert abs(record.global_liquid_residual_m3) <= 1.0e-14
    assert abs(record.global_gas_residual_kg) <= 2.0e-15


def test_three_single_run_contracts_vary_only_riser_diameter() -> None:
    candidates = [
        PersistentCampaign2Candidate(diameter)
        for diameter in SELECTED_DIAMETERS_M
    ]
    contracts = [item.solver_facing_contract() for item in candidates]

    assert all(
        _without_diameter(contract) == _without_diameter(contracts[0])
        for contract in contracts[1:]
    )
    assert {
        contract["riser_diameter_m"] for contract in contracts
    } == set(SELECTED_DIAMETERS_M)
    assert {
        contract["vertical_parameters"]["diameter_m"]
        for contract in contracts
    } == set(SELECTED_DIAMETERS_M)

    for item in candidates:
        horizontal = item.horizontal_owner.solver
        boundary = item.horizontal_owner.reservoir_boundary
        assert (
            item.solver_facing_contract()["vertical_pressure_closure"]
            == "dynamic isothermal EOS common-pressure faces; "
            "kernel default after conservative transport"
        )
        assert horizontal.valve_open_time == APPARATUS.valve_open_time_s == 0.20
        assert boundary is not None
        assert boundary.reservoir_head_from_invert_m == 0.66
        initial_height = (
            math.fsum(item.vertical_state.Al)
            * item.parameters.cell_length_m
            / item.parameters.full_area_m2
        )
        assert initial_height == pytest.approx(0.61, abs=2.0e-15)


def test_shared_step_contract_is_a_maximum_with_cfl_at_most_point45() -> None:
    numerics = CandidateNumerics()

    assert numerics.max_dt_s == 2.0e-4
    assert numerics.physical_step_s == numerics.max_dt_s
    assert numerics.shared_cfl == 0.45
    with pytest.raises(ValueError, match="shared_cfl"):
        CandidateNumerics(shared_cfl=np.nextafter(0.45, math.inf))


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_shared_paper_materials_and_exact_horizontal_faces(
    diameter: float,
) -> None:
    candidate = PersistentCampaign2Candidate(diameter)
    horizontal = candidate.horizontal_owner.solver
    apparatus = candidate.solver_facing_contract()["apparatus"]

    expected_materials = {
        "liquid_density_kg_m3": 998.0,
        "liquid_dynamic_viscosity_Pa_s": 0.001003,
        "liquid_bulk_modulus_Pa": 2.2e9,
        "atmospheric_pressure_Pa": 101_325.0,
        "gravity_m_s2": 9.81,
        "gas_constant_J_kg_K": 287.05,
        "gas_temperature_K": 296.15,
        "air_molar_mass_kg_kmol": 28.965,
        "air_dynamic_viscosity_Pa_s": 1.81e-5,
        "surface_tension_N_m": 0.072,
    }
    assert {
        name: apparatus[name] for name in expected_materials
    } == expected_materials
    assert horizontal.config.liquid_density == 998.0
    assert horizontal.liquid_dynamic_viscosity_Pa_s == 0.001003
    assert horizontal.liquid_bulk_modulus_Pa == 2.2e9
    assert horizontal.config.atmospheric_pressure == 101_325.0
    assert horizontal.config.gravity == 9.81
    assert horizontal.config.gas_constant == 287.05
    assert horizontal.config.temperature == 296.15
    assert candidate.parameters.liquid_density_kg_m3 == 998.0
    assert candidate.parameters.atmospheric_pressure_Pa == 101_325.0
    assert candidate.parameters.gravity_m_s2 == 9.81
    assert candidate.parameters.gas_constant_J_kg_K == 287.05
    assert candidate.parameters.gas_temperature_K == 296.15

    assert candidate.numerics.horizontal_dx_m == 0.010
    assert horizontal.ncell == 659
    assert horizontal.dx == pytest.approx(0.010, abs=1.0e-16)

    mirrored_valve_x = APPARATUS.tunnel_length_m - APPARATUS.valve_x_m
    mirrored_tee_x = APPARATUS.tunnel_length_m - APPARATUS.riser_x_m
    assert horizontal.config.valve_x == pytest.approx(0.610, abs=1.0e-15)
    assert horizontal.config.vent_x == pytest.approx(3.120, abs=1.0e-15)
    assert round(mirrored_valve_x / horizontal.dx) == 61
    assert round(mirrored_tee_x / horizontal.dx) == 312
    assert 61 * horizontal.dx == pytest.approx(
        mirrored_valve_x,
        abs=1.0e-15,
    )
    assert 312 * horizontal.dx == pytest.approx(
        mirrored_tee_x,
        abs=1.0e-15,
    )
    assert horizontal.physical_junction_face_index == 347
    assert horizontal.physical_junction_face_x == pytest.approx(
        APPARATUS.riser_x_m,
        abs=1.0e-15,
    )


def test_driver_has_no_measured_outcome_or_case_label_feedback() -> None:
    source = inspect.getsource(candidate_module)
    forbidden = (
        "EXPERIMENT_GEYSER",
        "BH1",
        "BH3",
        "BH6",
        "classification_match",
        "target_outcome",
    )
    assert all(token not in source for token in forbidden)
    parameters = set(
        inspect.signature(PersistentCampaign2Candidate).parameters
    )
    assert parameters == {"riser_diameter_m"}


def test_physical_rim_crossing_uses_only_integrated_top_liquid_flux() -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    tolerance = SHARED_CLOSURE.top_liquid_outflow_tolerance_m3

    assert candidate.physical_top_liquid_outflow_m3 == 0.0
    assert candidate.physical_rim_crossed is False
    candidate.vertical_state = replace(
        candidate.vertical_state,
        cumulative_top_liquid_outflow_m3=tolerance,
    )
    assert candidate.physical_rim_crossed is False
    candidate.vertical_state = replace(
        candidate.vertical_state,
        cumulative_top_liquid_outflow_m3=np.nextafter(tolerance, math.inf),
    )
    assert candidate.physical_rim_crossed is True


def test_initial_coupled_step_uses_vertical_isothermal_acoustic_limit() -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    numerics = candidate.numerics
    horizontal = candidate.horizontal_owner.solver
    horizontal_limit = (
        horizontal.stable_timestep(candidate.horizontal_owner.state)
        * numerics.shared_cfl
        / horizontal.config.cfl
    )
    gas_limit = (
        numerics.shared_cfl
        * numerics.vertical_dz_m
        / candidate.parameters.isothermal_gas_sound_speed_m_s
    )
    liquid_limit = (
        numerics.shared_cfl
        * numerics.vertical_dz_m
        / SHARED_CLOSURE.wave_speed_m_s
    )

    assert candidate.stable_coupled_timestep() == min(
        numerics.max_dt_s,
        horizontal_limit,
        gas_limit,
        liquid_limit,
    )
    assert candidate.stable_coupled_timestep() == pytest.approx(
        1.543397774503213e-5,
        rel=0.0,
        abs=2.0e-20,
    )


def test_coupled_step_responds_to_current_gas_and_liquid_velocities() -> None:
    gas_candidate = PersistentCampaign2Candidate(0.026)
    gas_state = gas_candidate.vertical_state
    gas_cell = next(
        cell
        for cell, (al, mg) in enumerate(zip(gas_state.Al, gas_state.Mg))
        if mg > 0.0 and gas_candidate.parameters.full_area_m2 - al > 0.0
    )
    gas_momentum = list(gas_state.Jg)
    gas_momentum[gas_cell] = gas_state.Mg[gas_cell] * 5.2
    gas_candidate.vertical_state = replace(
        gas_state,
        Jg=tuple(gas_momentum),
    )
    imposed_gas_velocity = (
        gas_candidate.vertical_state.Jg[gas_cell]
        / gas_candidate.vertical_state.Mg[gas_cell]
    )
    expected_gas = (
        gas_candidate.numerics.shared_cfl
        * (
            gas_candidate.parameters.cell_length_m
            / (
                abs(imposed_gas_velocity)
                + gas_candidate.parameters.isothermal_gas_sound_speed_m_s
            )
        )
    )
    assert gas_candidate.stable_coupled_timestep() == expected_gas

    liquid_candidate = PersistentCampaign2Candidate(0.026)
    liquid_state = liquid_candidate.vertical_state
    liquid_cell = next(
        cell for cell, al in enumerate(liquid_state.Al) if al > 0.0
    )
    liquid_flow = list(liquid_state.Ql)
    liquid_flow[liquid_cell] = liquid_state.Al[liquid_cell] * 1000.0
    liquid_candidate.vertical_state = replace(
        liquid_state,
        Ql=tuple(liquid_flow),
    )
    imposed_liquid_velocity = (
        liquid_candidate.vertical_state.Ql[liquid_cell]
        / liquid_candidate.vertical_state.Al[liquid_cell]
    )
    expected_liquid = (
        liquid_candidate.numerics.shared_cfl
        * (
            liquid_candidate.parameters.cell_length_m
            / (
                abs(imposed_liquid_velocity)
                + SHARED_CLOSURE.wave_speed_m_s
            )
        )
    )
    assert liquid_candidate.stable_coupled_timestep() == expected_liquid


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_upper_surface_geometric_cfl_uses_remaining_cell_inventory_for_all_diameters(
    diameter: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = PersistentCampaign2Candidate(diameter)
    area = candidate.parameters.full_area_m2
    liquid_area = area * 1.0e-6
    _install_thin_upper_cut_cell(
        candidate,
        liquid_area_m2=liquid_area,
    )
    # The imposed rate has one shared 0.02 m/s interface speed.  Scaling Q by
    # riser area makes the expected geometric time independent of diameter.
    bottom_flow = -area * 0.02
    preview = _replace_preview_bottom_liquid_flow(candidate, bottom_flow)
    monkeypatch.setattr(
        candidate,
        "_solve_current_tee_uncommitted",
        lambda: preview,
    )

    one_cell_crossing_time = (
        liquid_area * candidate.parameters.cell_length_m / (-bottom_flow)
    )
    expected = one_cell_crossing_time
    stable = candidate.stable_coupled_timestep()

    assert stable == expected
    assert stable / one_cell_crossing_time == 1.0
    assert -bottom_flow * stable == pytest.approx(
        liquid_area * candidate.parameters.cell_length_m,
        rel=0.0,
        abs=2.0e-24,
    )


def test_upper_surface_retreat_geometric_cfl_nextafter_is_rejected_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    liquid_area = 1.0e-10
    bottom_flow = -1.0e-7
    _install_thin_upper_cut_cell(
        candidate,
        liquid_area_m2=liquid_area,
    )
    preview = _replace_preview_bottom_liquid_flow(candidate, bottom_flow)
    monkeypatch.setattr(
        candidate,
        "_solve_current_tee_uncommitted",
        lambda: preview,
    )
    expected = (
        liquid_area
        * candidate.parameters.cell_length_m
        / (-bottom_flow)
    )
    before = _candidate_mutable_fingerprint(candidate)

    assert candidate.stable_coupled_timestep() == expected
    assert _candidate_mutable_fingerprint(candidate) == before
    with pytest.raises(CoupledTimestepLimitExceeded) as caught:
        candidate.advance_one_step(np.nextafter(expected, math.inf))

    assert caught.value.stable_dt_s == expected
    assert _candidate_mutable_fingerprint(candidate) == before


def test_upper_surface_advance_geometric_cfl_nextafter_is_rejected_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    area = candidate.parameters.full_area_m2
    gas_area = 1.0e-10
    bottom_flow = 1.0e-7
    _install_thin_upper_cut_cell(
        candidate,
        liquid_area_m2=area - gas_area,
    )
    interface_cell = next(
        cell
        for cell, (al, mg) in enumerate(
            zip(candidate.vertical_state.Al, candidate.vertical_state.Mg)
        )
        if 0.0 < al < area and mg > 0.0
    )
    stored_gas_area = area - candidate.vertical_state.Al[interface_cell]
    preview = _replace_preview_bottom_liquid_flow(candidate, bottom_flow)
    monkeypatch.setattr(
        candidate,
        "_solve_current_tee_uncommitted",
        lambda: preview,
    )
    expected = (
        stored_gas_area
        * candidate.parameters.cell_length_m
        / bottom_flow
    )
    before = _candidate_mutable_fingerprint(candidate)

    assert candidate.stable_coupled_timestep() == pytest.approx(
        expected,
        rel=0.0,
        abs=2.0e-20,
    )
    assert _candidate_mutable_fingerprint(candidate) == before
    with pytest.raises(CoupledTimestepLimitExceeded):
        candidate.advance_one_step(
            np.nextafter(candidate.stable_coupled_timestep(), math.inf)
        )

    assert _candidate_mutable_fingerprint(candidate) == before


def test_upper_surface_geometric_cfl_is_inactive_only_at_zero_flow() -> None:
    candidate = PersistentCampaign2Candidate(0.026)

    assert math.isinf(
        candidate._upper_surface_geometric_timestep_limit(0.0)
    )
    positive = candidate._upper_surface_geometric_timestep_limit(1.0e-8)
    assert positive == pytest.approx(
        candidate.parameters.full_area_m2
        * candidate.parameters.cell_length_m
        / 1.0e-8,
        rel=0.0,
        abs=5.0e-14,
    )


def test_explicit_step_over_current_limit_is_rejected_atomically() -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    before_stability_query = _candidate_mutable_fingerprint(candidate)
    stable = candidate.stable_coupled_timestep()
    assert _candidate_mutable_fingerprint(candidate) == before_stability_query
    before = _candidate_mutable_fingerprint(candidate)

    with pytest.raises(CoupledTimestepLimitExceeded) as caught:
        candidate.advance_one_step(np.nextafter(stable, math.inf))

    assert caught.value.stable_dt_s == stable
    assert caught.value.requested_dt_s > stable
    assert _candidate_mutable_fingerprint(candidate) == before
    record = candidate.advance_one_step()
    assert record.end_time_s - record.start_time_s == stable


def test_advance_to_recomputes_limits_and_lands_on_events_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    event = 2.3456789e-5
    target = 7.654321e-5
    accepted_ends: list[float] = []
    original_advance = candidate.advance_one_step

    def record_advance(dt=None):
        result = original_advance(dt)
        accepted_ends.append(result.end_time_s)
        return result

    monkeypatch.setattr(candidate, "advance_one_step", record_advance)
    advanced_steps = candidate.advance_to(
        target,
        event_times_s=(event,),
    )

    assert advanced_steps == len(accepted_ends)
    assert event in accepted_ends
    assert accepted_ends[-1] == target
    assert candidate.time_s == target

    near_valve = PersistentCampaign2Candidate(0.026)
    start = 0.20 - 1.0e-6
    near_valve.horizontal_owner.state = replace(
        near_valve.horizontal_owner.state,
        time=start,
    )
    near_valve.vertical_state = replace(
        near_valve.vertical_state,
        time_s=start,
    )
    assert near_valve.stable_coupled_timestep() == pytest.approx(
        1.0e-6,
        rel=0.0,
        abs=2.0e-18,
    )


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_first_step_uses_one_absolute_pressure_reference_for_every_diameter(
    diameter: float,
) -> None:
    candidate = PersistentCampaign2Candidate(diameter)
    pressure_faces = candidate._vertical_pressure_faces()
    reference = candidate._vertical_liquid_pressure_reference_abs_Pa(
        pressure_faces
    )
    preview = candidate._solve_current_tee_uncommitted()

    record = candidate.advance_one_step()

    assert reference + preview.liquid.node_gauge_pressure_Pa == (
        preview.gas.interface_pressure_abs_Pa
    )
    assert preview.first_bottom_gas_entry is not None
    assert preview.first_bottom_gas_entry.active is False
    assert preview.transaction.gas_volume_flow_to_riser_m3_s == 0.0
    assert abs(record.horizontal_tee_liquid_residual_m3) <= 2.0e-18
    assert abs(record.vertical_tee_liquid_residual_m3) <= 2.0e-18
    assert candidate.last_tee_solution is not None
    assert abs(
        candidate.last_tee_solution.liquid.continuity_residual_m3_s
    ) <= 2.0e-19


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_finite_bottom_pocket_uses_gas_t_and_blocks_riser_liquid_for_all_diameters(
    diameter: float,
) -> None:
    candidate = PersistentCampaign2Candidate(diameter)
    _install_finite_bottom_gas_pocket(
        candidate,
        liquid_fraction=0.80,
        common_velocity_m_s=0.01,
    )
    before = _candidate_mutable_fingerprint(candidate)
    preview = candidate._solve_current_tee_uncommitted()

    assert preview.finite_bottom_gas_pocket is True
    assert preview.first_bottom_gas_entry is None
    assert preview.liquid.riser_outward_flow_m3_s == 0.0
    assert preview.liquid.normal_momentum_to_riser_N == 0.0
    assert preview.transaction.liquid_flow_to_riser_m3_s == 0.0
    assert preview.transaction.liquid_normal_momentum_flow_N == 0.0
    assert preview.transaction.gas_volume_flow_to_riser_m3_s is not None
    assert preview.gas_open_area_m2 <= candidate.parameters.full_area_m2
    assert preview.vertical_bottom_gas.sound_speed_m_s == pytest.approx(
        candidate.parameters.isothermal_gas_sound_speed_m_s,
        rel=0.0,
        abs=2.0e-14,
    )
    gamma_speed = math.sqrt(
        1.4
        * preview.vertical_bottom_gas.pressure_abs_Pa
        / preview.vertical_bottom_gas.density_kg_m3
    )
    assert preview.vertical_bottom_gas.sound_speed_m_s != pytest.approx(
        gamma_speed,
        rel=1.0e-3,
    )
    assert preview.transaction.liquid_open_area_m2 == 0.0
    assert preview.transaction.gas_open_area_m2 == pytest.approx(
        preview.gas_open_area_m2
    )
    assert preview.transaction.blocked_riser_area_m2 == pytest.approx(
        candidate.parameters.full_area_m2 - preview.gas_open_area_m2
    )
    assert (
        preview.transaction.gas_open_area_m2
        + preview.transaction.liquid_open_area_m2
        + preview.transaction.blocked_riser_area_m2
        == pytest.approx(candidate.parameters.full_area_m2)
    )
    assert _candidate_mutable_fingerprint(candidate) == before

    expected_geometry = lower_material_front_geometric_timestep_limit(
        candidate.vertical_state,
        candidate.parameters,
        cfl=1.0,
        top=candidate.vertical_kernel.top_boundary,
    )
    stable = candidate.stable_coupled_timestep()
    assert stable <= expected_geometry
    before_rejection = _candidate_mutable_fingerprint(candidate)
    with pytest.raises(CoupledTimestepLimitExceeded):
        candidate.advance_one_step(np.nextafter(stable, math.inf))
    assert _candidate_mutable_fingerprint(candidate) == before_rejection
    record = candidate.advance_one_step(stable)
    assert candidate.last_tee_solution is not None
    assert candidate.last_tee_solution.finite_bottom_gas_pocket is True
    assert record.transaction.liquid_flow_to_riser_m3_s == 0.0
    assert abs(record.horizontal_tee_liquid_residual_m3) <= 2.0e-18
    assert abs(record.vertical_tee_liquid_residual_m3) <= 2.0e-18
    assert abs(record.horizontal_tee_gas_residual_kg) <= 2.0e-18
    assert abs(record.vertical_tee_gas_residual_kg) <= 2.0e-18
    assert abs(record.global_liquid_residual_m3) <= 1.0e-14
    assert abs(record.global_gas_residual_kg) <= 2.0e-15


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_public_driver_lower_front_crosses_two_faces_and_restarts_without_dt_collapse(
    diameter: float,
) -> None:
    """Constructed stress path; not a claim about a natural case trajectory."""

    candidate = PersistentCampaign2Candidate(diameter)
    _install_uniform_lower_front_checkpoint(
        candidate,
        front_cell=0,
        front_liquid_fraction=0.20,
        common_velocity_m_s=50.0,
    )
    area = candidate.parameters.full_area_m2
    visited = [0]
    exact_event_steps: list[float] = []
    steps_in_cell = 0

    while candidate.vertical_state.lower_material_front_cell < 2:
        before = candidate.vertical_state
        before_marker = before.lower_material_front_cell
        assert before_marker is not None
        geometric = lower_material_front_geometric_timestep_limit(
            before,
            candidate.parameters,
            cfl=1.0,
            top=candidate.vertical_kernel.top_boundary,
        )
        step = candidate.stable_coupled_timestep()
        record = candidate.advance_one_step(step)
        _assert_record_budgets_close(record)
        steps_in_cell += 1
        marker = candidate.vertical_state.lower_material_front_cell
        assert marker is not None
        assert candidate.vertical_state.lower_material_front_orientation == (
            "gas_below_liquid_above"
        )
        if marker != before_marker:
            assert marker == before_marker + 1
            assert step == geometric
            assert candidate.vertical_state.Al[marker] == area
            assert steps_in_cell <= 32
            exact_event_steps.append(step)
            visited.append(marker)
            steps_in_cell = 0
            if marker == 1:
                # Reconstruct every persisted field as a restart would.  No
                # orientation inference or gas seed is permitted.
                state = candidate.vertical_state
                restarted = VerticalTwoFluidState.from_iterables(
                    Al=state.Al,
                    Ql=state.Ql,
                    Mg=state.Mg,
                    Jg=state.Jg,
                    time_s=state.time_s,
                    cumulative_top_liquid_outflow_m3=(
                        state.cumulative_top_liquid_outflow_m3
                    ),
                    cumulative_top_gas_outflow_kg=(
                        state.cumulative_top_gas_outflow_kg
                    ),
                    cumulative_top_gas_inflow_kg=(
                        state.cumulative_top_gas_inflow_kg
                    ),
                    cumulative_bottom_liquid_exchange_m3=(
                        state.cumulative_bottom_liquid_exchange_m3
                    ),
                    cumulative_bottom_gas_exchange_kg=(
                        state.cumulative_bottom_gas_exchange_kg
                    ),
                    lower_material_front_cell=(
                        state.lower_material_front_cell
                    ),
                    lower_material_front_orientation=(
                        state.lower_material_front_orientation
                    ),
                )
                assert restarted == state
                candidate.vertical_state = restarted
                # An exact event is followed by a resolved positive step, not
                # an asymptotic sequence of near-face substeps.
                assert candidate.stable_coupled_timestep() > step

    assert visited == [0, 1, 2]
    assert len(exact_event_steps) == 2
    assert candidate.vertical_state.lower_material_front_cell == 2
    assert candidate.vertical_state.lower_material_front_orientation == (
        "gas_below_liquid_above"
    )


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_public_driver_reverse_checkpoint_evacuates_then_reenters_atomically(
    diameter: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise disappearance/re-entry from a declared conserved checkpoint.

    The high-speed binary-rational checkpoint makes both material events
    shorter than the acoustic ceiling.  It verifies event ownership and
    rollback; it is deliberately not presented as a naturally generated
    Campaign-2 reversal.
    """

    candidate = PersistentCampaign2Candidate(diameter)
    template = candidate._solve_current_tee_uncommitted()
    checkpoint_time = 0.21
    _install_horizontal_contact_checkpoint(
        candidate,
        time_s=checkpoint_time,
        pressure_abs_Pa=template.transaction.gas_interface_pressure_abs_Pa,
    )
    _install_uniform_lower_front_checkpoint(
        candidate,
        front_cell=0,
        front_liquid_fraction=0.75,
        common_velocity_m_s=-2048.0,
        lower_gas_density_kg_m3=1.20,
        time_s=checkpoint_time,
    )
    area = candidate.parameters.full_area_m2
    dz = candidate.parameters.cell_length_m

    def constructed_preview():
        marker = candidate.vertical_state.lower_material_front_cell
        if marker is not None:
            star = lower_material_front_star_state(
                candidate.vertical_state,
                candidate.parameters,
                candidate.vertical_kernel.top_boundary,
            )
            volume_flow = star.interface_volume_flow_m3_s
            density = star.gas_density_kg_m3
            velocity = volume_flow / area
            mass_flow = density * volume_flow
            momentum_flow = mass_flow * velocity
            pressure = star.interface_pressure_abs_Pa
            liquid_gauge = 0.0
            liquid_flow = 0.0
            liquid_momentum = 0.0
            gas_open_area = area
            liquid_open_area = 0.0
        else:
            # A fresh positive donor parcel starts from a truly saturated
            # bottom: there is no lower marker and no vertical gas seed.
            velocity = 16384.0
            gas_open_area = 0.50 * area
            liquid_open_area = area - gas_open_area
            volume_flow = gas_open_area * velocity
            density = (
                candidate.horizontal_owner.state.gas.mass
                / candidate.horizontal_owner.state.gas.volume
            )
            mass_flow = density * volume_flow
            momentum_flow = mass_flow * velocity
            liquid_flow = liquid_open_area * velocity
            liquid_momentum = (
                candidate.parameters.liquid_density_kg_m3
                * liquid_flow
                * velocity
            )
            pressure_faces = candidate._vertical_pressure_faces()
            pressure = pressure_faces[0]
            reference = (
                candidate._vertical_liquid_pressure_reference_abs_Pa(
                    pressure_faces
                )
            )
            liquid_gauge = pressure - reference
        transaction = TeeTransaction(
            west_liquid_flow_m3_s=liquid_flow,
            east_liquid_flow_m3_s=0.0,
            gas_mass_flow_to_riser_kg_s=mass_flow,
            gas_volume_flow_to_riser_m3_s=volume_flow,
            gas_normal_momentum_flow_N=momentum_flow,
            liquid_normal_momentum_flow_N=liquid_momentum,
            liquid_node_gauge_pressure_Pa=liquid_gauge,
            gas_interface_pressure_abs_Pa=pressure,
            riser_mouth_area_m2=area,
            gas_open_area_m2=gas_open_area,
            liquid_open_area_m2=liquid_open_area,
            blocked_riser_area_m2=0.0,
        )
        return replace(
            template,
            transaction=transaction,
            gas_open_area_m2=gas_open_area,
            finite_bottom_gas_pocket=marker is not None,
            first_bottom_gas_entry=None,
        )

    monkeypatch.setattr(
        candidate,
        "_solve_current_tee_uncommitted",
        constructed_preview,
    )

    retreat_step = candidate.stable_coupled_timestep()
    retreat_before = _candidate_mutable_fingerprint(candidate)
    with pytest.raises(CoupledTimestepLimitExceeded):
        candidate.advance_one_step(np.nextafter(retreat_step, math.inf))
    assert _candidate_mutable_fingerprint(candidate) == retreat_before
    retreat = candidate.advance_one_step(retreat_step)
    _assert_record_budgets_close(retreat)
    assert candidate.vertical_state.lower_material_front_cell is None
    assert candidate.vertical_state.lower_material_front_orientation is None
    assert candidate.vertical_state.Al[0] == area
    assert candidate.vertical_state.Mg[0] == 0.0
    assert candidate.vertical_state.Jg[0] == 0.0

    state = candidate.vertical_state
    restarted = VerticalTwoFluidState.from_iterables(
        Al=state.Al,
        Ql=state.Ql,
        Mg=state.Mg,
        Jg=state.Jg,
        time_s=state.time_s,
        cumulative_top_liquid_outflow_m3=state.cumulative_top_liquid_outflow_m3,
        cumulative_top_gas_outflow_kg=state.cumulative_top_gas_outflow_kg,
        cumulative_top_gas_inflow_kg=state.cumulative_top_gas_inflow_kg,
        cumulative_bottom_liquid_exchange_m3=(
            state.cumulative_bottom_liquid_exchange_m3
        ),
        cumulative_bottom_gas_exchange_kg=(
            state.cumulative_bottom_gas_exchange_kg
        ),
        lower_material_front_cell=None,
        lower_material_front_orientation=None,
    )
    assert restarted == state
    candidate.vertical_state = restarted

    reentry_preview = constructed_preview()
    qg = reentry_preview.transaction.gas_volume_flow_to_riser_m3_s
    assert qg is not None and qg > 0.0
    net_fill = (
        reentry_preview.transaction.liquid_flow_to_riser_m3_s + qg
    )
    bottom_event = area * dz / qg
    gas_only_upper_event = (
        candidate._upper_surface_geometric_timestep_limit(qg)
    )
    upper_event = candidate._upper_surface_geometric_timestep_limit(net_fill)
    assert upper_event < gas_only_upper_event
    assert upper_event < bottom_event
    reentry_step = candidate.stable_coupled_timestep()
    assert reentry_step == upper_event
    reentry_before = _candidate_mutable_fingerprint(candidate)
    with pytest.raises(CoupledTimestepLimitExceeded):
        candidate.advance_one_step(np.nextafter(reentry_step, math.inf))
    assert _candidate_mutable_fingerprint(candidate) == reentry_before

    reentry = candidate.advance_one_step(reentry_step)
    _assert_record_budgets_close(reentry)
    assert candidate.vertical_state.lower_material_front_cell == 0
    assert candidate.vertical_state.lower_material_front_orientation == (
        "gas_below_liquid_above"
    )
    expected_lower_gas_area = qg * reentry_step / dz
    assert area - candidate.vertical_state.Al[0] == pytest.approx(
        expected_lower_gas_area,
        abs=8.0 * math.ulp(area),
    )
    assert candidate.vertical_state.Mg[0] > 0.0
    assert candidate.stable_coupled_timestep() > reentry_step


def test_one_step_passes_the_same_transaction_once_to_both_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    transaction_ids: list[tuple[str, int]] = []
    original_kernel = candidate.vertical_kernel
    original_commit = candidate.horizontal_owner.commit_tee

    class VerticalSpy:
        parameters = original_kernel.parameters
        top_boundary = original_kernel.top_boundary

        def advance(self, state, *, dt, tee_transaction):
            transaction_ids.append(("vertical", id(tee_transaction)))
            return original_kernel.advance(
                state,
                dt=dt,
                tee_transaction=tee_transaction,
            )

    def horizontal_commit_spy(transaction, dt):
        transaction_ids.append(("horizontal", id(transaction)))
        return original_commit(transaction, dt)

    monkeypatch.setattr(candidate, "vertical_kernel", VerticalSpy())
    monkeypatch.setattr(
        candidate.horizontal_owner,
        "commit_tee",
        horizontal_commit_spy,
    )

    record = candidate.advance_one_step()

    assert [owner for owner, _ in transaction_ids] == [
        "vertical",
        "horizontal",
    ]
    assert len({identifier for _, identifier in transaction_ids}) == 1
    assert candidate.horizontal_owner.tee_transaction_count == 1
    assert candidate.step_count == 1
    assert record.vertical_tee_liquid_residual_m3 == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert record.vertical_tee_gas_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert record.horizontal_tee_liquid_residual_m3 == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert record.horizontal_tee_gas_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_late_failure_rolls_back_every_owner_then_next_step_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after all physical commits must leave no partial step."""

    candidate = PersistentCampaign2Candidate(0.026)
    candidate.advance_one_step()
    boundary = candidate.horizontal_owner.reservoir_boundary
    assert boundary is not None

    original_audit = candidate.global_budget.audit
    audit_calls = 0
    injected = RuntimeError("injected late global-budget audit failure")

    def fail_once(**kwargs):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 1:
            raise injected
        return original_audit(**kwargs)

    monkeypatch.setattr(candidate.global_budget, "audit", fail_once)
    before = _candidate_mutable_fingerprint(candidate)
    time_before = candidate.time_s
    safe_step_before = candidate.stable_coupled_timestep()
    step_count_before = candidate.step_count
    tee_count_before = candidate.horizontal_owner.tee_transaction_count
    reservoir_count_before = boundary.commit_count

    with pytest.raises(RuntimeError) as caught:
        candidate.advance_one_step()

    assert caught.value is injected
    assert audit_calls == 1
    assert _candidate_mutable_fingerprint(candidate) == before
    assert candidate.time_s == time_before
    assert candidate.step_count == step_count_before
    assert candidate.horizontal_owner.tee_transaction_count == tee_count_before
    assert boundary.commit_count == reservoir_count_before

    record = candidate.advance_one_step()

    assert audit_calls == 2
    assert record.start_time_s == time_before
    assert record.end_time_s == pytest.approx(
        time_before + safe_step_before,
        abs=2.0e-15,
    )
    assert candidate.step_count == step_count_before + 1
    assert candidate.horizontal_owner.tee_transaction_count == tee_count_before + 1
    assert boundary.commit_count == reservoir_count_before + 1


def test_production_driver_uses_dynamic_eos_pressure_and_kernel_default() -> None:
    source = inspect.getsource(candidate_module)
    step_source = inspect.getsource(
        PersistentCampaign2Candidate.advance_one_step
    )
    assert "mixture_hydrostatic_pressure_faces" not in source
    assert "pressure_faces_Pa=" not in step_source

    candidate = PersistentCampaign2Candidate(0.026)
    expected = isothermal_common_pressure_faces(
        candidate.vertical_state,
        candidate.parameters,
        candidate.vertical_kernel.top_boundary,
    )
    assert candidate._vertical_pressure_faces() == expected

    first_gas_cell = next(
        cell
        for cell, liquid_area in enumerate(candidate.vertical_state.Al)
        if (
            candidate.parameters.full_area_m2 - liquid_area
            > candidate.parameters.area_tolerance_m2
        )
    )
    gas_mass = list(candidate.vertical_state.Mg)
    gas_mass[first_gas_cell] *= 1.01
    compressed = replace(candidate.vertical_state, Mg=tuple(gas_mass))
    dynamic = candidate._vertical_pressure_faces(compressed)
    diagnostic_hydrostatic = mixture_hydrostatic_pressure_faces(
        compressed,
        candidate.parameters,
        candidate.vertical_kernel.top_boundary,
    )
    assert dynamic == isothermal_common_pressure_faces(
        compressed,
        candidate.parameters,
        candidate.vertical_kernel.top_boundary,
    )
    assert dynamic[first_gas_cell] != pytest.approx(
        diagnostic_hydrostatic[first_gas_cell], rel=1.0e-5
    )


def test_liquid_tee_roundoff_pin_covers_observed_h3_cancellation_only() -> None:
    """The 0.159 s H3 stop was binary64 cancellation, not resolved reflux."""

    traces = (
        LiquidBranchTrace(
            area_m2=0.0019634954084936209,
            outward_velocity_m_s=-1.4724850755164128e-14,
            gauge_pressure_Pa=5972.1317999996754,
            wave_speed_m_s=28.0,
            density_kg_m3=998.0,
        ),
        LiquidBranchTrace(
            area_m2=0.0019634954084936209,
            outward_velocity_m_s=2.951962007991591e-14,
            gauge_pressure_Pa=5972.1317999991534,
            wave_speed_m_s=28.0,
            density_kg_m3=998.0,
        ),
        LiquidBranchTrace(
            area_m2=0.00053092915845667505,
            outward_velocity_m_s=0.0,
            gauge_pressure_Pa=5972.1318000000028,
            wave_speed_m_s=28.0,
            density_kg_m3=998.0,
        ),
    )

    assert candidate_module._liquid_tee_is_roundoff_equilibrium(traces)
    observed = solve_liquid_tee(*traces)
    pinned = candidate_module._pin_liquid_tee_roundoff_flows(traces, observed)
    assert (
        pinned.west_outward_flow_m3_s,
        pinned.east_outward_flow_m3_s,
        pinned.riser_outward_flow_m3_s,
        pinned.continuity_residual_m3_s,
        pinned.normal_momentum_to_riser_N,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0)
    resolved = (
        traces[0],
        traces[1],
        replace(traces[2], gauge_pressure_Pa=traces[2].gauge_pressure_Pa + 1.0e-6),
    )
    assert not candidate_module._liquid_tee_is_roundoff_equilibrium(resolved)
    resolved_solution = solve_liquid_tee(*resolved)
    assert (
        candidate_module._pin_liquid_tee_roundoff_flows(
            resolved,
            resolved_solution,
        )
        == resolved_solution
    )


def test_equal_pressure_open_gas_interface_keeps_both_owners_stationary() -> None:
    """Regression for pressure being counted once, not in two formulations."""

    candidate = PersistentCampaign2Candidate(0.026)
    parameters = candidate.parameters
    vertical_before = atmospheric_empty_state(parameters)
    pressure_faces = isothermal_common_pressure_faces(
        vertical_before,
        parameters,
        candidate.vertical_kernel.top_boundary,
    )
    bottom_pressure = pressure_faces[0]
    rho_g = parameters.atmospheric_gas_density_kg_m3
    sound_speed = math.sqrt(1.4 * bottom_pressure / rho_g)
    equal_gas = GasTrace(
        pressure_abs_Pa=bottom_pressure,
        density_kg_m3=rho_g,
        normal_velocity_m_s=0.0,
        sound_speed_m_s=sound_speed,
    )
    gas = solve_gas_tee(
        equal_gas,
        equal_gas,
        open_area_m2=parameters.full_area_m2,
    )
    equal_liquid = LiquidBranchTrace(
        area_m2=parameters.full_area_m2,
        outward_velocity_m_s=0.0,
        gauge_pressure_Pa=0.0,
        wave_speed_m_s=SHARED_CLOSURE.wave_speed_m_s,
        density_kg_m3=parameters.liquid_density_kg_m3,
    )
    liquid = solve_liquid_tee_with_blocked_riser(
        equal_liquid,
        equal_liquid,
    )
    transaction = transaction_from_tee_solutions(
        liquid,
        gas,
        physical_riser_area_m2=parameters.full_area_m2,
    )

    assert parameters.full_area_m2 > 0.0
    assert gas.mass_flow_to_riser_kg_s == 0.0
    assert gas.normal_momentum_flow_N == 0.0
    assert gas.interface_pressure_force_N > 0.0

    horizontal_before = candidate.horizontal_owner.state
    horizontal_area_before = np.asarray(horizontal_before.area).copy()
    horizontal_discharge_before = np.asarray(
        horizontal_before.discharge
    ).copy()
    horizontal_gas_before = horizontal_before.gas
    increment = candidate.horizontal_owner.commit_tee(transaction, 2.0e-4)
    assert increment.liquid_volume_m3 == 0.0
    assert increment.gas_mass_kg == 0.0
    assert np.array_equal(
        candidate.horizontal_owner.state.area,
        horizontal_area_before,
    )
    assert np.array_equal(
        candidate.horizontal_owner.state.discharge,
        horizontal_discharge_before,
    )
    assert candidate.horizontal_owner.state.gas == horizontal_gas_before

    vertical_result = candidate.vertical_kernel.advance(
        vertical_before,
        dt=2.0e-4,
        tee_transaction=transaction,
        # The solved pressure equals the analytic hydrostatic boundary.  The
        # kernel's default branch preserves that fixed point exactly.
        pressure_faces_Pa=None,
    )
    assert vertical_result.state.Al == vertical_before.Al
    assert vertical_result.state.Ql == vertical_before.Ql
    assert vertical_result.state.Mg == vertical_before.Mg
    assert vertical_result.state.Jg == vertical_before.Jg
    assert vertical_result.bottom_gas_exchange_kg_s == 0.0
    assert vertical_result.budget.total_momentum_residual_kg_m_s == 0.0


@pytest.mark.parametrize("diameter", SELECTED_DIAMETERS_M)
def test_short_physical_smoke_closes_global_budgets(
    diameter: float,
) -> None:
    candidate = PersistentCampaign2Candidate(diameter)

    report = candidate.run_smoke(0.005)

    assert report["end_time_s"] == 0.005
    assert report["step_count"] > math.ceil(
        0.005 / candidate.numerics.max_dt_s
    )
    assert report["tee_transaction_count"] == report["step_count"]
    assert report["horizontal_owner_active"] is True
    assert report["physical_top_liquid_outflow_m3"] == 0.0
    assert report["physical_rim_crossed"] is False
    assert abs(report["global_budget"]["liquid_residual_m3"]) <= 1.0e-14
    assert abs(report["global_budget"]["gas_residual_kg"]) <= 2.0e-15
    assert candidate.global_budget.internal_tee_liquid_to_riser_m3 == pytest.approx(
        candidate.horizontal_owner.cumulative_liquid_to_riser_m3,
        abs=2.0e-18,
    )
    assert candidate.global_budget.internal_tee_gas_to_riser_kg == pytest.approx(
        candidate.horizontal_owner.cumulative_gas_to_riser_kg,
        abs=2.0e-18,
    )


def test_smoke_entry_point_refuses_a_formal_duration() -> None:
    candidate = PersistentCampaign2Candidate(0.026)
    with pytest.raises(ValueError, match="capped"):
        candidate.run_smoke(np.nextafter(SMOKE_END_LIMIT_S, math.inf))
