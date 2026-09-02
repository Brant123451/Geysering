from dataclasses import replace
import math

import pytest

from model.errors import ContractViolation, MissingPhysicalClosure
from model.flux import state_token
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import VerticalPressureVoidStageComponent
from model.port_contracts import (
    GrossNodePortFlux,
    PortKey,
    PortTraceState,
    TNodeTrial,
)
from model.state import VerticalState
from model.vertical_case1_adapter import (
    ATMOSPHERIC_PRESSURE_PA,
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    INITIAL_AIR_TEMPERATURE_K,
    PIPE_DIAMETER_M,
)
from model.vertical_pressure_void_component import (
    AtmosphericLiquidFallback,
    AtmosphericTopState,
    F0VerticalCapillaryOwner,
    F0VerticalPressureVoidStageComponent,
    conservative_void_remap,
    f0_smooth_pipe_darcy_factor,
)
from model.vertical_twostream_solver import (
    S1_GRAVITY_M_S2,
    S1_LIQUID_DENSITY_KG_M3,
    _component_state,
)


RHO_G_ATM = ATMOSPHERIC_PRESSURE_PA / (
    DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
)


def _trace(
    node: str,
    port: str,
    *,
    area: float,
    liquid_area: float,
    liquid_pressure: float,
    gas_pressure: float,
) -> PortTraceState:
    if port == "main_left":
        normal = (1.0, 0.0)
        component = "horizontal_main"
    elif port == "main_right":
        normal = (-1.0, 0.0)
        component = "horizontal_main"
    else:
        normal = (0.0, -1.0)
        component = "vertical_riser"
    return PortTraceState(
        key=PortKey(node, port),
        component_id=component,
        normal_into_node_x=normal[0],
        normal_into_node_z=normal[1],
        full_area_m2=area,
        liquid_area_m2=liquid_area,
        gas_area_m2=area - liquid_area,
        liquid_density_kg_m3=S1_LIQUID_DENSITY_KG_M3,
        gas_density_kg_m3=max(gas_pressure, 1.0)
        / (DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K),
        liquid_absolute_pressure_Pa=liquid_pressure,
        gas_absolute_pressure_Pa=gas_pressure,
    )


def _trial(
    assembly,
    *,
    dt: float,
    pressure: float,
    riser_flux: GrossNodePortFlux | None = None,
    state: VerticalState | None = None,
    physical_stage: str = "stage1_closed",
) -> TNodeTrial:
    vertical = assembly.state.vertical if state is None else state
    area = assembly.geometry.vertical_area_m2
    liquid_bottom = vertical.Aup[0] + vertical.Adown[0]
    traces = (
        _trace(
            "riser_T",
            "main_left",
            area=area,
            liquid_area=area,
            liquid_pressure=pressure,
            gas_pressure=pressure,
        ),
        _trace(
            "riser_T",
            "main_right",
            area=area,
            liquid_area=area,
            liquid_pressure=pressure,
            gas_pressure=pressure,
        ),
        _trace(
            "riser_T",
            "riser_bottom",
            area=area,
            liquid_area=liquid_bottom,
            liquid_pressure=pressure,
            gas_pressure=pressure,
        ),
    )
    flux = (
        GrossNodePortFlux(key=PortKey("riser_T", "riser_bottom"))
        if riser_flux is None
        else riser_flux
    )
    bottom_advective_to_component = (
        S1_LIQUID_DENSITY_KG_M3
        * (
            flux.liquid_into_node_m3_s * flux.liquid_into_node_speed_m_s
            + flux.liquid_out_of_node_m3_s * flux.liquid_out_of_node_speed_m_s
        )
        + flux.gas_into_node_kg_s * flux.gas_into_node_speed_m_s
        + flux.gas_out_of_node_kg_s * flux.gas_out_of_node_speed_m_s
    )
    flux = replace(
        flux,
        advective_momentum_to_node_z_N=-bottom_advective_to_component,
        pressure_traction_to_node_z_N=-pressure * area,
    )
    return TNodeTrial(
        trial_id="vertical-riser-stage-trial",
        base_state_token=state_token(assembly.state),
        node_name="riser_T",
        physical_stage=physical_stage,
        rk_stage=1,
        dt_s=dt,
        common_absolute_pressure_Pa=pressure,
        node_gas_area_fraction=(area - liquid_bottom) / area,
        port_traces=traces,
        gross_fluxes=(
            GrossNodePortFlux(key=PortKey("riser_T", "main_left")),
            GrossNodePortFlux(key=PortKey("riser_T", "main_right")),
            flux,
        ),
    )


def _after(state: VerticalState, delta, dt: float) -> VerticalState:
    def add(before, rate):
        return tuple(value + dt * change for value, change in zip(before, rate, strict=True))

    return VerticalState(
        Aup=add(state.Aup, delta.Aup),
        Qup=add(state.Qup, delta.Qup),
        Adown=add(state.Adown, delta.Adown),
        Qdown=add(state.Qdown, delta.Qdown),
        Mg=add(state.Mg, delta.Mg),
        Jg=add(state.Jg, delta.Jg),
    )


def _component(
    cell_count: int,
    *,
    fallback: AtmosphericLiquidFallback | None = None,
) -> F0VerticalPressureVoidStageComponent:
    return F0VerticalPressureVoidStageComponent(
        cell_count=cell_count,
        capillary_owner=F0VerticalCapillaryOwner(
            mode="planar_2d_zeroGradient_walls"
        ),
        atmospheric_top=AtmosphericTopState(liquid_fallback=fallback),
    )


def test_port_trace_reuses_the_FV_packing_band_without_mutating_inventory() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    state = component.initial_state
    full = component._solver.pipe_area_m2
    packing = component._solver._parameters.packing_tolerance
    numerical = replace(
        state,
        Aup=(state.Aup[0] + 0.5 * packing,) + state.Aup[1:],
    )
    before = repr(numerical)

    first = component.port_trace(numerical, assembly.geometry)
    second = component.port_trace(numerical, assembly.geometry)

    assert first == second
    assert first.liquid_area_m2 == full
    assert first.gas_area_m2 == 0.0
    assert repr(numerical) == before

    finite_overpack = replace(
        state,
        Aup=(state.Aup[0] + 1.01 * packing,) + state.Aup[1:],
    )
    with pytest.raises(ContractViolation, match="over-packs"):
        component.port_trace(finite_overpack, assembly.geometry)


def test_exact_dry_directional_roundoff_is_only_canonicalised_in_Case1_view() -> None:
    n = 8
    component = _component(n)
    state = component.initial_state
    numerical = replace(
        state,
        Qdown=(7.0e-40,) + state.Qdown[1:],
    )
    before = repr(numerical)

    view = _component_state(
        component._solver._runtime,
        numerical,
        dry_discharge_tolerance_m3_s=(
            component._solver._parameters.dry_area_tolerance
        ),
    )

    assert view.downward_discharge[0] == 0.0
    assert repr(numerical) == before
    finite = replace(
        state,
        Qdown=(
            1.01 * component._solver._parameters.dry_area_tolerance,
        )
        + state.Qdown[1:],
    )
    with pytest.raises(
        component._solver._runtime.fv.StateAdmissibilityError,
        match="dry downward stream",
    ):
        _component_state(
            component._solver._runtime,
            finite,
            dry_discharge_tolerance_m3_s=(
                component._solver._parameters.dry_area_tolerance
            ),
        )


def test_z0p5842_cut_cell_accepts_same_stage_bottom_gas_by_conservative_piston_remap() -> None:
    n = 40
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 0.5842
    )
    dt = 1.0e-6
    gas_volume_rate = 1.0e-7
    node_gas_density = pressure / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        gas_out_of_node_kg_s=node_gas_density * gas_volume_rate,
        gas_out_of_node_speed_m_s=0.1,
    )
    trial = _trial(
        assembly,
        dt=dt,
        pressure=pressure,
        riser_flux=flux,
        physical_stage="stage2_pressure_reservoir",
    )
    before = state_token(assembly.state)

    evaluation = component.evaluate_joint_stage(
        assembly.state.vertical,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage2_pressure_reservoir",
        dt_s=trial.dt_s,
    )

    assert evaluation.proposal.status == "accepted", evaluation.proposal.capacity_reject
    assert evaluation.diagnostics is not None
    piston = evaluation.diagnostics.bottom_gas_piston
    expected_volume = dt * gas_volume_rate
    assert piston.requested_gas_volume_m3 == pytest.approx(expected_volume)
    assert piston.displaced_liquid_volume_m3 == pytest.approx(expected_volume)
    assert piston.destination_cell == max(
        cell
        for cell, area in enumerate(assembly.state.vertical.Aup)
        if area > 0.0
    )
    assert piston.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-18)
    assert piston.liquid_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-18
    )
    final = _after(assembly.state.vertical, evaluation.proposal.delta, dt)
    area = assembly.geometry.vertical_area_m2
    assert area - final.Aup[0] - final.Adown[0] > 0.0
    assert final.Mg[0] > 0.0
    assert evaluation.diagnostics.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    assert evaluation.diagnostics.momentum_budget.residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-11
    )
    assert state_token(assembly.state) == before
    assert isinstance(component, VerticalPressureVoidStageComponent)


def test_z0p5842_single_liquid_label_reversal_is_not_a_cross_gap_merge() -> None:
    n = 40
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 0.5842
    )
    dt = 1.0e-4
    trial = _trial(assembly, dt=dt, pressure=pressure)

    evaluation = component.evaluate_joint_stage(
        assembly.state.vertical,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    source = assembly.state.vertical
    final = _after(source, evaluation.proposal.delta, dt)
    cut = max(cell for cell, area in enumerate(source.Aup) if area > 0.0)
    assert source.Aup[cut] > 0.0
    assert source.Adown[cut] == 0.0
    assert final.Aup[cut] == pytest.approx(0.0, abs=1.0e-18)
    assert final.Adown[cut] == pytest.approx(source.Aup[cut], abs=1.0e-18)
    assert evaluation.diagnostics.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    assert evaluation.diagnostics.momentum_budget.residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-11
    )


def test_void_remap_preserves_mass_momentum_and_populates_every_new_void() -> None:
    result = conservative_void_remap(
        old_void_area_m2=(0.0, 1.0e-4, 2.0e-4, 0.0),
        new_void_area_m2=(0.0, 0.5e-4, 1.5e-4, 1.0e-4),
        gas_mass_cell_kg=(0.0, 1.0e-6, 4.0e-6, 0.0),
        gas_momentum_cell_kg_m_s=(0.0, 2.0e-7, -1.0e-7, 0.0),
        cell_length_m=0.01,
    )
    assert sum(result.gas_mass_cell_kg) == pytest.approx(5.0e-6, abs=1.0e-18)
    assert sum(result.gas_momentum_cell_kg_m_s) == pytest.approx(1.0e-7, abs=1.0e-18)
    assert all(result.gas_mass_cell_kg[cell] > 0.0 for cell in (1, 2, 3))
    assert result.mass_residual_kg == pytest.approx(0.0, abs=1.0e-18)
    assert result.momentum_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-18)

    with pytest.raises(MissingPhysicalClosure, match="newly isolated"):
        conservative_void_remap(
            old_void_area_m2=(1.0e-4, 0.0, 0.0),
            new_void_area_m2=(1.0e-4, 0.0, 1.0e-4),
            gas_mass_cell_kg=(1.0e-6, 0.0, 0.0),
            gas_momentum_cell_kg_m_s=(0.0, 0.0, 0.0),
            cell_length_m=0.01,
        )


def _bottom_opening_case(*, with_gas_source: bool):
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    upward_area = 0.8 * area
    downward_area = area - upward_area
    upward_speed = 0.01
    upward_rate = upward_area * upward_speed
    downward_rate = 1.0e-7
    dt = 1.0e-6
    state = VerticalState(
        Aup=(upward_area, area) + (0.0,) * (n - 2),
        Qup=(upward_rate, upward_rate) + (0.0,) * (n - 2),
        Adown=(downward_area,) + (0.0,) * (n - 1),
        Qdown=(downward_rate,) + (0.0,) * (n - 1),
        Mg=(0.0, 0.0) + (RHO_G_ATM * area,) * (n - 2),
        Jg=(0.0,) * n,
    )
    gas_mass_rate = RHO_G_ATM * downward_rate if with_gas_source else 0.0
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        liquid_into_node_m3_s=downward_rate,
        liquid_into_node_speed_m_s=downward_rate / downward_area,
        liquid_out_of_node_m3_s=upward_rate,
        liquid_out_of_node_speed_m_s=upward_speed,
        gas_out_of_node_kg_s=gas_mass_rate,
        gas_out_of_node_speed_m_s=(0.1 if gas_mass_rate > 0.0 else 0.0),
    )
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 0.2
    )
    return (
        assembly,
        _component(n),
        state,
        _trial(
            assembly,
            dt=dt,
            pressure=pressure,
            riser_flux=flux,
            state=state,
            physical_stage="stage2_pressure_reservoir",
        ),
        gas_mass_rate,
    )


def test_first_bottom_gas_parcel_atomically_seeds_new_isolated_void_once() -> None:
    assembly, component, state, trial, gas_mass_rate = _bottom_opening_case(
        with_gas_source=True
    )
    dt = trial.dt_s

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage2_pressure_reservoir",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    remap = evaluation.diagnostics.void_remap
    assert remap.boundary_source_mass_kg == pytest.approx(
        dt * gas_mass_rate, rel=0.0, abs=1.0e-24
    )
    assert remap.boundary_source_momentum_kg_m_s == pytest.approx(
        dt * gas_mass_rate * 0.1, rel=0.0, abs=1.0e-24
    )
    final = _after(state, evaluation.proposal.delta, dt)
    area = assembly.geometry.vertical_area_m2
    assert area - final.Aup[0] - final.Adown[0] > 1.0e-14
    assert final.Mg[0] > 0.0
    initial_mass = sum(state.Mg) * component._solver.cell_length_m
    final_mass = sum(final.Mg) * component._solver.cell_length_m
    top = evaluation.diagnostics.top_gas
    expected_change = dt * (
        gas_mass_rate + top.inflow_kg_s - top.outflow_kg_s
    )
    assert final_mass - initial_mass == pytest.approx(
        expected_change, rel=0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    assert evaluation.diagnostics.momentum_budget.residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-11
    )


def test_new_isolated_bottom_void_without_boundary_gas_source_still_rejects() -> None:
    assembly, component, state, trial, _ = _bottom_opening_case(
        with_gas_source=False
    )

    proposal = component.propose_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage2_pressure_reservoir",
        dt_s=trial.dt_s,
    )

    assert proposal.status == "capacity_rejected"
    assert proposal.capacity_reject is not None
    assert proposal.capacity_reject.reason_code == "void_mass_pairing"
    assert "newly isolated" in proposal.capacity_reject.detail


def test_bottom_gas_piston_advances_only_into_the_nearest_internal_gas_gap() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    state = VerticalState(
        Aup=(area, area, 0.0, 0.5 * area) + (0.0,) * (n - 4),
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0, 0.0, RHO_G_ATM * area, RHO_G_ATM * 0.5 * area)
        + (RHO_G_ATM * area,) * (n - 4),
        Jg=(0.0,) * n,
    )
    pressure = ATMOSPHERIC_PRESSURE_PA
    gas_volume_rate = 1.0e-7
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        gas_out_of_node_kg_s=RHO_G_ATM * gas_volume_rate,
        gas_out_of_node_speed_m_s=0.1,
    )
    trial = _trial(
        assembly,
        dt=1.0e-6,
        pressure=pressure,
        riser_flux=flux,
        state=state,
        physical_stage="stage2_pressure_reservoir",
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage2_pressure_reservoir",
        dt_s=trial.dt_s,
    )

    assert evaluation.proposal.status == "accepted", evaluation.proposal.capacity_reject
    assert evaluation.diagnostics is not None
    piston = evaluation.diagnostics.bottom_gas_piston
    assert piston.receiving_cells == (2,)
    assert piston.traversed_gap_cells == (2,)
    assert piston.top_spill_volume_m3 == 0.0
    assert piston.deposited_liquid_volume_m3 == pytest.approx(
        piston.displaced_liquid_volume_m3
    )
    # The finite gas corridor remains finite and the liquid column on its far
    # side is not used as an alternate receiver.
    final = _after(state, evaluation.proposal.delta, trial.dt_s)
    assert area - final.Aup[2] - final.Adown[2] > 0.0
    assert 3 not in piston.receiving_cells


def test_bottom_piston_top_spill_is_an_explicit_conservative_boundary_parcel() -> None:
    n = 4
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    dz = component._solver.cell_length_m
    velocity = 0.23
    displacement = 0.15 * area * dz
    state = VerticalState(
        Aup=(area,) * n,
        Qup=(area * velocity,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )
    signed = _component_state(component._solver._runtime, state)

    remapped, piston = component._bottom_gas_piston_remap(
        liquid_state=signed,
        requested_gas_volume_m3=displacement,
        dt_s=1.0e-4,
    )

    assert piston.receiving_cells == ()
    assert piston.deposited_liquid_volume_m3 == 0.0
    assert piston.top_spill_volume_m3 == pytest.approx(displacement)
    assert piston.top_spill_momentum_kg_m_s == pytest.approx(
        S1_LIQUID_DENSITY_KG_M3 * displacement * velocity
    )
    assert piston.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-18)
    assert piston.liquid_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert remapped.upward_area[0] == pytest.approx(
        area - displacement / dz
    )
    assert remapped.upward_area[1:] == pytest.approx((area,) * (n - 1))


def test_full_stage_reports_bottom_piston_spill_at_the_atmospheric_boundary() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    state = VerticalState(
        Aup=(area,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )
    dt = 1.0e-8
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 1.02
    )
    gas_volume_rate = 1.0e-4
    node_density = pressure / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        gas_out_of_node_kg_s=node_density * gas_volume_rate,
        gas_out_of_node_speed_m_s=0.1,
    )
    trial = _trial(
        assembly,
        dt=dt,
        pressure=pressure,
        riser_flux=flux,
        state=state,
        physical_stage="stage2_pressure_reservoir",
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage2_pressure_reservoir",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted", evaluation.proposal.capacity_reject
    assert evaluation.diagnostics is not None
    piston = evaluation.diagnostics.bottom_gas_piston
    assert piston.top_spill_volume_m3 > 0.0
    assert evaluation.diagnostics.top_liquid.outflow_rate_m3_s >= (
        piston.top_spill_volume_m3 / dt
    )
    assert evaluation.proposal.external_exchange.liquid_outflow_m3_s == pytest.approx(
        evaluation.diagnostics.top_liquid.outflow_rate_m3_s
    )
    assert evaluation.diagnostics.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    assert evaluation.diagnostics.momentum_budget.residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-11
    )


def test_void_remap_never_teleports_a_closed_gas_gap_through_liquid() -> None:
    with pytest.raises(MissingPhysicalClosure, match="cross a resolved liquid column"):
        conservative_void_remap(
            old_void_area_m2=(0.0, 0.0, 1.0e-4, 0.0, 0.0),
            new_void_area_m2=(1.0e-4, 0.0, 0.0, 0.0, 0.0),
            gas_mass_cell_kg=(0.0, 0.0, 1.0e-6, 0.0, 0.0),
            gas_momentum_cell_kg_m_s=(0.0,) * 5,
            cell_length_m=0.01,
        )


def test_moving_water_front_remaps_existing_gas_without_massless_void() -> None:
    n = 40
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    source = assembly.state.vertical
    velocity = 0.01
    q = assembly.geometry.vertical_area_m2 * velocity
    state = VerticalState(
        Aup=source.Aup,
        Qup=tuple(q if area > 0.0 else 0.0 for area in source.Aup),
        Adown=source.Adown,
        Qdown=source.Qdown,
        Mg=source.Mg,
        Jg=source.Jg,
    )
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 0.5842
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        liquid_out_of_node_m3_s=q,
        liquid_out_of_node_speed_m_s=velocity,
    )
    dt = 1.0e-6
    trial = _trial(
        assembly,
        dt=dt,
        pressure=pressure,
        riser_flux=flux,
        state=state,
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    assert evaluation.diagnostics.void_remap.mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert evaluation.diagnostics.void_remap.momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-18
    )
    final = _after(state, evaluation.proposal.delta, dt)
    full = assembly.geometry.vertical_area_m2
    for up, down, mass in zip(final.Aup, final.Adown, final.Mg, strict=True):
        if full - up - down > 1.0e-14:
            assert mass > 0.0
    assert evaluation.diagnostics.capillary_interfaces[0].geometry_kind == (
        "declared_planar_semicircular_cap"
    )
    assert abs(
        evaluation.diagnostics.capillary_interfaces[0].record.curvature_1_m
    ) == pytest.approx(2.0 / PIPE_DIAMETER_M)


@pytest.mark.parametrize(
    ("pressure_ratio", "expected_direction"),
    ((1.05, "out"), (0.95, "in")),
)
def test_atmospheric_top_gas_Riemann_boundary_is_bidirectional(
    pressure_ratio: float, expected_direction: str
) -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    pressure = pressure_ratio * ATMOSPHERIC_PRESSURE_PA
    rho = pressure / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
    )
    state = VerticalState(
        Aup=(0.0,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(rho * area,) * n,
        Jg=(0.0,) * n,
    )
    dt = 1.0e-6
    trial = _trial(assembly, dt=dt, pressure=pressure, state=state)

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    top = evaluation.diagnostics.top_gas
    exchange = evaluation.proposal.external_exchange
    if expected_direction == "out":
        assert top.outflow_kg_s > 0.0
        assert top.inflow_kg_s == 0.0
        assert exchange.gas_outflow_kg_s == pytest.approx(top.outflow_kg_s)
    else:
        assert top.inflow_kg_s > 0.0
        assert top.outflow_kg_s == 0.0
        assert exchange.gas_inflow_kg_s == pytest.approx(top.inflow_kg_s)
    assert evaluation.diagnostics.gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.mixture_momentum_z_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    budget = evaluation.diagnostics.momentum_budget
    independently_known_external_impulse = (
        budget.liquid_pressure_impulse_kg_m_s
        + budget.gas_pressure_impulse_kg_m_s
        - budget.bottom_node_pressure_impulse_kg_m_s
        + budget.liquid_gravity_impulse_kg_m_s
        + budget.gas_gravity_impulse_kg_m_s
        + budget.wall_impulse_kg_m_s
        + budget.capillary_external_impulse_kg_m_s
    )
    assert (
        evaluation.proposal.external_exchange.external_force_z_N * dt
    ) == pytest.approx(independently_known_external_impulse, abs=2.0e-14)
    assert (
        budget.liquid_pressure_impulse_kg_m_s
        + budget.gas_pressure_impulse_kg_m_s
    ) == pytest.approx(
        budget.bottom_node_pressure_impulse_kg_m_s
        + budget.top_pressure_traction_impulse_kg_m_s
        + budget.discrete_pressure_traction_residual_kg_m_s,
        abs=2.0e-14,
    )
    assert abs(budget.discrete_pressure_traction_residual_kg_m_s) <= 2.0e-11


def _all_gas_case():
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    state = VerticalState(
        Aup=(0.0,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(RHO_G_ATM * area,) * n,
        Jg=(0.0,) * n,
    )
    dt = 1.0e-6
    trial = _trial(
        assembly,
        dt=dt,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        state=state,
    )
    return assembly, component, state, trial


def test_material_conservation_gate_catches_independent_mass_perturbation(
    monkeypatch,
) -> None:
    assembly, component, state, trial = _all_gas_case()
    original = F0VerticalPressureVoidStageComponent._transport_gas

    def tampered_transport(self, **kwargs):
        result = original(self, **kwargs)
        mass = list(result.gas_mass_cell_kg)
        mass[0] += 1.0e-8
        return replace(result, gas_mass_cell_kg=tuple(mass))

    monkeypatch.setattr(
        F0VerticalPressureVoidStageComponent,
        "_transport_gas",
        tampered_transport,
    )
    with pytest.raises(ContractViolation, match="material ledger"):
        component.evaluate_joint_stage(
            state,
            assembly.geometry,
            riser_node_trial=trial,
            physical_stage="stage1_closed",
            dt_s=trial.dt_s,
        )


def test_independent_momentum_gate_catches_final_momentum_perturbation(
    monkeypatch,
) -> None:
    assembly, component, state, trial = _all_gas_case()
    original = F0VerticalPressureVoidStageComponent._transport_gas

    def tampered_transport(self, **kwargs):
        result = original(self, **kwargs)
        momentum = list(result.gas_momentum_cell_kg_m_s)
        momentum[0] += 1.0e-6
        return replace(result, gas_momentum_cell_kg_m_s=tuple(momentum))

    monkeypatch.setattr(
        F0VerticalPressureVoidStageComponent,
        "_transport_gas",
        tampered_transport,
    )
    with pytest.raises(ContractViolation, match="independent physical budget"):
        component.evaluate_joint_stage(
            state,
            assembly.geometry,
            riser_node_trial=trial,
            physical_stage="stage1_closed",
            dt_s=trial.dt_s,
        )


def test_missing_bottom_pressure_traction_fails_closed_before_acceptance() -> None:
    assembly, component, state, trial = _all_gas_case()
    bad_fluxes = tuple(
        (
            replace(flux, pressure_traction_to_node_z_N=0.0)
            if flux.key.port_name == "riser_bottom"
            else flux
        )
        for flux in trial.gross_fluxes
    )
    bad_trial = replace(trial, gross_fluxes=bad_fluxes)

    proposal = component.propose_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=bad_trial,
        physical_stage="stage1_closed",
        dt_s=bad_trial.dt_s,
    )

    assert proposal.status == "capacity_rejected"
    assert proposal.capacity_reject is not None
    assert proposal.capacity_reject.reason_code == "missing_closure"
    assert "pressure traction" in proposal.capacity_reject.detail


def test_persistent_gross_streams_and_three_body_recoil_are_not_net_reconstructed() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    a_up = 0.30 * area
    a_down = 0.20 * area
    gas_area = 0.50 * area
    q_up = a_up * 0.20
    q_down = a_down * 0.15
    state = VerticalState(
        Aup=(a_up,) * n,
        Qup=(q_up,) * n,
        Adown=(a_down,) * n,
        Qdown=(q_down,) * n,
        Mg=(RHO_G_ATM * gas_area,) * n,
        Jg=(0.0,) * n,
    )
    fallback = AtmosphericLiquidFallback(
        donor_area_m2=a_down,
        downward_speed_m_s=0.15,
        available_volume_m3=1.0e-8,
    )
    component = _component(n, fallback=fallback)
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        liquid_into_node_m3_s=q_down,
        liquid_into_node_speed_m_s=0.15,
        liquid_out_of_node_m3_s=q_up,
        liquid_out_of_node_speed_m_s=0.20,
    )
    dt = 1.0e-7
    trial = _trial(
        assembly,
        dt=dt,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        riser_flux=flux,
        state=state,
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    final = _after(state, evaluation.proposal.delta, dt)
    assert all(value > 0.0 for value in final.Aup)
    assert all(value > 0.0 for value in final.Adown)
    assert all(value > 0.0 for value in final.Qup)
    assert all(value > 0.0 for value in final.Qdown)
    assert final.Qup != final.Qdown
    assert evaluation.diagnostics is not None
    assert evaluation.diagnostics.three_body_recoil_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-12
    )
    assert evaluation.diagnostics.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-14
    )


def test_dynamic_wall_friction_uses_local_Re_perimeter_and_hydraulic_diameter() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    velocity = 0.25
    q = area * velocity
    state = VerticalState(
        Aup=(area,) * n,
        Qup=(q,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )
    pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3 * S1_GRAVITY_M_S2 * 1.02
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        liquid_out_of_node_m3_s=q,
        liquid_out_of_node_speed_m_s=velocity,
    )
    dt = 1.0e-5
    trial = _trial(
        assembly,
        dt=dt,
        pressure=pressure,
        riser_flux=flux,
        state=state,
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    wall = evaluation.diagnostics.wall
    reynolds = (
        S1_LIQUID_DENSITY_KG_M3
        * velocity
        * PIPE_DIAMETER_M
        / 1.002e-3
    )
    expected = f0_smooth_pipe_darcy_factor(reynolds)
    assert wall.liquid_up_hydraulic_diameter_m == pytest.approx(
        (PIPE_DIAMETER_M,) * n
    )
    assert wall.liquid_up_darcy_factor == pytest.approx((expected,) * n)
    assert wall.wall_impulse_kg_m_s < 0.0
    final = _after(state, evaluation.proposal.delta, dt)
    assert max(final.Qup) < q


def test_capacity_rejection_has_no_delta_and_does_not_mutate_state() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    area = assembly.geometry.vertical_area_m2
    state = VerticalState(
        Aup=(0.0,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(RHO_G_ATM * area,) * n,
        Jg=(0.0,) * n,
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        gas_into_node_kg_s=1.0,
        gas_into_node_speed_m_s=1.0,
    )
    dt = 1.0e-3
    trial = _trial(
        assembly,
        dt=dt,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        riser_flux=flux,
        state=state,
    )
    before = repr(state)

    proposal = component.propose_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert proposal.status == "capacity_rejected"
    assert proposal.delta is None
    assert proposal.accepted_gross_fluxes == ()
    assert proposal.capacity_reject is not None
    assert proposal.capacity_reject.reason_code == "cfl"
    assert repr(state) == before


def test_massless_void_and_unseeded_void_from_countercurrent_liquid_fail_closed() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    component = _component(n)
    massless = VerticalState(
        Aup=(0.0,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )
    trial = _trial(
        assembly,
        dt=1.0e-5,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        state=massless,
    )
    rejected = component.propose_joint_stage(
        massless,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=trial.dt_s,
    )
    assert rejected.status == "capacity_rejected"
    assert rejected.capacity_reject.reason_code == "void_mass_pairing"

    falling = VerticalState(
        Aup=(0.8 * area,) * n,
        Qup=(0.0,) * n,
        Adown=(0.2 * area,) * n,
        Qdown=(0.2 * area * 0.1,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )
    falling_trial = _trial(
        assembly,
        dt=1.0e-6,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        state=falling,
    )
    falling_reject = component.propose_joint_stage(
        falling,
        assembly.geometry,
        riser_node_trial=falling_trial,
        physical_stage="stage1_closed",
        dt_s=falling_trial.dt_s,
    )
    assert falling_reject.status == "capacity_rejected"
    # Coexisting Aup/Adown in one cut cell is the pinned Case-1 local
    # two-fluid topology and is not itself an axial gas-gap crossing.  This
    # manufactured all-liquid column still fails closed when its motion opens
    # a finite void without an owned gas parcel.
    assert falling_reject.capacity_reject.reason_code == "void_mass_pairing"


def test_top_reentry_uses_only_a_finite_explicit_exterior_parcel() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    a_up = 0.30 * area
    a_down = 0.20 * area
    q_up = a_up * 0.20
    q_down = a_down * 0.15
    dt = 1.0e-7
    available = 0.25 * q_down * dt
    fallback = AtmosphericLiquidFallback(
        donor_area_m2=a_down,
        downward_speed_m_s=0.15,
        available_volume_m3=available,
    )
    component = _component(n, fallback=fallback)
    state = VerticalState(
        Aup=(a_up,) * n,
        Qup=(q_up,) * n,
        Adown=(a_down,) * n,
        Qdown=(q_down,) * n,
        Mg=(RHO_G_ATM * 0.50 * area,) * n,
        Jg=(0.0,) * n,
    )
    flux = GrossNodePortFlux(
        key=PortKey("riser_T", "riser_bottom"),
        liquid_into_node_m3_s=q_down,
        liquid_into_node_speed_m_s=0.15,
        liquid_out_of_node_m3_s=q_up,
        liquid_out_of_node_speed_m_s=0.20,
    )
    trial = _trial(
        assembly,
        dt=dt,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        riser_flux=flux,
        state=state,
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted"
    assert evaluation.diagnostics is not None
    top = evaluation.diagnostics.top_liquid
    assert top.reentry_demand_rate_m3_s == pytest.approx(q_down)
    assert top.reentry_rate_m3_s == pytest.approx(available / dt)
    assert top.stage_consumed_volume_m3 == pytest.approx(available)
    assert top.stage_consumed_volume_m3 <= top.exterior_available_volume_m3
    assert top.finite_exterior_inventory is True
    assert evaluation.proposal.external_exchange.liquid_inflow_m3_s == pytest.approx(
        available / dt
    )
    assert evaluation.diagnostics.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert evaluation.diagnostics.momentum_budget is not None
    assert evaluation.diagnostics.momentum_budget.residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-11
    )


def test_finite_falling_parcel_can_launch_reentry_from_zero_interior_qdown() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    a_up = 0.0
    a_down = 0.20 * area
    speed = 0.15
    dt = 1.0e-7
    rate = a_down * speed
    fallback = AtmosphericLiquidFallback(
        donor_area_m2=a_down,
        downward_speed_m_s=speed,
        available_volume_m3=2.0 * rate * dt,
        evidence_status="test_finite_incident_returning_queue",
    )
    component = _component(n, fallback=fallback)
    state = VerticalState(
        Aup=(a_up,) * n,
        Qup=(0.0,) * n,
        Adown=(a_down,) * n,
        Qdown=(0.0,) * n,
        Mg=(RHO_G_ATM * 0.80 * area,) * n,
        Jg=(0.0,) * n,
    )
    trial = _trial(
        assembly,
        dt=dt,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        state=state,
    )

    evaluation = component.evaluate_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=dt,
    )

    assert evaluation.proposal.status == "accepted", evaluation.proposal.capacity_reject
    assert evaluation.diagnostics is not None
    top = evaluation.diagnostics.top_liquid
    assert top.reentry_demand_rate_m3_s == pytest.approx(rate)
    assert top.reentry_rate_m3_s == pytest.approx(rate)
    assert top.reentry_speed_m_s == pytest.approx(speed)
    assert evaluation.proposal.external_exchange.liquid_inflow_m3_s == pytest.approx(
        rate
    )


def test_top_reentry_without_finite_exterior_inventory_fails_closed() -> None:
    n = 8
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    area = assembly.geometry.vertical_area_m2
    a_down = 0.20 * area
    state = VerticalState(
        Aup=(0.30 * area,) * n,
        Qup=(0.30 * area * 0.20,) * n,
        Adown=(a_down,) * n,
        Qdown=(a_down * 0.15,) * n,
        Mg=(RHO_G_ATM * 0.50 * area,) * n,
        Jg=(0.0,) * n,
    )
    component = _component(
        n,
        fallback=AtmosphericLiquidFallback(
            donor_area_m2=a_down,
            downward_speed_m_s=0.15,
        ),
    )
    trial = _trial(
        assembly,
        dt=1.0e-7,
        pressure=ATMOSPHERIC_PRESSURE_PA,
        state=state,
    )

    proposal = component.propose_joint_stage(
        state,
        assembly.geometry,
        riser_node_trial=trial,
        physical_stage="stage1_closed",
        dt_s=trial.dt_s,
    )

    assert proposal.status == "capacity_rejected"
    assert proposal.capacity_reject is not None
    assert proposal.capacity_reject.reason_code == "missing_closure"
    assert "finite exterior parcel" in proposal.capacity_reject.detail


def test_top_reentry_parcel_pressure_must_match_atmospheric_boundary() -> None:
    with pytest.raises(ContractViolation, match="atmospheric top pressure differ"):
        AtmosphericTopState(
            liquid_fallback=AtmosphericLiquidFallback(
                donor_area_m2=1.0e-6,
                downward_speed_m_s=0.1,
                available_volume_m3=1.0e-8,
                absolute_pressure_Pa=ATMOSPHERIC_PRESSURE_PA + 100.0,
            )
        )


def test_vertical_component_does_not_claim_production_readiness_early() -> None:
    component = _component(
        8,
        fallback=AtmosphericLiquidFallback(
            donor_area_m2=1.0e-6,
            downward_speed_m_s=0.1,
            available_volume_m3=1.0e-8,
        ),
    )

    assert component.capillary_owner.production_ready is True
    assert component.atmospheric_top.finite_stage_liquid_reentry_ready is True
    assert component.atmospheric_top.full_cycle_liquid_fallback_ready is False
    assert component.production_ready is False
    assert component.source_aligned_trajectory_ready is False


def test_capillary_owner_distinguishes_planar_and_3d_without_guessing_angle() -> None:
    n = 20
    assembly = build_s1_initial_assembly(vertical_cell_count=n)
    component = _component(n)
    state = component.initial_state
    moving = VerticalState(
        Aup=state.Aup,
        Qup=tuple(0.05 * area for area in state.Aup),
        Adown=state.Adown,
        Qdown=state.Qdown,
        Mg=state.Mg,
        Jg=state.Jg,
    )
    signed = _component_state(component._solver._runtime, moving)
    mass = tuple(value * component._solver.cell_length_m for value in moving.Mg)
    momentum = tuple(value * component._solver.cell_length_m for value in moving.Jg)
    planar = F0VerticalCapillaryOwner(mode="planar_2d_zeroGradient_walls")
    detected = planar.detect(
        signed,
        gas_mass_cell_kg=mass,
        gas_momentum_cell_kg_m_s=momentum,
        full_area_m2=component._solver.pipe_area_m2,
        diameter_m=PIPE_DIAMETER_M,
    )
    assert len(detected) == 1
    assert detected[0].gas_is_above is True
    assert detected[0].record.curvature_1_m == pytest.approx(
        2.0 / PIPE_DIAMETER_M
    )
    assert detected[0].record.contact_angle_deg is None

    # Reversing only the declared phase topology reverses the curvature sign;
    # no result observable or calibration switch participates in this choice.
    area = assembly.geometry.vertical_area_m2
    half = n // 2
    reversed_state = VerticalState(
        Aup=(0.0,) * half + (area,) * (n - half),
        Qup=(0.0,) * half + (0.05 * area,) * (n - half),
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(RHO_G_ATM * area,) * half + (0.0,) * (n - half),
        Jg=(0.0,) * n,
    )
    reversed_signed = _component_state(
        component._solver._runtime, reversed_state
    )
    reversed_mass = tuple(
        value * component._solver.cell_length_m for value in reversed_state.Mg
    )
    reversed_detected = planar.detect(
        reversed_signed,
        gas_mass_cell_kg=reversed_mass,
        gas_momentum_cell_kg_m_s=(0.0,) * n,
        full_area_m2=component._solver.pipe_area_m2,
        diameter_m=PIPE_DIAMETER_M,
    )
    assert len(reversed_detected) == 1
    assert reversed_detected[0].gas_is_above is False
    assert reversed_detected[0].record.curvature_1_m == pytest.approx(
        -2.0 / PIPE_DIAMETER_M
    )

    circular_missing = F0VerticalCapillaryOwner(mode="circular_3d_pipe")
    assert circular_missing.production_ready is False
    with pytest.raises(MissingPhysicalClosure, match="contact angle"):
        circular_missing.detect(
            signed,
            gas_mass_cell_kg=mass,
            gas_momentum_cell_kg_m_s=momentum,
            full_area_m2=component._solver.pipe_area_m2,
            diameter_m=PIPE_DIAMETER_M,
        )

    unselected = F0VerticalCapillaryOwner()
    assert unselected.production_ready is False
    with pytest.raises(MissingPhysicalClosure, match="unselected"):
        unselected.detect(
            signed,
            gas_mass_cell_kg=mass,
            gas_momentum_cell_kg_m_s=momentum,
            full_area_m2=component._solver.pipe_area_m2,
            diameter_m=PIPE_DIAMETER_M,
        )
