"""Regression tests for the independent Case-A S|P branch Euler core."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import casea_mixed_branch_fv as mixed_branch_fv  # noqa: E402
from casea_material_front_cutcell import (  # noqa: E402
    MaterialFrontCutCell,
    PressurisedFlux,
    PressurisedState,
    StratifiedFlux,
    StratifiedState,
)
from casea_material_front_rh_adapter import (  # noqa: E402
    build_casea_material_front_traces,
)
from casea_mixed_branch_fv import (  # noqa: E402
    GasRiemannDiagnostics,
    MixedBranchBoundActivationError,
    MixedBranchBoundaryFluxes,
    MixedBranchCrossingRequired,
    MixedBranchGasFallbackError,
    MixedBranchParameters,
    MixedBranchScopeError,
    MixedBranchSources,
    MixedBranchState,
    PressurisedSource,
    StratifiedSource,
    audit_stratified_state_bounds,
    build_mixed_branch_interface_traces,
    euler_mixed_branch_stage,
    pressurised_physical_flux,
    stratified_liquid_potential_offset,
    stratified_numerical_flux_with_diagnostics,
    stratified_physical_flux,
)
from casea_tjunction_shock_network import BranchGeometry  # noqa: E402


def _equilibrium_states(
    *, gas_pressure_offset: float = 0.0
) -> tuple[MixedBranchParameters, PressurisedState, StratifiedState]:
    geometry = BranchGeometry(0.094, 0.490, 28.0)
    params = MixedBranchParameters(geometry=geometry)
    section = geometry.section(params.gravity)
    depth = 0.070
    liquid_area = float(section.area_from_depth(depth))
    # Exact zero-speed traction balance at A_p=A_full:
    # g I_full = g I_1(h) + (p_g-p_atm) A_full/rho_l.
    pressure = (
        params.atmospheric_pressure
        + params.liquid_density
        * params.gravity
        * (
            section.full_hydrostatic_moment
            - float(section.hydrostatic_moment(depth))
        )
        / section.full_area
        + gas_pressure_offset
    )
    gas_area = section.full_area - liquid_area
    gas_density = pressure / (
        params.gas_constant * params.gas_temperature
    )
    return (
        params,
        PressurisedState(section.full_area, 0.0),
        StratifiedState(
            gas_density * gas_area,
            0.0,
            liquid_area,
            0.0,
        ),
    )


def _branch(
    pressurised: PressurisedState,
    stratified: StratifiedState,
    *,
    front_position: float = 0.06,
) -> MixedBranchState:
    faces = (0.0, 0.04, 0.08, 0.12)
    host = MaterialFrontCutCell(
        cell_faces=faces,
        host_index=1,
        front_position=front_position,
        pressurised_side="right",
        pressurised=pressurised,
        stratified=stratified,
    )
    return MixedBranchState(
        host=host,
        stratified_cells=(stratified,),
        pressurised_cells=(pressurised,),
    )


def _physical_boundaries(
    state: MixedBranchState,
    params: MixedBranchParameters,
) -> MixedBranchBoundaryFluxes:
    return MixedBranchBoundaryFluxes(
        left_stratified=stratified_physical_flux(
            state.stratified_cells[0]
            if state.stratified_cells
            else state.host.stratified,
            params,
        ),
        right_pressurised=pressurised_physical_flux(
            state.pressurised_cells[-1]
            if state.pressurised_cells
            else state.host.pressurised,
            params,
        ),
    )


def _assert_ledger_closed(result, *, atol: float = 5.0e-14) -> None:
    assert result.ledger.residual.vector() == pytest.approx(
        (0.0, 0.0, 0.0, 0.0), abs=atol
    )


def test_static_rh_equilibrium_preserves_all_regular_and_host_states() -> None:
    params, pressurised, stratified = _equilibrium_states()
    state = _branch(pressurised, stratified)
    traces = build_mixed_branch_interface_traces(state, params)
    assert abs(traces.speed) < 2.0e-12

    result = euler_mixed_branch_stage(
        state,
        1.0e-3,
        params=params,
        boundary_fluxes=_physical_boundaries(state, params),
    )

    assert result.state.host.front_position == pytest.approx(
        state.host.front_position, abs=2.0e-15
    )
    assert result.state.stratified_cells[0].vector() == pytest.approx(
        stratified.vector(), abs=2.0e-14
    )
    assert result.state.pressurised_cells[0].vector() == pytest.approx(
        pressurised.vector(), abs=2.0e-14
    )
    assert result.state.host.stratified.vector() == pytest.approx(
        stratified.vector(), abs=2.0e-14
    )
    assert result.state.host.pressurised.vector() == pytest.approx(
        pressurised.vector(), abs=2.0e-14
    )
    _assert_ledger_closed(result)


def test_zero_length_stratified_launch_grows_only_by_boundary_and_ale_flux() -> None:
    params, pressurised_foot, stratified_foot = _equilibrium_states(
        gas_pressure_offset=500.0
    )
    built = build_casea_material_front_traces(
        pressurised_foot,
        stratified_foot,
        front_position=0.0,
        geometry=params.geometry,
        atmospheric_pressure=params.atmospheric_pressure,
        liquid_density=params.liquid_density,
        gravity=params.gravity,
        gas_sound_speed=params.gas_sound_speed,
    )
    assert built.traces.speed > 0.0
    faces = (0.0, 0.04)
    host = MaterialFrontCutCell(
        cell_faces=faces,
        host_index=0,
        front_position=0.0,
        pressurised_side="right",
        pressurised=built.traces.pressurised_state,
        stratified=built.traces.stratified_state,
    )
    state = MixedBranchState(host, (), ())
    boundaries = MixedBranchBoundaryFluxes(
        left_stratified=built.traces.stratified_flux,
        right_pressurised=built.traces.pressurised_flux,
    )
    dt = 1.0e-3

    result = euler_mixed_branch_stage(
        state,
        dt,
        params=params,
        boundary_fluxes=boundaries,
        interface_traces=built.traces,
    )

    assert state.host.stratified_length == 0.0
    assert result.state.host.stratified_length == pytest.approx(
        built.traces.speed * dt, abs=2.0e-15
    )
    assert result.state.host.inventory().gas_mass == pytest.approx(
        dt * built.traces.stratified_flux.gas_mass,
        rel=2.0e-12,
        abs=1.0e-16,
    )
    _assert_ledger_closed(result)


def test_host_and_neighbour_use_the_identical_shared_face_flux_objects() -> None:
    params, pressurised, stratified = _equilibrium_states()
    state = _branch(pressurised, stratified)
    result = euler_mixed_branch_stage(
        state,
        1.0e-4,
        params=params,
        boundary_fluxes=_physical_boundaries(state, params),
    )

    assert (
        result.host_outer_fluxes.stratified
        is result.stratified_face_fluxes[-1]
    )
    assert (
        result.host_outer_fluxes.pressurised
        is result.pressurised_face_fluxes[0]
    )
    assert (
        result.host_advance.ledgers[0].interface_flux
        == result.ledger.interface_flux
    )


def test_full_branch_liquid_gas_ledger_closes_with_distributed_sources() -> None:
    params, pressurised, stratified = _equilibrium_states()
    state = _branch(pressurised, stratified)
    boundaries = MixedBranchBoundaryFluxes(
        left_stratified=StratifiedFlux(
            gas_mass=2.0e-5,
            gas_momentum=0.29,
            liquid_area=1.0e-5,
            liquid_momentum=0.0031,
        ),
        right_pressurised=PressurisedFlux(
            area=-0.5e-5,
            momentum=0.0030,
        ),
    )
    sources = MixedBranchSources(
        stratified=(
            StratifiedSource(1.0e-6, -2.0e-5, 3.0e-6, -4.0e-5),
        ),
        pressurised=(PressurisedSource(-2.0e-6, 5.0e-5),),
    )

    result = euler_mixed_branch_stage(
        state,
        2.0e-5,
        params=params,
        boundary_fluxes=boundaries,
        sources=sources,
    )

    actual = tuple(
        final - initial
        for final, initial in zip(
            result.ledger.final.vector(), result.ledger.initial.vector()
        )
    )
    assert actual == pytest.approx(
        result.ledger.expected_change.vector(), abs=5.0e-14
    )
    _assert_ledger_closed(result)


@pytest.mark.parametrize(
    ("pressure_offset", "expected_sign"),
    [(-500.0, -1), (500.0, 1)],
)
def test_default_rh_adapter_advances_positive_and_reverse_fronts(
    pressure_offset: float,
    expected_sign: int,
) -> None:
    params, pressurised, stratified = _equilibrium_states(
        gas_pressure_offset=pressure_offset
    )
    state = _branch(pressurised, stratified)
    traces = build_mixed_branch_interface_traces(state, params)
    assert math.copysign(1.0, traces.speed) == expected_sign
    dt = 1.0e-3

    result = euler_mixed_branch_stage(
        state,
        dt,
        params=params,
        boundary_fluxes=_physical_boundaries(state, params),
    )

    assert result.state.host.front_position == pytest.approx(
        state.host.front_position + traces.speed * dt,
        abs=3.0e-15,
    )
    _assert_ledger_closed(result)


@pytest.mark.parametrize(
    ("pressure_offset", "front_position", "expected_direction"),
    [(-500.0, 0.0401, -1), (500.0, 0.0799, 1)],
)
def test_crossing_inside_stage_is_rejected_with_exact_event_time(
    pressure_offset: float,
    front_position: float,
    expected_direction: int,
) -> None:
    params, pressurised, stratified = _equilibrium_states(
        gas_pressure_offset=pressure_offset
    )
    state = _branch(
        pressurised,
        stratified,
        front_position=front_position,
    )
    traces = build_mixed_branch_interface_traces(state, params)
    face = 0.04 if expected_direction < 0 else 0.08
    exact_time = (face - front_position) / traces.speed
    initial_inventory = state.inventory()

    with pytest.raises(MixedBranchCrossingRequired) as caught:
        euler_mixed_branch_stage(
            state,
            2.0 * exact_time,
            params=params,
            boundary_fluxes=_physical_boundaries(state, params),
        )

    event = caught.value
    assert event.crossing_time == pytest.approx(exact_time, abs=2.0e-15)
    assert event.face_position == face
    assert event.moving_direction == expected_direction
    assert state.inventory() == initial_inventory


def _state_with_area_and_density(
    params: MixedBranchParameters,
    *,
    liquid_area: float,
    gas_density: float,
    gas_velocity: float = 0.0,
) -> StratifiedState:
    area_full = params.geometry.section(params.gravity).full_area
    gas_area = area_full - liquid_area
    assert gas_area > 0.0
    gas_mass = gas_density * gas_area
    return StratifiedState(
        gas_mass,
        gas_mass * gas_velocity,
        liquid_area,
        0.0,
    )


def test_normal_case_a_face_reports_roe_and_no_fallback() -> None:
    params, _, stratified = _equilibrium_states()
    offset = stratified_liquid_potential_offset(stratified, params)

    evaluated = stratified_numerical_flux_with_diagnostics(
        stratified,
        stratified,
        params,
        liquid_potential_offset=offset,
    )

    gas = evaluated.diagnostics.gas
    assert gas.solver == "positive-density Roe"
    assert gas.roe_used is True
    assert gas.fallback_used is False
    assert gas.fallback_name is None
    assert gas.roe_density_floor_active is False
    assert evaluated.diagnostics.liquid_left.bound_audit.accepted_without_bound
    assert evaluated.diagnostics.liquid_right.bound_audit.accepted_without_bound


def test_stage_collects_face_solver_and_eq40_diagnostics() -> None:
    params, pressurised, stratified = _equilibrium_states()
    state = _branch(pressurised, stratified)

    result = euler_mixed_branch_stage(
        state,
        1.0e-4,
        params=params,
        boundary_fluxes=_physical_boundaries(state, params),
    )

    assert len(result.numerics.stratified_faces) == 1
    assert result.numerics.roe_face_count == 1
    assert result.numerics.fallback_face_count == 0
    assert result.numerics.eq40_nonpositive_state_count == 0
    potential = result.numerics.stratified_faces[0].liquid_left
    assert potential.eq40_floor_term_added is True
    assert potential.eq40_floor_term == pytest.approx(1.0e-6)
    assert potential.numerical_celerity_squared == pytest.approx(
        potential.physical_celerity_squared + potential.eq40_floor_term
    )


@pytest.mark.parametrize(
    ("kind", "area_fraction", "density_fraction", "expected_bound"),
    [
        ("geometry", 0.996, 1.0, "geometry_cap"),
        ("void", 1.0 - 0.5e-4, 1.0, "void_floor"),
        ("density-low", 0.80, 0.10, "gas_density_floor"),
        ("density-high", 0.80, 13.0, "gas_density_ceiling"),
    ],
)
def test_every_shared_pressure_bound_is_rejected_before_clipping(
    kind: str,
    area_fraction: float,
    density_fraction: float,
    expected_bound: str,
) -> None:
    del kind
    params, _, _ = _equilibrium_states()
    liquid = params.horizontal_liquid
    state = _state_with_area_and_density(
        params,
        liquid_area=area_fraction * liquid.area_full,
        gas_density=density_fraction * liquid.atmospheric_gas_density,
    )

    with pytest.raises(MixedBranchBoundActivationError) as caught:
        audit_stratified_state_bounds(state, params, state_label="test state")

    assert expected_bound in caught.value.audit.active_bounds
    assert expected_bound in str(caught.value)
    assert caught.value.audit.accepted_without_bound is False


def test_invalid_bound_is_rejected_before_pressure_operator_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params, _, _ = _equilibrium_states()
    liquid = params.horizontal_liquid
    invalid = _state_with_area_and_density(
        params,
        liquid_area=0.996 * liquid.area_full,
        gas_density=liquid.atmospheric_gas_density,
    )

    def forbidden_pressure_call(*_args, **_kwargs):
        raise AssertionError("pressure operator was called before the bound audit")

    monkeypatch.setattr(
        mixed_branch_fv,
        "pressure_potential_state",
        forbidden_pressure_call,
    )
    with pytest.raises(MixedBranchBoundActivationError):
        stratified_numerical_flux_with_diagnostics(
            invalid,
            invalid,
            params,
            liquid_potential_offset=0.0,
        )


def test_eq40_nonpositive_tangent_branch_is_retained_but_exposed() -> None:
    params, _, stratified = _equilibrium_states()
    high_slip = StratifiedState(
        stratified.gas_mass,
        50.0 * stratified.gas_mass,
        stratified.liquid_area,
        stratified.liquid_discharge,
    )
    offset = stratified_liquid_potential_offset(high_slip, params)

    evaluated = stratified_numerical_flux_with_diagnostics(
        high_slip,
        high_slip,
        params,
        liquid_potential_offset=offset,
    )

    potential = evaluated.diagnostics.liquid_left
    assert potential.physical_celerity_squared < 0.0
    assert potential.eq40_nonpositive_tangent_branch_active is True
    assert potential.numerical_celerity_squared == pytest.approx(
        potential.eq40_floor_term
    )
    assert evaluated.diagnostics.gas.roe_used is True
    assert evaluated.diagnostics.gas.fallback_used is False


def test_production_face_fails_closed_if_a_fallback_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params, _, stratified = _equilibrium_states()
    offset = stratified_liquid_potential_offset(stratified, params)

    fallback = GasRiemannDiagnostics(
        solver="Einfeldt",
        roe_used=False,
        fallback_used=True,
        fallback_name="Einfeldt/HLL two-wave flux",
        density_left=1.0,
        density_right=1.0,
        roe_internal_density_floor=1.0e-10,
        roe_density_floor_active=False,
    )

    def reported_fallback(*_args, **_kwargs):
        return 0.0, 1.0, fallback

    monkeypatch.setattr(
        mixed_branch_fv,
        "_audited_roe_gas_flux",
        reported_fallback,
    )
    with pytest.raises(MixedBranchGasFallbackError, match="fallback"):
        stratified_numerical_flux_with_diagnostics(
            stratified,
            stratified,
            params,
            liquid_potential_offset=offset,
        )


def test_horizontal_operator_explicitly_rejects_vertical_branch_geometry() -> None:
    vertical = BranchGeometry(0.0571, 0.305, 28.0, bed_slope=1.0)

    with pytest.raises(MixedBranchScopeError, match="horizontal-only"):
        MixedBranchParameters(geometry=vertical)
