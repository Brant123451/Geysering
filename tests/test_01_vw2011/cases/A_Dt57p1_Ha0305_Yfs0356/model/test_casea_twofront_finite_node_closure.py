"""Physical and conservation tests for the finite T-node launch closure."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_gas_coupled_front import GasCellTrace  # noqa: E402
from casea_material_front_cutcell import ALEInterfaceFlux  # noqa: E402
from casea_tjunction_shock_network import (  # noqa: E402
    IncompatibleZeroStoragePressure,
    PressureSolveError,
    solve_zero_storage_t_node,
)
from casea_twofront_finite_node_closure import (  # noqa: E402
    FiniteNodeGasState,
    PRODUCTION_READY,
    solve_finite_node_twofront_launch,
)
from casea_twofront_launch_closure import (  # noqa: E402
    PRODUCTION_READY as DECOUPLED_PRODUCTION_READY,
    evaluate_twofront_launch_candidate,
)
from test_casea_twofront_launch_closure import (  # noqa: E402
    C_G,
    RHO_L,
    _Fixture,
)


def _inputs(fixture: _Fixture) -> dict[str, object]:
    return {
        "west_gas_face_area": fixture.west_gas_area,
        "liquid_characteristics": fixture.characteristics,
        "liquid_areas": fixture.areas,
        "east_geometry": fixture.east,
        "vertical_geometry": fixture.vertical,
        "east_pressurised_foot": fixture.foot(
            fixture.east, fixture.east_depth
        ),
        "vertical_pressurised_foot": fixture.foot(
            fixture.vertical, fixture.vertical_depth
        ),
        "east_stratified_liquid_area": fixture.east_area,
        "vertical_stratified_liquid_area": fixture.vertical_area,
    }


def _trace(fixture: _Fixture, velocity: float) -> GasCellTrace:
    return GasCellTrace(
        density=fixture.pressure / C_G**2,
        velocity=velocity,
        sound_speed=C_G,
    )


def test_same_pressure_zero_storage_and_gas_launch_are_generically_incompatible() -> None:
    fixture = _Fixture()
    liquid_zero = solve_zero_storage_t_node(
        fixture.characteristics,
        fixture.areas,
        liquid_density=RHO_L,
    )
    trace = _trace(fixture, 0.20)
    # At the unique zero-storage pressure the independent gas launch balance
    # is nonzero.  Therefore no single pressure can satisfy both equations.
    gas_at_liquid_root = evaluate_twofront_launch_candidate(
        gas_node_pressure_abs=liquid_zero.node_pressure_abs,
        west_gas_trace=trace,
        west_gas_face_area=fixture.west_gas_area,
        liquid_node=liquid_zero,
        east_geometry=fixture.east,
        vertical_geometry=fixture.vertical,
        east_pressurised_foot=fixture.foot(
            fixture.east, fixture.east_depth
        ),
        vertical_pressurised_foot=fixture.foot(
            fixture.vertical, fixture.vertical_depth
        ),
        east_stratified_liquid_area=fixture.east_area,
        vertical_stratified_liquid_area=fixture.vertical_area,
    )
    assert abs(gas_at_liquid_root.gas_mass_balance_residual) > 1.0e-6

    # Conversely, the decoupled gas root cannot be imposed on the massless
    # liquid node.  The existing API reports the missing storage rate.
    decoupled = fixture.solve(0.20)
    with pytest.raises(IncompatibleZeroStoragePressure):
        solve_zero_storage_t_node(
            fixture.characteristics,
            fixture.areas,
            liquid_density=RHO_L,
            required_gas_pressure_abs=decoupled.gas_node_pressure_abs,
        )
    assert not DECOUPLED_PRODUCTION_READY
    # This historical closure diagnoses the missing finite storage, but its
    # state stores no elastic liquid inventory.  It is intentionally retained
    # as a non-production regression model; the production replacement is
    # casea_compressible_finite_node.
    assert not PRODUCTION_READY


def test_production_time_loop_does_not_import_decoupled_experimental_closure() -> None:
    production_loop = (MODEL_DIR / "vw2011_network_twofluid.py").read_text(
        encoding="utf-8"
    )
    assert "casea_twofront_launch_closure" not in production_loop
    assert "solve_twofront_launch_closure" not in production_loop


def test_finite_geometric_node_is_exactly_stationary_at_rest() -> None:
    fixture = _Fixture()
    volume = 1.0e-4
    initial = FiniteNodeGasState(
        gas_volume=volume,
        gas_mass=fixture.pressure * volume / C_G**2,
        node_total_volume=3.5e-4,
    )
    result = solve_finite_node_twofront_launch(
        initial,
        dt=1.0e-3,
        west_gas_trace=_trace(fixture, 0.0),
        **_inputs(fixture),
    )
    assert result.pressure_abs == fixture.pressure
    assert result.state == initial
    assert result.liquid_outward_volume_rate == 0.0
    assert result.west_gas_flux.mass_rate == 0.0
    assert result.receiver_mass_rate == 0.0
    assert result.eos_residual == 0.0


def test_exact_zero_launch_has_a_finite_dt_independent_pressure_limit() -> None:
    fixture = _Fixture()
    initial = FiniteNodeGasState(0.0, 0.0, 3.5e-4)
    results = [
        solve_finite_node_twofront_launch(
            initial,
            dt=dt,
            west_gas_trace=_trace(fixture, 0.20),
            **_inputs(fixture),
        )
        for dt in (1.0e-3, 1.0e-5, 1.0e-7)
    ]
    reference = results[0]
    assert reference.pressure_abs > fixture.pressure
    assert reference.east_traces.speed > 0.0
    assert reference.vertical_traces.speed > 0.0
    for result in results[1:]:
        assert math.isclose(
            result.pressure_abs,
            reference.pressure_abs,
            rel_tol=0.0,
            abs_tol=2.0e-7,
        )
        assert math.isclose(
            result.east_traces.speed,
            reference.east_traces.speed,
            rel_tol=0.0,
            abs_tol=2.0e-10,
        )
    # Inventory scales with dt while p=m*c^2/V remains finite and invariant.
    assert math.isclose(
        results[0].state.gas_volume / results[1].state.gas_volume,
        100.0,
        rel_tol=1.0e-9,
    )
    for result in results:
        assert math.isclose(
            result.state.gas_mass * C_G**2 / result.state.gas_volume,
            result.pressure_abs,
            rel_tol=1.0e-11,
        )


def test_one_pressure_drives_liquid_node_rh_and_west_gas_characteristic() -> None:
    fixture = _Fixture()
    result = solve_finite_node_twofront_launch(
        FiniteNodeGasState(0.0, 0.0, 3.5e-4),
        dt=1.0e-4,
        west_gas_trace=_trace(fixture, 0.20),
        **_inputs(fixture),
    )
    assert result.liquid_node.node_pressure_abs == result.pressure_abs
    assert result.west_gas_flux.pressure_abs == result.pressure_abs
    for name, flux in result.liquid_node.branch_fluxes.items():
        characteristic = getattr(fixture.characteristics, name)
        assert flux.face_pressure_abs == (
            result.pressure_abs + characteristic.pressure_offset
        )
    for traces, geometry in (
        (result.east_traces, fixture.east),
        (result.vertical_traces, fixture.vertical),
    ):
        gas_area = (
            geometry.section(9.81).full_area
            - traces.stratified_state.liquid_area
        )
        gas_density = traces.stratified_state.gas_mass / gas_area
        assert math.isclose(
            gas_density * C_G**2,
            result.pressure_abs,
            rel_tol=1.0e-12,
        )


def test_finite_node_liquid_gas_eos_and_rh_ledgers_close_without_double_count() -> None:
    fixture = _Fixture()
    initial = FiniteNodeGasState(0.0, 0.0, 3.5e-4)
    dt = 1.0e-4
    result = solve_finite_node_twofront_launch(
        initial,
        dt=dt,
        west_gas_trace=_trace(fixture, 0.20),
        **_inputs(fixture),
    )
    assert math.isclose(
        result.state.gas_volume - initial.gas_volume,
        dt * result.liquid_outward_volume_rate,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    assert math.isclose(
        result.state.liquid_volume - initial.liquid_volume,
        -dt * result.liquid_outward_volume_rate,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    assert math.isclose(
        result.state.gas_mass
        - initial.gas_mass
        + dt * result.receiver_mass_rate,
        dt * result.west_gas_flux.mass_rate,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    assert abs(result.liquid_storage_balance_residual) < 1.0e-18
    assert abs(result.gas_node_mass_balance_residual) < 1.0e-18
    assert abs(result.total_gas_mass_balance_residual) < 1.0e-18
    assert abs(result.nonlinear_residual) < 1.0e-8
    assert abs(result.east_liquid_rh_residual) < 1.0e-7
    assert abs(result.vertical_liquid_rh_residual) < 1.0e-7
    for traces in (result.east_traces, result.vertical_traces):
        ale = ALEInterfaceFlux.from_traces(traces)
        assert ale.gas_mass == 0.0
        assert abs(ale.liquid_area_residual) < 1.0e-10
        assert abs(ale.liquid_momentum_residual) < 1.0e-9


def test_explicit_geometric_capacity_is_never_silently_exceeded() -> None:
    fixture = _Fixture()
    with pytest.raises(PressureSolveError):
        solve_finite_node_twofront_launch(
            FiniteNodeGasState(0.0, 0.0, 1.0e-12),
            dt=1.0e-3,
            west_gas_trace=_trace(fixture, 0.20),
            **_inputs(fixture),
        )
