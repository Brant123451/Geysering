"""Resolved boundary-flux tests for the compressible finite T node."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    liquid_storage_factor,
    solve_compressible_node_pressure,
    state_from_pressure_and_gas_mass,
)
import casea_compressible_node_postlaunch_stage as postlaunch_module  # noqa: E402
from casea_compressible_node_postlaunch_stage import (  # noqa: E402
    PRODUCTION_READY,
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchParameters,
    ExactLaunchRequiresEventClosure,
    PostLaunchBoundActivationError,
    PostLaunchCFLInadmissible,
    euler_compressible_node_postlaunch_stage,
    liquid_characteristic_in_outward_coordinate,
    stratified_trace_in_outward_coordinate,
)
from casea_material_front_cutcell import StratifiedState  # noqa: E402
from casea_tjunction_shock_network import LiquidCharacteristic  # noqa: E402


P_ATM = 101_325.0
RHO_L = 998.0
A_NODE = 28.0
R_G = 287.05
T_G = 293.0
C_G = math.sqrt(R_G * T_G)
V_NODE = 0.10
FULL_AREA = 0.010
LIQUID_AREA = 0.005
GAS_AREA = FULL_AREA - LIQUID_AREA


def _params() -> CompressiblePostLaunchParameters:
    node = CompressibleFiniteNodeParameters(
        gas_sound_speed=C_G,
        liquid_density=RHO_L,
        liquid_wave_speed=A_NODE,
        reference_pressure_abs=P_ATM,
    )
    return CompressiblePostLaunchParameters(
        node=node,
        gas_constant=R_G,
        gas_temperature=T_G,
        atmospheric_pressure_abs=P_ATM,
    )


def _node_state(
    pressure: float = P_ATM,
    gas_volume_fraction: float = 0.40,
) -> CompressibleFiniteNodeState:
    params = _params().node
    gas_mass = (
        pressure * gas_volume_fraction * V_NODE / params.gas_sound_speed**2
    )
    return state_from_pressure_and_gas_mass(
        pressure_abs=pressure,
        gas_mass=gas_mass,
        node_total_volume=V_NODE,
        params=params,
    )


def _branch(
    *,
    node_pressure: float = P_ATM,
    gas_density_ratio: float = 1.0,
    gas_outward_velocity: float = 0.0,
    liquid_reference_pressure: float | None = None,
    liquid_reference_velocity: float = 0.0,
) -> CompressibleNodeResolvedBranch:
    density_node = node_pressure / C_G**2
    density = gas_density_ratio * density_node
    gas_mass = density * GAS_AREA
    characteristic_pressure = (
        node_pressure
        if liquid_reference_pressure is None
        else liquid_reference_pressure
    )
    return CompressibleNodeResolvedBranch(
        resolved=StratifiedState(
            gas_mass=gas_mass,
            gas_momentum=gas_mass * gas_outward_velocity,
            liquid_area=LIQUID_AREA,
            liquid_discharge=LIQUID_AREA * liquid_reference_velocity,
        ),
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=characteristic_pressure,
            reference_outward_velocity=liquid_reference_velocity,
            wave_speed=A_NODE,
        ),
        liquid_face_area=LIQUID_AREA,
        full_area=FULL_AREA,
        reference_liquid_face_pressure_abs=characteristic_pressure,
        reference_liquid_pressure_potential=0.0,
    )


def test_uniform_stationary_state_returns_complete_zero_gauge_fluxes() -> None:
    params = _params()
    state = _node_state()
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    branch = _branch(node_pressure=pressure)
    result = euler_compressible_node_postlaunch_stage(
        state,
        1.0e-4,
        west=branch,
        east=branch,
        vertical=branch,
        params=params,
    )
    assert PRODUCTION_READY
    for flux in (result.west, result.east, result.vertical):
        assert flux.gas_mass == pytest.approx(0.0, abs=2.0e-17)
        assert flux.gas_momentum == pytest.approx(0.0, abs=2.0e-12)
        assert flux.liquid_area == 0.0
        assert flux.liquid_momentum == 0.0
    for trace in (
        result.west_trace,
        result.east_trace,
        result.vertical_trace,
    ):
        assert trace.bound_audit.accepted_without_bound
        assert trace.gas_numerics.solver == "positive-density Roe"
        assert trace.gas_numerics.roe_used
        assert trace.gas_numerics.fallback_used is False
        assert trace.gas_numerics.fallback_name is None
        assert trace.gas_numerics.roe_density_floor_active is False
    assert result.node.state == state
    assert result.node.ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.node.ledger.liquid_inventory_balance_residual == 0.0


def test_one_branch_pressure_difference_changes_node_mass_only_through_that_flux() -> None:
    params = _params()
    state = _node_state()
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    west = _branch(node_pressure=pressure, gas_density_ratio=0.96)
    equal = _branch(node_pressure=pressure)
    dt = 1.0e-4
    result = euler_compressible_node_postlaunch_stage(
        state,
        dt,
        west=west,
        east=equal,
        vertical=equal,
        params=params,
    )
    assert result.west.gas_mass > 0.0
    assert result.east.gas_mass == pytest.approx(0.0, abs=2.0e-17)
    assert result.vertical.gas_mass == pytest.approx(0.0, abs=2.0e-17)
    assert result.node.state.gas_mass == pytest.approx(
        state.gas_mass - dt * result.west.gas_mass
    )


def test_three_complete_fluxes_drive_exact_node_ledgers_and_occupancy() -> None:
    params = _params()
    state = _node_state(pressure=103_000.0, gas_volume_fraction=0.35)
    p = solve_compressible_node_pressure(state, params.node).pressure_abs
    west = _branch(
        node_pressure=p,
        gas_density_ratio=0.98,
        gas_outward_velocity=0.20,
        liquid_reference_pressure=p - 80.0,
        liquid_reference_velocity=0.01,
    )
    east = _branch(
        node_pressure=p,
        gas_density_ratio=1.01,
        gas_outward_velocity=-0.10,
        liquid_reference_pressure=p + 50.0,
        liquid_reference_velocity=-0.005,
    )
    vertical = _branch(
        node_pressure=p,
        gas_density_ratio=1.00,
        gas_outward_velocity=0.05,
        liquid_reference_pressure=p,
        liquid_reference_velocity=0.002,
    )
    dt = 5.0e-5
    result = euler_compressible_node_postlaunch_stage(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    gas_sum = math.fsum(
        flux.gas_mass
        for flux in (result.west, result.east, result.vertical)
    )
    liquid_sum = math.fsum(
        flux.liquid_area
        for flux in (result.west, result.east, result.vertical)
    )
    ledger = result.node.ledger
    assert ledger.gas_mass_outward_rate == pytest.approx(gas_sum)
    assert ledger.liquid_equivalent_volume_outward_rate == pytest.approx(
        liquid_sum
    )
    assert ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert ledger.liquid_inventory_balance_residual == pytest.approx(
        0.0, abs=2.0e-17
    )
    pressure = result.node.pressure
    assert (
        pressure.gas_physical_volume + pressure.liquid_physical_volume
    ) == pytest.approx(V_NODE, abs=2.0e-13)


def test_west_global_axis_reversal_gives_same_outward_flux_as_east() -> None:
    params = _params()
    state = _node_state()
    density = P_ATM / C_G**2
    gas_mass = density * GAS_AREA
    west_global = StratifiedState(
        gas_mass=gas_mass,
        gas_momentum=-0.25 * gas_mass,
        liquid_area=LIQUID_AREA,
        liquid_discharge=-0.01 * LIQUID_AREA,
    )
    east_global = StratifiedState(
        gas_mass=gas_mass,
        gas_momentum=0.25 * gas_mass,
        liquid_area=LIQUID_AREA,
        liquid_discharge=0.01 * LIQUID_AREA,
    )
    axis_west_characteristic = LiquidCharacteristic(
        reference_pressure_abs=P_ATM,
        reference_outward_velocity=-0.01,
        wave_speed=A_NODE,
    )
    axis_east_characteristic = LiquidCharacteristic(
        reference_pressure_abs=P_ATM,
        reference_outward_velocity=0.01,
        wave_speed=A_NODE,
    )

    def converted(
        trace: StratifiedState,
        characteristic: LiquidCharacteristic,
        sign: int,
    ) -> CompressibleNodeResolvedBranch:
        return CompressibleNodeResolvedBranch(
            resolved=stratified_trace_in_outward_coordinate(trace, sign),
            liquid_characteristic=liquid_characteristic_in_outward_coordinate(
                characteristic, sign
            ),
            liquid_face_area=LIQUID_AREA,
            full_area=FULL_AREA,
            reference_liquid_face_pressure_abs=P_ATM,
            reference_liquid_pressure_potential=0.0,
        )

    west = converted(west_global, axis_west_characteristic, -1)
    east = converted(east_global, axis_east_characteristic, 1)
    result = euler_compressible_node_postlaunch_stage(
        state,
        1.0e-5,
        west=west,
        east=east,
        vertical=_branch(),
        params=params,
    )
    assert result.west.vector() == pytest.approx(result.east.vector())


def test_postlaunch_stage_rejects_exact_zero_gas_event_state() -> None:
    params = _params()
    pressure = 104_000.0
    state = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=(
            V_NODE * liquid_storage_factor(pressure, params.node)
        ),
        node_total_volume=V_NODE,
    )
    branch = _branch(node_pressure=pressure)
    with pytest.raises(ExactLaunchRequiresEventClosure):
        euler_compressible_node_postlaunch_stage(
            state,
            1.0e-5,
            west=branch,
            east=branch,
            vertical=branch,
            params=params,
        )


def test_explicit_node_cfl_is_rejected_instead_of_limited() -> None:
    params = _params()
    state = _node_state()
    branch = _branch()
    with pytest.raises(PostLaunchCFLInadmissible) as caught:
        euler_compressible_node_postlaunch_stage(
            state,
            1.0,
            west=branch,
            east=branch,
            vertical=branch,
            params=params,
        )
    assert caught.value.maximum_dt < 1.0


@pytest.mark.parametrize(
    ("bad_branch", "expected_bound"),
    [
        (_branch(gas_density_ratio=0.10), "trace_density_floor"),
        (_branch(gas_density_ratio=13.0), "trace_density_ceiling"),
        (
            replace(_branch(), liquid_face_area=0.00996),
            "face_geometry_cap",
        ),
        (
            replace(
                _branch(),
                resolved=StratifiedState(
                    gas_mass=0.5e-10 * GAS_AREA,
                    gas_momentum=0.0,
                    liquid_area=LIQUID_AREA,
                    liquid_discharge=0.0,
                ),
            ),
            "trace_roe_internal_floor",
        ),
    ],
)
def test_all_bounds_fail_closed_before_public_roe_call(
    monkeypatch: pytest.MonkeyPatch,
    bad_branch: CompressibleNodeResolvedBranch,
    expected_bound: str,
) -> None:
    params = _params()
    state = _node_state()
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    good = _branch(node_pressure=pressure)
    calls: list[object] = []

    def forbidden_roe(*args: object, **kwargs: object) -> tuple[float, float]:
        calls.append((args, kwargs))
        raise AssertionError("public Roe must not run after a failed raw audit")

    monkeypatch.setattr(
        postlaunch_module,
        "isothermal_ideal_gas_riemann_flux",
        forbidden_roe,
    )
    with pytest.raises(PostLaunchBoundActivationError) as caught:
        euler_compressible_node_postlaunch_stage(
            state,
            1.0e-5,
            west=good,
            east=bad_branch,
            vertical=good,
            params=params,
        )
    assert expected_bound in caught.value.audit.active_bounds
    assert calls == []


def test_repeated_euler_postlaunch_stage_is_first_order_in_dt() -> None:
    params = _params()
    initial = _node_state(pressure=102_000.0, gas_volume_fraction=0.30)
    branch = _branch(
        node_pressure=102_000.0,
        gas_density_ratio=0.97,
        gas_outward_velocity=0.05,
        liquid_reference_pressure=101_700.0,
        liquid_reference_velocity=0.003,
    )

    def integrate(n_steps: int) -> CompressibleFiniteNodeState:
        state = initial
        dt = 1.0e-3 / n_steps
        for _ in range(n_steps):
            state = euler_compressible_node_postlaunch_stage(
                state,
                dt,
                west=branch,
                east=branch,
                vertical=branch,
                params=params,
            ).node.state
        return state

    reference = integrate(4096)
    coarse = integrate(32)
    fine = integrate(64)

    def error(state: CompressibleFiniteNodeState) -> float:
        return abs(state.gas_mass - reference.gas_mass) + abs(
            state.liquid_equivalent_volume
            - reference.liquid_equivalent_volume
        )

    coarse_error = error(coarse)
    fine_error = error(fine)
    assert fine_error < coarse_error
    assert coarse_error / fine_error > 1.8
