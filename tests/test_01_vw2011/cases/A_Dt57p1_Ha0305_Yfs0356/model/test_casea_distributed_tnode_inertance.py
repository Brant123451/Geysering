"""Tests for the target-free distributed Case-A T-node flux owner."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_distributed_tnode_inertance import (  # noqa: E402
    DistributedTNodeGeometry,
    DistributedTNodeMomentumState,
    DistributedTNodePressureState,
    advance_distributed_tnode_inertance,
    measured_footprint_liquid_inventory,
)
from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    LiquidDonorInventories,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    DuplicateMouthFluxOwner,
    HorizontalNodeTopology,
    LegacyMouthPathActivity,
)


GEOMETRY = DistributedTNodeGeometry.case_a()
MATERIAL = VerticalMouthMaterialProperties(
    liquid_density=998.0,
    gas_density=1.20,
    liquid_dynamic_viscosity=1.003e-3,
)
WALLIS = WallisCounterCurrentParameters(constant=0.50, slope=1.0)
LOSSES = DirectionalMouthLosses(
    upward_turn=0.75,
    downward_turn=1.25,
    countercurrent_mixing=4.0,
)


def _phase(
    *,
    liquid_fraction: float = 0.25,
    liquid_velocity: float = 0.0,
    gas_velocity: float = 0.50,
) -> VerticalMouthPhaseState:
    liquid_area = liquid_fraction * GEOMETRY.mouth_area
    return VerticalMouthPhaseState(
        liquid_area=liquid_area,
        liquid_velocity=liquid_velocity,
        gas_area=GEOMETRY.mouth_area - liquid_area,
        gas_velocity=gas_velocity,
    )


def _donors(
    *,
    node_volume: float = 1.0e-3,
    riser_volume: float = 1.0e-3,
    dt: float = 1.0e-2,
) -> LiquidDonorInventories:
    return LiquidDonorInventories(
        finite_node_volume=node_volume,
        riser_volume=riser_volume,
        time_step=dt,
    )


def _pressure(
    *,
    gas: float = 101_325.0,
    liquid: float = 101_325.0,
    liquid_fraction: float = 0.50,
    vertical: float = 101_325.0,
) -> DistributedTNodePressureState:
    return DistributedTNodePressureState(
        horizontal_gas_pressure_abs=gas,
        horizontal_liquid_pressure_abs=liquid,
        horizontal_liquid_area=(
            liquid_fraction * GEOMETRY.horizontal_full_area
        ),
        vertical_mouth_pressure_abs=vertical,
    )


def _advance(
    state: DistributedTNodeMomentumState | None = None,
    *,
    dt: float = 1.0e-2,
    pressure: DistributedTNodePressureState | None = None,
    phase: VerticalMouthPhaseState | None = None,
    donors: LiquidDonorInventories | None = None,
    losses: DirectionalMouthLosses = LOSSES,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
):
    return advance_distributed_tnode_inertance(
        DistributedTNodeMomentumState(0.0) if state is None else state,
        dt=dt,
        pressure=_pressure() if pressure is None else pressure,
        geometry=GEOMETRY,
        phase=_phase() if phase is None else phase,
        material=MATERIAL,
        wallis=WALLIS,
        donors=_donors(dt=dt) if donors is None else donors,
        losses=losses,
        horizontal_axial_velocity=0.20,
        legacy_activity=legacy_activity,
    )


def test_case_a_geometry_uses_measured_pipe_and_tower_dimensions() -> None:
    assert GEOMETRY.horizontal_diameter == pytest.approx(0.094)
    assert GEOMETRY.riser_diameter == pytest.approx(0.0571)
    assert GEOMETRY.opening_footprint_length == pytest.approx(0.0571)
    assert GEOMETRY.opening_footprint_volume == pytest.approx(
        GEOMETRY.horizontal_full_area * GEOMETRY.opening_footprint_length
    )
    assert GEOMETRY.effective_inertance_length == pytest.approx(
        GEOMETRY.opening_footprint_volume / GEOMETRY.mouth_area
    )


def test_measured_footprint_inventory_is_grid_spacing_independent() -> None:
    full = GEOMETRY.horizontal_full_area
    two_cells = measured_footprint_liquid_inventory(
        [full, full],
        [0.50, 0.50],
        geometry=GEOMETRY,
    )
    six_cells = measured_footprint_liquid_inventory(
        [full] * 6,
        [1.0 / 6.0] * 6,
        geometry=GEOMETRY,
    )
    assert two_cells == pytest.approx(GEOMETRY.opening_footprint_volume)
    assert six_cells == pytest.approx(GEOMETRY.opening_footprint_volume)
    assert two_cells == pytest.approx(six_cells)


def test_horizontal_pressure_is_current_phase_contact_area_average() -> None:
    pressure = _pressure(
        gas=100_000.0,
        liquid=102_000.0,
        liquid_fraction=0.25,
        vertical=100_500.0,
    )
    assert pressure.horizontal_contact_pressure(GEOMETRY) == pytest.approx(
        100_500.0
    )
    result = _advance(pressure=pressure)
    assert result.q_net == pytest.approx(0.0, abs=1.0e-18)
    assert result.complementarity.driving_pressure_difference == pytest.approx(0.0)


def test_inertive_flux_is_persistent_without_drive_or_loss() -> None:
    q_initial = 2.0e-5
    state = DistributedTNodeMomentumState.from_net_flux(
        q_initial,
        geometry=GEOMETRY,
        liquid_density=MATERIAL.liquid_density,
    )
    no_loss = DirectionalMouthLosses(
        upward_turn=0.0,
        downward_turn=0.0,
        countercurrent_mixing=0.0,
    )
    result = _advance(state, losses=no_loss)
    assert result.q_net == pytest.approx(q_initial, rel=0.0, abs=1.0e-18)
    assert result.state.liquid_momentum == pytest.approx(state.liquid_momentum)
    assert result.complementarity.pressure_balance_residual == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_pressure_drive_advances_the_momentum_equation_not_an_assigned_flux() -> None:
    dt = 0.02
    delta_pressure = 250.0
    no_loss = DirectionalMouthLosses(
        upward_turn=0.0,
        downward_turn=0.0,
        countercurrent_mixing=0.0,
    )
    pressure_inertance = (
        MATERIAL.liquid_density
        * GEOMETRY.effective_inertance_length
        / GEOMETRY.mouth_area
    )
    expected_q = delta_pressure * dt / pressure_inertance
    result = _advance(
        dt=dt,
        pressure=_pressure(
            gas=101_575.0,
            liquid=101_575.0,
            vertical=101_325.0,
        ),
        donors=_donors(dt=dt),
        losses=no_loss,
    )
    assert result.q_net == pytest.approx(expected_q, rel=1.0e-13)
    assert result.complementarity.unconstrained_q_net == pytest.approx(expected_q)
    assert result.complementarity.actual_liquid_momentum_change == pytest.approx(
        GEOMETRY.mouth_area * dt * delta_pressure
    )
    assert result.complementarity.liquid_momentum_balance_residual == pytest.approx(
        0.0, abs=2.0e-13
    )


def test_nusselt_wallis_capacity_is_solved_as_pressure_complementarity() -> None:
    result = _advance(
        pressure=_pressure(
            gas=40_000.0,
            liquid=40_000.0,
            vertical=160_000.0,
        ),
        donors=_donors(node_volume=1.0e-2, riser_volume=1.0e-2),
    )
    ledger = result.complementarity
    assert ledger.unconstrained_q_net < ledger.lower_flux_bound
    assert result.q_net == pytest.approx(
        -ledger.physical_downward_capacity,
        rel=0.0,
        abs=2.0e-15,
    )
    assert ledger.downward_capacity_active
    assert ledger.lower_bound_owner in {"nusselt", "wallis", "nusselt_wallis_tie"}
    assert ledger.physical_downward_reaction_pressure > 0.0
    assert ledger.physical_downward_reaction_force == pytest.approx(
        GEOMETRY.mouth_area * ledger.physical_downward_reaction_pressure
    )
    assert ledger.physical_downward_reaction_impulse == pytest.approx(
        0.01 * ledger.physical_downward_reaction_force
    )
    assert ledger.signed_constraint_reaction_impulse > 0.0
    assert ledger.pressure_balance_residual == pytest.approx(0.0, abs=2.0e-8)
    assert ledger.physical_downward_gap == pytest.approx(0.0, abs=2.0e-15)
    assert ledger.physical_complementarity_product == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert result.mouth_plan.exchange.downward_flow == pytest.approx(
        ledger.physical_downward_capacity
    )
    assert result.mouth_plan.exchange.circulation_flow == pytest.approx(
        0.0, abs=2.0e-15
    )


def test_horizontal_donor_limit_has_its_own_reaction_and_no_silent_clip() -> None:
    dt = 0.01
    node_volume = 1.0e-7
    result = _advance(
        dt=dt,
        pressure=_pressure(
            gas=180_000.0,
            liquid=180_000.0,
            vertical=40_000.0,
        ),
        donors=_donors(
            node_volume=node_volume,
            riser_volume=1.0e-2,
            dt=dt,
        ),
    )
    ledger = result.complementarity
    expected_bound = node_volume / dt
    assert ledger.unconstrained_q_net > expected_bound
    assert result.q_net == pytest.approx(expected_bound)
    assert ledger.donor_upward_reaction_pressure > 0.0
    assert ledger.upper_complementarity_product == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert result.inventory_update.finite_node_liquid_volume == pytest.approx(
        0.0, abs=2.0e-18
    )


@pytest.mark.parametrize(
    "pressure",
    [
        _pressure(gas=102_000.0, liquid=102_000.0, vertical=101_325.0),
        _pressure(gas=100_700.0, liquid=100_700.0, vertical=101_325.0),
        _pressure(),
    ],
)
def test_gross_closure_and_combined_volume_are_exact(
    pressure: DistributedTNodePressureState,
) -> None:
    result = _advance(pressure=pressure)
    exchange = result.mouth_plan.exchange
    assert exchange.upward_flow - exchange.downward_flow == pytest.approx(
        result.q_net, rel=0.0, abs=3.0e-20
    )
    assert result.mouth_plan.horizontal_node_topology is (
        HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT
    )
    assert result.mouth_plan.combined_liquid_volume_rate == pytest.approx(
        0.0, abs=3.0e-20
    )
    assert result.combined_liquid_volume_residual == pytest.approx(
        0.0, abs=3.0e-18
    )
    assert result.inventory_update.combined_volume_after == pytest.approx(
        result.inventory_update.combined_volume_before,
        rel=0.0,
        abs=3.0e-18,
    )


def test_local_k_loss_is_in_the_implicit_pressure_and_momentum_ledgers() -> None:
    initial = DistributedTNodeMomentumState.from_net_flux(
        4.0e-5,
        geometry=GEOMETRY,
        liquid_density=MATERIAL.liquid_density,
    )
    result = _advance(initial)
    ledger = result.complementarity
    assert 0.0 < result.q_net < 4.0e-5
    assert ledger.signed_local_loss_pressure > 0.0
    assert ledger.inertive_pressure_change < 0.0
    assert ledger.pressure_balance_residual == pytest.approx(0.0, abs=2.0e-10)
    assert ledger.liquid_momentum_balance_residual == pytest.approx(
        0.0, abs=2.0e-13
    )


def test_closed_liquid_mouth_uses_a_recorded_constraint_reaction() -> None:
    phase = _phase(liquid_fraction=0.0, gas_velocity=1.0)
    result = _advance(
        phase=phase,
        pressure=_pressure(
            gas=120_000.0,
            liquid=120_000.0,
            vertical=101_325.0,
        ),
    )
    assert result.q_net == 0.0
    assert result.complementarity.closed_mouth_reaction_pressure > 0.0
    assert result.mouth_plan.exchange.upward_flow == 0.0
    assert result.mouth_plan.exchange.downward_flow == 0.0


def test_duplicate_legacy_flux_owner_is_rejected() -> None:
    with pytest.raises(DuplicateMouthFluxOwner):
        _advance(
            legacy_activity=LegacyMouthPathActivity(
                taylor_return_mass_flux_applied=True
            )
        )


def test_solver_source_contains_no_time_or_result_target() -> None:
    source = inspect.getsource(advance_distributed_tnode_inertance).lower()
    forbidden = (
        "8.85",
        "9.2",
        "0.104",
        "target_height",
        "target_volume",
        "openfoam",
        "frame",
        "animation",
        "current_time",
        "event_time",
    )
    for token in forbidden:
        assert token not in source
    assert "time_step" in source  # only the numerical donor interval


def test_geometry_and_pressure_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        DistributedTNodeGeometry(
            horizontal_diameter=0.094,
            riser_diameter=0.0571,
            opening_footprint_length=0.0,
            opening_footprint_volume=1.0e-4,
        )
    with pytest.raises(ValueError):
        _pressure(liquid_fraction=1.2).validate(GEOMETRY)
    assert math.isfinite(GEOMETRY.effective_inertance_length)
