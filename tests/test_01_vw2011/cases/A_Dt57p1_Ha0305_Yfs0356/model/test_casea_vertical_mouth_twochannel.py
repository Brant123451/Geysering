"""Tests for the target-free conservative two-channel riser-mouth closure."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    LiquidDonorInventories,
    NetFluxExceedsDownwardCapacity,
    NetFluxExceedsDonorCapacity,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
    close_vertical_mouth_twochannel_exchange,
)


GEOMETRY = VerticalMouthGeometry(diameter=0.0571, gravity=9.81)
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
    liquid_velocity: float = -0.05,
    gas_velocity: float = 0.50,
) -> VerticalMouthPhaseState:
    liquid_area = liquid_fraction * GEOMETRY.full_area
    return VerticalMouthPhaseState(
        liquid_area=liquid_area,
        liquid_velocity=liquid_velocity,
        gas_area=GEOMETRY.full_area - liquid_area,
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


def _close(
    q_net: float,
    *,
    phase: VerticalMouthPhaseState | None = None,
    donors: LiquidDonorInventories | None = None,
    losses: DirectionalMouthLosses = LOSSES,
):
    return close_vertical_mouth_twochannel_exchange(
        q_net,
        phase=_phase() if phase is None else phase,
        geometry=GEOMETRY,
        material=MATERIAL,
        wallis=WALLIS,
        donors=_donors() if donors is None else donors,
        losses=losses,
    )


@pytest.mark.parametrize("q_net", [2.0e-5, 0.0, -3.0e-5])
def test_gross_exchange_preserves_the_finite_node_net_flux(q_net: float) -> None:
    result = _close(q_net)
    assert result.circulation_flow > 0.0
    assert result.upward_flow >= 0.0
    assert result.downward_flow >= 0.0
    assert result.upward_flow - result.downward_flow == pytest.approx(
        q_net, rel=0.0, abs=3.0e-20
    )
    assert result.closure_residual == pytest.approx(0.0, abs=3.0e-20)

    # Applying the two gross streams to node and riser cancels exactly in the
    # combined liquid ledger, while each donor remains non-negative.
    dt = _donors().time_step
    node_change = dt * (result.downward_flow - result.upward_flow)
    riser_change = -node_change
    assert node_change + riser_change == 0.0
    assert result.upward_flow * dt <= _donors().finite_node_volume
    assert result.downward_flow * dt <= _donors().riser_volume


@pytest.mark.parametrize(
    "phase",
    [
        _phase(liquid_fraction=0.0, gas_velocity=0.50),
        _phase(liquid_fraction=1.0, gas_velocity=0.50),
        _phase(liquid_fraction=0.25, gas_velocity=0.0),
        _phase(liquid_fraction=0.25, gas_velocity=-0.25),
    ],
)
def test_zero_film_or_no_upward_gas_gives_zero_circulation(
    phase: VerticalMouthPhaseState,
) -> None:
    result = _close(0.0, phase=phase)
    assert result.circulation_flow == 0.0
    assert result.upward_flow == 0.0
    assert result.downward_flow == 0.0
    assert result.countercurrent_mixing_loss_power == 0.0


def test_wallis_capacity_decreases_monotonically_with_upward_gas_speed() -> None:
    speeds = (0.10, 0.50, 1.00, 2.00)
    results = [
        _close(0.0, phase=_phase(gas_velocity=speed))
        for speed in speeds
    ]
    capacities = [item.wallis_downward_capacity for item in results]
    circulations = [item.circulation_flow for item in results]
    assert all(
        left > right for left, right in zip(capacities, capacities[1:])
    )
    assert all(
        left >= right for left, right in zip(circulations, circulations[1:])
    )
    assert all(item.gravity_film_capacity > 0.0 for item in results)


def test_gravity_and_wallis_limit_total_downward_gross_flow() -> None:
    zero_net = _close(0.0)
    capacity = min(
        zero_net.gravity_film_capacity,
        zero_net.wallis_downward_capacity,
    )
    negative_net = 0.40 * capacity
    result = _close(-negative_net)

    assert result.downward_physical_capacity == pytest.approx(capacity)
    assert result.downward_physical_circulation_capacity == pytest.approx(
        capacity - negative_net
    )
    assert result.circulation_flow == pytest.approx(capacity - negative_net)
    assert result.downward_flow == pytest.approx(capacity)
    assert result.upward_flow == pytest.approx(capacity - negative_net)
    assert result.upward_flow - result.downward_flow == pytest.approx(
        -negative_net
    )


def test_inadmissible_negative_net_requires_finite_node_complementarity_resolve(
) -> None:
    reference = _close(0.0)
    capacity = reference.downward_physical_capacity
    requested_q_net = -(capacity + 1.0e-6)

    with pytest.raises(NetFluxExceedsDownwardCapacity) as caught:
        _close(requested_q_net)

    error = caught.value
    assert error.q_net == pytest.approx(requested_q_net)
    assert error.downward_capacity == pytest.approx(capacity)
    assert "re-solve" in str(error)
    assert "complementarity" in str(error)


def test_no_upward_gas_disables_circulation_but_not_admissible_net_drainage(
) -> None:
    phase = _phase(gas_velocity=0.0)
    result = _close(-2.0e-5, phase=phase)
    assert result.wallis_downward_capacity == 0.0
    assert result.downward_physical_capacity == pytest.approx(
        result.gravity_film_capacity
    )
    assert result.downward_physical_circulation_capacity == 0.0
    assert result.circulation_flow == 0.0
    assert result.upward_flow == 0.0
    assert result.downward_flow == pytest.approx(2.0e-5)


def test_donor_capacity_limits_only_equal_and_opposite_circulation() -> None:
    dt = 0.10
    limited = _close(
        2.0e-5,
        donors=_donors(
            node_volume=3.0e-6,
            riser_volume=9.0e-6,
            dt=dt,
        ),
    )
    # Node capacity is 3e-5 m3/s.  The prescribed net upward component uses
    # 2e-5, leaving exactly 1e-5 for the zero-net circulation.
    assert limited.finite_node_circulation_capacity == pytest.approx(1.0e-5)
    assert limited.circulation_flow == pytest.approx(1.0e-5)
    assert limited.upward_flow == pytest.approx(3.0e-5)
    assert limited.downward_flow == pytest.approx(1.0e-5)
    assert limited.upward_flow - limited.downward_flow == pytest.approx(2.0e-5)

    less_node_liquid = _close(
        0.0,
        donors=_donors(
            node_volume=2.0e-6,
            riser_volume=9.0e-6,
            dt=dt,
        ),
    )
    more_node_liquid = _close(
        0.0,
        donors=_donors(
            node_volume=4.0e-6,
            riser_volume=9.0e-6,
            dt=dt,
        ),
    )
    assert less_node_liquid.circulation_flow < more_node_liquid.circulation_flow


def test_net_flux_that_exhausts_a_donor_is_rejected_not_clipped() -> None:
    donors = _donors(node_volume=1.0e-7, riser_volume=1.0e-3, dt=0.10)
    with pytest.raises(NetFluxExceedsDonorCapacity):
        _close(2.0e-6, donors=donors)


def test_directional_losses_are_kept_separate() -> None:
    losses = DirectionalMouthLosses(
        upward_turn=0.50,
        downward_turn=2.00,
        countercurrent_mixing=7.00,
    )
    all_liquid = _phase(
        liquid_fraction=1.0,
        liquid_velocity=0.0,
        gas_velocity=0.0,
    )
    upward = _close(2.0e-5, phase=all_liquid, losses=losses)
    downward = _close(-2.0e-5, phase=all_liquid, losses=losses)

    assert upward.circulation_flow == 0.0
    assert upward.upward_turn_loss_power > 0.0
    assert upward.downward_turn_loss_power == 0.0
    assert upward.countercurrent_mixing_loss_power == 0.0
    upward_reference = (
        0.5
        * MATERIAL.liquid_density
        * upward.upward_flow
        * upward.upward_channel_velocity**2
    )
    assert upward.upward_turn_loss_power / upward_reference == pytest.approx(
        losses.upward_turn
    )

    assert downward.circulation_flow == 0.0
    assert downward.upward_turn_loss_power == 0.0
    assert downward.downward_turn_loss_power > 0.0
    assert downward.countercurrent_mixing_loss_power == 0.0
    downward_reference = (
        0.5
        * MATERIAL.liquid_density
        * downward.downward_flow
        * downward.downward_channel_velocity**2
    )
    assert (
        downward.downward_turn_loss_power / downward_reference
    ) == pytest.approx(losses.downward_turn)


def test_countercurrent_momentum_and_energy_diagnostics_are_dissipative() -> None:
    result = _close(0.0)
    assert result.circulation_flow > 0.0
    assert result.upward_channel_area + result.downward_channel_area == pytest.approx(
        _phase().liquid_area
    )
    assert result.gross_convective_momentum_flux > 0.0
    assert result.bulk_convective_momentum_flux == 0.0
    assert result.countercurrent_momentum_excess > 0.0
    assert result.gross_kinetic_power > 0.0
    assert result.signed_kinetic_energy_flux == pytest.approx(0.0, abs=1.0e-18)
    assert result.upward_turn_loss_power > 0.0
    assert result.downward_turn_loss_power > 0.0
    assert result.countercurrent_mixing_loss_power > 0.0
    assert result.total_dissipation_power == pytest.approx(
        result.upward_turn_loss_power
        + result.downward_turn_loss_power
        + result.countercurrent_mixing_loss_power
    )
    assert result.total_dissipation_power >= 0.0


def test_closure_source_contains_no_case_time_or_target_height() -> None:
    source = inspect.getsource(close_vertical_mouth_twochannel_exchange)
    lowered = source.lower()
    assert "8.85" not in source
    assert "0.104" not in source
    assert "target_height" not in lowered
    assert "event_time" not in lowered
    assert "current_time" not in lowered
    assert "frame" not in lowered
    assert "animation" not in lowered
    assert "render" not in lowered
    # The closure accesses the time step only through the donor-capacity
    # properties.  Keep the source audit tied to that physical/numerical role
    # rather than to the dataclass implementation detail.
    assert "finite_node_rate_capacity" in source
    assert "riser_rate_capacity" in source


def test_phase_partition_and_material_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        VerticalMouthPhaseState(
            liquid_area=0.25 * GEOMETRY.full_area,
            liquid_velocity=0.0,
            gas_area=0.50 * GEOMETRY.full_area,
            gas_velocity=0.2,
        ).validate(GEOMETRY)
    with pytest.raises(ValueError):
        VerticalMouthMaterialProperties(
            liquid_density=1.0,
            gas_density=2.0,
            liquid_dynamic_viscosity=1.0e-3,
        )
    assert math.isclose(
        GEOMETRY.full_area,
        math.pi * GEOMETRY.diameter**2 / 4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )
