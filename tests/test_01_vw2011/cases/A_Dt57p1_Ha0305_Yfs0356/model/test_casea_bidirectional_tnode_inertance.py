"""Component tests for the independent bidirectional Case-A T-node closure."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_bidirectional_tnode_inertance import (  # noqa: E402
    BidirectionalTNodeError,
    BidirectionalTNodeParameters,
    BidirectionalTNodeState,
    PersistentMouthTrace,
    advance_bidirectional_tnode_inertance,
    dynamic_annular_film_geometry,
    stage_bidirectional_tnode_coupling,
)
from casea_distributed_tnode_inertance import (  # noqa: E402
    DistributedTNodeGeometry,
    DistributedTNodePressureState,
)
from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    LiquidDonorInventories,
    VerticalMouthMaterialProperties,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    HorizontalNodeTopology,
    apply_twochannel_horizontal_footprint,
)


GEOMETRY = DistributedTNodeGeometry.case_a()
MATERIAL = VerticalMouthMaterialProperties(
    liquid_density=998.0,
    gas_density=1.20,
    liquid_dynamic_viscosity=1.003e-3,
)
WALLIS = WallisCounterCurrentParameters(constant=0.50, slope=1.0)
NO_LOCAL_LOSSES = DirectionalMouthLosses(
    upward_turn=0.0,
    downward_turn=0.0,
    countercurrent_mixing=0.0,
)
LOSSES = DirectionalMouthLosses(
    upward_turn=0.75,
    downward_turn=1.25,
    countercurrent_mixing=4.0,
)
PARAMETERS = BidirectionalTNodeParameters(riser_cell_length=0.020)


def _film_area(thickness: float) -> float:
    radius = 0.5 * GEOMETRY.riser_diameter
    return math.pi * (radius * radius - (radius - thickness) ** 2)


def _trace(
    *,
    thickness: float = 5.0e-4,
    upward_discharge: float = 0.0,
    downward_discharge: float = 0.0,
    gas_mass_flow: float = 0.0,
    no_gas_core: bool = False,
) -> PersistentMouthTrace:
    downward_area = _film_area(thickness)
    upward_area = (
        GEOMETRY.mouth_area - downward_area
        if no_gas_core
        else 0.10 * GEOMETRY.mouth_area
    )
    gas_area = 0.0 if no_gas_core else (
        GEOMETRY.mouth_area - upward_area - downward_area
    )
    gas_cell_mass = MATERIAL.gas_density * gas_area * PARAMETERS.riser_cell_length
    gas_cell_velocity = (
        gas_mass_flow / (MATERIAL.gas_density * gas_area)
        if gas_area > 0.0
        else 0.0
    )
    return PersistentMouthTrace(
        upward_area=upward_area,
        upward_discharge=upward_discharge,
        downward_area=downward_area,
        downward_discharge=downward_discharge,
        gas_area=gas_area,
        gas_mass_flow=gas_mass_flow,
        gas_cell_mass=gas_cell_mass,
        gas_cell_momentum=gas_cell_mass * gas_cell_velocity,
    )


def _pressure(delta: float = 0.0) -> DistributedTNodePressureState:
    horizontal = 101_325.0 + delta
    return DistributedTNodePressureState(
        horizontal_gas_pressure_abs=horizontal,
        horizontal_liquid_pressure_abs=horizontal,
        horizontal_liquid_area=GEOMETRY.horizontal_full_area,
        vertical_mouth_pressure_abs=101_325.0,
    )


def _donors(
    *,
    dt: float = 1.0e-2,
    node: float = 1.0e-3,
    riser: float = 1.0e-3,
) -> LiquidDonorInventories:
    return LiquidDonorInventories(
        finite_node_volume=node,
        riser_volume=riser,
        time_step=dt,
    )


def _advance(
    state: BidirectionalTNodeState = BidirectionalTNodeState(0.0, 0.0),
    *,
    dt: float = 1.0e-2,
    trace: PersistentMouthTrace | None = None,
    pressure: DistributedTNodePressureState | None = None,
    donors: LiquidDonorInventories | None = None,
    losses: DirectionalMouthLosses = LOSSES,
    riser_receiving_net_rate_capacity: float = math.inf,
):
    return advance_bidirectional_tnode_inertance(
        state,
        dt=dt,
        pressure=_pressure() if pressure is None else pressure,
        trace=_trace() if trace is None else trace,
        geometry=GEOMETRY,
        material=MATERIAL,
        wallis=WALLIS,
        donors=_donors(dt=dt) if donors is None else donors,
        losses=losses,
        parameters=PARAMETERS,
        riser_receiving_net_rate_capacity=(
            riser_receiving_net_rate_capacity
        ),
    )


class BidirectionalTNodeTests(unittest.TestCase):
    def assertClose(
        self,
        actual: float,
        expected: float,
        *,
        rel: float = 1.0e-10,
        absolute: float = 1.0e-14,
    ) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute),
            msg=f"{actual!r} != {expected!r}",
        )

    def test_gross_flow_identities_are_exact_for_both_net_directions(self) -> None:
        for q_net, q_c in ((2.0e-5, 3.0e-6), (-2.0e-5, 3.0e-6), (0.0, 3.0e-6)):
            state = BidirectionalTNodeState(q_net, q_c)
            self.assertClose(state.upward_flow, max(q_net, 0.0) + q_c)
            self.assertClose(state.downward_flow, max(-q_net, 0.0) + q_c)
            self.assertClose(state.upward_flow - state.downward_flow, q_net)
            reconstructed = BidirectionalTNodeState.from_gross_flows(
                upward_flow=state.upward_flow,
                downward_flow=state.downward_flow,
            )
            self.assertClose(reconstructed.q_net, q_net)
            self.assertClose(reconstructed.q_c, q_c)

    def test_net_flux_has_pressure_inertial_continuity(self) -> None:
        q_old = 2.0e-5
        result = _advance(
            BidirectionalTNodeState(q_old, 0.0),
            trace=_trace(no_gas_core=True),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertClose(result.q_net, q_old, absolute=2.0e-16)
        self.assertClose(result.net.pressure_residual, 0.0, absolute=1.0e-8)
        self.assertClose(result.net.momentum_residual, 0.0, absolute=1.0e-12)

    def test_circulation_persists_when_upward_gas_disappears(self) -> None:
        dt = 1.0e-4
        old = BidirectionalTNodeState(0.0, 2.0e-5)
        with_gas = _trace(
            upward_discharge=2.0e-5,
            downward_discharge=-2.0e-5,
            gas_mass_flow=2.0e-4,
        )
        first = _advance(old, dt=dt, trace=with_gas, losses=NO_LOCAL_LOSSES)
        without_gas = _trace(
            upward_discharge=first.upward_flow,
            downward_discharge=-first.downward_flow,
            gas_mass_flow=0.0,
        )
        second = _advance(
            first.state,
            dt=dt,
            trace=without_gas,
            losses=NO_LOCAL_LOSSES,
        )
        self.assertGreater(first.circulation_flow, 0.0)
        self.assertGreater(second.circulation_flow, 0.0)
        self.assertFalse(second.circulation.wallis_active)
        self.assertTrue(math.isfinite(second.circulation.circulation_pressure_inertance))

    def test_resolved_trace_initializes_real_circulation_inertia(self) -> None:
        initial = 2.0e-5
        trace = _trace(
            upward_discharge=initial,
            downward_discharge=-initial,
            no_gas_core=True,
        )
        state = BidirectionalTNodeState.from_persistent_trace(trace)
        result = _advance(
            state,
            dt=1.0e-5,
            trace=trace,
            losses=NO_LOCAL_LOSSES,
        )
        self.assertClose(state.q_net, 0.0)
        self.assertClose(state.circulation_flow, initial)
        self.assertGreater(result.circulation_flow, 0.0)
        self.assertLess(result.circulation_flow, initial)
        self.assertClose(result.circulation.gas_circulation_drive_pressure, 0.0)
        self.assertClose(result.circulation.net_circulation_gravity_pressure, 0.0)
        self.assertClose(result.circulation.state_trace_circulation_mismatch, 0.0)

    def test_horizontal_phase_pressure_difference_starts_counterflow(self) -> None:
        gas_pressure = 101_325.0
        liquid_pressure = 103_325.0
        pressure = DistributedTNodePressureState(
            horizontal_gas_pressure_abs=gas_pressure,
            horizontal_liquid_pressure_abs=liquid_pressure,
            horizontal_liquid_area=0.50 * GEOMETRY.horizontal_full_area,
            # Hold q_net at zero so this test isolates closed-loop work.
            vertical_mouth_pressure_abs=0.50 * (
                gas_pressure + liquid_pressure
            ),
        )

        result = _advance(
            BidirectionalTNodeState(0.0, 0.0),
            pressure=pressure,
            losses=NO_LOCAL_LOSSES,
        )

        self.assertClose(result.q_net, 0.0, absolute=1.0e-8)
        self.assertGreater(result.circulation_flow, 0.0)
        self.assertClose(
            result.upward_flow - result.downward_flow,
            result.q_net,
            absolute=1.0e-15,
        )
        self.assertClose(
            result.circulation.horizontal_phase_pressure_drive,
            liquid_pressure - gas_pressure,
        )
        self.assertClose(
            result.volume.combined_volume_residual,
            0.0,
            absolute=1.0e-18,
        )
        self.assertClose(
            result.circulation.circulation_momentum_residual,
            0.0,
            absolute=1.0e-12,
        )

    def test_phase_pressure_difference_does_not_reverse_falling_net_flux(self) -> None:
        pressure = DistributedTNodePressureState(
            horizontal_gas_pressure_abs=101_325.0,
            horizontal_liquid_pressure_abs=106_325.0,
            horizontal_liquid_area=0.50 * GEOMETRY.horizontal_full_area,
            vertical_mouth_pressure_abs=101_325.0,
        )
        old = BidirectionalTNodeState(q_net=-2.0e-5, circulation_flow=0.0)

        result = _advance(
            old,
            dt=1.0e-4,
            pressure=pressure,
            losses=NO_LOCAL_LOSSES,
        )

        self.assertLess(result.q_net, 0.0)
        self.assertGreater(result.circulation_flow, 0.0)
        self.assertClose(result.net.hydraulic_driving_pressure, 0.0)
        self.assertClose(
            result.circulation.horizontal_phase_pressure_drive,
            5_000.0,
        )

    def test_dry_upward_trace_rejects_stale_or_pressure_reversed_net_flow(self) -> None:
        """Keep the activation failure loud until its two inputs are coupled."""

        downward_area = 0.20 * GEOMETRY.mouth_area
        gas_area = GEOMETRY.mouth_area - downward_area
        gas_mass = MATERIAL.gas_density * gas_area * PARAMETERS.riser_cell_length
        trace = PersistentMouthTrace(
            upward_area=0.0,
            upward_discharge=0.0,
            downward_area=downward_area,
            downward_discharge=-6.5e-3,
            gas_area=gas_area,
            gas_mass_flow=0.0,
            gas_cell_mass=gas_mass,
            gas_cell_momentum=0.0,
        )

        # Failure mode 1: a stale legacy junction characteristic points upward,
        # although the post-Taylor-sweep face has no upward liquid path.
        with self.assertRaisesRegex(
            BidirectionalTNodeError,
            "positive upward gross flow has no persistent upward mouth area",
        ):
            _advance(
                BidirectionalTNodeState(q_net=7.0e-4, circulation_flow=0.0),
                dt=1.0e-3,
                trace=trace,
                donors=_donors(dt=1.0e-3, node=1.0, riser=1.0),
                losses=NO_LOCAL_LOSSES,
            )

        # Failure mode 2: even a non-upward node state reverses if the horizontal
        # connected-gas pressure is compared with a stale, lower legacy liquid
        # pressure at the vertical mouth.  This is the measured activation
        # mismatch (about 4.5 kPa), not a fitted test pressure.
        with self.assertRaisesRegex(
            BidirectionalTNodeError,
            "positive upward gross flow has no persistent upward mouth area",
        ):
            _advance(
                BidirectionalTNodeState(q_net=0.0, circulation_flow=0.0),
                dt=1.0e-3,
                trace=trace,
                pressure=_pressure(delta=4_500.0),
                donors=_donors(dt=1.0e-3, node=1.0, riser=1.0),
                losses=NO_LOCAL_LOSSES,
            )

        # A common normal stress removes that false reversal.  This assertion
        # deliberately does not prescribe the eventual boundary-Riemann split;
        # it only proves the pressure inconsistency is independently causal.
        common_pressure = _advance(
            BidirectionalTNodeState(q_net=0.0, circulation_flow=0.0),
            dt=1.0e-3,
            trace=trace,
            pressure=_pressure(delta=0.0),
            donors=_donors(dt=1.0e-3, node=1.0, riser=1.0),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertClose(common_pressure.q_net, 0.0)
        self.assertClose(common_pressure.upward_flow, 0.0)
        self.assertClose(common_pressure.downward_flow, 0.0)

    def test_equal_counterflow_gravity_work_cancels_and_cannot_start_qc(self) -> None:
        result = _advance(losses=NO_LOCAL_LOSSES)
        self.assertClose(result.circulation_flow, 0.0)
        self.assertFalse(result.circulation.wallis_active)
        self.assertEqual(result.circulation.upper_bound_owner, "inactive")
        self.assertGreater(
            result.circulation.upward_core_gravity_pressure_downward,
            0.0,
        )
        self.assertClose(
            result.circulation.upward_core_gravity_pressure_downward,
            result.circulation.falling_film_gravity_pressure_downward,
        )
        self.assertClose(result.circulation.net_circulation_gravity_pressure, 0.0)
        self.assertClose(result.circulation.gas_velocity_before_exchange, 0.0)
        self.assertClose(result.circulation.gas_velocity_after_exchange, 0.0)
        self.assertClose(result.circulation.gas_reaction_impulse_upward, 0.0)
        self.assertClose(result.circulation.interphase_dissipation_energy, 0.0)
        self.assertClose(result.circulation.pressure_residual, 0.0, absolute=1.0e-7)

    def test_nusselt_is_only_the_low_re_equilibrium_audit(self) -> None:
        thickness = 5.0e-5
        film = dynamic_annular_film_geometry(
            _film_area(thickness),
            geometry=GEOMETRY,
        )
        density_difference = MATERIAL.liquid_density - MATERIAL.gas_density
        nusselt_velocity = (
            density_difference
            * GEOMETRY.gravity
            * film.thickness**2
            / (3.0 * MATERIAL.liquid_dynamic_viscosity)
        )
        q_nusselt = film.area * nusselt_velocity
        result = _advance(
            BidirectionalTNodeState(0.0, q_nusselt),
            trace=_trace(
                thickness=thickness,
                upward_discharge=q_nusselt,
                downward_discharge=-q_nusselt,
                no_gas_core=True,
            ),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertTrue(result.circulation.nusselt_applicable)
        self.assertClose(
            result.circulation.nusselt_equilibrium_flow,
            q_nusselt,
            rel=2.0e-12,
        )
        # The falling-film Nusselt value is not the equilibrium of an equal
        # up/down closed loop; with no gas drive the transient must decay.
        self.assertLess(result.circulation_flow, q_nusselt)
        self.assertNotEqual(result.circulation.upper_bound_owner, "nusselt")

    def test_nusselt_value_does_not_clip_a_transient_circulation(self) -> None:
        thickness = 5.0e-5
        film = dynamic_annular_film_geometry(_film_area(thickness), geometry=GEOMETRY)
        density_difference = MATERIAL.liquid_density - MATERIAL.gas_density
        q_nusselt = film.area * (
            density_difference
            * GEOMETRY.gravity
            * film.thickness**2
            / (3.0 * MATERIAL.liquid_dynamic_viscosity)
        )
        initial = 5.0 * q_nusselt
        result = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=1.0e-5,
            trace=_trace(
                thickness=thickness,
                upward_discharge=initial,
                downward_discharge=-initial,
                no_gas_core=True,
            ),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertGreater(result.circulation_flow, q_nusselt)
        self.assertNotEqual(
            result.circulation.upper_bound_owner,
            "nusselt",
        )

    def test_wallis_is_an_inactive_or_active_upper_complementarity(self) -> None:
        dt = 1.0e-5
        initial = 2.0e-5
        untriggered = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=dt,
            trace=_trace(
                upward_discharge=initial,
                downward_discharge=-initial,
                gas_mass_flow=1.0e-5,
            ),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertTrue(untriggered.circulation.wallis_active)
        self.assertLess(
            untriggered.circulation_flow,
            untriggered.circulation.wallis_circulation_capacity,
        )
        self.assertEqual(untriggered.circulation.wallis_reaction_pressure, 0.0)

        active = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=dt,
            trace=_trace(
                upward_discharge=initial,
                downward_discharge=-initial,
                gas_mass_flow=3.0e-2,
            ),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertClose(active.circulation.wallis_circulation_capacity, 0.0)
        self.assertClose(active.circulation_flow, 0.0)
        self.assertEqual(active.circulation.upper_bound_owner, "wallis")
        self.assertGreater(active.circulation.wallis_reaction_pressure, 0.0)
        self.assertClose(
            active.circulation.upper_complementarity_product,
            0.0,
            absolute=1.0e-18,
        )

    def test_donor_is_an_upper_bound_with_its_own_reaction(self) -> None:
        dt = 1.0e-5
        initial = 2.0e-5
        result = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=dt,
            trace=_trace(
                upward_discharge=initial,
                downward_discharge=-initial,
            ),
            donors=_donors(dt=dt, node=0.0, riser=1.0e-3),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertClose(result.circulation_flow, 0.0)
        self.assertEqual(result.circulation.upper_bound_owner, "node_donor")
        self.assertGreater(result.circulation.node_donor_reaction_pressure, 0.0)
        self.assertClose(result.circulation.upper_complementarity_product, 0.0)

    def test_resolved_gas_shear_uses_both_interfaces_and_their_work_difference(self) -> None:
        initial = 2.0e-5
        dt = 1.0e-4
        common = dict(
            upward_discharge=initial,
            downward_discharge=-initial,
        )
        stationary = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=dt,
            trace=_trace(gas_mass_flow=0.0, **common),
            losses=NO_LOCAL_LOSSES,
        )
        rising = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=dt,
            trace=_trace(gas_mass_flow=2.0e-4, **common),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertNotEqual(rising.circulation.gas_upward_core_pressure_upward, 0.0)
        self.assertNotEqual(rising.circulation.gas_film_pressure_upward, 0.0)
        self.assertClose(
            rising.circulation.gas_circulation_drive_pressure,
            rising.circulation.gas_upward_core_pressure_upward
            - rising.circulation.gas_film_pressure_upward,
        )
        self.assertLess(rising.circulation.gas_circulation_drive_pressure, 0.0)
        self.assertLess(rising.circulation_flow, stationary.circulation_flow)

        # A resolved downward gas trace reverses both tractions; the stronger
        # action on the falling-film interface gives positive generalized work.
        falling_gas = _advance(
            BidirectionalTNodeState(0.0, 0.0),
            dt=dt,
            trace=_trace(gas_mass_flow=-2.0e-4),
            losses=NO_LOCAL_LOSSES,
        )
        self.assertGreater(
            falling_gas.circulation.gas_circulation_drive_pressure,
            0.0,
        )
        self.assertGreater(falling_gas.circulation_flow, 0.0)

    def test_implicit_interphase_exchange_monotonically_reduces_relative_motion(self) -> None:
        initial = 2.0e-5
        result = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=0.10,
            trace=_trace(
                upward_discharge=initial,
                downward_discharge=-initial,
                gas_mass_flow=2.0e-3,
            ),
            donors=_donors(dt=0.10, node=1.0, riser=1.0),
            losses=NO_LOCAL_LOSSES,
        )
        ledger = result.circulation
        relative_measure_before = (
            ledger.upward_core_gas_conductance
            * ledger.upward_core_gas_slip_before**2
            + ledger.falling_film_gas_conductance
            * ledger.falling_film_gas_slip_before**2
        )
        relative_measure_after = (
            ledger.upward_core_gas_conductance
            * ledger.upward_core_gas_slip_after**2
            + ledger.falling_film_gas_conductance
            * ledger.falling_film_gas_slip_after**2
        )
        self.assertGreater(relative_measure_before, 0.0)
        self.assertLess(relative_measure_after, relative_measure_before)
        self.assertGreater(ledger.interphase_dissipation_energy, 0.0)
        self.assertClose(
            ledger.interphase_dissipation_energy,
            0.10 * relative_measure_after,
            rel=2.0e-12,
        )

    def test_large_dt_and_tiny_gas_area_remain_finite_without_impulse_overshoot(self) -> None:
        gas_area = 1.0e-8 * GEOMETRY.mouth_area
        downward_area = _film_area(5.0e-4)
        upward_area = GEOMETRY.mouth_area - downward_area - gas_area
        initial = 2.0e-5
        gas_velocity = 50.0
        trace = PersistentMouthTrace(
            upward_area=upward_area,
            upward_discharge=initial,
            downward_area=downward_area,
            downward_discharge=-initial,
            gas_area=gas_area,
            gas_mass_flow=MATERIAL.gas_density * gas_area * gas_velocity,
            gas_cell_mass=(
                MATERIAL.gas_density * gas_area * PARAMETERS.riser_cell_length
            ),
            gas_cell_momentum=(
                MATERIAL.gas_density
                * gas_area
                * PARAMETERS.riser_cell_length
                * gas_velocity
            ),
        )
        result = _advance(
            BidirectionalTNodeState(0.0, initial),
            dt=1.0,
            trace=trace,
            donors=_donors(dt=1.0, node=1.0, riser=1.0),
            losses=NO_LOCAL_LOSSES,
        )
        ledger = result.circulation
        self.assertTrue(math.isfinite(result.circulation_flow))
        self.assertTrue(math.isfinite(ledger.gas_velocity_after_exchange))
        gas_momentum_before = ledger.gas_cell_mass * gas_velocity
        gas_momentum_after = (
            ledger.gas_cell_mass * ledger.gas_velocity_after_exchange
        )
        self.assertClose(
            gas_momentum_after - gas_momentum_before,
            ledger.gas_reaction_impulse_upward,
            rel=2.0e-10,
            absolute=2.0e-18,
        )
        # Backward Euler forms a non-negative weighted average of the old gas
        # velocity and the two accepted liquid velocities.
        liquid_up = result.upward_velocity
        liquid_down = -result.downward_velocity
        lower = min(gas_velocity, liquid_up, liquid_down)
        upper = max(gas_velocity, liquid_up, liquid_down)
        self.assertGreaterEqual(ledger.gas_velocity_after_exchange, lower - 1.0e-12)
        self.assertLessEqual(ledger.gas_velocity_after_exchange, upper + 1.0e-12)
        self.assertGreaterEqual(ledger.interphase_dissipation_energy, 0.0)

    def test_geometric_gas_aperture_without_gas_mass_has_no_interphase_force(self) -> None:
        base = _trace(
            upward_discharge=1.0e-5,
            downward_discharge=-1.0e-5,
        )
        trace = PersistentMouthTrace(
            upward_area=base.upward_area,
            upward_discharge=base.upward_discharge,
            downward_area=base.downward_area,
            downward_discharge=base.downward_discharge,
            gas_area=base.gas_area,
            gas_mass_flow=0.0,
            gas_cell_mass=0.0,
            gas_cell_momentum=0.0,
        )
        result = _advance(
            BidirectionalTNodeState.from_persistent_trace(trace),
            trace=trace,
            losses=NO_LOCAL_LOSSES,
        )

        self.assertClose(result.circulation.gas_reaction_impulse_upward, 0.0)
        self.assertClose(result.circulation.liquid_gas_impulse_upward, 0.0)
        self.assertClose(result.circulation.gas_momentum_residual, 0.0)
        self.assertClose(result.net.interphase_momentum_residual, 0.0)

    def test_volume_momentum_gas_and_complementarity_ledgers_close(self) -> None:
        result = _advance(
            BidirectionalTNodeState(1.0e-5, 2.0e-6),
            trace=_trace(
                upward_discharge=1.2e-5,
                downward_discharge=-2.0e-6,
                gas_mass_flow=2.0e-4,
            ),
        )
        self.assertClose(result.closure_residual, 0.0, absolute=1.0e-16)
        self.assertClose(result.volume.combined_volume_residual, 0.0, absolute=1.0e-14)
        self.assertClose(result.net.pressure_residual, 0.0, absolute=1.0e-7)
        self.assertClose(result.net.momentum_residual, 0.0, absolute=1.0e-12)
        self.assertClose(result.net.interphase_momentum_residual, 0.0)
        self.assertClose(
            result.net.liquid_interphase_impulse_upward,
            -result.circulation.gas_reaction_impulse_upward,
        )
        self.assertClose(
            result.net.liquid_interphase_impulse_upward,
            GEOMETRY.mouth_area
            * 1.0e-2
            * result.net.gas_interphase_drive_pressure,
        )
        self.assertClose(result.circulation.pressure_residual, 0.0, absolute=1.0e-7)
        self.assertClose(result.circulation.film_momentum_residual, 0.0, absolute=1.0e-12)
        self.assertClose(result.circulation.gas_liquid_momentum_residual, 0.0)
        self.assertClose(result.circulation.gas_momentum_residual, 0.0)
        self.assertClose(
            result.circulation.gas_momentum_change,
            result.circulation.gas_cell_mass
            * (
                result.circulation.gas_velocity_after_exchange
                - result.circulation.gas_velocity_before_exchange
            ),
        )
        self.assertClose(
            result.circulation.liquid_gas_impulse_upward,
            result.circulation.upward_core_gas_impulse_upward
            + result.circulation.falling_film_gas_impulse_upward,
        )
        self.assertClose(
            result.circulation.liquid_gas_impulse_upward,
            -result.circulation.gas_reaction_impulse_upward,
        )
        self.assertClose(result.circulation.trace_net_flow, 1.0e-5)
        self.assertClose(result.circulation.trace_circulation_flow, 2.0e-6)
        self.assertClose(result.circulation.state_trace_net_mismatch, 0.0)
        self.assertClose(result.circulation.state_trace_circulation_mismatch, 0.0)

    def test_riser_receiver_caps_net_inflow_inside_node_solve(self) -> None:
        capacity = 2.0e-5
        result = _advance(
            pressure=_pressure(100_000.0),
            donors=_donors(node=1.0, riser=1.0),
            riser_receiving_net_rate_capacity=capacity,
        )

        self.assertClose(result.q_net, capacity)
        self.assertEqual(result.net.upper_bound_owner, "riser_receiver")
        self.assertClose(result.upward_flow - result.downward_flow, capacity)
        self.assertClose(result.net.pressure_residual, 0.0, absolute=1.0e-7)
        self.assertClose(result.net.momentum_residual, 0.0, absolute=1.0e-12)

    def test_implicit_gas_inertia_uses_supplied_conservative_cell_state(self) -> None:
        base = _trace(
            upward_discharge=1.0e-5,
            downward_discharge=-1.0e-5,
            gas_mass_flow=1.0e-4,
        )
        supplied_mass = 7.5e-6
        supplied_velocity = 3.25
        trace = PersistentMouthTrace(
            upward_area=base.upward_area,
            upward_discharge=base.upward_discharge,
            downward_area=base.downward_area,
            downward_discharge=base.downward_discharge,
            gas_area=base.gas_area,
            gas_mass_flow=base.gas_mass_flow,
            gas_cell_mass=supplied_mass,
            gas_cell_momentum=supplied_mass * supplied_velocity,
        )
        result = _advance(
            BidirectionalTNodeState.from_persistent_trace(trace),
            dt=0.05,
            trace=trace,
            donors=_donors(dt=0.05, node=1.0, riser=1.0),
            losses=NO_LOCAL_LOSSES,
        )
        ledger = result.circulation
        self.assertClose(ledger.gas_cell_mass, supplied_mass)
        self.assertClose(ledger.gas_velocity_before_exchange, supplied_velocity)
        self.assertClose(
            ledger.gas_reaction_impulse_upward,
            supplied_mass
            * (ledger.gas_velocity_after_exchange - supplied_velocity),
        )

    def test_dynamic_film_geometry_is_inventory_driven_and_monotone(self) -> None:
        thin = dynamic_annular_film_geometry(_film_area(2.0e-4), geometry=GEOMETRY)
        thick = dynamic_annular_film_geometry(_film_area(8.0e-4), geometry=GEOMETRY)
        self.assertClose(thin.thickness, 2.0e-4, rel=2.0e-12)
        self.assertClose(thick.thickness, 8.0e-4, rel=2.0e-12)
        self.assertGreater(thick.area, thin.area)
        self.assertGreater(thick.hydraulic_diameter, thin.hydraulic_diameter)

    def test_dynamic_film_geometry_accepts_only_roundoff_above_bore(self) -> None:
        bore = GEOMETRY.mouth_area
        rounded = dynamic_annular_film_geometry(
            math.nextafter(bore, math.inf), geometry=GEOMETRY
        )
        self.assertEqual(rounded.area, bore)
        with self.assertRaisesRegex(ValueError, "exceeds the riser bore"):
            dynamic_annular_film_geometry(1.0001 * bore, geometry=GEOMETRY)

    def test_dynamic_step_builds_the_existing_shared_gross_plan(self) -> None:
        trace = _trace()
        result = _advance(trace=trace, losses=NO_LOCAL_LOSSES)
        plan = stage_bidirectional_tnode_coupling(
            result,
            trace=trace,
            geometry=GEOMETRY,
            material=MATERIAL,
            losses=NO_LOCAL_LOSSES,
            horizontal_axial_velocity=0.2,
            horizontal_node_topology=HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT,
        )
        self.assertClose(plan.exchange.upward_flow, result.upward_flow)
        self.assertClose(plan.exchange.downward_flow, result.downward_flow)
        self.assertClose(plan.exchange.circulation_flow, result.circulation_flow)
        self.assertClose(plan.vertical_boundary.total_volume_rate, result.q_net)
        self.assertClose(plan.combined_liquid_volume_rate, 0.0)

        update = apply_twochannel_horizontal_footprint(
            [GEOMETRY.horizontal_full_area, GEOMETRY.horizontal_full_area],
            [0.0, 0.0],
            [0.5, 0.5],
            cell_width=0.03,
            opening_length=GEOMETRY.opening_footprint_length,
            time_step=1.0e-2,
            plan=plan,
        )
        self.assertClose(
            update.net_horizontal_volume_change,
            -result.q_net * 1.0e-2,
            absolute=2.0e-14,
        )
        self.assertClose(update.volume_residual, 0.0, absolute=2.0e-14)

    def test_api_has_no_clock_target_or_external_field_input(self) -> None:
        signature = inspect.signature(advance_bidirectional_tnode_inertance)
        names = set(signature.parameters)
        self.assertFalse(
            names
            & {
                "time",
                "current_time",
                "target_volume",
                "target_height",
                "openfoam",
                "two_dimensional_field",
            }
        )
        self.assertIn("dt", names)
        self.assertIn("trace", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
