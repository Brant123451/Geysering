"""Standard-library tests for the isolated two-channel integration adapter."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    LiquidDonorInventories,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    DuplicateMouthFluxOwner,
    HorizontalNodeTopology,
    LegacyMouthPathActivity,
    SecondLiquidMomentumRequired,
    advance_lumped_liquid_inventories,
    apply_twochannel_horizontal_footprint,
    finite_width_node_liquid_inventory,
    stage_from_finite_node_ssprk2,
    stage_twochannel_mouth_coupling,
)


GEOMETRY = VerticalMouthGeometry(diameter=0.0571)
MATERIAL = VerticalMouthMaterialProperties(
    liquid_density=998.0,
    gas_density=1.20,
    liquid_dynamic_viscosity=1.003e-3,
)
WALLIS = WallisCounterCurrentParameters(constant=0.50)
LOSSES = DirectionalMouthLosses(
    upward_turn=0.75,
    downward_turn=1.25,
    countercurrent_mixing=4.0,
)


def _phase(*, liquid_fraction: float = 0.25, gas_velocity: float = 0.70):
    liquid_area = liquid_fraction * GEOMETRY.full_area
    return VerticalMouthPhaseState(
        liquid_area=liquid_area,
        liquid_velocity=0.0,
        gas_area=GEOMETRY.full_area - liquid_area,
        gas_velocity=gas_velocity,
    )


def _plan(
    q_net: float = 0.0,
    *,
    phase: VerticalMouthPhaseState | None = None,
    activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
):
    return stage_twochannel_mouth_coupling(
        q_net,
        phase=_phase() if phase is None else phase,
        geometry=GEOMETRY,
        material=MATERIAL,
        wallis=WALLIS,
        donors=LiquidDonorInventories(
            finite_node_volume=1.0e-3,
            riser_volume=1.0e-3,
            time_step=1.0e-2,
        ),
        losses=LOSSES,
        horizontal_axial_velocity=0.35,
        horizontal_node_topology=HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT,
        legacy_activity=activity,
    )


class TwoChannelIntegrationTests(unittest.TestCase):
    def test_plan_preserves_one_shared_net_flux(self) -> None:
        for q_net in (2.0e-5, 0.0, -3.0e-5):
            with self.subTest(q_net=q_net):
                plan = _plan(q_net)
                self.assertGreater(plan.exchange.circulation_flow, 0.0)
                self.assertAlmostEqual(
                    plan.exchange.upward_flow - plan.exchange.downward_flow,
                    q_net,
                    delta=3.0e-20,
                )
                self.assertAlmostEqual(
                    plan.vertical_boundary.total_volume_rate,
                    q_net,
                    delta=3.0e-20,
                )
                self.assertAlmostEqual(
                    plan.combined_liquid_volume_rate,
                    0.0,
                    delta=3.0e-20,
                )

    def test_each_legacy_mass_owner_is_rejected(self) -> None:
        activities = (
            LegacyMouthPathActivity(characteristic_bottom_flux_applied=True),
            LegacyMouthPathActivity(taylor_return_mass_flux_applied=True),
            LegacyMouthPathActivity(post_breakthrough_ccfl_net_flux_applied=True),
            LegacyMouthPathActivity(net_only_horizontal_side_source_applied=True),
        )
        for activity in activities:
            with self.subTest(active=activity.active_paths):
                with self.assertRaises(DuplicateMouthFluxOwner):
                    _plan(activity=activity)

    def test_countercurrent_second_moment_requires_second_momentum(self) -> None:
        countercurrent = _plan(0.0)
        self.assertGreater(countercurrent.exchange.circulation_flow, 0.0)
        self.assertGreater(
            countercurrent.vertical_boundary.total_convective_momentum_flux,
            0.0,
        )
        # q_net alone gives zero bulk convective momentum at this face.  The
        # positive gross second moment is independent information.
        self.assertEqual(countercurrent.exchange.q_net, 0.0)
        with self.assertRaises(SecondLiquidMomentumRequired):
            countercurrent.single_momentum_vertical_flux()

        no_gas = _plan(2.0e-5, phase=_phase(gas_velocity=0.0))
        self.assertEqual(no_gas.exchange.circulation_flow, 0.0)
        self.assertAlmostEqual(
            no_gas.single_momentum_vertical_flux(),
            no_gas.exchange.q_net**2 / no_gas.exchange.upward_channel_area,
        )

    def test_lumped_inventory_update_is_exactly_conservative(self) -> None:
        plan = _plan(2.0e-5)
        result = advance_lumped_liquid_inventories(
            4.0e-4,
            3.0e-4,
            time_step=0.02,
            plan=plan,
        )
        self.assertAlmostEqual(
            result.finite_node_liquid_volume,
            4.0e-4 - 2.0e-5 * 0.02,
        )
        self.assertAlmostEqual(
            result.riser_liquid_volume,
            3.0e-4 + 2.0e-5 * 0.02,
        )
        self.assertAlmostEqual(result.conservation_residual, 0.0, delta=1.0e-18)

    def test_gross_footprint_exchange_changes_momentum_at_zero_net_flow(self) -> None:
        plan = _plan(0.0)
        area = np.array([0.45, 0.70, 0.55]) * GEOMETRY.full_area
        discharge = area * np.array([0.30, 0.20, 0.10])
        weights = np.array([0.20, 0.60, 0.20])
        old_volume = float(np.sum(area) * 0.02)
        old_momentum = float(np.sum(discharge) * 0.02)
        update = apply_twochannel_horizontal_footprint(
            area,
            discharge,
            weights,
            cell_width=0.02,
            time_step=1.0e-3,
            plan=plan,
        )
        new_volume = float(np.sum(update.liquid_area) * 0.02)
        new_momentum = float(np.sum(update.liquid_discharge) * 0.02)
        self.assertAlmostEqual(new_volume, old_volume, delta=1.0e-18)
        self.assertAlmostEqual(
            update.removed_upward_volume,
            update.deposited_downward_volume,
            delta=1.0e-18,
        )
        self.assertLess(new_momentum, old_momentum)
        self.assertLess(update.axial_kinematic_momentum_change, 0.0)
        self.assertAlmostEqual(update.volume_residual, 0.0, delta=1.0e-18)

    def test_weighted_inventory_matches_the_footprint_donor_definition(self) -> None:
        area = np.array([0.20, 0.80]) * GEOMETRY.full_area
        weights = np.array([1.0, 3.0])
        inventory = finite_width_node_liquid_inventory(
            area,
            weights,
            cell_width=0.04,
        )
        expected = (0.25 * area[0] + 0.75 * area[1]) * 0.04
        self.assertAlmostEqual(inventory, expected)

    def test_measured_opening_length_makes_inventory_grid_independent(self) -> None:
        opening_length = 0.0571
        liquid_area = 0.60 * GEOMETRY.full_area
        coarse = finite_width_node_liquid_inventory(
            [liquid_area],
            [1.0],
            cell_width=0.08,
            opening_length=opening_length,
        )
        fine = finite_width_node_liquid_inventory(
            [liquid_area, liquid_area],
            [0.5, 0.5],
            cell_width=0.04,
            opening_length=opening_length,
        )
        expected = liquid_area * opening_length
        self.assertAlmostEqual(coarse, expected, delta=1.0e-18)
        self.assertAlmostEqual(fine, expected, delta=1.0e-18)

    def test_explicit_finite_node_adapter_does_not_allow_a_side_source(self) -> None:
        fake_result = SimpleNamespace(
            vertical=SimpleNamespace(liquid_area=1.0e-5),
            ledger=SimpleNamespace(
                dt=1.0e-3,
                initial_state=SimpleNamespace(liquid_equivalent_volume=2.0e-4),
            ),
        )
        plan = stage_from_finite_node_ssprk2(
            fake_result,
            phase=_phase(),
            geometry=GEOMETRY,
            material=MATERIAL,
            wallis=WALLIS,
            riser_liquid_donor_volume=2.0e-4,
            losses=LOSSES,
        )
        self.assertIs(
            plan.horizontal_node_topology,
            HorizontalNodeTopology.EXPLICIT_FINITE_NODE,
        )
        self.assertAlmostEqual(plan.exchange.q_net, 1.0e-5)
        with self.assertRaises(DuplicateMouthFluxOwner):
            apply_twochannel_horizontal_footprint(
                [0.5 * GEOMETRY.full_area],
                [0.0],
                [1.0],
                cell_width=0.02,
                time_step=1.0e-3,
                plan=plan,
            )

    def test_adapter_has_no_result_or_event_prescription(self) -> None:
        source = inspect.getsource(stage_twochannel_mouth_coupling).lower()
        forbidden = (
            "target_height",
            "event_time",
            "current_time",
            "frame",
            "animation",
            "render",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertNotIn("8.85", source)
        self.assertNotIn("0.104", source)

    def test_returned_energy_losses_are_nonnegative(self) -> None:
        plan = _plan(-2.0e-5)
        exchange = plan.exchange
        self.assertGreater(exchange.upward_flow, 0.0)
        self.assertGreater(exchange.downward_flow, 0.0)
        self.assertGreaterEqual(exchange.total_dissipation_power, 0.0)
        self.assertTrue(math.isfinite(exchange.total_dissipation_power))


if __name__ == "__main__":
    unittest.main()
