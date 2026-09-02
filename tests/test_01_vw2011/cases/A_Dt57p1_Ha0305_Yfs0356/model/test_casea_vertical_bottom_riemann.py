from __future__ import annotations

import math
import unittest

from casea_vertical_bottom_riemann import (
    BottomMouthRiemannError,
    resolve_bottom_mouth_riemann,
    solve_coupled_gross_mouth_characteristics,
    solve_upward_incoming_characteristic,
)


class BottomMouthRiemannTests(unittest.TestCase):
    def assertClose(
        self,
        actual: float,
        expected: float,
        *,
        absolute: float = 1.0e-14,
    ) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=absolute),
            msg=f"{actual!r} != {expected!r}",
        )

    @staticmethod
    def solve(**overrides):
        values = dict(
            incoming_upward_characteristic_rate=3.0e-3,
            incoming_upward_characteristic_speed=1.0,
            liquid_area_capacity=1.0e-2,
            first_cell_downward_area=5.0e-3,
            first_cell_downward_discharge=-1.0e-2,
            resolved_downward_mouth_area=2.0e-3,
            finite_node_liquid_volume=1.0,
            riser_downward_donor_volume=1.0,
            time_step=0.1,
            positive_net_receiving_capacity=math.inf,
            negative_net_receiving_capacity=math.inf,
            wallis_downward_capacity=math.inf,
            enforce_wallis_constraint=False,
        )
        values.update(overrides)
        return resolve_bottom_mouth_riemann(**values)

    def assertClosed(self, result) -> None:
        ledger = result.ledger
        self.assertClose(result.q_net, ledger.q_net)
        self.assertClose(ledger.net_flux_residual, 0.0)
        self.assertClose(ledger.momentum_residual, 0.0)
        self.assertClose(ledger.mouth_area_residual, 0.0)
        self.assertClose(ledger.combined_volume_residual, 0.0)

    def test_derives_net_rate_from_two_independent_gross_characteristics(self) -> None:
        result = self.solve()
        self.assertClose(result.flux.upward_rate, 3.0e-3)
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertClose(result.q_net, -1.0e-3)
        self.assertClosed(result)

    def test_maps_cell_to_narrower_mouth_by_preserving_velocity_not_discharge(self) -> None:
        result = self.solve()
        self.assertClose(result.ledger.first_cell_downward_rate, 1.0e-2)
        self.assertClose(result.ledger.first_cell_downward_speed, 2.0)
        self.assertClose(result.ledger.outgoing_mouth_downward_rate, 4.0e-3)
        self.assertClose(result.flux.downward_speed, 2.0)
        self.assertClose(result.downward_area, 2.0e-3)
        self.assertClosed(result)

    def test_same_step_return_does_not_fund_upward_node_donor(self) -> None:
        result = self.solve(finite_node_liquid_volume=1.0e-4)
        self.assertClose(result.flux.upward_rate, 1.0e-3)
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertIn("finite_node_donor", result.ledger.active_constraints)
        self.assertClosed(result)

    def test_riser_donor_limits_only_the_outgoing_stream(self) -> None:
        result = self.solve(riser_downward_donor_volume=2.5e-4)
        self.assertClose(result.flux.downward_rate, 2.5e-3)
        self.assertClose(result.downward_area, 2.0e-3)
        self.assertClose(result.flux.downward_speed, 1.25)
        self.assertClose(result.flux.upward_rate, 3.0e-3)
        self.assertClose(result.q_net, 0.5e-3)
        self.assertIn("riser_downward_donor", result.ledger.active_constraints)
        self.assertGreater(result.ledger.downward_constraint_reaction_flux, 0.0)
        self.assertClosed(result)

    def test_shared_aperture_reduces_only_the_incoming_stream(self) -> None:
        result = self.solve(liquid_area_capacity=4.0e-3)
        # Down occupies 0.002 m2; the remaining 0.002 m2 carries upward water.
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertClose(result.flux.upward_rate, 2.0e-3)
        self.assertClose(result.upward_area, 2.0e-3)
        self.assertClose(result.unused_mouth_area, 0.0)
        self.assertIn("shared_aperture", result.ledger.active_constraints)
        self.assertClosed(result)

    def test_positive_net_receiver_reduces_only_incoming_water(self) -> None:
        result = self.solve(
            incoming_upward_characteristic_rate=8.0e-3,
            positive_net_receiving_capacity=1.0e-3,
        )
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertClose(result.flux.upward_rate, 5.0e-3)
        self.assertClose(result.upward_area, 8.0e-3)
        self.assertClose(result.flux.upward_speed, 0.625)
        self.assertClose(result.q_net, 1.0e-3)
        self.assertIn("positive_net_receiver", result.ledger.active_constraints)
        self.assertClosed(result)

    def test_wallis_is_a_reference_for_an_inherited_churn_film(self) -> None:
        result = self.solve(wallis_downward_capacity=2.0e-3)
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertClose(result.ledger.wallis_excess_rate, 2.0e-3)
        self.assertFalse(result.ledger.wallis_constraint_applied)
        self.assertNotIn("wallis", result.ledger.active_constraints)
        self.assertClosed(result)

    def test_wallis_can_be_activated_for_a_quasi_steady_ccfl_topology(self) -> None:
        result = self.solve(
            wallis_downward_capacity=2.0e-3,
            enforce_wallis_constraint=True,
        )
        self.assertClose(result.flux.downward_rate, 2.0e-3)
        self.assertIn("wallis", result.ledger.active_constraints)
        self.assertGreater(result.ledger.downward_constraint_reaction_flux, 0.0)
        self.assertClosed(result)

    def test_pure_downward_outflow_requires_no_upward_characteristic(self) -> None:
        result = self.solve(
            incoming_upward_characteristic_rate=0.0,
            incoming_upward_characteristic_speed=0.0,
        )
        self.assertClose(result.flux.upward_rate, 0.0)
        self.assertClose(result.flux.downward_rate, 4.0e-3)
        self.assertClose(result.q_net, -4.0e-3)
        self.assertClosed(result)

    def test_rejects_downward_mouth_corridor_larger_than_liquid_aperture(self) -> None:
        with self.assertRaises(BottomMouthRiemannError):
            self.solve(
                liquid_area_capacity=1.0e-3,
                resolved_downward_mouth_area=2.0e-3,
            )

    def test_upward_characteristic_uses_liquid_pressure_and_implicit_turn_loss(self) -> None:
        result = solve_upward_incoming_characteristic(
            old_upward_speed=0.2,
            horizontal_liquid_pressure_abs=102_000.0,
            vertical_liquid_pressure_abs=101_000.0,
            liquid_density=1000.0,
            effective_inertance_length=0.2,
            time_step=0.1,
            upward_turn_loss_coefficient=1.5,
        )
        expected = 2.0 * 1400.0 / (2000.0 + math.sqrt(8_200_000.0))
        self.assertClose(result.unconstrained_speed, expected)
        self.assertGreater(result.accepted_speed, 0.0)
        self.assertClose(result.accepted_speed, result.unconstrained_speed)
        self.assertGreater(result.turn_loss_pressure, 0.0)
        self.assertClose(result.pressure_residual, 0.0, absolute=1.0e-10)

    def test_adverse_pressure_stops_incoming_characteristic_with_reaction(self) -> None:
        result = solve_upward_incoming_characteristic(
            old_upward_speed=0.1,
            horizontal_liquid_pressure_abs=100_000.0,
            vertical_liquid_pressure_abs=101_000.0,
            liquid_density=1000.0,
            effective_inertance_length=0.2,
            time_step=0.1,
            upward_turn_loss_coefficient=1.0,
        )
        self.assertLess(result.unconstrained_speed, 0.0)
        self.assertClose(result.accepted_speed, 0.0)
        self.assertGreater(result.lower_bound_reaction_pressure, 0.0)
        self.assertClose(result.pressure_residual, 0.0, absolute=1.0e-10)

    @staticmethod
    def coupled(**overrides):
        values = dict(
            old_upward_speed=0.30,
            raw_downward_speed=1.20,
            upward_area=3.0e-3,
            downward_area=2.0e-3,
            horizontal_liquid_pressure_abs=102_000.0,
            vertical_liquid_pressure_abs=101_000.0,
            liquid_density=1000.0,
            effective_inertance_length=0.20,
            time_step=0.01,
            downward_characteristic_celerity=1.20,
            upward_turn_loss_coefficient=0.75,
            downward_turn_loss_coefficient=0.75,
            countercurrent_mixing_coefficient=8.0,
        )
        values.update(overrides)
        return solve_coupled_gross_mouth_characteristics(**values)

    def test_coupled_mixing_has_exact_action_reaction_and_power(self) -> None:
        result = self.coupled()
        self.assertGreater(result.upward_speed, 0.0)
        self.assertGreater(result.downward_speed, 0.0)
        self.assertGreater(result.mixing_force, 0.0)
        self.assertClose(
            result.mixing_loss_power,
            result.mixing_force
            * (result.upward_speed + result.downward_speed),
            absolute=1.0e-12,
        )
        self.assertClose(
            result.mixing_kinematic_reaction_flux,
            result.mixing_force / 1000.0,
            absolute=1.0e-15,
        )
        self.assertClose(
            result.upward_pressure_residual,
            0.0,
            absolute=1.0e-8,
        )
        self.assertClose(
            result.downward_pressure_residual,
            0.0,
            absolute=1.0e-8,
        )

    def test_mixing_vanishes_when_either_stream_is_absent(self) -> None:
        no_down = self.coupled(raw_downward_speed=0.0)
        no_up = self.coupled(
            old_upward_speed=0.0,
            horizontal_liquid_pressure_abs=100_000.0,
            vertical_liquid_pressure_abs=101_000.0,
        )
        self.assertClose(no_down.mixing_force, 0.0)
        self.assertClose(no_up.upward_speed, 0.0)
        self.assertClose(no_up.mixing_force, 0.0)
        self.assertGreater(no_up.upward_lower_bound_reaction_pressure, 0.0)
        self.assertClose(
            no_up.upward_pressure_residual,
            0.0,
            absolute=1.0e-10,
        )

    def test_larger_physical_losses_do_not_accelerate_either_stream(self) -> None:
        no_mix = self.coupled(countercurrent_mixing_coefficient=0.0)
        mixed = self.coupled(countercurrent_mixing_coefficient=8.0)
        stronger_down_turn = self.coupled(
            downward_turn_loss_coefficient=2.0,
        )
        self.assertLessEqual(mixed.upward_speed, no_mix.upward_speed)
        self.assertLessEqual(mixed.downward_speed, no_mix.downward_speed)
        self.assertLessEqual(
            stronger_down_turn.downward_speed,
            mixed.downward_speed,
        )

    def test_physical_downward_speed_and_reaction_reach_the_boundary_ledger(self) -> None:
        result = self.solve(
            physical_downward_mouth_speed=1.5,
            downward_physical_reaction_flux=2.5e-4,
        )
        self.assertClose(result.flux.downward_rate, 3.0e-3)
        self.assertClose(result.flux.downward_speed, 1.5)
        self.assertClose(
            result.ledger.downward_physical_reaction_flux,
            2.5e-4,
        )
        self.assertGreater(
            result.ledger.downward_boundary_reaction_flux,
            result.ledger.downward_physical_reaction_flux,
        )
        self.assertClosed(result)

    def test_downward_acceleration_requires_an_explicit_pressure_source(self) -> None:
        with self.assertRaises(BottomMouthRiemannError):
            self.solve(
                first_cell_downward_area=2.0e-3,
                first_cell_downward_discharge=-4.0e-3,
                physical_downward_mouth_speed=2.5,
            )

        result = self.solve(
            first_cell_downward_area=2.0e-3,
            first_cell_downward_discharge=-4.0e-3,
            physical_downward_mouth_speed=2.5,
            downward_pressure_acceleration_flux=4.5e-3,
        )
        self.assertClose(result.flux.downward_rate, 5.0e-3)
        self.assertClose(result.flux.downward_speed, 2.5)
        self.assertClose(
            result.ledger.downward_pressure_acceleration_flux,
            4.5e-3,
        )
        self.assertClosed(result)


if __name__ == "__main__":
    unittest.main()
