from __future__ import annotations

import unittest

from campaign2_tee_riemann import (
    GasTrace,
    LiquidBranchTrace,
    _checked_liquid_continuity_residual_m3_s,
    solve_first_bottom_gas_entry,
    solve_gas_tee,
    solve_liquid_tee,
    solve_liquid_tee_with_blocked_riser,
)
from case1_persistent_coupling import transaction_from_tee_solutions


class Campaign2TeeRiemannTests(unittest.TestCase):
    def test_finite_pocket_blocks_riser_liquid_and_preserves_reference_shift(self) -> None:
        west = LiquidBranchTrace(1.9e-3, -0.12, 8400.0, 28.0)
        east = LiquidBranchTrace(1.7e-3, 0.08, 7600.0, 28.0)
        base = solve_liquid_tee_with_blocked_riser(west, east)
        shift = 12_345.0
        shifted = solve_liquid_tee_with_blocked_riser(
            LiquidBranchTrace(
                west.area_m2,
                west.outward_velocity_m_s,
                west.gauge_pressure_Pa + shift,
                west.wave_speed_m_s,
                west.density_kg_m3,
            ),
            LiquidBranchTrace(
                east.area_m2,
                east.outward_velocity_m_s,
                east.gauge_pressure_Pa + shift,
                east.wave_speed_m_s,
                east.density_kg_m3,
            ),
        )
        self.assertAlmostEqual(base.riser_outward_flow_m3_s, 0.0, places=18)
        self.assertAlmostEqual(base.normal_momentum_to_riser_N, 0.0, places=18)
        self.assertAlmostEqual(
            base.west_outward_flow_m3_s + base.east_outward_flow_m3_s,
            0.0,
            places=18,
        )
        self.assertAlmostEqual(
            shifted.west_outward_flow_m3_s,
            base.west_outward_flow_m3_s,
            places=17,
        )
        self.assertAlmostEqual(
            shifted.east_outward_flow_m3_s,
            base.east_outward_flow_m3_s,
            places=17,
        )
        self.assertAlmostEqual(
            shifted.node_gauge_pressure_Pa,
            base.node_gauge_pressure_Pa + shift,
            places=9,
        )

    def test_equal_liquid_state_is_exactly_stationary(self) -> None:
        trace = LiquidBranchTrace(1.0e-3, 0.0, 6000.0, 28.0)
        solution = solve_liquid_tee(trace, trace, trace)
        self.assertAlmostEqual(solution.node_gauge_pressure_Pa, 6000.0, places=10)
        self.assertAlmostEqual(solution.west_outward_flow_m3_s, 0.0, places=15)
        self.assertAlmostEqual(solution.east_outward_flow_m3_s, 0.0, places=15)
        self.assertAlmostEqual(solution.riser_outward_flow_m3_s, 0.0, places=15)

    def test_liquid_node_closes_and_maps_to_physical_face_flows(self) -> None:
        west = LiquidBranchTrace(1.9e-3, 0.0, 9000.0, 28.0)
        east = LiquidBranchTrace(1.9e-3, 0.0, 9000.0, 28.0)
        riser = LiquidBranchTrace(5.3e-4, 0.0, 2000.0, 28.0)
        solution = solve_liquid_tee(west, east, riser)
        self.assertGreater(solution.riser_outward_flow_m3_s, 0.0)
        self.assertAlmostEqual(solution.continuity_residual_m3_s, 0.0, places=15)
        self.assertAlmostEqual(
            solution.physical_west_flow_m3_s
            - solution.physical_east_flow_m3_s,
            solution.riser_outward_flow_m3_s,
            places=15,
        )

    def test_reversing_liquid_pressure_reverses_riser_flow(self) -> None:
        horizontal = LiquidBranchTrace(1.9e-3, 0.0, 2000.0, 28.0)
        riser = LiquidBranchTrace(5.3e-4, 0.0, 9000.0, 28.0)
        solution = solve_liquid_tee(horizontal, horizontal, riser)
        self.assertLess(solution.riser_outward_flow_m3_s, 0.0)

    def test_stationary_h3_roundoff_reproduction_is_not_rejected(self) -> None:
        # Exact traces captured at the former t ~= 0.1418 s failure.  The three
        # O(1e-17 m3/s) returned flows arise by cancelling O(1e-4 m3/s)
        # characteristic terms; their O(1e-21 m3/s) balance is roundoff, not a
        # physical source.  These values contain no case label or outcome hook.
        west = LiquidBranchTrace(
            area_m2=0.001963495408493621,
            outward_velocity_m_s=-2.2793453685581712e-15,
            gauge_pressure_Pa=5973.328620000023,
            wave_speed_m_s=28.0,
            density_kg_m3=998.2,
        )
        east = LiquidBranchTrace(
            area_m2=0.001963495408493621,
            outward_velocity_m_s=1.1382743128873317e-14,
            gauge_pressure_Pa=5973.3286199996755,
            wave_speed_m_s=28.0,
            density_kg_m3=998.2,
        )
        riser = LiquidBranchTrace(
            area_m2=0.000530929158456675,
            outward_velocity_m_s=0.0,
            gauge_pressure_Pa=5973.328620000393,
            wave_speed_m_s=28.0,
            density_kg_m3=998.2,
        )

        solution = solve_liquid_tee(west, east, riser)

        self.assertLess(
            abs(solution.continuity_residual_m3_s),
            1.0e-19,
        )
        self.assertLess(
            max(
                abs(solution.west_outward_flow_m3_s),
                abs(solution.east_outward_flow_m3_s),
                abs(solution.riser_outward_flow_m3_s),
            ),
            1.0e-15,
        )

    def test_resolved_liquid_continuity_defect_is_rejected(self) -> None:
        # A 1e-5 m3/s source on a 1e-4 m3/s characteristic-flow scale is many
        # orders above binary64 roundoff and must never be accepted.
        with self.assertRaisesRegex(
            FloatingPointError,
            "continuity did not close",
        ):
            _checked_liquid_continuity_residual_m3_s(
                (1.0e-4, -6.0e-5, -3.0e-5),
                (1.0e-4, -6.0e-5, -3.0e-5),
            )

    def test_equal_gas_state_has_zero_mass_flow(self) -> None:
        trace = GasTrace(101325.0, 1.19, 0.0, 345.0)
        solution = solve_gas_tee(trace, trace, open_area_m2=5.0e-4)
        self.assertAlmostEqual(solution.interface_velocity_to_riser_m_s, 0.0, places=15)
        self.assertAlmostEqual(solution.mass_flow_to_riser_kg_s, 0.0, places=15)
        self.assertEqual(solution.volume_flow_to_riser_m3_s, 0.0)
        self.assertAlmostEqual(solution.normal_momentum_flow_N, 0.0, places=15)
        self.assertGreater(solution.interface_pressure_force_N, 0.0)
        self.assertAlmostEqual(
            solution.total_conservative_momentum_flux_N,
            solution.interface_pressure_force_N,
            places=15,
        )

    def test_gas_pressure_jump_sets_physical_flow_direction(self) -> None:
        horizontal = GasTrace(108000.0, 1.27, 0.0, 345.0)
        riser = GasTrace(101325.0, 1.19, 0.0, 345.0)
        upward = solve_gas_tee(horizontal, riser, open_area_m2=5.0e-4)
        downward = solve_gas_tee(riser, horizontal, open_area_m2=5.0e-4)
        self.assertGreater(upward.mass_flow_to_riser_kg_s, 0.0)
        self.assertLess(downward.mass_flow_to_riser_kg_s, 0.0)
        self.assertAlmostEqual(
            upward.volume_flow_to_riser_m3_s,
            5.0e-4 * upward.interface_velocity_to_riser_m_s,
            places=15,
        )
        self.assertAlmostEqual(
            upward.mass_flow_to_riser_kg_s,
            horizontal.density_kg_m3 * upward.volume_flow_to_riser_m3_s,
            places=15,
        )

    def test_first_entry_common_pressure_ale_and_nonoverlap_close(self) -> None:
        west = LiquidBranchTrace(1.9e-3, 0.0, 0.0, 28.0, 998.0)
        riser = LiquidBranchTrace(5.3e-4, 0.0, 0.0, 28.0, 998.0)
        horizontal_gas = GasTrace(110_000.0, 1.30, 0.0, 340.0)
        reference = 101_325.0
        solution = solve_first_bottom_gas_entry(
            west,
            west,
            riser,
            horizontal_gas,
            liquid_pressure_reference_abs_Pa=reference,
            available_gas_open_area_m2=1.0e-4,
        )

        self.assertTrue(solution.active)
        self.assertGreater(solution.gas.volume_flow_to_riser_m3_s, 0.0)
        self.assertEqual(
            reference + solution.liquid.node_gauge_pressure_Pa,
            solution.gas.interface_pressure_abs_Pa,
        )
        self.assertAlmostEqual(solution.ale_volume_residual_m3_s, 0.0, places=17)
        self.assertAlmostEqual(
            solution.liquid_plug_flow_m3_s,
            solution.liquid.riser_outward_flow_m3_s
            + solution.gas.volume_flow_to_riser_m3_s,
            places=17,
        )
        self.assertEqual(
            solution.gas_open_area_m2 + solution.liquid_open_area_m2,
            riser.area_m2,
        )
        self.assertEqual(
            solution.gas.open_area_m2,
            solution.gas_open_area_m2,
        )
        self.assertEqual(
            solution.liquid.riser_open_area_m2,
            solution.liquid_open_area_m2,
        )
        transaction = transaction_from_tee_solutions(
            solution.liquid,
            solution.gas,
            physical_riser_area_m2=riser.area_m2,
        )
        self.assertEqual(transaction.gas_open_area_m2, solution.gas_open_area_m2)
        self.assertEqual(
            transaction.liquid_open_area_m2,
            solution.liquid_open_area_m2,
        )
        self.assertEqual(transaction.blocked_riser_area_m2, 0.0)
        expected_liquid_momentum = (
            riser.density_kg_m3
            * solution.liquid.riser_outward_flow_m3_s**2
            / solution.liquid_open_area_m2
        )
        self.assertAlmostEqual(
            solution.liquid.normal_momentum_to_riser_N,
            expected_liquid_momentum,
            places=15,
        )

    def test_first_entry_reference_shift_is_physically_invariant(self) -> None:
        west = LiquidBranchTrace(1.9e-3, 0.0, 0.0, 28.0, 998.0)
        riser = LiquidBranchTrace(5.3e-4, 0.0, 0.0, 28.0, 998.0)
        gas = GasTrace(110_000.0, 1.30, 0.0, 340.0)
        base = solve_first_bottom_gas_entry(
            west,
            west,
            riser,
            gas,
            liquid_pressure_reference_abs_Pa=101_325.0,
            available_gas_open_area_m2=1.0e-4,
        )
        shift = 137.0
        shifted_west = LiquidBranchTrace(
            west.area_m2,
            west.outward_velocity_m_s,
            west.gauge_pressure_Pa - shift,
            west.wave_speed_m_s,
            west.density_kg_m3,
        )
        shifted_riser = LiquidBranchTrace(
            riser.area_m2,
            riser.outward_velocity_m_s,
            riser.gauge_pressure_Pa - shift,
            riser.wave_speed_m_s,
            riser.density_kg_m3,
        )
        shifted = solve_first_bottom_gas_entry(
            shifted_west,
            shifted_west,
            shifted_riser,
            gas,
            liquid_pressure_reference_abs_Pa=101_325.0 + shift,
            available_gas_open_area_m2=1.0e-4,
        )

        self.assertEqual(
            shifted.common_pressure_abs_Pa,
            base.common_pressure_abs_Pa,
        )
        self.assertEqual(
            shifted.gas.volume_flow_to_riser_m3_s,
            base.gas.volume_flow_to_riser_m3_s,
        )
        self.assertAlmostEqual(
            shifted.liquid.riser_outward_flow_m3_s,
            base.liquid.riser_outward_flow_m3_s,
            places=18,
        )

    def test_zero_available_gas_area_is_exact_original_liquid_path(self) -> None:
        west = LiquidBranchTrace(1.9e-3, 0.01, 5_000.0, 28.0, 998.0)
        riser = LiquidBranchTrace(5.3e-4, -0.02, 6_000.0, 28.0, 998.0)
        closed = solve_liquid_tee(west, west, riser)
        solution = solve_first_bottom_gas_entry(
            west,
            west,
            riser,
            GasTrace(120_000.0, 1.4, 0.0, 340.0),
            liquid_pressure_reference_abs_Pa=101_325.0,
            available_gas_open_area_m2=0.0,
            closed_liquid_solution=closed,
        )

        self.assertFalse(solution.active)
        self.assertIs(solution.liquid, closed)
        self.assertEqual(solution.gas.volume_flow_to_riser_m3_s, 0.0)


if __name__ == "__main__":
    unittest.main()
