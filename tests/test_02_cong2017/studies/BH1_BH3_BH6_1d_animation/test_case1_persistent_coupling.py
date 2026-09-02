from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal
from campaign2_shared_contract import (
    QUALIFICATION_CASES,
    shared_solver_signature,
    solver_contract,
)
from case1_persistent_coupling import (
    PersistentHorizontalOwner,
    TeeTransaction,
    transaction_from_tee_solutions,
)
from campaign2_tee_riemann import (
    GasTrace,
    LiquidBranchTrace,
    solve_gas_tee,
    solve_liquid_tee,
)


def build_horizontal() -> Campaign2Case1MirroredHorizontal:
    return Campaign2Case1MirroredHorizontal(
        length=6.59,
        diameter=0.05,
        physical_valve_x=5.98,
        physical_riser_x=3.47,
        initial_water_head_from_invert=0.66,
        dx=0.02,
        wave_speed=28.0,
        valve_open_time=0.20,
        gas_temperature=296.15,
    )


class PersistentCase1CouplingTests(unittest.TestCase):
    def test_three_case_solver_contracts_only_vary_riser_diameter(self) -> None:
        signatures = [
            shared_solver_signature(case) for case in QUALIFICATION_CASES
        ]
        self.assertTrue(all(item == signatures[0] for item in signatures[1:]))
        contracts = [solver_contract(case) for case in QUALIFICATION_CASES]
        diameters = {
            item["case"]["riser_diameter_m"] for item in contracts
        }
        self.assertEqual(diameters, {0.016, 0.026, 0.041})
        for item in contracts:
            self.assertNotIn("experiment_geyser", item)

    def test_liquid_tee_transaction_is_equal_and_opposite(self) -> None:
        horizontal = build_horizontal()
        before = horizontal.initial_state()
        west_flow = 1.0e-6
        east_flow = 0.25e-6
        dt = 1.0e-3
        riser_gain = (west_flow - east_flow) * dt

        after = horizontal.apply_physical_junction_liquid_fluxes(
            before,
            west_flow=west_flow,
            east_flow=east_flow,
            dt=dt,
        )
        horizontal_change = float(
            np.sum(after.area - before.area) * horizontal.dx
        )
        self.assertAlmostEqual(
            horizontal_change + riser_gain,
            0.0,
            places=15,
        )

    def test_open_polytropic_mass_transaction_updates_pressure(self) -> None:
        horizontal = build_horizontal()
        state = horizontal.initial_state()
        ratio = 0.90
        gas = horizontal._with_open_polytropic_mass(
            state.gas,
            ratio * state.gas.mass,
        )
        self.assertAlmostEqual(gas.mass / state.gas.mass, ratio, places=14)
        self.assertAlmostEqual(
            gas.pressure_abs / state.gas.pressure_abs,
            ratio ** state.gas.gamma,
            places=14,
        )
        self.assertEqual(gas.volume, state.gas.volume)

    def test_gas_cannot_cross_tee_before_material_arrival(self) -> None:
        horizontal = build_horizontal()
        state = horizontal.initial_state()
        with self.assertRaisesRegex(ValueError, "material pocket"):
            horizontal.apply_physical_junction_gas_mass_flux(
                state,
                mass_flow_to_riser=1.0e-6,
                dt=1.0e-3,
            )

    def test_gas_transaction_conserves_horizontal_plus_riser_mass(self) -> None:
        horizontal = build_horizontal()
        state = horizontal.initial_state()
        # Place only the material-interface metadata at the discrete T face;
        # the thermodynamic transaction itself leaves volume/geometry fixed.
        mirrored_t = horizontal.physical_length - horizontal.physical_junction_face_x
        state = replace(state, interface_x=mirrored_t, vented=True)
        mass_flow = 2.0e-6
        dt = 2.5e-3
        riser_gain = mass_flow * dt
        after = horizontal.apply_physical_junction_gas_mass_flux(
            state,
            mass_flow_to_riser=mass_flow,
            dt=dt,
        )
        horizontal_change = after.gas.mass - state.gas.mass
        self.assertTrue(math.isfinite(after.air_pressure_abs))
        self.assertAlmostEqual(
            horizontal_change + riser_gain,
            0.0,
            places=15,
        )

    def test_persistent_owner_commits_one_shared_tee_transaction(self) -> None:
        horizontal = build_horizontal()
        owner = PersistentHorizontalOwner.initialize(horizontal)
        mirrored_t = horizontal.physical_length - horizontal.physical_junction_face_x
        owner.state = replace(owner.state, interface_x=mirrored_t, vented=True)
        liquid_before = float(np.sum(owner.state.area) * horizontal.dx)
        gas_before = float(owner.state.gas.mass)
        dt = 2.0e-3
        riser_area = 1.0e-3
        transaction = TeeTransaction(
            west_liquid_flow_m3_s=1.0e-6,
            east_liquid_flow_m3_s=0.4e-6,
            gas_mass_flow_to_riser_kg_s=1.5e-6,
            riser_mouth_area_m2=riser_area,
            gas_open_area_m2=0.5 * riser_area,
            liquid_open_area_m2=0.5 * riser_area,
            blocked_riser_area_m2=0.0,
            gas_volume_flow_to_riser_m3_s=1.25e-6,
            gas_normal_momentum_flow_N=2.0e-6,
            liquid_normal_momentum_flow_N=3.0e-7,
        )
        vertical = owner.commit_tee(transaction, dt)
        liquid_after = float(np.sum(owner.state.area) * horizontal.dx)
        gas_after = float(owner.state.gas.mass)

        self.assertAlmostEqual(
            liquid_after - liquid_before + vertical.liquid_volume_m3,
            0.0,
            places=15,
        )
        self.assertAlmostEqual(
            gas_after - gas_before + vertical.gas_mass_kg,
            0.0,
            places=15,
        )
        self.assertTrue(owner.horizontal_owner_active)
        self.assertEqual(owner.tee_transaction_count, 1)
        self.assertEqual(vertical.gas_volume_m3, 1.25e-6 * dt)
        self.assertEqual(
            vertical.liquid_normal_momentum_kg_m_s,
            3.0e-7 * dt,
        )
        state_identity = id(owner.state)
        owner.physical_snapshot()
        self.assertEqual(id(owner.state), state_identity)

    def test_riemann_solutions_form_the_same_committed_transaction(self) -> None:
        horizontal = LiquidBranchTrace(1.9e-3, 0.0, 9000.0, 28.0)
        riser_liquid = LiquidBranchTrace(5.3e-4, 0.0, 2000.0, 28.0)
        liquid = solve_liquid_tee(horizontal, horizontal, riser_liquid)
        gas = solve_gas_tee(
            GasTrace(108000.0, 1.27, 0.0, 345.0),
            GasTrace(101325.0, 1.19, 0.0, 345.0),
            open_area_m2=5.0e-4,
        )
        transaction = transaction_from_tee_solutions(
            liquid,
            gas,
            physical_riser_area_m2=(
                float(liquid.riser_open_area_m2)
                + float(gas.open_area_m2)
            ),
        )
        self.assertAlmostEqual(
            transaction.liquid_flow_to_riser_m3_s,
            liquid.riser_outward_flow_m3_s,
            places=15,
        )
        self.assertEqual(
            transaction.gas_mass_flow_to_riser_kg_s,
            gas.mass_flow_to_riser_kg_s,
        )
        self.assertEqual(
            transaction.gas_volume_flow_to_riser_m3_s,
            gas.volume_flow_to_riser_m3_s,
        )
        self.assertEqual(
            transaction.liquid_normal_momentum_flow_N,
            liquid.normal_momentum_to_riser_N,
        )
        self.assertEqual(
            transaction.liquid_node_gauge_pressure_Pa,
            liquid.node_gauge_pressure_Pa,
        )
        self.assertEqual(
            transaction.gas_interface_pressure_abs_Pa,
            gas.interface_pressure_abs_Pa,
        )

    def test_persistent_advance_commits_constant_head_boundary(self) -> None:
        horizontal = build_horizontal()
        owner = PersistentHorizontalOwner.initialize(
            horizontal,
            reservoir_head_from_invert_m=0.66,
        )
        commit = owner.advance(2.0e-4)
        self.assertIsNotNone(commit)
        assert commit is not None
        self.assertEqual(commit.mass_balance_residual_m3, 0.0)
        self.assertAlmostEqual(
            owner.cumulative_reservoir_liquid_inflow_m3,
            commit.liquid_volume_to_horizontal_m3,
            places=18,
        )
        self.assertTrue(owner.horizontal_owner_active)


if __name__ == "__main__":
    unittest.main()
