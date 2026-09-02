"""Acceptance tests for the experimental finite-node q_net transaction."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    state_from_pressure_and_gas_mass,
)
from casea_compressible_node_postlaunch_stage import (  # noqa: E402
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchParameters,
)
from casea_finite_node_qnet_owner import (  # noqa: E402
    DuplicateFiniteNodeCommit,
    IncompleteFiniteNodeCommit,
    advance_finite_node_qnet_owner,
    required_commit_keys,
    verify_atomic_branch_commit,
)
from casea_material_front_cutcell import StratifiedState  # noqa: E402
from casea_tjunction_shock_network import LiquidCharacteristic  # noqa: E402
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    DuplicateMouthFluxOwner,
    LegacyMouthPathActivity,
)


P_ATM = 101_325.0
RHO_L = 998.0
R_GAS = 287.05
T_GAS = 293.0
C_GAS = math.sqrt(R_GAS * T_GAS)
FULL_AREA = 0.010
LIQUID_AREA = 0.005
NODE_VOLUME = 0.10


def _parameters() -> CompressiblePostLaunchParameters:
    return CompressiblePostLaunchParameters(
        node=CompressibleFiniteNodeParameters(
            gas_sound_speed=C_GAS,
            liquid_density=RHO_L,
            liquid_wave_speed=28.0,
            reference_pressure_abs=P_ATM,
        ),
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure_abs=P_ATM,
    )


def _node_state():
    params = _parameters().node
    gas_volume = 0.04
    gas_mass = P_ATM * gas_volume / C_GAS**2
    return state_from_pressure_and_gas_mass(
        pressure_abs=P_ATM,
        gas_mass=gas_mass,
        node_total_volume=NODE_VOLUME,
        params=params,
    )


def _branch(*, liquid_velocity: float = 0.0) -> CompressibleNodeResolvedBranch:
    gas_area = FULL_AREA - LIQUID_AREA
    gas_density = P_ATM / C_GAS**2
    return CompressibleNodeResolvedBranch(
        resolved=StratifiedState(
            gas_mass=gas_density * gas_area,
            gas_momentum=0.0,
            liquid_area=LIQUID_AREA,
            liquid_discharge=LIQUID_AREA * liquid_velocity,
        ),
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=P_ATM,
            reference_outward_velocity=liquid_velocity,
            wave_speed=28.0,
        ),
        liquid_face_area=LIQUID_AREA,
        full_area=FULL_AREA,
        reference_liquid_face_pressure_abs=P_ATM,
        reference_liquid_pressure_potential=0.0,
    )


class FiniteNodeQnetOwnerTests(unittest.TestCase):
    def test_uniform_state_maps_all_three_shared_faces(self) -> None:
        branch = _branch()
        transaction = advance_finite_node_qnet_owner(
            _node_state(),
            dt=1.0e-4,
            west=branch,
            east=branch,
            vertical=branch,
            params=_parameters(),
        )
        self.assertAlmostEqual(transaction.q_net, 0.0, places=15)
        self.assertEqual(set(transaction.outward), {"west", "east", "vertical"})
        self.assertEqual(
            transaction.global_coordinate["west"].liquid_volume,
            -transaction.outward["west"].liquid_volume,
        )
        self.assertEqual(
            transaction.global_coordinate["west"].liquid_momentum,
            transaction.outward["west"].liquid_momentum,
        )
        verify_atomic_branch_commit(sorted(required_commit_keys()))

    def test_vertical_qnet_comes_only_from_finite_node(self) -> None:
        rest = _branch()
        transaction = advance_finite_node_qnet_owner(
            _node_state(),
            dt=1.0e-4,
            west=_branch(liquid_velocity=-0.03),
            east=rest,
            vertical=rest,
            params=_parameters(),
        )
        self.assertEqual(
            transaction.q_net,
            transaction.result.vertical.liquid_area,
        )
        self.assertAlmostEqual(transaction.liquid_inventory_residual, 0.0, places=15)
        self.assertAlmostEqual(transaction.gas_inventory_residual, 0.0, places=15)

    def test_any_legacy_mass_path_is_rejected_before_node_advance(self) -> None:
        branch = _branch()
        with self.assertRaises(DuplicateMouthFluxOwner):
            advance_finite_node_qnet_owner(
                _node_state(),
                dt=1.0e-4,
                west=branch,
                east=branch,
                vertical=branch,
                params=_parameters(),
                legacy_activity=LegacyMouthPathActivity(
                    net_only_horizontal_side_source_applied=True
                ),
            )

    def test_incomplete_or_duplicate_network_commit_fails_closed(self) -> None:
        keys = sorted(required_commit_keys())
        with self.assertRaises(IncompleteFiniteNodeCommit):
            verify_atomic_branch_commit(keys[:-1])
        with self.assertRaises(DuplicateFiniteNodeCommit):
            verify_atomic_branch_commit([*keys, keys[0]])


if __name__ == "__main__":
    unittest.main()
