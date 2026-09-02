"""Conservation tests for the isolated Case-A two-stream network adapter."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_twostream_network_adapter import (  # noqa: E402
    COMPLETE_CASEA_NETWORK_READY,
    GLOBAL_INTEGRATION_BLOCKERS,
    TWOSTREAM_NETWORK_ADAPTER_READY,
    TwoStreamNetworkAdapterError,
    advance_casea_twostream_riser_from_finite_node,
)
from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_twostream_fv import (  # noqa: E402
    DirectionalBoundaryFlux,
    PhysicalGasInterphaseState,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    hydrostatic_face_pressures,
)


def _transaction(q_net: float, *, dt: float, node_liquid_volume: float):
    """Small duck-typed finite-node transaction with the public contract."""

    result = SimpleNamespace(
        vertical=SimpleNamespace(liquid_area=float(q_net)),
        ledger=SimpleNamespace(
            dt=float(dt),
            initial_state=SimpleNamespace(
                liquid_equivalent_volume=float(node_liquid_volume)
            ),
        ),
    )
    return SimpleNamespace(q_net=float(q_net), result=result)


class CaseATwoStreamNetworkAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = VerticalTwoStreamParameters(
            cell_count=3,
            cell_length=0.05,
            diameter=0.0571,
            liquid_density=998.0,
            gravity=9.81,
            wall_friction_up=0.02,
            wall_friction_down=0.02,
            interstream_drag=0.0,
        )
        self.geometry = VerticalMouthGeometry(0.0571)
        self.total_area = 1.50e-3
        self.state = VerticalTwoStreamState.from_iterables(
            upward_area=[0.76e-3] * 3,
            upward_discharge=[0.20e-3] * 3,
            downward_area=[0.74e-3] * 3,
            downward_discharge=[-0.19e-3] * 3,
        )
        self.phase = VerticalMouthPhaseState(
            liquid_area=self.total_area,
            liquid_velocity=0.01e-3 / self.total_area,
            gas_area=self.geometry.full_area - self.total_area,
            gas_velocity=0.50,
        )

    def _advance(self, *, physical_gas=None):
        transaction = _transaction(
            0.01e-3,
            dt=1.0e-4,
            node_liquid_volume=2.0e-3,
        )
        pressure = hydrostatic_face_pressures(
            self.parameters,
            bottom_pressure=120_000.0,
        )
        return advance_casea_twostream_riser_from_finite_node(
            self.state,
            transaction,
            self.parameters,
            pressure_faces=pressure,
            phase=self.phase,
            geometry=self.geometry,
            material=VerticalMouthMaterialProperties(
                liquid_density=998.0,
                gas_density=1.2,
                liquid_dynamic_viscosity=1.0e-3,
            ),
            wallis=WallisCounterCurrentParameters(constant=0.50),
            losses=DirectionalMouthLosses(
                upward_turn=0.75,
                downward_turn=0.75,
                countercurrent_mixing=1.0,
            ),
            physical_gas=physical_gas,
            top_boundary=DirectionalBoundaryFlux(),
        )

    def test_gross_countercurrent_flux_preserves_node_owned_qnet(self) -> None:
        result = self._advance()
        self.assertGreater(result.gross_upward_rate, 0.0)
        self.assertGreater(result.gross_downward_rate, 0.0)
        self.assertAlmostEqual(
            result.gross_upward_rate - result.gross_downward_rate,
            result.q_net_owner_value,
            places=15,
        )
        self.assertAlmostEqual(result.q_net_residual, 0.0, places=15)
        self.assertAlmostEqual(result.riser_volume_residual, 0.0, places=15)
        self.assertTrue(result.global_branch_commit_pending)

    def test_physical_three_body_drag_closes_its_momentum_ledger(self) -> None:
        gas_area = self.geometry.full_area - self.total_area - 1.0e-5
        gas_mass = 1.2 * gas_area * self.parameters.cell_length
        gas = PhysicalGasInterphaseState.from_iterables(
            gas_mass=[gas_mass] * 3,
            gas_momentum=[gas_mass * 0.50] * 3,
            gas_area=[gas_area] * 3,
            upward_interface_perimeter=[0.04] * 3,
            downward_interface_perimeter=[0.08] * 3,
            upward_hydraulic_diameter=[0.02] * 3,
            downward_hydraulic_diameter=[0.01] * 3,
        )

        result = self._advance(physical_gas=gas)

        self.assertIsNotNone(result.physical_drag)
        assert result.physical_drag is not None
        self.assertAlmostEqual(
            result.physical_drag.total_momentum_residual,
            0.0,
            places=14,
        )
        self.assertAlmostEqual(result.riser_volume_residual, 0.0, places=15)

    def test_adapter_rejects_a_mouth_state_from_another_riser_cell(self) -> None:
        wrong = VerticalMouthPhaseState(
            liquid_area=self.total_area * 0.80,
            liquid_velocity=0.0,
            gas_area=self.geometry.full_area - self.total_area * 0.80,
            gas_velocity=0.50,
        )
        transaction = _transaction(0.0, dt=1.0e-4, node_liquid_volume=2.0e-3)
        with self.assertRaises(TwoStreamNetworkAdapterError):
            advance_casea_twostream_riser_from_finite_node(
                self.state,
                transaction,
                self.parameters,
                pressure_faces=hydrostatic_face_pressures(
                    self.parameters,
                    bottom_pressure=120_000.0,
                ),
                phase=wrong,
                geometry=self.geometry,
                material=VerticalMouthMaterialProperties(998.0, 1.2, 1.0e-3),
                wallis=WallisCounterCurrentParameters(0.50),
                losses=DirectionalMouthLosses(0.0, 0.0, 0.0),
            )

    def test_scope_and_missing_global_ownership_are_explicit(self) -> None:
        self.assertTrue(TWOSTREAM_NETWORK_ADAPTER_READY)
        self.assertFalse(COMPLETE_CASEA_NETWORK_READY)
        self.assertIn(
            "replace_legacy_G1_taylor_ccfl_and_distributed_side_source_owners",
            GLOBAL_INTEGRATION_BLOCKERS,
        )
        signature = inspect.signature(
            advance_casea_twostream_riser_from_finite_node
        )
        self.assertNotIn("target_time", signature.parameters)
        self.assertNotIn("target_height", signature.parameters)
        self.assertNotIn("target_volume", signature.parameters)


if __name__ == "__main__":
    unittest.main()
