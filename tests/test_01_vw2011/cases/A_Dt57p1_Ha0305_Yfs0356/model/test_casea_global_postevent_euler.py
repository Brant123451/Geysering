"""Conservation and ownership tests for the global Case-A Euler stage."""

from __future__ import annotations

import inspect
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
    CompressiblePostLaunchParameters,
)
from casea_global_postevent_euler import (  # noqa: E402
    GLOBAL_FIRST_ORDER_EULER_READY,
    MAIN_LOOP_INTEGRATED,
    GlobalPostEventEulerError,
    HorizontalBoundaryCellOperator,
    StratifiedStateRate,
    VerticalGasStageOperator,
    VerticalGasState,
    advance_casea_global_postevent_euler,
)
from casea_material_front_cutcell import StratifiedState  # noqa: E402
from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    VerticalMouthGeometry,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    DuplicateMouthFluxOwner,
    LegacyMouthPathActivity,
)
from casea_vertical_twostream_fv import (  # noqa: E402
    DirectionalBoundaryFlux,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
)


P_ATM = 101_325.0
RHO_L = 998.0
R_GAS = 287.05
T_GAS = 293.0
C_GAS = math.sqrt(R_GAS * T_GAS)
DIAMETER = 0.0571
FULL_AREA = math.pi * DIAMETER**2 / 4.0
LIQUID_AREA = 1.20e-3
GAS_AREA = FULL_AREA - LIQUID_AREA
RHO_G = P_ATM / C_GAS**2
DX = 0.04
DZ = 0.05
DT = 1.0e-5


def _node_parameters() -> CompressiblePostLaunchParameters:
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
        maximum_cfl=0.8,
    )


def _node_state():
    parameters = _node_parameters().node
    gas_volume = 2.0e-3
    gas_mass = P_ATM * gas_volume / C_GAS**2
    return state_from_pressure_and_gas_mass(
        pressure_abs=P_ATM,
        gas_mass=gas_mass,
        node_total_volume=5.0e-3,
        params=parameters,
    )


def _horizontal_state(*, liquid_discharge: float = 0.0) -> StratifiedState:
    return StratifiedState(
        gas_mass=RHO_G * GAS_AREA,
        gas_momentum=0.0,
        liquid_area=LIQUID_AREA,
        liquid_discharge=liquid_discharge,
    )


def _vertical_liquid_state() -> VerticalTwoStreamState:
    return VerticalTwoStreamState.from_iterables(
        upward_area=[0.61e-3] * 3,
        upward_discharge=[0.0] * 3,
        downward_area=[0.59e-3] * 3,
        downward_discharge=[0.0] * 3,
    )


def _vertical_gas_state(*, velocity: float = 0.0) -> VerticalGasState:
    mass = RHO_G * GAS_AREA
    return VerticalGasState.from_iterables(
        mass_per_length=[mass] * 3,
        momentum_per_length=[mass * velocity] * 3,
    )


def _horizontal_operator(sign: int, *, rhs=StratifiedStateRate()):
    return HorizontalBoundaryCellOperator(
        cell_length=DX,
        full_area=FULL_AREA,
        outward_axis_sign=sign,
        liquid_wave_speed=28.0,
        liquid_pressure_potential=0.0,
        non_node_rhs=rhs,
    )


def _vertical_parameters() -> VerticalTwoStreamParameters:
    # Gravity is zero only in this exact uniform-state unit test.  The mouth
    # constitutive law below still uses the measured positive gravity.
    return VerticalTwoStreamParameters(
        cell_count=3,
        cell_length=DZ,
        diameter=DIAMETER,
        liquid_density=RHO_L,
        gravity=0.0,
        wall_friction_up=0.0,
        wall_friction_down=0.0,
        interstream_drag=0.0,
    )


def _vertical_operator(*, drag: bool = False) -> VerticalGasStageOperator:
    return VerticalGasStageOperator.from_iterables(
        non_node_mass_rhs=[0.0] * 3,
        non_node_momentum_rhs=[0.0] * 3,
        liquid_wave_speed=28.0,
        liquid_pressure_potential=0.0,
        atmospheric_top_pressure_abs=P_ATM,
        apply_physical_interphase_drag=drag,
    )


def _advance(
    *,
    west_state=None,
    east_state=None,
    vertical_gas=None,
    node_state=None,
    two_stream=None,
    west_operator=None,
    east_operator=None,
    vertical_operator=None,
    legacy=LegacyMouthPathActivity(),
):
    return advance_casea_global_postevent_euler(
        dt=DT,
        west_state=_horizontal_state() if west_state is None else west_state,
        east_state=_horizontal_state() if east_state is None else east_state,
        vertical_gas_state=(
            _vertical_gas_state() if vertical_gas is None else vertical_gas
        ),
        node_state=_node_state() if node_state is None else node_state,
        two_stream_state=(
            _vertical_liquid_state() if two_stream is None else two_stream
        ),
        west_operator=(
            _horizontal_operator(-1) if west_operator is None else west_operator
        ),
        east_operator=(
            _horizontal_operator(+1) if east_operator is None else east_operator
        ),
        vertical_gas_operator=(
            _vertical_operator() if vertical_operator is None else vertical_operator
        ),
        node_parameters=_node_parameters(),
        two_stream_parameters=_vertical_parameters(),
        mouth_geometry=VerticalMouthGeometry(DIAMETER),
        liquid_dynamic_viscosity=1.0e-3,
        wallis=WallisCounterCurrentParameters(constant=0.50),
        mouth_losses=DirectionalMouthLosses(
            upward_turn=0.75,
            downward_turn=0.75,
            countercurrent_mixing=1.0,
        ),
        top_liquid_boundary=DirectionalBoundaryFlux(),
        legacy_activity=legacy,
    )


class CaseAGlobalPostEventEulerTests(unittest.TestCase):
    def assert_ledgers_close(self, result) -> None:
        self.assertAlmostEqual(result.ledger.gas_mass_residual, 0.0, places=14)
        self.assertAlmostEqual(result.ledger.liquid_volume_residual, 0.0, places=14)
        self.assertAlmostEqual(result.ledger.node_t_gas_residual, 0.0, places=14)
        self.assertAlmostEqual(result.ledger.node_t_liquid_residual, 0.0, places=14)
        self.assertAlmostEqual(result.ledger.mouth_q_net_residual, 0.0, places=14)
        self.assertAlmostEqual(
            result.ledger.vertical_upward_volume_residual, 0.0, places=14
        )
        self.assertAlmostEqual(
            result.ledger.vertical_downward_volume_residual, 0.0, places=14
        )
        self.assertAlmostEqual(
            result.ledger.three_body_momentum_residual, 0.0, places=14
        )

    def test_uniform_state_is_preserved_and_all_face_keys_are_committed(self) -> None:
        result = _advance()
        self.assert_ledgers_close(result)
        reference = _horizontal_state()
        self.assertAlmostEqual(result.state.west.gas_mass, reference.gas_mass, places=15)
        self.assertAlmostEqual(result.state.west.liquid_area, reference.liquid_area, places=15)
        self.assertAlmostEqual(result.state.west.gas_momentum, 0.0, places=15)
        self.assertAlmostEqual(result.state.west.liquid_discharge, 0.0, places=15)
        self.assertAlmostEqual(result.state.east.gas_mass, reference.gas_mass, places=15)
        self.assertAlmostEqual(result.state.east.liquid_area, reference.liquid_area, places=15)
        self.assertAlmostEqual(result.state.east.gas_momentum, 0.0, places=15)
        self.assertAlmostEqual(result.state.east.liquid_discharge, 0.0, places=15)
        vertical_reference = _vertical_liquid_state()
        for actual, expected in zip(
            result.state.vertical_liquid.upward_area,
            vertical_reference.upward_area,
        ):
            self.assertAlmostEqual(actual, expected, places=15)
        for actual, expected in zip(
            result.state.vertical_liquid.downward_area,
            vertical_reference.downward_area,
        ):
            self.assertAlmostEqual(actual, expected, places=15)
        for value in result.state.vertical_liquid.upward_discharge:
            self.assertAlmostEqual(value, 0.0, places=15)
        for value in result.state.vertical_liquid.downward_discharge:
            self.assertAlmostEqual(value, 0.0, places=15)
        self.assertEqual(len(result.ledger.committed_keys), 12)
        self.assertEqual(
            set(result.node_outward_fluxes), {"west", "east", "vertical"}
        )

    def test_natural_countercurrent_exchange_keeps_the_node_owned_net_rate(self) -> None:
        result = _advance(vertical_gas=_vertical_gas_state(velocity=0.50))
        self.assert_ledgers_close(result)
        self.assertGreater(result.mouth.exchange.upward_flow, 0.0)
        self.assertGreater(result.mouth.exchange.downward_flow, 0.0)
        self.assertAlmostEqual(
            result.mouth.exchange.upward_flow
            - result.mouth.exchange.downward_flow,
            result.node_stage.vertical.liquid_area,
            places=15,
        )
        # Equal and opposite gross exchange changes the directional fields,
        # not the combined liquid inventory.
        self.assertNotEqual(
            result.state.vertical_liquid.upward_area,
            _vertical_liquid_state().upward_area,
        )

    def test_physical_three_body_drag_returns_the_gas_reaction(self) -> None:
        initial_liquid = VerticalTwoStreamState.from_iterables(
            upward_area=[0.61e-3] * 3,
            upward_discharge=[0.08e-3] * 3,
            downward_area=[0.59e-3] * 3,
            downward_discharge=[-0.06e-3] * 3,
        )
        result = _advance(
            vertical_gas=_vertical_gas_state(velocity=0.50),
            two_stream=initial_liquid,
            vertical_operator=_vertical_operator(drag=True),
        )
        self.assertIsNotNone(result.physical_drag)
        self.assert_ledgers_close(result)
        assert result.physical_drag is not None
        self.assertAlmostEqual(
            result.physical_drag.total_momentum_residual,
            0.0,
            places=14,
        )

    def test_second_global_call_reconstructs_predictor_branch_traces(self) -> None:
        first = _advance(west_state=_horizontal_state(liquid_discharge=1.0e-5))
        self.assert_ledgers_close(first)
        second = _advance(
            west_state=first.state.west,
            east_state=first.state.east,
            vertical_gas=first.state.vertical_gas,
            node_state=first.state.node,
            two_stream=first.state.vertical_liquid,
        )
        self.assert_ledgers_close(second)
        self.assertNotAlmostEqual(
            first.node_stage.vertical.liquid_area,
            second.node_stage.vertical.liquid_area,
            places=16,
        )
        self.assertNotEqual(
            first.node_stage.pressure_abs,
            second.node_stage.pressure_abs,
        )

    def test_external_non_t_residual_is_counted_once(self) -> None:
        rhs = StratifiedStateRate(gas_mass=2.0e-7, liquid_area=3.0e-7)
        result = _advance(west_operator=_horizontal_operator(-1, rhs=rhs))
        self.assert_ledgers_close(result)
        self.assertAlmostEqual(
            result.ledger.external_gas_mass_change,
            DT * rhs.gas_mass * DX,
            places=18,
        )
        self.assertAlmostEqual(
            result.ledger.external_liquid_volume_change,
            DT * rhs.liquid_area * DX,
            places=18,
        )

    def test_duplicate_owner_fails_before_any_state_can_be_committed(self) -> None:
        west = _horizontal_state()
        with self.assertRaises(DuplicateMouthFluxOwner):
            _advance(
                west_state=west,
                legacy=LegacyMouthPathActivity(
                    taylor_return_mass_flux_applied=True
                ),
            )
        self.assertEqual(west, _horizontal_state())

    def test_vertical_rhs_cannot_hide_a_second_t_or_drag_owner(self) -> None:
        duplicate_t = VerticalGasStageOperator.from_iterables(
            non_node_mass_rhs=[0.0] * 3,
            non_node_momentum_rhs=[0.0] * 3,
            liquid_wave_speed=28.0,
            liquid_pressure_potential=0.0,
            contains_t_face_flux=True,
        )
        with self.assertRaises(GlobalPostEventEulerError):
            _advance(vertical_operator=duplicate_t)

        duplicate_horizontal = HorizontalBoundaryCellOperator(
            cell_length=DX,
            full_area=FULL_AREA,
            outward_axis_sign=-1,
            liquid_wave_speed=28.0,
            liquid_pressure_potential=0.0,
            contains_distributed_side_source=True,
        )
        with self.assertRaises(GlobalPostEventEulerError):
            _advance(west_operator=duplicate_horizontal)

    def test_scope_has_no_result_or_time_target_controls(self) -> None:
        self.assertTrue(GLOBAL_FIRST_ORDER_EULER_READY)
        self.assertFalse(MAIN_LOOP_INTEGRATED)
        signature = inspect.signature(advance_casea_global_postevent_euler)
        forbidden = {
            "time",
            "target_time",
            "target_height",
            "target_volume",
            "openfoam",
        }
        self.assertTrue(forbidden.isdisjoint(signature.parameters))


if __name__ == "__main__":
    unittest.main()
