"""Conservation tests for the independent Case-A EOS/front outer closure."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_tjunction_eos_front_coupler import (  # noqa: E402
    AtmosphericTopFace,
    EOSFrontParameters,
    EOSFrontState,
    IncompatiblePrescribedPressure,
    ResolvedGasVolumes,
    advance_eos_front_coupler,
    evaluate_eos_front_candidate_at_pressure,
)
from casea_tjunction_shock_network import (  # noqa: E402
    BranchGeometry,
    LiquidCharacteristic,
    MovingFrontState,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
)


class CaseATJunctionEOSFrontCouplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.east = BranchGeometry(0.094, 0.490, 28.0, 0.0)
        # Exact stationary tests use a horizontal gravitational source.  The
        # production vertical branch passes bed_slope=1 through the same API.
        self.vertical = BranchGeometry(0.0571, 0.610, 28.0, 0.0)
        self.params = EOSFrontParameters(
            east=self.east,
            vertical=self.vertical,
        )
        self.areas = ZeroStorageTBranchAreas(
            west=self.east.section(self.params.gravity).full_area,
            east=self.east.section(self.params.gravity).full_area,
            vertical=self.vertical.section(self.params.gravity).full_area,
        )
        self.closed_top = AtmosphericTopFace(False, self.areas.vertical, 0.0)
        self.open_top = AtmosphericTopFace(True, self.areas.vertical, 0.0)

    def _stationary_front(
        self,
        geometry: BranchGeometry,
        *,
        depth: float,
        gas_head: float,
        position: float,
    ) -> MovingFrontState:
        section = geometry.section(self.params.gravity)
        pressure_moment_head = (
            float(section.hydrostatic_moment(depth)) / section.full_area
        )
        return MovingFrontState(
            position=position,
            free_surface_depth=depth,
            pressurised_head_foot=(
                gas_head + 0.5 * geometry.diameter + pressure_moment_head
            ),
            pressurised_velocity_foot=0.0,
        )

    def _fixture(
        self,
        *,
        gas_head: float = 0.20,
    ) -> tuple[EOSFrontState, TeeLiquidCharacteristics, float]:
        pressure = (
            self.params.atmospheric_pressure
            + self.params.liquid_density * self.params.gravity * gas_head
        )
        east_front = self._stationary_front(
            self.east, depth=0.070, gas_head=gas_head, position=0.080
        )
        vertical_front = self._stationary_front(
            self.vertical, depth=0.040, gas_head=gas_head, position=0.120
        )
        east_gas_area = (
            self.east.section(self.params.gravity).full_area
            - float(
                self.east.section(self.params.gravity).area_from_depth(
                    east_front.free_surface_depth
                )
            )
        )
        vertical_gas_area = (
            self.vertical.section(self.params.gravity).full_area
            - float(
                self.vertical.section(self.params.gravity).area_from_depth(
                    vertical_front.free_surface_depth
                )
            )
        )
        volumes = ResolvedGasVolumes(
            west=2.5e-3,
            tee_void=1.5e-4,
            tee_control_volume=3.5e-4,
            east=east_gas_area * east_front.position,
            vertical=vertical_gas_area * vertical_front.position,
        )
        gas_mass = (
            pressure
            * volumes.total
            / (self.params.gas_constant * self.params.gas_temperature)
        )
        state = EOSFrontState(
            time=6.5,
            connected_gas_mass=gas_mass,
            volumes=volumes,
            east_front=east_front,
            vertical_front=vertical_front,
        )
        characteristic = LiquidCharacteristic(pressure, 0.0, 28.0)
        characteristics = TeeLiquidCharacteristics(
            west=characteristic,
            east=characteristic,
            vertical=characteristic,
            west_liquid_area=self.areas.west,
        )
        return state, characteristics, pressure

    def test_hydrostatic_rest_returns_complete_well_balanced_face_fluxes(self) -> None:
        state, characteristics, pressure = self._fixture()
        result = advance_eos_front_coupler(
            state,
            self.params,
            characteristics,
            self.areas,
            self.closed_top,
            dt=1.0e-3,
        )
        self.assertAlmostEqual(result.node_pressure_abs, pressure, delta=1.0e-5)
        self.assertAlmostEqual(
            result.east_solution.interface_speed, 0.0, delta=1.0e-9
        )
        self.assertAlmostEqual(
            result.vertical_solution.interface_speed, 0.0, delta=1.0e-9
        )
        for name, flux in result.liquid_branch_face_fluxes.items():
            self.assertAlmostEqual(flux.mass_flux, 0.0, delta=1.0e-11)
            self.assertAlmostEqual(
                flux.total_momentum_flux,
                pressure * getattr(self.areas, name),
                delta=1.0e-8,
            )
            self.assertAlmostEqual(
                flux.momentum_flux_increment, 0.0, delta=1.0e-8
            )

    def test_closed_network_preserves_connected_mass_and_gas_ledger(self) -> None:
        state, characteristics, _ = self._fixture()
        result = advance_eos_front_coupler(
            state,
            self.params,
            characteristics,
            self.areas,
            self.closed_top,
            dt=1.0e-3,
        )
        self.assertEqual(result.top_gas_flux.mass_rate, 0.0)
        self.assertEqual(result.top_gas_flux.momentum_flux, 0.0)
        self.assertEqual(result.top_mass_transfer, 0.0)
        self.assertAlmostEqual(
            result.state.connected_gas_mass,
            state.connected_gas_mass,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(result.gas_ledger_residual, 0.0, delta=1.0e-15)
        self.assertAlmostEqual(result.eos_residual, 0.0, delta=1.0e-8)

    def test_open_top_uses_riemann_mass_and_momentum_and_closes_ledger(self) -> None:
        state, characteristics, _ = self._fixture(gas_head=0.40)
        result = advance_eos_front_coupler(
            state,
            self.params,
            characteristics,
            self.areas,
            self.open_top,
            dt=1.0e-7,
        )
        self.assertGreater(result.top_gas_flux.mass_rate, 0.0)
        self.assertGreater(result.top_gas_flux.momentum_flux, 0.0)
        self.assertGreater(result.top_mass_transfer, 0.0)
        self.assertAlmostEqual(
            result.state.connected_gas_mass + result.top_mass_transfer,
            state.connected_gas_mass,
            delta=1.0e-14,
        )
        self.assertAlmostEqual(result.gas_ledger_residual, 0.0, delta=1.0e-14)
        self.assertAlmostEqual(
            result.state.cumulative_atmospheric_mass_out,
            result.top_mass_transfer,
            delta=1.0e-15,
        )

    def test_east_rh_front_is_signed_positive_zero_and_negative(self) -> None:
        gas_head = 0.3780169412545817
        state, characteristics, pressure = self._fixture(gas_head=gas_head)
        # Use the exact east-front datum from the independent RH regression.
        state = replace(
            state,
            east_front=MovingFrontState(
                position=0.10,
                free_surface_depth=0.070,
                pressurised_head_foot=0.450,
            ),
        )
        speeds: list[float] = []
        for head_offset in (-0.05, 0.0, 0.05):
            candidate = evaluate_eos_front_candidate_at_pressure(
                state,
                self.params,
                characteristics,
                self.areas,
                self.closed_top,
                dt=1.0e-5,
                node_pressure_abs=(
                    pressure
                    + self.params.liquid_density
                    * self.params.gravity
                    * head_offset
                ),
            )
            speeds.append(candidate.east_solution.interface_speed)
        self.assertLess(speeds[0], 0.0)
        self.assertAlmostEqual(speeds[1], 0.0, delta=1.0e-9)
        self.assertGreater(speeds[2], 0.0)

    def test_resolved_tee_void_not_a_fitted_total_volume_closes_storage(self) -> None:
        state, characteristics, pressure = self._fixture()
        driven = replace(
            characteristics,
            west=LiquidCharacteristic(pressure - 800.0, 0.0, 28.0, 0.75),
        )
        dt = 2.0e-4
        result = advance_eos_front_coupler(
            state,
            self.params,
            driven,
            self.areas,
            self.closed_top,
            dt=dt,
        )
        q_sum = math.fsum(
            flux.volume_flux
            for flux in result.liquid_branch_face_fluxes.values()
        )
        self.assertAlmostEqual(
            result.state.volumes.tee_void,
            state.volumes.tee_void + q_sum * dt,
            delta=1.0e-15,
        )
        self.assertEqual(
            result.state.volumes.tee_control_volume,
            state.volumes.tee_control_volume,
        )

    def test_incompatible_prescribed_pressure_is_rejected(self) -> None:
        state, characteristics, pressure = self._fixture()
        with self.assertRaises(IncompatiblePrescribedPressure) as caught:
            advance_eos_front_coupler(
                state,
                self.params,
                characteristics,
                self.areas,
                self.closed_top,
                dt=1.0e-3,
                prescribed_pressure_abs=pressure + 500.0,
            )
        self.assertNotEqual(caught.exception.eos_residual, 0.0)
        self.assertGreater(caught.exception.prescribed_pressure_abs, pressure)


if __name__ == "__main__":
    unittest.main()
