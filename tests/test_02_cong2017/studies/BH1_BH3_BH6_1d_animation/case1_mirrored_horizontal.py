"""Campaign-2 mirror adapter for the validated Campaign-1 horizontal core.

The numerical model is imported from the Case-A production source rather than
copied.  Campaign 2 is expressed in the mirrored coordinate ``x' = L - x``:
the downstream atmospheric pocket is therefore on the left of the mirrored
grid, exactly matching the dry/free-surface--pressurised topology solved by
``Tosan2021HorizontalShockFit``.

Only apparatus data and boundary orientation are adapted here.  Liquid area,
discharge, the moving wet/dry front, the Tosan pressure interface, and the
polytropic gas inventory are all advanced by the Case-1 implementation.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
TESTS_ROOT = HERE.parents[2]
CASE1_MODEL = (
    TESTS_ROOT
    / "test_01_vw2011"
    / "cases"
    / "A_Dt57p1_Ha0305_Yfs0356"
    / "model"
)
if str(CASE1_MODEL) not in sys.path:
    sys.path.insert(0, str(CASE1_MODEL))

from tosan2021_horizontal_shockfit import (  # noqa: E402
    HorizontalConfig,
    Tosan2021HorizontalShockFit,
)


CORE_SOURCE = CASE1_MODEL / "tosan2021_horizontal_shockfit.py"
EXPECTED_CORE_SHA256 = (
    "90e84da9afa0ec8465d80f87fc701dfb8f0fad6f97350ea708074a50192b6119"
)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Campaign2Case1MirroredHorizontal(Tosan2021HorizontalShockFit):
    """Case-1 horizontal solver in the mirrored Campaign-2 coordinate.

    ``right_boundary='transmissive'`` holds the initially prescribed reservoir
    state at the far end during the short release-wave stage.  Once the
    Case-1 wetting front reaches the closed downstream cap, the Campaign-2
    finite-volume network becomes the owner.  The latter must control the
    subsequent crown-pocket migration, tee compression and riser venting,
    because those are the branch-selecting mechanisms of Campaign 2.

    The paper's 0.20 s valve motion is represented by the same sine-squared
    effective-area law used in the Campaign-2 OpenFOAM solver.  It scales
    hydraulic time only during the short opening stroke; after the valve is
    fully open the Case-1 core advances in physical time one-for-one.
    """

    def __init__(
        self,
        *,
        length: float,
        diameter: float,
        physical_valve_x: float,
        physical_riser_x: float,
        initial_water_head_from_invert: float,
        dx: float,
        wave_speed: float,
        valve_open_time: float,
        liquid_density: float = 998.0,
        liquid_dynamic_viscosity: float = 0.001003,
        liquid_bulk_modulus: float = 2.2e9,
        atmospheric_pressure: float = 101_325.0,
        gravity: float = 9.81,
        gas_constant: float = 287.05,
        gas_temperature: float = 296.15,
        coupling_interval: float = 0.005,
    ) -> None:
        self.physical_length = float(length)
        self.physical_valve_x = float(physical_valve_x)
        self.physical_riser_x = float(physical_riser_x)
        self.valve_open_time = float(valve_open_time)
        self.coupling_interval = float(coupling_interval)
        self.liquid_dynamic_viscosity_Pa_s = float(
            liquid_dynamic_viscosity
        )
        self.liquid_bulk_modulus_Pa = float(liquid_bulk_modulus)
        if not np.isfinite(self.coupling_interval) or self.coupling_interval <= 0.0:
            raise ValueError("coupling_interval must be positive and finite")
        material_values = {
            "liquid_density": liquid_density,
            "liquid_dynamic_viscosity": self.liquid_dynamic_viscosity_Pa_s,
            "liquid_bulk_modulus": self.liquid_bulk_modulus_Pa,
            "atmospheric_pressure": atmospheric_pressure,
            "gravity": gravity,
            "gas_constant": gas_constant,
            "gas_temperature": gas_temperature,
        }
        if not all(
            np.isfinite(value) and float(value) > 0.0
            for value in material_values.values()
        ):
            raise ValueError(
                "all Campaign-2 material and ambient inputs must be "
                "positive and finite"
            )
        actual_core_hash = source_sha256(CORE_SOURCE)
        if actual_core_hash != EXPECTED_CORE_SHA256:
            raise RuntimeError(
                "Case-1 horizontal core changed during the Campaign-2 study: "
                f"expected {EXPECTED_CORE_SHA256}, found {actual_core_hash}"
            )
        mirrored_valve = self.physical_length - self.physical_valve_x
        mirrored_riser = self.physical_length - self.physical_riser_x
        config = HorizontalConfig(
            length=self.physical_length,
            diameter=float(diameter),
            valve_x=mirrored_valve,
            vent_x=mirrored_riser,
            initial_air_head=0.0,
            initial_water_head=float(initial_water_head_from_invert),
            dx=float(dx),
            wave_speed=float(wave_speed),
            gravity=float(gravity),
            liquid_density=float(liquid_density),
            atmospheric_pressure=float(atmospheric_pressure),
            gas_constant=float(gas_constant),
            gamma=1.4,
            temperature=float(gas_temperature),
            right_boundary="transmissive",
        )
        super().__init__(config)

    def initial_state(self):
        return self.case_b_initial_state(
            initial_air_gauge_head=0.0,
            initial_water_head=self.config.initial_water_head,
        )

    def step(self, state, dt: float, *, external_pressure_abs=None):
        """Advance one physical step including the prescribed valve stroke."""

        physical_dt = float(dt)
        if not np.isfinite(physical_dt) or physical_dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        midpoint = float(state.time) + 0.5 * physical_dt
        if self.valve_open_time <= 0.0:
            transmissivity = 1.0
        else:
            normalized_time = float(
                np.clip(midpoint / self.valve_open_time, 0.0, 1.0)
            )
            opening_sine = np.sin(0.5 * np.pi * normalized_time)
            transmissivity = float(opening_sine * opening_sine)
        hydraulic_dt = transmissivity * physical_dt
        if hydraulic_dt > 1.0e-15:
            advanced = super().step(
                state,
                hydraulic_dt,
                external_pressure_abs=external_pressure_abs,
            )
        else:
            advanced = state
        # The shock/core state carries physical experiment time.  The reduced
        # hydraulic increment is only the valve-area integral during opening.
        return replace(advanced, time=float(state.time) + physical_dt)

    def step_physical(self, state, dt: float):
        return self.step(state, dt, external_pressure_abs=None)

    def map_to_physical(self, state, *, x_target, full_area: float, dx: float):
        """Return Case-1 conserved fields in the paper's left-to-right axis."""

        x_target = np.asarray(x_target, dtype=float)
        if x_target.shape != self.x.shape:
            raise ValueError("mirrored Case-1 and Campaign-2 grids differ")
        if not np.allclose(x_target, self.x, rtol=0.0, atol=1.0e-12):
            raise ValueError("mirrored Case-1 and Campaign-2 grids differ")
        if not np.isclose(
            self.section.full_area,
            float(full_area),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("Case-1 and Campaign-2 pipe areas differ")
        area_m = np.asarray(state.area, dtype=float).copy()
        discharge_m = np.asarray(state.discharge, dtype=float).copy()
        void_volume = (
            np.maximum(
                float(full_area) - np.clip(area_m, 0.0, float(full_area)),
                0.0,
            )
            * float(dx)
        )
        connected = self.x < float(state.interface_x)
        void_volume = np.where(connected, void_volume, 0.0)
        total_void = float(np.sum(void_volume))
        if total_void <= 1.0e-14:
            raise FloatingPointError("Case-1 fitted state has no gas volume")
        gas_mass_m = float(state.gas.mass) * void_volume / total_void
        interface_x = max(float(state.interface_x), 0.5 * float(dx))
        gas_velocity_m = np.where(
            void_volume > 0.0,
            float(state.interface_speed)
            * np.clip(self.x / interface_x, 0.0, 1.0),
            0.0,
        )
        gas_momentum_m = gas_mass_m * gas_velocity_m
        return (
            area_m[::-1].copy(),
            -discharge_m[::-1].copy(),
            gas_mass_m[::-1].copy(),
            -gas_momentum_m[::-1].copy(),
        )

    def physical_fronts(self, state) -> dict[str, float | bool]:
        """Map the fitted interface metadata back to the experiment axis."""

        return {
            "wetting_front_x": float(
                self.physical_length - state.wetting_front_x
            ),
            "gas_nose_x": float(self.physical_length - state.interface_x),
            "vented": bool(state.vented),
        }

    @property
    def physical_junction_face_index(self) -> int:
        """Finite-volume face nearest the measured Campaign-2 side T."""

        return int(np.clip(
            round(self.physical_riser_x / self.dx),
            1,
            self.ncell - 1,
        ))

    @property
    def physical_junction_face_x(self) -> float:
        return float(self.physical_junction_face_index * self.dx)

    def apply_physical_junction_liquid_fluxes(
        self,
        state,
        *,
        west_flow: float,
        east_flow: float,
        dt: float,
    ):
        """Commit one conservative physical-coordinate side-T liquid flux.

        ``west_flow`` and ``east_flow`` are positive toward increasing physical
        x.  Thus ``(west_flow-east_flow)*dt`` is the volume delivered to the
        riser.  The identical transaction must be applied with the opposite
        sign to the vertical bottom control volume by the coupled driver.
        """

        step = float(dt)
        q_w = float(west_flow)
        q_e = float(east_flow)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not (math.isfinite(q_w) and math.isfinite(q_e)):
            raise ValueError("side-T branch flows must be finite")
        if abs(q_w - q_e) * step <= 1.0e-18:
            return state

        # Work in the paper's physical coordinate; map back exactly once.
        area = np.asarray(state.area, dtype=float)[::-1].copy()
        discharge = -np.asarray(state.discharge, dtype=float)[::-1].copy()
        face = self.physical_junction_face_index
        west = face - 1
        east = face
        reference_flow = 0.5 * (
            float(discharge[west]) + float(discharge[east])
        )
        old_area = area[[west, east]].copy()
        area[west] += step / self.dx * (reference_flow - q_w)
        area[east] += step / self.dx * (q_e - reference_flow)
        if np.any(area[[west, east]] <= 1.0e-10 * self.section.full_area):
            raise FloatingPointError(
                "side-T transaction emptied an adjacent horizontal cell"
            )

        # Liquid leaving through the 90-degree branch carries away its local
        # axial momentum; vertically returning liquid imports no prescribed
        # horizontal momentum.  The missing axial impulse is a wall reaction.
        for cell, area_before in zip((west, east), old_area, strict=True):
            if area[cell] < area_before:
                discharge[cell] *= area[cell] / area_before

        expected_change = (q_e - q_w) * step
        actual_change = float(
            np.sum(area[[west, east]] - old_area) * self.dx
        )
        if not math.isclose(
            actual_change,
            expected_change,
            rel_tol=1.0e-10,
            abs_tol=1.0e-16,
        ):
            raise FloatingPointError("side-T transaction lost liquid volume")

        area_mirrored = area[::-1].copy()
        discharge_mirrored = -discharge[::-1].copy()
        boundary_area = float(self.section.area_from_depth(
            state.interface_free_surface_depth
        ))
        gas_volume = self._connected_gas_volume(
            area_mirrored,
            float(state.interface_x),
            boundary_area,
            float(state.wetting_front_x),
        )
        gas = state.gas.with_volume(max(
            gas_volume,
            1.0e-9 * self.section.full_area * self.config.length,
        ))
        return replace(
            state,
            area=area_mirrored,
            discharge=discharge_mirrored,
            gas=gas,
            air_pressure_abs=gas.pressure_abs,
            wetting_front_x=self._wetting_front(
                area_mirrored,
                state.wetting_front_x,
            ),
        )

    @staticmethod
    def _with_open_polytropic_mass(gas, new_mass: float):
        """Update an open uniform pocket at fixed volume and entropy.

        At the transaction instant ``p_new/p_old=(m_new/m_old)^gamma``.
        Subsequent Case-1 volume changes retain the same polytropic invariant.
        """

        mass = float(new_mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("open-pocket gas mass must remain positive")
        old_mass = float(gas.mass)
        if not math.isfinite(old_mass) or old_mass <= 0.0:
            raise ValueError("existing gas mass must be positive and finite")
        pressure = float(gas.pressure_abs) * (
            mass / old_mass
        ) ** float(gas.gamma)
        return replace(
            gas,
            reference_volume=float(gas.volume),
            reference_pressure_abs=pressure,
            volume=float(gas.volume),
            mass=mass,
        )

    def apply_physical_junction_gas_mass_flux(
        self,
        state,
        *,
        mass_flow_to_riser: float,
        dt: float,
    ):
        """Commit gas mass leaving the horizontal pocket through the side T."""

        step = float(dt)
        mass_flow = float(mass_flow_to_riser)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(mass_flow):
            raise ValueError("side-T gas mass flow must be finite")
        transferred = mass_flow * step
        if abs(transferred) <= 1.0e-18:
            return state
        if transferred > 0.0:
            fronts = self.physical_fronts(state)
            if (
                float(fronts["gas_nose_x"])
                > self.physical_junction_face_x + 0.5 * self.dx
            ):
                raise ValueError(
                    "gas cannot leave before the material pocket reaches the side T"
                )
        gas = self._with_open_polytropic_mass(
            state.gas,
            float(state.gas.mass) - transferred,
        )
        return replace(
            state,
            gas=gas,
            air_pressure_abs=gas.pressure_abs,
        )

    @staticmethod
    def regularize_physical_momentum(area, discharge, *, full_area: float):
        """Use the Case-1 production wet/dry momentum desingularisation."""

        a = np.maximum(np.asarray(area, dtype=float), 0.0)
        q = np.asarray(discharge, dtype=float)
        if a.shape != q.shape:
            raise ValueError("area and discharge shapes differ")
        scale = 1.0e-3 * float(full_area)
        return q * a * a / (a * a + scale * scale)

    def provenance(self) -> dict[str, object]:
        return {
            "model": "Case-1 Tosan shock-fit + circular Saint-Venant wet/dry core",
            "coordinate_map": "x_mirror = L - x_physical",
            "core_source": str(CORE_SOURCE),
            "core_sha256": source_sha256(CORE_SOURCE),
            "conservative_map_source": str(Path(__file__).resolve()),
            "adapter_sha256": source_sha256(Path(__file__).resolve()),
            "conservative_map": "Case-1 same-grid void-volume map, frozen locally",
            "valve_open_time_s": self.valve_open_time,
            "liquid_density_kg_m3": self.config.liquid_density,
            "liquid_dynamic_viscosity_Pa_s": (
                self.liquid_dynamic_viscosity_Pa_s
            ),
            "liquid_bulk_modulus_Pa": self.liquid_bulk_modulus_Pa,
            "atmospheric_pressure_Pa": self.config.atmospheric_pressure,
            "gravity_m_s2": self.config.gravity,
            "gas_constant_J_kg_K": self.config.gas_constant,
            "gas_temperature_K": self.config.temperature,
            "viscosity_closure": (
                "material provenance only; Case-1 core retains its frozen "
                "Darcy-friction closure"
            ),
            "bulk_modulus_closure": (
                "material provenance only; effective water-hammer wave "
                "speed is supplied separately"
            ),
            "network_coupling_interval_s": self.coupling_interval,
            "valve_area_law": "phi=sin(pi/2*min(max(t/t_open,0),1))^2",
            "pre_handoff_reservoir_boundary": "fixed initial state (transmissive Case-1 MOC end)",
            "adapter_handoff_condition": (
                "selected by the coupled network; the adapter does not "
                "change ownership itself"
            ),
        }


__all__ = [
    "Campaign2Case1MirroredHorizontal",
    "CORE_SOURCE",
    "EXPECTED_CORE_SHA256",
    "source_sha256",
]
