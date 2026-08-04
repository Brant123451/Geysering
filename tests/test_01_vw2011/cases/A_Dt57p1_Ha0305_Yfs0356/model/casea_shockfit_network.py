"""Case-A valve-release shock fitting with conservative side-T face fluxes.

The copied Tosan-style core advances the moving air/water interface through the
physical side-T and remains the owner of the horizontal finite-volume state.
This adapter changes no interface position or field shape.  Its additional
operation replaces the uninterrupted horizontal face flux by the west/east
mass fluxes of the simultaneously solved three-branch node.  No hold, refill,
pocket projection, selected-cell compression, or cross-case call is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable

import numpy as np

import tosan2021_horizontal_shockfit as CASE_A_SHOCKFIT_CORE


CASE_A_SHOCKFIT_SOURCE = Path(__file__).with_name(
    "tosan2021_horizontal_shockfit.py"
)

VentPressureHook = Callable[[float, float, float], float]


@dataclass(frozen=True)
class JunctionEventAdvance:
    """Pre-arrival advance stopped on the last admissible side of the T.

    ``state.interface_x`` never exceeds ``junction_face_x``.  If ``reached``
    is true, the remaining geometric gap is within ``location_tolerance`` and
    the caller may perform the exact metadata-only topology event.  Any
    unconsumed part of the requested network step is returned explicitly; it
    must be advanced by the post-T graph, not by the retired straight-pipe
    interface.
    """

    state: CASE_A_SHOCKFIT_CORE.HorizontalState
    elapsed: float
    remaining: float
    reached: bool


class CaseASideTShockFit(CASE_A_SHOCKFIT_CORE.Tosan2021HorizontalShockFit):
    """Shock-fitting core plus a conservative side-T face coupling."""

    @property
    def junction_face_index(self) -> int:
        """Finite-volume face nearest the measured side-T centreline."""

        return int(np.clip(
            round(float(self.config.vent_x) / self.dx),
            1,
            self.ncell - 1,
        ))

    @property
    def junction_face_x(self) -> float:
        return float(self.junction_face_index * self.dx)

    def _effective_pressure(
        self,
        *,
        time: float,
        interface_x: float,
        closed_pressure_abs: float,
        external_pressure_abs: float | None,
    ) -> tuple[float, bool]:
        """Open the side branch only after the fitted front crosses its face.

        The measured T coordinate generally lies inside a finite-volume cell.
        Handing the state off there leaves the corresponding cell liquid-full
        even though the subcell front has touched the opening, so the network
        sees zero gas-mouth area.  Attaching a graph branch to the nearest
        finite-volume *face* is the consistent discrete topology; its position
        error is bounded by half a cell and converges under refinement.
        """

        vented = bool(interface_x >= self.junction_face_x)
        if external_pressure_abs is not None:
            pressure = float(external_pressure_abs)
        elif vented and self.vent_pressure_hook is not None:
            pressure = float(self.vent_pressure_hook(
                time, interface_x, closed_pressure_abs
            ))
        else:
            pressure = float(closed_pressure_abs)
        if not np.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("effective gas pressure must be positive and finite")
        return pressure, vented

    def step_until_junction(
        self,
        state,
        dt: float,
        *,
        external_pressure_abs: float | None = None,
        location_tolerance: float | None = None,
    ) -> JunctionEventAdvance:
        """Advance the pre-T solution without allowing its front through T.

        A trial crossing is located by deterministic bisection in physical
        time.  The returned state is always the last state on the west side of
        the junction.  This is an event locator only: it does not remap fields,
        create a tee volume, or advance either post-T branch front.
        """

        requested = float(dt)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("dt must be positive and finite")
        x_t = self.junction_face_x
        x0 = float(state.interface_x)
        if x0 > x_t:
            raise ValueError(
                "the legacy interface has already crossed the side T"
            )
        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(x_t), self.dx)
            if location_tolerance is None
            else float(location_tolerance)
        )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "location_tolerance must be finite and non-negative"
            )
        if x_t - x0 <= tolerance:
            return JunctionEventAdvance(
                state=state,
                elapsed=0.0,
                remaining=requested,
                reached=True,
            )

        trial = self.step(
            state,
            requested,
            external_pressure_abs=external_pressure_abs,
        )
        x_trial = float(trial.interface_x)
        if x_trial <= x_t:
            reached = bool(x_t - x_trial <= tolerance)
            return JunctionEventAdvance(
                state=trial,
                elapsed=requested,
                remaining=0.0,
                reached=reached,
            )

        low_time = 0.0
        low_state = state
        high_time = requested
        for _ in range(90):
            middle_time = 0.5 * (low_time + high_time)
            candidate = self.step(
                state,
                middle_time,
                external_pressure_abs=external_pressure_abs,
            )
            if float(candidate.interface_x) <= x_t:
                low_time = middle_time
                low_state = candidate
            else:
                high_time = middle_time
            if (
                x_t - float(low_state.interface_x) <= tolerance
                and high_time - low_time
                <= 8.0 * np.finfo(float).eps * max(1.0, requested)
            ):
                break

        if float(low_state.interface_x) > x_t:
            raise FloatingPointError(
                "junction event locator returned a crossed legacy front"
            )
        if x_t - float(low_state.interface_x) > tolerance:
            raise FloatingPointError(
                "junction event could not be located from the west side"
            )
        remaining = max(requested - low_time, 0.0)
        return JunctionEventAdvance(
            state=low_state,
            elapsed=low_time,
            remaining=remaining,
            reached=True,
        )

    def apply_junction_liquid_fluxes(
        self,
        state,
        *,
        west_flow: float,
        east_flow: float,
        dt: float,
    ):
        """Replace the uninterrupted-pipe flux by the two side-T fluxes.

        Positive ``west_flow`` and ``east_flow`` point east.  Hence the volume
        entering the vertical branch over this step is

        ``(west_flow - east_flow) * dt``.

        The underlying shock-fitting advance treats the horizontal pipe as
        uninterrupted.  This correction replaces its single internal face
        flux by the two characteristic branch fluxes on the west and east
        sides of the physical T.  It therefore changes the two adjacent
        control volumes through their faces; it does not compress or empty a
        selected T cell.  The horizontal volume change is exactly the
        negative of the vertical volume change.
        """

        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        q_w = float(west_flow)
        q_e = float(east_flow)
        if not (np.isfinite(q_w) and np.isfinite(q_e)):
            raise ValueError("side-T branch flows must be finite")
        if abs(q_w - q_e) * step <= 1.0e-18:
            return state

        area = np.asarray(state.area, dtype=float).copy()
        discharge = np.asarray(state.discharge, dtype=float).copy()
        face = self.junction_face_index
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
                "side-T face flux emptied an adjacent horizontal cell"
            )

        # Momentum exchange follows the control-volume balance.  Water
        # returning vertically through the side T has no incoming *axial*
        # momentum, so adding its volume leaves the stored horizontal momentum
        # Q unchanged.  When liquid leaves for the riser, it carries the local
        # axial velocity and therefore removes Q in the same proportion as A.
        # This replaces the former acoustic-Courant relaxation of cell-centre
        # discharge, which was not a momentum flux and could launch a
        # grid-local impulse at the junction.
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
            raise FloatingPointError("side-T face flux lost liquid volume")

        boundary_area = float(self.section.area_from_depth(
            state.interface_free_surface_depth
        ))
        gas_volume = self._connected_gas_volume(
            area,
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
            area=area,
            discharge=discharge,
            gas=gas,
            air_pressure_abs=gas.pressure_abs,
            wetting_front_x=self._wetting_front(
                area, state.wetting_front_x
            ),
        )


def case_a_config(
    *,
    dx: float = 0.010,
    wave_speed: float = 28.0,
    output_water_contour: float = 0.10,
):
    """Return Case-A geometry and material parameters."""

    return CASE_A_SHOCKFIT_CORE.HorizontalConfig(
        length=4.006,
        diameter=0.094,
        valve_x=0.546,
        vent_x=3.516,
        dx=dx,
        wave_speed=wave_speed,
        gamma=1.4,
        initial_air_head=0.305,
        # ``Yfs0`` is measured upward from the horizontal-pipe crown in the
        # experiment.  The shock-fitting section head is measured from the
        # pipe invert, so the hydrostatic full-pipe head is D + Yfs0.
        initial_water_head=0.094 + 0.356,
        wetting_front_report_fraction=output_water_contour,
    )


def build_case_a_shockfit_solver(
    *,
    dx: float = 0.010,
    wave_speed: float = 28.0,
    output_water_contour: float = 0.10,
    vent_pressure_hook: VentPressureHook | None = None,
):
    return CaseASideTShockFit(
        case_a_config(
            dx=dx,
            wave_speed=wave_speed,
            output_water_contour=output_water_contour,
        ),
        vent_pressure_hook=vent_pressure_hook,
    )


__all__ = [
    "CASE_A_SHOCKFIT_CORE",
    "CASE_A_SHOCKFIT_SOURCE",
    "CaseASideTShockFit",
    "JunctionEventAdvance",
    "build_case_a_shockfit_solver",
    "case_a_config",
]
