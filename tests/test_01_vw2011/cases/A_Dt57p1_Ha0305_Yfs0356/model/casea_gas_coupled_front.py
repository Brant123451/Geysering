"""Gas-characteristic closure for a Case-A material regime front.

The gas-side pressure at a moving pressurised--stratified interface is a
boundary pressure, not the centre pressure of the nearest gas cell.  For the
isothermal gas branch the incoming acoustic characteristic gives

    p_g,Gamma - p_g,1 - rho_g,1 c_g (w_Gamma - u_g,1) = 0.

This module couples that relation to the existing liquid Rankine--Hugoniot
solver.  The scalar pressure is solved without a speed cap, time window,
receiver prefill, or prescribed front trajectory.  At the returned material
front ``u_g,Gamma = w_Gamma`` and therefore the ALE gas-mass flux through the
front is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_tjunction_shock_network import (
    BranchGeometry,
    MovingFrontState,
    PressureSolveError,
    solve_front_rankine_hugoniot,
)
from tosan2021_horizontal_shockfit import TosanInterfaceSolution


@dataclass(frozen=True)
class GasCellTrace:
    """Resolved gas trace adjacent to the material interface."""

    density: float
    velocity: float
    sound_speed: float

    def __post_init__(self) -> None:
        values = (self.density, self.velocity, self.sound_speed)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gas trace must be finite")
        if self.density <= 0.0 or self.sound_speed <= 0.0:
            raise ValueError("gas density and sound speed must be positive")

    @property
    def pressure_abs(self) -> float:
        """Isothermal ideal-gas pressure ``rho*c_g**2``."""

        return self.density * self.sound_speed**2

    @property
    def acoustic_impedance(self) -> float:
        return self.density * self.sound_speed

    def right_going_boundary_velocity(self, pressure_abs: float) -> float:
        """Linear subsonic characteristic trace at a boundary to the right.

        The sign convention is the one already used by the material-front
        closure below: a right-going acoustic characteristic satisfies

        ``p_b-p_1-rho_1*c_g*(u_b-u_1)=0``.

        This returns the characteristic value without imposing a sonic cap.
        A caller whose boundary must remain subsonic must reject, rather than
        clip, a value outside that domain.
        """

        if not math.isfinite(pressure_abs) or pressure_abs <= 0.0:
            raise ValueError("gas boundary pressure must be positive and finite")
        return float(
            self.velocity
            + (pressure_abs - self.pressure_abs) / self.acoustic_impedance
        )

    def right_boundary_outflow_velocity(self, pressure_abs: float) -> float:
        """Subsonic outflow trace at the right face of a west gas branch.

        At a fixed right boundary, the characteristic arriving from the
        interior holds ``u+p/(rho*c)`` constant to first acoustic order, so

        ``u_b = u_1 + (p_1-p_b)/(rho_1*c)``.

        This is distinct from :meth:`right_going_boundary_velocity`, which is
        the moving-material-front convention retained by the existing local
        front closure.  Neither relation applies a sonic cap.
        """

        if not math.isfinite(pressure_abs) or pressure_abs <= 0.0:
            raise ValueError("gas boundary pressure must be positive and finite")
        return float(
            self.velocity
            + (self.pressure_abs - pressure_abs) / self.acoustic_impedance
        )


@dataclass(frozen=True)
class GasCoupledFrontSolution:
    """Joint gas-characteristic and liquid-RH interface solution."""

    gas_pressure_abs: float
    gas_velocity: float
    liquid: TosanInterfaceSolution
    characteristic_residual: float
    pressure_iterations: int

    @property
    def interface_speed(self) -> float:
        return float(self.liquid.interface_speed)

    @property
    def relative_gas_mass_flux_per_area(self) -> float:
        """Material-front ALE mass flux, identically zero by construction."""

        return 0.0


def _candidate(
    pressure_abs: float,
    *,
    front: MovingFrontState,
    geometry: BranchGeometry,
    gas_trace: GasCellTrace,
    atmospheric_pressure: float,
    liquid_density: float,
    gravity: float,
    free_surface_velocity: float,
    dt: float,
    liquid_tolerance: float,
    liquid_max_iterations: int,
) -> tuple[float, TosanInterfaceSolution]:
    liquid = solve_front_rankine_hugoniot(
        front,
        geometry,
        gas_pressure_abs=pressure_abs,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        free_surface_velocity=free_surface_velocity,
        dt=dt,
        tolerance=liquid_tolerance,
        max_iterations=liquid_max_iterations,
    )
    residual = gas_trace.acoustic_impedance * (
        gas_trace.right_going_boundary_velocity(pressure_abs)
        - liquid.interface_speed
    )
    return float(residual), liquid


def solve_gas_coupled_material_front(
    front: MovingFrontState,
    geometry: BranchGeometry,
    *,
    gas_trace: GasCellTrace,
    atmospheric_pressure: float,
    liquid_density: float,
    gravity: float,
    free_surface_velocity: float,
    dt: float = 0.0,
    pressure_tolerance: float = 1.0e-7,
    characteristic_tolerance: float = 1.0e-10,
    liquid_tolerance: float = 1.0e-10,
    max_iterations: int = 80,
) -> GasCoupledFrontSolution:
    """Solve the gas boundary pressure and liquid RH state simultaneously.

    The pressure bracket is expanded about the resolved cell pressure.  Failed
    liquid-RH samples are ignored; the final result is accepted only when the
    dimensional gas-characteristic residual is below its pressure-scaled
    tolerance.  No interface-speed bound is applied.
    """

    scalars = (
        atmospheric_pressure,
        liquid_density,
        gravity,
        free_surface_velocity,
        dt,
        pressure_tolerance,
        characteristic_tolerance,
        liquid_tolerance,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("front-closure data must be finite")
    if min(
        atmospheric_pressure,
        liquid_density,
        gravity,
        pressure_tolerance,
        characteristic_tolerance,
        liquid_tolerance,
    ) <= 0.0:
        raise ValueError("front-closure scales must be positive")
    if dt < 0.0 or max_iterations < 1:
        raise ValueError("dt and max_iterations are invalid")

    scale = max(gas_trace.pressure_abs, atmospheric_pressure, 1.0)
    accepted_residual = characteristic_tolerance * scale

    def evaluate(pressure: float) -> tuple[float, TosanInterfaceSolution] | None:
        if not math.isfinite(pressure) or pressure <= 0.0:
            return None
        try:
            return _candidate(
                pressure,
                front=front,
                geometry=geometry,
                gas_trace=gas_trace,
                atmospheric_pressure=atmospheric_pressure,
                liquid_density=liquid_density,
                gravity=gravity,
                free_surface_velocity=free_surface_velocity,
                dt=dt,
                liquid_tolerance=liquid_tolerance,
                liquid_max_iterations=max_iterations,
            )
        except (PressureSolveError, ValueError, FloatingPointError):
            return None

    centre_pressure = gas_trace.pressure_abs
    centre = evaluate(centre_pressure)
    if centre is not None and abs(centre[0]) <= accepted_residual:
        return GasCoupledFrontSolution(
            gas_pressure_abs=centre_pressure,
            gas_velocity=float(centre[1].interface_speed),
            liquid=centre[1],
            characteristic_residual=float(centre[0]),
            pressure_iterations=1,
        )

    samples: list[tuple[float, float, TosanInterfaceSolution]] = []
    if centre is not None:
        samples.append((centre_pressure, centre[0], centre[1]))
    delta = max(1.0, 1.0e-6 * centre_pressure)
    bracket: tuple[
        tuple[float, float, TosanInterfaceSolution],
        tuple[float, float, TosanInterfaceSolution],
    ] | None = None
    for _ in range(64):
        for pressure in (max(math.nextafter(0.0, 1.0), centre_pressure - delta), centre_pressure + delta):
            if any(pressure == item[0] for item in samples):
                continue
            value = evaluate(pressure)
            if value is not None:
                samples.append((pressure, value[0], value[1]))
        samples.sort(key=lambda item: item[0])
        for left, right in zip(samples[:-1], samples[1:]):
            if left[1] == 0.0 or right[1] == 0.0 or left[1] * right[1] < 0.0:
                bracket = (left, right)
                break
        if bracket is not None:
            break
        delta *= 2.0
    if bracket is None:
        best = min(samples, key=lambda item: abs(item[1])) if samples else None
        detail = "no admissible liquid-RH pressure samples"
        if best is not None:
            detail = (
                f"best residual={best[1]:.9g} Pa at "
                f"p={best[0]:.9g} Pa"
            )
        raise PressureSolveError(
            "could not bracket the gas-characteristic/material-front pressure; "
            + detail
        )

    left, right = bracket
    if left[1] == 0.0:
        chosen = left
        iterations = 1
    elif right[1] == 0.0:
        chosen = right
        iterations = 1
    else:
        chosen = min((left, right), key=lambda item: abs(item[1]))
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            pressure = 0.5 * (left[0] + right[0])
            middle_value = evaluate(pressure)
            if middle_value is None:
                raise PressureSolveError(
                    "liquid-RH solve failed inside a valid gas-pressure bracket"
                )
            middle = (pressure, middle_value[0], middle_value[1])
            chosen = middle
            if (
                abs(middle[1]) <= accepted_residual
                or right[0] - left[0] <= pressure_tolerance
            ):
                break
            if left[1] * middle[1] <= 0.0:
                right = middle
            else:
                left = middle
        else:
            raise PressureSolveError(
                "gas-characteristic/material-front pressure did not converge"
            )

    if abs(chosen[1]) > max(accepted_residual, pressure_tolerance):
        raise PressureSolveError(
            "gas-characteristic/material-front residual exceeds tolerance: "
            f"{chosen[1]:.9g} Pa"
        )
    return GasCoupledFrontSolution(
        gas_pressure_abs=float(chosen[0]),
        gas_velocity=float(chosen[2].interface_speed),
        liquid=chosen[2],
        characteristic_residual=float(chosen[1]),
        pressure_iterations=int(iterations),
    )


__all__ = [
    "GasCellTrace",
    "GasCoupledFrontSolution",
    "solve_gas_coupled_material_front",
]
