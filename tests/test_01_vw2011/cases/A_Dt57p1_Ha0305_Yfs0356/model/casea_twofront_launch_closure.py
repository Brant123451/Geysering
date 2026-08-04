"""EXPERIMENTAL non-production first-arrival algebraic diagnostic.

.. warning::
   This module solves the liquid zero-storage pressure and shared gas pressure
   as two independent unknowns.  ``LiquidCharacteristic`` defines its unknown
   as that same common gas-node pressure, so the separation amounts to an
   unmodelled interfacial pressure jump.  The routine is retained only to
   reproduce and test the over-constraint diagnosis.  Production integration
   must use the finite-geometric-node closure instead.

At the exact instant that the west gas pocket reaches the T, the east and
vertical stratified reaches have zero length.  Their material-front speeds
cannot be inferred from a pre-existing receiver gas cell.  This module solves
that topology event as one local algebraic problem; it owns no time loop and
creates no receiver inventory.

The liquid node is solved first from the three incoming liquid
characteristics and the zero-storage equation

``q_w + q_e + q_v = 0``.

Those three velocities are then held fixed while one shared gas pressure
``p_g,J`` is sought.  At every trial pressure the existing liquid
Rankine--Hugoniot closure gives the east and vertical material-front speeds.
The gas characteristic arriving from the west supplies exactly the mass
needed to create the two new gas volumes,

``m_dot_w = rho_J (A_g,e w_e + A_g,v w_v)``.

No branch split, launch speed, receiver prefill, pressure history, or target
waveform is prescribed.  The west gas-face area is an explicit input because
``GasCellTrace`` contains no geometry; inferring it from another branch would
silently add a Case-A-specific assumption to this local closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_gas_coupled_front import GasCellTrace
from casea_material_front_cutcell import InterfaceTraces
from casea_material_front_rh_adapter import (
    CaseAFixedPressureFrontTraceResult,
    build_casea_material_front_traces_at_pressure,
)
from casea_tjunction_shock_network import (
    BranchGeometry,
    PressureSolveError,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    ZeroStorageTNodeSolution,
    solve_zero_storage_t_node,
)
from casea_material_front_cutcell import PressurisedState


PRODUCTION_READY = False


class LaunchTopologyError(PressureSolveError):
    """The algebraic root is incompatible with two zero-length receivers."""


@dataclass(frozen=True)
class WestGasFaceFlux:
    """Gas flux integrated over the west face, positive into the T."""

    area: float
    pressure_abs: float
    density: float
    velocity: float
    mass_flux_per_area: float
    momentum_flux_per_area: float
    mass_rate: float
    momentum_rate: float


@dataclass(frozen=True)
class TwoFrontLaunchCandidate:
    """Complete residual evaluation at one trial shared gas pressure."""

    gas_node_pressure_abs: float
    east: CaseAFixedPressureFrontTraceResult
    vertical: CaseAFixedPressureFrontTraceResult
    west_gas_face_flux: WestGasFaceFlux
    east_gas_volume_creation_rate: float
    vertical_gas_volume_creation_rate: float
    gas_volume_creation_rate: float
    east_gas_mass_demand_rate: float
    vertical_gas_mass_demand_rate: float
    gas_mass_demand_rate: float
    gas_mass_balance_residual: float
    east_liquid_rh_residual: float
    vertical_liquid_rh_residual: float


@dataclass(frozen=True)
class TwoFrontLaunchSolution:
    """Accepted two-material-front launch state."""

    gas_node_pressure_abs: float
    liquid_node: ZeroStorageTNodeSolution
    east_traces: InterfaceTraces
    vertical_traces: InterfaceTraces
    west_gas_face_flux: WestGasFaceFlux
    west_gas_characteristic_mass_inflow: float
    east_gas_volume_creation_rate: float
    vertical_gas_volume_creation_rate: float
    gas_volume_creation_rate: float
    east_gas_mass_demand_rate: float
    vertical_gas_mass_demand_rate: float
    gas_mass_demand_rate: float
    gas_mass_balance_residual: float
    east_liquid_rh_residual: float
    vertical_liquid_rh_residual: float
    liquid_volume_balance_residual: float
    liquid_mass_balance_residual: float
    pressure_iterations: int


def _west_flux(
    trace: GasCellTrace,
    *,
    pressure_abs: float,
    face_area: float,
) -> WestGasFaceFlux:
    """Evaluate the west right-going subsonic characteristic face flux."""

    density = pressure_abs / trace.sound_speed**2
    velocity = trace.right_boundary_outflow_velocity(pressure_abs)
    if abs(velocity) >= trace.sound_speed:
        raise PressureSolveError(
            "the west characteristic trial is not subsonic; no velocity "
            "cap is applied"
        )
    mass_flux_per_area = density * velocity
    momentum_flux_per_area = density * velocity**2 + pressure_abs
    return WestGasFaceFlux(
        area=float(face_area),
        pressure_abs=float(pressure_abs),
        density=float(density),
        velocity=float(velocity),
        mass_flux_per_area=float(mass_flux_per_area),
        momentum_flux_per_area=float(momentum_flux_per_area),
        mass_rate=float(face_area * mass_flux_per_area),
        momentum_rate=float(face_area * momentum_flux_per_area),
    )


def evaluate_twofront_launch_candidate(
    *,
    gas_node_pressure_abs: float,
    west_gas_trace: GasCellTrace,
    west_gas_face_area: float,
    liquid_node: ZeroStorageTNodeSolution,
    east_geometry: BranchGeometry,
    vertical_geometry: BranchGeometry,
    east_pressurised_foot: PressurisedState,
    vertical_pressurised_foot: PressurisedState,
    east_stratified_liquid_area: float,
    vertical_stratified_liquid_area: float,
    atmospheric_pressure: float = 101_325.0,
    liquid_density: float = 998.0,
    gravity: float = 9.81,
    dt: float = 0.0,
    east_pressurised_friction_slope: float = 0.0,
    vertical_pressurised_friction_slope: float = 0.0,
) -> TwoFrontLaunchCandidate:
    """Evaluate the shared-pressure gas balance without changing state."""

    values = (
        gas_node_pressure_abs,
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
        dt,
        east_pressurised_friction_slope,
        vertical_pressurised_friction_slope,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("two-front launch inputs must be finite")
    if min(
        gas_node_pressure_abs,
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
    ) <= 0.0:
        raise ValueError("two-front launch physical inputs must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    east = build_casea_material_front_traces_at_pressure(
        east_pressurised_foot,
        stratified_liquid_area=east_stratified_liquid_area,
        free_surface_velocity=(
            liquid_node.branch_fluxes["east"].outward_velocity
        ),
        gas_pressure_abs=gas_node_pressure_abs,
        front_position=0.0,
        geometry=east_geometry,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        gas_sound_speed=west_gas_trace.sound_speed,
        dt=dt,
        pressurised_friction_slope=east_pressurised_friction_slope,
    )
    vertical = build_casea_material_front_traces_at_pressure(
        vertical_pressurised_foot,
        stratified_liquid_area=vertical_stratified_liquid_area,
        free_surface_velocity=(
            liquid_node.branch_fluxes["vertical"].outward_velocity
        ),
        gas_pressure_abs=gas_node_pressure_abs,
        front_position=0.0,
        geometry=vertical_geometry,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        gas_sound_speed=west_gas_trace.sound_speed,
        dt=dt,
        pressurised_friction_slope=vertical_pressurised_friction_slope,
    )
    west_flux = _west_flux(
        west_gas_trace,
        pressure_abs=gas_node_pressure_abs,
        face_area=west_gas_face_area,
    )
    east_volume_rate = east.gas_area * east.traces.speed
    vertical_volume_rate = vertical.gas_area * vertical.traces.speed
    volume_rate = math.fsum((east_volume_rate, vertical_volume_rate))
    density_node = gas_node_pressure_abs / west_gas_trace.sound_speed**2
    east_demand = density_node * east_volume_rate
    vertical_demand = density_node * vertical_volume_rate
    demand = math.fsum((east_demand, vertical_demand))
    residual = west_flux.mass_rate - demand
    return TwoFrontLaunchCandidate(
        gas_node_pressure_abs=float(gas_node_pressure_abs),
        east=east,
        vertical=vertical,
        west_gas_face_flux=west_flux,
        east_gas_volume_creation_rate=float(east_volume_rate),
        vertical_gas_volume_creation_rate=float(vertical_volume_rate),
        gas_volume_creation_rate=float(volume_rate),
        east_gas_mass_demand_rate=float(east_demand),
        vertical_gas_mass_demand_rate=float(vertical_demand),
        gas_mass_demand_rate=float(demand),
        gas_mass_balance_residual=float(residual),
        east_liquid_rh_residual=float(east.liquid.residual_linf),
        vertical_liquid_rh_residual=float(vertical.liquid.residual_linf),
    )


def solve_twofront_launch_closure(
    *,
    west_gas_trace: GasCellTrace,
    west_gas_face_area: float,
    liquid_characteristics: TeeLiquidCharacteristics,
    liquid_areas: ZeroStorageTBranchAreas,
    east_geometry: BranchGeometry,
    vertical_geometry: BranchGeometry,
    east_pressurised_foot: PressurisedState,
    vertical_pressurised_foot: PressurisedState,
    east_stratified_liquid_area: float,
    vertical_stratified_liquid_area: float,
    atmospheric_pressure: float = 101_325.0,
    liquid_density: float = 998.0,
    gravity: float = 9.81,
    dt: float = 0.0,
    east_pressurised_friction_slope: float = 0.0,
    vertical_pressurised_friction_slope: float = 0.0,
    pressure_hint_abs: float | None = None,
    mass_rate_absolute_tolerance: float = 1.0e-12,
    mass_rate_relative_tolerance: float = 1.0e-10,
    liquid_volume_tolerance: float = 1.0e-12,
    pressure_tolerance: float = 1.0e-7,
    launch_speed_tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> TwoFrontLaunchSolution:
    """Solve the exact first-arrival two-front launch pressure.

    The two zero-length receivers are admissible only when neither accepted
    material front recedes into the T.  A negative launch speed is rejected as
    a topology error; it is never replaced by zero.
    """

    scalars = (
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
        dt,
        mass_rate_absolute_tolerance,
        mass_rate_relative_tolerance,
        liquid_volume_tolerance,
        pressure_tolerance,
        launch_speed_tolerance,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("two-front solver inputs must be finite")
    if min(
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
        mass_rate_absolute_tolerance,
        mass_rate_relative_tolerance,
        liquid_volume_tolerance,
        pressure_tolerance,
        launch_speed_tolerance,
    ) <= 0.0:
        raise ValueError("two-front solver scales must be positive")
    if dt < 0.0 or max_iterations < 1:
        raise ValueError("dt or max_iterations is invalid")
    if pressure_hint_abs is not None and (
        not math.isfinite(pressure_hint_abs) or pressure_hint_abs <= 0.0
    ):
        raise ValueError("pressure hint must be positive and finite")

    liquid_node = solve_zero_storage_t_node(
        liquid_characteristics,
        liquid_areas,
        liquid_density=liquid_density,
        pressure_hint_abs=pressure_hint_abs or west_gas_trace.pressure_abs,
        volume_flux_tolerance=liquid_volume_tolerance,
        pressure_tolerance=pressure_tolerance,
        max_iterations=max_iterations,
    )

    def evaluate(pressure: float) -> TwoFrontLaunchCandidate | None:
        if not math.isfinite(pressure) or pressure <= 0.0:
            return None
        try:
            return evaluate_twofront_launch_candidate(
                gas_node_pressure_abs=pressure,
                west_gas_trace=west_gas_trace,
                west_gas_face_area=west_gas_face_area,
                liquid_node=liquid_node,
                east_geometry=east_geometry,
                vertical_geometry=vertical_geometry,
                east_pressurised_foot=east_pressurised_foot,
                vertical_pressurised_foot=vertical_pressurised_foot,
                east_stratified_liquid_area=east_stratified_liquid_area,
                vertical_stratified_liquid_area=(
                    vertical_stratified_liquid_area
                ),
                atmospheric_pressure=atmospheric_pressure,
                liquid_density=liquid_density,
                gravity=gravity,
                dt=dt,
                east_pressurised_friction_slope=(
                    east_pressurised_friction_slope
                ),
                vertical_pressurised_friction_slope=(
                    vertical_pressurised_friction_slope
                ),
            )
        except (PressureSolveError, ValueError, FloatingPointError):
            return None

    def accepted(candidate: TwoFrontLaunchCandidate) -> bool:
        scale = max(
            abs(candidate.west_gas_face_flux.mass_rate),
            abs(candidate.gas_mass_demand_rate),
            1.0e-30,
        )
        return abs(candidate.gas_mass_balance_residual) <= (
            mass_rate_absolute_tolerance
            + mass_rate_relative_tolerance * scale
        )

    centre_pressure = pressure_hint_abs or west_gas_trace.pressure_abs
    centres = {
        float(centre_pressure),
        float(west_gas_trace.pressure_abs),
        float(liquid_node.node_pressure_abs),
    }
    samples: list[TwoFrontLaunchCandidate] = []
    for pressure in sorted(centres):
        candidate = evaluate(pressure)
        if candidate is not None:
            if accepted(candidate):
                chosen = candidate
                iterations = 1
                break
            samples.append(candidate)
    else:
        chosen = None
        iterations = 0

    if chosen is None:
        delta = max(1.0, 1.0e-6 * centre_pressure)
        bracket: tuple[
            TwoFrontLaunchCandidate, TwoFrontLaunchCandidate
        ] | None = None
        for _ in range(64):
            for pressure in (
                max(math.nextafter(0.0, 1.0), centre_pressure - delta),
                centre_pressure + delta,
            ):
                if any(
                    math.isclose(
                        pressure,
                        item.gas_node_pressure_abs,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for item in samples
                ):
                    continue
                candidate = evaluate(pressure)
                if candidate is not None:
                    samples.append(candidate)
            samples.sort(key=lambda item: item.gas_node_pressure_abs)
            for left, right in zip(samples[:-1], samples[1:]):
                if (
                    left.gas_mass_balance_residual
                    * right.gas_mass_balance_residual
                    <= 0.0
                ):
                    bracket = (left, right)
                    break
            if bracket is not None:
                break
            delta *= 2.0
        if bracket is None:
            detail = "no admissible pressure samples"
            if samples:
                best = min(
                    samples,
                    key=lambda item: abs(item.gas_mass_balance_residual),
                )
                detail = (
                    f"best residual={best.gas_mass_balance_residual:.9g} "
                    f"kg/s at p={best.gas_node_pressure_abs:.9g} Pa"
                )
            raise PressureSolveError(
                "could not bracket the two-front gas-mass balance; " + detail
            )

        left, right = bracket
        chosen = min(
            (left, right),
            key=lambda item: abs(item.gas_mass_balance_residual),
        )
        for iterations in range(1, max_iterations + 1):
            pressure = 0.5 * (
                left.gas_node_pressure_abs + right.gas_node_pressure_abs
            )
            middle = evaluate(pressure)
            if middle is None:
                raise PressureSolveError(
                    "material-front solve failed inside the launch-pressure "
                    "bracket"
                )
            chosen = middle
            if accepted(middle) or (
                right.gas_node_pressure_abs - left.gas_node_pressure_abs
                <= pressure_tolerance
            ):
                break
            if (
                left.gas_mass_balance_residual
                * middle.gas_mass_balance_residual
                <= 0.0
            ):
                right = middle
            else:
                left = middle
        else:
            raise PressureSolveError(
                "two-front launch pressure did not converge"
            )

    if not accepted(chosen):
        raise PressureSolveError(
            "two-front gas-mass residual exceeds tolerance: "
            f"{chosen.gas_mass_balance_residual:.9g} kg/s"
        )
    speeds = (chosen.east.traces.speed, chosen.vertical.traces.speed)
    if any(speed < -launch_speed_tolerance for speed in speeds):
        raise LaunchTopologyError(
            "a zero-length receiver has a receding RH root; an active-set "
            "topology law is required and the speed was not clipped"
        )

    return TwoFrontLaunchSolution(
        gas_node_pressure_abs=chosen.gas_node_pressure_abs,
        liquid_node=liquid_node,
        east_traces=chosen.east.traces,
        vertical_traces=chosen.vertical.traces,
        west_gas_face_flux=chosen.west_gas_face_flux,
        west_gas_characteristic_mass_inflow=(
            chosen.west_gas_face_flux.mass_rate
        ),
        east_gas_volume_creation_rate=(
            chosen.east_gas_volume_creation_rate
        ),
        vertical_gas_volume_creation_rate=(
            chosen.vertical_gas_volume_creation_rate
        ),
        gas_volume_creation_rate=chosen.gas_volume_creation_rate,
        east_gas_mass_demand_rate=chosen.east_gas_mass_demand_rate,
        vertical_gas_mass_demand_rate=(
            chosen.vertical_gas_mass_demand_rate
        ),
        gas_mass_demand_rate=chosen.gas_mass_demand_rate,
        gas_mass_balance_residual=chosen.gas_mass_balance_residual,
        east_liquid_rh_residual=chosen.east_liquid_rh_residual,
        vertical_liquid_rh_residual=chosen.vertical_liquid_rh_residual,
        liquid_volume_balance_residual=(
            liquid_node.net_outward_volume_flux
        ),
        liquid_mass_balance_residual=liquid_node.net_outward_mass_flux,
        pressure_iterations=int(iterations),
    )


__all__ = [
    "PRODUCTION_READY",
    "LaunchTopologyError",
    "TwoFrontLaunchCandidate",
    "TwoFrontLaunchSolution",
    "WestGasFaceFlux",
    "evaluate_twofront_launch_candidate",
    "solve_twofront_launch_closure",
]
