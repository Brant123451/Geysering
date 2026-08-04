"""Finite-geometric-node closure for the Case-A two-front launch event.

All three liquid characteristics, both material-front RH problems, and the
west gas characteristic use one absolute junction pressure ``p_J``.  A
massless node would impose both ``sum(q_i)=0`` and an independent gas launch
balance on that one pressure and is generally over-constrained.  The missing
physical state is the resolved gas/liquid storage of the finite T control
volume.

For outward liquid branch rates ``q_i>0``, gas entering the node from the west
``m_dot_w>0``, and material fronts moving away from the node ``w_b>0``, one
implicit local step is

``V_g^{n+1} = V_g^n + dt sum(q_i)``,

``V_l^{n+1} = V_l^n - dt sum(q_i)``,

``m_J^{n+1} = m_J^n + dt [m_dot_w - rho_J sum(A_g,b w_b)]``,

``p_J V_g^{n+1} = m_J^{n+1} c_g^2``.

The east/vertical receiver mass is subtracted once from node mass and reported
separately in the total gas ledger; it is not included again in the node EOS.
The total node volume is an explicit caller-owned geometric input.  This
module neither infers it from a branch diameter nor fits it to a result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_gas_coupled_front import GasCellTrace
from casea_material_front_cutcell import InterfaceTraces, PressurisedState
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
    evaluate_zero_storage_t_node_at_pressure,
)


# This first finite-node draft stores only gas volume and gas mass.  It cannot
# preserve the pressurised liquid mass-equivalent inventory ``integral(A dx)``
# at the Case-A topology event.  In particular, a liquid-full node with
# ``A>A_f`` would lose its elastic storage and launch a spurious pressure
# impulse.  Keep the module as a regression/diagnostic implementation, but do
# not allow a production integrator to select it.  The replacement is
# ``casea_compressible_finite_node``.
PRODUCTION_READY = False


class FiniteNodeTopologyError(PressureSolveError):
    """A trial/accepted step leaves the geometric node state inadmissible."""


@dataclass(frozen=True)
class FiniteNodeGasState:
    """Resolved gas storage in one explicitly geometric T control volume."""

    gas_volume: float
    gas_mass: float
    node_total_volume: float

    def __post_init__(self) -> None:
        values = (self.gas_volume, self.gas_mass, self.node_total_volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("finite-node gas state must be finite")
        if self.node_total_volume <= 0.0:
            raise ValueError("node_total_volume must be a positive geometry")
        if self.gas_volume < 0.0 or self.gas_volume >= self.node_total_volume:
            raise ValueError(
                "gas volume must lie in [0, node_total_volume)"
            )
        if self.gas_mass < 0.0:
            raise ValueError("gas mass cannot be negative")
        if (self.gas_volume == 0.0) != (self.gas_mass == 0.0):
            raise ValueError(
                "exact-zero launch requires both gas volume and mass to be zero"
            )

    @property
    def liquid_volume(self) -> float:
        return float(self.node_total_volume - self.gas_volume)


@dataclass(frozen=True)
class FiniteNodeWestGasFlux:
    """West gas characteristic flux, positive from the west branch to node."""

    area: float
    pressure_abs: float
    density: float
    velocity: float
    mass_flux_per_area: float
    momentum_flux_per_area: float
    mass_rate: float
    momentum_rate: float


@dataclass(frozen=True)
class FiniteNodeLaunchCandidate:
    """One same-pressure evaluation of every local conservation equation."""

    pressure_abs: float
    liquid_node: ZeroStorageTNodeSolution
    east: CaseAFixedPressureFrontTraceResult
    vertical: CaseAFixedPressureFrontTraceResult
    west_gas_flux: FiniteNodeWestGasFlux
    liquid_outward_volume_rate: float
    east_receiver_volume_rate: float
    vertical_receiver_volume_rate: float
    receiver_volume_rate: float
    east_receiver_mass_rate: float
    vertical_receiver_mass_rate: float
    receiver_mass_rate: float
    next_node_gas_volume: float
    next_node_liquid_volume: float
    next_node_gas_mass: float
    eos_residual: float
    nonlinear_residual: float
    east_liquid_rh_residual: float
    vertical_liquid_rh_residual: float
    liquid_storage_balance_residual: float
    gas_node_mass_balance_residual: float
    total_gas_mass_balance_residual: float


@dataclass(frozen=True)
class FiniteNodeLaunchSolution:
    """Accepted one-step finite-node launch closure."""

    state: FiniteNodeGasState
    pressure_abs: float
    liquid_node: ZeroStorageTNodeSolution
    east_traces: InterfaceTraces
    vertical_traces: InterfaceTraces
    west_gas_flux: FiniteNodeWestGasFlux
    liquid_outward_volume_rate: float
    east_receiver_volume_rate: float
    vertical_receiver_volume_rate: float
    receiver_volume_rate: float
    east_receiver_mass_rate: float
    vertical_receiver_mass_rate: float
    receiver_mass_rate: float
    eos_residual: float
    nonlinear_residual: float
    east_liquid_rh_residual: float
    vertical_liquid_rh_residual: float
    liquid_storage_balance_residual: float
    gas_node_mass_balance_residual: float
    total_gas_mass_balance_residual: float
    pressure_iterations: int


def _west_gas_flux(
    trace: GasCellTrace,
    *,
    pressure_abs: float,
    face_area: float,
) -> FiniteNodeWestGasFlux:
    density = pressure_abs / trace.sound_speed**2
    velocity = trace.right_boundary_outflow_velocity(pressure_abs)
    if abs(velocity) >= trace.sound_speed:
        raise PressureSolveError(
            "west gas boundary left the subsonic characteristic domain; "
            "the velocity was not capped"
        )
    mass_flux_per_area = density * velocity
    momentum_flux_per_area = density * velocity**2 + pressure_abs
    return FiniteNodeWestGasFlux(
        area=float(face_area),
        pressure_abs=float(pressure_abs),
        density=float(density),
        velocity=float(velocity),
        mass_flux_per_area=float(mass_flux_per_area),
        momentum_flux_per_area=float(momentum_flux_per_area),
        mass_rate=float(face_area * mass_flux_per_area),
        momentum_rate=float(face_area * momentum_flux_per_area),
    )


def evaluate_finite_node_launch_candidate(
    state: FiniteNodeGasState,
    *,
    dt: float,
    pressure_abs: float,
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
    east_pressurised_friction_slope: float = 0.0,
    vertical_pressurised_friction_slope: float = 0.0,
) -> FiniteNodeLaunchCandidate:
    """Evaluate one trial pressure with no state mutation or clipping."""

    values = (
        dt,
        pressure_abs,
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
        east_pressurised_friction_slope,
        vertical_pressurised_friction_slope,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("finite-node launch inputs must be finite")
    if min(
        dt,
        pressure_abs,
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
    ) <= 0.0:
        raise ValueError("finite-node launch scales and dt must be positive")

    # This evaluator does not enforce zero storage.  It evaluates the three
    # characteristic fluxes at the one physical gas-node pressure; their sum
    # is the resolved node void-volume rate below.
    liquid_node = evaluate_zero_storage_t_node_at_pressure(
        liquid_characteristics,
        liquid_areas,
        node_pressure_abs=pressure_abs,
        liquid_density=liquid_density,
    )
    east = build_casea_material_front_traces_at_pressure(
        east_pressurised_foot,
        stratified_liquid_area=east_stratified_liquid_area,
        free_surface_velocity=(
            liquid_node.branch_fluxes["east"].outward_velocity
        ),
        gas_pressure_abs=pressure_abs,
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
        gas_pressure_abs=pressure_abs,
        front_position=0.0,
        geometry=vertical_geometry,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        gas_sound_speed=west_gas_trace.sound_speed,
        dt=dt,
        pressurised_friction_slope=vertical_pressurised_friction_slope,
    )
    west_flux = _west_gas_flux(
        west_gas_trace,
        pressure_abs=pressure_abs,
        face_area=west_gas_face_area,
    )

    q_liquid = liquid_node.net_outward_volume_flux
    next_gas_volume = state.gas_volume + dt * q_liquid
    next_liquid_volume = state.liquid_volume - dt * q_liquid
    if (
        next_gas_volume <= 0.0
        or next_gas_volume >= state.node_total_volume
        or next_liquid_volume <= 0.0
    ):
        raise FiniteNodeTopologyError(
            "trial pressure exhausts one phase of the finite geometric node"
        )

    east_volume_rate = east.gas_area * east.traces.speed
    vertical_volume_rate = vertical.gas_area * vertical.traces.speed
    receiver_volume_rate = math.fsum(
        (east_volume_rate, vertical_volume_rate)
    )
    density_node = pressure_abs / west_gas_trace.sound_speed**2
    east_mass_rate = density_node * east_volume_rate
    vertical_mass_rate = density_node * vertical_volume_rate
    receiver_mass_rate = math.fsum((east_mass_rate, vertical_mass_rate))
    next_gas_mass = state.gas_mass + dt * (
        west_flux.mass_rate - receiver_mass_rate
    )
    if next_gas_mass <= 0.0:
        raise FiniteNodeTopologyError(
            "trial pressure leaves non-positive gas mass in the finite node"
        )

    eos_residual = (
        pressure_abs * next_gas_volume
        - next_gas_mass * west_gas_trace.sound_speed**2
    )
    exact_zero_launch = state.gas_volume == 0.0
    nonlinear_residual = (
        eos_residual / dt if exact_zero_launch else eos_residual
    )
    liquid_storage_residual = (
        (next_liquid_volume - state.liquid_volume) + dt * q_liquid
    )
    gas_node_residual = (
        (next_gas_mass - state.gas_mass)
        - dt * (west_flux.mass_rate - receiver_mass_rate)
    )
    # Receiver gas is an external new inventory.  Adding it once to the node
    # change must recover the west characteristic inflow exactly.
    total_gas_residual = (
        (next_gas_mass - state.gas_mass)
        + dt * receiver_mass_rate
        - dt * west_flux.mass_rate
    )
    return FiniteNodeLaunchCandidate(
        pressure_abs=float(pressure_abs),
        liquid_node=liquid_node,
        east=east,
        vertical=vertical,
        west_gas_flux=west_flux,
        liquid_outward_volume_rate=float(q_liquid),
        east_receiver_volume_rate=float(east_volume_rate),
        vertical_receiver_volume_rate=float(vertical_volume_rate),
        receiver_volume_rate=float(receiver_volume_rate),
        east_receiver_mass_rate=float(east_mass_rate),
        vertical_receiver_mass_rate=float(vertical_mass_rate),
        receiver_mass_rate=float(receiver_mass_rate),
        next_node_gas_volume=float(next_gas_volume),
        next_node_liquid_volume=float(next_liquid_volume),
        next_node_gas_mass=float(next_gas_mass),
        eos_residual=float(eos_residual),
        nonlinear_residual=float(nonlinear_residual),
        east_liquid_rh_residual=float(east.liquid.residual_linf),
        vertical_liquid_rh_residual=float(vertical.liquid.residual_linf),
        liquid_storage_balance_residual=float(liquid_storage_residual),
        gas_node_mass_balance_residual=float(gas_node_residual),
        total_gas_mass_balance_residual=float(total_gas_residual),
    )


def solve_finite_node_twofront_launch(
    state: FiniteNodeGasState,
    *,
    dt: float,
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
    east_pressurised_friction_slope: float = 0.0,
    vertical_pressurised_friction_slope: float = 0.0,
    pressure_hint_abs: float | None = None,
    residual_absolute_tolerance: float = 1.0e-12,
    residual_relative_tolerance: float = 1.0e-10,
    pressure_tolerance: float = 1.0e-7,
    launch_speed_tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> FiniteNodeLaunchSolution:
    """Advance the finite T storage through one same-pressure implicit step."""

    scalars = (
        dt,
        west_gas_face_area,
        atmospheric_pressure,
        liquid_density,
        gravity,
        residual_absolute_tolerance,
        residual_relative_tolerance,
        pressure_tolerance,
        launch_speed_tolerance,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in scalars):
        raise ValueError("finite-node solver scales and dt must be positive")
    if max_iterations < 1:
        raise ValueError("at least one pressure iteration is required")
    if pressure_hint_abs is not None and (
        not math.isfinite(pressure_hint_abs) or pressure_hint_abs <= 0.0
    ):
        raise ValueError("pressure hint must be positive and finite")

    def evaluate(pressure: float) -> FiniteNodeLaunchCandidate | None:
        if not math.isfinite(pressure) or pressure <= 0.0:
            return None
        try:
            return evaluate_finite_node_launch_candidate(
                state,
                dt=dt,
                pressure_abs=pressure,
                west_gas_trace=west_gas_trace,
                west_gas_face_area=west_gas_face_area,
                liquid_characteristics=liquid_characteristics,
                liquid_areas=liquid_areas,
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
                east_pressurised_friction_slope=(
                    east_pressurised_friction_slope
                ),
                vertical_pressurised_friction_slope=(
                    vertical_pressurised_friction_slope
                ),
            )
        except (PressureSolveError, ValueError, FloatingPointError):
            return None

    def residual_scale(candidate: FiniteNodeLaunchCandidate) -> float:
        if state.gas_volume == 0.0:
            return max(
                abs(candidate.pressure_abs * candidate.liquid_outward_volume_rate),
                abs(
                    west_gas_trace.sound_speed**2
                    * (
                        candidate.west_gas_flux.mass_rate
                        - candidate.receiver_mass_rate
                    )
                ),
                1.0e-30,
            )
        return max(
            abs(candidate.pressure_abs * candidate.next_node_gas_volume),
            abs(
                candidate.next_node_gas_mass
                * west_gas_trace.sound_speed**2
            ),
            1.0e-30,
        )

    def accepted(candidate: FiniteNodeLaunchCandidate) -> bool:
        return abs(candidate.nonlinear_residual) <= (
            residual_absolute_tolerance
            + residual_relative_tolerance * residual_scale(candidate)
        )

    centres = {west_gas_trace.pressure_abs}
    if pressure_hint_abs is not None:
        centres.add(float(pressure_hint_abs))
    if state.gas_volume > 0.0:
        centres.add(
            state.gas_mass
            * west_gas_trace.sound_speed**2
            / state.gas_volume
        )
    centre_pressure = pressure_hint_abs or west_gas_trace.pressure_abs
    samples: list[FiniteNodeLaunchCandidate] = []
    chosen: FiniteNodeLaunchCandidate | None = None
    iterations = 0
    for pressure in sorted(centres):
        candidate = evaluate(pressure)
        if candidate is None:
            continue
        if accepted(candidate):
            chosen = candidate
            iterations = 1
            break
        samples.append(candidate)

    if chosen is None:
        delta = max(1.0, 1.0e-6 * centre_pressure)
        bracket: tuple[
            FiniteNodeLaunchCandidate, FiniteNodeLaunchCandidate
        ] | None = None
        for _ in range(64):
            for pressure in (
                max(math.nextafter(0.0, 1.0), centre_pressure - delta),
                centre_pressure + delta,
            ):
                if any(
                    math.isclose(
                        pressure,
                        item.pressure_abs,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for item in samples
                ):
                    continue
                candidate = evaluate(pressure)
                if candidate is not None:
                    samples.append(candidate)
            samples.sort(key=lambda item: item.pressure_abs)
            for left, right in zip(samples[:-1], samples[1:]):
                if left.nonlinear_residual * right.nonlinear_residual <= 0.0:
                    bracket = (left, right)
                    break
            if bracket is not None:
                break
            delta *= 2.0
        if bracket is None:
            detail = "no admissible finite-node pressure samples"
            if samples:
                best = min(samples, key=lambda item: abs(item.nonlinear_residual))
                detail = (
                    f"best residual={best.nonlinear_residual:.9g} at "
                    f"p={best.pressure_abs:.9g} Pa"
                )
            raise PressureSolveError(
                "could not bracket the finite-node EOS closure; " + detail
            )

        left, right = bracket
        chosen = min(
            (left, right), key=lambda item: abs(item.nonlinear_residual)
        )
        for iterations in range(1, max_iterations + 1):
            pressure = 0.5 * (left.pressure_abs + right.pressure_abs)
            middle = evaluate(pressure)
            if middle is None:
                raise PressureSolveError(
                    "finite-node state became inadmissible inside its EOS bracket"
                )
            chosen = middle
            if accepted(middle):
                break
            if left.nonlinear_residual * middle.nonlinear_residual <= 0.0:
                right = middle
            else:
                left = middle
            if right.pressure_abs - left.pressure_abs <= pressure_tolerance:
                denominator = (
                    right.nonlinear_residual - left.nonlinear_residual
                )
                if denominator != 0.0:
                    secant_pressure = (
                        left.pressure_abs
                        - left.nonlinear_residual
                        * (right.pressure_abs - left.pressure_abs)
                        / denominator
                    )
                    secant = evaluate(secant_pressure)
                    if secant is not None:
                        chosen = secant
                        if accepted(secant):
                            break
        else:
            raise PressureSolveError(
                "finite-node launch pressure did not converge"
            )

    if not accepted(chosen):
        raise PressureSolveError(
            "finite-node EOS residual exceeds tolerance: "
            f"{chosen.nonlinear_residual:.9g}"
        )
    speeds = (chosen.east.traces.speed, chosen.vertical.traces.speed)
    if any(speed < -launch_speed_tolerance for speed in speeds):
        raise FiniteNodeTopologyError(
            "a zero-length receiver has a receding RH root; no speed was clipped"
        )

    next_state = FiniteNodeGasState(
        gas_volume=chosen.next_node_gas_volume,
        gas_mass=chosen.next_node_gas_mass,
        node_total_volume=state.node_total_volume,
    )
    return FiniteNodeLaunchSolution(
        state=next_state,
        pressure_abs=chosen.pressure_abs,
        liquid_node=chosen.liquid_node,
        east_traces=chosen.east.traces,
        vertical_traces=chosen.vertical.traces,
        west_gas_flux=chosen.west_gas_flux,
        liquid_outward_volume_rate=chosen.liquid_outward_volume_rate,
        east_receiver_volume_rate=chosen.east_receiver_volume_rate,
        vertical_receiver_volume_rate=chosen.vertical_receiver_volume_rate,
        receiver_volume_rate=chosen.receiver_volume_rate,
        east_receiver_mass_rate=chosen.east_receiver_mass_rate,
        vertical_receiver_mass_rate=chosen.vertical_receiver_mass_rate,
        receiver_mass_rate=chosen.receiver_mass_rate,
        eos_residual=chosen.eos_residual,
        nonlinear_residual=chosen.nonlinear_residual,
        east_liquid_rh_residual=chosen.east_liquid_rh_residual,
        vertical_liquid_rh_residual=chosen.vertical_liquid_rh_residual,
        liquid_storage_balance_residual=(
            chosen.liquid_storage_balance_residual
        ),
        gas_node_mass_balance_residual=(
            chosen.gas_node_mass_balance_residual
        ),
        total_gas_mass_balance_residual=(
            chosen.total_gas_mass_balance_residual
        ),
        pressure_iterations=int(iterations),
    )


__all__ = [
    "PRODUCTION_READY",
    "FiniteNodeGasState",
    "FiniteNodeLaunchCandidate",
    "FiniteNodeLaunchSolution",
    "FiniteNodeTopologyError",
    "FiniteNodeWestGasFlux",
    "evaluate_finite_node_launch_candidate",
    "solve_finite_node_twofront_launch",
]
