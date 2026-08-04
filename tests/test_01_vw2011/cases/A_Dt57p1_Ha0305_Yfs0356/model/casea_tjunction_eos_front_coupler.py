"""Outer ideal-gas EOS/front closure for the resolved Case-A T junction.

This module deliberately owns no distributed finite-volume state.  A caller
supplies the currently resolved gas volumes, connected gas mass, two fitted
front states, and the *resolved* T-cell void and control-volume capacity.  The
single nonlinear unknown is the common connected-gas pressure ``p_J``.

At every trial pressure this module

1. evaluates all three conservative liquid face fluxes with the zero-storage
   T-node evaluator;
2. solves independent signed Rankine--Hugoniot problems for the east and
   vertical material fronts;
3. evaluates the atmospheric top gas Riemann flux when that face is open; and
4. closes the one-step isothermal EOS with the updated gas mass and resolved
   gas volume.

The liquid characteristics and the EOS are generally incompatible at a
strictly massless junction.  The only storage law used here is therefore

``V_T,g^(n+1) = V_T,g^n + dt * sum(q_liquid,out)``,

where both ``V_T,g^n`` and the T-cell capacity are caller-owned resolved FV
quantities.  There is no ``tee_total_volume`` parameter, fitted split, speed
cap, prescribed front motion, or presentation-time correction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from casea_coupled_gas_network import isothermal_ideal_gas_riemann_flux
from casea_tjunction_shock_network import (
    BranchGeometry,
    LiquidConservativeFaceFlux,
    MovingFrontState,
    PressureSolveError,
    StepSubdivisionRequired,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    ZeroStorageTNodeSolution,
    evaluate_zero_storage_t_node_at_pressure,
    solve_front_rankine_hugoniot,
    solve_zero_storage_t_node,
)
from tosan2021_horizontal_shockfit import TosanInterfaceSolution


class IncompatiblePrescribedPressure(PressureSolveError):
    """A prescribed pressure does not satisfy the one-step gas EOS."""

    def __init__(
        self,
        *,
        prescribed_pressure_abs: float,
        eos_residual: float,
        zero_storage_pressure_abs: float,
        net_outward_liquid_volume_flux: float,
    ) -> None:
        self.prescribed_pressure_abs = float(prescribed_pressure_abs)
        self.eos_residual = float(eos_residual)
        self.zero_storage_pressure_abs = float(zero_storage_pressure_abs)
        self.net_outward_liquid_volume_flux = float(
            net_outward_liquid_volume_flux
        )
        super().__init__(
            "prescribed common pressure is incompatible with the one-step "
            "resolved-volume gas EOS: "
            f"p={self.prescribed_pressure_abs:.9g} Pa, "
            f"EOS residual={self.eos_residual:.9g} Pa m^3, "
            f"p_zero={self.zero_storage_pressure_abs:.9g} Pa, "
            f"sum(q_out)={self.net_outward_liquid_volume_flux:.9g} m^3/s"
        )


@dataclass(frozen=True)
class ResolvedGasVolumes:
    """Caller-owned connected gas volumes at the beginning of a step."""

    west: float
    tee_void: float
    tee_control_volume: float
    east: float
    vertical: float

    def __post_init__(self) -> None:
        values = (
            self.west,
            self.tee_void,
            self.tee_control_volume,
            self.east,
            self.vertical,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("resolved gas volumes must be finite")
        if self.west <= 0.0 or self.east <= 0.0 or self.vertical <= 0.0:
            raise ValueError("resolved branch gas volumes must be positive")
        if self.tee_control_volume <= 0.0:
            raise ValueError("resolved T control-volume capacity must be positive")
        if not 0.0 < self.tee_void < self.tee_control_volume:
            raise ValueError(
                "resolved T void must lie strictly inside its FV capacity"
            )

    @property
    def total(self) -> float:
        return math.fsum((self.west, self.tee_void, self.east, self.vertical))


@dataclass(frozen=True)
class EOSFrontState:
    """Minimal uniquely owned state of the outer connected-gas closure."""

    time: float
    connected_gas_mass: float
    volumes: ResolvedGasVolumes
    east_front: MovingFrontState
    vertical_front: MovingFrontState
    cumulative_atmospheric_mass_out: float = 0.0


@dataclass(frozen=True)
class EOSFrontParameters:
    east: BranchGeometry
    vertical: BranchGeometry
    liquid_density: float = 998.0
    gravity: float = 9.81
    atmospheric_pressure: float = 101_325.0
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    entropy_fix_fraction: float = 0.10
    relative_eos_tolerance: float = 1.0e-10
    pressure_tolerance: float = 1.0e-7
    max_iterations: int = 100

    def __post_init__(self) -> None:
        values = (
            self.liquid_density,
            self.gravity,
            self.atmospheric_pressure,
            self.gas_constant,
            self.gas_temperature,
            self.entropy_fix_fraction,
            self.relative_eos_tolerance,
            self.pressure_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("EOS/front parameters must be finite")
        if min(values[:5]) <= 0.0:
            raise ValueError("physical EOS/front parameters must be positive")
        if self.entropy_fix_fraction < 0.0:
            raise ValueError("entropy-fix fraction must be non-negative")
        if self.relative_eos_tolerance <= 0.0 or self.pressure_tolerance <= 0.0:
            raise ValueError("nonlinear tolerances must be positive")
        if self.max_iterations < 1:
            raise ValueError("at least one nonlinear iteration is required")


@dataclass(frozen=True)
class AtmosphericTopFace:
    """Resolved top gas face, with the normal directed out of the network."""

    is_open: bool
    area: float
    network_outward_velocity: float
    atmospheric_velocity: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.area,
            self.network_outward_velocity,
            self.atmospheric_velocity,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("top-face data must be finite")
        if self.area <= 0.0:
            raise ValueError("top-face area must be positive")


@dataclass(frozen=True)
class GasConservativeFaceFlux:
    """Integrated gas flux on the outward atmospheric-top normal."""

    area: float
    mass_flux_per_area: float
    momentum_flux_per_area: float
    mass_rate: float
    momentum_flux: float


@dataclass(frozen=True)
class EOSFrontCandidate:
    """Complete one-step evaluation at one trial common pressure."""

    node_pressure_abs: float
    liquid_node: ZeroStorageTNodeSolution
    east_solution: TosanInterfaceSolution
    vertical_solution: TosanInterfaceSolution
    next_volumes: ResolvedGasVolumes
    next_connected_gas_mass: float
    top_gas_flux: GasConservativeFaceFlux
    top_mass_transfer: float
    eos_residual: float
    gas_ledger_residual: float


@dataclass(frozen=True)
class EOSFrontAdvance:
    state: EOSFrontState
    node_pressure_abs: float
    zero_storage_pressure_abs: float
    liquid_branch_face_fluxes: Mapping[str, LiquidConservativeFaceFlux]
    top_gas_flux: GasConservativeFaceFlux
    east_solution: TosanInterfaceSolution
    vertical_solution: TosanInterfaceSolution
    top_mass_transfer: float
    eos_residual: float
    gas_ledger_residual: float
    nonlinear_iterations: int


def _gas_area(
    front: MovingFrontState,
    geometry: BranchGeometry,
    *,
    gravity: float,
) -> float:
    section = geometry.section(gravity)
    return section.full_area - float(
        section.area_from_depth(front.free_surface_depth)
    )


def _validate_state(state: EOSFrontState, params: EOSFrontParameters) -> None:
    values = (
        state.time,
        state.connected_gas_mass,
        state.cumulative_atmospheric_mass_out,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("EOS/front state must be finite")
    if state.time < 0.0 or state.connected_gas_mass <= 0.0:
        raise ValueError("time must be non-negative and gas mass positive")
    state.east_front.validate(params.east)
    state.vertical_front.validate(params.vertical)


def connected_gas_pressure(
    state: EOSFrontState,
    params: EOSFrontParameters,
) -> float:
    """Return the current common pressure from the caller-owned inventory."""

    _validate_state(state, params)
    return (
        state.connected_gas_mass
        * params.gas_constant
        * params.gas_temperature
        / state.volumes.total
    )


def _top_flux_at_pressure(
    pressure: float,
    params: EOSFrontParameters,
    top_face: AtmosphericTopFace,
) -> GasConservativeFaceFlux:
    if not top_face.is_open:
        return GasConservativeFaceFlux(
            area=top_face.area,
            mass_flux_per_area=0.0,
            momentum_flux_per_area=0.0,
            mass_rate=0.0,
            momentum_flux=0.0,
        )
    density_network = pressure / (
        params.gas_constant * params.gas_temperature
    )
    density_atmosphere = params.atmospheric_pressure / (
        params.gas_constant * params.gas_temperature
    )
    mass_flux, momentum_flux = isothermal_ideal_gas_riemann_flux(
        density_network,
        top_face.network_outward_velocity,
        density_atmosphere,
        top_face.atmospheric_velocity,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
        entropy_fix_fraction=params.entropy_fix_fraction,
    )
    return GasConservativeFaceFlux(
        area=top_face.area,
        mass_flux_per_area=mass_flux,
        momentum_flux_per_area=momentum_flux,
        mass_rate=mass_flux * top_face.area,
        momentum_flux=momentum_flux * top_face.area,
    )


def evaluate_eos_front_candidate_at_pressure(
    state: EOSFrontState,
    params: EOSFrontParameters,
    characteristics: TeeLiquidCharacteristics,
    areas: ZeroStorageTBranchAreas,
    top_face: AtmosphericTopFace,
    *,
    dt: float,
    node_pressure_abs: float,
) -> EOSFrontCandidate:
    """Evaluate all liquid/gas faces, both RH fronts, and the EOS residual."""

    _validate_state(state, params)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(node_pressure_abs) or node_pressure_abs <= 0.0:
        raise ValueError("trial common pressure must be positive and finite")

    liquid_node = evaluate_zero_storage_t_node_at_pressure(
        characteristics,
        areas,
        node_pressure_abs=node_pressure_abs,
        liquid_density=params.liquid_density,
    )
    east_solution = solve_front_rankine_hugoniot(
        state.east_front,
        params.east,
        gas_pressure_abs=node_pressure_abs,
        atmospheric_pressure=params.atmospheric_pressure,
        liquid_density=params.liquid_density,
        gravity=params.gravity,
        free_surface_velocity=(
            liquid_node.branch_fluxes["east"].outward_velocity
        ),
        dt=dt,
        tolerance=params.relative_eos_tolerance,
        max_iterations=params.max_iterations,
    )
    vertical_solution = solve_front_rankine_hugoniot(
        state.vertical_front,
        params.vertical,
        gas_pressure_abs=node_pressure_abs,
        atmospheric_pressure=params.atmospheric_pressure,
        liquid_density=params.liquid_density,
        gravity=params.gravity,
        free_surface_velocity=(
            liquid_node.branch_fluxes["vertical"].outward_velocity
        ),
        dt=dt,
        tolerance=params.relative_eos_tolerance,
        max_iterations=params.max_iterations,
    )

    east_displacement = east_solution.interface_speed * dt
    vertical_displacement = vertical_solution.interface_speed * dt
    q_west = liquid_node.branch_fluxes["west"].volume_flux
    q_sum = liquid_node.net_outward_volume_flux
    volumes = state.volumes
    next_volumes = ResolvedGasVolumes(
        west=volumes.west - q_west * dt,
        tee_void=volumes.tee_void + q_sum * dt,
        tee_control_volume=volumes.tee_control_volume,
        east=(
            volumes.east
            + _gas_area(
                state.east_front, params.east, gravity=params.gravity
            )
            * east_displacement
        ),
        vertical=(
            volumes.vertical
            + _gas_area(
                state.vertical_front,
                params.vertical,
                gravity=params.gravity,
            )
            * vertical_displacement
        ),
    )
    top_flux = _top_flux_at_pressure(node_pressure_abs, params, top_face)
    top_mass_transfer = top_flux.mass_rate * dt
    next_mass = state.connected_gas_mass - top_mass_transfer
    if next_mass <= 0.0:
        raise StepSubdivisionRequired(
            "the atmospheric Riemann flux exhausts the connected gas mass"
        )
    east_position = state.east_front.position + east_displacement
    vertical_position = state.vertical_front.position + vertical_displacement
    if not 0.0 <= east_position <= params.east.length:
        raise StepSubdivisionRequired(
            "the east front reaches a branch boundary inside the step"
        )
    if not 0.0 <= vertical_position <= params.vertical.length:
        raise StepSubdivisionRequired(
            "the vertical front reaches a branch boundary inside the step"
        )

    eos_residual = (
        node_pressure_abs * next_volumes.total
        - next_mass * params.gas_constant * params.gas_temperature
    )
    gas_ledger_residual = (
        next_mass + top_mass_transfer - state.connected_gas_mass
    )
    return EOSFrontCandidate(
        node_pressure_abs=node_pressure_abs,
        liquid_node=liquid_node,
        east_solution=east_solution,
        vertical_solution=vertical_solution,
        next_volumes=next_volumes,
        next_connected_gas_mass=next_mass,
        top_gas_flux=top_flux,
        top_mass_transfer=top_mass_transfer,
        eos_residual=eos_residual,
        gas_ledger_residual=gas_ledger_residual,
    )


def _eos_scale(state: EOSFrontState, params: EOSFrontParameters) -> float:
    return max(
        state.connected_gas_mass
        * params.gas_constant
        * params.gas_temperature,
        1.0,
    )


def _solve_pressure(
    state: EOSFrontState,
    params: EOSFrontParameters,
    characteristics: TeeLiquidCharacteristics,
    areas: ZeroStorageTBranchAreas,
    top_face: AtmosphericTopFace,
    *,
    dt: float,
) -> tuple[EOSFrontCandidate, int]:
    pressure0 = connected_gas_pressure(state, params)
    zero_node = solve_zero_storage_t_node(
        characteristics,
        areas,
        liquid_density=params.liquid_density,
        pressure_hint_abs=pressure0,
        pressure_tolerance=params.pressure_tolerance,
        max_iterations=params.max_iterations,
    )
    seeds = sorted({pressure0, zero_node.node_pressure_abs})
    sampled: list[EOSFrontCandidate] = []

    def sample(pressure: float) -> None:
        if pressure <= 0.0 or any(
            math.isclose(
                pressure,
                item.node_pressure_abs,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for item in sampled
        ):
            return
        try:
            sampled.append(evaluate_eos_front_candidate_at_pressure(
                state,
                params,
                characteristics,
                areas,
                top_face,
                dt=dt,
                node_pressure_abs=pressure,
            ))
        except (PressureSolveError, StepSubdivisionRequired, ValueError):
            return

    for pressure in seeds:
        sample(pressure)
    scale = _eos_scale(state, params)
    if sampled:
        best = min(sampled, key=lambda item: abs(item.eos_residual))
        if abs(best.eos_residual) <= params.relative_eos_tolerance * scale:
            return best, 1

    delta = max(1.0, 1.0e-6 * pressure0)
    bracket: tuple[EOSFrontCandidate, EOSFrontCandidate] | None = None
    for _ in range(50):
        sample(max(1.0, pressure0 - delta))
        sample(pressure0 + delta)
        sampled.sort(key=lambda item: item.node_pressure_abs)
        for left, right in zip(sampled[:-1], sampled[1:]):
            if left.eos_residual * right.eos_residual <= 0.0:
                bracket = (left, right)
                break
        if bracket is not None:
            break
        delta *= 2.0
    if bracket is None:
        raise PressureSolveError(
            "could not bracket the resolved-volume one-step EOS pressure"
        )

    left, right = bracket
    for iteration in range(1, params.max_iterations + 1):
        middle_pressure = 0.5 * (
            left.node_pressure_abs + right.node_pressure_abs
        )
        middle = evaluate_eos_front_candidate_at_pressure(
            state,
            params,
            characteristics,
            areas,
            top_face,
            dt=dt,
            node_pressure_abs=middle_pressure,
        )
        if (
            abs(middle.eos_residual)
            <= params.relative_eos_tolerance * scale
            or right.node_pressure_abs - left.node_pressure_abs
            <= params.pressure_tolerance
        ):
            return middle, iteration + len(sampled)
        if left.eos_residual * middle.eos_residual <= 0.0:
            right = middle
        else:
            left = middle
    raise PressureSolveError("one-step EOS pressure solve did not converge")


def advance_eos_front_coupler(
    state: EOSFrontState,
    params: EOSFrontParameters,
    characteristics: TeeLiquidCharacteristics,
    areas: ZeroStorageTBranchAreas,
    top_face: AtmosphericTopFace,
    *,
    dt: float,
    prescribed_pressure_abs: float | None = None,
) -> EOSFrontAdvance:
    """Advance the independent EOS/front closure by one physical time step."""

    _validate_state(state, params)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    zero_node = solve_zero_storage_t_node(
        characteristics,
        areas,
        liquid_density=params.liquid_density,
        pressure_hint_abs=connected_gas_pressure(state, params),
        pressure_tolerance=params.pressure_tolerance,
        max_iterations=params.max_iterations,
    )
    if prescribed_pressure_abs is None:
        candidate, iterations = _solve_pressure(
            state,
            params,
            characteristics,
            areas,
            top_face,
            dt=dt,
        )
    else:
        if (
            not math.isfinite(prescribed_pressure_abs)
            or prescribed_pressure_abs <= 0.0
        ):
            raise ValueError("prescribed pressure must be positive and finite")
        candidate = evaluate_eos_front_candidate_at_pressure(
            state,
            params,
            characteristics,
            areas,
            top_face,
            dt=dt,
            node_pressure_abs=prescribed_pressure_abs,
        )
        tolerance = params.relative_eos_tolerance * _eos_scale(state, params)
        if abs(candidate.eos_residual) > tolerance:
            raise IncompatiblePrescribedPressure(
                prescribed_pressure_abs=prescribed_pressure_abs,
                eos_residual=candidate.eos_residual,
                zero_storage_pressure_abs=zero_node.node_pressure_abs,
                net_outward_liquid_volume_flux=(
                    candidate.liquid_node.net_outward_volume_flux
                ),
            )
        iterations = 1

    east_position = (
        state.east_front.position
        + candidate.east_solution.interface_speed * dt
    )
    vertical_position = (
        state.vertical_front.position
        + candidate.vertical_solution.interface_speed * dt
    )
    next_state = EOSFrontState(
        time=state.time + dt,
        connected_gas_mass=candidate.next_connected_gas_mass,
        volumes=candidate.next_volumes,
        east_front=replace(
            state.east_front,
            position=east_position,
            pressurised_head_foot=(
                candidate.east_solution.pressurised_head
            ),
            pressurised_velocity_foot=(
                candidate.east_solution.pressurised_velocity
            ),
        ),
        vertical_front=replace(
            state.vertical_front,
            position=vertical_position,
            pressurised_head_foot=(
                candidate.vertical_solution.pressurised_head
            ),
            pressurised_velocity_foot=(
                candidate.vertical_solution.pressurised_velocity
            ),
        ),
        cumulative_atmospheric_mass_out=(
            state.cumulative_atmospheric_mass_out
            + candidate.top_mass_transfer
        ),
    )
    _validate_state(next_state, params)
    return EOSFrontAdvance(
        state=next_state,
        node_pressure_abs=candidate.node_pressure_abs,
        zero_storage_pressure_abs=zero_node.node_pressure_abs,
        liquid_branch_face_fluxes=dict(candidate.liquid_node.branch_fluxes),
        top_gas_flux=candidate.top_gas_flux,
        east_solution=candidate.east_solution,
        vertical_solution=candidate.vertical_solution,
        top_mass_transfer=candidate.top_mass_transfer,
        eos_residual=candidate.eos_residual,
        gas_ledger_residual=candidate.gas_ledger_residual,
        nonlinear_iterations=iterations,
    )


__all__ = [
    "AtmosphericTopFace",
    "EOSFrontAdvance",
    "EOSFrontCandidate",
    "EOSFrontParameters",
    "EOSFrontState",
    "GasConservativeFaceFlux",
    "IncompatiblePrescribedPressure",
    "ResolvedGasVolumes",
    "advance_eos_front_coupler",
    "connected_gas_pressure",
    "evaluate_eos_front_candidate_at_pressure",
]
