"""Conservative post-arrival shock network for the Case-A side T.

The pre-arrival Case-A solution contains one fitted gas/water front in an
unbranched horizontal pipe.  That topology ceases to exist when the front
reaches the side T: there is then a west gas pocket, a finite intersection
control volume, an east dead-leg front, and a vertical/open branch.  Advancing
the original single ``interface_x`` through the T cannot represent that graph.

This module is the small, independent junction core for the topology after
arrival.  Its sign convention is:

* ``q_w, q_e, q_v > 0``: liquid leaves the tee and enters the named branch;
* ``s_dot_e, s_dot_v > 0``: a material front moves away from the tee;
* ``m_dot_top > 0``: gas leaves the network through the open riser top.

The connected gas is acoustically lumped.  This is justified here because its
acoustic crossing time is much shorter than a liquid/front time step.  West,
tee, east, and vertical gas volumes consequently share one pressure

``p_J = m_g R T / (V_w + V_T + V_e + V_v)``.

There is no prescribed branch split.  At a trial common pressure, each liquid
branch evaluates its own incoming characteristic and physical minor loss.  The
east and vertical fronts then solve characteristic compatibility plus the full
Rankine--Hugoniot mass and momentum jumps.  The scalar EOS residual closes the
node pressure.  The east-front solution is signed and may advance, stagnate,
or recede; no Froude cap, target speed, time window, or result-dependent source
appears in this module.

Only the atmospheric top face changes total gas mass.  Internal redistribution
is implicit in the common-pressure limit and therefore needs no arbitrary gas
mass split ratio.  A step that crosses a branch end or exhausts a physical
control-volume capacity raises :class:`StepSubdivisionRequired`; it is never
silently clipped to an admissible-looking result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np

from casea_coupled_gas_network import isothermal_ideal_gas_riemann_flux
from tosan2021_horizontal_shockfit import (
    CircularSection,
    TosanInterfaceData,
    TosanInterfaceSolution,
    solve_oriented_interface,
)


class PressureSolveError(RuntimeError):
    """The common-pressure EOS/RH nonlinear problem did not converge."""


class StepSubdivisionRequired(RuntimeError):
    """A physical topology/capacity event lies inside the requested step."""


class IncompatibleZeroStoragePressure(PressureSolveError):
    """A prescribed gas pressure cannot satisfy a zero-storage liquid node.

    If a gas EOS fixes ``p_J`` while the three incoming liquid
    characteristics give a nonzero sum of outward volume fluxes, the missing
    equation is a junction storage/interface-motion law.  Silently altering a
    branch flux or pressure would prescribe a split.  This exception exposes
    the required storage rate instead.
    """

    def __init__(
        self,
        *,
        prescribed_pressure_abs: float,
        zero_storage_pressure_abs: float,
        net_outward_volume_flux: float,
    ) -> None:
        self.prescribed_pressure_abs = float(prescribed_pressure_abs)
        self.zero_storage_pressure_abs = float(zero_storage_pressure_abs)
        self.net_outward_volume_flux = float(net_outward_volume_flux)
        # Positive outward liquid flux empties liquid from a finite tee and
        # would increase its gas volume at exactly this rate.
        self.required_tee_gas_volume_rate = float(net_outward_volume_flux)
        super().__init__(
            "prescribed gas pressure is incompatible with a zero-storage "
            "three-branch liquid node: "
            f"p_g={self.prescribed_pressure_abs:.9g} Pa, "
            f"p_zero={self.zero_storage_pressure_abs:.9g} Pa, "
            f"sum(q_out)={self.net_outward_volume_flux:.9g} m^3/s"
        )


@dataclass(frozen=True)
class BranchGeometry:
    """Geometry and wave data for one moving-front branch.

    ``bed_slope`` is zero for the horizontal east dead leg and one for a
    vertical coordinate directed upward.  The pressurised liquid is ahead of
    both Case-A gas fronts, hence the common ``pressurised_side='right'`` RH
    orientation used below.
    """

    diameter: float
    length: float
    wave_speed: float
    bed_slope: float = 0.0

    def __post_init__(self) -> None:
        values = (self.diameter, self.length, self.wave_speed, self.bed_slope)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("branch geometry must be finite")
        if self.diameter <= 0.0 or self.length <= 0.0:
            raise ValueError("branch diameter and length must be positive")
        if self.wave_speed <= 0.0:
            raise ValueError("branch wave speed must be positive")

    def section(self, gravity: float) -> CircularSection:
        return CircularSection(self.diameter, gravity, self.wave_speed)


@dataclass(frozen=True)
class MovingFrontState:
    """One branch-local gas/free-surface--pressurised-liquid front."""

    position: float
    free_surface_depth: float
    pressurised_head_foot: float
    pressurised_velocity_foot: float = 0.0
    friction_slope: float = 0.0

    def validate(self, geometry: BranchGeometry) -> None:
        values = (
            self.position,
            self.free_surface_depth,
            self.pressurised_head_foot,
            self.pressurised_velocity_foot,
            self.friction_slope,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("moving-front state must be finite")
        if not 0.0 <= self.position <= geometry.length:
            raise ValueError("front position lies outside its physical branch")
        if not 0.0 < self.free_surface_depth < geometry.diameter:
            raise ValueError(
                "the fitted free-surface trace must be strictly partial"
            )


@dataclass(frozen=True)
class LiquidCharacteristic:
    """Incoming liquid characteristic written in an outward branch frame.

    The boundary relation is

    ``(p_l,J-p*)/rho = c (u-u*) + K u|u|/2``.

    ``pressure_offset`` converts the common gas-node pressure to the local
    liquid pressure at the characteristic datum (for example a hydrostatic
    centroid offset).  The quadratic loss is solved analytically and is
    monotone; no velocity clipping is used.
    """

    reference_pressure_abs: float
    reference_outward_velocity: float
    wave_speed: float
    loss_coefficient: float = 0.0
    pressure_offset: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.reference_pressure_abs,
            self.reference_outward_velocity,
            self.wave_speed,
            self.loss_coefficient,
            self.pressure_offset,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("liquid characteristic must be finite")
        if self.reference_pressure_abs <= 0.0 or self.wave_speed <= 0.0:
            raise ValueError("characteristic pressure and wave speed must be positive")
        if self.loss_coefficient < 0.0:
            raise ValueError("minor-loss coefficient must be non-negative")

    def outward_velocity(
        self,
        gas_node_pressure_abs: float,
        *,
        liquid_density: float,
    ) -> float:
        """Return the signed characteristic velocity at the common node."""

        if gas_node_pressure_abs <= 0.0 or liquid_density <= 0.0:
            raise ValueError("node pressure and liquid density must be positive")
        rhs = (
            self.wave_speed * self.reference_outward_velocity
            + (
                gas_node_pressure_abs
                + self.pressure_offset
                - self.reference_pressure_abs
            )
            / liquid_density
        )
        if self.loss_coefficient == 0.0:
            return rhs / self.wave_speed
        magnitude = (
            2.0 * abs(rhs)
            / (
                self.wave_speed
                + math.sqrt(
                    self.wave_speed**2
                    + 2.0 * self.loss_coefficient * abs(rhs)
                )
            )
        )
        return math.copysign(magnitude, rhs)


@dataclass(frozen=True)
class TeeLiquidCharacteristics:
    west: LiquidCharacteristic
    east: LiquidCharacteristic
    vertical: LiquidCharacteristic
    west_liquid_area: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.west_liquid_area):
            raise ValueError("west liquid area must be finite")
        if self.west_liquid_area <= 0.0:
            raise ValueError("west liquid area must be positive")


@dataclass(frozen=True)
class ZeroStorageTBranchAreas:
    """Liquid face areas of the west, east, and vertical T branches."""

    west: float
    east: float
    vertical: float

    def __post_init__(self) -> None:
        values = (self.west, self.east, self.vertical)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all zero-storage branch areas must be positive")

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "west": float(self.west),
            "east": float(self.east),
            "vertical": float(self.vertical),
        }


@dataclass(frozen=True)
class LiquidConservativeFaceFlux:
    """Complete liquid flux in a coordinate directed away from the node.

    ``volume_flux`` and ``mass_flux`` are signed.  The normal momentum tensor
    flux ``rho*A*u^2 + p*A`` is positive in the local outward coordinate even
    for inflow, as required by the conservative momentum equation.  The
    ``momentum_flux_increment`` subtracts the incoming characteristic's static
    pressure face and is therefore exactly zero in hydrostatic rest; it is a
    useful well-balanced integration diagnostic, not a replacement flux.
    """

    area: float
    outward_velocity: float
    face_pressure_abs: float
    volume_flux: float
    mass_flux: float
    advective_momentum_flux: float
    pressure_force: float
    total_momentum_flux: float
    kinematic_momentum_flux: float
    momentum_flux_increment: float


@dataclass(frozen=True)
class ZeroStorageTNodeSolution:
    """One massless T-node boundary state shared by all three branches."""

    node_pressure_abs: float
    branch_fluxes: Mapping[str, LiquidConservativeFaceFlux]
    net_outward_volume_flux: float
    net_outward_mass_flux: float
    nonlinear_iterations: int



@dataclass(frozen=True)
class TJunctionParameters:
    east: BranchGeometry
    vertical: BranchGeometry
    tee_total_volume: float
    atmospheric_pressure: float = 101_325.0
    liquid_density: float = 998.0
    gravity: float = 9.81
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    vertical_outlet_area: float | None = None
    entropy_fix_fraction: float = 0.10
    nonlinear_tolerance: float = 1.0e-10
    nonlinear_max_iterations: int = 80

    def __post_init__(self) -> None:
        values = (
            self.tee_total_volume,
            self.atmospheric_pressure,
            self.liquid_density,
            self.gravity,
            self.gas_constant,
            self.gas_temperature,
            self.entropy_fix_fraction,
            self.nonlinear_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("T-junction parameters must be finite")
        if min(values[:6]) <= 0.0:
            raise ValueError("physical T-junction parameters must be positive")
        if self.entropy_fix_fraction < 0.0:
            raise ValueError("entropy-fix fraction must be non-negative")
        if self.nonlinear_tolerance <= 0.0:
            raise ValueError("nonlinear tolerance must be positive")
        if self.nonlinear_max_iterations < 1:
            raise ValueError("at least one nonlinear iteration is required")
        if self.vertical_outlet_area is not None:
            if (
                not math.isfinite(self.vertical_outlet_area)
                or self.vertical_outlet_area <= 0.0
            ):
                raise ValueError("vertical outlet area must be positive")


@dataclass(frozen=True)
class TJunctionShockState:
    """Post-arrival state of the acoustically connected gas graph."""

    time: float
    gas_mass: float
    west_gas_volume: float
    tee_gas_volume: float
    east_front: MovingFrontState
    vertical_front: MovingFrontState
    atmospheric_mass_exchange: float = 0.0

    def validate(self, params: TJunctionParameters) -> None:
        values = (
            self.time,
            self.gas_mass,
            self.west_gas_volume,
            self.tee_gas_volume,
            self.atmospheric_mass_exchange,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("T-junction state must be finite")
        if self.time < 0.0 or self.gas_mass <= 0.0:
            raise ValueError("time must be non-negative and gas mass positive")
        if self.west_gas_volume <= 0.0:
            raise ValueError("west gas volume must be positive")
        if not 0.0 < self.tee_gas_volume < params.tee_total_volume:
            raise ValueError("tee gas volume must lie inside its physical CV")
        self.east_front.validate(params.east)
        self.vertical_front.validate(params.vertical)


@dataclass(frozen=True)
class FrontAdvance:
    state: MovingFrontState
    solution: TosanInterfaceSolution | None


@dataclass(frozen=True)
class TJunctionShockAdvance:
    state: TJunctionShockState
    node_pressure_abs: float
    liquid_branch_flows: Mapping[str, float]
    branch_gas_masses: Mapping[str, float]
    branch_boundary_pressures_abs: Mapping[str, float]
    east: FrontAdvance
    vertical: FrontAdvance
    top_mass_transfer: float
    gas_mass_conservation_error: float
    eos_residual: float
    nonlinear_iterations: int


@dataclass(frozen=True)
class _Candidate:
    pressure: float
    gas_mass: float
    west_volume: float
    tee_volume: float
    east_position: float
    vertical_position: float
    east_solution: TosanInterfaceSolution
    vertical_solution: TosanInterfaceSolution | None
    flows: Mapping[str, float]
    top_mass_transfer: float
    total_volume: float
    residual: float


def _branch_gas_area(
    front: MovingFrontState,
    geometry: BranchGeometry,
    *,
    gravity: float,
) -> float:
    section = geometry.section(gravity)
    liquid_area = float(section.area_from_depth(front.free_surface_depth))
    return section.full_area - liquid_area


def _total_gas_volume(
    state: TJunctionShockState,
    params: TJunctionParameters,
) -> float:
    east_area = _branch_gas_area(
        state.east_front, params.east, gravity=params.gravity
    )
    vertical_area = _branch_gas_area(
        state.vertical_front, params.vertical, gravity=params.gravity
    )
    return (
        state.west_gas_volume
        + state.tee_gas_volume
        + east_area * state.east_front.position
        + vertical_area * state.vertical_front.position
    )


def common_node_pressure(
    state: TJunctionShockState,
    params: TJunctionParameters,
) -> float:
    """Return the unique connected-gas pressure from mass and geometry."""

    state.validate(params)
    volume = _total_gas_volume(state, params)
    if volume <= 0.0:
        raise ValueError("connected gas volume must be positive")
    return state.gas_mass * params.gas_constant * params.gas_temperature / volume


def evaluate_zero_storage_t_node_at_pressure(
    characteristics: TeeLiquidCharacteristics,
    areas: ZeroStorageTBranchAreas,
    *,
    node_pressure_abs: float,
    liquid_density: float,
) -> ZeroStorageTNodeSolution:
    """Evaluate all conservative liquid face fluxes at one trial ``p_J``.

    This function deliberately does not change gas mass, gas volume, a fitted
    front, or any branch state.  It is the residual evaluator needed by an
    outer gas-EOS/front solve: ``net_outward_volume_flux`` is the zero-storage
    residual.  All branch coordinates point away from the T, so conservation
    requires ``q_w + q_e + q_v = 0``.
    """

    if not math.isfinite(node_pressure_abs) or node_pressure_abs <= 0.0:
        raise ValueError("trial node pressure must be positive and finite")
    if not math.isfinite(liquid_density) or liquid_density <= 0.0:
        raise ValueError("liquid density must be positive and finite")

    characteristic_map = {
        "west": characteristics.west,
        "east": characteristics.east,
        "vertical": characteristics.vertical,
    }
    area_map = areas.as_mapping()
    fluxes: dict[str, LiquidConservativeFaceFlux] = {}
    for name in ("west", "east", "vertical"):
        characteristic = characteristic_map[name]
        area = area_map[name]
        face_pressure = node_pressure_abs + characteristic.pressure_offset
        if face_pressure <= 0.0:
            raise ValueError(
                f"{name} liquid face pressure is not positive at this trial"
            )
        velocity = characteristic.outward_velocity(
            node_pressure_abs,
            liquid_density=liquid_density,
        )
        volume_flux = area * velocity
        mass_flux = liquid_density * volume_flux
        advective = liquid_density * area * velocity * velocity
        pressure_force = face_pressure * area
        total_momentum = advective + pressure_force
        kinematic_momentum = (
            volume_flux * velocity
            + face_pressure * area / liquid_density
        )
        momentum_increment = (
            advective
            + (
                face_pressure
                - characteristic.reference_pressure_abs
            )
            * area
        )
        fluxes[name] = LiquidConservativeFaceFlux(
            area=area,
            outward_velocity=velocity,
            face_pressure_abs=face_pressure,
            volume_flux=volume_flux,
            mass_flux=mass_flux,
            advective_momentum_flux=advective,
            pressure_force=pressure_force,
            total_momentum_flux=total_momentum,
            kinematic_momentum_flux=kinematic_momentum,
            momentum_flux_increment=momentum_increment,
        )

    net_volume = math.fsum(
        flux.volume_flux for flux in fluxes.values()
    )
    net_mass = math.fsum(flux.mass_flux for flux in fluxes.values())
    return ZeroStorageTNodeSolution(
        node_pressure_abs=float(node_pressure_abs),
        branch_fluxes=fluxes,
        net_outward_volume_flux=net_volume,
        net_outward_mass_flux=net_mass,
        nonlinear_iterations=0,
    )


def solve_zero_storage_t_node(
    characteristics: TeeLiquidCharacteristics,
    areas: ZeroStorageTBranchAreas,
    *,
    liquid_density: float,
    pressure_hint_abs: float | None = None,
    required_gas_pressure_abs: float | None = None,
    volume_flux_tolerance: float = 1.0e-12,
    pressure_tolerance: float = 1.0e-7,
    max_iterations: int = 100,
) -> ZeroStorageTNodeSolution:
    """Solve the massless three-liquid-branch node pressure and face fluxes.

    The unknown is the one shared pressure ``p_J``.  Each incoming
    characteristic supplies a monotone relation ``q_i(p_J)``; zero junction
    storage supplies the closure ``sum(q_i)=0``.  Thus the liquid-only problem
    is square and has at most one positive-pressure solution.

    A gas EOS pressure is an *additional* equation.  Pass it as
    ``required_gas_pressure_abs`` only when an outer gas/front solve expects it
    to be compatible.  If its three characteristic fluxes do not sum to zero,
    this function raises :class:`IncompatibleZeroStoragePressure` and reports
    the tee gas-volume rate that would be required to close the equations.
    It never changes a branch flux to force agreement.  For an arbitrary EOS
    trial pressure, use :func:`evaluate_zero_storage_t_node_at_pressure` and
    return its volume-flux residual to the outer nonlinear solve.
    """

    scalars = (liquid_density, volume_flux_tolerance, pressure_tolerance)
    if not all(math.isfinite(value) and value > 0.0 for value in scalars):
        raise ValueError("node density and solver tolerances must be positive")
    if max_iterations < 1:
        raise ValueError("at least one zero-storage node iteration is required")
    optional_pressures = (pressure_hint_abs, required_gas_pressure_abs)
    if any(
        value is not None
        and (not math.isfinite(value) or value <= 0.0)
        for value in optional_pressures
    ):
        raise ValueError("optional node pressures must be positive and finite")

    characteristic_values = (
        characteristics.west,
        characteristics.east,
        characteristics.vertical,
    )
    # Keep every local absolute liquid pressure positive during bracketing.
    lower = max(
        1.0,
        *(
            1.0 - characteristic.pressure_offset
            for characteristic in characteristic_values
        ),
    )
    pressure_scale = max(
        *(
            characteristic.reference_pressure_abs
            - characteristic.pressure_offset
            + liquid_density
            * characteristic.wave_speed
            * abs(characteristic.reference_outward_velocity)
            for characteristic in characteristic_values
        ),
        pressure_hint_abs or 0.0,
        required_gas_pressure_abs or 0.0,
        lower + 1.0,
    )
    upper = max(2.0 * pressure_scale, lower + 1.0)

    lower_state = evaluate_zero_storage_t_node_at_pressure(
        characteristics,
        areas,
        node_pressure_abs=lower,
        liquid_density=liquid_density,
    )
    if abs(lower_state.net_outward_volume_flux) <= volume_flux_tolerance:
        zero_state = lower_state
        iterations = 1
    else:
        if lower_state.net_outward_volume_flux > 0.0:
            raise PressureSolveError(
                "the three incoming liquid characteristics require a "
                "non-positive absolute pressure for zero storage"
            )
        upper_state = evaluate_zero_storage_t_node_at_pressure(
            characteristics,
            areas,
            node_pressure_abs=upper,
            liquid_density=liquid_density,
        )
        for _ in range(40):
            if upper_state.net_outward_volume_flux >= 0.0:
                break
            upper *= 2.0
            upper_state = evaluate_zero_storage_t_node_at_pressure(
                characteristics,
                areas,
                node_pressure_abs=upper,
                liquid_density=liquid_density,
            )
        else:
            raise PressureSolveError(
                "could not bracket the positive zero-storage node pressure"
            )

        zero_state = upper_state
        for iterations in range(1, max_iterations + 1):
            middle = 0.5 * (lower + upper)
            middle_state = evaluate_zero_storage_t_node_at_pressure(
                characteristics,
                areas,
                node_pressure_abs=middle,
                liquid_density=liquid_density,
            )
            zero_state = middle_state
            if (
                abs(middle_state.net_outward_volume_flux)
                <= volume_flux_tolerance
                or upper - lower <= pressure_tolerance
            ):
                break
            if middle_state.net_outward_volume_flux > 0.0:
                upper = middle
                upper_state = middle_state
            else:
                lower = middle
                lower_state = middle_state
        else:
            raise PressureSolveError(
                "zero-storage node pressure did not converge"
            )

    zero_state = ZeroStorageTNodeSolution(
        node_pressure_abs=zero_state.node_pressure_abs,
        branch_fluxes=zero_state.branch_fluxes,
        net_outward_volume_flux=zero_state.net_outward_volume_flux,
        net_outward_mass_flux=zero_state.net_outward_mass_flux,
        nonlinear_iterations=iterations,
    )
    if required_gas_pressure_abs is None:
        return zero_state

    gas_state = evaluate_zero_storage_t_node_at_pressure(
        characteristics,
        areas,
        node_pressure_abs=required_gas_pressure_abs,
        liquid_density=liquid_density,
    )
    # Compatibility is the conservation equation itself.  Comparing two
    # independently terminated pressure roots can reject an exact static state
    # merely because the volume-flux tolerance was reached before the requested
    # pressure tolerance.
    if abs(gas_state.net_outward_volume_flux) > volume_flux_tolerance:
        raise IncompatibleZeroStoragePressure(
            prescribed_pressure_abs=required_gas_pressure_abs,
            zero_storage_pressure_abs=zero_state.node_pressure_abs,
            net_outward_volume_flux=gas_state.net_outward_volume_flux,
        )
    return ZeroStorageTNodeSolution(
        node_pressure_abs=gas_state.node_pressure_abs,
        branch_fluxes=gas_state.branch_fluxes,
        net_outward_volume_flux=gas_state.net_outward_volume_flux,
        net_outward_mass_flux=gas_state.net_outward_mass_flux,
        nonlinear_iterations=zero_state.nonlinear_iterations,
    )


def solve_front_rankine_hugoniot(
    front: MovingFrontState,
    geometry: BranchGeometry,
    *,
    gas_pressure_abs: float,
    atmospheric_pressure: float,
    liquid_density: float,
    gravity: float,
    free_surface_velocity: float,
    dt: float = 0.0,
    tolerance: float = 1.0e-10,
    max_iterations: int = 80,
) -> TosanInterfaceSolution:
    """Solve a signed branch front without a speed cap or direction lock."""

    front.validate(geometry)
    if gas_pressure_abs <= 0.0 or atmospheric_pressure <= 0.0:
        raise ValueError("front pressures must be positive")
    if liquid_density <= 0.0 or gravity <= 0.0 or dt < 0.0:
        raise ValueError("front material data and time step are invalid")
    if not math.isfinite(free_surface_velocity):
        raise ValueError("free-surface velocity must be finite")

    section = geometry.section(gravity)
    data = TosanInterfaceData(
        pressurised_velocity_foot=front.pressurised_velocity_foot,
        pressurised_head_foot=front.pressurised_head_foot,
        free_surface_velocity=free_surface_velocity,
        free_surface_depth=front.free_surface_depth,
        gas_pressure_head=(
            gas_pressure_abs - atmospheric_pressure
        ) / (liquid_density * gravity),
        dt=dt,
        pressurised_friction_slope=front.friction_slope,
        bed_slope=geometry.bed_slope,
    )
    solution = solve_oriented_interface(
        data,
        section=section,
        pressurised_side="right",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not solution.converged:
        raise PressureSolveError(
            "branch Rankine--Hugoniot solve did not converge"
        )
    return solution


def _candidate_at_pressure(
    state: TJunctionShockState,
    params: TJunctionParameters,
    characteristics: TeeLiquidCharacteristics,
    *,
    dt: float,
    pressure: float,
    vertical_open_height: float,
) -> _Candidate:
    east_section = params.east.section(params.gravity)
    vertical_section = params.vertical.section(params.gravity)
    east_liquid_area = float(
        east_section.area_from_depth(state.east_front.free_surface_depth)
    )
    vertical_liquid_area = float(
        vertical_section.area_from_depth(
            state.vertical_front.free_surface_depth
        )
    )
    u_w = characteristics.west.outward_velocity(
        pressure, liquid_density=params.liquid_density
    )
    u_e = characteristics.east.outward_velocity(
        pressure, liquid_density=params.liquid_density
    )
    u_v = characteristics.vertical.outward_velocity(
        pressure, liquid_density=params.liquid_density
    )
    flows = {
        "west": characteristics.west_liquid_area * u_w,
        "east": east_liquid_area * u_e,
        "vertical": vertical_liquid_area * u_v,
    }

    east_solution = solve_front_rankine_hugoniot(
        state.east_front,
        params.east,
        gas_pressure_abs=pressure,
        atmospheric_pressure=params.atmospheric_pressure,
        liquid_density=params.liquid_density,
        gravity=params.gravity,
        free_surface_velocity=u_e,
        dt=dt,
        tolerance=params.nonlinear_tolerance,
        max_iterations=params.nonlinear_max_iterations,
    )
    east_position = (
        state.east_front.position + east_solution.interface_speed * dt
    )

    vertical_is_open = bool(
        state.vertical_front.position
        >= vertical_open_height
        - 1.0e-14 * params.vertical.length
    )
    if vertical_is_open:
        vertical_solution = None
        # The material gas front opens when it meets the *current liquid
        # surface*.  The dry atmospheric headspace above that surface is not
        # part of the connected trapped-gas inventory and must not be added to
        # its EOS volume in one step.
        vertical_position = state.vertical_front.position
    else:
        vertical_solution = solve_front_rankine_hugoniot(
            state.vertical_front,
            params.vertical,
            gas_pressure_abs=pressure,
            atmospheric_pressure=params.atmospheric_pressure,
            liquid_density=params.liquid_density,
            gravity=params.gravity,
            free_surface_velocity=u_v,
            dt=dt,
            tolerance=params.nonlinear_tolerance,
            max_iterations=params.nonlinear_max_iterations,
        )
        vertical_position = (
            state.vertical_front.position
            + vertical_solution.interface_speed * dt
        )

    west_volume = state.west_gas_volume - flows["west"] * dt
    tee_volume = state.tee_gas_volume + math.fsum(flows.values()) * dt
    east_gas_area = east_section.full_area - east_liquid_area
    vertical_gas_area = vertical_section.full_area - vertical_liquid_area
    total_volume = (
        west_volume
        + tee_volume
        + east_gas_area * east_position
        + vertical_gas_area * vertical_position
    )

    top_mass_transfer = 0.0
    if vertical_is_open and dt > 0.0:
        density_network = pressure / (
            params.gas_constant * params.gas_temperature
        )
        density_atmosphere = params.atmospheric_pressure / (
            params.gas_constant * params.gas_temperature
        )
        mass_flux, _ = isothermal_ideal_gas_riemann_flux(
            density_network,
            0.0,
            density_atmosphere,
            0.0,
            gas_constant=params.gas_constant,
            temperature=params.gas_temperature,
            entropy_fix_fraction=params.entropy_fix_fraction,
        )
        outlet_area = (
            params.vertical_outlet_area
            if params.vertical_outlet_area is not None
            else vertical_gas_area
        )
        top_mass_transfer = mass_flux * outlet_area * dt
    gas_mass = state.gas_mass - top_mass_transfer
    residual = (
        pressure * total_volume
        - gas_mass * params.gas_constant * params.gas_temperature
    )
    return _Candidate(
        pressure=pressure,
        gas_mass=gas_mass,
        west_volume=west_volume,
        tee_volume=tee_volume,
        east_position=east_position,
        vertical_position=vertical_position,
        east_solution=east_solution,
        vertical_solution=vertical_solution,
        flows=flows,
        top_mass_transfer=top_mass_transfer,
        total_volume=total_volume,
        residual=residual,
    )


def _solve_common_pressure(
    state: TJunctionShockState,
    params: TJunctionParameters,
    characteristics: TeeLiquidCharacteristics,
    *,
    dt: float,
    vertical_open_height: float,
) -> tuple[_Candidate, int]:
    pressure0 = common_node_pressure(state, params)
    candidate0 = _candidate_at_pressure(
        state,
        params,
        characteristics,
        dt=dt,
        pressure=pressure0,
        vertical_open_height=vertical_open_height,
    )
    residual_scale = max(
        abs(state.gas_mass * params.gas_constant * params.gas_temperature),
        1.0,
    )
    if abs(candidate0.residual) <= params.nonlinear_tolerance * residual_scale:
        return candidate0, 1

    # A stable coupled step changes the gas pressure only locally.  Search
    # outward from the old EOS state rather than first evaluating at p/2 and
    # 2p: those remote trial states can lie on a different RH branch even when
    # the physical root is only a few pascals away.  This is a nonlinear-solver
    # bracket, not a bound on the accepted pressure.
    sampled: list[_Candidate] = [candidate0]
    delta = max(1.0, 1.0e-6 * pressure0)
    bracket: tuple[_Candidate, _Candidate] | None = None
    for _ in range(40):
        for trial_pressure in (
            max(1.0, pressure0 - delta),
            pressure0 + delta,
        ):
            if any(
                math.isclose(
                    trial_pressure,
                    item.pressure,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for item in sampled
            ):
                continue
            try:
                sampled.append(_candidate_at_pressure(
                    state,
                    params,
                    characteristics,
                    dt=dt,
                    pressure=trial_pressure,
                    vertical_open_height=vertical_open_height,
                ))
            except PressureSolveError:
                # A remote RH state may be inadmissible.  Adjacent successful
                # samples are still sufficient to bracket the EOS root.
                pass
        sampled.sort(key=lambda item: item.pressure)
        for left, right in zip(sampled[:-1], sampled[1:], strict=True):
            if left.residual * right.residual <= 0.0:
                bracket = (left, right)
                break
        if bracket is not None:
            break
        delta *= 2.0
    if bracket is None:
        raise PressureSolveError("could not bracket the common node pressure")
    lower_candidate, upper_candidate = bracket
    lower = lower_candidate.pressure
    upper = upper_candidate.pressure

    best = candidate0
    for iteration in range(1, params.nonlinear_max_iterations + 1):
        pressure = 0.5 * (lower + upper)
        candidate = _candidate_at_pressure(
            state,
            params,
            characteristics,
            dt=dt,
            pressure=pressure,
            vertical_open_height=vertical_open_height,
        )
        best = candidate
        if abs(candidate.residual) <= (
            params.nonlinear_tolerance * residual_scale
        ):
            return candidate, iteration + 3
        if lower_candidate.residual * candidate.residual <= 0.0:
            upper = pressure
            upper_candidate = candidate
        else:
            lower = pressure
            lower_candidate = candidate
    raise PressureSolveError(
        "common node pressure did not reach the requested EOS tolerance; "
        f"last residual={best.residual:.6e}"
    )


def _validate_completed_candidate(
    candidate: _Candidate,
    state: TJunctionShockState,
    params: TJunctionParameters,
    *,
    vertical_open_height: float,
) -> None:
    if candidate.gas_mass <= 0.0:
        raise StepSubdivisionRequired(
            "the atmospheric gas flux exhausts the connected inventory"
        )
    if candidate.west_volume <= 0.0:
        raise StepSubdivisionRequired("the west gas pocket closes inside the step")
    if not 0.0 < candidate.tee_volume < params.tee_total_volume:
        raise StepSubdivisionRequired(
            "the finite tee control volume changes topology inside the step"
        )
    positions = (
        ("east", candidate.east_position, params.east.length),
        ("vertical", candidate.vertical_position, params.vertical.length),
    )
    for name, position, length in positions:
        if position < 0.0 or position > length:
            raise StepSubdivisionRequired(
                f"the {name} front reaches a branch boundary inside the step"
            )
    vertical_tolerance = 1.0e-12 * params.vertical.length
    if (
        state.vertical_front.position
        < vertical_open_height - vertical_tolerance
        and candidate.vertical_position
        > vertical_open_height + vertical_tolerance
    ):
        raise StepSubdivisionRequired(
            "the vertical gas front reaches the current liquid surface "
            "inside the step"
        )
    if candidate.total_volume <= 0.0:
        raise StepSubdivisionRequired("connected gas volume vanishes inside the step")


def advance_tjunction_shock_network(
    state: TJunctionShockState,
    params: TJunctionParameters,
    characteristics: TeeLiquidCharacteristics,
    *,
    dt: float,
    vertical_liquid_surface_height: float | None = None,
) -> TJunctionShockAdvance:
    """Advance the conservative three-branch T-node by one implicit step.

    The only nonlinear unknown shared by all branches is the connected-gas
    pressure.  For each trial pressure the three liquid characteristic flows
    and the two signed RH fronts are recomputed; the ideal-gas EOS then closes
    the scalar residual.  ``vertical_liquid_surface_height`` is the current
    resolved top of the liquid column.  Gas becomes atmospheric when its
    material front reaches that moving surface, not only when it reaches the
    geometric riser rim.  This is a physical node solve, not a prescribed split
    or a post-processing correction.
    """

    state.validate(params)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    vertical_open_height = (
        params.vertical.length
        if vertical_liquid_surface_height is None
        else float(vertical_liquid_surface_height)
    )
    if (
        not math.isfinite(vertical_open_height)
        or vertical_open_height < 0.0
        or vertical_open_height > params.vertical.length
    ):
        raise ValueError(
            "vertical liquid-surface height must lie inside the riser"
        )
    candidate, iterations = _solve_common_pressure(
        state,
        params,
        characteristics,
        dt=dt,
        vertical_open_height=vertical_open_height,
    )
    _validate_completed_candidate(
        candidate,
        state,
        params,
        vertical_open_height=vertical_open_height,
    )

    east_state = replace(
        state.east_front,
        position=candidate.east_position,
        pressurised_head_foot=(
            candidate.east_solution.pressurised_head
        ),
        pressurised_velocity_foot=(
            candidate.east_solution.pressurised_velocity
        ),
    )
    if candidate.vertical_solution is None:
        vertical_state = state.vertical_front
    else:
        vertical_state = replace(
            state.vertical_front,
            position=candidate.vertical_position,
            pressurised_head_foot=(
                candidate.vertical_solution.pressurised_head
            ),
            pressurised_velocity_foot=(
                candidate.vertical_solution.pressurised_velocity
            ),
        )
    next_state = TJunctionShockState(
        time=state.time + dt,
        gas_mass=candidate.gas_mass,
        west_gas_volume=candidate.west_volume,
        tee_gas_volume=candidate.tee_volume,
        east_front=east_state,
        vertical_front=vertical_state,
        atmospheric_mass_exchange=(
            state.atmospheric_mass_exchange + candidate.top_mass_transfer
        ),
    )
    next_state.validate(params)

    volume_parts = {
        "west": next_state.west_gas_volume,
        "tee": next_state.tee_gas_volume,
        "east": _branch_gas_area(
            next_state.east_front, params.east, gravity=params.gravity
        )
        * next_state.east_front.position,
        "vertical": _branch_gas_area(
            next_state.vertical_front,
            params.vertical,
            gravity=params.gravity,
        )
        * next_state.vertical_front.position,
    }
    density = candidate.pressure / (
        params.gas_constant * params.gas_temperature
    )
    branch_masses = {
        name: density * volume for name, volume in volume_parts.items()
    }
    pressure_map = {
        name: candidate.pressure for name in volume_parts
    }
    mass_error = (
        next_state.gas_mass
        + candidate.top_mass_transfer
        - state.gas_mass
    )
    eos_residual = (
        candidate.pressure * math.fsum(volume_parts.values())
        - next_state.gas_mass
        * params.gas_constant
        * params.gas_temperature
    )
    return TJunctionShockAdvance(
        state=next_state,
        node_pressure_abs=candidate.pressure,
        liquid_branch_flows=dict(candidate.flows),
        branch_gas_masses=branch_masses,
        branch_boundary_pressures_abs=pressure_map,
        east=FrontAdvance(east_state, candidate.east_solution),
        vertical=FrontAdvance(vertical_state, candidate.vertical_solution),
        top_mass_transfer=candidate.top_mass_transfer,
        gas_mass_conservation_error=mass_error,
        eos_residual=eos_residual,
        nonlinear_iterations=iterations,
    )


__all__ = [
    "BranchGeometry",
    "FrontAdvance",
    "IncompatibleZeroStoragePressure",
    "LiquidConservativeFaceFlux",
    "LiquidCharacteristic",
    "MovingFrontState",
    "PressureSolveError",
    "StepSubdivisionRequired",
    "TJunctionParameters",
    "TJunctionShockAdvance",
    "TJunctionShockState",
    "TeeLiquidCharacteristics",
    "ZeroStorageTBranchAreas",
    "ZeroStorageTNodeSolution",
    "advance_tjunction_shock_network",
    "common_node_pressure",
    "evaluate_zero_storage_t_node_at_pressure",
    "solve_front_rankine_hugoniot",
    "solve_zero_storage_t_node",
]
