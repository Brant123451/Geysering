"""Build Case-A ALE cut-cell traces from the physical RH interface solve.

Case A has the stratified gas--liquid state on the T side and pressurised
liquid ahead of each east/upward material front (``S | P``).  This adapter
converts the two branch states to the local variables used by the existing
Rankine--Hugoniot solver, couples the gas acoustic boundary characteristic,
and returns the common ALE traces consumed by
``casea_material_front_cutcell``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_gas_coupled_front import (
    GasCellTrace,
    GasCoupledFrontSolution,
    solve_gas_coupled_material_front,
)
from casea_material_front_cutcell import (
    InterfaceTraces,
    PressurisedFlux,
    PressurisedState,
    StratifiedFlux,
    StratifiedState,
)
from casea_tjunction_shock_network import (
    BranchGeometry,
    MovingFrontState,
    solve_front_rankine_hugoniot,
)
from tosan2021_horizontal_shockfit import TosanInterfaceSolution


@dataclass(frozen=True)
class CaseAFrontTraceResult:
    traces: InterfaceTraces
    closure: GasCoupledFrontSolution
    gas_area: float
    gas_density: float
    liquid_depth: float


@dataclass(frozen=True)
class CaseAFixedPressureFrontTraceResult:
    """One ``S | P`` trace evaluated at a shared gas-node pressure.

    This is the trace form needed when more than one newly opened material
    front shares one junction pressure.  The pressure is supplied by the
    outer junction balance; it is not solved independently at each branch.
    """

    traces: InterfaceTraces
    liquid: TosanInterfaceSolution
    gas_pressure_abs: float
    gas_area: float
    gas_density: float
    liquid_depth: float


def _assemble_traces(
    solution: TosanInterfaceSolution,
    *,
    geometry: BranchGeometry,
    section,
    gas_pressure_abs: float,
    gas_sound_speed: float,
    atmospheric_pressure: float,
    liquid_density: float,
    gravity: float,
    fallback_liquid_depth: float,
    fallback_liquid_velocity: float,
) -> tuple[InterfaceTraces, float, float, float]:
    """Convert one accepted RH solution to conservative physical traces."""

    star_depth = (
        fallback_liquid_depth
        if solution.free_surface_depth is None
        else float(solution.free_surface_depth)
    )
    star_liquid_area = float(section.area_from_depth(star_depth))
    star_liquid_velocity = (
        fallback_liquid_velocity
        if solution.free_surface_velocity is None
        else float(solution.free_surface_velocity)
    )
    star_liquid_discharge = star_liquid_area * star_liquid_velocity
    area_full = float(section.full_area)
    star_gas_area = area_full - star_liquid_area
    if star_gas_area <= 0.0:
        raise ValueError("RH solution leaves no stratified gas area")
    star_gas_density = gas_pressure_abs / gas_sound_speed**2
    star_gas_mass = star_gas_density * star_gas_area
    speed = float(solution.interface_speed)
    star_gas_momentum = star_gas_mass * speed

    # The Tosan RH closure uses A_f at the liquid-full interface trace.  The
    # pressurised cell foot may carry elastic storage A!=A_f, but the material
    # surface itself is the full pipe section and its pressure is represented
    # by H_p,Gamma in the momentum flux.
    pressurised_star = PressurisedState(
        area=area_full,
        discharge=area_full * solution.pressurised_velocity,
    )
    pressurised_flux = PressurisedFlux(
        area=pressurised_star.discharge,
        momentum=(
            area_full * solution.pressurised_velocity**2
            + gravity
            * area_full
            * (solution.pressurised_head - 0.5 * geometry.diameter)
        ),
    )
    stratified_star = StratifiedState(
        gas_mass=star_gas_mass,
        gas_momentum=star_gas_momentum,
        liquid_area=star_liquid_area,
        liquid_discharge=star_liquid_discharge,
    )
    gas_gauge_pressure = gas_pressure_abs - atmospheric_pressure
    stratified_flux = StratifiedFlux(
        gas_mass=star_gas_momentum,
        gas_momentum=(
            star_gas_mass * speed**2
            + gas_gauge_pressure * star_gas_area
        ),
        liquid_area=star_liquid_discharge,
        liquid_momentum=(
            star_liquid_area * star_liquid_velocity**2
            + gravity * float(section.hydrostatic_moment(star_depth))
            + gas_gauge_pressure * area_full / liquid_density
        ),
    )
    return (
        InterfaceTraces(
            speed=speed,
            pressurised_state=pressurised_star,
            pressurised_flux=pressurised_flux,
            stratified_state=stratified_star,
            stratified_flux=stratified_flux,
        ),
        star_gas_area,
        star_gas_density,
        star_depth,
    )


def build_casea_material_front_traces_at_pressure(
    pressurised_foot: PressurisedState,
    *,
    stratified_liquid_area: float,
    free_surface_velocity: float,
    gas_pressure_abs: float,
    front_position: float,
    geometry: BranchGeometry,
    atmospheric_pressure: float = 101_325.0,
    liquid_density: float = 998.0,
    gravity: float = 9.81,
    gas_sound_speed: float = math.sqrt(287.05 * 293.0),
    dt: float = 0.0,
    pressurised_friction_slope: float = 0.0,
) -> CaseAFixedPressureFrontTraceResult:
    """Build one branch trace at an externally shared gas pressure.

    The function owns no pressure solve and applies no launch-speed rule.  It
    evaluates the existing liquid Rankine--Hugoniot closure and gives the gas
    trace the material velocity ``u_g=w`` required by the ALE interface.
    """

    values = (
        stratified_liquid_area,
        free_surface_velocity,
        gas_pressure_abs,
        front_position,
        atmospheric_pressure,
        liquid_density,
        gravity,
        gas_sound_speed,
        dt,
        pressurised_friction_slope,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fixed-pressure material-front inputs must be finite")
    if min(
        gas_pressure_abs,
        atmospheric_pressure,
        liquid_density,
        gravity,
        gas_sound_speed,
    ) <= 0.0:
        raise ValueError("fixed-pressure physical scales must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    section = geometry.section(gravity)
    if not 0.0 < stratified_liquid_area < section.full_area:
        raise ValueError(
            "stratified liquid area must lie strictly between zero and full"
        )
    liquid_depth = float(section.depth_from_area(stratified_liquid_area))
    local_front = MovingFrontState(
        position=float(front_position),
        free_surface_depth=liquid_depth,
        pressurised_head_foot=float(
            section.head_from_area(pressurised_foot.area)
        ),
        pressurised_velocity_foot=(
            pressurised_foot.discharge / pressurised_foot.area
        ),
        friction_slope=float(pressurised_friction_slope),
    )
    solution = solve_front_rankine_hugoniot(
        local_front,
        geometry,
        gas_pressure_abs=gas_pressure_abs,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        free_surface_velocity=free_surface_velocity,
        dt=dt,
    )
    traces, gas_area, gas_density, star_depth = _assemble_traces(
        solution,
        geometry=geometry,
        section=section,
        gas_pressure_abs=gas_pressure_abs,
        gas_sound_speed=gas_sound_speed,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        fallback_liquid_depth=liquid_depth,
        fallback_liquid_velocity=free_surface_velocity,
    )
    return CaseAFixedPressureFrontTraceResult(
        traces=traces,
        liquid=solution,
        gas_pressure_abs=float(gas_pressure_abs),
        gas_area=gas_area,
        gas_density=gas_density,
        liquid_depth=star_depth,
    )


def build_casea_material_front_traces(
    pressurised_foot: PressurisedState,
    stratified_foot: StratifiedState,
    *,
    front_position: float,
    geometry: BranchGeometry,
    atmospheric_pressure: float = 101_325.0,
    liquid_density: float = 998.0,
    gravity: float = 9.81,
    gas_sound_speed: float = math.sqrt(287.05 * 293.0),
    dt: float = 0.0,
    pressurised_friction_slope: float = 0.0,
) -> CaseAFrontTraceResult:
    """Return a gas-coupled ``S | P`` material-front trace.

    No state is clipped into the physical range.  A non-positive gas area or
    a front position outside the branch is rejected by the constitutive/front
    state constructors.
    """

    if not all(
        math.isfinite(value)
        for value in (
            front_position,
            atmospheric_pressure,
            liquid_density,
            gravity,
            gas_sound_speed,
            dt,
            pressurised_friction_slope,
        )
    ):
        raise ValueError("material-front adapter inputs must be finite")
    if min(
        atmospheric_pressure, liquid_density, gravity, gas_sound_speed
    ) <= 0.0:
        raise ValueError("material-front physical scales must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")

    section = geometry.section(gravity)
    area_full = section.full_area
    gas_area_foot = area_full - stratified_foot.liquid_area
    if gas_area_foot <= 0.0:
        raise ValueError("stratified foot must contain a finite gas area")
    gas_density_foot = stratified_foot.gas_mass / gas_area_foot
    liquid_depth_foot = float(
        section.depth_from_area(stratified_foot.liquid_area)
    )
    pressurised_head_foot = float(
        section.head_from_area(pressurised_foot.area)
    )
    local_front = MovingFrontState(
        position=float(front_position),
        free_surface_depth=liquid_depth_foot,
        pressurised_head_foot=pressurised_head_foot,
        pressurised_velocity_foot=(
            pressurised_foot.discharge / pressurised_foot.area
        ),
        friction_slope=float(pressurised_friction_slope),
    )
    closure = solve_gas_coupled_material_front(
        local_front,
        geometry,
        gas_trace=GasCellTrace(
            density=gas_density_foot,
            velocity=stratified_foot.gas_velocity,
            sound_speed=gas_sound_speed,
        ),
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        free_surface_velocity=(
            stratified_foot.liquid_discharge
            / stratified_foot.liquid_area
        ),
        dt=dt,
    )

    fallback_liquid_velocity = (
        stratified_foot.liquid_discharge / stratified_foot.liquid_area
    )
    traces, star_gas_area, star_gas_density, star_depth = _assemble_traces(
        closure.liquid,
        geometry=geometry,
        section=section,
        gas_pressure_abs=closure.gas_pressure_abs,
        gas_sound_speed=gas_sound_speed,
        atmospheric_pressure=atmospheric_pressure,
        liquid_density=liquid_density,
        gravity=gravity,
        fallback_liquid_depth=liquid_depth_foot,
        fallback_liquid_velocity=fallback_liquid_velocity,
    )
    return CaseAFrontTraceResult(
        traces=traces,
        closure=closure,
        gas_area=star_gas_area,
        gas_density=star_gas_density,
        liquid_depth=star_depth,
    )


__all__ = [
    "CaseAFixedPressureFrontTraceResult",
    "CaseAFrontTraceResult",
    "build_casea_material_front_traces",
    "build_casea_material_front_traces_at_pressure",
]
