"""Subsonic isothermal-gas Riemann core for the Case-A T junction.

Each branch coordinate points *away* from the junction.  For a subsonic
interior trace ``(rho_i, u_i)`` the characteristic entering the junction is

    u_J,i = u_i + c log(rho_J / rho_i).

The common junction density is obtained from the zero-storage mass balance

    sum_i rho_J A_i u_J,i = 0.

Because ``rho_J`` is common and positive, this scalar closure has an analytic
solution.  No branch allocation fraction, relaxation, clipping, or empirical
flow limiter is used.  The returned flux signs follow the outward branch
coordinates: a negative mass flux enters the junction and a positive mass
flux leaves it.

This module deliberately owns only the instantaneous gas junction Riemann
problem.  It does not advance cells, move material fronts, or prescribe a
coupling history.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class GasTJunctionError(ValueError):
    """Base class for invalid or inadmissible gas-junction states."""


class NonPositiveGasStateError(GasTJunctionError):
    """A density, area, or sound speed is non-positive."""


class SupersonicGasTraceError(GasTJunctionError):
    """An interior trace is not strictly subsonic."""


class NoAdmissibleSubsonicJunctionError(GasTJunctionError):
    """The characteristic closure has no strictly subsonic junction state."""


@dataclass(frozen=True)
class GasBranchTrace:
    """Interior trace of one branch in its outward local coordinate.

    ``tracer_fraction`` is optional globally: either all three branch traces
    provide it or all three omit it.
    """

    density: float
    outward_velocity: float
    area: float
    tracer_fraction: float | None = None


@dataclass(frozen=True)
class GasBranchJunctionFlux:
    """Junction state and outward numerical flux for one branch."""

    density: float
    outward_velocity: float
    area: float
    mass_flux: float
    momentum_flux: float
    tracer_flux: float | None


@dataclass(frozen=True)
class IsothermalGasTJunctionSolution:
    """Unique admissible subsonic solution of the three-branch closure."""

    common_density: float
    common_pressure: float
    sound_speed: float
    west: GasBranchJunctionFlux
    east: GasBranchJunctionFlux
    vertical: GasBranchJunctionFlux
    junction_tracer_fraction: float | None
    mass_residual: float
    tracer_residual: float | None

    @property
    def branches(self) -> tuple[GasBranchJunctionFlux, ...]:
        """Return branches in the fixed ``west, east, vertical`` order."""

        return (self.west, self.east, self.vertical)


def velocity_in_outward_coordinate(
    axis_velocity: float,
    outward_axis_sign: int,
) -> float:
    """Convert an axis-oriented velocity to a branch-outward velocity.

    For example, if global ``x`` is positive to the east, the west branch has
    ``outward_axis_sign=-1`` while the east branch has ``+1``.  Requiring an
    explicit sign prevents a silent west-branch coordinate reversal.
    """

    velocity = float(axis_velocity)
    if not math.isfinite(velocity):
        raise GasTJunctionError("axis velocity must be finite")
    if outward_axis_sign not in (-1, 1):
        raise GasTJunctionError("outward axis sign must be exactly -1 or +1")
    return float(outward_axis_sign) * velocity


def _validate_trace(
    trace: GasBranchTrace,
    *,
    branch_name: str,
    sound_speed: float,
) -> None:
    values = (trace.density, trace.outward_velocity, trace.area)
    if not all(math.isfinite(float(value)) for value in values):
        raise GasTJunctionError(f"{branch_name} trace must be finite")
    if trace.density <= 0.0:
        raise NonPositiveGasStateError(
            f"{branch_name} density must be strictly positive"
        )
    if trace.area <= 0.0:
        raise NonPositiveGasStateError(
            f"{branch_name} area must be strictly positive"
        )
    if abs(trace.outward_velocity) >= sound_speed:
        raise SupersonicGasTraceError(
            f"{branch_name} trace must satisfy |u_out| < c"
        )
    if trace.tracer_fraction is not None:
        fraction = float(trace.tracer_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise GasTJunctionError(
                f"{branch_name} tracer fraction must lie in [0, 1]"
            )


def solve_isothermal_gas_tjunction(
    west: GasBranchTrace,
    east: GasBranchTrace,
    vertical: GasBranchTrace,
    *,
    sound_speed: float,
) -> IsothermalGasTJunctionSolution:
    """Solve the zero-storage three-branch isothermal junction problem.

    Parameters
    ----------
    west, east, vertical:
        Interior branch traces.  Every ``outward_velocity`` is expressed in a
        local coordinate directed from the junction into that branch.
    sound_speed:
        Common positive isothermal sound speed ``sqrt(R T)``.

    Raises
    ------
    SupersonicGasTraceError
        If any supplied interior trace is sonic or supersonic.
    NoAdmissibleSubsonicJunctionError
        If the characteristic closure would make any junction branch state
        sonic/supersonic, or if no finite positive common density exists.
    """

    c = float(sound_speed)
    if not math.isfinite(c):
        raise GasTJunctionError("sound speed must be finite")
    if c <= 0.0:
        raise NonPositiveGasStateError("sound speed must be strictly positive")

    traces = (west, east, vertical)
    names = ("west", "east", "vertical")
    for name, trace in zip(names, traces):
        _validate_trace(trace, branch_name=name, sound_speed=c)

    tracer_presence = tuple(trace.tracer_fraction is not None for trace in traces)
    if any(tracer_presence) and not all(tracer_presence):
        raise GasTJunctionError(
            "tracer fractions must be supplied for all branches or none"
        )

    # Use density ratios relative to the first trace.  Besides avoiding an
    # unnecessary large-log cancellation, this makes a uniform quiescent
    # state exactly stationary in floating-point arithmetic.
    reference_log_density = math.log(float(west.density))
    total_area = math.fsum(float(trace.area) for trace in traces)
    log_density_offset_numerator = math.fsum(
        float(trace.area)
        * (
            math.log(float(trace.density))
            - reference_log_density
            - float(trace.outward_velocity) / c
        )
        for trace in traces
    )
    log_common_density = (
        reference_log_density + log_density_offset_numerator / total_area
    )
    try:
        common_density = math.exp(log_common_density)
    except OverflowError as error:
        raise NoAdmissibleSubsonicJunctionError(
            "junction density overflows the finite isothermal state space"
        ) from error
    if not math.isfinite(common_density) or common_density <= 0.0:
        raise NoAdmissibleSubsonicJunctionError(
            "no finite positive common junction density exists"
        )

    node_velocities = tuple(
        float(trace.outward_velocity)
        + c * (log_common_density - math.log(float(trace.density)))
        for trace in traces
    )
    for name, velocity in zip(names, node_velocities):
        if not math.isfinite(velocity) or abs(velocity) >= c:
            raise NoAdmissibleSubsonicJunctionError(
                f"characteristic closure gives a non-subsonic {name} state"
            )

    common_pressure = common_density * c * c
    if not math.isfinite(common_pressure):
        raise NoAdmissibleSubsonicJunctionError(
            "junction pressure is not finite"
        )

    mass_fluxes = tuple(
        common_density * float(trace.area) * velocity
        for trace, velocity in zip(traces, node_velocities)
    )
    mass_residual = math.fsum(mass_fluxes)

    junction_tracer_fraction: float | None = None
    tracer_fluxes: tuple[float | None, ...]
    tracer_residual: float | None
    if all(tracer_presence):
        incoming_mass_rate = -math.fsum(
            flux for flux in mass_fluxes if flux < 0.0
        )
        outgoing_mass_rate = math.fsum(
            flux for flux in mass_fluxes if flux > 0.0
        )
        throughput_scale = math.fsum(abs(flux) for flux in mass_fluxes)
        roundoff_scale = 32.0 * math.ulp(max(throughput_scale, 1.0))
        if incoming_mass_rate <= roundoff_scale and outgoing_mass_rate <= roundoff_scale:
            # No scalar crosses a stationary junction.  Its mixture value is
            # underdetermined, while every conservative tracer flux is zero.
            tracer_fluxes = (0.0, 0.0, 0.0)
        elif incoming_mass_rate <= 0.0 or outgoing_mass_rate <= 0.0:
            raise NoAdmissibleSubsonicJunctionError(
                "nonzero junction transport lacks both inflow and outflow"
            )
        else:
            incoming_tracer_rate = -math.fsum(
                flux * float(trace.tracer_fraction)
                for flux, trace in zip(mass_fluxes, traces)
                if flux < 0.0
            )
            junction_tracer_fraction = incoming_tracer_rate / outgoing_mass_rate
            # A residual at roundoff can place the ratio a few ulps outside
            # [0, 1].  Reject instead of clipping: clipping would hide an
            # inconsistent mass/tracer closure.
            if not 0.0 <= junction_tracer_fraction <= 1.0:
                raise NoAdmissibleSubsonicJunctionError(
                    "junction tracer mixture lies outside [0, 1]"
                )
            tracer_fluxes = tuple(
                flux
                * (
                    junction_tracer_fraction
                    if flux > 0.0
                    else float(trace.tracer_fraction)
                )
                for flux, trace in zip(mass_fluxes, traces)
            )
        tracer_residual = math.fsum(float(flux) for flux in tracer_fluxes)
    else:
        tracer_fluxes = (None, None, None)
        tracer_residual = None

    branch_fluxes = tuple(
        GasBranchJunctionFlux(
            density=common_density,
            outward_velocity=velocity,
            area=float(trace.area),
            mass_flux=mass_flux,
            momentum_flux=float(trace.area)
            * (common_density * velocity * velocity + common_pressure),
            tracer_flux=tracer_flux,
        )
        for trace, velocity, mass_flux, tracer_flux in zip(
            traces, node_velocities, mass_fluxes, tracer_fluxes
        )
    )

    return IsothermalGasTJunctionSolution(
        common_density=common_density,
        common_pressure=common_pressure,
        sound_speed=c,
        west=branch_fluxes[0],
        east=branch_fluxes[1],
        vertical=branch_fluxes[2],
        junction_tracer_fraction=junction_tracer_fraction,
        mass_residual=mass_residual,
        tracer_residual=tracer_residual,
    )


__all__ = [
    "GasBranchJunctionFlux",
    "GasBranchTrace",
    "GasTJunctionError",
    "IsothermalGasTJunctionSolution",
    "NoAdmissibleSubsonicJunctionError",
    "NonPositiveGasStateError",
    "SupersonicGasTraceError",
    "solve_isothermal_gas_tjunction",
    "velocity_in_outward_coordinate",
]
