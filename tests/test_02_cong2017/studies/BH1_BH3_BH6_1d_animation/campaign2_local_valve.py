"""Shared passive local-valve physics for Cong et al. Campaign 2.

This module is deliberately independent of the persistent horizontal driver.
It does not import a case identifier, riser diameter, experimental outcome, or
qualification threshold.  It freezes the valve law used by all three audited
Campaign-2 OpenFOAM cases::

    phi(t) = max(0.001, sin(pi/2 * clip(t/0.20, 0, 1))**2)
    K(t)   = phi(t)**(-2) - 1
    dp     = 0.5 * rho * K * abs(u) * u

The published experiment reports a manually operated ball valve opening in
approximately 0.20 s.  The sine-squared area history, the 0.001 numerical
minimum opening, and the passive Forchheimer representation are the common 2D
*numerical* valve contract; they are not measured discharge coefficients.

The liquid two-port solve uses the linear water-hammer characteristics already
used by the Case-1 pressurised-pipe core.  For physical flow positive from the
left port to the right port,

    C+_L = u_L + p_L/(rho_L*a_L)
    C-_R = u_R - p_R/(rho_R*a_R).

The zero-volume valve has one common volume flow ``Q`` and permits the signed
pressure jump above.  The resulting scalar equation is monotone and is solved
in closed form.  Instantaneous branch areas are supplied by the caller; this
module does not invent a free-surface pressure trace or a wet/dry closure.

The returned face momentum flows use the explicitly supplied valve flow area.
Their difference is the force exerted by the valve wall on the liquid.  A
separate ledger integrates signed through-volume, wall impulse, and strictly
non-negative dissipation.  No mass source is associated with the valve.

The clean free-surface branch is separate.  It uses the circular
Saint-Venant characteristics that can actually reach a stationary face.  Two
subcritical traces retain the tangent two-impedance closure; a one-sided trace
uses the exact nonlinear incoming characteristic with a native-flux offset,
while a dry or supercritical-outflow downstream trace supplies no condition.
A supercritical upstream trace may be
supply controlled (choked) only when its native Riemann flow has enough
specific-energy margin to cross the passive loss and feed a dry or
supercritical-outflow downstream state; otherwise an unresolved upstream
entropy shock is required and the stage is rejected.  This makes exact
``K=0`` the unmodified Case-1 flux and retains a
continuous passive ``K -> 0`` limit without clipping a Froude number or adding
a liquid film.  The moving shock/cut-cell integration is supplied by the
hash-pinned sibling in ``case1_local_valve_extension.py``; the constitutive
law remains single-sourced here.

Evidence sources (identical in B-H1, B-H3, and B-H6):

* ``openfoam/2d/case/constant/valveProperties``;
* ``openfoam/2d/solver/UEqn.H``;
* ``openfoam/2d/case_config.json`` (water density 998.0 kg/m3).

This is a tested physics component only.  The hash-locked Case-1 core and the
Campaign-2 driver must not be claimed to use it until an explicit conservative
internal-face integration is implemented and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math
from typing import Final


OPENING_DURATION_S: Final[float] = 0.20
MINIMUM_AREA_FRACTION: Final[float] = 0.001
RESISTANCE_LENGTH_M: Final[float] = 0.025
WATER_DENSITY_KG_M3: Final[float] = 998.0

VALVE_MODEL_NAME: Final[str] = "sineSquaredAreaForchheimer"
CHARACTERISTIC_MODEL_NAME: Final[str] = (
    "Case-1 linear water-hammer C+ (left) / C- (right) two-port"
)
FREE_SURFACE_CHARACTERISTIC_MODEL_NAME: Final[str] = (
    "directional Case-1 circular Saint-Venant characteristic supply about "
    "the native central-upwind Riemann face"
)


class FreeSurfaceValveControlRegime(str, Enum):
    """Characteristic count used by one free-surface valve solution."""

    EXACT_NATIVE = "exact_native"
    TWO_SIDED_CHARACTERISTIC = "two_sided_characteristic"
    ONE_SIDED_CHARACTERISTIC = "one_sided_characteristic"
    UPSTREAM_SUPPLY_CHOKED = "upstream_supply_choked"


class PressurisedMocValveControlRegime(str, Enum):
    """Characteristic count at a stationary valve in an elastic pipe."""

    EXACT_NATIVE = "exact_native"
    TWO_SIDED_SUBCRITICAL = "two_sided_subcritical"
    LEFT_SUPPLY_CHOKED = "left_supply_choked"
    RIGHT_SUPPLY_CHOKED = "right_supply_choked"


class PressurisedMocValveNoRootError(RuntimeError):
    """The supplied MOC traces need an entropy wave, not a local valve root."""


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ValveOpeningState:
    """The frozen shared opening state at one physical time."""

    time_s: float
    area_fraction: float
    loss_coefficient: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_s)):
            raise ValueError("time_s must be finite")
        if not (
            MINIMUM_AREA_FRACTION
            <= float(self.area_fraction)
            <= 1.0
        ):
            raise ValueError("area_fraction is outside the shared valve range")
        if (
            not math.isfinite(float(self.loss_coefficient))
            or float(self.loss_coefficient) < 0.0
        ):
            raise ValueError("loss_coefficient must be finite and non-negative")


def shared_opening_state(time_s: float) -> ValveOpeningState:
    """Return the common H1/H3/H6 sine-squared valve state.

    The clipping in the audited 2D contract also defines the schedule for a
    finite negative query time.  At and beyond 0.20 s the loss coefficient is
    returned as the exact floating-point value ``0.0``.
    """

    time = _finite(time_s, name="time_s")
    normalized = min(max(time / OPENING_DURATION_S, 0.0), 1.0)
    if normalized >= 1.0:
        area_fraction = 1.0
        coefficient = 0.0
    else:
        opening_sine = math.sin(0.5 * math.pi * normalized)
        area_fraction = max(
            MINIMUM_AREA_FRACTION,
            opening_sine * opening_sine,
        )
        coefficient = 1.0 / (area_fraction * area_fraction) - 1.0
    return ValveOpeningState(
        time_s=time,
        area_fraction=float(area_fraction),
        loss_coefficient=float(coefficient),
    )


@dataclass(frozen=True)
class LiquidValveTrace:
    """One physical-coordinate liquid trace adjacent to the valve.

    ``velocity_m_s`` is positive from the left port to the right port at both
    traces.  ``gauge_pressure_Pa`` must use one common pressure datum.  Area is
    the instantaneous liquid characteristic area, not the time-dependent valve
    opening fraction.  The default density is the common Campaign-2 2D water
    density from each case's ``case_config.json``.
    """

    area_m2: float
    velocity_m_s: float
    gauge_pressure_Pa: float
    wave_speed_m_s: float
    density_kg_m3: float = WATER_DENSITY_KG_M3

    def __post_init__(self) -> None:
        _positive_finite(self.area_m2, name="area_m2")
        _finite(self.velocity_m_s, name="velocity_m_s")
        _finite(self.gauge_pressure_Pa, name="gauge_pressure_Pa")
        _positive_finite(self.wave_speed_m_s, name="wave_speed_m_s")
        _positive_finite(self.density_kg_m3, name="density_kg_m3")

    @property
    def volume_flow_m3_s(self) -> float:
        return float(self.area_m2 * self.velocity_m_s)

    @property
    def acoustic_pressure_impedance_Pa_s_m(self) -> float:
        """Return ``rho*a`` for velocity-form water-hammer characteristics."""

        return float(self.density_kg_m3 * self.wave_speed_m_s)

    @property
    def acoustic_flow_impedance_Pa_s_m3(self) -> float:
        """Return ``rho*a/A`` for a volume-flow characteristic."""

        return float(
            self.acoustic_pressure_impedance_Pa_s_m / self.area_m2
        )

    @property
    def left_incoming_Cplus_m_s(self) -> float:
        return float(
            self.velocity_m_s
            + self.gauge_pressure_Pa
            / self.acoustic_pressure_impedance_Pa_s_m
        )

    @property
    def right_incoming_Cminus_m_s(self) -> float:
        return float(
            self.velocity_m_s
            - self.gauge_pressure_Pa
            / self.acoustic_pressure_impedance_Pa_s_m
        )


@dataclass(frozen=True)
class LiquidValveSolution:
    """One passive, mass-conservative zero-volume valve solution."""

    opening: ValveOpeningState
    valve_flow_area_m2: float
    volume_flow_left_to_right_m3_s: float
    valve_velocity_m_s: float
    left_gauge_pressure_Pa: float
    right_gauge_pressure_Pa: float
    signed_pressure_jump_Pa: float
    left_momentum_flow_N: float
    right_momentum_flow_N: float
    valve_wall_force_on_liquid_N: float
    dissipation_power_W: float
    left_characteristic_residual_m_s: float
    right_characteristic_residual_m_s: float
    pressure_jump_residual_Pa: float
    continuity_residual_m3_s: float
    upwind_density_kg_m3: float
    characteristic_model: str = CHARACTERISTIC_MODEL_NAME

    def __post_init__(self) -> None:
        _positive_finite(self.valve_flow_area_m2, name="valve_flow_area_m2")
        _positive_finite(self.upwind_density_kg_m3, name="upwind_density_kg_m3")
        scalar_values = (
            self.volume_flow_left_to_right_m3_s,
            self.valve_velocity_m_s,
            self.left_gauge_pressure_Pa,
            self.right_gauge_pressure_Pa,
            self.signed_pressure_jump_Pa,
            self.left_momentum_flow_N,
            self.right_momentum_flow_N,
            self.valve_wall_force_on_liquid_N,
            self.dissipation_power_W,
            self.left_characteristic_residual_m_s,
            self.right_characteristic_residual_m_s,
            self.pressure_jump_residual_Pa,
            self.continuity_residual_m3_s,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("all liquid-valve solution fields must be finite")
        if self.dissipation_power_W < 0.0:
            raise ValueError("a passive valve cannot have negative dissipation")

    @property
    def left_specific_momentum_flux_m4_s2(self) -> float:
        return float(self.left_momentum_flow_N / self.upwind_density_kg_m3)

    @property
    def right_specific_momentum_flux_m4_s2(self) -> float:
        return float(self.right_momentum_flow_N / self.upwind_density_kg_m3)

    @property
    def momentum_force_residual_N(self) -> float:
        """Return the face-flux minus valve-wall force closure residual."""

        return float(
            self.right_momentum_flow_N
            - self.left_momentum_flow_N
            - self.valve_wall_force_on_liquid_N
        )


@dataclass(frozen=True)
class PressurisedMocValveSolution:
    """Passive two-port closure after pressurised-MOC characteristic count.

    The subcritical branch contains one incoming characteristic from either
    side and delegates to the exact linear water-hammer closure above.  A
    genuinely super-acoustic upstream state fixes the supply only when the
    downstream state also carries both characteristics away from the valve.
    Mixed characteristic counts require a resolved entropy wave and are
    rejected rather than clipped.
    """

    opening: ValveOpeningState
    control_regime: PressurisedMocValveControlRegime
    valve_flow_area_m2: float
    volume_flow_left_to_right_m3_s: float
    valve_velocity_m_s: float
    left_gauge_pressure_Pa: float
    right_gauge_pressure_Pa: float
    signed_pressure_jump_Pa: float
    left_momentum_flow_N: float
    right_momentum_flow_N: float
    valve_wall_force_on_liquid_N: float
    dissipation_power_W: float
    left_characteristic_residual_m_s: float
    right_characteristic_residual_m_s: float
    continuity_residual_m3_s: float
    upwind_density_kg_m3: float
    left_incoming_characteristic_count: int
    right_incoming_characteristic_count: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.control_regime,
            PressurisedMocValveControlRegime,
        ):
            raise TypeError("control_regime must be a pressurised-MOC regime")
        _positive_finite(self.valve_flow_area_m2, name="valve_flow_area_m2")
        _positive_finite(self.upwind_density_kg_m3, name="upwind_density_kg_m3")
        values = (
            self.volume_flow_left_to_right_m3_s,
            self.valve_velocity_m_s,
            self.left_gauge_pressure_Pa,
            self.right_gauge_pressure_Pa,
            self.signed_pressure_jump_Pa,
            self.left_momentum_flow_N,
            self.right_momentum_flow_N,
            self.valve_wall_force_on_liquid_N,
            self.dissipation_power_W,
            self.left_characteristic_residual_m_s,
            self.right_characteristic_residual_m_s,
            self.continuity_residual_m3_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("all pressurised-MOC valve fields must be finite")
        if self.dissipation_power_W < 0.0:
            raise ValueError("a passive valve cannot have negative dissipation")
        for count in (
            self.left_incoming_characteristic_count,
            self.right_incoming_characteristic_count,
        ):
            if isinstance(count, bool) or count not in (0, 1, 2):
                raise ValueError("incoming characteristic counts must be 0, 1 or 2")

        expected_power = (
            self.signed_pressure_jump_Pa
            * self.volume_flow_left_to_right_m3_s
        )
        tolerance = 256.0 * math.ulp(
            max(1.0, abs(expected_power), abs(self.dissipation_power_W))
        )
        if not math.isclose(
            expected_power,
            self.dissipation_power_W,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("pressurised-MOC dissipation must equal dp*Q")
        if abs(self.momentum_force_residual_N) > 256.0 * math.ulp(
            max(
                1.0,
                abs(self.left_momentum_flow_N),
                abs(self.right_momentum_flow_N),
                abs(self.valve_wall_force_on_liquid_N),
            )
        ):
            raise ValueError("pressurised-MOC momentum ports do not close")

    @property
    def momentum_force_residual_N(self) -> float:
        return float(
            self.right_momentum_flow_N
            - self.left_momentum_flow_N
            - self.valve_wall_force_on_liquid_N
        )


def _stable_positive_quadratic_root(
    *,
    drive_pressure_Pa: float,
    flow_impedance_Pa_s_m3: float,
    quadratic_resistance_Pa_s2_m6: float,
) -> float:
    """Solve ``R*q**2 + Z*q = drive`` for the non-negative root."""

    drive = float(drive_pressure_Pa)
    impedance = _positive_finite(
        flow_impedance_Pa_s_m3,
        name="flow_impedance_Pa_s_m3",
    )
    resistance = float(quadratic_resistance_Pa_s2_m6)
    if not math.isfinite(resistance) or resistance < 0.0:
        raise ValueError("quadratic_resistance_Pa_s2_m6 must be non-negative")
    if drive < 0.0 or not math.isfinite(drive):
        raise ValueError("drive_pressure_Pa must be finite and non-negative")
    if drive == 0.0:
        return 0.0
    if resistance == 0.0:
        return float(drive / impedance)
    discriminant = math.sqrt(
        impedance * impedance + 4.0 * resistance * drive
    )
    # This form avoids subtracting two nearly equal positive values as the
    # valve becomes fully open.
    return float(2.0 * drive / (impedance + discriminant))


def _circular_area_depth_from_theta(
    theta: float,
    *,
    radius_m: float,
) -> tuple[float, float]:
    angle = float(theta)
    radius = _positive_finite(radius_m, name="radius_m")
    if abs(angle) < 1.0e-4:
        area_factor = (
            2.0 * angle**3 / 3.0
            - 2.0 * angle**5 / 15.0
            + 4.0 * angle**7 / 315.0
        )
    else:
        area_factor = angle - math.sin(angle) * math.cos(angle)
    area = radius * radius * area_factor
    depth = radius * (1.0 - math.cos(angle))
    return float(area), float(depth)


def _circular_theta_from_area(area_m2: float, full_area_m2: float) -> float:
    area = _finite(area_m2, name="area_m2")
    full_area = _positive_finite(full_area_m2, name="full_area_m2")
    if not (0.0 <= area <= full_area):
        raise ValueError("circular area must lie in [0, full_area]")
    if area == 0.0:
        return 0.0
    if area == full_area:
        return math.pi
    radius = math.sqrt(full_area / math.pi)
    lower = 0.0
    upper = math.pi
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        middle_area, _ = _circular_area_depth_from_theta(
            middle,
            radius_m=radius,
        )
        if middle_area < area:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper))


def _circular_depth_from_area(area_m2: float, full_area_m2: float) -> float:
    full_area = _positive_finite(full_area_m2, name="full_area_m2")
    radius = math.sqrt(full_area / math.pi)
    theta = _circular_theta_from_area(area_m2, full_area)
    _, depth = _circular_area_depth_from_theta(
        theta,
        radius_m=radius,
    )
    return depth


def _circular_characteristic_integrand(
    theta: float,
    *,
    radius_m: float,
    gravity_m_s2: float,
) -> float:
    if theta <= 0.0:
        return math.sqrt(3.0 * gravity_m_s2 * radius_m)
    if theta >= math.pi:
        return 0.0
    area, _ = _circular_area_depth_from_theta(theta, radius_m=radius_m)
    top_width = 2.0 * radius_m * math.sin(theta)
    celerity = math.sqrt(gravity_m_s2 * area / top_width)
    return float(gravity_m_s2 * radius_m * math.sin(theta) / celerity)


@lru_cache(maxsize=8)
def _circular_characteristic_table(
    full_area_m2: float,
    gravity_m_s2: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Build a deterministic high-order primitive table for one section."""

    full_area = _positive_finite(full_area_m2, name="full_area_m2")
    gravity = _positive_finite(gravity_m_s2, name="gravity_m_s2")
    radius = math.sqrt(full_area / math.pi)
    intervals = 4096
    step = math.pi / intervals
    values = tuple(
        _circular_characteristic_integrand(
            index * step,
            radius_m=radius,
            gravity_m_s2=gravity,
        )
        for index in range(intervals + 1)
    )
    cumulative = [0.0]
    for index in range(intervals):
        midpoint_value = _circular_characteristic_integrand(
            (index + 0.5) * step,
            radius_m=radius,
            gravity_m_s2=gravity,
        )
        increment = (
            step
            * (values[index] + 4.0 * midpoint_value + values[index + 1])
            / 6.0
        )
        cumulative.append(float(cumulative[-1] + increment))
    return step, values, tuple(cumulative)


def _circular_characteristic_primitive(
    theta: float,
    *,
    full_area_m2: float,
    gravity_m_s2: float,
) -> float:
    """Interpolate the cached primitive with endpoint-slope Hermite data."""

    angle = float(theta)
    if not (0.0 <= angle <= math.pi):
        raise ValueError("circular theta must lie in [0, pi]")
    step, values, cumulative = _circular_characteristic_table(
        float(full_area_m2),
        float(gravity_m_s2),
    )
    if angle == math.pi:
        return cumulative[-1]
    index = min(int(angle / step), len(values) - 2)
    fraction = (angle - index * step) / step
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    left_value = cumulative[index]
    right_value = cumulative[index + 1]
    return float(
        (2.0 * fraction3 - 3.0 * fraction2 + 1.0) * left_value
        + (fraction3 - 2.0 * fraction2 + fraction)
        * step
        * values[index]
        + (-2.0 * fraction3 + 3.0 * fraction2) * right_value
        + (fraction3 - fraction2) * step * values[index + 1]
    )


def _solve_native_offset_one_sided_characteristic(
    *,
    upstream: CircularSaintVenantValveTrace,
    native_volume_flow_m3_s: float,
    direction: float,
    loss_coefficient: float,
) -> tuple[float, float, float]:
    """Return ``(A*,Q*,head-rise)`` on the exact incoming characteristic."""

    native_flow = _finite(
        native_volume_flow_m3_s,
        name="native_volume_flow_m3_s",
    )
    coefficient = _finite(loss_coefficient, name="loss_coefficient")
    if native_flow == 0.0 or coefficient <= 0.0:
        raise ValueError("one-sided nonlinear solve requires nonzero Q0 and K")
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    area_start = upstream.area_m2
    full_area = upstream.full_area_m2
    gravity = upstream.gravity_m_s2
    radius = math.sqrt(full_area / math.pi)
    theta_start = _circular_theta_from_area(area_start, full_area)
    _, depth_start = _circular_area_depth_from_theta(
        theta_start,
        radius_m=radius,
    )
    primitive_start = _circular_characteristic_primitive(
        theta_start,
        full_area_m2=full_area,
        gravity_m_s2=gravity,
    )
    trace_flow_start = upstream.discharge_m3_s

    def evaluate(theta: float) -> tuple[float, float, float, float]:
        area, depth = _circular_area_depth_from_theta(
            theta,
            radius_m=radius,
        )
        primitive_change = float(
            _circular_characteristic_primitive(
                theta,
                full_area_m2=full_area,
                gravity_m_s2=gravity,
            )
            - primitive_start
        )
        velocity = float(
            upstream.velocity_m_s - direction * primitive_change
        )
        characteristic_flow = float(area * velocity)
        flow = float(
            native_flow + characteristic_flow - trace_flow_start
        )
        residual = float(
            gravity * (depth - depth_start)
            - 0.5 * coefficient * (flow / area) ** 2
        )
        return area, flow, depth, residual

    lower = theta_start
    upper = math.nextafter(math.pi, 0.0)
    _, _, _, lower_residual = evaluate(lower)
    if lower_residual >= 0.0:
        raise FloatingPointError("one-sided characteristic root lost its K>0 bracket")
    _, upper_flow, _, _ = evaluate(upper)
    if direction * upper_flow < 0.0:
        # Stop exactly at the zero-flow point; the passive root must occur
        # before a sign reversal of the native Riemann supply.
        flow_lower = native_flow
        theta_lower = lower
        theta_upper = upper
        for _ in range(100):
            theta_middle = 0.5 * (theta_lower + theta_upper)
            _, flow_middle, _, _ = evaluate(theta_middle)
            if direction * flow_middle > 0.0:
                theta_lower = theta_middle
                flow_lower = flow_middle
            else:
                theta_upper = theta_middle
        upper = float(0.5 * (theta_lower + theta_upper))
        if direction * flow_lower <= 0.0:
            raise FloatingPointError("failed to bracket the same-sign supply root")
    _, _, _, upper_residual = evaluate(upper)
    if upper_residual < 0.0:
        raise ValueError(
            "one-sided characteristic reaches full area before closing the valve loss"
        )
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        _, _, _, middle_residual = evaluate(middle)
        if middle_residual < 0.0:
            lower = middle
        else:
            upper = middle
    theta_solution = float(0.5 * (lower + upper))
    area_solution, flow_solution, depth_solution, _ = evaluate(theta_solution)
    if direction * flow_solution <= 0.0:
        raise FloatingPointError("one-sided valve root reversed the native flow")
    head_rise = float(depth_solution - depth_start)
    return area_solution, flow_solution, head_rise


def _circular_critical_specific_energy(
    *,
    discharge_m3_s: float,
    full_area_m2: float,
    gravity_m_s2: float,
) -> tuple[float, float]:
    """Return critical area and minimum specific energy for ``abs(Q)``."""

    discharge = abs(_finite(discharge_m3_s, name="discharge_m3_s"))
    full_area = _positive_finite(full_area_m2, name="full_area_m2")
    gravity = _positive_finite(gravity_m_s2, name="gravity_m_s2")
    if discharge == 0.0:
        return 0.0, 0.0
    radius = math.sqrt(full_area / math.pi)
    lower = 1.0e-12
    upper = math.pi - 1.0e-12

    def residual(theta: float) -> float:
        area, _ = _circular_area_depth_from_theta(theta, radius_m=radius)
        top_width = 2.0 * radius * math.sin(theta)
        celerity = math.sqrt(gravity * area / top_width)
        return float(discharge / area - celerity)

    if residual(lower) <= 0.0 or residual(upper) >= 0.0:
        raise FloatingPointError("failed to bracket the circular critical state")
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    theta = 0.5 * (lower + upper)
    critical_area, critical_depth = _circular_area_depth_from_theta(
        theta,
        radius_m=radius,
    )
    critical_energy = float(
        critical_depth
        + discharge * discharge
        / (2.0 * gravity * critical_area * critical_area)
    )
    return float(critical_area), critical_energy


def solve_passive_liquid_valve(
    left: LiquidValveTrace,
    right: LiquidValveTrace,
    *,
    time_s: float,
    valve_flow_area_m2: float,
) -> LiquidValveSolution:
    """Solve the shared passive liquid valve from two incoming traces.

    The left trace contributes the Case-1 water-hammer ``C+`` invariant and
    the right trace contributes ``C-``.  A positive result flows left-to-right.
    The same signed volume flow crosses both ports, so the valve owns no liquid
    storage.  Density in the Forchheimer jump is upwinded; Campaign 2 normally
    supplies the same 998.0 kg/m3 water density at both ports.

    ``valve_flow_area_m2`` is explicit because the 2D reference area
    ``D*extrusion`` is not a physical circular-pipe area that can silently be
    reused in 1D.  For a full 50 mm circular pipe the caller would supply
    ``pi*D**2/4``.
    """

    if not isinstance(left, LiquidValveTrace) or not isinstance(
        right, LiquidValveTrace
    ):
        raise TypeError("left and right must be LiquidValveTrace instances")
    flow_area = _positive_finite(
        valve_flow_area_m2,
        name="valve_flow_area_m2",
    )
    opening = shared_opening_state(time_s)

    z_left = left.acoustic_pressure_impedance_Pa_s_m
    z_right = right.acoustic_pressure_impedance_Pa_s_m
    flow_impedance = (
        left.acoustic_flow_impedance_Pa_s_m3
        + right.acoustic_flow_impedance_Pa_s_m3
    )
    drive_pressure = (
        z_left * left.left_incoming_Cplus_m_s
        + z_right * right.right_incoming_Cminus_m_s
    )
    direction = 0.0 if drive_pressure == 0.0 else math.copysign(1.0, drive_pressure)
    upwind_density = (
        left.density_kg_m3 if direction >= 0.0 else right.density_kg_m3
    )
    quadratic_resistance = (
        0.5
        * upwind_density
        * opening.loss_coefficient
        / (flow_area * flow_area)
    )
    magnitude = _stable_positive_quadratic_root(
        drive_pressure_Pa=abs(drive_pressure),
        flow_impedance_Pa_s_m3=flow_impedance,
        quadratic_resistance_Pa_s2_m6=quadratic_resistance,
    )
    volume_flow = float(direction * magnitude)
    valve_velocity = float(volume_flow / flow_area)
    expected_pressure_jump = float(
        0.5
        * upwind_density
        * opening.loss_coefficient
        * abs(valve_velocity)
        * valve_velocity
    )

    left_pressure_characteristic = float(
        z_left
        * (
            left.left_incoming_Cplus_m_s
            - volume_flow / left.area_m2
        )
    )
    right_pressure_characteristic = float(
        z_right
        * (
            volume_flow / right.area_m2
            - right.right_incoming_Cminus_m_s
        )
    )
    common_pressure_centre = 0.5 * (
        left_pressure_characteristic + right_pressure_characteristic
    )
    if opening.loss_coefficient == 0.0:
        # Preserve an exact lossless face: the two returned pressures and
        # momentum flows are bitwise equal, not merely close after cancellation.
        left_pressure = float(common_pressure_centre)
        right_pressure = float(common_pressure_centre)
        expected_pressure_jump = 0.0
    else:
        left_pressure = float(
            common_pressure_centre + 0.5 * expected_pressure_jump
        )
        # Construct the second pressure from the signed jump to keep the
        # pressure-loss ledger single-sourced.
        right_pressure = float(left_pressure - expected_pressure_jump)

    left_characteristic_velocity = (
        left.left_incoming_Cplus_m_s - left_pressure / z_left
    )
    right_characteristic_velocity = (
        right.right_incoming_Cminus_m_s + right_pressure / z_right
    )
    left_characteristic_residual = float(
        volume_flow / left.area_m2 - left_characteristic_velocity
    )
    right_characteristic_residual = float(
        volume_flow / right.area_m2 - right_characteristic_velocity
    )
    signed_jump = float(left_pressure - right_pressure)
    pressure_jump_residual = float(
        signed_jump - expected_pressure_jump
    )

    advective_momentum = float(
        upwind_density * volume_flow * volume_flow / flow_area
    )
    left_momentum = float(
        advective_momentum + left_pressure * flow_area
    )
    wall_force = float(-expected_pressure_jump * flow_area)
    # A zero-volume resistive face has one common volume flux.  Its two
    # momentum flows differ exactly by the external valve-wall force.
    right_momentum = float(left_momentum + wall_force)
    dissipation = float(expected_pressure_jump * volume_flow)
    if dissipation < 0.0:
        roundoff_scale = max(
            abs(expected_pressure_jump * volume_flow),
            1.0,
        )
        if abs(dissipation) <= 32.0 * math.ulp(roundoff_scale):
            dissipation = 0.0
        else:
            raise FloatingPointError("passive valve produced negative dissipation")

    return LiquidValveSolution(
        opening=opening,
        valve_flow_area_m2=flow_area,
        volume_flow_left_to_right_m3_s=volume_flow,
        valve_velocity_m_s=valve_velocity,
        left_gauge_pressure_Pa=left_pressure,
        right_gauge_pressure_Pa=right_pressure,
        signed_pressure_jump_Pa=signed_jump,
        left_momentum_flow_N=left_momentum,
        right_momentum_flow_N=right_momentum,
        valve_wall_force_on_liquid_N=wall_force,
        dissipation_power_W=dissipation,
        left_characteristic_residual_m_s=left_characteristic_residual,
        right_characteristic_residual_m_s=right_characteristic_residual,
        pressure_jump_residual_Pa=pressure_jump_residual,
        continuity_residual_m3_s=0.0,
        upwind_density_kg_m3=upwind_density,
    )


def solve_passive_pressurised_moc_valve(
    left: LiquidValveTrace,
    right: LiquidValveTrace,
    *,
    time_s: float,
    valve_flow_area_m2: float,
    nominal_pipe_area_m2: float,
) -> PressurisedMocValveSolution:
    """Close a stationary valve after counting elastic-pipe characteristics.

    ``left`` and ``right`` are the source-corrected incoming MOC traces at the
    same physical time.  In the normal subcritical branch the existing
    two-characteristic water-hammer solve is used without alteration.  A
    super-acoustic branch is admissible only as a one-way supply: the upstream
    trace fixes ``Q`` and the state reconstructed after the loss must still
    send both downstream characteristics away from the valve.  Every other
    characteristic count needs a resolved shock/rarefaction and raises
    :class:`PressurisedMocValveNoRootError`.

    The nominal area is used only to test the downstream elastic state,
    ``A=Af*(1+p/(rho*a**2))``.  It is not multiplied by the valve opening;
    the complete opening history is already contained in ``K``.
    """

    if not isinstance(left, LiquidValveTrace) or not isinstance(
        right,
        LiquidValveTrace,
    ):
        raise TypeError("left and right must be LiquidValveTrace instances")
    flow_area = _positive_finite(
        valve_flow_area_m2,
        name="valve_flow_area_m2",
    )
    nominal_area = _positive_finite(
        nominal_pipe_area_m2,
        name="nominal_pipe_area_m2",
    )
    opening = shared_opening_state(time_s)
    density_tolerance = 64.0 * math.ulp(
        max(left.density_kg_m3, right.density_kg_m3)
    )
    wave_tolerance = 64.0 * math.ulp(
        max(left.wave_speed_m_s, right.wave_speed_m_s)
    )
    if not math.isclose(
        left.density_kg_m3,
        right.density_kg_m3,
        rel_tol=0.0,
        abs_tol=density_tolerance,
    ):
        raise ValueError("pressurised-MOC valve traces must use one density")
    if not math.isclose(
        left.wave_speed_m_s,
        right.wave_speed_m_s,
        rel_tol=0.0,
        abs_tol=wave_tolerance,
    ):
        raise ValueError("pressurised-MOC valve traces must use one wave speed")

    def incoming_count(velocity: float, wave_speed: float, *, side: str) -> int:
        lambdas = (velocity - wave_speed, velocity + wave_speed)
        if any(value == 0.0 for value in lambdas):
            raise PressurisedMocValveNoRootError(
                "a stationary acoustic characteristic makes the local valve "
                "closure non-unique"
            )
        if side == "left":
            return sum(value > 0.0 for value in lambdas)
        return sum(value < 0.0 for value in lambdas)

    left_count = incoming_count(
        left.velocity_m_s,
        left.wave_speed_m_s,
        side="left",
    )
    right_count = incoming_count(
        right.velocity_m_s,
        right.wave_speed_m_s,
        side="right",
    )

    if opening.loss_coefficient == 0.0 or (
        left_count == 1 and right_count == 1
    ):
        base = solve_passive_liquid_valve(
            left,
            right,
            time_s=time_s,
            valve_flow_area_m2=flow_area,
        )
        left_port_velocity = (
            base.volume_flow_left_to_right_m3_s / left.area_m2
        )
        right_port_velocity = (
            base.volume_flow_left_to_right_m3_s / right.area_m2
        )
        if opening.loss_coefficient > 0.0 and (
            abs(left_port_velocity) >= left.wave_speed_m_s
            or abs(right_port_velocity) >= right.wave_speed_m_s
        ):
            raise PressurisedMocValveNoRootError(
                "the two-sided root leaves its subcritical characteristic branch"
            )
        regime = (
            PressurisedMocValveControlRegime.EXACT_NATIVE
            if opening.loss_coefficient == 0.0
            else PressurisedMocValveControlRegime.TWO_SIDED_SUBCRITICAL
        )
        return PressurisedMocValveSolution(
            opening=base.opening,
            control_regime=regime,
            valve_flow_area_m2=base.valve_flow_area_m2,
            volume_flow_left_to_right_m3_s=(
                base.volume_flow_left_to_right_m3_s
            ),
            valve_velocity_m_s=base.valve_velocity_m_s,
            left_gauge_pressure_Pa=base.left_gauge_pressure_Pa,
            right_gauge_pressure_Pa=base.right_gauge_pressure_Pa,
            signed_pressure_jump_Pa=base.signed_pressure_jump_Pa,
            left_momentum_flow_N=base.left_momentum_flow_N,
            right_momentum_flow_N=base.right_momentum_flow_N,
            valve_wall_force_on_liquid_N=(
                base.valve_wall_force_on_liquid_N
            ),
            dissipation_power_W=base.dissipation_power_W,
            left_characteristic_residual_m_s=(
                base.left_characteristic_residual_m_s
            ),
            right_characteristic_residual_m_s=(
                base.right_characteristic_residual_m_s
            ),
            continuity_residual_m3_s=base.continuity_residual_m3_s,
            upwind_density_kg_m3=base.upwind_density_kg_m3,
            left_incoming_characteristic_count=left_count,
            right_incoming_characteristic_count=right_count,
        )

    left_supply = left_count == 2 and right_count == 0
    right_supply = right_count == 2 and left_count == 0
    if not (left_supply or right_supply):
        raise PressurisedMocValveNoRootError(
            "the MOC characteristic count requires an entropy wave upstream "
            "of the local valve"
        )

    if left_supply:
        volume_flow = left.volume_flow_m3_s
        upstream_pressure = left.gauge_pressure_Pa
        upwind_density = left.density_kg_m3
        downstream_wave_speed = right.wave_speed_m_s
        regime = PressurisedMocValveControlRegime.LEFT_SUPPLY_CHOKED
    else:
        volume_flow = right.volume_flow_m3_s
        upstream_pressure = right.gauge_pressure_Pa
        upwind_density = right.density_kg_m3
        downstream_wave_speed = left.wave_speed_m_s
        regime = PressurisedMocValveControlRegime.RIGHT_SUPPLY_CHOKED
    if volume_flow == 0.0:
        raise PressurisedMocValveNoRootError(
            "a super-acoustic supply trace cannot carry zero flow"
        )
    expected_sign = 1.0 if left_supply else -1.0
    if math.copysign(1.0, volume_flow) != expected_sign:
        raise PressurisedMocValveNoRootError(
            "the super-acoustic characteristic direction opposes its supply flow"
        )

    valve_velocity = float(volume_flow / flow_area)
    pressure_jump = float(
        0.5
        * upwind_density
        * opening.loss_coefficient
        * abs(valve_velocity)
        * valve_velocity
    )
    if left_supply:
        left_pressure = float(upstream_pressure)
        right_pressure = float(left_pressure - pressure_jump)
        downstream_pressure = right_pressure
    else:
        right_pressure = float(upstream_pressure)
        left_pressure = float(right_pressure + pressure_jump)
        downstream_pressure = left_pressure
    downstream_area = float(
        nominal_area
        * (
            1.0
            + downstream_pressure
            / (upwind_density * downstream_wave_speed**2)
        )
    )
    if not math.isfinite(downstream_area) or downstream_area <= 0.0:
        raise PressurisedMocValveNoRootError(
            "the choked valve loss collapses the downstream elastic area"
        )
    downstream_velocity = float(volume_flow / downstream_area)
    if (
        left_supply and downstream_velocity <= downstream_wave_speed
    ) or (
        right_supply and downstream_velocity >= -downstream_wave_speed
    ):
        raise PressurisedMocValveNoRootError(
            "the post-loss downstream state sends a characteristic back to "
            "the valve and needs an entropy-wave solve"
        )

    advective_momentum = float(
        upwind_density * volume_flow * volume_flow / flow_area
    )
    left_momentum = float(
        advective_momentum + left_pressure * flow_area
    )
    wall_force = float(-pressure_jump * flow_area)
    right_momentum = float(left_momentum + wall_force)
    dissipation = float(pressure_jump * volume_flow)
    if dissipation < 0.0:
        raise FloatingPointError("super-acoustic valve branch is not passive")
    return PressurisedMocValveSolution(
        opening=opening,
        control_regime=regime,
        valve_flow_area_m2=flow_area,
        volume_flow_left_to_right_m3_s=volume_flow,
        valve_velocity_m_s=valve_velocity,
        left_gauge_pressure_Pa=left_pressure,
        right_gauge_pressure_Pa=right_pressure,
        signed_pressure_jump_Pa=pressure_jump,
        left_momentum_flow_N=left_momentum,
        right_momentum_flow_N=right_momentum,
        valve_wall_force_on_liquid_N=wall_force,
        dissipation_power_W=dissipation,
        left_characteristic_residual_m_s=0.0,
        right_characteristic_residual_m_s=0.0,
        continuity_residual_m3_s=0.0,
        upwind_density_kg_m3=upwind_density,
        left_incoming_characteristic_count=left_count,
        right_incoming_characteristic_count=right_count,
    )


@dataclass(frozen=True)
class CircularSaintVenantValveTrace:
    """One circular free-surface or exact-dry trace at the valve face.

    The celerity must be the Case-1 circular Saint-Venant value
    ``sqrt(d(g*I1)/dA)``.  A trace at or above the full circular area belongs
    to the elastic/MOC family and is deliberately rejected here.  Exact dry
    means ``A=Q=c=0``; no positive numerical film is manufactured.  Critical
    and supercritical wet traces are retained so the stationary-face
    characteristic count can be decided by the valve solver.
    """

    area_m2: float
    discharge_m3_s: float
    celerity_m_s: float
    full_area_m2: float
    density_kg_m3: float = WATER_DENSITY_KG_M3
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        area = _finite(self.area_m2, name="area_m2")
        full_area = _positive_finite(self.full_area_m2, name="full_area_m2")
        discharge = _finite(self.discharge_m3_s, name="discharge_m3_s")
        celerity = _finite(self.celerity_m_s, name="celerity_m_s")
        _positive_finite(self.density_kg_m3, name="density_kg_m3")
        _positive_finite(self.gravity_m_s2, name="gravity_m_s2")
        if area < 0.0:
            raise ValueError("area_m2 cannot be negative")
        if area >= full_area:
            raise ValueError(
                "a clean free-surface valve trace must satisfy area < full_area"
            )
        if area == 0.0:
            if discharge != 0.0 or celerity != 0.0:
                raise ValueError("an exact-dry trace must satisfy A=Q=c=0")
        elif celerity <= 0.0:
            raise ValueError(
                "a wet free-surface valve trace must have positive celerity"
            )

    @property
    def velocity_m_s(self) -> float:
        if self.is_dry:
            return 0.0
        return float(self.discharge_m3_s / self.area_m2)

    @property
    def is_dry(self) -> bool:
        return bool(self.area_m2 == 0.0)

    @property
    def is_subcritical(self) -> bool:
        return bool(
            not self.is_dry
            and abs(self.velocity_m_s) < self.celerity_m_s
        )

    @property
    def froude_number(self) -> float | None:
        if self.is_dry:
            return None
        return float(self.velocity_m_s / self.celerity_m_s)

    @property
    def left_incoming_flow_impedance_Pa_s_m3(self) -> float:
        """Return ``rho*c^2/[A(c-u)]`` from the incoming ``J+`` trace."""

        denominator = self.celerity_m_s - self.velocity_m_s
        if self.is_dry or denominator <= 0.0:
            raise ValueError("the left J+ trace has no finite positive impedance")
        return float(
            self.density_kg_m3
            * self.celerity_m_s
            * self.celerity_m_s
            / (self.area_m2 * denominator)
        )

    @property
    def right_incoming_flow_impedance_Pa_s_m3(self) -> float:
        """Return ``rho*c^2/[A(c+u)]`` from the incoming ``J-`` trace."""

        denominator = self.celerity_m_s + self.velocity_m_s
        if self.is_dry or denominator <= 0.0:
            raise ValueError("the right J- trace has no finite positive impedance")
        return float(
            self.density_kg_m3
            * self.celerity_m_s
            * self.celerity_m_s
            / (self.area_m2 * denominator)
        )


@dataclass(frozen=True)
class CircularSaintVenantValveSolution:
    """Passive clean-FV valve solution anchored to one native Case-1 face."""

    opening: ValveOpeningState
    native_volume_flow_m3_s: float
    native_specific_momentum_flux_m4_s2: float
    volume_flow_left_to_right_m3_s: float
    upwind_wetted_area_m2: float
    valve_velocity_m_s: float
    left_flow_impedance_Pa_s_m3: float
    right_flow_impedance_Pa_s_m3: float
    equivalent_flow_impedance_Pa_s_m3: float
    control_regime: FreeSurfaceValveControlRegime
    upstream_side: str | None
    left_characteristic_active: bool
    right_characteristic_active: bool
    native_offset_nonlinear_characteristic: bool
    signed_pressure_jump_Pa: float
    left_pressure_correction_Pa: float
    right_pressure_correction_Pa: float
    left_specific_momentum_flux_m4_s2: float
    right_specific_momentum_flux_m4_s2: float
    valve_wall_force_on_liquid_N: float
    dissipation_power_W: float
    valve_head_loss_m: float
    upstream_specific_energy_m: float | None
    critical_area_m2: float | None
    minimum_specific_energy_m: float | None
    supply_energy_margin_m: float | None
    impedance_residual_Pa: float
    pressure_partition_residual_Pa: float
    momentum_force_residual_N: float
    density_kg_m3: float
    gravity_m_s2: float
    characteristic_model: str = FREE_SURFACE_CHARACTERISTIC_MODEL_NAME

    def __post_init__(self) -> None:
        upwind_area = _finite(
            self.upwind_wetted_area_m2, name="upwind_wetted_area_m2"
        )
        if upwind_area < 0.0:
            raise ValueError("upwind_wetted_area_m2 cannot be negative")
        impedances = (
            self.left_flow_impedance_Pa_s_m3,
            self.right_flow_impedance_Pa_s_m3,
            self.equivalent_flow_impedance_Pa_s_m3,
        )
        if any(not math.isfinite(float(value)) or value < 0.0 for value in impedances):
            raise ValueError("active flow impedances must be finite and non-negative")
        _positive_finite(self.density_kg_m3, name="density_kg_m3")
        _positive_finite(self.gravity_m_s2, name="gravity_m_s2")
        if not isinstance(self.control_regime, FreeSurfaceValveControlRegime):
            raise TypeError("control_regime must be FreeSurfaceValveControlRegime")
        if self.upstream_side not in (None, "left", "right"):
            raise ValueError("upstream_side must be left, right, or None")
        if self.left_characteristic_active and self.left_flow_impedance_Pa_s_m3 <= 0.0:
            raise ValueError("an active left characteristic requires positive impedance")
        if self.right_characteristic_active and self.right_flow_impedance_Pa_s_m3 <= 0.0:
            raise ValueError("an active right characteristic requires positive impedance")
        if (
            self.native_offset_nonlinear_characteristic
            and self.control_regime
            is not FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
        ):
            raise ValueError(
                "native-offset nonlinear continuation requires one-sided control"
            )
        if self.control_regime in (
            FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC,
            FreeSurfaceValveControlRegime.TWO_SIDED_CHARACTERISTIC,
        ) and self.equivalent_flow_impedance_Pa_s_m3 <= 0.0:
            raise ValueError("a characteristic-controlled solution needs impedance")
        scalar_values = (
            self.native_volume_flow_m3_s,
            self.native_specific_momentum_flux_m4_s2,
            self.volume_flow_left_to_right_m3_s,
            self.valve_velocity_m_s,
            self.signed_pressure_jump_Pa,
            self.left_pressure_correction_Pa,
            self.right_pressure_correction_Pa,
            self.left_specific_momentum_flux_m4_s2,
            self.right_specific_momentum_flux_m4_s2,
            self.valve_wall_force_on_liquid_N,
            self.dissipation_power_W,
            self.valve_head_loss_m,
            self.impedance_residual_Pa,
            self.pressure_partition_residual_Pa,
            self.momentum_force_residual_N,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("all circular Saint-Venant valve fields must be finite")
        optional_energy_values = (
            self.upstream_specific_energy_m,
            self.critical_area_m2,
            self.minimum_specific_energy_m,
            self.supply_energy_margin_m,
        )
        if any(
            value is not None and not math.isfinite(float(value))
            for value in optional_energy_values
        ):
            raise ValueError("optional choked-energy fields must be finite")
        if self.valve_head_loss_m < 0.0:
            raise ValueError("valve_head_loss_m cannot be negative")
        if self.control_regime is FreeSurfaceValveControlRegime.UPSTREAM_SUPPLY_CHOKED:
            if any(value is None for value in optional_energy_values):
                raise ValueError("a choked solution requires its energy ledger")
            if self.critical_area_m2 is None or self.critical_area_m2 <= 0.0:
                raise ValueError("a choked solution requires positive critical area")
        if self.dissipation_power_W < 0.0:
            raise ValueError("a passive free-surface valve cannot dissipate negatively")

    @property
    def is_exact_native(self) -> bool:
        return bool(
            self.volume_flow_left_to_right_m3_s
            == self.native_volume_flow_m3_s
            and self.left_specific_momentum_flux_m4_s2
            == self.native_specific_momentum_flux_m4_s2
            and self.right_specific_momentum_flux_m4_s2
            == self.native_specific_momentum_flux_m4_s2
            and self.signed_pressure_jump_Pa == 0.0
        )


def solve_passive_circular_saint_venant_valve(
    left: CircularSaintVenantValveTrace,
    right: CircularSaintVenantValveTrace,
    *,
    native_volume_flow_m3_s: float,
    native_specific_momentum_flux_m4_s2: float,
    time_s: float,
) -> CircularSaintVenantValveSolution:
    """Solve a clean free-surface valve as a passive Riemann-flux correction.

    The exact circular invariants satisfy ``du +/- c dA/A = 0``.  With two
    participating traces their tangent flow impedances are used.  With only
    the upstream trace, its invariant is integrated nonlinearly and its flow
    increment is added to the native central-upwind flux.  Only a
    characteristic that can reach the stationary face participates.  If the
    upstream trace sends
    both characteristics into the face, the native Riemann supply is choked
    and remains fixed; the valve jump is then assigned wholly downstream.
    The native central-upwind volume flux is the exact lossless datum, not a
    full-liquid water-hammer characteristic.
    """

    if not isinstance(left, CircularSaintVenantValveTrace) or not isinstance(
        right,
        CircularSaintVenantValveTrace,
    ):
        raise TypeError(
            "left and right must be CircularSaintVenantValveTrace instances"
        )
    if left.full_area_m2 != right.full_area_m2:
        raise ValueError("both free-surface traces must use one circular section")
    if left.density_kg_m3 != right.density_kg_m3:
        raise ValueError("both free-surface traces must use one liquid density")
    if left.gravity_m_s2 != right.gravity_m_s2:
        raise ValueError("both free-surface traces must use one gravity")
    native_flow = _finite(
        native_volume_flow_m3_s,
        name="native_volume_flow_m3_s",
    )
    native_momentum = _finite(
        native_specific_momentum_flux_m4_s2,
        name="native_specific_momentum_flux_m4_s2",
    )
    opening = shared_opening_state(time_s)
    density = float(left.density_kg_m3)
    gravity = float(left.gravity_m_s2)
    if native_flow > 0.0:
        upstream_side = "left"
        upstream = left
        downstream = right
        direction = 1.0
    elif native_flow < 0.0:
        upstream_side = "right"
        upstream = right
        downstream = left
        direction = -1.0
    else:
        upstream_side = None
        upstream = left
        downstream = right
        direction = 0.0
    upwind_area = float(upstream.area_m2 if native_flow != 0.0 else 0.0)

    if opening.loss_coefficient == 0.0 or native_flow == 0.0:
        # This exact branch is the numerical transparency contract.  It uses
        # no trace admissibility threshold and returns the native flux bits.
        return CircularSaintVenantValveSolution(
            opening=opening,
            native_volume_flow_m3_s=native_flow,
            native_specific_momentum_flux_m4_s2=native_momentum,
            volume_flow_left_to_right_m3_s=native_flow,
            upwind_wetted_area_m2=upwind_area,
            valve_velocity_m_s=(
                float(native_flow / upwind_area) if upwind_area > 0.0 else 0.0
            ),
            left_flow_impedance_Pa_s_m3=0.0,
            right_flow_impedance_Pa_s_m3=0.0,
            equivalent_flow_impedance_Pa_s_m3=0.0,
            control_regime=FreeSurfaceValveControlRegime.EXACT_NATIVE,
            upstream_side=upstream_side,
            left_characteristic_active=False,
            right_characteristic_active=False,
            native_offset_nonlinear_characteristic=False,
            signed_pressure_jump_Pa=0.0,
            left_pressure_correction_Pa=0.0,
            right_pressure_correction_Pa=0.0,
            left_specific_momentum_flux_m4_s2=native_momentum,
            right_specific_momentum_flux_m4_s2=native_momentum,
            valve_wall_force_on_liquid_N=0.0,
            dissipation_power_W=0.0,
            valve_head_loss_m=0.0,
            upstream_specific_energy_m=None,
            critical_area_m2=None,
            minimum_specific_energy_m=None,
            supply_energy_margin_m=None,
            impedance_residual_Pa=0.0,
            pressure_partition_residual_Pa=0.0,
            momentum_force_residual_N=0.0,
            density_kg_m3=density,
            gravity_m_s2=gravity,
        )

    if upstream.is_dry:
        raise ValueError(
            "the native-flow donor is dry; no passive wet supply reaches the valve"
        )
    directional_upstream_velocity = direction * upstream.velocity_m_s
    if directional_upstream_velocity <= -upstream.celerity_m_s:
        raise ValueError(
            "the designated upstream trace has no characteristic reaching the valve"
        )

    quadratic_resistance = float(
        0.5
        * density
        * opening.loss_coefficient
        / (upwind_area * upwind_area)
    )
    left_active = False
    right_active = False
    z_left = 0.0
    z_right = 0.0
    upstream_specific_energy = None
    critical_area = None
    minimum_specific_energy = None
    supply_energy_margin = None
    nonlinear_characteristic = False

    if directional_upstream_velocity >= upstream.celerity_m_s:
        # Both upstream characteristics reach the stationary face.  A local
        # downstream condition cannot alter that Riemann supply; imposing a
        # smaller Q here would be an unphysical Froude clipping.  The passive
        # loss instead changes only the downstream momentum port until a
        # resolved upstream-moving wave changes the supplied trace.
        control_regime = (
            FreeSurfaceValveControlRegime.UPSTREAM_SUPPLY_CHOKED
        )
        if not downstream.is_dry:
            directional_downstream_velocity = (
                direction * downstream.velocity_m_s
            )
            if directional_downstream_velocity < downstream.celerity_m_s:
                raise ValueError(
                    "an upstream supply-choked valve requires a dry or "
                    "supercritical-outflow downstream trace"
                )
        z_equivalent = 0.0
        volume_flow = native_flow
        pressure_jump = float(
            quadratic_resistance * abs(volume_flow) * volume_flow
        )
        upstream_depth = _circular_depth_from_area(
            upstream.area_m2,
            upstream.full_area_m2,
        )
        upstream_specific_energy = float(
            upstream_depth
            + (native_flow / upwind_area) ** 2 / (2.0 * gravity)
        )
        critical_area, minimum_specific_energy = (
            _circular_critical_specific_energy(
                discharge_m3_s=native_flow,
                full_area_m2=upstream.full_area_m2,
                gravity_m_s2=gravity,
            )
        )
        choked_head_loss = float(
            abs(pressure_jump) / (density * gravity)
        )
        supply_energy_margin = float(
            upstream_specific_energy
            - choked_head_loss
            - minimum_specific_energy
        )
        energy_tolerance = float(
            512.0
            * math.ulp(1.0)
            * max(
                1.0,
                abs(upstream_specific_energy),
                abs(choked_head_loss),
                abs(minimum_specific_energy),
            )
        )
        if supply_energy_margin < -energy_tolerance:
            raise ValueError(
                "upstream supercritical supply cannot sustain the valve loss; "
                "a resolved upstream shock is required"
            )
        if upstream_side == "left":
            left_pressure_correction = 0.0
            right_pressure_correction = float(-pressure_jump)
        else:
            left_pressure_correction = pressure_jump
            right_pressure_correction = 0.0
        impedance_residual = 0.0
    else:
        if upstream_side == "left":
            left_active = True
            z_left = left.left_incoming_flow_impedance_Pa_s_m3
        else:
            right_active = True
            z_right = right.right_incoming_flow_impedance_Pa_s_m3

        if not downstream.is_dry:
            directional_downstream_velocity = (
                direction * downstream.velocity_m_s
            )
            if directional_downstream_velocity <= -downstream.celerity_m_s:
                raise ValueError(
                    "an opposing supercritical downstream trace sends two "
                    "characteristics into the valve"
                )
            if directional_downstream_velocity < downstream.celerity_m_s:
                if upstream_side == "left":
                    right_active = True
                    z_right = right.right_incoming_flow_impedance_Pa_s_m3
                else:
                    left_active = True
                    z_left = left.left_incoming_flow_impedance_Pa_s_m3

        z_equivalent = float(z_left + z_right)
        if z_equivalent <= 0.0:
            raise FloatingPointError(
                "a characteristic-controlled valve has no active impedance"
            )
        control_regime = (
            FreeSurfaceValveControlRegime.TWO_SIDED_CHARACTERISTIC
            if left_active and right_active
            else FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
        )
        if control_regime is FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC:
            (
                upwind_area,
                volume_flow,
                characteristic_head_rise,
            ) = _solve_native_offset_one_sided_characteristic(
                upstream=upstream,
                native_volume_flow_m3_s=native_flow,
                direction=direction,
                loss_coefficient=opening.loss_coefficient,
            )
            quadratic_resistance = float(
                0.5
                * density
                * opening.loss_coefficient
                / (upwind_area * upwind_area)
            )
            pressure_jump = float(
                quadratic_resistance * abs(volume_flow) * volume_flow
            )
            impedance_residual = float(
                direction
                * density
                * gravity
                * characteristic_head_rise
                - pressure_jump
            )
            nonlinear_characteristic = True
        else:
            magnitude = _stable_positive_quadratic_root(
                drive_pressure_Pa=z_equivalent * abs(native_flow),
                flow_impedance_Pa_s_m3=z_equivalent,
                quadratic_resistance_Pa_s2_m6=quadratic_resistance,
            )
            volume_flow = float(math.copysign(magnitude, native_flow))
            pressure_jump = float(
                quadratic_resistance * abs(volume_flow) * volume_flow
            )
            impedance_residual = float(
                z_equivalent * (native_flow - volume_flow) - pressure_jump
            )
        left_pressure_correction = float(
            z_left / z_equivalent * pressure_jump
        )
        right_pressure_correction = float(
            left_pressure_correction - pressure_jump
        )

    valve_velocity = float(volume_flow / upwind_area)
    common_advective_flux = float(
        native_momentum
        + (volume_flow * volume_flow - native_flow * native_flow)
        / upwind_area
    )
    left_specific_momentum = float(
        common_advective_flux
        + upwind_area * left_pressure_correction / density
    )
    wall_force = float(-pressure_jump * upwind_area)
    right_specific_momentum = float(
        left_specific_momentum + wall_force / density
    )
    dissipation = float(pressure_jump * volume_flow)
    if dissipation < 0.0:
        raise FloatingPointError(
            "passive circular Saint-Venant valve produced negative dissipation"
        )
    valve_head_loss = float(abs(pressure_jump) / (density * gravity))

    pressure_partition_residual = float(
        left_pressure_correction - right_pressure_correction - pressure_jump
    )
    momentum_force_residual = float(
        density
        * (right_specific_momentum - left_specific_momentum)
        - wall_force
    )
    return CircularSaintVenantValveSolution(
        opening=opening,
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=native_momentum,
        volume_flow_left_to_right_m3_s=volume_flow,
        upwind_wetted_area_m2=upwind_area,
        valve_velocity_m_s=valve_velocity,
        left_flow_impedance_Pa_s_m3=z_left,
        right_flow_impedance_Pa_s_m3=z_right,
        equivalent_flow_impedance_Pa_s_m3=z_equivalent,
        control_regime=control_regime,
        upstream_side=upstream_side,
        left_characteristic_active=left_active,
        right_characteristic_active=right_active,
        native_offset_nonlinear_characteristic=nonlinear_characteristic,
        signed_pressure_jump_Pa=pressure_jump,
        left_pressure_correction_Pa=left_pressure_correction,
        right_pressure_correction_Pa=right_pressure_correction,
        left_specific_momentum_flux_m4_s2=left_specific_momentum,
        right_specific_momentum_flux_m4_s2=right_specific_momentum,
        valve_wall_force_on_liquid_N=wall_force,
        dissipation_power_W=dissipation,
        valve_head_loss_m=valve_head_loss,
        upstream_specific_energy_m=upstream_specific_energy,
        critical_area_m2=critical_area,
        minimum_specific_energy_m=minimum_specific_energy,
        supply_energy_margin_m=supply_energy_margin,
        impedance_residual_Pa=impedance_residual,
        pressure_partition_residual_Pa=pressure_partition_residual,
        momentum_force_residual_N=momentum_force_residual,
        density_kg_m3=density,
        gravity_m_s2=gravity,
    )


@dataclass(frozen=True)
class LocalValveLedgerEntry:
    """One accepted equal-and-opposite valve transaction."""

    solution: LiquidValveSolution
    dt_s: float
    liquid_volume_left_to_right_m3: float
    left_liquid_volume_change_m3: float
    right_liquid_volume_change_m3: float
    liquid_volume_residual_m3: float
    valve_wall_impulse_on_liquid_N_s: float
    dissipated_energy_J: float
    cumulative_signed_through_volume_m3: float
    cumulative_absolute_through_volume_m3: float
    cumulative_valve_wall_impulse_on_liquid_N_s: float
    cumulative_dissipated_energy_J: float


@dataclass
class LocalValveLedger:
    """Integrate valve-only mass, wall-impulse, and dissipation accounts."""

    cumulative_signed_through_volume_m3: float = 0.0
    cumulative_absolute_through_volume_m3: float = 0.0
    cumulative_valve_wall_impulse_on_liquid_N_s: float = 0.0
    cumulative_dissipated_energy_J: float = 0.0
    commit_count: int = 0

    def commit(
        self,
        solution: LiquidValveSolution,
        *,
        dt_s: float,
    ) -> LocalValveLedgerEntry:
        """Commit one physical-time valve transaction exactly once."""

        if not isinstance(solution, LiquidValveSolution):
            raise TypeError("solution must be a LiquidValveSolution")
        step = _positive_finite(dt_s, name="dt_s")
        transferred = float(
            solution.volume_flow_left_to_right_m3_s * step
        )
        left_change = float(-transferred)
        right_change = float(transferred)
        residual = float(left_change + right_change)
        wall_impulse = float(
            solution.valve_wall_force_on_liquid_N * step
        )
        dissipated = float(solution.dissipation_power_W * step)
        if dissipated < 0.0:
            raise FloatingPointError("passive valve ledger received negative energy")

        self.cumulative_signed_through_volume_m3 += transferred
        self.cumulative_absolute_through_volume_m3 += abs(transferred)
        self.cumulative_valve_wall_impulse_on_liquid_N_s += wall_impulse
        self.cumulative_dissipated_energy_J += dissipated
        self.commit_count += 1
        return LocalValveLedgerEntry(
            solution=solution,
            dt_s=step,
            liquid_volume_left_to_right_m3=transferred,
            left_liquid_volume_change_m3=left_change,
            right_liquid_volume_change_m3=right_change,
            liquid_volume_residual_m3=residual,
            valve_wall_impulse_on_liquid_N_s=wall_impulse,
            dissipated_energy_J=dissipated,
            cumulative_signed_through_volume_m3=(
                self.cumulative_signed_through_volume_m3
            ),
            cumulative_absolute_through_volume_m3=(
                self.cumulative_absolute_through_volume_m3
            ),
            cumulative_valve_wall_impulse_on_liquid_N_s=(
                self.cumulative_valve_wall_impulse_on_liquid_N_s
            ),
            cumulative_dissipated_energy_J=(
                self.cumulative_dissipated_energy_J
            ),
        )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "commit_count": self.commit_count,
            "signed_through_volume_m3": (
                self.cumulative_signed_through_volume_m3
            ),
            "absolute_through_volume_m3": (
                self.cumulative_absolute_through_volume_m3
            ),
            "valve_wall_impulse_on_liquid_N_s": (
                self.cumulative_valve_wall_impulse_on_liquid_N_s
            ),
            "dissipated_energy_J": self.cumulative_dissipated_energy_J,
            "liquid_mass_source_m3": 0.0,
        }


def provenance() -> dict[str, object]:
    """Return the frozen shared physical/numerical contract."""

    return {
        "model": VALVE_MODEL_NAME,
        "opening_duration_s": OPENING_DURATION_S,
        "minimum_area_fraction": MINIMUM_AREA_FRACTION,
        "resistance_length_m": RESISTANCE_LENGTH_M,
        "default_water_density_kg_m3": WATER_DENSITY_KG_M3,
        "area_law": "phi=max(0.001,sin(pi/2*clip(t/0.20,0,1))^2)",
        "loss_coefficient": "K=phi^-2-1",
        "zero_thickness_jump": "dp=0.5*rho*K*abs(u)*u",
        "characteristic_model": CHARACTERISTIC_MODEL_NAME,
        "clean_free_surface_characteristic_model": (
            FREE_SURFACE_CHARACTERISTIC_MODEL_NAME
        ),
        "clean_free_surface_velocity_area": (
            "local upstream wetted area maps Q to liquid U: reconstructed "
            "for two-sided control and characteristic-solved for one-sided "
            "control; nominal/reference full area is not substituted"
        ),
        "clean_free_surface_control": (
            "directional characteristic count: two/one-sided subcritical "
            "control with native-offset nonlinear one-sided continuation, "
            "dry or supercritical-outflow downstream exclusion, and "
            "energy-gated upstream Riemann-supply choking"
        ),
        "openfoam_reference_flow_area_role": (
            "resistance-zone length audit only"
        ),
        "paper_support": "manual ball-valve opening takes approximately 0.2 s",
        "numerical_contract_support": (
            "identical Campaign-2 H1/H3/H6 OpenFOAM valveProperties and UEqn.H"
        ),
        "integration_status": (
            "directional wet/dry clean free-surface closure is available to "
            "the versioned Case-1 sibling; its shock/cut closure is integrated "
            "and tested, while MOC and coupled-driver integration remain pending"
        ),
    }


__all__ = [
    "CHARACTERISTIC_MODEL_NAME",
    "CircularSaintVenantValveSolution",
    "CircularSaintVenantValveTrace",
    "FREE_SURFACE_CHARACTERISTIC_MODEL_NAME",
    "FreeSurfaceValveControlRegime",
    "LocalValveLedger",
    "LocalValveLedgerEntry",
    "LiquidValveSolution",
    "LiquidValveTrace",
    "MINIMUM_AREA_FRACTION",
    "OPENING_DURATION_S",
    "PressurisedMocValveControlRegime",
    "PressurisedMocValveNoRootError",
    "PressurisedMocValveSolution",
    "RESISTANCE_LENGTH_M",
    "VALVE_MODEL_NAME",
    "ValveOpeningState",
    "WATER_DENSITY_KG_M3",
    "provenance",
    "shared_opening_state",
    "solve_passive_circular_saint_venant_valve",
    "solve_passive_liquid_valve",
    "solve_passive_pressurised_moc_valve",
]
