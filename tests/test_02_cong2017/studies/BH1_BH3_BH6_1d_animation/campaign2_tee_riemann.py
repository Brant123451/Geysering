"""Case-independent conservative side-T node solvers for Campaign 2.

All branch coordinates point outward from the zero-volume T node.  The liquid
solver closes three incoming water-hammer characteristics with one node
pressure and exact volume continuity.  The gas solver uses two acoustic
impedances to obtain one shared interface state.  Neither solver knows a case
identifier or a target geyser outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


# The three-branch solve below contains only a small, fixed number of binary64
# operations.  This multiplier covers their accumulated roundoff while keeping
# the admissible residual at machine precision relative to the *dimensional*
# branch-flux terms.  It is deliberately independent of case geometry and flow
# outcome.
_LIQUID_CONTINUITY_ROUNDOFF_FACTOR = 64.0


def _checked_liquid_continuity_residual_m3_s(
    flows_m3_s: Sequence[float],
    balance_terms_m3_s: Sequence[float],
) -> float:
    """Return the continuity residual or reject a resolved imbalance.

    Near a stationary pressurised node, each returned branch flow is the
    difference between two O(1e-4 m3/s) characteristic contributions.  Scaling
    the check by the tiny *net* flows therefore treats ordinary cancellation
    roundoff as a physical mass defect.  The correct dimensional scale is the
    sum of the magnitudes of the incoming-characteristic and pressure-correction
    fluxes used to form those flows.
    """

    flows = tuple(float(value) for value in flows_m3_s)
    terms = tuple(float(value) for value in balance_terms_m3_s)
    if not flows or not terms:
        raise ValueError("liquid continuity check requires flows and balance terms")
    if not all(math.isfinite(value) for value in (*flows, *terms)):
        raise FloatingPointError("liquid T-node continuity data are non-finite")

    residual = float(math.fsum(flows))
    characteristic_flow = float(math.fsum(abs(value) for value in terms))
    tolerance = (
        _LIQUID_CONTINUITY_ROUNDOFF_FACTOR
        * math.ulp(1.0)
        * characteristic_flow
    )
    if abs(residual) > tolerance:
        raise FloatingPointError(
            "liquid T-node continuity did not close: "
            f"residual={residual:.17g} m3/s exceeds binary64 roundoff "
            f"tolerance={tolerance:.17g} m3/s at characteristic flow "
            f"{characteristic_flow:.17g} m3/s"
        )
    return residual


@dataclass(frozen=True)
class LiquidBranchTrace:
    area_m2: float
    outward_velocity_m_s: float
    gauge_pressure_Pa: float
    wave_speed_m_s: float
    density_kg_m3: float = 998.0

    def __post_init__(self) -> None:
        positive = (
            self.area_m2,
            self.wave_speed_m_s,
            self.density_kg_m3,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in positive):
            raise ValueError("liquid branch area, wave speed and density must be positive")
        if not math.isfinite(float(self.outward_velocity_m_s)):
            raise ValueError("liquid branch velocity must be finite")
        if not math.isfinite(float(self.gauge_pressure_Pa)):
            raise ValueError("liquid branch pressure must be finite")

    @property
    def incoming_characteristic_m_s(self) -> float:
        return float(
            self.outward_velocity_m_s
            - self.gauge_pressure_Pa
            / (self.density_kg_m3 * self.wave_speed_m_s)
        )


@dataclass(frozen=True)
class LiquidTeeSolution:
    node_gauge_pressure_Pa: float
    west_outward_flow_m3_s: float
    east_outward_flow_m3_s: float
    riser_outward_flow_m3_s: float
    continuity_residual_m3_s: float
    # Convective normal momentum flux at the liquid part of the riser port.
    # The common node-pressure force is deliberately excluded.
    normal_momentum_to_riser_N: float = 0.0
    # Authoritative geometric liquid share of the physical riser mouth.  This
    # remains meaningful when both Ql and rho*Ql^2/Al are exactly zero, so a
    # receiver must never try to reconstruct it from momentum.
    riser_open_area_m2: float | None = None

    def __post_init__(self) -> None:
        if self.riser_open_area_m2 is not None and (
            not math.isfinite(float(self.riser_open_area_m2))
            or float(self.riser_open_area_m2) < 0.0
        ):
            raise ValueError("liquid riser opening must be finite and non-negative")

    @property
    def physical_west_flow_m3_s(self) -> float:
        """West-face flow in the paper coordinate, positive to physical east."""

        return -float(self.west_outward_flow_m3_s)

    @property
    def physical_east_flow_m3_s(self) -> float:
        return float(self.east_outward_flow_m3_s)


def solve_liquid_tee_with_blocked_riser(
    west: LiquidBranchTrace,
    east: LiquidBranchTrace,
) -> LiquidTeeSolution:
    """Close the horizontal liquid node when a finite riser gas pocket blocks it.

    Once the persisted lower material front exists, the physical bottom face of
    the one-dimensional riser belongs to gas.  The liquid slug above that front
    is therefore *not* a third liquid characteristic at the zero-volume T node.
    The two horizontal incoming characteristics retain one common liquid-node
    pressure and exact ``q_w + q_e = 0``; the riser liquid flux and its
    convective momentum are identically zero.  Gas pressure at the separate
    gas opening is solved independently by :func:`solve_gas_tee`.

    This is a topology closure, not a valve: it is selected only from the
    persisted ``gas_below_liquid_above`` material component.  A common shift of
    both liquid gauge traces shifts the returned gauge pressure by the same
    amount and leaves both physical flows unchanged.
    """

    branches = (west, east)
    incoming_fluxes = tuple(
        branch.area_m2 * branch.incoming_characteristic_m_s
        for branch in branches
    )
    compliance = math.fsum(
        branch.area_m2
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    if not math.isfinite(compliance) or compliance <= 0.0:
        raise FloatingPointError("blocked-riser liquid node has no compliance")
    pressure = -math.fsum(incoming_fluxes) / compliance
    pressure_corrections = tuple(
        branch.area_m2
        * pressure
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    flows = tuple(
        incoming + correction
        for incoming, correction in zip(incoming_fluxes, pressure_corrections)
    )
    _checked_liquid_continuity_residual_m3_s(
        flows,
        (*incoming_fluxes, *pressure_corrections),
    )
    # Project the already roundoff-closed pair onto the exact two-branch
    # constraint.  Otherwise west-east subtraction can expose an ulp-scale
    # fictitious riser flow after the physical liquid port has been blocked.
    through_flow = 0.5 * (flows[0] - flows[1])
    return LiquidTeeSolution(
        node_gauge_pressure_Pa=float(pressure),
        west_outward_flow_m3_s=float(through_flow),
        east_outward_flow_m3_s=float(-through_flow),
        riser_outward_flow_m3_s=0.0,
        continuity_residual_m3_s=0.0,
        normal_momentum_to_riser_N=0.0,
        riser_open_area_m2=0.0,
    )


def solve_liquid_tee(
    west: LiquidBranchTrace,
    east: LiquidBranchTrace,
    riser: LiquidBranchTrace,
) -> LiquidTeeSolution:
    """Solve the lossless three-characteristic liquid node exactly."""

    branches = (west, east, riser)
    incoming_fluxes = tuple(
        branch.area_m2 * branch.incoming_characteristic_m_s
        for branch in branches
    )
    numerator = math.fsum(incoming_fluxes)
    compliance = math.fsum(
        branch.area_m2
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    if not math.isfinite(compliance) or compliance <= 0.0:
        raise FloatingPointError("liquid T node has no positive compliance")
    pressure = -numerator / compliance
    pressure_correction_fluxes = tuple(
        branch.area_m2
        * pressure
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    flows = tuple(
        incoming_flux + pressure_correction_flux
        for incoming_flux, pressure_correction_flux in zip(
            incoming_fluxes,
            pressure_correction_fluxes,
        )
    )
    residual = _checked_liquid_continuity_residual_m3_s(
        flows,
        (*incoming_fluxes, *pressure_correction_fluxes),
    )
    return LiquidTeeSolution(
        node_gauge_pressure_Pa=float(pressure),
        west_outward_flow_m3_s=float(flows[0]),
        east_outward_flow_m3_s=float(flows[1]),
        riser_outward_flow_m3_s=float(flows[2]),
        continuity_residual_m3_s=residual,
        normal_momentum_to_riser_N=float(
            riser.density_kg_m3 * flows[2] * flows[2] / riser.area_m2
        ),
        riser_open_area_m2=float(riser.area_m2),
    )


@dataclass(frozen=True)
class GasTrace:
    pressure_abs_Pa: float
    density_kg_m3: float
    normal_velocity_m_s: float
    sound_speed_m_s: float

    def __post_init__(self) -> None:
        positive = (
            self.pressure_abs_Pa,
            self.density_kg_m3,
            self.sound_speed_m_s,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in positive):
            raise ValueError("gas pressure, density and sound speed must be positive")
        if not math.isfinite(float(self.normal_velocity_m_s)):
            raise ValueError("gas normal velocity must be finite")

    @property
    def impedance_Pa_s_m(self) -> float:
        return float(self.density_kg_m3 * self.sound_speed_m_s)


@dataclass(frozen=True)
class GasTeeSolution:
    interface_pressure_abs_Pa: float
    interface_velocity_to_riser_m_s: float
    mass_flow_to_riser_kg_s: float
    # Signed volume flux formed with the actual upwind donor density used by
    # the gas Riemann solve.  It cannot be reconstructed from a receiving EOS.
    volume_flow_to_riser_m3_s: float
    # Convective momentum flux ``mdot*u`` only.  The vertical FV kernel
    # applies T-face pressure separately through its common-pressure field.
    normal_momentum_flow_N: float
    interface_pressure_force_N: float = 0.0
    # Authoritative geometric gas share used by the Riemann solve.  Retaining
    # it avoids the singular Q/u reconstruction at a quiescent open port.
    open_area_m2: float | None = None

    def __post_init__(self) -> None:
        if self.open_area_m2 is not None and (
            not math.isfinite(float(self.open_area_m2))
            or float(self.open_area_m2) < 0.0
        ):
            raise ValueError("gas T opening must be finite and non-negative")

    @property
    def total_conservative_momentum_flux_N(self) -> float:
        """Return ``mdot*u + A*p`` for an unsplit Euler formulation."""

        return float(
            self.normal_momentum_flow_N + self.interface_pressure_force_N
        )


def solve_gas_tee(
    horizontal: GasTrace,
    riser: GasTrace,
    *,
    open_area_m2: float,
) -> GasTeeSolution:
    """Solve a two-impedance acoustic gas interface at the T opening."""

    area = float(open_area_m2)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("gas T opening area must be finite and non-negative")
    if area == 0.0:
        pressure = 0.5 * (
            horizontal.pressure_abs_Pa + riser.pressure_abs_Pa
        )
        return GasTeeSolution(
            float(pressure), 0.0, 0.0, 0.0, 0.0, 0.0, open_area_m2=0.0
        )

    z_h = horizontal.impedance_Pa_s_m
    z_r = riser.impedance_Pa_s_m
    denominator = z_h + z_r
    velocity = (
        horizontal.pressure_abs_Pa
        - riser.pressure_abs_Pa
        + z_h * horizontal.normal_velocity_m_s
        + z_r * riser.normal_velocity_m_s
    ) / denominator
    pressure = (
        z_r * horizontal.pressure_abs_Pa
        + z_h * riser.pressure_abs_Pa
        + z_h
        * z_r
        * (
            horizontal.normal_velocity_m_s
            - riser.normal_velocity_m_s
        )
    ) / denominator
    upwind_density = (
        horizontal.density_kg_m3
        if velocity >= 0.0
        else riser.density_kg_m3
    )
    volume_flow = area * velocity
    mass_flow = upwind_density * volume_flow
    convective_momentum_flow = mass_flow * velocity
    pressure_force = area * pressure
    values = (
        velocity,
        pressure,
        mass_flow,
        volume_flow,
        convective_momentum_flow,
        pressure_force,
    )
    if not all(math.isfinite(float(value)) for value in values) or pressure <= 0.0:
        raise FloatingPointError("gas T-node solve produced an inadmissible state")
    return GasTeeSolution(
        interface_pressure_abs_Pa=float(pressure),
        interface_velocity_to_riser_m_s=float(velocity),
        mass_flow_to_riser_kg_s=float(mass_flow),
        volume_flow_to_riser_m3_s=float(volume_flow),
        normal_momentum_flow_N=float(convective_momentum_flow),
        interface_pressure_force_N=float(pressure_force),
        open_area_m2=area,
    )


@dataclass(frozen=True)
class FirstBottomGasEntrySolution:
    """Common-pressure characteristic solution for a dry riser gas port.

    The riser receiver contains liquid but no gas.  Consequently it supplies a
    liquid-piston characteristic, not a fictitious atmospheric gas impedance.
    ``gas_open_area_m2`` is the geometric overlap of the material horizontal
    void with the riser mouth; the remaining non-overlapping mouth area belongs
    to the liquid port for the time-integrated first-entry transaction.
    """

    active: bool
    liquid: LiquidTeeSolution
    gas: GasTeeSolution
    common_pressure_abs_Pa: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    liquid_plug_flow_m3_s: float
    ale_volume_residual_m3_s: float
    common_pressure_residual_Pa: float


def solve_first_bottom_gas_entry(
    west: LiquidBranchTrace,
    east: LiquidBranchTrace,
    riser_liquid: LiquidBranchTrace,
    horizontal_gas: GasTrace,
    *,
    liquid_pressure_reference_abs_Pa: float,
    available_gas_open_area_m2: float,
    closed_liquid_solution: LiquidTeeSolution | None = None,
) -> FirstBottomGasEntrySolution:
    """Solve the unilateral first gas-entry/piston characteristic problem.

    With all branch coordinates outward from the T node, the liquid incoming
    characteristics give ``q_i(p*)``.  The horizontal gas supplies the one
    incoming acoustic relation

    ``Qg(p*) = Ag [u_h + (p_h-p*)/(rho_h*c_h)]``.

    The first-entry ALE identity is ``Ql,plug = Ql,0 + Qg`` while horizontal
    liquid continuity gives ``q_w+q_e+Ql,0=0``.  Therefore the scalar residual

    ``q_w(p*) + q_e(p*) + Ql,plug(p*) - Qg(p*)``

    has the strictly positive slope ``sum(A/(rho*c))+Ag/(rho_h*c_h)`` and one
    unique root.  If the closed, zero-gas solution predicts non-positive gas
    velocity, the unilateral gas port remains closed and the original liquid
    T solution is returned *exactly*.  No receiving-gas state or seed void is
    introduced.
    """

    p_ref = float(liquid_pressure_reference_abs_Pa)
    gas_area = float(available_gas_open_area_m2)
    if not math.isfinite(p_ref) or p_ref <= 0.0:
        raise ValueError("liquid pressure reference must be positive and finite")
    if not math.isfinite(gas_area) or gas_area < 0.0:
        raise ValueError("available gas opening must be finite and non-negative")
    if gas_area > riser_liquid.area_m2:
        raise ValueError("gas opening cannot exceed the physical riser area")

    branches = (west, east, riser_liquid)
    closed_liquid = (
        solve_liquid_tee(*branches)
        if closed_liquid_solution is None
        else closed_liquid_solution
    )
    closed_pressure_abs = p_ref + closed_liquid.node_gauge_pressure_Pa
    if gas_area == 0.0:
        # Exact transparent limit: retain the caller's already pinned/native
        # liquid solution byte-for-byte.  A closed gas port does not make an
        # otherwise diagnostic liquid trace inadmissible.
        zero_pressure = (
            closed_pressure_abs
            if math.isfinite(closed_pressure_abs) and closed_pressure_abs > 0.0
            else horizontal_gas.pressure_abs_Pa
        )
        zero_gas = GasTeeSolution(
            interface_pressure_abs_Pa=float(zero_pressure),
            interface_velocity_to_riser_m_s=0.0,
            mass_flow_to_riser_kg_s=0.0,
            volume_flow_to_riser_m3_s=0.0,
            normal_momentum_flow_N=0.0,
            interface_pressure_force_N=0.0,
            open_area_m2=0.0,
        )
        return FirstBottomGasEntrySolution(
            active=False,
            liquid=closed_liquid,
            gas=zero_gas,
            common_pressure_abs_Pa=float(zero_pressure),
            gas_open_area_m2=0.0,
            liquid_open_area_m2=float(riser_liquid.area_m2),
            liquid_plug_flow_m3_s=float(
                closed_liquid.riser_outward_flow_m3_s
            ),
            ale_volume_residual_m3_s=0.0,
            common_pressure_residual_Pa=0.0,
        )
    if not math.isfinite(closed_pressure_abs) or closed_pressure_abs <= 0.0:
        raise FloatingPointError("closed liquid T pressure is not positive")
    predicted_velocity = (
        horizontal_gas.normal_velocity_m_s
        + (horizontal_gas.pressure_abs_Pa - closed_pressure_abs)
        / horizontal_gas.impedance_Pa_s_m
    )
    if predicted_velocity <= 0.0:
        zero_gas = GasTeeSolution(
            interface_pressure_abs_Pa=float(closed_pressure_abs),
            interface_velocity_to_riser_m_s=0.0,
            mass_flow_to_riser_kg_s=0.0,
            volume_flow_to_riser_m3_s=0.0,
            normal_momentum_flow_N=0.0,
            interface_pressure_force_N=0.0,
            open_area_m2=0.0,
        )
        return FirstBottomGasEntrySolution(
            active=False,
            liquid=closed_liquid,
            gas=zero_gas,
            common_pressure_abs_Pa=float(closed_pressure_abs),
            gas_open_area_m2=0.0,
            liquid_open_area_m2=float(riser_liquid.area_m2),
            liquid_plug_flow_m3_s=float(
                closed_liquid.riser_outward_flow_m3_s
            ),
            ale_volume_residual_m3_s=0.0,
            common_pressure_residual_Pa=0.0,
        )

    incoming = tuple(
        branch.area_m2 * branch.incoming_characteristic_m_s
        for branch in branches
    )
    liquid_compliance = math.fsum(
        branch.area_m2 / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    gas_compliance = gas_area / horizontal_gas.impedance_Pa_s_m
    numerator = (
        gas_area
        * (
            horizontal_gas.normal_velocity_m_s
            + (horizontal_gas.pressure_abs_Pa - p_ref)
            / horizontal_gas.impedance_Pa_s_m
        )
        - math.fsum(incoming)
    )
    denominator = liquid_compliance + gas_compliance
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("first-entry characteristic has no compliance")
    pressure_gauge = numerator / denominator
    pressure_abs = p_ref + pressure_gauge
    gas_velocity = (
        horizontal_gas.normal_velocity_m_s
        + (horizontal_gas.pressure_abs_Pa - pressure_abs)
        / horizontal_gas.impedance_Pa_s_m
    )
    gas_volume_flow = gas_area * gas_velocity
    if (
        not _finite_solution_values(
            pressure_gauge,
            pressure_abs,
            gas_velocity,
            gas_volume_flow,
        )
        or pressure_abs <= 0.0
        or gas_velocity <= 0.0
        or gas_volume_flow <= 0.0
    ):
        raise FloatingPointError(
            "active first-entry characteristic produced an inadmissible state"
        )

    pressure_corrections = tuple(
        branch.area_m2
        * pressure_gauge
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    characteristic_flows = tuple(
        datum + correction
        for datum, correction in zip(incoming, pressure_corrections)
    )
    west_flow, east_flow, plug_flow = characteristic_flows
    bottom_liquid_flow = -(west_flow + east_flow)
    ale_residual = plug_flow - bottom_liquid_flow - gas_volume_flow
    residual_scale = math.fsum(
        abs(value)
        for value in (
            west_flow,
            east_flow,
            plug_flow,
            bottom_liquid_flow,
            gas_volume_flow,
        )
    )
    residual_tolerance = (
        _LIQUID_CONTINUITY_ROUNDOFF_FACTOR
        * math.ulp(1.0)
        * residual_scale
    )
    if abs(ale_residual) > residual_tolerance:
        raise FloatingPointError(
            "first-entry liquid/gas ALE characteristic did not close"
        )
    liquid_continuity = _checked_liquid_continuity_residual_m3_s(
        (west_flow, east_flow, bottom_liquid_flow),
        (*incoming[:2], *pressure_corrections[:2], bottom_liquid_flow),
    )

    liquid_area = riser_liquid.area_m2 - gas_area
    liquid_area_roundoff = 16.0 * math.ulp(riser_liquid.area_m2)
    if liquid_area < -liquid_area_roundoff:
        raise FloatingPointError("first-entry phase openings overlap")
    liquid_area = max(float(liquid_area), 0.0)
    if bottom_liquid_flow != 0.0 and liquid_area == 0.0:
        raise FloatingPointError(
            "first-entry solution requires liquid flow through a zero liquid opening"
        )
    liquid_normal_momentum = (
        0.0
        if bottom_liquid_flow == 0.0
        else riser_liquid.density_kg_m3
        * bottom_liquid_flow
        * bottom_liquid_flow
        / liquid_area
    )
    mass_flow = horizontal_gas.density_kg_m3 * gas_volume_flow
    gas_normal_momentum = mass_flow * gas_velocity
    gas_solution = GasTeeSolution(
        interface_pressure_abs_Pa=float(pressure_abs),
        interface_velocity_to_riser_m_s=float(gas_velocity),
        mass_flow_to_riser_kg_s=float(mass_flow),
        volume_flow_to_riser_m3_s=float(gas_volume_flow),
        normal_momentum_flow_N=float(gas_normal_momentum),
        interface_pressure_force_N=float(gas_area * pressure_abs),
        open_area_m2=float(gas_area),
    )
    liquid_solution = LiquidTeeSolution(
        node_gauge_pressure_Pa=float(pressure_gauge),
        west_outward_flow_m3_s=float(west_flow),
        east_outward_flow_m3_s=float(east_flow),
        riser_outward_flow_m3_s=float(bottom_liquid_flow),
        continuity_residual_m3_s=float(liquid_continuity),
        normal_momentum_to_riser_N=float(liquid_normal_momentum),
        riser_open_area_m2=float(liquid_area),
    )
    return FirstBottomGasEntrySolution(
        active=True,
        liquid=liquid_solution,
        gas=gas_solution,
        common_pressure_abs_Pa=float(pressure_abs),
        gas_open_area_m2=float(gas_area),
        liquid_open_area_m2=float(liquid_area),
        liquid_plug_flow_m3_s=float(plug_flow),
        ale_volume_residual_m3_s=float(ale_residual),
        common_pressure_residual_Pa=0.0,
    )


def _finite_solution_values(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


__all__ = [
    "FirstBottomGasEntrySolution",
    "GasTeeSolution",
    "GasTrace",
    "LiquidBranchTrace",
    "LiquidTeeSolution",
    "solve_gas_tee",
    "solve_first_bottom_gas_entry",
    "solve_liquid_tee",
    "solve_liquid_tee_with_blocked_riser",
]
