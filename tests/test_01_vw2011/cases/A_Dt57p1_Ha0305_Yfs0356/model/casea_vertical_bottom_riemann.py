"""Gross-first directional trace at the Case-A riser bottom.

The lower face of the persistent vertical two-stream model has two different
characteristic owners:

* upward liquid is an incoming characteristic supplied by the finite T node;
* falling liquid is an outgoing characteristic supplied by the first riser
  cell.

Those two gross rates are resolved independently and the signed node flux is
then the identity ``q_net = Q_up - Q_down``.  A preselected signed net flux is
not an input: imposing it in addition to the outgoing falling characteristic
over-determines the boundary and reflects the falling water back into the
riser.

The falling trace keeps the *cell velocity*.  When the mouth phase resolver
assigns a narrower falling-film corridor than the first-cell falling area, its
candidate rate is ``u_down,cell * A_down,mouth``.  Keeping the complete cell
discharge while shrinking the area would spuriously increase velocity.

Wallis is retained as a quasi-steady audit reference.  An inherited outgoing
film is not instantaneously clipped by that reference during three-stream
churn; distributed gas/liquid drag supplies its finite-time reaction.  A
caller may explicitly activate the Wallis inequality only for a topology in
which that quasi-steady closure is applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_vertical_twostream_fv import DirectionalBoundaryFlux


class BottomMouthRiemannError(RuntimeError):
    """An admissible bottom-face directional trace could not be formed."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _finite_or_positive_infinity(value: float) -> bool:
    number = float(value)
    return math.isfinite(number) or number == math.inf


@dataclass(frozen=True)
class UpwardIncomingCharacteristic:
    """One-sided liquid characteristic entering the riser from the T node."""

    old_speed: float
    unconstrained_speed: float
    accepted_speed: float
    hydraulic_driving_pressure: float
    pressure_inertance: float
    inertive_pressure: float
    turn_loss_pressure: float
    lower_bound_reaction_pressure: float
    pressure_residual: float


@dataclass(frozen=True)
class CoupledGrossMouthCharacteristic:
    """Implicit upward/downward mouth state with one mixing action--reaction."""

    old_upward_speed: float
    raw_downward_speed: float
    uncoupled_upward_speed: float
    uncoupled_downward_speed: float
    upward_speed: float
    downward_speed: float
    hydraulic_driving_pressure: float
    mixing_force: float
    mixing_kinematic_reaction_flux: float
    upward_inertive_pressure: float
    upward_turn_pressure: float
    upward_mixing_pressure: float
    upward_lower_bound_reaction_pressure: float
    downward_characteristic_pressure: float
    downward_turn_pressure: float
    downward_mixing_pressure: float
    upward_pressure_residual: float
    downward_pressure_residual: float
    downward_turn_reaction_flux: float
    upward_turn_loss_power: float
    downward_turn_loss_power: float
    mixing_loss_power: float
    iterations: int


def _positive_quadratic_root(
    *,
    linear: float,
    quadratic: float,
    right_hand_side: float,
) -> float:
    """Return the stable non-negative root of ``b*x+a*x**2=rhs``."""

    if right_hand_side <= 0.0:
        return 0.0
    if quadratic <= 0.0:
        return right_hand_side / linear
    return 2.0 * right_hand_side / (
        linear
        + math.sqrt(
            linear * linear + 4.0 * quadratic * right_hand_side
        )
    )


def solve_coupled_gross_mouth_characteristics(
    *,
    old_upward_speed: float,
    raw_downward_speed: float,
    upward_area: float,
    downward_area: float,
    horizontal_liquid_pressure_abs: float,
    vertical_liquid_pressure_abs: float,
    liquid_density: float,
    effective_inertance_length: float,
    time_step: float,
    downward_characteristic_celerity: float,
    upward_turn_loss_coefficient: float,
    downward_turn_loss_coefficient: float,
    countercurrent_mixing_coefficient: float,
    dry_area_tolerance: float = 1.0e-14,
) -> CoupledGrossMouthCharacteristic:
    """Solve the two gross liquid traces with a single dissipative mixing force.

    The mixing force is equal and opposite on the two liquid streams,

    ``F_mix = 0.5*rho*K_mix*min(Au*u, Ad*d)*(u+d)``.

    Consequently it contributes no net liquid momentum, while its dissipated
    power is exactly ``F_mix*(u+d)``.  The falling trace remains an outgoing
    characteristic of the first riser cell; the local turn and mixing forces
    reduce its mouth speed over the gravity-wave impedance instead of clipping
    its inherited discharge instantaneously.
    """

    values = (
        old_upward_speed,
        raw_downward_speed,
        upward_area,
        downward_area,
        horizontal_liquid_pressure_abs,
        vertical_liquid_pressure_abs,
        liquid_density,
        effective_inertance_length,
        time_step,
        downward_characteristic_celerity,
        upward_turn_loss_coefficient,
        downward_turn_loss_coefficient,
        countercurrent_mixing_coefficient,
        dry_area_tolerance,
    )
    if not _finite(*values):
        raise ValueError("coupled gross-mouth inputs must be finite")
    (
        old_u,
        raw_d,
        area_u,
        area_d,
        pressure_h,
        pressure_v,
        density,
        inertance_length,
        dt,
        celerity,
        loss_u,
        loss_d,
        loss_mix,
        area_tolerance,
    ) = (float(value) for value in values)
    if (
        min(old_u, raw_d, area_u, area_d, loss_u, loss_d, loss_mix) < 0.0
        or min(
            pressure_h,
            pressure_v,
            density,
            inertance_length,
            dt,
            celerity,
            area_tolerance,
        )
        <= 0.0
    ):
        raise ValueError(
            "gross-mouth speeds, areas and losses must be non-negative; "
            "pressures, material scales and tolerances must be positive"
        )

    drive = pressure_h - pressure_v
    inertance = inertance_length / dt
    upward_rhs = drive / density + inertance * old_u
    uncoupled_u = (
        _positive_quadratic_root(
            linear=inertance,
            quadratic=0.5 * loss_u,
            right_hand_side=upward_rhs,
        )
        if area_u > area_tolerance
        else 0.0
    )
    uncoupled_d = (
        _positive_quadratic_root(
            linear=celerity,
            quadratic=0.5 * loss_d,
            right_hand_side=celerity * raw_d,
        )
        if area_d > area_tolerance and raw_d > 0.0
        else 0.0
    )

    def state_residuals(u: float, d: float) -> tuple[float, float, float]:
        circulation = min(area_u * u, area_d * d)
        mixing_kinematic = (
            0.5 * loss_mix * circulation * (u + d)
        )
        upward_residual = (
            inertance * (u - old_u)
            + 0.5 * loss_u * u * u
            + (
                mixing_kinematic / area_u
                if area_u > area_tolerance
                else 0.0
            )
            - drive / density
        )
        downward_residual = (
            celerity * (d - raw_d)
            + 0.5 * loss_d * d * d
            + (
                mixing_kinematic / area_d
                if area_d > area_tolerance
                else 0.0
            )
        )
        return upward_residual, downward_residual, mixing_kinematic

    u = uncoupled_u
    d = uncoupled_d
    scale = max(
        abs(drive) / density,
        inertance * max(old_u, uncoupled_u),
        celerity * max(raw_d, uncoupled_d),
        1.0e-10,
    )
    tolerance = 2.0e-11 * scale + 2.0e-13
    iterations = 0
    converged = False
    if uncoupled_u == 0.0 or uncoupled_d == 0.0 or loss_mix == 0.0:
        iterations = 1
        converged = True
    else:
        for iterations in range(1, 17):
            ru, rd, mixing_kinematic = state_residuals(u, d)
            norm = max(abs(ru), abs(rd))
            if norm <= tolerance:
                converged = True
                break
            upward_limited = area_u * u <= area_d * d
            if upward_limited:
                dm_du = 0.5 * loss_mix * area_u * (2.0 * u + d)
                dm_dd = 0.5 * loss_mix * area_u * u
            else:
                dm_du = 0.5 * loss_mix * area_d * d
                dm_dd = 0.5 * loss_mix * area_d * (u + 2.0 * d)
            juu = inertance + loss_u * u + dm_du / area_u
            jud = dm_dd / area_u
            jdu = dm_du / area_d
            jdd = celerity + loss_d * d + dm_dd / area_d
            determinant = juu * jdd - jud * jdu
            if determinant <= 0.0 or not math.isfinite(determinant):
                break
            delta_u = (-ru * jdd + jud * rd) / determinant
            delta_d = (jdu * ru - juu * rd) / determinant
            accepted_trial = False
            damping = 1.0
            for _ in range(14):
                trial_u = min(max(u + damping * delta_u, 0.0), uncoupled_u)
                trial_d = min(max(d + damping * delta_d, 0.0), uncoupled_d)
                trial_ru, trial_rd, _ = state_residuals(trial_u, trial_d)
                trial_norm = max(abs(trial_ru), abs(trial_rd))
                if trial_norm < norm or damping <= 2.0**-13:
                    u, d = trial_u, trial_d
                    accepted_trial = True
                    break
                damping *= 0.5
            if not accepted_trial:
                break

    if not converged:
        # Monotone coordinate fallback.  This path is rare (normally only a
        # regime switch exactly at Au*u == Ad*d) and remains far cheaper than
        # the removed 80-iteration q_net/q_c fixed point.
        u = uncoupled_u
        d = uncoupled_d
        for outer in range(1, 13):
            if uncoupled_u > 0.0:
                lo, hi = 0.0, uncoupled_u
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    ru, _, _ = state_residuals(mid, d)
                    if ru > 0.0:
                        hi = mid
                    else:
                        lo = mid
                u = 0.5 * (lo + hi)
            if uncoupled_d > 0.0:
                lo, hi = 0.0, uncoupled_d
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    _, rd, _ = state_residuals(u, mid)
                    if rd > 0.0:
                        hi = mid
                    else:
                        lo = mid
                d = 0.5 * (lo + hi)
            ru, rd, _ = state_residuals(u, d)
            iterations = 16 + outer
            if max(abs(ru), abs(rd)) <= 20.0 * tolerance:
                converged = True
                break
    ru, rd, mixing_kinematic = state_residuals(u, d)
    if not converged and max(abs(ru), abs(rd)) > 50.0 * tolerance:
        raise BottomMouthRiemannError(
            "coupled gross-mouth characteristic solve did not converge"
        )

    mixing_force = density * mixing_kinematic
    upward_rate = area_u * u
    downward_rate = area_d * d
    upward_inertive_pressure = density * inertance * (u - old_u)
    upward_turn_pressure = 0.5 * density * loss_u * u * u
    upward_mixing_pressure = (
        mixing_force / area_u if area_u > area_tolerance else 0.0
    )
    downward_characteristic_pressure = density * celerity * (d - raw_d)
    downward_turn_pressure = 0.5 * density * loss_d * d * d
    downward_mixing_pressure = (
        mixing_force / area_d if area_d > area_tolerance else 0.0
    )
    raw_upward_pressure_residual = math.fsum(
        (
            upward_inertive_pressure,
            upward_turn_pressure,
            upward_mixing_pressure,
            -drive,
        )
    )
    upward_lower_bound_reaction_pressure = (
        max(raw_upward_pressure_residual, 0.0)
        if u == 0.0 and uncoupled_u == 0.0
        else 0.0
    )
    return CoupledGrossMouthCharacteristic(
        old_upward_speed=float(old_u),
        raw_downward_speed=float(raw_d),
        uncoupled_upward_speed=float(uncoupled_u),
        uncoupled_downward_speed=float(uncoupled_d),
        upward_speed=float(u),
        downward_speed=float(d),
        hydraulic_driving_pressure=float(drive),
        mixing_force=float(mixing_force),
        mixing_kinematic_reaction_flux=float(mixing_kinematic),
        upward_inertive_pressure=float(upward_inertive_pressure),
        upward_turn_pressure=float(upward_turn_pressure),
        upward_mixing_pressure=float(upward_mixing_pressure),
        upward_lower_bound_reaction_pressure=float(
            upward_lower_bound_reaction_pressure
        ),
        downward_characteristic_pressure=float(
            downward_characteristic_pressure
        ),
        downward_turn_pressure=float(downward_turn_pressure),
        downward_mixing_pressure=float(downward_mixing_pressure),
        upward_pressure_residual=float(
            raw_upward_pressure_residual
            - upward_lower_bound_reaction_pressure
        ),
        downward_pressure_residual=float(
            downward_characteristic_pressure
            + downward_turn_pressure
            + downward_mixing_pressure
        ),
        downward_turn_reaction_flux=float(
            0.5 * loss_d * downward_rate * d
        ),
        upward_turn_loss_power=float(
            0.5 * density * loss_u * upward_rate * u * u
        ),
        downward_turn_loss_power=float(
            0.5 * density * loss_d * downward_rate * d * d
        ),
        mixing_loss_power=float(mixing_force * (u + d)),
        iterations=int(iterations),
    )


def solve_upward_incoming_characteristic(
    *,
    old_upward_speed: float,
    horizontal_liquid_pressure_abs: float,
    vertical_liquid_pressure_abs: float,
    liquid_density: float,
    effective_inertance_length: float,
    time_step: float,
    upward_turn_loss_coefficient: float,
) -> UpwardIncomingCharacteristic:
    """Solve the non-negative incoming water characteristic and T loss.

    Pressure on the horizontal side is the resolved liquid pressure.  Using
    connected-gas pressure here would suppress the horizontal water tongue
    that can rise while the original riser water falls.
    """

    old_speed = float(old_upward_speed)
    horizontal_pressure = float(horizontal_liquid_pressure_abs)
    vertical_pressure = float(vertical_liquid_pressure_abs)
    density = float(liquid_density)
    inertance_length = float(effective_inertance_length)
    dt = float(time_step)
    loss = float(upward_turn_loss_coefficient)
    if not _finite(
        old_speed,
        horizontal_pressure,
        vertical_pressure,
        density,
        inertance_length,
        dt,
        loss,
    ):
        raise ValueError("upward characteristic inputs must be finite")
    if (
        old_speed < 0.0
        or min(
            horizontal_pressure,
            vertical_pressure,
            density,
            inertance_length,
            dt,
        )
        <= 0.0
        or loss < 0.0
    ):
        raise ValueError(
            "upward speed and loss must be non-negative; pressures, density "
            "inertance length and time step must be positive"
        )

    drive = horizontal_pressure - vertical_pressure
    pressure_inertance = density * inertance_length / dt
    rhs = drive + pressure_inertance * old_speed
    quadratic = 0.5 * density * loss
    if rhs > 0.0 and quadratic > 0.0:
        accepted = 2.0 * rhs / (
            pressure_inertance
            + math.sqrt(
                pressure_inertance * pressure_inertance
                + 4.0 * quadratic * rhs
            )
        )
    elif rhs > 0.0:
        accepted = rhs / pressure_inertance
    else:
        accepted = 0.0
    unconstrained = (
        math.copysign(
            2.0 * abs(rhs)
            / (
                pressure_inertance
                + math.sqrt(
                    pressure_inertance * pressure_inertance
                    + 4.0 * quadratic * abs(rhs)
                )
            ),
            rhs,
        )
        if quadratic > 0.0 and rhs != 0.0
        else rhs / pressure_inertance
    )
    inertive_pressure = pressure_inertance * (accepted - old_speed)
    turn_pressure = 0.5 * density * loss * accepted * accepted
    raw_residual = inertive_pressure + turn_pressure - drive
    lower_reaction = max(raw_residual, 0.0) if rhs <= 0.0 else 0.0
    residual = math.fsum(
        (inertive_pressure, turn_pressure, -drive, -lower_reaction)
    )
    return UpwardIncomingCharacteristic(
        old_speed=float(old_speed),
        unconstrained_speed=float(unconstrained),
        accepted_speed=float(accepted),
        hydraulic_driving_pressure=float(drive),
        pressure_inertance=float(pressure_inertance),
        inertive_pressure=float(inertive_pressure),
        turn_loss_pressure=float(turn_pressure),
        lower_bound_reaction_pressure=float(lower_reaction),
        pressure_residual=float(residual),
    )


@dataclass(frozen=True)
class BottomMouthRiemannLedger:
    """Volume, area and convective-momentum audit for one gross-first trace."""

    incoming_upward_rate: float
    incoming_upward_speed: float
    first_cell_downward_rate: float
    first_cell_downward_speed: float
    physical_mouth_downward_speed: float
    outgoing_mouth_downward_rate: float
    accepted_upward_rate: float
    accepted_downward_rate: float
    q_net: float
    node_upward_capacity: float
    riser_downward_capacity: float
    aperture_upward_capacity: float
    positive_net_receiving_capacity: float
    negative_net_receiving_capacity: float
    wallis_downward_reference: float
    wallis_constraint_applied: bool
    wallis_excess_rate: float
    active_constraints: tuple[str, ...]
    incoming_convective_momentum_flux: float
    accepted_convective_momentum_flux: float
    upward_constraint_reaction_flux: float
    downward_constraint_reaction_flux: float
    downward_geometric_reaction_flux: float
    downward_physical_reaction_flux: float
    downward_pressure_acceleration_flux: float
    downward_capacity_reaction_flux: float
    downward_boundary_reaction_flux: float
    momentum_residual: float
    net_flux_residual: float
    mouth_area_residual: float
    node_volume_before: float
    node_volume_after: float
    riser_volume_before: float
    riser_volume_after: float
    upward_gross_volume: float
    downward_gross_volume: float
    combined_volume_residual: float


@dataclass(frozen=True)
class BottomMouthRiemannResult:
    """Directional face state ready for ``VerticalTwoStreamBoundaries``."""

    flux: DirectionalBoundaryFlux
    upward_area: float
    downward_area: float
    unused_mouth_area: float
    ledger: BottomMouthRiemannLedger

    @property
    def q_net(self) -> float:
        return self.flux.net_rate


def resolve_bottom_mouth_riemann(
    *,
    incoming_upward_characteristic_rate: float,
    incoming_upward_characteristic_speed: float,
    liquid_area_capacity: float,
    first_cell_downward_area: float,
    first_cell_downward_discharge: float,
    resolved_downward_mouth_area: float,
    physical_downward_mouth_speed: float | None = None,
    downward_physical_reaction_flux: float = 0.0,
    downward_pressure_acceleration_flux: float = 0.0,
    finite_node_liquid_volume: float,
    riser_downward_donor_volume: float,
    time_step: float,
    positive_net_receiving_capacity: float = math.inf,
    negative_net_receiving_capacity: float = math.inf,
    wallis_downward_capacity: float = math.inf,
    enforce_wallis_constraint: bool = False,
    dry_area_tolerance: float = 1.0e-14,
) -> BottomMouthRiemannResult:
    """Resolve independent incoming/upgoing and outgoing/falling traces.

    Rates and speeds are non-negative magnitudes except
    ``first_cell_downward_discharge``, which uses the upward coordinate and is
    therefore non-positive.  The first-cell area and the resolved mouth area
    are deliberately separate: their ratio must never alter the outgoing cell
    velocity.

    Capacity projection is one-way.  Same-step falling water is not reused to
    fund the incoming stream.  Positive net receiving capacity can reduce only
    the incoming rate; negative net receiving capacity can reduce only the
    outgoing rate.
    """

    upward_candidate = float(incoming_upward_characteristic_rate)
    upward_speed = float(incoming_upward_characteristic_speed)
    area_capacity = float(liquid_area_capacity)
    cell_down_area = float(first_cell_downward_area)
    cell_down_discharge = float(first_cell_downward_discharge)
    mouth_down_area = float(resolved_downward_mouth_area)
    node_volume = float(finite_node_liquid_volume)
    riser_volume = float(riser_downward_donor_volume)
    dt = float(time_step)
    positive_net_capacity = float(positive_net_receiving_capacity)
    negative_net_capacity = float(negative_net_receiving_capacity)
    wallis_reference = float(wallis_downward_capacity)
    physical_down_speed_input = (
        None
        if physical_downward_mouth_speed is None
        else float(physical_downward_mouth_speed)
    )
    physical_down_reaction = float(downward_physical_reaction_flux)
    pressure_acceleration_flux = float(downward_pressure_acceleration_flux)
    area_tolerance = float(dry_area_tolerance)

    finite_values = (
        upward_candidate,
        upward_speed,
        area_capacity,
        cell_down_area,
        cell_down_discharge,
        mouth_down_area,
        node_volume,
        riser_volume,
        dt,
        area_tolerance,
        physical_down_reaction,
        pressure_acceleration_flux,
    )
    if physical_down_speed_input is not None:
        finite_values += (physical_down_speed_input,)
    if not _finite(*finite_values):
        raise ValueError("bottom-mouth Riemann inputs must be finite")
    for name, value in (
        ("positive net receiving capacity", positive_net_capacity),
        ("negative net receiving capacity", negative_net_capacity),
        ("Wallis capacity", wallis_reference),
    ):
        if not _finite_or_positive_infinity(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative finite or infinity")
    if (
        upward_candidate < 0.0
        or upward_speed < 0.0
        or area_capacity < 0.0
        or cell_down_area < 0.0
        or mouth_down_area < 0.0
        or node_volume < 0.0
        or riser_volume < 0.0
        or dt <= 0.0
        or area_tolerance <= 0.0
        or physical_down_reaction < 0.0
        or pressure_acceleration_flux < 0.0
        or (
            physical_down_speed_input is not None
            and physical_down_speed_input < 0.0
        )
    ):
        raise ValueError(
            "rates, areas, speeds, inventories and tolerances must be "
            "non-negative, with positive time step and tolerance"
        )
    if cell_down_discharge > 0.0:
        raise ValueError("first-cell downward discharge must use the upward sign")
    if upward_candidate > 0.0 and upward_speed <= 0.0:
        raise ValueError("positive incoming upward rate requires a positive speed")
    if cell_down_area <= area_tolerance and cell_down_discharge != 0.0:
        raise ValueError("a dry first-cell downward stream cannot carry discharge")
    geometric_tolerance = max(
        area_tolerance,
        2048.0 * math.ulp(max(area_capacity, mouth_down_area, 1.0)),
    )
    if mouth_down_area > area_capacity + geometric_tolerance:
        raise BottomMouthRiemannError(
            "resolved downward mouth corridor exceeds the liquid aperture"
        )

    first_cell_down_rate = max(-cell_down_discharge, 0.0)
    downward_speed = (
        first_cell_down_rate / cell_down_area
        if cell_down_area > area_tolerance and first_cell_down_rate > 0.0
        else 0.0
    )
    physical_downward_speed = (
        downward_speed
        if physical_down_speed_input is None
        else physical_down_speed_input
    )
    outgoing_raw_mouth_rate = downward_speed * mouth_down_area
    outgoing_down_candidate = physical_downward_speed * mouth_down_area
    raw_mouth_momentum = outgoing_raw_mouth_rate * downward_speed
    physical_mouth_momentum = (
        outgoing_down_candidate * physical_downward_speed
    )
    required_acceleration_flux = max(
        physical_mouth_momentum - raw_mouth_momentum,
        0.0,
    )
    momentum_source_tolerance = max(
        1.0e-15,
        2048.0
        * math.ulp(
            max(
                pressure_acceleration_flux,
                required_acceleration_flux,
                1.0,
            )
        ),
    )
    if (
        physical_downward_speed > downward_speed
        and required_acceleration_flux
        > pressure_acceleration_flux + momentum_source_tolerance
    ):
        raise BottomMouthRiemannError(
            "pressure-accelerated downward mouth trace lacks its momentum "
            f"source: required={required_acceleration_flux:.12e}, "
            f"provided={pressure_acceleration_flux:.12e}"
        )

    node_up_capacity = node_volume / dt
    riser_down_capacity = riser_volume / dt
    downward_caps = {
        "outgoing_characteristic": outgoing_down_candidate,
        "riser_downward_donor": riser_down_capacity,
    }
    if enforce_wallis_constraint:
        downward_caps["wallis"] = wallis_reference
    accepted_down = max(min(downward_caps.values()), 0.0)
    # The Riemann solve has already established the two geometric trace
    # corridors.  Inventory and receiving constraints act on their velocities,
    # not by making the corridors appear and disappear.  Reconstructing area as
    # Q/u retained the unconstrained velocity and made a capacity-limited stream
    # jump to a different area on the next step.
    accepted_down_area = mouth_down_area

    remaining_area = max(area_capacity - accepted_down_area, 0.0)
    aperture_up_capacity = (
        upward_speed * remaining_area if upward_speed > 0.0 else 0.0
    )
    upward_caps = {
        "incoming_characteristic": upward_candidate,
        "finite_node_donor": node_up_capacity,
        "shared_aperture": aperture_up_capacity,
    }
    accepted_up = max(min(upward_caps.values()), 0.0)

    if accepted_up - accepted_down > positive_net_capacity:
        accepted_up = max(accepted_down + positive_net_capacity, 0.0)
        upward_caps["positive_net_receiver"] = accepted_up
    if accepted_down - accepted_up > negative_net_capacity:
        accepted_down = max(accepted_up + negative_net_capacity, 0.0)
        downward_caps["negative_net_receiver"] = accepted_down
    accepted_up_area = remaining_area
    unused_area = area_capacity - accepted_up_area - accepted_down_area
    if unused_area < -geometric_tolerance:
        raise BottomMouthRiemannError(
            "gross-first projection over-packed the local liquid aperture"
        )
    unused_area = max(unused_area, 0.0)

    accepted_upward_speed = (
        accepted_up / accepted_up_area
        if accepted_up_area > area_tolerance and accepted_up > 0.0
        else 0.0
    )
    accepted_downward_speed = (
        accepted_down / accepted_down_area
        if accepted_down_area > area_tolerance and accepted_down > 0.0
        else 0.0
    )
    flux = DirectionalBoundaryFlux(
        upward_rate=float(accepted_up),
        upward_speed=float(accepted_upward_speed),
        downward_rate=float(accepted_down),
        downward_speed=float(accepted_downward_speed),
    )
    q_net = flux.net_rate
    flux_scale = max(
        upward_candidate,
        outgoing_down_candidate,
        accepted_up,
        accepted_down,
        1.0e-12,
    )
    flux_tolerance = max(
        1.0e-16,
        1.0e-10 * flux_scale,
        2048.0 * math.ulp(flux_scale),
    )
    active_constraints = tuple(
        name
        for name, capacity in tuple(upward_caps.items()) + tuple(downward_caps.items())
        if math.isfinite(capacity)
        and (
            abs(capacity - accepted_up) <= flux_tolerance
            if name
            in {
                "incoming_characteristic",
                "finite_node_donor",
                "shared_aperture",
                "positive_net_receiver",
            }
            else abs(capacity - accepted_down) <= flux_tolerance
        )
    )

    incoming_momentum = math.fsum(
        (
            upward_candidate * upward_speed,
            first_cell_down_rate * downward_speed,
        )
    )
    accepted_momentum = math.fsum(
        (flux.upward_momentum_flux, flux.downward_momentum_flux)
    )
    upward_reaction = max(
        upward_candidate * upward_speed
        - accepted_up * accepted_upward_speed,
        0.0,
    )
    downward_reaction = max(
        first_cell_down_rate * downward_speed
        + pressure_acceleration_flux
        - accepted_down * accepted_downward_speed,
        0.0,
    )
    geometric_reaction = max(
        first_cell_down_rate - outgoing_raw_mouth_rate,
        0.0,
    ) * downward_speed
    capacity_reaction = max(
        outgoing_down_candidate * physical_downward_speed
        - accepted_down * accepted_downward_speed,
        0.0,
    )
    boundary_reaction = math.fsum(
        (geometric_reaction, physical_down_reaction, capacity_reaction)
    )
    momentum_residual = math.fsum(
        (
            incoming_momentum,
            pressure_acceleration_flux,
            -accepted_momentum,
            -upward_reaction,
            -downward_reaction,
        )
    )

    upward_volume = accepted_up * dt
    downward_volume = accepted_down * dt
    node_after = node_volume - upward_volume + downward_volume
    riser_after = riser_volume + upward_volume - downward_volume
    volume_tolerance = max(
        1.0e-18,
        2048.0
        * math.ulp(
            max(
                node_volume,
                riser_volume,
                upward_volume,
                downward_volume,
                1.0e-18,
            )
        ),
    )
    if node_after < -volume_tolerance or riser_after < -volume_tolerance:
        raise BottomMouthRiemannError("gross-first trace exhausted a donor inventory")
    node_after = max(node_after, 0.0)
    riser_after = max(riser_after, 0.0)
    combined_volume_residual = math.fsum(
        (node_after, riser_after, -node_volume, -riser_volume)
    )
    area_residual = math.fsum(
        (accepted_up_area, accepted_down_area, unused_area, -area_capacity)
    )
    wallis_excess = (
        max(outgoing_down_candidate - wallis_reference, 0.0)
        if math.isfinite(wallis_reference)
        else 0.0
    )

    ledger = BottomMouthRiemannLedger(
        incoming_upward_rate=float(upward_candidate),
        incoming_upward_speed=float(upward_speed),
        first_cell_downward_rate=float(first_cell_down_rate),
        first_cell_downward_speed=float(downward_speed),
        physical_mouth_downward_speed=float(physical_downward_speed),
        outgoing_mouth_downward_rate=float(outgoing_down_candidate),
        accepted_upward_rate=float(accepted_up),
        accepted_downward_rate=float(accepted_down),
        q_net=float(q_net),
        node_upward_capacity=float(node_up_capacity),
        riser_downward_capacity=float(riser_down_capacity),
        aperture_upward_capacity=float(aperture_up_capacity),
        positive_net_receiving_capacity=float(positive_net_capacity),
        negative_net_receiving_capacity=float(negative_net_capacity),
        wallis_downward_reference=float(wallis_reference),
        wallis_constraint_applied=bool(enforce_wallis_constraint),
        wallis_excess_rate=float(wallis_excess),
        active_constraints=active_constraints,
        incoming_convective_momentum_flux=float(incoming_momentum),
        accepted_convective_momentum_flux=float(accepted_momentum),
        upward_constraint_reaction_flux=float(upward_reaction),
        downward_constraint_reaction_flux=float(downward_reaction),
        downward_geometric_reaction_flux=float(geometric_reaction),
        downward_physical_reaction_flux=float(physical_down_reaction),
        downward_pressure_acceleration_flux=float(pressure_acceleration_flux),
        downward_capacity_reaction_flux=float(capacity_reaction),
        downward_boundary_reaction_flux=float(boundary_reaction),
        momentum_residual=float(momentum_residual),
        net_flux_residual=float(math.fsum((accepted_up, -accepted_down, -q_net))),
        mouth_area_residual=float(area_residual),
        node_volume_before=float(node_volume),
        node_volume_after=float(node_after),
        riser_volume_before=float(riser_volume),
        riser_volume_after=float(riser_after),
        upward_gross_volume=float(upward_volume),
        downward_gross_volume=float(downward_volume),
        combined_volume_residual=float(combined_volume_residual),
    )
    return BottomMouthRiemannResult(
        flux=flux,
        upward_area=float(accepted_up_area),
        downward_area=float(accepted_down_area),
        unused_mouth_area=float(unused_area),
        ledger=ledger,
    )


__all__ = [
    "BottomMouthRiemannError",
    "BottomMouthRiemannLedger",
    "BottomMouthRiemannResult",
    "CoupledGrossMouthCharacteristic",
    "UpwardIncomingCharacteristic",
    "resolve_bottom_mouth_riemann",
    "solve_coupled_gross_mouth_characteristics",
    "solve_upward_incoming_characteristic",
]
