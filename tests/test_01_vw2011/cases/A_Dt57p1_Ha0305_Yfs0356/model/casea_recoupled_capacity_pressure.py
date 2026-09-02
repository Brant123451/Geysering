"""Coupled mouth-flux/capacity-pressure projection for Case A.

The older post-source projector treats the T-node transaction as immutable.
That split can be infeasible when the accepted bottom inflow over-packs a wet
column whose donor traces have no remaining pressure degree of freedom.  This
module keeps the bottom upward and downward *gross* rates in the same convex
solve as the column pressure impulse.

For wet-cell common velocity increments ``x`` and bottom-rate corrections
``du`` and ``dd`` the projection minimizes

``0.5 sum(rho*dz*A*x**2) + 0.5*Iu*du**2 + 0.5*Id*dd**2``

subject to the finite-volume liquid-capacity inequalities.  The upward rate
obeys ``0 <= Q_up <= Q_up,candidate`` because a non-negative capacity pressure
opposes filling.  The downward rate obeys
``0 <= Q_down <= V_down,donor/dt`` and may exceed the loss-reduced
no-capacity characteristic when bottom capacity pressure expels liquid.  It
cannot withdraw more than the frozen first-cell inventory.  ``Iu``
and ``Id`` are physical pressure-impulse inertances in kg/m4.  For example
``rho*L/A`` represents a finite liquid plug and ``rho*c*dt/A`` represents a
one-step outgoing characteristic.  No 2-D result or fitted target enters this
solve.

The kernel is side-effect free: it does not mutate an FV or T-node state.  The
production network solver calls it before committing one atomic horizontal /
vertical mouth transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from casea_capacity_pressure_projection import (
    CapacityPressureRecouplingRequired,
    _solve_unilateral_capacity_qp_dual,
)
from casea_vertical_twostream_fv import (
    DirectionalTopologyTransferResult,
    VerticalTwoStreamState,
    conservative_directional_topology_transfer,
)


Array = np.ndarray


def flux_inertance_from_plug(
    *, liquid_density: float, effective_length: float, flow_area: float
) -> float:
    """Return ``rho*L/A`` in kg/m4 for a finite liquid plug."""

    values = np.asarray((liquid_density, effective_length, flow_area), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("density, effective length, and flow area must be positive")
    return float(liquid_density * effective_length / flow_area)


def flux_inertance_from_characteristic(
    *,
    liquid_density: float,
    celerity: float,
    time_step: float,
    flow_area: float,
) -> float:
    """Return ``rho*c*dt/A`` for a one-step hydraulic characteristic."""

    values = np.asarray(
        (liquid_density, celerity, time_step, flow_area), dtype=float
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("density, celerity, time step, and flow area must be positive")
    return float(liquid_density * celerity * time_step / flow_area)


def _field(values: Iterable[float], *, name: str) -> Array:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional field")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _optional_field(
    values: Iterable[float] | float | None,
    *,
    shape: tuple[int, ...],
    name: str,
) -> Array:
    if values is None:
        return np.zeros(shape, dtype=float)
    result = np.asarray(values, dtype=float)
    if result.ndim == 0:
        result = np.full(shape, float(result), dtype=float)
    elif result.shape != shape:
        raise ValueError(f"{name} must be scalar or one value per cell")
    else:
        result = result.copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(values: Array) -> Array:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RecoupledCapacityPressureLedger:
    """Volume, column impulse, characteristic, and reaction audit.

    ``rejected_downward_volume`` is retained for archive compatibility and is
    signed: a negative value means capacity pressure accelerated bottom
    outflow above the no-capacity characteristic candidate.
    """

    initial_liquid_volume: float
    predicted_liquid_volume: float
    boundary_liquid_volume_change: float
    volume_balance_residual: float
    upward_gross_volume: float
    downward_gross_volume: float
    rejected_upward_volume: float
    rejected_downward_volume: float
    cell_momentum_impulse_upward: Array
    column_momentum_impulse_upward: float
    column_bottom_pressure_impulse_on_liquid: float
    column_top_pressure_impulse_on_liquid: float
    column_internal_area_pressure_impulse_on_liquid: float
    column_pressure_decomposition_residual: float
    upward_characteristic_pressure_impulse: float
    downward_characteristic_pressure_impulse: float
    characteristic_momentum_impulse_upward: float
    bottom_capacity_pressure_impulse: float
    bottom_capacity_pressure_impulse_on_column: float
    bottom_reaction_impulse_on_tnode: float
    candidate_convective_momentum_flux_upward: float
    accepted_convective_momentum_flux_upward: float
    convective_momentum_flux_change_upward: float


@dataclass(frozen=True)
class RecoupledCapacityPressureResult:
    """Admissible column state and bottom transaction from one coupled QP.

    ``rejected_bottom_downward_rate`` is signed for archive compatibility; a
    negative value denotes pressure-accelerated downward discharge.
    """

    corrected_upward_discharge: Array
    corrected_downward_discharge: Array
    common_velocity_increment: Array
    candidate_bottom_upward_rate: float
    candidate_bottom_downward_rate: float
    final_bottom_upward_rate: float
    final_bottom_downward_rate: float
    final_bottom_net_rate: float
    rejected_bottom_upward_rate: float
    rejected_bottom_downward_rate: float
    net_face_discharge: Array
    predicted_liquid_area: Array
    column_face_pressure_impulse: Array
    capacity_pressure_impulse: Array
    active_capacity_mask: Array
    upward_upper_bound_multiplier: float
    upward_lower_bound_multiplier: float
    downward_upper_bound_multiplier: float
    downward_lower_bound_multiplier: float
    iterations: int
    active_set_releases: int
    maximum_packing_residual: float
    maximum_bound_residual: float
    maximum_downward_donor_residual: float
    maximum_directional_sign_residual: float
    maximum_kkt_stationarity_residual: float
    maximum_complementarity_residual: float
    ledger: RecoupledCapacityPressureLedger


@dataclass(frozen=True)
class RecoupledCapacityTopologyResult:
    """Sign-admissible state after pressure/topology active-set closure."""

    donor_state: VerticalTwoStreamState
    state: VerticalTwoStreamState
    projection: RecoupledCapacityPressureResult
    topology_transfers: tuple[DirectionalTopologyTransferResult, ...]
    outer_iterations: int


def project_mouth_and_capacity_pressure(
    *,
    upward_area: Iterable[float],
    upward_discharge: Iterable[float],
    downward_area: Iterable[float],
    downward_discharge: Iterable[float],
    candidate_bottom_upward_rate: float,
    candidate_bottom_downward_rate: float,
    bottom_upward_flux_inertance: float,
    bottom_downward_flux_inertance: float,
    bottom_upward_characteristic_area: float,
    bottom_downward_characteristic_area: float,
    top_downward_rate: float,
    liquid_capacity_area: Iterable[float],
    current_liquid_area: Iterable[float],
    dt: float,
    dz: float,
    liquid_density: float,
    bottom_downward_donor_rate_capacity: float | None = None,
    capacity_area_rate: Iterable[float] | float | None = None,
    top_upward_rate: float | None = None,
    bottom_reaction_area: float | None = None,
    directional_area_tolerance: float | None = None,
    maximum_upward_speed: float | None = None,
    maximum_downward_speed: float | None = None,
    maximum_bottom_downward_speed: float | None = None,
    preserve_directional_signs: bool = True,
    maximum_iterations: int | None = None,
) -> RecoupledCapacityPressureResult:
    """Recouple bottom gross rates before accepting the FV transaction.

    Directional boundary arguments are non-negative magnitudes.  The upward
    rate is bounded between zero and its no-capacity candidate.  The downward
    rate is bounded between zero and the frozen first-cell inventory rate, so
    it can increase above its loss-reduced no-capacity candidate without
    recursively accelerating its own donor state.  Cell pressure preserves
    directional slip by applying one common velocity increment to both liquid
    corridors in each wet cell.

    ``bottom_*_flux_inertance`` has units kg/m4.  The characteristic areas are
    used for physical momentum bookkeeping.  The downward inventory bound is
    supplied explicitly as a rate; if omitted it is the first-cell falling
    volume divided by ``dt``.
    """

    au = _field(upward_area, name="upward_area")
    qu = _field(upward_discharge, name="upward_discharge")
    ad = _field(downward_area, name="downward_area")
    qd = _field(downward_discharge, name="downward_discharge")
    capacity = _field(liquid_capacity_area, name="liquid_capacity_area")
    current = _field(current_liquid_area, name="current_liquid_area")
    if any(field.shape != au.shape for field in (qu, ad, qd, capacity, current)):
        raise ValueError("all cell fields must have one common shape")
    n = au.size

    scalars = [
        candidate_bottom_upward_rate,
        candidate_bottom_downward_rate,
        bottom_upward_flux_inertance,
        bottom_downward_flux_inertance,
        bottom_upward_characteristic_area,
        bottom_downward_characteristic_area,
        top_downward_rate,
        dt,
        dz,
        liquid_density,
    ]
    if top_upward_rate is not None:
        scalars.append(top_upward_rate)
    if bottom_reaction_area is not None:
        scalars.append(bottom_reaction_area)
    if bottom_downward_donor_rate_capacity is not None:
        scalars.append(bottom_downward_donor_rate_capacity)
    if directional_area_tolerance is not None:
        scalars.append(directional_area_tolerance)
    if maximum_upward_speed is not None:
        scalars.append(maximum_upward_speed)
    if maximum_downward_speed is not None:
        scalars.append(maximum_downward_speed)
    if maximum_bottom_downward_speed is not None:
        scalars.append(maximum_bottom_downward_speed)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("boundary, inertia, grid, time, and density inputs must be finite")
    if min(dt, dz, liquid_density) <= 0.0:
        raise ValueError("dt, dz, and liquid_density must be positive")
    if min(bottom_upward_flux_inertance, bottom_downward_flux_inertance) <= 0.0:
        raise ValueError("bottom flux inertances must be positive")
    if min(
        candidate_bottom_upward_rate,
        candidate_bottom_downward_rate,
        bottom_upward_characteristic_area,
        bottom_downward_characteristic_area,
        top_downward_rate,
    ) < 0.0:
        raise ValueError("directional rates and characteristic areas cannot be negative")
    if top_upward_rate is not None and top_upward_rate < 0.0:
        raise ValueError("top_upward_rate cannot be negative")
    if directional_area_tolerance is not None and directional_area_tolerance < 0.0:
        raise ValueError("directional_area_tolerance cannot be negative")
    if (
        bottom_downward_donor_rate_capacity is not None
        and bottom_downward_donor_rate_capacity < 0.0
    ):
        raise ValueError("bottom_downward_donor_rate_capacity cannot be negative")
    if maximum_upward_speed is not None and maximum_upward_speed <= 0.0:
        raise ValueError("maximum_upward_speed must be positive")
    if maximum_downward_speed is not None and maximum_downward_speed <= 0.0:
        raise ValueError("maximum_downward_speed must be positive")
    if (
        maximum_bottom_downward_speed is not None
        and maximum_bottom_downward_speed <= 0.0
    ):
        raise ValueError("maximum_bottom_downward_speed must be positive")
    if candidate_bottom_upward_rate > 0.0 and bottom_upward_characteristic_area <= 0.0:
        raise ValueError("positive upward candidate requires a positive characteristic area")
    if candidate_bottom_downward_rate > 0.0 and bottom_downward_characteristic_area <= 0.0:
        raise ValueError("positive downward candidate requires a positive characteristic area")

    area_scale = max(
        float(np.max(np.abs(capacity), initial=0.0)),
        float(np.max(np.abs(current), initial=0.0)),
        bottom_upward_characteristic_area,
        bottom_downward_characteristic_area,
        1.0e-12,
    )
    area_tolerance = max(
        4096.0 * np.finfo(float).eps * area_scale,
        1.0e-15,
        0.0
        if directional_area_tolerance is None
        else float(directional_area_tolerance),
    )
    if np.any(au < -area_tolerance) or np.any(ad < -area_tolerance):
        raise ValueError("directional areas cannot be negative")
    au = np.maximum(au, 0.0)
    ad = np.maximum(ad, 0.0)
    reconstructed_current = au + ad
    if float(np.max(np.abs(reconstructed_current - current))) > area_tolerance:
        raise ValueError("current_liquid_area must equal upward_area+downward_area")
    current = reconstructed_current
    if np.any(capacity < current - area_tolerance) or np.any(capacity < -area_tolerance):
        raise ValueError("liquid capacity cannot be below the current liquid area")

    capacity_rate = _optional_field(
        capacity_area_rate, shape=au.shape, name="capacity_area_rate"
    )
    capacity_end = capacity + dt * capacity_rate
    if np.any(capacity_end < -area_tolerance):
        raise ValueError("capacity_area_rate produces negative end-of-step capacity")

    discharge_scale = max(
        float(np.max(np.abs(qu), initial=0.0)),
        float(np.max(np.abs(qd), initial=0.0)),
        candidate_bottom_upward_rate,
        candidate_bottom_downward_rate,
        top_downward_rate,
        0.0 if top_upward_rate is None else top_upward_rate,
        area_scale * dz / dt,
        1.0e-12,
    )
    rate_tolerance = max(
        8192.0 * np.finfo(float).eps * discharge_scale,
        1.0e-14 * discharge_scale,
        1.0e-15,
        0.0
        if directional_area_tolerance is None
        else float(directional_area_tolerance),
    )
    invalid_upward = (au <= area_tolerance) & (np.abs(qu) > rate_tolerance)
    invalid_downward = (ad <= area_tolerance) & (np.abs(qd) > rate_tolerance)
    if np.any(invalid_upward) or np.any(invalid_downward):
        upward = bool(np.any(invalid_upward))
        index = int(
            np.flatnonzero(invalid_upward if upward else invalid_downward)[0]
        )
        area = au[index] if upward else ad[index]
        discharge = qu[index] if upward else qd[index]
        label = "upward" if upward else "downward"
        raise ValueError(
            f"a dry {label} stream cannot carry discharge: cell={index}, "
            f"area={area:.12e}, discharge={discharge:.12e}, "
            f"area_tolerance={area_tolerance:.12e}, "
            f"rate_tolerance={rate_tolerance:.12e}"
        )

    wet_cells = np.flatnonzero(current > area_tolerance)
    column_for_cell = {int(cell): column for column, cell in enumerate(wet_cells)}
    upward_column = wet_cells.size
    downward_column = wet_cells.size + 1
    variable_count = wet_cells.size + 2

    # H maps the cell velocity impulses and both gross-rate corrections to the
    # exact donor traces used by the existing vertical FV discretisation.
    face_base = np.zeros(n + 1)
    face_map = np.zeros((n + 1, variable_count))
    face_base[0] = candidate_bottom_upward_rate - candidate_bottom_downward_rate
    face_map[0, upward_column] = 1.0
    face_map[0, downward_column] = -1.0
    for face in range(1, n):
        lower_cell = face - 1
        upper_cell = face
        face_base[face] = qu[lower_cell] + qd[upper_cell]
        if lower_cell in column_for_cell:
            face_map[face, column_for_cell[lower_cell]] += au[lower_cell]
        if upper_cell in column_for_cell:
            face_map[face, column_for_cell[upper_cell]] += ad[upper_cell]
    if top_upward_rate is None:
        face_base[n] = qu[-1] - top_downward_rate
        if n - 1 in column_for_cell:
            face_map[n, column_for_cell[n - 1]] += au[-1]
    else:
        face_base[n] = top_upward_rate - top_downward_rate

    divergence_base = face_base[:-1] - face_base[1:]
    capacity_matrix = face_map[:-1] - face_map[1:]
    available_volume_rate = (capacity - current) * dz / dt + dz * capacity_rate
    capacity_rhs = available_volume_rate - divergence_base

    # The upward correction is bounded by -candidate <= du <= 0.  Downward
    # capacity pressure is different: it can recover outflow above the
    # loss-reduced no-capacity characteristic candidate.  Its upper bound is
    # the resolved first-cell donor trace, not dd <= 0.  Keeping dd <= 0 here
    # makes a saturated bottom cell artificially infeasible: pressure may
    # neither accept more upward liquid nor expel the liquid already arriving
    # from the falling corridor.
    rows = [capacity_matrix]
    rhs = [capacity_rhs]
    bound_matrix = np.zeros((4, variable_count))
    bound_rhs = np.zeros(4)
    bound_matrix[0, upward_column] = 1.0
    bound_rhs[0] = 0.0
    bound_matrix[1, upward_column] = -1.0
    bound_rhs[1] = candidate_bottom_upward_rate
    bound_matrix[3, downward_column] = -1.0
    bound_rhs[3] = candidate_bottom_downward_rate

    # The bottom falling trace is an outflow supplied by the finite liquid
    # inventory of cell 0.  Its donor inequality is
    #
    #   Qdown,candidate + dd
    #       <= Vdown,donor / dt.
    #
    # A flux-based bound containing x[0] made this constraint drive x[0]
    # downward, enlarge its own bound, and create a positive-feedback discharge
    # spike.  The pressure-corrected cell momentum is committed by the FV
    # update; the same transaction is limited only by its frozen inventory.
    downward_donor_row_present = False
    downward_donor_follows_pressure = False
    downward_donor_capacity = 0.0
    if bottom_downward_characteristic_area > area_tolerance:
        if ad[0] <= area_tolerance:
            if candidate_bottom_downward_rate > rate_tolerance:
                raise CapacityPressureRecouplingRequired(
                    "positive bottom downflow has no first-cell donor corridor"
                )
        else:
            if bottom_downward_donor_rate_capacity is None:
                downward_donor_capacity = (
                    max(-qd[0], 0.0)
                    * bottom_downward_characteristic_area
                    / ad[0]
                )
                bound_matrix[2, column_for_cell[0]] = (
                    bottom_downward_characteristic_area
                )
                downward_donor_follows_pressure = True
            else:
                downward_donor_capacity = min(
                    ad[0] * dz / dt,
                    float(bottom_downward_donor_rate_capacity),
                )
            if maximum_downward_speed is not None:
                downward_donor_capacity = min(
                    downward_donor_capacity,
                    bottom_downward_characteristic_area
                    * maximum_downward_speed,
                )
            bound_matrix[2, downward_column] = 1.0
            bound_rhs[2] = (
                downward_donor_capacity - candidate_bottom_downward_rate
            )
            downward_donor_row_present = True
    if not downward_donor_row_present:
        # With no bottom falling corridor the downward candidate is zero and
        # dd must remain zero.  The paired rows keep the four multiplier slots
        # stable for diagnostics.
        bound_matrix[2, downward_column] = 1.0
        bound_rhs[2] = 0.0
        bound_rhs[3] = 0.0
    rows.append(bound_matrix)
    rhs.append(bound_rhs)
    bottom_downward_rate_ceiling = math.inf
    if (
        maximum_bottom_downward_speed is not None
        and bottom_downward_characteristic_area > area_tolerance
    ):
        bottom_downward_rate_ceiling = (
            bottom_downward_characteristic_area
            * maximum_bottom_downward_speed
        )
        mouth_speed_row = np.zeros(variable_count)
        mouth_speed_row[downward_column] = 1.0
        rows.append(mouth_speed_row[None, :])
        rhs.append(
            np.asarray(
                [
                    bottom_downward_rate_ceiling
                    - candidate_bottom_downward_rate
                ],
                dtype=float,
            )
        )

    # The pressure step may stop either labelled corridor but must not move it
    # through zero before the conservative topology owner has transported the
    # corresponding area.  Without these inequalities a common pressure
    # impulse can reverse a labelled stream, after which an immediate relabel
    # changes the donor-face map that the same capacity solve just audited.
    # A stopped branch can be transferred by the ordinary topology stage on a
    # later source/transport update; no velocity clipping is introduced here.
    if preserve_directional_signs:
        directional_rows: list[Array] = []
        directional_rhs: list[float] = []
        for cell in wet_cells:
            column = column_for_cell[int(cell)]
            if au[cell] > area_tolerance:
                row = np.zeros(variable_count)
                row[column] = -au[cell]
                directional_rows.append(row)
                directional_rhs.append(float(qu[cell]))
            if ad[cell] > area_tolerance:
                row = np.zeros(variable_count)
                row[column] = ad[cell]
                directional_rows.append(row)
                directional_rhs.append(float(-qd[cell]))
        if directional_rows:
            rows.append(np.vstack(directional_rows))
            rhs.append(np.asarray(directional_rhs, dtype=float))
    speed_rows: list[Array] = []
    speed_rhs: list[float] = []
    for cell in wet_cells:
        column = column_for_cell[int(cell)]
        if maximum_upward_speed is not None and au[cell] > area_tolerance:
            row = np.zeros(variable_count)
            row[column] = au[cell]
            speed_rows.append(row)
            speed_rhs.append(
                float(au[cell] * maximum_upward_speed - qu[cell])
            )
        if maximum_downward_speed is not None and ad[cell] > area_tolerance:
            row = np.zeros(variable_count)
            row[column] = -ad[cell]
            speed_rows.append(row)
            speed_rhs.append(
                float(ad[cell] * maximum_downward_speed + qd[cell])
            )
    if speed_rows:
        rows.append(np.vstack(speed_rows))
        rhs.append(np.asarray(speed_rhs, dtype=float))
    matrix = np.vstack(rows)
    right_hand_side = np.concatenate(rhs)

    mass = np.concatenate(
        (
            liquid_density * dz * current[wet_cells],
            np.asarray(
                (bottom_upward_flux_inertance, bottom_downward_flux_inertance),
                dtype=float,
            ),
        )
    )
    maximum_sweeps = (
        max(512, 64 * (n + variable_count + 4))
        if maximum_iterations is None
        else int(maximum_iterations)
    )
    if maximum_sweeps <= 0:
        raise ValueError("maximum_iterations must be positive")

    (
        increment,
        multiplier,
        active,
        iterations,
        releases,
        stationarity,
        complementarity,
    ) = _solve_unilateral_capacity_qp_dual(
        mass=mass,
        matrix=matrix,
        right_hand_side=right_hand_side,
        rate_tolerance=rate_tolerance,
        velocity_scale=1.0,
        maximum_sweeps=maximum_sweeps,
    )

    x = np.zeros(n)
    x[wet_cells] = increment[: wet_cells.size]
    du = float(increment[upward_column])
    dd = float(increment[downward_column])
    final_up = candidate_bottom_upward_rate + du
    final_down = candidate_bottom_downward_rate + dd
    corrected_qu = qu + au * x
    corrected_qd = qd + ad * x
    corrected_downward_donor_capacity = (
        max(-corrected_qd[0], 0.0)
        * bottom_downward_characteristic_area
        / ad[0]
        if downward_donor_follows_pressure
        else downward_donor_capacity
    )
    downward_donor_residual = (
        max(final_down - corrected_downward_donor_capacity, 0.0)
        if downward_donor_row_present
        else max(final_down, 0.0)
        if ad[0] <= area_tolerance
        else 0.0
    )
    bound_residual = max(
        -final_up,
        final_up - candidate_bottom_upward_rate,
        -final_down,
        downward_donor_residual,
        final_down - bottom_downward_rate_ceiling,
        0.0,
    )
    if bound_residual > 8.0 * rate_tolerance:
        raise CapacityPressureRecouplingRequired(
            "recoupled mouth projection failed its directional gross-rate audit"
        )
    if downward_donor_residual > 8.0 * rate_tolerance:
        raise CapacityPressureRecouplingRequired(
            "recoupled bottom downflow exceeds its corrected first-cell donor"
        )
    directional_residual = 0.0
    if preserve_directional_signs:
        directional_residual = max(
            float(np.max(-corrected_qu, initial=0.0)),
            float(np.max(corrected_qd, initial=0.0)),
            0.0,
        )
        if directional_residual > 8.0 * rate_tolerance:
            raise CapacityPressureRecouplingRequired(
                "recoupled mouth projection crossed a directional trace"
            )
        corrected_qu[np.abs(corrected_qu) <= rate_tolerance] = 0.0
        corrected_qd[np.abs(corrected_qd) <= rate_tolerance] = 0.0
    if abs(final_up) <= rate_tolerance:
        final_up = 0.0
    if abs(final_down) <= rate_tolerance:
        final_down = 0.0
    net_face = face_base + face_map @ increment
    divergence = net_face[:-1] - net_face[1:]
    predicted_area = current + dt / dz * divergence
    packing_residual = predicted_area - capacity_end

    pressure_impulse = np.zeros(n + 1)
    cell = 0
    while cell < n:
        if current[cell] <= area_tolerance:
            cell += 1
            continue
        start = cell
        while cell + 1 < n and current[cell + 1] > area_tolerance:
            cell += 1
        end = cell + 1
        for index in range(end - 1, start - 1, -1):
            pressure_impulse[index] = (
                pressure_impulse[index + 1] + liquid_density * dz * x[index]
            )
        cell = end

    cell_momentum = liquid_density * dz * current * x
    column_momentum = float(np.sum(cell_momentum))
    column_bottom_pressure = float(current[0] * pressure_impulse[0])
    column_top_pressure = float(-current[-1] * pressure_impulse[-1])
    column_internal_pressure = float(
        np.sum((current[1:] - current[:-1]) * pressure_impulse[1:-1])
    )


    pressure_residual = float(
        column_momentum
        - column_bottom_pressure
        - column_top_pressure
        - column_internal_pressure
    )

    capacity_multiplier = multiplier[:n]
    bound_multiplier = multiplier[n:]
    bottom_capacity_pressure = float(capacity_multiplier[0])
    reaction_area = current[0] if bottom_reaction_area is None else bottom_reaction_area
    if reaction_area < 0.0:
        raise ValueError("bottom_reaction_area cannot be negative")
    bottom_pressure_on_column = float(reaction_area * bottom_capacity_pressure)
    bottom_reaction_on_node = -bottom_pressure_on_column

    upward_characteristic_pressure = float(bottom_upward_flux_inertance * du)
    downward_characteristic_pressure = float(bottom_downward_flux_inertance * dd)
    characteristic_momentum = float(
        bottom_upward_characteristic_area * upward_characteristic_pressure
        - bottom_downward_characteristic_area * downward_characteristic_pressure
    )

    def signed_convective_momentum(up_rate: float, down_rate: float) -> float:
        up_term = (
            up_rate * up_rate / bottom_upward_characteristic_area
            if bottom_upward_characteristic_area > area_tolerance
            else 0.0
        )
        down_term = (
            down_rate * down_rate / bottom_downward_characteristic_area
            if bottom_downward_characteristic_area > area_tolerance
            else 0.0
        )
        return float(liquid_density * (up_term - down_term))

    candidate_momentum_flux = signed_convective_momentum(
        candidate_bottom_upward_rate, candidate_bottom_downward_rate
    )
    accepted_momentum_flux = signed_convective_momentum(final_up, final_down)

    initial_volume = float(dz * np.sum(current))
    predicted_volume = float(dz * np.sum(predicted_area))
    boundary_volume_change = float(dt * (net_face[0] - net_face[-1]))
    volume_residual = float(predicted_volume - initial_volume - boundary_volume_change)

    ledger = RecoupledCapacityPressureLedger(
        initial_liquid_volume=initial_volume,
        predicted_liquid_volume=predicted_volume,
        boundary_liquid_volume_change=boundary_volume_change,
        volume_balance_residual=volume_residual,
        upward_gross_volume=float(dt * final_up),
        downward_gross_volume=float(dt * final_down),
        rejected_upward_volume=float(dt * (candidate_bottom_upward_rate - final_up)),
        rejected_downward_volume=float(dt * (candidate_bottom_downward_rate - final_down)),
        cell_momentum_impulse_upward=_readonly(cell_momentum),
        column_momentum_impulse_upward=column_momentum,
        column_bottom_pressure_impulse_on_liquid=column_bottom_pressure,
        column_top_pressure_impulse_on_liquid=column_top_pressure,
        column_internal_area_pressure_impulse_on_liquid=column_internal_pressure,
        column_pressure_decomposition_residual=pressure_residual,
        upward_characteristic_pressure_impulse=upward_characteristic_pressure,
        downward_characteristic_pressure_impulse=downward_characteristic_pressure,
        characteristic_momentum_impulse_upward=characteristic_momentum,
        bottom_capacity_pressure_impulse=bottom_capacity_pressure,
        bottom_capacity_pressure_impulse_on_column=bottom_pressure_on_column,
        bottom_reaction_impulse_on_tnode=bottom_reaction_on_node,
        candidate_convective_momentum_flux_upward=candidate_momentum_flux,
        accepted_convective_momentum_flux_upward=accepted_momentum_flux,
        convective_momentum_flux_change_upward=float(
            accepted_momentum_flux - candidate_momentum_flux
        ),
    )
    return RecoupledCapacityPressureResult(
        corrected_upward_discharge=_readonly(corrected_qu),
        corrected_downward_discharge=_readonly(corrected_qd),
        common_velocity_increment=_readonly(x),
        candidate_bottom_upward_rate=float(candidate_bottom_upward_rate),
        candidate_bottom_downward_rate=float(candidate_bottom_downward_rate),
        final_bottom_upward_rate=float(final_up),
        final_bottom_downward_rate=float(final_down),
        final_bottom_net_rate=float(final_up - final_down),
        rejected_bottom_upward_rate=float(candidate_bottom_upward_rate - final_up),
        rejected_bottom_downward_rate=float(candidate_bottom_downward_rate - final_down),
        net_face_discharge=_readonly(net_face),
        predicted_liquid_area=_readonly(predicted_area),
        column_face_pressure_impulse=_readonly(pressure_impulse),
        capacity_pressure_impulse=_readonly(capacity_multiplier),
        active_capacity_mask=_readonly(active[:n]),
        upward_upper_bound_multiplier=float(bound_multiplier[0]),
        upward_lower_bound_multiplier=float(bound_multiplier[1]),
        downward_upper_bound_multiplier=float(bound_multiplier[2]),
        downward_lower_bound_multiplier=float(bound_multiplier[3]),
        iterations=int(iterations),
        active_set_releases=int(releases),
        maximum_packing_residual=float(np.max(packing_residual, initial=0.0)),
        maximum_bound_residual=float(bound_residual),
        maximum_downward_donor_residual=float(downward_donor_residual),
        maximum_directional_sign_residual=float(directional_residual),
        maximum_kkt_stationarity_residual=float(stationarity),
        maximum_complementarity_residual=float(complementarity),
        ledger=ledger,
    )


def project_state_mouth_and_capacity_pressure(
    state: VerticalTwoStreamState,
    *,
    preserve_stopped_partition: Iterable[bool] | None = None,
    maximum_topology_iterations: int = 32,
    **projection_arguments: object,
) -> RecoupledCapacityTopologyResult:
    """Close capacity pressure and directional topology before one FV commit.

    A sign-preserving pressure solve is attempted first.  If its active set is
    infeasible, one unrestricted pressure solve is used only to identify the
    conservative local topology change; the resulting remapped state is then
    submitted again to the sign-preserving solve.  Thus the returned pressure
    transaction always uses the donor map of the returned directional state.
    """

    if maximum_topology_iterations <= 0:
        raise ValueError("maximum_topology_iterations must be positive")
    forbidden = {
        "upward_area",
        "upward_discharge",
        "downward_area",
        "downward_discharge",
        "current_liquid_area",
        "preserve_directional_signs",
    }
    overlap = forbidden.intersection(projection_arguments)
    if overlap:
        raise ValueError(
            "state-owned projection arguments were supplied twice: "
            + ", ".join(sorted(overlap))
        )
    preserve = (
        (False,) * state.cell_count
        if preserve_stopped_partition is None
        else tuple(bool(value) for value in preserve_stopped_partition)
    )
    if len(preserve) != state.cell_count:
        raise ValueError("stopped-partition mask must contain one value per cell")
    topology_area_tolerance = float(
        projection_arguments.get("directional_area_tolerance", 0.0) or 0.0
    )

    work = state
    transfers: list[DirectionalTopologyTransferResult] = []
    for outer in range(1, maximum_topology_iterations + 1):
        common = dict(
            upward_area=work.upward_area,
            upward_discharge=work.upward_discharge,
            downward_area=work.downward_area,
            downward_discharge=work.downward_discharge,
            current_liquid_area=work.liquid_area,
            **projection_arguments,
        )
        try:
            projection = project_mouth_and_capacity_pressure(
                **common,
                preserve_directional_signs=True,
            )
        except CapacityPressureRecouplingRequired:
            unrestricted = project_mouth_and_capacity_pressure(
                **common,
                preserve_directional_signs=False,
            )
            transfer = conservative_directional_topology_transfer(
                upward_area=work.upward_area,
                upward_discharge=unrestricted.corrected_upward_discharge,
                downward_area=work.downward_area,
                downward_discharge=unrestricted.corrected_downward_discharge,
                preserve_stopped_partition=preserve,
                area_tolerance=topology_area_tolerance,
            )
            transfers.append(transfer)
            work = transfer.state
            continue

        final_state = VerticalTwoStreamState.from_iterables(
            upward_area=work.upward_area,
            upward_discharge=projection.corrected_upward_discharge,
            downward_area=work.downward_area,
            downward_discharge=projection.corrected_downward_discharge,
        )
        return RecoupledCapacityTopologyResult(
            donor_state=work,
            state=final_state,
            projection=projection,
            topology_transfers=tuple(transfers),
            outer_iterations=outer,
        )
    raise CapacityPressureRecouplingRequired(
        "capacity pressure/topology active set did not reach a sign-admissible state"
    )


__all__ = (
    "RecoupledCapacityPressureLedger",
    "RecoupledCapacityPressureResult",
    "RecoupledCapacityTopologyResult",
    "flux_inertance_from_characteristic",
    "flux_inertance_from_plug",
    "project_mouth_and_capacity_pressure",
    "project_state_mouth_and_capacity_pressure",
)
