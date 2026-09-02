"""Active-set capacity-pressure projection for the Case-A riser.

This module is deliberately independent of the vertical FV host.  It projects
the *actual directional donor traces* used by that host, rather than resetting
cell-total discharge to a plotted or face-averaged target.

The correction in a wet cell is one common velocity impulse ``x``::

    Q_up'   = Q_up   + A_up   x
    Q_down' = Q_down + A_down x

Consequently the up/down slip is unchanged.  Capacity constraints act on the
net face trace.  The accepted T-node rate remains a boundary-face flux; it is
not equated to the adjacent cell-average discharge.  An optional legacy bulk
match is kept only for isolated diagnostics because using it as an FV boundary
condition over-constrains a saturated wet block.

The returned face-pressure impulse is an incremental constraint pressure, not
an additional empirical source.  Its liquid impulse is decomposed exactly
into bottom, top, and internal area-change tractions so their opposite
reactions can be consumed by the T node, atmosphere, and gas/interface owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


class CapacityPressureProjectionError(RuntimeError):
    """Base class for a rejected capacity-pressure projection."""


class CapacityPressureRecouplingRequired(CapacityPressureProjectionError):
    """The accepted boundary transaction and preserved corridors are incompatible."""


@dataclass(frozen=True)
class CapacityPressureLedger:
    """Physical and kinematic impulse audit for one projection."""

    cell_kinematic_impulse: Array
    cell_physical_impulse: Array
    liquid_kinematic_impulse: float
    liquid_physical_impulse: float
    bottom_pressure_impulse_on_liquid: float
    top_pressure_impulse_on_liquid: float
    internal_area_pressure_impulse_on_liquid: float
    boundary_owner_reaction_impulse: float
    interface_owner_reaction_impulse: float
    pressure_decomposition_residual: float
    coupled_momentum_residual: float


@dataclass(frozen=True)
class CapacityPressureProjectionResult:
    """Corrected directional momenta and active-set diagnostics."""

    corrected_upward_discharge: Array
    corrected_downward_discharge: Array
    common_velocity_increment: Array
    net_face_discharge: Array
    predicted_liquid_area: Array
    face_pressure_impulse: Array
    active_capacity_mask: Array
    capacity_multiplier: Array
    lower_bound_multiplier: Array
    upper_bound_multiplier: Array
    iterations: int
    working_set_capacity_releases: int
    maximum_packing_residual: float
    maximum_active_constraint_residual: float
    maximum_kkt_stationarity_residual: float
    maximum_complementarity_residual: float
    bottom_bulk_anchor_residual: float
    top_bulk_anchor_residual: float
    ledger: CapacityPressureLedger

    @property
    def bottom_downward_anchor_residual(self) -> float:
        """Compatibility alias; the residual now audits the bottom bulk rate."""

        return self.bottom_bulk_anchor_residual

    @property
    def top_upward_anchor_residual(self) -> float:
        """Compatibility alias; the residual now audits the top bulk rate."""

        return self.top_bulk_anchor_residual


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


def _solve_minimum_mass_projection(
    *,
    mass: Array,
    rows: list[Array],
    right_hand_side: list[float],
    velocity_scale: float,
    feasibility_tolerances: list[float],
) -> tuple[Array, Array, float]:
    """Solve ``min 0.5*x'Mx`` subject to the supplied equalities."""

    variable_count = mass.size
    if not rows:
        return np.zeros(variable_count), np.zeros(0), 0.0
    matrix = np.vstack(rows)
    rhs = np.asarray(right_hand_side, dtype=float)
    row_scale = np.maximum(
        np.linalg.norm(matrix, axis=1),
        np.abs(rhs) / max(velocity_scale, np.finfo(float).tiny),
    )
    row_scale = np.maximum(row_scale, np.finfo(float).tiny)
    scaled_matrix = matrix / row_scale[:, None]
    scaled_rhs = rhs / row_scale
    # Solve the weighted equality problem directly.  Forming ``A M^-1 A'``
    # squares the condition number and falsely classified the long, thin
    # Taylor-film constraint chain as inconsistent at 7.53 s.
    sqrt_mass = np.sqrt(mass)
    weighted_matrix = scaled_matrix / sqrt_mass[None, :]
    weighted_increment = np.linalg.lstsq(
        weighted_matrix,
        scaled_rhs,
        rcond=1.0e-14,
    )[0]
    increment = weighted_increment / sqrt_mass
    scaled_multiplier = np.linalg.lstsq(
        scaled_matrix.T,
        -(mass * increment),
        rcond=1.0e-14,
    )[0]
    multiplier = scaled_multiplier / row_scale
    residual = matrix @ increment - rhs
    allowed = np.asarray(feasibility_tolerances, dtype=float)
    normalized_residual = float(
        np.max(np.abs(residual) / np.maximum(allowed, np.finfo(float).tiny))
    )
    if normalized_residual > 1.0:
        raise CapacityPressureRecouplingRequired(
            "capacity pressure equalities are incompatible with the accepted "
            "boundary transaction or preserved directional corridors; "
            f"max_abs_residual={float(np.max(np.abs(residual))):.12e}, "
            f"normalized_residual={normalized_residual:.6e}"
        )
    stationarity = mass * increment + matrix.T @ multiplier
    return increment, multiplier, float(np.max(np.abs(stationarity), initial=0.0))


def _solve_unilateral_capacity_qp_dual(
    *,
    mass: Array,
    matrix: Array,
    right_hand_side: Array,
    rate_tolerance: float,
    velocity_scale: float,
    maximum_sweeps: int,
) -> tuple[Array, Array, Array, int, int, float, float]:
    """Solve ``min 0.5*x'Mx`` subject to ``C*x <= r``.

    The non-negative dual is solved by a finite Lawson--Hanson-style active
    set.  Unlike a capacity-equality working set, this formulation permits any
    number of formerly active rows to become slack in one solve and ratio-
    pivots dependent normals without compromising primal packing.

    Returns the primal increment, original (unscaled) capacity multipliers,
    active mask, sweep count, dual releases, stationarity residual, and
    complementarity residual.
    """

    row_count, variable_count = matrix.shape
    increment = np.zeros(variable_count)
    multiplier = np.zeros(row_count)
    active = np.zeros(row_count, dtype=bool)
    if row_count == 0:
        return increment, multiplier, active, 1, 0, 0.0, 0.0
    if variable_count == 0:
        violation = -right_hand_side
        if float(np.max(violation, initial=0.0)) > rate_tolerance:
            cell = int(np.argmax(violation))
            raise CapacityPressureRecouplingRequired(
                "capacity QP is primal infeasible: no wet momentum degree of "
                f"freedom can satisfy row {cell}; rhs={right_hand_side[cell]:.12e}"
            )
        return increment, multiplier, active, 1, 0, 0.0, 0.0

    sqrt_mass = np.sqrt(mass)
    weighted_matrix = matrix / sqrt_mass[None, :]
    weighted_norm = np.linalg.norm(weighted_matrix, axis=1)
    norm_scale = max(float(np.max(weighted_norm, initial=0.0)), 1.0)
    zero_tolerance = 4096.0 * np.finfo(float).eps * norm_scale
    zero_rows = weighted_norm <= zero_tolerance
    infeasible_zero = zero_rows & (right_hand_side < -rate_tolerance)
    if np.any(infeasible_zero):
        cell = int(np.flatnonzero(infeasible_zero)[0])
        raise CapacityPressureRecouplingRequired(
            "capacity QP is primal infeasible: zero-mobility capacity row "
            f"{cell} requires {right_hand_side[cell]:.12e} < 0"
        )

    retained = np.flatnonzero(~zero_rows)
    if retained.size == 0:
        return increment, multiplier, active, 1, 0, 0.0, 0.0
    retained_matrix = weighted_matrix[retained]
    retained_rhs = right_hand_side[retained]
    row_scale = np.maximum(
        np.linalg.norm(retained_matrix, axis=1),
        np.abs(retained_rhs) / max(velocity_scale, np.finfo(float).tiny),
    )
    row_scale = np.maximum(row_scale, np.finfo(float).tiny)
    scaled_matrix = retained_matrix / row_scale[:, None]
    scaled_rhs = retained_rhs / row_scale
    dual = np.zeros(retained.size)
    scaled_increment = np.zeros(variable_count)
    scaled_slack = scaled_rhs.copy()
    scaled_rate_tolerance = np.maximum(
        rate_tolerance / row_scale,
        32768.0
        * np.finfo(float).eps
        * np.maximum(1.0, np.abs(scaled_rhs)),
    )
    dual_releases = 0
    converged = False
    passive: list[int] = []
    sweep = 0
    for sweep in range(1, maximum_sweeps + 1):
        passive_set = set(passive)
        violated = [
            index
            for index in range(retained.size)
            if index not in passive_set
            and scaled_slack[index] < -scaled_rate_tolerance[index]
        ]
        if not violated:
            converged = True
            break
        entering = min(
            violated,
            key=lambda index: (
                scaled_slack[index] / scaled_rate_tolerance[index],
                int(retained[index]),
            ),
        )

        # Keep the passive face-pressure rows linearly independent.  If the
        # entering normal is dependent, its negative slack supplies a zero-
        # curvature dual direction.  Move along it until a positive incumbent
        # multiplier reaches zero; if no incumbent can block the ray, the same
        # direction is a Farkas certificate of primal infeasibility.
        dependent = False
        coefficients = np.zeros(len(passive))
        if passive:
            passive_matrix = scaled_matrix[passive]
            coefficients = np.linalg.lstsq(
                passive_matrix.T,
                scaled_matrix[entering],
                rcond=1.0e-13,
            )[0]
            dependence_residual = float(
                np.linalg.norm(
                    scaled_matrix[entering]
                    - coefficients @ passive_matrix
                )
            )
            dependence_tolerance = max(
                32768.0
                * np.finfo(float).eps
                * max(
                    float(np.linalg.norm(scaled_matrix[entering])),
                    float(np.linalg.norm(coefficients @ passive_matrix)),
                    1.0,
                ),
                1.0e-12,
            )
            dependent = dependence_residual <= dependence_tolerance

        if dependent:
            positive_coefficients = [
                (dual[index] / coefficient, int(retained[index]), position)
                for position, (index, coefficient) in enumerate(
                    zip(passive, coefficients)
                )
                if coefficient > 1.0e-13
            ]
            dependency_gap = float(
                scaled_rhs[entering]
                - coefficients @ scaled_rhs[passive]
            )
            if not positive_coefficients:
                certificate = np.zeros(retained.size)
                certificate[entering] = 1.0
                certificate[passive] = -coefficients
                certificate = np.maximum(certificate, 0.0)
                certificate_gradient = float(
                    np.linalg.norm(scaled_matrix.T @ certificate)
                )
                certificate_rhs = float(scaled_rhs @ certificate)
                raise CapacityPressureRecouplingRequired(
                    "capacity QP is primal infeasible; dependent-row Farkas "
                    f"certificate gradient={certificate_gradient:.6e}, "
                    f"rhs={certificate_rhs:.6e}, gap={dependency_gap:.6e}"
                )
            step, _, _ = min(positive_coefficients)
            old_passive = list(passive)
            dual[old_passive] -= step * coefficients
            dual[entering] += step
            blocker_tolerance = max(
                4096.0
                * np.finfo(float).eps
                * max(float(np.max(dual, initial=0.0)), 1.0),
                1.0e-14,
            )
            blocked = {
                index
                for index in old_passive
                if dual[index] <= blocker_tolerance
            }
            if not blocked:
                # Roundoff in the ratio test: drop its deterministic minimiser.
                blocked = {
                    old_passive[
                        min(positive_coefficients, key=lambda item: (item[0], item[1]))[2]
                    ]
                }
            for index in blocked:
                dual[index] = 0.0
            dual_releases += len(blocked)
            passive = [index for index in old_passive if index not in blocked]
            passive.append(entering)
        else:
            passive.append(entering)

        # Lawson--Hanson inner loop.  Solve the exact restricted problem using
        # A rather than G=A A' so a long thin wet block does not square its
        # condition number.  Any non-positive candidate multiplier is reached
        # by a feasible line search and removed from the passive set.
        while passive:
            try:
                (
                    candidate_increment,
                    candidate_multiplier,
                    _,
                ) = _solve_minimum_mass_projection(
                    mass=np.ones(variable_count),
                    rows=[scaled_matrix[index].copy() for index in passive],
                    right_hand_side=[float(scaled_rhs[index]) for index in passive],
                    velocity_scale=velocity_scale,
                    feasibility_tolerances=[
                        float(scaled_rate_tolerance[index]) for index in passive
                    ],
                )
            except CapacityPressureRecouplingRequired as exc:
                # Numerically dependent passive basis.  Audit its smallest
                # left-null direction and ratio-pivot a deterministic blocker.
                basis = scaled_matrix[passive]
                _, singular_values, right_vectors = np.linalg.svd(
                    basis.T,
                    full_matrices=True,
                )
                dependency = right_vectors[-1].copy()
                dependency_residual = float(
                    np.linalg.norm(basis.T @ dependency)
                )
                dependency_rhs = float(scaled_rhs[passive] @ dependency)
                if dependency_rhs > 0.0:
                    dependency = -dependency
                    dependency_rhs = -dependency_rhs
                blockers = [
                    (
                        dual[index] / (-direction),
                        int(retained[index]),
                        position,
                    )
                    for position, (index, direction) in enumerate(
                        zip(passive, dependency)
                    )
                    if direction < -1.0e-13
                ]
                if not blockers:
                    raise CapacityPressureRecouplingRequired(
                        "capacity QP is primal infeasible; singular passive "
                        f"Farkas residual={dependency_residual:.6e}, "
                        f"rhs={dependency_rhs:.6e}"
                    ) from exc
                step, _, _ = min(blockers)
                old_passive = list(passive)
                dual[old_passive] += step * dependency
                leaving = old_passive[
                    min(blockers, key=lambda item: (item[0], item[1]))[2]
                ]
                dual[leaving] = 0.0
                passive.remove(leaving)
                dual_releases += 1
                continue

            positive_tolerance = max(
                4096.0
                * np.finfo(float).eps
                * max(
                    float(np.max(np.abs(candidate_multiplier), initial=0.0)),
                    1.0,
                ),
                1.0e-14,
            )
            nonpositive = np.flatnonzero(
                candidate_multiplier <= positive_tolerance
            )
            if nonpositive.size == 0:
                dual.fill(0.0)
                dual[passive] = candidate_multiplier
                scaled_increment = candidate_increment
                break

            current = dual[passive]
            direction = candidate_multiplier - current
            leaving_candidates = [
                (
                    current[position] / (-direction[position]),
                    int(retained[passive[position]]),
                    position,
                )
                for position in range(len(passive))
                if direction[position] < -positive_tolerance
            ]
            if leaving_candidates:
                step, _, _ = min(leaving_candidates)
                updated = current + step * direction
                dual[passive] = np.maximum(updated, 0.0)
                leaving_tolerance = max(positive_tolerance, 1.0e-13)
                leaving = {
                    passive[position]
                    for position in range(len(passive))
                    if dual[passive[position]] <= leaving_tolerance
                }
            else:
                # Degenerate zero candidate with no negative search direction.
                leaving = {
                    passive[int(nonpositive[0])]
                }
            for index in leaving:
                dual[index] = 0.0
            passive = [index for index in passive if index not in leaving]
            dual_releases += len(leaving)

        if not passive:
            dual.fill(0.0)
            scaled_increment.fill(0.0)
        # When ``passive`` is non-empty, retain ``candidate_increment`` from
        # the direct restricted solve above.  Reconstructing it here from the
        # dual would reintroduce the same cancellation that the final primal
        # audit is designed to avoid.
        scaled_slack = scaled_rhs - scaled_matrix @ scaled_increment

    if not converged:
        projected_gradient = scaled_slack.copy()
        positive = dual > 0.0
        projected_gradient[(~positive) & (scaled_slack >= 0.0)] = 0.0
        normalized_kkt = float(
            np.max(
                np.abs(projected_gradient) / scaled_rate_tolerance,
                initial=0.0,
            )
        )
        raise CapacityPressureRecouplingRequired(
            "capacity QP dual active set did not reach KKT convergence; "
            f"pivots={maximum_sweeps}, "
            f"projected_gradient={normalized_kkt:.6e}"
        )

    multiplier[retained] = dual / row_scale
    # Preserve the accurately constrained primal returned by the restricted
    # solve.  Reconstructing it from ``-M^-1 C'lambda`` subtracts large nearly
    # cancelling pressure terms in a thin/ill-conditioned wet block and can
    # turn a roundoff-level equality residual into a visible packing error.
    # The multiplier reconstruction remains an independent stationarity audit
    # below; it must not overwrite the higher-accuracy primal state.
    increment = scaled_increment / sqrt_mass
    slack = right_hand_side - matrix @ increment
    maximum_violation = float(np.max(-slack, initial=0.0))
    if maximum_violation > 8.0 * rate_tolerance:
        raise CapacityPressureRecouplingRequired(
            "capacity QP failed its primal packing audit; "
            f"maximum_violation={maximum_violation:.12e}"
        )
    multiplier_tolerance = max(
        4096.0
        * np.finfo(float).eps
        * max(float(np.max(multiplier, initial=0.0)), 1.0),
        1.0e-15,
    )
    active = multiplier > multiplier_tolerance
    stationarity = mass * increment + matrix.T @ multiplier
    complementarity = multiplier * slack
    return (
        increment,
        multiplier,
        active,
        sweep,
        dual_releases,
        float(np.max(np.abs(stationarity), initial=0.0)),
        float(np.max(np.abs(complementarity), initial=0.0)),
    )


def project_capacity_pressure_active_set(
    *,
    upward_area: Iterable[float],
    upward_discharge: Iterable[float],
    downward_area: Iterable[float],
    downward_discharge: Iterable[float],
    bottom_upward_rate: float,
    bottom_downward_rate: float,
    top_downward_rate: float,
    liquid_capacity_area: Iterable[float],
    current_liquid_area: Iterable[float],
    dt: float,
    dz: float,
    liquid_density: float,
    preserve_stopped_partition: Iterable[bool] | None = None,
    capacity_area_rate: Iterable[float] | float | None = None,
    top_upward_rate: float | None = None,
    enforce_boundary_cell_bulk_match: bool = False,
    maximum_iterations: int | None = None,
) -> CapacityPressureProjectionResult:
    """Project provisional two-stream momentum onto liquid capacity.

    Boundary-rate arguments are non-negative directional magnitudes.  Thus the
    accepted bottom net rate is ``bottom_upward_rate-bottom_downward_rate`` and
    a supplied top net rate is ``top_upward_rate-top_downward_rate``.

    ``capacity_area_rate`` has units m2/s.  The inequality advanced over this
    substep is

    ``A + dt/dz * (J_left-J_right) <= C + dt*dC/dt``.

    Capacity constraints are unilateral.  They enter the active set only when
    violated, and a constraint whose multiplier is negative is released rather
    than creating tensile capacity pressure.  The accepted boundary rate is a
    face flux and is therefore not equated to the adjacent cell-average bulk
    discharge.  ``enforce_boundary_cell_bulk_match`` retains that former
    diagnostic constraint only for isolated projector tests; it is not the FV
    pressure boundary condition.
    ``preserve_stopped_partition`` adds sign bounds so a finite directional
    corridor may stop but cannot be silently reversed and merged by this
    projection.
    """

    au = _field(upward_area, name="upward_area")
    qu = _field(upward_discharge, name="upward_discharge")
    ad = _field(downward_area, name="downward_area")
    qd = _field(downward_discharge, name="downward_discharge")
    capacity = _field(liquid_capacity_area, name="liquid_capacity_area")
    current = _field(current_liquid_area, name="current_liquid_area")
    fields = (qu, ad, qd, capacity, current)
    if any(field.shape != au.shape for field in fields):
        raise ValueError("all cell fields must have one common shape")
    n = au.size

    scalars = (
        bottom_upward_rate,
        bottom_downward_rate,
        top_downward_rate,
        dt,
        dz,
        liquid_density,
    )
    if top_upward_rate is not None:
        scalars += (top_upward_rate,)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("boundary, grid, time, and density inputs must be finite")
    if min(dt, dz, liquid_density) <= 0.0:
        raise ValueError("dt, dz, and liquid_density must be positive")
    if min(bottom_upward_rate, bottom_downward_rate, top_downward_rate) < 0.0:
        raise ValueError("directional boundary-rate magnitudes cannot be negative")
    if top_upward_rate is not None and top_upward_rate < 0.0:
        raise ValueError("top_upward_rate cannot be negative")

    area_scale = max(
        float(np.max(np.abs(capacity), initial=0.0)),
        float(np.max(np.abs(current), initial=0.0)),
        1.0e-12,
    )
    area_tolerance = max(4096.0 * np.finfo(float).eps * area_scale, 1.0e-15)
    if np.any(au < -area_tolerance) or np.any(ad < -area_tolerance):
        raise ValueError("directional areas cannot be negative")
    au = np.maximum(au, 0.0)
    ad = np.maximum(ad, 0.0)
    reconstructed_current = au + ad
    if np.max(np.abs(reconstructed_current - current)) > area_tolerance:
        raise ValueError("current_liquid_area must equal upward_area+downward_area")
    current = reconstructed_current
    if np.any(capacity < current - area_tolerance):
        raise ValueError("liquid capacity cannot be smaller than current liquid area")
    if np.any(capacity < -area_tolerance):
        raise ValueError("liquid capacity cannot be negative")

    capacity_rate = _optional_field(
        capacity_area_rate,
        shape=au.shape,
        name="capacity_area_rate",
    )
    preserve = (
        np.zeros(n, dtype=bool)
        if preserve_stopped_partition is None
        else np.asarray(tuple(preserve_stopped_partition), dtype=bool)
    )
    if preserve.shape != au.shape:
        raise ValueError("preserve_stopped_partition must have one value per cell")

    discharge_scale = max(
        float(np.max(np.abs(qu), initial=0.0)),
        float(np.max(np.abs(qd), initial=0.0)),
        bottom_upward_rate,
        bottom_downward_rate,
        top_downward_rate,
        0.0 if top_upward_rate is None else top_upward_rate,
        area_scale * dz / dt,
        1.0e-12,
    )
    rate_tolerance = max(
        8192.0 * np.finfo(float).eps * discharge_scale,
        1.0e-14 * discharge_scale,
        1.0e-15,
    )
    if np.any((au <= area_tolerance) & (np.abs(qu) > rate_tolerance)) or np.any(
        (ad <= area_tolerance) & (np.abs(qd) > rate_tolerance)
    ):
        raise ValueError("a dry directional stream cannot carry discharge")

    wet_cells = np.flatnonzero(current > area_tolerance)
    column_for_cell = {int(cell): column for column, cell in enumerate(wet_cells)}
    variable_count = wet_cells.size
    velocity_scale = 1.0
    if wet_cells.size:
        velocity_scale = max(
            velocity_scale,
            float(np.max(np.abs(qu[wet_cells]) / current[wet_cells])),
            float(np.max(np.abs(qd[wet_cells]) / current[wet_cells])),
        )
    velocity_tolerance = max(
        8192.0 * np.finfo(float).eps * velocity_scale,
        1.0e-13 * velocity_scale,
    )

    # H maps common cell velocity increments to the *actual* net donor trace.
    face_base = np.zeros(n + 1, dtype=float)
    face_map = np.zeros((n + 1, variable_count), dtype=float)
    face_base[0] = bottom_upward_rate - bottom_downward_rate
    for face in range(1, n):
        lower = face - 1
        upper = face
        face_base[face] = qu[lower] + qd[upper]
        if lower in column_for_cell:
            face_map[face, column_for_cell[lower]] += au[lower]
        if upper in column_for_cell:
            face_map[face, column_for_cell[upper]] += ad[upper]
    if top_upward_rate is None:
        face_base[n] = qu[n - 1] - top_downward_rate
        if n - 1 in column_for_cell:
            face_map[n, column_for_cell[n - 1]] += au[n - 1]
    else:
        face_base[n] = top_upward_rate - top_downward_rate

    divergence_base = face_base[:-1] - face_base[1:]
    constraint_matrix = face_map[:-1] - face_map[1:]
    available_volume_rate = (
        (capacity - current) * dz / dt + dz * capacity_rate
    )
    constraint_rhs = available_volume_rate - divergence_base
    capacity_end = capacity + dt * capacity_rate
    if np.any(capacity_end < -area_tolerance):
        raise ValueError("capacity_area_rate produces a negative end-of-step capacity")
    saturated = capacity - current <= area_tolerance

    # The accepted boundary transaction is a half-cell constraint on bulk
    # liquid momentum.  It must use the same common increment as every other
    # pressure row; anchoring q_down alone would alter the resolved slip and
    # introduce a hidden directional impulse.
    bottom_anchor_active = bool(
        enforce_boundary_cell_bulk_match
        and saturated[0]
        and current[0] > area_tolerance
    )
    top_anchor_active = bool(
        enforce_boundary_cell_bulk_match
        and top_upward_rate is not None
        and saturated[-1]
        and current[-1] > area_tolerance
    )

    lower = np.full(variable_count, -np.inf)
    upper = np.full(variable_count, np.inf)
    for cell in wet_cells:
        column = column_for_cell[int(cell)]
        if not preserve[cell]:
            continue
        if au[cell] > area_tolerance:
            lower[column] = -qu[cell] / au[cell]
        if ad[cell] > area_tolerance:
            upper[column] = -qd[cell] / ad[cell]
        if lower[column] > upper[column] + velocity_tolerance:
            raise CapacityPressureRecouplingRequired(
                f"preserved directional signs are already incompatible in cell {cell}"
            )

    mass = liquid_density * dz * current[wet_cells]
    # This source-stage projection follows an already accepted conservative
    # area update.  Only cells that are *currently* at capacity are eligible
    # for incompressibility pressure; a neighbouring cell with storage must
    # remain free to receive the next transaction, whose boundary rate will be
    # recomputed by the T Riemann solve.  Start with the mandatory accepted-face
    # anchors alone, then add an eligible capacity row only if that anchored
    # solution would compress it.  Pre-activating every saturated row can make
    # a draining (tensile-pressure) state look falsely infeasible.
    capacity_eligible = saturated.copy()
    active_capacity: set[int] = set()
    active_lower: set[int] = set()
    active_upper: set[int] = set()
    # Directional sign bounds are secondary topology guards.  Mandatory bulk
    # anchoring and non-overpacking are primary conservation constraints.  If
    # a sign equality makes those primary rows incompatible, release only that
    # guard and let the caller's conservative topology operator relabel the
    # crossed stream after this common-pressure step.
    released_lower: set[int] = set()
    released_upper: set[int] = set()
    maximum_iterations = (
        max(16, 8 * (n + variable_count + 1))
        if maximum_iterations is None
        else int(maximum_iterations)
    )
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    increment = np.zeros(variable_count)
    capacity_multiplier = np.zeros(n)
    lower_multiplier = np.zeros(variable_count)
    upper_multiplier = np.zeros(variable_count)
    stationarity = 0.0
    working_set_capacity_releases = 0
    converged = False

    # Production FV path: no diagnostic cell/boundary match and no directional
    # sign guard.  Solve the convex unilateral problem directly in the dual so
    # a rank-deficient saturated block can pivot several rows to slack without
    # ever compromising a capacity inequality.
    dual_capacity_path = bool(
        not bottom_anchor_active
        and not top_anchor_active
        and not np.any(np.isfinite(lower))
        and not np.any(np.isfinite(upper))
    )
    if dual_capacity_path:
        (
            increment,
            capacity_multiplier,
            dual_active_mask,
            iteration,
            working_set_capacity_releases,
            stationarity,
            _,
        ) = _solve_unilateral_capacity_qp_dual(
            mass=mass,
            matrix=constraint_matrix,
            right_hand_side=constraint_rhs,
            rate_tolerance=rate_tolerance,
            velocity_scale=velocity_scale,
            maximum_sweeps=max(
                512,
                maximum_iterations * max(4, min(n, 16)),
            ),
        )
        active_capacity = set(np.flatnonzero(dual_active_mask).tolist())
        converged = True

    for iteration in (() if converged else range(1, maximum_iterations + 1)):
        rows: list[Array] = []
        rhs: list[float] = []
        tolerances: list[float] = []
        metadata: list[tuple[str, int]] = []
        if bottom_anchor_active:
            row = np.zeros(variable_count)
            row[column_for_cell[0]] = current[0]
            rows.append(row)
            rhs.append(
                float(
                    bottom_upward_rate
                    - bottom_downward_rate
                    - qu[0]
                    - qd[0]
                )
            )
            tolerances.append(8.0 * rate_tolerance)
            metadata.append(("bottom_bulk", 0))
        if top_anchor_active:
            row = np.zeros(variable_count)
            row[column_for_cell[n - 1]] = current[-1]
            rows.append(row)
            rhs.append(
                float(
                    float(top_upward_rate)
                    - top_downward_rate
                    - qu[-1]
                    - qd[-1]
                )
            )
            tolerances.append(8.0 * rate_tolerance)
            metadata.append(("top_bulk", n - 1))
        for cell in sorted(active_capacity):
            rows.append(constraint_matrix[cell].copy())
            rhs.append(float(constraint_rhs[cell]))
            tolerances.append(8.0 * rate_tolerance)
            metadata.append(("capacity", cell))
        for column in sorted(active_lower):
            row = np.zeros(variable_count)
            row[column] = -1.0
            rows.append(row)
            rhs.append(float(-lower[column]))
            tolerances.append(8.0 * velocity_tolerance)
            metadata.append(("lower", column))
        for column in sorted(active_upper):
            row = np.zeros(variable_count)
            row[column] = 1.0
            rows.append(row)
            rhs.append(float(upper[column]))
            tolerances.append(8.0 * velocity_tolerance)
            metadata.append(("upper", column))

        try:
            increment, multipliers, stationarity = _solve_minimum_mass_projection(
                mass=mass,
                rows=rows,
                right_hand_side=rhs,
                velocity_scale=velocity_scale,
                feasibility_tolerances=tolerances,
            )
        except CapacityPressureRecouplingRequired as exc:
            released_bound: tuple[str, int] | None = None
            bound_positions = [
                position
                for position, (kind, _) in enumerate(metadata)
                if kind in ("lower", "upper")
            ]
            # The same primary rows were feasible before a sign guard became
            # active.  Test removal explicitly rather than accepting a least-
            # squares compromise in packing or in the accepted boundary rate.
            for position in reversed(bound_positions):
                trial_rows = [
                    row for row_index, row in enumerate(rows)
                    if row_index != position
                ]
                trial_rhs = [
                    value for row_index, value in enumerate(rhs)
                    if row_index != position
                ]
                trial_tolerances = [
                    value for row_index, value in enumerate(tolerances)
                    if row_index != position
                ]
                try:
                    _solve_minimum_mass_projection(
                        mass=mass,
                        rows=trial_rows,
                        right_hand_side=trial_rhs,
                        velocity_scale=velocity_scale,
                        feasibility_tolerances=trial_tolerances,
                    )
                except CapacityPressureRecouplingRequired:
                    continue
                released_bound = metadata[position]
                break
            if released_bound is not None:
                kind, index = released_bound
                if kind == "lower":
                    active_lower.remove(index)
                    released_lower.add(index)
                else:
                    active_upper.remove(index)
                    released_upper.add(index)
                continue

            # A unilateral capacity inequality may have entered a degenerate
            # working set as an equality even though the full inequality QP is
            # feasible with that row slack.  Audit one-row releases without
            # relaxing any packing condition: the trial is admissible only if
            # every capacity inequality, including the removed one and storage
            # cells, remains satisfied.  If no trial passes, this is genuine
            # primal incompatibility and must still be recoupled upstream.
            capacity_release_trials: list[tuple[float, int]] = []
            capacity_audit_tolerance = 8.0 * rate_tolerance
            for released_cell in sorted(active_capacity):
                position = metadata.index(("capacity", released_cell))
                trial_rows = [
                    row for row_index, row in enumerate(rows)
                    if row_index != position
                ]
                trial_rhs = [
                    value for row_index, value in enumerate(rhs)
                    if row_index != position
                ]
                trial_tolerances = [
                    value for row_index, value in enumerate(tolerances)
                    if row_index != position
                ]
                try:
                    trial_increment, _, _ = _solve_minimum_mass_projection(
                        mass=mass,
                        rows=trial_rows,
                        right_hand_side=trial_rhs,
                        velocity_scale=velocity_scale,
                        feasibility_tolerances=trial_tolerances,
                    )
                except CapacityPressureRecouplingRequired:
                    continue
                trial_capacity_violation = (
                    constraint_matrix @ trial_increment - constraint_rhs
                )
                if float(
                    np.max(trial_capacity_violation, initial=-np.inf)
                ) > capacity_audit_tolerance:
                    continue
                objective = 0.5 * float(
                    np.sum(mass * trial_increment * trial_increment)
                )
                capacity_release_trials.append((objective, released_cell))
            if capacity_release_trials:
                _, released_cell = min(
                    capacity_release_trials,
                    key=lambda candidate: (candidate[0], candidate[1]),
                )
                active_capacity.remove(released_cell)
                working_set_capacity_releases += 1
                continue
            raise CapacityPressureRecouplingRequired(
                f"{exc}; active_capacity={sorted(active_capacity)}, "
                f"active_lower={sorted(active_lower)}, "
                f"active_upper={sorted(active_upper)}, "
                f"rhs={[float(value) for value in rhs]}, "
                f"rows={[np.flatnonzero(np.abs(row) > 0.0).tolist() for row in rows]}, "
                f"bottom_rates=({bottom_upward_rate:.12e},"
                f"{bottom_downward_rate:.12e}), "
                f"top_rates=({top_upward_rate},{top_downward_rate:.12e})"
            ) from exc
        capacity_multiplier.fill(0.0)
        lower_multiplier.fill(0.0)
        upper_multiplier.fill(0.0)
        for (kind, index), value in zip(metadata, multipliers):
            if kind == "capacity":
                capacity_multiplier[index] = value
            elif kind == "lower":
                lower_multiplier[index] = value
            elif kind == "upper":
                upper_multiplier[index] = value

        active_values = [
            abs(capacity_multiplier[i]) for i in active_capacity
        ] + [abs(lower_multiplier[i]) for i in active_lower] + [
            abs(upper_multiplier[i]) for i in active_upper
        ]
        multiplier_tolerance = max(
            1.0e-12,
            1.0e-11 * max(active_values, default=1.0),
        )
        negative: list[tuple[float, str, int]] = []
        negative.extend(
            (capacity_multiplier[i], "capacity", i)
            for i in active_capacity
            if capacity_multiplier[i] < -multiplier_tolerance
        )
        negative.extend(
            (lower_multiplier[i], "lower", i)
            for i in active_lower
            if lower_multiplier[i] < -multiplier_tolerance
        )
        negative.extend(
            (upper_multiplier[i], "upper", i)
            for i in active_upper
            if upper_multiplier[i] < -multiplier_tolerance
        )
        if negative:
            _, kind, index = min(negative, key=lambda item: item[0])
            if kind == "capacity":
                active_capacity.remove(index)
            elif kind == "lower":
                active_lower.remove(index)
            else:
                active_upper.remove(index)
            continue

        capacity_violation = constraint_matrix @ increment - constraint_rhs
        # A formerly incompatible topology guard may be reconsidered if later
        # active-set changes bring the primary solution back to its admissible
        # side without enforcing it.
        released_lower = {
            index
            for index in released_lower
            if increment[index] < lower[index] - velocity_tolerance
        }
        released_upper = {
            index
            for index in released_upper
            if increment[index] > upper[index] + velocity_tolerance
        }
        candidates: list[tuple[float, str, int]] = []
        candidates.extend(
            (capacity_violation[i] / rate_tolerance, "capacity", i)
            for i in range(n)
            if capacity_eligible[i]
            and i not in active_capacity
            and capacity_violation[i] > rate_tolerance
        )
        candidates.extend(
            ((lower[i] - increment[i]) / velocity_tolerance, "lower", i)
            for i in range(variable_count)
            if i not in active_lower
            and i not in released_lower
            and np.isfinite(lower[i])
            and increment[i] < lower[i] - velocity_tolerance
        )
        candidates.extend(
            ((increment[i] - upper[i]) / velocity_tolerance, "upper", i)
            for i in range(variable_count)
            if i not in active_upper
            and i not in released_upper
            and np.isfinite(upper[i])
            and increment[i] > upper[i] + velocity_tolerance
        )
        if candidates:
            _, kind, index = max(candidates, key=lambda item: item[0])
            if kind == "capacity":
                active_capacity.add(index)
            elif kind == "lower":
                active_lower.add(index)
            else:
                active_upper.add(index)
            continue
        converged = True
        break
    if not converged:
        raise CapacityPressureProjectionError(
            "capacity-pressure active set did not converge"
        )

    x = np.zeros(n)
    x[wet_cells] = increment
    corrected_qu = qu + au * x
    corrected_qd = qd + ad * x
    corrected_qu[np.abs(corrected_qu) <= rate_tolerance] = 0.0
    corrected_qd[np.abs(corrected_qd) <= rate_tolerance] = 0.0
    net_face = face_base + face_map @ increment
    divergence = net_face[:-1] - net_face[1:]
    predicted_area = current + dt / dz * divergence
    packing_residual = predicted_area - capacity_end

    pressure_impulse = np.zeros(n + 1)
    # Each connected wet component uses its upper exposed face as pressure
    # gauge.  Enumerate components explicitly: clearing a dry cell must not
    # overwrite the already reconstructed lower face of the wet block above.
    cell = 0
    while cell < n:
        if current[cell] <= area_tolerance:
            cell += 1
            continue
        start = cell
        while cell + 1 < n and current[cell + 1] > area_tolerance:
            cell += 1
        end = cell + 1
        pressure_impulse[end] = 0.0
        for index in range(end - 1, start - 1, -1):
            pressure_impulse[index] = (
                pressure_impulse[index + 1]
                + liquid_density * dz * x[index]
            )
        cell = end

    delta_total_q = corrected_qu + corrected_qd - qu - qd
    cell_kinematic = dz * delta_total_q
    cell_physical = liquid_density * cell_kinematic
    liquid_kinematic = float(np.sum(cell_kinematic))
    liquid_physical = float(np.sum(cell_physical))
    bottom_pressure = float(current[0] * pressure_impulse[0])
    top_pressure = float(-current[-1] * pressure_impulse[-1])
    internal_pressure = float(
        np.sum((current[1:] - current[:-1]) * pressure_impulse[1:-1])
    )
    pressure_residual = (
        liquid_physical - bottom_pressure - top_pressure - internal_pressure
    )
    boundary_owner_reaction = -(bottom_pressure + top_pressure)
    interface_owner_reaction = -internal_pressure
    coupled_residual = (
        liquid_physical + boundary_owner_reaction + interface_owner_reaction
    )

    active_mask = np.zeros(n, dtype=bool)
    active_mask[list(active_capacity)] = True
    capacity_slack = constraint_rhs - constraint_matrix @ increment
    complementarity = float(
        np.max(np.abs(capacity_multiplier * capacity_slack), initial=0.0)
    )
    if variable_count:
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if np.any(finite_lower):
            complementarity = max(
                complementarity,
                float(
                    np.max(
                        np.abs(
                            lower_multiplier[finite_lower]
                            * (increment[finite_lower] - lower[finite_lower])
                        ),
                        initial=0.0,
                    )
                ),
            )
        if np.any(finite_upper):
            complementarity = max(
                complementarity,
                float(
                    np.max(
                        np.abs(
                            upper_multiplier[finite_upper]
                            * (upper[finite_upper] - increment[finite_upper])
                        ),
                        initial=0.0,
                    )
                ),
            )
    active_residual = (
        float(
            np.max(
                np.abs(divergence[active_mask] - available_volume_rate[active_mask]),
                initial=0.0,
            )
        )
        if np.any(active_mask)
        else 0.0
    )
    bottom_anchor_residual = (
        float(
            corrected_qu[0]
            + corrected_qd[0]
            - bottom_upward_rate
            + bottom_downward_rate
        )
        if bottom_anchor_active
        else 0.0
    )
    top_anchor_residual = (
        float(
            corrected_qu[-1]
            + corrected_qd[-1]
            - float(top_upward_rate)
            + top_downward_rate
        )
        if top_anchor_active
        else 0.0
    )

    ledger = CapacityPressureLedger(
        cell_kinematic_impulse=_readonly(cell_kinematic),
        cell_physical_impulse=_readonly(cell_physical),
        liquid_kinematic_impulse=liquid_kinematic,
        liquid_physical_impulse=liquid_physical,
        bottom_pressure_impulse_on_liquid=bottom_pressure,
        top_pressure_impulse_on_liquid=top_pressure,
        internal_area_pressure_impulse_on_liquid=internal_pressure,
        boundary_owner_reaction_impulse=boundary_owner_reaction,
        interface_owner_reaction_impulse=interface_owner_reaction,
        pressure_decomposition_residual=pressure_residual,
        coupled_momentum_residual=coupled_residual,
    )
    return CapacityPressureProjectionResult(
        corrected_upward_discharge=_readonly(corrected_qu),
        corrected_downward_discharge=_readonly(corrected_qd),
        common_velocity_increment=_readonly(x),
        net_face_discharge=_readonly(net_face),
        predicted_liquid_area=_readonly(predicted_area),
        face_pressure_impulse=_readonly(pressure_impulse),
        active_capacity_mask=_readonly(active_mask),
        capacity_multiplier=_readonly(capacity_multiplier),
        lower_bound_multiplier=_readonly(lower_multiplier),
        upper_bound_multiplier=_readonly(upper_multiplier),
        iterations=iteration,
        working_set_capacity_releases=working_set_capacity_releases,
        maximum_packing_residual=float(np.max(packing_residual, initial=0.0)),
        maximum_active_constraint_residual=active_residual,
        maximum_kkt_stationarity_residual=stationarity,
        maximum_complementarity_residual=complementarity,
        bottom_bulk_anchor_residual=bottom_anchor_residual,
        top_bulk_anchor_residual=top_anchor_residual,
        ledger=ledger,
    )


__all__ = (
    "CapacityPressureLedger",
    "CapacityPressureProjectionError",
    "CapacityPressureProjectionResult",
    "CapacityPressureRecouplingRequired",
    "project_capacity_pressure_active_set",
)
