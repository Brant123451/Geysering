"""Pure stage residual for the resolved post-T Case-A gas graph.

The production state owner stores cell-integrated conservative variables

``Mgt, Jgt``
    horizontal gas mass and axial momentum, and

``Mgr, Jgrs, Mgrs``
    vertical total-gas mass, vertical gas momentum, and tunnel-origin tracer
    mass.

This module evaluates, but never commits, one method-of-lines right-hand side
for those five fields.  Topology is supplied explicitly by independent east
and vertical material fronts.  In particular, the evaluator does not discover
or create a receiver cell, redistribute gas to equalise pressure, or keep a
second lumped gas inventory.  The same stage state can therefore be evaluated
twice without side effects, as required by SSP-RK schemes.

The T-mouth gas flux is one internal conservative face: it is subtracted from
the horizontal T cell and added to the vertical base cell by the underlying
network residual.  The atmospheric top flux is the only external gas-mass
flux.  A material-front speed is obtained from the conservative kinematic
identity ``s = m_dot/(rho A)``.  At a newly-created zero-length vertical
branch, ``m_dot`` is the actual T-face Riemann mass flux; no empirical launch
speed or pre-filled receiver is used.

The east material front follows the same rule.  Its leading, geometrically
open cut cell is filled only through the ordinary horizontal MUSCL--Riemann
face shared with the connected gas cell immediately to its west.  That single
face flux is already subtracted from the donor and added to the cut cell by
the finite-volume residual; the identical mass rate supplies the front
kinematics.  A metadata front can therefore neither move across a liquid-full
cell nor advance without an actual conservative gas transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from casea_coupled_gas_network import (
    CoupledGasParameters,
    _friction_factor,
    _gas_flux,
    _horizontal_geometry,
    _network_rhs,
    _slopes,
    _vertical_geometry,
    junction_mouth_area,
)
from casea_topology_event import BranchFrontTopology


Array = np.ndarray


@dataclass(frozen=True)
class ExternalGasFaceFlux:
    """Integrated flux on the atmospheric-top outward normal."""

    area: float
    mass_rate: float
    momentum_flux: float
    tracer_mass_rate: float


@dataclass(frozen=True)
class InternalTGasFlux:
    """One paired T-mouth gas flux, positive horizontal-to-vertical."""

    area: float
    mass_rate_horizontal_to_vertical: float
    normal_momentum_flux: float
    tracer_mass_rate_horizontal_to_vertical: float
    horizontal_mass_rate: float
    vertical_mass_rate: float
    internal_mass_residual: float


@dataclass(frozen=True)
class InternalEastFrontGasFlux:
    """Paired horizontal flux into the leading east material cut cell.

    ``mass_rate_west_to_east`` is the exact MUSCL--Riemann mass rate used by
    the horizontal finite-volume face.  The two signed rates below expose the
    conservative donor/receiver ledger without performing a remap.
    """

    area: float
    mass_rate_west_to_east: float
    normal_momentum_flux: float
    donor_index: int
    receiver_index: int
    donor_mass_rate: float
    receiver_mass_rate: float
    upwind_density: float
    internal_mass_residual: float


@dataclass(frozen=True)
class MaterialFrontRHS:
    """Conservative kinematics for one explicitly-owned material front."""

    branch: str
    position: float
    speed: float
    mass_rate: float
    area: float
    upwind_density: float
    source_face: str


@dataclass(frozen=True)
class ResolvedGasStageRHS:
    """Cell-integrated gas RHS and non-committing boundary diagnostics."""

    dMgt_dt: Array
    dJgt_dt: Array
    dMgr_dt: Array
    dJgrs_dt: Array
    dMgrs_dt: Array
    dJgt_drag_dt: Array
    dJgrs_drag_dt: Array
    dQlt_drag_dt: Array
    dQlr_drag_dt: Array
    top_flux: ExternalGasFaceFlux
    t_flux: InternalTGasFlux
    east_flux: InternalEastFrontGasFlux
    east_front: MaterialFrontRHS
    vertical_front: MaterialFrontRHS
    horizontal_active: Array
    vertical_bottom_active: Array
    vertical_top_active: Array
    total_gas_mass_ledger_residual: float
    tracer_mass_ledger_residual: float
    horizontal_interphase_momentum_residual: float
    vertical_interphase_momentum_residual: float


def _readonly(values: Array) -> Array:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _state_velocity(
    mass_per_length: float,
    momentum_per_length: float,
    gas_area: float,
    params: CoupledGasParameters,
) -> float:
    """Use exactly the velocity trace admitted by ``_network_rhs``."""

    if mass_per_length <= 0.0 or gas_area <= 0.0:
        return 0.0
    density = mass_per_length / gas_area
    lower = params.resolved_density_fraction * params.rho_atmospheric
    upper = params.resolved_density_ceiling * params.rho_atmospheric
    if not lower < density < upper:
        return 0.0
    return momentum_per_length / mass_per_length


def _validate_front(front: BranchFrontTopology, expected: str) -> None:
    if not isinstance(front, BranchFrontTopology):
        raise TypeError(f"{expected}_front must be BranchFrontTopology")
    if front.branch != expected:
        raise ValueError(f"{expected}_front owns the wrong branch")


def _front_cells(
    geometric_void: Array,
    *,
    first_index: int,
    position: float,
    cell_width: float,
    minimum_void: float,
    active: bool,
) -> Array:
    """Return cells geometrically swept by one outward material front.

    A cell whose branch-side face coincides with a zero-length front is active
    only when the liquid solve has already created finite geometric void in
    that cell.  This permits a zero front to initialise without manufacturing
    volume or gas mass in this evaluator.
    """

    mask = np.zeros(geometric_void.size, dtype=bool)
    if not active or first_index >= geometric_void.size:
        return mask
    branch_index = np.arange(
        geometric_void.size - first_index, dtype=float
    )
    cell_near_face = branch_index * cell_width
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(position), cell_width
    )
    swept_or_open_cut = cell_near_face <= position + tolerance
    mask[first_index:] = (
        swept_or_open_cut
        & (geometric_void[first_index:] > minimum_void)
    )
    return mask


def _top_component(
    geometric_void: Array,
    *,
    minimum_void: float,
    top_open: bool,
) -> Array:
    mask = np.zeros(geometric_void.size, dtype=bool)
    if not top_open:
        return mask
    for index in range(geometric_void.size - 1, -1, -1):
        if geometric_void[index] <= minimum_void:
            break
        mask[index] = True
    return mask


def _kinematics_from_mass_flux(
    *,
    branch: str,
    position: float,
    mass_rate: float,
    area: float,
    density_positive: float,
    density_negative: float,
    source_face: str,
) -> MaterialFrontRHS:
    density = density_positive if mass_rate >= 0.0 else density_negative
    if area <= 0.0 or density <= 0.0 or not math.isfinite(density):
        speed = 0.0
        density = max(float(density), 0.0) if math.isfinite(density) else 0.0
    else:
        speed = mass_rate / (density * area)
    return MaterialFrontRHS(
        branch=branch,
        position=float(position),
        speed=float(speed),
        mass_rate=float(mass_rate),
        area=float(area),
        upwind_density=float(density),
        source_face=source_face,
    )


def _closed_east_front_flux() -> InternalEastFrontGasFlux:
    return InternalEastFrontGasFlux(
        area=0.0,
        mass_rate_west_to_east=0.0,
        normal_momentum_flux=0.0,
        donor_index=-1,
        receiver_index=-1,
        donor_mass_rate=0.0,
        receiver_mass_rate=0.0,
        upwind_density=0.0,
        internal_mass_residual=0.0,
    )


def _east_front_riemann_flux(
    mass_per_length: Array,
    momentum_per_length: Array,
    gas_area: Array,
    face_area: Array,
    horizontal_active: Array,
    east_active: Array,
    *,
    junction_index: int,
    params: CoupledGasParameters,
) -> InternalEastFrontGasFlux:
    """Return the actual FV flux into the leading east cut cell.

    The leading cut cell is the outermost east cell currently admitted by the
    material-front geometry.  Its west neighbour must belong to the same
    connected gas component.  Reconstruction and Riemann evaluation duplicate
    the horizontal-face operations in ``_network_rhs`` exactly, so this
    diagnostic is the same flux already present with opposite signs in the
    donor and receiver residuals; it is not an additional source or remap.
    """

    east_indices = np.flatnonzero(east_active)
    if east_indices.size == 0:
        return _closed_east_front_flux()

    receiver = int(east_indices[-1])
    donor = receiver - 1
    face = receiver
    if (
        donor < int(junction_index)
        or not bool(horizontal_active[donor])
        or not bool(horizontal_active[receiver])
        or float(face_area[face]) <= 0.0
    ):
        return _closed_east_front_flux()

    density = np.maximum(
        np.asarray(mass_per_length, dtype=float)
        / np.maximum(np.asarray(gas_area, dtype=float), 1.0e-300),
        1.0e-10,
    )
    velocity = np.array(
        [
            _state_velocity(mass, momentum, area, params)
            for mass, momentum, area in zip(
                mass_per_length, momentum_per_length, gas_area
            )
        ],
        dtype=float,
    )
    density_slope = _slopes(density, params.limiter_theta)
    velocity_slope = _slopes(velocity, params.limiter_theta)
    density_left = max(
        float(density[donor] + 0.5 * density_slope[donor]), 1.0e-10
    )
    velocity_left = float(
        velocity[donor] + 0.5 * velocity_slope[donor]
    )
    density_right = max(
        float(density[receiver] - 0.5 * density_slope[receiver]), 1.0e-10
    )
    velocity_right = float(
        velocity[receiver] - 0.5 * velocity_slope[receiver]
    )
    mass_flux_per_area, momentum_flux_per_area = _gas_flux(
        density_left,
        velocity_left,
        density_right,
        velocity_right,
        params.sound_speed,
        params.rho_atmospheric,
        params.entropy_fix_fraction,
        params.resolved_density_fraction,
        params.resolved_density_ceiling,
    )
    area = float(face_area[face])
    mass_rate = float(mass_flux_per_area * area)
    momentum_flux = float(momentum_flux_per_area * area)
    upwind_density = density_left if mass_rate >= 0.0 else density_right
    return InternalEastFrontGasFlux(
        area=area,
        mass_rate_west_to_east=mass_rate,
        normal_momentum_flux=momentum_flux,
        donor_index=donor,
        receiver_index=receiver,
        donor_mass_rate=-mass_rate,
        receiver_mass_rate=mass_rate,
        upwind_density=float(upwind_density),
        internal_mass_residual=(-mass_rate + mass_rate),
    )


def _interphase_drag_rhs(
    gas_mass: Array,
    gas_momentum: Array,
    liquid_area: Array,
    liquid_discharge: Array,
    gas_area: Array,
    interface_perimeter: Array,
    hydraulic_diameter: Array,
    active: Array,
    *,
    cell_width: float,
    rho_l: float,
    gas_viscosity: float,
    liquid_holdup_drag_enhancement: float = 0.0,
) -> tuple[Array, Array, float]:
    """Return one equal-and-opposite, instantaneous interphase drag source.

    ``gas_mass`` and ``gas_momentum`` are cell-integrated.  The liquid
    conservative momentum in one cell is ``rho_l * Q_l * dx``.  Therefore an
    interfacial force ``F_i`` contributes ``-F_i`` to gas momentum and
    ``+F_i/(rho_l*dx)`` to the liquid-discharge RHS.  This evaluator contains
    no split-step receiver, floor, or velocity prescription; both phases are
    evaluated from the same Runge--Kutta stage state.
    """

    gm = np.asarray(gas_mass, dtype=float)
    gj = np.asarray(gas_momentum, dtype=float)
    al = np.asarray(liquid_area, dtype=float)
    ql = np.asarray(liquid_discharge, dtype=float)
    ag = np.asarray(gas_area, dtype=float)
    perimeter = np.asarray(interface_perimeter, dtype=float)
    diameter = np.asarray(hydraulic_diameter, dtype=float)
    mask = np.asarray(active, dtype=bool)
    if not (
        gm.shape == gj.shape == al.shape == ql.shape == ag.shape
        == perimeter.shape == diameter.shape == mask.shape
    ):
        raise ValueError("interphase-drag arrays must have equal shape")
    if liquid_holdup_drag_enhancement < 0.0:
        raise ValueError("liquid-holdup drag enhancement must be non-negative")

    d_j = np.zeros_like(gj)
    d_q = np.zeros_like(ql)
    for index in np.flatnonzero(
        mask & (gm > 1.0e-14) & (al > 0.0) & (ag > 1.0e-14)
        & (perimeter > 0.0)
    ):
        gas_velocity = gj[index] / gm[index]
        liquid_velocity = ql[index] / al[index]
        relative = gas_velocity - liquid_velocity
        if abs(relative) <= 1.0e-14:
            continue
        gas_density = gm[index] / (ag[index] * cell_width)
        reynolds = (
            gas_density * abs(relative) * max(diameter[index], 0.0)
            / max(gas_viscosity, 1.0e-18)
        )
        friction = float(_friction_factor(reynolds))
        liquid_holdup = min(
            max(al[index] / max(al[index] + ag[index], 1.0e-18), 0.0),
            1.0,
        )
        friction *= 1.0 + liquid_holdup_drag_enhancement * liquid_holdup
        coefficient = (
            0.5 * friction * gas_density * perimeter[index] * cell_width
        )
        force_on_liquid = coefficient * relative * abs(relative)
        d_j[index] = -force_on_liquid
        d_q[index] = force_on_liquid / (rho_l * cell_width)

    residual = float(
        np.sum(d_j, dtype=np.float64)
        + rho_l * cell_width * np.sum(d_q, dtype=np.float64)
    )
    return d_j, d_q, residual


def evaluate_resolved_gas_stage_rhs(
    Mgt: Array,
    Jgt: Array,
    Mgr: Array,
    Jgrs: Array,
    Mgrs: Array,
    Alt: Array,
    Alr: Array,
    Qlt: Array,
    Qlr: Array,
    *,
    dx: float,
    dz: float,
    junction_index: int,
    params: CoupledGasParameters,
    east_front: BranchFrontTopology,
    vertical_front: BranchFrontTopology,
    top_open: bool,
) -> ResolvedGasStageRHS:
    """Evaluate a pure resolved-gas stage residual.

    All five conservative inputs are cell-integrated.  Returned derivatives
    therefore have units of the corresponding cell-integrated quantity per
    second.  ``east_front`` and ``vertical_front`` are independent graph
    metadata; neither is inferred from the other or from a mass threshold.
    The topology is frozen for this residual evaluation and must be advanced by
    the caller together with the conservative fields.
    """

    _validate_front(east_front, "east")
    _validate_front(vertical_front, "vertical")
    if not isinstance(top_open, (bool, np.bool_)):
        raise TypeError("top_open must be boolean")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    if not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("dz must be positive and finite")

    hm = np.asarray(Mgt, dtype=float)
    hj = np.asarray(Jgt, dtype=float)
    vm = np.asarray(Mgr, dtype=float)
    vj = np.asarray(Jgrs, dtype=float)
    vc = np.asarray(Mgrs, dtype=float)
    h_al = np.asarray(Alt, dtype=float)
    v_al = np.asarray(Alr, dtype=float)
    h_ql = np.asarray(Qlt, dtype=float)
    v_ql = np.asarray(Qlr, dtype=float)
    if not (
        hm.ndim == 1 and hm.shape == hj.shape == h_al.shape == h_ql.shape
    ):
        raise ValueError(
            "Mgt, Jgt, Alt, and Qlt must be equal-length 1-D arrays"
        )
    if not (
        vm.ndim == 1
        and vm.shape == vj.shape == vc.shape == v_al.shape == v_ql.shape
    ):
        raise ValueError(
            "Mgr, Jgrs, Mgrs, Alr, and Qlr must be equal-length 1-D arrays"
        )
    if hm.size < 2 or vm.size < 1:
        raise ValueError("resolved gas grids must be non-empty")
    jx = int(junction_index)
    if jx != junction_index or not 0 <= jx < hm.size - 1:
        raise ValueError("junction_index must precede an east-branch cell")
    all_arrays = (hm, hj, vm, vj, vc, h_al, v_al, h_ql, v_ql)
    if any(np.any(~np.isfinite(values)) for values in all_arrays):
        raise ValueError("resolved gas stage inputs must be finite")
    if np.any(hm < 0.0) or np.any(vm < 0.0) or np.any(vc < 0.0):
        raise ValueError("gas and tracer masses must be non-negative")
    if np.any(vc > vm + 1.0e-13):
        raise ValueError("vertical tracer mass cannot exceed total gas mass")
    if np.any(h_al < 0.0):
        raise ValueError("horizontal liquid area cannot be negative")
    if np.any(v_al < 0.0):
        raise ValueError("vertical liquid area cannot be negative")
    # A_l>A_f is the elastic water-hammer state of the mixed-flow model, not
    # excess geometric gas volume.  The gas graph sees zero void there while
    # the liquid pressure law retains the conservative overfill unchanged.

    h_void_geometric = np.maximum(params.horizontal_area - h_al, 0.0)
    v_void_geometric = np.maximum(params.vertical_area - v_al, 0.0)
    horizontal_active = np.zeros(hm.size, dtype=bool)
    horizontal_active[: jx + 1] = (
        h_void_geometric[: jx + 1]
        > params.horizontal_capillary_void_fraction
        * params.horizontal_area
    )
    east_active = _front_cells(
        h_void_geometric,
        first_index=jx + 1,
        position=float(east_front.position),
        cell_width=float(dx),
        minimum_void=(
            params.horizontal_capillary_void_fraction
            * params.horizontal_area
        ),
        active=bool(east_front.active),
    )
    horizontal_active |= east_active

    vertical_bottom_active = _front_cells(
        v_void_geometric,
        first_index=0,
        position=float(vertical_front.position),
        cell_width=float(dz),
        minimum_void=(
            params.vertical_front_void_fraction * params.vertical_area
        ),
        active=bool(vertical_front.active),
    )
    vertical_top_active = _top_component(
        v_void_geometric,
        minimum_void=(
            params.vertical_front_void_fraction * params.vertical_area
        ),
        top_open=bool(top_open),
    )
    vertical_active = vertical_bottom_active | vertical_top_active

    h_topological_void = np.where(
        horizontal_active, h_void_geometric, 0.0
    )
    v_topological_void = np.where(
        vertical_active, v_void_geometric, 0.0
    )
    h_effective_liquid = params.horizontal_area - h_topological_void
    v_effective_liquid = params.vertical_area - v_topological_void
    _, h_ag, h_depth, h_pg, h_pi, h_dh = _horizontal_geometry(
        h_effective_liquid, params
    )
    _, v_ag, v_pg, v_pi, v_dh = _vertical_geometry(
        v_effective_liquid, params
    )

    h_faces = np.zeros(hm.size + 1)
    h_faces[0] = h_ag[0] if horizontal_active[0] else 0.0
    h_faces[-1] = h_ag[-1] if horizontal_active[-1] else 0.0
    h_faces[1:-1] = np.where(
        horizontal_active[:-1] & horizontal_active[1:],
        np.minimum(h_ag[:-1], h_ag[1:]),
        0.0,
    )
    v_faces = np.zeros(vm.size + 1)
    v_faces[1:-1] = np.where(
        vertical_active[:-1] & vertical_active[1:],
        np.minimum(v_ag[:-1], v_ag[1:]),
        0.0,
    )
    v_faces[-1] = (
        v_ag[-1]
        if bool(top_open) and vertical_top_active[-1]
        else 0.0
    )

    geometric_mouth = junction_mouth_area(
        h_topological_void[jx] / params.horizontal_area,
        params,
    )
    mouth_area = (
        min(geometric_mouth, float(v_ag[0]))
        if horizontal_active[jx] and vertical_bottom_active[0]
        else 0.0
    )
    v_faces[0] = mouth_area

    # Confined bottom gas receives the complementary liquid-pressure source;
    # a component connected to the atmospheric top does not.  This is the same
    # open-core distinction used by the complete network advance.
    connected_open_core = vertical_bottom_active & vertical_top_active
    open_core = connected_open_core & (
        v_topological_void
        >= params.vertical_gas_core_area_fraction * params.vertical_area
    )
    v_liquid_pressure_coupled = vertical_active & ~open_core

    h_mass_per_length = hm / dx
    h_momentum_per_length = hj / dx
    v_mass_per_length = vm / dz
    v_momentum_per_length = vj / dz
    v_tracer_per_length = vc / dz
    raw_rhs = _network_rhs(
        h_mass_per_length,
        h_momentum_per_length,
        v_mass_per_length,
        v_momentum_per_length,
        v_tracer_per_length,
        h_ag,
        h_depth,
        h_pg,
        h_dh,
        v_ag,
        v_pg,
        v_dh,
        h_faces,
        v_faces,
        horizontal_active,
        vertical_active,
        v_liquid_pressure_coupled,
        float(dx),
        float(dz),
        jx,
        float(mouth_area),
        params.rho_l,
        params.gravity,
        params.gas_viscosity,
        params.sound_speed,
        params.rho_atmospheric,
        params.limiter_theta,
        params.entropy_fix_fraction,
        params.resolved_density_fraction,
        params.resolved_density_ceiling,
    )
    d_hm = np.asarray(raw_rhs[0]) * dx
    d_hj = np.asarray(raw_rhs[1]) * dx
    d_vm = np.asarray(raw_rhs[2]) * dz
    d_vj = np.asarray(raw_rhs[3]) * dz
    d_vc = np.asarray(raw_rhs[4]) * dz

    horizontal_drag_j, horizontal_drag_q, horizontal_drag_residual = (
        _interphase_drag_rhs(
            hm,
            hj,
            h_effective_liquid,
            h_ql,
            h_ag,
            h_pi,
            h_dh,
            horizontal_active,
            cell_width=dx,
            rho_l=params.rho_l,
            gas_viscosity=params.gas_viscosity,
            liquid_holdup_drag_enhancement=(
                params.horizontal_holdup_drag_enhancement
            ),
        )
    )
    if params.vertical_confined_interface_kinematics:
        vertical_drag_j = np.zeros_like(vj)
        vertical_drag_q = np.zeros_like(v_ql)
        vertical_drag_residual = 0.0
    else:
        vertical_drag_j, vertical_drag_q, vertical_drag_residual = (
            _interphase_drag_rhs(
                vm,
                vj,
                v_effective_liquid,
                v_ql,
                v_ag,
                v_pi,
                v_dh,
                vertical_active,
                cell_width=dz,
                rho_l=params.rho_l,
                gas_viscosity=params.gas_viscosity,
            )
        )
    d_hj = d_hj + horizontal_drag_j
    d_vj = d_vj + vertical_drag_j

    # Re-evaluate the two reported Riemann momentum fluxes with exactly the
    # same cell traces used by ``_network_rhs``.  Their mass/tracer rates are
    # taken directly from that residual to make the ledger identity literal.
    h_rho_j = max(h_mass_per_length[jx] / h_ag[jx], 1.0e-10)
    h_u_j = _state_velocity(
        h_mass_per_length[jx],
        h_momentum_per_length[jx],
        h_ag[jx],
        params,
    )
    v_rho_0 = max(v_mass_per_length[0] / v_ag[0], 1.0e-10)
    v_u_0 = _state_velocity(
        v_mass_per_length[0],
        v_momentum_per_length[0],
        v_ag[0],
        params,
    )
    t_momentum_flux = 0.0
    if mouth_area > 0.0:
        _, t_momentum_per_area = _gas_flux(
            h_rho_j,
            0.0,
            v_rho_0,
            v_u_0,
            params.sound_speed,
            params.rho_atmospheric,
            params.entropy_fix_fraction,
            params.resolved_density_fraction,
            params.resolved_density_ceiling,
        )
        t_momentum_flux = t_momentum_per_area * mouth_area
    t_mass_rate = float(raw_rhs[7])
    if t_mass_rate >= 0.0:
        t_tracer_rate = t_mass_rate
    else:
        concentration = vc[0] / max(vm[0], 1.0e-14)
        t_tracer_rate = t_mass_rate * min(max(concentration, 0.0), 1.0)
    t_flux = InternalTGasFlux(
        area=float(mouth_area),
        mass_rate_horizontal_to_vertical=t_mass_rate,
        normal_momentum_flux=float(t_momentum_flux),
        tracer_mass_rate_horizontal_to_vertical=float(t_tracer_rate),
        horizontal_mass_rate=-t_mass_rate,
        vertical_mass_rate=t_mass_rate,
        internal_mass_residual=(-t_mass_rate + t_mass_rate),
    )

    top_mass_rate = float(raw_rhs[5])
    top_tracer_rate = float(raw_rhs[6])
    top_momentum_flux = 0.0
    if v_faces[-1] > 0.0:
        v_rho_top = max(v_mass_per_length[-1] / v_ag[-1], 1.0e-10)
        v_u_top = _state_velocity(
            v_mass_per_length[-1],
            v_momentum_per_length[-1],
            v_ag[-1],
            params,
        )
        _, top_momentum_per_area = _gas_flux(
            v_rho_top,
            v_u_top,
            params.rho_atmospheric,
            0.0,
            params.sound_speed,
            params.rho_atmospheric,
            params.entropy_fix_fraction,
            params.resolved_density_fraction,
            params.resolved_density_ceiling,
        )
        top_momentum_flux = top_momentum_per_area * v_faces[-1]
    top_flux = ExternalGasFaceFlux(
        area=float(v_faces[-1]),
        mass_rate=top_mass_rate,
        momentum_flux=float(top_momentum_flux),
        tracer_mass_rate=top_tracer_rate,
    )

    # The east front is tied to the same conservative Riemann face that fills
    # its leading geometric cut cell.  No east opening means no FV face area,
    # no gas-mass transfer, and therefore exactly zero metadata speed.
    east_flux = (
        _east_front_riemann_flux(
            h_mass_per_length,
            h_momentum_per_length,
            h_ag,
            h_faces,
            horizontal_active,
            east_active,
            junction_index=jx,
            params=params,
        )
        if bool(east_front.active)
        else _closed_east_front_flux()
    )
    if east_flux.area > 0.0:
        east_kinematics = _kinematics_from_mass_flux(
            branch="east",
            position=east_front.position,
            mass_rate=east_flux.mass_rate_west_to_east,
            area=east_flux.area,
            density_positive=east_flux.upwind_density,
            density_negative=east_flux.upwind_density,
            source_face="east_material_front_riemann",
        )
    else:
        east_kinematics = _kinematics_from_mass_flux(
            branch="east",
            position=east_front.position,
            mass_rate=0.0,
            area=0.0,
            density_positive=0.0,
            density_negative=0.0,
            source_face="closed",
        )

    # At zero vertical length there is no receiver state from which to invent a
    # velocity.  The paired T Riemann mass flux and its actual mouth area give
    # the unique conservative initial speed.  Thereafter the resolved gas trace
    # at the moving material front supplies the same m_dot/(rho A) identity.
    bottom_indices = np.flatnonzero(vertical_bottom_active)
    if vertical_front.active and vertical_front.position == 0.0:
        vertical_kinematics = _kinematics_from_mass_flux(
            branch="vertical",
            position=vertical_front.position,
            mass_rate=t_mass_rate,
            area=float(mouth_area),
            density_positive=h_rho_j,
            density_negative=v_rho_0,
            source_face="t_riemann",
        )
    elif vertical_front.active and bottom_indices.size:
        vertical_index = int(bottom_indices[-1])
        vertical_area = float(v_ag[vertical_index])
        vertical_density = max(
            v_mass_per_length[vertical_index] / vertical_area, 1.0e-10
        )
        vertical_mass_rate = float(v_momentum_per_length[vertical_index])
        vertical_kinematics = _kinematics_from_mass_flux(
            branch="vertical",
            position=vertical_front.position,
            mass_rate=vertical_mass_rate,
            area=vertical_area,
            density_positive=vertical_density,
            density_negative=vertical_density,
            source_face="vertical_material_front",
        )
    else:
        vertical_kinematics = _kinematics_from_mass_flux(
            branch="vertical",
            position=vertical_front.position,
            mass_rate=0.0,
            area=0.0,
            density_positive=0.0,
            density_negative=0.0,
            source_face="closed",
        )

    total_ledger = float(
        np.sum(d_hm, dtype=np.float64)
        + np.sum(d_vm, dtype=np.float64)
        + top_mass_rate
    )
    tracer_ledger = float(
        np.sum(d_hm, dtype=np.float64)
        + np.sum(d_vc, dtype=np.float64)
        + top_tracer_rate
    )
    return ResolvedGasStageRHS(
        dMgt_dt=_readonly(d_hm),
        dJgt_dt=_readonly(d_hj),
        dMgr_dt=_readonly(d_vm),
        dJgrs_dt=_readonly(d_vj),
        dMgrs_dt=_readonly(d_vc),
        dJgt_drag_dt=_readonly(horizontal_drag_j),
        dJgrs_drag_dt=_readonly(vertical_drag_j),
        dQlt_drag_dt=_readonly(horizontal_drag_q),
        dQlr_drag_dt=_readonly(vertical_drag_q),
        top_flux=top_flux,
        t_flux=t_flux,
        east_flux=east_flux,
        east_front=east_kinematics,
        vertical_front=vertical_kinematics,
        horizontal_active=_readonly(horizontal_active),
        vertical_bottom_active=_readonly(vertical_bottom_active),
        vertical_top_active=_readonly(vertical_top_active),
        total_gas_mass_ledger_residual=total_ledger,
        tracer_mass_ledger_residual=tracer_ledger,
        horizontal_interphase_momentum_residual=horizontal_drag_residual,
        vertical_interphase_momentum_residual=vertical_drag_residual,
    )


__all__ = [
    "ExternalGasFaceFlux",
    "InternalEastFrontGasFlux",
    "InternalTGasFlux",
    "MaterialFrontRHS",
    "ResolvedGasStageRHS",
    "evaluate_resolved_gas_stage_rhs",
]
