"""Pure conservative liquid stage for the Case-A post-arrival T network.

The horizontal pipe is split at a *face* into independent west and east
finite-volume branches.  The base of the vertical branch is the third face of
the same massless node.  A single zero-storage solve supplies all three face
volume fluxes.  Those face fluxes replace (rather than supplement) the three
ordinary numerical fluxes, so there is no side source and no ``mean +/- q/2``
partition.

The momentum face flux needs slightly more care than the volume flux.  The
barotropic horizontal operator stores a conservative pressure potential
``Psi``; it does not store the enormous, arbitrary ``p_abs*A/rho`` datum.  At
the node we therefore retain the adjacent cell's ``Psi`` and add only the
resolved pressure change

    Psi_J = Psi_* + (p_l,J - p_l,*) A_J / rho_l.

This puts the node and every interior Riemann face on the same gauge while
still carrying the complete advective and pressure-potential momentum flux.
In particular, a static uniform state has no pressure impulse at the tee.

Gas mass and momentum are frozen inputs to one liquid SSP-RK stage.  They are
never changed here; ownership of the gas EOS/front update belongs to the
outer coupled network solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import numpy as np

from casea_horizontal_liquid_operator import (
    PressurePotentialState,
    physical_liquid_flux,
    rusanov_face_flux,
)
from casea_tjunction_shock_network import (
    LiquidCharacteristic,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    ZeroStorageTNodeSolution,
    evaluate_zero_storage_t_node_at_pressure,
    solve_zero_storage_t_node,
)


BranchName = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class BranchPressureEvaluation:
    """Pressure-law data evaluated from one complete branch stage state.

    ``pressure`` is the conservative potential/Jacobian used at ordinary
    Riemann faces.  ``face_pressure_abs`` is the corresponding local liquid
    pressure datum used by the incoming node characteristic.
    ``node_pressure_offset`` maps the common node pressure to that datum,
    ``p_l,J = p_J + offset`` (for example a hydrostatic centroid offset).
    ``potential_pressure_abs`` identifies the absolute pressure datum that
    corresponds to the cell-centred conservative potential.  It may differ
    from the characteristic's boundary-face datum in a hydrostatic vertical
    cell; when omitted it is identical to ``face_pressure_abs``.
    ``momentum_source`` contains only a physical cell-volume source such as
    vertical gravity; a T-junction exchange is not permitted here.
    """

    pressure: PressurePotentialState
    face_pressure_abs: np.ndarray
    node_pressure_offset: np.ndarray
    momentum_source: np.ndarray
    potential_pressure_abs: np.ndarray | None = None


PressureCallback = Callable[
    [
        BranchName,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    BranchPressureEvaluation,
]


@dataclass(frozen=True)
class PostTLiquidGeometry:
    """Cell geometry and physical data for the massless T-node stage."""

    junction_face_index: int
    horizontal_cell_width: float
    vertical_cell_width: float
    liquid_density: float
    west_loss_coefficient: float = 0.0
    east_loss_coefficient: float = 0.0
    vertical_loss_coefficient: float = 0.0
    atmospheric_pressure_abs: float = 101_325.0
    node_volume_flux_tolerance: float = 1.0e-12
    node_pressure_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        positive = (
            self.horizontal_cell_width,
            self.vertical_cell_width,
            self.liquid_density,
            self.atmospheric_pressure_abs,
            self.node_volume_flux_tolerance,
            self.node_pressure_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive finite post-T liquid geometry required")
        losses = (
            self.west_loss_coefficient,
            self.east_loss_coefficient,
            self.vertical_loss_coefficient,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in losses):
            raise ValueError("finite non-negative T-node losses required")


@dataclass(frozen=True)
class NodeFluxDiagnostics:
    """Fluxes actually inserted into the three finite-volume faces."""

    solution: ZeroStorageTNodeSolution
    coordinate_volume_fluxes: Mapping[str, float]
    coordinate_momentum_fluxes: Mapping[str, float]
    pressure_potentials: Mapping[str, float]
    momentum_flux_changes: Mapping[str, float]
    node_volume_residual: float
    node_mass_residual: float
    vertical_top_volume_flux: float
    vertical_top_momentum_flux: float
    vertical_active_count: int


@dataclass(frozen=True)
class PostTLiquidStageRhs:
    rhs_horizontal_area: np.ndarray
    rhs_horizontal_discharge: np.ndarray
    rhs_vertical_area: np.ndarray
    rhs_vertical_discharge: np.ndarray
    diagnostics: NodeFluxDiagnostics


@dataclass(frozen=True)
class PostTLiquidAdvance:
    horizontal_area: np.ndarray
    horizontal_discharge: np.ndarray
    vertical_area: np.ndarray
    vertical_discharge: np.ndarray
    first_stage: NodeFluxDiagnostics
    second_stage: NodeFluxDiagnostics
    liquid_volume_change: float
    node_volume_integral: float
    conservation_error: float


def _as_state_array(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _pressure_slice(
    pressure: PressurePotentialState,
    selection: object,
) -> PressurePotentialState:
    """Slice every field of a pressure evaluation without changing its law."""

    return PressurePotentialState(
        potential=np.asarray(pressure.potential)[selection],
        derivative=np.asarray(pressure.derivative)[selection],
        discharge_derivative=np.asarray(pressure.discharge_derivative)[selection],
        celerity=np.asarray(pressure.celerity)[selection],
        eigenvalue_minus=np.asarray(pressure.eigenvalue_minus)[selection],
        eigenvalue_plus=np.asarray(pressure.eigenvalue_plus)[selection],
        lambda_value=np.asarray(pressure.lambda_value)[selection],
        lambda_derivative=np.asarray(pressure.lambda_derivative)[selection],
        stratified=np.asarray(pressure.stratified)[selection],
    )


def _validated_pressure_evaluation(
    evaluation: BranchPressureEvaluation,
    size: int,
    name: str,
) -> BranchPressureEvaluation:
    arrays = {
        "potential": np.asarray(evaluation.pressure.potential, dtype=float),
        "derivative": np.asarray(evaluation.pressure.derivative, dtype=float),
        "celerity": np.asarray(evaluation.pressure.celerity, dtype=float),
        "face_pressure_abs": np.asarray(
            evaluation.face_pressure_abs, dtype=float
        ),
        "node_pressure_offset": np.asarray(
            evaluation.node_pressure_offset, dtype=float
        ),
        "momentum_source": np.asarray(evaluation.momentum_source, dtype=float),
    }
    if evaluation.potential_pressure_abs is not None:
        arrays["potential_pressure_abs"] = np.asarray(
            evaluation.potential_pressure_abs, dtype=float
        )
    for field_name, field in arrays.items():
        if field.shape != (size,):
            raise ValueError(
                f"{name} {field_name} must have shape ({size},), got {field.shape}"
            )
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} {field_name} must be finite")
    if np.any(arrays["derivative"] < 0.0):
        raise ValueError(f"{name} pressure derivative must be non-negative")
    if np.any(arrays["face_pressure_abs"] <= 0.0):
        raise ValueError(f"{name} absolute liquid pressure must be positive")
    if (
        "potential_pressure_abs" in arrays
        and np.any(arrays["potential_pressure_abs"] <= 0.0)
    ):
        raise ValueError(
            f"{name} potential-reference pressure must be positive"
        )
    return evaluation


def _potential_reference_pressure(
    evaluation: BranchPressureEvaluation,
) -> np.ndarray:
    """Pressure datum corresponding to ``pressure.potential`` in each cell."""

    if evaluation.potential_pressure_abs is None:
        return np.asarray(evaluation.face_pressure_abs, dtype=float)
    return np.asarray(evaluation.potential_pressure_abs, dtype=float)


def _vertical_active_count(
    vertical_area: np.ndarray,
    *,
    vertical_active_mask: object | None,
    vertical_active_count: int | None,
) -> int:
    """Resolve one contiguous wet/two-phase prefix from the T upward."""

    if vertical_active_mask is not None and vertical_active_count is not None:
        raise ValueError("pass vertical_active_mask or vertical_active_count, not both")
    if vertical_active_mask is not None:
        mask = np.asarray(vertical_active_mask, dtype=bool)
        if mask.shape != vertical_area.shape:
            raise ValueError("vertical_active_mask must match the vertical state")
        inactive = np.flatnonzero(~mask)
        count = int(inactive[0]) if inactive.size else int(mask.size)
        expected = np.arange(mask.size) < count
        if not np.array_equal(mask, expected):
            raise ValueError(
                "vertical_active_mask must be one contiguous prefix from the T"
            )
    elif vertical_active_count is not None:
        if isinstance(vertical_active_count, bool):
            raise ValueError("vertical_active_count must be an integer")
        count = int(vertical_active_count)
        if count != vertical_active_count:
            raise ValueError("vertical_active_count must be an integer")
    else:
        count = int(vertical_area.size)
    if count < 1 or count > vertical_area.size:
        raise ValueError("at least one and at most all vertical cells must be active")
    return count


def _ordinary_branch_fluxes(
    area: np.ndarray,
    discharge: np.ndarray,
    pressure: PressurePotentialState,
) -> np.ndarray:
    """Return internal Rusanov faces; outer faces are reflecting walls."""

    ncell = area.size
    flux = np.zeros((ncell + 1, 2), dtype=float)
    physical = np.asarray(
        physical_liquid_flux(area, discharge, pressure), dtype=float
    )
    # Exact reflecting-wall flux: no liquid crosses the exterior boundary,
    # while the complete local normal momentum tensor remains on the face.
    flux[0, 1] = physical[0, 1]
    flux[-1, 1] = physical[-1, 1]
    if ncell > 1:
        internal, _ = rusanov_face_flux(
            area[:-1],
            discharge[:-1],
            _pressure_slice(pressure, slice(None, -1)),
            area[1:],
            discharge[1:],
            _pressure_slice(pressure, slice(1, None)),
        )
        flux[1:-1] = internal
    return flux


def _node_characteristic(
    *,
    area: float,
    discharge: float,
    face_pressure_abs: float,
    node_pressure_offset: float,
    celerity: float,
    outward_sign: float,
    loss_coefficient: float,
) -> LiquidCharacteristic:
    if area <= 0.0:
        raise ValueError("the three T-adjacent liquid traces must be wet")
    if celerity <= 0.0:
        raise ValueError("positive T-adjacent liquid celerity required")
    return LiquidCharacteristic(
        reference_pressure_abs=float(face_pressure_abs),
        reference_outward_velocity=float(outward_sign * discharge / area),
        wave_speed=float(celerity),
        loss_coefficient=float(loss_coefficient),
        pressure_offset=float(node_pressure_offset),
    )


def _gauge_consistent_node_momentum_flux(
    *,
    node_volume_flux: float,
    node_area: float,
    node_face_pressure_abs: float,
    reference_face_pressure_abs: float,
    reference_potential: float,
    liquid_density: float,
) -> tuple[float, float]:
    """Return ``q^2/A + Psi_J`` in the adjacent branch's potential gauge."""

    potential = (
        reference_potential
        + (node_face_pressure_abs - reference_face_pressure_abs)
        * node_area
        / liquid_density
    )
    momentum_flux = node_volume_flux**2 / node_area + potential
    return float(momentum_flux), float(potential)


def post_t_liquid_stage_rhs(
    horizontal_area: object,
    horizontal_discharge: object,
    vertical_area: object,
    vertical_discharge: object,
    horizontal_gas_mass: object,
    horizontal_gas_momentum: object,
    vertical_gas_mass: object,
    vertical_gas_momentum: object,
    *,
    geometry: PostTLiquidGeometry,
    pressure_callback: PressureCallback,
    vertical_active_mask: object | None = None,
    vertical_active_count: int | None = None,
) -> PostTLiquidStageRhs:
    """Evaluate one spatial liquid stage with one conservative T-node solve."""

    ah = _as_state_array(horizontal_area, "horizontal_area")
    qh = _as_state_array(horizontal_discharge, "horizontal_discharge")
    av = _as_state_array(vertical_area, "vertical_area")
    qv = _as_state_array(vertical_discharge, "vertical_discharge")
    mgh = _as_state_array(horizontal_gas_mass, "horizontal_gas_mass")
    jgh = _as_state_array(horizontal_gas_momentum, "horizontal_gas_momentum")
    mgv = _as_state_array(vertical_gas_mass, "vertical_gas_mass")
    jgv = _as_state_array(vertical_gas_momentum, "vertical_gas_momentum")
    if not (ah.shape == qh.shape == mgh.shape == jgh.shape):
        raise ValueError("horizontal liquid/gas stage arrays must have one shape")
    if not (av.shape == qv.shape == mgv.shape == jgv.shape):
        raise ValueError("vertical liquid/gas stage arrays must have one shape")
    active_count = _vertical_active_count(
        av,
        vertical_active_mask=vertical_active_mask,
        vertical_active_count=vertical_active_count,
    )
    if np.any(ah <= 0.0) or np.any(av[:active_count] <= 0.0):
        raise ValueError(
            "the horizontal branch and active vertical prefix require positive "
            "liquid areas"
        )
    if np.any(av[active_count:] < 0.0):
        raise ValueError("dry-suffix liquid areas cannot be negative")

    node_face = int(geometry.junction_face_index)
    if node_face <= 0 or node_face >= ah.size:
        raise ValueError(
            "junction_face_index must split at least one west and one east cell"
        )
    iw = node_face - 1
    ie = node_face

    ph = _validated_pressure_evaluation(
        pressure_callback(
            "horizontal", ah.copy(), qh.copy(), mgh.copy(), jgh.copy()
        ),
        ah.size,
        "horizontal",
    )
    pv = _validated_pressure_evaluation(
        pressure_callback(
            "vertical",
            av[:active_count].copy(),
            qv[:active_count].copy(),
            mgv[:active_count].copy(),
            jgv[:active_count].copy(),
        ),
        active_count,
        "vertical",
    )

    west = _node_characteristic(
        area=ah[iw],
        discharge=qh[iw],
        face_pressure_abs=np.asarray(ph.face_pressure_abs)[iw],
        node_pressure_offset=np.asarray(ph.node_pressure_offset)[iw],
        celerity=np.asarray(ph.pressure.celerity)[iw],
        outward_sign=-1.0,
        loss_coefficient=geometry.west_loss_coefficient,
    )
    east = _node_characteristic(
        area=ah[ie],
        discharge=qh[ie],
        face_pressure_abs=np.asarray(ph.face_pressure_abs)[ie],
        node_pressure_offset=np.asarray(ph.node_pressure_offset)[ie],
        celerity=np.asarray(ph.pressure.celerity)[ie],
        outward_sign=1.0,
        loss_coefficient=geometry.east_loss_coefficient,
    )
    vertical = _node_characteristic(
        area=av[0],
        discharge=qv[0],
        face_pressure_abs=np.asarray(pv.face_pressure_abs)[0],
        node_pressure_offset=np.asarray(pv.node_pressure_offset)[0],
        celerity=np.asarray(pv.pressure.celerity)[0],
        outward_sign=1.0,
        loss_coefficient=geometry.vertical_loss_coefficient,
    )
    areas = ZeroStorageTBranchAreas(
        west=float(ah[iw]), east=float(ah[ie]), vertical=float(av[0])
    )
    characteristics = TeeLiquidCharacteristics(
        west=west,
        east=east,
        vertical=vertical,
        west_liquid_area=float(ah[iw]),
    )
    pressure_hint = float(
        np.mean(
            [
                ph.face_pressure_abs[iw] - ph.node_pressure_offset[iw],
                ph.face_pressure_abs[ie] - ph.node_pressure_offset[ie],
                pv.face_pressure_abs[0] - pv.node_pressure_offset[0],
            ]
        )
    )
    # Test the physical stage pressure first.  Besides avoiding unnecessary
    # bisection, this preserves an exactly represented hydrostatic equilibrium
    # bit-for-bit instead of returning a nearby pressure-tolerance root.
    hinted_node = evaluate_zero_storage_t_node_at_pressure(
        characteristics,
        areas,
        node_pressure_abs=pressure_hint,
        liquid_density=geometry.liquid_density,
    )
    if (
        abs(hinted_node.net_outward_volume_flux)
        <= geometry.node_volume_flux_tolerance
    ):
        node = hinted_node
    else:
        node = solve_zero_storage_t_node(
            characteristics,
            areas,
            liquid_density=geometry.liquid_density,
            pressure_hint_abs=pressure_hint,
            volume_flux_tolerance=geometry.node_volume_flux_tolerance,
            pressure_tolerance=geometry.node_pressure_tolerance,
        )

    fh = _ordinary_branch_fluxes(ah, qh, ph.pressure)
    fv = _ordinary_branch_fluxes(
        av[:active_count], qv[:active_count], pv.pressure
    )

    outward_flows = {
        name: float(node.branch_fluxes[name].volume_flux)
        for name in ("west", "east", "vertical")
    }
    # Coordinate-volume mapping required by the graph convention.
    coordinate_volume = {
        "west": -outward_flows["west"],
        "east": outward_flows["east"],
        "vertical": outward_flows["vertical"],
    }
    reference_data = {
        "west": (
            float(ah[iw]),
            float(_potential_reference_pressure(ph)[iw]),
            float(ph.pressure.potential[iw]),
        ),
        "east": (
            float(ah[ie]),
            float(_potential_reference_pressure(ph)[ie]),
            float(ph.pressure.potential[ie]),
        ),
        "vertical": (
            float(av[0]),
            float(_potential_reference_pressure(pv)[0]),
            float(pv.pressure.potential[0]),
        ),
    }
    coordinate_momentum: dict[str, float] = {}
    node_potential: dict[str, float] = {}
    momentum_change: dict[str, float] = {}
    for name in ("west", "east", "vertical"):
        area_node, reference_pressure, reference_potential = reference_data[name]
        momentum, potential = _gauge_consistent_node_momentum_flux(
            node_volume_flux=coordinate_volume[name],
            node_area=area_node,
            node_face_pressure_abs=node.branch_fluxes[name].face_pressure_abs,
            reference_face_pressure_abs=reference_pressure,
            reference_potential=reference_potential,
            liquid_density=geometry.liquid_density,
        )
        coordinate_momentum[name] = momentum
        node_potential[name] = potential
        reference_flux = (
            ({"west": qh[iw], "east": qh[ie], "vertical": qv[0]}[name]) ** 2
            / area_node
            + reference_potential
        )
        momentum_change[name] = float(momentum - reference_flux)

    # Delete the ordinary horizontal face at the T and replace it by the two
    # branch faces.  They occupy the same array slot but are applied to the two
    # disjoint branch divergences below.
    west_flux = fh[: node_face + 1].copy()
    east_flux = fh[node_face:].copy()
    west_flux[-1] = (
        coordinate_volume["west"],
        coordinate_momentum["west"],
    )
    east_flux[0] = (
        coordinate_volume["east"],
        coordinate_momentum["east"],
    )
    fv[0] = (
        coordinate_volume["vertical"],
        coordinate_momentum["vertical"],
    )
    # The upper end of the active prefix is a material liquid/gas interface,
    # not a solid wall.  No liquid crosses that moving interface, while its
    # normal stress is atmospheric.  The pressure correction is made in the
    # same potential gauge as all interior and T-node faces.
    top_index = active_count - 1
    top_reference_pressure = float(_potential_reference_pressure(pv)[top_index])
    top_potential = float(
        pv.pressure.potential[top_index]
        + (
            geometry.atmospheric_pressure_abs - top_reference_pressure
        )
        * av[top_index]
        / geometry.liquid_density
    )
    fv[-1] = (0.0, top_potential)

    rhs_ah = np.empty_like(ah)
    rhs_qh = np.empty_like(qh)
    west_divergence = -(
        west_flux[1:] - west_flux[:-1]
    ) / geometry.horizontal_cell_width
    east_divergence = -(
        east_flux[1:] - east_flux[:-1]
    ) / geometry.horizontal_cell_width
    rhs_ah[:node_face] = west_divergence[:, 0]
    rhs_qh[:node_face] = west_divergence[:, 1]
    rhs_ah[node_face:] = east_divergence[:, 0]
    rhs_qh[node_face:] = east_divergence[:, 1]
    vertical_divergence = -(
        fv[1:] - fv[:-1]
    ) / geometry.vertical_cell_width
    rhs_av = np.zeros_like(av)
    rhs_qv = np.zeros_like(qv)
    rhs_av[:active_count] = vertical_divergence[:, 0]
    rhs_qv[:active_count] = vertical_divergence[:, 1]
    rhs_qh += np.asarray(ph.momentum_source, dtype=float)
    rhs_qv[:active_count] += np.asarray(pv.momentum_source, dtype=float)

    diagnostics = NodeFluxDiagnostics(
        solution=node,
        coordinate_volume_fluxes=coordinate_volume,
        coordinate_momentum_fluxes=coordinate_momentum,
        pressure_potentials=node_potential,
        momentum_flux_changes=momentum_change,
        node_volume_residual=float(node.net_outward_volume_flux),
        node_mass_residual=float(node.net_outward_mass_flux),
        vertical_top_volume_flux=0.0,
        vertical_top_momentum_flux=top_potential,
        vertical_active_count=active_count,
    )
    return PostTLiquidStageRhs(
        rhs_horizontal_area=rhs_ah,
        rhs_horizontal_discharge=rhs_qh,
        rhs_vertical_area=rhs_av,
        rhs_vertical_discharge=rhs_qv,
        diagnostics=diagnostics,
    )


def advance_post_t_liquid_ssprk2(
    horizontal_area: object,
    horizontal_discharge: object,
    vertical_area: object,
    vertical_discharge: object,
    horizontal_gas_mass: object,
    horizontal_gas_momentum: object,
    vertical_gas_mass: object,
    vertical_gas_momentum: object,
    *,
    dt: float,
    geometry: PostTLiquidGeometry,
    pressure_callback: PressureCallback,
    vertical_active_mask: object | None = None,
    vertical_active_count: int | None = None,
) -> PostTLiquidAdvance:
    """Advance the four liquid fields with a fully recomputed SSP-RK2 node."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite SSP-RK2 time step required")
    ah0 = _as_state_array(horizontal_area, "horizontal_area")
    qh0 = _as_state_array(horizontal_discharge, "horizontal_discharge")
    av0 = _as_state_array(vertical_area, "vertical_area")
    qv0 = _as_state_array(vertical_discharge, "vertical_discharge")
    active_count = _vertical_active_count(
        av0,
        vertical_active_mask=vertical_active_mask,
        vertical_active_count=vertical_active_count,
    )

    stage0 = post_t_liquid_stage_rhs(
        ah0, qh0, av0, qv0,
        horizontal_gas_mass, horizontal_gas_momentum,
        vertical_gas_mass, vertical_gas_momentum,
        geometry=geometry,
        pressure_callback=pressure_callback,
        vertical_active_count=active_count,
    )
    ah1 = ah0 + dt * stage0.rhs_horizontal_area
    qh1 = qh0 + dt * stage0.rhs_horizontal_discharge
    av1 = av0 + dt * stage0.rhs_vertical_area
    qv1 = qv0 + dt * stage0.rhs_vertical_discharge
    if np.any(ah1 <= 0.0) or np.any(av1[:active_count] <= 0.0):
        raise FloatingPointError(
            "SSP-RK2 predictor left the positive active-liquid state set"
        )
    stage1 = post_t_liquid_stage_rhs(
        ah1, qh1, av1, qv1,
        horizontal_gas_mass, horizontal_gas_momentum,
        vertical_gas_mass, vertical_gas_momentum,
        geometry=geometry,
        pressure_callback=pressure_callback,
        vertical_active_count=active_count,
    )
    ah2 = 0.5 * ah0 + 0.5 * (
        ah1 + dt * stage1.rhs_horizontal_area
    )
    qh2 = 0.5 * qh0 + 0.5 * (
        qh1 + dt * stage1.rhs_horizontal_discharge
    )
    av2 = 0.5 * av0 + 0.5 * (
        av1 + dt * stage1.rhs_vertical_area
    )
    qv2 = 0.5 * qv0 + 0.5 * (
        qv1 + dt * stage1.rhs_vertical_discharge
    )
    fields = (ah2, qh2, av2, qv2)
    if not all(np.all(np.isfinite(field)) for field in fields):
        raise FloatingPointError("non-finite post-T SSP-RK2 liquid state")
    if np.any(ah2 <= 0.0) or np.any(av2[:active_count] <= 0.0):
        raise FloatingPointError(
            "SSP-RK2 corrector left the positive active-liquid state set"
        )

    initial_volume = (
        float(np.sum(ah0)) * geometry.horizontal_cell_width
        + float(np.sum(av0)) * geometry.vertical_cell_width
    )
    final_volume = (
        float(np.sum(ah2)) * geometry.horizontal_cell_width
        + float(np.sum(av2)) * geometry.vertical_cell_width
    )
    node_integral = 0.5 * dt * (
        stage0.diagnostics.node_volume_residual
        + stage1.diagnostics.node_volume_residual
    )
    change = final_volume - initial_volume
    return PostTLiquidAdvance(
        horizontal_area=ah2,
        horizontal_discharge=qh2,
        vertical_area=av2,
        vertical_discharge=qv2,
        first_stage=stage0.diagnostics,
        second_stage=stage1.diagnostics,
        liquid_volume_change=change,
        node_volume_integral=node_integral,
        conservation_error=change - node_integral,
    )


__all__ = [
    "BranchPressureEvaluation",
    "NodeFluxDiagnostics",
    "PostTLiquidAdvance",
    "PostTLiquidGeometry",
    "PostTLiquidStageRhs",
    "PressureCallback",
    "advance_post_t_liquid_ssprk2",
    "post_t_liquid_stage_rhs",
]
