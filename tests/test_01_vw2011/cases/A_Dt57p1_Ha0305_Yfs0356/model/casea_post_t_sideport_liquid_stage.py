"""Conservative finite-width liquid coupling for the Case-A side port.

The ventilation tower in the Vasconcelos--Wright apparatus is connected
through a finite circular opening.  It is not a zero-volume, three-branch
mathematical point.  After the fitted horizontal gas front reaches that
opening, the horizontal finite-volume pipe therefore remains continuous and
the tower is coupled as a lateral face of finite width.

For a vertical volume flux ``q_v`` (positive upward), the same flux is used
on both sides of the coupling,

    integral_S dA_l/dt dx = -q_v,
    F_A,vertical(0) = q_v.

``S`` is the measured circular tower mouth.  Its cell weights are exact
integrals of circular chords, so they converge geometrically and contain no
chosen footprint or response shape.  Liquid leaving the horizontal pipe
carries its resolved axial momentum.  Liquid entering from the tower has no
incoming horizontal momentum; the ninety-degree turn is a pipe-wall reaction,
not an imposed pair of waves.

The vertical face state is the incoming vertical liquid characteristic
evaluated at the area-weighted horizontal mouth pressure.  There is no flow
cap, target waveform, filtering, receiver remap, or post-step assignment in
this module.  If an explicit step leaves the positive state set, the outer
integrator must reduce its timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from casea_post_t_liquid_stage import (
    BranchPressureEvaluation,
    PressureCallback,
    _as_state_array,
    _gauge_consistent_node_momentum_flux,
    _ordinary_branch_fluxes,
    _potential_reference_pressure,
    _validated_pressure_evaluation,
    _vertical_active_count,
)
from casea_tjunction_shock_network import LiquidCharacteristic


@dataclass(frozen=True)
class PostTSidePortGeometry:
    """Measured geometry and material constants of the lateral tower mouth."""

    horizontal_cell_width: float
    vertical_cell_width: float
    liquid_density: float
    junction_center_x: float
    opening_diameter: float
    atmospheric_pressure_abs: float = 101_325.0
    vertical_loss_coefficient: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.horizontal_cell_width,
            self.vertical_cell_width,
            self.liquid_density,
            self.opening_diameter,
            self.atmospheric_pressure_abs,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive finite side-port geometry required")
        if not math.isfinite(self.junction_center_x):
            raise ValueError("finite side-port centre required")
        if (
            not math.isfinite(self.vertical_loss_coefficient)
            or self.vertical_loss_coefficient < 0.0
        ):
            raise ValueError("finite non-negative vertical loss required")


@dataclass(frozen=True)
class SidePortDiagnostics:
    """Physical face data and exact semi-discrete conservation residuals."""

    opening_weights: np.ndarray
    junction_pressure_abs: float
    vertical_volume_flux: float
    vertical_face_velocity: float
    vertical_face_pressure_abs: float
    vertical_base_momentum_flux: float
    vertical_top_momentum_flux: float
    horizontal_volume_source_integral: float
    vertical_volume_source_integral: float
    liquid_volume_residual: float
    horizontal_wall_momentum_reaction_rate: float
    vertical_active_count: int


@dataclass(frozen=True)
class PostTSidePortLiquidStageRhs:
    rhs_horizontal_area: np.ndarray
    rhs_horizontal_discharge: np.ndarray
    rhs_vertical_area: np.ndarray
    rhs_vertical_discharge: np.ndarray
    diagnostics: SidePortDiagnostics


def _circle_primitive(x: np.ndarray, radius: float) -> np.ndarray:
    """Primitive of ``2 sqrt(R**2-x**2)`` on ``[-R,R]``."""

    local = np.clip(np.asarray(x, dtype=float), -radius, radius)
    root = np.sqrt(np.maximum(radius * radius - local * local, 0.0))
    return local * root + radius * radius * np.arcsin(local / radius)


def circular_side_port_weights(
    ncell: int,
    *,
    cell_width: float,
    centre_x: float,
    diameter: float,
) -> np.ndarray:
    """Return exact circular-mouth area fractions owned by horizontal cells.

    The horizontal coordinate cuts the tower mouth into circular chords.  A
    cell weight is its exact chord-area integral divided by ``pi R**2``.
    Hence the weights sum to one whenever the complete opening lies inside the
    horizontal domain, independently of grid spacing.
    """

    if isinstance(ncell, bool) or int(ncell) != ncell or ncell < 1:
        raise ValueError("ncell must be a positive integer")
    if not all(
        math.isfinite(value) for value in (cell_width, centre_x, diameter)
    ):
        raise ValueError("finite circular side-port geometry required")
    if cell_width <= 0.0 or diameter <= 0.0:
        raise ValueError("positive side-port width scales required")
    length = int(ncell) * cell_width
    radius = 0.5 * diameter
    if centre_x - radius < 0.0 or centre_x + radius > length:
        raise ValueError("the complete circular side port must lie in the pipe")

    left = np.arange(int(ncell), dtype=float) * cell_width - centre_x
    right = left + cell_width
    clipped_left = np.clip(left, -radius, radius)
    clipped_right = np.clip(right, -radius, radius)
    active = (right > -radius) & (left < radius)
    weights = np.zeros(int(ncell), dtype=float)
    weights[active] = (
        _circle_primitive(clipped_right[active], radius)
        - _circle_primitive(clipped_left[active], radius)
    ) / (math.pi * radius * radius)
    # The analytic integrals sum to one to roundoff.  Do not renormalise: a
    # non-unit sum would expose a geometry/discretisation defect instead of
    # silently prescribing a different source footprint.
    tolerance = 128.0 * np.finfo(float).eps * max(1, int(ncell))
    if not math.isclose(
        float(np.sum(weights, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise FloatingPointError("circular side-port quadrature lost mouth area")
    weights.setflags(write=False)
    return weights


def _top_atmospheric_momentum_flux(
    vertical_area: np.ndarray,
    evaluation: BranchPressureEvaluation,
    *,
    atmospheric_pressure_abs: float,
    liquid_density: float,
) -> float:
    top = vertical_area.size - 1
    reference_pressure = float(_potential_reference_pressure(evaluation)[top])
    return float(
        evaluation.pressure.potential[top]
        + (atmospheric_pressure_abs - reference_pressure)
        * vertical_area[top]
        / liquid_density
    )


def post_t_sideport_liquid_stage_rhs(
    horizontal_area: object,
    horizontal_discharge: object,
    vertical_area: object,
    vertical_discharge: object,
    horizontal_gas_mass: object,
    horizontal_gas_momentum: object,
    vertical_gas_mass: object,
    vertical_gas_momentum: object,
    *,
    geometry: PostTSidePortGeometry,
    pressure_callback: PressureCallback,
    vertical_active_mask: object | None = None,
    vertical_active_count: int | None = None,
) -> PostTSidePortLiquidStageRhs:
    """Evaluate one conservative liquid RHS for the finite tower side port."""

    ah = _as_state_array(horizontal_area, "horizontal_area")
    qh = _as_state_array(horizontal_discharge, "horizontal_discharge")
    av = _as_state_array(vertical_area, "vertical_area")
    qv = _as_state_array(vertical_discharge, "vertical_discharge")
    mgh = _as_state_array(horizontal_gas_mass, "horizontal_gas_mass")
    jgh = _as_state_array(horizontal_gas_momentum, "horizontal_gas_momentum")
    mgv = _as_state_array(vertical_gas_mass, "vertical_gas_mass")
    jgv = _as_state_array(vertical_gas_momentum, "vertical_gas_momentum")
    if not (ah.shape == qh.shape == mgh.shape == jgh.shape):
        raise ValueError("horizontal side-port fields must share one grid")
    if not (av.shape == qv.shape == mgv.shape == jgv.shape):
        raise ValueError("vertical side-port fields must share one grid")
    active_count = _vertical_active_count(
        av,
        vertical_active_mask=vertical_active_mask,
        vertical_active_count=vertical_active_count,
    )
    if np.any(ah <= 0.0) or np.any(av[:active_count] <= 0.0):
        raise ValueError("active liquid areas must remain positive")
    if np.any(av[active_count:] < 0.0):
        raise ValueError("inactive vertical liquid area cannot be negative")

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
    weights = circular_side_port_weights(
        ah.size,
        cell_width=geometry.horizontal_cell_width,
        centre_x=geometry.junction_center_x,
        diameter=geometry.opening_diameter,
    )
    p_junction = float(np.dot(weights, ph.face_pressure_abs))
    vertical_characteristic = LiquidCharacteristic(
        reference_pressure_abs=float(pv.face_pressure_abs[0]),
        reference_outward_velocity=float(qv[0] / av[0]),
        wave_speed=float(pv.pressure.celerity[0]),
        loss_coefficient=geometry.vertical_loss_coefficient,
        pressure_offset=float(pv.node_pressure_offset[0]),
    )
    vertical_velocity = vertical_characteristic.outward_velocity(
        p_junction,
        liquid_density=geometry.liquid_density,
    )
    vertical_flux = float(av[0] * vertical_velocity)
    vertical_face_pressure = float(
        p_junction + pv.node_pressure_offset[0]
    )
    vertical_momentum_flux, _ = _gauge_consistent_node_momentum_flux(
        node_volume_flux=vertical_flux,
        node_area=float(av[0]),
        node_face_pressure_abs=vertical_face_pressure,
        reference_face_pressure_abs=float(
            _potential_reference_pressure(pv)[0]
        ),
        reference_potential=float(pv.pressure.potential[0]),
        liquid_density=geometry.liquid_density,
    )

    fh = _ordinary_branch_fluxes(ah, qh, ph.pressure)
    fv = _ordinary_branch_fluxes(
        av[:active_count], qv[:active_count], pv.pressure
    )
    fv[0] = (vertical_flux, vertical_momentum_flux)
    top_momentum_flux = _top_atmospheric_momentum_flux(
        av[:active_count],
        pv,
        atmospheric_pressure_abs=geometry.atmospheric_pressure_abs,
        liquid_density=geometry.liquid_density,
    )
    fv[-1] = (0.0, top_momentum_flux)

    horizontal_divergence = -(
        fh[1:] - fh[:-1]
    ) / geometry.horizontal_cell_width
    rhs_ah = horizontal_divergence[:, 0]
    rhs_qh = horizontal_divergence[:, 1]
    side_area_source = (
        -vertical_flux * weights / geometry.horizontal_cell_width
    )
    rhs_ah = rhs_ah + side_area_source

    # Only outward liquid transports pre-existing horizontal axial momentum.
    # Orthogonal inflow brings zero x momentum; subsequent pressure fluxes
    # determine its horizontal acceleration without prescribing a split wave.
    wall_momentum_reaction = 0.0
    if vertical_flux > 0.0:
        local_velocity = qh / ah
        side_momentum_source = side_area_source * local_velocity
        rhs_qh = rhs_qh + side_momentum_source
        wall_momentum_reaction = float(
            np.sum(side_momentum_source, dtype=np.float64)
            * geometry.horizontal_cell_width
        )
    rhs_qh = rhs_qh + np.asarray(ph.momentum_source, dtype=float)

    vertical_divergence = -(
        fv[1:] - fv[:-1]
    ) / geometry.vertical_cell_width
    rhs_av = np.zeros_like(av)
    rhs_qv = np.zeros_like(qv)
    rhs_av[:active_count] = vertical_divergence[:, 0]
    rhs_qv[:active_count] = (
        vertical_divergence[:, 1]
        + np.asarray(pv.momentum_source, dtype=float)
    )

    horizontal_volume = float(
        np.sum(side_area_source, dtype=np.float64)
        * geometry.horizontal_cell_width
    )
    vertical_volume = float(vertical_flux)
    residual = horizontal_volume + vertical_volume
    scale = max(abs(horizontal_volume), abs(vertical_volume), 1.0e-30)
    if abs(residual) > 512.0 * np.finfo(float).eps * scale:
        raise FloatingPointError("finite-width side port lost liquid volume")
    diagnostics = SidePortDiagnostics(
        opening_weights=weights,
        junction_pressure_abs=p_junction,
        vertical_volume_flux=vertical_flux,
        vertical_face_velocity=float(vertical_velocity),
        vertical_face_pressure_abs=vertical_face_pressure,
        vertical_base_momentum_flux=float(vertical_momentum_flux),
        vertical_top_momentum_flux=float(top_momentum_flux),
        horizontal_volume_source_integral=horizontal_volume,
        vertical_volume_source_integral=vertical_volume,
        liquid_volume_residual=float(residual),
        horizontal_wall_momentum_reaction_rate=wall_momentum_reaction,
        vertical_active_count=active_count,
    )
    return PostTSidePortLiquidStageRhs(
        rhs_horizontal_area=np.asarray(rhs_ah, dtype=float),
        rhs_horizontal_discharge=np.asarray(rhs_qh, dtype=float),
        rhs_vertical_area=rhs_av,
        rhs_vertical_discharge=rhs_qv,
        diagnostics=diagnostics,
    )


__all__ = [
    "PostTSidePortGeometry",
    "PostTSidePortLiquidStageRhs",
    "SidePortDiagnostics",
    "circular_side_port_weights",
    "post_t_sideport_liquid_stage_rhs",
]
