"""Physical pressure and regular-source closure for the Case-A post-T graph.

This module supplies the :class:`~casea_post_t_liquid_stage.PressureCallback`
required by the conservative post-arrival liquid stage.  It contains no time
integration, junction exchange, filtering, waveform prescription, or plotting.

The stratified momentum flux follows the frozen-coefficient form used in the
companion method,

    F_Q = Q_l**2/A_l + C_component + 0.5*Lambda_d*A_l**2,
    c_Lambda**2 = max(Lambda_d*A_l, 0) + c_num**2.

``Lambda_d`` and the isothermal gas EOS are evaluated from the same conserved
state as :mod:`casea_horizontal_liquid_operator`.  The additive pressure-
potential constant is common to every cell of a connected material-gas
component and is fixed by traction continuity with an adjacent elastic-liquid
cell.  It is therefore a gauge choice, not a fitted wave forcing.

The vertical callback uses the theta=90 degree limit: the cross-sectional
buoyancy contribution (proportional to cos(theta)) vanishes, while axial
gravity appears once in ``momentum_source``.  A hydrostatic reference pressure
is carried in the flux so a constant-area column beneath an atmospheric free
surface is exactly well balanced by the finite-volume stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from casea_horizontal_liquid_operator import (
    HorizontalLiquidParameters,
    PressurePotentialState,
    _circular_depth_and_width,
    decoupled_lambda_and_derivative,
    pressure_potential_state,
)
from casea_post_t_liquid_stage import BranchPressureEvaluation


BranchName = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class CaseAPostTClosureParameters:
    """Geometry and constitutive constants for one frozen Case-A closure."""

    horizontal: HorizontalLiquidParameters
    vertical: HorizontalLiquidParameters
    vertical_cell_width: float
    gravity: float = 9.81
    kinematic_viscosity: float = 1.003e-3 / 998.0
    horizontal_darcy_factor: float = 0.025
    vertical_darcy_factor: float = 0.025
    active_void_fraction: float = 5.0e-4
    topology_density_fraction: float = 0.02
    resolved_density_fraction: float = 0.50

    def __post_init__(self) -> None:
        positive = (
            self.vertical_cell_width,
            self.gravity,
            self.kinematic_viscosity,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive finite post-T closure scales required")
        nonnegative = (
            self.horizontal_darcy_factor,
            self.vertical_darcy_factor,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("finite non-negative Darcy factors required")
        fractions = (
            self.active_void_fraction,
            self.topology_density_fraction,
            self.resolved_density_fraction,
        )
        if not all(np.isfinite(value) and 0.0 < value < 1.0 for value in fractions):
            raise ValueError("gas-topology fractions must lie in (0, 1)")
        if self.vertical.atmospheric_pressure != self.horizontal.atmospheric_pressure:
            raise ValueError("both branches require one atmospheric pressure")
        if self.vertical.rho_liquid != self.horizontal.rho_liquid:
            raise ValueError("both branches require one liquid density")
        if self.vertical.gas_constant != self.horizontal.gas_constant:
            raise ValueError("both branches require one gas constant")
        if self.vertical.gas_temperature != self.horizontal.gas_temperature:
            raise ValueError("both branches require one gas temperature")


def _material_gas_mask(
    area: np.ndarray,
    mass: np.ndarray,
    params: HorizontalLiquidParameters,
    closure: CaseAPostTClosureParameters,
) -> np.ndarray:
    """Identify connected, mass-backed gas without activating floor residue."""

    raw_void = np.maximum(params.area_full - area, 0.0)
    effective_void = np.maximum(
        raw_void, params.void_floor_fraction * params.area_full
    )
    reference_density = params.atmospheric_gas_density
    ordinary = (
        raw_void >= closure.active_void_fraction * params.area_full
    ) & (
        mass
        > closure.topology_density_fraction
        * reference_density
        * effective_void
        * params.cell_width
    )
    bridge = (
        raw_void > 1.5 * params.void_floor_fraction * params.area_full
    ) & (
        mass
        > closure.resolved_density_fraction
        * reference_density
        * effective_void
        * params.cell_width
    )
    supported = ordinary.copy()
    for _ in range(supported.size):
        adjacent = np.zeros_like(supported)
        if supported.size > 1:
            adjacent[1:] |= supported[:-1]
            adjacent[:-1] |= supported[1:]
        extended = supported | (bridge & adjacent)
        if np.array_equal(extended, supported):
            break
        supported = extended
    return supported


def _gas_density(
    area: np.ndarray,
    mass: np.ndarray,
    params: HorizontalLiquidParameters,
) -> np.ndarray:
    void = np.maximum(
        params.area_full - area,
        params.void_floor_fraction * params.area_full,
    )
    raw = mass / (void * params.cell_width)
    reference = params.atmospheric_gas_density
    return np.clip(
        raw,
        params.gas_density_floor_fraction * reference,
        params.gas_density_ceiling_fraction * reference,
    )


def _elastic_state(
    area: np.ndarray,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_area = np.maximum(area, params.elastic_separation_area)
    potential = (
        params.full_section_potential
        + 0.5
        * params.wave_speed**2
        * (active_area**2 - params.area_full**2)
        / params.area_full
    )
    tangent = np.where(
        area > params.elastic_separation_area,
        params.wave_speed**2 * area / params.area_full,
        0.0,
    )
    pressure = (
        params.atmospheric_pressure
        + params.rho_liquid
        * params.wave_speed**2
        * (active_area / params.area_full - 1.0)
    )
    return potential, tangent, pressure


def _vertical_lambda(
    area: np.ndarray,
    discharge: np.ndarray,
    gas_mass: np.ndarray,
    gas_momentum: np.ndarray,
    params: HorizontalLiquidParameters,
) -> np.ndarray:
    """Return Lambda_d in the vertical (cos(theta)=0) limit."""

    horizontal_lambda, _ = decoupled_lambda_and_derivative(
        area, discharge, gas_mass, gas_momentum, params
    )
    area_cap = np.minimum(area, params.geometry_cap_fraction * params.area_full)
    _, width = _circular_depth_and_width(area_cap, params)
    density = _gas_density(area, gas_mass, params)
    cross_buoyancy = (
        params.gravity
        * (1.0 - density / params.rho_liquid)
        / width
    )
    return horizontal_lambda - cross_buoyancy


def _component_offsets(
    area: np.ndarray,
    raw_potential: np.ndarray,
    supported: np.ndarray,
    elastic_potential: np.ndarray,
) -> np.ndarray:
    """Apply one traction-matched gauge to each connected gas component."""

    offsets = np.zeros_like(area)
    starts = np.flatnonzero(supported & ~np.r_[False, supported[:-1]])
    ends = np.flatnonzero(supported & ~np.r_[supported[1:], False]) + 1
    for first, last in zip(starts, ends):
        candidates: list[float] = []
        if first > 0:
            candidates.append(
                float(elastic_potential[first - 1] - raw_potential[first])
            )
        if last < area.size:
            candidates.append(
                float(elastic_potential[last] - raw_potential[last - 1])
            )
        offset = float(np.mean(candidates)) if candidates else 0.0
        offsets[first:last] = offset
    return offsets


def _frozen_pressure_state(
    area: np.ndarray,
    discharge: np.ndarray,
    gas_mass: np.ndarray,
    gas_momentum: np.ndarray,
    supported: np.ndarray,
    params: HorizontalLiquidParameters,
    closure: CaseAPostTClosureParameters,
    *,
    vertical: bool,
) -> tuple[PressurePotentialState, np.ndarray]:
    """Build the paper's frozen-Lambda pressure state and EOS pressure."""

    elastic_potential, elastic_tangent, elastic_pressure = _elastic_state(
        area, params
    )
    if not vertical:
        raw = np.zeros_like(area)
        if np.any(supported):
            lam, _ = decoupled_lambda_and_derivative(
                area[supported],
                discharge[supported],
                gas_mass[supported],
                gas_momentum[supported],
                params,
            )
            raw[supported] = 0.5 * lam * area[supported] ** 2
        offsets = _component_offsets(
            area, raw, supported, elastic_potential
        )
        # The horizontal branch is delegated to the canonical operator so its
        # EOS, frozen-Lambda speed, IKH regularisation, and diagnostics cannot
        # drift from the method implementation.
        state = pressure_potential_state(
            area,
            discharge,
            gas_mass,
            gas_momentum,
            supported,
            params,
            stratified_potential_offset=offsets,
        )
        gas_pressure = (
            _gas_density(area, gas_mass, params)
            * params.gas_constant
            * params.gas_temperature
        )
        pressure_abs = np.where(
            state.stratified, gas_pressure, elastic_pressure
        )
        return state, pressure_abs

    potential = elastic_potential.copy()
    lambda_value = np.zeros_like(area)
    lambda_derivative = np.zeros_like(area)
    if np.any(supported):
        lam = _vertical_lambda(
            area[supported],
            discharge[supported],
            gas_mass[supported],
            gas_momentum[supported],
            params,
        )
        raw = np.zeros_like(area)
        raw[supported] = 0.5 * lam * area[supported] ** 2
        offsets = _component_offsets(
            area, raw, supported, elastic_potential
        )
        potential[supported] = raw[supported] + offsets[supported]
        lambda_value[supported] = lam

    # Reuse Eq. (40)'s configured regularisation from the branch operator;
    # there is no second, closure-local wave-speed knob.
    floor2 = params.numerical_celerity_floor**2
    tangent = np.maximum(elastic_tangent, floor2)
    tangent[supported] = (
        np.maximum(lambda_value[supported] * area[supported], 0.0) + floor2
    )
    celerity = np.sqrt(tangent)
    velocity = discharge / area
    gas_pressure = (
        _gas_density(area, gas_mass, params)
        * params.gas_constant
        * params.gas_temperature
    )
    pressure_abs = np.where(supported, gas_pressure, elastic_pressure)
    state = PressurePotentialState(
        potential=potential,
        derivative=tangent,
        discharge_derivative=np.zeros_like(area),
        celerity=celerity,
        eigenvalue_minus=velocity - celerity,
        eigenvalue_plus=velocity + celerity,
        lambda_value=lambda_value,
        lambda_derivative=lambda_derivative,
        stratified=supported.copy(),
    )
    return state, pressure_abs


def physical_momentum_source(
    branch: BranchName,
    area: np.ndarray,
    discharge: np.ndarray,
    params: CaseAPostTClosureParameters,
) -> np.ndarray:
    """Return axial gravity plus dissipative pipe-wall friction exactly once."""

    local = params.horizontal if branch == "horizontal" else params.vertical
    darcy = (
        params.horizontal_darcy_factor
        if branch == "horizontal"
        else params.vertical_darcy_factor
    )
    velocity = discharge / area
    rate = (
        32.0 * params.kinematic_viscosity / local.diameter**2
        + darcy * np.abs(velocity) / (2.0 * local.diameter)
    )
    source = -rate * discharge
    if branch == "vertical":
        source = source - params.gravity * area
    return source


class CaseAPostTPhysicalClosure:
    """Callable pressure/source closure accepted by the post-T liquid stage."""

    def __init__(self, params: CaseAPostTClosureParameters) -> None:
        self.params = params

    def __call__(
        self,
        branch: BranchName,
        area: np.ndarray,
        discharge: np.ndarray,
        gas_mass: np.ndarray,
        gas_momentum: np.ndarray,
    ) -> BranchPressureEvaluation:
        area_a = np.asarray(area, dtype=float)
        q = np.asarray(discharge, dtype=float)
        mass = np.asarray(gas_mass, dtype=float)
        momentum = np.asarray(gas_momentum, dtype=float)
        if branch not in ("horizontal", "vertical"):
            raise ValueError("branch must be 'horizontal' or 'vertical'")
        if not (
            area_a.ndim == 1
            and area_a.size > 0
            and area_a.shape == q.shape == mass.shape == momentum.shape
        ):
            raise ValueError("closure fields must be equal non-empty 1-D arrays")
        if not all(
            np.all(np.isfinite(value))
            for value in (area_a, q, mass, momentum)
        ):
            raise ValueError("closure fields must be finite")
        if np.any(area_a <= 0.0) or np.any(mass < 0.0):
            raise ValueError("positive liquid area and non-negative gas mass required")

        local = self.params.horizontal if branch == "horizontal" else self.params.vertical
        supported = _material_gas_mask(area_a, mass, local, self.params)
        state, barotropic_pressure = _frozen_pressure_state(
            area_a,
            q,
            mass,
            momentum,
            supported,
            local,
            self.params,
            vertical=branch == "vertical",
        )
        source = physical_momentum_source(branch, area_a, q, self.params)
        if branch == "horizontal":
            face_pressure = barotropic_pressure.copy()
            potential_pressure = barotropic_pressure.copy()
        else:
            height = area_a.size * self.params.vertical_cell_width
            z = (
                np.arange(area_a.size, dtype=float) + 0.5
            ) * self.params.vertical_cell_width
            hydrostatic_head = self.params.gravity * (height - z)
            hydrostatic_potential = local.area_full * hydrostatic_head
            state = PressurePotentialState(
                potential=state.potential + hydrostatic_potential,
                derivative=state.derivative,
                discharge_derivative=state.discharge_derivative,
                celerity=state.celerity,
                eigenvalue_minus=state.eigenvalue_minus,
                eigenvalue_plus=state.eigenvalue_plus,
                lambda_value=state.lambda_value,
                lambda_derivative=state.lambda_derivative,
                stratified=state.stratified,
            )
            face_pressure = (
                barotropic_pressure
                + local.rho_liquid * hydrostatic_head
            )
            # The incoming characteristic at cell 0 is referenced at the T
            # face, half a control volume below its cell centre.
            face_pressure[0] += (
                0.5
                * local.rho_liquid
                * self.params.gravity
                * self.params.vertical_cell_width
            )
            potential_pressure = (
                barotropic_pressure
                + local.rho_liquid
                * hydrostatic_potential
                / area_a
            )

        if not (
            np.all(np.isfinite(state.potential))
            and np.all(np.isfinite(state.celerity))
            and np.all(state.celerity > 0.0)
            and np.all(np.isfinite(face_pressure))
            and np.all(face_pressure > 0.0)
            and np.all(np.isfinite(source))
        ):
            raise FloatingPointError("non-finite or non-positive closure state")
        return BranchPressureEvaluation(
            pressure=state,
            face_pressure_abs=face_pressure,
            node_pressure_offset=np.zeros_like(area_a),
            momentum_source=source,
            potential_pressure_abs=potential_pressure,
        )
