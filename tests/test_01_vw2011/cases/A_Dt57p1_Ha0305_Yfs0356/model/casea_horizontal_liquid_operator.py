"""Circular-pipe shallow-water liquid operator for the Case-A tunnel.

This module contains only local, side-effect-free numerical building blocks.
It deliberately does not know about the Case-A time loop, T-junction schedule,
or plotting.  The pressure potential and its Jacobian are evaluated together,
so a Riemann solver cannot use a wave speed from a different equation than the
one used in its momentum flux.

For a mass-supported free-surface cell the liquid momentum flux is

    F_Q = Q**2 / A + Psi(A),
    Psi(A) = C + g*I1(A),

where ``I1`` is the exact hydrostatic pressure moment of the wetted circular
segment.  Its squared gravity-wave celerity is

    c_l**2 = g*A/T,

with ``T`` the free-surface top width.  The gas EOS and gas momentum remain in
the coupled gas graph; gas pressure acts through the regular liquid pressure
source in the network solver.  No Kelvin--Helmholtz slip term is present in
this Case-A horizontal operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from casea_acceleration import njit


@dataclass(frozen=True)
class HorizontalLiquidParameters:
    area_full: float
    diameter: float
    wave_speed: float
    cell_width: float
    gravity: float = 9.81
    rho_liquid: float = 998.2
    gas_constant: float = 287.05
    gas_temperature: float = 293.15
    atmospheric_pressure: float = 101325.0
    tension_head: float = 0.05
    geometry_cap_fraction: float = 0.995
    void_floor_fraction: float = 1.0e-4
    gas_density_floor_fraction: float = 0.2
    gas_density_ceiling_fraction: float = 12.0
    # Small wet/dry numerical celerity used only at vanishing liquid area.
    numerical_celerity_floor: float = 1.0e-3

    def __post_init__(self) -> None:
        positive = {
            "area_full": self.area_full,
            "diameter": self.diameter,
            "wave_speed": self.wave_speed,
            "cell_width": self.cell_width,
            "gravity": self.gravity,
            "rho_liquid": self.rho_liquid,
            "gas_constant": self.gas_constant,
            "gas_temperature": self.gas_temperature,
            "atmospheric_pressure": self.atmospheric_pressure,
            "numerical_celerity_floor": self.numerical_celerity_floor,
        }
        bad = [name for name, value in positive.items() if value <= 0.0]
        if bad:
            raise ValueError(f"positive horizontal-liquid parameters required: {bad}")
        if not 0.0 < self.geometry_cap_fraction < 1.0:
            raise ValueError("geometry_cap_fraction must lie in (0, 1)")
        if not 0.0 < self.void_floor_fraction < 1.0:
            raise ValueError("void_floor_fraction must lie in (0, 1)")

    @property
    def atmospheric_gas_density(self) -> float:
        return (
            self.atmospheric_pressure
            / (self.gas_constant * self.gas_temperature)
        )

    @property
    def elastic_separation_area(self) -> float:
        fraction = 1.0 - (
            self.tension_head * self.gravity / self.wave_speed**2
        )
        return self.area_full * max(fraction, 0.0)

    @property
    def full_section_potential(self) -> float:
        # For a full circle I1 = integral(D-y)dA = A_full*D/2.
        return 0.5 * self.gravity * self.area_full * self.diameter


@dataclass(frozen=True)
class PressurePotentialState:
    potential: np.ndarray
    derivative: np.ndarray
    discharge_derivative: np.ndarray
    celerity: np.ndarray
    eigenvalue_minus: np.ndarray
    eigenvalue_plus: np.ndarray
    lambda_value: np.ndarray
    lambda_derivative: np.ndarray
    stratified: np.ndarray


@dataclass(frozen=True)
class PressurePotentialWaveState:
    """Only the pressure potential and celerity needed by a Riemann flux."""

    potential: np.ndarray
    celerity: np.ndarray


def _broadcast_float_arrays(*values: object) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(value, dtype=float)
        for value in np.broadcast_arrays(*values)
    )


@njit(cache=True)
def _circular_depth_and_width_kernel(
    target: np.ndarray,
    area_full: float,
    diameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled algebra for the exact safeguarded circular-segment solve."""

    fraction = target / area_full
    mirrored = fraction > 0.5
    reduced = np.where(mirrored, 1.0 - fraction, fraction)
    rhs = 2.0 * np.pi * reduced

    lo = np.zeros_like(reduced)
    hi = np.full_like(reduced, np.pi)
    phi = np.clip(
        np.cbrt(12.0 * np.pi * reduced),
        0.0,
        np.pi,
    )
    phi = np.where(reduced == 0.5, np.pi, phi)
    for _ in range(12):
        residual = phi - np.sin(phi) - rhs
        converged = np.abs(residual) <= (
            16.0
            * np.finfo(np.float64).eps
            * np.maximum(np.abs(rhs), 1.0)
        )
        if np.all(converged):
            break
        lo = np.where(residual < 0.0, phi, lo)
        hi = np.where(residual < 0.0, hi, phi)
        derivative = 1.0 - np.cos(phi)
        derivative_active = derivative > 1.0e-15
        safe_derivative = np.where(derivative_active, derivative, 1.0)
        newton = np.where(
            derivative_active,
            residual / safe_derivative,
            0.0,
        )
        candidate = phi - newton
        midpoint = 0.5 * (lo + hi)
        proposed = np.where(
            (candidate > lo) & (candidate < hi), candidate, midpoint
        )
        phi = np.where(converged, phi, proposed)

    radius = 0.5 * diameter
    reduced_depth = radius * (1.0 - np.cos(0.5 * phi))
    depth = np.where(
        mirrored,
        diameter - reduced_depth,
        reduced_depth,
    )
    depth = np.where(fraction == 0.5, radius, depth)
    depth = np.where(target <= 0.0, 0.0, depth)
    depth = np.where(target >= area_full, diameter, depth)
    width = 2.0 * np.sqrt(
        np.maximum(depth * (diameter - depth), 0.0)
    )
    width = np.where(fraction == 0.5, diameter, width)
    return depth, width


def _circular_depth_and_width(
    area: np.ndarray,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert circular-segment area with a safeguarded angle solve.

    For central angle ``phi`` the normalized segment area is

    ``A/Af = (phi - sin(phi))/(2*pi)``.

    The former implementation repeated a full-array bisection 55 times on
    every Riemann evaluation.  A symmetry-reduced Newton iteration with a
    maintained bracket reaches the same double-precision root in twelve
    iterations and contains no interpolation table or fitted geometry.
    """

    target = np.clip(np.asarray(area, dtype=float), 0.0, params.area_full)
    original_shape = target.shape
    depth, width = _circular_depth_and_width_kernel(
        np.atleast_1d(target).reshape(-1),
        float(params.area_full),
        float(params.diameter),
    )
    return depth.reshape(original_shape), width.reshape(original_shape)


def _circular_hydrostatic_state(
    area: object,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``g I1``, its tangent, ``g/T``, and ``d(g/T)/dA``.

    ``I1 = integral_0^h (h-y)b(y)dy`` is evaluated analytically for the
    circular segment.  Consequently ``d(gI1)/dA = gA/T`` and the Riemann
    celerity is exactly the Saint--Venant value ``sqrt(gA/T)``.
    """

    area_raw = np.asarray(area, dtype=float)
    if np.any(area_raw <= 0.0):
        raise ValueError("positive liquid area required for shallow water")
    area_eval = np.minimum(
        area_raw,
        params.geometry_cap_fraction * params.area_full,
    )
    depth, width = _circular_depth_and_width(area_eval, params)
    if np.any(width <= 0.0):
        raise ValueError("finite circular top width required for shallow water")

    radius = 0.5 * params.diameter
    cosine = np.clip((radius - depth) / radius, -1.0, 1.0)
    half_angle = np.arccos(cosine)
    angle = 2.0 * half_angle
    sine_half = np.sin(half_angle)
    segment_factor = angle - np.sin(angle)
    i1 = radius**3 * (
        -(0.5 * segment_factor * np.cos(half_angle))
        + (2.0 / 3.0) * sine_half**3
    )
    potential = params.gravity * i1
    tangent = params.gravity * area_eval / width
    coefficient = params.gravity / width

    # T=2*sqrt(h(D-h)); dT/dA=2(D-2h)/T**2.
    dwidth_darea = 2.0 * (params.diameter - 2.0 * depth) / width**2
    derivative = -params.gravity * dwidth_darea / width**2
    derivative = np.where(
        area_raw < params.geometry_cap_fraction * params.area_full,
        derivative,
        0.0,
    )
    return potential, tangent, coefficient, derivative


def _decoupled_lambda_derivatives(
    area: object,
    discharge: object,
    gas_mass: object,
    gas_momentum: object,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shallow-water coefficient ``g/T`` and its area derivative.

    ``discharge``, ``gas_mass``, and ``gas_momentum`` remain in the signature
    for API compatibility with the coupled network.  The shallow-water
    restoring law is independent of gas--liquid slip.
    """

    area_raw, _, mass, _ = _broadcast_float_arrays(
        area, discharge, gas_mass, gas_momentum
    )
    if np.any(mass <= 0.0):
        raise ValueError("positive resolved gas mass required for free-surface cells")
    _, _, coefficient, derivative = _circular_hydrostatic_state(
        area_raw, params
    )
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(derivative)):
        raise FloatingPointError("non-finite shallow-water restoring coefficient")
    return coefficient, derivative, np.zeros_like(coefficient)


def decoupled_lambda_and_derivative(
    area: object,
    discharge: object,
    gas_mass: object,
    gas_momentum: object,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Lambda`` and its partial derivative with respect to liquid area."""

    coefficient, derivative, _ = _decoupled_lambda_derivatives(
        area, discharge, gas_mass, gas_momentum, params
    )
    return coefficient, derivative


def _elastic_potential_and_derivative(
    area: np.ndarray,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray]:
    separation = params.elastic_separation_area
    active_area = np.maximum(area, separation)
    potential = (
        params.full_section_potential
        + 0.5 * params.wave_speed**2
        * (active_area**2 - params.area_full**2) / params.area_full
    )
    derivative = np.where(
        area > separation,
        params.wave_speed**2 * area / params.area_full,
        0.0,
    )
    return potential, derivative


def pressure_potential_wave_state(
    area: object,
    mass_supported: object,
    params: HorizontalLiquidParameters,
    *,
    stratified_potential_offset: object | None = None,
) -> PressurePotentialWaveState:
    """Evaluate only the two hybrid-pressure fields used by liquid fluxes.

    This is the same potential and tangent as :func:`pressure_potential_state`,
    without constructing discharge derivatives, eigenvalues or Lambda
    diagnostics that the Riemann caller immediately discarded.
    """

    area_a, support_f = np.broadcast_arrays(
        np.asarray(area, dtype=float),
        np.asarray(mass_supported, dtype=bool),
    )
    if np.any(area_a <= 0.0):
        raise ValueError("positive liquid area required")
    original_shape = area_a.shape
    area_f = np.asarray(area_a, dtype=float).reshape(-1)
    support = np.asarray(support_f, dtype=bool).reshape(-1)
    offset_f = None
    if stratified_potential_offset is not None:
        offset_f = np.broadcast_to(
            np.asarray(stratified_potential_offset, dtype=float),
            original_shape,
        ).reshape(-1)
        if not np.all(np.isfinite(offset_f)):
            raise ValueError("stratified pressure-potential offsets must be finite")

    potential, tangent = _elastic_potential_and_derivative(area_f, params)
    potential = np.asarray(potential).copy()
    tangent = np.asarray(tangent).copy()
    stratified = support & (area_f < params.elastic_separation_area)
    if np.any(stratified):
        hydro, hydro_tangent, _, _ = _circular_hydrostatic_state(
            area_f[stratified], params
        )
        potential[stratified] = hydro
        if offset_f is not None:
            potential[stratified] += offset_f[stratified]
        tangent[stratified] = hydro_tangent
    celerity = np.sqrt(
        np.maximum(tangent, 0.0) + params.numerical_celerity_floor**2
    )
    return PressurePotentialWaveState(
        potential=potential.reshape(original_shape),
        celerity=celerity.reshape(original_shape),
    )


def pressure_potential_state(
    area: object,
    discharge: object,
    gas_mass: object,
    gas_momentum: object,
    mass_supported: object,
    params: HorizontalLiquidParameters,
    *,
    stratified_potential_offset: object | None = None,
) -> PressurePotentialState:
    """Evaluate the shallow-water/elastic hybrid pressure potential.

    A mass-supported layer uses the exact circular Saint--Venant pressure
    moment below the finite-tension separation area.  Its natural gauge is
    ``g*I1``.  The network may supply one spatially constant offset for a
    connected free-surface component so its material-front traction matches
    the neighbouring elastic branch.  Gas pressure is deliberately absent
    here and is applied once as a regular pressure source by the network.
    """

    area_a, q_a, mass_a, momentum_a, support_f = np.broadcast_arrays(
        np.asarray(area, dtype=float),
        np.asarray(discharge, dtype=float),
        np.asarray(gas_mass, dtype=float),
        np.asarray(gas_momentum, dtype=float),
        np.asarray(mass_supported, dtype=bool),
    )
    support = support_f.astype(bool)
    if np.any(area_a <= 0.0):
        raise ValueError("positive liquid area required")

    original_shape = area_a.shape
    area_f = np.asarray(area_a, dtype=float).reshape(-1)
    q_f = np.asarray(q_a, dtype=float).reshape(-1)
    support_f = np.asarray(support, dtype=bool).reshape(-1)
    offset_f = None
    if stratified_potential_offset is not None:
        offset_f = np.broadcast_to(
            np.asarray(stratified_potential_offset, dtype=float),
            original_shape,
        ).reshape(-1)
        if not np.all(np.isfinite(offset_f)):
            raise ValueError("stratified pressure-potential offsets must be finite")

    elastic_potential, elastic_derivative = _elastic_potential_and_derivative(
        area_f, params
    )
    transition = params.elastic_separation_area
    stratified = support_f & (area_f < transition)

    lambda_value = np.zeros_like(area_f)
    lambda_derivative = np.zeros_like(area_f)
    lambda_discharge_derivative = np.zeros_like(area_f)
    potential = np.asarray(elastic_potential).copy()
    tangent = np.asarray(elastic_derivative).copy()
    pressure_q_derivative = np.zeros_like(area_f)
    if np.any(stratified):
        hydro, hydro_tangent, lam, dlam = _circular_hydrostatic_state(
            area_f[stratified], params
        )
        if offset_f is None:
            potential[stratified] = hydro
        else:
            potential[stratified] = hydro + offset_f[stratified]
        tangent[stratified] = hydro_tangent
        lambda_value[stratified] = lam
        lambda_derivative[stratified] = dlam

    velocity = q_f / area_f
    # The physical shallow-water/elastic tangent is non-negative.  The small
    # floor is active only in the dry limit and is not part of the flux.
    numerical_tangent = (
        np.maximum(tangent, 0.0) + params.numerical_celerity_floor**2
    )
    celerity = np.sqrt(numerical_tangent)
    eigenvalue_minus = velocity - celerity
    eigenvalue_plus = velocity + celerity
    return PressurePotentialState(
        potential=np.asarray(potential).reshape(original_shape),
        derivative=np.asarray(numerical_tangent).reshape(original_shape),
        discharge_derivative=np.asarray(pressure_q_derivative).reshape(original_shape),
        celerity=np.asarray(celerity).reshape(original_shape),
        eigenvalue_minus=np.asarray(eigenvalue_minus).reshape(original_shape),
        eigenvalue_plus=np.asarray(eigenvalue_plus).reshape(original_shape),
        lambda_value=np.asarray(lambda_value).reshape(original_shape),
        lambda_derivative=np.asarray(lambda_derivative).reshape(original_shape),
        stratified=np.asarray(stratified).reshape(original_shape),
    )


def physical_liquid_flux(
    area: object,
    discharge: object,
    pressure: PressurePotentialState,
) -> np.ndarray:
    area_a, q_a, psi = _broadcast_float_arrays(
        area, discharge, pressure.potential
    )
    if np.any(area_a <= 0.0):
        raise ValueError("positive liquid area required")
    return np.stack((q_a, q_a**2 / area_a + psi), axis=-1)


def characteristic_spectral_radius(
    area: object,
    discharge: object,
    pressure: PressurePotentialState,
) -> np.ndarray:
    eigenvalue_minus, eigenvalue_plus = _broadcast_float_arrays(
        pressure.eigenvalue_minus, pressure.eigenvalue_plus
    )
    return np.maximum(
        np.abs(eigenvalue_minus), np.abs(eigenvalue_plus)
    )


def rusanov_face_flux(
    area_left: object,
    discharge_left: object,
    pressure_left: PressurePotentialState,
    area_right: object,
    discharge_right: object,
    pressure_right: PressurePotentialState,
) -> tuple[np.ndarray, np.ndarray]:
    """Rusanov flux using the spectral radius of the actual pressure Jacobian."""

    flux_left = physical_liquid_flux(area_left, discharge_left, pressure_left)
    flux_right = physical_liquid_flux(area_right, discharge_right, pressure_right)
    state_left = np.stack(_broadcast_float_arrays(area_left, discharge_left), axis=-1)
    state_right = np.stack(
        _broadcast_float_arrays(area_right, discharge_right), axis=-1
    )
    speed = np.maximum(
        characteristic_spectral_radius(area_left, discharge_left, pressure_left),
        characteristic_spectral_radius(area_right, discharge_right, pressure_right),
    )
    flux = 0.5 * (flux_left + flux_right) - 0.5 * speed[..., None] * (
        state_right - state_left
    )
    return flux, speed


StageRhs = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def ssprk2_stage_step(
    area: object,
    discharge: object,
    time: float,
    dt: float,
    stage_rhs: StageRhs,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one SSP-RK2 step, recomputing flux and source at each stage.

    ``stage_rhs`` is intentionally a callback.  The caller may reconstruct gas
    variables, junction boundary fluxes, and physical source terms from each
    stage state.  Reusing a stage-0 flux/source in stage 1 is therefore
    impossible unless the caller explicitly chooses to do so.
    """

    if dt < 0.0:
        raise ValueError("non-negative SSP-RK2 step required")
    area0, discharge0 = _broadcast_float_arrays(area, discharge)
    rhs_a0, rhs_q0 = stage_rhs(area0.copy(), discharge0.copy(), float(time))
    rhs_a0, rhs_q0 = _broadcast_float_arrays(rhs_a0, rhs_q0)
    area1 = area0 + dt * rhs_a0
    discharge1 = discharge0 + dt * rhs_q0
    rhs_a1, rhs_q1 = stage_rhs(
        area1.copy(), discharge1.copy(), float(time + dt)
    )
    rhs_a1, rhs_q1 = _broadcast_float_arrays(rhs_a1, rhs_q1)
    area2 = 0.5 * area0 + 0.5 * (area1 + dt * rhs_a1)
    discharge2 = 0.5 * discharge0 + 0.5 * (
        discharge1 + dt * rhs_q1
    )
    if not np.all(np.isfinite(area2)) or not np.all(np.isfinite(discharge2)):
        raise FloatingPointError("non-finite SSP-RK2 liquid state")
    return area2, discharge2
