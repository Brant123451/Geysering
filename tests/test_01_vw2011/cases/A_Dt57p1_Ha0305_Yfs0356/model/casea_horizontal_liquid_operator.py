"""Consistent barotropic liquid operator for the Case-A horizontal pipe.

This module contains only local, side-effect-free numerical building blocks.
It deliberately does not know about the Case-A time loop, T-junction schedule,
or plotting.  The pressure potential and its Jacobian are evaluated together,
so a Riemann solver cannot use a wave speed from a different equation than the
one used in its momentum flux.

For a mass-supported stratified cell the liquid momentum flux is

    F_Q = Q**2 / A + Psi(A),
    Psi(A) = C + 0.5 * Lambda(A) * A**2.

The companion model treats ``Lambda`` as a frozen face coefficient when the
liquid Riemann problem is linearised.  Its squared liquid celerity is therefore

    c_l**2 = Lambda*A,

as stated by the methods-paper eigenvalues
``u_l +/- sqrt(Lambda*A_l)``.  Derivatives of ``Lambda`` are still exposed for
diagnostics, but they are not silently substituted into that published
frozen-coefficient Jacobian.  A negative ``Lambda*A`` is a loss of
hyperbolicity of the reduced closure and is reported explicitly rather than
hidden with ``max(value, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class LossOfHyperbolicity(RuntimeError):
    """Raised when the reduced pressure law has a negative tangent modulus."""


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
    # Eq. (40) of the companion numerical method: a small numerical celerity
    # keeps the Rusanov dissipation and CFL finite when Lambda_d*A_l reaches or
    # crosses the IKH neutral point.  It is not added to the physical flux.
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


def _broadcast_float_arrays(*values: object) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(value, dtype=float)
        for value in np.broadcast_arrays(*values)
    )


def _circular_depth_and_width(
    area: np.ndarray,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert circular-segment area without a table or fitted interpolation."""

    target = np.clip(np.asarray(area, dtype=float), 0.0, params.area_full)
    lo = np.zeros_like(target)
    hi = np.full_like(target, params.diameter)
    radius = 0.5 * params.diameter
    for _ in range(55):
        depth = 0.5 * (lo + hi)
        y = radius - depth
        root = np.sqrt(np.maximum(radius**2 - y**2, 0.0))
        segment_area = (
            radius**2 * np.arccos(np.clip(y / radius, -1.0, 1.0))
            - y * root
        )
        lo = np.where(segment_area < target, depth, lo)
        hi = np.where(segment_area < target, hi, depth)
    depth = 0.5 * (lo + hi)
    width = 2.0 * np.sqrt(
        np.maximum(depth * (params.diameter - depth), 0.0)
    )
    return depth, width


def _decoupled_lambda_derivatives(
    area: object,
    discharge: object,
    gas_mass: object,
    gas_momentum: object,
    params: HorizontalLiquidParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Eq. (A31) ``Lambda`` and its full partial derivative in ``A``.

    The derivative holds the other conserved variables ``Q, M_g, J_g`` fixed,
    exactly as required by the liquid-flux Jacobian.  Derivatives of gas
    density, liquid velocity, circular geometry, and slip are all retained.
    Existing density and crown-geometry bounds are treated as piecewise model
    definitions: their derivatives are zero only on an active bound.
    """

    area_raw, q, mass, momentum = _broadcast_float_arrays(
        area, discharge, gas_mass, gas_momentum
    )
    if np.any(area_raw <= 0.0):
        raise ValueError("positive liquid area required for the stratified law")
    if np.any(mass <= 0.0):
        raise ValueError("positive resolved gas mass required for stratified cells")

    area_cap = params.geometry_cap_fraction * params.area_full
    area_eval = np.minimum(area_raw, area_cap)
    darea_eval = (area_raw < area_cap).astype(float)

    void_unbounded = params.area_full - area_raw
    void_floor = params.void_floor_fraction * params.area_full
    gas_area = np.maximum(void_unbounded, void_floor)
    dgas_area = np.where(void_unbounded > void_floor, -1.0, 0.0)

    rho_raw = mass / (gas_area * params.cell_width)
    rho_floor = (
        params.gas_density_floor_fraction * params.atmospheric_gas_density
    )
    rho_ceiling = (
        params.gas_density_ceiling_fraction * params.atmospheric_gas_density
    )
    rho_g = np.clip(rho_raw, rho_floor, rho_ceiling)
    rho_is_free = (rho_raw > rho_floor) & (rho_raw < rho_ceiling)
    drho_raw = -rho_raw * dgas_area / gas_area
    drho = np.where(rho_is_free, drho_raw, 0.0)

    u_l = q / area_eval
    du_l = -q * darea_eval / area_eval**2
    u_g = momentum / mass
    delta_u = u_g - u_l
    ddelta_u = -du_l

    _, width = _circular_depth_and_width(area_eval, params)
    if np.any(width <= 0.0):
        raise ValueError("finite circular top width required for stratified cells")
    zeta = 1.0 / width
    # b=2*sqrt(h*(D-h)); db/dA=2*(D-2h)/b**2.
    depth, _ = _circular_depth_and_width(area_eval, params)
    dzeta_darea_eval = -2.0 * (params.diameter - 2.0 * depth) / width**4
    dzeta = dzeta_darea_eval * darea_eval

    gas_head = (
        rho_g * params.gas_constant * params.gas_temperature
        - params.atmospheric_pressure
    ) / (params.rho_liquid * params.gravity)
    dgas_head = (
        drho * params.gas_constant * params.gas_temperature
        / (params.rho_liquid * params.gravity)
    )

    term_pressure = 2.0 * params.gravity * gas_head / area_eval
    dterm_pressure = 2.0 * params.gravity * (
        dgas_head / area_eval
        - gas_head * darea_eval / area_eval**2
    )

    density_ratio = rho_g / params.rho_liquid
    ddensity_ratio = drho / params.rho_liquid
    term_buoyancy = params.gravity * (1.0 - density_ratio) * zeta
    dterm_buoyancy = params.gravity * (
        -ddensity_ratio * zeta + (1.0 - density_ratio) * dzeta
    )

    slip_over_void = rho_g * delta_u**2 / gas_area
    dslip_over_void = (
        drho * delta_u**2 / gas_area
        + 2.0 * rho_g * delta_u * ddelta_u / gas_area
        - rho_g * delta_u**2 * dgas_area / gas_area**2
    )
    term_slip = -slip_over_void / params.rho_liquid
    dterm_slip = -dslip_over_void / params.rho_liquid
    # Q is the second conserved liquid variable.  Retaining this derivative is
    # required because Lambda contains the liquid/gas slip velocity.
    ddelta_dq = -1.0 / area_eval
    dterm_slip_dq = -(
        2.0 * rho_g * delta_u * ddelta_dq / gas_area
    ) / params.rho_liquid

    coefficient = term_pressure + term_buoyancy + term_slip
    derivative = dterm_pressure + dterm_buoyancy + dterm_slip
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(derivative)):
        raise FloatingPointError("non-finite Lambda or dLambda/dA")
    return coefficient, derivative, dterm_slip_dq


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
    """Evaluate the hybrid potential and published frozen-coefficient speed.

    A mass-supported layer uses the stratified two-fluid potential below the
    finite-tension separation area.  By default its additive constant is
    chosen from the elastic potential at that same area.  A graph solver may
    instead supply ``stratified_potential_offset``: one spatially constant
    gauge per connected gas component, fixed by the resolved liquid-side
    traction at that component's fitted material front.  The offset changes
    no Jacobian eigenvalue and avoids inventing a different pressure zero in
    every cell of one acoustically connected pocket.
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
    mass_f = np.asarray(mass_a, dtype=float).reshape(-1)
    momentum_f = np.asarray(momentum_a, dtype=float).reshape(-1)
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

    lambda_value = np.zeros_like(area_a)
    lambda_derivative = np.zeros_like(area_a)
    lambda_value = np.zeros_like(area_f)
    lambda_derivative = np.zeros_like(area_f)
    lambda_discharge_derivative = np.zeros_like(area_f)
    potential = np.asarray(elastic_potential).copy()
    tangent = np.asarray(elastic_derivative).copy()
    pressure_q_derivative = np.zeros_like(area_f)
    if np.any(stratified):
        lam, dlam, dlam_dq = _decoupled_lambda_derivatives(
            area_f[stratified],
            q_f[stratified],
            mass_f[stratified],
            momentum_f[stratified],
            params,
        )
        a = area_f[stratified]
        if offset_f is None:
            transition_area = np.full(
                np.count_nonzero(stratified), transition
            )
            lam_transition, _, _ = _decoupled_lambda_derivatives(
                transition_area,
                q_f[stratified],
                mass_f[stratified],
                momentum_f[stratified],
                params,
            )
            elastic_transition, _ = _elastic_potential_and_derivative(
                transition_area, params
            )
            potential[stratified] = (
                elastic_transition
                + 0.5 * lam * a**2
                - 0.5 * lam_transition * transition**2
            )
        else:
            potential[stratified] = (
                0.5 * lam * a**2 + offset_f[stratified]
            )
        # The companion liquid block is quasi-linear: Lambda_d is evaluated
        # from the complete reconstructed stage state and then frozen over the
        # local liquid Riemann solve.  Its published eigenvalues are
        # u_l +/- sqrt(Lambda_d*A_l); dLambda/dA and dLambda/dQ therefore remain
        # diagnostics and do not alter this Riemann celerity.
        tangent[stratified] = lam * a
        lambda_value[stratified] = lam
        lambda_derivative[stratified] = dlam
        lambda_discharge_derivative[stratified] = dlam_dq

    velocity = q_f / area_f
    # The published scheme deliberately continues through the neutral/IKH
    # point with Eq. (40).  The sign of Lambda remains available in
    # ``lambda_value``; only the numerical spectral radius is regularised.
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
