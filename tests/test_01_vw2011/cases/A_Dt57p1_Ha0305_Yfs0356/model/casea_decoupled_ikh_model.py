"""Local four-equation stratified-flow branch for the Case-A IKH audit.

This module is an isolated transcription of the companion-paper model in
``E:/Research/论文/my sci``.  It implements the active stratified branch only:

    U = [A_g rho_g, rho_g Q_g, A_l, Q_l]

with the gas equations (A16)--(A17), the decoupled liquid equations (A30),
and the restoring coefficient (A31).  Primitive-variable MUSCL reconstruction,
block Rusanov fluxes, and SSP-RK2 follow Eqs. (31)--(32) of
``main_text_current_algorithm.tex``.

It is deliberately not a replacement for the pressurised cut-cell branch or
the Case-A T-junction.  The module is used first as a controlled local growth
test at the frame-187 state; that test decides whether a full network coupling
is scientifically warranted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    diameter: float = 0.094
    rho_l: float = 1000.0
    gravity: float = 9.80665
    gas_constant: float = 287.0
    gas_temperature: float = 293.15
    atmospheric_pressure: float = 101325.0
    c_num: float = 1.0e-4
    limiter_theta: float = 1.5

    @property
    def area_full(self) -> float:
        return 0.25 * math.pi * self.diameter**2

    @property
    def gas_sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.gas_temperature)


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(positive, magnitude, np.where(negative, -magnitude, 0.0))


def _periodic_reconstruction(
    values: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right primitive states at every periodic face."""

    delta_left = values - np.roll(values, 1)
    delta_right = np.roll(values, -1) - values
    delta_centre = 0.5 * (np.roll(values, -1) - np.roll(values, 1))
    slope = _minmod3(theta * delta_left, delta_centre, theta * delta_right)
    left = values + 0.5 * slope
    right = np.roll(values - 0.5 * slope, -1)
    return left, right


def gamma_from_holdup(alpha_l: np.ndarray) -> np.ndarray:
    """Full wetted central angle for a circular liquid segment."""

    alpha = np.clip(np.asarray(alpha_l, dtype=float), 1.0e-8, 1.0 - 1.0e-8)
    lo = np.full_like(alpha, 1.0e-10)
    hi = np.full_like(alpha, 2.0 * math.pi - 1.0e-10)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fraction = (mid - np.sin(mid)) / (2.0 * math.pi)
        lo = np.where(fraction < alpha, mid, lo)
        hi = np.where(fraction < alpha, hi, mid)
    return 0.5 * (lo + hi)


def liquid_depth(area_l: np.ndarray, params: ModelParameters) -> np.ndarray:
    gamma = gamma_from_holdup(area_l / params.area_full)
    return 0.5 * params.diameter * (1.0 - np.cos(0.5 * gamma))


def restoring_coefficient(
    area_l: np.ndarray,
    velocity_l: np.ndarray,
    density_g: np.ndarray,
    velocity_g: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """Return Lambda_d from companion-paper Eq. (A31)."""

    area_f = params.area_full
    area_l = np.clip(np.asarray(area_l, dtype=float), 1.0e-8 * area_f, (1.0 - 1.0e-8) * area_f)
    area_g = area_f - area_l
    gamma = gamma_from_holdup(area_l / area_f)
    top_width = np.maximum(params.diameter * np.sin(0.5 * gamma), 1.0e-12)
    zeta = 1.0 / top_width
    pressure_g = density_g * params.gas_constant * params.gas_temperature
    head_g = (pressure_g - params.atmospheric_pressure) / (
        params.rho_l * params.gravity
    )
    return (
        2.0 * params.gravity * head_g / area_l
        + (params.rho_l - density_g) / params.rho_l * params.gravity * zeta
        - density_g / params.rho_l * (velocity_g - velocity_l) ** 2 / area_g
    )


def primitives_to_conserved(
    density_g: np.ndarray,
    velocity_g: np.ndarray,
    area_l: np.ndarray,
    velocity_l: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    area_g = params.area_full - area_l
    return np.vstack(
        (
            area_g * density_g,
            area_g * density_g * velocity_g,
            area_l,
            area_l * velocity_l,
        )
    )


def conserved_to_primitives(
    conserved: np.ndarray, params: ModelParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area_f = params.area_full
    area_l = np.clip(conserved[2], 1.0e-8 * area_f, (1.0 - 1.0e-8) * area_f)
    area_g = area_f - area_l
    density_g = np.maximum(conserved[0] / area_g, 1.0e-8)
    velocity_g = conserved[1] / np.maximum(conserved[0], 1.0e-14)
    velocity_l = conserved[3] / area_l
    return density_g, velocity_g, area_l, velocity_l


def _physical_flux(
    density_g: np.ndarray,
    velocity_g: np.ndarray,
    area_l: np.ndarray,
    velocity_l: np.ndarray,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area_g = params.area_full - area_l
    pressure_g = density_g * params.gas_constant * params.gas_temperature
    lambda_d = restoring_coefficient(
        area_l, velocity_l, density_g, velocity_g, params
    )
    flux = np.vstack(
        (
            density_g * area_g * velocity_g,
            density_g * area_g * velocity_g**2 + pressure_g * area_g,
            area_l * velocity_l,
            area_l * velocity_l**2 + 0.5 * lambda_d * area_l**2,
        )
    )
    speed_g = np.abs(velocity_g) + params.gas_sound_speed
    speed_l = np.abs(velocity_l) + np.sqrt(
        np.maximum(lambda_d * area_l, 0.0) + params.c_num**2
    )
    return flux, speed_g, speed_l


def _spatial_operator(
    conserved: np.ndarray, dx: float, params: ModelParameters
) -> np.ndarray:
    density_g, velocity_g, area_l, velocity_l = conserved_to_primitives(
        conserved, params
    )
    primitive = (density_g, velocity_g, area_l, velocity_l)
    left_states = []
    right_states = []
    for values in primitive:
        left, right = _periodic_reconstruction(values, params.limiter_theta)
        left_states.append(left)
        right_states.append(right)

    area_floor = 1.0e-8 * params.area_full
    for states in (left_states, right_states):
        states[0] = np.maximum(states[0], 1.0e-8)
        states[2] = np.clip(
            states[2], area_floor, params.area_full - area_floor
        )

    u_left = primitives_to_conserved(*left_states, params)
    u_right = primitives_to_conserved(*right_states, params)
    flux_left, speed_g_left, speed_l_left = _physical_flux(
        *left_states, params
    )
    flux_right, speed_g_right, speed_l_right = _physical_flux(
        *right_states, params
    )
    speed_g = np.maximum(speed_g_left, speed_g_right)
    speed_l = np.maximum(speed_l_left, speed_l_right)
    dissipation = np.vstack((speed_g, speed_g, speed_l, speed_l))
    face_flux = 0.5 * (flux_left + flux_right) - 0.5 * dissipation * (
        u_right - u_left
    )
    operator = -(face_flux - np.roll(face_flux, 1, axis=1)) / dx

    # Non-conservative geometric terms in Eq. (A17).  The short periodic audit
    # is inviscid, so regular wall/interfacial source terms are intentionally
    # absent; this isolates the IKH mechanism represented by Eq. (A31).
    pressure_g = density_g * params.gas_constant * params.gas_temperature
    area_g = params.area_full - area_l
    d_area_g_dx = (np.roll(area_g, -1) - np.roll(area_g, 1)) / (2.0 * dx)
    depth_l = liquid_depth(area_l, params)
    d_depth_dx = (np.roll(depth_l, -1) - np.roll(depth_l, 1)) / (2.0 * dx)
    operator[1] += (
        pressure_g * d_area_g_dx
        - area_g * density_g * params.gravity * d_depth_dx
    )
    return operator


def stable_time_step(
    conserved: np.ndarray, dx: float, cfl: float, params: ModelParameters
) -> float:
    primitive = conserved_to_primitives(conserved, params)
    _, speed_g, speed_l = _physical_flux(*primitive, params)
    maximum = max(float(np.max(speed_g)), float(np.max(speed_l)), 1.0e-12)
    return cfl * dx / maximum


def advance_ssprk2(
    conserved: np.ndarray, dt: float, dx: float, params: ModelParameters
) -> np.ndarray:
    first = conserved + dt * _spatial_operator(conserved, dx, params)
    first = _enforce_physical_state(first, params)
    second = first + dt * _spatial_operator(first, dx, params)
    return _enforce_physical_state(0.5 * conserved + 0.5 * second, params)


def _enforce_physical_state(
    conserved: np.ndarray, params: ModelParameters
) -> np.ndarray:
    result = conserved.copy()
    area_floor = 1.0e-8 * params.area_full
    result[2] = np.clip(result[2], area_floor, params.area_full - area_floor)
    area_g = params.area_full - result[2]
    minimum_gas_mass = 1.0e-8 * area_g
    repair = result[0] < minimum_gas_mass
    result[0] = np.maximum(result[0], minimum_gas_mass)
    result[1, repair] = 0.0
    return result

