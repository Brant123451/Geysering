"""Conservative post-arrival gas dynamics for the Case-A west branch.

The shock-fitting solver remains responsible for the sharp cavity before it
reaches the side-T.  After arrival, this module advances the horizontal gas
mass and gas momentum instead of rebuilding a prescribed crown-pocket shape.

The evolved gas block is the first two equations of the companion four-
equation model,

    U_g = [A_g rho_g, A_g rho_g u_g],

with an isothermal equation of state.  A MUSCL-HLL finite-volume update and
SSP-RK2 time integration are used.  The acoustic gas block is subcycled on its
own CFL limit, so the much larger liquid/network step is not used for the gas
characteristics.  Quasi-one-dimensional face areas exactly balance the nozzle
pressure term; an actual interface-depth gradient remains a physical source.

Only the west branch (closed upstream wall to side-T) is advanced.  Both end
faces are impermeable in this operator; gas turning into the vertical branch
is a separate, local conservative sink applied by the network junction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - the pure-Python fallback is portable
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate


@dataclass(frozen=True)
class HorizontalGasParameters:
    diameter: float
    rho_l: float = 998.0
    gravity: float = 9.81
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    atmospheric_pressure: float = 101325.0
    gas_viscosity: float = 1.81e-5
    cfl: float = 0.45
    limiter_theta: float = 1.25
    void_floor_fraction: float = 1.0e-4
    active_void_fraction: float = 5.0e-4

    @property
    def area_full(self) -> float:
        return 0.25 * math.pi * self.diameter**2

    @property
    def rho_atmospheric(self) -> float:
        return self.atmospheric_pressure / (
            self.gas_constant * self.gas_temperature
        )

    @property
    def sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.gas_temperature)


@dataclass(frozen=True)
class HorizontalGasAdvance:
    mass: np.ndarray
    momentum: np.ndarray
    liquid_momentum_increment: np.ndarray
    substeps: int
    active_cells: int
    mass_error: float
    kinetic_energy: float
    centre_of_mass: float
    maximum_velocity: float


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(positive, magnitude, np.where(negative, -magnitude, 0.0))


def _limited_slopes(values: np.ndarray, theta: float) -> np.ndarray:
    slope = np.zeros_like(values)
    if values.size < 3:
        return slope
    left = values[1:-1] - values[:-2]
    right = values[2:] - values[1:-1]
    centre = 0.5 * (values[2:] - values[:-2])
    slope[1:-1] = _minmod3(theta * left, centre, theta * right)
    return slope


def _gamma_from_holdup(alpha_l: np.ndarray) -> np.ndarray:
    """Return the liquid wetted angle of a circular segment."""

    alpha = np.clip(np.asarray(alpha_l, dtype=float), 1.0e-10, 1.0 - 1.0e-10)
    lo = np.full_like(alpha, 1.0e-12)
    hi = np.full_like(alpha, 2.0 * math.pi - 1.0e-12)
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        fraction = (mid - np.sin(mid)) / (2.0 * math.pi)
        lo = np.where(fraction < alpha, mid, lo)
        hi = np.where(fraction < alpha, hi, mid)
    return 0.5 * (lo + hi)


def _friction_factor(reynolds: np.ndarray, gas_or_interface: bool) -> np.ndarray:
    re = np.maximum(np.asarray(reynolds, dtype=float), 1.0e-12)
    if gas_or_interface:
        laminar = 16.0 / re
        turbulent = 0.046 * re**-0.2
    else:
        laminar = 24.0 / re
        turbulent = 0.0262 * re**-0.139
    return np.clip(np.where(re < 2100.0, laminar, turbulent), 0.0, 4.0)


def _geometry(
    liquid_area: np.ndarray,
    params: HorizontalGasParameters,
) -> dict[str, np.ndarray]:
    area_full = params.area_full
    liquid = np.clip(liquid_area, 0.0, area_full)
    gas_raw = np.maximum(area_full - liquid, 0.0)
    gas = np.maximum(gas_raw, params.void_floor_fraction * area_full)
    gamma = _gamma_from_holdup(liquid / area_full)
    liquid_depth = 0.5 * params.diameter * (1.0 - np.cos(0.5 * gamma))
    perimeter_l = 0.5 * params.diameter * gamma
    perimeter_g = 0.5 * params.diameter * (2.0 * math.pi - gamma)
    interface_width = np.maximum(
        params.diameter * np.sin(0.5 * gamma), 1.0e-12
    )
    hydraulic_g = 4.0 * gas / np.maximum(
        perimeter_g + interface_width, 1.0e-12
    )
    return {
        "area_gas": gas,
        "area_gas_raw": gas_raw,
        "depth_liquid": liquid_depth,
        "perimeter_gas": perimeter_g,
        "interface_width": interface_width,
        "hydraulic_gas": hydraulic_g,
    }


def _hll_intrinsic_flux(
    rho_left: np.ndarray,
    velocity_left: np.ndarray,
    rho_right: np.ndarray,
    velocity_right: np.ndarray,
    params: HorizontalGasParameters,
) -> np.ndarray:
    """Isothermal Euler HLL flux per unit open gas area."""

    c = params.sound_speed
    rho_atm = params.rho_atmospheric
    pressure_left = (rho_left - rho_atm) * c**2
    pressure_right = (rho_right - rho_atm) * c**2
    conserved_left = np.vstack((rho_left, rho_left * velocity_left))
    conserved_right = np.vstack((rho_right, rho_right * velocity_right))
    flux_left = np.vstack(
        (
            rho_left * velocity_left,
            rho_left * velocity_left**2 + pressure_left,
        )
    )
    flux_right = np.vstack(
        (
            rho_right * velocity_right,
            rho_right * velocity_right**2 + pressure_right,
        )
    )
    speed_left = np.minimum(velocity_left - c, velocity_right - c)
    speed_right = np.maximum(velocity_left + c, velocity_right + c)
    denominator = np.maximum(speed_right - speed_left, 1.0e-12)
    middle = (
        speed_right * flux_left
        - speed_left * flux_right
        + speed_left
        * speed_right
        * (conserved_right - conserved_left)
    ) / denominator
    return np.where(
        (speed_left >= 0.0)[None, :],
        flux_left,
        np.where((speed_right <= 0.0)[None, :], flux_right, middle),
    )


def _source_terms(
    state: np.ndarray,
    liquid_discharge: np.ndarray,
    geometry: dict[str, np.ndarray],
    face_area: np.ndarray,
    dx: float,
    params: HorizontalGasParameters,
) -> tuple[np.ndarray, np.ndarray]:
    gas_area = geometry["area_gas"]
    density = np.maximum(state[0] / gas_area, 1.0e-10)
    velocity_g = state[1] / np.maximum(state[0], 1.0e-14)
    area_l = params.area_full - geometry["area_gas_raw"]
    velocity_l = liquid_discharge / np.maximum(
        area_l, 1.0e-4 * params.area_full
    )
    pressure_gauge = (
        density - params.rho_atmospheric
    ) * params.sound_speed**2

    d_depth_dx = np.gradient(
        geometry["depth_liquid"], dx, edge_order=1
    )
    pressure_area_source = pressure_gauge * (
        face_area[1:] - face_area[:-1]
    ) / dx
    hydrostatic_source = (
        -gas_area * density * params.gravity * d_depth_dx
    )

    re_g = (
        density
        * np.abs(velocity_g)
        * geometry["hydraulic_gas"]
        / params.gas_viscosity
    )
    relative = velocity_g - velocity_l
    re_i = (
        density
        * np.abs(relative)
        * np.maximum(geometry["hydraulic_gas"], 1.0e-12)
        / params.gas_viscosity
    )
    factor_g = _friction_factor(re_g, gas_or_interface=True)
    factor_i = _friction_factor(re_i, gas_or_interface=True)
    wall_force = (
        0.5
        * factor_g
        * density
        * velocity_g
        * np.abs(velocity_g)
        * geometry["perimeter_gas"]
    )
    interface_force = (
        0.5
        * factor_i
        * density
        * relative
        * np.abs(relative)
        * geometry["interface_width"]
    )
    momentum_source = (
        pressure_area_source
        + hydrostatic_source
        - wall_force
        - interface_force
    )
    return momentum_source, interface_force


def _operator(
    state: np.ndarray,
    liquid_discharge: np.ndarray,
    geometry: dict[str, np.ndarray],
    active: np.ndarray,
    dx: float,
    params: HorizontalGasParameters,
) -> tuple[np.ndarray, np.ndarray]:
    gas_area = geometry["area_gas"]
    density = np.maximum(state[0] / gas_area, 1.0e-10)
    velocity = state[1] / np.maximum(state[0], 1.0e-14)

    slope_density = _limited_slopes(density, params.limiter_theta)
    slope_velocity = _limited_slopes(velocity, params.limiter_theta)
    rho_left = np.maximum(
        density[:-1] + 0.5 * slope_density[:-1], 1.0e-10
    )
    rho_right = np.maximum(
        density[1:] - 0.5 * slope_density[1:], 1.0e-10
    )
    velocity_left = velocity[:-1] + 0.5 * slope_velocity[:-1]
    velocity_right = velocity[1:] - 0.5 * slope_velocity[1:]

    intrinsic_flux = _hll_intrinsic_flux(
        rho_left,
        velocity_left,
        rho_right,
        velocity_right,
        params,
    )
    face_area = np.zeros(state.shape[1] + 1)
    face_area[0] = gas_area[0] if active[0] else 0.0
    face_area[-1] = gas_area[-1] if active[-1] else 0.0
    connected = active[:-1] & active[1:]
    face_area[1:-1] = np.where(
        connected, np.minimum(gas_area[:-1], gas_area[1:]), 0.0
    )
    face_flux = np.zeros((2, state.shape[1] + 1))
    face_flux[:, 1:-1] = intrinsic_flux * face_area[1:-1]
    # Reflecting end walls: no gas mass crosses; the gauge-pressure traction is
    # retained so the quasi-1D area source remains exactly well balanced.
    pressure_gauge = (
        density - params.rho_atmospheric
    ) * params.sound_speed**2
    face_flux[1, 0] = pressure_gauge[0] * face_area[0]
    face_flux[1, -1] = pressure_gauge[-1] * face_area[-1]

    rhs = -(face_flux[:, 1:] - face_flux[:, :-1]) / dx
    source_momentum, interface_force = _source_terms(
        state,
        liquid_discharge,
        geometry,
        face_area,
        dx,
        params,
    )
    rhs[1] += source_momentum
    rhs[:, ~active] = 0.0
    interface_force = np.where(active, interface_force, 0.0)
    return rhs, interface_force


def _enforce_state(
    state: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    result = state.copy()
    result[0] = np.maximum(result[0], 0.0)
    empty = result[0] <= 1.0e-14
    result[1, empty | (~active)] = 0.0
    return result


@njit(cache=True)
def _minmod_scalar(a: float, b: float, c: float) -> float:
    if a > 0.0 and b > 0.0 and c > 0.0:
        return min(a, b, c)
    if a < 0.0 and b < 0.0 and c < 0.0:
        return max(a, b, c)
    return 0.0


@njit(cache=True)
def _factor_scalar(reynolds: float) -> float:
    reynolds = max(reynolds, 1.0e-12)
    if reynolds < 2100.0:
        factor = 16.0 / reynolds
    else:
        factor = 0.046 * reynolds**-0.2
    return min(max(factor, 0.0), 4.0)


@njit(cache=True)
def _compiled_rhs(
    state: np.ndarray,
    liquid_discharge: np.ndarray,
    liquid_area: np.ndarray,
    gas_area: np.ndarray,
    depth_liquid: np.ndarray,
    perimeter_gas: np.ndarray,
    interface_width: np.ndarray,
    hydraulic_gas: np.ndarray,
    face_area: np.ndarray,
    active: np.ndarray,
    dx: float,
    rho_l: float,
    gravity: float,
    gas_viscosity: float,
    sound_speed: float,
    rho_atmospheric: float,
    limiter_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = state.shape[1]
    density = np.empty(n)
    velocity = np.empty(n)
    velocity_l = np.empty(n)
    slope_density = np.zeros(n)
    slope_velocity = np.zeros(n)
    for i in range(n):
        density[i] = max(state[0, i] / gas_area[i], 1.0e-10)
        velocity[i] = state[1, i] / max(state[0, i], 1.0e-14)
        velocity_l[i] = liquid_discharge[i] / max(liquid_area[i], 1.0e-14)
    for i in range(1, n - 1):
        dl_rho = density[i] - density[i - 1]
        dr_rho = density[i + 1] - density[i]
        dc_rho = 0.5 * (density[i + 1] - density[i - 1])
        slope_density[i] = _minmod_scalar(
            limiter_theta * dl_rho,
            dc_rho,
            limiter_theta * dr_rho,
        )
        dl_u = velocity[i] - velocity[i - 1]
        dr_u = velocity[i + 1] - velocity[i]
        dc_u = 0.5 * (velocity[i + 1] - velocity[i - 1])
        slope_velocity[i] = _minmod_scalar(
            limiter_theta * dl_u,
            dc_u,
            limiter_theta * dr_u,
        )

    flux = np.zeros((2, n + 1))
    pressure_left_wall = (
        density[0] - rho_atmospheric
    ) * sound_speed * sound_speed
    pressure_right_wall = (
        density[n - 1] - rho_atmospheric
    ) * sound_speed * sound_speed
    flux[1, 0] = pressure_left_wall * face_area[0]
    flux[1, n] = pressure_right_wall * face_area[n]
    for face in range(1, n):
        if face_area[face] <= 0.0:
            continue
        left = face - 1
        right = face
        rho_left = max(
            density[left] + 0.5 * slope_density[left], 1.0e-10
        )
        rho_right = max(
            density[right] - 0.5 * slope_density[right], 1.0e-10
        )
        u_left = velocity[left] + 0.5 * slope_velocity[left]
        u_right = velocity[right] - 0.5 * slope_velocity[right]
        p_left = (rho_left - rho_atmospheric) * sound_speed * sound_speed
        p_right = (rho_right - rho_atmospheric) * sound_speed * sound_speed
        f_mass_left = rho_left * u_left
        f_mass_right = rho_right * u_right
        f_momentum_left = rho_left * u_left * u_left + p_left
        f_momentum_right = rho_right * u_right * u_right + p_right
        speed_left = min(u_left - sound_speed, u_right - sound_speed)
        speed_right = max(u_left + sound_speed, u_right + sound_speed)
        if speed_left >= 0.0:
            f_mass = f_mass_left
            f_momentum = f_momentum_left
        elif speed_right <= 0.0:
            f_mass = f_mass_right
            f_momentum = f_momentum_right
        else:
            denominator = max(speed_right - speed_left, 1.0e-12)
            f_mass = (
                speed_right * f_mass_left
                - speed_left * f_mass_right
                + speed_left * speed_right * (rho_right - rho_left)
            ) / denominator
            f_momentum = (
                speed_right * f_momentum_left
                - speed_left * f_momentum_right
                + speed_left
                * speed_right
                * (rho_right * u_right - rho_left * u_left)
            ) / denominator
        flux[0, face] = f_mass * face_area[face]
        flux[1, face] = f_momentum * face_area[face]

    rhs = np.zeros((2, n))
    interface_force = np.zeros(n)
    for i in range(n):
        if not active[i]:
            continue
        rhs[0, i] = -(flux[0, i + 1] - flux[0, i]) / dx
        rhs[1, i] = -(flux[1, i + 1] - flux[1, i]) / dx
        if i == 0:
            depth_gradient = (depth_liquid[1] - depth_liquid[0]) / dx
        elif i == n - 1:
            depth_gradient = (
                depth_liquid[n - 1] - depth_liquid[n - 2]
            ) / dx
        else:
            depth_gradient = (
                depth_liquid[i + 1] - depth_liquid[i - 1]
            ) / (2.0 * dx)
        pressure_gauge = (
            density[i] - rho_atmospheric
        ) * sound_speed * sound_speed
        pressure_area_source = pressure_gauge * (
            face_area[i + 1] - face_area[i]
        ) / dx
        hydrostatic_source = (
            -gas_area[i] * density[i] * gravity * depth_gradient
        )
        reynolds_g = (
            density[i]
            * abs(velocity[i])
            * hydraulic_gas[i]
            / gas_viscosity
        )
        relative = velocity[i] - velocity_l[i]
        reynolds_i = (
            density[i]
            * abs(relative)
            * hydraulic_gas[i]
            / gas_viscosity
        )
        wall_force = (
            0.5
            * _factor_scalar(reynolds_g)
            * density[i]
            * velocity[i]
            * abs(velocity[i])
            * perimeter_gas[i]
        )
        interface_force[i] = (
            0.5
            * _factor_scalar(reynolds_i)
            * density[i]
            * relative
            * abs(relative)
            * interface_width[i]
        )
        rhs[1, i] += (
            pressure_area_source
            + hydrostatic_source
            - wall_force
            - interface_force[i]
        )
    return rhs, interface_force


@njit(cache=True)
def _compiled_advance(
    state: np.ndarray,
    liquid_discharge: np.ndarray,
    liquid_area: np.ndarray,
    gas_area: np.ndarray,
    depth_liquid: np.ndarray,
    perimeter_gas: np.ndarray,
    interface_width: np.ndarray,
    hydraulic_gas: np.ndarray,
    face_area: np.ndarray,
    active: np.ndarray,
    dx: float,
    dt: float,
    rho_l: float,
    gravity: float,
    gas_viscosity: float,
    sound_speed: float,
    rho_atmospheric: float,
    cfl: float,
    limiter_theta: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n = state.shape[1]
    liquid_impulse = np.zeros(n)
    elapsed = 0.0
    substeps = 0
    maximum_velocity = 0.0
    while elapsed < dt - 1.0e-15:
        for i in range(n):
            if active[i]:
                velocity = abs(state[1, i] / max(state[0, i], 1.0e-14))
                maximum_velocity = max(maximum_velocity, velocity)
        stable = cfl * dx / max(sound_speed + maximum_velocity, 1.0e-12)
        sub_dt = min(stable, dt - elapsed)
        rhs_first, interface_first = _compiled_rhs(
            state,
            liquid_discharge,
            liquid_area,
            gas_area,
            depth_liquid,
            perimeter_gas,
            interface_width,
            hydraulic_gas,
            face_area,
            active,
            dx,
            rho_l,
            gravity,
            gas_viscosity,
            sound_speed,
            rho_atmospheric,
            limiter_theta,
        )
        first = state + sub_dt * rhs_first
        for i in range(n):
            first[0, i] = max(first[0, i], 0.0)
            if first[0, i] <= 1.0e-14 or not active[i]:
                first[1, i] = 0.0
        rhs_second, interface_second = _compiled_rhs(
            first,
            liquid_discharge,
            liquid_area,
            gas_area,
            depth_liquid,
            perimeter_gas,
            interface_width,
            hydraulic_gas,
            face_area,
            active,
            dx,
            rho_l,
            gravity,
            gas_viscosity,
            sound_speed,
            rho_atmospheric,
            limiter_theta,
        )
        second = first + sub_dt * rhs_second
        for i in range(n):
            second[0, i] = max(second[0, i], 0.0)
            if second[0, i] <= 1.0e-14 or not active[i]:
                second[1, i] = 0.0
            state[0, i] = max(
                0.5 * state[0, i] + 0.5 * second[0, i], 0.0
            )
            state[1, i] = 0.5 * state[1, i] + 0.5 * second[1, i]
            if state[0, i] <= 1.0e-14 or not active[i]:
                state[1, i] = 0.0
            liquid_impulse[i] += (
                0.5
                * sub_dt
                * (interface_first[i] + interface_second[i])
                / rho_l
            )
        elapsed += sub_dt
        substeps += 1
    return state, liquid_impulse, substeps, maximum_velocity


def advance_horizontal_gas(
    mass: np.ndarray,
    momentum: np.ndarray,
    liquid_area: np.ndarray,
    liquid_discharge: np.ndarray,
    x: np.ndarray,
    dx: float,
    dt: float,
    junction_index: int,
    params: HorizontalGasParameters,
) -> HorizontalGasAdvance:
    """Advance the physical gas state from the west wall to the side-T.

    ``mass`` and ``momentum`` are cell-integrated quantities.  The returned
    liquid momentum increment is the equal-and-opposite interfacial drag impulse
    for the same cells; the network adds it to the liquid discharge equation.
    """

    n_west = int(junction_index) + 1
    if n_west < 2:
        raise ValueError("the west branch must contain at least two cells")
    if not (
        mass.shape == momentum.shape == liquid_area.shape
        == liquid_discharge.shape == x.shape
    ):
        raise ValueError("all horizontal arrays must have the same shape")

    gas_mass = np.maximum(np.asarray(mass[:n_west], dtype=float), 0.0)
    gas_momentum = np.asarray(momentum[:n_west], dtype=float).copy()
    area_l = np.asarray(liquid_area[:n_west], dtype=float)
    discharge_l = np.asarray(liquid_discharge[:n_west], dtype=float)
    geometry = _geometry(area_l, params)

    inactive_mass_scale = (
        params.rho_atmospheric
        * params.void_floor_fraction
        * params.area_full
        * dx
    )
    active = (
        geometry["area_gas_raw"]
        > params.active_void_fraction * params.area_full
    ) | (gas_mass > 4.0 * inactive_mass_scale)

    state = np.vstack((gas_mass / dx, gas_momentum / dx))
    state = _enforce_state(state, active)
    initial_mass = float(np.sum(state[0]) * dx)
    face_area = np.zeros(n_west + 1)
    face_area[0] = geometry["area_gas"][0] if active[0] else 0.0
    face_area[-1] = geometry["area_gas"][-1] if active[-1] else 0.0
    face_area[1:-1] = np.where(
        active[:-1] & active[1:],
        np.minimum(
            geometry["area_gas"][:-1], geometry["area_gas"][1:]
        ),
        0.0,
    )
    state, liquid_impulse, substeps, maximum_velocity = _compiled_advance(
        state,
        discharge_l,
        params.area_full - geometry["area_gas_raw"],
        geometry["area_gas"],
        geometry["depth_liquid"],
        geometry["perimeter_gas"],
        geometry["interface_width"],
        geometry["hydraulic_gas"],
        face_area,
        active,
        dx,
        dt,
        params.rho_l,
        params.gravity,
        params.gas_viscosity,
        params.sound_speed,
        params.rho_atmospheric,
        params.cfl,
        params.limiter_theta,
    )

    final_mass = float(np.sum(state[0]) * dx)
    mass_out = np.asarray(mass, dtype=float).copy()
    momentum_out = np.asarray(momentum, dtype=float).copy()
    mass_out[:n_west] = state[0] * dx
    momentum_out[:n_west] = state[1] * dx

    velocity = state[1] / np.maximum(state[0], 1.0e-14)
    kinetic_energy = float(
        np.sum(0.5 * state[0] * velocity**2) * dx
    )
    resolved = np.maximum(state[0], 0.0) * dx
    resolved_total = float(np.sum(resolved))
    centre_of_mass = (
        float(np.sum(resolved * x[:n_west]) / resolved_total)
        if resolved_total > 0.0
        else float(x[0])
    )
    liquid_increment = np.zeros_like(liquid_discharge, dtype=float)
    liquid_increment[:n_west] = liquid_impulse
    return HorizontalGasAdvance(
        mass=mass_out,
        momentum=momentum_out,
        liquid_momentum_increment=liquid_increment,
        substeps=substeps,
        active_cells=int(np.count_nonzero(active)),
        mass_error=final_mass - initial_mass,
        kinetic_energy=kinetic_energy,
        centre_of_mass=centre_of_mass,
        maximum_velocity=maximum_velocity,
    )


__all__ = [
    "HorizontalGasAdvance",
    "HorizontalGasParameters",
    "advance_horizontal_gas",
]
