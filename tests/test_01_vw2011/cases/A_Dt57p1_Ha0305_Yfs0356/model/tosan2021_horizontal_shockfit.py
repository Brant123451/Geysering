"""Tosan (2021) Chapter-6 horizontal mixed-flow core for VW2011 Case B.

This module is intentionally independent of ``vw2011_network_twofluid.py``.
It contains four pieces that can be tested and coupled separately:

1. the exact circular-section Saint-Venant geometry in the free-surface
   branch and a continuous elastic continuation above the full area;
2. the uniform polytropic gas inventory, ``p_abs * V**gamma = constant``;
3. the explicitly tracked Tosan shock position,
   ``L(t + dt) = L(t) + w(t) * dt``, with the positive-interface equations
   (Tosan 2021, Eqs. 6-45--6-47);
4. a Case-B extension for the initially *truly dry* upstream chamber.

The dry-chamber extension is not part of Tosan's original Chapter-6 test,
which starts from a pre-wetted free-surface reach.  Here it is advanced by a
positivity-preserving MUSCL central-upwind finite-volume flux with SSP-RK2
time integration and a donor-cell draining limiter.  Zero area is a valid
state: no artificial water film is inserted.  The extension is deliberately
isolated in :func:`central_upwind_wet_dry_step` so that it cannot be mistaken
for the paper's shock-fitting equations.

The Case-B topology has a free-surface/gas region on the *left* and a
pressurised water column on the *right*.  :func:`solve_oriented_interface`
therefore supplies the reflected characteristic needed by that orientation,
while :func:`solve_tosan_positive_interface` remains a direct implementation
of Eqs. 6-45--6-47 and Appendix-C ``flxT4``.

The class at the bottom is a horizontal-core adapter, not a tower/jet model.
After the tracked front reaches ``vent_x``, a caller may relax the uniform gas
pressure through ``vent_pressure_hook`` or ``external_pressure_abs``.  Gas
mass loss and vertical-riser dynamics belong in the network adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np


Array = np.ndarray
BoundaryKind = Literal["wall", "transmissive"]
PressurisedSide = Literal["left", "right"]
VentPressureHook = Callable[[float, float, float], float]


@dataclass(frozen=True)
class HorizontalConfig:
    """Numerical and physical constants for the independent horizontal core."""

    length: float = 4.006
    diameter: float = 0.094
    valve_x: float = 0.546
    vent_x: float | None = 3.516
    initial_air_head: float = 0.610
    initial_water_head: float = 0.356
    dx: float = 0.010
    gravity: float = 9.81
    liquid_density: float = 998.2
    atmospheric_pressure: float = 101_325.0
    gas_constant: float = 287.05
    temperature: float = 293.15
    gamma: float = 1.4
    wave_speed: float = 100.0
    manning_n: float = 0.009
    darcy_friction: float = 0.018
    gravity_current_froude: float = 0.48
    bed_slope: float = 0.0
    cfl: float = 0.90
    dry_area_fraction: float = 1.0e-10
    wetting_front_report_fraction: float = 1.0e-3
    nonlinear_tolerance: float = 1.0e-10
    nonlinear_max_iterations: int = 40
    left_boundary: BoundaryKind = "wall"
    right_boundary: BoundaryKind = "wall"

    def __post_init__(self) -> None:
        positive = {
            "length": self.length,
            "diameter": self.diameter,
            "dx": self.dx,
            "gravity": self.gravity,
            "liquid_density": self.liquid_density,
            "atmospheric_pressure": self.atmospheric_pressure,
            "gas_constant": self.gas_constant,
            "temperature": self.temperature,
            "gamma": self.gamma,
            "wave_speed": self.wave_speed,
            "gravity_current_froude": self.gravity_current_froude,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not (0.0 <= self.valve_x <= self.length):
            raise ValueError("valve_x must lie inside the horizontal domain")
        if self.vent_x is not None and not (0.0 <= self.vent_x <= self.length):
            raise ValueError("vent_x must lie inside the horizontal domain")
        if not np.isfinite(self.initial_air_head):
            raise ValueError("initial_air_head must be finite")
        if not np.isfinite(self.initial_water_head) or self.initial_water_head <= 0.0:
            raise ValueError("initial_water_head must be positive and finite")
        if not (0.0 < self.cfl < 1.0):
            raise ValueError("cfl must lie in (0, 1)")
        if not (0.0 <= self.dry_area_fraction < 1.0):
            raise ValueError("dry_area_fraction must lie in [0, 1)")
        if not (0.0 < self.wetting_front_report_fraction < 1.0):
            raise ValueError("wetting_front_report_fraction must lie in (0, 1)")


class CircularSection:
    """Circular Saint-Venant geometry with an elastic full-pipe continuation."""

    def __init__(
        self,
        diameter: float,
        gravity: float = 9.81,
        wave_speed: float = 100.0,
    ) -> None:
        if diameter <= 0.0 or gravity <= 0.0 or wave_speed <= 0.0:
            raise ValueError("diameter, gravity and wave_speed must be positive")
        self.diameter = float(diameter)
        self.gravity = float(gravity)
        self.wave_speed = float(wave_speed)
        self.full_area = np.pi * self.diameter**2 / 4.0
        self.full_hydrostatic_moment = np.pi * self.diameter**3 / 8.0
        self._depth_lookup = np.linspace(0.0, self.diameter, 8193)
        lookup_theta = 2.0 * np.arccos(
            np.clip(
                1.0 - 2.0 * self._depth_lookup / self.diameter,
                -1.0,
                1.0,
            )
        )
        self._area_lookup = self.diameter**2 / 8.0 * (
            lookup_theta - np.sin(lookup_theta)
        )
        self._area_lookup[0] = 0.0
        self._area_lookup[-1] = self.full_area

    @staticmethod
    def _return_like_input(value: Array, original: object) -> float | Array:
        if np.ndim(original) == 0:
            return float(np.asarray(value))
        return np.asarray(value)

    def area_from_depth(self, depth: float | Sequence[float] | Array) -> float | Array:
        """Return wetted area ``A(h)`` for ``0 <= h <= D``."""

        original = depth
        h = np.clip(np.asarray(depth, dtype=float), 0.0, self.diameter)
        theta = 2.0 * np.arccos(
            np.clip(1.0 - 2.0 * h / self.diameter, -1.0, 1.0)
        )
        area = self.diameter**2 / 8.0 * (theta - np.sin(theta))
        area = np.where(h <= 0.0, 0.0, area)
        area = np.where(h >= self.diameter, self.full_area, area)
        return self._return_like_input(area, original)

    def depth_from_area(self, area: float | Sequence[float] | Array) -> float | Array:
        """Invert ``A(h)`` without a tabulated film or endpoint singularity."""

        original = area
        target = np.clip(np.asarray(area, dtype=float), 0.0, self.full_area)
        lo = np.zeros_like(target)
        hi = np.full_like(target, self.diameter)
        for _ in range(56):
            mid = 0.5 * (lo + hi)
            amid = np.asarray(self.area_from_depth(mid))
            lo = np.where(amid < target, mid, lo)
            hi = np.where(amid >= target, mid, hi)
        depth = 0.5 * (lo + hi)
        depth = np.where(target <= 0.0, 0.0, depth)
        depth = np.where(target >= self.full_area, self.diameter, depth)
        return self._return_like_input(depth, original)

    def _fast_depth_from_area(self, area: Array) -> Array:
        """Vectorised lookup used inside time-stepping kernels."""

        return np.interp(
            np.clip(area, 0.0, self.full_area),
            self._area_lookup,
            self._depth_lookup,
        )

    def top_width(self, depth: float | Sequence[float] | Array) -> float | Array:
        """Free-surface top width of the circular segment."""

        original = depth
        h = np.clip(np.asarray(depth, dtype=float), 0.0, self.diameter)
        width = 2.0 * np.sqrt(np.maximum(h * (self.diameter - h), 0.0))
        return self._return_like_input(width, original)

    def wetted_perimeter(
        self, depth: float | Sequence[float] | Array
    ) -> float | Array:
        original = depth
        h = np.clip(np.asarray(depth, dtype=float), 0.0, self.diameter)
        theta = 2.0 * np.arccos(
            np.clip(1.0 - 2.0 * h / self.diameter, -1.0, 1.0)
        )
        perimeter = 0.5 * self.diameter * theta
        return self._return_like_input(perimeter, original)

    def hydrostatic_moment(
        self, depth: float | Sequence[float] | Array
    ) -> float | Array:
        """Return ``I1 = integral_0^h (h-y)b(y)dy`` in m^3.

        This is the circular-section force moment used in Tosan Eq. (6-48);
        the momentum-flux contribution per unit density is ``g*I1``.
        """

        original = depth
        d = self.diameter
        h = np.clip(np.asarray(depth, dtype=float), 0.0, d)
        root = np.sqrt(np.maximum(h * (d - h), 0.0))
        ratio = np.divide(
            h,
            np.maximum(d - h, np.finfo(float).tiny),
            out=np.zeros_like(h),
            where=h > 0.0,
        )
        angle = np.arctan(np.sqrt(ratio))
        moment = (
            (3.0 * d**2 - 4.0 * d * h + 4.0 * h**2) * root
            - 3.0 * d**2 * (d - 2.0 * h) * angle
        ) / 12.0
        moment = np.where(h <= 0.0, 0.0, moment)
        moment = np.where(h >= d, self.full_hydrostatic_moment, moment)
        return self._return_like_input(moment, original)

    def hydraulic_radius(
        self, area: float | Sequence[float] | Array
    ) -> float | Array:
        original = area
        a = np.clip(np.asarray(area, dtype=float), 0.0, self.full_area)
        h = self._fast_depth_from_area(a)
        perimeter = np.asarray(self.wetted_perimeter(h))
        radius = np.divide(
            a,
            perimeter,
            out=np.zeros_like(a),
            where=perimeter > 0.0,
        )
        return self._return_like_input(radius, original)

    def head_from_area(self, area: float | Sequence[float] | Array) -> float | Array:
        """Piezometric depth/head for free and elastically full states.

        Above the full area, the linear water-hammer storage relation is

        ``A/Af = 1 + g(H-D)/a^2``.
        """

        original = area
        a = np.maximum(np.asarray(area, dtype=float), 0.0)
        free_depth = np.asarray(self.depth_from_area(np.minimum(a, self.full_area)))
        elastic_head = self.diameter + self.wave_speed**2 / self.gravity * (
            a / self.full_area - 1.0
        )
        head = np.where(a <= self.full_area, free_depth, elastic_head)
        return self._return_like_input(head, original)

    def area_from_head(self, head: float | Sequence[float] | Array) -> float | Array:
        """Inverse of :meth:`head_from_area`."""

        original = head
        h = np.asarray(head, dtype=float)
        free_area = np.asarray(self.area_from_depth(np.clip(h, 0.0, self.diameter)))
        elastic_area = self.full_area * (
            1.0 + self.gravity * (h - self.diameter) / self.wave_speed**2
        )
        area = np.where(h <= self.diameter, free_area, elastic_area)
        area = np.maximum(area, 0.0)
        return self._return_like_input(area, original)

    def pressure_flux(self, area: float | Sequence[float] | Array) -> float | Array:
        """Conservative pressure term in the discharge-momentum flux."""

        original = area
        a = np.maximum(np.asarray(area, dtype=float), 0.0)
        depth = self._fast_depth_from_area(np.minimum(a, self.full_area))
        free_flux = self.gravity * np.asarray(self.hydrostatic_moment(depth))
        elastic_flux = (
            self.gravity * self.full_hydrostatic_moment
            + 0.5
            * self.wave_speed**2
            * (a**2 - self.full_area**2)
            / self.full_area
        )
        flux = np.where(a <= self.full_area, free_flux, elastic_flux)
        return self._return_like_input(flux, original)

    def celerity(self, area: float | Sequence[float] | Array) -> float | Array:
        """Return circular gravity-wave or elastic water-hammer celerity."""

        original = area
        a = np.maximum(np.asarray(area, dtype=float), 0.0)
        depth = self._fast_depth_from_area(np.minimum(a, self.full_area))
        width = np.asarray(self.top_width(depth))
        free_c2 = np.divide(
            self.gravity * a,
            width,
            out=np.zeros_like(a),
            where=width > 0.0,
        )
        free_c = np.minimum(np.sqrt(np.maximum(free_c2, 0.0)), self.wave_speed)
        elastic_c = self.wave_speed * np.sqrt(
            np.maximum(a / self.full_area, 1.0)
        )
        c = np.where(a < self.full_area, free_c, elastic_c)
        c = np.where(a <= 0.0, 0.0, c)
        return self._return_like_input(c, original)

    def free_surface_celerity_from_depth(
        self, depth: float | Sequence[float] | Array
    ) -> float | Array:
        """Uncapped circular Saint-Venant celerity ``sqrt(gA/T)``.

        The value tends to infinity as an exactly full free surface closes.
        Callers handling a dry-bed Riemann problem should use the limiting
        integral, not reinterpret that endpoint as an elastic acoustic state.
        """

        original = depth
        h = np.clip(np.asarray(depth, dtype=float), 0.0, self.diameter)
        area = np.asarray(self.area_from_depth(h))
        width = np.asarray(self.top_width(h))
        c2 = np.divide(
            self.gravity * area,
            width,
            out=np.full_like(area, np.inf),
            where=width > 0.0,
        )
        c = np.sqrt(np.maximum(c2, 0.0))
        c = np.where(h <= 0.0, 0.0, c)
        return self._return_like_input(c, original)


@dataclass(frozen=True)
class PolytropicGasInventory:
    """Spatially uniform, closed gas pocket with an absolute-pressure EOS."""

    reference_volume: float
    reference_pressure_abs: float
    gamma: float
    volume: float
    mass: float

    @classmethod
    def from_gauge_head(
        cls,
        *,
        volume: float,
        gauge_head: float,
        atmospheric_pressure: float,
        liquid_density: float,
        gravity: float,
        gamma: float,
        gas_constant: float = 287.05,
        temperature: float = 293.15,
    ) -> "PolytropicGasInventory":
        if volume <= 0.0:
            raise ValueError("gas volume must be positive")
        pressure_abs = atmospheric_pressure + liquid_density * gravity * gauge_head
        if pressure_abs <= 0.0:
            raise ValueError("absolute gas pressure must be positive")
        mass = pressure_abs * volume / (gas_constant * temperature)
        return cls(
            reference_volume=float(volume),
            reference_pressure_abs=float(pressure_abs),
            gamma=float(gamma),
            volume=float(volume),
            mass=float(mass),
        )

    @property
    def invariant(self) -> float:
        return self.reference_pressure_abs * self.reference_volume**self.gamma

    @property
    def pressure_abs(self) -> float:
        return self.invariant / self.volume**self.gamma

    def pressure_head_gauge(
        self,
        atmospheric_pressure: float,
        liquid_density: float,
        gravity: float,
    ) -> float:
        return (self.pressure_abs - atmospheric_pressure) / (
            liquid_density * gravity
        )

    def with_volume(self, volume: float) -> "PolytropicGasInventory":
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("gas volume must remain positive")
        return replace(self, volume=float(volume))


@dataclass(frozen=True)
class WetDryState:
    """Cell averages for the zero-depth-capable finite-volume extension."""

    area: Array
    discharge: Array

    def __post_init__(self) -> None:
        area = np.asarray(self.area, dtype=float)
        discharge = np.asarray(self.discharge, dtype=float)
        if area.ndim != 1 or discharge.ndim != 1 or area.shape != discharge.shape:
            raise ValueError("area and discharge must be equal-length 1-D arrays")
        if np.any(~np.isfinite(area)) or np.any(~np.isfinite(discharge)):
            raise ValueError("state arrays must be finite")
        if np.any(area < 0.0):
            raise ValueError("area cannot be negative")


def _ghost_state(
    area: float,
    discharge: float,
    boundary: BoundaryKind,
) -> tuple[float, float]:
    if boundary == "wall":
        return area, -discharge
    if boundary == "transmissive":
        return area, discharge
    raise ValueError(f"unsupported boundary kind: {boundary}")


def _minmod3(a: Array, b: Array, c: Array) -> Array:
    """Return the componentwise three-argument minmod slope."""

    same_sign = (a * b > 0.0) & (a * c > 0.0)
    return np.where(
        same_sign,
        np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))),
        0.0,
    )


def _muscl_free_surface_face_states(
    area_ext: Array,
    discharge_ext: Array,
    section: CircularSection,
    dry_area: float,
) -> tuple[Array, Array, Array, Array]:
    """Reconstruct positive second-order states at all FV faces.

    Standard minmod (theta=1) is applied to liquid area and velocity.  Any
    stencil touching a dry cell or the elastic/full-pipe branch is reduced
    locally to its cell average.  Thus the resolved wet free-surface reach is
    second order without extrapolating water into a dry cell or smearing a
    fitted free-surface/pressurised interface.
    """

    area = np.asarray(area_ext, dtype=float)
    discharge = np.asarray(discharge_ext, dtype=float)
    if area.ndim != 1 or area.shape != discharge.shape or area.size < 3:
        raise ValueError("extended MUSCL states must be equal 1-D arrays")

    wet = area > dry_area
    velocity = np.divide(
        discharge,
        area,
        out=np.zeros_like(discharge),
        where=wet,
    )
    slope_area = np.zeros_like(area)
    slope_velocity = np.zeros_like(velocity)

    delta_area_left = area[1:-1] - area[:-2]
    delta_area_right = area[2:] - area[1:-1]
    delta_velocity_left = velocity[1:-1] - velocity[:-2]
    delta_velocity_right = velocity[2:] - velocity[1:-1]
    # The two ghost cells are deliberately piecewise constant.  A physical
    # cell is reconstructed only when its complete three-cell stencil belongs
    # to the wet free-surface family.
    free_stencil = (
        (area[:-2] > dry_area)
        & (area[1:-1] > dry_area)
        & (area[2:] > dry_area)
        & (area[:-2] < section.full_area)
        & (area[1:-1] < section.full_area)
        & (area[2:] < section.full_area)
    )
    limited_area = _minmod3(
        delta_area_left,
        0.5 * (delta_area_left + delta_area_right),
        delta_area_right,
    )
    limited_velocity = _minmod3(
        delta_velocity_left,
        0.5 * (delta_velocity_left + delta_velocity_right),
        delta_velocity_right,
    )
    slope_area[1:-1] = np.where(free_stencil, limited_area, 0.0)
    slope_velocity[1:-1] = np.where(
        free_stencil, limited_velocity, 0.0
    )

    # Rescale the area slope rather than clipping either face independently;
    # this preserves the cell average.  Keeping reconstructed free-surface
    # states one floating-point step below Af also prevents an accidental
    # switch to the elastic acoustic family at a smooth crest.
    upper_free_area = np.nextafter(section.full_area, 0.0)
    lower_face = area - 0.5 * slope_area
    upper_face = area + 0.5 * slope_area
    face_min = np.minimum(lower_face, upper_face)
    face_max = np.maximum(lower_face, upper_face)
    scale = np.ones_like(area)
    below = face_min < 0.0
    scale[below] = np.minimum(
        scale[below],
        np.divide(
            area[below],
            area[below] - face_min[below],
            out=np.zeros_like(area[below]),
            where=area[below] > face_min[below],
        ),
    )
    above = face_max > upper_free_area
    scale[above] = np.minimum(
        scale[above],
        np.divide(
            upper_free_area - area[above],
            face_max[above] - area[above],
            out=np.zeros_like(area[above]),
            where=face_max[above] > area[above],
        ),
    )
    slope_area *= np.clip(scale, 0.0, 1.0)

    area_left = np.maximum(area[:-1] + 0.5 * slope_area[:-1], 0.0)
    area_right = np.maximum(area[1:] - 0.5 * slope_area[1:], 0.0)
    velocity_left = velocity[:-1] + 0.5 * slope_velocity[:-1]
    velocity_right = velocity[1:] - 0.5 * slope_velocity[1:]
    discharge_left = area_left * velocity_left
    discharge_right = area_right * velocity_right
    discharge_left[area_left <= dry_area] = 0.0
    discharge_right[area_right <= dry_area] = 0.0
    return area_left, discharge_left, area_right, discharge_right


def _apply_donor_draining_limiter(
    mass_flux: Array,
    momentum_flux: Array,
    area: Array,
    *,
    dx: float,
    dt: float,
) -> tuple[Array, Array]:
    """Limit every face by the water volume owned by its physical donor."""

    mass = np.asarray(mass_flux, dtype=float).copy()
    momentum = np.asarray(momentum_flux, dtype=float).copy()
    cell_area = np.asarray(area, dtype=float)
    if mass.shape != momentum.shape or mass.shape != (cell_area.size + 1,):
        raise ValueError("face flux arrays must have one more entry than area")

    outgoing = np.maximum(mass[1:], 0.0) + np.maximum(-mass[:-1], 0.0)
    theta = np.ones(cell_area.size)
    draining = outgoing > 0.0
    theta[draining] = np.minimum(
        1.0,
        cell_area[draining] * dx / (dt * outgoing[draining]),
    )

    face_scale = np.ones(cell_area.size + 1)
    internal_flux = mass[1:-1]
    internal_faces = np.arange(1, cell_area.size)
    donor = np.where(internal_flux >= 0.0, internal_faces - 1, internal_faces)
    face_scale[1:-1] = theta[donor]
    # The previous limiter covered only internal faces.  A transmissive/custom
    # boundary can also drain a physical cell, so its paired flux must use the
    # same donor factor.  Boundary inflow is supplied by the ghost reservoir
    # and is not limited by a physical cell.
    if mass[0] < 0.0:
        face_scale[0] = theta[0]
    if mass[-1] > 0.0:
        face_scale[-1] = theta[-1]
    return mass * face_scale, momentum * face_scale


def _central_upwind_flux(
    area_left: Array,
    discharge_left: Array,
    area_right: Array,
    discharge_right: Array,
    section: CircularSection,
    dry_area: float,
) -> tuple[Array, Array]:
    """Local central-upwind flux for ``U=(A,Q)``."""

    wet_left = area_left > dry_area
    wet_right = area_right > dry_area
    velocity_left = np.divide(
        discharge_left,
        area_left,
        out=np.zeros_like(discharge_left),
        where=wet_left,
    )
    velocity_right = np.divide(
        discharge_right,
        area_right,
        out=np.zeros_like(discharge_right),
        where=wet_right,
    )
    c_left = np.asarray(section.celerity(area_left))
    c_right = np.asarray(section.celerity(area_right))
    # At a genuine wet/dry face the full-pipe acoustic branch must not leak
    # liquid into the dry reach at the water-hammer speed.  The dry-front
    # extension is a gravity-current Riemann problem, so its one-sided signal
    # speed is based on the local hydrostatic depth.  This follows from the
    # dry-bed Saint-Venant limit and introduces no fitted bore coefficient.
    dry_left = ~wet_left
    dry_right = ~wet_right
    wet_dry_face = dry_left ^ dry_right
    gravity_c_left = np.sqrt(
        section.gravity
        * section._fast_depth_from_area(
            np.minimum(area_left, section.full_area)
        )
    )
    gravity_c_right = np.sqrt(
        section.gravity
        * section._fast_depth_from_area(
            np.minimum(area_right, section.full_area)
        )
    )
    c_left = np.where(wet_dry_face & wet_left, gravity_c_left, c_left)
    c_right = np.where(wet_dry_face & wet_right, gravity_c_right, c_right)

    speed_minus = np.minimum(
        np.minimum(velocity_left - c_left, velocity_right - c_right), 0.0
    )
    speed_plus = np.maximum(
        np.maximum(velocity_left + c_left, velocity_right + c_right), 0.0
    )

    flux_mass_left = discharge_left
    flux_mass_right = discharge_right
    flux_momentum_left = np.divide(
        discharge_left**2,
        area_left,
        out=np.zeros_like(discharge_left),
        where=wet_left,
    ) + np.asarray(section.pressure_flux(area_left))
    flux_momentum_right = np.divide(
        discharge_right**2,
        area_right,
        out=np.zeros_like(discharge_right),
        where=wet_right,
    ) + np.asarray(section.pressure_flux(area_right))

    denominator = speed_plus - speed_minus
    active = denominator > np.finfo(float).eps
    mass = np.zeros_like(denominator)
    momentum = np.zeros_like(denominator)
    mass[active] = (
        speed_plus[active] * flux_mass_left[active]
        - speed_minus[active] * flux_mass_right[active]
        + speed_plus[active]
        * speed_minus[active]
        * (area_right[active] - area_left[active])
    ) / denominator[active]
    momentum[active] = (
        speed_plus[active] * flux_momentum_left[active]
        - speed_minus[active] * flux_momentum_right[active]
        + speed_plus[active]
        * speed_minus[active]
        * (discharge_right[active] - discharge_left[active])
    ) / denominator[active]
    return mass, momentum


def _central_upwind_wet_dry_euler_step(
    state: WetDryState,
    *,
    dx: float,
    dt: float,
    section: CircularSection,
    cfl: float = 0.45,
    dry_area_fraction: float = 1.0e-10,
    manning_n: float = 0.0,
    darcy_friction: float = 0.0,
    bed_slope: float = 0.0,
    left_boundary: BoundaryKind = "wall",
    right_boundary: BoundaryKind = "wall",
    left_ghost: tuple[float, float] | None = None,
    right_ghost: tuple[float, float] | None = None,
    left_face_flux: tuple[float, float] | None = None,
    right_face_flux: tuple[float, float] | None = None,
    interface_traction: (
        tuple[float, float] | tuple[float, float, PressurisedSide] | None
    ) = None,
) -> WetDryState:
    """Advance one positivity-preserving MUSCL forward-Euler stage.

    ``left_face_flux`` and ``right_face_flux`` are exact physical boundary
    fluxes ``(Q, Q^2/A+I_1)`` supplied by an external conservative network
    node.  They replace only the corresponding numerical boundary flux and
    are subjected to the same donor-volume limiter as every internal face.

    ``interface_traction=(x_interface, gas_gauge_head, water_side)`` adds the
    uniform gas pressure force to the water column on the selected side of
    the explicitly tracked face.  The two-item form defaults to ``"right"``,
    which is the Case-B topology.  It is a mechanical interface force, not a
    fictitious liquid layer.
    """

    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive")
    if not (0.0 < cfl < 1.0):
        raise ValueError("cfl must lie in (0, 1)")

    area = np.asarray(state.area, dtype=float).copy()
    discharge = np.asarray(state.discharge, dtype=float).copy()
    ncell = area.size
    if ncell == 0:
        return WetDryState(area, discharge)

    dry_area = dry_area_fraction * section.full_area
    velocity = np.divide(
        discharge,
        area,
        out=np.zeros_like(discharge),
        where=area > dry_area,
    )
    max_speed = float(np.max(np.abs(velocity) + np.asarray(section.celerity(area))))
    if max_speed > 0.0 and dt > cfl * dx / max_speed * (1.0 + 1.0e-12):
        raise ValueError("dt exceeds the requested central-upwind CFL limit")

    area_ext = np.empty(ncell + 2)
    discharge_ext = np.empty(ncell + 2)
    area_ext[1:-1] = area
    discharge_ext[1:-1] = discharge
    if left_ghost is None:
        area_ext[0], discharge_ext[0] = _ghost_state(
            area[0], discharge[0], left_boundary
        )
    else:
        area_ext[0], discharge_ext[0] = map(float, left_ghost)
    if right_ghost is None:
        area_ext[-1], discharge_ext[-1] = _ghost_state(
            area[-1], discharge[-1], right_boundary
        )
    else:
        area_ext[-1], discharge_ext[-1] = map(float, right_ghost)

    (
        area_left,
        discharge_left,
        area_right,
        discharge_right,
    ) = _muscl_free_surface_face_states(
        area_ext,
        discharge_ext,
        section,
        dry_area,
    )
    # Reconstruct the physical boundary trace first, then mirror/copy that
    # trace into an implicit wall/transmissive ghost.  Mirroring only the cell
    # average would leave two unequal second-order face states and create a
    # small artificial mass flux through a nominally closed wall.
    if left_ghost is None:
        area_left[0] = area_right[0]
        discharge_left[0] = (
            -discharge_right[0]
            if left_boundary == "wall"
            else discharge_right[0]
        )
    if right_ghost is None:
        area_right[-1] = area_left[-1]
        discharge_right[-1] = (
            -discharge_left[-1]
            if right_boundary == "wall"
            else discharge_left[-1]
        )
    mass_flux, momentum_flux = _central_upwind_flux(
        area_left,
        discharge_left,
        area_right,
        discharge_right,
        section,
        dry_area,
    )
    if left_face_flux is not None:
        mass_flux[0], momentum_flux[0] = map(float, left_face_flux)
    if right_face_flux is not None:
        mass_flux[-1], momentum_flux[-1] = map(float, right_face_flux)

    # A face may not remove more water than its physical donor owns during the
    # stage.  Scaling the paired momentum flux preserves a bounded donor
    # velocity while guaranteeing A >= 0.
    mass_flux, momentum_flux = _apply_donor_draining_limiter(
        mass_flux,
        momentum_flux,
        area,
        dx=dx,
        dt=dt,
    )

    area_new = area - dt / dx * (mass_flux[1:] - mass_flux[:-1])
    discharge_new = discharge - dt / dx * (
        momentum_flux[1:] - momentum_flux[:-1]
    )
    if interface_traction is not None:
        if len(interface_traction) == 2:
            interface_x, gas_head = interface_traction
            water_side: PressurisedSide = "right"
        else:
            interface_x, gas_head, water_side = interface_traction
        face = int(np.clip(np.rint(interface_x / dx), 1, ncell - 1))
        force_per_density = (
            section.gravity * float(gas_head) * section.full_area
        )
        if water_side == "right":
            discharge_new[face] += dt / dx * force_per_density
        elif water_side == "left":
            discharge_new[face - 1] -= dt / dx * force_per_density
        else:
            raise ValueError("interface traction water_side must be left or right")

    # Near a dry front, momentum divided by a vanishing cell average is not a
    # meaningful point velocity.  Use the standard smooth desingularisation
    # of Q/A and an entropy bound based on the dry-bed gravity-wave scale.
    # This acts only below 0.1% filling and therefore does not alter the
    # resolved free-surface or pressurised branches.
    regularisation_area = max(
        1.0e-3 * section.full_area,
        10.0 * dry_area,
    )
    denominator_velocity = np.sqrt(
        area_new**4
        + np.maximum(area_new**4, regularisation_area**4)
    )
    velocity_regularised = np.divide(
        np.sqrt(2.0) * area_new * discharge_new,
        denominator_velocity,
        out=np.zeros_like(discharge_new),
        where=denominator_velocity > 0.0,
    )
    dry_front_speed_bound = 2.0 * np.sqrt(
        section.gravity * section.diameter
    )
    velocity_regularised = np.clip(
        velocity_regularised,
        -dry_front_speed_bound,
        dry_front_speed_bound,
    )
    shallow = area_new < regularisation_area
    discharge_new[shallow] = (
        area_new[shallow] * velocity_regularised[shallow]
    )

    wet = area_new > dry_area
    velocity_new = np.divide(
        discharge_new,
        area_new,
        out=np.zeros_like(discharge_new),
        where=wet,
    )
    hydraulic_radius = np.asarray(
        section.hydraulic_radius(np.minimum(area_new, section.full_area))
    )
    free = wet & (area_new <= section.full_area)
    full = wet & ~free
    friction_slope = np.zeros(ncell)
    if manning_n > 0.0:
        friction_slope[free] = (
            manning_n**2
            * velocity_new[free]
            * np.abs(velocity_new[free])
            / np.maximum(hydraulic_radius[free], 1.0e-12) ** (4.0 / 3.0)
        )
    if darcy_friction > 0.0:
        friction_slope[full] = (
            darcy_friction
            * velocity_new[full]
            * np.abs(velocity_new[full])
            / (2.0 * section.diameter)
        )
    discharge_new += (
        dt
        * section.gravity
        * area_new
        * (float(bed_slope) - friction_slope)
    )

    # Only roundoff-sized negative areas are possible after the draining
    # limiter.  A materially negative value indicates a broken CFL/flux path.
    material_negative = area_new < -1.0e-12 * section.full_area
    if np.any(material_negative):
        raise FloatingPointError("wet/dry update produced a negative liquid area")
    area_new = np.maximum(area_new, 0.0)
    newly_dry = area_new <= dry_area
    area_new[newly_dry] = 0.0
    discharge_new[newly_dry] = 0.0
    return WetDryState(area_new, discharge_new)


def central_upwind_wet_dry_step(
    state: WetDryState,
    *,
    dx: float,
    dt: float,
    section: CircularSection,
    cfl: float = 0.45,
    dry_area_fraction: float = 1.0e-10,
    manning_n: float = 0.0,
    darcy_friction: float = 0.0,
    bed_slope: float = 0.0,
    left_boundary: BoundaryKind = "wall",
    right_boundary: BoundaryKind = "wall",
    left_ghost: tuple[float, float] | None = None,
    right_ghost: tuple[float, float] | None = None,
    left_face_flux: tuple[float, float] | None = None,
    right_face_flux: tuple[float, float] | None = None,
    interface_traction: (
        tuple[float, float] | tuple[float, float, PressurisedSide] | None
    ) = None,
) -> WetDryState:
    """Advance one fixed MUSCL central-upwind/SSP-RK2 wet/dry step.

    There is no first-order scheme switch.  Reconstruction falls back to cell
    averages only on local stencils that touch a dry cell or the fitted
    free-surface/pressurised branch boundary.  Each Euler stage is positive;
    their SSP convex combination therefore remains positive and conservative.
    """

    stage_keywords = dict(
        dx=dx,
        dt=dt,
        section=section,
        cfl=cfl,
        dry_area_fraction=dry_area_fraction,
        manning_n=manning_n,
        darcy_friction=darcy_friction,
        bed_slope=bed_slope,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        left_ghost=left_ghost,
        right_ghost=right_ghost,
        left_face_flux=left_face_flux,
        right_face_flux=right_face_flux,
        interface_traction=interface_traction,
    )
    first = _central_upwind_wet_dry_euler_step(state, **stage_keywords)
    second_euler = _central_upwind_wet_dry_euler_step(
        first, **stage_keywords
    )

    area_initial = np.asarray(state.area, dtype=float)
    discharge_initial = np.asarray(state.discharge, dtype=float)
    area_new = 0.5 * (area_initial + second_euler.area)
    discharge_new = 0.5 * (discharge_initial + second_euler.discharge)
    dry_area = dry_area_fraction * section.full_area
    if np.any(area_new < -1.0e-12 * section.full_area):
        raise FloatingPointError("SSP-RK2 wet/dry step produced negative area")
    area_new = np.maximum(area_new, 0.0)
    newly_dry = area_new <= dry_area
    area_new[newly_dry] = 0.0
    discharge_new[newly_dry] = 0.0
    return WetDryState(area_new, discharge_new)


def circular_dry_bed_gate_state(
    section: CircularSection,
    *,
    reservoir_depth: float | None = None,
    direction: Literal[-1, 1] = -1,
) -> tuple[float, float]:
    """Return the gravity-rarefaction state feeding a genuinely dry reach.

    For an arbitrary prismatic section, the simple-wave invariant is

    ``Phi(h) = integral_0^h g/c(eta) d eta``.

    At the stationary gate of a reservoir-to-dry-bed Riemann problem the
    self-similar condition is ``|u_*| = c(h_*)`` and
    ``|u_*| = Phi(h_0)-Phi(h_*)``.  Solving this relation supplies a circular
    counterpart of the familiar rectangular ``h_*=4h_0/9`` state without an
    empirical bore-speed coefficient.  ``direction=-1`` is the Case-B
    left-moving wetting front.
    """

    h0 = section.diameter if reservoir_depth is None else float(reservoir_depth)
    h0 = float(np.clip(h0, 1.0e-12 * section.diameter, section.diameter))
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    # h = h0*s^2 removes the integrable 1/sqrt(h) singularity of g/c at h=0.
    s = np.linspace(0.0, 1.0, 8193)
    h = h0 * s**2
    c = np.asarray(section.free_surface_celerity_from_depth(h))
    transformed = np.divide(
        section.gravity * 2.0 * h0 * s,
        c,
        out=np.zeros_like(s),
        where=np.isfinite(c) & (c > 0.0),
    )
    transformed[0] = transformed[1]
    ds = s[1] - s[0]
    phi = np.zeros_like(s)
    phi[1:] = np.cumsum(
        0.5 * (transformed[:-1] + transformed[1:]) * ds
    )
    phi0 = float(phi[-1])
    residual = phi0 - phi - c
    crossing = np.flatnonzero(residual <= 0.0)
    if crossing.size == 0:
        raise FloatingPointError("failed to bracket the circular dry-bed gate state")
    upper = int(crossing[0])
    lower = max(0, upper - 1)
    weight = residual[lower] / max(
        residual[lower] - residual[upper], np.finfo(float).tiny
    )
    h_star = float(h[lower] + weight * (h[upper] - h[lower]))
    c_star = float(section.free_surface_celerity_from_depth(h_star))
    return h_star, float(direction) * c_star


def pressurised_moc_step(
    state: WetDryState,
    *,
    dx: float,
    dt: float,
    section: CircularSection,
    interface_velocity: float,
    interface_head: float,
    darcy_friction: float = 0.0,
    bed_slope: float = 0.0,
    right_boundary: BoundaryKind = "wall",
) -> WetDryState:
    """Advance an explicitly pressurised branch with Tosan's MOC equations.

    The left node is the shock-fitted boundary state.  No numerical Riemann
    problem is formed between this branch and the dry/free-surface branch.
    """

    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive")
    area = np.asarray(state.area, dtype=float)
    discharge = np.asarray(state.discharge, dtype=float)
    if area.size == 0:
        return WetDryState(area.copy(), discharge.copy())
    if dt * section.wave_speed > dx * (1.0 + 1.0e-12):
        raise ValueError("pressurised MOC step crosses more than one grid interval")

    area_full = section.full_area
    g = section.gravity
    acoustic = section.wave_speed
    area_safe = np.maximum(area, 1.0e-12 * area_full)
    velocity = discharge / area_safe
    # In the explicitly pressurised region the elastic storage law remains
    # active even when a transient head is momentarily below the crown.
    head = section.diameter + acoustic**2 / g * (area / area_full - 1.0)
    friction = (
        darcy_friction
        * velocity
        * np.abs(velocity)
        / (2.0 * section.diameter)
    )

    ncell = area.size
    velocity_new = np.empty_like(velocity)
    head_new = np.empty_like(head)
    velocity_new[0] = float(interface_velocity)
    head_new[0] = float(interface_head)

    ratio = min(1.0, acoustic * dt / dx)
    if ncell > 2:
        centre = slice(1, -1)
        u_plus = velocity[centre] + ratio * (
            velocity[:-2] - velocity[centre]
        )
        h_plus = head[centre] + ratio * (head[:-2] - head[centre])
        sf_plus = friction[centre] + ratio * (
            friction[:-2] - friction[centre]
        )
        c_plus = (
            u_plus
            + g / acoustic * h_plus
            + g * (bed_slope - sf_plus) * dt
        )
        u_minus = velocity[centre] + ratio * (
            velocity[2:] - velocity[centre]
        )
        h_minus = head[centre] + ratio * (head[2:] - head[centre])
        sf_minus = friction[centre] + ratio * (
            friction[2:] - friction[centre]
        )
        c_minus = (
            u_minus
            - g / acoustic * h_minus
            + g * (bed_slope - sf_minus) * dt
        )
        velocity_new[centre] = 0.5 * (c_plus + c_minus)
        head_new[centre] = 0.5 * acoustic / g * (c_plus - c_minus)

    if ncell > 1:
        u_plus = velocity[-1] + ratio * (velocity[-2] - velocity[-1])
        h_plus = head[-1] + ratio * (head[-2] - head[-1])
        sf_plus = friction[-1] + ratio * (friction[-2] - friction[-1])
        c_plus = (
            u_plus
            + g / acoustic * h_plus
            + g * (bed_slope - sf_plus) * dt
        )
        if right_boundary == "wall":
            velocity_new[-1] = 0.0
            head_new[-1] = acoustic / g * c_plus
        elif right_boundary == "transmissive":
            velocity_new[-1] = velocity[-1]
            head_new[-1] = head[-1]
        else:
            raise ValueError(f"unsupported right boundary: {right_boundary}")

    area_new = area_full * (
        1.0 + g * (head_new - section.diameter) / acoustic**2
    )
    if np.any(~np.isfinite(area_new)) or np.any(area_new <= 0.0):
        raise FloatingPointError("pressurised MOC produced an inadmissible area")
    discharge_new = area_new * velocity_new
    return WetDryState(area_new, discharge_new)


@dataclass(frozen=True)
class TosanInterfaceData:
    """Characteristic-foot and free-surface data at one shock front."""

    pressurised_velocity_foot: float
    pressurised_head_foot: float
    free_surface_velocity: float
    free_surface_depth: float
    gas_pressure_head: float
    dt: float = 0.0
    pressurised_friction_slope: float = 0.0
    bed_slope: float = 0.0


@dataclass(frozen=True)
class TosanInterfaceSolution:
    """Solution of the interface compatibility and jump equations."""

    pressurised_velocity: float
    pressurised_head: float
    interface_speed: float
    residual: Array
    iterations: int
    converged: bool
    formulation: str
    free_surface_velocity: float | None = None
    free_surface_depth: float | None = None

    @property
    def residual_linf(self) -> float:
        return float(np.max(np.abs(self.residual)))


def _damped_newton(
    residual_and_jacobian: Callable[[Array], tuple[Array, Array]],
    initial: Array,
    scales: Array,
    *,
    tolerance: float,
    max_iterations: int,
    admissible: Callable[[Array], bool],
) -> tuple[Array, Array, int, bool]:
    """Small safeguarded Newton solve with a regularised least-squares fallback."""

    x = np.asarray(initial, dtype=float).copy()
    scales = np.maximum(np.asarray(scales, dtype=float), np.finfo(float).tiny)
    best_x = x.copy()
    best_residual, _ = residual_and_jacobian(x)
    best_norm = float(np.max(np.abs(best_residual / scales)))

    for iteration in range(1, max_iterations + 1):
        residual, jacobian = residual_and_jacobian(x)
        scaled_residual = residual / scales
        scaled_jacobian = jacobian / scales[:, None]
        norm = float(np.max(np.abs(scaled_residual)))
        if norm <= tolerance:
            return x, residual, iteration, True
        try:
            increment = np.linalg.solve(scaled_jacobian, -scaled_residual)
        except np.linalg.LinAlgError:
            damping = 1.0e-8 * max(
                1.0, float(np.linalg.norm(scaled_jacobian, ord=2) ** 2)
            )
            normal = scaled_jacobian.T @ scaled_jacobian
            increment = np.linalg.solve(
                normal + damping * np.eye(normal.shape[0]),
                -scaled_jacobian.T @ scaled_residual,
            )

        accepted = False
        step = 1.0
        for _ in range(24):
            candidate = x + step * increment
            if admissible(candidate):
                candidate_residual, _ = residual_and_jacobian(candidate)
                candidate_norm = float(
                    np.max(np.abs(candidate_residual / scales))
                )
                if candidate_norm < norm:
                    x = candidate
                    accepted = True
                    if candidate_norm < best_norm:
                        best_x = candidate.copy()
                        best_residual = candidate_residual.copy()
                        best_norm = candidate_norm
                    break
            step *= 0.5
        if not accepted:
            break

    return best_x, best_residual, max_iterations, best_norm <= tolerance


def solve_tosan_positive_interface(
    data: TosanInterfaceData,
    *,
    section: CircularSection,
    tolerance: float = 1.0e-10,
    max_iterations: int = 40,
) -> TosanInterfaceSolution:
    """Solve Tosan Eqs. (6-45)--(6-47) for an advancing interface.

    Unknowns are pressurised velocity ``u_p``, pressurised head ``H_p`` and
    shock speed ``w``.  The residuals reproduce Appendix-C ``flxT4`` after
    division by liquid density:

    * positive pressurised characteristic compatibility;
    * Rankine-Hugoniot liquid continuity;
    * Tosan's reduced momentum jump including uniform gas pressure.
    """

    g = section.gravity
    a = section.wave_speed
    area_full = section.full_area
    depth_fs = float(np.clip(data.free_surface_depth, 0.0, section.diameter))
    area_fs = float(section.area_from_depth(depth_fs))
    moment_fs = float(section.hydrostatic_moment(depth_fs))
    u_fs = float(data.free_surface_velocity)
    source = g * data.dt * (
        data.pressurised_friction_slope - data.bed_slope
    )

    def residual_and_jacobian(x: Array) -> tuple[Array, Array]:
        u_p, head_p, speed = x
        residual = np.array(
            [
                u_p
                - data.pressurised_velocity_foot
                + g / a * (head_p - data.pressurised_head_foot)
                + source,
                area_full * u_p
                - area_fs * u_fs
                - speed * (area_full - area_fs),
                g
                * area_full
                * (head_p - 0.5 * section.diameter - data.gas_pressure_head)
                - g * moment_fs
                + area_full * (speed - u_fs) * (u_fs - u_p),
            ],
            dtype=float,
        )
        jacobian = np.array(
            [
                [1.0, g / a, 0.0],
                [area_full, 0.0, -(area_full - area_fs)],
                [
                    area_full * (u_fs - speed),
                    g * area_full,
                    area_full * (u_fs - u_p),
                ],
            ],
            dtype=float,
        )
        return residual, jacobian

    denominator = max(area_full - area_fs, 1.0e-12 * area_full)
    speed0 = (
        area_full * data.pressurised_velocity_foot - area_fs * u_fs
    ) / denominator
    initial = np.array(
        [data.pressurised_velocity_foot, data.pressurised_head_foot, speed0]
    )
    gravity_speed = np.sqrt(g * section.diameter)
    scales = np.array(
        [
            max(gravity_speed, 1.0),
            area_full * max(gravity_speed, 1.0),
            g
            * area_full
            * max(
                section.diameter,
                abs(data.gas_pressure_head),
                abs(data.pressurised_head_foot),
                1.0e-3,
            ),
        ]
    )

    def admissible(x: Array) -> bool:
        return bool(
            np.all(np.isfinite(x))
            and -10.0 * section.diameter <= x[1] <= 1.0e4
            and abs(x[0]) <= 2.0 * a
            and abs(x[2]) <= 2.0 * a
        )

    solution, residual, iterations, converged = _damped_newton(
        residual_and_jacobian,
        initial,
        scales,
        tolerance=tolerance,
        max_iterations=max_iterations,
        admissible=admissible,
    )
    return TosanInterfaceSolution(
        pressurised_velocity=float(solution[0]),
        pressurised_head=float(solution[1]),
        interface_speed=float(solution[2]),
        residual=residual,
        iterations=iterations,
        converged=converged,
        formulation="tosan_positive_6_45_to_6_47",
    )


def solve_oriented_interface(
    data: TosanInterfaceData,
    *,
    section: CircularSection,
    pressurised_side: PressurisedSide,
    tolerance: float = 1.0e-10,
    max_iterations: int = 40,
) -> TosanInterfaceSolution:
    """Solve characteristic compatibility plus the full RH jumps.

    ``pressurised_side='right'`` is the Case-B orientation.  The incoming
    water-hammer characteristic is then ``u-gH/a``; for the canonical Tosan
    positive interface (pressurised side left) it is ``u+gH/a``.
    """

    sign = 1.0 if pressurised_side == "left" else -1.0
    if pressurised_side not in ("left", "right"):
        raise ValueError("pressurised_side must be 'left' or 'right'")
    g = section.gravity
    a = section.wave_speed
    area_full = section.full_area
    depth_fs = float(np.clip(data.free_surface_depth, 0.0, section.diameter))
    area_fs = float(section.area_from_depth(depth_fs))
    moment_fs = float(section.hydrostatic_moment(depth_fs))
    u_fs = float(data.free_surface_velocity)
    source = g * data.dt * (
        data.pressurised_friction_slope - data.bed_slope
    )

    def residual_and_jacobian(x: Array) -> tuple[Array, Array]:
        u_p, head_p, speed = x
        mass_momentum = area_full * u_p - area_fs * u_fs
        pressure_p = g * area_full * (head_p - 0.5 * section.diameter)
        pressure_fs_and_gas = (
            g * moment_fs + g * data.gas_pressure_head * area_full
        )
        residual = np.array(
            [
                u_p
                - data.pressurised_velocity_foot
                + sign * g / a * (head_p - data.pressurised_head_foot)
                + source,
                mass_momentum - speed * (area_full - area_fs),
                area_full * u_p**2
                + pressure_p
                - area_fs * u_fs**2
                - pressure_fs_and_gas
                - speed * mass_momentum,
            ],
            dtype=float,
        )
        jacobian = np.array(
            [
                [1.0, sign * g / a, 0.0],
                [area_full, 0.0, -(area_full - area_fs)],
                [
                    area_full * (2.0 * u_p - speed),
                    g * area_full,
                    -mass_momentum,
                ],
            ],
            dtype=float,
        )
        return residual, jacobian

    denominator = max(area_full - area_fs, 1.0e-12 * area_full)
    speed0 = (
        area_full * data.pressurised_velocity_foot - area_fs * u_fs
    ) / denominator
    head0 = max(
        data.pressurised_head_foot,
        0.5 * section.diameter + data.gas_pressure_head,
    )
    initial = np.array(
        [data.pressurised_velocity_foot, head0, speed0], dtype=float
    )
    gravity_speed = np.sqrt(g * section.diameter)
    scales = np.array(
        [
            max(gravity_speed, 1.0),
            area_full * max(gravity_speed, 1.0),
            g
            * area_full
            * max(
                section.diameter,
                abs(data.gas_pressure_head),
                abs(data.pressurised_head_foot),
                1.0e-3,
            ),
        ]
    )

    def admissible(x: Array) -> bool:
        return bool(
            np.all(np.isfinite(x))
            and -10.0 * section.diameter <= x[1] <= 1.0e4
            and abs(x[0]) <= 2.0 * a
            and abs(x[2]) <= 2.0 * a
        )

    solution, residual, iterations, converged = _damped_newton(
        residual_and_jacobian,
        initial,
        scales,
        tolerance=tolerance,
        max_iterations=max_iterations,
        admissible=admissible,
    )
    return TosanInterfaceSolution(
        pressurised_velocity=float(solution[0]),
        pressurised_head=float(solution[1]),
        interface_speed=float(solution[2]),
        residual=residual,
        iterations=iterations,
        converged=converged,
        formulation=f"full_rankine_hugoniot_pressurised_{pressurised_side}",
    )


def solve_tosan_negative_interface_case_b(
    data: TosanInterfaceData,
    *,
    section: CircularSection,
    manning_n: float = 0.009,
    interface_speed_upper: float | None = None,
    tolerance: float = 1.0e-9,
    max_iterations: int = 50,
) -> TosanInterfaceSolution:
    """Reflected Tosan negative-interface solve for the Case-B topology.

    Tosan Appendix-C ``flxT1`` treats the negative-interface speed explicitly
    from the preceding states, then solves the incoming pressurised and
    free-surface characteristic equations together with the mass and momentum
    jumps.  In physical Case-B coordinates, the pressurised branch is on the
    right, hence the incoming characteristics are ``C-`` in that branch and
    ``C+`` in the left free-surface branch.

    The zero-depth endpoint is intentionally excluded here; it is supplied by
    :func:`circular_dry_bed_gate_state` until a finite free-surface boundary
    state exists.
    """

    g = section.gravity
    acoustic = section.wave_speed
    area_full = section.full_area
    h_foot = float(np.clip(data.free_surface_depth, 0.0, section.diameter))
    area_foot = float(section.area_from_depth(h_foot))
    if area_foot <= 1.0e-12 * area_full:
        raise ValueError("negative-interface MOC requires a non-dry FS foot")
    c_foot = float(section.free_surface_celerity_from_depth(h_foot))
    u_foot = float(data.free_surface_velocity)
    u_p_foot = float(data.pressurised_velocity_foot)
    h_p_foot = float(data.pressurised_head_foot)
    denominator = max(area_full - area_foot, 1.0e-12 * area_full)
    speed_explicit = (
        area_full * u_p_foot - area_foot * u_foot
    ) / denominator
    if interface_speed_upper is not None:
        speed_explicit = min(speed_explicit, float(interface_speed_upper))
    hydraulic_radius_foot = float(section.hydraulic_radius(area_foot))
    fs_friction_foot = (
        manning_n**2
        * u_foot
        * abs(u_foot)
        / max(hydraulic_radius_foot, 1.0e-12) ** (4.0 / 3.0)
    )
    p_source = g * data.dt * (
        data.pressurised_friction_slope - data.bed_slope
    )
    fs_source = g * data.dt * (fs_friction_foot - data.bed_slope)

    def residual_and_jacobian(x: Array) -> tuple[Array, Array]:
        u_p, head_p, u_fs, depth_fs = x
        depth_fs = float(depth_fs)
        area_fs = float(section.area_from_depth(depth_fs))
        moment_fs = float(section.hydrostatic_moment(depth_fs))
        c_fs = float(section.free_surface_celerity_from_depth(depth_fs))
        c_fs = max(c_fs, 1.0e-12)
        top_width = float(section.top_width(depth_fs))
        mass_momentum = area_full * u_p - area_fs * u_fs
        residual = np.array(
            [
                u_p
                - u_p_foot
                - g / acoustic * (head_p - h_p_foot)
                + p_source,
                u_fs
                - u_foot
                + g * depth_fs / c_fs
                - g * h_foot / c_foot
                + fs_source,
                mass_momentum - speed_explicit * (area_full - area_fs),
                area_full * u_p**2
                + g * area_full * (head_p - 0.5 * section.diameter)
                - area_fs * u_fs**2
                - g * moment_fs
                - g * data.gas_pressure_head * area_full
                - speed_explicit * mass_momentum,
            ],
            dtype=float,
        )
        diameter = section.diameter
        dlog_width = (diameter - 2.0 * depth_fs) / (
            2.0
            * max(
                depth_fs * (diameter - depth_fs),
                1.0e-20 * diameter**2,
            )
        )
        dc_dh = 0.5 * c_fs * (
            top_width / max(area_fs, 1.0e-20 * area_full)
            - dlog_width
        )
        d_h_over_c = 1.0 / c_fs - depth_fs * dc_dh / c_fs**2
        jacobian = np.array(
            [
                [1.0, -g / acoustic, 0.0, 0.0],
                [0.0, 0.0, 1.0, g * d_h_over_c],
                [
                    area_full,
                    0.0,
                    -area_fs,
                    top_width * (speed_explicit - u_fs),
                ],
                [
                    area_full * (2.0 * u_p - speed_explicit),
                    g * area_full,
                    area_fs * (speed_explicit - 2.0 * u_fs),
                    top_width * u_fs * (speed_explicit - u_fs)
                    - g * area_fs,
                ],
            ],
            dtype=float,
        )
        return residual, jacobian

    def reduced_state(depth_fs: float) -> tuple[Array, float]:
        """Eliminate both characteristics and mass RH at a trial depth."""

        area_fs = float(section.area_from_depth(depth_fs))
        c_fs = max(
            float(section.free_surface_celerity_from_depth(depth_fs)),
            1.0e-12,
        )
        u_fs = (
            u_foot
            - g * depth_fs / c_fs
            + g * h_foot / c_foot
            - fs_source
        )
        u_p = (
            speed_explicit * (area_full - area_fs) + area_fs * u_fs
        ) / area_full
        head_p = h_p_foot + acoustic / g * (
            u_p - u_p_foot + p_source
        )
        x = np.array([u_p, head_p, u_fs, depth_fs], dtype=float)
        residual, jacobian = residual_and_jacobian(x)
        # Total derivative of the momentum residual along the three eliminated
        # equations.  This is the Schur complement of the 4x4 Newton system.
        top_width = float(section.top_width(depth_fs))
        c_derivative = 0.5 * c_fs * (
            top_width / max(area_fs, 1.0e-20 * area_full)
            - (section.diameter - 2.0 * depth_fs)
            / (
                2.0
                * max(
                    depth_fs * (section.diameter - depth_fs),
                    1.0e-20 * section.diameter**2,
                )
            )
        )
        d_h_over_c = 1.0 / c_fs - depth_fs * c_derivative / c_fs**2
        du_fs = -g * d_h_over_c
        du_p = (
            top_width * (u_fs - speed_explicit)
            + area_fs * du_fs
        ) / area_full
        dhead_p = acoustic / g * du_p
        total_derivative = (
            jacobian[3, 0] * du_p
            + jacobian[3, 1] * dhead_p
            + jacobian[3, 2] * du_fs
            + jacobian[3, 3]
        )
        return x, float(total_derivative)

    # The four equations reduce exactly to one scalar momentum equation after
    # eliminating the two characteristic relations and mass RH.  Solving this
    # Schur complement avoids a dense nonlinear solve at every acoustic step.
    depth_trial = float(
        np.clip(
            h_foot,
            1.0e-10 * section.diameter,
            (1.0 - 1.0e-9) * section.diameter,
        )
    )
    momentum_scale = (
        g
        * area_full
        * max(
            section.diameter,
            abs(data.gas_pressure_head),
            abs(h_p_foot),
            1.0e-3,
        )
    )
    reduced_converged = False
    reduced_iterations = 0
    reduced_x = np.array(
        [u_p_foot, h_p_foot, u_foot, depth_trial], dtype=float
    )
    reduced_residual, _ = residual_and_jacobian(reduced_x)
    for reduced_iterations in range(1, min(max_iterations, 16) + 1):
        reduced_x, derivative = reduced_state(depth_trial)
        reduced_residual, _ = residual_and_jacobian(reduced_x)
        norm = abs(float(reduced_residual[3])) / max(
            momentum_scale, np.finfo(float).tiny
        )
        if norm <= tolerance:
            reduced_converged = True
            break
        if not np.isfinite(derivative) or abs(derivative) <= 1.0e-20:
            break
        increment = float(
            np.clip(
                -reduced_residual[3] / derivative,
                -0.25 * section.diameter,
                0.25 * section.diameter,
            )
        )
        accepted = False
        step_fraction = 1.0
        for _ in range(12):
            candidate_depth = depth_trial + step_fraction * increment
            if (
                1.0e-10 * section.diameter
                < candidate_depth
                < (1.0 - 1.0e-9) * section.diameter
            ):
                candidate_x, _ = reduced_state(candidate_depth)
                candidate_residual, _ = residual_and_jacobian(candidate_x)
                if abs(candidate_residual[3]) < abs(reduced_residual[3]):
                    depth_trial = candidate_depth
                    accepted = True
                    break
            step_fraction *= 0.5
        if not accepted:
            break
    if reduced_converged:
        return TosanInterfaceSolution(
            pressurised_velocity=float(reduced_x[0]),
            pressurised_head=float(reduced_x[1]),
            interface_speed=float(speed_explicit),
            residual=reduced_residual,
            iterations=reduced_iterations,
            converged=True,
            formulation="tosan_negative_flxT1_reflected_case_b",
            free_surface_velocity=float(reduced_x[2]),
            free_surface_depth=float(reduced_x[3]),
        )

    initial = np.array(
        [u_p_foot, h_p_foot, u_foot, h_foot], dtype=float
    )
    gravity_speed = np.sqrt(g * section.diameter)
    scales = np.array(
        [
            max(gravity_speed, 1.0),
            max(gravity_speed, 1.0),
            area_full * max(gravity_speed, 1.0),
            g
            * area_full
            * max(
                section.diameter,
                abs(data.gas_pressure_head),
                abs(h_p_foot),
                1.0e-3,
            ),
        ]
    )

    def admissible(x: Array) -> bool:
        return bool(
            np.all(np.isfinite(x))
            and abs(x[0]) <= 2.0 * acoustic
            and abs(x[2]) <= 2.0 * acoustic
            and -10.0 * section.diameter <= x[1] <= 1.0e4
            and 1.0e-10 * section.diameter
            <= x[3]
            <= (1.0 - 1.0e-9) * section.diameter
        )

    solution, residual, iterations, converged = _damped_newton(
        residual_and_jacobian,
        initial,
        scales,
        tolerance=tolerance,
        max_iterations=max_iterations,
        admissible=admissible,
    )
    return TosanInterfaceSolution(
        pressurised_velocity=float(solution[0]),
        pressurised_head=float(solution[1]),
        interface_speed=float(speed_explicit),
        residual=residual,
        iterations=iterations,
        converged=converged,
        formulation="tosan_negative_flxT1_reflected_case_b",
        free_surface_velocity=float(solution[2]),
        free_surface_depth=float(solution[3]),
    )


def advance_shock_position(
    position: float,
    speed: float,
    dt: float,
    *,
    length: float,
) -> float:
    """Tosan Eq. (6-57): explicitly advance and clip the tracked front."""

    if dt < 0.0 or length <= 0.0:
        raise ValueError("dt must be nonnegative and length must be positive")
    return float(np.clip(position + speed * dt, 0.0, length))


@dataclass(frozen=True)
class HorizontalState:
    """Complete state returned by :class:`Tosan2021HorizontalShockFit`."""

    time: float
    area: Array
    discharge: Array
    gas: PolytropicGasInventory
    air_pressure_abs: float
    interface_x: float
    interface_speed: float
    interface_free_surface_depth: float
    interface_free_surface_velocity: float
    interface_pressurised_head: float
    interface_pressurised_velocity: float
    interface_residual_linf: float
    wetting_front_x: float
    vented: bool
    nonlinear_converged: bool
    liquid_volume_residual: float = 0.0
    cumulative_liquid_volume_residual: float = 0.0


class Tosan2021HorizontalShockFit:
    """Standalone Case-B horizontal solver and network-adapter surface."""

    def __init__(
        self,
        config: HorizontalConfig | None = None,
        *,
        vent_pressure_hook: VentPressureHook | None = None,
    ) -> None:
        self.config = config or HorizontalConfig()
        self.section = CircularSection(
            self.config.diameter,
            self.config.gravity,
            self.config.wave_speed,
        )
        self.ncell = max(4, int(np.ceil(self.config.length / self.config.dx)))
        self.dx = self.config.length / self.ncell
        self.x = (np.arange(self.ncell, dtype=float) + 0.5) * self.dx
        self.vent_pressure_hook = vent_pressure_hook
        self.dry_gate_depth, self.dry_gate_velocity = (
            circular_dry_bed_gate_state(self.section, direction=-1)
        )
        self.dry_gate_area = float(
            self.section.area_from_depth(self.dry_gate_depth)
        )
        # The analytical dry-bed trace and the liquid mass jump give an
        # energy-conserving upper speed.  Once the upstream reach is wetted,
        # Case B is a dissipative long air-cavity gravity current; its
        # independently measured circular-pipe range is Fr≈0.47--0.54.  The
        # configured Fr=0.48 sharpens (never raises) the RH upper bound and
        # prevents the one-front flxT1 acoustic branch from creating a
        # supercritical nose after the gas overpressure has relaxed.
        dry_bed_rh_bound = float(
            -self.dry_gate_area
            * self.dry_gate_velocity
            / max(
                self.section.full_area - self.dry_gate_area,
                1.0e-12 * self.section.full_area,
            )
        )
        gravity_current_bound = float(
            self.config.gravity_current_froude
            * np.sqrt(self.config.gravity * self.config.diameter)
        )
        self.case_b_entropy_speed_bound = min(
            dry_bed_rh_bound,
            gravity_current_bound,
        )

    def case_b_initial_state(
        self,
        *,
        initial_air_gauge_head: float | None = None,
        initial_water_head: float | None = None,
    ) -> HorizontalState:
        """Return the exact dry/full Case-B horizontal initial condition."""

        air_head = (
            self.config.initial_air_head
            if initial_air_gauge_head is None
            else float(initial_air_gauge_head)
        )
        water_head = (
            self.config.initial_water_head
            if initial_water_head is None
            else float(initial_water_head)
        )
        pressurised_area = float(self.section.area_from_head(water_head))
        area = np.where(
            self.x < self.config.valve_x,
            0.0,
            pressurised_area,
        )
        # The valve generally lies inside a finite-volume cell.  Store the
        # exact dry/full subcell average at t=0; leaving that cell wholly dry
        # makes the first shock projection create roughly half a cell of water.
        cut_index = int(np.clip(
            np.floor(self.config.valve_x / self.dx),
            0,
            self.ncell - 1,
        ))
        cut_left = cut_index * self.dx
        cut_right = cut_left + self.dx
        pressurised_fraction = float(np.clip(
            (cut_right - self.config.valve_x) / self.dx,
            0.0,
            1.0,
        ))
        area[cut_index] = pressurised_fraction * pressurised_area
        discharge = np.zeros_like(area)
        gas_volume = self.section.full_area * self.config.valve_x
        gas = PolytropicGasInventory.from_gauge_head(
            volume=gas_volume,
            gauge_head=air_head,
            atmospheric_pressure=self.config.atmospheric_pressure,
            liquid_density=self.config.liquid_density,
            gravity=self.config.gravity,
            gamma=self.config.gamma,
            gas_constant=self.config.gas_constant,
            temperature=self.config.temperature,
        )
        return HorizontalState(
            time=0.0,
            area=area,
            discharge=discharge,
            gas=gas,
            air_pressure_abs=gas.pressure_abs,
            interface_x=self.config.valve_x,
            interface_speed=0.0,
            interface_free_surface_depth=self.dry_gate_depth,
            interface_free_surface_velocity=self.dry_gate_velocity,
            interface_pressurised_head=water_head,
            interface_pressurised_velocity=0.0,
            interface_residual_linf=0.0,
            wetting_front_x=self.config.valve_x,
            vented=False,
            nonlinear_converged=True,
        )

    def stable_timestep(self, state: HorizontalState) -> float:
        """CFL step including the explicitly moving interface."""

        dry_area = self.config.dry_area_fraction * self.section.full_area
        velocity = np.divide(
            state.discharge,
            state.area,
            out=np.zeros_like(state.discharge),
            where=state.area > dry_area,
        )
        wave = np.asarray(self.section.celerity(state.area))
        speed = max(
            float(np.max(np.abs(velocity) + wave)),
            abs(float(state.interface_speed)),
            self.config.wave_speed,
            1.0e-12,
        )
        return self.config.cfl * self.dx / speed

    def _interface_cells(self, state: HorizontalState) -> tuple[int, int]:
        # Return the nearest *complete* FS and pressurised nodes.  The cell
        # containing the tracked shock lies between them and is never used as
        # a characteristic foot or MOC node.
        cut = int(
            np.clip(
                np.floor(state.interface_x / self.dx),
                1,
                self.ncell - 2,
            )
        )
        return cut - 1, cut + 1

    def _interface_solution(
        self,
        state: HorizontalState,
        *,
        dt: float = 0.0,
    ) -> TosanInterfaceSolution:
        free_index, pressurised_index = self._interface_cells(state)
        # The last FS cell and first pressurised cell receive the previous
        # fitted boundary values.  Characteristic feet must be one complete
        # spatial node inside each branch, never those boundary/cut nodes.
        free_foot_index = max(0, free_index - 1)
        free_foot_area = min(
            max(float(state.area[free_foot_index]), 0.0),
            self.section.full_area * (1.0 - 1.0e-9),
        )
        if free_foot_area >= self.dry_gate_area:
            depth_free = float(
                self.section.depth_from_area(free_foot_area)
            )
            velocity_free = float(
                state.discharge[free_foot_index] / free_foot_area
            )
        else:
            # Until a complete interior FV cell contains the analytical gate
            # state, the characteristic foot lies in the unresolved dry-bed
            # rarefaction.  Use that exact boundary trace rather than a
            # vanishing cell average or the previous shock node.
            depth_free = self.dry_gate_depth
            velocity_free = self.dry_gate_velocity
        pressure_foot_index = min(pressurised_index + 1, self.ncell - 1)
        area_pressurised = max(
            float(state.area[pressure_foot_index]), self.section.full_area
        )
        velocity_pressurised = float(
            state.discharge[pressure_foot_index] / area_pressurised
        )
        head_pressurised = float(
            self.section.diameter
            + self.config.wave_speed**2
            / self.config.gravity
            * (
                area_pressurised / self.section.full_area
                - 1.0
            )
        )
        gas_head = (
            state.air_pressure_abs - self.config.atmospheric_pressure
        ) / (self.config.liquid_density * self.config.gravity)
        data = TosanInterfaceData(
            pressurised_velocity_foot=velocity_pressurised,
            pressurised_head_foot=head_pressurised,
            free_surface_velocity=velocity_free,
            free_surface_depth=depth_free,
            gas_pressure_head=gas_head,
            dt=dt,
            pressurised_friction_slope=(
                self.config.darcy_friction
                * velocity_pressurised
                * abs(velocity_pressurised)
                / (2.0 * self.config.diameter)
            ),
            bed_slope=self.config.bed_slope,
        )
        negative = solve_tosan_negative_interface_case_b(
            data,
            section=self.section,
            manning_n=self.config.manning_n,
            interface_speed_upper=self.case_b_entropy_speed_bound,
            tolerance=max(self.config.nonlinear_tolerance, 1.0e-9),
            max_iterations=self.config.nonlinear_max_iterations,
        )
        if negative.converged:
            c_fs = float(
                self.section.free_surface_celerity_from_depth(
                    negative.free_surface_depth
                )
            )
            lower = float(negative.free_surface_velocity + c_fs)
            upper = float(
                negative.pressurised_velocity + self.config.wave_speed
            )
            if lower < negative.interface_speed < upper:
                return negative

        # A safeguarded full-RH solve preserves an admissible pressure boundary
        # if the four-state negative-interface Newton iteration stalls.
        fallback = solve_oriented_interface(
            data,
            section=self.section,
            pressurised_side="right",
            tolerance=self.config.nonlinear_tolerance,
            max_iterations=self.config.nonlinear_max_iterations,
        )
        return replace(
            fallback,
            formulation=(
                "case_b_negative_fallback_full_rankine_hugoniot"
            ),
            free_surface_velocity=velocity_free,
            free_surface_depth=depth_free,
        )

    def _connected_gas_volume(
        self,
        area: Array,
        interface_x: float,
        boundary_area: float,
        wetting_front_x: float,
    ) -> float:
        """Geometric ``Af*L_g - integral A_l dx`` over the connected pocket."""

        left_edges = self.x - 0.5 * self.dx
        right_edges = self.x + 0.5 * self.dx
        overlap = np.clip(
            np.minimum(right_edges, interface_x) - left_edges,
            0.0,
            self.dx,
        )
        liquid = np.minimum(np.maximum(area, 0.0), self.section.full_area)
        overlap_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(self.config.length, self.dx)
        )
        partial = (overlap > overlap_tolerance) & (
            overlap < self.dx - overlap_tolerance
        )
        liquid_volume = float(np.sum(overlap[~partial] * liquid[~partial]))
        if np.any(partial):
            # The interface cell may contain, from left to right, a dry part,
            # a free-surface part, and a pressurised part.  Only the portion
            # between the finite-speed wetting front and the tracked interface
            # subtracts liquid from the connected gas volume.  Treating the
            # whole overlap as the boundary layer makes water appear under the
            # pocket instantaneously at the first substep.
            boundary = float(np.clip(
                boundary_area, 0.0, self.section.full_area
            ))
            for index in np.flatnonzero(partial):
                free_length = max(
                    0.0,
                    float(interface_x)
                    - max(float(wetting_front_x), float(left_edges[index])),
                )
                free_length = min(free_length, float(overlap[index]))
                liquid_volume += boundary * free_length
        gas_geometric_volume = self.section.full_area * float(interface_x)
        return float(max(gas_geometric_volume - liquid_volume, 0.0))

    def _wetting_front(
        self,
        area: Array,
        fallback: float,
        *,
        area_fraction: float | None = None,
    ) -> float:
        fraction = (
            self.config.wetting_front_report_fraction
            if area_fraction is None
            else float(area_fraction)
        )
        threshold = fraction * self.section.full_area
        wet = np.flatnonzero(area > threshold)
        if wet.size == 0:
            return float(fallback)
        return float(max(0.0, self.x[int(wet[0])] - 0.5 * self.dx))

    def _effective_pressure(
        self,
        *,
        time: float,
        interface_x: float,
        closed_pressure_abs: float,
        external_pressure_abs: float | None,
    ) -> tuple[float, bool]:
        vented = bool(
            self.config.vent_x is not None
            and interface_x >= self.config.vent_x
        )
        if external_pressure_abs is not None:
            pressure = float(external_pressure_abs)
        elif vented and self.vent_pressure_hook is not None:
            pressure = float(
                self.vent_pressure_hook(time, interface_x, closed_pressure_abs)
            )
        else:
            pressure = float(closed_pressure_abs)
        if not np.isfinite(pressure) or pressure <= 0.0:
            raise ValueError("effective gas pressure must be positive and finite")
        return pressure, vented

    def _step_once(
        self,
        state: HorizontalState,
        dt: float,
        external_pressure_abs: float | None,
    ) -> HorizontalState:
        interface_solution = self._interface_solution(state, dt=dt)
        speed = (
            interface_solution.interface_speed
            if interface_solution.converged
            else state.interface_speed
        )
        fs_depth_for_bound = float(
            interface_solution.free_surface_depth
            if interface_solution.free_surface_depth is not None
            else state.interface_free_surface_depth
        )
        fs_velocity_for_bound = float(
            interface_solution.free_surface_velocity
            if interface_solution.free_surface_velocity is not None
            else state.interface_free_surface_velocity
        )
        characteristic_lower = fs_velocity_for_bound + float(
            self.section.free_surface_celerity_from_depth(fs_depth_for_bound)
        )
        characteristic_upper = (
            interface_solution.pressurised_velocity + self.config.wave_speed
        )
        characteristic_margin = 1.0e-10 * max(
            1.0, self.config.wave_speed
        )
        # Enforce the Lax/Tosan characteristic ordering.  Projection onto the
        # characteristic interval is an admissibility operation, not temporal
        # smoothing and contains no fitted relaxation coefficient.
        speed = float(
            np.clip(
                speed,
                characteristic_lower + characteristic_margin,
                characteristic_upper - characteristic_margin,
            )
        )
        interface_new = advance_shock_position(
            state.interface_x,
            speed,
            dt,
            length=self.config.length,
        )

        # Strict domain split: the zero-depth central-upwind extension sees
        # only the gas/free-surface region, while the full branch is advanced
        # only by MOC.  There is never an acoustic full/dry numerical face.
        _, pressurised_start = self._interface_cells(state)
        pressurised_start = int(
            np.clip(pressurised_start, 2, self.ncell - 1)
        )
        cut_old = pressurised_start - 1
        fs_stop = cut_old
        fs_area = np.asarray(state.area[:fs_stop], dtype=float).copy()
        fs_discharge = np.asarray(
            state.discharge[:fs_stop], dtype=float
        ).copy()
        p_area_input = np.asarray(
            state.area[pressurised_start:], dtype=float
        ).copy()
        p_discharge_input = np.asarray(
            state.discharge[pressurised_start:], dtype=float
        ).copy()
        boundary_depth = float(
            interface_solution.free_surface_depth
            if interface_solution.free_surface_depth is not None
            else self.dry_gate_depth
        )
        boundary_velocity = float(
            interface_solution.free_surface_velocity
            if interface_solution.free_surface_velocity is not None
            else self.dry_gate_velocity
        )
        boundary_area = float(self.section.area_from_depth(boundary_depth))
        boundary_discharge = boundary_area * boundary_velocity
        boundary_celerity = float(
            self.section.free_surface_celerity_from_depth(boundary_depth)
        )
        wetting_front_speed = min(
            boundary_velocity - 2.0 * boundary_celerity,
            0.0,
        )
        wetting_front_new = float(np.clip(
            state.wetting_front_x + wetting_front_speed * dt,
            0.0,
            interface_new,
        ))

        # A cell already passed by the front is represented by the fitted FS
        # state.  The actual cut cell is projected continuously below, so this
        # path is only a guard for a restart created by an older state layout.
        newly_free = fs_area >= self.section.full_area
        fs_area[newly_free] = boundary_area
        fs_discharge[newly_free] = boundary_discharge
        fs_next = central_upwind_wet_dry_step(
            WetDryState(fs_area, fs_discharge),
            dx=self.dx,
            dt=dt,
            section=self.section,
            cfl=self.config.cfl,
            dry_area_fraction=self.config.dry_area_fraction,
            manning_n=self.config.manning_n,
            darcy_friction=0.0,
            bed_slope=self.config.bed_slope,
            left_boundary=self.config.left_boundary,
            right_boundary="transmissive",
            right_ghost=(boundary_area, boundary_discharge),
        )

        pressurised_next = pressurised_moc_step(
            WetDryState(
                p_area_input,
                p_discharge_input,
            ),
            dx=self.dx,
            dt=dt,
            section=self.section,
            interface_velocity=interface_solution.pressurised_velocity,
            interface_head=interface_solution.pressurised_head,
            darcy_friction=self.config.darcy_friction,
            bed_slope=self.config.bed_slope,
            right_boundary=self.config.right_boundary,
        )
        area_new = np.empty(self.ncell, dtype=float)
        discharge_new = np.empty(self.ncell, dtype=float)
        area_new[:fs_stop] = fs_next.area
        discharge_new[:fs_stop] = fs_next.discharge
        area_new[pressurised_start:] = pressurised_next.area
        discharge_new[pressurised_start:] = pressurised_next.discharge
        # Provisional old cut state; it is overwritten by the continuous
        # projection below if the front remains in or leaves this cell.
        area_new[cut_old] = state.area[cut_old]
        discharge_new[cut_old] = state.discharge[cut_old]
        # Conservative-in-cell shock projection: the tracked interface sweeps
        # continuously through a cut cell instead of flipping its complete
        # state at a centre/face threshold.  This removes the grid-crossing
        # acoustic impulse while retaining the fitted states on both sides.
        cut_index = int(
            np.clip(
                np.floor(interface_new / self.dx),
                0,
                self.ncell - 1,
            )
        )
        if cut_index > cut_old:
            area_new[cut_old:cut_index] = boundary_area
            discharge_new[cut_old:cut_index] = boundary_discharge
        cut_left = cut_index * self.dx
        free_fraction = float(np.clip(
            (
                interface_new
                - max(wetting_front_new, cut_left)
            ) / self.dx,
            0.0,
            1.0,
        ))
        pressurised_fraction = float(np.clip(
            (cut_left + self.dx - interface_new) / self.dx,
            0.0,
            1.0,
        ))
        pressurised_cut_area = (
            area_new[cut_index]
            if cut_index >= pressurised_start
            else pressurised_next.area[0]
        )
        pressurised_cut_discharge = (
            discharge_new[cut_index]
            if cut_index >= pressurised_start
            else pressurised_next.discharge[0]
        )
        area_new[cut_index] = (
            free_fraction * boundary_area
            + pressurised_fraction * pressurised_cut_area
        )
        discharge_new[cut_index] = (
            free_fraction * boundary_discharge
            + pressurised_fraction * pressurised_cut_discharge
        )

        # The closed horizontal branch has no liquid mass flux through either
        # end.  MOC interpolation and the cut-cell projection are individually
        # consistent, but their composition leaves a mean-area residual.  That
        # residual is an elastic-volume correction of the *pressurised water
        # column*; it is not a second jump condition at the moving interface.
        #
        # An earlier implementation placed the complete correction in the cut
        # cell.  Because the fitted cut state is reconstructed on every substep,
        # the same stored surplus was removed and reinserted repeatedly.  By the
        # time the front reached the side T, the geometrically reconstructed cut
        # state was 0.60 A while the stored value was 1.12 A.  Handing that cell
        # to a finite-volume TPA solver therefore created a spurious O(100 kPa)
        # water-hammer impulse.
        #
        # A pressure correction cannot act outside the acoustic domain of
        # dependence.  The earlier implementation spread every substep's mean
        # area residual over the *complete* downstream water column.  A release
        # at the valve therefore changed the side-T pressure several metres
        # away in the first 1e-3 s substep, before a wave travelling at ``a``
        # could arrive.  The resulting nonlocal alternating pressure drove a
        # spurious drain/refill oscillation in the riser.
        #
        # The moving gas pocket is accompanied by a counter-current liquid
        # layer between the left-going wetting front and the right-going gas
        # nose.  The projection mismatch belongs first to that local layer,
        # not to a uniform compression of the remote full-water reach.  Apply
        # a bounded conservative correction over the geometrically wetted
        # free-surface interval.  Only a residual that cannot fit between zero
        # and the full circular section is placed in elastic storage, and that
        # fallback is restricted to the acoustic cone.  This preserves both
        # liquid volume and finite propagation.
        volume_before = float(np.sum(state.area) * self.dx)
        volume_raw = float(np.sum(area_new) * self.dx)
        volume_residual = volume_before - volume_raw
        cell_left = np.arange(self.ncell, dtype=float) * self.dx
        cell_right = cell_left + self.dx
        free_weight = np.clip(
            (
                np.minimum(cell_right, interface_new)
                - np.maximum(cell_left, wetting_front_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        pressurised_weight = np.clip(
            (
                np.minimum(cell_right, self.config.length)
                - np.maximum(cell_left, interface_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        free_cells = free_weight > 1.0e-14
        lower_area = np.zeros(self.ncell, dtype=float)
        upper_area = np.zeros(self.ncell, dtype=float)
        if np.any(free_cells):
            # Only the interface cut cell can contain both free-surface and
            # pressurised fractions.  Its pressure-side contribution is the
            # fitted state used in the conservative cut-cell projection.
            pressure_component = np.where(
                free_cells,
                pressurised_weight * pressurised_cut_area,
                0.0,
            )
            lower_area = pressure_component
            upper_area = (
                pressure_component
                + free_weight * self.section.full_area
            )
            # The FV step may already contain a small admissible reconstruction
            # overshoot.  The mass projection never clips that pre-existing
            # state; the bounds constrain only the new correction.
            lower_area = np.minimum(lower_area, area_new)
            upper_area = np.maximum(upper_area, area_new)

        remaining_volume = float(volume_residual)
        active = free_cells.copy()
        for _ in range(self.ncell + 1):
            weight_sum = float(np.sum(free_weight[active]))
            if weight_sum <= 1.0e-14 or abs(remaining_volume) <= 1.0e-16:
                break
            delta = remaining_volume / (self.dx * weight_sum)
            proposed = delta * free_weight[active]
            indices = np.flatnonzero(active)
            if remaining_volume > 0.0:
                allowed = upper_area[indices] - area_new[indices]
                applied = np.minimum(proposed, allowed)
            else:
                allowed = lower_area[indices] - area_new[indices]
                applied = np.maximum(proposed, allowed)
            area_new[indices] += applied
            used = float(np.sum(applied) * self.dx)
            remaining_volume -= used
            saturated = np.isclose(
                applied,
                allowed,
                rtol=0.0,
                atol=1.0e-16 * self.section.full_area,
            )
            if not np.any(saturated):
                break
            active[indices[saturated]] = False

        acoustic_front = min(
            self.config.length,
            self.config.valve_x
            + self.config.wave_speed * (state.time + dt),
        )
        elastic_weight = np.clip(
            (
                np.minimum(cell_right, acoustic_front)
                - np.maximum(cell_left, interface_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        old_elastic_area = area_new.copy()
        weight_sum = float(np.sum(elastic_weight))
        if abs(remaining_volume) > 1.0e-16:
            if weight_sum <= 1.0e-14:
                raise FloatingPointError(
                    "shock front leaves no local storage for mass projection"
                )
            area_new += (
                remaining_volume / (self.dx * weight_sum)
            ) * elastic_weight
        corrected = elastic_weight > 0.0
        if np.any(area_new[corrected] <= 0.0):
            raise FloatingPointError(
                "liquid mass projection would make a pressurised cell non-positive"
            )
        discharge_new[corrected] *= np.divide(
            area_new[corrected],
            np.maximum(
                old_elastic_area[corrected],
                1.0e-14 * self.section.full_area,
            ),
        )
        wetdry = WetDryState(area_new, discharge_new)

        volume_new = self._connected_gas_volume(
            wetdry.area,
            interface_new,
            boundary_area,
            wetting_front_new,
        )
        minimum_volume = (
            1.0e-9 * self.section.full_area * self.config.length
        )
        gas_new = state.gas.with_volume(max(volume_new, minimum_volume))
        pressure_new, vented = self._effective_pressure(
            time=state.time + dt,
            interface_x=interface_new,
            closed_pressure_abs=gas_new.pressure_abs,
            external_pressure_abs=external_pressure_abs,
        )
        return HorizontalState(
            time=state.time + dt,
            area=wetdry.area,
            discharge=wetdry.discharge,
            gas=gas_new,
            air_pressure_abs=pressure_new,
            interface_x=interface_new,
            interface_speed=speed,
            interface_free_surface_depth=boundary_depth,
            interface_free_surface_velocity=boundary_velocity,
            interface_pressurised_head=(
                interface_solution.pressurised_head
            ),
            interface_pressurised_velocity=(
                interface_solution.pressurised_velocity
            ),
            interface_residual_linf=interface_solution.residual_linf,
            wetting_front_x=wetting_front_new,
            vented=vented,
            nonlinear_converged=interface_solution.converged,
            liquid_volume_residual=volume_residual,
            cumulative_liquid_volume_residual=(
                state.cumulative_liquid_volume_residual + volume_residual
            ),
        )

    def step(
        self,
        state: HorizontalState,
        dt: float,
        *,
        external_pressure_abs: float | None = None,
    ) -> HorizontalState:
        """Advance by ``dt``, internally subcycling to the current CFL limit."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        current = state
        remaining = float(dt)
        while remaining > max(1.0e-14, 1.0e-12 * dt):
            stable = self.stable_timestep(current)
            substep = min(remaining, 0.999999 * stable)
            if substep <= 0.0 or not np.isfinite(substep):
                raise FloatingPointError("invalid horizontal-core time step")
            current = self._step_once(
                current, substep, external_pressure_abs
            )
            remaining -= substep
        return current

    def snapshot(self, state: HorizontalState) -> dict[str, object]:
        """Return fixed-grid fields and scalar keys used by frame adapters."""

        dry_area = self.config.dry_area_fraction * self.section.full_area
        velocity = np.divide(
            state.discharge,
            state.area,
            out=np.zeros_like(state.discharge),
            where=state.area > dry_area,
        )
        area_fraction = np.clip(
            state.area / self.section.full_area,
            0.0,
            None,
        )
        gas_head = (
            state.air_pressure_abs - self.config.atmospheric_pressure
        ) / (self.config.liquid_density * self.config.gravity)
        mode = (
            "wet_dry_extension+tosan_shockfit"
            if state.wetting_front_x > 0.5 * self.dx
            else "tosan_shockfit"
        )
        return {
            "time": float(state.time),
            "x": self.x.copy(),
            "area": state.area.copy(),
            "discharge": state.discharge.copy(),
            "velocity": velocity,
            "area_fraction": area_fraction,
            "interface_x": float(state.interface_x),
            "interface_speed": float(state.interface_speed),
            "interface_free_surface_depth": float(
                state.interface_free_surface_depth
            ),
            "interface_free_surface_velocity": float(
                state.interface_free_surface_velocity
            ),
            "interface_pressurised_head": float(
                state.interface_pressurised_head
            ),
            "interface_pressurised_velocity": float(
                state.interface_pressurised_velocity
            ),
            "interface_residual_linf": float(state.interface_residual_linf),
            "interface_converged": bool(state.nonlinear_converged),
            "air_pressure_abs": float(state.air_pressure_abs),
            "air_pressure_head_gauge": float(gas_head),
            "air_volume": float(state.gas.volume),
            "air_mass": float(state.gas.mass),
            "wetting_front_x": float(state.wetting_front_x),
            "numerical_wetting_front_x": self._wetting_front(
                state.area,
                state.wetting_front_x,
                area_fraction=self.config.dry_area_fraction,
            ),
            "vented": bool(state.vented),
            "mode": mode,
            "water_volume": float(np.sum(state.area) * self.dx),
            "liquid_volume_residual": float(state.liquid_volume_residual),
            "cumulative_liquid_volume_residual": float(
                state.cumulative_liquid_volume_residual
            ),
        }

    def run(
        self,
        state: HorizontalState,
        t_end: float,
        *,
        output_dt: float = 0.05,
        external_pressure_abs: float | None = None,
    ) -> list[dict[str, object]]:
        """Run on the full configured domain and return requested snapshots."""

        if t_end < state.time:
            raise ValueError("t_end cannot precede the input state")
        if output_dt <= 0.0:
            raise ValueError("output_dt must be positive")
        snapshots = [self.snapshot(state)]
        current = state
        next_output = state.time + output_dt
        tolerance = 1.0e-12 * max(1.0, t_end)
        while current.time < t_end - tolerance:
            target = min(next_output, t_end)
            current = self.step(
                current,
                target - current.time,
                external_pressure_abs=external_pressure_abs,
            )
            snapshots.append(self.snapshot(current))
            if target >= next_output - tolerance:
                next_output += output_dt
        return snapshots


__all__ = [
    "CircularSection",
    "HorizontalConfig",
    "HorizontalState",
    "PolytropicGasInventory",
    "Tosan2021HorizontalShockFit",
    "TosanInterfaceData",
    "TosanInterfaceSolution",
    "WetDryState",
    "advance_shock_position",
    "central_upwind_wet_dry_step",
    "circular_dry_bed_gate_state",
    "pressurised_moc_step",
    "solve_oriented_interface",
    "solve_tosan_negative_interface_case_b",
    "solve_tosan_positive_interface",
]
