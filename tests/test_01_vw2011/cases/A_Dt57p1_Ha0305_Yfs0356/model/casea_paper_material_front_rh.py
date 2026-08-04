"""Strict paper-equation RH traces for one Case-A material front.

This module is an independent audit replacement for the liquid/front part of
``casea_material_front_rh_adapter``.  It implements the *reduced time-marching*
closure in Eqs. (3), (12), (14), (15), (17), and (20)--(21) of
``model_algorithm_revised_20260803``.  In particular,

* the pressurised interface area ``A_p,Gamma`` is an unknown; it is not set to
  the undeformed full area ``A_f``;
* the stratified liquid pressure potential uses the complete ``Lambda_d``
  (gas-pressure, transverse gravity/buoyancy, and inviscid slip terms);
* the gas trace is material, ``u_g,Gamma=w_Gamma``; and
* the gas ALE momentum impulse is the absolute-pressure value
  ``P_g,Gamma A_g,Gamma``.

There is no prescribed interface speed, speed cap, state fill, state clip, or
fallback numerical flux here.  Polynomial roots are enumerated and checked
against the original dimensional balances.  If the equations have more than
one admissible root, the caller must provide a physical entropy/active-set
selection; this module does not silently choose a convenient trajectory.

The detailed paper Riemann problem has distinct stratified trace relations for
the slow, middle, and fast active sets.  The production time-marching text
instead uses the adjacent stratified trace in all three sets.  This module
faithfully represents that reduced closure and reports the active-set label;
it must not be described as the full branch-dependent reference problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.polynomial import Polynomial

from casea_material_front_cutcell import (
    InterfaceTraces,
    PressurisedFlux,
    PressurisedState,
    StratifiedFlux,
    StratifiedState,
)


PressurisedSide = Literal["left", "right"]
ActiveSet = Literal["slow", "middle", "fast"]


class PaperFrontClosureError(RuntimeError):
    """The strict reduced paper closure has no unique admissible solution."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class PaperFrontPhysics:
    diameter: float
    liquid_wave_speed: float
    liquid_density: float = 998.0
    gravity: float = 9.81
    reference_pressure: float = 101_325.0
    gas_sound_speed: float = math.sqrt(287.05 * 293.0)
    cos_inclination: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.diameter,
            self.liquid_wave_speed,
            self.liquid_density,
            self.gravity,
            self.reference_pressure,
            self.gas_sound_speed,
            self.cos_inclination,
        )
        if not _finite(*values):
            raise ValueError("paper-front physical data must be finite")
        if min(values[:-1]) <= 0.0:
            raise ValueError("paper-front physical scales must be positive")
        if not -1.0 <= self.cos_inclination <= 1.0:
            raise ValueError("cos(inclination) must lie in [-1, 1]")

    @property
    def full_area(self) -> float:
        return math.pi * self.diameter**2 / 4.0


@dataclass(frozen=True)
class AffineGasPressureLaw:
    """Interface pressure ``p_Gamma(w)=intercept+slope*w``.

    ``fixed`` is used when a finite junction supplies the common pressure.
    ``from_acoustic_trace`` is the paper's linear incoming gas characteristic

    ``p_Gamma=p_1+rho_1*c_g*(w-u_g,1)``.

    Positivity is checked on every candidate rather than enforced by a floor.
    """

    intercept: float
    slope: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.intercept, self.slope):
            raise ValueError("gas-pressure law must be finite")

    @classmethod
    def fixed(cls, pressure_absolute: float) -> "AffineGasPressureLaw":
        if not math.isfinite(pressure_absolute) or pressure_absolute <= 0.0:
            raise ValueError("fixed absolute gas pressure must be positive")
        return cls(float(pressure_absolute), 0.0)

    @classmethod
    def from_acoustic_trace(
        cls,
        *,
        density: float,
        velocity: float,
        sound_speed: float,
    ) -> "AffineGasPressureLaw":
        if not _finite(density, velocity, sound_speed):
            raise ValueError("gas acoustic trace must be finite")
        if density <= 0.0 or sound_speed <= 0.0:
            raise ValueError("gas density and sound speed must be positive")
        pressure = density * sound_speed**2
        impedance = density * sound_speed
        return cls(
            intercept=float(pressure - impedance * velocity),
            slope=float(impedance),
        )

    def pressure(self, speed: float) -> float:
        if not math.isfinite(speed):
            raise ValueError("front speed must be finite")
        return float(self.intercept + self.slope * speed)


@dataclass(frozen=True)
class PaperFrontCandidate:
    pressurised_side: PressurisedSide
    speed: float
    pressurised_area: float
    pressurised_discharge: float
    stratified_liquid_area: float
    stratified_liquid_discharge: float
    gas_pressure_absolute: float
    gas_density: float
    lambda_d: float
    adjacent_lambda_d: float
    active_set: ActiveSet | None
    incoming_pressurised_margin: float
    characteristic_residual: float
    liquid_mass_residual: float
    liquid_momentum_residual: float
    gas_pressure_residual: float

    @property
    def residual_linf(self) -> float:
        return max(
            abs(self.characteristic_residual),
            abs(self.liquid_mass_residual),
            abs(self.liquid_momentum_residual),
            abs(self.gas_pressure_residual),
        )


def _top_width_from_area(area: float, physics: PaperFrontPhysics) -> float:
    """Invert a circular segment without clipping an inadmissible state."""

    full = physics.full_area
    if not math.isfinite(area) or not 0.0 < area < full:
        raise ValueError("stratified liquid area must be strictly partial")
    lo = 0.0
    hi = physics.diameter
    radius = 0.5 * physics.diameter
    for _ in range(80):
        depth = 0.5 * (lo + hi)
        argument = 1.0 - depth / radius
        theta = 2.0 * math.acos(argument)
        trial = 0.5 * radius**2 * (theta - math.sin(theta))
        if trial < area:
            lo = depth
        else:
            hi = depth
    depth = 0.5 * (lo + hi)
    width = 2.0 * math.sqrt(depth * (physics.diameter - depth))
    if width <= 0.0:
        raise ValueError("partial circular section has zero top width")
    return width


def paper_lambda_d(
    *,
    liquid_area: float,
    liquid_discharge: float,
    gas_pressure_absolute: float,
    gas_density: float,
    gas_velocity: float,
    physics: PaperFrontPhysics,
) -> float:
    """Evaluate paper Eq. (3) without a positivity floor or regularisation."""

    full = physics.full_area
    if not _finite(
        liquid_area,
        liquid_discharge,
        gas_pressure_absolute,
        gas_density,
        gas_velocity,
    ):
        raise ValueError("Lambda_d inputs must be finite")
    if not 0.0 < liquid_area < full:
        raise ValueError("Lambda_d requires 0 < A_l < A_f")
    if gas_pressure_absolute <= 0.0 or gas_density <= 0.0:
        raise ValueError("gas pressure and density must be positive")
    gas_area = full - liquid_area
    liquid_velocity = liquid_discharge / liquid_area
    top_width = _top_width_from_area(liquid_area, physics)
    zeta = physics.cos_inclination / top_width
    gas_head = (
        gas_pressure_absolute - physics.reference_pressure
    ) / (physics.liquid_density * physics.gravity)
    return float(
        2.0 * physics.gravity * gas_head / liquid_area
        + (
            (physics.liquid_density - gas_density)
            / physics.liquid_density
        )
        * physics.gravity
        * zeta
        - gas_density
        / physics.liquid_density
        * (gas_velocity - liquid_velocity) ** 2
        / gas_area
    )


def _orientation_eta(side: PressurisedSide) -> float:
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    raise ValueError("pressurised_side must be 'left' or 'right'")


def _active_set(
    *,
    side: PressurisedSide,
    speed: float,
    liquid_velocity: float,
    adjacent_lambda_d: float,
    liquid_area: float,
) -> ActiveSet | None:
    if adjacent_lambda_d <= 0.0:
        return None
    celerity = math.sqrt(adjacent_lambda_d * liquid_area)
    # The paper convention is P|S.  Reflect both velocity and speed for S|P.
    if side == "left":
        local_speed = speed
        local_velocity = liquid_velocity
    else:
        local_speed = -speed
        local_velocity = -liquid_velocity
    if local_speed < local_velocity - celerity:
        return "slow"
    if local_speed > local_velocity + celerity:
        return "fast"
    return "middle"


def evaluate_candidate_from_pressurised_area(
    pressurised_area: float,
    *,
    pressurised_foot: PressurisedState,
    stratified_foot: StratifiedState,
    pressurised_side: PressurisedSide,
    pressure_law: AffineGasPressureLaw,
    physics: PaperFrontPhysics,
) -> PaperFrontCandidate:
    """Evaluate the strict reduced-paper balances at one ``A_p,Gamma``."""

    full = physics.full_area
    if not math.isfinite(pressurised_area) or pressurised_area <= 0.0:
        raise ValueError("pressurised interface area must be positive")
    al = stratified_foot.liquid_area
    if not 0.0 < al < full:
        raise ValueError("stratified liquid area must be strictly partial")
    gas_area = full - al
    gas_density_cell = stratified_foot.gas_mass / gas_area
    if gas_density_cell <= 0.0:
        raise ValueError("adjacent gas density must be positive")
    gas_velocity_cell = stratified_foot.gas_velocity
    gas_pressure_cell = gas_density_cell * physics.gas_sound_speed**2
    ul = stratified_foot.liquid_discharge / al

    eta = _orientation_eta(pressurised_side)
    up_foot = pressurised_foot.discharge / pressurised_foot.area
    up = up_foot + eta * physics.liquid_wave_speed * (
        pressurised_area - pressurised_foot.area
    ) / full
    qp = pressurised_area * up
    area_jump = pressurised_area - al
    if area_jump == 0.0:
        raise PaperFrontClosureError(
            "A_p,Gamma=A_l,Gamma makes the reduced mass equation singular"
        )
    speed = (qp - stratified_foot.liquid_discharge) / area_jump
    pressure = pressure_law.pressure(speed)
    if pressure <= 0.0:
        raise PaperFrontClosureError(
            "gas acoustic characteristic produced non-positive pressure"
        )
    gas_density = pressure / physics.gas_sound_speed**2
    lambda_front = paper_lambda_d(
        liquid_area=al,
        liquid_discharge=stratified_foot.liquid_discharge,
        gas_pressure_absolute=pressure,
        gas_density=gas_density,
        gas_velocity=speed,
        physics=physics,
    )
    adjacent_lambda = paper_lambda_d(
        liquid_area=al,
        liquid_discharge=stratified_foot.liquid_discharge,
        gas_pressure_absolute=gas_pressure_cell,
        gas_density=gas_density_cell,
        gas_velocity=gas_velocity_cell,
        physics=physics,
    )

    hs = physics.liquid_wave_speed**2 * (
        pressurised_area - full
    ) / (physics.gravity * full)
    pressurised_flux = qp**2 / pressurised_area + physics.gravity * full * hs
    stratified_flux = (
        stratified_foot.liquid_discharge**2 / al
        + 0.5 * lambda_front * al**2
    )
    characteristic = up - up_foot - eta * physics.liquid_wave_speed * (
        pressurised_area - pressurised_foot.area
    ) / full
    mass = qp - stratified_foot.liquid_discharge - speed * area_jump
    momentum = (
        pressurised_flux
        - stratified_flux
        - speed * (qp - stratified_foot.liquid_discharge)
    )
    gas_residual = pressure - pressure_law.pressure(speed)
    incoming_margin = (
        up + physics.liquid_wave_speed - speed
        if pressurised_side == "left"
        else speed - up + physics.liquid_wave_speed
    )
    return PaperFrontCandidate(
        pressurised_side=pressurised_side,
        speed=float(speed),
        pressurised_area=float(pressurised_area),
        pressurised_discharge=float(qp),
        stratified_liquid_area=float(al),
        stratified_liquid_discharge=float(
            stratified_foot.liquid_discharge
        ),
        gas_pressure_absolute=float(pressure),
        gas_density=float(gas_density),
        lambda_d=float(lambda_front),
        adjacent_lambda_d=float(adjacent_lambda),
        active_set=_active_set(
            side=pressurised_side,
            speed=speed,
            liquid_velocity=ul,
            adjacent_lambda_d=adjacent_lambda,
            liquid_area=al,
        ),
        incoming_pressurised_margin=float(incoming_margin),
        characteristic_residual=float(characteristic),
        liquid_mass_residual=float(mass),
        liquid_momentum_residual=float(momentum),
        gas_pressure_residual=float(gas_residual),
    )


def _selected_eq20_area(
    speed: float,
    *,
    pressurised_foot: PressurisedState,
    stratified_foot: StratifiedState,
    side: PressurisedSide,
    physics: PaperFrontPhysics,
) -> float | None:
    """Return the positive Eq. (20) root nearest the incoming foot area."""

    eta = _orientation_eta(side)
    full = physics.full_area
    u0 = pressurised_foot.discharge / pressurised_foot.area
    c1 = eta * physics.liquid_wave_speed / full
    c0 = u0 - c1 * pressurised_foot.area
    roots = np.roots(
        [
            c1,
            c0 - speed,
            speed * stratified_foot.liquid_area
            - stratified_foot.liquid_discharge,
        ]
    )
    positive = [
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1.0e-10 * max(1.0, abs(root.real))
        and root.real > 0.0
    ]
    if not positive:
        return None
    return min(positive, key=lambda value: abs(value - pressurised_foot.area))


def enumerate_paper_front_candidates(
    *,
    pressurised_foot: PressurisedState,
    stratified_foot: StratifiedState,
    pressurised_side: PressurisedSide,
    pressure_law: AffineGasPressureLaw,
    physics: PaperFrontPhysics,
    residual_relative_tolerance: float = 2.0e-8,
) -> tuple[PaperFrontCandidate, ...]:
    """Enumerate all admissible roots of the reduced paper RH equations.

    The scalar equation is converted to a polynomial in ``A_p,Gamma`` after
    multiplying only by powers of the nonzero area jump.  Every numerical root
    is substituted back into the unmultiplied balances, so a pole cannot be
    accepted as a physical solution.
    """

    if residual_relative_tolerance <= 0.0:
        raise ValueError("residual tolerance must be positive")
    eta = _orientation_eta(pressurised_side)
    full = physics.full_area
    al = stratified_foot.liquid_area
    if not 0.0 < al < full:
        raise ValueError("stratified liquid area must be strictly partial")
    gas_area = full - al
    ul = stratified_foot.liquid_discharge / al
    up0 = pressurised_foot.discharge / pressurised_foot.area

    x = Polynomial([0.0, 1.0])
    y = x - al
    c1 = eta * physics.liquid_wave_speed / full
    c0 = up0 - c1 * pressurised_foot.area
    u = Polynomial([c0, c1])
    q = x * u
    d = q - stratified_foot.liquid_discharge
    pressure_numerator = pressure_law.intercept * y + pressure_law.slope * d
    relative_velocity_numerator = d - ul * y
    top_width = _top_width_from_area(al, physics)
    zeta = physics.cos_inclination / top_width

    # F_f = C0 + Cp*p + Cs*p*(w-u_l)^2, with p=Pnum/y and
    # w-u_l=(d-u_l*y)/y.
    c_flux0 = (
        stratified_foot.liquid_discharge**2 / al
        - al * physics.reference_pressure / physics.liquid_density
        + 0.5 * al**2 * physics.gravity * zeta
    )
    c_pressure = (
        al / physics.liquid_density
        - 0.5
        * al**2
        * physics.gravity
        * zeta
        / (physics.liquid_density * physics.gas_sound_speed**2)
    )
    c_slip = -0.5 * al**2 / (
        physics.liquid_density * physics.gas_sound_speed**2 * gas_area
    )
    fp = x * u * u + physics.liquid_wave_speed**2 * (x - full)
    polynomial = (
        (fp - c_flux0) * y**3
        - c_pressure * pressure_numerator * y**2
        - c_slip * pressure_numerator * relative_velocity_numerator**2
        - d**2 * y**2
    )
    coefficients = np.asarray(polynomial.coef, dtype=float)
    coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
    while (
        coefficients.size > 1
        and abs(coefficients[-1])
        <= 1.0e-13 * coefficient_scale
    ):
        coefficients = coefficients[:-1]
    roots = np.polynomial.polynomial.polyroots(coefficients)

    candidates: list[PaperFrontCandidate] = []
    area_scale = max(full, pressurised_foot.area, 1.0e-12)
    for root in roots:
        if abs(root.imag) > 2.0e-8 * max(area_scale, abs(root.real)):
            continue
        ap = float(root.real)
        if ap <= 0.0 or abs(ap - al) <= 1.0e-10 * area_scale:
            continue
        try:
            candidate = evaluate_candidate_from_pressurised_area(
                ap,
                pressurised_foot=pressurised_foot,
                stratified_foot=stratified_foot,
                pressurised_side=pressurised_side,
                pressure_law=pressure_law,
                physics=physics,
            )
        except (ValueError, PaperFrontClosureError, FloatingPointError):
            continue
        selected_area = _selected_eq20_area(
            candidate.speed,
            pressurised_foot=pressurised_foot,
            stratified_foot=stratified_foot,
            side=pressurised_side,
            physics=physics,
        )
        if selected_area is None or abs(ap - selected_area) > 2.0e-7 * area_scale:
            continue
        if candidate.incoming_pressurised_margin <= 0.0:
            continue
        fp_scale = (
            abs(candidate.pressurised_discharge**2 / ap)
            + abs(physics.liquid_wave_speed**2 * (ap - full))
        )
        ff_scale = (
            abs(candidate.stratified_liquid_discharge**2 / al)
            + abs(0.5 * candidate.lambda_d * al**2)
        )
        momentum_scale = max(
            1.0e-12,
            fp_scale
            + ff_scale
            + abs(candidate.speed * candidate.pressurised_discharge)
            + abs(candidate.speed * candidate.stratified_liquid_discharge),
        )
        if (
            abs(candidate.liquid_momentum_residual)
            > residual_relative_tolerance * momentum_scale
        ):
            continue
        if any(
            abs(candidate.pressurised_area - old.pressurised_area)
            <= 2.0e-8 * area_scale
            for old in candidates
        ):
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda value: value.speed)
    return tuple(candidates)


def require_unique_paper_front_candidate(**kwargs) -> PaperFrontCandidate:
    candidates = enumerate_paper_front_candidates(**kwargs)
    if len(candidates) != 1:
        raise PaperFrontClosureError(
            "strict reduced paper closure requires exactly one admissible "
            f"root, found {len(candidates)}"
        )
    return candidates[0]


def candidate_to_ale_traces(
    candidate: PaperFrontCandidate,
    *,
    physics: PaperFrontPhysics,
) -> InterfaceTraces:
    """Convert a paper candidate to the exact cut-cell trace convention."""

    full = physics.full_area
    al = candidate.stratified_liquid_area
    gas_area = full - al
    if not 0.0 < gas_area < full:
        raise ValueError("paper candidate must retain a finite gas area")
    gas_mass = candidate.gas_density * gas_area
    gas_momentum = gas_mass * candidate.speed
    pressurised = PressurisedState(
        candidate.pressurised_area,
        candidate.pressurised_discharge,
    )
    stratified = StratifiedState(
        gas_mass,
        gas_momentum,
        al,
        candidate.stratified_liquid_discharge,
    )
    return InterfaceTraces(
        speed=candidate.speed,
        pressurised_state=pressurised,
        pressurised_flux=PressurisedFlux(
            area=candidate.pressurised_discharge,
            momentum=(
                candidate.pressurised_discharge**2
                / candidate.pressurised_area
                + physics.liquid_wave_speed**2
                * (candidate.pressurised_area - full)
            ),
        ),
        stratified_state=stratified,
        stratified_flux=StratifiedFlux(
            gas_mass=gas_momentum,
            gas_momentum=(
                gas_mass * candidate.speed**2
                + candidate.gas_pressure_absolute * gas_area
            ),
            liquid_area=candidate.stratified_liquid_discharge,
            liquid_momentum=(
                candidate.stratified_liquid_discharge**2 / al
                + 0.5 * candidate.lambda_d * al**2
            ),
        ),
    )


__all__ = [
    "ActiveSet",
    "AffineGasPressureLaw",
    "PaperFrontCandidate",
    "PaperFrontClosureError",
    "PaperFrontPhysics",
    "candidate_to_ale_traces",
    "enumerate_paper_front_candidates",
    "evaluate_candidate_from_pressurised_area",
    "paper_lambda_d",
    "require_unique_paper_front_candidate",
]
