"""Strict inclined stratified two-fluid branch core for Case A.

This module is a small, independent implementation of the *interior* branch
equations stated in ``model_algorithm_revised_20260803``.  It uses the four
conserved variables

``U = (A_g rho_g, rho_g Q_g, A_l, Q_l)``

and the physical flux in main-text Eq. (2).  The liquid restoring coefficient
is exactly Eq. (3), with

``zeta = cos(theta) / [D sin(gamma/2)]``.

The coordinate ``x`` points along the pipe.  Positive ``theta`` is uphill, so
the axial-gravity term is negative for upward flow.  The isothermal gas law is
``P_g = c_g**2 rho_g``.  The gas-pressure head in the liquid potential is a
gauge head, ``(P_g-P_ref)/(rho_l g)``; this is the pressure-perturbation form
used by the current model analysis and avoids assigning a liquid restoring
force to a spatially uniform atmospheric pressure.

The finite-volume face flux is the block Rusanov flux of Eq. (30): gas and
liquid blocks use their own physical spectral radii.  There is one and only
one returned flux per face, which is shared by its two cells.  This core has
no state clipping, area cap, density floor, celerity floor, prescribed wave,
or alternate Riemann-solver fallback.  A non-finite, non-admissible, or
elliptic (Lambda_d < 0) state rejects the whole Euler stage.

This file deliberately does not implement a moving pressurised--stratified
front, a T-junction boundary Riemann problem, topology conversion, or a vent
boundary.  Those are separate closures and must be supplied before this
interior operator can be used as the Case-A riser solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


NUMERICAL_FLUX = "block_rusanov_eq30"
RIEMANN_FALLBACK_AVAILABLE = False
INTERIOR_BRANCH_CORE_READY = True
COMPLETE_RISER_MODEL_READY = False
MISSING_RISER_CLOSURES = (
    "inclined_pressurised_stratified_front",
    "three_branch_tjunction_riemann_problem",
    "top_free_surface_and_vent_event",
    "vertical_wall_and_interfacial_shear",
    "vertical_phase_topology",
)


class InclinedTwoFluidError(RuntimeError):
    """Base class for rejected inclined-branch operations."""


class StateAdmissibilityError(InclinedTwoFluidError):
    """A supplied or updated conserved state is not physically admissible."""


class LossOfHyperbolicityError(InclinedTwoFluidError):
    """Eq. (3) gives ``Lambda_d < 0`` for a requested branch state."""


class CFLViolationError(InclinedTwoFluidError):
    """The requested explicit Euler stage violates the configured CFL bound."""


class StageAdmissibilityError(InclinedTwoFluidError):
    """An Euler candidate was rejected without mutating the input state."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class InclinedTwoFluidParameters:
    """Physical and numerical data for one uniform circular branch.

    ``inclination`` is measured from horizontal in radians.  The supported
    range is ``[-pi/2, pi/2]``; reverse-oriented graph edges should reverse
    their local coordinate instead of passing an angle outside this range.
    """

    diameter: float
    inclination: float
    liquid_density: float = 998.0
    gravity: float = 9.81
    gas_sound_speed: float = math.sqrt(287.05 * 293.0)
    reference_pressure: float = 101_325.0
    maximum_cfl: float = 1.0

    def __post_init__(self) -> None:
        if not _finite(
            self.diameter,
            self.inclination,
            self.liquid_density,
            self.gravity,
            self.gas_sound_speed,
            self.reference_pressure,
            self.maximum_cfl,
        ):
            raise ValueError("inclined two-fluid parameters must be finite")
        if min(
            self.diameter,
            self.liquid_density,
            self.gravity,
            self.gas_sound_speed,
            self.reference_pressure,
            self.maximum_cfl,
        ) <= 0.0:
            raise ValueError("inclined two-fluid physical scales must be positive")
        if not -0.5 * math.pi <= self.inclination <= 0.5 * math.pi:
            raise ValueError("inclination must lie in [-pi/2, pi/2]")

    @property
    def full_area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def reference_gas_density(self) -> float:
        return self.reference_pressure / self.gas_sound_speed**2

    @property
    def cosine(self) -> float:
        # Preserve the exact horizontal/vertical model limits instead of
        # allowing floating-point cos(pi/2) to leak a transverse-gravity term.
        if self.inclination == 0.0:
            return 1.0
        if abs(self.inclination) == 0.5 * math.pi:
            return 0.0
        return math.cos(self.inclination)

    @property
    def sine(self) -> float:
        if self.inclination == 0.0:
            return 0.0
        if self.inclination == 0.5 * math.pi:
            return 1.0
        if self.inclination == -0.5 * math.pi:
            return -1.0
        return math.sin(self.inclination)


@dataclass(frozen=True)
class InclinedTwoFluidState:
    """Per-unit-length conserved state ``(m_g, j_g, A_l, Q_l)``."""

    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_discharge: float

    def __post_init__(self) -> None:
        if not _finite(*self.vector()):
            raise StateAdmissibilityError("two-fluid conserved state must be finite")
        if self.gas_mass <= 0.0:
            raise StateAdmissibilityError("gas mass per unit length must be positive")
        if self.liquid_area <= 0.0:
            raise StateAdmissibilityError("liquid area must be positive")

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_discharge),
        )


@dataclass(frozen=True)
class CircularSegmentState:
    central_angle: float
    liquid_depth: float
    top_width: float
    depth_area_derivative: float


@dataclass(frozen=True)
class InclinedTwoFluidPrimitive:
    gas_area: float
    gas_density: float
    gas_velocity: float
    gas_pressure_absolute: float
    gas_pressure_gauge: float
    liquid_velocity: float
    liquid_depth: float
    top_width: float
    zeta: float
    lambda_d: float
    liquid_celerity: float

    @property
    def neutral_ikh_state(self) -> bool:
        return self.lambda_d == 0.0


@dataclass(frozen=True)
class InclinedTwoFluidFlux:
    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_momentum: float

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_momentum),
        )


@dataclass(frozen=True)
class MomentumClosureSource:
    """Optional already-decoupled wall/interfacial momentum sources.

    These values are additive rates per unit branch length.  No default
    friction law is hidden in this core; zero means a frictionless branch.
    """

    gas_momentum: float = 0.0
    liquid_momentum: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.gas_momentum, self.liquid_momentum):
            raise ValueError("extra momentum sources must be finite")


@dataclass(frozen=True)
class InclinedTwoFluidSource:
    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_momentum: float

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_momentum),
        )


@dataclass(frozen=True)
class BlockRusanovDiagnostics:
    method: str
    gas_spectral_radius: float
    liquid_spectral_radius: float
    lambda_left: float
    lambda_right: float
    lambda_face: float
    fallback_available: bool
    fallback_used: bool


@dataclass(frozen=True)
class BlockRusanovResult:
    flux: InclinedTwoFluidFlux
    diagnostics: BlockRusanovDiagnostics


@dataclass(frozen=True)
class InclinedBranchBoundaryStates:
    """Ghost-cell states immediately outside the left and right boundaries."""

    left: InclinedTwoFluidState
    right: InclinedTwoFluidState


@dataclass(frozen=True)
class InclinedBranchInventory:
    gas_mass: float
    gas_momentum: float
    liquid_volume: float
    liquid_discharge: float

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_volume),
            float(self.liquid_discharge),
        )


@dataclass(frozen=True)
class InclinedBranchStageLedger:
    dt: float
    cell_width: float
    initial: InclinedBranchInventory
    final: InclinedBranchInventory
    expected_change: InclinedBranchInventory
    residual: InclinedBranchInventory
    left_boundary_flux: InclinedTwoFluidFlux
    right_boundary_flux: InclinedTwoFluidFlux
    integrated_source: InclinedBranchInventory


@dataclass(frozen=True)
class InclinedBranchEulerResult:
    states: tuple[InclinedTwoFluidState, ...]
    face_fluxes: tuple[InclinedTwoFluidFlux, ...]
    face_diagnostics: tuple[BlockRusanovDiagnostics, ...]
    cell_sources: tuple[InclinedTwoFluidSource, ...]
    maximum_signal_speed: float
    cfl: float
    ledger: InclinedBranchStageLedger


def circular_segment_from_area(
    liquid_area: float,
    params: InclinedTwoFluidParameters,
) -> CircularSegmentState:
    """Invert exact circular-segment area without an area cap or floor."""

    if not math.isfinite(liquid_area):
        raise StateAdmissibilityError("liquid area must be finite")
    if not 0.0 < liquid_area < params.full_area:
        raise StateAdmissibilityError(
            "stratified liquid area must lie strictly inside (0, A_f)"
        )
    target = liquid_area / params.full_area
    lower = 0.0
    upper = 2.0 * math.pi
    # A/A_f = (gamma - sin(gamma))/(2*pi), monotone on (0, 2*pi).
    for _ in range(160):
        gamma = 0.5 * (lower + upper)
        fraction = (gamma - math.sin(gamma)) / (2.0 * math.pi)
        if fraction < target:
            lower = gamma
        else:
            upper = gamma
    gamma = 0.5 * (lower + upper)
    half = 0.5 * gamma
    top_width = params.diameter * math.sin(half)
    if not math.isfinite(top_width) or top_width <= 0.0:
        raise StateAdmissibilityError("circular-segment top width is non-positive")
    radius = 0.5 * params.diameter
    depth = radius * (1.0 - math.cos(half))
    return CircularSegmentState(
        central_angle=gamma,
        liquid_depth=depth,
        top_width=top_width,
        depth_area_derivative=1.0 / top_width,
    )


def primitive_state(
    state: InclinedTwoFluidState,
    params: InclinedTwoFluidParameters,
    *,
    require_hyperbolic: bool = True,
) -> InclinedTwoFluidPrimitive:
    """Return exact primitive variables and Eq. (3) diagnostics.

    The neutral state ``Lambda_d == 0`` is retained exactly.  A negative value
    is rejected when ``require_hyperbolic`` is true; it is never replaced by a
    numerical celerity.
    """

    gas_area = params.full_area - state.liquid_area
    if gas_area <= 0.0:
        raise StateAdmissibilityError(
            "stratified state must leave a strictly positive gas area"
        )
    geometry = circular_segment_from_area(state.liquid_area, params)
    gas_density = state.gas_mass / gas_area
    if not math.isfinite(gas_density) or gas_density <= 0.0:
        raise StateAdmissibilityError("derived gas density must be positive")
    gas_velocity = state.gas_momentum / state.gas_mass
    liquid_velocity = state.liquid_discharge / state.liquid_area
    pressure_absolute = params.gas_sound_speed**2 * gas_density
    pressure_gauge = pressure_absolute - params.reference_pressure
    zeta = params.cosine * geometry.depth_area_derivative
    pressure_term = 2.0 * pressure_gauge / (
        params.liquid_density * state.liquid_area
    )
    buoyancy_term = (
        (params.liquid_density - gas_density)
        / params.liquid_density
        * params.gravity
        * zeta
    )
    slip_term = -(
        gas_density
        / params.liquid_density
        * (gas_velocity - liquid_velocity) ** 2
        / gas_area
    )
    lambda_d = pressure_term + buoyancy_term + slip_term
    if not _finite(
        gas_velocity,
        liquid_velocity,
        pressure_absolute,
        pressure_gauge,
        zeta,
        lambda_d,
    ):
        raise StateAdmissibilityError("derived two-fluid primitive is non-finite")
    if pressure_absolute <= 0.0:
        raise StateAdmissibilityError("isothermal absolute gas pressure is non-positive")
    if lambda_d < 0.0 and require_hyperbolic:
        raise LossOfHyperbolicityError(
            "Eq. (3) gives Lambda_d < 0; the inviscid stratified subsystem "
            f"is not hyperbolic (Lambda_d={lambda_d:.12g})"
        )
    liquid_celerity = math.sqrt(lambda_d * state.liquid_area) if lambda_d >= 0.0 else math.nan
    return InclinedTwoFluidPrimitive(
        gas_area=gas_area,
        gas_density=gas_density,
        gas_velocity=gas_velocity,
        gas_pressure_absolute=pressure_absolute,
        gas_pressure_gauge=pressure_gauge,
        liquid_velocity=liquid_velocity,
        liquid_depth=geometry.liquid_depth,
        top_width=geometry.top_width,
        zeta=zeta,
        lambda_d=lambda_d,
        liquid_celerity=liquid_celerity,
    )


def physical_flux(
    state: InclinedTwoFluidState,
    params: InclinedTwoFluidParameters,
) -> InclinedTwoFluidFlux:
    """Evaluate the four-component physical flux in main-text Eq. (2)."""

    primitive = primitive_state(state, params)
    return InclinedTwoFluidFlux(
        gas_mass=state.gas_momentum,
        gas_momentum=(
            state.gas_momentum * primitive.gas_velocity
            + primitive.gas_pressure_absolute * primitive.gas_area
        ),
        liquid_area=state.liquid_discharge,
        liquid_momentum=(
            state.liquid_discharge * primitive.liquid_velocity
            + 0.5 * primitive.lambda_d * state.liquid_area**2
        ),
    )


def block_rusanov_flux(
    left: InclinedTwoFluidState,
    right: InclinedTwoFluidState,
    params: InclinedTwoFluidParameters,
) -> BlockRusanovResult:
    """Return the branch-dependent block Rusanov flux in Eq. (30).

    Following the text below Eq. (30), the liquid restoring coefficient in the
    central face flux is the arithmetic face average.  The exact physical
    celerities from Eq. (7) set the liquid dissipation; no regularized floor is
    introduced.
    """

    left_primitive = primitive_state(left, params)
    right_primitive = primitive_state(right, params)
    lambda_face = 0.5 * (
        left_primitive.lambda_d + right_primitive.lambda_d
    )
    gas_radius = max(
        abs(left_primitive.gas_velocity) + params.gas_sound_speed,
        abs(right_primitive.gas_velocity) + params.gas_sound_speed,
    )
    liquid_radius = max(
        abs(left_primitive.liquid_velocity) + left_primitive.liquid_celerity,
        abs(right_primitive.liquid_velocity) + right_primitive.liquid_celerity,
    )

    gas_mass_left = left.gas_momentum
    gas_mass_right = right.gas_momentum
    gas_momentum_left = (
        left.gas_momentum * left_primitive.gas_velocity
        + left_primitive.gas_pressure_absolute * left_primitive.gas_area
    )
    gas_momentum_right = (
        right.gas_momentum * right_primitive.gas_velocity
        + right_primitive.gas_pressure_absolute * right_primitive.gas_area
    )
    liquid_area_left = left.liquid_discharge
    liquid_area_right = right.liquid_discharge
    liquid_momentum_left = (
        left.liquid_discharge * left_primitive.liquid_velocity
        + 0.5 * lambda_face * left.liquid_area**2
    )
    liquid_momentum_right = (
        right.liquid_discharge * right_primitive.liquid_velocity
        + 0.5 * lambda_face * right.liquid_area**2
    )

    left_vector = left.vector()
    right_vector = right.vector()
    central = (
        0.5 * (gas_mass_left + gas_mass_right),
        0.5 * (gas_momentum_left + gas_momentum_right),
        0.5 * (liquid_area_left + liquid_area_right),
        0.5 * (liquid_momentum_left + liquid_momentum_right),
    )
    radii = (gas_radius, gas_radius, liquid_radius, liquid_radius)
    flux_vector = tuple(
        central[index]
        - 0.5 * radii[index] * (right_vector[index] - left_vector[index])
        for index in range(4)
    )
    return BlockRusanovResult(
        flux=InclinedTwoFluidFlux(*flux_vector),
        diagnostics=BlockRusanovDiagnostics(
            method=NUMERICAL_FLUX,
            gas_spectral_radius=gas_radius,
            liquid_spectral_radius=liquid_radius,
            lambda_left=left_primitive.lambda_d,
            lambda_right=right_primitive.lambda_d,
            lambda_face=lambda_face,
            fallback_available=RIEMANN_FALLBACK_AVAILABLE,
            fallback_used=False,
        ),
    )


def cell_source(
    left: InclinedTwoFluidState,
    center: InclinedTwoFluidState,
    right: InclinedTwoFluidState,
    cell_width: float,
    params: InclinedTwoFluidParameters,
    *,
    extra: MomentumClosureSource = MomentumClosureSource(),
) -> InclinedTwoFluidSource:
    """Discretize Eq. (2) and the gravity part of Eqs. (A20)/(A30).

    ``left`` and ``right`` are the neighbouring cell-centre (or ghost-cell)
    states on a uniform mesh.  The two nonconservative gas terms use the same
    centered derivative.  With friction omitted, Eq. (A20) gives

    ``S_g = -rho_g A_g g sin(theta)`` and
    ``S_d = -(1-rho_g/rho_l) A_l g sin(theta)``.
    """

    if not math.isfinite(cell_width) or cell_width <= 0.0:
        raise ValueError("cell width must be finite and positive")
    primitive_left = primitive_state(left, params)
    primitive_center = primitive_state(center, params)
    primitive_right = primitive_state(right, params)
    gas_area_gradient = (
        primitive_right.gas_area - primitive_left.gas_area
    ) / (2.0 * cell_width)
    liquid_depth_gradient = (
        primitive_right.liquid_depth - primitive_left.liquid_depth
    ) / (2.0 * cell_width)
    gas_regular_gravity = -center.gas_mass * params.gravity * params.sine
    gas_geometric = (
        primitive_center.gas_pressure_absolute * gas_area_gradient
        - primitive_center.gas_area
        * primitive_center.gas_density
        * params.gravity
        * params.cosine
        * liquid_depth_gradient
    )
    liquid_decoupled_gravity = -(
        1.0 - primitive_center.gas_density / params.liquid_density
    ) * center.liquid_area * params.gravity * params.sine
    return InclinedTwoFluidSource(
        gas_mass=0.0,
        gas_momentum=(
            gas_geometric + gas_regular_gravity + extra.gas_momentum
        ),
        liquid_area=0.0,
        liquid_momentum=(
            liquid_decoupled_gravity + extra.liquid_momentum
        ),
    )


def _inventory(
    states: Iterable[InclinedTwoFluidState],
    cell_width: float,
) -> InclinedBranchInventory:
    totals = [0.0, 0.0, 0.0, 0.0]
    for state in states:
        for index, value in enumerate(state.vector()):
            totals[index] += cell_width * value
    return InclinedBranchInventory(*totals)


def _subtract_inventory(
    left: InclinedBranchInventory,
    right: InclinedBranchInventory,
) -> InclinedBranchInventory:
    return InclinedBranchInventory(
        *(a - b for a, b in zip(left.vector(), right.vector()))
    )


def _add_scaled_flux_and_source(
    left_flux: InclinedTwoFluidFlux,
    right_flux: InclinedTwoFluidFlux,
    integrated_source: InclinedBranchInventory,
    dt: float,
) -> InclinedBranchInventory:
    return InclinedBranchInventory(
        *(
            dt * (fl - fr + source)
            for fl, fr, source in zip(
                left_flux.vector(),
                right_flux.vector(),
                integrated_source.vector(),
            )
        )
    )


def euler_inclined_branch_stage(
    states: tuple[InclinedTwoFluidState, ...],
    dt: float,
    *,
    cell_width: float,
    params: InclinedTwoFluidParameters,
    boundaries: InclinedBranchBoundaryStates,
    extra_sources: tuple[MomentumClosureSource, ...] | None = None,
) -> InclinedBranchEulerResult:
    """Advance one uniform-mesh Euler stage with fail-fast admissibility.

    The input tuple is immutable.  Every face flux, source, candidate state,
    and candidate ``Lambda_d`` is evaluated before the result is returned.  A
    failed check raises and exposes no partially advanced branch.
    """

    if not states:
        raise ValueError("at least one branch cell is required")
    if not _finite(dt, cell_width) or dt < 0.0 or cell_width <= 0.0:
        raise ValueError("dt must be non-negative and cell width positive")
    if extra_sources is None:
        extra_sources = tuple(MomentumClosureSource() for _ in states)
    if len(extra_sources) != len(states):
        raise ValueError("one extra source is required per branch cell")

    all_states = (*states, boundaries.left, boundaries.right)
    primitives = tuple(primitive_state(state, params) for state in all_states)
    maximum_signal_speed = max(
        max(
            abs(primitive.gas_velocity) + params.gas_sound_speed,
            abs(primitive.liquid_velocity) + primitive.liquid_celerity,
        )
        for primitive in primitives
    )
    cfl = dt * maximum_signal_speed / cell_width
    if cfl > params.maximum_cfl:
        raise CFLViolationError(
            f"explicit stage CFL={cfl:.12g} exceeds {params.maximum_cfl:.12g}"
        )

    chain = (boundaries.left, *states, boundaries.right)
    face_results = tuple(
        block_rusanov_flux(left, right, params)
        for left, right in zip(chain[:-1], chain[1:])
    )
    face_fluxes = tuple(result.flux for result in face_results)
    sources = tuple(
        cell_source(
            states[index - 1] if index > 0 else boundaries.left,
            state,
            states[index + 1] if index + 1 < len(states) else boundaries.right,
            cell_width,
            params,
            extra=extra_sources[index],
        )
        for index, state in enumerate(states)
    )

    candidates: list[InclinedTwoFluidState] = []
    for index, (state, source) in enumerate(zip(states, sources)):
        left_flux = face_fluxes[index].vector()
        right_flux = face_fluxes[index + 1].vector()
        candidate_vector = tuple(
            old
            + dt
            * (
                (left_flux[component] - right_flux[component]) / cell_width
                + source.vector()[component]
            )
            for component, old in enumerate(state.vector())
        )
        try:
            candidate = InclinedTwoFluidState(*candidate_vector)
            primitive_state(candidate, params)
        except InclinedTwoFluidError as exc:
            raise StageAdmissibilityError(
                f"inclined branch cell {index} rejected; reduce dt or revise "
                "the physical closure"
            ) from exc
        candidates.append(candidate)

    initial = _inventory(states, cell_width)
    final = _inventory(candidates, cell_width)
    integrated_source = InclinedBranchInventory(
        *(
            sum(cell_width * source.vector()[component] for source in sources)
            for component in range(4)
        )
    )
    expected = _add_scaled_flux_and_source(
        face_fluxes[0], face_fluxes[-1], integrated_source, dt
    )
    actual = _subtract_inventory(final, initial)
    residual = _subtract_inventory(actual, expected)
    ledger = InclinedBranchStageLedger(
        dt=dt,
        cell_width=cell_width,
        initial=initial,
        final=final,
        expected_change=expected,
        residual=residual,
        left_boundary_flux=face_fluxes[0],
        right_boundary_flux=face_fluxes[-1],
        integrated_source=integrated_source,
    )
    return InclinedBranchEulerResult(
        states=tuple(candidates),
        face_fluxes=face_fluxes,
        face_diagnostics=tuple(result.diagnostics for result in face_results),
        cell_sources=sources,
        maximum_signal_speed=maximum_signal_speed,
        cfl=cfl,
        ledger=ledger,
    )


__all__ = [
    "BlockRusanovDiagnostics",
    "BlockRusanovResult",
    "CFLViolationError",
    "CircularSegmentState",
    "InclinedBranchBoundaryStates",
    "InclinedBranchEulerResult",
    "InclinedBranchInventory",
    "InclinedBranchStageLedger",
    "InclinedTwoFluidError",
    "InclinedTwoFluidFlux",
    "InclinedTwoFluidParameters",
    "InclinedTwoFluidPrimitive",
    "InclinedTwoFluidSource",
    "InclinedTwoFluidState",
    "INTERIOR_BRANCH_CORE_READY",
    "LossOfHyperbolicityError",
    "MISSING_RISER_CLOSURES",
    "MomentumClosureSource",
    "NUMERICAL_FLUX",
    "COMPLETE_RISER_MODEL_READY",
    "RIEMANN_FALLBACK_AVAILABLE",
    "StageAdmissibilityError",
    "StateAdmissibilityError",
    "block_rusanov_flux",
    "cell_source",
    "circular_segment_from_area",
    "euler_inclined_branch_stage",
    "physical_flux",
    "primitive_state",
]
