"""One conservative finite-volume Euler stage for the Case-A horizontal branch.

The branch coordinate points away from the side T.  Cells strictly to the
left of the tracked material front store the resolved stratified variables

``(m_g, j_g, A_l, Q_l)``,

cells strictly to its right store the pressurised variables ``(A, Q)``, and
the single host cell is the ALE cut cell from
``casea_material_front_cutcell``.  The host's two outer-face fluxes are the
same Python objects used to update its neighbouring regular cells; no duplicate
or one-sided junction update exists in this module.

This file deliberately implements *one forward-Euler stage*.  A main time
integrator may call it at each SSP--RK stage, but this routine does not claim
to be SSP--RK2.  If the fitted front reaches a mesh face inside ``dt``, the
whole stage is rejected with :class:`MixedBranchCrossingRequired`.  The caller
must advance every branch to the reported event time, remap ownership, and
then restart the remaining stage.  Consequently no adjacent regular cell is
advanced past an unprocessed topology event.

There are no area/speed/result clips, dry fills, prescribed bubbles, target
waves, frozen cells, or empirical branch splits here.  A non-admissible Euler
state is rejected and the caller must reduce the time step.

This operator is intentionally **horizontal-only**.  Its stratified liquid
closure uses the horizontal circular-segment hydrostatic/area law.  A vertical
``bed_slope=1`` branch requires its own vertical two-fluid gravity and film
closure and is rejected here rather than being approximated by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_coupled_gas_network import isothermal_ideal_gas_riemann_flux
from casea_horizontal_liquid_operator import (
    HorizontalLiquidParameters,
    PressurePotentialState,
    decoupled_lambda_and_derivative,
    physical_liquid_flux,
    pressure_potential_state,
    rusanov_face_flux,
)
from casea_material_front_cutcell import (
    ALEInterfaceFlux,
    AdvanceResult,
    InterfaceTraces,
    MaterialFrontCutCell,
    OuterFaceFluxes,
    PressurisedFlux,
    PressurisedState,
    StratifiedFlux,
    StratifiedState,
    SubcellSources,
    advance_material_front_cutcell,
)
from casea_material_front_rh_adapter import (
    build_casea_material_front_traces,
    build_casea_material_front_traces_at_pressure,
)
from casea_tjunction_shock_network import BranchGeometry


class MixedBranchError(RuntimeError):
    """Base class for rejected mixed-branch Euler stages."""


class MixedBranchAdmissibilityError(MixedBranchError):
    """The explicit stage produced a non-physical regular-cell state."""


class MixedBranchBoundActivationError(MixedBranchAdmissibilityError):
    """A lower-level constitutive bound would alter a production state.

    The shared horizontal-liquid operator retains protective piecewise bounds
    for older exploratory drivers.  They are not admissible as silent state
    modifications in this fitted-front production path.  ``audit`` records
    the raw state and every bound that would have been active.
    """

    def __init__(self, audit: "StratifiedBoundAudit") -> None:
        self.audit = audit
        details = ", ".join(audit.active_bounds)
        super().__init__(
            f"{audit.state_label}: lower-level bound activation rejected "
            f"({details}); A_l={audit.liquid_area:.12g}, "
            f"A_cap={audit.geometry_area_cap:.12g}, "
            f"A_g={audit.raw_gas_area:.12g}, "
            f"A_g,floor={audit.void_area_floor:.12g}, "
            f"rho_g={audit.raw_gas_density:.12g}, "
            f"rho_range=[{audit.gas_density_floor:.12g}, "
            f"{audit.gas_density_ceiling:.12g}]"
        )


class MixedBranchGasFallbackError(MixedBranchError):
    """The gas face attempted to leave the required Roe production path."""


class MixedBranchScopeError(MixedBranchError):
    """A non-horizontal branch was passed to the horizontal-only operator."""


class MixedBranchCrossingRequired(MixedBranchError):
    """A material-front face crossing must be processed before this stage.

    ``crossing_time`` is measured from the beginning of the rejected Euler
    stage.  No state in the supplied branch has been mutated or advanced.
    """

    def __init__(
        self,
        *,
        crossing_time: float,
        face_position: float,
        moving_direction: int,
        old_host_index: int,
        new_host_index: int,
    ) -> None:
        self.crossing_time = float(crossing_time)
        self.face_position = float(face_position)
        self.moving_direction = int(moving_direction)
        self.old_host_index = int(old_host_index)
        self.new_host_index = int(new_host_index)
        super().__init__(
            "material front crosses a branch face inside the requested Euler "
            f"stage: dt_event={self.crossing_time:.12g}, "
            f"face={self.face_position:.12g}, direction={self.moving_direction:+d}"
        )


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class StratifiedBoundAudit:
    """Raw-state audit performed before the shared liquid closure is called."""

    state_label: str
    liquid_area: float
    geometry_area_cap: float
    raw_gas_area: float
    void_area_floor: float
    raw_gas_density: float
    gas_density_floor: float
    gas_density_ceiling: float
    roe_internal_density_floor: float
    geometry_cap_active: bool
    void_floor_active: bool
    gas_density_floor_active: bool
    gas_density_ceiling_active: bool
    roe_density_floor_active: bool

    @property
    def active_bounds(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.geometry_cap_active:
            names.append("geometry_cap")
        if self.void_floor_active:
            names.append("void_floor")
        if self.gas_density_floor_active:
            names.append("gas_density_floor")
        if self.gas_density_ceiling_active:
            names.append("gas_density_ceiling")
        if self.roe_density_floor_active:
            names.append("roe_internal_density_floor")
        return tuple(names)

    @property
    def accepted_without_bound(self) -> bool:
        return not self.active_bounds


@dataclass(frozen=True)
class LiquidPotentialDiagnostics:
    """Audit of the published Eq. (40) numerical celerity treatment.

    ``eq40_floor_term_added`` is true because the published numerical tangent
    always contains ``c_eps**2``.  ``eq40_nonpositive_tangent_branch_active``
    distinguishes the material case in which ``max(c_phys**2, 0)`` replaces a
    non-positive physical tangent; this is the activation that must remain
    visible to acceptance checks.
    """

    bound_audit: StratifiedBoundAudit
    physical_celerity_squared: float
    numerical_celerity_squared: float
    numerical_celerity: float
    eq40_floor_term: float
    eq40_floor_term_added: bool
    eq40_nonpositive_tangent_branch_active: bool


@dataclass(frozen=True)
class GasRiemannDiagnostics:
    """Solver provenance for one resolved-gas face.

    The called public gas interface is the positive-density Roe-only API; it
    has no Einfeldt/HLL fallback selector.  Densities that would activate its
    internal numerical floor are rejected by the preceding bound audit.
    """

    solver: str
    roe_used: bool
    fallback_used: bool
    fallback_name: str | None
    density_left: float
    density_right: float
    roe_internal_density_floor: float
    roe_density_floor_active: bool


@dataclass(frozen=True)
class StratifiedFaceNumericsDiagnostics:
    gas: GasRiemannDiagnostics
    liquid_left: LiquidPotentialDiagnostics
    liquid_right: LiquidPotentialDiagnostics


@dataclass(frozen=True)
class StratifiedNumericalFluxResult:
    flux: StratifiedFlux
    diagnostics: StratifiedFaceNumericsDiagnostics


@dataclass(frozen=True)
class MixedBranchNumericsDiagnostics:
    """Per-stage record of every ordinary stratified numerical face."""

    stratified_faces: tuple[StratifiedFaceNumericsDiagnostics, ...]

    @property
    def roe_face_count(self) -> int:
        return sum(int(face.gas.roe_used) for face in self.stratified_faces)

    @property
    def fallback_face_count(self) -> int:
        return sum(int(face.gas.fallback_used) for face in self.stratified_faces)

    @property
    def eq40_nonpositive_state_count(self) -> int:
        return sum(
            int(potential.eq40_nonpositive_tangent_branch_active)
            for face in self.stratified_faces
            for potential in (face.liquid_left, face.liquid_right)
        )


@dataclass(frozen=True)
class PressurisedSource:
    """Distributed pressurised source ``(S_A, S_Q)`` per unit length."""

    area: float = 0.0
    momentum: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.area, self.momentum):
            raise ValueError("pressurised source must be finite")

    def vector(self) -> tuple[float, float]:
        return (float(self.area), float(self.momentum))


@dataclass(frozen=True)
class StratifiedSource:
    """Distributed stratified source per unit branch length."""

    gas_mass: float = 0.0
    gas_momentum: float = 0.0
    liquid_area: float = 0.0
    liquid_momentum: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(*self.vector()):
            raise ValueError("stratified source must be finite")

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_momentum),
        )


@dataclass(frozen=True)
class MixedBranchSources:
    """Sources for every ordinary cell and both host subcells."""

    stratified: tuple[StratifiedSource, ...]
    pressurised: tuple[PressurisedSource, ...]
    host: SubcellSources = SubcellSources()

    @classmethod
    def zeros(cls, state: "MixedBranchState") -> "MixedBranchSources":
        return cls(
            stratified=tuple(
                StratifiedSource() for _ in state.stratified_cells
            ),
            pressurised=tuple(
                PressurisedSource() for _ in state.pressurised_cells
            ),
            host=SubcellSources(),
        )


@dataclass(frozen=True)
class MixedBranchParameters:
    """Physical data for the horizontal east branch and RH front adapter."""

    geometry: BranchGeometry
    atmospheric_pressure: float = 101_325.0
    liquid_density: float = 998.0
    gravity: float = 9.81
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    gas_entropy_fix_fraction: float = 0.10
    pressurised_friction_slope: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.atmospheric_pressure,
            self.liquid_density,
            self.gravity,
            self.gas_constant,
            self.gas_temperature,
            self.gas_entropy_fix_fraction,
            self.pressurised_friction_slope,
        )
        if not _finite(*values):
            raise ValueError("mixed-branch parameters must be finite")
        if min(
            self.atmospheric_pressure,
            self.liquid_density,
            self.gravity,
            self.gas_constant,
            self.gas_temperature,
        ) <= 0.0:
            raise ValueError("mixed-branch physical scales must be positive")
        if self.gas_entropy_fix_fraction < 0.0:
            raise ValueError("gas entropy fix must be non-negative")
        if self.geometry.bed_slope != 0.0:
            raise MixedBranchScopeError(
                "casea_mixed_branch_fv is horizontal-only (bed_slope=0); "
                "the vertical branch needs a separate vertical two-fluid "
                "gravity/film closure"
            )

    @property
    def gas_sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.gas_temperature)

    @property
    def horizontal_liquid(self) -> HorizontalLiquidParameters:
        """Published Case-A *horizontal* closure in per-unit-length variables.

        ``StratifiedState.gas_mass`` is gas mass per unit pipe length, whereas
        the lower-level operator accepts cell-integrated gas mass and divides
        by ``cell_width``.  Supplying a unit reference width makes those two
        representations identical without a mesh-dependent rescaling.
        """

        section = self.geometry.section(self.gravity)
        return HorizontalLiquidParameters(
            area_full=section.full_area,
            diameter=self.geometry.diameter,
            wave_speed=self.geometry.wave_speed,
            cell_width=1.0,
            gravity=self.gravity,
            rho_liquid=self.liquid_density,
            gas_constant=self.gas_constant,
            gas_temperature=self.gas_temperature,
            atmospheric_pressure=self.atmospheric_pressure,
        )


@dataclass(frozen=True)
class MixedBranchBoundaryFluxes:
    """Physical ``+x`` fluxes at the two ends of the branch."""

    left_stratified: StratifiedFlux
    right_pressurised: PressurisedFlux


@dataclass(frozen=True)
class MixedBranchInventory:
    """Branch-integrated conserved inventory.

    ``liquid_discharge`` is the integral of the discharge state and is the
    conserved liquid momentum variable used by the one-dimensional equations.
    """

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
class MixedBranchState:
    """Contiguous regular-cell state around one ``S | P`` ALE host."""

    host: MaterialFrontCutCell
    stratified_cells: tuple[StratifiedState, ...]
    pressurised_cells: tuple[PressurisedState, ...]

    def __post_init__(self) -> None:
        if self.host.pressurised_side != "right":
            raise ValueError("Case-A mixed branch requires pressurised_side='right'")
        n_cells = len(self.host.cell_faces) - 1
        if len(self.stratified_cells) != self.host.host_index:
            raise ValueError(
                "stratified_cells must own exactly indices [0, host_index)"
            )
        if len(self.pressurised_cells) != n_cells - self.host.host_index - 1:
            raise ValueError(
                "pressurised_cells must own exactly indices (host_index, n_cells)"
            )

    @property
    def cell_faces(self) -> tuple[float, ...]:
        return self.host.cell_faces

    @property
    def n_cells(self) -> int:
        return len(self.cell_faces) - 1

    def inventory(self) -> MixedBranchInventory:
        gas_mass = 0.0
        gas_momentum = 0.0
        liquid_volume = 0.0
        liquid_discharge = 0.0
        faces = self.cell_faces
        for index, state in enumerate(self.stratified_cells):
            width = faces[index + 1] - faces[index]
            gas_mass += width * state.gas_mass
            gas_momentum += width * state.gas_momentum
            liquid_volume += width * state.liquid_area
            liquid_discharge += width * state.liquid_discharge
        host_inventory = self.host.inventory()
        gas_mass += host_inventory.gas_mass
        gas_momentum += host_inventory.gas_momentum
        liquid_volume += host_inventory.liquid_area
        liquid_discharge += host_inventory.liquid_discharge
        start = self.host.host_index + 1
        for offset, state in enumerate(self.pressurised_cells):
            index = start + offset
            width = faces[index + 1] - faces[index]
            liquid_volume += width * state.area
            liquid_discharge += width * state.discharge
        return MixedBranchInventory(
            gas_mass=gas_mass,
            gas_momentum=gas_momentum,
            liquid_volume=liquid_volume,
            liquid_discharge=liquid_discharge,
        )


@dataclass(frozen=True)
class MixedBranchStageLedger:
    """Auditable componentwise balance for one accepted Euler stage."""

    dt: float
    initial: MixedBranchInventory
    final: MixedBranchInventory
    expected_change: MixedBranchInventory
    residual: MixedBranchInventory
    boundary_fluxes: MixedBranchBoundaryFluxes
    interface_flux: ALEInterfaceFlux
    integrated_source: MixedBranchInventory


@dataclass(frozen=True)
class MixedBranchEulerResult:
    state: MixedBranchState
    interface_traces: InterfaceTraces
    stratified_face_fluxes: tuple[StratifiedFlux, ...]
    pressurised_face_fluxes: tuple[PressurisedFlux, ...]
    host_outer_fluxes: OuterFaceFluxes
    host_advance: AdvanceResult
    ledger: MixedBranchStageLedger
    numerics: MixedBranchNumericsDiagnostics


def _gas_area(state: StratifiedState, params: MixedBranchParameters) -> float:
    area = params.geometry.section(params.gravity).full_area - state.liquid_area
    if area <= 0.0:
        raise ValueError("stratified state must leave a positive gas area")
    return float(area)


_ROE_INTERNAL_DENSITY_FLOOR = 1.0e-10


def audit_stratified_state_bounds(
    state: StratifiedState,
    params: MixedBranchParameters,
    *,
    state_label: str = "stratified state",
) -> StratifiedBoundAudit:
    """Reject every silent bound in the reused stratified pressure closure.

    The checks deliberately reproduce the *raw* predicates in
    :func:`pressure_potential_state` before calling it.  Equality is treated as
    active where the lower-level derivative switches branch at equality.  No
    value is modified here or in the accepted production path.
    """

    if not isinstance(state_label, str) or not state_label:
        raise ValueError("state_label must be a non-empty string")
    liquid = params.horizontal_liquid
    area_full = liquid.area_full
    liquid_area = float(state.liquid_area)
    area_cap = liquid.geometry_cap_fraction * area_full
    raw_gas_area = area_full - liquid_area
    void_floor = liquid.void_floor_fraction * area_full
    if raw_gas_area > 0.0:
        raw_density = float(
            state.gas_mass / (raw_gas_area * liquid.cell_width)
        )
    else:
        raw_density = math.inf
    density_floor = (
        liquid.gas_density_floor_fraction
        * liquid.atmospheric_gas_density
    )
    density_ceiling = (
        liquid.gas_density_ceiling_fraction
        * liquid.atmospheric_gas_density
    )
    audit = StratifiedBoundAudit(
        state_label=state_label,
        liquid_area=liquid_area,
        geometry_area_cap=float(area_cap),
        raw_gas_area=float(raw_gas_area),
        void_area_floor=float(void_floor),
        raw_gas_density=float(raw_density),
        gas_density_floor=float(density_floor),
        gas_density_ceiling=float(density_ceiling),
        roe_internal_density_floor=_ROE_INTERNAL_DENSITY_FLOOR,
        geometry_cap_active=bool(liquid_area >= area_cap),
        void_floor_active=bool(raw_gas_area <= void_floor),
        gas_density_floor_active=bool(raw_density <= density_floor),
        gas_density_ceiling_active=bool(raw_density >= density_ceiling),
        roe_density_floor_active=bool(
            raw_density < _ROE_INTERNAL_DENSITY_FLOOR
        ),
    )
    if not audit.accepted_without_bound:
        raise MixedBranchBoundActivationError(audit)
    return audit


def _evaluate_stratified_liquid_potential(
    state: StratifiedState,
    params: MixedBranchParameters,
    *,
    liquid_potential_offset: float,
    state_label: str,
) -> tuple[PressurePotentialState, LiquidPotentialDiagnostics]:
    """Evaluate one accepted state and expose the Eq. (40) branch choice."""

    audit = audit_stratified_state_bounds(
        state,
        params,
        state_label=state_label,
    )
    pressure = pressure_potential_state(
        state.liquid_area,
        state.liquid_discharge,
        state.gas_mass,
        state.gas_momentum,
        True,
        params.horizontal_liquid,
        stratified_potential_offset=liquid_potential_offset,
    )
    physical_tangent = float(pressure.lambda_value) * state.liquid_area
    numerical_tangent = float(pressure.derivative)
    floor_term = params.horizontal_liquid.numerical_celerity_floor**2
    expected = max(physical_tangent, 0.0) + floor_term
    if not math.isclose(
        numerical_tangent,
        expected,
        rel_tol=2.0e-13,
        abs_tol=2.0e-15,
    ):
        raise AssertionError(
            "shared liquid operator no longer matches the audited Eq. (40) law"
        )
    diagnostics = LiquidPotentialDiagnostics(
        bound_audit=audit,
        physical_celerity_squared=float(physical_tangent),
        numerical_celerity_squared=float(numerical_tangent),
        numerical_celerity=float(pressure.celerity),
        eq40_floor_term=float(floor_term),
        eq40_floor_term_added=True,
        eq40_nonpositive_tangent_branch_active=bool(
            physical_tangent <= 0.0
        ),
    )
    return pressure, diagnostics


def _audited_roe_gas_flux(
    density_left: float,
    velocity_left: float,
    density_right: float,
    velocity_right: float,
    params: MixedBranchParameters,
) -> tuple[float, float, GasRiemannDiagnostics]:
    """Call the Roe-only public API after rejecting its internal density floor."""

    density_floor_active = (
        density_left < _ROE_INTERNAL_DENSITY_FLOOR
        or density_right < _ROE_INTERNAL_DENSITY_FLOOR
    )
    if density_floor_active:
        raise MixedBranchAdmissibilityError(
            "resolved-gas Roe density floor would activate; production mixed "
            "branch rejects the face instead of modifying either state"
        )
    gas_mass, gas_momentum = isothermal_ideal_gas_riemann_flux(
        density_left,
        velocity_left,
        density_right,
        velocity_right,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
        entropy_fix_fraction=params.gas_entropy_fix_fraction,
    )
    diagnostics = GasRiemannDiagnostics(
        solver="positive-density Roe",
        roe_used=True,
        fallback_used=False,
        fallback_name=None,
        density_left=float(density_left),
        density_right=float(density_right),
        roe_internal_density_floor=_ROE_INTERNAL_DENSITY_FLOOR,
        roe_density_floor_active=False,
    )
    return float(gas_mass), float(gas_momentum), diagnostics


def _require_roe_only_gas_flux(diagnostics: GasRiemannDiagnostics) -> None:
    """Fail closed if a future adapter introduces an HLL/Einfeldt fallback."""

    if diagnostics.fallback_used or not diagnostics.roe_used:
        fallback = diagnostics.fallback_name or diagnostics.solver
        raise MixedBranchGasFallbackError(
            "production mixed branch requires the resolved positive-density "
            f"Roe flux; fallback '{fallback}' was rejected"
        )


def stratified_physical_flux(
    state: StratifiedState,
    params: MixedBranchParameters,
) -> StratifiedFlux:
    """Return the physical resolved two-fluid flux of one stratified state."""

    section = params.geometry.section(params.gravity)
    gas_area = _gas_area(state, params)
    gas_density = state.gas_mass / gas_area
    gas_velocity = state.gas_velocity
    gas_pressure_abs = gas_density * params.gas_constant * params.gas_temperature
    liquid_depth = float(section.depth_from_area(state.liquid_area))
    liquid_velocity = state.liquid_discharge / state.liquid_area
    return StratifiedFlux(
        gas_mass=state.gas_momentum,
        gas_momentum=(
            state.gas_momentum * gas_velocity
            + (gas_pressure_abs - params.atmospheric_pressure) * gas_area
        ),
        liquid_area=state.liquid_discharge,
        liquid_momentum=(
            state.liquid_discharge * liquid_velocity
            + params.gravity * float(section.hydrostatic_moment(liquid_depth))
            + (gas_pressure_abs - params.atmospheric_pressure)
            * section.full_area
            / params.liquid_density
        ),
    )


def stratified_liquid_potential_offset(
    reference: StratifiedState,
    params: MixedBranchParameters,
) -> float:
    """Fix the single pressure-potential gauge of one connected gas pocket.

    The reduced liquid operator determines only the derivative of its
    stratified pressure potential.  The additive constant is fixed once, at
    the fitted material-front foot, by the resolved hydrostatic plus gas
    traction.  Every S face in the connected branch then uses this same value;
    no cellwise pressure zero is introduced.
    """

    audit_stratified_state_bounds(
        reference,
        params,
        state_label="stratified potential reference",
    )
    liquid = params.horizontal_liquid
    lam, _ = decoupled_lambda_and_derivative(
        reference.liquid_area,
        reference.liquid_discharge,
        reference.gas_mass,
        reference.gas_momentum,
        liquid,
    )
    physical = stratified_physical_flux(reference, params)
    convective = reference.liquid_discharge**2 / reference.liquid_area
    return float(
        physical.liquid_momentum
        - convective
        - 0.5 * float(lam) * reference.liquid_area**2
    )


def stratified_numerical_flux(
    left: StratifiedState,
    right: StratifiedState,
    params: MixedBranchParameters,
    *,
    liquid_potential_offset: float,
) -> StratifiedFlux:
    """First-order Roe-gas/Rusanov-liquid flux on one shared S face.

    The existing Case-A positive-density isothermal Roe solver is evaluated on
    the arithmetic geometric face opening.  Its absolute-pressure momentum
    flux is shifted by the spatially constant atmospheric pressure so it uses
    the same gauge as the RH/ALE adapter.  This pressure-gauge change has no
    effect on the Roe waves or mass flux.
    """

    return stratified_numerical_flux_with_diagnostics(
        left,
        right,
        params,
        liquid_potential_offset=liquid_potential_offset,
    ).flux


def stratified_numerical_flux_with_diagnostics(
    left: StratifiedState,
    right: StratifiedState,
    params: MixedBranchParameters,
    *,
    liquid_potential_offset: float,
    left_label: str = "left stratified state",
    right_label: str = "right stratified state",
) -> StratifiedNumericalFluxResult:
    """Return the shared S-face flux plus an explicit numerical audit.

    This is the production implementation used by the Euler stage.  All
    lower-level cap/floor predicates are checked before either pressure state
    is evaluated.  The gas call is restricted to the public Roe-only API, so
    an Einfeldt/HLL fallback is neither selected nor silently available.
    """

    if not math.isfinite(liquid_potential_offset):
        raise ValueError("liquid pressure-potential offset must be finite")
    gas_area_left = _gas_area(left, params)
    gas_area_right = _gas_area(right, params)
    face_gas_area = 0.5 * (gas_area_left + gas_area_right)
    density_left = left.gas_mass / gas_area_left
    density_right = right.gas_mass / gas_area_right
    pressure_left, liquid_left_diagnostics = (
        _evaluate_stratified_liquid_potential(
            left,
            params,
            liquid_potential_offset=liquid_potential_offset,
            state_label=left_label,
        )
    )
    pressure_right, liquid_right_diagnostics = (
        _evaluate_stratified_liquid_potential(
            right,
            params,
            liquid_potential_offset=liquid_potential_offset,
            state_label=right_label,
        )
    )
    (
        gas_mass_per_area,
        gas_momentum_per_area,
        gas_diagnostics,
    ) = _audited_roe_gas_flux(
        density_left,
        left.gas_velocity,
        density_right,
        right.gas_velocity,
        params,
    )
    _require_roe_only_gas_flux(gas_diagnostics)
    liquid_flux, _ = rusanov_face_flux(
        left.liquid_area,
        left.liquid_discharge,
        pressure_left,
        right.liquid_area,
        right.liquid_discharge,
        pressure_right,
    )
    return StratifiedNumericalFluxResult(
        flux=StratifiedFlux(
            gas_mass=gas_mass_per_area * face_gas_area,
            gas_momentum=(
                gas_momentum_per_area - params.atmospheric_pressure
            )
            * face_gas_area,
            liquid_area=float(liquid_flux[..., 0]),
            liquid_momentum=float(liquid_flux[..., 1]),
        ),
        diagnostics=StratifiedFaceNumericsDiagnostics(
            gas=gas_diagnostics,
            liquid_left=liquid_left_diagnostics,
            liquid_right=liquid_right_diagnostics,
        ),
    )


def pressurised_physical_flux(
    state: PressurisedState,
    params: MixedBranchParameters,
) -> PressurisedFlux:
    """Return the conservative elastic/full-pipe liquid flux."""

    pressure = pressure_potential_state(
        state.area,
        state.discharge,
        0.0,
        0.0,
        False,
        params.horizontal_liquid,
    )
    flux = physical_liquid_flux(state.area, state.discharge, pressure)
    return PressurisedFlux(
        area=float(flux[..., 0]),
        momentum=float(flux[..., 1]),
    )


def pressurised_numerical_flux(
    left: PressurisedState,
    right: PressurisedState,
    params: MixedBranchParameters,
) -> PressurisedFlux:
    """Rusanov flux using the same circular/elastic pressure law."""

    liquid = params.horizontal_liquid
    pressure_left = pressure_potential_state(
        left.area, left.discharge, 0.0, 0.0, False, liquid
    )
    pressure_right = pressure_potential_state(
        right.area, right.discharge, 0.0, 0.0, False, liquid
    )
    flux, _ = rusanov_face_flux(
        left.area,
        left.discharge,
        pressure_left,
        right.area,
        right.discharge,
        pressure_right,
    )
    return PressurisedFlux(
        area=float(flux[..., 0]),
        momentum=float(flux[..., 1]),
    )


def build_mixed_branch_interface_traces(
    state: MixedBranchState,
    params: MixedBranchParameters,
    *,
    shared_gas_pressure_abs: float | None = None,
) -> InterfaceTraces:
    """Evaluate the existing Case-A RH adapter on the current host state."""

    host = state.host
    if shared_gas_pressure_abs is None:
        return build_casea_material_front_traces(
            host.pressurised,
            host.stratified,
            front_position=host.front_position,
            geometry=params.geometry,
            atmospheric_pressure=params.atmospheric_pressure,
            liquid_density=params.liquid_density,
            gravity=params.gravity,
            gas_sound_speed=params.gas_sound_speed,
            dt=0.0,
            pressurised_friction_slope=params.pressurised_friction_slope,
        ).traces
    if not math.isfinite(shared_gas_pressure_abs) or shared_gas_pressure_abs <= 0.0:
        raise ValueError("shared gas pressure must be finite and positive")
    return build_casea_material_front_traces_at_pressure(
        host.pressurised,
        stratified_liquid_area=host.stratified.liquid_area,
        free_surface_velocity=(
            host.stratified.liquid_discharge / host.stratified.liquid_area
        ),
        gas_pressure_abs=shared_gas_pressure_abs,
        front_position=host.front_position,
        geometry=params.geometry,
        atmospheric_pressure=params.atmospheric_pressure,
        liquid_density=params.liquid_density,
        gravity=params.gravity,
        gas_sound_speed=params.gas_sound_speed,
        dt=0.0,
        pressurised_friction_slope=params.pressurised_friction_slope,
    ).traces


def _crossing_within_stage(
    host: MaterialFrontCutCell,
    speed: float,
    dt: float,
) -> MixedBranchCrossingRequired | None:
    if speed == 0.0 or dt == 0.0:
        return None
    left, right = host.host_faces
    if speed > 0.0:
        face = right
        direction = 1
    else:
        face = left
        direction = -1
    crossing_time = (face - host.front_position) / speed
    if crossing_time < 0.0:
        raise MixedBranchError("front speed points away from its selected face")
    if crossing_time <= dt:
        return MixedBranchCrossingRequired(
            crossing_time=crossing_time,
            face_position=face,
            moving_direction=direction,
            old_host_index=host.host_index,
            new_host_index=host.host_index + direction,
        )
    return None


def _difference(
    final: MixedBranchInventory,
    initial: MixedBranchInventory,
) -> MixedBranchInventory:
    return MixedBranchInventory(
        *(right - left for right, left in zip(final.vector(), initial.vector()))
    )


def _subtract(
    left: MixedBranchInventory,
    right: MixedBranchInventory,
) -> MixedBranchInventory:
    return MixedBranchInventory(
        *(a - b for a, b in zip(left.vector(), right.vector()))
    )


def _integrated_sources(
    state: MixedBranchState,
    sources: MixedBranchSources,
) -> MixedBranchInventory:
    faces = state.cell_faces
    gas_mass = 0.0
    gas_momentum = 0.0
    liquid_volume = 0.0
    liquid_discharge = 0.0
    for index, source in enumerate(sources.stratified):
        width = faces[index + 1] - faces[index]
        gas_mass += width * source.gas_mass
        gas_momentum += width * source.gas_momentum
        liquid_volume += width * source.liquid_area
        liquid_discharge += width * source.liquid_momentum
    start = state.host.host_index + 1
    for offset, source in enumerate(sources.pressurised):
        index = start + offset
        width = faces[index + 1] - faces[index]
        liquid_volume += width * source.area
        liquid_discharge += width * source.momentum
    lp = state.host.pressurised_length
    ls = state.host.stratified_length
    gas_mass += ls * sources.host.stratified_gas_mass
    gas_momentum += ls * sources.host.stratified_gas_momentum
    liquid_volume += (
        lp * sources.host.pressurised_area
        + ls * sources.host.stratified_liquid_area
    )
    liquid_discharge += (
        lp * sources.host.pressurised_momentum
        + ls * sources.host.stratified_liquid_momentum
    )
    return MixedBranchInventory(
        gas_mass,
        gas_momentum,
        liquid_volume,
        liquid_discharge,
    )


def _validate_sources(
    state: MixedBranchState,
    sources: MixedBranchSources,
) -> None:
    if len(sources.stratified) != len(state.stratified_cells):
        raise ValueError("one stratified source is required per ordinary S cell")
    if len(sources.pressurised) != len(state.pressurised_cells):
        raise ValueError("one pressurised source is required per ordinary P cell")


def euler_mixed_branch_stage(
    state: MixedBranchState,
    dt: float,
    *,
    params: MixedBranchParameters,
    boundary_fluxes: MixedBranchBoundaryFluxes,
    sources: MixedBranchSources | None = None,
    time: float = 0.0,
    shared_gas_pressure_abs: float | None = None,
    interface_traces: InterfaceTraces | None = None,
) -> MixedBranchEulerResult:
    """Advance one complete ordinary-cell plus ALE-host Euler stage.

    ``interface_traces`` is intended for a simultaneously solved multi-branch
    T closure.  When omitted, this routine calls the Case-A RH adapter itself;
    ``shared_gas_pressure_abs`` selects its fixed-pressure form.  Supplying
    both a trace and a pressure is ambiguous and therefore rejected.
    """

    if not _finite(dt, time) or dt < 0.0:
        raise ValueError("Euler time and dt must be finite with dt non-negative")
    if interface_traces is not None and shared_gas_pressure_abs is not None:
        raise ValueError("supply either interface traces or shared pressure, not both")
    if sources is None:
        sources = MixedBranchSources.zeros(state)
    _validate_sources(state, sources)

    traces = (
        interface_traces
        if interface_traces is not None
        else build_mixed_branch_interface_traces(
            state,
            params,
            shared_gas_pressure_abs=shared_gas_pressure_abs,
        )
    )
    crossing = _crossing_within_stage(state.host, traces.speed, dt)
    if crossing is not None:
        raise crossing

    # One face table per phase topology.  The last S face and first P face are
    # passed directly to the ALE host and to their adjacent regular cells.
    stratified_face_fluxes: list[StratifiedFlux] = [
        boundary_fluxes.left_stratified
    ]
    stratified_chain = (*state.stratified_cells, state.host.stratified)
    liquid_potential_offset = stratified_liquid_potential_offset(
        state.host.stratified, params
    )
    stratified_numerics: list[StratifiedFaceNumericsDiagnostics] = []
    regular_s_count = len(state.stratified_cells)
    for face_index, (left, right) in enumerate(
        zip(stratified_chain[:-1], stratified_chain[1:])
    ):
        left_label = f"stratified cell {face_index}"
        right_label = (
            f"stratified cell {face_index + 1}"
            if face_index + 1 < regular_s_count
            else "ALE host stratified subcell"
        )
        evaluated = stratified_numerical_flux_with_diagnostics(
            left,
            right,
            params,
            liquid_potential_offset=liquid_potential_offset,
            left_label=left_label,
            right_label=right_label,
        )
        stratified_face_fluxes.append(evaluated.flux)
        stratified_numerics.append(evaluated.diagnostics)

    pressurised_chain = (state.host.pressurised, *state.pressurised_cells)
    pressurised_face_fluxes: list[PressurisedFlux] = []
    for left, right in zip(pressurised_chain[:-1], pressurised_chain[1:]):
        pressurised_face_fluxes.append(
            pressurised_numerical_flux(left, right, params)
        )
    pressurised_face_fluxes.append(boundary_fluxes.right_pressurised)

    host_outer = OuterFaceFluxes(
        pressurised=pressurised_face_fluxes[0],
        stratified=stratified_face_fluxes[-1],
    )
    initial = state.inventory()

    new_stratified: list[StratifiedState] = []
    faces = state.cell_faces
    for index, (old, source) in enumerate(
        zip(state.stratified_cells, sources.stratified)
    ):
        width = faces[index + 1] - faces[index]
        left_flux = stratified_face_fluxes[index].vector()
        right_flux = stratified_face_fluxes[index + 1].vector()
        old_vector = old.vector()
        source_vector = source.vector()
        updated = tuple(
            old_vector[component]
            + dt
            * (
                (left_flux[component] - right_flux[component]) / width
                + source_vector[component]
            )
            for component in range(4)
        )
        try:
            new_stratified.append(StratifiedState(*updated))
        except ValueError as exc:
            raise MixedBranchAdmissibilityError(
                f"non-admissible stratified cell {index}; reduce dt"
            ) from exc

    new_pressurised: list[PressurisedState] = []
    start = state.host.host_index + 1
    for offset, (old, source) in enumerate(
        zip(state.pressurised_cells, sources.pressurised)
    ):
        index = start + offset
        width = faces[index + 1] - faces[index]
        left_flux = pressurised_face_fluxes[offset].vector()
        right_flux = pressurised_face_fluxes[offset + 1].vector()
        old_vector = old.vector()
        source_vector = source.vector()
        updated = tuple(
            old_vector[component]
            + dt
            * (
                (left_flux[component] - right_flux[component]) / width
                + source_vector[component]
            )
            for component in range(2)
        )
        try:
            new_pressurised.append(PressurisedState(*updated))
        except ValueError as exc:
            raise MixedBranchAdmissibilityError(
                f"non-admissible pressurised cell {index}; reduce dt"
            ) from exc

    host_advance = advance_material_front_cutcell(
        state.host,
        dt,
        interface_provider=lambda _host, _time: traces,
        outer_flux_provider=lambda _host, _time: host_outer,
        source_provider=lambda _host, _time: sources.host,
        time=time,
    )
    if host_advance.crossings or len(host_advance.ledgers) != 1:
        raise AssertionError("prechecked no-crossing Euler host changed topology")
    final_state = MixedBranchState(
        host=host_advance.state,
        stratified_cells=tuple(new_stratified),
        pressurised_cells=tuple(new_pressurised),
    )
    final = final_state.inventory()
    interface_flux = host_advance.ledgers[0].interface_flux
    integrated_source = _integrated_sources(state, sources)
    rate = MixedBranchInventory(
        gas_mass=(
            boundary_fluxes.left_stratified.gas_mass
            + integrated_source.gas_mass
        ),
        gas_momentum=(
            boundary_fluxes.left_stratified.gas_momentum
            - interface_flux.gas_momentum
            + integrated_source.gas_momentum
        ),
        liquid_volume=(
            boundary_fluxes.left_stratified.liquid_area
            - boundary_fluxes.right_pressurised.area
            + integrated_source.liquid_volume
        ),
        liquid_discharge=(
            boundary_fluxes.left_stratified.liquid_momentum
            - boundary_fluxes.right_pressurised.momentum
            + integrated_source.liquid_discharge
        ),
    )
    expected = MixedBranchInventory(
        *(dt * value for value in rate.vector())
    )
    residual = _subtract(_difference(final, initial), expected)
    ledger = MixedBranchStageLedger(
        dt=dt,
        initial=initial,
        final=final,
        expected_change=expected,
        residual=residual,
        boundary_fluxes=boundary_fluxes,
        interface_flux=interface_flux,
        integrated_source=integrated_source,
    )
    return MixedBranchEulerResult(
        state=final_state,
        interface_traces=traces,
        stratified_face_fluxes=tuple(stratified_face_fluxes),
        pressurised_face_fluxes=tuple(pressurised_face_fluxes),
        host_outer_fluxes=host_outer,
        host_advance=host_advance,
        ledger=ledger,
        numerics=MixedBranchNumericsDiagnostics(
            stratified_faces=tuple(stratified_numerics)
        ),
    )


__all__ = [
    "MixedBranchAdmissibilityError",
    "MixedBranchBoundActivationError",
    "MixedBranchBoundaryFluxes",
    "MixedBranchCrossingRequired",
    "MixedBranchError",
    "MixedBranchEulerResult",
    "MixedBranchGasFallbackError",
    "MixedBranchInventory",
    "MixedBranchNumericsDiagnostics",
    "MixedBranchParameters",
    "MixedBranchScopeError",
    "MixedBranchSources",
    "MixedBranchStageLedger",
    "MixedBranchState",
    "GasRiemannDiagnostics",
    "LiquidPotentialDiagnostics",
    "PressurisedSource",
    "StratifiedBoundAudit",
    "StratifiedFaceNumericsDiagnostics",
    "StratifiedNumericalFluxResult",
    "StratifiedSource",
    "audit_stratified_state_bounds",
    "build_mixed_branch_interface_traces",
    "euler_mixed_branch_stage",
    "pressurised_numerical_flux",
    "pressurised_physical_flux",
    "stratified_liquid_potential_offset",
    "stratified_numerical_flux",
    "stratified_numerical_flux_with_diagnostics",
    "stratified_physical_flux",
]
