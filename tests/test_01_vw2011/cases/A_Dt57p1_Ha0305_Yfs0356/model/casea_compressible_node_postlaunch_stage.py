"""Resolved three-branch boundary fluxes for the compressible Case-A node.

This module adds the post-launch boundary Riemann layer to
``casea_compressible_finite_node``.  The finite node pressure is obtained from
its gas-mass/liquid-inventory occupancy equation.  At that same pressure:

* every liquid branch evaluates its incoming ``LiquidCharacteristic``;
* every gas branch uses an isothermal Roe flux from the stagnant node
  reservoir state to the resolved stratified trace;
* all rates are expressed in coordinates pointing away from the node; and
* the two conservative node inventories are advanced by the independent
  compressible finite-node Euler core.

The returned ``StratifiedFlux`` objects are complete branch FV boundary
fluxes.  Gas momentum uses the atmospheric gauge of the resolved Case-A
two-fluid equations.  Liquid momentum uses the adjacent cell's conservative
pressure-potential gauge, so a static state does not receive an arbitrary
absolute-pressure impulse.

This remains one Forward-Euler stage.  It neither implements SSP--RK nor
crosses a material-front/topology event.  Exact ``m_g=0`` launch is owned by
the event closure plus the inventory core; this post-launch reservoir flux
requires a finite gas volume and rejects the exact launch state explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_compressible_finite_node import (
    CompressibleFiniteNodeEulerResult,
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    CompressibleNodeBranchRates,
    euler_compressible_finite_node_stage,
    solve_compressible_node_pressure,
)
from casea_coupled_gas_network import isothermal_ideal_gas_riemann_flux
from casea_material_front_cutcell import StratifiedFlux, StratifiedState
from casea_tjunction_shock_network import LiquidCharacteristic


PRODUCTION_READY = True


class CompressiblePostLaunchError(RuntimeError):
    """Base class for a rejected post-launch boundary Euler stage."""


class ExactLaunchRequiresEventClosure(CompressiblePostLaunchError):
    """The post-launch reservoir Riemann problem has no finite gas state."""


class PostLaunchSubsonicTraceError(CompressiblePostLaunchError):
    """A resolved gas trace lies outside the intended subsonic domain."""


class PostLaunchBoundActivationError(CompressiblePostLaunchError):
    """A lower-level geometry/density bound would modify a raw state."""

    def __init__(self, audit: "PostLaunchGasBoundAudit") -> None:
        self.audit = audit
        super().__init__(
            f"{audit.branch_label} node-face state activates "
            f"{', '.join(audit.active_bounds)}; production post-launch "
            "coupling rejects the face before calling Roe"
        )


class PostLaunchCFLInadmissible(CompressiblePostLaunchError):
    """The requested explicit step exceeds the finite-node spectral CFL."""

    def __init__(
        self,
        *,
        gas_cfl: float,
        liquid_cfl: float,
        maximum_cfl: float,
        maximum_dt: float,
    ) -> None:
        self.gas_cfl = float(gas_cfl)
        self.liquid_cfl = float(liquid_cfl)
        self.maximum_cfl = float(maximum_cfl)
        self.maximum_dt = float(maximum_dt)
        super().__init__(
            "compressible finite-node boundary CFL is inadmissible: "
            f"CFL_g={gas_cfl:.12g}, CFL_l={liquid_cfl:.12g}, "
            f"limit={maximum_cfl:.12g}; use dt <= {maximum_dt:.12g}"
        )


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class CompressiblePostLaunchParameters:
    """Node constitutive law and branch-boundary gas constants."""

    node: CompressibleFiniteNodeParameters
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    atmospheric_pressure_abs: float = 101_325.0
    gas_entropy_fix_fraction: float = 0.10
    maximum_cfl: float = 1.0
    geometry_cap_fraction: float = 0.995
    void_floor_fraction: float = 1.0e-4
    gas_density_floor_fraction: float = 0.20
    gas_density_ceiling_fraction: float = 12.0

    def __post_init__(self) -> None:
        values = (
            self.gas_constant,
            self.gas_temperature,
            self.atmospheric_pressure_abs,
            self.gas_entropy_fix_fraction,
            self.maximum_cfl,
            self.geometry_cap_fraction,
            self.void_floor_fraction,
            self.gas_density_floor_fraction,
            self.gas_density_ceiling_fraction,
        )
        if not _finite(*values):
            raise ValueError("post-launch parameters must be finite")
        if min(
            self.gas_constant,
            self.gas_temperature,
            self.atmospheric_pressure_abs,
            self.maximum_cfl,
        ) <= 0.0:
            raise ValueError("post-launch physical scales must be positive")
        if self.gas_entropy_fix_fraction < 0.0:
            raise ValueError("gas entropy-fix fraction cannot be negative")
        if self.maximum_cfl > 1.0:
            raise ValueError("explicit post-launch CFL limit cannot exceed one")
        if not 0.0 < self.geometry_cap_fraction < 1.0:
            raise ValueError("geometry cap fraction must lie in (0, 1)")
        if not 0.0 < self.void_floor_fraction < 1.0:
            raise ValueError("void floor fraction must lie in (0, 1)")
        if (
            self.gas_density_floor_fraction <= 0.0
            or self.gas_density_ceiling_fraction <= 0.0
            or self.gas_density_floor_fraction
            >= self.gas_density_ceiling_fraction
        ):
            raise ValueError("gas density floor must be below its ceiling")
        sound_speed = math.sqrt(self.gas_constant * self.gas_temperature)
        if not math.isclose(
            sound_speed,
            self.node.gas_sound_speed,
            rel_tol=2.0e-14,
            abs_tol=2.0e-14,
        ):
            raise ValueError(
                "boundary gas constants and node gas sound speed disagree"
            )

    @property
    def gas_sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.gas_temperature)

    @property
    def atmospheric_gas_density(self) -> float:
        return self.atmospheric_pressure_abs / self.gas_sound_speed**2


_ROE_INTERNAL_DENSITY_FLOOR = 1.0e-10


@dataclass(frozen=True)
class PostLaunchGasBoundAudit:
    """Raw face audit performed before the Roe-only public API is called."""

    branch_label: str
    face_liquid_area: float
    trace_liquid_area: float
    full_area: float
    geometry_area_cap: float
    face_gas_area: float
    trace_gas_area: float
    void_area_floor: float
    node_gas_density: float
    trace_gas_density: float
    gas_density_floor: float
    gas_density_ceiling: float
    roe_internal_density_floor: float
    face_geometry_cap_active: bool
    trace_geometry_cap_active: bool
    face_void_floor_active: bool
    trace_void_floor_active: bool
    node_density_floor_active: bool
    trace_density_floor_active: bool
    node_density_ceiling_active: bool
    trace_density_ceiling_active: bool
    node_roe_floor_active: bool
    trace_roe_floor_active: bool

    @property
    def active_bounds(self) -> tuple[str, ...]:
        predicates = (
            ("face_geometry_cap", self.face_geometry_cap_active),
            ("trace_geometry_cap", self.trace_geometry_cap_active),
            ("face_void_floor", self.face_void_floor_active),
            ("trace_void_floor", self.trace_void_floor_active),
            ("node_density_floor", self.node_density_floor_active),
            ("trace_density_floor", self.trace_density_floor_active),
            ("node_density_ceiling", self.node_density_ceiling_active),
            ("trace_density_ceiling", self.trace_density_ceiling_active),
            ("node_roe_internal_floor", self.node_roe_floor_active),
            ("trace_roe_internal_floor", self.trace_roe_floor_active),
        )
        return tuple(name for name, active in predicates if active)

    @property
    def accepted_without_bound(self) -> bool:
        return not self.active_bounds


@dataclass(frozen=True)
class PostLaunchGasRiemannDiagnostics:
    """Fail-closed provenance of one accepted resolved-gas face."""

    solver: str
    roe_used: bool
    fallback_used: bool
    fallback_name: str | None
    density_left: float
    density_right: float
    roe_internal_density_floor: float
    roe_density_floor_active: bool


@dataclass(frozen=True)
class CompressibleNodeResolvedBranch:
    """One adjacent stratified trace in its outward branch coordinate."""

    resolved: StratifiedState
    liquid_characteristic: LiquidCharacteristic
    liquid_face_area: float
    full_area: float
    reference_liquid_face_pressure_abs: float
    reference_liquid_pressure_potential: float

    def __post_init__(self) -> None:
        values = (
            self.liquid_face_area,
            self.full_area,
            self.reference_liquid_face_pressure_abs,
            self.reference_liquid_pressure_potential,
        )
        if not _finite(*values):
            raise ValueError("resolved node branch data must be finite")
        if self.full_area <= 0.0:
            raise ValueError("branch full area must be positive")
        if not 0.0 < self.liquid_face_area < self.full_area:
            raise ValueError("node liquid face must be strictly partial")
        if not 0.0 < self.resolved.liquid_area < self.full_area:
            raise ValueError("resolved trace must leave a positive gas area")
        if self.reference_liquid_face_pressure_abs <= 0.0:
            raise ValueError("reference liquid pressure must be positive")

    @property
    def gas_face_area(self) -> float:
        return float(self.full_area - self.liquid_face_area)

    @property
    def trace_gas_area(self) -> float:
        return float(self.full_area - self.resolved.liquid_area)

    @property
    def trace_gas_density(self) -> float:
        return float(self.resolved.gas_mass / self.trace_gas_area)


@dataclass(frozen=True)
class CompressibleNodeBoundaryFlux:
    """Complete branch flux plus the states used in its Riemann solve."""

    flux: StratifiedFlux
    node_gas_density: float
    trace_gas_density: float
    trace_gas_outward_velocity: float
    gas_face_area: float
    liquid_face_pressure_abs: float
    liquid_outward_velocity: float
    bound_audit: PostLaunchGasBoundAudit
    gas_numerics: PostLaunchGasRiemannDiagnostics


@dataclass(frozen=True)
class CompressiblePostLaunchEulerResult:
    """Three branch FV fluxes and the advanced compressible node state."""

    node: CompressibleFiniteNodeEulerResult
    pressure_abs: float
    west: StratifiedFlux
    east: StratifiedFlux
    vertical: StratifiedFlux
    west_trace: CompressibleNodeBoundaryFlux
    east_trace: CompressibleNodeBoundaryFlux
    vertical_trace: CompressibleNodeBoundaryFlux
    gas_cfl: float
    liquid_cfl: float
    maximum_dt: float

    @property
    def branch_fluxes(self) -> dict[str, StratifiedFlux]:
        return {
            "west": self.west,
            "east": self.east,
            "vertical": self.vertical,
        }


def stratified_trace_in_outward_coordinate(
    trace: StratifiedState,
    outward_axis_sign: int,
) -> StratifiedState:
    """Convert both phase momenta from a global axis to an outward branch."""

    if outward_axis_sign not in (-1, 1):
        raise ValueError("outward axis sign must be exactly -1 or +1")
    return StratifiedState(
        gas_mass=trace.gas_mass,
        gas_momentum=float(outward_axis_sign) * trace.gas_momentum,
        liquid_area=trace.liquid_area,
        liquid_discharge=float(outward_axis_sign) * trace.liquid_discharge,
    )


def liquid_characteristic_in_outward_coordinate(
    characteristic: LiquidCharacteristic,
    outward_axis_sign: int,
) -> LiquidCharacteristic:
    """Convert a characteristic's reference velocity to outward form."""

    if outward_axis_sign not in (-1, 1):
        raise ValueError("outward axis sign must be exactly -1 or +1")
    return LiquidCharacteristic(
        reference_pressure_abs=characteristic.reference_pressure_abs,
        reference_outward_velocity=(
            float(outward_axis_sign)
            * characteristic.reference_outward_velocity
        ),
        wave_speed=characteristic.wave_speed,
        loss_coefficient=characteristic.loss_coefficient,
        pressure_offset=characteristic.pressure_offset,
    )


def audit_postlaunch_branch_bounds(
    branch: CompressibleNodeResolvedBranch,
    *,
    node_gas_density: float,
    params: CompressiblePostLaunchParameters,
    branch_label: str,
) -> PostLaunchGasBoundAudit:
    """Reject raw states that would activate a reused lower-level bound."""

    if not isinstance(branch_label, str) or not branch_label:
        raise ValueError("branch label must be a non-empty string")
    if not math.isfinite(node_gas_density) or node_gas_density <= 0.0:
        raise CompressiblePostLaunchError(
            "node gas density must be positive before Roe evaluation"
        )
    area_cap = params.geometry_cap_fraction * branch.full_area
    void_floor = params.void_floor_fraction * branch.full_area
    density_floor = (
        params.gas_density_floor_fraction
        * params.atmospheric_gas_density
    )
    density_ceiling = (
        params.gas_density_ceiling_fraction
        * params.atmospheric_gas_density
    )
    trace_density = branch.trace_gas_density
    audit = PostLaunchGasBoundAudit(
        branch_label=branch_label,
        face_liquid_area=float(branch.liquid_face_area),
        trace_liquid_area=float(branch.resolved.liquid_area),
        full_area=float(branch.full_area),
        geometry_area_cap=float(area_cap),
        face_gas_area=branch.gas_face_area,
        trace_gas_area=branch.trace_gas_area,
        void_area_floor=float(void_floor),
        node_gas_density=float(node_gas_density),
        trace_gas_density=float(trace_density),
        gas_density_floor=float(density_floor),
        gas_density_ceiling=float(density_ceiling),
        roe_internal_density_floor=_ROE_INTERNAL_DENSITY_FLOOR,
        face_geometry_cap_active=bool(branch.liquid_face_area >= area_cap),
        trace_geometry_cap_active=bool(
            branch.resolved.liquid_area >= area_cap
        ),
        face_void_floor_active=bool(branch.gas_face_area <= void_floor),
        trace_void_floor_active=bool(branch.trace_gas_area <= void_floor),
        node_density_floor_active=bool(node_gas_density <= density_floor),
        trace_density_floor_active=bool(trace_density <= density_floor),
        node_density_ceiling_active=bool(node_gas_density >= density_ceiling),
        trace_density_ceiling_active=bool(trace_density >= density_ceiling),
        node_roe_floor_active=bool(
            node_gas_density < _ROE_INTERNAL_DENSITY_FLOOR
        ),
        trace_roe_floor_active=bool(
            trace_density < _ROE_INTERNAL_DENSITY_FLOOR
        ),
    )
    if not audit.accepted_without_bound:
        raise PostLaunchBoundActivationError(audit)
    return audit


def _branch_flux(
    branch: CompressibleNodeResolvedBranch,
    *,
    pressure_abs: float,
    node_gas_density: float,
    bound_audit: PostLaunchGasBoundAudit,
    params: CompressiblePostLaunchParameters,
) -> CompressibleNodeBoundaryFlux:
    if not bound_audit.accepted_without_bound:
        raise PostLaunchBoundActivationError(bound_audit)
    trace_velocity = branch.resolved.gas_velocity
    if abs(trace_velocity) >= params.gas_sound_speed:
        raise PostLaunchSubsonicTraceError(
            "resolved gas trace must satisfy |u_out| < c_g"
        )
    gas_mass_per_area, gas_momentum_per_area = (
        isothermal_ideal_gas_riemann_flux(
            node_gas_density,
            0.0,
            branch.trace_gas_density,
            trace_velocity,
            gas_constant=params.gas_constant,
            temperature=params.gas_temperature,
            entropy_fix_fraction=params.gas_entropy_fix_fraction,
        )
    )
    gas_numerics = PostLaunchGasRiemannDiagnostics(
        solver="positive-density Roe",
        roe_used=True,
        fallback_used=False,
        fallback_name=None,
        density_left=float(node_gas_density),
        density_right=branch.trace_gas_density,
        roe_internal_density_floor=_ROE_INTERNAL_DENSITY_FLOOR,
        roe_density_floor_active=False,
    )

    characteristic = branch.liquid_characteristic
    liquid_velocity = characteristic.outward_velocity(
        pressure_abs,
        liquid_density=params.node.liquid_density,
    )
    liquid_volume_flux = branch.liquid_face_area * liquid_velocity
    liquid_face_pressure = pressure_abs + characteristic.pressure_offset
    if liquid_face_pressure <= 0.0:
        raise CompressiblePostLaunchError(
            "node pressure gives a non-positive liquid face pressure"
        )
    liquid_potential = (
        branch.reference_liquid_pressure_potential
        + (liquid_face_pressure - branch.reference_liquid_face_pressure_abs)
        * branch.liquid_face_area
        / params.node.liquid_density
    )
    liquid_momentum_flux = (
        liquid_volume_flux * liquid_velocity + liquid_potential
    )
    flux = StratifiedFlux(
        gas_mass=gas_mass_per_area * branch.gas_face_area,
        gas_momentum=(
            gas_momentum_per_area - params.atmospheric_pressure_abs
        )
        * branch.gas_face_area,
        liquid_area=liquid_volume_flux,
        liquid_momentum=liquid_momentum_flux,
    )
    return CompressibleNodeBoundaryFlux(
        flux=flux,
        node_gas_density=float(node_gas_density),
        trace_gas_density=branch.trace_gas_density,
        trace_gas_outward_velocity=float(trace_velocity),
        gas_face_area=branch.gas_face_area,
        liquid_face_pressure_abs=float(liquid_face_pressure),
        liquid_outward_velocity=float(liquid_velocity),
        bound_audit=bound_audit,
        gas_numerics=gas_numerics,
    )


def _spectral_rates(
    *,
    gas_volume: float,
    liquid_volume: float,
    branches: tuple[CompressibleNodeResolvedBranch, ...],
    fluxes: tuple[CompressibleNodeBoundaryFlux, ...],
    params: CompressiblePostLaunchParameters,
) -> tuple[float, float]:
    if gas_volume <= 0.0 or liquid_volume <= 0.0:
        raise ExactLaunchRequiresEventClosure(
            "post-launch boundary stage requires both physical phase volumes"
        )
    gas_rate = math.fsum(
        flux.gas_face_area
        * (params.gas_sound_speed + abs(flux.trace_gas_outward_velocity))
        for flux in fluxes
    ) / gas_volume
    liquid_rate = math.fsum(
        branch.liquid_face_area
        * (
            branch.liquid_characteristic.wave_speed
            + max(
                abs(branch.liquid_characteristic.reference_outward_velocity),
                abs(flux.liquid_outward_velocity),
            )
        )
        for branch, flux in zip(branches, fluxes)
    ) / liquid_volume
    return float(gas_rate), float(liquid_rate)


def euler_compressible_node_postlaunch_stage(
    state: CompressibleFiniteNodeState,
    dt: float,
    *,
    west: CompressibleNodeResolvedBranch,
    east: CompressibleNodeResolvedBranch,
    vertical: CompressibleNodeResolvedBranch,
    params: CompressiblePostLaunchParameters,
) -> CompressiblePostLaunchEulerResult:
    """Return three complete outward FV fluxes and advance one Euler stage."""

    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("post-launch Euler dt must be finite and non-negative")
    if state.gas_mass <= 0.0:
        raise ExactLaunchRequiresEventClosure(
            "exact m_g=0 must be advanced by the launch event closure"
        )
    pressure = solve_compressible_node_pressure(state, params.node)
    if pressure.gas_physical_volume <= 0.0:
        raise ExactLaunchRequiresEventClosure(
            "post-launch node has no finite gas reservoir volume"
        )
    # Use the isothermal EOS form directly.  It is algebraically identical to
    # m_g/V_g, but shares the exact arithmetic expression used to reconstruct
    # an equilibrium branch density and therefore preserves uniform rest.
    node_density = pressure.pressure_abs / params.gas_sound_speed**2
    branches = (west, east, vertical)
    bound_audits = tuple(
        audit_postlaunch_branch_bounds(
            branch,
            node_gas_density=node_density,
            params=params,
            branch_label=label,
        )
        for label, branch in zip(("west", "east", "vertical"), branches)
    )
    evaluated = tuple(
        _branch_flux(
            branch,
            pressure_abs=pressure.pressure_abs,
            node_gas_density=node_density,
            bound_audit=audit,
            params=params,
        )
        for branch, audit in zip(branches, bound_audits)
    )

    gas_rate, liquid_rate = _spectral_rates(
        gas_volume=pressure.gas_physical_volume,
        liquid_volume=pressure.liquid_physical_volume,
        branches=branches,
        fluxes=evaluated,
        params=params,
    )
    gas_cfl = dt * gas_rate
    liquid_cfl = dt * liquid_rate
    spectral_rate = max(gas_rate, liquid_rate)
    maximum_dt = params.maximum_cfl / spectral_rate
    cfl_roundoff = 64.0 * math.ulp(max(params.maximum_cfl, 1.0))
    if max(gas_cfl, liquid_cfl) > params.maximum_cfl + cfl_roundoff:
        raise PostLaunchCFLInadmissible(
            gas_cfl=gas_cfl,
            liquid_cfl=liquid_cfl,
            maximum_cfl=params.maximum_cfl,
            maximum_dt=maximum_dt,
        )

    rates = tuple(
        CompressibleNodeBranchRates(
            gas_mass_outward=item.flux.gas_mass,
            liquid_equivalent_volume_outward=item.flux.liquid_area,
            evaluation_pressure_abs=pressure.pressure_abs,
        )
        for item in evaluated
    )
    node_result = euler_compressible_finite_node_stage(
        state,
        dt,
        west=rates[0],
        east=rates[1],
        vertical=rates[2],
        params=params.node,
    )
    return CompressiblePostLaunchEulerResult(
        node=node_result,
        pressure_abs=float(pressure.pressure_abs),
        west=evaluated[0].flux,
        east=evaluated[1].flux,
        vertical=evaluated[2].flux,
        west_trace=evaluated[0],
        east_trace=evaluated[1],
        vertical_trace=evaluated[2],
        gas_cfl=float(gas_cfl),
        liquid_cfl=float(liquid_cfl),
        maximum_dt=float(maximum_dt),
    )


__all__ = [
    "PRODUCTION_READY",
    "CompressibleNodeBoundaryFlux",
    "CompressibleNodeResolvedBranch",
    "CompressiblePostLaunchError",
    "CompressiblePostLaunchEulerResult",
    "CompressiblePostLaunchParameters",
    "ExactLaunchRequiresEventClosure",
    "PostLaunchCFLInadmissible",
    "PostLaunchBoundActivationError",
    "PostLaunchGasBoundAudit",
    "PostLaunchGasRiemannDiagnostics",
    "PostLaunchSubsonicTraceError",
    "audit_postlaunch_branch_bounds",
    "euler_compressible_node_postlaunch_stage",
    "liquid_characteristic_in_outward_coordinate",
    "stratified_trace_in_outward_coordinate",
]
