"""Pure S1-1D-F0 pressure/void stage component for the vertical riser.

The component deliberately owns no committer and no conservation ledger.  It
consumes one immutable ``riser_T`` trial, evaluates the persistent
``Aup/Qup/Adown/Qdown/Mg/Jg`` riser state, and returns either a complete
``ComponentStageProposal`` or a non-committable capacity rejection.

The pinned Case-1 finite-volume liquid transport and its implicit physical
gas/up-liquid/down-liquid equal-recoil exchange are reused.  Gross liquid
streams remain independent state variables; they are never reconstructed from
their net discharge.  A conservative gas-volume remap pairs every finite void
with positive gas mass before the gas transport stage.  A persistent exterior
owner may supply an immutable finite falling parcel to one stage-specific
clone of this component; the component never invents exterior inventory.
Ordered multi-front piston motion and explicit top spill are supported.  A
trial that would teleport a closed gas pocket through a resolved liquid column,
or that needs two axial interfaces inside one cut cell, continues to fail
closed.

The component is exported for explicit network integration and testing.  The
joint nonlinear T-node solver and all remaining production gates must still
pass before any physical trajectory is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

from .errors import ContractViolation, MissingPhysicalClosure
from .flux import BoundaryExchange, VerticalDelta
from .port_contracts import (
    CapacityReject,
    CapillaryInterfaceOwnership,
    ComponentStageProposal,
    GrossNodePortFlux,
    PhysicalStage,
    PortKey,
    PortTraceState,
    TNodeTrial,
    validate_trial_set,
)
from .state import CoupledGeometry, CoupledState, VerticalState
from .vertical_case1_adapter import (
    ATMOSPHERIC_PRESSURE_PA,
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    INITIAL_AIR_TEMPERATURE_K,
    PIPE_DIAMETER_M,
)
from .vertical_twostream_solver import (
    S1_GAS_VISCOSITY_PA_S,
    S1_GRAVITY_M_S2,
    S1_LIQUID_DENSITY_KG_M3,
    S1VerticalClosures,
    S1VerticalTwoStreamSolver,
    _component_state,
)


S1_LIQUID_VISCOSITY_PA_S = 1.002e-3
F0_SURFACE_TENSION_N_M = 0.072
_VOID_TOLERANCE_M2 = 1.0e-14
_MASS_TOLERANCE_KG = 1.0e-16
_MOMENTUM_TOLERANCE_KG_M_S = 1.0e-16
_MATERIAL_LEDGER_TOLERANCE = 2.0e-14
_MOMENTUM_LEDGER_TOLERANCE = 2.0e-11
_DIRECTION_REVERSAL_RATE_TOLERANCE_M3_S = 1.0e-14

CapillaryGeometryMode = Literal[
    "planar_2d_zeroGradient_walls",
    "circular_3d_pipe",
]


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ContractViolation(f"{name} must be positive")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ContractViolation(f"{name} must be non-negative")
    return result


def f0_smooth_pipe_darcy_factor(reynolds: float) -> float:
    """Frozen F0 smooth-pipe Darcy law, including its transition blend."""

    re = _nonnegative("Re", reynolds)
    if re <= 1.0e-12:
        return 0.0
    if re <= 2300.0:
        return 64.0 / re
    turbulent = 0.3164 / re**0.25
    if re >= 4000.0:
        return turbulent
    laminar = 64.0 / re
    weight = (re - 2300.0) / 1700.0
    return (1.0 - weight) * laminar + weight * turbulent


@dataclass(frozen=True, slots=True)
class AtmosphericLiquidFallback:
    """One explicit finite exterior parcel available for rim re-entry.

    The parcel is a boundary input, not an implicit infinite reservoir.  Its
    available volume must be supplied by an exterior-plume state owner (for
    example, liquid previously recorded as rim outflow).  This component can
    consume at most that volume during one RK stage, but it deliberately does
    not advance the exterior owner; consequently this stage closure alone is
    not a complete repeated-cycle boundary.
    """

    donor_area_m2: float
    downward_speed_m_s: float
    available_volume_m3: float | None = None
    absolute_pressure_Pa: float = ATMOSPHERIC_PRESSURE_PA
    evidence_status: str = "explicit_finite_external_liquid_parcel"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "donor_area_m2", _positive("fallback donor area", self.donor_area_m2)
        )
        object.__setattr__(
            self,
            "downward_speed_m_s",
            _positive("fallback downward speed", self.downward_speed_m_s),
        )
        object.__setattr__(
            self,
            "absolute_pressure_Pa",
            _positive("fallback absolute pressure", self.absolute_pressure_Pa),
        )
        if self.available_volume_m3 is not None:
            object.__setattr__(
                self,
                "available_volume_m3",
                _nonnegative(
                    "fallback available volume", self.available_volume_m3
                ),
            )
        if not self.evidence_status.strip():
            raise ContractViolation("fallback evidence_status must be non-empty")

    @property
    def downward_rate_m3_s(self) -> float:
        return self.donor_area_m2 * self.downward_speed_m_s

    @property
    def finite_stage_inventory_ready(self) -> bool:
        return self.available_volume_m3 is not None

    def admissible_rate_m3_s(self, *, dt_s: float) -> float:
        dt = _positive("fallback RK-stage dt", dt_s)
        if self.available_volume_m3 is None:
            raise MissingPhysicalClosure(
                "riser-top liquid re-entry requires a finite exterior parcel "
                "volume from an explicit exterior state owner"
            )
        return min(self.downward_rate_m3_s, self.available_volume_m3 / dt)


@dataclass(frozen=True, slots=True)
class AtmosphericTopState:
    """Open gas reservoir and optional exterior liquid state above the rim."""

    absolute_pressure_Pa: float = ATMOSPHERIC_PRESSURE_PA
    temperature_K: float = INITIAL_AIR_TEMPERATURE_K
    gas_constant_J_kg_K: float = DRY_AIR_GAS_CONSTANT_J_KG_K
    gas_axial_velocity_m_s: float = 0.0
    liquid_fallback: AtmosphericLiquidFallback | None = None

    def __post_init__(self) -> None:
        for name in (
            "absolute_pressure_Pa",
            "temperature_K",
            "gas_constant_J_kg_K",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(
            self,
            "gas_axial_velocity_m_s",
            _finite("gas_axial_velocity_m_s", self.gas_axial_velocity_m_s),
        )
        if self.liquid_fallback is not None and not math.isclose(
            self.liquid_fallback.absolute_pressure_Pa,
            self.absolute_pressure_Pa,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ContractViolation(
                "finite exterior liquid parcel and atmospheric top pressure differ"
            )

    @property
    def gas_density_kg_m3(self) -> float:
        return self.absolute_pressure_Pa / (
            self.gas_constant_J_kg_K * self.temperature_K
        )

    @property
    def gas_sound_speed_m_s(self) -> float:
        return math.sqrt(self.gas_constant_J_kg_K * self.temperature_K)

    @property
    def full_cycle_liquid_fallback_ready(self) -> bool:
        # A finite stage parcel is conservative, but this component has no
        # persistent exterior-plume state in which to store prior outflow and
        # update the remaining parcel after re-entry.
        return False

    @property
    def finite_stage_liquid_reentry_ready(self) -> bool:
        return (
            self.liquid_fallback is not None
            and self.liquid_fallback.finite_stage_inventory_ready
        )


@dataclass(frozen=True, slots=True)
class DetectedCapillaryInterface:
    face_index: int
    gas_is_above: bool
    geometry_kind: str
    record: CapillaryInterfaceOwnership


@dataclass(frozen=True, slots=True)
class F0VerticalCapillaryOwner:
    """One declared owner for all axial riser free-surface/slug interfaces.

    ``planar_2d_zeroGradient_walls`` does not invent a contact angle.  A
    stationary source free surface is flat (kappa=0).  Once an interface is
    moving, the frozen 2-D analogue is a planar semicircular cap with
    |kappa|=2/D.  This is a declared model-form translation, not a claim that
    OpenFOAM's local CSF curvature is uniformly zero.

    ``circular_3d_pipe`` uses |kappa|=4*cos(theta)/D and therefore requires an
    explicit contact angle.  No default angle is supplied.
    """

    mode: CapillaryGeometryMode | None = None
    circular_contact_angle_deg: float | None = None
    surface_tension_N_m: float = F0_SURFACE_TENSION_N_M
    static_velocity_tolerance_m_s: float = 1.0e-12
    evidence_status: str = "declared_F0_capillary_geometry_translation"

    def __post_init__(self) -> None:
        if self.mode not in (
            None,
            "planar_2d_zeroGradient_walls",
            "circular_3d_pipe",
        ):
            raise ContractViolation(f"unsupported capillary geometry mode: {self.mode!r}")
        sigma = _positive("surface_tension_N_m", self.surface_tension_N_m)
        if not math.isclose(
            sigma, F0_SURFACE_TENSION_N_M, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ContractViolation("S1-1D-F0 surface tension must remain 0.072 N/m")
        object.__setattr__(self, "surface_tension_N_m", sigma)
        object.__setattr__(
            self,
            "static_velocity_tolerance_m_s",
            _nonnegative(
                "static_velocity_tolerance_m_s",
                self.static_velocity_tolerance_m_s,
            ),
        )
        angle = self.circular_contact_angle_deg
        if angle is not None:
            angle = _finite("circular_contact_angle_deg", angle)
            if not 0.0 <= angle <= 180.0:
                raise ContractViolation("circular contact angle must lie in [0, 180]")
            object.__setattr__(self, "circular_contact_angle_deg", angle)
        if self.mode == "circular_3d_pipe" and angle is None:
            # Construction is legal for a readiness audit; interface use will
            # fail closed instead of substituting a guessed angle.
            pass
        if not self.evidence_status.strip():
            raise ContractViolation("capillary evidence_status must be non-empty")

    @property
    def production_ready(self) -> bool:
        if self.mode == "planar_2d_zeroGradient_walls":
            return True
        return self.mode == "circular_3d_pipe" and self.circular_contact_angle_deg is not None

    def detect(
        self,
        state,
        *,
        gas_mass_cell_kg: tuple[float, ...],
        gas_momentum_cell_kg_m_s: tuple[float, ...],
        full_area_m2: float,
        diameter_m: float,
    ) -> tuple[DetectedCapillaryInterface, ...]:
        n = state.cell_count
        gas_area = tuple(
            max(full_area_m2 - up - down, 0.0)
            for up, down in zip(
                state.upward_area, state.downward_area, strict=True
            )
        )
        gas_dominant = tuple(area > 0.5 * full_area_m2 for area in gas_area)
        faces = tuple(
            face
            for face in range(1, n)
            if gas_dominant[face - 1] != gas_dominant[face]
        )
        if not faces:
            return ()
        if self.mode is None:
            raise MissingPhysicalClosure(
                "vertical capillary geometry mode is unselected"
            )
        if self.mode == "circular_3d_pipe" and self.circular_contact_angle_deg is None:
            raise MissingPhysicalClosure(
                "3-D circular-pipe capillarity requires an explicit contact angle"
            )

        velocities: list[float] = []
        for area, discharge in zip(
            state.upward_area, state.upward_discharge, strict=True
        ):
            if area > 0.0:
                velocities.append(abs(discharge / area))
        for area, discharge in zip(
            state.downward_area, state.downward_discharge, strict=True
        ):
            if area > 0.0:
                velocities.append(abs(discharge / area))
        for mass, momentum in zip(
            gas_mass_cell_kg, gas_momentum_cell_kg_m_s, strict=True
        ):
            if mass > 0.0:
                velocities.append(abs(momentum / mass))
        stationary_single_surface = (
            len(faces) == 1
            and max(velocities, default=0.0)
            <= self.static_velocity_tolerance_m_s
        )

        detected: list[DetectedCapillaryInterface] = []
        for face in faces:
            gas_above = gas_dominant[face]
            orientation = 1.0 if gas_above else -1.0
            if stationary_single_surface:
                curvature = 0.0
                geometry_kind = "flat_source_free_surface"
                contact_angle = None
            elif self.mode == "planar_2d_zeroGradient_walls":
                curvature = orientation * 2.0 / diameter_m
                geometry_kind = "declared_planar_semicircular_cap"
                contact_angle = None
            else:
                assert self.circular_contact_angle_deg is not None
                theta = math.radians(self.circular_contact_angle_deg)
                curvature = orientation * 4.0 * math.cos(theta) / diameter_m
                geometry_kind = "declared_circular_3d_meniscus"
                contact_angle = self.circular_contact_angle_deg
            jump = self.surface_tension_N_m * curvature
            record = CapillaryInterfaceOwnership(
                interface_id=f"vertical-riser-face-{face}",
                owner="vertical_riser",
                surface_tension_N_m=self.surface_tension_N_m,
                geometry_mode=self.mode,
                curvature_1_m=curvature,
                contact_angle_deg=contact_angle,
                pressure_jump_gas_minus_liquid_Pa=jump,
                evidence_status=(
                    f"{self.evidence_status}__{geometry_kind}"
                ),
            )
            detected.append(
                DetectedCapillaryInterface(
                    face_index=face,
                    gas_is_above=gas_above,
                    geometry_kind=geometry_kind,
                    record=record,
                )
            )
        return tuple(detected)


@dataclass(frozen=True, slots=True)
class VoidRemapResult:
    gas_mass_cell_kg: tuple[float, ...]
    gas_momentum_cell_kg_m_s: tuple[float, ...]
    connected_component_count: int
    mass_residual_kg: float
    momentum_residual_kg_m_s: float
    boundary_source_mass_kg: float = 0.0
    boundary_source_momentum_kg_m_s: float = 0.0


@dataclass(frozen=True, slots=True)
class BottomGasPistonRemap:
    """Conservative liquid displacement that opens a bottom gas cut cell.

    ``deposited_liquid_volume_m3`` and ``top_spill_volume_m3`` form an
    exhaustive partition of the one bottom donor parcel.  The parcel may fill
    more than one resolved front while a gas gap is being traversed, but it is
    removed from cell zero only once.  Existing directional labels are never
    relabelled or combined across a still-finite gas corridor.
    """

    requested_gas_volume_m3: float
    preexisting_bottom_void_volume_m3: float
    displaced_liquid_volume_m3: float
    destination_cell: int | None
    liquid_volume_residual_m3: float
    liquid_momentum_residual_kg_m_s: float
    deposited_liquid_volume_m3: float = 0.0
    top_spill_volume_m3: float = 0.0
    top_spill_momentum_kg_m_s: float = 0.0
    receiving_cells: tuple[int, ...] = ()
    traversed_gap_cells: tuple[int, ...] = ()


def _active_components(areas: tuple[float, ...], tolerance: float) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    current: list[int] = []
    for cell, area in enumerate(areas):
        if area > tolerance:
            current.append(cell)
        elif current:
            result.append(tuple(current))
            current = []
    if current:
        result.append(tuple(current))
    return tuple(result)


def conservative_void_remap(
    *,
    old_void_area_m2: tuple[float, ...],
    new_void_area_m2: tuple[float, ...],
    gas_mass_cell_kg: tuple[float, ...],
    gas_momentum_cell_kg_m_s: tuple[float, ...],
    cell_length_m: float,
    void_tolerance_m2: float = _VOID_TOLERANCE_M2,
    boundary_source_cell: int | None = None,
    boundary_source_mass_kg: float = 0.0,
    boundary_source_momentum_kg_m_s: float = 0.0,
) -> VoidRemapResult:
    """Conservatively remap gas inventory into a changed resolved void.

    Source parcels are assigned to the nearest connected new-void component.
    Within each component, overlap in normalized cumulative void volume maps
    mass and momentum without inventing either.  A newly isolated component
    with no source parcel is rejected rather than filled with background gas.

    ``boundary_source_*`` is the *already integrated* parcel admitted through
    one explicit boundary during this same RK stage.  It is part of this
    atomic remap, not a background fill.  Callers must consequently remove the
    same inflow from their subsequent boundary transport or it would be
    injected twice.  A boundary parcel may seed only the connected component
    containing ``boundary_source_cell``.
    """

    n = len(old_void_area_m2)
    if n == 0 or any(
        len(values) != n
        for values in (
            new_void_area_m2,
            gas_mass_cell_kg,
            gas_momentum_cell_kg_m_s,
        )
    ):
        raise ContractViolation("void remap vectors must have one common non-zero length")
    dz = _positive("cell_length_m", cell_length_m)
    tolerance = _nonnegative("void_tolerance_m2", void_tolerance_m2)
    old_area = tuple(_nonnegative("old void area", value) for value in old_void_area_m2)
    new_area = tuple(_nonnegative("new void area", value) for value in new_void_area_m2)
    mass = tuple(_nonnegative("gas cell mass", value) for value in gas_mass_cell_kg)
    momentum = tuple(_finite("gas cell momentum", value) for value in gas_momentum_cell_kg_m_s)
    source_mass = _nonnegative("boundary source mass", boundary_source_mass_kg)
    source_momentum = _finite(
        "boundary source momentum", boundary_source_momentum_kg_m_s
    )
    if source_mass == 0.0:
        if source_momentum != 0.0:
            raise ContractViolation("zero boundary gas mass cannot carry momentum")
        if boundary_source_cell is not None:
            raise ContractViolation(
                "boundary_source_cell requires a positive boundary gas parcel"
            )
    else:
        if not isinstance(boundary_source_cell, int) or not 0 <= boundary_source_cell < n:
            raise ContractViolation(
                "positive boundary gas parcel requires an in-range source cell"
            )
        if new_area[boundary_source_cell] <= tolerance:
            raise MissingPhysicalClosure(
                "boundary gas parcel has no finite receiving void at its source cell"
            )

    for cell, (area, cell_mass, cell_momentum) in enumerate(
        zip(old_area, mass, momentum, strict=True)
    ):
        if area <= tolerance and (
            cell_mass > _MASS_TOLERANCE_KG
            or abs(cell_momentum) > _MOMENTUM_TOLERANCE_KG_M_S
        ):
            raise MissingPhysicalClosure(
                f"gas inventory occupies zero old void in cell {cell}"
            )
        if area > tolerance and cell_mass <= _MASS_TOLERANCE_KG:
            raise MissingPhysicalClosure(
                f"old void cell {cell} has no gas mass"
            )

    components = _active_components(new_area, tolerance)
    old_components = _active_components(old_area, tolerance)
    old_total_mass = math.fsum(mass)
    old_total_momentum = math.fsum(momentum)
    total_mass = old_total_mass + source_mass
    total_momentum = old_total_momentum + source_momentum
    if not components:
        if total_mass > _MASS_TOLERANCE_KG or abs(total_momentum) > _MOMENTUM_TOLERANCE_KG_M_S:
            raise MissingPhysicalClosure("new liquid state has no capacity for existing gas")
        zero = (0.0,) * n
        return VoidRemapResult(
            zero,
            zero,
            0,
            0.0,
            0.0,
            boundary_source_mass_kg=source_mass,
            boundary_source_momentum_kg_m_s=source_momentum,
        )

    # A disappearing isolated gas pocket cannot be teleported through a
    # finite liquid column merely because another void exists elsewhere.  A
    # one-cell adjacency is the finite-volume representation of an interface
    # crossing a face; any larger separation needs a resolved gas transport
    # path and therefore fails closed.
    for old_component in old_components:
        if math.fsum(mass[cell] for cell in old_component) <= _MASS_TOLERANCE_KG:
            continue
        nearest = min(
            abs(old_cell - new_cell)
            for old_cell in old_component
            for destination in components
            for new_cell in destination
        )
        if nearest > 1:
            raise MissingPhysicalClosure(
                "a finite gas component would cross a resolved liquid column during void remap"
            )

    owner_by_cell: dict[int, int] = {}
    for component_index, cells in enumerate(components):
        for cell in cells:
            owner_by_cell[cell] = component_index

    assigned: list[list[int]] = [[] for _ in components]
    for cell, cell_mass in enumerate(mass):
        if cell_mass <= _MASS_TOLERANCE_KG:
            continue
        owner = owner_by_cell.get(cell)
        if owner is None:
            distances = tuple(
                min(abs(cell - candidate) for candidate in cells)
                for cells in components
            )
            owner = min(range(len(components)), key=lambda index: (distances[index], index))
        assigned[owner].append(cell)

    boundary_owner: int | None = None
    if source_mass > 0.0:
        assert boundary_source_cell is not None
        boundary_owner = owner_by_cell.get(boundary_source_cell)
        if boundary_owner is None:
            raise MissingPhysicalClosure(
                "boundary gas parcel source cell is outside every finite new void"
            )

    mapped_mass = [0.0] * n
    mapped_momentum = [0.0] * n
    for component_index, destination_cells in enumerate(components):
        source_cells = sorted(assigned[component_index])
        old_component_mass = math.fsum(mass[cell] for cell in source_cells)
        parcel_mass = source_mass if boundary_owner == component_index else 0.0
        component_mass = old_component_mass + parcel_mass
        if component_mass <= _MASS_TOLERANCE_KG:
            raise MissingPhysicalClosure(
                "a newly isolated finite void component has no conservative gas source"
            )
        destination_weights = [new_area[cell] * dz for cell in destination_cells]
        destination_total = math.fsum(destination_weights)
        if destination_total <= 0.0:
            raise MissingPhysicalClosure("finite void component has no destination volume")

        if source_cells:
            source_weights = [old_area[cell] * dz for cell in source_cells]
            source_total = math.fsum(source_weights)
            if source_total <= 0.0:
                raise MissingPhysicalClosure("gas source parcels have no old void volume")
            source_edges = [0.0]
            destination_edges = [0.0]
            for weight in source_weights:
                source_edges.append(source_edges[-1] + weight / source_total)
            for weight in destination_weights:
                destination_edges.append(
                    destination_edges[-1] + weight / destination_total
                )
            source_edges[-1] = 1.0
            destination_edges[-1] = 1.0

            for source_position, source_cell in enumerate(source_cells):
                s0 = source_edges[source_position]
                s1 = source_edges[source_position + 1]
                width = s1 - s0
                for destination_position, destination_cell in enumerate(
                    destination_cells
                ):
                    d0 = destination_edges[destination_position]
                    d1 = destination_edges[destination_position + 1]
                    overlap = max(min(s1, d1) - max(s0, d0), 0.0)
                    if overlap <= 0.0:
                        continue
                    fraction = overlap / width
                    mapped_mass[destination_cell] += mass[source_cell] * fraction
                    mapped_momentum[destination_cell] += (
                        momentum[source_cell] * fraction
                    )

        if parcel_mass > 0.0:
            for destination_cell, weight in zip(
                destination_cells, destination_weights, strict=True
            ):
                fraction = weight / destination_total
                mapped_mass[destination_cell] += parcel_mass * fraction
                mapped_momentum[destination_cell] += source_momentum * fraction

    for cell, area in enumerate(new_area):
        if area > tolerance and mapped_mass[cell] <= _MASS_TOLERANCE_KG:
            raise MissingPhysicalClosure(
                f"void remap left massless finite void in cell {cell}"
            )
        if area <= tolerance:
            mapped_mass[cell] = 0.0
            mapped_momentum[cell] = 0.0

    mass_residual = math.fsum(mapped_mass) - total_mass
    momentum_residual = math.fsum(mapped_momentum) - total_momentum
    scale = max(abs(total_mass), abs(total_momentum), 1.0)
    if max(abs(mass_residual), abs(momentum_residual)) > 2.0e-14 * scale:
        raise ContractViolation("conservative void remap ledger does not close")
    return VoidRemapResult(
        gas_mass_cell_kg=tuple(mapped_mass),
        gas_momentum_cell_kg_m_s=tuple(mapped_momentum),
        connected_component_count=len(components),
        mass_residual_kg=mass_residual,
        momentum_residual_kg_m_s=momentum_residual,
        boundary_source_mass_kg=source_mass,
        boundary_source_momentum_kg_m_s=source_momentum,
    )


@dataclass(frozen=True, slots=True)
class AtmosphericGasFlux:
    outflow_kg_s: float
    inflow_kg_s: float
    outflow_speed_m_s: float
    inflow_speed_m_s: float
    contact_velocity_m_s: float
    contact_pressure_Pa: float


@dataclass(frozen=True, slots=True)
class AtmosphericLiquidFlux:
    """Auditable liquid exchange selected at the atmospheric rim."""

    outflow_rate_m3_s: float
    outflow_speed_m_s: float
    reentry_demand_rate_m3_s: float
    reentry_rate_m3_s: float
    reentry_speed_m_s: float
    exterior_available_volume_m3: float | None
    stage_consumed_volume_m3: float
    finite_exterior_inventory: bool


@dataclass(frozen=True, slots=True)
class DynamicWallDiagnostics:
    liquid_up_darcy_factor: tuple[float, ...]
    liquid_down_darcy_factor: tuple[float, ...]
    gas_darcy_factor: tuple[float, ...]
    liquid_up_hydraulic_diameter_m: tuple[float, ...]
    liquid_down_hydraulic_diameter_m: tuple[float, ...]
    gas_hydraulic_diameter_m: tuple[float, ...]
    wall_impulse_kg_m_s: float


@dataclass(frozen=True, slots=True)
class GasTransportResult:
    gas_mass_cell_kg: tuple[float, ...]
    gas_momentum_cell_kg_m_s: tuple[float, ...]
    top: AtmosphericGasFlux
    mass_residual_kg: float
    momentum_residual_kg_m_s: float
    bottom_advective_impulse_kg_m_s: float
    top_advective_impulse_kg_m_s: float
    pressure_impulse_kg_m_s: float
    gravity_impulse_kg_m_s: float


@dataclass(frozen=True, slots=True)
class IndependentVerticalMomentumBudget:
    """Result-independent physical impulse budget for one riser stage.

    Every term is fixed by the accepted port packet or by an operator substep
    before the final state is assembled.  In particular, no term is obtained
    by subtracting the initial momentum from the final momentum.
    """

    bottom_node_advective_impulse_kg_m_s: float
    bottom_node_pressure_impulse_kg_m_s: float
    top_advective_impulse_kg_m_s: float
    top_pressure_traction_impulse_kg_m_s: float
    liquid_pressure_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    discrete_pressure_traction_residual_kg_m_s: float
    liquid_gravity_impulse_kg_m_s: float
    gas_gravity_impulse_kg_m_s: float
    wall_impulse_kg_m_s: float
    capillary_external_impulse_kg_m_s: float
    predicted_mixture_impulse_kg_m_s: float
    actual_mixture_impulse_kg_m_s: float
    residual_kg_m_s: float


@dataclass(frozen=True, slots=True)
class VerticalPressureVoidDiagnostics:
    common_pressure_faces_Pa: tuple[float, ...]
    capillary_interfaces: tuple[DetectedCapillaryInterface, ...]
    capillary_internal_recoil_residual_kg_m_s: float
    bottom_gas_piston: BottomGasPistonRemap
    void_remap: VoidRemapResult
    top_liquid: AtmosphericLiquidFlux
    top_gas: AtmosphericGasFlux
    wall: DynamicWallDiagnostics
    three_body_recoil_residual_kg_m_s: float
    liquid_volume_residual_m3: float
    gas_mass_residual_kg: float
    mixture_momentum_z_residual_kg_m_s: float
    momentum_budget: IndependentVerticalMomentumBudget | None = None
    hydrostatic_equilibrium_projection: bool = False


@dataclass(frozen=True, slots=True)
class VerticalComponentEvaluation:
    proposal: ComponentStageProposal
    diagnostics: VerticalPressureVoidDiagnostics | None


class _RejectSignal(Exception):
    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        retryable: bool,
        maximum_dt_s: float | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail
        self.retryable = retryable
        self.maximum_dt_s = maximum_dt_s


def _riser_flux(trial: TNodeTrial) -> GrossNodePortFlux:
    if trial.node_name != "riser_T":
        raise ContractViolation("vertical component requires the riser_T trial")
    for flux in trial.gross_fluxes:
        if flux.key.port_name == "riser_bottom":
            return flux
    raise ContractViolation("riser_T trial has no riser_bottom flux")


class F0VerticalPressureVoidStageComponent:
    """Pure pressure/void implementation of ``VerticalPressureVoidStageComponent``."""

    component_id = "vertical_riser"

    def __init__(
        self,
        *,
        cell_count: int = 160,
        capillary_owner: F0VerticalCapillaryOwner | None = None,
        atmospheric_top: AtmosphericTopState | None = None,
        gas_cfl_safety: float = 0.90,
    ) -> None:
        self._solver = S1VerticalTwoStreamSolver(
            cell_count=cell_count,
            closures=S1VerticalClosures.structural_zero_for_tests(),
        )
        self.capillary_owner = (
            F0VerticalCapillaryOwner()
            if capillary_owner is None
            else capillary_owner
        )
        self.atmospheric_top = (
            AtmosphericTopState() if atmospheric_top is None else atmospheric_top
        )
        safety = _positive("gas_cfl_safety", gas_cfl_safety)
        if safety > 1.0:
            raise ContractViolation("gas_cfl_safety cannot exceed one")
        self.gas_cfl_safety = safety

    @property
    def initial_state(self) -> VerticalState:
        return self._solver.initial_state

    @property
    def production_ready(self) -> bool:
        # Source cut-cell opening, ordered internal fronts, top spill and one
        # finite re-entry parcel have audited stage closures.  A closed gas
        # pocket may still not cross a resolved liquid column, a same-cell
        # double-interface remains unresolved, and the complete coupled
        # network/result gates are still open.  Do not advertise trajectory
        # readiness early.
        return False

    @property
    def source_aligned_trajectory_ready(self) -> bool:
        return False

    @property
    def joint_trial_ready(self) -> bool:
        """The existing six-state riser can consume a pure riser-node trial."""

        return True

    @property
    def persistent_exterior_trial_adapter_ready(self) -> bool:
        """A joint owner can attach one immutable finite plume parcel."""

        return True

    def with_liquid_fallback(
        self, fallback: AtmosphericLiquidFallback | None
    ) -> "F0VerticalPressureVoidStageComponent":
        """Return a pure stage clone with one explicitly owned rim parcel.

        The shared six-state source state and all frozen closures are
        unchanged.  Only the immutable atmospheric liquid parcel differs, so
        nonlinear trial evaluations cannot consume persistent exterior state
        before the whole-network atomic commit.
        """

        return F0VerticalPressureVoidStageComponent(
            cell_count=self._solver.cell_count,
            capillary_owner=self.capillary_owner,
            atmospheric_top=replace(
                self.atmospheric_top, liquid_fallback=fallback
            ),
            gas_cfl_safety=self.gas_cfl_safety,
        )

    def _validate_geometry(self, state: VerticalState, geometry: CoupledGeometry) -> None:
        if state.cell_count != self._solver.cell_count:
            raise ContractViolation("vertical component state and configured grid differ")
        if len(geometry.vertical_dz_m) != state.cell_count:
            raise ContractViolation("vertical component state and coupled grid differ")
        for dz in geometry.vertical_dz_m:
            if not math.isclose(
                dz,
                self._solver.cell_length_m,
                rel_tol=0.0,
                abs_tol=1.0e-13,
            ):
                raise ContractViolation("F0 vertical component requires its pinned uniform grid")
        if not math.isclose(
            geometry.vertical_area_m2,
            self._solver.pipe_area_m2,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ContractViolation("vertical component and coupled pipe areas differ")

    def bottom_trace_pressure_Pa(self, state: VerticalState) -> float:
        """Reconstruct the outgoing bottom trace from owned phase inventory.

        Starting at the bottom, the face-connected liquid height is integrated
        from the six-state areas.  The first connected gas cut cell supplies
        its exact isothermal EOS pressure; this pressure plus hydrostatic head
        gives the liquid-side bottom trace.  The source all-water fallback is
        the already declared initial pressure packet.  No node iterate or
        target result is used.
        """

        self._solver._validate_state(state)
        full = self._solver.pipe_area_m2
        dz = self._solver.cell_length_m
        rt = DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
        connected_height = 0.0
        gas_reference: float | None = None
        for up, down, mass in zip(
            state.Aup, state.Adown, state.Mg, strict=True
        ):
            liquid = min(max(up + down, 0.0), full)
            gas_area = full - liquid
            connected_height += dz * liquid / full
            if gas_area > _VOID_TOLERANCE_M2:
                if mass <= 0.0:
                    raise ContractViolation(
                        "connected riser gas cut cell has no state-owned mass"
                    )
                gas_reference = mass * rt / gas_area
                break
            if liquid <= _VOID_TOLERANCE_M2:
                break
        if gas_reference is None:
            for up, down, mass in zip(
                state.Aup, state.Adown, state.Mg, strict=True
            ):
                gas_area = full - up - down
                if gas_area > _VOID_TOLERANCE_M2 and mass > 0.0:
                    gas_reference = mass * rt / gas_area
                    break
        if gas_reference is None:
            gas_reference = ATMOSPHERIC_PRESSURE_PA
        # The zero-storage T aperture is at the horizontal-pipe crown, one
        # radius above the paper's horizontal centreline datum.  The vertical
        # state retains the published global z=0.5842 m water coordinate, so
        # only the head above that crown belongs in the port trace.  This is
        # the same source-coordinate translation already used by the main and
        # finite supply-branch traces.
        node_connected_height = max(connected_height - 0.5 * PIPE_DIAMETER_M, 0.0)
        pressure = gas_reference + (
            S1_LIQUID_DENSITY_KG_M3
            * S1_GRAVITY_M_S2
            * node_connected_height
        )
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ContractViolation("riser bottom trace pressure is invalid")
        return pressure

    def port_trace(
        self,
        state: VerticalState,
        geometry: CoupledGeometry,
        *,
        interface: CapillaryInterfaceOwnership | None = None,
    ) -> PortTraceState:
        """Expose the physical six-state riser-bottom trace to ``riser_T``."""

        self._validate_geometry(state, geometry)
        self._solver._validate_state(state)
        full = self._solver.pipe_area_m2
        raw_liquid_area = state.Aup[0] + state.Adown[0]
        overfill = max(raw_liquid_area - full, 0.0)
        if overfill > self._solver._parameters.packing_tolerance:
            raise ContractViolation(
                "riser-bottom trace exceeds the pinned FV packing tolerance"
            )
        # The persistent state and its conservation ledger are left untouched.
        # PortTraceState, however, is a face diagnostic and requires exactly
        # complementary non-negative areas.  Project only the roundoff-sized
        # packing band already admitted by the pinned FV state gate; never use
        # this view to hide finite gas inventory or to alter a committed cell.
        liquid_area = min(max(raw_liquid_area, 0.0), full)
        gas_area = max(full - liquid_area, 0.0)
        jump = (
            0.0
            if interface is None
            or interface.pressure_jump_gas_minus_liquid_Pa is None
            else interface.pressure_jump_gas_minus_liquid_Pa
        )
        rt = DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
        if gas_area > _VOID_TOLERANCE_M2:
            if state.Mg[0] <= 0.0:
                raise ContractViolation("riser bottom gas area has no gas mass")
            gas_density = state.Mg[0] / gas_area
            gas_pressure = gas_density * rt
            liquid_pressure = gas_pressure - jump
        else:
            liquid_pressure = self.bottom_trace_pressure_Pa(state)
            gas_pressure = liquid_pressure + jump
            gas_density = gas_pressure / rt
        if min(liquid_pressure, gas_pressure, gas_density) <= 0.0:
            raise ContractViolation("riser port trace has a non-positive phase state")
        liquid_velocity = (
            0.0
            if liquid_area <= _VOID_TOLERANCE_M2
            else (state.Qup[0] - state.Qdown[0]) / liquid_area
        )
        gas_velocity = 0.0 if state.Mg[0] <= 0.0 else state.Jg[0] / state.Mg[0]
        return PortTraceState(
            key=PortKey("riser_T", "riser_bottom"),
            component_id=self.component_id,
            normal_into_node_x=0.0,
            normal_into_node_z=-1.0,
            full_area_m2=full,
            liquid_area_m2=liquid_area,
            gas_area_m2=gas_area,
            liquid_density_kg_m3=S1_LIQUID_DENSITY_KG_M3,
            gas_density_kg_m3=gas_density,
            liquid_absolute_pressure_Pa=liquid_pressure,
            gas_absolute_pressure_Pa=gas_pressure,
            liquid_axial_velocity_m_s=liquid_velocity,
            gas_axial_velocity_m_s=gas_velocity,
            interface_id=None if interface is None else interface.interface_id,
            evidence_status=(
                "pinned_Case1_six_state_riser_bottom_trace__"
                "state_owned_EOS_and_face_connected_liquid_head"
            ),
        )

    def _rejection(
        self,
        trial: TNodeTrial,
        signal: _RejectSignal,
    ) -> VerticalComponentEvaluation:
        maximum = signal.maximum_dt_s
        if maximum is not None:
            maximum = min(maximum, trial.dt_s)
            if maximum <= 0.0:
                maximum = None
        rejection = CapacityReject(
            component_id=self.component_id,
            reason_code=signal.reason_code,
            detail=signal.detail,
            requested_dt_s=trial.dt_s,
            retryable=signal.retryable,
            maximum_admissible_dt_s=maximum,
        )
        proposal = ComponentStageProposal.rejected(
            component_id=self.component_id,
            base_state_token=trial.base_state_token,
            trials=(trial,),
            rejection=rejection,
            evidence_status="S1-1D-F0_vertical_pressure_void_fail_closed",
        )
        return VerticalComponentEvaluation(proposal=proposal, diagnostics=None)

    def _node_packet(self, trial: TNodeTrial):
        flux = _riser_flux(trial)
        fv = self._solver._runtime.fv
        return fv.DirectionalBoundaryFlux(
            upward_rate=flux.liquid_out_of_node_m3_s,
            upward_speed=flux.liquid_out_of_node_speed_m_s,
            downward_rate=flux.liquid_into_node_m3_s,
            downward_speed=flux.liquid_into_node_speed_m_s,
        )

    def _prepare_zero_momentum_bottom_direction(
        self,
        state: VerticalState,
        flux: GrossNodePortFlux,
    ) -> VerticalState:
        """Conservatively relabel a resting connected column at flow reversal.

        The pinned state has persistent directional labels.  At an exact zero
        crossing, however, a column initially stored entirely in ``Aup`` must
        be allowed to become the donor for a newly downward boundary flux (and
        conversely for ``Adown``).  Only zero-momentum, single-label cells in
        the bottom-connected liquid column are relabelled.  Area and signed
        momentum therefore remain exactly unchanged; any moving or two-label
        cell continues to fail closed instead of being silently reconstructed
        from net flow.
        """

        downward_requested = (
            flux.liquid_into_node_m3_s
            > _DIRECTION_REVERSAL_RATE_TOLERANCE_M3_S
            and flux.liquid_out_of_node_m3_s
            <= _DIRECTION_REVERSAL_RATE_TOLERANCE_M3_S
        )
        upward_requested = (
            flux.liquid_out_of_node_m3_s
            > _DIRECTION_REVERSAL_RATE_TOLERANCE_M3_S
            and flux.liquid_into_node_m3_s
            <= _DIRECTION_REVERSAL_RATE_TOLERANCE_M3_S
        )
        if not (downward_requested or upward_requested):
            return state
        up = list(state.Aup)
        qup = list(state.Qup)
        down = list(state.Adown)
        qdown = list(state.Qdown)
        changed = False
        for cell in range(state.cell_count):
            total = up[cell] + down[cell]
            if total <= _VOID_TOLERANCE_M2:
                break
            if qup[cell] > 1.0e-16 or qdown[cell] > 1.0e-16:
                break
            if up[cell] > _VOID_TOLERANCE_M2 and down[cell] > _VOID_TOLERANCE_M2:
                break
            if downward_requested and down[cell] <= _VOID_TOLERANCE_M2:
                down[cell] = up[cell]
                up[cell] = 0.0
                qup[cell] = 0.0
                qdown[cell] = 0.0
                changed = True
            elif upward_requested and up[cell] <= _VOID_TOLERANCE_M2:
                up[cell] = down[cell]
                down[cell] = 0.0
                qup[cell] = 0.0
                qdown[cell] = 0.0
                changed = True
            if total < self._solver.pipe_area_m2 - _VOID_TOLERANCE_M2:
                break
        if not changed:
            return state
        prepared = VerticalState(
            Aup=tuple(up),
            Qup=tuple(qup),
            Adown=tuple(down),
            Qdown=tuple(qdown),
            Mg=state.Mg,
            Jg=state.Jg,
        )
        before_area = tuple(
            a + b for a, b in zip(state.Aup, state.Adown, strict=True)
        )
        after_area = tuple(
            a + b for a, b in zip(prepared.Aup, prepared.Adown, strict=True)
        )
        before_signed_q = tuple(
            a - b for a, b in zip(state.Qup, state.Qdown, strict=True)
        )
        after_signed_q = tuple(
            a - b for a, b in zip(prepared.Qup, prepared.Qdown, strict=True)
        )
        momentum_projection = max(
            (
                abs(before - after)
                for before, after in zip(
                    before_signed_q, after_signed_q, strict=True
                )
            ),
            default=0.0,
        )
        if before_area != after_area or momentum_projection > 1.0e-16:
            raise ContractViolation(
                "zero-momentum bottom direction relabel changed area or finite momentum"
            )
        return prepared

    def _top_liquid_boundary(self, state, *, dt_s: float):
        fv = self._solver._runtime.fv
        area = state.upward_area[-1]
        rate = state.upward_discharge[-1]
        if area <= self._solver._parameters.dry_area_tolerance or rate <= 0.0:
            upward_rate = 0.0
            upward_speed = 0.0
        else:
            upward_rate = rate
            upward_speed = rate / area

        interior_downward_demand = max(-state.downward_discharge[-1], 0.0)
        fallback = self.atmospheric_top.liquid_fallback
        if fallback is not None:
            try:
                available_rate = fallback.admissible_rate_m3_s(dt_s=dt_s)
            except MissingPhysicalClosure as exc:
                raise _RejectSignal(
                    "missing_closure", str(exc), retryable=False
                ) from exc
            # The finite falling parcel is an incident upwind boundary state,
            # not merely a cap on an already-existing interior Qdown.  Its
            # atmospheric pressure, donor area and stored downward velocity
            # can therefore launch re-entry from a zero interior downward
            # trace.  The pinned FV donor/capacity projection remains the
            # fail-closed receiver gate.
            downward_demand = max(
                interior_downward_demand, fallback.downward_rate_m3_s
            )
            downward_rate = min(downward_demand, available_rate)
            downward_speed = (
                fallback.downward_speed_m_s if downward_rate > 0.0 else 0.0
            )
        else:
            downward_demand = interior_downward_demand
            downward_rate = 0.0
            downward_speed = 0.0
        boundary = fv.DirectionalBoundaryFlux(
            upward_rate=upward_rate,
            upward_speed=upward_speed,
            downward_rate=downward_rate,
            downward_speed=downward_speed,
        )
        diagnostics = AtmosphericLiquidFlux(
            outflow_rate_m3_s=upward_rate,
            outflow_speed_m_s=upward_speed,
            reentry_demand_rate_m3_s=downward_demand,
            reentry_rate_m3_s=downward_rate,
            reentry_speed_m_s=downward_speed,
            exterior_available_volume_m3=(
                None if fallback is None else fallback.available_volume_m3
            ),
            stage_consumed_volume_m3=dt_s * downward_rate,
            finite_exterior_inventory=(
                fallback is not None and fallback.finite_stage_inventory_ready
            ),
        )
        return boundary, diagnostics

    def _filled_pressure(self, bottom_pressure_Pa: float) -> tuple[float, ...]:
        dz = self._solver.cell_length_m
        pressure = tuple(
            bottom_pressure_Pa
            - S1_LIQUID_DENSITY_KG_M3
            * S1_GRAVITY_M_S2
            * (cell + 0.5)
            * dz
            for cell in range(self._solver.cell_count)
        )
        if min(pressure) <= 0.0:
            raise _RejectSignal(
                "pressure_bracket",
                "hydrostatic liquid pressure bracket became non-positive",
                retryable=False,
            )
        return pressure

    def _audit_directional_liquid_remap(
        self,
        *,
        state,
        topology_transfer,
        label: str,
    ) -> None:
        """Audit the pinned cell-local directional remap.

        The hash-pinned Case-1 owner defines this operation explicitly as a
        *cell-local* relabel/mixing step: it never reads a neighbouring cell
        and therefore cannot move a liquid parcel across an axial gas gap.
        Coexisting ``Aup``/``Adown`` streams in one resolved annular cut cell
        are the intended two-fluid state, not two axial columns.  We audit the
        pinned operation for per-cell area/momentum conservation and finite
        donor ownership; axial gap traversal remains owned exclusively by the
        ordered bottom-piston remap.
        """

        for cell, (
            up_change,
            down_change,
            up_momentum_change,
            down_momentum_change,
            up,
            down,
        ) in enumerate(
            zip(
                topology_transfer.upward_area_transfer,
                topology_transfer.downward_area_transfer,
                topology_transfer.upward_momentum_transfer,
                topology_transfer.downward_momentum_transfer,
                state.upward_area,
                state.downward_area,
                strict=True,
            )
        ):
            scale_area = max(abs(up), abs(down), self._solver.pipe_area_m2, 1.0)
            if abs(up_change + down_change) > 2.0e-14 * scale_area:
                raise ContractViolation(
                    f"{label} liquid directional remap does not conserve area in cell {cell}"
                )
            momentum_scale = max(
                abs(up_momentum_change), abs(down_momentum_change), 1.0
            )
            if (
                abs(up_momentum_change + down_momentum_change)
                > 2.0e-14 * momentum_scale
            ):
                raise ContractViolation(
                    f"{label} liquid directional remap does not conserve momentum in cell {cell}"
                )
            gas_area = self._solver.pipe_area_m2 - up - down
            if gas_area <= _VOID_TOLERANCE_M2:
                continue
            changed = max(abs(up_change), abs(down_change)) > 1.0e-14
            if not changed:
                continue
            old_up = up - up_change
            old_down = down - down_change
            if max(old_up, old_down) <= _VOID_TOLERANCE_M2:
                raise ContractViolation(
                    f"{label} created liquid inventory from an empty cut cell {cell}"
                )
            donor_tolerance = 2.0e-14 * max(
                abs(old_up), abs(old_down), self._solver.pipe_area_m2, 1.0e-300
            )
            if (
                up_change < 0.0
                and -up_change > old_up + donor_tolerance
            ) or (
                down_change < 0.0
                and -down_change > old_down + donor_tolerance
            ):
                raise ContractViolation(
                    f"{label} directional relabel consumed a liquid label more than once in cell {cell}"
                )

    def _bottom_gas_piston_remap(
        self,
        *,
        liquid_state,
        requested_gas_volume_m3: float,
        dt_s: float,
    ):
        """Open bottom gas capacity with one conservative piston transaction.

        The admitted EOS parcel is used once.  Its unresolved volume removes
        one finite ``Aup`` parcel at cell zero and advances that *same* liquid
        inventory through the ordered axial topology.  A receiving front is
        filled before the next liquid component can become connected.  Thus a
        finite internal gas corridor is never skipped and the pre-existing
        ``Aup``/``Adown`` labels on the far side are not merged or relabelled.
        Once a corridor has zero remaining capacity, the now-contacting liquid
        components form one carrier and the transaction can continue to the
        next front.  Any remainder after the rim is an explicit top spill,
        rather than a deleted parcel or a retry loop with no smaller physical
        time scale.

        No eruption height, duration, or unpublished coefficient enters this
        geometric remap.  Momentum follows the one bottom donor parcel; the
        later FV pressure, gravity, drag and wall operators remain its only
        force owners.
        """

        requested = _nonnegative(
            "same-stage bottom gas EOS volume", requested_gas_volume_m3
        )
        dz = self._solver.cell_length_m
        full = self._solver.pipe_area_m2
        total_area = tuple(
            up + down
            for up, down in zip(
                liquid_state.upward_area,
                liquid_state.downward_area,
                strict=True,
            )
        )
        preexisting = max(full - total_area[0], 0.0) * dz
        displacement = max(requested - preexisting, 0.0)
        if displacement <= 1.0e-18:
            return liquid_state, BottomGasPistonRemap(
                requested_gas_volume_m3=requested,
                preexisting_bottom_void_volume_m3=preexisting,
                displaced_liquid_volume_m3=0.0,
                destination_cell=None,
                liquid_volume_residual_m3=0.0,
                liquid_momentum_residual_kg_m_s=0.0,
                deposited_liquid_volume_m3=0.0,
                top_spill_volume_m3=0.0,
                top_spill_momentum_kg_m_s=0.0,
            )

        if total_area[0] <= _VOID_TOLERANCE_M2:
            raise _RejectSignal(
                "phase_capacity",
                "same-stage bottom gas parcel has no bottom-connected liquid piston",
                retryable=False,
            )
        bottom_up_volume = liquid_state.upward_area[0] * dz
        if displacement > bottom_up_volume * (1.0 + 1.0e-12):
            maximum_dt = (
                dt_s * bottom_up_volume / displacement
                if bottom_up_volume > 0.0
                else None
            )
            raise _RejectSignal(
                "cfl",
                "same-stage gas piston exceeds its one bottom Aup donor parcel",
                retryable=maximum_dt is not None,
                maximum_dt_s=maximum_dt,
            )

        up_area = list(liquid_state.upward_area)
        up_discharge = list(liquid_state.upward_discharge)
        down_area = tuple(liquid_state.downward_area)
        donor_velocity = (
            up_discharge[0] / up_area[0]
            if up_area[0] > self._solver._parameters.dry_area_tolerance
            else 0.0
        )
        initial_total_area = tuple(total_area)
        remaining = displacement
        deposited = 0.0
        top_spill = 0.0
        receiving_cells: list[int] = []
        traversed_gap_cells: list[int] = []

        # Plan additions before removing cell zero so an exactly exhausted
        # donor still advances against the topology that existed at the start
        # of this one atomic transaction.
        while remaining > 1.0e-18:
            work_total = tuple(
                up + down for up, down in zip(up_area, down_area, strict=True)
            )
            connected_end = 0
            while (
                connected_end + 1 < self._solver.cell_count
                and work_total[connected_end + 1] > _VOID_TOLERANCE_M2
            ):
                connected_end += 1

            front = connected_end
            capacity = max(full - work_total[front], 0.0) * dz
            if capacity <= 1.0e-18:
                front += 1
                if front >= self._solver.cell_count:
                    top_spill = remaining
                    remaining = 0.0
                    break
                capacity = max(full - work_total[front], 0.0) * dz
            if front == 0:
                raise _RejectSignal(
                    "missing_closure",
                    "bottom gas and upper liquid interfaces occupy one unresolved axial cut cell",
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                )
            if capacity <= 1.0e-18:
                raise ContractViolation(
                    "bottom piston topology search selected a receiver without capacity"
                )

            amount = min(remaining, capacity)
            up_area[front] += amount / dz
            up_discharge[front] += amount / dz * donor_velocity
            deposited += amount
            remaining -= amount
            if not receiving_cells or receiving_cells[-1] != front:
                receiving_cells.append(front)
            if (
                initial_total_area[front] <= _VOID_TOLERANCE_M2
                and front not in traversed_gap_cells
            ):
                traversed_gap_cells.append(front)

        up_area[0] -= displacement / dz
        up_discharge[0] -= displacement / dz * donor_velocity
        if min(up_area) < -self._solver._parameters.packing_tolerance:
            raise ContractViolation("bottom gas piston created negative upward area")
        up_area = [max(value, 0.0) for value in up_area]
        if any(
            area <= self._solver._parameters.dry_area_tolerance
            and abs(discharge) > 1.0e-14
            for area, discharge in zip(up_area, up_discharge, strict=True)
        ):
            raise _RejectSignal(
                "cfl",
                "bottom gas piston exhausted a liquid donor while retaining momentum",
                retryable=True,
                maximum_dt_s=0.5 * dt_s,
            )
        remapped = self._solver._runtime.fv.VerticalTwoStreamState.from_iterables(
            upward_area=up_area,
            upward_discharge=up_discharge,
            downward_area=down_area,
            downward_discharge=liquid_state.downward_discharge,
        )
        volume_before = dz * math.fsum(total_area)
        volume_after = dz * math.fsum(
            up + down
            for up, down in zip(
                remapped.upward_area,
                remapped.downward_area,
                strict=True,
            )
        )
        momentum_before = S1_LIQUID_DENSITY_KG_M3 * dz * math.fsum(
            up + down
            for up, down in zip(
                liquid_state.upward_discharge,
                liquid_state.downward_discharge,
                strict=True,
            )
        )
        momentum_after = S1_LIQUID_DENSITY_KG_M3 * dz * math.fsum(
            up + down
            for up, down in zip(
                remapped.upward_discharge,
                remapped.downward_discharge,
                strict=True,
            )
        )
        top_spill_momentum = (
            S1_LIQUID_DENSITY_KG_M3 * top_spill * donor_velocity
        )
        volume_residual = volume_after - volume_before + top_spill
        momentum_residual = (
            momentum_after - momentum_before + top_spill_momentum
        )
        parcel_residual = displacement - deposited - top_spill
        if (
            abs(volume_residual) > _MATERIAL_LEDGER_TOLERANCE
            or abs(momentum_residual) > _MOMENTUM_LEDGER_TOLERANCE
            or abs(parcel_residual) > _MATERIAL_LEDGER_TOLERANCE
        ):
            raise ContractViolation("bottom gas piston remap ledger does not close")
        return remapped, BottomGasPistonRemap(
            requested_gas_volume_m3=requested,
            preexisting_bottom_void_volume_m3=preexisting,
            displaced_liquid_volume_m3=displacement,
            destination_cell=(receiving_cells[-1] if receiving_cells else None),
            liquid_volume_residual_m3=volume_residual,
            liquid_momentum_residual_kg_m_s=momentum_residual,
            deposited_liquid_volume_m3=deposited,
            top_spill_volume_m3=top_spill,
            top_spill_momentum_kg_m_s=top_spill_momentum,
            receiving_cells=tuple(receiving_cells),
            traversed_gap_cells=tuple(traversed_gap_cells),
        )

    def _atmospheric_gas_flux(
        self,
        *,
        gas_area_m2: float,
        gas_mass_cell_kg: float,
        gas_momentum_cell_kg_m_s: float,
    ) -> AtmosphericGasFlux:
        if gas_area_m2 <= _VOID_TOLERANCE_M2:
            return AtmosphericGasFlux(0.0, 0.0, 0.0, 0.0, 0.0, self.atmospheric_top.absolute_pressure_Pa)
        dz = self._solver.cell_length_m
        if gas_mass_cell_kg <= _MASS_TOLERANCE_KG:
            raise _RejectSignal(
                "void_mass_pairing",
                "finite top gas void has no interior gas mass for the atmospheric Riemann solve",
                retryable=False,
            )
        rho_i = gas_mass_cell_kg / (gas_area_m2 * dz)
        u_i = gas_momentum_cell_kg_m_s / gas_mass_cell_kg
        rt = DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
        p_i = rho_i * rt
        rho_a = self.atmospheric_top.gas_density_kg_m3
        c = self.atmospheric_top.gas_sound_speed_m_s
        z_i = rho_i * c
        z_a = rho_a * c
        u_a = self.atmospheric_top.gas_axial_velocity_m_s
        contact_velocity = (
            z_i * u_i + z_a * u_a + p_i - self.atmospheric_top.absolute_pressure_Pa
        ) / (z_i + z_a)
        contact_pressure = (
            z_a * p_i
            + z_i * self.atmospheric_top.absolute_pressure_Pa
            + z_i * z_a * (u_i - u_a)
        ) / (z_i + z_a)
        if contact_pressure <= 0.0 or not math.isfinite(contact_pressure):
            raise _RejectSignal(
                "pressure_bracket",
                "atmospheric gas Riemann contact pressure is non-positive",
                retryable=True,
                maximum_dt_s=0.5,
            )
        if contact_velocity >= 0.0:
            outflow = rho_i * gas_area_m2 * contact_velocity
            return AtmosphericGasFlux(
                outflow_kg_s=outflow,
                inflow_kg_s=0.0,
                outflow_speed_m_s=contact_velocity,
                inflow_speed_m_s=0.0,
                contact_velocity_m_s=contact_velocity,
                contact_pressure_Pa=contact_pressure,
            )
        speed = -contact_velocity
        inflow = rho_a * gas_area_m2 * speed
        return AtmosphericGasFlux(
            outflow_kg_s=0.0,
            inflow_kg_s=inflow,
            outflow_speed_m_s=0.0,
            inflow_speed_m_s=speed,
            contact_velocity_m_s=contact_velocity,
            contact_pressure_Pa=contact_pressure,
        )

    def _transport_gas(
        self,
        *,
        liquid_state,
        remap: VoidRemapResult,
        bottom_flux: GrossNodePortFlux,
        common_pressure_faces_Pa: tuple[float, ...],
        dt_s: float,
    ):
        n = self._solver.cell_count
        dz = self._solver.cell_length_m
        # FV packing admits a tiny roundoff band around a full-water cell.
        # It is not a resolved gas control volume and therefore must not own a
        # pressure-force impulse.  Canonicalise only this derived transport
        # geometry; the immutable liquid areas and their conservation ledger
        # remain untouched.  Without this view, a 1e-15 m2 subtraction
        # residual creates O(1e-25) gas momentum in an exactly massless cell,
        # which the pinned physical-drag state correctly rejects.
        gas_area = tuple(
            0.0 if raw <= _VOID_TOLERANCE_M2 else raw
            for up, down in zip(
                liquid_state.upward_area,
                liquid_state.downward_area,
                strict=True,
            )
            for raw in (self._solver.pipe_area_m2 - up - down,)
        )
        mass = list(remap.gas_mass_cell_kg)
        momentum = list(remap.gas_momentum_cell_kg_m_s)
        velocity = [
            0.0 if cell_mass <= _MASS_TOLERANCE_KG else momentum[cell] / cell_mass
            for cell, cell_mass in enumerate(mass)
        ]
        top = self._atmospheric_gas_flux(
            gas_area_m2=gas_area[-1],
            gas_mass_cell_kg=mass[-1],
            gas_momentum_cell_kg_m_s=momentum[-1],
        )

        upward = [0.0] * (n + 1)
        downward = [0.0] * (n + 1)
        up_speed = [0.0] * (n + 1)
        down_speed = [0.0] * (n + 1)
        # The bottom inflow parcel was admitted during the atomic void remap.
        # It must not be applied again here.  Its momentum can nevertheless
        # drive interior donor transport during this same stage through the
        # remapped cell inventory above.
        upward[0] = 0.0
        up_speed[0] = 0.0
        downward[0] = bottom_flux.gas_into_node_kg_s
        down_speed[0] = bottom_flux.gas_into_node_speed_m_s
        for face in range(1, n):
            lower = face - 1
            upper = face
            if momentum[lower] > 0.0 and gas_area[upper] > _VOID_TOLERANCE_M2:
                upward[face] = momentum[lower] / dz
                up_speed[face] = velocity[lower]
            if momentum[upper] < 0.0 and gas_area[lower] > _VOID_TOLERANCE_M2:
                downward[face] = -momentum[upper] / dz
                down_speed[face] = -velocity[upper]
        upward[n] = top.outflow_kg_s
        up_speed[n] = top.outflow_speed_m_s
        downward[n] = top.inflow_kg_s
        down_speed[n] = top.inflow_speed_m_s

        maximum_dt = dt_s
        for cell in range(n):
            outgoing = upward[cell + 1] + downward[cell]
            if outgoing <= 0.0:
                continue
            allowed = self.gas_cfl_safety * mass[cell] / outgoing
            maximum_dt = min(maximum_dt, allowed)
        if maximum_dt < dt_s * (1.0 - 1.0e-13):
            raise _RejectSignal(
                "cfl",
                "gas donor transport would exhaust a finite void cell",
                retryable=True,
                maximum_dt_s=maximum_dt,
            )

        mass_flux = [up - down for up, down in zip(upward, downward, strict=True)]
        momentum_flux = [
            up * us + down * ds
            for up, us, down, ds in zip(
                upward, up_speed, downward, down_speed, strict=True
            )
        ]
        final_mass: list[float] = []
        final_momentum: list[float] = []
        pressure_impulse = 0.0
        gravity_impulse = 0.0
        for cell in range(n):
            new_mass = mass[cell] + dt_s * (
                mass_flux[cell] - mass_flux[cell + 1]
            )
            if gas_area[cell] > _VOID_TOLERANCE_M2 and new_mass <= _MASS_TOLERANCE_KG:
                raise _RejectSignal(
                    "void_mass_pairing",
                    f"gas transport left massless void in riser cell {cell}",
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                )
            p_impulse = dt_s * gas_area[cell] * (
                common_pressure_faces_Pa[cell]
                - common_pressure_faces_Pa[cell + 1]
            )
            g_impulse = -dt_s * mass[cell] * S1_GRAVITY_M_S2
            new_momentum = (
                momentum[cell]
                + dt_s * (momentum_flux[cell] - momentum_flux[cell + 1])
                + p_impulse
                + g_impulse
            )
            final_mass.append(max(new_mass, 0.0))
            final_momentum.append(new_momentum)
            pressure_impulse += p_impulse
            gravity_impulse += g_impulse

        mass_residual = (
            math.fsum(final_mass)
            - math.fsum(mass)
            - dt_s * (mass_flux[0] - mass_flux[-1])
        )
        momentum_residual = (
            math.fsum(final_momentum)
            - math.fsum(momentum)
            - dt_s * (momentum_flux[0] - momentum_flux[-1])
            - pressure_impulse
            - gravity_impulse
        )
        if (
            abs(mass_residual) > _MATERIAL_LEDGER_TOLERANCE
            or abs(momentum_residual) > _MOMENTUM_LEDGER_TOLERANCE
        ):
            raise ContractViolation("F0 vertical gas transport ledger does not close")
        return GasTransportResult(
            gas_mass_cell_kg=tuple(final_mass),
            gas_momentum_cell_kg_m_s=tuple(final_momentum),
            top=top,
            mass_residual_kg=mass_residual,
            momentum_residual_kg_m_s=momentum_residual,
            bottom_advective_impulse_kg_m_s=(
                remap.boundary_source_momentum_kg_m_s
                + dt_s * momentum_flux[0]
            ),
            top_advective_impulse_kg_m_s=dt_s * momentum_flux[-1],
            pressure_impulse_kg_m_s=pressure_impulse,
            gravity_impulse_kg_m_s=gravity_impulse,
        )

    def _apply_capillary_recoil(
        self,
        *,
        liquid_state,
        gas_mass_cell_kg: tuple[float, ...],
        gas_momentum_cell_kg_m_s: tuple[float, ...],
        dt_s: float,
    ):
        try:
            interfaces = self.capillary_owner.detect(
                liquid_state,
                gas_mass_cell_kg=gas_mass_cell_kg,
                gas_momentum_cell_kg_m_s=gas_momentum_cell_kg_m_s,
                full_area_m2=self._solver.pipe_area_m2,
                diameter_m=PIPE_DIAMETER_M,
            )
        except MissingPhysicalClosure as exc:
            raise _RejectSignal(
                "missing_closure", str(exc), retryable=False
            ) from exc
        if not interfaces:
            return liquid_state, gas_momentum_cell_kg_m_s, interfaces, 0.0
        up_q = list(liquid_state.upward_discharge)
        down_q = list(liquid_state.downward_discharge)
        gas_momentum = list(gas_momentum_cell_kg_m_s)
        total_liquid_impulse = 0.0
        total_gas_impulse = 0.0
        dz = self._solver.cell_length_m
        for interface in interfaces:
            face = interface.face_index
            gas_cell = face if interface.gas_is_above else face - 1
            liquid_cell = face - 1 if interface.gas_is_above else face
            jump = interface.record.pressure_jump_gas_minus_liquid_Pa
            assert jump is not None
            liquid_area = (
                liquid_state.upward_area[liquid_cell]
                + liquid_state.downward_area[liquid_cell]
            )
            if liquid_area <= self._solver._parameters.dry_area_tolerance:
                raise _RejectSignal(
                    "phase_capacity",
                    "capillary interface has no liquid receiving inventory",
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                )
            if gas_mass_cell_kg[gas_cell] <= _MASS_TOLERANCE_KG:
                raise _RejectSignal(
                    "void_mass_pairing",
                    "capillary interface has no gas recoil inventory",
                    retryable=False,
                )
            liquid_impulse = -jump * self._solver.pipe_area_m2 * dt_s
            discharge_change = liquid_impulse / (
                S1_LIQUID_DENSITY_KG_M3 * dz
            )
            up_fraction = liquid_state.upward_area[liquid_cell] / liquid_area
            down_fraction = liquid_state.downward_area[liquid_cell] / liquid_area
            up_q[liquid_cell] += discharge_change * up_fraction
            down_q[liquid_cell] += discharge_change * down_fraction
            gas_momentum[gas_cell] -= liquid_impulse
            total_liquid_impulse += liquid_impulse
            total_gas_impulse -= liquid_impulse

        topology = self._solver._runtime.fv.conservative_directional_topology_transfer(
            upward_area=liquid_state.upward_area,
            upward_discharge=up_q,
            downward_area=liquid_state.downward_area,
            downward_discharge=down_q,
            velocity_tolerance=1.0e-12,
        )
        residual = total_liquid_impulse + total_gas_impulse
        return topology.state, tuple(gas_momentum), interfaces, residual

    def _apply_dynamic_wall_friction(
        self,
        *,
        liquid_state,
        gas_mass_cell_kg: tuple[float, ...],
        gas_momentum_cell_kg_m_s: tuple[float, ...],
        dt_s: float,
    ):
        circumference = math.pi * PIPE_DIAMETER_M
        full = self._solver.pipe_area_m2
        dz = self._solver.cell_length_m
        up_q = list(liquid_state.upward_discharge)
        down_q = list(liquid_state.downward_discharge)
        gas_momentum = list(gas_momentum_cell_kg_m_s)
        f_up: list[float] = []
        f_down: list[float] = []
        f_gas: list[float] = []
        dh_up: list[float] = []
        dh_down: list[float] = []
        dh_gas: list[float] = []
        impulse = 0.0

        for cell, (a_up, a_down) in enumerate(
            zip(liquid_state.upward_area, liquid_state.downward_area, strict=True)
        ):
            a_gas = max(full - a_up - a_down, 0.0)
            # Coaxial F0 topology: downward liquid is the wall film.  With no
            # film, gas touches the wall if present; otherwise the upward
            # liquid occupies the full wall perimeter.
            p_up = circumference if a_up > 0.0 and a_down <= _VOID_TOLERANCE_M2 and a_gas <= _VOID_TOLERANCE_M2 else 0.0
            p_down = circumference if a_down > 0.0 else 0.0
            p_gas = circumference if a_gas > _VOID_TOLERANCE_M2 and a_down <= _VOID_TOLERANCE_M2 else 0.0

            local_dh_up = 0.0 if p_up == 0.0 else 4.0 * a_up / p_up
            local_dh_down = 0.0 if p_down == 0.0 else 4.0 * a_down / p_down
            local_dh_gas = 0.0 if p_gas == 0.0 else 4.0 * a_gas / p_gas
            dh_up.append(local_dh_up)
            dh_down.append(local_dh_down)
            dh_gas.append(local_dh_gas)

            if local_dh_up > 0.0 and a_up > 0.0:
                velocity = up_q[cell] / a_up
                re = (
                    S1_LIQUID_DENSITY_KG_M3
                    * abs(velocity)
                    * local_dh_up
                    / S1_LIQUID_VISCOSITY_PA_S
                )
                factor = f0_smooth_pipe_darcy_factor(re)
                relaxed = velocity / (
                    1.0 + factor * abs(velocity) * dt_s / (2.0 * local_dh_up)
                )
                old = up_q[cell]
                up_q[cell] = a_up * relaxed
                impulse += (
                    S1_LIQUID_DENSITY_KG_M3 * (up_q[cell] - old) * dz
                )
            else:
                factor = 0.0
            f_up.append(factor)

            if local_dh_down > 0.0 and a_down > 0.0:
                velocity = down_q[cell] / a_down
                re = (
                    S1_LIQUID_DENSITY_KG_M3
                    * abs(velocity)
                    * local_dh_down
                    / S1_LIQUID_VISCOSITY_PA_S
                )
                factor = f0_smooth_pipe_darcy_factor(re)
                relaxed = velocity / (
                    1.0 + factor * abs(velocity) * dt_s / (2.0 * local_dh_down)
                )
                old = down_q[cell]
                down_q[cell] = a_down * relaxed
                impulse += (
                    S1_LIQUID_DENSITY_KG_M3 * (down_q[cell] - old) * dz
                )
            else:
                factor = 0.0
            f_down.append(factor)

            if local_dh_gas > 0.0 and gas_mass_cell_kg[cell] > _MASS_TOLERANCE_KG:
                velocity = gas_momentum[cell] / gas_mass_cell_kg[cell]
                rho_g = gas_mass_cell_kg[cell] / (a_gas * dz)
                re = (
                    rho_g
                    * abs(velocity)
                    * local_dh_gas
                    / S1_GAS_VISCOSITY_PA_S
                )
                factor = f0_smooth_pipe_darcy_factor(re)
                relaxed = velocity / (
                    1.0 + factor * abs(velocity) * dt_s / (2.0 * local_dh_gas)
                )
                old = gas_momentum[cell]
                gas_momentum[cell] = gas_mass_cell_kg[cell] * relaxed
                impulse += gas_momentum[cell] - old
            else:
                factor = 0.0
            f_gas.append(factor)

        state = self._solver._runtime.fv.VerticalTwoStreamState.from_iterables(
            upward_area=liquid_state.upward_area,
            upward_discharge=up_q,
            downward_area=liquid_state.downward_area,
            downward_discharge=down_q,
        )
        diagnostics = DynamicWallDiagnostics(
            liquid_up_darcy_factor=tuple(f_up),
            liquid_down_darcy_factor=tuple(f_down),
            gas_darcy_factor=tuple(f_gas),
            liquid_up_hydraulic_diameter_m=tuple(dh_up),
            liquid_down_hydraulic_diameter_m=tuple(dh_down),
            gas_hydraulic_diameter_m=tuple(dh_gas),
            wall_impulse_kg_m_s=impulse,
        )
        return state, tuple(gas_momentum), diagnostics

    def evaluate_joint_stage(
        self,
        state: VerticalState,
        geometry: CoupledGeometry,
        *,
        riser_node_trial: TNodeTrial,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> VerticalComponentEvaluation:
        self._validate_geometry(state, geometry)
        trial = riser_node_trial
        if trial.physical_stage != physical_stage:
            raise ContractViolation("vertical trial changed the requested physical stage")
        if not math.isclose(trial.dt_s, dt_s, rel_tol=0.0, abs_tol=0.0):
            raise ContractViolation("vertical trial dt differs from component dt")
        flux = _riser_flux(trial)
        source_state = state
        state = self._prepare_zero_momentum_bottom_direction(state, flux)

        try:
            self._solver._validate_state(state)
            component = _component_state(
                self._solver._runtime,
                state,
                dry_discharge_tolerance_m3_s=(
                    self._solver._parameters.dry_area_tolerance
                ),
            )
            bottom = self._node_packet(trial)
            top, top_liquid = self._top_liquid_boundary(
                component, dt_s=dt_s
            )
            filled_pressure = self._filled_pressure(trial.common_absolute_pressure_Pa)
            old_mass = tuple(value * self._solver.cell_length_m for value in state.Mg)
            old_momentum = tuple(value * self._solver.cell_length_m for value in state.Jg)
            pressure_before = self._solver._runtime.closures.adapt_gas_void_and_pressure_faces(
                component,
                self._solver._parameters,
                gas_mass=old_mass,
                gas_momentum=old_momentum,
                gas=self._solver._gas_parameters,
                bottom_pressure=trial.common_absolute_pressure_Pa,
                liquid_filled_cell_pressure=filled_pressure,
            )
            transport = self._solver._runtime.fv.advance_vertical_two_stream_fv(
                component,
                self._solver._parameters,
                dt=dt_s,
                pressure_faces=pressure_before.common_pressure_faces,
                boundaries=self._solver._runtime.fv.VerticalTwoStreamBoundaries(
                    bottom=bottom,
                    top=top,
                ),
            )
            self._audit_directional_liquid_remap(
                state=transport.state,
                topology_transfer=transport.topology_transfer,
                label="liquid FV transport",
            )

            actual_boundaries = (
                transport.upward_area_flux[0],
                -transport.downward_area_flux[0],
                transport.upward_area_flux[-1],
                -transport.downward_area_flux[-1],
            )
            requested_boundaries = (
                bottom.upward_rate,
                bottom.downward_rate,
                top.upward_rate,
                top.downward_rate,
            )
            if any(
                not math.isclose(actual, requested, rel_tol=1.0e-11, abs_tol=1.0e-16)
                for actual, requested in zip(
                    actual_boundaries, requested_boundaries, strict=True
                )
            ):
                raise _RejectSignal(
                    "phase_capacity",
                    "liquid donor/capacity projection changed an explicit boundary flux",
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                )

            liquid_state_after_fv = transport.state
            admitted_bottom_gas_mass = dt_s * flux.gas_out_of_node_kg_s
            admitted_bottom_gas_momentum = (
                admitted_bottom_gas_mass * flux.gas_out_of_node_speed_m_s
            )
            node_gas_density = trial.common_absolute_pressure_Pa / (
                DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K
            )
            requested_bottom_gas_volume = (
                admitted_bottom_gas_mass / node_gas_density
            )
            piston_state, piston = self._bottom_gas_piston_remap(
                liquid_state=transport.state,
                requested_gas_volume_m3=requested_bottom_gas_volume,
                dt_s=dt_s,
            )
            if piston_state is not transport.state:
                transport = replace(transport, state=piston_state)
            piston_spill_rate = piston.top_spill_volume_m3 / dt_s
            if piston_spill_rate > 0.0:
                piston_spill_speed = (
                    piston.top_spill_momentum_kg_m_s
                    / (
                        S1_LIQUID_DENSITY_KG_M3
                        * piston.top_spill_volume_m3
                    )
                )
                combined_outflow = (
                    top_liquid.outflow_rate_m3_s + piston_spill_rate
                )
                combined_speed = (
                    top_liquid.outflow_rate_m3_s
                    * top_liquid.outflow_speed_m_s
                    + piston_spill_rate * piston_spill_speed
                ) / combined_outflow
                top_liquid = replace(
                    top_liquid,
                    outflow_rate_m3_s=combined_outflow,
                    outflow_speed_m_s=combined_speed,
                )

            old_void = tuple(
                max(self._solver.pipe_area_m2 - up - down, 0.0)
                for up, down in zip(
                    component.upward_area, component.downward_area, strict=True
                )
            )
            new_void = tuple(
                max(self._solver.pipe_area_m2 - up - down, 0.0)
                for up, down in zip(
                    transport.state.upward_area,
                    transport.state.downward_area,
                    strict=True,
                )
            )
            remap = conservative_void_remap(
                old_void_area_m2=old_void,
                new_void_area_m2=new_void,
                gas_mass_cell_kg=old_mass,
                gas_momentum_cell_kg_m_s=old_momentum,
                cell_length_m=self._solver.cell_length_m,
                boundary_source_cell=(
                    0 if admitted_bottom_gas_mass > 0.0 else None
                ),
                boundary_source_mass_kg=admitted_bottom_gas_mass,
                boundary_source_momentum_kg_m_s=admitted_bottom_gas_momentum,
            )
            pressure_after_liquid = self._solver._runtime.closures.adapt_gas_void_and_pressure_faces(
                transport.state,
                self._solver._parameters,
                gas_mass=remap.gas_mass_cell_kg,
                gas_momentum=remap.gas_momentum_cell_kg_m_s,
                gas=self._solver._gas_parameters,
                bottom_pressure=trial.common_absolute_pressure_Pa,
                liquid_filled_cell_pressure=filled_pressure,
            )
            gas_transport = self._transport_gas(
                liquid_state=transport.state,
                remap=remap,
                bottom_flux=flux,
                common_pressure_faces_Pa=pressure_after_liquid.common_pressure_faces,
                dt_s=dt_s,
            )
            gas_mass = gas_transport.gas_mass_cell_kg
            gas_momentum = gas_transport.gas_momentum_cell_kg_m_s
            top_gas = gas_transport.top

            (
                capillary_state,
                capillary_gas_momentum,
                capillary_interfaces,
                capillary_residual,
            ) = self._apply_capillary_recoil(
                liquid_state=transport.state,
                gas_mass_cell_kg=gas_mass,
                gas_momentum_cell_kg_m_s=gas_momentum,
                dt_s=dt_s,
            )
            pressure_for_drag = self._solver._runtime.closures.adapt_gas_void_and_pressure_faces(
                capillary_state,
                self._solver._parameters,
                gas_mass=gas_mass,
                gas_momentum=capillary_gas_momentum,
                gas=self._solver._gas_parameters,
                bottom_pressure=trial.common_absolute_pressure_Pa,
                liquid_filled_cell_pressure=filled_pressure,
            )
            drag = self._solver._runtime.fv.implicit_physical_three_body_drag_exchange(
                capillary_state,
                self._solver._parameters,
                pressure_for_drag.physical_drag_state,
                dt=dt_s,
            )
            self._audit_directional_liquid_remap(
                state=drag.state,
                topology_transfer=drag.topology_transfer,
                label="three-body recoil",
            )
            wall_state, wall_gas_momentum, wall = self._apply_dynamic_wall_friction(
                liquid_state=drag.state,
                gas_mass_cell_kg=gas_mass,
                gas_momentum_cell_kg_m_s=drag.gas_momentum,
                dt_s=dt_s,
            )
            final = VerticalState(
                Aup=wall_state.upward_area,
                Qup=wall_state.upward_discharge,
                Adown=wall_state.downward_area,
                Qdown=tuple(-value for value in wall_state.downward_discharge),
                Mg=tuple(value / self._solver.cell_length_m for value in gas_mass),
                Jg=tuple(
                    value / self._solver.cell_length_m
                    for value in wall_gas_momentum
                ),
            )
            self._solver._validate_state(final)
            dz = self._solver.cell_length_m
            initial_liquid = math.fsum(
                (up + down) * dz
                for up, down in zip(state.Aup, state.Adown, strict=True)
            )
            final_liquid = math.fsum(
                (up + down) * dz
                for up, down in zip(final.Aup, final.Adown, strict=True)
            )
            bottom_liquid_net = (
                flux.liquid_out_of_node_m3_s - flux.liquid_into_node_m3_s
            )
            top_liquid_out = actual_boundaries[2] + piston_spill_rate
            top_liquid_in = actual_boundaries[3]
            liquid_residual = (
                final_liquid
                - initial_liquid
                - dt_s
                * (bottom_liquid_net + top_liquid_in - top_liquid_out)
            )

            initial_gas = math.fsum(old_mass)
            final_gas = math.fsum(gas_mass)
            bottom_gas_net = flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
            gas_residual = (
                final_gas
                - initial_gas
                - dt_s
                * (
                    bottom_gas_net
                    + top_gas.inflow_kg_s
                    - top_gas.outflow_kg_s
                )
            )
            if (
                abs(liquid_residual) > _MATERIAL_LEDGER_TOLERANCE
                or abs(gas_residual) > _MATERIAL_LEDGER_TOLERANCE
            ):
                raise ContractViolation(
                    "vertical material ledger does not close against node and rim ports"
                )

            rho = S1_LIQUID_DENSITY_KG_M3
            initial_pz = math.fsum(
                (rho * (up - down) + gas_j) * dz
                for up, down, gas_j in zip(
                    state.Qup, state.Qdown, state.Jg, strict=True
                )
            )
            final_pz = math.fsum(
                (rho * (up - down) + gas_j) * dz
                for up, down, gas_j in zip(
                    final.Qup, final.Qdown, final.Jg, strict=True
                )
            )

            liquid_ledger = transport.ledger
            liquid_bottom_advective_impulse = rho * dt_s * (
                transport.upward_momentum_flux[0]
                + transport.downward_momentum_flux[0]
            )
            liquid_top_advective_impulse = rho * dt_s * (
                transport.upward_momentum_flux[-1]
                + transport.downward_momentum_flux[-1]
            ) + piston.top_spill_momentum_kg_m_s
            bottom_advective_impulse = (
                liquid_bottom_advective_impulse
                + gas_transport.bottom_advective_impulse_kg_m_s
            )
            top_advective_impulse = (
                liquid_top_advective_impulse
                + gas_transport.top_advective_impulse_kg_m_s
            )

            liquid_area_after_transport = tuple(
                up + down
                for up, down in zip(
                    liquid_state_after_fv.upward_area,
                    liquid_state_after_fv.downward_area,
                    strict=True,
                )
            )
            liquid_pressure_impulse = dt_s * math.fsum(
                area
                * (
                    pressure_before.common_pressure_faces[cell]
                    - pressure_before.common_pressure_faces[cell + 1]
                )
                for cell, area in enumerate(liquid_area_after_transport)
            )
            liquid_gravity_impulse = -dt_s * rho * S1_GRAVITY_M_S2 * math.fsum(
                area * dz for area in liquid_area_after_transport
            )
            liquid_pressure_gravity_from_fv = rho * (
                liquid_ledger.pressure_gravity_impulse
            )
            if not math.isclose(
                liquid_pressure_impulse + liquid_gravity_impulse,
                liquid_pressure_gravity_from_fv,
                rel_tol=2.0e-12,
                abs_tol=_MOMENTUM_LEDGER_TOLERANCE,
            ):
                raise ContractViolation(
                    "pinned liquid FV does not expose a reproducible pressure/gravity budget"
                )
            liquid_internal_exchange = rho * (
                liquid_ledger.interstream_upward_impulse
                + liquid_ledger.interstream_downward_impulse
                + liquid_ledger.gas_on_liquid_kinematic_impulse
            )
            liquid_fv_residual = rho * liquid_ledger.liquid_momentum_residual
            if max(
                abs(liquid_internal_exchange), abs(liquid_fv_residual)
            ) > _MOMENTUM_LEDGER_TOLERANCE:
                raise ContractViolation(
                    "pinned liquid FV internal momentum budget does not close"
                )
            if max(
                abs(capillary_residual),
                abs(drag.total_momentum_residual),
                abs(gas_transport.momentum_residual_kg_m_s),
            ) > _MOMENTUM_LEDGER_TOLERANCE:
                raise ContractViolation(
                    "vertical internal recoil or gas-transport momentum budget does not close"
                )

            fv_wall_impulse = rho * liquid_ledger.wall_impulse
            wall_impulse = fv_wall_impulse + wall.wall_impulse_kg_m_s
            bottom_pressure_impulse = (
                dt_s
                * trial.common_absolute_pressure_Pa
                * self._solver.pipe_area_m2
            )
            expected_advective_to_node_N = -bottom_advective_impulse / dt_s
            expected_pressure_to_node_N = -(
                trial.common_absolute_pressure_Pa * self._solver.pipe_area_m2
            )
            if not math.isclose(
                flux.advective_momentum_to_node_z_N,
                expected_advective_to_node_N,
                rel_tol=2.0e-11,
                abs_tol=2.0e-10,
            ):
                raise _RejectSignal(
                    "missing_closure",
                    "riser-bottom port lacks the independently computed advective momentum exchange",
                    retryable=False,
                )
            if not math.isclose(
                flux.pressure_traction_to_node_z_N,
                expected_pressure_to_node_N,
                rel_tol=2.0e-11,
                abs_tol=2.0e-10,
            ):
                raise _RejectSignal(
                    "missing_closure",
                    "riser-bottom port lacks the independently computed common-pressure traction",
                    retryable=False,
                )

            capillary_external_impulse = 0.0
            pressure_impulse = (
                liquid_pressure_impulse
                + gas_transport.pressure_impulse_kg_m_s
            )
            top_pressure_traction_impulse = (
                -dt_s
                * self.atmospheric_top.absolute_pressure_Pa
                * self._solver.pipe_area_m2
            )
            discrete_pressure_traction_residual = (
                pressure_impulse
                - bottom_pressure_impulse
                - top_pressure_traction_impulse
            )
            if (
                abs(discrete_pressure_traction_residual)
                > _MOMENTUM_LEDGER_TOLERANCE
            ):
                raise _RejectSignal(
                    "missing_closure",
                    "discrete common-pressure work cannot be decomposed into explicit bottom/top tractions",
                    retryable=False,
                )
            gravity_impulse = (
                liquid_gravity_impulse
                + gas_transport.gravity_impulse_kg_m_s
            )
            # Remove the bottom pressure traction carried by the internal
            # riser-T port.  Everything remaining here is an independently
            # known exterior/body/wall force; no final-state difference is
            # used to manufacture this value.
            external_force_impulse = (
                top_pressure_traction_impulse
                + discrete_pressure_traction_residual
                + gravity_impulse
                + wall_impulse
                + capillary_external_impulse
            )
            external_force_z_N = external_force_impulse / dt_s
            top_advective_out_N = top_advective_impulse / dt_s
            external = BoundaryExchange(
                liquid_inflow_m3_s=top_liquid_in,
                liquid_outflow_m3_s=top_liquid_out,
                gas_inflow_kg_s=top_gas.inflow_kg_s,
                gas_outflow_kg_s=top_gas.outflow_kg_s,
                momentum_z_out_N=top_advective_out_N,
                external_force_z_N=external_force_z_N,
            )
            bottom_momentum_to_component_N = -flux.mixture_momentum_to_node_z_N
            predicted_impulse = dt_s * (
                bottom_momentum_to_component_N
                + external.mixture_momentum_z_net_rate
            )
            actual_impulse = final_pz - initial_pz
            momentum_residual = (
                actual_impulse - predicted_impulse
            )
            if abs(momentum_residual) > _MOMENTUM_LEDGER_TOLERANCE:
                raise ContractViolation(
                    "vertical Pz conservation gate detected momentum not present in the independent physical budget"
                )

            momentum_budget = IndependentVerticalMomentumBudget(
                bottom_node_advective_impulse_kg_m_s=bottom_advective_impulse,
                bottom_node_pressure_impulse_kg_m_s=bottom_pressure_impulse,
                top_advective_impulse_kg_m_s=top_advective_impulse,
                top_pressure_traction_impulse_kg_m_s=(
                    top_pressure_traction_impulse
                ),
                liquid_pressure_impulse_kg_m_s=liquid_pressure_impulse,
                gas_pressure_impulse_kg_m_s=(
                    gas_transport.pressure_impulse_kg_m_s
                ),
                discrete_pressure_traction_residual_kg_m_s=(
                    discrete_pressure_traction_residual
                ),
                liquid_gravity_impulse_kg_m_s=liquid_gravity_impulse,
                gas_gravity_impulse_kg_m_s=(
                    gas_transport.gravity_impulse_kg_m_s
                ),
                wall_impulse_kg_m_s=wall_impulse,
                capillary_external_impulse_kg_m_s=capillary_external_impulse,
                predicted_mixture_impulse_kg_m_s=predicted_impulse,
                actual_mixture_impulse_kg_m_s=actual_impulse,
                residual_kg_m_s=momentum_residual,
            )

            def rate(after, before, *, nonnegative: bool):
                result: list[float] = []
                for target, base in zip(after, before, strict=True):
                    derivative = (target - base) / dt_s
                    # The component endpoint is admissible, but reconstructing
                    # an exact zero as ``base + dt * ((0-base)/dt)`` can land
                    # one ulp below zero.  Move only the rate representation
                    # upward until the immutable whole-network Euler view is
                    # non-negative.  The adjusted value is what the atomic
                    # packet and conservation ledger consume; no committed
                    # material or momentum is clipped afterwards.
                    if nonnegative and target >= 0.0:
                        for _ in range(4):
                            if base + dt_s * derivative >= 0.0:
                                break
                            derivative = math.nextafter(derivative, math.inf)
                        else:
                            raise ContractViolation(
                                "vertical non-negative endpoint cannot be represented as an RK rate"
                            )
                    result.append(derivative)
                return tuple(result)

            delta = VerticalDelta(
                Aup=rate(final.Aup, source_state.Aup, nonnegative=True),
                Qup=rate(final.Qup, source_state.Qup, nonnegative=True),
                Adown=rate(final.Adown, source_state.Adown, nonnegative=True),
                Qdown=rate(final.Qdown, source_state.Qdown, nonnegative=True),
                Mg=rate(final.Mg, source_state.Mg, nonnegative=True),
                Jg=rate(final.Jg, source_state.Jg, nonnegative=False),
            )
            proposal = ComponentStageProposal.accepted(
                component_id=self.component_id,
                base_state_token=trial.base_state_token,
                trials=(trial,),
                delta=delta,
                accepted_gross_fluxes=(flux,),
                external_exchange=external,
                evidence_status="S1-1D-F0_vertical_pressure_void_stage",
            )
            diagnostics = VerticalPressureVoidDiagnostics(
                common_pressure_faces_Pa=pressure_for_drag.common_pressure_faces,
                capillary_interfaces=capillary_interfaces,
                capillary_internal_recoil_residual_kg_m_s=capillary_residual,
                bottom_gas_piston=piston,
                void_remap=remap,
                top_liquid=top_liquid,
                top_gas=top_gas,
                wall=wall,
                three_body_recoil_residual_kg_m_s=drag.total_momentum_residual,
                liquid_volume_residual_m3=liquid_residual,
                gas_mass_residual_kg=gas_residual,
                mixture_momentum_z_residual_kg_m_s=momentum_residual,
                momentum_budget=momentum_budget,
            )
            return VerticalComponentEvaluation(proposal, diagnostics)
        except _RejectSignal as signal:
            return self._rejection(trial, signal)
        except MissingPhysicalClosure as exc:
            return self._rejection(
                trial,
                _RejectSignal(
                    "void_mass_pairing",
                    str(exc),
                    retryable=False,
                ),
            )
        except self._solver._runtime.fv.PackingViolationError as exc:
            return self._rejection(
                trial,
                _RejectSignal(
                    "phase_capacity",
                    str(exc),
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                ),
            )
        except self._solver._runtime.fv.StateAdmissibilityError as exc:
            return self._rejection(
                trial,
                _RejectSignal(
                    "phase_capacity",
                    str(exc),
                    retryable=True,
                    maximum_dt_s=0.5 * dt_s,
                ),
            )

    def propose_joint_stage(
        self,
        state: VerticalState,
        geometry: CoupledGeometry,
        *,
        riser_node_trial: TNodeTrial,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> ComponentStageProposal:
        """Return only the pure proposal required by the joint-stage protocol."""

        return self.evaluate_joint_stage(
            state,
            geometry,
            riser_node_trial=riser_node_trial,
            physical_stage=physical_stage,
            dt_s=dt_s,
        ).proposal

    def evaluate_trial(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> ComponentStageProposal:
        """Pure adapter used by the simultaneous two-node nonlinear owner."""

        validate_trial_set(trials)
        if len(trials) != 1 or trials[0].node_name != "riser_T":
            raise ContractViolation("vertical pure trial requires only riser_T")
        trial = trials[0]
        return self.propose_joint_stage(
            state.vertical,
            geometry,
            riser_node_trial=trial,
            physical_stage=trial.physical_stage,
            dt_s=trial.dt_s,
        )


__all__ = [
    "AtmosphericGasFlux",
    "AtmosphericLiquidFlux",
    "AtmosphericLiquidFallback",
    "AtmosphericTopState",
    "BottomGasPistonRemap",
    "CapillaryGeometryMode",
    "DetectedCapillaryInterface",
    "DynamicWallDiagnostics",
    "F0VerticalCapillaryOwner",
    "F0VerticalPressureVoidStageComponent",
    "GasTransportResult",
    "IndependentVerticalMomentumBudget",
    "S1_LIQUID_VISCOSITY_PA_S",
    "VerticalComponentEvaluation",
    "VerticalPressureVoidDiagnostics",
    "VoidRemapResult",
    "conservative_void_remap",
    "f0_smooth_pipe_darcy_factor",
]
