"""Pure F0 horizontal-main stage operator with both physical T junctions.

This module is the horizontal component needed by the whole-network
SSP-RK2 driver.  It deliberately does *not* own a time integrator.  Its
``HorizontalDelta`` is one finite-volume time derivative evaluated at one
global RK stage; :class:`model.joint_network_runner.S1JointNetworkRunner`
owns the two-stage composition and the sole atomic commit.

The liquid spatial operator is the hash-pinned Case-1 circular-pipe MUSCL
central-upwind operator and donor draining limiter.  The two physical T faces
split the main into three segments.  Each segment uses the unchanged Case-1
reconstruction/Riemann/limiter building blocks; only its endpoint flux is
replaced by the independently solved atomic T-port packet.  A moving resolved
gas nose is likewise an explicit phase-interface face, rather than a new
first-order stencil for the rest of the pipe.  Both Mahyawansi node trials are
consumed in the same proposal.  No finite Case-1 gas pocket, valve release,
fixed Darcy factor, fixed core fraction, Taylor/Wallis law, or lumped node
inertance is present.

Capillarity is routed through ``CapillaryInterfaceOwnership``.  For the frozen
2-D comparison, ``planar_2d_zeroGradient_walls`` means only that wall contact
angle is not imposed.  It does not mean zero curvature: a flat interface must
declare kappa=0, a declared semicircular nose/tail must declare kappa=+/-2/D,
and another local planar curvature must identify its geometry in the evidence
record.  The 3-D circular-meniscus mode remains available as an explicit
contract but cannot be inferred from the paper.

Table 1 uses different Fluent boundary semantics at the two water ends. The
upstream pressure inlet supplies total head and retains the outgoing Case-1
characteristic. The downstream pressure outlet supplies static piezometric
head; its characteristic ghost does not add kinetic head to the 0.584 m
datum. That outlet datum is the elastic storage zero, and the 0.5842 m Stage-1
field is its small positive elastic increment. Remaining network/top-boundary
and model-form gates keep this component non-production.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import numpy as np

from .errors import ConservationError, ContractViolation, MissingPhysicalClosure
from .flux import BoundaryExchange, HorizontalDelta
from .horizontal_case1_adapter import (
    FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S,
    Case1HorizontalLiquidAdapter,
)
from .horizontal_distributed import (
    HorizontalDistributedConfig,
    _darcy_factor,
    _isothermal_hll_density_flux,
    water_end_inlet_outlet_gas_flux,
)
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
from .state import CoupledGeometry, CoupledState, HorizontalState


Array = np.ndarray
CapillaryGeometryMode = Literal[
    "planar_2d_zeroGradient_walls",
    "circular_3d_meniscus",
]

PLANAR_2D_CAPILLARY_MODE = "planar_2d_zeroGradient_walls"
CIRCULAR_3D_CAPILLARY_MODE = "circular_3d_meniscus"
HORIZONTAL_COMPONENT_ID = "horizontal_main"
F0_STAGE_RATE_EVIDENCE = (
    "S1-1D-F0_hash_pinned_Case1_MUSCL_central_upwind_donor_draining__"
    "three_segments_two_atomic_T_faces__Table1_inlet_total_outlet_static__"
    "whole_network_SSP_RK2_owner"
)


def _tuple(values: Array) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _close(left: float, right: float, *, atol: float = 1.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2.0e-11, abs_tol=atol)


@dataclass(frozen=True, slots=True)
class HorizontalF0Readiness:
    """Result-independent readiness flags for this component only."""

    case1_geometry_pressure_law_derived: bool
    case1_circular_fv_lineage: bool
    both_t_ports_owned: bool
    capillary_geometry_mode_selected: bool
    liquid_eos_reconciled_with_2d: bool
    table1_characteristic_pressure_boundaries: bool

    @property
    def production_ready(self) -> bool:
        return all(
            (
                self.case1_geometry_pressure_law_derived,
                self.case1_circular_fv_lineage,
                self.both_t_ports_owned,
                self.capillary_geometry_mode_selected,
                self.liquid_eos_reconciled_with_2d,
                self.table1_characteristic_pressure_boundaries,
            )
        )


@dataclass(frozen=True, slots=True)
class _PortPhaseMomentum:
    liquid_to_node_x_N: float
    gas_to_node_x_N: float


@dataclass(slots=True)
class _MainArrays:
    Al: Array
    Ql: Array
    Mg: Array
    Jg: Array


@dataclass(slots=True)
class _CellFaceFluxes:
    liquid_volume_left_m3_s: Array
    liquid_volume_right_m3_s: Array
    liquid_momentum_left_m4_s2: Array
    liquid_momentum_right_m4_s2: Array
    gas_mass_left_kg_s: Array
    gas_mass_right_kg_s: Array
    gas_momentum_left_N: Array
    gas_momentum_right_N: Array
    liquid_pressure_area_left_m2: Array
    liquid_pressure_area_right_m2: Array


@dataclass(frozen=True, slots=True)
class _GasNoseFaceFlux:
    """One conservative one-sided gas-front transfer on a shared face."""

    liquid_volume_m3_s: float
    liquid_momentum_m4_s2: float
    gas_mass_kg_s: float
    gas_momentum_N: float
    liquid_face_area_m2: float


class F0HorizontalTwoTeeStageComponent:
    """Case-1-derived horizontal FV residual evaluated against two T trials."""

    component_id = HORIZONTAL_COMPONENT_ID
    source_aligned_trajectory_ready = False
    production_ready = False
    validation_only = True
    joint_trial_ready = True
    spatial_lineage = (
        "Case1 circular A(h)",
        "hash-pinned Case1 MUSCL central-upwind face kernel",
        "hash-pinned Case1 donor draining limiter",
        "whole-network SSP-RK2 using the Case1 forward-Euler spatial stage",
        "two atomic T endpoints plus explicit moving gas-interface faces",
        "Table-1 inlet-total/outlet-static characteristic ghosts",
    )

    def __init__(
        self,
        adapter: Case1HorizontalLiquidAdapter | None = None,
        *,
        config: HorizontalDistributedConfig | None = None,
        capillary_geometry_mode: CapillaryGeometryMode | None = None,
        liquid_eos_reconciled_with_2d: bool = False,
    ) -> None:
        self.adapter = Case1HorizontalLiquidAdapter() if adapter is None else adapter
        self.config = HorizontalDistributedConfig() if config is None else config
        if capillary_geometry_mode not in (
            None,
            PLANAR_2D_CAPILLARY_MODE,
            CIRCULAR_3D_CAPILLARY_MODE,
        ):
            raise ContractViolation("unsupported F0 horizontal capillary geometry mode")
        self.capillary_geometry_mode = capillary_geometry_mode
        self.liquid_eos_reconciled_with_2d = bool(liquid_eos_reconciled_with_2d)
        if self.liquid_eos_reconciled_with_2d and not _close(
            self.adapter.wave_speed_m_s,
            FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S,
            atol=1.0e-10,
        ):
            raise ContractViolation(
                "liquid_eos_reconciled_with_2d requires the frozen OpenFOAM "
                "perfectFluid tangent wave speed"
            )
        self.area_m2 = self.adapter.full_area_m2
        self.diameter_m = self.adapter.grid.diameter_m
        self.air_face = self.adapter.grid.air_tee_face_index
        self.riser_face = self.adapter.grid.riser_tee_face_index
        if not 0 < self.air_face < self.riser_face < self.adapter.grid.cell_count:
            raise ContractViolation("both S1 T junctions must be distinct internal faces")

    @property
    def readiness(self) -> HorizontalF0Readiness:
        return HorizontalF0Readiness(
            case1_geometry_pressure_law_derived=True,
            case1_circular_fv_lineage=True,
            both_t_ports_owned=True,
            capillary_geometry_mode_selected=self.capillary_geometry_mode is not None,
            liquid_eos_reconciled_with_2d=self.liquid_eos_reconciled_with_2d,
            table1_characteristic_pressure_boundaries=True,
        )

    @property
    def capillary_translation_evidence(self) -> str:
        if self.capillary_geometry_mode == PLANAR_2D_CAPILLARY_MODE:
            return (
                "declared_translation_of_frozen_2D_zeroGradient_walls__"
                "no_contact_angle_and_no_assumed_zero_curvature"
            )
        if self.capillary_geometry_mode == CIRCULAR_3D_CAPILLARY_MODE:
            return "3D_circular_meniscus_geometry_requires_explicit_contact_data"
        return "fail_closed_capillary_geometry_mode_not_selected"

    def assert_source_aligned_trajectory_ready(self) -> None:
        blockers: list[str] = []
        if self.capillary_geometry_mode is None:
            blockers.append("capillary geometry mode is not selected")
        if not self.liquid_eos_reconciled_with_2d:
            blockers.append("Case1 wave-speed EOS is not reconciled with frozen 2D")
        if blockers:
            raise MissingPhysicalClosure("; ".join(blockers))
        raise MissingPhysicalClosure(
            "horizontal unit component cannot authorize a trajectory before the "
            "simultaneous two-node and remaining F0 network gates pass"
        )

    def _validate_geometry(
        self, state: HorizontalState, geometry: CoupledGeometry
    ) -> None:
        if state.cell_count != self.adapter.grid.cell_count:
            raise ContractViolation("horizontal state does not use the frozen S1 grid")
        if len(geometry.horizontal_dx_m) != state.cell_count:
            raise ContractViolation("horizontal state/grid cell counts differ")
        if not _close(geometry.horizontal_area_m2, self.area_m2, atol=1.0e-15):
            raise ContractViolation("horizontal area differs from the Case1 circular adapter")
        if not _close(
            geometry.liquid_density_kg_m3,
            self.config.liquid_density_kg_m3,
            atol=1.0e-12,
        ):
            raise ContractViolation("horizontal liquid density differs from frozen F0")
        if any(
            not _close(dx, self.adapter.grid.dx_m, atol=1.0e-14)
            for dx in geometry.horizontal_dx_m
        ):
            raise ContractViolation(
                "horizontal FV cells must use the Case1-adapter grid without remapping"
            )
        required_overarea = self.config.maximum_elastic_overarea_fraction
        if geometry.horizontal_elastic_overarea_fraction + 1.0e-15 < required_overarea:
            raise ContractViolation(
                "coupled geometry does not declare the inherited elastic overarea bound"
            )
        self._validate_local_state(state, geometry)

    def _validate_local_state(
        self, state: HorizontalState, geometry: CoupledGeometry
    ) -> None:
        maximum = self.area_m2 * (
            1.0 + geometry.horizontal_elastic_overarea_fraction
        )
        area_tol = 5.0e-13 * self.area_m2
        mass_tol = self.config.gas_presence_mass_kg_m
        for index, (al, ql, mg, jg) in enumerate(
            zip(state.Al, state.Ql, state.Mg, state.Jg, strict=True)
        ):
            if not all(math.isfinite(value) for value in (al, ql, mg, jg)):
                raise ContractViolation(f"horizontal cell {index} is non-finite")
            if al < 0.0 or al > maximum:
                raise ContractViolation(f"horizontal cell {index} exceeds phase capacity")
            ag = max(self.area_m2 - al, 0.0)
            if mg > mass_tol and ag <= area_tol:
                raise ContractViolation(
                    f"horizontal cell {index} has gas mass without gas area"
                )
            if ag > area_tol and mg <= mass_tol:
                raise ContractViolation(
                    f"horizontal cell {index} has gas area without gas mass"
                )
            if mg <= mass_tol and abs(jg) > 1.0e-10:
                raise ContractViolation(
                    f"horizontal cell {index} has gas momentum without gas mass"
                )

    def _arrays(self, state: HorizontalState) -> _MainArrays:
        return _MainArrays(
            Al=np.asarray(state.Al, dtype=float),
            Ql=np.asarray(state.Ql, dtype=float),
            Mg=np.asarray(state.Mg, dtype=float),
            Jg=np.asarray(state.Jg, dtype=float),
        )

    def _gas_velocity(self, mass: float, momentum: float) -> float:
        if mass <= self.config.gas_presence_mass_kg_m:
            return 0.0
        return float(momentum) / float(mass)

    def _common_pressures(self, arrays: _MainArrays) -> Array:
        """Return the phase-interface/reference pressure used inside cells.

        Gas-occupied cells use the exact gas EOS.  Gas-free cells retain only
        the declared 0.584 m elastic-storage crown datum here because the
        area-dependent Case-1 force is already present in ``physical_flux``.
        The Table-1 inlet total head and outlet static head enter only through
        their distinct characteristic end ghosts.  A T-port needs that elastic
        force as an equivalent aperture pressure; :meth:`_liquid_port_pressure`
        adds it exactly once when the trace is formed.
        """

        gas_area = np.maximum(self.area_m2 - arrays.Al, 0.0)
        occupied = (
            (arrays.Mg > self.config.gas_presence_mass_kg_m)
            & (
                gas_area
                > self.config.gas_presence_area_fraction * self.area_m2
            )
        )
        pressure = np.empty_like(arrays.Al)
        pressure[occupied] = (
            arrays.Mg[occupied] * self.config.rt_J_kg / gas_area[occupied]
        )
        reference = self.config.atmospheric_pressure_Pa + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * (self.config.elastic_storage_reference_head_m - 0.5 * self.diameter_m)
        )
        pressure[~occupied] = reference
        if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise ContractViolation("horizontal common pressure must be positive and finite")
        return pressure

    def _liquid_port_pressure(self, liquid_area_m2: float, reference_Pa: float) -> float:
        """Absolute equivalent pressure presented by a full/elastic T trace.

        ``reference_Pa`` is the published-head crown datum.  The increment is
        derived from the pinned Case-1 conservative pressure flux and divided
        by the physical aperture, so ``Al > Af`` is neither clipped nor
        silently reset to the reference pressure.
        """

        area = float(liquid_area_m2)
        reference = float(reference_Pa)
        if area < self.area_m2:
            raise ContractViolation(
                "free-surface T pressure must come from gas/capillary closure"
            )
        increment = self.adapter.conservative_port_pressure_increment_Pa(
            area,
            self.config.liquid_density_kg_m3,
        )
        pressure = reference + increment
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ContractViolation("Case-1-derived T-port pressure is invalid")
        return pressure

    @staticmethod
    def _port_cells(face: int) -> tuple[tuple[str, int, float], ...]:
        return (("main_left", face - 1, 1.0), ("main_right", face, -1.0))

    def _all_port_cells(self) -> tuple[tuple[PortKey, int, float], ...]:
        result: list[tuple[PortKey, int, float]] = []
        for node_name, face in (
            ("air_supply_T", self.air_face),
            ("riser_T", self.riser_face),
        ):
            result.extend(
                (PortKey(node_name, port), cell, normal)
                for port, cell, normal in self._port_cells(face)
            )
        return tuple(result)

    def _validate_planar_interface(
        self, interface: CapillaryInterfaceOwnership
    ) -> None:
        if interface.contact_angle_deg is not None:
            raise ContractViolation(
                "frozen 2D walls use alpha.water zeroGradient; a contact angle "
                "cannot be inserted into the planar translation"
            )
        if (
            interface.curvature_1_m is None
            or interface.pressure_jump_gas_minus_liquid_Pa is None
        ):
            raise MissingPhysicalClosure(
                f"planar interface {interface.interface_id!r} has no explicit curvature"
            )
        evidence = interface.evidence_status.lower()
        curvature = interface.curvature_1_m
        cap = 2.0 / self.diameter_m
        if _close(curvature, 0.0, atol=1.0e-13):
            if "planar" not in evidence or "flat" not in evidence:
                raise MissingPhysicalClosure(
                    "kappa=0 is allowed only for an explicitly declared flat planar interface"
                )
        elif _close(abs(curvature), cap, atol=1.0e-10):
            if "planar" not in evidence or "semicircular" not in evidence:
                raise MissingPhysicalClosure(
                    "kappa=+/-2/D requires an explicitly declared planar semicircular cap"
                )
            is_nose = "gas_nose" in evidence
            is_tail = "gas_tail" in evidence
            if is_nose == is_tail:
                raise MissingPhysicalClosure(
                    "a planar semicircular cap must declare exactly one topology: "
                    "gas_nose or gas_tail"
                )
            topology_curvature = cap if is_nose else -cap
            if not _close(curvature, topology_curvature, atol=1.0e-10):
                raise MissingPhysicalClosure(
                    "semicircular curvature sign is topology-frozen: gas_nose=+2/D "
                    "and gas_tail=-2/D"
                )
        elif "local_planar_interface_geometry" not in evidence:
            raise MissingPhysicalClosure(
                "non-flat planar curvature needs an explicit local planar geometry record"
            )

    def _validate_capillary_interfaces(
        self,
        state: HorizontalState,
        trials: tuple[TNodeTrial, ...],
    ) -> CapacityReject | None:
        interfaces = {
            interface.interface_id: interface
            for trial in trials
            for interface in trial.interfaces
        }
        main_traces = {
            trace.key: trace
            for trial in trials
            for trace in trial.port_traces
            if trace.component_id == self.component_id
        }
        gross = {
            flux.key: flux
            for trial in trials
            for flux in trial.gross_fluxes
            if flux.key in main_traces
        }
        port_cells = {key: cell for key, cell, _ in self._all_port_cells()}
        required_ids: set[str] = set()
        area_tol = 5.0e-13 * self.area_m2
        for key, flux in gross.items():
            cell = port_cells[key]
            is_full_water = (
                state.Mg[cell] <= self.config.gas_presence_mass_kg_m
                and self.area_m2 - state.Al[cell] <= area_tol
            )
            if is_full_water and flux.gas_out_of_node_kg_s > flux.gas_into_node_kg_s:
                interface_id = main_traces[key].interface_id
                if interface_id is None:
                    return CapacityReject(
                        component_id=self.component_id,
                        reason_code="missing_closure",
                        detail=(
                            f"first gas entry at {key.label} has no unique capillary "
                            "interface ownership record"
                        ),
                        requested_dt_s=trials[0].dt_s,
                        retryable=False,
                    )
                required_ids.add(interface_id)
                record = interfaces.get(interface_id)
                if (
                    record is not None
                    and self.capillary_geometry_mode == PLANAR_2D_CAPILLARY_MODE
                    and "gas_nose" not in record.evidence_status.lower()
                ):
                    return CapacityReject(
                        component_id=self.component_id,
                        reason_code="missing_closure",
                        detail=(
                            f"first gas entry at {key.label} must use the topology-frozen "
                            "gas_nose curvature sign"
                        ),
                        requested_dt_s=trials[0].dt_s,
                        retryable=False,
                    )

        referenced_ids = {
            trace.interface_id
            for trace in main_traces.values()
            if trace.interface_id is not None
        }
        if not referenced_ids and not required_ids:
            return None
        if self.capillary_geometry_mode is None:
            return CapacityReject(
                component_id=self.component_id,
                reason_code="missing_closure",
                detail="capillary geometry mode was not explicitly selected",
                requested_dt_s=trials[0].dt_s,
                retryable=False,
            )
        try:
            for interface_id in referenced_ids | required_ids:
                interface = interfaces[interface_id]
                if self.capillary_geometry_mode == PLANAR_2D_CAPILLARY_MODE:
                    self._validate_planar_interface(interface)
                elif not interface.geometrically_resolved:
                    raise MissingPhysicalClosure(
                        "3D circular meniscus requires explicit curvature and contact geometry"
                    )
        except (KeyError, MissingPhysicalClosure, ContractViolation) as exc:
            return CapacityReject(
                component_id=self.component_id,
                reason_code="missing_closure",
                detail=str(exc),
                requested_dt_s=trials[0].dt_s,
                retryable=False,
            )
        return None

    def port_traces(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        *,
        interfaces_by_port: Mapping[
            PortKey, CapillaryInterfaceOwnership
        ] | None = None,
    ) -> tuple[PortTraceState, ...]:
        """Expose all four horizontal traces without solving either T node."""

        self._validate_geometry(state, geometry)
        interfaces = {} if interfaces_by_port is None else dict(interfaces_by_port)
        allowed = {key for key, _, _ in self._all_port_cells()}
        if any(key not in allowed for key in interfaces):
            raise ContractViolation("capillary interface was attached to a non-horizontal port")
        if interfaces and self.capillary_geometry_mode is None:
            raise MissingPhysicalClosure(
                "cannot attach a capillary interface before selecting its geometry mode"
            )
        arrays = self._arrays(state)
        common = self._common_pressures(arrays)
        result: list[PortTraceState] = []
        for key, cell, normal_x in self._all_port_cells():
            liquid_area = min(float(arrays.Al[cell]), self.area_m2)
            gas_area = self.area_m2 - liquid_area
            gas_present = (
                arrays.Mg[cell] > self.config.gas_presence_mass_kg_m
                and gas_area
                > self.config.gas_presence_area_fraction * self.area_m2
            )
            interface = interfaces.get(key)
            jump = (
                0.0
                if interface is None
                or interface.pressure_jump_gas_minus_liquid_Pa is None
                else interface.pressure_jump_gas_minus_liquid_Pa
            )
            if gas_present:
                gas_pressure = float(common[cell])
                liquid_pressure = gas_pressure - jump
                gas_density = float(arrays.Mg[cell] / gas_area)
            else:
                liquid_pressure = self._liquid_port_pressure(
                    float(arrays.Al[cell]),
                    float(common[cell]),
                )
                gas_pressure = liquid_pressure + jump
                gas_density = gas_pressure / self.config.rt_J_kg
            if liquid_pressure <= 0.0 or gas_pressure <= 0.0:
                raise ContractViolation("capillary trace produced non-positive phase pressure")
            liquid_velocity = (
                0.0
                if liquid_area <= 0.0
                else float(arrays.Ql[cell] / liquid_area)
            )
            result.append(
                PortTraceState(
                    key=key,
                    component_id=self.component_id,
                    normal_into_node_x=normal_x,
                    normal_into_node_z=0.0,
                    full_area_m2=self.area_m2,
                    liquid_area_m2=liquid_area,
                    gas_area_m2=gas_area,
                    liquid_density_kg_m3=self.config.liquid_density_kg_m3,
                    gas_density_kg_m3=gas_density,
                    liquid_absolute_pressure_Pa=liquid_pressure,
                    gas_absolute_pressure_Pa=gas_pressure,
                    liquid_axial_velocity_m_s=liquid_velocity,
                    gas_axial_velocity_m_s=self._gas_velocity(
                        float(arrays.Mg[cell]), float(arrays.Jg[cell])
                    ),
                    interface_id=None if interface is None else interface.interface_id,
                    evidence_status=(
                        "hash_pinned_Case1_MUSCL_central_upwind_cell_trace__"
                        "elastic_force_equivalent_physical_aperture_pressure__"
                        "Case1_donor_draining__S1_two_T_port__"
                        + self.capillary_translation_evidence
                    ),
                )
            )
        return tuple(result)

    def stationary_pressure_traction_to_node_N(
        self, trace: PortTraceState
    ) -> float:
        """Manufactured-equilibrium helper; this is not a node flux solver."""

        if trace.component_id != self.component_id:
            raise ContractViolation("stationary traction helper requires a main trace")
        liquid_case1 = (
            trace.liquid_density_kg_m3
            * self.adapter.physical_flux(trace.liquid_area_m2, 0.0).liquid_momentum_m4_s2
        )
        pressure = (
            liquid_case1
            + trace.liquid_absolute_pressure_Pa * trace.liquid_area_m2
            + trace.gas_absolute_pressure_Pa * trace.gas_area_m2
        )
        return trace.normal_into_node_x * pressure

    def _validate_trial_traces(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> dict[PortKey, PortTraceState]:
        interface_records = {
            interface.interface_id: interface
            for trial in trials
            for interface in trial.interfaces
        }
        supplied = {
            trace.key: trace
            for trial in trials
            for trace in trial.port_traces
            if trace.component_id == self.component_id
        }
        interface_by_port = {
            key: interface_records[trace.interface_id]
            for key, trace in supplied.items()
            if trace.interface_id is not None
        }
        expected = {
            trace.key: trace
            for trace in self.port_traces(
                state,
                geometry,
                interfaces_by_port=interface_by_port,
            )
        }
        if set(supplied) != set(expected):
            raise ContractViolation("two T trials do not expose all four horizontal ports")
        numerical_fields = (
            "normal_into_node_x",
            "normal_into_node_z",
            "full_area_m2",
            "liquid_area_m2",
            "gas_area_m2",
            "liquid_density_kg_m3",
            "gas_density_kg_m3",
            "liquid_absolute_pressure_Pa",
            "gas_absolute_pressure_Pa",
            "liquid_axial_velocity_m_s",
            "gas_axial_velocity_m_s",
        )
        for key, actual in supplied.items():
            wanted = expected[key]
            if actual.interface_id != wanted.interface_id:
                raise ContractViolation(f"stale interface ownership at {key.label}")
            for name in numerical_fields:
                if not _close(
                    getattr(actual, name),
                    getattr(wanted, name),
                    atol=1.0e-10 if "pressure" in name else 1.0e-12,
                ):
                    raise ContractViolation(
                        f"stale horizontal port trace {key.label}: {name} differs"
                    )
        return supplied

    @staticmethod
    def _weighted_split(total: float, liquid_weight: float, gas_weight: float) -> tuple[float, float]:
        denominator = liquid_weight + gas_weight
        if denominator <= 0.0:
            if abs(total) > 1.0e-12:
                raise ContractViolation("port momentum has no phase carrier")
            return 0.0, 0.0
        liquid = total * liquid_weight / denominator
        return liquid, total - liquid

    def _split_port_momentum(
        self, trace: PortTraceState, flux: GrossNodePortFlux
    ) -> _PortPhaseMomentum:
        if abs(flux.advective_momentum_to_node_z_N) > 1.0e-10 or abs(
            flux.pressure_traction_to_node_z_N
        ) > 1.0e-10:
            raise ContractViolation("a horizontal main port cannot carry z momentum")
        liquid_advective = trace.liquid_density_kg_m3 * (
            flux.liquid_into_node_m3_s * flux.liquid_into_node_speed_m_s
            + flux.liquid_out_of_node_m3_s * flux.liquid_out_of_node_speed_m_s
        )
        gas_advective = (
            flux.gas_into_node_kg_s * flux.gas_into_node_speed_m_s
            + flux.gas_out_of_node_kg_s * flux.gas_out_of_node_speed_m_s
        )
        liquid_adv_x, gas_adv_x = self._weighted_split(
            flux.advective_momentum_to_node_x_N,
            liquid_advective,
            gas_advective,
        )
        liquid_pressure_weight = (
            trace.liquid_density_kg_m3
            * self.adapter.physical_flux(
                trace.liquid_area_m2, 0.0
            ).liquid_momentum_m4_s2
            + trace.liquid_absolute_pressure_Pa * trace.liquid_area_m2
        )
        gas_pressure_weight = trace.gas_absolute_pressure_Pa * trace.gas_area_m2
        liquid_pressure_x, gas_pressure_x = self._weighted_split(
            flux.pressure_traction_to_node_x_N,
            liquid_pressure_weight,
            gas_pressure_weight,
        )
        liquid = liquid_adv_x + liquid_pressure_x
        gas = gas_adv_x + gas_pressure_x
        if not _close(
            liquid + gas,
            flux.mixture_momentum_to_node_x_N,
            atol=1.0e-9,
        ):
            raise ConservationError("phase split changed a T-port x-momentum flux")
        return _PortPhaseMomentum(liquid, gas)

    def _one_sided_gas_nose_flux(
        self,
        arrays: _MainArrays,
        pressure: Array,
        face: int,
    ) -> _GasNoseFaceFlux | None:
        """Return an atomic gas/void/liquid-displacement flux at one gas nose.

        The ordinary two-gas-cell stencil uses the common connection area
        ``min(AgL,AgR)``.  That area is identically zero when a resolved gas
        cell first meets a full-water cell and would pin the front forever.
        At exactly that one-sided topology, the gas donor aperture is used for
        an isothermal HLL gas-mass flux.  Its volume rate ``mdot/rho_donor`` is
        paired on the *same shared face* with an equal and opposite liquid
        volume rate, while its advective momentum is ``mdot*u_donor``.  The
        full-water-side pressure traction is added separately and cancels its
        absent-gas geometry source.  Thus the receiving full-water cell
        acquires positive gas area, mass and correctly signed momentum in one
        stage, while the donor gains the same liquid volume.  No donor gas
        means this branch is unavailable, so a newly created unpaired void
        still fails the later capacity gate.

        The counterflow pairing represents a stratified gas nose with a liquid
        return path beneath it.  If the donor has no remaining liquid aperture,
        this reduced closure cannot route displacement and returns a closed
        contact; it does not manufacture a plug-flow rule.
        """

        left = face - 1
        right = face
        ag_left = max(self.area_m2 - float(arrays.Al[left]), 0.0)
        ag_right = max(self.area_m2 - float(arrays.Al[right]), 0.0)
        area_presence = self.config.gas_presence_area_fraction * self.area_m2
        full_tolerance = 5.0e-13 * self.area_m2
        mass_presence = self.config.gas_presence_mass_kg_m
        gas_left = arrays.Mg[left] > mass_presence and ag_left > area_presence
        gas_right = arrays.Mg[right] > mass_presence and ag_right > area_presence
        full_left = arrays.Mg[left] <= mass_presence and ag_left <= full_tolerance
        full_right = arrays.Mg[right] <= mass_presence and ag_right <= full_tolerance
        if not ((gas_left and full_right) or (full_left and gas_right)):
            return None

        donor = left if gas_left else right
        full_cell = right if gas_left else left
        donor_area = ag_left if gas_left else ag_right
        donor_density = float(arrays.Mg[donor]) / donor_area
        donor_velocity = self._gas_velocity(
            float(arrays.Mg[donor]), float(arrays.Jg[donor])
        )
        gas_moves_toward_full_cell = (
            gas_left and donor_velocity > 0.0
        ) or (gas_right and donor_velocity < 0.0)
        liquid_face_area = self.area_m2 - donor_area
        face_pressure = float(pressure[full_cell])

        mass_flux = 0.0
        # At a non-advancing one-sided contact, use the full-water trace
        # pressure so its gas-pressure geometry source cancels exactly without
        # depositing momentum into a cell that still has zero gas mass.
        gas_momentum_flux = face_pressure * donor_area
        liquid_volume_flux = 0.0
        if gas_moves_toward_full_cell and liquid_face_area > area_presence:
            if gas_left:
                mass_density_flux, _ = (
                    _isothermal_hll_density_flux(
                        donor_density,
                        donor_velocity,
                        0.0,
                        0.0,
                        self.config.rt_J_kg,
                    )
                )
                direction_ok = mass_density_flux > 0.0
            else:
                mass_density_flux, _ = (
                    _isothermal_hll_density_flux(
                        0.0,
                        0.0,
                        donor_density,
                        donor_velocity,
                        self.config.rt_J_kg,
                    )
                )
                direction_ok = mass_density_flux < 0.0
            if direction_ok:
                mass_flux = mass_density_flux * donor_area
                gas_momentum_flux = (
                    face_pressure * donor_area + mass_flux * donor_velocity
                )
                gas_volume_flux = mass_flux / donor_density
                # Opposite signed phase rates across the same face preserve
                # total liquid volume while opening the receiving gas void.
                liquid_volume_flux = -gas_volume_flux

        liquid_momentum_flux = self.adapter.physical_flux(
            liquid_face_area,
            liquid_volume_flux,
        ).liquid_momentum_m4_s2
        liquid_momentum_flux += (
            face_pressure
            * liquid_face_area
            / self.config.liquid_density_kg_m3
        )
        values = (
            liquid_volume_flux,
            liquid_momentum_flux,
            mass_flux,
            gas_momentum_flux,
            liquid_face_area,
        )
        if not all(math.isfinite(value) for value in values):
            raise ContractViolation("one-sided gas-nose flux became non-finite")
        return _GasNoseFaceFlux(*values)

    def _table1_pressure_ghosts(
        self,
        arrays: _MainArrays,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return inlet-total/outlet-static Table-1 characteristic ghosts."""

        common = dict(
            reference_head_m=self.config.elastic_storage_reference_head_m,
            maximum_elastic_overarea_fraction=(
                self.config.maximum_elastic_overarea_fraction
            ),
        )
        left = self.adapter.dynamic_total_pressure_ghost(
            interior_area_m2=float(arrays.Al[0]),
            interior_discharge_m3_s=float(arrays.Ql[0]),
            prescribed_total_head_m=self.config.water_inlet_head_m,
            side="left",
            **common,
        )
        right = self.adapter.static_pressure_characteristic_ghost(
            interior_area_m2=float(arrays.Al[-1]),
            interior_discharge_m3_s=float(arrays.Ql[-1]),
            prescribed_static_head_m=self.config.water_outlet_head_m,
            side="right",
            **common,
        )
        return (
            (left.liquid_area_m2, left.liquid_discharge_m3_s),
            (right.liquid_area_m2, right.liquid_discharge_m3_s),
        )

    def _case1_segment_liquid_fluxes(
        self,
        arrays: _MainArrays,
        pressure: Array,
        traces: Mapping[PortKey, PortTraceState],
        gross: Mapping[PortKey, GrossNodePortFlux],
        noses: Mapping[int, _GasNoseFaceFlux],
        *,
        dt_s: float,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Assemble three exact Case-1 liquid segments around the two T nodes."""

        n = arrays.Al.size
        left_boundary_ghost, right_boundary_ghost = (
            self._table1_pressure_ghosts(arrays)
        )
        liquid_left = np.full(n, np.nan)
        liquid_right = np.full(n, np.nan)
        momentum_left = np.full(n, np.nan)
        momentum_right = np.full(n, np.nan)
        pressure_area_left = np.full(n, np.nan)
        pressure_area_right = np.full(n, np.nan)
        ag = np.maximum(self.area_m2 - arrays.Al, 0.0)
        rho_l = self.config.liquid_density_kg_m3
        reference_pressure = self.config.atmospheric_pressure_Pa + (
            rho_l
            * self.adapter.gravity_m_s2
            * (self.config.elastic_storage_reference_head_m - 0.5 * self.diameter_m)
        )
        node_at_face = {
            self.air_face: "air_supply_T",
            self.riser_face: "riser_T",
        }
        segments = (
            (0, self.air_face),
            (self.air_face, self.riser_face),
            (self.riser_face, n),
        )
        for start, end in segments:
            area_segment = arrays.Al[start:end]
            discharge_segment = arrays.Ql[start:end]
            left_ghost = (
                left_boundary_ghost
                if start == 0
                else (float(area_segment[0]), float(discharge_segment[0]))
            )
            right_ghost = (
                right_boundary_ghost
                if end == n
                else (float(area_segment[-1]), float(discharge_segment[-1]))
            )
            case1 = self.adapter.case1_muscl_central_upwind_face_fluxes(
                area_segment,
                discharge_segment,
                left_ghost=left_ghost,
                right_ghost=right_ghost,
            )
            mass = np.asarray(case1.liquid_volume_m3_s, dtype=float)
            momentum = np.asarray(case1.liquid_momentum_m4_s2, dtype=float)
            face_area = np.empty(end - start + 1)
            locked_faces: set[int] = set()

            for local_face, global_face in enumerate(range(start, end + 1)):
                node_name = node_at_face.get(global_face)
                if node_name is not None:
                    if global_face == end:
                        key = PortKey(node_name, "main_left")
                        sign = 1.0
                    elif global_face == start:
                        key = PortKey(node_name, "main_right")
                        sign = -1.0
                    else:  # pragma: no cover - segments are frozen at both T faces
                        raise ContractViolation("a T face appeared inside a Case-1 segment")
                    phase = self._split_port_momentum(traces[key], gross[key])
                    mass[local_face] = sign * gross[key].liquid_net_into_node_m3_s
                    momentum[local_face] = sign * phase.liquid_to_node_x_N / rho_l
                    face_area[local_face] = traces[key].liquid_area_m2
                    locked_faces.add(local_face)
                    continue

                nose = noses.get(global_face)
                if nose is not None:
                    mass[local_face] = nose.liquid_volume_m3_s
                    momentum[local_face] = nose.liquid_momentum_m4_s2
                    face_area[local_face] = nose.liquid_face_area_m2
                    locked_faces.add(local_face)
                    continue

                if global_face == 0:
                    # Table-1 alpha.water inletOutlet admits pure water on
                    # reverse flow, but it cannot overlap a resolved gas
                    # aperture already touching the end.  The Case-1
                    # characteristic supplies the finite liquid rate; this
                    # complementary aperture keeps liquid+gas traction equal
                    # to the physical pipe area during phase re-entry.
                    liquid_face_area = self.area_m2 - ag[0]
                    common_face_pressure = 0.5 * (
                        reference_pressure + float(pressure[0])
                    )
                elif global_face == n:
                    liquid_face_area = self.area_m2 - ag[-1]
                    common_face_pressure = 0.5 * (
                        float(pressure[-1]) + reference_pressure
                    )
                else:
                    connection_gas_area = max(
                        min(ag[global_face - 1], ag[global_face]), 0.0
                    )
                    liquid_face_area = self.area_m2 - connection_gas_area
                    common_face_pressure = 0.5 * (
                        float(pressure[global_face - 1])
                        + float(pressure[global_face])
                    )
                face_area[local_face] = liquid_face_area
                momentum[local_face] += (
                    common_face_pressure * liquid_face_area / rho_l
                )

            raw_mass = mass.copy()
            raw_momentum = momentum.copy()
            limited = self.adapter.case1_donor_draining_limit(
                area_segment,
                mass,
                momentum,
                dx_m=self.adapter.grid.dx_m,
                dt_s=dt_s,
            )
            mass = np.asarray(limited.liquid_volume_m3_s, dtype=float)
            momentum = np.asarray(limited.liquid_momentum_m4_s2, dtype=float)
            for face in locked_faces:
                if not (
                    _close(mass[face], raw_mass[face], atol=2.0e-14)
                    and _close(momentum[face], raw_momentum[face], atol=2.0e-11)
                ):
                    raise ContractViolation(
                        "Case-1 donor limiter would alter an atomic T/gas-interface flux; "
                        "retry the common network stage with a smaller dt"
                    )

            for local_cell, global_cell in enumerate(range(start, end)):
                liquid_left[global_cell] = mass[local_cell]
                liquid_right[global_cell] = mass[local_cell + 1]
                momentum_left[global_cell] = momentum[local_cell]
                momentum_right[global_cell] = momentum[local_cell + 1]
                pressure_area_left[global_cell] = face_area[local_cell]
                pressure_area_right[global_cell] = face_area[local_cell + 1]

        fields = (
            liquid_left,
            liquid_right,
            momentum_left,
            momentum_right,
            pressure_area_left,
            pressure_area_right,
        )
        if any(np.any(~np.isfinite(field)) for field in fields):
            raise ContractViolation("Case-1 segment assembly left an unowned liquid face")
        return fields

    def _build_face_fluxes(
        self,
        arrays: _MainArrays,
        traces: Mapping[PortKey, PortTraceState],
        gross: Mapping[PortKey, GrossNodePortFlux],
        *,
        dt_s: float,
    ) -> _CellFaceFluxes:
        n = arrays.Al.size
        pressure = self._common_pressures(arrays)
        ag = np.maximum(self.area_m2 - arrays.Al, 0.0)

        gas_left = np.full(n, np.nan)
        gas_right = np.full(n, np.nan)
        gas_momentum_left = np.full(n, np.nan)
        gas_momentum_right = np.full(n, np.nan)
        area_presence = self.config.gas_presence_area_fraction * self.area_m2
        left_end = water_end_inlet_outlet_gas_flux(
            side="left",
            gas_area_m2=float(ag[0]),
            gas_mass_kg_m=float(arrays.Mg[0]),
            gas_momentum_kg_s=float(arrays.Jg[0]),
            interior_absolute_pressure_Pa=float(pressure[0]),
            dx_m=self.adapter.grid.dx_m,
            dt_s=dt_s,
            gas_presence_mass_kg_m=self.config.gas_presence_mass_kg_m,
            gas_presence_area_m2=area_presence,
        )
        right_end = water_end_inlet_outlet_gas_flux(
            side="right",
            gas_area_m2=float(ag[-1]),
            gas_mass_kg_m=float(arrays.Mg[-1]),
            gas_momentum_kg_s=float(arrays.Jg[-1]),
            interior_absolute_pressure_Pa=float(pressure[-1]),
            dx_m=self.adapter.grid.dx_m,
            dt_s=dt_s,
            gas_presence_mass_kg_m=self.config.gas_presence_mass_kg_m,
            gas_presence_area_m2=area_presence,
        )
        gas_left[0] = left_end.gas_mass_left_to_right_kg_s
        gas_right[-1] = right_end.gas_mass_left_to_right_kg_s
        gas_momentum_left[0] = left_end.gas_momentum_left_to_right_N
        gas_momentum_right[-1] = right_end.gas_momentum_left_to_right_N

        t_faces = {self.air_face, self.riser_face}
        noses: dict[int, _GasNoseFaceFlux] = {}
        for face in range(1, n):
            if face in t_faces:
                continue
            nose = self._one_sided_gas_nose_flux(arrays, pressure, face)
            if nose is not None:
                noses[face] = nose
                gas_right[face - 1] = gas_left[face] = nose.gas_mass_kg_s
                gas_momentum_right[face - 1] = gas_momentum_left[face] = (
                    nose.gas_momentum_N
                )
                continue
            connection_gas_area = max(min(ag[face - 1], ag[face]), 0.0)
            if (
                connection_gas_area
                <= self.config.gas_presence_area_fraction * self.area_m2
            ):
                mass_flux = 0.0
                momentum_flux = 0.0
            else:
                rho_g_left = arrays.Mg[face - 1] / ag[face - 1]
                rho_g_right = arrays.Mg[face] / ag[face]
                mass_density, momentum_density = _isothermal_hll_density_flux(
                    float(rho_g_left),
                    self._gas_velocity(
                        float(arrays.Mg[face - 1]), float(arrays.Jg[face - 1])
                    ),
                    float(rho_g_right),
                    self._gas_velocity(float(arrays.Mg[face]), float(arrays.Jg[face])),
                    self.config.rt_J_kg,
                )
                mass_flux = mass_density * connection_gas_area
                momentum_flux = momentum_density * connection_gas_area
            gas_right[face - 1] = gas_left[face] = mass_flux
            gas_momentum_right[face - 1] = gas_momentum_left[face] = momentum_flux

        for node_name, face in (
            ("air_supply_T", self.air_face),
            ("riser_T", self.riser_face),
        ):
            left_key = PortKey(node_name, "main_left")
            right_key = PortKey(node_name, "main_right")
            left_cell = face - 1
            right_cell = face
            left_phase = self._split_port_momentum(traces[left_key], gross[left_key])
            right_phase = self._split_port_momentum(traces[right_key], gross[right_key])
            gas_right[left_cell] = gross[left_key].gas_net_into_node_kg_s
            gas_momentum_right[left_cell] = left_phase.gas_to_node_x_N
            gas_left[right_cell] = -gross[right_key].gas_net_into_node_kg_s
            gas_momentum_left[right_cell] = -right_phase.gas_to_node_x_N

        (
            liquid_left,
            liquid_right,
            liquid_momentum_left,
            liquid_momentum_right,
            pressure_area_left,
            pressure_area_right,
        ) = self._case1_segment_liquid_fluxes(
            arrays,
            pressure,
            traces,
            gross,
            noses,
            dt_s=dt_s,
        )
        fields = (
            liquid_left,
            liquid_right,
            liquid_momentum_left,
            liquid_momentum_right,
            gas_left,
            gas_right,
            gas_momentum_left,
            gas_momentum_right,
            pressure_area_left,
            pressure_area_right,
        )
        if any(np.any(~np.isfinite(field)) for field in fields):
            raise ContractViolation("horizontal face assembly left an unowned FV face")
        return _CellFaceFluxes(*fields)

    @staticmethod
    def _gross_boundary_flow(
        left_oriented: float, right_oriented: float
    ) -> tuple[float, float]:
        inflow = max(left_oriented, 0.0) + max(-right_oriented, 0.0)
        outflow = max(-left_oriented, 0.0) + max(right_oriented, 0.0)
        return inflow, outflow

    @staticmethod
    def _fanning_factor(reynolds: float) -> float:
        re = max(float(reynolds), 1.0e-12)
        value = 16.0 / re if re < 2100.0 else 0.046 * re**-0.2
        return min(max(value, 0.0), 4.0)

    def _rate_and_external(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        traces: Mapping[PortKey, PortTraceState],
        gross: Mapping[PortKey, GrossNodePortFlux],
        node_gas_density_by_key: Mapping[PortKey, float],
        *,
        dt_s: float,
    ) -> tuple[HorizontalDelta, BoundaryExchange]:
        arrays = self._arrays(state)
        pressure = self._common_pressures(arrays)
        ag = np.maximum(self.area_m2 - arrays.Al, 0.0)
        faces = self._build_face_fluxes(arrays, traces, gross, dt_s=dt_s)
        dx = np.asarray(geometry.horizontal_dx_m, dtype=float)
        rho_l = self.config.liquid_density_kg_m3

        dAl = -(
            faces.liquid_volume_right_m3_s - faces.liquid_volume_left_m3_s
        ) / dx
        dQl = -(
            faces.liquid_momentum_right_m4_s2
            - faces.liquid_momentum_left_m4_s2
        ) / dx
        dMg = -(faces.gas_mass_right_kg_s - faces.gas_mass_left_kg_s) / dx
        dJg = -(
            faces.gas_momentum_right_N - faces.gas_momentum_left_N
        ) / dx

        # A first gas parcel entering a full-water T port opens one gas-nose
        # volume in that receiving cell and displaces the same liquid volume
        # into the adjacent main cell away from the junction.  This is the
        # T-face analogue of the existing one-sided internal gas-nose flux:
        # gas mass, void opening and liquid displacement occur in the same
        # immutable stage, while total main-pipe liquid volume is unchanged.
        port_cells = {key: cell for key, cell, _ in self._all_port_cells()}
        for key, flux in gross.items():
            cell = port_cells[key]
            initially_full = (
                state.Mg[cell] <= self.config.gas_presence_mass_kg_m
                and self.area_m2 - state.Al[cell]
                <= 5.0e-13 * self.area_m2
            )
            incoming_gas = (
                flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
            )
            if not initially_full or incoming_gas <= 0.0:
                continue
            density = node_gas_density_by_key[key]
            opening = incoming_gas / density
            neighbor = cell - 1 if key.port_name == "main_left" else cell + 1
            if not 0 <= neighbor < state.cell_count:
                raise ContractViolation("T gas-nose displacement has no main neighbor")
            elastic_release = max(state.Al[cell] - self.area_m2, 0.0) * dx[cell]
            displaced = opening + elastic_release / dt_s
            dAl[cell] -= displaced / dx[cell]
            dAl[neighbor] += displaced / dx[neighbor]

        pressure_area_gradient = (
            faces.liquid_pressure_area_right_m2
            - faces.liquid_pressure_area_left_m2
        ) / dx
        dQl += pressure / rho_l * pressure_area_gradient
        dJg -= pressure * pressure_area_gradient

        gravity = self.adapter.gravity_m_s2
        slope = self.config.main_slope_sine
        dQl += arrays.Al * gravity * slope
        dJg += arrays.Mg * gravity * slope
        external_force_x = float(
            np.sum((rho_l * arrays.Al + arrays.Mg) * gravity * slope * dx)
        )

        circumference = math.pi * self.diameter_m
        for index in range(arrays.Al.size):
            al = float(arrays.Al[index])
            gas_area = float(max(ag[index], 0.0))
            ul = 0.0 if al <= 0.0 else float(arrays.Ql[index] / al)
            ug = self._gas_velocity(float(arrays.Mg[index]), float(arrays.Jg[index]))
            liquid_perimeter = self.adapter.wetted_perimeter_m(min(al, self.area_m2))
            gas_perimeter = max(circumference - liquid_perimeter, 0.0)

            hydraulic_l = 0.0 if liquid_perimeter <= 0.0 else 4.0 * al / liquid_perimeter
            re_l = rho_l * abs(ul) * hydraulic_l / self.config.liquid_viscosity_Pa_s
            f_l = _darcy_factor(re_l)
            liquid_wall_force = -f_l * rho_l * liquid_perimeter * ul * abs(ul) / 8.0
            dQl[index] += liquid_wall_force / rho_l
            external_force_x += liquid_wall_force * dx[index]

            if gas_area > 0.0 and arrays.Mg[index] > 0.0 and gas_perimeter > 0.0:
                rho_g = float(arrays.Mg[index] / gas_area)
                hydraulic_g = 4.0 * gas_area / gas_perimeter
                re_g = rho_g * abs(ug) * hydraulic_g / self.config.gas_viscosity_Pa_s
                f_g = _darcy_factor(re_g)
                gas_wall_force = -f_g * gas_perimeter * rho_g * ug * abs(ug) / 8.0
                dJg[index] += gas_wall_force
                external_force_x += gas_wall_force * dx[index]

                interface_perimeter = self.adapter.interface_width_m(min(al, self.area_m2))
                if interface_perimeter > 0.0:
                    hydraulic_i = 4.0 * gas_area / max(
                        gas_perimeter + interface_perimeter, 1.0e-300
                    )
                    slip = ug - ul
                    re_i = (
                        rho_g
                        * abs(slip)
                        * hydraulic_i
                        / self.config.gas_viscosity_Pa_s
                    )
                    fanning = self._fanning_factor(re_i)
                    force_on_liquid = (
                        0.5
                        * fanning
                        * rho_g
                        * abs(slip)
                        * slip
                        * interface_perimeter
                    )
                    dQl[index] += force_on_liquid / rho_l
                    dJg[index] -= force_on_liquid

        liquid_in, liquid_out = self._gross_boundary_flow(
            float(faces.liquid_volume_left_m3_s[0]),
            float(faces.liquid_volume_right_m3_s[-1]),
        )
        gas_in, gas_out = self._gross_boundary_flow(
            float(faces.gas_mass_left_kg_s[0]),
            float(faces.gas_mass_right_kg_s[-1]),
        )
        momentum_left = (
            rho_l * faces.liquid_momentum_left_m4_s2[0]
            + faces.gas_momentum_left_N[0]
        )
        momentum_right = (
            rho_l * faces.liquid_momentum_right_m4_s2[-1]
            + faces.gas_momentum_right_N[-1]
        )
        momentum_in, momentum_out = self._gross_boundary_flow(
            float(momentum_left), float(momentum_right)
        )
        external = BoundaryExchange(
            liquid_inflow_m3_s=liquid_in,
            liquid_outflow_m3_s=liquid_out,
            gas_inflow_kg_s=gas_in,
            gas_outflow_kg_s=gas_out,
            momentum_x_in_N=momentum_in,
            momentum_x_out_N=momentum_out,
            external_force_x_N=external_force_x,
        )
        delta = HorizontalDelta(_tuple(dAl), _tuple(dQl), _tuple(dMg), _tuple(dJg))
        self._audit_component_rate(delta, geometry, gross, external)
        return delta, external

    def _project_first_gas_entry_port_fluxes(
        self,
        state: HorizontalState,
        gross: Mapping[PortKey, GrossNodePortFlux],
    ) -> dict[PortKey, GrossNodePortFlux]:
        """Make a gas-occupied T aperture displace, rather than inject, water."""

        cells = {key: cell for key, cell, _ in self._all_port_cells()}
        result = dict(gross)
        for key, flux in gross.items():
            cell = cells[key]
            full_water = (
                state.Mg[cell] <= self.config.gas_presence_mass_kg_m
                and self.area_m2 - state.Al[cell]
                <= 5.0e-13 * self.area_m2
            )
            incoming_gas = (
                flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
            )
            if not full_water or incoming_gas <= 0.0:
                continue
            normal_x = 1.0 if key.port_name == "main_left" else -1.0
            gas_advective = (
                flux.gas_into_node_kg_s * flux.gas_into_node_speed_m_s
                + flux.gas_out_of_node_kg_s * flux.gas_out_of_node_speed_m_s
            )
            result[key] = GrossNodePortFlux(
                key=key,
                gas_into_node_kg_s=flux.gas_into_node_kg_s,
                gas_out_of_node_kg_s=flux.gas_out_of_node_kg_s,
                gas_into_node_speed_m_s=flux.gas_into_node_speed_m_s,
                gas_out_of_node_speed_m_s=flux.gas_out_of_node_speed_m_s,
                advective_momentum_to_node_x_N=normal_x * gas_advective,
                pressure_traction_to_node_x_N=flux.pressure_traction_to_node_x_N,
            )
        return result

    def _audit_component_rate(
        self,
        delta: HorizontalDelta,
        geometry: CoupledGeometry,
        gross: Mapping[PortKey, GrossNodePortFlux],
        external: BoundaryExchange,
    ) -> None:
        dx = geometry.horizontal_dx_m
        rho = self.config.liquid_density_kg_m3
        observed_liquid = sum(value * width for value, width in zip(delta.Al, dx, strict=True))
        observed_gas = sum(value * width for value, width in zip(delta.Mg, dx, strict=True))
        observed_px = sum(
            (rho * q_rate + gas_rate) * width
            for q_rate, gas_rate, width in zip(delta.Ql, delta.Jg, dx, strict=True)
        )
        expected_liquid = external.liquid_volume_net_rate + sum(
            flux.liquid_out_of_node_m3_s - flux.liquid_into_node_m3_s
            for flux in gross.values()
        )
        expected_gas = external.gas_mass_net_rate + sum(
            flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
            for flux in gross.values()
        )
        expected_px = external.mixture_momentum_x_net_rate - sum(
            flux.mixture_momentum_to_node_x_N for flux in gross.values()
        )
        checks = (
            ("liquid", observed_liquid, expected_liquid, 2.0e-11),
            ("gas", observed_gas, expected_gas, 2.0e-11),
            ("Px", observed_px, expected_px, 2.0e-7),
        )
        failed = [
            f"{name}={observed - expected:.6e}"
            for name, observed, expected, tolerance in checks
            if not math.isclose(observed, expected, rel_tol=2.0e-11, abs_tol=tolerance)
        ]
        if failed:
            raise ConservationError(
                "horizontal component/port stage ledger failed: " + ", ".join(failed)
            )

    def stable_timestep_s(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        *,
        trials: tuple[TNodeTrial, ...] | None = None,
    ) -> float:
        self._validate_geometry(state, geometry)
        maximum_speed = self.config.gas_sound_speed_m_s
        for area, discharge, mass, momentum in zip(
            state.Al, state.Ql, state.Mg, state.Jg, strict=True
        ):
            liquid_velocity = 0.0 if area <= 0.0 else abs(discharge / area)
            maximum_speed = max(
                maximum_speed,
                liquid_velocity + self.adapter.celerity_m_s(area),
            )
            if mass > self.config.gas_presence_mass_kg_m:
                maximum_speed = max(
                    maximum_speed,
                    abs(momentum / mass) + self.config.gas_sound_speed_m_s,
                )
        if trials is not None:
            for trial in trials:
                for flux in trial.gross_fluxes:
                    maximum_speed = max(
                        maximum_speed,
                        flux.liquid_into_node_speed_m_s,
                        flux.liquid_out_of_node_speed_m_s,
                        flux.gas_into_node_speed_m_s + self.config.gas_sound_speed_m_s
                        if flux.gas_into_node_kg_s > 0.0
                        else 0.0,
                        flux.gas_out_of_node_speed_m_s + self.config.gas_sound_speed_m_s
                        if flux.gas_out_of_node_kg_s > 0.0
                        else 0.0,
                    )
        return self.config.cfl * min(geometry.horizontal_dx_m) / maximum_speed

    def _capacity_rejection(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        delta: HorizontalDelta,
        trials: tuple[TNodeTrial, ...],
        gross: Mapping[PortKey, GrossNodePortFlux],
    ) -> CapacityReject | None:
        dt = trials[0].dt_s
        maximum_area = self.area_m2 * (
            1.0 + geometry.horizontal_elastic_overarea_fraction
        )
        candidate_al = np.asarray(state.Al) + dt * np.asarray(delta.Al)
        candidate_ql = np.asarray(state.Ql) + dt * np.asarray(delta.Ql)
        candidate_mg = np.asarray(state.Mg) + dt * np.asarray(delta.Mg)
        candidate_jg = np.asarray(state.Jg) + dt * np.asarray(delta.Jg)
        if any(
            np.any(~np.isfinite(values))
            for values in (candidate_al, candidate_ql, candidate_mg, candidate_jg)
        ):
            return CapacityReject(
                component_id=self.component_id,
                reason_code="nonfinite_trial",
                detail="horizontal candidate contains a non-finite conservative value",
                requested_dt_s=dt,
                retryable=True,
            )
        if np.any(candidate_al < 0.0) or np.any(candidate_al > maximum_area) or np.any(candidate_mg < 0.0):
            return CapacityReject(
                component_id=self.component_id,
                reason_code="phase_capacity",
                detail="horizontal candidate exceeds liquid-area or gas-mass capacity",
                requested_dt_s=dt,
                retryable=True,
            )
        gas_area = np.maximum(self.area_m2 - candidate_al, 0.0)
        area_tol = 5.0e-13 * self.area_m2
        mass_tol = self.config.gas_presence_mass_kg_m
        for index, (ag, mg, jg) in enumerate(
            zip(gas_area, candidate_mg, candidate_jg, strict=True)
        ):
            if mg > mass_tol and ag <= area_tol:
                return CapacityReject(
                    component_id=self.component_id,
                    reason_code="void_mass_pairing",
                    detail=f"cell {index} would receive gas mass without positive Ag",
                    requested_dt_s=dt,
                    retryable=False,
                )
            if ag > area_tol and mg <= mass_tol:
                return CapacityReject(
                    component_id=self.component_id,
                    reason_code="void_mass_pairing",
                    detail=f"cell {index} would create a massless void",
                    requested_dt_s=dt,
                    retryable=False,
                )
            if mg <= mass_tol and abs(jg) > 1.0e-10:
                return CapacityReject(
                    component_id=self.component_id,
                    reason_code="void_mass_pairing",
                    detail=f"cell {index} would carry gas momentum without gas mass",
                    requested_dt_s=dt,
                    retryable=False,
                )
            if mg > mass_tol and ag > area_tol:
                p_gas = mg * self.config.rt_J_kg / ag
                if not math.isfinite(p_gas) or p_gas <= 0.0:
                    return CapacityReject(
                        component_id=self.component_id,
                        reason_code="nonfinite_trial",
                        detail=f"cell {index} would have invalid isothermal gas pressure",
                        requested_dt_s=dt,
                        retryable=True,
                    )

        cell_by_key = {key: cell for key, cell, _ in self._all_port_cells()}
        for trial in trials:
            for flux in trial.gross_fluxes:
                if flux.key not in cell_by_key:
                    continue
                cell = cell_by_key[flux.key]
                initially_full = (
                    state.Mg[cell] <= mass_tol
                    and self.area_m2 - state.Al[cell] <= area_tol
                )
                net_gas_in = flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
                if not initially_full or net_gas_in <= 0.0:
                    continue
                rho_node = trial.common_absolute_pressure_Pa / self.config.rt_J_kg
                required_opening_m3_s = net_gas_in / rho_node
                actual_opening_m3_s = gas_area[cell] * geometry.horizontal_dx_m[cell] / dt
                if not math.isclose(
                    actual_opening_m3_s,
                    required_opening_m3_s,
                    rel_tol=2.0e-10,
                    abs_tol=2.0e-13,
                ):
                    return CapacityReject(
                        component_id=self.component_id,
                        reason_code="void_mass_pairing",
                        detail=(
                            f"first gas entry at {flux.key.label} does not displace "
                            "the same liquid volume in the same stage"
                        ),
                        requested_dt_s=dt,
                        retryable=False,
                    )
        return None

    def propose_joint_stage(
        self,
        state: HorizontalState,
        geometry: CoupledGeometry,
        *,
        air_node_trial: TNodeTrial,
        riser_node_trial: TNodeTrial,
        physical_stage: PhysicalStage,
        dt_s: float,
    ) -> ComponentStageProposal:
        trials = (air_node_trial, riser_node_trial)
        validate_trial_set(trials)
        if {trial.node_name for trial in trials} != {"air_supply_T", "riser_T"}:
            raise ContractViolation("horizontal component requires one trial from each T node")
        if physical_stage != air_node_trial.physical_stage:
            raise ContractViolation("horizontal physical stage differs from the T-node trials")
        if not math.isclose(float(dt_s), air_node_trial.dt_s, rel_tol=0.0, abs_tol=0.0):
            raise ContractViolation("horizontal dt differs from the T-node trials")
        self._validate_geometry(state, geometry)

        stable = self.stable_timestep_s(state, geometry, trials=trials)
        if dt_s > stable * (1.0 + 1.0e-12):
            rejection = CapacityReject(
                component_id=self.component_id,
                reason_code="cfl",
                detail=f"requested dt {dt_s:.6e} exceeds horizontal CFL {stable:.6e}",
                requested_dt_s=dt_s,
                retryable=True,
                maximum_admissible_dt_s=stable,
            )
            return ComponentStageProposal.rejected(
                component_id=self.component_id,
                base_state_token=air_node_trial.base_state_token,
                trials=trials,
                rejection=rejection,
                evidence_status=F0_STAGE_RATE_EVIDENCE,
            )

        capillary_rejection = self._validate_capillary_interfaces(state, trials)
        if capillary_rejection is not None:
            return ComponentStageProposal.rejected(
                component_id=self.component_id,
                base_state_token=air_node_trial.base_state_token,
                trials=trials,
                rejection=capillary_rejection,
                evidence_status=F0_STAGE_RATE_EVIDENCE,
            )
        traces = self._validate_trial_traces(state, geometry, trials)
        gross = {
            flux.key: flux
            for trial in trials
            for flux in trial.gross_fluxes
            if flux.key in traces
        }
        if set(gross) != set(traces):
            raise ContractViolation("horizontal component does not own exactly four T ports")
        gross = self._project_first_gas_entry_port_fluxes(state, gross)
        node_gas_density_by_key = {
            trace.key: trial.common_absolute_pressure_Pa / self.config.rt_J_kg
            for trial in trials
            for trace in trial.port_traces
            if trace.component_id == self.component_id
        }

        delta, external = self._rate_and_external(
            state,
            geometry,
            traces,
            gross,
            node_gas_density_by_key,
            dt_s=dt_s,
        )
        capacity_rejection = self._capacity_rejection(
            state, geometry, delta, trials, gross
        )
        if capacity_rejection is not None:
            return ComponentStageProposal.rejected(
                component_id=self.component_id,
                base_state_token=air_node_trial.base_state_token,
                trials=trials,
                rejection=capacity_rejection,
                evidence_status=F0_STAGE_RATE_EVIDENCE,
            )
        proposal = ComponentStageProposal.accepted(
            component_id=self.component_id,
            base_state_token=air_node_trial.base_state_token,
            trials=trials,
            delta=delta,
            accepted_gross_fluxes=tuple(gross[key] for key in sorted(gross)),
            external_exchange=external,
            evidence_status=F0_STAGE_RATE_EVIDENCE,
        )
        return proposal

    def evaluate_trial(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> ComponentStageProposal:
        """Adapter for ``evaluate_component_trial_pure`` contract tests."""

        validate_trial_set(trials)
        by_name = {trial.node_name: trial for trial in trials}
        if set(by_name) != {"air_supply_T", "riser_T"}:
            raise ContractViolation("horizontal pure evaluation needs both T-node trials")
        first = trials[0]
        return self.propose_joint_stage(
            state.horizontal,
            geometry,
            air_node_trial=by_name["air_supply_T"],
            riser_node_trial=by_name["riser_T"],
            physical_stage=first.physical_stage,
            dt_s=first.dt_s,
        )


__all__ = [
    "CIRCULAR_3D_CAPILLARY_MODE",
    "CapillaryGeometryMode",
    "F0HorizontalTwoTeeStageComponent",
    "F0_STAGE_RATE_EVIDENCE",
    "HORIZONTAL_COMPONENT_ID",
    "HorizontalF0Readiness",
    "PLANAR_2D_CAPILLARY_MODE",
]
