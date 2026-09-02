"""Source-aligned two-phase model of the finite S1 air-supply branch.

The OpenFOAM geometry contains a real vertical branch between the horizontal
pipe crown (``z=0.0127 m``) and the pressure inlet (``z=0.1500 m``).  It is
therefore not legal to replace that 0.1373 m water-filled volume by a
zero-length pressure source.  This module owns fourteen conservative cell
averages ``(Al, Ql, Mg, Jg)`` through the canonical
:class:`model.state.SupplyBranchState`.

The component is a conservative sharp-interface finite-volume translation:

* liquid volume and gas mass are advanced only through boundary fluxes;
* gas occupies a contiguous segment measured downwards from the top;
* each occupied cell satisfies ``Ag = A - Al`` and ``Mg > 0 iff Ag > 0``;
* Stage 1 has a material wall at the branch top;
* Stage 2 uses the published 5700 Pa-gauge, pure-air pressure reservoir;
* the bottom exchange is returned as one explicit gross liquid/gas packet,
  ready to participate in a global zero-storage T-node transaction.

The liquid circular geometry, pressure flux and celerity are read through the
hash-pinned Case-1 adapter. Gas uses the standard isothermal Euler HLL flux.
Both occupied plugs use the preregistered F0 smooth-pipe Darcy law, applied by
the same sign-preserving semi-implicit velocity relaxation used by the S1
riser. The paper does not publish the experimental valve/line-loss law, so
this component deliberately reports ``alignment_ready`` and
``production_ready`` as false. Passing its conservation/smoke tests is not an
accepted S1 trajectory or evidence of eruption.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

from .errors import ContractViolation
from .flux import BoundaryExchange, SupplyBranchDelta
from .horizontal_case1_adapter import Case1HorizontalLiquidAdapter
from .port_contracts import (
    CapacityReject,
    CapillaryInterfaceOwnership,
    ComponentStageProposal,
    GrossNodePortFlux,
    PortKey,
    PortTraceState,
    TNodeTrial,
    validate_trial_set,
)
from .pressure_reservoir import IsothermalIdealGasPressureReservoir
from .state import CoupledGeometry, CoupledState, SupplyBranchState


Stage = Literal["stage1_closed", "stage2_pressure_reservoir"]

SUPPLY_BRANCH_DIAMETER_M = 0.0254
SUPPLY_BRANCH_BOTTOM_Z_M = 0.0127
SUPPLY_BRANCH_TOP_Z_M = 0.1500
SUPPLY_BRANCH_LENGTH_M = SUPPLY_BRANCH_TOP_Z_M - SUPPLY_BRANCH_BOTTOM_Z_M
SUPPLY_BRANCH_CELL_COUNT = 14
SUPPLY_BRANCH_ALLOWED_CELL_COUNTS = (14, 28)
INITIAL_WATER_SURFACE_Z_M = 0.5842
PUBLISHED_GAS_GAUGE_PRESSURE_PA = 5700.0

CLOSURE_PROVENANCE = (
    "declared_F0_smooth_pipe_Darcy_sharp_interface_piston_and_isothermal_HLL__"
    "not_published_not_tuned"
)
GEOMETRY_PROVENANCE = "source_aligned_2D_geometry_z0p0127_to_z0p1500"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ContractViolation(f"{name} must be non-negative")
    return result


def _state_token(state: SupplyBranchState) -> str:
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SupplyBranchGeometry:
    """Frozen S1 supply-branch geometry and declared 1-D grid."""

    diameter_m: float = SUPPLY_BRANCH_DIAMETER_M
    z_bottom_m: float = SUPPLY_BRANCH_BOTTOM_Z_M
    z_top_m: float = SUPPLY_BRANCH_TOP_Z_M
    cell_count: int = SUPPLY_BRANCH_CELL_COUNT
    evidence_status: str = GEOMETRY_PROVENANCE

    def __post_init__(self) -> None:
        for name, expected in (
            ("diameter_m", SUPPLY_BRANCH_DIAMETER_M),
            ("z_bottom_m", SUPPLY_BRANCH_BOTTOM_Z_M),
            ("z_top_m", SUPPLY_BRANCH_TOP_Z_M),
        ):
            value = _finite(name, getattr(self, name))
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ContractViolation(
                    f"source-aligned supply geometry drifted: {name}={value}, "
                    f"expected {expected}"
                )
            object.__setattr__(self, name, value)
        if (
            int(self.cell_count) != self.cell_count
            or self.cell_count not in SUPPLY_BRANCH_ALLOWED_CELL_COUNTS
        ):
            raise ContractViolation(
                "the frozen S1 supply grid must contain exactly 14 or 28 cells"
            )
        if not self.evidence_status.strip():
            raise ContractViolation("geometry evidence status must be non-empty")

    @property
    def length_m(self) -> float:
        return self.z_top_m - self.z_bottom_m

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def dz_m(self) -> float:
        return self.length_m / self.cell_count

    @property
    def cell_volume_m3(self) -> float:
        return self.area_m2 * self.dz_m

    @property
    def total_volume_m3(self) -> float:
        return self.area_m2 * self.length_m


@dataclass(frozen=True, slots=True)
class SupplyBranchConfig:
    """Published/declared physical constants, with no result fitting knobs."""

    liquid_density_kg_m3: float = 998.4
    liquid_viscosity_Pa_s: float = 1.002e-3
    gas_viscosity_Pa_s: float = 1.78e-5
    atmospheric_pressure_Pa: float = 101325.0
    gas_constant_J_kg_K: float = 287.05
    temperature_K: float = 293.15
    initial_water_surface_z_m: float = INITIAL_WATER_SURFACE_Z_M
    stage2_top_alpha_water: float = 0.0
    cfl: float = 0.35
    maximum_substeps: int = 200000
    closure_provenance: str = CLOSURE_PROVENANCE

    def __post_init__(self) -> None:
        for name in (
            "liquid_density_kg_m3",
            "liquid_viscosity_Pa_s",
            "gas_viscosity_Pa_s",
            "atmospheric_pressure_Pa",
            "gas_constant_J_kg_K",
            "temperature_K",
            "initial_water_surface_z_m",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ContractViolation(f"{name} must be positive")
            object.__setattr__(self, name, value)
        cfl = _finite("cfl", self.cfl)
        if not 0.0 < cfl < 0.5:
            raise ContractViolation("supply-branch cfl must lie in (0, 0.5)")
        object.__setattr__(self, "cfl", cfl)
        alpha = _finite("stage2_top_alpha_water", self.stage2_top_alpha_water)
        if alpha != 0.0:
            raise ContractViolation("the published Stage-2 supply boundary is pure air")
        object.__setattr__(self, "stage2_top_alpha_water", alpha)
        if int(self.maximum_substeps) != self.maximum_substeps or self.maximum_substeps < 1:
            raise ContractViolation("maximum_substeps must be a positive integer")
        if not self.closure_provenance.strip():
            raise ContractViolation("closure provenance must be non-empty")

    @property
    def rt_J_kg(self) -> float:
        return self.gas_constant_J_kg_K * self.temperature_K

    @property
    def gas_sound_speed_m_s(self) -> float:
        return math.sqrt(self.rt_J_kg)


@dataclass(frozen=True, slots=True)
class SupplyBottomNodeCondition:
    """Algebraic state supplied by the global air T node.

    ``wall`` is useful for closed-component conservation tests.  During the
    experiment the lower connection is open.  ``gas_accepting`` is consulted
    only after the gas front reaches the bottom; before that instant the
    bottom packet necessarily carries liquid, not gas.
    """

    absolute_pressure_Pa: float
    wall: bool = False
    gas_accepting: bool = True
    gas_velocity_upward_m_s: float = 0.0

    def __post_init__(self) -> None:
        pressure = _finite("bottom-node absolute pressure", self.absolute_pressure_Pa)
        if pressure <= 0.0:
            raise ContractViolation("bottom-node absolute pressure must be positive")
        object.__setattr__(self, "absolute_pressure_Pa", pressure)
        object.__setattr__(
            self,
            "gas_velocity_upward_m_s",
            _finite("bottom-node gas velocity", self.gas_velocity_upward_m_s),
        )


@dataclass(frozen=True, slots=True)
class SupplyBranchGrossFluxPacket:
    """One explicit, immutable bottom-port transaction proposal.

    Upward rates enter the supply branch; downward rates leave it for the
    horizontal-pipe T node.  Gross directions remain separate even though the
    signed diagnostics are convenient for ledgers.
    """

    transaction_id: str
    base_state_token: str
    dt_s: float
    bottom_absolute_pressure_Pa: float
    bottom_momentum_flux_upward_N: float
    liquid_upward_rate_m3_s: float = 0.0
    liquid_downward_rate_m3_s: float = 0.0
    gas_upward_mass_rate_kg_s: float = 0.0
    gas_downward_mass_rate_kg_s: float = 0.0
    liquid_upward_speed_m_s: float = 0.0
    liquid_downward_speed_m_s: float = 0.0
    gas_upward_speed_m_s: float = 0.0
    gas_downward_speed_m_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.transaction_id.strip() or not self.base_state_token.strip():
            raise ContractViolation("gross packet transaction and state token are required")
        dt = _finite("gross packet dt_s", self.dt_s)
        pressure = _finite("gross packet pressure", self.bottom_absolute_pressure_Pa)
        if dt <= 0.0 or pressure <= 0.0:
            raise ContractViolation("gross packet dt and pressure must be positive")
        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "bottom_absolute_pressure_Pa", pressure)
        object.__setattr__(
            self,
            "bottom_momentum_flux_upward_N",
            _finite(
                "bottom momentum flux", self.bottom_momentum_flux_upward_N
            ),
        )
        rate_speed_pairs = (
            ("liquid_upward", self.liquid_upward_rate_m3_s, self.liquid_upward_speed_m_s),
            ("liquid_downward", self.liquid_downward_rate_m3_s, self.liquid_downward_speed_m_s),
            ("gas_upward", self.gas_upward_mass_rate_kg_s, self.gas_upward_speed_m_s),
            ("gas_downward", self.gas_downward_mass_rate_kg_s, self.gas_downward_speed_m_s),
        )
        for label, raw_rate, raw_speed in rate_speed_pairs:
            rate = _nonnegative(f"{label} rate", raw_rate)
            speed = _nonnegative(f"{label} speed", raw_speed)
            if rate == 0.0 and speed != 0.0:
                raise ContractViolation(f"{label} speed requires a positive gross rate")
            if rate > 0.0 and speed == 0.0:
                raise ContractViolation(f"{label} gross rate requires a positive speed")
        for name in (
            "liquid_upward_rate_m3_s",
            "liquid_downward_rate_m3_s",
            "gas_upward_mass_rate_kg_s",
            "gas_downward_mass_rate_kg_s",
            "liquid_upward_speed_m_s",
            "liquid_downward_speed_m_s",
            "gas_upward_speed_m_s",
            "gas_downward_speed_m_s",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def liquid_net_into_branch_m3_s(self) -> float:
        return self.liquid_upward_rate_m3_s - self.liquid_downward_rate_m3_s

    @property
    def gas_net_into_branch_kg_s(self) -> float:
        return self.gas_upward_mass_rate_kg_s - self.gas_downward_mass_rate_kg_s


@dataclass(frozen=True, slots=True)
class SupplyBranchInventory:
    liquid_volume_m3: float
    gas_volume_m3: float
    gas_mass_kg: float
    liquid_momentum_kg_m_s: float
    gas_momentum_kg_m_s: float

    @property
    def mixture_momentum_kg_m_s(self) -> float:
        return self.liquid_momentum_kg_m_s + self.gas_momentum_kg_m_s


@dataclass(frozen=True, slots=True)
class SupplyBranchWallDiagnostics:
    """Frozen F0 wall-shear audit for one pure component proposal."""

    liquid_darcy_factor: float
    gas_darcy_factor: float
    liquid_reynolds: float
    gas_reynolds: float
    liquid_wall_impulse_kg_m_s: float
    gas_wall_impulse_kg_m_s: float

    @property
    def total_wall_impulse_kg_m_s(self) -> float:
        return self.liquid_wall_impulse_kg_m_s + self.gas_wall_impulse_kg_m_s


@dataclass(frozen=True, slots=True)
class SupplyBranchLedgerEntry:
    stage: Stage
    before: SupplyBranchInventory
    after: SupplyBranchInventory
    top_gas_net_into_branch_kg: float
    bottom_liquid_net_into_branch_m3: float
    bottom_gas_net_into_branch_kg: float
    top_momentum_impulse_kg_m_s: float
    bottom_momentum_impulse_kg_m_s: float
    gravity_impulse_kg_m_s: float
    wall_momentum_impulse_kg_m_s: float
    liquid_wall_impulse_kg_m_s: float
    gas_wall_impulse_kg_m_s: float
    liquid_darcy_factor: float
    gas_darcy_factor: float
    acoustic_projection_impulse_kg_m_s: float
    liquid_volume_residual_m3: float
    gas_mass_residual_kg: float
    phase_volume_residual_m3: float
    mixture_momentum_residual_kg_m_s: float
    interface_recoil_residual_kg_m_s: float
    maximum_courant: float
    minimum_liquid_area_m2: float
    minimum_gas_mass_kg_m: float


@dataclass(frozen=True, slots=True)
class SupplyBranchStepResult:
    """Pure proposal: no shared state is mutated until the global commit."""

    state: SupplyBranchState
    delta: SupplyBranchDelta
    bottom: SupplyBranchGrossFluxPacket
    ledger: SupplyBranchLedgerEntry


@dataclass(frozen=True, slots=True)
class SupplyBranchAdvanceResult:
    state: SupplyBranchState
    packets: tuple[SupplyBranchGrossFluxPacket, ...]
    ledger: tuple[SupplyBranchLedgerEntry, ...]


def _isothermal_hll(
    rho_left: float,
    velocity_left: float,
    rho_right: float,
    velocity_right: float,
    rt: float,
) -> tuple[float, float]:
    """Isothermal Euler mass and momentum flux per unit connection area."""

    rho_l = _nonnegative("HLL left density", rho_left)
    rho_r = _nonnegative("HLL right density", rho_right)
    u_l = _finite("HLL left velocity", velocity_left)
    u_r = _finite("HLL right velocity", velocity_right)
    c = math.sqrt(rt)
    state_l = (rho_l, rho_l * u_l)
    state_r = (rho_r, rho_r * u_r)
    flux_l = (rho_l * u_l, rho_l * u_l * u_l + rho_l * rt)
    flux_r = (rho_r * u_r, rho_r * u_r * u_r + rho_r * rt)
    speed_l = min(u_l - c, u_r - c)
    speed_r = max(u_l + c, u_r + c)
    if speed_l >= 0.0:
        return flux_l
    if speed_r <= 0.0:
        return flux_r
    inverse = 1.0 / (speed_r - speed_l)
    return (
        (
            speed_r * flux_l[0]
            - speed_l * flux_r[0]
            + speed_l * speed_r * (state_r[0] - state_l[0])
        )
        * inverse,
        (
            speed_r * flux_l[1]
            - speed_l * flux_r[1]
            + speed_l * speed_r * (state_r[1] - state_l[1])
        )
        * inverse,
    )


def f0_supply_smooth_pipe_darcy_factor(reynolds: float) -> float:
    """Preregistered F0 smooth-pipe Darcy law with its frozen transition."""

    re = _nonnegative("supply Re", reynolds)
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


class SupplyBranchTwoPhaseSolver:
    """Four-field, sharp-interface S1 supply-branch component."""

    component_id = "air_supply_branch"

    def __init__(
        self,
        adapter: Case1HorizontalLiquidAdapter | None = None,
        *,
        geometry: SupplyBranchGeometry | None = None,
        config: SupplyBranchConfig | None = None,
        pressure_reservoir: IsothermalIdealGasPressureReservoir | None = None,
    ) -> None:
        self.adapter = Case1HorizontalLiquidAdapter() if adapter is None else adapter
        self.geometry = SupplyBranchGeometry() if geometry is None else geometry
        self.config = SupplyBranchConfig() if config is None else config
        self.pressure_reservoir = (
            IsothermalIdealGasPressureReservoir()
            if pressure_reservoir is None
            else pressure_reservoir
        )
        if not math.isclose(
            self.adapter.grid.diameter_m,
            self.geometry.diameter_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ContractViolation("Case-1 liquid adapter and supply branch diameters differ")
        if not math.isclose(
            self.adapter.full_area_m2,
            self.geometry.area_m2,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ContractViolation("Case-1 circular area and supply branch area differ")
        expected_pressure = self.config.atmospheric_pressure_Pa + PUBLISHED_GAS_GAUGE_PRESSURE_PA
        if not math.isclose(
            self.pressure_reservoir.reservoir_absolute_pressure_Pa,
            expected_pressure,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ContractViolation("Stage-2 supply pressure must remain 5700 Pa gauge")

    @property
    def alignment_ready(self) -> bool:
        return False

    @property
    def production_ready(self) -> bool:
        return False

    @property
    def joint_trial_ready(self) -> bool:
        """The finite branch can participate in a pure common-node trial.

        This component-level flag does not authorize a trajectory.  It only
        states that the existing pressure-reservoir/piston solver can consume
        one immutable ``air_supply_T`` pressure trial and return a conservative
        rate plus the accepted gross bottom packet.
        """

        return True

    @property
    def stage1_top_boundary(self) -> str:
        return "impermeable_wall"

    @property
    def stage2_top_boundary(self) -> str:
        return "5700Pa_gauge_pure_air_pressure_Riemann"

    @property
    def source_hydrostatic_bottom_pressure_Pa(self) -> float:
        return self.config.atmospheric_pressure_Pa + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * (self.config.initial_water_surface_z_m - self.geometry.z_bottom_m)
        )

    def default_bottom_condition(self) -> SupplyBottomNodeCondition:
        return SupplyBottomNodeCondition(
            absolute_pressure_Pa=self.source_hydrostatic_bottom_pressure_Pa
        )

    def _validate_coupled_geometry(
        self, state: SupplyBranchState, geometry: CoupledGeometry
    ) -> None:
        if state.cell_count != self.geometry.cell_count:
            raise ContractViolation("supply component and state cell counts differ")
        if len(geometry.supply_branch_dz_m) != state.cell_count:
            raise ContractViolation("supply state and coupled grid cell counts differ")
        if geometry.supply_branch_area_m2 is None or not math.isclose(
            geometry.supply_branch_area_m2,
            self.geometry.area_m2,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ContractViolation("supply component and coupled branch areas differ")
        if any(
            not math.isclose(
                dz, self.geometry.dz_m, rel_tol=0.0, abs_tol=1.0e-14
            )
            for dz in geometry.supply_branch_dz_m
        ):
            raise ContractViolation("supply component requires its frozen uniform grid")
        if not math.isclose(
            geometry.liquid_density_kg_m3,
            self.config.liquid_density_kg_m3,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ContractViolation("supply and coupled liquid densities differ")
        self.validate_state(state)

    def bottom_trace_pressure_Pa(self, state: SupplyBranchState) -> float:
        """Return the component-side bottom pressure before a node correction.

        The sharp interface is ordered gas-over-water.  Once gas exists, its
        state-owned isothermal pressure plus the hydrostatic liquid-plug head
        provides the outgoing bottom trace.  The source-aligned all-water
        state uses the declared initial hydrostatic datum.  No node trial or
        comparison result enters this reconstruction.
        """

        inventory = self.inventory(state)
        gas_pressure = self.gas_pressure_Pa(state)
        if gas_pressure is None:
            return self.source_hydrostatic_bottom_pressure_Pa
        liquid_height = inventory.liquid_volume_m3 / self.geometry.area_m2
        pressure = gas_pressure + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * liquid_height
        )
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ContractViolation("supply bottom trace pressure is invalid")
        return pressure

    def port_trace(
        self,
        state: SupplyBranchState,
        geometry: CoupledGeometry,
        *,
        interface: CapillaryInterfaceOwnership | None = None,
    ) -> PortTraceState:
        """Expose the physical lower face of the gas-over-water sharp plug."""

        self._validate_coupled_geometry(state, geometry)
        area = self.geometry.area_m2
        # Gas descends from the branch top.  A fractional bottom cell still
        # presents liquid at its lower face; gas reaches the T only when that
        # cell is fully gas.  Cell-average phase fractions are not substituted
        # for this sharp face topology.
        bottom_liquid_present = state.Al[0] > 64.0 * math.ulp(area)
        liquid_area = area if bottom_liquid_present else 0.0
        gas_area = area - liquid_area
        jump = (
            0.0
            if interface is None
            or interface.pressure_jump_gas_minus_liquid_Pa is None
            else interface.pressure_jump_gas_minus_liquid_Pa
        )
        trace_pressure = self.bottom_trace_pressure_Pa(state)
        if gas_area > 0.0:
            gas_pressure = self.gas_pressure_Pa(state)
            if gas_pressure is None:
                raise ContractViolation("gas bottom face has no state-owned pressure")
            liquid_pressure = gas_pressure - jump
            gas_density = gas_pressure / self.config.rt_J_kg
        else:
            liquid_pressure = trace_pressure
            gas_pressure = liquid_pressure + jump
            gas_density = gas_pressure / self.config.rt_J_kg
        if min(liquid_pressure, gas_pressure, gas_density) <= 0.0:
            raise ContractViolation("supply port trace has a non-positive phase state")
        liquid_velocity = (
            0.0 if liquid_area == 0.0 else state.Ql[0] / max(state.Al[0], area)
        )
        gas_velocity = 0.0 if state.Mg[0] <= 0.0 else state.Jg[0] / state.Mg[0]
        return PortTraceState(
            key=PortKey("air_supply_T", "supply_bottom"),
            component_id=self.component_id,
            normal_into_node_x=0.0,
            normal_into_node_z=-1.0,
            full_area_m2=area,
            liquid_area_m2=liquid_area,
            gas_area_m2=gas_area,
            liquid_density_kg_m3=self.config.liquid_density_kg_m3,
            gas_density_kg_m3=gas_density,
            liquid_absolute_pressure_Pa=liquid_pressure,
            gas_absolute_pressure_Pa=gas_pressure,
            liquid_axial_velocity_m_s=liquid_velocity,
            gas_axial_velocity_m_s=gas_velocity,
            interface_id=None if interface is None else interface.interface_id,
            evidence_status=(
                "source_aligned_finite_supply_branch__sharp_bottom_face__"
                "state_owned_isothermal_gas_and_hydrostatic_liquid_trace"
            ),
        )

    def initial_state(self) -> SupplyBranchState:
        """Return the source-aligned all-water, zero-velocity branch."""

        count = self.geometry.cell_count
        area = self.geometry.area_m2
        state = SupplyBranchState(
            Al=(area,) * count,
            Ql=(0.0,) * count,
            Mg=(0.0,) * count,
            Jg=(0.0,) * count,
        )
        self.validate_state(state)
        return state

    def inventory(self, state: SupplyBranchState) -> SupplyBranchInventory:
        self.validate_state(state)
        dz = self.geometry.dz_m
        area = self.geometry.area_m2
        liquid_volume = sum(state.Al) * dz
        gas_volume = sum(area - value for value in state.Al) * dz
        gas_mass = sum(state.Mg) * dz
        liquid_momentum = self.config.liquid_density_kg_m3 * sum(state.Ql) * dz
        gas_momentum = sum(state.Jg) * dz
        return SupplyBranchInventory(
            liquid_volume_m3=liquid_volume,
            gas_volume_m3=gas_volume,
            gas_mass_kg=gas_mass,
            liquid_momentum_kg_m_s=liquid_momentum,
            gas_momentum_kg_m_s=gas_momentum,
        )

    @staticmethod
    def delta(
        before: SupplyBranchState, after: SupplyBranchState
    ) -> SupplyBranchDelta:
        """Return the canonical delta consumed by ``AtomicFluxPacket``."""

        if before.cell_count != after.cell_count:
            raise ContractViolation("supply state/delta cell counts differ")
        return SupplyBranchDelta(
            Al=tuple(b - a for a, b in zip(before.Al, after.Al, strict=True)),
            Ql=tuple(b - a for a, b in zip(before.Ql, after.Ql, strict=True)),
            Mg=tuple(b - a for a, b in zip(before.Mg, after.Mg, strict=True)),
            Jg=tuple(b - a for a, b in zip(before.Jg, after.Jg, strict=True)),
        )

    def validate_state(self, state: SupplyBranchState) -> None:
        if state.cell_count != self.geometry.cell_count:
            raise ContractViolation("supply-branch state/grid cell counts differ")
        area = self.geometry.area_m2
        gas_seen = False
        fractional = 0
        liquid_velocities: list[float] = []
        gas_velocities: list[float] = []
        gas_pressures: list[float] = []
        for index, (al, ql, mg, jg) in enumerate(
            zip(state.Al, state.Ql, state.Mg, state.Jg, strict=True)
        ):
            if not 0.0 <= al <= area:
                raise ContractViolation(f"supply cell {index} liquid area lies outside [0,A]")
            ag = area - al
            # This is deliberately exact.  A finite void with zero mass is a
            # vacuum cell, while finite mass at Ag=0 is an infinite-pressure
            # cell; neither may be hidden behind a tolerance.
            if (ag > 0.0) != (mg > 0.0):
                raise ContractViolation(
                    f"supply cell {index} violates Ag>0 iff finite gas inventory"
                )
            if mg == 0.0 and jg != 0.0:
                raise ContractViolation(f"supply cell {index} has gas momentum in vacuum")
            if al == 0.0 and ql != 0.0:
                raise ContractViolation(f"supply cell {index} has liquid discharge without liquid")
            if 0.0 < al < area:
                fractional += 1
            if gas_seen and ag == 0.0:
                raise ContractViolation("gas cells must form one contiguous segment at the top")
            if ag > 0.0:
                gas_seen = True
                gas_velocities.append(jg / mg)
                gas_pressures.append(mg / ag * self.config.rt_J_kg)
            if al > 0.0:
                liquid_velocities.append(ql / al)
        if fractional > 1:
            raise ContractViolation("sharp-interface branch permits at most one fractional cell")
        for label, values in (
            ("liquid piston velocity", liquid_velocities),
            ("gas plug velocity", gas_velocities),
        ):
            if values and max(values) - min(values) > 2.0e-11 * max(1.0, max(map(abs, values))):
                raise ContractViolation(f"{label} is not uniform across its occupied segment")
        if gas_pressures and max(gas_pressures) - min(gas_pressures) > 2.0e-7 * max(
            gas_pressures
        ):
            raise ContractViolation("sharp-interface gas pressure is not spatially uniform")

    def state_from_bulk(
        self,
        *,
        gas_volume_m3: float,
        gas_mass_kg: float,
        liquid_velocity_upward_m_s: float = 0.0,
        gas_velocity_upward_m_s: float = 0.0,
    ) -> SupplyBranchState:
        """Map conservative bulk inventories to top-contiguous FV cells."""

        volume = _nonnegative("gas volume", gas_volume_m3)
        mass = _nonnegative("gas mass", gas_mass_kg)
        ul = _finite("liquid velocity", liquid_velocity_upward_m_s)
        ug = _finite("gas velocity", gas_velocity_upward_m_s)
        total = self.geometry.total_volume_m3
        tolerance = 32.0 * math.ulp(total)
        if volume > total + tolerance:
            raise ContractViolation("gas volume exceeds the finite supply branch")
        if abs(volume - total) <= tolerance:
            volume = total
        if volume <= tolerance:
            volume = 0.0
        if (volume > 0.0) != (mass > 0.0):
            raise ContractViolation("finite gas volume and finite gas mass must coexist")
        area = self.geometry.area_m2
        dz = self.geometry.dz_m
        density = 0.0 if volume == 0.0 else mass / volume
        remaining = volume
        ag = [0.0] * self.geometry.cell_count
        for index in range(self.geometry.cell_count - 1, -1, -1):
            cell_volume = min(max(remaining, 0.0), self.geometry.cell_volume_m3)
            if cell_volume >= self.geometry.cell_volume_m3 - tolerance:
                gas_area = area
            elif cell_volume <= tolerance:
                gas_area = 0.0
            else:
                gas_area = cell_volume / dz
            ag[index] = gas_area
            remaining -= gas_area * dz
        if abs(remaining) > 4.0 * tolerance:
            raise ContractViolation("sharp-interface cell mapping lost gas volume")
        al = tuple(area - value for value in ag)
        mg = tuple(density * value if value > 0.0 else 0.0 for value in ag)
        state = SupplyBranchState(
            Al=al,
            Ql=tuple(value * ul if value > 0.0 else 0.0 for value in al),
            Mg=mg,
            Jg=tuple(value * ug if value > 0.0 else 0.0 for value in mg),
        )
        self.validate_state(state)
        return state

    def gas_pressure_Pa(self, state: SupplyBranchState) -> float | None:
        inventory = self.inventory(state)
        if inventory.gas_volume_m3 == 0.0:
            return None
        return inventory.gas_mass_kg * self.config.rt_J_kg / inventory.gas_volume_m3

    def _liquid_velocity(self, inventory: SupplyBranchInventory) -> float:
        if inventory.liquid_volume_m3 == 0.0:
            return 0.0
        return inventory.liquid_momentum_kg_m_s / (
            self.config.liquid_density_kg_m3 * inventory.liquid_volume_m3
        )

    @staticmethod
    def _gas_velocity(inventory: SupplyBranchInventory) -> float:
        if inventory.gas_mass_kg == 0.0:
            return 0.0
        return inventory.gas_momentum_kg_m_s / inventory.gas_mass_kg

    def _apply_f0_wall_shear(
        self,
        state: SupplyBranchState,
        dt_s: float,
    ) -> tuple[SupplyBranchState, SupplyBranchWallDiagnostics]:
        """Apply the frozen wall law without reversing either plug velocity.

        The supply reduction is a sharp axial piston: liquid and gas occupy
        full circular sections in series, so each occupied plug has
        ``D_h=D`` and wall length ``V/A``.  For a mixed state the contact
        closure supplies one common provisional velocity.  The combined wall
        force is relaxed semi-implicitly,

        ``u_new = u/(1 + dt*K*abs(u)/M)``,

        where ``K=sum(f_D*rho*P*L/8)``.  This is the same sign-preserving
        discretization as the S1 riser wall step and introduces no tunable
        coefficient.  Phase wall impulses are split by their physical ``K``
        contributions and sum exactly to the mixture impulse.
        """

        self.validate_state(state)
        dt = _finite("supply wall dt_s", dt_s)
        if dt <= 0.0:
            raise ContractViolation("supply wall dt_s must be positive")
        inventory = self.inventory(state)
        liquid_mass = (
            self.config.liquid_density_kg_m3 * inventory.liquid_volume_m3
        )
        gas_mass = inventory.gas_mass_kg
        mixture_mass = liquid_mass + gas_mass
        if mixture_mass <= 0.0:
            raise ContractViolation("finite supply branch has no material mass")

        liquid_velocity = self._liquid_velocity(inventory)
        gas_velocity = self._gas_velocity(inventory)
        if liquid_mass > 0.0 and gas_mass > 0.0:
            if not math.isclose(
                liquid_velocity,
                gas_velocity,
                rel_tol=2.0e-11,
                abs_tol=2.0e-12,
            ):
                raise ContractViolation(
                    "mixed sharp-interface wall step requires the common contact velocity"
                )
            velocity = inventory.mixture_momentum_kg_m_s / mixture_mass
        elif liquid_mass > 0.0:
            velocity = liquid_velocity
        else:
            velocity = gas_velocity

        diameter = self.geometry.diameter_m
        perimeter = math.pi * diameter
        area = self.geometry.area_m2
        liquid_re = (
            0.0
            if liquid_mass == 0.0
            else (
                self.config.liquid_density_kg_m3
                * abs(velocity)
                * diameter
                / self.config.liquid_viscosity_Pa_s
            )
        )
        gas_density = (
            0.0
            if gas_mass == 0.0
            else gas_mass / inventory.gas_volume_m3
        )
        gas_re = (
            0.0
            if gas_mass == 0.0
            else (
                gas_density
                * abs(velocity)
                * diameter
                / self.config.gas_viscosity_Pa_s
            )
        )
        liquid_factor = f0_supply_smooth_pipe_darcy_factor(liquid_re)
        gas_factor = f0_supply_smooth_pipe_darcy_factor(gas_re)
        liquid_length = inventory.liquid_volume_m3 / area
        gas_length = inventory.gas_volume_m3 / area
        liquid_k = (
            liquid_factor
            * self.config.liquid_density_kg_m3
            * perimeter
            * liquid_length
            / 8.0
        )
        gas_k = gas_factor * gas_density * perimeter * gas_length / 8.0
        total_k = liquid_k + gas_k
        if velocity == 0.0 or total_k == 0.0:
            relaxed_velocity = velocity
            liquid_impulse = 0.0
            gas_impulse = 0.0
        else:
            relaxed_velocity = velocity / (
                1.0 + dt * total_k * abs(velocity) / mixture_mass
            )
            total_impulse = mixture_mass * (relaxed_velocity - velocity)
            liquid_impulse = total_impulse * liquid_k / total_k
            gas_impulse = total_impulse - liquid_impulse

        relaxed = self.state_from_bulk(
            gas_volume_m3=inventory.gas_volume_m3,
            gas_mass_kg=inventory.gas_mass_kg,
            liquid_velocity_upward_m_s=(
                relaxed_velocity if liquid_mass > 0.0 else 0.0
            ),
            gas_velocity_upward_m_s=(
                relaxed_velocity if gas_mass > 0.0 else 0.0
            ),
        )
        diagnostics = SupplyBranchWallDiagnostics(
            liquid_darcy_factor=liquid_factor,
            gas_darcy_factor=gas_factor,
            liquid_reynolds=liquid_re,
            gas_reynolds=gas_re,
            liquid_wall_impulse_kg_m_s=liquid_impulse,
            gas_wall_impulse_kg_m_s=gas_impulse,
        )
        observed = (
            self.inventory(relaxed).mixture_momentum_kg_m_s
            - inventory.mixture_momentum_kg_m_s
        )
        if not math.isclose(
            observed,
            diagnostics.total_wall_impulse_kg_m_s,
            rel_tol=2.0e-11,
            abs_tol=2.0e-15,
        ):
            raise ContractViolation("supply wall momentum ledger does not close")
        return relaxed, diagnostics

    def _nucleation_contact_velocity(
        self, bottom: SupplyBottomNodeCondition
    ) -> float:
        p_liquid_top = bottom.absolute_pressure_Pa - (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * self.geometry.length_m
        )
        p_reservoir = self.pressure_reservoir.reservoir_absolute_pressure_Pa
        rho_g = p_reservoir / self.config.rt_J_kg
        impedance_g = rho_g * self.config.gas_sound_speed_m_s
        impedance_l = (
            self.config.liquid_density_kg_m3
            * self.adapter.celerity_m_s(self.geometry.area_m2)
        )
        return (p_liquid_top - p_reservoir) / (impedance_l + impedance_g)

    def stable_timestep_s(
        self,
        state: SupplyBranchState,
        bottom: SupplyBottomNodeCondition | None = None,
    ) -> float:
        inventory = self.inventory(state)
        signed_ul = self._liquid_velocity(inventory)
        ul = abs(signed_ul)
        ug = abs(self._gas_velocity(inventory))
        if inventory.gas_volume_m3 == 0.0:
            node = self.default_bottom_condition() if bottom is None else bottom
            ug = max(ug, abs(self._nucleation_contact_velocity(node)))
        liquid_wave = self.adapter.celerity_m_s(self.geometry.area_m2)
        wave_dt = self.config.cfl * self.geometry.dz_m / max(
            ul + liquid_wave, ug + self.config.gas_sound_speed_m_s
        )
        event_dt = math.inf
        if signed_ul > 0.0 and inventory.gas_volume_m3 > 0.0:
            # Upward liquid motion consumes the top gas plug.
            event_dt = inventory.gas_volume_m3 / (self.geometry.area_m2 * ul)
        elif signed_ul > 0.0 and inventory.gas_volume_m3 == 0.0:
            event_dt = math.inf
        elif signed_ul < 0.0 and inventory.liquid_volume_m3 > 0.0:
            # Downward piston motion grows gas and can only exhaust liquid.
            event_dt = inventory.liquid_volume_m3 / (
                self.geometry.area_m2 * ul
            )
        return min(wave_dt, event_dt)

    def _liquid_bottom_momentum_flux_N(
        self, signed_liquid_rate_m3_s: float, bottom_pressure_Pa: float
    ) -> float:
        # The subtraction isolates Case-1's Q^2/A term while retaining its
        # exact circular/elastic implementation rather than duplicating it.
        moving = self.adapter.physical_flux(
            self.geometry.area_m2, signed_liquid_rate_m3_s
        )
        static = self.adapter.physical_flux(self.geometry.area_m2, 0.0)
        advective = self.config.liquid_density_kg_m3 * (
            moving.liquid_momentum_m4_s2 - static.liquid_momentum_m4_s2
        )
        return advective + bottom_pressure_Pa * self.geometry.area_m2

    def _packet(
        self,
        state: SupplyBranchState,
        dt_s: float,
        bottom: SupplyBottomNodeCondition,
        *,
        liquid_signed_into_branch_m3_s: float = 0.0,
        gas_signed_into_branch_kg_s: float = 0.0,
        liquid_speed_m_s: float = 0.0,
        gas_speed_m_s: float = 0.0,
        bottom_momentum_flux_upward_N: float | None = None,
        transaction_id: str,
    ) -> SupplyBranchGrossFluxPacket:
        ql = float(liquid_signed_into_branch_m3_s)
        mg = float(gas_signed_into_branch_kg_s)
        return SupplyBranchGrossFluxPacket(
            transaction_id=transaction_id,
            base_state_token=_state_token(state),
            dt_s=dt_s,
            bottom_absolute_pressure_Pa=bottom.absolute_pressure_Pa,
            bottom_momentum_flux_upward_N=(
                bottom.absolute_pressure_Pa * self.geometry.area_m2
                if bottom_momentum_flux_upward_N is None
                else bottom_momentum_flux_upward_N
            ),
            liquid_upward_rate_m3_s=max(ql, 0.0),
            liquid_downward_rate_m3_s=max(-ql, 0.0),
            gas_upward_mass_rate_kg_s=max(mg, 0.0),
            gas_downward_mass_rate_kg_s=max(-mg, 0.0),
            liquid_upward_speed_m_s=liquid_speed_m_s if ql > 0.0 else 0.0,
            liquid_downward_speed_m_s=liquid_speed_m_s if ql < 0.0 else 0.0,
            gas_upward_speed_m_s=gas_speed_m_s if mg > 0.0 else 0.0,
            gas_downward_speed_m_s=gas_speed_m_s if mg < 0.0 else 0.0,
        )

    def _static_closed_step(
        self,
        state: SupplyBranchState,
        dt_s: float,
        stage: Stage,
        bottom: SupplyBottomNodeCondition,
        transaction_id: str,
    ) -> SupplyBranchStepResult:
        before = self.inventory(state)
        packet = self._packet(
            state,
            dt_s,
            bottom,
            bottom_momentum_flux_upward_N=(
                bottom.absolute_pressure_Pa * self.geometry.area_m2
            ),
            transaction_id=transaction_id,
        )
        weight = (
            self.config.liquid_density_kg_m3 * before.liquid_volume_m3
            + before.gas_mass_kg
        ) * self.adapter.gravity_m_s2
        bottom_impulse = bottom.absolute_pressure_Pa * self.geometry.area_m2 * dt_s
        # The closed top supplies the hydrostatic reaction required by the
        # fixed branch; it is not a material or fitted source.
        top_impulse = -(bottom.absolute_pressure_Pa * self.geometry.area_m2 - weight) * dt_s
        gravity_impulse = -weight * dt_s
        residual = top_impulse + bottom_impulse + gravity_impulse
        return SupplyBranchStepResult(
            state=state,
            delta=SupplyBranchDelta.zeros(state.cell_count),
            bottom=packet,
            ledger=SupplyBranchLedgerEntry(
                stage=stage,
                before=before,
                after=before,
                top_gas_net_into_branch_kg=0.0,
                bottom_liquid_net_into_branch_m3=0.0,
                bottom_gas_net_into_branch_kg=0.0,
                top_momentum_impulse_kg_m_s=top_impulse,
                bottom_momentum_impulse_kg_m_s=bottom_impulse,
                gravity_impulse_kg_m_s=gravity_impulse,
                wall_momentum_impulse_kg_m_s=0.0,
                liquid_wall_impulse_kg_m_s=0.0,
                gas_wall_impulse_kg_m_s=0.0,
                liquid_darcy_factor=0.0,
                gas_darcy_factor=0.0,
                acoustic_projection_impulse_kg_m_s=0.0,
                liquid_volume_residual_m3=0.0,
                gas_mass_residual_kg=0.0,
                phase_volume_residual_m3=0.0,
                mixture_momentum_residual_kg_m_s=-residual,
                interface_recoil_residual_kg_m_s=0.0,
                maximum_courant=0.0,
                minimum_liquid_area_m2=min(state.Al),
                minimum_gas_mass_kg_m=min(state.Mg),
            ),
        )

    def _nucleation_step(
        self,
        state: SupplyBranchState,
        dt_s: float,
        bottom: SupplyBottomNodeCondition,
        transaction_id: str,
    ) -> SupplyBranchStepResult:
        before = self.inventory(state)
        p_reservoir = self.pressure_reservoir.reservoir_absolute_pressure_Pa
        p_liquid_top = bottom.absolute_pressure_Pa - (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * self.geometry.length_m
        )
        if p_liquid_top <= 0.0:
            raise ContractViolation("hydrostatic liquid pressure at the branch top is non-positive")
        rho_g = p_reservoir / self.config.rt_J_kg
        impedance_g = rho_g * self.config.gas_sound_speed_m_s
        # The linear acoustic contact bounds the pressure-reservoir Riemann
        # solution.  The actual sharp-piston velocity is obtained below from
        # the conservative whole-branch momentum balance, so nucleation does
        # not require an artificial initialization impulse.
        contact_velocity = self._nucleation_contact_velocity(bottom)
        if contact_velocity >= 0.0 or bottom.wall:
            return self._static_closed_step(
                state,
                dt_s,
                "stage2_pressure_reservoir",
                bottom,
                transaction_id,
            )
        area = self.geometry.area_m2
        total_volume = self.geometry.total_volume_m3
        rho_l = self.config.liquid_density_kg_m3
        gravity = self.adapter.gravity_m_s2

        def piston_balance(
            velocity: float,
        ) -> tuple[float, float, float, float, float, float]:
            interface_pressure = p_reservoir + impedance_g * velocity
            if interface_pressure <= 0.0:
                raise ContractViolation("nucleation Riemann pressure became non-positive")
            gas_volume = -area * velocity * dt_s
            if not 0.0 <= gas_volume < total_volume:
                raise ContractViolation("nucleation Riemann volume left the finite branch")
            gas_density = interface_pressure / self.config.rt_J_kg
            gas_mass = gas_density * gas_volume
            liquid_volume = total_volume - gas_volume
            liquid_signed = area * velocity
            bottom_flux = self._liquid_bottom_momentum_flux_N(
                liquid_signed, bottom.absolute_pressure_Pa
            )
            top_flux = (
                interface_pressure + gas_density * velocity * velocity
            ) * area
            average_mass = 0.5 * (
                rho_l * total_volume + rho_l * liquid_volume + gas_mass
            )
            impulse = (
                bottom_flux - top_flux - average_mass * gravity
            ) * dt_s
            final_momentum = (rho_l * liquid_volume + gas_mass) * velocity
            return (
                final_momentum - impulse,
                interface_pressure,
                gas_volume,
                gas_mass,
                bottom_flux,
                top_flux,
            )

        low = contact_velocity
        high = 0.0
        residual_low = piston_balance(low)[0]
        residual_high = piston_balance(high)[0]
        if not residual_low <= 0.0 <= residual_high:
            raise ContractViolation("could not bracket the gas/liquid nucleation Riemann root")
        for _ in range(90):
            middle = 0.5 * (low + high)
            residual_middle = piston_balance(middle)[0]
            if residual_middle > 0.0:
                high = middle
            else:
                low = middle
        interface_velocity = 0.5 * (low + high)
        (
            _,
            interface_pressure,
            gas_volume,
            gas_mass,
            bottom_flux_N,
            top_flux_N,
        ) = piston_balance(interface_velocity)
        liquid_signed = self.geometry.area_m2 * interface_velocity
        final = self.state_from_bulk(
            gas_volume_m3=gas_volume,
            gas_mass_kg=gas_mass,
            liquid_velocity_upward_m_s=interface_velocity,
            gas_velocity_upward_m_s=interface_velocity,
        )
        final, wall = self._apply_f0_wall_shear(final, dt_s)
        after = self.inventory(final)
        packet = self._packet(
            state,
            dt_s,
            bottom,
            liquid_signed_into_branch_m3_s=liquid_signed,
            liquid_speed_m_s=abs(interface_velocity),
            bottom_momentum_flux_upward_N=bottom_flux_N,
            transaction_id=transaction_id,
        )
        average_mass = 0.5 * (
            self.config.liquid_density_kg_m3
            * (before.liquid_volume_m3 + after.liquid_volume_m3)
            + before.gas_mass_kg
            + after.gas_mass_kg
        )
        top_impulse = -top_flux_N * dt_s
        bottom_impulse = bottom_flux_N * dt_s
        gravity_impulse = -average_mass * self.adapter.gravity_m_s2 * dt_s
        raw_impulse = top_impulse + bottom_impulse + gravity_impulse
        projection_impulse = 0.0
        liquid_residual = (
            after.liquid_volume_m3
            - before.liquid_volume_m3
            - liquid_signed * dt_s
        )
        gas_residual = after.gas_mass_kg - before.gas_mass_kg - gas_mass
        phase_residual = (
            after.liquid_volume_m3
            + after.gas_volume_m3
            - self.geometry.total_volume_m3
        )
        momentum_residual = (
            after.mixture_momentum_kg_m_s
            - before.mixture_momentum_kg_m_s
            - raw_impulse
            - wall.total_wall_impulse_kg_m_s
            - projection_impulse
        )
        return SupplyBranchStepResult(
            state=final,
            delta=self.delta(state, final),
            bottom=packet,
            ledger=SupplyBranchLedgerEntry(
                stage="stage2_pressure_reservoir",
                before=before,
                after=after,
                top_gas_net_into_branch_kg=gas_mass,
                bottom_liquid_net_into_branch_m3=liquid_signed * dt_s,
                bottom_gas_net_into_branch_kg=0.0,
                top_momentum_impulse_kg_m_s=top_impulse,
                bottom_momentum_impulse_kg_m_s=bottom_impulse,
                gravity_impulse_kg_m_s=gravity_impulse,
                wall_momentum_impulse_kg_m_s=wall.total_wall_impulse_kg_m_s,
                liquid_wall_impulse_kg_m_s=wall.liquid_wall_impulse_kg_m_s,
                gas_wall_impulse_kg_m_s=wall.gas_wall_impulse_kg_m_s,
                liquid_darcy_factor=wall.liquid_darcy_factor,
                gas_darcy_factor=wall.gas_darcy_factor,
                acoustic_projection_impulse_kg_m_s=projection_impulse,
                liquid_volume_residual_m3=liquid_residual,
                gas_mass_residual_kg=gas_residual,
                phase_volume_residual_m3=phase_residual,
                mixture_momentum_residual_kg_m_s=momentum_residual,
                interface_recoil_residual_kg_m_s=0.0,
                maximum_courant=(
                    (abs(interface_velocity) + self.config.gas_sound_speed_m_s)
                    * dt_s
                    / self.geometry.dz_m
                ),
                minimum_liquid_area_m2=min(final.Al),
                minimum_gas_mass_kg_m=min(final.Mg),
            ),
        )

    def _implicit_stage2_top_gas_exchange(
        self,
        *,
        gas_mass_before_kg: float,
        gas_volume_after_m3: float,
        gas_velocity_upward_m_s: float,
        bottom_gas_into_kg_s: float,
        dt_s: float,
    ) -> tuple[float, float, float]:
        """Positivity-preserving backward-Euler pressure-boundary exchange.

        The first source-aligned gas parcel is many floating-point ulps wide,
        but its acoustic residence time is far shorter than the network RK
        step.  An explicit HLL mass update can consequently remove more than
        the finite parcel during an RK2 pressure iterate.  That is a donor
        capacity violation, not permission to clip mass or pre-seed a larger
        bubble.

        This scalar solve keeps the published 5700 Pa reservoir and the same
        isothermal HLL flux.  It evaluates that flux at the end-of-stage gas
        density and solves the finite-inventory balance atomically.  The
        returned top rate is reconstructed from the solved balance, so the
        parcel is counted once and the gas ledger closes to roundoff.
        """

        mass_before = _nonnegative("stage2 gas mass before", gas_mass_before_kg)
        volume_after = _finite("stage2 gas volume after", gas_volume_after_m3)
        velocity = _finite("stage2 gas velocity", gas_velocity_upward_m_s)
        bottom_rate = _finite("stage2 bottom gas rate", bottom_gas_into_kg_s)
        dt = _finite("stage2 implicit gas dt", dt_s)
        if volume_after <= 0.0 or dt <= 0.0:
            raise ContractViolation(
                "Stage-2 implicit gas exchange requires positive volume and dt"
            )

        def boundary(candidate_mass: float):
            pressure = candidate_mass / volume_after * self.config.rt_J_kg
            return self.pressure_reservoir.evaluate(
                node_absolute_pressure_Pa=pressure,
                node_axial_velocity_m_s=-velocity,
                inlet_area_m2=self.geometry.area_m2,
            )

        def residual(candidate_mass: float) -> float:
            flux = boundary(candidate_mass)
            return candidate_mass - mass_before - dt * (
                flux.mass_flow_kg_s + bottom_rate
            )

        reservoir_mass = (
            self.pressure_reservoir.reservoir_absolute_pressure_Pa
            / self.config.rt_J_kg
            * volume_after
        )
        scale = max(mass_before, reservoir_mass, 1.0e-300)
        low = max(math.ulp(scale), 1.0e-300)
        residual_low = residual(low)
        if residual_low >= 0.0:
            raise ContractViolation(
                "Stage-2 finite gas parcel has no positive implicit HLL mass root"
            )
        high = 2.0 * max(
            scale,
            mass_before + dt * max(bottom_rate, 0.0),
            low,
        )
        residual_high = residual(high)
        for _ in range(80):
            if residual_high > 0.0:
                break
            high *= 2.0
            residual_high = residual(high)
        else:
            raise ContractViolation(
                "Stage-2 implicit HLL mass root could not be bracketed"
            )

        for _ in range(120):
            middle = 0.5 * (low + high)
            if residual(middle) > 0.0:
                high = middle
            else:
                low = middle
        mass_after = 0.5 * (low + high)
        final_flux = boundary(mass_after)
        top_rate = (mass_after - mass_before) / dt - bottom_rate
        balance_residual = mass_after - mass_before - dt * (
            top_rate + bottom_rate
        )
        balance_scale = max(
            mass_after,
            mass_before,
            abs(dt * top_rate),
            abs(dt * bottom_rate),
            1.0e-300,
        )
        if (
            mass_after <= 0.0
            or abs(balance_residual) > 2.0e-13 * balance_scale
        ):
            raise ContractViolation(
                "Stage-2 implicit HLL finite-inventory ledger did not close"
            )
        return (
            mass_after,
            top_rate,
            final_flux.axial_momentum_pressure_rate_N,
        )

    def _material_step(
        self,
        state: SupplyBranchState,
        dt_s: float,
        stage: Stage,
        bottom: SupplyBottomNodeCondition,
        transaction_id: str,
    ) -> SupplyBranchStepResult:
        before = self.inventory(state)
        area = self.geometry.area_m2
        gravity = self.adapter.gravity_m_s2
        gas_pressure = self.gas_pressure_Pa(state)
        if gas_pressure is None:
            raise ContractViolation("material step requires a finite gas inventory")
        liquid_velocity = self._liquid_velocity(before)
        gas_velocity = self._gas_velocity(before)

        if stage == "stage1_closed":
            top_mass_into = 0.0
            # Hydrostatic wall reaction at the upper face.
            top_flux_N = gas_pressure * area - before.gas_mass_kg * gravity
        elif stage == "stage2_pressure_reservoir":
            # Filled after the bottom packet and gas-volume update are known;
            # the finite newborn plug requires an implicit donor-capacity
            # balance at this acoustic boundary.
            top_mass_into = 0.0
            top_flux_N = 0.0
        else:
            raise ContractViolation(f"unknown supply-branch stage {stage!r}")

        total = self.geometry.total_volume_m3
        full_gas = before.liquid_volume_m3 <= 64.0 * math.ulp(total)
        bottom_gas_into = 0.0
        bottom_gas_flux_N = 0.0
        bottom_gas_speed = 0.0
        liquid_signed = 0.0
        bottom_liquid_flux_N = 0.0
        if bottom.wall:
            if abs(liquid_velocity) > 1.0e-12 or abs(gas_velocity) > 1.0e-12:
                raise ContractViolation("a wall cannot be imposed on a moving supply state")
            return self._static_closed_step(
                state, dt_s, stage, bottom, transaction_id
            )
        if full_gas:
            if not bottom.gas_accepting:
                bottom_gas_flux_N = gas_pressure * area
            else:
                rho_node = bottom.absolute_pressure_Pa / self.config.rt_J_kg
                rho_branch = before.gas_mass_kg / total
                mass_flux, momentum_flux = _isothermal_hll(
                    rho_node,
                    bottom.gas_velocity_upward_m_s,
                    rho_branch,
                    gas_velocity,
                    self.config.rt_J_kg,
                )
                bottom_gas_into = mass_flux * area
                bottom_gas_flux_N = momentum_flux * area
                donor_density = rho_node if bottom_gas_into > 0.0 else rho_branch
                if bottom_gas_into != 0.0:
                    bottom_gas_speed = abs(bottom_gas_into) / (donor_density * area)
        else:
            # The liquid plug is incompressible: its signed bottom flow is the
            # sharp-interface kinematic flux.  Gas cannot be written through
            # the lower face until the gas segment actually reaches it.
            liquid_signed = area * liquid_velocity
            bottom_liquid_flux_N = self._liquid_bottom_momentum_flux_N(
                liquid_signed, bottom.absolute_pressure_Pa
            )

        gas_volume_new = before.gas_volume_m3 - liquid_signed * dt_s
        if not 0.0 < gas_volume_new <= total:
            raise ContractViolation("supply phase inventory left its positive finite interval")
        if stage == "stage2_pressure_reservoir":
            (
                gas_mass_new,
                top_mass_into,
                top_flux_N,
            ) = self._implicit_stage2_top_gas_exchange(
                gas_mass_before_kg=before.gas_mass_kg,
                gas_volume_after_m3=gas_volume_new,
                gas_velocity_upward_m_s=gas_velocity,
                bottom_gas_into_kg_s=bottom_gas_into,
                dt_s=dt_s,
            )
        else:
            gas_mass_new = before.gas_mass_kg + dt_s * bottom_gas_into
        if gas_mass_new <= 0.0:
            raise ContractViolation("supply gas mass left its positive finite interval")
        liquid_volume_new = total - gas_volume_new

        if full_gas:
            gas_momentum_new = before.gas_momentum_kg_m_s + dt_s * (
                bottom_gas_flux_N
                - top_flux_N
                - before.gas_mass_kg * gravity
            )
            liquid_momentum_new = 0.0
        else:
            # Sharp piston/contact closure: pressure traction at the material
            # interface is an equal/opposite internal pair.  Advance the
            # conservative *mixture* momentum, then allocate it to the two
            # persistent phase momenta at their common contact velocity.  This
            # is the standard no-slip sharp-interface constraint; it avoids a
            # vanishing-volume gas cell acquiring unbounded velocity while
            # preserving the external momentum ledger exactly.
            mixture_momentum_new = before.mixture_momentum_kg_m_s + dt_s * (
                bottom_liquid_flux_N
                - top_flux_N
                - (
                    self.config.liquid_density_kg_m3 * before.liquid_volume_m3
                    + before.gas_mass_kg
                )
                * gravity
            )
            common_mass = (
                self.config.liquid_density_kg_m3 * liquid_volume_new
                + gas_mass_new
            )
            common_velocity = mixture_momentum_new / common_mass
            liquid_momentum_new = (
                self.config.liquid_density_kg_m3
                * liquid_volume_new
                * common_velocity
            )
            gas_momentum_new = gas_mass_new * common_velocity

        if full_gas:
            liquid_velocity_new = 0.0
            gas_velocity_new = gas_momentum_new / gas_mass_new
        else:
            liquid_velocity_new = common_velocity
            gas_velocity_new = common_velocity
        final = self.state_from_bulk(
            gas_volume_m3=gas_volume_new,
            gas_mass_kg=gas_mass_new,
            liquid_velocity_upward_m_s=liquid_velocity_new,
            gas_velocity_upward_m_s=gas_velocity_new,
        )
        final, wall = self._apply_f0_wall_shear(final, dt_s)
        after = self.inventory(final)
        packet = self._packet(
            state,
            dt_s,
            bottom,
            liquid_signed_into_branch_m3_s=liquid_signed,
            gas_signed_into_branch_kg_s=bottom_gas_into,
            liquid_speed_m_s=abs(liquid_velocity),
            gas_speed_m_s=bottom_gas_speed,
            bottom_momentum_flux_upward_N=(
                bottom_gas_flux_N if full_gas else bottom_liquid_flux_N
            ),
            transaction_id=transaction_id,
        )

        top_impulse = -top_flux_N * dt_s
        bottom_impulse = (
            bottom_gas_flux_N if full_gas else bottom_liquid_flux_N
        ) * dt_s
        gravity_impulse = -(
            self.config.liquid_density_kg_m3 * before.liquid_volume_m3
            + before.gas_mass_kg
        ) * gravity * dt_s
        material_impulse = (
            top_impulse
            + bottom_impulse
            + gravity_impulse
            + wall.total_wall_impulse_kg_m_s
        )
        liquid_residual = (
            after.liquid_volume_m3
            - before.liquid_volume_m3
            - liquid_signed * dt_s
        )
        gas_residual = (
            after.gas_mass_kg
            - before.gas_mass_kg
            - (top_mass_into + bottom_gas_into) * dt_s
        )
        phase_residual = (
            after.liquid_volume_m3 + after.gas_volume_m3 - total
        )
        momentum_residual = (
            after.mixture_momentum_kg_m_s
            - before.mixture_momentum_kg_m_s
            - material_impulse
        )
        interface_recoil = 0.0
        return SupplyBranchStepResult(
            state=final,
            delta=self.delta(state, final),
            bottom=packet,
            ledger=SupplyBranchLedgerEntry(
                stage=stage,
                before=before,
                after=after,
                top_gas_net_into_branch_kg=top_mass_into * dt_s,
                bottom_liquid_net_into_branch_m3=liquid_signed * dt_s,
                bottom_gas_net_into_branch_kg=bottom_gas_into * dt_s,
                top_momentum_impulse_kg_m_s=top_impulse,
                bottom_momentum_impulse_kg_m_s=bottom_impulse,
                gravity_impulse_kg_m_s=gravity_impulse,
                wall_momentum_impulse_kg_m_s=wall.total_wall_impulse_kg_m_s,
                liquid_wall_impulse_kg_m_s=wall.liquid_wall_impulse_kg_m_s,
                gas_wall_impulse_kg_m_s=wall.gas_wall_impulse_kg_m_s,
                liquid_darcy_factor=wall.liquid_darcy_factor,
                gas_darcy_factor=wall.gas_darcy_factor,
                acoustic_projection_impulse_kg_m_s=0.0,
                liquid_volume_residual_m3=liquid_residual,
                gas_mass_residual_kg=gas_residual,
                phase_volume_residual_m3=phase_residual,
                mixture_momentum_residual_kg_m_s=momentum_residual,
                interface_recoil_residual_kg_m_s=interface_recoil,
                maximum_courant=(
                    max(
                        abs(liquid_velocity) + self.adapter.celerity_m_s(area),
                        abs(gas_velocity) + self.config.gas_sound_speed_m_s,
                    )
                    * dt_s
                    / self.geometry.dz_m
                ),
                minimum_liquid_area_m2=min(final.Al),
                minimum_gas_mass_kg_m=min(final.Mg),
            ),
        )

    def propose_atomic_step(
        self,
        state: SupplyBranchState,
        dt_s: float,
        *,
        stage: Stage,
        bottom: SupplyBottomNodeCondition | None = None,
        transaction_id: str = "supply-branch-step",
    ) -> SupplyBranchStepResult:
        """Build one CFL-safe immutable component/T-node proposal."""

        self.validate_state(state)
        dt = _finite("supply-branch dt_s", dt_s)
        if dt <= 0.0:
            raise ContractViolation("supply-branch dt_s must be positive")
        if stage not in ("stage1_closed", "stage2_pressure_reservoir"):
            raise ContractViolation(f"unknown supply-branch stage {stage!r}")
        node = self.default_bottom_condition() if bottom is None else bottom
        stable = self.stable_timestep_s(state, node)
        if dt > stable * (1.0 + 1.0e-12):
            raise ContractViolation(
                f"requested supply dt {dt:.6e} exceeds CFL/event limit {stable:.6e}"
            )
        inventory = self.inventory(state)
        if inventory.gas_volume_m3 == 0.0:
            if stage == "stage1_closed":
                result = self._static_closed_step(
                    state, dt, stage, node, transaction_id
                )
            else:
                result = self._nucleation_step(
                    state, dt, node, transaction_id
                )
        else:
            result = self._material_step(
                state, dt, stage, node, transaction_id
            )
        ledger = result.ledger
        tolerances = (
            abs(ledger.liquid_volume_residual_m3),
            abs(ledger.gas_mass_residual_kg),
            abs(ledger.phase_volume_residual_m3),
            abs(ledger.mixture_momentum_residual_kg_m_s),
            abs(ledger.interface_recoil_residual_kg_m_s),
        )
        if max(tolerances) > 5.0e-11:
            raise ContractViolation(f"supply-branch conservative ledger failed: {tolerances!r}")
        if ledger.maximum_courant > self.config.cfl * (
            1.0 + 1.0e-10
        ) and inventory.gas_volume_m3 > 0.0:
            raise ContractViolation("accepted supply-branch step exceeded the CFL gate")
        return result

    @staticmethod
    def _rate_delta(delta: SupplyBranchDelta, dt_s: float) -> SupplyBranchDelta:
        inverse = 1.0 / dt_s
        return SupplyBranchDelta(
            Al=tuple(value * inverse for value in delta.Al),
            Ql=tuple(value * inverse for value in delta.Ql),
            Mg=tuple(value * inverse for value in delta.Mg),
            Jg=tuple(value * inverse for value in delta.Jg),
        )

    def _accepted_bottom_flux(
        self,
        trial: TNodeTrial,
        result: SupplyBranchStepResult,
    ) -> GrossNodePortFlux:
        """Translate the component's actual bottom packet to node convention."""

        packet = result.bottom
        key = PortKey("air_supply_T", "supply_bottom")
        pressure_traction_to_node = -(
            packet.bottom_absolute_pressure_Pa * self.geometry.area_m2
        )
        total_momentum_to_node = -packet.bottom_momentum_flux_upward_N
        return GrossNodePortFlux(
            key=key,
            liquid_into_node_m3_s=packet.liquid_downward_rate_m3_s,
            liquid_out_of_node_m3_s=packet.liquid_upward_rate_m3_s,
            gas_into_node_kg_s=packet.gas_downward_mass_rate_kg_s,
            gas_out_of_node_kg_s=packet.gas_upward_mass_rate_kg_s,
            liquid_into_node_speed_m_s=packet.liquid_downward_speed_m_s,
            liquid_out_of_node_speed_m_s=packet.liquid_upward_speed_m_s,
            gas_into_node_speed_m_s=packet.gas_downward_speed_m_s,
            gas_out_of_node_speed_m_s=packet.gas_upward_speed_m_s,
            advective_momentum_to_node_z_N=(
                total_momentum_to_node - pressure_traction_to_node
            ),
            pressure_traction_to_node_z_N=pressure_traction_to_node,
        )

    @staticmethod
    def _signed_momentum_exchange(
        signed_rate_N: float,
    ) -> tuple[float, float]:
        return max(signed_rate_N, 0.0), max(-signed_rate_N, 0.0)

    def _external_exchange(
        self, result: SupplyBranchStepResult, dt_s: float
    ) -> BoundaryExchange:
        ledger = result.ledger
        top_gas_rate = ledger.top_gas_net_into_branch_kg / dt_s
        top_momentum_rate = ledger.top_momentum_impulse_kg_m_s / dt_s
        if ledger.stage == "stage1_closed":
            # The closed upper face is an external wall reaction, not an
            # advective boundary momentum flux.
            momentum_in = momentum_out = 0.0
            top_wall_force = top_momentum_rate
        else:
            momentum_in, momentum_out = self._signed_momentum_exchange(
                top_momentum_rate
            )
            top_wall_force = 0.0
        return BoundaryExchange(
            gas_inflow_kg_s=max(top_gas_rate, 0.0),
            gas_outflow_kg_s=max(-top_gas_rate, 0.0),
            momentum_z_in_N=momentum_in,
            momentum_z_out_N=momentum_out,
            external_force_z_N=(
                top_wall_force
                + (
                    ledger.gravity_impulse_kg_m_s
                    + ledger.wall_momentum_impulse_kg_m_s
                    + ledger.acoustic_projection_impulse_kg_m_s
                )
                / dt_s
            ),
        )

    def evaluate_trial(
        self,
        state: CoupledState,
        geometry: CoupledGeometry,
        trials: tuple[TNodeTrial, ...],
    ) -> ComponentStageProposal:
        """Evaluate one immutable physical ``air_supply_T`` pressure trial.

        The nonlinear node's provisional characteristic packet is a pressure
        iterate.  The accepted port packet is always the one returned by the
        existing finite-branch pressure-reservoir/piston solver, so the node
        residual and the committed branch delta use exactly the same material
        and momentum exchange.
        """

        validate_trial_set(trials)
        if len(trials) != 1 or trials[0].node_name != "air_supply_T":
            raise ContractViolation("supply pure trial requires only air_supply_T")
        trial = trials[0]
        self._validate_coupled_geometry(state.supply_branch, geometry)
        provisional = next(
            (
                flux
                for flux in trial.gross_fluxes
                if flux.key == PortKey("air_supply_T", "supply_bottom")
            ),
            None,
        )
        if provisional is None:
            raise ContractViolation("air_supply_T trial has no supply-bottom flux")
        node_gas_velocity = (
            provisional.gas_out_of_node_speed_m_s
            if provisional.gas_out_of_node_kg_s > 0.0
            else -provisional.gas_into_node_speed_m_s
            if provisional.gas_into_node_kg_s > 0.0
            else 0.0
        )
        bottom = SupplyBottomNodeCondition(
            absolute_pressure_Pa=trial.common_absolute_pressure_Pa,
            gas_velocity_upward_m_s=node_gas_velocity,
        )
        try:
            result = self.propose_atomic_step(
                state.supply_branch,
                trial.dt_s,
                stage=trial.physical_stage,
                bottom=bottom,
                transaction_id=f"{trial.trial_id}-supply",
            )
        except ContractViolation as exc:
            text = str(exc)
            retryable = any(
                token in text.lower()
                for token in ("cfl", "courant", "bracket", "interval", "capacity")
            )
            rejection = CapacityReject(
                component_id=self.component_id,
                reason_code="cfl" if "cfl" in text.lower() else "phase_capacity",
                detail=text,
                requested_dt_s=trial.dt_s,
                retryable=retryable,
                maximum_admissible_dt_s=(
                    0.5 * trial.dt_s if retryable else None
                ),
            )
            return ComponentStageProposal.rejected(
                component_id=self.component_id,
                base_state_token=trial.base_state_token,
                trials=trials,
                rejection=rejection,
                evidence_status="S1-1D-F0_supply_physical_trial_fail_closed",
            )

        return ComponentStageProposal.accepted(
            component_id=self.component_id,
            base_state_token=trial.base_state_token,
            trials=trials,
            delta=self._rate_delta(result.delta, trial.dt_s),
            accepted_gross_fluxes=(self._accepted_bottom_flux(trial, result),),
            external_exchange=self._external_exchange(result, trial.dt_s),
            evidence_status=(
                "S1-1D-F0_finite_supply_pressure_reservoir_piston_physical_trial"
            ),
        )

    def advance(
        self,
        state: SupplyBranchState,
        duration_s: float,
        *,
        stage: Stage,
        bottom: SupplyBottomNodeCondition | None = None,
        transaction_prefix: str = "supply-branch",
    ) -> SupplyBranchAdvanceResult:
        """Advance with rejection-only positivity control; never clip material."""

        duration = _finite("supply-branch duration_s", duration_s)
        if duration <= 0.0:
            raise ContractViolation("supply-branch duration_s must be positive")
        current = state
        elapsed = 0.0
        packets: list[SupplyBranchGrossFluxPacket] = []
        entries: list[SupplyBranchLedgerEntry] = []
        for step_index in range(self.config.maximum_substeps):
            if elapsed >= duration - 2.0e-14 * max(1.0, duration):
                return SupplyBranchAdvanceResult(current, tuple(packets), tuple(entries))
            dt = min(self.stable_timestep_s(current, bottom), duration - elapsed)
            for _ in range(40):
                try:
                    result = self.propose_atomic_step(
                        current,
                        dt,
                        stage=stage,
                        bottom=bottom,
                        transaction_id=f"{transaction_prefix}-{step_index}",
                    )
                    break
                except ContractViolation:
                    dt *= 0.5
                    if dt <= 1.0e-13:
                        raise
            else:
                raise ContractViolation("supply positivity/CFL rejection did not recover")
            current = result.state
            elapsed += dt
            packets.append(result.bottom)
            entries.append(result.ledger)
        raise ContractViolation("maximum_substeps reached before supply duration")


__all__ = [
    "CLOSURE_PROVENANCE",
    "GEOMETRY_PROVENANCE",
    "INITIAL_WATER_SURFACE_Z_M",
    "PUBLISHED_GAS_GAUGE_PRESSURE_PA",
    "SUPPLY_BRANCH_BOTTOM_Z_M",
    "SUPPLY_BRANCH_ALLOWED_CELL_COUNTS",
    "SUPPLY_BRANCH_CELL_COUNT",
    "SUPPLY_BRANCH_DIAMETER_M",
    "SUPPLY_BRANCH_LENGTH_M",
    "SUPPLY_BRANCH_TOP_Z_M",
    "Stage",
    "SupplyBottomNodeCondition",
    "SupplyBranchAdvanceResult",
    "SupplyBranchConfig",
    "SupplyBranchGeometry",
    "SupplyBranchGrossFluxPacket",
    "SupplyBranchInventory",
    "SupplyBranchLedgerEntry",
    "SupplyBranchStepResult",
    "SupplyBranchTwoPhaseSolver",
    "SupplyBranchWallDiagnostics",
    "f0_supply_smooth_pipe_darcy_factor",
]
