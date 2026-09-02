"""Distributed Case-1-derived horizontal two-phase verification solver.

This module is deliberately narrower than the future production network.  It
advances the S1 horizontal main as cell averages ``(Al, Ql, Mg, Jg)`` and the
published 0.1373 m air stub as persistent gas-only ``(Mg, Jg)`` cells.  The
liquid circular geometry, hydrostatic/elastic physical flux and wave speed are
obtained from :mod:`horizontal_case1_adapter`; no finite Case-1 gas pocket,
valve-release shock fit or result target is imported.

The gas is an isothermal Euler phase.  In a gas-occupied main cell

``Ag = Apipe - Al`` and ``p = Mg*R*T/Ag``.

Gas-free full-pipe cells retain Case-1's elastic liquid storage and may have
``Al > Apipe``.  Such a cell has ``Ag=0`` and must have ``Mg=Jg=0``.  This
piecewise contract is essential: forbidding the Case-1 elastic branch would
make the published 0.586/0.584 m pressure heads unrepresentable.  It does not
relax the exact complement/EOS contract in any gas-occupied cell.

The two phase pressure contributions use complementary face areas.  The
geometric pressure sources are equal and opposite after the liquid equation
is multiplied by density, and interphase drag is likewise a strict recoil
pair.  Hence internal pressure/drag cannot create mixture momentum.

Important limitation
--------------------
Mahyawansi et al. do not publish horizontal wall/interphase closure
coefficients or the experimental valve/line loss law.  A solver therefore
requires an explicit :class:`HorizontalClosureSet`.  The supplied inviscid
closure is for conservation/static/smoke verification only and has
``alignment_ready=False``.  The smooth-pipe option is also declared and
unvalidated; it must not be tuned to the paper or promoted as an accepted
physical result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .errors import ContractViolation, MissingPhysicalClosure
from .horizontal_case1_adapter import Case1HorizontalLiquidAdapter
from .pressure_reservoir import IsothermalIdealGasPressureReservoir
from .state import HorizontalState


Array = np.ndarray
Stage = Literal["stage1_closed", "stage2_pressure_reservoir"]

AIR_STUB_LENGTH_M = 0.1500 - 0.0127
AIR_STUB_EVIDENCE = "published_2D_geometry__persistent_gas_only_FV_branch"
WATER_INLET_HEAD_M = 0.586
WATER_OUTLET_HEAD_M = 0.584
INITIAL_WATER_SURFACE_HEAD_M = 0.5842


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _tuple(values: Array) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


@dataclass(frozen=True, slots=True)
class AirStubGeometry:
    """The actual S1 side branch, from the main crown to the inlet top."""

    length_m: float = AIR_STUB_LENGTH_M
    cell_count: int = 14
    evidence_status: str = AIR_STUB_EVIDENCE

    def __post_init__(self) -> None:
        length = _finite("air-stub length", self.length_m)
        if not math.isclose(length, AIR_STUB_LENGTH_M, rel_tol=0.0, abs_tol=1.0e-12):
            raise ContractViolation("S1 air-stub length must remain 0.1373 m")
        if int(self.cell_count) != self.cell_count or self.cell_count < 2:
            raise ContractViolation("air-stub cell_count must be an integer >= 2")
        if not self.evidence_status.strip():
            raise ContractViolation("air-stub evidence status must be non-empty")

    @property
    def dz_m(self) -> float:
        return self.length_m / self.cell_count


@dataclass(frozen=True, slots=True)
class AirStubState:
    """Persistent gas mass/momentum per length in the 0.1373 m stub."""

    Mg: tuple[float, ...]
    Jg: tuple[float, ...]

    def __post_init__(self) -> None:
        mg = tuple(float(value) for value in self.Mg)
        jg = tuple(float(value) for value in self.Jg)
        if not mg or len(mg) != len(jg):
            raise ContractViolation("air-stub Mg/Jg must have one common non-empty grid")
        if not all(math.isfinite(value) for value in mg + jg):
            raise ContractViolation("air-stub state must be finite")
        if any(value <= 0.0 for value in mg):
            raise ContractViolation("the gas-only air stub must retain positive mass")
        object.__setattr__(self, "Mg", mg)
        object.__setattr__(self, "Jg", jg)

    @property
    def cell_count(self) -> int:
        return len(self.Mg)


@dataclass(frozen=True, slots=True)
class DistributedHorizontalState:
    time_s: float
    main: HorizontalState
    air_stub: AirStubState

    def __post_init__(self) -> None:
        time = _finite("time_s", self.time_s)
        if time < 0.0:
            raise ContractViolation("time_s must be non-negative")
        object.__setattr__(self, "time_s", time)


@dataclass(frozen=True, slots=True)
class HorizontalClosureSet:
    """Explicit, provenance-labelled wall/interphase closure selection."""

    wall_model: Literal["off", "smooth_pipe"]
    interphase_drag_coefficient: float
    evidence_status: str
    alignment_ready: bool = False

    def __post_init__(self) -> None:
        if self.wall_model not in ("off", "smooth_pipe"):
            raise ContractViolation("unsupported horizontal wall model")
        coefficient = _finite(
            "interphase_drag_coefficient", self.interphase_drag_coefficient
        )
        if coefficient < 0.0:
            raise ContractViolation("interphase drag coefficient must be non-negative")
        if not self.evidence_status.strip():
            raise ContractViolation("closure evidence status must be non-empty")
        if self.alignment_ready:
            raise ContractViolation(
                "no published S1 horizontal closure set is currently alignment-ready"
            )
        object.__setattr__(self, "interphase_drag_coefficient", coefficient)

    @classmethod
    def verification_inviscid(cls) -> "HorizontalClosureSet":
        return cls(
            wall_model="off",
            interphase_drag_coefficient=0.0,
            evidence_status="verification_only_zero_wall_and_interphase_drag",
        )

    @classmethod
    def declared_smooth_pipe_unvalidated(
        cls, *, interphase_drag_coefficient: float = 0.44
    ) -> "HorizontalClosureSet":
        return cls(
            wall_model="smooth_pipe",
            interphase_drag_coefficient=interphase_drag_coefficient,
            evidence_status=(
                "declared_generic_smooth_pipe_and_drag__not_published_not_tuned"
            ),
        )


@dataclass(frozen=True, slots=True)
class HorizontalDistributedConfig:
    liquid_density_kg_m3: float = 998.4
    liquid_viscosity_Pa_s: float = 1.002e-3
    gas_viscosity_Pa_s: float = 1.78e-5
    atmospheric_pressure_Pa: float = 101325.0
    gas_constant_J_kg_K: float = 287.05
    temperature_K: float = 293.15
    water_inlet_head_m: float = WATER_INLET_HEAD_M
    water_outlet_head_m: float = WATER_OUTLET_HEAD_M
    initial_water_surface_head_m: float = INITIAL_WATER_SURFACE_HEAD_M
    elastic_storage_reference_head_m: float = WATER_OUTLET_HEAD_M
    main_slope_sine: float = 0.0
    cfl: float = 0.35
    gas_presence_mass_kg_m: float = 1.0e-12
    gas_presence_area_fraction: float = 1.0e-8
    maximum_elastic_overarea_fraction: float = 0.02
    maximum_substeps: int = 200000

    def __post_init__(self) -> None:
        for name in (
            "liquid_density_kg_m3",
            "liquid_viscosity_Pa_s",
            "gas_viscosity_Pa_s",
            "atmospheric_pressure_Pa",
            "gas_constant_J_kg_K",
            "temperature_K",
            "water_inlet_head_m",
            "water_outlet_head_m",
            "initial_water_surface_head_m",
            "elastic_storage_reference_head_m",
            "gas_presence_mass_kg_m",
            "gas_presence_area_fraction",
            "maximum_elastic_overarea_fraction",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ContractViolation(f"{name} must be positive")
            object.__setattr__(self, name, value)
        slope = _finite("main_slope_sine", self.main_slope_sine)
        if abs(slope) > 1.0:
            raise ContractViolation("main_slope_sine must lie in [-1,1]")
        object.__setattr__(self, "main_slope_sine", slope)
        cfl = _finite("cfl", self.cfl)
        if not 0.0 < cfl < 0.5:
            raise ContractViolation("cfl must lie in (0,0.5) for the shared-face update")
        object.__setattr__(self, "cfl", cfl)
        if self.maximum_substeps < 1:
            raise ContractViolation("maximum_substeps must be positive")

    @property
    def rt_J_kg(self) -> float:
        return self.gas_constant_J_kg_K * self.temperature_K

    @property
    def gas_sound_speed_m_s(self) -> float:
        return math.sqrt(self.rt_J_kg)


@dataclass(frozen=True, slots=True)
class GasPositionObservation:
    gas_cell_count: int
    tail_x_m: float | None
    nose_x_m: float | None
    mass_centroid_x_m: float | None
    total_main_gas_mass_kg: float
    total_main_gas_volume_m3: float


@dataclass(frozen=True, slots=True)
class HorizontalInventory:
    liquid_volume_m3: float
    total_gas_mass_kg: float
    main_mixture_momentum_x_kg_m_s: float
    stub_gas_momentum_z_kg_m_s: float


@dataclass(frozen=True, slots=True)
class HorizontalLedgerEntry:
    time_start_s: float
    time_end_s: float
    stage: Stage
    before: HorizontalInventory
    after: HorizontalInventory
    liquid_boundary_exchange_m3: float
    reservoir_gas_exchange_kg: float
    main_momentum_impulse_kg_m_s: float
    stub_momentum_impulse_kg_m_s: float
    liquid_volume_residual_m3: float
    gas_mass_residual_kg: float
    main_momentum_residual_kg_m_s: float
    stub_momentum_residual_kg_m_s: float
    maximum_courant: float
    node_mass_residual_kg_s: float
    interphase_recoil_residual_N_per_m_integral: float


@dataclass(frozen=True, slots=True)
class _TeeFlux:
    pressure_Pa: float
    stub_mass_out_kg_s: float
    left_mass_out_kg_s: float
    right_mass_out_kg_s: float
    stub_momentum_flux_N: float
    left_momentum_flux_N: float
    right_momentum_flux_N: float

    @property
    def mass_residual_kg_s(self) -> float:
        return self.stub_mass_out_kg_s + self.left_mass_out_kg_s + self.right_mass_out_kg_s


@dataclass(slots=True)
class _Arrays:
    Al: Array
    Ql: Array
    Mg: Array
    Jg: Array
    stub_Mg: Array
    stub_Jg: Array

    def affine(self, scale: float, derivative: "_Arrays") -> "_Arrays":
        return _Arrays(
            self.Al + scale * derivative.Al,
            self.Ql + scale * derivative.Ql,
            self.Mg + scale * derivative.Mg,
            self.Jg + scale * derivative.Jg,
            self.stub_Mg + scale * derivative.stub_Mg,
            self.stub_Jg + scale * derivative.stub_Jg,
        )


@dataclass(frozen=True, slots=True)
class _BudgetRate:
    liquid_boundary_m3_s: float
    reservoir_gas_kg_s: float
    main_momentum_kg_m_s2: float
    stub_momentum_kg_m_s2: float
    node_mass_residual_kg_s: float
    interphase_recoil_residual_N_per_m_integral: float


WaterEndSide = Literal["left", "right"]
WaterEndPhaseMode = Literal[
    "water_only",
    "interior_gas_outflow",
    "pure_water_reentry_no_gas_inventory",
]


@dataclass(frozen=True, slots=True)
class WaterEndInletOutletGasFlux:
    """One finite gas-phase boundary flux at a Table-1 water end.

    Mahyawansi Table 1 and the frozen 2-D translation prescribe
    ``alpha.water inletOutlet`` with ``inletValue=1`` at *both* water ends.
    Thus an outward gas trace may leave with the interior phase state, while
    reversed flow may admit water only.  No external gas density, velocity or
    inventory is published, so gas re-entry is identically unavailable.

    The signed mass and momentum fluxes use the horizontal left-to-right face
    orientation.  The mass outflow is bounded by the donor cell inventory over
    the proposed stage; this is a finite conservative flux, not an implicit
    atmospheric material reservoir.
    """

    side: WaterEndSide
    mode: WaterEndPhaseMode
    gas_mass_left_to_right_kg_s: float
    gas_momentum_left_to_right_N: float
    gas_inflow_kg_s: float
    gas_outflow_kg_s: float
    gas_face_area_m2: float
    prescribed_reentry_alpha_water: float
    evidence_status: str

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ContractViolation("water-end side must be left or right")
        if self.mode not in (
            "water_only",
            "interior_gas_outflow",
            "pure_water_reentry_no_gas_inventory",
        ):
            raise ContractViolation("unsupported water-end phase mode")
        for name in (
            "gas_mass_left_to_right_kg_s",
            "gas_momentum_left_to_right_N",
            "gas_inflow_kg_s",
            "gas_outflow_kg_s",
            "gas_face_area_m2",
            "prescribed_reentry_alpha_water",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"water-end flux {name} must be finite")
            object.__setattr__(self, name, value)
        if self.gas_inflow_kg_s != 0.0:
            raise ContractViolation(
                "Table-1 water-end inletOutlet has no published external gas inventory"
            )
        if self.gas_outflow_kg_s < 0.0 or self.gas_face_area_m2 < 0.0:
            raise ContractViolation("water-end gas outflow/area must be non-negative")
        if self.prescribed_reentry_alpha_water != 1.0:
            raise ContractViolation("Table-1 water-end re-entry must be pure water")
        if not self.evidence_status.strip():
            raise ContractViolation("water-end phase evidence must be non-empty")
        expected_outflow = abs(self.gas_mass_left_to_right_kg_s)
        if not math.isclose(
            self.gas_outflow_kg_s,
            expected_outflow,
            rel_tol=2.0e-12,
            abs_tol=1.0e-18,
        ):
            raise ContractViolation("water-end gas outflow lost its signed face flux")
        if self.side == "left" and self.gas_mass_left_to_right_kg_s > 0.0:
            raise ContractViolation("left water end cannot import unpublished gas")
        if self.side == "right" and self.gas_mass_left_to_right_kg_s < 0.0:
            raise ContractViolation("right water end cannot import unpublished gas")


def water_end_inlet_outlet_gas_flux(
    *,
    side: WaterEndSide,
    gas_area_m2: float,
    gas_mass_kg_m: float,
    gas_momentum_kg_s: float,
    interior_absolute_pressure_Pa: float,
    dx_m: float,
    dt_s: float,
    gas_presence_mass_kg_m: float,
    gas_presence_area_m2: float,
) -> WaterEndInletOutletGasFlux:
    """Translate the published water-end phase switch without a gas source.

    For an outward interior gas characteristic, the zero-gradient/outflow
    branch is the identical-state isothermal Euler flux
    ``mdot=Mg*u`` and ``F=mdot*u+p*Ag``.  For an inward characteristic the
    frozen ``inletValue alpha.water=1`` branch supplies no gas.  The liquid
    total-head/static-head characteristic remains owned separately by the
    hash-pinned Case-1 adapter.
    """

    if side not in ("left", "right"):
        raise ContractViolation("water-end side must be left or right")
    values = {
        "gas_area_m2": gas_area_m2,
        "gas_mass_kg_m": gas_mass_kg_m,
        "gas_momentum_kg_s": gas_momentum_kg_s,
        "interior_absolute_pressure_Pa": interior_absolute_pressure_Pa,
        "dx_m": dx_m,
        "dt_s": dt_s,
        "gas_presence_mass_kg_m": gas_presence_mass_kg_m,
        "gas_presence_area_m2": gas_presence_area_m2,
    }
    parsed = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ContractViolation("water-end inletOutlet inputs must be finite")
    area = parsed["gas_area_m2"]
    mass = parsed["gas_mass_kg_m"]
    momentum = parsed["gas_momentum_kg_s"]
    pressure = parsed["interior_absolute_pressure_Pa"]
    dx = parsed["dx_m"]
    dt = parsed["dt_s"]
    mass_floor = parsed["gas_presence_mass_kg_m"]
    area_floor = parsed["gas_presence_area_m2"]
    if area < 0.0 or mass < 0.0 or mass_floor < 0.0 or area_floor < 0.0:
        raise ContractViolation("water-end phase amounts/tolerances must be non-negative")
    if pressure <= 0.0 or dx <= 0.0 or dt <= 0.0:
        raise ContractViolation("water-end pressure, dx and dt must be positive")
    gas_present = area > area_floor or mass > mass_floor
    if gas_present and not (area > area_floor and mass > mass_floor):
        raise ContractViolation("water-end gas area and mass must be paired")
    if not gas_present:
        if abs(momentum) > 1.0e-12:
            raise ContractViolation("water-only end cannot carry gas momentum")
        return WaterEndInletOutletGasFlux(
            side=side,
            mode="water_only",
            gas_mass_left_to_right_kg_s=0.0,
            gas_momentum_left_to_right_N=0.0,
            gas_inflow_kg_s=0.0,
            gas_outflow_kg_s=0.0,
            gas_face_area_m2=0.0,
            prescribed_reentry_alpha_water=1.0,
            evidence_status=(
                "published_Table1_alpha_water_inletOutlet_inletValue_1__"
                "water_only__no_external_gas_inventory"
            ),
        )

    velocity = momentum / mass
    raw_mass_flux = mass * velocity
    outward = (side == "left" and raw_mass_flux < 0.0) or (
        side == "right" and raw_mass_flux > 0.0
    )
    if not outward:
        return WaterEndInletOutletGasFlux(
            side=side,
            mode="pure_water_reentry_no_gas_inventory",
            gas_mass_left_to_right_kg_s=0.0,
            gas_momentum_left_to_right_N=pressure * area,
            gas_inflow_kg_s=0.0,
            gas_outflow_kg_s=0.0,
            gas_face_area_m2=area,
            prescribed_reentry_alpha_water=1.0,
            evidence_status=(
                "published_Table1_alpha_water_inletOutlet_inletValue_1__"
                "pure_water_reentry__no_external_gas_inventory"
            ),
        )

    finite_donor_rate = mass * dx / dt
    signed_limit = -finite_donor_rate if side == "left" else finite_donor_rate
    if side == "left":
        mass_flux = max(raw_mass_flux, signed_limit)
    else:
        mass_flux = min(raw_mass_flux, signed_limit)
    gas_outflow = abs(mass_flux)
    return WaterEndInletOutletGasFlux(
        side=side,
        mode="interior_gas_outflow",
        gas_mass_left_to_right_kg_s=mass_flux,
        gas_momentum_left_to_right_N=pressure * area + mass_flux * velocity,
        gas_inflow_kg_s=0.0,
        gas_outflow_kg_s=gas_outflow,
        gas_face_area_m2=area,
        prescribed_reentry_alpha_water=1.0,
        evidence_status=(
            "published_Table1_alpha_water_inletOutlet_inletValue_1__"
            "interior_phase_zeroGradient_outflow__finite_cell_donor__"
            "no_external_gas_inventory"
        ),
    )


def _isothermal_hll_density_flux(
    rho_left: float,
    velocity_left: float,
    rho_right: float,
    velocity_right: float,
    rt: float,
) -> tuple[float, float]:
    """Return isothermal Euler mass and momentum flux per connection area."""

    rho_l = max(float(rho_left), 0.0)
    rho_r = max(float(rho_right), 0.0)
    u_l = float(velocity_left)
    u_r = float(velocity_right)
    c = math.sqrt(rt)
    ul = np.array((rho_l, rho_l * u_l), dtype=float)
    ur = np.array((rho_r, rho_r * u_r), dtype=float)
    fl = np.array((rho_l * u_l, rho_l * u_l * u_l + rho_l * rt), dtype=float)
    fr = np.array((rho_r * u_r, rho_r * u_r * u_r + rho_r * rt), dtype=float)
    s_l = min(u_l - c, u_r - c)
    s_r = max(u_l + c, u_r + c)
    if s_l >= 0.0:
        flux = fl
    elif s_r <= 0.0:
        flux = fr
    else:
        flux = (s_r * fl - s_l * fr + s_l * s_r * (ur - ul)) / (s_r - s_l)
    return float(flux[0]), float(flux[1])


def _darcy_factor(reynolds: float) -> float:
    if reynolds <= 1.0e-12:
        return 0.0
    if reynolds <= 2300.0:
        return 64.0 / reynolds
    if reynolds >= 4000.0:
        return 0.3164 / reynolds**0.25
    laminar = 64.0 / reynolds
    turbulent = 0.3164 / reynolds**0.25
    weight = (reynolds - 2300.0) / 1700.0
    return (1.0 - weight) * laminar + weight * turbulent


class HorizontalDistributedSolver:
    """SSP-RK2 finite-volume main/stub solver for short verification runs."""

    def __init__(
        self,
        adapter: Case1HorizontalLiquidAdapter,
        *,
        closures: HorizontalClosureSet | None,
        config: HorizontalDistributedConfig | None = None,
        air_stub: AirStubGeometry | None = None,
        pressure_reservoir: IsothermalIdealGasPressureReservoir | None = None,
    ) -> None:
        if closures is None:
            raise MissingPhysicalClosure(
                "horizontal wall/interphase closures must be selected explicitly"
            )
        self.adapter = adapter
        self.closures = closures
        self.config = HorizontalDistributedConfig() if config is None else config
        self.air_stub = AirStubGeometry() if air_stub is None else air_stub
        self.pressure_reservoir = (
            IsothermalIdealGasPressureReservoir()
            if pressure_reservoir is None
            else pressure_reservoir
        )
        self.area = adapter.full_area_m2
        self.dx = adapter.grid.dx_m
        self.diameter = adapter.grid.diameter_m
        self.tee_face = adapter.grid.air_tee_face_index
        if not 0 < self.tee_face < adapter.grid.cell_count:
            raise ContractViolation("air tee must be an internal main-pipe face")
        if not math.isclose(
            self.pressure_reservoir.reservoir_absolute_pressure_Pa,
            self.config.atmospheric_pressure_Pa + 5700.0,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ContractViolation("Stage-2 reservoir must remain the published 5700 Pa gauge")

    @property
    def alignment_ready(self) -> bool:
        return self.closures.alignment_ready

    @property
    def source_aligned_trajectory_ready(self) -> bool:
        """Remain false until the air stub can admit a resolved liquid phase.

        The present edge is deliberately gas-only.  Stage-1 pressure loading
        can move the main-pipe liquid and the finite stub gas inventory, but a
        water tongue cannot enter the stub and reduce its gas volume.  That
        missing two-phase branch physics blocks a source-aligned trajectory.
        """

        return False

    def assert_source_aligned_trajectory_ready(self) -> None:
        raise MissingPhysicalClosure(
            "the 0.1373 m air stub has persistent Mg/Jg but no liquid-intrusion "
            "state; Stage-1 source-aligned settling and formal Stage-2 promotion "
            "remain fail-closed"
        )

    def initial_state(self) -> DistributedHorizontalState:
        main = self.adapter.build_stage1_initial_state()
        # Isothermal hydrostatic gas column.  The published 5700 Pa top
        # pressure is used as the initial reference; Stage 1 subsequently
        # closes that top face and conserves this finite inventory.
        z = (np.arange(self.air_stub.cell_count, dtype=float) + 0.5) * self.air_stub.dz_m
        p_top = self.pressure_reservoir.reservoir_absolute_pressure_Pa
        pressure = p_top * np.exp(
            self.adapter.gravity_m_s2
            * (self.air_stub.length_m - z)
            / self.config.rt_J_kg
        )
        mass_per_length = pressure / self.config.rt_J_kg * self.area
        state = DistributedHorizontalState(
            time_s=0.0,
            main=main,
            air_stub=AirStubState(
                Mg=_tuple(mass_per_length),
                Jg=(0.0,) * self.air_stub.cell_count,
            ),
        )
        self.validate_state(state)
        return state

    def validate_state(self, state: DistributedHorizontalState) -> None:
        if state.main.cell_count != self.adapter.grid.cell_count:
            raise ContractViolation("main state/grid cell counts differ")
        if state.air_stub.cell_count != self.air_stub.cell_count:
            raise ContractViolation("air-stub state/grid cell counts differ")
        tolerance = 5.0e-13 * self.area
        maximum_elastic_area = self.area * (
            1.0 + self.config.maximum_elastic_overarea_fraction
        )
        for area, gas_mass, gas_momentum in zip(
            state.main.Al, state.main.Mg, state.main.Jg, strict=True
        ):
            if area < -tolerance or area > maximum_elastic_area:
                raise ContractViolation("main liquid area left its gas/elastic admissible interval")
            gas_area = max(self.area - area, 0.0)
            if gas_mass > self.config.gas_presence_mass_kg_m and area >= self.area - tolerance:
                raise ContractViolation("positive main gas mass has no complementary gas area")
            if gas_mass <= self.config.gas_presence_mass_kg_m and abs(gas_momentum) > 1.0e-10:
                raise ContractViolation("gas-vacuum cell carries finite momentum")

    def _arrays(self, state: DistributedHorizontalState) -> _Arrays:
        return _Arrays(
            np.asarray(state.main.Al, dtype=float),
            np.asarray(state.main.Ql, dtype=float),
            np.asarray(state.main.Mg, dtype=float),
            np.asarray(state.main.Jg, dtype=float),
            np.asarray(state.air_stub.Mg, dtype=float),
            np.asarray(state.air_stub.Jg, dtype=float),
        )

    def _state(self, arrays: _Arrays, time_s: float) -> DistributedHorizontalState:
        # Roundoff-scale vacuum momenta are zeroed only when the corresponding
        # mass is also roundoff scale.  No material clipping is performed.
        jg = arrays.Jg.copy()
        jg[arrays.Mg <= self.config.gas_presence_mass_kg_m] = 0.0
        state = DistributedHorizontalState(
            time_s=time_s,
            main=HorizontalState(
                Al=_tuple(arrays.Al),
                Ql=_tuple(arrays.Ql),
                Mg=_tuple(arrays.Mg),
                Jg=_tuple(jg),
            ),
            air_stub=AirStubState(
                Mg=_tuple(arrays.stub_Mg),
                Jg=_tuple(arrays.stub_Jg),
            ),
        )
        self.validate_state(state)
        return state

    def _main_pressures(self, arrays: _Arrays) -> Array:
        ag = np.maximum(self.area - arrays.Al, 0.0)
        pressure = np.empty_like(arrays.Al)
        gas = (arrays.Mg > self.config.gas_presence_mass_kg_m) & (
            ag > self.config.gas_presence_area_fraction * self.area
        )
        pressure[gas] = arrays.Mg[gas] * self.config.rt_J_kg / ag[gas]
        # In gas-free cells the common (crown) pressure is only the published
        # reference offset.  The area-dependent hydrostatic/elastic increment
        # is already inside the Case-1 physical flux and must not be counted a
        # second time here.  Once gas occupies a cell, the common pressure is
        # replaced by the exact gas EOS above.
        # Table 1 heads are referenced to the pipe centreline (the paper's
        # vertical coordinate is zero there).  Case-1's circular section uses
        # depth/head above the invert, while this common pressure acts at the
        # crown.  The crown is one radius, not one diameter, above the source
        # datum; subtracting D here would understate pressure by rho*g*D/2.
        p_reference = self.config.atmospheric_pressure_Pa + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * (self.config.water_inlet_head_m - 0.5 * self.diameter)
        )
        pressure[~gas] = p_reference
        if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise ContractViolation("main common pressure must remain finite and positive")
        return pressure

    def gas_pressure_Pa(self, state: DistributedHorizontalState) -> tuple[float | None, ...]:
        """Return exact EOS pressures in occupied cells and ``None`` elsewhere."""

        result: list[float | None] = []
        for al, mg in zip(state.main.Al, state.main.Mg, strict=True):
            ag = max(self.area - al, 0.0)
            if (
                mg > self.config.gas_presence_mass_kg_m
                and ag > self.config.gas_presence_area_fraction * self.area
            ):
                result.append(mg * self.config.rt_J_kg / ag)
            else:
                result.append(None)
        return tuple(result)

    def _gas_velocity(self, mass: float, momentum: float) -> float:
        if mass <= self.config.gas_presence_mass_kg_m:
            return 0.0
        return momentum / mass

    def _tee_flux(self, arrays: _Arrays) -> _TeeFlux:
        left = self.tee_face - 1
        right = self.tee_face
        ag = np.maximum(self.area - arrays.Al, 0.0)
        rt = self.config.rt_J_kg

        stub_rho = arrays.stub_Mg[0] / self.area
        stub_u = self._gas_velocity(arrays.stub_Mg[0], arrays.stub_Jg[0])

        def branch(
            node_pressure: float, rho_cell: float, velocity_out: float, connection_area: float
        ) -> tuple[float, float]:
            if connection_area <= self.config.gas_presence_area_fraction * self.area:
                return 0.0, node_pressure * 0.0
            mass_flux, momentum_flux = _isothermal_hll_density_flux(
                node_pressure / rt, 0.0, rho_cell, velocity_out, rt
            )
            return mass_flux * connection_area, momentum_flux * connection_area

        def all_fluxes(node_pressure: float) -> tuple[tuple[float, float], ...]:
            stub = branch(node_pressure, stub_rho, stub_u, self.area)
            rho_left = 0.0 if ag[left] <= 0.0 else arrays.Mg[left] / ag[left]
            rho_right = 0.0 if ag[right] <= 0.0 else arrays.Mg[right] / ag[right]
            u_left_out = -self._gas_velocity(arrays.Mg[left], arrays.Jg[left])
            u_right_out = self._gas_velocity(arrays.Mg[right], arrays.Jg[right])
            return (
                stub,
                branch(node_pressure, rho_left, u_left_out, max(ag[left], 0.0)),
                branch(node_pressure, rho_right, u_right_out, max(ag[right], 0.0)),
            )

        def residual(node_pressure: float) -> float:
            return sum(pair[0] for pair in all_fluxes(node_pressure))

        cell_pressures = [arrays.stub_Mg[0] / self.area * rt]
        for index in (left, right):
            if ag[index] > self.config.gas_presence_area_fraction * self.area:
                cell_pressures.append(arrays.Mg[index] / ag[index] * rt)
        lo = max(1.0e-3, min(cell_pressures) * 0.05)
        hi = max(cell_pressures) * 4.0 + 1.0
        f_lo = residual(lo)
        f_hi = residual(hi)
        for _ in range(20):
            if f_lo <= 0.0 <= f_hi:
                break
            lo *= 0.25
            hi *= 4.0
            f_lo = residual(lo)
            f_hi = residual(hi)
        else:
            raise ContractViolation("could not bracket the zero-storage air-tee pressure")
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        pressure = 0.5 * (lo + hi)
        stub, left_flux, right_flux = all_fluxes(pressure)
        result = _TeeFlux(
            pressure_Pa=pressure,
            stub_mass_out_kg_s=stub[0],
            left_mass_out_kg_s=left_flux[0],
            right_mass_out_kg_s=right_flux[0],
            stub_momentum_flux_N=stub[1],
            left_momentum_flux_N=left_flux[1],
            right_momentum_flux_N=right_flux[1],
        )
        if abs(result.mass_residual_kg_s) > 1.0e-12:
            raise ContractViolation("air-tee Riemann solve did not conserve gas mass")
        return result

    def _liquid_rusanov(
        self, area_left: float, q_left: float, area_right: float, q_right: float
    ) -> tuple[float, float]:
        flux_left = self.adapter.physical_flux(area_left, q_left)
        flux_right = self.adapter.physical_flux(area_right, q_right)
        velocity_left = 0.0 if area_left <= 0.0 else q_left / area_left
        velocity_right = 0.0 if area_right <= 0.0 else q_right / area_right
        speed = max(
            abs(velocity_left) + self.adapter.celerity_m_s(area_left),
            abs(velocity_right) + self.adapter.celerity_m_s(area_right),
        )
        return (
            0.5 * (flux_left.liquid_volume_m3_s + flux_right.liquid_volume_m3_s)
            - 0.5 * speed * (area_right - area_left),
            0.5 * (flux_left.liquid_momentum_m4_s2 + flux_right.liquid_momentum_m4_s2)
            - 0.5 * speed * (q_right - q_left),
        )

    def _rhs(self, state: DistributedHorizontalState, stage: Stage) -> tuple[_Arrays, _BudgetRate]:
        arrays = self._arrays(state)
        n = arrays.Al.size
        if stage not in ("stage1_closed", "stage2_pressure_reservoir"):
            raise ContractViolation(f"unknown horizontal stage {stage!r}")
        ag = np.maximum(self.area - arrays.Al, 0.0)
        pressure = self._main_pressures(arrays)
        tee = self._tee_flux(arrays)

        liquid_mass_flux = np.zeros(n + 1)
        liquid_momentum_flux = np.zeros(n + 1)
        liquid_pressure_area = np.full(n + 1, self.area)

        p_left = self.config.atmospheric_pressure_Pa + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * (self.config.water_inlet_head_m - 0.5 * self.diameter)
        )
        p_right = self.config.atmospheric_pressure_Pa + (
            self.config.liquid_density_kg_m3
            * self.adapter.gravity_m_s2
            * (self.config.water_outlet_head_m - 0.5 * self.diameter)
        )
        liquid_mass_flux[0], base = self._liquid_rusanov(
            self.area, 0.0, arrays.Al[0], arrays.Ql[0]
        )
        liquid_momentum_flux[0] = base + p_left * self.area / self.config.liquid_density_kg_m3
        liquid_mass_flux[n], base = self._liquid_rusanov(
            arrays.Al[-1], arrays.Ql[-1], self.area, 0.0
        )
        liquid_momentum_flux[n] = base + p_right * self.area / self.config.liquid_density_kg_m3

        for face in range(1, n):
            liquid_mass_flux[face], base = self._liquid_rusanov(
                arrays.Al[face - 1],
                arrays.Ql[face - 1],
                arrays.Al[face],
                arrays.Ql[face],
            )
            connection_gas_area = max(min(ag[face - 1], ag[face]), 0.0)
            liquid_pressure_area[face] = self.area - connection_gas_area
            p_face = tee.pressure_Pa if face == self.tee_face else 0.5 * (
                pressure[face - 1] + pressure[face]
            )
            liquid_momentum_flux[face] = (
                base
                + p_face
                * liquid_pressure_area[face]
                / self.config.liquid_density_kg_m3
            )

        gas_left_mass = np.zeros(n)
        gas_right_mass = np.zeros(n)
        gas_left_momentum = np.zeros(n)
        gas_right_momentum = np.zeros(n)
        # Water reservoirs admit water only.  The gas phase is reflected until
        # a dedicated gas/water outlet closure is sourced; the short smoke must
        # never reach these ends.
        gas_left_momentum[0] = pressure[0] * max(ag[0], 0.0)
        gas_right_momentum[-1] = pressure[-1] * max(ag[-1], 0.0)

        for face in range(1, n):
            if face == self.tee_face:
                continue
            connection_area = max(min(ag[face - 1], ag[face]), 0.0)
            if connection_area <= self.config.gas_presence_area_fraction * self.area:
                mass_flux = 0.0
                momentum_flux = 0.0
            else:
                rho_l = arrays.Mg[face - 1] / max(ag[face - 1], 1.0e-300)
                rho_r = arrays.Mg[face] / max(ag[face], 1.0e-300)
                u_l = self._gas_velocity(arrays.Mg[face - 1], arrays.Jg[face - 1])
                u_r = self._gas_velocity(arrays.Mg[face], arrays.Jg[face])
                mass_density, momentum_density = _isothermal_hll_density_flux(
                    rho_l, u_l, rho_r, u_r, self.config.rt_J_kg
                )
                mass_flux = mass_density * connection_area
                momentum_flux = momentum_density * connection_area
            gas_right_mass[face - 1] = mass_flux
            gas_left_mass[face] = mass_flux
            gas_right_momentum[face - 1] = momentum_flux
            gas_left_momentum[face] = momentum_flux

        left_cell = self.tee_face - 1
        right_cell = self.tee_face
        gas_right_mass[left_cell] = -tee.left_mass_out_kg_s
        gas_right_momentum[left_cell] = tee.left_momentum_flux_N
        gas_left_mass[right_cell] = tee.right_mass_out_kg_s
        gas_left_momentum[right_cell] = tee.right_momentum_flux_N

        dAl = -(liquid_mass_flux[1:] - liquid_mass_flux[:-1]) / self.dx
        dQl = -(liquid_momentum_flux[1:] - liquid_momentum_flux[:-1]) / self.dx
        dMg = -(gas_right_mass - gas_left_mass) / self.dx
        dJg = -(gas_right_momentum - gas_left_momentum) / self.dx

        # Equal/opposite phase-area pressure geometry terms.
        d_area_dx = (liquid_pressure_area[1:] - liquid_pressure_area[:-1]) / self.dx
        dQl += pressure / self.config.liquid_density_kg_m3 * d_area_dx
        dJg -= pressure * d_area_dx

        gravity = self.adapter.gravity_m_s2
        dQl += arrays.Al * gravity * self.config.main_slope_sine
        dJg += arrays.Mg * gravity * self.config.main_slope_sine

        interphase_residual = 0.0
        if self.closures.wall_model == "smooth_pipe" or self.closures.interphase_drag_coefficient > 0.0:
            circumference = math.pi * self.diameter
            for index in range(n):
                al = arrays.Al[index]
                gas_area = max(ag[index], 0.0)
                ul = 0.0 if al <= 0.0 else arrays.Ql[index] / al
                ug = self._gas_velocity(arrays.Mg[index], arrays.Jg[index])
                liquid_perimeter = self.adapter.wetted_perimeter_m(min(al, self.area))
                gas_perimeter = max(circumference - liquid_perimeter, 0.0)
                if self.closures.wall_model == "smooth_pipe":
                    hydraulic_d_l = 0.0 if liquid_perimeter <= 0.0 else 4.0 * al / liquid_perimeter
                    re_l = (
                        self.config.liquid_density_kg_m3
                        * abs(ul)
                        * hydraulic_d_l
                        / self.config.liquid_viscosity_Pa_s
                    )
                    f_l = _darcy_factor(re_l)
                    dQl[index] -= f_l * liquid_perimeter * ul * abs(ul) / 8.0
                    if gas_area > 0.0 and arrays.Mg[index] > 0.0 and gas_perimeter > 0.0:
                        rho_g = arrays.Mg[index] / gas_area
                        hydraulic_d_g = 4.0 * gas_area / gas_perimeter
                        re_g = rho_g * abs(ug) * hydraulic_d_g / self.config.gas_viscosity_Pa_s
                        f_g = _darcy_factor(re_g)
                        dJg[index] -= f_g * gas_perimeter * rho_g * ug * abs(ug) / 8.0
                if (
                    self.closures.interphase_drag_coefficient > 0.0
                    and gas_area > 0.0
                    and arrays.Mg[index] > 0.0
                ):
                    rho_g = arrays.Mg[index] / gas_area
                    width = self.adapter.interface_width_m(min(al, self.area))
                    slip = ug - ul
                    force_on_liquid = (
                        0.5
                        * self.closures.interphase_drag_coefficient
                        * rho_g
                        * width
                        * slip
                        * abs(slip)
                    )
                    liquid_acceleration = force_on_liquid / self.config.liquid_density_kg_m3
                    dQl[index] += liquid_acceleration
                    dJg[index] -= force_on_liquid
                    interphase_residual += (
                        self.config.liquid_density_kg_m3 * liquid_acceleration
                        - force_on_liquid
                    ) * self.dx

        # Persistent gas-only vertical stub.  Positive flux/momentum points up.
        ns = arrays.stub_Mg.size
        stub_mass_flux = np.zeros(ns + 1)
        stub_momentum_flux = np.zeros(ns + 1)
        stub_mass_flux[0] = tee.stub_mass_out_kg_s
        stub_momentum_flux[0] = tee.stub_momentum_flux_N
        for face in range(1, ns):
            rho_l = arrays.stub_Mg[face - 1] / self.area
            rho_r = arrays.stub_Mg[face] / self.area
            u_l = self._gas_velocity(arrays.stub_Mg[face - 1], arrays.stub_Jg[face - 1])
            u_r = self._gas_velocity(arrays.stub_Mg[face], arrays.stub_Jg[face])
            fm, fj = _isothermal_hll_density_flux(
                rho_l, u_l, rho_r, u_r, self.config.rt_J_kg
            )
            stub_mass_flux[face] = fm * self.area
            stub_momentum_flux[face] = fj * self.area
        top_pressure = arrays.stub_Mg[-1] / self.area * self.config.rt_J_kg
        top_velocity = self._gas_velocity(arrays.stub_Mg[-1], arrays.stub_Jg[-1])
        if stage == "stage1_closed":
            stub_mass_flux[-1] = 0.0
            stub_momentum_flux[-1] = (
                arrays.stub_Mg[-1] / self.area * top_velocity * top_velocity
                + top_pressure
            ) * self.area
        else:
            reservoir_flux = self.pressure_reservoir.evaluate(
                node_absolute_pressure_Pa=top_pressure,
                node_axial_velocity_m_s=-top_velocity,
                inlet_area_m2=self.area,
            )
            stub_mass_flux[-1] = -reservoir_flux.mass_flow_kg_s
            stub_momentum_flux[-1] = reservoir_flux.axial_momentum_pressure_rate_N

        dz = self.air_stub.dz_m
        d_stub_Mg = -(stub_mass_flux[1:] - stub_mass_flux[:-1]) / dz
        d_stub_Jg = -(stub_momentum_flux[1:] - stub_momentum_flux[:-1]) / dz
        d_stub_Jg -= arrays.stub_Mg * gravity
        if self.closures.wall_model == "smooth_pipe":
            perimeter = math.pi * self.diameter
            for index in range(ns):
                rho_g = arrays.stub_Mg[index] / self.area
                velocity = self._gas_velocity(arrays.stub_Mg[index], arrays.stub_Jg[index])
                re = rho_g * abs(velocity) * self.diameter / self.config.gas_viscosity_Pa_s
                f_g = _darcy_factor(re)
                d_stub_Jg[index] -= f_g * perimeter * rho_g * velocity * abs(velocity) / 8.0

        derivative = _Arrays(dAl, dQl, dMg, dJg, d_stub_Mg, d_stub_Jg)
        liquid_boundary_rate = liquid_mass_flux[0] - liquid_mass_flux[-1]
        reservoir_gas_rate = -stub_mass_flux[-1]
        total_gas_rate = float(np.sum(dMg) * self.dx + np.sum(d_stub_Mg) * dz)
        if abs(total_gas_rate - reservoir_gas_rate) > 2.0e-11:
            raise ContractViolation("distributed gas RHS failed its mass ledger")
        total_liquid_rate = float(np.sum(dAl) * self.dx)
        if abs(total_liquid_rate - liquid_boundary_rate) > 2.0e-11:
            raise ContractViolation("distributed liquid RHS failed its volume ledger")
        main_momentum_rate = float(
            np.sum(self.config.liquid_density_kg_m3 * dQl + dJg) * self.dx
        )
        stub_momentum_rate = float(np.sum(d_stub_Jg) * dz)
        return derivative, _BudgetRate(
            liquid_boundary_m3_s=float(liquid_boundary_rate),
            reservoir_gas_kg_s=float(reservoir_gas_rate),
            main_momentum_kg_m_s2=main_momentum_rate,
            stub_momentum_kg_m_s2=stub_momentum_rate,
            node_mass_residual_kg_s=tee.mass_residual_kg_s,
            interphase_recoil_residual_N_per_m_integral=interphase_residual,
        )

    def inventory(self, state: DistributedHorizontalState) -> HorizontalInventory:
        liquid = sum(state.main.Al) * self.dx
        gas = sum(state.main.Mg) * self.dx + sum(state.air_stub.Mg) * self.air_stub.dz_m
        momentum_x = sum(
            self.config.liquid_density_kg_m3 * q + j
            for q, j in zip(state.main.Ql, state.main.Jg, strict=True)
        ) * self.dx
        momentum_z = sum(state.air_stub.Jg) * self.air_stub.dz_m
        return HorizontalInventory(liquid, gas, momentum_x, momentum_z)

    def stable_timestep_s(self, state: DistributedHorizontalState) -> float:
        arrays = self._arrays(state)
        liquid_speed = 0.0
        for area, discharge in zip(arrays.Al, arrays.Ql, strict=True):
            velocity = 0.0 if area <= 0.0 else abs(discharge / area)
            liquid_speed = max(liquid_speed, velocity + self.adapter.celerity_m_s(float(area)))
        gas_speed = self.config.gas_sound_speed_m_s
        for mass, momentum in zip(arrays.Mg, arrays.Jg, strict=True):
            gas_speed = max(gas_speed, abs(self._gas_velocity(mass, momentum)) + self.config.gas_sound_speed_m_s)
        stub_speed = max(
            abs(self._gas_velocity(mass, momentum)) + self.config.gas_sound_speed_m_s
            for mass, momentum in zip(arrays.stub_Mg, arrays.stub_Jg, strict=True)
        )
        return self.config.cfl * min(
            self.dx / max(liquid_speed, gas_speed),
            self.air_stub.dz_m / stub_speed,
        )

    def _ssprk2_step(
        self, state: DistributedHorizontalState, dt_s: float, stage: Stage
    ) -> tuple[DistributedHorizontalState, HorizontalLedgerEntry]:
        dt = _finite("dt_s", dt_s)
        if dt <= 0.0:
            raise ContractViolation("dt_s must be positive")
        stable = self.stable_timestep_s(state)
        if dt > stable * (1.0 + 1.0e-12):
            raise ContractViolation(f"requested dt {dt:.6e} exceeds CFL limit {stable:.6e}")
        base = self._arrays(state)
        derivative0, budget0 = self._rhs(state, stage)
        stage1_arrays = base.affine(dt, derivative0)
        stage1 = self._state(stage1_arrays, state.time_s + dt)
        derivative1, budget1 = self._rhs(stage1, stage)
        euler2 = stage1_arrays.affine(dt, derivative1)
        final_arrays = _Arrays(
            0.5 * (base.Al + euler2.Al),
            0.5 * (base.Ql + euler2.Ql),
            0.5 * (base.Mg + euler2.Mg),
            0.5 * (base.Jg + euler2.Jg),
            0.5 * (base.stub_Mg + euler2.stub_Mg),
            0.5 * (base.stub_Jg + euler2.stub_Jg),
        )
        final = self._state(final_arrays, state.time_s + dt)
        before = self.inventory(state)
        after = self.inventory(final)
        liquid_exchange = 0.5 * dt * (
            budget0.liquid_boundary_m3_s + budget1.liquid_boundary_m3_s
        )
        gas_exchange = 0.5 * dt * (
            budget0.reservoir_gas_kg_s + budget1.reservoir_gas_kg_s
        )
        impulse_x = 0.5 * dt * (
            budget0.main_momentum_kg_m_s2 + budget1.main_momentum_kg_m_s2
        )
        impulse_z = 0.5 * dt * (
            budget0.stub_momentum_kg_m_s2 + budget1.stub_momentum_kg_m_s2
        )
        residuals = (
            after.liquid_volume_m3 - before.liquid_volume_m3 - liquid_exchange,
            after.total_gas_mass_kg - before.total_gas_mass_kg - gas_exchange,
            after.main_mixture_momentum_x_kg_m_s
            - before.main_mixture_momentum_x_kg_m_s
            - impulse_x,
            after.stub_gas_momentum_z_kg_m_s
            - before.stub_gas_momentum_z_kg_m_s
            - impulse_z,
        )
        if max(abs(value) for value in residuals) > 5.0e-10:
            raise ContractViolation(f"SSP-RK2 conservation ledger failed: {residuals!r}")
        maximum_courant = dt / stable * self.config.cfl
        return final, HorizontalLedgerEntry(
            time_start_s=state.time_s,
            time_end_s=final.time_s,
            stage=stage,
            before=before,
            after=after,
            liquid_boundary_exchange_m3=liquid_exchange,
            reservoir_gas_exchange_kg=gas_exchange,
            main_momentum_impulse_kg_m_s=impulse_x,
            stub_momentum_impulse_kg_m_s=impulse_z,
            liquid_volume_residual_m3=residuals[0],
            gas_mass_residual_kg=residuals[1],
            main_momentum_residual_kg_m_s=residuals[2],
            stub_momentum_residual_kg_m_s=residuals[3],
            maximum_courant=maximum_courant,
            node_mass_residual_kg_s=max(
                abs(budget0.node_mass_residual_kg_s),
                abs(budget1.node_mass_residual_kg_s),
            ),
            interphase_recoil_residual_N_per_m_integral=max(
                abs(budget0.interphase_recoil_residual_N_per_m_integral),
                abs(budget1.interphase_recoil_residual_N_per_m_integral),
            ),
        )

    def advance(
        self, state: DistributedHorizontalState, duration_s: float, *, stage: Stage
    ) -> tuple[DistributedHorizontalState, tuple[HorizontalLedgerEntry, ...]]:
        duration = _finite("duration_s", duration_s)
        if duration <= 0.0:
            raise ContractViolation("duration_s must be positive")
        target = state.time_s + duration
        current = state
        ledger: list[HorizontalLedgerEntry] = []
        for _ in range(self.config.maximum_substeps):
            if current.time_s >= target - 1.0e-14 * max(1.0, target):
                return current, tuple(ledger)
            dt = min(self.stable_timestep_s(current), target - current.time_s)
            # Positivity is enforced by rejection, never by material clipping.
            for _attempt in range(30):
                try:
                    current, entry = self._ssprk2_step(current, dt, stage)
                    ledger.append(entry)
                    break
                except ContractViolation:
                    dt *= 0.5
                    if dt <= 1.0e-12:
                        raise
            else:
                raise ContractViolation("positivity/CFL step rejection did not recover")
        raise ContractViolation("maximum_substeps reached before requested duration")

    def gas_positions(self, state: DistributedHorizontalState) -> GasPositionObservation:
        centers = self.adapter.grid.x_left_m + (
            np.arange(state.main.cell_count, dtype=float) + 0.5
        ) * self.dx
        al = np.asarray(state.main.Al)
        mg = np.asarray(state.main.Mg)
        ag = np.maximum(self.area - al, 0.0)
        present = (mg > self.config.gas_presence_mass_kg_m) & (
            ag > self.config.gas_presence_area_fraction * self.area
        )
        total_mass = float(np.sum(mg) * self.dx)
        total_volume = float(np.sum(np.maximum(ag, 0.0)) * self.dx)
        if not np.any(present):
            return GasPositionObservation(0, None, None, None, total_mass, total_volume)
        indices = np.flatnonzero(present)
        weights = mg[present]
        return GasPositionObservation(
            gas_cell_count=int(indices.size),
            tail_x_m=float(centers[indices[0]] - 0.5 * self.dx),
            nose_x_m=float(centers[indices[-1]] + 0.5 * self.dx),
            mass_centroid_x_m=float(np.sum(centers[present] * weights) / np.sum(weights)),
            total_main_gas_mass_kg=total_mass,
            total_main_gas_volume_m3=total_volume,
        )


__all__ = [
    "AIR_STUB_EVIDENCE",
    "AIR_STUB_LENGTH_M",
    "AirStubGeometry",
    "AirStubState",
    "DistributedHorizontalState",
    "GasPositionObservation",
    "HorizontalClosureSet",
    "HorizontalDistributedConfig",
    "HorizontalDistributedSolver",
    "HorizontalInventory",
    "HorizontalLedgerEntry",
    "WaterEndInletOutletGasFlux",
    "WaterEndPhaseMode",
    "WaterEndSide",
    "WATER_INLET_HEAD_M",
    "WATER_OUTLET_HEAD_M",
    "water_end_inlet_outlet_gas_flux",
]
