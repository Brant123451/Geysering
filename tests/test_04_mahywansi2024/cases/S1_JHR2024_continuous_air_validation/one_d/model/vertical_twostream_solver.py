"""Component-level S1 riser time advance using the pinned Case-1 core.

This module is intentionally narrower than the complete coupled 1-D model.  It
owns a real finite-volume time advance for the S1 riser, but the finite riser
node still has to supply all gross bottom fluxes and the liquid pressure
closure explicitly.  Missing inputs or unpublished constitutive coefficients
therefore fail closed; a successful component smoke is *not* an eruption or a
production-trajectory claim.

The persistent S1 state is ``(Aup,Qup,Adown,Qdown,Mg,Jg)``.  The two liquid
rates are never reconstructed from their net value.  Internally the adapter
uses the pinned Case-1 signed convention, in which the downward discharge is
negative, and translates the sign once at the component boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Iterable

from .errors import ContractViolation, MissingPhysicalClosure
from .state import VerticalState
from .vertical_case1_adapter import (
    ATMOSPHERIC_PRESSURE_PA,
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    INITIAL_AIR_TEMPERATURE_K,
    PIPE_DIAMETER_M,
    RISER_BOTTOM_Z_M,
    RISER_TOP_Z_M,
    Case1VerticalComponentAdapter,
    build_s1_vertical_component,
)


S1_LIQUID_DENSITY_KG_M3 = 998.4
S1_GAS_VISCOSITY_PA_S = 1.78e-5
S1_GRAVITY_M_S2 = 9.81


@dataclass(frozen=True, slots=True)
class S1VerticalClosures:
    """Explicit constitutive inputs for the pinned liquid FV operator.

    Mahyawansi et al. do not publish these three one-dimensional
    coefficients.  The default therefore contains ``None`` and cannot
    advance.  ``structural_zero_for_tests`` is a declared numerical test
    closure used only for hydrostatic/conservation/smoke verification; it is
    never production-ready and must not be tuned against an eruption result.
    """

    wall_friction_up: float | None = None
    wall_friction_down: float | None = None
    interstream_drag: float | None = None
    provenance: str = "missing__not_published_for_s1"
    validation_only: bool = False
    apply_physical_three_body_drag: bool = True

    def __post_init__(self) -> None:
        for name in (
            "wall_friction_up",
            "wall_friction_down",
            "interstream_drag",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ContractViolation(f"{name} must be finite and non-negative")
        if not self.provenance.strip():
            raise ContractViolation("closure provenance must be non-empty")

    @property
    def computationally_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.wall_friction_up,
                self.wall_friction_down,
                self.interstream_drag,
            )
        )

    @property
    def production_ready(self) -> bool:
        # The complete horizontal/T-node/riser Riemann coupling remains open.
        return False

    def require_computational_closure(self) -> None:
        if not self.computationally_complete:
            raise MissingPhysicalClosure(
                "S1 riser advance requires explicit wall_friction_up, "
                "wall_friction_down, and interstream_drag; the paper does "
                "not publish them and this component does not infer them"
            )

    @classmethod
    def structural_zero_for_tests(cls) -> "S1VerticalClosures":
        return cls(
            wall_friction_up=0.0,
            wall_friction_down=0.0,
            interstream_drag=0.0,
            provenance="declared_zero__structural_validation_only",
            validation_only=True,
            apply_physical_three_body_drag=True,
        )


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractViolation(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class RiserNodeFluxPacket:
    """One explicit bottom-boundary transaction from the finite riser node.

    Liquid and gas directions are gross non-negative magnitudes.  Upward
    values enter the riser at its bottom; downward values leave it.  Speeds
    are required with every positive gross rate, so momentum cannot be
    silently reconstructed from a net flux.  The liquid-filled cell-pressure
    vector is the external common-pressure closure; omitting it is legal only
    when every riser cell contains a finite gas void.
    """

    bottom_pressure_pa: float
    liquid_filled_cell_pressure_pa: tuple[float, ...] | None = None
    liquid_upward_rate_m3_s: float = 0.0
    liquid_upward_speed_m_s: float = 0.0
    liquid_downward_rate_m3_s: float = 0.0
    liquid_downward_speed_m_s: float = 0.0
    gas_upward_mass_rate_kg_s: float = 0.0
    gas_upward_speed_m_s: float = 0.0
    gas_downward_mass_rate_kg_s: float = 0.0
    gas_downward_speed_m_s: float = 0.0

    def __post_init__(self) -> None:
        pressure = float(self.bottom_pressure_pa)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ContractViolation("bottom_pressure_pa must be finite and positive")
        object.__setattr__(self, "bottom_pressure_pa", pressure)
        if self.liquid_filled_cell_pressure_pa is not None:
            filled = tuple(float(value) for value in self.liquid_filled_cell_pressure_pa)
            if not filled or not all(math.isfinite(value) and value > 0.0 for value in filled):
                raise ContractViolation(
                    "liquid_filled_cell_pressure_pa must be a non-empty positive vector"
                )
            object.__setattr__(self, "liquid_filled_cell_pressure_pa", filled)
        for name in (
            "liquid_upward_rate_m3_s",
            "liquid_upward_speed_m_s",
            "liquid_downward_rate_m3_s",
            "liquid_downward_speed_m_s",
            "gas_upward_mass_rate_kg_s",
            "gas_upward_speed_m_s",
            "gas_downward_mass_rate_kg_s",
            "gas_downward_speed_m_s",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        for rate_name, speed_name in (
            ("liquid_upward_rate_m3_s", "liquid_upward_speed_m_s"),
            ("liquid_downward_rate_m3_s", "liquid_downward_speed_m_s"),
            ("gas_upward_mass_rate_kg_s", "gas_upward_speed_m_s"),
            ("gas_downward_mass_rate_kg_s", "gas_downward_speed_m_s"),
        ):
            if getattr(self, rate_name) > 0.0 and getattr(self, speed_name) <= 0.0:
                raise ContractViolation(f"positive {rate_name} requires positive {speed_name}")


@dataclass(frozen=True, slots=True)
class GasTransportLedger:
    initial_mass_kg: float
    final_mass_kg: float
    bottom_net_mass_rate_kg_s: float
    top_net_mass_rate_kg_s: float
    mass_residual_kg: float
    initial_momentum_kg_m_s: float
    final_momentum_kg_m_s: float
    boundary_advective_impulse_kg_m_s: float
    pressure_impulse_kg_m_s: float
    gravity_impulse_kg_m_s: float
    momentum_residual_kg_m_s: float


@dataclass(frozen=True, slots=True)
class VerticalTwoStreamAdvanceResult:
    """One accepted riser component stage and its hard conservation audits."""

    state: VerticalState
    common_pressure_faces_pa: tuple[float, ...]
    gas_pressure_cells_pa: tuple[float, ...]
    top_liquid_outflow_rate_m3_s: float
    top_liquid_outflow_volume_m3: float
    top_gas_outflow_rate_kg_s: float
    liquid_volume_residual_m3: float
    gas: GasTransportLedger
    three_body_momentum_residual_kg_m_s: float
    mixture_momentum_residual_kg_m_s: float
    maximum_packing_residual_m2: float
    validation_only: bool
    production_ready: bool = False


@dataclass(frozen=True, slots=True)
class _Case1Runtime:
    fv: ModuleType
    closures: ModuleType


_RUNTIME_LOCK = threading.RLock()
_RUNTIME_CACHE: dict[Path, _Case1Runtime] = {}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractViolation(f"cannot load pinned Case-1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_pinned_runtime(adapter: Case1VerticalComponentAdapter) -> _Case1Runtime:
    """Load the already hash-verified FV core and closure module as one type graph."""

    model_dir = adapter.pin.model_dir.resolve()
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(model_dir)
        if cached is not None:
            return cached
        tag = dict(adapter.pin.actual_sha256)["casea_vertical_twostream_fv.py"][:12].lower()
        fv_name = f"_s1_pinned_case1_vertical_fv_{tag}"
        closure_tag = dict(adapter.pin.actual_sha256)[
            "casea_vertical_twostream_closures.py"
        ][:12].lower()
        closure_name = f"_s1_pinned_case1_vertical_closures_{closure_tag}"
        fv = sys.modules.get(fv_name)
        if fv is None:
            fv = _load_module(fv_name, model_dir / "casea_vertical_twostream_fv.py")
        previous = sys.modules.get("casea_vertical_twostream_fv")
        sys.modules["casea_vertical_twostream_fv"] = fv
        try:
            closures = sys.modules.get(closure_name)
            if closures is None:
                closures = _load_module(
                    closure_name,
                    model_dir / "casea_vertical_twostream_closures.py",
                )
        finally:
            if previous is None:
                sys.modules.pop("casea_vertical_twostream_fv", None)
            else:
                sys.modules["casea_vertical_twostream_fv"] = previous
        if getattr(fv, "TWOSTREAM_FV_CORE_READY", False) is not True:
            raise ContractViolation("pinned Case-1 vertical FV readiness flag is false")
        if getattr(closures, "POST_EVENT_TWOSTREAM_CLOSURES_READY", False) is not True:
            raise ContractViolation("pinned Case-1 vertical closure readiness flag is false")
        runtime = _Case1Runtime(fv=fv, closures=closures)
        _RUNTIME_CACHE[model_dir] = runtime
        return runtime


def _component_state(
    runtime: _Case1Runtime,
    state: VerticalState,
    *,
    dry_discharge_tolerance_m3_s: float = 0.0,
):
    """Build the pinned Case-1 view without mutating conservative storage.

    The pinned FV dataclass requires an exactly dry directional label to have
    exactly zero discharge, while its numerical geometry gate admits a finite
    dry-discharge band.  Whole-network RK cancellation can leave an
    O(1e-40) discharge on an exactly zero-area label.  Canonicalise only that
    read-only Case-1 view when it lies inside the caller's pinned tolerance;
    the immutable :class:`VerticalState` and transaction ledger retain their
    original conservative value.  A finite dry-label discharge remains
    fail-closed in the pinned constructor.
    """

    tolerance = float(dry_discharge_tolerance_m3_s)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ContractViolation(
            "dry_discharge_tolerance_m3_s must be finite and non-negative"
        )

    def admissible(area: float, discharge: float) -> float:
        if area == 0.0 and abs(discharge) <= tolerance:
            return 0.0
        return discharge

    return runtime.fv.VerticalTwoStreamState.from_iterables(
        upward_area=state.Aup,
        upward_discharge=(
            admissible(area, discharge)
            for area, discharge in zip(state.Aup, state.Qup, strict=True)
        ),
        downward_area=state.Adown,
        downward_discharge=(
            admissible(area, -discharge)
            for area, discharge in zip(state.Adown, state.Qdown, strict=True)
        ),
    )


class S1VerticalTwoStreamSolver:
    """Pinned Case-1 FV/closure composition for the S1 riser only."""

    def __init__(
        self,
        *,
        cell_count: int = 160,
        closures: S1VerticalClosures | None = None,
        adapter: Case1VerticalComponentAdapter | None = None,
    ) -> None:
        self.adapter = (
            build_s1_vertical_component(cell_count=cell_count)
            if adapter is None
            else adapter
        )
        if self.adapter.cell_count != cell_count:
            raise ContractViolation("adapter and requested riser cell counts differ")
        self.closures = S1VerticalClosures() if closures is None else closures
        self._runtime = _load_pinned_runtime(self.adapter)
        self._parameters = self._runtime.fv.VerticalTwoStreamParameters(
            cell_count=cell_count,
            cell_length=self.adapter.cell_length_m,
            diameter=PIPE_DIAMETER_M,
            liquid_density=S1_LIQUID_DENSITY_KG_M3,
            gravity=S1_GRAVITY_M_S2,
            wall_friction_up=(
                0.0
                if self.closures.wall_friction_up is None
                else self.closures.wall_friction_up
            ),
            wall_friction_down=(
                0.0
                if self.closures.wall_friction_down is None
                else self.closures.wall_friction_down
            ),
            interstream_drag=(
                0.0
                if self.closures.interstream_drag is None
                else self.closures.interstream_drag
            ),
        )
        self._gas_parameters = self._runtime.closures.IsothermalGasClosureParameters(
            gas_constant=DRY_AIR_GAS_CONSTANT_J_KG_K,
            temperature=INITIAL_AIR_TEMPERATURE_K,
            atmospheric_pressure=ATMOSPHERIC_PRESSURE_PA,
            gas_viscosity=S1_GAS_VISCOSITY_PA_S,
        )

    @property
    def initial_state(self) -> VerticalState:
        return self.adapter.initial.own_state

    @property
    def cell_count(self) -> int:
        return self.adapter.cell_count

    @property
    def cell_length_m(self) -> float:
        return self.adapter.cell_length_m

    @property
    def pipe_area_m2(self) -> float:
        return self._parameters.full_area

    @property
    def production_ready(self) -> bool:
        return False

    def hydrostatic_filled_cell_pressures(
        self, *, bottom_pressure_pa: float
    ) -> tuple[float, ...]:
        """Cell-centre pressure supplied by a static liquid pressure solve."""

        bottom = float(bottom_pressure_pa)
        if not math.isfinite(bottom) or bottom <= 0.0:
            raise ContractViolation("bottom pressure must be finite and positive")
        dz = self.cell_length_m
        result = tuple(
            bottom
            - S1_LIQUID_DENSITY_KG_M3
            * S1_GRAVITY_M_S2
            * (cell + 0.5)
            * dz
            for cell in range(self.cell_count)
        )
        if min(result) <= 0.0:
            raise ContractViolation("hydrostatic cell pressure became non-positive")
        return result

    def source_initial_pressure_packet(self) -> RiserNodeFluxPacket:
        """Declared static pressure seed for the published z=0.5842 m state."""

        bottom = (
            ATMOSPHERIC_PRESSURE_PA
            + S1_LIQUID_DENSITY_KG_M3
            * S1_GRAVITY_M_S2
            * self.adapter.initial_water_level_m
        )
        return RiserNodeFluxPacket(
            bottom_pressure_pa=bottom,
            liquid_filled_cell_pressure_pa=self.hydrostatic_filled_cell_pressures(
                bottom_pressure_pa=bottom
            ),
        )

    def _validate_state(self, state: VerticalState) -> None:
        if state.cell_count != self.cell_count:
            raise ContractViolation("vertical state and solver grid differ")
        tol = self._parameters.packing_tolerance
        void_tol = self._gas_parameters.void_area_tolerance
        inventory_tol_per_length = (
            self._gas_parameters.gas_inventory_tolerance / self.cell_length_m
        )
        for cell, (up, down, mass, momentum) in enumerate(
            zip(state.Aup, state.Adown, state.Mg, state.Jg, strict=True)
        ):
            void = self.pipe_area_m2 - up - down
            if void < -tol:
                raise ContractViolation(f"liquid over-packs riser cell {cell}")
            if void <= void_tol and (
                mass > inventory_tol_per_length
                or abs(momentum) > inventory_tol_per_length
            ):
                raise MissingPhysicalClosure(
                    f"gas inventory occupies a zero-void riser cell {cell}"
                )
            if void > void_tol and mass <= inventory_tol_per_length:
                raise MissingPhysicalClosure(
                    "a finite gas void has no mass; an atomic gas/void Riemann "
                    f"closure is required in riser cell {cell}"
                )

    def _filled_pressure(
        self, state: VerticalState, packet: RiserNodeFluxPacket
    ) -> tuple[float, ...] | None:
        filled = packet.liquid_filled_cell_pressure_pa
        if filled is not None and len(filled) != self.cell_count:
            raise ContractViolation(
                "liquid-filled cell pressure and riser grid lengths differ"
            )
        if filled is None:
            for up, down in zip(state.Aup, state.Adown, strict=True):
                if self.pipe_area_m2 - up - down <= self._gas_parameters.void_area_tolerance:
                    raise MissingPhysicalClosure(
                        "liquid-filled riser cells require pressure from the coupled "
                        "liquid pressure solve"
                    )
        return filled

    def _gas_transport(
        self,
        *,
        component_state,
        gas_mass_cell_kg: tuple[float, ...],
        gas_momentum_cell_kg_m_s: tuple[float, ...],
        common_pressure_faces_pa: tuple[float, ...],
        packet: RiserNodeFluxPacket,
        dt_s: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], GasTransportLedger, float]:
        """Conservative donor gas transport plus common-pressure/body sources."""

        n = self.cell_count
        dz = self.cell_length_m
        raw_gas_area = tuple(
            self.pipe_area_m2 - up - down
            for up, down in zip(
                component_state.upward_area,
                component_state.downward_area,
                strict=True,
            )
        )
        void_tol = self._gas_parameters.void_area_tolerance
        gas_area = tuple(
            0.0 if value <= void_tol else value for value in raw_gas_area
        )
        mass = [value / dz for value in gas_mass_cell_kg]
        momentum = [value / dz for value in gas_momentum_cell_kg_m_s]
        velocity = [
            0.0 if value <= 0.0 else momentum[cell] / value
            for cell, value in enumerate(mass)
        ]

        upward = [0.0] * (n + 1)
        downward = [0.0] * (n + 1)  # non-negative downward magnitudes
        upward_speed = [0.0] * (n + 1)
        downward_speed = [0.0] * (n + 1)
        if packet.gas_upward_mass_rate_kg_s > 0.0 and gas_area[0] <= void_tol:
            raise MissingPhysicalClosure(
                "the riser node requested bottom gas inflow before creating a "
                "finite receiving void"
            )
        upward[0] = packet.gas_upward_mass_rate_kg_s
        upward_speed[0] = packet.gas_upward_speed_m_s
        downward[0] = packet.gas_downward_mass_rate_kg_s
        downward_speed[0] = packet.gas_downward_speed_m_s
        for face in range(1, n):
            lower = face - 1
            upper = face
            if momentum[lower] > 0.0 and gas_area[upper] > void_tol:
                upward[face] = momentum[lower]
                upward_speed[face] = velocity[lower]
            if momentum[upper] < 0.0 and gas_area[lower] > void_tol:
                downward[face] = -momentum[upper]
                downward_speed[face] = -velocity[upper]
        if momentum[-1] > 0.0:
            upward[n] = momentum[-1]
            upward_speed[n] = velocity[-1]
        # The atmospheric top is donor-outflow-only for this component.  A
        # downward atmospheric-gas Riemann state belongs to the global solve.
        downward[n] = 0.0

        requested_bottom_downward = downward[0]
        for cell in range(n):
            outgoing = upward[cell + 1] + downward[cell]
            if outgoing <= 0.0:
                continue
            available_rate = mass[cell] * dz / dt_s
            factor = min(1.0, available_rate / outgoing)
            upward[cell + 1] *= factor
            downward[cell] *= factor
        if not math.isclose(
            downward[0], requested_bottom_downward, rel_tol=1.0e-11, abs_tol=1.0e-16
        ):
            raise ContractViolation(
                "riser gas donor limiter changed the explicit bottom-node outflow; "
                "the node packet must be recomputed atomically"
            )

        mass_flux = [up - down for up, down in zip(upward, downward, strict=True)]
        momentum_flux = [
            up * u_up + down * u_down
            for up, u_up, down, u_down in zip(
                upward,
                upward_speed,
                downward,
                downward_speed,
                strict=True,
            )
        ]
        final_mass: list[float] = []
        final_momentum: list[float] = []
        pressure_change: list[float] = []
        gravity_change: list[float] = []
        for cell in range(n):
            m_new = mass[cell] + dt_s / dz * (
                mass_flux[cell] - mass_flux[cell + 1]
            )
            if m_new < -1.0e-14:
                raise ContractViolation("gas donor transport produced negative mass")
            m_new = max(m_new, 0.0)
            pressure_delta = (
                -dt_s
                * gas_area[cell]
                * (common_pressure_faces_pa[cell + 1] - common_pressure_faces_pa[cell])
                / dz
            )
            gravity_delta = -dt_s * mass[cell] * S1_GRAVITY_M_S2
            j_new = (
                momentum[cell]
                + dt_s / dz * (momentum_flux[cell] - momentum_flux[cell + 1])
                + pressure_delta
                + gravity_delta
            )
            if m_new == 0.0:
                if abs(j_new) > 1.0e-13:
                    raise MissingPhysicalClosure(
                        "a zero-mass gas cell retained momentum after the split stage"
                    )
                j_new = 0.0
            final_mass.append(m_new)
            final_momentum.append(j_new)
            pressure_change.append(pressure_delta)
            gravity_change.append(gravity_delta)

        initial_mass_total = sum(mass) * dz
        final_mass_total = sum(final_mass) * dz
        bottom_mass_rate = mass_flux[0]
        top_mass_rate = mass_flux[-1]
        mass_residual = (
            final_mass_total
            - initial_mass_total
            - dt_s * (bottom_mass_rate - top_mass_rate)
        )
        initial_momentum_total = sum(momentum) * dz
        final_momentum_total = sum(final_momentum) * dz
        boundary_impulse = dt_s * (momentum_flux[0] - momentum_flux[-1])
        pressure_impulse = sum(pressure_change) * dz
        gravity_impulse = sum(gravity_change) * dz
        momentum_residual = (
            final_momentum_total
            - initial_momentum_total
            - boundary_impulse
            - pressure_impulse
            - gravity_impulse
        )
        tolerance = 2.0e-12
        if abs(mass_residual) > tolerance or abs(momentum_residual) > tolerance:
            raise ContractViolation(
                "vertical gas finite-volume ledger does not close: "
                f"mass={mass_residual:.6e}, momentum={momentum_residual:.6e}"
            )
        ledger = GasTransportLedger(
            initial_mass_kg=initial_mass_total,
            final_mass_kg=final_mass_total,
            bottom_net_mass_rate_kg_s=bottom_mass_rate,
            top_net_mass_rate_kg_s=top_mass_rate,
            mass_residual_kg=mass_residual,
            initial_momentum_kg_m_s=initial_momentum_total,
            final_momentum_kg_m_s=final_momentum_total,
            boundary_advective_impulse_kg_m_s=boundary_impulse,
            pressure_impulse_kg_m_s=pressure_impulse,
            gravity_impulse_kg_m_s=gravity_impulse,
            momentum_residual_kg_m_s=momentum_residual,
        )
        return tuple(final_mass), tuple(final_momentum), ledger, upward[-1]

    def advance(
        self,
        state: VerticalState,
        *,
        dt_s: float,
        bottom: RiserNodeFluxPacket,
    ) -> VerticalTwoStreamAdvanceResult:
        """Advance one component stage or fail before returning a partial state."""

        self.closures.require_computational_closure()
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ContractViolation("dt_s must be finite and positive")
        self._validate_state(state)
        filled_pressure = self._filled_pressure(state, bottom)
        component = _component_state(
            self._runtime,
            state,
            dry_discharge_tolerance_m3_s=(
                self._parameters.dry_area_tolerance
            ),
        )
        liquid_bottom = self._runtime.fv.DirectionalBoundaryFlux(
            upward_rate=bottom.liquid_upward_rate_m3_s,
            upward_speed=bottom.liquid_upward_speed_m_s,
            downward_rate=bottom.liquid_downward_rate_m3_s,
            downward_speed=bottom.liquid_downward_speed_m_s,
        )
        gas_mass_cell = tuple(value * self.cell_length_m for value in state.Mg)
        gas_momentum_cell = tuple(value * self.cell_length_m for value in state.Jg)
        try:
            liquid_stage = self._runtime.closures.advance_post_event_core_film_stage(
                component,
                self._parameters,
                dt=dt,
                gas_mass=gas_mass_cell,
                gas_momentum=gas_momentum_cell,
                bottom_boundary=liquid_bottom,
                gas=self._gas_parameters,
                bottom_pressure=bottom.bottom_pressure_pa,
                liquid_filled_cell_pressure=filled_pressure,
                apply_physical_drag=self.closures.apply_physical_three_body_drag,
            )
        except self._runtime.closures.GasVoidStateError as exc:
            raise MissingPhysicalClosure(
                "liquid transport created a gas void/inventory mismatch; the "
                "atomic coupled gas/void pressure stage is required"
            ) from exc

        accepted_bottom_up = liquid_stage.transport.upward_area_flux[0]
        accepted_bottom_down = -liquid_stage.transport.downward_area_flux[0]
        for name, accepted, requested in (
            (
                "liquid_upward_rate_m3_s",
                accepted_bottom_up,
                bottom.liquid_upward_rate_m3_s,
            ),
            (
                "liquid_downward_rate_m3_s",
                accepted_bottom_down,
                bottom.liquid_downward_rate_m3_s,
            ),
        ):
            if not math.isclose(accepted, requested, rel_tol=1.0e-11, abs_tol=1.0e-16):
                raise ContractViolation(
                    f"Case-1 capacity projection changed explicit bottom {name}; "
                    "the riser-node packet must be recomputed atomically"
                )

        final_mass_per_length, final_momentum_per_length, gas_ledger, gas_top = (
            self._gas_transport(
                component_state=liquid_stage.state,
                gas_mass_cell_kg=gas_mass_cell,
                gas_momentum_cell_kg_m_s=liquid_stage.gas_momentum,
                common_pressure_faces_pa=(
                    liquid_stage.pressure_after_transport.common_pressure_faces
                ),
                packet=bottom,
                dt_s=dt,
            )
        )
        final = VerticalState(
            Aup=liquid_stage.state.upward_area,
            Qup=liquid_stage.state.upward_discharge,
            Adown=liquid_stage.state.downward_area,
            Qdown=tuple(-value for value in liquid_stage.state.downward_discharge),
            Mg=final_mass_per_length,
            Jg=final_momentum_per_length,
        )
        self._validate_state(final)

        # Validate the final gas EOS/void state and expose the pressure that the
        # next coupled stage must consume.  This does not invent missing mass.
        final_pressure = self._runtime.closures.adapt_gas_void_and_pressure_faces(
            _component_state(
                self._runtime,
                final,
                dry_discharge_tolerance_m3_s=(
                    self._parameters.dry_area_tolerance
                ),
            ),
            self._parameters,
            gas_mass=(value * self.cell_length_m for value in final.Mg),
            gas_momentum=(value * self.cell_length_m for value in final.Jg),
            gas=self._gas_parameters,
            bottom_pressure=bottom.bottom_pressure_pa,
            liquid_filled_cell_pressure=filled_pressure,
        )

        drag_residual = (
            0.0
            if liquid_stage.drag is None
            else liquid_stage.drag.total_momentum_residual
        )
        liquid_ledger = liquid_stage.transport.ledger
        rho = S1_LIQUID_DENSITY_KG_M3
        initial_mixture = (
            rho
            * self.cell_length_m
            * sum(up - down for up, down in zip(state.Qup, state.Qdown, strict=True))
            + self.cell_length_m * sum(state.Jg)
        )
        final_mixture = (
            rho
            * self.cell_length_m
            * sum(up - down for up, down in zip(final.Qup, final.Qdown, strict=True))
            + self.cell_length_m * sum(final.Jg)
        )
        liquid_external_impulse = rho * (
            liquid_ledger.boundary_momentum_impulse
            + liquid_ledger.pressure_gravity_impulse
            + liquid_ledger.wall_impulse
            + liquid_ledger.interstream_upward_impulse
            + liquid_ledger.interstream_downward_impulse
            + liquid_ledger.gas_on_liquid_kinematic_impulse
        )
        gas_external_impulse = (
            gas_ledger.boundary_advective_impulse_kg_m_s
            + gas_ledger.pressure_impulse_kg_m_s
            + gas_ledger.gravity_impulse_kg_m_s
        )
        mixture_residual = (
            final_mixture
            - initial_mixture
            - liquid_external_impulse
            - gas_external_impulse
        )
        if abs(drag_residual) > 2.0e-12 or abs(mixture_residual) > 5.0e-11:
            raise ContractViolation(
                "vertical mixture momentum ledger does not close: "
                f"three_body={drag_residual:.6e}, mixture={mixture_residual:.6e}"
            )

        top_liquid_rate = liquid_stage.transport.upward_area_flux[-1]
        return VerticalTwoStreamAdvanceResult(
            state=final,
            common_pressure_faces_pa=final_pressure.common_pressure_faces,
            gas_pressure_cells_pa=final_pressure.gas_pressure_cells,
            top_liquid_outflow_rate_m3_s=top_liquid_rate,
            top_liquid_outflow_volume_m3=dt * top_liquid_rate,
            top_gas_outflow_rate_kg_s=gas_top,
            liquid_volume_residual_m3=liquid_stage.inventory.total_volume_residual,
            gas=gas_ledger,
            three_body_momentum_residual_kg_m_s=drag_residual,
            mixture_momentum_residual_kg_m_s=mixture_residual,
            maximum_packing_residual_m2=(
                liquid_ledger.maximum_packing_residual
            ),
            validation_only=self.closures.validation_only,
            production_ready=False,
        )


__all__ = [
    "GasTransportLedger",
    "RiserNodeFluxPacket",
    "S1VerticalClosures",
    "S1VerticalTwoStreamSolver",
    "VerticalTwoStreamAdvanceResult",
]
