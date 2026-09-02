"""Read-only, fail-closed adapter for the pinned Case-1 riser components.

This module is deliberately limited to source verification and construction of
the S1 initial component state.  It does not expose a trajectory integrator.
The pinned Case-1 finite-volume core is locally ready, while its own complete
riser flag is false; the unresolved S1 closures therefore remain blockers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from .errors import ContractViolation, MissingPhysicalClosure
from .state import VerticalState


PIPE_DIAMETER_M = 0.0254
RISER_BOTTOM_Z_M = 0.0
RISER_TOP_Z_M = 1.02
INITIAL_WATER_LEVEL_Z_M = 0.5842
DEFAULT_CELL_COUNT = 160
ATMOSPHERIC_PRESSURE_PA = 101325.0
INITIAL_AIR_TEMPERATURE_K = 293.15
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05

PINNED_SHA256 = {
    "casea_vertical_twostream_fv.py":
        "262ACDA410E23ABD1DD67F6C23F7B774AACDAC71B54C708391E57790F91FB928",
    "casea_vertical_twostream_closures.py":
        "422892DAB98EAB73FB37BEF0F89C1037A5558735CD85301B5C21401D13A88710",
    "casea_bidirectional_tnode_inertance.py":
        "1AAF6AE32DB79D28FD75D10C5181FD404C2CFFDFC3D6A5A469DEABC968004AC2",
}

# These Case-1 apparatus-specific choices are evidence, not S1 inputs.
FORBIDDEN_CASE1_TRANSPLANTS = (
    "fixed Taylor-core area fraction (Case-1 alpha_core=0.80)",
    "Case-1 Taylor rise-speed floor 0.345*sqrt(g*D)",
    "Case-1 Wallis constants or flooding parameterization",
)

S1_MISSING_CLOSURES = (
    "S1 post-event core/film area evolution",
    "S1 liquid/liquid interfacial drag",
    "S1 gas-void pressure evolution at every coupled stage",
    "continuous-air T-junction net/circulation Riemann closure",
    "S1 top vent/free-surface liquid-exit boundary",
    "atomic horizontal/T-node/vertical stage coupling",
)


class Case1PinMismatch(ContractViolation):
    """A read-only Case-1 dependency no longer matches its reviewed hash."""


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tests" / "test_01_vw2011").is_dir():
            return candidate
    raise Case1PinMismatch("cannot locate the Geysering repository root")


def default_case1_model_dir() -> Path:
    return (
        _repository_root()
        / "tests"
        / "test_01_vw2011"
        / "cases"
        / "A_Dt57p1_Ha0305_Yfs0356"
        / "model"
    )


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


@dataclass(frozen=True, slots=True)
class Case1VerticalPinContract:
    model_dir: Path
    actual_sha256: tuple[tuple[str, str], ...]
    fv_core_ready: bool
    post_event_closures_ready: bool
    complete_riser_ready: bool
    case1_missing_physical_closures: tuple[str, ...]

    @property
    def production_ready(self) -> bool:
        return False


def verify_case1_vertical_pins(
    model_dir: Path | str | None = None,
) -> Case1VerticalPinContract:
    """Verify reviewed bytes and readiness literals without executing source."""

    source_dir = Path(model_dir) if model_dir is not None else default_case1_model_dir()
    actual: list[tuple[str, str]] = []
    for name, expected in PINNED_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise Case1PinMismatch(f"missing pinned Case-1 source: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if digest != expected:
            raise Case1PinMismatch(
                f"Case-1 source hash mismatch for {name}: {digest} != {expected}"
            )
        actual.append((name, digest))

    fv_literals = _literal_assignments(source_dir / "casea_vertical_twostream_fv.py")
    closure_literals = _literal_assignments(
        source_dir / "casea_vertical_twostream_closures.py"
    )
    core_ready = fv_literals.get("TWOSTREAM_FV_CORE_READY") is True
    closures_ready = closure_literals.get("POST_EVENT_TWOSTREAM_CLOSURES_READY") is True
    complete_ready = fv_literals.get("COMPLETE_CASEA_RISER_READY") is True
    if not core_ready or not closures_ready:
        raise Case1PinMismatch("pinned Case-1 local component readiness flag is false")
    if complete_ready:
        raise Case1PinMismatch(
            "reviewed contract expected COMPLETE_CASEA_RISER_READY=False"
        )
    missing = fv_literals.get("MISSING_PHYSICAL_CLOSURES")
    if not isinstance(missing, tuple) or not all(isinstance(item, str) for item in missing):
        raise Case1PinMismatch("Case-1 missing-closure declaration is absent or malformed")
    return Case1VerticalPinContract(
        model_dir=source_dir.resolve(),
        actual_sha256=tuple(actual),
        fv_core_ready=core_ready,
        post_event_closures_ready=closures_ready,
        complete_riser_ready=False,
        case1_missing_physical_closures=missing,
    )


def _load_pinned_fv_state_type(contract: Case1VerticalPinContract) -> type:
    """Load only the immutable state type after the byte/flag gate passes."""

    name = "_geysering_pinned_case1_vertical_twostream_fv_262acda4"
    module: ModuleType | None = sys.modules.get(name)
    if module is None:
        path = contract.model_dir / "casea_vertical_twostream_fv.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise Case1PinMismatch(f"cannot load pinned component state from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    if (
        getattr(module, "TWOSTREAM_FV_CORE_READY", False) is not True
        or getattr(module, "COMPLETE_CASEA_RISER_READY", True) is not False
    ):
        raise Case1PinMismatch("runtime Case-1 readiness flags violate the pin contract")
    return module.VerticalTwoStreamState


@dataclass(frozen=True, slots=True)
class S1VerticalInitialState:
    z_edges_m: tuple[float, ...]
    liquid_fraction: tuple[float, ...]
    gas_area_m2: tuple[float, ...]
    component_state: Any
    own_state: VerticalState
    air_density_kg_m3: float
    target_water_volume_m3: float
    represented_water_volume_m3: float

    @property
    def water_volume_error_m3(self) -> float:
        return self.represented_water_volume_m3 - self.target_water_volume_m3

    @property
    def two_stream_state(self) -> Any:
        """Compatibility alias for the signed Case-1 component state."""

        return self.component_state

    @property
    def represented_gas_mass_kg(self) -> float:
        dz = self.z_edges_m[1] - self.z_edges_m[0]
        return sum(mass_per_length * dz for mass_per_length in self.own_state.Mg)


@dataclass(frozen=True, slots=True)
class Case1VerticalComponentAdapter:
    pin: Case1VerticalPinContract
    diameter_m: float
    z_bottom_m: float
    z_top_m: float
    initial_water_level_m: float
    cell_count: int
    cell_length_m: float
    initial: S1VerticalInitialState
    missing_physical_closures: tuple[str, ...]
    forbidden_transplants: tuple[str, ...] = FORBIDDEN_CASE1_TRANSPLANTS

    @property
    def production_ready(self) -> bool:
        return False

    def require_production_trajectory(self) -> None:
        """Always fail closed until the declared physical closures are supplied."""

        blockers = "; ".join(self.missing_physical_closures)
        raise MissingPhysicalClosure(
            "Case1 vertical components are component-ready only; "
            f"production trajectory is forbidden. Missing: {blockers}"
        )

    def component_to_own_state(
        self,
        component_state: Any,
        *,
        gas_mass_per_length_kg_m: tuple[float, ...],
        gas_momentum_per_length_kg_s: tuple[float, ...] | None = None,
    ) -> VerticalState:
        """Translate signs without reconstructing either stream from net flow.

        The pinned Case-1 component stores ``downward_discharge <= 0`` in the
        upward-positive coordinate.  The S1-owned :class:`VerticalState`
        stores ``Qdown >= 0`` as a gross downward *magnitude*.  Therefore the
        only admissible bridge is ``Qdown = -downward_discharge`` while Qup is
        copied independently.
        """

        if getattr(component_state, "cell_count", None) != self.cell_count:
            raise ContractViolation("component state and S1 vertical grid differ")
        gas_mass = tuple(float(value) for value in gas_mass_per_length_kg_m)
        gas_momentum = (
            (0.0,) * self.cell_count
            if gas_momentum_per_length_kg_s is None
            else tuple(float(value) for value in gas_momentum_per_length_kg_s)
        )
        if len(gas_mass) != self.cell_count or len(gas_momentum) != self.cell_count:
            raise ContractViolation("gas state and S1 vertical grid differ")
        return _component_to_own_vertical_state(
            component_state,
            gas_mass_per_length_kg_m=gas_mass,
            gas_momentum_per_length_kg_s=gas_momentum,
        )


def _component_to_own_vertical_state(
    component_state: Any,
    *,
    gas_mass_per_length_kg_m: tuple[float, ...],
    gas_momentum_per_length_kg_s: tuple[float, ...],
) -> VerticalState:
    """Internal sign-explicit bridge used for initialization and later adapters."""

    downward_signed = tuple(float(value) for value in component_state.downward_discharge)
    if any(value > 0.0 for value in downward_signed):
        raise ContractViolation("Case-1 component downward discharge must be <= 0")
    return VerticalState(
        Aup=component_state.upward_area,
        Qup=component_state.upward_discharge,
        Adown=component_state.downward_area,
        Qdown=tuple(-value for value in downward_signed),
        Mg=gas_mass_per_length_kg_m,
        Jg=gas_momentum_per_length_kg_s,
    )


def build_s1_vertical_component(
    *,
    cell_count: int = DEFAULT_CELL_COUNT,
    case1_model_dir: Path | str | None = None,
) -> Case1VerticalComponentAdapter:
    """Construct the source-aligned S1 grid and persistent two-stream initial state.

    The cut cell at ``z=0.5842 m`` stores its exact cell-average water area.
    Below it, ``A_up=A_pipe``; above it, the section is gas.  ``A_down`` and
    both directional discharges start at zero.  No signed-net reconstruction,
    Taylor-core split, rise-speed floor, or Wallis closure is applied.
    """

    if not isinstance(cell_count, int) or cell_count <= 0:
        raise ContractViolation("cell_count must be a positive integer")
    pin = verify_case1_vertical_pins(case1_model_dir)
    state_type = _load_pinned_fv_state_type(pin)
    dz = (RISER_TOP_Z_M - RISER_BOTTOM_Z_M) / cell_count
    edges = tuple(RISER_BOTTOM_Z_M + index * dz for index in range(cell_count + 1))
    pipe_area = math.pi * PIPE_DIAMETER_M**2 / 4.0
    fractions = tuple(
        min(max((INITIAL_WATER_LEVEL_Z_M - edges[i]) / dz, 0.0), 1.0)
        for i in range(cell_count)
    )
    upward_area = tuple(pipe_area * fraction for fraction in fractions)
    zero = (0.0,) * cell_count
    state = state_type.from_iterables(
        upward_area=upward_area,
        upward_discharge=zero,
        downward_area=zero,
        downward_discharge=zero,
    )
    represented = sum(area * dz for area in state.liquid_area)
    target = pipe_area * (INITIAL_WATER_LEVEL_Z_M - RISER_BOTTOM_Z_M)
    gas_area = tuple(pipe_area - area for area in state.liquid_area)
    air_density = (
        ATMOSPHERIC_PRESSURE_PA
        / (DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K)
    )
    gas_mass_per_length = tuple(air_density * area for area in gas_area)
    gas_momentum_per_length = (0.0,) * cell_count
    own_state = _component_to_own_vertical_state(
        state,
        gas_mass_per_length_kg_m=gas_mass_per_length,
        gas_momentum_per_length_kg_s=gas_momentum_per_length,
    )
    initial = S1VerticalInitialState(
        z_edges_m=edges,
        liquid_fraction=fractions,
        gas_area_m2=gas_area,
        component_state=state,
        own_state=own_state,
        air_density_kg_m3=air_density,
        target_water_volume_m3=target,
        represented_water_volume_m3=represented,
    )
    missing = tuple(dict.fromkeys(pin.case1_missing_physical_closures + S1_MISSING_CLOSURES))
    return Case1VerticalComponentAdapter(
        pin=pin,
        diameter_m=PIPE_DIAMETER_M,
        z_bottom_m=RISER_BOTTOM_Z_M,
        z_top_m=RISER_TOP_Z_M,
        initial_water_level_m=INITIAL_WATER_LEVEL_Z_M,
        cell_count=cell_count,
        cell_length_m=dz,
        initial=initial,
        missing_physical_closures=missing,
    )


__all__ = [
    "ATMOSPHERIC_PRESSURE_PA",
    "Case1PinMismatch",
    "Case1VerticalComponentAdapter",
    "Case1VerticalPinContract",
    "DRY_AIR_GAS_CONSTANT_J_KG_K",
    "FORBIDDEN_CASE1_TRANSPLANTS",
    "INITIAL_AIR_TEMPERATURE_K",
    "PINNED_SHA256",
    "S1VerticalInitialState",
    "build_s1_vertical_component",
    "verify_case1_vertical_pins",
]
