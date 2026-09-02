"""Fail-closed Case-1 liquid-operator adapter for the Mahyawansi S1 case.

The circular-pipe liquid geometry, conservative physical flux, MUSCL face
reconstruction, central-upwind Riemann flux and donor draining limiter are
loaded from one hash-pinned Case-1 source.  The S1 network may call those
spatial building blocks on each main-pipe segment separated by a physical T
node.  Case-1's finite gas pocket, valve-release event and fitted moving
interface remain a different topology and are deliberately unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
from typing import Literal, Mapping, Sequence

import numpy as np

from .errors import ContractViolation
from .state import HorizontalState


CASE1_SEED_SHA256 = MappingProxyType(
    {
        "tosan2021_horizontal_shockfit.py": (
            "90E84DA9AFA0EC8465D80F87FC701DFB8F0FAD6F97350EA708074A50192B6119"
        ),
        "casea_shockfit_network.py": (
            "1B24C90C3DC997F0E17CFBEF9A720DB92510C14B86E42DF3902EA2AE8E6061B3"
        ),
    }
)

CONTINUOUS_AIR_TOPOLOGY = "mahywansi_continuous_side_air"
DEFAULT_HORIZONTAL_DX_M = 0.01
DEFAULT_GRAVITY_M_S2 = 9.81
DEFAULT_CASE1_WAVE_SPEED_M_S = 100.0
FROZEN_2D_WATER_PERFECT_FLUID_R_J_KG_K = 3000.0
FROZEN_2D_TEMPERATURE_K = 293.15
FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S = math.sqrt(
    FROZEN_2D_WATER_PERFECT_FLUID_R_J_KG_K * FROZEN_2D_TEMPERATURE_K
)
FROZEN_2D_WATER_EOS_EVIDENCE = (
    "declared_OpenFOAM_v2512_perfectFluid_isothermal_tangent__"
    "c_squared_equals_R_times_T"
)


class Case1SourceIntegrityError(ContractViolation):
    """A frozen Case-1 source is absent or differs from its accepted hash."""


class ForbiddenCase1Topology(ContractViolation):
    """A finite-pocket/valve-release Case-1 topology was requested for S1."""


@dataclass(frozen=True, slots=True)
class NumericalParameterProvenance:
    """The effective value and evidence class of a numerical parameter."""

    value: float
    evidence: str
    reference_value: float
    is_override: bool


@dataclass(frozen=True, slots=True)
class Case1FaceFluxes:
    """One segment's exact Case-1 MUSCL/central-upwind face fluxes."""

    liquid_volume_m3_s: tuple[float, ...]
    liquid_momentum_m4_s2: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TotalPressureGhostState:
    """Characteristic boundary state for a published total-pressure head."""

    side: Literal["left", "right"]
    liquid_area_m2: float
    liquid_discharge_m3_s: float
    piezometric_head_m: float
    total_head_m: float
    outgoing_characteristic_m_s: float
    total_head_residual_m: float
    evidence_status: str


@dataclass(frozen=True, slots=True)
class StaticPressureGhostState:
    """Characteristic ghost for a published static piezometric head."""

    side: Literal["left", "right"]
    liquid_area_m2: float
    liquid_discharge_m3_s: float
    piezometric_head_m: float
    prescribed_static_head_m: float
    outgoing_characteristic_m_s: float
    static_head_residual_m: float
    evidence_status: str


def _parameter_record(
    value: float, *, evidence: str, reference_value: float
) -> NumericalParameterProvenance:
    return NumericalParameterProvenance(
        value=float(value),
        evidence=evidence,
        reference_value=float(reference_value),
        is_override=not math.isclose(
            float(value), float(reference_value), rel_tol=0.0, abs_tol=1.0e-15
        ),
    )


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "tests" / "test_01_vw2011").is_dir():
            return parent
    raise Case1SourceIntegrityError("cannot locate the Geysering repository root")


CASE1_SEED_DIRECTORY = (
    _repository_root()
    / "tests"
    / "test_01_vw2011"
    / "cases"
    / "A_Dt57p1_Ha0305_Yfs0356"
    / "model"
)
CASE1_SEED_PATHS = MappingProxyType(
    {name: CASE1_SEED_DIRECTORY / name for name in CASE1_SEED_SHA256}
)


def _sha256_read_only(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Case1SourceIntegrityError(f"cannot read frozen Case-1 seed: {path}") from exc
    return digest.hexdigest().upper()


def verify_case1_seed_integrity(
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    """Read and verify both accepted Case-1 seeds without modifying them."""

    paths = CASE1_SEED_PATHS if source_paths is None else source_paths
    if set(paths) != set(CASE1_SEED_SHA256):
        raise Case1SourceIntegrityError("Case-1 seed set is incomplete or unexpected")
    observed: dict[str, str] = {}
    for name, expected in CASE1_SEED_SHA256.items():
        path = Path(paths[name])
        actual = _sha256_read_only(path)
        if actual != expected:
            raise Case1SourceIntegrityError(
                f"frozen Case-1 seed hash mismatch for {name}: {actual} != {expected}"
            )
        observed[name] = actual
    return observed


_CASE1_MODULE_NAME = "_geysering_frozen_case1_tosan_90e84da9"


def _load_verified_case1_liquid_seed() -> ModuleType:
    verify_case1_seed_integrity()
    existing = sys.modules.get(_CASE1_MODULE_NAME)
    if existing is not None:
        return existing
    source = CASE1_SEED_PATHS["tosan2021_horizontal_shockfit.py"]
    spec = importlib.util.spec_from_file_location(_CASE1_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise Case1SourceIntegrityError("cannot construct the Case-1 seed loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CASE1_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_CASE1_MODULE_NAME, None)
        raise
    return module


@dataclass(frozen=True, slots=True)
class MahyawansiHorizontalGrid:
    """Source-aligned S1 geometry with both side tees frozen on cell faces.

    ``dx_m=0.01`` is a declared 1-D numerical choice, not a published value.
    It places the air and riser tees exactly at face indices 31 and 183.
    """

    diameter_m: float = 0.0254
    x_left_m: float = -1.83
    air_tee_x_m: float = -1.52
    riser_tee_x_m: float = 0.0
    x_right_m: float = 1.27
    dx_m: float = DEFAULT_HORIZONTAL_DX_M

    def __post_init__(self) -> None:
        frozen = {
            "diameter_m": 0.0254,
            "x_left_m": -1.83,
            "air_tee_x_m": -1.52,
            "riser_tee_x_m": 0.0,
            "x_right_m": 1.27,
        }
        for name, expected in frozen.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not math.isclose(
                value, expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ContractViolation(
                    f"S1 source geometry drifted: {name}={value}, expected {expected}"
                )
            object.__setattr__(self, name, value)
        dx = float(self.dx_m)
        if not math.isfinite(dx) or dx <= 0.0:
            raise ContractViolation("horizontal dx_m must be finite and positive")
        object.__setattr__(self, "dx_m", dx)
        # Evaluate all three required face locations through the same integer map.
        for label, coordinate in (
            ("downstream end", self.x_right_m),
            ("air tee", self.air_tee_x_m),
            ("riser tee", self.riser_tee_x_m),
        ):
            raw = (coordinate - self.x_left_m) / dx
            if not math.isclose(raw, round(raw), rel_tol=0.0, abs_tol=1.0e-10):
                raise ContractViolation(f"{label} is not located on a 1-D cell face")

    @property
    def cell_count(self) -> int:
        return round((self.x_right_m - self.x_left_m) / self.dx_m)

    def face_index(self, x_m: float) -> int:
        raw = (float(x_m) - self.x_left_m) / self.dx_m
        index = round(raw)
        if not math.isclose(raw, index, rel_tol=0.0, abs_tol=1.0e-10):
            raise ContractViolation("requested coordinate is not a frozen grid face")
        if not 0 <= index <= self.cell_count:
            raise ContractViolation("requested coordinate lies outside the horizontal grid")
        return index

    @property
    def air_tee_face_index(self) -> int:
        return self.face_index(self.air_tee_x_m)

    @property
    def riser_tee_face_index(self) -> int:
        return self.face_index(self.riser_tee_x_m)

    @property
    def cell_lengths_m(self) -> tuple[float, ...]:
        return (self.dx_m,) * self.cell_count

    @property
    def dx_provenance(self) -> NumericalParameterProvenance:
        """Record the declared 1-D grid choice, including any explicit override."""

        return _parameter_record(
            self.dx_m,
            evidence="declared_1D_grid",
            reference_value=DEFAULT_HORIZONTAL_DX_M,
        )


def validate_mahywansi_initialization(
    *,
    topology: str = CONTINUOUS_AIR_TOPOLOGY,
    finite_gas_pocket_length_m: float | None = None,
    valve_release_shockfit: bool = False,
) -> None:
    """Fail closed if a Case-1 gas-pocket or valve-release state is requested."""

    if topology != CONTINUOUS_AIR_TOPOLOGY:
        raise ForbiddenCase1Topology(
            "S1 permits only continuous side-air supply; Case-1 shock-fit topology is forbidden"
        )
    if finite_gas_pocket_length_m is not None:
        raise ForbiddenCase1Topology("S1 initial horizontal pipe contains no finite gas pocket")
    if valve_release_shockfit:
        raise ForbiddenCase1Topology("S1 has no Case-1 valve-release shock-fit event")


@dataclass(frozen=True, slots=True)
class LiquidPhysicalFlux:
    """Case-1 flux ``(Q, Q^2/A + pressure_flux(A))`` per liquid density."""

    liquid_volume_m3_s: float
    liquid_momentum_m4_s2: float


class Case1HorizontalLiquidAdapter:
    """Verified circular-section liquid geometry; no Case-1 gas topology."""

    def __init__(
        self,
        grid: MahyawansiHorizontalGrid | None = None,
        *,
        gravity_m_s2: float = DEFAULT_GRAVITY_M_S2,
        wave_speed_m_s: float = DEFAULT_CASE1_WAVE_SPEED_M_S,
        wave_speed_evidence: str = "Case1_inherited_numerical_parameter",
    ) -> None:
        self.grid = MahyawansiHorizontalGrid() if grid is None else grid
        gravity = float(gravity_m_s2)
        wave_speed = float(wave_speed_m_s)
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ContractViolation("gravity_m_s2 must be finite and positive")
        if not math.isfinite(wave_speed) or wave_speed <= 0.0:
            raise ContractViolation("wave_speed_m_s must be finite and positive")
        if not wave_speed_evidence.strip():
            raise ContractViolation("wave_speed_evidence must be non-empty")
        self._gravity_m_s2 = gravity
        self._wave_speed_m_s = wave_speed
        self._wave_speed_evidence = wave_speed_evidence
        seed = _load_verified_case1_liquid_seed()
        self._case1_seed = seed
        self._section = seed.CircularSection(
            self.grid.diameter_m,
            gravity=gravity,
            wave_speed=wave_speed,
        )

    @property
    def full_area_m2(self) -> float:
        return float(self._section.full_area)

    @property
    def gravity_m_s2(self) -> float:
        return self._gravity_m_s2

    @property
    def wave_speed_m_s(self) -> float:
        return self._wave_speed_m_s

    @property
    def parameter_provenance(self) -> Mapping[str, NumericalParameterProvenance]:
        """Expose every inherited/declared scalar; overrides cannot be silent."""

        return MappingProxyType(
            {
                "dx_m": self.grid.dx_provenance,
                "gravity_m_s2": _parameter_record(
                    self.gravity_m_s2,
                    evidence="Case1_inherited_physical_constant",
                    reference_value=DEFAULT_GRAVITY_M_S2,
                ),
                "wave_speed_m_s": _parameter_record(
                    self.wave_speed_m_s,
                    evidence=self._wave_speed_evidence,
                    reference_value=DEFAULT_CASE1_WAVE_SPEED_M_S,
                ),
            }
        )

    def physical_flux(self, liquid_area_m2: float, liquid_discharge_m3_s: float) -> LiquidPhysicalFlux:
        area = float(liquid_area_m2)
        discharge = float(liquid_discharge_m3_s)
        if not (math.isfinite(area) and math.isfinite(discharge)) or area < 0.0:
            raise ContractViolation("horizontal liquid state must be finite with non-negative area")
        if area == 0.0 and discharge != 0.0:
            raise ContractViolation("a dry horizontal state cannot carry liquid discharge")
        advective = 0.0 if area == 0.0 else discharge * discharge / area
        return LiquidPhysicalFlux(
            liquid_volume_m3_s=discharge,
            liquid_momentum_m4_s2=advective + float(self._section.pressure_flux(area)),
        )

    def conservative_port_pressure_increment_Pa(
        self,
        liquid_area_m2: float,
        liquid_density_kg_m3: float,
    ) -> float:
        """Map Case-1 elastic storage to an equivalent rigid-port pressure.

        A T-port has the physical aperture ``Af`` even when the Case-1 elastic
        storage variable is slightly larger than ``Af``.  The conservative
        Case-1 pressure force per unit density is ``pressure_flux(A)``.
        Therefore the *incremental* force above the exactly-full state is

        ``rho * (pressure_flux(A) - pressure_flux(Af))``.

        Dividing that force by the physical port area gives the unique uniform
        pressure increment that preserves the Case-1 conservative traction at
        the T face.  It is zero at ``A=Af`` and, on the elastic branch, equals

        ``0.5*rho*a^2*((A/Af)^2 - 1)``.

        The atmospheric/Table-1 datum is intentionally not part of this
        method; the S1 component adds that independently so a numerical area
        law cannot silently change a published total-pressure reference.
        Free-surface states do not use this mapping because their interface
        pressure comes from the resolved gas EOS/capillary trace.
        """

        area = float(liquid_area_m2)
        density = float(liquid_density_kg_m3)
        if not math.isfinite(area) or area < 0.0:
            raise ContractViolation("liquid area must be finite and non-negative")
        if not math.isfinite(density) or density <= 0.0:
            raise ContractViolation("liquid density must be finite and positive")
        if area < self.full_area_m2:
            raise ContractViolation(
                "conservative rigid-port pressure mapping requires a full/elastic state"
            )
        pressure_flux = self.physical_flux(area, 0.0).liquid_momentum_m4_s2
        full_pressure_flux = self.physical_flux(
            self.full_area_m2, 0.0
        ).liquid_momentum_m4_s2
        increment = density * (pressure_flux - full_pressure_flux) / self.full_area_m2
        if not math.isfinite(increment) or increment < -1.0e-10:
            raise ContractViolation("Case-1 port pressure increment is invalid")
        return max(float(increment), 0.0)

    def celerity_m_s(self, liquid_area_m2: float) -> float:
        """Return the frozen Case-1 circular/elastic liquid wave speed."""

        area = float(liquid_area_m2)
        if not math.isfinite(area) or area < 0.0:
            raise ContractViolation("liquid area must be finite and non-negative")
        return float(self._section.celerity(area))

    def _characteristic_potential_m_s(self, liquid_area_m2: float) -> float:
        """Return ``integral_Af^A c(a)/a da`` for the pinned Case-1 law.

        On the elastic branch this is analytic.  The circular free-surface
        branch is integrated in ``log(A)`` so the dry endpoint does not create
        an artificial ``1/A`` quadrature singularity.  S1 total-pressure
        boundaries remain close to ``Af``; the wider implementation is kept
        solely to make the characteristic contract explicit and testable.
        """

        area = float(liquid_area_m2)
        if not math.isfinite(area) or area <= 0.0:
            raise ContractViolation(
                "total-pressure characteristic requires positive liquid area"
            )
        full = self.full_area_m2
        if area >= full:
            return 2.0 * self.wave_speed_m_s * (
                math.sqrt(area / full) - 1.0
            )
        lower = max(area, 1.0e-14 * full)
        log_lo = math.log(lower)
        log_hi = math.log(full)
        nodes, weights = np.polynomial.legendre.leggauss(48)
        midpoint = 0.5 * (log_lo + log_hi)
        half_width = 0.5 * (log_hi - log_lo)
        sample_area = np.exp(midpoint + half_width * nodes)
        integral = half_width * float(
            np.dot(weights, np.asarray(self._section.celerity(sample_area)))
        )
        return -integral

    def _piezometric_head_from_storage_m(
        self,
        liquid_area_m2: float,
        *,
        reference_head_m: float,
    ) -> float:
        """Map Case-1 storage to the S1 absolute head datum.

        ``A=Af`` is the source-aligned Stage-1 initial water head, not zero
        pressure.  Case-1 supplies only the dynamic/free-surface increment
        about that state.  This prevents a Table-1 boundary head from being
        mistaken for the initial pressure of every horizontal cell.
        """

        reference = float(reference_head_m)
        if not math.isfinite(reference):
            raise ContractViolation("horizontal reference head must be finite")
        return reference + self.head_from_area_m(liquid_area_m2) - self.grid.diameter_m

    def dynamic_total_pressure_ghost(
        self,
        *,
        interior_area_m2: float,
        interior_discharge_m3_s: float,
        prescribed_total_head_m: float,
        reference_head_m: float,
        side: Literal["left", "right"],
        maximum_elastic_overarea_fraction: float = 0.02,
    ) -> TotalPressureGhostState:
        """Solve a subcritical characteristic/total-head boundary state.

        The outgoing Case-1 Riemann invariant is retained from the adjacent
        water cell.  The incoming invariant is supplied by the published
        Table-1 total head ``H + u^2/(2g)``.  The returned state is used as the
        MUSCL ghost, so a changing interior velocity changes the boundary
        state; this is not a fixed reservoir cell or fixed static pressure.
        """

        if side not in ("left", "right"):
            raise ContractViolation("total-pressure boundary side must be left or right")
        area_i = float(interior_area_m2)
        discharge_i = float(interior_discharge_m3_s)
        total_head = float(prescribed_total_head_m)
        maximum = float(maximum_elastic_overarea_fraction)
        if (
            not all(math.isfinite(value) for value in (area_i, discharge_i, total_head, maximum))
            or area_i <= 0.0
            or maximum <= 0.0
        ):
            raise ContractViolation("dynamic total-pressure inputs are not admissible")
        velocity_i = discharge_i / area_i
        phi_i = self._characteristic_potential_m_s(area_i)
        outgoing = velocity_i - phi_i if side == "left" else velocity_i + phi_i
        interior_piezometric = self._piezometric_head_from_storage_m(
            area_i, reference_head_m=reference_head_m
        )
        interior_total = (
            interior_piezometric
            + velocity_i * velocity_i / (2.0 * self.gravity_m_s2)
        )
        # Preserve an exact discrete equilibrium.  The area/head inverse uses
        # finite-precision arithmetic, so the source 0.5842 m state can differ
        # from its reconstructed head by O(1e-11 m).  Re-solving that identical
        # physical state would inject a roundoff-scale boundary wave.
        if math.isclose(
            interior_total,
            total_head,
            rel_tol=0.0,
            abs_tol=2.0e-10,
        ):
            return TotalPressureGhostState(
                side=side,
                liquid_area_m2=area_i,
                liquid_discharge_m3_s=discharge_i,
                piezometric_head_m=float(interior_piezometric),
                total_head_m=float(total_head),
                outgoing_characteristic_m_s=float(outgoing),
                total_head_residual_m=float(interior_total - total_head),
                evidence_status=(
                    "published_Table1_total_pressure__Case1_outgoing_characteristic__"
                    "dynamic_MUSCL_ghost_declared_1D_translation"
                ),
            )

        def state_at(area: float) -> tuple[float, float, float]:
            phi = self._characteristic_potential_m_s(area)
            velocity = outgoing + phi if side == "left" else outgoing - phi
            piezometric = self._piezometric_head_from_storage_m(
                area, reference_head_m=reference_head_m
            )
            residual = (
                piezometric
                + velocity * velocity / (2.0 * self.gravity_m_s2)
                - total_head
            )
            return residual, velocity, piezometric

        # The published heads differ from the initial head by millimetres.
        # Bracket the physically connected, subcritical branch around Af and
        # fail closed if a requested state would leave that branch.
        lower = 0.90 * self.full_area_m2
        upper = (1.0 + maximum) * self.full_area_m2
        f_lower, _, _ = state_at(lower)
        f_upper, _, _ = state_at(upper)
        if f_lower == 0.0:
            area_b = lower
        elif f_upper == 0.0:
            area_b = upper
        elif f_lower * f_upper > 0.0:
            raise ContractViolation(
                "Table-1 total-pressure state has no bracketed subcritical Case-1 ghost"
            )
        else:
            lo = lower
            hi = upper
            flo = f_lower
            for _ in range(90):
                mid = 0.5 * (lo + hi)
                fmid, _, _ = state_at(mid)
                if abs(fmid) <= 2.0e-13:
                    lo = hi = mid
                    break
                if flo * fmid <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    flo = fmid
            area_b = 0.5 * (lo + hi)

        residual, velocity_b, piezometric_b = state_at(area_b)
        celerity = self.celerity_m_s(area_b)
        if not math.isfinite(celerity) or celerity <= 0.0:
            raise ContractViolation("dynamic total-pressure ghost has invalid celerity")
        if abs(velocity_b) >= celerity:
            raise ContractViolation(
                "Table-1 total-pressure boundary left the subcritical characteristic branch"
            )
        invariant_b = (
            velocity_b - self._characteristic_potential_m_s(area_b)
            if side == "left"
            else velocity_b + self._characteristic_potential_m_s(area_b)
        )
        if not math.isclose(invariant_b, outgoing, rel_tol=2.0e-12, abs_tol=2.0e-12):
            raise ContractViolation("dynamic total-pressure ghost lost outgoing invariant")
        if abs(residual) > 2.0e-10:
            raise ContractViolation("dynamic total-pressure ghost did not close total head")
        return TotalPressureGhostState(
            side=side,
            liquid_area_m2=float(area_b),
            liquid_discharge_m3_s=float(area_b * velocity_b),
            piezometric_head_m=float(piezometric_b),
            total_head_m=float(total_head),
            outgoing_characteristic_m_s=float(outgoing),
            total_head_residual_m=float(residual),
            evidence_status=(
                "published_Table1_total_pressure__Case1_outgoing_characteristic__"
                "dynamic_MUSCL_ghost_declared_1D_translation"
            ),
        )

    def static_pressure_characteristic_ghost(
        self,
        *,
        interior_area_m2: float,
        interior_discharge_m3_s: float,
        prescribed_static_head_m: float,
        reference_head_m: float,
        side: Literal["left", "right"],
        maximum_elastic_overarea_fraction: float = 0.02,
    ) -> StaticPressureGhostState:
        """Retain the outgoing invariant and impose a static pressure head.

        Mahyawansi Table 1 identifies the water outlet as a Fluent pressure
        outlet, whose specified pressure is static.  The prescribed
        piezometric head therefore fixes the Case-1 storage area directly;
        only the boundary velocity is recovered from the outgoing
        characteristic.  No kinetic-head term is added to the outlet datum.
        """

        if side not in ("left", "right"):
            raise ContractViolation("static-pressure boundary side must be left or right")
        area_i = float(interior_area_m2)
        discharge_i = float(interior_discharge_m3_s)
        static_head = float(prescribed_static_head_m)
        reference = float(reference_head_m)
        maximum = float(maximum_elastic_overarea_fraction)
        if (
            not all(
                math.isfinite(value)
                for value in (area_i, discharge_i, static_head, reference, maximum)
            )
            or area_i <= 0.0
            or maximum <= 0.0
        ):
            raise ContractViolation("static-pressure characteristic inputs are not admissible")

        velocity_i = discharge_i / area_i
        phi_i = self._characteristic_potential_m_s(area_i)
        outgoing = velocity_i - phi_i if side == "left" else velocity_i + phi_i
        case1_head = self.grid.diameter_m + static_head - reference
        if case1_head <= 0.0:
            raise ContractViolation("prescribed static head maps to a non-positive Case-1 depth")
        area_b = float(self._section.area_from_head(case1_head))
        upper = (1.0 + maximum) * self.full_area_m2
        if not 0.0 < area_b <= upper:
            raise ContractViolation(
                "Table-1 static-pressure state left the admitted Case-1 storage branch"
            )
        phi_b = self._characteristic_potential_m_s(area_b)
        velocity_b = outgoing + phi_b if side == "left" else outgoing - phi_b
        celerity = self.celerity_m_s(area_b)
        if not math.isfinite(celerity) or celerity <= 0.0:
            raise ContractViolation("static-pressure ghost has invalid celerity")
        if abs(velocity_b) >= celerity:
            raise ContractViolation(
                "Table-1 static-pressure boundary left the subcritical characteristic branch"
            )
        invariant_b = velocity_b - phi_b if side == "left" else velocity_b + phi_b
        if not math.isclose(invariant_b, outgoing, rel_tol=2.0e-12, abs_tol=2.0e-12):
            raise ContractViolation("static-pressure ghost lost outgoing invariant")
        piezometric = self._piezometric_head_from_storage_m(
            area_b, reference_head_m=reference
        )
        residual = piezometric - static_head
        if abs(residual) > 2.0e-10:
            raise ContractViolation("static-pressure ghost did not close prescribed head")
        return StaticPressureGhostState(
            side=side,
            liquid_area_m2=area_b,
            liquid_discharge_m3_s=area_b * velocity_b,
            piezometric_head_m=piezometric,
            prescribed_static_head_m=static_head,
            outgoing_characteristic_m_s=outgoing,
            static_head_residual_m=residual,
            evidence_status=(
                "published_Table1_pressure_outlet_static_head__"
                "Case1_outgoing_characteristic_dynamic_MUSCL_ghost_declared_1D_translation"
            ),
        )

    def case1_muscl_central_upwind_face_fluxes(
        self,
        liquid_area_m2: Sequence[float],
        liquid_discharge_m3_s: Sequence[float],
        *,
        left_ghost: tuple[float, float],
        right_ghost: tuple[float, float],
        dry_area_fraction: float = 1.0e-10,
    ) -> Case1FaceFluxes:
        """Call the hash-pinned Case-1 MUSCL and Riemann kernels unchanged."""

        area = np.asarray(tuple(liquid_area_m2), dtype=float)
        discharge = np.asarray(tuple(liquid_discharge_m3_s), dtype=float)
        if area.ndim != 1 or area.size == 0 or discharge.shape != area.shape:
            raise ContractViolation("Case-1 segment states must be equal non-empty vectors")
        if np.any(~np.isfinite(area)) or np.any(~np.isfinite(discharge)) or np.any(area < 0.0):
            raise ContractViolation("Case-1 segment state is not finite/admissible")
        dry_fraction = float(dry_area_fraction)
        if not math.isfinite(dry_fraction) or not 0.0 <= dry_fraction < 1.0:
            raise ContractViolation("Case-1 dry-area fraction is invalid")
        area_ext = np.empty(area.size + 2)
        discharge_ext = np.empty(discharge.size + 2)
        area_ext[1:-1] = area
        discharge_ext[1:-1] = discharge
        area_ext[0], discharge_ext[0] = map(float, left_ghost)
        area_ext[-1], discharge_ext[-1] = map(float, right_ghost)
        dry_area = dry_fraction * self.full_area_m2
        al, ql, ar, qr = self._case1_seed._muscl_free_surface_face_states(
            area_ext,
            discharge_ext,
            self._section,
            dry_area,
        )
        mass, momentum = self._case1_seed._central_upwind_flux(
            al,
            ql,
            ar,
            qr,
            self._section,
            dry_area,
        )
        return Case1FaceFluxes(
            liquid_volume_m3_s=tuple(float(value) for value in mass),
            liquid_momentum_m4_s2=tuple(float(value) for value in momentum),
        )

    def case1_donor_draining_limit(
        self,
        liquid_area_m2: Sequence[float],
        liquid_volume_face_flux_m3_s: Sequence[float],
        liquid_momentum_face_flux_m4_s2: Sequence[float],
        *,
        dx_m: float,
        dt_s: float,
    ) -> Case1FaceFluxes:
        """Call the pinned donor limiter on a complete S1 segment."""

        area = np.asarray(tuple(liquid_area_m2), dtype=float)
        mass = np.asarray(tuple(liquid_volume_face_flux_m3_s), dtype=float)
        momentum = np.asarray(tuple(liquid_momentum_face_flux_m4_s2), dtype=float)
        try:
            limited_mass, limited_momentum = (
                self._case1_seed._apply_donor_draining_limiter(
                    mass,
                    momentum,
                    area,
                    dx=float(dx_m),
                    dt=float(dt_s),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ContractViolation("Case-1 donor limiter rejected S1 segment inputs") from exc
        return Case1FaceFluxes(
            liquid_volume_m3_s=tuple(float(value) for value in limited_mass),
            liquid_momentum_m4_s2=tuple(float(value) for value in limited_momentum),
        )

    def head_from_area_m(self, liquid_area_m2: float) -> float:
        """Expose the Case-1 area-to-head map without exposing its topology."""

        area = float(liquid_area_m2)
        if not math.isfinite(area) or area < 0.0:
            raise ContractViolation("liquid area must be finite and non-negative")
        return float(self._section.head_from_area(area))

    def wetted_perimeter_m(self, liquid_area_m2: float) -> float:
        """Return the Case-1 circular wetted perimeter for wall closures."""

        area = float(liquid_area_m2)
        if not math.isfinite(area) or not 0.0 <= area <= self.full_area_m2:
            raise ContractViolation("liquid area must lie inside the rigid pipe")
        depth = float(self._section.depth_from_area(area))
        return float(self._section.wetted_perimeter(depth))

    def interface_width_m(self, liquid_area_m2: float) -> float:
        """Return the circular free-surface chord used by declared drag closures."""

        area = float(liquid_area_m2)
        if not math.isfinite(area) or not 0.0 <= area <= self.full_area_m2:
            raise ContractViolation("liquid area must lie inside the rigid pipe")
        if area == 0.0 or area == self.full_area_m2:
            return 0.0
        depth = float(self._section.depth_from_area(area))
        return float(self._section.top_width(depth))

    def build_stage1_initial_state(
        self,
        *,
        topology: str = CONTINUOUS_AIR_TOPOLOGY,
        finite_gas_pocket_length_m: float | None = None,
        valve_release_shockfit: bool = False,
        initial_piezometric_head_m: float | None = None,
        elastic_storage_reference_head_m: float | None = None,
    ) -> HorizontalState:
        validate_mahywansi_initialization(
            topology=topology,
            finite_gas_pocket_length_m=finite_gas_pocket_length_m,
            valve_release_shockfit=valve_release_shockfit,
        )
        if (initial_piezometric_head_m is None) != (
            elastic_storage_reference_head_m is None
        ):
            raise ContractViolation(
                "initial and elastic-reference heads must be supplied together"
            )
        initial_area = self.full_area_m2
        if initial_piezometric_head_m is not None:
            initial_head = float(initial_piezometric_head_m)
            reference_head = float(elastic_storage_reference_head_m)
            if not all(math.isfinite(value) for value in (initial_head, reference_head)):
                raise ContractViolation("Stage-1 horizontal heads must be finite")
            case1_head = self.grid.diameter_m + initial_head - reference_head
            if case1_head < self.grid.diameter_m:
                raise ContractViolation(
                    "gas-free S1 initialization cannot use a sub-full Case-1 area; "
                    "select an elastic reference no higher than the initial head"
                )
            initial_area = float(self._section.area_from_head(case1_head))
        count = self.grid.cell_count
        return HorizontalState(
            Al=(initial_area,) * count,
            Ql=(0.0,) * count,
            Mg=(0.0,) * count,
            Jg=(0.0,) * count,
        )


def build_s1_2d_eos_aligned_horizontal_adapter(
    grid: MahyawansiHorizontalGrid | None = None,
) -> Case1HorizontalLiquidAdapter:
    """Build the Case-1 geometry/pressure-law adapter with the 2-D EOS tangent.

    OpenFOAM v2512 ``perfectFluid`` uses ``rho=rho0+p/(R*T)``.  Its local
    isothermal pressure-density tangent is therefore ``dp/drho=R*T`` and the
    matching acoustic speed is ``sqrt(R*T)``.  This changes only the S1
    thermodynamic parameter; the Case-1 circular geometry and pressure-flux
    form remain unchanged.  The returned adapter still exposes the verified
    Case-1 MUSCL, central-upwind and donor-draining spatial kernels; selecting
    this tangent is not a claim of full thermodynamic EOS equivalence.
    """

    return Case1HorizontalLiquidAdapter(
        grid,
        wave_speed_m_s=FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S,
        wave_speed_evidence=FROZEN_2D_WATER_EOS_EVIDENCE,
    )


__all__ = [
    "CASE1_SEED_PATHS",
    "CASE1_SEED_SHA256",
    "CONTINUOUS_AIR_TOPOLOGY",
    "DEFAULT_CASE1_WAVE_SPEED_M_S",
    "DEFAULT_GRAVITY_M_S2",
    "DEFAULT_HORIZONTAL_DX_M",
    "FROZEN_2D_TEMPERATURE_K",
    "FROZEN_2D_WATER_EOS_EVIDENCE",
    "FROZEN_2D_WATER_PERFECT_FLUID_R_J_KG_K",
    "FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S",
    "Case1HorizontalLiquidAdapter",
    "Case1FaceFluxes",
    "Case1SourceIntegrityError",
    "ForbiddenCase1Topology",
    "LiquidPhysicalFlux",
    "MahyawansiHorizontalGrid",
    "NumericalParameterProvenance",
    "StaticPressureGhostState",
    "TotalPressureGhostState",
    "build_s1_2d_eos_aligned_horizontal_adapter",
    "validate_mahywansi_initialization",
    "verify_case1_seed_integrity",
]
