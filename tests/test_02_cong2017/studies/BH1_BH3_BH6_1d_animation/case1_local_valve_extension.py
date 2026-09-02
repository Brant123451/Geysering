"""Version-pinned infrastructure for a fixed Campaign-2 valve face.

This sibling deliberately leaves the hash-locked Case-1 source untouched.  It
provides immutable geometry, transaction, partition and event-plan types plus
the generic clean-free-surface finite-volume stage plumbing needed before a
conservative local-valve implementation can be added.

When ``local_face`` is ``None``, :meth:`Case1LocalValveExtension.step` dispatches
directly to the original Case-1 method.  An active clean free-surface face uses
the directional circular Saint-Venant characteristic count: a dry or
supercritical-outflow downstream trace supplies no incoming condition, while a
supercritical upstream trace is explicitly Riemann-supply controlled.  A
separate coupled cut-element closure handles ``cut in {60,61}``; a
conservative split-MOC two-port handles ``cut<=59``.  This module never clips a
Froude number, inserts a liquid film, changes elapsed time, or adds a post-step
force.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
TESTS_ROOT = HERE.parents[2]
CASE1_MODEL = (
    TESTS_ROOT
    / "test_01_vw2011"
    / "cases"
    / "A_Dt57p1_Ha0305_Yfs0356"
    / "model"
)
if str(CASE1_MODEL) not in sys.path:
    sys.path.insert(0, str(CASE1_MODEL))

import tosan2021_horizontal_shockfit as _case1_core  # noqa: E402
from tosan2021_horizontal_shockfit import (  # noqa: E402
    CircularSection,
    HorizontalConfig,
    HorizontalState,
    Tosan2021HorizontalShockFit,
    WetDryState,
)

from campaign2_local_valve import (  # noqa: E402
    CircularSaintVenantValveSolution,
    CircularSaintVenantValveTrace,
    LiquidValveTrace,
    OPENING_DURATION_S,
    PressurisedMocValveControlRegime,
    PressurisedMocValveNoRootError,
    PressurisedMocValveSolution,
    WATER_DENSITY_KG_M3,
    shared_opening_state,
    solve_passive_circular_saint_venant_valve,
    solve_passive_pressurised_moc_valve,
)


CORE_SOURCE = CASE1_MODEL / "tosan2021_horizontal_shockfit.py"
EXPECTED_CORE_SHA256 = (
    "90e84da9afa0ec8465d80f87fc701dfb8f0fad6f97350ea708074a50192b6119"
)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _require_pinned_core() -> str:
    """Verify the immutable dependency once per importing Python process."""

    actual_hash = source_sha256(CORE_SOURCE)
    if actual_hash != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "the Case-1 core does not match the version-pinned extension: "
            f"expected {EXPECTED_CORE_SHA256}, found {actual_hash}"
        )
    return actual_hash


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _roundoff_tolerance(*values: float, multiplier: float = 256.0) -> float:
    scale = max(1.0, *(abs(float(value)) for value in values))
    return multiplier * math.ulp(scale)


@dataclass(frozen=True)
class FixedInternalValveSpec:
    """Exact physical and mirrored geometry of one zero-volume valve face.

    ``valve_flow_area_m2`` is the nominal full circular area used to audit the
    apparatus and by the separate full-liquid two-port API.  The clean
    free-surface closure converts ``Q`` to the local 2D-style liquid velocity
    with its upstream *wetted* area (reconstructed for two-sided control and
    characteristic-solved for one-sided control); it never substitutes this
    nominal full area for a partially filled trace.
    """

    physical_domain_length_m: float
    physical_valve_x_m: float
    grid_dx_m: float
    pipe_diameter_m: float
    expected_mirrored_face_index: int = 61
    opening_duration_s: float = OPENING_DURATION_S
    liquid_density_kg_m3: float = WATER_DENSITY_KG_M3
    mirrored_valve_x_m: float = field(init=False)
    mirrored_face_index: int = field(init=False)
    mirrored_face_x_m: float = field(init=False)
    physical_face_x_m: float = field(init=False)
    valve_flow_area_m2: float = field(init=False)

    def __post_init__(self) -> None:
        length = _positive_finite(
            self.physical_domain_length_m,
            name="physical_domain_length_m",
        )
        physical_x = _finite(
            self.physical_valve_x_m,
            name="physical_valve_x_m",
        )
        dx = _positive_finite(self.grid_dx_m, name="grid_dx_m")
        diameter = _positive_finite(
            self.pipe_diameter_m,
            name="pipe_diameter_m",
        )
        opening_duration = _positive_finite(
            self.opening_duration_s,
            name="opening_duration_s",
        )
        density = _positive_finite(
            self.liquid_density_kg_m3,
            name="liquid_density_kg_m3",
        )
        opening_tolerance = _roundoff_tolerance(
            opening_duration,
            OPENING_DURATION_S,
        )
        if not math.isclose(
            opening_duration,
            OPENING_DURATION_S,
            rel_tol=0.0,
            abs_tol=opening_tolerance,
        ):
            raise ValueError(
                "the Campaign-2 valve opening duration must be 0.20 s"
            )
        expected_face = self.expected_mirrored_face_index
        if (
            isinstance(expected_face, bool)
            or not isinstance(expected_face, int)
            or expected_face <= 0
        ):
            raise ValueError(
                "expected_mirrored_face_index must be a positive integer"
            )
        if not (0.0 < physical_x < length):
            raise ValueError("physical_valve_x_m must lie inside the domain")

        mirrored_x = length - physical_x
        raw_face = mirrored_x / dx
        nearest_face = int(round(raw_face))
        alignment_tolerance = _roundoff_tolerance(raw_face)
        if abs(raw_face - nearest_face) > alignment_tolerance:
            raise ValueError(
                "the physical valve does not coincide with a finite-volume "
                f"face: mirrored x/dx={raw_face:.17g}"
            )
        if nearest_face != expected_face:
            raise ValueError(
                "the Campaign-2 valve must be mirrored face "
                f"{expected_face}, found face {nearest_face}"
            )

        face_x = nearest_face * dx
        physical_face_x = length - face_x
        position_tolerance = dx * alignment_tolerance
        if abs(mirrored_x - face_x) > position_tolerance:
            raise ValueError("the mirrored valve position is not face aligned")
        if abs(physical_x - physical_face_x) > position_tolerance:
            raise ValueError("the physical/mirrored valve mapping is inconsistent")

        object.__setattr__(self, "physical_domain_length_m", length)
        object.__setattr__(self, "physical_valve_x_m", physical_x)
        object.__setattr__(self, "grid_dx_m", dx)
        object.__setattr__(self, "pipe_diameter_m", diameter)
        object.__setattr__(self, "opening_duration_s", opening_duration)
        object.__setattr__(self, "liquid_density_kg_m3", density)
        object.__setattr__(self, "mirrored_valve_x_m", mirrored_x)
        object.__setattr__(self, "mirrored_face_index", nearest_face)
        object.__setattr__(self, "mirrored_face_x_m", face_x)
        object.__setattr__(self, "physical_face_x_m", physical_face_x)
        object.__setattr__(
            self,
            "valve_flow_area_m2",
            0.25 * math.pi * diameter * diameter,
        )

    def validate_against_solver(
        self,
        solver: Tosan2021HorizontalShockFit,
    ) -> None:
        """Reject a face that is not exactly the configured Case-1 face."""

        if not isinstance(solver, Tosan2021HorizontalShockFit):
            raise TypeError("solver must be a Tosan2021HorizontalShockFit")
        checks = (
            (
                "physical domain length",
                float(solver.config.length),
                self.physical_domain_length_m,
            ),
            ("grid spacing", float(solver.dx), self.grid_dx_m),
            (
                "pipe diameter",
                float(solver.config.diameter),
                self.pipe_diameter_m,
            ),
            (
                "mirrored valve position",
                float(solver.config.valve_x),
                self.mirrored_face_x_m,
            ),
            (
                "circular valve area",
                float(solver.section.full_area),
                self.valve_flow_area_m2,
            ),
            (
                "liquid density",
                float(solver.config.liquid_density),
                self.liquid_density_kg_m3,
            ),
        )
        for label, actual, expected in checks:
            tolerance = _roundoff_tolerance(actual, expected)
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    f"{label} differs between the valve and Case-1 solver: "
                    f"{actual:.17g} != {expected:.17g}"
                )
        if not (1 <= self.mirrored_face_index < solver.ncell):
            raise ValueError("the valve face must have two adjacent cells")

    @staticmethod
    def mirrored_volume_from_physical(value_m3: float) -> float:
        """Map signed west-to-east physical volume to increasing mirror x."""

        return -_finite(value_m3, name="physical signed volume")

    @staticmethod
    def physical_volume_from_mirrored(value_m3: float) -> float:
        return -_finite(value_m3, name="mirrored signed volume")

    @staticmethod
    def mirrored_wall_impulse_from_physical(value_N_s: float) -> float:
        """Axial force changes sign under ``x_mirror=L-x_physical``."""

        return -_finite(value_N_s, name="physical wall impulse")

    @staticmethod
    def physical_wall_impulse_from_mirrored(value_N_s: float) -> float:
        return -_finite(value_N_s, name="mirrored wall impulse")

    @staticmethod
    def mirrored_face_momentum_impulses_from_physical(
        *,
        physical_left_N_s: float,
        physical_right_N_s: float,
    ) -> tuple[float, float]:
        """Return ``(mirror_west, mirror_east)`` face momentum impulses.

        The scalar axial momentum flux is invariant to the coordinate reversal,
        while the two physical ports exchange order.
        """

        left = _finite(physical_left_N_s, name="physical_left_N_s")
        right = _finite(physical_right_N_s, name="physical_right_N_s")
        return right, left


@dataclass(frozen=True)
class IntegratedValveTransaction:
    """One uncommitted physical-step integral across the fixed valve."""

    start_time_s: float
    end_time_s: float
    physical_signed_through_volume_m3: float
    physical_left_momentum_impulse_N_s: float
    physical_right_momentum_impulse_N_s: float
    physical_wall_impulse_on_liquid_N_s: float
    dissipated_energy_J: float
    liquid_density_kg_m3: float = WATER_DENSITY_KG_M3
    stage_evaluation_count: int = 0
    substep_count: int = 0

    def __post_init__(self) -> None:
        start = _finite(self.start_time_s, name="start_time_s")
        end = _finite(self.end_time_s, name="end_time_s")
        if end < start:
            raise ValueError("end_time_s cannot precede start_time_s")
        density = _positive_finite(
            self.liquid_density_kg_m3,
            name="liquid_density_kg_m3",
        )
        scalar_fields = (
            self.physical_signed_through_volume_m3,
            self.physical_left_momentum_impulse_N_s,
            self.physical_right_momentum_impulse_N_s,
            self.physical_wall_impulse_on_liquid_N_s,
            self.dissipated_energy_J,
        )
        if not all(math.isfinite(float(value)) for value in scalar_fields):
            raise ValueError("all valve transaction integrals must be finite")
        if self.dissipated_energy_J < 0.0:
            raise ValueError("a passive valve cannot dissipate negative energy")
        for name, count in (
            ("stage_evaluation_count", self.stage_evaluation_count),
            ("substep_count", self.substep_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        momentum_residual = (
            float(self.physical_right_momentum_impulse_N_s)
            - float(self.physical_left_momentum_impulse_N_s)
            - float(self.physical_wall_impulse_on_liquid_N_s)
        )
        tolerance = _roundoff_tolerance(
            self.physical_right_momentum_impulse_N_s,
            self.physical_left_momentum_impulse_N_s,
            self.physical_wall_impulse_on_liquid_N_s,
        )
        if abs(momentum_residual) > tolerance:
            raise ValueError(
                "two-side momentum impulses do not close with valve-wall impulse"
            )
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "liquid_density_kg_m3", density)

    @classmethod
    def zero(
        cls,
        *,
        start_time_s: float,
        end_time_s: float,
        liquid_density_kg_m3: float = WATER_DENSITY_KG_M3,
    ) -> "IntegratedValveTransaction":
        return cls(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            physical_signed_through_volume_m3=0.0,
            physical_left_momentum_impulse_N_s=0.0,
            physical_right_momentum_impulse_N_s=0.0,
            physical_wall_impulse_on_liquid_N_s=0.0,
            dissipated_energy_J=0.0,
            liquid_density_kg_m3=liquid_density_kg_m3,
            stage_evaluation_count=0,
            substep_count=0,
        )

    @property
    def duration_s(self) -> float:
        return float(self.end_time_s - self.start_time_s)

    @property
    def mirrored_signed_through_volume_m3(self) -> float:
        return -float(self.physical_signed_through_volume_m3)

    @property
    def physical_signed_through_mass_kg(self) -> float:
        return float(
            self.liquid_density_kg_m3
            * self.physical_signed_through_volume_m3
        )

    @property
    def physical_left_liquid_mass_change_kg(self) -> float:
        return -self.physical_signed_through_mass_kg

    @property
    def physical_right_liquid_mass_change_kg(self) -> float:
        return self.physical_signed_through_mass_kg

    @property
    def liquid_mass_residual_kg(self) -> float:
        return float(
            self.physical_left_liquid_mass_change_kg
            + self.physical_right_liquid_mass_change_kg
        )

    @property
    def mirrored_wall_impulse_on_liquid_N_s(self) -> float:
        return -float(self.physical_wall_impulse_on_liquid_N_s)

    @property
    def momentum_impulse_residual_N_s(self) -> float:
        return float(
            self.physical_right_momentum_impulse_N_s
            - self.physical_left_momentum_impulse_N_s
            - self.physical_wall_impulse_on_liquid_N_s
        )

    def merged(
        self,
        other: "IntegratedValveTransaction",
    ) -> "IntegratedValveTransaction":
        """Merge two adjacent, still-uncommitted substep integrals."""

        if not isinstance(other, IntegratedValveTransaction):
            raise TypeError("other must be an IntegratedValveTransaction")
        time_tolerance = _roundoff_tolerance(self.end_time_s, other.start_time_s)
        if not math.isclose(
            self.end_time_s,
            other.start_time_s,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ):
            raise ValueError("valve transactions must be time-contiguous")
        density_tolerance = _roundoff_tolerance(
            self.liquid_density_kg_m3,
            other.liquid_density_kg_m3,
        )
        if not math.isclose(
            self.liquid_density_kg_m3,
            other.liquid_density_kg_m3,
            rel_tol=0.0,
            abs_tol=density_tolerance,
        ):
            raise ValueError("valve transactions must use one liquid density")
        return IntegratedValveTransaction(
            start_time_s=self.start_time_s,
            end_time_s=other.end_time_s,
            physical_signed_through_volume_m3=(
                self.physical_signed_through_volume_m3
                + other.physical_signed_through_volume_m3
            ),
            physical_left_momentum_impulse_N_s=(
                self.physical_left_momentum_impulse_N_s
                + other.physical_left_momentum_impulse_N_s
            ),
            physical_right_momentum_impulse_N_s=(
                self.physical_right_momentum_impulse_N_s
                + other.physical_right_momentum_impulse_N_s
            ),
            physical_wall_impulse_on_liquid_N_s=(
                self.physical_wall_impulse_on_liquid_N_s
                + other.physical_wall_impulse_on_liquid_N_s
            ),
            dissipated_energy_J=(
                self.dissipated_energy_J + other.dissipated_energy_J
            ),
            liquid_density_kg_m3=self.liquid_density_kg_m3,
            stage_evaluation_count=(
                self.stage_evaluation_count + other.stage_evaluation_count
            ),
            substep_count=self.substep_count + other.substep_count,
        )


class LocalValveRegime(str, Enum):
    CLEAN_FREE_SURFACE_FV = "clean_free_surface_fv"
    SHOCK_CUT_CELL = "shock_cut_cell"
    CLEAN_PRESSURISED_MOC = "clean_pressurised_moc"


@dataclass(frozen=True)
class LocalFacePartition:
    regime: LocalValveRegime
    fixed_face_index: int
    shock_cut_cell_index: int
    free_surface_stop_index: int
    pressurised_start_index: int
    interface_x_m: float


@dataclass(frozen=True)
class EventAlignedStepPlan:
    """A physical step split only at exact prescribed event times."""

    start_time_s: float
    end_time_s: float
    substeps_s: tuple[float, ...]
    interior_events_s: tuple[float, ...]

    def __post_init__(self) -> None:
        start = _finite(self.start_time_s, name="start_time_s")
        end = _finite(self.end_time_s, name="end_time_s")
        if end <= start:
            raise ValueError("an event-aligned step must have positive duration")
        if not self.substeps_s:
            raise ValueError("an event-aligned step needs at least one substep")
        substeps = tuple(
            _positive_finite(value, name="event-aligned substep")
            for value in self.substeps_s
        )
        events = tuple(
            _finite(value, name="interior event")
            for value in self.interior_events_s
        )
        if tuple(sorted(set(events))) != events:
            raise ValueError("interior events must be unique and increasing")
        if any(not (start < event < end) for event in events):
            raise ValueError("interior events must lie strictly inside the step")
        total = math.fsum(substeps)
        tolerance = _roundoff_tolerance(total, end - start)
        if not math.isclose(total, end - start, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("event-aligned substeps do not span the request")
        if len(substeps) != len(events) + 1:
            raise ValueError("each interior event must create one extra substep")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "substeps_s", substeps)
        object.__setattr__(self, "interior_events_s", events)


def plan_event_aligned_step(
    *,
    start_time_s: float,
    dt_s: float,
    event_times_s: Iterable[float],
) -> EventAlignedStepPlan:
    """Split a positive physical step at every strict interior event."""

    start = _finite(start_time_s, name="start_time_s")
    step = _positive_finite(dt_s, name="dt_s")
    end = start + step
    if not math.isfinite(end):
        raise ValueError("physical step end time must be finite")
    events_all = sorted({_finite(value, name="event time") for value in event_times_s})
    events = tuple(event for event in events_all if start < event < end)
    bounds = (start, *events, end)
    substeps = tuple(
        float(right - left)
        for left, right in zip(bounds, bounds[1:])
    )
    return EventAlignedStepPlan(
        start_time_s=start,
        end_time_s=end,
        substeps_s=substeps,
        interior_events_s=events,
    )


@dataclass(frozen=True)
class InternalFaceStageContext:
    """Reconstructed native data presented to one FV-stage callback.

    ``face_index`` uses the Case-1 finite-volume convention: face ``f`` lies
    between cells ``f-1`` (west) and ``f`` (east).  The callback receives only
    immutable scalars, so it cannot mutate the provisional stage state.
    """

    stage_index: int
    stage_time_s: float
    dt_s: float
    face_index: int
    west_area_m2: float
    west_discharge_m3_s: float
    east_area_m2: float
    east_discharge_m3_s: float
    native_shared_mass_flux_m3_s: float
    native_momentum_flux_m4_s2: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.stage_index, bool)
            or not isinstance(self.stage_index, int)
            or self.stage_index not in (1, 2)
        ):
            raise ValueError("stage_index must be SSPRK stage 1 or 2")
        if (
            isinstance(self.face_index, bool)
            or not isinstance(self.face_index, int)
            or self.face_index <= 0
        ):
            raise ValueError("face_index must be a positive internal-face index")
        _finite(self.stage_time_s, name="stage_time_s")
        _positive_finite(self.dt_s, name="dt_s")
        scalar_values = (
            self.west_area_m2,
            self.west_discharge_m3_s,
            self.east_area_m2,
            self.east_discharge_m3_s,
            self.native_shared_mass_flux_m3_s,
            self.native_momentum_flux_m4_s2,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("all reconstructed internal-face values must be finite")
        if self.west_area_m2 < 0.0 or self.east_area_m2 < 0.0:
            raise ValueError("reconstructed internal-face areas cannot be negative")


@dataclass(frozen=True)
class InternalFaceFluxPair:
    """One shared volume flux and the two momentum fluxes at a zero-volume face.

    The west momentum flux is the east-boundary flux seen by cell ``f-1``;
    the east momentum flux is the west-boundary flux seen by cell ``f``.  Their
    difference is therefore the resolved axial wall-force contribution per
    unit liquid density.  This type carries no valve constitutive law.
    """

    shared_mass_flux_m3_s: float
    west_momentum_flux_m4_s2: float
    east_momentum_flux_m4_s2: float

    def __post_init__(self) -> None:
        values = (
            self.shared_mass_flux_m3_s,
            self.west_momentum_flux_m4_s2,
            self.east_momentum_flux_m4_s2,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("all internal-face fluxes must be finite")

    @classmethod
    def native(cls, context: InternalFaceStageContext) -> "InternalFaceFluxPair":
        """Return the exact native Case-1 numerical flux on both ports."""

        if not isinstance(context, InternalFaceStageContext):
            raise TypeError("context must be an InternalFaceStageContext")
        return cls(
            shared_mass_flux_m3_s=context.native_shared_mass_flux_m3_s,
            west_momentum_flux_m4_s2=context.native_momentum_flux_m4_s2,
            east_momentum_flux_m4_s2=context.native_momentum_flux_m4_s2,
        )

    def is_exact_native(self, context: InternalFaceStageContext) -> bool:
        """Identify a transparent/K=0 decision without an alternate update."""

        return bool(
            self.shared_mass_flux_m3_s
            == context.native_shared_mass_flux_m3_s
            and self.west_momentum_flux_m4_s2
            == context.native_momentum_flux_m4_s2
            and self.east_momentum_flux_m4_s2
            == context.native_momentum_flux_m4_s2
        )


InternalFaceFluxCallback = Callable[
    [InternalFaceStageContext],
    InternalFaceFluxPair | None,
]


@dataclass(frozen=True)
class InternalFaceStageRecord:
    """Accepted, donor-limited face fluxes for one SSPRK Euler stage."""

    context: InternalFaceStageContext
    requested_flux: InternalFaceFluxPair | None
    accepted_flux: InternalFaceFluxPair
    donor_scale: float
    used_native_core_stage: bool

    def __post_init__(self) -> None:
        if not isinstance(self.context, InternalFaceStageContext):
            raise TypeError("context must be an InternalFaceStageContext")
        if self.requested_flux is not None and not isinstance(
            self.requested_flux,
            InternalFaceFluxPair,
        ):
            raise TypeError("requested_flux must be InternalFaceFluxPair or None")
        if not isinstance(self.accepted_flux, InternalFaceFluxPair):
            raise TypeError("accepted_flux must be an InternalFaceFluxPair")
        scale = _finite(self.donor_scale, name="donor_scale")
        if not (0.0 <= scale <= 1.0):
            raise ValueError("donor_scale must lie in [0, 1]")
        if not isinstance(self.used_native_core_stage, bool):
            raise TypeError("used_native_core_stage must be bool")


@dataclass(frozen=True)
class InternalFaceSsprk2Result:
    """Wet/dry SSPRK2 state and immutable accepted stage-face records."""

    state: WetDryState
    stage_records: tuple[InternalFaceStageRecord, ...]
    used_native_core_step: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, WetDryState):
            raise TypeError("state must be a WetDryState")
        if not all(
            isinstance(record, InternalFaceStageRecord)
            for record in self.stage_records
        ):
            raise TypeError("stage_records must contain InternalFaceStageRecord")
        if len(self.stage_records) not in (0, 2):
            raise ValueError("an SSPRK2 result must contain zero or two stage records")
        if not isinstance(self.used_native_core_step, bool):
            raise TypeError("used_native_core_step must be bool")


class ShockCutValveOrientation(str, Enum):
    """Which side of the fixed face contains the fitted shock subcell."""

    PRESSURISED_TOUCH = "pressurised_touch"
    FREE_SURFACE_TOUCH = "free_surface_touch"
    FACE_CONTACT = "face_contact"
    DRY_FACE_INJECTION = "dry_face_injection"


@dataclass(frozen=True)
class ShockCutValveStageSolution:
    """One implicit fixed-valve/moving-shock stage solution.

    The valve has one stationary mass flux and two momentum ports.  The
    moving Tosan interface is represented separately by its free-surface and
    pressurised traces.  Consequently the cut-cell storage can be rebuilt
    geometrically without inventing a wet film or losing either side of the
    valve transaction.
    """

    stage_index: int
    stage_time_s: float
    orientation: ShockCutValveOrientation
    fixed_face_index: int
    shock_cut_cell_index: int
    shared_mass_flux_m3_s: float
    left_momentum_flux_m4_s2: float
    right_momentum_flux_m4_s2: float
    signed_pressure_jump_Pa: float
    valve_wall_force_on_liquid_N: float
    dissipation_power_W: float
    valve_left_area_m2: float
    valve_right_area_m2: float
    interface_free_surface_area_m2: float
    interface_free_surface_discharge_m3_s: float
    interface_pressurised_velocity_m_s: float
    interface_pressurised_head_m: float
    interface_speed_m_s: float
    valve_left_head_m: float | None
    valve_right_head_m: float | None
    nonlinear_residual_linf: float
    nonlinear_evaluation_count: int
    left_nonpenetration_residual_m_s: float | None = None
    right_nonpenetration_residual_m_s: float | None = None

    def __post_init__(self) -> None:
        if self.stage_index not in (1, 2):
            raise ValueError("stage_index must be SSPRK stage 1 or 2")
        if not isinstance(self.orientation, ShockCutValveOrientation):
            raise TypeError("orientation must be ShockCutValveOrientation")
        if self.fixed_face_index <= 0 or self.shock_cut_cell_index < 0:
            raise ValueError("invalid fixed-face or shock-cut index")
        values = (
            self.stage_time_s,
            self.shared_mass_flux_m3_s,
            self.left_momentum_flux_m4_s2,
            self.right_momentum_flux_m4_s2,
            self.signed_pressure_jump_Pa,
            self.valve_wall_force_on_liquid_N,
            self.dissipation_power_W,
            self.valve_left_area_m2,
            self.valve_right_area_m2,
            self.interface_free_surface_area_m2,
            self.interface_free_surface_discharge_m3_s,
            self.interface_pressurised_velocity_m_s,
            self.interface_pressurised_head_m,
            self.interface_speed_m_s,
            self.nonlinear_residual_linf,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("all shock-cut stage fields must be finite")
        if min(
            self.valve_left_area_m2,
            self.valve_right_area_m2,
            self.interface_free_surface_area_m2,
        ) <= 0.0:
            raise ValueError("shock-cut liquid areas must be positive")
        if self.dissipation_power_W < 0.0:
            raise ValueError("a passive shock-cut valve cannot create energy")
        if self.nonlinear_residual_linf < 0.0:
            raise ValueError("nonlinear_residual_linf cannot be negative")
        if (
            isinstance(self.nonlinear_evaluation_count, bool)
            or self.nonlinear_evaluation_count <= 0
        ):
            raise ValueError("nonlinear_evaluation_count must be positive")
        for head in (self.valve_left_head_m, self.valve_right_head_m):
            if head is not None and not math.isfinite(float(head)):
                raise ValueError("optional valve heads must be finite")
        for residual in (
            self.left_nonpenetration_residual_m_s,
            self.right_nonpenetration_residual_m_s,
        ):
            if residual is not None and not math.isfinite(float(residual)):
                raise ValueError("optional nonpenetration residuals must be finite")
        if self.orientation is ShockCutValveOrientation.FACE_CONTACT:
            if (
                self.left_nonpenetration_residual_m_s is None
                or self.right_nonpenetration_residual_m_s is None
            ):
                raise ValueError(
                    "a face-contact stage needs both nonpenetration residuals"
                )
            if self.interface_speed_m_s != 0.0:
                raise ValueError("a face-contact stage must have exactly zero speed")
        if self.orientation is ShockCutValveOrientation.DRY_FACE_INJECTION:
            if self.interface_speed_m_s != 0.0:
                raise ValueError(
                    "a dry-face injection stage must have exactly zero speed"
                )

        density = WATER_DENSITY_KG_M3
        force_residual = (
            density
            * (self.right_momentum_flux_m4_s2 - self.left_momentum_flux_m4_s2)
            - self.valve_wall_force_on_liquid_N
        )
        tolerance = _roundoff_tolerance(
            density * self.right_momentum_flux_m4_s2,
            density * self.left_momentum_flux_m4_s2,
            self.valve_wall_force_on_liquid_N,
            multiplier=4096.0,
        )
        if abs(force_residual) > tolerance:
            raise ValueError("shock-cut momentum ports do not close with wall force")
        power = self.signed_pressure_jump_Pa * self.shared_mass_flux_m3_s
        if not math.isclose(
            power,
            self.dissipation_power_W,
            rel_tol=0.0,
            abs_tol=_roundoff_tolerance(
                power,
                self.dissipation_power_W,
                multiplier=4096.0,
            ),
        ):
            raise ValueError("shock-cut dissipation is not dp*Q")


@dataclass(frozen=True)
class _ShockCutFaceContactRoot:
    """Five-equation stationary valve/interface contact root.

    The free-surface valve has two traces and one shared discharge.  The
    pressurised trace satisfies the incoming water-hammer characteristic.
    Rankine--Hugoniot mass and momentum are imposed with ``w=0``; the missing
    free-surface valve-momentum equation is the fixed-face contact reaction,
    not an empirical speed constraint.
    """

    theta_left: float
    theta_right: float
    q_scaled: float
    pressure_velocity_scaled: float
    pressure_head_scaled: float
    residual_linf: float
    evaluation_count: int

    @property
    def free_branch_guess(self) -> np.ndarray:
        return np.array(
            (
                self.theta_left,
                self.theta_right,
                self.q_scaled,
                self.pressure_velocity_scaled,
                self.pressure_head_scaled,
                0.0,
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class PressurisedMocValveStageSolution:
    """One source-corrected split-MOC valve stage.

    ``native_shared_mass_flux_m3_s`` is the exact ``K=0`` two-characteristic
    datum formed from the same incoming feet.  Replacing it by the passive
    shared flux gives equal-and-opposite elastic storage corrections on the
    two cells adjacent to the zero-volume face.  Momentum has two ports; their
    difference is the stationary valve-wall force.
    """

    stage_index: int
    stage_time_s: float
    fixed_face_index: int
    shock_cut_cell_index: int
    control_regime: PressurisedMocValveControlRegime
    native_shared_mass_flux_m3_s: float
    shared_mass_flux_m3_s: float
    left_gauge_pressure_Pa: float
    right_gauge_pressure_Pa: float
    left_head_m: float
    right_head_m: float
    left_momentum_flux_m4_s2: float
    right_momentum_flux_m4_s2: float
    native_momentum_flux_m4_s2: float
    signed_pressure_jump_Pa: float
    valve_wall_force_on_liquid_N: float
    dissipation_power_W: float
    left_trace_area_m2: float
    right_trace_area_m2: float
    left_incoming_characteristic_count: int
    right_incoming_characteristic_count: int
    left_characteristic_residual_m_s: float
    right_characteristic_residual_m_s: float

    def __post_init__(self) -> None:
        if self.stage_index not in (1, 2):
            raise ValueError("stage_index must be SSPRK stage 1 or 2")
        if self.fixed_face_index <= 0 or self.shock_cut_cell_index < 0:
            raise ValueError("invalid fixed-face or shock-cut index")
        if not isinstance(
            self.control_regime,
            PressurisedMocValveControlRegime,
        ):
            raise TypeError("control_regime must be a pressurised-MOC regime")
        values = (
            self.stage_time_s,
            self.native_shared_mass_flux_m3_s,
            self.shared_mass_flux_m3_s,
            self.left_gauge_pressure_Pa,
            self.right_gauge_pressure_Pa,
            self.left_head_m,
            self.right_head_m,
            self.left_momentum_flux_m4_s2,
            self.right_momentum_flux_m4_s2,
            self.native_momentum_flux_m4_s2,
            self.signed_pressure_jump_Pa,
            self.valve_wall_force_on_liquid_N,
            self.dissipation_power_W,
            self.left_trace_area_m2,
            self.right_trace_area_m2,
            self.left_characteristic_residual_m_s,
            self.right_characteristic_residual_m_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("all pressurised-MOC stage values must be finite")
        if self.left_trace_area_m2 <= 0.0 or self.right_trace_area_m2 <= 0.0:
            raise ValueError("pressurised-MOC trace areas must be positive")
        if self.dissipation_power_W < 0.0:
            raise ValueError("a passive pressurised-MOC valve cannot create energy")
        density = WATER_DENSITY_KG_M3
        force_residual = (
            density
            * (self.right_momentum_flux_m4_s2 - self.left_momentum_flux_m4_s2)
            - self.valve_wall_force_on_liquid_N
        )
        if abs(force_residual) > _roundoff_tolerance(
            density * self.right_momentum_flux_m4_s2,
            density * self.left_momentum_flux_m4_s2,
            self.valve_wall_force_on_liquid_N,
            multiplier=4096.0,
        ):
            raise ValueError("pressurised-MOC momentum ports do not close")
        expected_power = self.signed_pressure_jump_Pa * self.shared_mass_flux_m3_s
        if not math.isclose(
            expected_power,
            self.dissipation_power_W,
            rel_tol=0.0,
            abs_tol=_roundoff_tolerance(
                expected_power,
                self.dissipation_power_W,
                multiplier=4096.0,
            ),
        ):
            raise ValueError("pressurised-MOC dissipation is not dp*Q")

    @property
    def left_elastic_volume_rate_m3_s(self) -> float:
        return float(
            self.native_shared_mass_flux_m3_s - self.shared_mass_flux_m3_s
        )

    @property
    def right_elastic_volume_rate_m3_s(self) -> float:
        return -self.left_elastic_volume_rate_m3_s

    @property
    def elastic_volume_rate_residual_m3_s(self) -> float:
        return float(
            self.left_elastic_volume_rate_m3_s
            + self.right_elastic_volume_rate_m3_s
        )


def _validated_internal_face_index(
    face_index: int,
    *,
    ncell: int,
) -> int:
    if isinstance(face_index, bool) or not isinstance(face_index, int):
        raise TypeError("face_index must be an integer")
    if not (1 <= face_index < ncell):
        raise ValueError("face_index must have one physical cell on each side")
    return face_index


def _split_momentum_donor_draining_limiter(
    mass_flux: np.ndarray,
    momentum_as_east_boundary: np.ndarray,
    momentum_as_west_boundary: np.ndarray,
    area: np.ndarray,
    *,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Case-1 donor factor to mass and both momentum ports.

    Except at the selected zero-volume face, the two momentum arrays are
    identical.  A single donor scale multiplies the shared mass flux and both
    face momenta, so limiting cannot create unequal side volume transactions.
    """

    mass = np.asarray(mass_flux, dtype=float).copy()
    momentum_east = np.asarray(momentum_as_east_boundary, dtype=float).copy()
    momentum_west = np.asarray(momentum_as_west_boundary, dtype=float).copy()
    cell_area = np.asarray(area, dtype=float)
    expected_shape = (cell_area.size + 1,)
    if (
        mass.shape != expected_shape
        or momentum_east.shape != expected_shape
        or momentum_west.shape != expected_shape
    ):
        raise ValueError("face flux arrays must have one more entry than area")
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive")

    outgoing = np.maximum(mass[1:], 0.0) + np.maximum(-mass[:-1], 0.0)
    theta = np.ones(cell_area.size)
    draining = outgoing > 0.0
    theta[draining] = np.minimum(
        1.0,
        cell_area[draining] * dx / (dt * outgoing[draining]),
    )

    face_scale = np.ones(cell_area.size + 1)
    internal_flux = mass[1:-1]
    internal_faces = np.arange(1, cell_area.size)
    donor = np.where(internal_flux >= 0.0, internal_faces - 1, internal_faces)
    face_scale[1:-1] = theta[donor]
    if mass[0] < 0.0:
        face_scale[0] = theta[0]
    if mass[-1] > 0.0:
        face_scale[-1] = theta[-1]
    return (
        mass * face_scale,
        momentum_east * face_scale,
        momentum_west * face_scale,
        face_scale,
    )


def _internal_face_wet_dry_euler_stage(
    state: WetDryState,
    *,
    dx: float,
    dt: float,
    section: CircularSection,
    face_index: int,
    callback: InternalFaceFluxCallback,
    stage_index: Literal[1, 2],
    stage_time_s: float,
    cfl: float = 0.45,
    dry_area_fraction: float = 1.0e-10,
    manning_n: float = 0.0,
    darcy_friction: float = 0.0,
    bed_slope: float = 0.0,
    left_boundary: Literal["wall", "transmissive"] = "wall",
    right_boundary: Literal["wall", "transmissive"] = "wall",
    left_ghost: tuple[float, float] | None = None,
    right_ghost: tuple[float, float] | None = None,
    left_face_flux: tuple[float, float] | None = None,
    right_face_flux: tuple[float, float] | None = None,
    interface_traction: (
        tuple[float, float]
        | tuple[float, float, Literal["left", "right"]]
        | None
    ) = None,
) -> tuple[WetDryState, InternalFaceStageRecord]:
    """Advance one Case-1 Euler stage with one optional split-momentum face."""

    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx and dt must be positive")
    if not (0.0 < cfl < 1.0):
        raise ValueError("cfl must lie in (0, 1)")
    if not callable(callback):
        raise TypeError("callback must be callable")
    time = _finite(stage_time_s, name="stage_time_s")

    area = np.asarray(state.area, dtype=float).copy()
    discharge = np.asarray(state.discharge, dtype=float).copy()
    ncell = area.size
    face = _validated_internal_face_index(face_index, ncell=ncell)

    dry_area = dry_area_fraction * section.full_area
    velocity = np.divide(
        discharge,
        area,
        out=np.zeros_like(discharge),
        where=area > dry_area,
    )
    max_speed = float(
        np.max(np.abs(velocity) + np.asarray(section.celerity(area)))
    )
    if max_speed > 0.0 and dt > cfl * dx / max_speed * (1.0 + 1.0e-12):
        raise ValueError("dt exceeds the requested central-upwind CFL limit")

    area_ext = np.empty(ncell + 2)
    discharge_ext = np.empty(ncell + 2)
    area_ext[1:-1] = area
    discharge_ext[1:-1] = discharge
    if left_ghost is None:
        area_ext[0], discharge_ext[0] = _case1_core._ghost_state(
            area[0], discharge[0], left_boundary
        )
    else:
        area_ext[0], discharge_ext[0] = map(float, left_ghost)
    if right_ghost is None:
        area_ext[-1], discharge_ext[-1] = _case1_core._ghost_state(
            area[-1], discharge[-1], right_boundary
        )
    else:
        area_ext[-1], discharge_ext[-1] = map(float, right_ghost)

    (
        area_left,
        discharge_left,
        area_right,
        discharge_right,
    ) = _case1_core._muscl_free_surface_face_states(
        area_ext,
        discharge_ext,
        section,
        dry_area,
    )
    if left_ghost is None:
        area_left[0] = area_right[0]
        discharge_left[0] = (
            -discharge_right[0]
            if left_boundary == "wall"
            else discharge_right[0]
        )
    if right_ghost is None:
        area_right[-1] = area_left[-1]
        discharge_right[-1] = (
            -discharge_left[-1]
            if right_boundary == "wall"
            else discharge_left[-1]
        )
    mass_flux, native_momentum_flux = _case1_core._central_upwind_flux(
        area_left,
        discharge_left,
        area_right,
        discharge_right,
        section,
        dry_area,
    )
    if left_face_flux is not None:
        mass_flux[0], native_momentum_flux[0] = map(float, left_face_flux)
    if right_face_flux is not None:
        mass_flux[-1], native_momentum_flux[-1] = map(float, right_face_flux)

    context = InternalFaceStageContext(
        stage_index=stage_index,
        stage_time_s=time,
        dt_s=dt,
        face_index=face,
        west_area_m2=float(area_left[face]),
        west_discharge_m3_s=float(discharge_left[face]),
        east_area_m2=float(area_right[face]),
        east_discharge_m3_s=float(discharge_right[face]),
        native_shared_mass_flux_m3_s=float(mass_flux[face]),
        native_momentum_flux_m4_s2=float(native_momentum_flux[face]),
    )
    requested_flux = callback(context)
    if requested_flux is not None and not isinstance(
        requested_flux,
        InternalFaceFluxPair,
    ):
        raise TypeError("internal-face callback must return InternalFaceFluxPair or None")
    use_native = bool(
        requested_flux is None or requested_flux.is_exact_native(context)
    )

    if use_native:
        (
            accepted_mass,
            accepted_momentum_as_east,
            accepted_momentum_as_west,
            native_face_scale,
        ) = _split_momentum_donor_draining_limiter(
            mass_flux,
            native_momentum_flux,
            native_momentum_flux,
            area,
            dx=dx,
            dt=dt,
        )
        advanced = _case1_core._central_upwind_wet_dry_euler_step(
            state,
            dx=dx,
            dt=dt,
            section=section,
            cfl=cfl,
            dry_area_fraction=dry_area_fraction,
            manning_n=manning_n,
            darcy_friction=darcy_friction,
            bed_slope=bed_slope,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            left_ghost=left_ghost,
            right_ghost=right_ghost,
            left_face_flux=left_face_flux,
            right_face_flux=right_face_flux,
            interface_traction=interface_traction,
        )
        accepted_flux = InternalFaceFluxPair(
            shared_mass_flux_m3_s=float(accepted_mass[face]),
            west_momentum_flux_m4_s2=float(
                accepted_momentum_as_east[face]
            ),
            east_momentum_flux_m4_s2=float(
                accepted_momentum_as_west[face]
            ),
        )
        return advanced, InternalFaceStageRecord(
            context=context,
            requested_flux=requested_flux,
            accepted_flux=accepted_flux,
            donor_scale=float(native_face_scale[face]),
            used_native_core_stage=True,
        )

    momentum_as_east_boundary = native_momentum_flux.copy()
    momentum_as_west_boundary = native_momentum_flux.copy()
    mass_flux[face] = requested_flux.shared_mass_flux_m3_s
    momentum_as_east_boundary[face] = (
        requested_flux.west_momentum_flux_m4_s2
    )
    momentum_as_west_boundary[face] = (
        requested_flux.east_momentum_flux_m4_s2
    )
    (
        mass_flux,
        momentum_as_east_boundary,
        momentum_as_west_boundary,
        face_scale,
    ) = _split_momentum_donor_draining_limiter(
        mass_flux,
        momentum_as_east_boundary,
        momentum_as_west_boundary,
        area,
        dx=dx,
        dt=dt,
    )

    area_new = area - dt / dx * (mass_flux[1:] - mass_flux[:-1])
    discharge_new = discharge - dt / dx * (
        momentum_as_east_boundary[1:] - momentum_as_west_boundary[:-1]
    )
    if interface_traction is not None:
        if len(interface_traction) == 2:
            interface_x, gas_head = interface_traction
            water_side = "right"
        else:
            interface_x, gas_head, water_side = interface_traction
        traction_face = int(
            np.clip(np.rint(interface_x / dx), 1, ncell - 1)
        )
        force_per_density = section.gravity * float(gas_head) * section.full_area
        if water_side == "right":
            discharge_new[traction_face] += dt / dx * force_per_density
        elif water_side == "left":
            discharge_new[traction_face - 1] -= dt / dx * force_per_density
        else:
            raise ValueError("interface traction water_side must be left or right")

    regularisation_area = max(
        1.0e-3 * section.full_area,
        10.0 * dry_area,
    )
    denominator_velocity = np.sqrt(
        area_new**4
        + np.maximum(area_new**4, regularisation_area**4)
    )
    velocity_regularised = np.divide(
        np.sqrt(2.0) * area_new * discharge_new,
        denominator_velocity,
        out=np.zeros_like(discharge_new),
        where=denominator_velocity > 0.0,
    )
    dry_front_speed_bound = 2.0 * np.sqrt(
        section.gravity * section.diameter
    )
    velocity_regularised = np.clip(
        velocity_regularised,
        -dry_front_speed_bound,
        dry_front_speed_bound,
    )
    shallow = area_new < regularisation_area
    discharge_new[shallow] = area_new[shallow] * velocity_regularised[shallow]

    wet = area_new > dry_area
    velocity_new = np.divide(
        discharge_new,
        area_new,
        out=np.zeros_like(discharge_new),
        where=wet,
    )
    hydraulic_radius = np.asarray(
        section.hydraulic_radius(np.minimum(area_new, section.full_area))
    )
    free = wet & (area_new <= section.full_area)
    full = wet & ~free
    friction_slope = np.zeros(ncell)
    if manning_n > 0.0:
        friction_slope[free] = (
            manning_n**2
            * velocity_new[free]
            * np.abs(velocity_new[free])
            / np.maximum(hydraulic_radius[free], 1.0e-12) ** (4.0 / 3.0)
        )
    if darcy_friction > 0.0:
        friction_slope[full] = (
            darcy_friction
            * velocity_new[full]
            * np.abs(velocity_new[full])
            / (2.0 * section.diameter)
        )
    discharge_new += (
        dt
        * section.gravity
        * area_new
        * (float(bed_slope) - friction_slope)
    )

    material_negative = area_new < -1.0e-12 * section.full_area
    if np.any(material_negative):
        raise FloatingPointError("wet/dry update produced a negative liquid area")
    area_new = np.maximum(area_new, 0.0)
    newly_dry = area_new <= dry_area
    area_new[newly_dry] = 0.0
    discharge_new[newly_dry] = 0.0
    accepted_flux = InternalFaceFluxPair(
        shared_mass_flux_m3_s=float(mass_flux[face]),
        west_momentum_flux_m4_s2=float(
            momentum_as_east_boundary[face]
        ),
        east_momentum_flux_m4_s2=float(
            momentum_as_west_boundary[face]
        ),
    )
    return WetDryState(area_new, discharge_new), InternalFaceStageRecord(
        context=context,
        requested_flux=requested_flux,
        accepted_flux=accepted_flux,
        donor_scale=float(face_scale[face]),
        used_native_core_stage=False,
    )


def internal_face_wet_dry_ssprk2_step(
    state: WetDryState,
    *,
    dx: float,
    dt: float,
    section: CircularSection,
    face_index: int,
    callback: InternalFaceFluxCallback | None,
    start_time_s: float,
    cfl: float = 0.45,
    dry_area_fraction: float = 1.0e-10,
    manning_n: float = 0.0,
    darcy_friction: float = 0.0,
    bed_slope: float = 0.0,
    left_boundary: Literal["wall", "transmissive"] = "wall",
    right_boundary: Literal["wall", "transmissive"] = "wall",
    left_ghost: tuple[float, float] | None = None,
    right_ghost: tuple[float, float] | None = None,
    left_face_flux: tuple[float, float] | None = None,
    right_face_flux: tuple[float, float] | None = None,
    interface_traction: (
        tuple[float, float]
        | tuple[float, float, Literal["left", "right"]]
        | None
    ) = None,
) -> InternalFaceSsprk2Result:
    """Advance the generic clean-FV branch with a stage-local face callback.

    ``callback=None`` delegates directly to the pinned Case-1 SSPRK2 routine.
    A callback is evaluated from each stage's reconstructed state and physical
    stage time.  Returning ``None`` or :meth:`InternalFaceFluxPair.native`
    delegates that Euler stage to the original core, which is the transparent
    (including a future ``K=0``) path.  No dry-start or valve law is supplied
    here; this is only conservative numerical plumbing.
    """

    _require_pinned_core()
    if not isinstance(state, WetDryState):
        raise TypeError("state must be a WetDryState")
    face = _validated_internal_face_index(
        face_index,
        ncell=np.asarray(state.area).size,
    )
    start = _finite(start_time_s, name="start_time_s")
    step = _positive_finite(dt, name="dt")

    core_keywords = dict(
        dx=dx,
        dt=step,
        section=section,
        cfl=cfl,
        dry_area_fraction=dry_area_fraction,
        manning_n=manning_n,
        darcy_friction=darcy_friction,
        bed_slope=bed_slope,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        left_ghost=left_ghost,
        right_ghost=right_ghost,
        left_face_flux=left_face_flux,
        right_face_flux=right_face_flux,
        interface_traction=interface_traction,
    )
    if callback is None:
        advanced = _case1_core.central_upwind_wet_dry_step(
            state,
            **core_keywords,
        )
        return InternalFaceSsprk2Result(
            state=advanced,
            stage_records=(),
            used_native_core_step=True,
        )
    if not callable(callback):
        raise TypeError("callback must be callable or None")

    stage_keywords = dict(core_keywords)
    stage_keywords.update(face_index=face, callback=callback)
    first, first_record = _internal_face_wet_dry_euler_stage(
        state,
        **stage_keywords,
        stage_index=1,
        stage_time_s=start,
    )
    second_euler, second_record = _internal_face_wet_dry_euler_stage(
        first,
        **stage_keywords,
        stage_index=2,
        stage_time_s=start + step,
    )

    area_initial = np.asarray(state.area, dtype=float)
    discharge_initial = np.asarray(state.discharge, dtype=float)
    area_new = 0.5 * (area_initial + second_euler.area)
    discharge_new = 0.5 * (discharge_initial + second_euler.discharge)
    dry_area = dry_area_fraction * section.full_area
    if np.any(area_new < -1.0e-12 * section.full_area):
        raise FloatingPointError("SSP-RK2 wet/dry step produced negative area")
    area_new = np.maximum(area_new, 0.0)
    newly_dry = area_new <= dry_area
    area_new[newly_dry] = 0.0
    discharge_new[newly_dry] = 0.0
    all_native = bool(
        first_record.used_native_core_stage
        and second_record.used_native_core_stage
    )
    return InternalFaceSsprk2Result(
        state=WetDryState(area_new, discharge_new),
        stage_records=(first_record, second_record),
        used_native_core_step=all_native,
    )


class LocalValvePathNotImplemented(NotImplementedError):
    """Base error for an active face whose conservative path is pending."""

    def __init__(
        self,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        super().__init__(
            "the active fixed valve was not advanced: conservative path "
            f"'{partition.regime.value}' is not implemented; "
            f"face={partition.fixed_face_index}, "
            f"shock_cut={partition.shock_cut_cell_index}, "
            f"planned_substeps={step_plan.substeps_s}"
        )


class FreeSurfaceValvePathNotImplemented(LocalValvePathNotImplemented):
    pass


class PressurisedMocValvePathNotImplemented(LocalValvePathNotImplemented):
    pass


class CleanFreeSurfaceValveTraceRejected(RuntimeError):
    """A clean-FV face lacked two resolvable wet Saint-Venant traces."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_context: InternalFaceStageContext,
        reason: str,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_context = stage_context
        self.reason = str(reason)
        super().__init__(
            "clean free-surface valve stage rejected atomically: "
            f"{self.reason}; face={partition.fixed_face_index}, "
            f"stage={stage_context.stage_index}, "
            f"stage_time={stage_context.stage_time_s:.17g}"
        )


class CleanFreeSurfaceValveDonorLimitRejected(RuntimeError):
    """The donor limiter would alter an already solved physical valve flux."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_record: InternalFaceStageRecord,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_record = stage_record
        super().__init__(
            "clean free-surface valve substep requires a smaller dt: "
            "the donor limiter would alter the physical face solution; "
            f"face={partition.fixed_face_index}, "
            f"stage={stage_record.context.stage_index}, "
            f"donor_scale={stage_record.donor_scale:.17g}"
        )


class ShockCutValveNonlinearRejected(RuntimeError):
    """The coupled valve/shock residual had no admissible passive root."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_index: int,
        stage_time_s: float,
        reason: str,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_index = int(stage_index)
        self.stage_time_s = float(stage_time_s)
        self.reason = str(reason)
        super().__init__(
            "shock-cut valve stage rejected atomically: "
            f"{self.reason}; face={partition.fixed_face_index}, "
            f"cut={partition.shock_cut_cell_index}, stage={stage_index}, "
            f"stage_time={stage_time_s:.17g}"
        )


class ShockCutValveDonorLimitRejected(RuntimeError):
    """A physical cut-valve flux would be altered by the FV donor limiter."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_solution: ShockCutValveStageSolution,
        donor_scale: float,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_solution = stage_solution
        self.donor_scale = float(donor_scale)
        super().__init__(
            "shock-cut valve substep requires a smaller dt: the donor "
            "limiter would alter the coupled physical flux; "
            f"face={partition.fixed_face_index}, "
            f"stage={stage_solution.stage_index}, "
            f"donor_scale={self.donor_scale:.17g}"
        )


class ShockCutValveCflRejected(RuntimeError):
    """A reconstructed shock/valve boundary needs a smaller FV substep."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_solution: ShockCutValveStageSolution,
        reason: str,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_solution = stage_solution
        self.reason = str(reason)
        super().__init__(
            "shock-cut valve substep requires a smaller dt: reconstructed "
            f"boundary CFL failed ({self.reason}); "
            f"face={partition.fixed_face_index}, "
            f"stage={stage_solution.stage_index}, "
            f"stage_time={stage_solution.stage_time_s:.17g}"
        )


class PressurisedMocValveStageRejected(RuntimeError):
    """A split-MOC root or its conservative elastic update is inadmissible."""

    def __init__(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_index: int,
        stage_time_s: float,
        reason: str,
    ) -> None:
        self.partition = partition
        self.step_plan = step_plan
        self.stage_index = int(stage_index)
        self.stage_time_s = float(stage_time_s)
        self.reason = str(reason)
        super().__init__(
            "pressurised split-MOC valve stage rejected atomically: "
            f"{self.reason}; face={partition.fixed_face_index}, "
            f"cut={partition.shock_cut_cell_index}, stage={stage_index}, "
            f"stage_time={stage_time_s:.17g}"
        )


@dataclass(frozen=True)
class HorizontalAdvanceResult:
    """A horizontal state plus one uncommitted local-valve transaction."""

    state: HorizontalState
    valve_transaction: IntegratedValveTransaction
    core_sha256: str = EXPECTED_CORE_SHA256
    partition: LocalFacePartition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, HorizontalState):
            raise TypeError("state must be a HorizontalState")
        if not isinstance(self.valve_transaction, IntegratedValveTransaction):
            raise TypeError(
                "valve_transaction must be an IntegratedValveTransaction"
            )
        if self.core_sha256 != EXPECTED_CORE_SHA256:
            raise ValueError("HorizontalAdvanceResult has the wrong core hash")
        tolerance = _roundoff_tolerance(
            self.state.time,
            self.valve_transaction.end_time_s,
        )
        if not math.isclose(
            float(self.state.time),
            self.valve_transaction.end_time_s,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("state time and valve transaction end time differ")


class Case1LocalValveExtension(Tosan2021HorizontalShockFit):
    """Hash-pinned Case-1 sibling with clean-FV and shock-cut valve branches."""

    def __init__(
        self,
        config: HorizontalConfig | None = None,
        *,
        local_face: FixedInternalValveSpec | None = None,
        vent_pressure_hook=None,
    ) -> None:
        _require_pinned_core()
        super().__init__(config, vent_pressure_hook=vent_pressure_hook)
        if local_face is not None and not isinstance(
            local_face,
            FixedInternalValveSpec,
        ):
            raise TypeError("local_face must be FixedInternalValveSpec or None")
        self.local_face = local_face
        if self.local_face is not None:
            self.local_face.validate_against_solver(self)

    def classify_local_face_regime(
        self,
        state: HorizontalState,
    ) -> LocalFacePartition:
        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        if not isinstance(state, HorizontalState):
            raise TypeError("state must be a HorizontalState")
        cut = int(math.floor(float(state.interface_x) / self.dx))
        face = self.local_face.mirrored_face_index
        if cut >= face + 1:
            regime = LocalValveRegime.CLEAN_FREE_SURFACE_FV
        elif cut <= face - 2:
            regime = LocalValveRegime.CLEAN_PRESSURISED_MOC
        else:
            regime = LocalValveRegime.SHOCK_CUT_CELL
        return LocalFacePartition(
            regime=regime,
            fixed_face_index=face,
            shock_cut_cell_index=cut,
            free_surface_stop_index=cut,
            pressurised_start_index=cut + 1,
            interface_x_m=float(state.interface_x),
        )

    def plan_physical_step(
        self,
        state: HorizontalState,
        dt: float,
    ) -> EventAlignedStepPlan:
        if not isinstance(state, HorizontalState):
            raise TypeError("state must be a HorizontalState")
        events: tuple[float, ...] = ()
        if self.local_face is not None:
            events = (self.local_face.opening_duration_s,)
        return plan_event_aligned_step(
            start_time_s=float(state.time),
            dt_s=dt,
            event_times_s=events,
        )

    def _circular_state_from_theta_fraction(
        self,
        theta_fraction: float,
    ) -> tuple[float, float, float, float]:
        """Return ``(A,h,c,I1)`` without a depth/area inverse iteration."""

        fraction = float(theta_fraction)
        if not (0.0 < fraction < 1.0):
            raise ValueError("circular theta fraction must lie in (0, 1)")
        theta = math.pi * fraction
        radius = 0.5 * self.config.diameter
        area = radius * radius * (
            theta - math.sin(theta) * math.cos(theta)
        )
        depth = radius * (1.0 - math.cos(theta))
        celerity = float(
            self.section.free_surface_celerity_from_depth(depth)
        )
        moment = float(self.section.hydrostatic_moment(depth))
        return float(area), float(depth), celerity, moment

    def _theta_fraction_from_area(self, area_m2: float) -> float:
        area = float(
            np.clip(
                area_m2,
                1.0e-14 * self.section.full_area,
                (1.0 - 1.0e-10) * self.section.full_area,
            )
        )
        depth = float(self.section.depth_from_area(area))
        radius = 0.5 * self.config.diameter
        cosine = float(np.clip(1.0 - depth / radius, -1.0, 1.0))
        return float(math.acos(cosine) / math.pi)

    def _shock_cut_zero_flow_theta_fraction(
        self,
        feet: dict[str, float],
    ) -> float | None:
        """Return the exact west-characteristic seed on the ``Q=0`` manifold.

        A nearly closed valve makes the two free-surface valve equations
        singular in ``(theta_L-theta_R, Q)``.  Their physical continuous limit
        is ``Q=0`` and ``theta_L=theta_R``.  This helper solves only the
        unchanged incoming west characteristic on that manifold and supplies
        a deterministic continuation seed; it changes neither equations nor
        the dimensionless residual acceptance gate.
        """

        gravity = float(self.config.gravity)
        gravity_speed = math.sqrt(gravity * self.config.diameter)
        free_foot_term = float(
            gravity * feet["free_depth"] / feet["free_celerity"]
        )

        def residual(theta_fraction: float) -> float:
            _area, depth, celerity, _moment = (
                self._circular_state_from_theta_fraction(theta_fraction)
            )
            return float(
                (
                    -feet["free_velocity"]
                    + gravity * depth / celerity
                    - free_foot_term
                    + feet["free_source"]
                )
                / gravity_speed
            )

        lower = 1.0e-8
        upper = 1.0 - 1.0e-9
        lower_residual = residual(lower)
        upper_residual = residual(upper)
        if not (
            math.isfinite(lower_residual)
            and math.isfinite(upper_residual)
            and lower_residual <= 0.0 <= upper_residual
        ):
            return None
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            if residual(middle) < 0.0:
                lower = middle
            else:
                upper = middle
        return float(0.5 * (lower + upper))

    def _pressurised_area_from_head(self, head_m: float) -> float:
        """Use the pinned MOC elastic law even for sub-crown pressure head."""

        area = float(
            self.section.full_area
            * (
                1.0
                + self.config.gravity
                * (float(head_m) - self.config.diameter)
                / self.config.wave_speed**2
            )
        )
        if not math.isfinite(area) or area <= 0.0:
            raise FloatingPointError(
                "shock-cut pressurised head gives inadmissible elastic area"
            )
        return area

    def _pressurised_moc_face_traces(
        self,
        state: HorizontalState,
        *,
        dt: float,
        partition: LocalFacePartition,
    ) -> tuple[LiquidValveTrace, LiquidValveTrace]:
        """Reconstruct the source-corrected incoming ``C+``/``C-`` feet.

        The face is cell aligned while the pinned MOC values are cell centred.
        Each characteristic foot is interpolated only from complete cells on
        its own side of the valve.  At the first real ``cut=59`` state the
        west branch has one complete elastic cell; its constant one-sided
        reconstruction is the available Case-1 datum and the conservative
        storage update below supplies the missing subcell evolution.
        """

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        if partition.regime is not LocalValveRegime.CLEAN_PRESSURISED_MOC:
            raise ValueError("split-MOC traces require the pressurised regime")
        step = _positive_finite(dt, name="dt")
        face = partition.fixed_face_index
        west_indices = np.arange(
            partition.pressurised_start_index,
            face,
            dtype=int,
        )
        east_indices = np.arange(face, self.ncell, dtype=int)
        if west_indices.size == 0 or east_indices.size == 0:
            raise ValueError("the split-MOC valve needs one cell on each side")

        area = np.asarray(state.area, dtype=float)
        discharge = np.asarray(state.discharge, dtype=float)
        full_area = float(self.section.full_area)
        gravity = float(self.config.gravity)
        acoustic = float(self.config.wave_speed)
        density = float(self.config.liquid_density)
        face_x = float(self.local_face.mirrored_face_x_m)

        def reconstruct(indices: np.ndarray, foot_x: float) -> LiquidValveTrace:
            coordinates = self.x[indices]
            if indices.size == 1:
                area_foot = float(area[indices[0]])
                discharge_foot = float(discharge[indices[0]])
            else:
                clipped_x = float(np.clip(foot_x, coordinates[0], coordinates[-1]))
                area_foot = float(np.interp(clipped_x, coordinates, area[indices]))
                discharge_foot = float(
                    np.interp(clipped_x, coordinates, discharge[indices])
                )
            if not math.isfinite(area_foot) or area_foot <= 0.0:
                raise FloatingPointError(
                    "a pressurised-MOC characteristic foot has non-positive area"
                )
            velocity = float(discharge_foot / area_foot)
            head = float(
                self.config.diameter
                + acoustic**2
                / gravity
                * (area_foot / full_area - 1.0)
            )
            friction = float(
                self.config.darcy_friction
                * velocity
                * abs(velocity)
                / (2.0 * self.config.diameter)
            )
            source_corrected_velocity = float(
                velocity
                + gravity * (self.config.bed_slope - friction) * step
            )
            return LiquidValveTrace(
                area_m2=area_foot,
                velocity_m_s=source_corrected_velocity,
                gauge_pressure_Pa=density * gravity * (
                    head - self.config.diameter
                ),
                wave_speed_m_s=acoustic,
                density_kg_m3=density,
            )

        left = reconstruct(
            west_indices,
            face_x - acoustic * step,
        )
        right = reconstruct(
            east_indices,
            face_x + acoustic * step,
        )
        return left, right

    def _solve_pressurised_moc_stage(
        self,
        state: HorizontalState,
        *,
        dt: float,
        stage_index: Literal[1, 2],
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
    ) -> PressurisedMocValveStageSolution:
        """Solve one passive split-MOC stage about the exact ``K=0`` datum."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        try:
            left, right = self._pressurised_moc_face_traces(
                state,
                dt=dt,
                partition=partition,
            )
            native = solve_passive_pressurised_moc_valve(
                left,
                right,
                time_s=OPENING_DURATION_S,
                valve_flow_area_m2=self.local_face.valve_flow_area_m2,
                nominal_pipe_area_m2=self.section.full_area,
            )
            solution = solve_passive_pressurised_moc_valve(
                left,
                right,
                time_s=float(state.time),
                valve_flow_area_m2=self.local_face.valve_flow_area_m2,
                nominal_pipe_area_m2=self.section.full_area,
            )
        except (PressurisedMocValveNoRootError, FloatingPointError, ValueError) as exc:
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason=str(exc),
            ) from exc

        density = float(self.config.liquid_density)
        gravity = float(self.config.gravity)
        diameter = float(self.config.diameter)
        return PressurisedMocValveStageSolution(
            stage_index=stage_index,
            stage_time_s=float(state.time),
            fixed_face_index=partition.fixed_face_index,
            shock_cut_cell_index=partition.shock_cut_cell_index,
            control_regime=solution.control_regime,
            native_shared_mass_flux_m3_s=(
                native.volume_flow_left_to_right_m3_s
            ),
            shared_mass_flux_m3_s=(
                solution.volume_flow_left_to_right_m3_s
            ),
            left_gauge_pressure_Pa=solution.left_gauge_pressure_Pa,
            right_gauge_pressure_Pa=solution.right_gauge_pressure_Pa,
            left_head_m=(
                diameter + solution.left_gauge_pressure_Pa / (density * gravity)
            ),
            right_head_m=(
                diameter + solution.right_gauge_pressure_Pa / (density * gravity)
            ),
            left_momentum_flux_m4_s2=(
                solution.left_momentum_flow_N / density
            ),
            right_momentum_flux_m4_s2=(
                solution.right_momentum_flow_N / density
            ),
            native_momentum_flux_m4_s2=(
                native.left_momentum_flow_N / density
            ),
            signed_pressure_jump_Pa=solution.signed_pressure_jump_Pa,
            valve_wall_force_on_liquid_N=(
                solution.valve_wall_force_on_liquid_N
            ),
            dissipation_power_W=solution.dissipation_power_W,
            left_trace_area_m2=left.area_m2,
            right_trace_area_m2=right.area_m2,
            left_incoming_characteristic_count=(
                solution.left_incoming_characteristic_count
            ),
            right_incoming_characteristic_count=(
                solution.right_incoming_characteristic_count
            ),
            left_characteristic_residual_m_s=(
                solution.left_characteristic_residual_m_s
            ),
            right_characteristic_residual_m_s=(
                solution.right_characteristic_residual_m_s
            ),
        )

    def _pressurised_moc_euler_stage(
        self,
        state: HorizontalState,
        *,
        dt: float,
        solution: PressurisedMocValveStageSolution,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        external_pressure_abs: float | None,
    ) -> HorizontalState:
        """Replace the pinned lossless MOC face by one conservative two-port."""

        native = super()._step_once(state, dt, external_pressure_abs)
        area = np.asarray(native.area, dtype=float).copy()
        discharge = np.asarray(native.discharge, dtype=float).copy()
        west = partition.fixed_face_index - 1
        east = partition.fixed_face_index
        pair_area_before = float(area[west] + area[east])
        area_increment = float(
            dt * solution.left_elastic_volume_rate_m3_s / self.dx
        )
        west_area = float(area[west] + area_increment)
        east_area = float(pair_area_before - west_area)
        if (
            not math.isfinite(west_area)
            or not math.isfinite(east_area)
            or west_area <= 0.0
            or east_area <= 0.0
        ):
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=solution.stage_index,
                stage_time_s=solution.stage_time_s,
                reason=(
                    "the equal-and-opposite elastic storage update would make "
                    "an adjacent MOC cell non-positive"
                ),
            )
        area[west] = west_area
        area[east] = east_area

        density = float(self.config.liquid_density)
        flux_scale = float(dt / self.dx)
        discharge[west] -= flux_scale * (
            solution.left_momentum_flux_m4_s2
            - solution.native_momentum_flux_m4_s2
        )
        discharge[east] += flux_scale * (
            solution.right_momentum_flux_m4_s2
            - solution.native_momentum_flux_m4_s2
        )
        if np.any(~np.isfinite(area)) or np.any(~np.isfinite(discharge)):
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=solution.stage_index,
                stage_time_s=solution.stage_time_s,
                reason="the split-MOC face correction produced a non-finite state",
            )
        pair_storage_residual = float(
            (area[west] - native.area[west])
            + (area[east] - native.area[east])
        )
        if abs(pair_storage_residual) > _roundoff_tolerance(
            area[west],
            area[east],
            native.area[west],
            native.area[east],
            multiplier=1024.0,
        ):
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=solution.stage_index,
                stage_time_s=solution.stage_time_s,
                reason="the two elastic storage corrections do not close",
            )
        momentum_rate_residual = float(
            density
            * self.dx
            / dt
            * (
                (discharge[west] - native.discharge[west])
                + (discharge[east] - native.discharge[east])
            )
            - solution.valve_wall_force_on_liquid_N
        )
        if abs(momentum_rate_residual) > _roundoff_tolerance(
            solution.valve_wall_force_on_liquid_N,
            density * self.dx / dt * discharge[west],
            density * self.dx / dt * discharge[east],
            multiplier=8192.0,
        ):
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=solution.stage_index,
                stage_time_s=solution.stage_time_s,
                reason="the two MOC momentum ports do not reproduce the wall force",
            )
        return HorizontalState(
            time=native.time,
            area=area,
            discharge=discharge,
            gas=native.gas,
            air_pressure_abs=native.air_pressure_abs,
            interface_x=native.interface_x,
            interface_speed=native.interface_speed,
            interface_free_surface_depth=native.interface_free_surface_depth,
            interface_free_surface_velocity=native.interface_free_surface_velocity,
            interface_pressurised_head=native.interface_pressurised_head,
            interface_pressurised_velocity=native.interface_pressurised_velocity,
            interface_residual_linf=native.interface_residual_linf,
            wetting_front_x=native.wetting_front_x,
            vented=native.vented,
            nonlinear_converged=native.nonlinear_converged,
            liquid_volume_residual=native.liquid_volume_residual,
            cumulative_liquid_volume_residual=(
                native.cumulative_liquid_volume_residual
            ),
        )

    def _shock_cut_characteristic_feet(
        self,
        state: HorizontalState,
        *,
        dt: float,
        free_foot_index_override: int | None = None,
        pressure_foot_index_override: int | None = None,
    ) -> dict[str, float]:
        """Reproduce the pinned Case-1 incoming feet without using the cut cell."""

        free_index, pressurised_index = self._interface_cells(state)
        free_foot_index = (
            max(0, free_index - 1)
            if free_foot_index_override is None
            else int(free_foot_index_override)
        )
        if not 0 <= free_foot_index < self.ncell:
            raise ValueError("free-foot index lies outside the horizontal grid")
        free_foot_area = min(
            max(float(state.area[free_foot_index]), 0.0),
            self.section.full_area * (1.0 - 1.0e-9),
        )
        if free_foot_area >= self.dry_gate_area:
            free_depth = float(self.section.depth_from_area(free_foot_area))
            free_velocity = float(
                state.discharge[free_foot_index] / free_foot_area
            )
        else:
            free_depth = float(self.dry_gate_depth)
            free_velocity = float(self.dry_gate_velocity)
            free_foot_area = float(self.dry_gate_area)
        free_celerity = float(
            self.section.free_surface_celerity_from_depth(free_depth)
        )
        free_radius = float(self.section.hydraulic_radius(free_foot_area))
        free_friction = float(
            self.config.manning_n**2
            * free_velocity
            * abs(free_velocity)
            / max(free_radius, 1.0e-12) ** (4.0 / 3.0)
        )

        pressure_foot_index = (
            min(pressurised_index + 1, self.ncell - 1)
            if pressure_foot_index_override is None
            else int(pressure_foot_index_override)
        )
        if not 0 <= pressure_foot_index < self.ncell:
            raise ValueError("pressure-foot index lies outside the horizontal grid")
        pressure_area = max(
            float(state.area[pressure_foot_index]),
            self.section.full_area,
        )
        pressure_velocity = float(
            state.discharge[pressure_foot_index] / pressure_area
        )
        pressure_head = float(
            self.config.diameter
            + self.config.wave_speed**2
            / self.config.gravity
            * (pressure_area / self.section.full_area - 1.0)
        )
        pressure_friction = float(
            self.config.darcy_friction
            * pressure_velocity
            * abs(pressure_velocity)
            / (2.0 * self.config.diameter)
        )
        gas_head = float(
            (state.air_pressure_abs - self.config.atmospheric_pressure)
            / (self.config.liquid_density * self.config.gravity)
        )
        return {
            "free_area": free_foot_area,
            "free_depth": free_depth,
            "free_velocity": free_velocity,
            "free_celerity": free_celerity,
            "free_source": self.config.gravity
            * dt
            * (free_friction - self.config.bed_slope),
            "pressure_velocity": pressure_velocity,
            "pressure_head": pressure_head,
            "pressure_source": self.config.gravity
            * dt
            * (pressure_friction - self.config.bed_slope),
            "gas_head": gas_head,
        }

    def _shock_cut_face_contact_feet(
        self,
        state: HorizontalState,
        *,
        dt: float,
    ) -> dict[str, float]:
        """Return the two complete incoming feet at exact face contact.

        Cell ``face-1`` is the reconstructed free-surface boundary cell and
        cell ``face`` is the reconstructed pressurised boundary cell.  The
        characteristic feet therefore lie at ``face-2`` and ``face+1``.
        This construction is independent of ``floor(interface_x/dx)`` and is
        the common topological limit used by both release orientations.
        """

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        face = self.local_face.mirrored_face_index
        return self._shock_cut_characteristic_feet(
            state,
            dt=dt,
            free_foot_index_override=face - 2,
            pressure_foot_index_override=face + 1,
        )

    def _solve_shock_cut_face_contact_root(
        self,
        state: HorizontalState,
        *,
        feet: dict[str, float],
        coefficient: float,
    ) -> _ShockCutFaceContactRoot:
        """Solve the stationary five-equation contact Riemann state.

        Unknowns are the two circular free-surface traces, shared valve flow,
        and the pressurised velocity/head.  The equations are the incoming
        free-surface characteristic, incoming water-hammer characteristic,
        stationary-interface mass and momentum, and the passive Forchheimer
        head jump.  No interface-speed equation is replaced or relaxed: the
        contact topology supplies the exact kinematic constraint ``w=0``.
        """

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        full_area = float(self.section.full_area)
        gravity = float(self.config.gravity)
        density = float(self.config.liquid_density)
        diameter = float(self.config.diameter)
        acoustic = float(self.config.wave_speed)
        gravity_speed = math.sqrt(gravity * diameter)
        free_foot_term = float(
            gravity * feet["free_depth"] / feet["free_celerity"]
        )

        def residual(x: np.ndarray) -> np.ndarray:
            (
                theta_left,
                theta_right,
                q_scaled,
                pressure_velocity_scaled,
                pressure_head_scaled,
            ) = x
            area_left, depth_left, celerity_left, _ = (
                self._circular_state_from_theta_fraction(theta_left)
            )
            area_right, depth_right, _, moment_right = (
                self._circular_state_from_theta_fraction(theta_right)
            )
            volume_flow = q_scaled * full_area * gravity_speed
            velocity_left = volume_flow / area_left
            velocity_right = volume_flow / area_right
            pressure_velocity = pressure_velocity_scaled * gravity_speed
            pressure_head = pressure_head_scaled * diameter
            upstream_area = area_right if volume_flow < 0.0 else area_left
            valve_velocity = volume_flow / upstream_area
            pressure_jump = float(
                0.5
                * density
                * coefficient
                * abs(valve_velocity)
                * valve_velocity
            )
            mass_jump = full_area * pressure_velocity - volume_flow
            momentum_jump = (
                full_area * pressure_velocity * pressure_velocity
                + gravity
                * full_area
                * (pressure_head - 0.5 * diameter)
                - volume_flow * volume_flow / area_right
                - gravity * moment_right
                - gravity * feet["gas_head"] * full_area
            )
            energy_jump = (
                depth_left
                + velocity_left * velocity_left / (2.0 * gravity)
                - depth_right
                - velocity_right * velocity_right / (2.0 * gravity)
                - pressure_jump / (density * gravity)
            )
            return np.array(
                [
                    (
                        velocity_left
                        - feet["free_velocity"]
                        + gravity * depth_left / celerity_left
                        - free_foot_term
                        + feet["free_source"]
                    )
                    / gravity_speed,
                    (
                        pressure_velocity
                        - feet["pressure_velocity"]
                        - gravity
                        / acoustic
                        * (pressure_head - feet["pressure_head"])
                        + feet["pressure_source"]
                    )
                    / gravity_speed,
                    mass_jump / (full_area * gravity_speed),
                    momentum_jump
                    / (full_area * gravity_speed * gravity_speed),
                    energy_jump / diameter,
                ],
                dtype=float,
            )

        face = self.local_face.mirrored_face_index
        left_area_guess = float(
            np.clip(
                state.area[face - 1],
                1.0e-12 * full_area,
                (1.0 - 1.0e-10) * full_area,
            )
        )
        interface_area_guess = float(
            self.section.area_from_depth(
                np.clip(
                    state.interface_free_surface_depth,
                    1.0e-12 * diameter,
                    (1.0 - 1.0e-10) * diameter,
                )
            )
        )
        theta_left_guess = self._theta_fraction_from_area(left_area_guess)
        theta_right_guess = self._theta_fraction_from_area(
            interface_area_guess
        )
        core_interface = self._interface_solution(state, dt=0.0)
        pressure_velocity_guess = float(
            core_interface.pressurised_velocity / gravity_speed
        )
        pressure_head_guess = float(
            core_interface.pressurised_head / diameter
        )
        native_flow = float(
            interface_area_guess
            * (
                core_interface.free_surface_velocity
                if core_interface.free_surface_velocity is not None
                else state.interface_free_surface_velocity
            )
        )
        flow_guess = float(
            native_flow
            / (full_area * gravity_speed)
            / max(1.0, math.sqrt(coefficient))
        )
        common_theta = 0.5 * (theta_left_guess + theta_right_guess)
        foot_theta = self._theta_fraction_from_area(feet["free_area"])
        guesses = (
            np.array(
                (
                    theta_left_guess,
                    theta_right_guess,
                    flow_guess,
                    pressure_velocity_guess,
                    pressure_head_guess,
                )
            ),
            np.array(
                (
                    common_theta,
                    common_theta,
                    flow_guess,
                    pressure_velocity_guess,
                    pressure_head_guess,
                )
            ),
            np.array(
                (
                    foot_theta,
                    foot_theta,
                    0.0,
                    pressure_velocity_guess,
                    pressure_head_guess,
                )
            ),
        )
        lower = np.array((1.0e-8, 1.0e-8, -5.0, -80.0, -10.0))
        upper = np.array(
            (1.0 - 1.0e-9, 1.0 - 1.0e-9, 5.0, 80.0, 1.0e4)
        )
        best = None
        best_norm = math.inf
        for guess in guesses:
            candidate = least_squares(
                residual,
                np.minimum(np.maximum(guess, lower), upper),
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=240,
            )
            norm = float(np.max(np.abs(residual(candidate.x))))
            if norm < best_norm:
                best = candidate
                best_norm = norm
            if norm <= 1.0e-6:
                break
        if best is not None and best_norm > 1.0e-6:
            continued = least_squares(
                residual,
                np.minimum(np.maximum(best.x, lower), upper),
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=720,
            )
            continued_norm = float(
                np.max(np.abs(residual(continued.x)))
            )
            if continued_norm < best_norm:
                best = continued
                best_norm = continued_norm
        if best is None or best_norm > 2.0e-6:
            raise RuntimeError(
                "stationary shock/valve contact root has dimensionless "
                f"residual Linf={best_norm:.6e}"
            )
        return _ShockCutFaceContactRoot(
            theta_left=float(best.x[0]),
            theta_right=float(best.x[1]),
            q_scaled=float(best.x[2]),
            pressure_velocity_scaled=float(best.x[3]),
            pressure_head_scaled=float(best.x[4]),
            residual_linf=best_norm,
            evaluation_count=int(best.nfev),
        )

    def _solve_shock_cut_dry_face_injection(
        self,
        state: HorizontalState,
        *,
        feet: dict[str, float],
        coefficient: float,
        stage_index: Literal[1, 2],
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
    ) -> ShockCutValveStageSolution:
        """Close the finite-opening Riemann problem at a dry valve face.

        Before a complete Case-1 free-surface characteristic foot exists west
        of the valve, a two-liquid-trace contact is not defined.  The receiving
        dry reach instead supplies the circular critical trace ``u=-c(h)``.
        The common discharge continues into the pressurised trace, whose head
        is fixed by the incoming ``C-`` characteristic.  The one remaining
        mixed-energy equation contains the unchanged passive Forchheimer loss.
        Its wall reaction absorbs the momentum mismatch while the pressure/free
        interface stays at the valve and the separate wetting front advances.

        As ``K -> infinity`` the unique root tends to ``h,Q -> 0``; no liquid
        film or seed is inserted.  Once the complete free foot is available,
        the caller leaves this one-sided closure and uses the ordinary
        release/contact complementarity problem.
        """

        full_area = float(self.section.full_area)
        gravity = float(self.config.gravity)
        density = float(self.config.liquid_density)
        diameter = float(self.config.diameter)
        acoustic = float(self.config.wave_speed)
        gravity_speed = math.sqrt(gravity * diameter)

        def trace(depth: float) -> tuple[float, ...]:
            area = float(self.section.area_from_depth(depth))
            celerity = float(
                self.section.free_surface_celerity_from_depth(depth)
            )
            free_velocity = -celerity
            volume_flow = float(area * free_velocity)
            pressure_velocity = float(volume_flow / full_area)
            pressure_head = float(
                feet["pressure_head"]
                + acoustic
                / gravity
                * (
                    pressure_velocity
                    - feet["pressure_velocity"]
                    + feet["pressure_source"]
                )
            )
            pressure_jump = float(
                0.5
                * density
                * coefficient
                * abs(pressure_velocity)
                * pressure_velocity
            )
            energy_residual = float(
                feet["gas_head"]
                + depth
                + free_velocity * free_velocity / (2.0 * gravity)
                - pressure_head
                - pressure_velocity * pressure_velocity / (2.0 * gravity)
                - pressure_jump / (density * gravity)
            )
            return (
                energy_residual,
                area,
                celerity,
                free_velocity,
                volume_flow,
                pressure_velocity,
                pressure_head,
                pressure_jump,
            )

        lower = 1.0e-12 * diameter
        upper = (1.0 - 1.0e-10) * diameter
        lower_residual = float(trace(lower)[0])
        upper_residual = float(trace(upper)[0])
        if not (
            math.isfinite(lower_residual)
            and math.isfinite(upper_residual)
            and lower_residual <= 0.0 <= upper_residual
        ):
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason=(
                    "dry-face critical injection has no bracketed passive root: "
                    f"lower={lower_residual:.6e}, upper={upper_residual:.6e}"
                ),
            )
        evaluation_count = 2
        for _ in range(120):
            middle = 0.5 * (lower + upper)
            middle_residual = float(trace(middle)[0])
            evaluation_count += 1
            if middle_residual < 0.0:
                lower = middle
            else:
                upper = middle
        depth = float(0.5 * (lower + upper))
        (
            energy_residual,
            area,
            celerity,
            free_velocity,
            volume_flow,
            pressure_velocity,
            pressure_head,
            pressure_jump,
        ) = trace(depth)
        moment = float(self.section.hydrostatic_moment(depth))
        flux_left = float(
            volume_flow * volume_flow / area
            + gravity * moment
            + gravity * feet["gas_head"] * full_area
        )
        flux_right = float(
            volume_flow * volume_flow / full_area
            + gravity * full_area * (pressure_head - 0.5 * diameter)
        )
        wall_force = float(density * (flux_right - flux_left))
        dissipation = float(pressure_jump * volume_flow)
        if dissipation < -_roundoff_tolerance(dissipation):
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason="dry-face critical injection is not passive",
            )
        characteristic_residual = float(
            (
                pressure_velocity
                - feet["pressure_velocity"]
                - gravity
                / acoustic
                * (pressure_head - feet["pressure_head"])
                + feet["pressure_source"]
            )
            / gravity_speed
        )
        critical_residual = float(
            (free_velocity + celerity) / gravity_speed
        )
        mass_residual = float(
            (full_area * pressure_velocity - volume_flow)
            / (full_area * gravity_speed)
        )
        residual_linf = float(
            max(
                abs(energy_residual / diameter),
                abs(characteristic_residual),
                abs(critical_residual),
                abs(mass_residual),
            )
        )
        if residual_linf > 2.0e-6:
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason=(
                    "dry-face critical injection residual Linf="
                    f"{residual_linf:.6e}"
                ),
            )
        return ShockCutValveStageSolution(
            stage_index=stage_index,
            stage_time_s=float(state.time),
            orientation=ShockCutValveOrientation.DRY_FACE_INJECTION,
            fixed_face_index=partition.fixed_face_index,
            shock_cut_cell_index=partition.shock_cut_cell_index,
            shared_mass_flux_m3_s=volume_flow,
            left_momentum_flux_m4_s2=flux_left,
            right_momentum_flux_m4_s2=flux_right,
            signed_pressure_jump_Pa=pressure_jump,
            valve_wall_force_on_liquid_N=wall_force,
            dissipation_power_W=max(0.0, dissipation),
            valve_left_area_m2=area,
            valve_right_area_m2=full_area,
            interface_free_surface_area_m2=area,
            interface_free_surface_discharge_m3_s=volume_flow,
            interface_pressurised_velocity_m_s=pressure_velocity,
            interface_pressurised_head_m=pressure_head,
            interface_speed_m_s=0.0,
            valve_left_head_m=None,
            valve_right_head_m=None,
            nonlinear_residual_linf=residual_linf,
            nonlinear_evaluation_count=evaluation_count,
        )

    def _solve_shock_cut_stage(
        self,
        state: HorizontalState,
        *,
        dt: float,
        stage_index: Literal[1, 2],
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        orientation_override: ShockCutValveOrientation | None = None,
        characteristic_feet_override: dict[str, float] | None = None,
        contact_root_override: _ShockCutFaceContactRoot | None = None,
        native_release_probe: bool = False,
        allow_dry_face_release: bool = True,
    ) -> ShockCutValveStageSolution:
        """Close one fitted cut element containing the fixed valve.

        ``cut=60`` has a pressurised subcell between the moving interface and
        face 61.  Its two full-pipe valve traces are joined to the reflected
        Case-1 Tosan interface.  ``cut=61`` has a free-surface subcell between
        face 61 and the fitted shock; both valve traces and the shock state are
        solved simultaneously.  No cut-cell average is used as a
        characteristic foot.
        """

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        opening = shared_opening_state(float(state.time))
        full_area = float(self.section.full_area)
        gravity = float(self.config.gravity)
        density = float(self.config.liquid_density)
        diameter = float(self.config.diameter)
        acoustic = float(self.config.wave_speed)
        gravity_speed = math.sqrt(gravity * diameter)
        face_x = float(self.local_face.mirrored_face_x_m)
        if orientation_override is not None and not isinstance(
            orientation_override,
            ShockCutValveOrientation,
        ):
            raise TypeError("orientation_override must be a shock-cut orientation")
        if not isinstance(native_release_probe, bool):
            raise TypeError("native_release_probe must be bool")
        if not isinstance(allow_dry_face_release, bool):
            raise TypeError("allow_dry_face_release must be bool")
        exact_face_contact = float(state.interface_x) == face_x
        feet = (
            dict(characteristic_feet_override)
            if characteristic_feet_override is not None
            else (
                self._shock_cut_face_contact_feet(state, dt=dt)
                if exact_face_contact
                else self._shock_cut_characteristic_feet(state, dt=dt)
            )
        )
        orientation = orientation_override or (
            ShockCutValveOrientation.FACE_CONTACT
            if exact_face_contact
            else (
                ShockCutValveOrientation.PRESSURISED_TOUCH
                if float(state.interface_x) < face_x
                else ShockCutValveOrientation.FREE_SURFACE_TOUCH
            )
        )
        coefficient = float(opening.loss_coefficient)
        free_foot_term = float(
            gravity * feet["free_depth"] / feet["free_celerity"]
        )

        def free_velocity(depth: float, celerity: float) -> float:
            return float(
                feet["free_velocity"]
                - gravity * depth / celerity
                + free_foot_term
                - feet["free_source"]
            )

        if orientation is ShockCutValveOrientation.FACE_CONTACT:
            face = self.local_face.mirrored_face_index
            free_foot_area = float(state.area[face - 2])
            if free_foot_area < self.dry_gate_area:
                # The analytical Case-1 dry-gate trace is also the authorized
                # K=0 continuation datum for detachment into the FREE compound
                # topology.  Probe that branch even before a complete stored
                # foot exists; otherwise the attached sonic solution would be
                # held until K=0 and create a finite state jump at 0.20 s.
                free_release = None
                if allow_dry_face_release:
                    try:
                        free_release = self._solve_shock_cut_stage(
                            state,
                            dt=dt,
                            stage_index=stage_index,
                            partition=partition,
                            step_plan=step_plan,
                            orientation_override=(
                                ShockCutValveOrientation.FREE_SURFACE_TOUCH
                            ),
                            characteristic_feet_override=feet,
                            native_release_probe=True,
                        )
                    except ShockCutValveNonlinearRejected:
                        free_release = None
                if free_release is not None:
                    release_speed = float(free_release.interface_speed_m_s)
                    release_flow = float(free_release.shared_mass_flux_m3_s)
                    release_power = float(
                        free_release.signed_pressure_jump_Pa * release_flow
                    )
                    if (
                        0.0 <= release_speed <= self.case_b_entropy_speed_bound
                        and release_flow < 0.0
                        and release_power >= 0.0
                    ):
                        return free_release
                return self._solve_shock_cut_dry_face_injection(
                    state,
                    feet=feet,
                    coefficient=coefficient,
                    stage_index=stage_index,
                    partition=partition,
                    step_plan=step_plan,
                )

        contact_root = contact_root_override

        if orientation is ShockCutValveOrientation.FACE_CONTACT:
            # A geometric equality ``interface_x == face_x`` is only a
            # candidate contact.  It is also the exact topological limit of
            # the two one-sided shock fits.  Solve those entropy limits first;
            # requiring the stationary five-equation root here used to make
            # the initially dry valve face falsely behave as a liquid-liquid
            # contact and no amount of time-step bisection could cure it.
            left_limit = self._solve_shock_cut_stage(
                state,
                dt=dt,
                stage_index=stage_index,
                partition=partition,
                step_plan=step_plan,
                orientation_override=(
                    ShockCutValveOrientation.PRESSURISED_TOUCH
                ),
                characteristic_feet_override=feet,
            )
            right_limit = self._solve_shock_cut_stage(
                state,
                dt=dt,
                stage_index=stage_index,
                partition=partition,
                step_plan=step_plan,
                orientation_override=(
                    ShockCutValveOrientation.FREE_SURFACE_TOUCH
                ),
                characteristic_feet_override=feet,
            )
            left_nonpenetration = float(left_limit.interface_speed_m_s)
            right_nonpenetration = float(-right_limit.interface_speed_m_s)

            if left_nonpenetration < 0.0 and right_nonpenetration < 0.0:
                raise ShockCutValveNonlinearRejected(
                    partition=partition,
                    step_plan=step_plan,
                    stage_index=stage_index,
                    stage_time_s=float(state.time),
                    reason=(
                        "face contact has two outward entropy releases: "
                        f"left={left_nonpenetration:.6e}, "
                        f"right={right_nonpenetration:.6e}"
                    ),
                )
            if left_nonpenetration < 0.0:
                return replace(
                    left_limit,
                    left_nonpenetration_residual_m_s=left_nonpenetration,
                    right_nonpenetration_residual_m_s=right_nonpenetration,
                )
            if right_nonpenetration < 0.0:
                return replace(
                    right_limit,
                    left_nonpenetration_residual_m_s=left_nonpenetration,
                    right_nonpenetration_residual_m_s=right_nonpenetration,
                )

            # Both unconstrained limits point into the face, so the
            # complementarity conditions now require the stationary contact
            # root.  The root is deliberately solved only after that physical
            # decision; root failure is not used to choose a release branch.
            if contact_root is None:
                try:
                    contact_root = self._solve_shock_cut_face_contact_root(
                        state,
                        feet=feet,
                        coefficient=coefficient,
                    )
                except RuntimeError as exc:
                    raise ShockCutValveNonlinearRejected(
                        partition=partition,
                        step_plan=step_plan,
                        stage_index=stage_index,
                        stage_time_s=float(state.time),
                        reason=str(exc),
                    ) from exc
            area_left, _, _, moment_left = (
                self._circular_state_from_theta_fraction(
                    contact_root.theta_left
                )
            )
            area_right, _, _, moment_right = (
                self._circular_state_from_theta_fraction(
                    contact_root.theta_right
                )
            )
            volume_flow = float(
                contact_root.q_scaled * full_area * gravity_speed
            )
            pressure_velocity = float(
                contact_root.pressure_velocity_scaled * gravity_speed
            )
            pressure_head = float(
                contact_root.pressure_head_scaled * diameter
            )
            upstream_area = area_right if volume_flow < 0.0 else area_left
            valve_velocity = volume_flow / upstream_area
            pressure_jump = float(
                0.5
                * density
                * coefficient
                * abs(valve_velocity)
                * valve_velocity
            )
            flux_left = float(
                volume_flow * volume_flow / area_left
                + gravity * moment_left
            )
            flux_right = float(
                volume_flow * volume_flow / area_right
                + gravity * moment_right
            )
            wall_force = float(density * (flux_right - flux_left))
            dissipation = float(pressure_jump * volume_flow)
            if dissipation < -_roundoff_tolerance(dissipation):
                raise ShockCutValveNonlinearRejected(
                    partition=partition,
                    step_plan=step_plan,
                    stage_index=stage_index,
                    stage_time_s=float(state.time),
                    reason="stationary face-contact root is not passive",
                )
            return ShockCutValveStageSolution(
                stage_index=stage_index,
                stage_time_s=float(state.time),
                orientation=ShockCutValveOrientation.FACE_CONTACT,
                fixed_face_index=partition.fixed_face_index,
                shock_cut_cell_index=partition.shock_cut_cell_index,
                shared_mass_flux_m3_s=volume_flow,
                left_momentum_flux_m4_s2=flux_left,
                right_momentum_flux_m4_s2=flux_right,
                signed_pressure_jump_Pa=pressure_jump,
                valve_wall_force_on_liquid_N=wall_force,
                dissipation_power_W=max(0.0, dissipation),
                valve_left_area_m2=area_left,
                valve_right_area_m2=area_right,
                interface_free_surface_area_m2=area_right,
                interface_free_surface_discharge_m3_s=volume_flow,
                interface_pressurised_velocity_m_s=pressure_velocity,
                interface_pressurised_head_m=pressure_head,
                interface_speed_m_s=0.0,
                valve_left_head_m=None,
                valve_right_head_m=None,
                nonlinear_residual_linf=contact_root.residual_linf,
                nonlinear_evaluation_count=contact_root.evaluation_count,
                left_nonpenetration_residual_m_s=left_nonpenetration,
                right_nonpenetration_residual_m_s=right_nonpenetration,
            )

        if orientation is ShockCutValveOrientation.PRESSURISED_TOUCH:
            denominator = max(
                full_area - feet["free_area"],
                1.0e-12 * full_area,
            )
            shock_speed = float(
                (
                    full_area * feet["pressure_velocity"]
                    - feet["free_area"] * feet["free_velocity"]
                )
                / denominator
            )
            shock_speed = min(shock_speed, self.case_b_entropy_speed_bound)

            def residual(x: np.ndarray) -> np.ndarray:
                theta_fraction, q_scaled, left_head_scaled, right_head_scaled = x
                area_fs, depth_fs, celerity_fs, moment_fs = (
                    self._circular_state_from_theta_fraction(theta_fraction)
                )
                velocity_fs = free_velocity(depth_fs, celerity_fs)
                volume_flow = q_scaled * full_area * gravity_speed
                velocity_full = volume_flow / full_area
                left_head = left_head_scaled * diameter
                right_head = right_head_scaled * diameter
                pressure_jump = float(
                    0.5
                    * density
                    * coefficient
                    * abs(velocity_full)
                    * velocity_full
                )
                mass_jump = (
                    volume_flow
                    - area_fs * velocity_fs
                    - shock_speed * (full_area - area_fs)
                )
                moving_momentum = volume_flow - area_fs * velocity_fs
                momentum_jump = (
                    volume_flow * volume_flow / full_area
                    + gravity * full_area * (left_head - 0.5 * diameter)
                    - area_fs * velocity_fs * velocity_fs
                    - gravity * moment_fs
                    - gravity * feet["gas_head"] * full_area
                    - shock_speed * moving_momentum
                )
                return np.array(
                    [
                        (
                            velocity_full
                            - feet["pressure_velocity"]
                            - gravity
                            / acoustic
                            * (right_head - feet["pressure_head"])
                            + feet["pressure_source"]
                        )
                        / gravity_speed,
                        (
                            gravity * (left_head - right_head)
                            - pressure_jump / density
                        )
                        / (gravity * diameter),
                        mass_jump / (full_area * gravity_speed),
                        momentum_jump
                        / (full_area * gravity_speed * gravity_speed),
                    ],
                    dtype=float,
                )

            interface_area_guess = float(
                self.section.area_from_depth(
                    np.clip(
                        state.interface_free_surface_depth,
                        1.0e-12 * diameter,
                        (1.0 - 1.0e-10) * diameter,
                    )
                )
            )
            theta_guess = self._theta_fraction_from_area(interface_area_guess)
            core_interface = self._interface_solution(state, dt=dt)
            native_flow = float(
                interface_area_guess
                * (
                    core_interface.free_surface_velocity
                    if core_interface.free_surface_velocity is not None
                    else state.interface_free_surface_velocity
                )
            )
            flow_guess = float(
                native_flow
                / (full_area * gravity_speed)
                / max(1.0, math.sqrt(coefficient))
            )
            head_left_guess = float(
                max(
                    0.5 * diameter + feet["gas_head"],
                    state.interface_pressurised_head,
                )
                / diameter
            )
            head_right_guess = float(feet["pressure_head"] / diameter)
            guesses = (
                np.array(
                    [
                        theta_guess,
                        flow_guess,
                        head_left_guess,
                        head_right_guess,
                    ]
                ),
                np.array(
                    [
                        self._theta_fraction_from_area(feet["free_area"]),
                        flow_guess,
                        head_left_guess,
                        head_right_guess,
                    ]
                ),
            )
            lower = np.array((1.0e-8, -5.0, -10.0, -10.0))
            upper = np.array((1.0 - 1.0e-9, 5.0, 1.0e4, 1.0e4))
        else:
            def residual_at_coefficient(
                x: np.ndarray,
                loss_coefficient: float,
            ) -> np.ndarray:
                (
                    theta_left,
                    theta_right,
                    q_scaled,
                    pressure_velocity_scaled,
                    pressure_head_scaled,
                    shock_speed_scaled,
                ) = x
                area_left, depth_left, celerity_left, moment_left = (
                    self._circular_state_from_theta_fraction(theta_left)
                )
                area_right, depth_right, _, moment_right = (
                    self._circular_state_from_theta_fraction(theta_right)
                )
                volume_flow = q_scaled * full_area * gravity_speed
                velocity_left = volume_flow / area_left
                velocity_right = volume_flow / area_right
                pressure_velocity = pressure_velocity_scaled * gravity_speed
                pressure_head = pressure_head_scaled * diameter
                shock_speed = shock_speed_scaled * gravity_speed
                upstream_area = area_right if volume_flow < 0.0 else area_left
                valve_velocity = volume_flow / upstream_area
                pressure_jump = float(
                    0.5
                    * density
                    * loss_coefficient
                    * abs(valve_velocity)
                    * valve_velocity
                )
                flux_left = (
                    volume_flow * volume_flow / area_left
                    + gravity * moment_left
                )
                flux_right = (
                    volume_flow * volume_flow / area_right
                    + gravity * moment_right
                )
                mass_jump = (
                    full_area * pressure_velocity
                    - volume_flow
                    - shock_speed * (full_area - area_right)
                )
                moving_momentum = (
                    full_area * pressure_velocity - volume_flow
                )
                momentum_jump = (
                    full_area * pressure_velocity * pressure_velocity
                    + gravity
                    * full_area
                    * (pressure_head - 0.5 * diameter)
                    - volume_flow * volume_flow / area_right
                    - gravity * moment_right
                    - gravity * feet["gas_head"] * full_area
                    - shock_speed * moving_momentum
                )
                energy_jump = (
                    depth_left
                    + velocity_left * velocity_left / (2.0 * gravity)
                    - depth_right
                    - velocity_right * velocity_right / (2.0 * gravity)
                    - pressure_jump / (density * gravity)
                )
                valve_momentum = (
                    flux_right
                    - flux_left
                    + pressure_jump * upstream_area / density
                )
                return np.array(
                    [
                        (
                            velocity_left
                            - feet["free_velocity"]
                            + gravity * depth_left / celerity_left
                            - free_foot_term
                            + feet["free_source"]
                        )
                        / gravity_speed,
                        (
                            pressure_velocity
                            - feet["pressure_velocity"]
                            - gravity
                            / acoustic
                            * (pressure_head - feet["pressure_head"])
                            + feet["pressure_source"]
                        )
                        / gravity_speed,
                        mass_jump / (full_area * gravity_speed),
                        momentum_jump
                        / (full_area * gravity_speed * gravity_speed),
                        energy_jump / diameter,
                        valve_momentum
                        / (full_area * gravity_speed * gravity_speed),
                    ],
                    dtype=float,
                )

            def residual(x: np.ndarray) -> np.ndarray:
                return residual_at_coefficient(x, coefficient)

            interface_area_guess = float(
                self.section.area_from_depth(
                    np.clip(
                        state.interface_free_surface_depth,
                        1.0e-12 * diameter,
                        (1.0 - 1.0e-10) * diameter,
                    )
                )
            )
            theta_right_guess = self._theta_fraction_from_area(
                interface_area_guess
            )
            left_area_guess = float(
                np.clip(
                    state.area[self.local_face.mirrored_face_index - 1],
                    1.0e-12 * full_area,
                    (1.0 - 1.0e-10) * full_area,
                )
            )
            theta_left_guess = self._theta_fraction_from_area(left_area_guess)
            core_interface = self._interface_solution(state, dt=dt)
            core_depth = float(
                core_interface.free_surface_depth
                if core_interface.free_surface_depth is not None
                else state.interface_free_surface_depth
            )
            core_area = float(
                self.section.area_from_depth(
                    np.clip(
                        core_depth,
                        1.0e-12 * diameter,
                        (1.0 - 1.0e-10) * diameter,
                    )
                )
            )
            core_theta = self._theta_fraction_from_area(core_area)
            core_velocity = float(
                core_interface.free_surface_velocity
                if core_interface.free_surface_velocity is not None
                else state.interface_free_surface_velocity
            )
            native_flow = float(
                interface_area_guess
                * core_velocity
            )
            flow_guess = float(
                native_flow
                / (full_area * gravity_speed)
                / max(1.0, math.sqrt(coefficient))
            )
            pressure_velocity_guess = float(
                core_interface.pressurised_velocity / gravity_speed
            )
            pressure_head_guess = float(
                core_interface.pressurised_head / diameter
            )
            shock_speed_guess = float(
                core_interface.interface_speed / gravity_speed
            )
            common = (
                flow_guess,
                pressure_velocity_guess,
                pressure_head_guess,
                shock_speed_guess,
            )
            # Start on the exact K=0 Case-1 interface branch.  In particular,
            # the state-carried dry-gate trace can also seed a mathematically
            # small-area root which is disconnected from the pinned native
            # solution.  The native root is the physical continuation datum;
            # no residual gate or equation is changed here.
            native_q_scaled = float(
                core_area * core_velocity / (full_area * gravity_speed)
            )
            native_common = (
                native_q_scaled,
                pressure_velocity_guess,
                pressure_head_guess,
                shock_speed_guess,
            )
            guess_list = [np.array([core_theta, core_theta, *native_common])]
            if not native_release_probe:
                guess_list.extend(
                    (
                        np.array(
                            [theta_left_guess, theta_right_guess, *common]
                        ),
                        np.array(
                            [theta_right_guess, theta_right_guess, *common]
                        ),
                        np.array(
                            (
                                0.5 * (theta_left_guess + theta_right_guess),
                                0.5 * (theta_left_guess + theta_right_guess),
                                *common,
                            )
                        ),
                    )
                )
            zero_flow_theta = self._shock_cut_zero_flow_theta_fraction(feet)
            if zero_flow_theta is not None and not native_release_probe:
                # The near-closed continuous branch has Q=0 and equal depths.
                # Keep the current Case-1 pressure/shock guesses; the unchanged
                # six-equation solve determines their compatible values.
                guess_list.append(
                    np.array(
                        (
                            zero_flow_theta,
                            zero_flow_theta,
                            0.0,
                            pressure_velocity_guess,
                            pressure_head_guess,
                            shock_speed_guess,
                        )
                        )
                    )
            if contact_root is not None and not native_release_probe:
                # The exact stationary contact root lies on the same
                # six-equation manifold at w=0 except for the release
                # equation.  It is therefore a physical warm start for the
                # one-sided entropy limit, not a relaxed acceptance rule.
                guess_list.append(contact_root.free_branch_guess)
            guesses = tuple(guess_list)
            lower = np.array((1.0e-8, 1.0e-8, -5.0, -80.0, -10.0, -80.0))
            upper = np.array(
                (1.0 - 1.0e-9, 1.0 - 1.0e-9, 5.0, 80.0, 1.0e4, 80.0)
            )

        best = None
        best_norm = math.inf
        for guess in guesses:
            bounded_guess = np.minimum(np.maximum(guess, lower), upper)
            candidate = least_squares(
                residual,
                bounded_guess,
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=32 if native_release_probe else 240,
            )
            candidate_residual = residual(candidate.x)
            norm = float(np.max(np.abs(candidate_residual)))
            if norm < best_norm:
                best = candidate
                best_norm = norm
            if norm <= 1.0e-6:
                break
        if (
            orientation is ShockCutValveOrientation.FREE_SURFACE_TOUCH
            and contact_root is not None
            and best_norm > 1.0e-6
        ):
            contact_continuation = least_squares(
                residual,
                np.minimum(
                    np.maximum(contact_root.free_branch_guess, lower),
                    upper,
                ),
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=960,
            )
            contact_norm = float(
                np.max(np.abs(residual(contact_continuation.x)))
            )
            if contact_norm < best_norm:
                best = contact_continuation
                best_norm = contact_norm
        if best is not None and best_norm > 1.0e-6:
            # Near valve/shock contact the unchanged energy and momentum
            # equations become weakly conditioned as Q -> 0 and
            # theta_L -> theta_R.  Continue the best dimensionless root with
            # tighter optimizer termination, while retaining the same bounds,
            # equations and 2e-6 physical acceptance gate.
            continued = least_squares(
                residual,
                np.minimum(np.maximum(best.x, lower), upper),
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=64 if native_release_probe else 720,
            )
            continued_residual = residual(continued.x)
            continued_norm = float(np.max(np.abs(continued_residual)))
            if continued_norm < best_norm:
                best = continued
                best_norm = continued_norm
        homotopy_evaluation_count = 0
        if (
            native_release_probe
            and best is not None
            and best_norm <= 2.0e-6
        ):
            direct_speed = float(best.x[5] * gravity_speed)
            direct_flow = float(best.x[2] * full_area * gravity_speed)
            speed_tolerance = _roundoff_tolerance(
                direct_speed,
                self.case_b_entropy_speed_bound,
                multiplier=4096.0,
            )
            if (
                direct_flow < 0.0
                and direct_speed >= -speed_tolerance
                and direct_speed
                <= self.case_b_entropy_speed_bound + speed_tolerance
            ):
                if orientation is not ShockCutValveOrientation.FREE_SURFACE_TOUCH:
                    raise RuntimeError("native release homotopy needs FREE topology")
                native_guess = np.minimum(
                    np.maximum(guesses[0], lower),
                    upper,
                )
                native_root = least_squares(
                    lambda x: residual_at_coefficient(x, 0.0),
                    native_guess,
                    bounds=(lower, upper),
                    x_scale="jac",
                    ftol=1.0e-13,
                    xtol=1.0e-13,
                    gtol=1.0e-13,
                    max_nfev=240,
                )
                native_norm = float(
                    np.max(
                        np.abs(
                            residual_at_coefficient(native_root.x, 0.0)
                        )
                    )
                )
                homotopy_evaluation_count += int(native_root.nfev)
                if native_norm > 2.0e-6:
                    raise ShockCutValveNonlinearRejected(
                        partition=partition,
                        step_plan=step_plan,
                        stage_index=stage_index,
                        stage_time_s=float(state.time),
                        reason=(
                            "K=0 native FREE anchor residual Linf="
                            f"{native_norm:.6e}"
                        ),
                    )

                previous_x = np.asarray(native_root.x, dtype=float)
                previous_speed = float(previous_x[5] * gravity_speed)
                previous_flow = float(
                    previous_x[2] * full_area * gravity_speed
                )
                target_path = float(math.log1p(coefficient))
                path_nodes = np.linspace(0.0, target_path, 9)[1:]
                previous_path = 0.0
                final_root = native_root
                final_norm = native_norm

                def continue_interval(
                    left_path: float,
                    left_x: np.ndarray,
                    left_speed: float,
                    left_flow: float,
                    right_path: float,
                ):
                    local_coefficient = float(math.expm1(right_path))
                    candidate = least_squares(
                        lambda x: residual_at_coefficient(
                            x,
                            local_coefficient,
                        ),
                        np.minimum(np.maximum(left_x, lower), upper),
                        bounds=(lower, upper),
                        x_scale="jac",
                        ftol=1.0e-12,
                        xtol=1.0e-12,
                        gtol=1.0e-12,
                        max_nfev=240,
                    )
                    norm = float(
                        np.max(
                            np.abs(
                                residual_at_coefficient(
                                    candidate.x,
                                    local_coefficient,
                                )
                            )
                        )
                    )
                    speed = float(candidate.x[5] * gravity_speed)
                    flow = float(candidate.x[2] * full_area * gravity_speed)
                    tolerance = _roundoff_tolerance(
                        speed,
                        left_speed,
                        self.case_b_entropy_speed_bound,
                        multiplier=4096.0,
                    )
                    admissible = bool(
                        norm <= 2.0e-6
                        and flow < 0.0
                        and speed >= -tolerance
                        and speed
                        <= self.case_b_entropy_speed_bound + tolerance
                        and speed <= left_speed + tolerance
                        and flow >= left_flow - _roundoff_tolerance(
                            flow,
                            left_flow,
                            multiplier=4096.0,
                        )
                    )
                    if admissible:
                        return candidate, norm, speed, flow, int(candidate.nfev)
                    midpoint = 0.5 * (left_path + right_path)
                    minimum_span = 64.0 * math.ulp(
                        max(1.0, abs(left_path), abs(right_path))
                    )
                    if not left_path < midpoint < right_path or (
                        right_path - left_path <= minimum_span
                    ):
                        raise ShockCutValveNonlinearRejected(
                            partition=partition,
                            step_plan=step_plan,
                            stage_index=stage_index,
                            stage_time_s=float(state.time),
                            reason=(
                                "K=0-native FREE homotopy lost its passive "
                                "monotone branch"
                            ),
                        )
                    first = continue_interval(
                        left_path,
                        left_x,
                        left_speed,
                        left_flow,
                        midpoint,
                    )
                    second = continue_interval(
                        midpoint,
                        np.asarray(first[0].x, dtype=float),
                        float(first[2]),
                        float(first[3]),
                        right_path,
                    )
                    return (
                        second[0],
                        second[1],
                        second[2],
                        second[3],
                        int(candidate.nfev) + int(first[4]) + int(second[4]),
                    )

                for path in path_nodes:
                    (
                        final_root,
                        final_norm,
                        previous_speed,
                        previous_flow,
                        evaluations,
                    ) = continue_interval(
                        previous_path,
                        previous_x,
                        previous_speed,
                        previous_flow,
                        float(path),
                    )
                    homotopy_evaluation_count += evaluations
                    previous_x = np.asarray(final_root.x, dtype=float)
                    previous_path = float(path)
                best = final_root
                best_norm = final_norm
        if best is None or best_norm > 2.0e-6:
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason=f"implicit residual Linf={best_norm:.6e}",
            )

        if orientation is ShockCutValveOrientation.PRESSURISED_TOUCH:
            theta_fraction, q_scaled, left_head_scaled, right_head_scaled = best.x
            area_fs, depth_fs, celerity_fs, moment_fs = (
                self._circular_state_from_theta_fraction(theta_fraction)
            )
            velocity_fs = free_velocity(depth_fs, celerity_fs)
            volume_flow = float(q_scaled * full_area * gravity_speed)
            left_head = float(left_head_scaled * diameter)
            right_head = float(right_head_scaled * diameter)
            velocity_full = float(volume_flow / full_area)
            pressure_jump = float(
                density * gravity * (left_head - right_head)
            )
            flux_left = float(
                volume_flow * volume_flow / full_area
                + gravity * full_area * (left_head - 0.5 * diameter)
            )
            flux_right = float(
                volume_flow * volume_flow / full_area
                + gravity * full_area * (right_head - 0.5 * diameter)
            )
            interface_discharge = float(area_fs * velocity_fs)
            pressure_velocity = velocity_full
            pressure_head = left_head
            valve_left_area = full_area
            valve_right_area = full_area
        else:
            (
                theta_left,
                theta_right,
                q_scaled,
                pressure_velocity_scaled,
                pressure_head_scaled,
                shock_speed_scaled,
            ) = best.x
            area_left, _, _, moment_left = (
                self._circular_state_from_theta_fraction(theta_left)
            )
            area_right, _, _, moment_right = (
                self._circular_state_from_theta_fraction(theta_right)
            )
            volume_flow = float(q_scaled * full_area * gravity_speed)
            pressure_velocity = float(
                pressure_velocity_scaled * gravity_speed
            )
            pressure_head = float(pressure_head_scaled * diameter)
            shock_speed = float(shock_speed_scaled * gravity_speed)
            upstream_area = area_right if volume_flow < 0.0 else area_left
            valve_velocity = volume_flow / upstream_area
            pressure_jump = float(
                0.5
                * density
                * coefficient
                * abs(valve_velocity)
                * valve_velocity
            )
            flux_left = float(
                volume_flow * volume_flow / area_left
                + gravity * moment_left
            )
            flux_right = float(
                volume_flow * volume_flow / area_right
                + gravity * moment_right
            )
            area_fs = area_right
            interface_discharge = volume_flow
            valve_left_area = area_left
            valve_right_area = area_right
            left_head = None
            right_head = None

        wall_force = float(density * (flux_right - flux_left))
        dissipation = float(pressure_jump * volume_flow)
        if dissipation < -_roundoff_tolerance(dissipation):
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=stage_index,
                stage_time_s=float(state.time),
                reason="coupled root is not passive",
            )
        dissipation = max(0.0, dissipation)
        return ShockCutValveStageSolution(
            stage_index=stage_index,
            stage_time_s=float(state.time),
            orientation=orientation,
            fixed_face_index=partition.fixed_face_index,
            shock_cut_cell_index=partition.shock_cut_cell_index,
            shared_mass_flux_m3_s=volume_flow,
            left_momentum_flux_m4_s2=flux_left,
            right_momentum_flux_m4_s2=flux_right,
            signed_pressure_jump_Pa=pressure_jump,
            valve_wall_force_on_liquid_N=wall_force,
            dissipation_power_W=dissipation,
            valve_left_area_m2=valve_left_area,
            valve_right_area_m2=valve_right_area,
            interface_free_surface_area_m2=area_fs,
            interface_free_surface_discharge_m3_s=interface_discharge,
            interface_pressurised_velocity_m_s=pressure_velocity,
            interface_pressurised_head_m=pressure_head,
            interface_speed_m_s=shock_speed,
            valve_left_head_m=left_head,
            valve_right_head_m=right_head,
            nonlinear_residual_linf=best_norm,
            nonlinear_evaluation_count=(
                int(best.nfev) + homotopy_evaluation_count
            ),
        )

    def _right_boundary_donor_scale(
        self,
        state: WetDryState,
        *,
        dt: float,
        right_ghost: tuple[float, float],
        right_face_flux: tuple[float, float],
    ) -> float:
        """Audit the exact Case-1 donor factor at a prescribed right port."""

        area = np.asarray(state.area, dtype=float)
        discharge = np.asarray(state.discharge, dtype=float)
        ncell = area.size
        if ncell < 1:
            raise ValueError("a cut-cell FV branch needs at least one cell")
        dry_area = self.config.dry_area_fraction * self.section.full_area
        area_ext = np.empty(ncell + 2)
        discharge_ext = np.empty(ncell + 2)
        area_ext[1:-1] = area
        discharge_ext[1:-1] = discharge
        area_ext[0], discharge_ext[0] = _case1_core._ghost_state(
            area[0],
            discharge[0],
            self.config.left_boundary,
        )
        area_ext[-1], discharge_ext[-1] = map(float, right_ghost)
        area_left, q_left, area_right, q_right = (
            _case1_core._muscl_free_surface_face_states(
                area_ext,
                discharge_ext,
                self.section,
                dry_area,
            )
        )
        area_left[0] = area_right[0]
        q_left[0] = (
            -q_right[0]
            if self.config.left_boundary == "wall"
            else q_right[0]
        )
        mass, momentum = _case1_core._central_upwind_flux(
            area_left,
            q_left,
            area_right,
            q_right,
            self.section,
            dry_area,
        )
        mass[-1], momentum[-1] = map(float, right_face_flux)
        _, _, _, scales = _split_momentum_donor_draining_limiter(
            mass,
            momentum,
            momentum,
            area,
            dx=self.dx,
            dt=dt,
        )
        return float(scales[-1])

    def _project_shock_cut_liquid_volume(
        self,
        *,
        state_before: HorizontalState,
        area_new: np.ndarray,
        discharge_new: np.ndarray,
        interface_new: float,
        wetting_front_new: float,
        interface_pressurised_area: float,
        end_time_s: float,
    ) -> float:
        """Apply the pinned Case-1 bounded local/elastic mass projection."""

        volume_before = float(np.sum(state_before.area) * self.dx)
        volume_raw = float(np.sum(area_new) * self.dx)
        volume_residual = volume_before - volume_raw
        cell_left = np.arange(self.ncell, dtype=float) * self.dx
        cell_right = cell_left + self.dx
        free_weight = np.clip(
            (
                np.minimum(cell_right, interface_new)
                - np.maximum(cell_left, wetting_front_new)
            )
            / self.dx,
            0.0,
            1.0,
        )
        pressurised_weight = np.clip(
            (
                np.minimum(cell_right, self.config.length)
                - np.maximum(cell_left, interface_new)
            )
            / self.dx,
            0.0,
            1.0,
        )
        free_cells = free_weight > 1.0e-14
        lower_area = np.zeros(self.ncell, dtype=float)
        upper_area = np.zeros(self.ncell, dtype=float)
        if np.any(free_cells):
            pressure_component = np.where(
                free_cells,
                pressurised_weight * interface_pressurised_area,
                0.0,
            )
            lower_area = np.minimum(pressure_component, area_new)
            upper_area = np.maximum(
                pressure_component
                + free_weight * self.section.full_area,
                area_new,
            )

        remaining = float(volume_residual)
        active = free_cells.copy()
        for _ in range(self.ncell + 1):
            weight_sum = float(np.sum(free_weight[active]))
            if weight_sum <= 1.0e-14 or abs(remaining) <= 1.0e-16:
                break
            delta = remaining / (self.dx * weight_sum)
            proposed = delta * free_weight[active]
            indices = np.flatnonzero(active)
            if remaining > 0.0:
                allowed = upper_area[indices] - area_new[indices]
                applied = np.minimum(proposed, allowed)
            else:
                allowed = lower_area[indices] - area_new[indices]
                applied = np.maximum(proposed, allowed)
            area_new[indices] += applied
            remaining -= float(np.sum(applied) * self.dx)
            saturated = np.isclose(
                applied,
                allowed,
                rtol=0.0,
                atol=1.0e-16 * self.section.full_area,
            )
            if not np.any(saturated):
                break
            active[indices[saturated]] = False

        acoustic_front = min(
            self.config.length,
            self.config.valve_x
            + self.config.wave_speed * float(end_time_s),
        )
        elastic_weight = np.clip(
            (
                np.minimum(cell_right, acoustic_front)
                - np.maximum(cell_left, interface_new)
            )
            / self.dx,
            0.0,
            1.0,
        )
        old_elastic_area = area_new.copy()
        weight_sum = float(np.sum(elastic_weight))
        if abs(remaining) > 1.0e-16:
            if weight_sum <= 1.0e-14:
                raise FloatingPointError(
                    "shock-cut front leaves no admissible mass-projection storage"
                )
            area_new += remaining / (self.dx * weight_sum) * elastic_weight
        corrected = elastic_weight > 0.0
        if np.any(area_new[corrected] <= 0.0):
            raise FloatingPointError(
                "shock-cut mass projection makes a liquid cell non-positive"
            )
        discharge_new[corrected] *= np.divide(
            area_new[corrected],
            np.maximum(
                old_elastic_area[corrected],
                1.0e-14 * self.section.full_area,
            ),
        )
        final_residual = volume_before - float(np.sum(area_new) * self.dx)
        tolerance = 4096.0 * math.ulp(
            max(1.0, abs(volume_before), abs(float(np.sum(area_new) * self.dx)))
        )
        if abs(final_residual) > tolerance:
            raise FloatingPointError(
                "shock-cut liquid-volume projection did not close"
            )
        return volume_residual

    def _shock_cut_euler_stage(
        self,
        state: HorizontalState,
        *,
        dt: float,
        solution: ShockCutValveStageSolution,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        external_pressure_abs: float | None,
    ) -> HorizontalState:
        """Advance one Euler stage from a coupled valve/shock root."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        face = self.local_face.mirrored_face_index
        face_x = self.local_face.mirrored_face_x_m
        step = _positive_finite(dt, name="dt")
        interface_new = _case1_core.advance_shock_position(
            state.interface_x,
            solution.interface_speed_m_s,
            step,
            length=self.config.length,
        )
        area_fs = float(solution.interface_free_surface_area_m2)
        q_fs = float(solution.interface_free_surface_discharge_m3_s)
        depth_fs = float(self.section.depth_from_area(area_fs))
        velocity_fs = float(q_fs / area_fs)
        celerity_fs = float(self.section.celerity(area_fs))
        wetting_front_speed = min(velocity_fs - 2.0 * celerity_fs, 0.0)
        wetting_front_new = float(
            np.clip(
                state.wetting_front_x + wetting_front_speed * step,
                0.0,
                interface_new,
            )
        )

        exact_face_touch = math.isclose(
            float(state.interface_x),
            face_x,
            rel_tol=0.0,
            abs_tol=_roundoff_tolerance(
                state.interface_x,
                face_x,
                multiplier=1024.0,
            ),
        )
        valve_is_fv_boundary = bool(
            solution.orientation
            in (
                ShockCutValveOrientation.FREE_SURFACE_TOUCH,
                ShockCutValveOrientation.FACE_CONTACT,
                ShockCutValveOrientation.DRY_FACE_INJECTION,
            )
            or exact_face_touch
        )
        if valve_is_fv_boundary:
            fs_stop = face
            boundary_flux = (
                solution.shared_mass_flux_m3_s,
                solution.left_momentum_flux_m4_s2,
            )
            boundary_ghost = (
                solution.valve_left_area_m2,
                solution.shared_mass_flux_m3_s,
            )
        else:
            fs_stop = max(1, partition.shock_cut_cell_index)
            boundary_flux = (
                q_fs,
                q_fs * q_fs / area_fs
                + float(self.section.pressure_flux(area_fs)),
            )
            boundary_ghost = (area_fs, q_fs)
        fs_input = WetDryState(
            np.asarray(state.area[:fs_stop], dtype=float).copy(),
            np.asarray(state.discharge[:fs_stop], dtype=float).copy(),
        )
        donor_scale = self._right_boundary_donor_scale(
            fs_input,
            dt=step,
            right_ghost=boundary_ghost,
            right_face_flux=boundary_flux,
        )
        if valve_is_fv_boundary and donor_scale != 1.0:
            raise ShockCutValveDonorLimitRejected(
                partition=partition,
                step_plan=step_plan,
                stage_solution=solution,
                donor_scale=donor_scale,
            )
        try:
            fs_next = _case1_core._central_upwind_wet_dry_euler_step(
                fs_input,
                dx=self.dx,
                dt=step,
                section=self.section,
                cfl=self.config.cfl,
                dry_area_fraction=self.config.dry_area_fraction,
                manning_n=self.config.manning_n,
                darcy_friction=0.0,
                bed_slope=self.config.bed_slope,
                left_boundary=self.config.left_boundary,
                right_boundary="transmissive",
                right_ghost=boundary_ghost,
                right_face_flux=boundary_flux,
            )
        except ValueError as exc:
            if str(exc) != "dt exceeds the requested central-upwind CFL limit":
                raise
            raise ShockCutValveCflRejected(
                partition=partition,
                step_plan=step_plan,
                stage_solution=solution,
                reason=str(exc),
            ) from exc

        if solution.orientation is ShockCutValveOrientation.PRESSURISED_TOUCH:
            pressurised_start = face
            moc_velocity = float(solution.shared_mass_flux_m3_s / self.section.full_area)
            if solution.valve_right_head_m is None:
                raise RuntimeError("pressurised-touch root lacks its right valve head")
            moc_head = float(solution.valve_right_head_m)
        elif solution.orientation in (
            ShockCutValveOrientation.FACE_CONTACT,
            ShockCutValveOrientation.DRY_FACE_INJECTION,
        ):
            pressurised_start = face
            moc_velocity = float(solution.interface_pressurised_velocity_m_s)
            moc_head = float(solution.interface_pressurised_head_m)
        else:
            pressurised_start = partition.shock_cut_cell_index + 1
            moc_velocity = float(solution.interface_pressurised_velocity_m_s)
            moc_head = float(solution.interface_pressurised_head_m)
        p_input = WetDryState(
            np.asarray(state.area[pressurised_start:], dtype=float).copy(),
            np.asarray(state.discharge[pressurised_start:], dtype=float).copy(),
        )
        p_next = _case1_core.pressurised_moc_step(
            p_input,
            dx=self.dx,
            dt=step,
            section=self.section,
            interface_velocity=moc_velocity,
            interface_head=moc_head,
            darcy_friction=self.config.darcy_friction,
            bed_slope=self.config.bed_slope,
            right_boundary=self.config.right_boundary,
        )

        area_new = np.asarray(state.area, dtype=float).copy()
        discharge_new = np.asarray(state.discharge, dtype=float).copy()
        area_new[:fs_stop] = fs_next.area
        discharge_new[:fs_stop] = fs_next.discharge
        area_new[pressurised_start:] = p_next.area
        discharge_new[pressurised_start:] = p_next.discharge

        new_cut = int(
            np.clip(
                math.floor(interface_new / self.dx),
                0,
                self.ncell - 1,
            )
        )
        old_cut = partition.shock_cut_cell_index
        if new_cut > old_cut:
            area_new[old_cut:new_cut] = area_fs
            discharge_new[old_cut:new_cut] = q_fs
        elif new_cut < old_cut:
            pressure_area_left = self._pressurised_area_from_head(
                solution.interface_pressurised_head_m
            )
            pressure_q_left = float(
                pressure_area_left
                * solution.interface_pressurised_velocity_m_s
            )
            area_new[new_cut + 1 : old_cut + 1] = pressure_area_left
            discharge_new[new_cut + 1 : old_cut + 1] = pressure_q_left

        cut_left = new_cut * self.dx
        free_fraction = float(
            np.clip(
                (interface_new - max(wetting_front_new, cut_left)) / self.dx,
                0.0,
                1.0,
            )
        )
        pressure_fraction = float(
            np.clip(
                (cut_left + self.dx - interface_new) / self.dx,
                0.0,
                1.0,
            )
        )
        pressure_area = self._pressurised_area_from_head(
            solution.interface_pressurised_head_m
        )
        pressure_q = float(
            pressure_area * solution.interface_pressurised_velocity_m_s
        )
        area_new[new_cut] = (
            free_fraction * area_fs + pressure_fraction * pressure_area
        )
        discharge_new[new_cut] = (
            free_fraction * q_fs + pressure_fraction * pressure_q
        )
        volume_residual = self._project_shock_cut_liquid_volume(
            state_before=state,
            area_new=area_new,
            discharge_new=discharge_new,
            interface_new=interface_new,
            wetting_front_new=wetting_front_new,
            interface_pressurised_area=pressure_area,
            end_time_s=state.time + step,
        )
        wetdry = WetDryState(area_new, discharge_new)
        gas_volume = self._connected_gas_volume(
            wetdry.area,
            interface_new,
            area_fs,
            wetting_front_new,
        )
        minimum_volume = 1.0e-9 * self.section.full_area * self.config.length
        gas_new = state.gas.with_volume(max(gas_volume, minimum_volume))
        pressure_new, vented = self._effective_pressure(
            time=state.time + step,
            interface_x=interface_new,
            closed_pressure_abs=gas_new.pressure_abs,
            external_pressure_abs=external_pressure_abs,
        )
        return HorizontalState(
            time=state.time + step,
            area=wetdry.area,
            discharge=wetdry.discharge,
            gas=gas_new,
            air_pressure_abs=pressure_new,
            interface_x=interface_new,
            interface_speed=solution.interface_speed_m_s,
            interface_free_surface_depth=depth_fs,
            interface_free_surface_velocity=velocity_fs,
            interface_pressurised_head=solution.interface_pressurised_head_m,
            interface_pressurised_velocity=(
                solution.interface_pressurised_velocity_m_s
            ),
            interface_residual_linf=solution.nonlinear_residual_linf,
            wetting_front_x=wetting_front_new,
            vented=vented,
            nonlinear_converged=True,
            liquid_volume_residual=volume_residual,
            cumulative_liquid_volume_residual=(
                state.cumulative_liquid_volume_residual + volume_residual
            ),
        )

    def _shock_cut_transaction(
        self,
        *,
        dt: float,
        stages: tuple[ShockCutValveStageSolution, ShockCutValveStageSolution],
    ) -> IntegratedValveTransaction:
        """Integrate a coupled cut-valve pair with SSPRK2 half weights."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        step = _positive_finite(dt, name="dt")
        density = float(self.config.liquid_density)
        mirrored_volume = float(
            0.5
            * step
            * math.fsum(stage.shared_mass_flux_m3_s for stage in stages)
        )
        physical_volume = self.local_face.physical_volume_from_mirrored(
            mirrored_volume
        )
        physical_left_impulse = float(
            0.5
            * step
            * density
            * math.fsum(stage.right_momentum_flux_m4_s2 for stage in stages)
        )
        physical_right_impulse = float(
            0.5
            * step
            * density
            * math.fsum(stage.left_momentum_flux_m4_s2 for stage in stages)
        )
        physical_wall_impulse = float(
            physical_right_impulse - physical_left_impulse
        )
        expected_wall = float(
            -0.5
            * step
            * math.fsum(stage.valve_wall_force_on_liquid_N for stage in stages)
        )
        if not math.isclose(
            physical_wall_impulse,
            expected_wall,
            rel_tol=0.0,
            abs_tol=_roundoff_tolerance(
                physical_wall_impulse,
                expected_wall,
                multiplier=4096.0,
            ),
        ):
            raise FloatingPointError(
                "shock-cut SSPRK momentum ports do not close with wall impulse"
            )
        energy = float(
            0.5
            * step
            * math.fsum(stage.dissipation_power_W for stage in stages)
        )
        return IntegratedValveTransaction(
            start_time_s=stages[0].stage_time_s,
            end_time_s=stages[0].stage_time_s + step,
            physical_signed_through_volume_m3=physical_volume,
            physical_left_momentum_impulse_N_s=physical_left_impulse,
            physical_right_momentum_impulse_N_s=physical_right_impulse,
            physical_wall_impulse_on_liquid_N_s=physical_wall_impulse,
            dissipated_energy_J=energy,
            liquid_density_kg_m3=density,
            stage_evaluation_count=2,
            substep_count=1,
        )

    def _step_once_shock_cut(
        self,
        state: HorizontalState,
        dt: float,
        external_pressure_abs: float | None,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        force_dry_attachment: bool = False,
    ) -> tuple[HorizontalState, IntegratedValveTransaction]:
        """Advance one coupled cut element with two reconstructed stages."""

        opening = shared_opening_state(float(state.time))
        if not isinstance(force_dry_attachment, bool):
            raise TypeError("force_dry_attachment must be bool")
        if opening.loss_coefficient == 0.0:
            native = super()._step_once(state, dt, external_pressure_abs)
            return native, IntegratedValveTransaction(
                start_time_s=float(state.time),
                end_time_s=float(native.time),
                physical_signed_through_volume_m3=0.0,
                physical_left_momentum_impulse_N_s=0.0,
                physical_right_momentum_impulse_N_s=0.0,
                physical_wall_impulse_on_liquid_N_s=0.0,
                dissipated_energy_J=0.0,
                liquid_density_kg_m3=self.config.liquid_density,
                stage_evaluation_count=2,
                substep_count=1,
            )
        first_solution = self._solve_shock_cut_stage(
            state,
            dt=dt,
            stage_index=1,
            partition=partition,
            step_plan=step_plan,
            allow_dry_face_release=not force_dry_attachment,
        )
        if self.local_face is None:
            raise RuntimeError("shock-cut stage lost its configured valve face")
        face_x = float(self.local_face.mirrored_face_x_m)
        speed_first = float(first_solution.interface_speed_m_s)
        predicted_first_x = float(state.interface_x + dt * speed_first)
        crosses_face = bool(
            (
                float(state.interface_x) < face_x
                and predicted_first_x >= face_x
                and speed_first > 0.0
            )
            or (
                float(state.interface_x) > face_x
                and predicted_first_x <= face_x
                and speed_first < 0.0
            )
        )
        if crosses_face:
            event_dt = float(
                (face_x - float(state.interface_x)) / speed_first
            )
            if not (0.0 < event_dt <= dt):
                raise FloatingPointError("invalid fixed-face contact event time")
            event_state = self._shock_cut_euler_stage(
                state,
                dt=event_dt,
                solution=first_solution,
                partition=partition,
                step_plan=step_plan,
                external_pressure_abs=external_pressure_abs,
            )
            event_partition = self.classify_local_face_regime(
                replace(event_state, interface_x=face_x)
            )
            gas_volume = self._connected_gas_volume(
                np.asarray(event_state.area, dtype=float),
                face_x,
                first_solution.interface_free_surface_area_m2,
                event_state.wetting_front_x,
            )
            minimum_volume = (
                1.0e-9 * self.section.full_area * self.config.length
            )
            gas_event = state.gas.with_volume(
                max(gas_volume, minimum_volume)
            )
            pressure_event, vented_event = self._effective_pressure(
                time=event_state.time,
                interface_x=face_x,
                closed_pressure_abs=gas_event.pressure_abs,
                external_pressure_abs=external_pressure_abs,
            )
            event_state = replace(
                event_state,
                interface_x=face_x,
                gas=gas_event,
                air_pressure_abs=pressure_event,
                vented=vented_event,
            )
            contact_solution = self._solve_shock_cut_stage(
                event_state,
                dt=event_dt,
                stage_index=2,
                partition=event_partition,
                step_plan=step_plan,
            )
            event_transaction = self._shock_cut_transaction(
                dt=event_dt,
                stages=(first_solution, contact_solution),
            )
            remaining = float(dt - event_dt)
            if remaining <= 0.0:
                return event_state, event_transaction
            end_state, end_transaction = self._step_once_shock_cut(
                event_state,
                remaining,
                external_pressure_abs,
                partition=event_partition,
                step_plan=step_plan,
            )
            return end_state, event_transaction.merged(end_transaction)
        first = self._shock_cut_euler_stage(
            state,
            dt=dt,
            solution=first_solution,
            partition=partition,
            step_plan=step_plan,
            external_pressure_abs=external_pressure_abs,
        )
        second_partition = self.classify_local_face_regime(first)
        if second_partition.regime is not LocalValveRegime.SHOCK_CUT_CELL:
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=2,
                stage_time_s=float(first.time),
                reason=(
                    "one Euler stage left the shock-cut regime; retry with a "
                    "smaller event-aligned substep"
                ),
            )
        second_solution = self._solve_shock_cut_stage(
            first,
            dt=dt,
            stage_index=2,
            partition=partition,
            step_plan=step_plan,
            allow_dry_face_release=not force_dry_attachment,
        )
        second_euler = self._shock_cut_euler_stage(
            first,
            dt=dt,
            solution=second_solution,
            partition=partition,
            step_plan=step_plan,
            external_pressure_abs=external_pressure_abs,
        )
        area_new = 0.5 * (
            np.asarray(state.area, dtype=float)
            + np.asarray(second_euler.area, dtype=float)
        )
        discharge_new = 0.5 * (
            np.asarray(state.discharge, dtype=float)
            + np.asarray(second_euler.discharge, dtype=float)
        )
        interface_new = float(
            state.interface_x
            + 0.5
            * dt
            * (
                first_solution.interface_speed_m_s
                + second_solution.interface_speed_m_s
            )
        )
        if (
            float(state.interface_x) < face_x < interface_new
            or interface_new < face_x < float(state.interface_x)
        ):
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=2,
                stage_time_s=float(first.time),
                reason=(
                    "the SSPRK mean crosses the fixed face after its second "
                    "stage; retry to resolve the contact event"
                ),
            )
        wetting_front_new = float(
            0.5 * (state.wetting_front_x + second_euler.wetting_front_x)
        )
        pressure_area = self._pressurised_area_from_head(
            second_solution.interface_pressurised_head_m
        )
        self._project_shock_cut_liquid_volume(
            state_before=state,
            area_new=area_new,
            discharge_new=discharge_new,
            interface_new=interface_new,
            wetting_front_new=wetting_front_new,
            interface_pressurised_area=pressure_area,
            end_time_s=state.time + dt,
        )
        gas_volume = self._connected_gas_volume(
            area_new,
            interface_new,
            second_solution.interface_free_surface_area_m2,
            wetting_front_new,
        )
        minimum_volume = 1.0e-9 * self.section.full_area * self.config.length
        gas_new = state.gas.with_volume(max(gas_volume, minimum_volume))
        pressure_new, vented = self._effective_pressure(
            time=state.time + dt,
            interface_x=interface_new,
            closed_pressure_abs=gas_new.pressure_abs,
            external_pressure_abs=external_pressure_abs,
        )
        final = HorizontalState(
            time=state.time + dt,
            area=area_new,
            discharge=discharge_new,
            gas=gas_new,
            air_pressure_abs=pressure_new,
            interface_x=interface_new,
            interface_speed=float(
                0.5
                * (
                    first_solution.interface_speed_m_s
                    + second_solution.interface_speed_m_s
                )
            ),
            interface_free_surface_depth=float(
                self.section.depth_from_area(
                    second_solution.interface_free_surface_area_m2
                )
            ),
            interface_free_surface_velocity=float(
                second_solution.interface_free_surface_discharge_m3_s
                / second_solution.interface_free_surface_area_m2
            ),
            interface_pressurised_head=(
                second_solution.interface_pressurised_head_m
            ),
            interface_pressurised_velocity=(
                second_solution.interface_pressurised_velocity_m_s
            ),
            interface_residual_linf=max(
                first_solution.nonlinear_residual_linf,
                second_solution.nonlinear_residual_linf,
            ),
            wetting_front_x=wetting_front_new,
            vented=vented,
            nonlinear_converged=True,
            liquid_volume_residual=float(
                0.5
                * (
                    first.liquid_volume_residual
                    + second_euler.liquid_volume_residual
                )
            ),
            cumulative_liquid_volume_residual=float(
                state.cumulative_liquid_volume_residual
                + 0.5
                * (
                    first.liquid_volume_residual
                    + second_euler.liquid_volume_residual
                )
            ),
        )
        return final, self._shock_cut_transaction(
            dt=dt,
            stages=(first_solution, second_solution),
        )

    def _pressurised_moc_transaction(
        self,
        *,
        dt: float,
        stages: tuple[
            PressurisedMocValveStageSolution,
            PressurisedMocValveStageSolution,
        ],
    ) -> IntegratedValveTransaction:
        """Integrate one split-MOC substep with SSPRK2 half weights."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        step = _positive_finite(dt, name="dt")
        density = float(self.config.liquid_density)
        mirrored_volume = float(
            0.5
            * step
            * math.fsum(stage.shared_mass_flux_m3_s for stage in stages)
        )
        physical_volume = self.local_face.physical_volume_from_mirrored(
            mirrored_volume
        )
        physical_left_impulse = float(
            0.5
            * step
            * density
            * math.fsum(stage.right_momentum_flux_m4_s2 for stage in stages)
        )
        physical_right_impulse = float(
            0.5
            * step
            * density
            * math.fsum(stage.left_momentum_flux_m4_s2 for stage in stages)
        )
        physical_wall_impulse = float(
            physical_right_impulse - physical_left_impulse
        )
        expected_wall = float(
            -0.5
            * step
            * math.fsum(stage.valve_wall_force_on_liquid_N for stage in stages)
        )
        if not math.isclose(
            physical_wall_impulse,
            expected_wall,
            rel_tol=0.0,
            abs_tol=_roundoff_tolerance(
                physical_wall_impulse,
                expected_wall,
                multiplier=4096.0,
            ),
        ):
            raise FloatingPointError(
                "split-MOC SSPRK momentum ports do not close with wall impulse"
            )
        energy = float(
            0.5
            * step
            * math.fsum(stage.dissipation_power_W for stage in stages)
        )
        return IntegratedValveTransaction(
            start_time_s=stages[0].stage_time_s,
            end_time_s=stages[0].stage_time_s + step,
            physical_signed_through_volume_m3=physical_volume,
            physical_left_momentum_impulse_N_s=physical_left_impulse,
            physical_right_momentum_impulse_N_s=physical_right_impulse,
            physical_wall_impulse_on_liquid_N_s=physical_wall_impulse,
            dissipated_energy_J=energy,
            liquid_density_kg_m3=density,
            stage_evaluation_count=2,
            substep_count=1,
        )

    def _step_once_pressurised_moc(
        self,
        state: HorizontalState,
        dt: float,
        external_pressure_abs: float | None,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
    ) -> tuple[HorizontalState, IntegratedValveTransaction]:
        """Advance a clean split-MOC face with two reconstructed stages."""

        opening = shared_opening_state(float(state.time))
        if opening.loss_coefficient == 0.0:
            native = super()._step_once(state, dt, external_pressure_abs)
            return native, IntegratedValveTransaction(
                start_time_s=float(state.time),
                end_time_s=float(native.time),
                physical_signed_through_volume_m3=0.0,
                physical_left_momentum_impulse_N_s=0.0,
                physical_right_momentum_impulse_N_s=0.0,
                physical_wall_impulse_on_liquid_N_s=0.0,
                dissipated_energy_J=0.0,
                liquid_density_kg_m3=self.config.liquid_density,
                stage_evaluation_count=2,
                substep_count=1,
            )

        first_solution = self._solve_pressurised_moc_stage(
            state,
            dt=dt,
            stage_index=1,
            partition=partition,
            step_plan=step_plan,
        )
        first = self._pressurised_moc_euler_stage(
            state,
            dt=dt,
            solution=first_solution,
            partition=partition,
            step_plan=step_plan,
            external_pressure_abs=external_pressure_abs,
        )
        second_partition = self.classify_local_face_regime(first)
        if second_partition.regime is not LocalValveRegime.CLEAN_PRESSURISED_MOC:
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=step_plan,
                stage_index=2,
                stage_time_s=float(first.time),
                reason=(
                    "one Euler stage left the clean split-MOC regime; retry "
                    "with a smaller event-aligned substep"
                ),
            )
        second_solution = self._solve_pressurised_moc_stage(
            first,
            dt=dt,
            stage_index=2,
            partition=second_partition,
            step_plan=step_plan,
        )
        second_euler = self._pressurised_moc_euler_stage(
            first,
            dt=dt,
            solution=second_solution,
            partition=second_partition,
            step_plan=step_plan,
            external_pressure_abs=external_pressure_abs,
        )

        area_new = 0.5 * (
            np.asarray(state.area, dtype=float)
            + np.asarray(second_euler.area, dtype=float)
        )
        discharge_new = 0.5 * (
            np.asarray(state.discharge, dtype=float)
            + np.asarray(second_euler.discharge, dtype=float)
        )
        interface_new = float(
            0.5 * (state.interface_x + second_euler.interface_x)
        )
        wetting_front_new = float(
            0.5 * (state.wetting_front_x + second_euler.wetting_front_x)
        )
        boundary_area = float(
            self.section.area_from_depth(
                second_euler.interface_free_surface_depth
            )
        )
        gas_volume = self._connected_gas_volume(
            area_new,
            interface_new,
            boundary_area,
            wetting_front_new,
        )
        minimum_volume = 1.0e-9 * self.section.full_area * self.config.length
        gas_new = state.gas.with_volume(max(gas_volume, minimum_volume))
        pressure_new, vented = self._effective_pressure(
            time=state.time + dt,
            interface_x=interface_new,
            closed_pressure_abs=gas_new.pressure_abs,
            external_pressure_abs=external_pressure_abs,
        )
        final = HorizontalState(
            time=state.time + dt,
            area=area_new,
            discharge=discharge_new,
            gas=gas_new,
            air_pressure_abs=pressure_new,
            interface_x=interface_new,
            interface_speed=float(
                0.5 * (first.interface_speed + second_euler.interface_speed)
            ),
            interface_free_surface_depth=(
                second_euler.interface_free_surface_depth
            ),
            interface_free_surface_velocity=(
                second_euler.interface_free_surface_velocity
            ),
            interface_pressurised_head=(
                second_euler.interface_pressurised_head
            ),
            interface_pressurised_velocity=(
                second_euler.interface_pressurised_velocity
            ),
            interface_residual_linf=max(
                first.interface_residual_linf,
                second_euler.interface_residual_linf,
            ),
            wetting_front_x=wetting_front_new,
            vented=vented,
            nonlinear_converged=bool(
                first.nonlinear_converged
                and second_euler.nonlinear_converged
            ),
            liquid_volume_residual=float(
                0.5
                * (
                    first.liquid_volume_residual
                    + second_euler.liquid_volume_residual
                )
            ),
            cumulative_liquid_volume_residual=float(
                state.cumulative_liquid_volume_residual
                + 0.5
                * (
                    first.liquid_volume_residual
                    + second_euler.liquid_volume_residual
                )
            ),
        )
        return final, self._pressurised_moc_transaction(
            dt=dt,
            stages=(first_solution, second_solution),
        )

    def _clean_free_surface_callback(
        self,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_solutions: list[CircularSaintVenantValveSolution],
    ) -> InternalFaceFluxCallback:
        """Build the two-stage physical closure for one clean-FV substep."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        dry_area = self.config.dry_area_fraction * self.section.full_area

        def solve_stage(
            context: InternalFaceStageContext,
        ) -> InternalFaceFluxPair:
            if context.face_index != partition.fixed_face_index:
                raise RuntimeError("clean-FV callback received the wrong face")
            trace_areas = (context.west_area_m2, context.east_area_m2)
            if any(area >= self.section.full_area for area in trace_areas):
                raise CleanFreeSurfaceValveTraceRejected(
                    partition=partition,
                    step_plan=step_plan,
                    stage_context=context,
                    reason=(
                        "both reconstructed traces must remain on the circular "
                        "Saint-Venant branch"
                    ),
                )

            try:
                def trace(area: float, discharge: float):
                    # Case-1 already classifies and zeroes reconstructed
                    # discharge at/below this dry area.  Map that exact dry
                    # state to A=Q=c=0 for the Riemann closure; no film or
                    # positive liquid volume is introduced.
                    if area <= dry_area:
                        return CircularSaintVenantValveTrace(
                            area_m2=0.0,
                            discharge_m3_s=0.0,
                            celerity_m_s=0.0,
                            full_area_m2=self.section.full_area,
                            density_kg_m3=self.config.liquid_density,
                            gravity_m_s2=self.config.gravity,
                        )
                    return CircularSaintVenantValveTrace(
                        area_m2=area,
                        discharge_m3_s=discharge,
                        celerity_m_s=float(self.section.celerity(area)),
                        full_area_m2=self.section.full_area,
                        density_kg_m3=self.config.liquid_density,
                        gravity_m_s2=self.config.gravity,
                    )

                west = trace(
                    context.west_area_m2,
                    context.west_discharge_m3_s,
                )
                east = trace(
                    context.east_area_m2,
                    context.east_discharge_m3_s,
                )
                solution = solve_passive_circular_saint_venant_valve(
                    west,
                    east,
                    native_volume_flow_m3_s=(
                        context.native_shared_mass_flux_m3_s
                    ),
                    native_specific_momentum_flux_m4_s2=(
                        context.native_momentum_flux_m4_s2
                    ),
                    time_s=context.stage_time_s,
                )
            except ValueError as error:
                raise CleanFreeSurfaceValveTraceRejected(
                    partition=partition,
                    step_plan=step_plan,
                    stage_context=context,
                    reason=str(error),
                ) from error

            stage_solutions.append(solution)
            if solution.is_exact_native:
                return InternalFaceFluxPair.native(context)
            return InternalFaceFluxPair(
                shared_mass_flux_m3_s=(
                    solution.volume_flow_left_to_right_m3_s
                ),
                west_momentum_flux_m4_s2=(
                    solution.left_specific_momentum_flux_m4_s2
                ),
                east_momentum_flux_m4_s2=(
                    solution.right_specific_momentum_flux_m4_s2
                ),
            )

        return solve_stage

    def _clean_free_surface_transaction(
        self,
        *,
        dt: float,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
        stage_records: tuple[InternalFaceStageRecord, ...],
        stage_solutions: tuple[CircularSaintVenantValveSolution, ...],
    ) -> IntegratedValveTransaction:
        """Integrate one accepted SSPRK2 valve transaction with RK weights."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        if len(stage_records) != 2 or len(stage_solutions) != 2:
            raise RuntimeError("a clean-FV SSPRK2 substep requires two stages")
        step = _positive_finite(dt, name="dt")
        density = float(self.config.liquid_density)

        for record, solution in zip(stage_records, stage_solutions):
            if solution.is_exact_native:
                continue
            requested = record.requested_flux
            if (
                record.donor_scale != 1.0
                or requested is None
                or record.accepted_flux != requested
            ):
                raise CleanFreeSurfaceValveDonorLimitRejected(
                    partition=partition,
                    step_plan=step_plan,
                    stage_record=record,
                )

        mirrored_volume = float(
            0.5
            * step
            * math.fsum(
                record.accepted_flux.shared_mass_flux_m3_s
                for record in stage_records
            )
        )
        physical_volume = self.local_face.physical_volume_from_mirrored(
            mirrored_volume
        )
        physical_left_impulse = float(
            0.5
            * step
            * density
            * math.fsum(
                record.accepted_flux.east_momentum_flux_m4_s2
                for record in stage_records
            )
        )
        physical_right_impulse = float(
            0.5
            * step
            * density
            * math.fsum(
                record.accepted_flux.west_momentum_flux_m4_s2
                for record in stage_records
            )
        )
        physical_wall_impulse = float(
            physical_right_impulse - physical_left_impulse
        )
        expected_physical_wall_impulse = float(
            -0.5
            * step
            * math.fsum(
                solution.valve_wall_force_on_liquid_N
                for solution in stage_solutions
            )
        )
        wall_tolerance = _roundoff_tolerance(
            physical_wall_impulse,
            expected_physical_wall_impulse,
            multiplier=1024.0,
        )
        if not math.isclose(
            physical_wall_impulse,
            expected_physical_wall_impulse,
            rel_tol=0.0,
            abs_tol=wall_tolerance,
        ):
            raise FloatingPointError(
                "SSPRK valve momentum ports do not close with wall impulse"
            )
        dissipated_energy = float(
            0.5
            * step
            * math.fsum(
                solution.dissipation_power_W
                for solution in stage_solutions
            )
        )
        return IntegratedValveTransaction(
            start_time_s=stage_records[0].context.stage_time_s,
            end_time_s=(
                stage_records[0].context.stage_time_s + step
            ),
            physical_signed_through_volume_m3=physical_volume,
            physical_left_momentum_impulse_N_s=physical_left_impulse,
            physical_right_momentum_impulse_N_s=physical_right_impulse,
            physical_wall_impulse_on_liquid_N_s=physical_wall_impulse,
            dissipated_energy_J=dissipated_energy,
            liquid_density_kg_m3=density,
            stage_evaluation_count=2,
            substep_count=1,
        )

    def _step_once_clean_free_surface(
        self,
        state: HorizontalState,
        dt: float,
        external_pressure_abs: float | None,
        *,
        partition: LocalFacePartition,
        step_plan: EventAlignedStepPlan,
    ) -> tuple[HorizontalState, IntegratedValveTransaction]:
        """Case-1 ``_step_once`` with only its clean-FV face replaced."""

        if self.local_face is None:
            raise ValueError("no active local valve face is configured")
        interface_solution = self._interface_solution(state, dt=dt)
        speed = (
            interface_solution.interface_speed
            if interface_solution.converged
            else state.interface_speed
        )
        fs_depth_for_bound = float(
            interface_solution.free_surface_depth
            if interface_solution.free_surface_depth is not None
            else state.interface_free_surface_depth
        )
        fs_velocity_for_bound = float(
            interface_solution.free_surface_velocity
            if interface_solution.free_surface_velocity is not None
            else state.interface_free_surface_velocity
        )
        characteristic_lower = fs_velocity_for_bound + float(
            self.section.free_surface_celerity_from_depth(fs_depth_for_bound)
        )
        characteristic_upper = (
            interface_solution.pressurised_velocity + self.config.wave_speed
        )
        characteristic_margin = 1.0e-10 * max(
            1.0, self.config.wave_speed
        )
        speed = float(
            np.clip(
                speed,
                characteristic_lower + characteristic_margin,
                characteristic_upper - characteristic_margin,
            )
        )
        interface_new = _case1_core.advance_shock_position(
            state.interface_x,
            speed,
            dt,
            length=self.config.length,
        )

        _, pressurised_start = self._interface_cells(state)
        pressurised_start = int(
            np.clip(pressurised_start, 2, self.ncell - 1)
        )
        cut_old = pressurised_start - 1
        fs_stop = cut_old
        fs_area = np.asarray(state.area[:fs_stop], dtype=float).copy()
        fs_discharge = np.asarray(
            state.discharge[:fs_stop], dtype=float
        ).copy()
        p_area_input = np.asarray(
            state.area[pressurised_start:], dtype=float
        ).copy()
        p_discharge_input = np.asarray(
            state.discharge[pressurised_start:], dtype=float
        ).copy()
        boundary_depth = float(
            interface_solution.free_surface_depth
            if interface_solution.free_surface_depth is not None
            else self.dry_gate_depth
        )
        boundary_velocity = float(
            interface_solution.free_surface_velocity
            if interface_solution.free_surface_velocity is not None
            else self.dry_gate_velocity
        )
        boundary_area = float(self.section.area_from_depth(boundary_depth))
        boundary_discharge = boundary_area * boundary_velocity
        boundary_celerity = float(
            self.section.free_surface_celerity_from_depth(boundary_depth)
        )
        wetting_front_speed = min(
            boundary_velocity - 2.0 * boundary_celerity,
            0.0,
        )
        wetting_front_new = float(np.clip(
            state.wetting_front_x + wetting_front_speed * dt,
            0.0,
            interface_new,
        ))

        newly_free = fs_area >= self.section.full_area
        fs_area[newly_free] = boundary_area
        fs_discharge[newly_free] = boundary_discharge
        if not (
            1
            <= partition.fixed_face_index
            < fs_area.size
        ):
            raise RuntimeError(
                "the classified clean-FV valve face is outside the FV branch"
            )
        stage_solutions: list[CircularSaintVenantValveSolution] = []
        callback = self._clean_free_surface_callback(
            partition=partition,
            step_plan=step_plan,
            stage_solutions=stage_solutions,
        )
        fv_result = internal_face_wet_dry_ssprk2_step(
            WetDryState(fs_area, fs_discharge),
            dx=self.dx,
            dt=dt,
            section=self.section,
            face_index=partition.fixed_face_index,
            callback=callback,
            start_time_s=float(state.time),
            cfl=self.config.cfl,
            dry_area_fraction=self.config.dry_area_fraction,
            manning_n=self.config.manning_n,
            darcy_friction=0.0,
            bed_slope=self.config.bed_slope,
            left_boundary=self.config.left_boundary,
            right_boundary="transmissive",
            right_ghost=(boundary_area, boundary_discharge),
        )
        stage_solution_tuple = tuple(stage_solutions)
        valve_transaction = self._clean_free_surface_transaction(
            dt=dt,
            partition=partition,
            step_plan=step_plan,
            stage_records=fv_result.stage_records,
            stage_solutions=stage_solution_tuple,
        )
        fs_next = fv_result.state

        pressurised_next = _case1_core.pressurised_moc_step(
            WetDryState(
                p_area_input,
                p_discharge_input,
            ),
            dx=self.dx,
            dt=dt,
            section=self.section,
            interface_velocity=interface_solution.pressurised_velocity,
            interface_head=interface_solution.pressurised_head,
            darcy_friction=self.config.darcy_friction,
            bed_slope=self.config.bed_slope,
            right_boundary=self.config.right_boundary,
        )
        area_new = np.empty(self.ncell, dtype=float)
        discharge_new = np.empty(self.ncell, dtype=float)
        area_new[:fs_stop] = fs_next.area
        discharge_new[:fs_stop] = fs_next.discharge
        area_new[pressurised_start:] = pressurised_next.area
        discharge_new[pressurised_start:] = pressurised_next.discharge
        area_new[cut_old] = state.area[cut_old]
        discharge_new[cut_old] = state.discharge[cut_old]
        cut_index = int(
            np.clip(
                np.floor(interface_new / self.dx),
                0,
                self.ncell - 1,
            )
        )
        if cut_index > cut_old:
            area_new[cut_old:cut_index] = boundary_area
            discharge_new[cut_old:cut_index] = boundary_discharge
        cut_left = cut_index * self.dx
        free_fraction = float(np.clip(
            (
                interface_new
                - max(wetting_front_new, cut_left)
            ) / self.dx,
            0.0,
            1.0,
        ))
        pressurised_fraction = float(np.clip(
            (cut_left + self.dx - interface_new) / self.dx,
            0.0,
            1.0,
        ))
        pressurised_cut_area = (
            area_new[cut_index]
            if cut_index >= pressurised_start
            else pressurised_next.area[0]
        )
        pressurised_cut_discharge = (
            discharge_new[cut_index]
            if cut_index >= pressurised_start
            else pressurised_next.discharge[0]
        )
        area_new[cut_index] = (
            free_fraction * boundary_area
            + pressurised_fraction * pressurised_cut_area
        )
        discharge_new[cut_index] = (
            free_fraction * boundary_discharge
            + pressurised_fraction * pressurised_cut_discharge
        )

        volume_before = float(np.sum(state.area) * self.dx)
        volume_raw = float(np.sum(area_new) * self.dx)
        volume_residual = volume_before - volume_raw
        cell_left = np.arange(self.ncell, dtype=float) * self.dx
        cell_right = cell_left + self.dx
        free_weight = np.clip(
            (
                np.minimum(cell_right, interface_new)
                - np.maximum(cell_left, wetting_front_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        pressurised_weight = np.clip(
            (
                np.minimum(cell_right, self.config.length)
                - np.maximum(cell_left, interface_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        free_cells = free_weight > 1.0e-14
        lower_area = np.zeros(self.ncell, dtype=float)
        upper_area = np.zeros(self.ncell, dtype=float)
        if np.any(free_cells):
            pressure_component = np.where(
                free_cells,
                pressurised_weight * pressurised_cut_area,
                0.0,
            )
            lower_area = pressure_component
            upper_area = (
                pressure_component
                + free_weight * self.section.full_area
            )
            lower_area = np.minimum(lower_area, area_new)
            upper_area = np.maximum(upper_area, area_new)

        remaining_volume = float(volume_residual)
        active = free_cells.copy()
        for _ in range(self.ncell + 1):
            weight_sum = float(np.sum(free_weight[active]))
            if weight_sum <= 1.0e-14 or abs(remaining_volume) <= 1.0e-16:
                break
            delta = remaining_volume / (self.dx * weight_sum)
            proposed = delta * free_weight[active]
            indices = np.flatnonzero(active)
            if remaining_volume > 0.0:
                allowed = upper_area[indices] - area_new[indices]
                applied = np.minimum(proposed, allowed)
            else:
                allowed = lower_area[indices] - area_new[indices]
                applied = np.maximum(proposed, allowed)
            area_new[indices] += applied
            used = float(np.sum(applied) * self.dx)
            remaining_volume -= used
            saturated = np.isclose(
                applied,
                allowed,
                rtol=0.0,
                atol=1.0e-16 * self.section.full_area,
            )
            if not np.any(saturated):
                break
            active[indices[saturated]] = False

        acoustic_front = min(
            self.config.length,
            self.config.valve_x
            + self.config.wave_speed * (state.time + dt),
        )
        elastic_weight = np.clip(
            (
                np.minimum(cell_right, acoustic_front)
                - np.maximum(cell_left, interface_new)
            ) / self.dx,
            0.0,
            1.0,
        )
        old_elastic_area = area_new.copy()
        weight_sum = float(np.sum(elastic_weight))
        if abs(remaining_volume) > 1.0e-16:
            if weight_sum <= 1.0e-14:
                raise FloatingPointError(
                    "shock front leaves no local storage for mass projection"
                )
            area_new += (
                remaining_volume / (self.dx * weight_sum)
            ) * elastic_weight
        corrected = elastic_weight > 0.0
        if np.any(area_new[corrected] <= 0.0):
            raise FloatingPointError(
                "liquid mass projection would make a pressurised cell non-positive"
            )
        discharge_new[corrected] *= np.divide(
            area_new[corrected],
            np.maximum(
                old_elastic_area[corrected],
                1.0e-14 * self.section.full_area,
            ),
        )
        wetdry = WetDryState(area_new, discharge_new)

        volume_new = self._connected_gas_volume(
            wetdry.area,
            interface_new,
            boundary_area,
            wetting_front_new,
        )
        minimum_volume = (
            1.0e-9 * self.section.full_area * self.config.length
        )
        gas_new = state.gas.with_volume(max(volume_new, minimum_volume))
        pressure_new, vented = self._effective_pressure(
            time=state.time + dt,
            interface_x=interface_new,
            closed_pressure_abs=gas_new.pressure_abs,
            external_pressure_abs=external_pressure_abs,
        )
        advanced = HorizontalState(
            time=state.time + dt,
            area=wetdry.area,
            discharge=wetdry.discharge,
            gas=gas_new,
            air_pressure_abs=pressure_new,
            interface_x=interface_new,
            interface_speed=speed,
            interface_free_surface_depth=boundary_depth,
            interface_free_surface_velocity=boundary_velocity,
            interface_pressurised_head=(
                interface_solution.pressurised_head
            ),
            interface_pressurised_velocity=(
                interface_solution.pressurised_velocity
            ),
            interface_residual_linf=interface_solution.residual_linf,
            wetting_front_x=wetting_front_new,
            vented=vented,
            nonlinear_converged=interface_solution.converged,
            liquid_volume_residual=volume_residual,
            cumulative_liquid_volume_residual=(
                state.cumulative_liquid_volume_residual + volume_residual
            ),
        )
        return advanced, valve_transaction

    def step(
        self,
        state: HorizontalState,
        dt: float,
        *,
        external_pressure_abs: float | None = None,
    ) -> HorizontalState:
        if self.local_face is None:
            return super().step(
                state,
                dt,
                external_pressure_abs=external_pressure_abs,
            )
        return self.advance_with_transaction(
            state,
            dt,
            external_pressure_abs=external_pressure_abs,
        ).state

    def advance_with_transaction(
        self,
        state: HorizontalState,
        dt: float,
        *,
        external_pressure_abs: float | None = None,
    ) -> HorizontalAdvanceResult:
        """Advance no-valve, shock-cut, or clean-FV paths atomically."""

        if self.local_face is None:
            advanced = super().step(
                state,
                dt,
                external_pressure_abs=external_pressure_abs,
            )
            transaction = IntegratedValveTransaction.zero(
                start_time_s=float(state.time),
                end_time_s=float(advanced.time),
                liquid_density_kg_m3=self.config.liquid_density,
            )
            return HorizontalAdvanceResult(
                state=advanced,
                valve_transaction=transaction,
            )

        plan = self.plan_physical_step(state, dt)
        initial_partition = self.classify_local_face_regime(state)

        current = state
        transaction = IntegratedValveTransaction.zero(
            start_time_s=float(state.time),
            end_time_s=float(state.time),
            liquid_density_kg_m3=self.config.liquid_density,
        )
        final_partition = initial_partition

        def advance_substep(substep: float) -> None:
            nonlocal current, transaction, final_partition

            def solve_interval(
                start_state: HorizontalState,
                interval: float,
                depth: int = 0,
            ) -> tuple[
                HorizontalState,
                IntegratedValveTransaction,
                LocalFacePartition,
            ]:
                interval_partition = self.classify_local_face_regime(start_state)
                try:
                    if (
                        interval_partition.regime
                        is LocalValveRegime.SHOCK_CUT_CELL
                    ):
                        advanced, interval_transaction = self._step_once_shock_cut(
                            start_state,
                            interval,
                            external_pressure_abs,
                            partition=interval_partition,
                            step_plan=plan,
                        )
                        path_label = "shock-cut"
                    elif (
                        interval_partition.regime
                        is LocalValveRegime.CLEAN_PRESSURISED_MOC
                    ):
                        advanced, interval_transaction = (
                            self._step_once_pressurised_moc(
                                start_state,
                                interval,
                                external_pressure_abs,
                                partition=interval_partition,
                                step_plan=plan,
                            )
                        )
                        path_label = "pressurised split-MOC"
                    else:
                        advanced, interval_transaction = (
                            self._step_once_clean_free_surface(
                                start_state,
                                interval,
                                external_pressure_abs,
                                partition=interval_partition,
                                step_plan=plan,
                            )
                        )
                        path_label = "clean free-surface"
                except (
                    ShockCutValveNonlinearRejected,
                    ShockCutValveDonorLimitRejected,
                    ShockCutValveCflRejected,
                    CleanFreeSurfaceValveDonorLimitRejected,
                    PressurisedMocValveStageRejected,
                ):
                    half = 0.5 * interval
                    minimum = max(
                        64.0 * math.ulp(max(1.0, abs(start_state.time))),
                        1.0e-12 * float(dt),
                    )
                    if depth >= 32 or half <= minimum:
                        raise
                    middle, first_transaction, _ = solve_interval(
                        start_state,
                        half,
                        depth + 1,
                    )
                    end, second_transaction, end_partition = solve_interval(
                        middle,
                        interval - half,
                        depth + 1,
                    )
                    return (
                        end,
                        first_transaction.merged(second_transaction),
                        end_partition,
                    )
                if advanced.time <= start_state.time:
                    raise FloatingPointError(
                        f"{path_label} valve step made no time progress"
                    )
                return advanced, interval_transaction, interval_partition

            advanced, subtransaction, final_partition = solve_interval(
                current,
                substep,
            )
            current = advanced
            transaction = transaction.merged(subtransaction)

        if not plan.interior_events_s:
            # Preserve the original Case-1 subtraction order when no event is
            # crossed.  In particular an exact K=0 request follows the same
            # CFL loop and receives the same dt bits as ``super().step``.
            remaining = float(dt)
            tolerance = max(1.0e-14, 1.0e-12 * dt)
            while remaining > tolerance:
                stable = self.stable_timestep(current)
                substep = min(remaining, 0.999999 * stable)
                if substep <= 0.0 or not math.isfinite(substep):
                    raise FloatingPointError(
                        "invalid local-valve time step"
                    )
                advance_substep(substep)
                remaining -= substep
        else:
            targets = (*plan.interior_events_s, plan.end_time_s)
            for target in targets:
                while current.time < target:
                    stable = self.stable_timestep(current)
                    remaining = float(target - current.time)
                    substep = min(remaining, 0.999999 * stable)
                    if substep <= 0.0 or not math.isfinite(substep):
                        raise FloatingPointError(
                            "invalid local-valve time step"
                        )
                    advance_substep(substep)
                if current.time != target:
                    raise FloatingPointError(
                        "local valve failed to land on an event time"
                    )

        return HorizontalAdvanceResult(
            state=current,
            valve_transaction=transaction,
            partition=final_partition,
        )


__all__ = [
    "CORE_SOURCE",
    "EXPECTED_CORE_SHA256",
    "Case1LocalValveExtension",
    "CleanFreeSurfaceValveDonorLimitRejected",
    "CleanFreeSurfaceValveTraceRejected",
    "EventAlignedStepPlan",
    "FixedInternalValveSpec",
    "FreeSurfaceValvePathNotImplemented",
    "HorizontalAdvanceResult",
    "InternalFaceFluxCallback",
    "InternalFaceFluxPair",
    "InternalFaceSsprk2Result",
    "InternalFaceStageContext",
    "InternalFaceStageRecord",
    "IntegratedValveTransaction",
    "LocalFacePartition",
    "LocalValvePathNotImplemented",
    "LocalValveRegime",
    "PressurisedMocValvePathNotImplemented",
    "PressurisedMocValveStageRejected",
    "PressurisedMocValveStageSolution",
    "ShockCutValveDonorLimitRejected",
    "ShockCutValveCflRejected",
    "ShockCutValveNonlinearRejected",
    "ShockCutValveOrientation",
    "ShockCutValveStageSolution",
    "internal_face_wet_dry_ssprk2_step",
    "plan_event_aligned_step",
    "source_sha256",
]
