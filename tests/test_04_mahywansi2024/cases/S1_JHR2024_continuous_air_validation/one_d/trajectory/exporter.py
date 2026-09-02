"""Build and atomically export canonical S1 1-D trajectory artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from model.conservation import LedgerEntry
from model.errors import MissingPhysicalClosure
from model.state import CoupledGeometry

from .contracts import (
    ArtifactRole,
    CommonAcceptedSample,
    ObservationContractError,
    ObserverContract,
    load_observer_contract,
)
from .observer import CumulativeLedgerResiduals, ObservedFrame, S1CanonicalObserver


FORBIDDEN_RESULT_MARKERS = (
    "RESULT_ACCEPTED",
    "RUN_COMPLETE_UNVALIDATED",
    "ERUPTION_ACCEPTED",
)

EXTRA_SCALAR_COLUMNS = (
    "P4_gauge_Pa",
    "P5_gauge_Pa",
    "P6_gauge_Pa",
    "mouth_liquid_inflow_m3_s",
    "mouth_gas_inflow_kg_s",
    "riser_mouth_state_Qup_m3_s",
    "riser_mouth_state_Qdown_m3_s",
    "air_supply_node_momentum_x_residual_N",
    "air_supply_node_momentum_z_residual_N",
    "riser_node_momentum_x_residual_N",
    "riser_node_momentum_z_residual_N",
    "internal_mouth_event_accepted_once",
    "internal_mouth_event_onset_s",
    "internal_mouth_event_acceptance_time_s",
    "derived_plume_height_proxy_m",
)

CORE_RUNTIME_INTERFACES_REQUIRED = (
    "accepted_state_callback_at_exact_0p10s_stage2_grid_including_t0",
    "every_accepted_LedgerEntry_since_previous_common_sample",
    "P1_through_P6_native_gauge_pressure_packet",
    "accepted_supply_and_mouth_gross_phase_flux_packet",
    "accepted_two_T_node_residual_and_reaction_packet",
    "native_internal_mouth_event_packet",
)


@dataclass(frozen=True, slots=True)
class CanonicalTrajectory:
    rows: tuple[Mapping[str, float], ...]
    frames: tuple[ObservedFrame, ...]
    ledger_rows: tuple[Mapping[str, Any], ...]
    contract: ObserverContract
    artifact_role: ArtifactRole
    stage2_origin_absolute_s: float
    operator_production_ready: bool
    unavailable_by_time: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class TrajectoryArtifactSet:
    canonical_csv: Path
    riser_profiles_npz: Path
    conservation_ledger_csv: Path
    metadata_json: Path


def require_production_operator(operator: object) -> None:
    """Fail closed before any formal long trajectory or artifact is authorized."""

    if getattr(operator, "production_ready", None) is not True:
        raise MissingPhysicalClosure(
            "formal S1 trajectory requires operator.production_ready is exactly True"
        )


def _ledger_row(entry: LedgerEntry, sample_time_s: float) -> dict[str, Any]:
    boundary = entry.boundary
    return {
        "sample_stage2_time_s": sample_time_s,
        "transaction_id": entry.transaction_id,
        "time_start_absolute_s": entry.time_start_s,
        "time_end_absolute_s": entry.time_end_s,
        "liquid_volume_residual_m3": entry.liquid_volume_residual_m3,
        "gas_mass_residual_kg": entry.gas_mass_residual_kg,
        "horizontal_momentum_residual_Ns": entry.mixture_momentum_x_residual_kg_m_s,
        "vertical_momentum_residual_Ns": entry.mixture_momentum_z_residual_kg_m_s,
        "liquid_inflow_m3_s": boundary.liquid_inflow_m3_s,
        "liquid_outflow_m3_s": boundary.liquid_outflow_m3_s,
        "gas_inflow_kg_s": boundary.gas_inflow_kg_s,
        "gas_outflow_kg_s": boundary.gas_outflow_kg_s,
        "boundary_momentum_x_impulse_Ns": entry.boundary_momentum_x_impulse_kg_m_s,
        "boundary_momentum_z_impulse_Ns": entry.boundary_momentum_z_impulse_kg_m_s,
        "external_force_x_impulse_Ns": entry.external_force_x_impulse_kg_m_s,
        "external_force_z_impulse_Ns": entry.external_force_z_impulse_kg_m_s,
    }


def _validate_interval(
    entries: tuple[LedgerEntry, ...],
    *,
    previous_absolute_s: float,
    current_absolute_s: float,
    seen_transactions: set[str],
) -> None:
    if not entries:
        raise ObservationContractError(
            "nonzero common-time interval omits its accepted conservation ledgers"
        )
    tolerance = 1.0e-10
    expected = previous_absolute_s
    for entry in entries:
        if entry.transaction_id in seen_transactions:
            raise ObservationContractError(
                f"duplicate accepted transaction in trajectory: {entry.transaction_id}"
            )
        seen_transactions.add(entry.transaction_id)
        if not math.isclose(entry.time_start_s, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ObservationContractError(
                "accepted ledger interval has a gap, overlap, or reordered transaction"
            )
        if entry.time_end_s <= entry.time_start_s:
            raise ObservationContractError("accepted ledger entry has non-positive duration")
        expected = entry.time_end_s
    if not math.isclose(expected, current_absolute_s, rel_tol=0.0, abs_tol=tolerance):
        raise ObservationContractError(
            "accepted ledger interval does not terminate at the common state"
        )


def build_canonical_trajectory(
    samples: Sequence[CommonAcceptedSample],
    *,
    geometry: CoupledGeometry,
    stage2_origin_absolute_s: float,
    artifact_role: ArtifactRole,
    operator: object | None = None,
    contract: ObserverContract | None = None,
) -> CanonicalTrajectory:
    """Build an in-memory artifact from exact accepted common-time samples.

    Formal mode first checks the production owner and then rejects every
    unavailable canonical diagnostic. Synthetic mode exists only for contract
    tests; it cannot authorize a run or create an acceptance marker.
    """

    if artifact_role not in ("formal_production", "synthetic_contract_test"):
        raise ObservationContractError(f"unsupported artifact_role: {artifact_role!r}")
    production_ready = getattr(operator, "production_ready", None) is True
    if artifact_role == "formal_production":
        if operator is None:
            raise MissingPhysicalClosure("formal S1 trajectory has no physical operator")
        require_production_operator(operator)
    origin = float(stage2_origin_absolute_s)
    if not math.isfinite(origin) or origin < 0.0:
        raise ObservationContractError(
            "stage2_origin_absolute_s must be finite and non-negative"
        )
    frozen = contract or load_observer_contract()
    if not samples:
        raise ObservationContractError("canonical trajectory requires at least one sample")
    observer = S1CanonicalObserver(geometry=geometry, contract=frozen)

    residual = CumulativeLedgerResiduals()
    rows: list[Mapping[str, float]] = []
    frames: list[ObservedFrame] = []
    ledgers: list[Mapping[str, Any]] = []
    unavailable: dict[str, Mapping[str, str]] = {}
    seen_transactions: set[str] = set()
    previous_state_time: float | None = None
    previous_common_time: float | None = None
    tolerance = 1.0e-10
    for index, sample in enumerate(samples):
        expected_common = index * frozen.common_dt_s
        if not math.isclose(
            sample.stage2_time_s, expected_common, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ObservationContractError(
                "samples must be contiguous exact Stage-2 t=0,0.1,... common times"
            )
        expected_absolute = origin + sample.stage2_time_s
        if not math.isclose(
            sample.state.time_s, expected_absolute, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ObservationContractError(
                "state time is not the declared unshifted Stage-2 time coordinate"
            )
        geometry.validate_state(sample.state)
        entries = sample.ledger_entries_since_previous_sample
        if index == 0:
            if entries:
                raise ObservationContractError(
                    "Stage-2 t=0 must not consume pre-Stage-2 ledger entries"
                )
        else:
            assert previous_state_time is not None and previous_common_time is not None
            _validate_interval(
                entries,
                previous_absolute_s=previous_state_time,
                current_absolute_s=sample.state.time_s,
                seen_transactions=seen_transactions,
            )
            residual = CumulativeLedgerResiduals(
                liquid_volume_m3=residual.liquid_volume_m3
                + sum(entry.liquid_volume_residual_m3 for entry in entries),
                gas_mass_kg=residual.gas_mass_kg
                + sum(entry.gas_mass_residual_kg for entry in entries),
                horizontal_momentum_Ns=residual.horizontal_momentum_Ns
                + sum(entry.mixture_momentum_x_residual_kg_m_s for entry in entries),
                vertical_momentum_Ns=residual.vertical_momentum_Ns
                + sum(entry.mixture_momentum_z_residual_kg_m_s for entry in entries),
            )
            ledgers.extend(_ledger_row(entry, sample.stage2_time_s) for entry in entries)
        frame = observer.observe(sample, cumulative=residual)
        if frame.unavailable:
            unavailable[f"{sample.stage2_time_s:.10g}"] = dict(frame.unavailable)
        rows.append(frame.row)
        frames.append(frame)
        previous_state_time = sample.state.time_s
        previous_common_time = sample.stage2_time_s

    if artifact_role == "formal_production" and unavailable:
        details = "; ".join(
            f"t={time}: {sorted(items)}" for time, items in unavailable.items()
        )
        raise ObservationContractError(
            "formal canonical trajectory has unavailable required diagnostics: " + details
        )
    return CanonicalTrajectory(
        rows=tuple(rows),
        frames=tuple(frames),
        ledger_rows=tuple(ledgers),
        contract=frozen,
        artifact_role=artifact_role,
        stage2_origin_absolute_s=origin,
        operator_production_ready=production_ready,
        unavailable_by_time=unavailable,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return Path(raw), descriptor


def _write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    temporary, descriptor = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for source in rows:
                row: dict[str, Any] = {}
                for name in columns:
                    value = source.get(name, "")
                    if isinstance(value, float) and not math.isfinite(value):
                        value = ""
                    row[name] = value
                writer.writerow(row)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    temporary, descriptor = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary, descriptor = _atomic_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _profile_payload(trajectory: CanonicalTrajectory) -> dict[str, np.ndarray]:
    frames = trajectory.frames
    reference_z = frames[0].profile.z_cell_center_m
    if any(
        not np.array_equal(frame.profile.z_cell_center_m, reference_z)
        for frame in frames[1:]
    ):
        raise ObservationContractError(
            "native riser cell-centre grid changed during a fixed-grid trajectory"
        )
    return {
        "time_s": np.asarray([float(row["time_s"]) for row in trajectory.rows]),
        "riser_z_cell_center_m": reference_z,
        "riser_Aup_m2": np.stack([frame.profile.Aup_m2 for frame in frames]),
        "riser_Qup_m3_s": np.stack([frame.profile.Qup_m3_s for frame in frames]),
        "riser_Adown_m2": np.stack([frame.profile.Adown_m2 for frame in frames]),
        "riser_Qdown_m3_s": np.stack([frame.profile.Qdown_m3_s for frame in frames]),
        "riser_gas_area_m2": np.stack([frame.profile.gas_area_m2 for frame in frames]),
        "riser_gas_mass_kg_m": np.stack([frame.profile.gas_mass_kg_m for frame in frames]),
        "riser_gas_velocity_m_s": np.stack(
            [frame.profile.gas_velocity_m_s for frame in frames]
        ),
        "riser_gas_velocity_available": np.stack(
            [frame.profile.gas_velocity_available for frame in frames]
        ),
    }


def write_trajectory_artifacts(
    trajectory: CanonicalTrajectory, output_dir: Path
) -> TrajectoryArtifactSet:
    """Write diagnostic artifacts; this function cannot write result markers."""

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / marker).exists() for marker in FORBIDDEN_RESULT_MARKERS):
        # Existing external markers are neither deleted nor modified. Refuse
        # before writing so a diagnostic export cannot be mistaken for the
        # process that created or promoted that marker.
        raise ObservationContractError(
            "output directory already contains a forbidden external result marker"
        )
    artifacts = TrajectoryArtifactSet(
        canonical_csv=output / "one_d_canonical_timeseries.csv",
        riser_profiles_npz=output / "riser_twofluid_profiles.npz",
        conservation_ledger_csv=output / "conservation_ledger.csv",
        metadata_json=output / "one_d_trajectory.metadata.json",
    )
    scalar_columns = tuple(trajectory.contract.canonical_series) + EXTRA_SCALAR_COLUMNS
    _write_csv_atomic(artifacts.canonical_csv, trajectory.rows, scalar_columns)
    _write_npz_atomic(artifacts.riser_profiles_npz, _profile_payload(trajectory))
    ledger_columns = (
        "sample_stage2_time_s",
        "transaction_id",
        "time_start_absolute_s",
        "time_end_absolute_s",
        "liquid_volume_residual_m3",
        "gas_mass_residual_kg",
        "horizontal_momentum_residual_Ns",
        "vertical_momentum_residual_Ns",
        "liquid_inflow_m3_s",
        "liquid_outflow_m3_s",
        "gas_inflow_kg_s",
        "gas_outflow_kg_s",
        "boundary_momentum_x_impulse_Ns",
        "boundary_momentum_z_impulse_Ns",
        "external_force_x_impulse_Ns",
        "external_force_z_impulse_Ns",
    )
    _write_csv_atomic(
        artifacts.conservation_ledger_csv, trajectory.ledger_rows, ledger_columns
    )
    metadata = {
        "schema_version": 1,
        "case_id": trajectory.contract.raw.get("case_id", "S1_JHR2024_continuous_air_validation"),
        "physical_condition_count": 1,
        "artifact_role": trajectory.artifact_role,
        "operator_production_ready_at_build": trajectory.operator_production_ready,
        "time_origin": "stage_2_air_opening",
        "stage2_origin_absolute_s": trajectory.stage2_origin_absolute_s,
        "time_shift_applied_s": 0.0,
        "common_grid_s": trajectory.contract.common_dt_s,
        "time_start_s": float(trajectory.rows[0]["time_s"]),
        "time_end_s": float(trajectory.rows[-1]["time_s"]),
        "sample_count": len(trajectory.rows),
        "canonical_columns": list(trajectory.contract.canonical_series),
        "extra_diagnostic_columns": list(EXTRA_SCALAR_COLUMNS),
        "unavailable_by_time": trajectory.unavailable_by_time,
        "fig8_velocity_semantics": trajectory.contract.raw["source_semantics"][
            "fig8_horizontal_slug_velocity"
        ],
        "pressure_probe_semantics": trajectory.contract.raw["source_semantics"][
            "published_pressure_probes"
        ],
        "derived_plume_height_semantics": trajectory.contract.raw["source_semantics"][
            "external_plume_height"
        ],
        "profile_semantics": {
            "native_states_preserved": ["Aup", "Qup", "Adown", "Qdown", "Mg", "Jg"],
            "Qup_Qdown_reconstructed_from_net": False,
            "gas_velocity": "Jg/Mg where Mg>0; NaN plus validity mask otherwise",
        },
        "core_runtime_interfaces_required": list(CORE_RUNTIME_INTERFACES_REQUIRED),
        "result_marker_written": False,
        "forbidden_result_markers": list(FORBIDDEN_RESULT_MARKERS),
        "contract_file": str(trajectory.contract.path),
        "contract_sha256": _sha256(trajectory.contract.path),
        "artifacts": {
            "canonical_csv": {
                "path": str(artifacts.canonical_csv),
                "sha256": _sha256(artifacts.canonical_csv),
            },
            "riser_profiles_npz": {
                "path": str(artifacts.riser_profiles_npz),
                "sha256": _sha256(artifacts.riser_profiles_npz),
            },
            "conservation_ledger_csv": {
                "path": str(artifacts.conservation_ledger_csv),
                "sha256": _sha256(artifacts.conservation_ledger_csv),
                "entry_count": len(trajectory.ledger_rows),
            },
        },
        "acceptance_scope": (
            "Evidence export only. This artifact cannot create or imply RESULT_ACCEPTED."
        ),
    }
    _write_json_atomic(artifacts.metadata_json, metadata)
    return artifacts


__all__ = [
    "CORE_RUNTIME_INTERFACES_REQUIRED",
    "CanonicalTrajectory",
    "FORBIDDEN_RESULT_MARKERS",
    "TrajectoryArtifactSet",
    "build_canonical_trajectory",
    "require_production_operator",
    "write_trajectory_artifacts",
]
