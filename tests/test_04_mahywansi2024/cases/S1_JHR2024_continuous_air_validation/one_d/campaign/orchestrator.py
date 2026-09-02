"""Fail-closed S1 campaign orchestration without any result acceptance logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import CampaignProtocolConfig
from .contracts import (
    AcceptedCommonState,
    AcceptedStepCallback,
    CommonStateCallback,
    ExactAdvanceRunner,
    ObservationBridge,
    Stage1AcceptanceCallback,
    Stage1AcceptanceCandidate,
    Stage1Observation,
    StateCodec,
)
from .stability import Stage1StabilityReport, evaluate_stage1_stability


class CampaignProtocolError(RuntimeError):
    """The runner, authorization, or state violated the frozen campaign contract."""


@dataclass(frozen=True, slots=True)
class Stage1Checkpoint:
    state_path: Path
    manifest_path: Path
    state_sha256: str
    manifest_sha256: str
    state_time_s: float
    codec_id: str


@dataclass(frozen=True, slots=True)
class PreproductionSmokeResult:
    state: Any
    duration_s: float
    physical_stage: str = "stage1_closed"
    completion_marker_written: bool = False
    validation_scope: str = "preproduction_smoke_only"


@dataclass(frozen=True, slots=True)
class FormalCampaignResult:
    state: Any
    stage1_report: Stage1StabilityReport
    checkpoint: Stage1Checkpoint
    stage1_end_absolute_time_s: float
    stage2_duration_s: float
    common_state_count: int
    same_in_memory_state_used_at_stage2_opening: bool
    status: str = "RUN_COMPLETE_UNVALIDATED"
    result_accepted: bool = False


def _canonical_json(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignProtocolError("campaign artifact is not finite canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _time(codec: StateCodec, state: Any, label: str) -> float:
    value = float(codec.time_s(state))
    if not math.isfinite(value) or value < 0.0:
        raise CampaignProtocolError(f"{label} state time is not finite and non-negative")
    return value


def _assert_exact_time(codec: StateCodec, state: Any, target: float, label: str) -> None:
    actual = _time(codec, state, label)
    tolerance = 2.0e-12 * max(1.0, abs(target))
    if not math.isclose(actual, target, rel_tol=0.0, abs_tol=tolerance):
        raise CampaignProtocolError(
            f"{label} returned t={actual:.17g}, not the exact event ceiling {target:.17g}; "
            "state interpolation is forbidden"
        )


class _MarkerStore:
    def __init__(
        self,
        config: CampaignProtocolConfig,
        authorization_dir: Path,
        run_dir: Path | None,
    ) -> None:
        self.config = config
        self.authorization_dir = authorization_dir.resolve()
        self.run_dir = None if run_dir is None else run_dir.resolve()

    def _authorization(self, name: str) -> None:
        path = self.authorization_dir / name
        if not path.is_file() or path.is_symlink():
            raise CampaignProtocolError(f"missing trusted authorization marker: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignProtocolError(f"authorization marker is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise CampaignProtocolError(f"authorization marker must contain an object: {path}")
        expected = {
            "schema_version": self.config.markers.authorization_schema,
            "case_id": self.config.case_id,
            "campaign_config_sha256": self.config.sha256,
            "authorized": True,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CampaignProtocolError(
                    f"authorization marker {path.name} has stale or invalid {key}"
                )

    def assert_smoke_authorized(self) -> None:
        self._authorization(self.config.markers.preproduction_smoke_authorization)

    def assert_formal_authorized(self, runner: ExactAdvanceRunner) -> None:
        if runner.production_ready is not True:
            raise CampaignProtocolError(
                "formal long run requires the concrete operator to be production_ready"
            )
        self._authorization(self.config.markers.model_implementation_acceptance)
        self._authorization(self.config.markers.formal_campaign_authorization)
        if self.run_dir is None:
            raise CampaignProtocolError("formal run has no output directory")
        for name in self.config.markers.forbidden_markers:
            if (self.run_dir / name).exists():
                raise CampaignProtocolError(f"forbidden stale result marker present: {name}")

    def write_progress(self, name: str, payload: dict[str, Any]) -> Path:
        if self.run_dir is None:
            raise CampaignProtocolError("smoke runs may not write campaign markers")
        allowed = {
            self.config.markers.stage1_steady_accepted,
            self.config.markers.stage1_checkpoint_accepted,
            self.config.markers.stage2_started,
            self.config.markers.only_terminal_campaign_marker,
        }
        if name not in allowed or name in self.config.markers.forbidden_markers:
            raise CampaignProtocolError(f"campaign controller is forbidden to write marker {name}")
        path = self.run_dir / name
        if path.exists():
            raise CampaignProtocolError(f"campaign marker already exists: {path}")
        record = {
            "schema_version": "s1_1d_campaign_marker_v1",
            "case_id": self.config.case_id,
            "campaign_config_sha256": self.config.sha256,
            "marker": name,
            **payload,
        }
        _atomic_write(path, _canonical_json(record))
        return path


def _write_checkpoint(
    output_dir: Path,
    codec: StateCodec,
    state: Any,
    report: Stage1StabilityReport,
    config: CampaignProtocolConfig,
) -> Stage1Checkpoint:
    state_payload = codec.encode(state)
    if not isinstance(state_payload, bytes) or not state_payload:
        raise CampaignProtocolError("state codec must return non-empty bytes")
    decoded = codec.decode(state_payload)
    if codec.encode(decoded) != state_payload:
        raise CampaignProtocolError("checkpoint codec is not deterministic on round trip")
    state_time = _time(codec, state, "Stage-1 accepted")
    decoded_time = _time(codec, decoded, "decoded Stage-1 accepted")
    if not math.isclose(state_time, decoded_time, rel_tol=0.0, abs_tol=1.0e-14):
        raise CampaignProtocolError("checkpoint round trip changed the accepted state time")

    state_path = output_dir / "stage1_accepted_state.checkpoint"
    manifest_path = output_dir / "stage1_accepted_state.manifest.json"
    if state_path.exists() or manifest_path.exists():
        raise CampaignProtocolError("Stage-1 checkpoint paths already exist")
    state_sha = hashlib.sha256(state_payload).hexdigest()
    report_bytes = _canonical_json(report.to_json_dict())
    manifest = {
        "schema_version": "s1_stage1_checkpoint_manifest_v1",
        "case_id": config.case_id,
        "campaign_config_sha256": config.sha256,
        "codec_id": codec.codec_id,
        "state_time_s": state_time,
        "state_sha256": state_sha,
        "stability_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "stage1_air_boundary": "closed_wall",
        "manual_acceptance_recorded": True,
        "stage2_must_receive_same_in_memory_state_without_reinitialization": True,
        "checkpoint_is_result_evidence": False,
    }
    manifest_bytes = _canonical_json(manifest)
    _atomic_write(state_path, state_payload)
    _atomic_write(manifest_path, manifest_bytes)
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != state_sha:
        raise CampaignProtocolError("written Stage-1 checkpoint failed its SHA-256 verification")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
        raise CampaignProtocolError("written checkpoint manifest failed SHA-256 verification")
    return Stage1Checkpoint(
        state_path=state_path,
        manifest_path=manifest_path,
        state_sha256=state_sha,
        manifest_sha256=manifest_sha,
        state_time_s=state_time,
        codec_id=codec.codec_id,
    )


def run_preproduction_smoke(
    *,
    config: CampaignProtocolConfig,
    runner: ExactAdvanceRunner,
    codec: StateCodec,
    authorization_dir: Path | str,
    accepted_step_callback: AcceptedStepCallback | None = None,
) -> PreproductionSmokeResult:
    """Run the hard-limited closed-port smoke without promoting the model."""

    markers = _MarkerStore(config, Path(authorization_dir), None)
    markers.assert_smoke_authorized()
    state = runner.source_initial_state(config.source_contract_path)
    _assert_exact_time(codec, state, 0.0, "source initial")
    target = config.smoke_duration_s
    advanced = runner.advance_exact(
        state,
        target_absolute_time_s=target,
        boundary=config.stage1.boundary,
        accepted_step_callback=accepted_step_callback,
    )
    _assert_exact_time(codec, advanced, target, "preproduction smoke")
    return PreproductionSmokeResult(state=advanced, duration_s=target)


def run_formal_campaign(
    *,
    config: CampaignProtocolConfig,
    runner: ExactAdvanceRunner,
    codec: StateCodec,
    observation_bridge: ObservationBridge,
    authorization_dir: Path | str,
    output_dir: Path | str,
    stage1_acceptance_callback: Stage1AcceptanceCallback,
    accepted_step_callback: AcceptedStepCallback | None = None,
    common_state_callback: CommonStateCallback | None = None,
) -> FormalCampaignResult:
    """Run Stage 1 to accepted steady state, then the unshifted 25 s Stage 2.

    This function performs no eruption classification and writes no result
    acceptance marker.  Successful solver completion is explicitly unvalidated.
    """

    output = Path(output_dir).resolve()
    markers = _MarkerStore(config, Path(authorization_dir), output)
    markers.assert_formal_authorized(runner)
    output.mkdir(parents=True, exist_ok=True)

    state = runner.source_initial_state(config.source_contract_path)
    _assert_exact_time(codec, state, 0.0, "source initial")
    observations: list[Stage1Observation] = []
    initial_observation = observation_bridge.observe_stage1(
        state,
        stage1_time_s=0.0,
        boundary=config.stage1.boundary,
    )
    if not math.isclose(initial_observation.stage1_time_s, 0.0, abs_tol=1.0e-14):
        raise CampaignProtocolError("Stage-1 observation bridge shifted the source time")
    observations.append(initial_observation)

    minimum_index = int(round(config.stage1.minimum_physical_time_s / config.stage1.sample_interval_s))
    recheck_steps = int(round(config.stage1.candidate_recheck_interval_s / config.stage1.sample_interval_s))
    maximum_index = int(round(config.stage1.maximum_declared_guard_time_s / config.stage1.sample_interval_s))
    for numerator, denominator, label in (
        (config.stage1.minimum_physical_time_s, config.stage1.sample_interval_s, "minimum time"),
        (config.stage1.candidate_recheck_interval_s, config.stage1.sample_interval_s, "recheck interval"),
        (config.stage1.maximum_declared_guard_time_s, config.stage1.sample_interval_s, "guard time"),
    ):
        if not math.isclose(numerator / denominator, round(numerator / denominator), rel_tol=0.0, abs_tol=1.0e-12):
            raise CampaignProtocolError(f"Stage-1 {label} is not integral on the common grid")

    accepted_report: Stage1StabilityReport | None = None
    for index in range(1, maximum_index + 1):
        target = index * config.stage1.sample_interval_s
        state = runner.advance_exact(
            state,
            target_absolute_time_s=target,
            boundary=config.stage1.boundary,
            accepted_step_callback=None,
        )
        _assert_exact_time(codec, state, target, "Stage-1 exact advance")
        observation = observation_bridge.observe_stage1(
            state,
            stage1_time_s=target,
            boundary=config.stage1.boundary,
        )
        if not math.isclose(observation.stage1_time_s, target, rel_tol=0.0, abs_tol=2.0e-12):
            raise CampaignProtocolError("Stage-1 observation bridge shifted an accepted state")
        observations.append(observation)
        if index < minimum_index or (index - minimum_index) % recheck_steps != 0:
            continue
        report = evaluate_stage1_stability(observations, config.stage1)
        if not report.stable_candidate:
            continue
        candidate = Stage1AcceptanceCandidate(
            state=state,
            report=report,
            campaign_config_sha256=config.sha256,
        )
        if stage1_acceptance_callback(candidate) is not True:
            raise CampaignProtocolError(
                "stable Stage-1 candidate was not manually accepted; no checkpoint or Stage 2 was created"
            )
        accepted_report = report
        break
    if accepted_report is None:
        raise CampaignProtocolError(
            "Stage 1 did not produce an accepted physical steady-state candidate within the declared 120 s guard"
        )

    stage1_accepted_state = state
    stage1_end = _time(codec, stage1_accepted_state, "Stage-1 accepted")
    checkpoint = _write_checkpoint(output, codec, stage1_accepted_state, accepted_report, config)
    report_sha = hashlib.sha256(_canonical_json(accepted_report.to_json_dict())).hexdigest()
    markers.write_progress(
        config.markers.stage1_steady_accepted,
        {
            "stage1_end_absolute_time_s": stage1_end,
            "decision": accepted_report.decision,
            "manual_acceptance": True,
            "stability_report_sha256": report_sha,
            "fixed_time_claimed_as_published": False,
        },
    )
    markers.write_progress(
        config.markers.stage1_checkpoint_accepted,
        {
            "state_sha256": checkpoint.state_sha256,
            "manifest_sha256": checkpoint.manifest_sha256,
            "codec_id": checkpoint.codec_id,
        },
    )

    # The accepted object is passed directly across the boundary switch.  The
    # checkpoint decoder was used only to verify persistence; its object never
    # replaces this state and source_initial_state is never called again.
    stage2_state = stage1_accepted_state
    same_object_at_opening = stage2_state is stage1_accepted_state
    if not same_object_at_opening:
        raise CampaignProtocolError("Stage-2 opening state was reinitialized")
    if common_state_callback is not None:
        common_state_callback(
            AcceptedCommonState(
                physical_stage="stage2_pressure_reservoir",
                stage_time_s=0.0,
                absolute_time_s=stage1_end,
                state=stage2_state,
            )
        )
    markers.write_progress(
        config.markers.stage2_started,
        {
            "stage2_time_origin_absolute_s": stage1_end,
            "stage2_time_s": 0.0,
            "air_gauge_pressure_pa": config.stage2.boundary.air_gauge_pressure_pa,
            "same_in_memory_stage1_state": same_object_at_opening,
            "source_reinitialization": False,
            "source_initialization_call_count_by_controller": 1,
        },
    )

    common_steps = int(
        round(config.stage2.planned_duration_s / config.stage2.common_event_ceiling_interval_s)
    )
    if not math.isclose(
        common_steps * config.stage2.common_event_ceiling_interval_s,
        config.stage2.planned_duration_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CampaignProtocolError("Stage-2 duration is not integral on the 0.1 s common grid")
    for index in range(1, common_steps + 1):
        stage_time = index * config.stage2.common_event_ceiling_interval_s
        target = stage1_end + stage_time
        stage2_state = runner.advance_exact(
            stage2_state,
            target_absolute_time_s=target,
            boundary=config.stage2.boundary,
            accepted_step_callback=accepted_step_callback,
        )
        _assert_exact_time(codec, stage2_state, target, "Stage-2 exact advance")
        if common_state_callback is not None:
            common_state_callback(
                AcceptedCommonState(
                    physical_stage="stage2_pressure_reservoir",
                    stage_time_s=stage_time,
                    absolute_time_s=target,
                    state=stage2_state,
                )
            )

    markers.write_progress(
        config.markers.only_terminal_campaign_marker,
        {
            "stage1_checkpoint_sha256": checkpoint.state_sha256,
            "stage2_planned_duration_s": config.stage2.planned_duration_s,
            "stage2_final_absolute_time_s": _time(codec, stage2_state, "Stage-2 final"),
            "common_event_ceiling_interval_s": config.stage2.common_event_ceiling_interval_s,
            "common_state_count_including_t0": common_steps + 1,
            "solver_run_complete": True,
            "eruption_classified": False,
            "result_accepted": False,
            "validation_required_next": True,
        },
    )
    return FormalCampaignResult(
        state=stage2_state,
        stage1_report=accepted_report,
        checkpoint=checkpoint,
        stage1_end_absolute_time_s=stage1_end,
        stage2_duration_s=config.stage2.planned_duration_s,
        common_state_count=common_steps + 1,
        same_in_memory_state_used_at_stage2_opening=same_object_at_opening,
    )


__all__ = [
    "CampaignProtocolError",
    "FormalCampaignResult",
    "PreproductionSmokeResult",
    "Stage1Checkpoint",
    "run_formal_campaign",
    "run_preproduction_smoke",
]
