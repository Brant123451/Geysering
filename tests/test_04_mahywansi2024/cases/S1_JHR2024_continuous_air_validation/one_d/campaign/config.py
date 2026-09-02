"""Mechanical loader for the preregistered S1 campaign protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import BoundaryCommand


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "S1_CAMPAIGN_PROTOCOL.json"


class CampaignConfigError(ValueError):
    """A campaign configuration drifted from the preregistered contract."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CampaignConfigError(f"{label} must be a JSON object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CampaignConfigError(f"{label} must be finite and positive")
    return result


def _expect(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        try:
            value = float(actual)
        except (TypeError, ValueError) as exc:
            raise CampaignConfigError(f"{label} is not numeric") from exc
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise CampaignConfigError(
                f"{label} drifted: expected {expected!r}, found {actual!r}"
            )
    elif actual != expected:
        raise CampaignConfigError(
            f"{label} drifted: expected {expected!r}, found {actual!r}"
        )


@dataclass(frozen=True, slots=True)
class PressureThresholds:
    maximum_absolute_slope_pa_s: float
    maximum_half_window_mean_shift_pa: float
    maximum_detrended_peak_to_peak_pa: float


@dataclass(frozen=True, slots=True)
class VelocityThresholds:
    maximum_slope_norm_m_s2: float
    maximum_half_window_mean_vector_change_m_s: float
    maximum_detrended_residual_vector_magnitude_m_s: float


@dataclass(frozen=True, slots=True)
class BoundaryFlowThresholds:
    minimum_mean_forward_flow_fraction_of_reference: float
    maximum_half_window_relative_mean_change: float
    maximum_detrended_peak_to_peak_fraction: float
    volume_flow_denominator_floor_m3_s: float
    mass_flow_denominator_floor_kg_s: float


@dataclass(frozen=True, slots=True)
class BalanceThresholds:
    maximum_mean_relative_imbalance: float
    maximum_p95_instantaneous_relative_imbalance: float


@dataclass(frozen=True, slots=True)
class Stage1SourceScales:
    ideal_head_velocity_m_s: float
    reference_volume_flow_m3_s: float
    reference_mass_flow_kg_s: float
    driving_pressure_difference_pa: float
    ideal_advective_time_s: float


@dataclass(frozen=True, slots=True)
class Stage1Config:
    boundary: BoundaryCommand
    sample_interval_s: float
    minimum_physical_time_s: float
    terminal_window_s: float
    candidate_recheck_interval_s: float
    maximum_declared_guard_time_s: float
    required_pressure_channels: tuple[str, ...]
    required_velocity_channels: tuple[str, ...]
    scales: Stage1SourceScales
    pressure: PressureThresholds
    velocity: VelocityThresholds
    boundary_flow: BoundaryFlowThresholds
    balance: BalanceThresholds


@dataclass(frozen=True, slots=True)
class Stage2Config:
    boundary: BoundaryCommand
    planned_duration_s: float
    common_event_ceiling_interval_s: float


@dataclass(frozen=True, slots=True)
class MarkerConfig:
    authorization_schema: str
    preproduction_smoke_authorization: str
    model_implementation_acceptance: str
    formal_campaign_authorization: str
    stage1_steady_accepted: str
    stage1_checkpoint_accepted: str
    stage2_started: str
    only_terminal_campaign_marker: str
    forbidden_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignProtocolConfig:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    case_id: str
    source_contract_path: Path
    frozen_2d_stage1_gate_path: Path
    frozen_2d_stage1_gate_sha256: str
    stage1: Stage1Config
    stage2: Stage2Config
    smoke_duration_s: float
    markers: MarkerConfig


def _boundary(raw: Mapping[str, Any], stage: str) -> BoundaryCommand:
    return BoundaryCommand(
        physical_stage=stage,  # type: ignore[arg-type]
        air_port_mode=str(raw.get("air_port_mode")),  # type: ignore[arg-type]
        air_gauge_pressure_pa=raw.get("air_gauge_pressure_pa"),
        evidence_status=str(raw.get("evidence_status", "")),
    )


def load_campaign_protocol_config(
    path: Path | str | None = None,
) -> CampaignProtocolConfig:
    """Load and fail closed on any science-relevant preregistration drift."""

    source = Path(path or DEFAULT_CONFIG_PATH).resolve()
    if not source.is_file():
        raise CampaignConfigError(f"missing campaign protocol: {source}")
    payload_bytes = source.read_bytes()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignConfigError("campaign protocol is not valid UTF-8 JSON") from exc
    root = _mapping(payload, "campaign protocol")

    _expect(root.get("schema_version"), "s1_1d_campaign_protocol_v1", "schema_version")
    _expect(root.get("case_id"), "S1_JHR2024_continuous_air_validation", "case_id")
    _expect(root.get("physical_condition_count"), 1, "physical_condition_count")
    registration = _mapping(root.get("registration"), "registration")
    _expect(
        registration.get("status"),
        "FROZEN_BEFORE_ANY_FORMAL_1D_TRAJECTORY",
        "registration.status",
    )

    stage1_raw = _mapping(root.get("stage1"), "stage1")
    stage2_raw = _mapping(root.get("stage2"), "stage2")
    smoke_raw = _mapping(root.get("preproduction_smoke"), "preproduction_smoke")
    marker_raw = _mapping(root.get("markers"), "markers")

    preregistered = {
        "stage1.sample_interval_s": (stage1_raw.get("sample_interval_s"), 0.1),
        "stage1.minimum_physical_time_s": (
            stage1_raw.get("minimum_physical_time_s"),
            16.0,
        ),
        "stage1.terminal_window_s": (stage1_raw.get("terminal_window_s"), 4.0),
        "stage1.candidate_recheck_interval_s": (
            stage1_raw.get("candidate_recheck_interval_s"),
            4.0,
        ),
        "stage1.maximum_declared_guard_time_s": (
            stage1_raw.get("maximum_declared_guard_time_s"),
            120.0,
        ),
        "stage1.fixed_published_settling_time_claimed": (
            stage1_raw.get("fixed_published_settling_time_claimed"),
            False,
        ),
        "stage1.manual_acceptance_required": (
            stage1_raw.get("manual_acceptance_required"),
            True,
        ),
        "stage2.reuse_same_state": (
            stage2_raw.get("reuse_same_stage1_accepted_state_without_reinitialization"),
            True,
        ),
        "stage2.planned_duration_s": (stage2_raw.get("planned_duration_s"), 25.0),
        "stage2.common_event_ceiling_interval_s": (
            stage2_raw.get("common_event_ceiling_interval_s"),
            0.1,
        ),
        "stage2.state_interpolation_allowed": (
            stage2_raw.get("state_interpolation_allowed"),
            False,
        ),
        "stage2.result_dependent_threshold_adjustment_allowed": (
            stage2_raw.get("result_dependent_threshold_adjustment_allowed"),
            False,
        ),
        "smoke.duration_s": (smoke_raw.get("duration_s"), 0.02),
        "smoke.operator_production_ready_required": (
            smoke_raw.get("operator_production_ready_required"),
            False,
        ),
        "smoke.may_create_campaign_completion_marker": (
            smoke_raw.get("may_create_campaign_completion_marker"),
            False,
        ),
    }
    for label, (actual, expected) in preregistered.items():
        _expect(actual, expected, label)

    pressures = tuple(str(item) for item in stage1_raw.get("required_pressure_channels", ()))
    velocities = tuple(str(item) for item in stage1_raw.get("required_velocity_channels", ()))
    expected_channels = ("P1", "P2", "P3", "P4", "P5", "P6")
    _expect(pressures, expected_channels, "required_pressure_channels")
    _expect(velocities, expected_channels, "required_velocity_channels")

    thresholds = _mapping(stage1_raw.get("thresholds"), "stage1.thresholds")
    scales_raw = _mapping(
        stage1_raw.get("source_and_declared_scales"),
        "stage1.source_and_declared_scales",
    )
    pressure_raw = _mapping(thresholds.get("pressure"), "pressure thresholds")
    velocity_raw = _mapping(thresholds.get("velocity"), "velocity thresholds")
    flow_raw = _mapping(thresholds.get("boundary_flow"), "boundary-flow thresholds")
    balance_raw = _mapping(thresholds.get("balance"), "balance thresholds")

    frozen_ref = _mapping(
        stage1_raw.get("frozen_2d_gate_reference"),
        "stage1.frozen_2d_gate_reference",
    )
    frozen_relative = frozen_ref.get("path")
    if not isinstance(frozen_relative, str) or not frozen_relative.strip():
        raise CampaignConfigError("frozen 2-D Stage-1 gate path is missing")
    frozen_path = (source.parent / frozen_relative).resolve()
    if not frozen_path.is_file():
        raise CampaignConfigError(f"missing frozen 2-D Stage-1 gate: {frozen_path}")
    frozen_bytes = frozen_path.read_bytes()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    _expect(
        frozen_ref.get("sha256"),
        "33800c9549ceaa262adc7809524ae17f18ac3b23b0d979861d3ebaec44724a47",
        "frozen 2-D Stage-1 gate registered SHA-256",
    )
    _expect(frozen_sha, str(frozen_ref.get("sha256")), "frozen 2-D Stage-1 gate file SHA-256")
    try:
        frozen_gate = json.loads(frozen_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignConfigError("frozen 2-D Stage-1 gate is not valid JSON") from exc
    frozen_root = _mapping(frozen_gate, "frozen 2-D Stage-1 gate")
    _expect(
        frozen_root.get("schema_version"),
        "case3_stage1_stability_gate_v1",
        "frozen 2-D Stage-1 gate schema",
    )
    frozen_coverage = _mapping(frozen_root.get("coverage"), "frozen 2-D coverage")
    frozen_thresholds = _mapping(
        frozen_root.get("thresholds"), "frozen 2-D thresholds"
    )
    frozen_pressure = _mapping(frozen_thresholds.get("pressure"), "frozen 2-D pressure")
    frozen_velocity = _mapping(frozen_thresholds.get("velocity"), "frozen 2-D velocity")
    frozen_flow = _mapping(frozen_thresholds.get("boundary_flow"), "frozen 2-D flow")
    frozen_balance = _mapping(frozen_thresholds.get("balance"), "frozen 2-D balance")
    frozen_scales = _mapping(
        frozen_root.get("source_and_declared_scales"), "frozen 2-D source scales"
    )
    cross_gate_values = {
        "coverage.minimum_stage1_time_s": (
            stage1_raw.get("minimum_physical_time_s"),
            frozen_coverage.get("minimum_stage1_time_s"),
        ),
        "coverage.terminal_window_s": (
            stage1_raw.get("terminal_window_s"),
            frozen_coverage.get("terminal_window_s"),
        ),
        "coverage.declared_saved_field_interval_s": (
            stage1_raw.get("sample_interval_s"),
            frozen_coverage.get("declared_saved_field_interval_s"),
        ),
        "pressure.maximum_absolute_slope": (
            pressure_raw.get("maximum_absolute_slope_pa_s"),
            frozen_pressure.get("maximum_absolute_slope_pa_per_s"),
        ),
        "pressure.maximum_half_window_mean_shift": (
            pressure_raw.get("maximum_half_window_mean_shift_pa"),
            frozen_pressure.get("maximum_half_window_mean_shift_pa"),
        ),
        "pressure.maximum_detrended_peak_to_peak": (
            pressure_raw.get("maximum_detrended_peak_to_peak_pa"),
            frozen_pressure.get("maximum_detrended_peak_to_peak_pa"),
        ),
        "velocity.maximum_slope_norm": (
            velocity_raw.get("maximum_slope_norm_m_s2"),
            frozen_velocity.get("maximum_slope_norm_m_per_s2"),
        ),
        "velocity.maximum_half_window_mean_vector_change": (
            velocity_raw.get("maximum_half_window_mean_vector_change_m_s"),
            frozen_velocity.get("maximum_half_window_mean_vector_change_m_per_s"),
        ),
        "velocity.maximum_detrended_residual_vector_magnitude": (
            velocity_raw.get("maximum_detrended_residual_vector_magnitude_m_s"),
            frozen_velocity.get("maximum_detrended_residual_vector_magnitude_m_per_s"),
        ),
        "flow.minimum_mean_forward_flow_fraction": (
            flow_raw.get("minimum_mean_forward_flow_fraction_of_reference"),
            frozen_flow.get("minimum_mean_forward_flow_fraction_of_reference"),
        ),
        "flow.maximum_half_window_relative_mean_change": (
            flow_raw.get("maximum_half_window_relative_mean_change"),
            frozen_flow.get("maximum_half_window_relative_mean_change"),
        ),
        "flow.maximum_detrended_peak_to_peak_fraction": (
            flow_raw.get("maximum_detrended_peak_to_peak_fraction"),
            frozen_flow.get("maximum_detrended_peak_to_peak_fraction"),
        ),
        "flow.volume_floor": (
            flow_raw.get("volume_flow_denominator_floor_m3_s"),
            frozen_flow.get("volume_flow_denominator_floor_m3_per_s"),
        ),
        "flow.mass_floor": (
            flow_raw.get("mass_flow_denominator_floor_kg_s"),
            frozen_flow.get("mass_flow_denominator_floor_kg_per_s"),
        ),
        "balance.maximum_mean_relative_imbalance": (
            balance_raw.get("maximum_mean_relative_imbalance"),
            frozen_balance.get("maximum_mean_relative_imbalance"),
        ),
        "balance.maximum_p95_instantaneous_relative_imbalance": (
            balance_raw.get("maximum_p95_instantaneous_relative_imbalance"),
            frozen_balance.get("maximum_p95_instantaneous_relative_imbalance"),
        ),
    }
    for label, (one_d_value, two_d_value) in cross_gate_values.items():
        try:
            expected = float(two_d_value)
        except (TypeError, ValueError) as exc:
            raise CampaignConfigError(f"frozen 2-D gate lacks {label}") from exc
        _expect(one_d_value, expected, f"1-D/2-D Stage-1 gate {label}")
    exact_thresholds = {
        "pressure.maximum_absolute_slope_pa_s": (
            pressure_raw.get("maximum_absolute_slope_pa_s"),
            0.5,
        ),
        "pressure.maximum_half_window_mean_shift_pa": (
            pressure_raw.get("maximum_half_window_mean_shift_pa"),
            2.0,
        ),
        "pressure.maximum_detrended_peak_to_peak_pa": (
            pressure_raw.get("maximum_detrended_peak_to_peak_pa"),
            10.0,
        ),
        "velocity.maximum_slope_norm_m_s2": (
            velocity_raw.get("maximum_slope_norm_m_s2"),
            0.0005,
        ),
        "velocity.maximum_half_window_mean_vector_change_m_s": (
            velocity_raw.get("maximum_half_window_mean_vector_change_m_s"),
            0.002,
        ),
        "velocity.maximum_detrended_residual_vector_magnitude_m_s": (
            velocity_raw.get("maximum_detrended_residual_vector_magnitude_m_s"),
            0.004,
        ),
        "flow.minimum_mean_forward_flow_fraction_of_reference": (
            flow_raw.get("minimum_mean_forward_flow_fraction_of_reference"),
            0.05,
        ),
        "flow.maximum_half_window_relative_mean_change": (
            flow_raw.get("maximum_half_window_relative_mean_change"),
            0.01,
        ),
        "flow.maximum_detrended_peak_to_peak_fraction": (
            flow_raw.get("maximum_detrended_peak_to_peak_fraction"),
            0.02,
        ),
        "flow.volume_flow_denominator_floor_m3_s": (
            flow_raw.get("volume_flow_denominator_floor_m3_s"),
            0.00000501870673037635,
        ),
        "flow.mass_flow_denominator_floor_kg_s": (
            flow_raw.get("mass_flow_denominator_floor_kg_s"),
            0.00501067679960775,
        ),
        "balance.maximum_mean_relative_imbalance": (
            balance_raw.get("maximum_mean_relative_imbalance"),
            0.01,
        ),
        "balance.maximum_p95_instantaneous_relative_imbalance": (
            balance_raw.get("maximum_p95_instantaneous_relative_imbalance"),
            0.02,
        ),
    }
    for label, (actual, expected) in exact_thresholds.items():
        _expect(actual, expected, label)

    exact_scales = {
        "ideal_head_velocity_m_s": 0.19809088823063,
        "reference_volume_flow_m3_s": 0.000100374134607527,
        "reference_mass_flow_kg_s": 0.100213535992155,
        "driving_pressure_difference_pa": 19.588608,
        "ideal_advective_time_s": 15.6493820977307,
    }
    for key, expected in exact_scales.items():
        _expect(scales_raw.get(key), expected, f"source scale {key}")
    frozen_scale_names = {
        "ideal_head_velocity_m_s": "ideal_head_velocity_m_per_s",
        "reference_volume_flow_m3_s": "reference_volume_flow_m3_per_s",
        "reference_mass_flow_kg_s": "reference_mass_flow_kg_per_s",
        "driving_pressure_difference_pa": "driving_pressure_difference_pa",
        "ideal_advective_time_s": "ideal_advective_time_s",
    }
    for one_d_name, two_d_name in frozen_scale_names.items():
        frozen_record = _mapping(
            frozen_scales.get(two_d_name), f"frozen 2-D source scale {two_d_name}"
        )
        _expect(
            scales_raw.get(one_d_name),
            float(frozen_record.get("value")),
            f"1-D/2-D source scale {one_d_name}",
        )

    decision = _mapping(stage1_raw.get("decision_policy"), "stage1.decision_policy")
    _expect(
        decision.get("all_automatic_gates_pass"),
        "STABLE_CANDIDATE_REQUIRES_MANUAL_ACCEPTANCE",
        "stage1 stable decision",
    )
    _expect(decision.get("automatic_acceptance"), False, "stage1 automatic acceptance")

    markers = MarkerConfig(
        authorization_schema=str(marker_raw.get("authorization_schema")),
        preproduction_smoke_authorization=str(
            marker_raw.get("preproduction_smoke_authorization")
        ),
        model_implementation_acceptance=str(
            marker_raw.get("model_implementation_acceptance")
        ),
        formal_campaign_authorization=str(marker_raw.get("formal_campaign_authorization")),
        stage1_steady_accepted=str(marker_raw.get("stage1_steady_accepted")),
        stage1_checkpoint_accepted=str(marker_raw.get("stage1_checkpoint_accepted")),
        stage2_started=str(marker_raw.get("stage2_started")),
        only_terminal_campaign_marker=str(marker_raw.get("only_terminal_campaign_marker")),
        forbidden_markers=tuple(str(item) for item in marker_raw.get("forbidden_markers", ())),
    )
    expected_markers = (
        (markers.authorization_schema, "s1_campaign_authorization_v1"),
        (markers.preproduction_smoke_authorization, "PREPRODUCTION_SMOKE_AUTHORIZED"),
        (markers.model_implementation_acceptance, "MODEL_IMPLEMENTATION_ACCEPTED"),
        (markers.formal_campaign_authorization, "FORMAL_CAMPAIGN_AUTHORIZED"),
        (markers.stage1_steady_accepted, "STAGE1_STEADY_ACCEPTED"),
        (markers.stage1_checkpoint_accepted, "STAGE1_CHECKPOINT_ACCEPTED"),
        (markers.stage2_started, "STAGE2_STARTED"),
        (markers.only_terminal_campaign_marker, "RUN_COMPLETE_UNVALIDATED"),
        (markers.forbidden_markers, ("RUN_COMPLETE", "RESULT_ACCEPTED", "ERUPTION_ACCEPTED")),
    )
    for index, (actual, expected) in enumerate(expected_markers):
        _expect(actual, expected, f"marker contract {index}")

    source_relative = root.get("source_contract")
    if not isinstance(source_relative, str) or not source_relative.strip():
        raise CampaignConfigError("source_contract must be a relative path")
    source_contract_path = (source.parent / source_relative).resolve()
    if not source_contract_path.is_file():
        raise CampaignConfigError(f"missing source-alignment contract: {source_contract_path}")

    stage1 = Stage1Config(
        boundary=_boundary(_mapping(stage1_raw.get("boundary"), "stage1.boundary"), "stage1_closed"),
        sample_interval_s=_finite_positive(stage1_raw["sample_interval_s"], "sample interval"),
        minimum_physical_time_s=_finite_positive(stage1_raw["minimum_physical_time_s"], "minimum Stage-1 time"),
        terminal_window_s=_finite_positive(stage1_raw["terminal_window_s"], "terminal window"),
        candidate_recheck_interval_s=_finite_positive(stage1_raw["candidate_recheck_interval_s"], "candidate recheck interval"),
        maximum_declared_guard_time_s=_finite_positive(stage1_raw["maximum_declared_guard_time_s"], "Stage-1 guard time"),
        required_pressure_channels=pressures,
        required_velocity_channels=velocities,
        scales=Stage1SourceScales(
            **{key: float(scales_raw[key]) for key in exact_scales}
        ),
        pressure=PressureThresholds(
            maximum_absolute_slope_pa_s=float(
                pressure_raw["maximum_absolute_slope_pa_s"]
            ),
            maximum_half_window_mean_shift_pa=float(
                pressure_raw["maximum_half_window_mean_shift_pa"]
            ),
            maximum_detrended_peak_to_peak_pa=float(
                pressure_raw["maximum_detrended_peak_to_peak_pa"]
            ),
        ),
        velocity=VelocityThresholds(
            maximum_slope_norm_m_s2=float(
                velocity_raw["maximum_slope_norm_m_s2"]
            ),
            maximum_half_window_mean_vector_change_m_s=float(
                velocity_raw["maximum_half_window_mean_vector_change_m_s"]
            ),
            maximum_detrended_residual_vector_magnitude_m_s=float(
                velocity_raw["maximum_detrended_residual_vector_magnitude_m_s"]
            ),
        ),
        boundary_flow=BoundaryFlowThresholds(
            minimum_mean_forward_flow_fraction_of_reference=float(
                flow_raw["minimum_mean_forward_flow_fraction_of_reference"]
            ),
            maximum_half_window_relative_mean_change=float(
                flow_raw["maximum_half_window_relative_mean_change"]
            ),
            maximum_detrended_peak_to_peak_fraction=float(
                flow_raw["maximum_detrended_peak_to_peak_fraction"]
            ),
            volume_flow_denominator_floor_m3_s=float(
                flow_raw["volume_flow_denominator_floor_m3_s"]
            ),
            mass_flow_denominator_floor_kg_s=float(
                flow_raw["mass_flow_denominator_floor_kg_s"]
            ),
        ),
        balance=BalanceThresholds(
            maximum_mean_relative_imbalance=float(
                balance_raw["maximum_mean_relative_imbalance"]
            ),
            maximum_p95_instantaneous_relative_imbalance=float(
                balance_raw["maximum_p95_instantaneous_relative_imbalance"]
            ),
        ),
    )
    stage2 = Stage2Config(
        boundary=_boundary(_mapping(stage2_raw.get("boundary"), "stage2.boundary"), "stage2_pressure_reservoir"),
        planned_duration_s=_finite_positive(stage2_raw["planned_duration_s"], "Stage-2 duration"),
        common_event_ceiling_interval_s=_finite_positive(stage2_raw["common_event_ceiling_interval_s"], "common event interval"),
    )
    return CampaignProtocolConfig(
        path=source,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        raw=MappingProxyType(dict(root)),
        case_id=str(root["case_id"]),
        source_contract_path=source_contract_path,
        frozen_2d_stage1_gate_path=frozen_path,
        frozen_2d_stage1_gate_sha256=frozen_sha,
        stage1=stage1,
        stage2=stage2,
        smoke_duration_s=float(smoke_raw["duration_s"]),
        markers=markers,
    )


__all__ = [
    "BalanceThresholds",
    "BoundaryFlowThresholds",
    "CampaignConfigError",
    "CampaignProtocolConfig",
    "DEFAULT_CONFIG_PATH",
    "MarkerConfig",
    "PressureThresholds",
    "Stage1Config",
    "Stage1SourceScales",
    "Stage2Config",
    "VelocityThresholds",
    "load_campaign_protocol_config",
]
