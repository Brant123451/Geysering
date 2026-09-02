import copy
import json
from pathlib import Path

import pytest

from campaign.config import (
    CampaignConfigError,
    DEFAULT_CONFIG_PATH,
    load_campaign_protocol_config,
)
from campaign.contracts import Stage1BoundaryFlows, Stage1Observation
from campaign.stability import evaluate_stage1_stability


CHANNELS = ("P1", "P2", "P3", "P4", "P5", "P6")


def _observation(time_s: float, *, pressure_slope: float = 0.0) -> Stage1Observation:
    q = 1.0e-4
    mdot = 998.4 * q
    return Stage1Observation(
        stage1_time_s=time_s,
        gauge_pressures_pa={name: 1000.0 + pressure_slope * time_s for name in CHANNELS},
        velocity_vectors_m_s={name: (0.05, 0.0, 0.0) for name in CHANNELS},
        boundary_flows=Stage1BoundaryFlows(
            qin_m3_s=q,
            qout_m3_s=q,
            mdot_in_kg_s=mdot,
            mdot_out_kg_s=mdot,
        ),
    )


def test_campaign_config_mechanically_freezes_timing_thresholds_and_markers() -> None:
    config = load_campaign_protocol_config()
    assert config.stage1.minimum_physical_time_s == 16.0
    assert config.stage1.terminal_window_s == 4.0
    assert config.stage1.sample_interval_s == 0.1
    assert config.stage1.pressure.maximum_absolute_slope_pa_s == 0.5
    assert config.stage1.velocity.maximum_slope_norm_m_s2 == 0.0005
    assert config.stage1.boundary_flow.maximum_half_window_relative_mean_change == 0.01
    assert config.stage1.balance.maximum_p95_instantaneous_relative_imbalance == 0.02
    assert config.frozen_2d_stage1_gate_sha256 == (
        "33800c9549ceaa262adc7809524ae17f18ac3b23b0d979861d3ebaec44724a47"
    )
    assert config.stage2.boundary.air_gauge_pressure_pa == 5700.0
    assert config.stage2.planned_duration_s == 25.0
    assert config.stage2.common_event_ceiling_interval_s == 0.1
    assert config.markers.only_terminal_campaign_marker == "RUN_COMPLETE_UNVALIDATED"
    assert "RESULT_ACCEPTED" in config.markers.forbidden_markers


def test_threshold_drift_fails_before_a_campaign_can_run(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["stage1"]["thresholds"]["pressure"]["maximum_absolute_slope_pa_s"] = 0.6
    canonical = load_campaign_protocol_config()
    payload["source_contract"] = str(canonical.source_contract_path)
    payload["stage1"]["frozen_2d_gate_reference"]["path"] = str(
        canonical.frozen_2d_stage1_gate_path
    )
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignConfigError, match="drifted"):
        load_campaign_protocol_config(path)


def test_every_1d_stage1_gate_value_is_mechanically_equal_to_frozen_2d_gate() -> None:
    config = load_campaign_protocol_config()
    frozen = json.loads(config.frozen_2d_stage1_gate_path.read_text(encoding="utf-8"))
    one_d = json.loads(config.path.read_text(encoding="utf-8"))["stage1"]
    comparisons = (
        (one_d["minimum_physical_time_s"], frozen["coverage"]["minimum_stage1_time_s"]),
        (one_d["terminal_window_s"], frozen["coverage"]["terminal_window_s"]),
        (one_d["sample_interval_s"], frozen["coverage"]["declared_saved_field_interval_s"]),
        (
            one_d["thresholds"]["pressure"]["maximum_absolute_slope_pa_s"],
            frozen["thresholds"]["pressure"]["maximum_absolute_slope_pa_per_s"],
        ),
        (
            one_d["thresholds"]["pressure"]["maximum_half_window_mean_shift_pa"],
            frozen["thresholds"]["pressure"]["maximum_half_window_mean_shift_pa"],
        ),
        (
            one_d["thresholds"]["pressure"]["maximum_detrended_peak_to_peak_pa"],
            frozen["thresholds"]["pressure"]["maximum_detrended_peak_to_peak_pa"],
        ),
        (
            one_d["thresholds"]["velocity"]["maximum_slope_norm_m_s2"],
            frozen["thresholds"]["velocity"]["maximum_slope_norm_m_per_s2"],
        ),
        (
            one_d["thresholds"]["velocity"]["maximum_half_window_mean_vector_change_m_s"],
            frozen["thresholds"]["velocity"]["maximum_half_window_mean_vector_change_m_per_s"],
        ),
        (
            one_d["thresholds"]["velocity"]["maximum_detrended_residual_vector_magnitude_m_s"],
            frozen["thresholds"]["velocity"]["maximum_detrended_residual_vector_magnitude_m_per_s"],
        ),
        (
            one_d["thresholds"]["boundary_flow"]["minimum_mean_forward_flow_fraction_of_reference"],
            frozen["thresholds"]["boundary_flow"]["minimum_mean_forward_flow_fraction_of_reference"],
        ),
        (
            one_d["thresholds"]["boundary_flow"]["maximum_half_window_relative_mean_change"],
            frozen["thresholds"]["boundary_flow"]["maximum_half_window_relative_mean_change"],
        ),
        (
            one_d["thresholds"]["boundary_flow"]["maximum_detrended_peak_to_peak_fraction"],
            frozen["thresholds"]["boundary_flow"]["maximum_detrended_peak_to_peak_fraction"],
        ),
        (
            one_d["thresholds"]["boundary_flow"]["volume_flow_denominator_floor_m3_s"],
            frozen["thresholds"]["boundary_flow"]["volume_flow_denominator_floor_m3_per_s"],
        ),
        (
            one_d["thresholds"]["boundary_flow"]["mass_flow_denominator_floor_kg_s"],
            frozen["thresholds"]["boundary_flow"]["mass_flow_denominator_floor_kg_per_s"],
        ),
        (
            one_d["thresholds"]["balance"]["maximum_mean_relative_imbalance"],
            frozen["thresholds"]["balance"]["maximum_mean_relative_imbalance"],
        ),
        (
            one_d["thresholds"]["balance"]["maximum_p95_instantaneous_relative_imbalance"],
            frozen["thresholds"]["balance"]["maximum_p95_instantaneous_relative_imbalance"],
        ),
    )
    assert all(one_d_value == two_d_value for one_d_value, two_d_value in comparisons)
    scale_names = {
        "ideal_head_velocity_m_s": "ideal_head_velocity_m_per_s",
        "reference_volume_flow_m3_s": "reference_volume_flow_m3_per_s",
        "reference_mass_flow_kg_s": "reference_mass_flow_kg_per_s",
        "driving_pressure_difference_pa": "driving_pressure_difference_pa",
        "ideal_advective_time_s": "ideal_advective_time_s",
    }
    for one_d_name, two_d_name in scale_names.items():
        assert one_d["source_and_declared_scales"][one_d_name] == frozen[
            "source_and_declared_scales"
        ][two_d_name]["value"]


def test_stability_gate_reports_scale_and_slope_but_never_auto_accepts() -> None:
    config = load_campaign_protocol_config()
    samples = [_observation(index * 0.1) for index in range(121, 161)]
    samples.insert(0, _observation(12.0))
    report = evaluate_stage1_stability(samples, config.stage1)
    assert report.stable_candidate
    assert report.automatic_acceptance is False
    assert report.thresholds_preregistered is True
    assert report.source_scales["ideal_head_velocity_m_s"] == pytest.approx(
        0.19809088823063
    )
    assert report.pressure_statistics["P1"].slope_per_s == pytest.approx(0.0, abs=1e-12)
    assert report.flow_statistics["qin_m3_s"].normalization_scale == pytest.approx(1.0e-4)
    assert all(check.passed for check in report.checks)


def test_pressure_drift_makes_complete_window_unstable() -> None:
    config = load_campaign_protocol_config()
    samples = [_observation(index * 0.1, pressure_slope=0.6) for index in range(120, 161)]
    report = evaluate_stage1_stability(samples, config.stage1)
    assert report.decision == "UNSTABLE"
    failed = [check for check in report.checks if not check.passed]
    assert any(check.check_id == "pressure.P1.absolute_slope_pa_s" for check in failed)


def test_before_16_seconds_is_inconclusive_not_steady() -> None:
    config = load_campaign_protocol_config()
    samples = [_observation(index * 0.1) for index in range(0, 151)]
    report = evaluate_stage1_stability(samples, config.stage1)
    assert report.decision == "INCONCLUSIVE"
    assert not report.stable_candidate
