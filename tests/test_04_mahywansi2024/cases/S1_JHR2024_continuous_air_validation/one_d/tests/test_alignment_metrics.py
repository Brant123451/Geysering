import pytest

from alignment import (
    AlignmentError,
    compare_event_branches,
    compare_one_d_to_three_meshes,
    compute_waveform_metrics,
)
from alignment.events import classify_internal_mouth_event


def test_rmse_and_bias_are_unshifted_and_have_no_auto_tolerance():
    metrics = compute_waveform_metrics(
        [0.0, 0.1, 0.2],
        reference=[0.0, 0.0, 0.0],
        candidate=[1.0, 1.0, 1.0],
    )
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mean_bias_candidate_minus_reference == pytest.approx(1.0)
    assert metrics.time_shift_applied_before_error_metrics_s == 0.0
    assert metrics.automatic_acceptance_applied is False
    assert not any("tolerance" in key for key in metrics.to_dict())


def test_phase_is_diagnostic_only_and_positive_when_candidate_is_late():
    time_s = [0.1 * index for index in range(8)]
    reference = [0.0, 1.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    candidate = [0.0, 0.0, 1.0, 2.0, 0.0, -1.0, 0.0, 0.0]
    metrics = compute_waveform_metrics(time_s, reference, candidate)
    assert metrics.diagnostic_phase_lag_s == pytest.approx(0.1)
    assert metrics.diagnostic_phase_correlation == pytest.approx(1.0)
    assert metrics.rmse > 0.0  # The diagnostic lag was not applied to RMSE.


def test_one_1d_record_is_compared_to_all_three_meshes():
    one_d = {"time_s": [0.0, 0.1, 0.2], "P1": [1.0, 2.0, 3.0]}
    meshes = {
        "coarse": {"time_s": [0.0, 0.1, 0.2], "P1": [1.0, 1.0, 3.0]},
        "medium_refine": {
            "time_s": [0.0, 0.1, 0.2],
            "P1": [1.0, 2.0, 2.5],
        },
        "refined": {"time_s": [0.0, 0.1, 0.2], "P1": [1.0, 2.0, 3.0]},
    }
    result = compare_one_d_to_three_meshes(one_d, meshes, series_names=["P1"])
    assert result["one_d_parameter_set_reused_for_all_meshes"] is True
    assert tuple(result["mesh_results"]) == ("coarse", "medium_refine", "refined")
    assert result["automatic_acceptance_applied"] is False
    assert result["mesh_results"]["refined"]["series_metrics"]["P1"]["rmse"] == 0.0


def test_campaign_rejects_nonidentical_physical_times():
    one_d = {"time_s": [0.0, 0.1], "P1": [1.0, 2.0]}
    meshes = {
        "coarse": {"time_s": [0.1, 0.2], "P1": [1.0, 2.0]},
        "medium_refine": {"time_s": [0.0, 0.1], "P1": [1.0, 2.0]},
        "refined": {"time_s": [0.0, 0.1], "P1": [1.0, 2.0]},
    }
    with pytest.raises(AlignmentError, match="identical physical times"):
        compare_one_d_to_three_meshes(one_d, meshes, series_names=["P1"])


def test_event_branch_comparison_is_exact_for_every_mesh():
    one_d_decision = classify_internal_mouth_event(
        [0.0, 0.1],
        [4.0e-5, 4.0e-5],
        water_path_connected=[True, True],
        topology_admissible=[True, True],
        conservation_admissible=[True, True],
    )
    result = compare_event_branches(
        one_d_decision,
        {"coarse": True, "medium_refine": True, "refined": False},
    )
    assert result["all_mesh_branches_match"] is False
    assert result["exact_branch_match_by_mesh"]["refined"] is False
    assert result["physics_alignment_branch_pass"] is False
    assert result["required_match_rule"] == "exact"


def test_all_models_agreeing_on_no_eruption_still_fails_paper_branch():
    one_d_decision = classify_internal_mouth_event(
        [0.0, 0.1],
        [0.0, 0.0],
        water_path_connected=[True, True],
        topology_admissible=[True, True],
        conservation_admissible=[True, True],
    )
    result = compare_event_branches(
        one_d_decision,
        {"coarse": False, "medium_refine": False, "refined": False},
    )
    assert result["all_mesh_branches_match"] is True
    assert result["paper_expected_erupted"] is True
    assert result["physics_alignment_branch_pass"] is False
    assert result["stable_but_no_eruption_classification"] == "physics_alignment_failure"


def test_event_branch_flags_must_be_explicit_booleans():
    one_d_decision = classify_internal_mouth_event(
        [0.0, 0.1],
        [0.0, 0.0],
        water_path_connected=[True, True],
        topology_admissible=[True, True],
        conservation_admissible=[True, True],
    )
    with pytest.raises(AlignmentError, match="explicit booleans"):
        compare_event_branches(
            one_d_decision,
            {"coarse": True, "medium_refine": "false", "refined": True},
        )
