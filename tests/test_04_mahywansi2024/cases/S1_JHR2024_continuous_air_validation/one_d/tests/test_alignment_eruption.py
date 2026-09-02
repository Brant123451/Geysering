import pytest

from alignment import (
    AlignmentError,
    ERUPTION_MEAN_FLOW_M3_S,
    ERUPTION_VOLUME_M3,
    classify_internal_mouth_event,
)


def _classify(flow, *, connected=None, topology=None, conservation=None):
    count = len(flow)
    time_s = [0.1 * index for index in range(count)]
    return classify_internal_mouth_event(
        time_s,
        flow,
        water_path_connected=connected if connected is not None else [True] * count,
        topology_admissible=topology if topology is not None else [True] * count,
        conservation_admissible=(
            conservation if conservation is not None else [True] * count
        ),
    )


def test_exact_threshold_boundary_is_an_event():
    decision = _classify([ERUPTION_MEAN_FLOW_M3_S] * 2)
    assert decision.eruption_detected is True
    assert decision.classification == "EVENT_DETECTED"
    assert decision.first_onset_s == pytest.approx(0.0)
    assert decision.episodes[0].duration_s == pytest.approx(0.10)
    assert decision.episodes[0].cumulative_liquid_outflow_m3 == pytest.approx(
        ERUPTION_VOLUME_M3
    )


def test_isolated_one_sample_noise_spike_does_not_trigger():
    decision = _classify([0.0, 100.0 * ERUPTION_MEAN_FLOW_M3_S, 0.0])
    assert decision.eruption_detected is False
    assert all(not window.qualifies for window in decision.windows)


def test_connected_path_must_persist_for_the_full_window():
    decision = _classify(
        [2.0 * ERUPTION_MEAN_FLOW_M3_S] * 2,
        connected=[True, False],
    )
    assert decision.eruption_detected is False
    assert decision.windows[0].mean_threshold_met is True
    assert decision.windows[0].water_path_connected is False


def test_no_event_is_reported_without_inventing_a_tolerance():
    decision = _classify([0.0, 0.0, 0.0])
    payload = decision.to_dict()
    assert payload["classification"] == "NO_EVENT"
    assert payload["event_count"] == 0
    assert payload["episodes"] == ()
    assert payload["thresholds"]["minimum_mean_liquid_outflow_m3_s"] == pytest.approx(
        ERUPTION_MEAN_FLOW_M3_S
    )


def test_topology_masks_require_explicit_booleans():
    with pytest.raises(AlignmentError, match="explicit booleans"):
        _classify(
            [2.0 * ERUPTION_MEAN_FLOW_M3_S] * 2,
            connected=[True, "false"],
        )
