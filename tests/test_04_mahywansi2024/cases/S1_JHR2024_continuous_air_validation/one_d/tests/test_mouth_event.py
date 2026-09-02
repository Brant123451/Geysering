import pytest

from model import ContractViolation, TopOutflowEventIntegrator


def test_event_requires_full_persistence_threshold() -> None:
    event = TopOutflowEventIntegrator(required_persistence_s=0.10)
    threshold_q = 3.2175924923e-5
    for _ in range(9):
        snapshot = event.advance(
            dt_s=0.01,
            q_up_out_m3_s=threshold_q,
            connected_water_to_mouth=True,
        )
    assert snapshot.active_persistence_s == pytest.approx(0.09)
    assert not snapshot.event_accepted

    snapshot = event.advance(
        dt_s=0.01,
        q_up_out_m3_s=threshold_q,
        connected_water_to_mouth=True,
    )
    assert snapshot.event_accepted
    assert snapshot.event_onset_s == pytest.approx(0.0)
    assert snapshot.acceptance_time_s == pytest.approx(0.10)
    assert snapshot.window_liquid_outflow_m3 == pytest.approx(3.2175924923e-6)
    assert snapshot.window_mean_liquid_outflow_m3_s == pytest.approx(threshold_q)


def test_gross_upflow_is_not_erased_by_larger_simultaneous_downflow() -> None:
    event = TopOutflowEventIntegrator(required_persistence_s=0.10)
    threshold_q = 3.2175924923e-5
    snapshot = event.advance(
        dt_s=0.10,
        q_up_out_m3_s=threshold_q,
        q_down_in_m3_s=2.0 * threshold_q,
        connected_water_to_mouth=True,
    )
    assert snapshot.event_accepted
    assert snapshot.gross_liquid_outflow_m3 == pytest.approx(3.2175924923e-6)
    assert snapshot.gross_liquid_inflow_m3 == pytest.approx(6.4351849846e-6)
    assert snapshot.net_liquid_outflow_m3 == pytest.approx(-3.2175924923e-6)


def test_topology_break_resets_unaccepted_persistence() -> None:
    event = TopOutflowEventIntegrator(required_persistence_s=0.10)
    threshold_q = 3.2175924923e-5
    event.advance(
        dt_s=0.06,
        q_up_out_m3_s=threshold_q,
        connected_water_to_mouth=True,
    )
    snapshot = event.advance(
        dt_s=0.01,
        q_up_out_m3_s=threshold_q,
        connected_water_to_mouth=False,
    )
    assert snapshot.active_persistence_s == 0.0
    assert not snapshot.event_accepted
    snapshot = event.advance(
        dt_s=0.04,
        q_up_out_m3_s=threshold_q,
        connected_water_to_mouth=True,
    )
    assert snapshot.active_persistence_s == pytest.approx(0.04)
    assert not snapshot.event_accepted


def test_below_frozen_mean_and_volume_threshold_never_triggers() -> None:
    event = TopOutflowEventIntegrator()
    below_threshold = 0.99 * event.minimum_mean_outflow_m3_s
    for _ in range(20):
        snapshot = event.advance(
            dt_s=0.01,
            q_up_out_m3_s=below_threshold,
            connected_water_to_mouth=True,
        )
    assert snapshot.active_persistence_s == pytest.approx(0.20)
    assert snapshot.window_liquid_outflow_m3 < event.minimum_cumulative_outflow_m3
    assert not snapshot.event_accepted


def test_short_high_flow_noise_does_not_trigger() -> None:
    event = TopOutflowEventIntegrator()
    snapshot = event.advance(
        dt_s=0.01,
        q_up_out_m3_s=100.0 * event.minimum_mean_outflow_m3_s,
        connected_water_to_mouth=True,
    )
    assert snapshot.window_liquid_outflow_m3 >= event.minimum_cumulative_outflow_m3
    assert not snapshot.event_accepted


def test_unconnected_outflow_does_not_trigger() -> None:
    event = TopOutflowEventIntegrator()
    snapshot = event.advance(
        dt_s=0.20,
        q_up_out_m3_s=10.0 * event.minimum_mean_outflow_m3_s,
        connected_water_to_mouth=False,
    )
    assert snapshot.gross_liquid_outflow_m3 > event.minimum_cumulative_outflow_m3
    assert snapshot.active_persistence_s == 0.0
    assert snapshot.window_liquid_outflow_m3 == 0.0
    assert not snapshot.event_accepted


def test_connectivity_flag_must_be_an_explicit_boolean() -> None:
    event = TopOutflowEventIntegrator()
    with pytest.raises(ContractViolation, match="explicit boolean"):
        event.advance(
            dt_s=0.01,
            q_up_out_m3_s=event.minimum_mean_outflow_m3_s,
            connected_water_to_mouth=1,
        )
