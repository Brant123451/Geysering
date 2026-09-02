from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from campaign import (
    CampaignProtocolError,
    Stage1BoundaryFlows,
    Stage1Observation,
    load_campaign_protocol_config,
    run_formal_campaign,
    run_preproduction_smoke,
)


CHANNELS = ("P1", "P2", "P3", "P4", "P5", "P6")


@dataclass(frozen=True)
class FakeState:
    time_s: float
    serial: int


@dataclass(frozen=True)
class FakeAcceptedStep:
    before: FakeState
    after: FakeState
    boundary_stage: str


class FakeCodec:
    codec_id = "fake-json-v1"

    def time_s(self, state: FakeState) -> float:
        return state.time_s

    def encode(self, state: FakeState) -> bytes:
        return json.dumps(
            {"serial": state.serial, "time_s": state.time_s},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, payload: bytes) -> FakeState:
        value = json.loads(payload.decode("utf-8"))
        return FakeState(time_s=float(value["time_s"]), serial=int(value["serial"]))


class FakeRunner:
    def __init__(self, *, production_ready: bool, exact_error_s: float = 0.0) -> None:
        self.production_ready = production_ready
        self.exact_error_s = exact_error_s
        self.source_calls = 0
        self.source_paths: list[Path] = []
        self.calls: list[dict[str, Any]] = []

    def source_initial_state(self, source_contract_path: Path) -> FakeState:
        self.source_calls += 1
        self.source_paths.append(source_contract_path)
        return FakeState(time_s=0.0, serial=0)

    def advance_exact(
        self,
        state: FakeState,
        *,
        target_absolute_time_s: float,
        boundary: Any,
        accepted_step_callback: Any,
    ) -> FakeState:
        self.calls.append(
            {
                "input_state": state,
                "target": target_absolute_time_s,
                "boundary": boundary,
            }
        )
        advanced = FakeState(
            time_s=target_absolute_time_s + self.exact_error_s,
            serial=state.serial + 1,
        )
        if accepted_step_callback is not None:
            accepted_step_callback(
                FakeAcceptedStep(
                    before=state,
                    after=advanced,
                    boundary_stage=boundary.physical_stage,
                )
            )
        return advanced


class StableBridge:
    def observe_stage1(
        self,
        state: FakeState,
        *,
        stage1_time_s: float,
        boundary: Any,
    ) -> Stage1Observation:
        assert boundary.physical_stage == "stage1_closed"
        assert state.time_s == pytest.approx(stage1_time_s, abs=1e-12)
        q = 1.0e-4
        mdot = 998.4 * q
        return Stage1Observation(
            stage1_time_s=stage1_time_s,
            gauge_pressures_pa={name: 1000.0 for name in CHANNELS},
            velocity_vectors_m_s={name: (0.05, 0.0, 0.0) for name in CHANNELS},
            boundary_flows=Stage1BoundaryFlows(q, q, mdot, mdot),
        )


def _authorize(directory: Path, name: str, config: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(
            {
                "schema_version": config.markers.authorization_schema,
                "case_id": config.case_id,
                "campaign_config_sha256": config.sha256,
                "authorized": True,
            }
        ),
        encoding="utf-8",
    )


def test_nonproduction_operator_may_run_only_authorized_closed_port_smoke(
    tmp_path: Path,
) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    _authorize(auth, config.markers.preproduction_smoke_authorization, config)
    runner = FakeRunner(production_ready=False)
    accepted: list[FakeAcceptedStep] = []
    result = run_preproduction_smoke(
        config=config,
        runner=runner,
        codec=FakeCodec(),
        authorization_dir=auth,
        accepted_step_callback=accepted.append,
    )
    assert result.duration_s == 0.02
    assert result.state.time_s == 0.02
    assert result.completion_marker_written is False
    assert runner.source_calls == 1
    assert len(runner.calls) == 1
    assert runner.calls[0]["boundary"].physical_stage == "stage1_closed"
    assert accepted[0].boundary_stage == "stage1_closed"
    assert not list(tmp_path.rglob("RUN_COMPLETE_UNVALIDATED"))


def test_smoke_fails_closed_without_its_own_authorization(tmp_path: Path) -> None:
    config = load_campaign_protocol_config()
    with pytest.raises(CampaignProtocolError, match="authorization"):
        run_preproduction_smoke(
            config=config,
            runner=FakeRunner(production_ready=False),
            codec=FakeCodec(),
            authorization_dir=tmp_path,
        )


def test_formal_campaign_requires_both_authorizations_and_production_flag(
    tmp_path: Path,
) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    _authorize(auth, config.markers.model_implementation_acceptance, config)
    _authorize(auth, config.markers.formal_campaign_authorization, config)
    with pytest.raises(CampaignProtocolError, match="production_ready"):
        run_formal_campaign(
            config=config,
            runner=FakeRunner(production_ready=False),
            codec=FakeCodec(),
            observation_bridge=StableBridge(),
            authorization_dir=auth,
            output_dir=tmp_path / "run",
            stage1_acceptance_callback=lambda candidate: True,
        )


@pytest.mark.parametrize(
    "present_marker,missing_marker",
    [
        ("MODEL_IMPLEMENTATION_ACCEPTED", "FORMAL_CAMPAIGN_AUTHORIZED"),
        ("FORMAL_CAMPAIGN_AUTHORIZED", "MODEL_IMPLEMENTATION_ACCEPTED"),
    ],
)
def test_formal_campaign_fails_when_either_authorization_layer_is_missing(
    tmp_path: Path, present_marker: str, missing_marker: str
) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    _authorize(auth, present_marker, config)
    with pytest.raises(CampaignProtocolError, match=missing_marker):
        run_formal_campaign(
            config=config,
            runner=FakeRunner(production_ready=True),
            codec=FakeCodec(),
            observation_bridge=StableBridge(),
            authorization_dir=auth,
            output_dir=tmp_path / "run",
            stage1_acceptance_callback=lambda candidate: True,
        )


def test_formal_stage1_checkpoint_same_state_switch_and_exact_stage2_grid(
    tmp_path: Path,
) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    output = tmp_path / "run"
    _authorize(auth, config.markers.model_implementation_acceptance, config)
    _authorize(auth, config.markers.formal_campaign_authorization, config)
    runner = FakeRunner(production_ready=True)
    accepted_candidates: list[Any] = []
    accepted_steps: list[FakeAcceptedStep] = []
    common_states: list[Any] = []

    def accept(candidate: Any) -> bool:
        accepted_candidates.append(candidate)
        return True

    result = run_formal_campaign(
        config=config,
        runner=runner,
        codec=FakeCodec(),
        observation_bridge=StableBridge(),
        authorization_dir=auth,
        output_dir=output,
        stage1_acceptance_callback=accept,
        accepted_step_callback=accepted_steps.append,
        common_state_callback=common_states.append,
    )

    assert runner.source_calls == 1
    assert result.stage1_end_absolute_time_s == 16.0
    assert result.state.time_s == 41.0
    assert result.status == "RUN_COMPLETE_UNVALIDATED"
    assert result.result_accepted is False
    assert result.same_in_memory_state_used_at_stage2_opening is True
    assert len(common_states) == 251
    assert [common_states[index].stage_time_s for index in (0, 1, 249, 250)] == [
        0.0,
        0.1,
        24.900000000000002,
        25.0,
    ]
    assert common_states[0].state is accepted_candidates[0].state

    stage2_calls = [
        call for call in runner.calls if call["boundary"].physical_stage == "stage2_pressure_reservoir"
    ]
    assert len(stage2_calls) == 250
    assert stage2_calls[0]["input_state"] is accepted_candidates[0].state
    assert stage2_calls[0]["boundary"].air_gauge_pressure_pa == 5700.0
    assert [call["target"] - 16.0 for call in stage2_calls[:3]] == pytest.approx(
        [0.1, 0.2, 0.3], abs=2e-15
    )
    assert len(accepted_steps) == 250

    checkpoint_bytes = result.checkpoint.state_path.read_bytes()
    assert hashlib.sha256(checkpoint_bytes).hexdigest() == result.checkpoint.state_sha256
    manifest = json.loads(result.checkpoint.manifest_path.read_text(encoding="utf-8"))
    assert manifest["state_sha256"] == result.checkpoint.state_sha256
    assert manifest["stage2_must_receive_same_in_memory_state_without_reinitialization"] is True

    expected_markers = {
        "STAGE1_STEADY_ACCEPTED",
        "STAGE1_CHECKPOINT_ACCEPTED",
        "STAGE2_STARTED",
        "RUN_COMPLETE_UNVALIDATED",
    }
    marker_names = {path.name for path in output.iterdir() if path.name in expected_markers}
    assert marker_names == expected_markers
    assert not (output / "RUN_COMPLETE").exists()
    assert not (output / "RESULT_ACCEPTED").exists()
    terminal = json.loads((output / "RUN_COMPLETE_UNVALIDATED").read_text(encoding="utf-8"))
    assert terminal["result_accepted"] is False
    assert terminal["eruption_classified"] is False


def test_manual_stage1_rejection_creates_no_checkpoint_or_run_marker(tmp_path: Path) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    output = tmp_path / "run"
    _authorize(auth, config.markers.model_implementation_acceptance, config)
    _authorize(auth, config.markers.formal_campaign_authorization, config)
    with pytest.raises(CampaignProtocolError, match="not manually accepted"):
        run_formal_campaign(
            config=config,
            runner=FakeRunner(production_ready=True),
            codec=FakeCodec(),
            observation_bridge=StableBridge(),
            authorization_dir=auth,
            output_dir=output,
            stage1_acceptance_callback=lambda candidate: False,
        )
    assert not (output / "stage1_accepted_state.checkpoint").exists()
    assert not (output / "STAGE1_STEADY_ACCEPTED").exists()
    assert not (output / "RUN_COMPLETE_UNVALIDATED").exists()


def test_interpolated_or_missed_event_ceiling_is_rejected(tmp_path: Path) -> None:
    config = load_campaign_protocol_config()
    auth = tmp_path / "auth"
    _authorize(auth, config.markers.model_implementation_acceptance, config)
    _authorize(auth, config.markers.formal_campaign_authorization, config)
    with pytest.raises(CampaignProtocolError, match="interpolation is forbidden"):
        run_formal_campaign(
            config=config,
            runner=FakeRunner(production_ready=True, exact_error_s=1.0e-5),
            codec=FakeCodec(),
            observation_bridge=StableBridge(),
            authorization_dir=auth,
            output_dir=tmp_path / "run",
            stage1_acceptance_callback=lambda candidate: True,
        )
