from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from campaign.config import load_campaign_protocol_config
from campaign.contracts import BoundaryCommand
from campaign.orchestrator import (
    CampaignProtocolError,
    run_formal_campaign,
    run_preproduction_smoke,
)
from model.flux import HorizontalDelta, SupplyBranchDelta, VerticalDelta, state_token
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import (
    AIR_NODE_PORTS,
    RISER_NODE_PORTS,
    CurrentS1PhysicalJointOperator,
    GrossComponentPortFlux,
    JointStageRate,
    S1JointNetworkRunner,
    ZeroStorageTNodeSolution,
    build_current_physical_operator,
)
from model.state import VerticalState
from runtime.campaign_adapter import (
    DEFAULT_SOURCE_CONTRACT_PATH,
    S1CampaignExactAdvanceAdapter,
    S1CampaignRuntimeError,
    S1CoupledStateCodec,
    S1Stage1ObservationBridge,
    _canonical_stage2_tick,
    build_current_s1_campaign_runtime,
)


def _node(name, ports):
    return ZeroStorageTNodeSolution(
        name=name,
        ports=tuple(GrossComponentPortFlux(port) for port in sorted(ports)),
    )


class _NativeZeroOperator:
    """No-motion test owner that still uses the real native P1--P6 closures."""

    validation_only = False

    def __init__(self, *, production_ready: bool, stable_dt_s: float) -> None:
        physical = build_current_physical_operator()
        self.horizontal_component = physical.horizontal_component
        self.vertical_component = physical.vertical_component
        self.production_ready = production_ready
        self.validation_only = not production_ready
        self.stable_dt_s = stable_dt_s

    def stable_timestep_s(self, state, geometry, *, physical_stage):
        del state, geometry, physical_stage
        return self.stable_dt_s

    def diagnostic_node_pressures(
        self, state, geometry, *, physical_stage, diagnostic_dt_s
    ):
        del state, geometry, physical_stage, diagnostic_dt_s
        return 107020.0, 107040.0

    def evaluate(self, state, geometry, *, physical_stage, rk_stage, dt_s):
        del geometry, rk_stage, dt_s
        return JointStageRate(
            physical_stage=physical_stage,
            horizontal=HorizontalDelta.zeros(state.horizontal.cell_count),
            supply_branch=SupplyBranchDelta.zeros(state.supply_branch.cell_count),
            vertical=VerticalDelta.zeros(state.vertical.cell_count),
            air_supply_node=_node("air_supply_T", AIR_NODE_PORTS),
            riser_node=_node("riser_T", RISER_NODE_PORTS),
            air_supply_node_common_absolute_pressure_Pa=107020.0,
            riser_node_common_absolute_pressure_Pa=107040.0,
            evidence_status="synthetic_zero_rate_real_native_diagnostic_contract",
        )


class _SelfReadyHorizontal:
    source_aligned_trajectory_ready = True
    production_ready = True
    joint_trial_ready = True

    def propose_joint_stage(self, *args, **kwargs):
        del args, kwargs


class _SelfReadySupply:
    production_ready = True
    joint_trial_ready = True

    def propose_atomic_step(self, *args, **kwargs):
        del args, kwargs


class _SelfReadyVertical:
    production_ready = True
    joint_trial_ready = True

    def propose_joint_stage(self, *args, **kwargs):
        del args, kwargs


class _SelfReadyNodes:
    algebraic_gate_ready = True


class _SelfReadyOwner:
    integration_owner_ready = True

    def evaluate(self, *args, **kwargs):
        del args, kwargs

    def stable_timestep_s(self, *args, **kwargs):
        del args, kwargs


def _boundary(stage: str) -> BoundaryCommand:
    if stage == "stage1_closed":
        return BoundaryCommand(
            physical_stage="stage1_closed",
            air_port_mode="closed_wall",
            air_gauge_pressure_pa=None,
            evidence_status="test_closed_published_boundary",
        )
    return BoundaryCommand(
        physical_stage="stage2_pressure_reservoir",
        air_port_mode="isothermal_pressure_reservoir",
        air_gauge_pressure_pa=5700.0,
        evidence_status="test_published_Table_1_boundary",
    )


def _authorize(directory: Path, name: str, config) -> None:
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


def test_lossless_codec_roundtrip_preserves_every_source_state_value() -> None:
    assembly = build_s1_initial_assembly()
    codec = S1CoupledStateCodec(assembly.geometry)

    payload = codec.encode(assembly.state)
    decoded = codec.decode(payload)

    assert decoded == assembly.state
    assert decoded is not assembly.state
    assert codec.encode(decoded) == payload
    assert state_token(decoded) == state_token(assembly.state)
    assert codec.time_s(decoded) == 0.0
    assert b'"schema_version":"s1_coupled_state_ieee754_hex_v1"' in payload


def test_codec_rejects_unknown_fields_instead_of_silently_dropping_state() -> None:
    assembly = build_s1_initial_assembly()
    codec = S1CoupledStateCodec(assembly.geometry)
    raw = json.loads(codec.encode(assembly.state))
    raw["unowned_inventory"] = "0x0.0p+0"
    payload = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(S1CampaignRuntimeError, match="fields differ"):
        codec.decode(payload)


def test_codec_rejects_duplicate_noncanonical_and_wrong_geometry_payloads() -> None:
    assembly = build_s1_initial_assembly()
    codec = S1CoupledStateCodec(assembly.geometry)
    payload = codec.encode(assembly.state)

    duplicate = payload.replace(
        b'"geometry_sha256":',
        b'"geometry_sha256":"duplicate","geometry_sha256":',
        1,
    )
    with pytest.raises(S1CampaignRuntimeError, match="duplicate JSON key"):
        codec.decode(duplicate)

    with pytest.raises(S1CampaignRuntimeError, match="canonical encoding"):
        codec.decode(b" " + payload)

    other_geometry = replace(
        assembly.geometry,
        liquid_density_kg_m3=assembly.geometry.liquid_density_kg_m3 + 0.1,
    )
    with pytest.raises(S1CampaignRuntimeError, match="geometry fingerprint"):
        S1CoupledStateCodec(other_geometry).decode(payload)


def test_codec_preserves_signed_zero_in_a_non_source_state() -> None:
    assembly = build_s1_initial_assembly()
    ql = list(assembly.state.horizontal.Ql)
    ql[0] = -0.0
    state = replace(
        assembly.state,
        horizontal=replace(assembly.state.horizontal, Ql=tuple(ql)),
    )
    codec = S1CoupledStateCodec(assembly.geometry)

    payload = codec.encode(state)
    decoded = codec.decode(payload)

    assert b'"-0x0.0p+0"' in payload
    assert decoded.horizontal.Ql[0] == 0.0
    assert math.copysign(1.0, decoded.horizontal.Ql[0]) == -1.0
    assert codec.encode(decoded) == payload


def test_real_owner_two_continuous_microsteps_emit_native_callbacks_and_ledgers() -> None:
    runtime = build_current_s1_campaign_runtime(
        scope="preproduction_validation_only",
        maximum_dt_s=1.0e-7,
        diagnostic_dt_s=1.0e-7,
    )
    adapter = runtime.exact_runner
    seen = []
    state = adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)

    advanced = adapter.advance_exact(
        state,
        target_absolute_time_s=2.0e-7,
        boundary=_boundary("stage1_closed"),
        accepted_step_callback=seen.append,
    )

    assert advanced.time_s == pytest.approx(2.0e-7, abs=2.0e-15)
    assert adapter.source_initialization_count == 1
    assert adapter.accepted_context_count == 2
    assert len(seen) == 2
    assert len(adapter.runner.committer.ledger.entries) == 2
    assert [context.actual_dt_s for context in seen] == pytest.approx([1.0e-7, 1.0e-7])
    assert all(context.physical_stage == "stage1_closed" for context in seen)
    assert all(context.diagnostics.pressure_after is not None for context in seen)
    assert all(context.diagnostics.accepted_native_interval is not None for context in seen)
    assert all(len(context.ledger_entries) == 1 for context in seen)
    ledger_before_observation = tuple(adapter.runner.committer.ledger.entries)
    accepted_observation = runtime.stage1_observation_bridge.observe_stage1(
        advanced,
        stage1_time_s=2.0e-7,
        boundary=_boundary("stage1_closed"),
    )
    assert set(accepted_observation.gauge_pressures_pa) == {
        "P1", "P2", "P3", "P4", "P5", "P6"
    }
    assert tuple(adapter.runner.committer.ledger.entries) == ledger_before_observation
    assert runtime.codec.decode(runtime.codec.encode(advanced)) == advanced
    with pytest.raises(S1CampaignRuntimeError, match="exactly once"):
        adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)


def test_real_stage1_observer_is_read_only_and_returns_native_p1_p6_vectors_flows() -> None:
    runtime = build_current_s1_campaign_runtime(
        scope="preproduction_validation_only",
        maximum_dt_s=1.0e-7,
        diagnostic_dt_s=1.0e-7,
    )
    state = runtime.exact_runner.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)
    token_before = state_token(state)
    ledger_before = tuple(runtime.exact_runner.runner.committer.ledger.entries)

    observation = runtime.stage1_observation_bridge.observe_stage1(
        state,
        stage1_time_s=0.0,
        boundary=_boundary("stage1_closed"),
    )

    assert set(observation.gauge_pressures_pa) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert set(observation.velocity_vectors_m_s) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert observation.velocity_vectors_m_s["P1"][:2] == (0.0, 0.0)
    assert observation.velocity_vectors_m_s["P4"][1:] == (0.0, 0.0)
    assert observation.boundary_flows.qin_m3_s >= 0.0
    assert observation.boundary_flows.qout_m3_s >= 0.0
    assert observation.boundary_flows.mdot_in_kg_s == pytest.approx(
        runtime.assembly.geometry.liquid_density_kg_m3
        * observation.boundary_flows.qin_m3_s
    )
    assert state_token(state) == token_before
    assert tuple(runtime.exact_runner.runner.committer.ledger.entries) == ledger_before


def test_stage1_observer_does_not_cancel_strong_countercurrent_riser_streams() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=1.0e-7)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    count = len(assembly.geometry.vertical_dz_m)
    half_area = 0.5 * assembly.geometry.vertical_area_m2
    vertical = VerticalState(
        Aup=(half_area,) * count,
        Qup=(half_area,) * count,
        Adown=(half_area,) * count,
        Qdown=(half_area,) * count,
        Mg=(0.0,) * count,
        Jg=(0.0,) * count,
    )
    state = replace(assembly.state, vertical=vertical)
    assembly.geometry.validate_state(state)

    observation = S1Stage1ObservationBridge(runner=runner).observe_stage1(
        state,
        stage1_time_s=0.0,
        boundary=_boundary("stage1_closed"),
    )

    for name in ("P1", "P2", "P3"):
        assert observation.velocity_vectors_m_s[name] == pytest.approx((0.0, 0.0, 1.0))


def test_self_declared_synthetic_production_operator_cannot_enter_formal_scope() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=True, stable_dt_s=0.07)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="formal_campaign",
        maximum_dt_s=0.07,
    )
    assert adapter.production_ready is False
    with pytest.raises(S1CampaignRuntimeError, match="CurrentS1PhysicalJointOperator"):
        adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)
    assert adapter.source_initialization_count == 0
    assert joint.committer.ledger.entries == []


def test_true_wrapper_around_self_ready_fake_components_has_no_factory_provenance() -> None:
    assembly = build_s1_initial_assembly()
    operator = CurrentS1PhysicalJointOperator(
        horizontal_component=_SelfReadyHorizontal(),
        supply_branch_component=_SelfReadySupply(),
        vertical_component=_SelfReadyVertical(),
        two_tnode_solver=_SelfReadyNodes(),
        joint_stage_owner=_SelfReadyOwner(),
    )
    assert operator.production_ready is True
    assert operator.integration_owner_ready is True
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="formal_campaign",
        maximum_dt_s=0.01,
    )

    assert adapter.production_ready is False
    with pytest.raises(S1CampaignRuntimeError, match="CurrentS1PhysicalJointOperator"):
        adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)
    assert adapter.source_initialization_count == 0
    assert joint.committer.ledger.entries == []


def test_source_hash_and_initial_assembly_provenance_are_fail_closed(
    tmp_path: Path,
) -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=0.01)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    changed_source = tmp_path / "S1_source_aligned.yaml"
    changed_source.write_bytes(DEFAULT_SOURCE_CONTRACT_PATH.read_bytes() + b"\n")
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="preproduction_validation_only",
        maximum_dt_s=0.01,
        source_contract_path=changed_source,
    )
    with pytest.raises(S1CampaignRuntimeError, match="content hash changed"):
        adapter.source_initial_state(changed_source)
    assert adapter.source_initialization_count == 0

    changed_assembly = replace(
        assembly,
        state=replace(assembly.state, time_s=1.0e-7),
    )
    with pytest.raises(S1CampaignRuntimeError, match="frozen source geometry"):
        S1CampaignExactAdvanceAdapter(
            assembly=changed_assembly,
            runner=joint,
            scope="preproduction_validation_only",
            maximum_dt_s=0.01,
        )


def test_canonical_stage2_tick_accepts_250_constructed_ticks_and_no_epsilon_snap() -> None:
    origin = 16.3
    for index in range(1, 251):
        target = origin + index * 0.1
        assert _canonical_stage2_tick(origin, target) == index
        with pytest.raises(S1CampaignRuntimeError, match="time snapping"):
            _canonical_stage2_tick(origin, math.nextafter(target, math.inf))
        with pytest.raises(S1CampaignRuntimeError, match="time snapping"):
            _canonical_stage2_tick(origin, math.nextafter(target, -math.inf))


def test_zero_step_stage1_cannot_create_accepted_evidence_or_unlock_transition() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=0.011)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="preproduction_validation_only",
        maximum_dt_s=0.011,
    )
    state = adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)
    seen = []

    with pytest.raises(S1CampaignRuntimeError, match="one-to-one accepted callback"):
        adapter.advance_exact(
            state,
            target_absolute_time_s=1.0e-15,
            boundary=_boundary("stage1_closed"),
            accepted_step_callback=seen.append,
        )

    assert adapter.stage1_advance_recorded is False
    assert adapter.accepted_context_count == 0
    assert joint.committer.ledger.entries == []
    assert seen == []
    assert adapter.terminal_fault is not None


def test_callback_failure_keeps_committed_successor_owned_and_faults_terminally() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=0.011)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="preproduction_validation_only",
        maximum_dt_s=0.011,
    )
    state = adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)

    def fail_after_commit(context) -> None:
        assert context.after_state.time_s == pytest.approx(0.011)
        raise RuntimeError("injected journal failure")

    with pytest.raises(RuntimeError, match="injected journal failure"):
        adapter.advance_exact(
            state,
            target_absolute_time_s=0.02,
            boundary=_boundary("stage1_closed"),
            accepted_step_callback=fail_after_commit,
        )

    assert len(joint.committer.ledger.entries) == 1
    assert adapter.accepted_context_count == 1
    assert adapter.last_accepted_context is not None
    assert adapter.last_accepted_state is adapter.last_accepted_context.after_state
    assert adapter.last_accepted_state.time_s == pytest.approx(0.011)
    assert adapter.terminal_fault is not None
    assert "injected journal failure" in adapter.terminal_fault
    with pytest.raises(S1CampaignRuntimeError, match="terminally faulted"):
        adapter.advance_exact(
            adapter.last_accepted_state,
            target_absolute_time_s=0.02,
            boundary=_boundary("stage1_closed"),
            accepted_step_callback=None,
        )
    assert len(joint.committer.ledger.entries) == 1


def test_runtime_rejects_decoded_copy_and_in_place_mutation_of_owned_state() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=0.01)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="preproduction_validation_only",
        maximum_dt_s=0.01,
    )
    state = adapter.source_initial_state(DEFAULT_SOURCE_CONTRACT_PATH)
    advanced = adapter.advance_exact(
        state,
        target_absolute_time_s=0.01,
        boundary=_boundary("stage1_closed"),
        accepted_step_callback=None,
    )
    codec = S1CoupledStateCodec(assembly.geometry)
    copied = codec.decode(codec.encode(advanced))
    ledger_count = len(joint.committer.ledger.entries)
    with pytest.raises(S1CampaignRuntimeError, match="identical last accepted"):
        adapter.advance_exact(
            copied,
            target_absolute_time_s=0.02,
            boundary=_boundary("stage1_closed"),
            accepted_step_callback=None,
        )
    assert len(joint.committer.ledger.entries) == ledger_count

    object.__setattr__(advanced, "time_s", math.nextafter(advanced.time_s, math.inf))
    with pytest.raises(S1CampaignRuntimeError, match="altered in place"):
        adapter.advance_exact(
            advanced,
            target_absolute_time_s=0.02,
            boundary=_boundary("stage1_closed"),
            accepted_step_callback=None,
        )
    assert len(joint.committer.ledger.entries) == ledger_count


def test_explicit_validation_scope_runs_only_authorized_closed_smoke(tmp_path: Path) -> None:
    config = load_campaign_protocol_config()
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=0.011)
    joint = S1JointNetworkRunner(assembly.geometry, operator)
    adapter = S1CampaignExactAdvanceAdapter(
        assembly=assembly,
        runner=joint,
        scope="preproduction_validation_only",
        maximum_dt_s=0.011,
    )
    auth = tmp_path / "auth"
    _authorize(auth, config.markers.preproduction_smoke_authorization, config)

    result = run_preproduction_smoke(
        config=config,
        runner=adapter,
        codec=S1CoupledStateCodec(assembly.geometry),
        authorization_dir=auth,
    )

    assert adapter.production_ready is False
    assert result.state.time_s == pytest.approx(0.02, abs=2.0e-14)
    assert adapter.accepted_context_count == 2
    assert result.completion_marker_written is False
    assert not list(tmp_path.rglob("RUN_COMPLETE_UNVALIDATED"))
    with pytest.raises(S1CampaignRuntimeError, match="cannot open"):
        adapter.advance_exact(
            result.state,
            target_absolute_time_s=0.0201,
            boundary=_boundary("stage2_pressure_reservoir"),
            accepted_step_callback=None,
        )


def test_current_real_formal_campaign_fails_before_source_or_output(tmp_path: Path) -> None:
    config = load_campaign_protocol_config()
    runtime = build_current_s1_campaign_runtime(
        scope="formal_campaign",
        maximum_dt_s=1.0e-7,
    )
    auth = tmp_path / "auth"
    _authorize(auth, config.markers.model_implementation_acceptance, config)
    _authorize(auth, config.markers.formal_campaign_authorization, config)

    with pytest.raises(CampaignProtocolError, match="production_ready"):
        run_formal_campaign(
            config=config,
            runner=runtime.exact_runner,
            codec=runtime.codec,
            observation_bridge=runtime.stage1_observation_bridge,
            authorization_dir=auth,
            output_dir=tmp_path / "formal",
            stage1_acceptance_callback=lambda candidate: True,
        )

    assert runtime.exact_runner.production_ready is False
    assert runtime.exact_runner.source_initialization_count == 0
    assert not (tmp_path / "formal").exists()
    assert (auth / "MODEL_IMPLEMENTATION_ACCEPTED").is_file()
    assert not list(tmp_path.rglob("RESULT_ACCEPTED"))


def test_stage1_observer_rejects_open_air_boundary_without_touching_ledger() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(production_ready=False, stable_dt_s=1.0e-7)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    observer = S1Stage1ObservationBridge(runner=runner)
    with pytest.raises(S1CampaignRuntimeError, match="closed-wall"):
        observer.observe_stage1(
            assembly.state,
            stage1_time_s=0.0,
            boundary=_boundary("stage2_pressure_reservoir"),
        )
    assert runner.committer.ledger.entries == []
