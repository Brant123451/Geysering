import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from model.errors import ContractViolation, MissingPhysicalClosure
from model.flux import HorizontalDelta, SupplyBranchDelta, VerticalDelta, state_token
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import (
    AIR_NODE_PORTS,
    RISER_NODE_PORTS,
    GrossComponentPortFlux,
    JointStageRate,
    S1JointNetworkRunner,
    ZeroStorageTNodeSolution,
    build_current_physical_operator,
)
from trajectory.exporter import build_canonical_trajectory, write_trajectory_artifacts
from trajectory.runtime_bridge import Stage2AcceptedTrajectoryBridge


def _node(name, ports):
    return ZeroStorageTNodeSolution(
        name=name,
        ports=tuple(GrossComponentPortFlux(port) for port in sorted(ports)),
    )


class _NativeZeroOperator:
    """Synthetic no-motion owner with real S1 pressure components."""

    production_ready = False
    validation_only = True

    def __init__(self, *, stable_dt_s=0.07, fail_rk2=False, omit_pressures=False):
        physical = build_current_physical_operator()
        self.horizontal_component = physical.horizontal_component
        self.vertical_component = physical.vertical_component
        self.stable_dt = stable_dt_s
        self.fail_rk2 = fail_rk2
        self.omit_pressures = omit_pressures

    def stable_timestep_s(self, state, geometry, *, physical_stage):
        del state, geometry, physical_stage
        return self.stable_dt

    def diagnostic_node_pressures(
        self, state, geometry, *, physical_stage, diagnostic_dt_s
    ):
        del state, geometry, physical_stage, diagnostic_dt_s
        return (107020.0, 107040.0)

    def evaluate(
        self, state, geometry, *, physical_stage, rk_stage, dt_s
    ):
        del geometry, dt_s
        if self.fail_rk2 and rk_stage == 2:
            raise ContractViolation("manufactured RK2 failure")
        kwargs = {}
        if not self.omit_pressures:
            kwargs = {
                "air_supply_node_common_absolute_pressure_Pa": 107020.0,
                "riser_node_common_absolute_pressure_Pa": 107040.0,
            }
        return JointStageRate(
            physical_stage=physical_stage,
            horizontal=HorizontalDelta.zeros(state.horizontal.cell_count),
            supply_branch=SupplyBranchDelta.zeros(
                state.supply_branch.cell_count
            ),
            vertical=VerticalDelta.zeros(state.vertical.cell_count),
            air_supply_node=_node("air_supply_T", AIR_NODE_PORTS),
            riser_node=_node("riser_T", RISER_NODE_PORTS),
            evidence_status="synthetic_native_observation_bridge_contract",
            **kwargs,
        )


def test_exact_event_ceiling_builds_unshifted_csv_and_npz(tmp_path: Path) -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(stable_dt_s=0.07)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    bridge = Stage2AcceptedTrajectoryBridge(
        stage2_origin_absolute_s=assembly.state.time_s
    )

    result = runner.advance(
        assembly.state,
        duration_s=0.20,
        maximum_dt_s=0.07,
        physical_stage="stage2_pressure_reservoir",
        transaction_prefix="synthetic-common-grid",
        require_production=False,
        accepted_step_callback=bridge,
        stage2_origin_absolute_s=assembly.state.time_s,
    )

    assert result.state.time_s == pytest.approx(0.20, abs=2.0e-14)
    assert [step.stage2_time_end_s for step in bridge.accepted_steps] == pytest.approx(
        [0.07, 0.10, 0.17, 0.20], abs=2.0e-14
    )
    assert [sample.stage2_time_s for sample in bridge.common_samples] == pytest.approx(
        [0.0, 0.1, 0.2], abs=2.0e-14
    )
    assert sum(len(step.ledger_entries) for step in bridge.accepted_steps) == 4
    assert len(bridge.common_samples[1].ledger_entries_since_previous_sample) == 2
    assert len(bridge.common_samples[2].ledger_entries_since_previous_sample) == 2
    assert all(
        step.pressure_semantics.temporal_semantics
        == "instantaneous_accepted_state__no_time_interpolation"
        for step in bridge.accepted_steps
    )
    assert all(
        step.diagnostics.gross_flux.mouth_liquid_outflow_m3_s == 0.0
        for step in bridge.accepted_steps
    )

    bridge.assert_complete_through(0.20)
    trajectory = build_canonical_trajectory(
        bridge.common_samples,
        geometry=assembly.geometry,
        stage2_origin_absolute_s=assembly.state.time_s,
        artifact_role="synthetic_contract_test",
        operator=operator,
    )
    artifacts = write_trajectory_artifacts(trajectory, tmp_path / "trajectory")
    with artifacts.canonical_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["time_s"]) for row in rows] == pytest.approx([0.0, 0.1, 0.2])
    with np.load(artifacts.riser_profiles_npz) as profiles:
        assert profiles["time_s"].tolist() == pytest.approx([0.0, 0.1, 0.2])
        assert profiles["riser_Aup_m2"].shape == (
            3,
            assembly.state.vertical.cell_count,
        )
        assert profiles["riser_Qdown_m3_s"].shape == (
            3,
            assembly.state.vertical.cell_count,
        )
    assert not (tmp_path / "trajectory" / "RESULT_ACCEPTED").exists()


def test_callback_runs_only_after_atomic_ledger_append() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(stable_dt_s=1.0e-7)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    seen = []

    def callback(context):
        assert len(runner.committer.ledger.entries) == 1
        assert runner.committer.ledger.entries[0] is context.ledger_entries[0]
        seen.append(context)

    result = runner.advance_one(
        assembly.state,
        dt_s=1.0e-7,
        physical_stage="stage2_pressure_reservoir",
        transaction_id="accepted-1e-7-callback",
        require_production=False,
        accepted_step_callback=callback,
        stage2_origin_absolute_s=assembly.state.time_s,
    )
    assert result.state.time_s == pytest.approx(1.0e-7)
    assert len(seen) == 1
    assert seen[0].before_state is assembly.state
    assert seen[0].after_state is result.state
    assert seen[0].actual_dt_s == pytest.approx(1.0e-7)
    assert seen[0].stage2_time_end_s == pytest.approx(1.0e-7)


def test_failed_rk2_rolls_back_and_never_calls_callback() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(fail_rk2=True)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    seen = []

    with pytest.raises(ContractViolation, match="manufactured RK2 failure"):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-7,
            physical_stage="stage2_pressure_reservoir",
            transaction_id="failed-rk2-no-callback",
            require_production=False,
            accepted_step_callback=seen.append,
            stage2_origin_absolute_s=assembly.state.time_s,
        )
    assert seen == []
    assert runner.committer.ledger.entries == []


def test_missing_native_pressure_packet_fails_before_commit_or_callback() -> None:
    assembly = build_s1_initial_assembly()
    operator = _NativeZeroOperator(omit_pressures=True)
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    seen = []

    with pytest.raises(MissingPhysicalClosure, match="omitted.*node pressures"):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-7,
            physical_stage="stage2_pressure_reservoir",
            transaction_id="missing-native-no-commit",
            require_production=False,
            accepted_step_callback=seen.append,
            stage2_origin_absolute_s=assembly.state.time_s,
        )
    assert seen == []
    assert runner.committer.ledger.entries == []


def test_current_real_stage1_microstep_commits_then_callback_and_repeats_diagnostics() -> None:
    """The real owner now supports a post-commit Stage-1 observation packet."""

    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    seen = []

    def callback(context):
        assert tuple(runner.committer.ledger.entries) == context.ledger_entries
        seen.append(context)

    result = runner.advance_one(
        assembly.state,
        dt_s=1.0e-7,
        physical_stage="stage1_closed",
        transaction_id="real-physical-stage1-observed-1e-7",
        require_production=False,
        accepted_step_callback=callback,
    )

    assert result.state.time_s == pytest.approx(1.0e-7)
    assert len(seen) == 1
    context = seen[0]
    assert context.before_state is assembly.state
    assert context.after_state is result.state
    assert context.actual_dt_s == pytest.approx(1.0e-7)
    assert context.stage2_time_start_s is None
    assert context.stage2_time_end_s is None
    assert context.ledger_entries == (result.ledger,)
    assert tuple(runner.committer.ledger.entries) == (result.ledger,)

    pressure_after = result.diagnostics.pressure_after
    assert pressure_after is not None
    state_token_before_diagnostics = state_token(result.state)
    ledger_before_diagnostics = tuple(runner.committer.ledger.entries)
    first = operator.diagnostic_node_pressures(
        result.state,
        assembly.geometry,
        physical_stage="stage1_closed",
        diagnostic_dt_s=1.0e-7,
    )
    second = operator.diagnostic_node_pressures(
        result.state,
        assembly.geometry,
        physical_stage="stage1_closed",
        diagnostic_dt_s=1.0e-7,
    )

    assert first == pytest.approx(second, rel=1.0e-13, abs=1.0e-10)
    assert first[1] - pressure_after.semantics.reference_absolute_pressure_Pa == pytest.approx(
        pressure_after.P1,
        rel=1.0e-13,
        abs=1.0e-10,
    )
    assert state_token(result.state) == state_token_before_diagnostics
    assert tuple(runner.committer.ledger.entries) == ledger_before_diagnostics
