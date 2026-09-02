from dataclasses import replace
import math

import pytest

from model.errors import (
    AtomicCommitError,
    ConservationError,
    ContractViolation,
    MissingPhysicalClosure,
)
from model.flux import BoundaryExchange, HorizontalDelta, SupplyBranchDelta, VerticalDelta
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import (
    AIR_NODE_PORTS,
    RISER_NODE_PORTS,
    GrossComponentPortFlux,
    JointStageRate,
    PUBLISHED_STAGE2_GAUGE_PRESSURE_PA,
    S1JointNetworkRunner,
    StructuralZeroJointOperator,
    ZeroStorageTNodeSolution,
    assert_source_initial_physical_smoke_ready,
    build_current_physical_operator,
    _boundary_with_node_reactions,
    run_structural_source_initial_atomic_check,
    stage_boundary_contract,
)


def _node(name: str, ports: frozenset[str]) -> ZeroStorageTNodeSolution:
    return ZeroStorageTNodeSolution(
        name=name,
        ports=tuple(GrossComponentPortFlux(name=port) for port in sorted(ports)),
    )


def _zero_rate(state, physical_stage):
    return JointStageRate(
        physical_stage=physical_stage,
        horizontal=HorizontalDelta.zeros(state.horizontal.cell_count),
        supply_branch=SupplyBranchDelta.zeros(state.supply_branch.cell_count),
        vertical=VerticalDelta.zeros(state.vertical.cell_count),
        air_supply_node=_node("air_supply_T", AIR_NODE_PORTS),
        riser_node=_node("riser_T", RISER_NODE_PORTS),
        evidence_status="unit_test_structural_rate",
    )


def test_stage_contract_keeps_wall_and_published_pressure_distinct() -> None:
    stage1 = stage_boundary_contract("stage1_closed")
    stage2 = stage_boundary_contract("stage2_pressure_reservoir")

    assert stage1.supply_top_kind == "wall"
    assert stage1.air_source_open is False
    assert stage1.gas_gauge_pressure_Pa is None
    assert stage2.supply_top_kind == "pressure_reservoir"
    assert stage2.air_source_open is True
    assert stage2.gas_gauge_pressure_Pa == pytest.approx(
        PUBLISHED_STAGE2_GAUGE_PRESSURE_PA
    )
    assert PUBLISHED_STAGE2_GAUGE_PRESSURE_PA == pytest.approx(5700.0)


def test_source_initial_structural_run_owns_water_filled_branch_and_is_not_physical() -> None:
    assembly = build_s1_initial_assembly()
    area = assembly.geometry.supply_branch_area_m2
    assert area is not None
    assert assembly.state.supply_branch.cell_count == 14
    assert assembly.state.supply_branch.Al == pytest.approx((area,) * 14)
    assert assembly.state.supply_branch.Ql == pytest.approx((0.0,) * 14)
    assert assembly.state.supply_branch.Mg == pytest.approx((0.0,) * 14)
    assert assembly.state.supply_branch.Jg == pytest.approx((0.0,) * 14)

    before = assembly.inventory
    result = run_structural_source_initial_atomic_check(assembly=assembly)
    after = result.entries[-1].after
    assert result.state.time_s == pytest.approx(0.02)
    assert len(result.entries) == 20
    assert result.state.horizontal == assembly.state.horizontal
    assert result.state.supply_branch == assembly.state.supply_branch
    assert result.state.vertical == assembly.state.vertical
    assert after == before
    assert result.validation_only is True
    assert result.production_ready is False
    assert result.status == "structural_atomic_validation_only"


class _CountingZeroOperator:
    production_ready = False
    validation_only = True

    def __init__(self) -> None:
        self.calls = []

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del geometry, dt_s
        self.calls.append((rk_stage, state.time_s))
        return _zero_rate(state, physical_stage)


def test_both_tnodes_are_solved_at_each_rk_stage_but_only_one_packet_commits() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    operator = _CountingZeroOperator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    result = runner.advance_one(
        assembly.state,
        dt_s=1.0e-3,
        physical_stage="stage1_closed",
        transaction_id="one-whole-network-packet",
        require_production=False,
    )

    assert operator.calls == [(1, 0.0), (2, pytest.approx(1.0e-3))]
    assert result.diagnostics.rk1_air_node.port_names == AIR_NODE_PORTS
    assert result.diagnostics.rk1_riser_node.port_names == RISER_NODE_PORTS
    assert result.diagnostics.rk2_air_node.port_names == AIR_NODE_PORTS
    assert result.diagnostics.rk2_riser_node.port_names == RISER_NODE_PORTS
    assert len(runner.committer.ledger.entries) == 1
    assert runner.committer.ledger.entries[0].transaction_id == "one-whole-network-packet"


class _FailsAtSecondStage:
    production_ready = False
    validation_only = True

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del geometry, dt_s
        if rk_stage == 2:
            raise MissingPhysicalClosure("synthetic missing second-stage node closure")
        return _zero_rate(state, physical_stage)


def test_second_rk_stage_failure_rolls_back_every_component_and_global_ledger() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(assembly.geometry, _FailsAtSecondStage())
    original = assembly.state
    with pytest.raises(MissingPhysicalClosure, match="second-stage"):
        runner.advance_one(
            original,
            dt_s=1.0e-3,
            physical_stage="stage1_closed",
            transaction_id="must-not-commit",
            require_production=False,
        )
    assert assembly.state is original
    assert runner.committer.ledger.entries == []


class _WritesGasIntoFullWaterMain:
    production_ready = False
    validation_only = True

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del geometry, rk_stage, dt_s
        result = _zero_rate(state, physical_stage)
        gas_rate = (1.0e-5,) + (0.0,) * (state.horizontal.cell_count - 1)
        return replace(
            result,
            horizontal=HorizontalDelta(
                Al=result.horizontal.Al,
                Ql=result.horizontal.Ql,
                Mg=gas_rate,
                Jg=result.horizontal.Jg,
            ),
        )


def test_direct_gas_write_into_full_water_main_fails_before_any_commit() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(assembly.geometry, _WritesGasIntoFullWaterMain())
    with pytest.raises(
        ContractViolation,
        match=(
            "contains gas and cannot use elastic overarea|"
            "gas mass but no positive complementary gas area"
        ),
    ):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-3,
            physical_stage="stage2_pressure_reservoir",
            transaction_id="illegal-gas-write",
            require_production=False,
        )
    assert runner.committer.ledger.entries == []


class _UnbalancedAirNode:
    production_ready = False
    validation_only = True

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del geometry, rk_stage, dt_s
        result = _zero_rate(state, physical_stage)
        ports = list(result.air_supply_node.ports)
        ports[0] = GrossComponentPortFlux(
            name=ports[0].name,
            liquid_into_component_m3_s=1.0e-6,
            liquid_into_speed_m_s=0.1,
        )
        return replace(
            result,
            air_supply_node=ZeroStorageTNodeSolution(
                name="air_supply_T", ports=tuple(ports)
            ),
        )


def test_either_node_residual_rejects_the_whole_stage_without_cancellation() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(assembly.geometry, _UnbalancedAirNode())
    with pytest.raises(AtomicCommitError, match="RK1 air-supply T node"):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-3,
            physical_stage="stage1_closed",
            transaction_id="node-residual",
            require_production=False,
        )
    assert runner.committer.ledger.entries == []


class _TwoAxisImpulseOperator:
    production_ready = False
    validation_only = True

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del rk_stage, dt_s
        result = _zero_rate(state, physical_stage)
        qx_rate = (1.0e-6,) + (0.0,) * (state.horizontal.cell_count - 1)
        qz_rate = (2.0e-6,) + (0.0,) * (state.supply_branch.cell_count - 1)
        px_force = (
            geometry.liquid_density_kg_m3 * qx_rate[0] * geometry.horizontal_dx_m[0]
        )
        pz_force = (
            geometry.liquid_density_kg_m3
            * qz_rate[0]
            * geometry.supply_branch_dz_m[0]
        )
        return replace(
            result,
            horizontal=HorizontalDelta(
                Al=result.horizontal.Al,
                Ql=qx_rate,
                Mg=result.horizontal.Mg,
                Jg=result.horizontal.Jg,
            ),
            supply_branch=SupplyBranchDelta(
                Al=result.supply_branch.Al,
                Ql=qz_rate,
                Mg=result.supply_branch.Mg,
                Jg=result.supply_branch.Jg,
            ),
            horizontal_external=BoundaryExchange(
                external_force_x_N=px_force,
            ),
            supply_external=BoundaryExchange(
                external_force_z_N=pz_force,
            ),
        )


def test_horizontal_px_and_vertical_supply_pz_close_as_separate_ledgers() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(assembly.geometry, _TwoAxisImpulseOperator())
    result = runner.advance_one(
        assembly.state,
        dt_s=1.0e-3,
        physical_stage="stage1_closed",
        transaction_id="two-axis-ledger",
        require_production=False,
    )
    entry = result.ledger
    assert entry.external_force_x_impulse_kg_m_s != 0.0
    assert entry.external_force_z_impulse_kg_m_s != 0.0
    assert entry.mixture_momentum_x_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-14)
    assert entry.mixture_momentum_z_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-14)
    assert math.isfinite(entry.after.mixture_momentum_x_kg_m_s)
    assert math.isfinite(entry.after.mixture_momentum_z_kg_m_s)


class _GloballyBalancedButMisroutedPz:
    production_ready = False
    validation_only = True

    def evaluate(
        self,
        state,
        geometry,
        *,
        physical_stage,
        rk_stage,
        dt_s,
    ):
        del rk_stage, dt_s
        result = _zero_rate(state, physical_stage)
        supply_q_rate = 1.0e-6
        riser_qdown_rate = (
            supply_q_rate
            * geometry.supply_branch_dz_m[0]
            / geometry.vertical_dz_m[0]
        )
        return replace(
            result,
            supply_branch=SupplyBranchDelta(
                Al=result.supply_branch.Al,
                Ql=(supply_q_rate,)
                + (0.0,) * (state.supply_branch.cell_count - 1),
                Mg=result.supply_branch.Mg,
                Jg=result.supply_branch.Jg,
            ),
            vertical=VerticalDelta(
                Aup=result.vertical.Aup,
                Qup=result.vertical.Qup,
                Adown=result.vertical.Adown,
                Qdown=(riser_qdown_rate,)
                + (0.0,) * (state.vertical.cell_count - 1),
                Mg=result.vertical.Mg,
                Jg=result.vertical.Jg,
            ),
        )


def test_component_port_ledgers_reject_global_cancellation_at_wrong_components() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(
        assembly.geometry, _GloballyBalancedButMisroutedPz()
    )
    with pytest.raises(ConservationError, match="supply component/port ledger"):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-3,
            physical_stage="stage1_closed",
            transaction_id="misrouted-global-cancellation",
            require_production=False,
        )
    assert runner.committer.ledger.entries == []


def test_current_components_expose_exact_physical_blockers_and_fail_closed() -> None:
    operator = build_current_physical_operator()
    assembly = build_s1_initial_assembly()
    assert operator.supply_branch_component.initial_state() == assembly.state.supply_branch
    assert operator.production_ready is False
    blocker_text = " | ".join(operator.readiness.blockers)
    assert "Case1 horizontal" in blocker_text
    assert "water-initial supply" in blocker_text
    assert "persistent riser" in blocker_text
    assert "two-zero-storage-T-node" not in blocker_text
    assert operator.integration_owner_ready is True
    node_capability = next(
        item
        for item in operator.readiness.capabilities
        if item.name == "simultaneous two-zero-storage-T-node solve"
    )
    assert node_capability.ready is True
    with pytest.raises(MissingPhysicalClosure, match="joint physical smoke is blocked"):
        assert_source_initial_physical_smoke_ready()


def test_structural_operator_cannot_be_promoted_by_default_runner_call() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(assembly.geometry, StructuralZeroJointOperator())
    with pytest.raises(MissingPhysicalClosure, match="non-production operator"):
        runner.advance_one(
            assembly.state,
            dt_s=1.0e-3,
            physical_stage="stage1_closed",
            transaction_id="promotion-forbidden",
        )
    assert runner.committer.ledger.entries == []


def test_internal_rim_liquid_pair_is_removed_from_global_gross_boundary() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    rate = replace(
        _zero_rate(assembly.state, "stage1_closed"),
        vertical_external=BoundaryExchange(
            liquid_inflow_m3_s=1.0e-6,
            liquid_outflow_m3_s=2.0e-6,
            gas_outflow_kg_s=3.0e-7,
            momentum_z_out_N=0.70,
            external_force_z_N=-2.0,
        ),
        exterior_plume_exchange=BoundaryExchange(
            liquid_inflow_m3_s=2.0e-6,
            liquid_outflow_m3_s=1.0e-6,
            momentum_z_in_N=0.50,
            external_force_z_N=-0.25,
        ),
    )

    boundary = _boundary_with_node_reactions(rate)

    assert boundary.liquid_inflow_m3_s == 0.0
    assert boundary.liquid_outflow_m3_s == 0.0
    assert boundary.gas_outflow_kg_s == pytest.approx(3.0e-7)
    assert boundary.momentum_z_out_N == pytest.approx(0.20)
    assert boundary.external_force_z_N == pytest.approx(-2.25)


def test_internal_rim_liquid_pair_mismatch_fails_before_global_ledger() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    rate = replace(
        _zero_rate(assembly.state, "stage1_closed"),
        vertical_external=BoundaryExchange(liquid_outflow_m3_s=2.0e-6),
        exterior_plume_exchange=BoundaryExchange(liquid_inflow_m3_s=1.0e-6),
    )

    with pytest.raises(ConservationError, match="rim liquid gross rates"):
        _boundary_with_node_reactions(rate)
