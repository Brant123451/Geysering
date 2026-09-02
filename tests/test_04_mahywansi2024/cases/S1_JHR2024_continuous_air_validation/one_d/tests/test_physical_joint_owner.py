from dataclasses import replace

import pytest

from model.flux import state_token
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import (
    AIR_NODE_PORTS,
    RISER_NODE_PORTS,
    S1JointNetworkRunner,
    _state_plus_rate,
    build_current_physical_operator,
)
from model.port_contracts import GrossNodePortFlux, PortKey
from model.simultaneous_two_tnode_solver import JointNodeSolveFailure


def test_real_owner_builds_six_same_state_ports_and_one_conservative_stage() -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    owner = operator.joint_stage_owner

    inputs = owner.build_physical_inputs(assembly.state, assembly.geometry)
    assert len(inputs.traces) == 6
    assert inputs.port_keys == {
        PortKey("air_supply_T", name) for name in AIR_NODE_PORTS
    } | {PortKey("riser_T", name) for name in RISER_NODE_PORTS}
    assert len({scale.key for scale in inputs.acoustic_scales}) == 6
    assert len(inputs.directional_seeds) == 1
    assert inputs.directional_seeds[0].key == PortKey(
        "riser_T", "riser_bottom"
    )
    for node_name in ("air_supply_T", "riser_T"):
        node_pressures = tuple(
            trace.liquid_absolute_pressure_Pa
            for trace in inputs.traces
            if trace.key.node_name == node_name
        )
        assert max(node_pressures) - min(node_pressures) == pytest.approx(
            0.0, abs=2.0e-6
        )

    rate = operator.evaluate(
        assembly.state,
        assembly.geometry,
        physical_stage="stage1_closed",
        rk_stage=1,
        dt_s=1.0e-7,
    )
    assert rate.air_supply_node.port_names == AIR_NODE_PORTS
    assert rate.riser_node.port_names == RISER_NODE_PORTS
    for node in (rate.air_supply_node, rate.riser_node):
        assert node.residual.liquid_volume_m3_s == pytest.approx(0.0, abs=5.0e-13)
        assert node.residual.gas_mass_kg_s == pytest.approx(0.0, abs=5.0e-13)
        assert node.residual.mixture_momentum_x_N == pytest.approx(0.0, abs=1.0e-12)
        assert node.residual.mixture_momentum_z_N == pytest.approx(0.0, abs=1.0e-12)
    assert "real_six_port_two_T_owner" in rate.evidence_status
    assert operator.integration_owner_ready is True
    assert operator.production_ready is False


def test_real_owner_two_rk_stages_commit_exactly_one_packet() -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)

    result = runner.advance_one(
        assembly.state,
        dt_s=1.0e-7,
        physical_stage="stage1_closed",
        transaction_id="real-owner-single-atomic-packet",
        require_production=False,
    )

    assert result.state.time_s == pytest.approx(1.0e-7)
    assert len(runner.committer.ledger.entries) == 1
    assert result.diagnostics.validation_only is True
    assert result.diagnostics.production_ready is False
    assert result.ledger.transaction_id == "real-owner-single-atomic-packet"


def test_source_aligned_stage2_first_air_microstep_is_atomic_and_continuable() -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    before = state_token(assembly.state)
    dt = 1.0e-7

    result = runner.advance_one(
        assembly.state,
        dt_s=dt,
        physical_stage="stage2_pressure_reservoir",
        transaction_id="source-aligned-first-air-microstep",
        require_production=False,
    )

    supply = result.state.supply_branch
    area = operator.supply_branch_component.geometry.area_m2
    gas_area = area - supply.Al[-1]
    assert 0.0 < gas_area < 1.0e-12
    assert supply.Mg[-1] > 0.0
    for al, mg in zip(supply.Al, supply.Mg, strict=True):
        assert ((area - al) > 0.0) == (mg > 0.0)
    operator.supply_branch_component.validate_state(supply)
    assembly.geometry.validate_state(result.state)
    assert result.state.time_s == pytest.approx(dt)
    assert len(runner.committer.ledger.entries) == 1
    assert state_token(assembly.state) == before


@pytest.mark.parametrize(
    ("physical_stage", "step_count"),
    (
        ("stage1_closed", 7),
        ("stage2_pressure_reservoir", 4),
    ),
)
def test_real_source_aligned_microsteps_remain_traceable_and_within_packing_band(
    physical_stage, step_count
) -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    original = state_token(assembly.state)
    state = assembly.state
    dt = 1.0e-7
    packing = operator.vertical_component._solver._parameters.packing_tolerance

    for step in range(1, step_count + 1):
        result = runner.advance_one(
            state,
            dt_s=dt,
            physical_stage=physical_stage,
            transaction_id=f"{physical_stage}-continuation-{step}",
            require_production=False,
        )
        state = result.state
        assembly.geometry.validate_state(state)
        first_trace = operator.vertical_component.port_trace(
            state.vertical, assembly.geometry
        )
        second_trace = operator.vertical_component.port_trace(
            state.vertical, assembly.geometry
        )
        assert first_trace == second_trace
        assert max(
            up + down - assembly.geometry.vertical_area_m2
            for up, down in zip(
                state.vertical.Aup, state.vertical.Adown, strict=True
            )
        ) <= packing
        assert result.ledger.liquid_volume_residual_m3 == pytest.approx(
            0.0, abs=2.0e-14
        )
        assert result.ledger.gas_mass_residual_kg == pytest.approx(
            0.0, abs=2.0e-14
        )
        assert result.ledger.mixture_momentum_x_residual_kg_m_s == pytest.approx(
            0.0, abs=2.0e-11
        )
        assert result.ledger.mixture_momentum_z_residual_kg_m_s == pytest.approx(
            0.0, abs=2.0e-11
        )
        assert len(runner.committer.ledger.entries) == step

    if physical_stage == "stage2_pressure_reservoir":
        area = operator.supply_branch_component.geometry.area_m2
        for liquid_area, gas_mass in zip(
            state.supply_branch.Al, state.supply_branch.Mg, strict=True
        ):
            assert ((area - liquid_area) > 0.0) == (gas_mass > 0.0)
    assert state_token(assembly.state) == original


def test_newton_failure_rolls_back_real_owner_state_and_ledger(monkeypatch) -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    accepted = runner.advance_one(
        assembly.state,
        dt_s=1.0e-7,
        physical_stage="stage1_closed",
        transaction_id="accepted-before-manufactured-failure",
        require_production=False,
    )
    before = state_token(accepted.state)
    ledger_before = tuple(runner.committer.ledger.entries)

    def fail(*args, **kwargs):
        del args, kwargs
        raise JointNodeSolveFailure("manufactured Newton failure before acceptance")

    monkeypatch.setattr(operator.two_tnode_solver, "solve_pure_stage", fail)
    with pytest.raises(JointNodeSolveFailure, match="manufactured Newton failure"):
        runner.advance_one(
            accepted.state,
            dt_s=1.0e-7,
            physical_stage="stage1_closed",
            transaction_id="must-rollback-real-owner",
            require_production=False,
        )

    assert state_token(accepted.state) == before
    assert tuple(runner.committer.ledger.entries) == ledger_before


def test_gas_interface_crosses_air_T_with_atomic_void_and_liquid_displacement() -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    supply = operator.supply_branch_component
    gas_density = 107025.0 / supply.config.rt_J_kg
    full_gas_supply = supply.state_from_bulk(
        gas_volume_m3=supply.geometry.total_volume_m3,
        gas_mass_kg=gas_density * supply.geometry.total_volume_m3,
        liquid_velocity_upward_m_s=0.0,
        gas_velocity_upward_m_s=0.0,
    )
    state = replace(assembly.state, supply_branch=full_gas_supply)
    assembly.geometry.validate_state(state)

    inputs = operator.joint_stage_owner.build_physical_inputs(
        state, assembly.geometry
    )
    air_interfaces = tuple(
        item for item in inputs.interfaces if item.owner == "air_supply_t_node"
    )
    assert len(air_interfaces) == 2
    assert all("gas_nose" in item.evidence_status for item in air_interfaces)
    assert all(item.production_ready for item in air_interfaces)

    dt_s = 1.0e-7
    rate = operator.evaluate(
        state,
        assembly.geometry,
        physical_stage="stage2_pressure_reservoir",
        rk_stage=1,
        dt_s=dt_s,
    )
    ports = {port.name: port for port in rate.air_supply_node.ports}
    assert ports["supply_bottom"].gas_net_into_component_kg_s < 0.0
    assert ports["main_left"].gas_net_into_component_kg_s > 0.0
    assert ports["main_right"].gas_net_into_component_kg_s > 0.0
    assert ports["main_left"].liquid_net_into_component_m3_s == 0.0
    assert ports["main_right"].liquid_net_into_component_m3_s == 0.0
    candidate = _state_plus_rate(state, rate, dt_s)
    assembly.geometry.validate_state(candidate)
    for port_name, cell in (
        ("main_left", operator.horizontal_component.air_face - 1),
        ("main_right", operator.horizontal_component.air_face),
    ):
        del port_name
        assert candidate.horizontal.Mg[cell] > 0.0
        assert (
            assembly.geometry.horizontal_area_m2 - candidate.horizontal.Al[cell]
            > 0.0
        )


@pytest.mark.parametrize(
    ("flux", "expected_up", "expected_down"),
    (
        (
            GrossNodePortFlux(
                key=PortKey("riser_T", "riser_bottom"),
                liquid_into_node_m3_s=1.0e-6,
                liquid_into_node_speed_m_s=0.1,
            ),
            False,
            True,
        ),
        (
            GrossNodePortFlux(
                key=PortKey("riser_T", "riser_bottom"),
                liquid_out_of_node_m3_s=1.0e-6,
                liquid_out_of_node_speed_m_s=0.1,
            ),
            True,
            False,
        ),
    ),
)
def test_zero_momentum_riser_direction_reversal_is_area_and_momentum_conservative(
    flux, expected_up, expected_down
) -> None:
    assembly = build_s1_initial_assembly()
    vertical = build_current_physical_operator().vertical_component
    state = assembly.state.vertical
    if expected_up:
        # Manufacture the opposite resting label first.
        state = replace(
            state,
            Aup=(0.0,) * state.cell_count,
            Adown=state.Aup,
            Qup=(0.0,) * state.cell_count,
            Qdown=(0.0,) * state.cell_count,
        )
    prepared = vertical._prepare_zero_momentum_bottom_direction(state, flux)

    assert (prepared.Aup[0] > 0.0) is expected_up
    assert (prepared.Adown[0] > 0.0) is expected_down
    assert tuple(
        up + down for up, down in zip(prepared.Aup, prepared.Adown, strict=True)
    ) == pytest.approx(
        tuple(up + down for up, down in zip(state.Aup, state.Adown, strict=True))
    )
    assert tuple(
        up - down for up, down in zip(prepared.Qup, prepared.Qdown, strict=True)
    ) == pytest.approx((0.0,) * state.cell_count)
