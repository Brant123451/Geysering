import math

import pytest

from model.flux import (
    HorizontalDelta,
    SupplyBranchDelta,
    VerticalDelta,
    state_token,
)
from model.port_contracts import (
    AIR_NODE_PORT_NAMES,
    RISER_NODE_PORT_NAMES,
    CapacityReject,
    ComponentStageProposal,
    PortKey,
    PortTraceState,
)
from model.simultaneous_two_tnode_solver import (
    F0SimultaneousTwoTNodeSolver,
    JointNodeSolveFailure,
    PortAcousticScale,
    PortDirectionalSeed,
)


P0 = 101325.0
RHO_L = 998.4
RT = 287.05 * 293.15
C_L = 100.0
C_G = math.sqrt(RT)


def _component(node, port):
    if port.startswith("main_"):
        return "horizontal_main"
    if node == "air_supply_T":
        return "air_supply_branch"
    return "vertical_riser"


def _normal(port):
    if port == "main_left":
        return 1.0, 0.0
    if port == "main_right":
        return -1.0, 0.0
    return 0.0, -1.0


def _all_traces(
    area,
    *,
    gas_fraction_by_key=None,
    pressure_by_key=None,
    liquid_velocity_by_key=None,
    gas_velocity_by_key=None,
):
    gas_fraction_by_key = {} if gas_fraction_by_key is None else gas_fraction_by_key
    pressure_by_key = {} if pressure_by_key is None else pressure_by_key
    liquid_velocity_by_key = (
        {} if liquid_velocity_by_key is None else liquid_velocity_by_key
    )
    gas_velocity_by_key = {} if gas_velocity_by_key is None else gas_velocity_by_key
    result = []
    for node, names in (
        ("air_supply_T", AIR_NODE_PORT_NAMES),
        ("riser_T", RISER_NODE_PORT_NAMES),
    ):
        for port in sorted(names):
            key = PortKey(node, port)
            alpha = gas_fraction_by_key.get(key, 0.0)
            pressure = pressure_by_key.get(key, P0)
            nx, nz = _normal(port)
            result.append(
                PortTraceState(
                    key=key,
                    component_id=_component(node, port),
                    normal_into_node_x=nx,
                    normal_into_node_z=nz,
                    full_area_m2=area,
                    liquid_area_m2=(1.0 - alpha) * area,
                    gas_area_m2=alpha * area,
                    liquid_density_kg_m3=RHO_L,
                    gas_density_kg_m3=pressure / RT,
                    liquid_absolute_pressure_Pa=pressure,
                    gas_absolute_pressure_Pa=pressure,
                    liquid_axial_velocity_m_s=liquid_velocity_by_key.get(key, 0.0),
                    gas_axial_velocity_m_s=gas_velocity_by_key.get(key, 0.0),
                )
            )
    return tuple(result)


def _acoustic(traces):
    return tuple(
        PortAcousticScale(
            key=trace.key,
            liquid_sound_speed_m_s=C_L,
            gas_sound_speed_m_s=C_G,
        )
        for trace in traces
    )


class _AcceptingComponent:
    def __init__(self, component_id):
        self.component_id = component_id

    def evaluate_trial(self, state, geometry, trials):
        del geometry
        fluxes = tuple(
            flux
            for trial in trials
            for flux in trial.gross_fluxes
            if any(
                trace.key == flux.key and trace.component_id == self.component_id
                for trace in trial.port_traces
            )
        )
        if self.component_id == "horizontal_main":
            delta = HorizontalDelta.zeros(state.horizontal.cell_count)
        elif self.component_id == "air_supply_branch":
            delta = SupplyBranchDelta.zeros(state.supply_branch.cell_count)
        else:
            delta = VerticalDelta.zeros(state.vertical.cell_count)
        return ComponentStageProposal.accepted(
            component_id=self.component_id,
            base_state_token=state_token(state),
            trials=trials,
            delta=delta,
            accepted_gross_fluxes=fluxes,
        )


class _RejectingSupply(_AcceptingComponent):
    def __init__(self):
        super().__init__("air_supply_branch")

    def evaluate_trial(self, state, geometry, trials):
        del state, geometry
        rejection = CapacityReject(
            component_id=self.component_id,
            reason_code="phase_capacity",
            detail="manufactured receiving-capacity rejection",
            requested_dt_s=trials[0].dt_s,
            retryable=True,
            maximum_admissible_dt_s=0.5 * trials[0].dt_s,
        )
        return ComponentStageProposal.rejected(
            component_id=self.component_id,
            base_state_token=trials[0].base_state_token,
            trials=trials,
            rejection=rejection,
        )


class _NoGasTransmissionHorizontal(_AcceptingComponent):
    """Manufacture the current upstream gas-front P0 as a hard rejection."""

    def __init__(self):
        super().__init__("horizontal_main")

    def evaluate_trial(self, state, geometry, trials):
        air_trial = next(trial for trial in trials if trial.node_name == "air_supply_T")
        supply = next(
            trace
            for trace in air_trial.port_traces
            if trace.key.port_name == "supply_bottom"
        )
        main = tuple(
            trace
            for trace in air_trial.port_traces
            if trace.key.port_name.startswith("main_")
        )
        if supply.gas_area_m2 > 0.0 and all(trace.gas_area_m2 == 0.0 for trace in main):
            rejection = CapacityReject(
                component_id=self.component_id,
                reason_code="missing_closure",
                detail=(
                    "gas-front transmission into an adjacent full-water main cell "
                    "is not available"
                ),
                requested_dt_s=trials[0].dt_s,
                retryable=False,
            )
            return ComponentStageProposal.rejected(
                component_id=self.component_id,
                base_state_token=trials[0].base_state_token,
                trials=trials,
                rejection=rejection,
            )
        return super().evaluate_trial(state, geometry, trials)


class _RejectingVerticalVoidRemap(_AcceptingComponent):
    """Manufacture the current riser same-stage bottom-gas ordering P0."""

    def __init__(self):
        super().__init__("vertical_riser")

    def evaluate_trial(self, state, geometry, trials):
        del state, geometry
        rejection = CapacityReject(
            component_id=self.component_id,
            reason_code="void_mass_pairing",
            detail=(
                "bottom gas capacity was opened after conservative void remap "
                "but before same-stage gas-source injection"
            ),
            requested_dt_s=trials[0].dt_s,
            retryable=False,
        )
        return ComponentStageProposal.rejected(
            component_id=self.component_id,
            base_state_token=trials[0].base_state_token,
            trials=trials,
            rejection=rejection,
        )


def _operators(*, rejecting_supply=False):
    return (
        _AcceptingComponent("horizontal_main"),
        _RejectingSupply()
        if rejecting_supply
        else _AcceptingComponent("air_supply_branch"),
        _AcceptingComponent("vertical_riser"),
    )


def _solve(coupled_state, geometry, traces, *, seeds=(), stage="stage1_closed"):
    return F0SimultaneousTwoTNodeSolver().solve_pure_stage(
        coupled_state,
        geometry,
        physical_stage=stage,
        rk_stage=1,
        dt_s=1.0e-3,
        traces=traces,
        acoustic_scales=_acoustic(traces),
        component_operators=_operators(),
        directional_seeds=seeds,
    )


def test_all_water_hydrostatic_two_nodes_are_exact_and_share_one_stage(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    result = _solve(coupled_state, geometry, traces)

    assert result.air_trial.common_absolute_pressure_Pa == pytest.approx(P0)
    assert result.riser_trial.common_absolute_pressure_Pa == pytest.approx(P0)
    assert result.air_trial.node_gas_area_fraction == 0.0
    assert result.riser_trial.node_gas_area_fraction == 0.0
    assert result.air_trial.base_state_token == result.riser_trial.base_state_token
    assert result.air_trial.rk_stage == result.riser_trial.rk_stage == 1
    assert result.air_trial.dt_s == result.riser_trial.dt_s == pytest.approx(1.0e-3)
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.normalized_residual_inf == pytest.approx(0.0)
    assert len(result.component_proposals) == 3


def test_stage2_first_air_enters_supply_T_and_is_routed_into_main_without_node_storage(
    coupled_state, geometry, pipe_area
):
    supply = PortKey("air_supply_T", "supply_bottom")
    left = PortKey("air_supply_T", "main_left")
    right = PortKey("air_supply_T", "main_right")
    traces = _all_traces(
        pipe_area,
        gas_fraction_by_key={supply: 1.0},
        liquid_velocity_by_key={left: 2.0, right: 3.0},
        gas_velocity_by_key={supply: -1.0, right: 3.0},
    )
    result = _solve(
        coupled_state,
        geometry,
        traces,
        stage="stage2_pressure_reservoir",
    )
    air_flux = {flux.key: flux for flux in result.air_trial.gross_fluxes}

    assert result.air_trial.node_gas_area_fraction == pytest.approx(1.0 / 3.0)
    assert air_flux[supply].gas_into_node_kg_s > 0.0
    assert air_flux[right].gas_out_of_node_kg_s > 0.0
    assert sum(flux.gas_net_into_node_kg_s for flux in air_flux.values()) == pytest.approx(
        0.0, abs=result.diagnostics.air_scale.gas_residual_tolerance_kg_s
    )
    assert sum(
        flux.liquid_net_into_node_m3_s for flux in air_flux.values()
    ) == pytest.approx(
        0.0, abs=result.diagnostics.air_scale.liquid_residual_tolerance_m3_s
    )


def test_riser_bottom_preserves_simultaneous_up_and_down_gross_streams(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    key = PortKey("riser_T", "riser_bottom")
    rate = 2.0e-5
    speed = 0.2
    seed = PortDirectionalSeed(
        key=key,
        liquid_into_node_m3_s=rate,
        liquid_out_of_node_m3_s=rate,
        liquid_into_node_speed_m_s=speed,
        liquid_out_of_node_speed_m_s=speed,
    )
    result = _solve(coupled_state, geometry, traces, seeds=(seed,))
    bottom = next(flux for flux in result.riser_trial.gross_fluxes if flux.key == key)

    assert bottom.liquid_into_node_m3_s == pytest.approx(rate)
    assert bottom.liquid_out_of_node_m3_s == pytest.approx(rate)
    assert bottom.liquid_net_into_node_m3_s == pytest.approx(0.0)
    component_bottom = next(
        port for port in result.riser_node.ports if port.name == "riser_bottom"
    )
    assert component_bottom.liquid_into_component_m3_s == pytest.approx(rate)
    assert component_bottom.liquid_out_of_component_m3_s == pytest.approx(rate)


def test_both_nodes_are_seen_by_one_horizontal_proposal_and_one_nonlinear_evaluation(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    result = _solve(coupled_state, geometry, traces)
    horizontal = next(
        proposal
        for proposal in result.component_proposals
        if proposal.component_id == "horizontal_main"
    )

    assert len(horizontal.trial_tokens) == 2
    assert set(horizontal.trial_tokens) == {
        result.air_trial.trial_token,
        result.riser_trial.trial_token,
    }
    assert result.diagnostics.component_evaluations == 1


def test_material_conservation_port_recoil_and_wall_reaction_are_explicit(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    result = _solve(coupled_state, geometry, traces)

    for node in (result.air_node, result.riser_node):
        assert node.residual.liquid_volume_m3_s == pytest.approx(0.0, abs=1.0e-15)
        assert node.residual.gas_mass_kg_s == pytest.approx(0.0, abs=1.0e-15)
        assert node.residual.mixture_momentum_x_N == pytest.approx(0.0, abs=1.0e-12)
        assert node.residual.mixture_momentum_z_N == pytest.approx(0.0, abs=1.0e-12)
        port_x = sum(port.mixture_momentum_to_component_x_N for port in node.ports)
        port_z = sum(port.mixture_momentum_to_component_z_N for port in node.ports)
        assert node.wall_reaction_on_fluid_x_N == pytest.approx(port_x)
        assert node.wall_reaction_on_fluid_z_N == pytest.approx(port_z)


def test_component_capacity_rejection_rolls_back_complete_group_without_state_effect(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    before = state_token(coupled_state)
    with pytest.raises(JointNodeSolveFailure, match="phase_capacity"):
        F0SimultaneousTwoTNodeSolver().solve_pure_stage(
            coupled_state,
            geometry,
            physical_stage="stage1_closed",
            rk_stage=1,
            dt_s=1.0e-3,
            traces=traces,
            acoustic_scales=_acoustic(traces),
            component_operators=_operators(rejecting_supply=True),
        )
    assert state_token(coupled_state) == before
    assert coupled_state.time_s == 0.0


def test_unavailable_gas_front_transmission_rejects_both_nodes_atomically(
    coupled_state, geometry, pipe_area
):
    supply = PortKey("air_supply_T", "supply_bottom")
    traces = _all_traces(
        pipe_area,
        gas_fraction_by_key={supply: 1.0},
    )
    before = state_token(coupled_state)
    operators = (
        _NoGasTransmissionHorizontal(),
        _AcceptingComponent("air_supply_branch"),
        _AcceptingComponent("vertical_riser"),
    )
    with pytest.raises(JointNodeSolveFailure, match="gas-front transmission"):
        F0SimultaneousTwoTNodeSolver().solve_pure_stage(
            coupled_state,
            geometry,
            physical_stage="stage2_pressure_reservoir",
            rk_stage=1,
            dt_s=1.0e-3,
            traces=traces,
            acoustic_scales=_acoustic(traces),
            component_operators=operators,
        )
    assert state_token(coupled_state) == before


def test_vertical_same_stage_void_mass_rejection_rolls_back_both_nodes(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    before = state_token(coupled_state)
    operators = (
        _AcceptingComponent("horizontal_main"),
        _AcceptingComponent("air_supply_branch"),
        _RejectingVerticalVoidRemap(),
    )
    with pytest.raises(JointNodeSolveFailure, match="same-stage gas-source injection"):
        F0SimultaneousTwoTNodeSolver().solve_pure_stage(
            coupled_state,
            geometry,
            physical_stage="stage2_pressure_reservoir",
            rk_stage=1,
            dt_s=1.0e-3,
            traces=traces,
            acoustic_scales=_acoustic(traces),
            component_operators=operators,
        )
    assert state_token(coupled_state) == before


def test_nonconvergent_pressure_outside_preregistered_acoustic_bracket_fails_closed(
    coupled_state, geometry, pipe_area
):
    liquid_velocity = {}
    for node, names in (
        ("air_supply_T", AIR_NODE_PORT_NAMES),
        ("riser_T", RISER_NODE_PORT_NAMES),
    ):
        for port in names:
            key = PortKey(node, port)
            nx, nz = _normal(port)
            projection = nx if abs(nx) > 0.5 else nz
            liquid_velocity[key] = 2.0 * C_L / projection
    traces = _all_traces(
        pipe_area,
        liquid_velocity_by_key=liquid_velocity,
    )
    before = state_token(coupled_state)
    with pytest.raises(
        JointNodeSolveFailure,
        match="preregistered bracket|did not find|did not converge",
    ):
        _solve(coupled_state, geometry, traces)
    assert state_token(coupled_state) == before


def test_pressure_brackets_and_tolerances_have_no_result_dependent_knobs(
    coupled_state, geometry, pipe_area
):
    traces = _all_traces(pipe_area)
    result = _solve(coupled_state, geometry, traces)
    scale = result.diagnostics.air_scale
    expected_pressure_scale = max(P0, RHO_L * C_L**2)

    assert scale.pressure_scale_Pa == pytest.approx(expected_pressure_scale)
    assert scale.pressure_lower_Pa == pytest.approx(
        math.ulp(1.0) * expected_pressure_scale
    )
    assert scale.pressure_upper_Pa == pytest.approx(P0 + expected_pressure_scale)
    assert scale.pressure_tolerance_Pa == pytest.approx(
        math.sqrt(math.ulp(1.0)) * expected_pressure_scale
    )
    assert "absolute_p_rho_c_dt_and_port_area" in scale.derivation


def test_algebraic_gate_is_owner_compatible_but_keeps_other_production_blockers():
    solver = F0SimultaneousTwoTNodeSolver()
    assert solver.algebraic_gate_ready is True
    assert solver.physical_owner_compatible is True
    assert solver.production_ready is False
    assert solver.validation_only is True
    assert not any("exterior-plume" in item for item in solver.upstream_production_blockers)
    assert any("water-end" in item for item in solver.upstream_production_blockers)
