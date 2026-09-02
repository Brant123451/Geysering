from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from model.coupled import AtomicCommitter
from model.errors import ContractViolation
from model.flux import BoundaryExchange, HorizontalDelta, state_token
from model.initialization import build_s1_initial_assembly
from model.port_contracts import (
    AIR_NODE_PORT_NAMES,
    CapacityReject,
    CapillaryInterfaceOwnership,
    ComponentStageProposal,
    F0_CAPILLARY_PRODUCTION_STATUS,
    F0_CLOSURE_SET_ID,
    F0_SURFACE_TENSION_N_M,
    GrossNodePortFlux,
    PortKey,
    PortTraceState,
    RISER_NODE_PORT_NAMES,
    TNodeTrial,
    evaluate_component_trial_pure,
    validate_trial_set,
)


P_ATM = 101325.0
RHO_L = 998.4
RHO_G = P_ATM / (287.05 * 293.15)


def _component(node: str, port: str) -> str:
    if port.startswith("main_"):
        return "horizontal_main"
    if node == "air_supply_T":
        return "air_supply_branch"
    return "vertical_riser"


def _normal(port: str) -> tuple[float, float]:
    if port == "main_left":
        return (1.0, 0.0)
    if port == "main_right":
        return (-1.0, 0.0)
    return (0.0, -1.0)


def _trial(
    *,
    node: str = "air_supply_T",
    base_token: str = "base-state-token",
    interface: CapillaryInterfaceOwnership | None = None,
    interface_port: str | None = None,
) -> TNodeTrial:
    names = AIR_NODE_PORT_NAMES if node == "air_supply_T" else RISER_NODE_PORT_NAMES
    area = 3.141592653589793 * 0.0254**2 / 4.0
    traces = []
    fluxes = []
    for port in sorted(names):
        key = PortKey(node, port)
        nx, nz = _normal(port)
        attached = interface is not None and port == interface_port
        jump = (
            interface.pressure_jump_gas_minus_liquid_Pa
            if attached and interface is not None
            else None
        )
        traces.append(
            PortTraceState(
                key=key,
                component_id=_component(node, port),
                normal_into_node_x=nx,
                normal_into_node_z=nz,
                full_area_m2=area,
                liquid_area_m2=area,
                gas_area_m2=0.0,
                liquid_density_kg_m3=RHO_L,
                gas_density_kg_m3=RHO_G,
                liquid_absolute_pressure_Pa=P_ATM,
                gas_absolute_pressure_Pa=P_ATM + (0.0 if jump is None else jump),
                interface_id=interface.interface_id if attached else None,
            )
        )
        fluxes.append(GrossNodePortFlux(key=key))
    return TNodeTrial(
        trial_id=f"trial-{node}",
        base_state_token=base_token,
        node_name=node,
        physical_stage="stage1_closed",
        rk_stage=1,
        dt_s=1.0e-3,
        common_absolute_pressure_Pa=P_ATM,
        node_gas_area_fraction=0.0,
        port_traces=tuple(traces),
        gross_fluxes=tuple(fluxes),
        interfaces=() if interface is None else (interface,),
    )


def test_contract_constants_match_preregistered_f0_yaml() -> None:
    path = Path(__file__).parents[1] / "config" / "S1_1D_F0_closures.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["closure_set_id"] == F0_CLOSURE_SET_ID
    assert data["shared_fluid_properties"]["surface_tension_N_m"]["value"] == pytest.approx(
        F0_SURFACE_TENSION_N_M
    )
    assert data["capillarity"]["production_status"] == F0_CAPILLARY_PRODUCTION_STATUS
    assert data["t_nodes"]["signed_flux_convention"] == "positive_into_node"


def test_port_trace_enforces_finite_complementary_areas_pressures_and_unit_normal() -> None:
    trial = _trial()
    trace = trial.port_traces[0]
    assert trace.liquid_area_m2 + trace.gas_area_m2 == pytest.approx(trace.full_area_m2)
    assert trace.liquid_absolute_pressure_Pa > 0.0
    assert trace.gas_absolute_pressure_Pa > 0.0
    assert trace.gas_area_fraction == pytest.approx(0.0)

    values = dict(
        key=trace.key,
        component_id=trace.component_id,
        normal_into_node_x=trace.normal_into_node_x,
        normal_into_node_z=trace.normal_into_node_z,
        full_area_m2=trace.full_area_m2,
        liquid_area_m2=trace.liquid_area_m2,
        gas_area_m2=trace.gas_area_m2,
        liquid_density_kg_m3=trace.liquid_density_kg_m3,
        gas_density_kg_m3=trace.gas_density_kg_m3,
        liquid_absolute_pressure_Pa=trace.liquid_absolute_pressure_Pa,
        gas_absolute_pressure_Pa=trace.gas_absolute_pressure_Pa,
    )
    with pytest.raises(ContractViolation, match="unit vector"):
        PortTraceState(**{**values, "normal_into_node_x": 0.5})
    with pytest.raises(ContractViolation, match="complementary"):
        PortTraceState(**{**values, "gas_area_m2": 1.0e-6})
    with pytest.raises(ContractViolation, match="must be positive"):
        PortTraceState(**{**values, "gas_absolute_pressure_Pa": 0.0})
    with pytest.raises(ContractViolation, match="must be finite"):
        PortTraceState(**{**values, "gas_density_kg_m3": float("nan")})


def test_tnode_trial_freezes_absolute_pressure_fraction_ports_and_gross_signs() -> None:
    trial = _trial()
    assert {trace.key.port_name for trace in trial.port_traces} == AIR_NODE_PORT_NAMES
    assert {flux.key.port_name for flux in trial.gross_fluxes} == AIR_NODE_PORT_NAMES
    assert trial.common_absolute_pressure_Pa == pytest.approx(P_ATM)
    assert len(trial.trial_token) == 64
    with pytest.raises(FrozenInstanceError):
        trial.common_absolute_pressure_Pa = P_ATM + 1.0

    flux = GrossNodePortFlux(
        key=PortKey("air_supply_T", "supply_bottom"),
        gas_into_node_kg_s=2.0e-5,
        gas_into_node_speed_m_s=0.25,
        advective_momentum_to_node_z_N=-1.0,
        pressure_traction_to_node_z_N=2.0,
    )
    assert flux.gas_net_into_node_kg_s == pytest.approx(2.0e-5)
    assert flux.mixture_momentum_to_node_z_N == pytest.approx(1.0)

    with pytest.raises(ContractViolation, match=r"\[0, 1\]"):
        TNodeTrial(
            trial_id=trial.trial_id,
            base_state_token=trial.base_state_token,
            node_name=trial.node_name,
            physical_stage=trial.physical_stage,
            rk_stage=trial.rk_stage,
            dt_s=trial.dt_s,
            common_absolute_pressure_Pa=trial.common_absolute_pressure_Pa,
            node_gas_area_fraction=1.1,
            port_traces=trial.port_traces,
            gross_fluxes=trial.gross_fluxes,
        )


def test_capillary_jump_is_f0_frozen_and_one_interface_has_one_owner() -> None:
    interface = CapillaryInterfaceOwnership(
        interface_id="air-entry-meniscus",
        owner="horizontal_main",
        curvature_1_m=10.0,
        contact_angle_deg=90.0,
        pressure_jump_gas_minus_liquid_Pa=0.72,
        evidence_status="declared_contract_test_geometry",
    )
    trial = _trial(interface=interface, interface_port="main_right")
    assert trial.interfaces[0].geometrically_resolved is True
    assert trial.interfaces[0].production_ready is False
    attached = next(trace for trace in trial.port_traces if trace.interface_id is not None)
    assert attached.phase_pressure_jump_gas_minus_liquid_Pa == pytest.approx(0.72)

    unresolved = CapillaryInterfaceOwnership(
        interface_id="unresolved-F0-interface",
        owner="air_supply_branch",
    )
    assert unresolved.production_ready is False

    planar = CapillaryInterfaceOwnership(
        interface_id="declared-planar-cap",
        owner="horizontal_main",
        geometry_mode="planar_2d_zeroGradient_walls",
        curvature_1_m=20.0,
        pressure_jump_gas_minus_liquid_Pa=1.44,
        evidence_status="declared_planar_semicircular_cap__not_result_tuned",
    )
    assert planar.production_ready is True
    assert planar.contact_angle_deg is None

    with pytest.raises(ContractViolation, match="must not invent"):
        CapillaryInterfaceOwnership(
            interface_id="bad-planar-angle",
            owner="horizontal_main",
            geometry_mode="planar_2d_zeroGradient_walls",
            curvature_1_m=20.0,
            contact_angle_deg=90.0,
            pressure_jump_gas_minus_liquid_Pa=1.44,
        )
    with pytest.raises(ContractViolation, match="requires an explicit contact angle"):
        CapillaryInterfaceOwnership(
            interface_id="unresolved-circular-3d",
            owner="horizontal_main",
            geometry_mode="circular_3d_pipe",
            curvature_1_m=20.0,
            pressure_jump_gas_minus_liquid_Pa=1.44,
        )

    with pytest.raises(ContractViolation, match=r"sigma\*kappa"):
        CapillaryInterfaceOwnership(
            interface_id="bad-jump",
            owner="horizontal_main",
            curvature_1_m=10.0,
            contact_angle_deg=90.0,
            pressure_jump_gas_minus_liquid_Pa=1.0,
        )

    duplicate_owner = CapillaryInterfaceOwnership(
        interface_id=interface.interface_id,
        owner="air_supply_t_node",
        curvature_1_m=10.0,
        contact_angle_deg=90.0,
        pressure_jump_gas_minus_liquid_Pa=0.72,
        evidence_status="declared_contract_test_geometry",
    )
    with pytest.raises(ContractViolation, match="more than one owner"):
        TNodeTrial(
            trial_id=trial.trial_id,
            base_state_token=trial.base_state_token,
            node_name=trial.node_name,
            physical_stage=trial.physical_stage,
            rk_stage=trial.rk_stage,
            dt_s=trial.dt_s,
            common_absolute_pressure_Pa=trial.common_absolute_pressure_Pa,
            node_gas_area_fraction=trial.node_gas_area_fraction,
            port_traces=trial.port_traces,
            gross_fluxes=trial.gross_fluxes,
            interfaces=(interface, duplicate_owner),
        )


def test_interface_owner_is_unique_across_the_joint_two_node_trial_set() -> None:
    air_interface = CapillaryInterfaceOwnership(
        interface_id="network-interface-7",
        owner="horizontal_main",
    )
    riser_interface = CapillaryInterfaceOwnership(
        interface_id="network-interface-7",
        owner="riser_t_node",
    )
    air = _trial(interface=air_interface, interface_port="main_right")
    riser = _trial(
        node="riser_T",
        interface=riser_interface,
        interface_port="main_left",
    )
    with pytest.raises(ContractViolation, match="more than one trial"):
        validate_trial_set((air, riser))


class _PureZeroHorizontalOperator:
    component_id = "horizontal_main"

    def evaluate_trial(self, state, geometry, trials):
        del geometry
        return ComponentStageProposal.accepted(
            component_id=self.component_id,
            base_state_token=state_token(state),
            trials=trials,
            delta=HorizontalDelta.zeros(state.horizontal.cell_count),
            accepted_gross_fluxes=tuple(
                flux
                for trial in trials
                for flux in trial.gross_fluxes
                if flux.key.port_name.startswith("main_")
            ),
        )


def test_pure_trial_response_does_not_mutate_state_or_ledger() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    token = state_token(assembly.state)
    trials = (_trial(base_token=token),)
    committer = AtomicCommitter(assembly.geometry)
    operator = _PureZeroHorizontalOperator()

    first = evaluate_component_trial_pure(
        operator, assembly.state, assembly.geometry, trials
    )
    second = evaluate_component_trial_pure(
        operator, assembly.state, assembly.geometry, trials
    )
    assert first == second
    assert first.committable is True
    assert state_token(assembly.state) == token
    assert assembly.state.time_s == pytest.approx(0.0)
    assert committer.ledger.entries == []


def test_capacity_rejection_has_no_delta_flux_or_external_exchange() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    token = state_token(assembly.state)
    trials = (_trial(base_token=token),)
    rejection = CapacityReject(
        component_id="horizontal_main",
        reason_code="phase_capacity",
        detail="receiving main cell has no admissible gas area",
        requested_dt_s=1.0e-3,
        retryable=True,
        maximum_admissible_dt_s=5.0e-4,
    )
    proposal = ComponentStageProposal.rejected(
        component_id="horizontal_main",
        base_state_token=token,
        trials=trials,
        rejection=rejection,
    )
    assert proposal.committable is False
    assert proposal.delta is None
    assert proposal.accepted_gross_fluxes == ()
    assert proposal.external_exchange == BoundaryExchange()

    with pytest.raises(ContractViolation, match="cannot contain a committable delta"):
        ComponentStageProposal(
            component_id="horizontal_main",
            base_state_token=token,
            trial_tokens=(trials[0].trial_token,),
            status="capacity_rejected",
            delta=HorizontalDelta.zeros(assembly.state.horizontal.cell_count),
            capacity_reject=rejection,
        )


def test_proposal_tokens_fail_closed_on_stale_or_mixed_trial_sets() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    token = state_token(assembly.state)
    air = _trial(base_token=token)
    proposal = ComponentStageProposal.accepted(
        component_id="horizontal_main",
        base_state_token=token,
        trials=(air,),
        delta=HorizontalDelta.zeros(assembly.state.horizontal.cell_count),
        accepted_gross_fluxes=tuple(
            flux for flux in air.gross_fluxes if flux.key.port_name.startswith("main_")
        ),
    )
    stale = _trial(base_token="different-state-token")
    with pytest.raises(ContractViolation, match="stale state"):
        proposal.validate_against_trials((stale,))
