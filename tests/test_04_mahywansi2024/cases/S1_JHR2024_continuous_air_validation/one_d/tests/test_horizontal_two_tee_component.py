from dataclasses import replace
import math

import numpy as np
import pytest

from model.errors import ContractViolation, MissingPhysicalClosure
from model.flux import state_token
from model.horizontal_case1_adapter import build_s1_2d_eos_aligned_horizontal_adapter
from model.horizontal_distributed import HorizontalDistributedConfig
from model.horizontal_distributed import water_end_inlet_outlet_gas_flux
from model.horizontal_two_tee_component import (
    F0HorizontalTwoTeeStageComponent,
    PLANAR_2D_CAPILLARY_MODE,
)
from model.initialization import build_s1_initial_assembly
from model.port_contracts import (
    CapillaryInterfaceOwnership,
    GrossNodePortFlux,
    PortKey,
    PortTraceState,
    TNodeTrial,
    evaluate_component_trial_pure,
)
from model.state import HorizontalState


P_ATM = 101325.0
DT = 1.0e-6


def _component(*, capillary_mode=None):
    # Equal source heads create a manufactured hydrostatic equilibrium.  This
    # is a contract test only; the source Table-1 0.586/0.584 m heads remain
    # the defaults in production configuration and are not altered on disk.
    config = HorizontalDistributedConfig(
        water_inlet_head_m=0.5842,
        water_outlet_head_m=0.5842,
        elastic_storage_reference_head_m=0.584,
    )
    return F0HorizontalTwoTeeStageComponent(
        adapter=build_s1_2d_eos_aligned_horizontal_adapter(),
        config=config,
        capillary_geometry_mode=capillary_mode,
    )


def _third_trace(node: str, area: float, pressure: float, rt: float) -> PortTraceState:
    if node == "air_supply_T":
        key = PortKey(node, "supply_bottom")
        component_id = "air_supply_branch"
    else:
        key = PortKey(node, "riser_bottom")
        component_id = "vertical_riser"
    return PortTraceState(
        key=key,
        component_id=component_id,
        normal_into_node_x=0.0,
        normal_into_node_z=-1.0,
        full_area_m2=area,
        liquid_area_m2=area,
        gas_area_m2=0.0,
        liquid_density_kg_m3=998.4,
        gas_density_kg_m3=pressure / rt,
        liquid_absolute_pressure_Pa=pressure,
        gas_absolute_pressure_Pa=pressure,
        evidence_status="manufactured_third_port_contract_trace__not_node_solution",
    )


def _trials(
    component,
    assembly,
    *,
    interfaces_by_port=None,
    flux_builders=None,
    third_fluxes=None,
    node_pressures=None,
    node_gas_fractions=None,
    rk_stage=1,
):
    interfaces_by_port = {} if interfaces_by_port is None else interfaces_by_port
    flux_builders = {} if flux_builders is None else flux_builders
    third_fluxes = {} if third_fluxes is None else third_fluxes
    node_pressures = {} if node_pressures is None else node_pressures
    node_gas_fractions = {} if node_gas_fractions is None else node_gas_fractions
    traces = {
        trace.key: trace
        for trace in component.port_traces(
            assembly.state.horizontal,
            assembly.geometry,
            interfaces_by_port=interfaces_by_port,
        )
    }
    token = state_token(assembly.state)
    trials = []
    for node in ("air_supply_T", "riser_T"):
        main = tuple(
            traces[PortKey(node, port)] for port in ("main_left", "main_right")
        )
        common_pressure = node_pressures.get(
            node, main[0].liquid_absolute_pressure_Pa
        )
        third = _third_trace(
            node,
            component.area_m2,
            common_pressure,
            component.config.rt_J_kg,
        )
        node_traces = main + (third,)
        fluxes = []
        for trace in node_traces:
            builder = flux_builders.get(trace.key)
            if builder is not None:
                fluxes.append(builder(trace, common_pressure))
            elif trace.key in third_fluxes:
                fluxes.append(third_fluxes[trace.key])
            elif trace.component_id == component.component_id:
                fluxes.append(
                    GrossNodePortFlux(
                        key=trace.key,
                        pressure_traction_to_node_x_N=(
                            component.stationary_pressure_traction_to_node_N(trace)
                        ),
                    )
                )
            else:
                fluxes.append(GrossNodePortFlux(key=trace.key))
        referenced = {
            trace.interface_id
            for trace in node_traces
            if trace.interface_id is not None
        }
        interface_records = tuple(
            interfaces_by_port[key]
            for key in sorted(interfaces_by_port)
            if key.node_name == node
            and interfaces_by_port[key].interface_id in referenced
        )
        trials.append(
            TNodeTrial(
                trial_id=f"manufactured-{node}-rk{rk_stage}",
                base_state_token=token,
                node_name=node,
                physical_stage="stage1_closed",
                rk_stage=rk_stage,
                dt_s=DT,
                common_absolute_pressure_Pa=common_pressure,
                node_gas_area_fraction=node_gas_fractions.get(node, 0.0),
                port_traces=node_traces,
                gross_fluxes=tuple(fluxes),
                interfaces=interface_records,
            )
        )
    return tuple(trials)


def _propose(component, assembly, trials):
    return component.propose_joint_stage(
        assembly.state.horizontal,
        assembly.geometry,
        air_node_trial=trials[0],
        riser_node_trial=trials[1],
        physical_stage="stage1_closed",
        dt_s=DT,
    )


def _semicircular_nose(component, interface_id):
    curvature = 2.0 / component.diameter_m
    return CapillaryInterfaceOwnership(
        interface_id=interface_id,
        owner="horizontal_main",
        geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        curvature_1_m=curvature,
        pressure_jump_gas_minus_liquid_Pa=0.072 * curvature,
        # No contact angle is supplied: frozen 2-D walls use zeroGradient.
        evidence_status=(
            "declared_planar_semicircular_gas_nose_2_over_D_translation__not_tuned"
        ),
    )


def test_2d_eos_reconciled_flag_requires_the_frozen_perfectfluid_tangent() -> None:
    with pytest.raises(ContractViolation, match="perfectFluid tangent"):
        F0HorizontalTwoTeeStageComponent(liquid_eos_reconciled_with_2d=True)

    component = F0HorizontalTwoTeeStageComponent(
        adapter=build_s1_2d_eos_aligned_horizontal_adapter(),
        capillary_geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        liquid_eos_reconciled_with_2d=True,
    )
    assert component.readiness.liquid_eos_reconciled_with_2d is True
    assert component.adapter.wave_speed_m_s == pytest.approx(math.sqrt(3000.0 * 293.15))


def test_hydrostatic_two_tee_stage_is_static_and_lineage_is_not_overclaimed() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component()
    trials = _trials(component, assembly)
    proposal = _propose(component, assembly, trials)

    assert proposal.committable is True
    assert component.air_face == 31
    assert component.riser_face == 183
    assert component.spatial_lineage == (
        "Case1 circular A(h)",
        "hash-pinned Case1 MUSCL central-upwind face kernel",
        "hash-pinned Case1 donor draining limiter",
        "whole-network SSP-RK2 using the Case1 forward-Euler spatial stage",
        "two atomic T endpoints plus explicit moving gas-interface faces",
        "Table-1 inlet-total/outlet-static characteristic ghosts",
    )
    assert component.readiness.case1_geometry_pressure_law_derived is True
    assert component.readiness.case1_circular_fv_lineage is True
    assert component.readiness.table1_characteristic_pressure_boundaries is True
    assert {flux.key for flux in proposal.accepted_gross_fluxes} == {
        PortKey("air_supply_T", "main_left"),
        PortKey("air_supply_T", "main_right"),
        PortKey("riser_T", "main_left"),
        PortKey("riser_T", "main_right"),
    }
    for field in ("Al", "Ql", "Mg", "Jg"):
        assert getattr(proposal.delta, field) == pytest.approx(
            (0.0,) * assembly.state.horizontal.cell_count, abs=2.0e-12
        )
    assert proposal.external_exchange.liquid_volume_net_rate == pytest.approx(0.0)
    assert proposal.external_exchange.mixture_momentum_x_net_rate == pytest.approx(
        0.0, abs=1.0e-10
    )
    assert component.production_ready is False
    assert component.readiness.liquid_eos_reconciled_with_2d is False
    with pytest.raises(MissingPhysicalClosure, match="capillary geometry mode"):
        component.assert_source_aligned_trajectory_ready()


def test_full_and_elastic_t_trace_preserve_case1_pressure_force_and_table1_datum() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    component = F0HorizontalTwoTeeStageComponent(
        adapter=adapter,
        capillary_geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        liquid_eos_reconciled_with_2d=True,
    )
    key = PortKey("air_supply_T", "main_left")
    cell = component.air_face - 1
    full_areas = list(assembly.state.horizontal.Al)
    full_areas[cell] = component.area_m2
    full_state = HorizontalState(
        Al=tuple(full_areas),
        Ql=assembly.state.horizontal.Ql,
        Mg=assembly.state.horizontal.Mg,
        Jg=assembly.state.horizontal.Jg,
    )
    full_trace = next(
        trace
        for trace in component.port_traces(
            full_state, assembly.geometry
        )
        if trace.key == key
    )

    epsilon = 1.0e-5
    elastic_area = component.area_m2 * (1.0 + epsilon)
    elastic_areas = list(full_state.Al)
    elastic_areas[cell] = elastic_area
    elastic_state = HorizontalState(
        Al=tuple(elastic_areas),
        Ql=assembly.state.horizontal.Ql,
        Mg=assembly.state.horizontal.Mg,
        Jg=assembly.state.horizontal.Jg,
    )
    elastic_trace = next(
        trace
        for trace in component.port_traces(elastic_state, assembly.geometry)
        if trace.key == key
    )
    crown_reference = component.config.atmospheric_pressure_Pa + (
        component.config.liquid_density_kg_m3
        * adapter.gravity_m_s2
        * (
            component.config.elastic_storage_reference_head_m
            - 0.5 * component.diameter_m
        )
    )
    expected_elastic = crown_reference + (
        adapter.conservative_port_pressure_increment_Pa(
            elastic_area, component.config.liquid_density_kg_m3
        )
    )

    assert full_trace.liquid_area_m2 == pytest.approx(component.area_m2)
    assert elastic_trace.liquid_area_m2 == pytest.approx(component.area_m2)
    assert full_trace.liquid_absolute_pressure_Pa == pytest.approx(crown_reference)
    assert elastic_trace.liquid_absolute_pressure_Pa == pytest.approx(expected_elastic)
    assert elastic_trace.liquid_absolute_pressure_Pa > full_trace.liquid_absolute_pressure_Pa
    assert "hash_pinned_Case1_MUSCL_central_upwind" in elastic_trace.evidence_status
    assert "Case1_donor_draining" in elastic_trace.evidence_status

    with pytest.raises(MissingPhysicalClosure, match="unit component"):
        component.assert_source_aligned_trajectory_ready()


def test_formal_water_ends_call_total_pressure_inlet_and_static_pressure_outlet(
    monkeypatch,
) -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = F0HorizontalTwoTeeStageComponent(
        adapter=build_s1_2d_eos_aligned_horizontal_adapter(),
        capillary_geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        liquid_eos_reconciled_with_2d=True,
    )
    calls = {"total": 0, "static": 0}
    original_total = component.adapter.dynamic_total_pressure_ghost
    original_static = component.adapter.static_pressure_characteristic_ghost

    def total(*args, **kwargs):
        calls["total"] += 1
        return original_total(*args, **kwargs)

    def static(*args, **kwargs):
        calls["static"] += 1
        return original_static(*args, **kwargs)

    monkeypatch.setattr(component.adapter, "dynamic_total_pressure_ghost", total)
    monkeypatch.setattr(
        component.adapter, "static_pressure_characteristic_ghost", static
    )
    component._table1_pressure_ghosts(
        component._arrays(assembly.state.horizontal)
    )

    assert calls == {"total": 1, "static": 1}


def test_air_and_riser_tee_trials_have_independent_faces_but_one_atomic_proposal() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component()
    baseline_trials = _trials(component, assembly)
    baseline = _propose(component, assembly, baseline_trials)
    impulse = 0.01

    def perturb(trace, _pressure):
        return GrossNodePortFlux(
            key=trace.key,
            pressure_traction_to_node_x_N=(
                component.stationary_pressure_traction_to_node_N(trace) + impulse
            ),
        )

    air_key = PortKey("air_supply_T", "main_left")
    riser_key = PortKey("riser_T", "main_left")
    air = _propose(
        component,
        assembly,
        _trials(component, assembly, flux_builders={air_key: perturb}),
    )
    riser = _propose(
        component,
        assembly,
        _trials(component, assembly, flux_builders={riser_key: perturb}),
    )
    assert air.committable and riser.committable
    air_changed = np.flatnonzero(
        np.abs(np.asarray(air.delta.Ql) - np.asarray(baseline.delta.Ql)) > 1.0e-12
    )
    riser_changed = np.flatnonzero(
        np.abs(np.asarray(riser.delta.Ql) - np.asarray(baseline.delta.Ql)) > 1.0e-12
    )
    assert air_changed.tolist() == [component.air_face - 1]
    assert riser_changed.tolist() == [component.riser_face - 1]

    mixed_stage = replace(
        baseline_trials[1],
        trial_id="mixed-rk-stage",
        rk_stage=2,
    )
    with pytest.raises(ContractViolation, match="share state, stage, RK index and dt"):
        _propose(component, assembly, (baseline_trials[0], mixed_stage))


def test_first_gas_entry_creates_Ag_and_Mg_and_displaces_equal_liquid_volume() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component(capillary_mode=PLANAR_2D_CAPILLARY_MODE)
    receiving_key = PortKey("air_supply_T", "main_right")
    interface = _semicircular_nose(component, "air-main-right-gas-nose")
    base_trace = next(
        trace
        for trace in component.port_traces(
            assembly.state.horizontal,
            assembly.geometry,
            interfaces_by_port={receiving_key: interface},
        )
        if trace.key == receiving_key
    )
    node_pressure = base_trace.gas_absolute_pressure_Pa
    rho_node = node_pressure / component.config.rt_J_kg
    gas_rate = 2.0e-7
    gas_opening_rate = gas_rate / rho_node
    cell = component.air_face
    elastic_release_rate = (
        (assembly.state.horizontal.Al[cell] - component.area_m2)
        * assembly.geometry.horizontal_dx_m[cell]
        / DT
    )
    liquid_rate = gas_opening_rate + elastic_release_rate
    speed = 0.08

    def gas_entry(trace, _pressure):
        normal = trace.normal_into_node_x
        return GrossNodePortFlux(
            key=trace.key,
            liquid_into_node_m3_s=liquid_rate,
            gas_out_of_node_kg_s=gas_rate,
            liquid_into_node_speed_m_s=speed,
            gas_out_of_node_speed_m_s=speed,
            advective_momentum_to_node_x_N=normal
            * (998.4 * liquid_rate * speed + gas_rate * speed),
            pressure_traction_to_node_x_N=(
                component.stationary_pressure_traction_to_node_N(trace)
            ),
        )

    supply_key = PortKey("air_supply_T", "supply_bottom")
    supply_balance = GrossNodePortFlux(
        key=supply_key,
        liquid_out_of_node_m3_s=liquid_rate,
        gas_into_node_kg_s=gas_rate,
        liquid_out_of_node_speed_m_s=speed,
        gas_into_node_speed_m_s=speed,
    )
    trials = _trials(
        component,
        assembly,
        interfaces_by_port={receiving_key: interface},
        flux_builders={receiving_key: gas_entry},
        third_fluxes={supply_key: supply_balance},
        node_pressures={"air_supply_T": node_pressure},
        node_gas_fractions={"air_supply_T": 0.05},
    )
    proposal = _propose(component, assembly, trials)
    assert proposal.committable is True
    d_liquid_area = -proposal.delta.Al[cell]
    d_geometric_gas_area = gas_opening_rate / assembly.geometry.horizontal_dx_m[cell]
    assert d_liquid_area > 0.0
    assert d_geometric_gas_area > 0.0
    assert proposal.delta.Mg[cell] > 0.0
    assert proposal.delta.Mg[cell] / d_geometric_gas_area == pytest.approx(rho_node)
    assert d_liquid_area * assembly.geometry.horizontal_dx_m[cell] == pytest.approx(
        liquid_rate
    )
    candidate_ag = d_geometric_gas_area * DT
    candidate_mg = proposal.delta.Mg[cell] * DT
    assert candidate_ag > 0.0 and candidate_mg > 0.0
    assert candidate_mg * component.config.rt_J_kg / candidate_ag == pytest.approx(
        node_pressure
    )
    assert interface.contact_angle_deg is None
    assert component.capillary_translation_evidence.startswith(
        "declared_translation_of_frozen_2D_zeroGradient_walls"
    )


@pytest.mark.parametrize(
    ("gas_velocity_m_s", "receiving_offset", "expected_momentum_sign"),
    ((0.20, 1, 1.0), (-0.20, -1, -1.0)),
)
def test_internal_gas_nose_crosses_into_full_neighbor_with_atomic_phase_pairing(
    gas_velocity_m_s, receiving_offset, expected_momentum_sign
) -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component()
    donor = 100
    receiver = donor + receiving_offset
    gas_area = 0.20 * component.area_m2
    liquid_area = component.area_m2 - gas_area
    gas_pressure = 106500.0
    gas_density = gas_pressure / component.config.rt_J_kg
    gas_mass = gas_density * gas_area

    al = list(assembly.state.horizontal.Al)
    mg = list(assembly.state.horizontal.Mg)
    jg = list(assembly.state.horizontal.Jg)
    al[donor] = liquid_area
    mg[donor] = gas_mass
    jg[donor] = gas_mass * gas_velocity_m_s
    horizontal = HorizontalState(
        Al=tuple(al),
        Ql=assembly.state.horizontal.Ql,
        Mg=tuple(mg),
        Jg=tuple(jg),
    )
    moved = replace(assembly, state=replace(assembly.state, horizontal=horizontal))
    proposal = _propose(component, moved, _trials(component, moved))

    assert proposal.committable is True
    opened_void_rate = -proposal.delta.Al[receiver]
    received_mass_rate = proposal.delta.Mg[receiver]
    received_momentum_rate = proposal.delta.Jg[receiver]
    assert opened_void_rate > 0.0
    assert received_mass_rate > 0.0
    assert expected_momentum_sign * received_momentum_rate > 0.0
    assert received_mass_rate / opened_void_rate == pytest.approx(
        gas_density, rel=2.0e-11
    )
    assert received_momentum_rate / received_mass_rate == pytest.approx(
        gas_velocity_m_s, rel=2.0e-11, abs=1.0e-12
    )

    dx = np.asarray(moved.geometry.horizontal_dx_m)
    assert np.dot(np.asarray(proposal.delta.Al), dx) == pytest.approx(
        0.0, abs=2.0e-12
    )
    assert np.dot(np.asarray(proposal.delta.Mg), dx) == pytest.approx(
        0.0, abs=2.0e-12
    )
    mixture_momentum_rate = np.dot(
        component.config.liquid_density_kg_m3 * np.asarray(proposal.delta.Ql)
        + np.asarray(proposal.delta.Jg),
        dx,
    )
    expected_momentum_rate = (
        proposal.external_exchange.mixture_momentum_x_net_rate
        - sum(
            flux.mixture_momentum_to_node_x_N
            for flux in proposal.accepted_gross_fluxes
        )
    )
    assert mixture_momentum_rate == pytest.approx(
        expected_momentum_rate, abs=2.0e-7
    )


def test_internal_liquid_perturbation_cannot_create_a_void_without_gas_source() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component()
    ql = list(assembly.state.horizontal.Ql)
    ql[100] = 1.0e-7
    horizontal = HorizontalState(
        Al=assembly.state.horizontal.Al,
        Ql=tuple(ql),
        Mg=assembly.state.horizontal.Mg,
        Jg=assembly.state.horizontal.Jg,
    )
    moved = replace(assembly, state=replace(assembly.state, horizontal=horizontal))
    proposal = _propose(component, moved, _trials(component, moved))

    assert proposal.committable is False
    assert proposal.capacity_reject.reason_code == "void_mass_pairing"
    assert "massless void" in proposal.capacity_reject.detail
    assert proposal.delta is None


def test_first_T_gas_entry_is_atomically_paired_but_wrong_cap_sign_rejects() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component(capillary_mode=PLANAR_2D_CAPILLARY_MODE)
    key = PortKey("air_supply_T", "main_right")
    interface = _semicircular_nose(component, "massless-entry-nose")
    trace = next(
        item
        for item in component.port_traces(
            assembly.state.horizontal,
            assembly.geometry,
            interfaces_by_port={key: interface},
        )
        if item.key == key
    )
    gas_rate = 2.0e-7

    def unpaired_gas(item, _pressure):
        return GrossNodePortFlux(
            key=item.key,
            gas_out_of_node_kg_s=gas_rate,
            gas_out_of_node_speed_m_s=0.08,
            advective_momentum_to_node_x_N=(
                item.normal_into_node_x * gas_rate * 0.08
            ),
            pressure_traction_to_node_x_N=(
                component.stationary_pressure_traction_to_node_N(item)
            ),
        )

    trials = _trials(
        component,
        assembly,
        interfaces_by_port={key: interface},
        flux_builders={key: unpaired_gas},
        node_pressures={"air_supply_T": trace.gas_absolute_pressure_Pa},
    )
    token_before = state_token(assembly.state)
    proposal = _propose(component, assembly, trials)
    assert proposal.committable is True
    assert proposal.capacity_reject is None
    assert proposal.delta is not None
    accepted = next(flux for flux in proposal.accepted_gross_fluxes if flux.key == key)
    assert accepted.gas_out_of_node_kg_s == pytest.approx(gas_rate)
    assert accepted.liquid_into_node_m3_s == 0.0
    assert accepted.liquid_out_of_node_m3_s == 0.0
    cell = component.air_face
    candidate_area = (
        assembly.state.horizontal.Al[cell] + DT * proposal.delta.Al[cell]
    )
    candidate_mass = (
        assembly.state.horizontal.Mg[cell] + DT * proposal.delta.Mg[cell]
    )
    assert component.area_m2 - candidate_area > 0.0
    assert candidate_mass > 0.0
    assert proposal.external_exchange.liquid_volume_net_rate == 0.0
    assert state_token(assembly.state) == token_before

    wrong_curvature = -2.0 / component.diameter_m
    wrong = CapillaryInterfaceOwnership(
        interface_id="wrong-sign-nose",
        owner="horizontal_main",
        geometry_mode=PLANAR_2D_CAPILLARY_MODE,
        curvature_1_m=wrong_curvature,
        pressure_jump_gas_minus_liquid_Pa=0.072 * wrong_curvature,
        evidence_status="declared_planar_semicircular_gas_nose_2_over_D_translation",
    )
    wrong_trials = _trials(
        component,
        assembly,
        interfaces_by_port={key: wrong},
    )
    wrong_proposal = _propose(component, assembly, wrong_trials)
    assert wrong_proposal.committable is False
    assert wrong_proposal.capacity_reject.reason_code == "missing_closure"
    assert "topology-frozen" in wrong_proposal.capacity_reject.detail


def test_horizontal_liquid_gas_and_Px_ledgers_close_against_ports_and_external() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component(capillary_mode=PLANAR_2D_CAPILLARY_MODE)
    key = PortKey("air_supply_T", "main_right")
    interface = _semicircular_nose(component, "ledger-entry-nose")
    trace = next(
        item
        for item in component.port_traces(
            assembly.state.horizontal,
            assembly.geometry,
            interfaces_by_port={key: interface},
        )
        if item.key == key
    )
    pressure = trace.gas_absolute_pressure_Pa
    rho_node = pressure / component.config.rt_J_kg
    gas_rate = 1.0e-7
    gas_opening_rate = gas_rate / rho_node
    cell = component.air_face
    elastic_release_rate = (
        (assembly.state.horizontal.Al[cell] - component.area_m2)
        * assembly.geometry.horizontal_dx_m[cell]
        / DT
    )
    liquid_rate = gas_opening_rate + elastic_release_rate

    def entry(item, _pressure):
        speed = 0.05
        return GrossNodePortFlux(
            key=item.key,
            liquid_into_node_m3_s=liquid_rate,
            gas_out_of_node_kg_s=gas_rate,
            liquid_into_node_speed_m_s=speed,
            gas_out_of_node_speed_m_s=speed,
            advective_momentum_to_node_x_N=item.normal_into_node_x
            * (998.4 * liquid_rate * speed + gas_rate * speed),
            pressure_traction_to_node_x_N=(
                component.stationary_pressure_traction_to_node_N(item)
            ),
        )

    trials = _trials(
        component,
        assembly,
        interfaces_by_port={key: interface},
        flux_builders={key: entry},
        node_pressures={"air_supply_T": pressure},
    )
    proposal = _propose(component, assembly, trials)
    assert proposal.committable
    dx = assembly.geometry.horizontal_dx_m
    rho_l = assembly.geometry.liquid_density_kg_m3
    observed_liquid = sum(a * width for a, width in zip(proposal.delta.Al, dx, strict=True))
    observed_gas = sum(m * width for m, width in zip(proposal.delta.Mg, dx, strict=True))
    observed_px = sum(
        (rho_l * q + j) * width
        for q, j, width in zip(
            proposal.delta.Ql, proposal.delta.Jg, dx, strict=True
        )
    )
    expected_liquid = proposal.external_exchange.liquid_volume_net_rate + sum(
        flux.liquid_out_of_node_m3_s - flux.liquid_into_node_m3_s
        for flux in proposal.accepted_gross_fluxes
    )
    expected_gas = proposal.external_exchange.gas_mass_net_rate + sum(
        flux.gas_out_of_node_kg_s - flux.gas_into_node_kg_s
        for flux in proposal.accepted_gross_fluxes
    )
    expected_px = proposal.external_exchange.mixture_momentum_x_net_rate - sum(
        flux.mixture_momentum_to_node_x_N
        for flux in proposal.accepted_gross_fluxes
    )
    assert observed_liquid == pytest.approx(expected_liquid, abs=2.0e-11)
    assert observed_gas == pytest.approx(expected_gas, abs=2.0e-11)
    assert observed_px == pytest.approx(expected_px, abs=2.0e-7)


def test_pure_evaluation_is_repeatable_and_rejection_does_not_pollute_state() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component()
    trials = _trials(component, assembly)
    token = state_token(assembly.state)
    first = evaluate_component_trial_pure(
        component, assembly.state, assembly.geometry, trials
    )
    second = evaluate_component_trial_pure(
        component, assembly.state, assembly.geometry, trials
    )
    assert first == second
    assert state_token(assembly.state) == token

    # A stale horizontal trace is a malformed nonlinear trial, and must fail
    # before any proposal or ledger side effect exists.
    air = trials[0]
    stale_trace = replace(
        air.port_traces[0],
        liquid_absolute_pressure_Pa=air.port_traces[0].liquid_absolute_pressure_Pa + 1.0,
        gas_absolute_pressure_Pa=air.port_traces[0].gas_absolute_pressure_Pa + 1.0,
    )
    stale_air = replace(
        air,
        trial_id="stale-horizontal-trace",
        port_traces=(stale_trace,) + air.port_traces[1:],
    )
    with pytest.raises(ContractViolation, match="stale horizontal port trace"):
        evaluate_component_trial_pure(
            component,
            assembly.state,
            assembly.geometry,
            (stale_air, trials[1]),
        )
    assert state_token(assembly.state) == token
    again = evaluate_component_trial_pure(
        component, assembly.state, assembly.geometry, trials
    )
    assert again == first


def _assembly_with_end_gas(*, side: str, gas_velocity_m_s: float):
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    component = _component(capillary_mode=PLANAR_2D_CAPILLARY_MODE)
    index = 0 if side == "left" else -1
    liquid_area = 0.75 * component.area_m2
    gas_area = component.area_m2 - liquid_area
    pressure = component.config.atmospheric_pressure_Pa
    gas_mass = pressure * gas_area / component.config.rt_J_kg
    areas = list(assembly.state.horizontal.Al)
    liquid = list(assembly.state.horizontal.Ql)
    gas = list(assembly.state.horizontal.Mg)
    gas_momentum = list(assembly.state.horizontal.Jg)
    areas[index] = liquid_area
    liquid[index] = 0.0
    gas[index] = gas_mass
    gas_momentum[index] = gas_mass * gas_velocity_m_s
    horizontal = HorizontalState(
        Al=tuple(areas),
        Ql=tuple(liquid),
        Mg=tuple(gas),
        Jg=tuple(gas_momentum),
    )
    return component, replace(
        assembly,
        state=replace(assembly.state, horizontal=horizontal),
    )


@pytest.mark.parametrize(
    ("side", "velocity", "expected_sign"),
    (("left", -20.0, -1.0), ("right", 20.0, 1.0)),
)
def test_water_end_inletOutlet_gas_outflow_is_one_sided_and_inventory_limited(
    side, velocity, expected_sign
) -> None:
    area = 2.0e-4
    mass_per_length = 3.0e-4
    dt = 0.02
    dx = 0.01
    flux = water_end_inlet_outlet_gas_flux(
        side=side,
        gas_area_m2=area,
        gas_mass_kg_m=mass_per_length,
        gas_momentum_kg_s=mass_per_length * velocity,
        interior_absolute_pressure_Pa=P_ATM,
        dx_m=dx,
        dt_s=dt,
        gas_presence_mass_kg_m=1.0e-12,
        gas_presence_area_m2=1.0e-12,
    )

    assert math.copysign(1.0, flux.gas_mass_left_to_right_kg_s) == expected_sign
    assert flux.gas_inflow_kg_s == 0.0
    assert flux.gas_outflow_kg_s == pytest.approx(
        abs(flux.gas_mass_left_to_right_kg_s)
    )
    assert flux.gas_outflow_kg_s == pytest.approx(mass_per_length * dx / dt)
    assert flux.prescribed_reentry_alpha_water == 1.0
    assert "Table1_alpha_water_inletOutlet" in flux.evidence_status
    assert "no_external_gas_inventory" in flux.evidence_status


def test_water_end_inletOutlet_rejects_unpaired_phase_state() -> None:
    common = dict(
        side="right",
        gas_momentum_kg_s=0.0,
        interior_absolute_pressure_Pa=P_ATM,
        dx_m=0.01,
        dt_s=0.02,
        gas_presence_mass_kg_m=1.0e-12,
        gas_presence_area_m2=1.0e-12,
    )
    with pytest.raises(ContractViolation, match="area and mass must be paired"):
        water_end_inlet_outlet_gas_flux(
            gas_area_m2=2.0e-4,
            gas_mass_kg_m=0.0,
            **common,
        )
    with pytest.raises(ContractViolation, match="area and mass must be paired"):
        water_end_inlet_outlet_gas_flux(
            gas_area_m2=0.0,
            gas_mass_kg_m=3.0e-4,
            **common,
        )


@pytest.mark.parametrize(
    ("side", "velocity"), (("left", 0.2), ("right", -0.2))
)
def test_water_end_inletOutlet_blocks_unpublished_gas_reentry(side, velocity) -> None:
    mass_per_length = 3.0e-4
    flux = water_end_inlet_outlet_gas_flux(
        side=side,
        gas_area_m2=2.0e-4,
        gas_mass_kg_m=mass_per_length,
        gas_momentum_kg_s=mass_per_length * velocity,
        interior_absolute_pressure_Pa=P_ATM,
        dx_m=0.01,
        dt_s=0.02,
        gas_presence_mass_kg_m=1.0e-12,
        gas_presence_area_m2=1.0e-12,
    )

    assert flux.gas_mass_left_to_right_kg_s == 0.0
    assert flux.gas_inflow_kg_s == 0.0
    assert flux.gas_outflow_kg_s == 0.0
    assert flux.mode == "pure_water_reentry_no_gas_inventory"


@pytest.mark.parametrize(
    ("side", "velocity"), (("left", -0.2), ("right", 0.2))
)
def test_gas_reaching_either_water_end_closes_external_gas_and_momentum_ledgers(
    side, velocity
) -> None:
    component, assembly = _assembly_with_end_gas(
        side=side, gas_velocity_m_s=velocity
    )
    token = state_token(assembly.state)
    trials = _trials(component, assembly)
    proposal = _propose(component, assembly, trials)

    assert proposal.committable
    assert proposal.external_exchange.gas_inflow_kg_s == 0.0
    assert proposal.external_exchange.gas_outflow_kg_s > 0.0
    dx = assembly.geometry.horizontal_dx_m
    observed_gas = sum(
        rate * width for rate, width in zip(proposal.delta.Mg, dx, strict=True)
    )
    assert observed_gas == pytest.approx(
        -proposal.external_exchange.gas_outflow_kg_s, abs=2.0e-11
    )
    assert math.isfinite(proposal.external_exchange.mixture_momentum_x_net_rate)
    assert state_token(assembly.state) == token
    assert _propose(component, assembly, trials) == proposal
    assert state_token(assembly.state) == token


@pytest.mark.parametrize(
    ("side", "velocity"), (("left", 0.2), ("right", -0.2))
)
def test_water_reentry_after_gas_arrival_is_finite_and_never_imports_gas(
    side, velocity
) -> None:
    component, assembly = _assembly_with_end_gas(
        side=side, gas_velocity_m_s=velocity
    )
    proposal = _propose(component, assembly, _trials(component, assembly))

    assert proposal.committable
    assert proposal.external_exchange.gas_inflow_kg_s == 0.0
    assert proposal.external_exchange.gas_outflow_kg_s == 0.0
    assert math.isfinite(proposal.external_exchange.liquid_inflow_m3_s)
    assert proposal.external_exchange.liquid_inflow_m3_s > 0.0


def test_unbracketed_upstream_water_reentry_characteristic_still_fails_closed() -> None:
    component, assembly = _assembly_with_end_gas(side="left", gas_velocity_m_s=0.0)
    areas = list(assembly.state.horizontal.Al)
    gas = list(assembly.state.horizontal.Mg)
    momentum = list(assembly.state.horizontal.Jg)
    areas[0] = 0.2 * component.area_m2
    gas_area = component.area_m2 - areas[0]
    gas[0] = P_ATM * gas_area / component.config.rt_J_kg
    momentum[0] = 0.0
    state = HorizontalState(
        Al=tuple(areas),
        Ql=assembly.state.horizontal.Ql,
        Mg=tuple(gas),
        Jg=tuple(momentum),
    )

    with pytest.raises(ContractViolation, match="no bracketed subcritical"):
        component._table1_pressure_ghosts(component._arrays(state))
