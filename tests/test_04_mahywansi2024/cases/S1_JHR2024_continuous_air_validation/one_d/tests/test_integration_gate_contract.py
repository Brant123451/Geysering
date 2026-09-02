from pathlib import Path

import yaml

from model.horizontal_two_tee_component import F0HorizontalTwoTeeStageComponent
from model.joint_network_runner import build_current_physical_operator
from model.physical_joint_owner import F0PhysicalTwoTNodeStageOwner
from model.simultaneous_two_tnode_solver import F0SimultaneousTwoTNodeSolver
from model.state import HorizontalState, VerticalState
from model.supply_branch_twophase import SupplyBranchTwoPhaseSolver
from model.vertical_pressure_void_component import F0VerticalPressureVoidStageComponent


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_alignment_contract_uses_only_declared_canonical_scalar_series() -> None:
    acceptance = _yaml("ACCEPTANCE.yaml")
    observables = _yaml("config/COMMON_OBSERVABLES.yaml")
    gate = acceptance["one_d_to_2d_gate"]
    required = set(gate["horizontal_required_series"])
    required.update(gate["vertical_required_series"])
    required.update(gate["pressure_required_series"])
    canonical = set(observables["canonical_series"])
    assert required <= canonical


def test_vertical_profile_contract_preserves_all_six_prognostic_states() -> None:
    acceptance = _yaml("ACCEPTANCE.yaml")
    observables = _yaml("config/COMMON_OBSERVABLES.yaml")
    required = set(acceptance["one_d_to_2d_gate"]["vertical_required_profile_fields"])
    declared = set(observables["required_profile_fields"])
    assert required <= declared
    assert required >= {
        "riser_Aup_m2",
        "riser_Qup_m3_s",
        "riser_Adown_m2",
        "riser_Qdown_m3_s",
        "riser_gas_mass_kg_m",
        "riser_gas_velocity_m_s",
    }
    assert tuple(VerticalState.__dataclass_fields__) == (
        "Aup",
        "Qup",
        "Adown",
        "Qdown",
        "Mg",
        "Jg",
    )
    assert tuple(HorizontalState.__dataclass_fields__) == ("Al", "Ql", "Mg", "Jg")


def test_current_physical_bundle_exposes_real_components_but_stays_closed() -> None:
    operator = build_current_physical_operator()
    assert isinstance(operator.horizontal_component, F0HorizontalTwoTeeStageComponent)
    assert isinstance(operator.supply_branch_component, SupplyBranchTwoPhaseSolver)
    assert isinstance(operator.vertical_component, F0VerticalPressureVoidStageComponent)
    assert isinstance(operator.two_tnode_solver, F0SimultaneousTwoTNodeSolver)
    assert isinstance(operator.joint_stage_owner, F0PhysicalTwoTNodeStageOwner)
    assert operator.integration_owner_ready is True
    assert operator.production_ready is False
    assert operator.horizontal_component.production_ready is False
    assert operator.supply_branch_component.production_ready is False
    assert operator.vertical_component.production_ready is False
    assert operator.two_tnode_solver.production_ready is False
    assert operator.joint_stage_owner.production_ready is False


def test_supply_component_implements_frozen_f0_wall_shear_without_promotion() -> None:
    f0 = _yaml("config/S1_1D_F0_closures.yaml")
    wall = f0["air_supply_branch"]["wall_shear"]
    assert wall["model"] == "smooth_pipe_darcy"
    assert wall["implementation_status"].startswith("implemented_sign_preserving")
    component = SupplyBranchTwoPhaseSolver()
    assert "smooth_pipe_Darcy" in component.config.closure_provenance
    assert component.production_ready is False


def test_table1_pressure_quantities_are_consistent_across_frozen_contracts() -> None:
    source = _yaml("config/S1_source_aligned.yaml")
    f0 = _yaml("config/S1_1D_F0_closures.yaml")
    acceptance = _yaml("ACCEPTANCE.yaml")

    boundaries = source["boundaries"]
    assert boundaries["water_inlet_head_m"]["quantity"] == "total_pressure_head"
    assert boundaries["water_outlet_head_m"]["quantity"] == "static_pressure_head"
    assert boundaries["one_d_native_top_mapping"]["pressure_quantity"] == "static_pressure"
    assert (
        boundaries["source_environment_boundaries"]["boundary_4"]["pressure_quantity"]
        == "static_pressure_outlet"
    )
    assert (
        boundaries["source_environment_boundaries"]["boundaries_5_and_6"]
        ["pressure_quantity"]
        == "static_pressure_outlet"
    )

    table1 = f0["horizontal_main"]["table1_water_pressure_boundaries"]
    assert table1["inlet_source_semantics"] == "published_pressure_inlet_total_head"
    assert table1["outlet_source_semantics"] == "published_pressure_outlet_static_head"
    assert "u^2/(2g)" in table1["inlet_imposed_relation"]
    assert "u^2/(2g)" not in table1["outlet_imposed_relation"]
    assert f0["riser_top_boundary"]["pressure_quantity"] == "static_absolute_pressure"

    numeric_gate = acceptance["source_boundary_numeric_gate"]
    assert numeric_gate["water_boundary_quantities"] == {
        "inlet": "total_pressure_head",
        "outlet": "static_pressure_head",
    }
    assert numeric_gate["one_d_characteristic_translations_required"] == {
        "inlet": "outgoing_invariant_plus_total_head",
        "outlet": "outgoing_invariant_plus_static_head",
    }
    assert numeric_gate["vertical_top_boundary_quantity"] == "static_pressure"
