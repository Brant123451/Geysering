import math

import pytest

from model.initialization import (
    S1_AIR_VISCOSITY_PA_S,
    S1_HORIZONTAL_ELASTIC_OVERAREA_FRACTION,
    S1_HORIZONTAL_ELASTIC_REFERENCE_HEAD_M,
    S1_HORIZONTAL_INITIAL_HEAD_M,
    S1_SUPPLY_BRANCH_LENGTH_M,
    S1_WATER_DENSITY_KG_M3,
    S1_WATER_VISCOSITY_PA_S,
    build_s1_initial_assembly,
)


def test_source_aligned_initial_assembly_and_inventories() -> None:
    assembly = build_s1_initial_assembly()
    area = math.pi * 0.0254**2 / 4.0
    horizontal_area = assembly.state.horizontal.Al[0]
    rho_air = 101325.0 / (287.05 * 293.15)

    assert assembly.state.horizontal.cell_count == 310
    assert assembly.state.vertical.cell_count == 160
    assert assembly.state.supply_branch.cell_count == 14
    assert horizontal_area > area
    assert assembly.state.horizontal.Al == pytest.approx((horizontal_area,) * 310)
    represented_head = (
        S1_HORIZONTAL_ELASTIC_REFERENCE_HEAD_M
        + assembly.horizontal_adapter.head_from_area_m(horizontal_area)
        - 0.0254
    )
    assert represented_head == pytest.approx(S1_HORIZONTAL_INITIAL_HEAD_M, abs=1.0e-10)
    assert assembly.state.supply_branch.Al == pytest.approx((area,) * 14)
    assert assembly.state.supply_branch.Mg == pytest.approx((0.0,) * 14)
    assert sum(assembly.state.vertical.Aup) * assembly.vertical_adapter.cell_length_m == pytest.approx(
        area * 0.5842
    )
    assert assembly.vertical_adapter.initial.represented_gas_mass_kg == pytest.approx(
        rho_air * area * (1.02 - 0.5842)
    )
    assert assembly.inventory.liquid_volume_m3 == pytest.approx(
        horizontal_area * 3.10
        + area * (S1_SUPPLY_BRANCH_LENGTH_M + 0.5842)
    )
    assert assembly.inventory.gas_mass_kg == pytest.approx(
        rho_air * area * (1.02 - 0.5842)
    )
    assert assembly.state.air_supply_node.gas_mass == 0.0
    assert assembly.state.riser_node.liquid_volume == 0.0
    assert assembly.geometry.horizontal_elastic_overarea_fraction == pytest.approx(
        S1_HORIZONTAL_ELASTIC_OVERAREA_FRACTION
    )


def test_initial_assembly_keeps_stage1_source_closed_and_declares_reductions() -> None:
    assembly = build_s1_initial_assembly()
    assert assembly.stage == "stage1_closed_air_pressure_driven_water_settling"
    assert assembly.air_source_open is False
    assert "zero_storage" in assembly.node_storage_status
    assert "0p1373m_water_initial_two_phase_Al_Ql_Mg_Jg_branch" in assembly.air_stub_status
    assert "real_six_port_joint_owner_integrated" in assembly.air_stub_status
    assert "global_production_blocked" in assembly.air_stub_status
    assert assembly.production_ready is False
    assert "not_effective_valve_area" in assembly.pressure_reservoir_inlet_area_status
    assert assembly.pressure_reservoir.reservoir_absolute_pressure_Pa == pytest.approx(107025.0)
    assert assembly.pressure_reservoir_inlet_area_m2 == pytest.approx(
        math.pi * 0.0254**2 / 4.0
    )


def test_declared_2d_consistent_material_properties_are_explicit() -> None:
    assert S1_WATER_DENSITY_KG_M3 == pytest.approx(998.4)
    assert S1_WATER_VISCOSITY_PA_S == pytest.approx(1.002e-3)
    assert S1_AIR_VISCOSITY_PA_S == pytest.approx(1.78e-5)
    assembly = build_s1_initial_assembly()
    assert assembly.geometry.liquid_density_kg_m3 == pytest.approx(998.4)
