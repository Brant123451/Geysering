import math

import pytest

from model.errors import ContractViolation
from model.pressure_reservoir import (
    DECLARED_DRY_AIR_GAS_CONSTANT_J_KG_K,
    DECLARED_REFERENCE_TEMPERATURE_K,
    IsothermalIdealGasPressureReservoir,
)


def test_declared_reference_and_published_pressure_translation() -> None:
    source = IsothermalIdealGasPressureReservoir()

    assert source.reservoir_absolute_pressure_Pa == pytest.approx(101325.0 + 5700.0)
    assert source.temperature_K == DECLARED_REFERENCE_TEMPERATURE_K == 293.15
    assert source.gas_constant_J_kg_K == DECLARED_DRY_AIR_GAS_CONSTANT_J_KG_K == 287.05
    assert source.pressure_evidence_status == "published_gauge_pressure"
    assert source.thermodynamic_reference_status == "declared_OpenFOAM_consistent_reference"
    assert source.valve_model_status.startswith("not_reproduced__")


def test_equilibrium_has_zero_mass_flux_and_static_pressure_momentum_flux() -> None:
    source = IsothermalIdealGasPressureReservoir()
    area = 2.5e-5
    result = source.evaluate(
        node_absolute_pressure_Pa=source.reservoir_absolute_pressure_Pa,
        node_axial_velocity_m_s=0.0,
        inlet_area_m2=area,
    )

    assert result.mass_flux_kg_m2_s == pytest.approx(0.0, abs=1.0e-14)
    assert result.mass_flow_kg_s == pytest.approx(0.0, abs=1.0e-18)
    assert result.axial_momentum_pressure_flux_Pa == pytest.approx(
        source.reservoir_absolute_pressure_Pa
    )
    assert result.axial_momentum_pressure_rate_N == pytest.approx(
        source.reservoir_absolute_pressure_Pa * area
    )


def test_pressure_difference_selects_inflow_or_backflow_without_clipping() -> None:
    source = IsothermalIdealGasPressureReservoir()
    area = 1.0e-5

    inflow = source.evaluate(
        node_absolute_pressure_Pa=101325.0,
        node_axial_velocity_m_s=0.0,
        inlet_area_m2=area,
    )
    backflow = source.evaluate(
        node_absolute_pressure_Pa=112000.0,
        node_axial_velocity_m_s=0.0,
        inlet_area_m2=area,
    )

    assert inflow.mass_flux_kg_m2_s > 0.0
    assert inflow.mass_flow_kg_s > 0.0
    assert backflow.mass_flux_kg_m2_s < 0.0
    assert backflow.mass_flow_kg_s < 0.0
    for result in (inflow, backflow):
        assert all(
            math.isfinite(value)
            for value in (
                result.mass_flux_kg_m2_s,
                result.mass_flow_kg_s,
                result.axial_momentum_pressure_flux_Pa,
                result.axial_momentum_pressure_rate_N,
            )
        )


def test_caller_supplied_area_scales_rates_but_not_flux_densities() -> None:
    source = IsothermalIdealGasPressureReservoir()
    common = dict(node_absolute_pressure_Pa=101325.0, node_axial_velocity_m_s=0.0)
    small = source.evaluate(inlet_area_m2=1.0e-5, **common)
    large = source.evaluate(inlet_area_m2=4.0e-5, **common)

    assert large.mass_flux_kg_m2_s == pytest.approx(small.mass_flux_kg_m2_s)
    assert large.axial_momentum_pressure_flux_Pa == pytest.approx(
        small.axial_momentum_pressure_flux_Pa
    )
    assert large.mass_flow_kg_s == pytest.approx(4.0 * small.mass_flow_kg_s)
    assert large.axial_momentum_pressure_rate_N == pytest.approx(
        4.0 * small.axial_momentum_pressure_rate_N
    )


def test_inlet_area_is_mandatory_and_invalid_inputs_fail_closed() -> None:
    source = IsothermalIdealGasPressureReservoir()
    with pytest.raises(TypeError):
        source.evaluate(  # type: ignore[call-arg]
            node_absolute_pressure_Pa=101325.0,
            node_axial_velocity_m_s=0.0,
        )
    with pytest.raises(ContractViolation, match="inlet area must be positive"):
        source.evaluate(
            node_absolute_pressure_Pa=101325.0,
            node_axial_velocity_m_s=0.0,
            inlet_area_m2=0.0,
        )
    with pytest.raises(ContractViolation, match="node absolute pressure must be positive"):
        source.evaluate(
            node_absolute_pressure_Pa=0.0,
            node_axial_velocity_m_s=0.0,
            inlet_area_m2=1.0e-5,
        )
    with pytest.raises(ContractViolation, match="must be finite"):
        source.evaluate(
            node_absolute_pressure_Pa=101325.0,
            node_axial_velocity_m_s=math.inf,
            inlet_area_m2=1.0e-5,
        )
