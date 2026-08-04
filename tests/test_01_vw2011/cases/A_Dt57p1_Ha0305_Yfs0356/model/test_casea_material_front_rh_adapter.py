"""Coupled RH/ALE regression tests for the Case-A S|P fronts."""

from __future__ import annotations

import math
from pathlib import Path
import sys


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_material_front_cutcell import (  # noqa: E402
    ALEInterfaceFlux,
    MaterialFrontCutCell,
    OuterFaceFluxes,
    PressurisedState,
    StratifiedState,
    advance_material_front_cutcell,
)
from casea_material_front_rh_adapter import (  # noqa: E402
    build_casea_material_front_traces,
)
from casea_tjunction_shock_network import BranchGeometry  # noqa: E402


RHO_L = 998.0
G = 9.81
P_ATM = 101_325.0
CG2 = 287.05 * 293.0


def _states(head_offset: float = 0.0):
    geometry = BranchGeometry(0.094, 0.490, 28.0)
    section = geometry.section(G)
    equilibrium_head = 0.3780169412545817
    pressure = P_ATM + RHO_L * G * (equilibrium_head + head_offset)
    al = float(section.area_from_depth(0.070))
    pressurised = PressurisedState(
        float(section.area_from_head(0.450)), 0.0
    )
    stratified = StratifiedState(
        gas_mass=(pressure / CG2) * (section.full_area - al),
        gas_momentum=0.0,
        liquid_area=al,
        liquid_discharge=0.0,
    )
    return geometry, pressurised, stratified


def test_adapter_returns_one_rh_ale_flux_and_zero_material_gas_flux() -> None:
    geometry, pressurised, stratified = _states()
    result = build_casea_material_front_traces(
        pressurised,
        stratified,
        front_position=0.10,
        geometry=geometry,
    )
    ale = ALEInterfaceFlux.from_traces(result.traces)
    assert abs(result.closure.interface_speed) < 2.0e-10
    assert abs(ale.liquid_area_residual) < 1.0e-12
    assert abs(ale.liquid_momentum_residual) < 1.0e-12
    assert ale.gas_mass == 0.0
    assert ale.gas_material_residual == 0.0


def test_adapter_front_direction_is_set_by_physical_pressure_state() -> None:
    speeds = []
    for offset in (-0.05, 0.05):
        geometry, pressurised, stratified = _states(offset)
        result = build_casea_material_front_traces(
            pressurised,
            stratified,
            front_position=0.10,
            geometry=geometry,
        )
        ALEInterfaceFlux.from_traces(result.traces)
        speeds.append(result.closure.interface_speed)
    assert speeds[0] < -0.08
    assert speeds[1] > 0.08


def test_zero_length_casea_stratified_host_grows_from_boundary_flux() -> None:
    geometry, pressurised, stratified = _states(0.05)
    built = build_casea_material_front_traces(
        pressurised,
        stratified,
        front_position=0.0,
        geometry=geometry,
    )
    assert built.traces.speed > 0.0
    faces = (0.0, 0.04, 0.08)
    state = MaterialFrontCutCell(
        cell_faces=faces,
        host_index=0,
        front_position=0.0,
        pressurised_side="right",
        pressurised=pressurised,
        # This trace has exactly zero initial inventory.
        stratified=built.traces.stratified_state,
    )
    dt = 1.0e-3
    advanced = advance_material_front_cutcell(
        state,
        dt,
        interface_provider=lambda _state, _time: built.traces,
        outer_flux_provider=lambda _state, _time: OuterFaceFluxes(
            pressurised=built.traces.pressurised_flux,
            stratified=built.traces.stratified_flux,
        ),
    )
    assert math.isclose(
        advanced.state.front_position,
        built.traces.speed * dt,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    assert advanced.state.stratified_length > 0.0
    expected_gas = (
        dt * built.traces.stratified_flux.gas_mass
    )
    assert math.isclose(
        advanced.state.inventory().gas_mass,
        expected_gas,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )
    ledger = advanced.ledgers[0]
    assert max(abs(value) for value in ledger.residual.vector()) < 1.0e-14

