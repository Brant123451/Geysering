from __future__ import annotations

import math

import pytest

from casea_material_front_cutcell import (
    ALEInterfaceFlux,
    PressurisedState,
    StratifiedState,
)
from casea_paper_material_front_rh import (
    AffineGasPressureLaw,
    PaperFrontPhysics,
    candidate_to_ale_traces,
    enumerate_paper_front_candidates,
    evaluate_candidate_from_pressurised_area,
)


def _exact_vertical_material_state(*, speed: float = 0.4):
    physics = PaperFrontPhysics(
        diameter=0.094,
        liquid_wave_speed=100.0,
        cos_inclination=0.0,
        gas_sound_speed=300.0,
    )
    af = physics.full_area
    al = 0.60 * af
    ap = 1.001 * af
    # With u_p=u_l=w, vertical zeta=0, and no slip, RH momentum gives this
    # absolute gas pressure exactly.
    pressure = physics.reference_pressure + (
        physics.liquid_density
        * physics.liquid_wave_speed**2
        * (ap - af)
        / al
    )
    ag = af - al
    rho_g = pressure / physics.gas_sound_speed**2
    pressurised = PressurisedState(ap, ap * speed)
    stratified = StratifiedState(
        rho_g * ag,
        rho_g * ag * speed,
        al,
        al * speed,
    )
    return physics, ap, pressure, pressurised, stratified


def test_vertical_paper_candidate_satisfies_all_dimensional_balances() -> None:
    physics, ap, pressure, pressurised, stratified = (
        _exact_vertical_material_state()
    )
    candidate = evaluate_candidate_from_pressurised_area(
        ap,
        pressurised_foot=pressurised,
        stratified_foot=stratified,
        pressurised_side="right",
        pressure_law=AffineGasPressureLaw.fixed(pressure),
        physics=physics,
    )
    assert candidate.speed == pytest.approx(0.4)
    assert candidate.characteristic_residual == pytest.approx(0.0, abs=1e-13)
    assert candidate.liquid_mass_residual == pytest.approx(0.0, abs=1e-13)
    assert candidate.liquid_momentum_residual == pytest.approx(0.0, abs=1e-11)
    assert candidate.lambda_d > 0.0
    assert candidate.active_set == "middle"


def test_polynomial_enumerator_recovers_exact_reduced_paper_root() -> None:
    physics, ap, pressure, pressurised, stratified = (
        _exact_vertical_material_state()
    )
    candidates = enumerate_paper_front_candidates(
        pressurised_foot=pressurised,
        stratified_foot=stratified,
        pressurised_side="right",
        pressure_law=AffineGasPressureLaw.fixed(pressure),
        physics=physics,
    )
    assert any(
        item.pressurised_area == pytest.approx(ap, rel=2e-7)
        and item.speed == pytest.approx(0.4, rel=2e-7)
        for item in candidates
    )


def test_cutcell_trace_has_absolute_gas_piston_impulse_and_exact_liquid_ale_flux() -> None:
    physics, ap, pressure, pressurised, stratified = (
        _exact_vertical_material_state()
    )
    candidate = evaluate_candidate_from_pressurised_area(
        ap,
        pressurised_foot=pressurised,
        stratified_foot=stratified,
        pressurised_side="right",
        pressure_law=AffineGasPressureLaw.fixed(pressure),
        physics=physics,
    )
    traces = candidate_to_ale_traces(candidate, physics=physics)
    ale = ALEInterfaceFlux.from_traces(
        traces,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )
    gas_area = physics.full_area - stratified.liquid_area
    assert ale.gas_mass == 0.0
    assert ale.gas_momentum == pytest.approx(pressure * gas_area)
    assert ale.liquid_area_residual == pytest.approx(0.0, abs=1e-12)
    assert ale.liquid_momentum_residual == pytest.approx(0.0, abs=1e-10)


def test_acoustic_pressure_law_matches_paper_linear_characteristic_without_floor() -> None:
    physics, ap, pressure, pressurised, stratified = (
        _exact_vertical_material_state()
    )
    speed = 0.4
    gas_velocity_cell = 0.0
    pressure_cell = pressure / (
        1.0 + (speed - gas_velocity_cell) / physics.gas_sound_speed
    )
    gas_area = physics.full_area - stratified.liquid_area
    density_cell = pressure_cell / physics.gas_sound_speed**2
    acoustic_stratified = StratifiedState(
        density_cell * gas_area,
        density_cell * gas_area * gas_velocity_cell,
        stratified.liquid_area,
        stratified.liquid_discharge,
    )
    law = AffineGasPressureLaw.from_acoustic_trace(
        density=density_cell,
        velocity=gas_velocity_cell,
        sound_speed=physics.gas_sound_speed,
    )
    candidate = evaluate_candidate_from_pressurised_area(
        ap,
        pressurised_foot=pressurised,
        stratified_foot=acoustic_stratified,
        pressurised_side="right",
        pressure_law=law,
        physics=physics,
    )
    assert candidate.gas_pressure_absolute == pytest.approx(pressure)
    assert candidate.gas_pressure_residual == 0.0
    assert candidate.liquid_momentum_residual == pytest.approx(0.0, abs=1e-10)


def test_no_hidden_two_wave_speed_cap() -> None:
    speed = 250.0
    physics, ap, pressure, pressurised, stratified = (
        _exact_vertical_material_state(speed=speed)
    )
    assert speed > 2.0 * physics.liquid_wave_speed
    candidate = evaluate_candidate_from_pressurised_area(
        ap,
        pressurised_foot=pressurised,
        stratified_foot=stratified,
        pressurised_side="right",
        pressure_law=AffineGasPressureLaw.fixed(pressure),
        physics=physics,
    )
    traces = candidate_to_ale_traces(candidate, physics=physics)
    ale = ALEInterfaceFlux.from_traces(
        traces,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-10,
    )
    assert candidate.speed == pytest.approx(speed)
    assert math.isfinite(ale.liquid_momentum)
