from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from functools import lru_cache
import inspect
import math

import numpy as np
import pytest

import case1_local_valve_extension as extension_module
from case1_local_valve_extension import (
    CORE_SOURCE,
    EXPECTED_CORE_SHA256,
    Case1LocalValveExtension,
    CleanFreeSurfaceValveDonorLimitRejected,
    CleanFreeSurfaceValveTraceRejected,
    FixedInternalValveSpec,
    InternalFaceFluxPair,
    InternalFaceStageContext,
    IntegratedValveTransaction,
    LocalValveRegime,
    PressurisedMocValveStageRejected,
    PressurisedMocValveStageSolution,
    ShockCutValveDonorLimitRejected,
    ShockCutValveCflRejected,
    ShockCutValveNonlinearRejected,
    ShockCutValveOrientation,
    ShockCutValveStageSolution,
    internal_face_wet_dry_ssprk2_step,
    plan_event_aligned_step,
    source_sha256,
)
from campaign2_local_valve import FreeSurfaceValveControlRegime
from tosan2021_horizontal_shockfit import (
    CircularSection,
    HorizontalConfig,
    HorizontalState,
    Tosan2021HorizontalShockFit,
    WetDryState,
    central_upwind_wet_dry_step,
)


def campaign2_horizontal_config() -> HorizontalConfig:
    return HorizontalConfig(
        length=6.590,
        diameter=0.050,
        valve_x=0.610,
        vent_x=3.120,
        initial_air_head=0.0,
        initial_water_head=0.660,
        dx=0.010,
        wave_speed=28.0,
        gravity=9.81,
        liquid_density=998.0,
        atmospheric_pressure=101_325.0,
        gas_constant=287.05,
        temperature=296.15,
        right_boundary="transmissive",
    )


def campaign2_valve_spec(**overrides) -> FixedInternalValveSpec:
    values = {
        "physical_domain_length_m": 6.590,
        "physical_valve_x_m": 5.980,
        "grid_dx_m": 0.010,
        "pipe_diameter_m": 0.050,
        "expected_mirrored_face_index": 61,
    }
    values.update(overrides)
    return FixedInternalValveSpec(**values)


@lru_cache(maxsize=None)
def baseline_state_at(time_s: float) -> HorizontalState:
    solver = Tosan2021HorizontalShockFit(campaign2_horizontal_config())
    return solver.step(solver.case_b_initial_state(), time_s)


def resolvable_clean_fv_state(
    time_s: float,
    *,
    discharge_scale: float = 0.8,
) -> HorizontalState:
    state = baseline_state_at(time_s)
    discharge = state.discharge.copy()
    cut = int(math.floor(state.interface_x / 0.010))
    discharge[: cut + 1] *= discharge_scale
    return replace(
        state,
        area=state.area.copy(),
        discharge=discharge,
    )


def pressurised_moc_restart_state(
    extension: Case1LocalValveExtension,
    *,
    cut: int,
    time_s: float = 0.10,
    west_head_m: float = 0.680,
    east_head_m: float = 0.640,
    velocity_m_s: float = 0.0,
) -> HorizontalState:
    """Build an exact dry/full restart with the valve inside the MOC reach."""

    if cut > 59 or cut < 2:
        raise ValueError("the helper is only for a clean pressurised-MOC split")
    state = extension.case_b_initial_state()
    face = extension.local_face.mirrored_face_index
    interface_x = (cut + 0.5) * extension.dx
    west_area = float(extension.section.area_from_head(west_head_m))
    east_area = float(extension.section.area_from_head(east_head_m))
    area = np.zeros_like(state.area)
    discharge = np.zeros_like(state.discharge)
    pressurised_fraction = float(
        (cut * extension.dx + extension.dx - interface_x) / extension.dx
    )
    area[cut] = pressurised_fraction * west_area
    area[cut + 1 : face] = west_area
    area[face:] = east_area
    discharge[cut] = pressurised_fraction * west_area * velocity_m_s
    discharge[cut + 1 : face] = west_area * velocity_m_s
    discharge[face:] = east_area * velocity_m_s
    gas_volume = extension.section.full_area * interface_x
    gas = state.gas.with_volume(gas_volume)
    return replace(
        state,
        time=time_s,
        area=area,
        discharge=discharge,
        gas=gas,
        air_pressure_abs=gas.pressure_abs,
        interface_x=interface_x,
        interface_speed=0.0,
        interface_pressurised_head=west_head_m,
        interface_pressurised_velocity=velocity_m_s,
        wetting_front_x=interface_x,
    )


def assert_horizontal_states_bitwise_equal(
    actual: HorizontalState,
    expected: HorizontalState,
) -> None:
    for item in fields(HorizontalState):
        actual_value = getattr(actual, item.name)
        expected_value = getattr(expected, item.name)
        if isinstance(actual_value, np.ndarray):
            assert np.array_equal(actual_value, expected_value), item.name
        else:
            assert actual_value == expected_value, item.name


def assert_wet_dry_states_bitwise_equal(
    actual: WetDryState,
    expected: WetDryState,
) -> None:
    assert np.array_equal(actual.area, expected.area)
    assert np.array_equal(actual.discharge, expected.discharge)


def fv_test_state(section: CircularSection) -> WetDryState:
    full = section.full_area
    return WetDryState(
        area=full
        * np.array((0.0, 0.35, 0.55, 0.70, 0.82, 1.01, 1.02, 0.91)),
        discharge=np.array(
            (0.0, -1.0e-5, -2.0e-5, -1.4e-5, -0.8e-5, 0.3e-5, 0.5e-5, 0.2e-5)
        ),
    )


def test_original_case1_core_hash_is_unchanged_and_pinned() -> None:
    assert EXPECTED_CORE_SHA256 == (
        "90e84da9afa0ec8465d80f87fc701dfb8f0fad6f97350ea708074a50192b6119"
    )
    assert source_sha256(CORE_SOURCE) == EXPECTED_CORE_SHA256


def test_none_step_is_bitwise_identical_for_multiple_states_and_subcycles() -> None:
    config = campaign2_horizontal_config()
    baseline = Tosan2021HorizontalShockFit(config)
    extension = Case1LocalValveExtension(config, local_face=None)
    baseline_state = baseline.case_b_initial_state(
        initial_air_gauge_head=0.0,
        initial_water_head=0.660,
    )
    extension_state = extension.case_b_initial_state(
        initial_air_gauge_head=0.0,
        initial_water_head=0.660,
    )
    assert_horizontal_states_bitwise_equal(extension_state, baseline_state)

    stable = baseline.stable_timestep(baseline_state)
    requested_steps = (1.0e-4, 2.25 * stable, 2.0e-4, 3.10 * stable)
    assert any(step > stable for step in requested_steps)
    for step in requested_steps:
        baseline_state = baseline.step(baseline_state, step)
        extension_state = extension.step(extension_state, step)
        assert_horizontal_states_bitwise_equal(extension_state, baseline_state)


def test_none_advance_result_wraps_the_exact_core_state_and_zero_transaction() -> None:
    config = campaign2_horizontal_config()
    baseline = Tosan2021HorizontalShockFit(config)
    extension = Case1LocalValveExtension(config, local_face=None)
    baseline_state = baseline.case_b_initial_state()
    extension_state = extension.case_b_initial_state()
    dt = 2.5 * baseline.stable_timestep(baseline_state)

    expected = baseline.step(baseline_state, dt)
    result = extension.advance_with_transaction(extension_state, dt)

    assert_horizontal_states_bitwise_equal(result.state, expected)
    transaction = result.valve_transaction
    assert transaction.start_time_s == extension_state.time
    assert transaction.end_time_s == expected.time
    assert transaction.physical_signed_through_volume_m3 == 0.0
    assert transaction.physical_wall_impulse_on_liquid_N_s == 0.0
    assert transaction.dissipated_energy_J == 0.0
    assert transaction.stage_evaluation_count == 0
    assert transaction.substep_count == 0


def test_none_dispatch_contains_no_alternate_elapsed_time_path() -> None:
    source = inspect.getsource(Case1LocalValveExtension.step)
    module_source = inspect.getsource(extension_module)
    assert "if self.local_face is None" in source
    assert "return super().step" in source
    assert "hydraulic_dt" not in module_source
    assert "transmissivity" not in module_source


def test_face61_geometry_area_and_physical_mirror_signs_are_exact() -> None:
    spec = campaign2_valve_spec()
    solver = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=spec,
    )

    assert spec.mirrored_face_index == 61
    assert spec.mirrored_valve_x_m == pytest.approx(0.610, abs=1.0e-15)
    assert spec.mirrored_face_x_m == pytest.approx(0.610, abs=1.0e-15)
    assert spec.physical_face_x_m == pytest.approx(5.980, abs=1.0e-15)
    assert spec.valve_flow_area_m2 == pytest.approx(
        0.25 * math.pi * 0.050**2,
        rel=0.0,
        abs=2.0e-18,
    )
    assert spec.valve_flow_area_m2 == pytest.approx(
        solver.section.full_area,
        rel=0.0,
        abs=2.0e-18,
    )
    assert spec.valve_flow_area_m2 != pytest.approx(0.050 * 0.001)
    assert spec.mirrored_volume_from_physical(3.0e-6) == -3.0e-6
    assert spec.physical_volume_from_mirrored(-3.0e-6) == 3.0e-6
    assert spec.mirrored_wall_impulse_from_physical(-2.0) == 2.0
    assert spec.physical_wall_impulse_from_mirrored(2.0) == -2.0
    assert spec.mirrored_face_momentum_impulses_from_physical(
        physical_left_N_s=4.0,
        physical_right_N_s=7.0,
    ) == (7.0, 4.0)


def test_non_face_geometry_wrong_face_and_solver_mismatch_are_rejected() -> None:
    with pytest.raises(ValueError, match="does not coincide"):
        campaign2_valve_spec(physical_valve_x_m=5.979)
    with pytest.raises(ValueError, match="must be mirrored face"):
        campaign2_valve_spec(expected_mirrored_face_index=60)
    with pytest.raises(ValueError, match="opening duration must be 0.20 s"):
        campaign2_valve_spec(opening_duration_s=0.19)

    wrong_density = replace(campaign2_horizontal_config(), liquid_density=998.2)
    with pytest.raises(ValueError, match="liquid density differs"):
        Case1LocalValveExtension(
            wrong_density,
            local_face=campaign2_valve_spec(),
        )


def test_active_face_classifies_pressurised_region_and_rejects_bad_restart() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    original = extension.case_b_initial_state()
    state = replace(original, interface_x=0.595)
    partition = extension.classify_local_face_regime(state)
    assert partition.regime is LocalValveRegime.CLEAN_PRESSURISED_MOC
    assert partition.fixed_face_index == 61
    assert partition.shock_cut_cell_index == 59
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()

    plan = extension.plan_physical_step(state, 1.0e-4)
    with pytest.raises(PressurisedMocValveStageRejected) as caught:
        extension._solve_pressurised_moc_stage(
            state,
            dt=1.0e-4,
            stage_index=1,
            partition=partition,
            step_plan=plan,
        )

    assert caught.value.partition == partition
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
    assert state.time == original.time


def test_real_initial_cut61_coupled_balance_and_physical_mirror_mapping(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    volume_before = float(np.sum(state.area) * extension.dx)
    captured: list[ShockCutValveStageSolution] = []
    original_solver = extension._solve_shock_cut_stage

    def capture(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        if kwargs.get("orientation_override") is None:
            captured.append(solution)
        return solution

    monkeypatch.setattr(extension, "_solve_shock_cut_stage", capture)
    result = extension.advance_with_transaction(state, 1.0e-4)
    transaction = result.valve_transaction

    assert result.state.time == pytest.approx(1.0e-4, abs=2.0e-19)
    assert len(captured) == 2
    assert captured[0].orientation is ShockCutValveOrientation.DRY_FACE_INJECTION
    assert captured[1].orientation is ShockCutValveOrientation.DRY_FACE_INJECTION
    assert all(stage.interface_speed_m_s == 0.0 for stage in captured)
    assert all(stage.shared_mass_flux_m3_s < 0.0 for stage in captured)
    assert all(stage.nonlinear_residual_linf <= 2.0e-6 for stage in captured)
    assert all(stage.dissipation_power_W >= 0.0 for stage in captured)
    assert transaction.stage_evaluation_count == 2
    assert transaction.substep_count == 1
    assert transaction.dissipated_energy_J > 0.0
    assert transaction.liquid_mass_residual_kg == 0.0
    assert transaction.momentum_impulse_residual_N_s == 0.0
    expected_mirrored_volume = 0.5e-4 * math.fsum(
        stage.shared_mass_flux_m3_s for stage in captured
    )
    assert transaction.physical_signed_through_volume_m3 == pytest.approx(
        -expected_mirrored_volume,
        rel=0.0,
        abs=2.0e-22,
    )
    assert transaction.physical_left_momentum_impulse_N_s == pytest.approx(
        0.5e-4
        * extension.config.liquid_density
        * math.fsum(stage.right_momentum_flux_m4_s2 for stage in captured),
        rel=0.0,
        abs=2.0e-20,
    )
    assert transaction.physical_right_momentum_impulse_N_s == pytest.approx(
        0.5e-4
        * extension.config.liquid_density
        * math.fsum(stage.left_momentum_flux_m4_s2 for stage in captured),
        rel=0.0,
        abs=2.0e-20,
    )
    assert float(np.sum(result.state.area) * extension.dx) == pytest.approx(
        volume_before,
        rel=0.0,
        abs=4.0e-18,
    )
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_dry_face_root_is_unique_critical_passive_and_seed_free() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()

    stage = extension._solve_shock_cut_stage(
        state,
        dt=1.0e-4,
        stage_index=1,
        partition=partition,
        step_plan=plan,
    )
    depth = extension.section.depth_from_area(
        stage.interface_free_surface_area_m2
    )
    celerity = extension.section.free_surface_celerity_from_depth(depth)
    free_velocity = (
        stage.interface_free_surface_discharge_m3_s
        / stage.interface_free_surface_area_m2
    )

    assert stage.orientation is ShockCutValveOrientation.DRY_FACE_INJECTION
    assert stage.interface_speed_m_s == 0.0
    assert free_velocity == pytest.approx(-celerity, rel=0.0, abs=2.0e-14)
    assert stage.shared_mass_flux_m3_s == pytest.approx(
        stage.interface_free_surface_discharge_m3_s,
        rel=0.0,
        abs=2.0e-20,
    )
    assert stage.shared_mass_flux_m3_s == pytest.approx(
        extension.section.full_area
        * stage.interface_pressurised_velocity_m_s,
        rel=0.0,
        abs=2.0e-20,
    )
    assert stage.shared_mass_flux_m3_s < 0.0
    assert stage.signed_pressure_jump_Pa < 0.0
    assert stage.dissipation_power_W > 0.0
    assert stage.nonlinear_residual_linf <= 2.0e-6
    assert depth == pytest.approx(0.0030462526556064924, abs=2.0e-14)
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_free_compound_k0_uses_native_case1_root_not_vacuum_root() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = replace(extension.case_b_initial_state(), time=0.20)
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    feet = extension._shock_cut_face_contact_feet(state, dt=1.0e-4)
    native = extension._interface_solution(state, dt=1.0e-4)

    stage = extension._solve_shock_cut_stage(
        state,
        dt=1.0e-4,
        stage_index=1,
        partition=partition,
        step_plan=plan,
        orientation_override=ShockCutValveOrientation.FREE_SURFACE_TOUCH,
        characteristic_feet_override=feet,
        native_release_probe=True,
    )
    native_area = extension.section.area_from_depth(native.free_surface_depth)

    assert stage.orientation is ShockCutValveOrientation.FREE_SURFACE_TOUCH
    assert stage.interface_speed_m_s == pytest.approx(
        native.interface_speed,
        rel=0.0,
        abs=2.0e-12,
    )
    assert stage.interface_free_surface_area_m2 == pytest.approx(
        native_area,
        rel=0.0,
        abs=2.0e-13,
    )
    assert stage.shared_mass_flux_m3_s == pytest.approx(
        native_area * native.free_surface_velocity,
        rel=0.0,
        abs=2.0e-12,
    )
    assert stage.interface_speed_m_s > 0.0
    assert stage.nonlinear_residual_linf <= 2.0e-6


def test_two_wet_inward_limits_select_stationary_face_contact() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    core = Tosan2021HorizontalShockFit(campaign2_horizontal_config())
    evolved = core.step(core.case_b_initial_state(), 0.12)
    state = replace(
        evolved,
        interface_x=extension.local_face.mirrored_face_x_m,
    )
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)

    stage = extension._solve_shock_cut_stage(
        state,
        dt=1.0e-4,
        stage_index=1,
        partition=partition,
        step_plan=plan,
    )

    assert stage.orientation is ShockCutValveOrientation.FACE_CONTACT
    assert stage.interface_speed_m_s == 0.0
    assert stage.left_nonpenetration_residual_m_s > 0.0
    assert stage.right_nonpenetration_residual_m_s > 0.0
    assert stage.nonlinear_residual_linf <= 2.0e-6
    assert stage.dissipation_power_W > 0.0


def test_dry_face_injection_pins_interface_and_keeps_exact_global_liquid_volume(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    initial_volume = float(np.sum(state.area) * extension.dx)
    cuts = []
    wetting_fronts = []
    orientations = []
    original_solver = extension._solve_shock_cut_stage

    def capture(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        if kwargs.get("orientation_override") is None:
            orientations.append(solution.orientation)
        return solution

    monkeypatch.setattr(extension, "_solve_shock_cut_stage", capture)
    for _ in range(3):
        state = extension.advance_with_transaction(state, 1.0e-4).state
        cuts.append(int(math.floor(state.interface_x / extension.dx)))
        wetting_fronts.append(state.wetting_front_x)
        assert float(np.sum(state.area) * extension.dx) == pytest.approx(
            initial_volume,
            rel=0.0,
            abs=4.0e-18,
        )
    assert cuts == [61, 61, 61]
    assert state.interface_x == extension.local_face.mirrored_face_x_m
    assert all(
        orientation is ShockCutValveOrientation.DRY_FACE_INJECTION
        for orientation in orientations
    )
    assert wetting_fronts[0] < extension.local_face.mirrored_face_x_m
    assert wetting_fronts == sorted(wetting_fronts, reverse=True)


@pytest.mark.parametrize("cut", (59, 58, 55))
def test_pressurised_moc_first_cut59_and_smaller_cuts_advance_conservatively(
    cut: int,
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(extension, cut=cut)
    initial_volume = float(np.sum(state.area) * extension.dx)
    captured: list[PressurisedMocValveStageSolution] = []
    original_solver = extension._solve_pressurised_moc_stage

    def capture(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        if kwargs.get("orientation_override") is None:
            captured.append(solution)
        return solution

    monkeypatch.setattr(extension, "_solve_pressurised_moc_stage", capture)
    result = extension.advance_with_transaction(state, 1.0e-4)

    assert result.state.time == pytest.approx(state.time + 1.0e-4, abs=2.0e-16)
    assert len(captured) == 2
    assert [stage.stage_index for stage in captured] == [1, 2]
    assert all(stage.shock_cut_cell_index <= 59 for stage in captured)
    assert all(stage.elastic_volume_rate_residual_m3_s == 0.0 for stage in captured)
    assert all(stage.signed_pressure_jump_Pa * stage.shared_mass_flux_m3_s >= 0.0 for stage in captured)
    assert float(np.sum(result.state.area) * extension.dx) == pytest.approx(
        initial_volume,
        rel=0.0,
        abs=5.0e-18,
    )
    transaction = result.valve_transaction
    expected_mirrored_volume = 0.5e-4 * math.fsum(
        stage.shared_mass_flux_m3_s for stage in captured
    )
    assert transaction.mirrored_signed_through_volume_m3 == pytest.approx(
        expected_mirrored_volume,
        rel=0.0,
        abs=2.0e-22,
    )
    assert transaction.liquid_mass_residual_kg == 0.0
    assert transaction.momentum_impulse_residual_N_s == pytest.approx(
        0.0,
        abs=5.0e-19,
    )
    assert transaction.dissipated_energy_J >= 0.0


def test_pressurised_moc_reverse_flow_and_physical_mirror_signs() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    forward_state = pressurised_moc_restart_state(
        extension,
        cut=58,
        west_head_m=0.680,
        east_head_m=0.640,
    )
    reverse_state = pressurised_moc_restart_state(
        extension,
        cut=58,
        west_head_m=0.640,
        east_head_m=0.680,
    )
    forward = extension.advance_with_transaction(forward_state, 5.0e-5)
    reverse = extension.advance_with_transaction(reverse_state, 5.0e-5)

    assert forward.valve_transaction.mirrored_signed_through_volume_m3 > 0.0
    assert reverse.valve_transaction.mirrored_signed_through_volume_m3 < 0.0
    assert forward.valve_transaction.physical_signed_through_volume_m3 < 0.0
    assert reverse.valve_transaction.physical_signed_through_volume_m3 > 0.0
    assert forward.valve_transaction.dissipated_energy_J > 0.0
    assert reverse.valve_transaction.dissipated_energy_J > 0.0


def test_pressurised_moc_k_tends_to_zero_and_exact_k0_is_bitwise_native() -> None:
    config = campaign2_horizontal_config()
    baseline = Tosan2021HorizontalShockFit(config)
    extension = Case1LocalValveExtension(
        config,
        local_face=campaign2_valve_spec(),
    )
    near = pressurised_moc_restart_state(
        extension,
        cut=58,
        time_s=math.nextafter(0.20, 0.0),
    )
    partition = extension.classify_local_face_regime(near)
    plan = extension.plan_physical_step(near, 1.0e-5)
    stage = extension._solve_pressurised_moc_stage(
        near,
        dt=1.0e-5,
        stage_index=1,
        partition=partition,
        step_plan=plan,
    )
    assert abs(stage.shared_mass_flux_m3_s - stage.native_shared_mass_flux_m3_s) < 2.0e-16
    assert abs(stage.signed_pressure_jump_Pa) < 2.0e-8

    exact = replace(near, time=0.20)
    expected = baseline.step(exact, 1.0e-4)
    actual = extension.advance_with_transaction(exact, 1.0e-4)
    assert_horizontal_states_bitwise_equal(actual.state, expected)
    assert actual.valve_transaction.dissipated_energy_J == 0.0
    assert actual.valve_transaction.physical_wall_impulse_on_liquid_N_s == 0.0


def test_pressurised_moc_opening_event_splits_and_reconstructs_each_stage(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(
        extension,
        cut=58,
        time_s=0.19995,
    )
    calls: list[tuple[int, float]] = []
    original_solver = extension._solve_pressurised_moc_stage

    def capture(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        calls.append((solution.stage_index, solution.stage_time_s))
        return solution

    monkeypatch.setattr(extension, "_solve_pressurised_moc_stage", capture)
    result = extension.advance_with_transaction(state, 1.0e-4)

    assert result.state.time == pytest.approx(0.20005, abs=3.0e-16)
    assert [stage for stage, _time in calls] == [1, 2]
    assert [time for _stage, time in calls] == pytest.approx(
        (0.19995, 0.20),
        abs=3.0e-16,
    )
    assert result.valve_transaction.stage_evaluation_count == 4
    assert result.valve_transaction.substep_count == 2


def test_pressurised_moc_second_stage_no_root_rolls_back_atomically(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(extension, cut=58)
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    original_solver = extension._solve_pressurised_moc_stage
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PressurisedMocValveStageRejected(
                partition=partition,
                step_plan=plan,
                stage_index=2,
                stage_time_s=state.time + 1.0e-4,
                reason="injected no-root",
            )
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(extension, "_solve_pressurised_moc_stage", fail_second)
    with pytest.raises(PressurisedMocValveStageRejected, match="injected no-root"):
        extension._step_once_pressurised_moc(
            state,
            1.0e-4,
            None,
            partition=partition,
            step_plan=plan,
        )
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
    assert state.time == 0.10


def test_pressurised_moc_real_characteristic_no_root_is_atomic() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(extension, cut=58)
    area = state.area.copy()
    discharge = state.discharge.copy()
    face = extension.local_face.mirrored_face_index
    discharge[state.area > 0.0] = state.area[state.area > 0.0] * 35.0
    discharge[face:] = 0.0
    state = replace(state, area=area, discharge=discharge)
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-5)

    with pytest.raises(PressurisedMocValveStageRejected, match="entropy wave"):
        extension._solve_pressurised_moc_stage(
            state,
            dt=1.0e-5,
            stage_index=1,
            partition=partition,
            step_plan=plan,
        )
    assert np.array_equal(state.area, area)
    assert np.array_equal(state.discharge, discharge)


def test_pressurised_moc_retry_subdivision_commits_only_accepted_halves(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(extension, cut=58)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    original_solver = extension._solve_pressurised_moc_stage
    rejected = 0

    def require_half_step(*args, **kwargs):
        nonlocal rejected
        if kwargs["dt"] > 5.0e-5:
            rejected += 1
            raise PressurisedMocValveStageRejected(
                partition=kwargs["partition"],
                step_plan=kwargs["step_plan"],
                stage_index=kwargs["stage_index"],
                stage_time_s=float(args[0].time),
                reason="injected full-step storage rejection",
            )
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(
        extension,
        "_solve_pressurised_moc_stage",
        require_half_step,
    )
    result = extension.advance_with_transaction(state, 1.0e-4)

    assert rejected == 1
    assert result.state.time == pytest.approx(0.1001, abs=2.0e-16)
    assert result.valve_transaction.stage_evaluation_count == 4
    assert result.valve_transaction.substep_count == 2
    assert result.valve_transaction.start_time_s == state.time
    assert result.valve_transaction.end_time_s == result.state.time
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_pressurised_moc_restart_is_bitwise_reproducible() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = pressurised_moc_restart_state(extension, cut=55)
    first = extension.advance_with_transaction(state, 1.0e-4)
    second = extension.advance_with_transaction(state, 1.0e-4)

    assert_horizontal_states_bitwise_equal(first.state, second.state)
    assert first.valve_transaction == second.valve_transaction


def test_shock_cut_donor_rejection_is_atomic(monkeypatch) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    monkeypatch.setattr(extension, "_right_boundary_donor_scale", lambda *a, **k: 0.5)

    with pytest.raises(ShockCutValveDonorLimitRejected):
        extension._step_once_shock_cut(
            state,
            1.0e-4,
            None,
            partition=partition,
            step_plan=plan,
        )

    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_shock_cut_reconstructed_boundary_cfl_rejection_is_atomic(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    solution = extension._solve_shock_cut_stage(
        state,
        dt=1.0e-4,
        stage_index=1,
        partition=partition,
        step_plan=plan,
    )
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()

    def reject_cfl(*args, **kwargs):
        raise ValueError("dt exceeds the requested central-upwind CFL limit")

    monkeypatch.setattr(
        extension_module._case1_core,
        "_central_upwind_wet_dry_euler_step",
        reject_cfl,
    )
    with pytest.raises(ShockCutValveCflRejected, match="boundary CFL"):
        extension._shock_cut_euler_stage(
            state,
            dt=1.0e-4,
            solution=solution,
            partition=partition,
            step_plan=plan,
            external_pressure_abs=None,
        )
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
    assert state.time == 0.0


def test_shock_cut_second_stage_failure_rolls_back_input_atomically(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = extension.case_b_initial_state()
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    original_solver = extension._solve_shock_cut_stage

    def fail_second(*args, **kwargs):
        if kwargs["stage_index"] == 2:
            raise ShockCutValveNonlinearRejected(
                partition=partition,
                step_plan=plan,
                stage_index=2,
                stage_time_s=1.0e-4,
                reason="injected stage-two failure",
            )
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(extension, "_solve_shock_cut_stage", fail_second)
    with pytest.raises(ShockCutValveNonlinearRejected):
        extension._step_once_shock_cut(
            state,
            1.0e-4,
            None,
            partition=partition,
            step_plan=plan,
        )
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
    assert state.time == 0.0


def test_artificial_near_full_wet_dry_start_rejects_without_a_liquid_film() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    original = extension.case_b_initial_state()
    state = replace(original, interface_x=0.630)
    partition = extension.classify_local_face_regime(state)
    assert partition.regime is LocalValveRegime.CLEAN_FREE_SURFACE_FV
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    assert state.area[60] == 0.0
    assert state.area[61] > 0.0

    with pytest.raises(
        CleanFreeSurfaceValveTraceRejected,
        match="reaches full area",
    ):
        extension.advance_with_transaction(state, 1.0e-5)

    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
    assert state.time == original.time


def test_resolvable_clean_fv_face_advances_with_passive_transaction() -> None:
    config = campaign2_horizontal_config()
    extension = Case1LocalValveExtension(
        config,
        local_face=campaign2_valve_spec(),
    )
    baseline = Tosan2021HorizontalShockFit(config)
    state = resolvable_clean_fv_state(0.120)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    partition = extension.classify_local_face_regime(state)
    assert partition.regime is LocalValveRegime.CLEAN_FREE_SURFACE_FV

    result = extension.advance_with_transaction(state, 1.0e-4)
    unvalved = baseline.step(state, 1.0e-4)
    transaction = result.valve_transaction

    assert result.state.time == state.time + 1.0e-4
    assert transaction.stage_evaluation_count == 2
    assert transaction.substep_count == 1
    assert transaction.physical_signed_through_volume_m3 > 0.0
    assert transaction.dissipated_energy_J > 0.0
    assert transaction.liquid_mass_residual_kg == 0.0
    assert transaction.momentum_impulse_residual_N_s == 0.0
    assert not np.array_equal(result.state.discharge, unvalved.discharge)
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_clean_fv_k0_whole_horizontal_step_is_bitwise_native() -> None:
    config = campaign2_horizontal_config()
    baseline = Tosan2021HorizontalShockFit(config)
    extension = Case1LocalValveExtension(
        config,
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.210)
    dt = 1.0e-4

    expected = baseline.step(state, dt)
    result = extension.advance_with_transaction(state, dt)

    assert_horizontal_states_bitwise_equal(result.state, expected)
    assert result.valve_transaction.stage_evaluation_count == 2
    assert result.valve_transaction.substep_count == 1
    assert result.valve_transaction.dissipated_energy_J == 0.0
    assert result.valve_transaction.physical_wall_impulse_on_liquid_N_s == 0.0


def test_clean_fv_crossing_opening_event_splits_exactly_and_recomputes_stages(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.19995)
    original_solver = extension_module.solve_passive_circular_saint_venant_valve
    stage_solutions = []

    def capture_solution(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        stage_solutions.append(solution)
        return solution

    monkeypatch.setattr(
        extension_module,
        "solve_passive_circular_saint_venant_valve",
        capture_solution,
    )
    result = extension.advance_with_transaction(state, 1.0e-4)

    assert result.state.time == state.time + 1.0e-4
    assert result.valve_transaction.stage_evaluation_count == 4
    assert result.valve_transaction.substep_count == 2
    assert [solution.opening.time_s for solution in stage_solutions] == pytest.approx(
        (0.19995, 0.20000, 0.20000, 0.20005),
        abs=3.0e-16,
    )
    assert stage_solutions[0].opening.loss_coefficient > 0.0
    assert all(
        solution.opening.loss_coefficient == 0.0
        for solution in stage_solutions[1:]
    )


def test_clean_fv_transaction_uses_ssprk2_half_weights_and_physical_mirror(
    monkeypatch,
) -> None:
    config = campaign2_horizontal_config()
    extension = Case1LocalValveExtension(
        config,
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.120)
    dt = 1.0e-4
    original_solver = extension_module.solve_passive_circular_saint_venant_valve
    stage_solutions = []

    def capture_solution(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        stage_solutions.append(solution)
        return solution

    monkeypatch.setattr(
        extension_module,
        "solve_passive_circular_saint_venant_valve",
        capture_solution,
    )
    transaction = extension.advance_with_transaction(
        state,
        dt,
    ).valve_transaction

    assert len(stage_solutions) == 2
    expected_physical_volume = -0.5 * dt * math.fsum(
        solution.volume_flow_left_to_right_m3_s
        for solution in stage_solutions
    )
    expected_physical_left_impulse = (
        0.5
        * dt
        * config.liquid_density
        * math.fsum(
            solution.right_specific_momentum_flux_m4_s2
            for solution in stage_solutions
        )
    )
    expected_physical_right_impulse = (
        0.5
        * dt
        * config.liquid_density
        * math.fsum(
            solution.left_specific_momentum_flux_m4_s2
            for solution in stage_solutions
        )
    )
    expected_energy = 0.5 * dt * math.fsum(
        solution.dissipation_power_W
        for solution in stage_solutions
    )
    assert transaction.physical_signed_through_volume_m3 == pytest.approx(
        expected_physical_volume,
        rel=0.0,
        abs=2.0e-22,
    )
    assert transaction.physical_left_momentum_impulse_N_s == pytest.approx(
        expected_physical_left_impulse,
        rel=0.0,
        abs=2.0e-20,
    )
    assert transaction.physical_right_momentum_impulse_N_s == pytest.approx(
        expected_physical_right_impulse,
        rel=0.0,
        abs=2.0e-20,
    )
    assert transaction.dissipated_energy_J == pytest.approx(
        expected_energy,
        rel=0.0,
        abs=2.0e-22,
    )


def test_actual_unscaled_t120_supercritical_outflow_uses_one_sided_characteristic(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.120, discharge_scale=1.0)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    captured = []
    original_solver = extension_module.solve_passive_circular_saint_venant_valve

    def capture_solution(left, right, **kwargs):
        solution = original_solver(left, right, **kwargs)
        captured.append((left, right, solution))
        return solution

    monkeypatch.setattr(
        extension_module,
        "solve_passive_circular_saint_venant_valve",
        capture_solution,
    )

    result = extension.advance_with_transaction(state, 1.0e-4)

    west, east, first_solution = captured[0]
    assert west.froude_number == pytest.approx(-1.0752750794595414)
    assert east.froude_number == pytest.approx(-0.9975255684112398)
    assert first_solution.control_regime is (
        FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
    )
    assert first_solution.upstream_side == "right"
    assert not first_solution.left_characteristic_active
    assert first_solution.right_characteristic_active
    assert first_solution.native_offset_nonlinear_characteristic
    assert first_solution.upwind_wetted_area_m2 == pytest.approx(
        0.0012225966465261749,
        rel=2.0e-12,
    )
    assert first_solution.volume_flow_left_to_right_m3_s == pytest.approx(
        -0.0003630218571911908,
        rel=2.0e-12,
    )
    assert first_solution.volume_flow_left_to_right_m3_s < 0.0
    assert first_solution.signed_pressure_jump_Pa < 0.0
    assert first_solution.dissipation_power_W > 0.0
    assert result.state.time == state.time + 1.0e-4
    assert result.valve_transaction.stage_evaluation_count == 2
    assert result.valve_transaction.dissipated_energy_J > 0.0

    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_clean_fv_callback_maps_case1_dry_trace_without_a_film() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.120)
    partition = extension.classify_local_face_regime(state)
    plan = extension.plan_physical_step(state, 1.0e-4)
    stage_solutions = []
    callback = extension._clean_free_surface_callback(
        partition=partition,
        step_plan=plan,
        stage_solutions=stage_solutions,
    )
    context = InternalFaceStageContext(
        stage_index=1,
        stage_time_s=0.120,
        dt_s=1.0e-4,
        face_index=61,
        west_area_m2=0.0,
        west_discharge_m3_s=0.0,
        east_area_m2=8.0e-4,
        east_discharge_m3_s=-2.0e-4,
        native_shared_mass_flux_m3_s=-2.2e-4,
        native_momentum_flux_m4_s2=0.003,
    )

    flux = callback(context)

    assert flux.shared_mass_flux_m3_s < 0.0
    assert len(stage_solutions) == 1
    solution = stage_solutions[0]
    assert solution.control_regime is (
        FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
    )
    assert solution.upstream_side == "right"
    assert not solution.left_characteristic_active
    assert solution.right_characteristic_active
    assert solution.dissipation_power_W > 0.0


def test_clean_fv_rejects_a_donor_scaled_physical_solution_atomically(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = resolvable_clean_fv_state(0.120)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    original_limiter = extension_module._split_momentum_donor_draining_limiter

    def force_valve_scale(*args, **kwargs):
        mass, momentum_east, momentum_west, scales = original_limiter(
            *args,
            **kwargs,
        )
        face = 61
        mass[face] *= 0.5
        momentum_east[face] *= 0.5
        momentum_west[face] *= 0.5
        scales[face] *= 0.5
        return mass, momentum_east, momentum_west, scales

    monkeypatch.setattr(
        extension_module,
        "_split_momentum_donor_draining_limiter",
        force_valve_scale,
    )
    with pytest.raises(CleanFreeSurfaceValveDonorLimitRejected):
        extension.advance_with_transaction(state, 1.0e-4)

    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_opening_event_at_point_two_seconds_is_an_exact_step_boundary() -> None:
    plan = plan_event_aligned_step(
        start_time_s=0.199,
        dt_s=0.003,
        event_times_s=(0.200,),
    )
    assert plan.start_time_s == 0.199
    assert plan.end_time_s == pytest.approx(0.202, abs=2.0e-16)
    assert plan.interior_events_s == (0.200,)
    assert plan.substeps_s == pytest.approx((0.001, 0.002), abs=2.0e-16)
    assert math.fsum(plan.substeps_s) == pytest.approx(0.003, abs=2.0e-16)

    ending_at_event = plan_event_aligned_step(
        start_time_s=0.198,
        dt_s=0.002,
        event_times_s=(0.200,),
    )
    starting_at_event = plan_event_aligned_step(
        start_time_s=0.200,
        dt_s=0.002,
        event_times_s=(0.200,),
    )
    assert ending_at_event.substeps_s == pytest.approx((0.002,))
    assert starting_at_event.substeps_s == pytest.approx((0.002,))


def test_active_shock_cut_plan_carries_the_exact_opening_event() -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = replace(extension.case_b_initial_state(), time=0.199)
    plan = extension.plan_physical_step(state, 0.003)
    assert plan.interior_events_s == (0.200,)
    assert plan.substeps_s == pytest.approx(
        (0.001, 0.002),
        abs=2.0e-16,
    )


def test_shock_cut_step_ending_at_opening_event_has_exact_k0_second_stage(
    monkeypatch,
) -> None:
    extension = Case1LocalValveExtension(
        campaign2_horizontal_config(),
        local_face=campaign2_valve_spec(),
    )
    state = replace(extension.case_b_initial_state(), time=0.19999)
    captured = []
    original_solver = extension._solve_shock_cut_stage

    def capture(*args, **kwargs):
        solution = original_solver(*args, **kwargs)
        captured.append(solution)
        return solution

    monkeypatch.setattr(extension, "_solve_shock_cut_stage", capture)
    result = extension.advance_with_transaction(state, 1.0e-5)

    assert result.state.time == 0.200
    assert len(captured) == 2
    assert captured[0].stage_time_s == 0.19999
    assert captured[0].dissipation_power_W > 0.0
    assert captured[1].stage_time_s == 0.200
    assert captured[1].dissipation_power_W == 0.0
    assert captured[1].signed_pressure_jump_Pa == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


@pytest.mark.parametrize("interface_x", (0.605, 0.610, 0.615))
def test_shock_cut_exact_k0_whole_step_is_bitwise_native(
    interface_x: float,
) -> None:
    config = campaign2_horizontal_config()
    baseline = Tosan2021HorizontalShockFit(config)
    extension = Case1LocalValveExtension(
        config,
        local_face=campaign2_valve_spec(),
    )
    state = replace(
        extension.case_b_initial_state(),
        time=0.200,
        interface_x=interface_x,
    )
    dt = 1.0e-4

    expected = baseline.step(state, dt)
    result = extension.advance_with_transaction(state, dt)

    assert_horizontal_states_bitwise_equal(result.state, expected)
    assert result.valve_transaction.stage_evaluation_count == 2
    assert result.valve_transaction.substep_count == 1
    assert result.valve_transaction.dissipated_energy_J == 0.0
    assert result.valve_transaction.physical_wall_impulse_on_liquid_N_s == 0.0


def test_transaction_zero_merge_mass_sign_and_momentum_invariants() -> None:
    zero = IntegratedValveTransaction.zero(
        start_time_s=0.0,
        end_time_s=0.05,
    )
    assert zero.duration_s == 0.05
    assert zero.mirrored_signed_through_volume_m3 == 0.0
    assert zero.liquid_mass_residual_kg == 0.0
    assert zero.momentum_impulse_residual_N_s == 0.0

    first = IntegratedValveTransaction(
        start_time_s=0.05,
        end_time_s=0.10,
        physical_signed_through_volume_m3=1.0e-4,
        physical_left_momentum_impulse_N_s=10.0,
        physical_right_momentum_impulse_N_s=7.0,
        physical_wall_impulse_on_liquid_N_s=-3.0,
        dissipated_energy_J=2.0,
        stage_evaluation_count=2,
        substep_count=1,
    )
    second = IntegratedValveTransaction(
        start_time_s=0.10,
        end_time_s=0.20,
        physical_signed_through_volume_m3=-2.0e-5,
        physical_left_momentum_impulse_N_s=4.0,
        physical_right_momentum_impulse_N_s=5.0,
        physical_wall_impulse_on_liquid_N_s=1.0,
        dissipated_energy_J=0.5,
        stage_evaluation_count=4,
        substep_count=2,
    )
    merged = zero.merged(first).merged(second)

    assert merged.start_time_s == 0.0
    assert merged.end_time_s == 0.20
    assert merged.physical_signed_through_volume_m3 == pytest.approx(8.0e-5)
    assert merged.mirrored_signed_through_volume_m3 == pytest.approx(-8.0e-5)
    assert merged.physical_signed_through_mass_kg == pytest.approx(998.0 * 8.0e-5)
    assert merged.physical_left_liquid_mass_change_kg == pytest.approx(
        -merged.physical_right_liquid_mass_change_kg
    )
    assert merged.liquid_mass_residual_kg == 0.0
    assert merged.physical_left_momentum_impulse_N_s == 14.0
    assert merged.physical_right_momentum_impulse_N_s == 12.0
    assert merged.physical_wall_impulse_on_liquid_N_s == -2.0
    assert merged.mirrored_wall_impulse_on_liquid_N_s == 2.0
    assert merged.momentum_impulse_residual_N_s == 0.0
    assert merged.dissipated_energy_J == 2.5
    assert merged.stage_evaluation_count == 6
    assert merged.substep_count == 3

    with pytest.raises(FrozenInstanceError):
        merged.end_time_s = 1.0  # type: ignore[misc]


def test_invalid_or_noncontiguous_transactions_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative energy"):
        IntegratedValveTransaction(
            start_time_s=0.0,
            end_time_s=0.1,
            physical_signed_through_volume_m3=0.0,
            physical_left_momentum_impulse_N_s=0.0,
            physical_right_momentum_impulse_N_s=0.0,
            physical_wall_impulse_on_liquid_N_s=0.0,
            dissipated_energy_J=-1.0,
        )
    with pytest.raises(ValueError, match="do not close"):
        IntegratedValveTransaction(
            start_time_s=0.0,
            end_time_s=0.1,
            physical_signed_through_volume_m3=0.0,
            physical_left_momentum_impulse_N_s=1.0,
            physical_right_momentum_impulse_N_s=1.0,
            physical_wall_impulse_on_liquid_N_s=1.0,
            dissipated_energy_J=0.0,
        )

    first = IntegratedValveTransaction.zero(start_time_s=0.0, end_time_s=0.1)
    gap = IntegratedValveTransaction.zero(start_time_s=0.2, end_time_s=0.3)
    with pytest.raises(ValueError, match="time-contiguous"):
        first.merged(gap)


def test_generic_fv_none_callback_is_direct_bitwise_case1_passthrough() -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    state = fv_test_state(section)
    keywords = dict(
        dx=0.010,
        dt=4.0e-5,
        section=section,
        cfl=0.45,
        dry_area_fraction=1.0e-10,
        manning_n=0.009,
        darcy_friction=0.018,
        bed_slope=0.002,
        left_boundary="wall",
        right_boundary="transmissive",
        right_ghost=(0.8 * section.full_area, -0.5e-5),
        left_face_flux=None,
        right_face_flux=(0.2e-5, 0.003),
        interface_traction=(0.060, 0.04, "left"),
    )
    expected = central_upwind_wet_dry_step(state, **keywords)
    result = internal_face_wet_dry_ssprk2_step(
        state,
        face_index=3,
        callback=None,
        start_time_s=0.137,
        **keywords,
    )

    assert_wet_dry_states_bitwise_equal(result.state, expected)
    assert result.stage_records == ()
    assert result.used_native_core_step


def test_generic_fv_explicit_native_k0_recomputes_both_stages_bitwise() -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    state = fv_test_state(section)
    contexts = []

    def transparent_k0(context):
        contexts.append(context)
        return InternalFaceFluxPair.native(context)

    keywords = dict(
        dx=0.010,
        dt=4.0e-5,
        section=section,
        cfl=0.45,
        dry_area_fraction=1.0e-10,
        manning_n=0.009,
        darcy_friction=0.018,
        bed_slope=-0.001,
        left_boundary="wall",
        right_boundary="wall",
        interface_traction=(0.050, -0.03, "right"),
    )
    expected = central_upwind_wet_dry_step(state, **keywords)
    result = internal_face_wet_dry_ssprk2_step(
        state,
        face_index=3,
        callback=transparent_k0,
        start_time_s=0.19996,
        **keywords,
    )

    assert_wet_dry_states_bitwise_equal(result.state, expected)
    assert result.used_native_core_step
    assert len(result.stage_records) == 2
    assert [context.stage_index for context in contexts] == [1, 2]
    assert [context.stage_time_s for context in contexts] == pytest.approx(
        (0.19996, 0.20000),
        abs=2.0e-16,
    )
    assert all(record.used_native_core_stage for record in result.stage_records)
    assert all(
        record.accepted_flux.west_momentum_flux_m4_s2
        == record.accepted_flux.east_momentum_flux_m4_s2
        for record in result.stage_records
    )
    # The callback sees the provisional state, not a cached stage-1 trace.
    assert (
        contexts[1].west_area_m2 != contexts[0].west_area_m2
        or contexts[1].west_discharge_m3_s
        != contexts[0].west_discharge_m3_s
        or contexts[1].east_area_m2 != contexts[0].east_area_m2
        or contexts[1].east_discharge_m3_s
        != contexts[0].east_discharge_m3_s
    )


def test_generic_fv_none_return_is_also_native_and_atomic() -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    state = fv_test_state(section)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()
    calls = []

    def no_override(context):
        calls.append((context.stage_index, context.stage_time_s))
        return None

    expected = central_upwind_wet_dry_step(
        state,
        dx=0.010,
        dt=3.0e-5,
        section=section,
    )
    result = internal_face_wet_dry_ssprk2_step(
        state,
        dx=0.010,
        dt=3.0e-5,
        section=section,
        face_index=4,
        callback=no_override,
        start_time_s=1.25,
    )

    assert_wet_dry_states_bitwise_equal(result.state, expected)
    assert [stage for stage, _time in calls] == [1, 2]
    assert [time for _stage, time in calls] == pytest.approx(
        (1.25, 1.25003),
        abs=2.0e-16,
    )
    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)


def test_split_momentum_ports_supply_wall_impulse_without_losing_volume() -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    state = WetDryState(
        area=np.full(10, 0.55 * section.full_area),
        discharge=np.zeros(10),
    )
    delta_west = 0.020
    delta_east = -0.010

    def split_ports(context):
        return InternalFaceFluxPair(
            shared_mass_flux_m3_s=context.native_shared_mass_flux_m3_s,
            west_momentum_flux_m4_s2=(
                context.native_momentum_flux_m4_s2 + delta_west
            ),
            east_momentum_flux_m4_s2=(
                context.native_momentum_flux_m4_s2 + delta_east
            ),
        )

    dt = 1.0e-5
    dx = 0.010
    result = internal_face_wet_dry_ssprk2_step(
        state,
        dx=dx,
        dt=dt,
        section=section,
        face_index=5,
        callback=split_ports,
        start_time_s=0.0,
        left_boundary="wall",
        right_boundary="wall",
    )

    assert not result.used_native_core_step
    assert len(result.stage_records) == 2
    assert np.sum(result.state.area) == pytest.approx(
        np.sum(state.area),
        rel=0.0,
        abs=5.0e-18,
    )
    expected_sum_discharge_change = 0.5 * dt / dx * math.fsum(
        record.accepted_flux.east_momentum_flux_m4_s2
        - record.accepted_flux.west_momentum_flux_m4_s2
        for record in result.stage_records
    )
    assert float(np.sum(result.state.discharge - state.discharge)) == pytest.approx(
        expected_sum_discharge_change,
        rel=0.0,
        abs=2.0e-15,
    )
    assert all(
        record.accepted_flux.west_momentum_flux_m4_s2
        != record.accepted_flux.east_momentum_flux_m4_s2
        for record in result.stage_records
    )


@pytest.mark.parametrize("flow_sign", (1.0, -1.0))
def test_shared_flux_donor_limiter_scales_both_ports_and_preserves_positivity(
    flow_sign: float,
) -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    tiny = 1.0e-8 * section.full_area
    if flow_sign > 0.0:
        area = np.array((0.4 * section.full_area, tiny, 0.0, 0.0, 0.0))
    else:
        area = np.array((0.0, 0.0, tiny, 0.4 * section.full_area, 0.0))
    state = WetDryState(area=area, discharge=np.zeros_like(area))
    requested_mass = flow_sign * 1.0
    requested_west_momentum = flow_sign * 0.002
    requested_east_momentum = flow_sign * 0.003

    def overdraw(_context):
        return InternalFaceFluxPair(
            shared_mass_flux_m3_s=requested_mass,
            west_momentum_flux_m4_s2=requested_west_momentum,
            east_momentum_flux_m4_s2=requested_east_momentum,
        )

    result = internal_face_wet_dry_ssprk2_step(
        state,
        dx=0.010,
        dt=1.0e-5,
        section=section,
        face_index=2,
        callback=overdraw,
        start_time_s=0.05,
        left_boundary="wall",
        right_boundary="wall",
    )

    first = result.stage_records[0]
    assert 0.0 < first.donor_scale < 1.0
    assert first.accepted_flux.shared_mass_flux_m3_s == pytest.approx(
        requested_mass * first.donor_scale
    )
    assert first.accepted_flux.west_momentum_flux_m4_s2 == pytest.approx(
        requested_west_momentum * first.donor_scale
    )
    assert first.accepted_flux.east_momentum_flux_m4_s2 == pytest.approx(
        requested_east_momentum * first.donor_scale
    )
    assert np.all(result.state.area >= 0.0)
    assert np.sum(result.state.area) == pytest.approx(
        np.sum(state.area),
        rel=0.0,
        abs=5.0e-18,
    )


def test_generic_fv_rejects_invalid_face_or_callback_without_mutation() -> None:
    section = CircularSection(0.050, gravity=9.81, wave_speed=28.0)
    state = fv_test_state(section)
    area_before = state.area.copy()
    discharge_before = state.discharge.copy()

    with pytest.raises(ValueError, match="one physical cell on each side"):
        internal_face_wet_dry_ssprk2_step(
            state,
            dx=0.010,
            dt=2.0e-5,
            section=section,
            face_index=0,
            callback=lambda context: None,
            start_time_s=0.0,
        )

    def invalid_callback(_context):
        return (1.0, 2.0, 3.0)

    with pytest.raises(TypeError, match="must return InternalFaceFluxPair"):
        internal_face_wet_dry_ssprk2_step(
            state,
            dx=0.010,
            dt=2.0e-5,
            section=section,
            face_index=3,
            callback=invalid_callback,
            start_time_s=0.0,
        )

    assert np.array_equal(state.area, area_before)
    assert np.array_equal(state.discharge, discharge_before)
