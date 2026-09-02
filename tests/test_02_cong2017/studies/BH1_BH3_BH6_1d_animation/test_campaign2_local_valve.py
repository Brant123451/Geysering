from __future__ import annotations

import inspect
import math

import pytest

from campaign2_local_valve import (
    CHARACTERISTIC_MODEL_NAME,
    CircularSaintVenantValveTrace,
    FREE_SURFACE_CHARACTERISTIC_MODEL_NAME,
    FreeSurfaceValveControlRegime,
    LocalValveLedger,
    LiquidValveTrace,
    MINIMUM_AREA_FRACTION,
    OPENING_DURATION_S,
    PressurisedMocValveControlRegime,
    PressurisedMocValveNoRootError,
    RESISTANCE_LENGTH_M,
    VALVE_MODEL_NAME,
    WATER_DENSITY_KG_M3,
    provenance,
    shared_opening_state,
    solve_passive_circular_saint_venant_valve,
    solve_passive_liquid_valve,
    solve_passive_pressurised_moc_valve,
)


PIPE_AREA_M2 = 0.25 * math.pi * 0.050**2
WAVE_SPEED_M_S = 28.0


def trace(*, pressure: float, velocity: float = 0.0) -> LiquidValveTrace:
    return LiquidValveTrace(
        area_m2=PIPE_AREA_M2,
        velocity_m_s=velocity,
        gauge_pressure_Pa=pressure,
        wave_speed_m_s=WAVE_SPEED_M_S,
    )


def test_frozen_shared_contract_and_documented_provenance() -> None:
    assert OPENING_DURATION_S == 0.20
    assert MINIMUM_AREA_FRACTION == 0.001
    assert RESISTANCE_LENGTH_M == 0.025
    assert WATER_DENSITY_KG_M3 == 998.0
    assert VALVE_MODEL_NAME == "sineSquaredAreaForchheimer"
    assert "water-hammer" in CHARACTERISTIC_MODEL_NAME
    assert "circular Saint-Venant" in FREE_SURFACE_CHARACTERISTIC_MODEL_NAME

    record = provenance()
    assert record["opening_duration_s"] == 0.20
    assert record["minimum_area_fraction"] == 0.001
    assert record["default_water_density_kg_m3"] == 998.0
    assert "approximately 0.2 s" in record["paper_support"]
    assert "OpenFOAM" in record["numerical_contract_support"]
    assert "shock/cut" in record["integration_status"]
    assert "local upstream wetted area" in record[
        "clean_free_surface_velocity_area"
    ]
    assert record["openfoam_reference_flow_area_role"] == (
        "resistance-zone length audit only"
    )
    assert "Riemann-supply choking" in record["clean_free_surface_control"]


@pytest.mark.parametrize(
    ("time_s", "area_fraction", "loss_coefficient"),
    (
        (0.00, 0.001, 999_999.0),
        (0.05, 0.14644660940672624, 45.62741699796952),
        (0.10, 0.5, 3.0),
        (0.15, 0.8535533905932735, 0.3725830020304801),
        (0.20, 1.0, 0.0),
        (0.25, 1.0, 0.0),
    ),
)
def test_shared_sine_squared_opening_law(
    time_s: float,
    area_fraction: float,
    loss_coefficient: float,
) -> None:
    state = shared_opening_state(time_s)
    assert state.area_fraction == pytest.approx(area_fraction, rel=2.0e-15)
    assert state.loss_coefficient == pytest.approx(
        loss_coefficient,
        rel=2.0e-15,
        abs=0.0,
    )


def test_schedule_clips_before_zero_and_is_exactly_lossless_after_opening() -> None:
    assert shared_opening_state(-1.0).area_fraction == 0.001
    assert shared_opening_state(-1.0).loss_coefficient == 999_999.0
    assert shared_opening_state(OPENING_DURATION_S).loss_coefficient == 0.0
    assert shared_opening_state(10.0).loss_coefficient == 0.0


def test_lossless_face_exactly_recovers_common_pressure_and_momentum_flux() -> None:
    left = trace(pressure=4_000.0, velocity=0.3)
    right = trace(pressure=4_000.0, velocity=0.3)
    solution = solve_passive_liquid_valve(
        left,
        right,
        time_s=0.20,
        valve_flow_area_m2=PIPE_AREA_M2,
    )

    assert solution.opening.loss_coefficient == 0.0
    assert solution.volume_flow_left_to_right_m3_s == pytest.approx(
        PIPE_AREA_M2 * 0.3,
        rel=2.0e-15,
    )
    assert solution.left_gauge_pressure_Pa == solution.right_gauge_pressure_Pa
    assert solution.signed_pressure_jump_Pa == 0.0
    assert solution.left_momentum_flow_N == solution.right_momentum_flow_N
    assert solution.valve_wall_force_on_liquid_N == 0.0
    assert solution.dissipation_power_W == 0.0
    assert solution.momentum_force_residual_N == 0.0
    assert solution.continuity_residual_m3_s == 0.0


def test_forward_pressure_drive_is_passive_and_closes_both_characteristics() -> None:
    solution = solve_passive_liquid_valve(
        trace(pressure=6_000.0),
        trace(pressure=0.0),
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )

    assert solution.volume_flow_left_to_right_m3_s > 0.0
    assert solution.valve_velocity_m_s > 0.0
    assert solution.signed_pressure_jump_Pa > 0.0
    assert solution.valve_wall_force_on_liquid_N < 0.0
    assert solution.dissipation_power_W > 0.0
    assert solution.upwind_density_kg_m3 == 998.0
    assert abs(solution.left_characteristic_residual_m_s) < 2.0e-15
    assert abs(solution.right_characteristic_residual_m_s) < 2.0e-15
    assert abs(solution.pressure_jump_residual_Pa) < 2.0e-12
    assert abs(solution.momentum_force_residual_N) < 2.0e-15


def test_reverse_pressure_drive_is_odd_and_remains_passive() -> None:
    forward = solve_passive_liquid_valve(
        trace(pressure=6_000.0),
        trace(pressure=0.0),
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    reverse = solve_passive_liquid_valve(
        trace(pressure=0.0),
        trace(pressure=6_000.0),
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )

    assert reverse.volume_flow_left_to_right_m3_s == pytest.approx(
        -forward.volume_flow_left_to_right_m3_s,
        rel=2.0e-15,
    )
    assert reverse.signed_pressure_jump_Pa == pytest.approx(
        -forward.signed_pressure_jump_Pa,
        rel=2.0e-15,
    )
    assert reverse.valve_wall_force_on_liquid_N == pytest.approx(
        -forward.valve_wall_force_on_liquid_N,
        rel=2.0e-15,
    )
    assert reverse.dissipation_power_W == pytest.approx(
        forward.dissipation_power_W,
        rel=2.0e-15,
    )
    assert reverse.valve_wall_force_on_liquid_N > 0.0
    assert reverse.dissipation_power_W > 0.0


def test_pressurised_moc_subcritical_closure_reuses_the_two_port_exactly() -> None:
    left = trace(pressure=6_000.0, velocity=0.15)
    right = trace(pressure=500.0, velocity=-0.05)
    expected = solve_passive_liquid_valve(
        left,
        right,
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    actual = solve_passive_pressurised_moc_valve(
        left,
        right,
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
        nominal_pipe_area_m2=PIPE_AREA_M2,
    )

    assert (
        actual.control_regime
        is PressurisedMocValveControlRegime.TWO_SIDED_SUBCRITICAL
    )
    assert actual.left_incoming_characteristic_count == 1
    assert actual.right_incoming_characteristic_count == 1
    for name in (
        "volume_flow_left_to_right_m3_s",
        "left_gauge_pressure_Pa",
        "right_gauge_pressure_Pa",
        "signed_pressure_jump_Pa",
        "left_momentum_flow_N",
        "right_momentum_flow_N",
        "valve_wall_force_on_liquid_N",
        "dissipation_power_W",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.momentum_force_residual_N == pytest.approx(
        0.0,
        abs=3.0e-16,
    )


def test_pressurised_moc_strong_resistance_and_reverse_flow_remain_passive() -> None:
    forward = solve_passive_pressurised_moc_valve(
        trace(pressure=8_000.0),
        trace(pressure=0.0),
        time_s=0.0,
        valve_flow_area_m2=PIPE_AREA_M2,
        nominal_pipe_area_m2=PIPE_AREA_M2,
    )
    reverse = solve_passive_pressurised_moc_valve(
        trace(pressure=0.0),
        trace(pressure=8_000.0),
        time_s=0.0,
        valve_flow_area_m2=PIPE_AREA_M2,
        nominal_pipe_area_m2=PIPE_AREA_M2,
    )

    assert 0.0 < forward.volume_flow_left_to_right_m3_s < 1.0e-5
    assert reverse.volume_flow_left_to_right_m3_s == pytest.approx(
        -forward.volume_flow_left_to_right_m3_s,
        rel=2.0e-15,
    )
    assert forward.signed_pressure_jump_Pa > 0.0
    assert reverse.signed_pressure_jump_Pa < 0.0
    assert forward.dissipation_power_W > 0.0
    assert reverse.dissipation_power_W > 0.0


@pytest.mark.parametrize("direction", (1.0, -1.0))
def test_pressurised_moc_super_acoustic_one_way_supply_is_explicit(
    direction: float,
) -> None:
    velocity = direction * 35.0
    left = trace(pressure=0.0, velocity=velocity)
    right = trace(pressure=0.0, velocity=velocity)
    solution = solve_passive_pressurised_moc_valve(
        left,
        right,
        time_s=0.19999,
        valve_flow_area_m2=PIPE_AREA_M2,
        nominal_pipe_area_m2=PIPE_AREA_M2,
    )

    expected_regime = (
        PressurisedMocValveControlRegime.LEFT_SUPPLY_CHOKED
        if direction > 0.0
        else PressurisedMocValveControlRegime.RIGHT_SUPPLY_CHOKED
    )
    assert solution.control_regime is expected_regime
    assert solution.volume_flow_left_to_right_m3_s == pytest.approx(
        PIPE_AREA_M2 * velocity,
        rel=0.0,
        abs=2.0e-20,
    )
    assert solution.signed_pressure_jump_Pa * direction > 0.0
    assert solution.dissipation_power_W > 0.0


@pytest.mark.parametrize(
    ("left_velocity", "right_velocity"),
    ((35.0, 0.0), (0.0, -35.0), (-35.0, 35.0), (28.0, 28.0)),
)
def test_pressurised_moc_incompatible_characteristic_counts_reject(
    left_velocity: float,
    right_velocity: float,
) -> None:
    with pytest.raises(PressurisedMocValveNoRootError):
        solve_passive_pressurised_moc_valve(
            trace(pressure=0.0, velocity=left_velocity),
            trace(pressure=0.0, velocity=right_velocity),
            time_s=0.10,
            valve_flow_area_m2=PIPE_AREA_M2,
            nominal_pipe_area_m2=PIPE_AREA_M2,
        )


def test_opening_monotonically_increases_flow_for_one_fixed_pressure_drive() -> None:
    left = trace(pressure=6_000.0)
    right = trace(pressure=0.0)
    flows = [
        solve_passive_liquid_valve(
            left,
            right,
            time_s=time,
            valve_flow_area_m2=PIPE_AREA_M2,
        ).volume_flow_left_to_right_m3_s
        for time in (0.0, 0.05, 0.10, 0.15, 0.20)
    ]
    assert all(next_flow > flow for flow, next_flow in zip(flows, flows[1:]))


def test_zero_drive_is_an_exact_quiescent_fixed_point_even_when_nearly_closed() -> None:
    solution = solve_passive_liquid_valve(
        trace(pressure=5_973.0),
        trace(pressure=5_973.0),
        time_s=0.0,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    assert solution.volume_flow_left_to_right_m3_s == 0.0
    assert solution.valve_velocity_m_s == 0.0
    assert solution.signed_pressure_jump_Pa == 0.0
    assert solution.valve_wall_force_on_liquid_N == 0.0
    assert solution.dissipation_power_W == 0.0


def test_ledger_is_equal_and_opposite_and_accumulates_wall_impulse_and_energy() -> None:
    ledger = LocalValveLedger()
    forward = solve_passive_liquid_valve(
        trace(pressure=6_000.0),
        trace(pressure=0.0),
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    reverse = solve_passive_liquid_valve(
        trace(pressure=0.0),
        trace(pressure=6_000.0),
        time_s=0.10,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    first = ledger.commit(forward, dt_s=2.0e-4)
    second = ledger.commit(reverse, dt_s=2.0e-4)

    assert first.left_liquid_volume_change_m3 == -first.right_liquid_volume_change_m3
    assert first.liquid_volume_residual_m3 == 0.0
    assert first.valve_wall_impulse_on_liquid_N_s < 0.0
    assert first.dissipated_energy_J > 0.0
    assert second.liquid_volume_residual_m3 == 0.0
    assert second.valve_wall_impulse_on_liquid_N_s > 0.0
    assert second.dissipated_energy_J > 0.0
    assert ledger.cumulative_signed_through_volume_m3 == pytest.approx(
        0.0,
        abs=1.0e-24,
    )
    assert ledger.cumulative_absolute_through_volume_m3 > 0.0
    assert ledger.cumulative_valve_wall_impulse_on_liquid_N_s == pytest.approx(
        0.0,
        abs=1.0e-18,
    )
    assert ledger.cumulative_dissipated_energy_J > first.dissipated_energy_J
    assert ledger.commit_count == 2
    assert ledger.snapshot()["liquid_mass_source_m3"] == 0.0


def test_explicit_valve_area_is_not_silently_replaced_by_2d_extrusion_area() -> None:
    signature = inspect.signature(solve_passive_liquid_valve)
    assert signature.parameters["valve_flow_area_m2"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        solve_passive_liquid_valve(
            trace(pressure=1_000.0),
            trace(pressure=0.0),
            time_s=0.1,
        )


@pytest.mark.parametrize(
    "bad_trace",
    (
        dict(area_m2=0.0, velocity_m_s=0.0, gauge_pressure_Pa=0.0, wave_speed_m_s=28.0),
        dict(area_m2=1.0, velocity_m_s=math.nan, gauge_pressure_Pa=0.0, wave_speed_m_s=28.0),
        dict(area_m2=1.0, velocity_m_s=0.0, gauge_pressure_Pa=0.0, wave_speed_m_s=0.0),
        dict(area_m2=1.0, velocity_m_s=0.0, gauge_pressure_Pa=0.0, wave_speed_m_s=28.0, density_kg_m3=-1.0),
    ),
)
def test_invalid_traces_are_rejected(bad_trace: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LiquidValveTrace(**bad_trace)


def test_invalid_area_time_and_ledger_step_are_rejected() -> None:
    left = trace(pressure=1_000.0)
    right = trace(pressure=0.0)
    with pytest.raises(ValueError):
        solve_passive_liquid_valve(
            left,
            right,
            time_s=0.1,
            valve_flow_area_m2=0.0,
        )
    with pytest.raises(ValueError):
        shared_opening_state(math.inf)
    solution = solve_passive_liquid_valve(
        left,
        right,
        time_s=0.1,
        valve_flow_area_m2=PIPE_AREA_M2,
    )
    with pytest.raises(ValueError):
        LocalValveLedger().commit(solution, dt_s=0.0)


def free_surface_trace(
    *,
    area: float = 8.0e-4,
    discharge: float = 8.0e-5,
    celerity: float = 1.2,
    density: float = WATER_DENSITY_KG_M3,
) -> CircularSaintVenantValveTrace:
    return CircularSaintVenantValveTrace(
        area_m2=area,
        discharge_m3_s=discharge,
        celerity_m_s=celerity,
        full_area_m2=PIPE_AREA_M2,
        density_kg_m3=density,
    )


def physical_supercritical_trace() -> CircularSaintVenantValveTrace:
    """A circular D=0.05 m trace with c(A) from the Case-1 geometry."""

    return free_surface_trace(
        area=8.0e-4,
        discharge=3.6e-4,
        celerity=0.3983191677460087,
    )


def test_circular_saint_venant_impedances_follow_incoming_invariants() -> None:
    left = free_surface_trace(area=8.0e-4, discharge=1.6e-4, celerity=1.1)
    right = free_surface_trace(area=9.0e-4, discharge=-9.0e-5, celerity=1.3)

    expected_left = (
        WATER_DENSITY_KG_M3
        * 1.1**2
        / (8.0e-4 * (1.1 - 0.2))
    )
    expected_right = (
        WATER_DENSITY_KG_M3
        * 1.3**2
        / (9.0e-4 * (1.3 - 0.1))
    )
    assert left.left_incoming_flow_impedance_Pa_s_m3 == pytest.approx(
        expected_left,
        rel=2.0e-15,
    )
    assert right.right_incoming_flow_impedance_Pa_s_m3 == pytest.approx(
        expected_right,
        rel=2.0e-15,
    )
    assert left.left_incoming_flow_impedance_Pa_s_m3 != pytest.approx(
        WATER_DENSITY_KG_M3 * WAVE_SPEED_M_S / left.area_m2
    )


def test_circular_saint_venant_k0_returns_native_flux_bits_exactly() -> None:
    native_flow = 1.7e-4
    native_momentum = 0.003456789012345
    solution = solve_passive_circular_saint_venant_valve(
        free_surface_trace(),
        free_surface_trace(area=9.0e-4, discharge=5.0e-5, celerity=1.4),
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=native_momentum,
        time_s=OPENING_DURATION_S,
    )

    assert solution.is_exact_native
    assert solution.volume_flow_left_to_right_m3_s == native_flow
    assert solution.left_specific_momentum_flux_m4_s2 == native_momentum
    assert solution.right_specific_momentum_flux_m4_s2 == native_momentum
    assert solution.signed_pressure_jump_Pa == 0.0
    assert solution.valve_wall_force_on_liquid_N == 0.0
    assert solution.dissipation_power_W == 0.0


def test_circular_saint_venant_forward_flow_is_passive_and_force_closed() -> None:
    native_flow = 2.0e-4
    solution = solve_passive_circular_saint_venant_valve(
        free_surface_trace(area=7.5e-4, discharge=1.0e-4, celerity=1.1),
        free_surface_trace(area=9.0e-4, discharge=0.0, celerity=1.3),
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=0.004,
        time_s=0.10,
    )

    assert 0.0 < solution.volume_flow_left_to_right_m3_s < native_flow
    assert solution.upwind_wetted_area_m2 == 7.5e-4
    assert solution.control_regime is (
        FreeSurfaceValveControlRegime.TWO_SIDED_CHARACTERISTIC
    )
    assert solution.left_characteristic_active
    assert solution.right_characteristic_active
    assert solution.signed_pressure_jump_Pa > 0.0
    assert solution.valve_wall_force_on_liquid_N < 0.0
    assert solution.dissipation_power_W == pytest.approx(
        solution.signed_pressure_jump_Pa
        * solution.volume_flow_left_to_right_m3_s,
        rel=0.0,
        abs=0.0,
    )
    assert solution.dissipation_power_W > 0.0
    assert abs(solution.impedance_residual_Pa) < 2.0e-10
    assert abs(solution.pressure_partition_residual_Pa) < 2.0e-13
    assert abs(solution.momentum_force_residual_N) < 2.0e-15


def test_circular_saint_venant_reverse_flow_uses_east_wetted_area_and_is_odd() -> None:
    west = free_surface_trace(area=7.0e-4, discharge=0.0, celerity=1.2)
    east = free_surface_trace(area=9.0e-4, discharge=0.0, celerity=1.2)
    forward = solve_passive_circular_saint_venant_valve(
        west,
        east,
        native_volume_flow_m3_s=1.5e-4,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.10,
    )
    reverse = solve_passive_circular_saint_venant_valve(
        east,
        west,
        native_volume_flow_m3_s=-1.5e-4,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.10,
    )

    assert forward.upwind_wetted_area_m2 == west.area_m2
    assert reverse.upwind_wetted_area_m2 == west.area_m2
    assert reverse.volume_flow_left_to_right_m3_s == pytest.approx(
        -forward.volume_flow_left_to_right_m3_s,
        rel=2.0e-15,
    )
    assert reverse.signed_pressure_jump_Pa == pytest.approx(
        -forward.signed_pressure_jump_Pa,
        rel=2.0e-15,
    )
    assert reverse.valve_wall_force_on_liquid_N == pytest.approx(
        -forward.valve_wall_force_on_liquid_N,
        rel=2.0e-15,
    )
    assert reverse.dissipation_power_W == pytest.approx(
        forward.dissipation_power_W,
        rel=2.0e-15,
    )


def test_circular_saint_venant_positive_k_converges_continuously_to_native() -> None:
    native_flow = 1.4e-4
    errors = []
    for time_s in (0.19, 0.199, 0.1999, 0.19999):
        solution = solve_passive_circular_saint_venant_valve(
            free_surface_trace(),
            free_surface_trace(area=9.0e-4, discharge=0.0, celerity=1.3),
            native_volume_flow_m3_s=native_flow,
            native_specific_momentum_flux_m4_s2=0.003,
            time_s=time_s,
        )
        assert solution.opening.loss_coefficient > 0.0
        errors.append(abs(native_flow - solution.volume_flow_left_to_right_m3_s))
    assert all(next_error < error for error, next_error in zip(errors, errors[1:]))
    exact = solve_passive_circular_saint_venant_valve(
        free_surface_trace(),
        free_surface_trace(area=9.0e-4, discharge=0.0, celerity=1.3),
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.20,
    )
    assert exact.volume_flow_left_to_right_m3_s == native_flow


def test_circular_saint_venant_zero_native_drive_is_exact_fixed_point() -> None:
    solution = solve_passive_circular_saint_venant_valve(
        free_surface_trace(),
        free_surface_trace(),
        native_volume_flow_m3_s=0.0,
        native_specific_momentum_flux_m4_s2=0.002,
        time_s=0.0,
    )
    assert solution.volume_flow_left_to_right_m3_s == 0.0
    assert solution.signed_pressure_jump_Pa == 0.0
    assert solution.dissipation_power_W == 0.0
    assert solution.left_specific_momentum_flux_m4_s2 == 0.002
    assert solution.right_specific_momentum_flux_m4_s2 == 0.002
    assert solution.is_exact_native


@pytest.mark.parametrize(
    "values",
    (
        dict(area_m2=0.0, discharge_m3_s=0.0, celerity_m_s=1.0),
        dict(area_m2=PIPE_AREA_M2, discharge_m3_s=0.0, celerity_m_s=1.0),
        dict(area_m2=-1.0e-9, discharge_m3_s=0.0, celerity_m_s=0.0),
        dict(area_m2=8.0e-4, discharge_m3_s=0.0, celerity_m_s=0.0),
    ),
)
def test_circular_saint_venant_rejects_invalid_dry_full_or_wet_traces(
    values: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        CircularSaintVenantValveTrace(
            **values,
            full_area_m2=PIPE_AREA_M2,
        )


def test_circular_saint_venant_keeps_exact_dry_critical_and_supercritical() -> None:
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    critical = free_surface_trace(
        area=8.0e-4,
        discharge=8.0e-4,
        celerity=1.0,
    )
    supercritical = free_surface_trace(
        area=8.0e-4,
        discharge=9.6e-4,
        celerity=1.0,
    )

    assert dry.is_dry
    assert dry.froude_number is None
    assert critical.froude_number == 1.0
    assert supercritical.froude_number == pytest.approx(1.2)


def test_one_sided_wet_to_dry_characteristic_branch_is_conservative_passive() -> None:
    wet = free_surface_trace(
        area=8.0e-4,
        discharge=2.0e-4,
        celerity=1.0,
    )
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    native_flow = 2.2e-4
    solution = solve_passive_circular_saint_venant_valve(
        wet,
        dry,
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.10,
    )

    assert solution.control_regime is (
        FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
    )
    assert solution.upstream_side == "left"
    assert solution.left_characteristic_active
    assert not solution.right_characteristic_active
    assert solution.native_offset_nonlinear_characteristic
    assert 0.0 < solution.volume_flow_left_to_right_m3_s < native_flow
    assert solution.signed_pressure_jump_Pa > 0.0
    assert solution.dissipation_power_W > 0.0
    assert abs(solution.impedance_residual_Pa) < 2.0e-9
    assert solution.momentum_force_residual_N == pytest.approx(0.0, abs=2e-15)


def test_reverse_wet_to_dry_branch_is_mirror_symmetric() -> None:
    wet = free_surface_trace(
        area=8.0e-4,
        discharge=-2.0e-4,
        celerity=1.0,
    )
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    forward = solve_passive_circular_saint_venant_valve(
        free_surface_trace(
            area=8.0e-4,
            discharge=2.0e-4,
            celerity=1.0,
        ),
        dry,
        native_volume_flow_m3_s=2.2e-4,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.10,
    )
    reverse = solve_passive_circular_saint_venant_valve(
        dry,
        wet,
        native_volume_flow_m3_s=-2.2e-4,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=0.10,
    )

    assert reverse.control_regime is (
        FreeSurfaceValveControlRegime.ONE_SIDED_CHARACTERISTIC
    )
    assert reverse.upstream_side == "right"
    assert reverse.volume_flow_left_to_right_m3_s == pytest.approx(
        -forward.volume_flow_left_to_right_m3_s,
        rel=2.0e-15,
    )
    assert reverse.signed_pressure_jump_Pa == pytest.approx(
        -forward.signed_pressure_jump_Pa,
        rel=2.0e-15,
    )
    assert reverse.dissipation_power_W == pytest.approx(
        forward.dissipation_power_W,
        rel=2.0e-15,
    )


def test_one_sided_nonlinear_branch_converges_to_native_without_a_k_threshold() -> None:
    wet = free_surface_trace(
        area=8.0e-4,
        discharge=2.0e-4,
        celerity=1.0,
    )
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    native_flow = 2.2e-4
    flow_errors = []
    area_errors = []
    for time_s in (0.19, 0.199, 0.1999, 0.19999):
        solution = solve_passive_circular_saint_venant_valve(
            wet,
            dry,
            native_volume_flow_m3_s=native_flow,
            native_specific_momentum_flux_m4_s2=0.003,
            time_s=time_s,
        )
        assert solution.opening.loss_coefficient > 0.0
        assert solution.native_offset_nonlinear_characteristic
        flow_errors.append(abs(native_flow - solution.volume_flow_left_to_right_m3_s))
        area_errors.append(abs(wet.area_m2 - solution.upwind_wetted_area_m2))
    assert all(next_error < error for error, next_error in zip(flow_errors, flow_errors[1:]))
    assert all(next_error < error for error, next_error in zip(area_errors, area_errors[1:]))

    exact = solve_passive_circular_saint_venant_valve(
        wet,
        dry,
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=0.003,
        time_s=OPENING_DURATION_S,
    )
    assert exact.volume_flow_left_to_right_m3_s == native_flow
    assert exact.left_specific_momentum_flux_m4_s2 == 0.003
    assert exact.right_specific_momentum_flux_m4_s2 == 0.003


def test_supercritical_upstream_is_explicit_supply_choked_not_fr_clipped() -> None:
    upstream = physical_supercritical_trace()
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    native_flow = 3.5e-4
    native_momentum = 0.001
    solution = solve_passive_circular_saint_venant_valve(
        upstream,
        dry,
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=native_momentum,
        time_s=0.195,
    )

    assert solution.control_regime is (
        FreeSurfaceValveControlRegime.UPSTREAM_SUPPLY_CHOKED
    )
    assert solution.volume_flow_left_to_right_m3_s == native_flow
    assert solution.left_pressure_correction_Pa == 0.0
    assert solution.right_pressure_correction_Pa == -solution.signed_pressure_jump_Pa
    assert solution.dissipation_power_W == pytest.approx(
        solution.signed_pressure_jump_Pa * native_flow,
        rel=0.0,
        abs=0.0,
    )
    assert solution.dissipation_power_W > 0.0
    assert solution.momentum_force_residual_N == pytest.approx(0.0, abs=2e-15)
    assert solution.upstream_specific_energy_m is not None
    assert solution.critical_area_m2 is not None
    assert solution.minimum_specific_energy_m is not None
    assert solution.supply_energy_margin_m is not None
    assert solution.supply_energy_margin_m > 0.0
    assert solution.valve_head_loss_m == pytest.approx(
        abs(solution.signed_pressure_jump_Pa)
        / (solution.density_kg_m3 * solution.gravity_m_s2)
    )


def test_supply_choke_rejects_loss_that_requires_an_upstream_entropy_shock() -> None:
    upstream = physical_supercritical_trace()
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    with pytest.raises(ValueError, match="resolved upstream shock"):
        solve_passive_circular_saint_venant_valve(
            upstream,
            dry,
            native_volume_flow_m3_s=3.5e-4,
            native_specific_momentum_flux_m4_s2=0.001,
            time_s=0.190,
        )


def test_supply_choke_rejects_a_downstream_feedback_characteristic() -> None:
    upstream = physical_supercritical_trace()
    downstream = free_surface_trace(
        area=8.0e-4,
        discharge=1.0e-4,
        celerity=0.3983191677460087,
    )
    with pytest.raises(ValueError, match="dry or supercritical-outflow"):
        solve_passive_circular_saint_venant_valve(
            upstream,
            downstream,
            native_volume_flow_m3_s=3.5e-4,
            native_specific_momentum_flux_m4_s2=0.001,
            time_s=0.195,
        )


def test_k0_is_native_even_for_dry_donor_and_supercritical_trace() -> None:
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    supercritical = physical_supercritical_trace()
    native_flow = 2.0e-4
    native_momentum = 0.003456789012345
    solution = solve_passive_circular_saint_venant_valve(
        dry,
        supercritical,
        native_volume_flow_m3_s=native_flow,
        native_specific_momentum_flux_m4_s2=native_momentum,
        time_s=OPENING_DURATION_S,
    )

    assert solution.control_regime is FreeSurfaceValveControlRegime.EXACT_NATIVE
    assert solution.volume_flow_left_to_right_m3_s == native_flow
    assert solution.left_specific_momentum_flux_m4_s2 == native_momentum
    assert solution.right_specific_momentum_flux_m4_s2 == native_momentum
    assert solution.is_exact_native


def test_positive_k_rejects_dry_donor_and_opposing_supercritical_stream() -> None:
    dry = CircularSaintVenantValveTrace(
        area_m2=0.0,
        discharge_m3_s=0.0,
        celerity_m_s=0.0,
        full_area_m2=PIPE_AREA_M2,
    )
    wet = free_surface_trace(area=8.0e-4, discharge=0.0, celerity=1.0)
    with pytest.raises(ValueError, match="donor is dry"):
        solve_passive_circular_saint_venant_valve(
            dry,
            wet,
            native_volume_flow_m3_s=2.0e-4,
            native_specific_momentum_flux_m4_s2=0.003,
            time_s=0.10,
        )

    opposing = free_surface_trace(
        area=8.0e-4,
        discharge=-9.6e-4,
        celerity=1.0,
    )
    with pytest.raises(ValueError, match="opposing supercritical"):
        solve_passive_circular_saint_venant_valve(
            wet,
            opposing,
            native_volume_flow_m3_s=2.0e-4,
            native_specific_momentum_flux_m4_s2=0.003,
            time_s=0.10,
        )


def test_circular_saint_venant_requires_one_section_and_density() -> None:
    left = free_surface_trace()
    wrong_section = CircularSaintVenantValveTrace(
        area_m2=8.0e-4,
        discharge_m3_s=0.0,
        celerity_m_s=1.2,
        full_area_m2=2.0e-3,
    )
    wrong_density = free_surface_trace(density=999.0)
    with pytest.raises(ValueError, match="one circular section"):
        solve_passive_circular_saint_venant_valve(
            left,
            wrong_section,
            native_volume_flow_m3_s=1.0e-4,
            native_specific_momentum_flux_m4_s2=0.002,
            time_s=0.1,
        )
    with pytest.raises(ValueError, match="one liquid density"):
        solve_passive_circular_saint_venant_valve(
            left,
            wrong_density,
            native_volume_flow_m3_s=1.0e-4,
            native_specific_momentum_flux_m4_s2=0.002,
            time_s=0.1,
        )
