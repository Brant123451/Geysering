from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import casea_post_t_coupled_stage as coupled_module  # noqa: E402
from casea_coupled_gas_network import CoupledGasParameters  # noqa: E402
from casea_horizontal_liquid_operator import PressurePotentialState  # noqa: E402
from casea_post_t_coupled_stage import (  # noqa: E402
    PostTCoupledGeometry,
    PostTCoupledState,
    advance_post_t_coupled_ssprk2,
    evaluate_post_t_coupled_stage_rhs,
)
from casea_post_t_liquid_stage import (  # noqa: E402
    BranchPressureEvaluation,
    PostTLiquidGeometry,
)
from casea_post_t_sideport_liquid_stage import (  # noqa: E402
    PostTSidePortGeometry,
)
from casea_topology_event import BranchFrontTopology  # noqa: E402


def _synthetic_pressure_callback(
    atmospheric_pressure: float,
    liquid_density: float,
    calls: list | None = None,
):
    def callback(branch, area, discharge, gas_mass, gas_momentum):
        if calls is not None:
            calls.append(
                (
                    branch,
                    np.array(area, copy=True),
                    np.array(discharge, copy=True),
                    np.array(gas_mass, copy=True),
                    np.array(gas_momentum, copy=True),
                )
            )
        area = np.asarray(area, dtype=float)
        discharge = np.asarray(discharge, dtype=float)
        velocity = discharge / area
        wave_speed = np.full_like(area, 28.0)
        potential = atmospheric_pressure * area / liquid_density
        pressure = PressurePotentialState(
            potential=potential,
            derivative=wave_speed**2,
            discharge_derivative=2.0 * velocity,
            celerity=wave_speed,
            eigenvalue_minus=velocity - wave_speed,
            eigenvalue_plus=velocity + wave_speed,
            lambda_value=np.zeros_like(area),
            lambda_derivative=np.zeros_like(area),
            stratified=np.ones_like(area, dtype=bool),
        )
        return BranchPressureEvaluation(
            pressure=pressure,
            face_pressure_abs=np.full_like(area, atmospheric_pressure),
            node_pressure_offset=np.zeros_like(area),
            momentum_source=np.zeros_like(area),
            potential_pressure_abs=np.full_like(area, atmospheric_pressure),
        )

    return callback


def _case(*, active_vertical_count: int = 6, pressure_ratio: float = 1.0):
    gas = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
    dx = 0.02
    dz = 0.02
    nh = 8
    nv = 6
    jx = 3
    Alt = np.full(nh, 0.60 * gas.horizontal_area)
    Qlt = np.zeros(nh)
    Alr = np.zeros(nv)
    Alr[:active_vertical_count] = 0.60 * gas.vertical_area
    Qlr = np.zeros(nv)
    rho = pressure_ratio * gas.rho_atmospheric
    Mgt = rho * (gas.horizontal_area - Alt) * dx
    Mgr = rho * (gas.vertical_area - Alr) * dz
    Jgt = np.zeros(nh)
    Jgrs = np.zeros(nv)
    Mgrs = Mgr.copy()
    state = PostTCoupledState(
        Alt=Alt,
        Qlt=Qlt,
        Alr=Alr,
        Qlr=Qlr,
        Mgt=Mgt,
        Jgt=Jgt,
        Mgr=Mgr,
        Jgrs=Jgrs,
        Mgrs=Mgrs,
        east_front=BranchFrontTopology("east", 0.08),
        vertical_front=BranchFrontTopology("vertical", 0.12),
    )
    geometry = PostTCoupledGeometry(
        liquid=PostTLiquidGeometry(
            junction_face_index=4,
            horizontal_cell_width=dx,
            vertical_cell_width=dz,
            liquid_density=gas.rho_l,
            atmospheric_pressure_abs=gas.atmospheric_pressure,
        ),
        gas=gas,
        gas_junction_index=jx,
        vertical_liquid_active_count=active_vertical_count,
    )
    callback = _synthetic_pressure_callback(
        gas.atmospheric_pressure, gas.rho_l
    )
    return state, geometry, callback


def _state_bytes(state: PostTCoupledState) -> tuple[bytes, ...]:
    return tuple(
        getattr(state, name).tobytes()
        for name in (
            "Alt", "Qlt", "Alr", "Qlr", "Mgt", "Jgt", "Mgr", "Jgrs",
            "Mgrs",
        )
    )


def test_stage_evaluation_is_repeatable_and_side_effect_free() -> None:
    state, geometry, callback = _case()
    before = _state_bytes(state)
    first = evaluate_post_t_coupled_stage_rhs(
        state, geometry=geometry, pressure_callback=callback, top_open=False
    )
    middle = _state_bytes(state)
    second = evaluate_post_t_coupled_stage_rhs(
        state, geometry=geometry, pressure_callback=callback, top_open=False
    )
    after = _state_bytes(state)

    assert before == middle == after
    for name in (
        "dAlt_dt", "dQlt_dt", "dAlr_dt", "dQlr_dt", "dMgt_dt",
        "dJgt_dt", "dMgr_dt", "dJgrs_dt", "dMgrs_dt",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_closed_advance_conserves_all_liquid_and_gas_mass() -> None:
    state, geometry, callback = _case()
    advanced = advance_post_t_coupled_ssprk2(
        state,
        dt=1.0e-8,
        geometry=geometry,
        pressure_callback=callback,
        top_open=False,
    )

    assert abs(advanced.atmospheric_mass_exchange) == 0.0
    assert abs(advanced.escaped_tracer_mass) == 0.0
    assert abs(advanced.liquid_volume_error) < 3.0e-18
    assert abs(advanced.gas_mass_error) < 3.0e-18
    assert abs(advanced.tracer_mass_error) < 3.0e-18
    assert abs(advanced.t_liquid_volume_residual_integral) < 1.0e-18
    assert abs(advanced.t_gas_mass_residual_integral) == 0.0


def test_finite_width_side_port_replaces_zero_storage_node_conservatively() -> None:
    state, old_geometry, callback = _case()
    side_port = PostTSidePortGeometry(
        horizontal_cell_width=old_geometry.liquid.horizontal_cell_width,
        vertical_cell_width=old_geometry.liquid.vertical_cell_width,
        liquid_density=old_geometry.liquid.liquid_density,
        # The synthetic grid is only 0.16 m long; place a 0.04 m circular
        # mouth at its centre.  Production Case A uses x=3.516 m, D=0.0571 m.
        junction_center_x=0.08,
        opening_diameter=0.04,
        atmospheric_pressure_abs=old_geometry.liquid.atmospheric_pressure_abs,
    )
    geometry = PostTCoupledGeometry(
        liquid=side_port,
        gas=old_geometry.gas,
        gas_junction_index=old_geometry.gas_junction_index,
        vertical_liquid_active_count=old_geometry.vertical_liquid_active_count,
    )
    advanced = advance_post_t_coupled_ssprk2(
        state,
        dt=1.0e-8,
        geometry=geometry,
        pressure_callback=callback,
        top_open=False,
    )

    assert abs(advanced.liquid_volume_error) < 3.0e-18
    assert abs(advanced.t_liquid_volume_residual_integral) < 3.0e-18
    assert isinstance(
        advanced.first_stage.liquid,
        coupled_module.PostTSidePortLiquidStageRhs,
    )


def test_t_internal_fluxes_cancel_and_drag_is_equal_opposite() -> None:
    state, geometry, callback = _case()
    Mgt = np.array(state.Mgt, copy=True)
    Jgt = np.array(state.Jgt, copy=True)
    Mgr = np.array(state.Mgr, copy=True)
    Jgrs = np.array(state.Jgrs, copy=True)
    Mgt *= 1.05
    Jgt[:] = 0.35 * Mgt
    Jgrs[:] = -0.18 * Mgr
    Qlt = np.array(state.Qlt, copy=True)
    Qlr = np.array(state.Qlr, copy=True)
    Qlt[:] = -0.08 * state.Alt
    Qlr[:] = 0.06 * state.Alr
    moving = PostTCoupledState(
        Alt=state.Alt,
        Qlt=Qlt,
        Alr=state.Alr,
        Qlr=Qlr,
        Mgt=Mgt,
        Jgt=Jgt,
        Mgr=Mgr,
        Jgrs=Jgrs,
        Mgrs=state.Mgrs,
        east_front=state.east_front,
        vertical_front=state.vertical_front,
    )
    stage = evaluate_post_t_coupled_stage_rhs(
        moving,
        geometry=geometry,
        pressure_callback=callback,
        top_open=False,
    )

    assert abs(stage.liquid_t_volume_residual) < 1.0e-12
    assert stage.gas.t_flux.mass_rate_horizontal_to_vertical > 0.0
    assert stage.gas_t_mass_residual == 0.0
    assert abs(stage.interphase_momentum_residual) < 3.0e-18
    np.testing.assert_allclose(
        stage.gas.dJgt_drag_dt
        + geometry.gas.rho_l * geometry.liquid.horizontal_cell_width
        * stage.gas.dQlt_drag_dt,
        0.0,
        atol=2.0e-18,
    )


def test_fronts_are_advanced_independently_by_their_own_heun_speeds() -> None:
    state, geometry, callback = _case()
    Jgt = np.array(state.Jgt, copy=True)
    Jgrs = np.array(state.Jgrs, copy=True)
    Jgt[geometry.gas_junction_index + 1 :] = (
        0.24 * state.Mgt[geometry.gas_junction_index + 1 :]
    )
    Jgrs[:] = -0.11 * state.Mgr
    moving = PostTCoupledState(
        Alt=state.Alt,
        Qlt=state.Qlt,
        Alr=state.Alr,
        Qlr=state.Qlr,
        Mgt=state.Mgt,
        Jgt=Jgt,
        Mgr=state.Mgr,
        Jgrs=Jgrs,
        Mgrs=state.Mgrs,
        east_front=state.east_front,
        vertical_front=state.vertical_front,
    )
    dt = 1.0e-8
    advanced = advance_post_t_coupled_ssprk2(
        moving,
        dt=dt,
        geometry=geometry,
        pressure_callback=callback,
        top_open=False,
    )
    expected_east = moving.east_front.position + 0.5 * dt * (
        advanced.first_stage.east_front_speed
        + advanced.second_stage.east_front_speed
    )
    expected_vertical = moving.vertical_front.position + 0.5 * dt * (
        advanced.first_stage.vertical_front_speed
        + advanced.second_stage.vertical_front_speed
    )

    assert np.isclose(advanced.state.east_front.position, expected_east)
    assert np.isclose(
        advanced.state.vertical_front.position, expected_vertical
    )
    assert advanced.first_stage.east_front_speed > 0.0
    assert advanced.first_stage.vertical_front_speed < 0.0
    assert advanced.state.east_front.position > moving.east_front.position
    assert advanced.state.vertical_front.position < moving.vertical_front.position


def test_ssprk2_recomputes_both_operators_from_the_predictor_state() -> None:
    state, geometry, _ = _case()
    Qlt = np.array(state.Qlt, copy=True)
    Qlt[geometry.liquid.junction_face_index - 1] = (
        0.04 * state.Alt[geometry.liquid.junction_face_index - 1]
    )
    moving = PostTCoupledState(
        Alt=state.Alt,
        Qlt=Qlt,
        Alr=state.Alr,
        Qlr=state.Qlr,
        Mgt=state.Mgt,
        Jgt=state.Jgt,
        Mgr=state.Mgr,
        Jgrs=state.Jgrs,
        Mgrs=state.Mgrs,
        east_front=state.east_front,
        vertical_front=state.vertical_front,
    )
    pressure_calls: list = []
    callback = _synthetic_pressure_callback(
        geometry.gas.atmospheric_pressure,
        geometry.gas.rho_l,
        pressure_calls,
    )
    gas_calls: list[tuple[bytes, bytes, float, float]] = []
    real_gas_evaluator = coupled_module.evaluate_resolved_gas_stage_rhs

    def recording_gas(*args, **kwargs):
        gas_calls.append(
            (
                np.asarray(args[0]).tobytes(),
                np.asarray(args[5]).tobytes(),
                kwargs["east_front"].position,
                kwargs["vertical_front"].position,
            )
        )
        return real_gas_evaluator(*args, **kwargs)

    with patch.object(
        coupled_module,
        "evaluate_resolved_gas_stage_rhs",
        side_effect=recording_gas,
    ):
        advanced = advance_post_t_coupled_ssprk2(
            moving,
            dt=1.0e-8,
            geometry=geometry,
            pressure_callback=callback,
            top_open=False,
        )

    assert len(pressure_calls) == 4  # horizontal + vertical at both stages
    assert len(gas_calls) == 2
    assert not np.array_equal(pressure_calls[0][1], pressure_calls[2][1])
    assert gas_calls[0] != gas_calls[1]
    assert advanced.first_stage is not advanced.second_stage


def test_open_top_mass_and_tracer_are_accumulated_with_heun_flux() -> None:
    state, geometry, callback = _case(pressure_ratio=1.10)
    dt = 1.0e-8
    advanced = advance_post_t_coupled_ssprk2(
        state,
        dt=dt,
        geometry=geometry,
        pressure_callback=callback,
        top_open=True,
    )
    expected_mass = 0.5 * dt * (
        advanced.first_stage.gas.top_flux.mass_rate
        + advanced.second_stage.gas.top_flux.mass_rate
    )
    expected_tracer = 0.5 * dt * (
        advanced.first_stage.gas.top_flux.tracer_mass_rate
        + advanced.second_stage.gas.top_flux.tracer_mass_rate
    )

    assert expected_mass > 0.0
    assert expected_tracer > 0.0
    assert advanced.atmospheric_mass_exchange == expected_mass
    assert advanced.escaped_tracer_mass == expected_tracer
    assert abs(advanced.gas_mass_error) < 3.0e-18
    assert abs(advanced.tracer_mass_error) < 3.0e-18


def test_vertical_dry_suffix_remains_exactly_dry() -> None:
    state, geometry, callback = _case(active_vertical_count=3)
    advanced = advance_post_t_coupled_ssprk2(
        state,
        dt=1.0e-8,
        geometry=geometry,
        pressure_callback=callback,
        top_open=False,
    )
    count = geometry.vertical_liquid_active_count

    assert np.array_equal(
        advanced.state.Alr[count:], np.zeros_like(advanced.state.Alr[count:])
    )
    assert np.array_equal(
        advanced.state.Qlr[count:], np.zeros_like(advanced.state.Qlr[count:])
    )
