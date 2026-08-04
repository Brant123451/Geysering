from __future__ import annotations

import numpy as np

from casea_coupled_gas_network import CoupledGasParameters
from casea_resolved_gas_stage import evaluate_resolved_gas_stage_rhs
from casea_topology_event import BranchFrontTopology


def _state(
    *,
    horizontal_pressure_ratio: float = 1.0,
    vertical_pressure_ratio: float = 1.0,
    horizontal_velocity: float = 0.0,
    vertical_velocity: float = 0.0,
    vertical_tracer_fraction: float = 1.0,
) -> tuple:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
    dx = 0.02
    dz = 0.02
    nh = 7
    nv = 5
    jx = 3
    Alt = np.zeros(nh)
    Alr = np.zeros(nv)
    Qlt = np.full(nh, horizontal_velocity * params.horizontal_area)
    Qlr = np.full(nv, vertical_velocity * params.vertical_area)
    rho_h = params.rho_atmospheric * horizontal_pressure_ratio
    rho_v = params.rho_atmospheric * vertical_pressure_ratio
    Mgt = np.full(nh, rho_h * params.horizontal_area * dx)
    Jgt = Mgt * horizontal_velocity
    Mgr = np.full(nv, rho_v * params.vertical_area * dz)
    Jgrs = Mgr * vertical_velocity
    Mgrs = Mgr * vertical_tracer_fraction
    return (
        params, dx, dz, jx, Mgt, Jgt, Mgr, Jgrs, Mgrs,
        Alt, Alr, Qlt, Qlr,
    )


def _evaluate(
    values: tuple,
    *,
    east_position: float,
    vertical_position: float,
    top_open: bool,
):
    (
        params,
        dx,
        dz,
        jx,
        Mgt,
        Jgt,
        Mgr,
        Jgrs,
        Mgrs,
        Alt,
        Alr,
        Qlt,
        Qlr,
    ) = values
    return evaluate_resolved_gas_stage_rhs(
        Mgt,
        Jgt,
        Mgr,
        Jgrs,
        Mgrs,
        Alt,
        Alr,
        Qlt,
        Qlr,
        dx=dx,
        dz=dz,
        junction_index=jx,
        params=params,
        east_front=BranchFrontTopology("east", east_position),
        vertical_front=BranchFrontTopology(
            "vertical", vertical_position
        ),
        top_open=top_open,
    )


def test_closed_stage_has_one_exact_internal_t_flux_and_no_mass_source() -> None:
    values = _state()
    result = _evaluate(
        values,
        east_position=0.03,
        vertical_position=0.03,
        top_open=False,
    )

    assert result.top_flux.area == 0.0
    assert result.top_flux.mass_rate == 0.0
    assert abs(result.t_flux.internal_mass_residual) == 0.0
    assert abs(result.total_gas_mass_ledger_residual) < 1.0e-14
    assert abs(result.tracer_mass_ledger_residual) < 1.0e-14
    assert abs(np.sum(result.dMgt_dt) + np.sum(result.dMgr_dt)) < 1.0e-14


def test_open_top_ledger_changes_only_by_reported_riemann_flux() -> None:
    values = _state(
        horizontal_pressure_ratio=1.10,
        vertical_pressure_ratio=1.10,
    )
    result = _evaluate(
        values,
        east_position=0.05,
        vertical_position=0.09,
        top_open=True,
    )

    assert result.top_flux.area > 0.0
    assert result.top_flux.mass_rate > 0.0
    assert result.top_flux.tracer_mass_rate > 0.0
    assert abs(
        np.sum(result.dMgt_dt)
        + np.sum(result.dMgr_dt)
        + result.top_flux.mass_rate
    ) < 2.0e-14
    assert abs(
        np.sum(result.dMgt_dt)
        + np.sum(result.dMgrs_dt)
        + result.top_flux.tracer_mass_rate
    ) < 2.0e-14
    assert abs(result.total_gas_mass_ledger_residual) < 2.0e-14
    assert abs(result.tracer_mass_ledger_residual) < 2.0e-14


def test_t_flux_is_subtracted_and_added_once() -> None:
    values = _state(
        horizontal_pressure_ratio=1.10,
        vertical_pressure_ratio=1.00,
    )
    result = _evaluate(
        values,
        east_position=0.03,
        vertical_position=0.03,
        top_open=False,
    )

    transfer = result.t_flux.mass_rate_horizontal_to_vertical
    assert transfer > 0.0
    assert result.t_flux.horizontal_mass_rate == -transfer
    assert result.t_flux.vertical_mass_rate == transfer
    assert abs(np.sum(result.dMgt_dt) + transfer) < 2.0e-14
    assert abs(np.sum(result.dMgr_dt) - transfer) < 2.0e-14
    assert result.t_flux.internal_mass_residual == 0.0


def test_zero_fronts_initialise_from_real_t_face_flux_without_receiver_remap() -> None:
    values = list(
        _state(
            horizontal_pressure_ratio=1.10,
            vertical_pressure_ratio=1.00,
            horizontal_velocity=0.25,
        )
    )
    before = tuple(array.tobytes() for array in values[4:])
    result = _evaluate(
        tuple(values),
        east_position=0.0,
        vertical_position=0.0,
        top_open=False,
    )

    assert result.east_front.source_face == "east_material_front_riemann"
    assert abs(result.east_front.speed - 0.25) < 1.0e-14
    assert result.vertical_front.source_face == "t_riemann"
    assert result.vertical_front.mass_rate == (
        result.t_flux.mass_rate_horizontal_to_vertical
    )
    assert result.vertical_front.area == result.t_flux.area
    assert result.vertical_front.area > 0.0
    assert result.vertical_front.speed > 0.0
    after = tuple(array.tobytes() for array in values[4:])
    assert after == before


def test_zero_front_with_no_geometric_opening_is_valid_and_stationary() -> None:
    values = list(_state(horizontal_pressure_ratio=1.10))
    params, _, _, jx = values[:4]
    Alt = values[9]
    Alr = values[10]
    Alt[jx + 1] = params.horizontal_area
    Alr[0] = params.vertical_area

    result = _evaluate(
        tuple(values),
        east_position=0.0,
        vertical_position=0.0,
        top_open=False,
    )

    assert not result.horizontal_active[jx + 1]
    assert not result.vertical_bottom_active[0]
    assert result.t_flux.area == 0.0
    assert result.t_flux.mass_rate_horizontal_to_vertical == 0.0
    assert result.east_flux.area == 0.0
    assert result.east_flux.mass_rate_west_to_east == 0.0
    assert result.east_front.source_face == "closed"
    assert result.east_front.speed == 0.0
    assert result.dMgt_dt[jx + 1] == 0.0
    assert result.vertical_front.speed == 0.0
    assert abs(result.total_gas_mass_ledger_residual) < 1.0e-14


def test_finite_east_void_uses_one_paired_riemann_flux_for_front_motion() -> None:
    values = list(_state())
    params, dx, _, jx = values[:4]
    Mgt, Jgt, Mgr, Jgrs, Mgrs = values[4:9]
    Alt, Alr, Qlt, Qlr = values[9:13]

    # Isolate the T-east material face: one west donor, one geometrically open
    # east cut cell, and liquid-full cells on every other horizontal face.
    Alt[:] = params.horizontal_area
    Alt[jx] = 0.50 * params.horizontal_area
    Alt[jx + 1] = 0.50 * params.horizontal_area
    Mgt[:] = 0.0
    Jgt[:] = 0.0
    gas_area = 0.50 * params.horizontal_area
    donor_density = 1.08 * params.rho_atmospheric
    receiver_density = 1.00 * params.rho_atmospheric
    Mgt[jx] = donor_density * gas_area * dx
    Mgt[jx + 1] = receiver_density * gas_area * dx
    Jgt[jx] = 0.18 * Mgt[jx]
    Alr[:] = params.vertical_area
    Mgr[:] = Jgrs[:] = Mgrs[:] = 0.0
    Qlt[:] = Qlr[:] = 0.0

    result = _evaluate(
        tuple(values),
        east_position=0.0,
        vertical_position=0.0,
        top_open=False,
    )

    transfer = result.east_flux.mass_rate_west_to_east
    assert result.east_flux.area == gas_area
    assert result.east_flux.donor_index == jx
    assert result.east_flux.receiver_index == jx + 1
    assert transfer > 0.0
    assert result.east_flux.donor_mass_rate == -transfer
    assert result.east_flux.receiver_mass_rate == transfer
    assert result.east_flux.internal_mass_residual == 0.0
    assert abs(result.dMgt_dt[jx] + transfer) < 2.0e-14
    assert abs(result.dMgt_dt[jx + 1] - transfer) < 2.0e-14
    assert result.east_front.mass_rate == transfer
    assert result.east_front.source_face == "east_material_front_riemann"
    assert result.east_front.speed > 0.0
    assert abs(
        result.east_front.speed
        - transfer / (result.east_flux.upwind_density * result.east_flux.area)
    ) < 2.0e-14
    assert abs(np.sum(result.dMgt_dt)) < 2.0e-14
    assert abs(result.total_gas_mass_ledger_residual) < 2.0e-14


def test_east_and_vertical_front_topologies_are_independent() -> None:
    values = _state(horizontal_pressure_ratio=1.04)
    base = _evaluate(
        values,
        east_position=0.0,
        vertical_position=0.01,
        top_open=False,
    )
    east_moved = _evaluate(
        values,
        east_position=0.05,
        vertical_position=0.01,
        top_open=False,
    )
    vertical_moved = _evaluate(
        values,
        east_position=0.0,
        vertical_position=0.07,
        top_open=False,
    )

    assert np.array_equal(
        base.vertical_bottom_active,
        east_moved.vertical_bottom_active,
    )
    assert np.array_equal(
        base.horizontal_active,
        vertical_moved.horizontal_active,
    )
    assert not np.array_equal(
        base.horizontal_active,
        east_moved.horizontal_active,
    )
    assert not np.array_equal(
        base.vertical_bottom_active,
        vertical_moved.vertical_bottom_active,
    )


def test_repeated_stage_evaluation_is_bitwise_side_effect_free() -> None:
    values = _state(
        horizontal_pressure_ratio=1.03,
        vertical_pressure_ratio=1.01,
        horizontal_velocity=0.08,
        vertical_velocity=0.03,
        vertical_tracer_fraction=0.4,
    )
    mutable_inputs = values[4:]
    before = tuple(array.tobytes() for array in mutable_inputs)
    first = _evaluate(
        values,
        east_position=0.035,
        vertical_position=0.055,
        top_open=True,
    )
    middle = tuple(array.tobytes() for array in mutable_inputs)
    second = _evaluate(
        values,
        east_position=0.035,
        vertical_position=0.055,
        top_open=True,
    )
    after = tuple(array.tobytes() for array in mutable_inputs)

    assert before == middle == after
    for name in (
        "dMgt_dt", "dJgt_dt", "dMgr_dt", "dJgrs_dt", "dMgrs_dt",
        "dJgt_drag_dt", "dJgrs_drag_dt",
        "dQlt_drag_dt", "dQlr_drag_dt",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.top_flux == second.top_flux
    assert first.t_flux == second.t_flux
    assert first.east_front == second.east_front
    assert first.vertical_front == second.vertical_front


def test_interphase_drag_is_equal_opposite_for_both_branches() -> None:
    values = list(_state())
    params, dx, dz = values[:3]
    Mgt, Jgt, Mgr, Jgrs, Mgrs = values[4:9]
    Alt, Alr, Qlt, Qlr = values[9:13]
    Alt[:] = 0.55 * params.horizontal_area
    Alr[:] = 0.55 * params.vertical_area
    gas_area_h = params.horizontal_area - Alt
    gas_area_v = params.vertical_area - Alr
    Mgt[:] = params.rho_atmospheric * gas_area_h * dx
    Mgr[:] = params.rho_atmospheric * gas_area_v * dz
    Mgrs[:] = Mgr
    Jgt[:] = 0.45 * Mgt
    Jgrs[:] = -0.30 * Mgr
    Qlt[:] = -0.10 * Alt
    Qlr[:] = 0.12 * Alr

    result = _evaluate(
        tuple(values),
        east_position=0.05,
        vertical_position=0.09,
        top_open=False,
    )

    np.testing.assert_allclose(
        result.dJgt_drag_dt + params.rho_l * dx * result.dQlt_drag_dt,
        0.0,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        result.dJgrs_drag_dt + params.rho_l * dz * result.dQlr_drag_dt,
        0.0,
        atol=2.0e-18,
    )
    assert np.any(result.dQlt_drag_dt > 0.0)
    assert np.any(result.dQlr_drag_dt < 0.0)
    assert abs(result.horizontal_interphase_momentum_residual) < 2.0e-18
    assert abs(result.vertical_interphase_momentum_residual) < 2.0e-18


def test_elastic_liquid_overfill_is_zero_gas_void_not_invalid_geometry() -> None:
    values = list(_state())
    params = values[0]
    Mgt, Jgt, Mgr, Jgrs, Mgrs = values[4:9]
    Alt, Alr, Qlt, Qlr = values[9:13]
    Alt[0] = 1.001 * params.horizontal_area
    Alr[1] = 1.001 * params.vertical_area
    Mgt[0] = Jgt[0] = 0.0
    Mgr[1] = Jgrs[1] = Mgrs[1] = 0.0
    Qlt[0] = Qlr[1] = 0.0

    result = _evaluate(
        tuple(values),
        east_position=0.03,
        vertical_position=0.03,
        top_open=False,
    )

    assert not result.horizontal_active[0]
    assert not result.vertical_bottom_active[1]
    assert np.all(np.isfinite(result.dMgt_dt))
    assert np.all(np.isfinite(result.dMgr_dt))
