import math
import shutil

import pytest

from model.errors import MissingPhysicalClosure
from model.vertical_case1_adapter import (
    ATMOSPHERIC_PRESSURE_PA,
    Case1PinMismatch,
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    FORBIDDEN_CASE1_TRANSPLANTS,
    INITIAL_AIR_TEMPERATURE_K,
    PINNED_SHA256,
    build_s1_vertical_component,
    default_case1_model_dir,
    verify_case1_vertical_pins,
)


def test_reviewed_case1_hashes_and_local_readiness_are_pinned() -> None:
    contract = verify_case1_vertical_pins()
    assert dict(contract.actual_sha256) == PINNED_SHA256
    assert contract.fv_core_ready is True
    assert contract.post_event_closures_ready is True
    assert contract.complete_riser_ready is False
    assert contract.production_ready is False
    assert contract.case1_missing_physical_closures


def test_any_pinned_source_change_fails_closed(tmp_path) -> None:
    source = default_case1_model_dir()
    for filename in PINNED_SHA256:
        shutil.copy2(source / filename, tmp_path / filename)
    changed = tmp_path / "casea_vertical_twostream_fv.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed\n")
    with pytest.raises(Case1PinMismatch, match="hash mismatch"):
        verify_case1_vertical_pins(tmp_path)


def test_s1_grid_and_initial_phase_volume_are_source_aligned() -> None:
    adapter = build_s1_vertical_component(cell_count=160)
    initial = adapter.initial
    area = math.pi * 0.0254**2 / 4.0

    assert adapter.diameter_m == pytest.approx(0.0254)
    assert initial.z_edges_m[0] == pytest.approx(0.0)
    assert initial.z_edges_m[-1] == pytest.approx(1.02)
    assert adapter.initial_water_level_m == pytest.approx(0.5842)
    assert abs(initial.water_volume_error_m3) <= 2.0e-18
    assert initial.target_water_volume_m3 == pytest.approx(area * 0.5842)

    state = initial.two_stream_state
    own = initial.own_state
    for index, (z0, z1) in enumerate(zip(initial.z_edges_m[:-1], initial.z_edges_m[1:])):
        if z1 <= 0.5842:
            assert state.upward_area[index] == pytest.approx(area)
            assert initial.gas_area_m2[index] == pytest.approx(0.0)
        elif z0 >= 0.5842:
            assert state.upward_area[index] == pytest.approx(0.0)
            assert initial.gas_area_m2[index] == pytest.approx(area)
    assert all(value == 0.0 for value in state.upward_discharge)
    assert all(value == 0.0 for value in state.downward_area)
    assert all(value == 0.0 for value in state.downward_discharge)
    assert own.Aup == pytest.approx(state.upward_area)
    assert own.Adown == pytest.approx(state.downward_area)
    assert own.Qup == pytest.approx(state.upward_discharge)
    assert own.Qdown == pytest.approx(tuple(-q for q in state.downward_discharge))
    assert all(value >= 0.0 for value in own.Qdown)
    assert own.Mg == pytest.approx(
        tuple(initial.air_density_kg_m3 * gas_area for gas_area in initial.gas_area_m2)
    )
    assert own.Jg == pytest.approx((0.0,) * adapter.cell_count)

    expected_density = (
        ATMOSPHERIC_PRESSURE_PA
        / (DRY_AIR_GAS_CONSTANT_J_KG_K * INITIAL_AIR_TEMPERATURE_K)
    )
    expected_gas_mass = expected_density * area * (1.02 - 0.5842)
    assert initial.air_density_kg_m3 == pytest.approx(expected_density)
    assert initial.represented_gas_mass_kg == pytest.approx(expected_gas_mass)


def test_two_directional_streams_remain_independent_when_net_flow_is_zero() -> None:
    adapter = build_s1_vertical_component(cell_count=8)
    state_type = type(adapter.initial.two_stream_state)
    area = math.pi * adapter.diameter_m**2 / 4.0
    state = state_type.from_iterables(
        upward_area=(0.40 * area,) * 8,
        upward_discharge=(1.0e-5,) * 8,
        downward_area=(0.20 * area,) * 8,
        downward_discharge=(-1.0e-5,) * 8,
    )
    assert state.liquid_discharge == pytest.approx((0.0,) * 8)
    assert state.gross_upward_flow == pytest.approx((1.0e-5,) * 8)
    assert state.gross_downward_flow == pytest.approx((1.0e-5,) * 8)
    assert state.upward_discharge != state.downward_discharge

    own = adapter.component_to_own_state(
        state,
        gas_mass_per_length_kg_m=(0.0,) * 8,
    )
    assert own.Qup == pytest.approx((1.0e-5,) * 8)
    assert own.Qdown == pytest.approx((1.0e-5,) * 8)
    assert own.net_liquid_discharge == pytest.approx((0.0,) * 8)
    # Case-1 uses a signed downward coordinate component; S1 stores its
    # non-negative gross magnitude.  Neither is reconstructed from the net.
    assert state.downward_discharge == pytest.approx(tuple(-q for q in own.Qdown))


def test_production_trajectory_and_case1_specific_closure_transplants_are_blocked() -> None:
    adapter = build_s1_vertical_component()
    assert adapter.production_ready is False
    assert adapter.pin.complete_riser_ready is False
    assert adapter.missing_physical_closures
    assert FORBIDDEN_CASE1_TRANSPLANTS == adapter.forbidden_transplants
    assert not hasattr(adapter, "alpha_core")
    assert not hasattr(adapter, "taylor_rise_speed")
    assert not hasattr(adapter, "wallis")
    with pytest.raises(MissingPhysicalClosure, match="production trajectory is forbidden"):
        adapter.require_production_trajectory()
