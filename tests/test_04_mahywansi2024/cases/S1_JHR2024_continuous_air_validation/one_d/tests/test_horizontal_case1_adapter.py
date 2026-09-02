import math

import numpy as np
import pytest

from model.errors import ContractViolation
from model.horizontal_case1_adapter import (
    CASE1_SEED_SHA256,
    FROZEN_2D_WATER_EOS_EVIDENCE,
    FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S,
    Case1HorizontalLiquidAdapter,
    ForbiddenCase1Topology,
    MahyawansiHorizontalGrid,
    build_s1_2d_eos_aligned_horizontal_adapter,
    verify_case1_seed_integrity,
)


def test_both_case1_seed_hashes_are_frozen_and_read_only_verified() -> None:
    assert verify_case1_seed_integrity() == dict(CASE1_SEED_SHA256)


def test_source_geometry_and_tee_face_indices_are_frozen() -> None:
    grid = MahyawansiHorizontalGrid()
    assert grid.diameter_m == pytest.approx(0.0254)
    assert (grid.x_left_m, grid.x_right_m) == pytest.approx((-1.83, 1.27))
    assert grid.cell_count == 310
    assert grid.air_tee_face_index == 31
    assert grid.riser_tee_face_index == 183
    assert grid.x_left_m + grid.air_tee_face_index * grid.dx_m == pytest.approx(-1.52)
    assert grid.x_left_m + grid.riser_tee_face_index * grid.dx_m == pytest.approx(0.0)
    assert len(grid.cell_lengths_m) == grid.cell_count


def test_geometry_drift_or_misaligned_tee_grid_fails_closed() -> None:
    with pytest.raises(ContractViolation, match="geometry drifted"):
        MahyawansiHorizontalGrid(diameter_m=0.026)
    with pytest.raises(ContractViolation, match="not located on a 1-D cell face"):
        MahyawansiHorizontalGrid(dx_m=0.007)


def test_stage1_horizontal_initial_state_is_full_water_and_at_rest() -> None:
    adapter = Case1HorizontalLiquidAdapter()
    state = adapter.build_stage1_initial_state()
    expected_area = math.pi * 0.0254**2 / 4.0
    assert state.cell_count == 310
    assert state.Al == pytest.approx((expected_area,) * 310)
    assert state.Ql == pytest.approx((0.0,) * 310)
    assert state.Mg == pytest.approx((0.0,) * 310)
    assert state.Jg == pytest.approx((0.0,) * 310)


def test_source_initial_head_maps_to_positive_elastic_storage_above_outlet_datum() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    state = adapter.build_stage1_initial_state(
        initial_piezometric_head_m=0.5842,
        elastic_storage_reference_head_m=0.584,
    )
    area = state.Al[0]
    assert area > adapter.full_area_m2
    represented = 0.584 + adapter.head_from_area_m(area) - adapter.grid.diameter_m
    assert represented == pytest.approx(0.5842, abs=1.0e-10)
    assert state.Al == pytest.approx((area,) * adapter.grid.cell_count)
    assert state.Ql == pytest.approx((0.0,) * adapter.grid.cell_count)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topology": "case1_finite_pocket_valve_release"},
        {"finite_gas_pocket_length_m": 0.5},
        {"valve_release_shockfit": True},
    ],
)
def test_case1_shockfit_and_finite_pocket_topologies_are_forbidden(kwargs) -> None:
    adapter = Case1HorizontalLiquidAdapter()
    with pytest.raises(ForbiddenCase1Topology):
        adapter.build_stage1_initial_state(**kwargs)


def test_case1_conservative_liquid_flux_contract() -> None:
    adapter = Case1HorizontalLiquidAdapter()
    area = adapter.full_area_m2
    flux = adapter.physical_flux(area, 0.0)
    assert flux.liquid_volume_m3_s == 0.0
    assert flux.liquid_momentum_m4_s2 > 0.0
    with pytest.raises(ContractViolation, match="dry horizontal state"):
        adapter.physical_flux(0.0, 1.0e-5)


def test_elastic_storage_maps_to_exact_case1_conservative_port_pressure() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    area = adapter.full_area_m2
    rho = 998.4
    epsilon = 1.0e-5
    full_increment = adapter.conservative_port_pressure_increment_Pa(area, rho)
    elastic_increment = adapter.conservative_port_pressure_increment_Pa(
        area * (1.0 + epsilon), rho
    )
    expected = 0.5 * rho * adapter.wave_speed_m_s**2 * (
        (1.0 + epsilon) ** 2 - 1.0
    )

    assert full_increment == pytest.approx(0.0, abs=1.0e-12)
    assert elastic_increment == pytest.approx(expected, rel=1.0e-11)
    with pytest.raises(ContractViolation, match="full/elastic"):
        adapter.conservative_port_pressure_increment_Pa(0.999 * area, rho)


def test_default_numerical_values_and_evidence_labels_are_exposed() -> None:
    adapter = Case1HorizontalLiquidAdapter()
    assert adapter.grid.dx_m == pytest.approx(0.01)
    assert adapter.gravity_m_s2 == pytest.approx(9.81)
    assert adapter.wave_speed_m_s == pytest.approx(100.0)
    records = adapter.parameter_provenance
    assert records["dx_m"].evidence == "declared_1D_grid"
    assert records["gravity_m_s2"].evidence == "Case1_inherited_physical_constant"
    assert (
        records["wave_speed_m_s"].evidence
        == "Case1_inherited_numerical_parameter"
    )
    assert not any(record.is_override for record in records.values())


def test_explicit_numerical_overrides_cannot_be_unrecorded() -> None:
    adapter = Case1HorizontalLiquidAdapter(
        MahyawansiHorizontalGrid(dx_m=0.005),
        gravity_m_s2=9.80665,
        wave_speed_m_s=80.0,
    )
    records = adapter.parameter_provenance
    assert records["dx_m"].value == pytest.approx(0.005)
    assert records["gravity_m_s2"].value == pytest.approx(9.80665)
    assert records["wave_speed_m_s"].value == pytest.approx(80.0)
    assert all(record.is_override for record in records.values())


def test_formal_s1_factory_reuses_case1_geometry_with_frozen_2d_eos_tangent() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    assert adapter.grid.cell_count == 310
    assert adapter.wave_speed_m_s == pytest.approx(math.sqrt(3000.0 * 293.15))
    assert adapter.wave_speed_m_s == pytest.approx(
        FROZEN_2D_WATER_TANGENT_WAVE_SPEED_M_S
    )
    record = adapter.parameter_provenance["wave_speed_m_s"]
    assert record.evidence == FROZEN_2D_WATER_EOS_EVIDENCE
    assert record.is_override is True


def test_segment_flux_calls_the_hash_pinned_case1_muscl_and_riemann_kernels(
    monkeypatch,
) -> None:
    adapter = Case1HorizontalLiquidAdapter()
    area = adapter.full_area_m2
    calls = {"muscl": 0, "riemann": 0, "draining": 0}
    seed = adapter._case1_seed
    original_muscl = seed._muscl_free_surface_face_states
    original_riemann = seed._central_upwind_flux
    original_draining = seed._apply_donor_draining_limiter

    def muscl(*args, **kwargs):
        calls["muscl"] += 1
        return original_muscl(*args, **kwargs)

    def riemann(*args, **kwargs):
        calls["riemann"] += 1
        return original_riemann(*args, **kwargs)

    def draining(*args, **kwargs):
        calls["draining"] += 1
        return original_draining(*args, **kwargs)

    monkeypatch.setattr(seed, "_muscl_free_surface_face_states", muscl)
    monkeypatch.setattr(seed, "_central_upwind_flux", riemann)
    monkeypatch.setattr(seed, "_apply_donor_draining_limiter", draining)
    raw = adapter.case1_muscl_central_upwind_face_fluxes(
        (area, area, area),
        (0.0, 0.0, 0.0),
        left_ghost=(area, 0.0),
        right_ghost=(area, 0.0),
    )
    limited = adapter.case1_donor_draining_limit(
        (area, area, area),
        raw.liquid_volume_m3_s,
        raw.liquid_momentum_m4_s2,
        dx_m=0.01,
        dt_s=1.0e-5,
    )

    assert calls == {"muscl": 1, "riemann": 1, "draining": 1}
    assert limited.liquid_volume_m3_s == pytest.approx((0.0,) * 4)
    assert np.all(np.isfinite(limited.liquid_momentum_m4_s2))


def test_dynamic_total_pressure_ghost_closes_head_and_changes_with_interior_velocity() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    area = adapter.full_area_m2
    common = dict(
        interior_area_m2=area,
        prescribed_total_head_m=0.586,
        reference_head_m=0.584,
        side="left",
    )
    at_rest = adapter.dynamic_total_pressure_ghost(
        interior_discharge_m3_s=0.0,
        **common,
    )
    moving = adapter.dynamic_total_pressure_ghost(
        interior_discharge_m3_s=0.05 * area,
        **common,
    )

    assert abs(at_rest.total_head_residual_m) < 2.0e-10
    assert abs(moving.total_head_residual_m) < 2.0e-10
    assert at_rest.liquid_discharge_m3_s > 0.0
    assert moving.liquid_area_m2 != pytest.approx(
        at_rest.liquid_area_m2, rel=0.0, abs=1.0e-13 * area
    )
    assert "dynamic_MUSCL_ghost" in at_rest.evidence_status


@pytest.mark.parametrize("side", ["left", "right"])
def test_total_pressure_equal_to_storage_datum_returns_static_full_ghost(side) -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    area = adapter.full_area_m2
    ghost = adapter.dynamic_total_pressure_ghost(
        interior_area_m2=area,
        interior_discharge_m3_s=0.0,
        prescribed_total_head_m=0.5842,
        reference_head_m=0.5842,
        side=side,
    )
    assert ghost.liquid_area_m2 == pytest.approx(area, rel=0.0, abs=2.0e-13 * area)
    assert ghost.liquid_discharge_m3_s == pytest.approx(0.0, abs=2.0e-12)


def test_static_pressure_outlet_fixes_head_without_adding_kinetic_head() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    area = adapter.full_area_m2
    common = dict(
        interior_area_m2=area,
        prescribed_static_head_m=0.584,
        reference_head_m=0.584,
        side="right",
    )
    at_rest = adapter.static_pressure_characteristic_ghost(
        interior_discharge_m3_s=0.0,
        **common,
    )
    moving = adapter.static_pressure_characteristic_ghost(
        interior_discharge_m3_s=0.08 * area,
        **common,
    )

    assert at_rest.liquid_area_m2 == pytest.approx(area)
    assert moving.liquid_area_m2 == pytest.approx(area)
    assert at_rest.piezometric_head_m == pytest.approx(0.584)
    assert moving.piezometric_head_m == pytest.approx(0.584)
    assert abs(at_rest.static_head_residual_m) < 2.0e-10
    assert abs(moving.static_head_residual_m) < 2.0e-10
    assert moving.liquid_discharge_m3_s == pytest.approx(0.08 * area)
    assert "pressure_outlet_static_head" in moving.evidence_status


def test_static_pressure_and_total_pressure_ghosts_are_not_interchangeable() -> None:
    adapter = build_s1_2d_eos_aligned_horizontal_adapter()
    area = adapter.full_area_m2
    discharge = 0.20 * area
    total = adapter.dynamic_total_pressure_ghost(
        interior_area_m2=area,
        interior_discharge_m3_s=discharge,
        prescribed_total_head_m=0.584,
        reference_head_m=0.584,
        side="right",
    )
    static = adapter.static_pressure_characteristic_ghost(
        interior_area_m2=area,
        interior_discharge_m3_s=discharge,
        prescribed_static_head_m=0.584,
        reference_head_m=0.584,
        side="right",
    )

    assert static.liquid_area_m2 == pytest.approx(area)
    assert total.liquid_area_m2 < area
    assert total.liquid_area_m2 != pytest.approx(
        static.liquid_area_m2, rel=0.0, abs=1.0e-13 * area
    )
