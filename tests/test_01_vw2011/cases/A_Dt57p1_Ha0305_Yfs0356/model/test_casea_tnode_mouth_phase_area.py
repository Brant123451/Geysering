"""Tests for target-free phase-area ownership at the Case-A T mouth."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_tnode_mouth_phase_area import (  # noqa: E402
    resolve_tnode_mouth_phase_areas,
)


def _partition(**overrides):
    inputs = {
        "resolved_upward_area": 0.73,
        "resolved_downward_area": 0.20,
        "horizontal_gas_opening_area": 0.80,
        "vertical_tracer_gas_mass": 0.0,
        "full_area": 1.0,
        "vertical_cell_length": 0.10,
        "reference_gas_density": 1.20,
        "topology_density_fraction": 0.02,
        "taylor_swept_fraction": 0.0875,
        "taylor_core_area_fraction": 0.80,
    }
    inputs.update(overrides)
    return resolve_tnode_mouth_phase_areas(**inputs)


def test_large_horizontal_opening_cannot_consume_unswept_vertical_liquid() -> None:
    result = _partition()

    assert result.resolved_vertical_void_area == pytest.approx(0.07)
    assert result.taylor_supported_vertical_void_area == pytest.approx(0.07)
    assert result.gas_area == pytest.approx(0.07)
    assert result.downward_area == pytest.approx(0.20)
    assert result.upward_area == pytest.approx(0.73)
    assert result.liquid_area + result.gas_area == pytest.approx(1.0)


def test_mass_supported_resolved_void_opens_without_taylor_cut() -> None:
    # Threshold = 0.02*1.2*0.15*0.1 = 3.6e-4 kg.
    result = _partition(
        resolved_upward_area=0.65,
        resolved_downward_area=0.20,
        vertical_tracer_gas_mass=4.0e-4,
        taylor_swept_fraction=0.0,
    )

    assert result.mass_supported_vertical_void_area == pytest.approx(0.15)
    assert result.vertical_material_gas_area == pytest.approx(0.15)
    assert result.gas_area == pytest.approx(0.15)


def test_massless_unswept_void_does_not_become_a_gas_aperture() -> None:
    result = _partition(
        resolved_upward_area=0.65,
        resolved_downward_area=0.20,
        vertical_tracer_gas_mass=0.0,
        taylor_swept_fraction=0.0,
    )

    assert result.resolved_vertical_void_area == pytest.approx(0.15)
    assert result.vertical_material_gas_area == 0.0
    assert result.gas_area == 0.0
    assert result.upward_area == pytest.approx(0.80)
    assert result.downward_area == pytest.approx(0.20)


def test_shared_gas_area_is_serial_overlap_not_union() -> None:
    result = _partition(
        resolved_upward_area=0.40,
        resolved_downward_area=0.20,
        horizontal_gas_opening_area=0.12,
        vertical_tracer_gas_mass=2.0e-3,
        taylor_swept_fraction=1.0,
    )

    assert result.vertical_material_gas_area == pytest.approx(0.40)
    assert result.gas_area == pytest.approx(0.12)
    assert result.upward_area == pytest.approx(0.68)
    assert result.downward_area == pytest.approx(0.20)
    assert result.partition_residual == pytest.approx(0.0)


def test_exact_case_a_scale_partition_preserves_resolved_liquid() -> None:
    diameter = 0.0571
    full_area = math.pi * diameter**2 / 4.0
    resolved_void = 1.81e-4
    film_area = 0.12 * full_area
    result = resolve_tnode_mouth_phase_areas(
        resolved_upward_area=full_area - resolved_void - film_area,
        resolved_downward_area=film_area,
        horizontal_gas_opening_area=0.80 * full_area,
        vertical_tracer_gas_mass=0.0,
        full_area=full_area,
        vertical_cell_length=0.02,
        reference_gas_density=1.20,
        topology_density_fraction=0.02,
        taylor_swept_fraction=(
            resolved_void / (0.80 * full_area)
        ),
        taylor_core_area_fraction=0.80,
    )

    assert result.gas_area == pytest.approx(resolved_void)
    assert result.upward_area + result.downward_area == pytest.approx(
        full_area - resolved_void
    )
    assert result.upward_area + result.downward_area + result.gas_area == pytest.approx(
        full_area
    )


def test_fully_swept_mouth_does_not_relabel_an_all_downward_cell() -> None:
    result = _partition(
        resolved_upward_area=0.0,
        resolved_downward_area=1.0,
        horizontal_gas_opening_area=0.0,
        vertical_tracer_gas_mass=0.0,
        taylor_swept_fraction=1.0,
        taylor_core_area_fraction=0.80,
    )

    assert result.gas_area == 0.0
    assert result.downward_area == pytest.approx(1.0)
    assert result.upward_area == pytest.approx(0.0)
    assert result.upward_area + result.downward_area == pytest.approx(1.0)


def test_unswept_mouth_does_not_relabel_a_resolved_downward_cell() -> None:
    result = _partition(
        resolved_upward_area=0.0,
        resolved_downward_area=1.0,
        horizontal_gas_opening_area=0.0,
        vertical_tracer_gas_mass=0.0,
        taylor_swept_fraction=0.0,
        taylor_core_area_fraction=0.80,
    )

    assert result.gas_area == 0.0
    assert result.downward_area == pytest.approx(1.0)
    assert result.upward_area == 0.0


def test_fully_swept_three_phase_trace_keeps_resolved_falling_area() -> None:
    result = _partition(
        resolved_upward_area=0.0,
        resolved_downward_area=0.70,
        horizontal_gas_opening_area=0.80,
        vertical_tracer_gas_mass=1.0,
        taylor_swept_fraction=1.0,
        taylor_core_area_fraction=0.80,
    )

    assert result.gas_area == pytest.approx(0.30)
    assert result.downward_area == pytest.approx(0.70)
    assert result.upward_area == pytest.approx(0.0)
    assert result.partition_residual == pytest.approx(0.0)


def test_almost_but_not_fully_swept_trace_keeps_prognostic_partition() -> None:
    result = _partition(
        resolved_upward_area=0.0,
        resolved_downward_area=0.70,
        horizontal_gas_opening_area=0.80,
        vertical_tracer_gas_mass=1.0,
        taylor_swept_fraction=1.0 - 1.0e-6,
        taylor_core_area_fraction=0.80,
    )

    assert result.gas_area == pytest.approx(0.30)
    assert result.downward_area == pytest.approx(0.70)
    assert result.upward_area == 0.0


def test_overpacked_two_stream_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="overpacks"):
        _partition(
            resolved_upward_area=0.81,
            resolved_downward_area=0.20,
        )


def test_helper_has_no_target_or_time_input() -> None:
    signature = inspect.signature(resolve_tnode_mouth_phase_areas)
    forbidden = {"time", "target", "reference_field", "desired_holdup"}
    assert forbidden.isdisjoint(signature.parameters)
