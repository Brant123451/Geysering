"""Conservation tests for the passive two-stream liquid-origin tracer."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_vertical_twostream_fv import (  # noqa: E402
    DirectionalBoundaryFlux,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamLiquidProvenanceState,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    advance_vertical_two_stream_fv,
    advance_vertical_two_stream_liquid_provenance,
    conservative_directional_topology_transfer,
    conservative_liquid_provenance_topology_transfer,
    hydrostatic_face_pressures,
)


def _parameters(*, cells: int, dz: float = 0.10) -> VerticalTwoStreamParameters:
    return VerticalTwoStreamParameters(
        cell_count=cells,
        cell_length=dz,
        diameter=0.094,
        liquid_density=998.0,
        gravity=9.81,
    )


def _hydrostatic(parameters: VerticalTwoStreamParameters) -> tuple[float, ...]:
    return hydrostatic_face_pressures(parameters, bottom_pressure=130_000.0)


def test_zero_initial_origin_map_does_not_create_horizontal_pipe_water() -> None:
    parameters = _parameters(cells=3)
    hydraulic_state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3, 1.0e-3, 1.0e-3],
        upward_discharge=[1.0e-4, 1.0e-4, 1.0e-4],
        downward_area=[0.0, 0.0, 0.0],
        downward_discharge=[0.0, 0.0, 0.0],
    )
    provenance = VerticalTwoStreamLiquidProvenanceState.initial_riser_water(
        hydraulic_state
    )
    dt = 0.01
    hydraulic_step = advance_vertical_two_stream_fv(
        hydraulic_state,
        parameters,
        dt=dt,
        pressure_faces=_hydrostatic(parameters),
    )

    result = advance_vertical_two_stream_liquid_provenance(
        provenance,
        hydraulic_state,
        hydraulic_step,
        parameters,
        dt=dt,
    )

    assert result.state.upward_source1_area == (0.0, 0.0, 0.0)
    assert result.state.downward_source1_area == (0.0, 0.0, 0.0)
    assert result.ledger.initial_source1_volume == 0.0
    assert result.ledger.final_source1_volume == 0.0
    assert result.ledger.boundary_source1_volume_change == 0.0
    assert result.ledger.source1_volume_residual == 0.0


def test_bottom_upward_boundary_injects_source_one_water_conservatively() -> None:
    parameters = _parameters(cells=2)
    hydraulic_state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3, 1.0e-3],
        upward_discharge=[0.0, 0.0],
        downward_area=[0.0, 0.0],
        downward_discharge=[0.0, 0.0],
    )
    provenance = VerticalTwoStreamLiquidProvenanceState.initial_riser_water(
        hydraulic_state
    )
    dt = 0.01
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=0.10,
        )
    )
    hydraulic_step = advance_vertical_two_stream_fv(
        hydraulic_state,
        parameters,
        dt=dt,
        pressure_faces=_hydrostatic(parameters),
        boundaries=boundaries,
    )

    result = advance_vertical_two_stream_liquid_provenance(
        provenance,
        hydraulic_state,
        hydraulic_step,
        parameters,
        dt=dt,
    )

    accepted_inflow = hydraulic_step.upward_area_flux[0]
    expected_area = dt / parameters.cell_length * accepted_inflow
    expected_volume = dt * accepted_inflow
    assert result.state.upward_source1_area == pytest.approx(
        [expected_area, 0.0]
    )
    assert result.state.downward_source1_area == (0.0, 0.0)
    assert result.upward_source1_area_flux[0] == pytest.approx(accepted_inflow)
    assert result.ledger.final_source1_volume == pytest.approx(expected_volume)
    assert result.ledger.boundary_source1_volume_change == pytest.approx(
        expected_volume
    )
    assert result.ledger.source1_volume_residual == pytest.approx(
        0.0,
        abs=2.0e-20,
    )


def test_topology_merge_conserves_source_one_inventory() -> None:
    hydraulic_transfer = conservative_directional_topology_transfer(
        upward_area=[1.0e-3],
        upward_discharge=[0.0],
        downward_area=[2.0e-3],
        downward_discharge=[-2.0e-4],
    )
    provenance = VerticalTwoStreamLiquidProvenanceState.from_iterables(
        upward_source1_area=[0.7e-3],
        downward_source1_area=[0.2e-3],
    )

    result = conservative_liquid_provenance_topology_transfer(
        provenance,
        hydraulic_transfer,
    )

    assert result.state.upward_source1_area == (0.0,)
    assert result.state.downward_source1_area == pytest.approx([0.9e-3])
    assert result.upward_source1_area_transfer == pytest.approx([-0.7e-3])
    assert result.downward_source1_area_transfer == pytest.approx([0.7e-3])
    assert result.source1_area_residual == pytest.approx(0.0, abs=2.0e-18)


def test_crossed_topology_swaps_complete_source_inventories() -> None:
    hydraulic_transfer = conservative_directional_topology_transfer(
        upward_area=[0.7e-3],
        upward_discharge=[-1.4e-4],
        downward_area=[1.1e-3],
        downward_discharge=[2.2e-4],
    )
    provenance = VerticalTwoStreamLiquidProvenanceState.from_iterables(
        upward_source1_area=[0.6e-3],
        downward_source1_area=[0.1e-3],
    )

    result = conservative_liquid_provenance_topology_transfer(
        provenance,
        hydraulic_transfer,
    )

    assert result.state.upward_source1_area == pytest.approx([0.1e-3])
    assert result.state.downward_source1_area == pytest.approx([0.6e-3])
    assert sum(result.state.source1_area) == pytest.approx(0.7e-3)
    assert result.source1_area_residual == pytest.approx(0.0, abs=2.0e-18)
