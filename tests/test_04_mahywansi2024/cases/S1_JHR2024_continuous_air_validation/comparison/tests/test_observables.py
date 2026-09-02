from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from observables import (
    EvidenceError,
    branch_status,
    compare_1d_to_2d,
    ensure_diagnostic_output_path,
    extract_2d_series,
    load_definitions,
    read_vtu,
    resample_rows,
    sha256_file,
    two_d_columns,
    write_csv,
)


HERE = Path(__file__).resolve().parents[1]
DEFINITIONS = HERE / "OBSERVABLE_DEFINITIONS.yaml"
THICKNESS = 0.0199491133502952


def _add_block(
    blocks: list[tuple[float, float, float, float]],
    x_edges: list[float],
    z_edges: list[float],
) -> None:
    for x0, x1 in zip(x_edges, x_edges[1:]):
        for z0, z1 in zip(z_edges, z_edges[1:]):
            blocks.append((x0, x1, z0, z1))


def _synthetic_cells() -> list[tuple[float, float, float, float]]:
    blocks: list[tuple[float, float, float, float]] = []
    _add_block(
        blocks,
        [-1.83, -1.5327, -1.5073, -1.0, -0.85, -0.80, -0.75, -0.10, -0.0127, 0.0127, 1.27],
        [-0.0127, 0.0, 0.0127],
    )
    _add_block(blocks, [-1.5327, -1.5073], [0.0127, 0.05, 0.10, 0.15])
    _add_block(blocks, [-0.0127, 0.0127], [0.0127, 0.30, 0.5842, 1.02])
    _add_block(blocks, [-0.10, -0.0127, 0.0127, 0.10], [1.02, 1.02635, 1.04, 1.10])
    return blocks


def _fields(
    cells: list[tuple[float, float, float, float]], *, erupt: bool, speed: float
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.ones(len(cells), dtype=float)
    velocity = np.zeros((len(cells), 3), dtype=float)
    for index, (x0, x1, z0, z1) in enumerate(cells):
        xmid = 0.5 * (x0 + x1)
        zmid = 0.5 * (z0 + z1)
        is_horizontal = -0.0127 <= z0 and z1 <= 0.0127
        is_supply = -1.5327 <= x0 and x1 <= -1.5073 and z0 >= 0.0127
        if is_supply:
            alpha[index] = 0.0
            velocity[index, 2] = -0.2
        if is_horizontal and (-1.5327 <= xmid < -0.85 or -0.75 < xmid <= -0.10):
            alpha[index] = 0.0
        if is_horizontal and -0.85 <= xmid <= -0.75:
            alpha[index] = 1.0
            velocity[index, 0] = speed
        if z0 >= 1.02:
            alpha[index] = 1.0 if erupt and abs(xmid) <= 0.0127 else 0.0
            velocity[index, 2] = 0.3 if alpha[index] > 0.5 else 0.0
        if 0.0127 <= zmid <= 1.02 and abs(xmid) <= 0.0127:
            alpha[index] = 1.0
            velocity[index, 2] = 0.1
    return alpha, velocity


def _write_vtu(
    path: Path,
    cells_xyz: list[tuple[float, float, float, float]],
    alpha: np.ndarray,
    velocity: np.ndarray,
) -> None:
    points: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}
    cells: list[list[int]] = []

    def point(value: tuple[float, float, float]) -> int:
        if value not in lookup:
            lookup[value] = len(points)
            points.append(value)
        return lookup[value]

    for x0, x1, z0, z1 in cells_xyz:
        cells.append(
            [
                point((x0, 0.0, z0)),
                point((x1, 0.0, z0)),
                point((x1, THICKNESS, z0)),
                point((x0, THICKNESS, z0)),
                point((x0, 0.0, z1)),
                point((x1, 0.0, z1)),
                point((x1, THICKNESS, z1)),
                point((x0, THICKNESS, z1)),
            ]
        )
    flat_points = " ".join(str(value) for item in points for value in item)
    connectivity = " ".join(str(value) for item in cells for value in item)
    offsets = " ".join(str(8 * (index + 1)) for index in range(len(cells)))
    types = " ".join("12" for _ in cells)
    alpha_text = " ".join(str(value) for value in alpha)
    velocity_text = " ".join(str(value) for row in velocity for value in row)
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">
<Points><DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">{flat_points}</DataArray></Points>
<Cells>
<DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray>
</Cells>
<CellData>
<DataArray type="Float64" Name="alpha.water" format="ascii">{alpha_text}</DataArray>
<DataArray type="Float64" Name="U" NumberOfComponents="3" format="ascii">{velocity_text}</DataArray>
</CellData>
</Piece></UnstructuredGrid></VTKFile>
""",
        encoding="utf-8",
    )


def _build_vtk_series(root: Path, *, erupt: bool = True) -> Path:
    root.mkdir(parents=True)
    cells = _synthetic_cells()
    entries = []
    for index, time_s in enumerate((10.0, 10.1, 10.2)):
        alpha, velocity = _fields(cells, erupt=erupt, speed=0.82 + 0.02 * index)
        vtu = root / f"internal_{index}.vtu"
        _write_vtu(vtu, cells, alpha, velocity)
        vtm = root / f"frame_{index}.vtm"
        vtm.write_text(
            f'<VTKFile><vtkMultiBlockDataSet><DataSet name="internal" file="{vtu.name}"/></vtkMultiBlockDataSet></VTKFile>\n',
            encoding="utf-8",
        )
        entries.append({"name": vtm.name, "time": time_s})
    series = root / "synthetic.vtm.series"
    series.write_text(json.dumps({"file-series-version": "1.0", "files": entries}), encoding="utf-8")
    return series


def _write_probes(case: Path) -> None:
    folder = case / "postProcessing" / "probesJHR" / "10"
    folder.mkdir(parents=True)
    locations = [(0, 0), (0, 0.3), (0, 0.45), (-0.8, 0), (-0.1, 0), (0.1, 0)]
    lines = [f"# Probe {index} ({x} {THICKNESS / 2} {z})" for index, (x, z) in enumerate(locations)]
    for step, time_s in enumerate((10.0, 10.05, 10.1, 10.15, 10.2)):
        values = [101325.0 + 100.0 * probe + step for probe in range(6)]
        lines.append(f"{time_s} " + " ".join(str(value) for value in values))
    (folder / "p").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile(path: Path, times: np.ndarray) -> None:
    z = np.array([0.1, 0.3, 0.5])
    shape = (len(times), len(z))
    np.savez(
        path,
        time_s=times,
        riser_z_cell_center_m=z,
        riser_Aup_m2=np.full(shape, 1.0e-4),
        riser_Qup_m3_s=np.full(shape, 2.0e-5),
        riser_Adown_m2=np.full(shape, 0.5e-4),
        riser_Qdown_m3_s=np.full(shape, 1.0e-5),
        riser_gas_area_m2=np.full(shape, 0.5e-4),
        riser_gas_mass_kg_m=np.full(shape, 1.0e-5),
        riser_gas_velocity_m_s=np.full(shape, 0.2),
    )


def _canonical_rows(
    defs,
    times: np.ndarray,
    *,
    event: bool,
    offset: float = 0.0,
    two_d_strict: bool = False,
):
    rows = []
    for index, time_s in enumerate(times):
        row = {name: 0.0 for name in defs.canonical}
        row["time_s"] = float(time_s)
        row["P1_gauge_Pa"] = 10.0 * time_s + offset
        row["P2_gauge_Pa"] = 20.0 * time_s + offset
        row["P3_gauge_Pa"] = 30.0 * time_s + offset
        row["horizontal_gas_nose_x_m"] = -1.0 + 0.1 * time_s + offset
        row["horizontal_gas_tail_x_m"] = -1.2 + 0.05 * time_s + offset
        row["horizontal_gas_centroid_x_m"] = -1.1 + 0.075 * time_s + offset
        row["horizontal_gas_volume_m3"] = 1e-5 + offset * 1e-6
        row["horizontal_slug_velocity_m_s"] = 0.85 + offset
        row["riser_connected_water_top_z_m"] = 1.02 + 0.1 * time_s + offset
        row["gas_arrival_at_riser"] = float(time_s >= 0.1)
        row["internal_mouth_event_active"] = float(event and 0.1 <= time_s <= 0.2)
        if two_d_strict:
            for unavailable_name in defs.data["observable_status"][
                "unavailable_from_alpha_water_and_U_only"
            ]:
                row[str(unavailable_name)] = math.nan
            active = bool(event and 0.1 <= time_s <= 0.2)
            minimum_volume = float(
                defs.result["hard_physics_gate"]["fixed_physical_event_test"]
                ["minimum_connected_water_volume_m3"]
            )
            volume = 2.0 * minimum_volume if active else 0.0
            top = 1.10 if active else math.nan
            row.update(
                {
                    "sample_index": index,
                    "target_time_s": float(time_s),
                    "actual_time_s": float(time_s),
                    "stage2_elapsed_s": float(time_s),
                    "launch_component_count": 1 if active else 0,
                    "component_water_volume_m3": volume,
                    "volume_over_minimum": volume / minimum_volume,
                    "bulk_top_q99_z_m": top,
                    "bulk_height_above_rim_m": top - 1.02 if active else math.nan,
                    "active_raw": float(active),
                    "active_persistent": float(active),
                    "water_weighted_uz_m_per_s": 0.3 if active else math.nan,
                }
            )
        rows.append(row)
    return rows


def _write_2d(
    path: Path,
    defs,
    times: np.ndarray,
    *,
    level: str,
    event: bool,
    offset: float = 0.0,
):
    rows = _canonical_rows(
        defs, times, event=event, offset=offset, two_d_strict=True
    )
    _write_2d_rows(path, defs, level, rows)


def _write_2d_rows(path: Path, defs, level: str, rows) -> None:
    write_csv(path, rows, two_d_columns(defs))
    metadata = {
        "case_id": defs.data["case_id"],
        "physical_condition_count": 1,
        "mesh_level": level,
        "time_origin": "stage_2_air_opening",
        "time_shift_applied_s": 0.0,
        "common_grid_s": defs.dt,
        "definition_sha256": sha256_file(defs.path),
        "result_acceptance_sha256": sha256_file(defs.result_path),
        "common_observables_sha256": sha256_file(defs.common_path),
        "csv_sha256": sha256_file(path),
        "result_marker_written": False,
        "vtk_sha256": {f"synthetic_{level}.vtu": "A" * 64},
    }
    path.with_name(path.name + ".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_source_semantics_and_ascii_synthetic_grid() -> None:
    defs = load_definitions(DEFINITIONS)
    semantic = defs.data["source_semantics"]["fig8_velocity"]
    assert semantic["target_m_s"] == [0.8, 0.9]
    assert semantic["quantity"].startswith("water_velocity_magnitude")
    assert "gas_nose_velocity" in semantic["forbidden_aliases"]


def test_end_to_end_2d_extraction_from_synthetic_vtk_and_probe_csv(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    case = tmp_path / "case"
    _write_probes(case)
    vtk = tmp_path / "vtk"
    _build_vtk_series(vtk)
    rows, meta = extract_2d_series(
        case_dir=case,
        vtk_dir=vtk,
        stage2_start_s=10.0,
        mesh_level="coarse",
        defs=defs,
    )
    assert [row["time_s"] for row in rows] == pytest.approx([0.0, 0.1, 0.2])
    assert rows[0]["P1_gauge_Pa"] == pytest.approx(0.0)
    assert rows[1]["P2_gauge_Pa"] == pytest.approx(102.0)
    assert rows[0]["horizontal_gas_nose_x_m"] == pytest.approx(-0.85)
    assert rows[0]["horizontal_slug_velocity_m_s"] == pytest.approx(0.82)
    assert rows[0]["horizontal_slug_nose_speed_m_s_proxy"] == pytest.approx(0.0)
    assert rows[0]["internal_mouth_event_active"] is True
    assert meta["time_shift_applied_s"] == 0.0
    assert meta["result_marker_written"] is False


def test_compare_csvs_reports_hard_branch_rmse_phase_and_mesh_spread(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.array([0.0, 0.1, 0.2])
    one = tmp_path / "one.csv"
    write_csv(one, _canonical_rows(defs, times, event=True), defs.canonical)
    meshes = {}
    for level, offset in (("coarse", 0.03), ("medium_refine", 0.01), ("refined", 0.0)):
        path = tmp_path / f"{level}.csv"
        _write_2d(path, defs, times, level=level, event=True, offset=offset)
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    result = compare_1d_to_2d(
        one_d_csv=one, one_d_profile_npz=profile, mesh_csvs=meshes, defs=defs
    )
    assert result["hard_eruption_gate"]["status"] == "PASS_ERUPTION_BRANCH_MATCH"
    metric = result["per_mesh"]["coarse"]["waveform_metrics"]["P1_gauge_Pa"]
    assert metric["status"] == "compared_at_zero_time_shift"
    assert metric["rmse"] == pytest.approx(0.03)
    assert metric["phase_peak_time_error_s_1d_minus_2d"] == pytest.approx(0.0)
    spread = result["mesh_spread_separate_from_1d_error"]["P1_gauge_Pa"]
    assert spread["maximum_pointwise_max_minus_min"] == pytest.approx(0.03)
    profile_check = result["one_d_profile_to_canonical_consistency"]
    assert profile_check["section_z_m"] == pytest.approx(0.30)
    assert profile_check["Qup_profile_vs_canonical_scalar"]["sample_count"] == 3
    assert result["result_marker_written"] is False


def test_complete_wrong_1d_eruption_branch_is_hard_failure(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.round(np.arange(0.0, 25.0 + 0.05, 0.1), 10)
    one = tmp_path / "one.csv"
    write_csv(one, _canonical_rows(defs, times, event=False), defs.canonical)
    meshes = {}
    for level in ("coarse", "medium_refine", "refined"):
        path = tmp_path / f"{level}.csv"
        _write_2d(
            path,
            defs,
            times,
            level=level,
            event=True,
            offset=0.001 * len(meshes),
        )
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    result = compare_1d_to_2d(
        one_d_csv=one, one_d_profile_npz=profile, mesh_csvs=meshes, defs=defs
    )
    assert result["hard_eruption_gate"]["status"] == "FAIL_PHYSICS_ALIGNMENT_ERUPTION_BRANCH"


def test_time_translation_is_rejected() -> None:
    rows = [{"time_s": 0.1, "x": 1.0}, {"time_s": 0.2, "x": 2.0}]
    with pytest.raises(EvidenceError, match="unshifted t=0"):
        resample_rows(rows, ["time_s", "x"], np.array([0.0, 0.1]), max_gap_s=0.1)


def test_no_acceptance_marker_name_is_an_output(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    assert "RESULT_ACCEPTED" not in two_d_columns(defs)
    assert defs.data["comparison"]["no_result_marker_written"] is True
    with pytest.raises(EvidenceError, match="cannot write solver acceptance markers"):
        ensure_diagnostic_output_path(HERE / "outputs" / "RESULT_ACCEPTED", HERE)


def test_2d_boolean_without_connected_volume_q99_evidence_is_rejected(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.array([0.0, 0.1, 0.2])
    one = tmp_path / "one.csv"
    write_csv(one, _canonical_rows(defs, times, event=True), defs.canonical)
    meshes = {}
    for index, level in enumerate(("coarse", "medium_refine", "refined")):
        path = tmp_path / f"{level}.csv"
        rows = _canonical_rows(
            defs, times, event=True, offset=0.001 * index, two_d_strict=True
        )
        if level == "refined":
            # Forge the canonical Boolean while removing the frozen volume predicate.
            rows[1]["component_water_volume_m3"] = 0.0
            rows[1]["volume_over_minimum"] = 0.0
        _write_2d_rows(path, defs, level, rows)
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    with pytest.raises(EvidenceError, match="active_raw is not the frozen"):
        compare_1d_to_2d(
            one_d_csv=one,
            one_d_profile_npz=profile,
            mesh_csvs=meshes,
            defs=defs,
        )


def test_single_sample_event_does_not_satisfy_0p10s_persistence() -> None:
    defs = load_definitions(DEFINITIONS)
    rows = [
        {"time_s": 0.0, "internal_mouth_event_active": 0.0},
        {"time_s": 0.1, "internal_mouth_event_active": 1.0},
        {"time_s": 0.2, "internal_mouth_event_active": 0.0},
    ]
    result = branch_status(rows, defs)
    assert result["eruption_decision"] is None
    assert result["discarded_short_event_count"] == 1


def test_2d_alpha_u_export_cannot_invent_gas_mass_flux(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.array([0.0, 0.1, 0.2])
    one = tmp_path / "one.csv"
    write_csv(one, _canonical_rows(defs, times, event=True), defs.canonical)
    meshes = {}
    for index, level in enumerate(("coarse", "medium_refine", "refined")):
        path = tmp_path / f"{level}.csv"
        rows = _canonical_rows(
            defs, times, event=True, offset=0.001 * index, two_d_strict=True
        )
        if level == "medium_refine":
            rows[0]["mouth_gas_outflow_kg_s"] = 1.0
        _write_2d_rows(path, defs, level, rows)
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    with pytest.raises(EvidenceError, match="invents unavailable observables"):
        compare_1d_to_2d(
            one_d_csv=one,
            one_d_profile_npz=profile,
            mesh_csvs=meshes,
            defs=defs,
        )


def test_no_event_is_inconclusive_before_25s_and_false_at_25s() -> None:
    defs = load_definitions(DEFINITIONS)
    early = [
        {"time_s": 0.0, "internal_mouth_event_active": 0.0},
        {"time_s": 24.9, "internal_mouth_event_active": 0.0},
    ]
    complete = [*early, {"time_s": 25.0, "internal_mouth_event_active": 0.0}]
    assert branch_status(early, defs)["eruption_decision"] is None
    assert branch_status(complete, defs)["eruption_decision"] is False


def test_three_mesh_labels_cannot_reuse_one_evidence_file(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.array([0.0, 0.1, 0.2])
    one = tmp_path / "one.csv"
    write_csv(one, _canonical_rows(defs, times, event=True), defs.canonical)
    meshes = {}
    for level in ("coarse", "medium_refine", "refined"):
        path = tmp_path / f"{level}.csv"
        _write_2d(path, defs, times, level=level, event=True)
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    with pytest.raises(EvidenceError, match="three distinct 2-D evidence files"):
        compare_1d_to_2d(
            one_d_csv=one,
            one_d_profile_npz=profile,
            mesh_csvs=meshes,
            defs=defs,
        )


def test_2d_external_height_uses_launch_connected_q99_not_riser_top(tmp_path: Path) -> None:
    defs = load_definitions(DEFINITIONS)
    times = np.array([0.0, 0.1, 0.2])
    one_rows = _canonical_rows(defs, times, event=True)
    for row in one_rows:
        row["riser_connected_water_top_z_m"] = 9.0
    one = tmp_path / "one.csv"
    write_csv(one, one_rows, defs.canonical)
    meshes = {}
    for index, level in enumerate(("coarse", "medium_refine", "refined")):
        path = tmp_path / f"{level}.csv"
        rows = _canonical_rows(
            defs, times, event=True, offset=0.001 * index, two_d_strict=True
        )
        for row in rows:
            row["riser_connected_water_top_z_m"] = 8.0
        _write_2d_rows(path, defs, level, rows)
        meshes[level] = path
    profile = tmp_path / "profiles.npz"
    _profile(profile, times)
    result = compare_1d_to_2d(
        one_d_csv=one, one_d_profile_npz=profile, mesh_csvs=meshes, defs=defs
    )
    assert (
        result["per_mesh"]["refined"]["two_d_signature"]
        ["maximum_external_launch_connected_bulk_q99_height_above_rim_m"]
        == pytest.approx(0.08)
    )
    assert (
        result["one_d_vs_published_targets"]["maximum_height_above_rim_vs_experiment_m"]
        ["status"]
        == "unavailable_without_resolved_external_free_surface_domain"
    )
