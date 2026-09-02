from __future__ import annotations

import base64
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("audit_physical_rim_outlet.py")
SPEC = importlib.util.spec_from_file_location("physical_rim_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _data_array(name: str, values: np.ndarray, components: int = 1) -> str:
    values = np.asarray(values)
    vtk_type = {
        np.dtype("float32"): "Float32",
        np.dtype("int32"): "Int32",
    }[values.dtype]
    payload = values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes()
    encoded = base64.b64encode(struct.pack("<Q", len(payload)) + payload).decode()
    component_text = "" if components == 1 else f" NumberOfComponents='{components}'"
    return (
        f"<DataArray type='{vtk_type}' Name='{name}'{component_text} "
        f"format='binary'>{encoded}</DataArray>"
    )


def _write_surface(
    root: Path,
    time_s: float,
    alpha: tuple[float, float],
    uz: tuple[float, float],
    *,
    include_u: bool = True,
) -> None:
    target = root / f"{time_s:g}" / "physicalRim.vtp"
    target.parent.mkdir(parents=True)
    # Two 1 m x 1 m opening faces at x=[1,2] and [2,3].
    points = np.array(
        [
            [1, 0, 1.8250001],
            [2, 0, 1.8250001],
            [2, 1, 1.8250001],
            [1, 1, 1.8250001],
            [2, 0, 1.8250001],
            [3, 0, 1.8250001],
            [3, 1, 1.8250001],
            [2, 1, 1.8250001],
        ],
        dtype=np.float32,
    )
    connectivity = np.arange(8, dtype=np.int32)
    offsets = np.array([4, 8], dtype=np.int32)
    velocity = np.zeros((2, 3), dtype=np.float32)
    velocity[:, 2] = uz
    u_array = _data_array("U", velocity, 3) if include_u else ""
    target.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<VTKFile type='PolyData' byte_order='LittleEndian' header_type='UInt64'>",
                "<PolyData>",
                f"<FieldData>{_data_array('TimeValue', np.array([time_s], dtype=np.float32))}</FieldData>",
                "<Piece NumberOfPoints='8' NumberOfPolys='2'>",
                f"<Points>{_data_array('Points', points, 3)}</Points>",
                "<Polys>",
                _data_array("connectivity", connectivity),
                _data_array("offsets", offsets),
                "</Polys>",
                "<CellData>",
                _data_array("alpha.water", np.array(alpha, dtype=np.float32)),
                u_array,
                "</CellData>",
                "</Piece></PolyData></VTKFile>",
            ]
        ),
        encoding="utf-8",
    )


def _make_inputs(tmp_path: Path, *, normal_end: bool = True) -> tuple[Path, Path, Path, Path]:
    config = tmp_path / "case_config.json"
    config.write_text(
        json.dumps(
            {
                "case_id": "synthetic",
                "paper_run": "synthetic",
                "physical_geometry_m": {
                    "tee_axis_x": 2.0,
                    "riser_rim_z": 1.825,
                },
                "planar_mapping": {
                    "physical_riser_diameter_m": 1.0,
                    "area_equivalent_riser_width_m": 2.0,
                    "extrusion_m": 1.0,
                    "formula": "synthetic area-equivalent map",
                },
                "mesh_m": {"riser_dx": 1.0, "external_dz": 0.5},
                "simulation": {"end_time_s": 2.0},
                # The auditor must ignore this deliberately contradictory label.
                "experiment": {"classification": "NO GEYSER"},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source"
    surfaces = tmp_path / "surfaces"
    for time_s in (0.0, 1.0, 2.0):
        (source / f"{time_s:g}").mkdir(parents=True)
    log = tmp_path / "log.solve"
    log.write_text(
        "Time = 0\nTime = 1\nTime = 2\n" + ("End\n" if normal_end else ""),
        encoding="utf-8",
    )
    return config, source, surfaces, log


def test_resolved_crossing_uses_interface_area_and_cell_volume(tmp_path: Path) -> None:
    config, source, surfaces, log = _make_inputs(tmp_path)
    _write_surface(surfaces, 0, (0, 0), (0, 0))
    _write_surface(surfaces, 1, (1, 0), (2, 0))
    _write_surface(surfaces, 2, (0, 0), (0, 0))

    report = AUDIT.audit(
        case_config_path=config,
        surface_root=surfaces,
        source_case=source,
        solver_log=log,
    )

    assert report["decision"]["classification"] == "GEYSER"
    assert report["decision"]["first_interface_supported_upward_crossing_time_s"] == 1
    assert report["decision"]["first_one_cell_volume_time_s"] == 1
    assert report["decision"]["experimental_label_used"] is False
    assert report["metrics"]["maximum_rim_plane_alpha"] == 1
    assert report["metrics"]["maximum_positive_alpha_weighted_flow_m3_s"] == 2
    assert report["metrics"]["cumulative_positive_liquid_volume_m3"] == 2


def test_dilute_positive_trace_is_not_resolved_crossing(tmp_path: Path) -> None:
    config, source, surfaces, log = _make_inputs(tmp_path)
    _write_surface(surfaces, 0, (0, 0), (0, 0))
    _write_surface(surfaces, 1, (1.0e-6, 0), (1, 0))
    _write_surface(surfaces, 2, (0, 0), (0, 0))

    report = AUDIT.audit(
        case_config_path=config,
        surface_root=surfaces,
        source_case=source,
        solver_log=log,
    )

    assert report["decision"]["classification"] == "NO_GEYSER"
    assert report["decision"]["resolved_crossing_gate_pass"] is False
    assert report["metrics"]["maximum_rim_plane_alpha"] < 0.5


def test_running_case_never_gets_final_classification(tmp_path: Path) -> None:
    config, source, surfaces, log = _make_inputs(tmp_path, normal_end=False)
    _write_surface(surfaces, 0, (0, 0), (0, 0))
    _write_surface(surfaces, 1, (1, 0), (2, 0))
    _write_surface(surfaces, 2, (0, 0), (0, 0))

    report = AUDIT.audit(
        case_config_path=config,
        surface_root=surfaces,
        source_case=source,
        solver_log=log,
    )

    assert report["decision"]["classification"].startswith("PROVISIONAL_")
    assert report["decision"]["final"] is False


def test_normal_early_end_cannot_establish_no_geyser(tmp_path: Path) -> None:
    config, source, surfaces, log = _make_inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["simulation"]["end_time_s"] = 3.0
    config.write_text(json.dumps(payload), encoding="utf-8")
    for time_s in (0, 1, 2):
        _write_surface(surfaces, time_s, (0, 0), (0, 0))

    report = AUDIT.audit(
        case_config_path=config,
        surface_root=surfaces,
        source_case=source,
        solver_log=log,
    )

    assert report["decision"]["classification"] == (
        "INCOMPLETE_NO_FINAL_CLASSIFICATION"
    )
    assert report["decision"]["final"] is False
    assert report["decision"]["declared_observation_end_reached"] is False


def test_missing_velocity_is_reported_not_inferred(tmp_path: Path) -> None:
    config, source, surfaces, log = _make_inputs(tmp_path)
    for time_s in (0, 1, 2):
        _write_surface(surfaces, time_s, (1, 1), (1, 1), include_u=False)

    report = AUDIT.audit(
        case_config_path=config,
        surface_root=surfaces,
        source_case=source,
        solver_log=log,
    )

    assert report["decision"]["classification"] == "INDETERMINATE_EVIDENCE_GAP"
    assert report["missing_metrics"] == ["U"]
    assert report["metrics"]["cumulative_positive_liquid_volume_m3"] is None


def test_trapfpe_startup_line_is_not_a_fatal_error(tmp_path: Path) -> None:
    log = tmp_path / "log.solve"
    log.write_text(
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
        "Time = 20\nEnd\n",
        encoding="utf-8",
    )
    status = AUDIT._parse_solver_log(log)
    assert status["normal_end"] is True
    assert status["fatal_error"] is False
