"""Extract raw Case-A acceptance profiles from existing OpenFOAM VTK fields.

This is a read-only diagnostic for the already-computed 2-D solution.  It
integrates ``alpha.water`` over each horizontal column, so the resulting
equivalent liquid depth is independent of render interpolation and contour
line styling.  The vertical liquid height is likewise obtained from liquid
volume per unit out-of-plane depth.  No 1-D result is read and no acceptance
threshold is fitted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VTK_ROOT = CASE_ROOT / "openfoam" / "2d" / "VTK_ACCEPTANCE_2D"
DEFAULT_OUTPUT = CASE_ROOT / "outputs" / "acceptance_reference_2d.json"

HORIZONTAL_DIAMETER = 0.094
TOWER_DIAMETER = 0.0571
TOWER_CENTRE_X = 3.516
TOWER_CROWN_Y = 0.5 * HORIZONTAL_DIAMETER


def _numeric_data_array(parent: ET.Element, name: str | None = None) -> np.ndarray:
    arrays = parent.findall("DataArray")
    if name is not None:
        arrays = [item for item in arrays if item.attrib.get("Name") == name]
    if len(arrays) != 1 or arrays[0].text is None:
        raise ValueError(f"expected one ASCII DataArray named {name!r}")
    if arrays[0].attrib.get("format") != "ascii":
        raise ValueError("acceptance extractor requires ASCII VTK output")
    return np.fromstring(arrays[0].text, sep=" ")


def _load_vtu(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError(f"missing VTK Piece in {path}")
    points_parent = piece.find("Points")
    cells_parent = piece.find("Cells")
    cell_data = piece.find("CellData")
    field_data = root.find("./UnstructuredGrid/FieldData")
    if any(
        item is None
        for item in (points_parent, cells_parent, cell_data, field_data)
    ):
        raise ValueError(f"incomplete VTK structure in {path}")

    points = _numeric_data_array(points_parent).reshape(-1, 3)
    connectivity = _numeric_data_array(cells_parent, "connectivity").astype(np.int64)
    offsets = _numeric_data_array(cells_parent, "offsets").astype(np.int64)
    alpha = _numeric_data_array(cell_data, "alpha.water")
    time_value = float(_numeric_data_array(field_data, "TimeValue")[0])
    if alpha.size != offsets.size:
        raise ValueError("cell alpha and connectivity offsets have different sizes")

    starts = np.r_[0, offsets[:-1]]
    centres = np.empty((offsets.size, 3), dtype=float)
    spans = np.empty((offsets.size, 3), dtype=float)
    volumes = np.empty(offsets.size, dtype=float)
    for index, (start, stop) in enumerate(zip(starts, offsets, strict=True)):
        vertices = points[connectivity[start:stop]]
        centres[index] = np.mean(vertices, axis=0)
        span = np.ptp(vertices, axis=0)
        spans[index] = span
        volumes[index] = float(np.prod(span))
    if np.any(volumes <= 0.0):
        raise ValueError("non-positive cell volume in exported 2-D mesh")
    return centres, spans, volumes, alpha, time_value


def _column_profile(
    centres: np.ndarray,
    volumes: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * HORIZONTAL_DIAMETER
    mask = (centres[:, 1] >= -half) & (centres[:, 1] <= half)
    x = centres[mask, 0]
    a = alpha[mask]
    v = volumes[mask]
    keys = np.round(x, decimals=8)
    unique = np.unique(keys)
    depth = np.empty_like(unique)
    for index, key in enumerate(unique):
        selected = keys == key
        # Dividing cell volume by its streamwise width is unnecessary for the
        # present structured mesh because all cells in one column share dx;
        # the volume-weighted fraction is exactly the cross-section average.
        fraction = float(np.sum(a[selected] * v[selected]) / np.sum(v[selected]))
        depth[index] = HORIZONTAL_DIAMETER * fraction
    return unique, depth


def _window_metrics(
    x: np.ndarray,
    gas_thickness: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, float | int]:
    selected = (x >= lower) & (x <= upper)
    values = gas_thickness[selected]
    if values.size == 0:
        raise ValueError(f"no horizontal columns in [{lower}, {upper}]")
    if values.size >= 3:
        first = np.diff(values)
        second = np.diff(values, n=2)
        total_variation = float(np.sum(np.abs(first)))
        curvature_rms = float(np.sqrt(np.mean(second**2)))
    else:
        total_variation = 0.0
        curvature_rms = 0.0
    return {
        "columns": int(values.size),
        "gas_thickness_min_m": float(np.min(values)),
        "gas_thickness_mean_m": float(np.mean(values)),
        "gas_thickness_max_m": float(np.max(values)),
        "gas_thickness_peak_to_peak_m": float(np.ptp(values)),
        "gas_thickness_total_variation_m": total_variation,
        "gas_thickness_second_difference_rms_m": curvature_rms,
    }


def _vertical_metrics(
    centres: np.ndarray,
    spans: np.ndarray,
    volumes: np.ndarray,
    alpha: np.ndarray,
) -> dict[str, float]:
    half_tower = 0.5 * TOWER_DIAMETER
    mask = (
        (centres[:, 0] >= TOWER_CENTRE_X - half_tower)
        & (centres[:, 0] <= TOWER_CENTRE_X + half_tower)
        & (centres[:, 1] >= TOWER_CROWN_Y)
    )
    if not np.any(mask):
        raise ValueError("tower cells were not found in the VTK mesh")
    # VTK is a thin 3-D extrusion of the 2-D plane.  Divide by the measured
    # tower width and the resolved extrusion thickness to obtain liquid height.
    extrusion = float(np.median(spans[mask, 2]))
    if extrusion <= 0.0:
        raise ValueError("non-positive extrusion thickness in tower mesh")
    liquid_volume = float(np.sum(alpha[mask] * volumes[mask]))
    equivalent_height = liquid_volume / (TOWER_DIAMETER * extrusion)
    return {
        "liquid_volume_per_extrusion_m2": float(liquid_volume / extrusion),
        "equivalent_liquid_height_above_crown_m": float(equivalent_height),
        "maximum_cell_alpha": float(np.max(alpha[mask])),
        "minimum_cell_alpha": float(np.min(alpha[mask])),
        "derived_extrusion_thickness_m": float(extrusion),
    }


def extract(vtk_root: Path) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for path in sorted(vtk_root.glob("2d_*/internal.vtu")):
        centres, spans, volumes, alpha, time_value = _load_vtu(path)
        x, liquid_depth = _column_profile(centres, volumes, alpha)
        gas = HORIZONTAL_DIAMETER - liquid_depth
        frames.append(
            {
                "time_s": time_value,
                "source": str(path.resolve()),
                "horizontal": {
                    "x_m": x.tolist(),
                    "equivalent_liquid_depth_m": liquid_depth.tolist(),
                    "equivalent_gas_thickness_m": gas.tolist(),
                    "whole_pipe": _window_metrics(x, gas, 0.0, 4.006),
                    "t_left_window": _window_metrics(x, gas, 2.4, 3.48745),
                    "t_neighbourhood": _window_metrics(x, gas, 3.0, 3.65),
                    "far_west": _window_metrics(x, gas, 0.0, 2.4),
                },
                "vertical": _vertical_metrics(
                    centres, spans, volumes, alpha
                ),
            }
        )
    if not frames:
        raise FileNotFoundError(f"no VTK frames found below {vtk_root}")
    frames.sort(key=lambda item: float(item["time_s"]))
    return {
        "provenance": {
            "kind": "raw_existing_openfoam_2d_cell_field",
            "field": "alpha.water",
            "one_dimensional_results_used": False,
            "rendered_images_used": False,
            "horizontal_diameter_m": HORIZONTAL_DIAMETER,
            "tower_diameter_m": TOWER_DIAMETER,
            "tower_centre_x_m": TOWER_CENTRE_X,
            "tower_crown_y_m": TOWER_CROWN_Y,
        },
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtk-root", type=Path, default=DEFAULT_VTK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = extract(args.vtk_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
