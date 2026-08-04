#!/usr/bin/env python3
"""Render the Case B external-plume OpenFOAM result for 1D--2D review.

The solver keeps the area-equivalent planar tower width
``W = Dt**2 / D``.  For display only, this script maps that slit to the
physical tower diameter ``Dt`` and applies a continuous piecewise map to the
external-air box so that the physical mouth and the computed plume meet
without a visual offset.  The outer edges of the external box are unchanged.

The physical tower walls stop at the experimental rim (y = 0.657 m).  No
wall, cap, dashed extension, or artificial duct is drawn above the rim.
External-air cells are hidden and only liquid with alpha.water >= 0.01 is
shown there, making lateral plume expansion visible against a white field.

This is a post-processing script: it neither changes nor runs the solver.
It expects ASCII ``foamToVTK`` output in ``openfoam/2d_external_plume/VTK``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_1D_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
SOURCE_1D_META = SOURCE_1D_ROOT / "frames_1d_caseB_tosan2021_meta.json"
OPENFOAM_2D = CASE_ROOT / "openfoam" / "2d_external_plume"
OUTPUT_ROOT = OPENFOAM_2D / "outputs_1d2d_compare"
FRAME_DIR = OUTPUT_ROOT / "frames_2d_external_plume"
MANIFEST = OUTPUT_ROOT / "frames_index_external_plume.json"
METADATA = OUTPUT_ROOT / "frames_2d_external_plume_meta.json"
SOURCE_RENDERER = OPENFOAM_2D / "_local_render_2d_frames.py"
VTK_DIR = OPENFOAM_2D / "VTK"

D = 0.094
DT = 0.0127
SLIT_WIDTH = DT * DT / D
X_TOWER = 3.516
X_VALVE = 0.546
X_END = 4.006
PIPE_BOTTOM = -0.5 * D
PIPE_CROWN = 0.5 * D
RIM_Y = PIPE_CROWN + 0.610
EXTERNAL_LEFT = 3.396
EXTERNAL_RIGHT = 3.636
EXTERNAL_TOP = 1.257
DISPLAY_TOP = 1.0
VISIBLE_PLUME_ALPHA = 0.01

WATER = "#2f7ff7"
AIR = "#f1f3f6"
WALL = "#333333"


def _load_vtu_reader() -> Callable[[Path], tuple[np.ndarray, np.ndarray]]:
    """Load the established ASCII-VTU cell-centre/alpha parser."""
    if not SOURCE_RENDERER.exists():
        raise FileNotFoundError(f"Missing source VTK parser: {SOURCE_RENDERER}")
    spec = importlib.util.spec_from_file_location(
        "caseb_external_plume_vtk_parser",
        SOURCE_RENDERER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SOURCE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_vtu_cell_centers_alpha


def _parse_series() -> list[tuple[float, Path]]:
    """Discover the VTM series without assuming the old case name ``2d``."""
    preferred = VTK_DIR / "2d_external_plume.vtm.series"
    if preferred.exists():
        series_path = preferred
    else:
        candidates = sorted(VTK_DIR.glob("*.vtm.series"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one VTM series in {VTK_DIR}; found {len(candidates)}"
            )
        series_path = candidates[0]

    data = json.loads(series_path.read_text(encoding="utf-8"))
    frames = []
    for item in data.get("files", []):
        source_time = float(item["time"])
        stem = Path(item["name"]).stem
        vtu = VTK_DIR / stem / "internal.vtu"
        if vtu.exists():
            frames.append((source_time, vtu))
    return frames


def _centres_to_edges(values: np.ndarray) -> np.ndarray:
    """Infer ordered cell edges from one-dimensional cell centres."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cell-centre coordinates must be a non-empty 1-D array")
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def _structured_cell_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place rectilinear cell data in a pcolormesh-compatible grid."""
    xr = np.round(np.asarray(x, dtype=float), 12)
    yr = np.round(np.asarray(y, dtype=float), 12)
    xu = np.unique(xr)
    yu = np.unique(yr)
    ix = np.searchsorted(xu, xr)
    iy = np.searchsorted(yu, yr)
    field = np.full((yu.size, xu.size), np.nan, dtype=float)
    field[iy, ix] = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return _centres_to_edges(xu), _centres_to_edges(yu), field


def _map_external_x(x: np.ndarray) -> np.ndarray:
    """Map the planar slit to Dt while fixing the exterior-box outer edges."""
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values)
    slit_left = X_TOWER - 0.5 * SLIT_WIDTH
    slit_right = X_TOWER + 0.5 * SLIT_WIDTH
    physical_left = X_TOWER - 0.5 * DT
    physical_right = X_TOWER + 0.5 * DT

    left = values < slit_left
    centre = (values >= slit_left) & (values <= slit_right)
    right = values > slit_right

    out[left] = EXTERNAL_LEFT + (
        (values[left] - EXTERNAL_LEFT)
        * (physical_left - EXTERNAL_LEFT)
        / (slit_left - EXTERNAL_LEFT)
    )
    out[centre] = X_TOWER + (values[centre] - X_TOWER) * (DT / SLIT_WIDTH)
    out[right] = physical_right + (
        (values[right] - slit_right)
        * (EXTERNAL_RIGHT - physical_right)
        / (EXTERNAL_RIGHT - slit_right)
    )
    return out


def _case_relative(path: Path) -> str:
    return str(path.relative_to(CASE_ROOT)).replace("\\", "/")


def _render_frame(
    *,
    index: int,
    target_time: float,
    source_time: float,
    vtu: Path,
    output: Path,
    read_vtu: Callable[[Path], tuple[np.ndarray, np.ndarray]],
    dpi: int,
) -> dict:
    centres, alpha = read_vtu(vtu)
    x = centres[:, 0]
    y = centres[:, 1]

    horizontal = (
        (x >= -1.0e-6)
        & (x <= X_END + 1.0e-6)
        & (y >= PIPE_BOTTOM - 1.0e-6)
        & (y <= PIPE_CROWN + 1.0e-10)
    )
    physical_tower = (
        (np.abs(x - X_TOWER) <= 0.5 * SLIT_WIDTH + 1.0e-8)
        & (y > PIPE_CROWN)
        & (y < RIM_Y - 1.0e-10)
    )
    external = (
        (x >= EXTERNAL_LEFT - 1.0e-8)
        & (x <= EXTERNAL_RIGHT + 1.0e-8)
        & (y >= RIM_Y - 1.0e-10)
        & (y <= EXTERNAL_TOP + 1.0e-8)
    )
    if not np.any(horizontal):
        raise RuntimeError(f"No horizontal-pipe cells found in {vtu}")
    if not np.any(physical_tower):
        raise RuntimeError(f"No physical-tower cells found in {vtu}")
    if not np.any(external):
        raise RuntimeError(f"No external-air cells found in {vtu}")

    hx, hy, ha = _structured_cell_field(x[horizontal], y[horizontal], alpha[horizontal])
    tx, ty, ta = _structured_cell_field(
        x[physical_tower],
        y[physical_tower],
        alpha[physical_tower],
    )
    ex, ey, ea = _structured_cell_field(x[external], y[external], alpha[external])

    tx = X_TOWER + (tx - X_TOWER) * (DT / SLIT_WIDTH)
    ex = _map_external_x(ex)
    external_water = np.ma.masked_where(
        np.isnan(ea) | (np.nan_to_num(ea, nan=0.0) < VISIBLE_PLUME_ALPHA),
        ea,
    )

    cmap = LinearSegmentedColormap.from_list(
        "caseb_external_air_water",
        [AIR, "#9ec5fe", WATER],
        N=256,
    )
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(12.4, 3.2))
        mesh_kw = {
            "cmap": cmap,
            "vmin": 0.0,
            "vmax": 1.0,
            "shading": "flat",
            "linewidth": 0.0,
            "rasterized": True,
            "zorder": 1,
        }
        ax.pcolormesh(hx, hy, ha, **mesh_kw)
        ax.pcolormesh(tx, ty, ta, **mesh_kw)
        ax.pcolormesh(ex, ey, external_water, **mesh_kw)

        tower_left = X_TOWER - 0.5 * DT
        tower_right = X_TOWER + 0.5 * DT
        line_kw = {"color": WALL, "linewidth": 0.8, "zorder": 3}

        # Connected pipe and tower.  The two tower walls end at the open rim;
        # deliberately draw no top cap, dashed rim, or extension above it.
        ax.plot([0.0, X_END], [PIPE_BOTTOM, PIPE_BOTTOM], **line_kw)
        ax.plot([0.0, 0.0], [PIPE_BOTTOM, PIPE_CROWN], **line_kw)
        ax.plot([X_END, X_END], [PIPE_BOTTOM, PIPE_CROWN], **line_kw)
        ax.plot([0.0, tower_left], [PIPE_CROWN, PIPE_CROWN], **line_kw)
        ax.plot([tower_right, X_END], [PIPE_CROWN, PIPE_CROWN], **line_kw)
        ax.plot([tower_left, tower_left], [PIPE_CROWN, RIM_Y], **line_kw)
        ax.plot([tower_right, tower_right], [PIPE_CROWN, RIM_Y], **line_kw)

        ax.axvline(
            X_VALVE,
            ymin=0.035,
            ymax=0.14,
            color="#202020",
            linestyle=":",
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            0.015,
            0.95,
            f"Time = {source_time:.2f} s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )
        ax.text(
            0.015,
            0.86,
            "OpenFOAM 2D  |  physical open rim + external atmosphere",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            X_VALVE,
            PIPE_CROWN + 0.025,
            "valve",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set_xlim(-0.04, 4.046)
        ax.set_ylim(PIPE_BOTTOM - 0.035, DISPLAY_TOP)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("vertical coordinate [m]")
        ax.legend(
            handles=[
                Patch(facecolor=WATER, label="water"),
                Patch(facecolor=AIR, edgecolor="#555555", label="air"),
            ],
            loc="upper right",
            frameon=False,
            prop={"family": "Times New Roman", "size": 9},
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.94)
        fig.savefig(output, dpi=dpi)
        plt.close(fig)

    return {
        "index": index,
        "target_time": float(target_time),
        "source_time": float(source_time),
        "time_offset": float(source_time - target_time),
        "file2d": _case_relative(output),
        "source_vtu": str(vtu.relative_to(OPENFOAM_2D)).replace("\\", "/"),
    }


def main(*, dpi: int = 140, max_time_offset: float = 0.031) -> None:
    read_vtu = _load_vtu_reader()
    series = _parse_series()
    if not series:
        raise RuntimeError(
            "No readable VTK series found. Reconstruct the selected times and "
            "run foamToVTK -ascii in openfoam/2d_external_plume first."
        )

    frames_1d = json.loads(SOURCE_1D_META.read_text(encoding="utf-8"))
    if not isinstance(frames_1d, list) or not frames_1d:
        raise ValueError(f"Expected a non-empty 1-D frame list in {SOURCE_1D_META}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in FRAME_DIR.glob("frame_*.png"):
        old_frame.unlink()

    paired_frames: list[dict] = []
    rendered_frames: list[dict] = []
    for index, frame_1d in enumerate(frames_1d):
        target_time = float(frame_1d["time"])
        source_time, vtu = min(series, key=lambda item: abs(item[0] - target_time))
        time_offset = abs(source_time - target_time)
        if time_offset > max_time_offset:
            raise RuntimeError(
                f"Nearest 2-D VTK time to {target_time:.6g} s is "
                f"{source_time:.6g} s (offset {time_offset:.6g} s), exceeding "
                f"the permitted {max_time_offset:.6g} s."
            )
        output = FRAME_DIR / f"frame_{index:04d}.png"
        frame_2d = _render_frame(
            index=index,
            target_time=target_time,
            source_time=source_time,
            vtu=vtu,
            output=output,
            read_vtu=read_vtu,
            dpi=dpi,
        )
        rendered_frames.append(frame_2d)

        pair = dict(frame_1d)
        pair["time"] = target_time
        pair["file1d"] = _case_relative(
            SOURCE_1D_ROOT / str(frame_1d["file"])
        )
        pair["file2d"] = frame_2d["file2d"]
        pair["sourceTime2d"] = source_time
        pair["dtMatch2d"] = float(source_time - target_time)
        paired_frames.append(pair)
        print(
            f"rendered {index + 1}/{len(frames_1d)}: "
            f"target={target_time:.2f} s, VTK={source_time:.2f} s",
            flush=True,
        )

    MANIFEST.write_text(
        json.dumps(paired_frames, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    METADATA.write_text(
        json.dumps(
            {
                "description": (
                    "Case B external-plume display. The physical tower walls "
                    "end at y=0.657 m and discharged liquid can expand in the "
                    "external atmospheric box."
                ),
                "scientific_role": "supporting exploratory 2D evidence",
                "solver_results_modified": False,
                "one_dimensional_frames_reused": True,
                "one_dimensional_source": _case_relative(SOURCE_1D_META),
                "horizontal_pipe_diameter": D,
                "physical_tower_diameter": DT,
                "computational_slit_width": SLIT_WIDTH,
                "physical_rim_y": RIM_Y,
                "external_domain": {
                    "x_left": EXTERNAL_LEFT,
                    "x_right": EXTERNAL_RIGHT,
                    "y_bottom": RIM_Y,
                    "y_top": EXTERNAL_TOP,
                },
                "display": {
                    "top_y": DISPLAY_TOP,
                    "external_air_cells_hidden": True,
                    "visible_plume_alpha_min": VISIBLE_PLUME_ALPHA,
                    "tower_and_external_x_mapping": (
                        "piecewise display-only map: W -> Dt at the mouth; "
                        "external-box outer edges fixed"
                    ),
                    "fake_rim_or_wall_extension_drawn": False,
                },
                "frames": rendered_frames,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"frames -> {FRAME_DIR}")
    print(f"manifest -> {MANIFEST}")
    print(f"metadata -> {METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--max-time-offset",
        type=float,
        default=0.031,
        help="maximum absolute 1-D/2-D pairing offset in seconds",
    )
    args = parser.parse_args()
    main(dpi=args.dpi, max_time_offset=args.max_time_offset)
