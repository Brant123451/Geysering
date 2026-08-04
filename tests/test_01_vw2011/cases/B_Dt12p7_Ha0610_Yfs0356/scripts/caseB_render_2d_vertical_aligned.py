#!/usr/bin/env python3
"""Redraw Case B OpenFOAM frames with a visually aligned standpipe.

The Case B 2-D calculation uses the area-equivalent planar slit
``W = Dt**2 / D``.  That width is essential to the calculation and is not
changed here.  For the 1-D/2-D viewer only, cells above the horizontal-pipe
crown are mapped laterally from the computational slit width ``W`` to the
physical standpipe diameter ``Dt``.  The alpha.water values and all vertical
coordinates remain unchanged.

This script deliberately writes a new frame directory.  The original
``frames_2d`` images and OpenFOAM VTK results are retained.
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
OPENFOAM_2D = CASE_ROOT / "openfoam" / "2d"
OUTPUT_ROOT = OPENFOAM_2D / "outputs_1d2d_compare"
FRAME_DIR = OUTPUT_ROOT / "frames_2d_caseB_tosan2021_aligned"
MANIFEST = OUTPUT_ROOT / "frames_index_tosan2021.json"
METADATA = OUTPUT_ROOT / "frames_2d_caseB_tosan2021_aligned_meta.json"
SOURCE_RENDERER = OPENFOAM_2D / "_local_render_2d_frames.py"

D = 0.094
DT = 0.0127
SLIT_WIDTH = DT * DT / D
X_TOWER = 3.516
X_VALVE = 0.546
X_END = 4.006
PIPE_BOTTOM = -0.5 * D
PIPE_CROWN = 0.5 * D
RIM_Y = PIPE_CROWN + 0.610
OBSERVATION_TOP = 0.957
VISIBLE_DISCHARGE_ALPHA = 0.01

WATER = "#2f7ff7"
AIR = "#f1f3f6"
WALL = "#333333"
RIM = "#ef4444"


def _load_source_parser() -> tuple[
    Callable[[], list[tuple[float, Path]]],
    Callable[[Path], tuple[np.ndarray, np.ndarray]],
]:
    """Load the established VTM-series and ASCII-VTU parsing functions."""
    if not SOURCE_RENDERER.exists():
        raise FileNotFoundError(f"Missing source VTK renderer: {SOURCE_RENDERER}")
    spec = importlib.util.spec_from_file_location(
        "caseb_existing_2d_renderer",
        SOURCE_RENDERER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SOURCE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_series, module.read_vtu_cell_centers_alpha


def _centres_to_edges(values: np.ndarray) -> np.ndarray:
    """Infer ordered cell edges from a one-dimensional centre coordinate."""
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
    """Place rectilinear cell-centred values in a pcolormesh-compatible grid."""
    # foamToVTK emits decimal ASCII coordinates.  Rounding suppresses harmless
    # last-digit differences at block joins without changing the mesh.
    xr = np.round(np.asarray(x, dtype=float), 12)
    yr = np.round(np.asarray(y, dtype=float), 12)
    xu = np.unique(xr)
    yu = np.unique(yr)
    ix = np.searchsorted(xu, xr)
    iy = np.searchsorted(yu, yr)
    field = np.full((yu.size, xu.size), np.nan, dtype=float)
    field[iy, ix] = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return _centres_to_edges(xu), _centres_to_edges(yu), field


def _plot_frame(
    source_time: float,
    vtu: Path,
    output: Path,
    read_vtu: Callable[[Path], tuple[np.ndarray, np.ndarray]],
    *,
    dpi: int,
) -> dict:
    centers, alpha = read_vtu(vtu)
    x = centers[:, 0]
    y = centers[:, 1]

    horizontal = (
        (x >= -0.001)
        & (x <= X_END + 0.001)
        & (y >= PIPE_BOTTOM - 0.001)
        & (y <= PIPE_CROWN + 1.0e-10)
    )
    tower = (
        (np.abs(x - X_TOWER) <= 0.5 * SLIT_WIDTH + 1.0e-8)
        & (y > PIPE_CROWN)
        & (y <= OBSERVATION_TOP + 0.001)
    )
    if not np.any(horizontal):
        raise RuntimeError(f"No horizontal-pipe cells found in {vtu}")
    if not np.any(tower):
        raise RuntimeError(f"No standpipe cells found in {vtu}")

    hx, hy, ha = _structured_cell_field(
        x[horizontal],
        y[horizontal],
        alpha[horizontal],
    )
    tx, ty, ta = _structured_cell_field(x[tower], y[tower], alpha[tower])
    # Display-only affine map: W-wide planar slit -> Dt-wide physical tube.
    tx = X_TOWER + (tx - X_TOWER) * (DT / SLIT_WIDTH)
    # The mesh extends above the experimental rim so that an expelled jet can
    # be resolved.  That numerical headroom is not part of the apparatus:
    # mask its air cells while retaining discharged liquid above the open rim.
    tower_row_y = 0.5 * (ty[:-1] + ty[1:])
    above_rim = tower_row_y[:, None] >= RIM_Y
    tower_display = np.ma.masked_invalid(ta)
    tower_display = np.ma.masked_where(
        above_rim & (np.nan_to_num(ta, nan=0.0) < VISIBLE_DISCHARGE_ALPHA),
        tower_display,
    )

    cmap = LinearSegmentedColormap.from_list(
        "caseb_air_water",
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
        ax.pcolormesh(
            hx,
            hy,
            ha,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            linewidth=0.0,
            rasterized=True,
            zorder=1,
        )
        ax.pcolormesh(
            tx,
            ty,
            tower_display,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            linewidth=0.0,
            rasterized=True,
            zorder=1,
        )

        tower_left = X_TOWER - 0.5 * DT
        tower_right = X_TOWER + 0.5 * DT
        line_kw = {"color": WALL, "linewidth": 0.8, "zorder": 3}

        # Connected horizontal pipe: omit its crown beneath the tower opening.
        ax.plot([0.0, X_END], [PIPE_BOTTOM, PIPE_BOTTOM], **line_kw)
        ax.plot([0.0, 0.0], [PIPE_BOTTOM, PIPE_CROWN], **line_kw)
        ax.plot([X_END, X_END], [PIPE_BOTTOM, PIPE_CROWN], **line_kw)
        ax.plot([0.0, tower_left], [PIPE_CROWN, PIPE_CROWN], **line_kw)
        ax.plot([tower_right, X_END], [PIPE_CROWN, PIPE_CROWN], **line_kw)

        # The physical tower ends at the open experimental rim.  No bottom edge
        # is drawn, preserving hydraulic contact with the horizontal pipe.
        ax.plot([tower_left, tower_left], [PIPE_CROWN, RIM_Y], **line_kw)
        ax.plot([tower_right, tower_right], [PIPE_CROWN, RIM_Y], **line_kw)

        # This is the visible rim/platform marker used in the 1-D rendering,
        # centred exactly at x=3.516 m.
        ax.plot(
            [3.470, 3.562],
            [RIM_Y, RIM_Y],
            color=RIM,
            linestyle="--",
            linewidth=1.0,
            zorder=4,
        )
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
            "OpenFOAM 2D  |  area-equivalent slit remapped to physical "
            r"$D_t$ for display",
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
        ax.set_ylim(PIPE_BOTTOM - 0.035, 1.0)
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
        "file": f"{FRAME_DIR.name}/{output.name}",
        "time": float(source_time),
        "source_vtu": str(vtu.relative_to(OPENFOAM_2D)).replace("\\", "/"),
        "computational_slit_width": SLIT_WIDTH,
        "display_tower_width": DT,
        "horizontal_scale_factor": DT / SLIT_WIDTH,
    }


def _update_manifest(rendered: list[dict]) -> None:
    """Point the Tosan comparison manifest at an already rendered frame set."""
    pairs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Expected a non-empty frame list in {MANIFEST}")
    if len(rendered) != len(pairs):
        raise RuntimeError(
            f"Aligned 2-D frame count {len(rendered)} does not match "
            f"comparison frame count {len(pairs)}"
        )
    for pair, frame in zip(pairs, rendered):
        output = FRAME_DIR / Path(str(frame["file"])).name
        if not output.exists():
            raise FileNotFoundError(f"Missing aligned 2-D frame: {output}")
        pair["file2d"] = frame["file"]
    MANIFEST.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_manifest_only() -> None:
    """Restore aligned paths after another script has rebuilt the manifest."""
    payload = json.loads(METADATA.read_text(encoding="utf-8"))
    rendered = payload.get("frames", [])
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(f"Expected rendered frame metadata in {METADATA}")
    _update_manifest(rendered)
    print(f"updated 2-D paths -> {MANIFEST}")


def main(*, dpi: int = 140) -> None:
    parse_series, read_vtu = _load_source_parser()
    series = parse_series()
    if not series:
        raise RuntimeError(
            f"No readable internal.vtu files listed by "
            f"{OPENFOAM_2D / 'VTK' / '2d.vtm.series'}"
        )
    pairs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Expected a non-empty frame list in {MANIFEST}")

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in FRAME_DIR.glob("frame_*.png"):
        old_frame.unlink()

    rendered = []
    for index, pair in enumerate(pairs):
        target_time = float(pair["time"])
        source_time, vtu = min(series, key=lambda item: abs(item[0] - target_time))
        filename = f"frame_{index:04d}.png"
        frame = _plot_frame(
            source_time,
            vtu,
            FRAME_DIR / filename,
            read_vtu,
            dpi=dpi,
        )
        frame["index"] = index
        frame["target_time"] = target_time
        frame["time_offset"] = float(source_time - target_time)
        rendered.append(frame)
        print(
            f"rendered {index + 1}/{len(pairs)}: "
            f"target={target_time:.2f} s, VTK={source_time:.2f} s",
            flush=True,
        )

    METADATA.write_text(
        json.dumps(
            {
                "description": (
                    "Display-only lateral remapping of the Case B OpenFOAM "
                    "area-equivalent tower slit to the physical tower diameter."
                ),
                "solver_results_modified": False,
                "horizontal_pipe_diameter": D,
                "physical_tower_diameter": DT,
                "computational_slit_width": SLIT_WIDTH,
                "tower_center_x": X_TOWER,
                "physical_rim_y": RIM_Y,
                "observation_top_y": OBSERVATION_TOP,
                "display_above_rim": (
                    "Numerical air cells are hidden; only discharged liquid "
                    "with alpha.water >= 0.01 is retained above the open rim."
                ),
                "frames": rendered,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _update_manifest(rendered)
    print(f"frames -> {FRAME_DIR}")
    print(f"metadata -> {METADATA}")
    print(f"updated 2-D paths -> {MANIFEST}")
    print(
        "Rebuild the HTML so its cache-busting asset version reflects the "
        "updated manifest."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Reuse rendered metadata and only restore aligned file2d paths.",
    )
    args = parser.parse_args()
    if args.manifest_only:
        update_manifest_only()
    else:
        main(dpi=args.dpi)
