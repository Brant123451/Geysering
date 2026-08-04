#!/usr/bin/env python3
"""Render Case B one-way top-plume frames and full-view replacement overlays.

The OpenFOAM top-only case uses local coordinates with the physical rim at
``y_local = 0`` and the inlet/source plane at ``y_local = -0.0014375 m``.
Every rendered record stores both clocks:

``source_time = local_time + 6.5 s``.

Air is transparent.  Only cells with ``alpha.water >= 0.01`` are rendered.
The solver's area-equivalent planar mouth is mapped to the physical tower
diameter for display, with a continuous piecewise map that fixes the outer
edges of the 0.24 m atmospheric box.  This script is post-processing only and
does not run, reconstruct, or modify the OpenFOAM solution.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle


CASE_ROOT = Path(__file__).resolve().parents[1]
OPENFOAM_CASE = CASE_ROOT / "openfoam" / "2d_top_plume"
VTK_DIR = OPENFOAM_CASE / "VTK"
DRIVER_CSV = CASE_ROOT / "outputs" / "caseB_2d_mouth_forcing_sanitized.csv"
OUTPUT_ROOT = OPENFOAM_CASE / "outputs_viewer"
SELECTION_JSON = OUTPUT_ROOT / "selected_times.json"
ZOOM_DIR = OUTPUT_ROOT / "frames_top_plume_zoom"
OVERLAY_DIR = OUTPUT_ROOT / "frames_top_plume_overlay"
SPACE_DIR = OUTPUT_ROOT / "frames_top_plume_space"
MANIFEST = OUTPUT_ROOT / "frames_index_top_plume.json"
METADATA = OUTPUT_ROOT / "frames_top_plume_meta.json"

SOURCE_TIME_OFFSET = 6.5
PIPE_D = 0.094
PHYSICAL_TOWER_D = 0.0127
SLIT_WIDTH = PHYSICAL_TOWER_D**2 / PIPE_D
X_TOWER = 3.516
LOCAL_INLET_Y = -0.0014375
LOCAL_RIM_Y = 0.0
LOCAL_EXTERNAL_TOP = 0.600
GLOBAL_RIM_Y = 0.657
GLOBAL_INLET_Y = GLOBAL_RIM_Y + LOCAL_INLET_Y
LOCAL_EXTERNAL_LEFT = -0.120
LOCAL_EXTERNAL_RIGHT = 0.120
GLOBAL_EXTERNAL_LEFT = X_TOWER + LOCAL_EXTERNAL_LEFT
GLOBAL_EXTERNAL_RIGHT = X_TOWER + LOCAL_EXTERNAL_RIGHT
VISIBLE_ALPHA = 0.01

# These values exactly match the established Case B full-domain frame canvas.
FULL_X_LIM = (-0.04, 4.046)
FULL_Y_LIM = (-0.5 * PIPE_D - 0.035, 1.0)
FULL_FIGSIZE = (12.4, 3.2)
FULL_SUBPLOT = {"left": 0.075, "right": 0.985, "bottom": 0.18, "top": 0.94}

WATER = "#2f7ff7"
WALL = "#333333"


def _floats(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _ints(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.int64)


def _data_array(raw: str, name: str) -> str:
    match = re.search(
        rf"<DataArray[^>]*Name=['\"]{re.escape(name)}['\"][^>]*>(.*?)</DataArray>",
        raw,
        flags=re.S,
    )
    if not match:
        raise RuntimeError(f"Missing DataArray {name!r}")
    return match.group(1)


def read_vtu_cell_bounds_alpha(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return exact x/y cell bounds and alpha.water from an ASCII VTU."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    points = _floats(_data_array(raw, "Points")).reshape(-1, 3)
    connectivity = _ints(_data_array(raw, "connectivity"))
    offsets = _ints(_data_array(raw, "offsets"))
    alpha = _floats(_data_array(raw, "alpha.water"))
    if len(alpha) != len(offsets):
        raise RuntimeError(
            f"alpha.water has {len(alpha)} values but {len(offsets)} cells in {path}"
        )

    bounds = np.empty((len(offsets), 4), dtype=np.float64)
    start = 0
    for index, end in enumerate(offsets):
        cell_points = points[connectivity[start:end]]
        bounds[index] = (
            np.min(cell_points[:, 0]),
            np.max(cell_points[:, 0]),
            np.min(cell_points[:, 1]),
            np.max(cell_points[:, 1]),
        )
        start = int(end)
    return bounds, np.clip(alpha, 0.0, 1.0)


def parse_vtk_series() -> list[tuple[float, Path]]:
    preferred = VTK_DIR / "2d_top_plume.vtm.series"
    if preferred.exists():
        series_path = preferred
    else:
        candidates = sorted(VTK_DIR.glob("*.vtm.series"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one VTM series in {VTK_DIR}; found {len(candidates)}"
            )
        series_path = candidates[0]
    payload = json.loads(series_path.read_text(encoding="utf-8"))
    frames = []
    for item in payload.get("files", []):
        local_time = float(item["time"])
        vtu = VTK_DIR / Path(item["name"]).stem / "internal.vtu"
        if vtu.exists():
            frames.append((local_time, vtu))
    return frames


def load_target_local_times() -> list[float]:
    if SELECTION_JSON.exists():
        selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
        times = [float(frame["target_local_time_s"]) for frame in selection["frames"]]
        if times:
            return times
    with DRIVER_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = [float(row["local_time_s"]) for row in rows]
    if not times or abs(times[0]) > 1.0e-12:
        raise ValueError(f"Invalid local-time sequence in {DRIVER_CSV}")
    return times


def map_local_x_for_display(values: np.ndarray) -> np.ndarray:
    """Map W to physical Dt while preserving exterior-box outer boundaries."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    slit_left = -0.5 * SLIT_WIDTH
    slit_right = 0.5 * SLIT_WIDTH
    physical_left = -0.5 * PHYSICAL_TOWER_D
    physical_right = 0.5 * PHYSICAL_TOWER_D
    left = values < slit_left
    centre = (values >= slit_left) & (values <= slit_right)
    right = values > slit_right
    out[left] = LOCAL_EXTERNAL_LEFT + (
        (values[left] - LOCAL_EXTERNAL_LEFT)
        * (physical_left - LOCAL_EXTERNAL_LEFT)
        / (slit_left - LOCAL_EXTERNAL_LEFT)
    )
    out[centre] = values[centre] * (PHYSICAL_TOWER_D / SLIT_WIDTH)
    out[right] = physical_right + (
        (values[right] - slit_right)
        * (LOCAL_EXTERNAL_RIGHT - physical_right)
        / (LOCAL_EXTERNAL_RIGHT - slit_right)
    )
    return out


def _water_polygons(
    bounds: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global display polygons, alpha, and source-cell global y maxima."""
    visible = np.isfinite(alpha) & (alpha >= VISIBLE_ALPHA)
    selected = bounds[visible]
    values = alpha[visible]
    if not len(selected):
        return np.empty((0, 4, 2)), values, np.empty(0)

    x_pair = map_local_x_for_display(selected[:, :2].reshape(-1)).reshape(-1, 2)
    x_pair += X_TOWER
    y_pair = selected[:, 2:4] + GLOBAL_RIM_Y
    polygons = np.stack(
        [
            np.column_stack([x_pair[:, 0], y_pair[:, 0]]),
            np.column_stack([x_pair[:, 1], y_pair[:, 0]]),
            np.column_stack([x_pair[:, 1], y_pair[:, 1]]),
            np.column_stack([x_pair[:, 0], y_pair[:, 1]]),
        ],
        axis=1,
    )
    return polygons, values, y_pair[:, 1]


def _add_water(ax: plt.Axes, polygons: np.ndarray, alpha: np.ndarray) -> None:
    if not len(polygons):
        return
    cmap = LinearSegmentedColormap.from_list(
        "top_plume_water",
        ["#b8d7ff", "#78adff", WATER],
        N=256,
    )
    colors = cmap(Normalize(vmin=VISIBLE_ALPHA, vmax=1.0, clip=True)(alpha))
    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        rasterized=True,
        zorder=2,
    )
    ax.add_collection(collection)


def _draw_short_physical_tower(ax: plt.Axes) -> None:
    left = X_TOWER - 0.5 * PHYSICAL_TOWER_D
    right = X_TOWER + 0.5 * PHYSICAL_TOWER_D
    ax.plot([left, left], [GLOBAL_INLET_Y, GLOBAL_RIM_Y], color=WALL, lw=0.8, zorder=4)
    ax.plot([right, right], [GLOBAL_INLET_Y, GLOBAL_RIM_Y], color=WALL, lw=0.8, zorder=4)


def render_zoom(
    *,
    local_time: float,
    source_time: float,
    polygons: np.ndarray,
    alpha: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 7.2))
        _add_water(ax, polygons, alpha)
        _draw_short_physical_tower(ax)
        ax.text(
            0.04,
            0.97,
            f"Local time = {local_time:.2f} s\nSource time = {source_time:.2f} s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.text(
            0.04,
            0.90,
            rf"water shown for $\alpha_{{water}} \geq {VISIBLE_ALPHA:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_xlim(GLOBAL_EXTERNAL_LEFT - 0.005, GLOBAL_EXTERNAL_RIGHT + 0.005)
        ax.set_ylim(GLOBAL_INLET_Y - 0.004, GLOBAL_RIM_Y + LOCAL_EXTERNAL_TOP + 0.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal coordinate [m]")
        ax.set_ylabel("physical vertical coordinate [m]")
        ax.legend(
            handles=[Patch(facecolor=WATER, label="water")],
            loc="upper right",
            frameon=False,
            prop={"family": "Times New Roman", "size": 9},
        )
        fig.subplots_adjust(left=0.20, right=0.96, bottom=0.10, top=0.97)
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)


def render_overlay(
    *,
    polygons: np.ndarray,
    alpha: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    """Write a full-canvas PNG aligned pixel-for-pixel with the old 2-D frame."""
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=FULL_FIGSIZE)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        # Opaque replacement window removes the old artificial extension and
        # its dashed rim marker before the new free-plume water is overlaid.
        ax.add_patch(
            Rectangle(
                (GLOBAL_EXTERNAL_LEFT, GLOBAL_INLET_Y),
                GLOBAL_EXTERNAL_RIGHT - GLOBAL_EXTERNAL_LEFT,
                FULL_Y_LIM[1] - GLOBAL_INLET_Y,
                facecolor="white",
                edgecolor="none",
                zorder=1,
            )
        )
        _add_water(ax, polygons, alpha)
        _draw_short_physical_tower(ax)
        ax.set_xlim(*FULL_X_LIM)
        ax.set_ylim(*FULL_Y_LIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        fig.subplots_adjust(**FULL_SUBPLOT)
        fig.savefig(output, dpi=dpi, transparent=True)
        plt.close(fig)


def render_space(
    *,
    polygons: np.ndarray,
    alpha: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    """Render an axis-free, visibly open air space immediately above the rim.

    The computational air domain extends 0.60 m above the rim.  This local
    presentation window shows the lower 0.08 m so that the physically small
    (centimetre-scale) liquid tongue remains visible without rescaling the
    water field itself.
    """
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 5.0))
        x_min = X_TOWER - 0.040
        x_max = X_TOWER + 0.040
        y_min = GLOBAL_INLET_Y - 0.003
        y_max = GLOBAL_RIM_Y + 0.080
        fig.patch.set_facecolor("#f2f8fc")
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="#f2f8fc",
                edgecolor="none",
                zorder=0,
            )
        )
        _add_water(ax, polygons, alpha)
        _draw_short_physical_tower(ax)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig.savefig(output, dpi=dpi, facecolor="#f2f8fc")
        plt.close(fig)


def _case_relative(path: Path) -> str:
    return str(path.relative_to(CASE_ROOT)).replace("\\", "/")


def main(*, dpi: int = 140, max_time_offset: float = 0.006) -> None:
    series = parse_vtk_series()
    if not series:
        raise RuntimeError(
            "No readable top-plume VTK series found. Run "
            "caseB_prepare_top_plume_vtk.sh only after the solver completes."
        )
    target_times = load_target_local_times()

    ZOOM_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    SPACE_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (ZOOM_DIR, OVERLAY_DIR, SPACE_DIR):
        for old in directory.glob("frame_*.png"):
            old.unlink()

    frames = []
    for index, target_local_time in enumerate(target_times):
        local_time, vtu = min(series, key=lambda item: abs(item[0] - target_local_time))
        offset = local_time - target_local_time
        if abs(offset) > max_time_offset:
            raise RuntimeError(
                f"Nearest VTU to local target {target_local_time:g} s is "
                f"{local_time:g} s (offset {offset:g} s)."
            )
        source_time = local_time + SOURCE_TIME_OFFSET
        bounds, alpha_all = read_vtu_cell_bounds_alpha(vtu)
        polygons, alpha, visible_y_max = _water_polygons(bounds, alpha_all)

        zoom = ZOOM_DIR / f"frame_{index:04d}.png"
        overlay = OVERLAY_DIR / f"frame_{index:04d}.png"
        space = SPACE_DIR / f"frame_{index:04d}.png"
        render_zoom(
            local_time=local_time,
            source_time=source_time,
            polygons=polygons,
            alpha=alpha,
            output=zoom,
            dpi=dpi,
        )
        render_overlay(polygons=polygons, alpha=alpha, output=overlay, dpi=dpi)
        render_space(polygons=polygons, alpha=alpha, output=space, dpi=dpi)

        frames.append(
            {
                "index": index,
                "target_local_time_s": float(target_local_time),
                "local_time_s": float(local_time),
                "source_time_s": float(source_time),
                "source_time_definition": "source_time_s = local_time_s + 6.5",
                "local_time_offset_s": float(offset),
                "file_zoom": _case_relative(zoom),
                "file_overlay": _case_relative(overlay),
                "file_space": _case_relative(space),
                "source_vtu": str(vtu.relative_to(OPENFOAM_CASE)).replace("\\", "/"),
                "visible_alpha_min": VISIBLE_ALPHA,
                "visible_water_cell_count": int(len(alpha)),
                "visible_water_cell_top_y_m": (
                    float(np.max(visible_y_max)) if len(visible_y_max) else None
                ),
            }
        )
        print(
            f"rendered {index + 1}/{len(target_times)}: "
            f"local={local_time:.2f} s, source={source_time:.2f} s",
            flush=True,
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    METADATA.write_text(
        json.dumps(
            {
                "description": (
                    "Case B top-only one-way-coupled free-plume frames and "
                    "full-domain replacement overlays."
                ),
                "scientific_role": "supporting exploratory 2D visualization",
                "source_time_definition": "source_time_s = local_time_s + 6.5",
                "source_time_offset_s": SOURCE_TIME_OFFSET,
                "physical_rim_y_m": GLOBAL_RIM_Y,
                "local_rim_y_m": LOCAL_RIM_Y,
                "local_inlet_y_m": LOCAL_INLET_Y,
                "global_inlet_y_m": GLOBAL_INLET_Y,
                "inlet_to_rim_distance_m": GLOBAL_RIM_Y - GLOBAL_INLET_Y,
                "computational_slit_width_m": SLIT_WIDTH,
                "physical_tower_diameter_m": PHYSICAL_TOWER_D,
                "external_domain_local_m": {
                    "x_left": LOCAL_EXTERNAL_LEFT,
                    "x_right": LOCAL_EXTERNAL_RIGHT,
                    "y_bottom": LOCAL_RIM_Y,
                    "y_top": LOCAL_EXTERNAL_TOP,
                },
                "display": {
                    "air_transparent": True,
                    "visible_alpha_min": VISIBLE_ALPHA,
                    "fake_wall_or_cap_above_rim": False,
                    "space_view_height_above_rim_m": 0.08,
                    "x_mapping": (
                        "display-only piecewise W-to-Dt mouth mapping with "
                        "fixed external-domain outer edges"
                    ),
                    "overlay_canvas_pixels_at_default_dpi": [
                        int(FULL_FIGSIZE[0] * dpi),
                        int(FULL_FIGSIZE[1] * dpi),
                    ],
                },
                "coupling_limit": (
                    "The overlay combines the archived full-domain 2D frame "
                    "with a separate one-way top-only plume calculation; it is "
                    "not a monolithic two-way-coupled CFD field."
                ),
                "frames": frames,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"zoom frames -> {ZOOM_DIR}")
    print(f"replacement overlays -> {OVERLAY_DIR}")
    print(f"open-air local frames -> {SPACE_DIR}")
    print(f"manifest -> {MANIFEST}")
    print(f"metadata -> {METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--max-time-offset", type=float, default=0.006)
    args = parser.parse_args()
    main(dpi=args.dpi, max_time_offset=args.max_time_offset)
