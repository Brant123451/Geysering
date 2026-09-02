#!/usr/bin/env python3
"""Build the Case-B full-domain three-frame 1D--2D comparison.

This figure intentionally uses the current Tosan-based one-dimensional
result and the archived, confined-headroom OpenFOAM 2-D result.  It does not
read or composite the separate top-only plume calculation, so every plotted
field remains a direct output of the stated full-domain calculation.

The two columns are redrawn in one physical coordinate system.  Every row is
paired at the same physical time without event alignment or time shifting.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
FRAME_ROOT = CASE_ROOT / "openfoam" / "2d" / "outputs_1d2d_compare"
OPENFOAM_2D = CASE_ROOT / "openfoam" / "2d"
PAPER_FIG = REPO_ROOT / "paper" / "figures"
OUTPUT_STEM = "caseB_1d2d_snapshots_3frame_current"
MANIFEST = CASE_ROOT / "outputs" / f"{OUTPUT_STEM}_manifest.json"

ONE_D_BUILDER = CASE_ROOT / "scripts" / "caseB_rebuild_1d_tosan2021.py"
TWO_D_RENDERER = CASE_ROOT / "scripts" / "caseB_render_2d_vertical_aligned.py"
TWO_D_META = FRAME_ROOT / "frames_2d_caseB_tosan2021_aligned_meta.json"

TARGET_TIMES = (6.35, 7.25, 7.70)
STAGES = (
    "horizontal gas--water interface approaching the standpipe",
    "gas entry and liquid-column lift",
    "above-rim liquid-column discharge",
)

D = 0.094
DT = 0.0127
DISPLAY_DT = 0.080
X_TOWER = 3.516
X_END = 4.006
TOWER_HEIGHT = 0.610
SLIT_WIDTH = DT * DT / D
GROUND_LENGTH = 0.075
VISIBLE_WATER_AREA_FRACTION = 0.10
VISIBLE_DISCHARGE_ALPHA = 0.01

WATER = "#2F7FF7"
AIR = "#F2F4F8"
WALL = "#4A4A4A"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _draw_outline(ax) -> None:
    """Draw a shared domain outline with display-widened standpipe walls."""
    tower_left = X_TOWER - 0.5 * DISPLAY_DT
    tower_right = X_TOWER + 0.5 * DISPLAY_DT
    wall = dict(color=WALL, linewidth=0.75, zorder=8)

    ax.plot([0.0, X_END], [-D, -D], **wall)
    ax.plot([0.0, 0.0], [-D, 0.0], **wall)
    ax.plot([X_END, X_END], [-D, 0.0], **wall)
    ax.plot([0.0, tower_left], [0.0, 0.0], **wall)
    ax.plot([tower_right, X_END], [0.0, 0.0], **wall)
    ax.plot([tower_left, tower_left], [0.0, TOWER_HEIGHT], **wall)
    ax.plot([tower_right, tower_right], [0.0, TOWER_HEIGHT], **wall)

    # Leave the tower open and draw two short ground-level strokes outside it.
    ax.plot(
        [tower_left - GROUND_LENGTH, tower_left],
        [TOWER_HEIGHT, TOWER_HEIGHT],
        **wall,
    )
    ax.plot(
        [tower_right, tower_right + GROUND_LENGTH],
        [TOWER_HEIGHT, TOWER_HEIGHT],
        **wall,
    )


def _format_panel(ax, letter: str) -> None:
    ax.set_xlim(-0.03, 4.036)
    ax.set_ylim(-0.124, 0.920)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(
        0.012,
        0.91,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=20,
    )


def _draw_1d_panel(ax, horizontal: dict, tower: dict) -> None:
    """Render the current Tosan horizontal state and retained tower branch."""
    builder = _load_module("caseb_current_1d_helpers", ONE_D_BUILDER)
    x = np.asarray(horizontal["x"], dtype=float)
    alpha = np.asarray(horizontal["area_fraction"], dtype=float)
    alpha_visible = np.where(alpha >= VISIBLE_WATER_AREA_FRACTION, alpha, 0.0)
    depths = D * builder._depth_fraction_from_area(alpha_visible)

    ax.add_patch(
        Rectangle((0.0, -D), X_END, D, facecolor=AIR, edgecolor="none", zorder=0)
    )
    ax.fill_between(
        x,
        -D,
        -D + depths,
        step="mid",
        color=WATER,
        linewidth=0.0,
        zorder=2,
    )

    tower_left = X_TOWER - 0.5 * DISPLAY_DT
    ax.add_patch(
        Rectangle(
            (tower_left, 0.0), DISPLAY_DT, TOWER_HEIGHT,
            facecolor=AIR, edgecolor="none", zorder=0,
        )
    )
    water_top = min(float(tower["wtop"]), TOWER_HEIGHT)
    z = np.asarray(tower["z"], dtype=float)
    dz = float(tower["dz"])
    for z_i, alpha_l, alpha_g in zip(z, tower["alpha_liquid"], tower["alpha_gas"]):
        z0 = float(z_i - 0.5 * dz)
        if z0 >= water_top:
            continue
        cell_height = min(dz, water_top - z0)
        if cell_height <= 0.0:
            continue
        gas_width = math.sqrt(float(np.clip(alpha_g, 0.0, 1.0))) * DISPLAY_DT
        film_width = max(0.5 * (DISPLAY_DT - gas_width), 0.0)
        if float(alpha_l) > 1.0e-4 and film_width > 0.0:
            ax.add_patch(
                Rectangle(
                    (tower_left, z0), film_width, cell_height,
                    facecolor=WATER, edgecolor="none", zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (tower_left + DISPLAY_DT - film_width, z0), film_width, cell_height,
                    facecolor=WATER, edgecolor="none", zorder=2,
                )
            )

    jet_top = float(tower["jet_height"])
    if jet_top > TOWER_HEIGHT:
        jet_width = 0.55 * DISPLAY_DT
        ax.add_patch(
            Rectangle(
                (X_TOWER - 0.5 * jet_width, TOWER_HEIGHT),
                jet_width,
                jet_top - TOWER_HEIGHT,
                facecolor=WATER,
                edgecolor="none",
                zorder=3,
            )
        )
    _draw_outline(ax)


def _select_2d_frames() -> list[dict]:
    payload = json.loads(TWO_D_META.read_text(encoding="utf-8"))
    frames = payload["frames"]
    selected = []
    for target in TARGET_TIMES:
        frame = min(frames, key=lambda item: abs(float(item["time"]) - target))
        delta = abs(float(frame["time"]) - target)
        if delta > 5.0e-7:
            raise FileNotFoundError(
                f"No archived 2-D frame at {target:.2f} s; nearest differs by {delta:.3g} s"
            )
        selected.append(frame)
    return selected


def _draw_2d_panel(ax, frame: dict, renderer, read_vtu) -> None:
    """Render the old confined 2-D phase field in the common physical frame."""
    source = OPENFOAM_2D / frame["source_vtu"]
    centres, alpha = read_vtu(source)
    x = centres[:, 0]
    y = centres[:, 1]
    pipe_bottom = -0.5 * D
    pipe_crown = 0.5 * D
    observation_top = float(
        json.loads(TWO_D_META.read_text(encoding="utf-8"))["observation_top_y"]
    )

    horizontal = (
        (x >= -0.001)
        & (x <= X_END + 0.001)
        & (y >= pipe_bottom - 0.001)
        & (y <= pipe_crown + 1.0e-10)
    )
    tower_cells = (
        (np.abs(x - X_TOWER) <= 0.5 * SLIT_WIDTH + 1.0e-8)
        & (y > pipe_crown)
        & (y <= observation_top + 0.001)
    )
    if not np.any(horizontal) or not np.any(tower_cells):
        raise RuntimeError(f"Incomplete 2-D pipe/tower field in {source}")

    hx, hy, ha = renderer._structured_cell_field(
        x[horizontal], y[horizontal] - pipe_crown, alpha[horizontal]
    )
    tx, ty, ta = renderer._structured_cell_field(
        x[tower_cells], y[tower_cells] - pipe_crown, alpha[tower_cells]
    )
    tx = X_TOWER + (tx - X_TOWER) * (DISPLAY_DT / SLIT_WIDTH)

    row_y = 0.5 * (ty[:-1] + ty[1:])
    above_rim = row_y[:, None] >= TOWER_HEIGHT
    tower_display = np.ma.masked_invalid(ta)
    tower_display = np.ma.masked_where(
        above_rim & (np.nan_to_num(ta, nan=0.0) < VISIBLE_DISCHARGE_ALPHA),
        tower_display,
    )

    cmap = LinearSegmentedColormap.from_list(
        "caseb_current_air_water", [AIR, "#9EC5FE", WATER], N=256
    )
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.pcolormesh(
        hx, hy, ha, cmap=cmap, vmin=0.0, vmax=1.0,
        shading="flat", linewidth=0.0, rasterized=True, zorder=1,
    )
    ax.pcolormesh(
        tx, ty, tower_display, cmap=cmap, vmin=0.0, vmax=1.0,
        shading="flat", linewidth=0.0, rasterized=True, zorder=1,
    )
    _draw_outline(ax)


def main() -> None:
    _configure_style()
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    builder = _load_module("caseb_current_1d_builder", ONE_D_BUILDER)
    _, _, vertical = builder._run_vertical_reference(list(TARGET_TIMES))
    _, _, horizontal = builder._run_horizontal(list(TARGET_TIMES), vertical)

    renderer = _load_module("caseb_current_2d_renderer", TWO_D_RENDERER)
    _, read_vtu = renderer._load_source_parser()
    frames_2d = _select_2d_frames()

    fig, axes = plt.subplots(
        3, 2, figsize=(7.20, 3.78), sharex=True, sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.025, right=0.995, bottom=0.025, top=0.895,
        wspace=0.055, hspace=0.48,
    )
    fig.text(
        0.267, 0.975, "Present 1D model", ha="center", va="top",
        fontsize=9.5, fontweight="bold",
    )
    fig.text(
        0.753, 0.975, "2D OpenFOAM", ha="center", va="top",
        fontsize=9.5, fontweight="bold",
    )

    selected = []
    for row, (target, stage, state_1d, tower, frame_2d) in enumerate(
        zip(TARGET_TIMES, STAGES, horizontal, vertical, frames_2d)
    ):
        letter = chr(ord("a") + row)
        _draw_1d_panel(axes[row, 0], state_1d, tower)
        _draw_2d_panel(axes[row, 1], frame_2d, renderer, read_vtu)
        _format_panel(axes[row, 0], letter)
        _format_panel(axes[row, 1], letter)

        selected.append(
            {
                "panel": letter,
                "stage": stage,
                "paired_time_s": target,
                "time_shift_s": 0.0,
                "one_d": {
                    "source": "current Tosan (2021) shock-fitting horizontal result with retained vertical branch",
                    "source_time_s": float(state_1d["time"]),
                    "interface_x_m": float(state_1d["interface_x"]),
                    "air_pressure_head_m": float(state_1d["air_pressure_head_gauge"]),
                    "Yfs_m": float(tower["wtop"]),
                    "Yint_m": float(tower["itop"]),
                    "jet_height_m": float(tower["jet_height"]),
                },
                "two_d": {
                    "source_time_s": float(frame_2d["time"]),
                    "source_vtu": f"openfoam/2d/{frame_2d['source_vtu']}",
                    "solver_domain": "old confined 0.30-m numerical headroom above the physical rim",
                    "display_remap": (
                        "area-equivalent slit remapped laterally to the common "
                        f"{DISPLAY_DT:.3f}-m display width"
                    ),
                },
            }
        )

    fig.canvas.draw()
    for row, target in enumerate(TARGET_TIMES):
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        fig.text(
            left.x0,
            max(left.y1, right.y1) + 0.007,
            f"Time = {target:.2f} s",
            ha="left",
            va="bottom",
            fontsize=8.0,
        )

    output_paths = []
    for extension in ("png", "pdf"):
        path = PAPER_FIG / f"{OUTPUT_STEM}.{extension}"
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        output_paths.append(path)
    plt.close(fig)

    manifest = {
        "case": "VW2011 Test 1 Case B",
        "artifact_status": "selected main-text full-domain comparison",
        "figure_claim": (
            "At identical physical times, the current 1-D and archived 2-D results "
            "show the progression from horizontal-interface approach through "
            "standpipe lift to above-rim liquid discharge."
        ),
        "claim_limit": (
            "The archived 2-D solver domain remains laterally confined above the "
            "physical rim; therefore the last row demonstrates only the initial "
            "above-rim response, not a freely developing external plume."
        ),
        "time_pairing": "same physical time; no time shift or event alignment",
        "selected_times_s": list(TARGET_TIMES),
        "one_d_role": "current Tosan-based Present 1D model",
        "two_d_role": "direct archived confined supporting planar VOF result",
        "two_d_geometry": (
            "area-equivalent W=Dt^2/D planar slit with 0.30-m confined numerical "
            f"headroom; display-only lateral remapping to {DISPLAY_DT:.3f} m"
        ),
        "presentation_geometry": {
            "physical_tower_diameter_m": DT,
            "display_tower_width_m": DISPLAY_DT,
            "vertical_scale_changed": False,
            "tower_top": "open; two short wall-coloured strokes denote ground level",
        },
        "solver_results_modified_for_figure": False,
        "panel_layout": "three rows by two columns; panel letters repeated across columns",
        "selected_frames": selected,
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
        "manuscript_status": (
            "selected for the Case-B main text; caption must retain the confined-"
            "headroom and no-free-external-plume limitation"
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
