"""Render an existing raw Case-A solver archive without rerunning the model.

The vertical image is built from the conserved per-cell liquid area.  It does
not substitute the mixed-phase envelope, tracer front, or a smoothed curve for
the liquid field.  Partial vertical cells are shown as area-preserving annular
films around a centred gas/air core.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
MODEL = CASE / "model"
OUTPUTS = CASE / "outputs"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from vw2011_network_twofluid import _depth_frac  # noqa: E402


PIPE_LENGTH = 4.006
PIPE_DIAMETER = 0.094
RISER_DIAMETER = 0.0571
RISER_HEIGHT = 0.610
RISER_X = 3.516
INITIAL_RISER_LEVEL = 0.356
WATER = "#2b7fff"
AIR = "#f2f4f8"


def _diagnostic_at(
    diagnostics: dict[str, object], name: str, index: int, default: float = 0.0
) -> float:
    values = diagnostics.get(name)
    if isinstance(values, list) and index < len(values):
        return float(values[index])
    return float(default)


def _draw_riser(
    ax: plt.Axes,
    *,
    x0: float,
    width: float,
    z: np.ndarray,
    dz: float,
    liquid_fraction: np.ndarray,
) -> None:
    liquid = np.clip(np.asarray(liquid_fraction, dtype=float), 0.0, 1.0)
    for centre, alpha_l in zip(z, liquid, strict=True):
        if alpha_l <= 1.0e-6:
            continue
        lower = max(float(centre - 0.5 * dz), 0.0)
        upper = min(float(centre + 0.5 * dz), RISER_HEIGHT)
        if upper <= lower:
            continue
        ax.add_patch(
            Rectangle(
                (x0, lower), width, upper - lower,
                facecolor=WATER, edgecolor="none",
            )
        )
        if alpha_l < 1.0 - 1.0e-6:
            core_width = math.sqrt(1.0 - float(alpha_l)) * width
            ax.add_patch(
                Rectangle(
                    (x0 + 0.5 * (width - core_width), lower),
                    core_width,
                    upper - lower,
                    facecolor=AIR,
                    edgecolor="none",
                )
            )


def render(
    fields_path: Path,
    diagnostics_path: Path,
    *,
    variant: str,
) -> Path:
    with np.load(fields_path) as saved:
        time = np.asarray(saved["time"], dtype=float)
        z = np.asarray(saved["z"], dtype=float)
        vertical_liquid = np.asarray(saved["alpha_l"], dtype=float)
        horizontal_liquid = np.asarray(
            saved["horizontal_alpha_l"], dtype=float
        )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if vertical_liquid.shape[0] != time.size:
        raise ValueError("vertical archive has an inconsistent time dimension")
    if horizontal_liquid.shape[0] != time.size:
        raise ValueError("horizontal archive has an inconsistent time dimension")
    if z.size < 2:
        raise ValueError("vertical grid must have at least two cells")

    dz = float(np.median(np.diff(z)))
    horizontal_count = horizontal_liquid.shape[1]
    dx = PIPE_LENGTH / horizontal_count
    x = (np.arange(horizontal_count, dtype=float) + 0.5) * dx
    initial_grid_height = float(np.sum(vertical_liquid[0]) * dz)
    initial_height_offset = initial_grid_height - INITIAL_RISER_LEVEL

    frames_dir = OUTPUTS / f"frames_{variant}"
    riser_dir = OUTPUTS / f"riser_frames_{variant}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    riser_dir.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUTS / f"frames_index_{variant}.json"
    wall = dict(color="0.35", lw=0.8, zorder=10)
    handles = [
        Patch(facecolor=WATER, label="water"),
        Patch(facecolor=AIR, edgecolor="0.5", label="air"),
    ]
    riser_left = RISER_X - 0.5 * RISER_DIAMETER
    riser_right = RISER_X + 0.5 * RISER_DIAMETER
    manifest: list[dict[str, object]] = []

    for index, time_value in enumerate(time):
        horizontal_alpha = np.clip(horizontal_liquid[index], 0.0, 1.0)
        horizontal_depth = PIPE_DIAMETER * _depth_frac(horizontal_alpha)
        surface_x = np.r_[0.0, x, PIPE_LENGTH]
        surface_y = -PIPE_DIAMETER + np.r_[
            horizontal_depth[0], horizontal_depth, horizontal_depth[-1]
        ]
        alpha_vertical = np.clip(vertical_liquid[index], 0.0, 1.0)
        liquid_equivalent_height = max(
            float(np.sum(alpha_vertical) * dz) - initial_height_offset,
            0.0,
        )
        visible = np.flatnonzero(alpha_vertical > 1.0e-3)
        visible_top = (
            min(float(z[visible[-1]] + 0.5 * dz), RISER_HEIGHT)
            if visible.size
            else 0.0
        )
        material_height = _diagnostic_at(
            diagnostics, "wtop", index, visible_top
        )
        tracer_top = _diagnostic_at(diagnostics, "itop", index)

        figure, ax = plt.subplots(figsize=(14.0, 3.6))
        ax.add_patch(
            Rectangle(
                (0.0, -PIPE_DIAMETER), PIPE_LENGTH, PIPE_DIAMETER,
                facecolor=AIR, edgecolor="none",
            )
        )
        ax.fill_between(
            surface_x,
            -PIPE_DIAMETER,
            surface_y,
            where=surface_y > -PIPE_DIAMETER + 1.0e-12,
            interpolate=True,
            facecolor=WATER,
            edgecolor="none",
        )
        ax.add_patch(
            Rectangle(
                (riser_left, 0.0), RISER_DIAMETER, RISER_HEIGHT,
                facecolor=AIR, edgecolor="none",
            )
        )
        _draw_riser(
            ax,
            x0=riser_left,
            width=RISER_DIAMETER,
            z=z,
            dz=dz,
            liquid_fraction=alpha_vertical,
        )
        ax.plot([0, PIPE_LENGTH], [-PIPE_DIAMETER, -PIPE_DIAMETER], **wall)
        ax.plot([0, 0], [-PIPE_DIAMETER, 0], **wall)
        ax.plot([PIPE_LENGTH, PIPE_LENGTH], [-PIPE_DIAMETER, 0], **wall)
        ax.plot([0, riser_left], [0, 0], **wall)
        ax.plot([riser_right, PIPE_LENGTH], [0, 0], **wall)
        ax.plot([riser_left, riser_left], [0, RISER_HEIGHT], **wall)
        ax.plot([riser_right, riser_right], [0, RISER_HEIGHT], **wall)
        ax.text(
            0.01,
            0.95,
            f"Time = {time_value:.2f} s    "
            f"riser liquid-equivalent height = {liquid_equivalent_height:.3f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xlim(-0.05, PIPE_LENGTH + 0.05)
        ax.set_ylim(-PIPE_DIAMETER - 0.04, RISER_HEIGHT + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title("Case A: conservative present 1D model", fontsize=10)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        figure.tight_layout()
        full_path = frames_dir / f"frame_{index:04d}.png"
        figure.savefig(full_path, dpi=130)
        plt.close(figure)

        figure, ax = plt.subplots(figsize=(2.6, 6.2))
        ax.add_patch(
            Rectangle(
                (0.0, 0.0), 1.0, RISER_HEIGHT,
                facecolor=AIR, edgecolor="none",
            )
        )
        _draw_riser(
            ax,
            x0=0.0,
            width=1.0,
            z=z,
            dz=dz,
            liquid_fraction=alpha_vertical,
        )
        ax.plot([0, 0], [0, RISER_HEIGHT], **wall)
        ax.plot([1, 1], [0, RISER_HEIGHT], **wall)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, RISER_HEIGHT)
        ax.set_xticks([])
        ax.set_ylabel("height above pipe crown [m]", fontsize=8)
        ax.set_title(f"tower zoom\nTime = {time_value:.2f} s", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        figure.tight_layout()
        zoom_path = riser_dir / f"riser_{index:04d}.png"
        figure.savefig(zoom_path, dpi=110)
        plt.close(figure)

        manifest.append(
            {
                "file": full_path.relative_to(CASE).as_posix(),
                "riserFile": zoom_path.relative_to(CASE).as_posix(),
                "time": round(float(time_value), 6),
                "wtop": round(liquid_equivalent_height, 6),
                "materialHeight": round(material_height, 6),
                "visibleWaterTop": round(visible_top, 6),
                "itop": round(tracer_top, 6),
                "coreMassMg": round(
                    1.0e6 * _diagnostic_at(diagnostics, "core_mass", index), 3
                ),
                "head": round(
                    _diagnostic_at(diagnostics, "pocket_head", index), 6
                ),
            }
        )
        if index % 25 == 0 or index + 1 == time.size:
            print(f"rendered {index + 1}/{time.size}", flush=True)

    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(index_path)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    render(args.fields, args.diagnostics, variant=args.variant)


if __name__ == "__main__":
    main()
