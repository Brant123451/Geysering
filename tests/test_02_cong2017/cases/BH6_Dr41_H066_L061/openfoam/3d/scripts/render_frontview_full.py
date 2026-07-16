#!/usr/bin/env python3
"""Full-apparatus front-elevation GIF: water/air only, no annotation markers.

Geometry matches Cong 2017 B-H6 paper audit (6.59 m pipe, tee 3.47 m,
valve/pocket 5.98-6.59 m, rim z=1.825 m). Horizontal air-front kinematics
use measured Ta and Uf; riser phases come from centreline alpha probes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

PIPE_D = 0.050
RISER_D = 0.041
PIPE_LEN = 6.590
TEE_X = 3.470
VALVE_X = 5.980
INVERT_Z = -0.025
SOFFIT_Z = 0.025
RIM_Z = 1.825
EXT_TOP_Z = 3.025
PIPE_R = PIPE_D / 2.0
RISER_R = RISER_D / 2.0
A_PIPE = np.pi * PIPE_R * PIPE_R

RISER_Z = np.arange(0.035, 1.815 + 1e-12, 0.020)
PLUME_Z = np.arange(1.835, 3.015 + 1e-12, 0.020)

WATER = "#2f7fdb"
AIR = "#dfe7ef"
WALL = "#1a2430"
BG = "#0a1520"


def load_case(probe_case: Path, series_path: Path, metrics_path: Path) -> dict:
    riser = np.loadtxt(
        probe_case / "postProcessing/riserCentreline/0/alpha.water", comments="#"
    )
    plume = np.loadtxt(
        probe_case / "postProcessing/plumeCentreline/0/alpha.water", comments="#"
    )
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    metrics = json.loads(metrics_path.read_text())
    return {
        "t": riser[:, 0],
        "a": np.clip(riser[:, 1:], 0.0, 1.0),
        "tp": plume[:, 0],
        "ap": np.clip(plume[:, 1:], 0.0, 1.0),
        "series": series,
        "metrics": metrics,
    }


def interp_series(series: np.ndarray, t: float) -> dict[str, float]:
    ts = series["time_s"]
    return {
        name: float(np.interp(t, ts, series[name]))
        for name in series.dtype.names
        if name != "time_s"
    }


def air_nose_x(t: float, ta: float, uf: float) -> float:
    """Leading air/water interface in the main pipe (moves upstream to the tee)."""
    if uf <= 0.0 or not np.isfinite(ta):
        return VALVE_X
    # t=0 → near valve; t=Ta → tee
    return float(np.clip(TEE_X + uf * (ta - t), TEE_X, PIPE_LEN))


def alpha_column(data: dict, t: float) -> tuple[np.ndarray, np.ndarray]:
    i = int(np.argmin(np.abs(data["t"] - t)))
    j = int(np.argmin(np.abs(data["tp"] - t)))
    z = np.concatenate([RISER_Z, PLUME_Z])
    a = np.concatenate([data["a"][i], data["ap"][j]])
    return z, a


def frame_times(ta: float, end: float = 13.0, n: int = 130) -> np.ndarray:
    # denser samples while the pocket travels and while the riser rises
    parts = [
        np.linspace(0.0, max(0.2, ta - 0.8), 28),
        np.linspace(max(0.0, ta - 0.8), min(end, ta + 3.2), 70),
        np.linspace(min(end, ta + 3.2), end, 32),
    ]
    key = np.unique(np.concatenate(parts))
    if len(key) > n:
        key = key[np.linspace(0, len(key) - 1, n).astype(int)]
    return key


def draw_phase_frame(
    ax,
    data: dict,
    t: float,
    *,
    show_external: bool = True,
) -> None:
    ax.clear()
    ax.set_facecolor(BG)
    ta = float(data["metrics"]["events"]["Ta_3d_s"])
    uf = float(data["metrics"]["events"]["horizontal_front_velocity_3d_m_s"])
    series = interp_series(data["series"], t)
    z_col, a_col = alpha_column(data, t)

    z_top = EXT_TOP_Z if show_external else (RIM_Z + 0.05)
    x0, x1 = -0.05, PIPE_LEN + 0.05
    z0, z1 = INVERT_Z - 0.04, z_top + 0.04

    # --- main pipe cavity ---
    ax.add_patch(
        Rectangle(
            (0.0, -PIPE_R),
            PIPE_LEN,
            PIPE_D,
            lw=0.0,
            fc=AIR,
            zorder=1,
        )
    )
    nose = air_nose_x(t, ta, uf)
    if t <= ta:
        # water upstream of advancing air nose; air from nose to cap
        ax.add_patch(
            Rectangle(
                (0.0, -PIPE_R * 0.98),
                max(0.0, nose),
                PIPE_D * 0.98,
                lw=0.0,
                fc=WATER,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (nose, -PIPE_R * 0.98),
                max(0.0, PIPE_LEN - nose),
                PIPE_D * 0.98,
                lw=0.0,
                fc=AIR,
                zorder=2,
            )
        )
    else:
        # after arrival: remaining downstream pocket volume → air length at the cap
        v_down = max(0.0, series["downstream_air_volume_m3"])
        air_len = float(np.clip(v_down / A_PIPE, 0.0, PIPE_LEN - TEE_X))
        water_end = PIPE_LEN - air_len
        ax.add_patch(
            Rectangle(
                (0.0, -PIPE_R * 0.98),
                max(0.0, water_end),
                PIPE_D * 0.98,
                lw=0.0,
                fc=WATER,
                zorder=2,
            )
        )
        if air_len > 1e-4:
            ax.add_patch(
                Rectangle(
                    (water_end, -PIPE_R * 0.98),
                    air_len,
                    PIPE_D * 0.98,
                    lw=0.0,
                    fc=AIR,
                    zorder=2,
                )
            )

    # --- riser / external cavity background (air) ---
    ax.add_patch(
        Rectangle(
            (TEE_X - RISER_R, SOFFIT_Z),
            RISER_D,
            RIM_Z - SOFFIT_Z,
            lw=0.0,
            fc=AIR,
            zorder=3,
        )
    )
    if show_external:
        ax.add_patch(
            Rectangle(
                (TEE_X - 0.12, RIM_Z),
                0.24,
                EXT_TOP_Z - RIM_Z,
                lw=0.0,
                fc=AIR,
                zorder=3,
            )
        )

    # paint contiguous water segments from centreline alpha (absolute z)
    wet = (a_col >= 0.5) & (z_col >= SOFFIT_Z) & (z_col <= z_top)
    if np.any(wet):
        idx = np.where(wet)[0]
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for i0, i1 in zip(starts, ends):
            z_lo = float(z_col[i0] - 0.010)
            z_hi = float(z_col[i1] + 0.010)
            # split at rim so external width can differ
            segments = []
            if z_lo < RIM_Z:
                segments.append((z_lo, min(z_hi, RIM_Z), RISER_R * 0.98))
            if show_external and z_hi > RIM_Z:
                segments.append((max(z_lo, RIM_Z), z_hi, 0.12))
            for za, zb, half in segments:
                if zb <= za:
                    continue
                ax.add_patch(
                    Rectangle(
                        (TEE_X - half, za),
                        2.0 * half,
                        zb - za,
                        lw=0.0,
                        fc=WATER,
                        zorder=4,
                    )
                )

    # subtle wall outlines only (no symbols / text)
    ax.add_patch(
        Rectangle(
            (0.0, -PIPE_R),
            PIPE_LEN,
            PIPE_D,
            lw=1.0,
            ec=WALL,
            fill=False,
            zorder=6,
        )
    )
    ax.add_patch(
        Rectangle(
            (TEE_X - RISER_R, SOFFIT_Z),
            RISER_D,
            RIM_Z - SOFFIT_Z,
            lw=1.0,
            ec=WALL,
            fill=False,
            zorder=6,
        )
    )
    if show_external:
        ax.add_patch(
            Rectangle(
                (TEE_X - 0.12, RIM_Z),
                0.24,
                EXT_TOP_Z - RIM_Z,
                lw=0.8,
                ec=WALL,
                fill=False,
                zorder=6,
            )
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")
    ax.axis("off")


def render_gif(
    data: dict,
    out_gif: Path,
    *,
    fps: int = 12,
    n_frames: int = 130,
    dpi: int = 120,
    show_external: bool = True,
) -> Path:
    ta = float(data["metrics"]["events"]["Ta_3d_s"])
    times = frame_times(ta, n=n_frames)
    z_top = EXT_TOP_Z if show_external else (RIM_Z + 0.05)
    width_m = PIPE_LEN + 0.10
    height_m = (z_top - INVERT_Z) + 0.08
    # keep physical aspect; scale figure to a readable width
    # True geometric aspect makes D=50 mm thin on a 6.59 m pipe; enlarge
    # figure and dpi so the full apparatus remains readable.
    fig_w = 22.0
    fig_h = max(5.0, fig_w * height_m / width_m)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    def update(i: int):
        draw_phase_frame(ax, data, float(times[i]), show_external=show_external)
        return []

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(out_gif, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)

    # stills at t=0, Ta, peak Yfs — still no markers
    series = data["series"]
    peak_t = float(series["time_s"][int(np.nanargmax(series["Yfs_3d_m"]))])
    for tag, tt in (("t0", 0.0), ("arrival", ta), ("peak", peak_t), ("end", 13.0)):
        fig_s, ax_s = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig_s.patch.set_facecolor(BG)
        draw_phase_frame(ax_s, data, tt, show_external=show_external)
        still = out_gif.with_name(f"{out_gif.stem}_{tag}.png")
        fig_s.savefig(still, facecolor=fig_s.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_s)
    return out_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--refined-case",
        type=Path,
        default=Path("/tmp/bh6-refined-work/refined"),
    )
    parser.add_argument(
        "--wallbl-case",
        type=Path,
        default=Path("/tmp/bh6-wall-bl-full"),
    )
    parser.add_argument("--profile", choices=("refined", "wall-bl-v6", "both"), default="both")
    parser.add_argument("--copy-artifacts", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=130)
    args = parser.parse_args()

    outdir = args.results_root / "animations"
    jobs: list[tuple[str, Path, Path, Path]] = []
    if args.profile in ("refined", "both") and args.refined_case.exists():
        jobs.append(
            (
                "refined",
                args.refined_case,
                args.results_root / "refined/series.csv",
                args.results_root / "refined/metrics.json",
            )
        )
    if args.profile in ("wall-bl-v6", "both") and args.wallbl_case.exists():
        jobs.append(
            (
                "wall-bl-v6",
                args.wallbl_case,
                args.results_root / "wall-bl-v6/series.csv",
                args.results_root / "wall-bl-v6/metrics.json",
            )
        )
    if not jobs:
        raise SystemExit("No probe cases available")

    name_map = {
        "refined": "BH6_refined_frontview_full.gif",
        "wall-bl-v6": "BH6_wallbl_v6_frontview_full.gif",
    }
    for label, probe, series, metrics in jobs:
        data = load_case(probe, series, metrics)
        out = outdir / name_map[label]
        render_gif(data, out, fps=args.fps, n_frames=args.frames)
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
        if args.copy_artifacts:
            args.copy_artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, args.copy_artifacts / out.name)
            for still in outdir.glob(f"{out.stem}_*.png"):
                shutil.copy2(still, args.copy_artifacts / still.name)


if __name__ == "__main__":
    main()
