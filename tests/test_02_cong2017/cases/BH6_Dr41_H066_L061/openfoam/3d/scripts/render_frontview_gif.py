#!/usr/bin/env python3
"""Render BH6 3-D front-elevation GIFs from riser/plume centreline alpha probes.

Volume fields are usually purged (purgeWrite=3); centreline probes retain the
full 13 s history and are used together with series.csv for Yfs/Yint overlays.
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

PIPE_D = 0.05
RISER_D = 0.041
SOFFIT = 0.025
RIM = 1.825
TEE_X = 3.47
L0 = 0.61
FS0 = 0.635
RISER_Z = np.arange(0.035, 1.815 + 1e-12, 0.020)
PLUME_Z = np.arange(1.835, 3.015 + 1e-12, 0.020)


def load_case(label: str, probe_case: Path, series_path: Path, metrics_path: Path) -> dict:
    riser = np.loadtxt(
        probe_case / "postProcessing/riserCentreline/0/alpha.water", comments="#"
    )
    plume = np.loadtxt(
        probe_case / "postProcessing/plumeCentreline/0/alpha.water", comments="#"
    )
    series = np.genfromtxt(series_path, delimiter=",", names=True)
    metrics = json.loads(metrics_path.read_text())
    return {
        "label": label,
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


def front_x(t: float, ta: float, uf: float) -> float:
    x0 = TEE_X - L0
    return float(np.clip(TEE_X - uf * (ta - t), x0, TEE_X + 0.05))


def make_gif(data: dict, out: Path, fps: int = 10, max_frames: int = 80) -> None:
    ta = data["metrics"]["events"]["Ta_3d_s"]
    uf = data["metrics"]["events"]["horizontal_front_velocity_3d_m_s"]
    ymax = data["metrics"]["events"]["Yfs_max_3d_m"]
    geyser = data["metrics"]["events"]["geyser_3d"]
    cells = data["metrics"]["mesh"]["cells"]
    t_end = 13.0
    key = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, max(0.5, ta - 1.0), 10),
                np.linspace(max(0.0, ta - 1.0), min(t_end, ta + 3.5), 50),
                np.linspace(min(t_end, ta + 3.5), t_end, 20),
            ]
        )
    )
    if len(key) > max_frames:
        key = key[np.linspace(0, len(key) - 1, max_frames).astype(int)]

    fig, ax = plt.subplots(figsize=(6.4, 8.6), dpi=120)
    water = "#3aa7e8"
    air = "#e8eef4"
    wall = "#cfd8e0"

    def alpha_at(t: float) -> tuple[np.ndarray, np.ndarray]:
        i = int(np.argmin(np.abs(data["t"] - t)))
        j = int(np.argmin(np.abs(data["tp"] - t)))
        return np.concatenate([RISER_Z, PLUME_Z]), np.concatenate(
            [data["a"][i], data["ap"][j]]
        )

    def draw(frame_i: int):
        ax.clear()
        ax.set_facecolor("#0b1c2c")
        fig.patch.set_facecolor("#0b1c2c")
        t = float(key[frame_i])
        z, alpha = alpha_at(t)
        series = interp_series(data["series"], t)
        yfs = series["Yfs_3d_m"]
        yint = series["Yint_3d_m"]
        xn = front_x(t, ta, uf)
        x0, x1 = TEE_X - 1.2, TEE_X + 1.35

        ax.add_patch(
            Rectangle(
                (x0, -PIPE_D / 2),
                x1 - x0,
                PIPE_D,
                lw=1.5,
                ec=wall,
                fc="#152536",
                zorder=1,
            )
        )
        if t < ta:
            ax.add_patch(
                Rectangle(
                    (x0, -PIPE_D / 2 * 0.92),
                    max(0.0, xn - x0),
                    PIPE_D * 0.92,
                    lw=0,
                    fc=water,
                    alpha=0.85,
                    zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (xn, -PIPE_D / 2 * 0.92),
                    max(0.0, x1 - xn),
                    PIPE_D * 0.92,
                    lw=0,
                    fc=air,
                    alpha=0.35,
                    zorder=2,
                )
            )
            ax.plot(
                [xn, xn],
                [-PIPE_D / 2, PIPE_D / 2],
                color="#ffcc66",
                lw=2,
                zorder=5,
            )
            ax.text(
                xn,
                PIPE_D / 2 + 0.05,
                "air front",
                color="#ffcc66",
                ha="center",
                fontsize=8,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x0, -PIPE_D / 2 * 0.92),
                    x1 - x0,
                    PIPE_D * 0.92,
                    lw=0,
                    fc=water,
                    alpha=0.55,
                    zorder=2,
                )
            )

        ax.add_patch(
            Rectangle(
                (TEE_X - RISER_D / 2, SOFFIT),
                RISER_D,
                RIM - SOFFIT,
                lw=1.6,
                ec=wall,
                fc="#101c28",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (TEE_X - 0.16, RIM),
                0.32,
                2.45 - RIM,
                lw=0.8,
                ec="#4a6074",
                fc="#0e2030",
                alpha=0.45,
                zorder=2,
            )
        )
        for zi, ai in zip(z, alpha):
            if ai < 0.05 or zi > 2.45:
                continue
            half = (RISER_D / 2) * 0.92 * (0.35 + 0.65 * ai)
            color = water if zi <= RIM else "#7ec8ff"
            ax.add_patch(
                Rectangle(
                    (TEE_X - half, zi - 0.01),
                    2 * half,
                    0.02,
                    lw=0,
                    fc=color,
                    alpha=0.25 + 0.75 * ai,
                    zorder=4,
                )
            )
        if np.isfinite(yfs) and yfs > 0:
            ax.plot(
                [TEE_X - RISER_D * 0.9, TEE_X + RISER_D * 0.9],
                [yfs, yfs],
                color="#ffe08a",
                lw=2.3,
                zorder=6,
            )
            ax.text(
                TEE_X + RISER_D * 0.95,
                yfs,
                f"Yfs={yfs:.2f}m",
                color="#ffe08a",
                va="center",
                fontsize=8,
            )
        if np.isfinite(yint) and yint > 0:
            ax.plot(
                [TEE_X - RISER_D * 0.75, TEE_X + RISER_D * 0.75],
                [yint, yint],
                color="#ff7a59",
                lw=1.7,
                ls="--",
                zorder=6,
            )
        ax.plot(
            [TEE_X - 0.12, TEE_X + 0.12],
            [RIM, RIM],
            color="#9ad1ff",
            lw=1.2,
            ls=":",
            zorder=6,
        )
        ax.text(TEE_X + 0.13, RIM, "rim", color="#9ad1ff", va="center", fontsize=8)
        ax.plot(
            [TEE_X - RISER_D / 2, TEE_X + RISER_D / 2],
            [FS0, FS0],
            color="#6f879c",
            lw=0.8,
            ls=":",
            zorder=5,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.12, 2.45)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)  front elevation at tee", color="#c5d3e0", fontsize=9)
        ax.set_ylabel("z (m)", color="#c5d3e0", fontsize=9)
        ax.tick_params(colors="#9fb0c0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#3a5166")
        status = "NO GEYSER" if not geyser else "GEYSER"
        ax.set_title(
            f"BH6 3D OpenFOAM — {data['label']}   t = {t:5.2f} s\n"
            f"cells={cells}   Ta={ta:.2f}s   Yfs={yfs:.3f}m "
            f"(max {ymax:.3f}m)   {status}",
            color="white",
            fontsize=11,
            pad=10,
        )
        ax.text(
            0.02,
            0.98,
            "Front view through tee midplane\n"
            "blue=water  yellow=Yfs  orange dashed=Yint",
            transform=ax.transAxes,
            va="top",
            color="#d5e2ee",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35", fc="#132333", ec="#3a5166", alpha=0.9
            ),
        )
        return []

    anim = FuncAnimation(fig, draw, frames=len(key), interval=100)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)

    timeseries = data["series"]["time_s"]
    yfs_series = data["series"]["Yfs_3d_m"]
    peak_t = float(timeseries[int(np.nanargmax(yfs_series))])
    for tag, tt in (("peak", peak_t), ("arrival", float(ta)), ("t0", 0.0)):
        key = np.array([tt])
        fig_s, ax_s = plt.subplots(figsize=(6.4, 8.6), dpi=130)
        # Rebind draw targets for a one-off still.
        ax = ax_s
        fig = fig_s
        draw(0)
        still = out.with_name(f"{out.stem}_{tag}.png")
        fig_s.savefig(still, facecolor=fig_s.get_facecolor(), bbox_inches="tight")
        plt.close(fig_s)


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
    parser.add_argument("--copy-artifacts", type=Path)
    args = parser.parse_args()

    outdir = args.results_root / "animations"
    cases = []
    if args.refined_case.exists():
        cases.append(
            (
                load_case(
                    "refined",
                    args.refined_case,
                    args.results_root / "refined/series.csv",
                    args.results_root / "refined/metrics.json",
                ),
                outdir / "BH6_refined_frontview.gif",
            )
        )
    if args.wallbl_case.exists():
        cases.append(
            (
                load_case(
                    "wall-bl-v6",
                    args.wallbl_case,
                    args.results_root / "wall-bl-v6/series.csv",
                    args.results_root / "wall-bl-v6/metrics.json",
                ),
                outdir / "BH6_wallbl_v6_frontview.gif",
            )
        )
    if not cases:
        raise SystemExit("No probe cases found for rendering")

    for data, out in cases:
        make_gif(data, out)
        print(f"wrote {out}")
        if args.copy_artifacts:
            args.copy_artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, args.copy_artifacts / out.name)
            for still in outdir.glob(f"{out.stem}_*.png"):
                shutil.copy2(still, args.copy_artifacts / still.name)


if __name__ == "__main__":
    main()
