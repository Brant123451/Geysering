#!/usr/bin/env python3
"""Render a compact visual QA sheet from the repaired NPZ archives."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "repaired" / "model_1d"
OUTPUT = HERE / "repaired" / "qa" / "campaign2_repaired_keyframes.png"
FRONT_OUTPUT = HERE / "repaired" / "qa" / "campaign2_repaired_wetting_fronts.png"
CASES = (("BH1", 0.016), ("BH3", 0.026), ("BH6", 0.041))
TIMES = (0.0, 0.2, 1.0, 8.0, 13.0)
PIPE_D = 0.050
PIPE_L = 6.59
RISER_X = 3.47
RISER_H = 1.80
BLUE = "#2778d8"
AIR = "#f7f9fc"
WALL = "#263442"


def draw_panel(ax, data, source_index: int, dr: float, label: str, time_s: float) -> None:
    xt = data["xt"]
    zr = data["zr"]
    dx = float(data["dx"][0])
    dz = float(data["dz"][0])
    alt = data["frames_alt"][source_index]
    alr = data["frames_alr"][source_index]
    agr = data["frames_agr"][source_index]

    ax.add_patch(Rectangle((0.0, -PIPE_D), PIPE_L, PIPE_D, facecolor=AIR, edgecolor=WALL, linewidth=0.7))
    for x, liquid in zip(xt, alt):
        fraction = float(np.clip(liquid, 0.0, 1.0))
        if fraction > 0.003:
            ax.add_patch(Rectangle((x - 0.5 * dx, -PIPE_D), dx, PIPE_D * fraction, facecolor=BLUE, edgecolor="none"))

    riser_width = max(dr, 0.035)
    x0 = RISER_X - 0.5 * riser_width
    ax.add_patch(Rectangle((x0, 0.0), riser_width, RISER_H, facecolor=AIR, edgecolor=WALL, linewidth=0.7))
    for z, liquid, gas in zip(zr, alr, agr):
        if liquid <= 0.002:
            continue
        gas_fraction = float(np.clip(gas, 0.0, 0.98))
        if gas_fraction < 0.01:
            ax.add_patch(Rectangle((x0, z - 0.5 * dz), riser_width, dz, facecolor=BLUE, edgecolor="none"))
        else:
            film = 0.5 * (1.0 - gas_fraction) * riser_width
            ax.add_patch(Rectangle((x0, z - 0.5 * dz), film, dz, facecolor=BLUE, edgecolor="none"))
            ax.add_patch(Rectangle((x0 + riser_width - film, z - 0.5 * dz), film, dz, facecolor=BLUE, edgecolor="none"))

    ax.axvline(5.98, ymin=0.0, ymax=0.06, color="#111827", linestyle="--", linewidth=0.8)
    ax.axhline(RISER_H, xmin=0.51, xmax=0.54, color="#d23b31", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.05, PIPE_L + 0.05)
    ax.set_ylim(-0.07, 1.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{label}  t={time_s:.2f} s", fontsize=9)
    ax.set_xticks((0, RISER_X, 5.98, PIPE_L))
    ax.set_yticks((0, 0.61, 1.8))
    ax.tick_params(labelsize=7)


def main() -> None:
    fig, axes = plt.subplots(len(CASES), len(TIMES), figsize=(22, 8.2), constrained_layout=True)
    for row, (case_key, dr) in enumerate(CASES):
        data = np.load(ROOT / case_key / "repaired_1d_frames.npz", allow_pickle=False)
        source_times = np.asarray(data["frames_t"], dtype=float)
        for col, target in enumerate(TIMES):
            index = int(np.argmin(np.abs(source_times - target)))
            draw_panel(axes[row, col], data, index, dr, case_key, float(source_times[index]))
            if col == 0:
                axes[row, col].set_ylabel("height (m)", fontsize=8)
    fig.suptitle("Cong 2017 Campaign 2 - repaired 1D key-frame QA", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)

    front_times = (0.0, 0.2, 0.5, 1.0, 1.5, 2.5)
    fig, axes = plt.subplots(len(CASES), 1, figsize=(11.5, 8.5), sharex=True, constrained_layout=True)
    for axis, (case_key, _dr) in zip(axes, CASES):
        data = np.load(ROOT / case_key / "repaired_1d_frames.npz", allow_pickle=False)
        source_times = np.asarray(data["frames_t"], dtype=float)
        release = data["xt"] > 5.98
        for target in front_times:
            index = int(np.argmin(np.abs(source_times - target)))
            axis.step(
                data["xt"][release],
                data["frames_alt"][index, release],
                where="mid",
                linewidth=1.4,
                label=f"{source_times[index]:.2f} s",
            )
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlim(5.96, 6.61)
        axis.set_ylabel(f"{case_key}\n$A_l/A$")
        axis.grid(alpha=0.22)
        axis.legend(ncol=6, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("x (m), valve at 5.98 m and sealed cap at 6.59 m")
    fig.suptitle("Finite wetting front after the true-dry initialization", fontsize=13)
    fig.savefig(FRONT_OUTPUT, dpi=180)
    plt.close(fig)
    print(FRONT_OUTPUT)


if __name__ == "__main__":
    main()
