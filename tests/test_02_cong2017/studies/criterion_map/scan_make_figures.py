# -*- coding: utf-8 -*-
"""Figures for the fully-synchronous rerun of the Series-B set and the
63-configuration criterion map.

  outputs/seriesB_fullsync.png        4-panel model-vs-measured (B-H1..B-H7)
  outputs/criterion_map_fullsync.png  (Dr/D, V*air) plane, model classification
                                      vs the paper's two-parameter criterion
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

C_MEAS = "#2563eb"
C_MODEL = "#d62728"
C_GEYSER = "#d62728"
C_NOGEYSER = "#2563eb"


def _num(row, k):
    v = row.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_seriesB() -> dict:
    rows = load_rows(OUT / "seriesB_fullsync.csv")
    rows.sort(key=lambda r: _num(r, "Dr_mm"))
    x = np.array([_num(r, "Dr_over_D") for r in rows])
    gm = np.array([_num(r, "geyser_model") for r in rows])
    ge = np.array([_num(r, "geyser_meas") for r in rows])
    n_match = int(np.sum(gm == ge))

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.2))

    ax = axes[0]
    ax.plot(x, ge, "s--", color=C_MEAS, ms=9, mfc="none", label="measured (camera)")
    ax.plot(x, gm, "o:", color=C_MODEL, ms=6, label="model (blind)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no geyser", "GEYSER"])
    ax.set_ylim(-0.35, 1.35)
    ax.axvspan(0.52, 0.62, color="#fde68a", alpha=0.5, zorder=0,
               label="measured branch flip 0.52..0.62")
    ax.set_title(f"classification ({n_match}/7 match)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left")

    ax = axes[1]
    ax.plot(x, [_num(r, "Ta_meas_s") for r in rows], "s--", color=C_MEAS, ms=9,
            mfc="none", label="measured")
    ax.plot(x, [_num(r, "Ta_model_s") for r in rows], "o:", color=C_MODEL, ms=6,
            label="model")
    ax.set_ylim(6.0, 11.0)
    ax.set_title("pocket arrival at riser $T_a$ [s]", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(x, [_num(r, "vfs_meas") for r in rows], "s--", color=C_MEAS, ms=9,
            mfc="none", label="measured")
    ax.plot(x, [_num(r, "v_fs_model") for r in rows], "o:", color=C_MODEL, ms=6,
            label="model (max 0.6 s climb)")
    ax.set_ylim(0, 1.6)
    ax.set_title("free-surface rise $v_{fs}$ [m/s]", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3]
    ax.plot(x, [_num(r, "vint_meas") for r in rows], "s--", color=C_MEAS, ms=9,
            mfc="none", label="measured")
    ax.plot(x, [_num(r, "v_int_model") for r in rows], "o:", color=C_MODEL, ms=6,
            label="model (max 0.6 s climb)")
    ax.set_ylim(0, 1.6)
    ax.set_title("gas-front climb $v_{int}$ [m/s]", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("$D_r/D$")
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
    fig.suptitle("Series B (B-H1..B-H7): fully-synchronous frozen solver vs high-speed-camera "
                 "measurements -- $H_0$=0.66 m, $L_0$=0.61 m, only $D_r$ varies", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "seriesB_fullsync.png", dpi=150)
    plt.close(fig)

    Ta_meas = np.array([_num(r, "Ta_meas_s") for r in rows])
    Ta_mod = np.array([_num(r, "Ta_model_s") for r in rows])
    return dict(n_match=n_match, n=len(rows),
                Ta_mae=float(np.nanmean(np.abs(Ta_mod - Ta_meas))))


def fig_criterion() -> dict:
    rows = []
    for p in sorted(OUT.glob("criterion_scan_fullsync*.csv")):
        rows += load_rows(p)
    # de-dup on config key (shards + combined file may overlap)
    seen, uniq = set(), []
    for r in rows:
        k = (r["Dr_mm"], r["L0_m"], r["H0_m"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    if not rows:
        return dict(n=0)

    x = np.array([_num(r, "Dr_over_D") for r in rows])
    y = np.array([_num(r, "Vair_star") for r in rows])
    gm = np.array([_num(r, "geyser_model") for r in rows])
    gc = np.array([_num(r, "criterion_geyser") for r in rows])
    ok = gm == gc
    n_agree = int(np.sum(ok))

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.axvspan(0.0, 0.62, ymin=0, ymax=1, color="#fee2e2", alpha=0.35, zorder=0)
    ax.axhline(3.42, color="#6b7280", ls="--", lw=1.4)
    ax.axvline(0.62, color="#6b7280", ls="--", lw=1.4)
    ax.text(0.33, 3.55, "paper criterion: geyser iff $D_r/D\\leq0.62$ and "
            "$V^*_{air}\\geq3.42$", fontsize=9, color="#374151")

    # jitter overlapping (Dr/D, V*) points from different L0/H0 combos is not
    # needed: V*air already separates them; size by Yfs_max for extra signal
    yfsmax = np.array([_num(r, "Yfs_max_m") for r in rows])
    m = gm == 1
    ax.scatter(x[m], y[m], s=80, marker="^", c=C_GEYSER, label="model: GEYSER",
               edgecolors="k", linewidths=0.5, zorder=3)
    near = (~m) & (yfsmax >= 1.35)
    ax.scatter(x[~m & ~near], y[~m & ~near], s=60, marker="v", c=C_NOGEYSER,
               label="model: no geyser", edgecolors="k", linewidths=0.5, zorder=3)
    if np.any(near):
        ax.scatter(x[near], y[near], s=60, marker="v", c="#93c5fd",
                   label="model: no geyser, near-miss ($Y_{fs,max}\\geq$1.35 m)",
                   edgecolors="k", linewidths=0.5, zorder=3)
    bad = ~ok
    if np.any(bad):
        ax.scatter(x[bad], y[bad], s=240, facecolors="none", edgecolors="#111827",
                   linewidths=1.6, zorder=4,
                   label=f"disagrees with criterion (n={int(np.sum(bad))}, all "
                         "criterion=geyser / model=no)")
        if np.sum(bad) <= 6:
            for xi, yi in zip(x[bad], y[bad]):
                ax.annotate(f"({xi:.2f}, {yi:.2f})", (xi, yi),
                            textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("$D_r/D$")
    ax.set_ylabel("$V^*_{air} = V_{air}/[(\\pi D_r^2/4)\\,H_0]$")
    ax.set_yscale("log")
    ax.set_title(f"63-configuration sweep, fully-synchronous solver: model blind classification "
                 f"vs paper two-parameter criterion -- agreement {n_agree}/{len(rows)}",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "criterion_map_fullsync.png", dpi=150)
    plt.close(fig)
    return dict(n=len(rows), n_agree=n_agree,
                disagree=[(float(xi), float(yi)) for xi, yi in zip(x[bad], y[bad])])


if __name__ == "__main__":
    s = fig_seriesB()
    print("seriesB:", s)
    c = fig_criterion()
    print("criterion:", c)
