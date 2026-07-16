# -*- coding: utf-8 -*-
"""Paper-quality Campaign 2 figures (fully-synchronous solver), written
directly into the paper's figures/ directory:

  cong2017_bh_series.png     4-panel Series-B model vs measured
  cong2017_criterion_map.png (Dr/D, V*air) plane, 63 configs vs criterion
  cong2017_signature.png     3-panel signature runs (B-H1 / B-H6 / pressures)
                             from the caseA/caseB frozen-solver series CSVs
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
TEST_ROOT = HERE.parents[1]
REPO_ROOT = TEST_ROOT.parents[1]
CASES = TEST_ROOT / "cases"
FIGDIR = REPO_ROOT / "paper" / "figures"

C_MEAS = "#2563eb"
C_MODEL = "#d62728"
H0 = 0.66
HR = 1.8

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def _num(row, k):
    v = row.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_series(path: Path) -> dict:
    rows = load_rows(path)
    return {k: np.array([_num(r, k) for r in rows])
            for k in ("t_s", "Yfs_m", "Yint_m", "pocket_head_m")}


def fig_bh_series() -> None:
    rows = load_rows(OUT / "seriesB_fullsync.csv")
    rows.sort(key=lambda r: _num(r, "Dr_mm"))
    x = np.array([_num(r, "Dr_over_D") for r in rows])
    gm = np.array([_num(r, "geyser_model") for r in rows])
    ge = np.array([_num(r, "geyser_meas") for r in rows])
    n_match = int(np.sum(gm == ge))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    (a, b), (c, d) = axes

    a.plot(x, ge, "s--", color=C_MEAS, ms=9, mfc="none", label="measured")
    a.plot(x, gm, "o:", color=C_MODEL, ms=6, label="model")
    a.axvspan(0.52, 0.62, color="#fde68a", alpha=0.55, zorder=0)
    a.set_yticks([0, 1])
    a.set_yticklabels(["no geyser", "geyser"])
    a.set_ylim(-0.3, 1.3)
    a.set_title(f"(a) classification ({n_match}/7 runs)")
    a.legend(frameon=False, loc="center left")

    b.plot(x, [_num(r, "Ta_meas_s") for r in rows], "s--", color=C_MEAS, ms=9,
           mfc="none", label="measured")
    b.plot(x, [_num(r, "Ta_model_s") for r in rows], "o:", color=C_MODEL, ms=6,
           label="model")
    b.set_ylim(6, 12)
    b.set_ylabel("$T_a$ (s)")
    b.set_title("(b) pocket arrival time at the riser")
    b.legend(frameon=False, loc="upper left")

    c.plot(x, [_num(r, "vfs_meas") for r in rows], "s--", color=C_MEAS, ms=9,
           mfc="none", label="measured")
    c.plot(x, [_num(r, "v_fs_model") for r in rows], "o:", color=C_MODEL, ms=6,
           label="model")
    c.set_ylim(0, 1.9)
    c.set_ylabel("$v_{fs}$ (m/s)")
    c.set_title("(c) free-surface rise speed")
    c.legend(frameon=False)

    d.plot(x, [_num(r, "vint_meas") for r in rows], "s--", color=C_MEAS, ms=9,
           mfc="none", label="measured")
    d.plot(x, [_num(r, "v_int_model") for r in rows], "o:", color=C_MODEL, ms=6,
           label="model")
    d.set_ylim(0, 1.9)
    d.set_ylabel("$v_{int}$ (m/s)")
    d.set_title("(d) gas-nose rise speed")
    d.legend(frameon=False)

    for ax in axes.flat:
        ax.set_xlabel("$D_r/D$")
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cong2017_bh_series.png", dpi=200)
    plt.close(fig)
    print(f"-> {FIGDIR / 'cong2017_bh_series.png'} ({n_match}/7)")


def fig_criterion_map() -> None:
    rows = []
    for p in sorted(OUT.glob("criterion_scan_fullsync_*.csv")):
        rows += load_rows(p)
    seen, uniq = set(), []
    for r in rows:
        k = (r["Dr_mm"], r["L0_m"], r["H0_m"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    x = np.array([_num(r, "Dr_over_D") for r in rows])
    y = np.array([_num(r, "Vair_star") for r in rows])
    gm = np.array([_num(r, "geyser_model") for r in rows])
    gc = np.array([_num(r, "criterion_geyser") for r in rows])
    yfs = np.array([_num(r, "Yfs_max_m") for r in rows])
    agree = gm == gc
    n_agree = int(np.sum(agree))

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.axhline(3.42, color="#6b7280", ls="--", lw=1.3)
    ax.axvline(0.62, color="#6b7280", ls="--", lw=1.3)
    m = gm == 1
    near = (~m) & (yfs >= 1.35)
    ax.scatter(x[m], y[m], s=85, marker="^", c=C_MODEL, edgecolors="k",
               linewidths=0.5, zorder=3, label="model: geyser")
    ax.scatter(x[~m & ~near], y[~m & ~near], s=62, marker="o", facecolors="none",
               edgecolors=C_MEAS, linewidths=1.2, zorder=3,
               label="model: no geyser")
    ax.scatter(x[near], y[near], s=62, marker="o", facecolors="#bfdbfe",
               edgecolors=C_MEAS, linewidths=1.2, zorder=3,
               label="model: no geyser, $Y_{fs,\\max}\\!\\geq\\!1.35$ m")
    bad = ~agree
    ax.scatter(x[bad], y[bad], s=230, facecolors="none", edgecolors="#111827",
               linewidths=1.4, zorder=4, label="differs from criterion")
    ax.set_xlabel("$D_r/D$")
    ax.set_ylabel("$V^{*}_{air}=V_{air}/[(\\pi D_r^2/4)H_0]$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "cong2017_criterion_map.png", dpi=200)
    plt.close(fig)
    print(f"-> {FIGDIR / 'cong2017_criterion_map.png'} ({n_agree}/{len(rows)})")


def fig_signature() -> None:
    sA = load_series(CASES / "BH1_Dr16_H066_L061" / "outputs"
                     / "caseA_model_series.csv")
    sB = load_series(CASES / "BH6_Dr41_H066_L061" / "outputs"
                     / "caseB_model_series.csv")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.6, 4.1))

    for ax, s, ttl in ((a, sA, "(a) B-H1 ($D_r=16$ mm): geyser"),
                       (b, sB, "(b) B-H6 ($D_r=41$ mm): no geyser")):
        ax.plot(s["t_s"], s["Yfs_m"], color=C_MODEL, lw=1.9,
                label="free surface $Y_{fs}$")
        ax.plot(s["t_s"], s["Yint_m"], color="#f59e0b", lw=1.7, ls="--",
                label="gas nose $Y_{int}$")
        ax.axhline(HR, color="#16a34a", ls=":", lw=1.2)
        ax.text(0.35, HR + 0.03, "riser top", color="#16a34a", fontsize=8)
        ax.axhline(H0, color="0.6", ls=":", lw=1.0)
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 1.95)
        ax.set_xlabel("$t$ (s)")
        ax.set_ylabel("$Y$ (m)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper left")

    win = 0.5

    def cyc_avg(t, h):
        out = np.full_like(h, np.nan)
        for i in range(len(h)):
            m = (t >= t[i] - win / 2) & (t <= t[i] + win / 2)
            if np.any(np.isfinite(h[m])):
                out[i] = np.nanmean(h[m])
        return out

    c.plot(sA["t_s"], cyc_avg(sA["t_s"], sA["pocket_head_m"]) / H0,
           color=C_MODEL, lw=1.9, label="B-H1 (geyser)")
    c.plot(sB["t_s"], cyc_avg(sB["t_s"], sB["pocket_head_m"]) / H0,
           color=C_MEAS, lw=1.9, label="B-H6 (no geyser)")
    c.axhline(1.9, color=C_MODEL, ls=":", lw=1.2)
    c.text(0.35, 1.93, "reported geyser surge $\\approx1.9H_0$",
           color=C_MODEL, fontsize=8)
    c.axhline(1.4, color=C_MEAS, ls=":", lw=1.2)
    c.text(0.35, 1.43, "reported no-geyser hump $\\approx1.4H_0$",
           color=C_MEAS, fontsize=8)
    c.set_xlim(0, 13)
    c.set_ylim(0, 2.2)
    c.set_xlabel("$t$ (s)")
    c.set_ylabel("$H/H_0$")
    c.set_title("(c) pocket pressure head (0.5 s average)")
    c.grid(alpha=0.3)
    c.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGDIR / "cong2017_signature.png", dpi=200)
    plt.close(fig)
    print(f"-> {FIGDIR / 'cong2017_signature.png'}")


if __name__ == "__main__":
    fig_bh_series()
    fig_criterion_map()
    fig_signature()
