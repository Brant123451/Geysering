# -*- coding: utf-8 -*-
"""Build publication-oriented B-H6 evidence and 1D/2D comparison assets.

The script is intentionally read-only with respect to the simulations: it
uses the archived geometry-matched 1D series, the completed OpenFOAM 2D
post-processing series, and digitized data from Cong et al. (2017).
No time shift or curve fitting is applied.  A short running median is used
only for the dense 2D traces and is stated in the figure captions/plan.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve()
CASE = HERE.parent.parent
REPO = next(p for p in CASE.parents if (p / "paper").is_dir() and (p / "references").is_dir())
OUT = CASE / "outputs" / "manuscript_comparison"
OUT.mkdir(parents=True, exist_ok=True)

H0 = 0.66
RISER_HEIGHT = 1.8

EXP_LEVELS = CASE / "data" / "digitized" / "fig7a_levels.csv"
EXP_PRESSURE = CASE / "data" / "digitized" / "fig10b_pt1.csv"
ONE_D = CASE / "outputs" / "paper_layout_1d" / "caseB_model_series.csv"
ONE_D_METRICS = CASE / "outputs" / "paper_layout_1d" / "caseB_comparison_metrics.json"
TWO_D = CASE / "openfoam" / "2d" / "results" / "openfoam_2d_riser_series.csv"
TWO_D_PRESSURE = CASE / "openfoam" / "2d" / "results" / "openfoam_2d_pt1_series.csv"
TWO_D_METRICS = CASE / "openfoam" / "2d" / "results" / "openfoam_2d_metrics.json"
SOURCE_PDF = REPO / "references" / "cong2017.pdf"
VIEWER = CASE / "outputs" / "1d2d_viewer"


def running_median(values: pd.Series, samples: int) -> pd.Series:
    """Centered rolling median with edges retained."""
    return values.rolling(samples, center=True, min_periods=1).median()


def interpolate_at(model: pd.DataFrame, times: np.ndarray, column: str) -> np.ndarray:
    ok = np.isfinite(model["t_s"]) & np.isfinite(model[column])
    return np.interp(times, model.loc[ok, "t_s"], model.loc[ok, column])


def rmse_against_markers(model: pd.DataFrame, markers: pd.DataFrame, column: str) -> float:
    predicted = interpolate_at(model, markers["t_s"].to_numpy(), column)
    return float(np.sqrt(np.mean((predicted - markers["Y_m"].to_numpy()) ** 2)))


def save_source_crops() -> None:
    """Create a small audit pack from the source PDF, not manuscript art."""
    import fitz

    audit = OUT / "source_crops"
    audit.mkdir(exist_ok=True)
    doc = fitz.open(SOURCE_PDF)
    crops = [
        # PDF coordinates; pages are zero-based here.
        (2, fitz.Rect(34, 268, 578, 585), "table2_seriesB.png"),
        (5, fitz.Rect(26, 317, 586, 770), "fig6_bh6_photos.png"),
        (6, fitz.Rect(25, 285, 587, 755), "fig7_bh6_panels.png"),
        (9, fitz.Rect(28, 34, 305, 744), "fig10_pressure_context.png"),
    ]
    for page_no, clip, name in crops:
        pix = doc[page_no].get_pixmap(dpi=220, clip=clip, alpha=False)
        pix.save(audit / name)


def style_axis(ax: plt.Axes, panel: str) -> None:
    ax.text(-0.12, 1.03, panel, transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.grid(True, color="#d1d5db", lw=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", width=0.8)


def build_level_figure(exp: pd.DataFrame, one: pd.DataFrame, two: pd.DataFrame) -> None:
    fs = exp.loc[exp["kind"] == "fs"].sort_values("t_s")
    gi = exp.loc[exp["kind"] == "int"].sort_values("t_s")

    # The 2D series is sampled every 0.01 s.  A 0.05-s running median removes
    # one-cell threshold flicker without shifting the event clock.
    two = two.copy()
    two["Yfs_plot"] = running_median(two["Yfs_m_above_crown"], 5)
    two["Yint_plot"] = running_median(two["Yint_m_above_crown"], 5)

    colors = {"exp": "#111827", "1d": "#d62728", "2d": "#2563eb"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), sharex=True, sharey=True)

    panels = [
        (axes[0], fs, "Yfs_m", "Yfs_plot", r"Free surface, $Y_{fs}$", "(a)"),
        (axes[1], gi, "Yint_m", "Yint_plot", r"Gas nose, $Y_{int}$", "(b)"),
    ]
    for ax, markers, col_1d, col_2d, title, panel in panels:
        ax.plot(markers["t_s"], markers["Y_m"], "o", ms=3.7, mfc="white",
                mec=colors["exp"], mew=0.9, label="Cong et al. (2017)", zorder=5)
        ax.plot(one["t_s"], one[col_1d], color=colors["1d"], lw=1.45,
                ls="--", label="1D")
        ax.plot(two["t_s"], two[col_2d], color=colors["2d"], lw=1.45,
                label="OpenFOAM 2D")
        ax.axhline(RISER_HEIGHT, color="#6b7280", lw=0.85, ls=":", label="riser rim")
        ax.axvline(8.10, color="#9ca3af", lw=0.75, ls=(0, (2, 2)))
        ax.set_xlim(8.0, 12.0)
        ax.set_ylim(0.0, 1.85)
        ax.set_xlabel("Time after valve opening, $t$ (s)")
        ax.set_title(title, fontsize=9.2)
        style_axis(ax, panel)
    axes[0].set_ylabel("Height above pipe crown, $Y$ (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=8.1,
               loc="upper center", bbox_to_anchor=(0.51, 1.015))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.82, wspace=0.10)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cong2017_bh6_1d2d_levels.{suffix}", dpi=350,
                    bbox_inches="tight")
    plt.close(fig)


def build_pressure_figure(exp: pd.DataFrame, one: pd.DataFrame, two: pd.DataFrame) -> None:
    # Fig. 10(b) is Run B-32: the same Dr/H0/L0 condition as B-H6, but a
    # companion video-camera run.  It is therefore supporting, not direct,
    # evidence.  The distinction is kept in the legend and figure plan.
    one = one.copy()
    two = two.copy()
    one["H_plot"] = running_median(one["tr_head_m"] / H0, 25)  # about 0.5 s
    two["H_plot"] = running_median(two["head_m_water"] / H0, 21)  # about 0.1 s

    fig, ax = plt.subplots(figsize=(6.9, 3.3))
    ax.fill_between(exp["t_s"], exp["HoverH0_min"], exp["HoverH0_max"],
                    color="#d1d5db", alpha=0.55, lw=0,
                    label="B-32 raster span")
    ax.plot(exp["t_s"], exp["HoverH0_med"], color="#111827", lw=1.1,
            label="B-32 digitized PT1")
    ax.plot(one["t_s"], one["H_plot"], color="#d62728", lw=1.45, ls="--",
            label="B-H6 1D (0.5-s median)")
    ax.plot(two["t_s"], two["H_plot"], color="#2563eb", lw=1.35,
            label="B-H6 OpenFOAM 2D (0.1-s median)")
    ax.axvline(8.10, color="#6b7280", lw=0.8, ls=":", label="B-H6 $T_a=8.10$ s")
    ax.set(xlim=(0, 13), ylim=(0, 2.1), xlabel="Time after valve opening, $t$ (s)",
           ylabel="Normalized PT1 head, $H/H_0$")
    style_axis(ax, "")
    ax.legend(frameon=False, fontsize=7.9, ncol=2, loc="upper right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cong2017_bh6_1d2d_pressure_supporting.{suffix}",
                    dpi=350, bbox_inches="tight")
    plt.close(fig)


def build_multiframe_figure() -> None:
    """Assemble three common-clock, complete-domain 1D/2D comparisons."""
    from matplotlib.patches import Patch

    times = [0.0, 8.7, 10.9]
    indices = [int(round(t * 10.0)) for t in times]
    # Crop the common complete-domain data rectangle from the archived viewer
    # frames.  Titles/ticks are redrawn outside the raster; phase pixels are
    # retained without filtering.
    x0, x1, y0, y1 = 174, 1568, 36, 449

    fig, axes = plt.subplots(3, 2, figsize=(7.3, 5.05), sharex=True, sharey=True)
    for row, (time_s, index) in enumerate(zip(times, indices)):
        for col, folder in enumerate(("frames_1d", "frames_2d")):
            path = VIEWER / folder / f"full_{index:04d}.png"
            image = plt.imread(path)[y0:y1, x0:x1]
            ax = axes[row, col]
            ax.imshow(image, extent=(0.0, 6.59, -0.05, 1.90),
                      origin="upper", aspect="auto", interpolation="nearest")
            ax.set_xlim(0.0, 6.59)
            ax.set_ylim(-0.05, 1.90)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([0.0, 0.6, 1.2, 1.8])
            ax.tick_params(labelsize=7, length=2.5)
            if col == 1:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#374151")
        axes[row, 0].text(-0.19, 0.94, f"({chr(97 + row)})\n$t={time_s:.1f}$ s",
                          transform=axes[row, 0].transAxes, ha="left", va="top",
                          fontsize=8.2, fontweight="bold")

    axes[0, 0].set_title("1D", fontsize=9.2, fontweight="bold", pad=4)
    axes[0, 1].set_title("OpenFOAM 2D", fontsize=9.2, fontweight="bold", pad=4)
    axes[2, 0].set_xlabel("Horizontal distance (m)", fontsize=8)
    axes[2, 1].set_xlabel("Horizontal distance (m)", fontsize=8)
    fig.text(0.018, 0.49, "Height above pipe axis (m)", rotation=90,
             va="center", ha="center", fontsize=8)
    fig.legend(handles=[Patch(facecolor="#1769d1", label="water"),
                        Patch(facecolor="#f4f7fa", edgecolor="#9ca3af", label="gas")],
               ncol=2, loc="upper center", bbox_to_anchor=(0.56, 0.995),
               frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.075, top=0.91,
                        wspace=0.035, hspace=0.12)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cong2017_bh6_1d2d_full_domain_3frame.{suffix}", dpi=350,
                    bbox_inches="tight")
    plt.close(fig)


def build_metrics(exp: pd.DataFrame, one: pd.DataFrame, two: pd.DataFrame,
                  one_metrics: dict, two_metrics: dict) -> dict:
    fs = exp.loc[exp["kind"] == "fs"].sort_values("t_s")
    gi = exp.loc[exp["kind"] == "int"].sort_values("t_s")

    one_for_rmse = one.rename(columns={"Yfs_m": "Yfs", "Yint_m": "Yint"})
    two_for_rmse = two.rename(columns={"Yfs_m_above_crown": "Yfs",
                                       "Yint_m_above_crown": "Yint"})
    table = {
        "classification": {"experiment": "no geyser", "1D": "no geyser", "2D": "no geyser"},
        "Ta_s": {
            "experiment": 8.10,
            "1D": one_metrics["model"]["Ta_gas_enters_riser_s"],
            "2D": two_metrics["model"]["Ta_s"],
        },
        "Yfs_max_m": {
            "experiment": float(fs["Y_m"].max()),
            "1D": float(one["Yfs_m"].max()),
            "2D": float(two["Yfs_m_above_crown"].max()),
        },
        "Yint_max_m": {
            "experiment": float(gi["Y_m"].max()),
            "1D": float(one["Yint_m"].max()),
            "2D": float(two["Yint_m_above_crown"].max()),
        },
        "Yfs_marker_RMSE_m": {
            "experiment": 0.0,
            "1D": rmse_against_markers(one_for_rmse, fs, "Yfs"),
            "2D": rmse_against_markers(two_for_rmse, fs, "Yfs"),
        },
        "Yint_marker_RMSE_m": {
            "experiment": 0.0,
            "1D": rmse_against_markers(one_for_rmse, gi, "Yint"),
            "2D": rmse_against_markers(two_for_rmse, gi, "Yint"),
        },
    }
    (OUT / "bh6_validation_metrics.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "bh6_validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["observable", "experiment", "1D", "2D"])
        for observable, values in table.items():
            writer.writerow([observable, values["experiment"], values["1D"], values["2D"]])
    return table


def build_plan(metrics: dict) -> None:
    ta_1d = metrics["Ta_s"]["1D"]
    ta_2d = metrics["Ta_s"]["2D"]
    yfs_1d = metrics["Yfs_marker_RMSE_m"]["1D"]
    yfs_2d = metrics["Yfs_marker_RMSE_m"]["2D"]
    yint_1d = metrics["Yint_marker_RMSE_m"]["1D"]
    yint_2d = metrics["Yint_marker_RMSE_m"]["2D"]
    plan = f"""# B-H6 manuscript figure and table plan

## Location and dominant claim

- Manuscript location: `paper/sections/tests.tex`, Campaign 2, Series B (`sec:tests:cong2017:seriesB`).
- Focused claim: both calculations reproduce the observed non-geyser branch; the 2D run captures pocket arrival accurately, while both reduced descriptions underpredict the measured riser lift and gas-nose rise.
- Evidence status: **partial support**. Classification is matched, but the trajectory amplitudes are not quantitatively closed.

## Main-text assets

1. `cong2017_bh6_1d2d_full_domain_3frame.pdf`: complete pipe--tee--riser views at 0.0, 8.7 and 10.9 s, with 1D and 2D side by side on the same physical clock. Its single job is to show the system-level morphology/timing contrast.
2. `cong2017_bh6_1d2d_levels.pdf`: two panels redrawn from the digitized markers in Cong et al. (2017), Fig. 7(a), with unshifted 1D and OpenFOAM 2D curves. Its single job is quantitative trajectory validation.
3. Retain the existing Series-B table (`tab:cong_bh`) as the only main table; it already transcribes the essential B-H6 row from the paper's Table 2. Use `bh6_validation_metrics.csv` as the audit record instead of adding a redundant case table.

## Supporting/audit assets

- `source_crops/table2_seriesB.png`: source audit for the B-H6 row in Table 2.
- `source_crops/fig6_bh6_photos.png`: morphology audit at 8.7, 9.3, 9.9, 10.5 and 10.9 s. Do not reproduce in a submitted manuscript without checking publisher permissions.
- `source_crops/fig7_bh6_panels.png`: audit of the four original panels. Only panel (a) is used quantitatively.
- `cong2017_bh6_1d2d_pressure_supporting.pdf`: pressure context. The experiment is the same-condition companion run B-32 from Fig. 10(b), not the exact B-H6 high-speed record, so it must remain supporting evidence.

## Excluded from the main figure

- Fig. 7(b): velocities are differentiated from the same sparse positions and would duplicate/noisily amplify panel (a).
- Fig. 7(c): water/air column lengths are algebraic transforms of the interface positions and add no independent validation.
- Fig. 7(d): the published air-pocket pressure ratio has no directly equivalent, identically sampled 2D observable in the archived post-processing.
- Fig. 10(b): useful only as same-parameter pressure context, because its run number is B-32.

## Reproducible quantitative readout

- Arrival time: experiment 8.10 s, 1D {ta_1d:.2f} s, 2D {ta_2d:.2f} s.
- Free-surface marker RMSE: 1D {yfs_1d:.3f} m, 2D {yfs_2d:.3f} m.
- Gas-nose marker RMSE: 1D {yint_1d:.3f} m, 2D {yint_2d:.3f} m.
- No time shift is used. The dense 2D level traces use a centered 0.05-s running median only for plotting.

## Method-scope caveat

The geometry-matched H6 1D archive is a case-specific reduced application variant. Its horizontal gas-front closure is the Benjamin-celerity implementation, not an explicit full KH/IKH flux term. It should not be described as an independent verification of the manuscript's generic KH-containing horizontal formulation unless that solver-path discrepancy is resolved.
"""
    (OUT / "BH6_FIGURE_PLAN.md").write_text(plan, encoding="utf-8")


def main() -> None:
    exp_levels = pd.read_csv(EXP_LEVELS)
    exp_pressure = pd.read_csv(EXP_PRESSURE)
    one = pd.read_csv(ONE_D)
    two = pd.read_csv(TWO_D)
    two_pressure = pd.read_csv(TWO_D_PRESSURE)
    one_metrics = json.loads(ONE_D_METRICS.read_text(encoding="utf-8"))
    two_metrics = json.loads(TWO_D_METRICS.read_text(encoding="utf-8"))

    save_source_crops()
    build_multiframe_figure()
    build_level_figure(exp_levels, one, two)
    build_pressure_figure(exp_pressure, one, two_pressure)
    metrics = build_metrics(exp_levels, one, two, one_metrics, two_metrics)
    build_plan(metrics)
    print(f"Wrote B-H6 manuscript comparison assets to {OUT}")


if __name__ == "__main__":
    main()
