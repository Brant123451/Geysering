# -*- coding: utf-8 -*-
"""Build B-H6 comparison figures anchored to Cong et al. (2017) artwork.

The published raster pixels are kept unchanged wherever they are shown.  The
script only crops the source figures, appends simulation frames, and overlays
the archived 1D/2D results through documented pixel-to-data transforms.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve()
CASE = HERE.parent.parent
OUT = CASE / "outputs" / "paper_matched_overlays"
OUT.mkdir(parents=True, exist_ok=True)
SCANS = CASE / "reference" / "paper_scans"
VIEWER = CASE / "outputs" / "1d2d_viewer"
ONE_D = CASE / "outputs" / "paper_layout_1d" / "caseB_model_series.csv"
TWO_D = CASE / "openfoam" / "2d" / "results" / "openfoam_2d_riser_series.csv"
TWO_D_P = CASE / "openfoam" / "2d" / "results" / "openfoam_2d_pt1_series.csv"
METRICS = CASE / "outputs" / "manuscript_comparison" / "bh6_validation_metrics.json"
TABLE2 = CASE / "outputs" / "manuscript_comparison" / "source_crops" / "table2_seriesB.png"
H0 = 0.66


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_both(fig: plt.Figure, stem: str, dpi: int = 350) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def running_median(series: pd.Series, samples: int) -> pd.Series:
    return series.rolling(samples, center=True, min_periods=1).median()


def build_fig6_augmented() -> None:
    """Published Fig. 6 photos plus 1D/2D rows at the identical five times."""
    source = Image.open(SCANS / "fig6_bh6_photos.png").convert("RGB")
    # Borders of the five published photographic frames in the 1700x2180 scan.
    boxes = [
        (207, 943, 350, 2053),
        (491, 943, 634, 2053),
        (776, 943, 918, 2053),
        (1062, 943, 1206, 2053),
        (1348, 943, 1492, 2053),
    ]
    times = [8.7, 9.3, 9.9, 10.5, 10.9]
    indices = [87, 93, 99, 105, 109]

    fig, axes = plt.subplots(
        3, 5, figsize=(7.3, 7.1),
        gridspec_kw={"height_ratios": [1.62, 1.0, 1.0], "hspace": 0.10, "wspace": 0.05},
    )
    for col, (box, time_s, index) in enumerate(zip(boxes, times, indices)):
        axes[0, col].imshow(source.crop(box), interpolation="nearest")
        axes[0, col].set_title(f"$t={time_s:.1f}$ s", fontsize=8.5, pad=3)
        for row, folder in ((1, "frames_1d"), (2, "frames_2d")):
            frame = Image.open(VIEWER / folder / f"zoom_{index:04d}.png").convert("RGB")
            # The common riser data rectangle; titles and axes are redrawn by the montage.
            axes[row, col].imshow(frame.crop((77, 37, 335, 731)), interpolation="nearest")
        for row in range(3):
            axes[row, col].axis("off")

    row_labels = ["Experiment\nCong et al. (2017), Fig. 6", "1D", "OpenFOAM 2D"]
    y_positions = [0.77, 0.40, 0.145]
    for label, y in zip(row_labels, y_positions):
        fig.text(0.012, y, label, rotation=90, ha="center", va="center",
                 fontsize=8.3, fontweight="bold" if label != row_labels[0] else "normal")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.015, top=0.955)
    save_both(fig, "cong2017_fig6_bh6_experiment_1d2d")


def build_fig6_full_domain_three_frame() -> None:
    """Three common-clock rows: complete 1D, source photo, complete 2D."""
    from matplotlib.patches import Patch

    source = Image.open(SCANS / "fig6_bh6_photos.png").convert("RGB")
    # Exact published frames at the early, developed, and late stages.
    photo_boxes = [
        (207, 943, 350, 2053),   # 8.7 s
        (776, 943, 918, 2053),   # 9.9 s
        (1348, 943, 1492, 2053), # 10.9 s
    ]
    times = [8.7, 9.9, 10.9]
    indices = [87, 99, 109]

    # Complete-domain data rectangle in the archived viewer frames.
    x0, x1, y0, y1 = 174, 1568, 36, 449
    fig = plt.figure(figsize=(7.3, 6.75))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[4.4, 0.72, 4.4], hspace=0.08, wspace=0.06
    )
    domain_axes: list[tuple[plt.Axes, plt.Axes]] = []

    for row, (time_s, index, box) in enumerate(zip(times, indices, photo_boxes)):
        ax_1d = fig.add_subplot(gs[row, 0])
        ax_exp = fig.add_subplot(gs[row, 1])
        ax_2d = fig.add_subplot(gs[row, 2])
        domain_axes.append((ax_1d, ax_2d))

        ax_exp.imshow(source.crop(box), interpolation="nearest")
        ax_exp.axis("off")

        for ax, folder in ((ax_1d, "frames_1d"), (ax_2d, "frames_2d")):
            path = VIEWER / folder / f"full_{index:04d}.png"
            frame = np.asarray(Image.open(path).convert("RGB"))[y0:y1, x0:x1]
            ax.imshow(frame, extent=(0.0, 6.59, -0.05, 1.90),
                      origin="upper", aspect="auto", interpolation="nearest")
            ax.set_xlim(0.0, 6.59)
            ax.set_ylim(-0.05, 1.90)
            ax.set_box_aspect(1.95 / 6.59)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([0.0, 0.6, 1.2, 1.8])
            ax.tick_params(labelsize=6.4, length=2.2, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#374151")

        ax_2d.tick_params(labelleft=False)
        if row < 2:
            ax_1d.tick_params(labelbottom=False)
            ax_2d.tick_params(labelbottom=False)
        else:
            ax_1d.set_xlabel("Horizontal distance (m)", fontsize=7.3, labelpad=2)
            ax_2d.set_xlabel("Horizontal distance (m)", fontsize=7.3, labelpad=2)

        ax_1d.text(0.025, 0.91, f"({chr(97 + row)})  $t={time_s:.1f}$ s",
                   transform=ax_1d.transAxes, ha="left", va="top",
                   fontsize=7.3, fontweight="bold",
                   bbox={"facecolor": "white", "edgecolor": "none",
                         "alpha": 0.88, "pad": 1.2})

    domain_axes[0][0].set_title("Complete 1D domain", fontsize=8.5,
                                fontweight="bold", pad=4)
    fig.axes[1].set_title("Experiment\n(Cong et al., Fig. 6)", fontsize=7.4,
                          fontweight="bold", pad=4)
    domain_axes[0][1].set_title("Complete OpenFOAM 2D domain", fontsize=8.5,
                                fontweight="bold", pad=4)
    fig.text(0.019, 0.50, "Height above pipe axis (m)", rotation=90,
             va="center", ha="center", fontsize=7.5)
    fig.legend(handles=[Patch(facecolor="#1769d1", label="water"),
                        Patch(facecolor="#f4f7fa", edgecolor="#9ca3af", label="gas")],
               ncol=2, loc="lower center", bbox_to_anchor=(0.55, 0.005),
               frameon=False, fontsize=7.3)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.065, top=0.955)
    save_both(fig, "cong2017_bh6_experiment_1d2d_full_domain_3frame")


def data_to_fig7_px(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = 202.0 + (t - 8.0) / 3.0 * (813.0 - 202.0)
    py = 1378.0 - y / 2.0 * (1378.0 - 926.0)
    return x, py


def build_fig7a_overlay(one: pd.DataFrame, two: pd.DataFrame) -> None:
    """Overlay model trajectories on the unaltered raster of Fig. 7(a)."""
    source = Image.open(SCANS / "fig7_bh6_riser.png").convert("RGB")
    left, top, right, bottom = 70, 890, 855, 1470
    bg = np.asarray(source.crop((left, top, right, bottom)))
    width, height = right - left, bottom - top

    fig = plt.figure(figsize=(width / 170.0, height / 170.0), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest")

    two_plot = two.copy()
    two_plot["Yfs_plot"] = running_median(two_plot["Yfs_m_above_crown"], 5)
    two_plot["Yint_plot"] = running_median(two_plot["Yint_m_above_crown"], 5)
    curves = [
        (one["t_s"], one["Yfs_m"], "#b2182b", "-", "1D $Y_{fs}$"),
        (one["t_s"], one["Yint_m"], "#b2182b", "--", "1D $Y_{int}$"),
        (two_plot["t_s"], two_plot["Yfs_plot"], "#1b7837", "-", "2D $Y_{fs}$"),
        (two_plot["t_s"], two_plot["Yint_plot"], "#1b7837", "--", "2D $Y_{int}$"),
    ]
    for t, y, color, ls, label in curves:
        mask = (t >= 8.0) & (t <= 11.0) & np.isfinite(y) & (y >= 0.0) & (y <= 2.0)
        xpx, ypx = data_to_fig7_px(t[mask].to_numpy(), y[mask].to_numpy())
        ax.plot(xpx - left, ypx - top, color=color, ls=ls, lw=1.35, label=label,
                solid_capstyle="round")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2,
                       frameon=True, framealpha=0.94, facecolor="white", edgecolor="0.45",
                       fontsize=6.3, handlelength=2.0, columnspacing=0.8, borderpad=0.35)
    legend.get_frame().set_linewidth(0.55)
    save_both(fig, "cong2017_fig7a_bh6_experiment_1d2d", dpi=350)


def data_to_fig10_px(t: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = 180.0 + t / 13.0 * (747.0 - 180.0)
    py = 970.0 - h / 4.0 * (970.0 - 550.0)
    return x, py


def build_fig10b_overlay(one: pd.DataFrame, two_p: pd.DataFrame) -> None:
    """Overlay B-H6 simulations on the published same-condition B-32 panel."""
    source = Image.open(SCANS / "fig10_pressure.png").convert("RGB")
    left, top, right, bottom = 100, 530, 790, 1065
    bg = np.asarray(source.crop((left, top, right, bottom)))
    width, height = right - left, bottom - top

    one_h = running_median(one["tr_head_m"] / H0, 25)
    two_h = running_median(two_p["head_m_water"] / H0, 21)
    fig = plt.figure(figsize=(width / 170.0, height / 170.0), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bg, extent=(0, width, height, 0), interpolation="nearest")
    for t, h, color, ls, label in [
        (one["t_s"], one_h, "#7b3294", "--", "B-H6 1D"),
        (two_p["t_s"], two_h, "#008837", "-", "B-H6 OpenFOAM 2D"),
    ]:
        mask = (t >= 0.0) & (t <= 13.0) & np.isfinite(h) & (h >= 0.0) & (h <= 4.0)
        xpx, ypx = data_to_fig10_px(t[mask].to_numpy(), h[mask].to_numpy())
        ax.plot(xpx - left, ypx - top, color=color, ls=ls, lw=1.45, label=label)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.15, 0.12), frameon=True,
                       framealpha=0.94, facecolor="white", edgecolor="0.45",
                       fontsize=7.0, borderpad=0.35)
    legend.get_frame().set_linewidth(0.55)
    save_both(fig, "cong2017_fig10b_b32_pressure_with_bh6_1d2d", dpi=350)


def build_table2_augmented(metrics: dict) -> None:
    """Keep the source Table 2 raster and append a source-matched B-H6 block."""
    source = np.asarray(Image.open(TABLE2).convert("RGB"))
    fig = plt.figure(figsize=(7.3, 5.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.9, 1.25], hspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(source, interpolation="nearest")
    # Highlight the B-H6 source row without obscuring any published pixel.
    ax0.add_patch(Rectangle((13, 825), 1635, 34, fill=False, ec="#b2182b", lw=1.35))
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    columns = ["Source", "Outcome", "$T_a$ (s)", "$Y_{fs,max}$ (m)",
               "$Y_{int,max}$ (m)", "RMSE $Y_{fs}$ (m)", "RMSE $Y_{int}$ (m)"]
    rows = [
        ["B-H6 experiment", "No", "8.10", "1.201", "1.178", "--", "--"],
        ["1D", "No", f"{metrics['Ta_s']['1D']:.2f}",
         f"{metrics['Yfs_max_m']['1D']:.3f}", f"{metrics['Yint_max_m']['1D']:.3f}",
         f"{metrics['Yfs_marker_RMSE_m']['1D']:.3f}",
         f"{metrics['Yint_marker_RMSE_m']['1D']:.3f}"],
        ["OpenFOAM 2D", "No", f"{metrics['Ta_s']['2D']:.2f}",
         f"{metrics['Yfs_max_m']['2D']:.3f}", f"{metrics['Yint_max_m']['2D']:.3f}",
         f"{metrics['Yfs_marker_RMSE_m']['2D']:.3f}",
         f"{metrics['Yint_marker_RMSE_m']['2D']:.3f}"],
    ]
    table = ax1.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                      loc="center", bbox=[0.0, 0.02, 1.0, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.45 if row > 0 else 0.8)
        cell.set_facecolor("white")
        if row == 0:
            cell.set_text_props(weight="bold")
    ax1.text(0.0, 1.04, "B-H6 model comparison added to the published Table 2 conditions",
             transform=ax1.transAxes, fontsize=8.3, fontweight="bold", ha="left")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.025, top=0.99)
    save_both(fig, "cong2017_table2_bh6_experiment_1d2d")


def write_manifest() -> None:
    manifest = {
        "source": "Cong, Chan and Lee (2017), J. Hydraul. Eng. 143(9), 04017039",
        "source_pdf": "references/cong2017.pdf",
        "principle": "published raster pixels retained; simulations appended or overlaid",
        "fig6": {
            "source_scan": "reference/paper_scans/fig6_bh6_photos.png",
            "times_s": [8.7, 9.3, 9.9, 10.5, 10.9],
            "simulation_frames": "outputs/1d2d_viewer/frames_{1d,2d}/zoom_*.png",
        },
        "fig6_full_domain_3frame": {
            "source_scan": "reference/paper_scans/fig6_bh6_photos.png",
            "times_s": [8.7, 9.9, 10.9],
            "layout": "complete 1D | published riser photograph | complete OpenFOAM 2D",
            "simulation_frames": "outputs/1d2d_viewer/frames_{1d,2d}/full_*.png",
            "time_alignment": "common physical clock; no shift or event alignment",
        },
        "fig7a": {
            "source_scan": "reference/paper_scans/fig7_bh6_riser.png",
            "source_axis_px": {"x": [202, 813], "t_s": [8, 11], "y": [1378, 926], "Y_m": [0, 2]},
            "smoothing": "2D curves: centered 0.05-s running median; no time shift",
        },
        "fig10b": {
            "source_run": "B-32 (same Dr/H0/L0 condition, not the B-H6 high-speed record)",
            "source_axis_px": {"x": [180, 747], "t_s": [0, 13], "y": [970, 550], "H_over_H0": [0, 4]},
            "smoothing": "1D 0.5-s and 2D 0.1-s running medians; no time shift",
        },
        "copyright_note": "Publisher permission may be required to reproduce the photographic/source rasters in a submitted manuscript.",
    }
    (OUT / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    one = pd.read_csv(ONE_D)
    two = pd.read_csv(TWO_D)
    two_p = pd.read_csv(TWO_D_P)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    build_fig6_augmented()
    build_fig6_full_domain_three_frame()
    build_fig7a_overlay(one, two)
    build_fig10b_overlay(one, two_p)
    build_table2_augmented(metrics)
    write_manifest()
    print(f"Wrote paper-matched overlays to {OUT}")


if __name__ == "__main__":
    main()
