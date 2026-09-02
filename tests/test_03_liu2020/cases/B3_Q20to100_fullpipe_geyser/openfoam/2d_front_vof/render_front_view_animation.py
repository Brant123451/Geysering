#!/usr/bin/env python3
"""Render a front-elevation (x–z) water/air motion animation for Liu2020 B3.

Volume fields were kept with purgeWrite=4, so intermediate VTK dumps are not
available.  This renderer rebuilds the front view from the completed baseline
run's dense postProcessing probes:

  - riserCentreline alpha.water  (z = 0.47 … 5.22 m, Δt = 0.002 s)
  - probesPT pressures           (PT1–PT4)
  - paper geometry from make_mesh.py

The animation is therefore the full 16.4 s geyser motion actually recorded by
the 3-D solver, drawn as a centreline-faithful elevation of the apparatus.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


HERE = Path(__file__).resolve().parent
DEFAULT_CASE = HERE / "case"
DEFAULT_OUT = HERE.parent / "outputs" / "openfoam_3d_front_view_motion.mp4"

# Paper / mesh geometry [m] (matches make_mesh.py).
LU, DU, SLOPE = 5.80, 0.20, 0.01
LC, WC, HC, DROP = 0.30, 0.30, 0.45, 0.18
LD, DD = 5.95, 0.28
DR, HR = 0.06, 1.22
RU, RD, RR = DU / 2.0, DD / 2.0, DR / 2.0
X_RISER = LC / 2.0
Z_LID = HC
Z_RIM = HC + HR
RAMP_START = 2.0
PATM = 101325.0

# Initial depths used by setFields (for schematic pipe fill only).
H_UPSTREAM0 = 0.08
H_CHAMBER0 = 0.30
H_DOWNSTREAM0 = DD  # full pipe

WATER = "#1f6f8b"
WATER_LIGHT = "#3d9bb0"
AIR = "#e8f1f4"
STEEL = "#3a3f44"
STEEL_LIGHT = "#6b7280"
BG_TOP = "#d9e6ec"
BG_BOT = "#f4f7f5"
ACCENT = "#b45309"


def configure_fonts() -> str:
    candidates = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    return "DejaVu Sans"


def z_axis_upstream(x: float | np.ndarray) -> float | np.ndarray:
    return DROP + RU - SLOPE * x


def read_probe_table(path: Path) -> tuple[list[str], np.ndarray]:
    headers: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("# Probe"):
                headers.append(line.strip())
            elif line.startswith("#") or not line.strip():
                continue
            else:
                break
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return headers, data


def load_series(case: Path) -> dict[str, np.ndarray]:
    alpha_headers, alpha = read_probe_table(
        case / "postProcessing" / "riserCentreline" / "0" / "alpha.water"
    )
    _, pressure = read_probe_table(case / "postProcessing" / "probesPT" / "0" / "p")
    z = np.array(
        [float(h.split("(")[1].split(")")[0].split()[2]) for h in alpha_headers],
        dtype=float,
    )
    # Align pressure to alpha times by nearest-neighbour (same writeInterval).
    t_alpha = alpha[:, 0]
    t_p = pressure[:, 0]
    idx = np.searchsorted(t_p, t_alpha, side="left")
    idx = np.clip(idx, 0, len(t_p) - 1)
    # Prefer closer neighbour.
    left = np.clip(idx - 1, 0, len(t_p) - 1)
    use_left = np.abs(t_p[left] - t_alpha) < np.abs(t_p[idx] - t_alpha)
    idx = np.where(use_left, left, idx)
    p_gauge_kpa = (pressure[idx, 1:] - PATM) / 1000.0
    return {
        "t": t_alpha,
        "report_t": t_alpha - RAMP_START,
        "z": z,
        "alpha": np.clip(alpha[:, 1:], 0.0, 1.0),
        "PT3": p_gauge_kpa[:, 0],
        "PT2": p_gauge_kpa[:, 1],
        "PT1": p_gauge_kpa[:, 2],
        "PT4": p_gauge_kpa[:, 3],
    }


def pipe_band(
    x0: float,
    x1: float,
    radius: float,
    z_centre_fn,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, n)
    zc = np.asarray(z_centre_fn(xs), dtype=float)
    upper = np.column_stack([xs, zc + radius])
    lower = np.column_stack([xs[::-1], (zc - radius)[::-1]])
    return np.vstack([upper, lower]), zc


def draw_apparatus(ax: plt.Axes) -> None:
    # Upstream pipe outline.
    up_poly, up_zc = pipe_band(-LU, 0.0, RU, z_axis_upstream)
    ax.add_patch(
        Polygon(up_poly, closed=True, facecolor="#cfd8dc", edgecolor=STEEL, lw=1.0, zorder=2)
    )
    # Downstream pipe (horizontal at z = RD).
    dn_xs = np.linspace(LC, LC + LD, 60)
    dn_upper = np.column_stack([dn_xs, np.full_like(dn_xs, 2 * RD)])
    dn_lower = np.column_stack([dn_xs[::-1], np.zeros_like(dn_xs)])
    ax.add_patch(
        Polygon(
            np.vstack([dn_upper, dn_lower]),
            closed=True,
            facecolor="#cfd8dc",
            edgecolor=STEEL,
            lw=1.0,
            zorder=2,
        )
    )
    # Chamber.
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            LC,
            HC,
            boxstyle="square,pad=0",
            facecolor="#b0bec5",
            edgecolor=STEEL,
            lw=1.2,
            zorder=3,
        )
    )
    # Riser tube.
    ax.add_patch(
        Rectangle(
            (X_RISER - RR, Z_LID),
            2 * RR,
            HR,
            facecolor="#90a4ae",
            edgecolor=STEEL,
            lw=1.0,
            zorder=3,
        )
    )
    # Atmosphere / plume box (dashed).
    ax.add_patch(
        Rectangle(
            (X_RISER - 0.30, Z_RIM),
            0.60,
            5.25 - Z_RIM,
            fill=False,
            edgecolor=STEEL_LIGHT,
            lw=0.8,
            ls="--",
            zorder=1,
        )
    )
    # Lid and rim markers.
    ax.plot([0.0, LC], [Z_LID, Z_LID], color=STEEL, lw=1.4, zorder=4)
    ax.plot(
        [X_RISER - RR, X_RISER + RR],
        [Z_RIM, Z_RIM],
        color=ACCENT,
        lw=1.6,
        zorder=4,
    )
    ax.text(X_RISER + 0.35, Z_RIM, "riser rim", color=ACCENT, fontsize=8, va="center")


def draw_schematic_pipe_water(ax: plt.Axes) -> None:
    """Static IC water fill in pipes (volume free-surface was not retained)."""
    # Upstream open-channel fill to H_UPSTREAM0 above invert.
    xs = np.linspace(-LU, 0.0, 120)
    zc = z_axis_upstream(xs)
    invert = zc - RU
    free = invert + H_UPSTREAM0
    crown = zc + RU
    free = np.minimum(free, crown)
    ax.fill_between(xs, invert, free, color=WATER, alpha=0.55, zorder=2.5, lw=0)
    # Downstream full.
    ax.add_patch(
        Rectangle((LC, 0.0), LD, DD, facecolor=WATER, alpha=0.55, edgecolor="none", zorder=2.5)
    )
    # Chamber initial stage (replaced each frame when alpha available near lid).
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            LC,
            H_CHAMBER0,
            facecolor=WATER,
            alpha=0.35,
            edgecolor="none",
            zorder=2.6,
            label="_chamber0",
        )
    )


def water_polygons_from_alpha(
    z: np.ndarray,
    alpha: np.ndarray,
    x_centre: float,
    half_width_riser: float,
    half_width_jet: float,
) -> list[np.ndarray]:
    """Build filled x–z polygons for contiguous wet segments along the centreline."""
    wet = alpha >= 0.05
    if not np.any(wet):
        return []
    polys: list[np.ndarray] = []
    start = None
    for i, flag in enumerate(wet):
        if flag and start is None:
            start = i
        if (not flag or i == len(wet) - 1) and start is not None:
            end = i if not flag else i
            if end < start:
                start = None
                continue
            segment = slice(start, end + 1)
            zs = z[segment]
            al = alpha[segment]
            # Widen slightly in the free plume above the rim.
            widths = np.where(
                zs <= Z_RIM,
                half_width_riser * (0.55 + 0.45 * al),
                half_width_jet * (0.35 + 0.65 * al),
            )
            left = np.column_stack([x_centre - widths, zs])
            right = np.column_stack([x_centre + widths[::-1], zs[::-1]])
            polys.append(np.vstack([left, right]))
            start = None
    return polys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dt",
        type=float,
        default=0.04,
        help="Animation sample interval in solver seconds (source Δt = 0.002 s)",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional solver-time end (default: full series)",
    )
    parser.add_argument(
        "--focus",
        choices=("full", "geyser"),
        default="geyser",
        help="full apparatus or chamber/riser zoom",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    font_name = configure_fonts()
    series = load_series(args.case.resolve())
    t = series["t"]
    if args.t_max is not None:
        keep = t <= args.t_max + 1e-9
        for key, value in list(series.items()):
            if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == len(t):
                series[key] = value[keep]
            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == len(t):
                series[key] = value[keep]
        t = series["t"]

    # Subsample frames.
    step = max(1, int(round(args.dt / 0.002)))
    indices = np.arange(0, len(t), step)
    # Always include the peak-height frame.
    heights = []
    for row in series["alpha"]:
        wet = np.flatnonzero(row >= 0.05)
        heights.append(series["z"][wet[-1]] if wet.size else series["z"][0])
    heights = np.asarray(heights)
    peak_i = int(np.argmax(heights))
    if peak_i not in set(indices.tolist()):
        indices = np.sort(np.append(indices, peak_i))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.5, 7.2), dpi=120)
    fig.patch.set_facecolor(BG_BOT)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[3.2, 1.15], height_ratios=[3.4, 1.1], hspace=0.28, wspace=0.22
    )
    ax = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    # Background gradient via imshow.
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(
        gradient,
        extent=(-LU - 0.4, LC + LD + 0.4, -0.15, 5.35),
        aspect="auto",
        cmap=mpl.colors.LinearSegmentedColormap.from_list("bg", [BG_BOT, BG_TOP]),
        origin="lower",
        zorder=0,
    )

    draw_apparatus(ax)
    draw_schematic_pipe_water(ax)

    chamber_water = Rectangle(
        (0.0, 0.0), LC, H_CHAMBER0, facecolor=WATER, alpha=0.75, edgecolor="none", zorder=3.2
    )
    ax.add_patch(chamber_water)
    jet_patches: list[Polygon] = []

    if args.focus == "geyser":
        ax.set_xlim(-1.2, 2.0)
        ax.set_ylim(-0.05, 3.2)
    else:
        ax.set_xlim(-LU - 0.3, LC + LD + 0.3)
        ax.set_ylim(-0.1, 5.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(
        "Liu2020 B3 · OpenFOAM 3-D front elevation (α.water centreline)",
        fontsize=12,
        pad=10,
    )
    ax.text(
        0.01,
        0.99,
        f"font={font_name} · source=riserCentreline probes · Δt_src=0.002 s",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color=STEEL_LIGHT,
    )

    time_text = ax.text(
        0.98,
        0.97,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=STEEL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # Pressure panel.
    ax_p.plot(series["report_t"], series["PT2"], color="#c2410c", lw=1.0, label="PT2")
    ax_p.plot(series["report_t"], series["PT3"], color="#0369a1", lw=1.0, label="PT3")
    ax_p.axhline(0.0, color=STEEL_LIGHT, lw=0.6)
    ax_p.axvline(0.0, color=STEEL_LIGHT, lw=0.6, ls="--")
    cursor_p = ax_p.axvline(series["report_t"][0], color=ACCENT, lw=1.2)
    ax_p.set_xlim(series["report_t"][0], series["report_t"][-1])
    ax_p.set_ylabel("gauge kPa")
    ax_p.set_title("Pressure probes", fontsize=10)
    ax_p.legend(fontsize=8, loc="upper right", frameon=False)
    ax_p.set_facecolor("#f8faf9")

    # Height panel.
    h_lid = np.maximum(heights - Z_LID, 0.0)
    ax_h.fill_between(series["report_t"], 0.0, h_lid, color=WATER, alpha=0.35)
    ax_h.plot(series["report_t"], h_lid, color=WATER, lw=1.1)
    ax_h.axhline(HR, color=ACCENT, lw=0.9, ls="--", label="rim")
    cursor_h = ax_h.axvline(series["report_t"][0], color=ACCENT, lw=1.2)
    ax_h.set_xlim(series["report_t"][0], series["report_t"][-1])
    ax_h.set_ylim(0.0, max(2.0, float(np.nanmax(h_lid)) * 1.15))
    ax_h.set_xlabel("t after ramp (s)")
    ax_h.set_ylabel("jet above lid (m)")
    ax_h.set_title("Geyser height", fontsize=10)
    ax_h.legend(fontsize=8, loc="upper right", frameon=False)
    ax_h.set_facecolor("#f8faf9")

    legend_bits = [
        plt.Line2D([0], [0], color=WATER, lw=6, solid_capstyle="butt", label="water (α≥0.05)"),
        plt.Line2D([0], [0], color="#cfd8dc", lw=6, solid_capstyle="butt", label="structure"),
        plt.Line2D([0], [0], color=STEEL_LIGHT, lw=1.2, ls="--", label="atmosphere box"),
    ]
    ax.legend(handles=legend_bits, loc="upper left", fontsize=8, frameon=False)

    def update(frame_i: int):
        nonlocal jet_patches
        i = int(indices[frame_i])
        alpha_row = series["alpha"][i]
        # Chamber fill: if base of riser is wet, treat chamber as connected/full to lid;
        # otherwise keep a residual pool from α near the lid probe.
        base_alpha = float(alpha_row[0])
        if base_alpha >= 0.05:
            chamber_water.set_height(HC)
            chamber_water.set_alpha(0.55 + 0.35 * base_alpha)
        else:
            chamber_water.set_height(max(0.05, H_CHAMBER0 * 0.7))
            chamber_water.set_alpha(0.35)

        for patch in jet_patches:
            patch.remove()
        jet_patches = []
        for poly in water_polygons_from_alpha(
            series["z"], alpha_row, X_RISER, RR * 1.15, RR * 3.5
        ):
            patch = Polygon(
                poly,
                closed=True,
                facecolor=WATER,
                edgecolor=WATER_LIGHT,
                lw=0.4,
                alpha=0.92,
                zorder=5,
            )
            ax.add_patch(patch)
            jet_patches.append(patch)

        rt = series["report_t"][i]
        ht = max(0.0, heights[i] - Z_LID)
        time_text.set_text(
            f"solver t = {series['t'][i]:.3f} s\n"
            f"after ramp = {rt:+.3f} s\n"
            f"jet above lid = {ht:.2f} m\n"
            f"PT2 = {series['PT2'][i]:+.1f} kPa"
        )
        cursor_p.set_xdata([rt, rt])
        cursor_h.set_xdata([rt, rt])
        return [chamber_water, time_text, cursor_p, cursor_h, *jet_patches]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000 / args.fps,
        blit=False,
    )
    writer = FFMpegWriter(
        fps=args.fps,
        metadata={
            "title": "Liu2020 B3 OpenFOAM 3-D front-view water/air motion",
            "artist": "Geysering B3 validation",
        },
        bitrate=2400,
    )
    print(
        f"rendering {len(indices)} frames → {args.output} "
        f"(dt={args.dt}s, fps={args.fps}, focus={args.focus})"
    )
    anim.save(args.output, writer=writer)
    plt.close(fig)

    # Also write a short GIF covering the critical geyser window for quick preview.
    gif_path = args.output.with_suffix(".gif")
    geyser_mask = (series["report_t"] >= -0.2) & (series["report_t"] <= 4.5)
    geyser_idx = np.flatnonzero(geyser_mask)
    if geyser_idx.size:
        step_g = max(1, int(round(0.05 / 0.002)))
        g_indices = geyser_idx[::step_g]
        fig2, ax2 = plt.subplots(figsize=(7.2, 7.0), dpi=90)
        fig2.patch.set_facecolor(BG_BOT)
        ax2.imshow(
            gradient,
            extent=(-0.8, 1.4, -0.05, 3.0),
            aspect="auto",
            cmap=mpl.colors.LinearSegmentedColormap.from_list("bg", [BG_BOT, BG_TOP]),
            origin="lower",
            zorder=0,
        )
        draw_apparatus(ax2)
        ax2.set_xlim(-0.6, 1.2)
        ax2.set_ylim(-0.02, 2.8)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("z (m)")
        ax2.set_title("B3 geyser window (front elevation)")
        chamber2 = Rectangle((0.0, 0.0), LC, H_CHAMBER0, facecolor=WATER, alpha=0.7, zorder=3.2)
        ax2.add_patch(chamber2)
        jet2: list[Polygon] = []
        txt2 = ax2.text(
            0.98,
            0.97,
            "",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        def update_gif(frame_i: int):
            nonlocal jet2
            i = int(g_indices[frame_i])
            for patch in jet2:
                patch.remove()
            jet2 = []
            for poly in water_polygons_from_alpha(
                series["z"], series["alpha"][i], X_RISER, RR * 1.2, RR * 3.8
            ):
                patch = Polygon(poly, closed=True, facecolor=WATER, alpha=0.92, zorder=5)
                ax2.add_patch(patch)
                jet2.append(patch)
            base = float(series["alpha"][i, 0])
            chamber2.set_height(HC if base >= 0.05 else max(0.05, H_CHAMBER0 * 0.7))
            txt2.set_text(
                f"t_ramp = {series['report_t'][i]:+.2f} s\n"
                f"h_lid = {max(0.0, heights[i] - Z_LID):.2f} m"
            )
            return [chamber2, txt2, *jet2]

        anim2 = FuncAnimation(fig2, update_gif, frames=len(g_indices), blit=False)
        anim2.save(gif_path, writer="pillow", fps=16)
        plt.close(fig2)
        print(f"wrote preview GIF → {gif_path} ({len(g_indices)} frames)")

    print("done")


if __name__ == "__main__":
    main()
