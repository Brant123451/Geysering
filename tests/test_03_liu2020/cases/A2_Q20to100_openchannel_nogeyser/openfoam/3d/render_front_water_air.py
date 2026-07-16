#!/usr/bin/env python3
"""Render complete A2 refined front-elevation water/air views.

1) True 3-D y=0 elevation slices from reconstructed VTK dumps
   (purgeWrite=3 kept volumes only at t=-12, 12, 13, 14 s).
   Each frame is a dual panel: full apparatus + horizontal-pipe zoom so
   shallow open-channel water is visible.
2) Full-timeline elevation motion from continuous probes
   (riser / chamber / tank / downstream wet area / bore), -12…14.4 s.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

CASE = Path(__file__).resolve().parent / "case"
OUT = Path(__file__).resolve().parents[2] / "outputs"
ART = Path("/opt/cursor/artifacts")
VTK_ROOT = CASE / "VTK"
PP = CASE / "postProcessing"

OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

pv.OFF_SCREEN = True
pv.global_theme.allow_empty_mesh = True

TIMES = [(-12, "case_0"), (12, "case_232258"), (13, "case_238055"), (14, "case_243803")]

# Apparatus outline (same coordinates as make_geometry.py)
LU, DU, SLOPE = 5.80, 0.20, 0.01
LC, WC, HC, DROP = 0.30, 0.30, 0.45, 0.18
DR, HR = 0.057, 1.22
LD, DD = 5.95, 0.28
LT, WT, HT = 0.57, 0.61, 0.89
Z_CREST, HW = 0.031, 0.40
RU, RD = DU / 2, DD / 2

WATER_CMAP = colors.LinearSegmentedColormap.from_list(
    "waterair", ["#f4f1ea", "#9ecae1", "#08519c"]
)
X_LIM = (-5.90, 6.90)
Z_FULL = (-0.45, 1.75)
Z_PIPE = (-0.05, 0.40)


def area_to_depth(A: float, D: float) -> float:
    R = D / 2.0
    A = float(np.clip(A, 0.0, math.pi * R * R))
    lo, hi = 0.0, D
    for _ in range(60):
        h = 0.5 * (lo + hi)
        a = R * R * math.acos((R - h) / R) - (R - h) * math.sqrt(max(0.0, 2 * R * h - h * h))
        if a < A:
            lo = h
        else:
            hi = h
    return 0.5 * (lo + hi)


def load_probe_scalar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:]


def draw_apparatus(ax, z_lim=Z_FULL):
    xs = np.linspace(-LU, 0.0, 80)
    zax = (DROP + RU) - SLOPE * xs
    ax.plot(xs, zax - RU, color="#555", lw=0.9, zorder=2)
    ax.plot(xs, zax + RU, color="#555", lw=0.9, zorder=2)
    ax.add_patch(Rectangle((0, 0), LC, HC, fill=False, ec="#333", lw=1.1, zorder=2))
    ax.plot([LC, LC + LD], [0.0, 0.0], color="#333", lw=1.0, zorder=2)
    ax.plot([LC, LC + LD], [DD, DD], color="#333", lw=1.0, zorder=2)
    ax.plot([LC + LD, LC + LD], [0.0, DD], color="#333", lw=1.0, zorder=2)
    xr0 = LC / 2 - DR / 2
    ax.add_patch(Rectangle((xr0, HC), DR, HR, fill=False, ec="#333", lw=1.1, zorder=2))
    tank_z0 = Z_CREST - HW
    ax.add_patch(
        Rectangle((LC + LD, tank_z0), LT, HT, fill=False, ec="#333", lw=1.1, zorder=2)
    )
    ax.plot(
        [LC + LD + LT / 2 - 0.15, LC + LD + LT / 2 + 0.15],
        [Z_CREST, Z_CREST],
        "k-",
        lw=1.4,
    )
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*z_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")


def vtk_slice_triangulation(folder: str) -> tuple[Triangulation, np.ndarray]:
    mesh = pv.read(VTK_ROOT / folder / "internal.vtu")
    sl = mesh.slice(normal="y", origin=(0.15, 0.0, 0.5))
    surf = sl.extract_surface(algorithm="dataset_surface").triangulate()
    faces = surf.faces.reshape(-1, 4)
    if faces.size == 0 or not np.all(faces[:, 0] == 3):
        raise RuntimeError(f"empty/non-triangle slice in {folder}")
    tri = faces[:, 1:]
    x = surf.points[:, 0]
    z = surf.points[:, 2]
    alpha = np.asarray(surf["alpha.water"], dtype=float)
    if alpha.shape[0] == surf.n_cells:
        ap = np.zeros(surf.n_points, dtype=float)
        cnt = np.zeros(surf.n_points, dtype=float)
        for c, (i, j, k) in enumerate(tri):
            for idx in (i, j, k):
                ap[idx] += alpha[c]
                cnt[idx] += 1.0
        alpha = ap / np.maximum(cnt, 1.0)
    return Triangulation(x, z, tri), alpha


def _paint_tripcolor(ax, triangulation: Triangulation, alpha: np.ndarray, z_lim):
    ax.tripcolor(
        triangulation,
        alpha,
        cmap=WATER_CMAP,
        vmin=0.0,
        vmax=1.0,
        shading="gouraud",
    )
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*z_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    # light apparatus outline on top of the field
    xs = np.linspace(-LU, 0.0, 80)
    zax = (DROP + RU) - SLOPE * xs
    ax.plot(xs, zax - RU, color="#222", lw=0.7, alpha=0.55)
    ax.plot(xs, zax + RU, color="#222", lw=0.7, alpha=0.55)
    ax.plot([0, LC, LC, 0, 0], [0, 0, HC, HC, 0], color="#222", lw=0.8, alpha=0.7)
    ax.plot([LC, LC + LD], [0.0, 0.0], color="#222", lw=0.7, alpha=0.7)
    ax.plot([LC, LC + LD], [DD, DD], color="#222", lw=0.7, alpha=0.7)
    xr0 = LC / 2 - DR / 2
    ax.plot(
        [xr0, xr0 + DR, xr0 + DR, xr0, xr0],
        [HC, HC, HC + HR, HC + HR, HC],
        color="#222",
        lw=0.8,
        alpha=0.7,
    )
    tank_z0 = Z_CREST - HW
    ax.plot(
        [LC + LD, LC + LD + LT, LC + LD + LT, LC + LD, LC + LD],
        [tank_z0, tank_z0, tank_z0 + HT, tank_z0 + HT, tank_z0],
        color="#222",
        lw=0.8,
        alpha=0.7,
    )


def render_vtk_front_views() -> list[Path]:
    """True field front elevations: collage + per-time dual panels."""
    written: list[Path] = []
    # ---- 2x2 collage of full elevations ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, (t, folder) in zip(axes.ravel(), TIMES):
        triangulation, alpha = vtk_slice_triangulation(folder)
        _paint_tripcolor(ax, triangulation, alpha, Z_FULL)
        wet = float(np.mean(alpha > 0.5))
        ax.set_title(
            f"t = {t:g} s   y=0 slice   ⟨α>0.5⟩={wet:.2f}",
            fontsize=11,
        )
    fig.suptitle(
        "Liu 2020 Case A2 — refined 3D interFoam COMPLETE front elevation\n"
        "True VTK alpha.water on y=0 plane (purgeWrite=3 kept t=-12, 12, 13, 14 s)",
        fontsize=13,
    )
    out = OUT / "openfoam_3d_refined_front_water_air.png"
    fig.savefig(out, dpi=170)
    fig.savefig(ART / out.name, dpi=170)
    plt.close(fig)
    written.append(out)

    # ---- dual-panel (full + pipe zoom) for each retained time ----
    for t, folder in TIMES:
        triangulation, alpha = vtk_slice_triangulation(folder)
        fig = plt.figure(figsize=(18, 8.2))
        ax_full = fig.add_axes([0.055, 0.42, 0.86, 0.50])
        ax_zoom = fig.add_axes([0.055, 0.08, 0.86, 0.28])
        cax = fig.add_axes([0.93, 0.20, 0.015, 0.55])
        _paint_tripcolor(ax_full, triangulation, alpha, Z_FULL)
        ax_full.set_title("Full apparatus front elevation (y = 0)", fontsize=12)
        mappable = ax_zoom.tripcolor(
            triangulation,
            alpha,
            cmap=WATER_CMAP,
            vmin=0.0,
            vmax=1.0,
            shading="gouraud",
        )
        ax_zoom.set_xlim(*X_LIM)
        ax_zoom.set_ylim(*Z_PIPE)
        ax_zoom.set_aspect("equal")
        ax_zoom.set_xlabel("x (m)")
        ax_zoom.set_ylabel("z (m)")
        ax_zoom.set_title(
            "Horizontal-pipe zoom — downstream open-channel water is the blue band",
            fontsize=12,
        )
        ax_zoom.axhline(DD, color="#222", lw=0.7, alpha=0.6)
        ax_zoom.axhline(0.0, color="#222", lw=0.7, alpha=0.6)
        fig.colorbar(mappable, cax=cax, label="alpha.water")
        fig.suptitle(
            f"A2 refined — complete front elevation render   t = {t:g} s",
            fontsize=14,
            y=0.98,
        )
        out_t = OUT / f"openfoam_3d_refined_front_full_t{str(t).replace('-', 'm')}.png"
        fig.savefig(out_t, dpi=170)
        fig.savefig(ART / out_t.name, dpi=170)
        plt.close(fig)
        written.append(out_t)

    # ---- stacked dual-panel strip of all 4 VTK times ----
    fig, axes = plt.subplots(4, 2, figsize=(18, 14), constrained_layout=True)
    for row, (t, folder) in enumerate(TIMES):
        triangulation, alpha = vtk_slice_triangulation(folder)
        _paint_tripcolor(axes[row, 0], triangulation, alpha, Z_FULL)
        axes[row, 0].set_title(f"t = {t:g} s  full", fontsize=11)
        _paint_tripcolor(axes[row, 1], triangulation, alpha, Z_PIPE)
        axes[row, 1].set_title(f"t = {t:g} s  pipe zoom", fontsize=11)
    fig.suptitle(
        "A2 refined — complete front elevation strip (true VTK y=0 field)",
        fontsize=14,
    )
    out_strip = OUT / "openfoam_3d_refined_front_complete_strip.png"
    fig.savefig(out_strip, dpi=160)
    fig.savefig(ART / out_strip.name, dpi=160)
    plt.close(fig)
    written.append(out_strip)
    return written


def water_surface_from_alpha_column(alphas: np.ndarray, z: np.ndarray, thr: float = 0.5) -> float:
    wet = np.where(alphas >= thr)[0]
    if wet.size == 0:
        return float(z[0])
    return float(z[wet[-1]])


def _load_motion_sources():
    t_riser, a_riser = load_probe_scalar(PP / "riserAlpha" / "-12" / "alpha.water")
    n_levels = 61
    z_riser = np.array([0.46 + 0.02 * i for i in range(n_levels)])
    a_levels = a_riser.reshape(-1, n_levels, 5).mean(axis=2)

    t_ch, a_ch_raw = load_probe_scalar(PP / "chamberLevel" / "-12" / "alpha.water")
    z_ch = np.array([0.02, 0.10, 0.20, 0.30, 0.44])
    a_ch = 0.5 * (a_ch_raw[:, 0:5] + a_ch_raw[:, 5:10])

    t_tank, a_tank = load_probe_scalar(PP / "tankLevel" / "-12" / "alpha.water")
    z_tank = np.array([-0.36 + 0.01 * i for i in range(a_tank.shape[1])])

    t_bore, a_bore = load_probe_scalar(PP / "boreAlpha" / "-12" / "alpha.water")
    # 2 stations × 5 samples; use mean of near-invert samples as wetness proxy
    a_bore_st = a_bore.reshape(-1, 2, 5).mean(axis=2)

    def wet_area(name: str):
        p = next((PP / name).rglob("surfaceFieldValue.dat"))
        t, a = load_probe_scalar(p)
        return t, a[:, 0]

    t060, A060 = wet_area("downstreamWetAreaX060")
    t325, A325 = wet_area("downstreamWetAreaX325")
    t600, A600 = wet_area("downstreamWetAreaX600")
    return {
        "t_riser": t_riser,
        "z_riser": z_riser,
        "a_levels": a_levels,
        "t_ch": t_ch,
        "z_ch": z_ch,
        "a_ch": a_ch,
        "t_tank": t_tank,
        "z_tank": z_tank,
        "a_tank": a_tank,
        "t_bore": t_bore,
        "a_bore_st": a_bore_st,
        "t060": t060,
        "A060": A060,
        "t325": t325,
        "A325": A325,
        "t600": t600,
        "A600": A600,
    }


def _draw_probe_frame(ax, t: float, src: dict, z_lim=Z_FULL):
    draw_apparatus(ax, z_lim=z_lim)
    # upstream: bore probes mark filling; before arrival keep ~0.08 m free surface
    xs = np.linspace(-LU, 0.0, 80)
    zax = (DROP + RU) - SLOPE * xs
    ib = int(np.argmin(np.abs(src["t_bore"] - t)))
    bore_wet = float(src["a_bore_st"][ib].mean())
    if t < 1.5 and bore_wet < 0.4:
        h_up = 0.08
    else:
        h_up = min(DU, 0.08 + 0.12 * min(1.0, max(0.0, (t - 1.5) / 2.0) + bore_wet))
    ax.fill_between(
        xs,
        zax - RU,
        np.minimum(zax - RU + h_up, zax + RU),
        color="#2b6cb0",
        alpha=0.88,
        zorder=1,
    )

    i = int(np.argmin(np.abs(src["t_ch"] - t)))
    z_free = water_surface_from_alpha_column(src["a_ch"][i], src["z_ch"])
    ax.add_patch(
        Rectangle((0.0, 0.0), LC, max(0.0, z_free), color="#2b6cb0", alpha=0.88, zorder=1)
    )

    i = int(np.argmin(np.abs(src["t_riser"] - t)))
    for zi, ai in zip(src["z_riser"], src["a_levels"][i]):
        if ai >= 0.5:
            ax.add_patch(
                Rectangle(
                    (LC / 2 - DR / 2, zi - 0.01),
                    DR,
                    0.02,
                    color="#2b6cb0",
                    alpha=0.92,
                    zorder=3,
                )
            )
        elif ai >= 0.2:
            ax.add_patch(
                Rectangle(
                    (LC / 2 - DR / 2, zi - 0.01),
                    DR,
                    0.02,
                    color="#9ecae1",
                    alpha=0.75,
                    zorder=3,
                )
            )

    hs = []
    for tx, Ax in (
        (src["t060"], src["A060"]),
        (src["t325"], src["A325"]),
        (src["t600"], src["A600"]),
    ):
        j = int(np.argmin(np.abs(tx - t)))
        hs.append(area_to_depth(Ax[j], DD))
    x_dn = np.linspace(LC, LC + LD, 120)
    h_dn = np.clip(np.interp(x_dn, [0.60, 3.25, 6.00], hs), 0.0, DD)
    ax.fill_between(x_dn, 0.0, h_dn, color="#2b6cb0", alpha=0.88, zorder=1)
    # annotate depths on zoom-friendly frames
    if z_lim[1] <= 0.5:
        for x_lab, h_lab in zip([0.60, 3.25, 6.00], hs):
            ax.plot(x_lab, h_lab, "o", color="#b22222", ms=4, zorder=4)
            ax.text(x_lab, h_lab + 0.012, f"{h_lab:.2f} m", color="#b22222", fontsize=8, ha="center")

    i = int(np.argmin(np.abs(src["t_tank"] - t)))
    z_free_t = water_surface_from_alpha_column(src["a_tank"][i], src["z_tank"])
    tank_x0 = LC + LD
    tank_z0 = Z_CREST - HW
    ax.add_patch(
        Rectangle(
            (tank_x0, tank_z0),
            LT,
            max(0.0, z_free_t - tank_z0),
            color="#2b6cb0",
            alpha=0.78,
            zorder=1,
        )
    )


def render_full_motion_collage() -> Path:
    src = _load_motion_sources()
    frames_t = np.array(
        [-12, -6, -1, 0, 0.4, 1.0, 1.6, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0]
    )
    frames_t = frames_t[(frames_t >= src["t_riser"][0]) & (frames_t <= src["t_riser"][-1])]

    nrows = len(frames_t)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 2.55 * nrows), constrained_layout=True)
    for row, t in enumerate(frames_t):
        _draw_probe_frame(axes[row, 0], float(t), src, Z_FULL)
        axes[row, 0].set_title(f"t = {t:g} s  full elevation", fontsize=10)
        _draw_probe_frame(axes[row, 1], float(t), src, Z_PIPE)
        axes[row, 1].set_title(f"t = {t:g} s  pipe zoom (water labeled)", fontsize=10)

    fig.suptitle(
        "A2 refined — complete front-elevation motion timeline\n"
        "(probe-reconstructed; blue=water; right column zooms pipes so hd is visible)",
        fontsize=13,
    )
    out = OUT / "openfoam_3d_refined_front_motion_timeline.png"
    fig.savefig(out, dpi=150)
    fig.savefig(ART / out.name, dpi=150)
    plt.close(fig)
    return out


def _adaptive_timeline(t0: float, t1: float) -> np.ndarray:
    """Dense sampling around the valve ramp / bore; coarser elsewhere."""
    parts = [
        np.arange(t0, min(0.0, t1) + 1e-9, 0.40),
        np.arange(max(0.0, t0), min(4.0, t1) + 1e-9, 0.08),
        np.arange(max(4.0, t0), t1 + 1e-9, 0.20),
        np.array([t0, t1], dtype=float),
    ]
    # Force true VTK dump times into the schedule.
    forced = np.array([t for t, _ in TIMES], dtype=float)
    forced = forced[(forced >= t0 - 1e-9) & (forced <= t1 + 1e-9)]
    t = np.unique(np.round(np.concatenate(parts + [forced]), 6))
    return t[(t >= t0 - 1e-9) & (t <= t1 + 1e-9)]


def _draw_vtk_dual(ax_full, ax_zoom, triangulation: Triangulation, alpha: np.ndarray, t: float):
    _paint_tripcolor(ax_full, triangulation, alpha, Z_FULL)
    ax_full.set_title(
        f"TRUE VTK y=0 field   t = {t:.2f} s   (volume dump)",
        fontsize=12,
        color="#0b3d5c",
    )
    _paint_tripcolor(ax_zoom, triangulation, alpha, Z_PIPE)
    ax_zoom.set_title(
        "Pipe zoom — true alpha.water (open-channel water = blue band)",
        fontsize=11,
        color="#0b3d5c",
    )


def _draw_probe_dual(ax_full, ax_zoom, t: float, src: dict):
    _draw_probe_frame(ax_full, t, src, Z_FULL)
    ax_full.set_title(f"Complete front elevation   t = {t:.2f} s", fontsize=12)
    _draw_probe_frame(ax_zoom, t, src, Z_PIPE)
    ax_zoom.set_title(
        "Pipe zoom — downstream depths labeled (probe-reconstructed)",
        fontsize=11,
    )


def _draw_time_scrubber(ax, t: float, t0: float, t1: float):
    ax.clear()
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.axvspan(0.0, 0.4, color="#f6ad55", alpha=0.35, label="Q0→Q1 valve")
    ax.axvline(1.60, color="#c53030", lw=1.0, ls="--", label="paper bore 1.60 s")
    ax.axvline(t, color="#1a365d", lw=2.5)
    ax.set_yticks([])
    ax.set_xlabel("simulation time t (s)")
    ax.set_title(f"timeline scrubber   now t = {t:.2f} s", fontsize=10)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)


def render_motion_gif() -> Path:
    """Legacy shorter dual-panel GIF (kept for compatibility)."""
    src = _load_motion_sources()
    frames_t = np.linspace(max(-12.0, src["t_riser"][0]), min(14.4, src["t_riser"][-1]), 72)

    fig = plt.figure(figsize=(14, 6.4))
    ax_full = fig.add_axes([0.06, 0.48, 0.90, 0.42])
    ax_zoom = fig.add_axes([0.06, 0.08, 0.90, 0.32])

    def update(k: int):
        t = float(frames_t[k])
        ax_full.clear()
        ax_zoom.clear()
        _draw_probe_dual(ax_full, ax_zoom, t, src)

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=100)
    out = OUT / "openfoam_3d_refined_front_motion.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    anim.save(ART / out.name, writer=PillowWriter(fps=10))
    plt.close(fig)
    return out


def render_complete_front_gif() -> Path:
    """Very complete dual-panel front-elevation GIF over the full run.

    Adaptive time grid (~130 frames): dense around valve/bore, coarser in
    Q0 init and late steady state. At retained VTK dump times the frame is
    the true y=0 alpha.water field; all other frames use continuous probes.
    """
    from PIL import Image

    src = _load_motion_sources()
    t0 = float(max(-12.0, src["t_riser"][0]))
    t1 = float(min(14.4, src["t_riser"][-1]))
    frames_t = _adaptive_timeline(t0, t1)

    print(f"  complete GIF frames: {len(frames_t)}  t=[{frames_t[0]:.3f}, {frames_t[-1]:.3f}]")
    print("  caching VTK triangulations for dump times...")
    vtk_cache: dict[float, tuple[Triangulation, np.ndarray]] = {}
    for t_vtk, folder in TIMES:
        if t0 - 0.05 <= t_vtk <= t1 + 0.05:
            vtk_cache[float(t_vtk)] = vtk_slice_triangulation(folder)

    tmp_dir = ART / "_complete_front_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    fig = plt.figure(figsize=(16.5, 8.2))
    ax_full = fig.add_axes([0.05, 0.50, 0.90, 0.42])
    ax_zoom = fig.add_axes([0.05, 0.14, 0.90, 0.30])
    ax_scrub = fig.add_axes([0.05, 0.035, 0.90, 0.055])

    for k, t in enumerate(frames_t):
        t = float(t)
        ax_full.clear()
        ax_zoom.clear()
        # Prefer true VTK within 0.05 s of a retained dump.
        dump_t = None
        for tv in vtk_cache:
            if abs(t - tv) <= 0.05:
                dump_t = tv
                break
        if dump_t is not None:
            triangulation, alpha = vtk_cache[dump_t]
            _draw_vtk_dual(ax_full, ax_zoom, triangulation, alpha, dump_t)
            mode = "VTK"
        else:
            _draw_probe_dual(ax_full, ax_zoom, t, src)
            mode = "probe"
        _draw_time_scrubber(ax_scrub, t, t0, t1)
        fig.suptitle(
            "Liu 2020 Case A2 refined — COMPLETE front elevation water/air motion\n"
            f"frame {k + 1}/{len(frames_t)}   source={mode}   "
            "blue=water, cream/white=air",
            fontsize=13,
            y=0.995,
        )
        fp = tmp_dir / f"frame_{k:04d}.png"
        fig.savefig(fp, dpi=110)
        frame_paths.append(fp)
        if k % 20 == 0 or k + 1 == len(frames_t):
            print(f"  rendered frame {k + 1}/{len(frames_t)}  t={t:.2f}s  ({mode})")

    plt.close(fig)

    print("  assembling GIF...")
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
    out = OUT / "openfoam_3d_refined_front_complete_motion.gif"
    # Hold first/last a bit longer for readability.
    durations = [80] * len(images)
    durations[0] = 500
    durations[-1] = 700
    for i, t in enumerate(frames_t):
        if any(abs(float(t) - tv) <= 0.05 for tv in vtk_cache):
            durations[i] = 450  # linger on true VTK frames
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    images[0].save(
        ART / out.name,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    # Cleanup frame PNGs to avoid clutter (keep GIF only).
    for p in frame_paths:
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return out


def render_downstream_zoom() -> Path:
    """Keep the dedicated downstream zoom panel for quick inspection."""
    src = _load_motion_sources()
    frames_t = np.array([-12, -1, 0, 1.6, 3.0, 7.0, 12.0, 14.0])
    frames_t = frames_t[(frames_t >= src["t_riser"][0]) & (frames_t <= src["t_riser"][-1])]
    ncols = 4
    nrows = int(math.ceil(len(frames_t) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, frames_t):
        _draw_probe_frame(ax, float(t), src, Z_PIPE)
        ax.set_title(f"t = {t:g} s", fontsize=10)
    for ax in axes[len(frames_t) :]:
        ax.set_axis_off()
    fig.suptitle(
        "Downstream pipe water is present: zoomed front elevation",
        fontsize=13,
    )
    out = OUT / "openfoam_3d_refined_front_downstream_zoom.png"
    fig.savefig(out, dpi=150)
    fig.savefig(ART / out.name, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print("rendering COMPLETE VTK front elevations (tripcolor y=0)...")
    paths = render_vtk_front_views()
    for p in paths:
        print("wrote", p)
    print("rendering full-timeline dual-panel collage...")
    p2 = render_full_motion_collage()
    print("wrote", p2)
    print("rendering dual-panel motion gif...")
    p3 = render_motion_gif()
    print("wrote", p3)
    print("rendering COMPLETE front-elevation motion gif...")
    p3b = render_complete_front_gif()
    print("wrote", p3b)
    print("rendering downstream zoom...")
    p4 = render_downstream_zoom()
    print("wrote", p4)


if __name__ == "__main__":
    main()
