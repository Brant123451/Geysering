#!/usr/bin/env python3
"""Render A2 refined front-elevation water/air views.

1) True 3-D centerline (y≈0) slices from reconstructed VTK times
   (-12, 12, 13, 14 s) — the only volume dumps retained under purgeWrite=3.
2) Full-timeline elevation motion collage from continuous probes
   (riser / chamber / tank / downstream wet area), covering -12…14.4 s.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

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


def draw_apparatus(ax):
    # upstream pipe (axis)
    xs = np.linspace(-LU, 0.0, 80)
    zax = (DROP + RU) - SLOPE * xs
    ax.fill_between(xs, zax - RU, zax + RU, color="#d9d9d9", alpha=0.55, zorder=0)
    # chamber
    ax.add_patch(Rectangle((0, 0), LC, HC, fill=False, ec="#333", lw=1.2, zorder=2))
    # downstream pipe
    ax.add_patch(
        FancyBboxPatch(
            (LC, 0),
            LD,
            DD,
            boxstyle="square,pad=0",
            fill=True,
            fc="#d9d9d9",
            ec="#333",
            alpha=0.55,
            zorder=0,
        )
    )
    # riser
    xr0 = LC / 2 - DR / 2
    ax.add_patch(Rectangle((xr0, HC), DR, HR, fill=False, ec="#333", lw=1.2, zorder=2))
    # tank
    tank_z0 = Z_CREST - HW
    ax.add_patch(
        Rectangle((LC + LD, tank_z0), LT, HT, fill=False, ec="#333", lw=1.2, zorder=2)
    )
    # weir crest mark
    ax.plot([LC + LD + LT / 2 - 0.15, LC + LD + LT / 2 + 0.15], [Z_CREST, Z_CREST], "k-", lw=1.5)
    ax.set_xlim(-6.0, 7.0)
    ax.set_ylim(-0.45, 1.75)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")


def render_vtk_front_views() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    cmap = colors.LinearSegmentedColormap.from_list(
        "waterair", ["#f7f4ef", "#9ecae1", "#2171b5"]
    )
    for ax, (t, folder) in zip(axes.ravel(), TIMES):
        mesh = pv.read(VTK_ROOT / folder / "internal.vtu")
        # centerline slice: y = 0 plane (front / elevation view)
        sl = mesh.slice(normal="y", origin=(0.15, 0.0, 0.5))
        pts = sl.points
        alpha = np.asarray(sl["alpha.water"])
        # triangulate scatter via tripcolor using PolyData faces if available
        if sl.n_cells > 0 and "alpha.water" in sl.array_names:
            # Use PyVista screenshot for robust triangulation, then also mpl fallback
            pl = pv.Plotter(off_screen=True, window_size=(1400, 700))
            pl.set_background("white")
            pl.add_mesh(
                sl,
                scalars="alpha.water",
                cmap=["#f7f4ef", "#6baed6", "#08519c"],
                clim=[0, 1],
                show_scalar_bar=False,
                lighting=False,
            )
            # outline walls lightly
            try:
                walls = pv.read(VTK_ROOT / folder / "boundary" / "solidWalls.vtp")
                pl.add_mesh(walls, color="#bbbbbb", opacity=0.15, lighting=False)
            except Exception:
                pass
            pl.camera_position = [
                (0.5, -28.0, 0.55),
                (0.5, 0.0, 0.55),
                (0.0, 0.0, 1.0),
            ]
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = 4.2
            # Crop to the apparatus elevation window in display space.
            pl.enable_parallel_projection()
            pl.reset_camera()
            # Look from -y at the xz plane; then manually frame the domain.
            pl.camera_position = "xz"
            pl.camera.SetPosition(0.5, -40.0, 0.6)
            pl.camera.SetFocalPoint(0.5, 0.0, 0.6)
            pl.camera.SetViewUp(0.0, 0.0, 1.0)
            pl.camera.parallel_scale = 7.2
            # Shift slightly so upstream and tank both fit.
            pl.camera.SetFocalPoint(0.6, 0.0, 0.55)
            tmp = ART / f"_tmp_front_t{t}.png"
            pl.show(screenshot=str(tmp))
            pl.close()
            img = plt.imread(tmp)
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(f"t = {t:g} s  (y=0 elevation slice, alpha.water)", fontsize=11)
        else:
            ax.text(0.5, 0.5, "empty slice", ha="center")
            ax.set_axis_off()
    fig.suptitle(
        "Liu 2020 Case A2 — refined 3D interFoam front elevation (water/air)\n"
        "(purgeWrite=3 kept volume fields only at t=-12, 12, 13, 14 s)",
        fontsize=13,
    )
    out = OUT / "openfoam_3d_refined_front_water_air.png"
    fig.savefig(out, dpi=160)
    fig.savefig(ART / "openfoam_3d_refined_front_water_air.png", dpi=160)
    plt.close(fig)
    return out


def sample_timeline(n: int = 48) -> np.ndarray:
    t_riser, a_riser = load_probe_scalar(PP / "riserAlpha" / "-12" / "alpha.water")
    t_end = min(14.4, float(t_riser[-1]))
    return np.linspace(-12.0, t_end, n)


def water_surface_from_alpha_column(alphas: np.ndarray, z: np.ndarray, thr: float = 0.5) -> float:
    wet = np.where(alphas >= thr)[0]
    if wet.size == 0:
        return float(z[0])
    return float(z[wet[-1]])


def render_full_motion_collage() -> Path:
    t_riser, a_riser = load_probe_scalar(PP / "riserAlpha" / "-12" / "alpha.water")
    # 61 levels × 5 radial → reshape mean over radial
    n_levels = 61
    z_riser = np.array([0.46 + 0.02 * i for i in range(n_levels)])
    a_levels = a_riser.reshape(-1, n_levels, 5).mean(axis=2)

    t_ch, a_ch_raw = load_probe_scalar(PP / "chamberLevel" / "-12" / "alpha.water")
    # Two vertical lines × five elevations (z=0.02,0.10,0.20,0.30,0.44).
    z_ch = np.array([0.02, 0.10, 0.20, 0.30, 0.44])
    a_ch = 0.5 * (a_ch_raw[:, 0:5] + a_ch_raw[:, 5:10])

    t_tank, a_tank = load_probe_scalar(PP / "tankLevel" / "-12" / "alpha.water")
    z_tank = np.array([-0.36 + 0.01 * i for i in range(a_tank.shape[1])])

    def wet_area(name: str):
        p = next((PP / name).rglob("surfaceFieldValue.dat"))
        t, a = load_probe_scalar(p)
        return t, a[:, 0]

    t060, A060 = wet_area("downstreamWetAreaX060")
    t325, A325 = wet_area("downstreamWetAreaX325")
    t600, A600 = wet_area("downstreamWetAreaX600")

    frames_t = np.array(
        [-12, -6, -1, 0, 0.4, 1.0, 1.6, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0]
    )
    frames_t = frames_t[(frames_t >= t_riser[0]) & (frames_t <= t_riser[-1])]

    ncols = 4
    nrows = int(math.ceil(len(frames_t) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, frames_t):
        draw_apparatus(ax)
        # upstream approximate free surface ~0.08 m above invert near t<=0
        xs = np.linspace(-LU, 0.0, 40)
        zinv = DROP - SLOPE * xs  # invert approx without radius offset complexity
        # better invert = axis - ru*cos... use axis-ru vertically for sketch
        zax = (DROP + RU) - SLOPE * xs
        if t <= 0:
            h_up = 0.08
        else:
            # after bore, treat as near-full for sketch if t>1.6
            h_up = 0.08 if t < 1.5 else min(DU, 0.18)
        ax.fill_between(xs, zax - RU, np.minimum(zax - RU + h_up, zax + RU), color="#2b6cb0", alpha=0.85, zorder=1)

        # chamber
        i = np.argmin(np.abs(t_ch - t))
        z_free = water_surface_from_alpha_column(a_ch[i], z_ch)
        ax.add_patch(Rectangle((0.0, 0.0), LC, max(0.0, z_free), color="#2b6cb0", alpha=0.85, zorder=1))

        # riser
        i = np.argmin(np.abs(t_riser - t))
        alphas = a_levels[i]
        for zi, ai in zip(z_riser, alphas):
            if ai >= 0.5:
                ax.add_patch(
                    Rectangle((LC / 2 - DR / 2, zi - 0.01), DR, 0.02, color="#2b6cb0", alpha=0.9, zorder=3)
                )
            elif ai >= 0.2:
                ax.add_patch(
                    Rectangle((LC / 2 - DR / 2, zi - 0.01), DR, 0.02, color="#9ecae1", alpha=0.7, zorder=3)
                )

        # downstream depths
        for tx, Ax, xx in (
            (t060, A060, 0.60),
            (t325, A325, 3.25),
            (t600, A600, 6.00),
        ):
            j = np.argmin(np.abs(tx - t))
            h = area_to_depth(Ax[j], DD)
            ax.add_patch(Rectangle((xx - 0.08, 0.0), 0.16, h, color="#2b6cb0", alpha=0.8, zorder=1))

        # tank stage
        i = np.argmin(np.abs(t_tank - t))
        z_free_t = water_surface_from_alpha_column(a_tank[i], z_tank)
        tank_x0 = LC + LD
        tank_z0 = Z_CREST - HW
        ax.add_patch(
            Rectangle(
                (tank_x0, tank_z0),
                LT,
                max(0.0, z_free_t - tank_z0),
                color="#2b6cb0",
                alpha=0.75,
                zorder=1,
            )
        )
        ax.set_title(f"t = {t:g} s", fontsize=10)

    for ax in axes[len(frames_t) :]:
        ax.set_axis_off()

    fig.suptitle(
        "A2 refined — full-timeline front elevation water/air motion\n"
        "(probe-reconstructed apparatus elevation; blue=water, light=mixture/air)",
        fontsize=13,
    )
    out = OUT / "openfoam_3d_refined_front_motion_timeline.png"
    fig.savefig(out, dpi=150)
    fig.savefig(ART / "openfoam_3d_refined_front_motion_timeline.png", dpi=150)
    plt.close(fig)
    return out


def render_motion_gif() -> Path:
    t_riser, a_riser = load_probe_scalar(PP / "riserAlpha" / "-12" / "alpha.water")
    n_levels = 61
    z_riser = np.array([0.46 + 0.02 * i for i in range(n_levels)])
    a_levels = a_riser.reshape(-1, n_levels, 5).mean(axis=2)
    t_ch, a_ch_raw = load_probe_scalar(PP / "chamberLevel" / "-12" / "alpha.water")
    z_ch = np.array([0.02, 0.10, 0.20, 0.30, 0.44])
    a_ch = 0.5 * (a_ch_raw[:, 0:5] + a_ch_raw[:, 5:10])
    t_tank, a_tank = load_probe_scalar(PP / "tankLevel" / "-12" / "alpha.water")
    z_tank = np.array([-0.36 + 0.01 * i for i in range(a_tank.shape[1])])

    def wet_area(name: str):
        p = next((PP / name).rglob("surfaceFieldValue.dat"))
        return load_probe_scalar(p)

    t060, A060 = wet_area("downstreamWetAreaX060")
    t325, A325 = wet_area("downstreamWetAreaX325")
    t600, A600 = wet_area("downstreamWetAreaX600")
    A060, A325, A600 = A060[:, 0], A325[:, 0], A600[:, 0]

    frames_t = np.linspace(max(-12.0, t_riser[0]), min(14.4, t_riser[-1]), 60)

    fig, ax = plt.subplots(figsize=(12, 4.8))

    def update(k: int):
        ax.clear()
        t = float(frames_t[k])
        draw_apparatus(ax)
        xs = np.linspace(-LU, 0.0, 40)
        zax = (DROP + RU) - SLOPE * xs
        h_up = 0.08 if t < 1.5 else min(DU, 0.18)
        ax.fill_between(xs, zax - RU, np.minimum(zax - RU + h_up, zax + RU), color="#2b6cb0", alpha=0.85)
        i = np.argmin(np.abs(t_ch - t))
        z_free = water_surface_from_alpha_column(a_ch[i], z_ch)
        ax.add_patch(Rectangle((0.0, 0.0), LC, max(0.0, z_free), color="#2b6cb0", alpha=0.85))
        i = np.argmin(np.abs(t_riser - t))
        for zi, ai in zip(z_riser, a_levels[i]):
            if ai >= 0.5:
                ax.add_patch(Rectangle((LC / 2 - DR / 2, zi - 0.01), DR, 0.02, color="#2b6cb0", alpha=0.9))
            elif ai >= 0.2:
                ax.add_patch(Rectangle((LC / 2 - DR / 2, zi - 0.01), DR, 0.02, color="#9ecae1", alpha=0.7))
        for tx, Ax, xx in ((t060, A060, 0.60), (t325, A325, 3.25), (t600, A600, 6.00)):
            j = np.argmin(np.abs(tx - t))
            h = area_to_depth(Ax[j], DD)
            ax.add_patch(Rectangle((xx - 0.08, 0.0), 0.16, h, color="#2b6cb0", alpha=0.8))
        i = np.argmin(np.abs(t_tank - t))
        z_free_t = water_surface_from_alpha_column(a_tank[i], z_tank)
        tank_x0 = LC + LD
        tank_z0 = Z_CREST - HW
        ax.add_patch(
            Rectangle((tank_x0, tank_z0), LT, max(0.0, z_free_t - tank_z0), color="#2b6cb0", alpha=0.75)
        )
        ax.set_title(f"A2 refined front elevation water/air motion  t = {t:.2f} s")

    anim = FuncAnimation(fig, update, frames=len(frames_t), interval=120)
    out = OUT / "openfoam_3d_refined_front_motion.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    anim.save(ART / "openfoam_3d_refined_front_motion.gif", writer=PillowWriter(fps=8))
    plt.close(fig)
    return out


def main() -> None:
    print("rendering VTK front slices...")
    p1 = render_vtk_front_views()
    print("wrote", p1)
    print("rendering full-timeline collage...")
    p2 = render_full_motion_collage()
    print("wrote", p2)
    print("rendering motion gif...")
    p3 = render_motion_gif()
    print("wrote", p3)


if __name__ == "__main__":
    main()
