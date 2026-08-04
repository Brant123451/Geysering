# -*- coding: utf-8 -*-
"""Case A interactive frame viewer assets, in the same design as the older
outputs/vw2011_network/report.html viewer:

  outputs/frames/frame_XXXX.png        -- global 1:1 view (tunnel + tower)
  outputs/riser_frames/riser_XXXX.png  -- tower zoom (area-preserving gas core)
  outputs/frames_index.json            -- [{file, riserFile, time, wtop, itop,
                                           coreMassMg, head}, ...]

caseA_digitize_and_compare.py embeds these into report.html as an interactive
viewer (prev/play/slider/next + arrow keys), replacing the static GIF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(CASE / "model"))

from vw2011_network_twofluid import NetworkCase, _depth_frac, run_network
from casea_shockfit_network import build_case_a_shockfit_solver

OUT = CASE / "outputs"
FRAMES = OUT / "frames"
RISER_FRAMES = OUT / "riser_frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="")
    parser.add_argument("--ds", type=float, default=0.01)
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--t-end", type=float, default=13.0)
    parser.add_argument("--phase-volume-cfl", type=float, default=0.25)
    parser.add_argument(
        "--momentum-viscosity",
        type=float,
        default=0.10,
        help="Horizontal Smagorinsky/mixing-length coefficient; the riser uses its independent film-scale closure.",
    )
    parser.add_argument(
        "--churn-friction",
        type=float,
        default=0.15,
        help="Additional Darcy coefficient active only in horizontally stratified gas-liquid cells.",
    )
    parser.add_argument(
        "--interfacial-enhancement",
        type=float,
        default=0.0,
        help=(
            "Horizontal stratified interfacial-friction coefficient C_h in "
            "lambda_i=lambda_g*(1+C_h*alpha_l)."
        ),
    )
    parser.add_argument(
        "--junction-loss",
        type=float,
        default=0.75,
        help="Single-phase side-T minor-loss coefficient K.",
    )
    parser.add_argument(
        "--glug-loss",
        type=float,
        default=8.0,
        help=(
            "Additional two-phase turn/mixing loss K active while gas and "
            "returning liquid share the T mouth."
        ),
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=0.05,
        help="Physical time between saved frames [s].",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help=(
            "Save conservative solver fields and diagnostics without creating "
            "PNG frames; this changes only post-processing runtime."
        ),
    )
    args = parser.parse_args()

    variant = args.variant.strip()
    frames_dir = OUT / (f"frames_{variant}" if variant else "frames")
    riser_frames_dir = OUT / (
        f"riser_frames_{variant}" if variant else "riser_frames"
    )
    index_path = OUT / (
        f"frames_index_{variant}.json" if variant else "frames_index.json"
    )
    diagnostics_path = OUT / (
        f"solver_diagnostics_{variant}.json"
        if variant
        else "solver_diagnostics.json"
    )
    fields_path = OUT / (
        f"vertical_fields_{variant}.npz"
        if variant
        else "vertical_fields.npz"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    frames_dir.mkdir(parents=True, exist_ok=True)
    riser_frames_dir.mkdir(parents=True, exist_ok=True)

    case = NetworkCase(
        Dr=0.0571,
        air_head=0.305,
        init_water_level=0.356,
        cfl=0.65,
        t_end=args.t_end,
        ds=args.ds,
        dz=args.dz,
        phase_volume_cfl=args.phase_volume_cfl,
        nu=args.momentum_viscosity,
        horizontal_churn_friction=args.churn_friction,
        horizontal_holdup_drag_enhancement=args.interfacial_enhancement,
        junction_loss_coeff=args.junction_loss,
        glug_loss_coeff=args.glug_loss,
    )
    external_horizontal_solver = build_case_a_shockfit_solver(
        dx=(
            case.L_tunnel
            / max(20, int(round(case.L_tunnel / case.ds)))
        ),
        wave_speed=case.a_wh,
    )
    rec = run_network(
        case,
        verbose=True,
        external_horizontal_solver=external_horizontal_solver,
        output_interval=args.output_interval,
    )

    diagnostic_keys = (
        "t", "wtop", "itop", "core_mass", "pocket_head", "base_q",
        "base_head", "junction_alpha", "left_mean_alpha", "right_mean_alpha",
        "right_max_alpha", "right_full_fraction", "tun_gas_mass",
        "tun_gas_vol", "tot_liq", "tot_liq_raw", "escaped_gas_mass",
        "total_resolved_gas_mass", "escaped_liquid_volume",
        "atmospheric_gas_mass_exchange",
        "total_gas_mass_including_atmosphere",
        "total_liquid_including_escape",
        "annular_film_return_volume", "riser_gas_front",
        "riser_gas_front_velocity", "riser_entry_cut_front",
        "riser_material_front",
        "riser_breakthrough",
        "junction_return_requested_volume",
        "junction_return_deposited_volume",
        "junction_return_unplaced_volume",
        "junction_wave_max_source_cells",
        "junction_east_liquid_flux",
        "junction_liquid_balance_correction",
        "junction_node_head",
        "junction_west_liquid_flux",
        "junction_east_node_liquid_flux",
        "junction_vertical_liquid_flux",
        "junction_gas_mouth_fraction",
        "junction_west_head",
        "junction_east_head",
        "junction_vertical_head",
        "dt_outer",
        "dt_phase_limit",
        "dt_junction_limit",
        "horizontal_gas_substeps",
        "horizontal_gas_active_cells",
        "horizontal_gas_mass_error",
        "horizontal_gas_kinetic_energy",
        "horizontal_gas_center_of_mass",
        "horizontal_gas_maximum_velocity",
        "coupled_gas_maximum_velocity",
        "right_branch_gas_mass",
        "side_t_east_cut_volume",
        "side_t_east_cut_gas_mass",
        "side_t_east_material_front",
        "dbg_hgas_u_max_current",
        "dbg_hgas_u_max_x",
        "dbg_hgas_u_max_rho_ratio",
        "dbg_vgas_u_max_current",
        "dbg_vgas_u_max_z",
        "dbg_vgas_u_max_rho_ratio",
    )
    diagnostics = {
        key: [float(value) for value in rec.get(key, [])]
        for key in diagnostic_keys
    }
    diagnostics.update(
        external_horizontal_used=bool(
            rec.get("external_horizontal_used", False)
        ),
        conservative_tjunction=True,
        external_horizontal_handoff_time=(
            None
            if rec.get("external_horizontal_handoff_time") is None
            else float(rec["external_horizontal_handoff_time"])
        ),
        initial_bore_impact_time=(
            None
            if rec.get("initial_bore_impact_time") is None
            else float(rec["initial_bore_impact_time"])
        ),
        initial_bore_reflection_time=(
            None
            if rec.get("initial_bore_reflection_time") is None
            else float(rec["initial_bore_reflection_time"])
        ),
        dbg_created={
            key: float(value)
            for key, value in rec.get("dbg_created", {}).items()
        },
        riser_film_closure={
            key: float(value)
            for key, value in rec.get("riser_film_closure", {}).items()
        },
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    L = case.riser_height
    nF = len(rec["frames_t"])
    # ``run_network`` now records at the requested physical interval.  Keep
    # every solver output instead of resampling to the historical fixed count
    # of 131, which silently truncated the visual comparison at 10 s.
    sel = np.arange(nF, dtype=int)
    if args.no_render:
        np.savez_compressed(
            fields_path,
            time=np.asarray([rec["frames_t"][k] for k in sel], dtype=float),
            z=np.asarray(zr, dtype=float),
            alpha_l=np.vstack(
                [np.asarray(rec["frames_alr"][k]) for k in sel]
            ),
            alpha_g=np.vstack(
                [np.asarray(rec["frames_agr"][k]) for k in sel]
            ),
            vertical_liquid_velocity=np.vstack(
                [np.asarray(rec["frames_ulr"][k]) for k in sel]
            ),
            horizontal_alpha_l=np.vstack(
                [np.asarray(rec["frames_alt"][k]) for k in sel]
            ),
            horizontal_alpha_l_raw=np.vstack(
                [np.asarray(rec["frames_alt_raw"][k]) for k in sel]
            ),
            horizontal_liquid_velocity=np.vstack(
                [np.asarray(rec["frames_ult"][k]) for k in sel]
            ),
            horizontal_gas_mass=np.vstack(
                [np.asarray(rec["frames_mgt"][k]) for k in sel]
            ),
        )
        print(f"solver fields only -> {fields_path}")
        return
    initial_liquid_equivalent_height = float(
        np.sum(np.asarray(rec["frames_alr"][0], dtype=float)) * dz
    )
    liquid_height_grid_offset = (
        initial_liquid_equivalent_height - case.init_water_level
    )

    # TRUE geometry, 1:1 aspect (reviewer request): tunnel bore = real D,
    # tower bore = real Dt.  At this scale the 57.1 mm tower is a thin column
    # on the 4 m tunnel axis -- in-tower detail is carried by the synced zoom
    # panel on the right of the viewer.
    pipe_h = case.D                     # 0.094 m
    riser_w = case.Dr                   # 0.0571 m
    C_W, C_A = "#2b7fff", "#f2f4f8"
    handles = [Patch(facecolor=C_W, label="water"),
               Patch(facecolor=C_A, edgecolor="0.5", label="air")]

    def draw_riser_phases(ax, x0, width, water_top, gas_nose, gas_fraction):
        """Draw axial 1-D riser fronts without extending a mixed cell.

        ``water_top`` and ``gas_nose`` are sub-cell material positions.  The
        cross-section is liquid-filled up to the bulk free surface; a centred
        gas core is then overpainted only below its own nose.  Consequently
        the liquid slug between the two fronts is full width and the top free
        surface cannot acquire a cell-rendering notch.
        """

        top = float(np.clip(water_top, 0.0, L))
        nose = float(np.clip(gas_nose, 0.0, top))
        if top > 0.0:
            ax.add_patch(Rectangle(
                (x0, 0.0), width, top, facecolor=C_W, edgecolor="none",
            ))
        for zi, g in zip(zr, np.clip(gas_fraction, 0.0, 1.0)):
            z0 = max(float(zi - 0.5 * dz), 0.0)
            z1 = min(float(zi + 0.5 * dz), nose)
            if g <= 0.01 or z1 <= z0:
                continue
            gas_width = math.sqrt(float(g)) * width
            ax.add_patch(Rectangle(
                (x0 + 0.5 * (width - gas_width), z0),
                gas_width, z1 - z0,
                facecolor=C_A, edgecolor="none",
            ))
        if top - nose > 1.0e-9:
            ax.plot(
                [x0, x0 + width], [top, top],
                color="#1d4ed8", linewidth=0.7, zorder=5,
            )

    index = []
    for n, k in enumerate(sel):
        t_k = float(rec["frames_t"][k])
        alt = rec["frames_alt"][k]
        alr = rec["frames_alr"][k]
        agr = np.clip(np.asarray(rec["frames_agr"][k]), 0.0, 1.0)
        # ``frames_alr`` is already the resolved liquid area fraction.
        Ar_frac = np.clip(np.asarray(alr), 0.0, 1.0)
        material_height = (
            float(rec["wtop"][k]) if k < len(rec["wtop"]) else 0.0
        )
        liquid_equivalent_height = max(
            float(np.sum(Ar_frac) * dz) - liquid_height_grid_offset,
            0.0,
        )
        visible_liquid = np.flatnonzero(Ar_frac > 0.08)
        visible_water_top = (
            float(zr[visible_liquid[-1]] + 0.5 * dz)
            if visible_liquid.size
            else 0.0
        )
        itop = float(rec["itop"][k]) if k < len(rec["itop"]) else 0.0
        core_mass = float(rec["core_mass"][k]) if k < len(rec["core_mass"]) else 0.0
        head = float(rec["pocket_head"][k]) if k < len(rec["pocket_head"]) else 0.0

        # ------- global view, TRUE geometry, equal aspect (x:y = 1:1) -------
        fig, ax = plt.subplots(figsize=(14.0, 3.6))
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A,
                               edgecolor="none"))
        # ``alt`` is circular-pipe wetted AREA fraction, not water-depth
        # fraction.  Convert it with the exact circular-segment geometry and
        # linearly reconstruct the plotted surface through the finite-volume
        # cell averages.  This preserves every computed value and wave
        # amplitude while avoiding a misleading 40-mm staircase of rectangles;
        # no spatial or temporal filter is applied to the solution.
        horizontal_alpha = np.clip(np.asarray(alt, dtype=float), 0.0, 1.0)
        horizontal_depth = pipe_h * _depth_frac(horizontal_alpha)
        surface_x = np.concatenate(([0.0], np.asarray(xt), [case.L_tunnel]))
        surface_y = -pipe_h + np.concatenate(
            ([horizontal_depth[0]], horizontal_depth, [horizontal_depth[-1]])
        )
        ax.fill_between(
            surface_x,
            -pipe_h,
            surface_y,
            where=surface_y > -pipe_h + 1.0e-12,
            interpolate=True,
            facecolor=C_W,
            edgecolor="none",
        )
        # Tower: map the two sub-cell axial material fronts.  Mixed-cell gas
        # is clipped at ``itop`` instead of being painted through the upper
        # liquid slug and bulk free surface.
        # ``x_riser`` is the physical centreline used by the OpenFOAM mesh.
        # Use the same convention here so the two animations overlay exactly.
        riser_left = x_r - 0.5 * riser_w
        riser_right = x_r + 0.5 * riser_w
        ax.add_patch(Rectangle((riser_left, 0), riser_w, L, facecolor=C_A,
                               edgecolor="none"))
        draw_riser_phases(
            ax, riser_left, riser_w, material_height, itop, agr
        )
        # One connected pipe--tower outline: interrupt the pipe crown across
        # the tower bore and never draw a separating bottom wall in the tower.
        wall = dict(color="0.35", lw=0.8, zorder=10)
        ax.plot([0, case.L_tunnel], [-pipe_h, -pipe_h], **wall)
        ax.plot([0, 0], [-pipe_h, 0], **wall)
        ax.plot([case.L_tunnel, case.L_tunnel], [-pipe_h, 0], **wall)
        ax.plot([0, riser_left], [0, 0], **wall)
        ax.plot([riser_right, case.L_tunnel], [0, 0], **wall)
        ax.plot([riser_left, riser_left], [0, L], **wall)
        ax.plot([riser_right, riser_right], [0, L], **wall)
        ax.text(0.01, 0.95,
                f"Time = {t_k:.2f} s    riser liquid-trace top = "
                f"{material_height:.3f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.04, L + 0.10)
        ax.set_aspect("equal", adjustable="box")   # true 1:1 proportions
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(f"Case A (true scale 1:1) -- D={case.D*1000:.0f} mm, "
                     f"Dt={case.Dr*1000:.1f} mm (Dt/D={case.Dr/case.D:.2f}), "
                     f"Ha0={case.air_head} m, WL0={case.init_water_level} m, L={L} m",
                     fontsize=9)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(frames_dir / f"frame_{n:04d}.png", dpi=130)
        plt.close(fig)

        # ---------- tower zoom (area-preserving centred circular gas core) ----------
        fig, ax = plt.subplots(figsize=(2.6, 6.2))
        ax.add_patch(Rectangle((0, 0), 1, L, facecolor=C_A, edgecolor="none"))
        draw_riser_phases(ax, 0.0, 1.0, material_height, itop, agr)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, L)
        ax.set_xticks([])
        ax.set_ylabel("height above pipe crown [m]", fontsize=8)
        ax.set_title(f"tower zoom\nt={t_k:.2f} s", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(riser_frames_dir / f"riser_{n:04d}.png", dpi=110)
        plt.close(fig)

        index.append(dict(
            file=(frames_dir / f"frame_{n:04d}.png").relative_to(CASE).as_posix(),
            riserFile=(riser_frames_dir / f"riser_{n:04d}.png").relative_to(CASE).as_posix(),
            time=round(t_k, 3),
            wtop=round(liquid_equivalent_height, 3),
            materialHeight=round(material_height, 3),
            visibleWaterTop=round(visible_water_top, 3),
            itop=round(itop, 3),
            coreMassMg=round(core_mass * 1e6, 2),
            head=round(head, 3),
        ))

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    np.savez_compressed(
        fields_path,
        time=np.asarray([rec["frames_t"][k] for k in sel], dtype=float),
        z=np.asarray(zr, dtype=float),
        alpha_l=np.vstack([np.asarray(rec["frames_alr"][k]) for k in sel]),
        alpha_g=np.vstack([np.asarray(rec["frames_agr"][k]) for k in sel]),
        vertical_liquid_velocity=np.vstack(
            [np.asarray(rec["frames_ulr"][k]) for k in sel]
        ),
        horizontal_alpha_l=np.vstack(
            [np.asarray(rec["frames_alt"][k]) for k in sel]
        ),
        horizontal_alpha_l_raw=np.vstack(
            [np.asarray(rec["frames_alt_raw"][k]) for k in sel]
        ),
        horizontal_liquid_velocity=np.vstack(
            [np.asarray(rec["frames_ult"][k]) for k in sel]
        ),
        horizontal_gas_mass=np.vstack(
            [np.asarray(rec["frames_mgt"][k]) for k in sel]
        ),
    )
    print(f"{len(index)} frames -> {frames_dir} / {riser_frames_dir}")
    print(f"-> {index_path}")
    print(f"-> {fields_path}")


if __name__ == "__main__":
    main()
