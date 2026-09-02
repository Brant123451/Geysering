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
from casea_coupled_gas_network import CoupledGasParameters
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
            "Optional horizontal holdup multiplier used only for a declared "
            "sensitivity run; the production companion-model closure is 0."
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
        "--vertical-taylor-core-fraction",
        type=float,
        default=0.80,
        help=(
            "Cross-section-averaged gas-core area fraction used by the "
            "side-fed Taylor shock fit.  Values below the production default "
            "retain more counter-current liquid holdup in the riser; the "
            "solver still advances one conservative net T-junction flux."
        ),
    )
    parser.add_argument(
        "--vertical-taylor-return-efficiency",
        type=float,
        default=1.0,
        help=(
            "Fraction of Taylor-core swept liquid returned through the "
            "counter-current film.  The complement remains in the resolved "
            "upper liquid slug; the T-junction update remains conservative."
        ),
    )
    parser.add_argument(
        "--vertical-ccfl-constant",
        type=float,
        default=0.50,
        help=(
            "Wallis counter-current-flow limitation constant used after "
            "riser material breakthrough."
        ),
    )
    parser.add_argument(
        "--tower-entry-alpha-min",
        type=float,
        default=None,
        help=(
            "Horizontal gas-area fraction required at the T mouth.  The "
            "default uses the capillary crown-opening fraction derived by "
            "CoupledGasParameters instead of an independently fitted cutoff."
        ),
    )
    parser.add_argument(
        "--disable-vertical-twostream",
        action="store_true",
        help=(
            "Keep the established conservative one-stream riser while "
            "auditing horizontal material-front changes."
        ),
    )
    parser.add_argument(
        "--disable-horizontal-front-retreat",
        action="store_true",
        help=(
            "Diagnostic sensitivity: reproduce the historical one-way east "
            "material front while leaving all other closures unchanged."
        ),
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=0.05,
        help="Physical time between saved frames [s].",
    )
    parser.add_argument(
        "--external-horizontal-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume the unchanged pre-T shock-fit branch from an exact "
            "checkpoint; intended for fast post-arrival closure tests."
        ),
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help=(
            "Save conservative solver fields and diagnostics without creating "
            "PNG frames; this changes only post-processing runtime."
        ),
    )
    parser.add_argument(
        "--reuse-saved-fields",
        action="store_true",
        help=(
            "Re-render the selected variant from its existing conservative "
            "NPZ fields without rerunning the 1D solver.  If the frame index "
            "is absent after --no-render, it is rebuilt from diagnostics."
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

    coupled_gas_parameters = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
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
        vertical_taylor_core_area_fraction=(
            args.vertical_taylor_core_fraction
        ),
        vertical_taylor_return_efficiency=(
            args.vertical_taylor_return_efficiency
        ),
        vertical_ccfl_constant=args.vertical_ccfl_constant,
        enable_vertical_twostream=not args.disable_vertical_twostream,
        allow_horizontal_front_retreat=(
            not args.disable_horizontal_front_retreat
        ),
        tower_entry_alpha_min=(
            coupled_gas_parameters.horizontal_capillary_void_fraction
            if args.tower_entry_alpha_min is None
            else args.tower_entry_alpha_min
        ),
    )
    external_horizontal_solver = build_case_a_shockfit_solver(
        dx=(
            case.L_tunnel
            / max(20, int(round(case.L_tunnel / case.ds)))
        ),
        wave_speed=case.a_wh,
    )
    if args.reuse_saved_fields:
        if not fields_path.exists():
            raise FileNotFoundError(
                "--reuse-saved-fields requires the selected variant's existing "
                f"field file: {fields_path}"
            )
        saved = np.load(fields_path)
        saved_time = np.asarray(saved["time"], dtype=float)
        saved_diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.exists()
            else {}
        )

        def diagnostic_series(key, default=0.0):
            values = saved_diagnostics.get(key, [])
            if len(values) != saved_time.size:
                return [float(default)] * saved_time.size
            return [float(value) for value in values]

        if index_path.exists():
            saved_index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            saved_wtop = diagnostic_series("wtop")
            saved_itop = diagnostic_series("itop")
            saved_core_mass = diagnostic_series("core_mass")
            saved_head = diagnostic_series("pocket_head")
            saved_index = [
                {
                    "time": float(time),
                    "materialHeight": saved_wtop[index],
                    "itop": saved_itop[index],
                    "coreMassMg": 1.0e6 * saved_core_mass[index],
                    "head": saved_head[index],
                }
                for index, time in enumerate(saved_time)
            ]
            print(
                "frame index absent; rebuilding render metadata from "
                f"{diagnostics_path}"
            )
        if len(saved_index) != saved_time.size:
            raise ValueError(
                "Saved frame index and conservative field archive have "
                "different frame counts."
            )
        saved_z = np.asarray(saved["z"], dtype=float)
        saved_horizontal = np.asarray(saved["horizontal_alpha_l"], dtype=float)
        n_tunnel = int(saved_horizontal.shape[1])
        saved_dx = case.L_tunnel / n_tunnel
        saved_dz = (
            float(np.median(np.diff(saved_z)))
            if saved_z.size > 1
            else float(args.dz)
        )

        def rows(key):
            return [np.asarray(row, dtype=float) for row in saved[key]]

        rec = dict(
            frames_t=saved_time.tolist(),
            xt=(np.arange(n_tunnel, dtype=float) + 0.5) * saved_dx,
            zr=saved_z,
            dx=saved_dx,
            dz=saved_dz,
            frames_alr=rows("alpha_l"),
            frames_agr=rows("alpha_g"),
            frames_mgr=rows("vertical_gas_mass"),
            frames_mgrs=rows("vertical_tracer_mass"),
            frames_jgrs=rows("vertical_gas_momentum"),
            frames_ulr=rows("vertical_liquid_velocity"),
            frames_riser_upward_area=rows("riser_upward_area"),
            frames_riser_downward_area=rows("riser_downward_area"),
            frames_riser_upward_discharge=rows("riser_upward_discharge"),
            frames_riser_downward_discharge=rows("riser_downward_discharge"),
            frames_alt=rows("horizontal_alpha_l"),
            frames_alt_raw=rows("horizontal_alpha_l_raw"),
            frames_ult=rows("horizontal_liquid_velocity"),
            frames_qlt=rows("horizontal_liquid_discharge"),
            frames_mgt=rows("horizontal_gas_mass"),
            frames_jgt=rows("horizontal_gas_momentum"),
            wtop=[float(item.get("materialHeight", 0.0)) for item in saved_index],
            itop=[float(item.get("itop", 0.0)) for item in saved_index],
            core_mass=[
                float(item.get("coreMassMg", 0.0)) * 1.0e-6
                for item in saved_index
            ],
            pocket_head=[float(item.get("head", 0.0)) for item in saved_index],
            riser_breakthrough=diagnostic_series("riser_breakthrough"),
            external_horizontal_used=True,
        )
        saved.close()
        print(f"reusing conservative solver fields -> {fields_path}")
    else:
        rec = run_network(
            case,
            verbose=True,
            external_horizontal_solver=external_horizontal_solver,
            external_horizontal_checkpoint=(
                args.external_horizontal_checkpoint
            ),
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
        "junction_vertical_characteristic_liquid_flux",
        "junction_taylor_return_liquid_flux",
        "junction_gas_mouth_fraction",
        "junction_gross_upward_liquid_flux",
        "junction_gross_downward_liquid_flux",
        "junction_countercurrent_circulation_flux",
        "twostream_upward_volume_residual",
        "twostream_downward_volume_residual",
        "twostream_provenance_volume_residual",
        "twostream_horizontal_source_volume",
        "twostream_initial_source_volume",
        "twostream_drag_momentum_residual",
        "twostream_bottom_0p1m_inventory",
        "twostream_active",
        "tnode_pressure_balance_residual",
        "tnode_pressure_raw_residual",
        "tnode_downward_pressure_balance_residual",
        "tnode_downward_pressure_raw_residual",
        "tnode_capacity_pressure_impulse",
        "tnode_capacity_pressure",
        "tnode_capacity_upward_rate_correction",
        "tnode_capacity_downward_rate_correction",
        "tnode_capacity_kkt_residual",
        "tnode_capacity_packing_residual",
        "tnode_capacity_donor_residual",
        "tnode_capacity_donor_multiplier",
        "tnode_capacity_active_cells",
        "tnode_capacity_topology_iterations",
        "tnode_momentum_balance_residual",
        "tnode_physical_reaction_pressure",
        "tnode_vertical_mouth_pressure",
        "twostream_bottom_pressure",
        "tnode_fv_mouth_pressure_residual",
        "tnode_gas_reaction_requested",
        "tnode_gas_reaction_applied",
        "tnode_gas_reaction_application_residual",
        "tnode_liquid_gas_action_residual",
        "combined_interphase_momentum_residual",
        "tnode_cell0_drag_length_fraction",
        "tnode_horizontal_liquid_pressure",
        "tnode_horizontal_liquid_pressure_raw",
        "tnode_vertical_liquid_pressure",
        "tnode_upward_old_speed",
        "tnode_upward_unconstrained_speed",
        "tnode_upward_characteristic_speed",
        "tnode_upward_characteristic_rate",
        "tnode_first_cell_downward_rate",
        "tnode_first_cell_downward_speed",
        "tnode_outgoing_mouth_downward_rate",
        "tnode_positive_net_receiving_capacity",
        "tnode_node_liquid_volume",
        "tnode_downward_donor_volume",
        "tnode_mouth_upward_area",
        "tnode_mouth_downward_area",
        "tnode_mouth_gas_area",
        "tnode_mouth_liquid_area",
        "tnode_wallis_downward_reference",
        "tnode_downward_constraint_reaction_flux",
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
        "side_t_east_topology_front",
        "side_t_east_material_front_velocity",
        "side_t_east_retired_cell_count",
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
        twostream_activated_time=(
            None
            if rec.get("twostream_activated_time") is None
            else float(rec["twostream_activated_time"])
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
    if not args.reuse_saved_fields:
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
            alpha_l_raw=np.vstack(
                [np.asarray(rec["frames_alr_raw"][k]) for k in sel]
            ),
            alpha_g=np.vstack(
                [np.asarray(rec["frames_agr"][k]) for k in sel]
            ),
            vertical_gas_mass=np.vstack(
                [np.asarray(rec["frames_mgr"][k]) for k in sel]
            ),
            vertical_tracer_mass=np.vstack(
                [np.asarray(rec["frames_mgrs"][k]) for k in sel]
            ),
            vertical_gas_momentum=np.vstack(
                [np.asarray(rec["frames_jgrs"][k]) for k in sel]
            ),
            vertical_liquid_velocity=np.vstack(
                [np.asarray(rec["frames_ulr"][k]) for k in sel]
            ),
            riser_upward_area=np.vstack(
                [np.asarray(rec["frames_riser_upward_area"][k]) for k in sel]
            ),
            riser_downward_area=np.vstack(
                [np.asarray(rec["frames_riser_downward_area"][k]) for k in sel]
            ),
            riser_upward_discharge=np.vstack(
                [np.asarray(rec["frames_riser_upward_discharge"][k]) for k in sel]
            ),
            riser_downward_discharge=np.vstack(
                [np.asarray(rec["frames_riser_downward_discharge"][k]) for k in sel]
            ),
            riser_horizontal_source_upward_area=np.vstack(
                [
                    np.asarray(
                        rec["frames_riser_horizontal_source_upward_area"][k]
                    )
                    for k in sel
                ]
            ),
            riser_horizontal_source_downward_area=np.vstack(
                [
                    np.asarray(
                        rec["frames_riser_horizontal_source_downward_area"][k]
                    )
                    for k in sel
                ]
            ),
            riser_initial_source_area=np.vstack(
                [
                    np.asarray(rec["frames_riser_initial_source_area"][k])
                    for k in sel
                ]
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
            horizontal_liquid_discharge=np.vstack(
                [np.asarray(rec["frames_qlt"][k]) for k in sel]
            ),
            horizontal_gas_mass=np.vstack(
                [np.asarray(rec["frames_mgt"][k]) for k in sel]
            ),
            horizontal_gas_momentum=np.vstack(
                [np.asarray(rec["frames_jgt"][k]) for k in sel]
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

    def draw_riser_phases(
        ax,
        x0,
        width,
        liquid_fraction,
        gas_fraction,
    ):
        """Render conserved riser phase areas with a planar upper free surface.

        ``liquid_fraction`` is the conserved liquid cross-sectional area in
        every axial cell.  Internal partial cells are drawn as an annular
        liquid film around a centred gas/air core whose diameter ratio is
        ``sqrt(1-alpha_l)``.  The uppermost liquid-bearing cell is different:
        it is the axial cut made by the bulk free surface, so its conserved
        volume is reconstructed as a full-width horizontal layer of height
        ``alpha_l*dz``.  This removes the nonphysical centre notch that the
        annular-core reconstruction otherwise creates at the top surface.

        ``gas_fraction`` is retained in the signature because it is the
        independently conserved tunnel-gas tracer used by the diagnostics.
        Air and tracer gas have the same colour in this phase-only viewer, so
        it does not alter the liquid geometry drawn here.
        """

        liquid = np.clip(np.asarray(liquid_fraction, dtype=float), 0.0, 1.0)
        np.asarray(gas_fraction, dtype=float)  # validate array-like input
        visible = np.flatnonzero(liquid > 1.0e-6)
        top_index = int(visible[-1]) if visible.size else -1
        for cell_index, (zi, alpha_l) in enumerate(zip(zr, liquid)):
            if alpha_l <= 1.0e-6:
                continue
            z0 = max(float(zi - 0.5 * dz), 0.0)
            z1 = min(float(zi + 0.5 * dz), L)
            if z1 <= z0:
                continue
            if cell_index == top_index and alpha_l < 1.0 - 1.0e-6:
                # A planar sub-cell cut preserves alpha_l*width*(z1-z0), hence
                # it changes only the visualization, not the computed volume.
                water_top = z0 + float(alpha_l) * (z1 - z0)
                ax.add_patch(Rectangle(
                    (x0, z0), width, water_top - z0,
                    facecolor=C_W, edgecolor="none",
                ))
                continue

            ax.add_patch(Rectangle(
                (x0, z0), width, z1 - z0,
                facecolor=C_W, edgecolor="none",
            ))
            if alpha_l < 1.0 - 1.0e-6:
                core_width = math.sqrt(1.0 - float(alpha_l)) * width
                ax.add_patch(Rectangle(
                    (x0 + 0.5 * (width - core_width), z0),
                    core_width, z1 - z0,
                    facecolor=C_A, edgecolor="none",
                ))

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
        visible_liquid = np.flatnonzero(Ar_frac > 1.0e-3)
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
        # Tower: draw the conservative per-cell liquid areas.  The material
        # envelope and tracer front remain diagnostics; neither is substituted
        # for the actual liquid field in the phase image.
        # ``x_riser`` is the physical centreline used by the OpenFOAM mesh.
        # Use the same convention here so the two animations overlay exactly.
        riser_left = x_r - 0.5 * riser_w
        riser_right = x_r + 0.5 * riser_w
        ax.add_patch(Rectangle((riser_left, 0), riser_w, L, facecolor=C_A,
                               edgecolor="none"))
        draw_riser_phases(
            ax,
            riser_left,
            riser_w,
            Ar_frac,
            agr,
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
                f"Time = {t_k:.2f} s    riser liquid-equivalent height = "
                f"{liquid_equivalent_height:.3f} m",
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
        draw_riser_phases(
            ax,
            0.0,
            1.0,
            Ar_frac,
            agr,
        )
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
        vertical_gas_mass=np.vstack(
            [np.asarray(rec["frames_mgr"][k]) for k in sel]
        ),
        vertical_tracer_mass=np.vstack(
            [np.asarray(rec["frames_mgrs"][k]) for k in sel]
        ),
        vertical_gas_momentum=np.vstack(
            [np.asarray(rec["frames_jgrs"][k]) for k in sel]
        ),
        vertical_liquid_velocity=np.vstack(
            [np.asarray(rec["frames_ulr"][k]) for k in sel]
        ),
        riser_upward_area=np.vstack(
            [np.asarray(rec["frames_riser_upward_area"][k]) for k in sel]
        ),
        riser_downward_area=np.vstack(
            [np.asarray(rec["frames_riser_downward_area"][k]) for k in sel]
        ),
        riser_upward_discharge=np.vstack(
            [np.asarray(rec["frames_riser_upward_discharge"][k]) for k in sel]
        ),
        riser_downward_discharge=np.vstack(
            [np.asarray(rec["frames_riser_downward_discharge"][k]) for k in sel]
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
        horizontal_liquid_discharge=np.vstack(
            [np.asarray(rec["frames_qlt"][k]) for k in sel]
        ),
        horizontal_gas_mass=np.vstack(
            [np.asarray(rec["frames_mgt"][k]) for k in sel]
        ),
        horizontal_gas_momentum=np.vstack(
            [np.asarray(rec["frames_jgt"][k]) for k in sel]
        ),
    )
    print(f"{len(index)} frames -> {frames_dir} / {riser_frames_dir}")
    print(f"-> {index_path}")
    print(f"-> {fields_path}")


if __name__ == "__main__":
    main()
