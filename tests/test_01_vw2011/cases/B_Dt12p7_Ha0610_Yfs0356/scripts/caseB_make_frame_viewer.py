# -*- coding: utf-8 -*-
"""Case B interactive frame viewer assets (same design as case A):

  outputs/frames/frame_XXXX.png        -- global 1:1 view (tunnel + tower)
  outputs/riser_frames/riser_XXXX.png  -- tower zoom (gas core width = alpha_g)
  outputs/frames_index.json            -- [{file, riserFile, time, wtop, itop,
                                           coreMassMg, head}, ...]

caseB_digitize_and_compare.py embeds these into report.html as an interactive
viewer (prev/play/slider/next + arrow keys).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
SCANS = CASE_ROOT / "reference" / "paper_scans"
OUTPUTS = CASE_ROOT / "outputs"
sys.path.insert(0, str(MODEL))

from vw2011_network_twofluid import NetworkCase, run_network

FRAMES = OUTPUTS / "frames"
RISER_FRAMES = OUTPUTS / "riser_frames"
N_FRAMES = 90


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    FRAMES.mkdir(parents=True, exist_ok=True)
    RISER_FRAMES.mkdir(parents=True, exist_ok=True)

    # t_end=9.0 s covers the full paper window (T*=5 is 8.64 s for Dt=12.7 mm)
    cfg = json.loads((CASE_ROOT / "config" / "case.json").read_text(encoding="utf-8"))
    case = NetworkCase(
        Dr=float(cfg["tower_diameter_m"]),
        air_head=float(cfg["initial_air_pressure_head_m"]),
        init_water_level=float(cfg["initial_water_level_m"]),
        t_end=9.0,
    )
    rec = run_network(case, verbose=False)

    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    L = case.riser_height
    nF = len(rec["frames_t"])
    sel = np.unique(np.linspace(0, nF - 1, min(N_FRAMES, nF)).astype(int))

    # TRUE geometry, 1:1 aspect: tunnel bore = real D (94 mm), tower bore =
    # real Dt (12.7 mm) -- at this scale the case-B tower is a hairline on the
    # 4 m tunnel; in-tower detail is carried by the synced zoom panel.
    pipe_h = case.D
    riser_w = case.Dr
    C_W, C_A = "#2b7fff", "#f2f4f8"
    handles = [Patch(facecolor=C_W, label="water"),
               Patch(facecolor=C_A, edgecolor="0.5", label="air")]

    index = []
    for n, k in enumerate(sel):
        t_k = float(rec["frames_t"][k])
        alt = rec["frames_alt"][k]
        alr = rec["frames_alr"][k]
        agr = np.clip(np.asarray(rec["frames_agr"][k]), 0.0, 1.0)
        Ar_frac = np.clip(np.asarray(alr), 0.0, 1.0)   # frames_alr already stored as fraction
        wtop = float(rec["wtop"][k]) if k < len(rec["wtop"]) else 0.0
        itop = float(rec["itop"][k]) if k < len(rec["itop"]) else 0.0
        core_mass = float(rec["core_mass"][k]) if k < len(rec["core_mass"]) else 0.0
        head = float(rec["pocket_head"][k]) if k < len(rec["pocket_head"]) else 0.0

        # ------- global view, TRUE geometry, equal aspect (x:y = 1:1) -------
        fig, ax = plt.subplots(figsize=(14.0, 3.6))
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A,
                               edgecolor="0.5", lw=0.8))
        for xi, ai in zip(xt, alt):
            f = float(min(max(ai, 0.0), 1.0))
            if f > 0.03:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, f * pipe_h,
                                       facecolor=C_W, edgecolor="none"))
        ax.add_patch(Rectangle((x_r, 0), riser_w, L, facecolor=C_A, edgecolor="0.5", lw=0.8))
        for zi, fl, g in zip(zr, Ar_frac, agr):
            if fl > 0.02:
                if fl >= 0.98 and g <= 0.01:
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), riser_w, dz,
                                           facecolor=C_W, edgecolor="none"))
                else:
                    fw = 0.5 * fl * riser_w
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), fw, dz,
                                           facecolor=C_W, edgecolor="none"))
                    ax.add_patch(Rectangle((x_r + riser_w - fw, zi - 0.5 * dz), fw, dz,
                                           facecolor=C_W, edgecolor="none"))
            if g > 0.01 and zi <= wtop:
                gw = g * riser_w
                ax.add_patch(Rectangle((x_r + 0.5 * (riser_w - gw), zi - 0.5 * dz), gw, dz,
                                       facecolor="white", edgecolor="none"))
        ax.plot([x_r - 0.06, x_r + riser_w + 0.06], [L, L], color="#ef4444", ls="--", lw=1.0)
        ax.text(0.01, 0.95, f"t = {t_k:.2f} s    tower visible water level = {wtop:.2f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.04, L + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(f"Case B (true scale 1:1) -- D={case.D*1000:.0f} mm, "
                     f"Dt={case.Dr*1000:.1f} mm (Dt/D={case.Dr/case.D:.3f}), "
                     f"Ha0={case.air_head} m, WL0={case.init_water_level} m, L={L} m",
                     fontsize=9)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(FRAMES / f"frame_{n:04d}.png", dpi=130)
        plt.close(fig)

        # ---------- tower zoom (gas-core width = local alpha_g) ----------
        fig, ax = plt.subplots(figsize=(2.6, 6.2))
        ax.add_patch(Rectangle((0, 0), 1, min(wtop, L), facecolor=C_W, edgecolor="none"))
        ax.add_patch(Rectangle((0, min(wtop, L)), 1, L - min(wtop, L),
                               facecolor=C_A, edgecolor="none"))
        for zi, g in zip(zr, agr):
            if zi <= wtop and g > 0.01:
                ax.add_patch(Rectangle((0.5 * (1 - g), zi - 0.5 * dz), g, dz,
                                       facecolor="white", edgecolor="none"))
        ax.axhline(wtop, color="#1f4ed8", lw=1.4)
        if itop > 0:
            ax.axhline(itop, color="#ef4444", lw=1.2, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, L)
        ax.set_xticks([])
        ax.set_ylabel("height above pipe crown [m]", fontsize=8)
        ax.set_title(f"tower zoom\nt={t_k:.2f} s", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(RISER_FRAMES / f"riser_{n:04d}.png", dpi=110)
        plt.close(fig)

        index.append(dict(
            file=f"frames/frame_{n:04d}.png",
            riserFile=f"riser_frames/riser_{n:04d}.png",
            time=round(t_k, 3),
            wtop=round(wtop, 3),
            itop=round(itop, 3),
            coreMassMg=round(core_mass * 1e6, 2),
            head=round(head, 3),
        ))

    (OUTPUTS / "frames_index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(index)} frames -> {FRAMES} / {RISER_FRAMES}")
    print(f"-> {OUTPUTS / 'frames_index.json'}")


if __name__ == "__main__":
    main()
