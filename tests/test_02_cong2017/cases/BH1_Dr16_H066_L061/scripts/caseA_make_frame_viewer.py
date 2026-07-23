# -*- coding: utf-8 -*-
"""B-H1 interactive frame viewer assets (same design as the VW2011 cases):

  outputs/frames/frame_XXXX.png        -- global 1:1 view (pipe + riser)
  outputs/riser_frames/riser_XXXX.png  -- riser zoom (gas core width = alpha_g)
  outputs/frames_index.json            -- [{file, riserFile, time, wtop, itop,
                                           coreMassMg, head}, ...]

caseA_run_and_compare.py embeds these into report.html as an interactive
viewer (prev/play/slider/next + arrow keys).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from cong2017_network_twofluid import NetworkCase, run_network

OUT = HERE / "outputs"
FRAMES = OUT / "frames"
RISER_FRAMES = OUT / "riser_frames"
N_FRAMES = 96

CASE_KW = dict(
    D=0.05, Dr=0.016, riser_height=1.8,
    L_up=2.88, L_mid=2.51, L_down=0.61,
    x_riser_at=2.88,
    pocket_downstream=True,
    reservoir_head=0.66,
    air_head=0.0,
    init_water_level=0.66,
    Hop_cap=10.0,
    x_transducer_at=5.85,
)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    FRAMES.mkdir(parents=True, exist_ok=True)
    RISER_FRAMES.mkdir(parents=True, exist_ok=True)

    case = NetworkCase(**CASE_KW, t_end=13.0)
    rec = run_network(case, verbose=False)

    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    HR = case.riser_height
    nF = len(rec["frames_t"])
    sel = np.unique(np.linspace(0, nF - 1, min(N_FRAMES, nF)).astype(int))

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
        Ar_frac = np.clip(np.asarray(alr), 0.0, 1.0)
        wtop = float(rec["wtop"][k]) if k < len(rec["wtop"]) else 0.0
        itop = float(rec["itop"][k]) if k < len(rec["itop"]) else 0.0
        core_mass = float(rec["core_mass"][k]) if k < len(rec["core_mass"]) else 0.0
        head = float(rec["pocket_head"][k]) if k < len(rec["pocket_head"]) else 0.0

        # ------- global view, TRUE geometry, equal aspect -------
        fig, ax = plt.subplots(figsize=(14.0, 4.6))
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A,
                               edgecolor="0.5", lw=0.8))
        for xi, ai in zip(xt, alt):
            f = float(min(max(ai, 0.0), 1.0))
            if f > 0.03:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, f * pipe_h,
                                       facecolor=C_W, edgecolor="none"))
        ax.add_patch(Rectangle((x_r, 0), riser_w, HR, facecolor=C_A, edgecolor="0.5", lw=0.8))
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
        # reservoir cartoon at the inlet
        ax.add_patch(Rectangle((-0.22, -pipe_h - 0.05), 0.2, 0.66 + pipe_h + 0.05,
                               facecolor="#dbeafe", edgecolor="0.5", lw=0.8))
        ax.text(-0.12, 0.70, "tank\n$H_0$", ha="center", va="bottom", fontsize=8)
        # valve + pocket markers
        ax.plot([case.x_valve, case.x_valve], [-pipe_h - 0.03, 0.05], color="#111827",
                ls=":", lw=1.0)
        ax.text(case.x_valve, 0.07, "ball valve x=5.39 m", ha="center", fontsize=7)
        ax.plot([x_r - 0.06, x_r + riser_w + 0.06], [HR, HR], color="#ef4444", ls="--", lw=1.0)
        ax.text(0.02, 0.93, f"t = {t_k:.2f} s    riser visible water level = {wtop:.2f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_xlim(-0.28, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.08, HR + 0.12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(f"Cong 2017 Run B-H1 (true scale 1:1) -- D={case.D*1000:.0f} mm, "
                     f"Dr={case.Dr*1000:.0f} mm (Dr/D={case.Dr/case.D:.2f}), "
                     f"H0=0.66 m, L0={case.L_down} m, riser {HR} m", fontsize=9)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(FRAMES / f"frame_{n:04d}.png", dpi=130)
        plt.close(fig)

        # ---------- riser zoom ----------
        fig, ax = plt.subplots(figsize=(2.6, 6.2))
        ax.add_patch(Rectangle((0, 0), 1, min(wtop, HR), facecolor=C_W, edgecolor="none"))
        ax.add_patch(Rectangle((0, min(wtop, HR)), 1, HR - min(wtop, HR),
                               facecolor=C_A, edgecolor="none"))
        for zi, g in zip(zr, agr):
            if zi <= wtop and g > 0.01:
                ax.add_patch(Rectangle((0.5 * (1 - g), zi - 0.5 * dz), g, dz,
                                       facecolor="white", edgecolor="none"))
        ax.axhline(wtop, color="#1f4ed8", lw=1.4)
        if itop > 0:
            ax.axhline(itop, color="#ef4444", lw=1.2, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, HR)
        ax.set_xticks([])
        ax.set_ylabel("height above riser entrance [m]", fontsize=8)
        ax.set_title(f"riser zoom\nt={t_k:.2f} s", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(RISER_FRAMES / f"riser_{n:04d}.png", dpi=110)
        plt.close(fig)

        index.append(dict(
            file=f"outputs/frames/frame_{n:04d}.png",
            riserFile=f"outputs/riser_frames/riser_{n:04d}.png",
            time=round(t_k, 3),
            wtop=round(wtop, 3),
            itop=round(itop, 3),
            coreMassMg=round(core_mass * 1e6, 2),
            head=round(head, 3),
        ))

    (OUT / "frames_index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(index)} frames -> {FRAMES} / {RISER_FRAMES}")
    print(f"-> {OUT / 'frames_index.json'}")


if __name__ == "__main__":
    main()
