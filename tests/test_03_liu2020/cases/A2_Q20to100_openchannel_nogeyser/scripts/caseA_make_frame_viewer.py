# -*- coding: utf-8 -*-
"""Liu2020 Case A2 interactive frame viewer assets (same design as the VW2011
Case A viewer): per-frame PNG renders of the whole rig + a JSON index; the
report builder embeds them with a play/slider/arrow-key control bar.

Outputs:
  outputs/frames/frame_XXXX.png
  outputs/frames_index.json   [{file, time, S, hr, Qin, Qout}, ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from liu2020_network_twofluid import LiuCase, run_case  # noqa: E402

OUT = HERE / "outputs"
FRAMES = OUT / "frames"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    FRAMES.mkdir(parents=True, exist_ok=True)
    case = LiuCase(t_end=14.0)
    rec = run_case(case, verbose=False)

    t_hist = np.asarray(rec["t"])
    hr_hist = np.asarray(rec["hr"])
    S_hist = np.asarray(rec["S"])
    qin_hist = np.asarray(rec["Qin"])
    qout_hist = np.asarray(rec["Qout"])

    up_x = np.asarray(rec["up_x"])
    dn_x = np.asarray(rec["dn_x"])
    up_zinv = np.asarray(rec["up_zinv"])
    dn_zinv = np.asarray(rec["dn_zinv"])

    # layout: upstream [0, Lu], chamber [Lu, Lu+Lc], downstream [Lu+Lc, ...]
    xc0 = case.Lu
    xc1 = case.Lu + case.Lc
    xd_off = xc1
    x_rise = 0.5 * (xc0 + xc1)
    w_rise_draw = case.dr                   # TRUE riser width (1:1 aspect below)

    C_W, C_A, C_ST = "#2b7fff", "#f2f4f8", "#94a3b8"
    C_MIX = "#8fc1ff"                       # aerated air-water mixture
    handles = [Patch(facecolor=C_W, label="water"),
               Patch(facecolor=C_MIX, label="air-water mixture"),
               Patch(facecolor=C_A, edgecolor="0.6", label="air")]

    # cell centers leave a half-cell white seam against the chamber walls at
    # 1:1 scale: pad the plot arrays out to the physical pipe ends
    def pad_x(x, L):
        return np.concatenate([[0.0], x, [L]])

    def pad_v(v):
        return np.concatenate([[v[0]], v, [v[-1]]])

    up_xp = pad_x(up_x, case.Lu)
    dn_xp = pad_x(dn_x, case.Ld)
    up_zp = pad_v(up_zinv)
    dn_zp = pad_v(dn_zinv)
    ch_x = np.asarray(rec["ch_x"])          # chamber cell centers, 0..Lc
    ch_xp = pad_x(ch_x, case.Lc)

    # velocity fields for the flow arrows (steady base flow LOOKS static in a
    # depth-only rendering; arrows make the through-flow visible)
    have_u = "frames_up_u" in rec and len(rec.get("frames_up_u", [])) == len(rec["frames_t"])

    nF = len(rec["frames_t"])
    index = []
    for k in range(nF):
        tk = float(rec["frames_t"][k])
        h_up = pad_v(np.asarray(rec["frames_up_h"][k]))
        h_dn = pad_v(np.asarray(rec["frames_dn_h"][k]))
        h_ch = pad_v(np.asarray(rec["frames_ch_h"][k]))
        S_k = float(np.interp(tk, t_hist, S_hist))
        hr_k = float(rec["frames_hr"][k])       # true dynamic column state
        ur_k = float(rec["frames_ur"][k])
        al_k = float(rec["frames_alpha"][k])    # chamber mixture void fraction
        alr_k = float(rec["frames_alpha_r"][k])  # column void fraction
        qin_k = float(np.interp(tk, t_hist, qin_hist))
        qout_k = float(np.interp(tk, t_hist, qout_hist))

        fig, ax = plt.subplots(figsize=(16.0, 3.4))
        # ---- air backgrounds (no edges -- the passages are OPEN through the
        # chamber: wall segments are drawn explicitly below, with gaps where
        # the pipes and the riser connect) ----
        ax.fill_between(up_xp, up_zp, up_zp + case.Du, color=C_A, edgecolor="none")
        ax.add_patch(Rectangle((xc0, 0), case.Lc, case.Hc,
                               facecolor=C_A, edgecolor="none"))
        ax.add_patch(Rectangle((x_rise - w_rise_draw / 2, case.Hc),
                               w_rise_draw, case.Hr,
                               facecolor=C_A, edgecolor="none"))
        xd = xd_off + dn_xp
        ax.fill_between(xd, dn_zp, dn_zp + case.Dd, color=C_A, edgecolor="none")
        # ---- water (continuous through the junctions) ----
        ax.fill_between(up_xp, up_zp, up_zp + np.minimum(h_up, case.Du),
                        color=C_W, edgecolor="none")
        # chamber: the TRUE per-cell free surface (tilted/cresting during the
        # bore arrival), clipped at the lid -- a flat mean-S rectangle hid the
        # crest that locally pins against the lid and feeds the riser
        ax.fill_between(xc0 + ch_xp, 0.0, np.minimum(h_ch, case.Hc),
                        color=C_W, edgecolor="none")
        # aerated mixture swell above the clear-water surface (plunging-jet
        # entrainment): mixture depth = h / (1 - alpha), capped at the lid --
        # it is this bubbly layer, not clear water, that reaches the lid
        if al_k > 0.01:
            h_mix = np.minimum(h_ch / max(1.0 - al_k, 0.55), case.Hc)
            ax.fill_between(xc0 + ch_xp, np.minimum(h_ch, case.Hc), h_mix,
                            color=C_MIX, edgecolor="none")
        else:
            h_mix = np.minimum(h_ch, case.Hc)
        # cells whose mixture is pinned against the lid: darker band under the
        # lid; these cells push mixture up the 0.06 m tap even while the MEAN
        # stage is still below the lid
        pinned = h_mix >= case.Hc - 1e-9
        if pinned.any():
            ax.fill_between(xc0 + ch_xp, case.Hc - 0.015, case.Hc,
                            where=pinned, color="#0b4aa2", edgecolor="none")
        if hr_k > 0:
            ax.add_patch(Rectangle((x_rise - w_rise_draw / 2, case.Hc),
                                   w_rise_draw, min(hr_k, case.Hr),
                                   facecolor=(C_MIX if alr_k > 0.05 else C_W),
                                   edgecolor="none"))
        # riser column velocity arrow (the column is a dynamic slug now)
        if hr_k > 0.02 and abs(ur_k) > 0.05:
            ya0 = case.Hc + 0.5 * min(hr_k, case.Hr)
            ua = float(np.clip(ur_k, -3.0, 3.0))
            ax.annotate("", xy=(x_rise, ya0 + 0.12 * ua), xytext=(x_rise, ya0),
                        arrowprops=dict(arrowstyle="-|>", color="#0b4aa2",
                                        lw=1.4, shrinkA=0, shrinkB=0))
        ax.fill_between(xd, dn_zp, dn_zp + np.minimum(h_dn, case.Dd),
                        color=C_W, edgecolor="none")
        # ---- structure lines: pipes ----
        ax.plot(up_xp, up_zp, color="0.4", lw=1.0)
        ax.plot(up_xp, up_zp + case.Du, color="0.4", lw=1.0)
        ax.plot(xd, dn_zp, color="0.4", lw=1.0)
        ax.plot(xd, dn_zp + case.Dd, color="0.4", lw=1.0)
        # ---- structure lines: chamber walls WITH OPENINGS ----
        zu_in = case.drop                    # upstream pipe invert at the wall
        # left wall: below the upstream pipe opening + above it
        ax.plot([xc0, xc0], [0.0, zu_in], color="0.4", lw=1.2)
        ax.plot([xc0, xc0], [zu_in + case.Du, case.Hc], color="0.4", lw=1.2)
        # right wall: above the downstream pipe opening only (pipe invert = 0)
        ax.plot([xc1, xc1], [case.Dd, case.Hc], color="0.4", lw=1.2)
        # bottom
        ax.plot([xc0, xc1], [0.0, 0.0], color="0.4", lw=1.2)
        # lid with the riser opening
        ax.plot([xc0, x_rise - w_rise_draw / 2], [case.Hc, case.Hc], color="0.4", lw=1.2)
        ax.plot([x_rise + w_rise_draw / 2, xc1], [case.Hc, case.Hc], color="0.4", lw=1.2)
        # riser walls (open top)
        ax.plot([x_rise - w_rise_draw / 2] * 2, [case.Hc, case.Hc + case.Hr],
                color="0.4", lw=1.0)
        ax.plot([x_rise + w_rise_draw / 2] * 2, [case.Hc, case.Hc + case.Hr],
                color="0.4", lw=1.0)
        ax.plot([x_rise - 0.15, x_rise + 0.15],
                [case.Hc + case.Hr, case.Hc + case.Hr],
                color="#ef4444", ls="--", lw=1.0)
        # weir marker at the end
        ax.plot([xd[-1] + 0.05, xd[-1] + 0.05], [0, 0.12], color="0.2", lw=3.0)
        ax.text(xd[-1] + 0.10, 0.14, "weir", fontsize=8, color="0.3")
        # ---- flow arrows (length ~ local velocity): a steady base flow has a
        # time-constant depth field and LOOKS static without them ----
        if have_u:
            u_up = np.asarray(rec["frames_up_u"][k])
            u_dn = np.asarray(rec["frames_dn_u"][k])
            n_up = len(u_up)
            for fx in (0.15, 0.4, 0.65, 0.9):
                i = int(fx * (n_up - 1))
                ua = float(np.clip(u_up[i], -4.0, 4.0))
                if abs(ua) < 0.05:
                    continue
                xa = up_x[i]
                ya = up_zinv[i] + 0.5 * min(float(rec["frames_up_h"][k][i]), case.Du)
                ax.annotate("", xy=(xa + 0.12 * ua, ya), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="-|>", color="#0b4aa2",
                                            lw=1.4, shrinkA=0, shrinkB=0))
            n_dn = len(u_dn)
            for fx in (0.12, 0.4, 0.68, 0.92):
                i = int(fx * (n_dn - 1))
                ua = float(np.clip(u_dn[i], -4.0, 4.0))
                if abs(ua) < 0.05:
                    continue
                xa = xd_off + dn_x[i]
                ya = 0.5 * min(float(rec["frames_dn_h"][k][i]), case.Dd)
                ax.annotate("", xy=(xa + 0.12 * ua, ya), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="-|>", color="#0b4aa2",
                                            lw=1.4, shrinkA=0, shrinkB=0))

        h_ch_max = float(np.max(h_ch))
        ax.text(0.01, 1.06,
                f"t = {tk:5.2f} s    chamber stage S = {S_k:.3f} m "
                f"(local max {h_ch_max:.3f})    void = {al_k:.2f}    "
                f"riser column = {hr_k:.3f} m @ {ur_k:+.2f} m/s    "
                f"Qin = {qin_k*1e3:5.1f} L/s    Qweir = {qout_k*1e3:5.1f} L/s",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
        ax.set_xlim(-0.1, xd[-1] + 0.45)
        ax.set_ylim(-0.08, case.Hc + case.Hr + 0.12)
        ax.set_aspect("equal", adjustable="box")     # TRUE 1:1 proportions
        ax.set_xlabel("distance along the flow path [m]   "
                      "(upstream pipe -> junction chamber + riser -> downstream pipe -> weir)")
        ax.set_ylabel("z [m]")
        ax.set_title(f"Liu2020 Case A2 (true scale 1:1) -- "
                     f"Q {case.Q0*1e3:.0f}->{case.Q1*1e3:.0f} L/s, "
                     f"downstream open channel", fontsize=10, pad=26)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(FRAMES / f"frame_{k:04d}.png", dpi=115)
        plt.close(fig)

        index.append(dict(
            file=f"outputs/frames/frame_{k:04d}.png",
            time=round(tk, 3),
            S=round(S_k, 4),
            hr=round(hr_k, 4),
            ur=round(ur_k, 3),
            alpha=round(al_k, 3),
            Qin=round(qin_k * 1e3, 1),
            Qout=round(qout_k * 1e3, 1),
        ))

    (OUT / "frames_index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(index)} frames -> {FRAMES}")
    print(f"-> {OUT / 'frames_index.json'}")


if __name__ == "__main__":
    main()
