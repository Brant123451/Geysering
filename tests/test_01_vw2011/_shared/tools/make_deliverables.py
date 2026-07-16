# -*- coding: utf-8 -*-
"""Run the two V&W(2011) target cases with our two-fluid model and produce, per case,
a trajectory CSV + figure into its dedicated sub-folder, plus a combined model-vs-paper
comparison (Table 2). One run of each case; full resolution by default.

Case A (Fig.4):  D_t=57.1 mm (D_t/D=0.607), Ha0=0.305, Yfs0=0.356  -> paper: NO geyser (rise<0.1L)
Case B (geyser): D_t=12.7 mm (D_t/D=0.135), Ha0=0.610, Yfs0=0.356  -> paper: geyser every run
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np

from vw2011_network_twofluid import NetworkCase, run_network, G

HERE = Path(__file__).resolve().parent

CASES = [
    {"tag": "caseA_Dt57p1_Ha0305_Yfs0356", "name": "Case A  D_t/D=0.607 (large riser)",
     "Dr": 0.0571, "Ha0": 0.305, "Yfs0": 0.356,
     "paper": {"V_fs": 0.048, "V_int": 0.39, "rise_frac": 0.10, "geyser": False}},
    {"tag": "caseB_Dt12p7_Ha0610_Yfs0356", "name": "Case B  D_t/D=0.135 (geyser)",
     "Dr": 0.0127, "Ha0": 0.610, "Yfs0": 0.356,
     "paper": {"V_fs": 0.44, "V_int": 1.43, "rise_frac": 1.00, "geyser": True}},
]


def _mean_velocity(t, y, y_lo, y_hi):
    t = np.asarray(t); y = np.asarray(y)
    above = np.where(y >= y_lo)[0]
    if above.size == 0:
        return 0.0
    i0 = int(above[0])
    reach = np.where(y >= y_hi)[0]
    i1 = int(reach[0]) if reach.size else int(np.argmax(y))
    if i1 <= i0 or t[i1] <= t[i0]:
        return 0.0
    return float((y[i1] - y[i0]) / (t[i1] - t[i0]))


def run_one(spec, ds, dz, tend):
    case = NetworkCase(Dr=spec["Dr"], air_head=spec["Ha0"], init_water_level=spec["Yfs0"],
                       t_end=tend, ds=ds, dz=dz)
    rec = run_network(case, verbose=False)
    t = np.asarray(rec["t"]); wtop = np.asarray(rec["wtop"]); itop = np.asarray(rec["itop"])
    pj = np.asarray(rec["pj_head"]); ph = np.asarray(rec["pocket_head"])
    n = min(len(t), len(pj))
    t, wtop, itop, ph = t[:n], wtop[:n], itop[:n], ph[:n]; pj = pj[:n]
    L = case.riser_height; sgd = math.sqrt(G * spec["Dr"])
    res = {
        "L": L, "sgd": sgd, "Yfs0": spec["Yfs0"],
        "Yfs_max": float(wtop.max()), "Yint_max": float(itop.max()),
        "reaches_top": bool(wtop.max() >= 0.98 * L),
        "rise_frac": float((wtop.max() - spec["Yfs0"]) / L),
        "V_fs_star": _mean_velocity(t, wtop, spec["Yfs0"] + 0.01,
                                    min(spec["Yfs0"] + 0.9 * (L - spec["Yfs0"]), L - 0.005)) / sgd,
        "V_int_star": _mean_velocity(t, itop, 0.05 * L, 0.85 * L) / sgd,
        "t": t, "Yfs": wtop, "Yint": itop, "pj_head": pj, "pocket_head": ph,
    }
    return case, res


def save_case_outputs(spec, case, res):
    out = HERE / spec["tag"] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    # trajectory CSV
    with (out / "trajectory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Yfs_m", "Yint_m", "Yfs_over_L", "Yint_over_L", "pj_head_m", "pocket_head_m"])
        for i in range(len(res["t"])):
            w.writerow([f"{res['t'][i]:.4f}", f"{res['Yfs'][i]:.4f}", f"{res['Yint'][i]:.4f}",
                        f"{res['Yfs'][i]/res['L']:.4f}", f"{res['Yint'][i]/res['L']:.4f}",
                        f"{res['pj_head'][i]:.4f}", f"{res['pocket_head'][i]:.4f}"])
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    L = res["L"]
    ax1.plot(res["t"], res["Yfs"] / L, color="#2b7fff", lw=2.2, label="free surface $Y_{fs}/L$ (model)")
    ax1.plot(res["t"], res["Yint"] / L, color="#ef4444", lw=1.8, ls="--", label="gas nose $Y_{int}/L$ (model)")
    ax1.axhline(1.0, color="#16a34a", ls=":", lw=1.4, label="tower top (geyser)")
    ax1.axhline(res["Yfs0"] / L, color="0.6", ls=":", lw=1.0)
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("height / L")
    ax1.set_ylim(0, 1.08); ax1.legend(frameon=False, fontsize=8); ax1.grid(alpha=0.25)
    ax1.set_title(f"{spec['name']}: surface & gas-nose trajectory")
    # speed comparison bars
    labels = ["rise/L", "V_fs*", "V_int*"]
    model_vals = [min(res["rise_frac"], 1.0) if not res["reaches_top"] else 1.0,
                  res["V_fs_star"], res["V_int_star"]]
    paper_vals = [spec["paper"]["rise_frac"], spec["paper"]["V_fs"], spec["paper"]["V_int"]]
    x = np.arange(len(labels)); ww = 0.36
    ax2.bar(x - ww / 2, model_vals, ww, color="#2b7fff", label="model")
    ax2.bar(x + ww / 2, paper_vals, ww, color="#9ca3af", label="paper (Table 2)")
    for xi, mv, pv in zip(x, model_vals, paper_vals):
        ax2.text(xi - ww / 2, mv, f"{mv:.2f}", ha="center", va="bottom", fontsize=8)
        ax2.text(xi + ww / 2, pv, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("model vs paper"); ax2.legend(frameon=False, fontsize=8); ax2.grid(alpha=0.25, axis="y")
    geys = "GEYSER (reaches top)" if res["reaches_top"] else "no geyser"
    fig.suptitle(f"{spec['name']}  ->  model: {geys};  paper: {'geyser' if spec['paper']['geyser'] else 'no geyser'}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "model_vs_paper.png", dpi=150)
    plt.close(fig)
    return out


def main(ds=0.02, dz=0.01, tend=10.0):
    rows = []
    for spec in CASES:
        case, res = run_one(spec, ds, dz, tend)
        out = save_case_outputs(spec, case, res)
        p = spec["paper"]
        print(f"\n=== {spec['name']} -> {out} ===")
        print(f"  reaches_top (geyser): model {res['reaches_top']}   paper {p['geyser']}")
        print(f"  rise/L : model {res['rise_frac']:.3f}   paper {p['rise_frac']}")
        print(f"  V_fs*  : model {res['V_fs_star']:.3f}   paper {p['V_fs']}")
        print(f"  V_int* : model {res['V_int_star']:.3f}   paper {p['V_int']}")
        rows.append((spec, res))
    return rows


if __name__ == "__main__":
    main()
