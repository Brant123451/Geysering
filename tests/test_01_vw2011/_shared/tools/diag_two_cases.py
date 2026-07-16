# -*- coding: utf-8 -*-
"""Diagnostic: run the two target V&W(2011) cases with the CURRENT model and
extract the free-surface (Y_fs) and gas-nose (Y_int) trajectories, mean rise
velocities (non-dimensionalised by sqrt(g*Dt)), and the gas accumulation, then
compare against the paper Table 2 values.

Case A (Fig.4): Dt=57.1 mm (Dt/D=0.607), Ha0=0.305 m, Yfs0=0.356 m  -> no geyser, rise<0.1L
Case B (ours):  Dt=12.7 mm (Dt/D=0.135), Ha0=0.610 m, Yfs0=0.356 m  -> geyser every run
"""
from __future__ import annotations
import json
import math

import numpy as np

from vw2011_network_twofluid import NetworkCase, run_network, G

# paper Table 2 (mean non-dimensional upward velocities, /sqrt(g Dt))
PAPER = {
    0.0571: {"Dt_over_D": 0.607, "V_fs": 0.048, "V_int": 0.39, "rise_frac": 0.10},
    0.0127: {"Dt_over_D": 0.135, "V_fs": 0.44, "V_int": 1.43, "rise_frac": 1.00},
}


def _mean_velocity(t, y, y_lo, y_hi):
    """Average upward speed while y climbs through [y_lo, y_hi] (paper-style)."""
    t = np.asarray(t)
    y = np.asarray(y)
    above = np.where(y >= y_lo)[0]
    if above.size == 0:
        return float("nan"), None, None
    i0 = int(above[0])
    reach = np.where(y >= y_hi)[0]
    i1 = int(reach[0]) if reach.size else int(np.argmax(y))
    if i1 <= i0 or t[i1] <= t[i0]:
        return 0.0, (t[i0], y[i0]), (t[i1], y[i1])
    v = (y[i1] - y[i0]) / (t[i1] - t[i0])
    return float(v), (float(t[i0]), float(y[i0])), (float(t[i1]), float(y[i1]))


def analyze(tag, Dr, Ha0, Yfs0, tend):
    case = NetworkCase(Dr=Dr, air_head=Ha0, init_water_level=Yfs0, t_end=tend)
    rec = run_network(case, verbose=False)
    t = np.asarray(rec["t"])
    wtop = np.asarray(rec["wtop"])      # Y_fs (free surface)
    itop = np.asarray(rec["itop"])      # Y_int (gas nose)
    cm = np.asarray(rec["core_mass"])   # resolved gas mass in riser [kg]
    ph = np.asarray(rec["pocket_head"]) # trapped-air gauge head [m]
    L = case.riser_height
    sgd = math.sqrt(G * Dr)

    rise = float(wtop.max() - Yfs0)
    rise_frac = rise / L

    # free-surface mean velocity while it climbs from just above start toward the top
    v_fs, fs0, fs1 = _mean_velocity(t, wtop, Yfs0 + 0.01, min(Yfs0 + 0.9 * (L - Yfs0), L - 0.005))
    # gas-nose mean velocity from near the base to near the top
    v_int, it0, it1 = _mean_velocity(t, itop, 0.05 * L, 0.85 * L)

    paper = PAPER.get(round(Dr, 4), {})
    out = {
        "tag": tag, "Dr": Dr, "Dt_over_D": Dr / case.D, "Ha0": Ha0, "Yfs0": Yfs0,
        "L": L, "sqrt_gDt": sgd, "tend": tend,
        "rise_m": rise, "rise_frac_model": rise_frac,
        "rise_frac_paper": paper.get("rise_frac"),
        "Yfs_max": float(wtop.max()), "Yint_max": float(itop.max()),
        "gas_mass_max_mg": float(cm.max() * 1e6),
        "pocket_head0": float(ph[0]) if ph.size else None,
        "pocket_head_end": float(ph[-1]) if ph.size else None,
        "V_fs_model": v_fs / sgd, "V_fs_paper": paper.get("V_fs"),
        "V_int_model": v_int / sgd, "V_int_paper": paper.get("V_int"),
        "fs_window": [fs0, fs1], "int_window": [it0, it1],
        "t": t.tolist(), "Yfs": wtop.tolist(), "Yint": itop.tolist(),
        "gas_mass_mg": (cm * 1e6).tolist(), "pocket_head": ph.tolist(),
    }
    print(f"\n=== {tag}  Dt/D={out['Dt_over_D']:.3f}  Ha0={Ha0} Yfs0={Yfs0} ===")
    print(f"  rise/L : model {rise_frac:.3f}   paper {paper.get('rise_frac')}")
    print(f"  V_fs*  : model {out['V_fs_model']:.3f}   paper {paper.get('V_fs')}")
    print(f"  V_int* : model {out['V_int_model']:.3f}   paper {paper.get('V_int')}")
    print(f"  Yfs_max={out['Yfs_max']:.3f}  Yint_max={out['Yint_max']:.3f}  L={L}")
    print(f"  gas_mass_max={out['gas_mass_max_mg']:.3f} mg  head0={out['pocket_head0']} head_end={out['pocket_head_end']}")
    # coarse trajectory print
    idx = np.linspace(0, len(t) - 1, 13).astype(int)
    print("  t / Yfs / Yint / gas[mg] :")
    for k in idx:
        print(f"    t={t[k]:5.2f}  Yfs={wtop[k]:.3f}  Yint={itop[k]:.3f}  gas={cm[k]*1e6:7.3f}")
    return out


if __name__ == "__main__":
    results = []
    results.append(analyze("CaseA_Dt57p1_Ha0305_Yfs0356", 0.0571, 0.305, 0.356, tend=10.0))
    results.append(analyze("CaseB_Dt12p7_Ha0610_Yfs0356", 0.0127, 0.610, 0.356, tend=10.0))
    with open("diag_two_cases_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved -> diag_two_cases_results.json")
