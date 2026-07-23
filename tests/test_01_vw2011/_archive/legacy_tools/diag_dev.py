# -*- coding: utf-8 -*-
"""Fast (coarse-grid) dev diagnostic for iterating the riser two-phase physics.
Runs both target cases and prints rise/L, V_fs*, V_int* against the paper.
Use coarse grid for speed during development; final validation uses full res.
"""
from __future__ import annotations
import math
import sys

import numpy as np

from vw2011_network_twofluid import NetworkCase, run_network, G

PAPER = {
    0.0571: {"Dt_over_D": 0.607, "V_fs": 0.048, "V_int": 0.39, "rise_frac": 0.10},
    0.0127: {"Dt_over_D": 0.135, "V_fs": 0.44, "V_int": 1.43, "rise_frac": 1.00},
}


def _mean_velocity(t, y, y_lo, y_hi):
    t = np.asarray(t); y = np.asarray(y)
    above = np.where(y >= y_lo)[0]
    if above.size == 0:
        return float("nan")
    i0 = int(above[0])
    reach = np.where(y >= y_hi)[0]
    i1 = int(reach[0]) if reach.size else int(np.argmax(y))
    if i1 <= i0 or t[i1] <= t[i0]:
        return 0.0
    return float((y[i1] - y[i0]) / (t[i1] - t[i0]))


def run(tag, Dr, Ha0, Yfs0, tend, ds, dz):
    case = NetworkCase(Dr=Dr, air_head=Ha0, init_water_level=Yfs0, t_end=tend, ds=ds, dz=dz)
    rec = run_network(case, verbose=False)
    t = np.asarray(rec["t"]); wtop = np.asarray(rec["wtop"]); itop = np.asarray(rec["itop"])
    cm = np.asarray(rec["core_mass"]); ph = np.asarray(rec["pocket_head"])
    pj = np.asarray(rec["pj_head"])
    L = case.riser_height; sgd = math.sqrt(G * Dr)
    rise = float(wtop.max() - Yfs0); rise_frac = rise / L
    v_fs = _mean_velocity(t, wtop, Yfs0 + 0.01, min(Yfs0 + 0.9 * (L - Yfs0), L - 0.005)) / sgd
    v_int = _mean_velocity(t, itop, 0.05 * L, 0.85 * L) / sgd
    p = PAPER[round(Dr, 4)]
    print(f"\n=== {tag} Dt/D={p['Dt_over_D']:.3f} (ds={ds},dz={dz},tend={tend}) ===")
    print(f"  rise/L : model {rise_frac:5.3f}  paper {p['rise_frac']}")
    print(f"  V_fs*  : model {v_fs:5.3f}  paper {p['V_fs']}")
    print(f"  V_int* : model {v_int:5.3f}  paper {p['V_int']}")
    print(f"  Yfs_max={wtop.max():.3f} Yint_max={itop.max():.3f} L={L} gas_max={cm.max()*1e6:.1f}mg")
    idx = np.linspace(0, len(t) - 1, 10).astype(int)
    for k in idx:
        pjk = pj[k] if k < len(pj) else float("nan")
        print(f"    t={t[k]:5.2f} Yfs={wtop[k]:.3f} Yint={itop[k]:.3f} gas={cm[k]*1e6:7.2f}mg pocket_head={ph[k]:.3f}m Pj_head={pjk:.3f}m")
    return rise_frac, v_fs, v_int


if __name__ == "__main__":
    ds = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    dz = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    tend = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
    run("CaseA_Dt57p1", 0.0571, 0.305, 0.356, tend, ds, dz)
    run("CaseB_Dt12p7", 0.0127, 0.610, 0.356, tend, ds, dz)
