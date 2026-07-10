# -*- coding: utf-8 -*-
"""Short Case B run to trace air-pocket conservation: gas mass, gas volume,
pocket head, junction pressure head, total liquid. Reveals why the pocket
overpressure dissipates before reaching the riser."""
from __future__ import annotations
import numpy as np
from vw2011_network_twofluid import NetworkCase, run_network

case = NetworkCase(Dr=0.0127, air_head=0.610, init_water_level=0.356, t_end=3.0, ds=0.04, dz=0.02)
rec = run_network(case, verbose=False)
t = np.asarray(rec["t"])
gm = np.asarray(rec["tun_gas_mass"]) * 1e3      # g
gv = np.asarray(rec["tun_gas_vol"]) * 1e3       # L
ph = np.asarray(rec["pocket_head"])
pj = np.asarray(rec["pj_head"])
tl = np.asarray(rec["tot_liq"]) * 1e3           # L
n = min(len(t), len(gm), len(pj))
print("  t    gas_mass[g]  gas_vol[L]  pocket_head[m]  Pj_head[m]  tot_liq[L]")
for k in range(0, n, max(1, n // 20)):
    print(f"  {t[k]:4.2f}   {gm[k]:8.4f}   {gv[k]:7.4f}    {ph[k]:7.3f}      {pj[k]:7.3f}    {tl[k]:8.4f}")
print(f"\ngas mass: t0={gm[0]:.4f} g  ->  t_end={gm[n-1]:.4f} g  (conserved if ~equal)")
print(f"gas vol : t0={gv[0]:.4f} L  ->  t_end={gv[n-1]:.4f} L")
print(f"tot liq : t0={tl[0]:.4f} L  ->  t_end={tl[n-1]:.4f} L  (should be ~constant)")
