# -*- coding: utf-8 -*-
"""Diagnostic: run Case A and track liquid-volume conservation, pocket state,
tower liquid volume, and junction exchange to locate non-physical behaviour."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))

from vw2011_network_twofluid import G, NetworkCase, run_network

case = NetworkCase(Dr=0.0571, air_head=0.305, init_water_level=0.356, t_end=9.0)
rec = run_network(case, verbose=False)

t = np.asarray(rec["t"])
up = np.asarray(rec["up_head"])
wtop = np.asarray(rec["wtop"])
itop = np.asarray(rec["itop"])
tun_gas_vol = np.asarray(rec["tun_gas_vol"])
tot_liq = np.asarray(rec["tot_liq"])
tgm = np.asarray(rec["tun_gas_mass"])

n = min(len(t) - 1, len(tot_liq))
L = case.riser_height
sgd = math.sqrt(G * case.Dr)
print(f"{'t':>6} {'T*':>6} {'up_head':>8} {'wtop':>6} {'itop':>6} "
      f"{'tunGasV[L]':>10} {'totLiq[L]':>10} {'dLiq[mL]':>9} {'gasM[g]':>8}")
liq0 = tot_liq[0]
for i in range(0, n, max(1, n // 60)):
    print(f"{t[i+1]:6.2f} {t[i+1]*sgd/L:6.2f} {up[i+1]:8.3f} {wtop[i+1]:6.3f} {itop[i+1]:6.3f} "
          f"{tun_gas_vol[i]*1e3:10.3f} {tot_liq[i]*1e3:10.4f} {(tot_liq[i]-liq0)*1e6:9.1f} {tgm[i]*1e3:8.3f}")
