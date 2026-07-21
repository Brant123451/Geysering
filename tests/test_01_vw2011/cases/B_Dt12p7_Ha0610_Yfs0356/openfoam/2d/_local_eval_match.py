#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

m = json.loads((Path(__file__).parent / "outputs/openfoam_2d_metrics.json").read_text())
yfs = float(m.get("free_surface_max_Ystar", 0) or 0)
hplateau = float(m.get("pressure_plateau_Hstar_mean_T1to7", -1) or -1)
hrmse = float(m.get("pressure_RMSE_Hstar_no_shift", 9) or 9)
tstar = float(m.get("simulation_end_Tstar", 0) or 0)
geyser = bool(m.get("geysering", False))
checks = {
    "ran_to_Tstar_ge_5.0": tstar >= 5.0,
    "geysering_Yfs_ge_1": geyser and yfs >= 1.0,
    "Hstar_plateau_in_0.35_0.95": (not (hplateau == hplateau)) or (0.35 <= hplateau <= 0.95),
    "Hstar_RMSE_lt_0.40": hrmse < 0.40,
}
ok = all(checks.values())
out = {
    "match_ok": ok,
    "checks": checks,
    "yfs_max": yfs,
    "Hstar_plateau": hplateau,
    "Hstar_RMSE": hrmse,
    "Tstar_end": tstar,
    "geysering": geyser,
    "note": "Area-equivalent tower W=Dt^2/D, sigma=0; require Yfs*>=1 (geysering).",
}
print(json.dumps(out, indent=2))
Path("outputs").mkdir(exist_ok=True)
Path("outputs/match_verdict.json").write_text(json.dumps(out, indent=2) + "\n")
raise SystemExit(0 if ok else 1)
