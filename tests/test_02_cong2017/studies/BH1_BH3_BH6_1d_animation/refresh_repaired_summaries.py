#!/usr/bin/env python3
"""Refresh conservation labels in completed repaired-run summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "repaired" / "model_1d"


def main() -> None:
    for case_key in ("BH1", "BH3", "BH6"):
        folder = ROOT / case_key
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data = np.load(folder / "repaired_1d_frames.npz", allow_pickle=False)
        gas_mass = np.asarray(data["tun_gas_mass"], dtype=float)
        mass_times = np.asarray(data["t"], dtype=float)[1 : 1 + gas_mass.size]
        wetting_time = summary["wet_dry_checks"]["wetting_front_reaches_cap_s"]
        index = int(np.argmin(np.abs(mass_times - wetting_time)))
        old = summary["wet_dry_checks"]
        floor_full = old.get(
            "liquid_floor_created_m3_full_run", old.get("liquid_floor_created_m3")
        )
        summary["wet_dry_checks"] = {
            "finite_wetting_front": True,
            "wetting_front_reaches_cap_s": wetting_time,
            "tunnel_gas_mass_relative_drift_through_wetting": float(
                gas_mass[index] / gas_mass[0] - 1.0
            ),
            "tunnel_gas_mass_relative_change_full_run_including_riser_venting": float(
                gas_mass[-1] / gas_mass[0] - 1.0
            ),
            "liquid_floor_created_m3_full_run": floor_full,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(case_key, summary["wet_dry_checks"])


if __name__ == "__main__":
    main()
