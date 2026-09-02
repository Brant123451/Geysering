#!/usr/bin/env python3
"""Summarize an exploratory B-H3 branch without promoting it as evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CASE_ROOT))
import postprocess as pp  # noqa: E402


def finite_max(values: np.ndarray) -> float | None:
    return float(np.nanmax(values)) if np.isfinite(values).any() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    t, yfs, yint = pp.read_riser_series(args.run_dir)
    start = float(manifest["branch_time_s"])
    actual_end = float(np.max(t)) if t.size else start
    t_rim = pp.first_passage(t, yfs, 0.98 * pp.RIM_HEIGHT, start)

    baseline = np.genfromtxt(args.baseline, delimiter=",", names=True)
    window = (baseline["t_s"] >= start) & (baseline["t_s"] <= actual_end + 1e-9)
    baseline_yfs = finite_max(baseline["Yfs_m_above_crown"][window])
    baseline_yint = finite_max(baseline["Yint_m_above_crown"][window])
    yfs_max = finite_max(yfs)
    yint_max = finite_max(yint)

    status = pp.log_status(args.run_dir)
    payload = {
        "schema_version": 1,
        "case": "BH3_Dr26_H066_L061",
        "evidence_status": "exploratory_not_manuscript_evidence",
        "candidate": manifest,
        "status": status,
        "n_riser_samples": int(t.size),
        "actual_end_time_s": actual_end,
        "model": {
            "Yfs_max_m_above_crown": yfs_max,
            "Yint_max_m_above_crown": yint_max,
            "t_free_surface_at_98pct_rim_s": t_rim,
            "geysering": t_rim is not None,
        },
        "same_window_baseline": {
            "Yfs_max_m_above_crown": baseline_yfs,
            "Yint_max_m_above_crown": baseline_yint,
        },
        "delta_from_baseline": {
            "Yfs_max_m": None if yfs_max is None or baseline_yfs is None else yfs_max - baseline_yfs,
            "Yint_max_m": None if yint_max is None or baseline_yint is None else yint_max - baseline_yint,
        },
        "selection_rule": "Prefer stable normal completion and larger coherent post-arrival Yint/Yfs; rerun selected scheme from t=0.",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
