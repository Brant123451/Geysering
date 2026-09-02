#!/usr/bin/env python3
"""Read-only live comparison for active qualification branches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CASE_ROOT))
import postprocess as pp  # noqa: E402


def maximum(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def main() -> None:
    baseline = np.genfromtxt(
        CASE_ROOT / "results" / "openfoam_2d_riser_series.csv",
        delimiter=",",
        names=True,
    )
    root = Path("/tmp/bh3-2d-qualification")
    for name in ("linearUpwind", "linearUpwind_cAlpha2"):
        t, yfs, yint = pp.read_riser_series(root / name)
        if not t.size:
            print(f"{name}: no samples")
            continue
        end = float(np.max(t))
        mask = (baseline["t_s"] >= 8.0) & (baseline["t_s"] <= end + 1e-9)
        print(
            f"{name}: t={end:.3f}, "
            f"Yfs={maximum(yfs):.4f} (base {maximum(baseline['Yfs_m_above_crown'][mask]):.4f}), "
            f"Yint={maximum(yint):.4f} (base {maximum(baseline['Yint_m_above_crown'][mask]):.4f})"
        )


if __name__ == "__main__":
    main()

