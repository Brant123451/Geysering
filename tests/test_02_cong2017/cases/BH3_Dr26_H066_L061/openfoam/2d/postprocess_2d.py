#!/usr/bin/env python3
"""Compact postprocess for BH3 2-D: Ta / free-surface / rim arrival vs paper."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PAPER = {
    "run": "B-H3",
    "Ta_s": 8.18,
    "vfs_m_per_s": 0.657,
    "vint_m_per_s": 0.916,
    "geyser": True,
    "H0_m": 0.66,
    "Dr_m": 0.026,
    "D_m": 0.05,
    "L0_m": 0.61,
}
Y_RIM = 1.85
X_TEE = 3.47


def latest_dat(root: Path, name: str, filename: str) -> Path | None:
    base = root / "postProcessing" / name
    if not base.exists():
        return None
    cands = sorted(base.glob(f"*/{filename}"), key=lambda p: float(p.parent.name))
    return cands[-1] if cands else None


def load_probes(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows = []
    header = None
    for ln in path.read_text(errors="ignore").splitlines():
        if not ln.strip():
            continue
        if ln.startswith("#"):
            if "Time" in ln and header is None:
                header = ln.lstrip("#").split()
            continue
        rows.append([float(x) for x in ln.split()])
    if not rows:
        return np.array([]), [], np.empty((0, 0))
    data = np.asarray(rows, dtype=float)
    return data[:, 0], header or [], data


def free_surface_from_riser(alpha_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Yfs(t) from riser centreline alpha probes (0.5 crossing)."""
    # Probe file layout from OpenFOAM probes: Time + one column per location
    # We stored alpha.water only in riserCentreline together with U,p — parse carefully.
    text = alpha_path.read_text(errors="ignore").splitlines()
    # Better: use alpha.water file if present
    return np.array([]), np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    case = args.case
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # riser alpha probes
    alpha = latest_dat(case, "riserCentreline", "alpha.water")
    p_probe = latest_dat(case, "pressureProbes", "p")
    metrics: dict = {
        "schema_version": 1,
        "run_id": "base_nominal_2d",
        "geometry": "2d_vertical_plane_fine",
        "paper": PAPER,
        "full_13s_window_completed": False,
    }

    yfs_t = []
    yfs_y = []
    if alpha and alpha.exists():
        # Read probe locations from header comments
        ys = []
        for ln in alpha.read_text(errors="ignore").splitlines():
            if ln.startswith("# Probe") and "(" in ln:
                # # Probe 0 (3.47 0.05 0)
                try:
                    xyz = ln.split("(")[1].split(")")[0].split()
                    ys.append(float(xyz[1]))
                except Exception:
                    pass
            if ln.startswith("#") and "Time" in ln:
                break
        ys = np.asarray(ys, dtype=float)
        times = []
        yfs = []
        for ln in alpha.read_text(errors="ignore").splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            vals = [float(x) for x in ln.split()]
            t = vals[0]
            a = np.asarray(vals[1 : 1 + len(ys)])
            # 0.5 isosurface from bottom
            y_fs = float(ys[0])
            for i in range(len(ys) - 1):
                if (a[i] - 0.5) * (a[i + 1] - 0.5) <= 0:
                    frac = (0.5 - a[i]) / (a[i + 1] - a[i] + 1e-30)
                    y_fs = float(ys[i] + frac * (ys[i + 1] - ys[i]))
                    break
            else:
                if a[-1] >= 0.5:
                    y_fs = float(ys[-1])
                elif a[0] < 0.5:
                    y_fs = float(ys[0])
            times.append(t)
            yfs.append(y_fs)
        yfs_t = np.asarray(times)
        yfs_y = np.asarray(yfs)
        metrics["simulated_end_time_s"] = float(yfs_t[-1]) if len(yfs_t) else 0.0
        metrics["full_13s_window_completed"] = bool(len(yfs_t) and yfs_t[-1] >= 12.99)
        metrics["Yfs_max_m"] = float(np.max(yfs_y)) if len(yfs_y) else None
        # Ta: first time Yfs reaches rim
        hit = np.where(yfs_y >= Y_RIM)[0]
        metrics["geysering"] = bool(len(hit))
        metrics["Ta_2d_s"] = float(yfs_t[hit[0]]) if len(hit) else None
        # vfs rough: max dYfs/dt near rise
        if len(yfs_t) > 5:
            dy = np.gradient(yfs_y, yfs_t)
            metrics["vfs_2d_m_per_s"] = float(np.max(dy))
        # write timeseries
        ts_path = out / "base_nominal_2d_timeseries.csv"
        with ts_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "Yfs_m"])
            for t, y in zip(yfs_t, yfs_y):
                w.writerow([f"{t:.8g}", f"{y:.8g}"])

    if p_probe and p_probe.exists():
        t, _, data = load_probes(p_probe)
        if len(t):
            metrics["pressure_probe_end_s"] = float(t[-1])
            metrics["pt_upstream_final_pa"] = float(data[-1, 1]) if data.shape[1] > 1 else None

    metrics["comparison"] = {
        "paper_Ta_s": PAPER["Ta_s"],
        "sim_Ta_s": metrics.get("Ta_2d_s"),
        "Ta_rel_error": (
            None
            if metrics.get("Ta_2d_s") is None
            else (metrics["Ta_2d_s"] - PAPER["Ta_s"]) / PAPER["Ta_s"]
        ),
        "paper_geyser": True,
        "sim_geyser": metrics.get("geysering"),
    }
    metrics["honesty_note"] = (
        "2-D vertical-plane diagnostic preserves paper lengths/diameters but not "
        "circular area ratio (Dr/D vs (Dr/D)^2). Instant valve opening (paper CFD baseline)."
    )

    (out / "base_nominal_2d_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
