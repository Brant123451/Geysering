#!/usr/bin/env python3
"""Compact postprocess for B-H2 2D planar OpenFOAM run vs paper Table-2 scalars."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
EXP = {
    "classification": "GEYSER",
    "Ta_s": 7.84,
    "vfs_m_s": 0.768,
    "vint_m_s": 1.022,
    "Yrim_m": 1.825,
}


def latest_run(run_id: str) -> Path:
    return CASE / "runs" / run_id


def read_riser_series(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return t, Yfs, Yint from sets/riserLine centreline alpha.water=0.5 crossings."""
    sets_root = run / "postProcessing" / "riserLine"
    if not sets_root.exists():
        # parallel reconstructed layout
        sets_root = run / "postProcessing"
    times = []
    yfs = []
    yint = []
    for td in sorted(sets_root.glob("*"), key=lambda p: float(p.name) if re.fullmatch(r"[0-9.eE+-]+", p.name) else -1):
        if not re.fullmatch(r"[0-9.eE+-]+", td.name):
            continue
        # find alpha file
        files = list(td.rglob("alpha.water*")) + list(td.rglob("*centreline*"))
        raw = None
        for f in td.iterdir() if td.is_dir() else []:
            if "alpha" in f.name or "centreline" in f.name:
                raw = f
                break
        # OpenFOAM sets raw: postProcessing/riserLine/<time>/centreline_alpha.water.xy
        cands = list(td.glob("*alpha.water*"))
        if not cands:
            continue
        data = np.loadtxt(cands[0])
        if data.ndim != 2 or data.shape[1] < 2:
            continue
        # columns: distance, alpha (or x y z alpha)
        if data.shape[1] >= 4:
            z = data[:, 2]
            a = data[:, -1]
        else:
            # distance along line from z=0.026
            z = 0.026 + data[:, 0]
            a = data[:, 1]
        # free surface: highest water->air crossing
        y_fs = None
        y_in = None
        for i in range(len(a) - 1):
            if a[i] >= 0.5 > a[i + 1]:
                # water to air
                frac = (a[i] - 0.5) / (a[i] - a[i + 1] + 1e-30)
                y_fs = z[i] + frac * (z[i + 1] - z[i])
            if a[i] <= 0.5 < a[i + 1]:
                frac = (0.5 - a[i]) / (a[i + 1] - a[i] + 1e-30)
                y_in = z[i] + frac * (z[i + 1] - z[i])
        times.append(float(td.name))
        yfs.append(y_fs if y_fs is not None else np.nan)
        yint.append(y_in if y_in is not None else np.nan)
    return np.asarray(times), np.asarray(yfs, float), np.asarray(yint, float)


def read_probe_fallback(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: plume probe alpha at rim as crude Yfs proxy."""
    pdir = run / "postProcessing" / "plumeProbes"
    if not pdir.exists():
        return np.array([]), np.array([])
    # OpenFOAM probes: postProcessing/plumeProbes/0/alpha.water
    files = sorted(pdir.glob("*/alpha.water"))
    if not files:
        return np.array([]), np.array([])
    # concatenate
    t_all = []
    a_rim = []
    for f in files:
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        if arr.shape[1] < 2:
            continue
        t_all.append(arr[:, 0])
        a_rim.append(arr[:, 1])  # first probe at rim
    if not t_all:
        return np.array([]), np.array([])
    t = np.concatenate(t_all)
    a = np.concatenate(a_rim)
    order = np.argsort(t)
    return t[order], a[order]


def metrics_from_series(t, yfs, yint) -> dict:
    out = {
        "Ta_s": None,
        "Yfs_max_m": None,
        "Yint_max_m": None,
        "vfs_m_s": None,
        "vint_m_s": None,
        "geyser_model": False,
        "t_rim_s": None,
    }
    if t.size == 0:
        return out
    finite = np.isfinite(yfs)
    if not finite.any():
        return out
    out["Yfs_max_m"] = float(np.nanmax(yfs))
    if np.isfinite(yint).any():
        out["Yint_max_m"] = float(np.nanmax(yint))
    # Ta: first time Yfs rises above initial ~0.635 by significant climb start
    # Use first time Yfs exceeds 0.70 as arrival proxy, refine with max climb window
    y0 = 0.635
    above = np.where(finite & (yfs > y0 + 0.02))[0]
    if above.size:
        out["Ta_s"] = float(t[above[0]])
    rim = np.where(finite & (yfs >= EXP["Yrim_m"] - 1e-3))[0]
    if rim.size:
        out["t_rim_s"] = float(t[rim[0]])
        out["geyser_model"] = True
    # velocities: max rolling climb over 0.6 s
    if out["Ta_s"] is not None:
        t0 = out["Ta_s"]
        mask = (t >= t0) & (t <= t0 + 3.0) & finite
        tt = t[mask]
        yy = yfs[mask]
        if tt.size > 5:
            window = 0.6
            best = 0.0
            for i, ti in enumerate(tt):
                j = np.searchsorted(tt, ti + window)
                if j <= i:
                    continue
                best = max(best, (yy[j - 1] - yy[i]) / max(tt[j - 1] - tt[i], 1e-9))
            out["vfs_m_s"] = float(best)
        if np.isfinite(yint).any():
            maski = (t >= t0) & (t <= t0 + 3.0) & np.isfinite(yint)
            tt = t[maski]
            yy = yint[maski]
            if tt.size > 5:
                best = 0.0
                for i, ti in enumerate(tt):
                    j = np.searchsorted(tt, ti + 0.6)
                    if j <= i:
                        continue
                    best = max(best, (yy[j - 1] - yy[i]) / max(tt[j - 1] - tt[i], 1e-9))
                out["vint_m_s"] = float(best)
    # also geyser if ejected / Yfs above rim significantly
    if out["Yfs_max_m"] is not None and out["Yfs_max_m"] >= EXP["Yrim_m"]:
        out["geyser_model"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fine_baseline")
    args = ap.parse_args()
    run = latest_run(args.run)
    results = CASE / "results"
    results.mkdir(exist_ok=True)

    t, yfs, yint = read_riser_series(run)
    source = "riserLine"
    if t.size == 0:
        # try processor0 postProcessing
        for p0 in sorted(run.glob("processor*/postProcessing/riserLine")):
            t, yfs, yint = read_riser_series(p0.parent.parent)
            if t.size:
                source = str(p0)
                break
    m = metrics_from_series(t, yfs, yint)
    # log progress fallback
    log = run / "log.solve"
    ended = False
    t_log = None
    if log.exists():
        text = log.read_text(errors="ignore")
        ended = bool(re.search(r"^End\b", text, re.M))
        times = [float(x) for x in re.findall(r"^Time = ([0-9.eE+-]+)", text, re.M)]
        t_log = times[-1] if times else None

    payload = {
        "run_id": args.run,
        "dim": "2D_planar_xz",
        "solver": "compressibleInterIsoFoam",
        "series_source": source,
        "n_samples": int(t.size),
        "log_time_s": t_log,
        "ended": ended,
        "experiment": EXP,
        "model": m,
        "deltas": {
            "Ta_delta_s": None if m["Ta_s"] is None else m["Ta_s"] - EXP["Ta_s"],
            "vfs_delta_m_s": None if m["vfs_m_s"] is None else m["vfs_m_s"] - EXP["vfs_m_s"],
            "vint_delta_m_s": None if m["vint_m_s"] is None else m["vint_m_s"] - EXP["vint_m_s"],
            "geyser_match": m["geyser_model"] is True,
        },
    }
    out_json = results / f"openfoam_2d_{args.run}_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    # series csv
    out_csv = results / f"openfoam_2d_{args.run}_series.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Yfs_m", "Yint_m"])
        for ti, a, b in zip(t, yfs, yint):
            w.writerow([ti, a, b])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
