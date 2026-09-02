#!/usr/bin/env python3
"""Extract B-H6 2D event metrics without fitting to the experiment."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CROWN_Z = 0.025
RIM_HEIGHT = 1.8
EXP = {
    "classification": "NO GEYSER",
    "Ta_s": 8.10,
    "vfs_m_s": 0.246,
    "vint_m_s": 0.476,
}


def numeric(name: str) -> bool:
    return bool(re.fullmatch(r"[0-9.eE+-]+", name))


def sample_roots(run: Path) -> list[Path]:
    roots = [run / "postProcessing" / "riserCentreline"]
    roots.extend(run.glob("processor*/postProcessing/riserCentreline"))
    return [p for p in roots if p.exists()]


def read_riser_series(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[np.ndarray]] = defaultdict(list)
    for root in sample_roots(run):
        for time_dir in root.iterdir():
            if not time_dir.is_dir() or not numeric(time_dir.name):
                continue
            candidates = sorted(time_dir.glob("*alpha.water*"))
            for path in candidates:
                try:
                    data = np.loadtxt(path, ndmin=2)
                except (OSError, ValueError):
                    continue
                if data.ndim == 2 and data.shape[1] >= 2:
                    grouped[float(time_dir.name)].append(data)
                    break

    times: list[float] = []
    yfs: list[float] = []
    yint: list[float] = []
    for time in sorted(grouped):
        data = np.concatenate(grouped[time], axis=0)
        order = np.argsort(data[:, 0])
        distance = data[order, 0]
        alpha = np.clip(data[order, 1], 0.0, 1.0)
        keep = np.r_[True, np.diff(distance) > 1e-10]
        distance = distance[keep]
        alpha = alpha[keep]
        z = CROWN_Z + 0.001 + distance

        water_to_air: list[float] = []
        air_to_water: list[float] = []
        for i in range(len(alpha) - 1):
            a0, a1 = alpha[i], alpha[i + 1]
            if (a0 - 0.5) * (a1 - 0.5) > 0 or abs(a1 - a0) < 1e-12:
                continue
            frac = (0.5 - a0) / (a1 - a0)
            cross = z[i] + frac * (z[i + 1] - z[i])
            if a0 >= 0.5 > a1:
                water_to_air.append(cross - CROWN_Z)
            elif a0 <= 0.5 < a1:
                air_to_water.append(cross - CROWN_Z)

        times.append(time)
        yfs.append(max(water_to_air) if water_to_air else np.nan)
        # The released pocket can break into several gas regions on the sampled
        # centreline.  The experimental Yint is the leading (uppermost) gas nose,
        # not the lowest gas-to-water crossing left behind by a liquid slug.
        # Since the atmospheric free surface is a water-to-air crossing, selecting
        # the uppermost air-to-water crossing remains below Yfs and tracks the
        # released pocket rather than the atmosphere above the column.
        yint.append(max(air_to_water) if air_to_water else np.nan)
    return np.asarray(times), np.asarray(yfs), np.asarray(yint)


def first_passage(t: np.ndarray, y: np.ndarray, level: float, start: float) -> float | None:
    valid = np.flatnonzero((t >= start) & np.isfinite(y) & (y >= level))
    if valid.size == 0:
        return None
    i = int(valid[0])
    if i == 0 or not np.isfinite(y[i - 1]) or y[i] == y[i - 1]:
        return float(t[i])
    fraction = (level - y[i - 1]) / (y[i] - y[i - 1])
    return float(t[i - 1] + fraction * (t[i] - t[i - 1]))


def interval_speed(t: np.ndarray, y: np.ndarray, low: float, high: float, start: float) -> float | None:
    t0 = first_passage(t, y, low, start)
    t1 = first_passage(t, y, high, start)
    if t0 is None or t1 is None or t1 <= t0:
        return None
    return (high - low) / (t1 - t0)


def read_pt1(run: Path) -> tuple[np.ndarray, np.ndarray]:
    paths = [run / "postProcessing" / "pressureProbes"]
    paths.extend(run.glob("processor*/postProcessing/pressureProbes"))
    rows: list[np.ndarray] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.glob("*/p"):
            try:
                data = np.loadtxt(path, ndmin=2)
            except (OSError, ValueError):
                continue
            if data.shape[1] >= 2:
                rows.append(data[:, :2])
    if not rows:
        return np.array([]), np.array([])
    data = np.concatenate(rows, axis=0)
    order = np.argsort(data[:, 0])
    data = data[order]
    keep = np.r_[True, np.diff(data[:, 0]) > 1e-12]
    return data[keep, 0], data[keep, 1]


def log_status(run: Path) -> dict[str, object]:
    logs = [
        run / "log.solve",
        run / "log.solve.resume",
        run / "log.solve.complete20",
        run / "log.smoke",
    ]
    text = "\n".join(path.read_text(errors="ignore") for path in logs if path.exists())
    times = [float(x) for x in re.findall(r"^Time = ([0-9.eE+-]+)", text, re.M)]
    return {
        "last_log_time_s": times[-1] if times else None,
        "ended_normally": bool(re.search(r"^End\s*$", text, re.M)),
        "fatal_error": "FOAM FATAL" in text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    t, yfs, yint = read_riser_series(args.run_dir)
    arrival_idx = np.flatnonzero((t >= 0.1) & np.isfinite(yint) & (yint >= 0.005))
    ta = float(t[arrival_idx[0]]) if arrival_idx.size else None
    start = ta if ta is not None else 0.0
    vfs = interval_speed(t, yfs, 0.65, 1.70, start)
    vint = interval_speed(t, yint, 0.05, 1.65, start)
    t_rim = first_passage(t, yfs, 0.98 * RIM_HEIGHT, start)
    max_yfs = float(np.nanmax(yfs)) if np.isfinite(yfs).any() else None
    max_yint = float(np.nanmax(yint)) if np.isfinite(yint).any() else None

    tp, pp = read_pt1(args.run_dir)
    head = (pp - 101325.0) / (998.0 * 9.81) if pp.size else np.array([])
    max_head = float(np.nanmax(head)) if head.size else None

    model = {
        "Ta_s": ta,
        "vfs_m_s": vfs,
        "vint_m_s": vint,
        "Yfs_max_m_above_crown": max_yfs,
        "Yint_max_m_above_crown": max_yint,
        "t_free_surface_at_98pct_rim_s": t_rim,
        "PT1_max_head_m_water": max_head,
        "geysering": t_rim is not None,
    }
    status = log_status(args.run_dir)
    payload = {
        "schema_version": 1,
        "case": "BH6_Dr41_H066_L061",
        "run_id": "paper_tau0p2_areaeq",
        "dimension": "2D_planar_area_equivalent",
        "solver": "bh6CompressibleInterFoam-v2512",
        "paper_contract": json.loads((HERE / "case_config.json").read_text()),
        "status": status,
        "n_riser_samples": int(t.size),
        "experiment": EXP,
        "model": model,
        "errors": {
            "Ta_s": None if ta is None else ta - EXP["Ta_s"],
            "vfs_m_s": None if vfs is None else vfs - EXP["vfs_m_s"],
            "vint_m_s": None if vint is None else vint - EXP["vint_m_s"],
            "classification_match": model["geysering"] is False,
        },
        "metric_definition": {
            "Ta": "first resolved gas-nose crossing at 5 mm above pipe crown",
            "vfs": "first-passage speed from 0.65 to 1.70 m above pipe crown",
            "vint": "first-passage speed of the uppermost enclosed gas nose from 0.05 to 1.65 m above pipe crown",
            "Yint": "uppermost air-to-water crossing below the continuous free surface on the riser centreline",
            "geyser": "free surface reaches 98% of the 1.8 m physical rim height",
        },
    }
    (output / "openfoam_2d_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (output / "openfoam_2d_riser_series.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "Yfs_m_above_crown", "Yint_m_above_crown"])
        writer.writerows(zip(t, yfs, yint))
    if tp.size:
        with (output / "openfoam_2d_pt1_series.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", "p_abs_Pa", "head_m_water"])
            writer.writerows(zip(tp, pp, head))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
        axes[0].plot(t, yfs, label="2D free surface", lw=1.7)
        axes[0].plot(t, yint, label="2D gas nose", lw=1.7)
        axes[0].axhline(RIM_HEIGHT, color="0.25", ls="--", label="physical rim")
        axes[0].set_ylabel("Height above pipe crown (m)")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)
        if tp.size:
            axes[1].plot(tp, head, color="#7a3db8", lw=1.4)
        axes[1].set_xlabel("Time after valve opening (s)")
        axes[1].set_ylabel("PT1 gauge head (m water)")
        axes[1].grid(alpha=0.25)
        fig.suptitle("Cong 2017 B-H6 - OpenFOAM 2D paper-layout run")
        fig.tight_layout()
        fig.savefig(output / "openfoam_2d_diagnostics.png", dpi=180)
        plt.close(fig)
    except Exception as exc:  # plotting is optional; metrics remain authoritative
        (output / "plot_error.log").write_text(f"{type(exc).__name__}: {exc}\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
