#!/usr/bin/env python3
"""Reduce ten old-2D mouth probes to a generic one-way coupling CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
MOUTH_WIDTH_M = 0.0017159574


def probe_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coords: dict[int, tuple[float, float, float]] = {}
    pattern = re.compile(
        r"^#\s*Probe\s+(\d+)\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)"
    )
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                coords[int(match.group(1))] = tuple(float(match.group(i)) for i in range(2, 5))
            elif coords and not line.startswith("#"):
                break
    if not coords:
        raise ValueError(f"No probe headers found in {path}")
    return [coords[i] for i in sorted(coords)]


def read_scalar_probes(path: Path) -> dict[float, list[float]]:
    result: dict[float, list[float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            result[round(values[0], 9)] = values[1:]
    return result


def read_vector_y_probes(path: Path) -> dict[float, list[float]]:
    result: dict[float, list[float]] = {}
    vector_pattern = re.compile(r"\(([^()]*)\)")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            time_value = round(float(line.split()[0]), 9)
            result[time_value] = [float(v.split()[1]) for v in vector_pattern.findall(line)]
    return result


def rolling_median(values: list[float], half_window: int) -> list[float]:
    if half_window <= 0:
        return values[:]
    return [
        float(statistics.median(values[max(0, i - half_window) : min(len(values), i + half_window + 1)]))
        for i in range(len(values))
    ]


def rolling_mean(values: list[float], half_window: int) -> list[float]:
    if half_window <= 0:
        return values[:]
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    output: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half_window)
        hi = min(len(values), i + half_window + 1)
        output.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-dir",
        type=Path,
        required=True,
        help="Directory containing the ten-probe alpha.water and U files",
    )
    parser.add_argument("--output", type=Path, default=CASE_DIR / "data" / "mouth_driver.csv")
    parser.add_argument("--source-start", type=float, default=6.5)
    parser.add_argument("--source-end", type=float, default=8.95)
    parser.add_argument("--smooth-seconds", type=float, default=0.025)
    parser.add_argument("--velocity-cap", type=float, default=6.0)
    parser.add_argument("--alpha-zero", type=float, default=1.0e-5)
    args = parser.parse_args()

    probe_dir = args.probe_dir.resolve()
    alpha_path = probe_dir / "alpha.water"
    velocity_path = probe_dir / "U"
    coords = probe_coordinates(alpha_path)
    alpha_map = read_scalar_probes(alpha_path)
    uy_map = read_vector_y_probes(velocity_path)
    times = sorted(set(alpha_map) & set(uy_map))
    times = [t for t in times if args.source_start <= t <= args.source_end]
    if len(times) < 3:
        raise ValueError("Fewer than three aligned source samples were found")
    if any(len(alpha_map[t]) != len(coords) or len(uy_map[t]) != len(coords) for t in times):
        raise ValueError("Probe-column count does not match the probe header")

    dt_values = [b - a for a, b in zip(times, times[1:]) if b > a]
    median_dt = statistics.median(dt_values)
    half_window = max(0, int(round(0.5 * args.smooth_seconds / median_dt)))

    alpha_raw: list[float] = []
    flux_raw: list[float] = []
    for t in times:
        alpha_cells = [min(1.0, max(0.0, a)) for a in alpha_map[t]]
        uy_cells = [max(-args.velocity_cap, min(args.velocity_cap, u)) for u in uy_map[t]]
        alpha_mean = statistics.fmean(alpha_cells)
        alpha_uy_mean = statistics.fmean(a * u for a, u in zip(alpha_cells, uy_cells))
        alpha_raw.append(alpha_mean)
        flux_raw.append(MOUTH_WIDTH_M * alpha_uy_mean)

    alpha = rolling_mean(rolling_median(alpha_raw, 1), half_window)
    line_flux = rolling_mean(rolling_median(flux_raw, 1), half_window)
    alpha = [0.0 if a < args.alpha_zero else min(1.0, max(0.0, a)) for a in alpha]
    uy_weighted = [
        0.0 if a == 0.0 else max(-args.velocity_cap, min(args.velocity_cap, q / (MOUTH_WIDTH_M * a)))
        for a, q in zip(alpha, line_flux)
    ]
    # Recompute flux after clipping so the CSV is internally self-consistent.
    line_flux = [MOUTH_WIDTH_M * a * u for a, u in zip(alpha, uy_weighted)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "local_time_s",
                "source_time_s",
                "alpha_mean",
                "Uy_alpha_weighted",
                "water_line_flux_m2_s",
            ]
        )
        for t, a, velocity, flux in zip(times, alpha, uy_weighted, line_flux):
            writer.writerow(
                [
                    f"{t - times[0]:.9f}",
                    f"{t:.9f}",
                    f"{a:.9g}",
                    f"{velocity:.9g}",
                    f"{flux:.9g}",
                ]
            )

    manifest = {
        "source_probe_directory": str(probe_dir),
        "source_probe_coordinates_m": coords,
        "source_time_range_s": [times[0], times[-1]],
        "local_time_range_s": [0.0, times[-1] - times[0]],
        "mouth_width_m": MOUTH_WIDTH_M,
        "source_median_dt_s": median_dt,
        "smoothing_window_s": args.smooth_seconds,
        "velocity_cap_m_per_s": args.velocity_cap,
        "coupling": "one-way section-mean alpha and liquid-alpha-weighted vertical velocity",
        "evidence_status": "exploratory/supporting",
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(times)} rows, {len(coords)} probes)")


if __name__ == "__main__":
    main()

