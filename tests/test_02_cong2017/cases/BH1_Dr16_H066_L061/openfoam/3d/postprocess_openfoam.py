#!/usr/bin/env python3
"""Reduce a BH1 OpenFOAM run to compact validation and conservation outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PATM = 101325.0
T0 = 296.15
R_AIR = 287.05
RHO_AIR_ATM = PATM / (R_AIR * T0)
RHO_WATER = 998.2
GRAVITY = 9.81
H0 = 0.66
RIM_HEIGHT = 1.80
POCKET_VOLUME_TARGET = math.pi * 0.050**2 * 0.610 / 4
POCKET_MASS_TARGET = POCKET_VOLUME_TARGET * RHO_AIR_ATM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "smoke", "full"), required=True)
    parser.add_argument("--valve-duration", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experimental-levels", type=Path, required=True)
    parser.add_argument("--experimental-pressure", type=Path, required=True)
    parser.add_argument("--one-d", type=Path, required=True)
    return parser.parse_args()


def newest_data_file(run_dir: Path, function_name: str, filename: str) -> Path | None:
    root = run_dir / "postProcessing" / function_name
    candidates = list(root.glob(f"*/{filename}"))
    if not candidates:
        return None

    def numeric_parent(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -math.inf

    return max(candidates, key=numeric_parent)


def parse_table(path: Path | None) -> tuple[list[str], np.ndarray]:
    if path is None or not path.exists():
        return [], np.empty((0, 0))
    header: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("# Time"):
            header = re.split(r"\s+", line[2:].strip())
        elif line and not line.startswith("#"):
            try:
                rows.append([float(value) for value in re.split(r"\s+", line)])
            except ValueError:
                continue
    if not rows:
        return header, np.empty((0, len(header)))
    width = min(len(row) for row in rows)
    return header[:width], np.asarray([row[:width] for row in rows], dtype=float)


def parse_scalar_probes(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.empty((0, 0))
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "(" in line:
            continue
        try:
            rows.append([float(value) for value in re.split(r"\s+", line)])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 0))
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=float)


def column(header: list[str], data: np.ndarray, needle: str) -> np.ndarray:
    for index, name in enumerate(header):
        if needle in name:
            return data[:, index]
    return np.full(data.shape[0], np.nan)


def interp(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_t) & np.isfinite(source_y)
    if np.count_nonzero(valid) < 2:
        return np.full(target_t.shape, np.nan)
    return np.interp(
        target_t,
        source_t[valid],
        source_y[valid],
        left=np.nan,
        right=np.nan,
    )


def first_crossing(t: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(np.isfinite(y) & (y >= threshold))
    return float(t[indices[0]]) if indices.size else None


def trapz_flux(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = np.zeros_like(t)
    valid = np.isfinite(q)
    if t.size < 2 or not np.any(valid):
        return np.full_like(t, np.nan)
    clean = np.where(valid, q, 0.0)
    result[1:] = np.cumsum(0.5 * (clean[1:] + clean[:-1]) * np.diff(t))
    return result


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
        if key != "kind"
    } | (
        {"kind": np.asarray([row["kind"] for row in rows])}
        if "kind" in rows[0]
        else {}
    )


def finite_relative(error: float, reference: float) -> float | None:
    if not math.isfinite(error) or reference == 0:
        return None
    return error / reference


def linear_speed(t: np.ndarray, y: np.ndarray, low: float, high: float) -> float | None:
    mask = np.isfinite(y) & (y >= low) & (y <= high)
    if np.count_nonzero(mask) < 3:
        return None
    return float(np.polyfit(t[mask], y[mask], 1)[0])


def profile_interfaces(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if profile.size == 0 or profile.shape[1] < 2:
        return np.array([]), np.array([]), np.array([])
    times = profile[:, 0]
    alpha = profile[:, 1:]
    heights = 0.005 + 0.010 * np.arange(alpha.shape[1])
    yfs = np.zeros(times.size)
    yint = np.zeros(times.size)
    for row_index, values in enumerate(alpha):
        liquid = np.flatnonzero(values >= 0.5)
        yfs[row_index] = heights[liquid[-1]] if liquid.size else 0.0

        # The gas core is counted only when it is connected to the first
        # centreline sample above the tee. Initial riser headspace is thus not
        # mistaken for the rising tunnel pocket.
        if values[0] < 0.5:
            core_end = 0
            while core_end + 1 < values.size and values[core_end + 1] < 0.5:
                core_end += 1
            yint[row_index] = heights[core_end]
    return times, yfs, yint


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_dir.name

    riser_profile = parse_scalar_probes(
        newest_data_file(args.run_dir, "riserCentreline", "alpha.water")
    )
    time, yfs, yint = profile_interfaces(riser_profile)
    if time.size == 0:
        raise SystemExit("No riserCentreline alpha.water data were written")

    pt_pressure = parse_scalar_probes(
        newest_data_file(args.run_dir, "pt1Pt2", "p")
    )
    pt1_head = (
        interp(pt_pressure[:, 0], (pt_pressure[:, 1] - PATM) / (RHO_WATER * GRAVITY), time)
        if pt_pressure.shape[1] >= 3
        else np.full(time.shape, np.nan)
    )
    pt2_head = (
        interp(pt_pressure[:, 0], (pt_pressure[:, 2] - PATM) / (RHO_WATER * GRAVITY), time)
        if pt_pressure.shape[1] >= 3
        else np.full(time.shape, np.nan)
    )

    function_specs = {
        "water_volume": ("waterVolume", "volFieldValue.dat", "alpha.water"),
        "gas_volume": ("gasVolume", "volFieldValue.dat", "alpha.air"),
        "water_mass": ("waterMass", "volFieldValue.dat", "alpha.water"),
        "gas_mass": ("gasMass", "volFieldValue.dat", "alpha.air"),
        "tunnel_gas_volume": (
            "tunnelGasVolume",
            "volFieldValue.dat",
            "alpha.air",
        ),
        "ejected_water": (
            "ejectedWaterInExterior",
            "volFieldValue.dat",
            "alpha.water",
        ),
        "riser_water": (
            "riserWaterVolume",
            "volFieldValue.dat",
            "alpha.water",
        ),
    }
    series: dict[str, np.ndarray] = {}
    for key, (function_name, filename, needle) in function_specs.items():
        header, data = parse_table(
            newest_data_file(args.run_dir, function_name, filename)
        )
        values = column(header, data, needle)
        series[key] = (
            interp(data[:, 0], values, time)
            if data.shape[0]
            else np.full(time.shape, np.nan)
        )

    pocket_header, pocket_data = parse_table(
        newest_data_file(args.run_dir, "endPocketGasAverage", "volFieldValue.dat")
    )
    pocket_pressure = column(pocket_header, pocket_data, "p")
    pocket_head = (
        interp(
            pocket_data[:, 0],
            (pocket_pressure - PATM) / (RHO_WATER * GRAVITY),
            time,
        )
        if pocket_data.shape[0]
        else np.full(time.shape, np.nan)
    )

    boundary_flux: dict[str, np.ndarray] = {}
    for prefix, function_name in (
        ("atmosphere", "atmosphereFluxes"),
        ("inlet", "inletFluxes"),
        ("rim", "riserMouthFluxes"),
    ):
        header, data = parse_table(
            newest_data_file(args.run_dir, function_name, "surfaceFieldValue.dat")
        )
        for key, needle in (
            ("total_phi", "sum(phi)"),
            ("water_phi", "alphaPhi0.water"),
            ("air_phi", "airPhi"),
            ("mass_phi", "rhoPhi"),
        ):
            values = column(header, data, needle)
            boundary_flux[f"{prefix}_{key}"] = (
                interp(data[:, 0], values, time)
                if data.shape[0]
                else np.full(time.shape, np.nan)
            )

    net_water_volume_out = (
        boundary_flux["atmosphere_water_phi"] + boundary_flux["inlet_water_phi"]
    )
    net_total_mass_out = (
        boundary_flux["atmosphere_mass_phi"] + boundary_flux["inlet_mass_phi"]
    )
    # rhoPhi is the solver's compressible mixture mass flux.  Subtracting the
    # nearly incompressible water contribution preserves the varying air
    # density instead of multiplying air volume flux by atmospheric density.
    net_water_mass_out = net_water_volume_out * RHO_WATER
    net_air_mass_out = net_total_mass_out - net_water_mass_out
    cumulative_water_out_mass = trapz_flux(time, net_water_mass_out)
    cumulative_air_out_mass = trapz_flux(time, net_air_mass_out)

    water_mass = series["water_mass"]
    gas_mass = series["gas_mass"]
    water_budget = water_mass + cumulative_water_out_mass
    gas_budget = gas_mass + cumulative_air_out_mass
    water_budget_error = (
        float(water_budget[-1] - water_budget[0])
        if np.all(np.isfinite(water_budget[[0, -1]]))
        else math.nan
    )
    gas_budget_error = (
        float(gas_budget[-1] - gas_budget[0])
        if np.all(np.isfinite(gas_budget[[0, -1]]))
        else math.nan
    )

    output_csv = args.output_dir / f"{run_name}-series.csv"
    columns = [
        "t_s",
        "Yfs_m",
        "Yint_m",
        "PT1_head_m",
        "PT2_head_m",
        "pocket_head_m",
        "tunnel_gas_volume_m3",
        "riser_water_volume_m3",
        "exterior_water_volume_m3",
        "rim_water_flow_m3_s",
        "atmosphere_water_flow_m3_s",
        "atmosphere_air_flow_m3_s",
        "atmosphere_total_mass_flow_kg_s",
        "inlet_total_mass_flow_kg_s",
        "net_air_mass_flow_kg_s",
        "water_mass_kg",
        "gas_mass_kg",
        "water_mass_budget_kg",
        "gas_mass_budget_kg",
    ]
    matrix = np.column_stack(
        [
            time,
            yfs,
            yint,
            pt1_head,
            pt2_head,
            pocket_head,
            series["tunnel_gas_volume"],
            series["riser_water"],
            series["ejected_water"],
            boundary_flux["rim_water_phi"],
            boundary_flux["atmosphere_water_phi"],
            boundary_flux["atmosphere_air_phi"],
            boundary_flux["atmosphere_mass_phi"],
            boundary_flux["inlet_mass_phi"],
            net_air_mass_out,
            water_mass,
            gas_mass,
            water_budget,
            gas_budget,
        ]
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(
            [[f"{value:.9g}" if math.isfinite(value) else "" for value in row] for row in matrix]
        )

    ta = first_crossing(time, yint, 0.02)
    t_rim = first_crossing(time, yfs, 0.98 * RIM_HEIGHT)
    vfs = linear_speed(time, yfs, 0.65, 1.70)
    vint = linear_speed(time, yint, 0.05, 1.65)
    initial_tunnel_gas = float(series["tunnel_gas_volume"][0])
    pocket_volume_error = initial_tunnel_gas - POCKET_VOLUME_TARGET
    reached_end = float(time[-1])
    metrics = {
        "run": run_name,
        "mode": args.mode,
        "solver": "compressibleInterFoam",
        "valve_duration_s": args.valve_duration,
        "reached_time_s": reached_end,
        "observed_3d_geyser": bool(np.nanmax(yfs) >= 0.98 * RIM_HEIGHT),
        "Ta_gas_enters_riser_s": ta,
        "t_free_surface_at_rim_s": t_rim,
        "vfs_fit_m_per_s": vfs,
        "vint_fit_m_per_s": vint,
        "Yfs_max_m": float(np.nanmax(yfs)),
        "Yint_max_m": float(np.nanmax(yint)),
        "PT1_peak_over_H0": float(np.nanmax(pt1_head / H0)),
        "pocket_peak_over_H0": float(np.nanmax(pocket_head / H0)),
        "ejected_water_max_m3": float(np.nanmax(series["ejected_water"])),
        "rim_water_flow_peak_m3_s": float(
            np.nanmax(np.abs(boundary_flux["rim_water_phi"]))
        ),
        "atmosphere_water_outflow_peak_m3_s": float(
            np.nanmax(boundary_flux["atmosphere_water_phi"])
        ),
        "initial_inventory": {
            "water_volume_m3": float(series["water_volume"][0]),
            "total_gas_volume_m3": float(series["gas_volume"][0]),
            "tunnel_pocket_volume_m3": initial_tunnel_gas,
            "analytic_pocket_volume_m3": POCKET_VOLUME_TARGET,
            "pocket_volume_relative_error": finite_relative(
                pocket_volume_error, POCKET_VOLUME_TARGET
            ),
            "analytic_pocket_air_mass_kg": POCKET_MASS_TARGET,
            "total_domain_gas_mass_kg": float(gas_mass[0]),
        },
        "conservation": {
            "water_budget_error_kg": water_budget_error,
            "water_budget_relative_error": finite_relative(
                water_budget_error, float(water_budget[0])
            ),
            "gas_budget_error_kg": gas_budget_error,
            "gas_budget_relative_error": finite_relative(
                gas_budget_error, float(gas_budget[0])
            ),
            "budget_definition": "final inventory + integrated outward boundary flux - initial inventory",
            "gas_flux_definition": (
                "solver rhoPhi minus alphaPhi0.water times 998.2 kg/m3"
            ),
        },
        "experimental_targets": {
            "classification": "GEYSER",
            "Ta_s": 8.07,
            "vfs_m_per_s": 0.924,
            "vint_m_per_s": 1.231,
        },
        "comparison_warning": (
            "Fig.10(a) is the same nominal B-1 condition; PT1 exact axial "
            "offset was not reported, so the 3-D curve is labelled PT1_proxy."
        ),
    }

    water_rel = metrics["conservation"]["water_budget_relative_error"]
    gas_rel = metrics["conservation"]["gas_budget_relative_error"]
    completed = reached_end >= {
        "static": 0.50,
        "smoke": 0.05,
        "full": 13.0,
    }[args.mode] - 1e-6
    if args.mode == "static":
        pt1_drift = float(np.nanmax(np.abs(pt1_head - pt1_head[0])))
        metrics["static_hold"] = {
            "PT1_max_drift_m_water": pt1_drift,
            "pass": bool(
                completed
                and pt1_drift <= 0.01
                and water_rel is not None
                and abs(water_rel) <= 1e-3
                and gas_rel is not None
                and abs(gas_rel) <= 1e-3
            ),
        }
    elif args.mode == "smoke":
        metrics["smoke"] = {
            "pass": bool(
                completed
                and water_rel is not None
                and abs(water_rel) <= 1e-2
                and gas_rel is not None
                and abs(gas_rel) <= 1e-2
            )
        }

    metrics_path = args.output_dir / f"{run_name}-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    exp_levels = read_csv_columns(args.experimental_levels)
    exp_pressure = read_csv_columns(args.experimental_pressure)
    one_d = read_csv_columns(args.one_d)
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), constrained_layout=True)

    axes[0].plot(time, yfs, label="3-D Yfs", color="#d62728")
    axes[0].plot(time, yint, label="3-D Yint", color="#1f77b4")
    if one_d:
        axes[0].plot(one_d["t_s"], one_d["Yfs_m"], "--", color="#d62728", alpha=0.5, label="1-D Yfs")
        axes[0].plot(one_d["t_s"], one_d["Yint_m"], "--", color="#1f77b4", alpha=0.5, label="1-D Yint")
    if exp_levels:
        fs = exp_levels["kind"] == "fs"
        interface = exp_levels["kind"] == "int"
        axes[0].scatter(exp_levels["t_s"][fs], exp_levels["Y_m"][fs], s=16, color="#d62728", marker="s", label="Fig.9(a) Yfs")
        axes[0].scatter(exp_levels["t_s"][interface], exp_levels["Y_m"][interface], s=16, facecolors="none", edgecolors="#1f77b4", label="Fig.9(a) Yint")
    axes[0].axhline(RIM_HEIGHT, color="0.4", linestyle=":", label="physical rim")
    axes[0].set_ylabel("height above crown [m]")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(time, pt1_head / H0, label="3-D PT1_proxy", color="#9467bd")
    axes[1].plot(time, pocket_head / H0, label="3-D pocket average", color="#ff7f0e")
    if one_d:
        axes[1].plot(one_d["t_s"], one_d["pocket_head_m"] / H0, "--", color="0.35", label="1-D pocket")
        if "tr_head_m" in one_d:
            axes[1].plot(
                one_d["t_s"],
                one_d["tr_head_m"] / H0,
                ":",
                color="#9467bd",
                alpha=0.65,
                label="1-D PT1 (different operator)",
            )
    if exp_pressure:
        axes[1].fill_between(
            exp_pressure["t_s"],
            exp_pressure["HoverH0_min"],
            exp_pressure["HoverH0_max"],
            color="#2ca02c",
            alpha=0.18,
        )
        axes[1].plot(exp_pressure["t_s"], exp_pressure["HoverH0_med"], color="#2ca02c", linewidth=1.0, label="Fig.10(a)")
    axes[1].set_ylabel("pressure head / H0")
    axes[1].legend(ncol=2, fontsize=8)

    axes[2].plot(time, series["tunnel_gas_volume"] * 1e3, label="tunnel gas [L]")
    axes[2].plot(time, series["ejected_water"] * 1e3, label="water above rim [L]")
    axes[2].set_xlabel("time from valve release [s]")
    axes[2].set_ylabel("volume [L]")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0, max(0.05, reached_end))
    figure.savefig(args.output_dir / f"{run_name}-comparison.png", dpi=150)
    plt.close(figure)

    if args.mode == "static" and not metrics["static_hold"]["pass"]:
        raise SystemExit("Closed-valve static-hold acceptance criteria failed")
    if args.mode == "smoke" and not metrics["smoke"]["pass"]:
        raise SystemExit("Open-valve smoke acceptance criteria failed")


if __name__ == "__main__":
    main()
