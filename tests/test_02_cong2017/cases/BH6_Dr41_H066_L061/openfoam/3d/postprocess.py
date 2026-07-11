#!/usr/bin/env python3
"""Create compact B-H6 3-D validation and conservation artifacts."""

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


P_ATM = 101325.0
RHO_W = 998.0
G = 9.81
H0 = 0.66
PIPE_D = 0.050
RISER_D = 0.041
L0 = 0.61
VALVE_X = 5.98
RIM_Y = 1.80  # physical riser height above horizontal-pipe soffit
RIM_Z = 1.825
R_AIR = 8314.46261815324 / 28.965
TEMPERATURE = 296.15
RISER_Z = np.arange(0.035, 1.815 + 1e-12, 0.020)
PLUME_Z = np.arange(1.835, 3.015 + 1e-12, 0.020)
AUDIT_COLUMNS = (
    "time_s",
    "water_volume_m3",
    "water_mass_kg",
    "air_volume_m3",
    "air_mass_kg",
    "apparatus_air_volume_m3",
    "downstream_air_volume_m3",
    "external_water_volume_m3",
    "reservoir_water_flux_out_m3_s",
    "atmosphere_water_flux_out_m3_s",
    "reservoir_air_mass_flux_out_kg_s",
    "atmosphere_air_mass_flux_out_kg_s",
    "max_speed_m_s",
    "min_pressure_Pa",
    "max_pressure_Pa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path.cwd())
    parser.add_argument("--reference-case-root", type=Path)
    parser.add_argument("--profile", default="custom")
    parser.add_argument("--opening-start", type=float, default=0.0)
    parser.add_argument("--opening-duration", type=float, default=0.2)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def numeric_directories(root: Path) -> list[Path]:
    paths = []
    for path in root.iterdir():
        if path.is_dir():
            try:
                float(path.name)
            except ValueError:
                continue
            paths.append(path)
    return sorted(paths, key=lambda path: float(path.name))


def probe_files(case: Path, name: str, field: str) -> list[Path]:
    root = case / "postProcessing" / name
    if not root.exists():
        raise FileNotFoundError(root)
    return [
        time_dir / field
        for time_dir in numeric_directories(root)
        if (time_dir / field).exists()
    ]


def read_scalar_probe(case: Path, name: str, field: str) -> np.ndarray:
    chunks = []
    for path in probe_files(case, name, field):
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size:
            chunks.append(data)
    if not chunks:
        raise RuntimeError(f"No probe data for {name}/{field}")
    data = np.vstack(chunks)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    _, reverse_indices = np.unique(data[::-1, 0], return_index=True)
    keep = np.sort(len(data) - 1 - reverse_indices)
    return data[keep]


def crossing(
    position0: float,
    position1: float,
    alpha0: float,
    alpha1: float,
    target: float = 0.5,
) -> float:
    if abs(alpha1 - alpha0) < 1e-12:
        return 0.5 * (position0 + position1)
    fraction = np.clip((target - alpha0) / (alpha1 - alpha0), 0.0, 1.0)
    return position0 + fraction * (position1 - position0)


def extract_riser_levels(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yint = np.zeros(alpha.shape[0])
    yfs = np.full(alpha.shape[0], np.nan)
    # Figure 7 interface trajectories are measured from the riser entrance at
    # the pipe soffit. H0 itself is separately defined from the pipe invert.
    elevations = RISER_Z - 0.025
    for row, profile in enumerate(alpha):
        wet = profile >= 0.5
        if not np.any(wet):
            yint[row] = np.nan
            continue
        first = int(np.argmax(wet))
        last = int(len(wet) - 1 - np.argmax(wet[::-1]))
        if first == 0:
            lower = elevations[0]
        else:
            lower = crossing(
                elevations[first - 1],
                elevations[first],
                profile[first - 1],
                profile[first],
            )
        if last == len(wet) - 1:
            upper = elevations[-1]
        else:
            upper = crossing(
                elevations[last],
                elevations[last + 1],
                profile[last],
                profile[last + 1],
            )
        yint[row] = max(lower, 0.0)
        yfs[row] = upper
    return yint, yfs


def first_time(
    time: np.ndarray,
    condition: np.ndarray,
    after: float | None = None,
) -> float:
    mask = condition & np.isfinite(time)
    if after is not None:
        mask &= time >= after
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if indices.size else float("nan")


def moving_average(
    time: np.ndarray, values: np.ndarray, width_s: float
) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    half = width_s / 2.0
    for index, current in enumerate(time):
        mask = (
            (time >= current - half)
            & (time <= current + half)
            & np.isfinite(values)
        )
        if np.any(mask):
            result[index] = float(np.mean(values[mask]))
    return result


def interpolation_rmse(
    model_time: np.ndarray,
    model_value: np.ndarray,
    observed_time: np.ndarray,
    observed_value: np.ndarray,
) -> float:
    finite_model = np.isfinite(model_time) & np.isfinite(model_value)
    finite_observed = np.isfinite(observed_time) & np.isfinite(observed_value)
    if np.count_nonzero(finite_model) < 2:
        return float("nan")
    lo = np.min(model_time[finite_model])
    hi = np.max(model_time[finite_model])
    mask = finite_observed & (observed_time >= lo) & (observed_time <= hi)
    if not np.any(mask):
        return float("nan")
    predicted = np.interp(
        observed_time[mask],
        model_time[finite_model],
        model_value[finite_model],
    )
    return float(np.sqrt(np.mean((predicted - observed_value[mask]) ** 2)))


def max_climb_rate(
    time: np.ndarray,
    level: np.ndarray,
    start: float,
    stop: float,
    width_s: float = 0.6,
) -> float:
    best = float("nan")
    for index, current in enumerate(time):
        if current < start or current + width_s > stop:
            continue
        mask = (
            (time >= current)
            & (time <= current + width_s)
            & np.isfinite(level)
        )
        if np.count_nonzero(mask) >= 3:
            slope = float(np.polyfit(time[mask], level[mask], 1)[0])
            best = slope if not np.isfinite(best) else max(best, slope)
    return best


def parse_audit(log_path: Path) -> np.ndarray:
    rows = []
    pattern = re.compile(r"BH6_AUDIT\s+(.+)$")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values = [float(value) for value in match.group(1).split()]
        if len(values) == len(AUDIT_COLUMNS):
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No BH6_AUDIT records in {log_path}")
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    _, reverse_indices = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - reverse_indices)]


def cumulative_trapezoid(time: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(time)
    if len(time) > 1:
        result[1:] = np.cumsum(
            0.5 * (value[1:] + value[:-1]) * np.diff(time)
        )
    return result


def parse_check_mesh(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    cells_match = re.search(r"\bcells:\s+(\d+)", text)
    return {
        "mesh_ok": "Mesh OK." in text,
        "cells": int(cells_match.group(1)) if cells_match else None,
        "max_aspect_ratio": number(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)"),
        "max_non_orthogonality_deg": number(
            r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)"
        ),
        "max_skewness": number(r"Max skewness\s*=\s*([0-9.eE+-]+)"),
        "all_geometry_and_topology": True,
    }


def write_csv(path: Path, header: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def load_named_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    case = args.case.resolve()
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    reference_case = (
        args.reference_case_root.resolve()
        if args.reference_case_root
        else case.parents[1]
    )

    pt1 = read_scalar_probe(case, "PT1", "p")
    riser = read_scalar_probe(case, "riserCentreline", "alpha.water")
    plume = read_scalar_probe(case, "plumeCentreline", "alpha.water")
    audit = parse_audit(case / "log.compressibleInterFoam")
    if riser.shape[1] - 1 != len(RISER_Z):
        raise RuntimeError(
            f"Expected {len(RISER_Z)} riser probes, got {riser.shape[1]-1}"
        )
    if plume.shape[1] - 1 != len(PLUME_Z):
        raise RuntimeError(
            f"Expected {len(PLUME_Z)} plume probes, got {plume.shape[1]-1}"
        )

    pt1_time = pt1[:, 0]
    pt1_head = (pt1[:, 1] - P_ATM) / (RHO_W * G * H0)
    pt1_smooth = moving_average(pt1_time, pt1_head, 0.10)
    riser_time = riser[:, 0]
    yint, yfs = extract_riser_levels(riser[:, 1:])
    plume_time = plume[:, 0]
    highest_water = np.full(len(plume_time), np.nan)
    for row, profile in enumerate(plume[:, 1:]):
        wet = np.flatnonzero(profile >= 0.05)
        if wet.size:
            highest_water[row] = PLUME_Z[wet[-1]] - RIM_Z

    ta = first_time(riser_time, yint >= 0.02)
    catch = first_time(
        riser_time,
        (yint >= 0.10) & ((yfs - yint) <= 0.05),
        after=ta if np.isfinite(ta) else None,
    )
    water_above_rim = bool(np.any(np.isfinite(highest_water)))
    yfs_max = float(np.nanmax(yfs)) if np.any(np.isfinite(yfs)) else float("nan")
    event_stop = catch if np.isfinite(catch) else float(riser_time[-1])
    vfs = (
        max_climb_rate(riser_time, yfs, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    vint = (
        max_climb_rate(riser_time, yint, ta, event_stop)
        if np.isfinite(ta)
        else float("nan")
    )
    pressure_peak = (
        float(np.nanmax(pt1_smooth[pt1_time >= ta]))
        if np.isfinite(ta) and np.any(pt1_time >= ta)
        else float("nan")
    )

    audit_time = audit[:, 0]
    water_flux = audit[:, 8] + audit[:, 9]
    gas_mass_flux = audit[:, 10] + audit[:, 11]
    cumulative_water_out = cumulative_trapezoid(audit_time, water_flux)
    cumulative_gas_out = cumulative_trapezoid(audit_time, gas_mass_flux)
    water_residual = (
        audit[:, 1] - audit[0, 1] + cumulative_water_out
    )
    gas_residual = audit[:, 4] - audit[0, 4] + cumulative_gas_out
    cumulative_far_water_out = cumulative_trapezoid(
        audit_time, np.maximum(audit[:, 9], 0.0)
    )
    expelled_water = audit[:, 7] + cumulative_far_water_out
    estimated_rim_flow = np.gradient(expelled_water, audit_time)

    balance_rows = np.column_stack(
        (
            audit,
            cumulative_water_out,
            water_residual,
            cumulative_gas_out,
            gas_residual,
            cumulative_far_water_out,
            expelled_water,
            estimated_rim_flow,
        )
    )
    balance_header = AUDIT_COLUMNS + (
        "cumulative_water_out_m3",
        "water_volume_balance_residual_m3",
        "cumulative_air_mass_out_kg",
        "air_mass_balance_residual_kg",
        "cumulative_farfield_water_out_m3",
        "expelled_water_volume_m3",
        "estimated_riser_outflow_m3_s",
    )
    write_csv(results / "mass_balance.csv", balance_header, balance_rows)

    common_time = riser_time
    series_rows = np.column_stack(
        (
            common_time,
            yfs,
            yint,
            np.interp(common_time, pt1_time, pt1_head),
            np.interp(common_time, pt1_time, pt1_smooth),
            np.interp(common_time, audit_time, audit[:, 5]),
            np.interp(common_time, audit_time, audit[:, 6]),
            np.interp(common_time, audit_time, expelled_water),
            np.interp(common_time, plume_time, highest_water),
        )
    )
    write_csv(
        results / "series.csv",
        (
            "time_s",
            "Yfs_3d_m",
            "Yint_3d_m",
            "PT1_H_over_H0_raw",
            "PT1_H_over_H0_smooth_0p10s",
            "apparatus_air_volume_m3",
            "downstream_air_volume_m3",
            "expelled_water_volume_m3",
            "sampled_water_height_above_rim_m",
        ),
        series_rows,
    )

    levels_exp = load_named_csv(
        reference_case / "data" / "digitized" / "fig7a_levels.csv"
    )
    pressure_exp = load_named_csv(
        reference_case / "data" / "digitized" / "fig10b_pt1.csv"
    )
    one_d = load_named_csv(reference_case / "outputs" / "caseB_model_series.csv")
    fs_mask = levels_exp["kind"] == "fs"
    int_mask = levels_exp["kind"] == "int"
    fs_rmse = interpolation_rmse(
        riser_time,
        yfs,
        levels_exp["t_s"][fs_mask],
        levels_exp["Y_m"][fs_mask],
    )
    int_rmse = interpolation_rmse(
        riser_time,
        yint,
        levels_exp["t_s"][int_mask],
        levels_exp["Y_m"][int_mask],
    )
    pressure_rmse = interpolation_rmse(
        pt1_time,
        pt1_smooth,
        pressure_exp["t_s"],
        pressure_exp["HoverH0_med"],
    )

    geometry = json.loads(
        (case / "geometry_audit.runtime.json").read_text(encoding="utf-8")
    )
    mesh = parse_check_mesh(case / "log.checkMesh")
    nominal_pocket_volume = math.pi * PIPE_D**2 * L0 / 4.0
    nominal_pocket_mass = (
        P_ATM * nominal_pocket_volume / (R_AIR * TEMPERATURE)
    )
    nominal_water_volume = (
        math.pi * PIPE_D**2 * VALVE_X / 4.0
        + math.pi * RISER_D**2 * (H0 - PIPE_D) / 4.0
    )
    numerical_pocket_volume = float(audit[0, 6])
    numerical_pocket_mass = (
        numerical_pocket_volume * P_ATM / (R_AIR * TEMPERATURE)
    )
    initial_audit = {
        "first_numerical_sample_time_s": float(audit_time[0]),
        "analytical_initial_water_volume_m3": nominal_water_volume,
        "numerical_first_sample_water_volume_m3": float(audit[0, 1]),
        "analytical_initial_pocket_volume_m3": nominal_pocket_volume,
        "numerical_first_sample_downstream_air_volume_m3": numerical_pocket_volume,
        "pocket_volume_relative_discretisation_error": (
            numerical_pocket_volume / nominal_pocket_volume - 1.0
        ),
        "analytical_initial_pocket_air_mass_kg": nominal_pocket_mass,
        "numerical_first_sample_pocket_air_mass_kg": numerical_pocket_mass,
        "whole_domain_first_sample_air_mass_kg": float(audit[0, 4]),
        "note": (
            "The whole-domain gas mass includes the external atmosphere. "
            "Pocket mass uses the valve-to-cap gas region at the first audit sample."
        ),
    }
    (results / "initial_volume_mass_audit.json").write_text(
        json.dumps(initial_audit, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    water_scale = max(abs(audit[0, 1]), 1e-30)
    gas_scale = max(abs(audit[0, 4]), 1e-30)
    metrics = {
        "case": "BH6_Dr41_H066_L061",
        "profile": args.profile,
        "simulation_end_s": float(
            min(pt1_time[-1], riser_time[-1], audit_time[-1])
        ),
        "solver": "compressibleInterFoam",
        "geometry": {
            "type": "true 3-D circular pipe, circular riser, T-junction, external air",
            "main_pipe_diameter_m": PIPE_D,
            "riser_diameter_m": RISER_D,
            "area_ratio": (RISER_D / PIPE_D) ** 2,
            "pipe_length_m": 6.59,
            "tee_x_m": 3.47,
            "valve_x_m": VALVE_X,
            "physical_riser_height_m": 1.8,
            "external_top_above_soffit_m": 3.0,
        },
        "mesh": {**mesh, **geometry.get("mesh_sizes_m", {})},
        "valve": {
            "opening_start_s": args.opening_start,
            "opening_duration_s": args.opening_duration,
            "representation": "variable-area cyclicACMI baffle",
            "table_samples": 40,
            "passive_loss_only": True,
        },
        "events": {
            "Ta_3d_s": ta,
            "interface_catch_3d_s": catch,
            "Yfs_max_3d_m": yfs_max,
            "vfs_max_sustained_0p6s_m_s": vfs,
            "vint_max_sustained_0p6s_m_s": vint,
            "water_above_rim": water_above_rim,
            "max_sampled_water_height_above_rim_m": (
                float(np.nanmax(highest_water))
                if water_above_rim
                else 0.0
            ),
            "geyser_3d": water_above_rim,
            "max_expelled_water_volume_m3": float(np.nanmax(expelled_water)),
        },
        "pressure": {
            "post_arrival_peak_H_over_H0": pressure_peak,
            "experiment_proxy": "Run B-32, same nominal Dr/H0/L0 as B-H6",
            "experiment_post_arrival_slow_peak_H_over_H0": 1.4,
            "RMSE_H_over_H0_no_time_shift": pressure_rmse,
        },
        "experiment": {
            "run": "B-H6",
            "Ta_s": 8.10,
            "vfs_m_s": 0.246,
            "vint_m_s": 0.476,
            "interface_catch_s": 10.5,
            "Yfs_peak_m": 1.21,
            "geyser": False,
        },
        "one_d": {
            "source": "outputs/caseB_model_series.csv",
            "geometry_difference": (
                "Frozen 1-D uses 6.0 m / tee x=2.88 m; "
                "paper-audited 3-D uses 6.59 m / tee x=3.47 m."
            ),
            "Yfs_max_m": float(np.nanmax(one_d["Yfs_m"])),
            "geyser": bool(np.nanmax(one_d["Yfs_m"]) >= 0.98 * 1.8),
        },
        "comparison": {
            "Yfs_RMSE_m_no_time_shift": fs_rmse,
            "Yint_RMSE_m_no_time_shift": int_rmse,
            "PT1_RMSE_H_over_H0_no_time_shift": pressure_rmse,
            "no_event_time_shift": True,
            "no_outcome_tuning": True,
        },
        "conservation": {
            "liquid_volume_final_residual_m3": float(water_residual[-1]),
            "liquid_volume_relative_residual": float(
                water_residual[-1] / water_scale
            ),
            "liquid_volume_max_relative_residual": float(
                np.nanmax(np.abs(water_residual)) / water_scale
            ),
            "gas_mass_final_residual_kg": float(gas_residual[-1]),
            "gas_mass_relative_residual": float(gas_residual[-1] / gas_scale),
            "gas_mass_max_relative_residual": float(
                np.nanmax(np.abs(gas_residual)) / gas_scale
            ),
            "open_boundary_fluxes_included": True,
        },
        "static_diagnostics": {
            "max_speed_m_s": float(np.nanmax(audit[:, 12])),
            "water_volume_change_m3": float(audit[-1, 1] - audit[0, 1]),
            "downstream_air_volume_change_m3": float(
                audit[-1, 6] - audit[0, 6]
            ),
        },
        "limitations": [
            "Paper gives PT1 topology but no numerical cap offset.",
            "Neutral 90 degree acrylic contact angle is not measured or calibrated.",
            "Laminar stress closure is common to the B-H1/B-H6 pair.",
            "Fig.10(b) pressure is B-32, a same-condition repeat, not the B-H6 camera run.",
        ],
    }
    (results / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True)
    axes[0].scatter(
        levels_exp["t_s"][fs_mask],
        levels_exp["Y_m"][fs_mask],
        s=16,
        facecolors="none",
        edgecolors="#b22222",
        label="experiment B-H6 $Y_{fs}$",
    )
    axes[0].scatter(
        levels_exp["t_s"][int_mask],
        levels_exp["Y_m"][int_mask],
        s=16,
        facecolors="none",
        edgecolors="#1f4e79",
        label="experiment B-H6 $Y_{int}$",
    )
    axes[0].plot(one_d["t_s"], one_d["Yfs_m"], color="0.55", lw=1, label="1-D $Y_{fs}$")
    axes[0].plot(
        one_d["t_s"], one_d["Yint_m"], color="0.55", lw=1, ls="--", label="1-D $Y_{int}$"
    )
    axes[0].plot(riser_time, yfs, color="#b22222", lw=1.4, label="3-D $Y_{fs}$")
    axes[0].plot(
        riser_time, yint, color="#1f4e79", lw=1.4, ls="--", label="3-D $Y_{int}$"
    )
    axes[0].axhline(RIM_Y, color="k", lw=0.8, ls=":", label="physical rim")
    axes[0].set_ylabel("level above pipe invert [m]")
    axes[0].legend(frameon=False, fontsize=7, ncol=3)

    axes[1].fill_between(
        pressure_exp["t_s"],
        pressure_exp["HoverH0_min"],
        pressure_exp["HoverH0_max"],
        color="0.88",
        label="B-32 digitised band",
    )
    axes[1].plot(
        pressure_exp["t_s"],
        pressure_exp["HoverH0_med"],
        color="0.25",
        lw=1,
        label="B-32 median",
    )
    axes[1].plot(pt1_time, pt1_smooth, color="#e66101", lw=1.4, label="3-D PT1")
    axes[1].set_ylabel("$H/H_0$")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        audit_time,
        1e3 * audit[:, 5],
        label="air below physical rim",
        color="#4c78a8",
    )
    axes[2].plot(
        audit_time,
        1e3 * audit[:, 6],
        label="air in initial pocket region",
        color="#72b7b2",
    )
    axes[2].plot(
        audit_time,
        1e6 * expelled_water,
        label="expelled water",
        color="#b22222",
    )
    axes[2].set_ylabel("air [L] / expelled water [mL]")
    axes[2].set_xlabel("time after valve motion begins [s]")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xlim(0, max(13.0, float(riser_time[-1])))
    fig.tight_layout()
    fig.savefig(results / "experiment_1d_3d_comparison.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
