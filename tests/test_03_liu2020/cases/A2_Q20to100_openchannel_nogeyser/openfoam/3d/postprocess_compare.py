#!/usr/bin/env python3
"""Post-process Liu2020 A2 experiment, frozen 1-D model, and 3-D interFoam.

The requested comparison clock has t=0 at the start of the 0.4 s inflow ramp.
The paper and its digitized Fig. 3 instead set t=0 when the valve is fully
open, so experimental times are shifted by +0.4 s here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
MODEL = CASE_ROOT / "model"
DIGITIZED = CASE_ROOT / "data" / "digitized"
OUT = CASE_ROOT / "outputs"

RAMP_DURATION_S = 0.4
PAPER_END_S = 14.0 + RAMP_DURATION_S
Z_LID = 0.45
RISER_HEIGHT = 1.22
RISER_LEVELS = np.arange(0.46, 1.66001, 0.02)
RISER_SAMPLES_PER_LEVEL = 5
MIXTURE_ALPHA_THRESHOLD = 0.10
PAPER = {
    "PT3_initial_kPa": 0.99,
    "PT2_final_kPa": 2.15,
    "PT3_final_kPa": 4.99,
    "bore_reach_paper_clock_s": 1.20,
    "bore_reach_ramp_clock_s": 1.60,
    "first_mixture_column_m": 0.13,
    "bernoulli_estimate_m": 0.33,
    "geyser": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=HERE / "case")
    parser.add_argument("--profile", choices=("base", "refined"))
    parser.add_argument(
        "--primary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write the five required profile-independent deliverables",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_numeric_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            row = [float(value) for value in text.replace("(", " ").replace(")", " ").split()]
        except ValueError:
            continue
        if len(row) >= 2:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no numeric rows in {path}")
    width = len(rows[0])
    rows = [row for row in rows if len(row) == width]
    array = np.asarray(rows, dtype=float)
    return array[:, 0], array[:, 1:]


def read_segments(root: Path, function_name: str, field_name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((root / "postProcessing" / function_name).glob(f"*/{field_name}"))
    if not paths:
        raise FileNotFoundError(f"missing {function_name} field {field_name}")
    by_time: dict[float, np.ndarray] = {}
    for path in paths:
        time, values = read_numeric_rows(path)
        for t_value, row in zip(time, values):
            by_time[float(t_value)] = row
    times = np.asarray(sorted(by_time), dtype=float)
    return times, np.vstack([by_time[t] for t in times])


def read_function_scalar(root: Path, function_name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((root / "postProcessing" / function_name).glob("*/*.dat"))
    if not paths:
        raise FileNotFoundError(f"missing function output {function_name}")
    by_time: dict[float, float] = {}
    for path in paths:
        time, values = read_numeric_rows(path)
        for t_value, row in zip(time, values):
            by_time[float(t_value)] = float(row[-1])
    times = np.asarray(sorted(by_time), dtype=float)
    return times, np.asarray([by_time[t] for t in times], dtype=float)


def interp_nan(target: np.ndarray, time: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.interp(target, time, values)
    result[(target < time.min()) | (target > time.max())] = np.nan
    return result


def window_mean(time: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    mask = (time >= start) & (time <= end) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    mask = np.isfinite(first) & np.isfinite(second)
    return (
        float(np.sqrt(np.mean((first[mask] - second[mask]) ** 2)))
        if np.any(mask)
        else float("nan")
    )


def bore_arrival(time: np.ndarray, pressure: np.ndarray) -> float | None:
    baseline = window_mean(time, pressure, -0.5, 0.0)
    above = (time >= RAMP_DURATION_S) & (pressure >= baseline + 0.20)
    candidates = np.flatnonzero(above)
    for index in candidates:
        end = np.searchsorted(time, time[index] + 0.02)
        if end > index and np.mean(above[index:end]) >= 0.8:
            return float(time[index])
    return None


def integrate_flux(time: np.ndarray, flux: np.ndarray) -> np.ndarray:
    integral = np.zeros_like(time)
    if len(time) > 1:
        integral[1:] = np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(time))
    return integral


def riser_measures(alpha_samples: np.ndarray) -> dict[str, np.ndarray]:
    expected = len(RISER_LEVELS) * RISER_SAMPLES_PER_LEVEL
    if alpha_samples.shape[1] != expected:
        raise RuntimeError(f"expected {expected} riser probes, got {alpha_samples.shape[1]}")
    level_alpha = alpha_samples.reshape(-1, len(RISER_LEVELS), RISER_SAMPLES_PER_LEVEL).mean(axis=2)
    wet = level_alpha >= MIXTURE_ALPHA_THRESHOLD
    equivalent = np.sum(np.clip(level_alpha, 0.0, 1.0), axis=1) * 0.02
    contiguous = np.zeros(len(level_alpha))
    front = np.zeros(len(level_alpha))
    for row_index, row in enumerate(wet):
        dry = np.flatnonzero(~row)
        contiguous[row_index] = 0.02 * (int(dry[0]) if len(dry) else len(row))
        wet_indices = np.flatnonzero(row)
        front[row_index] = 0.02 * (int(wet_indices[-1]) + 1) if len(wet_indices) else 0.0
    return {
        "level_alpha": level_alpha,
        "equivalent": np.clip(equivalent, 0.0, RISER_HEIGHT),
        "contiguous": np.clip(contiguous, 0.0, RISER_HEIGHT),
        "front": np.clip(front, 0.0, RISER_HEIGHT),
        "top_alpha": level_alpha[:, -1],
    }


def parse_run_metadata(case: Path) -> dict[str, object]:
    logs = sorted(case.glob("log.interFoam.full")) + sorted(case.glob("log.interFoam.resume.*"))
    log_texts = [path.read_text(errors="replace") for path in logs if path.exists()]
    text = "\n".join(log_texts)
    co = [
        float(value)
        for value in re.findall(r"(?m)^Courant Number mean: \S+ max: (\S+)", text)
    ]
    alpha_co = [
        float(value)
        for value in re.findall(r"Interface Courant Number mean: \S+ max: (\S+)", text)
    ]
    delta_t = [float(value) for value in re.findall(r"deltaT = (\S+)", text)]
    clock = []
    for one_log in log_texts:
        values = re.findall(r"ClockTime = (\S+) s", one_log)
        if values:
            clock.append(float(values[-1]))
    build = re.search(r"Build\s+:\s+(.+)", text)

    check_text = (case / "log.checkMesh").read_text(errors="replace")
    def match_float(pattern: str) -> float | None:
        found = re.search(pattern, check_text)
        return float(found.group(1)) if found else None

    cells_match = re.search(r"\bcells:\s+(\d+)", check_text)
    return {
        "solver": "interFoam",
        "openfoam_build": build.group(1).strip() if build else "OpenFOAM v2512",
        "cells": int(cells_match.group(1)) if cells_match else None,
        "mesh_ok": "Mesh OK." in check_text and not re.search(r"Failed [1-9]", check_text),
        "max_non_orthogonality": match_float(r"non-orthogonality Max:\s+(\S+)"),
        "max_skewness": match_float(r"Max skewness =\s+(\S+)"),
        "min_cell_determinant": match_float(r"minimum:\s+(\S+) average:"),
        "minimum_delta_t_s": min(delta_t) if delta_t else None,
        "maximum_courant_number": max(co) if co else None,
        "maximum_interface_courant_number": max(alpha_co) if alpha_co else None,
        "clock_time_s": sum(clock) if clock else None,
    }


def load_experiment() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in ("PT1", "PT2", "PT3"):
        path = DIGITIZED / f"fig3_{name}.csv"
        result[name] = np.genfromtxt(path, delimiter=",", names=True)
    return result


def write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in zip(*columns):
            writer.writerow([f"{float(value):.8g}" if np.isfinite(value) else "" for value in row])


def finite(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def json_safe(value):
    """Recursively replace non-finite diagnostics with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return finite(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def grid_sensitivity(base: dict, refined: dict) -> dict[str, object]:
    def get(document: dict, *keys: str) -> float | None:
        value = document
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return float(value) if value is not None else None

    quantities = {
        "PT2_paper_window_kPa": ("openfoam_3d", "PT2_paper_window_kPa"),
        "PT3_paper_window_kPa": ("openfoam_3d", "PT3_paper_window_kPa"),
        "bore_arrival_ramp_clock_s": ("openfoam_3d", "bore_arrival_ramp_clock_s"),
        "first_contiguous_mixture_column_m": (
            "openfoam_3d",
            "first_contiguous_mixture_column_m",
        ),
        "maximum_contiguous_mixture_column_m": (
            "openfoam_3d",
            "maximum_contiguous_mixture_column_m",
        ),
        "maximum_mixture_front_m": ("openfoam_3d", "maximum_mixture_front_m"),
    }
    comparisons = {}
    for name, path in quantities.items():
        base_value = get(base, *path)
        refined_value = get(refined, *path)
        if base_value is None or refined_value is None:
            comparisons[name] = None
            continue
        delta = refined_value - base_value
        comparisons[name] = {
            "base": base_value,
            "refined": refined_value,
            "refined_minus_base": delta,
            "absolute_relative_change_percent": (
                100.0 * abs(delta) / max(abs(refined_value), 1.0e-12)
            ),
        }
    return {
        "base_cells": get(base, "openfoam_3d", "cells"),
        "refined_cells": get(refined, "openfoam_3d", "cells"),
        "quantities": comparisons,
    }


def main() -> None:
    args = parse_args()
    case = args.case_dir.resolve()
    profile = args.profile
    if profile is None:
        profile_file = case / "mesh.profile"
        if not profile_file.exists():
            raise RuntimeError("--profile required when mesh.profile is absent")
        profile = profile_file.read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)

    pressure_time, pressure_pa = read_segments(case, "probesPT", "p")
    alpha_time, alpha_samples = read_segments(case, "riserAlpha", "alpha.water")
    if pressure_time.max() < 14.39 and not args.allow_incomplete:
        raise RuntimeError(f"3-D run incomplete: last pressure time {pressure_time.max():.6g} s")
    pt3d = {
        "PT1": pressure_pa[:, 0] / 1000.0,
        "PT2": pressure_pa[:, 1] / 1000.0,
        "PT3": pressure_pa[:, 2] / 1000.0,
    }
    riser = riser_measures(alpha_samples)

    sys.path.insert(0, str(MODEL))
    from liu2020_network_twofluid import LiuCase, run_case

    one_d = run_case(LiuCase(t_end=PAPER_END_S), verbose=False)
    t1d = np.asarray(one_d["t"], dtype=float)
    p1d = {name: np.asarray(one_d[name], dtype=float) for name in ("PT1", "PT2", "PT3")}
    h1d = np.asarray(one_d["hr"], dtype=float)
    experiment = load_experiment()

    grid = np.arange(0.0, PAPER_END_S + 0.0001, 0.005)
    pressure_columns: list[np.ndarray] = [grid]
    pressure_header = ["t_ramp_start_s"]
    for name in ("PT1", "PT2", "PT3"):
        exp = experiment[name]
        exp_time = exp["t_s"] + RAMP_DURATION_S
        pressure_header.extend(
            [
                f"{name}_experiment_lo_kPa",
                f"{name}_experiment_med_kPa",
                f"{name}_experiment_hi_kPa",
                f"{name}_model_1d_kPa",
                f"{name}_openfoam_3d_kPa",
            ]
        )
        pressure_columns.extend(
            [
                interp_nan(grid, exp_time, exp["p_lo_kPa"]),
                interp_nan(grid, exp_time, exp["p_med_kPa"]),
                interp_nan(grid, exp_time, exp["p_hi_kPa"]),
                interp_nan(grid, t1d, p1d[name]),
                interp_nan(grid, pressure_time, pt3d[name]),
            ]
        )

    riser_grid = np.arange(0.0, PAPER_END_S + 0.0001, 0.01)
    riser_columns = [
        riser_grid,
        interp_nan(riser_grid, t1d, h1d),
        interp_nan(riser_grid, alpha_time, riser["equivalent"]),
        interp_nan(riser_grid, alpha_time, riser["contiguous"]),
        interp_nan(riser_grid, alpha_time, riser["front"]),
        interp_nan(riser_grid, alpha_time, riser["top_alpha"]),
    ]
    riser_header = [
        "t_ramp_start_s",
        "model_1d_column_height_m",
        "openfoam_3d_water_equivalent_height_m",
        "openfoam_3d_contiguous_mixture_column_m",
        "openfoam_3d_mixture_front_m",
        "openfoam_3d_top_mean_alpha_water",
    ]

    profile_pressure = OUT / f"openfoam_3d_{profile}_pressure_series.csv"
    profile_riser = OUT / f"openfoam_3d_{profile}_riser_series.csv"
    write_csv(profile_pressure, pressure_header, pressure_columns)
    write_csv(profile_riser, riser_header, riser_columns)

    volume_time, volume = read_function_scalar(case, "waterVolume")
    flux_names = (
        "waterFluxInlet",
        "waterFluxOutletAir",
        "waterFluxOutletWater",
        "waterFluxHeadboxAtmosphere",
        "waterFluxRiserOutlet",
    )
    flux = {}
    for name in flux_names:
        time, values = read_function_scalar(case, name)
        flux[name] = interp_nan(volume_time, time, values)
    total_flux = np.sum(np.vstack([flux[name] for name in flux_names]), axis=0)
    residual = volume - volume[0] + integrate_flux(volume_time, total_flux)
    inflow_volume = -integrate_flux(volume_time, flux["waterFluxInlet"])
    transient_mask = volume_time >= 0.0
    pre_mask = (volume_time >= -0.5) & (volume_time <= 0.0)
    pre_outflow = (
        flux["waterFluxOutletAir"]
        + flux["waterFluxOutletWater"]
        + flux["waterFluxHeadboxAtmosphere"]
        + flux["waterFluxRiserOutlet"]
    )
    if np.count_nonzero(pre_mask) >= 2:
        pre_volume_slope = float(np.polyfit(volume_time[pre_mask], volume[pre_mask], 1)[0])
    else:
        pre_volume_slope = float("nan")
    riser_outward_volume = integrate_flux(
        volume_time, np.maximum(flux["waterFluxRiserOutlet"], 0.0)
    )

    first_window = (alpha_time >= 0.0) & (alpha_time <= 3.0)
    transient_alpha = alpha_time >= 0.0
    first_column = (
        float(np.max(riser["contiguous"][first_window])) if np.any(first_window) else 0.0
    )
    max_column = (
        float(np.max(riser["contiguous"][transient_alpha])) if np.any(transient_alpha) else 0.0
    )
    max_front = (
        float(np.max(riser["front"][transient_alpha])) if np.any(transient_alpha) else 0.0
    )
    reached_top = bool(max_front >= RISER_HEIGHT - 0.01)
    discharged_riser = float(riser_outward_volume[-1]) > 1.0e-7
    geyser = bool(reached_top and discharged_riser)

    exp_interp = {}
    for name in ("PT1", "PT2", "PT3"):
        exp = experiment[name]
        exp_interp[name] = interp_nan(
            grid, exp["t_s"] + RAMP_DURATION_S, exp["p_med_kPa"]
        )
    p3_grid = {name: interp_nan(grid, pressure_time, pt3d[name]) for name in pt3d}
    p1_grid = {name: interp_nan(grid, t1d, p1d[name]) for name in p1d}
    steady_start = 7.0 + RAMP_DURATION_S

    metrics: dict[str, object] = {
        "case": "Liu2020 A2 Q20to100 open-channel no-geyser",
        "profile": profile,
        "time_origin": {
            "simulation_and_1d": "t=0 at inflow-ramp start",
            "paper_and_digitized_figure": "t=0 when valve fully open",
            "experimental_shift_applied_s": RAMP_DURATION_S,
        },
        "paper": PAPER,
        "openfoam_3d": {
            **parse_run_metadata(case),
            "simulation_start_s": float(pressure_time.min()),
            "simulation_end_s": float(pressure_time.max()),
            "PT1_initial_kPa": window_mean(pressure_time, pt3d["PT1"], -0.5, 0.0),
            "PT2_initial_kPa": window_mean(pressure_time, pt3d["PT2"], -0.5, 0.0),
            "PT3_initial_kPa": window_mean(pressure_time, pt3d["PT3"], -0.5, 0.0),
            "PT2_paper_window_kPa": window_mean(
                pressure_time, pt3d["PT2"], steady_start, PAPER_END_S
            ),
            "PT3_paper_window_kPa": window_mean(
                pressure_time, pt3d["PT3"], steady_start, PAPER_END_S
            ),
            "bore_arrival_ramp_clock_s": bore_arrival(pressure_time, pt3d["PT3"]),
            "first_contiguous_mixture_column_m": first_column,
            "maximum_contiguous_mixture_column_m": max_column,
            "maximum_mixture_front_m": max_front,
            "maximum_water_equivalent_riser_height_m": (
                float(np.max(riser["equivalent"][transient_alpha]))
                if np.any(transient_alpha)
                else 0.0
            ),
            "reached_riser_top": reached_top,
            "riser_water_discharge_m3": float(riser_outward_volume[-1]),
            "geyser": geyser,
        },
        "model_1d": {
            "PT2_paper_window_kPa": window_mean(t1d, p1d["PT2"], steady_start, PAPER_END_S),
            "PT3_paper_window_kPa": window_mean(t1d, p1d["PT3"], steady_start, PAPER_END_S),
            "maximum_riser_height_m": float(np.max(h1d)),
            "bore_arrival_ramp_clock_s": bore_arrival(t1d, p1d["PT3"]),
            "mass_error_L": float(one_d["mass_error"] * 1000.0),
            "geyser": bool(one_d["geyser"]),
        },
        "mass_conservation": {
            "initial_water_volume_m3": float(volume[0]),
            "final_water_volume_m3": float(volume[-1]),
            "final_balance_residual_m3": float(residual[-1]),
            "maximum_abs_balance_residual_m3": float(np.max(np.abs(residual))),
            "final_residual_percent_of_inflow": float(
                100.0 * residual[-1] / max(inflow_volume[-1], 1.0e-12)
            ),
            "pre_ramp_inlet_m3s": (
                float(-np.mean(flux["waterFluxInlet"][pre_mask]))
                if np.any(pre_mask)
                else float("nan")
            ),
            "pre_ramp_outlet_m3s": (
                float(np.mean(pre_outflow[pre_mask]))
                if np.any(pre_mask)
                else float("nan")
            ),
            "pre_ramp_volume_slope_m3s": pre_volume_slope,
            "transient_samples": int(np.count_nonzero(transient_mask)),
        },
        "errors": {
            "PT3_initial_minus_paper_kPa": window_mean(
                pressure_time, pt3d["PT3"], -0.5, 0.0
            )
            - PAPER["PT3_initial_kPa"],
            "PT2_steady_minus_paper_kPa": window_mean(
                pressure_time, pt3d["PT2"], steady_start, PAPER_END_S
            )
            - PAPER["PT2_final_kPa"],
            "PT3_steady_minus_paper_kPa": window_mean(
                pressure_time, pt3d["PT3"], steady_start, PAPER_END_S
            )
            - PAPER["PT3_final_kPa"],
            "first_column_minus_paper_m": first_column - PAPER["first_mixture_column_m"],
            "PT1_RMSE_vs_digitized_kPa": rmse(p3_grid["PT1"], exp_interp["PT1"]),
            "PT2_RMSE_vs_digitized_kPa": rmse(p3_grid["PT2"], exp_interp["PT2"]),
            "PT3_RMSE_vs_digitized_kPa": rmse(p3_grid["PT3"], exp_interp["PT3"]),
            "PT3_1d_RMSE_vs_digitized_kPa": rmse(p1_grid["PT3"], exp_interp["PT3"]),
        },
        "limitations": [
            "Unreported downstream tank/weir geometry is replaced by a fixed-stage hd=0.070 m boundary.",
            "Probe circumferential/in-plane coordinates are not reported by Liu et al.",
            "interFoam omits acoustic water hammer, air compressibility, bubble slip, breakup and coalescence.",
            "The experimental 0.13 m riser datum is a digitized first-column scalar, not a time series.",
        ],
    }

    other_profile = "refined" if profile == "base" else "base"
    other_path = OUT / f"openfoam_3d_{other_profile}_metrics.json"
    if other_path.exists():
        other_metrics = json.loads(other_path.read_text())
        base_metrics = metrics if profile == "base" else other_metrics
        refined_metrics = metrics if profile == "refined" else other_metrics
        metrics["grid_sensitivity"] = grid_sensitivity(base_metrics, refined_metrics)

    metrics = json_safe(metrics)
    profile_metrics = OUT / f"openfoam_3d_{profile}_metrics.json"
    profile_metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    if args.primary:
        write_csv(OUT / "openfoam_3d_pressure_series.csv", pressure_header, pressure_columns)
        write_csv(OUT / "openfoam_3d_riser_series.csv", riser_header, riser_columns)
        (OUT / "openfoam_3d_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"PT1": "#777777", "PT2": "#d95f0e", "PT3": "#2b5f9e"}
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
        for axis, name in zip(axes, ("PT3", "PT2", "PT1")):
            exp = experiment[name]
            exp_t = exp["t_s"] + RAMP_DURATION_S
            axis.fill_between(
                exp_t, exp["p_lo_kPa"], exp["p_hi_kPa"],
                color=colors[name], alpha=0.18, label=f"experiment {name} envelope",
            )
            axis.plot(exp_t, exp["p_med_kPa"], color=colors[name], lw=1.0, label="experiment median")
            axis.plot(t1d, p1d[name], color="#c81e3c", lw=1.4, label="existing 1-D model")
            axis.plot(pressure_time, pt3d[name], color="#1f6feb", lw=1.0, label=f"3-D {profile}")
            axis.set_ylabel(f"{name} [kPa]")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8, ncol=2)
        axes[0].set_ylim(-1, 10)
        axes[1].set_ylim(-3, 9)
        axes[2].set_ylim(-1.5, 2)
        axes[-1].set_xlim(0, PAPER_END_S)
        axes[-1].set_xlabel("t [s], ramp-start clock (paper traces shifted +0.4 s)")
        axes[0].set_title("Liu2020 A2: experiment vs 1-D model vs 3-D interFoam")
        fig.tight_layout()
        fig.savefig(OUT / "openfoam_3d_pressure_comparison.png", dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(11, 4.6))
        axis.plot(t1d, h1d, color="#c81e3c", lw=1.4, label="existing 1-D column")
        axis.plot(alpha_time, riser["contiguous"], color="#1f6feb", lw=1.1, label=f"3-D contiguous mixture ({profile})")
        axis.plot(alpha_time, riser["equivalent"], color="#60a5fa", lw=1.0, label="3-D water-equivalent height")
        axis.axhline(PAPER["first_mixture_column_m"], color="black", ls="--", lw=1.0, label="experiment first column 0.13 m")
        axis.axhline(RISER_HEIGHT, color="#16a34a", ls=":", lw=1.0, label="riser top / geyser threshold")
        axis.set(xlim=(0, PAPER_END_S), ylim=(0, 1.3), xlabel="t [s], ramp-start clock", ylabel="height above chamber lid [m]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        axis.set_title("Liu2020 A2 riser response")
        fig.tight_layout()
        fig.savefig(OUT / "openfoam_3d_riser_comparison.png", dpi=160)
        plt.close(fig)

    elif "grid_sensitivity" in metrics:
        primary_metrics = OUT / "openfoam_3d_metrics.json"
        if primary_metrics.exists():
            primary_document = json.loads(primary_metrics.read_text())
            primary_document["grid_sensitivity"] = metrics["grid_sensitivity"]
            primary_metrics.write_text(
                json.dumps(primary_document, indent=2, allow_nan=False) + "\n"
            )

    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"wrote profile={profile} primary={args.primary} to {OUT}")


if __name__ == "__main__":
    main()
