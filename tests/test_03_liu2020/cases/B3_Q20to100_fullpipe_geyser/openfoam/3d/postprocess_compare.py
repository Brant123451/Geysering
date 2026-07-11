#!/usr/bin/env python3
"""Post-process Liu2020 B3 OpenFOAM probes and compare with paper and 1-D data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
OUT = CASE_ROOT / "outputs"
DIG = CASE_ROOT / "data" / "digitized"
MODEL = CASE_ROOT / "model"

PATM = 101325.0
RHO_WATER = 998.2
G = 9.81
RAMP_START = 2.0
RAMP_END = 2.4
CHAMBER_LID_Z = 0.45
RISER_LENGTH = 1.22
RISER_RIM_Z = CHAMBER_LID_Z + RISER_LENGTH
PLUME_TOP_Z = 5.25
ALPHA_WET = 0.05
RISER_Z = np.arange(0.47, 5.2200001, 0.05)

PAPER = {
    "source_note": (
        "Values frozen from the paper quotations/parsed metadata in this "
        "commit; the PDF and page scans are absent (see openfoam/3d/PAPER_AUDIT.md)."
    ),
    "geyser": True,
    "bore_reach_chamber_s": 1.20,
    "PT2_peak_kPa": 55.03,
    "PT3_peak_kPa": 51.76,
    "t_peak_s": 1.47,
    "PT1_min_kPa": -8.30,
    "PT2_min_kPa": -20.26,
    "PT3_min_kPa": -17.77,
    "osc_periods_s": [0.51, 0.37, 0.37],
    "PT1_final_kPa": 0.0,
    "PT2_final_kPa": 1.82,
    "PT3_final_kPa": 4.65,
    "fig7a_slope": 0.6943,
    "fig7a_intercept_m": 0.3086,
    "fig7a_r2": 0.97,
    "t_mist_s": 1.47,
    "t_jet_out_s": 1.51,
    "t_column_top_s": 1.65,
    "t_break_s": [1.70, 1.89],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=HERE / "case")
    parser.add_argument(
        "--mesh-label",
        default="baseline",
        help="Row key written to openfoam_3d_mesh_sensitivity.csv",
    )
    parser.add_argument(
        "--primary",
        action="store_true",
        help="Write the required primary CSV/JSON/PNG outputs",
    )
    return parser.parse_args()


def numeric_dirs(path: Path) -> list[Path]:
    result: list[tuple[float, Path]] = []
    for entry in path.iterdir() if path.exists() else ():
        try:
            result.append((float(entry.name), entry))
        except ValueError:
            continue
    return [entry for _, entry in sorted(result)]


def deduplicate_time(table: np.ndarray) -> np.ndarray:
    if table.ndim == 1:
        table = table[None, :]
    order = np.argsort(table[:, 0], kind="stable")
    table = table[order]
    # Restart folders can repeat their first sample. Keep the later occurrence.
    _, reverse_indices = np.unique(table[::-1, 0], return_index=True)
    keep = len(table) - 1 - reverse_indices
    return table[np.sort(keep)]


def read_scalar_probes(case: Path, function: str, field: str) -> np.ndarray:
    root = case / "postProcessing" / function
    chunks: list[np.ndarray] = []
    for directory in numeric_dirs(root):
        source = directory / field
        if source.exists():
            data = np.loadtxt(source, comments="#")
            if data.size:
                chunks.append(np.atleast_2d(data))
    if not chunks:
        raise FileNotFoundError(f"No probe data: {root}/*/{field}")
    return deduplicate_time(np.vstack(chunks))


def read_vector_probes(case: Path, function: str, field: str) -> np.ndarray:
    root = case / "postProcessing" / function
    rows: list[list[float]] = []
    for directory in numeric_dirs(root):
        source = directory / field
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            groups = re.findall(r"\(([^()]*)\)", line)
            prefix = line.split("(", 1)[0].strip().split()
            if not prefix:
                continue
            row = [float(prefix[0])]
            for group in groups:
                values = [float(value) for value in group.split()]
                if len(values) != 3:
                    raise ValueError(f"Expected vector probe value in {source}: {group}")
                row.extend(values)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No vector probe data: {root}/*/{field}")
    return deduplicate_time(np.asarray(rows, dtype=float))


def read_function_table(case: Path, function: str) -> np.ndarray:
    root = case / "postProcessing" / function
    chunks: list[np.ndarray] = []
    for directory in numeric_dirs(root):
        candidates = sorted(directory.glob("*.dat"))
        if not candidates:
            continue
        data = np.loadtxt(candidates[0], comments="#")
        if data.size:
            chunks.append(np.atleast_2d(data))
    if not chunks:
        raise FileNotFoundError(f"No function-object table: {root}")
    return deduplicate_time(np.vstack(chunks))


def load_experiment(sensor: str) -> np.ndarray:
    source = DIG / f"fig5b_{sensor}.csv"
    data = np.genfromtxt(source, delimiter=",", names=True)
    return np.column_stack(
        [data["t_s"], data["p_lo_kPa"], data["p_med_kPa"], data["p_hi_kPa"]]
    )


def load_one_dimensional() -> dict[str, np.ndarray | float | bool]:
    sys.path.insert(0, str(MODEL))
    try:
        from liu2020_network_twofluid import LiuCase, run_case
    finally:
        sys.path.pop(0)
    record = run_case(LiuCase(t_end=14.0, downstream_full=True), verbose=False)
    return {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in record.items()
    }


def interp_nan(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp, left=np.nan, right=np.nan)


def relative_error(value: float, target: float) -> float | None:
    if not np.isfinite(value):
        return None
    if abs(target) < 1e-12:
        return None
    return 100.0 * abs(value - target) / abs(target)


def first_sustained_crossing(
    time: np.ndarray, values: np.ndarray, threshold: float, count: int = 4
) -> float | None:
    hit = values >= threshold
    if len(hit) < count:
        return None
    run = np.convolve(hit.astype(int), np.ones(count, dtype=int), mode="valid")
    indices = np.flatnonzero(run == count)
    return float(time[indices[0]]) if indices.size else None


def oscillation_periods(
    time: np.ndarray, values: np.ndarray, main_peak_index: int
) -> list[float]:
    mask = (
        (time >= time[main_peak_index] + 0.10)
        & (time <= min(time[main_peak_index] + 2.0, 4.5))
    )
    tt, yy = time[mask], values[mask]
    if len(tt) < 9:
        return []
    dt = float(np.median(np.diff(tt)))
    # Match the established B3 comparison: about 50 ms total smoothing,
    # positive peaks only, and at least 0.20 s between successive peaks.
    half_window = max(1, int(round(0.025 / max(dt, 1e-9))))
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    smooth = np.convolve(yy, kernel, mode="same")
    candidates: list[int] = []
    neighbourhood = half_window
    minimum_separation = 0.20
    for i in range(neighbourhood, len(smooth) - neighbourhood):
        window = smooth[i - neighbourhood : i + neighbourhood + 1]
        if smooth[i] == np.max(window) and smooth[i] > 3.0:
            if not candidates or tt[i] - tt[candidates[-1]] >= minimum_separation:
                candidates.append(i)
            elif smooth[i] > smooth[candidates[-1]]:
                candidates[-1] = i
    return [
        float(tt[candidates[i + 1]] - tt[candidates[i]])
        for i in range(min(3, len(candidates) - 1))
    ]


def pressure_metrics(time: np.ndarray, pressure: dict[str, np.ndarray]) -> dict:
    event = (time >= 0.0) & (time <= 4.5)
    if not np.any(event):
        raise RuntimeError("OpenFOAM result does not cover the event window")
    event_indices = np.flatnonzero(event)
    reference_peak_index = event_indices[
        int(np.argmax(pressure["PT2"][event]))
    ]
    # Paper points D/E are the first rebound after point C, not the lowest
    # value of every later oscillation.  A 0.62 s window includes the first
    # trough while excluding the next trough for the reported 0.51/0.37 s
    # oscillation periods.
    rebound = (
        (time >= time[reference_peak_index] + 0.02)
        & (time <= time[reference_peak_index] + 0.62)
        & (time <= 4.5)
    )
    if not np.any(rebound):
        raise RuntimeError("OpenFOAM result does not cover the first rebound window")
    rebound_indices = np.flatnonzero(rebound)
    result: dict[str, dict[str, object]] = {}
    for sensor, values in pressure.items():
        peak_index = event_indices[int(np.argmax(values[event]))]
        minimum_index = rebound_indices[int(np.argmin(values[rebound]))]
        final = (time >= 10.0) & (time <= 14.4)
        result[sensor] = {
            "peak_kPa": float(values[peak_index]),
            "peak_time_s": float(time[peak_index]),
            "minimum_kPa": float(values[minimum_index]),
            "minimum_time_s": float(time[minimum_index]),
            "final_mean_kPa": float(np.mean(values[final])) if np.any(final) else None,
            "final_std_kPa": float(np.std(values[final])) if np.any(final) else None,
            "minimum_definition": (
                "minimum from 0.02 to 0.62 s after the PT2 principal peak "
                "(paper first-rebound window)"
            ),
        }
    return result


def parse_check_mesh(case: Path) -> dict[str, float | int | str | None]:
    source = case / "log.checkMesh"
    text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""

    def number(pattern: str, cast=float):
        match = re.search(pattern, text, re.MULTILINE)
        return cast(match.group(1)) if match else None

    return {
        "cells": number(r"^\s*cells:\s+(\d+)", int),
        "max_aspect_ratio": number(r"Max aspect ratio\s*=\s*([0-9.eE+-]+)"),
        "max_non_orthogonality_deg": number(
            r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)"
        ),
        "mean_non_orthogonality_deg": number(
            r"Mesh non-orthogonality Max:\s*[0-9.eE+-]+\s+average:\s*([0-9.eE+-]+)"
        ),
        "max_skewness": number(r"Max skewness\s*=\s*([0-9.eE+-]+)"),
        "min_cell_volume_m3": number(
            r"Min volume\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        ),
        "underdetermined_tetrahedra": number(
            r"Cells with small determinant \(< 0\.001\) found, number of cells:\s*(\d+)",
            int,
        ),
        "low_interpolation_weight_faces": number(
            r"Faces with small interpolation weight \(< 0\.05\) found, number of faces:\s*(\d+)",
            int,
        )
        or 0,
        "failed_checks": number(r"Failed\s+(\d+)\s+mesh checks", int) or 0,
        "check_mesh_ok": "Mesh OK." in text,
        "command": "checkMesh -allGeometry -allTopology",
    }


def parse_velocity_limiter(case: Path) -> dict[str, object]:
    source = case / "log.compressibleInterFoam"
    if not source.exists():
        source = case / "log.compressibleInterFoam.smoke"
    text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
    counts = [
        int(value)
        for value in re.findall(
            r"limitVelocity\s+\S+\s+Limited\s+(\d+)\s+\(", text
        )
    ]
    return {
        "velocity_limit_m_s": 50.0,
        "limiter_calls": len(counts),
        "calls_with_activation": sum(value > 0 for value in counts),
        "maximum_cells_limited_in_one_correction": max(counts, default=0),
        "activated": any(value > 0 for value in counts),
        "interpretation": (
            "The 50 m/s bound is outside the expected B3 state. Any activation "
            "is reported as a numerical-quality warning, not a physical result."
        ),
    }


def calculate_mass_balance(case: Path) -> dict:
    volume = read_function_table(case, "waterVolume")[:, :2]
    fluxes = {}
    for name in ("waterFluxInlet", "waterFluxOutlet", "waterFluxAtmosphere"):
        table = read_function_table(case, name)
        fluxes[name] = interp_nan(volume[:, 0], table[:, 0], table[:, 1])

    valid = np.ones(len(volume), dtype=bool)
    for values in fluxes.values():
        valid &= np.isfinite(values)
    time = volume[valid, 0]
    water_volume = volume[valid, 1]
    if len(time) < 2:
        raise RuntimeError("Insufficient volume/flux samples for mass balance")
    outward_flux = sum(values[valid] for values in fluxes.values())
    integrated = np.zeros(len(time))
    integrated[1:] = np.cumsum(
        0.5 * (outward_flux[1:] + outward_flux[:-1]) * np.diff(time)
    )
    residual = water_volume - water_volume[0] + integrated
    inlet_flux = fluxes["waterFluxInlet"][valid]
    inlet_volume = np.zeros(len(time))
    inlet_volume[1:] = np.cumsum(
        -0.5 * (inlet_flux[1:] + inlet_flux[:-1]) * np.diff(time)
    )
    reference = max(water_volume[0] + max(inlet_volume[-1], 0.0), 1e-12)
    signed_percent = float(100 * residual[-1] / reference)
    return {
        "initial_water_volume_m3": float(water_volume[0]),
        "final_water_volume_m3": float(water_volume[-1]),
        "net_integrated_outward_water_flux_m3": float(integrated[-1]),
        "numerical_mass_error_m3": float(residual[-1]),
        "numerical_mass_error_L": float(1000 * residual[-1]),
        "numerical_mass_error_percent": abs(signed_percent),
        "signed_numerical_mass_error_percent": signed_percent,
        "maximum_absolute_mass_error_L": float(1000 * np.max(np.abs(residual))),
        "method": (
            "Liquid-volume closure: V(alpha.water) change plus trapezoidal "
            "integral of alpha.water-weighted phi over inlet, submerged outlet "
            "and atmosphere. Small physical volume changes from water "
            "compressibility are included in this residual."
        ),
    }


def inlet_flow_check(case: Path) -> dict:
    table = read_function_table(case, "mixtureFluxInlet")
    time, flow_in = table[:, 0], -table[:, 1]
    settle = (time >= 0.2) & (time <= 1.9)
    final = time >= 2.5
    q0 = float(np.mean(flow_in[settle])) if np.any(settle) else math.nan
    q1 = float(np.mean(flow_in[final])) if np.any(final) else math.nan
    return {
        "Q0_mean_m3_s": q0,
        "Q0_relative_error_percent": relative_error(q0, 0.020),
        "Q1_mean_m3_s": q1,
        "Q1_relative_error_percent": relative_error(q1, 0.100),
    }


def write_pressure_csv(
    path: Path,
    solver_time: np.ndarray,
    report_time: np.ndarray,
    pressure: dict[str, np.ndarray],
    one_d: dict,
    experiment: dict[str, np.ndarray],
) -> None:
    one_time = np.asarray(one_d["t"])
    headers = ["solver_time_s", "time_after_ramp_s"]
    columns: list[np.ndarray] = [solver_time, report_time]
    for sensor in ("PT1", "PT2", "PT3"):
        headers.extend(
            [
                f"openfoam_{sensor}_kPa",
                f"one_dimensional_{sensor}_kPa",
                f"experiment_{sensor}_median_kPa",
                f"experiment_{sensor}_low_kPa",
                f"experiment_{sensor}_high_kPa",
            ]
        )
        exp = experiment[sensor]
        columns.extend(
            [
                pressure[sensor],
                interp_nan(report_time, one_time, np.asarray(one_d[sensor])),
                interp_nan(report_time, exp[:, 0], exp[:, 2]),
                interp_nan(report_time, exp[:, 0], exp[:, 1]),
                interp_nan(report_time, exp[:, 0], exp[:, 3]),
            ]
        )
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.9g",
    )


def write_riser_csv(
    path: Path,
    solver_time: np.ndarray,
    report_time: np.ndarray,
    alpha: np.ndarray,
    velocity: np.ndarray,
) -> dict:
    highest_z = np.full(len(report_time), np.nan)
    riser_column = np.zeros(len(report_time))
    rim_alpha = alpha[:, int(np.argmin(np.abs(RISER_Z - RISER_RIM_Z)))]
    max_vertical_velocity = np.max(velocity[:, :, 2], axis=1)
    for row, profile in enumerate(alpha):
        wet = np.flatnonzero(profile >= ALPHA_WET)
        if wet.size:
            highest_z[row] = RISER_Z[wet[-1]]
            riser_wet = wet[RISER_Z[wet] <= RISER_RIM_Z + 1e-9]
            if riser_wet.size:
                riser_column[row] = max(
                    0.0, min(RISER_LENGTH, RISER_Z[riser_wet[-1]] - CHAMBER_LID_Z)
                )
    height_above_lid = np.where(
        np.isfinite(highest_z), np.maximum(highest_z - CHAMBER_LID_Z, 0.0), 0.0
    )
    height_above_rim = np.maximum(height_above_lid - RISER_LENGTH, 0.0)
    geyser = height_above_rim >= 0.05 - 1e-9
    top_crossing = first_sustained_crossing(report_time, rim_alpha, ALPHA_WET, 2)
    plume_crossing = first_sustained_crossing(
        report_time, height_above_rim, 0.05 - 1e-9, 2
    )

    np.savetxt(
        path,
        np.column_stack(
            [
                solver_time,
                report_time,
                riser_column,
                highest_z,
                height_above_lid,
                height_above_rim,
                rim_alpha,
                max_vertical_velocity,
                geyser.astype(int),
            ]
        ),
        delimiter=",",
        header=(
            "solver_time_s,time_after_ramp_s,riser_column_height_m,"
            "highest_wet_z_m,geyser_height_above_lid_m,height_above_rim_m,"
            "rim_alpha_water,max_vertical_velocity_m_s,water_above_rim"
        ),
        comments="",
        fmt="%.9g",
    )
    return {
        "geyser": bool(np.any(geyser)),
        "first_reach_riser_top_s": top_crossing,
        "first_water_above_rim_s": plume_crossing,
        "maximum_geyser_height_above_lid_m": float(np.max(height_above_lid)),
        "maximum_height_above_rim_m": float(np.max(height_above_rim)),
        "maximum_sampled_vertical_velocity_m_s": float(np.max(max_vertical_velocity)),
        "height_detection": (
            f"highest centreline alpha.water >= {ALPHA_WET} at 0.05 m spacing; "
            "height is measured above the chamber lid to match the Fig. 7 threshold"
        ),
        "plume_domain_top_height_above_lid_m": PLUME_TOP_Z - CHAMBER_LID_Z,
        "height_censored_by_domain": bool(
            np.max(height_above_lid) >= PLUME_TOP_Z - CHAMBER_LID_Z - 0.051
        ),
    }


def write_pressure_plot(
    path: Path,
    report_time: np.ndarray,
    pressure: dict[str, np.ndarray],
    one_d: dict,
    experiment: dict[str, np.ndarray],
    metrics: dict,
) -> None:
    colours = {"PT1": "#2b3f9e", "PT2": "#c81e3c", "PT3": "#1d8a4a"}
    one_time = np.asarray(one_d["t"])
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True)
    for ax, sensor in zip(axes, ("PT2", "PT3", "PT1")):
        exp = experiment[sensor]
        ax.fill_between(
            exp[:, 0],
            exp[:, 1],
            exp[:, 3],
            color=colours[sensor],
            alpha=0.18,
            label="paper Fig. 5(b) digitised envelope",
        )
        ax.plot(
            exp[:, 0], exp[:, 2], color=colours[sensor], lw=1.0,
            label="paper digitised median",
        )
        ax.plot(
            one_time,
            np.asarray(one_d[sensor]),
            color="#7c3aed",
            ls="--",
            lw=1.2,
            label="existing 1-D model",
        )
        ax.plot(
            report_time,
            pressure[sensor],
            color="#111827",
            lw=1.1,
            label="3-D compressibleInterFoam",
        )
        target_peak = PAPER.get(f"{sensor}_peak_kPa")
        target_min = PAPER[f"{sensor}_min_kPa"]
        if target_peak is not None:
            ax.axhline(target_peak, color="#dc2626", ls=":", lw=0.9)
        ax.axhline(target_min, color="#2563eb", ls=":", lw=0.9)
        ax.axvline(PAPER["bore_reach_chamber_s"], color="#64748b", ls=":", lw=0.8)
        ax.axvline(PAPER["t_peak_s"], color="#dc2626", ls=":", lw=0.8)
        final_target = PAPER[f"{sensor}_final_kPa"]
        final_model = metrics["pressure"][sensor]["final_mean_kPa"]
        final_text = (
            f"final mean: exp {final_target:.2f}, "
            f"3-D {final_model:.2f} kPa"
            if final_model is not None
            else f"final target: {final_target:.2f} kPa (run too short)"
        )
        ax.text(
            0.01, 0.04, final_text, transform=ax.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        ax.set_ylabel(f"{sensor} [kPa]")
        ax.set_xlim(-0.5, 4.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylim(-30, 70)
    axes[1].set_ylim(-30, 70)
    axes[2].set_ylim(-15, 15)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("time after start of 0.4 s inflow ramp [s]")
    axes[0].set_title(
        "Liu2020 B3 pressure validation — paper, existing 1-D, and 3-D OpenFOAM\n"
        "vertical lines: bore arrival 1.20 s and main peak 1.47 s"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_height_plot(
    path: Path,
    riser_csv: Path,
    one_d: dict,
    pressure_metrics_: dict,
    riser_metrics: dict,
) -> None:
    riser = np.genfromtxt(riser_csv, delimiter=",", names=True)
    one_time = np.asarray(one_d["t"])
    one_hr = np.asarray(one_d["hr"])
    paper_height = (
        PAPER["fig7a_slope"]
        * PAPER["PT2_peak_kPa"]
        * 1000
        / (RHO_WATER * G)
        + PAPER["fig7a_intercept_m"]
    )
    one_pmax = float(np.max(np.asarray(one_d["PT2"])))
    of_pmax = float(pressure_metrics_["PT2"]["peak_kPa"])
    one_height = float(one_d["h_jet"])
    of_height = float(riser_metrics["maximum_geyser_height_above_lid_m"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    ax.plot(
        one_time, one_hr, color="#7c3aed", ls="--", lw=1.3,
        label="existing 1-D riser column",
    )
    ax.plot(
        riser["time_after_ramp_s"],
        riser["riser_column_height_m"],
        color="#111827",
        lw=1.2,
        label="3-D riser column",
    )
    ax.plot(
        riser["time_after_ramp_s"],
        riser["geyser_height_above_lid_m"],
        color="#c81e3c",
        lw=1.0,
        alpha=0.85,
        label="3-D directly resolved wet height",
    )
    ax.axhline(RISER_LENGTH, color="#16a34a", ls=":", label="physical riser rim")
    for event, label in (
        (PAPER["t_jet_out_s"], "paper jet out"),
        (PAPER["t_column_top_s"], "paper column at top"),
    ):
        ax.axvline(event, color="#64748b", ls=":", lw=0.8)
        ax.text(event + 0.015, 0.05, label, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, max(1.4, paper_height * 1.12, of_height * 1.12))
    ax.set_xlabel("time after ramp start [s]")
    ax.set_ylabel("height above chamber lid [m]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    pressure_head = np.linspace(0, 6.5, 100)
    ax.plot(
        pressure_head,
        PAPER["fig7a_slope"] * pressure_head + PAPER["fig7a_intercept_m"],
        "k--",
        lw=1.2,
        label=f"paper Fig. 7(a), $R^2={PAPER['fig7a_r2']:.2f}$",
    )
    ax.scatter(
        PAPER["PT2_peak_kPa"] * 1000 / (RHO_WATER * G),
        paper_height,
        s=70,
        facecolors="none",
        edgecolors="#1d8a4a",
        linewidths=1.6,
        label="B3 paper peak on regression",
    )
    ax.scatter(
        one_pmax * 1000 / (RHO_WATER * G),
        one_height,
        marker="s",
        s=55,
        color="#7c3aed",
        label="existing 1-D ballistic estimate",
    )
    ax.scatter(
        of_pmax * 1000 / (RHO_WATER * G),
        of_height,
        marker="^",
        s=70,
        color="#c81e3c",
        label="3-D resolved wet height",
    )
    ax.axhline(RISER_LENGTH, color="#16a34a", ls=":", lw=1.0)
    ax.set_xlabel(r"$P_{max}/(\rho g)$ [m]")
    ax.set_ylabel("geyser height above chamber lid [m]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"$h=0.6943P_{max}/(\rho g)+0.3086$")
    fig.suptitle("Liu2020 B3 geyser timing and Fig. 7(a) height comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def update_sensitivity(path: Path, row: dict[str, object]) -> None:
    fields = [
        "mesh_label",
        "cells",
        "check_mesh_ok",
        "max_aspect_ratio",
        "max_non_orthogonality_deg",
        "mean_non_orthogonality_deg",
        "max_skewness",
        "underdetermined_tetrahedra",
        "low_interpolation_weight_faces",
        "failed_checks",
        "end_time_after_ramp_s",
        "full_14s_window",
        "PT2_peak_kPa",
        "PT2_min_kPa",
        "PT3_peak_kPa",
        "PT3_min_kPa",
        "maximum_geyser_height_m",
        "geyser",
        "mass_error_percent",
        "max_velocity_limited_cells",
    ]
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    replacement = {key: row.get(key, "") for key in fields}
    updated = False
    for index, old in enumerate(existing):
        if old.get("mesh_label") == row["mesh_label"]:
            existing[index] = {key: str(replacement[key]) for key in fields}
            updated = True
    if not updated:
        existing.append({key: str(replacement[key]) for key in fields})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    args = parse_args()
    case = args.case.resolve()
    OUT.mkdir(exist_ok=True)

    p_probe = read_scalar_probes(case, "probesPT", "p")
    solver_time = p_probe[:, 0]
    report_time = solver_time - RAMP_START
    if p_probe.shape[1] < 5:
        raise RuntimeError(f"Expected four pressure probes, got shape {p_probe.shape}")
    pressure = {
        "PT3": (p_probe[:, 1] - PATM) / 1000,
        "PT2": (p_probe[:, 2] - PATM) / 1000,
        "PT1": (p_probe[:, 3] - PATM) / 1000,
        "PT4": (p_probe[:, 4] - PATM) / 1000,
    }

    alpha_probe = read_scalar_probes(case, "riserCentreline", "alpha.water")
    velocity_probe = read_vector_probes(case, "riserCentreline", "U")
    alpha = np.column_stack(
        [
            interp_nan(solver_time, alpha_probe[:, 0], alpha_probe[:, i + 1])
            for i in range(len(RISER_Z))
        ]
    )
    velocity_flat = np.column_stack(
        [
            interp_nan(solver_time, velocity_probe[:, 0], velocity_probe[:, i + 1])
            for i in range(3 * len(RISER_Z))
        ]
    )
    velocity = velocity_flat.reshape(len(solver_time), len(RISER_Z), 3)

    one_d = load_one_dimensional()
    experiment = {sensor: load_experiment(sensor) for sensor in ("PT1", "PT2", "PT3")}
    pressure_metrics_ = pressure_metrics(
        report_time, {sensor: pressure[sensor] for sensor in ("PT1", "PT2", "PT3")}
    )
    main_peak_index = int(
        np.nanargmax(np.where((report_time >= 0) & (report_time <= 4.5), pressure["PT2"], np.nan))
    )
    periods = oscillation_periods(report_time, pressure["PT2"], main_peak_index)

    pre_ramp = (report_time >= -0.4) & (report_time <= -0.05)
    baseline_pt2 = (
        float(np.mean(pressure["PT2"][pre_ramp])) if np.any(pre_ramp) else pressure["PT2"][0]
    )
    after_ramp = report_time >= 0
    bore_time = first_sustained_crossing(
        report_time[after_ramp],
        pressure["PT2"][after_ramp] - baseline_pt2,
        1.0,
        4,
    )

    riser_path = OUT / "openfoam_3d_riser_series.csv"
    riser_metrics = write_riser_csv(
        riser_path, solver_time, report_time, alpha, velocity
    )
    mass = calculate_mass_balance(case)
    inlet = inlet_flow_check(case)
    mesh = parse_check_mesh(case)
    limiter = parse_velocity_limiter(case)

    paper_height = (
        PAPER["fig7a_slope"]
        * PAPER["PT2_peak_kPa"]
        * 1000
        / (RHO_WATER * G)
        + PAPER["fig7a_intercept_m"]
    )
    one_pt2 = np.asarray(one_d["PT2"])
    one_pt3 = np.asarray(one_d["PT3"])
    one_pt1 = np.asarray(one_d["PT1"])
    end_report_time = float(np.nanmax(report_time))

    metrics = {
        "case": "Liu2020 B3 Q20to100 full-pipe single-shoot geyser",
        "time_origin": "t=0 at start of the inflow ramp (OpenFOAM solver t=2.0 s)",
        "solver": {
            "name": "compressibleInterFoam",
            "version": "OpenFOAM-v2512",
            "reason": (
                "Two compressible immiscible phases are required for the 55 kPa "
                "slam and negative rebound; incompressible interFoam cannot validate "
                "their water-hammer amplitude."
            ),
            "water_equation_of_state": "perfectFluid, rho0=998.2 kg/m3, R=2.2e6 m2/s2",
            "water_sound_speed_m_s": math.sqrt(2.2e6),
            "air_equation_of_state": "perfectGas",
        },
        "run": {
            "mesh_label": args.mesh_label,
            "solver_start_s": float(np.min(solver_time)),
            "solver_end_s": float(np.max(solver_time)),
            "end_time_after_ramp_s": end_report_time,
            "full_14s_post_ramp_window": end_report_time >= 14.4 - 0.01,
            "Q0_settle_duration_s": RAMP_START,
            "ramp_duration_s": RAMP_END - RAMP_START,
        },
        "mesh": mesh,
        "paper": {**PAPER, "B3_regression_height_m": paper_height},
        "one_dimensional": {
            "geyser": bool(one_d["geyser"]),
            "PT1_peak_kPa": float(np.max(one_pt1)),
            "PT1_min_kPa": float(np.min(one_pt1)),
            "PT2_peak_kPa": float(np.max(one_pt2)),
            "PT2_min_kPa": float(np.min(one_pt2)),
            "PT3_peak_kPa": float(np.max(one_pt3)),
            "PT3_min_kPa": float(np.min(one_pt3)),
            "peak_time_s": float(np.asarray(one_d["t"])[int(np.argmax(one_pt2))]),
            "maximum_geyser_height_m": float(one_d["h_jet"]),
            "height_method": (
                "ballistic estimate from the 1-D riser ejection velocity, "
                "measured above the chamber lid"
            ),
            "mass_error_L": float(one_d["mass_error"] * 1000),
        },
        "openfoam_3d": {
            "pressure": pressure_metrics_,
            "bore_arrival_time_s": bore_time,
            "bore_arrival_definition": (
                "first four samples with PT2 >= pre-ramp mean + 1 kPa"
            ),
            "oscillation_periods_s": periods,
            **riser_metrics,
            "mass_balance": mass,
            "inlet_flow": inlet,
            "numerical_safeguards": limiter,
        },
        "relative_error_percent": {
            "PT1_min": relative_error(
                pressure_metrics_["PT1"]["minimum_kPa"], PAPER["PT1_min_kPa"]
            ),
            "PT2_peak": relative_error(
                pressure_metrics_["PT2"]["peak_kPa"], PAPER["PT2_peak_kPa"]
            ),
            "PT2_peak_time": relative_error(
                pressure_metrics_["PT2"]["peak_time_s"], PAPER["t_peak_s"]
            ),
            "PT2_min": relative_error(
                pressure_metrics_["PT2"]["minimum_kPa"], PAPER["PT2_min_kPa"]
            ),
            "PT3_peak": relative_error(
                pressure_metrics_["PT3"]["peak_kPa"], PAPER["PT3_peak_kPa"]
            ),
            "PT3_peak_time": relative_error(
                pressure_metrics_["PT3"]["peak_time_s"], PAPER["t_peak_s"]
            ),
            "PT3_min": relative_error(
                pressure_metrics_["PT3"]["minimum_kPa"], PAPER["PT3_min_kPa"]
            ),
            "bore_arrival_time": relative_error(
                bore_time if bore_time is not None else math.nan,
                PAPER["bore_reach_chamber_s"],
            ),
            "maximum_geyser_height_vs_Fig7_regression": relative_error(
                riser_metrics["maximum_geyser_height_above_lid_m"], paper_height
            ),
            "oscillation_periods": [
                relative_error(value, PAPER["osc_periods_s"][index])
                for index, value in enumerate(periods[: len(PAPER["osc_periods_s"])])
            ],
            "PT1_final_absolute_error_kPa": (
                abs(
                    pressure_metrics_["PT1"]["final_mean_kPa"]
                    - PAPER["PT1_final_kPa"]
                )
                if pressure_metrics_["PT1"]["final_mean_kPa"] is not None
                else None
            ),
            "PT2_final": relative_error(
                pressure_metrics_["PT2"]["final_mean_kPa"]
                if pressure_metrics_["PT2"]["final_mean_kPa"] is not None
                else math.nan,
                PAPER["PT2_final_kPa"],
            ),
            "PT3_final": relative_error(
                pressure_metrics_["PT3"]["final_mean_kPa"]
                if pressure_metrics_["PT3"]["final_mean_kPa"] is not None
                else math.nan,
                PAPER["PT3_final_kPa"],
            ),
        },
        "unresolved_physics_and_input_uncertainty": [
            "Paper PDF/page scans are absent from the required base commit.",
            "Tail-gate opening and receiving-tank level were not reported; H_tail=Dd is the declared minimum full-pipe boundary realisation.",
            "Initial B3 chamber level was not reported; 0.30 m is a declared initialisation, not a fitted datum.",
            "Rigid walls omit PVC pipe-wall compliance because thickness and modulus were not reported.",
            "VOF does not resolve sub-grid dispersed-air entrainment, bubble-size transport, cavitation or phase change.",
            "Exact PT2 in-plane and PT3 horizontal coordinates are not reported; A2 sampling coordinates are retained.",
        ],
    }

    sensitivity_row = {
        "mesh_label": args.mesh_label,
        **mesh,
        "end_time_after_ramp_s": end_report_time,
        "full_14s_window": end_report_time >= 14.4 - 0.01,
        "PT2_peak_kPa": pressure_metrics_["PT2"]["peak_kPa"],
        "PT2_min_kPa": pressure_metrics_["PT2"]["minimum_kPa"],
        "PT3_peak_kPa": pressure_metrics_["PT3"]["peak_kPa"],
        "PT3_min_kPa": pressure_metrics_["PT3"]["minimum_kPa"],
        "maximum_geyser_height_m": riser_metrics["maximum_geyser_height_above_lid_m"],
        "geyser": riser_metrics["geyser"],
        "mass_error_percent": mass["numerical_mass_error_percent"],
        "max_velocity_limited_cells": limiter[
            "maximum_cells_limited_in_one_correction"
        ],
    }
    update_sensitivity(OUT / "openfoam_3d_mesh_sensitivity.csv", sensitivity_row)

    if args.primary:
        write_pressure_csv(
            OUT / "openfoam_3d_pressure_series.csv",
            solver_time,
            report_time,
            pressure,
            one_d,
            experiment,
        )
        (OUT / "openfoam_3d_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        write_pressure_plot(
            OUT / "openfoam_3d_pressure_comparison.png",
            report_time,
            pressure,
            one_d,
            experiment,
            metrics["openfoam_3d"],
        )
        write_height_plot(
            OUT / "openfoam_3d_geyser_height_comparison.png",
            riser_path,
            one_d,
            pressure_metrics_,
            riser_metrics,
        )

    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
