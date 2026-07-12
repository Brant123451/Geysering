#!/usr/bin/env python3
"""Reduce OpenFOAM function-object output to compact validation artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
CFG = json.loads((HERE / "case_config.json").read_text(encoding="utf-8"))
FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="single runs/<mesh>_<valve> directory id")
    return parser.parse_args()


def numeric_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(v) for v in FLOAT.findall(line)]
        if values:
            rows.append(values)
    if not rows:
        return np.empty((0, 0))
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=float)


def object_table(run: Path, name: str) -> np.ndarray:
    files = sorted((run / "postProcessing" / name).glob("**/*.dat"))
    chunks = [numeric_rows(path) for path in files]
    chunks = [chunk for chunk in chunks if chunk.size and chunk.shape[1] >= 2]
    if not chunks:
        return np.empty((0, 0))
    data = np.vstack(chunks)
    data = data[np.argsort(data[:, 0], kind="stable")]
    # For restarted runs, keep the latest occurrence of each time.
    _, reverse_index = np.unique(data[::-1, 0], return_index=True)
    keep = np.sort(len(data) - 1 - reverse_index)
    return data[keep]


def probe_table(run: Path, name: str, field: str) -> np.ndarray:
    files = sorted((run / "postProcessing" / name).glob(f"**/{field}"))
    chunks = [numeric_rows(path) for path in files]
    chunks = [chunk for chunk in chunks if chunk.size and chunk.shape[1] >= 2]
    if not chunks:
        return np.empty((0, 0))
    data = np.vstack(chunks)
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, reverse_index = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - reverse_index)]


def interp(
    table: np.ndarray,
    times: np.ndarray,
    column: int,
    default: float = math.nan,
    *,
    finite_only: bool = False,
) -> np.ndarray:
    if table.size == 0 or table.shape[1] <= column:
        return np.full_like(times, default, dtype=float)
    source_t = table[:, 0]
    source_y = table[:, column]
    if not finite_only:
        return np.interp(times, source_t, source_y)
    valid = np.isfinite(source_t) & np.isfinite(source_y)
    if not np.any(valid):
        return np.full_like(times, default, dtype=float)
    source_t = source_t[valid]
    source_y = source_y[valid]
    output = np.full_like(times, default, dtype=float)
    inside = (times >= source_t[0]) & (times <= source_t[-1])
    output[inside] = np.interp(times[inside], source_t, source_y)
    return output


def alpha_crossings(
    z: np.ndarray, alpha: np.ndarray
) -> tuple[list[float], list[float]]:
    water_to_air: list[float] = []
    air_to_water: list[float] = []
    for i in range(len(z) - 1):
        a0, a1 = alpha[i], alpha[i + 1]
        if not np.isfinite(a0 + a1) or (a0 - 0.5) * (a1 - 0.5) > 0:
            continue
        if a1 == a0:
            continue
        f = (0.5 - a0) / (a1 - a0)
        crossing = float(z[i] + f * (z[i + 1] - z[i]))
        if a1 < a0:
            water_to_air.append(crossing)
        else:
            air_to_water.append(crossing)
    return water_to_air, air_to_water


def level_series(run: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    root = run / "postProcessing" / "riserLine"
    for path in sorted(root.glob("**/centreline*")):
        try:
            time = float(path.parent.name)
        except ValueError:
            continue
        data = numeric_rows(path)
        if data.size == 0 or data.shape[1] < 2:
            continue
        order = np.argsort(data[:, 0])
        distance = data[order, 0]
        alpha = data[order, 1]
        z = 0.026 + distance
        water_to_air, air_to_water = alpha_crossings(z, alpha)
        # Yfs is the highest water-to-air transition, or the physical rim
        # when the centreline is still water-wet at its upper endpoint.
        yfs = (
            1.8
            if alpha[-1] >= 0.5
            else (
                max(water_to_air) - 0.025
                if water_to_air
                else math.nan
            )
        )
        # The pocket nose is an upward air-to-water transition.  Direction,
        # rather than simply the lowest crossing, avoids treating the initial
        # free surface as an air-pocket front.
        yint = min(air_to_water) - 0.025 if air_to_water else math.nan
        rows.append((time, yfs, yint))
    if not rows:
        return np.empty((0, 3))
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, reverse_index = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - reverse_index)]


def cumulative_integral(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    if len(t) > 1:
        out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(t))
    return out


def first_crossing(
    t: np.ndarray,
    y: np.ndarray,
    threshold: float,
    *,
    missing_previous: float | None = None,
) -> float | None:
    valid = np.isfinite(y)
    indices = np.flatnonzero(valid & (y >= threshold))
    if len(indices) == 0:
        return None
    i = int(indices[0])
    if i == 0:
        return float(t[i])
    previous = y[i - 1]
    if not np.isfinite(previous):
        if missing_previous is None:
            return float(t[i])
        previous = missing_previous
    if y[i] == previous:
        return float(t[i])
    f = float(np.clip((threshold - previous) / (y[i] - previous), 0.0, 1.0))
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def max_climb_rate(
    t: np.ndarray,
    y: np.ndarray,
    start: float | None,
    stop: float | None,
    window: float = 0.6,
) -> float | None:
    if start is None or stop is None or stop < start:
        return None
    mask = (t >= start) & (t <= stop) & np.isfinite(y)
    selected_t = t[mask]
    selected_y = y[mask]
    if len(selected_t) < 4:
        return None
    best: float | None = None
    first = 0
    for last in range(len(selected_t)):
        while selected_t[last] - selected_t[first] > window:
            first += 1
        if last - first >= 2 and selected_t[last] > selected_t[first]:
            slope = float(
                (selected_y[last] - selected_y[first])
                / (selected_t[last] - selected_t[first])
            )
            best = slope if best is None else max(best, slope)
    return best


def time_of_finite_max(t: np.ndarray, values: np.ndarray) -> float | None:
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return None
    index = int(valid[np.argmax(values[valid])])
    return float(t[index])


def finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else None


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def reduce_run(run_id: str) -> dict:
    run = HERE / "runs" / run_id
    if not run.is_dir():
        raise FileNotFoundError(run)
    mesh_name, valve_name = run_id.split("_", 1)

    pressure = probe_table(run, "pressureProbes", "p")
    levels = level_series(run)
    total = object_table(run, "totalMass")
    pocket = object_table(run, "pocketIntegrals")
    pocket_p = object_table(run, "pocketPressure")
    riser = object_table(run, "riserIntegrals")
    external = object_table(run, "externalWater")
    inlet = object_table(run, "inletFlux")
    inlet_water = object_table(run, "inletWaterFlux")
    atmosphere = object_table(run, "atmosphereFlux")
    atmosphere_water = object_table(run, "atmosphereWaterFlux")
    if valve_name != "closed" and levels.size == 0:
        raise RuntimeError(f"Missing riserLine output in event run {run}")

    candidates = [table[:, 0] for table in (pressure, levels, total) if table.size]
    if not candidates:
        raise RuntimeError(f"No function-object time series found in {run}")
    t = max(candidates, key=len)

    p_pt1 = interp(pressure, t, 1)
    p_pt2 = interp(pressure, t, 2)
    yfs = interp(levels, t, 1, finite_only=True)
    yint = interp(levels, t, 2, finite_only=True)
    total_mass = interp(total, t, 1)
    water_volume = interp(total, t, 2)
    pocket_mass = interp(pocket, t, 1)
    pocket_water = interp(pocket, t, 2)
    p_pocket = interp(pocket_p, t, 1)
    riser_mass = interp(riser, t, 1)
    riser_water = interp(riser, t, 2)
    external_water = interp(external, t, 1, 0.0)
    q_in = interp(inlet, t, 1, 0.0)
    mdot_in = interp(inlet, t, 2, 0.0)
    q_water_in = interp(inlet_water, t, 1, 0.0)
    q_atm = interp(atmosphere, t, 1, 0.0)
    mdot_atm = interp(atmosphere, t, 2, 0.0)
    q_water_atm = interp(atmosphere_water, t, 1, 0.0)

    rho_w = 998.0
    p_atm = 101325.0
    g = 9.81
    h0 = 0.66
    water_mass = rho_w * water_volume
    gas_mass = total_mass - water_mass
    pocket_gas_mass = pocket_mass - rho_w * pocket_water
    riser_gas_mass = riser_mass - rho_w * riser_water

    net_mdot_out = mdot_in + mdot_atm
    net_water_mdot_out = rho_w * (q_water_in + q_water_atm)
    net_gas_mdot_out = net_mdot_out - net_water_mdot_out
    mass_closure = total_mass - total_mass[0] + cumulative_integral(t, net_mdot_out)
    water_closure = water_mass - water_mass[0] + cumulative_integral(t, net_water_mdot_out)
    gas_closure = gas_mass - gas_mass[0] + cumulative_integral(t, net_gas_mdot_out)
    # Conservation over the external cell zone: water that crossed the
    # physical rim is either still in that zone or has crossed its far-field
    # atmosphere patch.  The signed patch integral handles possible re-entry.
    escaped_water_net = cumulative_integral(t, q_water_atm)
    eject_volume = np.maximum(
        external_water - external_water[0] + escaped_water_net, 0.0
    )

    level_t = levels[:, 0] if levels.size else np.empty(0)
    level_yfs = levels[:, 1] if levels.size else np.empty(0)
    level_yint = levels[:, 2] if levels.size else np.empty(0)
    ta = first_crossing(
        level_t, level_yint, 0.02, missing_previous=0.0
    )
    t_rim = first_crossing(level_t, level_yfs, 0.98 * 1.8)
    geyser = t_rim is not None
    velocity_stop = (
        t_rim
        if t_rim is not None
        else time_of_finite_max(level_t, level_yint)
    )
    vfs = max_climb_rate(level_t, level_yfs, ta, velocity_stop)
    vint = max_climb_rate(level_t, level_yint, ta, velocity_stop)

    head_pt1 = (p_pt1 - p_atm) / (rho_w * g)
    head_pt2 = (p_pt2 - p_atm) / (rho_w * g)
    head_pocket = (p_pocket - p_atm) / (rho_w * g)

    series_path = RESULTS / f"openfoam_{run_id}_series.csv"
    write_csv(
        series_path,
        (
            "t_s",
            "Yfs_m",
            "Yint_m",
            "p_pt1_Pa",
            "p_pt1_head_m",
            "p_pt2_Pa",
            "p_pt2_head_m",
            "p_pocket_Pa",
            "p_pocket_head_m",
            "Q_in_m3_s",
            "Q_water_in_m3_s",
            "Q_atmosphere_m3_s",
            "Q_water_atmosphere_m3_s",
            "V_ejected_m3",
            "m_water_domain_kg",
            "m_gas_domain_kg",
            "m_gas_pocket_kg",
            "m_gas_riser_kg",
            "mass_closure_kg",
            "water_closure_kg",
            "gas_closure_kg",
        ),
        zip(
            t,
            yfs,
            yint,
            p_pt1,
            head_pt1,
            p_pt2,
            head_pt2,
            p_pocket,
            head_pocket,
            q_in,
            q_water_in,
            q_atm,
            q_water_atm,
            eject_volume,
            water_mass,
            gas_mass,
            pocket_gas_mass,
            riser_gas_mass,
            mass_closure,
            water_closure,
            gas_closure,
        ),
    )

    mesh_stats = json.loads((run / "mesh_stats.json").read_text(encoding="utf-8"))
    initial_audit = json.loads((run / "initial_audit.json").read_text(encoding="utf-8"))
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "mesh": mesh_name,
        "valve": valve_name,
        "valve_opening_s": {"baseline": 0.2, "instant": 0.0, "closed": None}[valve_name],
        "end_time_s": float(t[-1]),
        "mesh_cells": mesh_stats["cells_3d"],
        "geyser_model": geyser,
        "Ta_s": ta,
        "t_rim_s": t_rim,
        "Yfs_max_m": finite_max(yfs),
        "Yint_max_m": finite_max(yint),
        "vfs_m_s": vfs,
        "vint_m_s": vint,
        "velocity_metric": {
            "method": "maximum rolling climb rate",
            "window_s": 0.6,
            "start_s": ta,
            "stop_s": velocity_stop,
        },
        "p_pt1_peak_over_H0": finite_max(head_pt1 / h0),
        "p_pocket_peak_over_H0": finite_max(head_pocket / h0),
        "ejected_water_final_m3": float(eject_volume[-1]),
        "mass_conservation": {
            "max_abs_total_error_kg": finite_max(np.abs(mass_closure)),
            "max_abs_water_error_kg": finite_max(np.abs(water_closure)),
            "max_abs_gas_error_kg": finite_max(np.abs(gas_closure)),
            "max_abs_total_relative_initial": finite_max(
                np.abs(mass_closure) / max(abs(total_mass[0]), 1e-30)
            ),
            "max_abs_gas_relative_initial": finite_max(
                np.abs(gas_closure) / max(abs(gas_mass[0]), 1e-30)
            ),
        },
        "closed_hold": {
            "max_abs_Yfs_drift_m": finite_max(np.abs(yfs - yfs[0])),
            "max_abs_PT1_drift_Pa": finite_max(np.abs(p_pt1 - p_pt1[0])),
            "max_abs_pocket_mass_drift_kg": finite_max(
                np.abs(pocket_gas_mass - pocket_gas_mass[0])
            ),
        }
        if valve_name == "closed"
        else None,
        "initial_audit": initial_audit,
        "series_csv": series_path.name,
        "notes": [
            "No experimental classification was used as a model input.",
            "PT1 millimetre offset is unreported; pocket gas-weighted pressure is also provided.",
            "Yfs uses upward water-to-air and Yint upward air-to-water alpha.water=0.5 centreline crossings.",
            "Ejected volume is external-domain water inventory plus signed cumulative far-field water outflow.",
            "Water density is exactly constant in this configured model, so total minus water mass gives gas mass.",
        ],
    }
    metrics_path = RESULTS / f"openfoam_{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def read_experiment() -> dict:
    path = CASE_ROOT / "data" / "series_b_measurement.csv"
    with path.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    return {
        "run": row["run"],
        "Dr_mm": float(row["Dr_mm"]),
        "L0_m": float(row["L0_m"]),
        "H0_m": float(row["H0_m"]),
        "Ta_s": float(row["Ta_meas_s"]),
        "vfs_m_s": float(row["vfs_meas"]),
        "vint_m_s": float(row["vint_meas"]),
        "geyser": bool(int(row["geyser_meas"])),
    }


def read_1d() -> dict:
    path = CASE_ROOT / "outputs" / "series_b_model_summary.csv"
    with path.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))

    def number(name: str) -> float | None:
        text = row.get(name, "")
        return float(text) if text not in ("", None) else None

    return {
        "geometry": "legacy 6.0 m main, tee x=2.88 m",
        "geyser": bool(int(row["geyser_model"])),
        "Ta_s": number("Ta_model_s"),
        "Yfs_max_m": number("Yfs_max_m"),
        "Yint_max_m": number("Yint_max_m"),
        "vfs_m_s": number("v_fs_model"),
        "vint_m_s": number("v_int_model"),
        "pocket_peak_over_H0": number("pocket_peak_over_H0"),
        "match": row["match"],
        "source": str(path.relative_to(CASE_ROOT)),
    }


def load_metrics() -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("openfoam_*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics[data["run_id"]] = data
    return metrics


def plot_outputs(metrics: dict[str, dict]) -> None:
    if not metrics:
        return
    series: dict[str, np.ndarray] = {}
    for run_id in metrics:
        path = RESULTS / f"openfoam_{run_id}_series.csv"
        series[run_id] = np.genfromtxt(path, delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for run_id, data in series.items():
        if run_id.endswith("_closed"):
            continue
        ax.plot(data["t_s"], data["Yfs_m"], label=f"{run_id} Yfs")
        ax.plot(data["t_s"], data["Yint_m"], "--", label=f"{run_id} Yint")
    ax.axhline(1.8, color="k", lw=0.8, label="physical rim")
    ax.set(xlabel="t (s)", ylabel="height above main soffit (m)", xlim=(0, 13))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_levels.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for run_id, data in series.items():
        if run_id.endswith("_closed"):
            continue
        ax.plot(
            data["t_s"],
            data["p_pocket_head_m"] / 0.66,
            label=f"{run_id} pocket",
        )
        ax.plot(
            data["t_s"],
            data["p_pt1_head_m"] / 0.66,
            "--",
            alpha=0.75,
            label=f"{run_id} PT1",
        )
    ax.set(xlabel="t (s)", ylabel="gauge pressure head / H0", xlim=(0, 13))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_pressure.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.3), sharex=True)
    for run_id, data in series.items():
        if run_id.endswith("_closed"):
            continue
        axes[0].plot(data["t_s"], data["Q_water_in_m3_s"], label=f"{run_id} inlet")
        axes[0].plot(
            data["t_s"], data["Q_water_atmosphere_m3_s"], "--", label=f"{run_id} far field"
        )
        axes[1].plot(data["t_s"], data["V_ejected_m3"] * 1000, label=run_id)
    axes[0].set_ylabel("water flow (m3/s)")
    axes[1].set_ylabel("ejected water (L)")
    axes[1].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_flow_ejection.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for run_id, data in series.items():
        ax.plot(data["t_s"], data["gas_closure_kg"], label=f"{run_id} gas")
        ax.plot(
            data["t_s"], data["water_closure_kg"], "--", label=f"{run_id} water"
        )
    ax.set(xlabel="t (s)", ylabel="mass closure residual (kg)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "mass_conservation.png", dpi=160)
    plt.close(fig)

    production = [
        run_id
        for run_id in ("base_baseline", "refined_baseline", "base_instant")
        if run_id in metrics
    ]
    if production:
        labels = production
        ta = [metrics[k]["Ta_s"] if metrics[k]["Ta_s"] is not None else np.nan for k in labels]
        ymax = [metrics[k]["Yfs_max_m"] for k in labels]
        pmax = [metrics[k]["p_pocket_peak_over_H0"] for k in labels]
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.8))
        for ax, values, title in zip(
            axes, (ta, ymax, pmax), ("Ta (s)", "max Yfs (m)", "pocket peak/H0")
        ):
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / "variant_sensitivity.png", dpi=160)
        plt.close(fig)

    experiment = read_experiment()
    model_1d = read_1d()
    base = metrics.get("base_baseline")
    labels = ["experiment", "legacy 1-D"]
    records: list[tuple[dict, str]] = [
        (experiment, "geyser"),
        (model_1d, "geyser"),
    ]
    if base:
        labels.append("3-D base")
        records.append((base, "geyser_model"))
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.7))
    values_by_panel = (
        [float(record[key]) for record, key in records],
        [
            record.get("Ta_s", np.nan)
            if record.get("Ta_s") is not None
            else np.nan
            for record, _ in records
        ],
        [
            record.get("vfs_m_s", np.nan)
            if record.get("vfs_m_s") is not None
            else np.nan
            for record, _ in records
        ],
        [
            record.get("vint_m_s", np.nan)
            if record.get("vint_m_s") is not None
            else np.nan
            for record, _ in records
        ],
    )
    for ax, values, title in zip(
        axes,
        values_by_panel,
        ("geyser", "Ta (s)", "vfs (m/s)", "vint (m/s)"),
    ):
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_yticks((0, 1), ("no", "yes"))
    fig.tight_layout()
    fig.savefig(RESULTS / "experiment_1d_3d.png", dpi=160)
    plt.close(fig)


def aggregate() -> None:
    metrics = load_metrics()
    experiment = read_experiment()
    model_1d = read_1d()
    base = metrics.get("base_baseline")
    refined = metrics.get("refined_baseline")
    instant = metrics.get("base_instant")

    def delta(a: dict | None, b: dict | None, key: str) -> float | None:
        if not a or not b or a.get(key) is None or b.get(key) is None:
            return None
        return float(a[key] - b[key])

    def compare_with_experiment(
        model: dict | None, geyser_key: str
    ) -> dict:
        def error(key: str) -> float | None:
            if not model or model.get(key) is None:
                return None
            return float(model[key] - experiment[key])

        return {
            "available": model is not None,
            "geyser_match": (
                bool(model[geyser_key]) == experiment["geyser"]
                if model and model.get(geyser_key) is not None
                else None
            ),
            "Ta_delta_s": error("Ta_s"),
            "vfs_delta_m_s": error("vfs_m_s"),
            "vint_delta_m_s": error("vint_m_s"),
        }

    comparison = {
        "schema_version": 1,
        "case": {
            "run": "B-H2",
            "geometry_3d": "paper-audited 6.59 m main, tee x=3.47 m",
            "geometry_1d": model_1d["geometry"],
            "solver_3d": "OpenFOAM v2512 compressibleInterIsoFoam",
        },
        "experiment": experiment,
        "model_1d": model_1d,
        "openfoam_3d": metrics,
        "experiment_comparison": {
            "legacy_1d": compare_with_experiment(model_1d, "geyser"),
            "base_3d": compare_with_experiment(base, "geyser_model"),
        },
        "sensitivity": {
            "refined_minus_base_Ta_s": delta(refined, base, "Ta_s"),
            "refined_minus_base_Yfs_max_m": delta(refined, base, "Yfs_max_m"),
            "refined_minus_base_pocket_peak_over_H0": delta(
                refined, base, "p_pocket_peak_over_H0"
            ),
            "baseline_minus_instant_Ta_s": delta(base, instant, "Ta_s"),
            "baseline_minus_instant_Yfs_max_m": delta(base, instant, "Yfs_max_m"),
            "classification_flip_mesh": (
                base["geyser_model"] != refined["geyser_model"]
                if base and refined
                else None
            ),
            "classification_flip_valve": (
                base["geyser_model"] != instant["geyser_model"]
                if base and instant
                else None
            ),
        },
        "comparison_limits": [
            "B-H2 has no Case-owned digitized level or pressure history; scalar Table 2 errors only.",
            "The legacy 1-D and audited 3-D geometries differ and are not time-shifted.",
            "The reported experimental classification was not used to tune the 3-D model.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison_metrics.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_outputs(metrics)


def main() -> None:
    cli = arguments()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cli.run:
        reduce_run(cli.run)
    aggregate()


if __name__ == "__main__":
    main()
