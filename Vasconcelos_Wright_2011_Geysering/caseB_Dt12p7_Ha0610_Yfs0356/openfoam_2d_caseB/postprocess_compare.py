#!/usr/bin/env python3
"""Extract OpenFOAM histories and compare with V&W (2011) Case B."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DIGITIZED = HERE / "digitized"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

P_ATM = 101325.0
RHO_W = 998.2
G = 9.81
PIPE_D = 0.094
DT = 0.0127
L_TOWER = 0.610
CROWN_Y = 0.047
TIME_SCALE = math.sqrt(G * DT) / L_TOWER
PROBE_Y = np.arange(0.049, 0.654 + 0.0025, 0.005)
OPENFOAM_VERSION = "v2512"
MESH_CELLS = 51984


def probe_files(name: str, field: str) -> list[Path]:
    root = HERE / "postProcessing" / name
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(root.glob(f"*/{field}"), key=lambda path: float(path.parent.name))


def read_probe(name: str, field: str) -> np.ndarray:
    chunks = []
    for path in probe_files(name, field):
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size:
            chunks.append(data)
    if not chunks:
        raise RuntimeError(f"No probe data for {name}/{field}")
    data = np.vstack(chunks)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    # A resumed run repeats its checkpoint row. Keep the newest copy.
    _, reverse_index = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - reverse_index)]


def moving_average(time: np.ndarray, value: np.ndarray, width_s: float) -> np.ndarray:
    out = np.full_like(value, np.nan, dtype=float)
    half = width_s / 2.0
    for index, centre in enumerate(time):
        mask = (
            (time >= centre - half)
            & (time <= centre + half)
            & np.isfinite(value)
        )
        if np.any(mask):
            out[index] = np.mean(value[mask])
    return out


def alpha_crossing(
    y0: float, y1: float, alpha0: float, alpha1: float, target: float = 0.5
) -> float:
    if abs(alpha1 - alpha0) < 1e-12:
        return 0.5 * (y0 + y1)
    fraction = np.clip((target - alpha0) / (alpha1 - alpha0), 0.0, 1.0)
    return y0 + fraction * (y1 - y0)


def largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    if not len(starts):
        return None
    lengths = ends - starts + 1
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def extract_levels(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    interface = np.full(alpha.shape[0], np.nan)
    free_surface = np.full(alpha.shape[0], np.nan)
    for row, profile in enumerate(alpha):
        run = largest_true_run(np.asarray(profile) >= 0.5)
        if run is None:
            continue
        first, last = run
        if first == 0:
            lower = CROWN_Y
        else:
            lower = alpha_crossing(
                PROBE_Y[first - 1],
                PROBE_Y[first],
                profile[first - 1],
                profile[first],
            )
        if last == len(profile) - 1:
            upper = CROWN_Y + L_TOWER
        else:
            upper = alpha_crossing(
                PROBE_Y[last],
                PROBE_Y[last + 1],
                profile[last],
                profile[last + 1],
            )
        interface[row] = np.clip((lower - CROWN_Y) / L_TOWER, 0.0, 1.0)
        free_surface[row] = np.clip((upper - CROWN_Y) / L_TOWER, 0.0, 1.0)
    return interface, free_surface


def interp_rmse(
    x_model: np.ndarray,
    y_model: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
) -> float:
    valid_model = np.isfinite(x_model) & np.isfinite(y_model)
    valid_obs = np.isfinite(x_obs) & np.isfinite(y_obs)
    if not np.any(valid_model) or not np.any(valid_obs):
        return float("nan")
    xm = x_model[valid_model]
    ym = y_model[valid_model]
    order = np.argsort(xm)
    xm, ym = xm[order], ym[order]
    valid_obs &= (x_obs >= xm[0]) & (x_obs <= xm[-1])
    if not np.any(valid_obs):
        return float("nan")
    predicted = np.interp(x_obs[valid_obs], xm, ym)
    return float(np.sqrt(np.mean((predicted - y_obs[valid_obs]) ** 2)))


def threshold_time(
    time: np.ndarray,
    value: np.ndarray,
    threshold: float,
    *,
    rising: bool,
    after: float = -np.inf,
) -> float:
    valid = np.isfinite(time) & np.isfinite(value) & (time >= after)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return float("nan")
    condition = value >= threshold if rising else value <= threshold
    hits = indices[condition[indices]]
    if not len(hits):
        return float("nan")
    current = int(hits[0])
    previous_candidates = indices[indices < current]
    if not len(previous_candidates):
        return float(time[current])
    previous = int(previous_candidates[-1])
    dv = value[current] - value[previous]
    if abs(dv) < 1e-12:
        return float(time[current])
    fraction = np.clip((threshold - value[previous]) / dv, 0.0, 1.0)
    return float(time[previous] + fraction * (time[current] - time[previous]))


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def timing_error_seconds(model_tstar: float, experiment_tstar: float) -> float:
    return (model_tstar - experiment_tstar) / TIME_SCALE


def median_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(array)) if np.any(np.isfinite(array)) else float("nan")


def write_series(
    time: np.ndarray,
    tstar: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
) -> None:
    with (OUT / "openfoam_2d_series.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "Tstar", "Hstar_raw", "Hstar_smooth"])
        writer.writerows(zip(time, tstar, raw, smooth))


def write_levels(
    time: np.ndarray,
    tstar: np.ndarray,
    interface: np.ndarray,
    free_surface: np.ndarray,
) -> None:
    with (OUT / "openfoam_2d_levels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "Tstar", "Yint_star", "Yfs_star"])
        writer.writerows(zip(time, tstar, interface, free_surface))


def main() -> None:
    transducer = read_probe("transducer", "p")
    tower = read_probe("towerCentreline", "alpha.water")
    if tower.shape[1] - 1 != len(PROBE_Y):
        raise RuntimeError(
            f"Expected {len(PROBE_Y)} tower probes, found {tower.shape[1] - 1}"
        )

    time = transducer[:, 0]
    pressure = transducer[:, 1]
    tower_time = tower[:, 0]
    alpha = tower[:, 1:]

    hstar_raw = (pressure - P_ATM) / (RHO_W * G * L_TOWER)
    hstar = moving_average(time, hstar_raw, 0.10)
    tstar = time * TIME_SCALE
    tower_tstar = tower_time * TIME_SCALE
    yint, yfs = extract_levels(alpha)

    pressure_exp = np.genfromtxt(
        DIGITIZED / "fig6_caseB_pressure_envelope.csv",
        delimiter=",",
        names=True,
    )
    levels_exp = np.genfromtxt(
        DIGITIZED / "fig8_caseB_levels.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    fs_mask = levels_exp["kind"] == "fs"
    int_mask = levels_exp["kind"] == "int"

    pressure_rmse = interp_rmse(
        tstar, hstar, pressure_exp["Tstar"], pressure_exp["Hstar_med"]
    )
    fs_rmse = interp_rmse(
        tower_tstar,
        yfs,
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
    )
    int_rmse = interp_rmse(
        tower_tstar,
        yint,
        levels_exp["Tstar"][int_mask],
        levels_exp["Ystar"][int_mask],
    )

    exp_release = threshold_time(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
        0.4,
        rising=False,
        after=3.5,
    )
    model_release = threshold_time(
        tstar, hstar, 0.4, rising=False, after=3.5
    )
    exp_geyser = threshold_time(
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
        0.99,
        rising=True,
        after=3.0,
    )
    model_geyser = threshold_time(
        tower_tstar, yfs, 0.99, rising=True, after=3.0
    )

    interface_85_by_run = []
    for run in np.unique(levels_exp["run"][int_mask]):
        run_mask = int_mask & (levels_exp["run"] == run)
        interface_85_by_run.append(
            threshold_time(
                levels_exp["Tstar"][run_mask],
                levels_exp["Ystar"][run_mask],
                0.85,
                rising=True,
                after=3.0,
            )
        )
    exp_interface_85 = median_or_nan(interface_85_by_run)
    model_interface_85 = threshold_time(
        tower_tstar, yint, 0.85, rising=True, after=3.0
    )

    exp_plateau_mask = (
        (pressure_exp["Tstar"] >= 1.0)
        & (pressure_exp["Tstar"] <= 3.5)
    )
    model_plateau_mask = (tstar >= 1.0) & (tstar <= 3.5)
    exp_plateau = float(np.mean(pressure_exp["Hstar_med"][exp_plateau_mask]))
    model_plateau = (
        float(np.mean(hstar[model_plateau_mask]))
        if np.any(model_plateau_mask)
        else float("nan")
    )

    metrics = {
        "case": "Vasconcelos_Wright_2011_Case_B_Dt12p7_Ha0610_Yfs0356",
        "openfoam_version": OPENFOAM_VERSION,
        "solver": "compressibleInterFoam",
        "mesh_cells": MESH_CELLS,
        "simulation_end_s": float(time[-1]),
        "simulation_end_Tstar": float(tstar[-1]),
        "sample_counts": {
            "model_pressure": int(len(time)),
            "model_tower_profiles": int(len(tower_time)),
            "tower_probes_per_profile": int(alpha.shape[1]),
            "experimental_pressure_trace": int(len(pressure_exp)),
            "experimental_free_surface": int(np.count_nonzero(fs_mask)),
            "experimental_interface": int(np.count_nonzero(int_mask)),
        },
        "definitions": {
            "Tstar": "t*sqrt(g*Dt)/L",
            "Hstar": "(p-p_atm)/(rho_w*g*L)",
            "Ystar": "(y-pipe_crown_y)/L",
            "time_shift_applied": False,
            "pressure_smoothing_s": 0.10,
            "level_extraction": (
                "alpha.water=0.5 crossings around the largest contiguous "
                "water segment on the tower centreline"
            ),
        },
        "pressure": {
            "experimental_plateau_Hstar_mean_T1to3p5": exp_plateau,
            "model_plateau_Hstar_mean_T1to3p5": finite_or_none(model_plateau),
            "plateau_bias_Hstar": finite_or_none(model_plateau - exp_plateau),
            "RMSE_Hstar_no_shift": finite_or_none(pressure_rmse),
            "experimental_release_Tstar_at_Hstar_0p4": finite_or_none(exp_release),
            "model_release_Tstar_at_Hstar_0p4": finite_or_none(model_release),
            "model_release_observed": bool(np.isfinite(model_release)),
            "release_timing_error_Tstar": finite_or_none(
                model_release - exp_release
            ),
            "release_timing_error_s": finite_or_none(
                timing_error_seconds(model_release, exp_release)
            ),
            "release_delay_lower_bound_Tstar_if_censored": (
                finite_or_none(tstar[-1] - exp_release)
                if not np.isfinite(model_release)
                else None
            ),
            "release_delay_lower_bound_s_if_censored": (
                finite_or_none((tstar[-1] - exp_release) / TIME_SCALE)
                if not np.isfinite(model_release)
                else None
            ),
        },
        "levels": {
            "experimental_geyser_onset_Tstar_at_Yfs_0p99": finite_or_none(
                exp_geyser
            ),
            "model_geyser_onset_Tstar_at_Yfs_0p99": finite_or_none(model_geyser),
            "model_geyser_onset_observed": bool(np.isfinite(model_geyser)),
            "geyser_onset_timing_error_Tstar": finite_or_none(
                model_geyser - exp_geyser
            ),
            "geyser_onset_timing_error_s": finite_or_none(
                timing_error_seconds(model_geyser, exp_geyser)
            ),
            "geyser_delay_lower_bound_Tstar_if_censored": (
                finite_or_none(tower_tstar[-1] - exp_geyser)
                if not np.isfinite(model_geyser)
                else None
            ),
            "geyser_delay_lower_bound_s_if_censored": (
                finite_or_none((tower_tstar[-1] - exp_geyser) / TIME_SCALE)
                if not np.isfinite(model_geyser)
                else None
            ),
            "experimental_interface_Tstar_at_Yint_0p85_median": finite_or_none(
                exp_interface_85
            ),
            "model_interface_Tstar_at_Yint_0p85": finite_or_none(
                model_interface_85
            ),
            "model_interface_0p85_observed": bool(
                np.isfinite(model_interface_85)
            ),
            "interface_0p85_timing_error_Tstar": finite_or_none(
                model_interface_85 - exp_interface_85
            ),
            "interface_0p85_timing_error_s": finite_or_none(
                timing_error_seconds(model_interface_85, exp_interface_85)
            ),
            "interface_0p85_delay_lower_bound_Tstar_if_censored": (
                finite_or_none(tower_tstar[-1] - exp_interface_85)
                if not np.isfinite(model_interface_85)
                else None
            ),
            "interface_0p85_delay_lower_bound_s_if_censored": (
                finite_or_none(
                    (tower_tstar[-1] - exp_interface_85) / TIME_SCALE
                )
                if not np.isfinite(model_interface_85)
                else None
            ),
            "model_interface_max_Ystar": finite_or_none(np.nanmax(yint)),
            "model_free_surface_max_Ystar": finite_or_none(np.nanmax(yfs)),
            "free_surface_RMSE_Ystar_no_shift": finite_or_none(fs_rmse),
            "interface_RMSE_Ystar_no_shift": finite_or_none(int_rmse),
            "experimental_geysering": True,
            "model_geysering": bool(np.nanmax(yfs) >= 0.99),
            "model_geysering_criterion": "tower free surface reaches Ystar>=0.99",
        },
        "geometry_ratios": {
            "planar_2d_Dt_over_D": DT / PIPE_D,
            "physical_circular_area_ratio": (DT / PIPE_D) ** 2,
        },
        "digitization_uncertainty": {
            "figure8_Tstar": 4.0 / (475.0 - 263.0),
            "figure8_Ystar": 2.0 / (306.0 - 173.0),
            "note": "Approximate +/-2 pixel marker-centre uncertainty.",
        },
        "limitations": [
            "A constant-depth planar model preserves diameters but not the circular pipe/tower area ratio.",
            "The open tower ends at the rim, so an above-rim jet is not represented.",
            "The laminar 2-D VOF model cannot resolve the experimental three-dimensional annular wall film.",
            "No parameter was tuned and no event-time shift was applied.",
        ],
    }

    write_series(time, tstar, hstar_raw, hstar)
    write_levels(tower_time, tower_tstar, yint, yfs)
    (OUT / "openfoam_2d_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    figure, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.fill_between(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_min"],
        pressure_exp["Hstar_max"],
        color="0.84",
        label="Experiment: three-run raster envelope",
    )
    axis.plot(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
        color="0.15",
        lw=1.4,
        label="Experiment: digitized median",
    )
    axis.plot(
        tstar,
        hstar_raw,
        color="#4c78a8",
        lw=0.45,
        alpha=0.35,
        label="OpenFOAM raw",
    )
    axis.plot(
        tstar,
        hstar,
        color="#c62828",
        lw=1.5,
        label="OpenFOAM 0.10 s mean",
    )
    axis.set(
        xlabel=r"$T^*=t\sqrt{gD_t}/L$",
        ylabel=r"$H^*=(p-p_{atm})/(\rho_w gL)$",
        xlim=(0, 5.0),
            ylim=(-0.05, 1.50),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "openfoam_2d_pressure_comparison.png", dpi=300)
    figure.savefig(OUT / "openfoam_2d_pressure_comparison.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.errorbar(
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
        xerr=levels_exp["Tstar_uncertainty"][fs_mask],
        yerr=levels_exp["Ystar_uncertainty"][fs_mask],
        fmt="^",
        mfc="none",
        mec="0.25",
        ecolor="0.75",
        ms=4,
        lw=0.5,
        label="Experiment $Y^*_{fs}$ (overlap centreline)",
    )
    markers = {"1": "s", "2": "o", "3": "D"}
    for run in np.unique(levels_exp["run"][int_mask]):
        run_mask = int_mask & (levels_exp["run"] == run)
        axis.scatter(
            levels_exp["Tstar"][run_mask],
            levels_exp["Ystar"][run_mask],
            marker=markers.get(str(run), "o"),
            facecolors="none",
            edgecolors="0.35",
            s=20,
            label=f"Experiment $Y^*_{{int}}$, run {run}",
        )
    axis.plot(
        tower_tstar,
        yfs,
        color="#c62828",
        lw=1.5,
        label="OpenFOAM $Y^*_{fs}$",
    )
    axis.plot(
        tower_tstar,
        yint,
        color="#c62828",
        lw=1.5,
        ls="--",
        label="OpenFOAM $Y^*_{int}$",
    )
    axis.set(
        xlabel=r"$T^*=t\sqrt{gD_t}/L$",
        ylabel=r"$Y^*=(y-y_{crown})/L$",
        xlim=(3.0, 4.5),
        ylim=(-0.02, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    figure.tight_layout()
    figure.savefig(OUT / "openfoam_2d_levels_comparison.png", dpi=300)
    figure.savefig(OUT / "openfoam_2d_levels_comparison.pdf")
    plt.close(figure)

    figure, (heat_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(9.0, 4.5), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    ystar_probes = (PROBE_Y - CROWN_Y) / L_TOWER
    image = heat_axis.pcolormesh(
        tower_tstar,
        ystar_probes,
        alpha.T,
        shading="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    heat_axis.set(
        xlabel=r"$T^*$",
        ylabel=r"$Y^*$",
        xlim=(0, min(5.2, tower_tstar[-1])),
        ylim=(0, 1),
        title=r"Tower centreline $\alpha_{water}$",
    )
    figure.colorbar(image, ax=heat_axis, label=r"$\alpha_{water}$")

    targets = [0.0, 3.5, 3.9, 4.2, 5.0]
    for target in targets:
        index = int(np.argmin(np.abs(tower_tstar - target)))
        profile_axis.plot(
            alpha[index],
            ystar_probes,
            lw=1.2,
            label=rf"$T^*={tower_tstar[index]:.2f}$",
        )
    profile_axis.set(
        xlabel=r"$\alpha_{water}$",
        ylabel=r"$Y^*$",
        xlim=(-0.03, 1.03),
        ylim=(0, 1),
        title="Selected phase profiles",
    )
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "openfoam_2d_phase_snapshots.png", dpi=300)
    figure.savefig(OUT / "openfoam_2d_phase_snapshots.pdf")
    plt.close(figure)

    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
