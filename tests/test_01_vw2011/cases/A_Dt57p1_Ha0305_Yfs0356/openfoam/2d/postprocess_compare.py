"""Extract OpenFOAM probe histories and compare them with V&W2011 Case A."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CASE_A = HERE.parents[1]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

P_ATM = 101325.0
RHO_W = 998.2
G = 9.81
DT = 0.0571
L_TOWER = 0.610
CROWN_Y = 0.047
TRANSDUCER_Y = -0.043
TIME_SCALE = math.sqrt(G * DT) / L_TOWER
PROBE_Y = np.arange(0.052, 0.653, 0.010)
PRESSURE_DATUM_CORRECTION_HSTAR = (CROWN_Y - TRANSDUCER_Y) / L_TOWER
PRESSURE_PLATEAU_TARGET = 0.54
FREE_SURFACE_TARGET = 0.63
INTERFACE_VELOCITY_TARGET = 0.39


def probe_files(name: str, field: str) -> list[Path]:
    root = HERE / "postProcessing" / name
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        root.glob(f"*/{field}"),
        key=lambda p: float(p.parent.name),
    )


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
    # Keep the newest row when a resumed run repeats a checkpoint time.
    _, rev_idx = np.unique(data[::-1, 0], return_index=True)
    return data[np.sort(len(data) - 1 - rev_idx)]


def moving_average(time: np.ndarray, value: np.ndarray, width_s: float) -> np.ndarray:
    out = np.full_like(value, np.nan, dtype=float)
    half = 0.5 * width_s
    for i, ti in enumerate(time):
        mask = (time >= ti - half) & (time <= ti + half) & np.isfinite(value)
        if np.any(mask):
            out[i] = np.mean(value[mask])
    return out


def crossing(y0: float, y1: float, a0: float, a1: float, target: float = 0.5) -> float:
    if abs(a1 - a0) < 1e-12:
        return 0.5 * (y0 + y1)
    f = np.clip((target - a0) / (a1 - a0), 0.0, 1.0)
    return y0 + f * (y1 - y0)


def extract_levels(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yint = np.zeros(alpha.shape[0])
    yfs = np.zeros(alpha.shape[0])
    for row, profile in enumerate(alpha):
        wet = profile >= 0.5
        if not np.any(wet):
            yint[row] = np.nan
            yfs[row] = np.nan
            continue

        first = int(np.argmax(wet))
        last = int(len(wet) - 1 - np.argmax(wet[::-1]))

        if first == 0:
            lower = CROWN_Y
        else:
            lower = crossing(
                PROBE_Y[first - 1],
                PROBE_Y[first],
                profile[first - 1],
                profile[first],
            )
        if last == len(wet) - 1:
            upper = PROBE_Y[-1]
        else:
            upper = crossing(
                PROBE_Y[last],
                PROBE_Y[last + 1],
                profile[last],
                profile[last + 1],
            )

        yint[row] = max(lower - CROWN_Y, 0.0) / L_TOWER
        yfs[row] = max(upper - CROWN_Y, 0.0) / L_TOWER
    return yint, yfs


def interp_rmse(x_model, y_model, x_obs, y_obs) -> tuple[float, int]:
    """Return RMSE and coverage without interpolating across missing model data."""
    x_model = np.asarray(x_model, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    finite_x = np.isfinite(x_model)
    x_model = x_model[finite_x]
    y_model = y_model[finite_x]
    if not x_model.size:
        return float("nan"), 0

    residuals = []
    for obs_x, obs_y in zip(x_obs, y_obs):
        if (
            not np.isfinite(obs_x)
            or not np.isfinite(obs_y)
            or obs_x < x_model[0]
            or obs_x > x_model[-1]
        ):
            continue

        right = int(np.searchsorted(x_model, obs_x, side="left"))
        if right < len(x_model) and x_model[right] == obs_x:
            prediction = y_model[right]
        elif 0 < right < len(x_model):
            left = right - 1
            if not np.isfinite(y_model[left]) or not np.isfinite(y_model[right]):
                continue
            fraction = (obs_x - x_model[left]) / (
                x_model[right] - x_model[left]
            )
            prediction = y_model[left] + fraction * (
                y_model[right] - y_model[left]
            )
        else:
            continue

        if np.isfinite(prediction):
            residuals.append(prediction - obs_y)

    if not residuals:
        return float("nan"), 0
    return float(np.sqrt(np.mean(np.square(residuals)))), len(residuals)


def fit_interface_run(run: np.ndarray) -> dict[str, float]:
    """Fit the unambiguous climb below the converged free-surface markers."""
    fit_points = run[(run[:, 1] >= 0.04) & (run[:, 1] <= 0.55)]
    if fit_points.shape[0] < 2:
        return {
            "climb_velocity_Vstar": float("nan"),
            "liftoff_Tstar_fit": float("nan"),
            "catch_Tstar_fit_at_Ystar_0p63": float("nan"),
        }
    slope, intercept = np.polyfit(fit_points[:, 0], fit_points[:, 1], 1)
    return {
        "climb_velocity_Vstar": float(slope),
        "liftoff_Tstar_fit": float(-intercept / slope),
        "catch_Tstar_fit_at_Ystar_0p63": float(
            (FREE_SURFACE_TARGET - intercept) / slope
        ),
    }


def cluster_interface_repetitions(time: np.ndarray, level: np.ndarray) -> list[np.ndarray]:
    """Separate the three approximately parallel Fig. 7 interface traces."""
    finite = np.isfinite(time) & np.isfinite(level) & (level > 0.03)
    points = np.column_stack((time[finite], level[finite]))
    if points.shape[0] < 6:
        return [points] if points.size else []

    repetition_count = 3
    nominal_slope = 0.55
    intercept_coordinate = points[:, 0] - points[:, 1] / nominal_slope
    centroids = np.quantile(intercept_coordinate, [1 / 6, 3 / 6, 5 / 6])
    labels = np.zeros(points.shape[0], dtype=int)

    for _ in range(20):
        labels = np.argmin(
            np.abs(intercept_coordinate[:, None] - centroids[None, :]),
            axis=1,
        )
        updated = np.array(
            [
                (
                    intercept_coordinate[labels == cluster].mean()
                    if np.any(labels == cluster)
                    else centroids[cluster]
                )
                for cluster in range(repetition_count)
            ]
        )
        if np.allclose(updated, centroids):
            break
        centroids = updated

    # Refine the intercept grouping against a fitted line for each repetition.
    for _ in range(4):
        lines = []
        for cluster in range(repetition_count):
            cluster_points = points[labels == cluster]
            lines.append(
                np.polyfit(cluster_points[:, 0], cluster_points[:, 1], 1)
                if cluster_points.shape[0] >= 2
                else None
            )
        for point_index, point in enumerate(points):
            distances = [
                (
                    abs(np.polyval(line, point[0]) - point[1])
                    if line is not None
                    else float("inf")
                )
                for line in lines
            ]
            labels[point_index] = int(np.argmin(distances))

    runs = []
    for cluster in range(repetition_count):
        run = points[labels == cluster]
        if run.shape[0] >= 2:
            runs.append(run[np.argsort(run[:, 0])])
    return sorted(runs, key=lambda run: fit_interface_run(run)["liftoff_Tstar_fit"])


def first_time(time: np.ndarray, condition: np.ndarray) -> float:
    idx = np.where(condition)[0]
    return float(time[idx[0]]) if idx.size else float("nan")


def main() -> None:
    transducer = read_probe("transducer", "p")
    tower = read_probe("towerCentreline", "alpha.water")

    time = transducer[:, 0]
    p = transducer[:, 1]
    t_tower = tower[:, 0]
    alpha = tower[:, 1:]
    if alpha.shape[1] != len(PROBE_Y):
        raise RuntimeError(f"Expected {len(PROBE_Y)} tower probes, found {alpha.shape[1]}")

    hstar_probe_raw = (p - P_ATM) / (RHO_W * G * L_TOWER)
    hstar_probe = moving_average(time, hstar_probe_raw, 0.10)
    # The pressure tap is sampled just above the pipe invert, while the paper's
    # calibrated H* is referenced to the tower-base/pipe-crown elevation.
    hstar_raw = hstar_probe_raw - PRESSURE_DATUM_CORRECTION_HSTAR
    hstar = moving_average(time, hstar_raw, 0.10)
    tstar = time * TIME_SCALE
    tstar_tower = t_tower * TIME_SCALE
    yint, yfs = extract_levels(alpha)

    pressure_exp = np.genfromtxt(
        CASE_A / "data" / "digitized" / "fig5_caseA_Hstar_band.csv",
        delimiter=",",
        names=True,
    )
    levels_exp = np.genfromtxt(
        CASE_A / "data" / "digitized" / "fig7_caseA_levels.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    fs_mask = levels_exp["kind"] == "fs"
    int_mask = levels_exp["kind"] == "int"

    pressure_rmse, pressure_samples = interp_rmse(
        tstar,
        hstar,
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
    )
    fs_rmse, fs_samples = interp_rmse(
        tstar_tower,
        yfs,
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
    )
    int_rmse, int_samples = interp_rmse(
        tstar_tower,
        yint,
        levels_exp["Tstar"][int_mask],
        levels_exp["Ystar"][int_mask],
    )
    interface_runs = cluster_interface_repetitions(
        levels_exp["Tstar"][int_mask],
        levels_exp["Ystar"][int_mask],
    )
    interface_repetition_metrics = []
    for repetition, run in enumerate(interface_runs, start=1):
        repetition_rmse, repetition_samples = interp_rmse(
            tstar_tower,
            yint,
            run[:, 0],
            run[:, 1],
        )
        interface_repetition_metrics.append(
            {
                "repetition": repetition,
                "RMSE_Ystar_no_shift": repetition_rmse,
                "samples_used": repetition_samples,
                "digitized_samples": int(run.shape[0]),
                **fit_interface_run(run),
            }
        )

    plateau = (tstar >= 1.0) & (tstar <= min(7.0, np.nanmax(tstar)))
    liftoff = first_time(tstar_tower, yint > 0.02)
    catch = first_time(tstar_tower, (yint > 0.05) & ((yfs - yint) < 0.02))
    model_climb = (
        np.isfinite(yint)
        & (yint >= 0.10)
        & (yint <= 0.55)
        & (tstar_tower <= catch)
    )
    if np.count_nonzero(model_climb) >= 2:
        model_slope, model_intercept = np.polyfit(
            tstar_tower[model_climb],
            yint[model_climb],
            1,
        )
        model_catch_fit = (
            FREE_SURFACE_TARGET - model_intercept
        ) / model_slope
    else:
        model_slope = float("nan")
        model_catch_fit = float("nan")
    repetition_rmses = [
        item["RMSE_Ystar_no_shift"]
        for item in interface_repetition_metrics
        if np.isfinite(item["RMSE_Ystar_no_shift"])
    ]
    catch_repetition_fits = [
        item["catch_Tstar_fit_at_Ystar_0p63"]
        for item in interface_repetition_metrics
        if np.isfinite(item["catch_Tstar_fit_at_Ystar_0p63"])
    ]
    metrics = {
        "simulation_end_s": float(time[-1]),
        "simulation_end_Tstar": float(tstar[-1]),
        "pressure_plateau_Hstar_mean_T1to7": (
            float(np.nanmean(hstar[plateau])) if np.any(plateau) else float("nan")
        ),
        "pressure_probe_plateau_Hstar_mean_T1to7": (
            float(np.nanmean(hstar_probe[plateau]))
            if np.any(plateau)
            else float("nan")
        ),
        "pressure_RMSE_Hstar_no_shift": pressure_rmse,
        "pressure_datum": {
            "comparison": "pipe crown / ventilation-tower base",
            "probe_y_m": TRANSDUCER_Y,
            "comparison_y_m": CROWN_Y,
            "subtracted_Hstar": PRESSURE_DATUM_CORRECTION_HSTAR,
            "probe_pressure_retained_in_series": True,
        },
        "free_surface_max_Ystar": float(np.nanmax(yfs)),
        "free_surface_RMSE_Ystar_no_shift": fs_rmse,
        "interface_RMSE_Ystar_no_shift": int_rmse,
        "interface_RMSE_best_repetition_no_shift": (
            float(min(repetition_rmses)) if repetition_rmses else float("nan")
        ),
        "interface_repetition_comparison": interface_repetition_metrics,
        "interface_climb_velocity_Vstar_fit": float(model_slope),
        "rmse_sample_coverage": {
            "pressure": {
                "used": pressure_samples,
                "digitized_finite": int(
                    np.count_nonzero(
                        np.isfinite(pressure_exp["Tstar"])
                        & np.isfinite(pressure_exp["Hstar_med"])
                    )
                ),
            },
            "free_surface": {
                "used": fs_samples,
                "digitized_finite": int(np.count_nonzero(fs_mask)),
            },
            "interface": {
                "used": int_samples,
                "digitized_finite": int(np.count_nonzero(int_mask)),
            },
        },
        "interface_liftoff_Tstar": liftoff,
        "interface_catch_Tstar": catch,
        "interface_catch_Tstar_fit_at_Ystar_0p63": float(model_catch_fit),
        "geysering": bool(np.nanmax(yfs) >= 0.98),
        "comparison_targets": {
            "pressure_plateau_Hstar": PRESSURE_PLATEAU_TARGET,
            "free_surface_max_Ystar": FREE_SURFACE_TARGET,
            "interface_liftoff_Tstar_repetitions": [7.3, 7.8, 7.9],
            "interface_catch_Tstar_earliest_repetition": 8.4,
            "interface_catch_Tstar_repetition_fits": catch_repetition_fits,
            "interface_climb_velocity_Vstar": INTERFACE_VELOCITY_TARGET,
            "observed_geysering": False,
        },
        "caveat": (
            "Planar 2-D area ratio Dt/D=0.607; physical circular area ratio "
            "(Dt/D)^2=0.369. No event-time shift was applied."
        ),
    }

    with (OUT / "openfoam_2d_series.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "time_s",
                "Tstar",
                "Hstar_raw",
                "Hstar_smooth",
                "Hstar_probe_raw",
                "Hstar_probe_smooth",
            ]
        )
        writer.writerows(
            zip(
                time,
                tstar,
                hstar_raw,
                hstar,
                hstar_probe_raw,
                hstar_probe,
            )
        )
    with (OUT / "openfoam_2d_levels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "Tstar", "Yint_star", "Yfs_star"])
        writer.writerows(zip(t_tower, tstar_tower, yint, yfs))
    (OUT / "openfoam_2d_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ax.fill_between(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_min"],
        pressure_exp["Hstar_max"],
        color="0.86",
        label="Experiment: three-run envelope",
    )
    ax.plot(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
        color="0.15",
        lw=1.4,
        label="Experiment: median",
    )
    ax.plot(
        tstar,
        hstar_probe,
        color="#4c78a8",
        lw=0.8,
        alpha=0.30,
        label="OpenFOAM probe datum (audit)",
    )
    ax.plot(
        tstar,
        hstar_raw,
        color="#c62828",
        lw=0.45,
        alpha=0.30,
        label="OpenFOAM crown datum, raw",
    )
    ax.plot(
        tstar,
        hstar,
        color="#c62828",
        lw=1.5,
        label="OpenFOAM crown datum, 0.10 s mean",
    )
    ax.set(xlabel=r"$T^*$", ylabel=r"$H^*$", xlim=(0, 10), ylim=(-0.05, 0.85))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "openfoam_2d_pressure_comparison.png", dpi=300)
    fig.savefig(OUT / "openfoam_2d_pressure_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ax.scatter(
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
        marker="^",
        facecolors="none",
        edgecolors="0.35",
        s=22,
        label="Experiment $Y^*_{fs}$",
    )
    ax.scatter(
        levels_exp["Tstar"][int_mask],
        levels_exp["Ystar"][int_mask],
        marker="o",
        facecolors="none",
        edgecolors="0.15",
        s=22,
        label="Experiment $Y^*_{int}$",
    )
    ax.plot(tstar_tower, yfs, color="#c62828", lw=1.5, label="OpenFOAM $Y^*_{fs}$")
    ax.plot(
        tstar_tower,
        yint,
        color="#c62828",
        lw=1.5,
        ls="--",
        label="OpenFOAM $Y^*_{int}$",
    )
    ax.set(xlabel=r"$T^*$", ylabel=r"$Y^*$", xlim=(7, 10), ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "openfoam_2d_levels_comparison.png", dpi=300)
    fig.savefig(OUT / "openfoam_2d_levels_comparison.pdf")
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
