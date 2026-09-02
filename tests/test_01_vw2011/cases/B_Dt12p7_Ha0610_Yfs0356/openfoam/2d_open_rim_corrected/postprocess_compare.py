"""Extract OpenFOAM probe histories and compare them with V&W2011 Case B."""
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
CASE_ROOT = HERE.parent.parent
DATA = CASE_ROOT / "data" / "digitized"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

P_ATM = 101325.0
RHO_W = 998.2
G = 9.81
DT = 0.0127
L_TOWER = 0.610
CROWN_Y = 0.047
RIM_Y = CROWN_Y + L_TOWER
TIME_SCALE = math.sqrt(G * DT) / L_TOWER
# Through the physical tower to the final cell centre below y=0.657 m.
PROBE_Y = np.arange(0.052, 0.653, 0.010)


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
            # A wet final probe means the connected column has reached the
            # directly open physical rim.
            upper = RIM_Y
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


def interp_rmse(x_model, y_model, x_obs, y_obs) -> float:
    mask = (
        np.isfinite(x_obs)
        & np.isfinite(y_obs)
        & (x_obs >= np.nanmin(x_model))
        & (x_obs <= np.nanmax(x_model))
    )
    if not np.any(mask):
        return float("nan")
    pred = np.interp(x_obs[mask], x_model, y_model)
    return float(np.sqrt(np.mean((pred - y_obs[mask]) ** 2)))


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

    hstar_raw = (p - P_ATM) / (RHO_W * G * L_TOWER)
    hstar = moving_average(time, hstar_raw, 0.10)
    tstar = time * TIME_SCALE
    tstar_tower = t_tower * TIME_SCALE
    yint, yfs = extract_levels(alpha)

    pressure_exp = np.genfromtxt(
        DATA / "fig6_caseB_pressure_mean_range_v3.csv",
        delimiter=",",
        names=True,
    )
    levels_exp = np.genfromtxt(
        DATA / "fig8_caseB_levels_runs_v2.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    fs_mask = (levels_exp["kind"] == "fs") & (levels_exp["role"] == "rising_track")
    int_mask = (levels_exp["kind"] == "int") & (levels_exp["role"] == "rising_track")

    pressure_rmse = interp_rmse(
        tstar,
        hstar,
        pressure_exp["Tstar"],
        pressure_exp["Hstar_mean"],
    )
    fs_rmse = interp_rmse(
        tstar_tower,
        yfs,
        levels_exp["Tstar"][fs_mask],
        levels_exp["Ystar"][fs_mask],
    )
    int_rmse = interp_rmse(
        tstar_tower,
        yint,
        levels_exp["Tstar"][int_mask],
        levels_exp["Ystar"][int_mask],
    )

    plateau = (tstar >= 1.0) & (tstar <= min(3.0, np.nanmax(tstar)))
    liftoff = first_time(tstar_tower, yint > 0.02)
    catch = first_time(tstar_tower, (yint > 0.05) & ((yfs - yint) < 0.02))
    metrics = {
        "case": "VW2011 Test 1 Case B",
        "dimension": "2d_planar_open_physical_rim",
        "simulation_end_s": float(time[-1]),
        "simulation_end_Tstar": float(tstar[-1]),
        "pressure_plateau_Hstar_mean_T1to3": (
            float(np.nanmean(hstar[plateau])) if np.any(plateau) else float("nan")
        ),
        "pressure_RMSE_Hstar_no_shift": pressure_rmse,
        "free_surface_max_Ystar": float(np.nanmax(yfs)),
        "free_surface_RMSE_Ystar_no_shift": fs_rmse,
        "interface_RMSE_Ystar_no_shift": int_rmse,
        "interface_liftoff_Tstar": liftoff,
        "interface_catch_Tstar": catch,
        "geysering": bool(np.nanmax(yfs) >= 1.0),
        "geyser_before_catch": bool(
            np.isfinite(first_time(tstar_tower, yfs >= 1.0))
            and (
                not np.isfinite(catch)
                or first_time(tstar_tower, yfs >= 1.0) <= catch
            )
        ),
        "comparison_targets": {
            "observed_geysering": True,
            "figure_pressure": "Fig.6",
            "figure_levels": "Fig.8",
            "pressure_plateau_Hstar_exp_mean_approx": 0.758,
        },
        "paper_inputs": {
            "D_m": 0.094,
            "Dt_m": 0.0127,
            "L_m": 0.610,
            "Ha0_m": 0.610,
            "Yfs0_m": 0.356,
            "chamber_m": 0.546,
            "middle_m": 2.970,
            "downstream_m": 0.490,
        },
        "caveat": (
            "Planar 2-D keeps paper lengths, D, Ha0, Yfs0 and BCs. Tower width "
            "uses hydraulic equivalence W=Dt^2/D so W/D=(Dt/D)^2; sigma=0 to "
            "avoid capillary lock in the thin bore. Valve opening instantaneous "
            "(paper: manual <1 s). The physical rim at y=0.657 m opens directly "
            "to atmosphere; no confined headroom is retained."
        ),
        "planar_tower_width_m": 0.0127**2 / 0.094,
        "sigma": 0.0,
    }

    with (OUT / "openfoam_2d_series.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "Tstar", "Hstar_raw", "Hstar_smooth"])
        writer.writerows(zip(time, tstar, hstar_raw, hstar))
    with (OUT / "openfoam_2d_levels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "Tstar", "Yint_star", "Yfs_star"])
        writer.writerows(zip(t_tower, tstar_tower, yint, yfs))
    (OUT / "openfoam_2d_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ax.plot(
        pressure_exp["Tstar"],
        pressure_exp["Hstar_mean"],
        color="0.15",
        lw=1.4,
        label="Experiment: mean (n=3)",
    )
    ax.plot(tstar, hstar_raw, color="#4c78a8", lw=0.45, alpha=0.35, label="OpenFOAM raw")
    ax.plot(tstar, hstar, color="#c62828", lw=1.5, label="OpenFOAM 0.10 s mean")
    ax.set(xlabel=r"$T^*$", ylabel=r"$H^*$", xlim=(0, 10), ylim=(-0.05, 1.2))
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
    ax.set(xlabel=r"$T^*$", ylabel=r"$Y^*$", xlim=(0, 10), ylim=(0, 1.35))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "openfoam_2d_levels_comparison.png", dpi=300)
    fig.savefig(OUT / "openfoam_2d_levels_comparison.pdf")
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
