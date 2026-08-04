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


def _main_wet_segment(wet: np.ndarray, min_cells: int = 3) -> tuple[int, int] | None:
    """Return the inclusive bounds of the principal contiguous water column.

    A first-wet-probe rule is not robust after gas enters the tower: isolated
    droplets or residual wet cells below the main column can cross
    ``alpha.water = 0.5`` and make the inferred interface jump to the tower
    base.  The physically relevant free-surface column is the longest
    contiguous wet run.  Runs shorter than ``min_cells`` are treated as
    unresolved droplets rather than a column.
    """
    padded = np.r_[False, wet, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    if starts.size == 0:
        return None

    lengths = stops - starts + 1
    valid = lengths >= min_cells
    if not np.any(valid):
        return None

    candidate_indices = np.flatnonzero(valid)
    max_length = int(np.max(lengths[valid]))
    longest = candidate_indices[lengths[candidate_indices] == max_length]
    # In the unlikely event of a tie, retain the higher column segment.
    selected = int(longest[np.argmax(stops[longest])])
    return int(starts[selected]), int(stops[selected])


def extract_levels(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yint = np.full(alpha.shape[0], np.nan)
    yfs = np.full(alpha.shape[0], np.nan)
    for row, profile in enumerate(alpha):
        wet = profile >= 0.5
        main_segment = _main_wet_segment(wet)
        if main_segment is None:
            continue

        first, last = main_segment

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

    # Once the connected upper water column fragments, a centreline profile
    # no longer has a unique gas-front/free-surface pair.  Do not let the
    # tracker alternate between the upper column and bottom residual liquid;
    # terminate the level history at the first topology-loss jump.
    for row in range(1, len(yint)):
        if not (
            np.isfinite(yint[row - 1])
            and np.isfinite(yint[row])
            and np.isfinite(yfs[row - 1])
            and np.isfinite(yfs[row])
        ):
            continue
        interface_drop = yint[row - 1] - yint[row]
        surface_drop = yfs[row - 1] - yfs[row]
        if yint[row - 1] > 0.05 and (interface_drop > 0.10 or surface_drop > 0.20):
            yint[row:] = np.nan
            yfs[row:] = np.nan
            break
    return yint, yfs


def interp_rmse(x_model, y_model, x_obs, y_obs) -> float:
    model_mask = np.isfinite(x_model) & np.isfinite(y_model)
    if not np.any(model_mask):
        return float("nan")
    x_valid = np.asarray(x_model)[model_mask]
    y_valid = np.asarray(y_model)[model_mask]
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    mask = (
        np.isfinite(x_obs)
        & np.isfinite(y_obs)
        & (x_obs >= x_valid[0])
        & (x_obs <= x_valid[-1])
    )
    if not np.any(mask):
        return float("nan")
    pred = np.interp(x_obs[mask], x_valid, y_valid)
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

    # The numerical probe is close to the pipe invert, while the published
    # elevation and pressure comparison uses the pipe-crown datum.  Remove
    # only that hydrostatic elevation difference; this is a datum conversion,
    # not a fitted vertical shift.
    head_at_probe = (p - P_ATM) / (RHO_W * G)
    head_at_crown = head_at_probe - (CROWN_Y - TRANSDUCER_Y)
    hstar_raw = head_at_crown / L_TOWER
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

    pressure_rmse = interp_rmse(
        tstar,
        hstar,
        pressure_exp["Tstar"],
        pressure_exp["Hstar_med"],
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

    plateau = (tstar >= 1.0) & (tstar <= min(7.0, np.nanmax(tstar)))
    liftoff = first_time(tstar_tower, yint > 0.02)
    catch = first_time(tstar_tower, (yint > 0.05) & ((yfs - yint) < 0.02))
    metrics = {
        "simulation_end_s": float(time[-1]),
        "simulation_end_Tstar": float(tstar[-1]),
        "pressure_plateau_Hstar_mean_T1to7": (
            float(np.nanmean(hstar[plateau])) if np.any(plateau) else float("nan")
        ),
        "pressure_RMSE_Hstar_no_shift": pressure_rmse,
        "free_surface_max_Ystar": float(np.nanmax(yfs)),
        "free_surface_RMSE_Ystar_no_shift": fs_rmse,
        "interface_RMSE_Ystar_no_shift": int_rmse,
        "interface_liftoff_Tstar": liftoff,
        "interface_catch_Tstar": catch,
        "connected_level_trace_end_Tstar": (
            float(tstar_tower[np.flatnonzero(np.isfinite(yfs))[-1]])
            if np.any(np.isfinite(yfs))
            else float("nan")
        ),
        "geysering": bool(np.nanmax(yfs) >= 0.98),
        "comparison_targets": {
            "pressure_plateau_Hstar": 0.54,
            "free_surface_max_Ystar": 0.63,
            "interface_liftoff_Tstar_repetitions": [7.3, 7.8, 7.9],
            "interface_catch_Tstar": 8.4,
            "observed_geysering": False,
        },
        "caveat": (
            "Planar 2-D area ratio Dt/D=0.607; physical circular area ratio "
            "(Dt/D)^2=0.369. Pressure was converted from the y=-0.043 m "
            "probe to the pipe-crown datum; no fitted pressure or event-time "
            "shift was applied."
        ),
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
    ax.plot(tstar, hstar_raw, color="#4c78a8", lw=0.45, alpha=0.35, label="OpenFOAM raw")
    ax.plot(tstar, hstar, color="#c62828", lw=1.5, label="OpenFOAM 0.10 s mean")
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
