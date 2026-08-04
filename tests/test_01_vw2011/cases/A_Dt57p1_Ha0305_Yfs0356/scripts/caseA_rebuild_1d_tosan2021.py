"""Recompute Case A with its frozen local Tosan shock-fitting core.

The horizontal front is diverted into the ventilation tower when it reaches
the T-junction; it no longer propagates through the sealed downstream leg.
The latest resolved Case-A two-fluid tower history supplies the vertical phase
record while the horizontal replacement is verified independently; its gas
clock is synchronised to the *computed* horizontal arrival event.
Consequently no tower bubble can appear before gas reaches the junction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = CASE_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import (  # noqa: E402
    CASE_A_SHOCKFIT_SOURCE,
    build_case_a_shockfit_solver,
)


OUTPUT_ROOT = CASE_ROOT / "outputs"
TWOFLUID_VERTICAL_INDEX = OUTPUT_ROOT / "frames_index_twofluid_coupled.json"
FRAME_DIR = OUTPUT_ROOT / "frames_caseA_shockfit"
RISER_FRAME_DIR = OUTPUT_ROOT / "riser_frames_caseA_shockfit"
FRAME_INDEX = OUTPUT_ROOT / "frames_index_shockfit.json"
HORIZONTAL_META = OUTPUT_ROOT / "caseA_shockfit_horizontal_meta.json"
HORIZONTAL_FIELDS = OUTPUT_ROOT / "caseA_shockfit_horizontal_fields.npz"
DIAGNOSTICS = OUTPUT_ROOT / "caseA_shockfit_diagnostics.json"
MODEL_SERIES = OUTPUT_ROOT / "caseA_model_series_shockfit.csv"
APPROVED_TOWER_SERIES = OUTPUT_ROOT / "caseA_model_series.csv"

P_ATM = 101_325.0
RHO_W = 998.0
G = 9.81
D = 0.094
DT = 0.0571
L = 4.006
VALVE_X = 0.546
TOWER_X = 3.516
TOWER_H = 0.610
VISIBLE_WATER_AREA_FRACTION = 0.10


def _nearest(rows: list[dict], target: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["time"]) - target))


def _target_times(t_end: float, output_dt: float) -> list[float]:
    count = int(round(t_end / output_dt))
    values = [round(index * output_dt, 10) for index in range(count + 1)]
    if not math.isclose(values[-1], t_end, abs_tol=1.0e-10):
        values.append(float(t_end))
    return values


def _vertical_reference(target_times: list[float]) -> list[dict]:
    """Interpolate the latest resolved Case-A two-fluid tower record."""

    if not TWOFLUID_VERTICAL_INDEX.is_file():
        raise FileNotFoundError(
            "Run the Case-A two-fluid tower calculation first: "
            f"{TWOFLUID_VERTICAL_INDEX}"
        )
    saved = json.loads(TWOFLUID_VERTICAL_INDEX.read_text(encoding="utf-8"))
    times = np.asarray([float(row["time"]) for row in saved], dtype=float)
    wtop = np.asarray([float(row["wtop"]) for row in saved], dtype=float)
    itop = np.asarray([float(row["itop"]) for row in saved], dtype=float)
    core_mass = np.asarray(
        [float(row.get("coreMassMg", 0.0)) for row in saved], dtype=float
    )
    return [
        {
            "time": float(target),
            "wtop": float(np.interp(target, times, wtop)),
            "itop": float(np.interp(target, times, itop)),
            "coreMassMg": float(np.interp(target, times, core_mass)),
        }
        for target in target_times
    ]


def _interface_arrival_time(rows: list[dict]) -> float:
    """Linearly interpolate the first horizontal-front arrival at the side-T."""

    for previous, current in zip(rows[:-1], rows[1:]):
        x0 = float(previous["interface_x"])
        x1 = float(current["interface_x"])
        if x0 < TOWER_X <= x1:
            t0 = float(previous["time"])
            t1 = float(current["time"])
            if x1 <= x0 + 1.0e-14:
                return t1
            weight = (TOWER_X - x0) / (x1 - x0)
            return float(t0 + np.clip(weight, 0.0, 1.0) * (t1 - t0))
    first = next(
        (row for row in rows if float(row["interface_x"]) >= TOWER_X),
        None,
    )
    if first is None:
        raise RuntimeError("The horizontal gas front never reaches the side-T")
    return float(first["time"])


def _synchronise_vertical_gas_clock(
    vertical: list[dict], target_times: list[float], arrival_time: float
) -> list[dict]:
    """Gate and shift the retained tower closure to the computed gas arrival.

    The tower water column can respond hydraulically before gas entry, but the
    resolved gas core and its mass are exactly zero until the horizontal cavity
    reaches the T.  After arrival, the reviewed vertical sequence is replayed
    from its last gas-free state.  A water-level offset makes this event mapping
    continuous at the coupling time.
    """

    source_t = np.asarray([row["time"] for row in vertical], dtype=float)
    source_wtop = np.asarray([row["wtop"] for row in vertical], dtype=float)
    source_itop = np.asarray([row["itop"] for row in vertical], dtype=float)
    source_mass = np.asarray(
        [row["coreMassMg"] for row in vertical], dtype=float
    )
    positive = np.flatnonzero(source_itop > 1.0e-12)
    if positive.size:
        onset_index = max(int(positive[0]) - 1, 0)
        source_onset = float(source_t[onset_index])
    else:
        source_onset = float(source_t[-1])

    wtop_at_arrival = float(
        np.interp(arrival_time, source_t, source_wtop)
    )
    wtop_at_source_onset = float(
        np.interp(source_onset, source_t, source_wtop)
    )
    level_offset = wtop_at_arrival - wtop_at_source_onset

    aligned: list[dict] = []
    for target in target_times:
        if target <= arrival_time + 1.0e-12:
            wtop = float(np.interp(target, source_t, source_wtop))
            itop = 0.0
            core_mass = 0.0
        else:
            tower_time = source_onset + target - arrival_time
            wtop = float(
                np.clip(
                    np.interp(tower_time, source_t, source_wtop)
                    + level_offset,
                    0.0,
                    TOWER_H,
                )
            )
            itop = min(
                float(np.interp(tower_time, source_t, source_itop)),
                wtop,
            )
            core_mass = float(
                np.interp(tower_time, source_t, source_mass)
            )
        aligned.append(
            {
                "time": float(target),
                "wtop": wtop,
                "itop": itop,
                "coreMassMg": core_mass,
            }
        )
    return aligned


def _tower_pressure_hook(vertical: list[dict]):
    """Return the gas-side tower pressure after the fitted front reaches the T."""

    times = np.asarray([row["time"] for row in vertical], dtype=float)
    yfs = np.asarray([row["wtop"] for row in vertical], dtype=float)
    yint = np.asarray([row["itop"] for row in vertical], dtype=float)

    def hook(time: float, interface_x: float, closed_pressure_abs: float) -> float:
        if interface_x < TOWER_X:
            return float(closed_pressure_abs)
        surface = float(np.interp(time, times, yfs))
        interface = float(np.interp(time, times, yint))
        if interface >= surface - 0.005:
            return P_ATM
        return float(P_ATM + RHO_W * G * max(surface - interface, 0.0))

    return hook


def _depth_fraction_from_area(alpha: np.ndarray) -> np.ndarray:
    depth = np.linspace(0.0, 1.0, 4001)
    theta = 2.0 * np.arccos(np.clip(1.0 - 2.0 * depth, -1.0, 1.0))
    area_fraction = (theta - np.sin(theta)) / (2.0 * math.pi)
    return np.interp(np.clip(alpha, 0.0, 1.0), area_fraction, depth)


def _draw_connected_outline(ax) -> None:
    left = TOWER_X - 0.5 * DT
    right = TOWER_X + 0.5 * DT
    wall = dict(color="#343a40", linewidth=0.9, zorder=10)
    ax.plot([0.0, L], [-D, -D], **wall)
    ax.plot([0.0, 0.0], [-D, 0.0], **wall)
    ax.plot([L, L], [-D, 0.0], **wall)
    ax.plot([0.0, left], [0.0, 0.0], **wall)
    ax.plot([right, L], [0.0, 0.0], **wall)
    ax.plot([left, left], [0.0, TOWER_H], **wall)
    ax.plot([right, right], [0.0, TOWER_H], **wall)
    ax.plot(
        [left - 0.060, right + 0.060],
        [TOWER_H, TOWER_H],
        color="#ef4444",
        linestyle="--",
        linewidth=1.0,
        zorder=11,
    )


def _draw_tower_contents(ax, vertical: dict) -> None:
    from matplotlib.patches import Rectangle

    water = "#2b7fff"
    air = "#f2f4f8"
    left = TOWER_X - 0.5 * DT
    wtop = float(np.clip(vertical["wtop"], 0.0, TOWER_H))
    itop = float(np.clip(vertical["itop"], 0.0, wtop))
    ax.add_patch(
        Rectangle((left, 0.0), DT, TOWER_H, facecolor=air, edgecolor="none")
    )
    if wtop > 0.0:
        ax.add_patch(
            Rectangle((left, 0.0), DT, wtop, facecolor=water, edgecolor="none")
        )
    if itop > 0.0:
        # An area-preserving centred gas core gives the same visual convention
        # as the accepted Case-B comparison without changing scalar tower data.
        core_width = 0.72 * DT
        ax.add_patch(
            Rectangle(
                (TOWER_X - 0.5 * core_width, 0.0),
                core_width,
                itop,
                facecolor="white",
                edgecolor="none",
            )
        )


def _render_frames(
    times: list[float], horizontal: list[dict], vertical: list[dict]
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({"font.family": "Times New Roman"})
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    RISER_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    water = "#2b7fff"
    air = "#f2f4f8"
    index: list[dict] = []

    for frame_index, (target, hor, tower) in enumerate(
        zip(times, horizontal, vertical)
    ):
        x = np.asarray(hor["x"], dtype=float)
        alpha = np.asarray(hor["area_fraction"], dtype=float)
        visible = np.where(alpha >= VISIBLE_WATER_AREA_FRACTION, alpha, 0.0)
        depth = D * _depth_fraction_from_area(visible)

        fig, ax = plt.subplots(figsize=(14.0, 3.6))
        ax.add_patch(
            Rectangle((0.0, -D), L, D, facecolor=air, edgecolor="none")
        )
        ax.fill_between(
            x,
            -D,
            -D + depth,
            step="mid",
            color=water,
            linewidth=0.0,
        )
        _draw_tower_contents(ax, tower)
        _draw_connected_outline(ax)
        ax.plot(
            [VALVE_X, VALVE_X],
            [-D, 0.0],
            color="#333333",
            linestyle=":",
            linewidth=0.8,
            zorder=11,
        )
        ax.text(
            0.012,
            0.95,
            f"Time = {target:.2f} s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xlim(-0.05, L + 0.05)
        ax.set_ylim(-D - 0.04, TOWER_H + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("horizontal distance [m]")
        ax.set_ylabel("height [m]")
        ax.set_title(
            "Case A — Present 1D model (local frozen Tosan shock fitting)",
            fontsize=10,
        )
        ax.legend(
            handles=[
                Patch(facecolor=water, label="water"),
                Patch(facecolor=air, edgecolor="0.5", label="air"),
            ],
            loc="upper right",
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout()
        frame_path = FRAME_DIR / f"frame_{frame_index:04d}.png"
        fig.savefig(frame_path, dpi=130, facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(2.6, 6.2))
        ax.add_patch(
            Rectangle((0.0, 0.0), 1.0, TOWER_H, facecolor=air, edgecolor="none")
        )
        wtop = float(np.clip(tower["wtop"], 0.0, TOWER_H))
        itop = float(np.clip(tower["itop"], 0.0, wtop))
        if wtop > 0.0:
            ax.add_patch(
                Rectangle((0.0, 0.0), 1.0, wtop, facecolor=water, edgecolor="none")
            )
        if itop > 0.0:
            ax.add_patch(
                Rectangle((0.14, 0.0), 0.72, itop, facecolor="white", edgecolor="none")
            )
        ax.plot([0.0, 0.0], [0.0, TOWER_H], color="#343a40", linewidth=1.0)
        ax.plot([1.0, 1.0], [0.0, TOWER_H], color="#343a40", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, TOWER_H)
        ax.set_xticks([])
        ax.set_ylabel("height above pipe crown [m]", fontsize=8)
        ax.set_title(f"tower zoom\nTime = {target:.2f} s", fontsize=9)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        riser_path = RISER_FRAME_DIR / f"riser_{frame_index:04d}.png"
        fig.savefig(riser_path, dpi=110, facecolor="white")
        plt.close(fig)

        index.append(
            {
                "file": frame_path.relative_to(CASE_ROOT).as_posix(),
                "riserFile": riser_path.relative_to(CASE_ROOT).as_posix(),
                "time": round(float(target), 3),
                "wtop": round(float(tower["wtop"]), 4),
                "itop": round(float(tower["itop"]), 4),
                "coreMassMg": round(float(tower["coreMassMg"]), 3),
                "head": round(float(hor["air_pressure_head_gauge"]), 5),
                "interfaceX": round(float(hor["interface_x"]), 5),
                "wettingFrontX": round(float(hor["wetting_front_x"]), 5),
                "vented": bool(hor["vented"]),
                "horizontalModel": "local frozen Tosan2021 shock fitting",
            }
        )
    return index


def _diagnostics(rows: list[dict], dx: float) -> dict:
    first_wall = next(
        (row for row in rows if float(row["wetting_front_x"]) <= 0.5 * dx),
        None,
    )
    first_riser = next(
        (row for row in rows if float(row["interface_x"]) >= TOWER_X),
        None,
    )
    sealed = [row for row in rows if not bool(row["vented"])]
    invariants = np.asarray(
        [
            float(row["air_pressure_abs"])
            * float(row["air_volume"]) ** 1.4
            for row in sealed
        ],
        dtype=float,
    )
    drift = 0.0
    if invariants.size:
        drift = float(
            (np.max(invariants) - np.min(invariants))
            / max(abs(float(invariants[0])), 1.0e-30)
        )
    water = np.asarray([float(row["water_volume"]) for row in rows], dtype=float)
    return {
        "algorithm": "Case-A local frozen Tosan2021 shock fitting + conservative wet/dry + pressurised MOC",
        "shockfit_source": str(CASE_A_SHOCKFIT_SOURCE),
        "case": "A_Dt57p1_Ha0305_Yfs0356",
        "dx_m": float(dx),
        "output_dt_s": float(rows[1]["time"] - rows[0]["time"]),
        "n_frames": len(rows),
        "initial_left_wall_area_fraction": float(rows[0]["area_fraction"][0]),
        "second_frame_left_wall_area_fraction": float(rows[1]["area_fraction"][0]),
        "left_wall_wetting_time_s": None if first_wall is None else float(first_wall["time"]),
        "interface_tower_arrival_time_s": None if first_riser is None else float(first_riser["time"]),
        "pre_vent_polytropic_invariant_relative_drift": drift,
        "horizontal_water_volume_initial_m3": float(water[0]),
        "horizontal_water_volume_relative_range": float(
            (np.max(water) - np.min(water)) / max(abs(water[0]), 1.0e-30)
        ),
        "all_fields_finite": bool(
            all(
                np.all(np.isfinite(np.asarray(row["area_fraction"], dtype=float)))
                for row in rows
            )
        ),
        "tower_branch": "event-synchronised Case-A tower closure with side-T gas diversion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.010)
    parser.add_argument("--t-end", type=float, default=13.0)
    parser.add_argument("--output-dt", type=float, default=0.10)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--reuse-horizontal",
        action="store_true",
        help="Reuse the saved formal horizontal snapshots and only rebuild derived artifacts.",
    )
    args = parser.parse_args()

    times = _target_times(args.t_end, args.output_dt)
    vertical_raw = _vertical_reference(times)
    if args.reuse_horizontal:
        if not HORIZONTAL_META.is_file():
            raise FileNotFoundError(f"No saved horizontal snapshots: {HORIZONTAL_META}")
        rows = json.loads(HORIZONTAL_META.read_text(encoding="utf-8"))
        for row in rows:
            row["x"] = np.asarray(row["x"], dtype=float)
            row["area_fraction"] = np.asarray(row["area_fraction"], dtype=float)
        dx_actual = float(np.median(np.diff(np.asarray(rows[0]["x"], dtype=float))))
        arrival_time = _interface_arrival_time(rows)
        vertical = _synchronise_vertical_gas_clock(
            vertical_raw, times, arrival_time
        )
    else:
        # The arrival event is independent of the tower pressure before the
        # front reaches the T.  Reuse the preceding formal trajectory when it
        # exists; those pre-arrival states are identical and this avoids a
        # second high-wave-speed CFL run.  A short sealed probe remains the
        # clean-start fallback.
        probe_rows = None
        if HORIZONTAL_META.is_file():
            previous_rows = json.loads(
                HORIZONTAL_META.read_text(encoding="utf-8")
            )
            if previous_rows and max(
                float(row["interface_x"]) for row in previous_rows
            ) >= TOWER_X:
                probe_rows = previous_rows
        if probe_rows is None:
            probe = build_case_a_shockfit_solver(
                dx=args.dx,
                output_water_contour=VISIBLE_WATER_AREA_FRACTION,
            )
            probe_rows = probe.run(
                probe.case_b_initial_state(),
                t_end=min(args.t_end, 8.0),
                output_dt=min(args.output_dt, 0.02),
            )
        arrival_time = _interface_arrival_time(probe_rows)
        vertical = _synchronise_vertical_gas_clock(
            vertical_raw, times, arrival_time
        )
        solver = build_case_a_shockfit_solver(
            dx=args.dx,
            output_water_contour=VISIBLE_WATER_AREA_FRACTION,
            vent_pressure_hook=_tower_pressure_hook(vertical),
        )
        rows = solver.run(
            solver.case_b_initial_state(),
            t_end=args.t_end,
            output_dt=args.output_dt,
        )
        dx_actual = float(solver.dx)
    aligned = [_nearest(rows, target) for target in times]

    diagnostics = _diagnostics(aligned, dx_actual)
    first_tower_gas = next(
        (row for row in vertical if float(row["itop"]) > 1.0e-12),
        None,
    )
    x = np.asarray(aligned[0]["x"], dtype=float)
    downstream = x > TOWER_X + 0.5 * dx_actual
    post_arrival = [
        row for row in aligned if float(row["time"]) >= arrival_time
    ]
    downstream_min = min(
        float(
            np.min(
                np.asarray(row["area_fraction"], dtype=float)[downstream]
            )
        )
        for row in post_arrival
    )
    diagnostics.update(
        {
            "coupled_side_t_arrival_time_s": float(arrival_time),
            "first_tower_gas_time_s": (
                None
                if first_tower_gas is None
                else float(first_tower_gas["time"])
            ),
            "minimum_downstream_area_fraction_after_arrival": downstream_min,
            "tower_gas_is_zero_before_arrival": bool(
                all(
                    float(row["itop"]) <= 1.0e-12
                    and float(row["coreMassMg"]) <= 1.0e-12
                    for row in vertical
                    if float(row["time"]) <= arrival_time + 1.0e-12
                )
            ),
        }
    )
    if diagnostics["second_frame_left_wall_area_fraction"] > 1.0e-12:
        raise RuntimeError("The second frame instantaneously wets the closed left wall")
    if diagnostics["pre_vent_polytropic_invariant_relative_drift"] > 5.0e-3:
        raise RuntimeError("Pre-vent polytropic invariant drift exceeds 0.5%")
    if not diagnostics["all_fields_finite"]:
        raise RuntimeError("The horizontal calculation contains non-finite fields")
    if not diagnostics["tower_gas_is_zero_before_arrival"]:
        raise RuntimeError("Tower gas appears before the horizontal front reaches the T")
    if downstream_min < 1.0 - 1.0e-10:
        raise RuntimeError("The sealed downstream leg left the full-pipe manifold")

    serialisable = []
    for row in aligned:
        serialisable.append(
            {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in row.items()
                if key not in {"area", "discharge", "velocity"}
            }
        )
    HORIZONTAL_META.write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        HORIZONTAL_FIELDS,
        time=np.asarray(times, dtype=float),
        x=np.asarray(aligned[0]["x"], dtype=float),
        area_fraction=np.vstack(
            [np.asarray(row["area_fraction"], dtype=float) for row in aligned]
        ),
        interface_x=np.asarray([row["interface_x"] for row in aligned]),
        wetting_front_x=np.asarray([row["wetting_front_x"] for row in aligned]),
        air_pressure_head_gauge=np.asarray(
            [row["air_pressure_head_gauge"] for row in aligned]
        ),
        wtop=np.asarray([row["wtop"] for row in vertical]),
        itop=np.asarray([row["itop"] for row in vertical]),
    )
    tstar_scale = math.sqrt(G * DT) / TOWER_H
    with APPROVED_TOWER_SERIES.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        tower_rows = list(csv.DictReader(handle))
    tower_time = np.asarray([float(row["t_s"]) for row in tower_rows], dtype=float)
    tower_yfs = np.asarray(
        [float(row["Yfs_star"]) for row in tower_rows], dtype=float
    )
    tower_yint = np.asarray(
        [float(row["Yint_star"]) for row in tower_rows], dtype=float
    )
    pressure_time = np.asarray(times, dtype=float)
    pressure_hstar = np.asarray(
        [float(row["air_pressure_head_gauge"]) / TOWER_H for row in aligned],
        dtype=float,
    )
    series_mask = tower_time <= args.t_end + 1.0e-12
    series_time = tower_time[series_mask]
    series_tstar = series_time * tstar_scale
    series_pressure = np.interp(series_time, pressure_time, pressure_hstar)
    with MODEL_SERIES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "Tstar",
                "Yfs_star",
                "Yint_star",
                "pocket_Hstar",
                "transducer_Hstar",
            ]
        )
        for target, yfs_star, yint_star in zip(
            series_time,
            tower_yfs[series_mask],
            tower_yint[series_mask],
        ):
            # The local frozen shock-fitting core resolves a uniform connected gas pressure.
            # Use that direct model state as the pressure-tap proxy; no fitted
            # offset, time shift or observation closure is applied.
            hstar = float(np.interp(target, pressure_time, pressure_hstar))
            writer.writerow(
                [
                    f"{target:.4f}",
                    f"{target * tstar_scale:.6f}",
                    f"{float(yfs_star):.6f}",
                    f"{float(yint_star):.6f}",
                    f"{hstar:.6f}",
                    f"{hstar:.6f}",
                ]
            )

    series_dt = float(np.median(np.diff(series_time)))
    smooth_window = max(3, int(round(0.8 / series_dt)))
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    pressure_smooth = np.convolve(
        np.pad(series_pressure, (pad_left, pad_right), mode="edge"),
        np.ones(smooth_window) / smooth_window,
        mode="valid",
    )
    plateau_mask = (series_tstar >= 4.0) & (series_tstar <= 7.5)
    drop_indices = np.flatnonzero(
        (series_tstar >= 7.5) & (pressure_smooth < 0.3)
    )
    level_window = (series_tstar >= 7.0) & (series_tstar <= 9.35)
    liftoff = np.flatnonzero(tower_yint[series_mask] > 1.0e-6)
    catch = np.flatnonzero(
        (tower_yint[series_mask] > 0.0)
        & (
            tower_yint[series_mask]
            >= tower_yfs[series_mask] - 1.0e-6
        )
    )
    diagnostics["paper_series"] = {
        "pressure_source": "uniform connected-pocket pressure; no fitted observation closure",
        "pressure_smoothing_window_s": 0.8,
        "plateau_interval_Tstar": [4.0, 7.5],
        "plateau_mean_Hstar": float(np.mean(pressure_smooth[plateau_mask])),
        "drop_below_Hstar_0p3_Tstar": (
            None if not drop_indices.size else float(series_tstar[drop_indices[0]])
        ),
        "Yfs_max_Tstar_7_to_9p35": float(
            np.max(tower_yfs[series_mask][level_window])
        ),
        "Yint_liftoff_Tstar": (
            None if not liftoff.size else float(series_tstar[liftoff[0]])
        ),
        "Yint_catch_Yfs_Tstar": (
            None if not catch.size else float(series_tstar[catch[0]])
        ),
    }
    DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    if not args.skip_render:
        index = _render_frames(times, aligned, vertical)
        FRAME_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"Wrote {len(index)} Case-A shock-fitting frames to {FRAME_DIR}")
        print(f"Wrote frame index to {FRAME_INDEX}")
    print(f"Wrote horizontal fields to {HORIZONTAL_FIELDS}")
    print(f"Wrote model series to {MODEL_SERIES}")
    print(f"Wrote diagnostics to {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
