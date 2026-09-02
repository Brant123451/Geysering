"""Formal Case-B tower-level observer based on inlet-connected gas.

This script reads the archived tower-centreline
``alpha.water`` probes, rejects short detached wet islands, and defines the
gas--water interface as the alpha=0.5 lower edge of the first persistent
liquid barrier encountered above the tower inlet.  Equivalently, this is the
top of the gas core connected to the horizontal-pipe/tower junction.

No time translation, event alignment, or model-field modification is used.
All products are written below this frozen observer directory.
"""
from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OPENFOAM_CASE = HERE.parents[1]
CASE_ROOT = OPENFOAM_CASE.parent.parent
DATA = CASE_ROOT / "data" / "digitized"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

G = 9.81
DT = 0.0127
L_TOWER = 0.610
CROWN_Y = 0.047
RIM_Y = CROWN_Y + L_TOWER
TIME_SCALE = math.sqrt(G * DT) / L_TOWER

ALPHA_THRESHOLD = 0.5
# A one- or two-probe wet island spans at most 0.02 m and is treated as a
# detached droplet, not as a liquid barrier closing the full tower section.
MIN_PERSISTENT_WET_PROBES = 3


def probe_files(name: str, field: str) -> list[Path]:
    root = OPENFOAM_CASE / "postProcessing" / name
    paths = sorted(root.glob(f"*/{field}"), key=lambda p: float(p.parent.name))
    if not paths:
        raise FileNotFoundError(f"No probe files found below {root} for {field}")
    return paths


def read_probe(name: str, field: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in probe_files(name, field):
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.size:
            chunks.append(data)
    if not chunks:
        raise RuntimeError(f"No numeric probe rows for {name}/{field}")
    data = np.vstack(chunks)
    data = data[np.argsort(data[:, 0], kind="stable")]
    # A resumed OpenFOAM run can repeat checkpoint times.  Preserve the newest
    # occurrence at every physical time, matching the production postprocessor.
    _, reverse_index = np.unique(data[::-1, 0], return_index=True)
    keep = np.sort(len(data) - 1 - reverse_index)
    return data[keep]


def read_probe_y(path: Path) -> np.ndarray:
    pattern = re.compile(
        r"^#\s*Probe\s+\d+\s+\(\s*[-+0-9.eE]+\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s*\)"
    )
    y: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            match = pattern.match(line.strip())
            if match:
                y.append(float(match.group(1)))
    if not y:
        raise RuntimeError(f"Could not parse probe locations from {path}")
    values = np.asarray(y, dtype=float)
    if np.any(np.diff(values) <= 0.0):
        raise RuntimeError("Tower-centreline probe elevations are not increasing")
    return values


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, state in enumerate(mask):
        if state and start is None:
            start = index
        if start is not None and (not state or index == len(mask) - 1):
            end = index - 1 if not state else index
            runs.append((start, end))
            start = None
    return runs


def alpha_crossing(
    y0: float,
    y1: float,
    alpha0: float,
    alpha1: float,
    target: float = ALPHA_THRESHOLD,
) -> float:
    if abs(alpha1 - alpha0) < 1.0e-12:
        return 0.5 * (y0 + y1)
    fraction = np.clip((target - alpha0) / (alpha1 - alpha0), 0.0, 1.0)
    return float(y0 + fraction * (y1 - y0))


def legacy_levels(profile: np.ndarray, probe_y: np.ndarray) -> tuple[float, float]:
    """Reproduce the first-wet/last-wet production observer for comparison."""
    alpha = np.clip(np.asarray(profile, dtype=float), 0.0, 1.0)
    wet = alpha >= ALPHA_THRESHOLD
    if not np.any(wet):
        return math.nan, math.nan
    first = int(np.argmax(wet))
    last = int(len(wet) - 1 - np.argmax(wet[::-1]))
    lower = (
        CROWN_Y
        if first == 0
        else alpha_crossing(
            probe_y[first - 1], probe_y[first], alpha[first - 1], alpha[first]
        )
    )
    upper = (
        probe_y[-1]
        if last == len(wet) - 1
        else alpha_crossing(
            probe_y[last], probe_y[last + 1], alpha[last], alpha[last + 1]
        )
    )
    return max(lower - CROWN_Y, 0.0) / L_TOWER, max(
        upper - CROWN_Y, 0.0
    ) / L_TOWER


def connected_core_levels(
    profile: np.ndarray, probe_y: np.ndarray
) -> tuple[float, float, int, int, int, int]:
    """Return the connected-core interface and coherent free-surface levels.

    Short wet islands below the persistent water column are detached droplets;
    they cannot block the full tower cross-section.  The first persistent wet
    run therefore terminates the gas segment connected to the tower inlet.
    Both interface positions are interpolated on the unmodified alpha profile.
    """
    alpha = np.clip(np.asarray(profile, dtype=float), 0.0, 1.0)
    wet = alpha >= ALPHA_THRESHOLD
    all_runs = contiguous_runs(wet)
    persistent = [
        (start, end)
        for start, end in all_runs
        if end - start + 1 >= MIN_PERSISTENT_WET_PROBES
    ]
    if not persistent:
        return math.nan, math.nan, -1, -1, 0, int(np.count_nonzero(wet))

    # The first persistent water barrier is precisely where the inlet-connected
    # gas core terminates.  Any shorter wet runs beneath it are entrained drops.
    start, end = persistent[0]
    detached_below = int(np.count_nonzero(wet[:start]))
    lower = (
        CROWN_Y
        if start == 0
        else alpha_crossing(
            probe_y[start - 1], probe_y[start], alpha[start - 1], alpha[start]
        )
    )
    upper = (
        probe_y[-1]
        if end == len(wet) - 1
        else alpha_crossing(
            probe_y[end], probe_y[end + 1], alpha[end], alpha[end + 1]
        )
    )
    yint = max(lower - CROWN_Y, 0.0) / L_TOWER
    yfs_raw = max(upper - CROWN_Y, 0.0) / L_TOWER
    return yint, yfs_raw, start, end, end - start + 1, detached_below


def comparison_errors(
    model_t: np.ndarray,
    model_y: np.ndarray,
    obs_t: np.ndarray,
    obs_y: np.ndarray,
) -> dict[str, float | int]:
    valid_model = np.isfinite(model_t) & np.isfinite(model_y)
    if np.count_nonzero(valid_model) < 2:
        return {"n": 0, "rmse": math.nan, "bias": math.nan, "mae": math.nan}
    x = model_t[valid_model]
    y = model_y[valid_model]
    valid_obs = (
        np.isfinite(obs_t)
        & np.isfinite(obs_y)
        & (obs_t >= x[0])
        & (obs_t <= x[-1])
    )
    predicted = np.interp(obs_t[valid_obs], x, y)
    residual = predicted - obs_y[valid_obs]
    return {
        "n": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
    }


def first_upcrossing(time: np.ndarray, value: np.ndarray, target: float) -> float:
    valid = np.isfinite(time) & np.isfinite(value)
    index = np.flatnonzero(valid & (value >= target))
    if not index.size:
        return math.nan
    i1 = int(index[0])
    if i1 == 0 or not np.isfinite(value[i1 - 1]) or value[i1 - 1] >= target:
        return float(time[i1])
    i0 = i1 - 1
    if abs(value[i1] - value[i0]) < 1.0e-12:
        return float(time[i1])
    fraction = np.clip((target - value[i0]) / (value[i1] - value[i0]), 0.0, 1.0)
    return float(time[i0] + fraction * (time[i1] - time[i0]))


def experiment_event_times(
    experiment: np.ndarray, kind: str, targets: list[float]
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for target in targets:
        run_times: list[float] = []
        for run in sorted(set(experiment["run"])):
            mask = (
                (experiment["kind"] == kind)
                & (experiment["role"] == "rising_track")
                & (experiment["run"] == run)
            )
            y = np.asarray(experiment["Ystar"][mask], dtype=float)
            t = np.asarray(experiment["Tstar"][mask], dtype=float)
            order = np.argsort(y)
            y = y[order]
            t = t[order]
            if y.size >= 2 and target >= y[0] and target <= y[-1]:
                run_times.append(float(np.interp(target, y, t)))
        rows.append(
            {
                "kind": kind,
                "target_Ystar": target,
                "experiment_run_count": len(run_times),
                "experiment_Tstar_median": (
                    float(np.median(run_times)) if run_times else math.nan
                ),
                "experiment_Tstar_min": min(run_times) if run_times else math.nan,
                "experiment_Tstar_max": max(run_times) if run_times else math.nan,
            }
        )
    return rows


def rollback_counts(
    time: np.ndarray, yint: np.ndarray, window_end: float
) -> dict[str, int]:
    active = np.isfinite(yint) & (time >= 3.5) & (time <= window_end)
    values = yint[active]
    if values.size < 2:
        return {"large_negative_step_dY_lt_minus_0p03": 0, "return_to_zero": 0}
    delta = np.diff(values)
    return {
        "large_negative_step_dY_lt_minus_0p03": int(np.count_nonzero(delta < -0.03)),
        "return_to_zero": int(
            np.count_nonzero((values[:-1] > 0.03) & (values[1:] <= 0.01))
        ),
    }


def vtk_connected_core_cross_check(
    target_time_s: float, model_time_s: np.ndarray, model_yint: np.ndarray
) -> dict[str, float | int | str]:
    """Cross-check the probe observer against one archived 2-D field snapshot."""
    series_path = OPENFOAM_CASE / "VTK" / "2d.vtm.series"
    series = json.loads(series_path.read_text(encoding="utf-8"))
    entry = min(series["files"], key=lambda row: abs(float(row["time"]) - target_time_s))
    vtm_name = Path(entry["name"])
    vtu_path = OPENFOAM_CASE / "VTK" / vtm_name.stem / "internal.vtu"
    root = ET.parse(vtu_path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise RuntimeError(f"No Piece element in {vtu_path}")

    def array(parent: str, name: str | None = None) -> np.ndarray:
        element = (
            piece.find(f'./{parent}/DataArray[@Name="{name}"]')
            if name is not None
            else piece.find(f"./{parent}/DataArray")
        )
        if element is None or element.text is None:
            raise RuntimeError(f"Missing {parent}/{name} in {vtu_path}")
        return np.fromstring(element.text, sep=" ")

    points = array("Points").reshape(-1, 3)
    connectivity = array("Cells", "connectivity").astype(int)
    offsets = array("Cells", "offsets").astype(int)
    alpha_cell = array("CellData", "alpha.water")
    starts = np.r_[0, offsets[:-1]]
    centres = np.asarray(
        [points[connectivity[start:end]].mean(axis=0) for start, end in zip(starts, offsets)]
    )

    tower_width = DT**2 / 0.094
    x_centre = 3.516
    in_tower = (
        (centres[:, 0] >= x_centre - 0.5 * tower_width - 1.0e-9)
        & (centres[:, 0] <= x_centre + 0.5 * tower_width + 1.0e-9)
        & (centres[:, 1] >= CROWN_Y - 1.0e-9)
        & (centres[:, 1] <= 0.957 + 1.0e-9)
    )
    x = np.unique(np.round(centres[in_tower, 0], 12))
    y = np.unique(np.round(centres[in_tower, 1], 12))
    field = np.full((y.size, x.size), np.nan)
    for cx, cy, value in zip(
        centres[in_tower, 0], centres[in_tower, 1], alpha_cell[in_tower]
    ):
        row = int(np.argmin(np.abs(y - cy)))
        column = int(np.argmin(np.abs(x - cx)))
        field[row, column] = value
    if np.any(~np.isfinite(field)):
        raise RuntimeError("Incomplete structured tower field in VTK cross-check")

    gas = field < ALPHA_THRESHOLD
    connected = np.zeros_like(gas, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for column in range(x.size):
        if gas[0, column]:
            connected[0, column] = True
            queue.append((0, column))
    while queue:
        row, column = queue.popleft()
        for drow, dcolumn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = row + drow
            cc = column + dcolumn
            if (
                0 <= rr < y.size
                and 0 <= cc < x.size
                and gas[rr, cc]
                and not connected[rr, cc]
            ):
                connected[rr, cc] = True
                queue.append((rr, cc))

    centre_columns = np.argsort(np.abs(x - x_centre))[:2]
    crossings: list[float] = []
    for column in centre_columns:
        rows = np.flatnonzero(connected[:, column])
        if not rows.size:
            continue
        top = int(rows[-1])
        if top + 1 < y.size:
            crossings.append(
                alpha_crossing(
                    y[top], y[top + 1], field[top, column], field[top + 1, column]
                )
            )
        else:
            crossings.append(float(y[top]))
    field_yint = (
        (float(np.mean(crossings)) - CROWN_Y) / L_TOWER if crossings else math.nan
    )
    vtk_time = float(entry["time"])
    probe_yint = float(np.interp(vtk_time, model_time_s, model_yint))
    return {
        "vtk_file": str(vtu_path.relative_to(CASE_ROOT)),
        "vtk_time_s": vtk_time,
        "vtk_Tstar": vtk_time * TIME_SCALE,
        "tower_grid_rows": int(y.size),
        "tower_grid_columns": int(x.size),
        "field_connected_core_Yint_star": field_yint,
        "probe_connected_core_Yint_star_interpolated": probe_yint,
        "probe_minus_field_Yint_star": probe_yint - field_yint,
    }


def json_safe(value: object) -> object:
    """Return strict-JSON data, mapping non-finite floats to null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    alpha_paths = probe_files("towerCentreline", "alpha.water")
    probe_y = read_probe_y(alpha_paths[0])
    tower = read_probe("towerCentreline", "alpha.water")
    time_s = tower[:, 0]
    tstar = time_s * TIME_SCALE
    alpha = tower[:, 1:]
    if alpha.shape[1] != probe_y.size:
        raise RuntimeError(
            f"Probe header has {probe_y.size} positions but data has {alpha.shape[1]} values"
        )

    legacy = np.asarray([legacy_levels(row, probe_y) for row in alpha])
    connected = np.asarray([connected_core_levels(row, probe_y) for row in alpha])
    yint_legacy = legacy[:, 0]
    yfs_legacy_raw = legacy[:, 1]
    yint = connected[:, 0]
    yfs_raw = connected[:, 1]
    yfs_compare = np.minimum(yfs_raw, 1.0)

    experiment = np.genfromtxt(
        DATA / "fig8_caseB_levels_runs_v2.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    masks = {
        kind: (experiment["kind"] == kind)
        & (experiment["role"] == "rising_track")
        for kind in ("int", "fs")
    }

    event_rows = experiment_event_times(
        experiment, "int", [0.02, 0.05, 0.10, 0.50, 0.80, 0.95]
    )
    # The simulated free surface first exceeds 0.85 during its early filling
    # history, before the Fig. 8 rising branch.  The unambiguous comparable
    # free-surface event is therefore arrival at the physical rim only.
    event_rows += experiment_event_times(experiment, "fs", [1.00])
    for event in event_rows:
        series = yint if event["kind"] == "int" else yfs_compare
        model_time = first_upcrossing(tstar, series, float(event["target_Ystar"]))
        event["connected_core_Tstar"] = model_time
        median = float(event["experiment_Tstar_median"])
        event["connected_core_minus_experiment_median_Tstar"] = (
            model_time - median
            if np.isfinite(model_time) and np.isfinite(median)
            else math.nan
        )

    rising_end = first_upcrossing(tstar, yint, 0.95)
    if not np.isfinite(rising_end):
        rising_end = 4.55

    # Locate the largest old/new discrepancy while experimental Yint rising
    # observations still exist, then use the nearest archived field for an
    # independent 2-D connected-component check.
    int_obs_max_tstar = float(np.max(experiment["Tstar"][masks["int"]]))
    disagreement = np.abs(yint - yint_legacy)
    audit_window = (
        (tstar >= 3.5)
        & (tstar <= int_obs_max_tstar)
        & np.isfinite(disagreement)
    )
    audit_index = int(np.nanargmax(np.where(audit_window, disagreement, np.nan)))
    field_cross_check = vtk_connected_core_cross_check(
        float(time_s[audit_index]), time_s, yint
    )

    metrics = {
        "case": "VW2011 Test 1 Case B",
        "observer_status": "formal",
        "promoted_to_formal_evidence": True,
        "source_probe": str(alpha_paths[0].relative_to(CASE_ROOT)),
        "source_experiment": "data/digitized/fig8_caseB_levels_runs_v2.csv",
        "time_shift_Tstar": 0.0,
        "time_definition": "Tstar = time_s * sqrt(g*Dt)/L_tower",
        "observer": {
            "definition": (
                "alpha.water=0.5 lower edge of the first persistent wet run above "
                "the inlet; this is the top of the inlet-connected gas core"
            ),
            "alpha_threshold": ALPHA_THRESHOLD,
            "minimum_persistent_wet_probes": MIN_PERSISTENT_WET_PROBES,
            "probe_spacing_m_median": float(np.median(np.diff(probe_y))),
            "shorter_wet_runs_below_interface": "classified as detached droplets",
            "free_surface_for_Fig8": "upper edge of the same coherent liquid run; capped at Y*=1 only for Fig.8 comparison",
        },
        "simulation_end_s": float(time_s[-1]),
        "simulation_end_Tstar": float(tstar[-1]),
        "fig8_rising_track_no_time_shift": {
            "connected_core_Yint": comparison_errors(
                tstar,
                yint,
                experiment["Tstar"][masks["int"]],
                experiment["Ystar"][masks["int"]],
            ),
            "legacy_Yint": comparison_errors(
                tstar,
                yint_legacy,
                experiment["Tstar"][masks["int"]],
                experiment["Ystar"][masks["int"]],
            ),
            "connected_core_Yfs": comparison_errors(
                tstar,
                yfs_compare,
                experiment["Tstar"][masks["fs"]],
                experiment["Ystar"][masks["fs"]],
            ),
            "legacy_Yfs": comparison_errors(
                tstar,
                np.minimum(yfs_legacy_raw, 1.0),
                experiment["Tstar"][masks["fs"]],
                experiment["Ystar"][masks["fs"]],
            ),
        },
        "rollback_diagnostic_rising_branch": {
            "window_Tstar": [3.5, rising_end],
            "connected_core": rollback_counts(tstar, yint, rising_end),
            "legacy": rollback_counts(tstar, yint_legacy, rising_end),
        },
        "largest_probe_operator_disagreement_within_Fig8_window": {
            "time_s": float(time_s[audit_index]),
            "Tstar": float(tstar[audit_index]),
            "connected_core_Yint_star": float(yint[audit_index]),
            "legacy_Yint_star": float(yint_legacy[audit_index]),
            "detached_wet_probes_below": int(connected[audit_index, 5]),
        },
        "archived_2d_field_connectivity_cross_check": field_cross_check,
        "events": event_rows,
        "interpretation": (
            "The observer removes false downward interface excursions caused by "
            "detached bottom droplets. Any remaining timing bias is a simulation "
            "discrepancy and is not corrected by shifting the time axis."
        ),
    }

    with (OUT / "levels_connected_core.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "Tstar",
                "Yint_star_connected_core",
                "Yfs_star_coherent_raw",
                "Yfs_star_Fig8_capped",
                "Yint_star_legacy_first_wet",
                "Yfs_star_legacy_last_wet_raw",
                "selected_start_probe",
                "selected_end_probe",
                "persistent_wet_probe_count",
                "detached_wet_probes_below",
            ]
        )
        for index in range(time_s.size):
            writer.writerow(
                [
                    time_s[index],
                    tstar[index],
                    yint[index],
                    yfs_raw[index],
                    yfs_compare[index],
                    yint_legacy[index],
                    yfs_legacy_raw[index],
                    int(connected[index, 2]),
                    int(connected[index, 3]),
                    int(connected[index, 4]),
                    int(connected[index, 5]),
                ]
            )

    # Exact-schema companion used by the formal manuscript plotting script.
    # The field solution is unchanged; this file freezes only the observer.
    with (OUT / "openfoam_2d_levels_connected_core.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "Tstar", "Yint_star", "Yfs_star"])
        writer.writerows(zip(time_s, tstar, yint, yfs_raw))

    with (OUT / "event_times_connected_core.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = list(event_rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(event_rows)

    (OUT / "metrics_connected_core.json").write_text(
        json.dumps(json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8"
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True, sharey=True)
    marker_map = {1: "D", 2: "s", 3: "o"}
    for axis, kind, candidate_y, legacy_y, title in (
        (axes[0], "int", yint, yint_legacy, r"Gas--water interface, $Y^*_{int}$"),
        (
            axes[1],
            "fs",
            yfs_compare,
            np.minimum(yfs_legacy_raw, 1.0),
            r"Free surface, $Y^*_{fs}$",
        ),
        ):
        for run in (1, 2, 3):
            mask = masks[kind] & (experiment["run"] == run)
            axis.scatter(
                experiment["Tstar"][mask],
                experiment["Ystar"][mask],
                marker=marker_map[run],
                facecolors="none",
                edgecolors="0.25",
                linewidths=0.8,
                s=22,
                label=f"Experiment run {run}",
                zorder=4,
            )
        stop_tstar = (
            rising_end
            if kind == "int"
            else first_upcrossing(tstar, yfs_compare, 1.0)
        )
        plot_mask = np.isfinite(candidate_y) & (tstar <= stop_tstar + 1.0e-12)
        legacy_plot_mask = np.isfinite(legacy_y) & (tstar <= stop_tstar + 1.0e-12)
        axis.plot(
            tstar[legacy_plot_mask],
            legacy_y[legacy_plot_mask],
            color="0.62",
            lw=0.9,
            ls=":",
            label="Legacy first/last-wet observer",
        )
        axis.plot(
            tstar[plot_mask],
            candidate_y[plot_mask],
            color="#2166ac",
            lw=1.7,
            label="Connected-core observer",
        )
        axis.set_title(title)
        axis.set_xlim(3.45, 4.52)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(r"$T^*$")
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$Y^*$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.99, wspace=0.12)
    fig.savefig(OUT / "levels_connected_core.png", dpi=300)
    fig.savefig(OUT / "levels_connected_core.pdf")
    plt.close(fig)

    # A profile audit at the largest legacy/candidate disagreement documents
    # exactly why the old first-wet rule fails.
    audit_alpha = np.clip(alpha[audit_index], 0.0, 1.0)
    start = int(connected[audit_index, 2])
    end = int(connected[audit_index, 3])
    fig, axis = plt.subplots(figsize=(4.4, 5.2))
    axis.plot(audit_alpha, (probe_y - CROWN_Y) / L_TOWER, "-o", ms=2.5, lw=1.0)
    axis.axvline(ALPHA_THRESHOLD, color="0.25", ls="--", lw=0.9)
    if start >= 0:
        axis.axhspan(
            (probe_y[start] - CROWN_Y) / L_TOWER,
            (probe_y[end] - CROWN_Y) / L_TOWER,
            color="#2166ac",
            alpha=0.14,
            label="selected persistent liquid run",
        )
    axis.axhline(yint_legacy[audit_index], color="#b2182b", ls=":", lw=1.2, label="legacy $Y^*_{int}$")
    axis.axhline(yint[audit_index], color="#2166ac", lw=1.2, label="connected-core $Y^*_{int}$")
    axis.set(
        xlabel=r"$\alpha_w$",
        ylabel=r"$Y^*$",
        xlim=(-0.03, 1.03),
        ylim=(0.0, 1.05),
        title=rf"Profile audit at $T^*={tstar[audit_index]:.3f}$",
    )
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "connected_core_profile_audit.png", dpi=300)
    fig.savefig(OUT / "connected_core_profile_audit.pdf")
    plt.close(fig)

    print(json.dumps(json_safe(metrics), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
