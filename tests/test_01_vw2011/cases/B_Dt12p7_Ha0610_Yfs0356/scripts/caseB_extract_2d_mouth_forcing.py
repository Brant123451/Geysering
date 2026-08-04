#!/usr/bin/env python3
"""Extract Case-B 2-D riser-mouth forcing without rerunning OpenFOAM.

The archived OpenFOAM solution is decomposed.  This script maps the ten cells
immediately below the physical rim back to their processor-local field values,
then reads the binary ``alpha.water`` and ``U`` files directly.  It therefore
avoids reconstructing or converting every saved time directory and does not
interfere with the separate ``2d_external_plume`` run.

Two data products are deliberately kept separate:

* ``*_raw.csv`` is a faithful extraction of every archived field time.
* ``*_sanitized.csv`` is the one-way local-plume input.  It excludes the known
  failed terminal step, bounds alpha to [0, 1], and zeros only sub-threshold
  numerical phase traces.  It does not smooth or clip velocity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
OF_DIR = CASE_DIR / "openfoam" / "2d"
OUT_DIR = CASE_DIR / "outputs"

PIPE_D = 0.094
RISER_D = 0.0127
AREA_EQ_SLIT_WIDTH = RISER_D**2 / PIPE_D
COMPUTATIONAL_X_LEFT = 3.5151420213
COMPUTATIONAL_X_RIGHT = 3.5168579787
SLIT_WIDTH = COMPUTATIONAL_X_RIGHT - COMPUTATIONAL_X_LEFT
RIM_Y = 0.657
WATER_DENSITY = 998.2
ACCEPTED_END_TIME = 8.95
SOURCE_START_TIME = 6.50
ALPHA_TRACE_FLOOR = 1.0e-4
FAILED_STEP_VELOCITY_GUARD = 5.0
TERMINAL_PROBE_FAILURE_GUARD = 10.0


def _data_array(text: str, name: str, section: str | None = None) -> np.ndarray:
    scope = text
    if section:
        match = re.search(rf"<{section}>(.*?)</{section}>", text, flags=re.S)
        if not match:
            raise RuntimeError(f"missing <{section}> in reference VTK")
        scope = match.group(1)
    match = re.search(
        rf"<DataArray[^>]*Name='{re.escape(name)}'[^>]*>(.*?)</DataArray>",
        scope,
        flags=re.S,
    )
    if not match:
        raise RuntimeError(f"missing DataArray {name!r} in reference VTK")
    return np.fromstring(match.group(1), sep=" ", dtype=np.float64)


def reference_cell_centres(vtu: Path) -> np.ndarray:
    text = vtu.read_text(encoding="utf-8", errors="ignore")
    points = _data_array(text, "Points").reshape(-1, 3)
    connectivity = _data_array(text, "connectivity").astype(np.int64)
    offsets = _data_array(text, "offsets").astype(np.int64)
    centres = np.empty((len(offsets), 3), dtype=np.float64)
    start = 0
    for index, end in enumerate(offsets):
        centres[index] = points[connectivity[start:end]].mean(axis=0)
        start = int(end)
    return centres


def find_reference_vtu() -> Path:
    preferred = OF_DIR / "VTK" / "2d_0" / "internal.vtu"
    if preferred.exists():
        return preferred
    candidates = sorted((OF_DIR / "VTK").glob("2d_*/internal.vtu"))
    if not candidates:
        raise FileNotFoundError("no archived ASCII internal.vtu found under openfoam/2d/VTK")
    return candidates[0]


def mouth_global_cells(centres: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x_left = COMPUTATIONAL_X_LEFT
    x_right = COMPUTATIONAL_X_RIGHT
    in_tower = (
        (centres[:, 0] > x_left)
        & (centres[:, 0] < x_right)
        & (centres[:, 1] < RIM_Y)
        & (centres[:, 1] > 0.047)
    )
    if not np.any(in_tower):
        raise RuntimeError("could not locate tower cells in reference VTK")
    sample_y = float(np.max(centres[in_tower, 1]))
    row = np.flatnonzero(in_tower & np.isclose(centres[:, 1], sample_y, atol=2.0e-7))
    row = row[np.argsort(centres[row, 0])]
    if len(row) != 10:
        raise RuntimeError(f"expected 10 cells across the mouth, found {len(row)}")
    dx = SLIT_WIDTH / len(row)
    exact_x = x_left + (np.arange(len(row), dtype=np.float64) + 0.5) * dx
    return row, exact_x, sample_y


def read_label_list(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    matches = list(re.finditer(rb"\n(\d+)\s*\n?\(", raw))
    if not matches:
        raise RuntimeError(f"cannot find binary label list in {path}")
    match = matches[-1]
    count = int(match.group(1))
    values = np.frombuffer(raw, dtype="<i4", count=count, offset=match.end()).copy()
    if len(values) != count:
        raise RuntimeError(f"short label list in {path}")
    return values


def read_internal_field(path: Path, field_type: str, minimum_count: int = 1) -> np.ndarray:
    components = {"scalar": 1, "vector": 3}[field_type]
    raw = path.read_bytes()
    match = re.search(
        rb"internalField\s+nonuniform\s+List<" + field_type.encode() + rb">\s+(\d+)\s*\(",
        raw,
    )
    if not match:
        if field_type == "scalar":
            uniform = re.search(rb"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", raw)
            if uniform:
                return np.full(minimum_count, float(uniform.group(1)), dtype=np.float64)
        else:
            uniform = re.search(
                rb"internalField\s+uniform\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)\s*;",
                raw,
            )
            if uniform:
                vector = np.asarray([float(value) for value in uniform.groups()], dtype=np.float64)
                return np.tile(vector, (minimum_count, 1))
        raise RuntimeError(f"unsupported internal field in {path}")
    count = int(match.group(1))
    values = np.frombuffer(
        raw,
        dtype="<f8",
        count=count * components,
        offset=match.end(),
    ).copy()
    if values.size != count * components:
        raise RuntimeError(f"short binary field in {path}")
    return values.reshape(count, components) if components > 1 else values


def processor_lookup(global_cells: np.ndarray) -> dict[int, tuple[int, int]]:
    wanted = set(int(value) for value in global_cells)
    lookup: dict[int, tuple[int, int]] = {}
    for proc in range(6):
        path = OF_DIR / f"processor{proc}" / "constant" / "polyMesh" / "cellProcAddressing"
        addressing = read_label_list(path)
        for local, global_id in enumerate(addressing):
            global_id = int(global_id)
            if global_id in wanted:
                if global_id in lookup:
                    raise RuntimeError(f"global cell {global_id} occurs on two processors")
                lookup[global_id] = (proc, local)
    missing = wanted.difference(lookup)
    if missing:
        raise RuntimeError(f"mouth cells missing from processor maps: {sorted(missing)}")
    return lookup


def common_field_times() -> list[tuple[float, str]]:
    per_proc: list[dict[float, str]] = []
    for proc in range(6):
        mapping: dict[float, str] = {}
        for path in (OF_DIR / f"processor{proc}").iterdir():
            if not path.is_dir():
                continue
            try:
                time = float(path.name)
            except ValueError:
                continue
            if (path / "alpha.water").exists() and (path / "U").exists():
                mapping[time] = path.name
        per_proc.append(mapping)
    common = set(per_proc[0])
    for mapping in per_proc[1:]:
        common.intersection_update(mapping)
    return [(time, per_proc[0][time]) for time in sorted(common)]


def gather_profiles(
    cells: np.ndarray,
    lookup: dict[int, tuple[int, int]],
) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    by_proc: dict[int, list[tuple[int, int]]] = {}
    for position, global_id in enumerate(cells):
        proc, local = lookup[int(global_id)]
        by_proc.setdefault(proc, []).append((position, local))

    for time, folder in common_field_times():
        alpha = np.empty(len(cells), dtype=np.float64)
        velocity = np.empty((len(cells), 3), dtype=np.float64)
        for proc, requests in by_proc.items():
            time_dir = OF_DIR / f"processor{proc}" / folder
            minimum_count = max(local for _, local in requests) + 1
            alpha_local = read_internal_field(time_dir / "alpha.water", "scalar", minimum_count)
            velocity_local = read_internal_field(time_dir / "U", "vector", minimum_count)
            for position, local in requests:
                alpha[position] = alpha_local[local]
                velocity[position] = velocity_local[local]
        profiles.append({"time": time, "alpha": alpha, "U": velocity})
    return profiles


def summary_row(time: float, alpha: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
    uy = velocity[:, 1]
    mean_alpha = float(np.mean(alpha))
    mean_alpha_uy = float(np.mean(alpha * uy))
    line_flux = SLIT_WIDTH * mean_alpha_uy
    circular_area = math.pi * RISER_D**2 / 4.0
    return {
        "time_s": time,
        "alpha_area_mean": mean_alpha,
        "Uy_area_mean_m_per_s": float(np.mean(uy)),
        "alpha_Uy_area_mean_m_per_s": mean_alpha_uy,
        "alpha_weighted_Uy_m_per_s": (
            mean_alpha_uy / mean_alpha if abs(mean_alpha) > 1.0e-12 else math.nan
        ),
        "water_line_flux_m2_per_s": line_flux,
        "water_line_flux_upward_m2_per_s": SLIT_WIDTH * float(np.mean(alpha * np.maximum(uy, 0.0))),
        "water_line_flux_downward_m2_per_s": SLIT_WIDTH * float(np.mean(alpha * np.minimum(uy, 0.0))),
        "Q_water_circular_equiv_m3_per_s": circular_area * mean_alpha_uy,
        "water_mass_flow_circular_equiv_kg_per_s": WATER_DENSITY * circular_area * mean_alpha_uy,
        "max_abs_U_m_per_s": float(np.max(np.linalg.norm(velocity, axis=1))),
        "wetted_cell_fraction_alpha_ge_0p5": float(np.mean(alpha >= 0.5)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_probe_coordinates(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    pattern = re.compile(r"^# Probe \d+ \(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                coords.append(tuple(float(v) for v in match.groups()))
            elif coords:
                break
    return coords


def parse_scalar_probes(path: Path) -> dict[float, np.ndarray]:
    rows: dict[float, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            values = np.fromstring(line, sep=" ", dtype=np.float64)
            rows[float(values[0])] = values[1:]
    return rows


def parse_vector_probes(path: Path) -> dict[float, np.ndarray]:
    vector_pattern = re.compile(r"\(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)")
    rows: dict[float, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            time = float(line.split(maxsplit=1)[0])
            rows[time] = np.asarray(
                [[float(value) for value in match.groups()] for match in vector_pattern.finditer(line)],
                dtype=np.float64,
            )
    return rows


def write_terminal_probe_diagnostic(sample_y: float) -> dict[str, object]:
    probe_dir = OF_DIR / "postProcessing" / "towerCentreline" / "0"
    alpha_path = probe_dir / "alpha.water"
    velocity_path = probe_dir / "U"
    if not alpha_path.exists() or not velocity_path.exists():
        return {"available": False}
    coords = parse_probe_coordinates(velocity_path)
    centre_index = min(range(len(coords)), key=lambda index: abs(coords[index][1] - sample_y))
    alpha_rows = parse_scalar_probes(alpha_path)
    velocity_rows = parse_vector_probes(velocity_path)
    common = sorted(set(alpha_rows).intersection(velocity_rows))
    output: list[dict[str, object]] = []
    first_failed: float | None = None
    global_peak = 0.0
    for time in common:
        vectors = velocity_rows[time]
        max_abs_uy = float(np.max(np.abs(vectors[:, 1])))
        global_peak = max(global_peak, max_abs_uy)
        # Valid ejection reaches about 7 m/s at a few lower-tower probes.
        # The failed terminal step is qualitatively separate (55.8 m/s), so a
        # 10 m/s diagnostic guard isolates it without labelling the valid peak.
        failed = max_abs_uy > TERMINAL_PROBE_FAILURE_GUARD
        if failed and first_failed is None:
            first_failed = time
        output.append(
            {
                "time_s": time,
                "alpha_near_mouth_centre": float(alpha_rows[time][centre_index]),
                "Uy_near_mouth_centre_m_per_s": float(vectors[centre_index, 1]),
                "max_abs_Uy_all_tower_probes_m_per_s": max_abs_uy,
                "failed_global_step": int(failed),
            }
        )
    path = OUT_DIR / "caseB_2d_tower_probe_terminal_diagnostic_raw.csv"
    write_csv(path, output, list(output[0]))
    return {
        "available": True,
        "path": str(path.relative_to(CASE_DIR)),
        "near_mouth_probe_index": centre_index,
        "near_mouth_probe_y_m": coords[centre_index][1],
        "first_velocity_guard_failure_time_s": first_failed,
        "max_abs_Uy_all_tower_probes_m_per_s": global_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-start", type=float, default=SOURCE_START_TIME)
    parser.add_argument("--accepted-end", type=float, default=ACCEPTED_END_TIME)
    parser.add_argument("--alpha-floor", type=float, default=ALPHA_TRACE_FLOOR)
    args = parser.parse_args()

    reference = find_reference_vtu()
    centres = reference_cell_centres(reference)
    cells, x_coords, sample_y = mouth_global_cells(centres)
    lookup = processor_lookup(cells)
    profiles = gather_profiles(cells, lookup)
    if not profiles:
        raise RuntimeError("no common decomposed field times were found")

    raw_profile_rows: list[dict[str, object]] = []
    raw_summary_rows: list[dict[str, object]] = []
    sanitized_profile_rows: list[dict[str, object]] = []
    sanitized_summary_rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for profile in profiles:
        time = float(profile["time"])
        alpha = np.asarray(profile["alpha"], dtype=np.float64)
        velocity = np.asarray(profile["U"], dtype=np.float64)
        summary = summary_row(time, alpha, velocity)
        raw_summary_rows.append(summary)
        for index, (x, a, vector) in enumerate(zip(x_coords, alpha, velocity)):
            raw_profile_rows.append(
                {
                    "time_s": time,
                    "cell_index_from_left": index,
                    "x_m": float(x),
                    "sample_y_m": sample_y,
                    "alpha_water": float(a),
                    "Ux_m_per_s": float(vector[0]),
                    "Uy_m_per_s": float(vector[1]),
                    "alpha_Uy_m_per_s": float(a * vector[1]),
                }
            )

        reasons: list[str] = []
        if time < args.source_start - 1.0e-12:
            reasons.append("before_source_start")
        if time > args.accepted_end + 1.0e-12:
            reasons.append("after_accepted_end")
        if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(velocity)):
            reasons.append("non_finite")
        if float(summary["max_abs_U_m_per_s"]) > FAILED_STEP_VELOCITY_GUARD:
            reasons.append("velocity_guard")
        if reasons:
            rejected.append({"time_s": time, "reasons": reasons})
            continue

        alpha_clean = np.clip(alpha, 0.0, 1.0)
        alpha_clean[alpha_clean < args.alpha_floor] = 0.0
        clean_values = summary_row(time, alpha_clean, velocity)
        clean_summary = {
            "source_time_s": time,
            "local_time_s": time - args.source_start,
            **{key: value for key, value in clean_values.items() if key != "time_s"},
        }
        # The per-cell trace filter above is the only activation threshold.
        # If any retained water remains, keep the bulk forcing exactly
        # flux-conservative instead of applying a second mean-alpha cutoff.
        clean_summary["forcing_active"] = int(clean_summary["alpha_area_mean"] > 0.0)
        if clean_summary["forcing_active"]:
            clean_summary["forcing_alpha_uniform"] = clean_summary["alpha_area_mean"]
            clean_summary["forcing_Uy_uniform_m_per_s"] = (
                clean_summary["alpha_Uy_area_mean_m_per_s"] / clean_summary["alpha_area_mean"]
            )
        else:
            clean_summary["forcing_alpha_uniform"] = 0.0
            clean_summary["forcing_Uy_uniform_m_per_s"] = 0.0
        sanitized_summary_rows.append(clean_summary)
        for index, (x, a, vector) in enumerate(zip(x_coords, alpha_clean, velocity)):
            sanitized_profile_rows.append(
                {
                    "source_time_s": time,
                    "local_time_s": time - args.source_start,
                    "cell_index_from_left": index,
                    "x_m": float(x),
                    "sample_y_m": sample_y,
                    "alpha_water": float(a),
                    "Ux_m_per_s": float(vector[0]),
                    "Uy_m_per_s": float(vector[1]),
                    "alpha_Uy_m_per_s": float(a * vector[1]),
                }
            )

    raw_summary = OUT_DIR / "caseB_2d_mouth_forcing_raw.csv"
    raw_profile = OUT_DIR / "caseB_2d_mouth_profile_raw.csv"
    clean_summary = OUT_DIR / "caseB_2d_mouth_forcing_sanitized.csv"
    clean_profile = OUT_DIR / "caseB_2d_mouth_profile_sanitized.csv"
    write_csv(raw_summary, raw_summary_rows, list(raw_summary_rows[0]))
    write_csv(raw_profile, raw_profile_rows, list(raw_profile_rows[0]))
    write_csv(clean_summary, sanitized_summary_rows, list(sanitized_summary_rows[0]))
    write_csv(clean_profile, sanitized_profile_rows, list(sanitized_profile_rows[0]))

    terminal_probe = write_terminal_probe_diagnostic(sample_y)
    raw_times = [float(row["time_s"]) for row in raw_summary_rows]
    accepted_times = [float(row["source_time_s"]) for row in sanitized_summary_rows]
    local_times = np.asarray([float(row["local_time_s"]) for row in sanitized_summary_rows])
    line_flux = np.asarray(
        [float(row["water_line_flux_m2_per_s"]) for row in sanitized_summary_rows]
    )
    forcing_identity = np.asarray(
        [
            SLIT_WIDTH
            * float(row["forcing_alpha_uniform"])
            * float(row["forcing_Uy_uniform_m_per_s"])
            for row in sanitized_summary_rows
        ]
    )
    circular_flow = np.asarray(
        [float(row["Q_water_circular_equiv_m3_per_s"]) for row in sanitized_summary_rows]
    )
    peak_up = int(np.argmax(line_flux))
    peak_down = int(np.argmin(line_flux))
    forcing_statistics = {
        "row_count": len(sanitized_summary_rows),
        "sample_interval_s": 0.05,
        "peak_upward_line_flux_m2_per_s": float(line_flux[peak_up]),
        "peak_upward_source_time_s": accepted_times[peak_up],
        "peak_upward_local_time_s": float(local_times[peak_up]),
        "peak_downward_line_flux_m2_per_s": float(line_flux[peak_down]),
        "peak_downward_source_time_s": accepted_times[peak_down],
        "peak_downward_local_time_s": float(local_times[peak_down]),
        "signed_integrated_water_area_per_depth_m2": float(np.trapz(line_flux, local_times)),
        "upward_integrated_water_area_per_depth_m2": float(
            np.trapz(np.maximum(line_flux, 0.0), local_times)
        ),
        "downward_integrated_water_area_per_depth_m2": float(
            np.trapz(np.minimum(line_flux, 0.0), local_times)
        ),
        "signed_circular_equivalent_water_volume_m3": float(
            np.trapz(circular_flow, local_times)
        ),
        "max_abs_q_minus_W_alpha_Uy_m2_per_s": float(
            np.max(np.abs(line_flux - forcing_identity))
        ),
    }
    metadata = {
        "source_case": "Case B archived extended-tower 2-D OpenFOAM result",
        "source_reference_vtu": str(reference.relative_to(CASE_DIR)),
        "sampling": {
            "physical_rim_y_m": RIM_Y,
            "sample_cell_centre_y_m": sample_y,
            "distance_below_rim_m": RIM_Y - sample_y,
            "cell_count_across_slit": len(cells),
            "x_cell_centres_m": [float(value) for value in x_coords],
            "computational_slit_width_m": SLIT_WIDTH,
            "intended_area_equivalent_slit_width_Dt2_over_D_m": AREA_EQ_SLIT_WIDTH,
            "relative_width_difference": SLIT_WIDTH / AREA_EQ_SLIT_WIDTH - 1.0,
        },
        "raw_field_window_s": [min(raw_times), max(raw_times)],
        "accepted_source_forcing_window_s": [min(accepted_times), max(accepted_times)],
        "accepted_local_forcing_window_s": [
            min(accepted_times) - args.source_start,
            max(accepted_times) - args.source_start,
        ],
        "local_time_definition": f"local_time_s = source_time_s - {args.source_start:g} s",
        "recommended_top_only_control_endTime_s": 2.5,
        "known_failed_terminal_step": {
            "time_s_approx": 8.975,
            "policy": "excluded in full; last accepted archived field is t=8.95 s",
            "reason": "tower-probe velocities jump from O(1) to tens of m/s, indicating solver failure",
        },
        "sanitization": {
            "temporal_smoothing": "none",
            "velocity_clipping": "none",
            "row_rejection": f"source time outside [{args.source_start:g},{args.accepted_end:g}] s, non-finite data, or max|U|>{FAILED_STEP_VELOCITY_GUARD:g} m/s",
            "alpha_bounds": "clipped to [0,1] only in sanitized files",
            "alpha_trace_floor": args.alpha_floor,
            "trace_policy": "values below the floor are set to zero only in sanitized files",
            "time_interpolation_recommendation": "piecewise linear in the local-plume boundary condition",
            "rejected_rows": rejected,
        },
        "forcing_statistics": forcing_statistics,
        "definitions": {
            "water_line_flux_m2_per_s": "W * mean_i(alpha_i * Uy_i), signed upward; per unit out-of-plane depth",
            "alpha_area_mean": "mean_i(alpha_i) because all ten slit cells have equal width",
            "alpha_weighted_Uy_m_per_s": "mean_i(alpha_i*Uy_i) / mean_i(alpha_i)",
            "Q_water_circular_equiv_m3_per_s": "pi*Dt^2/4 * mean_i(alpha_i*Uy_i)",
            "water_mass_flow_circular_equiv_kg_per_s": "rho_water * Q_water_circular_equiv",
        },
        "evidence_limitations": [
            "The source is the exploratory planar area-equivalent slit model, not a circular 3-D riser.",
            "Flux is reconstructed from cell-centred alpha and velocity, not the solver's conservative face alphaPhi field.",
            "The sample row is 1.5375 mm below the physical rim because the archived mesh does not place a face exactly at y=0.657 m.",
            "One-way forcing cannot transmit pressure or fallback feedback from the external plume to the lower pipe system.",
        ],
        "terminal_probe_diagnostic": terminal_probe,
        "outputs": {
            "raw_summary": str(raw_summary.relative_to(CASE_DIR)),
            "raw_profile": str(raw_profile.relative_to(CASE_DIR)),
            "sanitized_summary": str(clean_summary.relative_to(CASE_DIR)),
            "sanitized_profile": str(clean_profile.relative_to(CASE_DIR)),
        },
    }
    metadata_path = OUT_DIR / "caseB_2d_mouth_forcing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"mouth row: y={sample_y:.7f} m, {len(cells)} cells")
    print(f"raw field times: {len(raw_times)}, {min(raw_times):g}--{max(raw_times):g} s")
    print(f"accepted forcing times: {len(accepted_times)}, {min(accepted_times):g}--{max(accepted_times):g} s")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
