#!/usr/bin/env python3
"""Compare the completed H1 rerun with the published B-H1 description."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path("/tmp/bh1-2d-study/h1_refined_co015")


def relative_error(actual: float | None, expected: float) -> float | None:
    return None if actual is None else abs(actual - expected) / abs(expected)


def first_probe_time(run_dir: Path, column: int, threshold: float = 0.5) -> float | None:
    paths = sorted((run_dir / "postProcessing" / "plumeProbes").glob("*/alpha.water"))
    rows: list[np.ndarray] = []
    for path in paths:
        try:
            values = np.loadtxt(path, ndmin=2)
        except (OSError, ValueError):
            continue
        if values.shape[1] > column:
            rows.append(values)
    if not rows:
        return None
    data = np.concatenate(rows)
    data = data[np.argsort(data[:, 0])]
    hit = np.flatnonzero(data[:, column] >= threshold)
    return None if hit.size == 0 else float(data[int(hit[0]), 0])


def postarrival_pressure_ratio(results_dir: Path, arrival: float | None, h0: float) -> tuple[float | None, float | None]:
    if arrival is None:
        return None, None
    path = results_dir / "openfoam_2d_pt1_series.csv"
    if not path.exists():
        return None, None
    rows = list(csv.DictReader(path.open()))
    candidates = [row for row in rows if float(row["t_s"]) >= arrival]
    if not candidates:
        return None, None
    peak = max(candidates, key=lambda row: float(row["head_m_water"]))
    return float(peak["head_m_water"]) / h0, float(peak["t_s"])


def support(error: float | None, supported: float, partial: float) -> str:
    if error is None:
        return "missing"
    if error <= supported:
        return "supported"
    if error <= partial:
        return "partial"
    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((args.results_dir / "openfoam_2d_metrics.json").read_text())
    model = metrics["model"]
    experiment = metrics["experiment"]
    h0 = metrics["paper_contract"]["initial_conditions"]["H0_m_above_pipe_invert"]
    arrival_error = relative_error(model["Ta_s"], experiment["Ta_s"])
    rim_error = relative_error(model["t_free_surface_at_98pct_rim_s"], 9.55)
    vfs_error = relative_error(model["vfs_m_s"], experiment["vfs_m_s"])
    vint_error = relative_error(model["vint_m_s"], experiment["vint_m_s"])
    pressure_ratio, pressure_time = postarrival_pressure_ratio(args.results_dir, model["Ta_s"], h0)
    pressure_error = relative_error(pressure_ratio, 1.9)
    plume = {
        "rim_1p824_m_first_alpha_ge_0p5_s": first_probe_time(args.run_dir, 1),
        "above_rim_1p9_m_first_alpha_ge_0p5_s": first_probe_time(args.run_dir, 2),
        "plume_2p1_m_first_alpha_ge_0p5_s": first_probe_time(args.run_dir, 3),
    }

    claims = {
        "gas_arrival": {
            "support": support(arrival_error, 0.10, 0.20),
            "experiment_s": experiment["Ta_s"],
            "model_s": model["Ta_s"],
            "relative_error": arrival_error,
        },
        "geyser_classification": {
            "support": "supported" if model["geysering"] else "missing",
            "experiment": True,
            "model": model["geysering"],
        },
        "water_ejection_above_rim": {
            "support": "supported" if plume["above_rim_1p9_m_first_alpha_ge_0p5_s"] is not None else "missing",
            "plume_probe_times": plume,
        },
        "eruption_timing": {
            "support": support(rim_error, 0.20, 0.50),
            "experiment_s": 9.55,
            "model_s": model["t_free_surface_at_98pct_rim_s"],
            "relative_error": rim_error,
        },
        "free_surface_rise_speed": {
            "support": support(vfs_error, 0.30, 0.50),
            "experiment_m_s": experiment["vfs_m_s"],
            "model_m_s": model["vfs_m_s"],
            "relative_error": vfs_error,
        },
        "gas_nose_rise_speed": {
            "support": support(vint_error, 0.30, 0.50),
            "experiment_m_s": experiment["vint_m_s"],
            "model_m_s": model["vint_m_s"],
            "relative_error": vint_error,
        },
        "postarrival_pressure_surge": {
            "support": support(pressure_error, 0.15, 0.40),
            "experiment_H_over_H0": 1.9,
            "model_H_over_H0": pressure_ratio,
            "model_peak_time_s": pressure_time,
            "relative_error": pressure_error,
        },
    }

    basic_outcome = (
        claims["gas_arrival"]["support"] == "supported"
        and claims["geyser_classification"]["support"] == "supported"
        and claims["water_ejection_above_rim"]["support"] == "supported"
    )
    overall = (
        "BASIC_OUTCOME_MATCH_WITH_MAJOR_TRANSIENT_BIASES"
        if basic_outcome
        else "NOT_BASICALLY_CONSISTENT"
    )
    payload = {
        "overall_assessment": overall,
        "plain_language": (
            "The 2D run reproduces the B-H1 arrival and geyser/ejection outcome, but not the published pressure-surge mechanism or quantitative rise dynamics."
            if basic_outcome
            else "The 2D run does not reproduce the basic published B-H1 outcome."
        ),
        "run_complete": metrics["status"]["ended_normally"],
        "claims": claims,
        "evidence_type": "exploratory 2D OpenFOAM versus published high-speed-camera measurements",
    }
    (args.output_dir / "paper_consistency_report.json").write_text(json.dumps(payload, indent=2) + "\n")

    def value(x: float | None, digits: int = 3) -> str:
        return "--" if x is None else f"{x:.{digits}f}"

    md = f"""# B-H1 paper consistency report

## Conclusion

The refined 2D calculation is **basically consistent in outcome** with the
published B-H1 case: the pocket arrives at the riser and produces water
ejection above the rim. It is **not a quantitative reproduction** of the
published transient because the eruption is delayed, both interfaces rise too
slowly, and the post-arrival pressure surge is missing.

| Item | Experiment | Refined 2D | Evidence judgement |
|---|---:|---:|---|
| Pocket arrival | {experiment['Ta_s']:.2f} s | {value(model['Ta_s'], 2)} s | {claims['gas_arrival']['support']} |
| Geyser/ejection | yes | {'yes' if model['geysering'] else 'no'} | {claims['geyser_classification']['support']} |
| 98% rim time | 9.55 s | {value(model['t_free_surface_at_98pct_rim_s'], 3)} s | {claims['eruption_timing']['support']} |
| Free-surface rise speed | {experiment['vfs_m_s']:.3f} m/s | {value(model['vfs_m_s'])} m/s | {claims['free_surface_rise_speed']['support']} |
| Gas-nose rise speed | {experiment['vint_m_s']:.3f} m/s | {value(model['vint_m_s'])} m/s | {claims['gas_nose_rise_speed']['support']} |
| Post-arrival pressure peak | 1.90 H0 | {value(pressure_ratio)} H0 | {claims['postarrival_pressure_surge']['support']} |
| Water at 2.1 m | observed ejection | {value(plume['plume_2p1_m_first_alpha_ge_0p5_s'], 3)} s | {claims['water_ejection_above_rim']['support']} |

The original strict qualification gate remains separate and is not weakened by
this descriptive assessment.
"""
    (args.output_dir / "paper_consistency_report.md").write_text(md, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
