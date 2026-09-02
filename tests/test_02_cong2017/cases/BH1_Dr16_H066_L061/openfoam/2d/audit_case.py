#!/usr/bin/env python3
"""Fail-fast paper-contract audit for the prepared B-H1 2D case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def close(actual: float, expected: float, tol: float = 1e-10) -> bool:
    return abs(actual - expected) <= tol


def dictionary_scalar(path: Path, key: str) -> float:
    import re

    match = re.search(rf"(?m)^\s*{re.escape(key)}\s+([0-9.eE+-]+)\s*;", path.read_text())
    if not match:
        raise ValueError(f"Missing scalar {key} in {path}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "case_config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    g = cfg["physical_geometry_m"]
    ic = cfg["initial_conditions"]
    pm = cfg["planar_mapping"]

    checks = {
        "D_0p05": close(g["pipe_inner_diameter"], 0.05),
        "Dr_0p016": close(g["riser_inner_diameter"], 0.016),
        "H0_0p66": close(ic["H0_m_above_pipe_invert"], 0.66),
        "L0_0p61": close(ic["pocket_length_m"], 0.61),
        "tee_x_3p47": close(g["tee_axis_x"], 3.47),
        "valve_x_5p98": close(g["release_valve_x"], 5.98),
        "cap_x_6p59": close(g["closed_cap_x"], 6.59),
        "free_surface_z_0p635": close(ic["free_surface_z_m"], 0.635),
        "pocket_atmospheric": close(ic["pocket_pressure_Pa_abs"], 101325.0),
        "area_ratio_mapping": close(
            pm["area_equivalent_riser_width_m"] / g["pipe_inner_diameter"],
            (g["riser_inner_diameter"] / g["pipe_inner_diameter"]) ** 2,
        ),
    }

    control = args.run_dir / "system" / "controlDict"
    sim = cfg["simulation"]
    checks.update(
        {
            "control_end_time": close(dictionary_scalar(control, "endTime"), sim["end_time_s"]),
            "control_max_Courant": close(dictionary_scalar(control, "maxCo"), sim["max_Courant"]),
            "control_max_alpha_Courant": close(
                dictionary_scalar(control, "maxAlphaCo"), sim["max_alpha_Courant"]
            ),
            "control_max_delta_t": close(dictionary_scalar(control, "maxDeltaT"), sim["max_delta_t_s"]),
        }
    )

    required_text = {
        "system/setFieldsDict": ["box (5.98", "0.635", "101325"],
        "constant/valveProperties": ["sineSquaredAreaForchheimer", "openingDuration 0.2", "referenceFlowArea 0.00005"],
        "0.orig/p_rgh": ["107541.89130", "101332.42484"],
    }
    for rel, needles in required_text.items():
        content = (args.run_dir / rel).read_text()
        for needle in needles:
            checks[f"{rel}:{needle}"] = needle in content

    mesh_stats = json.loads((args.run_dir / "mesh_stats.json").read_text())
    checks["mesh_nonempty"] = mesh_stats["cells_total"] > 50000
    checks["standard_checkMesh_pass"] = "Mesh OK." in (args.run_dir / "log.checkMesh").read_text()
    payload = {
        "schema_version": 1,
        "case": cfg["case_id"],
        "paper_run": cfg["paper_run"],
        "run_id": sim["primary_run_id"],
        "config": str(args.config.resolve()),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mesh": mesh_stats,
        "declared_2d_limitation": pm["limitation"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
