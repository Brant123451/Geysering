#!/usr/bin/env python3
"""Create an intermediate-mesh B-H3 qualification config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.source.read_text())
    cfg["simulation"]["primary_run_id"] = "medium_iso_valve_qualification"
    cfg["simulation"]["end_time_s"] = 13.0
    cfg["mesh_m"] = {
        "pipe_dx": 0.004,
        "pipe_dz": 0.003125,
        "riser_dx": 0.00169,
        "riser_dz": 0.003,
        "external_dx": 0.008,
        "external_dz": 0.008,
    }
    cfg["mesh_min_cells"] = {
        "pipe_z": 16,
        "pipe_left_x": 40,
        "riser_x": 8,
        "pipe_mid_x": 40,
        "pocket_x": 20,
        "riser_z": 100,
        "external_x": 6,
        "external_z": 40,
    }
    cfg["qualification_metadata"] = {
        "role": "intermediate_mesh_isoAdvector_qualification",
        "starts_from_s": 0.0,
        "changed_numerics": {
            "interface_advection": "isoAdvector",
            "energy_equation": "official OpenFOAM v2512 form",
            "mesh": "declared intermediate qualification mesh",
            "div(rhoPhi,U)": "Gauss linearUpwind grad(U)",
            "maxCo": 0.3,
            "maxAlphaCo": 0.25,
            "maxDeltaT_s": 0.0015,
        },
        "unchanged_physics": [
            "paper geometry",
            "initial and boundary conditions",
            "0.2 s passive valve law",
            "materials",
        ],
        "evidence_status": "qualification_not_manuscript_evidence",
    }
    args.output.write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
