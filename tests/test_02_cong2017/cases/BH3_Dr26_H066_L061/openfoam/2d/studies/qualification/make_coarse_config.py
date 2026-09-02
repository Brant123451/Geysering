#!/usr/bin/env python3
"""Create a declared coarse screening config from the frozen paper config."""
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
    cfg["simulation"]["primary_run_id"] = "coarse_full_linearUpwind_screen"
    cfg["simulation"]["end_time_s"] = 10.5
    cfg["mesh_m"] = {
        "pipe_dx": 0.00625,
        "pipe_dz": 0.004166666666666667,
        "riser_dx": 0.00338,
        "riser_dz": 0.0045,
        "external_dx": 0.012,
        "external_dz": 0.012,
    }
    cfg["mesh_min_cells"] = {
        "pipe_z": 12,
        "pipe_left_x": 40,
        "riser_x": 8,
        "pipe_mid_x": 40,
        "pocket_x": 20,
        "riser_z": 100,
        "external_x": 6,
        "external_z": 40,
    }
    cfg["qualification_metadata"] = {
        "role": "exploratory_coarse_full_process_screen",
        "starts_from_s": 0.0,
        "changed_numerics": {
            "mesh": "declared coarse screening mesh",
            "div(rhoPhi,U)": "Gauss linearUpwind grad(U)",
            "maxCo": 0.4,
            "maxAlphaCo": 0.3,
            "maxDeltaT_s": 0.003,
        },
        "unchanged_physics": [
            "paper geometry",
            "initial and boundary conditions",
            "valve law",
            "materials",
            "Courant limits",
        ],
        "evidence_status": "exploratory_not_manuscript_evidence",
    }
    args.output.write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
