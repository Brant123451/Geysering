#!/usr/bin/env python3
"""Run the frozen H1 1D model with the axial layout drawn in Fig. 1(b)."""
from __future__ import annotations

import json
from pathlib import Path

import caseA_run_and_compare as base


CASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = CASE_ROOT / "outputs" / "paper_layout_1d"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.CASE_KW = dict(
        D=0.05,
        Dr=0.016,
        riser_height=1.8,
        L_up=3.47,
        L_mid=2.51,
        L_down=0.61,
        x_riser_at=3.47,
        pocket_downstream=True,
        reservoir_head=0.66,
        air_head=0.0,
        init_water_level=0.66,
        Hop_cap=10.0,
        x_transducer_at=6.44,
    )
    base.OUT = OUTPUT
    base.build_report = lambda metrics: None
    base.main()

    metrics_path = OUTPUT / "caseA_comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["variant"] = "paper_Fig1b_axial_layout"
    metrics["notes"][0] = (
        "Geometry-matched comparison variant: tee x=3.47 m and selected "
        "release valve x=5.98 m from Fig. 1(b), with L0=0.61 m to the "
        "sealed cap at x=6.59 m. No parameter was fitted to the H1 outcome."
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
