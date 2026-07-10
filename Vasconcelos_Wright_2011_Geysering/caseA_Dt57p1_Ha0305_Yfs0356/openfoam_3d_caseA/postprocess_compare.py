"""Post-process the 3-D Case A probes and explicitly detect above-rim water."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_A = HERE.parent
OUT = HERE / "outputs"
TOWER_RIM_Y = 0.657
PLUME_Y = np.arange(0.672, 1.853, 0.020)


def load_common_postprocessor():
    source = HERE.parent / "openfoam_2d_caseA" / "postprocess_compare.py"
    spec = importlib.util.spec_from_file_location("case_a_common_postprocess", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared postprocessor from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.CASE_A = CASE_A
    module.OUT = OUT
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common_postprocessor()
    common.main()

    for suffix in (
        "series.csv",
        "levels.csv",
        "pressure_comparison.png",
        "pressure_comparison.pdf",
        "levels_comparison.png",
        "levels_comparison.pdf",
    ):
        (OUT / f"openfoam_2d_{suffix}").replace(OUT / f"openfoam_3d_{suffix}")

    plume = common.read_probe("plumeCentreline", "alpha.water")
    time = plume[:, 0]
    alpha = plume[:, 1:]
    if alpha.shape[1] != len(PLUME_Y):
        raise RuntimeError(
            f"Expected {len(PLUME_Y)} plume probes, found {alpha.shape[1]}"
        )

    highest_water_y = np.full(len(time), np.nan)
    for row, profile in enumerate(alpha):
        wet = np.where(profile >= 0.05)[0]
        if wet.size:
            highest_water_y[row] = PLUME_Y[wet[-1]]

    water_above_rim = bool(np.any(np.isfinite(highest_water_y)))
    if water_above_rim:
        max_height = float(np.nanmax(highest_water_y) - TOWER_RIM_Y)
    else:
        max_height = 0.0

    with (OUT / "openfoam_3d_plume.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "highest_water_y_m", "height_above_rim_m"])
        for sample_time, elevation in zip(time, highest_water_y):
            height = elevation - TOWER_RIM_Y if np.isfinite(elevation) else np.nan
            writer.writerow([sample_time, elevation, height])

    metrics_path = OUT / "openfoam_2d_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "geometry_model": "3-D circular pipe and circular tower",
            "circular_tower_to_pipe_area_ratio": (0.0571 / 0.094) ** 2,
            "plume_domain_height_above_rim_m": 1.2,
            "water_detected_above_rim": water_above_rim,
            "max_sampled_water_height_above_rim_m": max_height,
            "geysering": water_above_rim,
            "geyser_detection": (
                "alpha.water >= 0.05 on the external-atmosphere centreline; "
                "20 mm vertical sampling"
            ),
            "caveat": (
                "Circular 3-D apparatus with an external atmosphere. No event-time "
                "shift or fitted wall/turbulence parameters were applied."
            ),
        }
    )
    metrics_path.replace(OUT / "openfoam_3d_metrics.json")
    (OUT / "openfoam_3d_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
