#!/usr/bin/env python3
"""Cheap preflight checks for geometry, initial inventories and source scope."""
from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
config = json.loads((CASE_ROOT / "config" / "case.json").read_text())

expected = {
    "tower_diameter_m": 0.0127,
    "initial_air_pressure_head_m": 0.610,
    "initial_water_level_m": 0.356,
    "observed_branch": "geyser",
}
if config != expected:
    raise SystemExit(f"Authoritative case mismatch: {config!r} != {expected!r}")

pipe_diameter = 0.094
pipe_area = math.pi * pipe_diameter**2 / 4
tower_area = math.pi * config["tower_diameter_m"] ** 2 / 4
air_volume = pipe_area * 0.546
water_volume = pipe_area * (4.006 - 0.546) + tower_area * 0.356
pressure_abs = (
    101325
    + 998.2 * 9.81 * config["initial_air_pressure_head_m"]
)
time_scale = 0.610 / math.sqrt(9.81 * config["tower_diameter_m"])

mesh_source = (HERE / "make_mesh.py").read_text()
required_fragments = [
    "occ.addCylinder",
    "TOWER_DIAMETER = 0.0127",
    "ATMOSPHERE_HEIGHT = 1.200",
    '"refined"',
]
missing = [fragment for fragment in required_fragments if fragment not in mesh_source]
if missing:
    raise SystemExit(f"Mesh source is missing required fragments: {missing}")

result = {
    "case_definition_ok": True,
    "geometry_source_check": (
        "configured 3-D circular Boolean pipe/tower/exterior atmosphere; "
        "generated mesh remains authoritative"
    ),
    "pipe_length_m": 4.006,
    "pipe_diameter_m": pipe_diameter,
    "tower_diameter_m": config["tower_diameter_m"],
    "tower_height_above_crown_m": 0.610,
    "tower_rim_y_m": 0.657,
    "exterior_top_y_m": 1.857,
    "exterior_bottom_y_m": 0.257,
    "assumed_tower_wall_thickness_m": 0.002,
    "initial_air_pocket_volume_m3": air_volume,
    "initial_water_volume_m3": water_volume,
    "initial_air_absolute_pressure_Pa": pressure_abs,
    "initial_free_surface_y_m": 0.047 + config["initial_water_level_m"],
    "dimensionless_time_scale_s": time_scale,
    "Tstar_6_physical_time_s": 6 * time_scale,
}
runtime = HERE / "outputs" / "runtime"
runtime.mkdir(parents=True, exist_ok=True)
(runtime / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
