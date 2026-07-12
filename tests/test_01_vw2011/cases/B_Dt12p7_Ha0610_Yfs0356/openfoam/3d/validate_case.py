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
    "valve_plane_conformal",
    'setPhysicalName(2, valve_group, "valvePlane")',
    "TOWER_DIAMETER = 0.0127",
    "ATMOSPHERE_HEIGHT = 1.200",
    '"refined"',
]
missing = [fragment for fragment in required_fragments if fragment not in mesh_source]
if missing:
    raise SystemExit(f"Mesh source is missing required fragments: {missing}")

solver_sources = "\n".join(
    (
        (HERE / "system" / "controlDict").read_text(),
        (HERE / "system" / "runSettings.default").read_text(),
        (HERE / "system" / "alphaCorrectors.default").read_text(),
        (HERE / "system" / "pimpleCorrectors.default").read_text(),
        (HERE / "system" / "runControl.default").read_text(),
        (HERE / "system" / "createBafflesDict.hold").read_text(),
        (HERE / "constant" / "thermophysicalProperties").read_text(),
        (HERE / "constant" / "fvOptions").read_text(),
        (HERE / "constant" / "surfaceForces.default").read_text(),
        (HERE / "Allrun").read_text(),
    )
)
required_solver_fragments = [
    "application     compressibleInterFlow",
    "advectionScheme         isoAdvection",
    "reconstructionScheme    plicRDF",
    "iterations              5",
    "tol                     1e-6",
    "interpolateNormal       false",
    "surfaceTensionForceModel    RDF",
    "nAlphaCorr      1",
    "nAlphaSubCycles 2",
    "nOuterCorrectors         1",
    "nCorrectors              2",
    "nNonOrthogonalCorrectors 0",
    "writeControl    runTime",
    "alphaAtMax",
    "CASEB_FORCE_BALANCE",
    "hydrostaticForceResidual",
    "surfaceTensionForce",
    "zoneName    valvePlane",
    'if (mode != "closed")',
    "pureMovingPhaseModel",
    "de9826f9ffb24f4b635ac97fd388ebd560cfc174",
]
missing_solver = [
    fragment
    for fragment in required_solver_fragments
    if fragment not in solver_sources
]
if missing_solver:
    raise SystemExit(
        f"Solver source is missing required fragments: {missing_solver}"
    )

pressure_initialisation = (HERE / "system" / "setExprFieldsDict").read_text()
reduced_pressure_initialisation = (
    HERE / "system" / "setExprFieldsReducedPressureDict"
).read_text()
alpha_initialisation = (HERE / "system" / "setAlphaFieldDict").read_text()
allrun_source = (HERE / "Allrun").read_text()
pressure_call = "setExprFields -dict system/setExprFieldsDict.runtime"
reduced_pressure_call = (
    "setExprFields -dict system/setExprFieldsReducedPressureDict.runtime"
)
if (
    "mixtureReducedPressure" in pressure_initialisation
    or "mixtureReducedPressure" not in reduced_pressure_initialisation
    or "readFields (alpha.water p p_rgh T);" not in reduced_pressure_initialisation
    or pressure_call not in allrun_source
    or reduced_pressure_call not in allrun_source
    or allrun_source.index(pressure_call) > allrun_source.index(reduced_pressure_call)
):
    raise SystemExit(
        "Absolute and reduced pressure must be initialised by ordered, "
        "independent setExprFields processes"
    )
if "radius     0.00735;" not in alpha_initialisation:
    raise SystemExit(
        "Tower-water selector must end inside the solid wall gap so all "
        "tower-fluid boundary cells start fully wet"
    )

result = {
    "case_definition_ok": True,
    "solver_source_check": (
        "compressibleInterFlow at pinned TwoPhaseFlow commit with "
        "isoAdvection, plicRDF reconstruction without normal interpolation, "
        "materialised reconstruction convergence controls, RDF curvature and "
        "TwoPhaseFlow-reference corrector counts; conformal no-slip closed "
        "baffle and runTime output control"
    ),
    "geometry_source_check": (
        "configured 3-D circular Boolean pipe/tower/exterior atmosphere with "
        "a conformal valve-plane face zone; generated mesh remains authoritative"
    ),
    "initial_pressure_source_check": (
        "absolute p is written before a separate process reloads it to construct "
        "mixture-density p_rgh"
    ),
    "initial_alpha_source_check": (
        "tower-water geometric selector extends halfway into the assumed solid "
        "wall and remains 1 mm inside the exterior fluid boundary"
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
