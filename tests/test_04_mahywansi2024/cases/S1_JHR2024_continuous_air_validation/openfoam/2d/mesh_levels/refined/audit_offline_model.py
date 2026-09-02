#!/usr/bin/env python3
"""Static source and initial-condition audit; does not launch OpenFOAM."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def read(relative: str) -> str:
    return (CASE / relative).read_text(encoding="utf-8")


block = read("system/blockMeshDict")
fv_schemes = read("system/fvSchemes")
fv_solution = read("system/fvSolution")
stage1_control = read("system/controlDict.stage1")
set_fields = read("system/setFieldsDict")
set_expr = read("system/setExprFieldsDict")
u0 = read("0/U")
k0 = read("0/k")
omega0 = read("0/omega")
stage1_turb = read("constant/turbulenceProperties.stage1")
stage2_turb = read("constant/turbulenceProperties.stage2")

cell_blocks = [
    (int(nx), int(ny), int(nz))
    for nx, ny, nz in re.findall(
        r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", block
    )
]
cell_count = sum(nx * ny * nz for nx, ny, nz in cell_blocks)

checks = {
    "geometry_upstream_x_minus_1p83": "(-1.8300" in block,
    "geometry_air_tee_x_minus_1p52": "air tee centre x = -1.52 m" in block,
    "geometry_riser_x_zero": "(-0.0127" in block and "( 0.0127" in block,
    "geometry_downstream_x_plus_1p27": "( 1.2700" in block,
    "geometry_riser_rim_z_1p02": "1.0200" in block,
    "mesh_expected_243646_cells": cell_count == 243646,
    "mesh_pipe_32_cells_per_D": "(374 1 32)" in block and "(1882 1 32)" in block,
    "initial_water_surface_0p5842": "0.5842" in set_fields and "0.5842" in set_expr,
    "stage1_supply_branch_water_filled": (
        "no pre-existing gas pocket" in set_fields
        and "volScalarFieldValue alpha.water 1" in set_fields.split(
            "no pre-existing gas pocket", 1
        )[1]
        and "closedWaterStubPressure" in set_expr
        and "closedAirStubPressure" not in set_expr
        and "107026.772153" not in set_fields
    ),
    "initial_U_zero": "internalField   uniform (0 0 0);" in u0,
    "setFields_U_zero": "volVectorFieldValue U (0 0 0)" in set_fields,
    "continuous_air_k_is_declared_floor": "unreported numerical floor" in k0.lower()
        or "declared positive SST floor" in k0,
    "continuous_air_omega_is_declared_floor": "unreported numerical floor" in omega0.lower()
        or "declared positive SST floor" in omega0,
    "finite_pocket_0p06_not_in_U_or_setFields": "0.06" not in u0 and "0.06" not in set_fields,
    "stage1_SST": "kOmegaSST" in stage1_turb,
    "stage2_Smagorinsky": "Smagorinsky" in stage2_turb,
    "time_CrankNicolson_0p9": "CrankNicolson 0.9" in fv_schemes,
    "time_backward_absent": not re.search(r"^\s*default\s+backward", fv_schemes, re.M),
    "alpha_correctors_2": re.search(r"nAlphaCorr\s+2;", fv_solution) is not None,
    "alpha_subcycles_1": re.search(r"nAlphaSubCycles\s+1;", fv_solution) is not None,
    "alpha_limiter_iterations_5": re.search(r"nLimiterIter\s+5;", fv_solution) is not None,
    "stage1_control_is_public_template": re.search(r"endTime\s+3\.0;", stage1_control) is not None,
    "table1_water_inlet_head": "107064.462144" in read("0/p_rgh"),
    "table1_water_outlet_head": "107044.873536" in read("0/p_rgh"),
    "stage2_air_supply_5700Pa_gauge": "107025" in read("prepare_stage2.sh"),
}

report = {
    "audit_type": "offline_static_no_OpenFOAM_process",
    "case": str(CASE),
    "derived_2d_cell_count": cell_count,
    "block_count": len(cell_blocks),
    "checks": checks,
    "passed": all(checks.values()),
    "provenance": {
        "U": "unreported continuous-air seed; declared zero for Stage 1",
        "k": "unreported positive numerical floor; not a paper initial field",
        "omega": "unreported positive numerical floor; not a paper initial field",
        "finite_pocket_U_0p06_transferred": False,
        "stage1_supply_branch_phase": (
            "method-inferred water from simple-water settling and closed "
            "isolation boundary; no pre-existing gas pocket"
        ),
    },
    "execution_gate": {
        "blockMesh_checkMesh_initial_fields_authorized_after_load_gate": True,
        "solver_authorized": False,
        "future_smoke_cap_s": 0.02,
        "note": "The frozen public smoke launcher is not used until CASE3 updates or separately authorizes a capped launcher.",
    },
}
(ROOT / "offline_model_audit.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
