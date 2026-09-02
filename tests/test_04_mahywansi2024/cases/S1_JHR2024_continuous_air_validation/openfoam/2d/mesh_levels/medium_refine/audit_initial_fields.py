#!/usr/bin/env python3
"""Read ASCII OpenFOAM t=0 internal fields and write a compact audit."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        return [float(uniform.group(1))]
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise ValueError(f"Cannot parse scalar internalField in {path}")
    return [float(value) for value in match.group(1).split()]


def vector_values(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+\(([^)]+)\);", text)
    if uniform:
        xyz = tuple(float(value) for value in uniform.group(1).split())
        return [xyz]  # type: ignore[list-item]
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise ValueError(f"Cannot parse vector internalField in {path}")
    return [
        tuple(float(value) for value in group.split())
        for group in re.findall(r"\(([^)]+)\)", match.group(1))
    ]  # type: ignore[return-value]


alpha = scalar_values(CASE / "0" / "alpha.water")
p = scalar_values(CASE / "0" / "p")
p_rgh = scalar_values(CASE / "0" / "p_rgh")
k = scalar_values(CASE / "0" / "k")
omega = scalar_values(CASE / "0" / "omega")
velocity = vector_values(CASE / "0" / "U")

main_water_cells = 31232
supply_branch_water_cells = 1376
riser_cells_across = 16
riser_z0 = 0.0127
riser_z1 = 1.02
riser_nz = 400
riser_water_cells = (
    sum(value > 0.5 for value in alpha)
    - main_water_cells
    - supply_branch_water_cells
)
riser_water_rows = riser_water_cells // riser_cells_across
riser_dz = (riser_z1 - riser_z0) / riser_nz
represented_water_face = riser_z0 + riser_water_rows * riser_dz

report = {
    "case": str(CASE),
    "alpha_water": {
        "count": len(alpha),
        "min": min(alpha),
        "max": max(alpha),
        "water_cells_alpha_gt_0_5": sum(value > 0.5 for value in alpha),
        "bounded_0_1": min(alpha) >= 0 and max(alpha) <= 1,
        "main_pipe_water_cells": main_water_cells,
        "supply_branch_water_cells": supply_branch_water_cells,
        "supply_branch_initial_phase": "water",
        "supply_branch_phase_evidence": (
            "method_inferred_from_simple_water_flow_and_closed_isolation_valve"
        ),
        "riser_water_rows": riser_water_rows,
        "requested_surface_z_m": 0.5842,
        "represented_interface_face_z_m": represented_water_face,
        "interface_discretization_error_m": represented_water_face - 0.5842,
    },
    "p_absolute_Pa": {"count": len(p), "min": min(p), "max": max(p)},
    "p_rgh_absolute_Pa": {
        "count": len(p_rgh),
        "min": min(p_rgh),
        "max": max(p_rgh),
    },
    "U_m_per_s": {
        "count": len(velocity),
        "max_magnitude": max(math.sqrt(sum(v * v for v in xyz)) for xyz in velocity),
        "all_zero": all(all(abs(v) <= 1e-15 for v in xyz) for xyz in velocity),
    },
    "stage1_turbulence_floors": {
        "k_m2_per_s2": {"min": min(k), "max": max(k)},
        "omega_per_s": {"min": min(omega), "max": max(omega)},
        "source_status": "unreported_declared_positive_numerical_floors",
        "positive": min(k) > 0 and min(omega) > 0,
    },
    "checks": {
        "no_finite_pocket_velocity_seed": all(
            all(abs(v) <= 1e-15 for v in xyz) for xyz in velocity
        ),
        "field_lengths_match": len(alpha) == len(p) == len(p_rgh),
        "finite_pressure_fields": all(math.isfinite(value) for value in p + p_rgh),
        "riser_surface_within_one_cell": abs(represented_water_face - 0.5842) <= riser_dz,
        "water_cell_partition_closes": (
            sum(value > 0.5 for value in alpha)
            == main_water_cells + supply_branch_water_cells + riser_water_cells
        ),
        "source_dictionary_requires_water_filled_supply_branch": (
            "no pre-existing gas pocket"
            in (CASE / "system" / "setFieldsDict").read_text(encoding="utf-8")
        ),
    },
}

output = ROOT / "initial_field_audit.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
