#!/usr/bin/env python3
"""Audit the common source-aligned Stage-1 fields for any 2-D mesh level.

The block order is part of the frozen S1 topology: five horizontal blocks,
then the short supply branch, then the riser.  This lets the audit derive the
expected water-cell partition from each level's own blockMeshDict instead of
hard-coding coarse/medium/refined counts.  It launches no OpenFOAM process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import apply_total_pressure_profiles as total_pressure


WATER_LEVEL_M = 0.5842
RISER_BOTTOM_M = 0.0127
TABLE1_P_RGH_PA = {
    "waterInlet": 107064.462144,
    "waterOutlet": 107044.873536,
}
TABLE1_REWRITE_TOLERANCE_PA = 0.01


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
        raise ValueError(f"cannot parse scalar internalField in {path}")
    return [float(value) for value in match.group(1).split()]


def vector_values(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+\(([^)]+)\);", text)
    if uniform:
        xyz = tuple(float(value) for value in uniform.group(1).split())
        if len(xyz) != 3:
            raise ValueError(f"expected a three-component vector in {path}")
        return [xyz]  # type: ignore[list-item]
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not match:
        raise ValueError(f"cannot parse vector internalField in {path}")
    values = [
        tuple(float(value) for value in group.split())
        for group in re.findall(r"\(([^)]+)\)", match.group(1))
    ]
    if not all(len(value) == 3 for value in values):
        raise ValueError(f"expected three-component vectors in {path}")
    return values  # type: ignore[return-value]


def block_cell_counts(block_mesh: str) -> list[tuple[int, int, int]]:
    return [
        (int(nx), int(ny), int(nz))
        for nx, ny, nz in re.findall(
            r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", block_mesh
        )
    ]


def boundary_scalar_values(
    field_text: str, field_path: Path, patch: str, entry: str = "value"
) -> tuple[str, list[float]]:
    """Return a patch type and a uniform/nonuniform scalar entry.

    The total-pressure converter writes facewise ``nonuniform List<scalar>``
    entries.  Keeping this parser shared with that converter prevents the
    initial-field gate from silently accepting the superseded uniform-static-
    pressure campaign.
    """

    blocks, _ = total_pressure._find_field_patch_blocks(field_text, field_path)
    if patch not in blocks:
        raise ValueError(f"cannot locate boundary patch {patch!r}")
    body = total_pressure._patch_body(field_text, blocks[patch])
    patch_type = total_pressure._single_type(body, f"{field_path}:{patch}")
    values = total_pressure._nonuniform_scalar_entry(
        body,
        entry,
        f"{field_path}:{patch}",
        required=True,
    )
    if not values:
        raise ValueError(f"cannot parse {entry!r} for boundary patch {patch!r}")
    return patch_type, values


def total_pressure_profile_gate(case: Path, p_rgh_text: str) -> dict[str, object]:
    """Cross-check the converted field against its deterministic audit."""

    audit_path = case / "total_pressure_profile_audit.json"
    if not audit_path.is_file():
        raise ValueError(f"missing total-pressure audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    field_path = case / "0" / "p_rgh"
    field_sha256 = hashlib.sha256(field_path.read_bytes()).hexdigest()

    constants = audit.get("constants", {})
    profiles = audit.get("profiles", {})
    expected_constants = {
        "Patm_pa": total_pressure.PATM_PA,
        "g_m_s2": total_pressure.G_M_S2,
        "rho_water_kg_m3": total_pressure.RHO_WATER_KG_M3,
        "rho_air_kg_m3": total_pressure.RHO_AIR_KG_M3,
        "ambient_rim_z_m": total_pressure.AMBIENT_RIM_Z_M,
        "formula_tolerance_pa": total_pressure.FORMULA_TOLERANCE_PA,
    }
    constants_exact = all(
        constants.get(key) == expected for key, expected in expected_constants.items()
    )

    patch_reports: dict[str, object] = {}
    patch_checks: list[bool] = []
    for spec in total_pressure.PROFILE_SPECS:
        patch_type, values = boundary_scalar_values(
            p_rgh_text, field_path, spec.name, "value"
        )
        profile = profiles.get(spec.name, {}) if isinstance(profiles, dict) else {}
        n_faces = profile.get("n_faces") if isinstance(profile, dict) else None
        if len(values) == 1 and isinstance(n_faces, int) and n_faces > 1:
            expanded_values = values * n_faces
        else:
            expanded_values = values
        max_error = max(
            abs(value - spec.expected_zero_u_prgh_pa)
            for value in expanded_values
        )
        profile_formula_ok = (
            isinstance(profile, dict)
            and profile.get("expected_zero_u_p_rgh_pa")
            == spec.expected_zero_u_prgh_pa
            and profile.get("zero_u_consistency_max_abs_error_pa", math.inf)
            <= total_pressure.FORMULA_TOLERANCE_PA
            and profile.get("rendered_p0_max_abs_error_pa", math.inf)
            <= total_pressure.FORMULA_TOLERANCE_PA
            and profile.get("rendered_zero_u_p_rgh_max_abs_error_pa", math.inf)
            <= total_pressure.FORMULA_TOLERANCE_PA
        )
        face_count_ok = isinstance(n_faces, int) and len(expanded_values) == n_faces
        patch_ok = (
            patch_type == "prghTotalPressure"
            and face_count_ok
            and max_error <= TABLE1_REWRITE_TOLERANCE_PA
            and profile_formula_ok
        )
        patch_checks.append(patch_ok)
        patch_reports[spec.name] = {
            "type": patch_type,
            "n_values": len(expanded_values),
            "audit_n_faces": n_faces,
            "value_min_pa": min(expanded_values),
            "value_max_pa": max(expanded_values),
            "expected_zero_u_p_rgh_pa": spec.expected_zero_u_prgh_pa,
            "max_abs_error_pa": max_error,
            "formula_audit_passed": profile_formula_ok,
            "passed": patch_ok,
        }

    air_type, _ = boundary_scalar_values(
        p_rgh_text, field_path, total_pressure.PRESERVED_STAGE1_PATCH, "value"
    )
    audit_bc = audit.get("openfoam_boundary_condition", {})
    checks = {
        "audit_status_passed": audit.get("status") == "passed",
        "field_sha256_matches_audit": audit.get("field_sha256") == field_sha256,
        "converter_constants_exact": constants_exact,
        "openfoam_v2512_prgh_total_pressure_translation": (
            isinstance(audit_bc, dict)
            and audit_bc.get("version") == "OpenFOAM v2512"
            and audit_bc.get("type") == "prghTotalPressure"
        ),
        "all_five_total_pressure_profiles_pass": all(patch_checks),
        "stage1_air_inlet_remains_closed_fixed_flux_pressure": (
            air_type == "fixedFluxPressure"
        ),
    }
    return {
        "audit_path": str(audit_path),
        "field_sha256": field_sha256,
        "expected_constants": expected_constants,
        "patches": patch_reports,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(case: Path, output: Path) -> dict[str, object]:
    case = case.resolve()
    block_text = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    set_fields = (case / "system" / "setFieldsDict").read_text(encoding="utf-8")
    blocks = block_cell_counts(block_text)
    if len(blocks) < 7:
        raise ValueError("S1 topology requires at least seven ordered pipe blocks")

    cells = [nx * ny * nz for nx, ny, nz in blocks]
    main_water_cells = sum(cells[:5])
    supply_water_cells = cells[5]
    riser_nx, riser_ny, riser_nz = blocks[6]
    if riser_ny != 1:
        raise ValueError("the quasi-2-D riser must retain one transverse cell")
    riser_dz = (1.02 - RISER_BOTTOM_M) / riser_nz
    riser_water_rows = math.floor(
        (WATER_LEVEL_M - RISER_BOTTOM_M) / riser_dz + 0.5
    )
    riser_water_cells = riser_nx * riser_water_rows
    expected_water_cells = main_water_cells + supply_water_cells + riser_water_cells

    alpha = scalar_values(case / "0" / "alpha.water")
    pressure = scalar_values(case / "0" / "p")
    reduced_pressure = scalar_values(case / "0" / "p_rgh")
    velocity = vector_values(case / "0" / "U")
    k = scalar_values(case / "0" / "k")
    omega = scalar_values(case / "0" / "omega")
    actual_water_cells = sum(value > 0.5 for value in alpha)
    represented_face = RISER_BOTTOM_M + riser_water_rows * riser_dz

    p_rgh_path = case / "0" / "p_rgh"
    p_rgh_text = p_rgh_path.read_text(encoding="utf-8")
    total_pressure_gate = total_pressure_profile_gate(case, p_rgh_text)
    table1_boundary_profiles = {
        patch: total_pressure_gate["patches"][patch]
        for patch in TABLE1_P_RGH_PA
    }
    supply_block = set_fields.split("no pre-existing gas pocket", 1)
    supply_dictionary_water = (
        len(supply_block) == 2
        and "volScalarFieldValue alpha.water 1" in supply_block[1]
        and "107026.772153" not in supply_block[1]
    )
    checks = {
        "alpha_bounded_0_1": min(alpha) >= 0.0 and max(alpha) <= 1.0,
        "water_cell_partition_exact": actual_water_cells == expected_water_cells,
        "stage1_supply_branch_dictionary_is_water": supply_dictionary_water,
        "riser_surface_within_one_cell": (
            abs(represented_face - WATER_LEVEL_M) <= riser_dz
        ),
        "initial_velocity_zero": all(
            all(abs(component) <= 1.0e-15 for component in vector)
            for vector in velocity
        ),
        "pressure_fields_finite_positive": (
            all(math.isfinite(value) and value > 0.0 for value in pressure)
            and all(math.isfinite(value) and value > 0.0 for value in reduced_pressure)
        ),
        "field_lengths_match_mesh": (
            len(alpha) == len(pressure) == len(reduced_pressure) == sum(cells)
        ),
        "positive_unreported_turbulence_floors": min(k) > 0.0 and min(omega) > 0.0,
        "source_aligned_total_pressure_profile_gate": bool(
            total_pressure_gate["passed"]
        ),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "case": str(case),
        "classification": "source_aligned_stage1_initial_field_audit",
        "stage1_supply_branch_phase": {
            "phase": "water",
            "evidence": (
                "method_inferred_from_simple_water_flow_and_closed_isolation_valve"
            ),
            "pre_existing_gas_allowed": False,
        },
        "mesh": {
            "cell_count": sum(cells),
            "main_water_cells": main_water_cells,
            "supply_branch_water_cells": supply_water_cells,
            "riser_water_cells": riser_water_cells,
            "expected_total_water_cells": expected_water_cells,
        },
        "alpha_water": {
            "min": min(alpha),
            "max": max(alpha),
            "actual_water_cells_alpha_gt_0_5": actual_water_cells,
            "requested_riser_surface_z_m": WATER_LEVEL_M,
            "represented_interface_face_z_m": represented_face,
            "interface_discretization_error_m": represented_face - WATER_LEVEL_M,
        },
        "p_absolute_Pa": {"min": min(pressure), "max": max(pressure)},
        "p_rgh_absolute_Pa": {
            "min": min(reduced_pressure),
            "max": max(reduced_pressure),
        },
        "table1_water_boundary_zero_U_p_rgh_Pa": {
            "expected": TABLE1_P_RGH_PA,
            "profile_summary": table1_boundary_profiles,
            "absolute_tolerance_Pa": TABLE1_REWRITE_TOLERANCE_PA,
        },
        "total_pressure_profile_gate": total_pressure_gate,
        "U_m_per_s": {
            "max_magnitude": max(
                math.sqrt(sum(component * component for component in vector))
                for vector in velocity
            )
        },
        "stage1_turbulence_floors": {
            "k_min": min(k),
            "omega_min": min(omega),
            "source_status": "unreported_declared_positive_numerical_floors",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.case, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
