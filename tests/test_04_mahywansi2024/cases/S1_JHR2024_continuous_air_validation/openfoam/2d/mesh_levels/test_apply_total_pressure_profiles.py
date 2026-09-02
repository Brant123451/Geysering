#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


# These compatibility tests exercise the corrected recovery campaign, not the
# superseded all-total-pressure converter that remains beside this file as an
# invalidated historical artifact.
RECOVERY_ROOT = Path(__file__).resolve().parent / "recovery_total_pressure_v1"
if str(RECOVERY_ROOT) in sys.path:
    sys.path.remove(str(RECOVERY_ROOT))
sys.path.insert(0, str(RECOVERY_ROOT))

import apply_total_pressure_profiles as converter


def _foam_header(object_name: str, class_name: str, *, file_format: str = "ascii") -> str:
    return f"""FoamFile
{{
    version 2.0;
    format {file_format};
    class {class_name};
    object {object_name};
}}
"""


def _make_case(root: Path) -> tuple[Path, dict[str, list[float]]]:
    case = root / "case"
    mesh = case / "constant" / "polyMesh"
    zero = case / "0"
    mesh.mkdir(parents=True)
    zero.mkdir()

    patch_z = {
        "waterInlet": [-0.010, 0.010],
        "waterOutlet": [-0.005, 0.015],
        "airInlet": [0.150],
        "ambientFloor": [1.020, 1.020],
        "ambientSides": [1.100, 1.600],
        "ambientTop": [2.000],
    }
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    patch_ranges: dict[str, tuple[int, int]] = {}
    for patch_i, (name, z_values) in enumerate(patch_z.items()):
        start = len(faces)
        for face_i, z_m in enumerate(z_values):
            x = float(patch_i)
            y = float(face_i)
            base = len(points)
            points.extend(
                (
                    (x - 0.1, y - 0.1, z_m),
                    (x + 0.1, y - 0.1, z_m),
                    (x + 0.1, y + 0.1, z_m),
                    (x - 0.1, y + 0.1, z_m),
                )
            )
            faces.append((base, base + 1, base + 2, base + 3))
        patch_ranges[name] = (start, len(z_values))

    points_text = _foam_header("points", "vectorField")
    points_text += f"{len(points)}\n(\n"
    points_text += "\n".join(f"({x} {y} {z})" for x, y, z in points)
    points_text += "\n)\n"
    (mesh / "points").write_text(points_text, encoding="utf-8", newline="\n")

    faces_text = _foam_header("faces", "faceList")
    faces_text += f"{len(faces)}\n(\n"
    faces_text += "\n".join(
        f"4({a} {b} {c} {d})" for a, b, c, d in faces
    )
    faces_text += "\n)\n"
    (mesh / "faces").write_text(faces_text, encoding="utf-8", newline="\n")

    boundary_text = _foam_header("boundary", "polyBoundaryMesh")
    boundary_text += f"{len(patch_ranges)}\n(\n"
    for name, (start, count) in patch_ranges.items():
        boundary_text += f"""    {name}
    {{
        type patch;
        nFaces {count};
        startFace {start};
    }}
"""
    boundary_text += ")\n"
    (mesh / "boundary").write_text(boundary_text, encoding="utf-8", newline="\n")

    field_text = _foam_header("p_rgh", "volScalarField")
    field_text += """dimensions [1 -1 -2 0 0 0 0];
internalField uniform 101325;
boundaryField
{
"""
    for name in patch_ranges:
        if name == "airInlet":
            field_text += f"""    {name}
    {{
        type fixedFluxPressure;
        gradient uniform 0;
        value uniform 101325;
    }}
"""
        else:
            field_text += f"""    {name}
    {{
        type fixedValue;
        value uniform 101325;
    }}
"""
    field_text += "}\n"
    (zero / "p_rgh").write_text(field_text, encoding="utf-8", newline="\n")
    return case, patch_z


def _profile_values(case: Path, patch: str, entry: str) -> list[float]:
    path = case / "0" / "p_rgh"
    text = path.read_text(encoding="utf-8")
    blocks, _ = converter._find_field_patch_blocks(text, path)
    body = converter._patch_body(text, blocks[patch])
    values = converter._nonuniform_scalar_entry(
        body, entry, f"test:{patch}", required=True
    )
    assert values is not None
    return converter._expanded(values, len(values), f"test:{patch}:{entry}")


class ApplyTotalPressureProfilesTests(unittest.TestCase):
    def test_table1_inlet_total_and_outlet_static_profiles_use_frozen_densities(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, patch_z = _make_case(Path(temp))
            audit = converter.apply_total_pressure_profiles(case)

            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["schema_version"], 2)
            constants = audit["constants"]
            self.assertEqual(constants["rho_water_kg_m3"], 998.4)
            self.assertEqual(constants["rho_air_kg_m3"], 1.204317575)
            self.assertEqual(
                audit["profiles"]["waterInlet"]["invariant_reduced_pressure_pa"],
                107064.462144,
            )
            self.assertEqual(
                audit["profiles"]["waterOutlet"]["invariant_reduced_pressure_pa"],
                107044.873536,
            )
            self.assertAlmostEqual(
                audit["profiles"]["ambientSides"]["invariant_reduced_pressure_pa"],
                101337.050643,
                places=6,
            )
            self.assertEqual(
                audit["openfoam_boundary_conditions"]["pressure_inlet"],
                {
                    "type": "prghTotalPressure",
                    "entry": "p0",
                    "semantics": "physical absolute total pressure",
                    "zero_velocity_equation": "p_rgh=p0-rho*((g.Cf)-ghRef)",
                    "inflow_dynamic_term": "-0.5*rho*neg(phi)*magSqr(U)",
                },
            )
            self.assertEqual(
                audit["openfoam_boundary_conditions"]["pressure_outlet"],
                {
                    "type": "prghPressure",
                    "entry": "p",
                    "semantics": "physical absolute static pressure",
                    "equation": "p_rgh=p-rho*((g.Cf)-ghRef)",
                },
            )

            inlet = _profile_values(case, "waterInlet", "p0")
            expected_inlet = [
                101325.0 + 998.4 * 9.81 * (0.586 - z_m)
                for z_m in patch_z["waterInlet"]
            ]
            self.assertEqual(len(inlet), 2)
            for actual, expected in zip(inlet, expected_inlet):
                self.assertAlmostEqual(actual, expected, places=9)
            self.assertNotEqual(inlet[0], inlet[1])

            outlet = _profile_values(case, "waterOutlet", "p")
            expected_outlet = [
                101325.0 + 998.4 * 9.81 * (0.584 - z_m)
                for z_m in patch_z["waterOutlet"]
            ]
            for actual, expected in zip(outlet, expected_outlet):
                self.assertAlmostEqual(actual, expected, places=9)

            sides = _profile_values(case, "ambientSides", "p")
            expected_sides = [
                101325.0 + 1.204317575 * 9.81 * (1.02 - z_m)
                for z_m in patch_z["ambientSides"]
            ]
            for actual, expected in zip(sides, expected_sides):
                self.assertAlmostEqual(actual, expected, places=9)
            self.assertNotEqual(sides[0], sides[1])

            for patch in converter.TARGET_PATCH_NAMES:
                values = _profile_values(case, patch, "value")
                spec = converter.PROFILE_BY_NAME[patch]
                expected_values = [
                    spec.reduced_pressure_pa(z_m) for z_m in patch_z[patch]
                ]
                for actual, expected in zip(values, expected_values):
                    self.assertAlmostEqual(actual, expected, places=8)

            persisted = json.loads((case / "total_pressure_profile_audit.json").read_text())
            self.assertEqual(persisted, audit)
            self.assertIn("thermophysicalProperties/setExprFields", persisted["evidence"]["density_source"])

    def test_boundary4_static_p_is_atmospheric_and_5_6_are_hydrostatic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, patch_z = _make_case(Path(temp))
            converter.apply_total_pressure_profiles(case)
            floor = _profile_values(case, "ambientFloor", "p")
            self.assertEqual(floor, [101325.0, 101325.0])

            for patch in ("ambientSides", "ambientTop"):
                static_p = _profile_values(case, patch, "p")
                expected = [
                    101325.0 + 1.204317575 * 9.81 * (1.02 - z_m)
                    for z_m in patch_z[patch]
                ]
                for actual, wanted in zip(static_p, expected):
                    self.assertAlmostEqual(actual, wanted, places=9)

            text = (case / "0" / "p_rgh").read_text(encoding="utf-8")
            blocks, _ = converter._find_field_patch_blocks(
                text, case / "0" / "p_rgh"
            )
            for patch in ("ambientFloor", "ambientSides", "ambientTop"):
                body = converter._patch_body(text, blocks[patch])
                self.assertEqual(
                    converter._single_type(body, f"test:{patch}"), "prghPressure"
                )
                self.assertIsNone(
                    converter._nonuniform_scalar_entry(
                        body, "p0", f"test:{patch}", required=False
                    )
                )

    def test_field_and_audit_are_byte_idempotent_and_air_inlet_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, _ = _make_case(Path(temp))
            field = case / "0" / "p_rgh"
            before = field.read_text(encoding="utf-8")
            before_blocks, _ = converter._find_field_patch_blocks(before, field)
            before_air = before[
                before_blocks["airInlet"].brace_start : before_blocks["airInlet"].brace_end + 1
            ]

            converter.apply_total_pressure_profiles(case)
            field_first = field.read_bytes()
            audit_first = (case / "total_pressure_profile_audit.json").read_bytes()
            after = field_first.decode("utf-8")
            after_blocks, _ = converter._find_field_patch_blocks(after, field)
            after_air = after[
                after_blocks["airInlet"].brace_start : after_blocks["airInlet"].brace_end + 1
            ]
            self.assertEqual(after_air, before_air)

            converter.apply_total_pressure_profiles(case)
            self.assertEqual(field.read_bytes(), field_first)
            self.assertEqual((case / "total_pressure_profile_audit.json").read_bytes(), audit_first)

    def test_missing_patch_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, _ = _make_case(Path(temp))
            boundary = case / "constant" / "polyMesh" / "boundary"
            boundary.write_text(
                boundary.read_text(encoding="utf-8").replace("ambientTop", "missingAmbientTop"),
                encoding="utf-8",
                newline="\n",
            )
            field = case / "0" / "p_rgh"
            original = field.read_bytes()
            with self.assertRaisesRegex(converter.ConversionError, "ambientTop.*missing"):
                converter.apply_total_pressure_profiles(case)
            self.assertEqual(field.read_bytes(), original)
            self.assertFalse((case / "total_pressure_profile_audit.json").exists())

    def test_binary_out_of_range_and_declared_count_meshes_are_rejected(self) -> None:
        mutations = ("binary", "point_index", "declared_face_count")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                case, _ = _make_case(Path(temp))
                mesh = case / "constant" / "polyMesh"
                if mutation == "binary":
                    points = mesh / "points"
                    points.write_text(
                        points.read_text(encoding="utf-8").replace("format ascii;", "format binary;"),
                        encoding="utf-8",
                        newline="\n",
                    )
                elif mutation == "point_index":
                    faces = mesh / "faces"
                    faces.write_text(
                        faces.read_text(encoding="utf-8").replace("4(0 1 2 3)", "4(999 1 2 3)", 1),
                        encoding="utf-8",
                        newline="\n",
                    )
                else:
                    faces = mesh / "faces"
                    text = faces.read_text(encoding="utf-8")
                    text = re.sub(r"(object faces;\s*}\s*)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 1), text, count=1)
                    faces.write_text(text, encoding="utf-8", newline="\n")

                field = case / "0" / "p_rgh"
                original = field.read_bytes()
                with self.assertRaises(converter.ConversionError):
                    converter.apply_total_pressure_profiles(case)
                self.assertEqual(field.read_bytes(), original)
                self.assertFalse((case / "total_pressure_profile_audit.json").exists())

    def test_existing_list_count_and_formula_tampering_fail_closed(self) -> None:
        for mutation in ("list_count", "formula"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                case, _ = _make_case(Path(temp))
                converter.apply_total_pressure_profiles(case)
                field = case / "0" / "p_rgh"
                text = field.read_text(encoding="utf-8")
                blocks, _ = converter._find_field_patch_blocks(text, field)
                block = blocks["waterInlet"]
                body = converter._patch_body(text, block)
                if mutation == "list_count":
                    changed_body, count = re.subn(
                        r"(\bp0\s+nonuniform\s+List<scalar>\s*)2(\s*\()",
                        r"\g<1>3\g<2>",
                        body,
                        count=1,
                    )
                else:
                    match = re.search(
                        rf"\bp0\s+nonuniform\s+List<scalar>\s+2\s*\(\s*({converter._FLOAT_PATTERN})",
                        body,
                    )
                    self.assertIsNotNone(match)
                    assert match is not None
                    wrong = format(float(match.group(1)) + 1.0, ".17g")
                    changed_body = body[: match.start(1)] + wrong + body[match.end(1) :]
                    count = 1
                self.assertEqual(count, 1)
                mutated = text[: block.brace_start + 1] + changed_body + text[block.brace_end :]
                field.write_text(mutated, encoding="utf-8", newline="\n")
                original = field.read_bytes()
                with self.assertRaises(converter.ConversionError):
                    converter.apply_total_pressure_profiles(case)
                self.assertEqual(field.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
