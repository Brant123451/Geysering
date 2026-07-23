#!/usr/bin/env python3
"""Fail-fast audit of the B-H1 source parameters against an OpenFOAM case.

The experimental facts below are transcribed from Cong et al. (2017), while
the explicitly labelled numerical-method comparison is from Chan et al.
(2018).  Experimental outcomes are reported but never used as forcing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PATM = 101325.0
R_AIR = 287.05
EXPERIMENT = {
    "run": "B-H1",
    "D_m": 0.050,
    "Dr_m": 0.016,
    "H0_above_invert_m": 0.66,
    "L0_m": 0.61,
    "pipe_length_model_m": 6.60,
    "tee_axis_x_m": 3.47,
    "riser_height_above_crown_m": 1.80,
    "temperature_K": 296.15,
    "water_density_kg_m3": 998.2,
    "surface_tension_N_m": 0.072,
    "pocket_pressure_absolute_Pa": PATM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def main() -> None:
    args = parse_args()
    model_dir = Path(__file__).resolve().parent
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def numeric(
        name: str,
        actual: float,
        expected: float,
        source: str,
        tolerance: float = 1e-9,
    ) -> None:
        checks.append(
            {
                "name": name,
                "source": source,
                "expected": expected,
                "actual": actual,
                "absolute_tolerance": tolerance,
                "pass": abs(actual - expected) <= tolerance,
            }
        )

    def contains(name: str, relative: str, needles: list[str], source: str) -> None:
        path = args.case_dir / relative
        text = path.read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in text]
        checks.append(
            {
                "name": name,
                "source": source,
                "file": relative,
                "required_fragments": needles,
                "missing_fragments": missing,
                "pass": not missing,
            }
        )

    primary = "Cong et al. (2017), DOI 10.1061/(ASCE)HY.1943-7900.0001332"
    table2 = f"{primary}, Table 2 row B-H1"
    fig1 = f"{primary}, Fig. 1(b)"
    procedure = f"{primary}, Experimental Procedure and Measurements"

    numeric(
        "horizontal pipe diameter",
        nested(config, "geometry", "pipe_inner_diameter_m"),
        EXPERIMENT["D_m"],
        table2,
    )
    numeric(
        "riser diameter",
        nested(config, "geometry", "riser_inner_diameter_m"),
        EXPERIMENT["Dr_m"],
        table2,
    )
    numeric(
        "upstream head above pipe invert",
        nested(config, "initial_state", "H0_above_pipe_invert_m"),
        EXPERIMENT["H0_above_invert_m"],
        table2,
    )
    numeric(
        "initial pocket length",
        nested(config, "initial_state", "pocket_length_m"),
        EXPERIMENT["L0_m"],
        table2,
    )
    numeric(
        "tee axis",
        nested(config, "geometry", "tee_axis_x_m"),
        EXPERIMENT["tee_axis_x_m"],
        fig1,
    )
    numeric(
        "full computational pipe length",
        nested(config, "geometry", "pipe_length_m"),
        EXPERIMENT["pipe_length_model_m"],
        "Fig. 1 chain gives 6.59 m; Chan et al. (2018) uses x=0..6.6 m",
    )
    numeric(
        "physical riser height above crown",
        nested(config, "geometry", "riser_height_above_pipe_crown_m"),
        EXPERIMENT["riser_height_above_crown_m"],
        f"{primary}, Observation of Geysers",
    )
    numeric(
        "initial temperature",
        nested(config, "initial_state", "temperature_K"),
        EXPERIMENT["temperature_K"],
        f"{primary}, pressure-measurement paragraph (laboratory 23 degC)",
    )
    numeric(
        "initial pocket absolute pressure",
        nested(config, "initial_state", "pocket_pressure_absolute_Pa"),
        EXPERIMENT["pocket_pressure_absolute_Pa"],
        procedure,
    )
    numeric(
        "water reference density",
        nested(config, "materials", "water_reference_density_kg_per_m3"),
        EXPERIMENT["water_density_kg_m3"],
        f"{primary}, pressure-measurement paragraph",
        tolerance=0.05,
    )
    numeric(
        "surface tension",
        nested(config, "materials", "surface_tension_N_per_m"),
        EXPERIMENT["surface_tension_N_m"],
        f"{primary}, Eotvos-number discussion",
    )

    expected_volume = math.pi * EXPERIMENT["D_m"] ** 2 * EXPERIMENT["L0_m"] / 4
    expected_vstar = expected_volume / (
        math.pi * EXPERIMENT["Dr_m"] ** 2 * EXPERIMENT["H0_above_invert_m"] / 4
    )
    expected_air_mass = (
        PATM * expected_volume / (R_AIR * EXPERIMENT["temperature_K"])
    )
    numeric(
        "analytic initial pocket volume",
        nested(config, "initial_state", "analytic_pocket_volume_m3"),
        expected_volume,
        f"{primary}, notation Vair=pi*D^2*L0/4",
        tolerance=5e-11,
    )
    numeric(
        "CAD pipe diameter",
        geometry["pipe_diameter_m"],
        EXPERIMENT["D_m"],
        "generated CAD metadata",
    )
    numeric(
        "CAD riser diameter",
        geometry["riser_diameter_m"],
        EXPERIMENT["Dr_m"],
        "generated CAD metadata",
    )
    numeric(
        "CAD tee axis",
        geometry["tee_x_m"],
        EXPERIMENT["tee_axis_x_m"],
        "generated CAD metadata",
    )
    numeric(
        "CAD riser rim in model y datum",
        geometry["riser_rim_y_m"],
        EXPERIMENT["riser_height_above_crown_m"] + EXPERIMENT["D_m"] / 2,
        "1.8 m above crown; model y=0 is pipe axis",
    )

    contains(
        "initial phase, pressure and temperature fields",
        "system/setFieldsDict",
        [
            "volScalarFieldValue alpha.water 1",
            "volScalarFieldValue T 296.15",
            "box (5.989 -0.030 -0.030) (6.601 0.030 0.030)",
            "volScalarFieldValue alpha.water 0",
            "volScalarFieldValue p 101325",
            "box (3.319 0.635 -0.151) (3.621 3.026 0.151)",
        ],
        f"{table2}; {procedure}",
    )
    contains(
        "constant-head and atmospheric pressure boundaries",
        "0.orig/p_rgh",
        [
            "type            fixedValue;",
            "value           uniform 107543.13717;",
            "value           uniform 101332.42484;",
        ],
        f"{primary}, Series B boundary description",
    )
    contains(
        "water-only reservoir and air-only atmospheric inflow",
        "0.orig/alpha.water",
        [
            "type            fixedValue;",
            "value           uniform 1;",
            "type            inletOutlet;",
            "inletValue      uniform 0;",
        ],
        f"{primary}, Series B initial condition; Chan et al. (2018) boundary conditions",
    )
    contains(
        "gravity",
        "constant/g",
        ["value           (0 -9.81 0);"],
        f"{primary}, notation",
    )
    contains(
        "instantaneous baseline valve is not a hidden source",
        "constant/valveProperties",
        ["active                  false;"],
        "Chan et al. (2018) instantaneous CFD opening; experimental duration audited separately",
    )

    passed = all(check["pass"] for check in checks)
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "case": "Cong2017 B-H1",
        "source_provenance": {
            "primary": {
                "citation": primary,
                "repository_file": "tests/test_02_cong2017/_shared/reference/paper_source/cong2017_JHE2017_offprint.pdf",
                "sha256": "6a2fd77ae65f6361ec5479780a3226e5f5cf69fde643866f692279588c16aa3e",
            },
            "companion_cfd": {
                "citation": "Chan et al. (2018), DOI 10.1061/(ASCE)HY.1943-7900.0001416",
                "role": "methodological comparison only; its fine-mesh B1 uses H0=0.88 m, not B-H1 H0=0.66 m",
            },
        },
        "derived_targets": {
            "initial_pocket_volume_m3": expected_volume,
            "initial_pocket_air_mass_kg": expected_air_mass,
            "Vair_star": expected_vstar,
            "initial_free_surface_height_above_crown_m": (
                EXPERIMENT["H0_above_invert_m"] - EXPERIMENT["D_m"]
            ),
        },
        "experimental_outputs_not_used_as_inputs": {
            "geyser": True,
            "Ta_s": 8.07,
            "vfs_m_per_s": 0.924,
            "vint_m_per_s": 1.231,
        },
        "comparison_scope": {
            "Fig9a": "Run B-H1 high-speed Yfs/Yint; direct quantitative comparison",
            "Fig10a": "Run B-1 PT1, same nominal D/Dr/H0/L0 but a different realization; morphology only",
        },
        "declared_method_differences_from_chan2018": {
            "target_run": "B-H1 H0=0.66 m, whereas companion fine-mesh B1 has H0=0.88 m",
            "temperature": "296.15 K measured in Cong2017, rather than companion CFD 300 K",
            "riser_top": "physical 1.8 m rim plus open exterior atmosphere, rather than a confined 3.0 m riser",
            "turbulence": "laminar preregistered OpenFOAM baseline; companion uses standard k-epsilon but does not report inlet/initial turbulence data",
            "valve": "instantaneous baseline plus separately labelled 0.2/0.5 s resistance sensitivities",
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("B-H1 paper-to-case audit failed")


if __name__ == "__main__":
    main()
