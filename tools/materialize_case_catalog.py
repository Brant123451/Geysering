"""Create manifests and missing Case folders from the existing experiment catalog."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from tools.scaffold_case import scaffold_case


def load_series_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["run"]: row for row in csv.DictReader(stream)}


def write_single_row_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest(
    test: str,
    case_id: str,
    source: str,
    status: str,
    entrypoints: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "test": test,
        "id": case_id,
        "source": source,
        "status": status,
        "self_contained": True,
        "entrypoints": entrypoints,
    }


def _materialize_vw(root: Path) -> None:
    cases = root / "tests" / "test_01_vw2011" / "cases"
    source = "Vasconcelos & Wright (2011), Journal of Hydraulic Engineering"
    definitions = {
        "A_Dt57p1_Ha0305_Yfs0356": {
            "status": "validated_primary_case",
            "config": {
                "tower_diameter_m": 0.0571,
                "initial_air_pressure_head_m": 0.305,
                "initial_water_level_m": 0.356,
                "observed_branch": "no_geyser",
            },
            "entrypoints": [
                "scripts/caseA_digitize_and_compare.py",
                "openfoam/2d/Allrun",
                "openfoam/3d/Allrun",
            ],
        },
        "B_Dt12p7_Ha0610_Yfs0356": {
            "status": "validated_primary_case",
            "config": {
                "tower_diameter_m": 0.0127,
                "initial_air_pressure_head_m": 0.610,
                "initial_water_level_m": 0.356,
                "observed_branch": "geyser",
            },
            "entrypoints": ["scripts/caseB_digitize_and_compare.py"],
        },
    }
    for case_id, definition in definitions.items():
        scaffold_case(
            cases / case_id,
            manifest=_manifest(
                "test_01_vw2011",
                case_id,
                source,
                str(definition["status"]),
                list(definition["entrypoints"]),
            ),
            config=dict(definition["config"]),
        )

    variants = {
        "Fig10_Dt57p1_Ha0305_Yfs0254": {
            "model": cases
            / "A_Dt57p1_Ha0305_Yfs0356"
            / "model"
            / "vw2011_network_twofluid.py",
            "config": {
                "tower_diameter_m": 0.0571,
                "initial_air_pressure_head_m": 0.305,
                "initial_water_level_m": 0.254,
                "paper_figure": "Figure 10",
            },
            "entrypoints": ["scripts/caseA_fig10_compare.py"],
        },
        "Fig11_Dt12p7_Ha0305_Yfs0254": {
            "model": cases
            / "B_Dt12p7_Ha0610_Yfs0356"
            / "model"
            / "vw2011_network_twofluid.py",
            "config": {
                "tower_diameter_m": 0.0127,
                "initial_air_pressure_head_m": 0.305,
                "initial_water_level_m": 0.254,
                "paper_figure": "Figure 11",
            },
            "entrypoints": ["scripts/caseB_fig11_compare.py"],
        },
    }
    for case_id, definition in variants.items():
        case = cases / case_id
        scaffold_case(
            case,
            manifest=_manifest(
                "test_01_vw2011",
                case_id,
                source,
                "validated_auxiliary_case",
                list(definition["entrypoints"]),
            ),
            config=dict(definition["config"]),
            model_sources=[Path(definition["model"])],
        )
        case.joinpath("README.md").write_text(
            f"# {case_id}\n\n"
            f"VW2011 {definition['config']['paper_figure']} auxiliary comparison. "
            "This folder isolates the parameter-specific code, data, and outputs "
            "that were previously mixed into a primary Case directory.\n",
            encoding="utf-8",
        )


def _coerce(value: str) -> object:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _materialize_cong(root: Path) -> None:
    test_root = root / "tests" / "test_02_cong2017"
    cases = test_root / "cases"
    study = test_root / "studies" / "criterion_map"
    rows = load_series_rows(study / "outputs" / "seriesB_fullsync.csv")
    source_model = study / "model" / "cong2017_network_twofluid.py"
    scan_common = (study / "scan_common.py").read_text(encoding="utf-8")
    scan_common = scan_common.replace(
        "HERE = Path(__file__).resolve().parent",
        "HERE = Path(__file__).resolve().parents[1]",
    )
    source = "Cong, Chan & Lee (2017), Journal of Hydraulic Engineering"

    for run, row in rows.items():
        number = run.replace("B-H", "")
        case_id = f"BH{number}_Dr{row['Dr_mm']}_H066_L061"
        case = cases / case_id
        config = {
            key: _coerce(row[key])
            for key in ("run", "Dr_mm", "Dr_over_D", "L0_m", "H0_m", "Vair_star")
        }
        config["observed_geyser"] = bool(int(row["geyser_meas"]))
        config["modeled_geyser"] = bool(int(row["geyser_model"]))
        scaffold_case(
            case,
            manifest=_manifest(
                "test_02_cong2017",
                case_id,
                source,
                "validated_detailed_case" if run in {"B-H1", "B-H6"} else "series_scan_case",
                ["scripts/run_series_case.py"],
            ),
            config=config,
            model_sources=[source_model],
        )

        measured_fields = (
            "run",
            "Dr_mm",
            "L0_m",
            "H0_m",
            "Ta_meas_s",
            "vfs_meas",
            "vint_meas",
            "geyser_meas",
        )
        model_fields = tuple(key for key in row if key not in measured_fields)
        write_single_row_csv(
            case / "data" / "series_b_measurement.csv",
            {key: row[key] for key in measured_fields},
        )
        write_single_row_csv(
            case / "outputs" / "series_b_model_summary.csv",
            {"run": run, **{key: row[key] for key in model_fields if key != "run"}},
        )
        (case / "scripts" / "scan_common.py").write_text(scan_common, encoding="utf-8")
        runner = f'''"""Run Cong 2017 {run} with this Case's frozen model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scan_common import run_one

CASE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((CASE_ROOT / "config" / "case.json").read_text(encoding="utf-8"))
RESULT = run_one(CONFIG["Dr_mm"], CONFIG["L0_m"], CONFIG["H0_m"])
RESULT["run"] = CONFIG["run"]
output = CASE_ROOT / "outputs" / "rerun_result.json"
output.write_text(json.dumps(RESULT, indent=2) + "\\n", encoding="utf-8")
print(output)
'''
        (case / "scripts" / "run_series_case.py").write_text(runner, encoding="utf-8")
        if run not in {"B-H1", "B-H6"}:
            case.joinpath("README.md").write_text(
                f"# {run} — Cong, Chan & Lee (2017)\n\n"
                f"Series B case with `Dr={row['Dr_mm']} mm`, `H0={row['H0_m']} m`, "
                f"and `L0={row['L0_m']} m`.\n\n"
                f"- Measured geyser classification: `{row['geyser_meas']}`\n"
                f"- Model classification: `{row['geyser_model']}` (`{row['match']}`)\n"
                "- Re-run: `python scripts/run_series_case.py`\n"
                "- Original campaign scan: `../../studies/criterion_map`\n",
                encoding="utf-8",
            )


def _materialize_liu(root: Path) -> None:
    cases = root / "tests" / "test_03_liu2020" / "cases"
    source = "Liu, Shao & Zhu (2020), Journal of Hydraulic Engineering"
    definitions = {
        "A2_Q20to100_openchannel_nogeyser": {
            "status": "validated_primary_case",
            "config": {
                "case": "A2",
                "flow_lps_initial": 20,
                "flow_lps_final": 100,
                "downstream": "open_channel",
                "observed_branch": "no_geyser",
            },
            "entrypoints": ["scripts/caseA_digitize_and_compare.py", "openfoam/3d/Allrun"],
        },
        "B3_Q20to100_fullpipe_geyser": {
            "status": "validated_primary_case",
            "config": {
                "case": "B3",
                "flow_lps_initial": 20,
                "flow_lps_final": 100,
                "downstream": "full_pipe",
                "observed_branch": "geyser",
            },
            "entrypoints": ["scripts/caseB_digitize_and_compare.py"],
        },
        "C9_Q25to40_hr03_airpocket": {
            "status": "validated_phase1_only",
            "config": {
                "case": "C9",
                "flow_lps_initial": 25,
                "flow_lps_final": 40,
                "initial_riser_level_m": 0.3,
                "air_pocket": True,
                "computed_scope": "phase_1",
            },
            "entrypoints": ["scripts/caseC_digitize_and_compare.py"],
        },
    }
    for case_id, definition in definitions.items():
        scaffold_case(
            cases / case_id,
            manifest=_manifest(
                "test_03_liu2020",
                case_id,
                source,
                str(definition["status"]),
                list(definition["entrypoints"]),
            ),
            config=dict(definition["config"]),
        )


def materialize(root: Path) -> None:
    _materialize_vw(root)
    _materialize_cong(root)
    _materialize_liu(root)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    materialize(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
