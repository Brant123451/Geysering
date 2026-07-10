"""Validate the reorganized Geysering Test/Case directory contract."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

EXPECTED_CASES = {
    "test_01_vw2011": (
        "A_Dt57p1_Ha0305_Yfs0356",
        "B_Dt12p7_Ha0610_Yfs0356",
        "Fig10_Dt57p1_Ha0305_Yfs0254",
        "Fig11_Dt12p7_Ha0305_Yfs0254",
    ),
    "test_02_cong2017": (
        "BH1_Dr16_H066_L061",
        "BH2_Dr21_H066_L061",
        "BH3_Dr26_H066_L061",
        "BH4_Dr31_H066_L061",
        "BH5_Dr36_H066_L061",
        "BH6_Dr41_H066_L061",
        "BH7_Dr46_H066_L061",
    ),
    "test_03_liu2020": (
        "A2_Q20to100_openchannel_nogeyser",
        "B3_Q20to100_fullpipe_geyser",
        "C9_Q25to40_hr03_airpocket",
    ),
}

REQUIRED_FILES = ("README.md", "manifest.yaml")
REQUIRED_DIRS = ("config", "data", "model", "scripts", "reference", "outputs")
FORBIDDEN_PATTERNS = ("_tmp*", "_dev*", "_slugdbg*", "*.bak", "*.bak-*", "*.bak.*")
OLD_CAMPAIGN_DIRS = (
    "Vasconcelos_Wright_2011_Geysering",
    "Cong_Chan_Lee_2017_Geyser_Horizontal_Pipe_Vertical_Shaft",
    "Liu_Shao_Zhu_2020_Junction_Chamber_Geyser",
)


def validate_case(case: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (case / name).is_file():
            errors.append(f"missing {name}")
    for name in REQUIRED_DIRS:
        if not (case / name).is_dir():
            errors.append(f"missing directory: {name}")
    return errors


def _forbidden_files(directory: Path, root: Path) -> list[str]:
    if not directory.is_dir():
        return []
    errors: list[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_PATTERNS):
            relative = path.relative_to(root).as_posix()
            errors.append(f"forbidden temporary file: {relative}")
    return errors


def validate_repository(
    root: Path,
    *,
    test_id: str | None = None,
    check_old_campaigns: bool = False,
) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    selected = (
        {test_id: EXPECTED_CASES[test_id]}
        if test_id is not None
        else EXPECTED_CASES
    )

    errors.extend(_forbidden_files(root, root))
    for current_test, case_ids in selected.items():
        test_root = root / "tests" / current_test
        if not (test_root / "README.md").is_file():
            errors.append(f"missing Test README: tests/{current_test}/README.md")
        if not (test_root / "_shared").is_dir():
            errors.append(f"missing shared directory: tests/{current_test}/_shared")
        errors.extend(_forbidden_files(test_root, root))
        for case_id in case_ids:
            case = test_root / "cases" / case_id
            relative = case.relative_to(root).as_posix()
            if not case.is_dir():
                errors.append(f"missing Case directory: {relative}")
                continue
            errors.extend(f"{relative}: {error}" for error in validate_case(case))

    if check_old_campaigns:
        for name in OLD_CAMPAIGN_DIRS:
            if (root / name).exists():
                errors.append(f"old campaign directory remains: {name}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test", choices=tuple(EXPECTED_CASES))
    parser.add_argument("--check-old-campaigns", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_repository(
        args.root,
        test_id=args.test,
        check_old_campaigns=args.check_old_campaigns,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"{len(errors)} layout error(s)")
        return 1
    print("Layout validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
