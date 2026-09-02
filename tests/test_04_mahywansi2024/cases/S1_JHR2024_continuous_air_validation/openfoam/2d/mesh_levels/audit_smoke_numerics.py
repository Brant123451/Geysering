#!/usr/bin/env python3
"""Read-only numerical acceptance audit for the three S1 2-D smoke runs.

The script never invokes OpenFOAM and never creates run-state markers.  It
reads ``log.smoke`` plus the written fields at t=0.02 and writes one
``smoke_audit.json`` inside each selected mesh-level case directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
TARGET_TIME = 0.02
TIME_TOLERANCE = 1.0e-12
LEVELS = {
    "coarse": ROOT / "coarse",
    "medium_refine": ROOT / "medium_refine" / "case",
    "refined": ROOT / "refined" / "case",
}
REQUIRED_FIELDS = ("alpha.water", "p", "p_rgh", "U")

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER_RE = re.compile(NUMBER)
NONFINITE_TOKEN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[+-]?(?:nan|inf(?:inity)?|1\.#(?:inf|nan)))"
    r"(?![A-Za-z0-9_])"
)
TIME_RE = re.compile(rf"(?m)^\s*Time\s*=\s*({NUMBER})\s*$")
CO_RE = re.compile(
    rf"(?m)^\s*Courant Number mean:\s*({NUMBER})\s+max:\s*({NUMBER})\s*$"
)
ALPHA_CO_RE = re.compile(
    rf"(?m)^\s*Interface Courant Number mean:\s*({NUMBER})\s+max:\s*({NUMBER})\s*$"
)
DELTAT_RE = re.compile(rf"(?m)^\s*deltaT\s*=\s*({NUMBER})\s*$")
END_RE = re.compile(r"(?m)^\s*End\s*$")
WRAPPER_EXIT_RE = re.compile(r"wrapper_returncode=([+-]?\d+)")
FATAL_RE = re.compile(
    r"(?i)FOAM\s+FATAL|fatal error|segmentation fault|"
    r"floating point exception(?!\s+trapping enabled)|(?<!FOAM_)sigfpe|"
    r"core dumped|abort(?:ed)?"
)
PRESSURE_CLIP_RE = re.compile(
    r"(?i)(?:\bpressure\b|\bp_rgh\b|\bp\b).{0,48}"
    r"(?:clip(?:ped|ping)?|bound(?:ed|ing)?|limit(?:ed|ing)?)|"
    r"(?:clip(?:ped|ping)?|bounding|limit(?:ed|ing)?).{0,48}"
    r"(?:\bpressure\b|\bp_rgh\b|\bp\b)"
)


class FieldParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedField:
    name: str
    path: Path
    kind: str
    representation: str
    declared_count: int | None
    values: tuple[float, ...] | tuple[tuple[float, float, float], ...]


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    return path.read_text(encoding="utf-8", errors="replace")


def field_path(time_dir: Path, name: str) -> Path | None:
    plain = time_dir / name
    compressed = time_dir / f"{name}.gz"
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    return None


def matching_parenthesis(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise FieldParseError("unterminated internalField list")


def parse_internal_field(path: Path, expected_name: str) -> ParsedField:
    text = read_text(path)
    if not re.search(r"(?m)^\s*format\s+ascii\s*;", text):
        raise FieldParseError("only ASCII OpenFOAM fields are accepted")
    object_match = re.search(r"(?m)^\s*object\s+([^;]+)\s*;", text)
    if object_match is None or object_match.group(1).strip() != expected_name:
        raise FieldParseError(f"field object is not {expected_name!r}")

    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text, re.DOTALL)
    if uniform is not None:
        payload = uniform.group(1).strip()
        if NONFINITE_TOKEN_RE.search(payload):
            raise FieldParseError("non-finite token in uniform internalField")
        if payload.startswith("("):
            numbers = tuple(float(item) for item in NUMBER_RE.findall(payload))
            if len(numbers) != 3 or not all(math.isfinite(item) for item in numbers):
                raise FieldParseError("invalid uniform vector internalField")
            values: tuple[tuple[float, float, float], ...] = (numbers,)  # type: ignore[assignment]
            return ParsedField(expected_name, path, "vector", "uniform", None, values)
        numbers = tuple(float(item) for item in NUMBER_RE.findall(payload))
        if len(numbers) != 1 or not math.isfinite(numbers[0]):
            raise FieldParseError("invalid uniform scalar internalField")
        return ParsedField(expected_name, path, "scalar", "uniform", None, numbers)

    header = re.search(
        r"internalField\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(",
        text,
        re.DOTALL,
    )
    if header is None:
        raise FieldParseError("internalField is neither supported uniform nor nonuniform")
    kind = header.group(1)
    declared_count = int(header.group(2))
    opening = header.end() - 1
    closing = matching_parenthesis(text, opening)
    payload = text[opening + 1 : closing]
    if NONFINITE_TOKEN_RE.search(payload):
        raise FieldParseError("non-finite token in nonuniform internalField")

    if kind == "scalar":
        scalar_values = tuple(float(item) for item in NUMBER_RE.findall(payload))
        if len(scalar_values) != declared_count:
            raise FieldParseError(
                f"declared {declared_count} scalar values but parsed {len(scalar_values)}"
            )
        if not all(math.isfinite(item) for item in scalar_values):
            raise FieldParseError("non-finite scalar internalField value")
        return ParsedField(
            expected_name,
            path,
            kind,
            "nonuniform",
            declared_count,
            scalar_values,
        )

    vector_groups = re.findall(r"\(([^()]*)\)", payload)
    vectors: list[tuple[float, float, float]] = []
    for group in vector_groups:
        components = tuple(float(item) for item in NUMBER_RE.findall(group))
        if len(components) != 3 or not all(math.isfinite(item) for item in components):
            raise FieldParseError("invalid/non-finite vector internalField value")
        vectors.append(components)  # type: ignore[arg-type]
    if len(vectors) != declared_count:
        raise FieldParseError(
            f"declared {declared_count} vectors but parsed {len(vectors)}"
        )
    return ParsedField(
        expected_name,
        path,
        kind,
        "nonuniform",
        declared_count,
        tuple(vectors),
    )


def control_value(case: Path, key: str) -> float | None:
    control = case / "system" / "controlDict"
    if not control.is_file():
        return None
    text = read_text(control)
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s+({NUMBER})\s*;", text)
    return float(match.group(1)) if match else None


def matching_time_directory(case: Path) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if math.isfinite(value) and abs(value - TARGET_TIME) <= TIME_TOLERANCE:
            candidates.append((abs(value - TARGET_TIME), path))
    return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def matching_lines(text: str, pattern: re.Pattern[str], limit: int = 20) -> list[str]:
    return [line.strip() for line in text.splitlines() if pattern.search(line)][:limit]


def finite_nonnegative(items: Iterable[float]) -> bool:
    return all(math.isfinite(item) and item >= 0.0 for item in items)


def base_report(level: str, case: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_type": "read_only_no_OpenFOAM_process",
        "level": level,
        "case_directory": str(case),
        "target_time_s": TARGET_TIME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "creates_run_markers": False,
    }


def audit_level(level: str, case: Path) -> dict[str, Any]:
    report = base_report(level, case)
    log_path = case / "log.smoke"
    time_dir = matching_time_directory(case)
    missing_inputs: list[str] = []
    if not log_path.is_file():
        missing_inputs.append("log.smoke")
    if time_dir is None:
        missing_inputs.append("time_directory_0.02")

    log_text = read_text(log_path) if log_path.is_file() else ""
    end_present = bool(END_RE.search(log_text))
    fatal_lines = matching_lines(log_text, FATAL_RE)
    nonfinite_lines = matching_lines(log_text, NONFINITE_TOKEN_RE)
    clipping_lines = matching_lines(log_text, PRESSURE_CLIP_RE)
    times = [float(item) for item in TIME_RE.findall(log_text)]
    co_pairs = [(float(mean), float(maximum)) for mean, maximum in CO_RE.findall(log_text)]
    alpha_co_pairs = [
        (float(mean), float(maximum)) for mean, maximum in ALPHA_CO_RE.findall(log_text)
    ]
    delta_ts = [float(item) for item in DELTAT_RE.findall(log_text)]
    wrapper_codes = [int(item) for item in WRAPPER_EXIT_RE.findall(log_text)]

    log_metrics = {
        "time_samples": len(times),
        "last_time_s": times[-1] if times else None,
        "max_time_s": max(times) if times else None,
        "target_time_reached": any(
            abs(item - TARGET_TIME) <= TIME_TOLERANCE for item in times
        ),
        "did_not_advance_past_target": bool(times)
        and max(times) <= TARGET_TIME + TIME_TOLERANCE,
        "End_present": end_present,
        "wrapper_exit_codes": wrapper_codes,
        "wrapper_exit_zero": bool(wrapper_codes) and wrapper_codes[-1] == 0,
        "fatal_lines": fatal_lines,
        "nonfinite_lines": nonfinite_lines,
        "pressure_clipping_lines": clipping_lines,
        "maxCo": max((item[1] for item in co_pairs), default=None),
        "maxAlphaCo": max((item[1] for item in alpha_co_pairs), default=None),
        "deltaT": delta_ts[-1] if delta_ts else None,
        "deltaT_min": min(delta_ts) if delta_ts else None,
        "deltaT_max": max(delta_ts) if delta_ts else None,
        "configured_maxCo": control_value(case, "maxCo"),
        "configured_maxAlphaCo": control_value(case, "maxAlphaCo"),
        "configured_maxDeltaT": control_value(case, "maxDeltaT"),
    }
    report["inputs"] = {
        "log": str(log_path),
        "time_directory": str(time_dir) if time_dir else None,
        "missing": missing_inputs,
    }
    report["log"] = log_metrics

    # A still-running/not-yet-written smoke is incomplete, not a numerical
    # failure.  A fatal token is a hard failure even without End.
    if missing_inputs or not end_present:
        if fatal_lines or nonfinite_lines or clipping_lines:
            report.update(
                {
                    "status": "failed",
                    "complete": False,
                    "passed": False,
                    "hard_gates": {
                        "no_fatal": not fatal_lines,
                        "no_log_nonfinite": not nonfinite_lines,
                        "no_pressure_clipping": not clipping_lines,
                    },
                    "failure_reasons": [
                        item
                        for item, failed in (
                            ("fatal log token", bool(fatal_lines)),
                            ("NaN/Inf log token", bool(nonfinite_lines)),
                            ("pressure clipping log token", bool(clipping_lines)),
                        )
                        if failed
                    ],
                }
            )
            return report
        reasons = list(missing_inputs)
        if not end_present:
            reasons.append("log.smoke has not reached End")
        report.update(
            {
                "status": "incomplete",
                "complete": False,
                "passed": False,
                "hard_gates": {},
                "incomplete_reasons": reasons,
            }
        )
        return report

    assert time_dir is not None
    parsed: dict[str, ParsedField] = {}
    field_errors: dict[str, str] = {}
    for name in REQUIRED_FIELDS:
        path = field_path(time_dir, name)
        if path is None:
            field_errors[name] = "field file is missing"
            continue
        try:
            parsed[name] = parse_internal_field(path, name)
        except (OSError, FieldParseError, ValueError) as exc:
            field_errors[name] = str(exc)

    write_precision = control_value(case, "writePrecision")
    if write_precision is None:
        write_precision = 8.0
    alpha_tolerance = 10.0 ** (-int(write_precision))

    field_report: dict[str, Any] = {"errors": field_errors}
    if "alpha.water" in parsed:
        alpha = parsed["alpha.water"]
        alpha_values = alpha.values
        assert alpha.kind == "scalar"
        alpha_min = min(alpha_values)  # type: ignore[arg-type]
        alpha_max = max(alpha_values)  # type: ignore[arg-type]
        field_report["alpha.water"] = {
            "path": str(alpha.path),
            "representation": alpha.representation,
            "count": alpha.declared_count,
            "min": alpha_min,
            "max": alpha_max,
            "write_precision": int(write_precision),
            "bounds_tolerance": alpha_tolerance,
            "bounded_within_write_tolerance": (
                alpha_min >= -alpha_tolerance and alpha_max <= 1.0 + alpha_tolerance
            ),
        }
    for name in ("p", "p_rgh"):
        if name not in parsed:
            continue
        field = parsed[name]
        assert field.kind == "scalar"
        values = field.values
        minimum = min(values)  # type: ignore[arg-type]
        maximum = max(values)  # type: ignore[arg-type]
        field_report[name] = {
            "path": str(field.path),
            "representation": field.representation,
            "count": field.declared_count,
            "min": minimum,
            "max": maximum,
            "finite": all(math.isfinite(item) for item in values),  # type: ignore[arg-type]
            "strictly_positive": minimum > 0.0,
        }
    if "U" in parsed:
        velocity = parsed["U"]
        assert velocity.kind == "vector"
        vectors = velocity.values
        components = [component for vector in vectors for component in vector]  # type: ignore[union-attr]
        magnitudes = [math.sqrt(sum(component * component for component in vector)) for vector in vectors]  # type: ignore[union-attr]
        field_report["U"] = {
            "path": str(velocity.path),
            "representation": velocity.representation,
            "count": velocity.declared_count,
            "finite": all(math.isfinite(item) for item in components),
            "max_magnitude": max(magnitudes),
        }
    report["fields"] = field_report

    configured_max_co = log_metrics["configured_maxCo"]
    configured_max_alpha_co = log_metrics["configured_maxAlphaCo"]
    max_co = log_metrics["maxCo"]
    max_alpha_co = log_metrics["maxAlphaCo"]
    gates = {
        "target_time_reached": bool(log_metrics["target_time_reached"]),
        "did_not_advance_past_target": bool(log_metrics["did_not_advance_past_target"]),
        "End_present": end_present,
        "wrapper_exit_zero": bool(log_metrics["wrapper_exit_zero"]),
        "no_fatal": not fatal_lines,
        "no_log_nonfinite": not nonfinite_lines,
        "no_pressure_clipping": not clipping_lines,
        "Co_metrics_present_and_finite": (
            max_co is not None
            and max_alpha_co is not None
            and bool(delta_ts)
            and finite_nonnegative([max_co, max_alpha_co, *delta_ts])
        ),
        "maxCo_within_configured_limit": (
            max_co is not None
            and configured_max_co is not None
            and max_co <= configured_max_co + 1.0e-12
        ),
        "maxAlphaCo_within_configured_limit": (
            max_alpha_co is not None
            and configured_max_alpha_co is not None
            and max_alpha_co <= configured_max_alpha_co + 1.0e-12
        ),
        "all_required_fields_parsed": not field_errors
        and all(name in parsed for name in REQUIRED_FIELDS),
        "alpha_bounded_within_write_tolerance": bool(
            field_report.get("alpha.water", {}).get("bounded_within_write_tolerance")
        ),
        "p_finite_and_positive": bool(
            field_report.get("p", {}).get("finite")
            and field_report.get("p", {}).get("strictly_positive")
        ),
        "p_rgh_finite_and_positive": bool(
            field_report.get("p_rgh", {}).get("finite")
            and field_report.get("p_rgh", {}).get("strictly_positive")
        ),
        "U_finite": bool(field_report.get("U", {}).get("finite")),
    }
    failed_gates = [name for name, accepted in gates.items() if not accepted]
    report.update(
        {
            "complete": True,
            "hard_gates": gates,
            "failed_gates": failed_gates,
            "status": "passed" if not failed_gates else "failed",
            "passed": not failed_gates,
        }
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        action="append",
        choices=tuple(LEVELS),
        help="audit one level; repeat as needed (default: all levels)",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="return nonzero if any selected level is failed or incomplete",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = args.level or list(LEVELS)
    reports: dict[str, dict[str, Any]] = {}
    for level in selected:
        case = LEVELS[level]
        report = audit_level(level, case)
        output = case / "smoke_audit.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports[level] = report

    summary = {
        "audit_type": "read_only_no_OpenFOAM_process",
        "levels": {
            name: {"status": item["status"], "passed": item["passed"]}
            for name, item in reports.items()
        },
        "all_selected_passed": all(item["passed"] for item in reports.values()),
    }
    print(json.dumps(summary, indent=2))
    if args.strict_exit and not summary["all_selected_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
