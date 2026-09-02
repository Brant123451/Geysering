#!/usr/bin/env python3
"""Apply the preregistered numerical and experimental gate to the H1 rerun."""
from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def relative_error(actual: float | None, expected: float) -> float | None:
    return None if actual is None else abs(actual - expected) / abs(expected)


def log_courant(path: Path) -> tuple[float | None, float | None, int]:
    courant: list[float] = []
    interface: list[float] = []
    fatal = 0
    for line in path.read_text(errors="ignore").splitlines():
        fatal += int("FOAM FATAL" in line)
        match = re.match(r"^Courant Number mean: .* max: ([0-9.eE+-]+)$", line)
        if match:
            courant.append(float(match.group(1)))
        match = re.match(r"^Interface Courant Number mean: .* max: ([0-9.eE+-]+)$", line)
        if match:
            interface.append(float(match.group(1)))

    def p99(values: list[float]) -> float | None:
        if not values:
            return None
        values.sort()
        return values[int(0.99 * (len(values) - 1))]

    return p99(courant), p99(interface), fatal


def main() -> None:
    policy = json.loads((HERE / "qualification_policy.json").read_text())
    metrics = json.loads((RESULTS / "openfoam_2d_metrics.json").read_text())
    audit = json.loads((RESULTS / "run_record" / "paper_audit.json").read_text())
    model = metrics["model"]
    experiment = metrics["experiment"]
    co_p99, alpha_co_p99, fatal_count = log_courant(RESULTS / "run_record" / "log.solve")

    arrival_error = relative_error(model["Ta_s"], experiment["Ta_s"])
    rim_error = relative_error(model["t_free_surface_at_98pct_rim_s"], policy["experiment_rim_time_s"])
    vfs_error = relative_error(model["vfs_m_s"], experiment["vfs_m_s"])
    vint_error = relative_error(model["vint_m_s"], experiment["vint_m_s"])
    checks = {
        "paper_contract_pass": audit["status"] == "PASS",
        "formal_end_reached": metrics["status"]["ended_normally"]
        and metrics["status"]["last_log_time_s"] >= 16.0,
        "no_fatal_error": not metrics["status"]["fatal_error"] and fatal_count == 0,
        "classification_geyser": model["geysering"] is True,
        "arrival_within_10pct": arrival_error is not None
        and arrival_error <= policy["arrival_relative_tolerance"],
        "rim_time_within_20pct": rim_error is not None
        and rim_error <= policy["rim_time_relative_tolerance"],
        "free_surface_speed_within_30pct": vfs_error is not None
        and vfs_error <= policy["rise_speed_relative_tolerance"],
        "gas_front_speed_within_30pct": vint_error is not None
        and vint_error <= policy["rise_speed_relative_tolerance"],
        "courant_p99_within_gate": co_p99 is not None and co_p99 <= policy["courant_p99_max"],
        "interface_courant_p99_within_gate": alpha_co_p99 is not None
        and alpha_co_p99 <= policy["interface_courant_p99_max"],
    }
    payload = {
        "state": "PASS" if all(checks.values()) else "FAIL",
        "run_id": metrics["run_id"],
        "policy": policy,
        "checks": checks,
        "diagnostics": {
            "arrival_relative_error": arrival_error,
            "rim_time_relative_error": rim_error,
            "free_surface_speed_relative_error": vfs_error,
            "gas_front_speed_relative_error": vint_error,
            "courant_p99": co_p99,
            "interface_courant_p99": alpha_co_p99,
        },
        "model": model,
    }
    (HERE / "qualification_status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
