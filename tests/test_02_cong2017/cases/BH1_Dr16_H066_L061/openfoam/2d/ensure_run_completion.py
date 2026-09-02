#!/usr/bin/env python3
"""Resume the detached formal B-H1 run only if it unexpectedly stops."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DRIVER_LOG = HERE / "full_run_driver.log"
DRIVER_ERR = HERE / "full_run_driver.err.log"
STATUS = HERE / "results" / "completion_watchdog_status.json"


def save(state: str, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps({"state": state, "detail": detail, "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def solver_running() -> bool:
    check = subprocess.run(
        [
            "wsl.exe",
            "bash",
            "-lc",
            "pgrep -f '^/tmp/bh1-2d-build/bin/bh1CompressibleInterFoam( |$)' >/dev/null",
        ],
        capture_output=True,
    )
    return check.returncode == 0


def complete_marker() -> bool:
    return DRIVER_LOG.exists() and "SOLVE_DONE" in DRIVER_LOG.read_text(errors="ignore")


def main() -> int:
    deadline = time.monotonic() + 24 * 60 * 60
    stopped_polls = 0
    resume_attempts = 0
    save("monitoring", "Formal B-H1 solver is active; no restart requested")
    while time.monotonic() < deadline:
        if complete_marker():
            save("complete", "Formal solve and post-processing completed")
            return 0
        if solver_running():
            stopped_polls = 0
            time.sleep(30)
            continue

        stopped_polls += 1
        if stopped_polls < 4:
            save("grace", "Solver exited; allowing the active driver to finish post-processing")
            time.sleep(30)
            continue
        if complete_marker():
            save("complete", "Formal solve and post-processing completed")
            return 0
        if resume_attempts >= 2:
            save("failed", "Two checkpoint resume attempts failed")
            return 2

        resume_attempts += 1
        save("resuming", f"Starting checkpoint resume attempt {resume_attempts}")
        command = (
            "cd /mnt/e/Geysering/tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/2d "
            "&& OPENFOAM_NP=1 ./Allrun resume"
        )
        completed = subprocess.run(["wsl.exe", "bash", "-lc", command], capture_output=True, text=True)
        with DRIVER_LOG.open("a", encoding="utf-8") as handle:
            handle.write("\n" + completed.stdout)
        with DRIVER_ERR.open("a", encoding="utf-8") as handle:
            handle.write("\n" + completed.stderr)
        if completed.returncode == 0 and "SOLVE_DONE" in completed.stdout:
            save("complete", "Checkpoint resume reached 13 s and post-processing completed")
            return 0
        stopped_polls = 0
        time.sleep(30)

    save("timed_out", "Formal run did not complete within the 24-hour monitor window")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
