#!/usr/bin/env python3
"""Watch the detached H1 qualification run and resume from checkpoints if needed."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DRIVER_LOG = HERE / "full_run_driver.log"
DRIVER_ERR = HERE / "full_run_driver.err.log"
STATUS = HERE / "run_status.json"
CASE_ROOT = "/mnt/e/Geysering/tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/2d"
QUAL_ROOT = f"{CASE_ROOT}/qualification/h1_refined_co015"
RUN_DIR = "/tmp/bh1-2d-study/h1_refined_co015"


def save(state: str, detail: str, latest_time: float | None = None) -> None:
    STATUS.write_text(
        json.dumps(
            {
                "state": state,
                "detail": detail,
                "latest_time_s": latest_time,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def wsl(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["wsl.exe", "bash", "-lc", command], capture_output=True, text=True)


def latest_time() -> float | None:
    result = wsl(
        f"tail -n 500 {RUN_DIR}/log.solve {RUN_DIR}/log.solve.resume 2>/dev/null "
        "| sed -n 's/^Time = //p' | tail -1"
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def solver_running() -> bool:
    # The dedicated wrapper remains alive through prepare, solve and post-process,
    # so it is a more reliable run identifier than matching a generic solver PID.
    return wsl("pgrep -f '/qualification/h1_refined_co015/run_formal.sh' >/dev/null").returncode == 0


def complete() -> bool:
    return DRIVER_LOG.exists() and "QUALIFICATION_DONE" in DRIVER_LOG.read_text(errors="ignore")


def launch_resume() -> subprocess.Popen[bytes]:
    command = (
        f"cd {CASE_ROOT} && "
        "BH1_RUN_ID=h1_refined_co015 "
        f"BH1_CONFIG_PATH={QUAL_ROOT}/case_config.json "
        f"BH1_RESULTS_DIR={QUAL_ROOT}/results "
        "OPENFOAM_NP=1 ./Allrun resume && "
        f"python3 {QUAL_ROOT}/evaluate_qualification.py && "
        f"echo QUALIFICATION_DONE {QUAL_ROOT}"
    )
    stdout = DRIVER_LOG.open("ab")
    stderr = DRIVER_ERR.open("ab")
    return subprocess.Popen(
        ["wsl.exe", "bash", "-lc", command],
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main() -> int:
    deadline = time.monotonic() + 72 * 60 * 60
    stopped_polls = 0
    resume_attempts = 0
    save("monitoring", "H1 qualification run launched", latest_time())
    while time.monotonic() < deadline:
        if complete():
            result = json.loads((HERE / "qualification_status.json").read_text())
            save("complete", f"Qualification gate: {result['state']}", latest_time())
            return 0
        text = DRIVER_LOG.read_text(errors="ignore") if DRIVER_LOG.exists() else ""
        err = DRIVER_ERR.read_text(errors="ignore") if DRIVER_ERR.exists() else ""
        if "FOAM FATAL" in text + err or "Traceback (most recent call last)" in text + err:
            save("failed", "Formal run reported a fatal error", latest_time())
            return 2
        if solver_running():
            stopped_polls = 0
            save("running", "H1 refined solve is active", latest_time())
            time.sleep(60)
            continue
        stopped_polls += 1
        if stopped_polls < 5:
            save("grace", "Solver stopped; allowing driver post-processing to finish", latest_time())
            time.sleep(60)
            continue
        if complete():
            continue
        if resume_attempts >= 2:
            save("failed", "Two checkpoint resume attempts failed", latest_time())
            return 3
        resume_attempts += 1
        save("resuming", f"Checkpoint resume attempt {resume_attempts}", latest_time())
        launch_resume()
        stopped_polls = 0
        time.sleep(60)
    save("timed_out", "H1 qualification run exceeded the 72-hour monitor window", latest_time())
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
