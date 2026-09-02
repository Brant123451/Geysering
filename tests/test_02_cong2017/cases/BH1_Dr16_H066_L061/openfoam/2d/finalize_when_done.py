#!/usr/bin/env python3
"""Finish the comparison after a detached full OpenFOAM run exits."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DRIVER_LOG = HERE / "full_run_driver.log"
STATUS = HERE / "results" / "finalizer_status.json"


def save(state: str, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "detail": detail,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    STATUS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    deadline = time.monotonic() + 24 * 60 * 60
    save("waiting", f"Waiting for SOLVE_DONE in {DRIVER_LOG}")
    while time.monotonic() < deadline:
        text = DRIVER_LOG.read_text(errors="ignore") if DRIVER_LOG.exists() else ""
        if "SOLVE_DONE" in text:
            command = [
                sys.executable,
                str(HERE / "compare_1d_2d.py"),
                "--results-dir",
                str(HERE / "results"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode == 0:
                save("complete", "Full-run 1D-2D-experiment comparison generated")
                return 0
            save("failed", completed.stderr[-4000:])
            return completed.returncode
        if "FOAM FATAL" in text or "Traceback (most recent call last)" in text:
            save("failed", "Detached run reported a fatal error; comparison was not finalized")
            return 2
        time.sleep(30)
    save("timed_out", "No SOLVE_DONE marker was observed within 24 hours")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
