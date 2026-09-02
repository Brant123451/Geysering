#!/usr/bin/env python3
"""Generate the final B-H1 paper-consistency report after the formal solve."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "full_run_driver.log"
STATUS = HERE / "paper_consistency_finalizer_status.json"
WSL_HERE = "/mnt/e/Geysering/tests/test_02_cong2017/cases/BH1_Dr16_H066_L061/openfoam/2d/qualification/h1_refined_co015"


def save(state: str, detail: str) -> None:
    STATUS.write_text(
        json.dumps(
            {"state": state, "detail": detail, "updated_utc": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    deadline = time.monotonic() + 120 * 60 * 60
    save("waiting", "Waiting for the completed 16 s H1 solve and qualification output")
    while time.monotonic() < deadline:
        text = DRIVER.read_text(errors="ignore") if DRIVER.exists() else ""
        if "QUALIFICATION_DONE" in text:
            completed = subprocess.run(
                [
                    "wsl.exe",
                    "python3",
                    f"{WSL_HERE}/paper_consistency.py",
                    "--results-dir",
                    f"{WSL_HERE}/results",
                    "--run-dir",
                    "/tmp/bh1-2d-study/h1_refined_co015",
                    "--output-dir",
                    WSL_HERE,
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                report = json.loads((HERE / "paper_consistency_report.json").read_text())
                save("complete", report["overall_assessment"])
                return 0
            save("failed", completed.stderr[-4000:])
            return completed.returncode
        if "FOAM FATAL" in text or "Traceback (most recent call last)" in text:
            save("failed", "Formal H1 driver reported an error")
            return 2
        time.sleep(60)
    save("timed_out", "Formal H1 run did not finish within 120 hours")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
