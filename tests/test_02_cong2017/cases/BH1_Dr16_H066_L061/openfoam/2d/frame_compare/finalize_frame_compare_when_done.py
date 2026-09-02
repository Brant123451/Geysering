#!/usr/bin/env python3
"""Wait for the detached H1 run and build the formal frame viewer in WSL."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OF2D = HERE.parent
DRIVER_LOG = OF2D / "full_run_driver.log"
STATUS = HERE / "finalizer_status.json"


def save(state: str, detail: str) -> None:
    STATUS.write_text(
        json.dumps({"state": state, "detail": detail, "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def main() -> int:
    deadline = time.monotonic() + 24 * 60 * 60
    save("waiting", "Waiting for the formal OpenFOAM SOLVE_DONE marker")
    while time.monotonic() < deadline:
        text = DRIVER_LOG.read_text(errors="ignore") if DRIVER_LOG.exists() else ""
        if "SOLVE_DONE" in text:
            builder = wsl_path(HERE / "build_frame_compare.py")
            completed = subprocess.run(
                ["wsl.exe", "bash", "-lc", f"nice -n 10 python3 '{builder}'"],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                save("complete", "Formal 13 s 1D--2D frame viewer generated")
                return 0
            save("failed", completed.stderr[-5000:] or completed.stdout[-5000:])
            return completed.returncode
        if "FOAM FATAL" in text:
            save("failed", "OpenFOAM driver reported a fatal error")
            return 2
        time.sleep(30)
    save("timed_out", "No SOLVE_DONE marker within 24 hours")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
