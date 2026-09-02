#!/usr/bin/env python3
"""Wait for the B3 front-view solve, then render its real alpha.water movie."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"
LOG = CASE / "log.compressibleInterIsoFoam"
SLICES = CASE / "postProcessing" / "frontCentrePlane"
OUTPUTS = ROOT / "outputs"
PROGRESS = OUTPUTS / "run_progress.json"
TARGET = 5.3


def tail(path: Path, size: int = 262_144) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode("utf-8", errors="replace")


def latest_time(text: str) -> float | None:
    matches = re.findall(r"^Time = ([0-9.eE+-]+)$", text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def write_progress(status: str, current: float | None, frames: int, detail: str = "") -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "solver_time_s": current,
        "target_solver_time_s": TARGET,
        "front_slice_frames": frames,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(PROGRESS)


def main() -> None:
    while True:
        text = tail(LOG)
        current = latest_time(text)
        frames = len(list(SLICES.glob("*/frontCentre.vtp"))) if SLICES.exists() else 0
        if "FOAM FATAL ERROR" in text:
            write_progress("failed", current, frames, "FOAM FATAL ERROR in solver log")
            raise SystemExit("OpenFOAM failed; see log.compressibleInterIsoFoam")
        if current is not None and current >= TARGET - 1e-7 and "\nEnd\n" in text:
            write_progress("rendering", current, frames)
            break
        write_progress("running", current, frames)
        time.sleep(30)

    movie = OUTPUTS / "B3_openfoam_quasi2d_front_view.mp4"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "render_real_front_slice.py"),
            "--fps", "50",
            "--mp4", str(movie),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = len(list(SLICES.glob("*/frontCentre.vtp")))
    write_progress("complete", TARGET, frames, str(movie))
    builder = ROOT.parent.parent / "scripts" / "caseB_build_real_vof_html.py"
    subprocess.run([sys.executable, str(builder)], cwd=builder.parent, check=True)


if __name__ == "__main__":
    main()
