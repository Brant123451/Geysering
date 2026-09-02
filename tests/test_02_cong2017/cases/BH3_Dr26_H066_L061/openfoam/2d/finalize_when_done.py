#!/usr/bin/env python3
"""Wait for the B-H3 13 s run, then build the archived 1D--2D viewer."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = Path("/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq")
DRIVER_OUT = HERE / "results" / "full_run_driver.stdout.log"
DRIVER_ERR = HERE / "results" / "full_run_driver.stderr.log"
RESUME_OUT = HERE / "results" / "resume_run_driver.stdout.log"
RESUME_ERR = HERE / "results" / "resume_run_driver.stderr.log"
STATUS = HERE / "results" / "finalizer_status.json"
COMPARISON = HERE / "comparison"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(state: str, detail: str, steps: list[dict[str, object]] | None = None) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "case": "BH3_Dr26_H066_L061",
        "paper_run": "B-H3",
        "state": state,
        "detail": detail,
        "updated_utc": now(),
        "steps": steps or [],
    }
    STATUS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_step(name: str, command: list[str], steps: list[dict[str, object]]) -> None:
    log_root = COMPARISON / "finalizer_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    save("running", name, steps)
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    (log_root / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    record = {"name": name, "returncode": completed.returncode, "finished_utc": now()}
    steps.append(record)
    if completed.returncode != 0:
        save("failed", f"{name} failed with return code {completed.returncode}", steps)
        raise RuntimeError(f"{name} failed: {completed.stderr[-2000:]}")


def one_d_is_aligned() -> bool:
    path = COMPARISON / "model_1d" / "summary.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        initial = payload["initial_conditions"]
        return (
            payload.get("status") == "COMPLETE"
            and abs(float(initial["model_head_above_pipe_crown_m"]) - 0.61) < 1.0e-10
            and abs(float(payload["geometry_m"]["tank_to_riser"]) - 3.47) < 1.0e-10
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    deadline = time.monotonic() + 36 * 60 * 60
    save("waiting", "Waiting for the complete 13 s OpenFOAM solve and postprocess")
    while time.monotonic() < deadline:
        solve_log = "\n".join(
            path.read_text(errors="ignore")
            for path in (RUN / "log.solve", RUN / "log.solve.resume")
            if path.exists()
        )
        driver_out = DRIVER_OUT.read_text(errors="ignore") if DRIVER_OUT.exists() else ""
        driver_err = DRIVER_ERR.read_text(errors="ignore") if DRIVER_ERR.exists() else ""
        resume_out = RESUME_OUT.read_text(errors="ignore") if RESUME_OUT.exists() else ""
        resume_err = RESUME_ERR.read_text(errors="ignore") if RESUME_ERR.exists() else ""
        if "FOAM FATAL" in solve_log or "FOAM FATAL" in resume_err or "Traceback (most recent call last)" in driver_err + resume_err:
            save("failed", "The OpenFOAM driver reported a fatal error")
            return 2
        complete = (
            "\nEnd\n" in f"\n{solve_log}\n"
            and ("SOLVE_DONE" in driver_out or "RESUME_DONE" in resume_out)
            and (HERE / "results" / "openfoam_2d_metrics.json").exists()
        )
        if complete:
            break
        time.sleep(30)
    else:
        save("timed_out", "No complete 13 s result was observed within 36 hours")
        return 3

    # The geometry-aligned 1D run normally finishes long before OpenFOAM.
    one_d_deadline = time.monotonic() + 60 * 60
    while not one_d_is_aligned() and time.monotonic() < one_d_deadline:
        time.sleep(15)
    if not one_d_is_aligned():
        save("failed", "Geometry-aligned 1D output is missing or uses the wrong crown datum")
        return 4

    steps: list[dict[str, object]] = []
    try:
        run_step("export_openfoam_vtk", ["bash", str(COMPARISON / "export_openfoam_vtk.sh")], steps)
        run_step("build_1d_frames", [sys.executable, str(COMPARISON / "build_1d_frames.py")], steps)
        run_step("build_2d_frames", [sys.executable, str(COMPARISON / "build_2d_frames.py")], steps)
        run_step("build_html", [sys.executable, str(COMPARISON / "build_html.py")], steps)
    except RuntimeError:
        return 5

    one_frames = json.loads((COMPARISON / "model_1d" / "frames.json").read_text(encoding="utf-8"))
    two_frames = json.loads((COMPARISON / "openfoam_2d" / "frames.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "case": "BH3_Dr26_H066_L061",
        "paper_run": "B-H3",
        "status": "COMPLETE",
        "completed_utc": now(),
        "openfoam_time_window_s": [two_frames[0]["time"], two_frames[-1]["time"]],
        "frame_counts": {"model_1d": len(one_frames), "openfoam_2d": len(two_frames)},
        "artifacts": {
            "viewer": "BH3_1d_openfoam2d_frame_compare_13s.html",
            "openfoam_metrics": "../results/openfoam_2d_metrics.json",
            "paper_audit": "../results/run_record/paper_audit.json",
        },
        "comparison_policy": {
            "native_time_axes": True,
            "time_shift_applied": False,
            "outcome_fitting_applied": False,
            "height_datum": "pipe crown",
        },
        "steps": steps,
    }
    (COMPARISON / "completion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save("complete", "B-H3 OpenFOAM run and 1D--2D HTML comparison completed", steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
