#!/usr/bin/env python3
"""Prepare or execute the declared C9 sensitivity matrix.

Run directories are intentionally ignored by Git.  Each row is a genuine
OpenFOAM case generated from the same source; absent runs are reported as
`prepared`, never populated with one-dimensional or synthetic CFD results.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_CASE = HERE / "case"
RUNS = HERE / "runs"
OUTPUT = HERE.parents[1] / "outputs" / "openfoam_3d_mesh_sensitivity.csv"

VARIANTS = {
    "base": [],
    "mesh_refined": ["--mesh-profile", "refined"],
    "time_tight": ["--max-co", "0.10", "--max-alpha-co", "0.075", "--max-dt", "0.000125"],
    "pocket_small": ["--pocket-profile", "pocket_small"],
    "pocket_large": ["--pocket-profile", "pocket_large"],
    "isothermal_air": ["--thermo", "isothermal"],
    "gate_low": ["--gate-area", "0.00672"],
    "gate_high": ["--gate-area", "0.01008"],
    "contact_angle_60": ["--contact-angle", "60"],
    "contact_angle_120": ["--contact-angle", "120"],
    "interface_compression_05": ["--c-alpha", "0.5"],
    "interface_compression_15": ["--c-alpha", "1.5"],
}


def source_ignore(directory, names):
    ignored = set()
    for name in names:
        if name in {"polyMesh", "postProcessing", "0"}:
            ignored.add(name)
        elif name.startswith("processor") or name.startswith("log."):
            ignored.add(name)
        else:
            try:
                float(name)
            except ValueError:
                pass
            else:
                ignored.add(name)
    return ignored


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "mesh", "initialize", "smoke", "phase1", "full"),
        default="prepare",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        for name in selected:
            generator = [sys.executable, "prepare_case.py", "--np", "4", *VARIANTS[name]]
            run(generator, HERE)
            destination = RUNS / name
            if destination.exists() and args.fresh:
                shutil.rmtree(destination)
            if not destination.exists():
                shutil.copytree(SOURCE_CASE, destination, ignore=source_ignore)

            status = "prepared"
            error = ""
            try:
                if args.stage != "prepare":
                    run(["./Allrun.mesh"], destination)
                    status = "mesh_checked"
                if args.stage == "initialize":
                    run(["./Allrun.initialize"], destination)
                    status = "initialized"
                elif args.stage == "smoke":
                    run(["./Allrun.smoke"], destination)
                    status = "smoke_complete"
                elif args.stage == "phase1":
                    run(["./Allrun.phase1"], destination)
                    status = "phase1_complete"
                elif args.stage == "full":
                    run(["./Allrun.solve"], destination)
                    status = "full_complete"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"{exc.cmd!r} exited {exc.returncode}"

            result_dir = destination / "results"
            try:
                run(
                    [
                        sys.executable,
                        str(HERE / "postprocess_openfoam.py"),
                        "--case",
                        str(destination),
                        "--output-dir",
                        str(result_dir),
                    ],
                    HERE,
                )
                metrics = json.loads((result_dir / "openfoam_3d_metrics.json").read_text())
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
                metrics = {}
                if not error:
                    error = str(exc)

            generated = json.loads((destination / "generated_case.json").read_text())
            mesh = metrics.get("mesh", {})
            phase1 = metrics.get("phase_1", {})
            rows.append(
                {
                    "variant": name,
                    "status": status,
                    "error": error,
                    "solver": generated.get("application"),
                    "mesh_profile": generated.get("mesh_profile"),
                    "cells": mesh.get("cells"),
                    "checkMesh_passed": mesh.get("checkMesh_passed"),
                    "maxCo": generated.get("maxCo"),
                    "maxDeltaT_s": generated.get("maxDeltaT"),
                    "pocket_profile": generated.get("pocket_profile"),
                    "analytic_pocket_volume_m3": generated.get("analytic_initial_air_volume_m3"),
                    "gate_area_m2": generated.get("gate_area_m2"),
                    "contact_angle_deg": generated.get("contact_angle_deg"),
                    "cAlpha": generated.get("interface_compression"),
                    "P1m_kPa": phase1.get("P1m_kPa"),
                    "first_top_s": phase1.get("first_riser_top_s"),
                    "geyser_count": metrics.get("simulated_geyser_count"),
                    "air_arrival_s": metrics.get("simulated_air_pocket_arrival_s"),
                    "mass_error": metrics.get("mass_conservation_relative_error"),
                    "gas_mass_error": metrics.get("gas_mass_conservation_relative_error"),
                }
            )
    finally:
        # Keep the tracked source case deterministic after preparing variants.
        run([sys.executable, "prepare_case.py", "--np", "4"], HERE)

    fieldnames = [
        "variant",
        "status",
        "error",
        "solver",
        "mesh_profile",
        "cells",
        "checkMesh_passed",
        "maxCo",
        "maxDeltaT_s",
        "pocket_profile",
        "analytic_pocket_volume_m3",
        "gate_area_m2",
        "contact_angle_deg",
        "cAlpha",
        "P1m_kPa",
        "first_top_s",
        "geyser_count",
        "air_arrival_s",
        "mass_error",
        "gas_mass_error",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
