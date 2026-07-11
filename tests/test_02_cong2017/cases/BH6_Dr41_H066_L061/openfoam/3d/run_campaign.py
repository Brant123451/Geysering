#!/usr/bin/env python3
"""Run the required B-H6 checks in disposable work directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = (
    "static",
    "smoke",
    "base",
    "refined",
    "valve-fast",
    "valve-slow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profiles",
        nargs="*",
        choices=DEFAULT_PROFILES,
        default=list(DEFAULT_PROFILES),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "results",
    )
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--np", type=int, default=min(os.cpu_count() or 1, 6))
    return parser.parse_args()


def source_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        "results",
        "0",
        "postProcessing",
        "dynamicCode",
        "__pycache__",
        "bh6-3d.msh",
        "geometry_audit.runtime.json",
        "checkMesh.runtime.json",
        "initial_audit.runtime.json",
        "valve_schedule.runtime.tsv",
    }
    ignored.update(name for name in names if name.startswith("processor"))
    ignored.update(name for name in names if name.startswith("log."))
    ignored.update(
        name
        for name in names
        if name.replace(".", "", 1).isdigit() and name != "0.orig"
    )
    return ignored.intersection(names)


def run_profile(
    profile: str,
    work_root: Path,
    results_root: Path,
    ranks: int,
) -> None:
    work_case = work_root / profile
    shutil.copytree(HERE, work_case, ignore=source_ignore)
    result_dir = (results_root / profile).resolve()
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        {
            "BH6_PROFILE": profile,
            "BH6_RESULTS_DIR": str(result_dir),
            "BH6_REFERENCE_CASE_ROOT": str(HERE.parents[1]),
            "OPENFOAM_NP": str(ranks),
        }
    )
    print(f"[campaign] {profile}: work={work_case} results={result_dir}", flush=True)
    subprocess.run(["bash", "Allrun"], cwd=work_case, env=env, check=True)


def aggregate(results_root: Path, profiles: list[str], work_root: Path) -> None:
    rows: list[dict] = []
    for profile in profiles:
        path = results_root / profile / "metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "profile": profile,
                    "simulation_end_s": metrics.get("simulation_end_s"),
                    "cells": metrics.get("mesh", {}).get("cells"),
                    "opening_duration_s": metrics.get("valve", {}).get(
                        "opening_duration_s"
                    ),
                    "Ta_3d_s": metrics.get("events", {}).get("Ta_3d_s"),
                    "catch_time_3d_s": metrics.get("events", {}).get(
                        "interface_catch_3d_s"
                    ),
                    "Yfs_max_3d_m": metrics.get("events", {}).get(
                        "Yfs_max_3d_m"
                    ),
                    "PT1_post_arrival_peak_H_over_H0": metrics.get(
                        "pressure", {}
                    ).get("post_arrival_peak_H_over_H0"),
                    "water_above_rim": metrics.get("events", {}).get(
                        "water_above_rim"
                    ),
                    "liquid_balance_relative_error": metrics.get(
                        "conservation", {}
                    ).get("liquid_volume_relative_residual"),
                    "gas_balance_relative_error": metrics.get(
                        "conservation", {}
                    ).get("gas_mass_relative_residual"),
                }
            )

    summary = {
        "case": "BH6_Dr41_H066_L061",
        "work_root": str(work_root),
        "profiles_requested": profiles,
        "profiles_completed": [row["profile"] for row in rows],
        "results": rows,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    full_rows = [
        row
        for row in rows
        if row["profile"] in {"base", "refined", "valve-fast", "valve-slow"}
    ]
    if not full_rows:
        return
    labels = [row["profile"] for row in full_rows]
    yfs = [row["Yfs_max_3d_m"] for row in full_rows]
    pressure = [
        row["PT1_post_arrival_peak_H_over_H0"] for row in full_rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].bar(labels, yfs, color="#4c78a8")
    axes[0].axhline(1.21, color="0.2", ls="--", label="experiment")
    axes[0].axhline(1.8, color="#b22222", ls=":", label="physical rim")
    axes[0].set_ylabel("maximum $Y_{fs}$ [m]")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(labels, pressure, color="#f58518")
    axes[1].axhline(1.4, color="0.2", ls="--", label="experiment")
    axes[1].set_ylabel("post-arrival PT1 peak $H/H_0$")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(results_root / "campaign_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = args.results_dir.resolve()
    if args.work_root:
        work_root = args.work_root.resolve()
        work_root.mkdir(parents=True, exist_ok=False)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="cong-bh6-3d-"))
    print(f"[campaign] disposable work root: {work_root}", flush=True)

    completed: list[str] = []
    try:
        for profile in args.profiles:
            run_profile(profile, work_root, results_root, args.np)
            completed.append(profile)
    finally:
        aggregate(results_root, completed, work_root)


if __name__ == "__main__":
    main()
