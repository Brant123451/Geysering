#!/usr/bin/env python3
"""Run isolated BH3 mesh, hold, event, and sensitivity variants.

Runtime OpenFOAM state is kept under /tmp by default. Only compact
CSV/JSON/PNG products are copied back to this source directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Variant:
    run_id: str
    group: str
    mesh: str = "base"
    mode: str = "event"
    valve: str = "instant"
    end_time: float = 13.0
    c_alpha: float = 1.0
    max_co: float = 0.25
    max_alpha_co: float = 0.15
    max_delta_t: float = 5.0e-4


VARIANTS = (
    Variant("closed_base", "smoke", mode="closed", valve="closed", end_time=1.0),
    Variant("open_smoke", "smoke", end_time=0.02),
    Variant("base_nominal", "core"),
    Variant("refined_nominal", "core", mesh="refined"),
    Variant(
        "dt_fine",
        "sensitivity",
        max_co=0.125,
        max_alpha_co=0.075,
        max_delta_t=2.5e-4,
    ),
    Variant("valve_0p2", "sensitivity", valve="0.2"),
    Variant("valve_0p5", "sensitivity", valve="0.5"),
    Variant("interface_diffuse", "sensitivity", c_alpha=0.5),
    Variant("interface_sharp", "sensitivity", c_alpha=2.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=("smoke", "core", "sensitivity", "all"),
        default="smoke",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(item.run_id for item in VARIANTS),
        help="run only named variant; may be repeated",
    )
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/bh3-study"))
    parser.add_argument("--np", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_source(destination: Path) -> None:
    shutil.copytree(
        HERE,
        destination,
        ignore=shutil.ignore_patterns(
            "0",
            "[1-9]*",
            "processor*",
            "postProcessing",
            "polyMesh",
            "triSurface",
            "log.*",
            "_work",
            "__pycache__",
        ),
    )


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[{cwd.name}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_mesh(profile: str, root: Path, force: bool) -> Path:
    cache = root / f"mesh-{profile}"
    marker = cache / "constant" / "polyMesh" / "boundary"
    if force and cache.exists():
        shutil.rmtree(cache)
    if marker.exists():
        return cache
    if cache.exists():
        shutil.rmtree(cache)
    copy_source(cache)
    env = os.environ.copy()
    env["MESH_PROFILE"] = profile
    run(["bash", "./Allmesh"], cache, env)
    return cache


def copy_products(runtime: Path, source_output: Path, run_id: str) -> None:
    source_output.mkdir(parents=True, exist_ok=True)
    for product in (runtime / "outputs").glob(f"{run_id}_*"):
        shutil.copy2(product, source_output / product.name)
    for product in (runtime / "outputs").glob("mesh_*.json"):
        shutil.copy2(product, source_output / product.name)


def main() -> None:
    args = parse_args()
    if args.np < 1:
        raise ValueError("--np must be positive")
    selected = [
        item
        for item in VARIANTS
        if (
            (args.variant and item.run_id in args.variant)
            or (not args.variant and (args.group == "all" or item.group == args.group))
        )
    ]
    args.work_root.mkdir(parents=True, exist_ok=True)
    source_output = HERE / "outputs"

    for variant in selected:
        final_metrics = source_output / f"{variant.run_id}_metrics.json"
        if final_metrics.exists() and not args.force:
            print(f"[{variant.run_id}] compact output exists; skipping")
            continue
        mesh_cache = ensure_mesh(variant.mesh, args.work_root, args.force)
        runtime = args.work_root / variant.run_id
        if runtime.exists():
            shutil.rmtree(runtime)
        copy_source(runtime)
        shutil.copytree(
            mesh_cache / "constant" / "polyMesh",
            runtime / "constant" / "polyMesh",
        )
        for mesh_product in (mesh_cache / "outputs").glob("mesh_*.json"):
            (runtime / "outputs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(mesh_product, runtime / "outputs" / mesh_product.name)

        env = os.environ.copy()
        env.update(
            {
                "RUN_ID": variant.run_id,
                "RUN_MODE": variant.mode,
                "VALVE_OPENING": variant.valve,
                "END_TIME": str(variant.end_time),
                "C_ALPHA": str(variant.c_alpha),
                "MAX_CO": str(variant.max_co),
                "MAX_ALPHA_CO": str(variant.max_alpha_co),
                "MAX_DELTA_T": str(variant.max_delta_t),
                "OPENFOAM_NP": str(args.np),
                "REFERENCE_ROOT": str(HERE.parents[1]),
            }
        )
        run(["bash", "./Allrun"], runtime, env)
        copy_products(runtime, source_output, variant.run_id)

    subprocess.run(
        ["python3", "summarize_sensitivities.py", "--outputs", "outputs"],
        cwd=HERE,
        check=True,
    )


if __name__ == "__main__":
    main()
