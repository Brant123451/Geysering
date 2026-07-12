#!/usr/bin/env python3
"""Run isolated BH3 mesh, hold, event, and sensitivity variants.

Runtime OpenFOAM state is kept under /tmp by default. Only compact
CSV/JSON/PNG products are copied back to this source directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
FORMAL_HOLD_DURATION = 1.0
PRODUCT_SUFFIXES = (
    "_metrics.json",
    "_timeseries.csv",
    "_comparison.csv",
    "_summary.png",
)
MESH_INPUTS = (
    Path("Allmesh"),
    Path("make_geometry.py"),
    Path("make_runtime_config.py"),
    Path("mesh_audit.py"),
    Path("system/changeDictionaryDict"),
    Path("system/topoSetDict.regions"),
)


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
    alpha_smooth_curvature: int = 0
    sample_interval: float = 5.0e-3
    surface_tension: float = 0.072
    initial_interface_thickness: float = 0.015


VARIANTS = (
    Variant(
        "closed_base",
        "smoke",
        mode="closed",
        valve="closed",
        end_time=1.0,
        sample_interval=1.0e-3,
    ),
    Variant(
        "closed_sigma_zero",
        "diagnostic",
        mode="closed",
        valve="closed",
        end_time=0.05,
        sample_interval=1.0e-3,
        surface_tension=0.0,
    ),
    Variant(
        "closed_sharp_sigma_zero",
        "diagnostic",
        mode="closed",
        valve="closed",
        end_time=0.05,
        sample_interval=1.0e-3,
        surface_tension=0.0,
        initial_interface_thickness=0.0,
    ),
    Variant(
        "closed_refined_sigma_zero",
        "diagnostic",
        mesh="refined",
        mode="closed",
        valve="closed",
        end_time=0.05,
        sample_interval=1.0e-3,
        surface_tension=0.0,
    ),
    Variant(
        "closed_refined_sigma_072",
        "diagnostic",
        mesh="refined",
        mode="closed",
        valve="closed",
        end_time=0.05,
        sample_interval=1.0e-3,
    ),
    Variant("open_smoke", "smoke", end_time=0.02, sample_interval=1.0e-3),
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
    Variant(
        "initial_sharp",
        "sensitivity",
        initial_interface_thickness=0.0,
    ),
    Variant("sigma_zero", "sensitivity", surface_tension=0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=("smoke", "diagnostic", "core", "sensitivity", "all"),
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
            "outputs",
            "_work",
            "__pycache__",
            "*.runtime",
            "linux*",
        ),
    )


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[{cwd.name}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def fingerprint(paths: list[Path] | tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(relative.as_posix().encode())
        digest.update((HERE / relative).read_bytes())
    return digest.hexdigest()


def source_fingerprint() -> str:
    paths = [
        path.relative_to(HERE)
        for path in HERE.rglob("*")
        if path.is_file()
        and "outputs" not in path.relative_to(HERE).parts
        and "__pycache__" not in path.relative_to(HERE).parts
        and "polyMesh" not in path.relative_to(HERE).parts
        and "triSurface" not in path.relative_to(HERE).parts
        and "postProcessing" not in path.relative_to(HERE).parts
        and not any(
            part == "0"
            or part.startswith("processor")
            or part.startswith("linux")
            for part in path.relative_to(HERE).parts
        )
        and not path.name.endswith(".runtime")
        and not path.name.startswith("log.")
        and path.suffix != ".msh"
    ]
    return fingerprint(paths)


def mesh_cache_valid(cache: Path, profile: str, expected_fingerprint: str) -> bool:
    boundary = cache / "constant" / "polyMesh" / "boundary"
    audit = cache / "outputs" / f"mesh_{profile}.json"
    if not boundary.is_file() or not audit.is_file():
        return False
    try:
        data = json.loads(audit.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("profile") == profile
        and data.get("checkMesh_allGeometry_allTopology") is True
        and data.get("source_fingerprint") == expected_fingerprint
    )


def ensure_mesh(profile: str, root: Path, force: bool) -> Path:
    cache = root / f"mesh-{profile}"
    expected_fingerprint = fingerprint(MESH_INPUTS)
    if force and cache.exists():
        shutil.rmtree(cache)
    if mesh_cache_valid(cache, profile, expected_fingerprint):
        return cache
    if cache.exists():
        shutil.rmtree(cache)
    copy_source(cache)
    env = os.environ.copy()
    env["MESH_PROFILE"] = profile
    run(["bash", "./Allmesh"], cache, env)
    audit = cache / "outputs" / f"mesh_{profile}.json"
    data = json.loads(audit.read_text(encoding="utf-8"))
    data["source_fingerprint"] = expected_fingerprint
    audit.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if not mesh_cache_valid(cache, profile, expected_fingerprint):
        raise RuntimeError(f"Mesh cache validation failed for profile {profile}")
    return cache


def copy_products(runtime: Path, source_output: Path, run_id: str) -> None:
    source_output.mkdir(parents=True, exist_ok=True)
    for product in (runtime / "outputs").glob(f"{run_id}_*"):
        shutil.copy2(product, source_output / product.name)
    for product in (runtime / "outputs").glob("mesh_*.json"):
        shutil.copy2(product, source_output / product.name)


def outputs_complete(
    source_output: Path,
    variant: Variant,
    expected_fingerprint: str,
) -> bool:
    products = [source_output / f"{variant.run_id}{suffix}" for suffix in PRODUCT_SUFFIXES]
    if not all(path.is_file() and path.stat().st_size > 0 for path in products):
        return False
    try:
        data = json.loads(products[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    complete = (
        data.get("run_id") == variant.run_id
        and data.get("run_mode") == variant.mode
        and float(data.get("simulated_end_time_s", -1.0)) >= variant.end_time - 1.0e-9
        and data.get("source_fingerprint") == expected_fingerprint
    )
    if not complete:
        return False
    if (
        variant.mode == "closed"
        and variant.end_time >= FORMAL_HOLD_DURATION - 1.0e-9
    ):
        return data.get("closed_hold", {}).get("pass") is True
    return True


def annotate_metrics(
    runtime: Path, variant: Variant, expected_fingerprint: str
) -> None:
    path = runtime / "outputs" / f"{variant.run_id}_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_fingerprint"] = expected_fingerprint
    data["numerical_controls"] = {
        "mesh_profile": variant.mesh,
        "c_alpha": variant.c_alpha,
        "initial_interface_thickness_m": variant.initial_interface_thickness,
        "alpha_smooth_curvature": variant.alpha_smooth_curvature,
        "max_co": variant.max_co,
        "max_alpha_co": variant.max_alpha_co,
        "max_delta_t_s": variant.max_delta_t,
        "sample_interval_s": variant.sample_interval,
        "surface_tension_n_per_m": variant.surface_tension,
    }
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    expected_fingerprint = source_fingerprint()
    closed_variant = next(item for item in VARIANTS if item.run_id == "closed_base")
    needs_closed_gate = any(
        item.group in {"core", "sensitivity"} for item in selected
    )
    if (
        needs_closed_gate
        and closed_variant not in selected
        and not outputs_complete(
            source_output, closed_variant, expected_fingerprint
        )
    ):
        raise RuntimeError(
            "Current-source closed_base output is absent or failed; "
            "run --variant closed_base and pass the static gate first"
        )

    for variant in selected:
        if (
            outputs_complete(source_output, variant, expected_fingerprint)
            and not args.force
        ):
            print(f"[{variant.run_id}] complete current compact output exists; skipping")
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
                "SAMPLE_INTERVAL": str(variant.sample_interval),
                "ALPHA_SMOOTH_CURVATURE": str(
                    variant.alpha_smooth_curvature
                ),
                "SURFACE_TENSION": str(variant.surface_tension),
                "INITIAL_INTERFACE_THICKNESS": str(
                    variant.initial_interface_thickness
                ),
                "MESH_PROFILE": variant.mesh,
                "OPENFOAM_NP": str(args.np),
                "REFERENCE_ROOT": str(HERE.parents[1]),
            }
        )
        try:
            run(["bash", "./Allrun"], runtime, env)
        except subprocess.CalledProcessError:
            metrics_path = runtime / "outputs" / f"{variant.run_id}_metrics.json"
            if metrics_path.is_file():
                annotate_metrics(runtime, variant, expected_fingerprint)
                copy_products(runtime, source_output, variant.run_id)
            raise
        annotate_metrics(runtime, variant, expected_fingerprint)
        copy_products(runtime, source_output, variant.run_id)
        if variant.mode == "closed" and not outputs_complete(
            source_output, variant, expected_fingerprint
        ):
            raise RuntimeError(
                f"{variant.run_id} completed but failed the closed-hold gate"
            )

    subprocess.run(
        ["python3", "summarize_sensitivities.py", "--outputs", "outputs"],
        cwd=HERE,
        check=True,
    )


if __name__ == "__main__":
    main()
