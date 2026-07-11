#!/usr/bin/env python3
"""Restore an immutable solver run environment from its manifest."""
from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path


ENVIRONMENT_KEYS = {
    "stage": "CASEB_STAGE",
    "end_time_s": "CASEB_END_TIME",
    "mesh_preset": "CASEB_MESH",
    "valve_mode": "CASEB_VALVE_MODE",
    "valve_open_time_s": "CASEB_VALVE_OPEN_TIME",
    "valve_seal_speed_m_per_s": "CASEB_VALVE_SEAL_SPEED",
    "initial_air_head_m": "CASEB_HA0",
    "gas_equation_of_state": "CASEB_GAS_EOS",
    "max_co": "CASEB_MAX_CO",
    "max_alpha_co": "CASEB_MAX_ALPHA_CO",
    "max_delta_t_s": "CASEB_MAX_DELTA_T",
    "field_write_interval_s": "CASEB_WRITE_INTERVAL",
    "c_alpha": "CASEB_C_ALPHA",
    "alpha_smooth_curvature_iterations": "CASEB_ALPHA_SMOOTH_CURVATURE",
}

NUMERIC_KEYS = {
    "end_time_s",
    "valve_open_time_s",
    "valve_seal_speed_m_per_s",
    "initial_air_head_m",
    "max_co",
    "max_alpha_co",
    "max_delta_t_s",
    "field_write_interval_s",
    "c_alpha",
    "alpha_smooth_curvature_iterations",
}


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Run manifest does not exist: {path}")
    manifest = json.loads(path.read_text())
    missing = sorted(set(ENVIRONMENT_KEYS) - set(manifest))
    if missing:
        raise ValueError(f"Run manifest is missing controls: {', '.join(missing)}")
    if manifest["stage"] not in {"hold", "smoke", "full"}:
        raise ValueError("Only solver-stage manifests can be resumed")
    if manifest["mesh_preset"] not in {"base", "refined"}:
        raise ValueError("Unknown mesh preset in run manifest")
    if manifest["valve_mode"] not in {"opening", "closed", "instant"}:
        raise ValueError("Unknown valve mode in run manifest")
    if manifest["gas_equation_of_state"] not in {"perfectGas", "rhoConst"}:
        raise ValueError("Unknown gas equation of state in run manifest")
    for key in NUMERIC_KEYS:
        value = float(manifest[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key} in run manifest")
    if not float(manifest["alpha_smooth_curvature_iterations"]).is_integer():
        raise ValueError("Curvature smoothing iterations must be an integer")
    return manifest


def shell_exports(manifest: dict) -> str:
    rows = []
    for manifest_key, environment_key in ENVIRONMENT_KEYS.items():
        value = shlex.quote(str(manifest[manifest_key]))
        rows.append(f"export {environment_key}={value}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(shell_exports(manifest))


if __name__ == "__main__":
    main()
