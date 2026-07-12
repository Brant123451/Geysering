#!/usr/bin/env python3
"""Restore an immutable solver run environment from its manifest."""
from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path


TWOPHASEFLOW_COMMIT = "de9826f9ffb24f4b635ac97fd388ebd560cfc174"

ENVIRONMENT_KEYS = {
    "stage": "CASEB_STAGE",
    "end_time_s": "CASEB_END_TIME",
    "mesh_preset": "CASEB_MESH",
    "valve_mode": "CASEB_VALVE_MODE",
    "valve_open_time_s": "CASEB_VALVE_OPEN_TIME",
    "valve_seal_speed_m_per_s": "CASEB_VALVE_SEAL_SPEED",
    "initial_air_head_m": "CASEB_HA0",
    "gas_equation_of_state": "CASEB_GAS_EOS",
    "hydrostatic_initialization": "CASEB_HYDROSTATIC_INITIALIZATION",
    "n_hydrostatic_correctors": "CASEB_HYDROSTATIC_CORRECTORS",
    "max_co": "CASEB_MAX_CO",
    "max_alpha_co": "CASEB_MAX_ALPHA_CO",
    "max_capillary_num": "CASEB_MAX_CAPILLARY_NUM",
    "max_delta_t_s": "CASEB_MAX_DELTA_T",
    "field_write_interval_s": "CASEB_WRITE_INTERVAL",
    "c_alpha": "CASEB_C_ALPHA",
    "advection_scheme": "CASEB_ADVECTION_SCHEME",
    "reconstruction_scheme": "CASEB_RECONSTRUCTION_SCHEME",
    "reconstruction_iterations": "CASEB_RECONSTRUCTION_ITERATIONS",
    "reconstruction_tolerance": "CASEB_RECONSTRUCTION_TOL",
    "interpolate_normal": "CASEB_INTERPOLATE_NORMAL",
    "curvature_model": "CASEB_CURVATURE_MODEL",
    "curvature_value_per_m": "CASEB_CURVATURE_VALUE",
    "curvature_from_trace": "CASEB_CURV_FROM_TR",
    "n_alpha_bounds": "CASEB_N_ALPHA_BOUNDS",
    "n_alpha_corr": "CASEB_N_ALPHA_CORR",
    "n_alpha_subcycles": "CASEB_N_ALPHA_SUBCYCLES",
    "n_outer_correctors": "CASEB_N_OUTER_CORRECTORS",
    "n_pressure_correctors": "CASEB_N_CORRECTORS",
    "n_non_orthogonal_correctors": "CASEB_N_NON_ORTHOGONAL_CORRECTORS",
    "alpha_clip": "CASEB_ALPHA_CLIP",
}

NUMERIC_KEYS = {
    "end_time_s",
    "valve_open_time_s",
    "valve_seal_speed_m_per_s",
    "initial_air_head_m",
    "n_hydrostatic_correctors",
    "max_co",
    "max_alpha_co",
    "max_capillary_num",
    "max_delta_t_s",
    "field_write_interval_s",
    "c_alpha",
    "reconstruction_iterations",
    "reconstruction_tolerance",
    "curvature_value_per_m",
    "n_alpha_bounds",
    "n_alpha_corr",
    "n_alpha_subcycles",
    "n_outer_correctors",
    "n_pressure_correctors",
    "n_non_orthogonal_correctors",
}


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Run manifest does not exist: {path}")
    manifest = json.loads(path.read_text())
    # Manifests predating the discrete initializer used the analytic fields.
    manifest.setdefault("hydrostatic_initialization", "analytic")
    manifest.setdefault("n_hydrostatic_correctors", 5)
    required = set(ENVIRONMENT_KEYS) | {
        "solver",
        "two_phase_flow_commit",
        "valve_representation",
        "time_control",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Run manifest is missing controls: {', '.join(missing)}")
    if manifest["stage"] not in {"hold", "smoke", "full"}:
        raise ValueError("Only solver-stage manifests can be resumed")
    if manifest["mesh_preset"] not in {"base", "refined"}:
        raise ValueError("Unknown mesh preset in run manifest")
    if manifest["valve_mode"] not in {"opening", "closed", "instant"}:
        raise ValueError("Unknown valve mode in run manifest")
    expected_valve = (
        "conformalNoSlipBaffle"
        if manifest["valve_mode"] == "closed"
        else "dissipativeResistance"
    )
    if manifest["valve_representation"] != expected_valve:
        raise ValueError("Run manifest has an inconsistent valve representation")
    if manifest["time_control"] != "runTime":
        raise ValueError("Run manifest uses a timestep-disturbing output control")
    if manifest["gas_equation_of_state"] not in {"perfectGas", "rhoConst"}:
        raise ValueError("Unknown gas equation of state in run manifest")
    if manifest["hydrostatic_initialization"] not in {"analytic", "discrete"}:
        raise ValueError("Unknown hydrostatic initialization in run manifest")
    if manifest["solver"] != "compressibleInterFlow":
        raise ValueError("Run manifest was created by a different solver")
    if manifest["two_phase_flow_commit"] != TWOPHASEFLOW_COMMIT:
        raise ValueError("Run manifest uses a different TwoPhaseFlow commit")
    if manifest["advection_scheme"] not in {"isoAdvection", "MULESScheme"}:
        raise ValueError("Unknown advection scheme in run manifest")
    if manifest["reconstruction_scheme"] not in {
        "plicRDF",
        "isoAlpha",
        "gradAlpha",
    }:
        raise ValueError("Unknown reconstruction scheme in run manifest")
    if manifest["curvature_model"] not in {
        "RDF",
        "fitParaboloid",
        "gradAlpha",
        "constantCurvature",
    }:
        raise ValueError("Unknown curvature model in run manifest")
    if (
        manifest["curvature_model"] == "constantCurvature"
        and manifest["stage"] != "hold"
    ):
        raise ValueError("constantCurvature is only valid for hold diagnostics")
    if not isinstance(manifest["interpolate_normal"], bool):
        raise ValueError("interpolate_normal must be a boolean")
    if not isinstance(manifest["curvature_from_trace"], bool):
        raise ValueError("curvature_from_trace must be a boolean")
    if not isinstance(manifest["alpha_clip"], bool):
        raise ValueError("alpha_clip must be a boolean")
    for key in NUMERIC_KEYS:
        value = float(manifest[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key} in run manifest")
    if float(manifest["reconstruction_tolerance"]) <= 0:
        raise ValueError("reconstruction_tolerance must be positive")
    positive_integer_keys = {
        "n_alpha_bounds",
        "reconstruction_iterations",
        "n_alpha_corr",
        "n_alpha_subcycles",
        "n_outer_correctors",
        "n_pressure_correctors",
        "n_hydrostatic_correctors",
    }
    for key in positive_integer_keys:
        if not float(manifest[key]).is_integer() or int(manifest[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    non_orthogonal = manifest["n_non_orthogonal_correctors"]
    if (
        not float(non_orthogonal).is_integer()
        or int(non_orthogonal) < 0
    ):
        raise ValueError(
            "n_non_orthogonal_correctors must be a non-negative integer"
        )
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
