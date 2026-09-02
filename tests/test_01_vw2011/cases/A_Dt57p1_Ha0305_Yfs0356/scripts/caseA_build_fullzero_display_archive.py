"""Build a continuous 0-s display archive for the fast Case-A event run.

The fast event workflow starts the coupled side-T/riser model from the exact
pre-arrival shock-fit checkpoint.  This utility reconstructs the omitted
display history with that same pre-arrival owner, keeps the as-yet unconnected
riser in its event-initial state, and then appends the coupled event archive.
It is a display archive only; it does not replace the solver field archive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = CASE_ROOT / "model"
OUTPUT_DIR = CASE_ROOT / "outputs"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import CaseASideTShockFit, case_a_config  # noqa: E402


PIPE_LENGTH = 4.006
PIPE_DIAMETER = 0.094
TOWER_CENTRE_X = 3.516
TOWER_DIAMETER = 0.0571
PORT_WEST_X = TOWER_CENTRE_X - 0.5 * TOWER_DIAMETER
ATMOSPHERIC_PRESSURE = 101325.0
LIQUID_DENSITY = 998.0
GRAVITY = 9.81


def _scalar(saved: np.lib.npyio.NpzFile, name: str) -> float:
    values = np.asarray(saved[name], dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"checkpoint field {name!r} must be one finite scalar")
    return float(values[0])


def _phase_fraction(
    *, area: np.ndarray, centres: np.ndarray, interface_x: float
) -> np.ndarray:
    full_area = math.pi * PIPE_DIAMETER**2 / 4.0
    raw_fraction = np.asarray(area, dtype=float) / full_area
    return np.where(
        centres <= float(interface_x) + 1.0e-12,
        np.clip(raw_fraction, 0.0, 1.0),
        1.0,
    )


def build_archive(
    *,
    post_variant: str,
    output_variant: str,
    checkpoint_path: Path,
    integration_dt: float,
    save_every: int,
) -> tuple[Path, Path, Path]:
    if integration_dt <= 0.0:
        raise ValueError("integration_dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be at least one")

    post_fields_path = OUTPUT_DIR / f"vertical_fields_{post_variant}.npz"
    post_index_path = OUTPUT_DIR / f"frames_index_{post_variant}_render.json"
    if not post_fields_path.is_file():
        raise FileNotFoundError(post_fields_path)
    if not post_index_path.is_file():
        raise FileNotFoundError(post_index_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with np.load(post_fields_path) as post_saved:
        post = {name: np.asarray(post_saved[name]) for name in post_saved.files}
    post_manifest = json.loads(post_index_path.read_text(encoding="utf-8"))
    post_time = np.asarray(post["time"], dtype=float)
    if post_time.ndim != 1 or post_time.size < 2:
        raise ValueError("post-event archive has an invalid time axis")
    if len(post_manifest) != post_time.size:
        raise ValueError("post-event fields and frame index differ in length")

    with np.load(checkpoint_path) as checkpoint_saved:
        checkpoint = {
            name: np.asarray(checkpoint_saved[name])
            for name in checkpoint_saved.files
        }
    event_time = _scalar(checkpoint, "time")
    if abs(float(post_time[0]) - event_time) > 1.0e-10:
        raise ValueError("post-event archive does not start at the checkpoint")

    cells = int(np.asarray(checkpoint["area"]).size)
    base = case_a_config(dx=PIPE_LENGTH / cells)
    solver = CaseASideTShockFit(replace(base, vent_x=PORT_WEST_X))
    state = solver.case_b_initial_state()
    snapshots: list[tuple[float, np.ndarray, float, float]] = [
        (
            float(state.time),
            np.asarray(state.area, dtype=float).copy(),
            float(state.interface_x),
            float(state.air_pressure_abs),
        )
    ]
    calls = 0
    tolerance = (
        128.0
        * np.finfo(float).eps
        * max(1.0, float(solver.junction_face_x))
    )
    while True:
        calls += 1
        advance = solver.step_until_junction(
            state,
            integration_dt,
            location_tolerance=tolerance,
        )
        state = advance.state
        if calls % save_every == 0 or advance.reached:
            snapshots.append(
                (
                    float(state.time),
                    np.asarray(state.area, dtype=float).copy(),
                    float(state.interface_x),
                    float(state.air_pressure_abs),
                )
            )
        if advance.reached:
            break
        if state.time > event_time + integration_dt:
            raise RuntimeError("pre-arrival replay passed the saved event")

    checkpoint_area = np.asarray(checkpoint["area"], dtype=float)
    checkpoint_discharge = np.asarray(checkpoint["discharge"], dtype=float)
    if abs(float(state.time) - event_time) > 5.0e-13:
        raise RuntimeError("pre-arrival replay did not recover the event time")
    if not np.array_equal(np.asarray(state.area), checkpoint_area):
        raise RuntimeError("pre-arrival replay did not recover checkpoint area")
    if not np.array_equal(np.asarray(state.discharge), checkpoint_discharge):
        raise RuntimeError("pre-arrival replay did not recover checkpoint discharge")

    event_phase = _phase_fraction(
        area=np.asarray(state.area),
        centres=np.asarray(solver.x),
        interface_x=float(state.interface_x),
    )
    if not np.array_equal(event_phase, post["horizontal_alpha_l"][0]):
        raise RuntimeError("display phase field is discontinuous at the event")

    prefix = [item for item in snapshots if item[0] < event_time - 1.0e-12]
    prefix_time = np.asarray([item[0] for item in prefix], dtype=float)
    prefix_horizontal = np.vstack([
        _phase_fraction(
            area=item[1],
            centres=np.asarray(solver.x),
            interface_x=item[2],
        )
        for item in prefix
    ])
    prefix_count = prefix_time.size
    full_time = np.concatenate((prefix_time, post_time))
    if np.any(np.diff(full_time) <= 0.0):
        raise RuntimeError("combined display time must be strictly increasing")

    initial_vertical_liquid = np.asarray(post["alpha_l"][0], dtype=float)
    initial_vertical_gas = np.asarray(post["alpha_g"][0], dtype=float)
    combined_vertical_liquid = np.concatenate(
        (
            np.repeat(initial_vertical_liquid[None, :], prefix_count, axis=0),
            np.asarray(post["alpha_l"], dtype=float),
        ),
        axis=0,
    )
    combined_vertical_gas = np.concatenate(
        (
            np.repeat(initial_vertical_gas[None, :], prefix_count, axis=0),
            np.asarray(post["alpha_g"], dtype=float),
        ),
        axis=0,
    )
    combined_horizontal = np.concatenate(
        (prefix_horizontal, np.asarray(post["horizontal_alpha_l"], dtype=float)),
        axis=0,
    )

    fields_path = OUTPUT_DIR / f"vertical_fields_{output_variant}.npz"
    index_path = OUTPUT_DIR / f"frames_index_{output_variant}.json"
    metadata_path = OUTPUT_DIR / f"fullzero_display_{output_variant}.json"
    np.savez_compressed(
        fields_path,
        time=full_time,
        z=np.asarray(post["z"], dtype=float),
        alpha_l=combined_vertical_liquid,
        alpha_g=combined_vertical_gas,
        horizontal_alpha_l=combined_horizontal,
    )

    event_frame = post_manifest[0]
    prefix_manifest = []
    for time_s, _area, _interface_x, pressure_abs in prefix:
        prefix_manifest.append({
            "time": float(time_s),
            "wtop": float(event_frame["wtop"]),
            "materialHeight": float(event_frame["materialHeight"]),
            "itop": 0.0,
            "coreMassMg": 0.0,
            "head": float(
                (pressure_abs - ATMOSPHERIC_PRESSURE)
                / (LIQUID_DENSITY * GRAVITY)
            ),
            "eastMaterialFront": TOWER_CENTRE_X,
        })
    combined_manifest = prefix_manifest + post_manifest
    if len(combined_manifest) != full_time.size:
        raise RuntimeError("combined frame index differs from field archive")
    index_path.write_text(
        json.dumps(combined_manifest, indent=2), encoding="utf-8"
    )

    metadata = {
        "role": "display-only 0-s archive",
        "post_event_fields": str(post_fields_path.resolve()),
        "exact_event_checkpoint": str(checkpoint_path.resolve()),
        "pre_event_owner": "CaseASideTShockFit",
        "pre_event_riser_state": "unchanged event-initial state",
        "integration_dt_s": integration_dt,
        "saved_every_n_steps": save_every,
        "pre_event_frames": int(prefix_count),
        "post_event_frames": int(post_time.size),
        "total_frames": int(full_time.size),
        "event_time_s": event_time,
        "event_area_byte_exact": True,
        "event_discharge_byte_exact": True,
        "event_display_phase_byte_exact": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return fields_path, index_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--post-variant", default="recoupled_stable_fast_v106"
    )
    parser.add_argument(
        "--output-variant", default="recoupled_stable_fast_v106_fullzero_display"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "casea_port_west_event_dx40_checkpoint.npz",
    )
    parser.add_argument("--integration-dt", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=2)
    arguments = parser.parse_args()
    for path in build_archive(
        post_variant=arguments.post_variant,
        output_variant=arguments.output_variant,
        checkpoint_path=arguments.checkpoint,
        integration_dt=arguments.integration_dt,
        save_every=arguments.save_every,
    ):
        print(path)


if __name__ == "__main__":
    main()
