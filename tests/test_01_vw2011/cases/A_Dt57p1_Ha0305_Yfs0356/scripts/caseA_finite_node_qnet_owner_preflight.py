"""Fail-closed Case-A finite-node q_net integration preflight.

The entry point starts from the validated pre-arrival Case-A shock-fit event
checkpoint, performs the conservative finite-port handoff, and evaluates the
first exact west-to-node launch step.  It then audits whether the repository
contains every state and spatial operator required to continue the *same*
finite-node transaction to the requested end time.

This is intentionally not a fallback runner.  If the network cannot commit
all west/east/vertical gas and liquid face fluxes at both SSP--RK stages, the
program exits with status 2 and writes no simulated trajectory.  In
particular, it never resumes the legacy ``G1[0]``/Taylor/CCFL/side-source path
to manufacture a nominal 9.2-s result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
MODEL_DIR = CASE_DIR / "model"
OUTPUT_DIR = CASE_DIR / "outputs"
MAIN_MODEL = MODEL_DIR / "vw2011_network_twofluid.py"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    solve_compressible_node_pressure,
)
from casea_compressible_node_launch import (  # noqa: E402
    advance_compressible_node_west_launch,
)
from casea_compressible_node_ssprk2 import (  # noqa: E402
    PRODUCTION_READY as NODE_SSPRK2_PRODUCTION_READY,
)
from casea_inclined_twofluid_branch import (  # noqa: E402
    COMPLETE_RISER_MODEL_READY,
    MISSING_RISER_CLOSURES,
)
from casea_paper_material_front_rh import PaperFrontPhysics  # noqa: E402
from casea_port_event_handoff import (  # noqa: E402
    CaseAPortGeometry,
    build_casea_port_event_handoff,
)
from casea_tjunction_shock_network import (  # noqa: E402
    LiquidCharacteristic,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
)


P_ATM = 101_325.0
RHO_L = 998.0
R_GAS = 287.05
T_GAS = 293.0
C_GAS = math.sqrt(R_GAS * T_GAS)
LIQUID_WAVE_SPEED = 28.0
VERTICAL_LOSS_COEFFICIENT = 0.75


class FiniteNodeIntegrationBlocked(RuntimeError):
    """The current repository cannot make one conservative network commit."""


@dataclass(frozen=True)
class MainLoopOwnershipAudit:
    characteristic_owner_sites: int
    postbreak_ccfl_sites: int
    taylor_mass_replacement_sites: int
    distributed_side_source_sites: int
    source_file: str

    @property
    def legacy_paths_present(self) -> bool:
        return any(
            value > 0
            for value in (
                self.characteristic_owner_sites,
                self.postbreak_ccfl_sites,
                self.taylor_mass_replacement_sites,
                self.distributed_side_source_sites,
            )
        )


def audit_main_loop_ownership(source_file: Path = MAIN_MODEL) -> MainLoopOwnershipAudit:
    """Count exact legacy ownership anchors without changing the main file."""

    source = source_file.read_text(encoding="utf-8")
    anchors: Mapping[str, str] = {
        "characteristic_owner_sites": (
            "G1[0] = junction_liquid_area * u_t_liquid"
        ),
        "postbreak_ccfl_sites": (
            "junction_vertical_node_flow = _countercurrent_flooding_liquid_flow("
        ),
        "taylor_mass_replacement_sites": "G1[0] = -material_return_flow",
        "distributed_side_source_sites": (
            "Alt_new, Qlt_new = _apply_finite_width_side_t_exchange("
        ),
    }
    counts = {name: source.count(anchor) for name, anchor in anchors.items()}
    missing = sorted(name for name, count in counts.items() if count != 1)
    if missing:
        details = ", ".join(f"{name}={counts[name]}" for name in missing)
        raise FiniteNodeIntegrationBlocked(
            "main-loop ownership anchors changed; refusing a stale patch plan: "
            + details
        )
    return MainLoopOwnershipAudit(
        **counts,
        source_file=str(source_file.resolve()),
    )


def _elastic_pressure(area: float, full_area: float) -> float:
    return P_ATM + RHO_L * LIQUID_WAVE_SPEED**2 * (
        area / full_area - 1.0
    )


def _first_exact_launch(checkpoint: Path, *, dt: float) -> dict[str, object]:
    """Evaluate the sourced topology handoff and one exact launch substep."""

    with np.load(checkpoint) as saved:
        raw = {name: np.asarray(saved[name]).copy() for name in saved.files}
    geometry = CaseAPortGeometry()
    handoff = build_casea_port_event_handoff(raw, geometry=geometry)
    west = handoff.west_cells[-1]
    east = handoff.east_pressurised_cells[0]
    vertical = handoff.vertical_pressurised_cells[0]
    horizontal_full = geometry.horizontal_area
    vertical_full = geometry.vertical_area
    west_gas_area = horizontal_full - west.liquid_area
    if west_gas_area <= 0.0:
        raise FiniteNodeIntegrationBlocked(
            "event west trace has no resolved gas area"
        )
    west_pressure = west.gas_mass / west_gas_area * C_GAS**2
    east_pressure = _elastic_pressure(east.area, horizontal_full)
    vertical_pressure = P_ATM + RHO_L * geometry.gravity * geometry.riser_liquid_height
    characteristics = TeeLiquidCharacteristics(
        west=LiquidCharacteristic(
            reference_pressure_abs=west_pressure,
            reference_outward_velocity=-west.liquid_discharge / west.liquid_area,
            wave_speed=LIQUID_WAVE_SPEED,
        ),
        east=LiquidCharacteristic(
            reference_pressure_abs=east_pressure,
            reference_outward_velocity=east.discharge / east.area,
            wave_speed=LIQUID_WAVE_SPEED,
        ),
        vertical=LiquidCharacteristic(
            reference_pressure_abs=vertical_pressure,
            reference_outward_velocity=vertical.discharge / vertical.area,
            wave_speed=LIQUID_WAVE_SPEED,
            loss_coefficient=VERTICAL_LOSS_COEFFICIENT,
        ),
        west_liquid_area=west.liquid_area,
    )
    node_params = CompressibleFiniteNodeParameters(
        gas_sound_speed=C_GAS,
        liquid_density=RHO_L,
        liquid_wave_speed=LIQUID_WAVE_SPEED,
        reference_pressure_abs=P_ATM,
    )
    distance_to_first_branch = geometry.tower_centre_x - float(
        handoff.diagnostics.event_face_x
    )
    if distance_to_first_branch <= 0.0:
        raise FiniteNodeIntegrationBlocked(
            "discrete event face lies at or beyond the first branch point"
        )
    initial_pressure = solve_compressible_node_pressure(
        handoff.node, node_params
    )
    launch = advance_compressible_node_west_launch(
        handoff.node,
        dt=dt,
        node_params=node_params,
        # The liquid-full node-side trace at this event is inherited from the
        # conservative east remap; no state or pressure is prescribed here.
        west_pressurised_foot=east,
        west_stratified_foot=west,
        west_physics=PaperFrontPhysics(
            diameter=geometry.horizontal_diameter,
            liquid_wave_speed=LIQUID_WAVE_SPEED,
            liquid_density=RHO_L,
            gravity=geometry.gravity,
            reference_pressure=P_ATM,
            gas_sound_speed=C_GAS,
            cos_inclination=1.0,
        ),
        liquid_characteristics=characteristics,
        liquid_areas=ZeroStorageTBranchAreas(
            west=west.liquid_area,
            east=horizontal_full,
            vertical=vertical_full,
        ),
        distance_to_first_branch=distance_to_first_branch,
    )
    diagnostics = handoff.diagnostics
    return {
        "event_time": float(np.asarray(raw["time"]).reshape(-1)[0]),
        "event_face_x": diagnostics.event_face_x,
        "physical_port_west_x": diagnostics.physical_port_west_x,
        "physical_port_east_x": diagnostics.physical_port_east_x,
        "event_position_error": diagnostics.event_position_error,
        "node_total_volume": diagnostics.node_total_volume,
        "node_liquid_equivalent_volume": (
            handoff.node.liquid_equivalent_volume
        ),
        "node_initial_pressure_abs": initial_pressure.pressure_abs,
        "horizontal_liquid_inventory_error": (
            diagnostics.horizontal_liquid_inventory_error
        ),
        "horizontal_discharge_inventory_error": (
            diagnostics.horizontal_discharge_inventory_error
        ),
        "west_gas_mass_error": diagnostics.west_gas_mass_error,
        "west_gas_volume_error": diagnostics.west_gas_volume_error,
        "launch_dt": dt,
        "launch_front_speed": launch.front_candidate.speed,
        "launch_front_distance": launch.front_distance,
        "distance_to_first_branch": distance_to_first_branch,
        "launch_node_gas_mass": launch.state.gas_mass,
        "launch_next_pressure_abs": launch.next_pressure.pressure_abs,
        "launch_gas_mass_balance_residual": launch.gas_mass_balance_residual,
        "launch_liquid_inventory_balance_residual": (
            launch.liquid_inventory_balance_residual
        ),
        "launch_geometric_volume_residual": launch.geometric_volume_residual,
    }


def build_report(
    checkpoint: Path,
    *,
    target_time: float,
    launch_dt: float,
) -> dict[str, object]:
    if not math.isfinite(target_time) or target_time <= 0.0:
        raise ValueError("target_time must be positive and finite")
    if not math.isfinite(launch_dt) or launch_dt <= 0.0:
        raise ValueError("launch_dt must be positive and finite")
    source_audit = audit_main_loop_ownership()
    launch = _first_exact_launch(checkpoint, dt=launch_dt)
    blockers: list[dict[str, object]] = []
    if not NODE_SSPRK2_PRODUCTION_READY:
        blockers.append(
            {
                "code": "node_ssprk2_frozen_branch_traces",
                "detail": (
                    "casea_compressible_node_ssprk2 freezes adjacent branch "
                    "traces during its local RK2 step; the network predictor "
                    "must recompute all three traces and use the same averaged "
                    "face fluxes"
                ),
            }
        )
    if not COMPLETE_RISER_MODEL_READY:
        blockers.append(
            {
                "code": "vertical_twofluid_branch_incomplete",
                "detail": (
                    "the strict inclined two-fluid core has no complete "
                    "pressurised/stratified front, top free-surface/vent, "
                    "wall/interfacial shear, or vertical phase-topology owner"
                ),
                "missing": list(MISSING_RISER_CLOSURES),
            }
        )
    if source_audit.legacy_paths_present:
        blockers.append(
            {
                "code": "legacy_qnet_owners_still_present",
                "detail": (
                    "the current main loop still applies the characteristic "
                    "G1 bottom flux, post-breakthrough CCFL, Taylor-return "
                    "replacement, and distributed side source; a finite-node "
                    "entry must replace all four in one patch, not add to them"
                ),
            }
        )
    reached = float(launch["event_time"])
    report: dict[str, object] = {
        "status": "blocked_fail_closed" if blockers else "ready",
        "requested_end_time": target_time,
        "last_conservatively_evaluated_time": reached + launch_dt,
        "checkpoint": str(checkpoint.resolve()),
        "main_model_modified": False,
        "main_loop_ownership_audit": asdict(source_audit),
        "event_and_launch": launch,
        "blockers": blockers,
        "required_atomic_patch": [
            "create the explicit finite node at the exact port event and remove its geometric volume from adjacent branch cells",
            "evaluate west/east/vertical branch traces from the same RK predictor state",
            "replace both horizontal node faces and the vertical bottom face with the finite-node gas/liquid mass and momentum fluxes",
            "advance node gas mass and liquid equivalent volume with those same time-averaged face fluxes",
            "disable legacy G1 ownership, Taylor mass replacement, post-breakthrough CCFL on q_net, and the distributed side source",
            "advance the complete vertical two-fluid/free-surface/vent topology without a target trajectory or fallback fill",
        ],
    }
    if blockers:
        report["refused_action"] = (
            f"did not fabricate continuation from {reached + launch_dt:.9f} s "
            f"to {target_time:.9f} s"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "casea_port_west_event_dx40_checkpoint.npz",
    )
    parser.add_argument("--target-time", type=float, default=9.2)
    parser.add_argument("--launch-dt", type=float, default=1.0e-5)
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        help="return zero only when the strict preflight blocks as expected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.checkpoint,
        target_time=args.target_time,
        launch_dt=args.launch_dt,
    )
    print(json.dumps(report, indent=2))
    blocked = report["status"] == "blocked_fail_closed"
    if args.expect_blocked:
        if not blocked:
            raise SystemExit("preflight unexpectedly became ready")
        return
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
