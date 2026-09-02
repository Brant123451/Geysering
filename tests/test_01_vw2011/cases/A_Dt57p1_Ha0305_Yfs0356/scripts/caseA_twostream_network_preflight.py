"""Preflight the real Case-A finite-node/two-stream integration to 9.2 s.

The command evaluates the exact west-port event and first conservative launch,
then audits the code contracts needed for a global continuation.  It is
fail-closed: while the finite-node face traces are not recomputed at both
global SSP-RK stages and the legacy mouth owners remain in the main loop, no
trajectory, frame viewer, or nominal 9.2-s result is written.

This script is an integration screen, not a simulator and not a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
MODEL_DIR = CASE_DIR / "model"
OUTPUT_DIR = CASE_DIR / "outputs"
MAIN_MODEL = MODEL_DIR / "vw2011_network_twofluid.py"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from caseA_finite_node_qnet_owner_preflight import (  # noqa: E402
    _first_exact_launch,
    audit_main_loop_ownership,
)
from casea_compressible_node_ssprk2 import (  # noqa: E402
    PRODUCTION_READY as FINITE_NODE_SSPRK2_READY,
)
from casea_twostream_network_adapter import (  # noqa: E402
    COMPLETE_CASEA_NETWORK_READY,
    GLOBAL_INTEGRATION_BLOCKERS,
    TWOSTREAM_NETWORK_ADAPTER_READY,
)
from casea_vertical_twostream_fv import (  # noqa: E402
    COMPLETE_CASEA_RISER_READY,
    MISSING_PHYSICAL_CLOSURES,
    TWOSTREAM_FV_CORE_READY,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    checkpoint: Path,
    *,
    requested_end_time: float,
    launch_dt: float,
) -> dict[str, object]:
    if not math.isfinite(requested_end_time) or requested_end_time <= 0.0:
        raise ValueError("requested end time must be positive and finite")
    if not math.isfinite(launch_dt) or launch_dt <= 0.0:
        raise ValueError("launch dt must be positive and finite")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    launch = _first_exact_launch(checkpoint, dt=launch_dt)
    ownership = audit_main_loop_ownership(MAIN_MODEL)
    blockers: list[dict[str, object]] = []
    if not FINITE_NODE_SSPRK2_READY:
        blockers.append(
            {
                "code": "finite_node_traces_frozen_inside_local_ssprk2",
                "required_change": (
                    "move the node Euler residual into the global SSP-RK2 "
                    "stage and rebuild west/east/vertical traces from each "
                    "predictor state"
                ),
            }
        )
    if not COMPLETE_CASEA_RISER_READY:
        blockers.append(
            {
                "code": "casea_riser_not_yet_a_complete_network_branch",
                "missing_physical_closures": list(MISSING_PHYSICAL_CLOSURES),
            }
        )
    if ownership.legacy_paths_present:
        blockers.append(
            {
                "code": "legacy_mouth_flux_owners_are_still_live",
                "required_change": (
                    "replace G1 bottom ownership, Taylor return mass "
                    "replacement, post-breakthrough CCFL-on-qnet, and the "
                    "distributed side source in one atomic main-loop patch"
                ),
            }
        )
    if not COMPLETE_CASEA_NETWORK_READY:
        blockers.append(
            {
                "code": "global_two_stream_network_commit_not_installed",
                "required_contracts": list(GLOBAL_INTEGRATION_BLOCKERS),
            }
        )

    last_time = float(launch["event_time"]) + float(launch_dt)
    return {
        "status": "blocked_fail_closed" if blockers else "ready_for_9p2s_run",
        "requested_end_time_s": float(requested_end_time),
        "last_conservatively_evaluated_time_s": last_time,
        "trajectory_generated": False,
        "html_generated": False,
        "checkpoint": str(checkpoint.resolve()),
        "source": {
            "main_model": str(MAIN_MODEL.resolve()),
            "main_model_sha256": _sha256(MAIN_MODEL),
        },
        "component_readiness": {
            "finite_node_local_ssprk2": bool(FINITE_NODE_SSPRK2_READY),
            "vertical_twostream_fv_core": bool(TWOSTREAM_FV_CORE_READY),
            "vertical_twostream_complete_casea_branch": bool(
                COMPLETE_CASEA_RISER_READY
            ),
            "twostream_network_adapter": bool(TWOSTREAM_NETWORK_ADAPTER_READY),
            "complete_casea_network": bool(COMPLETE_CASEA_NETWORK_READY),
        },
        "main_loop_ownership_audit": {
            "characteristic_owner_sites": ownership.characteristic_owner_sites,
            "postbreak_ccfl_sites": ownership.postbreak_ccfl_sites,
            "taylor_mass_replacement_sites": (
                ownership.taylor_mass_replacement_sites
            ),
            "distributed_side_source_sites": (
                ownership.distributed_side_source_sites
            ),
        },
        "exact_event_and_first_launch": launch,
        "blockers": blockers,
        "required_atomic_main_patch": [
            "persist A_up,Q_up,A_down,Q_down after the Taylor breakthrough topology event",
            "construct the two-stream state with the conservative Taylor event map, not from a requested hold-up",
            "evaluate finite-node and all three adjacent branch residuals at both global SSP-RK2 stages",
            "decompose only the node-owned vertical q_net into gross Q_up and Q_down",
            "use those gross rates as the two vertical bottom boundary fluxes and retain their separate momenta",
            "apply the equal-and-opposite physical gas-drag impulse to the resolved gas momentum",
            "commit every west/east/vertical gas and liquid face component once",
            "remove all four legacy mouth mass owners in the same patch",
            "write raw conservation, gross-flow, bottom-inventory, and gas-Mach diagnostics before rendering",
        ],
        "refused_action": (
            None
            if not blockers
            else (
                f"did not fabricate continuation from {last_time:.9f} s "
                f"to {requested_end_time:.9f} s"
            )
        ),
    }


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
        "--output",
        type=Path,
        default=OUTPUT_DIR / "casea_twostream_network_preflight_9p2s.json",
    )
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        help="return zero only when the honest preflight blocks as expected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.checkpoint,
        requested_end_time=args.target_time,
        launch_dt=args.launch_dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    blocked = report["status"] == "blocked_fail_closed"
    if args.expect_blocked:
        if not blocked:
            raise SystemExit("preflight unexpectedly became ready")
        return
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
