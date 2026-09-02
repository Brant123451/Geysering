"""Run one conservative finite-node + gross two-channel Case-A snapshot step.

This is a local executable integration smoke test, not a trajectory solver.
It reconstructs one explicit finite T control volume and its three adjacent
resolved branch traces from an existing raw Case-A state, advances that node
with the isolated SSP--RK2 operator, and passes its unchanged vertical
``q_net`` to the two-channel mouth closure.  It demonstrates that the local
node and gross counter-current exchange close their inventories without any
target-dependent multiplier.

The script deliberately does not write the returned fluxes back into the
network arrays.  Doing that safely requires the atomic three-face network
commit audited by ``caseA_finite_node_qnet_owner_preflight.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
MODEL_DIR = CASE_DIR / "model"
OUTPUT_DIR = CASE_DIR / "outputs"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    solve_compressible_node_pressure,
)
from casea_compressible_node_postlaunch_stage import (  # noqa: E402
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchParameters,
)
from casea_finite_node_qnet_owner import (  # noqa: E402
    advance_finite_node_qnet_owner,
    required_commit_keys,
)
from casea_horizontal_liquid_operator import (  # noqa: E402
    HorizontalLiquidParameters,
)
from casea_material_front_cutcell import StratifiedState  # noqa: E402
from casea_post_t_physical_closure import (  # noqa: E402
    CaseAPostTClosureParameters,
    CaseAPostTPhysicalClosure,
)
from casea_tjunction_shock_network import LiquidCharacteristic  # noqa: E402
from casea_vertical_mouth_twochannel import (  # noqa: E402
    DirectionalMouthLosses,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (  # noqa: E402
    stage_from_finite_node_ssprk2,
)


HORIZONTAL_LENGTH = 4.006
HORIZONTAL_DIAMETER = 0.094
RISER_HEIGHT = 0.610
RISER_DIAMETER = 0.0571
TOWER_CENTRE_X = 3.516
RHO_L = 998.0
MU_L = 1.003e-3
R_GAS = 287.05
T_GAS = 293.0
P_ATM = 101_325.0
C_GAS = math.sqrt(R_GAS * T_GAS)
LIQUID_WAVE_SPEED = 28.0


def _full_area(diameter: float) -> float:
    return 0.25 * math.pi * diameter**2


def _branch(
    *,
    area: np.ndarray,
    discharge: np.ndarray,
    gas_mass: np.ndarray,
    gas_momentum: np.ndarray,
    cell_width: float,
    full_area: float,
    pressure,
    index: int,
    outward_sign: float,
    loss_coefficient: float,
) -> CompressibleNodeResolvedBranch:
    liquid_area = float(area[index])
    if not 0.0 < liquid_area < full_area:
        raise ValueError(
            f"branch cell {index} is not a strictly stratified trace: "
            f"A_l/A_f={liquid_area / full_area:.9g}"
        )
    mass_per_length = float(gas_mass[index] / cell_width)
    momentum_per_length = float(gas_momentum[index] / cell_width)
    if mass_per_length <= 0.0:
        raise ValueError(f"branch cell {index} has no resolved gas mass")
    face_pressure = float(pressure.face_pressure_abs[index])
    potential_pressure = float(
        pressure.potential_pressure_abs[index]
        if pressure.potential_pressure_abs is not None
        else pressure.face_pressure_abs[index]
    )
    return CompressibleNodeResolvedBranch(
        resolved=StratifiedState(
            gas_mass=mass_per_length,
            gas_momentum=outward_sign * momentum_per_length,
            liquid_area=liquid_area,
            liquid_discharge=outward_sign * float(discharge[index]),
        ),
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=face_pressure,
            reference_outward_velocity=(
                outward_sign * float(discharge[index]) / liquid_area
            ),
            wave_speed=float(pressure.pressure.celerity[index]),
            loss_coefficient=loss_coefficient,
            pressure_offset=float(pressure.node_pressure_offset[index]),
        ),
        liquid_face_area=liquid_area,
        full_area=full_area,
        reference_liquid_face_pressure_abs=potential_pressure,
        reference_liquid_pressure_potential=float(
            pressure.pressure.potential[index]
        ),
    )


def run_snapshot(path: Path, *, requested_time: float, dt: float) -> dict[str, object]:
    with np.load(path) as saved:
        time = np.asarray(saved["time"], dtype=float)
        frame = int(np.argmin(np.abs(time - requested_time)))
        horizontal_alpha = np.asarray(
            saved["horizontal_alpha_l_raw"], dtype=float
        )[frame]
        horizontal_q = np.asarray(
            saved["horizontal_liquid_discharge"], dtype=float
        )[frame]
        horizontal_m = np.asarray(
            saved["horizontal_gas_mass"], dtype=float
        )[frame]
        horizontal_j = np.asarray(
            saved["horizontal_gas_momentum"], dtype=float
        )[frame]
        vertical_alpha = np.asarray(saved["alpha_l"], dtype=float)[frame]
        vertical_u = np.asarray(
            saved["vertical_liquid_velocity"], dtype=float
        )[frame]
        vertical_m = np.asarray(
            saved["vertical_gas_mass"], dtype=float
        )[frame]
        vertical_j = np.asarray(
            saved["vertical_gas_momentum"], dtype=float
        )[frame]

    ah_full = _full_area(HORIZONTAL_DIAMETER)
    av_full = _full_area(RISER_DIAMETER)
    dx = HORIZONTAL_LENGTH / horizontal_alpha.size
    dz = RISER_HEIGHT / vertical_alpha.size
    junction = int(round(TOWER_CENTRE_X / dx - 0.5))
    if not 1 <= junction < horizontal_alpha.size - 1:
        raise ValueError("snapshot grid has no two-sided T cell")
    ah = horizontal_alpha * ah_full
    av = vertical_alpha * av_full
    qv = vertical_u * av
    inactive_vertical = np.flatnonzero(av <= 0.0)
    vertical_active_count = (
        int(inactive_vertical[0]) if inactive_vertical.size else int(av.size)
    )
    if vertical_active_count < 1:
        raise ValueError("snapshot has no active vertical liquid prefix")

    horizontal_parameters = HorizontalLiquidParameters(
        area_full=ah_full,
        diameter=HORIZONTAL_DIAMETER,
        wave_speed=LIQUID_WAVE_SPEED,
        cell_width=dx,
        rho_liquid=RHO_L,
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure=P_ATM,
    )
    vertical_parameters = HorizontalLiquidParameters(
        area_full=av_full,
        diameter=RISER_DIAMETER,
        wave_speed=LIQUID_WAVE_SPEED,
        cell_width=dz,
        rho_liquid=RHO_L,
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure=P_ATM,
    )
    closure = CaseAPostTPhysicalClosure(
        CaseAPostTClosureParameters(
            horizontal=horizontal_parameters,
            vertical=vertical_parameters,
            vertical_cell_width=dz,
        )
    )
    ph = closure("horizontal", ah, horizontal_q, horizontal_m, horizontal_j)
    # The dry atmospheric suffix is not part of the liquid pressure operator.
    # Restrict the closure to the contiguous active prefix, as the post-T
    # network stage does; this also makes its hydrostatic height the resolved
    # instantaneous liquid-column height rather than the fixed riser rim.
    pv = closure(
        "vertical",
        av[:vertical_active_count],
        qv[:vertical_active_count],
        vertical_m[:vertical_active_count],
        vertical_j[:vertical_active_count],
    )

    node_parameters = CompressibleFiniteNodeParameters(
        gas_sound_speed=C_GAS,
        liquid_density=RHO_L,
        liquid_wave_speed=LIQUID_WAVE_SPEED,
        reference_pressure_abs=P_ATM,
    )
    node = CompressibleFiniteNodeState(
        gas_mass=float(horizontal_m[junction]),
        liquid_equivalent_volume=float(ah[junction] * dx),
        node_total_volume=float(ah_full * dx),
    )
    node_pressure = solve_compressible_node_pressure(node, node_parameters)
    postlaunch = CompressiblePostLaunchParameters(
        node=node_parameters,
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure_abs=P_ATM,
    )
    west = _branch(
        area=ah,
        discharge=horizontal_q,
        gas_mass=horizontal_m,
        gas_momentum=horizontal_j,
        cell_width=dx,
        full_area=ah_full,
        pressure=ph,
        index=junction - 1,
        outward_sign=-1.0,
        loss_coefficient=0.0,
    )
    east = _branch(
        area=ah,
        discharge=horizontal_q,
        gas_mass=horizontal_m,
        gas_momentum=horizontal_j,
        cell_width=dx,
        full_area=ah_full,
        pressure=ph,
        index=junction + 1,
        outward_sign=1.0,
        loss_coefficient=0.0,
    )
    vertical = _branch(
        area=av,
        discharge=qv,
        gas_mass=vertical_m,
        gas_momentum=vertical_j,
        cell_width=dz,
        full_area=av_full,
        pressure=pv,
        index=0,
        outward_sign=1.0,
        loss_coefficient=0.75,
    )
    transaction = advance_finite_node_qnet_owner(
        node,
        dt=dt,
        west=west,
        east=east,
        vertical=vertical,
        params=postlaunch,
    )

    gas_area = av_full - av[0]
    gas_density = float(vertical_m[0] / (gas_area * dz))
    gas_velocity = float(vertical_j[0] / max(vertical_m[0], 1.0e-300))
    plan = stage_from_finite_node_ssprk2(
        transaction.result,
        phase=VerticalMouthPhaseState(
            liquid_area=float(av[0]),
            liquid_velocity=float(vertical_u[0]),
            gas_area=float(gas_area),
            gas_velocity=gas_velocity,
        ),
        geometry=VerticalMouthGeometry(diameter=RISER_DIAMETER),
        material=VerticalMouthMaterialProperties(
            liquid_density=RHO_L,
            gas_density=gas_density,
            liquid_dynamic_viscosity=MU_L,
        ),
        wallis=WallisCounterCurrentParameters(constant=0.50),
        riser_liquid_donor_volume=float(np.sum(av) * dz),
        losses=DirectionalMouthLosses(
            upward_turn=0.75,
            downward_turn=0.75,
            countercurrent_mixing=8.0,
        ),
        horizontal_axial_velocity_diagnostic=float(
            horizontal_q[junction] / ah[junction]
        ),
    )
    exchange = plan.exchange
    return {
        "source": str(path.resolve()),
        "requested_time": requested_time,
        "snapshot_time": float(time[frame]),
        "frame_index": frame,
        "dt": dt,
        "junction_cell_index": junction,
        "node_pressure_abs": node_pressure.pressure_abs,
        "q_net_lps": 1000.0 * transaction.q_net,
        "q_up_lps": 1000.0 * exchange.upward_flow,
        "q_down_lps": 1000.0 * exchange.downward_flow,
        "q_circulation_lps": 1000.0 * exchange.circulation_flow,
        "gross_identity_residual_m3s": exchange.closure_residual,
        "finite_node_liquid_ledger_residual_m3": (
            transaction.liquid_inventory_residual
        ),
        "finite_node_gas_ledger_residual_kg": (
            transaction.gas_inventory_residual
        ),
        "combined_liquid_rate_residual_m3s": (
            plan.combined_liquid_volume_rate
        ),
        "requires_second_liquid_momentum": (
            plan.requires_second_liquid_momentum
        ),
        "required_network_commit_components": len(required_commit_keys()),
        "network_state_committed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            OUTPUT_DIR
            / "vertical_fields_natural_tjunction_taylor_ccfl_v3_dx160_cfl1_9p2s.npz"
        ),
    )
    parser.add_argument("--time", type=float, default=8.5)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_snapshot(args.input, requested_time=args.time, dt=args.dt),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
