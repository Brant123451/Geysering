"""Audit the strict first west-port launch from the exact Case-A checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
MODEL = CASE / "model"
OUTPUTS = CASE / "outputs"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
)
from casea_compressible_node_launch import (  # noqa: E402
    advance_compressible_node_west_launch,
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
R_GAS = 287.05
T_GAS = 293.0


def run(checkpoint: Path, output: Path, dt: float) -> Path:
    geometry = CaseAPortGeometry()
    with np.load(checkpoint) as saved:
        raw = {name: saved[name] for name in saved.files}
    handoff = build_casea_port_event_handoff(raw, geometry=geometry)
    west = handoff.west_cells[-1]
    east = handoff.east_pressurised_cells[0]
    vertical = handoff.vertical_pressurised_cells[0]
    gas_sound_speed = math.sqrt(R_GAS * T_GAS)
    node_params = CompressibleFiniteNodeParameters(
        gas_sound_speed=gas_sound_speed,
        liquid_density=geometry.liquid_density,
        liquid_wave_speed=geometry.horizontal_wave_speed,
    )

    def elastic_pressure(area: float, full_area: float, speed: float) -> float:
        return P_ATM + geometry.liquid_density * speed**2 * (
            area / full_area - 1.0
        )

    west_gas_area = geometry.horizontal_area - west.liquid_area
    west_pressure = west.gas_mass / west_gas_area * gas_sound_speed**2
    liquid_characteristics = TeeLiquidCharacteristics(
        west=LiquidCharacteristic(
            reference_pressure_abs=west_pressure,
            reference_outward_velocity=(-west.liquid_discharge / west.liquid_area),
            wave_speed=geometry.horizontal_wave_speed,
        ),
        east=LiquidCharacteristic(
            reference_pressure_abs=elastic_pressure(
                east.area,
                geometry.horizontal_area,
                geometry.horizontal_wave_speed,
            ),
            reference_outward_velocity=east.discharge / east.area,
            wave_speed=geometry.horizontal_wave_speed,
        ),
        vertical=LiquidCharacteristic(
            reference_pressure_abs=(
                P_ATM
                + geometry.liquid_density
                * geometry.gravity
                * geometry.riser_liquid_height
            ),
            reference_outward_velocity=vertical.discharge / vertical.area,
            wave_speed=geometry.vertical_wave_speed,
            loss_coefficient=0.75,
        ),
        west_liquid_area=west.liquid_area,
    )
    result = advance_compressible_node_west_launch(
        handoff.node,
        dt=dt,
        node_params=node_params,
        west_pressurised_foot=east,
        west_stratified_foot=west,
        west_physics=PaperFrontPhysics(
            diameter=geometry.horizontal_diameter,
            liquid_wave_speed=geometry.horizontal_wave_speed,
            liquid_density=geometry.liquid_density,
            gravity=geometry.gravity,
            reference_pressure=P_ATM,
            gas_sound_speed=gas_sound_speed,
            cos_inclination=1.0,
        ),
        liquid_characteristics=liquid_characteristics,
        liquid_areas=ZeroStorageTBranchAreas(
            west=west.liquid_area,
            east=geometry.horizontal_area,
            vertical=geometry.vertical_area,
        ),
        distance_to_first_branch=(
            geometry.tower_centre_x - handoff.diagnostics.event_face_x
        ),
    )
    payload = {
        "kind": "strict_casea_finite_tnode_west_launch",
        "checkpoint": str(checkpoint.resolve()),
        "dt_s": dt,
        "current_node_pressure_pa": result.current_pressure.pressure_abs,
        "next_node_pressure_pa": result.next_pressure.pressure_abs,
        "selected_active_set": result.front_candidate.active_set,
        "front_speed_m_per_s": result.front_candidate.speed,
        "front_distance_m": result.front_distance,
        "distance_to_first_branch_m": (
            geometry.tower_centre_x - handoff.diagnostics.event_face_x
        ),
        "west_gas_mass_rate_into_node_kg_per_s": (
            result.west_gas_mass_rate_into_node
        ),
        "east_gas_mass_rate_kg_per_s": 0.0,
        "vertical_gas_mass_rate_kg_per_s": 0.0,
        "west_liquid_volume_rate_outward_m3_per_s": (
            result.west_liquid_volume_rate_outward
        ),
        "east_liquid_volume_rate_outward_m3_per_s": (
            result.east_liquid_volume_rate_outward
        ),
        "vertical_liquid_volume_rate_outward_m3_per_s": (
            result.vertical_liquid_volume_rate_outward
        ),
        "gas_mass_balance_residual_kg": result.gas_mass_balance_residual,
        "liquid_inventory_balance_residual_m3": (
            result.liquid_inventory_balance_residual
        ),
        "node_occupancy_residual_m3": result.next_pressure.occupancy_residual,
        "swept_gas_volume_m3": result.swept_gas_volume,
        "node_gas_volume_m3": result.node_gas_volume,
        "geometric_volume_residual_m3": result.geometric_volume_residual,
        "receiver_topology": (
            "gas enters finite node from west only; east and vertical remain closed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUTS / "casea_port_west_event_dx40_checkpoint.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "casea_strict_tnode_west_launch_dx40.json",
    )
    parser.add_argument("--dt", type=float, default=1.0e-5)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run(arguments.checkpoint, arguments.output, arguments.dt))
