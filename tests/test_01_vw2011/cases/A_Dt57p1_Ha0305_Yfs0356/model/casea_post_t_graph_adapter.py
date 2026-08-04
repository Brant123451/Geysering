"""Finite-volume adapter between the Case-A west arm and the side-T graph.

After the tracked gas front reaches the side tee, the upstream horizontal arm
is no longer bounded by a pressurised-liquid shock.  It is a stratified
free-surface branch connected to a three-way network node.  This module keeps
that arm on the same positivity-preserving MUSCL central-upwind equations used
before arrival, but replaces its east boundary numerical flux by the *single*
liquid flux returned by the conservative T-node solve.

No wave shape is prescribed.  A change of the node flux enters the last west
control volume once, and the finite-volume equations determine the resulting
gravity wave.  The paired mass and momentum flux is also the one used by the
node inventory, so horizontal liquid volume cannot be created or deleted at
the topology hand-off.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from casea_tjunction_shock_network import LiquidCharacteristic
from tosan2021_horizontal_shockfit import (
    CircularSection,
    WetDryState,
    central_upwind_wet_dry_step,
)


@dataclass(frozen=True)
class WestBranchAdvance:
    state: WetDryState
    coordinate_mass_flux: float
    coordinate_momentum_flux: float
    liquid_volume_change: float
    conservation_error: float


def west_branch_characteristic(
    state: WetDryState,
    *,
    section: CircularSection,
    gas_pressure_abs: float,
    liquid_density: float,
    loss_coefficient: float = 0.0,
) -> tuple[LiquidCharacteristic, float]:
    """Return the incoming west-arm characteristic in the graph sign frame.

    The finite-volume coordinate points east, whereas the graph's outward west
    direction points west.  Consequently ``u_out=-Q/A``.  Gas pressure acts on
    the free surface and the hydrostatic depth offset converts it to the local
    liquid pressure datum.  With zero local fitting loss, the current gas
    pressure reproduces the resolved boundary velocity exactly; a non-zero
    loss then acts through the node relation itself.
    """

    if liquid_density <= 0.0 or gas_pressure_abs <= 0.0:
        raise ValueError("physical pressure and density must be positive")
    area = float(np.asarray(state.area, dtype=float)[-1])
    discharge = float(np.asarray(state.discharge, dtype=float)[-1])
    if area <= 0.0:
        raise ValueError("the west T-adjacent liquid trace must be wet")
    depth = float(section.depth_from_area(min(area, section.full_area)))
    celerity = float(section.celerity(min(area, section.full_area)))
    hydrostatic_offset = liquid_density * section.gravity * depth
    characteristic = LiquidCharacteristic(
        reference_pressure_abs=gas_pressure_abs + hydrostatic_offset,
        reference_outward_velocity=-discharge / area,
        wave_speed=max(celerity, 1.0e-8),
        loss_coefficient=loss_coefficient,
        pressure_offset=hydrostatic_offset,
    )
    return characteristic, area


def advance_west_branch(
    state: WetDryState,
    *,
    outward_west_liquid_flow: float,
    node_liquid_area: float,
    dx: float,
    dt: float,
    section: CircularSection,
    cfl: float = 0.45,
    manning_n: float = 0.0,
    darcy_friction: float = 0.0,
) -> WestBranchAdvance:
    """Advance the west arm with the exact conservative T-node face flux.

    ``outward_west_liquid_flow`` is positive from the tee into the west arm.
    In the east-positive finite-volume coordinate the boundary mass flux is
    therefore negative.  The momentum flux uses the same node liquid area and
    the branch's conservative pressure potential; it is not a separate force
    or a tuned impulse.
    """

    flow_out = float(outward_west_liquid_flow)
    area_node = float(node_liquid_area)
    if not all(math.isfinite(value) for value in (flow_out, area_node, dx, dt)):
        raise ValueError("west-branch coupling data must be finite")
    if area_node <= 0.0 or dx <= 0.0 or dt <= 0.0:
        raise ValueError("west-branch coupling geometry must be positive")

    coordinate_mass_flux = -flow_out
    coordinate_momentum_flux = (
        coordinate_mass_flux**2 / area_node
        + float(section.pressure_flux(area_node))
    )
    initial_volume = float(np.sum(state.area) * dx)
    advanced = central_upwind_wet_dry_step(
        state,
        dx=dx,
        dt=dt,
        section=section,
        cfl=cfl,
        manning_n=manning_n,
        darcy_friction=darcy_friction,
        left_boundary="wall",
        right_face_flux=(
            coordinate_mass_flux,
            coordinate_momentum_flux,
        ),
    )
    final_volume = float(np.sum(advanced.area) * dx)
    expected_change = flow_out * dt
    actual_change = final_volume - initial_volume
    return WestBranchAdvance(
        state=advanced,
        coordinate_mass_flux=coordinate_mass_flux,
        coordinate_momentum_flux=coordinate_momentum_flux,
        liquid_volume_change=actual_change,
        conservation_error=actual_change - expected_change,
    )


__all__ = [
    "WestBranchAdvance",
    "advance_west_branch",
    "west_branch_characteristic",
]
