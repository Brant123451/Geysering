"""Conservative handoff at the Case-A west edge of the finite T opening.

The pre-arrival shock-fit state owns the complete horizontal pipe.  At the
first topology event this module partitions that *same* state into

* a resolved west stratified branch,
* a finite geometric T control volume spanning the measured opening, and
* an east pressurised branch beginning at the east edge of that opening.

The partition is an overlap integral of the old piecewise-constant finite-
volume field.  The finite node is initially liquid full, but its conserved
liquid inventory is *not* replaced by its geometric volume.  The exact old
``integral(A dx)`` is retained as the node's elastic mass-equivalent liquid
volume.  This prevents the small ``A>A_f`` storage from being injected into
the first east cell as a spurious local pressure pulse.  The node geometry is
the measured horizontal footprint between the event face and the east edge of
the circular side-port opening; it is not fitted to a result.

The zero-dimensional node has no horizontal liquid-momentum degree of
freedom.  The small old axial-discharge inventory of that footprint is
therefore projected once onto the first east pressurised cell.  This projection
is stated explicitly in the diagnostics; unlike the former elastic-area
transfer, it changes no pressure inventory and closes the horizontal
discharge ledger exactly.

The old lumped gas state has no momentum degree of freedom.  Resolving it
requires one explicit kinematic reconstruction.  We use the affine
closed-wall/material-front field ``u_g(x)=w*x/L``: it satisfies ``u_g(0)=0``
and ``u_g(L)=w`` and is the constant-strain continuation of a uniformly
compressed one-dimensional pocket.  This assumption is reported in the
diagnostics and is never adjusted using a 2-D result or target waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from casea_compressible_finite_node import CompressibleFiniteNodeState
from casea_material_front_cutcell import PressurisedState, StratifiedState
from tosan2021_horizontal_shockfit import CircularSection


@dataclass(frozen=True)
class CaseAPortGeometry:
    horizontal_length: float = 4.006
    horizontal_diameter: float = 0.094
    tower_centre_x: float = 3.516
    tower_diameter: float = 0.0571
    riser_liquid_height: float = 0.356
    riser_total_height: float = 0.610
    horizontal_wave_speed: float = 28.0
    vertical_wave_speed: float = 28.0
    gravity: float = 9.81
    liquid_density: float = 998.0

    def __post_init__(self) -> None:
        values = (
            self.horizontal_length,
            self.horizontal_diameter,
            self.tower_centre_x,
            self.tower_diameter,
            self.riser_liquid_height,
            self.riser_total_height,
            self.horizontal_wave_speed,
            self.vertical_wave_speed,
            self.gravity,
            self.liquid_density,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Case-A port geometry must be positive and finite")
        if self.port_west_x <= 0.0 or self.port_east_x >= self.horizontal_length:
            raise ValueError("the measured T opening lies outside the pipe")
        if self.riser_liquid_height >= self.riser_total_height:
            raise ValueError("initial riser liquid height must lie below its top")

    @property
    def port_west_x(self) -> float:
        return self.tower_centre_x - 0.5 * self.tower_diameter

    @property
    def port_east_x(self) -> float:
        return self.tower_centre_x + 0.5 * self.tower_diameter

    @property
    def horizontal_area(self) -> float:
        return 0.25 * math.pi * self.horizontal_diameter**2

    @property
    def vertical_area(self) -> float:
        return 0.25 * math.pi * self.tower_diameter**2


@dataclass(frozen=True)
class CaseAPortHandoffDiagnostics:
    event_face_x: float
    physical_port_west_x: float
    physical_port_east_x: float
    event_position_error: float
    node_length: float
    node_total_volume: float
    old_node_liquid_inventory: float
    node_geometric_liquid_volume: float
    node_liquid_equivalent_volume: float
    node_elastic_inventory_excess: float
    east_elastic_inventory_correction: float
    old_node_discharge_inventory: float
    west_gas_mass_error: float
    west_gas_volume_error: float
    horizontal_liquid_inventory_error: float
    horizontal_discharge_inventory_error: float
    gas_velocity_reconstruction: str
    reconstructed_gas_kinetic_energy: float


@dataclass(frozen=True)
class CaseAPortEventHandoff:
    west_faces: tuple[float, ...]
    west_cells: tuple[StratifiedState, ...]
    node: CompressibleFiniteNodeState
    east_faces: tuple[float, ...]
    east_pressurised_cells: tuple[PressurisedState, ...]
    vertical_faces: tuple[float, ...]
    vertical_pressurised_cells: tuple[PressurisedState, ...]
    vertical_open_surface_height: float
    diagnostics: CaseAPortHandoffDiagnostics


def _scalar(data: Mapping[str, object], name: str) -> float:
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"checkpoint {name!r} must contain one scalar")
    result = float(value.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"checkpoint {name!r} is not finite")
    return result


def _integrate_piecewise_constant(
    old_faces: np.ndarray,
    old_values: np.ndarray,
    left: float,
    right: float,
) -> float:
    if not (math.isfinite(left) and math.isfinite(right) and right > left):
        raise ValueError("integration interval must be finite and non-empty")
    lo = np.maximum(old_faces[:-1], left)
    hi = np.minimum(old_faces[1:], right)
    overlap = np.maximum(hi - lo, 0.0)
    return float(np.sum(overlap * old_values, dtype=np.float64))


def _remap_piecewise_constant(
    old_faces: np.ndarray,
    old_values: np.ndarray,
    new_faces: np.ndarray,
) -> np.ndarray:
    result = np.empty(new_faces.size - 1, dtype=float)
    for index, (left, right) in enumerate(
        zip(new_faces[:-1], new_faces[1:], strict=True)
    ):
        result[index] = (
            _integrate_piecewise_constant(old_faces, old_values, left, right)
            / (right - left)
        )
    return result


def _vertical_faces(height: float, target_width: float) -> np.ndarray:
    if target_width <= 0.0:
        raise ValueError("vertical target width must be positive")
    full_count = int(math.floor(height / target_width))
    faces = [index * target_width for index in range(full_count + 1)]
    if not math.isclose(faces[-1], height, rel_tol=0.0, abs_tol=1.0e-14):
        faces.append(height)
    return np.asarray(faces, dtype=float)


def build_casea_port_event_handoff(
    checkpoint: Mapping[str, object],
    *,
    geometry: CaseAPortGeometry = CaseAPortGeometry(),
    vertical_target_width: float = 0.020,
) -> CaseAPortEventHandoff:
    """Return the conservative finite-node topology state at first contact."""

    area = np.asarray(checkpoint["area"], dtype=float)
    discharge = np.asarray(checkpoint["discharge"], dtype=float)
    if area.ndim != 1 or area.size == 0 or discharge.shape != area.shape:
        raise ValueError("checkpoint area/discharge must be equal non-empty vectors")
    if not (np.all(np.isfinite(area)) and np.all(np.isfinite(discharge))):
        raise ValueError("checkpoint horizontal fields must be finite")
    if np.any(area <= 0.0):
        raise ValueError("checkpoint horizontal liquid area must be positive")

    dx = _scalar(checkpoint, "dx")
    event_face_index = int(round(_scalar(checkpoint, "junction_face_index")))
    event_face_x = _scalar(checkpoint, "junction_face_x")
    interface_x = _scalar(checkpoint, "interface_x")
    if not 0 < event_face_index < area.size:
        raise ValueError("junction face must split the horizontal field")
    old_faces = np.arange(area.size + 1, dtype=float) * dx
    if not math.isclose(old_faces[-1], geometry.horizontal_length, abs_tol=1.0e-12):
        raise ValueError("checkpoint mesh length differs from Case-A geometry")
    if not math.isclose(
        event_face_x, old_faces[event_face_index], rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("checkpoint junction face is inconsistent with its mesh")
    if not math.isclose(interface_x, event_face_x, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("checkpoint is not located at the exact topology event")
    if event_face_x >= geometry.port_east_x:
        raise ValueError("discrete event face lies beyond the measured T opening")

    full_area = geometry.horizontal_area
    west_area = area[:event_face_index]
    west_discharge = discharge[:event_face_index]
    west_gas_area = full_area - west_area
    if np.any(west_gas_area <= 0.0):
        raise ValueError("every west event cell must contain resolved gas volume")

    gas_mass = _scalar(checkpoint, "gas_mass")
    gas_volume = _scalar(checkpoint, "gas_volume")
    interface_speed = _scalar(checkpoint, "interface_speed")
    if gas_mass <= 0.0 or gas_volume <= 0.0:
        raise ValueError("pre-event gas mass and volume must be positive")
    reconstructed_volume = float(np.sum(west_gas_area) * dx)
    if not math.isclose(reconstructed_volume, gas_volume, rel_tol=1.0e-10, abs_tol=1.0e-14):
        raise ValueError("checkpoint gas volume is inconsistent with the west field")
    gas_density = gas_mass / gas_volume
    west_centres = 0.5 * (old_faces[:event_face_index] + old_faces[1:event_face_index + 1])
    gas_velocity = interface_speed * west_centres / event_face_x
    west_cells = tuple(
        StratifiedState(
            gas_mass=gas_density * gas_cross_section,
            gas_momentum=gas_density * gas_cross_section * velocity,
            liquid_area=liquid_cross_section,
            liquid_discharge=flow,
        )
        for gas_cross_section, velocity, liquid_cross_section, flow in zip(
            west_gas_area,
            gas_velocity,
            west_area,
            west_discharge,
            strict=True,
        )
    )
    reconstructed_mass = float(
        math.fsum(cell.gas_mass * dx for cell in west_cells)
    )
    reconstructed_kinetic = float(
        math.fsum(
            0.5 * cell.gas_mass * dx * cell.gas_velocity**2
            for cell in west_cells
        )
    )

    node_left = event_face_x
    node_right = geometry.port_east_x
    node_length = node_right - node_left
    node_volume = full_area * node_length
    old_node_area_integral = _integrate_piecewise_constant(
        old_faces, area, node_left, node_right
    )
    old_node_q_integral = _integrate_piecewise_constant(
        old_faces, discharge, node_left, node_right
    )
    node = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=old_node_area_integral,
        node_total_volume=node_volume,
    )

    retained_old_faces = old_faces[old_faces > node_right]
    east_faces = np.r_[node_right, retained_old_faces]
    if east_faces.size < 2 or not math.isclose(
        east_faces[-1], geometry.horizontal_length, abs_tol=1.0e-12
    ):
        raise ValueError("east branch remap has no valid downstream reach")
    east_area = _remap_piecewise_constant(old_faces, area, east_faces)
    east_q = _remap_piecewise_constant(old_faces, discharge, east_faces)
    first_width = east_faces[1] - east_faces[0]
    # The compressible finite node, not the first east cell, owns the old
    # elastic liquid inventory.  Keeping this value explicitly zero is useful
    # in machine-readable diagnostics and guards against reintroducing the old
    # pressure-pulse remap.
    elastic_correction = 0.0
    east_q[0] += old_node_q_integral / first_width
    if np.any(east_area <= 0.0):
        raise ValueError("conservative east remap produced non-positive area")
    east_cells = tuple(
        PressurisedState(a, q)
        for a, q in zip(east_area, east_q, strict=True)
    )

    vertical_faces = _vertical_faces(
        geometry.riser_liquid_height, vertical_target_width
    )
    vertical_centres = 0.5 * (vertical_faces[:-1] + vertical_faces[1:])
    vertical_pressure_head = geometry.riser_liquid_height - vertical_centres
    vertical_full_area = geometry.vertical_area
    # Standard water-hammer storage with A=Af at zero gauge pressure.  The
    # vertical RH adapter must retain this reference; it must not reinterpret
    # A<Af as a horizontal circular free surface.
    vertical_area = vertical_full_area * (
        1.0
        + geometry.gravity
        * vertical_pressure_head
        / geometry.vertical_wave_speed**2
    )
    vertical_cells = tuple(
        PressurisedState(a, 0.0) for a in vertical_area
    )

    old_horizontal_liquid = float(np.sum(area) * dx)
    new_horizontal_liquid = float(
        math.fsum(cell.liquid_area * dx for cell in west_cells)
        + node.liquid_equivalent_volume
        + math.fsum(
            cell.area * (east_faces[index + 1] - east_faces[index])
            for index, cell in enumerate(east_cells)
        )
    )
    old_horizontal_discharge = float(np.sum(discharge) * dx)
    new_horizontal_discharge = float(
        math.fsum(cell.liquid_discharge * dx for cell in west_cells)
        + math.fsum(
            cell.discharge * (east_faces[index + 1] - east_faces[index])
            for index, cell in enumerate(east_cells)
        )
    )
    diagnostics = CaseAPortHandoffDiagnostics(
        event_face_x=event_face_x,
        physical_port_west_x=geometry.port_west_x,
        physical_port_east_x=geometry.port_east_x,
        event_position_error=event_face_x - geometry.port_west_x,
        node_length=node_length,
        node_total_volume=node_volume,
        old_node_liquid_inventory=old_node_area_integral,
        node_geometric_liquid_volume=node_volume,
        node_liquid_equivalent_volume=node.liquid_equivalent_volume,
        node_elastic_inventory_excess=(
            node.liquid_equivalent_volume - node.node_total_volume
        ),
        east_elastic_inventory_correction=elastic_correction,
        old_node_discharge_inventory=old_node_q_integral,
        west_gas_mass_error=reconstructed_mass - gas_mass,
        west_gas_volume_error=reconstructed_volume - gas_volume,
        horizontal_liquid_inventory_error=(
            new_horizontal_liquid - old_horizontal_liquid
        ),
        horizontal_discharge_inventory_error=(
            new_horizontal_discharge - old_horizontal_discharge
        ),
        gas_velocity_reconstruction=(
            "affine constant-strain u_g(x)=w_interface*x/x_event; "
            "closed wall and material-front kinematics"
        ),
        reconstructed_gas_kinetic_energy=reconstructed_kinetic,
    )
    return CaseAPortEventHandoff(
        west_faces=tuple(float(value) for value in old_faces[:event_face_index + 1]),
        west_cells=west_cells,
        node=node,
        east_faces=tuple(float(value - node_right) for value in east_faces),
        east_pressurised_cells=east_cells,
        vertical_faces=tuple(float(value) for value in vertical_faces),
        vertical_pressurised_cells=vertical_cells,
        vertical_open_surface_height=geometry.riser_liquid_height,
        diagnostics=diagnostics,
    )


__all__ = [
    "CaseAPortEventHandoff",
    "CaseAPortGeometry",
    "CaseAPortHandoffDiagnostics",
    "build_casea_port_event_handoff",
]
