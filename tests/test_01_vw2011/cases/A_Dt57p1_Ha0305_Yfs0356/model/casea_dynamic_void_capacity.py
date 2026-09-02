"""Dynamic material-gas void capacity for the Case-A riser liquid step.

The liquid and gas equations are advanced in split conservative stages.  A
liquid receiving-capacity projection therefore needs a gas-volume constraint
*before* it advances: otherwise a cell that still contains conserved tunnel
gas can be filled to the bore area and the following gas stage sees a
zero-volume material state.

This helper converts the conservative gas inventory into that constraint.  In
a cell containing tunnel-origin tracer, the isothermal gas volume compatible
with the current liquid-side absolute pressure target is

``V_g,* = m_g R T / p_l,*``.

The corresponding liquid-area capacity is recomputed from gas mass and
pressure every stage.  It is consequently neither a fixed Taylor-core
fraction nor the current liquid area frozen as a permanent corridor.  The
capacity is never made smaller than the liquid already in the cell; any gas
volume that the current split state cannot provide is reported explicitly as
an unresolved compression deficit and must be relieved by the pressure /
momentum stage rather than by clipping liquid or deleting gas.

Call :func:`compute_dynamic_material_void_capacity` at the beginning of a
liquid FV stage with the same conservative gas mass and tracer that will be
advanced by the paired gas stage.  Hold the returned capacity fixed for that
liquid stage, then recompute it after the gas inventory and pressure target
have changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class DynamicMaterialVoidCapacity:
    """Cellwise gas-volume constraint and diagnostics.

    ``liquid_capacity_area`` is the array supplied to the vertical liquid
    receiving-capacity owner.  ``eos_void_area`` is the unconstrained local
    ideal-gas area at the requested absolute pressure.  A cell can provide at
    most one bore area, represented by ``required_void_area``.  If the current
    liquid inventory has already compressed the gas below that target,
    ``compression_deficit_area`` is positive and the returned liquid capacity
    equals the current liquid area, preventing any further same-stage closure.

    ``cell_expansion_excess_area`` is positive when the gas inventory would
    require more than the entire cell at the requested pressure.  It is a
    signal that gas transport to neighbouring cells is required; a local
    capacity cannot resolve that expansion by itself.
    """

    liquid_capacity_area: Array
    material_gas_mask: Array
    topology_void_mask: Array
    eos_void_area: Array
    minimum_topology_void_area: Array
    required_void_area: Array
    reserved_void_area: Array
    compression_deficit_area: Array
    cell_expansion_excess_area: Array
    available_liquid_filling_area: Array


def _one_dimensional_field(
    values: Iterable[float],
    *,
    name: str,
) -> Array:
    field = np.asarray(values, dtype=float)
    if field.ndim != 1 or field.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional field")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite")
    return field.copy()


def _pressure_field(values: Iterable[float] | float, shape: tuple[int, ...]) -> Array:
    pressure = np.asarray(values, dtype=float)
    if pressure.ndim == 0:
        pressure = np.full(shape, float(pressure), dtype=float)
    elif pressure.shape != shape:
        raise ValueError("liquid_pressure_target must be scalar or one value per cell")
    else:
        pressure = pressure.copy()
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("liquid_pressure_target must be finite, absolute, and positive")
    return pressure


def compute_dynamic_material_void_capacity(
    *,
    gas_mass: Iterable[float],
    tracer_mass: Iterable[float],
    liquid_pressure_target: Iterable[float] | float,
    current_liquid_area: Iterable[float],
    full_area: float,
    cell_length: float,
    gas_constant: float = 287.05,
    gas_temperature: float = 293.0,
    tracer_mass_tolerance: float = 0.0,
    area_tolerance: float = 0.0,
    minimum_topology_void_area: Iterable[float] | float = 0.0,
) -> DynamicMaterialVoidCapacity:
    """Return a mass-backed, pressure-dependent liquid-area capacity.

    Tunnel-origin tracer is a topology label, not the volume estimate.  Any
    cell with ``tracer_mass > tracer_mass_tolerance`` is material-gas active;
    the *total* gas mass in that cell then sets its isothermal void because all
    gas constituents share the same pressure and volume.  Cells without
    material tracer retain the full liquid bore capacity even if they contain
    a positivity-floor gas mass.

    The returned capacity satisfies

    ``current_liquid_area <= liquid_capacity_area <= full_area``

    (up to ``area_tolerance`` on input).  Thus applying it before a liquid FV
    stage never clips existing liquid.  For every feasible material cell it
    also enforces

    ``full_area - liquid_capacity_area = m_g R T / (p_l,* dz) > 0``.

    If the current state is already more compressed than that relation, the
    helper holds the liquid capacity at the current area and reports the
    missing void in ``compression_deficit_area``.  It does not relabel,
    redistribute, or delete gas/tracer mass.
    """

    mass = _one_dimensional_field(gas_mass, name="gas_mass")
    tracer = _one_dimensional_field(tracer_mass, name="tracer_mass")
    liquid = _one_dimensional_field(
        current_liquid_area,
        name="current_liquid_area",
    )
    if mass.shape != tracer.shape or mass.shape != liquid.shape:
        raise ValueError("gas, tracer, and liquid fields must have one common shape")
    pressure = _pressure_field(liquid_pressure_target, mass.shape)
    minimum_void = np.asarray(minimum_topology_void_area, dtype=float)
    if minimum_void.ndim == 0:
        minimum_void = np.full(mass.shape, float(minimum_void), dtype=float)
    elif minimum_void.shape != mass.shape:
        raise ValueError(
            "minimum_topology_void_area must be scalar or one value per cell"
        )
    else:
        minimum_void = minimum_void.copy()
    if not np.all(np.isfinite(minimum_void)) or np.any(minimum_void < 0.0):
        raise ValueError("minimum_topology_void_area must be finite and non-negative")
    if np.any(minimum_void > full_area + area_tolerance):
        raise ValueError("minimum topology void exceeds the riser bore")
    minimum_void = np.clip(minimum_void, 0.0, full_area)

    scalar_values = (
        full_area,
        cell_length,
        gas_constant,
        gas_temperature,
        tracer_mass_tolerance,
        area_tolerance,
    )
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("capacity geometry and thermodynamic inputs must be finite")
    if min(full_area, cell_length, gas_constant, gas_temperature) <= 0.0:
        raise ValueError("capacity geometry and thermodynamic scales must be positive")
    if tracer_mass_tolerance < 0.0 or area_tolerance < 0.0:
        raise ValueError("capacity tolerances cannot be negative")
    if np.any(mass < 0.0) or np.any(tracer < 0.0):
        raise ValueError("conservative gas and tracer masses cannot be negative")

    mass_scale = max(
        float(np.max(mass, initial=0.0)),
        float(np.max(tracer, initial=0.0)),
        np.finfo(float).tiny,
    )
    mass_roundoff = 128.0 * np.finfo(float).eps * mass_scale
    if np.any(tracer > mass + mass_roundoff):
        raise ValueError("tracer mass cannot exceed total gas mass")
    if np.any(liquid < -area_tolerance) or np.any(
        liquid > full_area + area_tolerance
    ):
        raise ValueError("current liquid area lies outside the riser bore")
    liquid = np.clip(liquid, 0.0, full_area)

    material = tracer > tracer_mass_tolerance
    eos_void = np.zeros_like(mass)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eos_void[material] = (
            mass[material]
            * gas_constant
            * gas_temperature
            / (pressure[material] * cell_length)
        )
    if not np.all(np.isfinite(eos_void)):
        raise FloatingPointError("material-gas EOS void area is non-finite")

    topology_void = minimum_void > area_tolerance
    required_void = np.maximum(
        np.where(material, np.minimum(eos_void, full_area), 0.0),
        minimum_void,
    )
    present_void = full_area - liquid
    desired_liquid_capacity = full_area - required_void

    # A capacity owner must not create the missing gas volume by clipping an
    # existing liquid inventory.  Keeping the larger of the current area and
    # the EOS-compatible capacity blocks further closure and leaves the
    # pressure/momentum stage responsible for an already overcompressed cell.
    liquid_capacity = np.maximum(liquid, desired_liquid_capacity)
    liquid_capacity = np.clip(liquid_capacity, 0.0, full_area)
    reserved_void = full_area - liquid_capacity
    compression_deficit = np.maximum(required_void - present_void, 0.0)
    expansion_excess = np.where(
        material,
        np.maximum(eos_void - full_area, 0.0),
        0.0,
    )
    available_filling = np.maximum(liquid_capacity - liquid, 0.0)

    return DynamicMaterialVoidCapacity(
        liquid_capacity_area=liquid_capacity,
        material_gas_mask=material,
        topology_void_mask=topology_void,
        eos_void_area=eos_void,
        minimum_topology_void_area=minimum_void,
        required_void_area=required_void,
        reserved_void_area=reserved_void,
        compression_deficit_area=compression_deficit,
        cell_expansion_excess_area=expansion_excess,
        available_liquid_filling_area=available_filling,
    )


__all__ = (
    "DynamicMaterialVoidCapacity",
    "compute_dynamic_material_void_capacity",
)
