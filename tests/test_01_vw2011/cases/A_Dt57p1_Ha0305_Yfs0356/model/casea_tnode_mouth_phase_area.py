"""Target-free phase-area ownership for the Case-A shared T mouth.

After the persistent two-stream riser is activated, the horizontal crown
opening and the resolved void in the first riser cell form two apertures in
series.  The gas portion of their *shared* face is consequently their overlap,
not their union.  In particular, a large crown opening cannot remove liquid
area from a still mostly liquid-filled riser cut cell.

The vertical void is admitted as material gas geometry only when it is backed
by tunnel-origin gas mass or belongs to the currently swept Taylor cut cell.
The closure reads no clock, comparison field, desired holdup, or fitted flow.
It changes only the boundary trace areas; cell inventories remain prognostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TNodeMouthPhaseAreas:
    """One exact partition of the shared circular T-mouth aperture."""

    upward_area: float
    downward_area: float
    gas_area: float
    resolved_vertical_void_area: float
    mass_supported_vertical_void_area: float
    taylor_supported_vertical_void_area: float
    vertical_material_gas_area: float
    partition_residual: float

    @property
    def liquid_area(self) -> float:
        return self.upward_area + self.downward_area


def resolve_tnode_mouth_phase_areas(
    *,
    resolved_upward_area: float,
    resolved_downward_area: float,
    horizontal_gas_opening_area: float,
    vertical_tracer_gas_mass: float,
    full_area: float,
    vertical_cell_length: float,
    reference_gas_density: float,
    topology_density_fraction: float,
    taylor_swept_fraction: float,
    taylor_core_area_fraction: float,
    area_tolerance: float = 2.0e-12,
) -> TNodeMouthPhaseAreas:
    """Return a conservative, material-supported phase partition at the tee.

    ``horizontal_gas_opening_area`` is the crown-exposed tower-bore area from
    a mass-supported horizontal gas component.  The vertical side is obtained
    from the post-sweep two-stream inventories.  A resolved vertical void is
    material-supported by either

    * tunnel-origin tracer mass at the existing topology threshold, or
    * the gas-core portion of the currently swept Taylor cut cell.

    These two supports are alternatives for the same vertical void and are
    therefore combined with ``max``.  The horizontal and vertical apertures,
    however, are in series and are combined with ``min``.  Since the resulting
    gas area cannot exceed the resolved vertical void, the available liquid
    face is never smaller than the two resolved liquid inventories.  Before
    breakthrough, existing directional liquid areas are retained.  Taylor
    sweep history may support a material gas opening, but it does not relabel
    resolved liquid from the falling stream into an upward stream.  A new
    upward corridor is a pressure-characteristic decision made by the T-node
    solver, not a permanent geometric consequence of an earlier sweep.
    """

    values = (
        resolved_upward_area,
        resolved_downward_area,
        horizontal_gas_opening_area,
        vertical_tracer_gas_mass,
        full_area,
        vertical_cell_length,
        reference_gas_density,
        topology_density_fraction,
        taylor_swept_fraction,
        taylor_core_area_fraction,
        area_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("T-mouth phase-area inputs must be finite")
    if full_area <= 0.0 or vertical_cell_length <= 0.0:
        raise ValueError("T-mouth area and vertical cell length must be positive")
    if reference_gas_density <= 0.0:
        raise ValueError("reference gas density must be positive")
    if not 0.0 < topology_density_fraction < 1.0:
        raise ValueError("topology density fraction must lie in (0, 1)")
    if not 0.0 <= taylor_swept_fraction <= 1.0:
        raise ValueError("Taylor swept fraction must lie in [0, 1]")
    if not 0.0 < taylor_core_area_fraction <= 1.0:
        raise ValueError("Taylor core area fraction must lie in (0, 1]")
    if area_tolerance < 0.0:
        raise ValueError("area tolerance cannot be negative")
    if min(
        resolved_upward_area,
        resolved_downward_area,
        horizontal_gas_opening_area,
        vertical_tracer_gas_mass,
    ) < 0.0:
        raise ValueError("T-mouth areas and gas mass cannot be negative")

    upward = float(resolved_upward_area)
    downward = float(resolved_downward_area)
    area = float(full_area)
    resolved_liquid = upward + downward
    if resolved_liquid > area + float(area_tolerance):
        raise ValueError(
            "resolved two-stream liquid overpacks the T mouth: "
            f"upward={upward:.12e}, downward={downward:.12e}, "
            f"total={resolved_liquid:.12e}, full={area:.12e}, "
            f"excess={resolved_liquid - area:.12e}"
        )
    # Remove roundoff-only overpacking without changing either conserved cell.
    resolved_liquid = min(resolved_liquid, area)
    resolved_void = max(area - resolved_liquid, 0.0)

    supported_mass = (
        float(topology_density_fraction)
        * float(reference_gas_density)
        * resolved_void
        * float(vertical_cell_length)
    )
    mass_supported_void = (
        resolved_void
        if resolved_void > 0.0
        and float(vertical_tracer_gas_mass) > supported_mass
        else 0.0
    )
    taylor_cut_opening = min(
        resolved_void,
        float(taylor_swept_fraction)
        * float(taylor_core_area_fraction)
        * area,
    )
    vertical_material_gas = min(
        resolved_void,
        max(mass_supported_void, taylor_cut_opening),
    )
    horizontal_opening = min(float(horizontal_gas_opening_area), area)
    gas_area = min(horizontal_opening, vertical_material_gas)
    liquid_area = area - gas_area

    # gas_area <= resolved_void guarantees liquid_area >= resolved_liquid.
    # Preserve the prognostic falling area at the face.  Any shared aperture
    # that is not occupied by resolved liquid or material gas remains an
    # available upward entrance, but a historical Taylor sweep alone may not
    # manufacture that entrance by cutting the falling trace to a fixed film
    # fraction.  The latter was the source of the persistent 20%-area outlet
    # bottleneck in the post-breakthrough Case-A run.
    mouth_downward = min(downward, liquid_area)
    mouth_upward = liquid_area - mouth_downward
    residual = mouth_upward + mouth_downward + gas_area - area
    closure_scale = max(area, 1.0)
    if abs(residual) > max(float(area_tolerance), 64.0 * math.ulp(closure_scale)):
        raise FloatingPointError("T-mouth phase-area partition does not close")
    if mouth_upward + float(area_tolerance) < upward:
        raise FloatingPointError("T-mouth partition contracted resolved upward liquid")

    return TNodeMouthPhaseAreas(
        upward_area=float(mouth_upward),
        downward_area=float(mouth_downward),
        gas_area=float(gas_area),
        resolved_vertical_void_area=float(resolved_void),
        mass_supported_vertical_void_area=float(mass_supported_void),
        taylor_supported_vertical_void_area=float(taylor_cut_opening),
        vertical_material_gas_area=float(vertical_material_gas),
        partition_residual=float(residual),
    )


__all__ = [
    "TNodeMouthPhaseAreas",
    "resolve_tnode_mouth_phase_areas",
]
