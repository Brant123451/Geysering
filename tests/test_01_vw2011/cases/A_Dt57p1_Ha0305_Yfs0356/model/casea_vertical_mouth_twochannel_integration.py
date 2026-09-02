"""Conservative integration adapter for the Case-A two-channel mouth closure.

This module is deliberately separate from :mod:`vw2011_network_twofluid`.
It defines the ownership and residual contracts needed to connect a finite
T-node to a riser that resolves simultaneous upward and downward liquid.  It
does **not** select an event time, prescribe a liquid height, inspect a plotted
result, or mutate the production solver.

The essential distinction is between

* the signed finite-node volume flux ``q_net``; and
* the two gross mouth fluxes ``Q_up`` and ``Q_down``.

Only ``q_net = Q_up - Q_down`` enters the combined liquid inventory ledger.
The gross streams nevertheless carry different axial momentum and kinetic
energy.  Consequently a one-momentum riser may consume ``q_net`` safely but
may use the gross quantities only as diagnostics.  A production calculation
that claims to resolve counter-current motion must retain a second liquid
momentum (and either a resolved or algebraically closed stream-area split).

The adapter also makes the side-T ownership explicit.  Once it owns the mouth,
the following legacy operations are mutually exclusive and must be disabled:

* a separately applied characteristic bottom-face liquid flux;
* the Taylor-sweep return as an additional mass flux;
* a post-breakthrough CCFL limiter applied to the signed net flux; and
* the old net-only horizontal side source.

Taylor-front geometry may still be advanced.  Its film geometry supplies the
local phase state used by the two-channel constitutive closure; it simply may
not move the same liquid a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np

from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    LiquidDonorInventories,
    TwoChannelMouthResult,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
    close_vertical_mouth_twochannel_exchange,
)


class MouthIntegrationError(RuntimeError):
    """Base class for an inadmissible two-channel integration stage."""


class DuplicateMouthFluxOwner(MouthIntegrationError):
    """More than one algorithm attempts to update the same T-mouth flux."""


class SecondLiquidMomentumRequired(MouthIntegrationError):
    """Gross counter-current momentum was requested from a one-momentum state."""


class HorizontalNodeTopology(str, Enum):
    """Which control volume owns the horizontal side of the mouth.

    The two choices are exclusive.  ``EXPLICIT_FINITE_NODE`` means that the
    finite-node west/east branch fluxes already connect the horizontal cells;
    no distributed side source may then be applied.  The footprint option is
    only a transitional adapter for the current solver, whose horizontal T
    cell itself acts as the node.
    """

    DISTRIBUTED_FOOTPRINT = "distributed_horizontal_footprint"
    EXPLICIT_FINITE_NODE = "explicit_finite_node"


@dataclass(frozen=True)
class LegacyMouthPathActivity:
    """Whether a legacy operation has already been applied in the current step.

    Computing a characteristic as a *candidate* for the finite-node solve is
    harmless.  Set ``characteristic_bottom_flux_applied`` only if its signed
    flux has already changed either connected liquid inventory.
    """

    characteristic_bottom_flux_applied: bool = False
    taylor_return_mass_flux_applied: bool = False
    post_breakthrough_ccfl_net_flux_applied: bool = False
    net_only_horizontal_side_source_applied: bool = False

    @property
    def active_paths(self) -> tuple[str, ...]:
        names = (
            (
                "characteristic_bottom_flux",
                self.characteristic_bottom_flux_applied,
            ),
            ("taylor_return_mass_flux", self.taylor_return_mass_flux_applied),
            (
                "post_breakthrough_ccfl_net_flux",
                self.post_breakthrough_ccfl_net_flux_applied,
            ),
            (
                "net_only_horizontal_side_source",
                self.net_only_horizontal_side_source_applied,
            ),
        )
        return tuple(name for name, active in names if active)


def require_exclusive_twochannel_ownership(
    activity: LegacyMouthPathActivity,
) -> None:
    """Reject a stage in which a legacy path already moved mouth liquid."""

    conflicts = activity.active_paths
    if conflicts:
        raise DuplicateMouthFluxOwner(
            "two-channel mouth must be the sole liquid-flux owner; "
            "already applied: " + ", ".join(conflicts)
        )


@dataclass(frozen=True)
class TwoLiquidMomentumBoundaryResidual:
    """Bottom-face residual for a riser with two liquid momentum states.

    ``upward_volume_rate`` is positive into the riser.  The downward stream
    leaves through the bottom, hence its signed inventory rate is negative.
    Both convective momentum fluxes are positive in the upward-coordinate
    conservation flux ``A u^2``.  Their sum is not recoverable from ``q_net``.
    """

    upward_volume_rate: float
    downward_volume_rate: float
    upward_convective_momentum_flux: float
    downward_convective_momentum_flux: float

    @property
    def total_volume_rate(self) -> float:
        return self.upward_volume_rate + self.downward_volume_rate

    @property
    def total_convective_momentum_flux(self) -> float:
        return (
            self.upward_convective_momentum_flux
            + self.downward_convective_momentum_flux
        )


@dataclass(frozen=True)
class TwoChannelMouthCouplingPlan:
    """One conservative, exclusively owned mouth-coupling stage."""

    exchange: TwoChannelMouthResult
    vertical_boundary: TwoLiquidMomentumBoundaryResidual
    horizontal_liquid_volume_rate: float
    vertical_liquid_volume_rate: float
    horizontal_axial_kinematic_momentum_rate: float
    horizontal_node_topology: HorizontalNodeTopology
    legacy_paths_to_disable: tuple[str, ...]

    @property
    def requires_second_liquid_momentum(self) -> bool:
        return self.exchange.circulation_flow > 0.0

    @property
    def combined_liquid_volume_rate(self) -> float:
        return self.horizontal_liquid_volume_rate + self.vertical_liquid_volume_rate

    def single_momentum_vertical_flux(self) -> float:
        """Return the safe bulk flux only when no counter-current stream exists.

        The returned quantity has the units of the production solver's
        convective discharge flux, ``A u^2``.  Silently substituting the gross
        two-stream second moment into a single-velocity cell would create an
        unresolved Reynolds stress and is therefore rejected.
        """

        if self.requires_second_liquid_momentum:
            raise SecondLiquidMomentumRequired(
                "Q_c is nonzero: retain a second riser liquid momentum state "
                "or use the gross exchange as an audit only"
            )
        return self.vertical_boundary.total_convective_momentum_flux


def stage_twochannel_mouth_coupling(
    q_net: float,
    *,
    phase: VerticalMouthPhaseState,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    donors: LiquidDonorInventories,
    losses: DirectionalMouthLosses,
    horizontal_axial_velocity: float,
    horizontal_node_topology: HorizontalNodeTopology,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> TwoChannelMouthCouplingPlan:
    """Build the only admissible liquid update at one finite-node mouth.

    ``q_net`` must come from the conservative finite-node stage and is never
    altered here.  Positive values leave the horizontal node and enter the
    riser.  Upward liquid removes its donor's horizontal axial momentum; water
    returning through the orthogonal branch enters with zero horizontal axial
    momentum.  No directional impulse is prescribed.
    """

    require_exclusive_twochannel_ownership(legacy_activity)
    if not math.isfinite(horizontal_axial_velocity):
        raise ValueError("horizontal mouth velocity must be finite")
    exchange = close_vertical_mouth_twochannel_exchange(
        q_net,
        phase=phase,
        geometry=geometry,
        material=material,
        wallis=wallis,
        donors=donors,
        losses=losses,
    )
    vertical = TwoLiquidMomentumBoundaryResidual(
        upward_volume_rate=exchange.upward_flow,
        downward_volume_rate=-exchange.downward_flow,
        upward_convective_momentum_flux=(
            exchange.upward_flow * exchange.upward_channel_velocity
        ),
        downward_convective_momentum_flux=(
            exchange.downward_flow * abs(exchange.downward_channel_velocity)
        ),
    )
    horizontal_momentum_rate = -(
        exchange.upward_flow * float(horizontal_axial_velocity)
    )
    plan = TwoChannelMouthCouplingPlan(
        exchange=exchange,
        vertical_boundary=vertical,
        horizontal_liquid_volume_rate=-exchange.q_net,
        vertical_liquid_volume_rate=exchange.q_net,
        horizontal_axial_kinematic_momentum_rate=horizontal_momentum_rate,
        horizontal_node_topology=HorizontalNodeTopology(horizontal_node_topology),
        legacy_paths_to_disable=(
            "characteristic_bottom_flux_as_update",
            "taylor_return_as_mass_flux",
            "post_breakthrough_ccfl_on_q_net",
            "net_only_horizontal_side_source",
        ),
    )
    tolerance = 128.0 * math.ulp(
        max(
            abs(exchange.upward_flow),
            abs(exchange.downward_flow),
            abs(exchange.q_net),
            1.0,
        )
    )
    if abs(plan.combined_liquid_volume_rate) > tolerance:
        raise FloatingPointError("two-channel plan violates combined liquid conservation")
    if not math.isclose(
        vertical.total_volume_rate,
        exchange.q_net,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise FloatingPointError("two-liquid boundary residual does not recover q_net")
    return plan


def stage_from_finite_node_ssprk2(
    finite_node_result,
    *,
    phase: VerticalMouthPhaseState,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    riser_liquid_donor_volume: float,
    losses: DirectionalMouthLosses,
    horizontal_axial_velocity_diagnostic: float = 0.0,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> TwoChannelMouthCouplingPlan:
    """Adapt the isolated finite-node SSP--RK2 result without re-advancing it.

    The finite node has already updated its own gas/liquid inventories with the
    time-averaged west, east, and vertical *net* branch fluxes.  This function
    therefore only builds the matching riser two-stream boundary residual; the
    caller must apply ``q_net`` to the riser branch once and must not call the
    distributed horizontal-footprint update.

    The duck-typed result is expected to provide
    ``vertical.liquid_area`` (outward volume rate),
    ``ledger.initial_state.liquid_equivalent_volume``, and ``ledger.dt``.  These
    are the public fields of ``CompressibleNodeSSPRK2Result`` and avoid a hard
    dependency from this constitutive adapter to the finite-node integrator.
    """

    try:
        q_net = float(finite_node_result.vertical.liquid_area)
        node_liquid = float(
            finite_node_result.ledger.initial_state.liquid_equivalent_volume
        )
        dt = float(finite_node_result.ledger.dt)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid finite-node SSP-RK2 result contract") from exc
    riser_liquid = float(riser_liquid_donor_volume)
    if not math.isfinite(riser_liquid) or riser_liquid < 0.0:
        raise ValueError("riser liquid donor volume must be finite and non-negative")
    donors = LiquidDonorInventories(
        finite_node_volume=node_liquid,
        riser_volume=riser_liquid,
        time_step=dt,
    )
    return stage_twochannel_mouth_coupling(
        q_net,
        phase=phase,
        geometry=geometry,
        material=material,
        wallis=wallis,
        donors=donors,
        losses=losses,
        horizontal_axial_velocity=horizontal_axial_velocity_diagnostic,
        horizontal_node_topology=HorizontalNodeTopology.EXPLICIT_FINITE_NODE,
        legacy_activity=legacy_activity,
    )


@dataclass(frozen=True)
class LumpedInventoryUpdate:
    """Exact node/riser inventory update using the shared signed net flux."""

    finite_node_liquid_volume: float
    riser_liquid_volume: float
    combined_volume_before: float
    combined_volume_after: float
    conservation_residual: float


def advance_lumped_liquid_inventories(
    finite_node_liquid_volume: float,
    riser_liquid_volume: float,
    *,
    time_step: float,
    plan: TwoChannelMouthCouplingPlan,
) -> LumpedInventoryUpdate:
    """Apply ``q_net`` once to the two connected liquid inventories."""

    node = float(finite_node_liquid_volume)
    riser = float(riser_liquid_volume)
    dt = float(time_step)
    if not all(math.isfinite(value) for value in (node, riser, dt)):
        raise ValueError("lumped mouth inventory data must be finite")
    if node < 0.0 or riser < 0.0 or dt <= 0.0:
        raise ValueError("non-negative inventories and positive time step required")
    node_new = node + dt * plan.horizontal_liquid_volume_rate
    riser_new = riser + dt * plan.vertical_liquid_volume_rate
    tolerance = 128.0 * math.ulp(max(node, riser, 1.0))
    if node_new < -tolerance or riser_new < -tolerance:
        raise MouthIntegrationError("two-channel net update exhausted a liquid inventory")
    before = node + riser
    after = max(node_new, 0.0) + max(riser_new, 0.0)
    residual = after - before
    if abs(residual) > tolerance:
        raise FloatingPointError("lumped two-channel update lost liquid volume")
    return LumpedInventoryUpdate(
        finite_node_liquid_volume=max(node_new, 0.0),
        riser_liquid_volume=max(riser_new, 0.0),
        combined_volume_before=before,
        combined_volume_after=after,
        conservation_residual=residual,
    )


def finite_width_node_liquid_inventory(
    liquid_area: Iterable[float],
    opening_weights: Iterable[float],
    *,
    cell_width: float,
    opening_length: float | None = None,
) -> float:
    """Return the liquid inventory inside the measured T footprint.

    ``opening_weights`` are normalized overlap lengths.  Consequently the
    physical overlap represented by cell ``i`` is ``w_i * opening_length``.
    The former ``sum(w_i A_i) * cell_width`` expression was correct only when
    the mouth happened to be exactly one grid cell wide; at any other
    resolution it made the node donor inventory grid dependent.  Omitting
    ``opening_length`` retains that one-cell convention for existing callers.
    """

    area = np.asarray(tuple(liquid_area), dtype=float)
    weights = np.asarray(tuple(opening_weights), dtype=float)
    width = float(cell_width)
    footprint_length = (
        width if opening_length is None else float(opening_length)
    )
    if area.ndim != 1 or area.shape != weights.shape or area.size == 0:
        raise ValueError("finite-width node arrays must be equal nonempty vectors")
    if not np.all(np.isfinite(area)) or not np.all(np.isfinite(weights)):
        raise ValueError("finite-width node arrays must be finite")
    if (
        np.any(area < 0.0)
        or np.any(weights < 0.0)
        or width <= 0.0
        or not math.isfinite(footprint_length)
        or footprint_length <= 0.0
    ):
        raise ValueError(
            "non-negative areas/weights and positive grid/footprint lengths required"
        )
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("finite-width node requires a positive opening weight")
    normalized = weights / total_weight
    overlap_length = normalized * footprint_length
    tolerance = 128.0 * math.ulp(max(width, footprint_length, 1.0))
    if np.any(overlap_length > width + tolerance):
        raise ValueError(
            "normalized weights and opening_length overlap more than one cell"
        )
    return float(np.sum(area * overlap_length))


@dataclass(frozen=True)
class HorizontalFootprintUpdate:
    """Gross-aware horizontal side-source replacement and its exact ledger."""

    liquid_area: np.ndarray
    liquid_discharge: np.ndarray
    removed_upward_volume: float
    deposited_downward_volume: float
    net_horizontal_volume_change: float
    axial_kinematic_momentum_change: float
    volume_residual: float


def apply_twochannel_horizontal_footprint(
    liquid_area: Iterable[float],
    liquid_discharge: Iterable[float],
    opening_weights: Iterable[float],
    *,
    cell_width: float,
    opening_length: float | None = None,
    time_step: float,
    plan: TwoChannelMouthCouplingPlan,
) -> HorizontalFootprintUpdate:
    """Replace the old net-only side source with one gross-aware update.

    Upward liquid is removed from the pre-step footprint in proportion to its
    weighted local inventory and carries the corresponding horizontal parcel
    momentum.  Downward liquid is then deposited with zero horizontal momentum,
    because the T wall supplies the ninety-degree turning reaction.  The volume
    update is exactly ``-q_net * time_step`` and is applied nowhere else.
    """

    if plan.horizontal_node_topology is not HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT:
        raise DuplicateMouthFluxOwner(
            "explicit finite-node west/east fluxes already own the horizontal "
            "connection; a distributed side source would count it twice"
        )

    area = np.asarray(tuple(liquid_area), dtype=float)
    discharge = np.asarray(tuple(liquid_discharge), dtype=float)
    weights = np.asarray(tuple(opening_weights), dtype=float)
    width = float(cell_width)
    footprint_length = (
        width if opening_length is None else float(opening_length)
    )
    dt = float(time_step)
    if not (area.ndim == 1 and area.shape == discharge.shape == weights.shape):
        raise ValueError("horizontal footprint arrays must be equal vectors")
    if (
        area.size == 0
        or width <= 0.0
        or not math.isfinite(footprint_length)
        or footprint_length <= 0.0
        or dt <= 0.0
    ):
        raise ValueError(
            "nonempty footprint and positive grid/footprint/time scales required"
        )
    if not (
        np.all(np.isfinite(area))
        and np.all(np.isfinite(discharge))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("horizontal footprint state must be finite")
    if np.any(area < 0.0) or np.any(weights < 0.0):
        raise ValueError("horizontal footprint areas and weights cannot be negative")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("horizontal footprint requires positive opening weights")
    normalized = weights / total_weight
    overlap_length = normalized * footprint_length
    overlap_tolerance = 128.0 * math.ulp(
        max(width, footprint_length, 1.0)
    )
    if np.any(overlap_length > width + overlap_tolerance):
        raise ValueError(
            "normalized weights and opening_length overlap more than one cell"
        )

    old_volume = area * width
    old_integrated_momentum = discharge * width
    # Only the physical slice beneath the circular opening can donate directly
    # to the riser.  This is the exact overlap integral A_l * dL, independent
    # of whether the opening spans one, two, or many numerical cells.
    weighted_inventory = area * overlap_length
    donor_inventory = float(np.sum(weighted_inventory))
    upward_volume = plan.exchange.upward_flow * dt
    downward_volume = plan.exchange.downward_flow * dt
    tolerance = 128.0 * math.ulp(max(donor_inventory, upward_volume, 1.0))
    if upward_volume > donor_inventory + tolerance:
        raise MouthIntegrationError(
            "gross upward stream exceeds the weighted horizontal donor inventory"
        )
    if donor_inventory > 0.0 and upward_volume > 0.0:
        removal = upward_volume * weighted_inventory / donor_inventory
    else:
        removal = np.zeros_like(old_volume)
    removal = np.minimum(removal, old_volume)
    removed = float(np.sum(removal))
    if not math.isclose(removed, upward_volume, rel_tol=1.0e-12, abs_tol=tolerance):
        raise FloatingPointError("horizontal gross-up allocation lost liquid")

    local_velocity = np.divide(
        discharge,
        area,
        out=np.zeros_like(discharge),
        where=area > 0.0,
    )
    momentum_removed = removal * local_velocity
    # Returning liquid enters normal to the horizontal axis.  The geometric
    # footprint weights are the only distribution; no directional wave shape
    # or preferred horizontal branch is introduced.
    deposition = downward_volume * normalized
    new_volume = old_volume - removal + deposition
    new_integrated_momentum = old_integrated_momentum - momentum_removed
    if np.any(new_volume < -tolerance):
        raise FloatingPointError("gross-aware horizontal exchange made a cell negative")
    new_volume = np.maximum(new_volume, 0.0)
    new_area = new_volume / width
    new_discharge = new_integrated_momentum / width

    actual_change = float(np.sum(new_volume - old_volume))
    expected_change = -plan.exchange.q_net * dt
    residual = actual_change - expected_change
    if not math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=tolerance):
        raise FloatingPointError("gross-aware horizontal exchange lost net volume")
    momentum_change = float(
        np.sum(new_integrated_momentum - old_integrated_momentum)
    )
    return HorizontalFootprintUpdate(
        liquid_area=new_area,
        liquid_discharge=new_discharge,
        removed_upward_volume=removed,
        deposited_downward_volume=float(np.sum(deposition)),
        net_horizontal_volume_change=actual_change,
        axial_kinematic_momentum_change=momentum_change,
        volume_residual=residual,
    )


__all__ = [
    "DuplicateMouthFluxOwner",
    "HorizontalFootprintUpdate",
    "HorizontalNodeTopology",
    "LegacyMouthPathActivity",
    "LumpedInventoryUpdate",
    "MouthIntegrationError",
    "SecondLiquidMomentumRequired",
    "TwoChannelMouthCouplingPlan",
    "TwoLiquidMomentumBoundaryResidual",
    "advance_lumped_liquid_inventories",
    "apply_twochannel_horizontal_footprint",
    "finite_width_node_liquid_inventory",
    "require_exclusive_twochannel_ownership",
    "stage_twochannel_mouth_coupling",
    "stage_from_finite_node_ssprk2",
]
