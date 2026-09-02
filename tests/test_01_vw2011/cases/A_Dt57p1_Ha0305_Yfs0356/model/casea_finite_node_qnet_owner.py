"""Transactional ownership adapter for the experimental Case-A finite T node.

This module does not modify :mod:`vw2011_network_twofluid`.  It wraps the
isolated compressible-node SSP--RK2 step in the minimum network-integration
contract needed to make that node the *only* post-breakthrough owner of the
three-branch exchange.

The important point is that ``q_net`` is not a scalar source that may be
inserted independently into the riser.  The finite node returns one complete
gas/liquid flux on each of its west, east, and vertical faces.  Those six
conservative flux components must be committed to the adjacent branch cells
as one transaction.  The old characteristic ``G1[0]`` update, Taylor-return
mass replacement, post-breakthrough CCFL scaling of the signed net flux, and
distributed side-T source are mutually exclusive with this transaction.

No target time, target height, target flux, plotted curve, or empirical
water-entry multiplier exists here.  An incomplete or duplicate commit fails
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from casea_compressible_finite_node import CompressibleFiniteNodeState
from casea_compressible_node_postlaunch_stage import (
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchParameters,
)
from casea_compressible_node_ssprk2 import (
    CompressibleNodeSSPRK2Result,
    ssprk2_compressible_node_postlaunch_step,
)
from casea_material_front_cutcell import StratifiedFlux
from casea_vertical_mouth_twochannel_integration import (
    LegacyMouthPathActivity,
    require_exclusive_twochannel_ownership,
)


class FiniteNodeQnetOwnerError(RuntimeError):
    """Base class for a rejected finite-node network transaction."""


class IncompleteFiniteNodeCommit(FiniteNodeQnetOwnerError):
    """Not every finite-node face flux was committed to its neighbour."""


class DuplicateFiniteNodeCommit(FiniteNodeQnetOwnerError):
    """A finite-node face or component was committed more than once."""


@dataclass(frozen=True)
class CoordinateBranchFlux:
    """One finite-node face flux in the branch's outward coordinate."""

    gas_mass: float
    gas_momentum: float
    liquid_volume: float
    liquid_momentum: float

    @classmethod
    def from_stratified_flux(cls, flux: StratifiedFlux) -> "CoordinateBranchFlux":
        return cls(
            gas_mass=float(flux.gas_mass),
            gas_momentum=float(flux.gas_momentum),
            liquid_volume=float(flux.liquid_area),
            liquid_momentum=float(flux.liquid_momentum),
        )

    def values(self) -> tuple[float, float, float, float]:
        return (
            self.gas_mass,
            self.gas_momentum,
            self.liquid_volume,
            self.liquid_momentum,
        )


@dataclass(frozen=True)
class GlobalBranchFlux:
    """Flux mapped to the positive coordinate of the connected branch.

    West points opposite the global horizontal coordinate, so its gas-mass
    and liquid-volume fluxes change sign.  Momentum fluxes do not: under an
    axis reversal both momentum density and the face normal reverse, leaving
    the normal momentum tensor ``rho*u**2+p`` invariant.
    """

    gas_mass: float
    gas_momentum: float
    liquid_volume: float
    liquid_momentum: float


@dataclass(frozen=True)
class FiniteNodeQnetTransaction:
    """One accepted node step and all face fluxes that must be committed."""

    result: CompressibleNodeSSPRK2Result
    outward: Mapping[str, CoordinateBranchFlux]
    global_coordinate: Mapping[str, GlobalBranchFlux]
    q_net: float
    liquid_inventory_residual: float
    gas_inventory_residual: float

    @property
    def next_node_state(self) -> CompressibleFiniteNodeState:
        return self.result.state


_BRANCHES = ("west", "east", "vertical")
_COMPONENTS = ("gas_mass", "gas_momentum", "liquid_volume", "liquid_momentum")


def _map_global(name: str, flux: CoordinateBranchFlux) -> GlobalBranchFlux:
    if name not in _BRANCHES:
        raise ValueError(f"unknown finite-node branch {name!r}")
    orientation = -1.0 if name == "west" else 1.0
    return GlobalBranchFlux(
        gas_mass=orientation * flux.gas_mass,
        gas_momentum=flux.gas_momentum,
        liquid_volume=orientation * flux.liquid_volume,
        liquid_momentum=flux.liquid_momentum,
    )


def advance_finite_node_qnet_owner(
    state: CompressibleFiniteNodeState,
    *,
    dt: float,
    west: CompressibleNodeResolvedBranch,
    east: CompressibleNodeResolvedBranch,
    vertical: CompressibleNodeResolvedBranch,
    params: CompressiblePostLaunchParameters,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> FiniteNodeQnetTransaction:
    """Advance the sole finite-node owner and return an atomic commit plan.

    Branch traces are frozen inside the isolated SSP--RK2 wrapper, exactly as
    documented by that experimental core.  A network integrator must still
    evaluate its branch predictor states at the same RK stages before this
    adapter can be promoted to production.  This function nevertheless makes
    the ownership and conservative face mapping executable and testable.
    """

    require_exclusive_twochannel_ownership(legacy_activity)
    result = ssprk2_compressible_node_postlaunch_step(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    outward = {
        name: CoordinateBranchFlux.from_stratified_flux(
            result.branch_fluxes[name]
        )
        for name in _BRANCHES
    }
    global_coordinate = {
        name: _map_global(name, outward[name]) for name in _BRANCHES
    }
    q_net = outward["vertical"].liquid_volume
    liquid_residual = float(result.ledger.liquid_inventory_balance_residual)
    gas_residual = float(result.ledger.gas_mass_balance_residual)
    values = [
        dt,
        q_net,
        liquid_residual,
        gas_residual,
        *(value for flux in outward.values() for value in flux.values()),
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise FiniteNodeQnetOwnerError("finite-node transaction is non-finite")
    scale = max(
        abs(result.ledger.actual_liquid_equivalent_volume_change),
        abs(result.ledger.actual_gas_mass_change),
        state.node_total_volume,
        1.0e-15,
    )
    tolerance = 512.0 * math.ulp(scale)
    if abs(liquid_residual) > tolerance or abs(gas_residual) > tolerance:
        raise FiniteNodeQnetOwnerError(
            "finite-node SSP-RK2 inventory ledger did not close"
        )
    return FiniteNodeQnetTransaction(
        result=result,
        outward=outward,
        global_coordinate=global_coordinate,
        q_net=float(q_net),
        liquid_inventory_residual=liquid_residual,
        gas_inventory_residual=gas_residual,
    )


def required_commit_keys() -> frozenset[str]:
    """Return the exact branch/component keys required for one commit."""

    return frozenset(
        f"{branch}.{component}"
        for branch in _BRANCHES
        for component in _COMPONENTS
    )


def verify_atomic_branch_commit(
    committed_keys: object,
) -> None:
    """Reject missing, duplicate, or extraneous finite-node face commits.

    ``committed_keys`` must be an iterable, not a set, so duplicate writes can
    be detected.  A caller should append a key only after the corresponding
    shared face flux has been inserted into the neighbouring branch residual.
    """

    try:
        keys = tuple(str(value) for value in committed_keys)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("committed_keys must be iterable") from exc
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise DuplicateFiniteNodeCommit(
            "finite-node face components committed more than once: "
            + ", ".join(duplicates)
        )
    required = required_commit_keys()
    supplied = frozenset(keys)
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise IncompleteFiniteNodeCommit(
            "finite-node branch transaction was not atomic: "
            + "; ".join(details)
        )


__all__ = [
    "CoordinateBranchFlux",
    "DuplicateFiniteNodeCommit",
    "FiniteNodeQnetOwnerError",
    "FiniteNodeQnetTransaction",
    "GlobalBranchFlux",
    "IncompleteFiniteNodeCommit",
    "advance_finite_node_qnet_owner",
    "required_commit_keys",
    "verify_atomic_branch_commit",
]
