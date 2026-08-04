"""Exact pre-T to post-T topology event for the Case-A shock fit.

The event represented here is deliberately *only* a change of graph
metadata.  It does not remap a finite-volume field, solve a junction, or add
storage at the tee.  When the fitted west-pocket interface reaches the
discrete junction face, its one-dimensional straight-pipe topology ceases to
exist.  The conservative horizontal snapshot is retained bit for bit, the old
interface is retired at the junction face, and independent east and vertical
branch fronts are created at zero branch distance.

This small type boundary is intentional: a :class:`HorizontalState` can be
advanced by the pre-arrival shock fitter, whereas a
:class:`PostTTopologyState` cannot.  Consequently, code using this module
cannot accidentally continue the legacy ``interface_x`` into the east dead
leg.  A later coupled node/branch solver must own the post-event evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from tosan2021_horizontal_shockfit import (
    HorizontalState,
    PolytropicGasInventory,
)


Array = np.ndarray
BranchName = Literal["east", "vertical"]


class TopologyEventError(RuntimeError):
    """Base class for an invalid or non-conservative topology event."""


class EventNotAtJunction(TopologyEventError):
    """Raised when the pre-arrival front has not been located at the T face."""


class InactiveLegacyInterfaceError(TopologyEventError):
    """Raised if a caller tries to advance the retired straight-pipe front."""


def _readonly_bit_copy(values: Array, *, name: str) -> Array:
    """Copy a one-dimensional floating field without changing any bits."""

    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(source.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if np.any(~np.isfinite(source)):
        raise ValueError(f"{name} must be finite")
    copied = np.array(source, copy=True, order="K", subok=False)
    copied.setflags(write=False)
    return copied


def _same_array_bits(left: Array, right: Array) -> bool:
    """Return true only for equal shape, dtype, and stored floating bits."""

    a = np.asarray(left)
    b = np.asarray(right)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.tobytes(order="C") == b.tobytes(order="C")
    )


@dataclass(frozen=True)
class ConservativeHorizontalSnapshot:
    """Read-only horizontal fields retained across the topology event.

    ``discharge`` is the liquid momentum variable per unit density used by the
    governing finite-volume equations.  Keeping the complete array bitwise
    unchanged is stronger than merely preserving its integral.
    """

    time: float
    dx: float
    area: Array = field(repr=False)
    discharge: Array = field(repr=False)
    gas: PolytropicGasInventory
    air_pressure_abs: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("snapshot time must be finite")
        if not np.isfinite(self.dx) or self.dx <= 0.0:
            raise ValueError("snapshot dx must be positive and finite")
        area = np.asarray(self.area)
        discharge = np.asarray(self.discharge)
        if area.ndim != 1 or area.shape != discharge.shape:
            raise ValueError("area and discharge must be equal-length 1-D arrays")
        if np.any(~np.isfinite(area)) or np.any(~np.isfinite(discharge)):
            raise ValueError("conservative fields must be finite")
        if np.any(area < 0.0):
            raise ValueError("liquid area cannot be negative")
        if not np.isfinite(self.air_pressure_abs) or self.air_pressure_abs <= 0.0:
            raise ValueError("absolute coupling pressure must be positive")
        if (
            not np.isfinite(self.gas.mass)
            or not np.isfinite(self.gas.volume)
            or not np.isfinite(self.gas.pressure_abs)
            or self.gas.mass <= 0.0
            or self.gas.volume <= 0.0
            or self.gas.pressure_abs <= 0.0
        ):
            raise ValueError("gas mass and volume must be positive")

    @property
    def liquid_volume(self) -> float:
        return float(np.sum(self.area, dtype=np.float64) * self.dx)

    @property
    def discharge_integral(self) -> float:
        """Integral of the per-density liquid momentum field."""

        return float(np.sum(self.discharge, dtype=np.float64) * self.dx)

    @property
    def gas_mass(self) -> float:
        return float(self.gas.mass)

    @property
    def gas_volume(self) -> float:
        return float(self.gas.volume)

    @property
    def gas_eos_pressure_abs(self) -> float:
        return float(self.gas.pressure_abs)


@dataclass(frozen=True)
class RetiredLegacyInterface:
    """The old one-front coordinate, permanently inactive at the T face."""

    junction_face_x: float
    position: float
    active: Literal[False] = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.junction_face_x) or self.junction_face_x <= 0.0:
            raise ValueError("junction face must be positive and finite")
        if not np.isfinite(self.position):
            raise ValueError("legacy interface position must be finite")
        if self.position > self.junction_face_x:
            raise ValueError("retired legacy interface cannot pass the T face")

    def advance(self, _distance: float) -> "RetiredLegacyInterface":
        """Reject every attempt to reuse the pre-arrival front coordinate."""

        raise InactiveLegacyInterfaceError(
            "the pre-T interface is inactive; advance east/vertical fronts "
            "through the coupled post-T graph instead"
        )


@dataclass(frozen=True)
class BranchFrontTopology:
    """One post-T material front measured outward from the junction."""

    branch: BranchName
    position: float = 0.0
    active: bool = True

    def __post_init__(self) -> None:
        if self.branch not in ("east", "vertical"):
            raise ValueError("post-T front branch must be east or vertical")
        if not np.isfinite(self.position) or self.position < 0.0:
            raise ValueError("branch-front position must be finite and nonnegative")

    def at(self, position: float) -> "BranchFrontTopology":
        """Return this branch at a new position without touching its sibling."""

        return replace(self, position=float(position))


@dataclass(frozen=True)
class PostTTopologyState:
    """Conservative event snapshot plus independent post-T front metadata."""

    horizontal: ConservativeHorizontalSnapshot
    legacy_interface: RetiredLegacyInterface
    east_front: BranchFrontTopology
    vertical_front: BranchFrontTopology
    topology: Literal["post_t_graph"] = field(
        default="post_t_graph", init=False
    )

    def __post_init__(self) -> None:
        if self.east_front.branch != "east":
            raise ValueError("east_front must own the east branch")
        if self.vertical_front.branch != "vertical":
            raise ValueError("vertical_front must own the vertical branch")
        if self.legacy_interface.active:
            raise ValueError("legacy interface must remain inactive")

    def with_east_front_position(self, position: float) -> "PostTTopologyState":
        return replace(self, east_front=self.east_front.at(position))

    def with_vertical_front_position(
        self, position: float
    ) -> "PostTTopologyState":
        return replace(self, vertical_front=self.vertical_front.at(position))


def create_post_t_topology_event(
    state: HorizontalState,
    *,
    junction_face_x: float,
    dx: float,
    location_tolerance: float | None = None,
) -> PostTTopologyState:
    """Retire the fitted straight-pipe front at the exact Case-A T event.

    The event locator must subdivide a crossing step before calling this
    function.  A state marginally below the face is accepted only within the
    supplied nonlinear tolerance and its *metadata* is attached to the exact
    face.  A state even marginally beyond the face is rejected: accepting it
    would silently let the old topology enter the east branch.

    No finite tee volume is introduced.  No liquid or gas field is projected.
    """

    x_t = float(junction_face_x)
    spacing = float(dx)
    if not np.isfinite(x_t) or x_t <= 0.0:
        raise ValueError("junction_face_x must be positive and finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("dx must be positive and finite")
    default_tolerance = (
        64.0 * np.finfo(float).eps * max(1.0, abs(x_t), spacing)
    )
    tolerance = (
        default_tolerance
        if location_tolerance is None
        else float(location_tolerance)
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("location_tolerance must be finite and nonnegative")
    interface_x = float(state.interface_x)
    if not np.isfinite(interface_x):
        raise EventNotAtJunction("pre-T interface position must be finite")
    if interface_x > x_t:
        raise EventNotAtJunction(
            "pre-T front has crossed the junction; subdivide the trial step "
            "and recreate the state at the event"
        )
    if x_t - interface_x > tolerance:
        raise EventNotAtJunction(
            "pre-T front has not yet reached the junction face"
        )

    area = _readonly_bit_copy(state.area, name="area")
    discharge = _readonly_bit_copy(state.discharge, name="discharge")
    snapshot = ConservativeHorizontalSnapshot(
        time=float(state.time),
        dx=spacing,
        area=area,
        discharge=discharge,
        gas=state.gas,
        air_pressure_abs=state.air_pressure_abs,
    )
    post = PostTTopologyState(
        horizontal=snapshot,
        legacy_interface=RetiredLegacyInterface(
            junction_face_x=x_t,
            position=x_t,
        ),
        east_front=BranchFrontTopology("east", 0.0),
        vertical_front=BranchFrontTopology("vertical", 0.0),
    )
    assert_exact_event_identity(state, post, dx=spacing)
    return post


def assert_exact_event_identity(
    before: HorizontalState,
    after: PostTTopologyState,
    *,
    dx: float,
) -> None:
    """Raise if a topology transition changed any physical event state."""

    spacing = float(dx)
    horizontal = after.horizontal
    if horizontal.dx != spacing:
        raise TopologyEventError("event snapshot changed the grid spacing")
    if not _same_array_bits(before.area, horizontal.area):
        raise TopologyEventError("topology event changed liquid-area bits")
    if not _same_array_bits(before.discharge, horizontal.discharge):
        raise TopologyEventError("topology event changed discharge/momentum bits")

    liquid_before = float(
        np.sum(np.asarray(before.area), dtype=np.float64) * spacing
    )
    momentum_before = float(
        np.sum(np.asarray(before.discharge), dtype=np.float64) * spacing
    )
    exact_scalars = (
        ("event time", float(before.time), float(horizontal.time)),
        ("liquid volume", liquid_before, horizontal.liquid_volume),
        ("momentum integral", momentum_before, horizontal.discharge_integral),
        ("gas mass", float(before.gas.mass), horizontal.gas_mass),
        ("gas volume", float(before.gas.volume), horizontal.gas_volume),
        (
            "gas EOS pressure",
            float(before.gas.pressure_abs),
            horizontal.gas_eos_pressure_abs,
        ),
        (
            "coupling pressure",
            float(before.air_pressure_abs),
            float(horizontal.air_pressure_abs),
        ),
    )
    for name, old, new in exact_scalars:
        if old != new:
            raise TopologyEventError(f"topology event changed {name}")
    if after.legacy_interface.active:
        raise TopologyEventError("legacy interface was not retired")
    if after.legacy_interface.position > after.legacy_interface.junction_face_x:
        raise TopologyEventError("legacy interface passed the T face")
    if after.east_front.position != 0.0 or after.vertical_front.position != 0.0:
        raise TopologyEventError("new branch fronts did not start at zero")


__all__ = [
    "BranchFrontTopology",
    "ConservativeHorizontalSnapshot",
    "EventNotAtJunction",
    "InactiveLegacyInterfaceError",
    "PostTTopologyState",
    "RetiredLegacyInterface",
    "TopologyEventError",
    "assert_exact_event_identity",
    "create_post_t_topology_event",
]
