"""Conservative ALE cut cell for one pressurised--stratified material front.

This module is deliberately independent of the Case-A time loop.  It is the
local finite-volume operation described by Eq. (28), Eqs. (33)--(36) of the
current methods text (Eqs. (42)--(44) in the earlier numbering): a host cell is
split by a tracked material front into a pressurised liquid subcell and a
stratified gas--liquid subcell.

The operator has no dry-state fill, area clipping, target waveform, or global
redistribution.  A subcell is allowed to have *exactly* zero length at a
topology event.  Its state is an interface trace and therefore carries zero
inventory.  If a disappearing subcell has a nonzero inventory at the event,
the step is rejected instead of silently discarding that inventory.

Both orientations are supported explicitly:

``pressurised_side='left'``
    ``P | S`` (the orientation used in the methods-paper sketch).

``pressurised_side='right'``
    ``S | P`` (the east and vertical Case-A fronts directed away from the T).

The caller supplies physical face fluxes and an interface closure.  The latter
may be obtained from ``solve_front_rankine_hugoniot``; this module only requires
the two liquid traces, the stratified gas trace, their physical fluxes, and the
front speed.  It verifies the Rankine--Hugoniot/ALE identities before updating
any inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Protocol, TypeAlias


PressurisedSide: TypeAlias = Literal["left", "right"]
BranchKind: TypeAlias = Literal["pressurised", "stratified"]


class MaterialFrontError(RuntimeError):
    """Base class for rejected material-front updates."""


class InterfaceClosureError(MaterialFrontError):
    """The supplied interface traces do not satisfy the ALE jump conditions."""


class VanishingSubcellInventoryError(MaterialFrontError):
    """A zero-measure subcell would retain a finite conserved inventory."""


class MultipleCrossingsError(MaterialFrontError):
    """More than one face crossing was requested in one accepted step."""


class DomainBoundaryCrossing(MaterialFrontError):
    """The tracked front reached the end of the supplied one-dimensional mesh."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _close_residual(value: float, scale: float, *, atol: float, rtol: float) -> bool:
    return abs(value) <= atol + rtol * max(1.0, abs(scale))


@dataclass(frozen=True)
class PressurisedState:
    """Cell-average pressurised liquid state ``(A, Q)``."""

    area: float
    discharge: float

    def __post_init__(self) -> None:
        if not _finite(self.area, self.discharge):
            raise ValueError("pressurised state must be finite")
        if self.area <= 0.0:
            raise ValueError("pressurised liquid area must be positive")

    def vector(self) -> tuple[float, float]:
        return (float(self.area), float(self.discharge))


@dataclass(frozen=True)
class StratifiedState:
    """Cell-average stratified state ``(m_g, j_g, A_l, Q_l)``.

    ``m_g=rho_g A_g`` and ``j_g=rho_g Q_g`` are gas mass and axial
    momentum per unit pipe length.  The gas velocity is therefore ``j_g/m_g``.
    """

    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_discharge: float

    def __post_init__(self) -> None:
        if not _finite(
            self.gas_mass,
            self.gas_momentum,
            self.liquid_area,
            self.liquid_discharge,
        ):
            raise ValueError("stratified state must be finite")
        if self.gas_mass <= 0.0:
            raise ValueError("stratified gas mass per length must be positive")
        if self.liquid_area <= 0.0:
            raise ValueError("stratified liquid area must be positive")

    @property
    def gas_velocity(self) -> float:
        return float(self.gas_momentum / self.gas_mass)

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_discharge),
        )


@dataclass(frozen=True)
class PressurisedFlux:
    """Physical ``+x`` liquid flux ``(F_A, F_Q)``."""

    area: float
    momentum: float

    def __post_init__(self) -> None:
        if not _finite(self.area, self.momentum):
            raise ValueError("pressurised flux must be finite")

    def vector(self) -> tuple[float, float]:
        return (float(self.area), float(self.momentum))


@dataclass(frozen=True)
class StratifiedFlux:
    """Physical ``+x`` flux ``(F_mg, F_jg, F_Al, F_Ql)``."""

    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_momentum: float

    def __post_init__(self) -> None:
        if not _finite(
            self.gas_mass,
            self.gas_momentum,
            self.liquid_area,
            self.liquid_momentum,
        ):
            raise ValueError("stratified flux must be finite")

    def vector(self) -> tuple[float, float, float, float]:
        return (
            float(self.gas_mass),
            float(self.gas_momentum),
            float(self.liquid_area),
            float(self.liquid_momentum),
        )


@dataclass(frozen=True)
class InterfaceTraces:
    """Two-sided physical states and fluxes returned by the front closure."""

    speed: float
    pressurised_state: PressurisedState
    pressurised_flux: PressurisedFlux
    stratified_state: StratifiedState
    stratified_flux: StratifiedFlux

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed):
            raise ValueError("interface speed must be finite")


@dataclass(frozen=True)
class ALEInterfaceFlux:
    """Unique ALE interface flux used on both sides of the cut cell."""

    speed: float
    liquid_area: float
    liquid_momentum: float
    gas_mass: float
    gas_momentum: float
    liquid_area_residual: float
    liquid_momentum_residual: float
    gas_material_residual: float

    @classmethod
    def from_traces(
        cls,
        traces: InterfaceTraces,
        *,
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-10,
    ) -> "ALEInterfaceFlux":
        """Evaluate ``G=F-wU`` and enforce one liquid flux and material gas.

        For an exact RH solve the two liquid ALE fluxes are identical.  As in
        Eq. (28), their average is the unique conservative value after the
        bounded nonlinear residual has passed the stated tolerance.
        """

        w = float(traces.speed)
        up = traces.pressurised_state.vector()
        fp = traces.pressurised_flux.vector()
        us = traces.stratified_state.vector()
        fs = traces.stratified_flux.vector()

        gp = (fp[0] - w * up[0], fp[1] - w * up[1])
        gs_l = (fs[2] - w * us[2], fs[3] - w * us[3])
        residual_a = gp[0] - gs_l[0]
        residual_q = gp[1] - gs_l[1]
        if not _close_residual(
            residual_a,
            max(abs(gp[0]), abs(gs_l[0])),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ) or not _close_residual(
            residual_q,
            max(abs(gp[1]), abs(gs_l[1])),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            raise InterfaceClosureError(
                "liquid traces do not share one ALE Rankine--Hugoniot flux: "
                f"residual=({residual_a:.9g}, {residual_q:.9g})"
            )

        # The first physical gas flux is j_g.  Checking this identity prevents
        # a caller from satisfying u_g=w in the state while supplying a
        # contradictory gas-mass flux.
        flux_identity_residual = fs[0] - us[1]
        material_residual = us[1] - w * us[0]
        if not _close_residual(
            flux_identity_residual,
            max(abs(fs[0]), abs(us[1])),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            raise InterfaceClosureError(
                "gas mass flux is not the conserved gas momentum j_g"
            )
        if not _close_residual(
            material_residual,
            max(abs(us[1]), abs(w * us[0])),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            raise InterfaceClosureError(
                "gas trace violates the material condition u_g=w: "
                f"j_g-w*m_g={material_residual:.9g}"
            )

        return cls(
            speed=w,
            liquid_area=0.5 * (gp[0] + gs_l[0]),
            liquid_momentum=0.5 * (gp[1] + gs_l[1]),
            # This is exactly zero by the accepted material condition.  It is
            # not an epsilon or a numerical mask.
            gas_mass=0.0,
            gas_momentum=fs[1] - w * us[1],
            liquid_area_residual=residual_a,
            liquid_momentum_residual=residual_q,
            gas_material_residual=material_residual,
        )


@dataclass(frozen=True)
class OuterFaceFluxes:
    """Physical fluxes at the two branch-side faces of the current host.

    Both vectors use the global ``+x`` sign.  Which one is the left/right host
    face follows from ``pressurised_side``.
    """

    pressurised: PressurisedFlux
    stratified: StratifiedFlux


@dataclass(frozen=True)
class SubcellSources:
    """Distributed source vectors per unit subcell length."""

    pressurised_area: float = 0.0
    pressurised_momentum: float = 0.0
    stratified_gas_mass: float = 0.0
    stratified_gas_momentum: float = 0.0
    stratified_liquid_area: float = 0.0
    stratified_liquid_momentum: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(
            self.pressurised_area,
            self.pressurised_momentum,
            self.stratified_gas_mass,
            self.stratified_gas_momentum,
            self.stratified_liquid_area,
            self.stratified_liquid_momentum,
        ):
            raise ValueError("subcell sources must be finite")

    def pressurised_vector(self) -> tuple[float, float]:
        return (self.pressurised_area, self.pressurised_momentum)

    def stratified_vector(self) -> tuple[float, float, float, float]:
        return (
            self.stratified_gas_mass,
            self.stratified_gas_momentum,
            self.stratified_liquid_area,
            self.stratified_liquid_momentum,
        )


@dataclass(frozen=True)
class CutCellInventory:
    """Volume-integrated conserved inventory of one cut cell."""

    gas_mass: float
    gas_momentum: float
    liquid_area: float
    liquid_discharge: float

    def vector(self) -> tuple[float, float, float, float]:
        return (
            self.gas_mass,
            self.gas_momentum,
            self.liquid_area,
            self.liquid_discharge,
        )


@dataclass(frozen=True)
class MaterialFrontCutCell:
    """One host cell split at ``front_position`` into P and S subcells."""

    cell_faces: tuple[float, ...]
    host_index: int
    front_position: float
    pressurised_side: PressurisedSide
    pressurised: PressurisedState
    stratified: StratifiedState

    def __post_init__(self) -> None:
        if len(self.cell_faces) < 2:
            raise ValueError("cell_faces must contain at least two faces")
        if not all(math.isfinite(face) for face in self.cell_faces):
            raise ValueError("cell faces must be finite")
        if not all(
            self.cell_faces[index + 1] > self.cell_faces[index]
            for index in range(len(self.cell_faces) - 1)
        ):
            raise ValueError("cell faces must be strictly increasing")
        if not 0 <= self.host_index < len(self.cell_faces) - 1:
            raise ValueError("host index lies outside the mesh")
        if self.pressurised_side not in ("left", "right"):
            raise ValueError("pressurised_side must be 'left' or 'right'")
        if not math.isfinite(self.front_position):
            raise ValueError("front position must be finite")
        left, right = self.host_faces
        if not left <= self.front_position <= right:
            raise ValueError("front must lie inside the host cell")

    @property
    def host_faces(self) -> tuple[float, float]:
        return (
            float(self.cell_faces[self.host_index]),
            float(self.cell_faces[self.host_index + 1]),
        )

    @property
    def host_width(self) -> float:
        left, right = self.host_faces
        return right - left

    @property
    def pressurised_length(self) -> float:
        left, right = self.host_faces
        if self.pressurised_side == "left":
            return self.front_position - left
        return right - self.front_position

    @property
    def stratified_length(self) -> float:
        return self.host_width - self.pressurised_length

    def inventory(self) -> CutCellInventory:
        lp = self.pressurised_length
        ls = self.stratified_length
        return CutCellInventory(
            gas_mass=ls * self.stratified.gas_mass,
            gas_momentum=ls * self.stratified.gas_momentum,
            liquid_area=(
                lp * self.pressurised.area
                + ls * self.stratified.liquid_area
            ),
            liquid_discharge=(
                lp * self.pressurised.discharge
                + ls * self.stratified.liquid_discharge
            ),
        )


@dataclass(frozen=True)
class StageLedger:
    """Auditable balance for one no-crossing cut-cell substep."""

    dt: float
    initial: CutCellInventory
    final: CutCellInventory
    expected_change: CutCellInventory
    residual: CutCellInventory
    interface_flux: ALEInterfaceFlux


@dataclass(frozen=True)
class CompletedCell:
    """Old host after a face crossing has become a regular branch cell."""

    index: int
    branch: BranchKind
    pressurised: PressurisedState | None = None
    stratified: StratifiedState | None = None


@dataclass(frozen=True)
class CrossingRequest:
    """Data needed from the external branch solver at the exact event time."""

    event_time: float
    face_position: float
    old_host_index: int
    new_host_index: int
    moving_direction: int
    completed_branch: BranchKind
    existing_new_host_branch: BranchKind
    interface_traces: InterfaceTraces


@dataclass(frozen=True)
class CrossingEvent:
    """Exact one-face topology event and its inventory-neutral remap."""

    request: CrossingRequest
    completed_cell: CompletedCell
    zero_length_branch: BranchKind
    remap_residual: CutCellInventory


@dataclass(frozen=True)
class AdvanceResult:
    state: MaterialFrontCutCell
    elapsed: float
    ledgers: tuple[StageLedger, ...]
    crossings: tuple[CrossingEvent, ...]


class InterfaceProvider(Protocol):
    def __call__(
        self, state: MaterialFrontCutCell, time: float
    ) -> InterfaceTraces: ...


class OuterFluxProvider(Protocol):
    def __call__(
        self, state: MaterialFrontCutCell, time: float
    ) -> OuterFaceFluxes: ...


class SourceProvider(Protocol):
    def __call__(
        self, state: MaterialFrontCutCell, time: float
    ) -> SubcellSources: ...


NewHostState: TypeAlias = PressurisedState | StratifiedState
NewHostProvider: TypeAlias = Callable[[CrossingRequest], NewHostState]


def _difference(
    final: CutCellInventory, initial: CutCellInventory
) -> CutCellInventory:
    return CutCellInventory(
        gas_mass=final.gas_mass - initial.gas_mass,
        gas_momentum=final.gas_momentum - initial.gas_momentum,
        liquid_area=final.liquid_area - initial.liquid_area,
        liquid_discharge=final.liquid_discharge - initial.liquid_discharge,
    )


def _subtract(
    left: CutCellInventory, right: CutCellInventory
) -> CutCellInventory:
    return CutCellInventory(
        gas_mass=left.gas_mass - right.gas_mass,
        gas_momentum=left.gas_momentum - right.gas_momentum,
        liquid_area=left.liquid_area - right.liquid_area,
        liquid_discharge=left.liquid_discharge - right.liquid_discharge,
    )


def _add(
    left: CutCellInventory, right: CutCellInventory
) -> CutCellInventory:
    return CutCellInventory(
        gas_mass=left.gas_mass + right.gas_mass,
        gas_momentum=left.gas_momentum + right.gas_momentum,
        liquid_area=left.liquid_area + right.liquid_area,
        liquid_discharge=left.liquid_discharge + right.liquid_discharge,
    )


def _full_cell_inventory(
    branch: BranchKind,
    state: PressurisedState | StratifiedState,
    width: float,
) -> CutCellInventory:
    if branch == "pressurised":
        if not isinstance(state, PressurisedState):
            raise TypeError("pressurised full cell requires PressurisedState")
        return CutCellInventory(
            gas_mass=0.0,
            gas_momentum=0.0,
            liquid_area=width * state.area,
            liquid_discharge=width * state.discharge,
        )
    if not isinstance(state, StratifiedState):
        raise TypeError("stratified full cell requires StratifiedState")
    return CutCellInventory(
        gas_mass=width * state.gas_mass,
        gas_momentum=width * state.gas_momentum,
        liquid_area=width * state.liquid_area,
        liquid_discharge=width * state.liquid_discharge,
    )


def _state2_from_content(
    content: tuple[float, float],
    length: float,
    *,
    zero_trace: PressurisedState,
    tolerance: float,
) -> PressurisedState:
    if length > 0.0:
        return PressurisedState(content[0] / length, content[1] / length)
    if length != 0.0:
        raise MaterialFrontError("negative pressurised subcell length")
    if any(abs(value) > tolerance for value in content):
        raise VanishingSubcellInventoryError(
            "pressurised subcell reached zero length with finite inventory: "
            f"{content!r}"
        )
    return zero_trace


def _state4_from_content(
    content: tuple[float, float, float, float],
    length: float,
    *,
    zero_trace: StratifiedState,
    tolerance: float,
) -> StratifiedState:
    if length > 0.0:
        return StratifiedState(*(value / length for value in content))
    if length != 0.0:
        raise MaterialFrontError("negative stratified subcell length")
    if any(abs(value) > tolerance for value in content):
        raise VanishingSubcellInventoryError(
            "stratified subcell reached zero length with finite inventory: "
            f"{content!r}"
        )
    return zero_trace


def _advance_without_crossing(
    state: MaterialFrontCutCell,
    dt: float,
    traces: InterfaceTraces,
    outer_fluxes: OuterFaceFluxes,
    sources: SubcellSources,
    *,
    event_face: float | None = None,
    closure_absolute_tolerance: float,
    closure_relative_tolerance: float,
    vanishing_inventory_tolerance: float,
) -> tuple[MaterialFrontCutCell, StageLedger]:
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("substep dt must be finite and non-negative")

    ale = ALEInterfaceFlux.from_traces(
        traces,
        absolute_tolerance=closure_absolute_tolerance,
        relative_tolerance=closure_relative_tolerance,
    )
    initial = state.inventory()
    lp = state.pressurised_length
    ls = state.stratified_length
    cp = (
        lp * state.pressurised.area,
        lp * state.pressurised.discharge,
    )
    cs = (
        ls * state.stratified.gas_mass,
        ls * state.stratified.gas_momentum,
        ls * state.stratified.liquid_area,
        ls * state.stratified.liquid_discharge,
    )
    fp = outer_fluxes.pressurised.vector()
    fs = outer_fluxes.stratified.vector()
    sp = sources.pressurised_vector()
    ss = sources.stratified_vector()
    gi = (
        ale.gas_mass,
        ale.gas_momentum,
        ale.liquid_area,
        ale.liquid_momentum,
    )

    if state.pressurised_side == "left":
        dcp = (
            fp[0] - ale.liquid_area + lp * sp[0],
            fp[1] - ale.liquid_momentum + lp * sp[1],
        )
        dcs = tuple(
            gi[index] - fs[index] + ls * ss[index]
            for index in range(4)
        )
        expected_rate = CutCellInventory(
            gas_mass=-fs[0] + ls * ss[0],
            gas_momentum=ale.gas_momentum - fs[1] + ls * ss[1],
            liquid_area=(
                fp[0] - fs[2] + lp * sp[0] + ls * ss[2]
            ),
            liquid_discharge=(
                fp[1] - fs[3] + lp * sp[1] + ls * ss[3]
            ),
        )
    else:
        dcp = (
            ale.liquid_area - fp[0] + lp * sp[0],
            ale.liquid_momentum - fp[1] + lp * sp[1],
        )
        dcs = tuple(
            fs[index] - gi[index] + ls * ss[index]
            for index in range(4)
        )
        expected_rate = CutCellInventory(
            gas_mass=fs[0] + ls * ss[0],
            gas_momentum=fs[1] - ale.gas_momentum + ls * ss[1],
            liquid_area=(
                fs[2] - fp[0] + lp * sp[0] + ls * ss[2]
            ),
            liquid_discharge=(
                fs[3] - fp[1] + lp * sp[1] + ls * ss[3]
            ),
        )

    cp_new = tuple(cp[index] + dt * dcp[index] for index in range(2))
    cs_new = tuple(cs[index] + dt * dcs[index] for index in range(4))
    front_new = (
        float(event_face)
        if event_face is not None
        else state.front_position + traces.speed * dt
    )
    provisional = MaterialFrontCutCell(
        cell_faces=state.cell_faces,
        host_index=state.host_index,
        front_position=front_new,
        pressurised_side=state.pressurised_side,
        pressurised=state.pressurised,
        stratified=state.stratified,
    )
    pressurised_new = _state2_from_content(
        cp_new,
        provisional.pressurised_length,
        zero_trace=traces.pressurised_state,
        tolerance=vanishing_inventory_tolerance,
    )
    stratified_new = _state4_from_content(
        cs_new,
        provisional.stratified_length,
        zero_trace=traces.stratified_state,
        tolerance=vanishing_inventory_tolerance,
    )
    final_state = MaterialFrontCutCell(
        cell_faces=state.cell_faces,
        host_index=state.host_index,
        front_position=front_new,
        pressurised_side=state.pressurised_side,
        pressurised=pressurised_new,
        stratified=stratified_new,
    )
    final = final_state.inventory()
    expected = CutCellInventory(
        gas_mass=dt * expected_rate.gas_mass,
        gas_momentum=dt * expected_rate.gas_momentum,
        liquid_area=dt * expected_rate.liquid_area,
        liquid_discharge=dt * expected_rate.liquid_discharge,
    )
    residual = _subtract(_difference(final, initial), expected)
    return final_state, StageLedger(
        dt=dt,
        initial=initial,
        final=final,
        expected_change=expected,
        residual=residual,
        interface_flux=ale,
    )


def _next_crossing(
    state: MaterialFrontCutCell, speed: float, dt: float
) -> tuple[float, float, int] | None:
    if speed == 0.0:
        return None
    left, right = state.host_faces
    if speed > 0.0:
        face = right
        direction = 1
    else:
        face = left
        direction = -1
    crossing_time = (face - state.front_position) / speed
    if crossing_time < 0.0:
        raise MaterialFrontError("front speed points away from its selected face")
    if crossing_time <= dt:
        return (crossing_time, face, direction)
    return None


def _remap_at_crossing(
    crossed_state: MaterialFrontCutCell,
    traces: InterfaceTraces,
    *,
    event_time: float,
    face: float,
    direction: int,
    new_host_provider: NewHostProvider,
) -> tuple[MaterialFrontCutCell, CrossingEvent]:
    side_sign = 1 if crossed_state.pressurised_side == "left" else -1
    pressurised_expands = side_sign * traces.speed > 0.0
    completed_branch: BranchKind = (
        "pressurised" if pressurised_expands else "stratified"
    )
    existing_branch: BranchKind = (
        "stratified" if pressurised_expands else "pressurised"
    )
    new_index = crossed_state.host_index + direction
    if not 0 <= new_index < len(crossed_state.cell_faces) - 1:
        raise DomainBoundaryCrossing(
            f"front crossed mesh boundary from cell {crossed_state.host_index}"
        )

    request = CrossingRequest(
        event_time=float(event_time),
        face_position=float(face),
        old_host_index=crossed_state.host_index,
        new_host_index=new_index,
        moving_direction=direction,
        completed_branch=completed_branch,
        existing_new_host_branch=existing_branch,
        interface_traces=traces,
    )
    existing_state = new_host_provider(request)
    if existing_branch == "stratified":
        if not isinstance(existing_state, StratifiedState):
            raise TypeError("new host provider must return a stratified state")
        pressurised = traces.pressurised_state
        stratified = existing_state
        completed = CompletedCell(
            index=crossed_state.host_index,
            branch="pressurised",
            pressurised=crossed_state.pressurised,
        )
        zero_branch: BranchKind = "pressurised"
    else:
        if not isinstance(existing_state, PressurisedState):
            raise TypeError("new host provider must return a pressurised state")
        pressurised = existing_state
        stratified = traces.stratified_state
        completed = CompletedCell(
            index=crossed_state.host_index,
            branch="stratified",
            stratified=crossed_state.stratified,
        )
        zero_branch = "stratified"

    new_host = MaterialFrontCutCell(
        cell_faces=crossed_state.cell_faces,
        host_index=new_index,
        front_position=face,
        pressurised_side=crossed_state.pressurised_side,
        pressurised=pressurised,
        stratified=stratified,
    )
    new_width = (
        crossed_state.cell_faces[new_index + 1]
        - crossed_state.cell_faces[new_index]
    )
    old_width = crossed_state.host_width
    before_remap = _add(
        crossed_state.inventory(),
        _full_cell_inventory(existing_branch, existing_state, new_width),
    )
    completed_state: PressurisedState | StratifiedState
    if completed_branch == "pressurised":
        completed_state = crossed_state.pressurised
    else:
        completed_state = crossed_state.stratified
    after_remap = _add(
        _full_cell_inventory(completed_branch, completed_state, old_width),
        new_host.inventory(),
    )
    event = CrossingEvent(
        request=request,
        completed_cell=completed,
        zero_length_branch=zero_branch,
        # The remap only changes ownership.  The emerging trace has exactly
        # zero length and the old disappearing trace has already reached zero.
        remap_residual=_difference(after_remap, before_remap),
    )
    return new_host, event


def advance_material_front_cutcell(
    state: MaterialFrontCutCell,
    dt: float,
    *,
    interface_provider: InterfaceProvider,
    outer_flux_provider: OuterFluxProvider,
    new_host_provider: NewHostProvider | None = None,
    source_provider: SourceProvider | None = None,
    time: float = 0.0,
    closure_absolute_tolerance: float = 1.0e-12,
    closure_relative_tolerance: float = 1.0e-10,
    vanishing_inventory_tolerance: float = 1.0e-11,
) -> AdvanceResult:
    """Advance one local ALE stage with at most one exact face crossing.

    The interface and outer-face providers are reevaluated after a crossing.
    That makes the routine suitable as the geometry-aware stage kernel inside
    either Euler or each stage of SSP--RK2.  It does not itself update regular
    branch cells; ``new_host_provider`` must therefore return the adjacent
    branch cell state already advanced to the exact event time.
    """

    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if closure_absolute_tolerance < 0.0 or closure_relative_tolerance < 0.0:
        raise ValueError("closure tolerances must be non-negative")
    if vanishing_inventory_tolerance < 0.0:
        raise ValueError("inventory tolerance must be non-negative")

    if source_provider is None:
        source_provider = lambda _state, _time: SubcellSources()

    current = state
    current_time = float(time)
    remaining = float(dt)
    ledgers: list[StageLedger] = []
    events: list[CrossingEvent] = []

    traces = interface_provider(current, current_time)
    crossing = _next_crossing(current, traces.speed, remaining)
    if crossing is None:
        current, ledger = _advance_without_crossing(
            current,
            remaining,
            traces,
            outer_flux_provider(current, current_time),
            source_provider(current, current_time),
            closure_absolute_tolerance=closure_absolute_tolerance,
            closure_relative_tolerance=closure_relative_tolerance,
            vanishing_inventory_tolerance=vanishing_inventory_tolerance,
        )
        ledgers.append(ledger)
        return AdvanceResult(current, dt, tuple(ledgers), tuple(events))

    if new_host_provider is None:
        raise MaterialFrontError(
            "a face crossing requires the adjacent branch state at event time"
        )
    crossing_dt, face, direction = crossing
    crossed, ledger = _advance_without_crossing(
        current,
        crossing_dt,
        traces,
        outer_flux_provider(current, current_time),
        source_provider(current, current_time),
        event_face=face,
        closure_absolute_tolerance=closure_absolute_tolerance,
        closure_relative_tolerance=closure_relative_tolerance,
        vanishing_inventory_tolerance=vanishing_inventory_tolerance,
    )
    ledgers.append(ledger)
    current_time += crossing_dt
    remaining -= crossing_dt
    current, event = _remap_at_crossing(
        crossed,
        traces,
        event_time=current_time,
        face=face,
        direction=direction,
        new_host_provider=new_host_provider,
    )
    events.append(event)

    if remaining > 0.0:
        traces = interface_provider(current, current_time)
        if _next_crossing(current, traces.speed, remaining) is not None:
            raise MultipleCrossingsError(
                "accepted material-front stage contains more than one face crossing"
            )
        current, ledger = _advance_without_crossing(
            current,
            remaining,
            traces,
            outer_flux_provider(current, current_time),
            source_provider(current, current_time),
            closure_absolute_tolerance=closure_absolute_tolerance,
            closure_relative_tolerance=closure_relative_tolerance,
            vanishing_inventory_tolerance=vanishing_inventory_tolerance,
        )
        ledgers.append(ledger)

    return AdvanceResult(current, dt, tuple(ledgers), tuple(events))


__all__ = [
    "ALEInterfaceFlux",
    "AdvanceResult",
    "CompletedCell",
    "CrossingEvent",
    "CrossingRequest",
    "CutCellInventory",
    "DomainBoundaryCrossing",
    "InterfaceClosureError",
    "InterfaceTraces",
    "MaterialFrontCutCell",
    "MaterialFrontError",
    "MultipleCrossingsError",
    "OuterFaceFluxes",
    "PressurisedFlux",
    "PressurisedState",
    "StageLedger",
    "StratifiedFlux",
    "StratifiedState",
    "SubcellSources",
    "VanishingSubcellInventoryError",
    "advance_material_front_cutcell",
]
