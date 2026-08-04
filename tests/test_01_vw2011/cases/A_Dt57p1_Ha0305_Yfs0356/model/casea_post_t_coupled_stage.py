"""One-owner, fully coupled post-T SSP-RK2 stage for Case A.

This module is the time-integration boundary between the pure liquid graph
operator and the resolved gas graph operator.  Every Runge--Kutta stage is
evaluated from one simultaneous state containing the four liquid fields, the
five gas/tracer fields, and two independent material-front positions.  The
predictor is therefore not allowed to freeze gas while advancing liquid (or
vice versa), and the corrector recomputes both spatial operators and both
front speeds.

There is exactly one gas owner: ``Mgt, Jgt, Mgr, Jgrs, Mgrs``.  No lumped
pocket, receiver cell, remap, positivity floor, prescribed wave, or filtered
display field exists here.  If a trial Runge--Kutta state leaves the physical
invariant set, the step is rejected with ``FloatingPointError`` so the caller
can reduce ``dt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from casea_coupled_gas_network import CoupledGasParameters
from casea_post_t_liquid_stage import (
    PostTLiquidGeometry,
    PostTLiquidStageRhs,
    PressureCallback,
    post_t_liquid_stage_rhs,
)
from casea_post_t_sideport_liquid_stage import (
    PostTSidePortGeometry,
    PostTSidePortLiquidStageRhs,
    post_t_sideport_liquid_stage_rhs,
)
from casea_resolved_gas_stage import (
    ResolvedGasStageRHS,
    evaluate_resolved_gas_stage_rhs,
)
from casea_topology_event import BranchFrontTopology


Array = np.ndarray


def _readonly_state_array(value: object, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PostTCoupledState:
    """Complete conservative state owned by the post-T graph integrator."""

    Alt: Array = field(repr=False)
    Qlt: Array = field(repr=False)
    Alr: Array = field(repr=False)
    Qlr: Array = field(repr=False)
    Mgt: Array = field(repr=False)
    Jgt: Array = field(repr=False)
    Mgr: Array = field(repr=False)
    Jgrs: Array = field(repr=False)
    Mgrs: Array = field(repr=False)
    east_front: BranchFrontTopology
    vertical_front: BranchFrontTopology

    def __post_init__(self) -> None:
        for name in (
            "Alt", "Qlt", "Alr", "Qlr", "Mgt", "Jgt", "Mgr", "Jgrs",
            "Mgrs",
        ):
            object.__setattr__(
                self, name, _readonly_state_array(getattr(self, name), name)
            )
        if not (
            self.Alt.shape == self.Qlt.shape == self.Mgt.shape == self.Jgt.shape
        ):
            raise ValueError("horizontal liquid and gas fields must share a grid")
        if not (
            self.Alr.shape == self.Qlr.shape == self.Mgr.shape
            == self.Jgrs.shape == self.Mgrs.shape
        ):
            raise ValueError("vertical liquid and gas fields must share a grid")
        if np.any(self.Alt < 0.0) or np.any(self.Alr < 0.0):
            raise ValueError("liquid area cannot be negative")
        if np.any(self.Mgt < 0.0) or np.any(self.Mgr < 0.0):
            raise ValueError("gas mass cannot be negative")
        if np.any(self.Mgrs < 0.0) or np.any(self.Mgrs > self.Mgr + 1.0e-13):
            raise ValueError("vertical tracer must lie between zero and gas mass")
        if self.east_front.branch != "east":
            raise ValueError("east_front must own the east branch")
        if self.vertical_front.branch != "vertical":
            raise ValueError("vertical_front must own the vertical branch")


@dataclass(frozen=True)
class PostTCoupledGeometry:
    """Geometry shared by the liquid and gas stage evaluators."""

    liquid: PostTLiquidGeometry | PostTSidePortGeometry
    gas: CoupledGasParameters
    gas_junction_index: int
    vertical_liquid_active_count: int

    def __post_init__(self) -> None:
        index = int(self.gas_junction_index)
        count = int(self.vertical_liquid_active_count)
        if index != self.gas_junction_index or index < 0:
            raise ValueError("gas_junction_index must be a non-negative integer")
        if count != self.vertical_liquid_active_count or count < 1:
            raise ValueError("vertical_liquid_active_count must be positive")


@dataclass(frozen=True)
class PostTCoupledStageRHS:
    """One pure simultaneous liquid/gas/front residual evaluation."""

    dAlt_dt: Array
    dQlt_dt: Array
    dAlr_dt: Array
    dQlr_dt: Array
    dMgt_dt: Array
    dJgt_dt: Array
    dMgr_dt: Array
    dJgrs_dt: Array
    dMgrs_dt: Array
    east_front_speed: float
    vertical_front_speed: float
    liquid: PostTLiquidStageRhs | PostTSidePortLiquidStageRhs
    gas: ResolvedGasStageRHS
    liquid_t_volume_residual: float
    gas_t_mass_residual: float
    interphase_momentum_residual: float


@dataclass(frozen=True)
class PostTCoupledAdvance:
    """Accepted SSP-RK2 state and conservative Heun ledgers."""

    state: PostTCoupledState
    first_stage: PostTCoupledStageRHS
    second_stage: PostTCoupledStageRHS
    atmospheric_mass_exchange: float
    escaped_tracer_mass: float
    liquid_volume_change: float
    liquid_volume_error: float
    gas_mass_change: float
    gas_mass_error: float
    tracer_mass_change: float
    tracer_mass_error: float
    t_liquid_volume_residual_integral: float
    t_gas_mass_residual_integral: float
    interphase_momentum_residual_integral: float


def _readonly_rhs(value: object) -> Array:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def evaluate_post_t_coupled_stage_rhs(
    state: PostTCoupledState,
    *,
    geometry: PostTCoupledGeometry,
    pressure_callback: PressureCallback,
    top_open: bool,
) -> PostTCoupledStageRHS:
    """Evaluate liquid, gas, drag, and front RHS from the same stage state."""

    if not isinstance(state, PostTCoupledState):
        raise TypeError("state must be PostTCoupledState")
    if not isinstance(top_open, (bool, np.bool_)):
        raise TypeError("top_open must be boolean")
    if geometry.gas_junction_index >= state.Alt.size - 1:
        raise ValueError("gas junction must precede an east-branch cell")
    if geometry.vertical_liquid_active_count > state.Alr.size:
        raise ValueError("vertical liquid active prefix exceeds its grid")
    if np.any(state.Alt <= 0.0):
        raise ValueError("all horizontal liquid cells must remain positive")
    active_count = geometry.vertical_liquid_active_count
    if np.any(state.Alr[:active_count] <= 0.0):
        raise ValueError("active vertical liquid cells must remain positive")
    if np.any(state.Alr[active_count:] != 0.0):
        raise ValueError("inactive vertical liquid suffix must remain exactly dry")

    if isinstance(geometry.liquid, PostTSidePortGeometry):
        liquid = post_t_sideport_liquid_stage_rhs(
            state.Alt,
            state.Qlt,
            state.Alr,
            state.Qlr,
            state.Mgt,
            state.Jgt,
            state.Mgr,
            state.Jgrs,
            geometry=geometry.liquid,
            pressure_callback=pressure_callback,
            vertical_active_count=active_count,
        )
        liquid_volume_residual = float(
            liquid.diagnostics.liquid_volume_residual
        )
    else:
        liquid = post_t_liquid_stage_rhs(
            state.Alt,
            state.Qlt,
            state.Alr,
            state.Qlr,
            state.Mgt,
            state.Jgt,
            state.Mgr,
            state.Jgrs,
            geometry=geometry.liquid,
            pressure_callback=pressure_callback,
            vertical_active_count=active_count,
        )
        liquid_volume_residual = float(
            liquid.diagnostics.node_volume_residual
        )
    gas = evaluate_resolved_gas_stage_rhs(
        state.Mgt,
        state.Jgt,
        state.Mgr,
        state.Jgrs,
        state.Mgrs,
        state.Alt,
        state.Alr,
        state.Qlt,
        state.Qlr,
        dx=geometry.liquid.horizontal_cell_width,
        dz=geometry.liquid.vertical_cell_width,
        junction_index=geometry.gas_junction_index,
        params=geometry.gas,
        east_front=state.east_front,
        vertical_front=state.vertical_front,
        top_open=bool(top_open),
    )

    # Interphase drag appears once in each phase with opposite momentum.  It
    # is added here, not in the liquid graph evaluator, so both contributions
    # use precisely the same gas/liquid stage state.
    d_qh = liquid.rhs_horizontal_discharge + gas.dQlt_drag_dt
    d_qv = liquid.rhs_vertical_discharge + gas.dQlr_drag_dt
    interphase_residual = float(
        gas.horizontal_interphase_momentum_residual
        + gas.vertical_interphase_momentum_residual
    )
    return PostTCoupledStageRHS(
        dAlt_dt=_readonly_rhs(liquid.rhs_horizontal_area),
        dQlt_dt=_readonly_rhs(d_qh),
        dAlr_dt=_readonly_rhs(liquid.rhs_vertical_area),
        dQlr_dt=_readonly_rhs(d_qv),
        dMgt_dt=_readonly_rhs(gas.dMgt_dt),
        dJgt_dt=_readonly_rhs(gas.dJgt_dt),
        dMgr_dt=_readonly_rhs(gas.dMgr_dt),
        dJgrs_dt=_readonly_rhs(gas.dJgrs_dt),
        dMgrs_dt=_readonly_rhs(gas.dMgrs_dt),
        east_front_speed=float(gas.east_front.speed),
        vertical_front_speed=float(gas.vertical_front.speed),
        liquid=liquid,
        gas=gas,
        liquid_t_volume_residual=liquid_volume_residual,
        gas_t_mass_residual=float(gas.t_flux.internal_mass_residual),
        interphase_momentum_residual=interphase_residual,
    )


def _front_predictor(
    front: BranchFrontTopology,
    speed: float,
    dt: float,
) -> BranchFrontTopology:
    position = front.position + dt * speed
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, front.position, abs(dt * speed)
    )
    if position < -tolerance:
        raise FloatingPointError(
            f"{front.branch} material front crossed behind the T; reduce dt"
        )
    # Only roundoff at the graph origin is removed.  A finite negative trial
    # is rejected above rather than projected or remapped.
    if position < 0.0:
        position = 0.0
    return front.at(position)


def _trial_state(
    base: PostTCoupledState,
    rhs: PostTCoupledStageRHS,
    dt: float,
) -> PostTCoupledState:
    return PostTCoupledState(
        Alt=base.Alt + dt * rhs.dAlt_dt,
        Qlt=base.Qlt + dt * rhs.dQlt_dt,
        Alr=base.Alr + dt * rhs.dAlr_dt,
        Qlr=base.Qlr + dt * rhs.dQlr_dt,
        Mgt=base.Mgt + dt * rhs.dMgt_dt,
        Jgt=base.Jgt + dt * rhs.dJgt_dt,
        Mgr=base.Mgr + dt * rhs.dMgr_dt,
        Jgrs=base.Jgrs + dt * rhs.dJgrs_dt,
        Mgrs=base.Mgrs + dt * rhs.dMgrs_dt,
        east_front=_front_predictor(
            base.east_front, rhs.east_front_speed, dt
        ),
        vertical_front=_front_predictor(
            base.vertical_front, rhs.vertical_front_speed, dt
        ),
    )


def _heun_state(
    initial: PostTCoupledState,
    predictor: PostTCoupledState,
    second: PostTCoupledStageRHS,
    dt: float,
) -> PostTCoupledState:
    def combine(name: str) -> Array:
        return 0.5 * getattr(initial, name) + 0.5 * (
            getattr(predictor, name) + dt * getattr(second, f"d{name}_dt")
        )

    def combine_front(
        original: BranchFrontTopology,
        predicted: BranchFrontTopology,
        speed: float,
    ) -> BranchFrontTopology:
        position = 0.5 * original.position + 0.5 * (
            predicted.position + dt * speed
        )
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, original.position, predicted.position, abs(dt * speed)
        )
        if position < -tolerance:
            raise FloatingPointError(
                f"{original.branch} SSP-RK2 front crossed behind the T"
            )
        return original.at(max(position, 0.0))

    return PostTCoupledState(
        Alt=combine("Alt"),
        Qlt=combine("Qlt"),
        Alr=combine("Alr"),
        Qlr=combine("Qlr"),
        Mgt=combine("Mgt"),
        Jgt=combine("Jgt"),
        Mgr=combine("Mgr"),
        Jgrs=combine("Jgrs"),
        Mgrs=combine("Mgrs"),
        east_front=combine_front(
            initial.east_front,
            predictor.east_front,
            second.east_front_speed,
        ),
        vertical_front=combine_front(
            initial.vertical_front,
            predictor.vertical_front,
            second.vertical_front_speed,
        ),
    )


def advance_post_t_coupled_ssprk2(
    state: PostTCoupledState,
    *,
    dt: float,
    geometry: PostTCoupledGeometry,
    pressure_callback: PressureCallback,
    top_open: bool,
) -> PostTCoupledAdvance:
    """Advance every post-T conserved field and both fronts with SSP-RK2."""

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    first = evaluate_post_t_coupled_stage_rhs(
        state,
        geometry=geometry,
        pressure_callback=pressure_callback,
        top_open=top_open,
    )
    try:
        predictor = _trial_state(state, first, dt)
    except ValueError as error:
        raise FloatingPointError(
            "post-T SSP-RK2 predictor left the conservative invariant set"
        ) from error
    second = evaluate_post_t_coupled_stage_rhs(
        predictor,
        geometry=geometry,
        pressure_callback=pressure_callback,
        top_open=top_open,
    )
    try:
        final = _heun_state(state, predictor, second, dt)
    except ValueError as error:
        raise FloatingPointError(
            "post-T SSP-RK2 corrector left the conservative invariant set"
        ) from error

    top_mass = 0.5 * dt * (
        first.gas.top_flux.mass_rate + second.gas.top_flux.mass_rate
    )
    top_tracer = 0.5 * dt * (
        first.gas.top_flux.tracer_mass_rate
        + second.gas.top_flux.tracer_mass_rate
    )
    liquid_initial = float(
        np.sum(state.Alt, dtype=np.float64)
        * geometry.liquid.horizontal_cell_width
        + np.sum(state.Alr, dtype=np.float64)
        * geometry.liquid.vertical_cell_width
    )
    liquid_final = float(
        np.sum(final.Alt, dtype=np.float64)
        * geometry.liquid.horizontal_cell_width
        + np.sum(final.Alr, dtype=np.float64)
        * geometry.liquid.vertical_cell_width
    )
    gas_initial = float(
        np.sum(state.Mgt, dtype=np.float64)
        + np.sum(state.Mgr, dtype=np.float64)
    )
    gas_final = float(
        np.sum(final.Mgt, dtype=np.float64)
        + np.sum(final.Mgr, dtype=np.float64)
    )
    tracer_initial = float(
        np.sum(state.Mgt, dtype=np.float64)
        + np.sum(state.Mgrs, dtype=np.float64)
    )
    tracer_final = float(
        np.sum(final.Mgt, dtype=np.float64)
        + np.sum(final.Mgrs, dtype=np.float64)
    )
    t_liquid = 0.5 * dt * (
        first.liquid_t_volume_residual + second.liquid_t_volume_residual
    )
    t_gas = 0.5 * dt * (
        first.gas_t_mass_residual + second.gas_t_mass_residual
    )
    drag_residual = 0.5 * dt * (
        first.interphase_momentum_residual
        + second.interphase_momentum_residual
    )
    liquid_change = liquid_final - liquid_initial
    gas_change = gas_final - gas_initial
    tracer_change = tracer_final - tracer_initial
    return PostTCoupledAdvance(
        state=final,
        first_stage=first,
        second_stage=second,
        atmospheric_mass_exchange=float(top_mass),
        escaped_tracer_mass=float(top_tracer),
        liquid_volume_change=liquid_change,
        liquid_volume_error=liquid_change,
        gas_mass_change=gas_change,
        gas_mass_error=gas_change + top_mass,
        tracer_mass_change=tracer_change,
        tracer_mass_error=tracer_change + top_tracer,
        t_liquid_volume_residual_integral=t_liquid,
        t_gas_mass_residual_integral=t_gas,
        interphase_momentum_residual_integral=drag_residual,
    )


__all__ = [
    "PostTCoupledAdvance",
    "PostTCoupledGeometry",
    "PostTCoupledStageRHS",
    "PostTCoupledState",
    "advance_post_t_coupled_ssprk2",
    "evaluate_post_t_coupled_stage_rhs",
]
