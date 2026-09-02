#!/usr/bin/env python3
"""Persistent single-run Campaign-2 one-dimensional coupling candidate.

The horizontal state is owned and advanced by the mirrored Campaign-1 Tosan
shock-fit solver for the entire physical run.  A fixed-head characteristic at
physical ``x=0`` supplies the published 0.66 m reservoir condition.  The
vertical state is owned by :mod:`campaign2_vertical_twofluid_kernel` and starts
with the published 0.61 m column measured above the horizontal-pipe crown.

At every physical step this driver

1. advances the one and only horizontal state (including the shared 0.20 s
   sine-squared valve law and fixed-head reservoir transaction),
2. reads the two horizontal traces adjacent to the side T and the vertical
   bottom trace,
3. solves the shared liquid and gas T nodes exactly once,
4. passes the resulting *same* :class:`TeeTransaction` to both state owners,
   and
5. books only the reservoir and open riser rim as external boundaries.

No case identifier or measured outcome enters a solver call.  The riser
diameter is the sole physical argument that differs between single-run
instances.  Ejection is diagnosed only from the time-integrated positive
liquid flux through the physical top face.

This file intentionally exposes a smoke runner capped at 0.05 s.  The vertical
kernel itself declares that the complete Campaign-2 constitutive closure is
not yet ready, so this candidate must not be promoted to qualification or
manuscript evidence merely because the ownership and conservation smoke tests
pass.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from campaign2_global_budget import Campaign2GlobalBudget
from campaign2_shared_contract import APPARATUS, SHARED_CLOSURE
from campaign2_tee_riemann import (
    GasTeeSolution,
    GasTrace,
    LiquidBranchTrace,
    LiquidTeeSolution,
    FirstBottomGasEntrySolution,
    solve_gas_tee,
    solve_first_bottom_gas_entry,
    solve_liquid_tee,
    solve_liquid_tee_with_blocked_riser,
)
from campaign2_vertical_twofluid_kernel import (
    COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY,
    MISSING_PHYSICAL_CLOSURES,
    AtmosphericTopBoundary,
    Campaign2VerticalTwoFluidKernel,
    VerticalTwoFluidParameters,
    VerticalTwoFluidState,
    canonicalize_upper_free_surface_roundoff,
    hydrostatic_column_state,
    isothermal_common_pressure_faces,
    lower_material_front_geometric_timestep_limit,
)
from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal
from case1_persistent_coupling import (
    PersistentHorizontalOwner,
    TeeTransaction,
    VerticalTeeIncrement,
    transaction_from_tee_solutions,
)


HERE = Path(__file__).resolve().parent
SMOKE_END_LIMIT_S = 0.05
GAS_HEAT_CAPACITY_RATIO = 1.4
_TEE_EQUILIBRIUM_ROUNDOFF_FACTOR = 4096.0


class CouplingClosureUnavailable(RuntimeError):
    """A phase trace cannot be represented by the current shared closures."""


class CoupledTimestepLimitExceeded(ValueError):
    """A requested physical step is larger than the current shared CFL limit."""

    def __init__(self, *, requested_dt_s: float, stable_dt_s: float) -> None:
        self.requested_dt_s = float(requested_dt_s)
        self.stable_dt_s = float(stable_dt_s)
        super().__init__(
            "requested dt exceeds the current coupled CFL/event limit: "
            f"{self.requested_dt_s:.17g} s > {self.stable_dt_s:.17g} s"
        )


@dataclass(frozen=True)
class CandidateNumerics:
    """One shared numerical contract; none of these values is case-dependent."""

    horizontal_dx_m: float = 0.010
    vertical_dz_m: float = 0.010
    max_dt_s: float = 2.0e-4
    shared_cfl: float = 0.45

    def __post_init__(self) -> None:
        values = (
            self.horizontal_dx_m,
            self.vertical_dz_m,
            self.max_dt_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all candidate numerical spacings must be positive")
        if not math.isfinite(self.shared_cfl) or not (
            0.0 < self.shared_cfl <= 0.45
        ):
            raise ValueError("shared_cfl must lie in (0, 0.45]")

    @property
    def physical_step_s(self) -> float:
        """Backward-compatible name for the shared maximum physical step."""

        return self.max_dt_s


SHARED_NUMERICS = CandidateNumerics()


@dataclass(frozen=True)
class TeeSolveResult:
    """The one shared T-node solution used by both phase owners."""

    transaction: TeeTransaction
    liquid: LiquidTeeSolution
    gas: GasTeeSolution
    gas_open_area_m2: float
    horizontal_west_liquid: LiquidBranchTrace
    horizontal_east_liquid: LiquidBranchTrace
    vertical_bottom_liquid: LiquidBranchTrace
    horizontal_gas: GasTrace
    vertical_bottom_gas: GasTrace
    first_bottom_gas_entry: FirstBottomGasEntrySolution | None = None
    finite_bottom_gas_pocket: bool = False


@dataclass(frozen=True)
class CoupledStepRecord:
    """Auditable result of one completed physical step."""

    start_time_s: float
    end_time_s: float
    transaction: TeeTransaction
    gas_open_area_m2: float
    vertical_increment: VerticalTeeIncrement
    reservoir_liquid_exchange_m3: float
    top_liquid_outflow_m3: float
    top_gas_net_outflow_kg: float
    horizontal_tee_liquid_residual_m3: float
    horizontal_tee_gas_residual_kg: float
    vertical_tee_liquid_residual_m3: float
    vertical_tee_gas_residual_kg: float
    global_liquid_residual_m3: float
    global_gas_residual_kg: float


@dataclass(frozen=True)
class _MutableAttributesSnapshot:
    """Deep snapshot of one owner's non-structural instance attributes.

    Solver, kernel, and owner identities are intentionally retained across a
    rollback.  Everything that those owners mutate during a physical step is
    copied and restored in place, including attributes added while a failing
    step is in progress.  Keeping this helper generic prevents a newly added
    cumulative ledger from silently falling outside the transaction.
    """

    target: Any
    structural_attributes: frozenset[str]
    values: dict[str, Any]

    @classmethod
    def capture(
        cls,
        target: Any,
        *,
        structural_attributes: tuple[str, ...] = (),
    ) -> "_MutableAttributesSnapshot":
        structural = frozenset(structural_attributes)
        values = {
            name: copy.deepcopy(value)
            for name, value in vars(target).items()
            if name not in structural
        }
        return cls(
            target=target,
            structural_attributes=structural,
            values=values,
        )

    def restore(self) -> None:
        current_mutable_names = {
            name
            for name in vars(self.target)
            if name not in self.structural_attributes
        }
        for name in current_mutable_names.difference(self.values):
            delattr(self.target, name)
        for name, value in self.values.items():
            setattr(self.target, name, copy.deepcopy(value))


@dataclass(frozen=True)
class _CoupledStepSnapshot:
    """All mutable owners participating in one coupled physical step."""

    owner_snapshots: tuple[_MutableAttributesSnapshot, ...]

    @classmethod
    def capture(
        cls,
        candidate: "PersistentCampaign2Candidate",
    ) -> "_CoupledStepSnapshot":
        horizontal = candidate.horizontal_owner
        snapshots = [
            _MutableAttributesSnapshot.capture(
                candidate,
                structural_attributes=(
                    "riser_diameter_m",
                    "numerics",
                    "horizontal_owner",
                    "vertical_kernel",
                    "global_budget",
                ),
            ),
            _MutableAttributesSnapshot.capture(
                horizontal,
                structural_attributes=("solver", "reservoir_boundary"),
            ),
            _MutableAttributesSnapshot.capture(candidate.global_budget),
        ]
        if horizontal.reservoir_boundary is not None:
            snapshots.append(
                _MutableAttributesSnapshot.capture(
                    horizontal.reservoir_boundary,
                    structural_attributes=("solver",),
                )
            )
        return cls(owner_snapshots=tuple(snapshots))

    def restore(self) -> None:
        # Restore in reverse ownership order, analogous to unwinding a stack.
        for snapshot in reversed(self.owner_snapshots):
            snapshot.restore()


def _liquid_volume_horizontal(owner: PersistentHorizontalOwner) -> float:
    return float(
        np.sum(np.asarray(owner.state.area, dtype=float))
        * owner.solver.dx
    )


def _gas_mass_horizontal(owner: PersistentHorizontalOwner) -> float:
    return float(owner.state.gas.mass)


def _liquid_volume_vertical(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> float:
    dz = float(parameters.cell_length_m)
    return float(math.fsum(al * dz for al in state.Al))


def _gas_mass_vertical(state: VerticalTwoFluidState) -> float:
    return float(math.fsum(state.Mg))


def _liquid_tee_is_roundoff_equilibrium(
    branches: tuple[LiquidBranchTrace, LiquidBranchTrace, LiquidBranchTrace],
) -> bool:
    """Recognise only a binary64-scale quiescent three-branch T state.

    Each gauge pressure is reconstructed through a different horizontal or
    vertical path.  Near the exact hydrostatic fixed point those paths can
    differ by roughly 1e-9 Pa, and the corresponding characteristic flows are
    differences of O(1e-4) m3/s terms.  The tolerance is therefore tied to
    machine epsilon and the dimensional pressure/acoustic-velocity scales; it
    is not a physical deadband or a case-dependent flow threshold.
    """

    pressures = tuple(float(branch.gauge_pressure_Pa) for branch in branches)
    velocities = tuple(float(branch.outward_velocity_m_s) for branch in branches)
    pressure_scale = max(*(abs(value) for value in pressures), 1.0)
    velocity_scale = max(
        *(abs(value) for value in velocities),
        *(float(branch.wave_speed_m_s) for branch in branches),
        1.0,
    )
    epsilon = math.ulp(1.0)
    pressure_roundoff = (
        _TEE_EQUILIBRIUM_ROUNDOFF_FACTOR * epsilon * pressure_scale
    )
    velocity_roundoff = (
        _TEE_EQUILIBRIUM_ROUNDOFF_FACTOR * epsilon * velocity_scale
    )
    return bool(
        max(pressures) - min(pressures) <= pressure_roundoff
        and max(abs(value) for value in velocities) <= velocity_roundoff
    )


def _pin_liquid_tee_roundoff_flows(
    branches: tuple[LiquidBranchTrace, LiquidBranchTrace, LiquidBranchTrace],
    solution: LiquidTeeSolution,
) -> LiquidTeeSolution:
    """Pin only unresolved characteristic cancellation to exact zero.

    The returned branch fluxes subtract incoming and pressure-correction terms
    of order 1e-4 m3/s.  Reconstructed branch traces can carry a few hundred
    binary64 ulps before this node solve.  If *every* branch result lies within
    4096 ulps of the dimensional terms that formed it, the only conservative
    fixed point is the exact three-zero transaction.  Any one resolved branch
    leaves the full solution untouched.
    """

    pressure = float(solution.node_gauge_pressure_Pa)
    incoming = tuple(
        branch.area_m2 * branch.incoming_characteristic_m_s
        for branch in branches
    )
    corrections = tuple(
        branch.area_m2
        * pressure
        / (branch.density_kg_m3 * branch.wave_speed_m_s)
        for branch in branches
    )
    scale = math.fsum(abs(value) for value in (*incoming, *corrections))
    tolerance = _TEE_EQUILIBRIUM_ROUNDOFF_FACTOR * math.ulp(1.0) * scale
    flows = (
        solution.west_outward_flow_m3_s,
        solution.east_outward_flow_m3_s,
        solution.riser_outward_flow_m3_s,
    )
    if max(abs(float(value)) for value in flows) <= tolerance:
        return replace(
            solution,
            west_outward_flow_m3_s=0.0,
            east_outward_flow_m3_s=0.0,
            riser_outward_flow_m3_s=0.0,
            continuity_residual_m3_s=0.0,
            # Convective momentum is rho*Q^2/A.  Pinning every branch volume
            # flux to the exact hydrostatic fixed point therefore also pins
            # the riser momentum transaction to exact zero.  Retaining the
            # pre-pin, ulp-scale Q^2/A value would create an algebraically
            # impossible transaction (Q=0, Pi>0) and make the receiver reject
            # an otherwise exact roundoff equilibrium.
            normal_momentum_to_riser_N=0.0,
        )
    return solution


class PersistentCampaign2Candidate:
    """Single-run persistent horizontal/T/vertical owner.

    Parameters
    ----------
    riser_diameter_m:
        The only physical input allowed to differ among Campaign-2 runs.
    The numerical contract is the module-level :data:`SHARED_NUMERICS`; it is
    intentionally not a per-run constructor argument.
    """

    def __init__(
        self,
        riser_diameter_m: float,
    ) -> None:
        diameter = float(riser_diameter_m)
        if not math.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("riser_diameter_m must be positive and finite")
        if diameter > APPARATUS.pipe_diameter_m:
            raise ValueError("riser diameter cannot exceed the tunnel diameter")
        self.riser_diameter_m = diameter
        numerics = SHARED_NUMERICS
        self.numerics = numerics

        horizontal = Campaign2Case1MirroredHorizontal(
            length=APPARATUS.tunnel_length_m,
            diameter=APPARATUS.pipe_diameter_m,
            physical_valve_x=APPARATUS.valve_x_m,
            physical_riser_x=APPARATUS.riser_x_m,
            initial_water_head_from_invert=(
                APPARATUS.initial_head_from_invert_m
            ),
            dx=numerics.horizontal_dx_m,
            wave_speed=SHARED_CLOSURE.wave_speed_m_s,
            valve_open_time=APPARATUS.valve_open_time_s,
            liquid_density=APPARATUS.liquid_density_kg_m3,
            liquid_dynamic_viscosity=(
                APPARATUS.liquid_dynamic_viscosity_Pa_s
            ),
            liquid_bulk_modulus=APPARATUS.liquid_bulk_modulus_Pa,
            atmospheric_pressure=APPARATUS.atmospheric_pressure_Pa,
            gravity=APPARATUS.gravity_m_s2,
            gas_constant=APPARATUS.gas_constant_J_kg_K,
            gas_temperature=APPARATUS.gas_temperature_K,
            coupling_interval=numerics.max_dt_s,
        )
        acoustic_limit = horizontal.dx / horizontal.section.wave_speed
        if numerics.max_dt_s > acoustic_limit * (1.0 + 1.0e-12):
            raise ValueError(
                "max_dt_s exceeds one horizontal acoustic cell"
            )
        self.horizontal_owner = PersistentHorizontalOwner.initialize(
            horizontal,
            reservoir_head_from_invert_m=(
                APPARATUS.initial_head_from_invert_m
            ),
        )

        cell_count = round(
            APPARATUS.riser_height_m / numerics.vertical_dz_m
        )
        if cell_count <= 0 or not math.isclose(
            cell_count * numerics.vertical_dz_m,
            APPARATUS.riser_height_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "vertical_dz_m must divide the physical riser height exactly"
            )
        vertical_parameters = VerticalTwoFluidParameters(
            cell_count=cell_count,
            cell_length_m=numerics.vertical_dz_m,
            diameter_m=diameter,
            # Use the Case-1 material value so a shared pressure head has one
            # physical density on every side of the T.
            liquid_density_kg_m3=horizontal.config.liquid_density,
            gas_constant_J_kg_K=horizontal.config.gas_constant,
            gas_temperature_K=APPARATUS.gas_temperature_K,
            liquid_wave_speed_m_s=SHARED_CLOSURE.wave_speed_m_s,
            atmospheric_pressure_Pa=horizontal.config.atmospheric_pressure,
            gravity_m_s2=horizontal.config.gravity,
        )
        top = AtmosphericTopBoundary(
            pressure_abs_Pa=horizontal.config.atmospheric_pressure
        )
        self.vertical_kernel = Campaign2VerticalTwoFluidKernel(
            vertical_parameters,
            top,
        )
        self.vertical_state = hydrostatic_column_state(
            vertical_parameters,
            liquid_height_m=APPARATUS.initial_head_from_crown_m,
        )

        initial_liquid = (
            _liquid_volume_horizontal(self.horizontal_owner)
            + _liquid_volume_vertical(
                self.vertical_state,
                vertical_parameters,
            )
        )
        initial_gas = (
            _gas_mass_horizontal(self.horizontal_owner)
            + _gas_mass_vertical(self.vertical_state)
        )
        self.global_budget = Campaign2GlobalBudget(
            initial_liquid_volume_m3=initial_liquid,
            initial_gas_mass_kg=initial_gas,
        )
        self.step_count = 0
        self.last_tee_solution: TeeSolveResult | None = None
        self.last_step: CoupledStepRecord | None = None
        self.maximum_abs_global_liquid_residual_m3 = 0.0
        self.maximum_abs_global_gas_residual_kg = 0.0

    @property
    def parameters(self) -> VerticalTwoFluidParameters:
        return self.vertical_kernel.parameters

    @property
    def time_s(self) -> float:
        horizontal_time = float(self.horizontal_owner.state.time)
        vertical_time = float(self.vertical_state.time_s)
        if not math.isclose(
            horizontal_time,
            vertical_time,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise FloatingPointError(
                "horizontal and vertical physical clocks have diverged"
            )
        return horizontal_time

    @property
    def physical_top_liquid_outflow_m3(self) -> float:
        return float(self.vertical_state.cumulative_top_liquid_outflow_m3)

    @property
    def physical_rim_crossed(self) -> bool:
        """True only after finite liquid volume crosses the physical rim."""

        return bool(
            self.physical_top_liquid_outflow_m3
            > SHARED_CLOSURE.top_liquid_outflow_tolerance_m3
        )

    @property
    def qualification_ready(self) -> bool:
        return bool(COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY)

    def solver_facing_contract(self) -> dict[str, Any]:
        """Return shared physics plus the sole per-run physical variable."""

        return {
            "apparatus": asdict(APPARATUS),
            "shared_closure": asdict(SHARED_CLOSURE),
            "numerics": asdict(self.numerics),
            "riser_diameter_m": self.riser_diameter_m,
            "initial_riser_liquid_height_above_crown_m": (
                APPARATUS.initial_head_from_crown_m
            ),
            "horizontal_model": self.horizontal_owner.solver.provenance(),
            "horizontal_owner_persistent": True,
            "constant_head_boundary": (
                self.horizontal_owner.reservoir_boundary.provenance()
                if self.horizontal_owner.reservoir_boundary is not None
                else None
            ),
            "vertical_model": "conservative Al/Ql/Mg/Jg two-fluid FV kernel",
            "vertical_pressure_closure": (
                "dynamic isothermal EOS common-pressure faces; "
                "kernel default after conservative transport"
            ),
            "vertical_parameters": asdict(self.parameters),
            "vertical_top_boundary": asdict(
                self.vertical_kernel.top_boundary
            ),
            "qualification_ready": self.qualification_ready,
            "missing_vertical_closures": list(MISSING_PHYSICAL_CLOSURES),
        }

    def _horizontal_liquid_trace(
        self,
        area_m2: float,
        discharge_m3_s: float,
        *,
        outward_sign: float,
    ) -> LiquidBranchTrace:
        solver = self.horizontal_owner.solver
        full_area = float(solver.section.full_area)
        area = float(area_m2)
        if area <= solver.config.dry_area_fraction * full_area:
            raise CouplingClosureUnavailable(
                "a dry horizontal T branch has no liquid characteristic"
            )
        flow_area = min(area, full_area)
        head_from_invert = float(solver.section.head_from_area(area))
        # The vertical branch origin is the horizontal crown.  A partially
        # full free surface below the crown cannot carry positive crown
        # pressure; this is a state projection, not a fitted pressure offset.
        gauge_head = max(
            head_from_invert - APPARATUS.pipe_diameter_m,
            0.0,
        )
        density = float(solver.config.liquid_density)
        return LiquidBranchTrace(
            area_m2=flow_area,
            outward_velocity_m_s=(
                float(outward_sign) * float(discharge_m3_s) / flow_area
            ),
            gauge_pressure_Pa=(
                density * solver.config.gravity * gauge_head
            ),
            wave_speed_m_s=SHARED_CLOSURE.wave_speed_m_s,
            density_kg_m3=density,
        )

    def _vertical_pressure_faces(
        self,
        state: VerticalTwoFluidState | None = None,
    ) -> tuple[float, ...]:
        """Return the production EOS/common-pressure reconstruction.

        This field is used only to construct the current vertical traces for
        the shared T solve.  The subsequent FV update reconstructs pressure
        again from the *transported* conserved state and consumes the same T
        transaction through the kernel's default pressure path.
        """

        current = self.vertical_state if state is None else state
        return isothermal_common_pressure_faces(
            current,
            self.parameters,
            self.vertical_kernel.top_boundary,
        )

    def _vertical_liquid_trace(
        self,
        pressure_faces: tuple[float, ...],
    ) -> LiquidBranchTrace:
        area = float(self.vertical_state.Al[0])
        if area <= self.parameters.area_tolerance_m2:
            raise CouplingClosureUnavailable(
                "a dry vertical bottom has no liquid characteristic"
            )
        velocity = float(self.vertical_state.Ql[0]) / area
        full_area = self.parameters.full_area_m2
        bottom_has_gas = (
            full_area - area > self.parameters.area_tolerance_m2
        )
        if bottom_has_gas:
            # The present common-pressure closure has no phase-pressure jump
            # inside a mixed mouth cell.  Liquid gauge pressure is therefore
            # zero relative to the local gas pressure used by the gas trace.
            liquid_pressure = 0.0
        else:
            gas_reference = self._vertical_liquid_pressure_reference_abs_Pa(
                pressure_faces
            )
            liquid_pressure = float(pressure_faces[0]) - gas_reference
        pressure_tolerance = (
            512.0
            * np.finfo(float).eps
            * max(abs(float(pressure_faces[0])), 1.0)
        )
        if liquid_pressure < -pressure_tolerance:
            raise CouplingClosureUnavailable(
                "dynamic vertical liquid trace has negative gauge pressure"
            )
        liquid_pressure = max(float(liquid_pressure), 0.0)
        return LiquidBranchTrace(
            area_m2=area,
            outward_velocity_m_s=velocity,
            gauge_pressure_Pa=float(liquid_pressure),
            wave_speed_m_s=SHARED_CLOSURE.wave_speed_m_s,
            density_kg_m3=self.parameters.liquid_density_kg_m3,
        )

    def _lower_front_liquid_trace(
        self,
        pressure_faces: tuple[float, ...],
        pressure_reference_abs_Pa: float,
    ) -> LiquidBranchTrace:
        """Return a diagnostic liquid-slug trace above a persisted gas pocket.

        This trace is deliberately not connected to the physical T liquid
        node.  It records the current slug state for the shared audit object;
        :func:`solve_liquid_tee_with_blocked_riser` consumes only west/east.
        """

        front = self.vertical_state.lower_material_front_cell
        if front is None:
            raise CouplingClosureUnavailable("no persisted lower material front")
        al = float(self.vertical_state.Al[front])
        if al <= 0.0:
            raise CouplingClosureUnavailable("lower front has no liquid trace")
        velocity = float(self.vertical_state.Ql[front]) / al
        absolute_pressure = float(pressure_faces[front])
        return LiquidBranchTrace(
            area_m2=float(self.parameters.full_area_m2),
            outward_velocity_m_s=velocity,
            gauge_pressure_Pa=absolute_pressure - pressure_reference_abs_Pa,
            wave_speed_m_s=SHARED_CLOSURE.wave_speed_m_s,
            density_kg_m3=self.parameters.liquid_density_kg_m3,
        )

    def _vertical_liquid_pressure_reference_abs_Pa(
        self,
        pressure_faces: tuple[float, ...],
    ) -> float:
        """Return the absolute datum used by every liquid gauge trace.

        At first bottom entry this is the base pressure of the existing
        top-connected riser gas column (or atmosphere for an all-liquid riser),
        never the fictitious density formerly assigned to an empty bottom gas
        phase.  The same explicit datum is passed into the coupled gas/piston
        characteristic so ``p_ref + p_gauge == p*`` is auditable.
        """

        full_area = self.parameters.full_area_m2
        reference = float(
            self.vertical_kernel.top_boundary.pressure_abs_Pa
            if self.vertical_kernel.top_boundary is not None
            else self.parameters.atmospheric_pressure_Pa
        )
        for cell, liquid_area in enumerate(self.vertical_state.Al):
            if (
                full_area - float(liquid_area) > 0.0
                and float(self.vertical_state.Mg[cell]) > 0.0
            ):
                return float(pressure_faces[cell])
        return reference

    def _gas_traces_and_open_area(
        self,
        area: np.ndarray,
        gas_mass: np.ndarray,
        gas_momentum: np.ndarray,
        pressure_faces: tuple[float, ...],
    ) -> tuple[GasTrace, GasTrace, float]:
        horizontal = self.horizontal_owner
        solver = horizontal.solver
        face = solver.physical_junction_face_index
        west, east = face - 1, face
        full_h = float(solver.section.full_area)
        horizontal_void = max(
            max(full_h - min(float(area[west]), full_h), 0.0),
            max(full_h - min(float(area[east]), full_h), 0.0),
        )
        local_horizontal_gas_mass = float(gas_mass[west] + gas_mass[east])
        if local_horizontal_gas_mass <= 0.0:
            horizontal_void = 0.0

        finite_lower_pocket = bool(
            self.vertical_state.lower_material_front_orientation
            == "gas_below_liquid_above"
        )
        gas_area_vertical = max(
            self.parameters.full_area_m2 - float(self.vertical_state.Al[0]),
            0.0,
        )
        if self.vertical_state.Mg[0] <= 0.0:
            gas_area_vertical = 0.0
        # A resolved bottom gas parcel supplies an ordinary two-gas opening.
        # For an exactly saturated bottom, the *available* first-entry opening
        # is instead the real horizontal material-void/riser overlap.  It must
        # not be multiplied by the zero pre-entry vertical void, which caused
        # the old zero-area deadlock.
        vertical_limit = (
            self.parameters.full_area_m2
            if finite_lower_pocket
            else (
                gas_area_vertical
                if gas_area_vertical > 0.0 and self.vertical_state.Mg[0] > 0.0
                else self.parameters.full_area_m2
            )
        )
        open_area = min(
            horizontal_void,
            vertical_limit,
            self.parameters.full_area_m2,
        )

        p_h = float(horizontal.state.air_pressure_abs)
        rho_h = float(horizontal.state.gas.mass / horizontal.state.gas.volume)
        c_h = math.sqrt(GAS_HEAT_CAPACITY_RATIO * p_h / rho_h)
        horizontal_trace = GasTrace(
            pressure_abs_Pa=p_h,
            density_kg_m3=rho_h,
            # Axial pocket momentum turns at the side branch; it is not a
            # prescribed vertical normal velocity.
            normal_velocity_m_s=0.0,
            sound_speed_m_s=c_h,
        )

        p_v = float(pressure_faces[0])
        if gas_area_vertical > self.parameters.area_tolerance_m2:
            rho_v = float(
                self.vertical_state.Mg[0]
                / (gas_area_vertical * self.parameters.cell_length_m)
            )
            u_v = float(
                self.vertical_state.Jg[0] / self.vertical_state.Mg[0]
            )
        else:
            rho_v = float(self.parameters.atmospheric_gas_density_kg_m3)
            u_v = 0.0
        # The vertical owner is barotropic/isothermal: p=rho*R*T.  Its
        # characteristic impedance must therefore use c=sqrt(R*T), matching
        # the vertical EOS, acoustic CFL, and lower material-star solve.  The
        # horizontal donor may retain its own closure independently.
        c_v = float(self.parameters.isothermal_gas_sound_speed_m_s)
        vertical_trace = GasTrace(
            pressure_abs_Pa=p_v,
            density_kg_m3=rho_v,
            normal_velocity_m_s=u_v,
            sound_speed_m_s=c_v,
        )
        return horizontal_trace, vertical_trace, float(open_area)

    def _solve_current_tee_uncommitted(self) -> TeeSolveResult:
        """Return the current T solution without changing any owned state.

        The shared timestep selector needs the current bottom liquid flux to
        bound motion of the reconstructed upper free surface.  Keeping that
        preview pure is essential: rejecting an explicitly oversized step
        must not even change ``last_tee_solution``.
        """

        area, discharge, gas_mass, gas_momentum = (
            self.horizontal_owner.physical_snapshot()
        )
        face = self.horizontal_owner.solver.physical_junction_face_index
        west = self._horizontal_liquid_trace(
            area[face - 1],
            discharge[face - 1],
            outward_sign=-1.0,
        )
        east = self._horizontal_liquid_trace(
            area[face],
            discharge[face],
            outward_sign=1.0,
        )
        pressure_faces = self._vertical_pressure_faces()
        pressure_reference = (
            self._vertical_liquid_pressure_reference_abs_Pa(pressure_faces)
        )
        # LiquidBranchTrace stores gauge pressure, so all three incoming
        # characteristics must use one explicitly named absolute datum.  The
        # horizontal section's head reconstruction is relative to atmosphere;
        # the vertical trace is already relative to ``pressure_reference``.
        # Shifting a datum changes no absolute pressure and prevents the first
        # top-gas hydrostatic head from being silently mixed with atmosphere.
        horizontal_reference = float(
            self.horizontal_owner.solver.config.atmospheric_pressure
        )
        gauge_shift = horizontal_reference - pressure_reference
        west = replace(
            west,
            gauge_pressure_Pa=west.gauge_pressure_Pa + gauge_shift,
        )
        east = replace(
            east,
            gauge_pressure_Pa=east.gauge_pressure_Pa + gauge_shift,
        )
        finite_lower_pocket = bool(
            self.vertical_state.lower_material_front_orientation
            == "gas_below_liquid_above"
        )
        if finite_lower_pocket:
            riser = self._lower_front_liquid_trace(
                pressure_faces,
                pressure_reference,
            )
            liquid = solve_liquid_tee_with_blocked_riser(west, east)
        else:
            riser = self._vertical_liquid_trace(pressure_faces)
            # Preserve the exact hydrostatic fixed point when independently
            # evaluated but analytically identical pressure heads differ only by
            # floating-point summation order.  This is a roundoff pin, bounded by
            # machine epsilon; it cannot alter a resolved pressure transient.
            if _liquid_tee_is_roundoff_equilibrium((west, east, riser)):
                common_pressure = float(west.gauge_pressure_Pa)
                west = replace(
                    west,
                    gauge_pressure_Pa=common_pressure,
                    outward_velocity_m_s=0.0,
                )
                east = replace(
                    east,
                    gauge_pressure_Pa=common_pressure,
                    outward_velocity_m_s=0.0,
                )
                riser = replace(
                    riser,
                    gauge_pressure_Pa=common_pressure,
                    outward_velocity_m_s=0.0,
                )
                # This is the analytic lossless-node solution.  Calling the
                # generic floating-point solve here can leave O(1e-19 m3/s)
                # branch roundoff and then reject its own nominal zero-flow
                # residual for small riser areas (notably Dr=0.016 m).
                liquid = LiquidTeeSolution(
                    node_gauge_pressure_Pa=common_pressure,
                    west_outward_flow_m3_s=0.0,
                    east_outward_flow_m3_s=0.0,
                    riser_outward_flow_m3_s=0.0,
                    continuity_residual_m3_s=0.0,
                    riser_open_area_m2=float(riser.area_m2),
                )
            else:
                liquid = solve_liquid_tee(west, east, riser)
                liquid = _pin_liquid_tee_roundoff_flows(
                    (west, east, riser),
                    liquid,
                )
        horizontal_gas, vertical_gas, open_area = (
            self._gas_traces_and_open_area(
                area,
                gas_mass,
                gas_momentum,
                pressure_faces,
            )
        )
        first_entry: FirstBottomGasEntrySolution | None = None
        bottom_is_exactly_saturated = bool(
            self.vertical_state.Al[0] == self.parameters.full_area_m2
            and self.vertical_state.Mg[0] == 0.0
            and self.vertical_state.Jg[0] == 0.0
            and self.vertical_state.lower_material_front_cell is None
            and self.vertical_state.lower_material_front_orientation is None
        )
        if bottom_is_exactly_saturated:
            first_entry = solve_first_bottom_gas_entry(
                west,
                east,
                riser,
                horizontal_gas,
                liquid_pressure_reference_abs_Pa=pressure_reference,
                available_gas_open_area_m2=open_area,
                closed_liquid_solution=liquid,
            )
            liquid = first_entry.liquid
            gas = first_entry.gas
            open_area = first_entry.gas_open_area_m2
        else:
            gas = solve_gas_tee(
                horizontal_gas,
                vertical_gas,
                open_area_m2=open_area,
            )
        transaction = transaction_from_tee_solutions(
            liquid,
            gas,
            physical_riser_area_m2=self.parameters.full_area_m2,
        )
        solved = TeeSolveResult(
            transaction=transaction,
            liquid=liquid,
            gas=gas,
            gas_open_area_m2=open_area,
            horizontal_west_liquid=west,
            horizontal_east_liquid=east,
            vertical_bottom_liquid=riser,
            horizontal_gas=horizontal_gas,
            vertical_bottom_gas=vertical_gas,
            first_bottom_gas_entry=first_entry,
            finite_bottom_gas_pocket=finite_lower_pocket,
        )
        return solved

    def solve_current_tee(self) -> TeeSolveResult:
        """Solve and record one case-independent T transaction from state."""

        solved = self._solve_current_tee_uncommitted()
        self.last_tee_solution = solved
        return solved

    def _upper_surface_geometric_timestep_limit(
        self,
        bottom_liquid_flow_m3_s: float,
    ) -> float:
        """Bound either direction of one upper material-interface motion.

        A negative T-node liquid flux removes liquid inventory ``Al_i*dz``
        from the reconstructed interface cell.  A positive flux rewets the
        grid-aligned first gas cell or the existing cut cell and consumes its
        gas volume ``Ag_i*dz``.  Dividing the appropriate phase volume by
        ``|Q_T|`` gives the exact directional crossing event.  Acoustic and
        ordinary transport bounds retain the shared 0.45 CFL, but the final
        material-event substep must land on the face rather than approach it
        geometrically.

        This routine mirrors only the kernel's admissible monotone
        liquid-below/gas-above geometry.  If that topology is absent, it adds
        no numerical bound and leaves the kernel to report the corresponding
        physical closure gap; it never repairs or reclassifies a state.
        """

        flow = float(bottom_liquid_flow_m3_s)
        if not math.isfinite(flow):
            raise FloatingPointError(
                "the current T-node liquid flow is not finite"
            )
        if flow == 0.0:
            return math.inf
        if self.vertical_state.lower_material_front_cell is not None:
            # The finite-pocket lower-front helper bounds both ends of the
            # translated liquid plug from one common material-star flux.
            return math.inf

        state = canonicalize_upper_free_surface_roundoff(
            self.vertical_state,
            self.parameters,
        )
        parameters = self.parameters
        full_area = float(parameters.full_area_m2)
        area_tolerance = float(parameters.area_tolerance_m2)
        cell_count = int(parameters.cell_count)

        def full_liquid_like(cell: int) -> bool:
            liquid_area = float(state.Al[cell])
            gas_area = max(full_area - liquid_area, 0.0)
            return bool(
                liquid_area > area_tolerance
                and gas_area <= area_tolerance
                and float(state.Mg[cell]) == 0.0
            )

        def resolved_cut(cell: int) -> bool:
            liquid_area = float(state.Al[cell])
            gas_area = max(full_area - liquid_area, 0.0)
            return bool(
                liquid_area > 0.0
                and gas_area > 0.0
                and float(state.Mg[cell]) > 0.0
            )

        def top_gas(cell: int) -> bool:
            liquid_area = float(state.Al[cell])
            gas_area = max(full_area - liquid_area, 0.0)
            return bool(
                liquid_area <= 0.0
                and gas_area > area_tolerance
                and float(state.Mg[cell]) > 0.0
            )

        full_count = 0
        while full_count < cell_count and full_liquid_like(full_count):
            full_count += 1
        if full_count == 0:
            return math.inf

        cut_cell: int | None = None
        cursor = full_count
        if cursor < cell_count and resolved_cut(cursor):
            cut_cell = cursor
            cursor += 1
        if cursor >= cell_count or not all(
            top_gas(cell) for cell in range(cursor, cell_count)
        ):
            return math.inf
        interface_cell = (
            cut_cell
            if cut_cell is not None
            else (full_count - 1 if flow < 0.0 else full_count)
        )
        if interface_cell < 0 or interface_cell >= cell_count:
            return math.inf
        if flow < 0.0 and interface_cell + 1 >= cell_count:
            return math.inf

        liquid_area = float(state.Al[interface_cell])
        directional_phase_area = (
            liquid_area
            if flow < 0.0
            else max(full_area - liquid_area, 0.0)
        )
        directional_phase_volume_m3 = (
            directional_phase_area * float(parameters.cell_length_m)
        )
        if directional_phase_volume_m3 <= 0.0:
            return math.inf
        one_cell_crossing_time_s = (
            directional_phase_volume_m3 / abs(flow)
        )
        limit = one_cell_crossing_time_s
        if not math.isfinite(limit) or limit <= 0.0:
            raise FloatingPointError(
                "upper free-surface geometric CFL limit is not positive"
            )
        return float(limit)

    def _lower_surface_geometric_timestep_limit(
        self,
        gas_volume_flow_to_riser_m3_s: float | None,
        bottom_liquid_flow_m3_s: float,
    ) -> float:
        """Bound first-entry and finite-pocket material displacements.

        The bound uses the exact positive donor-derived gas volume flux carried
        by the T transaction.  It never substitutes ``mdot/(p/RT)`` from the
        receiving riser and is inactive only when the gas port is closed.

        On first entry, ``Qg`` creates the lower gas cut while the connected
        liquid plug translates at the actual net fill ``Ql + Qg``.  A prior
        pocket evacuation can leave a cut upper free surface even though the
        lower marker has disappeared.  The next shared step must therefore
        stop at whichever event occurs first: exhaustion of the saturated
        bottom cell or exhaustion of the directional phase inventory at that
        upper surface.  Omitting the latter lets the nominally stable driver
        offer a step that the conservative receiver must reject.
        """

        state = self.vertical_state
        parameters = self.parameters
        if (
            state.lower_material_front_orientation
            == "gas_below_liquid_above"
        ):
            return lower_material_front_geometric_timestep_limit(
                state,
                parameters,
                cfl=1.0,
                top=self.vertical_kernel.top_boundary,
            )
        if gas_volume_flow_to_riser_m3_s is None:
            return math.inf
        qg = float(gas_volume_flow_to_riser_m3_s)
        if not math.isfinite(qg):
            raise FloatingPointError("bottom gas volume flow is not finite")
        if qg <= 0.0:
            return math.inf
        if (
            state.Al[0] == parameters.full_area_m2
            and state.Mg[0] == 0.0
            and state.Jg[0] == 0.0
        ):
            front = 0
        else:
            return math.inf
        remaining_liquid_volume = (
            float(state.Al[front]) * parameters.cell_length_m
        )
        if remaining_liquid_volume <= 0.0:
            return math.inf
        bottom_limit = remaining_liquid_volume / qg
        plug_volume_flow = (
            float(bottom_liquid_flow_m3_s) + qg
        )
        upper_limit = (
            math.inf
            if plug_volume_flow == 0.0
            else self._upper_surface_geometric_timestep_limit(
                plug_volume_flow
            )
        )
        limit = min(bottom_limit, upper_limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise FloatingPointError(
                "lower material-interface geometric CFL is not positive"
            )
        return float(limit)

    def _audit_global_budget(self) -> dict[str, float]:
        return self.global_budget.audit(
            final_horizontal_liquid_m3=(
                _liquid_volume_horizontal(self.horizontal_owner)
            ),
            final_vertical_liquid_m3=_liquid_volume_vertical(
                self.vertical_state,
                self.parameters,
            ),
            final_horizontal_gas_kg=(
                _gas_mass_horizontal(self.horizontal_owner)
            ),
            final_vertical_gas_kg=_gas_mass_vertical(self.vertical_state),
        )

    def stable_coupled_timestep(
        self,
        *,
        target_time_s: float | None = None,
        event_times_s: Iterable[float] = (),
    ) -> float:
        """Return the current shared horizontal/vertical physical time step.

        ``max_dt_s`` is only a ceiling.  Every call recomputes the horizontal
        Case-1 stability limit, the vertical liquid and isothermal-gas
        acoustic limits, and the upper-free-surface geometric limit from the
        currently owned states.  Only cells with a strictly positive phase
        inventory and phase area participate.  The next valve, caller event,
        or requested target is also an exact upper bound, so accepted steps
        cannot jump across a physical event.
        """

        current = self.time_s
        numerics = self.numerics
        horizontal = self.horizontal_owner.solver
        horizontal_state = self.horizontal_owner.state
        native_horizontal_dt = float(
            horizontal.stable_timestep(horizontal_state)
        )
        horizontal_cfl = float(horizontal.config.cfl)
        if (
            not math.isfinite(native_horizontal_dt)
            or native_horizontal_dt <= 0.0
            or not math.isfinite(horizontal_cfl)
            or horizontal_cfl <= 0.0
        ):
            raise FloatingPointError(
                "horizontal solver returned an invalid stability limit"
            )
        horizontal_limit = (
            native_horizontal_dt * numerics.shared_cfl / horizontal_cfl
        )

        state = self.vertical_state
        parameters = self.parameters
        full_area = float(parameters.full_area_m2)
        dz = float(parameters.cell_length_m)
        gas_sound_speed = float(parameters.isothermal_gas_sound_speed_m_s)
        liquid_wave_speed = float(SHARED_CLOSURE.wave_speed_m_s)
        gas_limit = math.inf
        liquid_limit = math.inf
        for liquid_area, liquid_flow, gas_mass, gas_momentum in zip(
            state.Al,
            state.Ql,
            state.Mg,
            state.Jg,
        ):
            al = float(liquid_area)
            ag = full_area - al
            mg = float(gas_mass)
            if mg > 0.0 and ag > 0.0:
                gas_velocity = float(gas_momentum) / mg
                cell_limit = dz / (abs(gas_velocity) + gas_sound_speed)
                gas_limit = min(gas_limit, numerics.shared_cfl * cell_limit)
            if al > 0.0:
                liquid_velocity = float(liquid_flow) / al
                cell_limit = dz / (abs(liquid_velocity) + liquid_wave_speed)
                liquid_limit = min(
                    liquid_limit,
                    numerics.shared_cfl * cell_limit,
                )

        current_tee = self._solve_current_tee_uncommitted()
        upper_surface_geometric_limit = (
            self._upper_surface_geometric_timestep_limit(
                current_tee.transaction.liquid_flow_to_riser_m3_s
            )
        )
        lower_surface_geometric_limit = (
            self._lower_surface_geometric_timestep_limit(
                current_tee.transaction.gas_volume_flow_to_riser_m3_s,
                current_tee.transaction.liquid_flow_to_riser_m3_s,
            )
        )

        limits = [
            float(numerics.max_dt_s),
            horizontal_limit,
            gas_limit,
            liquid_limit,
            upper_surface_geometric_limit,
            lower_surface_geometric_limit,
        ]
        if target_time_s is not None:
            target = float(target_time_s)
            if not math.isfinite(target) or target < current:
                raise ValueError(
                    "target_time_s must be finite and not precede current time"
                )
            if target == current:
                return 0.0
            limits.append(target - current)

        events = (float(horizontal.valve_open_time), *event_times_s)
        for raw_event in events:
            event = float(raw_event)
            if not math.isfinite(event):
                raise ValueError("event times must be finite")
            if event > current:
                limits.append(event - current)

        stable = float(min(limits))
        if not math.isfinite(stable) or stable <= 0.0:
            raise FloatingPointError("coupled stability limit is not positive")
        return stable

    def advance_one_step(self, dt: float | None = None) -> CoupledStepRecord:
        """Atomically advance and commit one physical Lie-split step.

        A horizontal advance, reservoir commit, shared T solve, vertical
        advance, and all ledgers form one transaction.  If any participant or
        audit raises, every mutable owner is restored before the original
        exception is re-raised.
        """

        if dt is None:
            step = self.stable_coupled_timestep()
        else:
            step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        stable = self.stable_coupled_timestep()
        if step > stable:
            raise CoupledTimestepLimitExceeded(
                requested_dt_s=step,
                stable_dt_s=stable,
            )

        snapshot = _CoupledStepSnapshot.capture(self)
        try:
            return self._advance_one_step_transaction_body(step)
        except BaseException:
            snapshot.restore()
            raise

    def _advance_one_step_transaction_body(
        self,
        step: float,
    ) -> CoupledStepRecord:
        """Execute a validated step inside :meth:`advance_one_step`."""

        start = self.time_s

        # Freeze the one simultaneous T transaction from the same start state
        # used by ``stable_coupled_timestep``.  In particular, a material-event
        # time is computed from this transaction and must not be invalidated by
        # first advancing the horizontal owner and silently replacing Q with an
        # end-of-step value.  The horizontal PDE/reservoir substep remains
        # first in the Lie update; only the already-agreed interface flux is
        # frozen before it.
        solved = self.solve_current_tee()
        transaction = solved.transaction

        reservoir = self.horizontal_owner.advance(step)
        reservoir_exchange = (
            0.0
            if reservoir is None
            else float(reservoir.liquid_volume_to_horizontal_m3)
        )
        self.global_budget.book_reservoir_liquid(reservoir_exchange)

        vertical_before = self.vertical_state
        horizontal_liquid_before = _liquid_volume_horizontal(
            self.horizontal_owner
        )
        horizontal_gas_before = _gas_mass_horizontal(self.horizontal_owner)
        vertical_result = self.vertical_kernel.advance(
            vertical_before,
            dt=step,
            tee_transaction=transaction,
        )
        increment = self.horizontal_owner.commit_tee(transaction, step)
        horizontal_liquid_after = _liquid_volume_horizontal(
            self.horizontal_owner
        )
        horizontal_gas_after = _gas_mass_horizontal(self.horizontal_owner)

        horizontal_liquid_residual = (
            horizontal_liquid_after
            - horizontal_liquid_before
            + increment.liquid_volume_m3
        )
        horizontal_gas_residual = (
            horizontal_gas_after
            - horizontal_gas_before
            + increment.gas_mass_kg
        )
        vertical_liquid_residual = (
            vertical_result.budget.bottom_liquid_exchange_m3
            - increment.liquid_volume_m3
        )
        vertical_gas_residual = (
            vertical_result.budget.bottom_gas_exchange_kg
            - increment.gas_mass_kg
        )
        tolerance = 5.0e-15
        residuals = (
            horizontal_liquid_residual,
            horizontal_gas_residual,
            vertical_liquid_residual,
            vertical_gas_residual,
        )
        if any(abs(value) > tolerance for value in residuals):
            raise FloatingPointError(
                "the shared T transaction was not equal and opposite"
            )

        self.vertical_state = vertical_result.state
        self.global_budget.book_internal_tee(
            liquid_to_riser_m3=increment.liquid_volume_m3,
            gas_to_riser_kg=increment.gas_mass_kg,
        )
        top_liquid = float(vertical_result.budget.top_liquid_outflow_m3)
        top_gas_net = float(
            vertical_result.budget.top_gas_outflow_kg
            - vertical_result.budget.top_gas_inflow_kg
        )
        self.global_budget.book_top_liquid(top_liquid)
        self.global_budget.book_top_gas(top_gas_net)
        audit = self._audit_global_budget()
        self.maximum_abs_global_liquid_residual_m3 = max(
            self.maximum_abs_global_liquid_residual_m3,
            abs(audit["liquid_residual_m3"]),
        )
        self.maximum_abs_global_gas_residual_kg = max(
            self.maximum_abs_global_gas_residual_kg,
            abs(audit["gas_residual_kg"]),
        )
        self.step_count += 1
        record = CoupledStepRecord(
            start_time_s=start,
            end_time_s=self.time_s,
            transaction=transaction,
            gas_open_area_m2=solved.gas_open_area_m2,
            vertical_increment=increment,
            reservoir_liquid_exchange_m3=reservoir_exchange,
            top_liquid_outflow_m3=top_liquid,
            top_gas_net_outflow_kg=top_gas_net,
            horizontal_tee_liquid_residual_m3=(
                horizontal_liquid_residual
            ),
            horizontal_tee_gas_residual_kg=horizontal_gas_residual,
            vertical_tee_liquid_residual_m3=vertical_liquid_residual,
            vertical_tee_gas_residual_kg=vertical_gas_residual,
            global_liquid_residual_m3=audit["liquid_residual_m3"],
            global_gas_residual_kg=audit["gas_residual_kg"],
        )
        self.last_step = record
        return record

    def advance_to(
        self,
        target_time_s: float,
        *,
        event_times_s: Iterable[float] = (),
    ) -> int:
        """Advance with a freshly recomputed coupled limit to an exact target."""

        target = float(target_time_s)
        if not math.isfinite(target) or target < self.time_s:
            raise ValueError("target time must be finite and non-decreasing")
        events = tuple(float(value) for value in event_times_s)
        if any(not math.isfinite(value) for value in events):
            raise ValueError("event times must be finite")
        initial_step_count = self.step_count
        while self.time_s < target:
            step = self.stable_coupled_timestep(
                target_time_s=target,
                event_times_s=events,
            )
            if step <= 0.0:
                raise FloatingPointError(
                    "coupled stepper stalled before the requested target"
                )
            self.advance_one_step(step)
        if self.time_s != target:
            raise FloatingPointError(
                "coupled stepper did not land exactly on the requested target"
            )
        return self.step_count - initial_step_count

    def run_smoke(self, end_time_s: float) -> dict[str, Any]:
        """Advance to at most 0.05 s; never launch a qualification run."""

        target = float(end_time_s)
        if not math.isfinite(target) or target < self.time_s:
            raise ValueError("smoke end time must be finite and non-decreasing")
        if target > SMOKE_END_LIMIT_S:
            raise ValueError(
                f"smoke runner is capped at {SMOKE_END_LIMIT_S:.2f} s"
            )
        self.advance_to(target)
        audit = self._audit_global_budget()
        return {
            "role": "coupling smoke only; not qualification or manuscript evidence",
            "end_time_s": self.time_s,
            "step_count": self.step_count,
            "solver_facing_contract": self.solver_facing_contract(),
            "physical_top_liquid_outflow_m3": (
                self.physical_top_liquid_outflow_m3
            ),
            "physical_rim_crossed": self.physical_rim_crossed,
            "horizontal_owner_active": (
                self.horizontal_owner.horizontal_owner_active
            ),
            "tee_transaction_count": (
                self.horizontal_owner.tee_transaction_count
            ),
            "global_budget": audit,
            "maximum_abs_global_liquid_residual_m3": (
                self.maximum_abs_global_liquid_residual_m3
            ),
            "maximum_abs_global_gas_residual_kg": (
                self.maximum_abs_global_gas_residual_kg
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Campaign-2 persistent-coupling smoke case"
    )
    parser.add_argument("--riser-diameter-m", type=float, required=True)
    parser.add_argument("--smoke-end", type=float, default=0.01)
    args = parser.parse_args()
    candidate = PersistentCampaign2Candidate(args.riser_diameter_m)
    print(
        json.dumps(
            candidate.run_smoke(args.smoke_end),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CandidateNumerics",
    "CoupledStepRecord",
    "CoupledTimestepLimitExceeded",
    "CouplingClosureUnavailable",
    "PersistentCampaign2Candidate",
    "SHARED_NUMERICS",
    "SMOKE_END_LIMIT_S",
    "TeeSolveResult",
]
