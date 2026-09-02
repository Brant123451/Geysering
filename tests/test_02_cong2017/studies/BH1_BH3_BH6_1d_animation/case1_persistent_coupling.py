"""Conservative transaction layer for the persistent Campaign-2 1D coupler.

This module deliberately contains no geyser criterion and no case identifier.
It is the ownership boundary between the Case-1 horizontal state and the
future extracted vertical two-fluid kernel.  Mapping arrays are snapshots only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal
from case1_constant_head_boundary import (
    Case1ConstantHeadBoundary,
    ConstantHeadBoundaryCommit,
)
from campaign2_tee_riemann import GasTeeSolution, LiquidTeeSolution


@dataclass(frozen=True)
class TeeTransaction:
    """One simultaneous horizontal/T/vertical physical exchange."""

    west_liquid_flow_m3_s: float
    east_liquid_flow_m3_s: float
    gas_mass_flow_to_riser_kg_s: float
    # Authoritative, non-overlapping partition of the physical riser mouth.
    # These members remain defined at zero phase flow, where Q/u and
    # rho*Q^2/Pi are singular.
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float
    # Signed gas volume flux produced with the horizontal/riser Riemann
    # solver's actual upwind donor density.  The vertical receiver must never
    # infer this member from its own isothermal EOS.
    gas_volume_flow_to_riser_m3_s: float | None = None
    # Convective gas momentum flux ``mdot*u``.  T-face pressure is carried by
    # ``gas_interface_pressure_abs_Pa`` and applied by the vertical pressure
    # source; it must not be folded into this member.
    gas_normal_momentum_flow_N: float = 0.0
    # Convective liquid normal momentum flux; the common T pressure force is
    # excluded and consumed once by the vertical pressure-face field.
    liquid_normal_momentum_flow_N: float | None = None
    liquid_node_gauge_pressure_Pa: float = 0.0
    gas_interface_pressure_abs_Pa: float = 101_325.0

    def __post_init__(self) -> None:
        values = (
            self.west_liquid_flow_m3_s,
            self.east_liquid_flow_m3_s,
            self.gas_mass_flow_to_riser_kg_s,
            self.gas_normal_momentum_flow_N,
            self.liquid_node_gauge_pressure_Pa,
            self.gas_interface_pressure_abs_Pa,
            self.riser_mouth_area_m2,
            self.gas_open_area_m2,
            self.liquid_open_area_m2,
            self.blocked_riser_area_m2,
        )
        optional_values = tuple(
            value
            for value in (
                self.gas_volume_flow_to_riser_m3_s,
                self.liquid_normal_momentum_flow_N,
            )
            if value is not None
        )
        if not all(
            math.isfinite(float(value)) for value in (*values, *optional_values)
        ):
            raise ValueError("all T-junction transaction fields must be finite")
        if self.gas_interface_pressure_abs_Pa <= 0.0:
            raise ValueError("gas interface absolute pressure must be positive")
        mouth = float(self.riser_mouth_area_m2)
        gas = float(self.gas_open_area_m2)
        liquid = float(self.liquid_open_area_m2)
        blocked = float(self.blocked_riser_area_m2)
        if mouth <= 0.0 or min(gas, liquid, blocked) < 0.0:
            raise ValueError(
                "riser mouth must be positive and phase openings non-negative"
            )
        partition_residual = gas + liquid + blocked - mouth
        partition_roundoff = 64.0 * math.ulp(max(mouth, 1.0e-300))
        if abs(partition_residual) > partition_roundoff:
            raise ValueError(
                "gas, liquid, and blocked riser areas must partition the mouth"
            )

    @property
    def liquid_flow_to_riser_m3_s(self) -> float:
        return float(
            self.west_liquid_flow_m3_s - self.east_liquid_flow_m3_s
        )


@dataclass(frozen=True)
class VerticalTeeIncrement:
    """Exactly the phase inventory received by the vertical bottom face."""

    liquid_volume_m3: float
    gas_mass_kg: float
    gas_volume_m3: float
    gas_normal_momentum_kg_m_s: float
    liquid_normal_momentum_kg_m_s: float
    liquid_node_gauge_pressure_Pa: float
    gas_interface_pressure_abs_Pa: float


def transaction_from_tee_solutions(
    liquid: LiquidTeeSolution,
    gas: GasTeeSolution,
    *,
    physical_riser_area_m2: float,
) -> TeeTransaction:
    """Create the single transaction consumed by both phase owners."""

    mouth = float(physical_riser_area_m2)
    if not math.isfinite(mouth) or mouth <= 0.0:
        raise ValueError("physical riser area must be finite and positive")
    if gas.open_area_m2 is None or liquid.riser_open_area_m2 is None:
        raise ValueError(
            "T solutions must carry authoritative phase openings"
        )
    gas_open = float(gas.open_area_m2)
    liquid_open = float(liquid.riser_open_area_m2)
    raw_blocked = mouth - gas_open - liquid_open
    area_roundoff = 64.0 * math.ulp(mouth)
    if raw_blocked < -area_roundoff:
        raise ValueError("T-solution phase openings overlap the riser mouth")
    blocked = max(raw_blocked, 0.0)

    return TeeTransaction(
        west_liquid_flow_m3_s=liquid.physical_west_flow_m3_s,
        east_liquid_flow_m3_s=liquid.physical_east_flow_m3_s,
        gas_mass_flow_to_riser_kg_s=gas.mass_flow_to_riser_kg_s,
        gas_volume_flow_to_riser_m3_s=gas.volume_flow_to_riser_m3_s,
        gas_normal_momentum_flow_N=gas.normal_momentum_flow_N,
        liquid_normal_momentum_flow_N=(
            liquid.normal_momentum_to_riser_N
        ),
        liquid_node_gauge_pressure_Pa=liquid.node_gauge_pressure_Pa,
        gas_interface_pressure_abs_Pa=gas.interface_pressure_abs_Pa,
        riser_mouth_area_m2=mouth,
        gas_open_area_m2=gas_open,
        liquid_open_area_m2=liquid_open,
        blocked_riser_area_m2=blocked,
    )


@dataclass
class PersistentHorizontalOwner:
    """The sole mutable owner of the Case-1 horizontal conserved state."""

    solver: Campaign2Case1MirroredHorizontal
    state: object
    reservoir_boundary: Case1ConstantHeadBoundary | None = None
    cumulative_liquid_to_riser_m3: float = 0.0
    cumulative_gas_to_riser_kg: float = 0.0
    tee_transaction_count: int = 0

    @classmethod
    def initialize(
        cls,
        solver: Campaign2Case1MirroredHorizontal,
        *,
        reservoir_head_from_invert_m: float | None = 0.66,
    ) -> "PersistentHorizontalOwner":
        boundary = (
            None
            if reservoir_head_from_invert_m is None
            else Case1ConstantHeadBoundary(
                solver,
                reservoir_head_from_invert_m=(
                    reservoir_head_from_invert_m
                ),
            )
        )
        return cls(
            solver=solver,
            state=solver.initial_state(),
            reservoir_boundary=boundary,
        )

    def advance(self, dt: float) -> ConstantHeadBoundaryCommit | None:
        """Advance Case-1, then commit the physical reservoir face flux."""

        step = float(dt)
        self.state = self.solver.step_physical(self.state, step)
        if self.reservoir_boundary is None:
            return None
        committed = self.reservoir_boundary.commit(self.state, step)
        self.state = committed.state
        return committed

    def commit_tee(
        self,
        transaction: TeeTransaction,
        dt: float,
    ) -> VerticalTeeIncrement:
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        liquid_volume = transaction.liquid_flow_to_riser_m3_s * step
        gas_mass = transaction.gas_mass_flow_to_riser_kg_s * step
        gas_volume = (
            0.0
            if transaction.gas_volume_flow_to_riser_m3_s is None
            else transaction.gas_volume_flow_to_riser_m3_s * step
        )
        gas_momentum = transaction.gas_normal_momentum_flow_N * step
        liquid_momentum = (
            0.0
            if transaction.liquid_normal_momentum_flow_N is None
            else transaction.liquid_normal_momentum_flow_N * step
        )

        updated = self.solver.apply_physical_junction_liquid_fluxes(
            self.state,
            west_flow=transaction.west_liquid_flow_m3_s,
            east_flow=transaction.east_liquid_flow_m3_s,
            dt=step,
        )
        updated = self.solver.apply_physical_junction_gas_mass_flux(
            updated,
            mass_flow_to_riser=transaction.gas_mass_flow_to_riser_kg_s,
            dt=step,
        )
        self.state = updated
        self.cumulative_liquid_to_riser_m3 += liquid_volume
        self.cumulative_gas_to_riser_kg += gas_mass
        self.tee_transaction_count += 1
        return VerticalTeeIncrement(
            liquid_volume_m3=liquid_volume,
            gas_mass_kg=gas_mass,
            gas_volume_m3=gas_volume,
            gas_normal_momentum_kg_m_s=gas_momentum,
            liquid_normal_momentum_kg_m_s=liquid_momentum,
            liquid_node_gauge_pressure_Pa=(
                transaction.liquid_node_gauge_pressure_Pa
            ),
            gas_interface_pressure_abs_Pa=(
                transaction.gas_interface_pressure_abs_Pa
            ),
        )

    def physical_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a read-only mapped view; never replace the owned state."""

        return self.solver.map_to_physical(
            self.state,
            x_target=self.solver.x,
            full_area=self.solver.section.full_area,
            dx=self.solver.dx,
        )

    @property
    def horizontal_owner_active(self) -> bool:
        return True

    @property
    def cumulative_reservoir_liquid_inflow_m3(self) -> float:
        if self.reservoir_boundary is None:
            return 0.0
        return float(
            self.reservoir_boundary.cumulative_liquid_to_horizontal_m3
        )


__all__ = [
    "PersistentHorizontalOwner",
    "TeeTransaction",
    "VerticalTeeIncrement",
    "transaction_from_tee_solutions",
]
