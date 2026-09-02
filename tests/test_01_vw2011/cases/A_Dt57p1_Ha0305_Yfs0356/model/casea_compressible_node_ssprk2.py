"""Isolated SSP--RK2 wrapper for the compressible finite Case-A T node.

The lower-level post-launch operator returns one Forward-Euler update of the
two node inventories and three complete, outward-oriented branch fluxes.  This
module composes two such evaluations into the standard two-stage SSP--RK2
formula

``U(1) = U(n) + dt L(U(n))``

``U(n+1) = 1/2 U(n) + 1/2 [U(1) + dt L(U(1))]``.

The adjacent branch traces are intentionally frozen during this *local* node
step.  The node pressure and all six phase boundary rates are nevertheless
recomputed from the predictor inventory at stage two.  A future network
integrator must recompute its branch-cell traces at the same RK stages and use
the returned time-averaged fluxes on the branch side of each shared face.

There is no clipping, target trajectory, time-window switch, or independent
per-branch flux scaling here.  If either Euler stage violates the CFL or
exhausts an inventory, the existing fail-closed exception is propagated.
This file is deliberately not wired into the Case-A main time loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_compressible_finite_node import (
    CompressibleFiniteNodeState,
    CompressibleNodePressureState,
    solve_compressible_node_pressure,
)
from casea_compressible_node_postlaunch_stage import (
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchEulerResult,
    CompressiblePostLaunchParameters,
    euler_compressible_node_postlaunch_stage,
)
from casea_material_front_cutcell import StratifiedFlux


PRODUCTION_READY = False


@dataclass(frozen=True)
class CompressibleNodeSSPRK2Ledger:
    """Conservative audit of one accepted frozen-trace SSP--RK2 step."""

    dt: float
    initial_state: CompressibleFiniteNodeState
    predictor_state: CompressibleFiniteNodeState
    final_state: CompressibleFiniteNodeState
    west_time_average_flux: StratifiedFlux
    east_time_average_flux: StratifiedFlux
    vertical_time_average_flux: StratifiedFlux
    gas_mass_outward_rate: float
    liquid_equivalent_volume_outward_rate: float
    expected_gas_mass_change: float
    actual_gas_mass_change: float
    gas_mass_balance_residual: float
    expected_liquid_equivalent_volume_change: float
    actual_liquid_equivalent_volume_change: float
    liquid_inventory_balance_residual: float
    fixed_geometric_volume_change: float
    final_occupancy_residual: float


@dataclass(frozen=True)
class CompressibleNodeSSPRK2Result:
    """Final node state and the two boundary-flux stage evaluations."""

    state: CompressibleFiniteNodeState
    pressure: CompressibleNodePressureState
    west: StratifiedFlux
    east: StratifiedFlux
    vertical: StratifiedFlux
    first_stage: CompressiblePostLaunchEulerResult
    second_stage: CompressiblePostLaunchEulerResult
    ledger: CompressibleNodeSSPRK2Ledger
    gas_cfl: float
    liquid_cfl: float
    maximum_dt: float

    @property
    def branch_fluxes(self) -> dict[str, StratifiedFlux]:
        """Time-averaged face fluxes for conservative branch updates."""

        return {
            "west": self.west,
            "east": self.east,
            "vertical": self.vertical,
        }


def _average_flux(first: StratifiedFlux, second: StratifiedFlux) -> StratifiedFlux:
    return StratifiedFlux(
        gas_mass=0.5 * (first.gas_mass + second.gas_mass),
        gas_momentum=0.5 * (first.gas_momentum + second.gas_momentum),
        liquid_area=0.5 * (first.liquid_area + second.liquid_area),
        liquid_momentum=0.5 * (
            first.liquid_momentum + second.liquid_momentum
        ),
    )


def ssprk2_compressible_node_postlaunch_step(
    state: CompressibleFiniteNodeState,
    dt: float,
    *,
    west: CompressibleNodeResolvedBranch,
    east: CompressibleNodeResolvedBranch,
    vertical: CompressibleNodeResolvedBranch,
    params: CompressiblePostLaunchParameters,
) -> CompressibleNodeSSPRK2Result:
    """Advance the isolated finite node by one conservative SSP--RK2 step.

    All branch fluxes use coordinates pointing away from the node.  The
    returned ``west/east/vertical`` fluxes are the equal-weight stage average,
    so using their negatives in the neighbouring finite-volume cells gives
    the same gas and liquid transfer as the node ledger.
    """

    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("compressible finite-node RK2 dt must be finite and non-negative")

    first = euler_compressible_node_postlaunch_stage(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    second = euler_compressible_node_postlaunch_stage(
        first.node.state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )

    final_state = CompressibleFiniteNodeState(
        gas_mass=0.5 * (state.gas_mass + second.node.state.gas_mass),
        liquid_equivalent_volume=0.5
        * (
            state.liquid_equivalent_volume
            + second.node.state.liquid_equivalent_volume
        ),
        node_total_volume=state.node_total_volume,
    )
    final_pressure = solve_compressible_node_pressure(final_state, params.node)

    west_average = _average_flux(first.west, second.west)
    east_average = _average_flux(first.east, second.east)
    vertical_average = _average_flux(first.vertical, second.vertical)
    averages = (west_average, east_average, vertical_average)
    gas_outward = math.fsum(flux.gas_mass for flux in averages)
    liquid_outward = math.fsum(flux.liquid_area for flux in averages)

    expected_gas_change = -dt * gas_outward
    actual_gas_change = final_state.gas_mass - state.gas_mass
    expected_liquid_change = -dt * liquid_outward
    actual_liquid_change = (
        final_state.liquid_equivalent_volume
        - state.liquid_equivalent_volume
    )
    ledger = CompressibleNodeSSPRK2Ledger(
        dt=float(dt),
        initial_state=state,
        predictor_state=first.node.state,
        final_state=final_state,
        west_time_average_flux=west_average,
        east_time_average_flux=east_average,
        vertical_time_average_flux=vertical_average,
        gas_mass_outward_rate=float(gas_outward),
        liquid_equivalent_volume_outward_rate=float(liquid_outward),
        expected_gas_mass_change=float(expected_gas_change),
        actual_gas_mass_change=float(actual_gas_change),
        gas_mass_balance_residual=float(
            actual_gas_change - expected_gas_change
        ),
        expected_liquid_equivalent_volume_change=float(
            expected_liquid_change
        ),
        actual_liquid_equivalent_volume_change=float(actual_liquid_change),
        liquid_inventory_balance_residual=float(
            actual_liquid_change - expected_liquid_change
        ),
        fixed_geometric_volume_change=float(
            final_state.node_total_volume - state.node_total_volume
        ),
        final_occupancy_residual=float(final_pressure.occupancy_residual),
    )
    return CompressibleNodeSSPRK2Result(
        state=final_state,
        pressure=final_pressure,
        west=west_average,
        east=east_average,
        vertical=vertical_average,
        first_stage=first,
        second_stage=second,
        ledger=ledger,
        gas_cfl=max(first.gas_cfl, second.gas_cfl),
        liquid_cfl=max(first.liquid_cfl, second.liquid_cfl),
        maximum_dt=min(first.maximum_dt, second.maximum_dt),
    )


__all__ = [
    "PRODUCTION_READY",
    "CompressibleNodeSSPRK2Ledger",
    "CompressibleNodeSSPRK2Result",
    "ssprk2_compressible_node_postlaunch_step",
]
