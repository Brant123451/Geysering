"""Conservative west-port launch into the finite Case-A T control volume.

At first contact the measured T fitting is still liquid-full and contains
exactly zero gas mass.  Gas cannot be created simultaneously in the east and
vertical branches: it must first cross the west edge of the finite fitting.
This module advances that one topology event.

The west material-front trace is obtained from the strict reduced equations
implemented in :mod:`casea_paper_material_front_rh`.  Of the enumerated RH
roots, the launch continuation is the unique positive-speed middle-family
root.  The high-speed slow-family root is a different Riemann topology and is
reported by the strict enumerator rather than silently selected here.

During this event only the west gas face is open.  East and vertical gas rates
are identically zero until the gas-filled part of the finite fitting reaches
their measured openings.  Liquid remains connected to all three branches and
is updated with one common current node pressure.  No seed gas, receiver split,
speed cap, state fill, or result-dependent switch is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from casea_compressible_finite_node import (
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    CompressibleNodePressureState,
    solve_compressible_node_pressure,
)
from casea_material_front_cutcell import (
    InterfaceTraces,
    PressurisedState,
    StratifiedState,
)
from casea_paper_material_front_rh import (
    AffineGasPressureLaw,
    PaperFrontCandidate,
    PaperFrontClosureError,
    PaperFrontPhysics,
    candidate_to_ale_traces,
    enumerate_paper_front_candidates,
)
from casea_tjunction_shock_network import (
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    ZeroStorageTNodeSolution,
    evaluate_zero_storage_t_node_at_pressure,
)


class CompressibleNodeLaunchError(RuntimeError):
    """The exact west-port launch is not admissible for the current state."""


class LaunchStepCrossesNodeBranchPoint(CompressibleNodeLaunchError):
    """The requested step reaches a new geometric branch event."""

    def __init__(self, *, requested_distance: float, available_distance: float) -> None:
        self.requested_distance = float(requested_distance)
        self.available_distance = float(available_distance)
        super().__init__(
            "west-port launch step crosses the first finite-node branch point: "
            f"requested={requested_distance:.12g} m, "
            f"available={available_distance:.12g} m"
        )


@dataclass(frozen=True)
class CompressibleNodeWestLaunch:
    """One accepted exact-zero west-port launch step."""

    state: CompressibleFiniteNodeState
    current_pressure: CompressibleNodePressureState
    next_pressure: CompressibleNodePressureState
    front_candidate: PaperFrontCandidate
    front_traces: InterfaceTraces
    liquid_node: ZeroStorageTNodeSolution
    front_distance: float
    west_gas_mass_rate_into_node: float
    west_liquid_volume_rate_outward: float
    east_liquid_volume_rate_outward: float
    vertical_liquid_volume_rate_outward: float
    gas_mass_balance_residual: float
    liquid_inventory_balance_residual: float
    swept_gas_volume: float
    node_gas_volume: float
    geometric_volume_residual: float


def _unique_positive_middle_candidate(
    *,
    pressure_abs: float,
    west_pressurised_foot: PressurisedState,
    west_stratified_foot: StratifiedState,
    west_physics: PaperFrontPhysics,
) -> PaperFrontCandidate:
    candidates = enumerate_paper_front_candidates(
        pressurised_foot=west_pressurised_foot,
        stratified_foot=west_stratified_foot,
        pressurised_side="right",
        pressure_law=AffineGasPressureLaw.fixed(pressure_abs),
        physics=west_physics,
    )
    launch = tuple(
        candidate
        for candidate in candidates
        if candidate.active_set == "middle" and candidate.speed > 0.0
    )
    if len(launch) != 1:
        summary = ", ".join(
            f"w={candidate.speed:.9g}/{candidate.active_set}"
            for candidate in candidates
        ) or "no strict RH roots"
        raise CompressibleNodeLaunchError(
            "west gas cannot launch into the finite node through one unique "
            f"positive middle-family root at p={pressure_abs:.9g} Pa; {summary}"
        )
    return launch[0]


def advance_compressible_node_west_launch(
    state: CompressibleFiniteNodeState,
    *,
    dt: float,
    node_params: CompressibleFiniteNodeParameters,
    west_pressurised_foot: PressurisedState,
    west_stratified_foot: StratifiedState,
    west_physics: PaperFrontPhysics,
    liquid_characteristics: TeeLiquidCharacteristics,
    liquid_areas: ZeroStorageTBranchAreas,
    distance_to_first_branch: float,
) -> CompressibleNodeWestLaunch:
    """Advance gas from the west branch into an initially liquid-full T node.

    ``distance_to_first_branch`` is measured from the fixed west handoff face
    to the first physical receiver opening.  The caller must subdivide a step
    that reaches it and create the next topology explicitly.
    """

    if state.gas_mass != 0.0:
        raise ValueError("west launch requires an exactly gas-free node")
    if not all(math.isfinite(value) for value in (dt, distance_to_first_branch)):
        raise ValueError("launch step and geometry must be finite")
    if dt <= 0.0 or distance_to_first_branch <= 0.0:
        raise ValueError("launch step and branch distance must be positive")
    if not math.isclose(
        west_physics.gas_sound_speed,
        node_params.gas_sound_speed,
        rel_tol=2.0e-14,
        abs_tol=2.0e-14,
    ):
        raise ValueError("west and finite-node gas sound speeds disagree")
    if not math.isclose(
        west_physics.liquid_density,
        node_params.liquid_density,
        rel_tol=2.0e-14,
        abs_tol=2.0e-14,
    ):
        raise ValueError("west and finite-node liquid densities disagree")

    current_pressure = solve_compressible_node_pressure(state, node_params)
    candidate = _unique_positive_middle_candidate(
        pressure_abs=current_pressure.pressure_abs,
        west_pressurised_foot=west_pressurised_foot,
        west_stratified_foot=west_stratified_foot,
        west_physics=west_physics,
    )
    front_distance = candidate.speed * dt
    crossing_tolerance = 64.0 * math.ulp(max(distance_to_first_branch, 1.0))
    if front_distance > distance_to_first_branch + crossing_tolerance:
        raise LaunchStepCrossesNodeBranchPoint(
            requested_distance=front_distance,
            available_distance=distance_to_first_branch,
        )

    traces = candidate_to_ale_traces(candidate, physics=west_physics)
    gas_area = west_physics.full_area - candidate.stratified_liquid_area
    gas_mass_rate = candidate.gas_density * gas_area * candidate.speed

    liquid_node = evaluate_zero_storage_t_node_at_pressure(
        liquid_characteristics,
        liquid_areas,
        node_pressure_abs=current_pressure.pressure_abs,
        liquid_density=node_params.liquid_density,
    )
    # The strict RH pressurised trace, rather than a second approximate west
    # characteristic, is the unique west liquid flux at this topology event.
    west_liquid_outward = -candidate.pressurised_discharge
    east_liquid_outward = liquid_node.branch_fluxes["east"].volume_flux
    vertical_liquid_outward = liquid_node.branch_fluxes["vertical"].volume_flux
    liquid_outward = math.fsum(
        (west_liquid_outward, east_liquid_outward, vertical_liquid_outward)
    )

    next_gas_mass = dt * gas_mass_rate
    next_liquid_inventory = (
        state.liquid_equivalent_volume - dt * liquid_outward
    )
    if next_gas_mass <= 0.0:
        raise CompressibleNodeLaunchError("strict launch supplied no gas mass")
    if next_liquid_inventory <= 0.0:
        raise CompressibleNodeLaunchError("launch step exhausts node liquid")
    next_state = CompressibleFiniteNodeState(
        gas_mass=next_gas_mass,
        liquid_equivalent_volume=next_liquid_inventory,
        node_total_volume=state.node_total_volume,
    )
    next_pressure = solve_compressible_node_pressure(next_state, node_params)

    gas_balance = next_state.gas_mass - dt * gas_mass_rate
    liquid_balance = (
        next_state.liquid_equivalent_volume
        - state.liquid_equivalent_volume
        + dt * liquid_outward
    )
    swept_volume = gas_area * front_distance
    node_gas_volume = next_pressure.gas_physical_volume
    return CompressibleNodeWestLaunch(
        state=next_state,
        current_pressure=current_pressure,
        next_pressure=next_pressure,
        front_candidate=candidate,
        front_traces=traces,
        liquid_node=liquid_node,
        front_distance=float(front_distance),
        west_gas_mass_rate_into_node=float(gas_mass_rate),
        west_liquid_volume_rate_outward=float(west_liquid_outward),
        east_liquid_volume_rate_outward=float(east_liquid_outward),
        vertical_liquid_volume_rate_outward=float(vertical_liquid_outward),
        gas_mass_balance_residual=float(gas_balance),
        liquid_inventory_balance_residual=float(liquid_balance),
        swept_gas_volume=float(swept_volume),
        node_gas_volume=float(node_gas_volume),
        geometric_volume_residual=float(node_gas_volume - swept_volume),
    )


__all__ = [
    "CompressibleNodeLaunchError",
    "CompressibleNodeWestLaunch",
    "LaunchStepCrossesNodeBranchPoint",
    "advance_compressible_node_west_launch",
]
