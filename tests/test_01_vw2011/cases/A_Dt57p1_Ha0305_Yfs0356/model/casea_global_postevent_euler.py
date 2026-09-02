"""Atomic first-order post-event Euler stage for the Case-A T network.

The local :mod:`casea_compressible_node_ssprk2` component evaluates its
second stage with frozen adjacent-branch traces.  That is useful for a node
unit test but is not a consistent network time integrator.  This module
provides the production alternative used to remove that obstruction: one
globally first-order Forward-Euler stage which

* reconstructs the west, east, and vertical node traces from the *current*
  branch states on every call;
* advances the finite compressible node once and makes its three complete
  face fluxes the sole T-junction owner;
* applies those same fluxes once to the adjacent west/east/vertical branch
  inventories;
* decomposes the node-owned vertical liquid rate into conservative gross
  upward/downward rates and advances the persistent two-stream riser; and
* applies physical gas--upstream--film drag with an equal and opposite gas
  impulse.

The operation is transactional.  All inputs are immutable and no new state
is returned until every positivity, ownership, and conservation audit has
passed.  Calling this function at a predictor state therefore recomputes all
three node traces at that predictor; no branch trace is stored in the result.

Only the T-adjacent horizontal cells are represented here.  Their supplied
``non_node_rhs`` values are the current-stage residuals from the ordinary
branch finite-volume operator with the T face omitted.  Likewise, the
vertical gas ``non_node_*_rhs`` contains every current-stage gas contribution
except the T face and interphase drag.  This separation lets the module be
inserted into the existing Case-A solver without duplicating its interior
fluxes.

There is no clock, target height, target volume, OpenFOAM field, plotting
state, or result-dependent multiplier in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from casea_compressible_finite_node import CompressibleFiniteNodeState
from casea_compressible_node_postlaunch_stage import (
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchEulerResult,
    CompressiblePostLaunchParameters,
    euler_compressible_node_postlaunch_stage,
    stratified_trace_in_outward_coordinate,
)
from casea_finite_node_qnet_owner import (
    CoordinateBranchFlux,
    required_commit_keys,
    verify_atomic_branch_commit,
)
from casea_material_front_cutcell import StratifiedFlux, StratifiedState
from casea_tjunction_shock_network import LiquidCharacteristic
from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    LiquidDonorInventories,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (
    HorizontalNodeTopology,
    LegacyMouthPathActivity,
    TwoChannelMouthCouplingPlan,
    require_exclusive_twochannel_ownership,
    stage_twochannel_mouth_coupling,
)
from casea_vertical_twostream_fv import (
    DirectionalBoundaryFlux,
    PhysicalGasInterphaseState,
    PhysicalThreeBodyDragResult,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    VerticalTwoStreamStepResult,
    advance_vertical_two_stream_fv,
    implicit_physical_three_body_drag_exchange,
)


GLOBAL_FIRST_ORDER_EULER_READY = True
MAIN_LOOP_INTEGRATED = False
REMAINING_MAIN_INTEGRATION_WORK = (
    "splice_returned_west_and_east_cells_into_the_full_horizontal_arrays",
    "supply_current_non_t_branch_residuals_from_the_main_fv_operator",
    "persist_all_four_vertical_two_stream_fields_after_breakthrough",
    "disable_legacy_G1_taylor_ccfl_and_distributed_side_source_owners",
)


class GlobalPostEventEulerError(RuntimeError):
    """A global stage failed its ownership, admissibility, or ledger audit."""


@dataclass(frozen=True)
class StratifiedStateRate:
    """Current-stage conservative RHS in a branch's global coordinate."""

    gas_mass: float = 0.0
    gas_momentum: float = 0.0
    liquid_area: float = 0.0
    liquid_discharge: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(
            self.gas_mass,
            self.gas_momentum,
            self.liquid_area,
            self.liquid_discharge,
        ):
            raise ValueError("stratified branch RHS must be finite")


@dataclass(frozen=True)
class HorizontalBoundaryCellOperator:
    """Geometry and non-T residual of one T-adjacent horizontal cell.

    ``outward_axis_sign`` converts the main solver's global horizontal
    coordinate to the branch coordinate pointing away from the node.  It is
    ``-1`` for the west cell and ``+1`` for the east cell.

    ``liquid_pressure_potential`` is the conservative hydrostatic/slot
    pressure flux evaluated from the current cell state by the caller's
    branch constitutive law.  It is not a force correction.
    """

    cell_length: float
    full_area: float
    outward_axis_sign: int
    liquid_wave_speed: float
    liquid_pressure_potential: float
    liquid_pressure_offset: float = 0.0
    liquid_loss_coefficient: float = 0.0
    non_node_rhs: StratifiedStateRate = field(default_factory=StratifiedStateRate)
    contains_t_face_flux: bool = False
    contains_distributed_side_source: bool = False

    def __post_init__(self) -> None:
        if self.outward_axis_sign not in (-1, 1):
            raise ValueError("outward_axis_sign must be exactly -1 or +1")
        if not _finite(
            self.cell_length,
            self.full_area,
            self.liquid_wave_speed,
            self.liquid_pressure_potential,
            self.liquid_pressure_offset,
            self.liquid_loss_coefficient,
        ):
            raise ValueError("horizontal boundary-cell operator must be finite")
        if min(self.cell_length, self.full_area, self.liquid_wave_speed) <= 0.0:
            raise ValueError("horizontal cell geometry and wave speed must be positive")
        if self.liquid_loss_coefficient < 0.0:
            raise ValueError("liquid loss coefficient cannot be negative")


@dataclass(frozen=True)
class VerticalGasState:
    """Conservative gas fields per unit riser length."""

    mass_per_length: tuple[float, ...]
    momentum_per_length: tuple[float, ...]

    @classmethod
    def from_iterables(
        cls,
        *,
        mass_per_length: Iterable[float],
        momentum_per_length: Iterable[float],
    ) -> "VerticalGasState":
        return cls(
            mass_per_length=_finite_tuple(
                mass_per_length, name="vertical gas mass"
            ),
            momentum_per_length=_finite_tuple(
                momentum_per_length, name="vertical gas momentum"
            ),
        )

    def __post_init__(self) -> None:
        if len(self.mass_per_length) != len(self.momentum_per_length):
            raise ValueError("vertical gas arrays need one common length")
        if not self.mass_per_length:
            raise ValueError("vertical gas state needs at least one cell")
        if not _finite(*self.mass_per_length, *self.momentum_per_length):
            raise ValueError("vertical gas state must be finite")
        if min(self.mass_per_length) <= 0.0:
            raise ValueError("post-event vertical gas mass must remain positive")

    @property
    def cell_count(self) -> int:
        return len(self.mass_per_length)


@dataclass(frozen=True)
class VerticalGasStageOperator:
    """Current-stage vertical gas residual with the T face and drag omitted."""

    non_node_mass_rhs: tuple[float, ...]
    non_node_momentum_rhs: tuple[float, ...]
    liquid_wave_speed: float
    liquid_pressure_potential: float
    liquid_pressure_offset: float = 0.0
    liquid_loss_coefficient: float = 0.0
    atmospheric_top_pressure_abs: float = 101_325.0
    apply_physical_interphase_drag: bool = True
    contains_t_face_flux: bool = False
    contains_interphase_drag: bool = False

    @classmethod
    def from_iterables(
        cls,
        *,
        non_node_mass_rhs: Iterable[float],
        non_node_momentum_rhs: Iterable[float],
        liquid_wave_speed: float,
        liquid_pressure_potential: float,
        liquid_pressure_offset: float = 0.0,
        liquid_loss_coefficient: float = 0.0,
        atmospheric_top_pressure_abs: float = 101_325.0,
        apply_physical_interphase_drag: bool = True,
        contains_t_face_flux: bool = False,
        contains_interphase_drag: bool = False,
    ) -> "VerticalGasStageOperator":
        return cls(
            non_node_mass_rhs=_finite_tuple(
                non_node_mass_rhs, name="vertical non-node gas-mass RHS"
            ),
            non_node_momentum_rhs=_finite_tuple(
                non_node_momentum_rhs,
                name="vertical non-node gas-momentum RHS",
            ),
            liquid_wave_speed=float(liquid_wave_speed),
            liquid_pressure_potential=float(liquid_pressure_potential),
            liquid_pressure_offset=float(liquid_pressure_offset),
            liquid_loss_coefficient=float(liquid_loss_coefficient),
            atmospheric_top_pressure_abs=float(atmospheric_top_pressure_abs),
            apply_physical_interphase_drag=bool(apply_physical_interphase_drag),
            contains_t_face_flux=bool(contains_t_face_flux),
            contains_interphase_drag=bool(contains_interphase_drag),
        )

    def __post_init__(self) -> None:
        if len(self.non_node_mass_rhs) != len(self.non_node_momentum_rhs):
            raise ValueError("vertical gas RHS arrays need one common length")
        if not self.non_node_mass_rhs:
            raise ValueError("vertical gas RHS needs at least one cell")
        if not _finite(
            *self.non_node_mass_rhs,
            *self.non_node_momentum_rhs,
            self.liquid_wave_speed,
            self.liquid_pressure_potential,
            self.liquid_pressure_offset,
            self.liquid_loss_coefficient,
            self.atmospheric_top_pressure_abs,
        ):
            raise ValueError("vertical gas-stage operator must be finite")
        if self.liquid_wave_speed <= 0.0:
            raise ValueError("vertical liquid wave speed must be positive")
        if self.liquid_loss_coefficient < 0.0:
            raise ValueError("vertical liquid loss coefficient cannot be negative")
        if self.atmospheric_top_pressure_abs <= 0.0:
            raise ValueError("top pressure must be positive")


@dataclass(frozen=True)
class GlobalEulerState:
    """The complete state advanced atomically by one Euler call."""

    west: StratifiedState
    east: StratifiedState
    vertical_gas: VerticalGasState
    node: CompressibleFiniteNodeState
    vertical_liquid: VerticalTwoStreamState


@dataclass(frozen=True)
class GlobalEulerLedger:
    """Network inventory and unique-face ownership audit."""

    initial_gas_mass: float
    final_gas_mass: float
    external_gas_mass_change: float
    gas_mass_residual: float
    initial_liquid_volume: float
    final_liquid_volume: float
    external_liquid_volume_change: float
    liquid_volume_residual: float
    node_t_gas_residual: float
    node_t_liquid_residual: float
    mouth_q_net_residual: float
    vertical_upward_volume_residual: float
    vertical_downward_volume_residual: float
    three_body_momentum_residual: float
    committed_keys: tuple[str, ...]


@dataclass(frozen=True)
class GlobalPostEventEulerResult:
    """Accepted global state plus all current-stage flux diagnostics."""

    state: GlobalEulerState
    node_stage: CompressiblePostLaunchEulerResult
    mouth: TwoChannelMouthCouplingPlan
    vertical_liquid_step: VerticalTwoStreamStepResult
    physical_drag: PhysicalThreeBodyDragResult | None
    node_outward_fluxes: dict[str, CoordinateBranchFlux]
    pressure_faces: tuple[float, ...]
    ledger: GlobalEulerLedger


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _finite_tuple(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not _finite(*result):
        raise ValueError(f"{name} must be a non-empty finite sequence")
    return result


def _gas_pressure_abs(
    gas_mass_per_length: float,
    gas_area: float,
    *,
    gas_sound_speed: float,
) -> float:
    if gas_mass_per_length <= 0.0 or gas_area <= 0.0:
        raise GlobalPostEventEulerError(
            "a post-event gas trace needs positive mass and void area"
        )
    pressure = gas_sound_speed**2 * gas_mass_per_length / gas_area
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise GlobalPostEventEulerError("gas EOS produced an invalid pressure")
    return float(pressure)


def _resolved_branch(
    state: StratifiedState,
    operator: HorizontalBoundaryCellOperator,
    params: CompressiblePostLaunchParameters,
) -> CompressibleNodeResolvedBranch:
    """Rebuild one node trace from the current boundary-cell state."""

    if not 0.0 < state.liquid_area < operator.full_area:
        raise GlobalPostEventEulerError(
            "T-adjacent liquid area must leave a finite gas opening"
        )
    gas_area = operator.full_area - state.liquid_area
    gas_pressure = _gas_pressure_abs(
        state.gas_mass,
        gas_area,
        gas_sound_speed=params.gas_sound_speed,
    )
    outward = stratified_trace_in_outward_coordinate(
        state, operator.outward_axis_sign
    )
    liquid_reference_pressure = gas_pressure + operator.liquid_pressure_offset
    return CompressibleNodeResolvedBranch(
        resolved=outward,
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=liquid_reference_pressure,
            reference_outward_velocity=(
                operator.outward_axis_sign
                * state.liquid_discharge
                / state.liquid_area
            ),
            wave_speed=operator.liquid_wave_speed,
            loss_coefficient=operator.liquid_loss_coefficient,
            pressure_offset=operator.liquid_pressure_offset,
        ),
        liquid_face_area=state.liquid_area,
        full_area=operator.full_area,
        reference_liquid_face_pressure_abs=liquid_reference_pressure,
        reference_liquid_pressure_potential=(
            operator.liquid_pressure_potential
        ),
    )


def _vertical_resolved_branch(
    gas: VerticalGasState,
    liquid: VerticalTwoStreamState,
    operator: VerticalGasStageOperator,
    parameters: VerticalTwoStreamParameters,
    node_parameters: CompressiblePostLaunchParameters,
) -> CompressibleNodeResolvedBranch:
    liquid_area = liquid.liquid_area[0]
    liquid_discharge = liquid.liquid_discharge[0]
    full_area = parameters.full_area
    if not 0.0 < liquid_area < full_area:
        raise GlobalPostEventEulerError(
            "vertical mouth must contain both liquid and gas after the event"
        )
    gas_area = full_area - liquid_area
    gas_pressure = _gas_pressure_abs(
        gas.mass_per_length[0],
        gas_area,
        gas_sound_speed=node_parameters.gas_sound_speed,
    )
    liquid_reference_pressure = gas_pressure + operator.liquid_pressure_offset
    resolved = StratifiedState(
        gas_mass=gas.mass_per_length[0],
        gas_momentum=gas.momentum_per_length[0],
        liquid_area=liquid_area,
        liquid_discharge=liquid_discharge,
    )
    return CompressibleNodeResolvedBranch(
        resolved=resolved,
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=liquid_reference_pressure,
            reference_outward_velocity=liquid_discharge / liquid_area,
            wave_speed=operator.liquid_wave_speed,
            loss_coefficient=operator.liquid_loss_coefficient,
            pressure_offset=operator.liquid_pressure_offset,
        ),
        liquid_face_area=liquid_area,
        full_area=full_area,
        reference_liquid_face_pressure_abs=liquid_reference_pressure,
        reference_liquid_pressure_potential=(
            operator.liquid_pressure_potential
        ),
    )


def _advance_horizontal_cell(
    state: StratifiedState,
    operator: HorizontalBoundaryCellOperator,
    outward_flux: StratifiedFlux,
    *,
    dt: float,
) -> StratifiedState:
    """Apply the shared T face once plus the branch's non-T residual."""

    scale = dt / operator.cell_length
    sign = float(operator.outward_axis_sign)
    rhs = operator.non_node_rhs
    candidate = StratifiedState(
        gas_mass=(
            state.gas_mass + dt * rhs.gas_mass + scale * outward_flux.gas_mass
        ),
        gas_momentum=(
            state.gas_momentum
            + dt * rhs.gas_momentum
            + sign * scale * outward_flux.gas_momentum
        ),
        liquid_area=(
            state.liquid_area
            + dt * rhs.liquid_area
            + scale * outward_flux.liquid_area
        ),
        liquid_discharge=(
            state.liquid_discharge
            + dt * rhs.liquid_discharge
            + sign * scale * outward_flux.liquid_momentum
        ),
    )
    if candidate.liquid_area >= operator.full_area:
        raise GlobalPostEventEulerError(
            "T-face Euler update filled a horizontal gas opening; reduce dt"
        )
    return candidate


def _current_vertical_gas_geometry(
    gas: VerticalGasState,
    liquid: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    gas_sound_speed: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return gas areas/pressures and parameter-free core/film geometry."""

    full_area = parameters.full_area
    radius = 0.5 * parameters.diameter
    gas_area: list[float] = []
    gas_pressure: list[float] = []
    up_perimeter: list[float] = []
    down_perimeter: list[float] = []
    up_diameter: list[float] = []
    down_diameter: list[float] = []
    for index, (mass, a_up, a_down) in enumerate(
        zip(
            gas.mass_per_length,
            liquid.upward_area,
            liquid.downward_area,
        )
    ):
        area_g = full_area - a_up - a_down
        if area_g <= parameters.packing_tolerance:
            raise GlobalPostEventEulerError(
                f"vertical gas void vanished in cell {index}; reduce dt"
            )
        gas_area.append(area_g)
        gas_pressure.append(
            _gas_pressure_abs(
                mass,
                area_g,
                gas_sound_speed=gas_sound_speed,
            )
        )

        # Equivalent circular upward tongue and annular falling film.  These
        # are geometric reconstructions from the evolved areas, not fitted
        # exchange multipliers.
        if a_up > parameters.dry_area_tolerance:
            r_up = math.sqrt(a_up / math.pi)
            p_up = 2.0 * math.pi * r_up
            d_up = 4.0 * a_up / p_up
        else:
            p_up = 0.0
            d_up = 0.0
        if a_down > parameters.dry_area_tolerance:
            inner_radius = math.sqrt(max(radius * radius - a_down / math.pi, 0.0))
            p_inner = 2.0 * math.pi * inner_radius
            p_wall = 2.0 * math.pi * radius
            p_wetted = p_inner + p_wall
            d_down = 4.0 * a_down / p_wetted if p_wetted > 0.0 else 0.0
        else:
            p_inner = 0.0
            d_down = 0.0
        up_perimeter.append(p_up)
        down_perimeter.append(p_inner)
        up_diameter.append(d_up)
        down_diameter.append(d_down)
    return (
        tuple(gas_area),
        tuple(gas_pressure),
        tuple(up_perimeter),
        tuple(down_perimeter),
        tuple(up_diameter),
        tuple(down_diameter),
    )


def _pressure_faces(
    *,
    node_pressure_abs: float,
    cell_pressures_abs: tuple[float, ...],
    atmospheric_top_pressure_abs: float,
) -> tuple[float, ...]:
    values = [float(node_pressure_abs)]
    values.extend(
        0.5 * (left + right)
        for left, right in zip(cell_pressures_abs[:-1], cell_pressures_abs[1:])
    )
    values.append(float(atmospheric_top_pressure_abs))
    # The node occupancy solve and the gas EOS may reconstruct the same
    # uniform pressure through different floating-point expressions.  Remove
    # only that roundoff-width spread so the well-balanced liquid operator
    # does not interpret it as a real pressure gradient and trigger a stream
    # topology change.  Any resolved hydrostatic/dynamic gradient is orders
    # of magnitude larger and is left untouched.
    pressure_scale = max(*(abs(value) for value in values), 1.0)
    if max(values) - min(values) <= 256.0 * math.ulp(pressure_scale):
        equilibrium = math.fsum(values) / len(values)
        values = [equilibrium] * len(values)
    return tuple(values)


def _advance_vertical_gas_without_drag(
    gas: VerticalGasState,
    operator: VerticalGasStageOperator,
    *,
    bottom_flux: StratifiedFlux,
    dt: float,
    dz: float,
) -> VerticalGasState:
    if gas.cell_count != len(operator.non_node_mass_rhs):
        raise ValueError("vertical gas state and RHS cell counts differ")
    mass = [
        old + dt * rhs
        for old, rhs in zip(gas.mass_per_length, operator.non_node_mass_rhs)
    ]
    momentum = [
        old + dt * rhs
        for old, rhs in zip(
            gas.momentum_per_length, operator.non_node_momentum_rhs
        )
    ]
    mass[0] += dt * bottom_flux.gas_mass / dz
    momentum[0] += dt * bottom_flux.gas_momentum / dz
    if min(mass) <= 0.0:
        raise GlobalPostEventEulerError(
            "vertical gas Euler update exhausted a cell; reduce dt"
        )
    return VerticalGasState.from_iterables(
        mass_per_length=mass,
        momentum_per_length=momentum,
    )


def _network_gas_mass(
    *,
    node: CompressibleFiniteNodeState,
    west: StratifiedState,
    east: StratifiedState,
    vertical: VerticalGasState,
    west_dx: float,
    east_dx: float,
    dz: float,
) -> float:
    return math.fsum(
        (
            node.gas_mass,
            west.gas_mass * west_dx,
            east.gas_mass * east_dx,
            dz * math.fsum(vertical.mass_per_length),
        )
    )


def _network_liquid_volume(
    *,
    node: CompressibleFiniteNodeState,
    west: StratifiedState,
    east: StratifiedState,
    vertical: VerticalTwoStreamState,
    west_dx: float,
    east_dx: float,
    dz: float,
) -> float:
    return math.fsum(
        (
            node.liquid_equivalent_volume,
            west.liquid_area * west_dx,
            east.liquid_area * east_dx,
            dz * math.fsum(vertical.liquid_area),
        )
    )


def _ledger_tolerance(*values: float) -> float:
    scale = max(*(abs(float(value)) for value in values), 1.0e-15)
    return max(4096.0 * math.ulp(scale), 2.0e-12 * scale, 1.0e-18)


def advance_casea_global_postevent_euler(
    *,
    dt: float,
    west_state: StratifiedState,
    east_state: StratifiedState,
    vertical_gas_state: VerticalGasState,
    node_state: CompressibleFiniteNodeState,
    two_stream_state: VerticalTwoStreamState,
    west_operator: HorizontalBoundaryCellOperator,
    east_operator: HorizontalBoundaryCellOperator,
    vertical_gas_operator: VerticalGasStageOperator,
    node_parameters: CompressiblePostLaunchParameters,
    two_stream_parameters: VerticalTwoStreamParameters,
    mouth_geometry: VerticalMouthGeometry,
    liquid_dynamic_viscosity: float,
    wallis: WallisCounterCurrentParameters,
    mouth_losses: DirectionalMouthLosses,
    top_liquid_boundary: DirectionalBoundaryFlux = DirectionalBoundaryFlux(),
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> GlobalPostEventEulerResult:
    """Advance one current-state, globally conservative post-event stage.

    A higher-order global integrator may call this routine once at ``U^n``
    and again at its global predictor.  Because the routine accepts conserved
    branch states rather than prebuilt traces, every call reconstructs all
    three node Riemann states and the mouth phase state from that stage.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("global Euler dt must be finite and positive")
    if west_operator.outward_axis_sign != -1:
        raise ValueError("west branch must use outward_axis_sign=-1")
    if east_operator.outward_axis_sign != 1:
        raise ValueError("east branch must use outward_axis_sign=+1")
    if vertical_gas_operator.contains_t_face_flux:
        raise GlobalPostEventEulerError(
            "vertical gas non-node RHS already contains the T face"
        )
    if vertical_gas_operator.contains_interphase_drag:
        raise GlobalPostEventEulerError(
            "vertical gas non-node RHS already contains interphase drag"
        )
    for name, operator in (("west", west_operator), ("east", east_operator)):
        if operator.contains_t_face_flux:
            raise GlobalPostEventEulerError(
                f"{name} non-node RHS already contains the T face"
            )
        if operator.contains_distributed_side_source:
            raise GlobalPostEventEulerError(
                f"{name} non-node RHS already contains the distributed T source"
            )
    require_exclusive_twochannel_ownership(legacy_activity)

    n = two_stream_parameters.cell_count
    if two_stream_state.cell_count != n or vertical_gas_state.cell_count != n:
        raise ValueError("vertical gas/liquid state and parameter cell counts differ")
    if len(vertical_gas_operator.non_node_mass_rhs) != n:
        raise ValueError("vertical gas operator and state cell counts differ")
    if not math.isclose(
        mouth_geometry.full_area,
        two_stream_parameters.full_area,
        rel_tol=2.0e-12,
        abs_tol=2.0e-14,
    ):
        raise ValueError("mouth and vertical-grid full areas disagree")
    if not math.isfinite(liquid_dynamic_viscosity) or liquid_dynamic_viscosity <= 0.0:
        raise ValueError("liquid dynamic viscosity must be finite and positive")

    west_trace = _resolved_branch(west_state, west_operator, node_parameters)
    east_trace = _resolved_branch(east_state, east_operator, node_parameters)
    vertical_trace = _vertical_resolved_branch(
        vertical_gas_state,
        two_stream_state,
        vertical_gas_operator,
        two_stream_parameters,
        node_parameters,
    )
    node_stage = euler_compressible_node_postlaunch_stage(
        node_state,
        dt,
        west=west_trace,
        east=east_trace,
        vertical=vertical_trace,
        params=node_parameters,
    )
    outward_fluxes = {
        "west": CoordinateBranchFlux.from_stratified_flux(node_stage.west),
        "east": CoordinateBranchFlux.from_stratified_flux(node_stage.east),
        "vertical": CoordinateBranchFlux.from_stratified_flux(
            node_stage.vertical
        ),
    }

    current_gas_area = (
        two_stream_parameters.full_area - two_stream_state.liquid_area[0]
    )
    current_gas_pressure = _gas_pressure_abs(
        vertical_gas_state.mass_per_length[0],
        current_gas_area,
        gas_sound_speed=node_parameters.gas_sound_speed,
    )
    current_gas_density = current_gas_pressure / node_parameters.gas_sound_speed**2
    current_gas_velocity = (
        vertical_gas_state.momentum_per_length[0]
        / vertical_gas_state.mass_per_length[0]
    )
    current_liquid_area = two_stream_state.liquid_area[0]
    current_liquid_velocity = (
        two_stream_state.liquid_discharge[0] / current_liquid_area
    )
    # A falling stream can leave through the bottom face only from the
    # bottom-cell film donor.  Using the whole riser inventory here lets the
    # algebraic mouth request a downward rate that the local FV donor limiter
    # must subsequently remove, which breaks Q_up-Q_down=q_net.  The local
    # donor volume makes the constitutive and transport capacities identical.
    riser_downward_donor_volume = (
        two_stream_parameters.cell_length
        * two_stream_state.downward_area[0]
    )
    mouth = stage_twochannel_mouth_coupling(
        node_stage.vertical.liquid_area,
        phase=VerticalMouthPhaseState(
            liquid_area=current_liquid_area,
            liquid_velocity=current_liquid_velocity,
            gas_area=current_gas_area,
            gas_velocity=current_gas_velocity,
        ),
        geometry=mouth_geometry,
        material=VerticalMouthMaterialProperties(
            liquid_density=two_stream_parameters.liquid_density,
            gas_density=current_gas_density,
            liquid_dynamic_viscosity=liquid_dynamic_viscosity,
        ),
        wallis=wallis,
        donors=LiquidDonorInventories(
            finite_node_volume=node_state.liquid_equivalent_volume,
            riser_volume=riser_downward_donor_volume,
            time_step=dt,
        ),
        losses=mouth_losses,
        # The explicit finite node is a mixing volume and owns no resolved
        # horizontal momentum.  The two horizontal branch momentum fluxes
        # above already carry the physical axial momentum across its faces.
        horizontal_axial_velocity=0.0,
        horizontal_node_topology=HorizontalNodeTopology.EXPLICIT_FINITE_NODE,
        legacy_activity=legacy_activity,
    )
    bottom = DirectionalBoundaryFlux(
        upward_rate=mouth.exchange.upward_flow,
        upward_speed=mouth.exchange.upward_channel_velocity,
        downward_rate=mouth.exchange.downward_flow,
        downward_speed=abs(mouth.exchange.downward_channel_velocity),
    )

    old_geometry = _current_vertical_gas_geometry(
        vertical_gas_state,
        two_stream_state,
        two_stream_parameters,
        gas_sound_speed=node_parameters.gas_sound_speed,
    )
    pressure_faces = _pressure_faces(
        node_pressure_abs=node_stage.pressure_abs,
        cell_pressures_abs=old_geometry[1],
        atmospheric_top_pressure_abs=(
            vertical_gas_operator.atmospheric_top_pressure_abs
        ),
    )
    vertical_step = advance_vertical_two_stream_fv(
        two_stream_state,
        two_stream_parameters,
        dt=dt,
        pressure_faces=pressure_faces,
        boundaries=VerticalTwoStreamBoundaries(
            bottom=bottom,
            top=top_liquid_boundary,
        ),
    )

    west_new = _advance_horizontal_cell(
        west_state,
        west_operator,
        node_stage.west,
        dt=dt,
    )
    east_new = _advance_horizontal_cell(
        east_state,
        east_operator,
        node_stage.east,
        dt=dt,
    )
    gas_transport = _advance_vertical_gas_without_drag(
        vertical_gas_state,
        vertical_gas_operator,
        bottom_flux=node_stage.vertical,
        dt=dt,
        dz=two_stream_parameters.cell_length,
    )

    final_liquid = vertical_step.state
    final_gas = gas_transport
    physical_drag: PhysicalThreeBodyDragResult | None = None
    if vertical_gas_operator.apply_physical_interphase_drag:
        final_geometry = _current_vertical_gas_geometry(
            gas_transport,
            final_liquid,
            two_stream_parameters,
            gas_sound_speed=node_parameters.gas_sound_speed,
        )
        dz = two_stream_parameters.cell_length
        physical_gas = PhysicalGasInterphaseState.from_iterables(
            gas_mass=(value * dz for value in gas_transport.mass_per_length),
            gas_momentum=(
                value * dz for value in gas_transport.momentum_per_length
            ),
            gas_area=final_geometry[0],
            upward_interface_perimeter=final_geometry[2],
            downward_interface_perimeter=final_geometry[3],
            upward_hydraulic_diameter=final_geometry[4],
            downward_hydraulic_diameter=final_geometry[5],
        )
        physical_drag = implicit_physical_three_body_drag_exchange(
            final_liquid,
            two_stream_parameters,
            physical_gas,
            dt=dt,
        )
        final_liquid = physical_drag.state
        final_gas = VerticalGasState.from_iterables(
            mass_per_length=gas_transport.mass_per_length,
            momentum_per_length=(
                value / dz for value in physical_drag.gas_momentum
            ),
        )

    committed_keys = tuple(sorted(required_commit_keys()))
    verify_atomic_branch_commit(committed_keys)

    dz = two_stream_parameters.cell_length
    initial_gas = _network_gas_mass(
        node=node_state,
        west=west_state,
        east=east_state,
        vertical=vertical_gas_state,
        west_dx=west_operator.cell_length,
        east_dx=east_operator.cell_length,
        dz=dz,
    )
    final_gas_mass = _network_gas_mass(
        node=node_stage.node.state,
        west=west_new,
        east=east_new,
        vertical=final_gas,
        west_dx=west_operator.cell_length,
        east_dx=east_operator.cell_length,
        dz=dz,
    )
    external_gas = dt * math.fsum(
        (
            west_operator.non_node_rhs.gas_mass * west_operator.cell_length,
            east_operator.non_node_rhs.gas_mass * east_operator.cell_length,
            dz * math.fsum(vertical_gas_operator.non_node_mass_rhs),
        )
    )
    gas_residual = final_gas_mass - initial_gas - external_gas

    initial_liquid = _network_liquid_volume(
        node=node_state,
        west=west_state,
        east=east_state,
        vertical=two_stream_state,
        west_dx=west_operator.cell_length,
        east_dx=east_operator.cell_length,
        dz=dz,
    )
    final_liquid_volume = _network_liquid_volume(
        node=node_stage.node.state,
        west=west_new,
        east=east_new,
        vertical=final_liquid,
        west_dx=west_operator.cell_length,
        east_dx=east_operator.cell_length,
        dz=dz,
    )
    external_liquid = dt * math.fsum(
        (
            west_operator.non_node_rhs.liquid_area * west_operator.cell_length,
            east_operator.non_node_rhs.liquid_area * east_operator.cell_length,
            -top_liquid_boundary.net_rate,
        )
    )
    liquid_residual = final_liquid_volume - initial_liquid - external_liquid

    node_t_gas = math.fsum(
        (
            node_stage.node.state.gas_mass - node_state.gas_mass,
            dt * node_stage.west.gas_mass,
            dt * node_stage.east.gas_mass,
            dt * node_stage.vertical.gas_mass,
        )
    )
    node_t_liquid = math.fsum(
        (
            node_stage.node.state.liquid_equivalent_volume
            - node_state.liquid_equivalent_volume,
            dt * node_stage.west.liquid_area,
            dt * node_stage.east.liquid_area,
            dt * node_stage.vertical.liquid_area,
        )
    )
    mouth_residual = math.fsum(
        (
            mouth.exchange.upward_flow,
            -mouth.exchange.downward_flow,
            -node_stage.vertical.liquid_area,
        )
    )
    drag_residual = (
        0.0
        if physical_drag is None
        else physical_drag.total_momentum_residual
    )

    residuals = (
        gas_residual,
        liquid_residual,
        node_t_gas,
        node_t_liquid,
        mouth_residual,
        vertical_step.ledger.upward_volume_residual,
        vertical_step.ledger.downward_volume_residual,
        drag_residual,
    )
    tolerance = _ledger_tolerance(
        initial_gas,
        final_gas_mass,
        initial_liquid,
        final_liquid_volume,
        dt * mouth.exchange.upward_flow,
        dt * mouth.exchange.downward_flow,
    )
    if any(abs(value) > tolerance for value in residuals):
        raise GlobalPostEventEulerError(
            "global post-event Euler conservation ledger did not close: "
            + ", ".join(f"{value:.12g}" for value in residuals)
            + f"; liquid(initial={initial_liquid:.12g}, "
            f"final={final_liquid_volume:.12g}, "
            f"external={external_liquid:.12g})"
        )

    state = GlobalEulerState(
        west=west_new,
        east=east_new,
        vertical_gas=final_gas,
        node=node_stage.node.state,
        vertical_liquid=final_liquid,
    )
    ledger = GlobalEulerLedger(
        initial_gas_mass=initial_gas,
        final_gas_mass=final_gas_mass,
        external_gas_mass_change=external_gas,
        gas_mass_residual=gas_residual,
        initial_liquid_volume=initial_liquid,
        final_liquid_volume=final_liquid_volume,
        external_liquid_volume_change=external_liquid,
        liquid_volume_residual=liquid_residual,
        node_t_gas_residual=node_t_gas,
        node_t_liquid_residual=node_t_liquid,
        mouth_q_net_residual=mouth_residual,
        vertical_upward_volume_residual=(
            vertical_step.ledger.upward_volume_residual
        ),
        vertical_downward_volume_residual=(
            vertical_step.ledger.downward_volume_residual
        ),
        three_body_momentum_residual=drag_residual,
        committed_keys=committed_keys,
    )
    return GlobalPostEventEulerResult(
        state=state,
        node_stage=node_stage,
        mouth=mouth,
        vertical_liquid_step=vertical_step,
        physical_drag=physical_drag,
        node_outward_fluxes=outward_fluxes,
        pressure_faces=pressure_faces,
        ledger=ledger,
    )


__all__ = [
    "GLOBAL_FIRST_ORDER_EULER_READY",
    "MAIN_LOOP_INTEGRATED",
    "REMAINING_MAIN_INTEGRATION_WORK",
    "GlobalEulerLedger",
    "GlobalEulerState",
    "GlobalPostEventEulerError",
    "GlobalPostEventEulerResult",
    "HorizontalBoundaryCellOperator",
    "StratifiedStateRate",
    "VerticalGasStageOperator",
    "VerticalGasState",
    "advance_casea_global_postevent_euler",
]
