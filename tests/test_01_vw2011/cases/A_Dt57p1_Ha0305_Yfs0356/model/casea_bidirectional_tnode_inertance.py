"""Independent net/circulation inertance closure for the Case-A T mouth.

This module replaces the algebraic identification of counter-current
circulation with a capacity.  Two fluxes persist between time steps:

``q_net``
    signed liquid flux from the horizontal footprint into the riser;

``q_c``
    non-negative equal-and-opposite circulation component.

The gross rates are identities, not fitted decompositions,

``q_up = max(q_net, 0) + q_c`` and
``q_down = max(-q_net, 0) + q_c``.

``q_net`` follows the finite-footprint pressure/inertance and directional
turn-loss equation.  ``q_c`` follows a separate closed-loop half-cell
generalized momentum equation whose inertia contains both the upward core and
the downward film.  Its source terms use the current persistent mouth stream
areas/discharges, the dynamically reconstructed annular-film thickness,
laminar/turbulent wall shear, turn and counter-current mixing losses, and the
difference between resolved gas shear on the upward liquid core and on the
falling film.  Gravity does not drive the equal-and-opposite circulation:
its work on equal upward and downward volume rates over the same half-cell
cancels exactly.  Hydrostatic gravity acting on a one-way net flow remains in
the resolved pressure difference used by ``q_net``.

The Nusselt relation is reported only as a low-Re equilibrium audit; it is
never assigned to ``q_c`` and is never an instantaneous cap.  Wallis flooding
and finite donor inventories are upper inequalities.  When active, their
reaction pressures and complementarity products are explicit in the ledger.

No clock, target water volume, 2-D field, or rendered result enters the API.
The module is isolated and does not import or modify the production main loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from casea_distributed_tnode_inertance import (
    DistributedTNodeGeometry,
    DistributedTNodePressureState,
)
from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    LiquidDonorInventories,
    TwoChannelMouthResult,
    VerticalMouthMaterialProperties,
    WallisCounterCurrentParameters,
)
from casea_vertical_mouth_twochannel_integration import (
    HorizontalNodeTopology,
    LegacyMouthPathActivity,
    TwoChannelMouthCouplingPlan,
    TwoLiquidMomentumBoundaryResidual,
    require_exclusive_twochannel_ownership,
)


class BidirectionalTNodeError(RuntimeError):
    """A bidirectional node state or conservative step was rejected."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class BidirectionalTNodeState:
    """Persistent signed net flux and non-negative circulation flux."""

    q_net: float
    circulation_flow: float

    def __post_init__(self) -> None:
        if not _finite(self.q_net, self.circulation_flow):
            raise ValueError("bidirectional T-node state must be finite")
        if self.circulation_flow < 0.0:
            raise ValueError("circulation flow cannot be negative")

    @property
    def upward_flow(self) -> float:
        return max(self.q_net, 0.0) + self.circulation_flow

    @property
    def downward_flow(self) -> float:
        return max(-self.q_net, 0.0) + self.circulation_flow

    @property
    def q_c(self) -> float:
        """Short, equation-facing alias for the persistent circulation flux."""

        return self.circulation_flow

    @classmethod
    def from_gross_flows(
        cls,
        *,
        upward_flow: float,
        downward_flow: float,
    ) -> "BidirectionalTNodeState":
        upward = float(upward_flow)
        downward = float(downward_flow)
        if not _finite(upward, downward) or min(upward, downward) < 0.0:
            raise ValueError("gross mouth flows must be finite and non-negative")
        return cls(
            q_net=upward - downward,
            circulation_flow=min(upward, downward),
        )

    @classmethod
    def from_persistent_trace(
        cls,
        trace: "PersistentMouthTrace",
    ) -> "BidirectionalTNodeState":
        """Initialize the inertial state from resolved directional face fluxes."""

        upward = float(trace.upward_discharge)
        downward = float(-trace.downward_discharge)
        if not _finite(upward, downward) or min(upward, downward) < 0.0:
            raise ValueError("persistent trace violates directional liquid flux signs")
        return cls.from_gross_flows(
            upward_flow=upward,
            downward_flow=downward,
        )


@dataclass(frozen=True)
class PersistentMouthTrace:
    """Current first-riser-cell two-liquid state and resolved gas face flux.

    Liquid discharges use the upward coordinate: ``Q_up >= 0`` and
    ``Q_down <= 0``.  ``gas_mass_flow`` is the resolved mouth-face mass rate,
    positive upward, and is used by Wallis.  ``gas_cell_mass`` and
    ``gas_cell_momentum`` are the actual first-riser-cell conservative state;
    their ratio, not ``rho * A_mouth * dz``, supplies the old velocity and
    finite inertia for the implicit interphase exchange.
    """

    upward_area: float
    upward_discharge: float
    downward_area: float
    downward_discharge: float
    gas_area: float
    gas_mass_flow: float
    gas_cell_mass: float
    gas_cell_momentum: float

    def validate(self, geometry: DistributedTNodeGeometry) -> None:
        values = (
            self.upward_area,
            self.upward_discharge,
            self.downward_area,
            self.downward_discharge,
            self.gas_area,
            self.gas_mass_flow,
            self.gas_cell_mass,
            self.gas_cell_momentum,
        )
        if not _finite(*values):
            raise ValueError("persistent mouth trace must be finite")
        if min(self.upward_area, self.downward_area, self.gas_area) < 0.0:
            raise ValueError("mouth phase areas cannot be negative")
        if self.gas_cell_mass < 0.0:
            raise ValueError("resolved first-cell gas mass cannot be negative")
        if self.gas_cell_mass == 0.0 and self.gas_cell_momentum != 0.0:
            raise ValueError("zero-mass first gas cell carries momentum")
        if self.upward_discharge < 0.0 or self.downward_discharge > 0.0:
            raise ValueError("mouth liquid discharges violate directional labels")
        if self.upward_area == 0.0 and self.upward_discharge != 0.0:
            raise ValueError("dry upward mouth stream carries discharge")
        if self.downward_area == 0.0 and self.downward_discharge != 0.0:
            raise ValueError("dry downward mouth stream carries discharge")
        tolerance = 512.0 * math.ulp(max(geometry.mouth_area, 1.0))
        occupied = self.upward_area + self.downward_area + self.gas_area
        if not math.isclose(
            occupied,
            geometry.mouth_area,
            rel_tol=1.0e-10,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "upward liquid, downward film, and gas must partition the mouth"
            )

    @property
    def upward_velocity(self) -> float:
        return (
            self.upward_discharge / self.upward_area
            if self.upward_area > 0.0
            else 0.0
        )

    @property
    def downward_velocity(self) -> float:
        return (
            self.downward_discharge / self.downward_area
            if self.downward_area > 0.0
            else 0.0
        )

    @property
    def gas_cell_velocity(self) -> float:
        return (
            self.gas_cell_momentum / self.gas_cell_mass
            if self.gas_cell_mass > 0.0
            else 0.0
        )


@dataclass(frozen=True)
class BidirectionalTNodeParameters:
    """Physical half-cell, gas, and transition parameters."""

    riser_cell_length: float
    gas_dynamic_viscosity: float = 1.81e-5
    laminar_film_reynolds_limit: float = 1600.0
    turbulent_film_reynolds_limit: float = 3000.0
    dry_area_tolerance: float = 1.0e-14
    root_relative_tolerance: float = 1.0e-12
    root_absolute_tolerance: float = 1.0e-16
    root_pressure_tolerance: float = 1.0e-10
    max_root_iterations: int = 160

    def __post_init__(self) -> None:
        values = (
            self.riser_cell_length,
            self.gas_dynamic_viscosity,
            self.laminar_film_reynolds_limit,
            self.turbulent_film_reynolds_limit,
            self.dry_area_tolerance,
            self.root_relative_tolerance,
            self.root_absolute_tolerance,
            self.root_pressure_tolerance,
        )
        if not _finite(*values) or min(values) <= 0.0:
            raise ValueError("bidirectional T-node parameters must be positive")
        if self.turbulent_film_reynolds_limit <= self.laminar_film_reynolds_limit:
            raise ValueError("turbulent transition must exceed laminar transition")
        if self.max_root_iterations < 1:
            raise ValueError("at least one scalar root iteration is required")

    @property
    def half_cell_length(self) -> float:
        return 0.5 * self.riser_cell_length


@dataclass(frozen=True)
class FilmGeometry:
    area: float
    thickness: float
    inner_radius: float
    wall_perimeter: float
    interface_perimeter: float
    hydraulic_diameter: float


@dataclass(frozen=True)
class GasLiquidInterfaceAction:
    """Resolved gas shear on both liquid interfaces at the mouth.

    Pressures and forces use the upward coordinate.  The circulation-driving
    generalized pressure is ``p_on_upward_core - p_on_falling_film`` because
    the two liquid velocities associated with ``q_c`` have opposite signs.
    """

    gas_velocity: float
    gas_hydraulic_diameter: float
    upward_core_interface_perimeter: float
    falling_film_interface_perimeter: float
    upward_core_reynolds: float
    falling_film_reynolds: float
    upward_core_shear_upward: float
    falling_film_shear_upward: float
    upward_core_force_conductance: float
    falling_film_force_conductance: float
    upward_core_pressure_upward: float
    falling_film_pressure_upward: float
    circulation_drive_pressure: float
    upward_core_force_upward: float
    falling_film_force_upward: float


@dataclass(frozen=True)
class ImplicitGasLiquidExchange:
    """Backward-Euler gas/interface exchange for one candidate ``q_c``."""

    action: GasLiquidInterfaceAction
    gas_velocity_before: float
    gas_velocity_after: float
    gas_cell_mass: float
    gas_momentum_change: float
    gas_momentum_residual: float
    upward_core_slip_before: float
    upward_core_slip_after: float
    falling_film_slip_before: float
    falling_film_slip_after: float
    dissipation_energy: float


@dataclass(frozen=True)
class _CirculationSolve:
    base_up: float
    base_down: float
    node_capacity: float
    riser_capacity: float
    wallis_capacity: float
    upper_bound: float
    effective_area: float
    pressure_inertance: float
    q_free: float
    q_new: float
    inertive_pressure: float
    total_loss_pressure: float
    drive_pressure: float
    wall_pressure: float
    upward_turn_pressure: float
    downward_turn_pressure: float
    mixing_pressure: float
    film_reynolds: float
    lower_reaction: float
    upper_reaction: float
    upper_owner: str
    wallis_reaction: float
    node_reaction: float
    riser_reaction: float
    pressure_residual: float
    lower_gap: float
    upper_gap: float
    lower_product: float
    upper_product: float
    gas_exchange: ImplicitGasLiquidExchange


@dataclass(frozen=True)
class NetFluxLedger:
    old_q_net: float
    unconstrained_q_net: float
    accepted_q_net: float
    hydraulic_driving_pressure: float
    gas_interphase_drive_pressure: float
    driving_pressure: float
    pressure_inertance: float
    inertive_pressure: float
    turn_loss_pressure: float
    lower_bound: float
    upper_bound: float
    lower_reaction_pressure: float
    upper_reaction_pressure: float
    lower_bound_owner: str
    upper_bound_owner: str
    pressure_residual: float
    lower_gap: float
    upper_gap: float
    lower_complementarity_product: float
    upper_complementarity_product: float
    actual_momentum_change: float
    expected_momentum_change: float
    momentum_residual: float
    liquid_interphase_impulse_upward: float
    gas_interphase_impulse_upward: float
    interphase_momentum_residual: float


@dataclass(frozen=True)
class CirculationLedger:
    old_circulation_flow: float
    unconstrained_circulation_flow: float
    accepted_circulation_flow: float
    film: FilmGeometry
    film_reynolds: float
    wall_regime_blend: float
    circulation_effective_area: float
    circulation_pressure_inertance: float
    inertive_pressure: float
    buoyant_gravity_pressure: float
    upward_core_gravity_pressure_downward: float
    falling_film_gravity_pressure_downward: float
    net_circulation_gravity_pressure: float
    wall_loss_pressure: float
    upward_turn_loss_pressure: float
    downward_turn_loss_pressure: float
    countercurrent_mixing_loss_pressure: float
    gas_upward_core_pressure_upward: float
    gas_film_pressure_upward: float
    gas_drive_pressure_downward: float
    gas_circulation_drive_pressure: float
    horizontal_phase_pressure_drive: float
    gas_superficial_velocity: float
    gas_velocity_before_exchange: float
    gas_velocity_after_exchange: float
    gas_cell_mass: float
    upward_core_gas_conductance: float
    falling_film_gas_conductance: float
    upward_core_gas_slip_before: float
    upward_core_gas_slip_after: float
    falling_film_gas_slip_before: float
    falling_film_gas_slip_after: float
    interphase_dissipation_energy: float
    gas_momentum_change: float
    gas_momentum_residual: float
    trace_net_flow: float
    trace_circulation_flow: float
    state_trace_net_mismatch: float
    state_trace_circulation_mismatch: float
    wallis_downward_capacity: float
    wallis_active: bool
    node_circulation_capacity: float
    riser_circulation_capacity: float
    wallis_circulation_capacity: float
    upper_circulation_bound: float
    upper_bound_owner: str
    lower_reaction_pressure: float
    wallis_reaction_pressure: float
    node_donor_reaction_pressure: float
    riser_donor_reaction_pressure: float
    pressure_residual: float
    lower_gap: float
    upper_gap: float
    lower_complementarity_product: float
    upper_complementarity_product: float
    nusselt_applicable: bool
    nusselt_equilibrium_flow: float
    nusselt_thin_film_flow: float
    nusselt_flow_residual: float
    nusselt_force_residual: float
    actual_circulation_momentum_change: float
    expected_circulation_momentum_change: float
    circulation_momentum_residual: float
    upward_core_gas_impulse_upward: float
    falling_film_gas_impulse_upward: float
    liquid_gas_impulse_upward: float
    gas_reaction_impulse_upward: float
    gas_liquid_momentum_residual: float

    @property
    def actual_film_momentum_change(self) -> float:
        """Compatibility alias for the reduced circulation momentum."""

        return self.actual_circulation_momentum_change

    @property
    def expected_film_momentum_change(self) -> float:
        return self.expected_circulation_momentum_change

    @property
    def film_momentum_residual(self) -> float:
        return self.circulation_momentum_residual


@dataclass(frozen=True)
class BidirectionalVolumeLedger:
    node_volume_before: float
    node_volume_after: float
    riser_donor_volume_before: float
    riser_donor_volume_after: float
    upward_gross_volume: float
    downward_gross_volume: float
    node_volume_change: float
    riser_volume_change: float
    combined_volume_residual: float


@dataclass(frozen=True)
class BidirectionalTNodeStepResult:
    state: BidirectionalTNodeState
    upward_flow: float
    downward_flow: float
    upward_velocity: float
    downward_velocity: float
    net: NetFluxLedger
    circulation: CirculationLedger
    volume: BidirectionalVolumeLedger

    @property
    def closure_residual(self) -> float:
        return math.fsum((self.upward_flow, -self.downward_flow, -self.state.q_net))

    @property
    def q_net(self) -> float:
        """Accepted signed node-to-riser liquid flux."""

        return self.state.q_net

    @property
    def circulation_flow(self) -> float:
        return self.state.circulation_flow


def dynamic_annular_film_geometry(
    falling_film_area: float,
    *,
    geometry: DistributedTNodeGeometry,
) -> FilmGeometry:
    """Reconstruct the current annular film directly from its inventory."""

    area = float(falling_film_area)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("falling-film area must be finite and non-negative")
    bore_area = float(geometry.mouth_area)
    packing_roundoff = 128.0 * math.ulp(max(area, bore_area))
    if area > bore_area:
        if area <= bore_area + packing_roundoff:
            area = bore_area
        else:
            raise ValueError(
                "falling-film area exceeds the riser bore: "
                f"area={area:.16g}, bore={bore_area:.16g}, "
                f"excess={area - bore_area:.16g}"
            )
    radius = 0.5 * geometry.riser_diameter
    inner = math.sqrt(max(radius * radius - area / math.pi, 0.0))
    thickness = radius - inner
    wall = 2.0 * math.pi * radius if area > 0.0 else 0.0
    interface = 2.0 * math.pi * inner if area > 0.0 and inner > 0.0 else 0.0
    wetted = wall + interface
    hydraulic = 4.0 * area / wetted if wetted > 0.0 else 0.0
    return FilmGeometry(
        area=area,
        thickness=float(thickness),
        inner_radius=float(inner),
        wall_perimeter=float(wall),
        interface_perimeter=float(interface),
        hydraulic_diameter=float(hydraulic),
    )


def _wallis_capacity(
    trace: PersistentMouthTrace,
    *,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
) -> tuple[bool, float, float]:
    if trace.gas_area <= 0.0 or trace.gas_mass_flow <= 0.0:
        return False, 0.0, math.inf
    gas_superficial_velocity = trace.gas_mass_flow / (
        material.gas_density * geometry.mouth_area
    )
    density_difference = material.liquid_density - material.gas_density
    jg_star = gas_superficial_velocity * math.sqrt(
        material.gas_density
        / (geometry.gravity * geometry.riser_diameter * density_difference)
    )
    remaining = max(wallis.constant - math.sqrt(max(jg_star, 0.0)), 0.0)
    jl_star = (remaining / wallis.slope) ** 2
    liquid_velocity_scale = math.sqrt(
        geometry.gravity
        * geometry.riser_diameter
        * density_difference
        / material.liquid_density
    )
    capacity = jl_star * liquid_velocity_scale * geometry.mouth_area
    return True, float(gas_superficial_velocity), float(max(capacity, 0.0))


def _rationalized_quadratic_root(linear: float, quadratic: float, rhs: float) -> float:
    magnitude = abs(rhs)
    if quadratic <= 0.0:
        root = magnitude / linear
    else:
        root = 2.0 * magnitude / (
            linear + math.sqrt(linear * linear + 4.0 * quadratic * magnitude)
        )
    return math.copysign(root, rhs)


def _directional_net_pressure_state(
    state: BidirectionalTNodeState,
    pressure: DistributedTNodePressureState,
    geometry: DistributedTNodeGeometry,
) -> DistributedTNodePressureState:
    """Return the common-mode pressure seen by the signed net coordinate.

    At a two-phase side opening the upward liquid characteristic is connected
    to the horizontal liquid pressure, whereas the falling-film receiver is
    connected to the horizontal gas-contact pressure.  Their difference does
    closed-loop work and is handled by ``q_c``.  Area-averaging the same two
    pressures into ``q_net`` double-counts that work and converts a falling
    film directly into a rising core when the signed coordinate crosses zero.

    A moving positive/negative net state therefore keeps its corresponding
    phase pressure.  At rest, a vertical pressure lying between the two
    horizontal phase pressures is a valid nonsmooth equilibrium for ``q_net``;
    only circulation is driven.  If both horizontal phases lie on one side of
    the vertical pressure, their nearest common envelope drives net flow.
    """

    pressure.validate(geometry)
    gas_pressure = float(pressure.horizontal_gas_pressure_abs)
    liquid_pressure = float(pressure.horizontal_liquid_pressure_abs)
    vertical_pressure = float(pressure.vertical_mouth_pressure_abs)
    flux_tolerance = 512.0 * math.ulp(
        max(abs(state.q_net), 1.0e-12)
    )
    if state.q_net > flux_tolerance:
        common_pressure = liquid_pressure
    elif state.q_net < -flux_tolerance:
        common_pressure = gas_pressure
    else:
        lower = min(gas_pressure, liquid_pressure)
        upper = max(gas_pressure, liquid_pressure)
        if vertical_pressure < lower:
            common_pressure = lower
        elif vertical_pressure > upper:
            common_pressure = upper
        else:
            common_pressure = vertical_pressure
    return replace(
        pressure,
        horizontal_gas_pressure_abs=float(common_pressure),
        horizontal_liquid_pressure_abs=float(common_pressure),
    )


def _advance_net_flux(
    q_old: float,
    *,
    dt: float,
    pressure: DistributedTNodePressureState,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    losses: DirectionalMouthLosses,
    node_rate_capacity: float,
    riser_rate_capacity: float,
    wallis_active: bool,
    wallis_downward_capacity: float,
    riser_receiving_net_rate_capacity: float = math.inf,
    gas_interphase_drive_pressure: float = 0.0,
) -> tuple[float, NetFluxLedger]:
    rho = material.liquid_density
    area = geometry.mouth_area
    length = geometry.effective_inertance_length
    inertance = rho * length / area
    horizontal_pressure = pressure.horizontal_contact_pressure(geometry)
    hydraulic_drive = horizontal_pressure - pressure.vertical_mouth_pressure_abs
    gas_drive = float(gas_interphase_drive_pressure)
    if not math.isfinite(gas_drive):
        raise ValueError("gas interphase drive pressure must be finite")
    drive = hydraulic_drive + gas_drive
    linear = inertance / dt
    rhs = drive + linear * q_old
    coefficient = losses.upward_turn if rhs >= 0.0 else losses.downward_turn
    quadratic = 0.5 * rho * coefficient / area**2
    q_free = _rationalized_quadratic_root(linear, quadratic, rhs)

    physical_down_capacity = (
        wallis_downward_capacity if wallis_active else math.inf
    )
    downward_capacity = min(riser_rate_capacity, physical_down_capacity)
    lower = -downward_capacity
    upper = min(node_rate_capacity, riser_receiving_net_rate_capacity)
    if upper < lower:
        raise BidirectionalTNodeError(
            "riser receiving capacity and downward donor capacity form an "
            "empty net-flux interval"
        )
    q_new = min(max(q_free, lower), upper)
    turn_coefficient = losses.upward_turn if q_new >= 0.0 else losses.downward_turn
    velocity = q_new / area
    turn_pressure = 0.5 * rho * turn_coefficient * velocity * abs(velocity)
    inertive_pressure = inertance * (q_new - q_old) / dt
    raw_residual = inertive_pressure + turn_pressure - drive
    tolerance = max(
        1.0e-16,
        512.0
        * math.ulp(
            max(abs(q_free), abs(q_new), abs(lower), abs(upper), 1.0e-12)
        ),
    )
    lower_reaction = 0.0
    upper_reaction = 0.0
    lower_owner = "inactive"
    upper_owner = "inactive"
    if q_free < lower - tolerance:
        lower_reaction = max(raw_residual, 0.0)
        if wallis_active and wallis_downward_capacity <= riser_rate_capacity:
            lower_owner = "wallis"
        else:
            lower_owner = "riser_donor"
    elif q_free > upper + tolerance:
        upper_reaction = max(-raw_residual, 0.0)
        upper_owner = (
            "riser_receiver"
            if riser_receiving_net_rate_capacity
            <= node_rate_capacity + tolerance
            else "node_donor"
        )
    residual = math.fsum(
        (inertive_pressure, turn_pressure, -drive, -lower_reaction, upper_reaction)
    )
    lower_gap = q_new - lower
    upper_gap = upper - q_new
    lower_product = lower_reaction * lower_gap
    upper_product = upper_reaction * upper_gap
    old_momentum = rho * length * q_old
    new_momentum = rho * length * q_new
    actual_momentum = new_momentum - old_momentum
    expected_momentum = area * dt * math.fsum(
        (drive, -turn_pressure, lower_reaction, -upper_reaction)
    )
    momentum_residual = actual_momentum - expected_momentum
    return q_new, NetFluxLedger(
        old_q_net=float(q_old),
        unconstrained_q_net=float(q_free),
        accepted_q_net=float(q_new),
        hydraulic_driving_pressure=float(hydraulic_drive),
        gas_interphase_drive_pressure=float(gas_drive),
        driving_pressure=float(drive),
        pressure_inertance=float(inertance),
        inertive_pressure=float(inertive_pressure),
        turn_loss_pressure=float(turn_pressure),
        lower_bound=float(lower),
        upper_bound=float(upper),
        lower_reaction_pressure=float(lower_reaction),
        upper_reaction_pressure=float(upper_reaction),
        lower_bound_owner=lower_owner,
        upper_bound_owner=upper_owner,
        pressure_residual=float(residual),
        lower_gap=float(lower_gap),
        upper_gap=float(upper_gap),
        lower_complementarity_product=float(lower_product),
        upper_complementarity_product=float(upper_product),
        actual_momentum_change=float(actual_momentum),
        expected_momentum_change=float(expected_momentum),
        momentum_residual=float(momentum_residual),
        liquid_interphase_impulse_upward=float(area * dt * gas_drive),
        gas_interphase_impulse_upward=float(-area * dt * gas_drive),
        interphase_momentum_residual=0.0,
    )


def _smoothstep(value: float) -> float:
    x = min(max(float(value), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _film_wall_loss_pressure(
    q_down: float,
    film: FilmGeometry,
    *,
    half_length: float,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    parameters: BidirectionalTNodeParameters,
) -> tuple[float, float, float, float]:
    if q_down <= 0.0 or film.area <= 0.0 or film.thickness <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    velocity = q_down / film.area
    film_re = (
        4.0
        * material.liquid_density
        * q_down
        / (math.pi * geometry.riser_diameter * material.liquid_dynamic_viscosity)
    )
    laminar_pressure = (
        3.0
        * material.liquid_dynamic_viscosity
        * half_length
        * velocity
        / film.thickness**2
    )
    hydraulic_re = (
        material.liquid_density
        * velocity
        * max(film.hydraulic_diameter, 1.0e-30)
        / material.liquid_dynamic_viscosity
    )
    darcy_factor = (
        0.3164 / max(hydraulic_re, 1.0) ** 0.25
    )
    turbulent_pressure = (
        0.5
        * material.liquid_density
        * darcy_factor
        * half_length
        / max(film.hydraulic_diameter, 1.0e-30)
        * velocity**2
    )
    blend = _smoothstep(
        (
            film_re - parameters.laminar_film_reynolds_limit
        )
        / (
            parameters.turbulent_film_reynolds_limit
            - parameters.laminar_film_reynolds_limit
        )
    )
    pressure = (1.0 - blend) * laminar_pressure + blend * turbulent_pressure
    return float(pressure), float(film_re), float(blend), float(laminar_pressure)


def _gas_liquid_interface_action(
    trace: PersistentMouthTrace,
    film: FilmGeometry,
    *,
    half_length: float,
    material: VerticalMouthMaterialProperties,
    parameters: BidirectionalTNodeParameters,
) -> GasLiquidInterfaceAction:
    """Resolve signed gas shear on the upward core and falling film.

    This is a two-interface action/reaction calculation.  Using only the
    upward gas shear on the falling film would correctly inhibit downward
    liquid, but would omit the gas work on the upward liquid core and would
    not form the generalized force conjugate to the equal counterflow.
    """

    upward_radius = (
        math.sqrt(trace.upward_area / math.pi)
        if trace.upward_area > 0.0
        else 0.0
    )
    upward_perimeter = 2.0 * math.pi * upward_radius
    falling_perimeter = film.interface_perimeter
    wetted_gas_perimeter = upward_perimeter + falling_perimeter
    if trace.gas_area <= 0.0 or wetted_gas_perimeter <= 0.0:
        return GasLiquidInterfaceAction(
            gas_velocity=0.0,
            gas_hydraulic_diameter=0.0,
            upward_core_interface_perimeter=float(upward_perimeter),
            falling_film_interface_perimeter=float(falling_perimeter),
            upward_core_reynolds=0.0,
            falling_film_reynolds=0.0,
            upward_core_shear_upward=0.0,
            falling_film_shear_upward=0.0,
            upward_core_force_conductance=0.0,
            falling_film_force_conductance=0.0,
            upward_core_pressure_upward=0.0,
            falling_film_pressure_upward=0.0,
            circulation_drive_pressure=0.0,
            upward_core_force_upward=0.0,
            falling_film_force_upward=0.0,
        )
    gas_velocity = trace.gas_cell_velocity
    gas_hydraulic = 4.0 * trace.gas_area / wetted_gas_perimeter

    def interface_shear(liquid_velocity_upward: float) -> tuple[float, float]:
        slip = gas_velocity - liquid_velocity_upward
        reynolds = (
            material.gas_density
            * abs(slip)
            * gas_hydraulic
            / parameters.gas_dynamic_viscosity
        )
        if reynolds <= 0.0:
            return 0.0, 0.0
        darcy = (
            64.0 / reynolds
            if reynolds < 2300.0
            else 0.3164 / reynolds**0.25
        )
        shear = 0.125 * darcy * material.gas_density * slip * abs(slip)
        return float(reynolds), float(shear)

    upward_re, upward_shear = interface_shear(trace.upward_velocity)
    falling_re, falling_shear = interface_shear(trace.downward_velocity)
    upward_force = upward_shear * upward_perimeter * half_length
    falling_force = falling_shear * falling_perimeter * half_length
    upward_slip = gas_velocity - trace.upward_velocity
    falling_slip = gas_velocity - trace.downward_velocity
    laminar_upward_conductance = (
        8.0
        * parameters.gas_dynamic_viscosity
        * upward_perimeter
        * half_length
        / max(gas_hydraulic, 1.0e-30)
    )
    laminar_falling_conductance = (
        8.0
        * parameters.gas_dynamic_viscosity
        * falling_perimeter
        * half_length
        / max(gas_hydraulic, 1.0e-30)
    )
    upward_conductance = (
        upward_force / upward_slip
        if abs(upward_slip) > 1.0e-14
        else laminar_upward_conductance
    )
    falling_conductance = (
        falling_force / falling_slip
        if abs(falling_slip) > 1.0e-14
        else laminar_falling_conductance
    )
    upward_pressure = (
        upward_force / trace.upward_area if trace.upward_area > 0.0 else 0.0
    )
    falling_pressure = (
        falling_force / film.area if film.area > 0.0 else 0.0
    )
    return GasLiquidInterfaceAction(
        gas_velocity=float(gas_velocity),
        gas_hydraulic_diameter=float(gas_hydraulic),
        upward_core_interface_perimeter=float(upward_perimeter),
        falling_film_interface_perimeter=float(falling_perimeter),
        upward_core_reynolds=float(upward_re),
        falling_film_reynolds=float(falling_re),
        upward_core_shear_upward=float(upward_shear),
        falling_film_shear_upward=float(falling_shear),
        upward_core_force_conductance=float(max(upward_conductance, 0.0)),
        falling_film_force_conductance=float(max(falling_conductance, 0.0)),
        upward_core_pressure_upward=float(upward_pressure),
        falling_film_pressure_upward=float(falling_pressure),
        circulation_drive_pressure=float(upward_pressure - falling_pressure),
        upward_core_force_upward=float(upward_force),
        falling_film_force_upward=float(falling_force),
    )


def _implicit_gas_liquid_exchange(
    q_c: float,
    q_net: float,
    *,
    dt: float,
    trace: PersistentMouthTrace,
    film: FilmGeometry,
    material: VerticalMouthMaterialProperties,
    parameters: BidirectionalTNodeParameters,
) -> ImplicitGasLiquidExchange:
    """Apply one energy-stable semi-implicit gas/liquid drag exchange.

    Darcy conductances are frozen from the resolved old face state, but both
    liquid velocities and the finite first-cell gas velocity are evaluated at
    the new state.  With non-negative conductances this backward-Euler solve
    cannot overshoot the conductance-weighted velocity equilibrium and removes
    ``dt * k * slip_new**2`` from relative kinetic energy.

    The gas mass uses the complete first riser cell because the returned
    impulse is applied to that cell's momentum inventory ``Jgrs[0]``.  The
    interfacial force itself acts over the geometric half-cell length inside
    :func:`_gas_liquid_interface_action`.
    """

    # A newly opened Taylor-core trace can precede the conservative gas
    # transport into that control volume by one outer stage.  Geometry alone
    # is not a momentum owner: applying shear to a massless gas aperture would
    # accelerate the liquids while providing no equal-and-opposite gas impulse.
    # Leave interphase exchange inactive until actual gas mass reaches the
    # first cell; pressure, gravity, wall and turn terms still advance both
    # liquid directions during that stage.
    if trace.gas_cell_mass <= 0.0:
        return _inactive_gas_liquid_exchange(
            trace,
            film,
            material=material,
            parameters=parameters,
        )

    old_action = _gas_liquid_interface_action(
        trace,
        film,
        half_length=parameters.half_cell_length,
        material=material,
        parameters=parameters,
    )
    gas_velocity_old = old_action.gas_velocity
    gas_mass = trace.gas_cell_mass
    base_up = max(q_net, 0.0)
    base_down = max(-q_net, 0.0)
    upward_velocity_new = (
        (base_up + q_c) / trace.upward_area
        if trace.upward_area > 0.0
        else 0.0
    )
    falling_velocity_new = (
        -(base_down + q_c) / film.area
        if film.area > 0.0
        else 0.0
    )
    k_up = old_action.upward_core_force_conductance
    k_down = old_action.falling_film_force_conductance
    if gas_mass > 0.0:
        inertial_conductance = gas_mass / dt
        denominator = inertial_conductance + k_up + k_down
        gas_velocity_new = (
            inertial_conductance * gas_velocity_old
            + k_up * upward_velocity_new
            + k_down * falling_velocity_new
        ) / denominator
    else:
        gas_velocity_new = gas_velocity_old

    upward_slip_before = gas_velocity_old - trace.upward_velocity
    falling_slip_before = gas_velocity_old - trace.downward_velocity
    upward_slip_after = gas_velocity_new - upward_velocity_new
    falling_slip_after = gas_velocity_new - falling_velocity_new
    upward_force = k_up * upward_slip_after
    falling_force = k_down * falling_slip_after
    upward_pressure = (
        upward_force / trace.upward_area if trace.upward_area > 0.0 else 0.0
    )
    falling_pressure = (
        falling_force / film.area if film.area > 0.0 else 0.0
    )
    half_length = parameters.half_cell_length
    upward_shear = (
        upward_force
        / (old_action.upward_core_interface_perimeter * half_length)
        if old_action.upward_core_interface_perimeter > 0.0
        else 0.0
    )
    falling_shear = (
        falling_force
        / (old_action.falling_film_interface_perimeter * half_length)
        if old_action.falling_film_interface_perimeter > 0.0
        else 0.0
    )
    upward_re = (
        material.gas_density
        * abs(upward_slip_after)
        * old_action.gas_hydraulic_diameter
        / parameters.gas_dynamic_viscosity
    )
    falling_re = (
        material.gas_density
        * abs(falling_slip_after)
        * old_action.gas_hydraulic_diameter
        / parameters.gas_dynamic_viscosity
    )
    action = GasLiquidInterfaceAction(
        gas_velocity=float(gas_velocity_new),
        gas_hydraulic_diameter=float(old_action.gas_hydraulic_diameter),
        upward_core_interface_perimeter=float(
            old_action.upward_core_interface_perimeter
        ),
        falling_film_interface_perimeter=float(
            old_action.falling_film_interface_perimeter
        ),
        upward_core_reynolds=float(upward_re),
        falling_film_reynolds=float(falling_re),
        upward_core_shear_upward=float(upward_shear),
        falling_film_shear_upward=float(falling_shear),
        upward_core_force_conductance=float(k_up),
        falling_film_force_conductance=float(k_down),
        upward_core_pressure_upward=float(upward_pressure),
        falling_film_pressure_upward=float(falling_pressure),
        circulation_drive_pressure=float(upward_pressure - falling_pressure),
        upward_core_force_upward=float(upward_force),
        falling_film_force_upward=float(falling_force),
    )
    gas_momentum_change = gas_mass * (gas_velocity_new - gas_velocity_old)
    gas_momentum_residual = math.fsum(
        (gas_momentum_change, dt * upward_force, dt * falling_force)
    )
    dissipation = dt * math.fsum(
        (
            k_up * upward_slip_after**2,
            k_down * falling_slip_after**2,
        )
    )
    return ImplicitGasLiquidExchange(
        action=action,
        gas_velocity_before=float(gas_velocity_old),
        gas_velocity_after=float(gas_velocity_new),
        gas_cell_mass=float(gas_mass),
        gas_momentum_change=float(gas_momentum_change),
        gas_momentum_residual=float(gas_momentum_residual),
        upward_core_slip_before=float(upward_slip_before),
        upward_core_slip_after=float(upward_slip_after),
        falling_film_slip_before=float(falling_slip_before),
        falling_film_slip_after=float(falling_slip_after),
        dissipation_energy=float(dissipation),
    )


def _inactive_gas_liquid_exchange(
    trace: PersistentMouthTrace,
    film: FilmGeometry,
    *,
    material: VerticalMouthMaterialProperties,
    parameters: BidirectionalTNodeParameters,
) -> ImplicitGasLiquidExchange:
    """Return a zero exchange when two liquid directions do not both exist."""

    old = _gas_liquid_interface_action(
        trace,
        film,
        half_length=parameters.half_cell_length,
        material=material,
        parameters=parameters,
    )
    action = replace(
        old,
        upward_core_shear_upward=0.0,
        falling_film_shear_upward=0.0,
        upward_core_force_conductance=0.0,
        falling_film_force_conductance=0.0,
        upward_core_pressure_upward=0.0,
        falling_film_pressure_upward=0.0,
        circulation_drive_pressure=0.0,
        upward_core_force_upward=0.0,
        falling_film_force_upward=0.0,
    )
    return ImplicitGasLiquidExchange(
        action=action,
        gas_velocity_before=float(old.gas_velocity),
        gas_velocity_after=float(old.gas_velocity),
        gas_cell_mass=float(trace.gas_cell_mass),
        gas_momentum_change=0.0,
        gas_momentum_residual=0.0,
        upward_core_slip_before=float(old.gas_velocity - trace.upward_velocity),
        upward_core_slip_after=float(old.gas_velocity - trace.upward_velocity),
        falling_film_slip_before=float(old.gas_velocity - trace.downward_velocity),
        falling_film_slip_after=float(old.gas_velocity - trace.downward_velocity),
        dissipation_energy=0.0,
    )


def _circulation_losses_and_drive(
    q_c: float,
    q_net: float,
    *,
    trace: PersistentMouthTrace,
    film: FilmGeometry,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    parameters: BidirectionalTNodeParameters,
    losses: DirectionalMouthLosses,
    gas_circulation_drive_pressure: float,
    horizontal_phase_pressure_drive: float,
) -> tuple[float, float, float, float, float, float, float]:
    base_up = max(q_net, 0.0)
    base_down = max(-q_net, 0.0)
    q_up = base_up + q_c
    q_down = base_down + q_c
    total_wall, film_re, blend, laminar_wall = _film_wall_loss_pressure(
        q_down,
        film,
        half_length=parameters.half_cell_length,
        geometry=geometry,
        material=material,
        parameters=parameters,
    )
    base_wall, _, _, _ = _film_wall_loss_pressure(
        base_down,
        film,
        half_length=parameters.half_cell_length,
        geometry=geometry,
        material=material,
        parameters=parameters,
    )
    # Only the incremental wall work caused by q_c belongs to the closed-loop
    # coordinate.  Wall drag on a one-way negative q_net is owned by q_net.
    wall = max(total_wall - base_wall, 0.0)
    upward_speed_c = q_c / trace.upward_area if trace.upward_area > 0.0 else 0.0
    downward_speed_c = q_c / film.area if film.area > 0.0 else 0.0
    upward_turn = 0.5 * material.liquid_density * losses.upward_turn * upward_speed_c**2
    downward_turn = (
        0.5 * material.liquid_density * losses.downward_turn * downward_speed_c**2
    )
    if q_up > 0.0 and q_down > 0.0:
        predicted_activation = 4.0 * q_up * q_down / (q_up + q_down) ** 2
    else:
        predicted_activation = 0.0
    predicted_relative_speed = upward_speed_c + downward_speed_c

    # Retain the already-resolved first-cell two-stream momentum in the
    # source evaluation.  This is a trapezoidal (old/new) mixing work, not an
    # instantaneous capacity assignment.  In particular, both persistent
    # Qup and Qdown affect the next q_c even when q_net happens to be zero.
    trace_up = max(trace.upward_discharge, 0.0)
    trace_down = max(-trace.downward_discharge, 0.0)
    trace_gross = trace_up + trace_down
    trace_activation = (
        4.0 * trace_up * trace_down / trace_gross**2
        if trace_gross > 0.0
        else 0.0
    )
    trace_relative_speed = max(trace.upward_velocity, 0.0) + max(
        -trace.downward_velocity,
        0.0,
    )
    mixing = (
        0.25
        * material.liquid_density
        * losses.countercurrent_mixing
        * (
            trace_activation * trace_relative_speed**2
            + predicted_activation * predicted_relative_speed**2
        )
    )
    # Equal upward/downward volume rates traverse the same elevation.  The
    # downward-film gravity power +Delta(rho) g L q_c is exactly cancelled by
    # the upward-core gravity power -Delta(rho) g L q_c.  The remaining closed-
    # loop work has two resolved owners: differential gas shear, and the
    # pressure difference between the horizontal liquid donor under the
    # upward core and the horizontal gas-contact receiver of the falling film.
    # The common vertical mouth pressure cancels from those two equal and
    # opposite volume rates.
    total_loss = wall + upward_turn + downward_turn + mixing
    drive = math.fsum(
        (
            gas_circulation_drive_pressure,
            horizontal_phase_pressure_drive,
        )
    )
    return (
        float(total_loss),
        float(drive),
        float(wall),
        float(upward_turn),
        float(downward_turn),
        float(mixing),
        float(film_re),
    )


def _positive_monotone_root(
    function,
    *,
    parameters: BidirectionalTNodeParameters,
) -> float:
    lower = 0.0
    f_lower = float(function(lower))
    if f_lower >= 0.0:
        return 0.0
    upper = max(parameters.root_absolute_tolerance, 1.0e-12)
    f_upper = float(function(upper))
    for _ in range(parameters.max_root_iterations):
        if f_upper >= 0.0:
            break
        upper *= 2.0
        f_upper = float(function(upper))
    else:
        raise BidirectionalTNodeError("failed to bracket circulation momentum root")
    best_q = lower
    best_residual = abs(f_lower)
    for _ in range(parameters.max_root_iterations):
        old_lower = lower
        old_upper = upper
        midpoint = 0.5 * (lower + upper)
        f_mid = float(function(midpoint))
        if abs(f_mid) < best_residual:
            best_q = midpoint
            best_residual = abs(f_mid)
        if abs(f_mid) <= parameters.root_pressure_tolerance:
            return midpoint
        if f_mid >= 0.0:
            upper = midpoint
        else:
            lower = midpoint
        if midpoint == old_lower or midpoint == old_upper:
            break
    # Return the representable point with the smallest pressure residual.
    # A flux-width stopping criterion alone is unsafe because film inertance
    # can be very large when the dynamic annulus is thin.
    for candidate in (lower, upper, 0.5 * (lower + upper)):
        residual = abs(float(function(candidate)))
        if residual < best_residual:
            best_q = candidate
            best_residual = residual
    return float(best_q)


def _solve_circulation_given_net(
    q_net: float,
    state: BidirectionalTNodeState,
    *,
    dt: float,
    trace: PersistentMouthTrace,
    film: FilmGeometry,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    losses: DirectionalMouthLosses,
    parameters: BidirectionalTNodeParameters,
    node_rate: float,
    riser_rate: float,
    wallis_active: bool,
    wallis_downward_capacity: float,
    horizontal_phase_pressure_drive: float,
) -> _CirculationSolve:
    """Solve the constrained ``q_c``/gas exchange for one candidate q_net."""

    base_up = max(q_net, 0.0)
    base_down = max(-q_net, 0.0)
    node_capacity = max(node_rate - base_up, 0.0)
    riser_capacity = max(riser_rate - base_down, 0.0)
    wallis_capacity = (
        max(wallis_downward_capacity - base_down, 0.0)
        if wallis_active
        else math.inf
    )
    upper_bound = min(node_capacity, riser_capacity, wallis_capacity)

    dry = (
        film.area <= parameters.dry_area_tolerance
        or trace.upward_area <= parameters.dry_area_tolerance
    )
    if dry:
        if state.circulation_flow > parameters.root_absolute_tolerance:
            raise BidirectionalTNodeError(
                "persistent circulation cannot be carried by a dry two-stream mouth"
            )
        effective_area = 0.0
        inertance = 0.0
        q_free = 0.0
        q_new = 0.0
        inertive = 0.0
        total_loss = 0.0
        drive = 0.0
        wall_pressure = 0.0
        up_turn = 0.0
        down_turn = 0.0
        mixing = 0.0
        film_re = 0.0
        lower_reaction = 0.0
        upper_reaction = 0.0
        gas_exchange = _inactive_gas_liquid_exchange(
            trace,
            film,
            material=material,
            parameters=parameters,
        )
    else:
        effective_area = 1.0 / (1.0 / trace.upward_area + 1.0 / film.area)
        inertance = material.liquid_density * parameters.half_cell_length / effective_area

        def gas_at(q_c: float) -> ImplicitGasLiquidExchange:
            return _implicit_gas_liquid_exchange(
                q_c,
                q_net,
                dt=dt,
                trace=trace,
                film=film,
                material=material,
                parameters=parameters,
            )

        def residual_at(q_c: float) -> float:
            exchange = gas_at(q_c)
            loss_drive = _circulation_losses_and_drive(
                q_c,
                q_net,
                trace=trace,
                film=film,
                geometry=geometry,
                material=material,
                parameters=parameters,
                losses=losses,
                gas_circulation_drive_pressure=(
                    exchange.action.circulation_drive_pressure
                ),
                horizontal_phase_pressure_drive=(
                    horizontal_phase_pressure_drive
                ),
            )
            return (
                inertance * (q_c - state.circulation_flow) / dt
                + loss_drive[0]
                - loss_drive[1]
            )

        q_free = _positive_monotone_root(residual_at, parameters=parameters)
        q_new = min(q_free, upper_bound)
        gas_exchange = gas_at(q_new)
        (
            total_loss,
            drive,
            wall_pressure,
            up_turn,
            down_turn,
            mixing,
            film_re,
        ) = _circulation_losses_and_drive(
            q_new,
            q_net,
            trace=trace,
            film=film,
            geometry=geometry,
            material=material,
            parameters=parameters,
            losses=losses,
            gas_circulation_drive_pressure=(
                gas_exchange.action.circulation_drive_pressure
            ),
            horizontal_phase_pressure_drive=(
                horizontal_phase_pressure_drive
            ),
        )
        inertive = inertance * (q_new - state.circulation_flow) / dt
        raw = inertive + total_loss - drive
        lower_reaction = 0.0
        upper_reaction = 0.0
        root_tolerance = parameters.root_absolute_tolerance + (
            parameters.root_relative_tolerance
            * max(q_free, upper_bound, 1.0e-30)
        )
        if q_free <= root_tolerance and residual_at(0.0) > 0.0:
            lower_reaction = max(raw, 0.0)
        elif q_free > upper_bound + root_tolerance:
            upper_reaction = max(-raw, 0.0)

    owner_tolerance = max(
        1.0e-16,
        512.0
        * math.ulp(
            max(
                node_capacity,
                riser_capacity,
                wallis_capacity if math.isfinite(wallis_capacity) else 0.0,
                1.0e-12,
            )
        ),
    )
    upper_owner = "inactive"
    wallis_reaction = 0.0
    node_reaction = 0.0
    riser_reaction = 0.0
    if upper_reaction > 0.0:
        if (
            wallis_active
            and wallis_capacity <= min(node_capacity, riser_capacity) + owner_tolerance
        ):
            upper_owner = "wallis"
            wallis_reaction = upper_reaction
        elif node_capacity <= riser_capacity + owner_tolerance:
            upper_owner = "node_donor"
            node_reaction = upper_reaction
        else:
            upper_owner = "riser_donor"
            riser_reaction = upper_reaction
    residual = math.fsum(
        (inertive, total_loss, -drive, -lower_reaction, upper_reaction)
    )
    lower_gap = q_new
    upper_gap = upper_bound - q_new
    return _CirculationSolve(
        base_up=float(base_up),
        base_down=float(base_down),
        node_capacity=float(node_capacity),
        riser_capacity=float(riser_capacity),
        wallis_capacity=float(wallis_capacity),
        upper_bound=float(upper_bound),
        effective_area=float(effective_area),
        pressure_inertance=float(inertance),
        q_free=float(q_free),
        q_new=float(q_new),
        inertive_pressure=float(inertive),
        total_loss_pressure=float(total_loss),
        drive_pressure=float(drive),
        wall_pressure=float(wall_pressure),
        upward_turn_pressure=float(up_turn),
        downward_turn_pressure=float(down_turn),
        mixing_pressure=float(mixing),
        film_reynolds=float(film_re),
        lower_reaction=float(lower_reaction),
        upper_reaction=float(upper_reaction),
        upper_owner=upper_owner,
        wallis_reaction=float(wallis_reaction),
        node_reaction=float(node_reaction),
        riser_reaction=float(riser_reaction),
        pressure_residual=float(residual),
        lower_gap=float(lower_gap),
        upper_gap=float(upper_gap),
        lower_product=float(lower_reaction * lower_gap),
        upper_product=float(upper_reaction * upper_gap),
        gas_exchange=gas_exchange,
    )


def advance_bidirectional_tnode_inertance(
    state: BidirectionalTNodeState,
    *,
    dt: float,
    pressure: DistributedTNodePressureState,
    trace: PersistentMouthTrace,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    donors: LiquidDonorInventories,
    losses: DirectionalMouthLosses,
    parameters: BidirectionalTNodeParameters,
    riser_receiving_net_rate_capacity: float = math.inf,
) -> BidirectionalTNodeStepResult:
    """Advance persistent net and circulation fluxes by one conservative step."""

    step = float(dt)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("bidirectional T-node time step must be positive")
    if math.isnan(float(riser_receiving_net_rate_capacity)):
        raise ValueError("riser receiving capacity cannot be NaN")
    if not math.isclose(
        step,
        donors.time_step,
        rel_tol=1.0e-12,
        abs_tol=512.0 * math.ulp(max(step, donors.time_step, 1.0)),
    ):
        raise ValueError("node and donor time steps must match")
    pressure.validate(geometry)
    trace.validate(geometry)
    net_pressure = _directional_net_pressure_state(
        state,
        pressure,
        geometry,
    )
    horizontal_area_tolerance = max(
        128.0 * math.ulp(max(geometry.horizontal_full_area, 1.0)),
        parameters.dry_area_tolerance,
    )
    horizontal_two_phase_contact = bool(
        pressure.horizontal_liquid_area > horizontal_area_tolerance
        and pressure.horizontal_liquid_area
        < geometry.horizontal_full_area - horizontal_area_tolerance
    )
    horizontal_phase_pressure_drive = (
        pressure.horizontal_liquid_pressure_abs
        - pressure.horizontal_gas_pressure_abs
        if horizontal_two_phase_contact
        else 0.0
    )
    film = dynamic_annular_film_geometry(
        trace.downward_area,
        geometry=geometry,
    )
    wallis_active, gas_superficial, wallis_capacity = _wallis_capacity(
        trace,
        geometry=geometry,
        material=material,
        wallis=wallis,
    )
    node_rate = donors.finite_node_rate_capacity
    riser_rate = donors.riser_rate_capacity
    # Fixed-point coupling is only across the dissipative interphase operator;
    # each inner gas/qc solve is backward Euler.  At convergence the exact
    # same F_up + F_down drives q_net and reacts on the finite gas inventory.
    q_net, net_ledger = _advance_net_flux(
        state.q_net,
        dt=step,
        pressure=net_pressure,
        geometry=geometry,
        material=material,
        losses=losses,
        node_rate_capacity=node_rate,
        riser_rate_capacity=riser_rate,
        wallis_active=wallis_active,
        wallis_downward_capacity=wallis_capacity,
        riser_receiving_net_rate_capacity=(
            riser_receiving_net_rate_capacity
        ),
    )
    coupled_tolerance = max(
        parameters.root_absolute_tolerance,
        parameters.root_relative_tolerance * max(abs(q_net), 1.0e-12),
    )
    circulation_solve = _solve_circulation_given_net(
        q_net,
        state,
        dt=step,
        trace=trace,
        film=film,
        geometry=geometry,
        material=material,
        losses=losses,
        parameters=parameters,
        node_rate=node_rate,
        riser_rate=riser_rate,
        wallis_active=wallis_active,
        wallis_downward_capacity=wallis_capacity,
        horizontal_phase_pressure_drive=(
            horizontal_phase_pressure_drive
        ),
    )
    converged = False
    for _ in range(80):
        gas_force = math.fsum(
            (
                circulation_solve.gas_exchange.action.upward_core_force_upward,
                circulation_solve.gas_exchange.action.falling_film_force_upward,
            )
        )
        gas_pressure = gas_force / geometry.mouth_area
        q_target, target_ledger = _advance_net_flux(
            state.q_net,
            dt=step,
            pressure=net_pressure,
            geometry=geometry,
            material=material,
            losses=losses,
            node_rate_capacity=node_rate,
            riser_rate_capacity=riser_rate,
            wallis_active=wallis_active,
            wallis_downward_capacity=wallis_capacity,
            riser_receiving_net_rate_capacity=(
                riser_receiving_net_rate_capacity
            ),
            gas_interphase_drive_pressure=gas_pressure,
        )
        if abs(q_target - q_net) <= coupled_tolerance:
            q_net = q_target
            net_ledger = target_ledger
            circulation_solve = _solve_circulation_given_net(
                q_net,
                state,
                dt=step,
                trace=trace,
                film=film,
                geometry=geometry,
                material=material,
                losses=losses,
                parameters=parameters,
                node_rate=node_rate,
                riser_rate=riser_rate,
                wallis_active=wallis_active,
                wallis_downward_capacity=wallis_capacity,
                horizontal_phase_pressure_drive=(
                    horizontal_phase_pressure_drive
                ),
            )
            converged = True
            break
        # Under-relax only the nonlinear iterate, never the accepted ledger.
        q_net = 0.5 * (q_net + q_target)
        circulation_solve = _solve_circulation_given_net(
            q_net,
            state,
            dt=step,
            trace=trace,
            film=film,
            geometry=geometry,
            material=material,
            losses=losses,
            parameters=parameters,
            node_rate=node_rate,
            riser_rate=riser_rate,
            wallis_active=wallis_active,
            wallis_downward_capacity=wallis_capacity,
            horizontal_phase_pressure_drive=(
                horizontal_phase_pressure_drive
            ),
        )
    if not converged:
        raise BidirectionalTNodeError("gas/qnet/qc semi-implicit coupling did not converge")

    base_up = circulation_solve.base_up
    base_down = circulation_solve.base_down
    node_qc_capacity = circulation_solve.node_capacity
    riser_qc_capacity = circulation_solve.riser_capacity
    wallis_qc_capacity = circulation_solve.wallis_capacity
    qc_upper = circulation_solve.upper_bound
    circulation_effective_area = circulation_solve.effective_area
    circulation_inertance = circulation_solve.pressure_inertance
    qc_free = circulation_solve.q_free
    qc_new = circulation_solve.q_new
    inertive_pressure = circulation_solve.inertive_pressure
    total_loss = circulation_solve.total_loss_pressure
    drive = circulation_solve.drive_pressure
    wall_pressure = circulation_solve.wall_pressure
    up_turn = circulation_solve.upward_turn_pressure
    down_turn = circulation_solve.downward_turn_pressure
    mixing = circulation_solve.mixing_pressure
    film_re = circulation_solve.film_reynolds
    lower_reaction = circulation_solve.lower_reaction
    upper_reaction = circulation_solve.upper_reaction
    upper_owner = circulation_solve.upper_owner
    wallis_reaction = circulation_solve.wallis_reaction
    node_reaction = circulation_solve.node_reaction
    riser_reaction = circulation_solve.riser_reaction
    total_reaction_residual = circulation_solve.pressure_residual
    lower_gap = circulation_solve.lower_gap
    upper_gap = circulation_solve.upper_gap
    lower_product = circulation_solve.lower_product
    upper_product = circulation_solve.upper_product
    gas_exchange = circulation_solve.gas_exchange

    density_difference = material.liquid_density - material.gas_density
    nusselt_velocity = (
        density_difference
        * geometry.gravity
        * film.thickness**2
        / (3.0 * material.liquid_dynamic_viscosity)
        if film.thickness > 0.0
        else 0.0
    )
    nusselt_flow = film.area * nusselt_velocity
    nusselt_thin = (
        math.pi
        * geometry.riser_diameter
        * density_difference
        * geometry.gravity
        * film.thickness**3
        / (3.0 * material.liquid_dynamic_viscosity)
        if film.thickness > 0.0
        else 0.0
    )
    q_down = base_down + qc_new
    q_up = base_up + qc_new
    wall_blend = _smoothstep(
        (film_re - parameters.laminar_film_reynolds_limit)
        / (
            parameters.turbulent_film_reynolds_limit
            - parameters.laminar_film_reynolds_limit
        )
    )
    _, _, _, laminar_wall = _film_wall_loss_pressure(
        q_down,
        film,
        half_length=parameters.half_cell_length,
        geometry=geometry,
        material=material,
        parameters=parameters,
    )
    branch_gravity = (
        density_difference * geometry.gravity * parameters.half_cell_length
    )
    net_circulation_gravity = math.fsum((branch_gravity, -branch_gravity))
    nusselt_force_residual = branch_gravity - laminar_wall
    nusselt_applicable = bool(
        film.area > parameters.dry_area_tolerance
        and film_re <= parameters.laminar_film_reynolds_limit
    )

    old_circulation_momentum = (
        material.liquid_density
        * parameters.half_cell_length
        * state.circulation_flow
    )
    new_circulation_momentum = (
        material.liquid_density
        * parameters.half_cell_length
        * qc_new
    )
    actual_circulation_momentum = (
        new_circulation_momentum - old_circulation_momentum
    )
    expected_circulation_momentum = circulation_effective_area * step * math.fsum(
        (drive, -total_loss, lower_reaction, -upper_reaction)
    )
    circulation_momentum_residual = (
        actual_circulation_momentum - expected_circulation_momentum
    )
    gas_action = gas_exchange.action
    upward_core_gas_impulse = gas_action.upward_core_force_upward * step
    falling_film_gas_impulse = gas_action.falling_film_force_upward * step
    liquid_gas_impulse_upward = math.fsum(
        (upward_core_gas_impulse, falling_film_gas_impulse)
    )
    gas_reaction_impulse = gas_exchange.gas_momentum_change
    gas_liquid_residual = liquid_gas_impulse_upward + gas_reaction_impulse
    net_interphase_residual = math.fsum(
        (net_ledger.liquid_interphase_impulse_upward, gas_reaction_impulse)
    )
    net_ledger = replace(
        net_ledger,
        gas_interphase_impulse_upward=float(gas_reaction_impulse),
        interphase_momentum_residual=float(net_interphase_residual),
    )
    gas_impulse_scale = max(
        abs(liquid_gas_impulse_upward),
        abs(gas_reaction_impulse),
        1.0e-30,
    )
    gas_impulse_tolerance = max(
        1.0e-18,
        1.0e-10 * gas_impulse_scale,
        8192.0 * math.ulp(gas_impulse_scale),
    )
    if abs(gas_exchange.gas_momentum_residual) > gas_impulse_tolerance:
        raise BidirectionalTNodeError(
            "implicit gas momentum ledger did not close: "
            f"residual={gas_exchange.gas_momentum_residual:.12e}, "
            f"tolerance={gas_impulse_tolerance:.12e}, "
            f"gas_mass={gas_exchange.gas_cell_mass:.12e}, "
            f"u_old={gas_exchange.gas_velocity_before:.12e}, "
            f"u_new={gas_exchange.gas_velocity_after:.12e}, "
            f"dJg={gas_reaction_impulse:.12e}, "
            f"Il={liquid_gas_impulse_upward:.12e}, "
            f"Aup={trace.upward_area:.12e}, "
            f"Adown={trace.downward_area:.12e}, "
            f"Ag={trace.gas_area:.12e}, dt={step:.12e}"
        )
    if abs(gas_liquid_residual) > gas_impulse_tolerance:
        raise BidirectionalTNodeError("gas/liquid action-reaction ledger did not close")
    if abs(net_interphase_residual) > gas_impulse_tolerance:
        raise BidirectionalTNodeError(
            "q_net did not consume the realized gas interphase impulse"
        )
    if gas_exchange.dissipation_energy < -gas_impulse_tolerance:
        raise BidirectionalTNodeError("interphase exchange created relative energy")
    trace_net_flow = math.fsum(
        (trace.upward_discharge, trace.downward_discharge)
    )
    trace_circulation_flow = min(
        max(trace.upward_discharge, 0.0),
        max(-trace.downward_discharge, 0.0),
    )

    upward_volume = q_up * step
    downward_volume = q_down * step
    node_after = donors.finite_node_volume - upward_volume + downward_volume
    riser_after = donors.riser_volume + upward_volume - downward_volume
    volume_tolerance = max(
        1.0e-18,
        2048.0
        * math.ulp(
            max(
                donors.finite_node_volume,
                donors.riser_volume,
                upward_volume,
                downward_volume,
                1.0e-18,
            )
        ),
    )
    if node_after < -volume_tolerance or riser_after < -volume_tolerance:
        raise BidirectionalTNodeError("accepted gross flow exhausted a donor")
    node_after = max(node_after, 0.0)
    riser_after = max(riser_after, 0.0)
    node_change = node_after - donors.finite_node_volume
    riser_change = riser_after - donors.riser_volume
    combined_residual = math.fsum((node_change, riser_change))

    closure_residual = math.fsum((q_up, -q_down, -q_net))
    pressure_scale = max(
        abs(net_ledger.inertive_pressure),
        abs(net_ledger.turn_loss_pressure),
        abs(net_ledger.driving_pressure),
        abs(total_loss),
        abs(drive),
        abs(lower_reaction),
        abs(upper_reaction),
        1.0,
    )
    pressure_tolerance = max(
        32.0 * parameters.root_pressure_tolerance,
        1.0e-10 * pressure_scale,
        8192.0 * math.ulp(pressure_scale),
    )
    momentum_scale = max(
        abs(actual_circulation_momentum),
        abs(expected_circulation_momentum),
        1.0e-12,
    )
    momentum_tolerance = max(
        1.0e-14,
        1.0e-10 * momentum_scale,
        8192.0 * math.ulp(momentum_scale),
    )
    flux_scale = max(q_up, q_down, abs(q_net), 1.0e-12)
    flux_tolerance = max(
        parameters.root_absolute_tolerance,
        1.0e-10 * flux_scale,
        8192.0 * math.ulp(flux_scale),
    )
    complementarity_tolerance = pressure_tolerance * flux_scale
    net_momentum_scale = max(
        abs(net_ledger.actual_momentum_change),
        abs(net_ledger.expected_momentum_change),
        1.0e-12,
    )
    net_momentum_tolerance = max(
        1.0e-14,
        1.0e-10 * net_momentum_scale,
        8192.0 * math.ulp(net_momentum_scale),
    )
    if abs(net_ledger.pressure_residual) > pressure_tolerance:
        raise BidirectionalTNodeError("net-flux pressure ledger did not close")
    if abs(net_ledger.momentum_residual) > net_momentum_tolerance:
        raise BidirectionalTNodeError("net-flux momentum ledger did not close")
    if max(
        abs(net_ledger.lower_complementarity_product),
        abs(net_ledger.upper_complementarity_product),
    ) > complementarity_tolerance:
        raise BidirectionalTNodeError("net-flux complementarity product is nonzero")
    if abs(total_reaction_residual) > pressure_tolerance:
        raise BidirectionalTNodeError("circulation pressure ledger did not close")
    if abs(circulation_momentum_residual) > momentum_tolerance:
        raise BidirectionalTNodeError("circulation momentum ledger did not close")
    if max(abs(lower_product), abs(upper_product)) > complementarity_tolerance:
        raise BidirectionalTNodeError("circulation complementarity product is nonzero")
    if abs(combined_residual) > volume_tolerance or abs(closure_residual) > flux_tolerance:
        raise BidirectionalTNodeError("bidirectional mouth volume ledger did not close")

    if q_up > flux_tolerance and trace.upward_area <= parameters.dry_area_tolerance:
        raise BidirectionalTNodeError(
            "positive upward gross flow has no persistent upward mouth area "
            f"(q_old={state.q_net:.17g}, q_new={q_net:.17g}, "
            f"q_c_old={state.circulation_flow:.17g}, q_c_new={qc_new:.17g}, "
            f"q_up={q_up:.17g}, A_up={trace.upward_area:.17g}, "
            f"Q_up_trace={trace.upward_discharge:.17g}, "
            f"A_down={trace.downward_area:.17g}, "
            f"Q_down_trace={trace.downward_discharge:.17g}, "
            f"A_g={trace.gas_area:.17g})"
        )
    if q_down > flux_tolerance and trace.downward_area <= parameters.dry_area_tolerance:
        raise BidirectionalTNodeError(
            "positive downward gross flow has no persistent falling-film area"
        )

    upward_velocity = q_up / trace.upward_area if trace.upward_area > 0.0 else 0.0
    downward_velocity = q_down / trace.downward_area if trace.downward_area > 0.0 else 0.0
    next_state = BidirectionalTNodeState(
        q_net=float(q_net),
        circulation_flow=float(qc_new),
    )
    circulation_ledger = CirculationLedger(
        old_circulation_flow=float(state.circulation_flow),
        unconstrained_circulation_flow=float(qc_free),
        accepted_circulation_flow=float(qc_new),
        film=film,
        film_reynolds=float(film_re),
        wall_regime_blend=float(wall_blend),
        circulation_effective_area=float(circulation_effective_area),
        circulation_pressure_inertance=float(circulation_inertance),
        inertive_pressure=float(inertive_pressure),
        # Retained compatibility spelling: this is the *net* circulation
        # gravity pressure, which is identically zero for equal gross rates.
        buoyant_gravity_pressure=float(net_circulation_gravity),
        upward_core_gravity_pressure_downward=float(branch_gravity),
        falling_film_gravity_pressure_downward=float(branch_gravity),
        net_circulation_gravity_pressure=float(net_circulation_gravity),
        wall_loss_pressure=float(wall_pressure),
        upward_turn_loss_pressure=float(up_turn),
        downward_turn_loss_pressure=float(down_turn),
        countercurrent_mixing_loss_pressure=float(mixing),
        gas_upward_core_pressure_upward=float(
            gas_action.upward_core_pressure_upward
        ),
        gas_film_pressure_upward=float(
            gas_action.falling_film_pressure_upward
        ),
        gas_drive_pressure_downward=float(
            -gas_action.falling_film_pressure_upward
        ),
        gas_circulation_drive_pressure=float(
            gas_action.circulation_drive_pressure
        ),
        horizontal_phase_pressure_drive=float(
            horizontal_phase_pressure_drive
        ),
        gas_superficial_velocity=float(gas_superficial),
        gas_velocity_before_exchange=float(gas_exchange.gas_velocity_before),
        gas_velocity_after_exchange=float(gas_exchange.gas_velocity_after),
        gas_cell_mass=float(gas_exchange.gas_cell_mass),
        upward_core_gas_conductance=float(
            gas_action.upward_core_force_conductance
        ),
        falling_film_gas_conductance=float(
            gas_action.falling_film_force_conductance
        ),
        upward_core_gas_slip_before=float(
            gas_exchange.upward_core_slip_before
        ),
        upward_core_gas_slip_after=float(
            gas_exchange.upward_core_slip_after
        ),
        falling_film_gas_slip_before=float(
            gas_exchange.falling_film_slip_before
        ),
        falling_film_gas_slip_after=float(
            gas_exchange.falling_film_slip_after
        ),
        interphase_dissipation_energy=float(
            gas_exchange.dissipation_energy
        ),
        gas_momentum_change=float(gas_exchange.gas_momentum_change),
        gas_momentum_residual=float(gas_exchange.gas_momentum_residual),
        trace_net_flow=float(trace_net_flow),
        trace_circulation_flow=float(trace_circulation_flow),
        state_trace_net_mismatch=float(state.q_net - trace_net_flow),
        state_trace_circulation_mismatch=float(
            state.circulation_flow - trace_circulation_flow
        ),
        wallis_downward_capacity=float(wallis_capacity),
        wallis_active=bool(wallis_active),
        node_circulation_capacity=float(node_qc_capacity),
        riser_circulation_capacity=float(riser_qc_capacity),
        wallis_circulation_capacity=float(wallis_qc_capacity),
        upper_circulation_bound=float(qc_upper),
        upper_bound_owner=upper_owner,
        lower_reaction_pressure=float(lower_reaction),
        wallis_reaction_pressure=float(wallis_reaction),
        node_donor_reaction_pressure=float(node_reaction),
        riser_donor_reaction_pressure=float(riser_reaction),
        pressure_residual=float(total_reaction_residual),
        lower_gap=float(lower_gap),
        upper_gap=float(upper_gap),
        lower_complementarity_product=float(lower_product),
        upper_complementarity_product=float(upper_product),
        nusselt_applicable=nusselt_applicable,
        nusselt_equilibrium_flow=float(nusselt_flow),
        nusselt_thin_film_flow=float(nusselt_thin),
        nusselt_flow_residual=float(q_down - nusselt_flow),
        nusselt_force_residual=float(nusselt_force_residual),
        actual_circulation_momentum_change=float(
            actual_circulation_momentum
        ),
        expected_circulation_momentum_change=float(
            expected_circulation_momentum
        ),
        circulation_momentum_residual=float(circulation_momentum_residual),
        upward_core_gas_impulse_upward=float(upward_core_gas_impulse),
        falling_film_gas_impulse_upward=float(falling_film_gas_impulse),
        liquid_gas_impulse_upward=float(liquid_gas_impulse_upward),
        gas_reaction_impulse_upward=float(gas_reaction_impulse),
        gas_liquid_momentum_residual=float(gas_liquid_residual),
    )
    volume_ledger = BidirectionalVolumeLedger(
        node_volume_before=float(donors.finite_node_volume),
        node_volume_after=float(node_after),
        riser_donor_volume_before=float(donors.riser_volume),
        riser_donor_volume_after=float(riser_after),
        upward_gross_volume=float(upward_volume),
        downward_gross_volume=float(downward_volume),
        node_volume_change=float(node_change),
        riser_volume_change=float(riser_change),
        combined_volume_residual=float(combined_residual),
    )
    return BidirectionalTNodeStepResult(
        state=next_state,
        upward_flow=float(q_up),
        downward_flow=float(q_down),
        upward_velocity=float(upward_velocity),
        downward_velocity=float(downward_velocity),
        net=net_ledger,
        circulation=circulation_ledger,
        volume=volume_ledger,
    )


def twochannel_result_from_bidirectional_step(
    step_result: BidirectionalTNodeStepResult,
    *,
    trace: PersistentMouthTrace,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    losses: DirectionalMouthLosses,
) -> TwoChannelMouthResult:
    """Expose the dynamic solution through the existing gross-flow contract.

    The adapter does not call the old algebraic closure and therefore cannot
    overwrite the integrated ``q_c`` with a capacity.  The compatibility
    fields named ``gravity_film_capacity`` retain the historical dataclass
    spelling, but contain the Nusselt *equilibrium diagnostic* only.  They
    are never consumed by the dynamic step.
    """

    trace.validate(geometry)
    q_net = step_result.q_net
    q_up = step_result.upward_flow
    q_down = step_result.downward_flow
    q_c = step_result.circulation_flow
    tolerance = max(
        1.0e-16,
        1.0e-10 * max(q_up, q_down, abs(q_net), 1.0e-12),
    )
    if q_up > tolerance and trace.upward_area <= 0.0:
        raise BidirectionalTNodeError(
            "two-stream plan requires a resolved upward liquid mouth area"
        )
    if q_down > tolerance and trace.downward_area <= 0.0:
        raise BidirectionalTNodeError(
            "two-stream plan requires a resolved downward liquid mouth area"
        )

    upward_velocity = q_up / trace.upward_area if trace.upward_area > 0.0 else 0.0
    downward_velocity = (
        -q_down / trace.downward_area if trace.downward_area > 0.0 else 0.0
    )
    resolved_liquid_area = trace.upward_area + trace.downward_area
    resolved_liquid_flux = math.fsum(
        (trace.upward_discharge, trace.downward_discharge)
    )
    resolved_liquid_velocity = (
        resolved_liquid_flux / resolved_liquid_area
        if resolved_liquid_area > 0.0
        else 0.0
    )
    rho = material.liquid_density
    gross_momentum = rho * (
        q_up * upward_velocity + q_down * abs(downward_velocity)
    )
    bulk_momentum = (
        rho * q_net**2 / resolved_liquid_area
        if resolved_liquid_area > 0.0
        else 0.0
    )
    momentum_excess = max(gross_momentum - bulk_momentum, 0.0)
    gross_kinetic_power = 0.5 * rho * (
        q_up * upward_velocity**2 + q_down * downward_velocity**2
    )
    signed_kinetic_flux = 0.5 * rho * (
        q_up * upward_velocity**2 - q_down * downward_velocity**2
    )
    upward_loss_power = (
        0.5 * rho * losses.upward_turn * q_up * upward_velocity**2
    )
    downward_loss_power = (
        0.5 * rho * losses.downward_turn * q_down * downward_velocity**2
    )
    relative_velocity = upward_velocity - downward_velocity
    mixing_loss_power = (
        0.5
        * rho
        * losses.countercurrent_mixing
        * q_c
        * relative_velocity**2
    )
    total_dissipation = math.fsum(
        (upward_loss_power, downward_loss_power, mixing_loss_power)
    )
    density_difference = material.liquid_density - material.gas_density
    wallis_jg = (
        step_result.circulation.gas_superficial_velocity
        * math.sqrt(
            material.gas_density
            / (
                geometry.gravity
                * geometry.riser_diameter
                * density_difference
            )
        )
    )
    wallis_capacity = step_result.circulation.wallis_downward_capacity
    wallis_circulation_capacity = (
        step_result.circulation.wallis_circulation_capacity
    )
    downward_physical_capacity = (
        wallis_capacity
        if step_result.circulation.wallis_active
        else math.inf
    )
    downward_physical_circulation_capacity = (
        wallis_circulation_capacity
        if step_result.circulation.wallis_active
        else math.inf
    )
    return TwoChannelMouthResult(
        q_net=float(q_net),
        upward_flow=float(q_up),
        downward_flow=float(q_down),
        circulation_flow=float(q_c),
        closure_residual=float(step_result.closure_residual),
        film_thickness=float(step_result.circulation.film.thickness),
        gravity_film_capacity=float(
            step_result.circulation.nusselt_equilibrium_flow
        ),
        wallis_downward_capacity=float(wallis_capacity),
        downward_physical_capacity=float(downward_physical_capacity),
        downward_physical_circulation_capacity=float(
            downward_physical_circulation_capacity
        ),
        finite_node_circulation_capacity=float(
            step_result.circulation.node_circulation_capacity
        ),
        riser_circulation_capacity=float(
            step_result.circulation.riser_circulation_capacity
        ),
        gas_superficial_velocity=float(
            step_result.circulation.gas_superficial_velocity
        ),
        wallis_gas_parameter=float(wallis_jg),
        upward_channel_area=float(trace.upward_area),
        downward_channel_area=float(trace.downward_area),
        upward_channel_velocity=float(upward_velocity),
        downward_channel_velocity=float(downward_velocity),
        resolved_liquid_velocity=float(resolved_liquid_velocity),
        resolved_net_flux_mismatch=float(q_net - resolved_liquid_flux),
        gross_convective_momentum_flux=float(gross_momentum),
        bulk_convective_momentum_flux=float(bulk_momentum),
        countercurrent_momentum_excess=float(momentum_excess),
        gross_kinetic_power=float(gross_kinetic_power),
        signed_kinetic_energy_flux=float(signed_kinetic_flux),
        upward_turn_loss_power=float(upward_loss_power),
        downward_turn_loss_power=float(downward_loss_power),
        countercurrent_mixing_loss_power=float(mixing_loss_power),
        total_dissipation_power=float(total_dissipation),
    )


def stage_bidirectional_tnode_coupling(
    step_result: BidirectionalTNodeStepResult,
    *,
    trace: PersistentMouthTrace,
    geometry: DistributedTNodeGeometry,
    material: VerticalMouthMaterialProperties,
    losses: DirectionalMouthLosses,
    horizontal_axial_velocity: float,
    horizontal_node_topology: HorizontalNodeTopology,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> TwoChannelMouthCouplingPlan:
    """Create the shared horizontal/vertical plan from one dynamic ledger.

    This is the compatibility seam for ``apply_twochannel_horizontal_footprint``
    and the existing two-liquid vertical boundary.  Both consumers receive
    exactly the same accepted gross rates; no second closure is evaluated.
    """

    require_exclusive_twochannel_ownership(legacy_activity)
    horizontal_velocity = float(horizontal_axial_velocity)
    if not math.isfinite(horizontal_velocity):
        raise ValueError("horizontal mouth velocity must be finite")
    exchange = twochannel_result_from_bidirectional_step(
        step_result,
        trace=trace,
        geometry=geometry,
        material=material,
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
    plan = TwoChannelMouthCouplingPlan(
        exchange=exchange,
        vertical_boundary=vertical,
        horizontal_liquid_volume_rate=-exchange.q_net,
        vertical_liquid_volume_rate=exchange.q_net,
        horizontal_axial_kinematic_momentum_rate=(
            -exchange.upward_flow * horizontal_velocity
        ),
        horizontal_node_topology=HorizontalNodeTopology(horizontal_node_topology),
        legacy_paths_to_disable=(
            "characteristic_bottom_flux_as_update",
            "taylor_return_as_mass_flux",
            "post_breakthrough_ccfl_on_q_net",
            "net_only_horizontal_side_source",
        ),
    )
    scale = max(exchange.upward_flow, exchange.downward_flow, abs(exchange.q_net), 1.0e-12)
    tolerance = max(1.0e-16, 1.0e-10 * scale)
    if abs(plan.combined_liquid_volume_rate) > tolerance:
        raise BidirectionalTNodeError("dynamic two-channel plan lost liquid volume")
    if abs(vertical.total_volume_rate - exchange.q_net) > tolerance:
        raise BidirectionalTNodeError(
            "dynamic two-liquid boundary does not recover the accepted q_net"
        )
    return plan


__all__ = [
    "BidirectionalTNodeError",
    "BidirectionalTNodeParameters",
    "BidirectionalTNodeState",
    "BidirectionalTNodeStepResult",
    "BidirectionalVolumeLedger",
    "CirculationLedger",
    "FilmGeometry",
    "GasLiquidInterfaceAction",
    "ImplicitGasLiquidExchange",
    "NetFluxLedger",
    "PersistentMouthTrace",
    "advance_bidirectional_tnode_inertance",
    "dynamic_annular_film_geometry",
    "stage_bidirectional_tnode_coupling",
    "twochannel_result_from_bidirectional_step",
]
