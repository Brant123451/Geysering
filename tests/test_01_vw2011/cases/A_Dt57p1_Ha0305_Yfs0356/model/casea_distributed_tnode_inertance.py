"""Target-free inertive liquid-flux owner for the Case-A distributed T node.

The existing horizontal finite-volume cells under the measured tower opening
form a finite-width T-junction control volume.  This module gives that control
volume one persistent *net-mouth momentum* instead of recomputing a signed
vertical flux algebraically at every step.  For a mouth flow ``Q`` (positive
from the horizontal pipe into the riser), the stored momentum is

``P = rho_l * L_eff * Q`` with ``L_eff = V_footprint / A_mouth``.

Equivalently, the pressure inertance is

``I_p = rho_l * L_eff / A_mouth``

and the implicit one-step momentum equation is

``I_p (Q[n+1]-Q[n])/dt + dp_loss(Q[n+1])``
``    = p_horizontal - p_vertical + lambda_down - lambda_up``.

``p_horizontal`` is the instantaneous phase-contact-area average of the
resolved horizontal gas and liquid pressures over the node cross-section.
No result, elapsed time, target hold-up, or prescribed entry pulse enters the
equation.

The Nusselt/Wallis closure constrains *total downward gross flow*.  Therefore
its capacity also supplies the lower admissible bound for negative ``Q``.
When the unconstrained momentum solution crosses that bound, this module
solves the pressure/flux complementarity condition and records the constraint
reaction pressure and impulse explicitly.  It never silently clips a flux.
Donor-positivity bounds are handled by the same box-complementarity solve.

After the unique net flux is accepted, the existing Case-A two-channel mouth
closure decomposes it into simultaneous gross upward/downward streams.  The
same signed net flux is then applied once, with opposite signs, to the finite
horizontal node and riser inventories.  The returned ledger verifies exact
combined liquid-volume conservation.

This is an isolated integration component.  It deliberately does not import
or modify :mod:`vw2011_network_twofluid`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    LiquidDonorInventories,
    VerticalMouthGeometry,
    VerticalMouthMaterialProperties,
    VerticalMouthPhaseState,
    WallisCounterCurrentParameters,
    close_vertical_mouth_twochannel_exchange,
)
from casea_vertical_mouth_twochannel_integration import (
    HorizontalNodeTopology,
    LegacyMouthPathActivity,
    LumpedInventoryUpdate,
    TwoChannelMouthCouplingPlan,
    advance_lumped_liquid_inventories,
    stage_twochannel_mouth_coupling,
)


class DistributedTNodeError(RuntimeError):
    """A distributed T-node state or complementarity solve was rejected."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class DistributedTNodeGeometry:
    """Measured pipe/mouth geometry used by the finite-width node.

    ``opening_footprint_volume`` is the fixed geometric volume represented by
    the horizontal cells beneath the tower opening.  On the native Case-A
    geometry it is the horizontal full-pipe area times the measured opening
    length ``D_r``.  A grid adapter may instead provide the exactly integrated
    physical-overlap volume; neither choice introduces a numerical spreading
    length.
    """

    horizontal_diameter: float
    riser_diameter: float
    opening_footprint_length: float
    opening_footprint_volume: float
    gravity: float = 9.81

    def __post_init__(self) -> None:
        values = (
            self.horizontal_diameter,
            self.riser_diameter,
            self.opening_footprint_length,
            self.opening_footprint_volume,
            self.gravity,
        )
        if not _finite(*values):
            raise ValueError("distributed T-node geometry must be finite")
        if min(values) <= 0.0:
            raise ValueError("distributed T-node geometry must be positive")

    @classmethod
    def case_a(cls, *, gravity: float = 9.81) -> "DistributedTNodeGeometry":
        """Return the measured Case-A geometry, without calibration factors."""

        horizontal_diameter = 0.094
        riser_diameter = 0.0571
        footprint_length = riser_diameter
        horizontal_area = 0.25 * math.pi * horizontal_diameter**2
        return cls(
            horizontal_diameter=horizontal_diameter,
            riser_diameter=riser_diameter,
            opening_footprint_length=footprint_length,
            opening_footprint_volume=horizontal_area * footprint_length,
            gravity=float(gravity),
        )

    @property
    def horizontal_full_area(self) -> float:
        return 0.25 * math.pi * self.horizontal_diameter**2

    @property
    def mouth_area(self) -> float:
        return 0.25 * math.pi * self.riser_diameter**2

    @property
    def effective_inertance_length(self) -> float:
        """Liquid slug length having the footprint volume and mouth area."""

        return self.opening_footprint_volume / self.mouth_area


@dataclass(frozen=True)
class DistributedTNodePressureState:
    """Current resolved pressures on the two sides of the liquid mouth.

    The horizontal liquid-area fraction is used only as a geometric
    phase-contact fraction.  It is not a fitted activation function.
    """

    horizontal_gas_pressure_abs: float
    horizontal_liquid_pressure_abs: float
    horizontal_liquid_area: float
    vertical_mouth_pressure_abs: float

    def validate(self, geometry: DistributedTNodeGeometry) -> None:
        if not _finite(
            self.horizontal_gas_pressure_abs,
            self.horizontal_liquid_pressure_abs,
            self.horizontal_liquid_area,
            self.vertical_mouth_pressure_abs,
        ):
            raise ValueError("distributed T-node pressure state must be finite")
        if min(
            self.horizontal_gas_pressure_abs,
            self.horizontal_liquid_pressure_abs,
            self.vertical_mouth_pressure_abs,
        ) <= 0.0:
            raise ValueError("distributed T-node absolute pressures must be positive")
        tolerance = 128.0 * math.ulp(max(geometry.horizontal_full_area, 1.0))
        if self.horizontal_liquid_area < -tolerance:
            raise ValueError("horizontal node liquid area cannot be negative")
        if self.horizontal_liquid_area > geometry.horizontal_full_area + tolerance:
            raise ValueError("horizontal node liquid area exceeds the pipe area")

    def horizontal_contact_pressure(
        self,
        geometry: DistributedTNodeGeometry,
    ) -> float:
        """Area-weight the resolved gas/liquid pressures at the T footprint."""

        self.validate(geometry)
        liquid_fraction = min(
            max(self.horizontal_liquid_area / geometry.horizontal_full_area, 0.0),
            1.0,
        )
        return math.fsum(
            (
                liquid_fraction * self.horizontal_liquid_pressure_abs,
                (1.0 - liquid_fraction) * self.horizontal_gas_pressure_abs,
            )
        )


@dataclass(frozen=True)
class DistributedTNodeMomentumState:
    """Persistent liquid momentum ``rho_l L_eff Q`` at the T mouth."""

    liquid_momentum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.liquid_momentum):
            raise ValueError("distributed T-node momentum must be finite")

    @classmethod
    def from_net_flux(
        cls,
        q_net: float,
        *,
        geometry: DistributedTNodeGeometry,
        liquid_density: float,
    ) -> "DistributedTNodeMomentumState":
        q = float(q_net)
        rho = float(liquid_density)
        if not _finite(q, rho) or rho <= 0.0:
            raise ValueError("finite net flux and positive liquid density required")
        return cls(
            liquid_momentum=(
                rho * geometry.effective_inertance_length * q
            )
        )

    def net_flux(
        self,
        *,
        geometry: DistributedTNodeGeometry,
        liquid_density: float,
    ) -> float:
        rho = float(liquid_density)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("positive finite liquid density required")
        return self.liquid_momentum / (
            rho * geometry.effective_inertance_length
        )


@dataclass(frozen=True)
class PressureFluxComplementarityLedger:
    """Pressure, reaction, momentum, and inequality audit for one step."""

    old_q_net: float
    unconstrained_q_net: float
    accepted_q_net: float
    horizontal_contact_pressure_abs: float
    vertical_mouth_pressure_abs: float
    driving_pressure_difference: float
    pressure_inertance: float
    inertive_pressure_change: float
    signed_local_loss_pressure: float
    nusselt_downward_capacity: float
    wallis_downward_capacity: float
    physical_downward_capacity: float
    riser_donor_downward_capacity: float
    horizontal_donor_upward_capacity: float
    lower_flux_bound: float
    upper_flux_bound: float
    lower_bound_owner: str
    physical_downward_reaction_pressure: float
    donor_downward_reaction_pressure: float
    donor_upward_reaction_pressure: float
    closed_mouth_lower_reaction_pressure: float
    closed_mouth_upper_reaction_pressure: float
    total_lower_reaction_pressure: float
    total_upper_reaction_pressure: float
    physical_downward_reaction_force: float
    physical_downward_reaction_impulse: float
    signed_constraint_reaction_force: float
    signed_constraint_reaction_impulse: float
    pressure_balance_residual: float
    physical_downward_gap: float
    lower_bound_gap: float
    upper_bound_gap: float
    physical_complementarity_product: float
    lower_complementarity_product: float
    upper_complementarity_product: float
    old_liquid_momentum: float
    new_liquid_momentum: float
    expected_liquid_momentum_change: float
    actual_liquid_momentum_change: float
    liquid_momentum_balance_residual: float

    @property
    def downward_capacity_active(self) -> bool:
        return self.physical_downward_reaction_pressure > 0.0

    @property
    def donor_bound_active(self) -> bool:
        return (
            self.donor_downward_reaction_pressure > 0.0
            or self.donor_upward_reaction_pressure > 0.0
        )

    @property
    def closed_mouth_reaction_pressure(self) -> float:
        """Magnitude of the reaction closing either flux direction."""

        return (
            self.closed_mouth_lower_reaction_pressure
            + self.closed_mouth_upper_reaction_pressure
        )


@dataclass(frozen=True)
class DistributedTNodeStepResult:
    """Accepted persistent state, gross exchange, and exact ledgers."""

    state: DistributedTNodeMomentumState
    q_net: float
    mouth_plan: TwoChannelMouthCouplingPlan
    inventory_update: LumpedInventoryUpdate
    complementarity: PressureFluxComplementarityLedger

    @property
    def combined_liquid_volume_residual(self) -> float:
        return self.inventory_update.conservation_residual


def measured_footprint_liquid_inventory(
    horizontal_liquid_area: Iterable[float],
    normalized_opening_weights: Iterable[float],
    *,
    geometry: DistributedTNodeGeometry,
) -> float:
    """Integrate liquid volume over the measured opening footprint.

    The main solver's opening weights are ``overlap_i / L_open`` and sum to
    one.  Consequently the physical overlap integral is

    ``sum(A_i * weight_i) * L_open``.

    Multiplying the weighted mean by the horizontal grid spacing would be
    grid-dependent and is correct only in the accidental case
    ``dx == L_open``.
    """

    area = tuple(float(value) for value in horizontal_liquid_area)
    weights = tuple(float(value) for value in normalized_opening_weights)
    if not area or len(area) != len(weights):
        raise ValueError("footprint areas and weights must be equal nonempty vectors")
    if not _finite(*area, *weights):
        raise ValueError("footprint areas and weights must be finite")
    if min(area) < 0.0 or min(weights) < 0.0:
        raise ValueError("footprint areas and weights cannot be negative")
    total_weight = math.fsum(weights)
    tolerance = 512.0 * math.ulp(max(abs(total_weight), 1.0))
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("opening weights must be normalized physical overlaps")
    weighted_area = math.fsum(
        liquid_area * weight for liquid_area, weight in zip(area, weights)
    )
    return weighted_area * geometry.opening_footprint_length


def _signed_local_loss_pressure(
    q_net: float,
    *,
    mouth_area: float,
    liquid_density: float,
    losses: DirectionalMouthLosses,
) -> float:
    q = float(q_net)
    if q == 0.0:
        return 0.0
    coefficient = losses.upward_turn if q > 0.0 else losses.downward_turn
    velocity = q / mouth_area
    return 0.5 * liquid_density * coefficient * velocity * abs(velocity)


def _unconstrained_implicit_flux(
    q_old: float,
    *,
    dt: float,
    pressure_inertance: float,
    driving_pressure: float,
    mouth_area: float,
    liquid_density: float,
    losses: DirectionalMouthLosses,
) -> float:
    """Solve the monotone implicit inertia-plus-directional-loss equation."""

    linear = pressure_inertance / dt
    right_hand_side = driving_pressure + linear * q_old
    if right_hand_side == 0.0:
        return 0.0
    coefficient = (
        losses.upward_turn if right_hand_side > 0.0 else losses.downward_turn
    )
    quadratic = 0.5 * liquid_density * coefficient / mouth_area**2
    magnitude_rhs = abs(right_hand_side)
    if quadratic == 0.0:
        magnitude = magnitude_rhs / linear
    else:
        # The rationalized positive quadratic root avoids cancellation when
        # the loss is weak relative to the inertive term.
        magnitude = (
            2.0 * magnitude_rhs
            / (
                linear
                + math.sqrt(linear**2 + 4.0 * quadratic * magnitude_rhs)
            )
        )
    return math.copysign(magnitude, right_hand_side)


def _capacity_owner(reference) -> str:
    wallis_active = (
        reference.gas_superficial_velocity > 0.0
        and reference.wallis_downward_capacity >= 0.0
    )
    if not wallis_active:
        return "nusselt"
    tolerance = 128.0 * math.ulp(
        max(
            reference.gravity_film_capacity,
            reference.wallis_downward_capacity,
            1.0,
        )
    )
    difference = (
        reference.gravity_film_capacity - reference.wallis_downward_capacity
    )
    if abs(difference) <= tolerance:
        return "nusselt_wallis_tie"
    if difference > 0.0:
        return "wallis"
    return "nusselt"


def advance_distributed_tnode_inertance(
    state: DistributedTNodeMomentumState,
    *,
    dt: float,
    pressure: DistributedTNodePressureState,
    geometry: DistributedTNodeGeometry,
    phase: VerticalMouthPhaseState,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    donors: LiquidDonorInventories,
    losses: DirectionalMouthLosses,
    horizontal_axial_velocity: float,
    legacy_activity: LegacyMouthPathActivity = LegacyMouthPathActivity(),
) -> DistributedTNodeStepResult:
    """Advance the sole distributed-node net flux and close gross exchange.

    The pressure/flux solve is implicit in the directional local loss and
    explicit only in the supplied current pressures.  Nusselt/Wallis and donor
    capacities enter as complementarity constraints in that solve.  The
    two-channel closure receives the already admissible net flux and therefore
    never has to repair or clip it.
    """

    step = float(dt)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("distributed T-node time step must be positive")
    if not math.isclose(
        step,
        donors.time_step,
        rel_tol=1.0e-12,
        abs_tol=128.0 * math.ulp(max(step, donors.time_step, 1.0)),
    ):
        raise ValueError("node and donor time steps must be identical")
    if not math.isfinite(horizontal_axial_velocity):
        raise ValueError("horizontal axial velocity must be finite")
    pressure.validate(geometry)
    mouth_geometry = VerticalMouthGeometry(
        diameter=geometry.riser_diameter,
        gravity=geometry.gravity,
    )
    phase.validate(mouth_geometry)

    rho = material.liquid_density
    area = geometry.mouth_area
    length = geometry.effective_inertance_length
    pressure_inertance = rho * length / area
    q_old = state.net_flux(geometry=geometry, liquid_density=rho)
    horizontal_pressure = pressure.horizontal_contact_pressure(geometry)
    driving_pressure = (
        horizontal_pressure - pressure.vertical_mouth_pressure_abs
    )
    q_unconstrained = _unconstrained_implicit_flux(
        q_old,
        dt=step,
        pressure_inertance=pressure_inertance,
        driving_pressure=driving_pressure,
        mouth_area=area,
        liquid_density=rho,
        losses=losses,
    )

    # Query the same constitutive law that will close the accepted gross
    # streams.  q=0 exposes its total downward physical capacity without
    # consuming either donor or changing an inventory.
    capacity_reference = close_vertical_mouth_twochannel_exchange(
        0.0,
        phase=phase,
        geometry=mouth_geometry,
        material=material,
        wallis=wallis,
        donors=donors,
        losses=losses,
    )
    physical_downward_capacity = (
        capacity_reference.downward_physical_capacity
    )
    riser_donor_capacity = donors.riser_rate_capacity
    horizontal_donor_capacity = donors.finite_node_rate_capacity

    liquid_open = phase.liquid_area > 0.0
    if liquid_open:
        physical_lower = -physical_downward_capacity
        donor_lower = -riser_donor_capacity
        lower_bound = max(physical_lower, donor_lower)
        upper_bound = horizontal_donor_capacity
    else:
        physical_lower = 0.0
        donor_lower = 0.0
        lower_bound = 0.0
        upper_bound = 0.0

    if lower_bound > upper_bound:
        raise DistributedTNodeError("inconsistent distributed-node flux bounds")
    q_accepted = min(max(q_unconstrained, lower_bound), upper_bound)
    tolerance = 512.0 * math.ulp(
        max(
            abs(q_unconstrained),
            abs(q_accepted),
            abs(lower_bound),
            abs(upper_bound),
            1.0,
        )
    )
    inertive_pressure = pressure_inertance * (q_accepted - q_old) / step
    loss_pressure = _signed_local_loss_pressure(
        q_accepted,
        mouth_area=area,
        liquid_density=rho,
        losses=losses,
    )
    equation_left_minus_drive = (
        inertive_pressure + loss_pressure - driving_pressure
    )

    physical_reaction = 0.0
    donor_down_reaction = 0.0
    donor_up_reaction = 0.0
    closed_lower_reaction = 0.0
    closed_upper_reaction = 0.0
    lower_owner = "inactive"
    if q_unconstrained < lower_bound - tolerance:
        lower_reaction = max(equation_left_minus_drive, 0.0)
        if not liquid_open:
            closed_lower_reaction = lower_reaction
            lower_owner = "closed_mouth"
        elif physical_lower >= donor_lower - tolerance:
            physical_reaction = lower_reaction
            lower_owner = _capacity_owner(capacity_reference)
        else:
            donor_down_reaction = lower_reaction
            lower_owner = "riser_donor"
    elif q_unconstrained > upper_bound + tolerance:
        upper_reaction = max(-equation_left_minus_drive, 0.0)
        if not liquid_open:
            closed_upper_reaction = upper_reaction
            lower_owner = "closed_mouth"
        else:
            donor_up_reaction = upper_reaction
            lower_owner = "horizontal_donor"

    total_lower_reaction = (
        physical_reaction + donor_down_reaction + closed_lower_reaction
    )
    total_upper_reaction = donor_up_reaction + closed_upper_reaction
    pressure_residual = math.fsum(
        (
            inertive_pressure,
            loss_pressure,
            -driving_pressure,
            -total_lower_reaction,
            total_upper_reaction,
        )
    )

    new_state = DistributedTNodeMomentumState.from_net_flux(
        q_accepted,
        geometry=geometry,
        liquid_density=rho,
    )
    actual_momentum_change = (
        new_state.liquid_momentum - state.liquid_momentum
    )
    expected_momentum_change = area * step * math.fsum(
        (
            driving_pressure,
            -loss_pressure,
            total_lower_reaction,
            -total_upper_reaction,
        )
    )
    momentum_residual = actual_momentum_change - expected_momentum_change

    physical_gap = q_accepted + physical_downward_capacity
    lower_gap = q_accepted - lower_bound
    upper_gap = upper_bound - q_accepted
    physical_product = physical_reaction * physical_gap
    lower_product = total_lower_reaction * lower_gap
    upper_product = total_upper_reaction * upper_gap
    residual_scale = max(
        abs(driving_pressure),
        abs(inertive_pressure),
        abs(loss_pressure),
        total_lower_reaction,
        total_upper_reaction,
        1.0,
    )
    residual_tolerance = 2048.0 * math.ulp(residual_scale)
    if abs(pressure_residual) > residual_tolerance:
        raise DistributedTNodeError(
            "pressure/flux complementarity balance did not close"
        )
    if physical_gap < -tolerance or lower_gap < -tolerance or upper_gap < -tolerance:
        raise DistributedTNodeError("accepted flux violates a complementarity bound")
    product_tolerance = residual_tolerance * max(
        abs(q_accepted), abs(lower_bound), abs(upper_bound), 1.0
    )
    if max(
        abs(physical_product),
        abs(lower_product),
        abs(upper_product),
    ) > product_tolerance:
        raise DistributedTNodeError("pressure/flux complementarity product is nonzero")
    momentum_tolerance = 4096.0 * math.ulp(
        max(
            abs(state.liquid_momentum),
            abs(new_state.liquid_momentum),
            abs(expected_momentum_change),
            1.0,
        )
    )
    if abs(momentum_residual) > momentum_tolerance:
        raise DistributedTNodeError("distributed-node momentum ledger did not close")

    plan = stage_twochannel_mouth_coupling(
        q_accepted,
        phase=phase,
        geometry=mouth_geometry,
        material=material,
        wallis=wallis,
        donors=donors,
        losses=losses,
        horizontal_axial_velocity=horizontal_axial_velocity,
        horizontal_node_topology=HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT,
        legacy_activity=legacy_activity,
    )
    inventories = advance_lumped_liquid_inventories(
        donors.finite_node_volume,
        donors.riser_volume,
        time_step=step,
        plan=plan,
    )
    ledger = PressureFluxComplementarityLedger(
        old_q_net=float(q_old),
        unconstrained_q_net=float(q_unconstrained),
        accepted_q_net=float(q_accepted),
        horizontal_contact_pressure_abs=float(horizontal_pressure),
        vertical_mouth_pressure_abs=float(pressure.vertical_mouth_pressure_abs),
        driving_pressure_difference=float(driving_pressure),
        pressure_inertance=float(pressure_inertance),
        inertive_pressure_change=float(inertive_pressure),
        signed_local_loss_pressure=float(loss_pressure),
        nusselt_downward_capacity=float(
            capacity_reference.gravity_film_capacity
        ),
        wallis_downward_capacity=float(
            capacity_reference.wallis_downward_capacity
        ),
        physical_downward_capacity=float(physical_downward_capacity),
        riser_donor_downward_capacity=float(riser_donor_capacity),
        horizontal_donor_upward_capacity=float(horizontal_donor_capacity),
        lower_flux_bound=float(lower_bound),
        upper_flux_bound=float(upper_bound),
        lower_bound_owner=lower_owner,
        physical_downward_reaction_pressure=float(physical_reaction),
        donor_downward_reaction_pressure=float(donor_down_reaction),
        donor_upward_reaction_pressure=float(donor_up_reaction),
        closed_mouth_lower_reaction_pressure=float(closed_lower_reaction),
        closed_mouth_upper_reaction_pressure=float(closed_upper_reaction),
        total_lower_reaction_pressure=float(total_lower_reaction),
        total_upper_reaction_pressure=float(total_upper_reaction),
        physical_downward_reaction_force=float(area * physical_reaction),
        physical_downward_reaction_impulse=float(
            area * step * physical_reaction
        ),
        signed_constraint_reaction_force=float(
            area * (total_lower_reaction - total_upper_reaction)
        ),
        signed_constraint_reaction_impulse=float(
            area
            * step
            * (total_lower_reaction - total_upper_reaction)
        ),
        pressure_balance_residual=float(pressure_residual),
        physical_downward_gap=float(physical_gap),
        lower_bound_gap=float(lower_gap),
        upper_bound_gap=float(upper_gap),
        physical_complementarity_product=float(physical_product),
        lower_complementarity_product=float(lower_product),
        upper_complementarity_product=float(upper_product),
        old_liquid_momentum=float(state.liquid_momentum),
        new_liquid_momentum=float(new_state.liquid_momentum),
        expected_liquid_momentum_change=float(expected_momentum_change),
        actual_liquid_momentum_change=float(actual_momentum_change),
        liquid_momentum_balance_residual=float(momentum_residual),
    )
    return DistributedTNodeStepResult(
        state=new_state,
        q_net=float(q_accepted),
        mouth_plan=plan,
        inventory_update=inventories,
        complementarity=ledger,
    )


__all__ = [
    "DistributedTNodeError",
    "DistributedTNodeGeometry",
    "DistributedTNodeMomentumState",
    "DistributedTNodePressureState",
    "DistributedTNodeStepResult",
    "PressureFluxComplementarityLedger",
    "advance_distributed_tnode_inertance",
    "measured_footprint_liquid_inventory",
]
