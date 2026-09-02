"""Local conservative two-channel liquid exchange at the Case-A riser mouth.

The planar calculation resolves simultaneous upward and downward liquid at
the side-T opening.  A conventional one-dimensional boundary stores only the
net signed volume flux and therefore cannot report that counter-current
exchange.  This module supplies the smallest algebraic extension that keeps
the finite-node flux unchanged:

``Q_up = max(q_net, 0) + Q_c``

``Q_down = max(-q_net, 0) + Q_c``.

``Q_c`` is not a fitted exchange rate.  The gravity-driven Nusselt film
capacity and, when gas rises through the opening, the Wallis
counter-current-flow (CCFL) capacity constrain the **total downward gross
flow** ``Q_down``.  The liquid available in the finite T node and in the
resolved riser during the numerical step likewise constrain the corresponding
gross donor flows.  ``Q_c`` is the capacity left after the compulsory part of
the signed net flow has been accounted for.

If a supplied negative ``q_net`` alone exceeds the admissible total downward
capacity, this algebraic closure cannot preserve both that net flow and the
film/CCFL inequality.  It therefore fails closed.  The finite-node stage must
then re-solve its pressure/flux complementarity problem; clipping ``q_net``
inside this post-closure would break the node equations.

Consequently ``Q_up - Q_down == q_net`` to roundoff and the two-channel
closure cannot change either phase inventory.  The caller must apply the two
gross fluxes with opposite signs to the same node/riser pair, or apply only
their net value to a single-velocity finite-volume branch.  Momentum and
energy quantities returned here are diagnostics for a future local
two-stream mixing control volume; they must not be inserted into a
single-momentum cell without that additional state.

The module contains no physical time, event time, target liquid height, or
result-dependent switch.  ``time_step`` appears only in the donor-positivity
capacity, as it must in an explicit conservative finite-volume update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class TwoChannelMouthError(RuntimeError):
    """Base class for an inadmissible local two-channel state."""


class NetFluxExceedsDonorCapacity(TwoChannelMouthError):
    """The finite-node net flux already exhausts its physical donor."""


class NetFluxExceedsDownwardCapacity(TwoChannelMouthError):
    """A negative net flux is incompatible with the downward-flow envelope.

    This is not a request to clip the finite-node flux.  The caller must
    re-solve the node pressure/flux complementarity problem with the active
    downward-capacity inequality.
    """

    def __init__(self, q_net: float, downward_capacity: float) -> None:
        self.q_net = float(q_net)
        self.downward_capacity = float(downward_capacity)
        super().__init__(
            "negative finite-node net flux exceeds the admissible total "
            "downward mouth capacity; re-solve the finite-node "
            "pressure/flux complementarity problem"
        )


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class VerticalMouthGeometry:
    """Measured vertical-mouth geometry and gravity."""

    diameter: float
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if not _finite(self.diameter, self.gravity):
            raise ValueError("vertical-mouth geometry must be finite")
        if self.diameter <= 0.0 or self.gravity <= 0.0:
            raise ValueError("vertical-mouth diameter and gravity must be positive")

    @property
    def full_area(self) -> float:
        return 0.25 * math.pi * self.diameter**2


@dataclass(frozen=True)
class VerticalMouthPhaseState:
    """Cross-section-averaged phase state at the first riser face.

    Both velocities use the vertical coordinate, positive upward.  Areas must
    partition the complete measured mouth.  A positive gas velocity is thus
    the only state in which a counter-current downward liquid film can be
    assigned a nonzero Wallis exchange capacity.
    """

    liquid_area: float
    liquid_velocity: float
    gas_area: float
    gas_velocity: float

    def validate(self, geometry: VerticalMouthGeometry) -> None:
        values = (
            self.liquid_area,
            self.liquid_velocity,
            self.gas_area,
            self.gas_velocity,
        )
        if not _finite(*values):
            raise ValueError("vertical-mouth phase state must be finite")
        if self.liquid_area < 0.0 or self.gas_area < 0.0:
            raise ValueError("vertical-mouth phase areas cannot be negative")
        tolerance = 128.0 * math.ulp(max(geometry.full_area, 1.0))
        if not math.isclose(
            self.liquid_area + self.gas_area,
            geometry.full_area,
            rel_tol=1.0e-12,
            abs_tol=tolerance,
        ):
            raise ValueError("liquid and gas areas must partition the full mouth")


@dataclass(frozen=True)
class VerticalMouthMaterialProperties:
    """Gas/liquid properties used by the film and Wallis closures."""

    liquid_density: float
    gas_density: float
    liquid_dynamic_viscosity: float

    def __post_init__(self) -> None:
        values = (
            self.liquid_density,
            self.gas_density,
            self.liquid_dynamic_viscosity,
        )
        if not _finite(*values):
            raise ValueError("vertical-mouth material properties must be finite")
        if min(values) <= 0.0:
            raise ValueError("vertical-mouth material properties must be positive")
        if self.gas_density >= self.liquid_density:
            raise ValueError("gas density must be lower than liquid density")


@dataclass(frozen=True)
class WallisCounterCurrentParameters:
    """Dimensionless Wallis flooding-envelope coefficients."""

    constant: float
    slope: float = 1.0

    def __post_init__(self) -> None:
        if not _finite(self.constant, self.slope):
            raise ValueError("Wallis parameters must be finite")
        if self.constant <= 0.0 or self.slope <= 0.0:
            raise ValueError("Wallis parameters must be positive")


@dataclass(frozen=True)
class DirectionalMouthLosses:
    """Non-negative loss coefficients used only in the energy ledger.

    The upward and downward turn coefficients remain separate so a
    counter-current glug loss is never silently applied to co-current upward
    liquid.  ``countercurrent_mixing`` acts only on the equal-and-opposite
    circulation ``Q_c``.
    """

    upward_turn: float
    downward_turn: float
    countercurrent_mixing: float

    def __post_init__(self) -> None:
        values = (
            self.upward_turn,
            self.downward_turn,
            self.countercurrent_mixing,
        )
        if not _finite(*values):
            raise ValueError("directional mouth losses must be finite")
        if min(values) < 0.0:
            raise ValueError("directional mouth losses cannot be negative")


@dataclass(frozen=True)
class LiquidDonorInventories:
    """Liquid volumes available on each side of the mouth for one FV step."""

    finite_node_volume: float
    riser_volume: float
    time_step: float

    def __post_init__(self) -> None:
        values = (
            self.finite_node_volume,
            self.riser_volume,
            self.time_step,
        )
        if not _finite(*values):
            raise ValueError("liquid donor inventories must be finite")
        if self.finite_node_volume < 0.0 or self.riser_volume < 0.0:
            raise ValueError("liquid donor inventories cannot be negative")
        if self.time_step <= 0.0:
            raise ValueError("donor-capacity time step must be positive")

    @property
    def finite_node_rate_capacity(self) -> float:
        return self.finite_node_volume / self.time_step

    @property
    def riser_rate_capacity(self) -> float:
        return self.riser_volume / self.time_step


@dataclass(frozen=True)
class TwoChannelMouthResult:
    """Gross exchange, exact net flux, and local momentum/energy ledger."""

    q_net: float
    upward_flow: float
    downward_flow: float
    circulation_flow: float
    closure_residual: float
    film_thickness: float
    gravity_film_capacity: float
    wallis_downward_capacity: float
    downward_physical_capacity: float
    downward_physical_circulation_capacity: float
    finite_node_circulation_capacity: float
    riser_circulation_capacity: float
    gas_superficial_velocity: float
    wallis_gas_parameter: float
    upward_channel_area: float
    downward_channel_area: float
    upward_channel_velocity: float
    downward_channel_velocity: float
    resolved_liquid_velocity: float
    resolved_net_flux_mismatch: float
    gross_convective_momentum_flux: float
    bulk_convective_momentum_flux: float
    countercurrent_momentum_excess: float
    gross_kinetic_power: float
    signed_kinetic_energy_flux: float
    upward_turn_loss_power: float
    downward_turn_loss_power: float
    countercurrent_mixing_loss_power: float
    total_dissipation_power: float


def _annular_film_capacity(
    liquid_area: float,
    *,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
) -> tuple[float, float]:
    """Return Nusselt gravity-film thickness and downward volume capacity."""

    area = min(max(float(liquid_area), 0.0), geometry.full_area)
    if area == 0.0:
        return 0.0, 0.0
    radius = 0.5 * geometry.diameter
    core_radius = math.sqrt(max(radius * radius - area / math.pi, 0.0))
    thickness = radius - core_radius
    flow = (
        math.pi
        * geometry.gravity
        * geometry.diameter
        * (material.liquid_density - material.gas_density)
        * thickness**3
        / (3.0 * material.liquid_dynamic_viscosity)
    )
    return float(thickness), float(max(flow, 0.0))


def _wallis_downward_capacity(
    phase: VerticalMouthPhaseState,
    *,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
) -> tuple[float, float, float]:
    """Return gas superficial velocity, ``Jg*``, and liquid CCFL capacity."""

    if phase.gas_area == 0.0 or phase.gas_velocity <= 0.0:
        return 0.0, 0.0, 0.0
    gas_superficial_velocity = (
        phase.gas_velocity * phase.gas_area / geometry.full_area
    )
    density_difference = material.liquid_density - material.gas_density
    gas_scale = math.sqrt(
        material.gas_density
        / (geometry.gravity * geometry.diameter * density_difference)
    )
    jg_star = gas_superficial_velocity * gas_scale
    remaining = max(wallis.constant - math.sqrt(max(jg_star, 0.0)), 0.0)
    jl_star_capacity = (remaining / wallis.slope) ** 2
    liquid_velocity_scale = math.sqrt(
        geometry.gravity
        * geometry.diameter
        * density_difference
        / material.liquid_density
    )
    capacity = jl_star_capacity * liquid_velocity_scale * geometry.full_area
    return (
        float(gas_superficial_velocity),
        float(jg_star),
        float(max(capacity, 0.0)),
    )


def close_vertical_mouth_twochannel_exchange(
    q_net: float,
    *,
    phase: VerticalMouthPhaseState,
    geometry: VerticalMouthGeometry,
    material: VerticalMouthMaterialProperties,
    wallis: WallisCounterCurrentParameters,
    donors: LiquidDonorInventories,
    losses: DirectionalMouthLosses,
) -> TwoChannelMouthResult:
    """Close one local gross counter-current exchange conservatively.

    ``q_net`` is the signed finite-node liquid flux, positive from the node
    into the riser.  It is never clipped or fitted here.  If that already
    exceeds the corresponding donor inventory, the state is rejected so the
    caller can reduce its finite-volume step.  If a negative ``q_net`` already
    exceeds the gravity/CCFL envelope for total downward gross flow, the state
    is also rejected; the finite node must then re-solve its coupled
    pressure/flux complementarity condition rather than clip the flux here.

    The returned stream areas are a diagnostic, parameter-free partition of
    the resolved liquid mouth area in proportion to the two gross flows.  It
    gives both channels the same speed magnitude and makes their summed area
    exactly the resolved liquid area.  A production momentum coupling must
    retain a local two-stream state; these diagnostics are not permission to
    inject a counter-current Reynolds stress into a single-velocity cell.
    """

    if not math.isfinite(q_net):
        raise ValueError("finite-node net liquid flux must be finite")
    phase.validate(geometry)

    q = float(q_net)
    base_upward = max(q, 0.0)
    base_downward = max(-q, 0.0)
    node_rate = donors.finite_node_rate_capacity
    riser_rate = donors.riser_rate_capacity
    tolerance = 128.0 * math.ulp(max(abs(q), node_rate, riser_rate, 1.0))
    if base_upward > node_rate + tolerance:
        raise NetFluxExceedsDonorCapacity(
            "positive finite-node net flux exhausts node liquid inventory"
        )
    if base_downward > riser_rate + tolerance:
        raise NetFluxExceedsDonorCapacity(
            "negative finite-node net flux exhausts riser liquid inventory"
        )
    if phase.liquid_area == 0.0 and q != 0.0:
        raise NetFluxExceedsDonorCapacity(
            "nonzero liquid net flux has no liquid-open mouth area"
        )

    film_thickness, gravity_capacity = _annular_film_capacity(
        phase.liquid_area,
        geometry=geometry,
        material=material,
    )
    gas_superficial_velocity, jg_star, wallis_capacity = (
        _wallis_downward_capacity(
            phase,
            geometry=geometry,
            material=material,
            wallis=wallis,
        )
    )
    wallis_active = phase.gas_area > 0.0 and phase.gas_velocity > 0.0
    downward_physical_capacity = gravity_capacity
    if wallis_active:
        downward_physical_capacity = min(
            downward_physical_capacity,
            wallis_capacity,
        )
    if base_downward > downward_physical_capacity + tolerance:
        raise NetFluxExceedsDownwardCapacity(q, downward_physical_capacity)

    # The gravity/Wallis envelope constrains Q_down itself, not merely the
    # equal-and-opposite circulation.  With q_net fixed, the compulsory
    # downward part is max(-q_net, 0), so only the residual envelope may be
    # assigned to Q_c.  Wallis is inactive when no gas rises; that state has no
    # counter-current driver, although a one-way gravity-driven downward net
    # flux remains admissible up to the Nusselt capacity.
    downward_physical_circulation_capacity = max(
        downward_physical_capacity - base_downward,
        0.0,
    )
    if not wallis_active:
        downward_physical_circulation_capacity = 0.0

    node_circulation_capacity = max(node_rate - base_upward, 0.0)
    riser_circulation_capacity = max(riser_rate - base_downward, 0.0)
    circulation = min(
        downward_physical_circulation_capacity,
        node_circulation_capacity,
        riser_circulation_capacity,
    )
    circulation = max(float(circulation), 0.0)

    if q >= 0.0:
        downward = circulation
        upward = q + downward
    else:
        upward = circulation
        downward = upward - q
    closure_residual = math.fsum((upward, -downward, -q))
    closure_tolerance = 128.0 * math.ulp(
        max(abs(upward), abs(downward), abs(q), 1.0)
    )
    if abs(closure_residual) > closure_tolerance:
        raise FloatingPointError("two-channel exchange failed to preserve q_net")

    gross = upward + downward
    liquid_area = phase.liquid_area
    if gross > 0.0 and liquid_area > 0.0:
        channel_speed = gross / liquid_area
        upward_area = liquid_area * upward / gross
        downward_area = liquid_area - upward_area
        upward_velocity = channel_speed if upward > 0.0 else 0.0
        downward_velocity = -channel_speed if downward > 0.0 else 0.0
    else:
        upward_area = 0.0
        downward_area = 0.0
        upward_velocity = 0.0
        downward_velocity = 0.0

    rho = material.liquid_density
    gross_momentum = rho * (
        upward * upward_velocity
        + downward * abs(downward_velocity)
    )
    bulk_momentum = (
        rho * q * q / liquid_area if liquid_area > 0.0 else 0.0
    )
    momentum_excess = max(gross_momentum - bulk_momentum, 0.0)
    gross_kinetic_power = 0.5 * rho * (
        upward * upward_velocity**2
        + downward * downward_velocity**2
    )
    signed_kinetic_flux = 0.5 * rho * (
        upward * upward_velocity**2
        - downward * downward_velocity**2
    )
    upward_loss = (
        0.5
        * rho
        * losses.upward_turn
        * upward
        * upward_velocity**2
    )
    downward_loss = (
        0.5
        * rho
        * losses.downward_turn
        * downward
        * downward_velocity**2
    )
    relative_velocity = upward_velocity - downward_velocity
    mixing_loss = (
        0.5
        * rho
        * losses.countercurrent_mixing
        * circulation
        * relative_velocity**2
    )
    total_dissipation = upward_loss + downward_loss + mixing_loss

    return TwoChannelMouthResult(
        q_net=q,
        upward_flow=float(upward),
        downward_flow=float(downward),
        circulation_flow=circulation,
        closure_residual=float(closure_residual),
        film_thickness=film_thickness,
        gravity_film_capacity=gravity_capacity,
        wallis_downward_capacity=wallis_capacity,
        downward_physical_capacity=float(downward_physical_capacity),
        downward_physical_circulation_capacity=float(
            downward_physical_circulation_capacity
        ),
        finite_node_circulation_capacity=node_circulation_capacity,
        riser_circulation_capacity=riser_circulation_capacity,
        gas_superficial_velocity=gas_superficial_velocity,
        wallis_gas_parameter=jg_star,
        upward_channel_area=float(upward_area),
        downward_channel_area=float(downward_area),
        upward_channel_velocity=float(upward_velocity),
        downward_channel_velocity=float(downward_velocity),
        resolved_liquid_velocity=float(phase.liquid_velocity),
        resolved_net_flux_mismatch=float(
            q - phase.liquid_area * phase.liquid_velocity
        ),
        gross_convective_momentum_flux=float(gross_momentum),
        bulk_convective_momentum_flux=float(bulk_momentum),
        countercurrent_momentum_excess=float(momentum_excess),
        gross_kinetic_power=float(gross_kinetic_power),
        signed_kinetic_energy_flux=float(signed_kinetic_flux),
        upward_turn_loss_power=float(upward_loss),
        downward_turn_loss_power=float(downward_loss),
        countercurrent_mixing_loss_power=float(mixing_loss),
        total_dissipation_power=float(total_dissipation),
    )


__all__ = [
    "DirectionalMouthLosses",
    "LiquidDonorInventories",
    "NetFluxExceedsDownwardCapacity",
    "NetFluxExceedsDonorCapacity",
    "TwoChannelMouthError",
    "TwoChannelMouthResult",
    "VerticalMouthGeometry",
    "VerticalMouthMaterialProperties",
    "VerticalMouthPhaseState",
    "WallisCounterCurrentParameters",
    "close_vertical_mouth_twochannel_exchange",
]
