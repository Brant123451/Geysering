"""Compressible finite-geometric T-node inventory core for Case A.

The material-front event cannot hand a pressurised ``A>A_f`` cell to an
incompressible node whose liquid volume is simply ``V_tot-V_g``.  Doing so
discards the elastic liquid inventory and creates a pressure impulse when the
discarded inventory is placed in a neighbouring cell.  This module keeps the
two required conservative inventories instead:

``m_g``
    gas mass in the finite node,

``M_l/rho_ref = integral(A dx)``
    liquid equivalent volume (liquid mass divided by the reference density).

At a common absolute pressure ``p`` their *physical occupying volumes* are

``V_g(p) = m_g c_g**2 / p``

and

``V_l(p) = V_l,eq / [1 + (p-p_ref)/(rho_l a_node**2)]``.

The node pressure is the unique admissible root of

``V_g(p) + V_l(p) = V_tot``.

For Case A, ``p_ref=p_atm``.  The horizontal-slot storage law then becomes

``A/A_f = 1 + (p_abs-p_atm)/(rho_l a**2)``

which is identical to ``1 + g(H-D)/a**2`` because
``p_abs-p_atm=rho_l g(H-D)`` at the pipe crown.

This is an inventory/occupancy core, not yet a branch boundary Riemann
solver.  One Forward-Euler stage consumes three outward gas-mass and liquid
equivalent-volume rates that were all evaluated at the same current node
pressure, updates the two conservative inventories, and solves the final
occupancy equation implicitly.  The main network integrator remains
responsible for SSP--RK stage recomputation and topology-event splitting.
No empirical split, clipping, fill, or pressure assignment is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


PRODUCTION_READY = True


class CompressibleFiniteNodeError(RuntimeError):
    """Base class for a rejected finite-node state or Euler stage."""


class OccupancyPressureError(CompressibleFiniteNodeError):
    """The inventory pair has no positive finite occupancy pressure."""


class InventoryExhaustionError(CompressibleFiniteNodeError):
    """An Euler step would make a conservative phase inventory negative."""


class InconsistentFluxPressureError(CompressibleFiniteNodeError):
    """The three supplied branch rates were not evaluated at one pressure."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class CompressibleFiniteNodeParameters:
    """Thermodynamic and elastic constants of the finite T control volume."""

    gas_sound_speed: float
    liquid_density: float
    liquid_wave_speed: float
    reference_pressure_abs: float = 101_325.0
    pressure_absolute_tolerance: float = 1.0e-8
    pressure_relative_tolerance: float = 1.0e-12
    occupancy_absolute_tolerance: float = 1.0e-15
    occupancy_relative_tolerance: float = 1.0e-12
    max_pressure_iterations: int = 160

    def __post_init__(self) -> None:
        scalars = (
            self.gas_sound_speed,
            self.liquid_density,
            self.liquid_wave_speed,
            self.reference_pressure_abs,
            self.pressure_absolute_tolerance,
            self.pressure_relative_tolerance,
            self.occupancy_absolute_tolerance,
            self.occupancy_relative_tolerance,
        )
        if not _finite(*scalars):
            raise ValueError("compressible finite-node parameters must be finite")
        if min(scalars) <= 0.0:
            raise ValueError("compressible finite-node scales must be positive")
        if self.max_pressure_iterations < 1:
            raise ValueError("at least one pressure iteration is required")

    @property
    def liquid_bulk_modulus(self) -> float:
        """Linear barotropic modulus ``rho_l a_node**2``."""

        return self.liquid_density * self.liquid_wave_speed**2


@dataclass(frozen=True)
class CompressibleFiniteNodeState:
    """Two conservative phase inventories in a fixed geometric node."""

    gas_mass: float
    liquid_equivalent_volume: float
    node_total_volume: float

    def __post_init__(self) -> None:
        values = (
            self.gas_mass,
            self.liquid_equivalent_volume,
            self.node_total_volume,
        )
        if not _finite(*values):
            raise ValueError("compressible finite-node state must be finite")
        if self.gas_mass < 0.0:
            raise ValueError("finite-node gas mass cannot be negative")
        if self.liquid_equivalent_volume < 0.0:
            raise ValueError("finite-node liquid inventory cannot be negative")
        if self.node_total_volume <= 0.0:
            raise ValueError("finite-node geometric volume must be positive")
        if self.gas_mass == 0.0 and self.liquid_equivalent_volume == 0.0:
            raise ValueError("an empty fixed node has no occupancy pressure")


@dataclass(frozen=True)
class CompressibleNodePressureState:
    """Unique same-pressure physical occupancy of one inventory state."""

    pressure_abs: float
    gas_physical_volume: float
    liquid_physical_volume: float
    liquid_storage_factor: float
    occupancy_residual: float
    iterations: int


@dataclass(frozen=True)
class CompressibleNodeBranchRates:
    """One branch's rates in a coordinate pointing away from the node.

    Positive rates leave the node and enter the branch.  Negative rates enter
    the node.  ``evaluation_pressure_abs`` records the one node pressure at
    which the external branch boundary solver evaluated this rate; all three
    branches must provide the current occupancy pressure.
    """

    gas_mass_outward: float
    liquid_equivalent_volume_outward: float
    evaluation_pressure_abs: float

    def __post_init__(self) -> None:
        if not _finite(
            self.gas_mass_outward,
            self.liquid_equivalent_volume_outward,
            self.evaluation_pressure_abs,
        ):
            raise ValueError("finite-node branch rates must be finite")
        if self.evaluation_pressure_abs <= 0.0:
            raise ValueError("branch evaluation pressure must be positive")


@dataclass(frozen=True)
class CompressibleFiniteNodeLedger:
    """Exact inventory and occupancy audit for one accepted Euler stage."""

    dt: float
    initial_state: CompressibleFiniteNodeState
    final_state: CompressibleFiniteNodeState
    initial_pressure: CompressibleNodePressureState
    final_pressure: CompressibleNodePressureState
    west: CompressibleNodeBranchRates
    east: CompressibleNodeBranchRates
    vertical: CompressibleNodeBranchRates
    gas_mass_outward_rate: float
    liquid_equivalent_volume_outward_rate: float
    expected_gas_mass_change: float
    actual_gas_mass_change: float
    gas_mass_balance_residual: float
    expected_liquid_equivalent_volume_change: float
    actual_liquid_equivalent_volume_change: float
    liquid_inventory_balance_residual: float
    fixed_geometric_volume_change: float
    initial_occupancy_residual: float
    final_occupancy_residual: float


@dataclass(frozen=True)
class CompressibleFiniteNodeEulerResult:
    """Final state, implicit pressure, and conservation ledger."""

    state: CompressibleFiniteNodeState
    pressure: CompressibleNodePressureState
    ledger: CompressibleFiniteNodeLedger


def liquid_storage_factor(
    pressure_abs: float,
    params: CompressibleFiniteNodeParameters,
) -> float:
    """Return ``A/A_ref`` for the linear elastic slot storage law."""

    if not math.isfinite(pressure_abs) or pressure_abs <= 0.0:
        raise ValueError("absolute pressure must be positive and finite")
    factor = 1.0 + (
        pressure_abs - params.reference_pressure_abs
    ) / params.liquid_bulk_modulus
    if not math.isfinite(factor) or factor <= 0.0:
        raise OccupancyPressureError(
            "pressure lies outside the positive linear-storage domain"
        )
    return float(factor)


def _occupancy_at_pressure(
    state: CompressibleFiniteNodeState,
    pressure_abs: float,
    params: CompressibleFiniteNodeParameters,
) -> tuple[float, float, float, float]:
    factor = liquid_storage_factor(pressure_abs, params)
    gas_volume = (
        state.gas_mass * params.gas_sound_speed**2 / pressure_abs
    )
    liquid_volume = state.liquid_equivalent_volume / factor
    residual = gas_volume + liquid_volume - state.node_total_volume
    return (
        float(gas_volume),
        float(liquid_volume),
        float(factor),
        float(residual),
    )


def _occupancy_tolerance(
    state: CompressibleFiniteNodeState,
    params: CompressibleFiniteNodeParameters,
) -> float:
    return (
        params.occupancy_absolute_tolerance
        + params.occupancy_relative_tolerance * state.node_total_volume
    )


def _analytic_mixed_phase_pressure(
    state: CompressibleFiniteNodeState,
    params: CompressibleFiniteNodeParameters,
    *,
    pressure_floor: float,
) -> float | None:
    """Return the admissible quadratic root, refined on the occupancy law.

    Multiplying the two EOS denominators gives a quadratic in ``p``.  The
    stable ``q`` form avoids cancellation between its roots; a few Newton
    iterations on the original (unmultiplied) occupancy equation then remove
    multiplication roundoff.  Safeguarded bisection below remains the fallback
    for an extreme coefficient set.
    """

    modulus = params.liquid_bulk_modulus
    offset = modulus - params.reference_pressure_abs
    gas_energy = state.gas_mass * params.gas_sound_speed**2
    coefficient_a = state.node_total_volume
    coefficient_b = (
        state.node_total_volume * offset
        - gas_energy
        - state.liquid_equivalent_volume * modulus
    )
    coefficient_c = -gas_energy * offset
    discriminant = (
        coefficient_b * coefficient_b
        - 4.0 * coefficient_a * coefficient_c
    )
    if not math.isfinite(discriminant) or discriminant < 0.0:
        return None
    root_discriminant = math.sqrt(discriminant)
    q_value = -0.5 * (
        coefficient_b + math.copysign(root_discriminant, coefficient_b)
    )
    candidates: list[float] = []
    if q_value != 0.0:
        candidates.extend(
            (q_value / coefficient_a, coefficient_c / q_value)
        )
    elif coefficient_b != 0.0:
        candidates.append(-coefficient_b / coefficient_a)

    admissible = [
        value
        for value in candidates
        if math.isfinite(value) and value > max(pressure_floor, 0.0)
    ]
    if not admissible:
        return None
    pressure = min(
        admissible,
        key=lambda value: abs(
            _occupancy_at_pressure(state, value, params)[3]
        ),
    )
    for _ in range(6):
        _, _, factor, residual = _occupancy_at_pressure(
            state, pressure, params
        )
        derivative = (
            -gas_energy / pressure**2
            - state.liquid_equivalent_volume / (modulus * factor**2)
        )
        if derivative == 0.0 or not math.isfinite(derivative):
            break
        candidate = pressure - residual / derivative
        if (
            not math.isfinite(candidate)
            or candidate <= max(pressure_floor, 0.0)
        ):
            break
        if candidate == pressure:
            break
        pressure = candidate
    return float(pressure)


def solve_compressible_node_pressure(
    state: CompressibleFiniteNodeState,
    params: CompressibleFiniteNodeParameters,
) -> CompressibleNodePressureState:
    """Solve the unique positive same-pressure occupancy state.

    The exact ``m_g=0`` topology event is handled by the liquid elastic
    inventory alone.  For ``m_g>0`` the occupancy residual is strictly
    decreasing throughout its admissible pressure interval, so safeguarded
    bisection gives the unique root without a pressure clip or fitted value.
    """

    modulus = params.liquid_bulk_modulus
    pressure_floor = max(
        0.0, params.reference_pressure_abs - modulus
    )
    tolerance = _occupancy_tolerance(state, params)

    if state.gas_mass == 0.0:
        if state.liquid_equivalent_volume <= 0.0:
            raise OccupancyPressureError(
                "zero-gas node requires a positive elastic liquid inventory"
            )
        pressure = (
            params.reference_pressure_abs
            + modulus
            * (
                state.liquid_equivalent_volume / state.node_total_volume
                - 1.0
            )
        )
        if pressure <= pressure_floor or pressure <= 0.0:
            raise OccupancyPressureError(
                "liquid-only inventory has no positive admissible pressure"
            )
        gas_volume, liquid_volume, factor, residual = _occupancy_at_pressure(
            state, pressure, params
        )
        if abs(residual) > tolerance:
            raise OccupancyPressureError(
                "liquid-only occupancy residual exceeds tolerance"
            )
        return CompressibleNodePressureState(
            pressure_abs=float(pressure),
            gas_physical_volume=gas_volume,
            liquid_physical_volume=liquid_volume,
            liquid_storage_factor=factor,
            occupancy_residual=residual,
            iterations=0,
        )

    analytic_pressure = _analytic_mixed_phase_pressure(
        state, params, pressure_floor=pressure_floor
    )
    if analytic_pressure is not None:
        gas_volume, liquid_volume, factor, residual = _occupancy_at_pressure(
            state, analytic_pressure, params
        )
        if abs(residual) <= tolerance:
            return CompressibleNodePressureState(
                pressure_abs=analytic_pressure,
                gas_physical_volume=gas_volume,
                liquid_physical_volume=liquid_volume,
                liquid_storage_factor=factor,
                occupancy_residual=residual,
                iterations=1,
            )

    # Approach the positive-storage boundary from within its domain.  At
    # this lower state at least one phase volume diverges, hence f(lower)>0.
    lower = math.nextafter(pressure_floor, math.inf)
    if lower <= 0.0:
        lower = math.nextafter(0.0, math.inf)

    gas_pressure_scale = (
        state.gas_mass * params.gas_sound_speed**2
        / state.node_total_volume
    )
    liquid_pressure_scale = (
        params.reference_pressure_abs
        + modulus
        * (
            state.liquid_equivalent_volume / state.node_total_volume
            - 1.0
        )
    )
    upper = max(
        params.reference_pressure_abs,
        gas_pressure_scale,
        liquid_pressure_scale,
        1.0,
    )
    if upper <= lower:
        upper = max(2.0 * lower, lower + 1.0)

    _, _, _, upper_residual = _occupancy_at_pressure(state, upper, params)
    bracket_expansions = 0
    while upper_residual > 0.0:
        upper *= 2.0
        if not math.isfinite(upper):
            raise OccupancyPressureError(
                "could not bracket a finite occupancy pressure"
            )
        _, _, _, upper_residual = _occupancy_at_pressure(
            state, upper, params
        )
        bracket_expansions += 1
        if bracket_expansions > params.max_pressure_iterations:
            raise OccupancyPressureError(
                "occupancy pressure bracket did not close"
            )

    chosen = upper
    iterations = 0
    for iterations in range(1, params.max_pressure_iterations + 1):
        middle = 0.5 * (lower + upper)
        gas_volume, liquid_volume, factor, residual = _occupancy_at_pressure(
            state, middle, params
        )
        chosen = middle
        pressure_width_tolerance = (
            params.pressure_absolute_tolerance
            + params.pressure_relative_tolerance * abs(middle)
        )
        if (
            abs(residual) <= tolerance
            and upper - lower <= pressure_width_tolerance
        ):
            return CompressibleNodePressureState(
                pressure_abs=float(middle),
                gas_physical_volume=gas_volume,
                liquid_physical_volume=liquid_volume,
                liquid_storage_factor=factor,
                occupancy_residual=residual,
                iterations=iterations,
            )
        if residual > 0.0:
            lower = middle
        else:
            upper = middle

    gas_volume, liquid_volume, factor, residual = _occupancy_at_pressure(
        state, chosen, params
    )
    raise OccupancyPressureError(
        "occupancy pressure did not converge: "
        f"p={chosen:.12g}, residual={residual:.12g}, "
        f"Vg={gas_volume:.12g}, Vl={liquid_volume:.12g}"
    )


def state_from_pressure_and_gas_mass(
    *,
    pressure_abs: float,
    gas_mass: float,
    node_total_volume: float,
    params: CompressibleFiniteNodeParameters,
) -> CompressibleFiniteNodeState:
    """Construct an exactly occupied state without prescribing its pressure.

    This helper is for initialization and tests.  The caller supplies a
    physical pressure and gas inventory; the complementary physical liquid
    volume is converted to the conservative elastic liquid inventory.
    """

    if not _finite(pressure_abs, gas_mass, node_total_volume):
        raise ValueError("finite state-construction inputs required")
    if pressure_abs <= 0.0 or gas_mass < 0.0 or node_total_volume <= 0.0:
        raise ValueError("positive pressure/volume and non-negative gas mass required")
    gas_volume = gas_mass * params.gas_sound_speed**2 / pressure_abs
    liquid_physical_volume = node_total_volume - gas_volume
    if liquid_physical_volume < 0.0:
        raise OccupancyPressureError(
            "supplied gas mass occupies more than the geometric node"
        )
    factor = liquid_storage_factor(pressure_abs, params)
    return CompressibleFiniteNodeState(
        gas_mass=float(gas_mass),
        liquid_equivalent_volume=float(liquid_physical_volume * factor),
        node_total_volume=float(node_total_volume),
    )


def _pressure_matches(
    supplied: float,
    required: float,
    params: CompressibleFiniteNodeParameters,
) -> bool:
    return math.isclose(
        supplied,
        required,
        rel_tol=params.pressure_relative_tolerance,
        abs_tol=params.pressure_absolute_tolerance,
    )


def euler_compressible_finite_node_stage(
    state: CompressibleFiniteNodeState,
    dt: float,
    *,
    west: CompressibleNodeBranchRates,
    east: CompressibleNodeBranchRates,
    vertical: CompressibleNodeBranchRates,
    params: CompressibleFiniteNodeParameters,
) -> CompressibleFiniteNodeEulerResult:
    """Advance one conservative Forward-Euler inventory stage.

    Each branch rate must have been evaluated at the current pressure returned
    by :func:`solve_compressible_node_pressure`.  Final pressure is not copied
    or extrapolated: it is obtained by solving the final two-inventory
    occupancy equation.  Exact zero gas mass is an admissible event state;
    a negative updated inventory is not and causes explicit rejection.
    """

    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("compressible finite-node dt must be finite and non-negative")
    initial_pressure = solve_compressible_node_pressure(state, params)
    branches = (west, east, vertical)
    for name, rates in zip(("west", "east", "vertical"), branches):
        if not _pressure_matches(
            rates.evaluation_pressure_abs,
            initial_pressure.pressure_abs,
            params,
        ):
            raise InconsistentFluxPressureError(
                f"{name} rates were evaluated at {rates.evaluation_pressure_abs:.12g} "
                f"Pa, not the common node pressure "
                f"{initial_pressure.pressure_abs:.12g} Pa"
            )

    gas_outward = math.fsum(rate.gas_mass_outward for rate in branches)
    liquid_outward = math.fsum(
        rate.liquid_equivalent_volume_outward for rate in branches
    )
    next_gas_mass = state.gas_mass - dt * gas_outward
    next_liquid_inventory = (
        state.liquid_equivalent_volume - dt * liquid_outward
    )
    if next_gas_mass < 0.0:
        raise InventoryExhaustionError(
            "Euler branch fluxes make node gas mass negative"
        )
    if next_liquid_inventory < 0.0:
        raise InventoryExhaustionError(
            "Euler branch fluxes make node liquid inventory negative"
        )
    if next_gas_mass == 0.0 and next_liquid_inventory == 0.0:
        raise InventoryExhaustionError(
            "Euler branch fluxes empty both node phase inventories"
        )

    next_state = CompressibleFiniteNodeState(
        gas_mass=float(next_gas_mass),
        liquid_equivalent_volume=float(next_liquid_inventory),
        node_total_volume=state.node_total_volume,
    )
    final_pressure = solve_compressible_node_pressure(next_state, params)

    expected_gas_change = -dt * gas_outward
    actual_gas_change = next_state.gas_mass - state.gas_mass
    expected_liquid_change = -dt * liquid_outward
    actual_liquid_change = (
        next_state.liquid_equivalent_volume
        - state.liquid_equivalent_volume
    )
    ledger = CompressibleFiniteNodeLedger(
        dt=float(dt),
        initial_state=state,
        final_state=next_state,
        initial_pressure=initial_pressure,
        final_pressure=final_pressure,
        west=west,
        east=east,
        vertical=vertical,
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
            next_state.node_total_volume - state.node_total_volume
        ),
        initial_occupancy_residual=float(
            initial_pressure.occupancy_residual
        ),
        final_occupancy_residual=float(final_pressure.occupancy_residual),
    )
    return CompressibleFiniteNodeEulerResult(
        state=next_state,
        pressure=final_pressure,
        ledger=ledger,
    )


__all__ = [
    "PRODUCTION_READY",
    "CompressibleFiniteNodeError",
    "CompressibleFiniteNodeEulerResult",
    "CompressibleFiniteNodeLedger",
    "CompressibleFiniteNodeParameters",
    "CompressibleFiniteNodeState",
    "CompressibleNodeBranchRates",
    "CompressibleNodePressureState",
    "InconsistentFluxPressureError",
    "InventoryExhaustionError",
    "OccupancyPressureError",
    "euler_compressible_finite_node_stage",
    "liquid_storage_factor",
    "solve_compressible_node_pressure",
    "state_from_pressure_and_gas_mass",
]
