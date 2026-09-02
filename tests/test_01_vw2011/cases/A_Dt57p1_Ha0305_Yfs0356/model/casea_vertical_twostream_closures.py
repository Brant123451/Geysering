"""Physical closure adapters for the Case-A post-breakthrough riser.

The persistent liquid state is the conservative two-stream state defined in
``casea_vertical_twostream_fv``.  This module supplies the pieces that are
needed around that interior operator:

* a coaxial core--gas--wall-film geometry computed from the *current* cell
  inventories;
* an isothermal gas-void/EOS adapter and common pressure-face reconstruction;
* an atmospheric, liquid-outflow-only upper boundary; and
* one callable post-event stage which transports the two liquid inventories
  and then applies the conservative three-body drag exchange.

There is deliberately no result-derived film-area set point.  Core and film
areas remain prognostic inventories: they change through shared finite-volume
face fluxes, the geometric moving-Taylor-interface event, and conservative
local topology transfer when a labelled stream stops or reverses.  In
particular, this file contains no elapsed-time branch, requested water depth,
reference field, or rendering correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from casea_vertical_twostream_fv import (
    DirectionalBoundaryFlux,
    PhysicalGasInterphaseState,
    PhysicalThreeBodyDragResult,
    StateAdmissibilityError,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    VerticalTwoStreamStepResult,
    advance_vertical_two_stream_fv,
    implicit_physical_three_body_drag_exchange,
)


POST_EVENT_TWOSTREAM_CLOSURES_READY = True


class VerticalTwoStreamClosureError(RuntimeError):
    """Base class for a rejected physical-closure stage."""


class GasVoidStateError(VerticalTwoStreamClosureError):
    """Gas mass, momentum, and geometric void are mutually inconsistent."""


@dataclass(frozen=True)
class TaylorSweepExtensionResult:
    """Conservative incremental activation of newly Taylor-swept material."""

    state: VerticalTwoStreamState
    previous_swept_fraction: tuple[float, ...]
    new_swept_fraction: tuple[float, ...]
    newly_swept_fraction: tuple[float, ...]
    added_film_area: tuple[float, ...]
    added_film_discharge: tuple[float, ...]
    area_residual: tuple[float, ...]
    momentum_residual: tuple[float, ...]


@dataclass(frozen=True)
class TaylorGasCoreDisplacementResult:
    """Conservative liquid displacement caused by newly opened gas core."""

    state: VerticalTwoStreamState
    requested_core_area: tuple[float, ...]
    opened_core_area: tuple[float, ...]
    source_shortfall_area: tuple[float, ...]
    protected_film_area: tuple[float, ...]
    film_requirement_shortfall_area: tuple[float, ...]
    removed_upward_area: tuple[float, ...]
    removed_upward_discharge: tuple[float, ...]
    removed_downward_area: tuple[float, ...]
    removed_downward_discharge: tuple[float, ...]
    received_upward_area: tuple[float, ...]
    received_upward_discharge: tuple[float, ...]
    received_downward_area: tuple[float, ...]
    received_downward_discharge: tuple[float, ...]
    upward_overflow_liquid_volume: float
    upward_overflow_kinematic_momentum: float
    downward_overflow_liquid_volume: float
    downward_overflow_kinematic_momentum: float
    overflow_liquid_volume: float
    overflow_kinematic_momentum: float
    requested_core_volume: float
    opened_core_volume: float
    source_shortfall_volume: float
    deposited_liquid_volume: float
    initial_liquid_volume: float
    final_liquid_volume: float
    initial_axial_kinematic_momentum: float
    final_axial_kinematic_momentum: float
    volume_residual_including_overflow: float
    momentum_residual_including_overflow: float

    @property
    def has_overflow(self) -> bool:
        return self.overflow_liquid_volume > 0.0

    @property
    def gas_core_fully_opened(self) -> bool:
        return self.source_shortfall_volume == 0.0


@dataclass(frozen=True)
class TaylorSweepGeometryResult:
    """Recommended film-formation plus gas-core-opening event result."""

    film_extension: TaylorSweepExtensionResult
    gas_core_displacement: TaylorGasCoreDisplacementResult

    @property
    def state(self) -> VerticalTwoStreamState:
        return self.gas_core_displacement.state


def extend_taylor_sweep_in_persistent_state(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    previous_swept_fraction: Iterable[float],
    new_swept_fraction: Iterable[float],
    taylor_core_area_fraction: float,
    taylor_rise_velocity: float,
) -> TaylorSweepExtensionResult:
    """Activate only the material newly swept by the rising Taylor core.

    The persistent two-stream state is initialized once, when gas first
    connects through the T mouth.  During the subsequent rise, this function
    expands the wall-film corridor cell by cell.  A cell whose swept fraction
    did not increase is copied exactly; an already swept cell is therefore
    never reconstructed from the legacy net state.

    For an increment ``dS``, at most

    ``dA_f = dS (1-alpha_core) A_r``

    is transferred from the current upward inventory to the film inventory.
    The new film material receives the Davies--Taylor return velocity
    ``-alpha_core U_T/(1-alpha_core)``.  The equal and opposite discharge is
    retained by the upward liquid while that labelled stream has finite area.
    If the transfer exhausts it, the vanishing label is collapsed locally into
    the remaining directional stream according to the signed total discharge.
    Thus no dry stream carries momentum, while cell liquid area and total axial
    momentum remain exactly conservative.  Pressure work at the moving
    material interface may change the split kinetic energy; no unrecorded mass
    is created.
    """

    if state.cell_count != parameters.cell_count:
        raise StateAdmissibilityError("state and Taylor sweep cell counts differ")
    old = _finite_tuple(previous_swept_fraction, name="previous_swept_fraction")
    new = _finite_tuple(new_swept_fraction, name="new_swept_fraction")
    if len(old) != parameters.cell_count or len(new) != parameters.cell_count:
        raise ValueError("Taylor sweep vectors need one value per riser cell")
    if not math.isfinite(taylor_core_area_fraction) or not (
        0.0 < taylor_core_area_fraction < 1.0
    ):
        raise ValueError("taylor_core_area_fraction must lie in (0, 1)")
    if not math.isfinite(taylor_rise_velocity) or taylor_rise_velocity <= 0.0:
        raise ValueError("taylor_rise_velocity must be finite and positive")
    tolerance = 128.0 * math.ulp(1.0)
    for cell, (old_fraction, new_fraction) in enumerate(zip(old, new)):
        if (
            old_fraction < 0.0
            or old_fraction > 1.0
            or new_fraction < 0.0
            or new_fraction > 1.0
        ):
            raise ValueError(f"Taylor sweep fraction outside [0, 1] in cell {cell}")
        if new_fraction + tolerance < old_fraction:
            raise ValueError(f"Taylor sweep cannot retreat in cell {cell}")

    film_velocity = -(
        taylor_core_area_fraction
        * taylor_rise_velocity
        / (1.0 - taylor_core_area_fraction)
    )
    up_area = list(state.upward_area)
    up_discharge = list(state.upward_discharge)
    down_area = list(state.downward_area)
    down_discharge = list(state.downward_discharge)
    increments: list[float] = []
    added_area: list[float] = []
    added_discharge: list[float] = []
    area_residual: list[float] = []
    momentum_residual: list[float] = []

    for cell, (old_fraction, new_fraction) in enumerate(zip(old, new)):
        increment = max(new_fraction - old_fraction, 0.0)
        increments.append(increment)
        if increment <= tolerance:
            added_area.append(0.0)
            added_discharge.append(0.0)
            area_residual.append(0.0)
            momentum_residual.append(0.0)
            continue
        old_total_area = up_area[cell] + down_area[cell]
        old_total_discharge = up_discharge[cell] + down_discharge[cell]
        geometric_increment = (
            increment
            * (1.0 - taylor_core_area_fraction)
            * parameters.full_area
        )
        transfer_area = min(geometric_increment, up_area[cell])
        transfer_discharge = transfer_area * film_velocity
        up_area[cell] -= transfer_area
        down_area[cell] += transfer_area
        # The negative film discharge is balanced by the core.  This is the
        # conservative moving-interface impulse, not a second mouth flux.
        down_discharge[cell] += transfer_discharge
        up_discharge[cell] -= transfer_discharge
        # A geometric film increment can consume the last upward-liquid area.
        # Retaining the equal-and-opposite Taylor impulse on that now-dry label
        # creates an inadmissible zero-area/nonzero-momentum state.  Resolve the
        # local topology event immediately: the complete liquid inventory is
        # represented by the channel matching its signed total discharge.  In
        # the usual Case-A event the total remains downward, so only the tiny
        # orphaned upward impulse is mixed into the finite falling film.
        dry_tolerance = max(parameters.dry_area_tolerance, tolerance)
        if up_area[cell] <= dry_tolerance:
            total_area = up_area[cell] + down_area[cell]
            total_discharge = up_discharge[cell] + down_discharge[cell]
            if total_discharge <= 0.0:
                up_area[cell] = 0.0
                up_discharge[cell] = 0.0
                down_area[cell] = total_area
                down_discharge[cell] = total_discharge
            else:
                up_area[cell] = total_area
                up_discharge[cell] = total_discharge
                down_area[cell] = 0.0
                down_discharge[cell] = 0.0
        added_area.append(transfer_area)
        added_discharge.append(transfer_discharge)
        area_residual.append(
            up_area[cell] + down_area[cell] - old_total_area
        )
        momentum_residual.append(
            up_discharge[cell] + down_discharge[cell] - old_total_discharge
        )

    mapped = VerticalTwoStreamState.from_iterables(
        upward_area=up_area,
        upward_discharge=up_discharge,
        downward_area=down_area,
        downward_discharge=down_discharge,
    )
    return TaylorSweepExtensionResult(
        state=mapped,
        previous_swept_fraction=old,
        new_swept_fraction=new,
        newly_swept_fraction=tuple(increments),
        added_film_area=tuple(added_area),
        added_film_discharge=tuple(added_discharge),
        area_residual=tuple(area_residual),
        momentum_residual=tuple(momentum_residual),
    )


def _require_bottom_contiguous_sweep(
    fractions: tuple[float, ...],
    *,
    tolerance: float,
    name: str,
) -> None:
    """Require full cells, at most one cut cell, then unswept cells."""

    for cell in range(1, len(fractions)):
        if fractions[cell] > tolerance and fractions[cell - 1] < 1.0 - tolerance:
            raise ValueError(
                f"{name} is not a bottom-connected material front at cell {cell}"
            )


def displace_taylor_core_liquid_into_unswept_cells(
    state_after_film: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    previous_swept_fraction: Iterable[float],
    new_swept_fraction: Iterable[float],
    taylor_core_area_fraction: float,
) -> TaylorGasCoreDisplacementResult:
    """Open newly swept gas core and displace its liquid conservatively.

    This function is called immediately after
    :func:`extend_taylor_sweep_in_persistent_state`.  In each newly swept
    source cell it requests only the *additional* gas-core area that is not
    already present after the finite-volume film/gas update,

    ``dA_g = max(alpha_core A_r S_new - A_g, 0)``.

    Here ``A_g = A_r-A_up-A_down`` is measured from ``state_after_film``.
    Consequently, gas void already opened by the resolved gas/liquid finite-
    volume step is credited to the Taylor core instead of causing the same
    liquid volume to be displaced a second time.  When the existing void is
    exactly the previous swept-core area, this reduces to the usual increment
    ``alpha_core A_r (S_new-S_old)``.

    The area is removed first from the source cell's remaining upward liquid.
    If that is insufficient, it is removed from downward liquid in excess of
    the cumulative geometric film requirement

    ``S_new (1-alpha_core) A_r``.

    Thus an initially full downward-moving cell can still become an annular
    film plus gas core.  Every removed parcel retains its signed stream
    velocity.  Displaced upward and downward parcels are placed into their
    matching directional streams, nearest unswept receiver first, without an
    artificial moving-interface impulse.

    If the unswept receiver cells cannot hold all displaced liquid, the
    remainder is returned as ``overflow_liquid_volume`` and
    ``overflow_kinematic_momentum``.  The caller must route that explicit
    inventory through the atmospheric rim or reject the global stage; it must
    never discard it.  The returned combined state-plus-overflow ledgers close
    volume and axial momentum to roundoff.
    """

    if state_after_film.cell_count != parameters.cell_count:
        raise StateAdmissibilityError("state and Taylor displacement cell counts differ")
    old = _finite_tuple(previous_swept_fraction, name="previous_swept_fraction")
    new = _finite_tuple(new_swept_fraction, name="new_swept_fraction")
    n = parameters.cell_count
    if len(old) != n or len(new) != n:
        raise ValueError("Taylor sweep vectors need one value per riser cell")
    if not math.isfinite(taylor_core_area_fraction) or not (
        0.0 < taylor_core_area_fraction < 1.0
    ):
        raise ValueError("taylor_core_area_fraction must lie in (0, 1)")
    fraction_tolerance = 128.0 * math.ulp(1.0)
    for cell, (old_fraction, new_fraction) in enumerate(zip(old, new)):
        if (
            old_fraction < 0.0
            or old_fraction > 1.0
            or new_fraction < 0.0
            or new_fraction > 1.0
        ):
            raise ValueError(f"Taylor sweep fraction outside [0, 1] in cell {cell}")
        if new_fraction + fraction_tolerance < old_fraction:
            raise ValueError(f"Taylor sweep cannot retreat in cell {cell}")
    _require_bottom_contiguous_sweep(
        old,
        tolerance=fraction_tolerance,
        name="previous_swept_fraction",
    )
    _require_bottom_contiguous_sweep(
        new,
        tolerance=fraction_tolerance,
        name="new_swept_fraction",
    )

    full_area = parameters.full_area
    dz = parameters.cell_length
    area_tolerance = parameters.dry_area_tolerance
    up_area = list(state_after_film.upward_area)
    up_discharge = list(state_after_film.upward_discharge)
    down_area = list(state_after_film.downward_area)
    down_discharge = list(state_after_film.downward_discharge)
    requested = [0.0] * n
    opened = [0.0] * n
    shortfall = [0.0] * n
    protected_film_area = [0.0] * n
    film_requirement_shortfall = [0.0] * n
    removed_up_area = [0.0] * n
    removed_up_discharge = [0.0] * n
    removed_down_area = [0.0] * n
    removed_down_discharge = [0.0] * n

    initial_volume = dz * math.fsum(state_after_film.liquid_area)
    initial_momentum = dz * math.fsum(state_after_film.liquid_discharge)

    for cell, (old_fraction, new_fraction) in enumerate(zip(old, new)):
        increment = max(new_fraction - old_fraction, 0.0)
        if increment <= fraction_tolerance:
            continue
        target_core_area = taylor_core_area_fraction * full_area * new_fraction
        existing_gas_area = max(
            full_area - up_area[cell] - down_area[cell],
            0.0,
        )
        request = max(target_core_area - existing_gas_area, 0.0)
        available_up = up_area[cell]
        removal_up = min(request, available_up)
        if removal_up > 0.0 and available_up > 0.0:
            parcel_velocity_up = up_discharge[cell] / available_up
            parcel_discharge_up = removal_up * parcel_velocity_up
        else:
            parcel_discharge_up = 0.0
        up_area[cell] = available_up - removal_up
        up_discharge[cell] -= parcel_discharge_up
        if removal_up >= available_up:
            up_area[cell] = 0.0
            up_discharge[cell] = 0.0

        remaining_request = request - removal_up
        film_requirement = (
            new_fraction * (1.0 - taylor_core_area_fraction) * full_area
        )
        protected_film = min(down_area[cell], film_requirement)
        removable_down = max(down_area[cell] - protected_film, 0.0)
        removal_down = min(remaining_request, removable_down)
        if removal_down > 0.0 and down_area[cell] > 0.0:
            parcel_velocity_down = down_discharge[cell] / down_area[cell]
            parcel_discharge_down = removal_down * parcel_velocity_down
        else:
            parcel_discharge_down = 0.0
        old_down_area = down_area[cell]
        down_area[cell] -= removal_down
        down_discharge[cell] -= parcel_discharge_down
        if removal_down >= old_down_area:
            down_area[cell] = 0.0
            down_discharge[cell] = 0.0

        removal = removal_up + removal_down
        requested[cell] = request
        opened[cell] = removal
        shortfall[cell] = request - removal
        protected_film_area[cell] = protected_film
        film_requirement_shortfall[cell] = max(
            film_requirement - protected_film,
            0.0,
        )
        removed_up_area[cell] = removal_up
        removed_up_discharge[cell] = parcel_discharge_up
        removed_down_area[cell] = removal_down
        removed_down_discharge[cell] = parcel_discharge_down

    displaced_area = math.fsum(opened)
    remaining_up_area = math.fsum(removed_up_area)
    remaining_up_discharge = math.fsum(removed_up_discharge)
    remaining_down_area = math.fsum(removed_down_area)
    remaining_down_discharge = math.fsum(removed_down_discharge)
    received_up_area = [0.0] * n
    received_up_discharge = [0.0] * n
    received_down_area = [0.0] * n
    received_down_discharge = [0.0] * n

    swept_cells = [cell for cell, fraction in enumerate(new) if fraction > fraction_tolerance]
    first_receiver = 0 if not swept_cells else max(swept_cells) + 1
    for cell in range(first_receiver, n):
        remaining_area = remaining_up_area + remaining_down_area
        if new[cell] > fraction_tolerance or remaining_area <= area_tolerance:
            continue
        capacity = max(full_area - up_area[cell] - down_area[cell], 0.0)
        deposit_total = min(capacity, remaining_area)
        if deposit_total <= 0.0:
            continue
        # Share receiver capacity in the remaining parcel-area proportion;
        # neither direction receives an arbitrary priority.
        if remaining_down_area <= 0.0:
            deposit_up = deposit_total
            deposit_down = 0.0
        elif remaining_up_area <= 0.0:
            deposit_up = 0.0
            deposit_down = deposit_total
        else:
            deposit_up = deposit_total * remaining_up_area / remaining_area
            deposit_down = max(deposit_total - deposit_up, 0.0)
        parcel_discharge_up = (
            remaining_up_discharge * deposit_up / remaining_up_area
            if remaining_up_area > 0.0
            else 0.0
        )
        parcel_discharge_down = (
            remaining_down_discharge * deposit_down / remaining_down_area
            if remaining_down_area > 0.0
            else 0.0
        )
        up_area[cell] += deposit_up
        up_discharge[cell] += parcel_discharge_up
        down_area[cell] += deposit_down
        down_discharge[cell] += parcel_discharge_down
        received_up_area[cell] = deposit_up
        received_up_discharge[cell] = parcel_discharge_up
        received_down_area[cell] = deposit_down
        received_down_discharge[cell] = parcel_discharge_down
        remaining_up_area -= deposit_up
        remaining_up_discharge -= parcel_discharge_up
        remaining_down_area -= deposit_down
        remaining_down_discharge -= parcel_discharge_down
        if remaining_up_area <= 0.0:
            remaining_up_area = 0.0
            remaining_up_discharge = 0.0
        if remaining_down_area <= 0.0:
            remaining_down_area = 0.0
            remaining_down_discharge = 0.0

    area_roundoff = 1024.0 * math.ulp(max(displaced_area, full_area, 1.0e-30))
    if remaining_up_area <= area_roundoff:
        remaining_up_area = 0.0
        remaining_up_discharge = 0.0
    if remaining_down_area <= area_roundoff:
        remaining_down_area = 0.0
        remaining_down_discharge = 0.0

    mapped = VerticalTwoStreamState.from_iterables(
        upward_area=up_area,
        upward_discharge=up_discharge,
        downward_area=down_area,
        downward_discharge=down_discharge,
    )

    upward_overflow_volume = remaining_up_area * dz
    upward_overflow_momentum = remaining_up_discharge * dz
    downward_overflow_volume = remaining_down_area * dz
    downward_overflow_momentum = remaining_down_discharge * dz
    overflow_volume = upward_overflow_volume + downward_overflow_volume
    overflow_momentum = upward_overflow_momentum + downward_overflow_momentum
    final_volume = dz * math.fsum(mapped.liquid_area)
    final_momentum = dz * math.fsum(mapped.liquid_discharge)
    volume_residual = final_volume + overflow_volume - initial_volume
    momentum_residual = final_momentum + overflow_momentum - initial_momentum
    ledger_tolerance = 1024.0 * math.ulp(
        max(initial_volume, abs(initial_momentum), 1.0)
    )
    if max(abs(volume_residual), abs(momentum_residual)) > ledger_tolerance:
        raise FloatingPointError("Taylor gas-core displacement ledger does not close")

    return TaylorGasCoreDisplacementResult(
        state=mapped,
        requested_core_area=tuple(requested),
        opened_core_area=tuple(opened),
        source_shortfall_area=tuple(shortfall),
        protected_film_area=tuple(protected_film_area),
        film_requirement_shortfall_area=tuple(film_requirement_shortfall),
        removed_upward_area=tuple(removed_up_area),
        removed_upward_discharge=tuple(removed_up_discharge),
        removed_downward_area=tuple(removed_down_area),
        removed_downward_discharge=tuple(removed_down_discharge),
        received_upward_area=tuple(received_up_area),
        received_upward_discharge=tuple(received_up_discharge),
        received_downward_area=tuple(received_down_area),
        received_downward_discharge=tuple(received_down_discharge),
        upward_overflow_liquid_volume=upward_overflow_volume,
        upward_overflow_kinematic_momentum=upward_overflow_momentum,
        downward_overflow_liquid_volume=downward_overflow_volume,
        downward_overflow_kinematic_momentum=downward_overflow_momentum,
        overflow_liquid_volume=overflow_volume,
        overflow_kinematic_momentum=overflow_momentum,
        requested_core_volume=dz * math.fsum(requested),
        opened_core_volume=dz * displaced_area,
        source_shortfall_volume=dz * math.fsum(shortfall),
        deposited_liquid_volume=dz
        * (math.fsum(received_up_area) + math.fsum(received_down_area)),
        initial_liquid_volume=initial_volume,
        final_liquid_volume=final_volume,
        initial_axial_kinematic_momentum=initial_momentum,
        final_axial_kinematic_momentum=final_momentum,
        volume_residual_including_overflow=volume_residual,
        momentum_residual_including_overflow=momentum_residual,
    )


def advance_taylor_sweep_geometry(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    previous_swept_fraction: Iterable[float],
    new_swept_fraction: Iterable[float],
    taylor_core_area_fraction: float,
    taylor_rise_velocity: float,
) -> TaylorSweepGeometryResult:
    """Apply the film and gas-core parts of one material-front increment."""

    old = tuple(float(value) for value in previous_swept_fraction)
    new = tuple(float(value) for value in new_swept_fraction)
    film = extend_taylor_sweep_in_persistent_state(
        state,
        parameters,
        previous_swept_fraction=old,
        new_swept_fraction=new,
        taylor_core_area_fraction=taylor_core_area_fraction,
        taylor_rise_velocity=taylor_rise_velocity,
    )
    displacement = displace_taylor_core_liquid_into_unswept_cells(
        film.state,
        parameters,
        previous_swept_fraction=old,
        new_swept_fraction=new,
        taylor_core_area_fraction=taylor_core_area_fraction,
    )
    return TaylorSweepGeometryResult(
        film_extension=film,
        gas_core_displacement=displacement,
    )


def _finite_tuple(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


@dataclass(frozen=True)
class IsothermalGasClosureParameters:
    """Thermodynamic data for the vertical gas adapter."""

    gas_constant: float = 287.05
    temperature: float = 293.0
    atmospheric_pressure: float = 101325.0
    gas_viscosity: float = 1.81e-5
    void_area_tolerance: float = 1.0e-14
    gas_inventory_tolerance: float = 1.0e-14
    pressure_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        values = (
            self.gas_constant,
            self.temperature,
            self.atmospheric_pressure,
            self.gas_viscosity,
            self.void_area_tolerance,
            self.gas_inventory_tolerance,
            self.pressure_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gas closure parameters must be finite")
        if min(values[:4]) <= 0.0 or min(values[4:]) < 0.0:
            raise ValueError("thermodynamic data must be positive and tolerances non-negative")

    @property
    def sound_speed_squared(self) -> float:
        return self.gas_constant * self.temperature


@dataclass(frozen=True)
class CoaxialCoreFilmGeometry:
    """Cell geometry for a liquid core, gas annulus, and wall liquid film.

    From the axis outwards, the idealized topology is upward liquid, gas, and
    downward liquid film.  The topology is used only when the corresponding
    areas are present; a missing phase has zero interface perimeter.
    """

    gas_area: tuple[float, ...]
    upward_core_radius: tuple[float, ...]
    film_inner_radius: tuple[float, ...]
    upward_gas_interface_perimeter: tuple[float, ...]
    downward_gas_interface_perimeter: tuple[float, ...]
    gas_hydraulic_diameter: tuple[float, ...]
    upward_liquid_hydraulic_diameter: tuple[float, ...]
    downward_film_hydraulic_diameter: tuple[float, ...]
    upward_wall_perimeter: tuple[float, ...]
    downward_wall_perimeter: tuple[float, ...]


def coaxial_core_film_geometry(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
) -> CoaxialCoreFilmGeometry:
    """Derive all interface lengths from the instantaneous conserved areas.

    For pipe radius ``R``, upward-core radius ``r_c`` and the inner film
    radius ``r_f`` are

    ``r_c=sqrt(A_up/pi)`` and ``r_f=sqrt((A_pipe-A_down)/pi)``.

    Hence the gas annulus is exactly ``pi*(r_f**2-r_c**2)``.  The gas
    hydraulic diameter uses both gas--liquid interfaces, ``4 A_g/(P_c+P_f)``.
    This is a geometry closure, not an adjustable drag multiplier.
    """

    if state.cell_count != parameters.cell_count:
        raise StateAdmissibilityError("state and geometry cell counts differ")
    full_area = parameters.full_area
    radius = 0.5 * parameters.diameter
    pipe_perimeter = 2.0 * math.pi * radius
    tol = max(parameters.dry_area_tolerance, 0.0)

    gas_area: list[float] = []
    core_radius: list[float] = []
    film_radius: list[float] = []
    p_up: list[float] = []
    p_down: list[float] = []
    dh_gas: list[float] = []
    dh_up: list[float] = []
    dh_down: list[float] = []
    wall_up: list[float] = []
    wall_down: list[float] = []

    for cell, (area_up, area_down) in enumerate(
        zip(state.upward_area, state.downward_area)
    ):
        occupied = area_up + area_down
        if occupied > full_area + parameters.packing_tolerance:
            raise StateAdmissibilityError(f"liquid over-packs riser cell {cell}")
        area_gas = max(full_area - occupied, 0.0)
        r_core = math.sqrt(max(area_up, 0.0) / math.pi)
        r_film = math.sqrt(max(full_area - area_down, 0.0) / math.pi)
        if r_core > r_film:
            # The area-space admissibility test above is authoritative.  The
            # nonlinear square-root map can turn an allowed packing-tolerance
            # excess into a radius difference slightly larger than a
            # first-order ``dA/(2*pi*r)`` estimate.  Since any larger physical
            # over-pack has already raised, this remaining case is a closed
            # gas annulus at solver tolerance: use one common liquid--liquid
            # contact radius without clipping either conserved liquid area.
            r_film = r_core

        interface_up = (
            2.0 * math.pi * r_core
            if area_up > tol and area_gas > tol
            else 0.0
        )
        interface_down = (
            2.0 * math.pi * r_film
            if area_down > tol and area_gas > tol
            else 0.0
        )
        gas_wetted = interface_up + interface_down
        gas_diameter = 4.0 * area_gas / gas_wetted if gas_wetted > 0.0 else 0.0
        # A core is bounded by its gas interface.  The annular film is bounded
        # by both the wall and its gas interface.
        up_diameter = 4.0 * area_up / interface_up if interface_up > 0.0 else 0.0
        down_wetted = pipe_perimeter + interface_down if area_down > tol else 0.0
        down_diameter = 4.0 * area_down / down_wetted if down_wetted > 0.0 else 0.0

        gas_area.append(area_gas)
        core_radius.append(r_core)
        film_radius.append(r_film)
        p_up.append(interface_up)
        p_down.append(interface_down)
        dh_gas.append(gas_diameter)
        dh_up.append(up_diameter)
        dh_down.append(down_diameter)
        wall_up.append(0.0)
        wall_down.append(pipe_perimeter if area_down > tol else 0.0)

    return CoaxialCoreFilmGeometry(
        gas_area=tuple(gas_area),
        upward_core_radius=tuple(core_radius),
        film_inner_radius=tuple(film_radius),
        upward_gas_interface_perimeter=tuple(p_up),
        downward_gas_interface_perimeter=tuple(p_down),
        gas_hydraulic_diameter=tuple(dh_gas),
        upward_liquid_hydraulic_diameter=tuple(dh_up),
        downward_film_hydraulic_diameter=tuple(dh_down),
        upward_wall_perimeter=tuple(wall_up),
        downward_wall_perimeter=tuple(wall_down),
    )


@dataclass(frozen=True)
class GasVoidPressureAdapter:
    """EOS pressure and drag geometry evaluated on one liquid stage."""

    geometry: CoaxialCoreFilmGeometry
    gas_density: tuple[float, ...]
    gas_velocity: tuple[float, ...]
    gas_pressure_cells: tuple[float, ...]
    common_pressure_cells: tuple[float, ...]
    common_pressure_faces: tuple[float, ...]
    active_gas: tuple[bool, ...]
    physical_drag_state: PhysicalGasInterphaseState


def adapt_gas_void_and_pressure_faces(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    gas_mass: Iterable[float],
    gas_momentum: Iterable[float],
    gas: IsothermalGasClosureParameters = IsothermalGasClosureParameters(),
    bottom_pressure: float | None = None,
    liquid_filled_cell_pressure: Iterable[float] | None = None,
) -> GasVoidPressureAdapter:
    """Map gas inventories to void, EOS pressure faces, and drag geometry.

    ``gas_mass`` and ``gas_momentum`` are cell inventories (kg and kg m/s).
    A finite gas void must contain finite positive mass; the adapter never
    creates atmospheric background mass.  A liquid-filled cell has no gas EOS
    pressure, so its common pressure must be supplied by the caller's liquid
    pressure solve through ``liquid_filled_cell_pressure``.  Post-breakthrough
    cells with finite gas void need no such auxiliary value.

    Interior faces use centred reconstruction.  The upper face is exactly
    atmospheric and the lower face is either supplied by the finite T node or
    linearly extrapolated from the first two cell pressures.  A linear
    hydrostatic pressure field is therefore reconstructed exactly.
    """

    geometry = coaxial_core_film_geometry(state, parameters)
    mass = _finite_tuple(gas_mass, name="gas_mass")
    momentum = _finite_tuple(gas_momentum, name="gas_momentum")
    n = parameters.cell_count
    if len(mass) != n or len(momentum) != n:
        raise ValueError("gas inventories and liquid state cell counts differ")
    if min(mass) < 0.0:
        raise GasVoidStateError("gas mass cannot be negative")
    if liquid_filled_cell_pressure is None:
        filled_pressure = (math.nan,) * n
    else:
        filled_pressure = _finite_tuple(
            liquid_filled_cell_pressure,
            name="liquid_filled_cell_pressure",
        )
        if len(filled_pressure) != n:
            raise ValueError("liquid-filled pressure needs one value per cell")

    density: list[float] = []
    velocity: list[float] = []
    eos_pressure: list[float] = []
    common_pressure: list[float] = []
    active: list[bool] = []
    for cell, (area_gas, cell_mass, cell_momentum) in enumerate(
        zip(geometry.gas_area, mass, momentum)
    ):
        has_void = area_gas > gas.void_area_tolerance
        if has_void:
            if cell_mass <= 0.0:
                raise GasVoidStateError(
                    f"finite gas void has no gas mass in cell {cell}"
                )
            cell_density = cell_mass / (area_gas * parameters.cell_length)
            cell_velocity = cell_momentum / cell_mass
            cell_pressure = cell_density * gas.sound_speed_squared
            if not math.isfinite(cell_pressure) or cell_pressure <= 0.0:
                raise GasVoidStateError(f"invalid EOS pressure in cell {cell}")
            common = cell_pressure
        else:
            if (
                cell_mass > gas.gas_inventory_tolerance
                or abs(cell_momentum) > gas.gas_inventory_tolerance
            ):
                raise GasVoidStateError(
                    f"gas inventory occupies a zero-void cell {cell}"
                )
            if not math.isfinite(filled_pressure[cell]) or filled_pressure[cell] <= 0.0:
                raise GasVoidStateError(
                    "liquid-filled cells require pressure from the liquid pressure solve"
                )
            cell_density = 0.0
            cell_velocity = 0.0
            cell_pressure = math.nan
            common = filled_pressure[cell]
        density.append(cell_density)
        velocity.append(cell_velocity)
        eos_pressure.append(cell_pressure)
        common_pressure.append(common)
        active.append(has_void)

    pressure_faces = [0.0] * (n + 1)
    for face in range(1, n):
        pressure_faces[face] = 0.5 * (
            common_pressure[face - 1] + common_pressure[face]
        )
    pressure_faces[n] = gas.atmospheric_pressure
    if bottom_pressure is not None:
        bottom = float(bottom_pressure)
        if not math.isfinite(bottom) or bottom <= 0.0:
            raise ValueError("bottom_pressure must be finite and positive")
        pressure_faces[0] = bottom
    elif n == 1:
        pressure_faces[0] = 2.0 * common_pressure[0] - pressure_faces[1]
    else:
        pressure_faces[0] = 1.5 * common_pressure[0] - 0.5 * common_pressure[1]
    if min(pressure_faces) <= 0.0 or not all(
        math.isfinite(value) for value in pressure_faces
    ):
        raise GasVoidStateError("reconstructed common pressure face is non-positive")

    # Preserve a stationary liquid column exactly.  If the reconstructed
    # profile already agrees with the discrete hydrostatic line to pressure
    # roundoff, use a single recurrence for all faces.  This is the standard
    # well-balanced equilibrium projection; it cannot alter a resolved
    # transient pressure gradient because the comparison tolerance is in Pa.
    if bottom_pressure is not None:
        hydrostatic_faces = tuple(
            float(bottom_pressure)
            - face
            * parameters.liquid_density
            * parameters.gravity
            * parameters.cell_length
            for face in range(n + 1)
        )
        mismatch = max(
            abs(candidate - equilibrium)
            for candidate, equilibrium in zip(pressure_faces, hydrostatic_faces)
        )
        if mismatch <= gas.pressure_tolerance:
            pressure_faces[:] = hydrostatic_faces

    drag_state = PhysicalGasInterphaseState.from_iterables(
        gas_mass=mass,
        gas_momentum=momentum,
        gas_area=geometry.gas_area,
        upward_interface_perimeter=geometry.upward_gas_interface_perimeter,
        downward_interface_perimeter=geometry.downward_gas_interface_perimeter,
        upward_hydraulic_diameter=geometry.gas_hydraulic_diameter,
        downward_hydraulic_diameter=geometry.gas_hydraulic_diameter,
        gas_viscosity=gas.gas_viscosity,
    )
    return GasVoidPressureAdapter(
        geometry=geometry,
        gas_density=tuple(density),
        gas_velocity=tuple(velocity),
        gas_pressure_cells=tuple(eos_pressure),
        common_pressure_cells=tuple(common_pressure),
        common_pressure_faces=tuple(pressure_faces),
        active_gas=tuple(active),
        physical_drag_state=drag_state,
    )


@dataclass(frozen=True)
class AtmosphericTopLiquidBoundary:
    """One-way liquid boundary at the open riser rim."""

    flux: DirectionalBoundaryFlux
    atmospheric_pressure: float
    connected_upward_area: float
    rejected_liquid_inflow: float = 0.0


def atmospheric_top_liquid_outflow(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    atmospheric_pressure: float = 101325.0,
) -> AtmosphericTopLiquidBoundary:
    """Return a donor outflow at the rim and prohibit liquid inflow.

    The top upward stream uses the interior donor state.  No liquid reservoir
    exists above the apparatus, so the downward boundary rate is identically
    zero.  Pressure at the same face is atmospheric and is supplied separately
    to the common-pressure adapter.
    """

    pressure = float(atmospheric_pressure)
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("atmospheric pressure must be finite and positive")
    if state.cell_count != parameters.cell_count:
        raise StateAdmissibilityError("state and parameter cell counts differ")
    area = state.upward_area[-1]
    rate = state.upward_discharge[-1]
    if area <= parameters.dry_area_tolerance or rate <= 0.0:
        rate = 0.0
        speed = 0.0
    else:
        speed = rate / area
    return AtmosphericTopLiquidBoundary(
        flux=DirectionalBoundaryFlux(
            upward_rate=rate,
            upward_speed=speed,
            downward_rate=0.0,
            downward_speed=0.0,
        ),
        atmospheric_pressure=pressure,
        connected_upward_area=area,
    )


@dataclass(frozen=True)
class CoreFilmInventoryEvolution:
    """Area ledger for a post-event prognostic core/film stage."""

    initial_upward_volume: float
    final_upward_volume: float
    initial_downward_volume: float
    final_downward_volume: float
    upward_boundary_volume_change: float
    downward_boundary_volume_change: float
    upward_topology_transfer: float
    downward_topology_transfer: float
    upward_volume_residual: float
    downward_volume_residual: float
    total_volume_residual: float


@dataclass(frozen=True)
class PostEventCoreFilmStep:
    """Result returned to a global Case-A stage."""

    state: VerticalTwoStreamState
    gas_momentum: tuple[float, ...]
    pressure_before: GasVoidPressureAdapter
    pressure_after_transport: GasVoidPressureAdapter
    top_boundary: AtmosphericTopLiquidBoundary
    transport: VerticalTwoStreamStepResult
    drag: PhysicalThreeBodyDragResult | None
    inventory: CoreFilmInventoryEvolution


def advance_post_event_core_film_stage(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    dt: float,
    gas_mass: Iterable[float],
    gas_momentum: Iterable[float],
    bottom_boundary: DirectionalBoundaryFlux = DirectionalBoundaryFlux(),
    gas: IsothermalGasClosureParameters = IsothermalGasClosureParameters(),
    bottom_pressure: float | None = None,
    liquid_filled_cell_pressure: Iterable[float] | None = None,
    apply_physical_drag: bool = True,
) -> PostEventCoreFilmStep:
    """Advance one target-free post-breakthrough core/film closure stage.

    The caller supplies the finite-T-node gross bottom rates.  This function
    owns the riser liquid update exactly once.  It recomputes void, pressure,
    geometry, and the atmospheric top boundary from the current stage state;
    transports both liquid inventories; then recomputes geometry before the
    optional implicit gas/up/down drag solve.

    Gas *mass transport* remains owned by the coupled gas network.  The gas
    momentum returned here includes only the equal-and-opposite drag reaction
    and must replace, not augment, the pre-drag vertical gas momentum.
    """

    mass = _finite_tuple(gas_mass, name="gas_mass")
    momentum = _finite_tuple(gas_momentum, name="gas_momentum")
    if len(mass) != parameters.cell_count or len(momentum) != parameters.cell_count:
        raise ValueError("gas inventories and riser state cell counts differ")
    pressure_before = adapt_gas_void_and_pressure_faces(
        state,
        parameters,
        gas_mass=mass,
        gas_momentum=momentum,
        gas=gas,
        bottom_pressure=bottom_pressure,
        liquid_filled_cell_pressure=liquid_filled_cell_pressure,
    )
    top = atmospheric_top_liquid_outflow(
        state,
        parameters,
        atmospheric_pressure=gas.atmospheric_pressure,
    )
    transport = advance_vertical_two_stream_fv(
        state,
        parameters,
        dt=dt,
        pressure_faces=pressure_before.common_pressure_faces,
        boundaries=VerticalTwoStreamBoundaries(
            bottom=bottom_boundary,
            top=top.flux,
        ),
    )
    pressure_after = adapt_gas_void_and_pressure_faces(
        transport.state,
        parameters,
        gas_mass=mass,
        gas_momentum=momentum,
        gas=gas,
        bottom_pressure=bottom_pressure,
        liquid_filled_cell_pressure=liquid_filled_cell_pressure,
    )
    if apply_physical_drag:
        drag = implicit_physical_three_body_drag_exchange(
            transport.state,
            parameters,
            pressure_after.physical_drag_state,
            dt=dt,
        )
        final_state = drag.state
        final_gas_momentum = drag.gas_momentum
    else:
        drag = None
        final_state = transport.state
        final_gas_momentum = momentum

    dz = parameters.cell_length
    initial_up = dz * sum(state.upward_area)
    initial_down = dz * sum(state.downward_area)
    final_up = dz * sum(final_state.upward_area)
    final_down = dz * sum(final_state.downward_area)
    up_boundary = transport.ledger.upward_boundary_volume_change
    down_boundary = transport.ledger.downward_boundary_volume_change
    drag_up_topology = final_up - dz * sum(transport.state.upward_area)
    drag_down_topology = final_down - dz * sum(transport.state.downward_area)
    up_topology = (
        transport.ledger.upward_topology_volume_transfer + drag_up_topology
    )
    down_topology = (
        transport.ledger.downward_topology_volume_transfer + drag_down_topology
    )
    up_residual = final_up - initial_up - up_boundary - up_topology
    down_residual = final_down - initial_down - down_boundary - down_topology
    total_residual = (
        (final_up + final_down)
        - (initial_up + initial_down)
        - up_boundary
        - down_boundary
    )
    tolerance = 512.0 * math.ulp(
        max(initial_up + initial_down, abs(up_boundary) + abs(down_boundary), 1.0)
    )
    if abs(total_residual) > tolerance:
        raise FloatingPointError("post-event core/film stage lost liquid volume")
    if abs(up_topology + down_topology) > tolerance:
        raise FloatingPointError("post-drag topology transfer lost liquid volume")
    if max(abs(up_residual), abs(down_residual)) > tolerance:
        raise FloatingPointError("post-event directional inventory ledger does not close")
    inventory = CoreFilmInventoryEvolution(
        initial_upward_volume=initial_up,
        final_upward_volume=final_up,
        initial_downward_volume=initial_down,
        final_downward_volume=final_down,
        upward_boundary_volume_change=up_boundary,
        downward_boundary_volume_change=down_boundary,
        upward_topology_transfer=up_topology,
        downward_topology_transfer=down_topology,
        upward_volume_residual=up_residual,
        downward_volume_residual=down_residual,
        total_volume_residual=total_residual,
    )
    return PostEventCoreFilmStep(
        state=final_state,
        gas_momentum=tuple(final_gas_momentum),
        pressure_before=pressure_before,
        pressure_after_transport=pressure_after,
        top_boundary=top,
        transport=transport,
        drag=drag,
        inventory=inventory,
    )


__all__ = [
    "AtmosphericTopLiquidBoundary",
    "CoaxialCoreFilmGeometry",
    "CoreFilmInventoryEvolution",
    "GasVoidPressureAdapter",
    "GasVoidStateError",
    "IsothermalGasClosureParameters",
    "POST_EVENT_TWOSTREAM_CLOSURES_READY",
    "PostEventCoreFilmStep",
    "TaylorGasCoreDisplacementResult",
    "TaylorSweepExtensionResult",
    "TaylorSweepGeometryResult",
    "VerticalTwoStreamClosureError",
    "adapt_gas_void_and_pressure_faces",
    "advance_post_event_core_film_stage",
    "advance_taylor_sweep_geometry",
    "atmospheric_top_liquid_outflow",
    "coaxial_core_film_geometry",
    "displace_taylor_core_liquid_into_unswept_cells",
    "extend_taylor_sweep_in_persistent_state",
]
