"""Conservative finite-volume core for counter-current liquid in the Case-A riser.

The vertical coordinate ``z`` is positive upward.  Two liquid continua are
resolved in every cell:

``U = (A_up, Q_up, A_down, Q_down)``.

``Q_up`` is non-negative and ``Q_down`` is non-positive while this topology is
active.  Consequently the legacy one-liquid variables are recovered without
fitting or reconstruction,

``A_l = A_up + A_down`` and ``Q_l = Q_up + Q_down``.

The gross rates are ``Q_up`` and ``-Q_down`` and their difference is the net
rate.  Both streams satisfy finite-volume area and axial-momentum balances.
The source operator contains axial gravity, a supplied common-pressure
gradient, Darcy wall friction, liquid--liquid momentum exchange, and an
optional gas-drag reaction.  Liquid--liquid exchange is exactly equal and
opposite.  Optional liquid--gas exchange returns the opposite impulse that a
coupled gas solver must consume.

This is an intentionally isolated interior/boundary core.  It does not read a
clock, a requested liquid height, an OpenFOAM field, or a plotted result.
Simultaneous counter-current arrivals share a conservative receiving-capacity
projection, so a nearly full cell cannot accept both directional fluxes beyond
the physical cross-section.  The projection changes the single shared face
flux (and its matching momentum flux), never clips or redistributes a cell
inventory after the update.  A stopped or reversed directional stream is
handled by an explicit area-and-momentum-conservative topology transfer, not a
velocity clamp.  A production integration still needs constitutive closures
for the evolving stream-area split, pressure/void coupling, and the T-mouth
and top-vent Riemann problems.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from casea_capacity_pressure_projection import (
    project_capacity_pressure_active_set,
)


TWOSTREAM_FV_CORE_READY = True
COMPLETE_CASEA_RISER_READY = False
MISSING_PHYSICAL_CLOSURES = (
    "post_event_core_film_area_evolution",
    "calibrated_liquid_liquid_interfacial_drag",
    "gas_void_pressure_evolution_and_stage_coupling",
    "finite_tjunction_two_stream_riemann_problem",
    "top_vent_and_free_surface_boundary",
)


class VerticalTwoStreamError(RuntimeError):
    """Base class for rejected two-stream operations."""


class StateAdmissibilityError(VerticalTwoStreamError):
    """The supplied state is non-finite or violates geometry/topology."""


class PackingViolationError(VerticalTwoStreamError):
    """A finite-volume stage tries to exceed the riser cross-section."""


class DirectionalTopologyError(VerticalTwoStreamError):
    """A stream reverses and therefore requires an explicit topology event."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _tuple(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one cell")
    if not _finite(*result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class VerticalTwoStreamParameters:
    """Uniform-grid geometry and local constitutive coefficients.

    ``wall_friction_*`` are Darcy friction factors.  The two hydraulic
    diameters default to the full riser diameter; replacing them by a
    film/core geometry is part of the missing area-partition closure.

    ``interstream_drag`` has units ``1/m`` in

    ``S_up = -K A_mix (u_up-u_down)|u_up-u_down|``.

    It is a physical input, not a result-dependent control.
    """

    cell_count: int
    cell_length: float
    diameter: float
    liquid_density: float = 998.0
    gravity: float = 9.81
    wall_friction_up: float = 0.0
    wall_friction_down: float = 0.0
    interstream_drag: float = 0.0
    hydraulic_diameter_up: float | None = None
    hydraulic_diameter_down: float | None = None
    dry_area_tolerance: float = 1.0e-14
    packing_tolerance: float = 2.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.cell_count, int) or self.cell_count <= 0:
            raise ValueError("cell_count must be a positive integer")
        values = (
            self.cell_length,
            self.diameter,
            self.liquid_density,
            self.gravity,
            self.wall_friction_up,
            self.wall_friction_down,
            self.interstream_drag,
            self.dry_area_tolerance,
            self.packing_tolerance,
        )
        if not _finite(*values):
            raise ValueError("two-stream parameters must be finite")
        if min(self.cell_length, self.diameter, self.liquid_density) <= 0.0:
            raise ValueError("grid, diameter, and density must be positive")
        if min(
            self.gravity,
            self.wall_friction_up,
            self.wall_friction_down,
            self.interstream_drag,
            self.dry_area_tolerance,
            self.packing_tolerance,
        ) < 0.0:
            raise ValueError("gravity, friction, drag, and tolerances cannot be negative")
        for name, value in (
            ("hydraulic_diameter_up", self.hydraulic_diameter_up),
            ("hydraulic_diameter_down", self.hydraulic_diameter_down),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def full_area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def up_diameter(self) -> float:
        return self.diameter if self.hydraulic_diameter_up is None else self.hydraulic_diameter_up

    @property
    def down_diameter(self) -> float:
        return (
            self.diameter
            if self.hydraulic_diameter_down is None
            else self.hydraulic_diameter_down
        )


@dataclass(frozen=True)
class VerticalTwoStreamState:
    """Cell-centred conserved state.

    Areas have units m2 and discharges have units m3/s.  The tuples are
    immutable so a rejected step cannot partly mutate the caller's state.
    """

    upward_area: tuple[float, ...]
    upward_discharge: tuple[float, ...]
    downward_area: tuple[float, ...]
    downward_discharge: tuple[float, ...]

    def __post_init__(self) -> None:
        arrays = (
            self.upward_area,
            self.upward_discharge,
            self.downward_area,
            self.downward_discharge,
        )
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise StateAdmissibilityError(
                "all two-stream state arrays need one common nonzero length"
            )
        if not _finite(*(value for values in arrays for value in values)):
            raise StateAdmissibilityError("two-stream state must be finite")
        for index, (area, discharge) in enumerate(
            zip(self.upward_area, self.upward_discharge)
        ):
            if area < 0.0:
                raise StateAdmissibilityError(f"negative upward area in cell {index}")
            if discharge < 0.0:
                raise DirectionalTopologyError(f"upward stream reversed in cell {index}")
            if area == 0.0 and discharge != 0.0:
                raise StateAdmissibilityError(
                    f"dry upward stream carries momentum in cell {index}"
                )
        for index, (area, discharge) in enumerate(
            zip(self.downward_area, self.downward_discharge)
        ):
            if area < 0.0:
                raise StateAdmissibilityError(f"negative downward area in cell {index}")
            if discharge > 0.0:
                raise DirectionalTopologyError(f"downward stream reversed in cell {index}")
            if area == 0.0 and discharge != 0.0:
                raise StateAdmissibilityError(
                    f"dry downward stream carries momentum in cell {index}"
                )

    @classmethod
    def from_iterables(
        cls,
        *,
        upward_area: Iterable[float],
        upward_discharge: Iterable[float],
        downward_area: Iterable[float],
        downward_discharge: Iterable[float],
    ) -> "VerticalTwoStreamState":
        return cls(
            upward_area=_tuple(upward_area, name="upward_area"),
            upward_discharge=_tuple(upward_discharge, name="upward_discharge"),
            downward_area=_tuple(downward_area, name="downward_area"),
            downward_discharge=_tuple(downward_discharge, name="downward_discharge"),
        )

    @classmethod
    def from_total_and_film(
        cls,
        *,
        liquid_area: Iterable[float],
        liquid_discharge: Iterable[float],
        falling_film_area: Iterable[float],
        falling_film_discharge: Iterable[float],
        tolerance: float = 1.0e-14,
    ) -> "VerticalTwoStreamState":
        """Map ``(A_l,Q_l,A_f,Q_f)`` to explicit core/film states.

        No area split is inferred: the caller supplies the falling-film state.
        """

        al = _tuple(liquid_area, name="liquid_area")
        ql = _tuple(liquid_discharge, name="liquid_discharge")
        af = _tuple(falling_film_area, name="falling_film_area")
        qf = _tuple(falling_film_discharge, name="falling_film_discharge")
        if len({len(al), len(ql), len(af), len(qf)}) != 1:
            raise StateAdmissibilityError("total and film arrays need one common length")
        up_area = []
        up_q = []
        for index, (a_total, q_total, a_film, q_film) in enumerate(zip(al, ql, af, qf)):
            a_core = a_total - a_film
            q_core = q_total - q_film
            if a_core < -tolerance:
                raise StateAdmissibilityError(f"film area exceeds total area in cell {index}")
            up_area.append(max(a_core, 0.0))
            up_q.append(0.0 if abs(q_core) <= tolerance else q_core)
        return cls.from_iterables(
            upward_area=up_area,
            upward_discharge=up_q,
            downward_area=af,
            downward_discharge=qf,
        )

    @classmethod
    def from_legacy_single_stream(
        cls,
        liquid_area: Iterable[float],
        liquid_discharge: Iterable[float],
    ) -> "VerticalTwoStreamState":
        """Embed a one-liquid state as a degenerate two-stream state."""

        area = _tuple(liquid_area, name="liquid_area")
        discharge = _tuple(liquid_discharge, name="liquid_discharge")
        if len(area) != len(discharge):
            raise StateAdmissibilityError("legacy area and discharge lengths differ")
        return cls.from_iterables(
            upward_area=(a if q >= 0.0 else 0.0 for a, q in zip(area, discharge)),
            upward_discharge=(max(q, 0.0) for q in discharge),
            downward_area=(a if q < 0.0 else 0.0 for a, q in zip(area, discharge)),
            downward_discharge=(min(q, 0.0) for q in discharge),
        )

    @property
    def cell_count(self) -> int:
        return len(self.upward_area)

    @property
    def liquid_area(self) -> tuple[float, ...]:
        """The exact array to map to the existing ``Alr`` field."""

        return tuple(a_up + a_down for a_up, a_down in zip(self.upward_area, self.downward_area))

    @property
    def liquid_discharge(self) -> tuple[float, ...]:
        return tuple(
            q_up + q_down
            for q_up, q_down in zip(
                self.upward_discharge, self.downward_discharge
            )
        )

    @property
    def gross_upward_flow(self) -> tuple[float, ...]:
        return self.upward_discharge

    @property
    def gross_downward_flow(self) -> tuple[float, ...]:
        return tuple(-q for q in self.downward_discharge)


@dataclass(frozen=True)
class DirectionalBoundaryFlux:
    """Gross two-stream flux at one boundary face.

    Both rates and speeds are non-negative magnitudes.  The coordinate fluxes
    are ``+upward_rate`` and ``-downward_rate``; hence ``net_rate`` is their
    difference.  At the bottom, upward liquid enters and downward liquid
    leaves.  At the top, the donor roles are reversed.
    """

    upward_rate: float = 0.0
    upward_speed: float = 0.0
    downward_rate: float = 0.0
    downward_speed: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.upward_rate,
            self.upward_speed,
            self.downward_rate,
            self.downward_speed,
        )
        if not _finite(*values) or min(values) < 0.0:
            raise ValueError(
                "directional boundary rates and speeds must be finite and non-negative"
            )
        if self.upward_rate > 0.0 and self.upward_speed <= 0.0:
            raise ValueError("positive upward boundary rate requires a positive speed")
        if self.downward_rate > 0.0 and self.downward_speed <= 0.0:
            raise ValueError("positive downward boundary rate requires a positive speed")

    @property
    def net_rate(self) -> float:
        return self.upward_rate - self.downward_rate

    @property
    def signed_downward_rate(self) -> float:
        return -self.downward_rate

    @property
    def upward_momentum_flux(self) -> float:
        return self.upward_rate * self.upward_speed

    @property
    def downward_momentum_flux(self) -> float:
        # (-Q_down)(-u_down) is positive in the axial momentum flux.
        return self.downward_rate * self.downward_speed


@dataclass(frozen=True)
class VerticalTwoStreamBoundaries:
    bottom: DirectionalBoundaryFlux = DirectionalBoundaryFlux()
    top: DirectionalBoundaryFlux = DirectionalBoundaryFlux()


@dataclass(frozen=True)
class GasMomentumCoupling:
    """Frozen gas trace used for an operator-split liquid--gas drag stage.

    The returned gas reaction must be applied to the coupled gas momentum.
    ``drag_coefficient`` has units 1/m.  The frozen-trace approximation is
    first-order; a fully coupled implicit exchange remains an integration task.
    """

    gas_area: tuple[float, ...]
    gas_velocity: tuple[float, ...]
    drag_coefficient: float

    @classmethod
    def from_iterables(
        cls,
        *,
        gas_area: Iterable[float],
        gas_velocity: Iterable[float],
        drag_coefficient: float,
    ) -> "GasMomentumCoupling":
        return cls(
            gas_area=_tuple(gas_area, name="gas_area"),
            gas_velocity=_tuple(gas_velocity, name="gas_velocity"),
            drag_coefficient=float(drag_coefficient),
        )

    def __post_init__(self) -> None:
        if len(self.gas_area) != len(self.gas_velocity) or not self.gas_area:
            raise ValueError("gas coupling arrays need one common nonzero length")
        if not _finite(*self.gas_area, *self.gas_velocity, self.drag_coefficient):
            raise ValueError("gas coupling data must be finite")
        if min(self.gas_area) < 0.0 or self.drag_coefficient < 0.0:
            raise ValueError("gas area and drag coefficient cannot be negative")


@dataclass(frozen=True)
class PhysicalGasInterphaseState:
    """Resolved gas state and measured gas--liquid interface geometry.

    ``gas_mass`` and ``gas_momentum`` are cell inventories in kg and kg m/s.
    The two interface perimeters and hydraulic diameters determine the
    gas-side friction factor in the same manner as
    ``casea_coupled_gas_network._implicit_interphase_drag_exchange``.  There
    is deliberately no free ``1/m`` drag multiplier.
    """

    gas_mass: tuple[float, ...]
    gas_momentum: tuple[float, ...]
    gas_area: tuple[float, ...]
    upward_interface_perimeter: tuple[float, ...]
    downward_interface_perimeter: tuple[float, ...]
    upward_hydraulic_diameter: tuple[float, ...]
    downward_hydraulic_diameter: tuple[float, ...]
    gas_viscosity: float = 1.81e-5

    @classmethod
    def from_iterables(
        cls,
        *,
        gas_mass: Iterable[float],
        gas_momentum: Iterable[float],
        gas_area: Iterable[float],
        upward_interface_perimeter: Iterable[float],
        downward_interface_perimeter: Iterable[float],
        upward_hydraulic_diameter: Iterable[float],
        downward_hydraulic_diameter: Iterable[float],
        gas_viscosity: float = 1.81e-5,
    ) -> "PhysicalGasInterphaseState":
        return cls(
            gas_mass=_tuple(gas_mass, name="gas_mass"),
            gas_momentum=_tuple(gas_momentum, name="gas_momentum"),
            gas_area=_tuple(gas_area, name="gas_area"),
            upward_interface_perimeter=_tuple(
                upward_interface_perimeter,
                name="upward_interface_perimeter",
            ),
            downward_interface_perimeter=_tuple(
                downward_interface_perimeter,
                name="downward_interface_perimeter",
            ),
            upward_hydraulic_diameter=_tuple(
                upward_hydraulic_diameter,
                name="upward_hydraulic_diameter",
            ),
            downward_hydraulic_diameter=_tuple(
                downward_hydraulic_diameter,
                name="downward_hydraulic_diameter",
            ),
            gas_viscosity=float(gas_viscosity),
        )

    def __post_init__(self) -> None:
        arrays = (
            self.gas_mass,
            self.gas_momentum,
            self.gas_area,
            self.upward_interface_perimeter,
            self.downward_interface_perimeter,
            self.upward_hydraulic_diameter,
            self.downward_hydraulic_diameter,
        )
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise ValueError("physical gas-drag arrays need one common nonzero length")
        if not _finite(
            *(value for values in arrays for value in values),
            self.gas_viscosity,
        ):
            raise ValueError("physical gas-drag data must be finite")
        if self.gas_viscosity <= 0.0:
            raise ValueError("gas viscosity must be positive")
        for index, (mass, momentum, area) in enumerate(
            zip(self.gas_mass, self.gas_momentum, self.gas_area)
        ):
            if mass < 0.0 or area < 0.0:
                raise ValueError(f"negative gas inventory in cell {index}")
            if mass == 0.0 and momentum != 0.0:
                raise ValueError(f"zero gas mass carries momentum in cell {index}")
            if mass > 0.0 and area <= 0.0:
                raise ValueError(f"positive gas mass needs positive area in cell {index}")
        for name, values in (
            ("upward_interface_perimeter", self.upward_interface_perimeter),
            ("downward_interface_perimeter", self.downward_interface_perimeter),
        ):
            if min(values) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name, values in (
            ("upward_hydraulic_diameter", self.upward_hydraulic_diameter),
            ("downward_hydraulic_diameter", self.downward_hydraulic_diameter),
        ):
            if min(values) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for index, (perimeter, diameter) in enumerate(
            zip(
                self.upward_interface_perimeter,
                self.upward_hydraulic_diameter,
            )
        ):
            if perimeter > 0.0 and diameter <= 0.0:
                raise ValueError(
                    f"upward interface needs positive hydraulic diameter in cell {index}"
                )
        for index, (perimeter, diameter) in enumerate(
            zip(
                self.downward_interface_perimeter,
                self.downward_hydraulic_diameter,
            )
        ):
            if perimeter > 0.0 and diameter <= 0.0:
                raise ValueError(
                    f"downward interface needs positive hydraulic diameter in cell {index}"
                )


@dataclass(frozen=True)
class DirectionalTopologyTransferResult:
    """Conservative local remap when a labelled stream stops or reverses."""

    state: VerticalTwoStreamState
    upward_area_transfer: tuple[float, ...]
    downward_area_transfer: tuple[float, ...]
    upward_momentum_transfer: tuple[float, ...]
    downward_momentum_transfer: tuple[float, ...]
    kinematic_energy_loss: float
    area_residual: float
    momentum_residual: float


@dataclass(frozen=True)
class PhysicalThreeBodyDragResult:
    """Implicit gas/upward-liquid/downward-liquid exchange result."""

    state: VerticalTwoStreamState
    gas_momentum: tuple[float, ...]
    gas_impulse: tuple[float, ...]
    upward_liquid_impulse: tuple[float, ...]
    downward_liquid_impulse: tuple[float, ...]
    upward_friction_factor: tuple[float, ...]
    downward_friction_factor: tuple[float, ...]
    cell_momentum_residual: tuple[float, ...]
    topology_transfer: DirectionalTopologyTransferResult

    @property
    def total_momentum_residual(self) -> float:
        return sum(self.cell_momentum_residual)


@dataclass(frozen=True)
class TaylorBreakthroughMappingResult:
    """Conservative map from the legacy riser state at Taylor breakthrough."""

    state: VerticalTwoStreamState
    geometric_film_area: tuple[float, ...]
    falling_film_velocity: float
    area_residual: tuple[float, ...]
    momentum_residual: tuple[float, ...]
    kinematic_energy_change: float


def _kinematic_energy(area: float, discharge: float) -> float:
    if area <= 0.0:
        return 0.0
    return 0.5 * discharge * discharge / area


def conservative_directional_topology_transfer(
    *,
    upward_area: Iterable[float],
    upward_discharge: Iterable[float],
    downward_area: Iterable[float],
    downward_discharge: Iterable[float],
    velocity_tolerance: float = 1.0e-12,
    area_tolerance: float = 0.0,
    preserve_stopped_partition: Iterable[bool] | None = None,
) -> DirectionalTopologyTransferResult:
    """Return a valid directional state without losing area or momentum.

    A stopped stream is mixed inelastically into the still-moving stream.  A
    stream that reverses is transferred to the matching directional channel.
    If both labels have crossed simultaneously, their complete states are
    swapped, which is conservative and non-dissipative.  If both streams are
    stationary, their area partition is retained exactly.  A caller may also
    retain a stopped branch in selected geometrically separated cells.  In
    those cells, a one-sided reversal transfers only the minimum donor area
    required to carry the reversed momentum without increasing kinetic energy;
    the remaining stationary corridor is not deleted by a sign crossing of
    arbitrarily small magnitude.

    This is a topology operation, not a velocity clamp.  The post-transfer
    kinetic energy cannot exceed the pre-transfer value.
    """

    au = _tuple(upward_area, name="upward_area")
    qu = _tuple(upward_discharge, name="upward_discharge")
    ad = _tuple(downward_area, name="downward_area")
    qd = _tuple(downward_discharge, name="downward_discharge")
    if len({len(au), len(qu), len(ad), len(qd)}) != 1:
        raise StateAdmissibilityError("topology-transfer arrays need equal lengths")
    if not math.isfinite(velocity_tolerance) or velocity_tolerance < 0.0:
        raise ValueError("velocity_tolerance must be finite and non-negative")
    if not math.isfinite(area_tolerance) or area_tolerance < 0.0:
        raise ValueError("area_tolerance must be finite and non-negative")
    preserve = (
        (False,) * len(au)
        if preserve_stopped_partition is None
        else tuple(bool(value) for value in preserve_stopped_partition)
    )
    if len(preserve) != len(au):
        raise ValueError("stopped-partition mask must contain one value per cell")

    final_au: list[float] = []
    final_qu: list[float] = []
    final_ad: list[float] = []
    final_qd: list[float] = []
    energy_before = 0.0
    energy_after = 0.0
    for index, (a_u, q_u, a_d, q_d) in enumerate(zip(au, qu, ad, qd)):
        if a_u < 0.0 or a_d < 0.0:
            raise StateAdmissibilityError(
                f"negative area supplied to topology transfer in cell {index}"
            )
        if a_u == 0.0 and q_u != 0.0:
            raise StateAdmissibilityError(
                f"dry upward channel carries momentum in cell {index}"
            )
        if a_d == 0.0 and q_d != 0.0:
            raise StateAdmissibilityError(
                f"dry downward channel carries momentum in cell {index}"
            )
        u_u = 0.0 if a_u == 0.0 else q_u / a_u
        u_d = 0.0 if a_d == 0.0 else q_d / a_d
        energy_before += _kinematic_energy(a_u, q_u)
        energy_before += _kinematic_energy(a_d, q_d)

        if q_u < 0.0 and q_d > 0.0:
            # Both physical streams remain resolved; only their labels crossed.
            new_au, new_qu = a_d, q_d
            new_ad, new_qd = a_u, q_u
        elif preserve[index] and q_d > 0.0:
            # Only the downward-labelled branch crossed.  Relabel the minimum
            # amount of its area that can carry the combined upward momentum
            # without increasing kinetic energy.  This is the entropy-stable
            # projection of the provisional state onto q_up>=0, q_down=0.
            # In particular, an O(eps) sign crossing cannot erase an O(1)
            # falling-film corridor.  If no upward receiver exists the whole
            # reversed branch transfers, so a dry channel never receives
            # finite momentum.
            total_discharge = q_u + q_d
            twice_energy = (
                (q_u * q_u / a_u if a_u > 0.0 else 0.0)
                + q_d * q_d / a_d
            )
            required_moving_area = (
                total_discharge * total_discharge / twice_energy
                if twice_energy > 0.0
                else 0.0
            )
            transfer_area = min(
                max(required_moving_area - a_u, 0.0),
                a_d,
            )
            if a_d - transfer_area <= max(
                128.0 * math.ulp(a_d), area_tolerance
            ):
                transfer_area = a_d
            new_au = a_u + transfer_area
            new_qu = total_discharge
            new_ad = a_d - transfer_area
            new_qd = 0.0
        elif preserve[index] and q_u < 0.0:
            # Symmetric projection for an upward-labelled branch that crossed
            # into the downward direction.
            total_discharge = q_u + q_d
            twice_energy = (
                q_u * q_u / a_u
                + (q_d * q_d / a_d if a_d > 0.0 else 0.0)
            )
            required_moving_area = (
                total_discharge * total_discharge / twice_energy
                if twice_energy > 0.0
                else 0.0
            )
            transfer_area = min(
                max(required_moving_area - a_d, 0.0),
                a_u,
            )
            if a_u - transfer_area <= max(
                128.0 * math.ulp(a_u), area_tolerance
            ):
                transfer_area = a_u
            new_au = a_u - transfer_area
            new_qu = 0.0
            new_ad = a_d + transfer_area
            new_qd = total_discharge
        else:
            up_invalid = q_u < 0.0
            down_invalid = q_d > 0.0
            up_stopped = (
                not preserve[index]
                and
                a_u > 0.0
                and abs(u_u) <= velocity_tolerance
                and u_d < -velocity_tolerance
            )
            down_stopped = (
                not preserve[index]
                and
                a_d > 0.0
                and abs(u_d) <= velocity_tolerance
                and u_u > velocity_tolerance
            )
            if up_invalid or down_invalid or up_stopped or down_stopped:
                total_area = a_u + a_d
                total_discharge = q_u + q_d
                if total_discharge > 0.0:
                    new_au, new_qu = total_area, total_discharge
                    new_ad, new_qd = 0.0, 0.0
                elif total_discharge < 0.0:
                    new_au, new_qu = 0.0, 0.0
                    new_ad, new_qd = total_area, total_discharge
                else:
                    # A zero-total-momentum collision becomes a stationary
                    # single continuum in the channel with the larger area.
                    if a_u >= a_d:
                        new_au, new_qu = total_area, 0.0
                        new_ad, new_qd = 0.0, 0.0
                    else:
                        new_au, new_qu = 0.0, 0.0
                        new_ad, new_qd = total_area, 0.0
            else:
                new_au, new_qu = a_u, q_u
                new_ad, new_qd = a_d, q_d

        final_au.append(new_au)
        final_qu.append(new_qu)
        final_ad.append(new_ad)
        final_qd.append(new_qd)
        energy_after += _kinematic_energy(new_au, new_qu)
        energy_after += _kinematic_energy(new_ad, new_qd)

    final_state = VerticalTwoStreamState.from_iterables(
        upward_area=final_au,
        upward_discharge=final_qu,
        downward_area=final_ad,
        downward_discharge=final_qd,
    )
    up_area_transfer = tuple(new - old for new, old in zip(final_au, au))
    down_area_transfer = tuple(new - old for new, old in zip(final_ad, ad))
    up_momentum_transfer = tuple(new - old for new, old in zip(final_qu, qu))
    down_momentum_transfer = tuple(new - old for new, old in zip(final_qd, qd))
    area_residual = sum(up_area_transfer) + sum(down_area_transfer)
    momentum_residual = sum(up_momentum_transfer) + sum(down_momentum_transfer)
    energy_loss = energy_before - energy_after
    if energy_loss < -1.0e-12 * max(energy_before, 1.0):
        raise StateAdmissibilityError("topology transfer created kinetic energy")
    return DirectionalTopologyTransferResult(
        state=final_state,
        upward_area_transfer=up_area_transfer,
        downward_area_transfer=down_area_transfer,
        upward_momentum_transfer=up_momentum_transfer,
        downward_momentum_transfer=down_momentum_transfer,
        kinematic_energy_loss=max(energy_loss, 0.0),
        area_residual=area_residual,
        momentum_residual=momentum_residual,
    )


def map_taylor_breakthrough_to_twostream(
    liquid_area: Iterable[float],
    liquid_discharge: Iterable[float],
    parameters: VerticalTwoStreamParameters,
    *,
    taylor_core_area_fraction: float,
    taylor_rise_velocity: float,
    swept_fraction: Iterable[float] | None = None,
) -> TaylorBreakthroughMappingResult:
    """Initialize film/core states at a Taylor-front breakthrough event.

    The falling-film corridor in cell ``i`` is

    ``swept_i (1-alpha_core) A_r``.

    The Davies--Taylor displacement balance supplies a diagnostic film speed,

    ``u_f = -alpha_core U_T/(1-alpha_core)``,

    but the transition does not impose that speed.  The cell's complete
    pre-existing liquid inventory is assigned to the stream matching its
    inherited discharge.  Resting liquid is assigned to the downward owner
    because it is initial riser water; the geometrical upward corridor is a
    boundary trace, not a second stationary cell inventory.  This preserves
    area and axial momentum cell by cell without manufacturing an upward parcel
    at breakthrough.  Subsequent boundary transport, gravity, wall stress and
    three-body drag determine both directional inventories prognostically.
    """

    area = _tuple(liquid_area, name="liquid_area")
    discharge = _tuple(liquid_discharge, name="liquid_discharge")
    if len(area) != len(discharge) or len(area) != parameters.cell_count:
        raise StateAdmissibilityError(
            "breakthrough state and parameter cell counts differ"
        )
    if not _finite(taylor_core_area_fraction, taylor_rise_velocity):
        raise ValueError("Taylor closure inputs must be finite")
    if not 0.0 < taylor_core_area_fraction < 1.0:
        raise ValueError("Taylor core area fraction must lie in (0, 1)")
    if taylor_rise_velocity <= 0.0:
        raise ValueError("Taylor rise velocity must be positive")
    if swept_fraction is None:
        swept = (1.0,) * parameters.cell_count
    else:
        swept = _tuple(swept_fraction, name="swept_fraction")
        if len(swept) != parameters.cell_count:
            raise ValueError("swept_fraction must contain one value per cell")
        if min(swept) < 0.0 or max(swept) > 1.0:
            raise ValueError("swept fractions must lie in [0, 1]")

    film_velocity = -(
        taylor_core_area_fraction
        * taylor_rise_velocity
        / (1.0 - taylor_core_area_fraction)
    )
    up_area: list[float] = []
    up_q: list[float] = []
    down_area: list[float] = []
    down_q: list[float] = []
    geometric: list[float] = []
    energy_before = 0.0
    energy_after = 0.0
    for index, (a_total, q_total, swept_cell) in enumerate(
        zip(area, discharge, swept)
    ):
        if a_total < 0.0 or a_total > parameters.full_area + parameters.packing_tolerance:
            raise StateAdmissibilityError(
                f"inadmissible breakthrough liquid area in cell {index}"
            )
        if a_total == 0.0 and q_total != 0.0:
            raise StateAdmissibilityError(
                f"dry breakthrough cell carries momentum in cell {index}"
            )
        film_capacity = (
            swept_cell
            * (1.0 - taylor_core_area_fraction)
            * parameters.full_area
        )
        if q_total > 0.0:
            a_core, q_core = a_total, q_total
            a_film, q_film = 0.0, 0.0
        else:
            a_core, q_core = 0.0, 0.0
            a_film, q_film = a_total, q_total
        up_area.append(a_core)
        up_q.append(q_core)
        down_area.append(a_film)
        down_q.append(q_film)
        geometric.append(film_capacity)
        energy_before += _kinematic_energy(a_total, q_total)
        energy_after += _kinematic_energy(a_core, q_core)
        energy_after += _kinematic_energy(a_film, q_film)

    mapped = VerticalTwoStreamState.from_iterables(
        upward_area=up_area,
        upward_discharge=up_q,
        downward_area=down_area,
        downward_discharge=down_q,
    )
    area_residual = tuple(
        mapped_area - original
        for mapped_area, original in zip(mapped.liquid_area, area)
    )
    momentum_residual = tuple(
        mapped_q - original
        for mapped_q, original in zip(mapped.liquid_discharge, discharge)
    )
    return TaylorBreakthroughMappingResult(
        state=mapped,
        geometric_film_area=tuple(geometric),
        falling_film_velocity=film_velocity,
        area_residual=area_residual,
        momentum_residual=momentum_residual,
        kinematic_energy_change=energy_after - energy_before,
    )


def _gas_friction_factor(reynolds: float) -> float:
    """Gas-side Fanning factor used by the existing coupled gas network."""

    re = max(float(reynolds), 1.0e-12)
    if re < 2100.0:
        value = 16.0 / re
    else:
        value = 0.046 * re**-0.2
    return min(max(value, 0.0), 4.0)


def implicit_physical_three_body_drag_exchange(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    gas: PhysicalGasInterphaseState,
    *,
    dt: float,
    topology_velocity_tolerance: float = 1.0e-12,
    preserve_stopped_partition: Iterable[bool] | None = None,
) -> PhysicalThreeBodyDragResult:
    """Implicitly exchange momentum among gas, core liquid, and wall film.

    For each gas--liquid interface the force coefficient is

    ``C_i = 0.5 f_i rho_g P_i dz``

    with ``f_i`` obtained from the gas Reynolds number and hydraulic diameter.
    The quadratic magnitude is frozen at the start of the source step, while
    all three final velocities are solved in one coupled implicit system.  If
    only one liquid stream exists, the result reduces algebraically to
    ``r_new = r/(1+beta |r| dt)`` used by the existing gas network.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    _validate_state_geometry(state, parameters)
    n = parameters.cell_count
    if len(gas.gas_mass) != n:
        raise ValueError("gas and liquid cell counts differ")
    preserved_partition = (
        (False,) * n
        if preserve_stopped_partition is None
        else tuple(bool(value) for value in preserve_stopped_partition)
    )
    if len(preserved_partition) != n:
        raise ValueError(
            "stopped-partition mask must contain one value per cell"
        )
    for cell, (gas_area, liquid_area) in enumerate(
        zip(gas.gas_area, state.liquid_area)
    ):
        if gas_area + liquid_area > parameters.full_area + parameters.packing_tolerance:
            raise PackingViolationError(
                f"physical gas and liquid areas over-pack cell {cell}"
            )

    rho_l = parameters.liquid_density
    dz = parameters.cell_length
    q_up_new = list(state.upward_discharge)
    q_down_new = list(state.downward_discharge)
    gas_momentum_new = list(gas.gas_momentum)
    gas_impulse: list[float] = []
    up_impulse: list[float] = []
    down_impulse: list[float] = []
    f_up_values: list[float] = []
    f_down_values: list[float] = []
    residuals: list[float] = []

    for cell in range(n):
        mg = gas.gas_mass[cell]
        ag = gas.gas_area[cell]
        initial_gas_momentum = gas.gas_momentum[cell]
        ml_up = rho_l * state.upward_area[cell] * dz
        ml_down = rho_l * state.downward_area[cell] * dz
        initial_up_momentum = rho_l * state.upward_discharge[cell] * dz
        initial_down_momentum = rho_l * state.downward_discharge[cell] * dz
        if mg <= 0.0 or ag <= 0.0:
            gas_impulse.append(0.0)
            up_impulse.append(0.0)
            down_impulse.append(0.0)
            f_up_values.append(0.0)
            f_down_values.append(0.0)
            residuals.append(0.0)
            continue

        gas_velocity = initial_gas_momentum / mg
        rho_g = mg / (ag * dz)
        coefficients: list[tuple[float, float, float]] = []
        # Tuple entries are liquid mass, old velocity, and frozen K [kg/s].
        for liquid_mass, area, discharge, perimeter, diameter in (
            (
                ml_up,
                state.upward_area[cell],
                state.upward_discharge[cell],
                gas.upward_interface_perimeter[cell],
                gas.upward_hydraulic_diameter[cell],
            ),
            (
                ml_down,
                state.downward_area[cell],
                state.downward_discharge[cell],
                gas.downward_interface_perimeter[cell],
                gas.downward_hydraulic_diameter[cell],
            ),
        ):
            if liquid_mass <= 0.0 or area <= 0.0 or perimeter <= 0.0:
                coefficients.append((liquid_mass, 0.0, 0.0))
                continue
            liquid_velocity = discharge / area
            relative = gas_velocity - liquid_velocity
            reynolds = (
                rho_g
                * abs(relative)
                * max(diameter, 0.0)
                / gas.gas_viscosity
            )
            friction = _gas_friction_factor(reynolds)
            force_coefficient = 0.5 * friction * rho_g * perimeter * dz
            coefficients.append(
                (
                    liquid_mass,
                    liquid_velocity,
                    force_coefficient * abs(relative),
                )
            )
        f_up_values.append(
            0.0
            if coefficients[0][2] == 0.0
            else _gas_friction_factor(
                rho_g
                * abs(gas_velocity - coefficients[0][1])
                * gas.upward_hydraulic_diameter[cell]
                / gas.gas_viscosity
            )
        )
        f_down_values.append(
            0.0
            if coefficients[1][2] == 0.0
            else _gas_friction_factor(
                rho_g
                * abs(gas_velocity - coefficients[1][1])
                * gas.downward_hydraulic_diameter[cell]
                / gas.gas_viscosity
            )
        )

        lambdas = []
        for liquid_mass, _, frozen_k in coefficients:
            if liquid_mass <= 0.0 or frozen_k <= 0.0:
                lambdas.append(0.0)
            else:
                lambdas.append(
                    dt * frozen_k * liquid_mass
                    / (liquid_mass + dt * frozen_k)
                )
        denominator = mg + sum(lambdas)
        gas_velocity_new = (
            mg * gas_velocity
            + sum(
                relaxation * data[1]
                for relaxation, data in zip(lambdas, coefficients)
            )
        ) / denominator

        liquid_velocities_new: list[float] = []
        for liquid_mass, liquid_velocity, frozen_k in coefficients:
            if liquid_mass <= 0.0 or frozen_k <= 0.0:
                liquid_velocities_new.append(liquid_velocity)
            else:
                liquid_velocities_new.append(
                    (
                        liquid_mass * liquid_velocity
                        + dt * frozen_k * gas_velocity_new
                    )
                    / (liquid_mass + dt * frozen_k)
                )
        final_up_momentum = ml_up * liquid_velocities_new[0]
        final_down_momentum = ml_down * liquid_velocities_new[1]
        initial_total = (
            initial_gas_momentum
            + initial_up_momentum
            + initial_down_momentum
        )
        # Close the three-body ledger exactly; this removes only roundoff from
        # the analytically conservative implicit solve.
        final_gas_momentum = (
            initial_total - final_up_momentum - final_down_momentum
        )
        gas_momentum_new[cell] = final_gas_momentum
        q_up_new[cell] = final_up_momentum / (rho_l * dz)
        q_down_new[cell] = final_down_momentum / (rho_l * dz)
        gas_impulse.append(final_gas_momentum - initial_gas_momentum)
        up_impulse.append(final_up_momentum - initial_up_momentum)
        down_impulse.append(final_down_momentum - initial_down_momentum)
        residuals.append(
            gas_impulse[-1] + up_impulse[-1] + down_impulse[-1]
        )

    topology = conservative_directional_topology_transfer(
        upward_area=state.upward_area,
        upward_discharge=q_up_new,
        downward_area=state.downward_area,
        downward_discharge=q_down_new,
        velocity_tolerance=topology_velocity_tolerance,
        preserve_stopped_partition=preserved_partition,
    )
    return PhysicalThreeBodyDragResult(
        state=topology.state,
        gas_momentum=tuple(gas_momentum_new),
        gas_impulse=tuple(gas_impulse),
        upward_liquid_impulse=tuple(up_impulse),
        downward_liquid_impulse=tuple(down_impulse),
        upward_friction_factor=tuple(f_up_values),
        downward_friction_factor=tuple(f_down_values),
        cell_momentum_residual=tuple(residuals),
        topology_transfer=topology,
    )


@dataclass(frozen=True)
class VerticalTwoStreamLedger:
    """Domain-integrated conservation audit for one accepted stage.

    Liquid momentum fields are kinematic (the conserved discharge integrated
    over ``z``).  ``gas_on_liquid_impulse`` and ``gas_reaction_impulse`` are
    physical impulses, obtained by multiplying the kinematic liquid change by
    ``rho_l``; they can therefore be transferred to a mass-based gas momentum
    equation without a unit mismatch.
    """

    initial_upward_volume: float
    final_upward_volume: float
    upward_boundary_volume_change: float
    upward_topology_volume_transfer: float
    upward_volume_residual: float
    initial_downward_volume: float
    final_downward_volume: float
    downward_boundary_volume_change: float
    downward_topology_volume_transfer: float
    downward_volume_residual: float
    initial_liquid_momentum: float
    final_liquid_momentum: float
    boundary_momentum_impulse: float
    pressure_gravity_impulse: float
    wall_impulse: float
    interstream_upward_impulse: float
    interstream_downward_impulse: float
    topology_upward_momentum_transfer: float
    topology_downward_momentum_transfer: float
    topology_kinematic_energy_loss: float
    gas_on_liquid_kinematic_impulse: float
    gas_on_liquid_impulse: float
    gas_reaction_impulse: float
    liquid_momentum_residual: float
    requested_upward_boundary_volume_change: float = 0.0
    requested_downward_boundary_volume_change: float = 0.0
    upward_capacity_boundary_volume_change: float = 0.0
    downward_capacity_boundary_volume_change: float = 0.0
    requested_boundary_momentum_impulse: float = 0.0
    capacity_constraint_momentum_impulse: float = 0.0
    capacity_pressure_kinematic_impulse: float = 0.0
    capacity_pressure_physical_impulse: float = 0.0
    capacity_pressure_bottom_impulse_on_liquid: float = 0.0
    capacity_pressure_top_impulse_on_liquid: float = 0.0
    capacity_pressure_internal_area_impulse_on_liquid: float = 0.0
    capacity_pressure_boundary_owner_reaction_impulse: float = 0.0
    capacity_pressure_interface_owner_reaction_impulse: float = 0.0
    capacity_pressure_decomposition_residual: float = 0.0
    capacity_pressure_coupled_momentum_residual: float = 0.0
    capacity_pressure_maximum_active_residual: float = 0.0
    capacity_pressure_maximum_kkt_residual: float = 0.0
    capacity_pressure_maximum_complementarity_residual: float = 0.0
    capacity_pressure_bottom_bulk_anchor_residual: float = 0.0
    capacity_pressure_working_set_releases: int = 0
    bottom_downward_reaction_momentum_impulse: float = 0.0
    maximum_packing_residual: float = 0.0
    capacity_projection_iterations: int = 0

    @property
    def total_volume_residual(self) -> float:
        return self.upward_volume_residual + self.downward_volume_residual

    @property
    def interstream_momentum_residual(self) -> float:
        return self.interstream_upward_impulse + self.interstream_downward_impulse

    @property
    def liquid_gas_exchange_residual(self) -> float:
        return self.gas_on_liquid_impulse + self.gas_reaction_impulse

    @property
    def capacity_boundary_volume_residual(self) -> float:
        """Close accepted boundary volume against request plus constraint."""

        upward = (
            self.upward_boundary_volume_change
            - self.requested_upward_boundary_volume_change
            - self.upward_capacity_boundary_volume_change
        )
        downward = (
            self.downward_boundary_volume_change
            - self.requested_downward_boundary_volume_change
            - self.downward_capacity_boundary_volume_change
        )
        return upward + downward

    @property
    def capacity_momentum_ledger_residual(self) -> float:
        """Close accepted momentum flux against request plus reaction."""

        return (
            self.boundary_momentum_impulse
            - self.requested_boundary_momentum_impulse
            - self.capacity_constraint_momentum_impulse
        )


@dataclass(frozen=True)
class VerticalTwoStreamStepResult:
    state: VerticalTwoStreamState
    upward_area_flux: tuple[float, ...]
    downward_area_flux: tuple[float, ...]
    upward_momentum_flux: tuple[float, ...]
    downward_momentum_flux: tuple[float, ...]
    upward_donor_factor: tuple[float, ...]
    downward_donor_factor: tuple[float, ...]
    topology_transfer: DirectionalTopologyTransferResult
    ledger: VerticalTwoStreamLedger
    upward_receiving_factor: tuple[float, ...] = ()
    downward_receiving_factor: tuple[float, ...] = ()
    upward_capacity_volume_correction: tuple[float, ...] = ()
    downward_capacity_volume_correction: tuple[float, ...] = ()
    upward_capacity_momentum_impulse: tuple[float, ...] = ()
    downward_capacity_momentum_impulse: tuple[float, ...] = ()
    capacity_pressure_cell_impulse: tuple[float, ...] = ()
    capacity_pressure_face_momentum_flux: tuple[float, ...] = ()


@dataclass(frozen=True)
class VerticalTwoStreamLiquidProvenanceState:
    """Conservative source-one liquid inventory in the two streams.

    The transported scalar is ``A chi`` rather than the non-conservative
    marker ``chi`` itself.  ``chi=0`` denotes liquid initially present in the
    riser and ``chi=1`` denotes liquid entering from the horizontal pipe at
    the bottom boundary.  A value between zero and the corresponding liquid
    area represents a mixed cell.  Source-zero inventory is recovered exactly
    as ``A - A chi``, so only one additional conserved field per stream is
    required.
    """

    upward_source1_area: tuple[float, ...]
    downward_source1_area: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.upward_source1_area
            or len(self.upward_source1_area) != len(self.downward_source1_area)
        ):
            raise StateAdmissibilityError(
                "liquid-provenance arrays need one common nonzero length"
            )
        if not _finite(
            *self.upward_source1_area,
            *self.downward_source1_area,
        ):
            raise StateAdmissibilityError("liquid-provenance state must be finite")
        if min(*self.upward_source1_area, *self.downward_source1_area) < 0.0:
            raise StateAdmissibilityError(
                "liquid-provenance inventories cannot be negative"
            )

    @classmethod
    def from_iterables(
        cls,
        *,
        upward_source1_area: Iterable[float],
        downward_source1_area: Iterable[float],
    ) -> "VerticalTwoStreamLiquidProvenanceState":
        return cls(
            upward_source1_area=_tuple(
                upward_source1_area,
                name="upward_source1_area",
            ),
            downward_source1_area=_tuple(
                downward_source1_area,
                name="downward_source1_area",
            ),
        )

    @classmethod
    def initial_riser_water(
        cls,
        hydraulic_state: VerticalTwoStreamState,
    ) -> "VerticalTwoStreamLiquidProvenanceState":
        """Label every initially present liquid parcel as source zero."""

        zeros = (0.0,) * hydraulic_state.cell_count
        return cls(
            upward_source1_area=zeros,
            downward_source1_area=zeros,
        )

    @property
    def cell_count(self) -> int:
        return len(self.upward_source1_area)

    @property
    def source1_area(self) -> tuple[float, ...]:
        return tuple(
            upward + downward
            for upward, downward in zip(
                self.upward_source1_area,
                self.downward_source1_area,
            )
        )

    def source1_fraction(
        self,
        hydraulic_state: VerticalTwoStreamState,
        *,
        dry_area_tolerance: float = 1.0e-14,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return diagnostic ``chi`` values for the upward/downward streams."""

        _validate_liquid_provenance_state(
            self,
            hydraulic_state,
            tolerance=dry_area_tolerance,
        )
        upward = tuple(
            0.0 if area <= dry_area_tolerance else marked / area
            for marked, area in zip(
                self.upward_source1_area,
                hydraulic_state.upward_area,
            )
        )
        downward = tuple(
            0.0 if area <= dry_area_tolerance else marked / area
            for marked, area in zip(
                self.downward_source1_area,
                hydraulic_state.downward_area,
            )
        )
        return upward, downward


@dataclass(frozen=True)
class VerticalTwoStreamLiquidProvenanceBoundaries:
    """Source-one fractions on the two possible external inflow faces.

    At the bottom only the upward stream is an inflow; at the top only the
    downward stream is an inflow.  The other two boundary fluxes are outflows
    and automatically use the adjacent cell's donor fraction.
    """

    bottom_upward_source1_fraction: float = 1.0
    top_downward_source1_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.bottom_upward_source1_fraction,
            self.top_downward_source1_fraction,
        )
        if not _finite(*values) or min(values) < 0.0 or max(values) > 1.0:
            raise ValueError(
                "liquid-provenance boundary fractions must lie in [0, 1]"
            )


@dataclass(frozen=True)
class LiquidProvenanceTopologyTransferResult:
    """Source-one inventory after one hydraulic topology remap."""

    state: VerticalTwoStreamLiquidProvenanceState
    upward_source1_area_transfer: tuple[float, ...]
    downward_source1_area_transfer: tuple[float, ...]
    source1_area_residual: float


@dataclass(frozen=True)
class VerticalTwoStreamLiquidProvenanceLedger:
    """Domain-integrated source-one volume audit for one FV stage."""

    initial_upward_source1_volume: float
    final_upward_source1_volume: float
    upward_boundary_source1_volume_change: float
    upward_topology_source1_volume_transfer: float
    upward_source1_volume_residual: float
    initial_downward_source1_volume: float
    final_downward_source1_volume: float
    downward_boundary_source1_volume_change: float
    downward_topology_source1_volume_transfer: float
    downward_source1_volume_residual: float

    @property
    def initial_source1_volume(self) -> float:
        return (
            self.initial_upward_source1_volume
            + self.initial_downward_source1_volume
        )

    @property
    def final_source1_volume(self) -> float:
        return self.final_upward_source1_volume + self.final_downward_source1_volume

    @property
    def boundary_source1_volume_change(self) -> float:
        return (
            self.upward_boundary_source1_volume_change
            + self.downward_boundary_source1_volume_change
        )

    @property
    def topology_source1_volume_residual(self) -> float:
        return (
            self.upward_topology_source1_volume_transfer
            + self.downward_topology_source1_volume_transfer
        )

    @property
    def source1_volume_residual(self) -> float:
        return (
            self.upward_source1_volume_residual
            + self.downward_source1_volume_residual
        )


@dataclass(frozen=True)
class VerticalTwoStreamLiquidProvenanceStepResult:
    """Passive provenance update driven by an accepted hydraulic FV step."""

    state: VerticalTwoStreamLiquidProvenanceState
    upward_source1_area_flux: tuple[float, ...]
    downward_source1_area_flux: tuple[float, ...]
    topology_transfer: LiquidProvenanceTopologyTransferResult
    ledger: VerticalTwoStreamLiquidProvenanceLedger


def hydrostatic_face_pressures(
    parameters: VerticalTwoStreamParameters,
    *,
    bottom_pressure: float,
) -> tuple[float, ...]:
    """Return face pressures whose gradient exactly balances axial gravity."""

    if not math.isfinite(bottom_pressure) or bottom_pressure <= 0.0:
        raise ValueError("bottom pressure must be finite and positive")
    rho_g_dz = parameters.liquid_density * parameters.gravity * parameters.cell_length
    return tuple(bottom_pressure - face * rho_g_dz for face in range(parameters.cell_count + 1))


def _validate_state_geometry(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
) -> None:
    if state.cell_count != parameters.cell_count:
        raise StateAdmissibilityError("state and parameter cell counts differ")
    for index, area in enumerate(state.liquid_area):
        if area > parameters.full_area + parameters.packing_tolerance:
            raise PackingViolationError(f"liquid area exceeds the pipe area in cell {index}")
    for index, (area, discharge) in enumerate(zip(state.upward_area, state.upward_discharge)):
        if (
            area <= parameters.dry_area_tolerance
            and abs(discharge) > parameters.dry_area_tolerance
        ):
            raise StateAdmissibilityError(
                f"near-dry upward stream carries discharge in cell {index}"
            )
    for index, (area, discharge) in enumerate(zip(state.downward_area, state.downward_discharge)):
        if (
            area <= parameters.dry_area_tolerance
            and abs(discharge) > parameters.dry_area_tolerance
        ):
            raise StateAdmissibilityError(
                f"near-dry downward stream carries discharge in cell {index}"
            )


def _velocity(area: float, discharge: float, tolerance: float) -> float:
    return 0.0 if area <= tolerance else discharge / area


def _donor_factor(area: float, outgoing_rate: float, dt_over_dz: float) -> float:
    requested = dt_over_dz * max(outgoing_rate, 0.0)
    if requested <= area or requested == 0.0:
        return 1.0
    return area / requested


def bottom_net_receiving_rate_capacity(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    dt: float,
    top: DirectionalBoundaryFlux = DirectionalBoundaryFlux(),
    liquid_capacity_area: Iterable[float] | None = None,
) -> float:
    """Return the largest admissible signed inflow at the bottom face.

    The T node owns the bottom boundary flux while this FV owner enforces the
    shared cross-sectional capacity of every riser cell.  This predictor is
    the suffix-feasibility part of :func:`_project_shared_receiving_capacity`
    with an unconstrained bottom face.  It uses the same old directional
    state, donor limiter, top boundary, geometry tolerance and time step as the
    subsequent FV stage.  Therefore a node flux capped by this value is not
    silently reduced later, and the horizontal and vertical branches can
    commit one atomic liquid transaction.

    ``liquid_capacity_area`` optionally supplies a physical liquid-area cap
    for each cell.  This is used after Taylor breakthrough to preserve an
    already opened material gas corridor; it limits only net filling, so
    simultaneous equal upward and downward circulation remains available.

    The returned quantity is a *net* rate: simultaneous equal upward and
    downward circulation consumes no receiving volume.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("receiving-capacity time step must be finite and positive")
    _validate_state_geometry(state, parameters)
    n = parameters.cell_count
    dz = parameters.cell_length
    dt_over_dz = dt / dz

    upward = [0.0] * (n + 1)
    downward = [0.0] * (n + 1)
    upward[n] = top.upward_rate
    downward[n] = top.downward_rate
    for face in range(1, n):
        upward[face] = state.upward_discharge[face - 1]
        downward[face] = -state.downward_discharge[face]

    # Match the actual FV donor limiter on all faces downstream of the unknown
    # bottom trace.  Upward flow is donated by the lower cell; downward flow by
    # the upper cell.  The top downward rate is an imposed inflow and has no
    # in-domain donor.
    alpha_up = [
        _donor_factor(state.upward_area[cell], upward[cell + 1], dt_over_dz)
        for cell in range(n)
    ]
    alpha_down = [
        _donor_factor(state.downward_area[cell], downward[cell], dt_over_dz)
        for cell in range(n)
    ]
    for face in range(1, n + 1):
        upward[face] *= alpha_up[face - 1]
    for face in range(1, n):
        downward[face] *= alpha_down[face]

    if liquid_capacity_area is None:
        physical_capacity = [parameters.full_area] * n
    else:
        physical_capacity = list(
            _tuple(liquid_capacity_area, name="liquid_capacity_area")
        )
        if len(physical_capacity) != n:
            raise ValueError(
                "liquid_capacity_area must contain one value per cell"
            )
        for cell, capacity in enumerate(physical_capacity):
            present = state.upward_area[cell] + state.downward_area[cell]
            if capacity < present - parameters.packing_tolerance:
                raise PackingViolationError(
                    "liquid capacity is smaller than the existing inventory "
                    f"in cell {cell}"
                )
            if capacity > parameters.full_area + parameters.packing_tolerance:
                raise PackingViolationError(
                    f"liquid capacity exceeds the pipe area in cell {cell}"
                )
    admissible_capacity_area = [
        min(capacity, parameters.full_area) + parameters.packing_tolerance
        for capacity in physical_capacity
    ]
    area_scale = max(
        *admissible_capacity_area,
        *(a_up + a_down for a_up, a_down in zip(
            state.upward_area,
            state.downward_area,
        )),
    )
    roundoff_area = max(
        256.0 * 2.220446049250313e-16 * area_scale,
        1.0e-18,
    )
    capacity_area = [
        capacity - roundoff_area
        for capacity in admissible_capacity_area
    ]
    void_rate = [
        (
            capacity_area[cell]
            - state.upward_area[cell]
            - state.downward_area[cell]
        )
        / dt_over_dz
        for cell in range(n)
    ]
    flux_scale = max(
        *upward[1:],
        *downward[1:],
        max(capacity_area) / dt_over_dz,
        1.0e-30,
    )
    roundoff_flux = 256.0 * 2.220446049250313e-16 * flux_scale

    feasible_high = [0.0] * (n + 1)
    feasible_high[n] = upward[n]
    for face in range(n - 1, 0, -1):
        lower = -downward[face]
        feasible_high[face] = min(
            upward[face],
            feasible_high[face + 1] + void_rate[face],
        )
        if feasible_high[face] < lower - roundoff_flux:
            raise PackingViolationError(
                "no downstream donor-limited flux can receive a bottom "
                f"T-node inflow upstream of cell {face}"
            )
        feasible_high[face] = max(feasible_high[face], lower)

    return float(feasible_high[1] + void_rate[0])


def _quadratic_relaxation_velocity(
    velocity: float,
    target_velocity: float,
    coefficient: float,
    dt: float,
) -> float:
    relative = velocity - target_velocity
    return target_velocity + relative / (1.0 + coefficient * abs(relative) * dt)


@dataclass(frozen=True)
class _ReceivingCapacityProjection:
    """Accepted shared fluxes and the conservative projection audit.

    The proposed directional fluxes already satisfy donor positivity.  This
    second finite-volume limiter enforces the complementary receiver bound

    ``A_up[i] + A_down[i] <= A_r``.

    The two directional magnitudes are decomposed into a common circulation
    and one signed net flux per geometric face.  A finite forward/backward
    interval solve projects only the net flux divergence onto the cell
    capacity while retaining the largest admissible common circulation.
    Because an interior face has one stored value, every correction is shared
    by its two neighbours; no liquid is clipped, created, or moved between the
    directional labels.
    """

    upward_area_flux: tuple[float, ...]
    downward_area_flux: tuple[float, ...]
    upward_momentum_flux: tuple[float, ...]
    downward_momentum_flux: tuple[float, ...]
    upward_receiving_factor: tuple[float, ...]
    downward_receiving_factor: tuple[float, ...]
    upward_volume_correction: tuple[float, ...]
    downward_volume_correction: tuple[float, ...]
    upward_momentum_impulse: tuple[float, ...]
    downward_momentum_impulse: tuple[float, ...]
    maximum_packing_residual: float
    iterations: int


def _project_shared_receiving_capacity(
    *,
    upward_area: tuple[float, ...],
    downward_area: tuple[float, ...],
    full_area: float,
    packing_tolerance: float,
    dt: float,
    dz: float,
    upward_area_flux: list[float],
    downward_area_flux: list[float],
    upward_momentum_flux: list[float],
    downward_momentum_flux: list[float],
    liquid_capacity_area: Iterable[float] | None = None,
) -> _ReceivingCapacityProjection:
    """Project donor-limited fluxes onto the shared cell capacity.

    Let ``U_f >= 0`` and ``D_f >= 0`` be the upward and downward magnitudes on
    face ``f``.  Cell packing depends only on their signed net flux

    ``J_f = U_f - D_f``

    through ``A_new[i] = A_old[i] + dt/dz*(J_i-J_{i+1})``.  Reducing the
    already donor-limited directional magnitudes permits exactly the interval
    ``-D_f <= J_f <= U_f``.  A backward pass computes the feasible upper
    interval at every face and one forward pass selects the requested net flux
    whenever it remains feasible.  The directional pair is then recovered by
    reducing only the magnitude required by the net correction, which retains
    the maximum possible counter-current circulation.

    This is a finite flux-space complementarity projection.  It has no
    convergence loop and performs no post-update area clipping or directional
    inventory transfer.
    """

    n = len(upward_area)
    if len(downward_area) != n:
        raise ValueError("directional area arrays need one common length")
    expected_faces = n + 1
    arrays = (
        upward_area_flux,
        downward_area_flux,
        upward_momentum_flux,
        downward_momentum_flux,
    )
    if any(len(values) != expected_faces for values in arrays):
        raise ValueError("capacity projection needs N+1 fluxes per field")

    requested_up_area = tuple(upward_area_flux)
    requested_down_area = tuple(downward_area_flux)
    requested_up_momentum = tuple(upward_momentum_flux)
    requested_down_momentum = tuple(downward_momentum_flux)
    dt_over_dz = dt / dz
    # The admissible state definition already treats ``packing_tolerance`` as
    # geometry tolerance.  Project to that same closed set, then resolve the
    # interval algebra to machine roundoff; using ``full_area`` here would make
    # an otherwise admissible nearly closed cell spuriously infeasible.
    if liquid_capacity_area is None:
        physical_capacity = [full_area] * n
    else:
        physical_capacity = list(
            _tuple(liquid_capacity_area, name="liquid_capacity_area")
        )
        if len(physical_capacity) != n:
            raise ValueError(
                "liquid_capacity_area must contain one value per cell"
            )
        for cell, capacity in enumerate(physical_capacity):
            present = upward_area[cell] + downward_area[cell]
            if capacity < present - packing_tolerance:
                raise PackingViolationError(
                    "liquid capacity is smaller than the existing inventory "
                    f"in cell {cell}"
                )
            if capacity > full_area + packing_tolerance:
                raise PackingViolationError(
                    f"liquid capacity exceeds the pipe area in cell {cell}"
                )
    admissible_capacity_area = [
        min(capacity, full_area) + packing_tolerance
        for capacity in physical_capacity
    ]
    area_scale = max(
        *admissible_capacity_area,
        *(a_up + a_down for a_up, a_down in zip(upward_area, downward_area)),
    )
    roundoff_area = max(
        256.0 * 2.220446049250313e-16 * area_scale,
        1.0e-18,
    )
    # Aim one floating-point guard band inside the public admissible set.  The
    # guard is not extra packing allowance; it prevents reconstruction
    # roundoff from crossing ``full_area + packing_tolerance`` downstream.
    capacity_area = [
        capacity - roundoff_area
        for capacity in admissible_capacity_area
    ]
    upward_magnitude = [max(value, 0.0) for value in requested_up_area]
    downward_magnitude = [max(-value, 0.0) for value in requested_down_area]
    if any(value < 0.0 for value in requested_up_area) or any(
        value > 0.0 for value in requested_down_area
    ):
        raise StateAdmissibilityError(
            "capacity projection requires directional donor-limited flux signs"
        )
    requested_net = [
        up - down
        for up, down in zip(upward_magnitude, downward_magnitude)
    ]
    requested_total_area = [
        upward_area[cell]
        + downward_area[cell]
        + dt_over_dz * (requested_net[cell] - requested_net[cell + 1])
        for cell in range(n)
    ]
    requested_maximum_excess = max(
        (
            area - admissible_capacity_area[cell]
            for cell, area in enumerate(requested_total_area)
        ),
        default=-full_area,
    )

    if requested_maximum_excess <= 0.0:
        accepted_net = requested_net
        iterations = 0
    else:
        lower = [-down for down in downward_magnitude]
        upper = list(upward_magnitude)
        void_rate = [
            (
                capacity_area[cell]
                - upward_area[cell]
                - downward_area[cell]
            )
            / dt_over_dz
            for cell in range(n)
        ]
        flux_scale = max(
            *upward_magnitude,
            *downward_magnitude,
            max(capacity_area) / dt_over_dz,
            1.0e-30,
        )
        roundoff_flux = 256.0 * 2.220446049250313e-16 * flux_scale

        # Suffix feasibility: face i can be no larger than the greatest
        # feasible face i+1 value plus cell i's available receiving rate.
        feasible_high = [0.0] * (n + 1)
        feasible_high[n] = upper[n]
        for face in range(n - 1, -1, -1):
            feasible_high[face] = min(
                upper[face],
                feasible_high[face + 1] + void_rate[face],
            )
            if feasible_high[face] < lower[face] - roundoff_flux:
                raise PackingViolationError(
                    "no donor-limited shared-face flux can satisfy receiving "
                    f"capacity upstream of cell {face}"
                )
            feasible_high[face] = max(feasible_high[face], lower[face])

        accepted_net = [0.0] * (n + 1)
        accepted_net[0] = min(
            max(requested_net[0], lower[0]),
            feasible_high[0],
        )
        for cell in range(n):
            feasible_low = max(
                lower[cell + 1],
                accepted_net[cell] - void_rate[cell],
            )
            if feasible_low > feasible_high[cell + 1] + roundoff_flux:
                raise PackingViolationError(
                    "shared receiving-capacity interval became empty at "
                    f"cell {cell}"
                )
            feasible_low = min(feasible_low, feasible_high[cell + 1])
            accepted_net[cell + 1] = min(
                max(requested_net[cell + 1], feasible_low),
                feasible_high[cell + 1],
            )
        iterations = 1

    # For a prescribed accepted net flux, change only the directional
    # magnitude needed to realise that net value.  This is the maximum-gross-
    # circulation member of the admissible pair and never increases either
    # donor-limited directional magnitude.
    accepted_upward = []
    accepted_downward = []
    for face, (net_requested, net_accepted) in enumerate(
        zip(requested_net, accepted_net)
    ):
        delta_net = net_accepted - net_requested
        if delta_net >= 0.0:
            up = upward_magnitude[face]
            down = downward_magnitude[face] - delta_net
        else:
            up = upward_magnitude[face] + delta_net
            down = downward_magnitude[face]
        if up < -roundoff_area / dt_over_dz or down < -roundoff_area / dt_over_dz:
            raise PackingViolationError(
                f"capacity projection exceeded a directional face bound at {face}"
            )
        accepted_upward.append(max(up, 0.0))
        accepted_downward.append(max(down, 0.0))

    fau = accepted_upward
    fad = [-value for value in accepted_downward]
    fmu = [
        requested_up_momentum[face]
        * (1.0 if requested_up_area[face] == 0.0 else fau[face] / requested_up_area[face])
        for face in range(n + 1)
    ]
    fmd = [
        requested_down_momentum[face]
        * (
            1.0
            if requested_down_area[face] == 0.0
            else fad[face] / requested_down_area[face]
        )
        for face in range(n + 1)
    ]

    final_total_area = [
        upward_area[cell]
        + downward_area[cell]
        + dt_over_dz
        * (
            fau[cell]
            - fau[cell + 1]
            + fad[cell]
            - fad[cell + 1]
        )
        for cell in range(n)
    ]
    maximum_packing_residual = max(
        (
            area - admissible_capacity_area[cell]
            for cell, area in enumerate(final_total_area)
        ),
        default=-full_area,
    )
    if maximum_packing_residual > 0.0:
        raise PackingViolationError(
            "finite receiving-capacity projection left a non-roundoff area "
            f"excess={maximum_packing_residual:.6e} m2"
        )

    up_volume_correction = tuple(
        dt
        * (
            (fau[cell] - requested_up_area[cell])
            - (fau[cell + 1] - requested_up_area[cell + 1])
        )
        for cell in range(n)
    )
    down_volume_correction = tuple(
        dt
        * (
            (fad[cell] - requested_down_area[cell])
            - (fad[cell + 1] - requested_down_area[cell + 1])
        )
        for cell in range(n)
    )
    up_momentum_impulse = tuple(
        dt
        * (
            (fmu[cell] - requested_up_momentum[cell])
            - (fmu[cell + 1] - requested_up_momentum[cell + 1])
        )
        for cell in range(n)
    )
    down_momentum_impulse = tuple(
        dt
        * (
            (fmd[cell] - requested_down_momentum[cell])
            - (fmd[cell + 1] - requested_down_momentum[cell + 1])
        )
        for cell in range(n)
    )

    def factors(accepted: list[float], requested: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            1.0 if value == 0.0 else accepted_value / value
            for accepted_value, value in zip(accepted, requested)
        )

    return _ReceivingCapacityProjection(
        upward_area_flux=tuple(fau),
        downward_area_flux=tuple(fad),
        upward_momentum_flux=tuple(fmu),
        downward_momentum_flux=tuple(fmd),
        upward_receiving_factor=factors(fau, requested_up_area),
        downward_receiving_factor=factors(fad, requested_down_area),
        upward_volume_correction=up_volume_correction,
        downward_volume_correction=down_volume_correction,
        upward_momentum_impulse=up_momentum_impulse,
        downward_momentum_impulse=down_momentum_impulse,
        maximum_packing_residual=maximum_packing_residual,
        iterations=iterations,
    )


def advance_vertical_two_stream_fv(
    state: VerticalTwoStreamState,
    parameters: VerticalTwoStreamParameters,
    *,
    dt: float,
    pressure_faces: Iterable[float],
    boundaries: VerticalTwoStreamBoundaries = VerticalTwoStreamBoundaries(),
    gas_coupling: GasMomentumCoupling | None = None,
    preserve_stopped_partition: Iterable[bool] | None = None,
    liquid_capacity_area: Iterable[float] | None = None,
    bottom_downward_reaction_flux: float = 0.0,
    enable_capacity_pressure_projection: bool = False,
) -> VerticalTwoStreamStepResult:
    """Advance one conservative first-order finite-volume stage.

    Area transport is directional donor-cell advection.  A local draining-time
    factor scales every outgoing face flux and its matching momentum flux, so
    no cell can donate more liquid than it contains.  A subsequent shared
    receiving-capacity projection prevents simultaneous counter-current
    arrivals from over-packing a cell.  Both operations act on one face value
    shared by its adjacent cells, so neither operation clips cell inventories.

    Pressure is supplied at the ``N+1`` faces.  With
    ``dp/dz = -rho_l g``, the discrete pressure and gravity sources cancel to
    roundoff, which gives an exact stationary hydrostatic column.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    bottom_reaction_flux = float(bottom_downward_reaction_flux)
    if not math.isfinite(bottom_reaction_flux) or bottom_reaction_flux < 0.0:
        raise ValueError(
            "bottom downward momentum reaction flux must be finite and non-negative"
        )
    _validate_state_geometry(state, parameters)
    pressure = tuple(float(value) for value in pressure_faces)
    n = parameters.cell_count
    if len(pressure) != n + 1 or not _finite(*pressure):
        raise ValueError("pressure_faces must contain N+1 finite values")
    if gas_coupling is not None:
        if len(gas_coupling.gas_area) != n:
            raise ValueError("gas coupling and liquid state cell counts differ")
        gas_liquid_areas = zip(gas_coupling.gas_area, state.liquid_area)
        for index, (gas_area, liquid_area) in enumerate(gas_liquid_areas):
            if gas_area + liquid_area > parameters.full_area + parameters.packing_tolerance:
                raise PackingViolationError(f"gas and liquid areas over-pack cell {index}")

    tol = parameters.dry_area_tolerance
    dz = parameters.cell_length
    dt_over_dz = dt / dz
    preserved_partition = (
        (False,) * n
        if preserve_stopped_partition is None
        else tuple(bool(value) for value in preserve_stopped_partition)
    )
    if len(preserved_partition) != n:
        raise ValueError(
            "stopped-partition mask must contain one value per cell"
        )
    capacity_area = (
        None
        if liquid_capacity_area is None
        else _tuple(liquid_capacity_area, name="liquid_capacity_area")
    )
    if capacity_area is not None and len(capacity_area) != n:
        raise ValueError("liquid_capacity_area must contain one value per cell")
    u_up = [
        _velocity(a, q, tol) for a, q in zip(state.upward_area, state.upward_discharge)
    ]
    u_down = [
        _velocity(a, q, tol) for a, q in zip(state.downward_area, state.downward_discharge)
    ]

    # One shared flux per face.  Interior upward flux uses its lower donor;
    # interior downward flux uses its upper donor.
    fau = [0.0] * (n + 1)
    fad = [0.0] * (n + 1)
    fmu = [0.0] * (n + 1)
    fmd = [0.0] * (n + 1)
    fau[0] = boundaries.bottom.upward_rate
    fmu[0] = boundaries.bottom.upward_momentum_flux
    fad[0] = boundaries.bottom.signed_downward_rate
    fmd[0] = boundaries.bottom.downward_momentum_flux
    fau[n] = boundaries.top.upward_rate
    fmu[n] = boundaries.top.upward_momentum_flux
    fad[n] = boundaries.top.signed_downward_rate
    fmd[n] = boundaries.top.downward_momentum_flux
    for face in range(1, n):
        lower = face - 1
        upper = face
        fau[face] = state.upward_discharge[lower]
        fmu[face] = state.upward_discharge[lower] * u_up[lower]
        fad[face] = state.downward_discharge[upper]
        fmd[face] = state.downward_discharge[upper] * u_down[upper]

    # Local donor positivity.  Upward liquid leaves through a cell's upper
    # face; downward liquid leaves through its lower face.
    alpha_up = [
        _donor_factor(state.upward_area[cell], fau[cell + 1], dt_over_dz)
        for cell in range(n)
    ]
    alpha_down = [
        _donor_factor(state.downward_area[cell], -fad[cell], dt_over_dz)
        for cell in range(n)
    ]
    for face in range(1, n + 1):
        donor = face - 1
        fau[face] *= alpha_up[donor]
        fmu[face] *= alpha_up[donor]
    for face in range(0, n):
        donor = face
        fad[face] *= alpha_down[donor]
        fmd[face] *= alpha_down[donor]

    donor_limited_fau = tuple(fau)
    donor_limited_fad = tuple(fad)
    donor_limited_fmu = tuple(fmu)
    donor_limited_fmd = tuple(fmd)
    capacity_projection = _project_shared_receiving_capacity(
        upward_area=state.upward_area,
        downward_area=state.downward_area,
        full_area=parameters.full_area,
        packing_tolerance=parameters.packing_tolerance,
        dt=dt,
        dz=dz,
        upward_area_flux=fau,
        downward_area_flux=fad,
        upward_momentum_flux=fmu,
        downward_momentum_flux=fmd,
        liquid_capacity_area=capacity_area,
    )
    fau = list(capacity_projection.upward_area_flux)
    fad = list(capacity_projection.downward_area_flux)
    fmu = list(capacity_projection.upward_momentum_flux)
    fmd = list(capacity_projection.downward_momentum_flux)
    # ``bottom_reaction_flux`` is only the physical turn-loss traction returned
    # by the mouth characteristic.  Apply that finite upward force to the
    # falling donor.  The much larger geometric/capacity rejection belongs to
    # the fixed T structure and is deliberately excluded by the caller; adding
    # the full structural reaction here caused the false 8-s reversal.
    fmd[0] += bottom_reaction_flux

    a_up = []
    a_down = []
    m_up = []
    m_down = []
    for cell in range(n):
        new_a_up = state.upward_area[cell] + dt_over_dz * (fau[cell] - fau[cell + 1])
        new_a_down = state.downward_area[cell] + dt_over_dz * (fad[cell] - fad[cell + 1])
        new_m_up = state.upward_discharge[cell] + dt_over_dz * (fmu[cell] - fmu[cell + 1])
        new_m_down = state.downward_discharge[cell] + dt_over_dz * (fmd[cell] - fmd[cell + 1])
        if new_a_up < -tol or new_a_down < -tol:
            raise StateAdmissibilityError("donor limiter failed to preserve non-negative area")
        a_up.append(max(new_a_up, 0.0))
        a_down.append(max(new_a_down, 0.0))
        m_up.append(0.0 if abs(new_a_up) <= tol else new_m_up)
        m_down.append(0.0 if abs(new_a_down) <= tol else new_m_down)

    pressure_gravity_up = [0.0] * n
    pressure_gravity_down = [0.0] * n
    wall_up = [0.0] * n
    wall_down = [0.0] * n
    exchange_up = [0.0] * n
    exchange_down = [0.0] * n
    gas_up = [0.0] * n
    gas_down = [0.0] * n

    # Common pressure and gravity impulse.
    for cell in range(n):
        pressure_gradient = (pressure[cell + 1] - pressure[cell]) / dz
        pressure_acceleration = -pressure_gradient / parameters.liquid_density
        accel = pressure_acceleration - parameters.gravity
        well_balanced_tolerance = (
            128.0
            * 2.220446049250313e-16
            * max(abs(pressure_acceleration), parameters.gravity, 1.0)
        )
        if abs(accel) <= well_balanced_tolerance:
            accel = 0.0
        pressure_gravity_up[cell] = dt * a_up[cell] * accel
        pressure_gravity_down[cell] = dt * a_down[cell] * accel
        m_up[cell] += pressure_gravity_up[cell]
        m_down[cell] += pressure_gravity_down[cell]

    # Darcy wall friction is integrated exactly for frozen area.
    for cell in range(n):
        if a_up[cell] > tol:
            velocity = m_up[cell] / a_up[cell]
            coefficient = parameters.wall_friction_up / (2.0 * parameters.up_diameter)
            relaxed = _quadratic_relaxation_velocity(velocity, 0.0, coefficient, dt)
            wall_up[cell] = a_up[cell] * (relaxed - velocity)
            m_up[cell] += wall_up[cell]
        if a_down[cell] > tol:
            velocity = m_down[cell] / a_down[cell]
            coefficient = parameters.wall_friction_down / (2.0 * parameters.down_diameter)
            relaxed = _quadratic_relaxation_velocity(velocity, 0.0, coefficient, dt)
            wall_down[cell] = a_down[cell] * (relaxed - velocity)
            m_down[cell] += wall_down[cell]

    # Exact frozen-area quadratic exchange.  Total liquid momentum is held
    # fixed while the relative velocity decays monotonically.
    for cell in range(n):
        if a_up[cell] <= tol or a_down[cell] <= tol or parameters.interstream_drag == 0.0:
            continue
        velocity_up = m_up[cell] / a_up[cell]
        velocity_down = m_down[cell] / a_down[cell]
        relative = velocity_up - velocity_down
        mix_area = a_up[cell] * a_down[cell] / (a_up[cell] + a_down[cell])
        relative_coefficient = (
            parameters.interstream_drag
            * mix_area
            * (1.0 / a_up[cell] + 1.0 / a_down[cell])
        )
        relative_new = relative / (1.0 + relative_coefficient * abs(relative) * dt)
        total_momentum = m_up[cell] + m_down[cell]
        velocity_down_new = (
            total_momentum - a_up[cell] * relative_new
        ) / (a_up[cell] + a_down[cell])
        velocity_up_new = velocity_down_new + relative_new
        exchange_up[cell] = a_up[cell] * velocity_up_new - m_up[cell]
        exchange_down[cell] = a_down[cell] * velocity_down_new - m_down[cell]
        m_up[cell] += exchange_up[cell]
        m_down[cell] += exchange_down[cell]

    # Optional liquid--gas exchange.  Gas velocity is frozen during this
    # substep; the exact opposite impulse is returned in the ledger.
    if gas_coupling is not None and gas_coupling.drag_coefficient > 0.0:
        for cell in range(n):
            gas_area = gas_coupling.gas_area[cell]
            gas_velocity = gas_coupling.gas_velocity[cell]
            if gas_area <= tol:
                continue
            for area, momentum, sink in (
                (a_up[cell], m_up[cell], gas_up),
                (a_down[cell], m_down[cell], gas_down),
            ):
                if area <= tol:
                    continue
                velocity = momentum / area
                contact_area = area * gas_area / parameters.full_area
                coefficient = gas_coupling.drag_coefficient * contact_area / area
                relaxed = _quadratic_relaxation_velocity(
                    velocity, gas_velocity, coefficient, dt
                )
                sink[cell] = area * (relaxed - velocity)
            m_up[cell] += gas_up[cell]
            m_down[cell] += gas_down[cell]

    # The face-capacity transaction above changes inventory but cannot by
    # itself supply the pressure reaction that makes a saturated liquid block
    # incompressible.  Project only the common/bulk velocity here, so the
    # directional slip is unchanged.  The accepted T-node rate remains the
    # boundary-face flux; a cell-average discharge is not the same quantity and
    # must not be forced to match it as an extra pressure equation.
    capacity_pressure_cell_impulse = [0.0] * n
    capacity_pressure_face_momentum_flux = [0.0] * (n + 1)
    capacity_pressure_physical = 0.0
    capacity_pressure_bottom = 0.0
    capacity_pressure_top = 0.0
    capacity_pressure_internal = 0.0
    capacity_pressure_boundary_reaction = 0.0
    capacity_pressure_interface_reaction = 0.0
    capacity_pressure_decomposition_residual = 0.0
    capacity_pressure_coupled_residual = 0.0
    capacity_pressure_active_residual = 0.0
    capacity_pressure_kkt_residual = 0.0
    capacity_pressure_complementarity_residual = 0.0
    capacity_pressure_bottom_anchor_residual = 0.0
    capacity_pressure_working_set_releases = 0
    # This projection is retained as an isolated diagnostic only.  It cannot
    # be enabled in the production split solve until the T-node boundary flux
    # is recoupled with the riser capacity pressure: a fixed accepted mouth
    # flux can be provably incompatible with the downstream donor mobility.
    if enable_capacity_pressure_projection and capacity_area is not None:
        provisional_up = tuple(m_up)
        provisional_down = tuple(m_down)
        current_pressure_area = [
            upward + downward
            for upward, downward in zip(a_up, a_down)
        ]
        # The conservative face solve admits the public packing roundoff band.
        # Treat that sub-tolerance excess as the current active cap rather than
        # asking the pressure projector to remove liquid inventory.
        pressure_capacity_area = [
            max(capacity, current)
            for capacity, current in zip(capacity_area, current_pressure_area)
        ]
        pressure_projection = project_capacity_pressure_active_set(
            upward_area=a_up,
            upward_discharge=provisional_up,
            downward_area=a_down,
            downward_discharge=provisional_down,
            bottom_upward_rate=fau[0],
            bottom_downward_rate=-fad[0],
            top_downward_rate=-fad[n],
            top_upward_rate=None,
            liquid_capacity_area=pressure_capacity_area,
            current_liquid_area=current_pressure_area,
            dt=dt,
            dz=dz,
            liquid_density=parameters.liquid_density,
            # Pressure may drive a labelled stream through zero.  Directional
            # relabelling is handled conservatively by the topology operator
            # immediately below; imposing the same sign guard in this solve can
            # over-constrain otherwise feasible packing rows.
            preserve_stopped_partition=None,
            enforce_boundary_cell_bulk_match=False,
        )
        m_up = list(pressure_projection.corrected_upward_discharge)
        m_down = list(pressure_projection.corrected_downward_discharge)
        capacity_pressure_cell_impulse = [
            corrected_up
            + corrected_down
            - old_up
            - old_down
            for corrected_up, corrected_down, old_up, old_down in zip(
                m_up,
                m_down,
                provisional_up,
                provisional_down,
            )
        ]

        # Equivalent kinematic face traction, with one zero-pressure gauge at
        # the upper face of each connected wet component.  It reproduces the
        # cell correction exactly through dt/dz*(F_left-F_right).
        cell = 0
        while cell < n:
            if a_up[cell] + a_down[cell] <= tol:
                cell += 1
                continue
            start = cell
            while (
                cell + 1 < n
                and a_up[cell + 1] + a_down[cell + 1] > tol
            ):
                cell += 1
            end = cell + 1
            capacity_pressure_face_momentum_flux[end] = 0.0
            for index in range(end - 1, start - 1, -1):
                capacity_pressure_face_momentum_flux[index] = (
                    capacity_pressure_face_momentum_flux[index + 1]
                    + dz / dt * capacity_pressure_cell_impulse[index]
                )
            cell = end

        pressure_ledger = pressure_projection.ledger
        capacity_pressure_physical = pressure_ledger.liquid_physical_impulse
        capacity_pressure_bottom = (
            pressure_ledger.bottom_pressure_impulse_on_liquid
        )
        capacity_pressure_top = pressure_ledger.top_pressure_impulse_on_liquid
        capacity_pressure_internal = (
            pressure_ledger.internal_area_pressure_impulse_on_liquid
        )
        capacity_pressure_boundary_reaction = (
            pressure_ledger.boundary_owner_reaction_impulse
        )
        capacity_pressure_interface_reaction = (
            pressure_ledger.interface_owner_reaction_impulse
        )
        capacity_pressure_decomposition_residual = (
            pressure_ledger.pressure_decomposition_residual
        )
        capacity_pressure_coupled_residual = (
            pressure_ledger.coupled_momentum_residual
        )
        capacity_pressure_active_residual = (
            pressure_projection.maximum_active_constraint_residual
        )
        capacity_pressure_kkt_residual = (
            pressure_projection.maximum_kkt_stationarity_residual
        )
        capacity_pressure_complementarity_residual = (
            pressure_projection.maximum_complementarity_residual
        )
        capacity_pressure_bottom_anchor_residual = (
            pressure_projection.bottom_bulk_anchor_residual
        )
        capacity_pressure_working_set_releases = (
            pressure_projection.working_set_capacity_releases
        )

    for cell in range(n):
        if a_up[cell] + a_down[cell] > parameters.full_area + parameters.packing_tolerance:
            raise PackingViolationError(
                f"two-stream stage over-packed cell {cell}; reduce dt or couple void pressure"
            )

    topology = conservative_directional_topology_transfer(
        upward_area=a_up,
        upward_discharge=m_up,
        downward_area=a_down,
        downward_discharge=m_down,
        velocity_tolerance=1.0e-12,
        preserve_stopped_partition=preserved_partition,
    )
    final_state = topology.state
    _validate_state_geometry(final_state, parameters)

    initial_up_volume = dz * sum(state.upward_area)
    final_up_volume = dz * sum(final_state.upward_area)
    up_boundary_change = dt * (fau[0] - fau[n])
    requested_up_boundary_change = dt * (
        donor_limited_fau[0] - donor_limited_fau[n]
    )
    up_capacity_boundary_change = (
        up_boundary_change - requested_up_boundary_change
    )
    up_topology_volume = dz * sum(topology.upward_area_transfer)
    initial_down_volume = dz * sum(state.downward_area)
    final_down_volume = dz * sum(final_state.downward_area)
    down_boundary_change = dt * (fad[0] - fad[n])
    requested_down_boundary_change = dt * (
        donor_limited_fad[0] - donor_limited_fad[n]
    )
    down_capacity_boundary_change = (
        down_boundary_change - requested_down_boundary_change
    )
    down_topology_volume = dz * sum(topology.downward_area_transfer)
    initial_liquid_momentum = dz * (
        sum(state.upward_discharge) + sum(state.downward_discharge)
    )
    final_liquid_momentum = dz * (
        sum(final_state.upward_discharge) + sum(final_state.downward_discharge)
    )
    boundary_momentum = dt * ((fmu[0] + fmd[0]) - (fmu[n] + fmd[n]))
    requested_boundary_momentum = dt * (
        (donor_limited_fmu[0] + donor_limited_fmd[0])
        - (donor_limited_fmu[n] + donor_limited_fmd[n])
    )
    capacity_constraint_momentum = (
        boundary_momentum - requested_boundary_momentum
    )
    pressure_gravity = dz * (
        sum(pressure_gravity_up) + sum(pressure_gravity_down)
    )
    wall = dz * (sum(wall_up) + sum(wall_down))
    exchange_up_total = dz * sum(exchange_up)
    exchange_down_total = dz * sum(exchange_down)
    topology_up_momentum = dz * sum(topology.upward_momentum_transfer)
    topology_down_momentum = dz * sum(topology.downward_momentum_transfer)
    gas_on_liquid_kinematic = dz * (sum(gas_up) + sum(gas_down))
    capacity_pressure_kinematic = dz * sum(
        capacity_pressure_cell_impulse
    )
    gas_on_liquid_physical = (
        parameters.liquid_density * gas_on_liquid_kinematic
    )
    expected_liquid_momentum = (
        initial_liquid_momentum
        + boundary_momentum
        + pressure_gravity
        + wall
        + exchange_up_total
        + exchange_down_total
        + gas_on_liquid_kinematic
        + capacity_pressure_kinematic
    )
    ledger = VerticalTwoStreamLedger(
        initial_upward_volume=initial_up_volume,
        final_upward_volume=final_up_volume,
        upward_boundary_volume_change=up_boundary_change,
        upward_topology_volume_transfer=up_topology_volume,
        upward_volume_residual=(
            final_up_volume
            - initial_up_volume
            - up_boundary_change
            - up_topology_volume
        ),
        initial_downward_volume=initial_down_volume,
        final_downward_volume=final_down_volume,
        downward_boundary_volume_change=down_boundary_change,
        downward_topology_volume_transfer=down_topology_volume,
        downward_volume_residual=(
            final_down_volume
            - initial_down_volume
            - down_boundary_change
            - down_topology_volume
        ),
        initial_liquid_momentum=initial_liquid_momentum,
        final_liquid_momentum=final_liquid_momentum,
        boundary_momentum_impulse=boundary_momentum,
        pressure_gravity_impulse=pressure_gravity,
        wall_impulse=wall,
        interstream_upward_impulse=exchange_up_total,
        interstream_downward_impulse=exchange_down_total,
        topology_upward_momentum_transfer=topology_up_momentum,
        topology_downward_momentum_transfer=topology_down_momentum,
        topology_kinematic_energy_loss=topology.kinematic_energy_loss * dz,
        gas_on_liquid_kinematic_impulse=gas_on_liquid_kinematic,
        gas_on_liquid_impulse=gas_on_liquid_physical,
        gas_reaction_impulse=-gas_on_liquid_physical,
        liquid_momentum_residual=final_liquid_momentum - expected_liquid_momentum,
        requested_upward_boundary_volume_change=requested_up_boundary_change,
        requested_downward_boundary_volume_change=requested_down_boundary_change,
        upward_capacity_boundary_volume_change=up_capacity_boundary_change,
        downward_capacity_boundary_volume_change=down_capacity_boundary_change,
        requested_boundary_momentum_impulse=requested_boundary_momentum,
        capacity_constraint_momentum_impulse=capacity_constraint_momentum,
        capacity_pressure_kinematic_impulse=(
            capacity_pressure_kinematic
        ),
        capacity_pressure_physical_impulse=capacity_pressure_physical,
        capacity_pressure_bottom_impulse_on_liquid=capacity_pressure_bottom,
        capacity_pressure_top_impulse_on_liquid=capacity_pressure_top,
        capacity_pressure_internal_area_impulse_on_liquid=(
            capacity_pressure_internal
        ),
        capacity_pressure_boundary_owner_reaction_impulse=(
            capacity_pressure_boundary_reaction
        ),
        capacity_pressure_interface_owner_reaction_impulse=(
            capacity_pressure_interface_reaction
        ),
        capacity_pressure_decomposition_residual=(
            capacity_pressure_decomposition_residual
        ),
        capacity_pressure_coupled_momentum_residual=(
            capacity_pressure_coupled_residual
        ),
        capacity_pressure_maximum_active_residual=(
            capacity_pressure_active_residual
        ),
        capacity_pressure_maximum_kkt_residual=capacity_pressure_kkt_residual,
        capacity_pressure_maximum_complementarity_residual=(
            capacity_pressure_complementarity_residual
        ),
        capacity_pressure_bottom_bulk_anchor_residual=(
            capacity_pressure_bottom_anchor_residual
        ),
        capacity_pressure_working_set_releases=(
            capacity_pressure_working_set_releases
        ),
        bottom_downward_reaction_momentum_impulse=(
            dt * bottom_reaction_flux
        ),
        maximum_packing_residual=capacity_projection.maximum_packing_residual,
        capacity_projection_iterations=capacity_projection.iterations,
    )
    return VerticalTwoStreamStepResult(
        state=final_state,
        upward_area_flux=tuple(fau),
        downward_area_flux=tuple(fad),
        upward_momentum_flux=tuple(fmu),
        downward_momentum_flux=tuple(fmd),
        upward_donor_factor=tuple(alpha_up),
        downward_donor_factor=tuple(alpha_down),
        upward_receiving_factor=capacity_projection.upward_receiving_factor,
        downward_receiving_factor=capacity_projection.downward_receiving_factor,
        upward_capacity_volume_correction=(
            capacity_projection.upward_volume_correction
        ),
        downward_capacity_volume_correction=(
            capacity_projection.downward_volume_correction
        ),
        upward_capacity_momentum_impulse=(
            capacity_projection.upward_momentum_impulse
        ),
        downward_capacity_momentum_impulse=(
            capacity_projection.downward_momentum_impulse
        ),
        capacity_pressure_cell_impulse=tuple(
            capacity_pressure_cell_impulse
        ),
        capacity_pressure_face_momentum_flux=tuple(
            capacity_pressure_face_momentum_flux
        ),
        topology_transfer=topology,
        ledger=ledger,
    )


def _validate_liquid_provenance_areas(
    provenance: VerticalTwoStreamLiquidProvenanceState,
    upward_area: tuple[float, ...],
    downward_area: tuple[float, ...],
    *,
    tolerance: float,
) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("liquid-provenance tolerance must be finite and non-negative")
    if (
        provenance.cell_count != len(upward_area)
        or len(upward_area) != len(downward_area)
    ):
        raise StateAdmissibilityError(
            "liquid-provenance and hydraulic cell counts differ"
        )
    for direction, marked_values, area_values in (
        ("upward", provenance.upward_source1_area, upward_area),
        ("downward", provenance.downward_source1_area, downward_area),
    ):
        for cell, (marked, area) in enumerate(zip(marked_values, area_values)):
            if area < -tolerance:
                raise StateAdmissibilityError(
                    f"negative {direction} hydraulic area in provenance cell {cell}"
                )
            if marked > max(area, 0.0) + tolerance:
                raise StateAdmissibilityError(
                    f"{direction} source-one area exceeds liquid area in cell {cell}: "
                    f"marked={marked:.12e}, area={area:.12e}, "
                    f"excess={marked - max(area, 0.0):.12e}, "
                    f"tolerance={tolerance:.12e}"
                )


def _validate_liquid_provenance_state(
    provenance: VerticalTwoStreamLiquidProvenanceState,
    hydraulic_state: VerticalTwoStreamState,
    *,
    tolerance: float,
) -> None:
    _validate_liquid_provenance_areas(
        provenance,
        hydraulic_state.upward_area,
        hydraulic_state.downward_area,
        tolerance=tolerance,
    )


def conservative_liquid_provenance_topology_transfer(
    provenance: VerticalTwoStreamLiquidProvenanceState,
    hydraulic_transfer: DirectionalTopologyTransferResult,
    *,
    area_tolerance: float = 1.0e-14,
) -> LiquidProvenanceTopologyTransferResult:
    """Apply the hydraulic topology event to source-one inventory.

    The hydraulic transfer stores ``new-old`` for area and momentum, so the
    complete pre-topology state is recoverable without repeating any momentum
    closure.  A two-label crossing swaps the complete provenance inventories.
    A merge carries both inventories into the surviving directional channel.
    The proportional fallback supports a future partial area transfer by
    moving the donor channel's local concentration with the transferred area.
    """

    final = hydraulic_transfer.state
    n = final.cell_count
    if provenance.cell_count != n:
        raise StateAdmissibilityError(
            "liquid-provenance and topology-transfer cell counts differ"
        )
    arrays = (
        hydraulic_transfer.upward_area_transfer,
        hydraulic_transfer.downward_area_transfer,
        hydraulic_transfer.upward_momentum_transfer,
        hydraulic_transfer.downward_momentum_transfer,
    )
    if any(len(values) != n for values in arrays):
        raise StateAdmissibilityError(
            "hydraulic topology-transfer arrays need one value per cell"
        )
    if not math.isfinite(area_tolerance) or area_tolerance < 0.0:
        raise ValueError("topology provenance tolerance must be finite and non-negative")

    before_up_area = tuple(
        area - transfer
        for area, transfer in zip(
            final.upward_area,
            hydraulic_transfer.upward_area_transfer,
        )
    )
    before_down_area = tuple(
        area - transfer
        for area, transfer in zip(
            final.downward_area,
            hydraulic_transfer.downward_area_transfer,
        )
    )
    before_up_discharge = tuple(
        discharge - transfer
        for discharge, transfer in zip(
            final.upward_discharge,
            hydraulic_transfer.upward_momentum_transfer,
        )
    )
    before_down_discharge = tuple(
        discharge - transfer
        for discharge, transfer in zip(
            final.downward_discharge,
            hydraulic_transfer.downward_momentum_transfer,
        )
    )
    _validate_liquid_provenance_areas(
        provenance,
        before_up_area,
        before_down_area,
        tolerance=area_tolerance,
    )

    final_up_marked: list[float] = []
    final_down_marked: list[float] = []
    for cell in range(n):
        before_up = before_up_area[cell]
        before_down = before_down_area[cell]
        marked_up = provenance.upward_source1_area[cell]
        marked_down = provenance.downward_source1_area[cell]
        after_up = final.upward_area[cell]
        after_down = final.downward_area[cell]
        total_area = before_up + before_down
        total_marked = marked_up + marked_down

        if before_up_discharge[cell] < 0.0 and before_down_discharge[cell] > 0.0:
            # The hydraulic operation swapped the complete physical streams,
            # not merely their difference in cross-sectional area.
            new_marked_up = marked_down
            new_marked_down = marked_up
        elif (
            abs(after_up - before_up) <= area_tolerance
            and abs(after_down - before_down) <= area_tolerance
        ):
            new_marked_up = marked_up
            new_marked_down = marked_down
        elif (
            after_down <= area_tolerance
            and abs(after_up - total_area) <= area_tolerance
        ):
            new_marked_up = total_marked
            new_marked_down = 0.0
        elif (
            after_up <= area_tolerance
            and abs(after_down - total_area) <= area_tolerance
        ):
            new_marked_up = 0.0
            new_marked_down = total_marked
        else:
            delta_up = after_up - before_up
            delta_down = after_down - before_down
            if abs(delta_up + delta_down) > area_tolerance:
                raise StateAdmissibilityError(
                    f"hydraulic topology transfer is not area-conservative in cell {cell}"
                )
            if delta_up > area_tolerance:
                if before_down <= 0.0 or delta_up > before_down + area_tolerance:
                    raise StateAdmissibilityError(
                        f"invalid downward-to-upward topology transfer in cell {cell}"
                    )
                moved_marked = delta_up * marked_down / before_down
                new_marked_up = marked_up + moved_marked
                new_marked_down = marked_down - moved_marked
            elif delta_up < -area_tolerance:
                moved_area = -delta_up
                if before_up <= 0.0 or moved_area > before_up + area_tolerance:
                    raise StateAdmissibilityError(
                        f"invalid upward-to-downward topology transfer in cell {cell}"
                    )
                moved_marked = moved_area * marked_up / before_up
                new_marked_up = marked_up - moved_marked
                new_marked_down = marked_down + moved_marked
            else:
                new_marked_up = marked_up
                new_marked_down = marked_down

        if -area_tolerance <= new_marked_up < 0.0:
            new_marked_up = 0.0
        if -area_tolerance <= new_marked_down < 0.0:
            new_marked_down = 0.0
        if after_up <= area_tolerance and new_marked_up <= area_tolerance:
            new_marked_up = 0.0
        elif new_marked_up > after_up:
            if new_marked_up <= after_up + area_tolerance:
                new_marked_up = after_up
            else:
                raise StateAdmissibilityError(
                    f"upward source-one topology inventory exceeds area in cell {cell}"
                )
        if after_down <= area_tolerance and new_marked_down <= area_tolerance:
            new_marked_down = 0.0
        elif new_marked_down > after_down:
            if new_marked_down <= after_down + area_tolerance:
                new_marked_down = after_down
            else:
                raise StateAdmissibilityError(
                    f"downward source-one topology inventory exceeds area in cell {cell}"
                )
        final_up_marked.append(new_marked_up)
        final_down_marked.append(new_marked_down)

    final_provenance = VerticalTwoStreamLiquidProvenanceState.from_iterables(
        upward_source1_area=final_up_marked,
        downward_source1_area=final_down_marked,
    )
    _validate_liquid_provenance_state(
        final_provenance,
        final,
        tolerance=area_tolerance,
    )
    upward_transfer = tuple(
        after - before
        for after, before in zip(
            final_provenance.upward_source1_area,
            provenance.upward_source1_area,
        )
    )
    downward_transfer = tuple(
        after - before
        for after, before in zip(
            final_provenance.downward_source1_area,
            provenance.downward_source1_area,
        )
    )
    return LiquidProvenanceTopologyTransferResult(
        state=final_provenance,
        upward_source1_area_transfer=upward_transfer,
        downward_source1_area_transfer=downward_transfer,
        source1_area_residual=sum(upward_transfer) + sum(downward_transfer),
    )


def advance_vertical_two_stream_liquid_provenance(
    provenance: VerticalTwoStreamLiquidProvenanceState,
    hydraulic_state: VerticalTwoStreamState,
    hydraulic_step: VerticalTwoStreamStepResult,
    parameters: VerticalTwoStreamParameters,
    *,
    dt: float,
    boundaries: VerticalTwoStreamLiquidProvenanceBoundaries = (
        VerticalTwoStreamLiquidProvenanceBoundaries()
    ),
) -> VerticalTwoStreamLiquidProvenanceStepResult:
    """Advance liquid origin with an already accepted hydraulic FV step.

    This routine never proposes a second flux.  It uses the hydraulic step's
    donor- and receiver-limited face area fluxes, then applies that same
    step's topology map.  Consequently provenance cannot alter the hydraulic
    solution and cannot cross a face that accepted no liquid flux.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("liquid-provenance dt must be finite and positive")
    _validate_state_geometry(hydraulic_state, parameters)
    tolerance = max(
        parameters.dry_area_tolerance,
        parameters.packing_tolerance,
        256.0 * 2.220446049250313e-16 * parameters.full_area,
    )
    _validate_liquid_provenance_state(
        provenance,
        hydraulic_state,
        tolerance=tolerance,
    )
    if hydraulic_step.state != hydraulic_step.topology_transfer.state:
        raise StateAdmissibilityError(
            "hydraulic step and its topology-transfer final states differ"
        )

    n = parameters.cell_count
    face_arrays = (
        hydraulic_step.upward_area_flux,
        hydraulic_step.downward_area_flux,
    )
    if any(len(values) != n + 1 for values in face_arrays):
        raise StateAdmissibilityError(
            "hydraulic step needs N+1 directional area fluxes for provenance"
        )
    if any(value < 0.0 for value in hydraulic_step.upward_area_flux) or any(
        value > 0.0 for value in hydraulic_step.downward_area_flux
    ):
        raise StateAdmissibilityError(
            "hydraulic provenance driver has invalid directional flux signs"
        )

    def fraction(marked: float, area: float) -> float:
        if area <= tolerance:
            return 0.0
        if marked < -tolerance or marked > area + tolerance:
            raise StateAdmissibilityError(
                "liquid-provenance donor inventory lies outside its liquid area"
            )
        value = marked / area
        return min(max(value, 0.0), 1.0)

    upward_fraction = tuple(
        fraction(marked, area)
        for marked, area in zip(
            provenance.upward_source1_area,
            hydraulic_state.upward_area,
        )
    )
    downward_fraction = tuple(
        fraction(marked, area)
        for marked, area in zip(
            provenance.downward_source1_area,
            hydraulic_state.downward_area,
        )
    )
    upward_marker_flux = [0.0] * (n + 1)
    downward_marker_flux = [0.0] * (n + 1)
    upward_marker_flux[0] = (
        hydraulic_step.upward_area_flux[0]
        * boundaries.bottom_upward_source1_fraction
    )
    for face in range(1, n + 1):
        upward_marker_flux[face] = (
            hydraulic_step.upward_area_flux[face]
            * upward_fraction[face - 1]
        )
    for face in range(0, n):
        downward_marker_flux[face] = (
            hydraulic_step.downward_area_flux[face]
            * downward_fraction[face]
        )
    downward_marker_flux[n] = (
        hydraulic_step.downward_area_flux[n]
        * boundaries.top_downward_source1_fraction
    )

    dt_over_dz = dt / parameters.cell_length
    transported_up = []
    transported_down = []
    for cell in range(n):
        new_up = provenance.upward_source1_area[cell] + dt_over_dz * (
            upward_marker_flux[cell] - upward_marker_flux[cell + 1]
        )
        new_down = provenance.downward_source1_area[cell] + dt_over_dz * (
            downward_marker_flux[cell] - downward_marker_flux[cell + 1]
        )
        if new_up < -tolerance or new_down < -tolerance:
            raise StateAdmissibilityError(
                "hydraulic donor flux did not preserve provenance positivity"
            )
        transported_up.append(0.0 if new_up < 0.0 else new_up)
        transported_down.append(0.0 if new_down < 0.0 else new_down)

    pre_topology_up_area = tuple(
        area - transfer
        for area, transfer in zip(
            hydraulic_step.state.upward_area,
            hydraulic_step.topology_transfer.upward_area_transfer,
        )
    )
    pre_topology_down_area = tuple(
        area - transfer
        for area, transfer in zip(
            hydraulic_step.state.downward_area,
            hydraulic_step.topology_transfer.downward_area_transfer,
        )
    )
    reconstructed_up_area = tuple(
        hydraulic_state.upward_area[cell]
        + dt_over_dz
        * (
            hydraulic_step.upward_area_flux[cell]
            - hydraulic_step.upward_area_flux[cell + 1]
        )
        for cell in range(n)
    )
    reconstructed_down_area = tuple(
        hydraulic_state.downward_area[cell]
        + dt_over_dz
        * (
            hydraulic_step.downward_area_flux[cell]
            - hydraulic_step.downward_area_flux[cell + 1]
        )
        for cell in range(n)
    )
    for direction, reconstructed, stored in (
        ("upward", reconstructed_up_area, pre_topology_up_area),
        ("downward", reconstructed_down_area, pre_topology_down_area),
    ):
        for cell, (expected, actual) in enumerate(zip(reconstructed, stored)):
            if abs(expected - actual) > tolerance:
                raise StateAdmissibilityError(
                    f"hydraulic {direction} step is inconsistent before topology "
                    f"in cell {cell}"
                )

    transported = VerticalTwoStreamLiquidProvenanceState.from_iterables(
        upward_source1_area=transported_up,
        downward_source1_area=transported_down,
    )
    _validate_liquid_provenance_areas(
        transported,
        pre_topology_up_area,
        pre_topology_down_area,
        tolerance=tolerance,
    )
    topology = conservative_liquid_provenance_topology_transfer(
        transported,
        hydraulic_step.topology_transfer,
        area_tolerance=tolerance,
    )

    dz = parameters.cell_length
    initial_up = dz * sum(provenance.upward_source1_area)
    final_up = dz * sum(topology.state.upward_source1_area)
    boundary_up = dt * (upward_marker_flux[0] - upward_marker_flux[n])
    topology_up = dz * sum(topology.upward_source1_area_transfer)
    initial_down = dz * sum(provenance.downward_source1_area)
    final_down = dz * sum(topology.state.downward_source1_area)
    boundary_down = dt * (downward_marker_flux[0] - downward_marker_flux[n])
    topology_down = dz * sum(topology.downward_source1_area_transfer)
    ledger = VerticalTwoStreamLiquidProvenanceLedger(
        initial_upward_source1_volume=initial_up,
        final_upward_source1_volume=final_up,
        upward_boundary_source1_volume_change=boundary_up,
        upward_topology_source1_volume_transfer=topology_up,
        upward_source1_volume_residual=(
            final_up - initial_up - boundary_up - topology_up
        ),
        initial_downward_source1_volume=initial_down,
        final_downward_source1_volume=final_down,
        downward_boundary_source1_volume_change=boundary_down,
        downward_topology_source1_volume_transfer=topology_down,
        downward_source1_volume_residual=(
            final_down - initial_down - boundary_down - topology_down
        ),
    )
    return VerticalTwoStreamLiquidProvenanceStepResult(
        state=topology.state,
        upward_source1_area_flux=tuple(upward_marker_flux),
        downward_source1_area_flux=tuple(downward_marker_flux),
        topology_transfer=topology,
        ledger=ledger,
    )


__all__ = [
    "COMPLETE_CASEA_RISER_READY",
    "DirectionalBoundaryFlux",
    "bottom_net_receiving_rate_capacity",
    "DirectionalTopologyTransferResult",
    "DirectionalTopologyError",
    "GasMomentumCoupling",
    "MISSING_PHYSICAL_CLOSURES",
    "LiquidProvenanceTopologyTransferResult",
    "PackingViolationError",
    "PhysicalGasInterphaseState",
    "PhysicalThreeBodyDragResult",
    "StateAdmissibilityError",
    "TWOSTREAM_FV_CORE_READY",
    "TaylorBreakthroughMappingResult",
    "VerticalTwoStreamBoundaries",
    "VerticalTwoStreamError",
    "VerticalTwoStreamLedger",
    "VerticalTwoStreamLiquidProvenanceBoundaries",
    "VerticalTwoStreamLiquidProvenanceLedger",
    "VerticalTwoStreamLiquidProvenanceState",
    "VerticalTwoStreamLiquidProvenanceStepResult",
    "VerticalTwoStreamParameters",
    "VerticalTwoStreamState",
    "VerticalTwoStreamStepResult",
    "advance_vertical_two_stream_fv",
    "advance_vertical_two_stream_liquid_provenance",
    "conservative_liquid_provenance_topology_transfer",
    "conservative_directional_topology_transfer",
    "hydrostatic_face_pressures",
    "implicit_physical_three_body_drag_exchange",
    "map_taylor_breakthrough_to_twostream",
]
