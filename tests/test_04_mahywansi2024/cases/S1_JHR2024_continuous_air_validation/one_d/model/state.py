"""Immutable conservative state owned by the coupled 1-D network.

The cell states are axial finite-volume averages.  ``Mg`` is gas mass per
axial length [kg/m], and ``Jg`` is signed gas momentum per axial length
[kg/s].  Liquid discharges have units [m3/s].  In the riser, ``Qup`` and
``Qdown`` are non-negative gross directional discharges.  They are persistent
degrees of freedom; the model never reconstructs them from their difference.

The air-supply branch is a real, finite branch, not a pressure condition placed
directly on the horizontal main.  It therefore owns the same conservative
variables ``(Al, Ql, Mg, Jg)`` on its own vertical grid.  The two T nodes are
zero-storage algebraic junctions: they own port balances, but no fictitious
material volume or momentum inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .errors import ContractViolation


Vector = tuple[float, ...]


# The Case-1 horizontal FV owner uses this as its conservative vacuum-
# momentum admissibility band.  The coupled geometry gate must not apply a
# stricter *exact-zero* test to the very same immutable state: RK face-force
# cancellation can leave paired values of order 1e-34--1e-43 kg/s while gas
# mass is exactly zero.  This is only an admissibility threshold; neither the
# state nor its ledger is projected or clipped here.
_HORIZONTAL_VACUUM_MOMENTUM_TOLERANCE_KG_S = 1.0e-10


def _finite_vector(values: Iterable[float], name: str) -> Vector:
    try:
        result = tuple(float(value) for value in values)
    except TypeError as exc:
        raise ContractViolation(f"{name} must be an iterable of real values") from exc
    if not result:
        raise ContractViolation(f"{name} must contain at least one cell")
    if not all(math.isfinite(value) for value in result):
        raise ContractViolation(f"{name} contains a non-finite value")
    return result


def _same_length(name: str, *vectors: Vector) -> None:
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise ContractViolation(f"{name} fields must have identical cell counts")


def _nonnegative(vector: Vector, name: str) -> None:
    if any(value < 0.0 for value in vector):
        raise ContractViolation(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class HorizontalState:
    """Horizontal conservative variables ``(Al, Ql, Mg, Jg)``."""

    Al: Vector
    Ql: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Al", "Ql", "Mg", "Jg"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        _same_length("horizontal", self.Al, self.Ql, self.Mg, self.Jg)
        _nonnegative(self.Al, "horizontal Al")
        _nonnegative(self.Mg, "horizontal Mg")

    @property
    def cell_count(self) -> int:
        return len(self.Al)


@dataclass(frozen=True, slots=True)
class SupplyBranchState:
    """Persistent vertical air-supply branch state ``(Al, Ql, Mg, Jg)``.

    ``Mg`` and ``Jg`` are gas mass and signed vertical gas momentum per unit
    branch length.  ``Ql`` is signed positive upward.  Gas cross-sectional
    area is not an independent state: it is the complement of ``Al`` in the
    branch area and is checked by :class:`CoupledGeometry`.
    """

    Al: Vector
    Ql: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Al", "Ql", "Mg", "Jg"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        _same_length("supply branch", self.Al, self.Ql, self.Mg, self.Jg)
        _nonnegative(self.Al, "supply-branch Al")
        _nonnegative(self.Mg, "supply-branch Mg")

    @property
    def cell_count(self) -> int:
        return len(self.Al)

@dataclass(frozen=True, slots=True)
class VerticalState:
    """Riser state with persistent up/down liquid streams and resolved gas."""

    Aup: Vector
    Qup: Vector
    Adown: Vector
    Qdown: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Aup", "Qup", "Adown", "Qdown", "Mg", "Jg"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        _same_length(
            "vertical", self.Aup, self.Qup, self.Adown, self.Qdown, self.Mg, self.Jg
        )
        _nonnegative(self.Aup, "vertical Aup")
        _nonnegative(self.Qup, "vertical Qup")
        _nonnegative(self.Adown, "vertical Adown")
        _nonnegative(self.Qdown, "vertical Qdown")
        _nonnegative(self.Mg, "vertical Mg")

    @property
    def cell_count(self) -> int:
        return len(self.Aup)

    @property
    def net_liquid_discharge(self) -> Vector:
        """Diagnostic only; never used to reconstruct the gross streams."""

        return tuple(up - down for up, down in zip(self.Qup, self.Qdown, strict=True))


@dataclass(frozen=True, slots=True)
class TNodeState:
    """Compatibility shell for a zero-storage algebraic T junction.

    The fields remain so stale callers fail with an informative contract error
    rather than an unexpected constructor error.  A valid coupled state must
    have all four fields exactly zero.  Port conservation belongs to the
    atomic flux packet, not to a fabricated node control volume.
    """

    liquid_volume: float = 0.0
    gas_mass: float = 0.0
    liquid_momentum: float = 0.0
    gas_momentum: float = 0.0

    def __post_init__(self) -> None:
        for name in ("liquid_volume", "gas_mass", "liquid_momentum", "gas_momentum"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"T-node {name} must be finite")
            object.__setattr__(self, name, value)
        if any(
            getattr(self, name) != 0.0
            for name in ("liquid_volume", "gas_mass", "liquid_momentum", "gas_momentum")
        ):
            raise ContractViolation(
                "T-node is a zero-storage algebraic junction; all inventory fields must be zero"
            )

    @property
    def is_zero_storage(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ExteriorPlumeState:
    """Persistent reduced-order liquid inventory above the atmospheric rim.

    Airborne liquid and the finite returning queue at the rim remain separate
    gross populations, so simultaneous outflow and re-entry cannot disappear
    into one net scalar.  ``airborne_liquid_first_moment_m4`` is retained only
    to decide when a falling lump reaches the mouth; any reconstructed height
    is a declared reduced-order/derived proxy, not a simulated external
    free-surface elevation.
    """

    airborne_liquid_volume_m3: float = 0.0
    airborne_vertical_momentum_kg_m_s: float = 0.0
    airborne_liquid_first_moment_m4: float = 0.0
    returning_liquid_volume_m3: float = 0.0
    returning_downward_momentum_kg_m_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "airborne_liquid_volume_m3",
            "airborne_vertical_momentum_kg_m_s",
            "airborne_liquid_first_moment_m4",
            "returning_liquid_volume_m3",
            "returning_downward_momentum_kg_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"exterior plume {name} must be finite")
            object.__setattr__(self, name, value)
        if self.airborne_liquid_volume_m3 < 0.0:
            raise ContractViolation("airborne plume volume must be non-negative")
        if self.returning_liquid_volume_m3 < 0.0:
            raise ContractViolation("returning plume volume must be non-negative")
        if self.returning_downward_momentum_kg_m_s < 0.0:
            raise ContractViolation("returning plume momentum must be non-negative")
        if self.airborne_liquid_first_moment_m4 < 0.0:
            raise ContractViolation(
                "airborne plume liquid first moment must be non-negative"
            )
        if self.airborne_liquid_volume_m3 == 0.0 and (
            self.airborne_vertical_momentum_kg_m_s != 0.0
            or self.airborne_liquid_first_moment_m4 != 0.0
        ):
            raise ContractViolation(
                "empty airborne plume cannot retain momentum or height moment"
            )
        if self.returning_liquid_volume_m3 == 0.0 and (
            self.returning_downward_momentum_kg_m_s != 0.0
        ):
            raise ContractViolation(
                "empty returning plume cannot retain downward momentum"
            )

    @property
    def liquid_volume_m3(self) -> float:
        return self.airborne_liquid_volume_m3 + self.returning_liquid_volume_m3

    @property
    def vertical_momentum_kg_m_s(self) -> float:
        return (
            self.airborne_vertical_momentum_kg_m_s
            - self.returning_downward_momentum_kg_m_s
        )

    @property
    def liquid_first_moment_m4(self) -> float:
        return self.airborne_liquid_first_moment_m4

    @property
    def derived_centroid_height_proxy_m(self) -> float:
        if self.airborne_liquid_volume_m3 == 0.0:
            return 0.0
        return (
            self.airborne_liquid_first_moment_m4
            / self.airborne_liquid_volume_m3
        )

    @property
    def height_evidence_status(self) -> str:
        return "declared_reduced_order_derived_proxy__not_external_free_surface"


@dataclass(frozen=True, slots=True)
class CoupledState:
    """One immutable snapshot of all state owned by the 1-D network."""

    time_s: float
    horizontal: HorizontalState
    vertical: VerticalState
    supply_branch: SupplyBranchState
    exterior_plume: ExteriorPlumeState = ExteriorPlumeState()
    air_supply_node: TNodeState = TNodeState()
    riser_node: TNodeState = TNodeState()

    def __post_init__(self) -> None:
        value = float(self.time_s)
        if not math.isfinite(value) or value < 0.0:
            raise ContractViolation("time_s must be finite and non-negative")
        object.__setattr__(self, "time_s", value)


@dataclass(frozen=True, slots=True)
class CoupledGeometry:
    """Cell lengths and pipe areas needed to turn state into inventories."""

    horizontal_dx_m: Vector
    vertical_dz_m: Vector
    horizontal_area_m2: float
    vertical_area_m2: float
    liquid_density_kg_m3: float = 998.2
    supply_branch_dz_m: Vector = (0.1373,)
    supply_branch_area_m2: float | None = None
    horizontal_elastic_overarea_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizontal_dx_m", _finite_vector(self.horizontal_dx_m, "horizontal_dx_m")
        )
        object.__setattr__(
            self, "vertical_dz_m", _finite_vector(self.vertical_dz_m, "vertical_dz_m")
        )
        object.__setattr__(
            self,
            "supply_branch_dz_m",
            _finite_vector(self.supply_branch_dz_m, "supply_branch_dz_m"),
        )
        for name in ("horizontal_area_m2", "vertical_area_m2", "liquid_density_kg_m3"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractViolation(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        supply_area = (
            self.horizontal_area_m2
            if self.supply_branch_area_m2 is None
            else float(self.supply_branch_area_m2)
        )
        if not math.isfinite(supply_area) or supply_area <= 0.0:
            raise ContractViolation("supply_branch_area_m2 must be finite and positive")
        object.__setattr__(self, "supply_branch_area_m2", supply_area)
        overarea = float(self.horizontal_elastic_overarea_fraction)
        if not math.isfinite(overarea) or overarea < 0.0:
            raise ContractViolation(
                "horizontal_elastic_overarea_fraction must be a finite non-negative declaration"
            )
        object.__setattr__(self, "horizontal_elastic_overarea_fraction", overarea)
        _nonnegative(self.horizontal_dx_m, "horizontal cell lengths")
        _nonnegative(self.vertical_dz_m, "vertical cell lengths")
        _nonnegative(self.supply_branch_dz_m, "supply-branch cell lengths")
        if any(
            value == 0.0
            for value in (
                self.horizontal_dx_m + self.vertical_dz_m + self.supply_branch_dz_m
            )
        ):
            raise ContractViolation("cell lengths must be strictly positive")

    def validate_state(self, state: CoupledState, tolerance: float = 1.0e-12) -> None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ContractViolation("state validation tolerance must be finite and non-negative")
        if state.horizontal.cell_count != len(self.horizontal_dx_m):
            raise ContractViolation("horizontal state/grid cell counts differ")
        if state.vertical.cell_count != len(self.vertical_dz_m):
            raise ContractViolation("vertical state/grid cell counts differ")
        if state.supply_branch.cell_count != len(self.supply_branch_dz_m):
            raise ContractViolation("supply-branch state/grid cell counts differ")

        horizontal_elastic_limit = self.horizontal_area_m2 * (
            1.0 + self.horizontal_elastic_overarea_fraction
        )
        for index, (liquid_area, gas_mass, gas_momentum) in enumerate(
            zip(
                state.horizontal.Al,
                state.horizontal.Mg,
                state.horizontal.Jg,
                strict=True,
            )
        ):
            if (
                gas_mass == 0.0
                and abs(gas_momentum)
                > _HORIZONTAL_VACUUM_MOMENTUM_TOLERANCE_KG_S
            ):
                raise ContractViolation(
                    f"horizontal cell {index} has gas momentum without gas mass"
                )
            if liquid_area > self.horizontal_area_m2 + tolerance:
                if gas_mass != 0.0:
                    raise ContractViolation(
                        f"horizontal cell {index} contains gas and cannot use elastic overarea"
                    )
                if liquid_area > horizontal_elastic_limit + tolerance:
                    raise ContractViolation(
                        f"horizontal cell {index} exceeds the declared elastic overarea bound"
                    )
            # ``tolerance`` is a packing/zero-inventory roundoff band, not a
            # minimum physical bubble size.  A finite, representable positive
            # complement must remain admissible when it is paired with gas
            # mass in the same atomic proposal.
            if gas_mass > 0.0 and self.horizontal_area_m2 - liquid_area <= 0.0:
                raise ContractViolation(
                    f"horizontal cell {index} has gas mass but no positive complementary gas area"
                )
            if gas_mass == 0.0 and self.horizontal_area_m2 - liquid_area > tolerance:
                raise ContractViolation(
                    f"horizontal cell {index} has positive gas area but zero gas mass"
                )

        for index, (up, down, gas_mass, gas_momentum) in enumerate(
            zip(
                state.vertical.Aup,
                state.vertical.Adown,
                state.vertical.Mg,
                state.vertical.Jg,
                strict=True,
            )
        ):
            liquid_area = up + down
            if liquid_area > self.vertical_area_m2 + tolerance:
                raise ContractViolation("vertical directional liquid areas exceed pipe area")
            if gas_mass == 0.0 and gas_momentum != 0.0:
                raise ContractViolation(f"vertical cell {index} has gas momentum without gas mass")
            if gas_mass > 0.0 and self.vertical_area_m2 - liquid_area <= 0.0:
                raise ContractViolation(
                    f"vertical cell {index} has gas mass but no positive complementary gas area"
                )
            if gas_mass == 0.0 and self.vertical_area_m2 - liquid_area > tolerance:
                raise ContractViolation(
                    f"vertical cell {index} has positive gas area but zero gas mass"
                )

        assert self.supply_branch_area_m2 is not None
        for index, (liquid_area, gas_mass, gas_momentum) in enumerate(
            zip(
                state.supply_branch.Al,
                state.supply_branch.Mg,
                state.supply_branch.Jg,
                strict=True,
            )
        ):
            if liquid_area > self.supply_branch_area_m2 + tolerance:
                raise ContractViolation("supply-branch liquid area exceeds pipe area")
            if gas_mass == 0.0 and gas_momentum != 0.0:
                raise ContractViolation(
                    f"supply-branch cell {index} has gas momentum without gas mass"
                )
            if gas_mass > 0.0 and self.supply_branch_area_m2 - liquid_area <= 0.0:
                raise ContractViolation(
                    f"supply-branch cell {index} has gas mass but no positive complementary gas area"
                )
            if gas_mass == 0.0 and self.supply_branch_area_m2 - liquid_area > tolerance:
                raise ContractViolation(
                    f"supply-branch cell {index} has positive gas area but zero gas mass"
                )

        # The constructors already enforce this, but keeping the production
        # invariant at the geometry gate makes the zero-storage ownership
        # explicit at every atomic commit.
        if not state.air_supply_node.is_zero_storage or not state.riser_node.is_zero_storage:
            raise ContractViolation("T nodes must remain zero-storage algebraic junctions")
