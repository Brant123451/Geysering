"""Single-packet state changes for one atomic full-network transaction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

from .errors import ContractViolation
from .state import CoupledState, Vector


def _delta_vector(values: Iterable[float], name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ContractViolation(f"{name} must be a non-empty finite vector")
    return result


@dataclass(frozen=True, slots=True)
class HorizontalDelta:
    Al: Vector
    Ql: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Al", "Ql", "Mg", "Jg"):
            object.__setattr__(self, name, _delta_vector(getattr(self, name), name))
        if len({len(self.Al), len(self.Ql), len(self.Mg), len(self.Jg)}) != 1:
            raise ContractViolation("horizontal delta fields must have identical lengths")

    @classmethod
    def zeros(cls, cell_count: int) -> "HorizontalDelta":
        zero = (0.0,) * cell_count
        return cls(zero, zero, zero, zero)


@dataclass(frozen=True, slots=True)
class SupplyBranchDelta:
    """Change in persistent air-supply branch ``(Al, Ql, Mg, Jg)``."""

    Al: Vector
    Ql: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Al", "Ql", "Mg", "Jg"):
            object.__setattr__(self, name, _delta_vector(getattr(self, name), name))
        if len({len(self.Al), len(self.Ql), len(self.Mg), len(self.Jg)}) != 1:
            raise ContractViolation("supply-branch delta fields must have identical lengths")

    @classmethod
    def zeros(cls, cell_count: int) -> "SupplyBranchDelta":
        zero = (0.0,) * cell_count
        return cls(zero, zero, zero, zero)


@dataclass(frozen=True, slots=True)
class VerticalDelta:
    Aup: Vector
    Qup: Vector
    Adown: Vector
    Qdown: Vector
    Mg: Vector
    Jg: Vector

    def __post_init__(self) -> None:
        for name in ("Aup", "Qup", "Adown", "Qdown", "Mg", "Jg"):
            object.__setattr__(self, name, _delta_vector(getattr(self, name), name))
        if len(
            {len(self.Aup), len(self.Qup), len(self.Adown), len(self.Qdown), len(self.Mg), len(self.Jg)}
        ) != 1:
            raise ContractViolation("vertical delta fields must have identical lengths")

    @classmethod
    def zeros(cls, cell_count: int) -> "VerticalDelta":
        zero = (0.0,) * cell_count
        return cls(zero, zero, zero, zero, zero, zero)


@dataclass(frozen=True, slots=True)
class ExteriorPlumeDelta:
    """Rate or committed change of the persistent exterior liquid state."""

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
                raise ContractViolation(f"exterior plume delta {name} must be finite")
            object.__setattr__(self, name, value)

    @classmethod
    def zeros(cls) -> "ExteriorPlumeDelta":
        return cls()


@dataclass(frozen=True, slots=True)
class TNodeDelta:
    """Compatibility delta for a zero-storage node; valid packets keep it zero."""

    liquid_volume: float = 0.0
    gas_mass: float = 0.0
    liquid_momentum: float = 0.0
    gas_momentum: float = 0.0

    def __post_init__(self) -> None:
        for name in ("liquid_volume", "gas_mass", "liquid_momentum", "gas_momentum"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"T-node delta {name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def is_zero(self) -> bool:
        return all(
            getattr(self, name) == 0.0
            for name in ("liquid_volume", "gas_mass", "liquid_momentum", "gas_momentum")
        )


@dataclass(frozen=True, slots=True)
class TNodePortResidual:
    """Signed sum of all oriented port rates at one algebraic T node.

    A zero-storage node has no time derivative with which to hide a mismatch.
    Consequently all four residuals are mandatory packet data and are checked
    before any proposed state or ledger entry can be committed.
    """

    liquid_volume_m3_s: float = 0.0
    gas_mass_kg_s: float = 0.0
    mixture_momentum_x_N: float = 0.0
    mixture_momentum_z_N: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "liquid_volume_m3_s",
            "gas_mass_kg_s",
            "mixture_momentum_x_N",
            "mixture_momentum_z_N",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"T-node port residual {name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class BoundaryExchange:
    """External material rates and Cartesian momentum sources.

    Boundary momentum fluxes are non-negative magnitudes assigned to an
    explicit Cartesian in/out direction.  External forces are signed.  Their
    time integrals are recorded separately by the conservation ledger.
    Internal T-node exchange never appears here; it must cancel in each node's
    :class:`TNodePortResidual`.

    The legacy scalar momentum fields remain zero-only constructor shims.  A
    non-zero scalar has no defensible mapping to x or z and is rejected rather
    than silently added to both axes or to an unphysical scalar sum.
    """

    liquid_inflow_m3_s: float = 0.0
    liquid_outflow_m3_s: float = 0.0
    gas_inflow_kg_s: float = 0.0
    gas_outflow_kg_s: float = 0.0
    momentum_x_in_N: float = 0.0
    momentum_x_out_N: float = 0.0
    momentum_z_in_N: float = 0.0
    momentum_z_out_N: float = 0.0
    external_force_x_N: float = 0.0
    external_force_z_N: float = 0.0
    mixture_momentum_in_N: float = 0.0
    mixture_momentum_out_N: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "liquid_inflow_m3_s",
            "liquid_outflow_m3_s",
            "gas_inflow_kg_s",
            "gas_outflow_kg_s",
            "momentum_x_in_N",
            "momentum_x_out_N",
            "momentum_z_in_N",
            "momentum_z_out_N",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"boundary exchange {name} must be finite")
            if value < 0.0:
                raise ContractViolation(f"boundary exchange {name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("external_force_x_N", "external_force_z_N"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"boundary exchange {name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("mixture_momentum_in_N", "mixture_momentum_out_N"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractViolation(f"legacy boundary exchange {name} must be non-negative")
            if value != 0.0:
                raise ContractViolation(
                    "legacy scalar momentum exchange is ambiguous; use explicit x/z fields"
                )
            object.__setattr__(self, name, value)

    @property
    def liquid_volume_net_rate(self) -> float:
        return self.liquid_inflow_m3_s - self.liquid_outflow_m3_s

    @property
    def gas_mass_net_rate(self) -> float:
        return self.gas_inflow_kg_s - self.gas_outflow_kg_s

    @property
    def mixture_momentum_x_boundary_rate(self) -> float:
        return self.momentum_x_in_N - self.momentum_x_out_N

    @property
    def mixture_momentum_z_boundary_rate(self) -> float:
        return self.momentum_z_in_N - self.momentum_z_out_N

    @property
    def mixture_momentum_x_net_rate(self) -> float:
        return self.mixture_momentum_x_boundary_rate + self.external_force_x_N

    @property
    def mixture_momentum_z_net_rate(self) -> float:
        return self.mixture_momentum_z_boundary_rate + self.external_force_z_N

    @property
    def mixture_momentum_net_rate(self) -> float:
        """Deprecated one-axis-only compatibility view.

        Returning x+z would recreate the dimensional error this contract is
        intended to remove.  The scalar view is therefore available only when
        at most one component is non-zero.
        """

        x_rate = self.mixture_momentum_x_net_rate
        z_rate = self.mixture_momentum_z_net_rate
        if x_rate != 0.0 and z_rate != 0.0:
            raise ContractViolation(
                "mixture momentum has both x and z components; use the vector properties"
            )
        return x_rate if x_rate != 0.0 else z_rate


def state_token(state: CoupledState) -> str:
    """Return a deterministic stale-packet guard for an immutable state."""

    payload = repr(state).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AtomicFluxPacket:
    """The only legal unit of full supply/main/two-node/riser advancement."""

    transaction_id: str
    base_state_token: str
    dt_s: float
    horizontal: HorizontalDelta
    vertical: VerticalDelta
    supply_branch: SupplyBranchDelta = SupplyBranchDelta.zeros(1)
    exterior_plume: ExteriorPlumeDelta = ExteriorPlumeDelta()
    air_supply_node: TNodeDelta = TNodeDelta()
    riser_node: TNodeDelta = TNodeDelta()
    air_supply_node_ports: TNodePortResidual = TNodePortResidual()
    riser_node_ports: TNodePortResidual = TNodePortResidual()
    boundary: BoundaryExchange = BoundaryExchange()

    def __post_init__(self) -> None:
        if not self.transaction_id.strip():
            raise ContractViolation("transaction_id must be non-empty")
        if not self.base_state_token.strip():
            raise ContractViolation("base_state_token must be non-empty")
        value = float(self.dt_s)
        if not math.isfinite(value) or value <= 0.0:
            raise ContractViolation("packet dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", value)

    @classmethod
    def zero(
        cls, state: CoupledState, dt_s: float, transaction_id: str = "zero-flux"
    ) -> "AtomicFluxPacket":
        return cls(
            transaction_id=transaction_id,
            base_state_token=state_token(state),
            dt_s=dt_s,
            horizontal=HorizontalDelta.zeros(state.horizontal.cell_count),
            vertical=VerticalDelta.zeros(state.vertical.cell_count),
            supply_branch=SupplyBranchDelta.zeros(state.supply_branch.cell_count),
            exterior_plume=ExteriorPlumeDelta.zeros(),
        )
