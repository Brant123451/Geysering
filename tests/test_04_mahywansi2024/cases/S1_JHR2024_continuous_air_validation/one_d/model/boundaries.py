"""Continuous gas-source boundary contract.

The paper gives a 5700 Pa gauge pressure but not the full valve/line/tank loss
law required to convert it to gas mass and momentum flux.  The unresolved
implementation therefore fails closed instead of inventing a flow rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from .errors import ContractViolation, MissingPhysicalClosure


@dataclass(frozen=True, slots=True)
class GasSourceContext:
    time_s: float
    dt_s: float
    local_absolute_pressure_Pa: float
    local_gas_density_kg_m3: float

    def __post_init__(self) -> None:
        for name in (
            "time_s",
            "dt_s",
            "local_absolute_pressure_Pa",
            "local_gas_density_kg_m3",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContractViolation(f"gas-source context {name} must be finite")
            object.__setattr__(self, name, value)
        if self.time_s < 0.0 or self.dt_s <= 0.0:
            raise ContractViolation("gas-source time must be non-negative and dt positive")
        if self.local_absolute_pressure_Pa <= 0.0 or self.local_gas_density_kg_m3 <= 0.0:
            raise ContractViolation("gas-source pressure and density must be positive")


@dataclass(frozen=True, slots=True)
class GasSourceFlux:
    mass_inflow_kg_s: float
    momentum_in_N: float

    def __post_init__(self) -> None:
        for name in ("mass_inflow_kg_s", "momentum_in_N"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractViolation(f"gas-source {name} must be finite and non-negative")
            object.__setattr__(self, name, value)


class ContinuousGasSourceBoundary(ABC):
    """Interface for a pressure-driven source evaluated every time step."""

    @abstractmethod
    def evaluate(self, context: GasSourceContext) -> GasSourceFlux:
        """Return external gas mass/momentum rates for the atomic packet."""


@dataclass(frozen=True, slots=True)
class UnresolvedPressureGasSource(ContinuousGasSourceBoundary):
    """Published pressure boundary with an intentionally absent flow closure."""

    source_gauge_pressure_Pa: float = 5700.0
    evidence_status: str = "published_pressure__missing_valve_line_flow_closure"

    def evaluate(self, context: GasSourceContext) -> GasSourceFlux:
        del context
        raise MissingPhysicalClosure(
            "5700 Pa is a pressure boundary, not a gas mass-flow prescription; "
            "the valve/line/tank closure is not yet evidence-backed"
        )
