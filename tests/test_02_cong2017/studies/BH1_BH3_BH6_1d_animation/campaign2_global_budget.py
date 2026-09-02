"""Global phase-inventory ledger for the persistent Campaign-2 1D model."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Campaign2GlobalBudget:
    initial_liquid_volume_m3: float
    initial_gas_mass_kg: float
    reservoir_liquid_inflow_m3: float = 0.0
    top_liquid_outflow_m3: float = 0.0
    top_gas_outflow_kg: float = 0.0
    internal_tee_liquid_to_riser_m3: float = 0.0
    internal_tee_gas_to_riser_kg: float = 0.0
    numerical_liquid_correction_m3: float = 0.0
    numerical_gas_correction_kg: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.initial_liquid_volume_m3))
            or self.initial_liquid_volume_m3 < 0.0
        ):
            raise ValueError("initial liquid volume must be finite and non-negative")
        if (
            not math.isfinite(float(self.initial_gas_mass_kg))
            or self.initial_gas_mass_kg < 0.0
        ):
            raise ValueError("initial gas mass must be finite and non-negative")

    @staticmethod
    def _finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def book_reservoir_liquid(self, volume_into_domain_m3: float) -> None:
        self.reservoir_liquid_inflow_m3 += self._finite(
            volume_into_domain_m3,
            "reservoir liquid volume",
        )

    def book_top_liquid(self, volume_out_m3: float) -> None:
        volume = self._finite(volume_out_m3, "top liquid volume")
        if volume < 0.0:
            raise ValueError("the atmosphere cannot donate liquid at the riser top")
        self.top_liquid_outflow_m3 += volume

    def book_top_gas(self, mass_out_kg: float) -> None:
        # Signed: a negative value is atmospheric gas entering a draining riser.
        self.top_gas_outflow_kg += self._finite(mass_out_kg, "top gas mass")

    def book_internal_tee(self, *, liquid_to_riser_m3: float, gas_to_riser_kg: float) -> None:
        self.internal_tee_liquid_to_riser_m3 += self._finite(
            liquid_to_riser_m3,
            "internal T liquid volume",
        )
        self.internal_tee_gas_to_riser_kg += self._finite(
            gas_to_riser_kg,
            "internal T gas mass",
        )

    def liquid_residual_m3(
        self,
        *,
        final_horizontal_volume_m3: float,
        final_vertical_volume_m3: float,
    ) -> float:
        actual = self._finite(
            final_horizontal_volume_m3,
            "final horizontal liquid volume",
        ) + self._finite(
            final_vertical_volume_m3,
            "final vertical liquid volume",
        )
        expected = (
            self.initial_liquid_volume_m3
            + self.reservoir_liquid_inflow_m3
            - self.top_liquid_outflow_m3
            + self.numerical_liquid_correction_m3
        )
        return float(actual - expected)

    def gas_residual_kg(
        self,
        *,
        final_horizontal_mass_kg: float,
        final_vertical_mass_kg: float,
    ) -> float:
        actual = self._finite(
            final_horizontal_mass_kg,
            "final horizontal gas mass",
        ) + self._finite(
            final_vertical_mass_kg,
            "final vertical gas mass",
        )
        expected = (
            self.initial_gas_mass_kg
            - self.top_gas_outflow_kg
            + self.numerical_gas_correction_kg
        )
        return float(actual - expected)

    def audit(
        self,
        *,
        final_horizontal_liquid_m3: float,
        final_vertical_liquid_m3: float,
        final_horizontal_gas_kg: float,
        final_vertical_gas_kg: float,
    ) -> dict[str, float]:
        liquid_residual = self.liquid_residual_m3(
            final_horizontal_volume_m3=final_horizontal_liquid_m3,
            final_vertical_volume_m3=final_vertical_liquid_m3,
        )
        gas_residual = self.gas_residual_kg(
            final_horizontal_mass_kg=final_horizontal_gas_kg,
            final_vertical_mass_kg=final_vertical_gas_kg,
        )
        return {
            "liquid_residual_m3": liquid_residual,
            "gas_residual_kg": gas_residual,
            "liquid_relative_residual": liquid_residual
            / max(self.initial_liquid_volume_m3, 1.0e-30),
            "gas_relative_residual": gas_residual
            / max(self.initial_gas_mass_kg, 1.0e-30),
            "numerical_liquid_correction_m3": float(
                self.numerical_liquid_correction_m3
            ),
            "numerical_gas_correction_kg": float(
                self.numerical_gas_correction_kg
            ),
        }


__all__ = ["Campaign2GlobalBudget"]
