"""Explicit material and Cartesian mixture-momentum conservation ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .errors import ConservationError
from .flux import BoundaryExchange
from .state import CoupledGeometry, CoupledState


@dataclass(frozen=True, slots=True)
class ConservationSnapshot:
    liquid_volume_m3: float
    gas_mass_kg: float
    mixture_momentum_x_kg_m_s: float
    mixture_momentum_z_kg_m_s: float

    @property
    def mixture_axial_momentum_kg_m_s(self) -> float:
        """Deprecated scalar compatibility diagnostic (vector magnitude).

        This property is deliberately not used by the ledger.  In particular,
        it does not add orthogonal x and z components.
        """

        return math.hypot(
            self.mixture_momentum_x_kg_m_s,
            self.mixture_momentum_z_kg_m_s,
        )

    @classmethod
    def from_state(
        cls, state: CoupledState, geometry: CoupledGeometry
    ) -> "ConservationSnapshot":
        geometry.validate_state(state)
        rho_l = geometry.liquid_density_kg_m3
        horizontal_liquid = sum(
            area * dx
            for area, dx in zip(state.horizontal.Al, geometry.horizontal_dx_m, strict=True)
        )
        vertical_liquid = sum(
            (up + down) * dz
            for up, down, dz in zip(
                state.vertical.Aup,
                state.vertical.Adown,
                geometry.vertical_dz_m,
                strict=True,
            )
        )
        supply_liquid = sum(
            area * dz
            for area, dz in zip(
                state.supply_branch.Al,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        )
        horizontal_gas = sum(
            mass * dx
            for mass, dx in zip(state.horizontal.Mg, geometry.horizontal_dx_m, strict=True)
        )
        vertical_gas = sum(
            mass * dz
            for mass, dz in zip(state.vertical.Mg, geometry.vertical_dz_m, strict=True)
        )
        supply_gas = sum(
            mass * dz
            for mass, dz in zip(
                state.supply_branch.Mg,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        )
        horizontal_momentum = sum(
            (rho_l * discharge + gas_momentum) * dx
            for discharge, gas_momentum, dx in zip(
                state.horizontal.Ql,
                state.horizontal.Jg,
                geometry.horizontal_dx_m,
                strict=True,
            )
        )
        vertical_momentum = sum(
            (rho_l * (up - down) + gas_momentum) * dz
            for up, down, gas_momentum, dz in zip(
                state.vertical.Qup,
                state.vertical.Qdown,
                state.vertical.Jg,
                geometry.vertical_dz_m,
                strict=True,
            )
        )
        supply_momentum = sum(
            (rho_l * discharge + gas_momentum) * dz
            for discharge, gas_momentum, dz in zip(
                state.supply_branch.Ql,
                state.supply_branch.Jg,
                geometry.supply_branch_dz_m,
                strict=True,
            )
        )
        exterior_liquid = state.exterior_plume.liquid_volume_m3
        exterior_momentum = state.exterior_plume.vertical_momentum_kg_m_s
        return cls(
            liquid_volume_m3=(
                horizontal_liquid
                + vertical_liquid
                + supply_liquid
                + exterior_liquid
            ),
            gas_mass_kg=horizontal_gas + vertical_gas + supply_gas,
            mixture_momentum_x_kg_m_s=horizontal_momentum,
            mixture_momentum_z_kg_m_s=(
                vertical_momentum + supply_momentum + exterior_momentum
            ),
        )


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    transaction_id: str
    time_start_s: float
    time_end_s: float
    before: ConservationSnapshot
    after: ConservationSnapshot
    boundary: BoundaryExchange
    boundary_momentum_x_impulse_kg_m_s: float
    boundary_momentum_z_impulse_kg_m_s: float
    external_force_x_impulse_kg_m_s: float
    external_force_z_impulse_kg_m_s: float
    liquid_volume_residual_m3: float
    gas_mass_residual_kg: float
    mixture_momentum_x_residual_kg_m_s: float
    mixture_momentum_z_residual_kg_m_s: float

    @property
    def mixture_momentum_residual_kg_m_s(self) -> float:
        """Deprecated scalar compatibility diagnostic (residual norm)."""

        return math.hypot(
            self.mixture_momentum_x_residual_kg_m_s,
            self.mixture_momentum_z_residual_kg_m_s,
        )


@dataclass(slots=True)
class ConservationLedger:
    """Append-only ledger; failed transactions are never appended."""

    absolute_tolerance: float = 1.0e-12
    relative_tolerance: float = 1.0e-10
    entries: list[LedgerEntry] = field(default_factory=list)

    def _accepted(self, residual: float, before: float, after: float, expected: float) -> bool:
        scale = max(abs(before), abs(after), abs(expected), 1.0)
        return abs(residual) <= self.absolute_tolerance + self.relative_tolerance * scale

    def evaluate(
        self,
        transaction_id: str,
        before_state: CoupledState,
        after_state: CoupledState,
        geometry: CoupledGeometry,
        dt_s: float,
        boundary: BoundaryExchange,
    ) -> LedgerEntry:
        before = ConservationSnapshot.from_state(before_state, geometry)
        after = ConservationSnapshot.from_state(after_state, geometry)
        expected_liquid = dt_s * boundary.liquid_volume_net_rate
        expected_gas = dt_s * boundary.gas_mass_net_rate
        boundary_x_impulse = dt_s * boundary.mixture_momentum_x_boundary_rate
        boundary_z_impulse = dt_s * boundary.mixture_momentum_z_boundary_rate
        external_x_impulse = dt_s * boundary.external_force_x_N
        external_z_impulse = dt_s * boundary.external_force_z_N
        expected_momentum_x = boundary_x_impulse + external_x_impulse
        expected_momentum_z = boundary_z_impulse + external_z_impulse
        liquid_residual = after.liquid_volume_m3 - before.liquid_volume_m3 - expected_liquid
        gas_residual = after.gas_mass_kg - before.gas_mass_kg - expected_gas
        momentum_x_residual = (
            after.mixture_momentum_x_kg_m_s
            - before.mixture_momentum_x_kg_m_s
            - expected_momentum_x
        )
        momentum_z_residual = (
            after.mixture_momentum_z_kg_m_s
            - before.mixture_momentum_z_kg_m_s
            - expected_momentum_z
        )
        checks = (
            self._accepted(
                liquid_residual, before.liquid_volume_m3, after.liquid_volume_m3, expected_liquid
            ),
            self._accepted(gas_residual, before.gas_mass_kg, after.gas_mass_kg, expected_gas),
            self._accepted(
                momentum_x_residual,
                before.mixture_momentum_x_kg_m_s,
                after.mixture_momentum_x_kg_m_s,
                expected_momentum_x,
            ),
            self._accepted(
                momentum_z_residual,
                before.mixture_momentum_z_kg_m_s,
                after.mixture_momentum_z_kg_m_s,
                expected_momentum_z,
            ),
        )
        if not all(checks) or not all(
            math.isfinite(value)
            for value in (
                liquid_residual,
                gas_residual,
                momentum_x_residual,
                momentum_z_residual,
            )
        ):
            raise ConservationError(
                "atomic packet failed conservation: "
                f"liquid={liquid_residual:.6e} m3, gas={gas_residual:.6e} kg, "
                f"Px={momentum_x_residual:.6e} kg m/s, "
                f"Pz={momentum_z_residual:.6e} kg m/s"
            )
        return LedgerEntry(
            transaction_id=transaction_id,
            time_start_s=before_state.time_s,
            time_end_s=after_state.time_s,
            before=before,
            after=after,
            boundary=boundary,
            boundary_momentum_x_impulse_kg_m_s=boundary_x_impulse,
            boundary_momentum_z_impulse_kg_m_s=boundary_z_impulse,
            external_force_x_impulse_kg_m_s=external_x_impulse,
            external_force_z_impulse_kg_m_s=external_z_impulse,
            liquid_volume_residual_m3=liquid_residual,
            gas_mass_residual_kg=gas_residual,
            mixture_momentum_x_residual_kg_m_s=momentum_x_residual,
            mixture_momentum_z_residual_kg_m_s=momentum_z_residual,
        )

    def append(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
