"""Persistent reduced-order liquid state above the atmospheric riser rim.

The exterior reduction owns two gross populations: one airborne lump and one
finite returning queue at the rim.  Their volumes and vertical momenta enter
the whole-network conservation ledger.  The airborne volume first moment is
only a declared kinematic/derived height proxy used to locate the return
event; it is not an external free-surface prediction.

No eruption height, duration or period enters this closure.  The returning
donor uses the stored downward momentum and the physical riser aperture, not
``g*dt`` or a result-fitted velocity.  Falling airborne state is relabelled to
the returning queue by a zero-time conservative preparation before either RK
stage; it is never represented as a ``-V/dt`` ordinary RK rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .errors import ContractViolation
from .flux import BoundaryExchange, ExteriorPlumeDelta
from .state import CoupledGeometry, ExteriorPlumeState
from .vertical_pressure_void_component import (
    AtmosphericLiquidFallback,
    AtmosphericLiquidFlux,
)


F0_EXTERIOR_GRAVITY_M_S2 = 9.81
_MATERIAL_TOLERANCE_M3 = 2.0e-14
_MOMENTUM_TOLERANCE_KG_M_S = 2.0e-11


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ContractViolation(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class ExteriorPlumeStageDiagnostics:
    before: ExteriorPlumeState
    after: ExteriorPlumeState
    top_liquid: AtmosphericLiquidFlux
    liquid_port_momentum_rate_N: float
    gravity_force_z_N: float
    returning_support_force_z_N: float
    liquid_volume_residual_m3: float
    vertical_momentum_residual_kg_m_s: float
    airborne_first_moment_residual_m4: float
    height_evidence_status: str = (
        "declared_reduced_order_derived_proxy__not_external_free_surface"
    )


@dataclass(frozen=True, slots=True)
class ExteriorPlumeStageEvaluation:
    delta: ExteriorPlumeDelta
    component_exchange: BoundaryExchange
    diagnostics: ExteriorPlumeStageDiagnostics


class F0AtmosphericExteriorPlumeOwner:
    """Conservative airborne/returning exterior-liquid state owner."""

    persistent_cycle_ready = True
    validation_only = True
    evidence_status = (
        "S1-1D-F0_declared_zeroD_airborne_lump_plus_finite_return_queue__"
        "full_riser_aperture_return_cap__no_restitution_loss__"
        "derived_height_proxy__not_published_external_jet_geometry"
    )

    def __init__(self, *, gravity_m_s2: float = F0_EXTERIOR_GRAVITY_M_S2) -> None:
        self.gravity_m_s2 = _positive("exterior plume gravity", gravity_m_s2)

    @staticmethod
    def _diameter_m(geometry: CoupledGeometry) -> float:
        return math.sqrt(4.0 * geometry.vertical_area_m2 / math.pi)

    def _contact_tolerance_m(self, geometry: CoupledGeometry) -> float:
        # Numerical event tolerance, not a physical film or fitted plume scale.
        return 1.0e-12 * self._diameter_m(geometry)

    def airborne_at_rim(
        self, state: ExteriorPlumeState, geometry: CoupledGeometry
    ) -> bool:
        contact = self._contact_tolerance_m(geometry)
        # The event-limited RK sequence approaches the declared numerical
        # plane monotonically.  Admit only a few representable numbers above
        # that plane so floating-point roundoff cannot stall it forever; this
        # is not an added physical film thickness.
        roundoff = 8.0 * math.ulp(contact)
        return (
            state.airborne_liquid_volume_m3 > 0.0
            and state.derived_centroid_height_proxy_m
            <= contact + roundoff
        )

    def prepare_atomic_state(
        self, state: ExteriorPlumeState, geometry: CoupledGeometry
    ) -> ExteriorPlumeState:
        """Conservatively relabel a returned airborne lump before both RK stages."""

        if not self.airborne_at_rim(state, geometry):
            return state
        if state.airborne_vertical_momentum_kg_m_s >= 0.0:
            return state
        prepared = ExteriorPlumeState(
            airborne_liquid_volume_m3=0.0,
            airborne_vertical_momentum_kg_m_s=0.0,
            airborne_liquid_first_moment_m4=0.0,
            returning_liquid_volume_m3=(
                state.returning_liquid_volume_m3
                + state.airborne_liquid_volume_m3
            ),
            returning_downward_momentum_kg_m_s=(
                state.returning_downward_momentum_kg_m_s
                - state.airborne_vertical_momentum_kg_m_s
            ),
        )
        if not math.isclose(
            prepared.liquid_volume_m3,
            state.liquid_volume_m3,
            rel_tol=0.0,
            abs_tol=_MATERIAL_TOLERANCE_M3,
        ) or not math.isclose(
            prepared.vertical_momentum_kg_m_s,
            state.vertical_momentum_kg_m_s,
            rel_tol=0.0,
            abs_tol=_MOMENTUM_TOLERANCE_KG_M_S,
        ):
            raise ContractViolation("exterior rim-event relabel is not conservative")
        return prepared

    def finite_reentry_fallback(
        self,
        state: ExteriorPlumeState,
        geometry: CoupledGeometry,
        *,
        dt_s: float,
    ) -> AtmosphericLiquidFallback | None:
        """Return a finite state-derived donor for the resolved riser top."""

        _positive("exterior plume RK-stage dt", dt_s)
        volume = state.returning_liquid_volume_m3
        momentum = state.returning_downward_momentum_kg_m_s
        if volume == 0.0 or momentum == 0.0:
            return None
        speed = momentum / (geometry.liquid_density_kg_m3 * volume)
        return AtmosphericLiquidFallback(
            # Declared maximum geometric aperture.  This is not a published
            # exterior-jet area and is disclosed in ``evidence_status``.
            donor_area_m2=geometry.vertical_area_m2,
            downward_speed_m_s=speed,
            available_volume_m3=volume,
            evidence_status=(
                "persistent_return_queue_state_speed__declared_full_rim_aperture__"
                "not_published_external_jet_geometry"
            ),
        )

    def stable_timestep_s(
        self, state: ExteriorPlumeState, geometry: CoupledGeometry
    ) -> float:
        volume = state.airborne_liquid_volume_m3
        if volume == 0.0:
            return math.inf
        rho = geometry.liquid_density_kg_m3
        height = state.derived_centroid_height_proxy_m
        speed = state.airborne_vertical_momentum_kg_m_s / (rho * volume)
        if self.airborne_at_rim(state, geometry) and speed <= 0.0:
            return math.inf
        height_above_event = max(
            height - self._contact_tolerance_m(geometry), 0.0
        )
        discriminant = (
            speed * speed + 2.0 * self.gravity_m_s2 * height_above_event
        )
        root = math.sqrt(max(discriminant, 0.0))
        if speed < 0.0:
            # Algebraically identical positive root without subtractive
            # cancellation when a falling lump is already very near the rim.
            event_time = 2.0 * height_above_event / (root - speed)
        else:
            event_time = (speed + root) / self.gravity_m_s2
        if not math.isfinite(event_time) or event_time <= 0.0:
            raise ContractViolation("exterior airborne lump has no positive rim-event time")
        return 0.25 * event_time

    def evaluate_stage(
        self,
        state: ExteriorPlumeState,
        geometry: CoupledGeometry,
        *,
        top_liquid: AtmosphericLiquidFlux,
        dt_s: float,
    ) -> ExteriorPlumeStageEvaluation:
        """Return one pure semidiscrete exterior rate from accepted rim fluxes."""

        dt = _positive("exterior plume RK-stage dt", dt_s)
        if (
            state.airborne_liquid_volume_m3 > 0.0
            and state.airborne_liquid_first_moment_m4 == 0.0
            and state.airborne_vertical_momentum_kg_m_s < 0.0
        ):
            raise ContractViolation(
                "falling airborne plume must be atomically prepared before RK evaluation"
            )
        rho = geometry.liquid_density_kg_m3
        q_out = top_liquid.outflow_rate_m3_s
        u_out = top_liquid.outflow_speed_m_s
        q_return = top_liquid.reentry_rate_m3_s
        u_return = top_liquid.reentry_speed_m_s
        if q_out == 0.0 and u_out != 0.0:
            raise ContractViolation("zero rim outflow cannot carry a donor speed")
        if q_return == 0.0 and u_return != 0.0:
            raise ContractViolation("zero rim re-entry cannot carry a donor speed")

        fallback = self.finite_reentry_fallback(state, geometry, dt_s=dt)
        if q_return > 0.0:
            if fallback is None:
                raise ContractViolation(
                    "riser consumed liquid without a finite returning queue"
                )
            if not top_liquid.finite_exterior_inventory:
                raise ContractViolation("riser re-entry did not declare finite inventory")
            if not math.isclose(
                u_return,
                fallback.downward_speed_m_s,
                rel_tol=2.0e-12,
                abs_tol=1.0e-14,
            ):
                raise ContractViolation(
                    "riser re-entry speed differs from the returning queue state"
                )
            if (
                top_liquid.stage_consumed_volume_m3
                > state.returning_liquid_volume_m3 + _MATERIAL_TOLERANCE_M3
            ):
                raise ContractViolation("riser re-entry exceeds returning inventory")

        airborne_volume_rate = q_out
        airborne_momentum_rate = (
            rho * q_out * u_out
            - rho * self.gravity_m_s2 * state.airborne_liquid_volume_m3
        )
        airborne_first_moment_rate = (
            state.airborne_vertical_momentum_kg_m_s / rho
        )
        returning_volume_rate = -q_return
        returning_momentum_rate = -rho * q_return * u_return

        after_returning_volume = (
            state.returning_liquid_volume_m3 + dt * returning_volume_rate
        )
        after_returning_momentum = (
            state.returning_downward_momentum_kg_m_s
            + dt * returning_momentum_rate
        )
        if after_returning_volume < -_MATERIAL_TOLERANCE_M3:
            raise ContractViolation("returning queue stage created negative volume")
        if after_returning_momentum < -_MOMENTUM_TOLERANCE_KG_M_S:
            raise ContractViolation("returning queue stage created negative momentum")
        if after_returning_volume <= _MATERIAL_TOLERANCE_M3:
            if abs(after_returning_momentum) > _MOMENTUM_TOLERANCE_KG_M_S:
                raise ContractViolation("depleted return queue retains ghost momentum")
            after_returning_volume = 0.0
            after_returning_momentum = 0.0
        elif after_returning_momentum < 0.0:
            after_returning_momentum = 0.0

        after = ExteriorPlumeState(
            airborne_liquid_volume_m3=(
                state.airborne_liquid_volume_m3 + dt * airborne_volume_rate
            ),
            airborne_vertical_momentum_kg_m_s=(
                state.airborne_vertical_momentum_kg_m_s
                + dt * airborne_momentum_rate
            ),
            airborne_liquid_first_moment_m4=(
                state.airborne_liquid_first_moment_m4
                + dt * airborne_first_moment_rate
            ),
            returning_liquid_volume_m3=after_returning_volume,
            returning_downward_momentum_kg_m_s=after_returning_momentum,
        )

        delta = ExteriorPlumeDelta(
            airborne_liquid_volume_m3=(
                after.airborne_liquid_volume_m3
                - state.airborne_liquid_volume_m3
            )
            / dt,
            airborne_vertical_momentum_kg_m_s=(
                after.airborne_vertical_momentum_kg_m_s
                - state.airborne_vertical_momentum_kg_m_s
            )
            / dt,
            airborne_liquid_first_moment_m4=(
                after.airborne_liquid_first_moment_m4
                - state.airborne_liquid_first_moment_m4
            )
            / dt,
            returning_liquid_volume_m3=(
                after.returning_liquid_volume_m3
                - state.returning_liquid_volume_m3
            )
            / dt,
            returning_downward_momentum_kg_m_s=(
                after.returning_downward_momentum_kg_m_s
                - state.returning_downward_momentum_kg_m_s
            )
            / dt,
        )

        liquid_port_momentum = rho * (
            q_out * u_out + q_return * u_return
        )
        gravity_force = -rho * self.gravity_m_s2 * state.liquid_volume_m3
        return_support = (
            rho * self.gravity_m_s2 * state.returning_liquid_volume_m3
        )
        exchange = BoundaryExchange(
            liquid_inflow_m3_s=q_out,
            liquid_outflow_m3_s=q_return,
            momentum_z_in_N=liquid_port_momentum,
            external_force_z_N=gravity_force + return_support,
        )
        volume_residual = (
            after.liquid_volume_m3
            - state.liquid_volume_m3
            - dt * exchange.liquid_volume_net_rate
        )
        momentum_residual = (
            after.vertical_momentum_kg_m_s
            - state.vertical_momentum_kg_m_s
            - dt * exchange.mixture_momentum_z_net_rate
        )
        first_moment_residual = (
            after.airborne_liquid_first_moment_m4
            - state.airborne_liquid_first_moment_m4
            - dt * state.airborne_vertical_momentum_kg_m_s / rho
        )
        first_moment_scale = max(
            abs(after.airborne_liquid_first_moment_m4),
            abs(state.airborne_liquid_first_moment_m4),
            abs(dt * state.airborne_vertical_momentum_kg_m_s / rho),
            1.0e-30,
        )
        if (
            abs(volume_residual) > _MATERIAL_TOLERANCE_M3
            or abs(momentum_residual) > _MOMENTUM_TOLERANCE_KG_M_S
            or abs(first_moment_residual) > 2.0e-14 * first_moment_scale
        ):
            raise ContractViolation("persistent exterior plume ledger does not close")
        diagnostics = ExteriorPlumeStageDiagnostics(
            before=state,
            after=after,
            top_liquid=top_liquid,
            liquid_port_momentum_rate_N=liquid_port_momentum,
            gravity_force_z_N=gravity_force,
            returning_support_force_z_N=return_support,
            liquid_volume_residual_m3=volume_residual,
            vertical_momentum_residual_kg_m_s=momentum_residual,
            airborne_first_moment_residual_m4=first_moment_residual,
        )
        return ExteriorPlumeStageEvaluation(delta, exchange, diagnostics)


__all__ = [
    "ExteriorPlumeStageDiagnostics",
    "ExteriorPlumeStageEvaluation",
    "F0AtmosphericExteriorPlumeOwner",
    "F0_EXTERIOR_GRAVITY_M_S2",
]
