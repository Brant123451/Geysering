"""Post-commit bridge from the atomic runner to canonical trajectory samples."""

from __future__ import annotations

from dataclasses import dataclass
import math

from model.accepted_observation_diagnostics import (
    InstantaneousGaugePressures,
    NativeIntervalDiagnostics,
    PressureSemantics,
)
from model.conservation import LedgerEntry
from model.errors import ContractViolation, MissingPhysicalClosure
from model.events import TopOutflowEventIntegrator, TopOutflowEventSnapshot
from model.joint_network_runner import AcceptedStepContext
from model.state import CoupledState

from .contracts import (
    AcceptedGrossFluxPacket,
    AcceptedNodePacket,
    AcceptedStepDiagnostics,
    CommonAcceptedSample,
    GaugePressurePacket,
    InternalMouthEventPacket,
    ObservationContractError,
    ObserverContract,
    load_observer_contract,
)


@dataclass(frozen=True, slots=True)
class AcceptedTrajectoryStep:
    """Complete evidence packet emitted after one successful atomic commit."""

    before_state: CoupledState
    after_state: CoupledState
    actual_dt_s: float
    ledger_entries: tuple[LedgerEntry, ...]
    stage2_time_start_s: float
    stage2_time_end_s: float
    diagnostics: AcceptedStepDiagnostics
    pressure_semantics: PressureSemantics
    mouth_event_snapshot: TopOutflowEventSnapshot


def _pressure_packet(values: InstantaneousGaugePressures) -> GaugePressurePacket:
    return GaugePressurePacket(
        P1=values.P1,
        P2=values.P2,
        P3=values.P3,
        P4=values.P4,
        P5=values.P5,
        P6=values.P6,
    )


def _gross_packet(
    values: NativeIntervalDiagnostics,
    *,
    cumulative_mouth_liquid_outflow_m3: float,
) -> AcceptedGrossFluxPacket:
    return AcceptedGrossFluxPacket(
        supply_branch_liquid_outflow_m3_s=(
            values.supply_branch_liquid_outflow_m3_s
        ),
        supply_branch_gas_inflow_kg_s=values.supply_branch_gas_inflow_kg_s,
        mouth_liquid_outflow_m3_s=values.mouth_liquid_outflow_m3_s,
        mouth_liquid_inflow_m3_s=values.mouth_liquid_inflow_m3_s,
        mouth_gas_outflow_kg_s=values.mouth_gas_outflow_kg_s,
        mouth_gas_inflow_kg_s=values.mouth_gas_inflow_kg_s,
        cumulative_mouth_liquid_outflow_m3=(
            cumulative_mouth_liquid_outflow_m3
        ),
    )


def _node_packet(
    values: NativeIntervalDiagnostics,
    *,
    cumulative_node_reaction_impulse_Ns: float,
) -> AcceptedNodePacket:
    return AcceptedNodePacket(
        air_supply_liquid_volume_residual_m3_s=(
            values.air_supply_liquid_volume_residual_m3_s
        ),
        air_supply_gas_mass_residual_kg_s=values.air_supply_gas_mass_residual_kg_s,
        air_supply_momentum_x_residual_N=values.air_supply_momentum_x_residual_N,
        air_supply_momentum_z_residual_N=values.air_supply_momentum_z_residual_N,
        riser_liquid_volume_residual_m3_s=values.riser_liquid_volume_residual_m3_s,
        riser_gas_mass_residual_kg_s=values.riser_gas_mass_residual_kg_s,
        riser_momentum_x_residual_N=values.riser_momentum_x_residual_N,
        riser_momentum_z_residual_N=values.riser_momentum_z_residual_N,
        node_reaction_impulse_Ns=cumulative_node_reaction_impulse_Ns,
    )


def _event_packet(snapshot: TopOutflowEventSnapshot) -> InternalMouthEventPacket:
    return InternalMouthEventPacket(
        active=(
            snapshot.connected_water_to_mouth
            and snapshot.active_persistence_s > 0.0
        ),
        accepted_once=snapshot.event_accepted,
        onset_s=snapshot.event_onset_s,
        acceptance_time_s=snapshot.acceptance_time_s,
        evidence_status=(
            "native_internal_gross_mouth_outflow_persistence__accepted_steps_only"
        ),
    )


def _diagnostics(
    *,
    pressure: InstantaneousGaugePressures,
    interval: NativeIntervalDiagnostics,
    cumulative_mouth_liquid_outflow_m3: float,
    cumulative_node_reaction_impulse_Ns: float,
    event: TopOutflowEventSnapshot,
) -> AcceptedStepDiagnostics:
    return AcceptedStepDiagnostics(
        pressure=_pressure_packet(pressure),
        gross_flux=_gross_packet(
            interval,
            cumulative_mouth_liquid_outflow_m3=(
                cumulative_mouth_liquid_outflow_m3
            ),
        ),
        nodes=_node_packet(
            interval,
            cumulative_node_reaction_impulse_Ns=(
                cumulative_node_reaction_impulse_Ns
            ),
        ),
        mouth_event=_event_packet(event),
    )


class Stage2AcceptedTrajectoryBridge:
    """Collect every accepted transaction and emit exact 0.1 s samples.

    The runner calls this object only after its atomic ledger append.  The
    bridge never interpolates a state in time.  A common sample is emitted only
    when the runner's event ceiling made the accepted state land exactly on
    ``0, 0.1, 0.2, ...`` seconds from Stage-2 opening.
    """

    def __init__(
        self,
        *,
        stage2_origin_absolute_s: float,
        contract: ObserverContract | None = None,
        mouth_event_integrator: TopOutflowEventIntegrator | None = None,
    ) -> None:
        self.contract = contract or load_observer_contract()
        origin = float(stage2_origin_absolute_s)
        if not math.isfinite(origin) or origin < 0.0:
            raise ObservationContractError(
                "Stage-2 bridge origin must be finite and non-negative"
            )
        self.stage2_origin_absolute_s = origin
        self.mouth_event_integrator = (
            TopOutflowEventIntegrator()
            if mouth_event_integrator is None
            else mouth_event_integrator
        )
        if self.mouth_event_integrator.time_s != 0.0:
            raise ObservationContractError(
                "a fresh Stage-2 bridge requires an event integrator at t=0"
            )
        self._steps: list[AcceptedTrajectoryStep] = []
        self._samples: list[CommonAcceptedSample] = []
        self._pending_ledgers: list[LedgerEntry] = []
        self._last_stage2_time_s: float | None = None
        self._node_reaction_impulse_Ns = 0.0

    @property
    def accepted_steps(self) -> tuple[AcceptedTrajectoryStep, ...]:
        return tuple(self._steps)

    @property
    def common_samples(self) -> tuple[CommonAcceptedSample, ...]:
        return tuple(self._samples)

    @property
    def stage2_elapsed_s(self) -> float:
        return 0.0 if self._last_stage2_time_s is None else self._last_stage2_time_s

    def _is_common_time(self, value: float) -> bool:
        index = round(value / self.contract.common_dt_s)
        return math.isclose(
            value,
            index * self.contract.common_dt_s,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        )

    def _append_initial_sample(self, context: AcceptedStepContext) -> None:
        if self._samples:
            return
        pressure = context.diagnostics.pressure_before
        interval = context.diagnostics.rk1_native_interval
        if pressure is None or interval is None:
            raise MissingPhysicalClosure(
                "initial accepted callback omitted native before-state diagnostics"
            )
        event = self.mouth_event_integrator.snapshot()
        diagnostics = _diagnostics(
            pressure=pressure,
            interval=interval,
            cumulative_mouth_liquid_outflow_m3=0.0,
            cumulative_node_reaction_impulse_Ns=0.0,
            event=event,
        )
        self._samples.append(
            CommonAcceptedSample(
                stage2_time_s=0.0,
                state=context.before_state,
                diagnostics=diagnostics,
                ledger_entries_since_previous_sample=(),
            )
        )

    def __call__(self, context: AcceptedStepContext) -> None:
        """Consume one already committed step; no rejected step can reach here."""

        if not isinstance(context, AcceptedStepContext):
            raise ObservationContractError(
                "Stage-2 bridge callback requires AcceptedStepContext"
            )
        if (
            context.physical_stage != "stage2_pressure_reservoir"
            or context.stage2_time_start_s is None
            or context.stage2_time_end_s is None
        ):
            raise ObservationContractError(
                "Stage-2 bridge cannot consume a Stage-1 callback packet"
            )
        expected_origin = context.before_state.time_s - context.stage2_time_start_s
        if not math.isclose(
            expected_origin,
            self.stage2_origin_absolute_s,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ObservationContractError("callback changed the Stage-2 time origin")
        expected_start = 0.0 if self._last_stage2_time_s is None else self._last_stage2_time_s
        if not math.isclose(
            context.stage2_time_start_s,
            expected_start,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ObservationContractError(
                "accepted callback intervals have a gap, overlap or reordering"
            )
        self._append_initial_sample(context)

        interval = context.diagnostics.accepted_native_interval
        pressure = context.diagnostics.pressure_after
        if interval is None or pressure is None:
            raise MissingPhysicalClosure(
                "accepted callback omitted native post-state or interval diagnostics"
            )
        event = self.mouth_event_integrator.advance(
            context.actual_dt_s,
            interval.mouth_liquid_outflow_m3_s,
            interval.mouth_liquid_inflow_m3_s,
            connected_water_to_mouth=interval.connected_water_to_mouth,
        )
        if not math.isclose(
            event.time_s,
            context.stage2_time_end_s,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ObservationContractError(
                "native mouth-event clock drifted from Stage-2 physical time"
            )
        self._node_reaction_impulse_Ns += (
            context.actual_dt_s * interval.node_reaction_rate_magnitude_N
        )
        self._pending_ledgers.extend(context.ledger_entries)
        diagnostics = _diagnostics(
            pressure=pressure,
            interval=interval,
            cumulative_mouth_liquid_outflow_m3=event.gross_liquid_outflow_m3,
            cumulative_node_reaction_impulse_Ns=self._node_reaction_impulse_Ns,
            event=event,
        )
        step = AcceptedTrajectoryStep(
            before_state=context.before_state,
            after_state=context.after_state,
            actual_dt_s=context.actual_dt_s,
            ledger_entries=context.ledger_entries,
            stage2_time_start_s=context.stage2_time_start_s,
            stage2_time_end_s=context.stage2_time_end_s,
            diagnostics=diagnostics,
            pressure_semantics=pressure.semantics,
            mouth_event_snapshot=event,
        )
        self._steps.append(step)
        self._last_stage2_time_s = context.stage2_time_end_s

        if self._is_common_time(context.stage2_time_end_s):
            self._samples.append(
                CommonAcceptedSample(
                    stage2_time_s=context.stage2_time_end_s,
                    state=context.after_state,
                    diagnostics=diagnostics,
                    ledger_entries_since_previous_sample=tuple(
                        self._pending_ledgers
                    ),
                )
            )
            self._pending_ledgers.clear()

    def assert_complete_through(self, stage2_time_s: float) -> None:
        target = float(stage2_time_s)
        if not math.isfinite(target) or target < 0.0:
            raise ContractViolation("bridge completion target must be non-negative")
        if not self._is_common_time(target):
            raise ObservationContractError(
                "bridge completion target must lie on the 0.1 s common grid"
            )
        if not self._samples or not math.isclose(
            self._samples[-1].stage2_time_s,
            target,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ObservationContractError(
                "accepted trajectory has not reached the requested common time"
            )
        if self._pending_ledgers:
            raise ObservationContractError(
                "accepted ledger entries remain beyond the last common sample"
            )


__all__ = [
    "AcceptedTrajectoryStep",
    "Stage2AcceptedTrajectoryBridge",
]
