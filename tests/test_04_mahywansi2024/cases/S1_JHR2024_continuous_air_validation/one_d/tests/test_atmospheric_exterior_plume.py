from dataclasses import replace
import math

import pytest

from model.atmospheric_exterior_plume import F0AtmosphericExteriorPlumeOwner
from model.errors import ContractViolation
from model.flux import ExteriorPlumeDelta, state_token
from model.initialization import build_s1_initial_assembly
from model.joint_network_runner import (
    S1JointNetworkRunner,
    StructuralZeroJointOperator,
    build_current_physical_operator,
)
from model.state import ExteriorPlumeState
from model.vertical_pressure_void_component import AtmosphericLiquidFlux


def _top_flux(
    *,
    outflow: float = 0.0,
    outflow_speed: float = 0.0,
    reentry: float = 0.0,
    reentry_speed: float = 0.0,
    available: float | None = None,
    dt_s: float = 1.0,
) -> AtmosphericLiquidFlux:
    return AtmosphericLiquidFlux(
        outflow_rate_m3_s=outflow,
        outflow_speed_m_s=outflow_speed,
        reentry_demand_rate_m3_s=reentry,
        reentry_rate_m3_s=reentry,
        reentry_speed_m_s=reentry_speed,
        exterior_available_volume_m3=available,
        stage_consumed_volume_m3=dt_s * reentry,
        finite_exterior_inventory=available is not None,
    )


def _apply_rate(
    state: ExteriorPlumeState, rate: ExteriorPlumeDelta, dt_s: float
) -> ExteriorPlumeState:
    return ExteriorPlumeState(
        airborne_liquid_volume_m3=(
            state.airborne_liquid_volume_m3
            + dt_s * rate.airborne_liquid_volume_m3
        ),
        airborne_vertical_momentum_kg_m_s=(
            state.airborne_vertical_momentum_kg_m_s
            + dt_s * rate.airborne_vertical_momentum_kg_m_s
        ),
        airborne_liquid_first_moment_m4=(
            state.airborne_liquid_first_moment_m4
            + dt_s * rate.airborne_liquid_first_moment_m4
        ),
        returning_liquid_volume_m3=(
            state.returning_liquid_volume_m3
            + dt_s * rate.returning_liquid_volume_m3
        ),
        returning_downward_momentum_kg_m_s=(
            state.returning_downward_momentum_kg_m_s
            + dt_s * rate.returning_downward_momentum_kg_m_s
        ),
    )


def _rk2_step(owner, state, geometry, *, top, dt_s):
    first = owner.evaluate_stage(state, geometry, top_liquid=top, dt_s=dt_s)
    predictor = _apply_rate(state, first.delta, dt_s)
    second = owner.evaluate_stage(predictor, geometry, top_liquid=top, dt_s=dt_s)
    return ExteriorPlumeState(
        airborne_liquid_volume_m3=state.airborne_liquid_volume_m3
        + 0.5
        * dt_s
        * (
            first.delta.airborne_liquid_volume_m3
            + second.delta.airborne_liquid_volume_m3
        ),
        airborne_vertical_momentum_kg_m_s=state.airborne_vertical_momentum_kg_m_s
        + 0.5
        * dt_s
        * (
            first.delta.airborne_vertical_momentum_kg_m_s
            + second.delta.airborne_vertical_momentum_kg_m_s
        ),
        airborne_liquid_first_moment_m4=state.airborne_liquid_first_moment_m4
        + 0.5
        * dt_s
        * (
            first.delta.airborne_liquid_first_moment_m4
            + second.delta.airborne_liquid_first_moment_m4
        ),
        returning_liquid_volume_m3=state.returning_liquid_volume_m3
        + 0.5
        * dt_s
        * (
            first.delta.returning_liquid_volume_m3
            + second.delta.returning_liquid_volume_m3
        ),
        returning_downward_momentum_kg_m_s=(
            state.returning_downward_momentum_kg_m_s
            + 0.5
            * dt_s
            * (
                first.delta.returning_downward_momentum_kg_m_s
                + second.delta.returning_downward_momentum_kg_m_s
            )
        ),
    )


def test_exterior_state_owns_gross_airborne_and_returning_populations() -> None:
    state = ExteriorPlumeState(
        airborne_liquid_volume_m3=2.0e-6,
        airborne_vertical_momentum_kg_m_s=1.0e-4,
        airborne_liquid_first_moment_m4=6.0e-7,
        returning_liquid_volume_m3=1.0e-6,
        returning_downward_momentum_kg_m_s=2.0e-5,
    )
    assert state.liquid_volume_m3 == pytest.approx(3.0e-6)
    assert state.vertical_momentum_kg_m_s == pytest.approx(8.0e-5)
    assert state.derived_centroid_height_proxy_m == pytest.approx(0.3)
    assert "not_external_free_surface" in state.height_evidence_status
    with pytest.raises(ContractViolation, match="empty airborne plume"):
        ExteriorPlumeState(airborne_vertical_momentum_kg_m_s=1.0e-9)


def test_zero_inventory_has_no_fallback_and_zero_stage_rate(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    state = ExteriorPlumeState()
    assert owner.finite_reentry_fallback(state, geometry, dt_s=0.01) is None
    result = owner.evaluate_stage(state, geometry, top_liquid=_top_flux(), dt_s=0.01)
    assert result.delta == ExteriorPlumeDelta()
    assert result.component_exchange.liquid_volume_net_rate == 0.0
    assert result.component_exchange.mixture_momentum_z_net_rate == 0.0


def test_positive_outflow_enters_airborne_inventory_with_momentum(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    dt = 0.02
    q = 4.0e-6
    speed = 0.35
    result = owner.evaluate_stage(
        ExteriorPlumeState(),
        geometry,
        top_liquid=_top_flux(outflow=q, outflow_speed=speed),
        dt_s=dt,
    )
    rho = geometry.liquid_density_kg_m3
    after = result.diagnostics.after
    assert after.airborne_liquid_volume_m3 == pytest.approx(dt * q)
    assert after.airborne_vertical_momentum_kg_m_s == pytest.approx(
        dt * rho * q * speed
    )
    assert after.returning_liquid_volume_m3 == 0.0
    assert result.component_exchange.liquid_inflow_m3_s == q


def test_ssp_rk2_ballistic_first_moment_counts_gravity_once(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    rho = geometry.liquid_density_kg_m3
    volume = 1.0e-6
    speed = 0.4
    height = 0.2
    dt = 0.01
    state = ExteriorPlumeState(
        airborne_liquid_volume_m3=volume,
        airborne_vertical_momentum_kg_m_s=rho * volume * speed,
        airborne_liquid_first_moment_m4=volume * height,
    )
    after = _rk2_step(owner, state, geometry, top=_top_flux(), dt_s=dt)
    assert after.airborne_vertical_momentum_kg_m_s == pytest.approx(
        rho * volume * (speed - owner.gravity_m_s2 * dt)
    )
    assert after.airborne_liquid_first_moment_m4 == pytest.approx(
        volume * (height + speed * dt - 0.5 * owner.gravity_m_s2 * dt * dt)
    )
    audit = owner.evaluate_stage(
        state, geometry, top_liquid=_top_flux(), dt_s=dt
    ).diagnostics
    assert audit.airborne_first_moment_residual_m4 == pytest.approx(
        0.0, abs=1.0e-22
    )


def test_rim_event_relabel_preserves_total_volume_and_signed_momentum(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    state = ExteriorPlumeState(
        airborne_liquid_volume_m3=3.0e-7,
        airborne_vertical_momentum_kg_m_s=-7.0e-5,
        airborne_liquid_first_moment_m4=0.0,
        returning_liquid_volume_m3=2.0e-7,
        returning_downward_momentum_kg_m_s=4.0e-5,
    )
    prepared = owner.prepare_atomic_state(state, geometry)
    assert prepared.airborne_liquid_volume_m3 == 0.0
    assert prepared.returning_liquid_volume_m3 == pytest.approx(5.0e-7)
    assert prepared.returning_downward_momentum_kg_m_s == pytest.approx(1.1e-4)
    assert prepared.liquid_volume_m3 == pytest.approx(state.liquid_volume_m3)
    assert prepared.vertical_momentum_kg_m_s == pytest.approx(
        state.vertical_momentum_kg_m_s
    )


def test_rim_event_uses_only_an_ulp_scale_numerical_band(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    contact = owner._contact_tolerance_m(geometry)
    just_above = ExteriorPlumeState(
        airborne_liquid_volume_m3=1.0,
        airborne_vertical_momentum_kg_m_s=-1.0,
        airborne_liquid_first_moment_m4=math.nextafter(contact, math.inf),
    )
    clearly_above = replace(
        just_above,
        airborne_liquid_first_moment_m4=contact + 32.0 * math.ulp(contact),
    )

    assert owner.airborne_at_rim(just_above, geometry) is True
    assert owner.airborne_at_rim(clearly_above, geometry) is False


def test_return_speed_is_state_derived_and_not_dt_dependent(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    rho = geometry.liquid_density_kg_m3
    state = ExteriorPlumeState(
        returning_liquid_volume_m3=1.0e-6,
        returning_downward_momentum_kg_m_s=rho * 1.0e-6 * 0.2,
    )
    coarse = owner.finite_reentry_fallback(state, geometry, dt_s=1.0e-3)
    fine = owner.finite_reentry_fallback(state, geometry, dt_s=2.5e-4)
    assert coarse is not None and fine is not None
    assert coarse.downward_speed_m_s == pytest.approx(0.2)
    assert fine.downward_speed_m_s == pytest.approx(0.2)
    assert coarse.downward_rate_m3_s == pytest.approx(fine.downward_rate_m3_s)
    assert "full_rim_aperture" in coarse.evidence_status


def _integrate_fixed_return(owner, state, geometry, *, dt, duration, demand):
    current = state
    initial = state.returning_liquid_volume_m3
    for _ in range(round(duration / dt)):
        fallback = owner.finite_reentry_fallback(current, geometry, dt_s=dt)
        assert fallback is not None
        q = min(demand, fallback.downward_rate_m3_s, current.returning_liquid_volume_m3 / dt)
        top = _top_flux(
            reentry=q,
            reentry_speed=fallback.downward_speed_m_s,
            available=current.returning_liquid_volume_m3,
            dt_s=dt,
        )
        current = _rk2_step(owner, current, geometry, top=top, dt_s=dt)
    return initial - current.returning_liquid_volume_m3


def test_fixed_demand_return_volume_converges_across_dt(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    rho = geometry.liquid_density_kg_m3
    state = ExteriorPlumeState(
        returning_liquid_volume_m3=1.0e-6,
        returning_downward_momentum_kg_m_s=rho * 1.0e-6 * 0.2,
    )
    duration = 2.0e-3
    demand = 2.0e-5
    volumes = tuple(
        _integrate_fixed_return(
            owner, state, geometry, dt=dt, duration=duration, demand=demand
        )
        for dt in (2.0e-4, 1.0e-4, 5.0e-5)
    )
    assert volumes == pytest.approx((demand * duration,) * 3, rel=2.0e-12)


def test_simultaneous_outflow_and_reentry_keep_two_gross_inventories(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    rho = geometry.liquid_density_kg_m3
    state = ExteriorPlumeState(
        returning_liquid_volume_m3=1.0e-6,
        returning_downward_momentum_kg_m_s=rho * 1.0e-6 * 0.15,
    )
    dt = 1.0e-3
    top = _top_flux(
        outflow=3.0e-6,
        outflow_speed=0.25,
        reentry=2.0e-6,
        reentry_speed=0.15,
        available=state.returning_liquid_volume_m3,
        dt_s=dt,
    )
    result = owner.evaluate_stage(state, geometry, top_liquid=top, dt_s=dt)
    after = result.diagnostics.after
    assert after.airborne_liquid_volume_m3 == pytest.approx(3.0e-9)
    assert after.returning_liquid_volume_m3 == pytest.approx(0.998e-6)
    assert result.component_exchange.liquid_inflow_m3_s == 3.0e-6
    assert result.component_exchange.liquid_outflow_m3_s == 2.0e-6


def test_persistent_owner_completes_two_storage_return_cycles(geometry) -> None:
    owner = F0AtmosphericExteriorPlumeOwner()
    rho = geometry.liquid_density_kg_m3
    first = ExteriorPlumeState(
        returning_liquid_volume_m3=1.0e-8,
        returning_downward_momentum_kg_m_s=rho * 1.0e-8 * 0.2,
    )
    dt_return = 0.01
    fallback = owner.finite_reentry_fallback(first, geometry, dt_s=dt_return)
    assert fallback is not None
    first_empty = owner.evaluate_stage(
        first,
        geometry,
        top_liquid=_top_flux(
            reentry=first.returning_liquid_volume_m3 / dt_return,
            reentry_speed=fallback.downward_speed_m_s,
            available=first.returning_liquid_volume_m3,
            dt_s=dt_return,
        ),
        dt_s=dt_return,
    ).diagnostics.after
    assert first_empty == ExteriorPlumeState()

    current = _rk2_step(
        owner,
        first_empty,
        geometry,
        top=_top_flux(outflow=1.0e-6, outflow_speed=0.2),
        dt_s=0.01,
    )
    for _ in range(260):
        if owner.airborne_at_rim(current, geometry) and (
            current.airborne_vertical_momentum_kg_m_s < 0.0
        ):
            break
        ceiling = owner.stable_timestep_s(current, geometry)
        dt = 1.0e-3 if math.isinf(ceiling) else min(1.0e-3, ceiling)
        current = _rk2_step(owner, current, geometry, top=_top_flux(), dt_s=dt)
    prepared = owner.prepare_atomic_state(current, geometry)
    assert prepared.airborne_liquid_volume_m3 == 0.0
    assert prepared.returning_liquid_volume_m3 > 0.0
    second = owner.finite_reentry_fallback(prepared, geometry, dt_s=dt_return)
    assert second is not None
    second_empty = owner.evaluate_stage(
        prepared,
        geometry,
        top_liquid=_top_flux(
            reentry=prepared.returning_liquid_volume_m3 / dt_return,
            reentry_speed=second.downward_speed_m_s,
            available=prepared.returning_liquid_volume_m3,
            dt_s=dt_return,
        ),
        dt_s=dt_return,
    ).diagnostics.after
    assert second_empty == ExteriorPlumeState()


def test_second_rk_rejection_rolls_back_event_relabel_and_ledger(monkeypatch) -> None:
    assembly = build_s1_initial_assembly()
    operator = build_current_physical_operator()
    runner = S1JointNetworkRunner(assembly.geometry, operator)
    state = replace(
        assembly.state,
        exterior_plume=ExteriorPlumeState(
            airborne_liquid_volume_m3=1.0e-8,
            airborne_vertical_momentum_kg_m_s=-1.0e-6,
        ),
    )
    before_token = state_token(state)
    original = operator.joint_stage_owner.evaluate

    def reject_second(*args, **kwargs):
        if kwargs.get("rk_stage") == 2:
            raise ContractViolation("manufactured plume RK2 rejection")
        return original(*args, **kwargs)

    monkeypatch.setattr(operator.joint_stage_owner, "evaluate", reject_second)
    with pytest.raises(ContractViolation, match="manufactured plume RK2 rejection"):
        runner.advance_one(
            state,
            dt_s=1.0e-7,
            physical_stage="stage1_closed",
            transaction_id="plume-stage2-reject",
            require_production=False,
        )
    assert state_token(state) == before_token
    assert runner.committer.ledger.entries == []


class _PreparingStructuralZeroOperator:
    """Validation-only harness for the runner's zero-time plume event path."""

    production_ready = False
    validation_only = True

    def __init__(self) -> None:
        self.plume = F0AtmosphericExteriorPlumeOwner()
        self.zero = StructuralZeroJointOperator()

    def prepare_atomic_state(self, state, geometry, **kwargs):
        del kwargs
        return replace(
            state,
            exterior_plume=self.plume.prepare_atomic_state(
                state.exterior_plume, geometry
            ),
        )

    def evaluate(self, state, geometry, **kwargs):
        return self.zero.evaluate(state, geometry, **kwargs)


def test_successful_rim_relabel_commits_once_with_no_fake_boundary_flux() -> None:
    assembly = build_s1_initial_assembly(vertical_cell_count=20)
    runner = S1JointNetworkRunner(
        assembly.geometry, _PreparingStructuralZeroOperator()
    )
    before = replace(
        assembly.state,
        exterior_plume=ExteriorPlumeState(
            airborne_liquid_volume_m3=1.0e-8,
            airborne_vertical_momentum_kg_m_s=-1.0e-6,
        ),
    )

    result = runner.advance_one(
        before,
        dt_s=1.0e-4,
        physical_stage="stage1_closed",
        transaction_id="plume-event-single-commit",
        require_production=False,
    )

    assert before.exterior_plume.airborne_liquid_volume_m3 == 1.0e-8
    assert result.state.exterior_plume == ExteriorPlumeState(
        returning_liquid_volume_m3=1.0e-8,
        returning_downward_momentum_kg_m_s=1.0e-6,
    )
    assert len(runner.committer.ledger.entries) == 1
    assert runner.committer.ledger.entries[0].transaction_id == (
        "plume-event-single-commit"
    )
    assert result.ledger.boundary.liquid_inflow_m3_s == 0.0
    assert result.ledger.boundary.liquid_outflow_m3_s == 0.0
