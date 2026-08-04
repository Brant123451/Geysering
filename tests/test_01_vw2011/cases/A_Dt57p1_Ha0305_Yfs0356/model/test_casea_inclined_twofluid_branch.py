"""Equation and conservation tests for the strict inclined branch core."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_inclined_twofluid_branch import (  # noqa: E402
    CFLViolationError,
    COMPLETE_RISER_MODEL_READY,
    INTERIOR_BRANCH_CORE_READY,
    InclinedBranchBoundaryStates,
    InclinedTwoFluidParameters,
    InclinedTwoFluidState,
    LossOfHyperbolicityError,
    MISSING_RISER_CLOSURES,
    NUMERICAL_FLUX,
    RIEMANN_FALLBACK_AVAILABLE,
    StateAdmissibilityError,
    block_rusanov_flux,
    cell_source,
    euler_inclined_branch_stage,
    physical_flux,
    primitive_state,
)


def _state_from_primitive(
    params: InclinedTwoFluidParameters,
    *,
    liquid_fraction: float,
    gas_density: float | None = None,
    gas_velocity: float = 0.0,
    liquid_velocity: float = 0.0,
) -> InclinedTwoFluidState:
    liquid_area = liquid_fraction * params.full_area
    gas_area = params.full_area - liquid_area
    if gas_density is None:
        gas_density = params.reference_gas_density
    gas_mass = gas_density * gas_area
    return InclinedTwoFluidState(
        gas_mass=gas_mass,
        gas_momentum=gas_mass * gas_velocity,
        liquid_area=liquid_area,
        liquid_discharge=liquid_area * liquid_velocity,
    )


def _assert_ledger_closed(result, *, atol: float = 2.0e-14) -> None:
    assert result.ledger.residual.vector() == pytest.approx(
        (0.0, 0.0, 0.0, 0.0), abs=atol
    )


def test_lambda_d_is_exactly_current_tex_equation_three() -> None:
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=math.radians(31.0),
    )
    state = _state_from_primitive(
        params,
        liquid_fraction=0.42,
        gas_density=1.27,
        gas_velocity=0.34,
        liquid_velocity=-0.08,
    )
    primitive = primitive_state(state, params, require_hyperbolic=False)
    gas_head = primitive.gas_pressure_gauge / (
        params.liquid_density * params.gravity
    )
    expected = (
        2.0 * params.gravity * gas_head / state.liquid_area
        + (params.liquid_density - primitive.gas_density)
        / params.liquid_density
        * params.gravity
        * params.cosine
        / primitive.top_width
        - primitive.gas_density
        / params.liquid_density
        * (primitive.gas_velocity - primitive.liquid_velocity) ** 2
        / primitive.gas_area
    )
    assert primitive.zeta == pytest.approx(
        params.cosine / primitive.top_width, rel=2.0e-15
    )
    assert primitive.lambda_d == pytest.approx(expected, rel=2.0e-15)


def test_scope_flags_prevent_treating_interior_core_as_complete_riser() -> None:
    assert INTERIOR_BRANCH_CORE_READY is True
    assert COMPLETE_RISER_MODEL_READY is False
    assert "inclined_pressurised_stratified_front" in MISSING_RISER_CLOSURES
    assert "three_branch_tjunction_riemann_problem" in MISSING_RISER_CLOSURES
    assert "top_free_surface_and_vent_event" in MISSING_RISER_CLOSURES


def test_horizontal_uniform_rest_is_preserved_exactly() -> None:
    params = InclinedTwoFluidParameters(diameter=0.094, inclination=0.0)
    state = _state_from_primitive(params, liquid_fraction=0.57)
    states = (state, state, state, state)
    result = euler_inclined_branch_stage(
        states,
        2.0e-5,
        cell_width=0.04,
        params=params,
        boundaries=InclinedBranchBoundaryStates(state, state),
    )
    for candidate, original in zip(result.states, states):
        assert candidate.vector() == pytest.approx(original.vector(), abs=1.0e-16)
    assert all(source.vector() == (0.0, 0.0, 0.0, 0.0) for source in result.cell_sources)
    _assert_ledger_closed(result)


def test_horizontal_low_density_limit_reduces_to_saint_venant_celerity() -> None:
    # A tiny but positive reference pressure makes rho_g/rho_l negligible
    # without any density floor.  At zero gauge pressure and zero slip,
    # Eq. (3) tends to g/T for theta=0.
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=0.0,
        reference_pressure=1.0e-7,
    )
    state = _state_from_primitive(params, liquid_fraction=0.36)
    primitive = primitive_state(state, params)
    expected_lambda = (
        (params.liquid_density - primitive.gas_density)
        / params.liquid_density
        * params.gravity
        / primitive.top_width
    )
    expected_saint_venant = params.gravity / primitive.top_width
    assert primitive.lambda_d == pytest.approx(expected_lambda, rel=2.0e-15)
    assert primitive.lambda_d == pytest.approx(
        expected_saint_venant, rel=2.0e-12
    )
    assert primitive.liquid_celerity == pytest.approx(
        math.sqrt(expected_saint_venant * state.liquid_area), rel=2.0e-12
    )


def test_vertical_limit_has_zero_transverse_geometry_and_exact_axial_gravity() -> None:
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=0.5 * math.pi,
    )
    # Positive gauge pressure keeps the liquid block strictly hyperbolic while
    # the source itself is checked independently of any time integration.
    gas_density = (params.reference_pressure + 2_000.0) / params.gas_sound_speed**2
    state = _state_from_primitive(
        params,
        liquid_fraction=0.61,
        gas_density=gas_density,
    )
    primitive = primitive_state(state, params)
    source = cell_source(state, state, state, 0.02, params)
    assert params.cosine == 0.0
    assert params.sine == 1.0
    assert primitive.zeta == 0.0
    assert source.gas_momentum == pytest.approx(
        -state.gas_mass * params.gravity, rel=2.0e-15
    )
    assert source.liquid_momentum == pytest.approx(
        -(
            1.0 - primitive.gas_density / params.liquid_density
        )
        * state.liquid_area
        * params.gravity,
        rel=2.0e-15,
    )


def test_periodic_horizontal_stage_strictly_conserves_all_four_inventories() -> None:
    params = InclinedTwoFluidParameters(diameter=0.094, inclination=0.0)
    states = (
        _state_from_primitive(
            params,
            liquid_fraction=0.45,
            gas_density=params.reference_gas_density,
            gas_velocity=0.04,
            liquid_velocity=0.01,
        ),
        _state_from_primitive(
            params,
            liquid_fraction=0.45,
            gas_density=params.reference_gas_density,
            gas_velocity=-0.02,
            liquid_velocity=0.00,
        ),
        _state_from_primitive(
            params,
            liquid_fraction=0.45,
            gas_density=params.reference_gas_density,
            gas_velocity=0.01,
            liquid_velocity=-0.01,
        ),
    )
    result = euler_inclined_branch_stage(
        states,
        1.0e-6,
        cell_width=0.03,
        params=params,
        # These ghost states make the two physical boundary faces identical:
        # both evaluate the ordered pair (last, first).
        boundaries=InclinedBranchBoundaryStates(states[-1], states[0]),
    )
    assert result.face_fluxes[0] == result.face_fluxes[-1]
    assert result.ledger.integrated_source.vector() == pytest.approx(
        (0.0, 0.0, 0.0, 0.0), abs=1.0e-16
    )
    assert result.ledger.final.vector() == pytest.approx(
        result.ledger.initial.vector(), abs=3.0e-16
    )
    _assert_ledger_closed(result)


def test_face_flux_has_one_declared_method_and_no_fallback_path() -> None:
    params = InclinedTwoFluidParameters(diameter=0.094, inclination=0.0)
    left = _state_from_primitive(params, liquid_fraction=0.4, gas_velocity=0.02)
    right = _state_from_primitive(params, liquid_fraction=0.5, gas_velocity=-0.01)
    result = block_rusanov_flux(left, right, params)
    assert result.diagnostics.method == NUMERICAL_FLUX
    assert RIEMANN_FALLBACK_AVAILABLE is False
    assert result.diagnostics.fallback_available is False
    assert result.diagnostics.fallback_used is False


def test_near_full_area_and_tiny_positive_density_are_not_bounded() -> None:
    tiny_density = 1.0e-9
    gas_sound_speed = 290.0
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=0.0,
        gas_sound_speed=gas_sound_speed,
        reference_pressure=tiny_density * gas_sound_speed**2,
    )
    state = _state_from_primitive(
        params,
        liquid_fraction=0.999999,
        gas_density=tiny_density,
    )
    primitive = primitive_state(state, params)
    assert primitive.gas_area == pytest.approx(
        params.full_area * 1.0e-6, rel=2.0e-10
    )
    assert primitive.gas_density == pytest.approx(tiny_density, rel=2.0e-10)
    assert primitive.lambda_d > 0.0
    assert all(math.isfinite(value) for value in physical_flux(state, params).vector())


def test_negative_lambda_fails_instead_of_using_a_celerity_floor() -> None:
    params = InclinedTwoFluidParameters(diameter=0.094, inclination=0.0)
    unstable = _state_from_primitive(
        params,
        liquid_fraction=0.5,
        gas_velocity=80.0,
        liquid_velocity=-20.0,
    )
    diagnostic = primitive_state(unstable, params, require_hyperbolic=False)
    assert diagnostic.lambda_d < 0.0
    with pytest.raises(LossOfHyperbolicityError, match="Lambda_d < 0"):
        primitive_state(unstable, params)
    with pytest.raises(LossOfHyperbolicityError):
        block_rusanov_flux(unstable, unstable, params)


def test_vertical_atmospheric_zero_slip_keeps_exact_neutral_celerity() -> None:
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=0.5 * math.pi,
    )
    state = _state_from_primitive(params, liquid_fraction=0.5)
    primitive = primitive_state(state, params)
    assert primitive.lambda_d == 0.0
    assert primitive.liquid_celerity == 0.0
    assert primitive.neutral_ikh_state is True


def test_cfl_violation_rejects_without_mutating_input_tuple() -> None:
    params = InclinedTwoFluidParameters(
        diameter=0.094,
        inclination=0.0,
        maximum_cfl=0.5,
    )
    state = _state_from_primitive(params, liquid_fraction=0.5)
    states = (state, state)
    before = tuple(item.vector() for item in states)
    with pytest.raises(CFLViolationError):
        euler_inclined_branch_stage(
            states,
            1.0,
            cell_width=0.02,
            params=params,
            boundaries=InclinedBranchBoundaryStates(state, state),
        )
    assert tuple(item.vector() for item in states) == before


def test_invalid_stratified_area_fails_instead_of_clipping() -> None:
    params = InclinedTwoFluidParameters(diameter=0.094, inclination=0.0)
    state = InclinedTwoFluidState(1.0e-4, 0.0, params.full_area, 0.0)
    with pytest.raises(StateAdmissibilityError, match="positive gas area"):
        primitive_state(state, params)
