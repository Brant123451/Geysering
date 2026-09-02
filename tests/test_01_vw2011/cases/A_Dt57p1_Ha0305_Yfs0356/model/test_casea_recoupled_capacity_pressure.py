"""Tests for the independent mouth/capacity pressure recoupling prototype."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_capacity_pressure_projection import (  # noqa: E402
    CapacityPressureRecouplingRequired,
    project_capacity_pressure_active_set,
)
from casea_recoupled_capacity_pressure import (  # noqa: E402
    flux_inertance_from_characteristic,
    flux_inertance_from_plug,
    project_mouth_and_capacity_pressure,
    project_state_mouth_and_capacity_pressure,
)
from casea_vertical_twostream_fv import VerticalTwoStreamState  # noqa: E402


RHO = 998.0
AREA = 6.0e-3
DZ = 0.02
DT = 1.0e-3


def _project(**overrides: object):
    arguments: dict[str, object] = dict(
        upward_area=[0.0],
        upward_discharge=[0.0],
        downward_area=[AREA],
        downward_discharge=[-0.20e-3],
        candidate_bottom_upward_rate=0.60e-3,
        candidate_bottom_downward_rate=0.20e-3,
        bottom_upward_flux_inertance=flux_inertance_from_plug(
            liquid_density=RHO, effective_length=0.20, flow_area=AREA
        ),
        bottom_downward_flux_inertance=flux_inertance_from_characteristic(
            liquid_density=RHO, celerity=2.0, time_step=DT, flow_area=AREA
        ),
        bottom_upward_characteristic_area=AREA,
        bottom_downward_characteristic_area=AREA,
        bottom_downward_donor_rate_capacity=0.20e-3,
        top_downward_rate=0.0,
        liquid_capacity_area=[AREA],
        current_liquid_area=[AREA],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
    )
    arguments.update(overrides)
    return project_mouth_and_capacity_pressure(**arguments)


def test_adjustable_bottom_characteristic_removes_fixed_flux_farkas_case() -> None:
    """A saturated all-falling cell has no column mobility for fixed inflow."""

    with pytest.raises(
        CapacityPressureRecouplingRequired,
        match="zero-mobility capacity row",
    ):
        project_capacity_pressure_active_set(
            upward_area=[0.0],
            upward_discharge=[0.0],
            downward_area=[AREA],
            downward_discharge=[0.0],
            bottom_upward_rate=0.60e-3,
            bottom_downward_rate=0.20e-3,
            top_downward_rate=0.0,
            liquid_capacity_area=[AREA],
            current_liquid_area=[AREA],
            dt=DT,
            dz=DZ,
            liquid_density=RHO,
        )

    result = _project()

    # Here the loss-reduced candidate already equals the resolved donor, so the
    # conservative response is to reduce only the incoming branch.
    assert result.final_bottom_upward_rate == pytest.approx(0.20e-3)
    assert result.final_bottom_downward_rate == pytest.approx(0.20e-3)
    assert result.final_bottom_upward_rate == pytest.approx(
        result.final_bottom_downward_rate
    )
    assert result.final_bottom_net_rate == pytest.approx(0.0, abs=2.0e-18)
    assert result.rejected_bottom_upward_rate == pytest.approx(0.40e-3)
    assert result.rejected_bottom_downward_rate == pytest.approx(0.0)
    corrected_downward_donor = max(-result.corrected_downward_discharge[0], 0.0)
    assert result.final_bottom_downward_rate <= corrected_downward_donor + 2.0e-15
    assert result.maximum_packing_residual <= 2.0e-15
    assert result.maximum_bound_residual <= 2.0e-15
    assert result.maximum_downward_donor_residual <= 2.0e-15
    assert result.ledger.bottom_capacity_pressure_impulse > 0.0
    assert result.ledger.bottom_reaction_impulse_on_tnode < 0.0


def test_physical_inertances_split_capacity_response_analytically() -> None:
    """Capacity pressure shares work between column inertia and mouth inertia."""

    mouth_inertance = flux_inertance_from_plug(
        liquid_density=RHO, effective_length=0.10, flow_area=AREA
    )
    result = _project(
        upward_area=[AREA],
        downward_area=[0.0],
        candidate_bottom_upward_rate=0.40e-3,
        candidate_bottom_downward_rate=0.0,
        bottom_upward_flux_inertance=mouth_inertance,
        bottom_downward_flux_inertance=mouth_inertance,
        downward_discharge=[0.0],
    )

    # Active equality: du - A*x = -Q.  Minimising
    # 0.5*(rho*dz*A)*x^2 + 0.5*I*du^2 gives this closed form.
    cell_mass = RHO * DZ * AREA
    pressure_impulse = 0.40e-3 / (1.0 / mouth_inertance + AREA**2 / cell_mass)
    expected_du = -pressure_impulse / mouth_inertance
    expected_x = pressure_impulse * AREA / cell_mass
    assert result.final_bottom_upward_rate == pytest.approx(
        0.40e-3 + expected_du, rel=2.0e-11
    )
    assert result.common_velocity_increment[0] == pytest.approx(
        expected_x, rel=2.0e-11
    )
    assert result.capacity_pressure_impulse[0] == pytest.approx(
        pressure_impulse, rel=2.0e-11
    )
    assert result.maximum_kkt_stationarity_residual < 2.0e-12


def test_inactive_capacity_is_exact_candidate_noop() -> None:
    result = _project(
        liquid_capacity_area=[2.0 * AREA],
        current_liquid_area=[AREA],
    )

    assert result.final_bottom_upward_rate == pytest.approx(0.60e-3)
    assert result.final_bottom_downward_rate == pytest.approx(0.20e-3)
    np.testing.assert_array_equal(result.common_velocity_increment, [0.0])
    assert not np.any(result.active_capacity_mask)
    assert result.ledger.bottom_capacity_pressure_impulse == 0.0


def test_capacity_pressure_can_accelerate_downflow_below_donor_limit() -> None:
    """The pressure reaction is not an artificial Qdown<=candidate clip."""

    result = _project(
        downward_discharge=[-1.0e-3],
        bottom_downward_donor_rate_capacity=1.0e-3,
    )

    expected_rate = 0.596039603960396e-3
    assert result.final_bottom_upward_rate == pytest.approx(expected_rate)
    assert result.final_bottom_downward_rate == pytest.approx(expected_rate)
    assert result.final_bottom_downward_rate > result.candidate_bottom_downward_rate
    assert result.final_bottom_downward_rate < 1.0e-3
    assert result.downward_upper_bound_multiplier == pytest.approx(0.0, abs=2.0e-15)
    assert result.rejected_bottom_downward_rate < 0.0
    assert result.ledger.rejected_downward_volume == pytest.approx(
        DT * result.rejected_bottom_downward_rate
    )


def test_bounds_volume_and_momentum_ledgers_are_explicit() -> None:
    result = _project()

    assert 0.0 <= result.final_bottom_upward_rate <= 0.60e-3
    corrected_downward_donor = max(-result.corrected_downward_discharge[0], 0.0)
    assert 0.0 <= result.final_bottom_downward_rate <= (
        corrected_downward_donor + 2.0e-15
    )
    assert result.ledger.volume_balance_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.upward_gross_volume == pytest.approx(
        DT * result.final_bottom_upward_rate
    )
    assert result.ledger.downward_gross_volume == pytest.approx(
        DT * result.final_bottom_downward_rate
    )
    assert result.ledger.rejected_upward_volume == pytest.approx(
        DT * result.rejected_bottom_upward_rate
    )
    assert result.ledger.rejected_downward_volume == pytest.approx(
        DT * result.rejected_bottom_downward_rate
    )
    assert result.ledger.column_pressure_decomposition_residual == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert np.isfinite(result.ledger.characteristic_momentum_impulse_upward)
    assert np.isfinite(result.ledger.accepted_convective_momentum_flux_upward)
    assert result.maximum_complementarity_residual < 2.0e-12


def test_capacity_pressure_stops_but_does_not_reverse_directional_labels() -> None:
    result = _project(
        upward_area=[0.25 * AREA],
        upward_discharge=[0.0],
        downward_area=[0.75 * AREA],
        downward_discharge=[-0.10e-3],
        candidate_bottom_upward_rate=0.60e-3,
        candidate_bottom_downward_rate=0.0,
    )

    assert result.corrected_upward_discharge[0] >= -2.0e-15
    assert result.corrected_downward_discharge[0] <= 2.0e-15
    assert result.maximum_directional_sign_residual <= 2.0e-15


def test_inertance_helpers_reject_nonphysical_inputs() -> None:
    with pytest.raises(ValueError):
        flux_inertance_from_plug(
            liquid_density=RHO, effective_length=0.0, flow_area=AREA
        )
    with pytest.raises(ValueError):
        flux_inertance_from_characteristic(
            liquid_density=RHO, celerity=2.0, time_step=DT, flow_area=0.0
        )


def test_state_wrapper_returns_one_sign_admissible_atomic_state() -> None:
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.0],
        upward_discharge=[0.0],
        downward_area=[AREA],
        downward_discharge=[-0.20e-3],
    )
    arguments = dict(
        candidate_bottom_upward_rate=0.60e-3,
        candidate_bottom_downward_rate=0.20e-3,
        bottom_upward_flux_inertance=flux_inertance_from_plug(
            liquid_density=RHO, effective_length=0.20, flow_area=AREA
        ),
        bottom_downward_flux_inertance=flux_inertance_from_characteristic(
            liquid_density=RHO, celerity=2.0, time_step=DT, flow_area=AREA
        ),
        bottom_upward_characteristic_area=AREA,
        bottom_downward_characteristic_area=AREA,
        top_downward_rate=0.0,
        liquid_capacity_area=[AREA],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
    )

    result = project_state_mouth_and_capacity_pressure(state, **arguments)

    assert result.state.upward_discharge[0] >= 0.0
    assert result.state.downward_discharge[0] <= 0.0
    assert result.projection.final_bottom_net_rate == pytest.approx(0.0)
    assert result.outer_iterations == 1
