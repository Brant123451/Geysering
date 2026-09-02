"""Tests for the independent Case-A capacity-pressure active-set projector."""

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
    _solve_unilateral_capacity_qp_dual,
    project_capacity_pressure_active_set,
)


RHO = 998.0
DZ = 0.02
DT = 0.001


def test_bottom_accepted_outflow_removes_hidden_cell0_downward_momentum() -> None:
    """The v68-type -4.6 L/s cell state may not bypass a -0.63 L/s T face."""

    area = 6.0e-3
    result = project_capacity_pressure_active_set(
        upward_area=[0.0],
        upward_discharge=[0.0],
        downward_area=[area],
        downward_discharge=[-4.60e-3],
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.63e-3,
        top_downward_rate=0.63e-3,
        liquid_capacity_area=[area],
        current_liquid_area=[area],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True],
        enforce_boundary_cell_bulk_match=True,
    )

    assert result.corrected_downward_discharge[0] == pytest.approx(-0.63e-3)
    np.testing.assert_allclose(result.net_face_discharge, [-0.63e-3] * 2)
    assert result.bottom_bulk_anchor_residual == pytest.approx(0.0)
    assert result.common_velocity_increment[0] == pytest.approx(
        (-0.63e-3 + 4.60e-3) / area
    )
    assert result.ledger.bottom_pressure_impulse_on_liquid > 0.0
    assert result.ledger.coupled_momentum_residual == pytest.approx(
        0.0, abs=3.0e-16
    )


def test_projection_targets_actual_donor_faces_not_equal_cell_totals() -> None:
    """Equal cell totals can still hide a non-uniform donor-face net flux."""

    area = 6.0e-3
    half = 0.5 * area
    # Every provisional cell has Qu+Qd=0, so a block-mean cell reset is a no-op.
    result = project_capacity_pressure_active_set(
        upward_area=[half] * 3,
        upward_discharge=[0.60e-3, 0.50e-3, 0.50e-3],
        downward_area=[half] * 3,
        downward_discharge=[-0.60e-3, -0.50e-3, -0.50e-3],
        bottom_upward_rate=0.80e-3,
        bottom_downward_rate=0.50e-3,
        top_downward_rate=0.40e-3,
        liquid_capacity_area=[area] * 3,
        current_liquid_area=[area] * 3,
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True] * 3,
    )

    assert np.max(np.abs(result.common_velocity_increment)) > 0.0
    np.testing.assert_allclose(
        result.net_face_discharge,
        [0.30e-3] * 4,
        atol=2.0e-18,
    )
    assert np.count_nonzero(result.active_capacity_mask) >= 2
    assert result.maximum_active_constraint_residual < 2.0e-18
    assert result.maximum_packing_residual <= 2.0e-18


def test_common_pressure_keeps_both_corridors_and_exact_cellwise_slip() -> None:
    area = 6.0e-3
    au = 0.60 * area
    ad = 0.40 * area
    qu = 0.72e-3
    qd = -0.72e-3
    initial_slip = qu / au - qd / ad

    result = project_capacity_pressure_active_set(
        upward_area=[au],
        upward_discharge=[qu],
        downward_area=[ad],
        downward_discharge=[qd],
        bottom_upward_rate=1.50e-3,
        bottom_downward_rate=0.60e-3,
        top_downward_rate=0.0,
        liquid_capacity_area=[area],
        current_liquid_area=[area],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True],
        enforce_boundary_cell_bulk_match=True,
    )

    assert result.corrected_upward_discharge[0] > 0.0
    assert result.corrected_downward_discharge[0] < 0.0
    final_slip = (
        result.corrected_upward_discharge[0] / au
        - result.corrected_downward_discharge[0] / ad
    )
    assert final_slip == pytest.approx(initial_slip, abs=2.0e-15)
    assert (
        result.corrected_upward_discharge[0]
        + result.corrected_downward_discharge[0]
    ) == pytest.approx(0.90e-3)
    assert result.bottom_bulk_anchor_residual == pytest.approx(0.0)


def test_face_pressure_gradient_and_boundary_interface_ledger_close() -> None:
    areas = np.array([6.0e-3, 4.0e-3])
    result = project_capacity_pressure_active_set(
        upward_area=[0.0, 0.0],
        upward_discharge=[0.0, 0.0],
        downward_area=areas,
        downward_discharge=[-4.60e-3, -1.00e-3],
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.63e-3,
        top_downward_rate=0.63e-3,
        liquid_capacity_area=areas,
        current_liquid_area=areas,
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True, True],
    )

    pressure = result.face_pressure_impulse
    recovered_x = -(pressure[1:] - pressure[:-1]) / (RHO * DZ)
    np.testing.assert_allclose(recovered_x, result.common_velocity_increment)
    assert result.ledger.internal_area_pressure_impulse_on_liquid != 0.0
    assert result.ledger.pressure_decomposition_residual == pytest.approx(
        0.0, abs=3.0e-16
    )
    assert result.ledger.coupled_momentum_residual == pytest.approx(
        0.0, abs=3.0e-16
    )


def test_tensile_capacity_multiplier_is_released_and_cell_may_drain() -> None:
    area = 6.0e-3
    result = project_capacity_pressure_active_set(
        upward_area=[0.5 * area, area],
        upward_discharge=[0.0, 1.0e-3],
        downward_area=[0.0, 0.0],
        downward_discharge=[0.0, 0.0],
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.0,
        top_downward_rate=0.0,
        liquid_capacity_area=[area, area],
        current_liquid_area=[0.5 * area, area],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True, True],
    )

    assert not result.active_capacity_mask[1]
    assert result.capacity_multiplier[1] == 0.0
    np.testing.assert_array_equal(result.common_velocity_increment, [0.0, 0.0])
    assert result.predicted_liquid_area[1] < area
    assert result.iterations >= 1


def test_inactive_capacity_is_an_exact_noop() -> None:
    result = project_capacity_pressure_active_set(
        upward_area=[2.0e-3, 2.0e-3],
        upward_discharge=[0.20e-3, 0.20e-3],
        downward_area=[1.0e-3, 1.0e-3],
        downward_discharge=[-0.10e-3, -0.10e-3],
        bottom_upward_rate=0.20e-3,
        bottom_downward_rate=0.10e-3,
        top_downward_rate=0.10e-3,
        liquid_capacity_area=[6.0e-3, 6.0e-3],
        current_liquid_area=[3.0e-3, 3.0e-3],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True, True],
    )

    np.testing.assert_array_equal(result.common_velocity_increment, [0.0, 0.0])
    np.testing.assert_allclose(result.corrected_upward_discharge, [0.20e-3] * 2)
    np.testing.assert_allclose(result.corrected_downward_discharge, [-0.10e-3] * 2)
    assert not np.any(result.active_capacity_mask)
    assert result.ledger.liquid_physical_impulse == 0.0


def test_bottom_anchor_is_bulk_not_a_separate_downward_stream_overwrite() -> None:
    area = 6.0e-3
    result = project_capacity_pressure_active_set(
        upward_area=[0.5 * area],
        upward_discharge=[0.10e-3],
        downward_area=[0.5 * area],
        downward_discharge=[-1.00e-3],
        bottom_upward_rate=0.10e-3,
        bottom_downward_rate=0.20e-3,
        top_downward_rate=0.0,
        liquid_capacity_area=[area],
        current_liquid_area=[area],
        dt=0.01,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[False],
        enforce_boundary_cell_bulk_match=True,
    )

    assert result.common_velocity_increment[0] == pytest.approx(0.8e-3 / area)
    assert sum(
        (
            result.corrected_upward_discharge[0],
            result.corrected_downward_discharge[0],
        )
    ) == pytest.approx(-0.10e-3)
    assert result.corrected_downward_discharge[0] != pytest.approx(-0.20e-3)
    initial_slip = 0.10e-3 / (0.5 * area) - (-1.00e-3) / (0.5 * area)
    final_slip = (
        result.corrected_upward_discharge[0] / (0.5 * area)
        - result.corrected_downward_discharge[0] / (0.5 * area)
    )
    assert final_slip == pytest.approx(initial_slip)


def test_projection_is_idempotent_when_reapplied_to_its_corrected_state() -> None:
    area = 6.0e-3
    first = project_capacity_pressure_active_set(
        upward_area=[0.0],
        upward_discharge=[0.0],
        downward_area=[area],
        downward_discharge=[-4.60e-3],
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.63e-3,
        top_downward_rate=0.63e-3,
        liquid_capacity_area=[area],
        current_liquid_area=[area],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True],
        enforce_boundary_cell_bulk_match=True,
    )
    second = project_capacity_pressure_active_set(
        upward_area=[0.0],
        upward_discharge=first.corrected_upward_discharge,
        downward_area=[area],
        downward_discharge=first.corrected_downward_discharge,
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.63e-3,
        top_downward_rate=0.63e-3,
        liquid_capacity_area=[area],
        current_liquid_area=[area],
        dt=DT,
        dz=DZ,
        liquid_density=RHO,
        preserve_stopped_partition=[True],
        enforce_boundary_cell_bulk_match=True,
    )

    np.testing.assert_allclose(second.common_velocity_increment, [0.0], atol=2.0e-15)
    np.testing.assert_allclose(
        second.corrected_downward_discharge,
        first.corrected_downward_discharge,
    )
    assert second.maximum_kkt_stationarity_residual < 2.0e-15


def test_bottom_anchor_may_release_a_saturated_cell_that_drains() -> None:
    """Mandatory accepted-flow anchoring precedes unilateral capacity rows."""

    result = project_capacity_pressure_active_set(
        upward_area=[0.5],
        upward_discharge=[0.1],
        downward_area=[0.5],
        downward_discharge=[-1.0],
        bottom_upward_rate=0.1,
        bottom_downward_rate=0.2,
        top_downward_rate=0.0,
        liquid_capacity_area=[1.0],
        current_liquid_area=[1.0],
        dt=0.01,
        dz=0.02,
        liquid_density=1000.0,
        preserve_stopped_partition=[False],
        enforce_boundary_cell_bulk_match=True,
    )

    assert result.corrected_upward_discharge[0] == pytest.approx(0.5)
    assert result.corrected_downward_discharge[0] == pytest.approx(-0.6)
    assert result.common_velocity_increment[0] == pytest.approx(0.8)
    assert result.bottom_bulk_anchor_residual == pytest.approx(0.0)
    assert not result.active_capacity_mask[0]
    assert result.predicted_liquid_area[0] < 1.0


def test_dry_gap_does_not_erase_pressure_of_upper_wet_component() -> None:
    result = project_capacity_pressure_active_set(
        upward_area=[1.0, 0.0, 1.0],
        upward_discharge=[0.0, 0.0, 0.0],
        downward_area=[0.0, 0.0, 0.0],
        downward_discharge=[0.0, 0.0, 0.0],
        bottom_upward_rate=0.0,
        bottom_downward_rate=0.0,
        top_downward_rate=1.0,
        liquid_capacity_area=[2.0, 0.0, 1.0],
        current_liquid_area=[1.0, 0.0, 1.0],
        dt=0.01,
        dz=0.02,
        liquid_density=1000.0,
        preserve_stopped_partition=[False, False, False],
    )

    assert result.common_velocity_increment[2] != 0.0
    recovered = -np.diff(result.face_pressure_impulse) / (1000.0 * 0.02)
    np.testing.assert_allclose(recovered[[0, 2]], result.common_velocity_increment[[0, 2]])
    assert result.ledger.coupled_momentum_residual == pytest.approx(
        0.0, abs=3.0e-13
    )


def test_incompatible_sign_guard_yields_to_capacity_then_topology() -> None:
    """Primary packing may require a preserved falling label to cross zero."""

    result = project_capacity_pressure_active_set(
        upward_area=[0.5, 0.5],
        upward_discharge=[0.1, 0.15],
        downward_area=[0.5, 0.5],
        downward_discharge=[-0.1, -0.1],
        bottom_upward_rate=0.3,
        bottom_downward_rate=0.0,
        top_downward_rate=0.0,
        liquid_capacity_area=[1.0, 1.0],
        current_liquid_area=[1.0, 1.0],
        dt=0.01,
        dz=0.02,
        liquid_density=1000.0,
        preserve_stopped_partition=[False, True],
        enforce_boundary_cell_bulk_match=True,
    )

    # Mandatory bulk and the active capacity row require x1=0.3, whereas the
    # old q_down<=0 guard would require x1<=0.2.  Conservation wins; the host
    # topology transfer will relabel the positive downward-labelled branch.
    np.testing.assert_allclose(result.common_velocity_increment, [0.3, 0.3])
    assert result.corrected_downward_discharge[1] == pytest.approx(0.05)
    np.testing.assert_allclose(result.net_face_discharge, [0.3, 0.3, 0.3])
    assert result.maximum_packing_residual <= 2.0e-15
    assert result.bottom_bulk_anchor_residual == pytest.approx(0.0)
    assert result.ledger.coupled_momentum_residual == pytest.approx(
        0.0, abs=3.0e-13
    )


def test_degenerate_two_wet_blocks_release_only_a_slack_working_row() -> None:
    """A storage gap may make the equality basis dependent, not the QP infeasible."""

    capacity = np.array([1.0, 0.4, 0.95, 0.4, 0.7])
    result = project_capacity_pressure_active_set(
        upward_area=[1.0, 0.32, 0.7, 0.0, 0.35],
        upward_discharge=[0.60, 0.10, 0.59, 0.0, 0.27],
        downward_area=[0.0, 0.08, 0.0, 0.4, 0.35],
        downward_discharge=[0.0, -0.05, 0.0, -0.12, -0.28],
        bottom_upward_rate=0.52,
        bottom_downward_rate=0.23,
        top_downward_rate=0.0,
        liquid_capacity_area=capacity,
        current_liquid_area=[1.0, 0.4, 0.7, 0.4, 0.7],
        dt=0.05,
        dz=0.10,
        liquid_density=1000.0,
        preserve_stopped_partition=[False] * 5,
        enforce_boundary_cell_bulk_match=True,
    )

    # Cells 0--1 and 3--4 start saturated, with storage only in cell 2.  One
    # capacity row enters a rank-deficient equality basis, but is slack in the
    # feasible inequality solution and can therefore be removed exactly.
    assert result.working_set_capacity_releases == 1
    np.testing.assert_array_equal(
        result.active_capacity_mask,
        [True, False, False, True, False],
    )
    assert np.all(result.predicted_liquid_area <= capacity + 2.0e-14)
    assert result.maximum_packing_residual <= 2.0e-14
    assert result.maximum_active_constraint_residual <= 2.0e-14
    assert result.bottom_bulk_anchor_residual == pytest.approx(0.0)
    assert result.ledger.coupled_momentum_residual == pytest.approx(
        0.0, abs=3.0e-12
    )


def test_dual_active_set_pivots_multiple_violated_rows_to_slack() -> None:
    """The formal no-anchor path resolves a multi-row degenerate basis finitely."""

    capacity = np.array([1.0, 1.0, 1.3, 0.4, 1.0, 0.4])
    current = np.array([1.0, 1.0, 1.0, 0.4, 1.0, 0.4])
    result = project_capacity_pressure_active_set(
        upward_area=[0.0, 0.8, 0.2, 0.4, 0.5, 0.0],
        upward_discharge=[0.0, 0.12, 0.01, 0.21, 0.30, 0.0],
        downward_area=[1.0, 0.2, 0.8, 0.0, 0.5, 0.4],
        downward_discharge=[-0.44, -0.17, -0.64, 0.0, -0.24, -0.22],
        bottom_upward_rate=0.60,
        bottom_downward_rate=0.36,
        top_downward_rate=0.0,
        liquid_capacity_area=capacity,
        current_liquid_area=current,
        dt=0.05,
        dz=0.10,
        liquid_density=1000.0,
        preserve_stopped_partition=None,
        enforce_boundary_cell_bulk_match=False,
    )

    # Rows 0, 1, 3 and 5 are violated at x=0.  The optimal unilateral solve
    # instead activates 0, 2 and 5, leaving both 1 and 3 with finite slack.
    np.testing.assert_array_equal(
        result.active_capacity_mask,
        [True, False, True, False, False, True],
    )
    area_slack = capacity - result.predicted_liquid_area
    assert area_slack[1] > 0.09
    assert area_slack[3] > 0.04
    assert result.iterations <= 8
    assert np.all(result.capacity_multiplier >= 0.0)
    assert result.maximum_packing_residual <= 2.0e-14
    assert result.maximum_kkt_stationarity_residual <= 5.0e-13
    assert result.maximum_complementarity_residual <= 1.0e-11


def test_dual_active_set_rejects_violated_zero_mobility_row() -> None:
    with pytest.raises(
        CapacityPressureRecouplingRequired,
        match="zero-mobility capacity row",
    ):
        project_capacity_pressure_active_set(
            upward_area=[0.0],
            upward_discharge=[0.0],
            downward_area=[1.0],
            downward_discharge=[0.0],
            bottom_upward_rate=1.0,
            bottom_downward_rate=0.0,
            top_downward_rate=0.0,
            top_upward_rate=0.0,
            liquid_capacity_area=[1.0],
            current_liquid_area=[1.0],
            dt=0.01,
            dz=0.02,
            liquid_density=1000.0,
            preserve_stopped_partition=None,
            enforce_boundary_cell_bulk_match=False,
        )


def test_ill_conditioned_dual_keeps_direct_restricted_primal() -> None:
    """Large cancelling pressure multipliers must not reconstruct the primal."""

    epsilon = 1.0e-8
    matrix = np.array(
        [
            [1.0, 0.0],
            [-1.0, epsilon],
        ]
    )
    rhs = np.array([1.0, -1.0 - epsilon])
    (
        increment,
        multiplier,
        _,
        _,
        _,
        stationarity,
        _,
    ) = _solve_unilateral_capacity_qp_dual(
        mass=np.ones(2),
        matrix=matrix,
        right_hand_side=rhs,
        rate_tolerance=1.0e-10,
        velocity_scale=2.0,
        maximum_sweeps=100,
    )

    # The directly solved primal satisfies both faces to roundoff.  Rebuilding
    # it from two O(1e8) cancelling dual tractions loses O(1e-8), which is the
    # v88 failure mode; stationarity is audited but does not replace x.
    assert np.max(matrix @ increment - rhs) <= 5.0e-15
    reconstructed = -(matrix.T @ multiplier)
    assert np.max(matrix @ reconstructed - rhs) > 1.0e-9
    assert stationarity <= 2.0e-8
