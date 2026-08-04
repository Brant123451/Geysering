from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from casea_horizontal_liquid_operator import PressurePotentialState  # noqa: E402
from casea_post_t_liquid_stage import BranchPressureEvaluation  # noqa: E402
from casea_post_t_sideport_liquid_stage import (  # noqa: E402
    PostTSidePortGeometry,
    circular_side_port_weights,
    post_t_sideport_liquid_stage_rhs,
)


def _constant_pressure_callback(
    *,
    horizontal_pressure: float,
    vertical_pressure: float,
    rho: float,
    celerity: float = 28.0,
):
    def callback(branch, area, discharge, gas_mass, gas_momentum):
        area = np.asarray(area, dtype=float)
        q = np.asarray(discharge, dtype=float)
        pressure_abs = (
            horizontal_pressure if branch == "horizontal" else vertical_pressure
        )
        pressure = PressurePotentialState(
            potential=pressure_abs * area / rho,
            derivative=np.full_like(area, celerity**2),
            discharge_derivative=2.0 * q / area,
            celerity=np.full_like(area, celerity),
            eigenvalue_minus=q / area - celerity,
            eigenvalue_plus=q / area + celerity,
            lambda_value=np.zeros_like(area),
            lambda_derivative=np.zeros_like(area),
            stratified=np.zeros_like(area, dtype=bool),
        )
        return BranchPressureEvaluation(
            pressure=pressure,
            face_pressure_abs=np.full_like(area, pressure_abs),
            node_pressure_offset=np.zeros_like(area),
            momentum_source=np.zeros_like(area),
            potential_pressure_abs=np.full_like(area, pressure_abs),
        )

    return callback


def _case(*, horizontal_pressure=101_325.0, vertical_pressure=101_325.0):
    dx = 0.04
    dz = 0.02
    nh = 100
    nv = 18
    ah = np.full(nh, 0.0060)
    qh = np.zeros(nh)
    av = np.full(nv, 0.0020)
    qv = np.zeros(nv)
    mgh = np.zeros(nh)
    jgh = np.zeros(nh)
    mgv = np.zeros(nv)
    jgv = np.zeros(nv)
    geometry = PostTSidePortGeometry(
        horizontal_cell_width=dx,
        vertical_cell_width=dz,
        liquid_density=998.0,
        junction_center_x=3.516,
        opening_diameter=0.0571,
    )
    callback = _constant_pressure_callback(
        horizontal_pressure=horizontal_pressure,
        vertical_pressure=vertical_pressure,
        rho=geometry.liquid_density,
    )
    return (ah, qh, av, qv, mgh, jgh, mgv, jgv), geometry, callback


def test_circular_opening_weights_are_exact_and_grid_independent() -> None:
    for dx in (0.08, 0.04, 0.02, 0.01):
        ncell = int(np.ceil(4.006 / dx))
        exact_dx = 4.006 / ncell
        weights = circular_side_port_weights(
            ncell,
            cell_width=exact_dx,
            centre_x=3.516,
            diameter=0.0571,
        )
        assert abs(float(np.sum(weights)) - 1.0) < 5.0e-14
        assert np.all(weights >= 0.0)
        assert np.count_nonzero(weights) >= 1
        cell_centres = (np.arange(ncell) + 0.5) * exact_dx
        # The source is stored as a cell average, so its represented first
        # moment is second-order accurate and can differ from the continuous
        # circular-mouth centroid by at most half a cell on a coarse grid.
        assert (
            abs(float(np.dot(weights, cell_centres)) - 3.516)
            <= 0.5 * exact_dx + 1.0e-14
        )


def test_static_equal_pressure_has_zero_side_port_mass_exchange() -> None:
    fields, geometry, callback = _case()
    rhs = post_t_sideport_liquid_stage_rhs(
        *fields,
        geometry=geometry,
        pressure_callback=callback,
        vertical_active_count=18,
    )
    assert abs(rhs.diagnostics.vertical_volume_flux) < 1.0e-16
    assert abs(rhs.diagnostics.liquid_volume_residual) < 1.0e-18
    assert np.max(np.abs(rhs.rhs_horizontal_area)) < 1.0e-15
    assert np.max(np.abs(rhs.rhs_vertical_area)) < 1.0e-15


def test_pressure_driven_exchange_is_equal_and_opposite() -> None:
    fields, geometry, callback = _case(
        horizontal_pressure=103_000.0,
        vertical_pressure=101_325.0,
    )
    rhs = post_t_sideport_liquid_stage_rhs(
        *fields,
        geometry=geometry,
        pressure_callback=callback,
        vertical_active_count=18,
    )
    diagnostics = rhs.diagnostics
    assert diagnostics.vertical_volume_flux > 0.0
    horizontal = float(np.sum(rhs.rhs_horizontal_area) * geometry.horizontal_cell_width)
    vertical = float(np.sum(rhs.rhs_vertical_area) * geometry.vertical_cell_width)
    assert abs(horizontal + vertical) < 3.0e-18
    assert abs(horizontal + diagnostics.vertical_volume_flux) < 3.0e-18
    assert abs(diagnostics.liquid_volume_residual) < 3.0e-18


def test_downward_tower_inflow_adds_no_prescribed_horizontal_momentum() -> None:
    fields, geometry, callback = _case(
        horizontal_pressure=101_325.0,
        vertical_pressure=103_000.0,
    )
    ah, qh, av, qv, mgh, jgh, mgv, jgv = fields
    qh[:] = 2.0e-4
    rhs = post_t_sideport_liquid_stage_rhs(
        ah, qh, av, qv, mgh, jgh, mgv, jgv,
        geometry=geometry,
        pressure_callback=callback,
        vertical_active_count=18,
    )
    assert rhs.diagnostics.vertical_volume_flux < 0.0
    assert rhs.diagnostics.horizontal_wall_momentum_reaction_rate == 0.0


def test_upward_outflow_carries_local_axial_momentum_only() -> None:
    fields, geometry, callback = _case(
        horizontal_pressure=103_000.0,
        vertical_pressure=101_325.0,
    )
    ah, qh, av, qv, mgh, jgh, mgv, jgv = fields
    qh[:] = 2.0e-4
    rhs = post_t_sideport_liquid_stage_rhs(
        ah, qh, av, qv, mgh, jgh, mgv, jgv,
        geometry=geometry,
        pressure_callback=callback,
        vertical_active_count=18,
    )
    q_side = rhs.diagnostics.vertical_volume_flux
    expected = -q_side * float(qh[0] / ah[0])
    assert q_side > 0.0
    assert abs(
        rhs.diagnostics.horizontal_wall_momentum_reaction_rate - expected
    ) < 3.0e-16
