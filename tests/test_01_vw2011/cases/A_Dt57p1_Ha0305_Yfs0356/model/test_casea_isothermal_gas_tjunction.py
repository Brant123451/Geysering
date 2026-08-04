import math

import pytest

from casea_isothermal_gas_tjunction import (
    GasBranchTrace,
    GasTJunctionError,
    NoAdmissibleSubsonicJunctionError,
    NonPositiveGasStateError,
    SupersonicGasTraceError,
    solve_isothermal_gas_tjunction,
    velocity_in_outward_coordinate,
)


def _trace(
    density: float,
    velocity: float,
    area: float,
    tracer: float | None = None,
) -> GasBranchTrace:
    return GasBranchTrace(density, velocity, area, tracer)


def test_quiescent_equal_pressure_state_is_exactly_stationary() -> None:
    c = 12.0
    rho = 1.27
    result = solve_isothermal_gas_tjunction(
        _trace(rho, 0.0, 0.0060),
        _trace(rho, 0.0, 0.0040),
        _trace(rho, 0.0, 0.0015),
        sound_speed=c,
    )

    assert result.common_density == rho
    assert result.common_pressure == rho * c * c
    assert result.mass_residual == 0.0
    for branch in result.branches:
        assert branch.outward_velocity == 0.0
        assert branch.mass_flux == 0.0
        assert branch.momentum_flux == branch.area * result.common_pressure
        assert branch.tracer_flux is None


def test_unequal_pressure_traces_close_mass_without_flow_allocation() -> None:
    c = 300.0
    result = solve_isothermal_gas_tjunction(
        _trace(1.30, -8.0, 0.00693),
        _trace(1.00, 3.0, 0.00693),
        _trace(0.90, 1.0, 0.00256),
        sound_speed=c,
    )

    scale = math.fsum(abs(branch.mass_flux) for branch in result.branches)
    assert result.common_pressure == result.common_density * c * c
    assert abs(result.mass_residual) <= 2.0e-14 * scale
    assert any(branch.mass_flux < 0.0 for branch in result.branches)
    assert any(branch.mass_flux > 0.0 for branch in result.branches)
    for trace, branch in zip(
        (
            _trace(1.30, -8.0, 0.00693),
            _trace(1.00, 3.0, 0.00693),
            _trace(0.90, 1.0, 0.00256),
        ),
        result.branches,
    ):
        characteristic_velocity = trace.outward_velocity + c * math.log(
            result.common_density / trace.density
        )
        assert math.isclose(
            branch.outward_velocity,
            characteristic_velocity,
            rel_tol=0.0,
            abs_tol=3.0e-13,
        )
        assert math.isclose(
            branch.mass_flux,
            result.common_density * trace.area * branch.outward_velocity,
            rel_tol=2.0e-16,
            abs_tol=0.0,
        )


def test_west_axis_reversal_matches_direct_outward_coordinate() -> None:
    c = 10.0
    west_outward = velocity_in_outward_coordinate(2.0, -1)
    east_outward = velocity_in_outward_coordinate(1.0, +1)
    vertical_outward = velocity_in_outward_coordinate(1.0, +1)

    converted = solve_isothermal_gas_tjunction(
        _trace(1.0, west_outward, 1.0),
        _trace(1.0, east_outward, 1.0),
        _trace(1.0, vertical_outward, 1.0),
        sound_speed=c,
    )
    direct = solve_isothermal_gas_tjunction(
        _trace(1.0, -2.0, 1.0),
        _trace(1.0, +1.0, 1.0),
        _trace(1.0, +1.0, 1.0),
        sound_speed=c,
    )

    assert converted == direct
    assert converted.west.mass_flux < 0.0
    assert converted.east.mass_flux > 0.0
    assert converted.vertical.mass_flux > 0.0


def test_strictly_positive_tiny_branch_area_is_supported() -> None:
    result = solve_isothermal_gas_tjunction(
        _trace(1.01, -0.2, 6.93e-3),
        _trace(0.99, +0.1, 6.93e-3),
        _trace(1.00, 0.0, 1.0e-14),
        sound_speed=20.0,
    )

    scale = math.fsum(abs(branch.mass_flux) for branch in result.branches)
    assert result.vertical.area == 1.0e-14
    assert math.isfinite(result.vertical.mass_flux)
    assert abs(result.mass_residual) <= 3.0e-14 * scale


def test_tracer_is_mixed_from_inflow_and_upwinded_to_outflows() -> None:
    result = solve_isothermal_gas_tjunction(
        _trace(1.0, -2.0, 1.0, 0.8),
        _trace(1.0, +1.0, 1.0, 0.1),
        _trace(1.0, +1.0, 1.0, 0.2),
        sound_speed=10.0,
    )

    assert result.junction_tracer_fraction == 0.8
    assert result.west.tracer_flux == -1.6
    assert result.east.tracer_flux == 0.8
    assert result.vertical.tracer_flux == 0.8
    assert result.tracer_residual == 0.0


def test_partial_tracer_input_is_rejected() -> None:
    with pytest.raises(GasTJunctionError, match="all branches or none"):
        solve_isothermal_gas_tjunction(
            _trace(1.0, -1.0, 1.0, 0.5),
            _trace(1.0, +0.5, 1.0),
            _trace(1.0, +0.5, 1.0),
            sound_speed=10.0,
        )


@pytest.mark.parametrize(
    "bad_trace",
    [
        _trace(0.0, 0.0, 1.0),
        _trace(-1.0, 0.0, 1.0),
        _trace(1.0, 0.0, 0.0),
        _trace(1.0, 0.0, -1.0),
    ],
)
def test_nonpositive_density_or_area_is_rejected(bad_trace) -> None:
    with pytest.raises(NonPositiveGasStateError):
        solve_isothermal_gas_tjunction(
            bad_trace,
            _trace(1.0, 0.0, 1.0),
            _trace(1.0, 0.0, 1.0),
            sound_speed=10.0,
        )


@pytest.mark.parametrize("velocity", [-10.0, 10.0, -10.1, 10.1])
def test_sonic_or_supersonic_trace_is_explicitly_rejected(velocity) -> None:
    with pytest.raises(SupersonicGasTraceError):
        solve_isothermal_gas_tjunction(
            _trace(1.0, velocity, 1.0),
            _trace(1.0, 0.0, 1.0),
            _trace(1.0, 0.0, 1.0),
            sound_speed=10.0,
        )


def test_characteristic_closure_without_subsonic_solution_is_rejected() -> None:
    with pytest.raises(
        NoAdmissibleSubsonicJunctionError,
        match="non-subsonic",
    ):
        solve_isothermal_gas_tjunction(
            _trace(1.0e-4, 0.0, 1.0),
            _trace(1.0, 0.0, 1.0),
            _trace(1.0, 0.0, 1.0),
            sound_speed=10.0,
        )


def test_invalid_coordinate_sign_is_rejected() -> None:
    with pytest.raises(GasTJunctionError, match=r"exactly -1 or \+1"):
        velocity_in_outward_coordinate(1.0, 0)
