import pytest

from alignment import (
    AlignmentGapError,
    TimeShiftNotAllowedError,
    make_common_grid,
    resample_to_common_grid,
    validate_strict_common_grid,
)


def test_resampling_uses_exact_unshifted_common_grid():
    target = make_common_grid(0.0, 0.3)
    aligned = resample_to_common_grid(
        "pressure",
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        target,
    )
    assert aligned.time_s == pytest.approx((0.0, 0.1, 0.2, 0.3))
    assert aligned.values == pytest.approx((0.0, 1.0, 2.0, 3.0))
    assert aligned.time_shift_applied_s == 0.0


def test_interpolation_across_missing_solver_interval_is_rejected():
    with pytest.raises(AlignmentGapError, match="refusing to interpolate"):
        resample_to_common_grid(
            "pressure",
            [0.0, 0.1, 0.3],
            [0.0, 1.0, 3.0],
            [0.0, 0.1, 0.2, 0.3],
        )


def test_nonzero_time_shift_is_rejected():
    with pytest.raises(TimeShiftNotAllowedError):
        resample_to_common_grid(
            "pressure",
            [0.0, 0.1],
            [0.0, 1.0],
            [0.0, 0.1],
            time_shift_s=0.1,
        )


def test_common_grid_cannot_skip_a_tick():
    with pytest.raises(AlignmentGapError):
        validate_strict_common_grid([0.0, 0.2])
