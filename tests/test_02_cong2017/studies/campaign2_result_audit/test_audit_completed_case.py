from __future__ import annotations

from pathlib import Path

import numpy as np

from audit_completed_case import (
    centreline_bubble_state,
    first_confirmed_catchup,
    quantitative_support,
)


def test_quantitative_support_uses_fixed_common_bands() -> None:
    assert quantitative_support(1.10, 1.0) == "supported"
    assert quantitative_support(1.099, 1.0) == "supported"
    assert quantitative_support(1.299, 1.0) == "partial"
    assert quantitative_support(1.301, 1.0) == "failed"
    assert quantitative_support(None, 1.0) == "missing"


def test_catchup_requires_three_consecutive_samples() -> None:
    time = np.arange(7, dtype=float)
    yfs = np.full(7, 0.6)
    yint = np.asarray([0.0, 0.1, 0.591, 0.2, 0.591, 0.592, 0.593])
    catchup, threshold, initial_gap = first_confirmed_catchup(
        time, yfs, yint, ta_s=0.0, riser_dz_m=0.001
    )
    assert initial_gap == 0.6
    assert threshold == 0.012
    assert catchup == 4.0


def test_centreline_pressure_uses_uppermost_enclosed_gas_component(
    tmp_path: Path,
) -> None:
    path = tmp_path / "centreline_alpha.water_p_U.xy"
    distance = np.arange(8, dtype=float) * 0.01
    alpha = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    pressure = np.asarray([101500, 101500, 106000, 106000, 102000, 102000, 101325, 101325])
    velocity = np.zeros((8, 3))
    np.savetxt(path, np.column_stack([distance, alpha, pressure, velocity]))

    state = centreline_bubble_state(path)
    assert state is not None
    # The uppermost water-to-air crossing is the free surface; the enclosed
    # component is the lower gas pocket, not the atmosphere above it.
    assert np.isclose(state["Yfs_m_above_crown"], 0.056)
    assert np.isclose(state["Yint_m_above_crown"], 0.036)
    assert np.isclose(state["pocket_pressure_Pa_abs"], 106000.0)
