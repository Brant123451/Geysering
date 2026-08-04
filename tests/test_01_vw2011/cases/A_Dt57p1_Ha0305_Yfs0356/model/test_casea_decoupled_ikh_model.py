from __future__ import annotations

import numpy as np

from casea_decoupled_ikh_model import (
    ModelParameters,
    advance_ssprk2,
    primitives_to_conserved,
    restoring_coefficient,
    stable_time_step,
)


def test_uniform_state_is_preserved() -> None:
    params = ModelParameters()
    n = 64
    al = np.full(n, 0.72 * params.area_full)
    rho_g = np.full(n, params.atmospheric_pressure / (params.gas_constant * params.gas_temperature))
    ug = np.full(n, 0.4)
    ul = np.full(n, 0.2)
    state = primitives_to_conserved(rho_g, ug, al, ul, params)
    before = state.copy()
    dx = 1.0 / n
    state = advance_ssprk2(state, stable_time_step(state, dx, 0.35, params), dx, params)
    np.testing.assert_allclose(state, before, rtol=1.0e-11, atol=1.0e-12)


def test_ikh_sign_changes_above_critical_slip() -> None:
    params = ModelParameters()
    al = np.asarray([0.75 * params.area_full])
    rho_g = np.asarray([1.3])
    ul = np.asarray([0.2])
    base = restoring_coefficient(al, ul, rho_g, np.asarray([0.2]), params)[0]
    ag = params.area_full - al[0]
    critical = np.sqrt(
        params.rho_l / rho_g[0] * ag * base
    )
    stable = restoring_coefficient(
        al, ul, rho_g, ul + 0.95 * critical, params
    )[0]
    unstable = restoring_coefficient(
        al, ul, rho_g, ul + 1.05 * critical, params
    )[0]
    assert stable > 0.0
    assert unstable < 0.0
