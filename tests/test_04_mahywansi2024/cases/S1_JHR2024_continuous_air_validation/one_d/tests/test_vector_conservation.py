from dataclasses import replace
import math

import pytest

from model import (
    AtomicCommitter,
    AtomicFluxPacket,
    BoundaryExchange,
    ConservationSnapshot,
    ContractViolation,
    HorizontalDelta,
    VerticalDelta,
)
from model.flux import SupplyBranchDelta, state_token


def test_inventory_keeps_horizontal_px_separate_from_vertical_and_supply_pz(
    coupled_state, geometry
) -> None:
    state = replace(
        coupled_state,
        horizontal=replace(coupled_state.horizontal, Ql=(1.0e-5, 0.0)),
        supply_branch=replace(coupled_state.supply_branch, Jg=(2.0e-4, 0.0)),
    )
    snapshot = ConservationSnapshot.from_state(state, geometry)

    expected_px = (
        geometry.liquid_density_kg_m3 * 1.0e-5 * geometry.horizontal_dx_m[0]
    )
    expected_vertical_pz = sum(
        geometry.liquid_density_kg_m3 * (up - down) * dz
        for up, down, dz in zip(
            state.vertical.Qup,
            state.vertical.Qdown,
            geometry.vertical_dz_m,
            strict=True,
        )
    )
    expected_supply_pz = 2.0e-4 * geometry.supply_branch_dz_m[0]

    assert snapshot.mixture_momentum_x_kg_m_s == pytest.approx(expected_px)
    assert snapshot.mixture_momentum_z_kg_m_s == pytest.approx(
        expected_vertical_pz + expected_supply_pz
    )
    assert snapshot.mixture_axial_momentum_kg_m_s == pytest.approx(
        math.hypot(expected_px, expected_vertical_pz + expected_supply_pz)
    )
    assert snapshot.mixture_axial_momentum_kg_m_s != pytest.approx(
        expected_px + expected_vertical_pz + expected_supply_pz
    )


def test_ledger_records_axis_specific_boundary_and_external_impulses(
    coupled_state, geometry
) -> None:
    dt_s = 0.01
    dql = 1.0e-6
    djg_supply = 2.0e-6
    px_change = (
        geometry.liquid_density_kg_m3 * dql * geometry.horizontal_dx_m[0]
    )
    pz_change = djg_supply * geometry.supply_branch_dz_m[0]
    boundary = BoundaryExchange(
        momentum_x_in_N=px_change / dt_s,
        external_force_z_N=pz_change / dt_s,
    )
    packet = AtomicFluxPacket(
        transaction_id="axis-specific-impulses",
        base_state_token=state_token(coupled_state),
        dt_s=dt_s,
        horizontal=HorizontalDelta(
            Al=(0.0, 0.0),
            Ql=(dql, 0.0),
            Mg=(0.0, 0.0),
            Jg=(0.0, 0.0),
        ),
        vertical=VerticalDelta.zeros(coupled_state.vertical.cell_count),
        supply_branch=SupplyBranchDelta(
            Al=(0.0, 0.0),
            Ql=(0.0, 0.0),
            Mg=(0.0, 0.0),
            Jg=(djg_supply, 0.0),
        ),
        boundary=boundary,
    )

    _, entry = AtomicCommitter(geometry).commit(coupled_state, packet)
    assert entry.boundary_momentum_x_impulse_kg_m_s == pytest.approx(px_change)
    assert entry.boundary_momentum_z_impulse_kg_m_s == pytest.approx(0.0)
    assert entry.external_force_x_impulse_kg_m_s == pytest.approx(0.0)
    assert entry.external_force_z_impulse_kg_m_s == pytest.approx(pz_change)
    assert entry.mixture_momentum_x_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-14)
    assert entry.mixture_momentum_z_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-14)

    with pytest.raises(ContractViolation, match="both x and z"):
        _ = boundary.mixture_momentum_net_rate


def test_legacy_nonzero_scalar_momentum_exchange_is_rejected() -> None:
    with pytest.raises(ContractViolation, match="explicit x/z"):
        BoundaryExchange(mixture_momentum_in_N=1.0)
