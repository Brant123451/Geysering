import pytest

from model import (
    AtomicCommitError,
    AtomicCommitter,
    AtomicFluxPacket,
    ConservationError,
    HorizontalDelta,
    TNodeDelta,
    VerticalDelta,
)
from model.flux import SupplyBranchDelta, TNodePortResidual, state_token


def test_zero_flux_step_is_static_and_has_zero_ledger(coupled_state, geometry) -> None:
    committer = AtomicCommitter(geometry)
    packet = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="static-1")
    advanced, entry = committer.commit(coupled_state, packet)

    assert advanced.time_s == pytest.approx(0.01)
    assert advanced.horizontal == coupled_state.horizontal
    assert advanced.vertical == coupled_state.vertical
    assert advanced.supply_branch == coupled_state.supply_branch
    assert advanced.air_supply_node == coupled_state.air_supply_node
    assert advanced.riser_node == coupled_state.riser_node
    assert entry.liquid_volume_residual_m3 == pytest.approx(0.0, abs=1.0e-15)
    assert entry.gas_mass_residual_kg == pytest.approx(0.0, abs=1.0e-15)
    assert entry.mixture_momentum_x_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-15)
    assert entry.mixture_momentum_z_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-15)
    assert entry.mixture_momentum_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-15)


def test_internal_transfer_commits_as_one_packet(coupled_state, geometry) -> None:
    # Move liquid volume from horizontal cell 1 (dx=0.5) to riser up-cell 0
    # (dz=0.25). The equal and opposite inventory changes close exactly.
    d_area_horizontal = -2.0e-7
    d_area_vertical = 4.0e-7
    packet = AtomicFluxPacket(
        transaction_id="internal-transfer-1",
        base_state_token=state_token(coupled_state),
        dt_s=0.01,
        horizontal=HorizontalDelta(
            Al=(0.0, d_area_horizontal),
            Ql=(0.0, 0.0),
            Mg=(0.0, 0.0),
            Jg=(0.0, 0.0),
        ),
        vertical=VerticalDelta(
            Aup=(d_area_vertical, 0.0),
            Qup=(0.0, 0.0),
            Adown=(0.0, 0.0),
            Qdown=(0.0, 0.0),
            Mg=(0.0, 0.0),
            Jg=(0.0, 0.0),
        ),
        supply_branch=SupplyBranchDelta.zeros(coupled_state.supply_branch.cell_count),
    )
    advanced, entry = AtomicCommitter(geometry).commit(coupled_state, packet)
    assert advanced.horizontal.Al[1] == pytest.approx(
        coupled_state.horizontal.Al[1] + d_area_horizontal
    )
    assert advanced.vertical.Aup[0] == pytest.approx(
        coupled_state.vertical.Aup[0] + d_area_vertical
    )
    assert entry.liquid_volume_residual_m3 == pytest.approx(0.0, abs=1.0e-15)


def test_one_packet_atomically_transfers_gas_from_supply_branch_to_main(
    coupled_state, geometry
) -> None:
    # Cell-integrated gas mass is Mg*cell_length.  The unequal cell lengths
    # therefore require unequal per-length deltas for one conservative transfer.
    supply_delta = -1.0e-5
    horizontal_delta = -supply_delta * geometry.supply_branch_dz_m[0] / geometry.horizontal_dx_m[0]
    packet = AtomicFluxPacket(
        transaction_id="supply-main-transfer-1",
        base_state_token=state_token(coupled_state),
        dt_s=0.01,
        horizontal=HorizontalDelta(
            Al=(0.0, 0.0),
            Ql=(0.0, 0.0),
            Mg=(horizontal_delta, 0.0),
            Jg=(0.0, 0.0),
        ),
        vertical=VerticalDelta.zeros(coupled_state.vertical.cell_count),
        supply_branch=SupplyBranchDelta(
            Al=(0.0, 0.0),
            Ql=(0.0, 0.0),
            Mg=(supply_delta, 0.0),
            Jg=(0.0, 0.0),
        ),
        air_supply_node_ports=TNodePortResidual(),
    )
    advanced, entry = AtomicCommitter(geometry).commit(coupled_state, packet)
    assert advanced.supply_branch.Mg[0] == pytest.approx(
        coupled_state.supply_branch.Mg[0] + supply_delta
    )
    assert advanced.horizontal.Mg[0] == pytest.approx(
        coupled_state.horizontal.Mg[0] + horizontal_delta
    )
    assert entry.gas_mass_residual_kg == pytest.approx(0.0, abs=1.0e-15)


def test_failed_packet_changes_neither_state_nor_ledger(coupled_state, geometry) -> None:
    committer = AtomicCommitter(geometry)
    bad_packet = AtomicFluxPacket(
        transaction_id="unbalanced-1",
        base_state_token=state_token(coupled_state),
        dt_s=0.01,
        horizontal=HorizontalDelta(
            Al=(1.0e-7, 0.0),
            Ql=(0.0, 0.0),
            Mg=(0.0, 0.0),
            Jg=(0.0, 0.0),
        ),
        vertical=VerticalDelta.zeros(coupled_state.vertical.cell_count),
        supply_branch=SupplyBranchDelta.zeros(coupled_state.supply_branch.cell_count),
    )
    original = coupled_state
    with pytest.raises(ConservationError):
        committer.commit(coupled_state, bad_packet)
    assert coupled_state is original
    assert committer.ledger.entries == []


def test_packet_cannot_be_committed_twice(coupled_state, geometry) -> None:
    committer = AtomicCommitter(geometry)
    packet = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="once-only")
    committer.commit(coupled_state, packet)
    with pytest.raises(AtomicCommitError):
        committer.commit(coupled_state, packet)


def test_base_state_cannot_receive_a_second_t_node_packet(coupled_state, geometry) -> None:
    committer = AtomicCommitter(geometry)
    first = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="owner-1")
    second = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="owner-2")
    committer.commit(coupled_state, first)
    with pytest.raises(AtomicCommitError, match="duplicate ownership"):
        committer.commit(coupled_state, second)


def test_stale_packet_is_rejected_without_ledger_mutation(coupled_state, geometry) -> None:
    seed_committer = AtomicCommitter(geometry)
    first = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="advance-base")
    advanced, _ = seed_committer.commit(coupled_state, first)
    stale = AtomicFluxPacket.zero(coupled_state, dt_s=0.01, transaction_id="stale")
    committer = AtomicCommitter(geometry)
    with pytest.raises(AtomicCommitError, match="different or stale"):
        committer.commit(advanced, stale)
    assert committer.ledger.entries == []


@pytest.mark.parametrize("port_field", ("air_supply_node_ports", "riser_node_ports"))
def test_nonzero_node_port_residual_rolls_back_entire_step(
    coupled_state, geometry, port_field
) -> None:
    committer = AtomicCommitter(geometry)
    transaction_id = f"node-residual-retry-{port_field}"
    bad = AtomicFluxPacket(
        transaction_id=transaction_id,
        base_state_token=state_token(coupled_state),
        dt_s=0.01,
        horizontal=HorizontalDelta.zeros(coupled_state.horizontal.cell_count),
        vertical=VerticalDelta.zeros(coupled_state.vertical.cell_count),
        supply_branch=SupplyBranchDelta.zeros(coupled_state.supply_branch.cell_count),
        **{port_field: TNodePortResidual(gas_mass_kg_s=2.0e-10)},
    )
    with pytest.raises(AtomicCommitError, match="port balance failed"):
        committer.commit(coupled_state, bad)
    assert committer.ledger.entries == []

    # The failed ID and base token were not consumed: the corrected whole
    # packet can be committed, proving rollback rather than partial ownership.
    corrected = AtomicFluxPacket.zero(
        coupled_state,
        dt_s=0.01,
        transaction_id=transaction_id,
    )
    advanced, _ = committer.commit(coupled_state, corrected)
    assert advanced.time_s == pytest.approx(0.01)
    assert len(committer.ledger.entries) == 1


def test_node_inventory_delta_is_forbidden_and_rolls_back(coupled_state, geometry) -> None:
    committer = AtomicCommitter(geometry)
    bad = AtomicFluxPacket(
        transaction_id="node-inventory-forbidden",
        base_state_token=state_token(coupled_state),
        dt_s=0.01,
        horizontal=HorizontalDelta.zeros(coupled_state.horizontal.cell_count),
        vertical=VerticalDelta.zeros(coupled_state.vertical.cell_count),
        supply_branch=SupplyBranchDelta.zeros(coupled_state.supply_branch.cell_count),
        riser_node=TNodeDelta(liquid_volume=1.0e-9),
    )
    with pytest.raises(AtomicCommitError, match="inventory deltas are forbidden"):
        committer.commit(coupled_state, bad)
    assert committer.ledger.entries == []
