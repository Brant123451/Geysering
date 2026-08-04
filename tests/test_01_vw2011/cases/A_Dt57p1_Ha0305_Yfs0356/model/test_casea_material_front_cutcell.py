from __future__ import annotations

from dataclasses import replace

import pytest

from casea_material_front_cutcell import (
    ALEInterfaceFlux,
    InterfaceClosureError,
    InterfaceTraces,
    MaterialFrontCutCell,
    OuterFaceFluxes,
    PressurisedFlux,
    PressurisedState,
    StratifiedFlux,
    StratifiedState,
    SubcellSources,
    VanishingSubcellInventoryError,
    advance_material_front_cutcell,
)


def _uniform_front(
    speed: float,
    *,
    liquid_ale_momentum: float = 5.0,
    gas_piston_momentum: float = 3.0,
) -> tuple[InterfaceTraces, OuterFaceFluxes]:
    """Constant states whose physical fluxes translate exactly with the front."""

    pressurised = PressurisedState(area=1.0, discharge=speed)
    stratified = StratifiedState(
        gas_mass=0.6,
        gas_momentum=0.6 * speed,
        liquid_area=0.4,
        liquid_discharge=0.4 * speed,
    )
    pressurised_flux = PressurisedFlux(
        area=pressurised.discharge,
        momentum=speed * pressurised.discharge + liquid_ale_momentum,
    )
    stratified_flux = StratifiedFlux(
        gas_mass=stratified.gas_momentum,
        gas_momentum=(
            speed * stratified.gas_momentum + gas_piston_momentum
        ),
        liquid_area=stratified.liquid_discharge,
        liquid_momentum=(
            speed * stratified.liquid_discharge + liquid_ale_momentum
        ),
    )
    traces = InterfaceTraces(
        speed=speed,
        pressurised_state=pressurised,
        pressurised_flux=pressurised_flux,
        stratified_state=stratified,
        stratified_flux=stratified_flux,
    )
    return traces, OuterFaceFluxes(pressurised_flux, stratified_flux)


def _constant_provider(value):
    return lambda _state, _time: value


def _assert_inventory_zero(inventory, *, atol: float = 2.0e-14) -> None:
    assert inventory.vector() == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=atol)


def test_stationary_front_preserves_both_subcell_states() -> None:
    traces, outer = _uniform_front(0.0)
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0),
        host_index=0,
        front_position=0.37,
        pressurised_side="left",
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )

    result = advance_material_front_cutcell(
        state,
        0.8,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
    )

    assert result.state.front_position == 0.37
    assert result.state.pressurised == state.pressurised
    assert result.state.stratified == state.stratified
    assert result.crossings == ()
    _assert_inventory_zero(result.ledgers[0].residual)


@pytest.mark.parametrize("pressurised_side", ["left", "right"])
def test_uniform_translation_is_exact_in_both_orientations(
    pressurised_side: str,
) -> None:
    traces, outer = _uniform_front(0.3)
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0),
        host_index=0,
        front_position=0.25,
        pressurised_side=pressurised_side,
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )

    result = advance_material_front_cutcell(
        state,
        0.5,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
    )

    assert result.state.front_position == pytest.approx(0.4, abs=1.0e-15)
    assert result.state.pressurised.vector() == pytest.approx(
        state.pressurised.vector(), abs=2.0e-15
    )
    assert result.state.stratified.vector() == pytest.approx(
        state.stratified.vector(), abs=2.0e-15
    )
    _assert_inventory_zero(result.ledgers[0].residual)


@pytest.mark.parametrize("pressurised_side", ["left", "right"])
def test_liquid_and_gas_stage_ledgers_close_componentwise(
    pressurised_side: str,
) -> None:
    traces, _ = _uniform_front(0.2)
    outer = OuterFaceFluxes(
        pressurised=PressurisedFlux(area=0.13, momentum=4.7),
        stratified=StratifiedFlux(
            gas_mass=0.07,
            gas_momentum=0.8,
            liquid_area=0.08,
            liquid_momentum=4.2,
        ),
    )
    sources = SubcellSources(
        pressurised_area=0.011,
        pressurised_momentum=-0.17,
        stratified_gas_mass=0.013,
        stratified_gas_momentum=-0.12,
        stratified_liquid_area=-0.007,
        stratified_liquid_momentum=0.09,
    )
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0),
        host_index=0,
        front_position=0.4,
        pressurised_side=pressurised_side,
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )
    dt = 0.01

    result = advance_material_front_cutcell(
        state,
        dt,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
        source_provider=_constant_provider(sources),
    )
    ledger = result.ledgers[0]
    lp = state.pressurised_length
    ls = state.stratified_length
    gas_piston = 3.0
    if pressurised_side == "left":
        expected = (
            dt * (-outer.stratified.gas_mass + ls * sources.stratified_gas_mass),
            dt
            * (
                gas_piston
                - outer.stratified.gas_momentum
                + ls * sources.stratified_gas_momentum
            ),
            dt
            * (
                outer.pressurised.area
                - outer.stratified.liquid_area
                + lp * sources.pressurised_area
                + ls * sources.stratified_liquid_area
            ),
            dt
            * (
                outer.pressurised.momentum
                - outer.stratified.liquid_momentum
                + lp * sources.pressurised_momentum
                + ls * sources.stratified_liquid_momentum
            ),
        )
    else:
        expected = (
            dt * (outer.stratified.gas_mass + ls * sources.stratified_gas_mass),
            dt
            * (
                outer.stratified.gas_momentum
                - gas_piston
                + ls * sources.stratified_gas_momentum
            ),
            dt
            * (
                outer.stratified.liquid_area
                - outer.pressurised.area
                + lp * sources.pressurised_area
                + ls * sources.stratified_liquid_area
            ),
            dt
            * (
                outer.stratified.liquid_momentum
                - outer.pressurised.momentum
                + lp * sources.pressurised_momentum
                + ls * sources.stratified_liquid_momentum
            ),
        )
    assert ledger.expected_change.vector() == pytest.approx(expected, abs=1.0e-15)
    actual = tuple(
        final - initial
        for final, initial in zip(ledger.final.vector(), ledger.initial.vector())
    )
    assert actual == pytest.approx(expected, abs=2.0e-15)
    _assert_inventory_zero(ledger.residual)
    assert ledger.interface_flux.gas_mass == 0.0


def test_case_a_s_left_p_right_positive_crossing_starts_exact_zero_s_subcell() -> None:
    """Away from the Case-A T, S|P advances toward increasing branch x."""

    traces, outer = _uniform_front(0.4)
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0, 3.0),
        host_index=0,
        front_position=0.8,
        pressurised_side="right",
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )

    def new_host(request):
        assert request.existing_new_host_branch == "pressurised"
        assert request.completed_branch == "stratified"
        return traces.pressurised_state

    result = advance_material_front_cutcell(
        state,
        0.5,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
        new_host_provider=new_host,
    )

    assert result.state.host_index == 1
    assert result.state.front_position == 1.0
    assert result.state.stratified_length == 0.0
    assert result.state.pressurised_length == 1.0
    assert result.state.stratified == traces.stratified_state
    assert result.crossings[0].zero_length_branch == "stratified"
    assert result.crossings[0].completed_cell.branch == "stratified"
    _assert_inventory_zero(result.crossings[0].remap_residual)
    _assert_inventory_zero(result.ledgers[0].residual)


def test_zero_length_new_host_advances_without_fill_or_special_assignment() -> None:
    traces, outer = _uniform_front(0.4)
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0, 3.0),
        host_index=0,
        front_position=0.8,
        pressurised_side="right",
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )

    result = advance_material_front_cutcell(
        state,
        0.75,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
        new_host_provider=lambda request: (
            traces.pressurised_state
            if request.existing_new_host_branch == "pressurised"
            else traces.stratified_state
        ),
    )

    # The first substep ends at a truly zero S length.  The remaining 0.25 s
    # then grows that subcell from the ALE balance to 0.1 m; there is no seed.
    assert len(result.ledgers) == 2
    assert result.state.host_index == 1
    assert result.state.front_position == pytest.approx(1.1, abs=2.0e-15)
    assert result.state.stratified_length == pytest.approx(0.1, abs=2.0e-15)
    assert result.state.stratified.vector() == pytest.approx(
        traces.stratified_state.vector(), abs=3.0e-15
    )
    assert result.state.pressurised.vector() == pytest.approx(
        traces.pressurised_state.vector(), abs=3.0e-15
    )
    for ledger in result.ledgers:
        _assert_inventory_zero(ledger.residual)


def test_reverse_front_crossing_is_conservative_for_s_left_p_right() -> None:
    traces, outer = _uniform_front(-0.4)
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0, 3.0),
        host_index=1,
        front_position=1.2,
        pressurised_side="right",
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )

    def new_host(request):
        assert request.moving_direction == -1
        assert request.completed_branch == "pressurised"
        assert request.existing_new_host_branch == "stratified"
        return traces.stratified_state

    result = advance_material_front_cutcell(
        state,
        0.75,
        interface_provider=_constant_provider(traces),
        outer_flux_provider=_constant_provider(outer),
        new_host_provider=new_host,
    )

    assert result.state.host_index == 0
    assert result.state.front_position == pytest.approx(0.9, abs=2.0e-15)
    assert result.state.pressurised_length == pytest.approx(0.1, abs=2.0e-15)
    assert result.crossings[0].completed_cell.branch == "pressurised"
    assert result.crossings[0].zero_length_branch == "pressurised"
    assert result.state.pressurised.vector() == pytest.approx(
        traces.pressurised_state.vector(), abs=3.0e-15
    )
    assert result.state.stratified.vector() == pytest.approx(
        traces.stratified_state.vector(), abs=3.0e-15
    )
    for ledger in result.ledgers:
        _assert_inventory_zero(ledger.residual)


def test_interface_rejects_nonmaterial_gas_and_mismatched_liquid_flux() -> None:
    traces, _ = _uniform_front(0.3)
    bad_gas = replace(
        traces,
        stratified_state=replace(traces.stratified_state, gas_momentum=0.21),
        stratified_flux=replace(traces.stratified_flux, gas_mass=0.21),
    )
    with pytest.raises(InterfaceClosureError, match="material condition"):
        ALEInterfaceFlux.from_traces(bad_gas)

    bad_liquid = replace(
        traces,
        stratified_flux=replace(
            traces.stratified_flux,
            liquid_momentum=traces.stratified_flux.liquid_momentum + 0.1,
        ),
    )
    with pytest.raises(InterfaceClosureError, match="Rankine"):
        ALEInterfaceFlux.from_traces(bad_liquid)


def test_crossing_rejects_finite_inventory_in_disappearing_subcell() -> None:
    traces, outer = _uniform_front(0.4)
    # S|P with w>0 makes P disappear.  Altering its outer liquid flux breaks
    # the exact material transport, so the event must be rejected, not remapped.
    bad_outer = replace(
        outer,
        pressurised=replace(
            outer.pressurised,
            area=outer.pressurised.area + 0.01,
        ),
    )
    state = MaterialFrontCutCell(
        cell_faces=(0.0, 1.0, 2.0),
        host_index=0,
        front_position=0.8,
        pressurised_side="right",
        pressurised=traces.pressurised_state,
        stratified=traces.stratified_state,
    )
    with pytest.raises(VanishingSubcellInventoryError, match="pressurised"):
        advance_material_front_cutcell(
            state,
            0.5,
            interface_provider=_constant_provider(traces),
            outer_flux_provider=_constant_provider(bad_outer),
            new_host_provider=lambda _request: traces.pressurised_state,
        )
