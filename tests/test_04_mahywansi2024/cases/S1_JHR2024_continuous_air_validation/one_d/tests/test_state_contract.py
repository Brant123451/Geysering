from dataclasses import FrozenInstanceError, fields, replace

import pytest

from model import ContractViolation, CoupledState, HorizontalState, TNodeState, VerticalState
from model.state import SupplyBranchState


def test_required_state_field_names_are_frozen() -> None:
    assert [field.name for field in fields(HorizontalState)] == ["Al", "Ql", "Mg", "Jg"]
    assert [field.name for field in fields(SupplyBranchState)] == ["Al", "Ql", "Mg", "Jg"]
    assert [field.name for field in fields(VerticalState)] == [
        "Aup",
        "Qup",
        "Adown",
        "Qdown",
        "Mg",
        "Jg",
    ]


def test_state_is_immutable_and_normalizes_to_tuples(coupled_state) -> None:
    assert isinstance(coupled_state.horizontal.Al, tuple)
    with pytest.raises(FrozenInstanceError):
        coupled_state.horizontal.Al = (0.0, 0.0)


def test_vertical_two_streams_persist_independently(coupled_state) -> None:
    vertical = coupled_state.vertical
    assert vertical.Qup == (2.0e-5, 2.0e-5)
    assert vertical.Qdown == (1.5e-5, 1.5e-5)
    assert vertical.net_liquid_discharge == pytest.approx((0.5e-5, 0.5e-5))
    assert not hasattr(VerticalState, "from_net_liquid_discharge")


def test_two_physical_t_nodes_are_explicit_and_singular_node_is_absent() -> None:
    assert [field.name for field in fields(CoupledState)] == [
        "time_s",
        "horizontal",
        "vertical",
        "supply_branch",
        "exterior_plume",
        "air_supply_node",
        "riser_node",
    ]
    assert not hasattr(CoupledState, "node")


def test_t_nodes_are_zero_storage_algebraic_junctions() -> None:
    assert TNodeState().is_zero_storage
    with pytest.raises(ContractViolation, match="zero-storage"):
        TNodeState(gas_mass=1.0e-9)
    with pytest.raises(ContractViolation, match="zero-storage"):
        TNodeState(liquid_momentum=1.0e-9)


def test_declared_elastic_horizontal_overarea_is_gas_free_only(
    coupled_state, geometry, pipe_area
) -> None:
    elastic_horizontal = replace(
        coupled_state.horizontal,
        Al=(1.01 * pipe_area, coupled_state.horizontal.Al[1]),
        Mg=(0.0, coupled_state.horizontal.Mg[1]),
        Jg=(0.0, coupled_state.horizontal.Jg[1]),
    )
    elastic_state = replace(coupled_state, horizontal=elastic_horizontal)
    geometry.validate_state(elastic_state)

    undeclared_geometry = replace(geometry, horizontal_elastic_overarea_fraction=0.0)
    with pytest.raises(ContractViolation, match="elastic overarea bound"):
        undeclared_geometry.validate_state(elastic_state)

    gas_overarea = replace(
        elastic_state,
        horizontal=replace(
            elastic_horizontal,
            Mg=(1.0e-9, elastic_horizontal.Mg[1]),
        ),
    )
    with pytest.raises(ContractViolation, match="contains gas"):
        geometry.validate_state(gas_overarea)


def test_gas_mass_requires_positive_complementary_area(
    coupled_state, geometry, pipe_area
) -> None:
    full_gas_cell = replace(
        coupled_state,
        horizontal=replace(
            coupled_state.horizontal,
            Al=(pipe_area, coupled_state.horizontal.Al[1]),
        ),
    )
    with pytest.raises(ContractViolation, match="no positive complementary gas area"):
        geometry.validate_state(full_gas_cell)

    full_supply_cell = replace(
        coupled_state,
        supply_branch=replace(
            coupled_state.supply_branch,
            Al=(pipe_area, coupled_state.supply_branch.Al[1]),
        ),
    )
    with pytest.raises(ContractViolation, match="no positive complementary gas area"):
        geometry.validate_state(full_supply_cell)


def test_gas_momentum_cannot_exist_without_gas_mass(coupled_state, geometry) -> None:
    invalid = replace(
        coupled_state,
        supply_branch=replace(
            coupled_state.supply_branch,
            Mg=(0.0, coupled_state.supply_branch.Mg[1]),
            Jg=(1.0e-6, coupled_state.supply_branch.Jg[1]),
        ),
    )
    with pytest.raises(ContractViolation, match="momentum without gas mass"):
        geometry.validate_state(invalid)


def test_horizontal_vacuum_roundoff_momentum_uses_the_physical_owner_band(
    coupled_state, geometry
) -> None:
    horizontal = replace(
        coupled_state.horizontal,
        Al=(geometry.horizontal_area_m2, coupled_state.horizontal.Al[1]),
        Mg=(0.0, coupled_state.horizontal.Mg[1]),
        Jg=(5.0e-34, coupled_state.horizontal.Jg[1]),
    )
    numerical = replace(coupled_state, horizontal=horizontal)
    before = numerical.horizontal.Jg

    geometry.validate_state(numerical)

    # Validation is an admissibility decision, not a conservative projection.
    assert numerical.horizontal.Jg == before
    finite = replace(
        numerical,
        horizontal=replace(horizontal, Jg=(1.01e-10, horizontal.Jg[1])),
    )
    with pytest.raises(ContractViolation, match="momentum without gas mass"):
        geometry.validate_state(finite)


@pytest.mark.parametrize("owner", ["horizontal", "vertical", "supply_branch"])
def test_positive_gas_area_cannot_be_a_massless_vacuum(
    coupled_state, geometry, pipe_area, owner
) -> None:
    if owner == "horizontal":
        invalid = replace(
            coupled_state,
            horizontal=replace(
                coupled_state.horizontal,
                Al=(0.9 * pipe_area, coupled_state.horizontal.Al[1]),
                Mg=(0.0, coupled_state.horizontal.Mg[1]),
                Jg=(0.0, coupled_state.horizontal.Jg[1]),
            ),
        )
    elif owner == "vertical":
        invalid = replace(
            coupled_state,
            vertical=replace(
                coupled_state.vertical,
                Aup=(0.2 * pipe_area, coupled_state.vertical.Aup[1]),
                Adown=(0.7 * pipe_area, coupled_state.vertical.Adown[1]),
                Mg=(0.0, coupled_state.vertical.Mg[1]),
                Jg=(0.0, coupled_state.vertical.Jg[1]),
            ),
        )
    else:
        invalid = replace(
            coupled_state,
            supply_branch=replace(
                coupled_state.supply_branch,
                Al=(0.9 * pipe_area, coupled_state.supply_branch.Al[1]),
                Mg=(0.0, coupled_state.supply_branch.Mg[1]),
                Jg=(0.0, coupled_state.supply_branch.Jg[1]),
            ),
        )
    with pytest.raises(ContractViolation, match="positive gas area but zero gas mass"):
        geometry.validate_state(invalid)
