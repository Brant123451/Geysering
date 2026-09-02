import math

import pytest

from model import CoupledGeometry, CoupledState, HorizontalState, TNodeState, VerticalState
from model.state import SupplyBranchState


@pytest.fixture
def pipe_area() -> float:
    return math.pi * 0.0254**2 / 4.0


@pytest.fixture
def geometry(pipe_area: float) -> CoupledGeometry:
    return CoupledGeometry(
        horizontal_dx_m=(0.5, 0.5),
        vertical_dz_m=(0.25, 0.25),
        horizontal_area_m2=pipe_area,
        vertical_area_m2=pipe_area,
        liquid_density_kg_m3=998.2,
        supply_branch_dz_m=(0.06865, 0.06865),
        supply_branch_area_m2=pipe_area,
        horizontal_elastic_overarea_fraction=0.02,
    )


@pytest.fixture
def coupled_state(pipe_area: float) -> CoupledState:
    return CoupledState(
        time_s=0.0,
        horizontal=HorizontalState(
            Al=(0.6 * pipe_area, 0.5 * pipe_area),
            Ql=(0.0, 0.0),
            Mg=(0.001, 0.002),
            Jg=(0.0, 0.0),
        ),
        vertical=VerticalState(
            Aup=(0.2 * pipe_area, 0.2 * pipe_area),
            Qup=(2.0e-5, 2.0e-5),
            Adown=(0.3 * pipe_area, 0.3 * pipe_area),
            Qdown=(1.5e-5, 1.5e-5),
            Mg=(0.001, 0.001),
            Jg=(0.0, 0.0),
        ),
        supply_branch=SupplyBranchState(
            Al=(0.0, 0.0),
            Ql=(0.0, 0.0),
            Mg=(5.0e-4, 5.0e-4),
            Jg=(0.0, 0.0),
        ),
        air_supply_node=TNodeState(),
        riser_node=TNodeState(),
    )
