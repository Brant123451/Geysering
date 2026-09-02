"""Source-aligned S1 initial assembly without advancing a physical trajectory."""

from __future__ import annotations

from dataclasses import dataclass

from .conservation import ConservationSnapshot
from .horizontal_case1_adapter import (
    Case1HorizontalLiquidAdapter,
    MahyawansiHorizontalGrid,
    build_s1_2d_eos_aligned_horizontal_adapter,
)
from .pressure_reservoir import IsothermalIdealGasPressureReservoir
from .state import (
    CoupledGeometry,
    CoupledState,
    ExteriorPlumeState,
    SupplyBranchState,
    TNodeState,
)
from .vertical_case1_adapter import (
    Case1VerticalComponentAdapter,
    build_s1_vertical_component,
)


S1_WATER_DENSITY_KG_M3 = 998.4
S1_WATER_VISCOSITY_PA_S = 1.002e-3
S1_AIR_VISCOSITY_PA_S = 1.78e-5
S1_SUPPLY_BRANCH_LENGTH_M = 0.1373
S1_HORIZONTAL_ELASTIC_OVERAREA_FRACTION = 0.02
S1_HORIZONTAL_INITIAL_HEAD_M = 0.5842
S1_HORIZONTAL_ELASTIC_REFERENCE_HEAD_M = 0.584


@dataclass(frozen=True, slots=True)
class S1InitialAssembly:
    """Validated Stage-1 state plus frozen numerical/source metadata.

    Both T nodes are zero-storage algebraic nodes at initialization. The gas
    pressure reservoir is present as boundary metadata but remains closed in
    Stage 1. The finite 0.1373 m water-filled supply branch owns distributed
    ``Al/Ql/Mg/Jg`` state so Stage-2 air can only enter through its top and
    displace that water conservatively. The assembly remains non-production-
    ready until one joint closure advances the branch, both T nodes, the main
    and the riser. No settling step or Stage-2 air injection is performed.
    """

    state: CoupledState
    geometry: CoupledGeometry
    inventory: ConservationSnapshot
    horizontal_adapter: Case1HorizontalLiquidAdapter
    vertical_adapter: Case1VerticalComponentAdapter
    pressure_reservoir: IsothermalIdealGasPressureReservoir
    pressure_reservoir_inlet_area_m2: float
    pressure_reservoir_inlet_area_status: str
    stage: str
    air_source_open: bool
    node_storage_status: str
    air_stub_status: str
    production_ready: bool


def build_s1_initial_assembly(
    *,
    horizontal_dx_m: float = 0.01,
    vertical_cell_count: int = 160,
    supply_branch_cell_count: int = 14,
) -> S1InitialAssembly:
    """Assemble the published/declared S1 Stage-1 initial distribution.

    The horizontal main pipe is water-filled and at rest.  The riser contains
    water to z=0.5842 m and atmospheric air above it, including the exact
    cut-cell phase fraction.  The continuous 5700 Pa gas source is closed
    during Stage 1.  The two physical T locations remain distinct even though
    this reduction assigns them no storage volume.
    """

    grid = MahyawansiHorizontalGrid(dx_m=horizontal_dx_m)
    horizontal = build_s1_2d_eos_aligned_horizontal_adapter(grid=grid)
    horizontal_state = horizontal.build_stage1_initial_state(
        initial_piezometric_head_m=S1_HORIZONTAL_INITIAL_HEAD_M,
        elastic_storage_reference_head_m=(
            S1_HORIZONTAL_ELASTIC_REFERENCE_HEAD_M
        ),
    )
    vertical = build_s1_vertical_component(cell_count=vertical_cell_count)
    vertical_state = vertical.initial.own_state
    if supply_branch_cell_count <= 0:
        raise ValueError("supply_branch_cell_count must be positive")
    supply_dz = S1_SUPPLY_BRANCH_LENGTH_M / supply_branch_cell_count
    supply_state = SupplyBranchState(
        Al=(horizontal.full_area_m2,) * supply_branch_cell_count,
        Ql=(0.0,) * supply_branch_cell_count,
        Mg=(0.0,) * supply_branch_cell_count,
        Jg=(0.0,) * supply_branch_cell_count,
    )

    geometry = CoupledGeometry(
        horizontal_dx_m=grid.cell_lengths_m,
        vertical_dz_m=(vertical.cell_length_m,) * vertical.cell_count,
        horizontal_area_m2=horizontal.full_area_m2,
        vertical_area_m2=horizontal.full_area_m2,
        liquid_density_kg_m3=S1_WATER_DENSITY_KG_M3,
        supply_branch_dz_m=(supply_dz,) * supply_branch_cell_count,
        supply_branch_area_m2=horizontal.full_area_m2,
        horizontal_elastic_overarea_fraction=(
            S1_HORIZONTAL_ELASTIC_OVERAREA_FRACTION
        ),
    )
    state = CoupledState(
        time_s=0.0,
        horizontal=horizontal_state,
        vertical=vertical_state,
        supply_branch=supply_state,
        exterior_plume=ExteriorPlumeState(),
        air_supply_node=TNodeState(),
        riser_node=TNodeState(),
    )
    geometry.validate_state(state)
    inventory = ConservationSnapshot.from_state(state, geometry)
    source = IsothermalIdealGasPressureReservoir()
    return S1InitialAssembly(
        state=state,
        geometry=geometry,
        inventory=inventory,
        horizontal_adapter=horizontal,
        vertical_adapter=vertical,
        pressure_reservoir=source,
        pressure_reservoir_inlet_area_m2=horizontal.full_area_m2,
        pressure_reservoir_inlet_area_status=(
            "declared_equal_diameter_1d_translation__not_effective_valve_area"
        ),
        stage="stage1_closed_air_pressure_driven_water_settling",
        air_source_open=False,
        node_storage_status="declared_zero_storage_algebraic_two_node_reduction",
        air_stub_status=(
            "owned_0p1373m_water_initial_two_phase_Al_Ql_Mg_Jg_branch_"
            "real_six_port_joint_owner_integrated__global_production_blocked"
        ),
        production_ready=False,
    )


__all__ = [
    "S1_AIR_VISCOSITY_PA_S",
    "S1InitialAssembly",
    "S1_HORIZONTAL_ELASTIC_OVERAREA_FRACTION",
    "S1_HORIZONTAL_ELASTIC_REFERENCE_HEAD_M",
    "S1_HORIZONTAL_INITIAL_HEAD_M",
    "S1_SUPPLY_BRANCH_LENGTH_M",
    "S1_WATER_DENSITY_KG_M3",
    "S1_WATER_VISCOSITY_PA_S",
    "build_s1_initial_assembly",
]
