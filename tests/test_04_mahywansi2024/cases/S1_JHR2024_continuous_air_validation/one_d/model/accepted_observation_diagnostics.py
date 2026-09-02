"""Native diagnostics carried by an accepted S1 whole-network step.

This module is observational only.  It does not change a flux, state, time
step or boundary condition.  Instantaneous pressures are reconstructed from
the same Case-1 horizontal pressure closure and six-state riser pressure/void
closure used by the physical owner.  Gross flows and zero-storage-node forces
remain interval quantities and are combined with the SSP-RK2 weights used by
the atomic packet.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .errors import ContractViolation, MissingPhysicalClosure
from .state import CoupledGeometry, CoupledState


ATMOSPHERIC_ABSOLUTE_PRESSURE_PA = 101325.0
PUBLISHED_PRESSURE_COORDINATES = {
    "P1": (0.0, 0.0),
    "P2": (0.0, 0.30),
    "P3": (0.0, 0.45),
    "P4": (-0.80, 0.0),
    "P5": (-0.10, 0.0),
    "P6": (0.10, 0.0),
}


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ContractViolation(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PressureSemantics:
    """Exact meaning of the six pressure values in a trajectory packet."""

    reference_absolute_pressure_Pa: float = ATMOSPHERIC_ABSOLUTE_PRESSURE_PA
    temporal_semantics: str = "instantaneous_accepted_state__no_time_interpolation"
    P1_semantics: str = (
        "riser_T_zero_storage_common_static_pressure_from_pure_post_state_node_solve"
    )
    P2_P3_semantics: str = (
        "vertical_native_common_pressure_face_closure__linear_spatial_sample_only"
    )
    P4_P6_semantics: str = (
        "Case1_horizontal_native_common_cell_pressure__segment_local_linear_spatial_sample_only"
    )
    gauge_semantics: str = "p_gauge=p_absolute-101325_Pa"

    def __post_init__(self) -> None:
        reference = _finite(
            "pressure reference absolute pressure",
            self.reference_absolute_pressure_Pa,
        )
        if reference <= 0.0:
            raise ContractViolation("pressure reference must be positive")
        object.__setattr__(self, "reference_absolute_pressure_Pa", reference)
        for name in (
            "temporal_semantics",
            "P1_semantics",
            "P2_P3_semantics",
            "P4_P6_semantics",
            "gauge_semantics",
        ):
            if not str(getattr(self, name)).strip():
                raise ContractViolation(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class InstantaneousGaugePressures:
    """P1--P6 at one immutable accepted state, all in gauge Pa."""

    P1: float
    P2: float
    P3: float
    P4: float
    P5: float
    P6: float
    semantics: PressureSemantics = PressureSemantics()

    def __post_init__(self) -> None:
        for name in ("P1", "P2", "P3", "P4", "P5", "P6"):
            object.__setattr__(self, name, _finite(f"{name} gauge pressure", getattr(self, name)))


@dataclass(frozen=True, slots=True)
class NativeIntervalDiagnostics:
    """One RK-stage or accepted SSP-RK2 interval diagnostic packet."""

    supply_branch_liquid_outflow_m3_s: float
    supply_branch_gas_inflow_kg_s: float
    mouth_liquid_outflow_m3_s: float
    mouth_liquid_inflow_m3_s: float
    mouth_gas_outflow_kg_s: float
    mouth_gas_inflow_kg_s: float
    air_supply_liquid_volume_residual_m3_s: float
    air_supply_gas_mass_residual_kg_s: float
    air_supply_momentum_x_residual_N: float
    air_supply_momentum_z_residual_N: float
    riser_liquid_volume_residual_m3_s: float
    riser_gas_mass_residual_kg_s: float
    riser_momentum_x_residual_N: float
    riser_momentum_z_residual_N: float
    air_supply_reaction_x_N: float
    air_supply_reaction_z_N: float
    riser_reaction_x_N: float
    riser_reaction_z_N: float
    connected_water_to_mouth: bool
    temporal_semantics: str = "native_RK_stage"

    def __post_init__(self) -> None:
        for name in (
            "supply_branch_liquid_outflow_m3_s",
            "supply_branch_gas_inflow_kg_s",
            "mouth_liquid_outflow_m3_s",
            "mouth_liquid_inflow_m3_s",
            "mouth_gas_outflow_kg_s",
            "mouth_gas_inflow_kg_s",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        for name in (
            "air_supply_liquid_volume_residual_m3_s",
            "air_supply_gas_mass_residual_kg_s",
            "air_supply_momentum_x_residual_N",
            "air_supply_momentum_z_residual_N",
            "riser_liquid_volume_residual_m3_s",
            "riser_gas_mass_residual_kg_s",
            "riser_momentum_x_residual_N",
            "riser_momentum_z_residual_N",
            "air_supply_reaction_x_N",
            "air_supply_reaction_z_N",
            "riser_reaction_x_N",
            "riser_reaction_z_N",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not isinstance(self.connected_water_to_mouth, bool):
            raise ContractViolation("connected_water_to_mouth must be bool")
        if not self.temporal_semantics.strip():
            raise ContractViolation("interval temporal semantics must be non-empty")

    @property
    def node_reaction_rate_magnitude_N(self) -> float:
        """Sum of both node reaction-vector magnitudes; cancellation is forbidden."""

        return math.hypot(
            self.air_supply_reaction_x_N, self.air_supply_reaction_z_N
        ) + math.hypot(self.riser_reaction_x_N, self.riser_reaction_z_N)


def _linear_sample(
    coordinates: Iterable[float], values: Iterable[float], target: float, *, label: str
) -> float:
    x = tuple(_finite(f"{label} coordinate", value) for value in coordinates)
    y = tuple(_finite(f"{label} value", value) for value in values)
    point = _finite(f"{label} target", target)
    if len(x) != len(y) or not x:
        raise MissingPhysicalClosure(f"{label} coordinate/value packet is incomplete")
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ContractViolation(f"{label} coordinates must be strictly increasing")
    tolerance = 5.0e-13 * max(1.0, abs(point))
    if point < x[0] - tolerance or point > x[-1] + tolerance:
        raise MissingPhysicalClosure(f"{label} target lies outside native pressure support")
    if point <= x[0]:
        return y[0]
    if point >= x[-1]:
        return y[-1]
    for index, (left, right) in enumerate(zip(x, x[1:])):
        if left - tolerance <= point <= right + tolerance:
            fraction = (point - left) / (right - left)
            return y[index] + fraction * (y[index + 1] - y[index])
    raise MissingPhysicalClosure(f"{label} target has no bracketing native samples")


def _horizontal_pressures_absolute_Pa(
    state: CoupledState, horizontal_component: object
) -> tuple[float, ...]:
    arrays_method = getattr(horizontal_component, "_arrays", None)
    pressure_method = getattr(horizontal_component, "_common_pressures", None)
    adapter = getattr(horizontal_component, "adapter", None)
    grid = getattr(adapter, "grid", None)
    if not callable(arrays_method) or not callable(pressure_method) or grid is None:
        raise MissingPhysicalClosure(
            "Case1 horizontal component has no native common-pressure diagnostic"
        )
    arrays = arrays_method(state.horizontal)
    pressures = tuple(float(value) for value in pressure_method(arrays))
    if len(pressures) != state.horizontal.cell_count:
        raise MissingPhysicalClosure("horizontal pressure diagnostic cell count drifted")
    centres = tuple(
        float(grid.x_left_m) + (index + 0.5) * float(grid.dx_m)
        for index in range(len(pressures))
    )

    # Probe brackets are segment-local.  In particular P5 and P6 must never
    # interpolate through the zero-storage riser-T face at x=0.
    segments = (
        (0, int(horizontal_component.air_face)),
        (int(horizontal_component.air_face), int(horizontal_component.riser_face)),
        (int(horizontal_component.riser_face), len(pressures)),
    )
    result: list[float] = []
    for name in ("P4", "P5", "P6"):
        target = PUBLISHED_PRESSURE_COORDINATES[name][0]
        selected = next(
            (
                (start, end)
                for start, end in segments
                if centres[start] <= target <= centres[end - 1]
            ),
            None,
        )
        if selected is None:
            raise MissingPhysicalClosure(
                f"{name} cannot be sampled inside one Case1 horizontal segment"
            )
        start, end = selected
        result.append(
            _linear_sample(
                centres[start:end],
                pressures[start:end],
                target,
                label=f"{name} Case1 horizontal pressure",
            )
        )
    return tuple(result)


def _vertical_pressure_faces_absolute_Pa(
    state: CoupledState,
    geometry: CoupledGeometry,
    vertical_component: object,
    *,
    bottom_common_absolute_pressure_Pa: float,
) -> tuple[float, ...]:
    """Evaluate the native pressure/void closure without advancing a state."""

    from .vertical_pressure_void_component import _component_state

    solver = getattr(vertical_component, "_solver", None)
    if solver is None:
        raise MissingPhysicalClosure("vertical component has no native six-state solver")
    validate = getattr(vertical_component, "_validate_geometry", None)
    filled_method = getattr(vertical_component, "_filled_pressure", None)
    if not callable(validate) or not callable(filled_method):
        raise MissingPhysicalClosure("vertical component pressure diagnostic is incomplete")
    validate(state.vertical, geometry)
    runtime = getattr(solver, "_runtime", None)
    parameters = getattr(solver, "_parameters", None)
    gas = getattr(solver, "_gas_parameters", None)
    closures = None if runtime is None else getattr(runtime, "closures", None)
    adapt = None if closures is None else getattr(
        closures, "adapt_gas_void_and_pressure_faces", None
    )
    if not callable(adapt) or parameters is None or gas is None:
        raise MissingPhysicalClosure("vertical native pressure/void closure is unavailable")
    component_state = _component_state(
        runtime,
        state.vertical,
        dry_discharge_tolerance_m3_s=parameters.dry_area_tolerance,
    )
    dz = tuple(float(value) for value in geometry.vertical_dz_m)
    gas_mass = tuple(
        value * width for value, width in zip(state.vertical.Mg, dz, strict=True)
    )
    gas_momentum = tuple(
        value * width for value, width in zip(state.vertical.Jg, dz, strict=True)
    )
    pressure = adapt(
        component_state,
        parameters,
        gas_mass=gas_mass,
        gas_momentum=gas_momentum,
        gas=gas,
        bottom_pressure=_finite(
            "riser-T common absolute pressure", bottom_common_absolute_pressure_Pa
        ),
        liquid_filled_cell_pressure=filled_method(
            bottom_common_absolute_pressure_Pa
        ),
    )
    faces = tuple(float(value) for value in pressure.common_pressure_faces)
    if len(faces) != state.vertical.cell_count + 1:
        raise MissingPhysicalClosure("vertical native pressure face count drifted")
    return faces


def build_instantaneous_gauge_pressures(
    state: CoupledState,
    geometry: CoupledGeometry,
    *,
    horizontal_component: object,
    vertical_component: object,
    riser_node_common_absolute_pressure_Pa: float,
) -> InstantaneousGaugePressures:
    """Build P1--P6 from one immutable state and its pure node solution."""

    geometry.validate_state(state)
    atmospheric = getattr(
        getattr(vertical_component, "atmospheric_top", None),
        "absolute_pressure_Pa",
        None,
    )
    if atmospheric is None or not math.isclose(
        float(atmospheric),
        ATMOSPHERIC_ABSOLUTE_PRESSURE_PA,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise MissingPhysicalClosure(
            "native pressure reference is not the frozen 101325 Pa atmosphere"
        )
    p1 = _finite(
        "P1 riser-T common absolute pressure",
        riser_node_common_absolute_pressure_Pa,
    )
    faces = _vertical_pressure_faces_absolute_Pa(
        state,
        geometry,
        vertical_component,
        bottom_common_absolute_pressure_Pa=p1,
    )
    z_faces = [0.0]
    for width in geometry.vertical_dz_m:
        z_faces.append(z_faces[-1] + width)
    p2 = _linear_sample(z_faces, faces, 0.30, label="P2 vertical common pressure")
    p3 = _linear_sample(z_faces, faces, 0.45, label="P3 vertical common pressure")
    p4, p5, p6 = _horizontal_pressures_absolute_Pa(state, horizontal_component)
    reference = ATMOSPHERIC_ABSOLUTE_PRESSURE_PA
    return InstantaneousGaugePressures(
        P1=p1 - reference,
        P2=p2 - reference,
        P3=p3 - reference,
        P4=p4 - reference,
        P5=p5 - reference,
        P6=p6 - reference,
    )


def interval_diagnostics_from_stage_rate(
    rate: object,
    *,
    vertical_area_m2: float,
) -> NativeIntervalDiagnostics:
    """Extract one RK-stage packet without reducing gross flow to net flow."""

    air = getattr(rate, "air_supply_node", None)
    riser = getattr(rate, "riser_node", None)
    supply_external = getattr(rate, "supply_external", None)
    vertical_external = getattr(rate, "vertical_external", None)
    if any(value is None for value in (air, riser, supply_external, vertical_external)):
        raise MissingPhysicalClosure("joint stage omitted an accepted native diagnostic")
    supply_port = next(
        (port for port in air.ports if port.name == "supply_bottom"), None
    )
    if supply_port is None:
        raise MissingPhysicalClosure("air-supply node omitted supply_bottom gross flux")
    area = _nonnegative("vertical area", vertical_area_m2)
    if area == 0.0:
        raise ContractViolation("vertical area must be positive")
    # A positive gross liquid outflow is itself proof that the native top
    # boundary presented connected liquid during this stage.  A stationary
    # wet mouth is supplied separately from the accepted-state topology by the
    # trajectory bridge and does not trigger an eruption event.
    connected = vertical_external.liquid_outflow_m3_s > 0.0
    return NativeIntervalDiagnostics(
        supply_branch_liquid_outflow_m3_s=(
            supply_port.liquid_out_of_component_m3_s
        ),
        supply_branch_gas_inflow_kg_s=supply_external.gas_inflow_kg_s,
        mouth_liquid_outflow_m3_s=vertical_external.liquid_outflow_m3_s,
        mouth_liquid_inflow_m3_s=vertical_external.liquid_inflow_m3_s,
        mouth_gas_outflow_kg_s=vertical_external.gas_outflow_kg_s,
        mouth_gas_inflow_kg_s=vertical_external.gas_inflow_kg_s,
        air_supply_liquid_volume_residual_m3_s=air.residual.liquid_volume_m3_s,
        air_supply_gas_mass_residual_kg_s=air.residual.gas_mass_kg_s,
        air_supply_momentum_x_residual_N=air.residual.mixture_momentum_x_N,
        air_supply_momentum_z_residual_N=air.residual.mixture_momentum_z_N,
        riser_liquid_volume_residual_m3_s=riser.residual.liquid_volume_m3_s,
        riser_gas_mass_residual_kg_s=riser.residual.gas_mass_kg_s,
        riser_momentum_x_residual_N=riser.residual.mixture_momentum_x_N,
        riser_momentum_z_residual_N=riser.residual.mixture_momentum_z_N,
        air_supply_reaction_x_N=air.wall_reaction_on_fluid_x_N,
        air_supply_reaction_z_N=air.wall_reaction_on_fluid_z_N,
        riser_reaction_x_N=riser.wall_reaction_on_fluid_x_N,
        riser_reaction_z_N=riser.wall_reaction_on_fluid_z_N,
        connected_water_to_mouth=connected,
    )


def average_interval_diagnostics(
    rk1: NativeIntervalDiagnostics, rk2: NativeIntervalDiagnostics
) -> NativeIntervalDiagnostics:
    """Apply the exact 1/2,1/2 SSP-RK2 weights to accepted interval data."""

    numeric_names = tuple(
        name
        for name in NativeIntervalDiagnostics.__dataclass_fields__
        if name not in ("connected_water_to_mouth", "temporal_semantics")
    )
    values = {
        name: 0.5 * (getattr(rk1, name) + getattr(rk2, name))
        for name in numeric_names
    }
    return NativeIntervalDiagnostics(
        **values,
        connected_water_to_mouth=(
            rk1.connected_water_to_mouth or rk2.connected_water_to_mouth
        ),
        temporal_semantics="accepted_interval_SSPRK2_half_RK1_plus_half_RK2",
    )


__all__ = [
    "ATMOSPHERIC_ABSOLUTE_PRESSURE_PA",
    "InstantaneousGaugePressures",
    "NativeIntervalDiagnostics",
    "PUBLISHED_PRESSURE_COORDINATES",
    "PressureSemantics",
    "average_interval_diagnostics",
    "build_instantaneous_gauge_pressures",
    "interval_diagnostics_from_stage_rate",
]
