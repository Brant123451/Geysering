"""Conservative 1D two-fluid kernel for the Cong-2017 Campaign-2 riser.

The vertical coordinate is positive upward.  The cell-centred conserved state is

``(Al, Ql, Mg, Jg)``

where ``Al`` is liquid area [m2], ``Ql = Al*u_l`` is liquid discharge
[m3/s], ``Mg`` is gas mass in a cell [kg], and ``Jg = Mg*u_g`` is gas
momentum [kg m/s].  Thus the physical liquid momentum stored in a cell is
``rho_l*Ql*dz``.  This is the same convention used by the existing Campaign-2
network solver.

The kernel owns only the vertical finite-volume state.  A Case-1 horizontal
owner supplies one already-agreed :class:`case1_persistent_coupling.TeeTransaction`
at the bottom face; this module never reads a case identifier or an experimental
geyser outcome.  The top face is an atmospheric, linear-characteristic open
boundary.
Liquid is allowed to leave that face but can never enter from the atmosphere.
The accumulated liquid and gas top-face fluxes are therefore physical boundary
integrals, not height- or plotting-based geyser criteria.

The gas--liquid drag source is the single-liquid reduction of Case 1's
``implicit_physical_three_body_drag_exchange``.  It uses the same gas-side
Fanning factor and an implicit pair solve, then closes the gas momentum from
the total cell momentum so that the two impulses are exactly equal and
opposite (apart from the final floating-point subtraction).

This file is an integration kernel, not yet the final Campaign-2 constitutive
model.  Its production default is an algebraic isothermal common-pressure
closure: every resolved gas volume obtains ``p = Mg*R*T/(Ag*dz)``, pure-liquid
cells inherit pressure from the nearest connected gas/atmospheric datum, and
the resulting cell pressures are reconstructed to one shared face field.  The
mixture-hydrostatic helper remains available only for initialization and
diagnostic base states; it is not the transient default.

The four conserved variables do not contain gas energy, liquid compressibility,
or an independent pressure-relaxation variable.  Consequently this is an
honest barotropic/isothermal intermediate closure, not a claim of a complete
two-pressure compressible solver.  The following production closures have
deliberately *not* been invented here:

* finite-speed liquid compressibility and a pressure-relaxation equation;
* a flow-regime-dependent annular/slug interfacial-perimeter closure;
* a finite-volume interior acoustic Riemann flux (interior advection remains
  first-order upwind while pressure is applied through the shared face field);
* liquid entrainment, breakup/coalescence, and regime-transition closures.

None of those omissions is replaced by a case-specific coefficient, a target
geyser flag, an imposed jet, or a scripted top outflow.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Protocol, Sequence


VERTICAL_TWOFLUID_KERNEL_READY = True
COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY = False
MISSING_PHYSICAL_CLOSURES = (
    "finite_speed_liquid_compressibility_and_pressure_relaxation",
    "flow_regime_dependent_interfacial_geometry",
    "finite_volume_interior_acoustic_riemann_flux",
    "entrainment_breakup_coalescence_and_regime_transition",
)

# These are the two independent conservative rates that a first-entry
# TeeTransaction must carry.  Existing mdot and mdot*u fix gas velocity, but
# not donor density/volume flux; liquid volume flow alone does not fix its
# convective momentum flux at a newly two-phase mouth.  The kernel validates
# both rates before it creates the first lower cut cell.
MINIMUM_FIRST_BOTTOM_GAS_INTRUSION_FIELDS = (
    "gas_volume_flow_to_riser_m3_s",
    "liquid_normal_momentum_flow_N",
    "riser_mouth_area_m2",
    "gas_open_area_m2",
    "liquid_open_area_m2",
    "blocked_riser_area_m2",
)
UPPER_INTERFACE_GEOMETRY_ROUNDOFF_ULPS = 16.0


class VerticalTwoFluidError(RuntimeError):
    """Base class for rejected vertical-kernel operations."""


class StateAdmissibilityError(VerticalTwoFluidError):
    """The supplied state is non-finite or violates phase geometry."""


class TeeTransactionRejected(VerticalTwoFluidError):
    """The requested atomic T-face exchange exceeds donor/capacity limits."""


class TeeTransactionLike(Protocol):
    """Structural interface implemented by ``case1_persistent_coupling``."""

    west_liquid_flow_m3_s: float
    east_liquid_flow_m3_s: float
    gas_mass_flow_to_riser_kg_s: float
    gas_volume_flow_to_riser_m3_s: float | None
    gas_normal_momentum_flow_N: float
    liquid_normal_momentum_flow_N: float | None
    liquid_node_gauge_pressure_Pa: float
    gas_interface_pressure_abs_Pa: float
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float

    @property
    def liquid_flow_to_riser_m3_s(self) -> float: ...


class FirstBottomGasIntrusionTransactionLike(TeeTransactionLike, Protocol):
    """Minimal extension needed by a conservative first-entry ALE closure.

    ``gas_volume_flow_to_riser_m3_s`` must be produced by the gas T Riemann
    solve from its actual upwind donor state (not reconstructed from the
    receiving riser's isothermal EOS).  ``liquid_normal_momentum_flow_N`` is
    the convective liquid momentum flux, excluding the common pressure force.
    The inherited liquid/gas pressure members must resolve to one common T
    pressure before either phase consumes it.
    """

    gas_volume_flow_to_riser_m3_s: float
    liquid_normal_momentum_flow_N: float
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _as_tuple(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise StateAdmissibilityError(f"{name} needs at least one cell")
    if not _finite(*result):
        raise StateAdmissibilityError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class VerticalTwoFluidParameters:
    """Uniform-grid geometry, material data, and shared local closures.

    ``liquid_wall_friction`` and ``gas_wall_friction`` are Darcy factors.  The
    homogeneous interfacial perimeter used by the drag source is multiplied by
    ``interphase_drag_multiplier``.  It is one shared physical coefficient; it
    is not keyed by riser diameter or case outcome.
    """

    cell_count: int
    cell_length_m: float
    diameter_m: float
    liquid_density_kg_m3: float = 998.0
    gas_constant_J_kg_K: float = 287.05
    gas_temperature_K: float = 296.15
    liquid_wave_speed_m_s: float = 28.0
    atmospheric_pressure_Pa: float = 101_325.0
    gravity_m_s2: float = 9.81
    gas_viscosity_Pa_s: float = 1.81e-5
    liquid_wall_friction: float = 0.0
    gas_wall_friction: float = 0.0
    interphase_drag_multiplier: float = 1.0
    area_tolerance_m2: float = 2.0e-14
    mass_tolerance_kg: float = 2.0e-16
    transaction_tolerance: float = 2.0e-13

    def __post_init__(self) -> None:
        if not isinstance(self.cell_count, int) or self.cell_count <= 0:
            raise ValueError("cell_count must be a positive integer")
        values = (
            self.cell_length_m,
            self.diameter_m,
            self.liquid_density_kg_m3,
            self.gas_constant_J_kg_K,
            self.gas_temperature_K,
            self.liquid_wave_speed_m_s,
            self.atmospheric_pressure_Pa,
            self.gravity_m_s2,
            self.gas_viscosity_Pa_s,
            self.liquid_wall_friction,
            self.gas_wall_friction,
            self.interphase_drag_multiplier,
            self.area_tolerance_m2,
            self.mass_tolerance_kg,
            self.transaction_tolerance,
        )
        if not _finite(*values):
            raise ValueError("vertical two-fluid parameters must be finite")
        if min(
            self.cell_length_m,
            self.diameter_m,
            self.liquid_density_kg_m3,
            self.gas_constant_J_kg_K,
            self.gas_temperature_K,
            self.liquid_wave_speed_m_s,
            self.atmospheric_pressure_Pa,
            self.gas_viscosity_Pa_s,
        ) <= 0.0:
            raise ValueError("geometry and thermodynamic properties must be positive")
        if min(
            self.gravity_m_s2,
            self.liquid_wall_friction,
            self.gas_wall_friction,
            self.interphase_drag_multiplier,
            self.area_tolerance_m2,
            self.mass_tolerance_kg,
            self.transaction_tolerance,
        ) < 0.0:
            raise ValueError("gravity, friction, drag, and tolerances cannot be negative")

    @property
    def full_area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def atmospheric_gas_density_kg_m3(self) -> float:
        return self.atmospheric_pressure_Pa / (
            self.gas_constant_J_kg_K * self.gas_temperature_K
        )

    @property
    def isothermal_gas_sound_speed_m_s(self) -> float:
        """Barotropic sound speed consistent with ``p=rho*R*T``."""

        return math.sqrt(self.gas_constant_J_kg_K * self.gas_temperature_K)


@dataclass(frozen=True)
class AtmosphericTopBoundary:
    """Pressure-anchored linear-acoustic opening at the physical riser rim."""

    pressure_abs_Pa: float = 101_325.0
    allow_gas_inflow: bool = True
    prevent_liquid_inflow: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_abs_Pa) or self.pressure_abs_Pa <= 0.0:
            raise ValueError("top atmospheric pressure must be finite and positive")
        if not self.prevent_liquid_inflow:
            raise ValueError("Campaign-2 atmosphere cannot supply liquid at the rim")


@dataclass(frozen=True)
class VerticalTwoFluidState:
    """Immutable cell state and accepted boundary-flux integrals."""

    Al: tuple[float, ...]
    Ql: tuple[float, ...]
    Mg: tuple[float, ...]
    Jg: tuple[float, ...]
    time_s: float = 0.0
    cumulative_top_liquid_outflow_m3: float = 0.0
    cumulative_top_gas_outflow_kg: float = 0.0
    cumulative_top_gas_inflow_kg: float = 0.0
    cumulative_bottom_liquid_exchange_m3: float = 0.0
    cumulative_bottom_gas_exchange_kg: float = 0.0
    # A lower material front cannot be reconstructed from Al/Ql/Mg/Jg alone:
    # the same mixed average also represents liquid below/gas above.  These two
    # checkpoint members are created only by a conservative bottom-intrusion
    # transaction and remain immutable restart data thereafter.
    lower_material_front_cell: int | None = None
    lower_material_front_orientation: str | None = None

    def __post_init__(self) -> None:
        arrays = (self.Al, self.Ql, self.Mg, self.Jg)
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise StateAdmissibilityError(
                "Al, Ql, Mg, and Jg need one common nonzero cell count"
            )
        if not _finite(*(value for values in arrays for value in values)):
            raise StateAdmissibilityError("all conserved arrays must be finite")
        scalars = (
            self.time_s,
            self.cumulative_top_liquid_outflow_m3,
            self.cumulative_top_gas_outflow_kg,
            self.cumulative_top_gas_inflow_kg,
            self.cumulative_bottom_liquid_exchange_m3,
            self.cumulative_bottom_gas_exchange_kg,
        )
        if not _finite(*scalars):
            raise StateAdmissibilityError("state clocks and ledgers must be finite")
        if self.time_s < 0.0 or min(scalars[1:4]) < 0.0:
            raise StateAdmissibilityError(
                "time and one-way top-boundary integrals cannot be negative"
            )
        marker_pair = (
            self.lower_material_front_cell,
            self.lower_material_front_orientation,
        )
        if (marker_pair[0] is None) != (marker_pair[1] is None):
            raise StateAdmissibilityError(
                "lower material-front cell and orientation must be stored together"
            )
        if marker_pair[0] is not None:
            if not isinstance(marker_pair[0], int) or not (
                0 <= marker_pair[0] < len(self.Al)
            ):
                raise StateAdmissibilityError(
                    "lower material-front cell is outside the vertical grid"
                )
            if marker_pair[1] != "gas_below_liquid_above":
                raise StateAdmissibilityError(
                    "lower material-front orientation is not recognised"
                )

    @classmethod
    def from_iterables(
        cls,
        *,
        Al: Iterable[float],
        Ql: Iterable[float],
        Mg: Iterable[float],
        Jg: Iterable[float],
        **ledger: float,
    ) -> "VerticalTwoFluidState":
        return cls(
            Al=_as_tuple(Al, name="Al"),
            Ql=_as_tuple(Ql, name="Ql"),
            Mg=_as_tuple(Mg, name="Mg"),
            Jg=_as_tuple(Jg, name="Jg"),
            **ledger,
        )

    @property
    def liquid_area(self) -> tuple[float, ...]:
        return self.Al

    @property
    def liquid_discharge(self) -> tuple[float, ...]:
        return self.Ql

    @property
    def gas_mass(self) -> tuple[float, ...]:
        return self.Mg

    @property
    def gas_momentum(self) -> tuple[float, ...]:
        return self.Jg


@dataclass(frozen=True)
class InterphaseDragLedger:
    gas_impulse_kg_m_s: tuple[float, ...]
    liquid_impulse_kg_m_s: tuple[float, ...]
    cell_residual_kg_m_s: tuple[float, ...]
    fanning_factor: tuple[float, ...]
    interface_perimeter_m: tuple[float, ...]

    @property
    def total_gas_impulse_kg_m_s(self) -> float:
        return math.fsum(self.gas_impulse_kg_m_s)

    @property
    def total_liquid_impulse_kg_m_s(self) -> float:
        return math.fsum(self.liquid_impulse_kg_m_s)

    @property
    def exchange_residual_kg_m_s(self) -> float:
        return math.fsum(self.cell_residual_kg_m_s)


@dataclass(frozen=True)
class VerticalTwoFluidBudget:
    initial_liquid_volume_m3: float
    final_liquid_volume_m3: float
    bottom_liquid_exchange_m3: float
    top_liquid_outflow_m3: float
    liquid_volume_residual_m3: float
    initial_gas_mass_kg: float
    final_gas_mass_kg: float
    bottom_gas_exchange_kg: float
    top_gas_outflow_kg: float
    top_gas_inflow_kg: float
    gas_mass_residual_kg: float
    initial_total_momentum_kg_m_s: float
    final_total_momentum_kg_m_s: float
    boundary_momentum_impulse_kg_m_s: float
    pressure_gravity_impulse_kg_m_s: float
    wall_impulse_kg_m_s: float
    interphase_exchange_residual_kg_m_s: float
    total_momentum_residual_kg_m_s: float
    initial_liquid_momentum_kg_m_s: float
    final_liquid_momentum_kg_m_s: float
    liquid_boundary_momentum_impulse_kg_m_s: float
    liquid_pressure_impulse_kg_m_s: float
    liquid_gravity_impulse_kg_m_s: float
    liquid_wall_impulse_kg_m_s: float
    liquid_interphase_impulse_kg_m_s: float
    liquid_momentum_residual_kg_m_s: float
    initial_gas_momentum_kg_m_s: float
    final_gas_momentum_kg_m_s: float
    gas_boundary_momentum_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    gas_gravity_impulse_kg_m_s: float
    gas_wall_impulse_kg_m_s: float
    gas_interphase_impulse_kg_m_s: float
    gas_momentum_residual_kg_m_s: float


@dataclass(frozen=True)
class UpperFreeSurfaceRetreatLedger:
    """Auditable conservative transfer at one liquid-below/gas-above front."""

    interface_cell: int
    interface_face: int
    interface_velocity_m_s: float
    swept_gas_volume_m3: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    gas_mass_flux_kg_s: float
    gas_momentum_flux_N: float
    liquid_volume_residual_m3: float
    receiver_gas_mass_residual_kg: float
    receiver_gas_momentum_residual_kg_m_s: float
    interface_pressure_abs_Pa: float
    liquid_pressure_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    paired_pressure_impulse_residual_kg_m_s: float


@dataclass(frozen=True)
class UpperFreeSurfaceAdvanceLedger:
    """Auditable conservative rewetting of one upper material front."""

    interface_cell: int
    interface_face: int
    interface_velocity_m_s: float
    swept_liquid_volume_m3: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    gas_mass_flux_kg_s: float
    gas_momentum_flux_N: float
    liquid_volume_residual_m3: float
    donor_gas_mass_residual_kg: float
    donor_gas_momentum_residual_kg_m_s: float
    interface_pressure_abs_Pa: float
    liquid_pressure_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    paired_pressure_impulse_residual_kg_m_s: float


@dataclass(frozen=True)
class FirstBottomGasIntrusionLedger:
    """Conservative ALE audit for creation of the lower material front."""

    lower_front_cell: int
    lower_front_orientation: str
    common_pressure_abs_Pa: float
    gas_volume_flow_m3_s: float
    gas_mass_flow_kg_s: float
    gas_momentum_flow_N: float
    liquid_bottom_volume_flow_m3_s: float
    liquid_plug_volume_flow_m3_s: float
    liquid_momentum_flow_N: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float
    swept_gas_volume_m3: float
    liquid_volume_residual_m3: float
    mixture_volume_residual_m3: float
    gas_mass_residual_kg: float
    gas_momentum_residual_kg_m_s: float
    liquid_momentum_flux_residual_N: float
    liquid_pressure_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    paired_pressure_impulse_residual_kg_m_s: float


@dataclass(frozen=True)
class BottomGasStorageLedger:
    """Auditable finite-pocket T-mouth exchange for one accepted step.

    Gas volume flux is retained independently from mass and momentum.  The
    difference between bottom volume inflow and geometric pocket growth is the
    compressive storage rate of the isothermal gas; it is not treated as a mass
    source and is never reconstructed from the receiving EOS.
    """

    common_bottom_pressure_abs_Pa: float
    gas_volume_flow_m3_s: float
    gas_mass_flow_kg_s: float
    gas_momentum_flow_N: float
    donor_gas_density_kg_m3: float | None
    donor_gas_velocity_m_s: float | None
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float
    lower_front_volume_flow_m3_s: float
    gas_pocket_volume_change_m3: float
    gas_pocket_geometry_residual_m3: float
    compressive_storage_volume_m3: float
    bottom_liquid_flow_m3_s: float
    bottom_liquid_momentum_flow_N: float


@dataclass(frozen=True)
class LowerMaterialFrontLedger:
    """Sharp gas-below/liquid-above interface and liquid-plug audit."""

    old_front_cell: int
    new_front_cell: int | None
    old_grid_aligned: bool
    new_grid_aligned: bool
    interface_velocity_m_s: float
    interface_volume_flow_m3_s: float
    interface_pressure_abs_Pa: float
    swept_volume_m3: float
    liquid_plug_volume_residual_m3: float
    gas_pocket_volume_change_m3: float
    gas_pocket_mass_change_kg: float
    bottom_gas_mass_exchange_kg: float
    top_gas_component_mass_change_kg: float
    gas_component_mass_residual_kg: float
    liquid_pressure_impulse_kg_m_s: float
    gas_pressure_impulse_kg_m_s: float
    paired_pressure_impulse_residual_kg_m_s: float


@dataclass(frozen=True)
class VerticalTwoFluidStepResult:
    state: VerticalTwoFluidState
    budget: VerticalTwoFluidBudget
    pressure_faces_Pa: tuple[float, ...]
    liquid_volume_flux_faces_m3_s: tuple[float, ...]
    gas_mass_flux_faces_kg_s: tuple[float, ...]
    liquid_momentum_flux_faces_N: tuple[float, ...]
    gas_momentum_flux_faces_N: tuple[float, ...]
    drag: InterphaseDragLedger
    upper_free_surface_retreat: UpperFreeSurfaceRetreatLedger | None = None
    upper_free_surface_advance: UpperFreeSurfaceAdvanceLedger | None = None
    first_bottom_gas_intrusion: FirstBottomGasIntrusionLedger | None = None
    bottom_gas_storage: BottomGasStorageLedger | None = None
    lower_material_front: LowerMaterialFrontLedger | None = None

    @property
    def top_liquid_outflow_m3_s(self) -> float:
        return self.liquid_volume_flux_faces_m3_s[-1]

    @property
    def top_gas_mass_flux_kg_s(self) -> float:
        return self.gas_mass_flux_faces_kg_s[-1]

    @property
    def bottom_liquid_exchange_m3_s(self) -> float:
        return self.liquid_volume_flux_faces_m3_s[0]

    @property
    def bottom_gas_exchange_kg_s(self) -> float:
        return self.gas_mass_flux_faces_kg_s[0]


def validate_state(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> None:
    """Validate geometry and the zero-inventory/zero-momentum invariants."""

    if len(state.Al) != parameters.cell_count:
        raise StateAdmissibilityError("state and parameter cell counts differ")
    area = parameters.full_area_m2
    at = parameters.area_tolerance_m2
    for cell, (al, ql, mg, jg) in enumerate(
        zip(state.Al, state.Ql, state.Mg, state.Jg)
    ):
        if al < -at or al > area + at:
            raise StateAdmissibilityError(f"liquid area outside pipe in cell {cell}")
        if mg < 0.0:
            raise StateAdmissibilityError(f"negative gas mass in cell {cell}")
        if al <= at and abs(ql) > at:
            raise StateAdmissibilityError(
                f"dry liquid phase carries discharge in cell {cell}"
            )
        gas_area = area - al
        # Resolution tolerances may suppress absent-phase source terms, but
        # they cannot decide whether a strictly positive conserved gas parcel
        # exists.  A restart has only (Al,Ql,Mg,Jg), so exact positivity is the
        # persistent material marker: every Ag>0,Mg>0 parcel owns its EOS and
        # velocity continuously on either side of the public tolerances.
        if gas_area <= 0.0 and mg > 0.0:
            raise StateAdmissibilityError(
                f"gas mass has no positive volume in cell {cell}"
            )
        if mg == 0.0 and jg != 0.0:
            raise StateAdmissibilityError(
                f"zero-mass gas carries momentum in cell {cell}"
            )
        if gas_area > at and mg == 0.0:
            raise StateAdmissibilityError(
                f"resolved gas volume has no gas mass in cell {cell}"
            )
    if state.lower_material_front_cell is not None:
        front = state.lower_material_front_cell
        assert front is not None
        front_al = state.Al[front]
        front_ag = area - front_al
        resolved_cut = bool(
            0.0 < front_al < area
            and front_ag > 0.0
            and state.Mg[front] > 0.0
        )
        grid_aligned = bool(
            front > 0
            and front_al == area
            and state.Mg[front] == 0.0
            and state.Jg[front] == 0.0
        )
        if not (resolved_cut or grid_aligned):
            raise StateAdmissibilityError(
                "persisted lower material front must own a paired cut cell or "
                "the first full-liquid cell above a grid-aligned gas pocket"
            )
        for cell in range(front):
            if not (
                state.Al[cell] == 0.0
                and state.Mg[cell] > 0.0
            ):
                raise StateAdmissibilityError(
                    "cells below a lower material front must be connected pure gas"
                )


def atmospheric_empty_state(
    parameters: VerticalTwoFluidParameters,
) -> VerticalTwoFluidState:
    """Return the discrete isothermal atmospheric hydrostatic gas column."""

    return hydrostatic_column_state(parameters, liquid_height_m=0.0)


def hydrostatic_column_state(
    parameters: VerticalTwoFluidParameters,
    *,
    liquid_height_m: float,
) -> VerticalTwoFluidState:
    """Build a bottom-anchored column with atmospheric gas above it.

    A height cutting a cell creates a homogeneous mixed interface cell.  Such a
    cell is admissible but is not an exact phase-by-phase equilibrium because the
    unresolved phases have different densities.  For a grid-aligned interface,
    gas mass is initialized from the same discrete isothermal EOS/hydrostatic
    relation used by :func:`isothermal_common_pressure_faces`; therefore the
    pure liquid and pure gas cells are exactly at rest.
    """

    if not math.isfinite(liquid_height_m) or liquid_height_m < 0.0:
        raise ValueError("liquid height must be finite and non-negative")
    dz = parameters.cell_length_m
    area = parameters.full_area_m2
    rt = parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    Al: list[float] = []
    for cell in range(parameters.cell_count):
        z0 = cell * dz
        liquid_fraction = min(max((liquid_height_m - z0) / dz, 0.0), 1.0)
        liquid_area = liquid_fraction * area
        Al.append(liquid_area)

    # March from the atmospheric rim downward.  With centred gravity, the
    # common cell pressure obeys
    # pbar = p_top + 0.5*g*dz*(rho_l*alpha_l + pbar/(RT)*alpha_g).
    # Solving that scalar relation makes every pure cell exactly well balanced
    # by the production face reconstruction, including the small gas head.
    Mg = [0.0] * parameters.cell_count
    upper_face_pressure = parameters.atmospheric_pressure_Pa
    for cell in range(parameters.cell_count - 1, -1, -1):
        alpha_l = Al[cell] / area
        alpha_g = 1.0 - alpha_l
        denominator = 1.0 - 0.5 * parameters.gravity_m_s2 * dz * alpha_g / rt
        if denominator <= 0.0:
            raise ValueError("cell is too tall for the discrete isothermal base")
        pbar = (
            upper_face_pressure
            + 0.5
            * parameters.gravity_m_s2
            * dz
            * parameters.liquid_density_kg_m3
            * alpha_l
        ) / denominator
        rho_g = pbar / rt
        gas_area = area - Al[cell]
        Mg[cell] = rho_g * gas_area * dz
        mixture_density = (
            parameters.liquid_density_kg_m3 * alpha_l
            + rho_g * alpha_g
        )
        upper_face_pressure = (
            pbar
            + 0.5 * mixture_density * parameters.gravity_m_s2 * dz
        )
    state = VerticalTwoFluidState.from_iterables(
        Al=Al,
        Ql=[0.0] * parameters.cell_count,
        Mg=Mg,
        Jg=[0.0] * parameters.cell_count,
    )
    validate_state(state, parameters)
    return state


def mixture_hydrostatic_pressure_faces(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary | None = None,
) -> tuple[float, ...]:
    """Integrate mixture weight downward for a base-state diagnostic.

    This helper is intentionally *not* the production transient default.  It
    remains useful when constructing or auditing a hydrostatic reference.
    Production stepping uses :func:`isothermal_common_pressure_faces`, whose
    gas pressure changes with the conserved gas mass and gas volume.
    """

    validate_state(state, parameters)
    boundary = top or AtmosphericTopBoundary(
        pressure_abs_Pa=parameters.atmospheric_pressure_Pa
    )
    n = parameters.cell_count
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    rho_l = parameters.liquid_density_kg_m3

    rho_atm = boundary.pressure_abs_Pa / (
        parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    )
    faces = [0.0] * (n + 1)
    faces[-1] = boundary.pressure_abs_Pa
    for cell in range(n - 1, -1, -1):
        gas_area = max(area - state.Al[cell], 0.0)
        rho_g = (
            state.Mg[cell] / (gas_area * dz)
            if gas_area > 0.0 and state.Mg[cell] > 0.0
            else rho_atm
        )
        mixture_density = (
            rho_l * state.Al[cell] + rho_g * gas_area
        ) / area
        faces[cell] = (
            faces[cell + 1]
            + mixture_density * parameters.gravity_m_s2 * dz
        )
    return tuple(faces)


def isothermal_gas_pressure_cells(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> tuple[float | None, ...]:
    """Return EOS pressure in every resolved gas volume.

    ``None`` denotes a genuinely full-liquid cell or a sub-resolution geometric
    void with exactly zero material gas.  Any strictly positive paired
    ``Ag,Mg`` inventory owns its EOS on both sides of the public resolution
    tolerances.  Gas mass without positive volume, or a resolved void without
    mass, is rejected by :func:`validate_state` before this function is reached.
    No pressure floor, target outcome, or artificial gas source is used.
    """

    validate_state(state, parameters)
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    rt = parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    result: list[float | None] = []
    for al, mg in zip(state.Al, state.Mg):
        ag = max(area - al, 0.0)
        if ag <= 0.0 or mg <= 0.0:
            result.append(None)
            continue
        pressure = mg * rt / (ag * dz)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise StateAdmissibilityError(
                "isothermal gas pressure must be finite and positive"
            )
        result.append(pressure)
    return tuple(result)


def _common_pressure_cell_centres(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
) -> tuple[float, ...]:
    """Close one common cell pressure from EOS gas and liquid continuity.

    Gas-containing cells are algebraically fixed by ``Mg`` and ``Ag``.  A
    pure-liquid cell has no independent pressure state in the present four
    variables, so its pressure is extended hydrostatically from the face above.
    If a lower gas pocket disagrees with that extension, the mismatch remains
    in the reconstructed face gradient and drives the transient; it is never
    overwritten by a hydrostatic profile.
    """

    eos = isothermal_gas_pressure_cells(state, parameters)
    n = parameters.cell_count
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    gravity = parameters.gravity_m_s2
    rho_l = parameters.liquid_density_kg_m3
    centres = [0.0] * n
    pressure_at_upper_face = top.pressure_abs_Pa
    for cell in range(n - 1, -1, -1):
        al = state.Al[cell]
        gas_line_mass = state.Mg[cell] / dz
        mixture_density = (rho_l * al + gas_line_mass) / area
        if eos[cell] is None:
            centre = pressure_at_upper_face + 0.5 * rho_l * gravity * dz
            mixture_density = rho_l
        else:
            centre = float(eos[cell])
        centres[cell] = centre
        pressure_at_upper_face = (
            centre + 0.5 * mixture_density * gravity * dz
        )
    if not _finite(*centres) or min(centres) <= 0.0:
        raise StateAdmissibilityError("common cell pressure is inadmissible")
    return tuple(centres)


def _pressure_face_predictions(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    cell_pressures_Pa: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return each cell's lower- and upper-face pressure predictions."""

    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    gravity = parameters.gravity_m_s2
    rho_l = parameters.liquid_density_kg_m3
    lower: list[float] = []
    upper: list[float] = []
    for al, mg, centre in zip(state.Al, state.Mg, cell_pressures_Pa):
        mixture_density = (rho_l * al + mg / dz) / area
        half_weight = 0.5 * mixture_density * gravity * dz
        lower.append(float(centre) + half_weight)
        upper.append(float(centre) - half_weight)
    return tuple(lower), tuple(upper)


def _transaction_bottom_common_pressure(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
    transaction: TeeTransactionLike,
    base_faces_Pa: Sequence[float],
) -> float:
    """Resolve the two T-pressure members to one physical bottom face.

    Liquid gauge pressure is relative to the connected gas-column datum.  For
    a liquid-filled bottom, that datum is the reconstructed pressure at the
    base of the lowest resolved gas cell; for an all-liquid riser it is the
    atmospheric rim pressure.  A mixed bottom has one common pressure and the
    independently supplied liquid and gas values must agree.  Rejecting an
    incompatible pair is preferable to averaging it into a hidden source.
    """

    liquid_gauge = float(transaction.liquid_node_gauge_pressure_Pa)
    gas_absolute = float(transaction.gas_interface_pressure_abs_Pa)
    if not _finite(liquid_gauge, gas_absolute) or gas_absolute <= 0.0:
        raise TeeTransactionRejected("T-interface pressures are inadmissible")

    area = parameters.full_area_m2
    if state.lower_material_front_orientation == "gas_below_liquid_above":
        # The persisted lower gas component owns the physical bottom face.
        # Horizontal liquid pressure belongs to a separate, blocked two-branch
        # node and must not be averaged with the gas T pressure.
        return gas_absolute
    bottom_al = state.Al[0]
    bottom_ag = max(area - bottom_al, 0.0)
    has_liquid = bottom_al > parameters.area_tolerance_m2
    has_gas = bottom_ag > 0.0 and state.Mg[0] > 0.0
    if not has_liquid and not has_gas:
        raise StateAdmissibilityError("bottom cell contains no resolved phase")

    gas_reference = top.pressure_abs_Pa
    for cell, al in enumerate(state.Al):
        if area - al > 0.0 and state.Mg[cell] > 0.0:
            gas_reference = float(base_faces_Pa[cell])
            break
    # When gas is present at the mouth, its solved interface pressure is the
    # local liquid gauge datum.  With a liquid-only mouth, use the base of the
    # connected gas column reconstructed from the conserved riser state.
    liquid_absolute = (
        gas_absolute + liquid_gauge
        if has_gas
        else gas_reference + liquid_gauge
    )
    if not math.isfinite(liquid_absolute) or liquid_absolute <= 0.0:
        raise TeeTransactionRejected("liquid T pressure is not positive")

    if has_liquid and has_gas:
        tolerance = 1.0e-8 * max(
            abs(liquid_absolute), abs(gas_absolute), 1.0
        )
        if abs(liquid_absolute - gas_absolute) > tolerance:
            raise TeeTransactionRejected(
                "mixed bottom cell received incompatible liquid/gas T pressures"
            )
        return 0.5 * (liquid_absolute + gas_absolute)
    if has_liquid:
        return liquid_absolute
    return gas_absolute


def isothermal_common_pressure_faces(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary | None = None,
    tee_transaction: TeeTransactionLike | None = None,
) -> tuple[float, ...]:
    """Build the production shared pressure-face field.

    The reconstruction is centred and exactly matches the discrete base state
    returned by :func:`hydrostatic_column_state` when the interface is grid
    aligned.  Away from that base, adjacent EOS pressures are not flattened:
    their two half-cell predictions are averaged at the shared face, so gas
    compression produces a real pressure gradient.  The top face is atmospheric
    and an optional T transaction supplies the bottom face exactly once.
    """

    validate_state(state, parameters)
    boundary = top or AtmosphericTopBoundary(
        pressure_abs_Pa=parameters.atmospheric_pressure_Pa
    )
    centres = _common_pressure_cell_centres(state, parameters, boundary)
    lower, upper = _pressure_face_predictions(state, parameters, centres)
    n = parameters.cell_count
    faces = [0.0] * (n + 1)
    faces[0] = lower[0]
    for face in range(1, n):
        faces[face] = 0.5 * (upper[face - 1] + lower[face])
    faces[-1] = boundary.pressure_abs_Pa
    if tee_transaction is not None:
        faces[0] = _transaction_bottom_common_pressure(
            state,
            parameters,
            boundary,
            tee_transaction,
            faces,
        )
    if not _finite(*faces) or min(faces) <= 0.0:
        raise StateAdmissibilityError("common pressure faces are inadmissible")
    return tuple(faces)


def _gas_fanning_factor(reynolds: float) -> float:
    """Case-1 gas-side Fanning-factor closure."""

    re = max(float(reynolds), 1.0e-12)
    value = 16.0 / re if re < 2100.0 else 0.046 * re**-0.2
    return min(max(value, 0.0), 4.0)


def _homogeneous_interface_perimeter(
    liquid_fraction: float,
    diameter_m: float,
) -> float:
    """Smooth provisional perimeter; zero in either pure-phase limit."""

    alpha_l = min(max(float(liquid_fraction), 0.0), 1.0)
    return math.pi * diameter_m * 4.0 * alpha_l * (1.0 - alpha_l)


def apply_equal_and_opposite_interphase_drag(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    *,
    dt: float,
) -> tuple[VerticalTwoFluidState, InterphaseDragLedger]:
    """Apply Case-1-form gas--liquid drag with an exact momentum ledger."""

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    validate_state(state, parameters)
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    rho_l = parameters.liquid_density_kg_m3
    q_new = list(state.Ql)
    j_new = list(state.Jg)
    gas_impulses: list[float] = []
    liquid_impulses: list[float] = []
    residuals: list[float] = []
    fanning: list[float] = []
    perimeters: list[float] = []

    for cell, (al, ql, mg, jg) in enumerate(
        zip(state.Al, state.Ql, state.Mg, state.Jg)
    ):
        ag = max(area - al, 0.0)
        ml = rho_l * al * dz
        liquid_is_resolved = al > parameters.area_tolerance_m2
        gas_is_resolved = ag > 0.0 and mg > 0.0
        if not liquid_is_resolved or not gas_is_resolved:
            if not liquid_is_resolved:
                q_new[cell] = 0.0
            if not gas_is_resolved:
                j_new[cell] = 0.0
            gas_impulses.append(0.0)
            liquid_impulses.append(0.0)
            residuals.append(0.0)
            fanning.append(0.0)
            perimeters.append(0.0)
            continue
        ul = ql / al
        ug = jg / mg
        relative = ug - ul
        perimeter = _homogeneous_interface_perimeter(al / area, parameters.diameter_m)
        hydraulic_diameter = max(
            min(4.0 * ag / max(perimeter, 1.0e-14), parameters.diameter_m),
            1.0e-9,
        )
        rho_g = mg / (ag * dz)
        reynolds = (
            rho_g
            * abs(relative)
            * hydraulic_diameter
            / parameters.gas_viscosity_Pa_s
        )
        friction = _gas_fanning_factor(reynolds)
        force_coefficient = (
            0.5
            * friction
            * rho_g
            * perimeter
            * dz
            * parameters.interphase_drag_multiplier
        )
        frozen_k = force_coefficient * abs(relative)
        if frozen_k <= 0.0:
            gas_impulses.append(0.0)
            liquid_impulses.append(0.0)
            residuals.append(0.0)
            fanning.append(friction)
            perimeters.append(perimeter)
            continue

        relaxation = dt * frozen_k * ml / (ml + dt * frozen_k)
        ug_new = (mg * ug + relaxation * ul) / (mg + relaxation)
        ul_new = (ml * ul + dt * frozen_k * ug_new) / (
            ml + dt * frozen_k
        )
        initial_total = jg + ml * ul
        final_liquid = ml * ul_new
        # Close the pair ledger exactly in the stored representation.
        final_gas = initial_total - final_liquid
        liquid_impulse = final_liquid - ml * ul
        gas_impulse = final_gas - jg
        q_new[cell] = final_liquid / (rho_l * dz)
        j_new[cell] = final_gas
        gas_impulses.append(gas_impulse)
        liquid_impulses.append(liquid_impulse)
        residuals.append(gas_impulse + liquid_impulse)
        fanning.append(friction)
        perimeters.append(perimeter)

    updated = replace(state, Ql=tuple(q_new), Jg=tuple(j_new))
    validate_state(updated, parameters)
    return updated, InterphaseDragLedger(
        gas_impulse_kg_m_s=tuple(gas_impulses),
        liquid_impulse_kg_m_s=tuple(liquid_impulses),
        cell_residual_kg_m_s=tuple(residuals),
        fanning_factor=tuple(fanning),
        interface_perimeter_m=tuple(perimeters),
    )


def _phase_velocities(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> tuple[list[float], list[float]]:
    ul = [
        q / a if a > parameters.area_tolerance_m2 else 0.0
        for a, q in zip(state.Al, state.Ql)
    ]
    ug = [
        j / m if m > 0.0 else 0.0
        for m, j in zip(state.Mg, state.Jg)
    ]
    return ul, ug


def _donor_limited_faces(
    fluxes: Sequence[float],
    inventory: Sequence[float],
    *,
    dt: float,
    minimum_inventory: float | Sequence[float] = 0.0,
) -> list[float]:
    """Limit signed face fluxes by the total inventory leaving each cell."""

    n = len(inventory)
    result = [float(value) for value in fluxes]
    outgoing = [0.0] * n
    donors: list[int | None] = []
    for face, flux in enumerate(result):
        donor: int | None = None
        if flux > 0.0 and face > 0:
            donor = face - 1
        elif flux < 0.0 and face < n:
            donor = face
        donors.append(donor)
        if donor is not None:
            outgoing[donor] += abs(flux)
    if isinstance(minimum_inventory, (int, float)):
        minima = [float(minimum_inventory)] * n
    else:
        minima = [float(value) for value in minimum_inventory]
        if len(minima) != n or not _finite(*minima):
            raise ValueError("minimum inventories must contain n finite values")
    factors = [
        min(
            1.0,
            max(float(inventory[cell]) - minima[cell], 0.0) / (dt * rate),
        )
        if rate > 0.0
        else 1.0
        for cell, rate in enumerate(outgoing)
    ]
    for face, donor in enumerate(donors):
        if donor is not None:
            result[face] *= factors[donor]
    return result


def _liquid_capacity_limited_faces(
    fluxes: Sequence[float],
    liquid_volume: Sequence[float],
    *,
    cell_capacity_m3: float | Sequence[float],
    dt: float,
) -> list[float]:
    """Limit incoming face fluxes by the receiver's post-outflow capacity."""

    n = len(liquid_volume)
    result = [float(value) for value in fluxes]
    incoming = [0.0] * n
    outgoing = [0.0] * n
    receivers: list[int | None] = []
    for face, flux in enumerate(result):
        receiver: int | None = None
        if flux > 0.0:
            if face < n:
                receiver = face
            if face > 0:
                outgoing[face - 1] += flux
        elif flux < 0.0:
            if face > 0:
                receiver = face - 1
            if face < n:
                outgoing[face] += -flux
        receivers.append(receiver)
        if receiver is not None:
            incoming[receiver] += abs(flux)
    if isinstance(cell_capacity_m3, (int, float)):
        capacities = [float(cell_capacity_m3)] * n
    else:
        capacities = [float(value) for value in cell_capacity_m3]
        if len(capacities) != n or not _finite(*capacities):
            raise ValueError("cell capacities must contain n finite values")
    factors: list[float] = []
    for cell in range(n):
        available_rate = (
            max(capacities[cell] - liquid_volume[cell], 0.0) / dt
            + outgoing[cell]
        )
        factors.append(
            min(1.0, available_rate / incoming[cell])
            if incoming[cell] > 0.0
            else 1.0
        )
    for face, receiver in enumerate(receivers):
        if receiver is not None:
            result[face] *= factors[receiver]
    return result


def _gas_void_limited_faces(
    fluxes: Sequence[float],
    gas_volume: Sequence[float],
    *,
    minimum_resolved_volume_m3: float,
    joint_interface_faces: frozenset[int] = frozenset(),
) -> list[float]:
    """Forbid gas inflow into a receiver with no simultaneously opened void.

    Gas is compressible, so a positive resolved void has no arbitrary mass
    capacity.  A zero-volume receiver, however, cannot accept any gas mass.
    The bottom transaction is checked after this limiter and is rejected
    atomically if it attempted that impossible exchange.  A geometrically
    reconstructed upper material interface may bypass only its one paired
    face: its liquid retreat and gas influx create exactly the same positive
    sub-cell volume in this step and are audited together downstream.
    """

    n = len(gas_volume)
    result = [float(value) for value in fluxes]
    for face, flux in enumerate(result):
        receiver: int | None = None
        if flux > 0.0 and face < n:
            receiver = face
        elif flux < 0.0 and face > 0:
            receiver = face - 1
        if (
            receiver is not None
            and face not in joint_interface_faces
            and gas_volume[receiver] <= minimum_resolved_volume_m3
        ):
            result[face] = 0.0
    return result


def _bottom_connected_saturated_liquid_cell_count(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> int:
    """Return the pure-liquid prefix hydraulically connected to the T face."""

    area = parameters.full_area_m2
    count = 0
    for al, mg in zip(state.Al, state.Mg):
        gas_area = max(area - al, 0.0)
        if gas_area <= parameters.area_tolerance_m2 and mg == 0.0:
            count += 1
            continue
        break
    return count


@dataclass(frozen=True)
class _MonotoneUpperFreeSurfaceTopology:
    full_liquid_cell_count: int
    interface_cell: int
    interface_face: int
    has_resolved_cut_cell: bool
    gas_donor_cell: int


@dataclass(frozen=True)
class _UpperFreeSurfaceRetreatFlux:
    interface_cell: int
    interface_face: int
    common_liquid_flux_m3_s: float
    swept_gas_volume_m3: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    requested_gas_mass_flux_kg_s: float
    lands_on_geometric_event: bool


@dataclass(frozen=True)
class _UpperFreeSurfaceAdvanceFlux:
    interface_cell: int
    interface_face: int
    common_liquid_flux_m3_s: float
    swept_liquid_volume_m3: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    requested_gas_mass_flux_kg_s: float
    lands_on_geometric_event: bool


@dataclass(frozen=True)
class _FirstBottomGasIntrusionFlux:
    gas_volume_flow_m3_s: float
    gas_mass_flow_kg_s: float
    gas_momentum_flow_N: float
    liquid_bottom_volume_flow_m3_s: float
    liquid_plug_volume_flow_m3_s: float
    liquid_momentum_flow_N: float
    donor_gas_density_kg_m3: float
    donor_gas_velocity_m_s: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float
    common_pressure_abs_Pa: float
    swept_gas_volume_m3: float
    lands_on_geometric_event: bool


@dataclass(frozen=True)
class _BottomGasStorageFlux:
    gas_volume_flow_m3_s: float
    gas_mass_flow_kg_s: float
    gas_momentum_flow_N: float
    donor_gas_density_kg_m3: float | None
    donor_gas_velocity_m_s: float | None
    riser_mouth_area_m2: float
    gas_open_area_m2: float
    liquid_open_area_m2: float
    blocked_riser_area_m2: float
    common_bottom_pressure_abs_Pa: float


@dataclass(frozen=True)
class _LowerMaterialFrontTopology:
    front_cell: int
    is_grid_aligned: bool
    gas_trace_cell: int
    liquid_trace_cell: int
    upper_interface_cell: int | None
    upper_interface_face: int | None
    upper_has_cut_cell: bool
    first_top_gas_cell: int | None


@dataclass(frozen=True)
class _LowerMaterialFrontStar:
    topology: _LowerMaterialFrontTopology
    interface_velocity_m_s: float
    interface_volume_flow_m3_s: float
    interface_pressure_abs_Pa: float
    gas_density_kg_m3: float
    gas_velocity_m_s: float
    liquid_velocity_m_s: float


@dataclass(frozen=True)
class _LowerMaterialFrontFlux:
    star: _LowerMaterialFrontStar
    swept_volume_m3: float
    projected_liquid_face_first: int
    projected_liquid_face_last: int
    upper_retreat: _UpperFreeSurfaceRetreatFlux | None
    upper_advance: _UpperFreeSurfaceAdvanceFlux | None
    lower_lands_on_geometric_event: bool


def _exact_geometric_event_flux(
    available_volume_m3: float,
    dt_s: float,
    reference_flux_m3_s: float,
) -> float:
    """Return the nearest signed flux whose binary64 product lands exactly.

    The event time and Riemann flux are independently rounded binary64
    numbers, so ``(V/|Q|)*|Q|`` can be one ulp to either side of ``V``.  An
    event step must not turn that double-rounding residue into a phase film.
    Use the represented quotient and, when one adjacent float has an exactly
    representable product, prefer that neighbour.  Some binary64 ``(V, dt)``
    pairs admit no flux whose rounded product is exactly ``V``; the event
    ledger therefore owns ``V`` as the exact integrated geometric exchange.
    This is an explicit event projection, not a tolerance snap.
    """

    available = float(available_volume_m3)
    dt = float(dt_s)
    reference = float(reference_flux_m3_s)
    if (
        not _finite(available, dt, reference)
        or available <= 0.0
        or dt <= 0.0
        or reference == 0.0
    ):
        raise ValueError(
            "geometric event projection requires positive V, dt, and nonzero Q"
        )
    magnitude = available / dt
    product = magnitude * dt
    if product != available:
        direction = math.inf if product < available else 0.0
        adjacent = math.nextafter(magnitude, direction)
        if adjacent * dt == available:
            magnitude = adjacent
    return math.copysign(magnitude, reference)


@dataclass(frozen=True)
class _SaturatedLiquidFluxProjection:
    liquid_fluxes_m3_s: tuple[float, ...]
    gas_fluxes_kg_s: tuple[float, ...]
    common_liquid_face_last: int
    common_liquid_face_first: int
    common_liquid_flux_m3_s: float
    retreat: _UpperFreeSurfaceRetreatFlux | None
    advance: _UpperFreeSurfaceAdvanceFlux | None
    first_bottom_gas_intrusion: _FirstBottomGasIntrusionFlux | None
    lower_material_front: _LowerMaterialFrontFlux | None = None


def _reconstruct_monotone_upper_free_surface(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> _MonotoneUpperFreeSurfaceTopology:
    """Reconstruct exactly one bottom-liquid/top-gas axial interface.

    The admissible ordering is

    ``full liquid ... [one L-below/G-above cut cell] ... full gas``.

    Every exactly positive paired phase inventory is retained as the same cut
    cell on restart, including a microscopic liquid parcel created by upward
    rewetting.  Public resolution tolerances suppress absent-phase source
    degrees of freedom; they do not erase material connectivity.  No result
    flag or case identity enters the reconstruction.  Opposite, disconnected,
    and two-interface arrangements are rejected because their face occupancy
    is not uniquely encoded by the four cell averages.
    """

    n = parameters.cell_count
    area = parameters.full_area_m2
    at = parameters.area_tolerance_m2

    def full_liquid_like(cell: int) -> bool:
        al = state.Al[cell]
        ag = max(area - al, 0.0)
        mg = state.Mg[cell]
        return al > at and ag <= at and mg == 0.0

    def resolved_cut(cell: int) -> bool:
        al = state.Al[cell]
        ag = max(area - al, 0.0)
        return al > 0.0 and ag > 0.0 and state.Mg[cell] > 0.0

    def top_gas(cell: int) -> bool:
        return (
            state.Al[cell] <= 0.0
            and max(area - state.Al[cell], 0.0) > at
            and state.Mg[cell] > 0.0
        )

    full_count = 0
    while full_count < n and full_liquid_like(full_count):
        full_count += 1
    if full_count == 0:
        raise TeeTransactionRejected(
            "upper free-surface displacement requires bottom-connected liquid "
            "with liquid-below/gas-above orientation"
        )

    cut_cell: int | None = None
    cursor = full_count
    if cursor < n and resolved_cut(cursor):
        cut_cell = cursor
        cursor += 1
    gas_start = cursor
    if gas_start >= n or not all(top_gas(cell) for cell in range(gas_start, n)):
        raise TeeTransactionRejected(
            "upper free-surface topology is not one monotone "
            "liquid-below/gas-above component"
        )

    interface_cell = full_count - 1 if cut_cell is None else cut_cell
    interface_face = interface_cell + 1
    if interface_face >= n:
        raise TeeTransactionRejected(
            "upper free-surface displacement has no material top-gas cell"
        )
    return _MonotoneUpperFreeSurfaceTopology(
        full_liquid_cell_count=full_count,
        interface_cell=interface_cell,
        interface_face=interface_face,
        has_resolved_cut_cell=cut_cell is not None,
        gas_donor_cell=interface_face,
    )


def canonicalize_upper_free_surface_roundoff(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> VerticalTwoFluidState:
    """Canonicalize only an ulp-scale exhausted phase at the upper front.

    Exact-positive inventories normally preserve the cut-cell orientation over
    a restart.  Repeated geometric-CFL substeps can eventually leave a phase
    smaller than the spacing of representable full-cell area.  Such a number
    no longer locates an interface: retaining it drives ``dt`` toward zero and
    makes the pressure projection divide by a meaningless time scale.

    The pin is tied solely to ``ulp(full_area)`` and is far below the public
    physical resolution tolerances.  Exhausted liquid is set to exact zero.
    Exhausted gas mass and momentum are transferred to the adjacent
    top-connected gas cell before the interface cell becomes exactly full, so
    no gas is created or destroyed.  Resolved cuts are returned bit-for-bit.
    """

    validate_state(state, parameters)
    try:
        topology = _reconstruct_monotone_upper_free_surface(
            state, parameters
        )
    except TeeTransactionRejected:
        return state
    if not topology.has_resolved_cut_cell:
        return state

    cell = topology.interface_cell
    area = parameters.full_area_m2
    geometric_roundoff = (
        UPPER_INTERFACE_GEOMETRY_ROUNDOFF_ULPS * math.ulp(area)
    )
    liquid_area = state.Al[cell]
    gas_area = max(area - liquid_area, 0.0)
    if 0.0 < liquid_area <= geometric_roundoff:
        al = list(state.Al)
        ql = list(state.Ql)
        al[cell] = 0.0
        ql[cell] = 0.0
        canonical = replace(state, Al=tuple(al), Ql=tuple(ql))
        validate_state(canonical, parameters)
        return canonical
    if 0.0 < gas_area <= geometric_roundoff:
        receiver = topology.interface_face
        if receiver >= parameters.cell_count:
            return state
        al = list(state.Al)
        mg = list(state.Mg)
        jg = list(state.Jg)
        al[cell] = area
        mg[receiver] += mg[cell]
        jg[receiver] += jg[cell]
        mg[cell] = 0.0
        jg[cell] = 0.0
        canonical = replace(
            state,
            Al=tuple(al),
            Mg=tuple(mg),
            Jg=tuple(jg),
        )
        validate_state(canonical, parameters)
        return canonical
    return state


def _reconstruct_lower_material_front(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> _LowerMaterialFrontTopology:
    """Reconstruct one gas-pocket/liquid-plug/top-gas axial ordering.

    ``lower_material_front_cell`` is the first cell containing liquid.  It owns
    either a gas-below/liquid-above cut or, at an exact crossing event, the
    first full-liquid cell above a grid-aligned gas pocket.  This convention is
    restart-complete without a floating interface coordinate.
    """

    validate_state(state, parameters)
    marker = state.lower_material_front_cell
    if marker is None:
        raise TeeTransactionRejected("no persisted lower material front exists")
    area = parameters.full_area_m2
    n = parameters.cell_count
    al_front = state.Al[marker]
    cut = bool(0.0 < al_front < area and state.Mg[marker] > 0.0)
    aligned = bool(
        marker > 0
        and al_front == area
        and state.Mg[marker] == 0.0
        and state.Jg[marker] == 0.0
    )
    if not (cut or aligned):
        raise StateAdmissibilityError(
            "lower material-front marker is neither cut nor grid aligned"
        )

    gas_trace_cell = marker if cut else marker - 1
    liquid_trace_cell = marker
    cursor = marker + 1 if cut else marker
    while cursor < n and state.Al[cursor] == area and state.Mg[cursor] == 0.0:
        cursor += 1

    upper_cut = False
    upper_interface_cell: int | None
    upper_interface_face: int | None
    first_top_gas: int | None
    if cursor == n:
        upper_interface_cell = None
        upper_interface_face = None
        first_top_gas = None
    else:
        al = state.Al[cursor]
        ag = area - al
        if 0.0 < al < area and ag > 0.0 and state.Mg[cursor] > 0.0:
            upper_cut = True
            upper_interface_cell = cursor
            upper_interface_face = cursor + 1
            first_top_gas = cursor + 1
            cursor += 1
        else:
            upper_interface_cell = cursor - 1
            upper_interface_face = cursor
            first_top_gas = cursor
        if first_top_gas is not None and first_top_gas >= n:
            # A cut in the rim cell has atmosphere above it; there need not be
            # a material gas cell beyond the physical domain.
            first_top_gas = None
        if cursor < n:
            for cell in range(cursor, n):
                if not (
                    state.Al[cell] == 0.0
                    and state.Mg[cell] > 0.0
                    and state.Ql[cell] == 0.0
                ):
                    raise StateAdmissibilityError(
                        "lower-front state is not one gas/liquid-plug/top-gas component"
                    )
    return _LowerMaterialFrontTopology(
        front_cell=marker,
        is_grid_aligned=aligned,
        gas_trace_cell=gas_trace_cell,
        liquid_trace_cell=liquid_trace_cell,
        upper_interface_cell=upper_interface_cell,
        upper_interface_face=upper_interface_face,
        upper_has_cut_cell=upper_cut,
        first_top_gas_cell=first_top_gas,
    )


def lower_material_front_star_state(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary | None = None,
) -> _LowerMaterialFrontStar:
    """Solve the gas-below/liquid-above linear acoustic material star state."""

    boundary = top or AtmosphericTopBoundary(
        pressure_abs_Pa=parameters.atmospheric_pressure_Pa
    )
    topology = _reconstruct_lower_material_front(state, parameters)
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    gas_cell = topology.gas_trace_cell
    liquid_cell = topology.liquid_trace_cell
    ag = area - state.Al[gas_cell]
    mg = state.Mg[gas_cell]
    al = state.Al[liquid_cell]
    if ag <= 0.0 or mg <= 0.0 or al <= 0.0:
        raise StateAdmissibilityError(
            "lower material front lacks a positive gas or liquid trace"
        )
    rho_g = mg / (ag * dz)
    u_g = state.Jg[gas_cell] / mg
    u_l = state.Ql[liquid_cell] / al
    cell_pressures = _common_pressure_cell_centres(state, parameters, boundary)
    p_g = cell_pressures[gas_cell]
    p_l = cell_pressures[liquid_cell]
    z_g = rho_g * parameters.isothermal_gas_sound_speed_m_s
    z_l = parameters.liquid_density_kg_m3 * parameters.liquid_wave_speed_m_s
    denominator = z_g + z_l
    velocity = (p_g - p_l + z_g * u_g + z_l * u_l) / denominator
    pressure = (
        z_l * p_g
        + z_g * p_l
        + z_g * z_l * (u_g - u_l)
    ) / denominator
    volume_flow = area * velocity
    if not _finite(rho_g, u_g, u_l, velocity, pressure, volume_flow):
        raise FloatingPointError("lower material-front star state is non-finite")
    if pressure <= 0.0:
        raise StateAdmissibilityError(
            "lower material-front star pressure is not positive"
        )
    return _LowerMaterialFrontStar(
        topology=topology,
        interface_velocity_m_s=float(velocity),
        interface_volume_flow_m3_s=float(volume_flow),
        interface_pressure_abs_Pa=float(pressure),
        gas_density_kg_m3=float(rho_g),
        gas_velocity_m_s=float(u_g),
        liquid_velocity_m_s=float(u_l),
    )


def lower_material_front_geometric_timestep_limit(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    *,
    cfl: float,
    top: AtmosphericTopBoundary | None = None,
) -> float:
    """Return a directional one-cell limit for both translated plug fronts."""

    factor = float(cfl)
    if not math.isfinite(factor) or not (0.0 < factor <= 1.0):
        raise ValueError("lower material-front CFL must lie in (0, 1]")
    star = lower_material_front_star_state(state, parameters, top)
    q = star.interface_volume_flow_m3_s
    if q == 0.0:
        return math.inf
    topology = star.topology
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    lower_cell = topology.front_cell
    if q > 0.0:
        lower_available = state.Al[lower_cell] * dz
    elif topology.is_grid_aligned:
        lower_available = area * dz
    else:
        lower_available = (area - state.Al[lower_cell]) * dz

    upper_available = math.inf
    if topology.upper_interface_cell is not None:
        upper = topology.upper_interface_cell
        if q > 0.0:
            target = upper if topology.upper_has_cut_cell else upper + 1
            if target < parameters.cell_count:
                upper_available = (area - state.Al[target]) * dz
        else:
            upper_available = state.Al[upper] * dz
    available = min(lower_available, upper_available)
    if available <= 0.0:
        raise TeeTransactionRejected(
            "lower material front has no directional phase volume"
        )
    return float(factor * available / abs(q))


def _authoritative_riser_mouth_partition(
    transaction: TeeTransactionLike,
    parameters: VerticalTwoFluidParameters,
    *,
    context: str,
) -> tuple[float, float, float]:
    """Validate and return gas/liquid/blocked areas from the T owner.

    Flow and convective momentum cannot identify an open but quiescent phase
    share.  The receiver therefore accepts only the explicit geometric
    partition produced by the same Riemann/T-topology solve.
    """

    raw = (
        getattr(transaction, "riser_mouth_area_m2", None),
        getattr(transaction, "gas_open_area_m2", None),
        getattr(transaction, "liquid_open_area_m2", None),
        getattr(transaction, "blocked_riser_area_m2", None),
    )
    if any(value is None for value in raw):
        raise TeeTransactionRejected(
            f"{context} requires an authoritative riser-mouth phase partition"
        )
    mouth, gas, liquid, blocked = (float(value) for value in raw)
    if not _finite(mouth, gas, liquid, blocked):
        raise TeeTransactionRejected(
            f"{context} riser-mouth partition is non-finite"
        )
    area = parameters.full_area_m2
    area_roundoff = 64.0 * math.ulp(area)
    if abs(mouth - area) > area_roundoff:
        raise TeeTransactionRejected(
            f"{context} riser-mouth area differs from the vertical geometry"
        )
    if min(gas, liquid, blocked) < 0.0:
        raise TeeTransactionRejected(
            f"{context} phase openings cannot be negative"
        )
    if abs(gas + liquid + blocked - mouth) > area_roundoff:
        raise TeeTransactionRejected(
            f"{context} gas/liquid/blocked areas do not partition the mouth"
        )
    return gas, liquid, blocked


def _prepare_bottom_gas_storage(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    transaction: TeeTransactionLike | None,
) -> _BottomGasStorageFlux | None:
    """Validate a finite bottom gas-pocket T transaction without EOS inference."""

    if state.lower_material_front_cell is None:
        return None
    if transaction is None:
        return _BottomGasStorageFlux(
            gas_volume_flow_m3_s=0.0,
            gas_mass_flow_kg_s=0.0,
            gas_momentum_flow_N=0.0,
            donor_gas_density_kg_m3=None,
            donor_gas_velocity_m_s=None,
            riser_mouth_area_m2=parameters.full_area_m2,
            gas_open_area_m2=0.0,
            liquid_open_area_m2=0.0,
            blocked_riser_area_m2=parameters.full_area_m2,
            common_bottom_pressure_abs_Pa=float(
                isothermal_gas_pressure_cells(state, parameters)[0]
            ),
        )
    gas_open, liquid_open, blocked = _authoritative_riser_mouth_partition(
        transaction,
        parameters,
        context="finite-pocket T transaction",
    )
    area_roundoff = 64.0 * math.ulp(parameters.full_area_m2)
    if liquid_open != 0.0:
        raise TeeTransactionRejected(
            "finite bottom gas pocket requires an exactly blocked liquid riser opening"
        )
    if abs(gas_open + blocked - parameters.full_area_m2) > area_roundoff:
        raise TeeTransactionRejected(
            "finite-pocket gas and blocked areas do not cover the riser mouth"
        )
    ql = float(transaction.liquid_flow_to_riser_m3_s)
    pi_l_raw = getattr(transaction, "liquid_normal_momentum_flow_N", None)
    pi_l = 0.0 if pi_l_raw is None else float(pi_l_raw)
    if ql != 0.0 or pi_l != 0.0:
        raise TeeTransactionRejected(
            "a finite bottom gas pocket blocks liquid exchange at the physical T face"
        )
    mdot = float(transaction.gas_mass_flow_to_riser_kg_s)
    qg_raw = getattr(transaction, "gas_volume_flow_to_riser_m3_s", None)
    pi_g = float(transaction.gas_normal_momentum_flow_N)
    qg = 0.0 if qg_raw is None else float(qg_raw)
    pressure = float(transaction.gas_interface_pressure_abs_Pa)
    if not _finite(mdot, qg, pi_g, pressure) or pressure <= 0.0:
        raise TeeTransactionRejected("finite-pocket gas T transaction is inadmissible")
    if mdot == 0.0 and qg == 0.0:
        if pi_g != 0.0:
            raise TeeTransactionRejected(
                "zero finite-pocket gas flow cannot carry convective momentum"
            )
        return _BottomGasStorageFlux(
            gas_volume_flow_m3_s=0.0,
            gas_mass_flow_kg_s=0.0,
            gas_momentum_flow_N=0.0,
            donor_gas_density_kg_m3=None,
            donor_gas_velocity_m_s=None,
            riser_mouth_area_m2=parameters.full_area_m2,
            gas_open_area_m2=gas_open,
            liquid_open_area_m2=liquid_open,
            blocked_riser_area_m2=blocked,
            common_bottom_pressure_abs_Pa=pressure,
        )
    if qg_raw is None or mdot == 0.0 or qg == 0.0 or pi_g <= 0.0:
        raise TeeTransactionRejected(
            "finite-pocket gas exchange requires independent signed volume, mass, and momentum fluxes"
        )
    density = mdot / qg
    velocity = pi_g / mdot
    opening = qg / velocity
    if not _finite(density, velocity, opening) or density <= 0.0 or opening <= 0.0:
        raise TeeTransactionRejected(
            "finite-pocket donor density, velocity, or opening is inadmissible"
        )
    if math.copysign(1.0, velocity) != math.copysign(1.0, qg):
        raise TeeTransactionRejected("finite-pocket gas velocity and volume-flow signs differ")
    if opening > parameters.full_area_m2 + area_roundoff:
        raise TeeTransactionRejected(
            "finite-pocket gas opening exceeds the physical riser mouth"
        )
    if abs(opening - gas_open) > area_roundoff:
        raise TeeTransactionRejected(
            "finite-pocket gas flux does not use its authoritative opening"
        )
    return _BottomGasStorageFlux(
        gas_volume_flow_m3_s=qg,
        gas_mass_flow_kg_s=mdot,
        gas_momentum_flow_N=pi_g,
        donor_gas_density_kg_m3=density,
        donor_gas_velocity_m_s=velocity,
        riser_mouth_area_m2=parameters.full_area_m2,
        gas_open_area_m2=gas_open,
        liquid_open_area_m2=liquid_open,
        blocked_riser_area_m2=blocked,
        common_bottom_pressure_abs_Pa=pressure,
    )


def _finite_pocket_applied_bottom_pressure(
    storage: _BottomGasStorageFlux,
    parameters: VerticalTwoFluidParameters,
    closed_wall_pressure_abs_Pa: float,
) -> float:
    """Area-average the open T traction and the closed-mouth reaction.

    A finite lower gas pocket blocks the liquid part of the riser mouth.  The
    gas Riemann pressure acts only on ``gas_open_area_m2``; the complementary
    blocked area is a wall and therefore carries the receiver's own closed
    boundary pressure.  In particular, an exactly closed gas port cannot
    transmit an externally supplied T pressure into the vertical momentum
    equation.
    """

    closed = float(closed_wall_pressure_abs_Pa)
    if not math.isfinite(closed) or closed <= 0.0:
        raise FloatingPointError(
            "finite-pocket closed-wall pressure must be positive and finite"
        )
    gas_area = float(storage.gas_open_area_m2)
    blocked_area = float(storage.blocked_riser_area_m2)
    area = float(parameters.full_area_m2)
    if gas_area == 0.0:
        return closed
    if blocked_area == 0.0:
        return float(storage.common_bottom_pressure_abs_Pa)
    return float(
        math.fsum(
            (
                gas_area * float(storage.common_bottom_pressure_abs_Pa),
                blocked_area * closed,
            )
        )
        / area
    )


def _prepare_first_bottom_gas_intrusion(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
    transaction: TeeTransactionLike | None,
    *,
    dt: float,
) -> _FirstBottomGasIntrusionFlux | None:
    """Validate and freeze the no-seed first lower-front transaction.

    This gate is intentionally exact in phase topology.  Public area/mass
    tolerances may suppress source terms, but they may not create a gas seed or
    decide whether a conserved parcel exists.  A strictly positive incoming gas
    mass at an exactly saturated receiver therefore needs the two independent
    transaction members specified by the design: donor-derived gas volume flow
    and liquid convective normal momentum.
    """

    if transaction is None:
        return None
    mdot = float(transaction.gas_mass_flow_to_riser_kg_s)
    qg_raw = getattr(transaction, "gas_volume_flow_to_riser_m3_s", None)
    pi_g = float(transaction.gas_normal_momentum_flow_N)
    pi_l_raw = getattr(transaction, "liquid_normal_momentum_flow_N", None)
    qg = 0.0 if qg_raw is None else float(qg_raw)
    pi_l = 0.0 if pi_l_raw is None else float(pi_l_raw)
    if not _finite(mdot, qg, pi_g, pi_l):
        raise TeeTransactionRejected(
            "first bottom gas-intrusion transaction is non-finite"
        )

    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    bottom_is_saturated = bool(
        state.Al[0] == area
        and state.Mg[0] == 0.0
        and state.Jg[0] == 0.0
        and state.lower_material_front_cell is None
        and state.lower_material_front_orientation is None
    )
    if not bottom_is_saturated:
        return None

    if mdot <= 0.0 and qg <= 0.0:
        if mdot == 0.0 and qg == 0.0 and (pi_g != 0.0 or pi_l < 0.0):
            raise TeeTransactionRejected(
                "zero first-entry gas flow cannot carry gas momentum or negative liquid momentum"
            )
        return None
    if mdot <= 0.0 or qg_raw is None or qg <= 0.0:
        raise TeeTransactionRejected(
            "positive first bottom gas exchange requires both actual donor "
            "gas_volume_flow_to_riser_m3_s and gas mass flow"
        )
    if pi_g <= 0.0:
        raise TeeTransactionRejected(
            "positive first bottom gas exchange requires positive convective gas momentum"
        )

    donor_density = mdot / qg
    donor_velocity = pi_g / mdot
    kinematic_gas_open_area = qg / donor_velocity
    if not _finite(donor_density, donor_velocity, kinematic_gas_open_area) or min(
        donor_density, donor_velocity, kinematic_gas_open_area
    ) <= 0.0:
        raise TeeTransactionRejected(
            "first-entry donor density, velocity, and gas opening must be positive"
        )
    gas_momentum_residual = pi_g - mdot * donor_velocity
    gas_volume_residual = qg - kinematic_gas_open_area * donor_velocity
    gas_scale = math.fsum((abs(pi_g), abs(mdot * donor_velocity)))
    volume_scale = math.fsum(
        (abs(qg), abs(kinematic_gas_open_area * donor_velocity))
    )
    if (
        abs(gas_momentum_residual)
        > 64.0 * math.ulp(1.0) * gas_scale
        or abs(gas_volume_residual)
        > 64.0 * math.ulp(1.0) * volume_scale
    ):
        raise TeeTransactionRejected(
            "first-entry gas mass/volume/momentum transaction is inconsistent"
        )

    gas_open_area, liquid_open_area, blocked_area = (
        _authoritative_riser_mouth_partition(
            transaction,
            parameters,
            context="first-entry T transaction",
        )
    )
    area_roundoff = 64.0 * math.ulp(area)
    if blocked_area != 0.0:
        raise TeeTransactionRejected(
            "first-entry gas and liquid openings must cover the whole riser mouth"
        )
    if abs(kinematic_gas_open_area - gas_open_area) > area_roundoff:
        raise TeeTransactionRejected(
            "first-entry gas flux does not use its authoritative opening"
        )

    ql0 = float(transaction.liquid_flow_to_riser_m3_s)
    if pi_l < 0.0:
        raise TeeTransactionRejected(
            "liquid convective normal momentum flux cannot be negative"
        )
    if ql0 == 0.0:
        if pi_l_raw is None:
            pi_l = 0.0
        if pi_l != 0.0:
            raise TeeTransactionRejected(
                "zero bottom liquid volume flow cannot carry convective momentum"
            )
    else:
        if pi_l_raw is None or pi_l <= 0.0:
            raise TeeTransactionRejected(
                "nonzero bottom liquid flow requires its Riemann-solved normal momentum"
            )
        kinematic_liquid_open_area = (
            parameters.liquid_density_kg_m3 * ql0 * ql0 / pi_l
        )
        if (
            not math.isfinite(kinematic_liquid_open_area)
            or kinematic_liquid_open_area <= 0.0
        ):
            raise TeeTransactionRejected(
                "first-entry liquid opening reconstructed from momentum is inadmissible"
            )
        if abs(kinematic_liquid_open_area - liquid_open_area) > area_roundoff:
            raise TeeTransactionRejected(
                "first-entry liquid flux does not use its authoritative opening"
            )

    if gas_open_area > area + area_roundoff:
        raise TeeTransactionRejected(
            "first-entry gas opening exceeds the physical riser mouth"
        )
    if liquid_open_area > area + area_roundoff:
        raise TeeTransactionRejected(
            "first-entry liquid opening exceeds the physical riser mouth"
        )
    if gas_open_area + liquid_open_area > area + area_roundoff:
        raise TeeTransactionRejected(
            "first-entry gas and liquid T-port areas overlap"
        )

    base_faces = isothermal_common_pressure_faces(state, parameters, top)
    liquid_absolute = _transaction_bottom_common_pressure(
        state,
        parameters,
        top,
        transaction,
        base_faces,
    )
    gas_absolute = float(transaction.gas_interface_pressure_abs_Pa)
    pressure_scale = max(abs(liquid_absolute), abs(gas_absolute), 1.0)
    pressure_roundoff = 64.0 * math.ulp(pressure_scale)
    if abs(liquid_absolute - gas_absolute) > pressure_roundoff:
        raise TeeTransactionRejected(
            "first-entry liquid and gas characteristics do not share one absolute pressure"
        )

    available_volume = state.Al[0] * dz
    crossing_time = available_volume / qg
    if dt > crossing_time:
        raise TeeTransactionRejected(
            "lower material-interface CFL would cross the first liquid cell"
        )
    lands_on_event = bool(dt == crossing_time)
    if lands_on_event:
        qg = _exact_geometric_event_flux(available_volume, dt, qg)
        donor_velocity = qg / gas_open_area
        mdot = donor_density * qg
        pi_g = mdot * donor_velocity
        swept_volume = available_volume
    else:
        swept_volume = qg * dt
    swept_area = swept_volume / dz
    if swept_volume <= 0.0 or swept_area <= 0.0 or area - swept_area == area:
        raise TeeTransactionRejected(
            "first-entry displacement is not representable on the binary64 grid"
        )
    return _FirstBottomGasIntrusionFlux(
        gas_volume_flow_m3_s=qg,
        gas_mass_flow_kg_s=mdot,
        gas_momentum_flow_N=pi_g,
        liquid_bottom_volume_flow_m3_s=ql0,
        liquid_plug_volume_flow_m3_s=ql0 + qg,
        liquid_momentum_flow_N=pi_l,
        donor_gas_density_kg_m3=donor_density,
        donor_gas_velocity_m_s=donor_velocity,
        gas_open_area_m2=gas_open_area,
        liquid_open_area_m2=liquid_open_area,
        blocked_riser_area_m2=blocked_area,
        common_pressure_abs_Pa=gas_absolute,
        swept_gas_volume_m3=swept_volume,
        lands_on_geometric_event=lands_on_event,
    )


def _project_lower_material_front_fluxes(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
    liquid_fluxes_m3_s: Sequence[float],
    gas_fluxes_kg_s: Sequence[float],
    *,
    dt: float,
) -> _SaturatedLiquidFluxProjection:
    """Translate the finite-pocket lower front and the incompressible liquid slug.

    The lower and upper material surfaces share one liquid-plug volume flux.
    No liquid crosses the gas-owned bottom face and no phase crosses either
    material surface.  Eulerian face fluxes at a grid-aligned crossing create
    the newly swept phase from its actual donor state; a cut-cell continuation
    changes geometry and storage through its two physical bounding faces.
    """

    liquid = [float(value) for value in liquid_fluxes_m3_s]
    gas = [float(value) for value in gas_fluxes_kg_s]
    star = lower_material_front_star_state(state, parameters, top)
    topology = star.topology
    q = star.interface_volume_flow_m3_s
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    n = parameters.cell_count
    front = topology.front_cell

    if liquid[0] != 0.0:
        raise TeeTransactionRejected(
            "finite bottom gas pocket cannot accept a liquid T-face flux"
        )

    if q > 0.0:
        lower_available = state.Al[front] * dz
        lower_target = front
        plug_first_face = front + 1
    elif q < 0.0 and topology.is_grid_aligned:
        lower_available = area * dz
        lower_target = front - 1
        plug_first_face = front
    else:
        lower_available = (area - state.Al[front]) * dz
        lower_target = front
        plug_first_face = front + 1

    upper_cell = topology.upper_interface_cell
    upper_face = topology.upper_interface_face
    upper_available = math.inf
    if q > 0.0 and upper_cell is not None:
        upper_target = (
            upper_cell if topology.upper_has_cut_cell else upper_cell + 1
        )
        if upper_target < n:
            upper_available = (area - state.Al[upper_target]) * dz
    elif q < 0.0:
        upper_event_cell = n - 1 if upper_cell is None else upper_cell
        upper_available = state.Al[upper_event_cell] * dz

    event_available = min(lower_available, upper_available)
    if q == 0.0:
        swept = 0.0
        lands_on_event = False
    else:
        event_time = event_available / abs(q)
        if dt > event_time:
            raise TeeTransactionRejected(
                "lower material-interface CFL would cross more than one cell"
            )
        lands_on_event = bool(dt == event_time)
        if lands_on_event:
            # Event splitting owns the last substep.  Use the exactly swept
            # phase volume as the common plug flux so both material surfaces
            # land conservatively instead of relying on a later ulp snap.
            q = _exact_geometric_event_flux(event_available, dt, q)
            swept = event_available
            star = replace(
                star,
                interface_velocity_m_s=q / area,
                interface_volume_flow_m3_s=q,
            )
        else:
            swept = abs(q) * dt
    lower_lands = bool(
        lands_on_event and lower_available == event_available
    )
    upper_lands = bool(
        lands_on_event and upper_available == event_available
    )

    # Every face below the material interface is gas-owned.  Every liquid face
    # below the first translated plug face is therefore identically closed.
    for face in range(0, plug_first_face):
        liquid[face] = 0.0
    if topology.is_grid_aligned:
        gas[front] = (
            star.gas_density_kg_m3 * q if q > 0.0 else 0.0
        )
        if q > 0.0 and gas[front] <= 0.0:
            raise TeeTransactionRejected(
                "grid-aligned lower-front advance has no gas donor flux"
            )
    gas[front + 1] = 0.0

    upper_retreat: _UpperFreeSurfaceRetreatFlux | None = None
    upper_advance: _UpperFreeSurfaceAdvanceFlux | None = None
    projected_last = plug_first_face - 1
    if q > 0.0:
        if upper_cell is None:
            # The liquid plug already reaches the physical rim.  Translation
            # ejects the same volume through that real boundary.
            for face in range(plug_first_face, n + 1):
                liquid[face] = q
            projected_last = n
        else:
            target = upper_cell if topology.upper_has_cut_cell else upper_cell + 1
            if target >= n:
                # A cut in the rim cell discharges directly through the rim.
                for face in range(plug_first_face, n + 1):
                    liquid[face] = q
                projected_last = n
            else:
                available_gas = (area - state.Al[target]) * dz
                if swept > available_gas:
                    raise TeeTransactionRejected(
                        "translated upper interface would advance across more than one gas cell"
                    )
                donor_mass = state.Mg[target]
                if available_gas <= 0.0 or donor_mass <= 0.0:
                    raise TeeTransactionRejected(
                        "translated upper interface has no positive gas donor"
                    )
                donor_density = donor_mass / available_gas
                donor_velocity = state.Jg[target] / donor_mass
                material_face = target + 1
                for face in range(plug_first_face, target + 1):
                    liquid[face] = q
                liquid[material_face] = 0.0
                gas[target] = 0.0
                gas[material_face] = donor_density * q
                projected_last = target
                upper_advance = _UpperFreeSurfaceAdvanceFlux(
                    interface_cell=target,
                    interface_face=material_face,
                    common_liquid_flux_m3_s=q,
                    swept_liquid_volume_m3=swept,
                    donor_gas_density_kg_m3=donor_density,
                    donor_gas_velocity_m_s=donor_velocity,
                    requested_gas_mass_flux_kg_s=donor_density * q,
                    lands_on_geometric_event=upper_lands,
                )
    elif q < 0.0:
        if upper_cell is None:
            upper_cell = n - 1
            upper_face = n
        assert upper_face is not None
        available_liquid = state.Al[upper_cell] * dz
        if swept > available_liquid:
            raise TeeTransactionRejected(
                "translated upper interface would retreat across more than one liquid cell"
            )
        if upper_face < n:
            donor = upper_face
            donor_volume = (area - state.Al[donor]) * dz
            donor_mass = state.Mg[donor]
            if donor_volume <= 0.0 or donor_mass <= 0.0:
                raise TeeTransactionRejected(
                    "translated upper retreat has no top-gas donor"
                )
            donor_density = donor_mass / donor_volume
            donor_velocity = state.Jg[donor] / donor_mass
        else:
            if not top.allow_gas_inflow:
                raise TeeTransactionRejected(
                    "translated rim retreat requires atmospheric gas inflow"
                )
            donor_density = top.pressure_abs_Pa / (
                parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
            )
            donor_velocity = 0.0
        for face in range(plug_first_face, upper_cell + 1):
            liquid[face] = q
        liquid[upper_face] = 0.0
        if upper_cell > lower_target:
            gas[upper_cell] = 0.0
        gas[upper_face] = donor_density * q
        projected_last = upper_cell
        upper_retreat = _UpperFreeSurfaceRetreatFlux(
            interface_cell=upper_cell,
            interface_face=upper_face,
            common_liquid_flux_m3_s=q,
            swept_gas_volume_m3=swept,
            donor_gas_density_kg_m3=donor_density,
            donor_gas_velocity_m_s=donor_velocity,
            requested_gas_mass_flux_kg_s=donor_density * q,
            lands_on_geometric_event=upper_lands,
        )
    else:
        # A static material interface is a phase wall for both components.
        if topology.is_grid_aligned:
            liquid[front] = 0.0
            gas[front] = 0.0
        else:
            liquid[front] = 0.0
            gas[front + 1] = 0.0

    return _SaturatedLiquidFluxProjection(
        liquid_fluxes_m3_s=tuple(liquid),
        gas_fluxes_kg_s=tuple(gas),
        common_liquid_face_last=projected_last,
        common_liquid_face_first=plug_first_face,
        common_liquid_flux_m3_s=q,
        retreat=upper_retreat,
        advance=upper_advance,
        first_bottom_gas_intrusion=None,
        lower_material_front=_LowerMaterialFrontFlux(
            star=star,
            swept_volume_m3=swept,
            projected_liquid_face_first=plug_first_face,
            projected_liquid_face_last=projected_last,
            upper_retreat=upper_retreat,
            upper_advance=upper_advance,
            lower_lands_on_geometric_event=lower_lands,
        ),
    )


def _resolve_lower_front_after_transport(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    lower: _LowerMaterialFrontFlux | None,
    al_new: list[float],
    liquid_momentum_new: list[float],
    mg_new: list[float],
    gas_momentum_new: list[float],
) -> tuple[int | None, str | None]:
    """Event-split one lower-front crossing and preserve all phase momenta."""

    if lower is None:
        return state.lower_material_front_cell, state.lower_material_front_orientation
    topology = lower.star.topology
    q = lower.star.interface_volume_flow_m3_s
    area = parameters.full_area_m2
    if q > 0.0:
        target = topology.front_cell
        if lower.lower_lands_on_geometric_event:
            next_cell = target + 1
            if next_cell >= parameters.cell_count or al_new[next_cell] <= 0.0:
                raise TeeTransactionRejected(
                    "lower material front collided with the upper gas component"
                )
            liquid_momentum_new[next_cell] += liquid_momentum_new[target]
            liquid_momentum_new[target] = 0.0
            al_new[target] = 0.0
            return next_cell, "gas_below_liquid_above"
        if al_new[target] <= 0.0:
            raise TeeTransactionRejected(
                "lower material front reached a face without an event split"
            )
        return target, "gas_below_liquid_above"
    if q < 0.0:
        target = (
            topology.front_cell - 1
            if topology.is_grid_aligned
            else topology.front_cell
        )
        gas_area = area - al_new[target]
        if lower.lower_lands_on_geometric_event:
            if target > 0:
                mg_new[target - 1] += mg_new[target]
                gas_momentum_new[target - 1] += gas_momentum_new[target]
                mg_new[target] = 0.0
                gas_momentum_new[target] = 0.0
                al_new[target] = area
                return target, "gas_below_liquid_above"
            if mg_new[target] == 0.0 and gas_momentum_new[target] == 0.0:
                al_new[target] = area
                return None, None
            raise TeeTransactionRejected(
                "bottom gas pocket cannot disappear while it retains mass"
            )
        if gas_area <= 0.0:
            raise TeeTransactionRejected(
                "lower material front reached a face without an event split"
            )
        return target, "gas_below_liquid_above"
    return state.lower_material_front_cell, state.lower_material_front_orientation


def _resolve_upper_front_after_transport(
    parameters: VerticalTwoFluidParameters,
    retreat: _UpperFreeSurfaceRetreatFlux | None,
    advance: _UpperFreeSurfaceAdvanceFlux | None,
    al_new: list[float],
    liquid_momentum_new: list[float],
    mg_new: list[float],
    gas_momentum_new: list[float],
) -> None:
    """Pin only an event-exhausted upper phase and transfer residual momentum."""

    area = parameters.full_area_m2
    if advance is not None:
        cell = advance.interface_cell
        gas_area = area - al_new[cell]
        if gas_area < 0.0 and not advance.lands_on_geometric_event:
            raise TeeTransactionRejected(
                "upper material front advanced across more than one gas cell"
            )
        if advance.lands_on_geometric_event:
            receiver = cell + 1
            if receiver >= parameters.cell_count:
                if mg_new[cell] != 0.0 or gas_momentum_new[cell] != 0.0:
                    raise TeeTransactionRejected(
                        "upper gas cannot vanish at the rim with positive inventory"
                    )
            else:
                mg_new[receiver] += mg_new[cell]
                gas_momentum_new[receiver] += gas_momentum_new[cell]
            mg_new[cell] = 0.0
            gas_momentum_new[cell] = 0.0
            al_new[cell] = area
    if retreat is not None:
        cell = retreat.interface_cell
        liquid_area = al_new[cell]
        if liquid_area < 0.0 and not retreat.lands_on_geometric_event:
            raise TeeTransactionRejected(
                "upper material front retreated across more than one liquid cell"
            )
        if retreat.lands_on_geometric_event:
            receiver = cell - 1
            if receiver < 0 or al_new[receiver] <= 0.0:
                raise TeeTransactionRejected(
                    "upper material front collided with the lower gas component"
                )
            liquid_momentum_new[receiver] += liquid_momentum_new[cell]
            liquid_momentum_new[cell] = 0.0
            al_new[cell] = 0.0


def _lower_gas_component_inventory(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
) -> tuple[float, float, float]:
    """Return lower-pocket volume, mass and momentum from persisted topology."""

    marker = state.lower_material_front_cell
    if marker is None:
        return 0.0, 0.0, 0.0
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    last_gas_cell = marker if state.Al[marker] < area else marker - 1
    if last_gas_cell < 0:
        return 0.0, 0.0, 0.0
    volume = math.fsum(
        (area - state.Al[cell]) * dz for cell in range(last_gas_cell + 1)
    )
    mass = math.fsum(state.Mg[: last_gas_cell + 1])
    momentum = math.fsum(state.Jg[: last_gas_cell + 1])
    return float(volume), float(mass), float(momentum)


def _project_saturated_liquid_volume_fluxes(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
    liquid_fluxes_m3_s: Sequence[float],
    gas_fluxes_kg_s: Sequence[float],
    *,
    bottom_gas_flux_kg_s: float,
    first_bottom_gas_intrusion: _FirstBottomGasIntrusionFlux | None,
    dt: float,
) -> _SaturatedLiquidFluxProjection:
    """Enforce one volume flux through a bottom-connected saturated block.

    With no gas at the T face, a connected incompressible liquid block cannot
    change ``Al`` by accumulating volume in one of its saturated cells.  Its
    bottom volume flux is therefore the same at every face up to the represented
    free-surface cell.  This changes internal face fluxes only: total liquid
    volume remains governed exclusively by the physical bottom and top faces.

    For a negative bottom flux, the unique monotone upper free surface retreats
    geometrically: the common liquid flux stops at the lower face of the
    interface cell, the liquid flux through its upper material face is zero,
    and top-connected donor gas crosses that same upper face downward.

    For a positive bottom flux below an upper free surface, the same geometry
    is reversed.  The common liquid flux reaches only the lower material face
    of the current interface cell.  Its upper material face carries no liquid
    from the gas cell.  The liquid-swept gas volume leaves that interface cell
    upward using the cell's actual donor density and velocity.  Both directions
    use conservative face fluxes only; no atmospheric fill, seed inventory, or
    case-dependent rule is used.
    """

    if state.lower_material_front_cell is not None:
        if first_bottom_gas_intrusion is not None:
            raise TeeTransactionRejected(
                "first entry and finite-pocket continuation cannot share one step"
            )
        return _project_lower_material_front_fluxes(
            state,
            parameters,
            top,
            liquid_fluxes_m3_s,
            gas_fluxes_kg_s,
            dt=dt,
        )

    liquid = [float(value) for value in liquid_fluxes_m3_s]
    gas = [float(value) for value in gas_fluxes_kg_s]
    count = _bottom_connected_saturated_liquid_cell_count(state, parameters)
    gas_flux = float(bottom_gas_flux_kg_s)
    if count > 0 and gas_flux > 0.0 and first_bottom_gas_intrusion is None:
        raise TeeTransactionRejected(
            "first bottom gas exchange/intrusion into a saturated liquid block "
            "requires a joint gas/liquid displacement closure and transaction "
            "fields gas_volume_flow_to_riser_m3_s plus "
            "liquid_normal_momentum_flow_N"
        )

    common_flux = (
        float(first_bottom_gas_intrusion.liquid_plug_volume_flow_m3_s)
        if first_bottom_gas_intrusion is not None
        else float(liquid[0])
    )
    if first_bottom_gas_intrusion is not None:
        gas[0] = first_bottom_gas_intrusion.gas_mass_flow_kg_s
    common_face_first = 1 if first_bottom_gas_intrusion is not None else 0
    if common_flux < 0.0:
        topology = _reconstruct_monotone_upper_free_surface(
            state, parameters
        )
        interface_cell = topology.interface_cell
        interface_face = topology.interface_face
        available_liquid = state.Al[interface_cell] * parameters.cell_length_m
        crossing_time = available_liquid / abs(common_flux)
        if dt > crossing_time:
            raise TeeTransactionRejected(
                "upper free-surface interface CFL would retreat across "
                "more than one liquid cell"
            )
        lands_on_event = bool(dt == crossing_time)
        if lands_on_event:
            common_flux = _exact_geometric_event_flux(
                available_liquid,
                dt,
                common_flux,
            )
            if first_bottom_gas_intrusion is None:
                liquid[0] = common_flux
            swept_volume = available_liquid
        else:
            swept_volume = -common_flux * dt
        donor = topology.gas_donor_cell
        donor_gas_area = max(
            parameters.full_area_m2 - state.Al[donor], 0.0
        )
        donor_mass = state.Mg[donor]
        donor_volume = donor_gas_area * parameters.cell_length_m
        if donor_volume <= 0.0 or donor_mass <= 0.0:
            raise TeeTransactionRejected(
                "upper free-surface retreat has no positive top-gas donor"
            )
        donor_density = donor_mass / donor_volume
        donor_velocity = state.Jg[donor] / donor_mass
        requested_gas_flux = donor_density * common_flux
        requested_mass = -requested_gas_flux * dt
        retained_mass = (
            2.0 * parameters.mass_tolerance_kg
            if donor_mass > parameters.mass_tolerance_kg
            else 0.0
        )
        available_mass = max(donor_mass - retained_mass, 0.0)
        mass_scale = max(requested_mass, available_mass)
        mass_roundoff = 128.0 * math.ulp(1.0) * mass_scale
        if requested_mass > available_mass + mass_roundoff:
            raise TeeTransactionRejected(
                "top-gas donor cannot supply the retreating free surface"
            )

        # Faces [0, interface_face) carry the common downward liquid flux.
        # The material interface face itself carries no liquid from the dry
        # donor; its gas flux is the paired, opposite-direction displacement.
        for face in range(1, interface_face):
            liquid[face] = common_flux
        liquid[interface_face] = 0.0
        # The cut cell contains gas only above its reconstructed material
        # interface.  Its lower face borders the bottom-connected saturated
        # liquid and therefore cannot carry gas, even when the cut-cell gas
        # velocity points downward.  This is a geometric face occupancy, not
        # a flux dead zone.
        if interface_cell > 0:
            gas[interface_cell] = 0.0
        gas[interface_face] = requested_gas_flux
        return _SaturatedLiquidFluxProjection(
            liquid_fluxes_m3_s=tuple(liquid),
            gas_fluxes_kg_s=tuple(gas),
            common_liquid_face_last=interface_face - 1,
            common_liquid_face_first=common_face_first,
            common_liquid_flux_m3_s=common_flux,
            retreat=_UpperFreeSurfaceRetreatFlux(
                interface_cell=interface_cell,
                interface_face=interface_face,
                common_liquid_flux_m3_s=common_flux,
                swept_gas_volume_m3=swept_volume,
                donor_gas_density_kg_m3=donor_density,
                donor_gas_velocity_m_s=donor_velocity,
                requested_gas_mass_flux_kg_s=requested_gas_flux,
                lands_on_geometric_event=lands_on_event,
            ),
            advance=None,
            first_bottom_gas_intrusion=first_bottom_gas_intrusion,
        )

    if common_flux > 0.0 and 0 < count < parameters.cell_count:
        topology = _reconstruct_monotone_upper_free_surface(
            state, parameters
        )
        # A grid-aligned surface advances into the first gas cell.  Once a cut
        # cell exists, that same cell remains the material-interface owner.
        interface_cell = (
            topology.interface_cell
            if topology.has_resolved_cut_cell
            else topology.full_liquid_cell_count
        )
        interface_face = interface_cell + 1
        if interface_cell >= parameters.cell_count:
            raise TeeTransactionRejected(
                "upper free-surface advance has no gas interface cell"
            )
        gas_area = max(
            parameters.full_area_m2 - state.Al[interface_cell], 0.0
        )
        available_gas_volume = gas_area * parameters.cell_length_m
        crossing_time = available_gas_volume / common_flux
        if dt > crossing_time:
            raise TeeTransactionRejected(
                "upper free-surface interface CFL would advance across "
                "more than one gas cell"
            )
        lands_on_event = bool(dt == crossing_time)
        if lands_on_event:
            common_flux = _exact_geometric_event_flux(
                available_gas_volume,
                dt,
                common_flux,
            )
            if first_bottom_gas_intrusion is None:
                liquid[0] = common_flux
            swept_volume = available_gas_volume
        else:
            swept_volume = common_flux * dt
        donor_mass = state.Mg[interface_cell]
        if available_gas_volume <= 0.0 or donor_mass <= 0.0:
            raise TeeTransactionRejected(
                "upper free-surface advance has no positive interface-cell gas donor"
            )
        donor_density = donor_mass / available_gas_volume
        donor_velocity = state.Jg[interface_cell] / donor_mass
        requested_gas_flux = donor_density * common_flux
        requested_mass = requested_gas_flux * dt
        mass_scale = max(requested_mass, donor_mass)
        mass_roundoff = 128.0 * math.ulp(1.0) * mass_scale
        if requested_mass > donor_mass + mass_roundoff:
            raise TeeTransactionRejected(
                "interface-cell gas donor cannot supply the advancing free surface"
            )

        # Faces [0, interface_cell] carry the common upward liquid flux into
        # the interface cell.  Its upper material face has zero liquid flux;
        # the displaced gas crosses that face upward.  The lower face remains
        # occupied only by the bottom-connected liquid phase.
        for face in range(1, interface_cell + 1):
            liquid[face] = common_flux
        liquid[interface_face] = 0.0
        if interface_cell > 0:
            gas[interface_cell] = 0.0
        gas[interface_face] = requested_gas_flux
        return _SaturatedLiquidFluxProjection(
            liquid_fluxes_m3_s=tuple(liquid),
            gas_fluxes_kg_s=tuple(gas),
            common_liquid_face_last=interface_cell,
            common_liquid_face_first=common_face_first,
            common_liquid_flux_m3_s=common_flux,
            retreat=None,
            advance=_UpperFreeSurfaceAdvanceFlux(
                interface_cell=interface_cell,
                interface_face=interface_face,
                common_liquid_flux_m3_s=common_flux,
                swept_liquid_volume_m3=swept_volume,
                donor_gas_density_kg_m3=donor_density,
                donor_gas_velocity_m_s=donor_velocity,
                requested_gas_mass_flux_kg_s=requested_gas_flux,
                lands_on_geometric_event=lands_on_event,
            ),
            first_bottom_gas_intrusion=first_bottom_gas_intrusion,
        )

    if count == 0:
        return _SaturatedLiquidFluxProjection(
            liquid_fluxes_m3_s=tuple(liquid),
            gas_fluxes_kg_s=tuple(gas),
            common_liquid_face_last=0,
            common_liquid_face_first=common_face_first,
            common_liquid_flux_m3_s=common_flux,
            retreat=None,
            advance=None,
            first_bottom_gas_intrusion=first_bottom_gas_intrusion,
        )
    for face in range(1, count + 1):
        liquid[face] = common_flux
    return _SaturatedLiquidFluxProjection(
        liquid_fluxes_m3_s=tuple(liquid),
        gas_fluxes_kg_s=tuple(gas),
        common_liquid_face_last=count,
        common_liquid_face_first=common_face_first,
        common_liquid_flux_m3_s=common_flux,
        retreat=None,
        advance=None,
        first_bottom_gas_intrusion=first_bottom_gas_intrusion,
    )


def _verify_saturated_liquid_volume_flux_projection(
    liquid_fluxes_m3_s: Sequence[float],
    *,
    common_liquid_face_first: int,
    common_liquid_face_last: int,
    common_flux_m3_s: float,
) -> None:
    """Reject any donor/capacity limiter that breaks the projected block."""

    first = int(common_liquid_face_first)
    if common_liquid_face_last < first:
        return
    projected = tuple(
        float(liquid_fluxes_m3_s[face])
        for face in range(first, common_liquid_face_last + 1)
    )
    common = float(common_flux_m3_s)
    scale = math.fsum(abs(value) for value in (*projected, common))
    tolerance = 64.0 * math.ulp(1.0) * scale
    if any(abs(value - common) > tolerance for value in projected):
        raise TeeTransactionRejected(
            "the represented free surface cannot accept the projected "
            "saturated-column liquid flux"
        )


def _project_saturated_liquid_pressure_faces(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    pressure_faces_Pa: Sequence[float],
    transported_liquid_momentum_kg_m_s: Sequence[float],
    *,
    dt: float,
) -> tuple[float, ...]:
    """Apply the algebraic incompressible pressure constraint to the prefix.

    Bottom T pressure and the already reconstructed liquid-surface pressure are
    retained as the two physical endpoints.  Interior pressures are the unique
    one-dimensional projection that gives every saturated cell one common
    post-source discharge, including the existing gravity and wall terms.  No
    liquid volume or momentum is added by hand; pressure remains the Lagrange
    multiplier of the incompressibility constraint.
    """

    count = _bottom_connected_saturated_liquid_cell_count(state, parameters)
    faces = [float(value) for value in pressure_faces_Pa]
    if count == 0:
        return tuple(faces)
    if len(faces) != parameters.cell_count + 1:
        raise ValueError("pressure projection requires n+1 face values")
    if len(transported_liquid_momentum_kg_m_s) != parameters.cell_count:
        raise ValueError("pressure projection requires n liquid momenta")

    rho_l = parameters.liquid_density_kg_m3
    dz = parameters.cell_length_m
    gravity = parameters.gravity_m_s2
    inv_areas: list[float] = []
    discharges: list[float] = []
    wall_forces: list[float] = []
    for cell in range(count):
        al = state.Al[cell]
        momentum = float(transported_liquid_momentum_kg_m_s[cell])
        ml = rho_l * al * dz
        velocity = momentum / ml
        wall_force = (
            -0.5
            * parameters.liquid_wall_friction
            * ml
            / parameters.diameter_m
            * velocity
            * abs(velocity)
        )
        inv_areas.append(1.0 / al)
        discharges.append(momentum / (rho_l * dz))
        wall_forces.append(wall_force)

    inv_area_sum = math.fsum(inv_areas)
    endpoint_drop = faces[count] - faces[0]
    weighted_discharge = math.fsum(
        discharge * inv_area
        for discharge, inv_area in zip(discharges, inv_areas)
    )
    weighted_wall = math.fsum(
        wall_force * inv_area
        for wall_force, inv_area in zip(wall_forces, inv_areas)
    )
    target_discharge = (
        weighted_discharge
        + dt
        / (rho_l * dz)
        * (
            -endpoint_drop
            - count * rho_l * gravity * dz
            + weighted_wall
        )
    ) / inv_area_sum

    pressure_drops = [
        -rho_l
        * dz
        / (dt * al)
        * (target_discharge - discharge)
        - rho_l * gravity * dz
        + wall_force / al
        for al, discharge, wall_force in zip(
            state.Al[:count],
            discharges,
            wall_forces,
        )
    ]
    closure = endpoint_drop - math.fsum(pressure_drops)
    pressure_drops = [
        drop + closure * inv_area / inv_area_sum
        for drop, inv_area in zip(pressure_drops, inv_areas)
    ]

    pressure = faces[0]
    for cell, drop in enumerate(pressure_drops):
        pressure += drop
        faces[cell + 1] = pressure
    # The weighted closure above makes this assignment a roundoff pin only;
    # retain the independently reconstructed physical surface endpoint exactly.
    faces[count] = float(pressure_faces_Pa[count])
    if not _finite(*faces) or min(faces) <= 0.0:
        raise StateAdmissibilityError(
            "saturated-liquid pressure projection is inadmissible"
        )
    return tuple(faces)


def _transaction_fluxes(
    transaction: TeeTransactionLike | None,
) -> tuple[float, float, float, float | None]:
    if transaction is None:
        return 0.0, 0.0, 0.0, None
    values = (
        float(transaction.liquid_flow_to_riser_m3_s),
        float(transaction.gas_mass_flow_to_riser_kg_s),
        float(transaction.gas_normal_momentum_flow_N),
        (
            None
            if transaction.liquid_normal_momentum_flow_N is None
            else float(transaction.liquid_normal_momentum_flow_N)
        ),
        float(transaction.liquid_node_gauge_pressure_Pa),
        float(transaction.gas_interface_pressure_abs_Pa),
    )
    finite_values = tuple(value for value in values if value is not None)
    if not _finite(*finite_values):
        raise ValueError("TeeTransaction fields must be finite")
    if values[5] <= 0.0:
        raise ValueError("gas T-interface pressure must be positive")
    return values[:4]


def _linear_characteristic_top_gas_flux(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
) -> tuple[float, float]:
    """Return conservative top gas mass flux and its convective velocity.

    At the upward boundary the outgoing acoustic invariant gives
    ``u* = u_i + (p_i,face-p_atm)/(rho_i*c_iso)``.  The interior EOS pressure
    is extrapolated through the upper half-cell gas weight before comparison
    with atmosphere, making the discrete isothermal hydrostatic state exactly
    quiescent.  Density is upwinded for mass conservation; the returned
    momentum flux is later formed only as ``mdot*u*``.
    """

    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    ag = max(area - state.Al[-1], 0.0)
    mg = state.Mg[-1]
    if ag <= 0.0 or mg <= 0.0:
        return 0.0, 0.0
    rho_i = mg / (ag * dz)
    p_cell = (
        rho_i
        * parameters.gas_constant_J_kg_K
        * parameters.gas_temperature_K
    )
    p_face_interior = p_cell - 0.5 * rho_i * parameters.gravity_m_s2 * dz
    u_i = state.Jg[-1] / mg
    sound_speed = parameters.isothermal_gas_sound_speed_m_s
    pressure_residual = p_face_interior - top.pressure_abs_Pa
    scale = max(abs(p_face_interior), abs(top.pressure_abs_Pa), 1.0)
    if abs(pressure_residual) <= 8.0e-15 * scale and abs(u_i) <= 1.0e-14:
        return 0.0, 0.0
    u_star = u_i + pressure_residual / (rho_i * sound_speed)
    if u_star < 0.0 and not top.allow_gas_inflow:
        return 0.0, 0.0
    rho_flux = (
        rho_i
        if u_star >= 0.0
        else top.pressure_abs_Pa
        / (parameters.gas_constant_J_kg_K * parameters.gas_temperature_K)
    )
    return rho_flux * ag * u_star, u_star


def _upwind_face_fluxes(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    top: AtmosphericTopBoundary,
    transaction: TeeTransactionLike | None,
    *,
    first_bottom_gas_intrusion: _FirstBottomGasIntrusionFlux | None,
    dt: float,
) -> tuple[
    list[float],
    list[float],
    float,
    float | None,
    float,
    _UpperFreeSurfaceRetreatFlux | None,
    _UpperFreeSurfaceAdvanceFlux | None,
    _LowerMaterialFrontFlux | None,
]:
    """Build and positivity-limit liquid-volume and gas-mass face fluxes."""

    n = parameters.cell_count
    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    ul, ug = _phase_velocities(state, parameters)
    (
        bottom_liquid,
        bottom_gas,
        bottom_gas_momentum,
        bottom_liquid_momentum,
    ) = _transaction_fluxes(transaction)

    liquid = [0.0] * (n + 1)
    gas = [0.0] * (n + 1)
    liquid[0] = bottom_liquid
    gas[0] = bottom_gas
    for face in range(1, n):
        u_l_face = 0.5 * (ul[face - 1] + ul[face])
        liquid[face] = (
            u_l_face * state.Al[face - 1]
            if u_l_face >= 0.0
            else u_l_face * state.Al[face]
        )
        u_g_face = 0.5 * (ug[face - 1] + ug[face])
        gas_line_density = (
            state.Mg[face - 1] / dz
            if u_g_face >= 0.0
            else state.Mg[face] / dz
        )
        gas[face] = u_g_face * gas_line_density

    # A physical atmosphere contains no liquid reservoir above the rim.
    liquid[-1] = max(state.Ql[-1], 0.0)
    gas[-1], top_gas_velocity = _linear_characteristic_top_gas_flux(
        state, parameters, top
    )

    projection = _project_saturated_liquid_volume_fluxes(
        state,
        parameters,
        top,
        liquid,
        gas,
        bottom_gas_flux_kg_s=gas[0],
        first_bottom_gas_intrusion=first_bottom_gas_intrusion,
        dt=dt,
    )
    liquid = list(projection.liquid_fluxes_m3_s)
    gas = list(projection.gas_fluxes_kg_s)

    if projection.first_bottom_gas_intrusion is not None:
        bottom_gas_momentum = (
            projection.first_bottom_gas_intrusion.gas_momentum_flow_N
        )
    if liquid[0] != bottom_liquid and bottom_liquid_momentum is not None:
        if transaction is None:
            raise TeeTransactionRejected(
                "an event-projected T flux requires its transaction"
            )
        liquid_open_raw = getattr(
            transaction,
            "liquid_open_area_m2",
            None,
        )
        if liquid_open_raw is None or float(liquid_open_raw) <= 0.0:
            raise TeeTransactionRejected(
                "an event-projected liquid T flux requires its open area"
            )
        bottom_liquid_momentum = (
            parameters.liquid_density_kg_m3
            * liquid[0]
            * liquid[0]
            / float(liquid_open_raw)
        )

    liquid_requested = liquid[0]
    gas_requested = gas[0]
    liquid = _donor_limited_faces(
        liquid,
        [value * dz for value in state.Al],
        dt=dt,
    )
    liquid = _liquid_capacity_limited_faces(
        liquid,
        [value * dz for value in state.Al],
        cell_capacity_m3=area * dz,
        dt=dt,
    )
    # The projected saturated plug has already checked its sole directional
    # material-interface donor/capacity.  Every interior projected cell has
    # identical inflow and outflow, so applying independent per-face limiters
    # can only break incompressibility through floating-point factor ordering.
    # Preserve the exact common flux; this is the pressure-constrained block,
    # not a limiter bypass for an unverified donor.
    for face in range(
        projection.common_liquid_face_first,
        projection.common_liquid_face_last + 1,
    ):
        liquid[face] = projection.common_liquid_flux_m3_s
    gas_minimum_inventory = [
        2.0 * parameters.mass_tolerance_kg
        if mg > parameters.mass_tolerance_kg
        else 0.0
        for mg in state.Mg
    ]
    if projection.advance is not None:
        # The joint interface flux already limits displaced mass by the exact
        # donor inventory and geometric swept volume.  A public resolution
        # reserve here would create an artificial terminal gas film.
        gas_minimum_inventory[projection.advance.interface_cell] = 0.0
    if (
        projection.lower_material_front is not None
        and projection.lower_material_front.star.interface_volume_flow_m3_s < 0.0
    ):
        lower_topology = projection.lower_material_front.star.topology
        retreat_cell = (
            lower_topology.front_cell - 1
            if lower_topology.is_grid_aligned
            else lower_topology.front_cell
        )
        gas_minimum_inventory[retreat_cell] = 0.0
    gas = _donor_limited_faces(
        gas,
        state.Mg,
        dt=dt,
        minimum_inventory=gas_minimum_inventory,
    )

    # Gas may enter only where the simultaneous liquid update has opened a
    # real void.  This prevents Mg from ever being placed in Ag=0.
    liquid_area_trial = [
        state.Al[cell]
        - dt / dz * (liquid[cell + 1] - liquid[cell])
        for cell in range(n)
    ]
    gas_volume_trial = [
        max(area - al, 0.0) * dz for al in liquid_area_trial
    ]
    gas = _gas_void_limited_faces(
        gas,
        gas_volume_trial,
        minimum_resolved_volume_m3=parameters.area_tolerance_m2 * dz,
        joint_interface_faces=frozenset(
            (
                *(
                    item.interface_face
                    for item in (projection.retreat, projection.advance)
                    if item is not None
                ),
                *(
                    (0,)
                    if projection.first_bottom_gas_intrusion is not None
                    else ()
                ),
                *(
                    (
                        projection.lower_material_front.star.topology.front_cell,
                    )
                    if (
                        projection.lower_material_front is not None
                        and projection.lower_material_front.star.topology.is_grid_aligned
                        and projection.lower_material_front.star.interface_volume_flow_m3_s
                        > 0.0
                    )
                    else ()
                ),
            )
        ),
    )

    # Conversely, liquid cannot erase the last resolved gas volume while gas
    # mass remains after its conservative face update.
    gas_mass_trial = [
        state.Mg[cell] - dt * (gas[cell + 1] - gas[cell])
        for cell in range(n)
    ]
    capacities = []
    for cell, mg in enumerate(gas_mass_trial):
        is_joint_advance_donor = bool(
            projection.advance is not None
            and cell == projection.advance.interface_cell
        )
        is_first_entry_receiver = bool(
            projection.first_bottom_gas_intrusion is not None and cell == 0
        )
        is_lower_front_receiver = bool(
            projection.lower_material_front is not None
            and cell
            in {
                projection.lower_material_front.star.topology.front_cell,
                max(
                    projection.lower_material_front.star.topology.front_cell - 1,
                    0,
                ),
            }
        )
        reserve = (
            4.0 * parameters.area_tolerance_m2 * dz
            if (
                mg > 0.0
                and not is_joint_advance_donor
                and not is_first_entry_receiver
                and not is_lower_front_receiver
            )
            else 0.0
        )
        capacities.append(area * dz - reserve)
    liquid = _liquid_capacity_limited_faces(
        liquid,
        [value * dz for value in state.Al],
        cell_capacity_m3=capacities,
        dt=dt,
    )
    for face in range(
        projection.common_liquid_face_first,
        projection.common_liquid_face_last + 1,
    ):
        liquid[face] = projection.common_liquid_flux_m3_s
    _verify_saturated_liquid_volume_flux_projection(
        liquid,
        common_liquid_face_first=projection.common_liquid_face_first,
        common_liquid_face_last=projection.common_liquid_face_last,
        common_flux_m3_s=projection.common_liquid_flux_m3_s,
    )
    if projection.retreat is not None:
        face = projection.retreat.interface_face
        lower_face = projection.retreat.interface_cell
        requested = projection.retreat.requested_gas_mass_flux_kg_s
        gas_scale = math.fsum((abs(gas[face]), abs(requested)))
        gas_roundoff = 64.0 * math.ulp(1.0) * gas_scale
        if abs(gas[face] - requested) > gas_roundoff:
            raise TeeTransactionRejected(
                "top-gas donor cannot supply the conservative retreat flux"
            )
        if liquid[face] != 0.0:
            raise TeeTransactionRejected(
                "the upper material interface cannot carry liquid from gas"
            )
        if lower_face > 0 and gas[lower_face] != 0.0:
            raise TeeTransactionRejected(
                "the lower face of an upper cut cell cannot carry gas"
            )
    if projection.advance is not None:
        face = projection.advance.interface_face
        lower_face = projection.advance.interface_cell
        requested = projection.advance.requested_gas_mass_flux_kg_s
        gas_scale = math.fsum((abs(gas[face]), abs(requested)))
        gas_roundoff = 64.0 * math.ulp(1.0) * gas_scale
        if abs(gas[face] - requested) > gas_roundoff:
            raise TeeTransactionRejected(
                "interface-cell gas donor cannot supply the conservative advance flux"
            )
        if liquid[face] != 0.0:
            raise TeeTransactionRejected(
                "the upper material interface cannot draw liquid from gas"
            )
        if lower_face > 0 and gas[lower_face] != 0.0:
            raise TeeTransactionRejected(
                "the lower face of an advancing upper cut cell cannot carry gas"
            )
    if projection.first_bottom_gas_intrusion is not None:
        first = projection.first_bottom_gas_intrusion
        if gas[0] != first.gas_mass_flow_kg_s:
            raise TeeTransactionRejected(
                "first-entry bottom gas mass flux was changed by a limiter"
            )
        if len(gas) > 1 and gas[1] != 0.0:
            raise TeeTransactionRejected(
                "new lower gas front cannot cross its upper liquid material face"
            )
        ale_residual = liquid[1] - liquid[0] - first.gas_volume_flow_m3_s
        ale_scale = math.fsum(
            (
                abs(liquid[1]),
                abs(liquid[0]),
                abs(first.gas_volume_flow_m3_s),
            )
        )
        if abs(ale_residual) > 64.0 * math.ulp(1.0) * ale_scale:
            raise TeeTransactionRejected(
                "first-entry ALE liquid/gas volume identity was changed by a limiter"
            )
    if projection.lower_material_front is not None:
        lower = projection.lower_material_front
        topology = lower.star.topology
        q = lower.star.interface_volume_flow_m3_s
        if topology.is_grid_aligned and q > 0.0:
            face = topology.front_cell
            requested = lower.star.gas_density_kg_m3 * q
            scale = math.fsum((abs(gas[face]), abs(requested)))
            if abs(gas[face] - requested) > 64.0 * math.ulp(1.0) * scale:
                raise TeeTransactionRejected(
                    "lower-front gas donor cannot supply a grid-aligned advance"
                )
        material_gas_face = topology.front_cell + 1
        if gas[material_gas_face] != 0.0:
            raise TeeTransactionRejected(
                "gas cannot cross the lower gas/liquid material surface"
            )
        _verify_saturated_liquid_volume_flux_projection(
            liquid,
            common_liquid_face_first=lower.projected_liquid_face_first,
            common_liquid_face_last=lower.projected_liquid_face_last,
            common_flux_m3_s=q,
        )

    tolerance = parameters.transaction_tolerance
    if abs(liquid[0] - liquid_requested) > tolerance * max(
        1.0, abs(liquid_requested)
    ):
        raise TeeTransactionRejected(
            "bottom liquid exchange cannot be accepted atomically"
        )
    if abs(gas[0] - gas_requested) > tolerance * max(1.0, abs(gas_requested)):
        raise TeeTransactionRejected(
            "bottom gas exchange cannot be accepted atomically"
        )
    return (
        liquid,
        gas,
        bottom_gas_momentum,
        bottom_liquid_momentum,
        top_gas_velocity,
        projection.retreat,
        projection.advance,
        projection.lower_material_front,
    )


def _momentum_fluxes(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    liquid_flux: Sequence[float],
    gas_flux: Sequence[float],
    bottom_gas_momentum_flux_N: float,
    bottom_liquid_momentum_flux_N: float | None,
    top_gas_velocity_m_s: float,
    upper_surface_advance: _UpperFreeSurfaceAdvanceFlux | None,
) -> tuple[list[float], list[float]]:
    n = parameters.cell_count
    area = parameters.full_area_m2
    rho_l = parameters.liquid_density_kg_m3
    ul, ug = _phase_velocities(state, parameters)
    liquid_momentum = [0.0] * (n + 1)
    gas_momentum = [0.0] * (n + 1)

    if bottom_liquid_momentum_flux_N is None:
        # No transaction means a closed physical T face.  Retain the original
        # native zero/default construction bit-for-bit.
        if liquid_flux[0] >= 0.0:
            bottom_liquid_speed = liquid_flux[0] / area
        else:
            bottom_liquid_speed = ul[0]
        liquid_momentum[0] = rho_l * liquid_flux[0] * bottom_liquid_speed
    else:
        supplied_liquid_momentum = float(bottom_liquid_momentum_flux_N)
        if supplied_liquid_momentum < 0.0:
            raise TeeTransactionRejected(
                "bottom liquid convective momentum flux cannot be negative"
            )
        if liquid_flux[0] == 0.0 and supplied_liquid_momentum != 0.0:
            raise TeeTransactionRejected(
                "zero liquid volume flux cannot carry convective momentum"
            )
        if liquid_flux[0] != 0.0 and supplied_liquid_momentum == 0.0:
            raise TeeTransactionRejected(
                "nonzero liquid T flow requires its Riemann-solved momentum flux"
            )
        liquid_momentum[0] = supplied_liquid_momentum
    # The transaction carries only the convective part ``mdot*u``.  Pressure
    # is represented below by the common-pressure face field.  Passing
    # ``mdot*u + A*p`` here would double-count the T-face pressure and create
    # motion in an equal-pressure, zero-flow equilibrium.
    if bottom_gas_momentum_flux_N < -parameters.transaction_tolerance:
        raise TeeTransactionRejected(
            "bottom gas convective momentum flux cannot be negative"
        )
    if (
        abs(gas_flux[0]) <= parameters.transaction_tolerance
        and abs(bottom_gas_momentum_flux_N)
        > parameters.transaction_tolerance
    ):
        raise TeeTransactionRejected(
            "zero gas mass flux cannot carry convective momentum"
        )
    gas_momentum[0] = bottom_gas_momentum_flux_N

    for face in range(1, n):
        donor_l = face - 1 if liquid_flux[face] >= 0.0 else face
        donor_g = face - 1 if gas_flux[face] >= 0.0 else face
        liquid_momentum[face] = (
            rho_l * liquid_flux[face] * ul[donor_l]
        )
        gas_momentum[face] = gas_flux[face] * ug[donor_g]
    liquid_momentum[-1] = rho_l * liquid_flux[-1] * ul[-1]
    gas_momentum[-1] = gas_flux[-1] * top_gas_velocity_m_s
    if upper_surface_advance is not None:
        face = upper_surface_advance.interface_face
        gas_momentum[face] = (
            upper_surface_advance.requested_gas_mass_flux_kg_s
            * upper_surface_advance.donor_gas_velocity_m_s
        )
    return liquid_momentum, gas_momentum


def advance_vertical_twofluid(
    state: VerticalTwoFluidState,
    parameters: VerticalTwoFluidParameters,
    *,
    dt: float,
    tee_transaction: TeeTransactionLike | None = None,
    top_boundary: AtmosphericTopBoundary | None = None,
    pressure_faces_Pa: Sequence[float] | None = None,
) -> VerticalTwoFluidStepResult:
    """Advance both phases through one conservative finite-volume step."""

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    validate_state(state, parameters)
    state = canonicalize_upper_free_surface_roundoff(state, parameters)
    top = top_boundary or AtmosphericTopBoundary(
        pressure_abs_Pa=parameters.atmospheric_pressure_Pa
    )
    n = parameters.cell_count
    dz = parameters.cell_length_m
    area = parameters.full_area_m2
    rho_l = parameters.liquid_density_kg_m3

    first_bottom_gas_intrusion = _prepare_first_bottom_gas_intrusion(
        state,
        parameters,
        top,
        tee_transaction,
        dt=dt,
    )
    bottom_gas_storage = _prepare_bottom_gas_storage(
        state,
        parameters,
        tee_transaction,
    )

    (
        liquid_flux,
        gas_flux,
        bottom_gas_momentum,
        bottom_liquid_momentum,
        top_gas_velocity,
        upper_surface_retreat,
        upper_surface_advance,
        lower_material_front,
    ) = _upwind_face_fluxes(
        state,
        parameters,
        top,
        tee_transaction,
        first_bottom_gas_intrusion=first_bottom_gas_intrusion,
        dt=dt,
    )
    liquid_momentum_flux, gas_momentum_flux = _momentum_fluxes(
        state,
        parameters,
        liquid_flux,
        gas_flux,
        bottom_gas_momentum,
        bottom_liquid_momentum,
        top_gas_velocity,
        upper_surface_advance,
    )

    Al_new: list[float] = []
    Mg_new: list[float] = []
    liquid_momentum_new: list[float] = []
    gas_momentum_new: list[float] = []
    for cell in range(n):
        Al_new.append(
            state.Al[cell]
            - dt / dz * (liquid_flux[cell + 1] - liquid_flux[cell])
        )
        Mg_new.append(
            state.Mg[cell]
            - dt * (gas_flux[cell + 1] - gas_flux[cell])
        )
        liquid_momentum_new.append(
            rho_l * state.Ql[cell] * dz
            - dt
            * (
                liquid_momentum_flux[cell + 1]
                - liquid_momentum_flux[cell]
            )
        )
        gas_momentum_new.append(
            state.Jg[cell]
            - dt
            * (gas_momentum_flux[cell + 1] - gas_momentum_flux[cell])
        )

    lower_front_cell, lower_front_orientation = _resolve_lower_front_after_transport(
        state,
        parameters,
        lower_material_front,
        Al_new,
        liquid_momentum_new,
        Mg_new,
        gas_momentum_new,
    )
    _resolve_upper_front_after_transport(
        parameters,
        upper_surface_retreat,
        upper_surface_advance,
        Al_new,
        liquid_momentum_new,
        Mg_new,
        gas_momentum_new,
    )
    if first_bottom_gas_intrusion is not None:
        if first_bottom_gas_intrusion.lands_on_geometric_event:
            next_cell = 1
            if next_cell >= n or Al_new[next_cell] <= 0.0:
                raise TeeTransactionRejected(
                    "first bottom gas entry collided with the upper gas component"
                )
            liquid_momentum_new[next_cell] += liquid_momentum_new[0]
            liquid_momentum_new[0] = 0.0
            Al_new[0] = 0.0
            lower_front_cell = next_cell
        else:
            lower_front_cell = 0
        lower_front_orientation = "gas_below_liquid_above"
    transported = VerticalTwoFluidState.from_iterables(
        Al=Al_new,
        Ql=[value / (rho_l * dz) for value in liquid_momentum_new],
        Mg=Mg_new,
        Jg=gas_momentum_new,
        time_s=state.time_s,
        cumulative_top_liquid_outflow_m3=state.cumulative_top_liquid_outflow_m3,
        cumulative_top_gas_outflow_kg=state.cumulative_top_gas_outflow_kg,
        cumulative_top_gas_inflow_kg=state.cumulative_top_gas_inflow_kg,
        cumulative_bottom_liquid_exchange_m3=state.cumulative_bottom_liquid_exchange_m3,
        cumulative_bottom_gas_exchange_kg=state.cumulative_bottom_gas_exchange_kg,
        lower_material_front_cell=lower_front_cell,
        lower_material_front_orientation=lower_front_orientation,
    )
    validate_state(transported, parameters)

    if pressure_faces_Pa is None:
        pressure_faces = isothermal_common_pressure_faces(
            transported,
            parameters,
            top,
            (
                None
                if (
                    first_bottom_gas_intrusion is not None
                    or bottom_gas_storage is not None
                )
                else tee_transaction
            ),
        )
        if first_bottom_gas_intrusion is not None:
            pressure_faces = (
                first_bottom_gas_intrusion.common_pressure_abs_Pa,
                *pressure_faces[1:],
            )
        elif bottom_gas_storage is not None:
            pressure_faces = (
                _finite_pocket_applied_bottom_pressure(
                    bottom_gas_storage,
                    parameters,
                    pressure_faces[0],
                ),
                *pressure_faces[1:],
            )
    else:
        pressure_faces = tuple(float(value) for value in pressure_faces_Pa)
        if len(pressure_faces) != n + 1 or not _finite(*pressure_faces):
            raise ValueError("pressure_faces_Pa must contain n+1 finite values")
        pressure_tolerance = 1.0e-9 * max(top.pressure_abs_Pa, 1.0)
        if abs(pressure_faces[-1] - top.pressure_abs_Pa) > pressure_tolerance:
            raise ValueError("top pressure face must equal the atmospheric boundary")
        if tee_transaction is not None:
            dynamic_base = isothermal_common_pressure_faces(
                transported, parameters, top
            )
            expected_bottom = (
                first_bottom_gas_intrusion.common_pressure_abs_Pa
                if first_bottom_gas_intrusion is not None
                else (
                    _finite_pocket_applied_bottom_pressure(
                        bottom_gas_storage,
                        parameters,
                        dynamic_base[0],
                    )
                    if bottom_gas_storage is not None
                    else _transaction_bottom_common_pressure(
                    transported,
                    parameters,
                    top,
                    tee_transaction,
                    dynamic_base,
                    )
                )
            )
            bottom_tolerance = 1.0e-8 * max(
                abs(expected_bottom), abs(pressure_faces[0]), 1.0
            )
            if abs(pressure_faces[0] - expected_bottom) > bottom_tolerance:
                raise TeeTransactionRejected(
                    "caller-supplied bottom face does not consume the T pressure"
                )

    pressure_faces = _project_saturated_liquid_pressure_faces(
        transported,
        parameters,
        pressure_faces,
        liquid_momentum_new,
        dt=dt,
    )
    aligned_lower_interface_pressure_abs_Pa: float | None = None
    if (
        transported.lower_material_front_cell is not None
        and transported.Al[transported.lower_material_front_cell] == area
        and (
            lower_material_front is not None
            or first_bottom_gas_intrusion is not None
        )
    ):
        aligned_face = transported.lower_material_front_cell
        aligned_lower_interface_pressure_abs_Pa = (
            lower_material_front.star.interface_pressure_abs_Pa
            if lower_material_front is not None
            else first_bottom_gas_intrusion.common_pressure_abs_Pa
        )
        pressure_faces = (
            *pressure_faces[:aligned_face],
            aligned_lower_interface_pressure_abs_Pa,
            *pressure_faces[aligned_face + 1 :],
        )

    lower_cut_cell: int | None = None
    lower_cut_pressure_abs_Pa: float | None = None
    if (
        transported.lower_material_front_cell is not None
        and 0.0
        < transported.Al[transported.lower_material_front_cell]
        < area
    ):
        if lower_material_front is not None:
            lower_cut_cell = transported.lower_material_front_cell
            lower_cut_pressure_abs_Pa = (
                lower_material_front.star.interface_pressure_abs_Pa
            )
        elif first_bottom_gas_intrusion is not None:
            # The first-entry transport has already created a genuine
            # gas-below/liquid-above cut.  Its zero-pocket-limit common
            # characteristic pressure is the material-interface pressure for
            # this same source step; using the old pre-entry marker would omit
            # the equal-and-opposite interface force.
            lower_cut_cell = transported.lower_material_front_cell
            lower_cut_pressure_abs_Pa = (
                first_bottom_gas_intrusion.common_pressure_abs_Pa
            )

    liquid_pressure_forces: list[float] = []
    liquid_gravity_forces: list[float] = []
    gas_pressure_forces: list[float] = []
    gas_gravity_forces: list[float] = []
    liquid_wall_forces: list[float] = []
    gas_wall_forces: list[float] = []
    applied_lower_interface_liquid_pressure_force_N = 0.0
    applied_lower_interface_gas_pressure_force_N = 0.0
    aligned_lower_interface_face = (
        transported.lower_material_front_cell
        if aligned_lower_interface_pressure_abs_Pa is not None
        else None
    )
    for cell in range(n):
        al = transported.Al[cell]
        ag = max(area - al, 0.0)
        mg = transported.Mg[cell]
        dp = pressure_faces[cell + 1] - pressure_faces[cell]
        ml = rho_l * al * dz
        liquid_is_resolved = al > parameters.area_tolerance_m2
        gas_is_resolved = ag > 0.0 and mg > 0.0
        is_lower_cut = bool(
            lower_cut_pressure_abs_Pa is not None
            and lower_cut_cell == cell
            and 0.0 < al < area
            and ag > 0.0
        )

        if liquid_is_resolved:
            liquid_pressure_force = (
                area
                * (
                    lower_cut_pressure_abs_Pa
                    - pressure_faces[cell + 1]
                )
                if is_lower_cut
                else -al * dp
            )
            liquid_gravity_force = -ml * parameters.gravity_m_s2
            liquid_force = liquid_pressure_force + liquid_gravity_force
            # Absolute pressures are O(1e5 Pa), whereas a hydrostatic face
            # difference may be many orders smaller.  Cancel only the
            # roundoff-sized residual of the discrete balance.
            liquid_balance_scale = (
                abs(al * dp) + abs(ml * parameters.gravity_m_s2)
            )
            if abs(liquid_force) <= 1.0e-7 * max(
                liquid_balance_scale, 1.0e-30
            ):
                liquid_force = 0.0
                liquid_pressure_force = -liquid_gravity_force
            ul = liquid_momentum_new[cell] / ml
            liquid_wall = (
                -0.5
                * parameters.liquid_wall_friction
                * ml
                / parameters.diameter_m
                * ul
                * abs(ul)
            )
            liquid_momentum_new[cell] += dt * (
                liquid_force + liquid_wall
            )
            if is_lower_cut:
                applied_lower_interface_liquid_pressure_force_N = (
                    area * float(lower_cut_pressure_abs_Pa)
                )
            elif aligned_lower_interface_face == cell:
                applied_lower_interface_liquid_pressure_force_N = (
                    area * float(aligned_lower_interface_pressure_abs_Pa)
                )
        else:
            # A phase below the geometry resolution has no velocity degree of
            # freedom.  Keep its conserved momentum identically zero instead
            # of letting pressure on a roundoff-sized area create momentum
            # without inventory.
            liquid_momentum_new[cell] = 0.0
            liquid_force = 0.0
            liquid_pressure_force = 0.0
            liquid_gravity_force = 0.0
            liquid_wall = 0.0

        if gas_is_resolved:
            gas_pressure_force = (
                area
                * (
                    pressure_faces[cell]
                    - lower_cut_pressure_abs_Pa
                )
                if is_lower_cut
                else -ag * dp
            )
            gas_gravity_force = -mg * parameters.gravity_m_s2
            gas_force = gas_pressure_force + gas_gravity_force
            gas_balance_scale = (
                abs(ag * dp) + abs(mg * parameters.gravity_m_s2)
            )
            if abs(gas_force) <= 1.0e-7 * max(
                gas_balance_scale, 1.0e-30
            ):
                gas_force = 0.0
                gas_pressure_force = -gas_gravity_force
            ug = gas_momentum_new[cell] / mg
            gas_wall = (
                -0.5
                * parameters.gas_wall_friction
                * mg
                / parameters.diameter_m
                * ug
                * abs(ug)
            )
            gas_momentum_new[cell] += dt * (gas_force + gas_wall)
            if is_lower_cut:
                applied_lower_interface_gas_pressure_force_N = (
                    -area * float(lower_cut_pressure_abs_Pa)
                )
            elif aligned_lower_interface_face == cell + 1:
                applied_lower_interface_gas_pressure_force_N = (
                    -area * float(aligned_lower_interface_pressure_abs_Pa)
                )
        else:
            # validate_state has already ruled out a resolved void with no gas
            # mass (and gas mass without a void).  This branch is therefore a
            # genuinely absent/unresolved gas phase, not a small but physical
            # gas inventory whose pressure force may be discarded.
            gas_momentum_new[cell] = 0.0
            gas_force = 0.0
            gas_pressure_force = 0.0
            gas_gravity_force = 0.0
            gas_wall = 0.0
        liquid_pressure_forces.append(liquid_pressure_force)
        liquid_gravity_forces.append(liquid_gravity_force)
        gas_pressure_forces.append(gas_pressure_force)
        gas_gravity_forces.append(gas_gravity_force)
        liquid_wall_forces.append(liquid_wall)
        gas_wall_forces.append(gas_wall)

    sourced = replace(
        transported,
        Ql=tuple(value / (rho_l * dz) for value in liquid_momentum_new),
        Jg=tuple(gas_momentum_new),
    )
    sourced, drag = apply_equal_and_opposite_interphase_drag(
        sourced,
        parameters,
        dt=dt,
    )

    top_liquid_outflow = liquid_flux[-1] * dt
    top_gas_outflow = max(gas_flux[-1], 0.0) * dt
    top_gas_inflow = max(-gas_flux[-1], 0.0) * dt
    bottom_liquid_exchange = liquid_flux[0] * dt
    if lower_material_front is None and first_bottom_gas_intrusion is None:
        if (
            upper_surface_retreat is not None
            and upper_surface_retreat.lands_on_geometric_event
        ):
            bottom_liquid_exchange = -upper_surface_retreat.swept_gas_volume_m3
        elif (
            upper_surface_advance is not None
            and upper_surface_advance.lands_on_geometric_event
        ):
            bottom_liquid_exchange = upper_surface_advance.swept_liquid_volume_m3
    bottom_gas_exchange = gas_flux[0] * dt
    final = replace(
        sourced,
        time_s=state.time_s + dt,
        cumulative_top_liquid_outflow_m3=(
            state.cumulative_top_liquid_outflow_m3 + top_liquid_outflow
        ),
        cumulative_top_gas_outflow_kg=(
            state.cumulative_top_gas_outflow_kg + top_gas_outflow
        ),
        cumulative_top_gas_inflow_kg=(
            state.cumulative_top_gas_inflow_kg + top_gas_inflow
        ),
        cumulative_bottom_liquid_exchange_m3=(
            state.cumulative_bottom_liquid_exchange_m3
            + bottom_liquid_exchange
        ),
        cumulative_bottom_gas_exchange_kg=(
            state.cumulative_bottom_gas_exchange_kg + bottom_gas_exchange
        ),
    )
    validate_state(final, parameters)

    initial_liquid_volume = math.fsum(al * dz for al in state.Al)
    final_liquid_volume = math.fsum(al * dz for al in final.Al)
    initial_gas_mass = math.fsum(state.Mg)
    final_gas_mass = math.fsum(final.Mg)
    initial_liquid_momentum = rho_l * math.fsum(q * dz for q in state.Ql)
    final_liquid_momentum = rho_l * math.fsum(q * dz for q in final.Ql)
    initial_gas_momentum = math.fsum(state.Jg)
    final_gas_momentum = math.fsum(final.Jg)
    initial_momentum = initial_liquid_momentum + initial_gas_momentum
    final_momentum = final_liquid_momentum + final_gas_momentum
    liquid_boundary_impulse = dt * (
        liquid_momentum_flux[0] - liquid_momentum_flux[-1]
    )
    gas_boundary_impulse = dt * (
        gas_momentum_flux[0] - gas_momentum_flux[-1]
    )
    boundary_momentum_impulse = (
        liquid_boundary_impulse + gas_boundary_impulse
    )
    liquid_pressure_impulse = dt * math.fsum(liquid_pressure_forces)
    liquid_gravity_impulse = dt * math.fsum(liquid_gravity_forces)
    gas_pressure_impulse = dt * math.fsum(gas_pressure_forces)
    gas_gravity_impulse = dt * math.fsum(gas_gravity_forces)
    # Preserve the source loop's cellwise cancellation for the total budget;
    # the separate phase sums below intentionally expose their own roundoff.
    pressure_gravity_impulse = dt * math.fsum(
        lp + lg + gp + gg
        for lp, lg, gp, gg in zip(
            liquid_pressure_forces,
            liquid_gravity_forces,
            gas_pressure_forces,
            gas_gravity_forces,
        )
    )
    liquid_wall_impulse = dt * math.fsum(liquid_wall_forces)
    gas_wall_impulse = dt * math.fsum(gas_wall_forces)
    wall_impulse = dt * math.fsum(
        liquid + gas
        for liquid, gas in zip(liquid_wall_forces, gas_wall_forces)
    )
    liquid_interphase_impulse = drag.total_liquid_impulse_kg_m_s
    gas_interphase_impulse = drag.total_gas_impulse_kg_m_s
    liquid_momentum_residual = (
        final_liquid_momentum
        - initial_liquid_momentum
        - liquid_boundary_impulse
        - liquid_pressure_impulse
        - liquid_gravity_impulse
        - liquid_wall_impulse
        - liquid_interphase_impulse
    )
    gas_momentum_residual = (
        final_gas_momentum
        - initial_gas_momentum
        - gas_boundary_impulse
        - gas_pressure_impulse
        - gas_gravity_impulse
        - gas_wall_impulse
        - gas_interphase_impulse
    )
    budget = VerticalTwoFluidBudget(
        initial_liquid_volume_m3=initial_liquid_volume,
        final_liquid_volume_m3=final_liquid_volume,
        bottom_liquid_exchange_m3=bottom_liquid_exchange,
        top_liquid_outflow_m3=top_liquid_outflow,
        liquid_volume_residual_m3=(
            final_liquid_volume
            - initial_liquid_volume
            - bottom_liquid_exchange
            + top_liquid_outflow
        ),
        initial_gas_mass_kg=initial_gas_mass,
        final_gas_mass_kg=final_gas_mass,
        bottom_gas_exchange_kg=bottom_gas_exchange,
        top_gas_outflow_kg=top_gas_outflow,
        top_gas_inflow_kg=top_gas_inflow,
        gas_mass_residual_kg=(
            final_gas_mass
            - initial_gas_mass
            - bottom_gas_exchange
            + top_gas_outflow
            - top_gas_inflow
        ),
        initial_total_momentum_kg_m_s=initial_momentum,
        final_total_momentum_kg_m_s=final_momentum,
        boundary_momentum_impulse_kg_m_s=boundary_momentum_impulse,
        pressure_gravity_impulse_kg_m_s=pressure_gravity_impulse,
        wall_impulse_kg_m_s=wall_impulse,
        interphase_exchange_residual_kg_m_s=drag.exchange_residual_kg_m_s,
        total_momentum_residual_kg_m_s=(
            final_momentum
            - initial_momentum
            - boundary_momentum_impulse
            - pressure_gravity_impulse
            - wall_impulse
        ),
        initial_liquid_momentum_kg_m_s=initial_liquid_momentum,
        final_liquid_momentum_kg_m_s=final_liquid_momentum,
        liquid_boundary_momentum_impulse_kg_m_s=liquid_boundary_impulse,
        liquid_pressure_impulse_kg_m_s=liquid_pressure_impulse,
        liquid_gravity_impulse_kg_m_s=liquid_gravity_impulse,
        liquid_wall_impulse_kg_m_s=liquid_wall_impulse,
        liquid_interphase_impulse_kg_m_s=liquid_interphase_impulse,
        liquid_momentum_residual_kg_m_s=liquid_momentum_residual,
        initial_gas_momentum_kg_m_s=initial_gas_momentum,
        final_gas_momentum_kg_m_s=final_gas_momentum,
        gas_boundary_momentum_impulse_kg_m_s=gas_boundary_impulse,
        gas_pressure_impulse_kg_m_s=gas_pressure_impulse,
        gas_gravity_impulse_kg_m_s=gas_gravity_impulse,
        gas_wall_impulse_kg_m_s=gas_wall_impulse,
        gas_interphase_impulse_kg_m_s=gas_interphase_impulse,
        gas_momentum_residual_kg_m_s=gas_momentum_residual,
    )
    retreat_ledger: UpperFreeSurfaceRetreatLedger | None = None
    if upper_surface_retreat is not None:
        receiver = upper_surface_retreat.interface_cell
        face = upper_surface_retreat.interface_face
        swept_volume = upper_surface_retreat.swept_gas_volume_m3
        expected_mass = (
            upper_surface_retreat.donor_gas_density_kg_m3 * swept_volume
        )
        liquid_volume_loss = (
            state.Al[receiver] - transported.Al[receiver]
        ) * dz
        receiver_mass_gain = transported.Mg[receiver] - state.Mg[receiver]
        receiver_momentum_gain = (
            transported.Jg[receiver] - state.Jg[receiver]
        )
        expected_momentum_gain = -dt * (
            gas_momentum_flux[face] - gas_momentum_flux[face - 1]
        )
        interface_pressure = float(pressure_faces[face])
        pressure_impulse = interface_pressure * area * dt
        retreat_ledger = UpperFreeSurfaceRetreatLedger(
            interface_cell=receiver,
            interface_face=face,
            interface_velocity_m_s=(
                upper_surface_retreat.common_liquid_flux_m3_s / area
            ),
            swept_gas_volume_m3=swept_volume,
            donor_gas_density_kg_m3=(
                upper_surface_retreat.donor_gas_density_kg_m3
            ),
            donor_gas_velocity_m_s=(
                upper_surface_retreat.donor_gas_velocity_m_s
            ),
            gas_mass_flux_kg_s=(
                upper_surface_retreat.requested_gas_mass_flux_kg_s
            ),
            gas_momentum_flux_N=float(gas_momentum_flux[face]),
            liquid_volume_residual_m3=(liquid_volume_loss - swept_volume),
            receiver_gas_mass_residual_kg=(
                receiver_mass_gain - expected_mass
            ),
            receiver_gas_momentum_residual_kg_m_s=(
                receiver_momentum_gain - expected_momentum_gain
            ),
            interface_pressure_abs_Pa=interface_pressure,
            liquid_pressure_impulse_kg_m_s=-pressure_impulse,
            gas_pressure_impulse_kg_m_s=pressure_impulse,
            paired_pressure_impulse_residual_kg_m_s=0.0,
        )
    advance_ledger: UpperFreeSurfaceAdvanceLedger | None = None
    if upper_surface_advance is not None:
        donor = upper_surface_advance.interface_cell
        face = upper_surface_advance.interface_face
        swept_volume = upper_surface_advance.swept_liquid_volume_m3
        expected_mass = (
            upper_surface_advance.donor_gas_density_kg_m3 * swept_volume
        )
        liquid_volume_gain = (
            transported.Al[donor] - state.Al[donor]
        ) * dz
        donor_mass_loss = state.Mg[donor] - transported.Mg[donor]
        donor_momentum_loss = state.Jg[donor] - transported.Jg[donor]
        expected_momentum_loss = dt * (
            gas_momentum_flux[face] - gas_momentum_flux[donor]
        )
        interface_pressure = float(pressure_faces[face])
        pressure_impulse = interface_pressure * area * dt
        advance_ledger = UpperFreeSurfaceAdvanceLedger(
            interface_cell=donor,
            interface_face=face,
            interface_velocity_m_s=(
                upper_surface_advance.common_liquid_flux_m3_s / area
            ),
            swept_liquid_volume_m3=swept_volume,
            donor_gas_density_kg_m3=(
                upper_surface_advance.donor_gas_density_kg_m3
            ),
            donor_gas_velocity_m_s=(
                upper_surface_advance.donor_gas_velocity_m_s
            ),
            gas_mass_flux_kg_s=(
                upper_surface_advance.requested_gas_mass_flux_kg_s
            ),
            gas_momentum_flux_N=float(gas_momentum_flux[face]),
            liquid_volume_residual_m3=(
                liquid_volume_gain - swept_volume
            ),
            donor_gas_mass_residual_kg=(donor_mass_loss - expected_mass),
            donor_gas_momentum_residual_kg_m_s=(
                donor_momentum_loss - expected_momentum_loss
            ),
            interface_pressure_abs_Pa=interface_pressure,
            liquid_pressure_impulse_kg_m_s=-pressure_impulse,
            gas_pressure_impulse_kg_m_s=pressure_impulse,
            paired_pressure_impulse_residual_kg_m_s=0.0,
        )
    first_entry_ledger: FirstBottomGasIntrusionLedger | None = None
    if first_bottom_gas_intrusion is not None:
        first = first_bottom_gas_intrusion
        swept = first.swept_gas_volume_m3
        liquid_loss = (state.Al[0] - transported.Al[0]) * dz
        gas_area_after = area - transported.Al[0]
        mixture_volume_after = (
            transported.Al[0] + gas_area_after
        ) * dz
        gas_mass_gain = transported.Mg[0] - state.Mg[0]
        gas_momentum_gain = transported.Jg[0] - state.Jg[0]
        liquid_interface_impulse = (
            applied_lower_interface_liquid_pressure_force_N * dt
        )
        gas_interface_impulse = (
            applied_lower_interface_gas_pressure_force_N * dt
        )
        first_entry_ledger = FirstBottomGasIntrusionLedger(
            lower_front_cell=int(transported.lower_material_front_cell),
            lower_front_orientation="gas_below_liquid_above",
            common_pressure_abs_Pa=first.common_pressure_abs_Pa,
            gas_volume_flow_m3_s=first.gas_volume_flow_m3_s,
            gas_mass_flow_kg_s=first.gas_mass_flow_kg_s,
            gas_momentum_flow_N=first.gas_momentum_flow_N,
            liquid_bottom_volume_flow_m3_s=(
                first.liquid_bottom_volume_flow_m3_s
            ),
            liquid_plug_volume_flow_m3_s=(
                first.liquid_plug_volume_flow_m3_s
            ),
            liquid_momentum_flow_N=first.liquid_momentum_flow_N,
            donor_gas_density_kg_m3=first.donor_gas_density_kg_m3,
            donor_gas_velocity_m_s=first.donor_gas_velocity_m_s,
            riser_mouth_area_m2=area,
            gas_open_area_m2=first.gas_open_area_m2,
            liquid_open_area_m2=first.liquid_open_area_m2,
            blocked_riser_area_m2=first.blocked_riser_area_m2,
            swept_gas_volume_m3=swept,
            liquid_volume_residual_m3=liquid_loss - swept,
            mixture_volume_residual_m3=(
                mixture_volume_after - area * dz
            ),
            gas_mass_residual_kg=(
                gas_mass_gain - dt * first.gas_mass_flow_kg_s
            ),
            gas_momentum_residual_kg_m_s=(
                gas_momentum_gain - dt * first.gas_momentum_flow_N
            ),
            liquid_momentum_flux_residual_N=(
                liquid_momentum_flux[0] - first.liquid_momentum_flow_N
            ),
            liquid_pressure_impulse_kg_m_s=liquid_interface_impulse,
            gas_pressure_impulse_kg_m_s=gas_interface_impulse,
            paired_pressure_impulse_residual_kg_m_s=(
                liquid_interface_impulse + gas_interface_impulse
            ),
        )
    storage_ledger: BottomGasStorageLedger | None = None
    lower_front_ledger: LowerMaterialFrontLedger | None = None
    if bottom_gas_storage is not None and lower_material_front is not None:
        old_pocket_volume, old_pocket_mass, _ = _lower_gas_component_inventory(
            state, parameters
        )
        new_pocket_volume, new_pocket_mass, _ = _lower_gas_component_inventory(
            final, parameters
        )
        front_flow = lower_material_front.star.interface_volume_flow_m3_s
        pocket_volume_change = new_pocket_volume - old_pocket_volume
        storage_ledger = BottomGasStorageLedger(
            common_bottom_pressure_abs_Pa=(
                bottom_gas_storage.common_bottom_pressure_abs_Pa
            ),
            gas_volume_flow_m3_s=bottom_gas_storage.gas_volume_flow_m3_s,
            gas_mass_flow_kg_s=bottom_gas_storage.gas_mass_flow_kg_s,
            gas_momentum_flow_N=bottom_gas_storage.gas_momentum_flow_N,
            donor_gas_density_kg_m3=(
                bottom_gas_storage.donor_gas_density_kg_m3
            ),
            donor_gas_velocity_m_s=(
                bottom_gas_storage.donor_gas_velocity_m_s
            ),
            riser_mouth_area_m2=bottom_gas_storage.riser_mouth_area_m2,
            gas_open_area_m2=bottom_gas_storage.gas_open_area_m2,
            liquid_open_area_m2=bottom_gas_storage.liquid_open_area_m2,
            blocked_riser_area_m2=bottom_gas_storage.blocked_riser_area_m2,
            lower_front_volume_flow_m3_s=front_flow,
            gas_pocket_volume_change_m3=pocket_volume_change,
            gas_pocket_geometry_residual_m3=(
                pocket_volume_change - front_flow * dt
            ),
            compressive_storage_volume_m3=(
                (bottom_gas_storage.gas_volume_flow_m3_s - front_flow) * dt
            ),
            bottom_liquid_flow_m3_s=liquid_flux[0],
            bottom_liquid_momentum_flow_N=liquid_momentum_flux[0],
        )
        old_total_gas = math.fsum(state.Mg)
        new_total_gas = math.fsum(final.Mg)
        old_top_gas = old_total_gas - old_pocket_mass
        new_top_gas = new_total_gas - new_pocket_mass
        bottom_mass = gas_flux[0] * dt
        top_component_change = new_top_gas - old_top_gas
        gas_component_residual = (
            new_pocket_mass
            - old_pocket_mass
            - bottom_mass
            + top_component_change
            + top_gas_outflow
            - top_gas_inflow
        )
        old_topology = lower_material_front.star.topology
        new_marker = final.lower_material_front_cell
        new_grid_aligned = bool(
            new_marker is not None and final.Al[new_marker] == area
        )
        liquid_interface_impulse = (
            applied_lower_interface_liquid_pressure_force_N * dt
        )
        gas_interface_impulse = (
            applied_lower_interface_gas_pressure_force_N * dt
        )
        lower_front_ledger = LowerMaterialFrontLedger(
            old_front_cell=old_topology.front_cell,
            new_front_cell=new_marker,
            old_grid_aligned=old_topology.is_grid_aligned,
            new_grid_aligned=new_grid_aligned,
            interface_velocity_m_s=(
                lower_material_front.star.interface_velocity_m_s
            ),
            interface_volume_flow_m3_s=front_flow,
            interface_pressure_abs_Pa=(
                lower_material_front.star.interface_pressure_abs_Pa
            ),
            swept_volume_m3=lower_material_front.swept_volume_m3,
            liquid_plug_volume_residual_m3=budget.liquid_volume_residual_m3,
            gas_pocket_volume_change_m3=pocket_volume_change,
            gas_pocket_mass_change_kg=new_pocket_mass - old_pocket_mass,
            bottom_gas_mass_exchange_kg=bottom_mass,
            top_gas_component_mass_change_kg=top_component_change,
            gas_component_mass_residual_kg=gas_component_residual,
            liquid_pressure_impulse_kg_m_s=liquid_interface_impulse,
            gas_pressure_impulse_kg_m_s=gas_interface_impulse,
            paired_pressure_impulse_residual_kg_m_s=(
                liquid_interface_impulse + gas_interface_impulse
            ),
        )
    return VerticalTwoFluidStepResult(
        state=final,
        budget=budget,
        pressure_faces_Pa=tuple(pressure_faces),
        liquid_volume_flux_faces_m3_s=tuple(liquid_flux),
        gas_mass_flux_faces_kg_s=tuple(gas_flux),
        liquid_momentum_flux_faces_N=tuple(liquid_momentum_flux),
        gas_momentum_flux_faces_N=tuple(gas_momentum_flux),
        drag=drag,
        upper_free_surface_retreat=retreat_ledger,
        upper_free_surface_advance=advance_ledger,
        first_bottom_gas_intrusion=first_entry_ledger,
        bottom_gas_storage=storage_ledger,
        lower_material_front=lower_front_ledger,
    )


@dataclass(frozen=True)
class Campaign2VerticalTwoFluidKernel:
    """Small state-free facade suitable for the persistent network coupler."""

    parameters: VerticalTwoFluidParameters
    top_boundary: AtmosphericTopBoundary | None = None

    def advance(
        self,
        state: VerticalTwoFluidState,
        *,
        dt: float,
        tee_transaction: TeeTransactionLike | None = None,
        pressure_faces_Pa: Sequence[float] | None = None,
    ) -> VerticalTwoFluidStepResult:
        return advance_vertical_twofluid(
            state,
            self.parameters,
            dt=dt,
            tee_transaction=tee_transaction,
            top_boundary=self.top_boundary,
            pressure_faces_Pa=pressure_faces_Pa,
        )


__all__ = [
    "AtmosphericTopBoundary",
    "COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY",
    "Campaign2VerticalTwoFluidKernel",
    "BottomGasStorageLedger",
    "FirstBottomGasIntrusionLedger",
    "FirstBottomGasIntrusionTransactionLike",
    "InterphaseDragLedger",
    "LowerMaterialFrontLedger",
    "MINIMUM_FIRST_BOTTOM_GAS_INTRUSION_FIELDS",
    "MISSING_PHYSICAL_CLOSURES",
    "StateAdmissibilityError",
    "TeeTransactionLike",
    "TeeTransactionRejected",
    "UPPER_INTERFACE_GEOMETRY_ROUNDOFF_ULPS",
    "UpperFreeSurfaceAdvanceLedger",
    "UpperFreeSurfaceRetreatLedger",
    "VERTICAL_TWOFLUID_KERNEL_READY",
    "VerticalTwoFluidBudget",
    "VerticalTwoFluidError",
    "VerticalTwoFluidParameters",
    "VerticalTwoFluidState",
    "VerticalTwoFluidStepResult",
    "advance_vertical_twofluid",
    "apply_equal_and_opposite_interphase_drag",
    "atmospheric_empty_state",
    "canonicalize_upper_free_surface_roundoff",
    "hydrostatic_column_state",
    "isothermal_common_pressure_faces",
    "isothermal_gas_pressure_cells",
    "lower_material_front_geometric_timestep_limit",
    "lower_material_front_star_state",
    "mixture_hydrostatic_pressure_faces",
    "validate_state",
]
