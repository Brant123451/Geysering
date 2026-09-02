"""Conservative gas network for the Case-A horizontal pipe and riser.

The horizontal and vertical gas phases are advanced in the same acoustic
substeps.  A single side-branch Riemann flux is subtracted from the horizontal
T cell and added to the first riser cell.  Consequently the T is an internal
face of the gas network rather than a post-step mass-removal rule.

The gas equations are the isothermal block of the companion four-equation
model,

    d(A_g rho_g)/dt + d(A_g rho_g u_g)/dx = 0,
    d(A_g rho_g u_g)/dt
        + d[A_g(rho_g u_g**2 + p_g)]/dx = S_g.

MUSCL reconstruction, a Roe two-wave flux with an entropy fix, and SSP-RK2
are used.  At an unresolved gas-vacuum face only, an invariant-domain
Einfeldt flux replaces Roe because Roe is not positivity preserving there.
Horizontal and vertical momenta remain separate at the 90-degree
turn: mass turns through the branch, vertical momentum comes from the normal
Riemann flux, and the fitting supplies the corresponding vector reaction.
No front location, transfer rate, gas velocity, or result curve is prescribed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from casea_acceleration import njit


@dataclass(frozen=True)
class CoupledGasParameters:
    horizontal_diameter: float
    vertical_diameter: float
    rho_l: float = 998.0
    gravity: float = 9.81
    gas_constant: float = 287.05
    gas_temperature: float = 293.0
    atmospheric_pressure: float = 101325.0
    gas_viscosity: float = 1.81e-5
    # Stratified gas--liquid interfacial friction of the companion model:
    # lambda_i = lambda_g * (1 + C_h * alpha_l).  Taitel--Dukler-type
    # holdup enhancement increases shear only where both phases coexist and
    # therefore damps slip-driven short waves without diffusing phase area.
    horizontal_holdup_drag_enhancement: float = 0.0
    surface_tension: float = 0.072
    cfl: float = 0.32
    limiter_theta: float = 1.20
    entropy_fix_fraction: float = 0.10
    void_floor_fraction: float = 1.0e-4
    active_void_fraction: float = 5.0e-4
    # A massless elastic rarefaction away from the material front is not an
    # air path.  The receiver below is therefore restricted to one cell next
    # to mass-supported gas; within that local stencil, any void above the
    # positivity scale is a physically open crown volume and may receive gas.
    vertical_front_void_fraction: float = 0.05
    # Gas topology and gas-momentum resolution are different questions.  A
    # rarefied but non-empty cell must remain connected to its neighbours so
    # pressure can refill it; otherwise the 50%-atmosphere momentum threshold
    # turns a continuous pocket into alternating inactive cells.  Point
    # velocity is still excluded below ``resolved_density_fraction``.
    topology_density_fraction: float = 0.02
    # A confined Taylor bubble retains an annular counter-current liquid film.
    # The area-averaged junction therefore never assigns the complete tower
    # bore to the gas core, even when the horizontal crown fully covers it.
    vertical_gas_core_area_fraction: float = 0.80
    # A shock-fitted vertical front supplies material topology independently
    # of the interphase-momentum closure.  Keep receiver geometry separate
    # from ``vertical_confined_interface_kinematics``, which selects who owns
    # gas--liquid drag.
    vertical_fitted_front_receivers: bool = False
    # In the fitted Taylor-core regime, pressure work on the upper liquid slug
    # is already transmitted by the moving material interface.  Applying the
    # distributed form-drag exchange as well counts that coupling twice and
    # accelerates the annular film with the acoustic blow-down velocity.  Leave
    # the switch off for a generic dispersed two-fluid network; the Case-A
    # Taylor closure enables it explicitly.
    vertical_confined_interface_kinematics: bool = False
    # Diagnostic sensitivity only.  The physical default admits signed motion;
    # disabling it reproduces the historical one-way shock-fit front and is
    # used to isolate regressions in the retreat closure.
    allow_horizontal_front_retreat: bool = True
    # The isothermal Euler vacuum limit has an unbounded velocity tail.  In
    # this apparatus the physical gas releases from approximately 1.0--1.04
    # atm to an atmospheric opening; states below 50% atmospheric density are
    # therefore unresolved gas/vacuum fronts rather than the main gas body.
    # Their mass is still transported and conserved, but their point velocity
    # is excluded until the cell contains a resolved gas state.
    resolved_density_fraction: float = 0.50
    # Above this density the Riemann solver switches from Roe to the robust
    # Einfeldt flux.  It is not a phase-velocity cutoff: a compressed gas cell
    # remains a resolved momentum control volume.
    resolved_density_ceiling: float = 2.0

    @property
    def horizontal_area(self) -> float:
        return 0.25 * math.pi * self.horizontal_diameter**2

    @property
    def vertical_area(self) -> float:
        return 0.25 * math.pi * self.vertical_diameter**2

    @property
    def rho_atmospheric(self) -> float:
        return self.atmospheric_pressure / (
            self.gas_constant * self.gas_temperature
        )

    @property
    def sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.gas_temperature)

    @property
    def horizontal_capillary_void_fraction(self) -> float:
        """Minimum connected crown segment that exceeds capillary sealing.

        A sub-capillary crown gap is treated as a liquid bridge rather than a
        propagating gas passage.  The threshold follows from the water--air
        capillary length and circular-segment geometry; it is not a fitted
        percentage of the Case-A result.
        """

        density_jump = max(self.rho_l - self.rho_atmospheric, 1.0e-12)
        if self.gravity <= 0.0 or self.surface_tension <= 0.0:
            return self.active_void_fraction
        capillary_length = math.sqrt(
            self.surface_tension / (density_jump * self.gravity)
        )
        radius = 0.5 * self.horizontal_diameter
        cap_height = min(max(capillary_length, 0.0), radius)
        segment = (
            radius * radius * math.acos((radius - cap_height) / radius)
            - (radius - cap_height)
            * math.sqrt(max(2.0 * radius * cap_height - cap_height**2, 0.0))
        )
        return max(
            self.active_void_fraction,
            segment / self.horizontal_area,
        )

    @property
    def vertical_capillary_core_fraction(self) -> float:
        """Minimum open axial gas core that is not capillary sealed.

        In a vertical circular conduit, an axial gas passage with radius below
        the water--air capillary length behaves as a closed meniscus rather
        than a connected Taylor-bubble core.  The corresponding area ratio is
        ``(l_c/R)^2``.  This supplies a geometry/property-derived topology
        floor for an already swept gas path; it is not a fitted Taylor-core
        fraction and does not prescribe the resolved gas volume when the EOS
        requires a larger core.
        """

        density_jump = max(self.rho_l - self.rho_atmospheric, 1.0e-12)
        if self.gravity <= 0.0 or self.surface_tension <= 0.0:
            return self.active_void_fraction
        capillary_length = math.sqrt(
            self.surface_tension / (density_jump * self.gravity)
        )
        radius = 0.5 * self.vertical_diameter
        core_radius = min(max(capillary_length, 0.0), radius)
        return max(
            self.active_void_fraction,
            (core_radius / radius) ** 2,
        )


@dataclass(frozen=True)
class CoupledGasAdvance:
    horizontal_mass: np.ndarray
    horizontal_momentum: np.ndarray
    vertical_total_mass: np.ndarray
    vertical_momentum: np.ndarray
    vertical_tracer_mass: np.ndarray
    horizontal_liquid_momentum_increment: np.ndarray
    vertical_liquid_momentum_increment: np.ndarray
    escaped_tracer_mass: float
    atmospheric_mass_exchange: float
    junction_mass_transfer: float
    total_mass_error: float
    tracer_mass_error: float
    substeps: int
    maximum_velocity: float
    junction_mouth_area: float
    downstream_front_position: float | None
    downstream_topology_front_position: float | None
    downstream_front_velocity: float
    downstream_retired_cell_count: int


@dataclass(frozen=True)
class OpenIsothermalGasInventory:
    """A spatially lumped ideal-gas inventory with an externally set volume.

    The inventory is *open*: its mass is advanced by resolved boundary fluxes.
    Its volume is supplied by the moving liquid interfaces of the host solver.
    Pressure is therefore a state result, ``p = m R T / V``, rather than a
    prescribed pocket-pressure history.
    """

    mass: float
    volume: float
    gas_constant: float = 287.05
    temperature: float = 293.0

    def __post_init__(self) -> None:
        values = (self.mass, self.volume, self.gas_constant, self.temperature)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gas-inventory state must be finite")
        if self.mass < 0.0:
            raise ValueError("gas-inventory mass must be non-negative")
        if self.volume <= 0.0:
            raise ValueError("gas-inventory volume must be positive")
        if self.gas_constant <= 0.0 or self.temperature <= 0.0:
            raise ValueError("ideal-gas constants must be positive")

    @property
    def density(self) -> float:
        return self.mass / self.volume

    @property
    def pressure_absolute(self) -> float:
        return self.mass * self.gas_constant * self.temperature / self.volume

    @property
    def sound_speed(self) -> float:
        return math.sqrt(self.gas_constant * self.temperature)

    def with_state(
        self,
        *,
        mass: float | None = None,
        volume: float | None = None,
    ) -> "OpenIsothermalGasInventory":
        """Return the same thermodynamic inventory at a new mass/volume."""

        return OpenIsothermalGasInventory(
            mass=self.mass if mass is None else float(mass),
            volume=self.volume if volume is None else float(volume),
            gas_constant=self.gas_constant,
            temperature=self.temperature,
        )


@dataclass(frozen=True)
class LumpedSideTGasAdvance:
    """One conservative gas-mass exchange across a side-T mouth.

    ``mass_transfer`` is positive from the horizontal lumped pocket into the
    vertical receiver.  ``normal_momentum_flux`` is returned for a host solver
    to include in its balanced junction momentum residual; this isolated mass
    component deliberately does not invent a 90-degree momentum remap.
    """

    horizontal_inventory: OpenIsothermalGasInventory
    vertical_mass: np.ndarray
    mass_flux: float
    normal_momentum_flux: float
    raw_mass_transfer: float
    mass_transfer: float
    conservation_error: float
    donor_limited: bool


@dataclass(frozen=True)
class LumpedPocketVerticalAdvance:
    """Result of a lumped horizontal pocket coupled to the riser FV graph."""

    horizontal_inventory: OpenIsothermalGasInventory
    vertical_total_mass: np.ndarray
    vertical_momentum: np.ndarray
    vertical_tracer_mass: np.ndarray
    vertical_liquid_momentum_increment: np.ndarray
    escaped_tracer_mass: float
    atmospheric_mass_exchange: float
    junction_mass_transfer: float
    total_mass_error: float
    tracer_mass_error: float
    substeps: int
    maximum_velocity: float
    junction_mouth_area: float


def _gamma_from_holdup(alpha_l: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha_l, dtype=float), 1.0e-10, 1.0 - 1.0e-10)
    lo = np.full_like(alpha, 1.0e-12)
    hi = np.full_like(alpha, 2.0 * math.pi - 1.0e-12)
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        fraction = (mid - np.sin(mid)) / (2.0 * math.pi)
        lo = np.where(fraction < alpha, mid, lo)
        hi = np.where(fraction < alpha, hi, mid)
    return 0.5 * (lo + hi)


def _horizontal_geometry(
    liquid_area: np.ndarray,
    params: CoupledGasParameters,
) -> tuple[np.ndarray, ...]:
    area = params.horizontal_area
    liquid = np.clip(np.asarray(liquid_area, dtype=float), 0.0, area)
    raw = np.maximum(area - liquid, 0.0)
    gas = np.maximum(raw, params.void_floor_fraction * area)
    gamma = _gamma_from_holdup(liquid / area)
    depth = 0.5 * params.horizontal_diameter * (
        1.0 - np.cos(0.5 * gamma)
    )
    perimeter_g = 0.5 * params.horizontal_diameter * (
        2.0 * math.pi - gamma
    )
    interface = np.maximum(
        params.horizontal_diameter * np.sin(0.5 * gamma), 1.0e-12
    )
    hydraulic = 4.0 * gas / np.maximum(perimeter_g + interface, 1.0e-12)
    return raw, gas, depth, perimeter_g, interface, hydraulic


def _vertical_geometry(
    liquid_area: np.ndarray,
    params: CoupledGasParameters,
) -> tuple[np.ndarray, ...]:
    area = params.vertical_area
    liquid = np.clip(np.asarray(liquid_area, dtype=float), 0.0, area)
    raw = np.maximum(area - liquid, 0.0)
    gas = np.maximum(raw, params.void_floor_fraction * area)
    alpha_g = np.clip(raw / area, 0.0, 1.0)
    # A smooth core/annular interface estimate.  It vanishes in either pure
    # phase and peaks when both phases occupy comparable cross-sectional area.
    interface = (
        math.pi
        * params.vertical_diameter
        * 2.0
        * np.sqrt(np.maximum(alpha_g * (1.0 - alpha_g), 0.0))
    )
    perimeter_g = math.pi * params.vertical_diameter * alpha_g
    hydraulic = 4.0 * gas / np.maximum(perimeter_g + interface, 1.0e-12)
    return raw, gas, perimeter_g, interface, hydraulic


def _mass_backed_gas_topology(
    raw_void_area: np.ndarray,
    gas_mass: np.ndarray,
    *,
    full_area: float,
    cell_width: float,
    rho_reference: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    topology_density_fraction: float,
    resolved_density_fraction: float,
) -> np.ndarray:
    """Return a connected material-gas mask without orphaning compressed mass.

    Ordinary gas cells require the geometric active-void threshold.  A narrow
    cell between such cells may remain connected below that threshold only
    when it contains a resolved-density gas state; this covers a closing front
    at roundoff scale without turning isolated positivity-floor residue into a
    domain-wide pneumatic slit.
    """

    raw = np.maximum(np.asarray(raw_void_area, dtype=float), 0.0)
    mass = np.maximum(np.asarray(gas_mass, dtype=float), 0.0)
    if raw.shape != mass.shape or raw.ndim != 1:
        raise ValueError("gas-topology arrays must be equal and one-dimensional")
    if full_area <= 0.0 or cell_width <= 0.0 or rho_reference <= 0.0:
        raise ValueError("positive gas-topology geometry and density required")
    effective = np.maximum(raw, void_floor_fraction * full_area)
    ordinary = (
        raw >= active_void_fraction * full_area
    ) & (
        mass
        > topology_density_fraction
        * rho_reference
        * effective
        * cell_width
    )
    compressed_bridge = (
        raw > 1.5 * void_floor_fraction * full_area
    ) & (
        mass
        > resolved_density_fraction
        * rho_reference
        * effective
        * cell_width
    )
    supported = ordinary.copy()
    for _ in range(raw.size):
        adjacent = np.zeros_like(supported)
        if supported.size > 1:
            adjacent[1:] |= supported[:-1]
            adjacent[:-1] |= supported[1:]
        extended = supported | (compressed_bridge & adjacent)
        if np.array_equal(extended, supported):
            break
        supported = extended
    return supported


def _top_connected_active_component(active: np.ndarray) -> np.ndarray:
    """Return the active vertical gas component connected to the open lip.

    The acoustic graph connects adjacent active cells with a positive shared
    face.  Its atmospheric component is therefore the contiguous active run
    descending from the top cell.  Local gas area is not a second connectivity
    criterion: a capillary neck that remains an active Riemann face cannot be
    treated simultaneously as a sealed liquid-supported bubble.
    """

    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("vertical active-gas mask must be one-dimensional")
    connected = np.zeros_like(mask)
    for index in range(mask.size - 1, -1, -1):
        if not mask[index]:
            break
        connected[index] = True
    return connected


def junction_mouth_area(
    horizontal_void_fraction: float,
    params: CoupledGasParameters,
) -> float:
    """Geometric area of the tower bore exposed to the crown gas layer."""

    alpha = min(max(float(horizontal_void_fraction), 0.0), 1.0)
    if alpha <= 0.0:
        return 0.0
    crown_depth = params.horizontal_diameter * min(
        (3.0 * alpha / (2.0 * math.sqrt(2.0))) ** (2.0 / 3.0),
        1.0,
    )
    exposed = min(crown_depth / params.vertical_diameter, 1.0)
    return params.vertical_area * min(
        exposed, params.vertical_gas_core_area_fraction
    )


def _horizontal_downstream_material_fraction(
    cell_count: int,
    cell_width: float,
    junction_index: int,
    front_position: float | None,
) -> np.ndarray:
    """Return the axial gas occupancy of the fitted east-branch front.

    ``junction_index`` is the west donor cell adjacent to the face-aligned T.
    That cell and the upstream branch retain their ordinary gas topology.  In
    the east branch, a material front at ``x_f`` occupies the
    fraction ``clip((x_f-x_left)/dx, 0, 1)`` of each control volume.  This
    removes the former cell-centre switch: a front can now advance and retreat
    continuously without leaving a full-size gas cell pinned behind it.
    """

    if cell_count <= 0 or cell_width <= 0.0:
        raise ValueError("material-front grid must have positive size and width")
    if not 0 <= int(junction_index) < int(cell_count):
        raise ValueError("junction index lies outside the material-front grid")
    fraction = np.ones(int(cell_count), dtype=float)
    if front_position is None:
        return fraction
    indices = np.arange(int(cell_count), dtype=int)
    east = indices > int(junction_index)
    left_faces = indices.astype(float) * float(cell_width)
    fraction[east] = np.clip(
        (float(front_position) - left_faces[east]) / float(cell_width),
        0.0,
        1.0,
    )
    return fraction


def _equilibrate_horizontal_front_receivers(
    mass: np.ndarray,
    momentum: np.ndarray,
    raw_gas_area: np.ndarray,
    mass_supported: np.ndarray,
    front_receiver: np.ndarray,
    *,
    cell_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialise a newly swept gas cut-cell without a gas--vacuum jet.

    The receiver volume is created by the liquid moving-interface solve.  It
    is not a pre-existing gas control volume and therefore must not enter the
    isothermal Euler solver as near vacuum.  Over the sub-cell acoustic time,
    the neighbouring material gas and the new cut volume share one density and
    velocity.  This exact two-volume remap conserves their combined mass and
    axial momentum; the subsequent Roe/MUSCL solve advances ordinary resolved
    gas states.
    """

    m = np.asarray(mass, dtype=float).copy()
    j = np.asarray(momentum, dtype=float).copy()
    area = np.maximum(np.asarray(raw_gas_area, dtype=float), 0.0)
    supported = np.asarray(mass_supported, dtype=bool)
    receiver = np.asarray(front_receiver, dtype=bool)
    if not (m.shape == j.shape == area.shape == supported.shape == receiver.shape):
        raise ValueError("horizontal front-remap arrays must have equal shape")
    if m.ndim != 1 or cell_width <= 0.0:
        raise ValueError("horizontal front remap requires a 1-D positive-width grid")

    for index in np.flatnonzero(receiver & ~supported):
        candidates: list[int] = []
        if index > 0 and supported[index - 1]:
            candidates.append(index - 1)
        if index + 1 < m.size and supported[index + 1]:
            candidates.append(index + 1)
        if not candidates or area[index] <= 0.0:
            continue
        # The receiver is a full finite-volume gas state, rather than a true
        # fractional axial cut cell.  Equalise it over the attached acoustic
        # component before the Riemann solve; restricting this operation to a
        # single small donor launched a gas--vacuum jet and drove the Case-A
        # front to the closed end in less than one output interval.  This
        # opening projection is the established stable whole-cell treatment.
        # Retiring cells use the separate local merge below, so a centimetre-
        # scale reversal never reprojects the already established pocket.
        donor_set: set[int] = set()
        for candidate in candidates:
            first = candidate
            while first > 0 and supported[first - 1]:
                first -= 1
            last = candidate + 1
            while last < m.size and supported[last]:
                last += 1
            donor_set.update(range(first, last))
        donor_indices = np.asarray(sorted(donor_set), dtype=int)
        donor_volume = float(np.sum(area[donor_indices]) * cell_width)
        receiver_volume = area[index] * cell_width
        total_mass = float(np.sum(m[donor_indices]) + m[index])
        total_momentum = float(np.sum(j[donor_indices]) + j[index])
        total_volume = donor_volume + receiver_volume
        if total_mass <= 0.0 or total_volume <= 0.0:
            continue
        common_density = total_mass / total_volume
        common_velocity = total_momentum / total_mass
        m[donor_indices] = common_density * area[donor_indices] * cell_width
        m[index] = common_density * receiver_volume
        j[donor_indices] = m[donor_indices] * common_velocity
        j[index] = m[index] * common_velocity
    return m, j


def _collapse_horizontal_front_cells(
    mass: np.ndarray,
    momentum: np.ndarray,
    raw_gas_area: np.ndarray,
    mass_supported: np.ndarray,
    closing: np.ndarray,
    *,
    cell_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge completely closed front cells into their west neighbour.

    A partially occupied cut cell remains an active gas volume and is not
    passed to this helper.  Only when its axial occupancy reaches zero is the
    residual mass and momentum transferred to the immediately adjacent west
    cell and the retired storage cleared.  This operation is local, exactly
    conservative, and leaves the remote pocket bit-for-bit unchanged.
    """

    m = np.asarray(mass, dtype=float).copy()
    j = np.asarray(momentum, dtype=float).copy()
    area = np.maximum(np.asarray(raw_gas_area, dtype=float), 0.0)
    supported = np.asarray(mass_supported, dtype=bool)
    close = np.asarray(closing, dtype=bool) & supported
    if not (
        m.shape == j.shape == area.shape == supported.shape == close.shape
    ):
        raise ValueError("horizontal front-collapse arrays must have equal shape")
    if m.ndim != 1 or cell_width <= 0.0:
        raise ValueError(
            "horizontal front collapse requires a 1-D positive-width grid"
        )

    index = 0
    while index < m.size:
        if not close[index]:
            index += 1
            continue
        first_retired = index
        while index < m.size and close[index]:
            index += 1
        donor = first_retired - 1
        # The fitted east-branch front retreats westward, so its adjacent
        # retained cut cell must lie immediately west of the retiring run.
        if donor < 0 or not supported[donor] or close[donor]:
            continue
        retiring = np.arange(first_retired, index, dtype=int)
        m[donor] = float(m[donor] + np.sum(m[retiring]))
        j[donor] = float(j[donor] + np.sum(j[retiring]))
        m[retiring] = 0.0
        j[retiring] = 0.0
    return m, j


def _equilibrate_vertical_front_receivers(
    horizontal_mass: np.ndarray,
    horizontal_momentum: np.ndarray,
    vertical_mass: np.ndarray,
    vertical_momentum: np.ndarray,
    vertical_tracer: np.ndarray,
    horizontal_gas_area: np.ndarray,
    vertical_gas_area: np.ndarray,
    horizontal_supported: np.ndarray,
    vertical_supported: np.ndarray,
    vertical_receiver: np.ndarray,
    *,
    junction_index: int,
    horizontal_width: float,
    vertical_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill a newly opened riser cut cell from the attached tunnel gas.

    A material sweep creates finite gas volume before a cell-centred gas mass
    equation has transported mass into it.  Sending that new volume into an
    isothermal Riemann solve with only positivity-floor mass creates a nearly
    vacuum state and a spurious sonic jet.  Over the much shorter acoustic
    crossing time of the T fitting, the complete horizontal gas component and
    all newly opened cut volumes share one density.  This component projection
    conserves total gas mass and tunnel-origin tracer exactly; no pressure,
    velocity, or transfer fraction is prescribed.

    Axial gas momentum is not rotated through the 90-degree fitting by this
    remap.  A removed tunnel parcel carries its tangential momentum out of the
    horizontal control volume, while the vertical Riemann problem subsequently
    generates normal momentum from its resolved pressure difference.  The tee
    wall supplies the corresponding turning reaction.
    """

    hm = np.asarray(horizontal_mass, dtype=float).copy()
    hj = np.asarray(horizontal_momentum, dtype=float).copy()
    vm = np.asarray(vertical_mass, dtype=float).copy()
    vj = np.asarray(vertical_momentum, dtype=float).copy()
    vc = np.asarray(vertical_tracer, dtype=float).copy()
    hag = np.maximum(np.asarray(horizontal_gas_area, dtype=float), 0.0)
    vag = np.maximum(np.asarray(vertical_gas_area, dtype=float), 0.0)
    h_supported = np.asarray(horizontal_supported, dtype=bool)
    supported = np.asarray(vertical_supported, dtype=bool)
    receiver = np.asarray(vertical_receiver, dtype=bool)
    if not (hm.shape == hj.shape == hag.shape == h_supported.shape):
        raise ValueError("horizontal side-T gas arrays must have equal shape")
    if not (
        vm.shape == vj.shape == vc.shape == vag.shape
        == supported.shape == receiver.shape
    ):
        raise ValueError("vertical side-T gas arrays must have equal shape")
    junction = int(junction_index)
    if not 0 <= junction < hm.size:
        raise ValueError("side-T gas donor lies outside the horizontal grid")
    if horizontal_width <= 0.0 or vertical_width <= 0.0:
        raise ValueError("positive side-T control-volume widths are required")

    if not h_supported[junction]:
        return hm, hj, vm, vj, vc

    # Use the complete mass-supported horizontal component touching the tee as
    # the acoustic donor.  Taking all of the required mass from the one tee
    # cell can empty that cell when the riser cut volume is wider than the
    # horizontal cell; the next Riemann solve then sees another artificial
    # vacuum.  Proportional removal from the connected component is the
    # conservative finite-volume remap for adding a cut-cell volume and leaves
    # every donor velocity unchanged.
    first = junction
    while first > 0 and h_supported[first - 1]:
        first -= 1
    last = junction + 1
    while last < hm.size and h_supported[last]:
        last += 1
    donor_indices = np.arange(first, last, dtype=int)
    donor_volume = float(
        np.sum(hag[donor_indices]) * horizontal_width
    )
    donor_mass = float(np.sum(hm[donor_indices]))
    if donor_volume <= 0.0 or donor_mass <= 0.0:
        return hm, hj, vm, vj, vc

    receiver_indices = np.flatnonzero(receiver & ~supported)
    receiver_volumes = vag[receiver_indices] * vertical_width
    positive_geometry = receiver_volumes > 0.0
    receiver_indices = receiver_indices[positive_geometry]
    receiver_volumes = receiver_volumes[positive_geometry]
    if receiver_indices.size == 0:
        return hm, hj, vm, vj, vc

    donor_mass = float(np.sum(hm[donor_indices]))
    total_mass = donor_mass + float(np.sum(vm[receiver_indices]))
    total_volume = donor_volume + float(np.sum(receiver_volumes))
    if total_mass <= 0.0 or total_volume <= 0.0:
        return hm, hj, vm, vj, vc
    common_density = total_mass / total_volume
    requested = np.maximum(
        common_density * receiver_volumes - vm[receiver_indices], 0.0
    )
    total_transfer = min(float(np.sum(requested)), donor_mass)
    if total_transfer <= 0.0:
        # This routine only initialises newly opened volume.  Any reverse gas
        # motion is advanced by the conservative Riemann flux below; returning
        # non-tracer gas here would be relabelled as horizontal tracer mass.
        return hm, hj, vm, vj, vc
    if float(np.sum(requested)) > total_transfer:
        requested *= total_transfer / float(np.sum(requested))
    retained_fraction = max(
        (donor_mass - total_transfer) / donor_mass, 0.0
    )
    hm[donor_indices] *= retained_fraction
    hj[donor_indices] *= retained_fraction
    vm[receiver_indices] += requested
    vc[receiver_indices] += requested
    vc[receiver_indices] = np.minimum(
        np.maximum(vc[receiver_indices], 0.0), vm[receiver_indices]
    )
    return hm, hj, vm, vj, vc


def _displace_gas_from_closed_vertical_material_cells(
    total_mass: np.ndarray,
    momentum: np.ndarray,
    tracer_mass: np.ndarray,
    raw_gas_area: np.ndarray,
    *,
    full_area: float,
    cell_width: float,
    rho_reference: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    topology_density_fraction: float,
    resolved_density_fraction: float,
    resolved_density_ceiling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Move a material gas parcel out of a liquid-closed riser cell.

    Liquid and gas are advanced in split conservative stages.  During the
    liquid stage a previously open material-gas cell can become geometrically
    full.  Keeping its finite tracer inventory in the numerical void floor
    stores an arbitrarily large delayed EOS pressure.  The liquid closure is
    instead a moving gas boundary: immediately adjacent open void receives the
    displaced parcel, carrying total mass, tunnel tracer and gas momentum
    together.

    Only contiguous cells whose void has fallen to the numerical floor and
    which contain either material tracer or resolved compressed gas are
    remapped.  This also covers atmospheric headspace displaced by a rising
    liquid parcel without touching the ordinary positivity background.  A
    block is sent to its adjacent open side;
    when both sides are open, the sign of its resolved gas momentum selects
    the downstream side, with a volume-weighted split only for a stagnant
    block.  Remote gas cells are unchanged and all three conserved inventories
    close to roundoff.
    """

    mass = np.maximum(np.asarray(total_mass, dtype=float), 0.0).copy()
    gas_momentum = np.asarray(momentum, dtype=float).copy()
    tracer = np.maximum(np.asarray(tracer_mass, dtype=float), 0.0).copy()
    raw = np.maximum(np.asarray(raw_gas_area, dtype=float), 0.0)
    if not (
        mass.shape == gas_momentum.shape == tracer.shape == raw.shape
    ) or mass.ndim != 1:
        raise ValueError("vertical closure-remap arrays must be equal 1-D fields")
    if (
        full_area <= 0.0
        or cell_width <= 0.0
        or rho_reference <= 0.0
        or not 0.0 < void_floor_fraction < active_void_fraction < 1.0
        or not 0.0 < topology_density_fraction < resolved_density_fraction < 1.0
        or resolved_density_ceiling <= 1.0
    ):
        raise ValueError("invalid vertical closure-remap scales")
    if np.any(tracer > mass + 1.0e-14):
        raise ValueError("vertical tracer mass exceeds total gas mass")

    tracer_threshold = (
        topology_density_fraction
        * rho_reference
        * void_floor_fraction
        * full_area
        * cell_width
    )
    compressed_mass_threshold = (
        resolved_density_ceiling
        * rho_reference
        * void_floor_fraction
        * full_area
        * cell_width
    )
    closed_material = (
        raw <= 1.5 * void_floor_fraction * full_area
    ) & (
        (tracer > tracer_threshold)
        | (mass > compressed_mass_threshold)
    )
    open_void = raw >= active_void_fraction * full_area
    displaced = 0.0
    index = 0
    while index < mass.size:
        if not closed_material[index]:
            index += 1
            continue
        first = index
        while index < mass.size and closed_material[index]:
            index += 1
        last = index
        recipients: list[int] = []
        if first > 0 and open_void[first - 1]:
            recipients.append(first - 1)
        if last < mass.size and open_void[last]:
            recipients.append(last)
        if not recipients:
            # A genuinely sealed capsule remains a compressed gas state; it
            # cannot be teleported through liquid to a remote void.
            continue

        block = np.arange(first, last, dtype=int)
        parcel_mass = float(np.sum(mass[block]))
        parcel_tracer = float(np.sum(tracer[block]))
        parcel_momentum = float(np.sum(gas_momentum[block]))
        if parcel_mass <= 0.0:
            continue
        if len(recipients) == 1:
            weights = np.ones(1, dtype=float)
        elif parcel_momentum > 1.0e-18:
            weights = np.array([0.0, 1.0])
        elif parcel_momentum < -1.0e-18:
            weights = np.array([1.0, 0.0])
        else:
            recipient_void = raw[np.asarray(recipients, dtype=int)]
            weights = recipient_void / float(np.sum(recipient_void))

        mass[block] = 0.0
        tracer[block] = 0.0
        gas_momentum[block] = 0.0
        for receiver, weight in zip(recipients, weights):
            mass[receiver] += weight * parcel_mass
            tracer[receiver] += weight * parcel_tracer
            gas_momentum[receiver] += weight * parcel_momentum
        displaced += parcel_mass

    return mass, gas_momentum, tracer, displaced


def _apply_side_t_phase_separation(
    front_receiver: np.ndarray,
    mass_supported: np.ndarray,
    *,
    junction_index: int,
    vertical_branch_receiving: bool,
) -> np.ndarray:
    """Prefer an actually receiving upper branch at the side tee.

    A massless area deficit in the liquid-full downstream dead leg is elastic
    pressure storage, not automatically a material gas receiver.  Suppress the
    new east receiver only while the riser has a geometrically open base and
    its gas Riemann predictor is directed upward.  If the vertical branch is
    closed or its predicted flux reverses, the ordinary east receiver remains
    available.  An east cell that already contains supported gas is never
    disconnected.  Thus the rule is a local state-dependent branch competition,
    not a permanent topology diode or a prescribed gas-transfer fraction.
    """

    receiver = np.asarray(front_receiver, dtype=bool).copy()
    supported = np.asarray(mass_supported, dtype=bool)
    if receiver.shape != supported.shape or receiver.ndim != 1:
        raise ValueError("side-T topology arrays must be equal and one-dimensional")
    junction = int(junction_index)
    if not 0 <= junction < receiver.size:
        raise ValueError("side-T junction lies outside the horizontal topology")
    east = junction + 1
    if (
        east < receiver.size
        and bool(vertical_branch_receiving)
        and supported[junction]
        and not supported[east]
    ):
        receiver[east] = False
    return receiver


def _apply_downstream_material_front_kinematics(
    front_receiver: np.ndarray,
    mass_supported: np.ndarray,
    liquid_area: np.ndarray,
    liquid_discharge: np.ndarray,
    *,
    junction_index: int,
) -> np.ndarray:
    """Open downstream gas receivers only behind an advancing liquid front.

    The TPA liquid state can contain a long massless area deficit in the closed
    downstream leg: that is elastic rarefaction, not a pre-existing gas path.
    A material gas front may sweep the next cell only when the liquid interface
    at that face is moving away from the gas.  This is the discrete kinematic
    condition ``dx_f/dt = u_interface``.  It prevents an acoustic gas solve from
    instantaneously filling the complete elastic deficit while still allowing
    pressure-driven east penetration and its later arrest or retreat.
    """

    receiver = np.asarray(front_receiver, dtype=bool).copy()
    supported = np.asarray(mass_supported, dtype=bool)
    area = np.maximum(np.asarray(liquid_area, dtype=float), 0.0)
    discharge = np.asarray(liquid_discharge, dtype=float)
    if not (
        receiver.shape == supported.shape == area.shape == discharge.shape
    ) or receiver.ndim != 1:
        raise ValueError("downstream-front arrays must be equal and one-dimensional")
    junction = int(junction_index)
    if not 0 <= junction < receiver.size:
        raise ValueError("downstream-front junction lies outside the grid")
    if receiver.size < 2 or junction >= receiver.size - 1:
        return receiver

    velocity = np.divide(
        discharge,
        np.maximum(area, 1.0e-14),
        out=np.zeros_like(discharge),
        where=area > 1.0e-14,
    )
    face_velocity = 0.5 * (velocity[:-1] + velocity[1:])
    for index in range(junction + 1, receiver.size):
        if not receiver[index] or supported[index]:
            continue
        from_west = bool(
            supported[index - 1] and face_velocity[index - 1] > 0.0
        )
        from_east = bool(
            index + 1 < receiver.size
            and supported[index + 1]
            and face_velocity[index] < 0.0
        )
        if not (from_west or from_east):
            receiver[index] = False
    return receiver


@njit(cache=True)
def _minmod3(a: float, b: float, c: float) -> float:
    if a > 0.0 and b > 0.0 and c > 0.0:
        return min(a, b, c)
    if a < 0.0 and b < 0.0 and c < 0.0:
        return max(a, b, c)
    return 0.0


@njit(cache=True)
def _friction_factor(reynolds: float) -> float:
    re = max(reynolds, 1.0e-12)
    if re < 2100.0:
        value = 16.0 / re
    else:
        value = 0.046 * re**-0.2
    return min(max(value, 0.0), 4.0)


@njit(cache=True)
def _entropy_abs(value: float, delta: float) -> float:
    magnitude = abs(value)
    if magnitude >= delta:
        return magnitude
    return 0.5 * (value * value / delta + delta)


@njit(cache=True)
def _roe_flux(
    rho_l: float,
    velocity_l: float,
    rho_r: float,
    velocity_r: float,
    sound_speed: float,
    rho_atm: float,
    entropy_fraction: float,
) -> tuple[float, float]:
    """Two-wave Roe flux for isothermal Euler, per unit open area."""

    rho_l = max(rho_l, 1.0e-10)
    rho_r = max(rho_r, 1.0e-10)
    p_l = (rho_l - rho_atm) * sound_speed * sound_speed
    p_r = (rho_r - rho_atm) * sound_speed * sound_speed
    mass_l = rho_l * velocity_l
    mass_r = rho_r * velocity_r
    flux_mass_l = mass_l
    flux_mass_r = mass_r
    flux_momentum_l = mass_l * velocity_l + p_l
    flux_momentum_r = mass_r * velocity_r + p_r

    root_l = math.sqrt(rho_l)
    root_r = math.sqrt(rho_r)
    u_roe = (
        root_l * velocity_l + root_r * velocity_r
    ) / max(root_l + root_r, 1.0e-14)
    delta_rho = rho_r - rho_l
    delta_momentum = mass_r - mass_l
    alpha_minus = (
        (u_roe + sound_speed) * delta_rho - delta_momentum
    ) / (2.0 * sound_speed)
    alpha_plus = (
        delta_momentum - (u_roe - sound_speed) * delta_rho
    ) / (2.0 * sound_speed)
    entropy_delta = entropy_fraction * sound_speed
    lambda_minus = _entropy_abs(u_roe - sound_speed, entropy_delta)
    lambda_plus = _entropy_abs(u_roe + sound_speed, entropy_delta)
    diss_mass = lambda_minus * alpha_minus + lambda_plus * alpha_plus
    diss_momentum = (
        lambda_minus * alpha_minus * (u_roe - sound_speed)
        + lambda_plus * alpha_plus * (u_roe + sound_speed)
    )
    return (
        0.5 * (flux_mass_l + flux_mass_r) - 0.5 * diss_mass,
        0.5 * (flux_momentum_l + flux_momentum_r)
        - 0.5 * diss_momentum,
    )


@njit(cache=True)
def _einfeldt_flux(
    rho_l: float,
    velocity_l: float,
    rho_r: float,
    velocity_r: float,
    sound_speed: float,
    rho_atm: float,
) -> tuple[float, float]:
    """Positivity-preserving two-wave flux for a gas-vacuum face."""

    rho_l = max(rho_l, 1.0e-12)
    rho_r = max(rho_r, 1.0e-12)
    p_l = (rho_l - rho_atm) * sound_speed * sound_speed
    p_r = (rho_r - rho_atm) * sound_speed * sound_speed
    m_l = rho_l * velocity_l
    m_r = rho_r * velocity_r
    fm_l = m_l
    fm_r = m_r
    fj_l = m_l * velocity_l + p_l
    fj_r = m_r * velocity_r + p_r
    speed_l = min(velocity_l - sound_speed, velocity_r - sound_speed)
    speed_r = max(velocity_l + sound_speed, velocity_r + sound_speed)
    if speed_l >= 0.0:
        return fm_l, fj_l
    if speed_r <= 0.0:
        return fm_r, fj_r
    denominator = max(speed_r - speed_l, 1.0e-14)
    return (
        (
            speed_r * fm_l - speed_l * fm_r
            + speed_l * speed_r * (rho_r - rho_l)
        ) / denominator,
        (
            speed_r * fj_l - speed_l * fj_r
            + speed_l * speed_r * (m_r - m_l)
        ) / denominator,
    )


@njit(cache=True)
def _gas_flux(
    rho_l: float,
    velocity_l: float,
    rho_r: float,
    velocity_r: float,
    sound_speed: float,
    rho_atm: float,
    entropy_fraction: float,
    resolved_density_fraction: float,
    resolved_density_ceiling: float,
) -> tuple[float, float]:
    density_min = min(rho_l, rho_r)
    density_max = max(rho_l, rho_r)
    unresolved_vacuum = (
        density_min < resolved_density_fraction * rho_atm
        or density_max > resolved_density_ceiling * rho_atm
        or density_min < 1.0e-5 * density_max
    )
    if unresolved_vacuum:
        return _einfeldt_flux(
            rho_l,
            velocity_l,
            rho_r,
            velocity_r,
            sound_speed,
            rho_atm,
        )
    return _roe_flux(
        rho_l,
        velocity_l,
        rho_r,
        velocity_r,
        sound_speed,
        rho_atm,
        entropy_fraction,
    )


def isothermal_ideal_gas_riemann_flux(
    density_left: float,
    velocity_left: float,
    density_right: float,
    velocity_right: float,
    *,
    gas_constant: float,
    temperature: float,
    entropy_fix_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return an isothermal ideal-gas Riemann flux per unit mouth area.

    The pressure law is ``p = rho R T`` and the returned pair contains mass
    and normal-momentum fluxes.  This is the ordinary positive-density Roe
    two-wave problem; unlike the distributed network's gas-vacuum fallback,
    no topology threshold or density-relative switch enters this interface.
    """

    values = (
        density_left,
        velocity_left,
        density_right,
        velocity_right,
        gas_constant,
        temperature,
        entropy_fix_fraction,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Riemann states must be finite")
    if density_left <= 0.0 or density_right <= 0.0:
        raise ValueError("Riemann-state densities must be positive")
    if gas_constant <= 0.0 or temperature <= 0.0:
        raise ValueError("ideal-gas constants must be positive")
    if entropy_fix_fraction < 0.0:
        raise ValueError("entropy-fix fraction must be non-negative")

    sound_speed = math.sqrt(gas_constant * temperature)
    mass_flux, momentum_flux = _roe_flux(
        float(density_left),
        float(velocity_left),
        float(density_right),
        float(velocity_right),
        sound_speed,
        0.0,
        float(entropy_fix_fraction),
    )
    return float(mass_flux), float(momentum_flux)


def advance_lumped_isothermal_side_t(
    horizontal_inventory: OpenIsothermalGasInventory,
    vertical_mass: np.ndarray,
    *,
    vertical_receiver_volume: float,
    mouth_area: float,
    dt: float,
    receiver_index: int = 0,
    horizontal_normal_velocity: float = 0.0,
    vertical_normal_velocity: float = 0.0,
    entropy_fix_fraction: float = 0.10,
) -> LumpedSideTGasAdvance:
    """Advance one conservative open-pocket/vertical-receiver mass exchange.

    Positive normal direction is from the horizontal pocket into the riser.
    The flux is determined only by the two instantaneous ideal-gas states and
    their normal velocities.  The sole post-flux operation is a conservative
    donor-positivity bound, needed when a caller supplies a time step larger
    than the local gas CFL limit.
    """

    masses = np.asarray(vertical_mass, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("vertical_mass must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("vertical gas masses must be finite and non-negative")
    if not isinstance(receiver_index, (int, np.integer)):
        raise TypeError("receiver_index must be an integer")
    receiver = int(receiver_index)
    if receiver < 0 or receiver >= masses.size:
        raise IndexError("receiver_index is outside vertical_mass")

    scalars = (
        vertical_receiver_volume,
        mouth_area,
        dt,
        horizontal_normal_velocity,
        vertical_normal_velocity,
        entropy_fix_fraction,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("side-T state and geometry must be finite")
    if vertical_receiver_volume <= 0.0:
        raise ValueError("vertical receiver volume must be positive")
    if mouth_area < 0.0 or dt < 0.0:
        raise ValueError("mouth area and time step must be non-negative")

    receiver_mass = float(masses[receiver])
    if horizontal_inventory.mass <= 0.0 or receiver_mass <= 0.0:
        raise ValueError(
            "both Riemann states require positive gas mass; reduce the prior "
            "time step before a donor is exhausted"
        )
    receiver_density = receiver_mass / vertical_receiver_volume
    mass_flux, momentum_flux = isothermal_ideal_gas_riemann_flux(
        horizontal_inventory.density,
        horizontal_normal_velocity,
        receiver_density,
        vertical_normal_velocity,
        gas_constant=horizontal_inventory.gas_constant,
        temperature=horizontal_inventory.temperature,
        entropy_fix_fraction=entropy_fix_fraction,
    )

    raw_transfer = mass_flux * mouth_area * dt
    transfer = min(
        max(raw_transfer, -receiver_mass),
        horizontal_inventory.mass,
    )
    horizontal_after_mass = horizontal_inventory.mass - transfer
    vertical_after = masses.copy()
    vertical_after[receiver] = receiver_mass + transfer

    # The clamp above is algebraically positivity preserving.  Remove only a
    # possible signed zero so downstream state checks are unambiguous.
    if horizontal_after_mass == 0.0:
        horizontal_after_mass = 0.0
    if vertical_after[receiver] == 0.0:
        vertical_after[receiver] = 0.0

    total_before = math.fsum([horizontal_inventory.mass, *masses.tolist()])
    total_after = math.fsum([horizontal_after_mass, *vertical_after.tolist()])
    return LumpedSideTGasAdvance(
        horizontal_inventory=horizontal_inventory.with_state(
            mass=horizontal_after_mass
        ),
        vertical_mass=vertical_after,
        mass_flux=mass_flux,
        normal_momentum_flux=momentum_flux,
        raw_mass_transfer=raw_transfer,
        mass_transfer=transfer,
        conservation_error=total_after - total_before,
        donor_limited=(transfer != raw_transfer),
    )


@njit(cache=True)
def _slopes(values: np.ndarray, theta: float) -> np.ndarray:
    n = values.size
    slope = np.zeros(n)
    for i in range(1, n - 1):
        left = values[i] - values[i - 1]
        right = values[i + 1] - values[i]
        centre = 0.5 * (values[i + 1] - values[i - 1])
        slope[i] = _minmod3(theta * left, centre, theta * right)
    return slope


@njit(cache=True)
def _network_rhs(
    h_mass: np.ndarray,
    h_momentum: np.ndarray,
    v_mass: np.ndarray,
    v_momentum: np.ndarray,
    v_tracer: np.ndarray,
    h_area_g: np.ndarray,
    h_depth_l: np.ndarray,
    h_perimeter_g: np.ndarray,
    h_hydraulic: np.ndarray,
    v_area_g: np.ndarray,
    v_perimeter_g: np.ndarray,
    v_hydraulic: np.ndarray,
    h_face_area: np.ndarray,
    v_face_area: np.ndarray,
    h_active: np.ndarray,
    v_active: np.ndarray,
    v_liquid_pressure_coupled: np.ndarray,
    dx: float,
    dz: float,
    junction_index: int,
    mouth_area: float,
    rho_l: float,
    gravity: float,
    gas_viscosity: float,
    sound_speed: float,
    rho_atm: float,
    limiter_theta: float,
    entropy_fraction: float,
    resolved_density_fraction: float,
    resolved_density_ceiling: float,
) -> tuple:
    nh = h_mass.size
    nv = v_mass.size
    h_rho = np.empty(nh)
    h_u = np.zeros(nh)
    v_rho = np.empty(nv)
    v_u = np.zeros(nv)
    for i in range(nh):
        h_rho[i] = max(h_mass[i] / h_area_g[i], 1.0e-10)
        if h_mass[i] > resolved_density_fraction * rho_atm * h_area_g[i]:
            h_u[i] = h_momentum[i] / h_mass[i]
    for i in range(nv):
        v_rho[i] = max(v_mass[i] / v_area_g[i], 1.0e-10)
        if v_mass[i] > resolved_density_fraction * rho_atm * v_area_g[i]:
            v_u[i] = v_momentum[i] / v_mass[i]

    h_sr = _slopes(h_rho, limiter_theta)
    h_su = _slopes(h_u, limiter_theta)
    v_sr = _slopes(v_rho, limiter_theta)
    v_su = _slopes(v_u, limiter_theta)
    v_concentration = np.zeros(nv)
    for i in range(nv):
        if v_mass[i] > 1.0e-14:
            v_concentration[i] = min(
                max(v_tracer[i] / v_mass[i], 0.0), 1.0
            )
    v_sc = _slopes(v_concentration, limiter_theta)

    hf_mass = np.zeros(nh + 1)
    hf_momentum = np.zeros(nh + 1)
    hf_momentum[0] = (
        (h_rho[0] - rho_atm) * sound_speed * sound_speed
        * h_face_area[0]
    )
    hf_momentum[nh] = (
        (h_rho[nh - 1] - rho_atm) * sound_speed * sound_speed
        * h_face_area[nh]
    )
    for face in range(1, nh):
        if h_face_area[face] <= 0.0:
            continue
        left = face - 1
        right = face
        fm, fj = _gas_flux(
            max(h_rho[left] + 0.5 * h_sr[left], 1.0e-10),
            h_u[left] + 0.5 * h_su[left],
            max(h_rho[right] - 0.5 * h_sr[right], 1.0e-10),
            h_u[right] - 0.5 * h_su[right],
            sound_speed,
            rho_atm,
            entropy_fraction,
            resolved_density_fraction,
            resolved_density_ceiling,
        )
        hf_mass[face] = fm * h_face_area[face]
        hf_momentum[face] = fj * h_face_area[face]

    vf_mass = np.zeros(nv + 1)
    vf_momentum = np.zeros(nv + 1)
    vf_tracer = np.zeros(nv + 1)
    for face in range(1, nv):
        if v_face_area[face] <= 0.0:
            continue
        left = face - 1
        right = face
        fm, fj = _gas_flux(
            max(v_rho[left] + 0.5 * v_sr[left], 1.0e-10),
            v_u[left] + 0.5 * v_su[left],
            max(v_rho[right] - 0.5 * v_sr[right], 1.0e-10),
            v_u[right] - 0.5 * v_su[right],
            sound_speed,
            rho_atm,
            entropy_fraction,
            resolved_density_fraction,
            resolved_density_ceiling,
        )
        vf_mass[face] = fm * v_face_area[face]
        vf_momentum[face] = fj * v_face_area[face]
        if vf_mass[face] >= 0.0:
            concentration = v_concentration[left] + 0.5 * v_sc[left]
        else:
            concentration = v_concentration[right] - 0.5 * v_sc[right]
        vf_tracer[face] = vf_mass[face] * min(max(concentration, 0.0), 1.0)

    # Atmospheric opening at the top.  Incoming atmosphere carries no pocket
    # tracer; outgoing gas carries the local upwind tracer concentration.
    if v_face_area[nv] > 0.0:
        fm_top, fj_top = _gas_flux(
            v_rho[nv - 1],
            v_u[nv - 1],
            rho_atm,
            0.0,
            sound_speed,
            rho_atm,
            entropy_fraction,
            resolved_density_fraction,
            resolved_density_ceiling,
        )
        vf_mass[nv] = fm_top * v_face_area[nv]
        vf_momentum[nv] = fj_top * v_face_area[nv]
        if vf_mass[nv] > 0.0:
            concentration = (
                v_concentration[nv - 1] + 0.5 * v_sc[nv - 1]
            )
            vf_tracer[nv] = vf_mass[nv] * min(max(concentration, 0.0), 1.0)

    # Shared horizontal-to-vertical T face.  The horizontal axial velocity is
    # tangential to this face; its normal trace is zero.  The vertical trace is
    # the first riser-cell state.  The Roe mass flux is therefore determined by
    # the local pressures and the resolved vertical normal velocity.
    junction_mass_flux = 0.0
    junction_momentum_flux = 0.0
    junction_tracer_flux = 0.0
    if mouth_area > 0.0 and h_active[junction_index]:
        fm_t, fj_t = _gas_flux(
            h_rho[junction_index],
            0.0,
            v_rho[0],
            v_u[0],
            sound_speed,
            rho_atm,
            entropy_fraction,
            resolved_density_fraction,
            resolved_density_ceiling,
        )
        junction_mass_flux = fm_t * mouth_area
        junction_momentum_flux = fj_t * mouth_area
        if junction_mass_flux >= 0.0:
            junction_tracer_flux = junction_mass_flux
        else:
            concentration = v_tracer[0] / max(v_mass[0], 1.0e-14)
            junction_tracer_flux = junction_mass_flux * min(
                max(concentration, 0.0), 1.0
            )
        vf_mass[0] = junction_mass_flux
        vf_momentum[0] = junction_momentum_flux
        vf_tracer[0] = junction_tracer_flux

    rhs_hm = np.zeros(nh)
    rhs_hj = np.zeros(nh)
    rhs_vm = np.zeros(nv)
    rhs_vj = np.zeros(nv)
    rhs_vc = np.zeros(nv)

    for i in range(nh):
        if not h_active[i]:
            continue
        rhs_hm[i] = -(hf_mass[i + 1] - hf_mass[i]) / dx
        rhs_hj[i] = -(hf_momentum[i + 1] - hf_momentum[i]) / dx
        if nh == 1:
            # A lumped horizontal reservoir has no resolved axial geometry
            # gradient.  Its only open gas face is the normal side-T mouth.
            depth_gradient = 0.0
        elif i == 0:
            depth_gradient = (h_depth_l[1] - h_depth_l[0]) / dx
        elif i == nh - 1:
            depth_gradient = (h_depth_l[nh - 1] - h_depth_l[nh - 2]) / dx
        else:
            depth_gradient = (h_depth_l[i + 1] - h_depth_l[i - 1]) / (2.0 * dx)
        pressure = (h_rho[i] - rho_atm) * sound_speed * sound_speed
        rhs_hj[i] += pressure * (h_face_area[i + 1] - h_face_area[i]) / dx
        rhs_hj[i] -= h_area_g[i] * h_rho[i] * gravity * depth_gradient
        re_wall = h_rho[i] * abs(h_u[i]) * h_hydraulic[i] / gas_viscosity
        wall = 0.5 * _friction_factor(re_wall) * h_rho[i] * h_u[i] * abs(h_u[i]) * h_perimeter_g[i]
        # Gas--liquid drag is a stiff, internal momentum exchange.  It is
        # applied after gas transport by the conservative semi-implicit solve
        # in ``_implicit_interphase_drag_exchange``.  Accumulating it here
        # while holding the liquid velocity fixed over all gas substeps
        # over-transfers momentum into thin liquid layers.
        rhs_hj[i] -= wall

    if mouth_area > 0.0:
        rhs_hm[junction_index] -= junction_mass_flux / dx
        if junction_mass_flux > 0.0:
            rhs_hj[junction_index] -= (
                junction_mass_flux * h_u[junction_index] / dx
            )

    for i in range(nv):
        if not v_active[i]:
            continue
        rhs_vm[i] = -(vf_mass[i + 1] - vf_mass[i]) / dz
        rhs_vj[i] = -(vf_momentum[i + 1] - vf_momentum[i]) / dz
        rhs_vc[i] = -(vf_tracer[i + 1] - vf_tracer[i]) / dz
        pressure = (v_rho[i] - rho_atm) * sound_speed * sound_speed
        rhs_vj[i] += pressure * (v_face_area[i + 1] - v_face_area[i]) / dz
        # The phases share the vertical hydrostatic pressure gradient.  The
        # liquid equation receives its area-weighted part in the riser solve;
        # the gas equation must receive the complementary ``A_g rho_l g``
        # pressure force.  Together with gas weight below this gives the
        # resolved buoyancy ``(rho_l-rho_g) g A_g``.  Omitting this term left
        # tracer gas stationary after its fitted nose reached the open top, so
        # the nominal atmospheric boundary carried almost no gas mass.
        if v_liquid_pressure_coupled[i]:
            rhs_vj[i] += rho_l * gravity * v_area_g[i]
        rhs_vj[i] -= v_mass[i] * gravity
        re_wall = v_rho[i] * abs(v_u[i]) * v_hydraulic[i] / gas_viscosity
        wall = 0.5 * _friction_factor(re_wall) * v_rho[i] * v_u[i] * abs(v_u[i]) * v_perimeter_g[i]
        rhs_vj[i] -= wall

    return (
        rhs_hm,
        rhs_hj,
        rhs_vm,
        rhs_vj,
        rhs_vc,
        vf_mass[nv],
        vf_tracer[nv],
        junction_mass_flux,
    )


@njit(cache=True)
def _valid_state(
    h_mass: np.ndarray,
    v_mass: np.ndarray,
    v_tracer: np.ndarray,
) -> bool:
    for value in h_mass:
        if not math.isfinite(value) or value < -1.0e-13:
            return False
    for i in range(v_mass.size):
        if not math.isfinite(v_mass[i]) or v_mass[i] < -1.0e-13:
            return False
        if not math.isfinite(v_tracer[i]) or v_tracer[i] < -1.0e-13:
            return False
        if v_tracer[i] > v_mass[i] + 1.0e-12:
            return False
    return True


@njit(cache=True)
def _compiled_advance(
    h_mass: np.ndarray,
    h_momentum: np.ndarray,
    v_mass: np.ndarray,
    v_momentum: np.ndarray,
    v_tracer: np.ndarray,
    h_area_g: np.ndarray,
    h_depth_l: np.ndarray,
    h_perimeter_g: np.ndarray,
    h_hydraulic: np.ndarray,
    v_area_g: np.ndarray,
    v_perimeter_g: np.ndarray,
    v_hydraulic: np.ndarray,
    h_face_area: np.ndarray,
    v_face_area: np.ndarray,
    h_active: np.ndarray,
    v_active: np.ndarray,
    v_liquid_pressure_coupled: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    junction_index: int,
    mouth_area: float,
    rho_l: float,
    gravity: float,
    gas_viscosity: float,
    sound_speed: float,
    rho_atm: float,
    cfl: float,
    limiter_theta: float,
    entropy_fraction: float,
    resolved_density_fraction: float,
    resolved_density_ceiling: float,
) -> tuple:
    elapsed = 0.0
    escaped_tracer = 0.0
    atmospheric_exchange = 0.0
    junction_transfer = 0.0
    maximum_velocity = 0.0
    substeps = 0

    while elapsed < dt - 1.0e-15:
        current_max = 0.0
        for i in range(h_mass.size):
            if (
                h_active[i]
                and h_mass[i] > resolved_density_fraction * rho_atm * h_area_g[i]
            ):
                current_max = max(current_max, abs(h_momentum[i] / h_mass[i]))
        for i in range(v_mass.size):
            if (
                v_active[i]
                and v_mass[i] > resolved_density_fraction * rho_atm * v_area_g[i]
            ):
                current_max = max(current_max, abs(v_momentum[i] / v_mass[i]))
        maximum_velocity = max(maximum_velocity, current_max)
        trial_dt = min(
            cfl * min(dx, dz) / max(sound_speed + current_max, 1.0e-12),
            dt - elapsed,
        )

        accepted = False
        for _attempt in range(24):
            first_rhs = _network_rhs(
                h_mass, h_momentum, v_mass, v_momentum, v_tracer,
                h_area_g, h_depth_l, h_perimeter_g, h_hydraulic,
                v_area_g, v_perimeter_g, v_hydraulic,
                h_face_area, v_face_area, h_active, v_active,
                v_liquid_pressure_coupled,
                dx, dz, junction_index, mouth_area, rho_l, gravity,
                gas_viscosity, sound_speed, rho_atm, limiter_theta,
                entropy_fraction, resolved_density_fraction,
                resolved_density_ceiling,
            )
            h_mass_1 = h_mass + trial_dt * first_rhs[0]
            h_momentum_1 = h_momentum + trial_dt * first_rhs[1]
            v_mass_1 = v_mass + trial_dt * first_rhs[2]
            v_momentum_1 = v_momentum + trial_dt * first_rhs[3]
            v_tracer_1 = v_tracer + trial_dt * first_rhs[4]
            if not _valid_state(h_mass_1, v_mass_1, v_tracer_1):
                trial_dt *= 0.5
                continue
            for i in range(h_mass_1.size):
                if h_mass_1[i] <= resolved_density_fraction * rho_atm * h_area_g[i]:
                    h_momentum_1[i] = 0.0
            for i in range(v_mass_1.size):
                if v_mass_1[i] <= resolved_density_fraction * rho_atm * v_area_g[i]:
                    v_momentum_1[i] = 0.0

            second_rhs = _network_rhs(
                h_mass_1, h_momentum_1, v_mass_1, v_momentum_1, v_tracer_1,
                h_area_g, h_depth_l, h_perimeter_g, h_hydraulic,
                v_area_g, v_perimeter_g, v_hydraulic,
                h_face_area, v_face_area, h_active, v_active,
                v_liquid_pressure_coupled,
                dx, dz, junction_index, mouth_area, rho_l, gravity,
                gas_viscosity, sound_speed, rho_atm, limiter_theta,
                entropy_fraction, resolved_density_fraction,
                resolved_density_ceiling,
            )
            h_mass_2 = h_mass_1 + trial_dt * second_rhs[0]
            h_momentum_2 = h_momentum_1 + trial_dt * second_rhs[1]
            v_mass_2 = v_mass_1 + trial_dt * second_rhs[2]
            v_momentum_2 = v_momentum_1 + trial_dt * second_rhs[3]
            v_tracer_2 = v_tracer_1 + trial_dt * second_rhs[4]
            if not _valid_state(h_mass_2, v_mass_2, v_tracer_2):
                trial_dt *= 0.5
                continue
            for i in range(h_mass_2.size):
                if h_mass_2[i] <= resolved_density_fraction * rho_atm * h_area_g[i]:
                    h_momentum_2[i] = 0.0
            for i in range(v_mass_2.size):
                if v_mass_2[i] <= resolved_density_fraction * rho_atm * v_area_g[i]:
                    v_momentum_2[i] = 0.0

            h_mass = 0.5 * h_mass + 0.5 * h_mass_2
            h_momentum = 0.5 * h_momentum + 0.5 * (
                h_momentum_2
            )
            v_mass = 0.5 * v_mass + 0.5 * v_mass_2
            v_momentum = 0.5 * v_momentum + 0.5 * (
                v_momentum_2
            )
            v_tracer = 0.5 * v_tracer + 0.5 * v_tracer_2
            atmospheric_exchange += 0.5 * trial_dt * (
                first_rhs[5] + second_rhs[5]
            )
            escaped_tracer += 0.5 * trial_dt * (
                first_rhs[6] + second_rhs[6]
            )
            junction_transfer += 0.5 * trial_dt * (
                first_rhs[7] + second_rhs[7]
            )
            accepted = True
            break
        if not accepted:
            raise FloatingPointError("gas network could not find a positive SSP-RK2 step")

        for i in range(h_mass.size):
            if h_mass[i] < 0.0 and h_mass[i] > -1.0e-13:
                h_mass[i] = 0.0
            if h_mass[i] <= resolved_density_fraction * rho_atm * h_area_g[i]:
                h_momentum[i] = 0.0
        for i in range(v_mass.size):
            if v_mass[i] < 0.0 and v_mass[i] > -1.0e-13:
                v_mass[i] = 0.0
            if v_tracer[i] < 0.0 and v_tracer[i] > -1.0e-13:
                v_tracer[i] = 0.0
            if v_mass[i] <= resolved_density_fraction * rho_atm * v_area_g[i]:
                v_momentum[i] = 0.0
        elapsed += trial_dt
        substeps += 1

    return (
        h_mass, h_momentum, v_mass, v_momentum, v_tracer,
        escaped_tracer, atmospheric_exchange,
        junction_transfer, substeps, maximum_velocity,
    )


def _implicit_interphase_drag_exchange(
    gas_mass: np.ndarray,
    gas_momentum: np.ndarray,
    liquid_area: np.ndarray,
    liquid_discharge: np.ndarray,
    gas_area: np.ndarray,
    interface_perimeter: np.ndarray,
    hydraulic_diameter: np.ndarray,
    *,
    cell_width: float,
    dt: float,
    rho_l: float,
    gas_viscosity: float,
    form_drag_diameter: float | None = None,
    total_area: float | None = None,
    liquid_viscosity: float = 1.0e-3,
    confined_bubble_froude: float | None = None,
    liquid_holdup_drag_enhancement: float = 0.0,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange gas/liquid momentum without changing mixture momentum.

    The quadratic interfacial force is frozen at the beginning of the split
    source step, while the relative velocity is integrated implicitly.  For
    ``r = u_g-u_l`` this gives ``r_new = r/(1+beta*|r|*dt)``.  The cell mixture
    velocity is invariant, so the two updated momenta are exactly equal and
    opposite.  In a stratified horizontal cell the companion model uses
    The baseline companion-model closure uses the gas-side friction factor at
    the interface.  ``liquid_holdup_drag_enhancement`` is retained only as an
    explicit sensitivity parameter and is zero in the production Case-A run;
    no fitted enhancement is silently applied.  This is the standard stable
    treatment of a stiff internal drag source and contains no velocity,
    distance, or response-history limiter.
    """

    gm = np.asarray(gas_mass, dtype=float)
    gj = np.asarray(gas_momentum, dtype=float).copy()
    al = np.asarray(liquid_area, dtype=float)
    ql = np.asarray(liquid_discharge, dtype=float)
    ag = np.asarray(gas_area, dtype=float)
    perimeter = np.asarray(interface_perimeter, dtype=float)
    dh = np.asarray(hydraulic_diameter, dtype=float)
    if not (
        gm.shape == gj.shape == al.shape == ql.shape
        == ag.shape == perimeter.shape == dh.shape
    ):
        raise ValueError("drag-exchange arrays must have equal shape")
    if liquid_holdup_drag_enhancement < 0.0:
        raise ValueError("liquid-holdup drag enhancement must be non-negative")

    q_new = ql.copy()
    liquid_mass = rho_l * np.maximum(al, 0.0) * cell_width
    active = (
        (gm > 1.0e-14)
        & (liquid_mass > 1.0e-14)
        & (ag > 1.0e-14)
        & (perimeter > 0.0)
    )
    if active_mask is not None:
        resolved = np.asarray(active_mask, dtype=bool)
        if resolved.shape != active.shape:
            raise ValueError("drag active mask must match the phase arrays")
        active &= resolved
    for index in np.flatnonzero(active):
        mg = float(gm[index])
        ml = float(liquid_mass[index])
        ug = float(gj[index] / mg)
        ul = float(ql[index] / max(al[index], 1.0e-14))
        relative = ug - ul
        if abs(relative) <= 1.0e-14:
            continue
        rho_g = mg / max(float(ag[index]) * cell_width, 1.0e-18)
        if form_drag_diameter is None:
            reynolds = (
                rho_g * abs(relative) * max(float(dh[index]), 0.0)
                / max(gas_viscosity, 1.0e-18)
            )
            friction = float(_friction_factor(reynolds))
            liquid_holdup = min(
                max(
                    float(al[index])
                    / max(float(al[index] + ag[index]), 1.0e-18),
                    0.0,
                ),
                1.0,
            )
            friction *= (
                1.0
                + float(liquid_holdup_drag_enhancement) * liquid_holdup
            )
            force_coefficient = (
                0.5 * friction * rho_g * float(perimeter[index]) * cell_width
            )
        else:
            if total_area is None or total_area <= 0.0:
                raise ValueError(
                    "total_area is required for dispersed/slug form drag"
                )
            bubble_diameter = max(float(form_drag_diameter), 1.0e-12)
            reynolds = (
                rho_l * abs(relative) * bubble_diameter
                / max(liquid_viscosity, 1.0e-18)
            )
            if reynolds < 1000.0:
                drag_coefficient = (
                    24.0 / max(reynolds, 1.0e-12)
                    * (1.0 + 0.15 * reynolds**0.687)
                )
            else:
                drag_coefficient = 0.44
            if confined_bubble_froude is not None:
                froude = max(float(confined_bubble_froude), 1.0e-6)
                # A wall-confined Taylor bubble has
                # U_t = Fr*sqrt(gD).  The equivalent quadratic form-drag
                # coefficient follows from buoyancy/drag balance and prevents
                # the unconfined-sphere C_D=0.44 limit from overpredicting its
                # rise speed in the narrow tower.
                drag_coefficient = max(
                    drag_coefficient,
                    4.0 / (3.0 * froude * froude),
                )
            alpha_g = min(
                max(float(ag[index]) / float(total_area), 0.0), 1.0
            )
            # F_i/V = 3/4 C_D rho_l alpha_g |u_r|u_r/d_b.
            # Multiplication by the cell volume gives the coefficient used in
            # the relative-velocity source equation below.
            force_coefficient = (
                0.75
                * drag_coefficient
                * rho_l
                * alpha_g
                * float(total_area)
                / bubble_diameter
                * cell_width
            )
        beta = force_coefficient * (1.0 / mg + 1.0 / ml)
        relative_new = relative / (1.0 + beta * abs(relative) * dt)
        mixture_velocity = (mg * ug + ml * ul) / (mg + ml)
        ug_new = mixture_velocity + ml / (mg + ml) * relative_new
        ul_new = mixture_velocity - mg / (mg + ml) * relative_new
        gj[index] = mg * ug_new
        q_new[index] = al[index] * ul_new
    return gj, q_new


def advance_coupled_gas_network(
    horizontal_mass: np.ndarray,
    horizontal_momentum: np.ndarray,
    vertical_total_mass: np.ndarray,
    vertical_momentum: np.ndarray,
    vertical_tracer_mass: np.ndarray,
    horizontal_liquid_area: np.ndarray,
    horizontal_liquid_discharge: np.ndarray,
    vertical_liquid_area: np.ndarray,
    vertical_liquid_discharge: np.ndarray,
    *,
    dx: float,
    dz: float,
    dt: float,
    junction_index: int,
    params: CoupledGasParameters,
    vertical_pocket_front_height: float | None = None,
    vertical_liquid_surface_height: float | None = None,
    vertical_branch_confined: bool = False,
    vertical_branch_receiving_hint: bool = False,
    horizontal_downstream_front_position: float | None = None,
    horizontal_downstream_topology_front_position: float | None = None,
    prefer_vertical_branch: bool = True,
) -> CoupledGasAdvance:
    """Advance the complete Case-A gas graph over one liquid-network step.

    ``junction_index`` is always the west cell adjacent to the discrete T face;
    the first downstream cell is therefore ``junction_index + 1``.  The same
    west cell is the conservative donor for the vertical mouth flux.
    """

    hm = np.asarray(horizontal_mass, dtype=float).copy()
    hj = np.asarray(horizontal_momentum, dtype=float).copy()
    vm = np.asarray(vertical_total_mass, dtype=float).copy()
    vj = np.asarray(vertical_momentum, dtype=float).copy()
    vc = np.asarray(vertical_tracer_mass, dtype=float).copy()
    h_al = np.asarray(horizontal_liquid_area, dtype=float).copy()
    h_ql = np.asarray(horizontal_liquid_discharge, dtype=float)
    v_al = np.asarray(vertical_liquid_area, dtype=float).copy()
    v_ql = np.asarray(vertical_liquid_discharge, dtype=float)
    if not (hm.shape == hj.shape == h_al.shape == h_ql.shape):
        raise ValueError("horizontal gas and liquid arrays must have equal shape")
    if not (vm.shape == vj.shape == vc.shape == v_al.shape == v_ql.shape):
        raise ValueError("vertical gas and liquid arrays must have equal shape")
    tracer_initial = float(np.sum(hm) + np.sum(vc))
    total_initial = float(np.sum(hm) + np.sum(vm))
    if not 0 <= int(junction_index) < hm.size:
        raise ValueError("junction index lies outside the horizontal grid")
    if (
        vertical_pocket_front_height is not None
        and float(vertical_pocket_front_height) < 0.0
    ):
        raise ValueError("vertical pocket front height must be non-negative")
    if (
        vertical_liquid_surface_height is not None
        and float(vertical_liquid_surface_height) < 0.0
    ):
        raise ValueError("vertical liquid surface height must be non-negative")
    if (
        horizontal_downstream_front_position is not None
        and float(horizontal_downstream_front_position) < 0.0
    ):
        raise ValueError("horizontal downstream front position must be non-negative")
    if (
        horizontal_downstream_topology_front_position is not None
        and float(horizontal_downstream_topology_front_position) < 0.0
    ):
        raise ValueError(
            "horizontal downstream topology front position must be non-negative"
        )
    if not 0.0 < params.resolved_density_fraction < 1.0:
        raise ValueError("resolved density fraction must lie in (0, 1)")
    if not 0.0 < params.topology_density_fraction < 1.0:
        raise ValueError("topology density fraction must lie in (0, 1)")
    if params.topology_density_fraction >= params.resolved_density_fraction:
        raise ValueError(
            "topology density fraction must be below momentum resolution"
        )
    if params.resolved_density_ceiling <= 1.0:
        raise ValueError("resolved density ceiling must exceed one atmosphere")
    if params.horizontal_holdup_drag_enhancement < 0.0:
        raise ValueError(
            "horizontal holdup-drag enhancement must be non-negative"
        )
    if not 0.0 < params.vertical_front_void_fraction < 1.0:
        raise ValueError("vertical front void fraction must lie in (0, 1)")
    if not 0.0 < params.vertical_gas_core_area_fraction < 1.0:
        raise ValueError("vertical gas-core area fraction must lie in (0, 1)")

    h_raw_geometric, _, _, _, _, _ = _horizontal_geometry(h_al, params)
    v_raw_geometric, _, _, _, _ = _vertical_geometry(v_al, params)
    vm, vj, vc, _ = _displace_gas_from_closed_vertical_material_cells(
        vm,
        vj,
        vc,
        v_raw_geometric,
        full_area=params.vertical_area,
        cell_width=dz,
        rho_reference=params.rho_atmospheric,
        void_floor_fraction=params.void_floor_fraction,
        active_void_fraction=params.active_void_fraction,
        topology_density_fraction=params.topology_density_fraction,
        resolved_density_fraction=params.resolved_density_fraction,
        resolved_density_ceiling=params.resolved_density_ceiling,
    )
    downstream_front_position = (
        None
        if horizontal_downstream_front_position is None
        else float(horizontal_downstream_front_position)
    )
    downstream_topology_front_position = (
        downstream_front_position
        if horizontal_downstream_topology_front_position is None
        else float(horizontal_downstream_topology_front_position)
    )
    old_downstream_fraction = _horizontal_downstream_material_fraction(
        h_al.size,
        dx,
        int(junction_index),
        downstream_front_position,
    )
    # A_l<A can mean either a true gas-filled crown volume or elastic pressure
    # storage on the rarefaction side of a still-full liquid pipe.  Only void
    # backed by gas mass belongs to the gas topology.  Add one adjacent empty
    # cell as the Riemann-front receiver so mass can propagate conservatively;
    # disconnected massless rarefaction streaks remain liquid-only.
    h_mass_supported = _mass_backed_gas_topology(
        h_raw_geometric,
        hm,
        full_area=params.horizontal_area,
        cell_width=dx,
        rho_reference=params.rho_atmospheric,
        void_floor_fraction=params.void_floor_fraction,
        active_void_fraction=params.active_void_fraction,
        topology_density_fraction=params.topology_density_fraction,
        resolved_density_fraction=params.resolved_density_fraction,
    )
    h_front_receiver = np.zeros_like(h_mass_supported)
    if h_mass_supported.size > 1:
        h_front_receiver[1:] |= h_mass_supported[:-1]
        h_front_receiver[:-1] |= h_mass_supported[1:]
    h_front_receiver &= (
        h_raw_geometric
        > params.horizontal_capillary_void_fraction
        * params.horizontal_area
    )
    raw_mouth_area = junction_mouth_area(
        h_raw_geometric[int(junction_index)] / params.horizontal_area,
        params,
    )
    # The resolved two-fluid path has no fitted Taylor-front flag.  Its branch
    # competition is instead supplied by the liquid/gas pressure characteristic
    # at the side tee (``vertical_branch_receiving_hint``).  Once a supported
    # gas cell exists at the riser foot, retain that decision from the actual
    # upward gas momentum as well.  This closes the former production-path hole
    # in which ``vertical_branch_confined=False`` made the east dead leg win by
    # construction even while the gas pressure was opening the riser.
    vertical_supported_pre = _mass_backed_gas_topology(
        v_raw_geometric,
        vc,
        full_area=params.vertical_area,
        cell_width=dz,
        rho_reference=params.rho_atmospheric,
        void_floor_fraction=params.void_floor_fraction,
        active_void_fraction=params.active_void_fraction,
        topology_density_fraction=params.topology_density_fraction,
        resolved_density_fraction=params.resolved_density_fraction,
    )
    vertical_base_supported = bool(vertical_supported_pre[0])
    resolved_vertical_upflow = bool(
        vertical_base_supported and float(vj[0]) > 0.0
    )
    vertical_branch_receiving = bool(
        vertical_branch_receiving_hint
        or resolved_vertical_upflow
        or (
            vertical_branch_confined
            and h_mass_supported[int(junction_index)]
            and raw_mouth_area > 0.0
            and vertical_pocket_front_height is not None
            and float(vertical_pocket_front_height) > 0.0
        )
    )
    if prefer_vertical_branch:
        # Before pneumatic breakthrough the right dead leg is still a
        # liquid-full elastic receiver.  Require motion of that liquid contact
        # before material gas can enter.  Once the riser core is open to the
        # atmosphere, however, the measured T supports counter-current branch
        # flow: gas may turn east along the crown while liquid returns west.
        # In that regime the liquid plug velocity is not the gas-front speed;
        # the conservative gas Riemann flux and one-cell receiver topology set
        # the advance instead.
        h_front_receiver = _apply_downstream_material_front_kinematics(
            h_front_receiver,
            h_mass_supported,
            h_al,
            h_ql,
            junction_index=int(junction_index),
        )
        if horizontal_downstream_front_position is not None:
            # Before pneumatic breakthrough the open riser is the only
            # material-gas outlet from the side T; the east branch is a
            # water-filled closed dead leg.  A transient elastic crown-area
            # deficit there is not a second gas path.  Keep existing supported
            # gas connected, but do not create a new east receiver until the
            # caller releases this topology after breakthrough.
            east = np.arange(h_al.size) > int(junction_index)
            h_front_receiver[east & ~h_mass_supported] = False
    front_velocity = 0.0
    retired_cell_count = 0
    if downstream_front_position is not None:
        # Separate the signed material contact from the whole-cell acoustic
        # topology envelope.  The latter is the established finite-volume
        # receiver front and remains one-way; it may contain one pressure guard
        # cell when material gas retracts.  The signed contact is used for
        # material kinematics and rendering, so internal pressure storage is
        # never mistaken for a pinned gas tongue.  A true variable-length ALE
        # cut cell would combine these roles, but coupling a signed sub-cell
        # position directly to whole-cell opening/closing creates repeated
        # O(dx) compression impulses.
        indices = np.arange(h_al.size, dtype=int)
        centres = (np.arange(h_al.size, dtype=float) + 0.5) * dx
        first_downstream = int(junction_index) + 1
        topology_front_position = float(
            max(
                downstream_front_position,
                downstream_topology_front_position,
            )
        )
        if (
            not prefer_vertical_branch
            and first_downstream < h_al.size
            and h_mass_supported[int(junction_index)]
        ):
            material_donors = np.flatnonzero(
                h_mass_supported
                & (indices >= int(junction_index))
                & (
                    (indices <= int(junction_index))
                    | (old_downstream_fraction > 0.0)
                )
            )
            if material_donors.size:
                donor = int(material_donors[-1])
                gas_velocity = float(
                    hj[donor] / max(hm[donor], 1.0e-18)
                )
                # The material gas boundary is advected by the resolved gas
                # trace.  Its sign is retained; counter-current liquid motion
                # does not turn the material surface into a check valve.
                front_velocity = gas_velocity
                if (
                    not params.allow_horizontal_front_retreat
                    and front_velocity < 0.0
                ):
                    front_velocity = 0.0
            # The fitted contact is explicit, so enforce its own geometric CFL
            # in addition to the subcycled acoustic CFL of the gas network.
            displacement = float(np.clip(
                front_velocity * dt,
                -params.cfl * dx,
                params.cfl * dx,
            ))
            downstream_front_position += displacement

            topology_leads = np.flatnonzero(
                (indices >= first_downstream)
                & (centres > topology_front_position)
            )
            topology_lead = (
                int(topology_leads[0])
                if topology_leads.size
                else h_al.size
            )
            topology_donors = np.flatnonzero(
                h_mass_supported
                & (indices >= int(junction_index))
                & (indices < topology_lead)
            )
            if topology_donors.size:
                topology_donor = int(topology_donors[-1])
                topology_velocity = float(
                    hj[topology_donor] / max(hm[topology_donor], 1.0e-18)
                )
                topology_displacement = float(np.clip(
                    max(topology_velocity, 0.0) * dt,
                    0.0,
                    params.cfl * dx,
                ))
                topology_front_position += topology_displacement
        downstream_front_anchor = (
            float(junction_index) + 0.5
        ) * dx
        downstream_front_position = float(np.clip(
            downstream_front_position,
            downstream_front_anchor,
            h_al.size * dx,
        ))
        topology_front_position = float(np.clip(
            max(topology_front_position, downstream_front_position),
            downstream_front_anchor,
            h_al.size * dx,
        ))
        downstream_topology_front_position = topology_front_position
        # Retain one pressure-active guard cell immediately ahead of a
        # retreating material contact.  The present gas solver uses whole
        # finite volumes (not variable-length cut cells), so deleting the tail
        # exactly at the west face compresses one full-cell inventory in a
        # single outer step.  A one-cell guard is the bounded first-order
        # moving-boundary closure: it is excluded from material-front velocity
        # and phase rendering, remains acoustically reversible if the front
        # advances again, and is locally merged only after it lies one complete
        # grid interval behind the contact.
        cell_left_faces = indices.astype(float) * dx
        completely_unswept = (
            (indices >= first_downstream)
            & (cell_left_faces >= topology_front_position + dx)
        )
        retiring_downstream = completely_unswept & h_mass_supported
        retired_cell_count = int(np.count_nonzero(retiring_downstream))
        if np.any(retiring_downstream):
            # A tail cell stays pressure-active while any part of it remains
            # swept.  Once the front crosses its west face, merge only that
            # cell into its west neighbour and clear it exactly; no distant
            # gas state is projected or hidden outside the transport graph.
            hm, hj = _collapse_horizontal_front_cells(
                hm,
                hj,
                h_raw_geometric,
                h_mass_supported,
                retiring_downstream,
                cell_width=dx,
            )
            h_mass_supported = _mass_backed_gas_topology(
                h_raw_geometric,
                hm,
                full_area=params.horizontal_area,
                cell_width=dx,
                rho_reference=params.rho_atmospheric,
                void_floor_fraction=params.void_floor_fraction,
                active_void_fraction=params.active_void_fraction,
                topology_density_fraction=params.topology_density_fraction,
                resolved_density_fraction=params.resolved_density_fraction,
            )
        h_mass_supported[completely_unswept] = False
        h_front_receiver[completely_unswept] = False
        # Opening remains tied to the cell-centre crossing used by the stable
        # historical finite-volume candidate.  Existing tail cells are not
        # removed when a retreat merely crosses their centre.
        pending_open = (
            (indices >= first_downstream)
            & (centres > topology_front_position)
            & ~h_mass_supported
        )
        h_front_receiver[pending_open] = False
    # A confined Taylor core is itself the resolved upward-receiving state: its
    # base has been opened by the conservative liquid displacement in this
    # outer step.  Use that geometric state here rather than a gas Riemann
    # predictor evaluated against the still-background base-cell gas mass; the
    # latter has a one-step start-up lag and incorrectly lights the elastic
    # east rarefaction.  Once the core breaks through, ``vertical_branch_confined``
    # becomes false and ordinary east/vertical Riemann competition resumes.
    h_front_receiver = _apply_side_t_phase_separation(
        h_front_receiver,
        h_mass_supported,
        junction_index=int(junction_index),
        vertical_branch_receiving=(
            prefer_vertical_branch
            and (
                vertical_branch_receiving
                or horizontal_downstream_front_position is not None
            )
        ),
    )
    hm, hj = _equilibrate_horizontal_front_receivers(
        hm,
        hj,
        h_raw_geometric,
        h_mass_supported,
        h_front_receiver,
        cell_width=dx,
    )
    # The remapped receiver is now a material gas cut-cell, not a vacuum
    # buffer.  Re-evaluate support before building face topology.
    h_mass_supported = _mass_backed_gas_topology(
        h_raw_geometric,
        hm,
        full_area=params.horizontal_area,
        cell_width=dx,
        rho_reference=params.rho_atmospheric,
        void_floor_fraction=params.void_floor_fraction,
        active_void_fraction=params.active_void_fraction,
        topology_density_fraction=params.topology_density_fraction,
        resolved_density_fraction=params.resolved_density_fraction,
    )
    if downstream_front_position is not None:
        # The topology detector is intentionally mass based and therefore
        # cannot know that a fitted front has completely vacated an east cell.
        # Reapply the material-domain mask after the receiver remap so retired
        # storage cannot be silently reactivated by this second support pass.
        h_mass_supported[completely_unswept] = False
        h_front_receiver[completely_unswept | pending_open] = False
    h_topology = h_mass_supported | h_front_receiver
    h_raw_topological = np.where(
        h_topology,
        np.maximum(
            h_raw_geometric,
            params.void_floor_fraction * params.horizontal_area,
        ),
        0.0,
    )
    h_effective_liquid = params.horizontal_area - h_raw_topological
    h_raw, h_ag, h_depth, h_pg, h_pi, h_dh = _horizontal_geometry(
        h_effective_liquid, params
    )
    geometric_mouth_area = junction_mouth_area(
        h_raw[int(junction_index)] / params.horizontal_area,
        params,
    )
    # As in the horizontal TPA state, an axial area deficit can be elastic
    # storage rather than a physical gas passage.  Build two genuine vertical
    # gas components: tunnel-origin gas connected upward from the base, and
    # atmospheric gas connected downward from the open top.  Disconnected
    # massless rarefaction cells remain liquid-only.
    v_tracer_supported = _mass_backed_gas_topology(
        v_raw_geometric,
        vc,
        full_area=params.vertical_area,
        cell_width=dz,
        rho_reference=params.rho_atmospheric,
        void_floor_fraction=params.void_floor_fraction,
        active_void_fraction=params.active_void_fraction,
        topology_density_fraction=params.topology_density_fraction,
        resolved_density_fraction=params.resolved_density_fraction,
    )
    v_front_receiver = np.zeros_like(v_tracer_supported)
    if v_front_receiver.size > 1:
        v_front_receiver[1:] |= v_tracer_supported[:-1]
        v_front_receiver[:-1] |= v_tracer_supported[1:]
    v_front_receiver &= (
        v_raw_geometric
        > params.vertical_front_void_fraction * params.vertical_area
    )
    if (
        geometric_mouth_area > 0.0
        and v_raw_geometric[0]
        > params.vertical_front_void_fraction * params.vertical_area
    ):
        v_front_receiver[0] = True

    v_top_connected = np.zeros_like(v_tracer_supported)
    for index in range(v_top_connected.size - 1, -1, -1):
        if (
            v_raw_geometric[index]
            <= params.vertical_front_void_fraction * params.vertical_area
        ):
            break
        v_top_connected[index] = True
    if vertical_pocket_front_height is not None:
        cell_bottom = np.arange(v_tracer_supported.size, dtype=float) * dz
        cell_top = cell_bottom + dz
        # A fitted material front is stronger topology evidence than the
        # generic 5%-void receiver threshold used for unlabelled elastic area
        # deficits.  Its cut cell must be acoustically filled as soon as the
        # swept void exceeds the solver's geometric void floor; otherwise the
        # first several Taylor increments contain only positivity-floor mass
        # and are advanced as a 0.2--2 kPa near-vacuum beside a 109 kPa pocket.
        # This marks geometry only.  The conservative component remap below
        # determines the transferred mass from the donor/receiver volumes and
        # subtracts it from the attached horizontal gas component.
        material_swept_receiver = (
            cell_bottom < float(vertical_pocket_front_height)
        ) & (
            v_raw_geometric
            > params.void_floor_fraction * params.vertical_area
        )
        if params.vertical_fitted_front_receivers:
            v_front_receiver |= material_swept_receiver
        # The fitted coordinate already owns interface kinematics.  Admit its
        # intersected cut cell, but never a complete numerical halo ahead of
        # it.  The former ``front + dz`` domain could leave tracer gas in an
        # unswept cell that the liquid-contact projection subsequently filled,
        # creating an enormous false EOS pressure on the next time step.
        bottom_front_domain = cell_bottom < float(vertical_pocket_front_height)
        v_tracer_supported &= bottom_front_domain
        v_front_receiver &= bottom_front_domain
        if (
            vertical_liquid_surface_height is not None
            and float(vertical_pocket_front_height)
            < float(vertical_liquid_surface_height)
        ):
            # The atmospheric headspace and tunnel-origin pocket are separate
            # gas components until the material nose catches the bulk liquid
            # surface.  A transient area deficit scattered through the upper
            # liquid slug is not a pneumatic short circuit between them.
            cell_top = (
                np.arange(v_top_connected.size, dtype=float) + 1.0
            ) * dz
            v_top_connected &= (
                cell_top > float(vertical_liquid_surface_height)
            )

    # When the fitted material front first exposes a finite void slice in the
    # riser, acoustically fill that cut volume from the gas cell attached to
    # the tee before solving the gas Riemann problem.  Otherwise the receiver
    # contains only positivity-floor mass, so a perfectly ordinary geometric
    # hand-off is misread as expansion into a vacuum and launches a spurious
    # near-sonic jet.  The connected-component remap is conservative and is
    # applied only to newly opened, tunnel-connected bottom receivers; the
    # separate atmospheric component above the bulk surface is never touched.
    bottom_front_receiver = (
        v_front_receiver & ~v_tracer_supported & ~v_top_connected
    )
    if (
        geometric_mouth_area > 0.0
        and h_topology[int(junction_index)]
        and np.any(bottom_front_receiver)
    ):
        hm, hj, vm, vj, vc = _equilibrate_vertical_front_receivers(
            hm,
            hj,
            vm,
            vj,
            vc,
            h_raw_topological,
            v_raw_geometric,
            h_mass_supported,
            v_tracer_supported,
            bottom_front_receiver,
            junction_index=int(junction_index),
            horizontal_width=dx,
            vertical_width=dz,
        )
        v_tracer_supported = _mass_backed_gas_topology(
            v_raw_geometric,
            vc,
            full_area=params.vertical_area,
            cell_width=dz,
            rho_reference=params.rho_atmospheric,
            void_floor_fraction=params.void_floor_fraction,
            active_void_fraction=params.active_void_fraction,
            topology_density_fraction=params.topology_density_fraction,
            resolved_density_fraction=params.resolved_density_fraction,
        )
        if vertical_pocket_front_height is not None:
            v_tracer_supported &= bottom_front_domain
    v_topology = (
        v_tracer_supported | v_front_receiver | v_top_connected
    )
    v_raw_topological = np.where(
        v_topology,
        np.maximum(
            v_raw_geometric,
            params.void_floor_fraction * params.vertical_area,
        ),
        0.0,
    )
    v_effective_liquid = params.vertical_area - v_raw_topological
    v_raw, v_ag, v_pg, v_pi, v_dh = _vertical_geometry(
        v_effective_liquid, params
    )
    h_background = params.rho_atmospheric * h_ag * dx
    v_background = params.rho_atmospheric * v_ag * dz
    h_active = h_topology & (
        (h_raw > params.active_void_fraction * params.horizontal_area)
        | (hm > 0.30 * h_background)
    )
    v_active = v_topology & (
        (v_raw > params.active_void_fraction * params.vertical_area)
        | (vm > 0.30 * v_background)
        | (vc > 1.0e-10 * np.maximum(v_background, 1.0e-18))
    )
    # A confined bubble inside a bulk liquid column feels the liquid pressure
    # gradient and therefore receives the complementary buoyancy source in the
    # gas momentum equation.  That closure must end for the complete gas
    # component actually connected through active Riemann faces to the open
    # atmospheric lip.  The former local ``A_g >= 0.8 A`` test could call a
    # capillary neck sealed even while the acoustic graph carried flux through
    # it.  It then imposed ``rho_l g`` on an open gas path and created the false
    # post-breakthrough pressure ramp that held the liquid at about 8 s.
    open_atmospheric_gas_component = _top_connected_active_component(v_active)
    v_liquid_pressure_coupled = v_active & ~open_atmospheric_gas_component
    # The gas cannot enter a liquid-full riser cell.  The horizontal pressure
    # first accelerates/displaces that liquid through the coupled liquid node;
    # gas then occupies the resolved base-cell void.  Using the full geometric
    # bore against a void-floor cell injects finite mass into O(1e-4) of the
    # bore area and creates a nonphysical near-vacuum jet with unbounded speed.
    mouth_area = min(
        geometric_mouth_area,
        max(float(v_raw_topological[0]), 0.0),
    )
    if mouth_area > 0.0:
        h_active[int(junction_index)] = True
        v_active[0] = True

    h_faces = np.zeros(hm.size + 1)
    h_faces[0] = h_ag[0] if h_active[0] else 0.0
    h_faces[-1] = h_ag[-1] if h_active[-1] else 0.0
    h_faces[1:-1] = np.where(
        h_active[:-1] & h_active[1:],
        np.minimum(h_ag[:-1], h_ag[1:]),
        0.0,
    )
    v_faces = np.zeros(vm.size + 1)
    v_faces[0] = mouth_area
    v_faces[1:-1] = np.where(
        v_active[:-1] & v_active[1:],
        np.minimum(v_ag[:-1], v_ag[1:]),
        0.0,
    )
    # The atmospheric Riemann face exists only when the top gas control volume
    # belongs to the active topology.  Opening a floor-area face beside an
    # inactive, liquid-filled cell records atmospheric mass flux without
    # applying the equal cell update; near breakthrough that was the entire
    # gas-ledger defect and generated an unbounded rarefaction velocity.
    v_faces[-1] = v_ag[-1] if v_active[-1] else 0.0

    result = _compiled_advance(
        hm / dx,
        hj / dx,
        vm / dz,
        vj / dz,
        vc / dz,
        h_ag,
        h_depth,
        h_pg,
        h_dh,
        v_ag,
        v_pg,
        v_dh,
        h_faces,
        v_faces,
        h_active,
        v_active,
        v_liquid_pressure_coupled,
        float(dx),
        float(dz),
        float(dt),
        int(junction_index),
        float(mouth_area),
        params.rho_l,
        params.gravity,
        params.gas_viscosity,
        params.sound_speed,
        params.rho_atmospheric,
        params.cfl,
        params.limiter_theta,
        params.entropy_fix_fraction,
        params.resolved_density_fraction,
        params.resolved_density_ceiling,
    )
    hm_out = result[0] * dx
    hj_out = result[1] * dx
    vm_out = result[2] * dz
    vj_out = result[3] * dz
    vc_out = result[4] * dz
    # Complete the operator split with one conservative, semi-implicit
    # gas--liquid momentum exchange.  The transport solve above already
    # includes gas wall friction but deliberately excludes interphase drag.
    h_drag_resolved = (
        hm_out
        > params.resolved_density_fraction
        * params.rho_atmospheric
        * h_ag
        * dx
    )
    hj_out, h_q_after_drag = _implicit_interphase_drag_exchange(
        hm_out,
        hj_out,
        h_al,
        h_ql,
        h_ag,
        h_pi,
        h_dh,
        cell_width=dx,
        dt=dt,
        rho_l=params.rho_l,
        gas_viscosity=params.gas_viscosity,
        liquid_holdup_drag_enhancement=(
            params.horizontal_holdup_drag_enhancement
        ),
        active_mask=h_drag_resolved,
    )
    if params.vertical_confined_interface_kinematics:
        v_q_after_drag = v_ql.copy()
    else:
        # The resolved vertical gas is coupled to the liquid by perimeter shear.
        # This equal-and-opposite exchange introduces neither a prescribed flow
        # history nor an external momentum source.
        v_drag_resolved = (
            vm_out
            > params.resolved_density_fraction
            * params.rho_atmospheric
            * v_ag
            * dz
        )
        vj_out, v_q_after_drag = _implicit_interphase_drag_exchange(
            vm_out,
            vj_out,
            v_al,
            v_ql,
            v_ag,
            v_pi,
            v_dh,
            cell_width=dz,
            dt=dt,
            rho_l=params.rho_l,
            gas_viscosity=params.gas_viscosity,
            # A fitted, wall-confined Taylor core transfers buoyancy through
            # pressure/form drag on the liquid-density scale.  Gas-side skin
            # friction alone is O(rho_g/rho_l) too weak and left the complete
            # pre-handoff liquid column in near free fall.  The Davies--Taylor
            # Froude number is the same geometry closure that advances the
            # material nose; it is not a response-history or case-time fit.
            form_drag_diameter=(
                params.vertical_diameter
                if vertical_pocket_front_height is not None
                else None
            ),
            total_area=(
                params.vertical_area
                if vertical_pocket_front_height is not None
                else None
            ),
            confined_bubble_froude=(
                0.345
                if vertical_pocket_front_height is not None
                else None
            ),
            active_mask=v_drag_resolved,
        )
    escaped = float(result[5])
    tracer_final = float(np.sum(hm_out) + np.sum(vc_out) + escaped)
    total_final = float(
        np.sum(hm_out) + np.sum(vm_out) + float(result[6])
    )
    return CoupledGasAdvance(
        horizontal_mass=hm_out,
        horizontal_momentum=hj_out,
        vertical_total_mass=vm_out,
        vertical_momentum=vj_out,
        vertical_tracer_mass=vc_out,
        horizontal_liquid_momentum_increment=h_q_after_drag - h_ql,
        vertical_liquid_momentum_increment=v_q_after_drag - v_ql,
        escaped_tracer_mass=escaped,
        atmospheric_mass_exchange=float(result[6]),
        junction_mass_transfer=float(result[7]),
        total_mass_error=total_final - total_initial,
        tracer_mass_error=tracer_final - tracer_initial,
        substeps=int(result[8]),
        maximum_velocity=float(result[9]),
        junction_mouth_area=float(mouth_area),
        downstream_front_position=downstream_front_position,
        downstream_topology_front_position=(
            downstream_topology_front_position
        ),
        downstream_front_velocity=float(front_velocity),
        downstream_retired_cell_count=int(retired_cell_count),
    )


def advance_lumped_pocket_vertical_network(
    horizontal_inventory: OpenIsothermalGasInventory,
    horizontal_t_void_area: float,
    vertical_total_mass: np.ndarray,
    vertical_momentum: np.ndarray,
    vertical_tracer_mass: np.ndarray,
    vertical_liquid_area: np.ndarray,
    vertical_liquid_discharge: np.ndarray,
    *,
    dz: float,
    dt: float,
    params: CoupledGasParameters,
    vertical_pocket_front_height: float | None = None,
    vertical_liquid_surface_height: float | None = None,
    vertical_branch_confined: bool = False,
) -> LumpedPocketVerticalAdvance:
    """Couple one physical lumped pocket to the resolved vertical FV graph.

    The pocket is represented by one closed-ended horizontal reservoir control
    volume with ``dx = V_pocket / A_g``.  Its axial momentum is identically
    zero, so the side-T Riemann problem sees the correct zero *normal* trace
    from the horizontal pocket.  The riser transport, SSP-RK2 substepping,
    interphase exchange, and atmospheric top boundary are exactly those used
    by :func:`advance_coupled_gas_network`.

    No pressure, transfer-rate, interface history, or result curve is imposed:
    the inventory pressure is ``m R T / V`` and its mass changes only through
    the conservative T-mouth flux returned by the shared solver.
    """

    scalars = (horizontal_t_void_area, dz, dt)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("lumped-pocket geometry and step must be finite")
    if horizontal_t_void_area <= 0.0:
        raise ValueError("horizontal T-cell void area must be positive")
    if horizontal_t_void_area > params.horizontal_area:
        raise ValueError("horizontal T-cell void exceeds the pipe area")
    if dz <= 0.0 or dt < 0.0:
        raise ValueError("vertical spacing must be positive and dt non-negative")
    if horizontal_inventory.mass <= 0.0:
        raise ValueError("the lumped horizontal pocket must contain gas")
    if not math.isclose(
        horizontal_inventory.gas_constant,
        params.gas_constant,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        horizontal_inventory.temperature,
        params.gas_temperature,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "horizontal and vertical gas phases must share R and temperature"
        )

    vm = np.asarray(vertical_total_mass, dtype=float)
    if vm.ndim != 1 or vm.size == 0:
        raise ValueError("vertical gas fields must be non-empty one-dimensional arrays")

    reservoir_width = horizontal_inventory.volume / horizontal_t_void_area
    horizontal_liquid_area = np.asarray(
        [params.horizontal_area - horizontal_t_void_area],
        dtype=float,
    )
    result = advance_coupled_gas_network(
        np.asarray([horizontal_inventory.mass], dtype=float),
        np.zeros(1, dtype=float),
        vertical_total_mass,
        vertical_momentum,
        vertical_tracer_mass,
        horizontal_liquid_area,
        np.zeros(1, dtype=float),
        vertical_liquid_area,
        vertical_liquid_discharge,
        dx=reservoir_width,
        dz=dz,
        dt=dt,
        junction_index=0,
        params=params,
        vertical_pocket_front_height=vertical_pocket_front_height,
        vertical_liquid_surface_height=vertical_liquid_surface_height,
        vertical_branch_confined=vertical_branch_confined,
    )
    return LumpedPocketVerticalAdvance(
        horizontal_inventory=horizontal_inventory.with_state(
            mass=float(result.horizontal_mass[0])
        ),
        vertical_total_mass=result.vertical_total_mass,
        vertical_momentum=result.vertical_momentum,
        vertical_tracer_mass=result.vertical_tracer_mass,
        vertical_liquid_momentum_increment=(
            result.vertical_liquid_momentum_increment
        ),
        escaped_tracer_mass=result.escaped_tracer_mass,
        atmospheric_mass_exchange=result.atmospheric_mass_exchange,
        junction_mass_transfer=result.junction_mass_transfer,
        total_mass_error=result.total_mass_error,
        tracer_mass_error=result.tracer_mass_error,
        substeps=result.substeps,
        maximum_velocity=result.maximum_velocity,
        junction_mouth_area=result.junction_mouth_area,
    )


__all__ = [
    "CoupledGasAdvance",
    "CoupledGasParameters",
    "LumpedPocketVerticalAdvance",
    "LumpedSideTGasAdvance",
    "OpenIsothermalGasInventory",
    "advance_lumped_pocket_vertical_network",
    "advance_lumped_isothermal_side_t",
    "advance_coupled_gas_network",
    "isothermal_ideal_gas_riemann_flux",
    "junction_mouth_area",
]
