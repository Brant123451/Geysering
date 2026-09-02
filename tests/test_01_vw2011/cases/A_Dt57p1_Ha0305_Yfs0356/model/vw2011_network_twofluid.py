# -*- coding: utf-8 -*-
"""
Single-test reproduction of Vasconcelos & Wright (2011), JHE 137(5):543-555.

The selected test is one representative geysering case from the paper's 36-run
matrix:
    D_t = 12.7 mm, H_a0 = 0.610 m, Y_fs0 = 0.356 m.

The apparatus and initial state follow the parsed JHE paper values:
  * horizontal pipe diameter D = 0.094 m throughout;
  * upstream air chamber length 0.546 m, initially pressurized air;
  * middle water-filled pipe length 2.970 m;
  * ventilation tower at x = 3.516 m, length L = 0.610 m, open top;
  * downstream pipe length 0.490 m, closed end;
  * downstream of the butterfly valve is initially water-filled.

The horizontal pipe uses a shock-fitted pressurised--free-surface front before
the side-T event and a circular-pipe Saint--Venant shallow-water liquid branch
after handoff.  Gas mass and momentum remain conservative with an isothermal EOS,
and gas pressure couples through the liquid pressure source; no KH/IKH slip term
is used in the Case-A horizontal liquid equation.  The vertical tower is treated
    as the same 1D model rotated to theta=90 degrees, driven at its base by the tunnel
    junction pressure and open to atmosphere at the top.  When air enters the tower,
    the vertical pipe is displayed as an axial gas/liquid volume-fraction field rather
    than a prescribed bubble shape. The result is obtained by evolving the
    model from the experimental initial condition; no prescribed geyser height,
    release-rate script, or result fitting is imposed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from casea_coupled_gas_network import (
    CoupledGasParameters,
    _mass_backed_gas_topology,
    advance_coupled_gas_network,
    junction_mouth_area,
)
from casea_face_aligned_t import face_aligned_t_indices
from casea_horizontal_liquid_operator import (
    HorizontalLiquidParameters,
    pressure_potential_state,
    pressure_potential_wave_state,
)
from casea_vertical_mouth_twochannel import (
    DirectionalMouthLosses,
    TwoChannelMouthResult,
)
from casea_vertical_mouth_twochannel_integration import (
    HorizontalNodeTopology,
    TwoChannelMouthCouplingPlan,
    TwoLiquidMomentumBoundaryResidual,
    apply_twochannel_horizontal_footprint,
)
from casea_distributed_tnode_inertance import (
    DistributedTNodeGeometry,
    measured_footprint_liquid_inventory,
)
from casea_tnode_mouth_phase_area import (
    resolve_tnode_mouth_phase_areas,
)
from casea_dynamic_void_capacity import (
    compute_dynamic_material_void_capacity,
)
from casea_recoupled_capacity_pressure import (
    flux_inertance_from_characteristic,
    flux_inertance_from_plug,
    project_state_mouth_and_capacity_pressure,
)
from casea_vertical_bottom_riemann import (
    resolve_bottom_mouth_riemann,
    solve_coupled_gross_mouth_characteristics,
)
from casea_vertical_twostream_closures import (
    advance_taylor_sweep_geometry,
    atmospheric_top_liquid_outflow,
    coaxial_core_film_geometry,
)
from casea_vertical_twostream_fv import (
    DirectionalBoundaryFlux,
    PhysicalGasInterphaseState,
    VerticalTwoStreamLiquidProvenanceState,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    advance_vertical_two_stream_fv,
    advance_vertical_two_stream_liquid_provenance,
    conservative_liquid_provenance_topology_transfer,
    implicit_physical_three_body_drag_exchange,
    map_taylor_breakthrough_to_twostream,
)

HERE = Path(__file__).resolve().parent
G = 9.81
RHO_L = 998.0
P_ATM = 101325.0
R_GAS = 287.05
T_GAS = 293.0
MU_L = 1.003e-3
EPS = 1.0e-12
U_FLUX_MAX = 25.0   # safety clamp on advective face velocity [m/s] (prevents empty-cell blow-up)
TENSION_HEAD = 0.05  # max elastic tension of a rarefying full cell [m of head].
                     # Kept SHORT on purpose: any flat clamp in P(alpha) is an
                     # asymmetric spring, and under grid-scale ringing an asymmetric
                     # spring rectifies oscillation into net pumping work (observed
                     # as the pocket over-compressing far past equilibrium).  The
                     # ringing itself is removed by the acoustic bulk viscosity in
                     # run_network; this band only needs to recluse micro-voids.


@dataclass
class NetworkCase:
    """Vasconcelos & Wright (2011) JHE apparatus (Fig. 2 / Table 1), exact values.

    Geometry: upstream air pipe (0.546 m) | butterfly valve | middle pipe (2.97 m) |
    PVC coupling w/ ventilation tower | downstream pipe (0.490 m, closed).  Pipe D=0.094 m.
    Both far ends closed.  The upstream pipe holds the pressurized air pocket; opening the
    valve releases it toward the tower.  Three swept variables: tower diameter D_t, air-phase
    initial pressure head H_a0, initial tower water level Y_fs0."""
    D: float = 0.094            # pipe diameter [m]  (V&W2011)
    Dr: float = 0.0127          # selected ventilation tower diameter D_t [m]
    air_head: float = 0.610     # selected air-phase initial pressure head H_a0 [m]
    init_water_level: float = 0.356  # Y_fs0 above the horizontal-pipe crown [m]
    riser_height: float = 0.610      # ventilation tower length L [m]
    L_up: float = 0.546         # upstream (air) pipe length [m]
    L_mid: float = 2.970        # middle pipe length [m]
    L_down: float = 0.490       # downstream pipe length [m] (closed end)
    gamma_gas: float = 1.4      # isentropic exponent
    # numerics
    ds: float = 0.01
    dz: float = 0.01
    t_end: float = 8.0
    cfl: float = 0.35
    phase_volume_cfl: float = 1.0
    # Numerical elastic speed of the full-water branch.  It only needs to be far
    # above the gravity-wave scale sqrt(gD)~1 m/s so the full reach is effectively
    # incompressible on pocket/tower timescales; the grid-scale ring amplitude of a
    # micro-void scales with rho*a^2 (a=60 turned a 0.1% area flicker into 0.37 m of
    # head and the rectified ringing pumped the tower), so keep it as low as the
    # scale separation allows.
    a_wh: float = 28.0
    nu: float = 0.10             # horizontal Smagorinsky coefficient
    nu_riser: float = 0.10       # riser coefficient (film friction is separate)
    horizontal_churn_friction: float = 0.15
                                    # Darcy-friction increment for cells containing
                                    # both phases; the 4*alpha_g*alpha_l activation
                                    # vanishes in single-phase reaches and supplies
                                    # the unresolved interfacial/churn loss only.
    horizontal_holdup_drag_enhancement: float = 0.0
                                    # companion two-fluid closure coefficient in
                                    # lambda_i=lambda_g*(1+C_h*alpha_l); it acts
                                    # only through conservative gas-liquid slip.
    max_steps: int = 6_000_000
    use_vw_tower_closure: bool = False
    tower_entry_alpha_min: float = 0.02
    junction_loss_coeff: float = 0.75
    glug_loss_coeff: float = 8.0    # extra T-mouth loss while the junction carries gas:
                                    # the counter-current air-up/water-down exchange
                                    # (inverted-bottle glug) is intensely dissipative --
                                    # bubbly churn in the mouth.  Without it the tower
                                    # column rings undamped on the pocket gas spring
                                    # (~2 Hz, +-0.25 L of head) from arrival to the end.
    vertical_taylor_core_area_fraction: float = 0.80
                                    # Cross-section-averaged gas-core area used by
                                    # the side-fed Taylor-bubble shock fit.  The
                                    # remaining 20% is the resolved counter-current
                                    # wall-film corridor; this is the frozen network
                                    # closure, not a Case-A timing or height target.
    vertical_taylor_return_efficiency: float = 1.0
                                    # Fraction of the Taylor-core swept liquid that
                                    # returns through the counter-current wall film.
                                    # The complement remains in the connected upper
                                    # liquid slug as resolved liquid entrainment.  A
                                     # constant value preserves the conservative T
                                     # balance and contains no event-time prescription.
    vertical_ccfl_constant: float = 0.50
                                    # Wallis counter-current-flow limitation
                                    # constant for the open riser.  It acts only
                                    # after material breakthrough and limits the
                                    # downward liquid branch by the resolved
                                    # upward-gas superficial velocity.
    enable_vertical_twostream: bool = True
                                    # Keep the in-development persistent
                                    # two-stream riser available for dedicated
                                    # studies.  Horizontal-front verification
                                    # can retain the established conservative
                                    # one-stream riser so the two changes are
                                    # not conflated.
    allow_horizontal_front_retreat: bool = True
                                    # Physical default: the downstream material
                                    # contact may reverse.  False is retained
                                    # only as a diagnostic reproduction of the
                                    # historical one-way front.
    valve_loss_coeff: float = 2.0   # butterfly-valve disc stays in the flow when open
                                    # (V&W2011 Fig.2): local K on the through-flow at
                                    # x=L_up damps the release piston mode physically
    valve_open_time: float = 0.25   # hand-turned butterfly valve opening time [s].
                                    # The release is NOT a 0-ms diaphragm burst: the
                                    # loss K ~ K_open/theta(t)^2 throttles the first
                                    # ~0.2 s so the column accelerates as a body
                                    # instead of the contact cell alone cavitating.

    @property
    def L_tunnel(self) -> float:
        return self.L_up + self.L_mid + self.L_down

    @property
    def x_riser(self) -> float:
        return self.L_up + self.L_mid          # tower at the coupling after the middle pipe

    @property
    def x_transducer(self) -> float:
        return self.L_up + 1.07                # V&W2011: transducer 1.07 m downstream of the valve

    @property
    def V_air(self) -> float:
        return self.L_up * self.A              # air pocket = upstream pipe volume

    @property
    def A(self) -> float:
        return 0.25 * math.pi * self.D * self.D

    @property
    def Ar(self) -> float:
        return 0.25 * math.pi * self.Dr * self.Dr


SELECTED_TAG = "vw2011_single_Dt12p7_Ha0610_Yfs0356"
SELECTED_LABEL = (
    "V&W(2011) JHE selected test: Dt=12.7 mm (Dt/D=0.135), "
    "Ha0=0.610 m, Yfs0=0.356 m"
)

FORMULATION_NOTE = (
    "有限体积一维双流体版：水平管和竖管的气相均按质量和动量守恒推进，"
    "气相密度由水气共压力和理想气体状态方程计算；液相用面积/动量方程推进。"
    "竖管顶部为大气开口，底部由水平管气相压力与竖管底部共压力构成双向节点边界。"
    "HTML 只把求解得到的 alpha_g(z,t) 映射成水气分布图。"
)


def selected_case(t_end: float = 10.0) -> NetworkCase:
    """The only test case retained for this reproduction."""
    return NetworkCase(Dr=0.0127, air_head=0.610, init_water_level=0.356, t_end=t_end)


# --------------------------------------------------------------- apparatus figure
def draw_apparatus_schematic(out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    # scales (schematic, not to scale; dimensions labelled)
    tun_y = 0.0
    tun_h = 0.12              # drawn tunnel bore
    tun_x0, tun_x1 = 0.0, 6.0
    res_w, res_h = 0.9, 1.6
    xr = 1.2                 # riser location (near upstream)
    riser_w = tun_h * 0.55
    riser_h = 2.4

    # reservoir (constant head) at the upstream (left)
    ax.add_patch(Rectangle((tun_x0 - res_w, tun_y - 0.2), res_w, res_h, facecolor="#cfe3ff", edgecolor="0.3"))
    ax.plot([tun_x0 - res_w, tun_x0], [tun_y - 0.2 + res_h, tun_y - 0.2 + res_h], color="#2b7fff", lw=1.0)
    ax.text(tun_x0 - res_w * 0.5, tun_y - 0.2 + res_h + 0.08, "constant-head\nreservoir", ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(tun_x0 - res_w - 0.18, tun_y), xytext=(tun_x0 - res_w - 0.18, tun_y - 0.2 + res_h),
                arrowprops={"arrowstyle": "<->", "color": "0.3"})
    ax.text(tun_x0 - res_w - 0.24, tun_y + 0.6, "$H_{res}$", ha="right", va="center", fontsize=10)

    # horizontal tunnel (capped both ends)
    ax.add_patch(Rectangle((tun_x0, tun_y - tun_h), tun_x1 - tun_x0, tun_h, facecolor="#eef2f7", edgecolor="0.3"))
    # water in tunnel (full)
    ax.add_patch(Rectangle((tun_x0, tun_y - tun_h), xr + 1.6, tun_h, facecolor="#2b7fff", edgecolor="none", alpha=0.9))
    # air capsule (right of valve)
    cap_x0, cap_x1 = 3.6, 5.0
    ax.add_patch(Rectangle((cap_x0, tun_y - tun_h), cap_x1 - cap_x0, tun_h, facecolor="#ffffff", edgecolor="0.5"))
    ax.text(0.5 * (cap_x0 + cap_x1), tun_y - tun_h - 0.16, "air capsule $V_{air}$", ha="center", va="top", fontsize=8)
    # downstream water short bit + capped ends
    ax.add_patch(Rectangle((cap_x1, tun_y - tun_h), tun_x1 - cap_x1, tun_h, facecolor="#2b7fff", edgecolor="none", alpha=0.9))
    ax.plot([tun_x0, tun_x0], [tun_y - tun_h, tun_y], color="0.2", lw=2.5)   # upstream cap
    ax.plot([tun_x1, tun_x1], [tun_y - tun_h, tun_y], color="0.2", lw=2.5)   # downstream cap
    # butterfly valve
    ax.plot([cap_x0, cap_x0], [tun_y - tun_h - 0.03, tun_y + 0.03], color="#b91c1c", lw=2.5)
    ax.text(cap_x0, tun_y + 0.06, "valve", ha="center", va="bottom", fontsize=8, color="#b91c1c")

    # vertical riser (side-T on the crown, near upstream)
    ax.add_patch(Rectangle((xr - 0.5 * riser_w, tun_y), riser_w, riser_h, facecolor="#eef2f7", edgecolor="0.3"))
    ax.add_patch(Rectangle((xr - 0.5 * riser_w, tun_y), riser_w, 0.9, facecolor="#2b7fff", edgecolor="none", alpha=0.9))
    ax.text(xr + 0.12, riser_h * 0.6, "riser\n$D_r$, open top", ha="left", va="center", fontsize=8)
    ax.annotate("", xy=(xr + 0.32, tun_y), xytext=(xr + 0.32, tun_y + 0.9),
                arrowprops={"arrowstyle": "<->", "color": "0.3"})
    ax.text(xr + 0.40, 0.45, "initial\nsurcharge", ha="left", va="center", fontsize=7)

    # arrows: air migrates to riser, water lifted up
    ax.annotate("", xy=(xr + 0.2, tun_y - tun_h * 0.5), xytext=(cap_x0 - 0.1, tun_y - tun_h * 0.5),
                arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 1.4})
    ax.text(2.4, tun_y - tun_h - 0.16, "air migration", ha="center", va="top", fontsize=8, color="0.45")
    ax.annotate("", xy=(xr, riser_h + 0.15), xytext=(xr, riser_h - 0.2),
                arrowprops={"arrowstyle": "->", "color": "#2b7fff", "lw": 1.6})

    # dimension labels
    ax.annotate("", xy=(tun_x0, tun_y - tun_h - 0.35), xytext=(tun_x1, tun_y - tun_h - 0.35),
                arrowprops={"arrowstyle": "<->", "color": "0.3"})
    ax.text(3.0, tun_y - tun_h - 0.42, "tunnel  $D=0.095$ m (capped both ends)", ha="center", va="top", fontsize=9)
    ax.text(xr, riser_h + 0.25, "$\\sim$0.8 m", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(tun_x0 - res_w - 0.7, tun_x1 + 0.4)
    ax.set_ylim(tun_y - tun_h - 0.7, riser_h + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V&W (2011) air-capsule geyser apparatus (after Lewis & Wright 2012, Fig 2.9)\n"
                 "horizontal tunnel + constant-head reservoir + side-T vertical riser + air capsule", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "vw2011_apparatus_schematic.png", dpi=150)
    plt.close(fig)
    return out_dir / "vw2011_apparatus_schematic.png"


def _regions(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def _pocket_mask(Al, A, Mg, ds, gas_min):
    """Cells that belong to a GENUINE gas pocket: they carry resolved void AND a
    gas mass consistent with that void at (a sizeable fraction of) atmospheric
    density.  The crown-exchange transport moves mass WITH volume, so every true
    cavity cell -- including a building nose -- satisfies this by construction.
    A transient rarefaction streak opened in the water column by the release
    hammer / column stretch has void but (almost) NO gas mass: counting such
    cells as pocket (the old void-only test) let the streak JOIN the capsule
    region, which (a) teleported the pocket nose metres ahead of the true
    Benjamin front, (b) diluted the region EOS with phantom volume (recorded
    head sawtooth), and (c) painted pocket pressure over stretched water --
    the positive feedback that kept the whole column ringing."""
    rho_ref = P_ATM / (R_GAS * T_GAS)
    void = np.maximum(A - Al, 0.0)
    return ((void / A) > gas_min) & (Mg > 0.3 * rho_ref * void * ds)


def _connected_pocket_inventory(
    liquid_area,
    gas_mass,
    pocket_mask,
    *,
    index: int,
    full_area: float,
    cell_width: float,
) -> tuple[float, float]:
    """Mass and void volume of the gas component touching one cell.

    Pressure at a junction is set by the gas pocket hydraulically connected to
    that junction, not by disconnected micro-voids elsewhere in the pipe.  The
    one-cell mass halo matches :func:`_pressure`: a material front can deposit
    gas in its receiver before the receiver crosses the pocket-area threshold,
    but the halo contributes no unconnected void volume.
    """

    area = np.asarray(liquid_area, dtype=float)
    mass = np.asarray(gas_mass, dtype=float)
    mask = np.asarray(pocket_mask, dtype=bool)
    if area.ndim != 1 or not (area.shape == mass.shape == mask.shape):
        raise ValueError("connected-pocket arrays must be equal and one-dimensional")
    cell = int(index)
    if not 0 <= cell < area.size:
        raise ValueError("connected-pocket index lies outside the grid")
    if full_area <= 0.0 or cell_width <= 0.0:
        raise ValueError("positive connected-pocket geometry required")
    if not mask[cell]:
        return 0.0, 0.0

    first = cell
    while first > 0 and mask[first - 1]:
        first -= 1
    last = cell + 1
    while last < area.size and mask[last]:
        last += 1
    connected_mass = float(
        np.sum(mass[max(first - 1, 0):min(last + 1, area.size)])
    )
    connected_volume = float(
        np.sum(
            np.maximum(
                full_area - np.clip(area[first:last], 0.0, full_area),
                0.0,
            )
        )
        * cell_width
    )
    return connected_mass, connected_volume


def _pressure(Al, A, Mg, ds, a2, vent_top, p_floor=0.0, gas_min=0.5, tension_head=0.0,
              mass_consistent=False):
    """Unified interfacial pressure: water-hammer in the liquid reach, quasi-uniform
    compressible isothermal gas in trapped pockets, atmospheric in the vented region.

    gas_min: a cell is treated as carrying a free gas pocket when its gas fraction
    alpha_g = 1 - Al/A exceeds gas_min.  Any *connected* gas-bearing region (even a
    thin STRATIFIED crown layer with alpha_g < 0.5) then shares one compressible
    isothermal gas pressure P = m R T / V_gas.  This is essential: a compressed air
    pocket that spreads along the pipe crown must KEEP its pressure (mass and volume
    are conserved), otherwise the pocket's driving overpressure non-physically
    vanishes as soon as it stratifies.  The crown-exchange gas transport moves void
    and mass TOGETHER, so the region's mass and its void volume are always
    collocated and the straightforward sums below are exact.

    mass_consistent: additionally require the cell's own gas mass to back its void
    (see _pocket_mask) -- rejects hammer-rarefaction streaks in the tunnel.

    p_floor: a trapped pocket cannot fall below the surrounding (reservoir-pressurised)
    liquid it is in mechanical contact with -> P = max(m R T / V, p_floor)."""
    n = len(Al)
    alpha = Al / A
    alpha_g = 1.0 - alpha
    Ag = np.maximum(A - Al, 1.0e-4 * A)
    wet = alpha >= 0.5
    if mass_consistent:
        gassy = _pocket_mask(Al, A, Mg, ds, gas_min)
    else:
        gassy = alpha_g > gas_min
    # Elastic water-hammer in essentially-full cells.  The compression branch is
    # exact; on the rarefaction side a LIMITED tension band (down to -tension_head
    # of gauge head) gives a micro-void a smooth restoring stiffness so the
    # neighbours reclose it, WITHOUT the deep-suction pathology of an unbounded
    # tension branch (which turned every rarefied stretch into a suction pump).
    # Below the tension floor the section has separated: it reads its ambient.
    dev = np.where(alpha >= 1.0,
                   alpha - 1.0,
                   -np.minimum(1.0 - alpha, tension_head * RHO_L * G / (RHO_L * a2)))
    P = np.where(~gassy, P_ATM + RHO_L * a2 * dev, P_ATM)
    for (i0, i1) in _regions(gassy):
        connected_vent = vent_top and (i1 >= n)
        if connected_vent:
            P[i0:i1] = P_ATM
            continue
        # one-cell HALO on the mass sum: the crown-exchange front deposits mass+void
        # into the nose cell over several steps BEFORE its void crosses gas_min; that
        # building nose holds up to ~5% of the pocket inventory, and excluding it
        # made the EOS under-read by ~0.4 m of head every few steps (the sawtooth
        # ringing of the recorded pocket head).  The halo cell's own void is <
        # gas_min*A*ds (negligible), so only its MASS is added.
        m = float(np.sum(Mg[max(i0 - 1, 0):min(i1 + 1, n)]))
        v = float(np.sum(Ag[i0:i1]) * ds)
        # Mass-consistency guard: a void region carrying (almost) no gas mass is NOT
        # an air pocket -- it is a numerically rarefied stretch of the water column
        # (transient momentum divergence).  Reading the EOS there returns near-zero
        # absolute pressure (-10 m head), which then violently sucks the surrounding
        # liquid, cavitates the whole reach and destroys the solution.  Such a region
        # relaxes at ~atmospheric pressure and lets the neighbours refill it.
        # (threshold: the mass fills < 20% of the void at atmospheric density)
        if m * R_GAS * T_GAS < 0.2 * P_ATM * max(v, 1e-12):
            P[i0:i1] = P_ATM
            continue
        P[i0:i1] = min(max(m * R_GAS * T_GAS / max(v, 1e-12), p_floor), 12.0 * P_ATM)
    return P, wet, Ag


_SEG_TABLE = None


def _depth_frac(alpha_l):
    """Water depth fraction h/D at liquid area fraction alpha_l in a circular pipe
    (stratified layer at the invert, gas at the crown)."""
    global _SEG_TABLE
    if _SEG_TABLE is None:
        hs = np.linspace(0.0, 1.0, 2001)
        width = 2.0 * np.sqrt(np.clip(hs * (1.0 - hs), 0.0, None))
        dh = hs[1] - hs[0]
        area = np.zeros_like(hs)
        area[1:] = np.cumsum(0.5 * (width[:-1] + width[1:]) * dh)
        alpha = area / area[-1]
        # Hydrostatic pressure moment I1 = integral_0^h (h-y)b(y)dy;
        # dI1/dh = A(h).  Values are dimensionless for D=1.
        i1 = np.zeros_like(hs)
        i1[1:] = np.cumsum(0.5 * (area[:-1] + area[1:]) * dh)
        _SEG_TABLE = (alpha, hs, i1)
    return np.interp(alpha_l, _SEG_TABLE[0], _SEG_TABLE[1])


def _section_hydrostatic(alpha_l, diameter):
    """Circular-section depth and hydrostatic moment for 0<=alpha_l<=1.

    Returns h [m] and I1 [m^3], where g*I1 is the Saint-Venant pressure
    contribution to the discharge-momentum flux.
    """
    global _SEG_TABLE
    alpha = np.clip(np.asarray(alpha_l, dtype=float), 0.0, 1.0)
    if _SEG_TABLE is None:
        _depth_frac(alpha)
    h_frac = np.interp(alpha, _SEG_TABLE[0], _SEG_TABLE[1])
    i1_frac = np.interp(alpha, _SEG_TABLE[0], _SEG_TABLE[2])
    return diameter * h_frac, diameter ** 3 * i1_frac


def _tpa_pressure_flux_and_celerity(area_l, area_full, diameter, wave_speed):
    """Two-component pressure law for free-surface/pressurised pipe flow.

    For A <= Af the momentum pressure flux is g*I1(A), i.e. the exact
    Saint-Venant hydrostatic moment of a circular section.  For A > Af it is
    continued with the elastic water-hammer law.  This is the conservative
    pressure split used by the Case-B wet/dry horizontal solver.
    """
    area = np.maximum(np.asarray(area_l, dtype=float), 0.0)
    alpha = np.clip(area / area_full, 0.0, 1.0)
    depth, i1 = _section_hydrostatic(alpha, diameter)
    pressure_flux = G * i1

    _, i1_full = _section_hydrostatic(np.asarray([1.0]), diameter)
    psi_full = G * float(i1_full[0])
    overfull = area > area_full
    pressure_flux = np.where(
        overfull,
        psi_full + 0.5 * wave_speed * wave_speed
        * (area * area - area_full * area_full) / area_full,
        pressure_flux,
    )

    top_width = 2.0 * np.sqrt(
        np.maximum(depth * (diameter - depth), 0.0)
    )
    celerity = np.zeros_like(area)
    free = (area > 1.0e-12 * area_full) & (~overfull)
    celerity[free] = np.sqrt(
        G * area[free] / np.maximum(top_width[free], 1.0e-10 * diameter)
    )
    # The Preissmann/TPA transition caps the vanishing-top-width singularity
    # at the prescribed elastic wave speed.
    celerity = np.minimum(celerity, wave_speed)
    celerity[overfull] = wave_speed * np.sqrt(
        np.maximum(area[overfull] / area_full, 1.0)
    )
    at_full = np.abs(area - area_full) <= 1.0e-10 * area_full
    celerity[at_full] = wave_speed
    return pressure_flux, celerity


def _minmod3(a, b, c):
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    return np.where(
        same_sign,
        np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))),
        0.0,
    )


def _tpa_hllc_flux(area_l, discharge_l, area_r, discharge_r,
                   area_full, diameter, wave_speed):
    """Vectorised low-diffusion HLLC flux for the Case-B TPA pipe equations."""
    dry = 1.0e-9 * area_full
    al = np.maximum(np.asarray(area_l, dtype=float), 0.0)
    ar = np.maximum(np.asarray(area_r, dtype=float), 0.0)
    ql = np.where(al > dry, np.asarray(discharge_l, dtype=float), 0.0)
    qr = np.where(ar > dry, np.asarray(discharge_r, dtype=float), 0.0)
    ul = np.where(al > dry, ql / np.maximum(al, dry), 0.0)
    ur = np.where(ar > dry, qr / np.maximum(ar, dry), 0.0)
    psil, cl = _tpa_pressure_flux_and_celerity(
        al, area_full, diameter, wave_speed
    )
    psir, cr = _tpa_pressure_flux_and_celerity(
        ar, area_full, diameter, wave_speed
    )
    # A full cell touching a free-surface/dry state is the crown-opening
    # transition, not a water-hammer discontinuity.  Use the gravity-wave
    # family on that face; the elastic family is active only across two
    # pressurised cells.  Otherwise a nominally full cell launches the dry
    # front at 2*a_wh instead of the observed O(sqrt(gD)) bore speed.
    free_surface_face = np.minimum(al, ar) < 0.995 * area_full
    # A partially filled section follows its gravity-wave family; the elastic
    # family is reserved for two pressurised states.
    gravity_cap = math.sqrt(G * diameter)
    cl = np.where(free_surface_face, np.minimum(cl, gravity_cap), cl)
    cr = np.where(free_surface_face, np.minimum(cr, gravity_cap), cr)
    f1l = ql
    f1r = qr
    f2l = ql * ul + psil
    f2r = qr * ur + psir

    sl = np.minimum(ul - cl, ur - cr)
    sr = np.maximum(ul + cl, ur + cr)
    left_dry = al <= dry
    right_dry = ar <= dry
    # Einfeldt dry-front estimates recover the finite dam-break fan speed.
    sl = np.where(left_dry & (~right_dry), ur - 2.0 * cr, sl)
    sr = np.where(left_dry & (~right_dry), ur + cr, sr)
    sl = np.where(right_dry & (~left_dry), ul - cl, sl)
    sr = np.where(right_dry & (~left_dry), ul + 2.0 * cl, sr)

    denom = al * (sl - ul) - ar * (sr - ur)
    safe_denom = np.where(np.abs(denom) > 1.0e-14, denom, -1.0e-14)
    sm = (
        psir - psil
        + al * ul * (sl - ul)
        - ar * ur * (sr - ur)
    ) / safe_denom
    sm = np.clip(sm, sl, sr)

    safe_l = np.where(np.abs(sl - sm) > 1.0e-14, sl - sm, -1.0e-14)
    safe_r = np.where(np.abs(sr - sm) > 1.0e-14, sr - sm, 1.0e-14)
    astar_l = np.maximum(al * (sl - ul) / safe_l, 0.0)
    astar_r = np.maximum(ar * (sr - ur) / safe_r, 0.0)
    qstar_l = astar_l * sm
    qstar_r = astar_r * sm
    f1star_l = f1l + sl * (astar_l - al)
    f2star_l = f2l + sl * (qstar_l - ql)
    f1star_r = f1r + sr * (astar_r - ar)
    f2star_r = f2r + sr * (qstar_r - qr)

    f1 = np.where(
        sl >= 0.0, f1l,
        np.where(sm >= 0.0, f1star_l,
                 np.where(sr > 0.0, f1star_r, f1r)),
    )
    f2 = np.where(
        sl >= 0.0, f2l,
        np.where(sm >= 0.0, f2star_l,
                 np.where(sr > 0.0, f2star_r, f2r)),
    )
    both_dry = left_dry & right_dry
    f1[both_dry] = 0.0
    f2[both_dry] = 0.0
    return f1, f2


def _tpa_muscl_faces(
    area,
    discharge,
    area_full,
    diameter,
    *,
    first_order=False,
):
    """Positivity-preserving, monotone MUSCL reconstruction.

    Use the standard minmod limiter (theta=1) for the liquid area and
    velocity.  The former generalized-minmod factor of 1.5 was sharper, but
    after the side-T starts returning water it amplified alternating
    cell-scale extrema into a visibly serrated free surface.  Standard
    minmod remains second order in smooth monotone reaches while suppressing
    those non-physical grid modes; it neither filters an archived result nor
    prescribes a wave amplitude.
    """
    n = len(area)
    dry = 1.0e-9 * area_full
    velocity = np.where(
        area > dry, discharge / np.maximum(area, dry), 0.0
    )
    free_surface_cap = 2.0 * math.sqrt(G * diameter)
    velocity_cap = np.where(
        area < 0.995 * area_full,
        free_surface_cap,
        25.0,
    )
    velocity = np.clip(velocity, -velocity_cap, velocity_cap)
    slope_a = np.zeros(n)
    slope_u = np.zeros(n)
    if n > 2 and not first_order:
        slope_a[1:-1] = _minmod3(
            area[1:-1] - area[:-2],
            0.5 * (area[2:] - area[:-2]),
            area[2:] - area[1:-1],
        )
        slope_u[1:-1] = _minmod3(
            velocity[1:-1] - velocity[:-2],
            0.5 * (velocity[2:] - velocity[:-2]),
            velocity[2:] - velocity[1:-1],
        )
    slope_a = np.clip(slope_a, -2.0 * area, 2.0 * area)

    al = np.empty(n + 1)
    ar = np.empty(n + 1)
    ql = np.empty(n + 1)
    qr = np.empty(n + 1)
    al[1:-1] = np.maximum(area[:-1] + 0.5 * slope_a[:-1], 0.0)
    ar[1:-1] = np.maximum(area[1:] - 0.5 * slope_a[1:], 0.0)
    if first_order:
        ql[1:-1] = discharge[:-1]
        qr[1:-1] = discharge[1:]
    else:
        ul = velocity[:-1] + 0.5 * slope_u[:-1]
        ur = velocity[1:] - 0.5 * slope_u[1:]
        ql[1:-1] = al[1:-1] * ul
        qr[1:-1] = ar[1:-1] * ur

    # Reflecting end walls.
    al[0] = area[0]; ar[0] = area[0]
    ql[0] = -discharge[0]; qr[0] = discharge[0]
    al[-1] = area[-1]; ar[-1] = area[-1]
    ql[-1] = discharge[-1]; qr[-1] = -discharge[-1]
    return al, ql, ar, qr


def _decoupled_restoring_coefficient(
    area_l,
    discharge_l,
    gas_mass,
    gas_momentum,
    *,
    area_full,
    diameter,
    cell_width,
):
    """Circular-pipe shallow-water restoring coefficient ``g/T``.

    The gas conserved variables remain arguments for call-site compatibility,
    but neither gas--liquid slip nor a KH term enters the Case-A liquid wave
    speed.  Gas pressure is coupled separately through the pressure source.
    """

    area = np.clip(
        np.asarray(area_l, dtype=float),
        1.0e-6 * area_full,
        0.995 * area_full,
    )
    depth = diameter * _depth_frac(np.clip(area / area_full, 0.0, 1.0))
    top_width = 2.0 * np.sqrt(
        np.maximum(depth * (diameter - depth), 0.0)
    )
    coefficient = G / np.maximum(top_width, 1.0e-10 * diameter)
    if not np.all(np.isfinite(coefficient)):
        raise FloatingPointError(
            "non-finite shallow-water coefficient in T-junction branch"
        )
    return coefficient


@lru_cache(maxsize=16)
def _horizontal_liquid_parameters_cached(
    area_full: float,
    diameter: float,
    wave_speed: float,
    cell_width: float,
    tension_head: float,
) -> HorizontalLiquidParameters:
    """Reuse the immutable horizontal liquid closure across FV stages."""

    return HorizontalLiquidParameters(
        area_full=float(area_full),
        diameter=float(diameter),
        wave_speed=float(wave_speed),
        cell_width=float(cell_width),
        gravity=G,
        rho_liquid=RHO_L,
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure=P_ATM,
        tension_head=float(tension_head),
    )


def _connected_shallow_water_potential_offsets(
    area,
    free_surface_supported,
    params,
    *,
    gas_mass=None,
    cell_width=None,
):
    """Match each connected shallow-water component to elastic traction.

    ``g*I1`` is defined up to a constant in the momentum balance.  A component
    touching an elastic full-pipe state receives one constant gauge offset from
    its boundary traction(s); an isolated all-free-surface component retains
    the natural ``g*I1`` gauge.  The offset changes neither celerity nor wave
    shape and contains no gas velocity or KH contribution.
    """

    area_a = np.asarray(area, dtype=float)
    support = np.asarray(free_surface_supported, dtype=bool)
    if area_a.shape != support.shape:
        raise ValueError("connected-potential area and topology must have equal shape")
    offsets = np.zeros_like(area_a)
    if not np.any(support):
        return offsets

    mass = None
    if gas_mass is not None:
        mass = np.asarray(gas_mass, dtype=float)
        if mass.shape != area_a.shape or not np.all(np.isfinite(mass)):
            raise ValueError(
                "connected-potential gas mass and area must be finite and equal"
            )
        if cell_width is None or float(cell_width) <= 0.0:
            raise ValueError("positive cell width is required with gas mass")
    safe_area = np.maximum(area_a, 1.0e-9 * params.area_full)
    for i0, i1 in _regions(support):
        natural_indices = []
        target_indices = []
        if i0 > 0 and not support[i0 - 1]:
            target = i0 - 1
            if mass is not None:
                while target >= 0 and (
                    support[target] or mass[target] > 1.0e-14
                ):
                    target -= 1
            if target >= 0 and not support[target]:
                natural_indices.append(i0)
                target_indices.append(target)
        if i1 < area_a.size and not support[i1]:
            target = i1
            if mass is not None:
                while target < area_a.size and (
                    support[target] or mass[target] > 1.0e-14
                ):
                    target += 1
            if target < area_a.size and not support[target]:
                natural_indices.append(i1 - 1)
                target_indices.append(target)
        if natural_indices:
            natural = pressure_potential_wave_state(
                safe_area[np.asarray(natural_indices, dtype=int)],
                np.ones(len(natural_indices), dtype=bool),
                params,
            ).potential
            target = pressure_potential_wave_state(
                safe_area[np.asarray(target_indices, dtype=int)],
                np.zeros(len(target_indices), dtype=bool),
                params,
            ).potential
            offsets[i0:i1] = math.fsum(
                float(value) for value in np.asarray(target - natural)
            ) / len(natural_indices)
    return offsets


def _decoupled_liquid_rusanov_flux(
    area,
    discharge,
    gas_mass,
    gas_momentum,
    *,
    area_full,
    diameter,
    wave_speed,
    cell_width,
    tension_head=TENSION_HEAD,
    minimum_stratified_void_fraction=5.0e-4,
    first_order=False,
    force_hll=False,
):
    """Topology-aware MUSCL-HLLC liquid flux.

    The conserved TPA area has two different meanings.  In a cell carrying
    resolved gas mass it is the *physical* wetted area and follows the circular
    free-surface pressure law.  In a gas-free cell it is elastic storage in a
    still liquid-full pipe and follows the water-hammer continuation through
    ``A=Af``.  Treating every ``A<Af`` state as a free surface nucleates gas in
    a rarefaction; using an unshifted elastic flux creates an O(a^2 Af) jump at
    the moving interface.  On a genuine free-surface cell the liquid momentum
    flux is the circular-pipe Saint--Venant flux

    ``Q_l**2/A_l + g*I1(A_l)``.

    The KH/IKH slip term is absent.  Gas pressure remains conservative in the
    gas graph and acts on the liquid once through the regular pressure source.
    """

    al, ql, ar, qr = _tpa_muscl_faces(
        area,
        discharge,
        area_full,
        diameter,
        first_order=first_order,
    )
    # A sub-full liquid area is a genuine free-surface/two-fluid state only
    # when resolved gas mass backs its void.  Otherwise it is the elastic
    # rarefaction side of a still-liquid-full pipe.
    area_cell = np.asarray(area, dtype=float)
    void_cell = np.maximum(area_full - np.clip(area_cell, 0.0, area_full), 0.0)
    rho_atm = P_ATM / (R_GAS * T_GAS)
    gas_supported = _mass_backed_gas_topology(
        void_cell,
        np.asarray(gas_mass, dtype=float),
        full_area=area_full,
        cell_width=cell_width,
        rho_reference=rho_atm,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    # A mass-backed but sub-capillary crown sliver is not a resolved
    # free-surface control volume.  Below the capillary opening the
    # section belongs to the elastic/full-pipe branch; gas mass remains on the
    # conservative gas graph and is transported to an open neighbouring void.
    # The production threshold is derived from capillary length and circular
    # geometry by CoupledGasParameters, rather than fitted to this transient.
    gas_supported &= (
        void_cell
        >= float(minimum_stratified_void_fraction) * float(area_full)
    )
    support_g = np.concatenate(
        ([gas_supported[0]], gas_supported, [gas_supported[-1]])
    )
    pressurised_l = ~support_g[:-1]
    pressurised_r = ~support_g[1:]
    mixed_topology = pressurised_l ^ pressurised_r

    # A slope reconstructed from one pressure law must not cross into the
    # other pressure law.  Use the standard
    # piecewise-constant, well-balanced trace only on mixed-topology faces;
    # second-order MUSCL reconstruction remains active inside each branch.
    area_ghost = np.concatenate(([area_cell[0]], area_cell, [area_cell[-1]]))
    discharge_cell = np.asarray(discharge, dtype=float)
    discharge_ghost = np.concatenate(
        ([-discharge_cell[0]], discharge_cell, [-discharge_cell[-1]])
    )
    al[mixed_topology] = area_ghost[:-1][mixed_topology]
    ar[mixed_topology] = area_ghost[1:][mixed_topology]
    ql[mixed_topology] = discharge_ghost[:-1][mixed_topology]
    qr[mixed_topology] = discharge_ghost[1:][mixed_topology]

    # The pressure potential and characteristic speed are evaluated together:
    # exact ``g*I1`` with ``sqrt(g*A/T)`` on a free surface, and the elastic
    # water-hammer continuation in gas-free full-pipe cells.  Gas conserved
    # variables select the cell topology and component gauge; they do not need
    # a second face reconstruction in the decoupled liquid pressure law.
    dry = 1.0e-9 * area_full
    al_eval = np.maximum(al, dry)
    ar_eval = np.maximum(ar, dry)
    ql_eval = np.where(al > dry, ql, 0.0)
    qr_eval = np.where(ar > dry, qr, 0.0)
    liquid_params = _horizontal_liquid_parameters_cached(
        float(area_full),
        float(diameter),
        float(wave_speed),
        float(cell_width),
        float(tension_head),
    )
    component_offset_cell = _connected_shallow_water_potential_offsets(
        area_cell,
        gas_supported,
        liquid_params,
    )
    component_offset_ghost = np.concatenate(
        (
            [component_offset_cell[0]],
            component_offset_cell,
            [component_offset_cell[-1]],
        )
    )
    pressure_l = pressure_potential_wave_state(
        al_eval,
        ~pressurised_l,
        liquid_params,
        stratified_potential_offset=component_offset_ghost[:-1],
    )
    pressure_r = pressure_potential_wave_state(
        ar_eval,
        ~pressurised_r,
        liquid_params,
        stratified_potential_offset=component_offset_ghost[1:],
    )
    psi_l = pressure_l.potential
    psi_r = pressure_r.potential
    c_l = pressure_l.celerity
    c_r = pressure_r.celerity
    ul = np.where(al > dry, ql_eval / al_eval, 0.0)
    ur = np.where(ar > dry, qr_eval / ar_eval, 0.0)

    sl = np.minimum(ul - c_l, ur - c_r)
    sr = np.maximum(ul + c_l, ur + c_r)
    left_dry = al <= dry
    right_dry = ar <= dry
    sl = np.where(left_dry & (~right_dry), ur - 2.0 * c_r, sl)
    sr = np.where(left_dry & (~right_dry), ur + c_r, sr)
    sl = np.where(right_dry & (~left_dry), ul - c_l, sl)
    sr = np.where(right_dry & (~left_dry), ul + 2.0 * c_l, sr)

    denom = al * (sl - ul) - ar * (sr - ur)
    safe_denom = np.where(
        np.abs(denom) > 1.0e-14, denom, -1.0e-14
    )
    sm = (
        psi_r - psi_l
        + al * ul * (sl - ul)
        - ar * ur * (sr - ur)
    ) / safe_denom
    sm = np.clip(sm, sl, sr)
    safe_l = np.where(
        np.abs(sl - sm) > 1.0e-14, sl - sm, -1.0e-14
    )
    safe_r = np.where(
        np.abs(sr - sm) > 1.0e-14, sr - sm, 1.0e-14
    )
    astar_l = np.maximum(al * (sl - ul) / safe_l, 0.0)
    astar_r = np.maximum(ar * (sr - ur) / safe_r, 0.0)
    qstar_l = astar_l * sm
    qstar_r = astar_r * sm
    f1l = ql
    f1r = qr
    f2l = ql * ul + psi_l
    f2r = qr * ur + psi_r
    f1star_l = f1l + sl * (astar_l - al)
    f2star_l = f2l + sl * (qstar_l - ql)
    f1star_r = f1r + sr * (astar_r - ar)
    f2star_r = f2r + sr * (qstar_r - qr)
    f1 = np.where(
        sl >= 0.0,
        f1l,
        np.where(sm >= 0.0, f1star_l, np.where(sr > 0.0, f1star_r, f1r)),
    )
    f2 = np.where(
        sl >= 0.0,
        f2l,
        np.where(sm >= 0.0, f2star_l, np.where(sr > 0.0, f2star_r, f2r)),
    )
    # HLLC loses its star state when the contact speed coalesces with an outer
    # wave.  Dividing by ``S_R-S_M`` (or ``S_L-S_M``) then produces an
    # arbitrarily large star area and a non-physical liquid impulse.  A
    # positivity-preserving HLL fallback on only those degenerate faces is the
    # standard entropy-stable repair; regular contacts still use HLLC and keep
    # its low diffusion.
    wave_span = np.maximum(sr - sl, 1.0e-12)
    star_scale = np.maximum.reduce((al, ar, np.full_like(al, dry)))
    degenerate_star = (
        (np.abs(sl - sm) <= 1.0e-8 * wave_span)
        | (np.abs(sr - sm) <= 1.0e-8 * wave_span)
        | (~np.isfinite(astar_l))
        | (~np.isfinite(astar_r))
        | (astar_l > 10.0 * star_scale)
        | (astar_r > 10.0 * star_scale)
    )
    hll_denominator = np.where(
        wave_span > 1.0e-14, wave_span, 1.0e-14
    )
    f1_hll_middle = (
        sr * f1l - sl * f1r + sl * sr * (ar - al)
    ) / hll_denominator
    f2_hll_middle = (
        sr * f2l - sl * f2r + sl * sr * (qr - ql)
    ) / hll_denominator
    f1_hll = np.where(sl >= 0.0, f1l, np.where(sr <= 0.0, f1r, f1_hll_middle))
    f2_hll = np.where(sl >= 0.0, f2l, np.where(sr <= 0.0, f2r, f2_hll_middle))
    # Degeneracy is a property of the HLLC star construction, not of the
    # topology label.  Apply the positivity-preserving HLL fallback on every
    # degenerate HLLC face.  Fully stratified faces are overwritten by the
    # companion model's Rusanov flux below, so this changes only an
    # inadmissible elastic/mixed HLLC state and imposes no velocity cap.
    fallback = degenerate_star | bool(force_hll)
    f1 = np.where(fallback, f1_hll, f1)
    f2 = np.where(fallback, f2_hll, f2)
    # The free-surface shallow-water block uses scalar Rusanov dissipation with
    # its gravity-wave characteristic speed.  Retain HLLC only on an elastic or mixed-
    # topology face, where its contact resolution is needed by the fitted
    # pressurised front.
    both_stratified = (~pressurised_l) & (~pressurised_r)
    rusanov_speed = np.maximum(np.abs(ul) + c_l, np.abs(ur) + c_r)
    f1_rusanov = 0.5 * (f1l + f1r) - 0.5 * rusanov_speed * (ar - al)
    f2_rusanov = 0.5 * (f2l + f2r) - 0.5 * rusanov_speed * (qr - ql)
    f1 = np.where(both_stratified, f1_rusanov, f1)
    f2 = np.where(both_stratified, f2_rusanov, f2)
    both_dry = left_dry & right_dry
    f1[both_dry] = 0.0
    f2[both_dry] = 0.0
    return f1, f2, al, ar


def _limit_antidiffusive_liquid_flux(
    area,
    discharge,
    high_volume_flux,
    high_momentum_flux,
    low_volume_flux,
    low_momentum_flux,
    *,
    cell_width,
    dt,
    momentum_limiter_face_mask=None,
):
    """Convex-limit a high-order flux about an invariant-domain HLL update.

    The low-order update supplies the monotone area state.  A Zalesak limiter
    then admits as much of the paired high-order mass/momentum correction as is
    compatible with the local area envelope.  One face coefficient multiplies
    both equations, preserving conservation and avoiding a velocity clip.
    """

    a = np.asarray(area, dtype=float)
    q = np.asarray(discharge, dtype=float)
    fh1 = np.asarray(high_volume_flux, dtype=float)
    fh2 = np.asarray(high_momentum_flux, dtype=float)
    fl1 = np.asarray(low_volume_flux, dtype=float)
    fl2 = np.asarray(low_momentum_flux, dtype=float)
    if (
        a.ndim != 1
        or q.shape != a.shape
        or fh1.shape != (a.size + 1,)
        or fh2.shape != fh1.shape
        or fl1.shape != fh1.shape
        or fl2.shape != fh1.shape
    ):
        raise ValueError("FCT liquid states and face fluxes have incompatible shapes")
    lam = float(dt) / float(cell_width)
    low_area = a - lam * (fl1[1:] - fl1[:-1])

    lower = a.copy()
    upper = a.copy()
    if a.size > 1:
        lower[1:] = np.minimum(lower[1:], a[:-1])
        lower[:-1] = np.minimum(lower[:-1], a[1:])
        upper[1:] = np.maximum(upper[1:], a[:-1])
        upper[:-1] = np.maximum(upper[:-1], a[1:])
    lower = np.minimum(lower, low_area)
    upper = np.maximum(upper, low_area)

    anti = fh1 - fl1
    from_left = lam * anti[:-1]
    from_right = -lam * anti[1:]
    positive = np.maximum(from_left, 0.0) + np.maximum(from_right, 0.0)
    negative = np.minimum(from_left, 0.0) + np.minimum(from_right, 0.0)
    room_up = np.maximum(upper - low_area, 0.0)
    room_down = np.maximum(low_area - lower, 0.0)
    r_up = np.ones_like(a)
    r_down = np.ones_like(a)
    mask_up = positive > 1.0e-30
    mask_down = negative < -1.0e-30
    r_up[mask_up] = np.minimum(1.0, room_up[mask_up] / positive[mask_up])
    r_down[mask_down] = np.minimum(
        1.0,
        room_down[mask_down] / (-negative[mask_down]),
    )

    theta = np.ones_like(fh1)
    for face in range(1, a.size):
        if anti[face] >= 0.0:
            theta[face] = min(r_down[face - 1], r_up[face])
        else:
            theta[face] = min(r_up[face - 1], r_down[face])

    # The area bound alone cannot stop a pressure-law switch from creating an
    # alternating momentum mode at nearly unchanged area.  Apply the identical
    # convex-limiting construction to the conserved discharge, then use the
    # more restrictive face factor for the coupled (A,Q) correction.
    low_q = q - lam * (fl2[1:] - fl2[:-1])
    q_lower = q.copy()
    q_upper = q.copy()
    if q.size > 1:
        q_lower[1:] = np.minimum(q_lower[1:], q[:-1])
        q_lower[:-1] = np.minimum(q_lower[:-1], q[1:])
        q_upper[1:] = np.maximum(q_upper[1:], q[:-1])
        q_upper[:-1] = np.maximum(q_upper[:-1], q[1:])
    q_lower = np.minimum(q_lower, low_q)
    q_upper = np.maximum(q_upper, low_q)
    anti_q = fh2 - fl2
    q_from_left = lam * anti_q[:-1]
    q_from_right = -lam * anti_q[1:]
    q_positive = np.maximum(q_from_left, 0.0) + np.maximum(q_from_right, 0.0)
    q_negative = np.minimum(q_from_left, 0.0) + np.minimum(q_from_right, 0.0)
    q_r_up = np.ones_like(q)
    q_r_down = np.ones_like(q)
    q_mask_up = q_positive > 1.0e-30
    q_mask_down = q_negative < -1.0e-30
    q_r_up[q_mask_up] = np.minimum(
        1.0,
        (q_upper[q_mask_up] - low_q[q_mask_up]) / q_positive[q_mask_up],
    )
    q_r_down[q_mask_down] = np.minimum(
        1.0,
        (low_q[q_mask_down] - q_lower[q_mask_down])
        / (-q_negative[q_mask_down]),
    )
    if momentum_limiter_face_mask is None:
        momentum_mask = np.ones_like(theta, dtype=bool)
    else:
        momentum_mask = np.asarray(momentum_limiter_face_mask, dtype=bool)
        if momentum_mask.shape != theta.shape:
            raise ValueError("momentum FCT face mask has incompatible shape")
    for face in range(1, q.size):
        if not momentum_mask[face]:
            continue
        if anti_q[face] >= 0.0:
            theta_q = min(q_r_down[face - 1], q_r_up[face])
        else:
            theta_q = min(q_r_up[face - 1], q_r_down[face])
        theta[face] = min(theta[face], theta_q)
    return (
        fl1 + theta * (fh1 - fl1),
        fl2 + theta * (fh2 - fl2),
        theta,
    )


def _advance_horizontal_liquid_hyperbolic_ssprk2(
    area,
    discharge,
    gas_mass,
    gas_momentum,
    *,
    area_full: float,
    diameter: float,
    wave_speed: float,
    cell_width: float,
    dt: float,
    valve_face: int,
    valve_transmissivity: float,
    junction_wave_active: bool,
    rho_reference: float,
    coupled_gas_parameters: CoupledGasParameters,
    phase_volume_cfl: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the horizontal hyperbolic liquid operator with SSP-RK2.

    The MUSCL spatial reconstruction is second order and must not be advanced
    by a single Forward-Euler stage after the shock-fit hand-off.  This helper
    recomputes the Riemann fluxes and invariant-domain limiters in both SSP
    stages.  The returned face flux is the time-integrated SSP average, so the
    T-junction diagnostics see the same conservative update as the cells.
    Gas conserved variables are frozen only over this liquid operator split;
    the coupled gas graph is advanced immediately afterwards.
    """

    area0 = np.asarray(area, dtype=float)
    q0 = np.asarray(discharge, dtype=float)
    if area0.shape != q0.shape or area0.ndim != 1:
        raise ValueError("horizontal SSP states must be equal one-dimensional arrays")
    if dt <= 0.0 or cell_width <= 0.0:
        raise ValueError("positive horizontal SSP timestep and spacing required")

    def limited_flux(stage_area, stage_q):
        def candidate(*, first_order, force_hll):
            f1, f2, _, _ = _decoupled_liquid_rusanov_flux(
                stage_area,
                stage_q,
                gas_mass,
                gas_momentum,
                area_full=area_full,
                diameter=diameter,
                wave_speed=wave_speed,
                cell_width=cell_width,
                minimum_stratified_void_fraction=(
                    coupled_gas_parameters.horizontal_capillary_void_fraction
                ),
                first_order=first_order,
                force_hll=force_hll,
            )
            f1[0] = 0.0
            f1[-1] = 0.0
            if valve_transmissivity < 1.0:
                f1[valve_face] *= valve_transmissivity
                f2[valve_face] *= valve_transmissivity
            f1, f2 = _limit_liquid_donor_flux(
                stage_area,
                f1,
                f2,
                cell_width=cell_width,
                dt=dt,
                retained_fraction=0.10,
            )
            if junction_wave_active:
                f1, f2 = _limit_gas_void_closure_flux(
                    stage_area,
                    gas_mass,
                    f1,
                    f2,
                    full_area=area_full,
                    cell_width=cell_width,
                    dt=dt,
                    rho_reference=rho_reference,
                    density_fraction=(
                        coupled_gas_parameters.topology_density_fraction
                    ),
                    density_ceiling=(
                        coupled_gas_parameters.resolved_density_ceiling
                    ),
                    void_floor_fraction=(
                        coupled_gas_parameters.void_floor_fraction
                    ),
                    active_void_fraction=(
                        coupled_gas_parameters.active_void_fraction
                    ),
                    closure_fraction=phase_volume_cfl,
                )
            return f1, f2

        high1, high2 = candidate(first_order=False, force_hll=False)
        low1, low2 = candidate(first_order=True, force_hll=True)
        stage_velocity = np.divide(
            stage_q,
            np.maximum(stage_area, 1.0e-9 * area_full),
        )
        high_area_trial = stage_area - dt / cell_width * (
            high1[1:] - high1[:-1]
        )
        high_q_trial = stage_q - dt / cell_width * (
            high2[1:] - high2[:-1]
        )
        high_velocity_trial = np.divide(
            high_q_trial,
            np.maximum(high_area_trial, 1.0e-9 * area_full),
        )
        bore_speed = 2.0 * math.sqrt(G * diameter)
        unresolved_cell = (
            (np.abs(stage_velocity) > bore_speed)
            | (np.abs(high_velocity_trial) > bore_speed)
        )
        momentum_face_mask = np.zeros(stage_area.size + 1, dtype=bool)
        momentum_face_mask[:-1] |= unresolved_cell
        momentum_face_mask[1:] |= unresolved_cell
        # Include one adjacent face so the low-order momentum flux can remove,
        # rather than merely confine, a newly detected two-cell grid mode.
        momentum_face_mask[:-1] |= momentum_face_mask[1:]
        momentum_face_mask[1:] |= momentum_face_mask[:-1]
        f1, f2, _ = _limit_antidiffusive_liquid_flux(
            stage_area,
            stage_q,
            high1,
            high2,
            low1,
            low2,
            cell_width=cell_width,
            dt=dt,
            momentum_limiter_face_mask=momentum_face_mask,
        )
        return f1, f2

    f1_0, f2_0 = limited_flux(area0, q0)
    area1 = area0 - dt / cell_width * (f1_0[1:] - f1_0[:-1])
    q1 = q0 - dt / cell_width * (f2_0[1:] - f2_0[:-1])
    if np.any(area1 < -1.0e-12) or not (
        np.all(np.isfinite(area1)) and np.all(np.isfinite(q1))
    ):
        raise FloatingPointError("invalid first SSP-RK2 horizontal liquid stage")
    area1 = np.maximum(area1, 0.0)

    f1_1, f2_1 = limited_flux(area1, q1)
    area2 = 0.5 * area0 + 0.5 * (
        area1 - dt / cell_width * (f1_1[1:] - f1_1[:-1])
    )
    q2 = 0.5 * q0 + 0.5 * (
        q1 - dt / cell_width * (f2_1[1:] - f2_1[:-1])
    )
    if np.any(area2 < -1.0e-12) or not (
        np.all(np.isfinite(area2)) and np.all(np.isfinite(q2))
    ):
        raise FloatingPointError("invalid completed SSP-RK2 horizontal liquid stage")
    area2 = np.maximum(area2, 0.0)
    return area2, q2, 0.5 * (f1_0 + f1_1), 0.5 * (f2_0 + f2_1)


def _branch_consistent_external_pressure_gradient(
    pressure,
    stratified_supported,
    *,
    cell_width: float,
):
    """Differentiate only the residual pressure inside one TPA branch.

    The conservative liquid flux already resolves the traction jump at a
    stratified/elastic material face.  ``pressure`` contains only the
    residual source left after that conservative pressure law is subtracted.
    Centred differencing across a change of branch therefore applies the
    material-face traction a second time and launches a grid-scale impulse.

    This routine writes the centred derivative as the average of its two face
    differences and sets a face difference to zero only where the governing
    pressure branch changes.  Ordinary gradients inside either branch are
    unchanged; closed end faces remain zero-gradient.  It is a well-balanced
    source discretisation, not a velocity or wave-amplitude limiter.
    """

    values = np.asarray(pressure, dtype=float)
    topology = np.asarray(stratified_supported, dtype=bool)
    if values.ndim != 1 or topology.shape != values.shape:
        raise ValueError(
            "pressure and topology must be equal one-dimensional arrays"
        )
    if cell_width <= 0.0:
        raise ValueError("positive cell width required")
    if not np.all(np.isfinite(values)):
        raise ValueError("external pressure must be finite")

    face_jump = np.zeros(values.size + 1, dtype=float)
    same_branch = topology[:-1] == topology[1:]
    face_jump[1:-1] = np.where(
        same_branch,
        values[1:] - values[:-1],
        0.0,
    )
    return (face_jump[:-1] + face_jump[1:]) / (2.0 * cell_width)


def _liquid_surface_height(z, dz, Al, A, threshold=0.08):
    """Top of any resolved liquid in the riser, including thin side films."""
    idx = np.where(Al / A > threshold)[0]
    return float(z[idx[-1]] + 0.5 * dz) if idx.size else 0.0


def _column_material_height(Al, alpha_g, A, dz, initial_volume_offset=0.0):
    """Free-surface height from conservative liquid-plus-gas column volume."""
    occupied = np.clip(Al / A + alpha_g, 0.0, 1.0)
    volume = float(np.sum(occupied) * A * dz) - initial_volume_offset
    return float(max(volume / A, 0.0))


def _bulk_material_reaches_riser_outlet(
    liquid_area,
    material_gas_fraction,
    *,
    full_area: float,
    cell_width: float,
    riser_height: float,
    initial_volume_offset: float = 0.0,
) -> bool:
    """Return whether the bulk material free surface has reached the lip.

    A cell-centred phase solver can advect a very small liquid remnant into the
    top cell while the conserved liquid--gas column still ends far below the
    outlet.  Treating that remnant as an open-water donor ejects liquid through
    the boundary before the physical free surface reaches the lip.  The gas
    boundary remains independently open to atmosphere; this predicate gates
    only *liquid* outflow.

    The criterion is geometric and conservative: the liquid area plus the
    tunnel-origin material-gas volume reconstructs the bulk occupied height.
    No clock, case label, observed event time, or comparison-field value enters
    the decision.
    """

    area = np.asarray(liquid_area, dtype=float)
    gas = np.asarray(material_gas_fraction, dtype=float)
    if area.shape != gas.shape or area.ndim != 1:
        raise ValueError("riser liquid and material-gas arrays must match")
    if full_area <= 0.0 or cell_width <= 0.0 or riser_height <= 0.0:
        raise ValueError("positive riser geometry required")
    material_height = _column_material_height(
        area,
        gas,
        full_area,
        cell_width,
        initial_volume_offset,
    )
    tolerance = max(
        128.0 * np.finfo(float).eps * float(riser_height),
        1.0e-12,
    )
    return bool(material_height >= float(riser_height) - tolerance)


def _vw_laminar_film_closure(
    diameter: float,
    *,
    rho_l: float = RHO_L,
    rho_g: float | None = None,
    mu_l: float = MU_L,
    gravity: float = G,
) -> tuple[float, float, float, float]:
    """Return the V&W (2011) Taylor-core/film closure for one riser.

    Vasconcelos and Wright balance the liquid displaced by a Davies--Taylor
    bubble against the gravity-driven annular film at the start of the rise,

    ``A_core U_inf = pi g D (rho_l-rho_g) delta^3 / (3 mu_l)``.

    Solving this equation supplies the film thickness instead of prescribing
    an 80% gas-core area for every tower diameter.  The returned values are
    ``(delta, core_area_fraction, film_flow, film_velocity)``.  The velocity is
    negative because the wall film drains toward the horizontal conduit.
    """

    diameter = float(diameter)
    rho_g = (
        P_ATM / (R_GAS * T_GAS)
        if rho_g is None
        else float(rho_g)
    )
    if (
        diameter <= 0.0
        or rho_l <= rho_g
        or mu_l <= 0.0
        or gravity <= 0.0
    ):
        raise ValueError("positive film geometry and density contrast required")
    terminal_speed = 0.345 * math.sqrt(gravity * diameter)

    def residual(thickness: float) -> float:
        core_diameter = max(diameter - 2.0 * thickness, 0.0)
        displaced = 0.25 * math.pi * core_diameter**2 * terminal_speed
        film = (
            math.pi
            * gravity
            * diameter
            * (rho_l - rho_g)
            * thickness**3
            / (3.0 * mu_l)
        )
        return displaced - film

    lower = 0.0
    upper = 0.5 * diameter * (1.0 - 1.0e-12)
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    thickness = 0.5 * (lower + upper)
    full_area = 0.25 * math.pi * diameter**2
    core_area = 0.25 * math.pi * (diameter - 2.0 * thickness) ** 2
    core_fraction = core_area / full_area
    film_area = max(full_area - core_area, EPS)
    film_flow = core_area * terminal_speed
    film_velocity = -film_flow / film_area
    return thickness, core_fraction, film_flow, film_velocity


def _open_riser_annular_film_flux(
    liquid_area: float,
    *,
    diameter: float,
    maximum_flow: float,
    rho_l: float = RHO_L,
    rho_g: float | None = None,
    mu_l: float = MU_L,
    gravity: float = G,
) -> tuple[float, float]:
    """Return the gravity-driven base flux of an open annular film.

    After breakthrough the Taylor nose is no longer confined and its fixed
    material jump must not be imposed at the riser base.  The remaining film
    follows the Nusselt wall balance.  Its thickness is recovered exactly
    from the resolved annular liquid area, and the resulting flow is capped by
    the pre-breakthrough terminal-film discharge.  Thus the boundary is
    continuous at breakthrough and decays cubically as the film thins.
    """

    area = max(float(liquid_area), 0.0)
    diameter = float(diameter)
    rho_g = P_ATM / (R_GAS * T_GAS) if rho_g is None else float(rho_g)
    if (
        diameter <= 0.0
        or maximum_flow < 0.0
        or rho_l <= rho_g
        or mu_l <= 0.0
        or gravity <= 0.0
    ):
        raise ValueError("invalid open-film geometry or material properties")
    full_area = 0.25 * math.pi * diameter**2
    annular_area = min(area, full_area)
    if annular_area <= EPS or maximum_flow <= 0.0:
        return 0.0, 0.0
    radius = 0.5 * diameter
    core_radius = math.sqrt(
        max(radius * radius - annular_area / math.pi, 0.0)
    )
    thickness = radius - core_radius
    flow = -(
        math.pi
        * gravity
        * diameter
        * (rho_l - rho_g)
        * thickness**3
        / (3.0 * mu_l)
    )
    flow = max(flow, -float(maximum_flow))
    velocity = flow / annular_area
    return flow, velocity


def _mass_supported_vertical_gas_mouth(
    liquid_area: float,
    gas_mass: float,
    *,
    full_area: float,
    cell_width: float,
    rho_reference: float,
    density_fraction: float,
    maximum_gas_area_fraction: float,
) -> float:
    """Return the gas-open part of the riser base from its resolved state.

    A temporarily rewetted horizontal T cell cannot close an already open
    vertical gas core.  Conversely, the small atmospheric mass assigned to a
    numerically full liquid cell is not a physical gas path.  The mouth is
    therefore opened only by geometric void carrying resolved gas mass and is
    capped by the Taylor/annular core area used by the gas-network Riemann
    problem.
    """

    if (
        full_area <= 0.0
        or cell_width <= 0.0
        or rho_reference <= 0.0
        or not 0.0 < density_fraction < 1.0
        or not 0.0 < maximum_gas_area_fraction <= 1.0
    ):
        raise ValueError("invalid vertical gas-mouth geometry")
    void = max(
        float(full_area) - min(max(float(liquid_area), 0.0), float(full_area)),
        0.0,
    )
    if void <= EPS:
        return 0.0
    supported_mass = (
        float(density_fraction)
        * float(rho_reference)
        * void
        * float(cell_width)
    )
    if float(gas_mass) <= supported_mass:
        return 0.0
    return min(void, float(maximum_gas_area_fraction) * float(full_area))


def _vertical_two_phase_mouth_pressure(
    *,
    liquid_trace_pressure: float,
    connected_gas_pressure: float,
    gas_mouth_area: float,
    full_area: float,
) -> float:
    """Return the single normal-stress trace at the vertical T mouth.

    Before gas opens the mouth, the lower liquid characteristic supplies the
    pressure at ``z=0``.  Once a mass-supported gas aperture is present, the
    liquid streams and gas share one resolved interface normal stress, namely
    the EOS pressure of the gas component connected through the T.  Continuing
    to use the legacy liquid-column trace in that topology creates a pressure
    jump at one and the same geometric face.

    The returned value is consumed by both the finite T-node inertance (whose
    pressure segment ends at ``z=0``) and the vertical FV operator (whose first
    segment starts at ``z=0``).  Sharing one scalar therefore joins two
    adjacent pressure-work segments; it does not apply either segment twice.
    """

    p_liquid = float(liquid_trace_pressure)
    p_gas = float(connected_gas_pressure)
    area_gas = float(gas_mouth_area)
    area_total = float(full_area)
    if not all(math.isfinite(value) for value in (p_liquid, p_gas, area_gas, area_total)):
        raise ValueError("vertical mouth pressure inputs must be finite")
    if p_liquid <= 0.0 or area_total <= 0.0:
        raise ValueError("vertical liquid pressure and full area must be positive")
    tolerance = 512.0 * math.ulp(max(area_total, 1.0))
    if area_gas < -tolerance or area_gas > area_total + tolerance:
        raise ValueError("vertical gas-mouth area lies outside the riser bore")
    if area_gas > tolerance:
        if p_gas <= 0.0:
            raise ValueError("an open gas mouth requires positive connected pressure")
        return p_gas
    return p_liquid


def _regularize_near_dry_momentum(
    area,
    discharge,
    *,
    full_area: float,
    dry_fraction: float = 1.0e-3,
) -> np.ndarray:
    """Smoothly remove the undefined velocity of a vanishing liquid phase.

    The conserved discharge must tend to zero at least as fast as liquid area.
    A smooth ``A^2/(A^2+A_eps^2)`` factor leaves resolved films unchanged while
    preventing a finite round-off momentum from becoming ``Q/A -> infinity``
    in a numerically dry cell.  This is the standard wet/dry desingularisation;
    it neither changes liquid volume nor imposes a physical velocity cap.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    q = np.asarray(discharge, dtype=float)
    if a.shape != q.shape:
        raise ValueError("near-dry area and discharge must have equal shape")
    if full_area <= 0.0 or not 0.0 < dry_fraction < 1.0:
        raise ValueError("valid full area and dry fraction required")
    scale = dry_fraction * float(full_area)
    return q * a * a / (a * a + scale * scale)


def _riser_liquid_friction_rate(
    area,
    discharge,
    annular_film_mask,
    *,
    full_area: float,
    diameter: float,
    film_thickness: float,
    kinematic_viscosity: float = MU_L / RHO_L,
) -> np.ndarray:
    """Return the local dissipative rate for bulk and annular liquid.

    Full/slug cells retain the ordinary pipe-wall expression.  A Taylor-bubble
    wall film is treated as annular internal flow with hydraulic diameter
    ``D_h = 2 delta``.  Its Darcy factor is ``64/Re`` while laminar and is
    blended continuously to the smooth-pipe Blasius law between ``Re=2000``
    and ``Re=4000``.  This matters in Case A: the old Nusselt-laminar terminal
    value corresponds to ``Re_h`` of several thousand and therefore cannot be
    used self-consistently as either a film stress or a base-flow condition.

    Gas--film interfacial drag is still applied, equal and opposite, in the
    conservative two-fluid gas step.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    q = np.asarray(discharge, dtype=float)
    film = np.asarray(annular_film_mask, dtype=bool)
    if a.shape != q.shape or a.shape != film.shape or a.ndim != 1:
        raise ValueError("riser friction arrays must have equal one-dimensional shape")
    thickness = np.asarray(film_thickness, dtype=float)
    if thickness.ndim == 0:
        thickness = np.full_like(a, float(thickness))
    elif thickness.shape != a.shape:
        raise ValueError(
            "film thickness must be scalar or match the riser state"
        )
    if (
        full_area <= 0.0
        or diameter <= 0.0
        or kinematic_viscosity <= 0.0
        or np.any(~np.isfinite(thickness))
        or np.any(thickness[film] <= 0.0)
    ):
        raise ValueError("positive riser friction geometry and viscosity required")
    velocity = np.divide(
        q,
        np.maximum(a, 1.0e-3 * full_area),
        out=np.zeros_like(q),
        where=a > 0.0,
    )
    bulk_rate = (
        32.0 * kinematic_viscosity / (diameter * diameter)
        + 0.025 / (2.0 * diameter) * np.abs(velocity)
    )
    film_hydraulic_diameter = np.maximum(
        2.0 * thickness,
        1.0e-12 * diameter,
    )
    reynolds = (
        np.abs(velocity) * film_hydraulic_diameter / kinematic_viscosity
    )
    laminar_rate = (
        32.0 * kinematic_viscosity / (film_hydraulic_diameter**2)
    )
    turbulent_factor = 0.3164 / np.maximum(reynolds, 1.0) ** 0.25
    turbulent_rate = (
        turbulent_factor
        * np.abs(velocity)
        / (2.0 * film_hydraulic_diameter)
    )
    blend = np.clip((reynolds - 2000.0) / 2000.0, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    film_rate = (1.0 - blend) * laminar_rate + blend * turbulent_rate
    return np.where(film, film_rate, bulk_rate)


def _top_connected_atmospheric_gas_mask(
    gas_area,
    tracer_mass,
    *,
    cell_length: float,
    reference_density: float,
    dry_area_tolerance: float,
    tracer_density_fraction: float = 0.02,
) -> np.ndarray:
    """Identify the pure ambient headspace connected to the open riser lip.

    The tower initially contains an atmospheric gas column above its free
    surface.  It is an open reservoir, not a sealed cell inventory.  A
    Taylor-core parcel injected from the tunnel is distinguished by the
    conservative gas tracer; scanning downward from the lip stops at either a
    liquid seal or the first material-gas cell.  Consequently this mask cannot
    depressurise a confined tunnel-origin pocket.
    """

    area = np.asarray(gas_area, dtype=float)
    tracer = np.maximum(np.asarray(tracer_mass, dtype=float), 0.0)
    if area.shape != tracer.shape or area.ndim != 1:
        raise ValueError("gas area and tracer mass must be equal 1D arrays")
    if (
        cell_length <= 0.0
        or reference_density <= 0.0
        or dry_area_tolerance < 0.0
        or not 0.0 <= tracer_density_fraction < 1.0
    ):
        raise ValueError("invalid open-headspace geometry or density scale")

    mask = np.zeros(area.size, dtype=bool)
    for index in range(area.size - 1, -1, -1):
        local_area = max(float(area[index]), 0.0)
        if local_area <= dry_area_tolerance:
            break
        tracer_threshold = (
            tracer_density_fraction
            * reference_density
            * local_area
            * cell_length
        )
        if float(tracer[index]) > tracer_threshold:
            break
        mask[index] = True
    return mask


def _riser_material_gas_mask(
    gas_area,
    tracer_mass,
    *,
    full_area: float,
    cell_length: float,
    reference_density: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    topology_density_fraction: float,
    resolved_density_fraction: float,
) -> np.ndarray:
    """Return cells whose void is backed by tunnel-origin material gas.

    Total riser gas mass includes the initially atmospheric headspace and the
    small positivity inventory assigned to nearly full cells.  It therefore
    cannot by itself select an EOS pressure or an interphase drag interface.
    The conservative tunnel-gas tracer supplies the missing material topology,
    while the ordinary active-void threshold rejects sub-grid crown slivers.
    """

    return _mass_backed_gas_topology(
        np.maximum(np.asarray(gas_area, dtype=float), 0.0),
        np.maximum(np.asarray(tracer_mass, dtype=float), 0.0),
        full_area=full_area,
        cell_width=cell_length,
        rho_reference=reference_density,
        void_floor_fraction=void_floor_fraction,
        active_void_fraction=active_void_fraction,
        topology_density_fraction=topology_density_fraction,
        resolved_density_fraction=resolved_density_fraction,
    )


def _equilibrate_open_riser_headspace(
    total_mass,
    gas_momentum,
    tracer_mass,
    gas_area,
    *,
    cell_length: float,
    reference_density: float,
    dry_area_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Keep the tracer-free, top-connected riser gas at atmosphere.

    A change of free-surface-cell void volume exchanges ambient gas through the
    open lip on an acoustic time scale.  The projection removes or admits that
    *ambient* mass at the local gas velocity and returns the signed outward mass
    for the global atmospheric ledger.  Material-gas tracer is never changed.
    """

    mass = np.maximum(np.asarray(total_mass, dtype=float), 0.0).copy()
    momentum = np.asarray(gas_momentum, dtype=float).copy()
    tracer = np.maximum(np.asarray(tracer_mass, dtype=float), 0.0)
    area = np.maximum(np.asarray(gas_area, dtype=float), 0.0)
    if not (mass.shape == momentum.shape == tracer.shape == area.shape):
        raise ValueError("open-headspace gas arrays must have equal shape")
    mask = _top_connected_atmospheric_gas_mask(
        area,
        tracer,
        cell_length=cell_length,
        reference_density=reference_density,
        dry_area_tolerance=dry_area_tolerance,
    )
    old_mass = mass.copy()
    target_mass = np.maximum(
        reference_density * area * cell_length,
        tracer,
    )
    velocity = np.divide(
        momentum,
        old_mass,
        out=np.zeros_like(momentum),
        where=old_mass > 1.0e-14,
    )
    mass[mask] = target_mass[mask]
    momentum[mask] = target_mass[mask] * velocity[mask]
    outward_exchange = float(np.sum(old_mass[mask] - mass[mask]))
    return mass, momentum, outward_exchange, mask


def _implicit_smagorinsky_momentum_diffusion(
    area,
    discharge,
    *,
    full_area: float,
    diameter: float,
    spacing: float,
    dt: float,
    coefficient: float,
    molecular_viscosity: float = MU_L / RHO_L,
) -> np.ndarray:
    """Dissipate unresolved shear without prescribing a spatial wave window.

    A one-dimensional section average cannot resolve the recirculating shear
    layer and three-dimensional mixing generated when tower water returns
    through the side-T.  The missing stress is closed with the local mixing-
    length form

    ``nu_t = (C_s D)^2 |du/dx|``.

    The stress is applied as the conservative face flux
    ``A_face (nu + nu_t) du/dx``.  Consequently it changes neither liquid
    volume nor the domain-integrated axial discharge at closed ends, and it is
    exactly zero for uniform translation.  A backward-Euler tridiagonal solve
    removes an explicit viscosity time-step cap; no distance from the T and no
    target waveform enters the closure.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    q = np.asarray(discharge, dtype=float)
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("area and discharge must be equal one-dimensional arrays")
    if (
        full_area <= 0.0
        or diameter <= 0.0
        or spacing <= 0.0
        or dt <= 0.0
        or coefficient < 0.0
        or molecular_viscosity < 0.0
    ):
        raise ValueError("invalid momentum-diffusion parameters")
    if q.size < 2 or coefficient == 0.0 and molecular_viscosity == 0.0:
        return q.copy()

    a_eps = 1.0e-3 * float(full_area)
    a_eff = np.maximum(a, a_eps)
    velocity = q / a_eff
    wet_face = (a[:-1] > a_eps) & (a[1:] > a_eps)
    gradient = (velocity[1:] - velocity[:-1]) / spacing
    velocity_jump = np.abs(velocity[1:] - velocity[:-1])
    unresolved_jump = np.maximum(
        velocity_jump - 2.0 * math.sqrt(G * diameter),
        0.0,
    )
    # A topology-pressure impulse first appears as a two-cell velocity jump.
    # Add the conservative local-Lax--Friedrichs viscosity needed to resolve
    # that jump on this mesh.  It vanishes identically for smooth/translated
    # states and redistributes, rather than clips or deletes, axial momentum.
    shock_viscosity = 0.5 * spacing * unresolved_jump
    nu_face = (
        molecular_viscosity
        + (coefficient * diameter) ** 2 * np.abs(gradient)
        + shock_viscosity
    )
    face_area = 2.0 * a[:-1] * a[1:] / np.maximum(a[:-1] + a[1:], a_eps)
    conductance = np.where(
        wet_face,
        face_area * nu_face / (spacing * spacing),
        0.0,
    )

    n = q.size
    lower = np.zeros(n - 1)
    diagonal = np.ones(n)
    upper = np.zeros(n - 1)
    for i in range(n):
        left = conductance[i - 1] if i > 0 else 0.0
        right = conductance[i] if i < n - 1 else 0.0
        diagonal[i] += dt * (left + right) / a_eff[i]
        if i > 0:
            lower[i - 1] = -dt * left / a_eff[i - 1]
        if i < n - 1:
            upper[i] = -dt * right / a_eff[i + 1]

    # Thomas algorithm for the frozen-coefficient backward-Euler operator.
    rhs = q.copy()
    for i in range(1, n):
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    result = np.empty_like(q)
    result[-1] = rhs[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / diagonal[i]
    return result


def _project_single_liquid_column(area, discharge, full_area, dz):
    """Conservative sharp-interface projection for a gas-free riser column.

    Before tunnel gas reaches the side-T, the one-dimensional riser contains
    one connected incompressible liquid column and atmospheric gas above it.
    A cell-centred area update must not nucleate internal axial vacuum pockets.
    This projection retains the exact liquid volume and total axial momentum,
    packs the liquid continuously from the bottom, and leaves at most one
    fractional free-surface cell.  Once tunnel gas reaches the T the caller
    disables this single-column limit and the full two-fluid fields evolve.
    """

    # Elastic compression in a nominally full riser cell is still liquid
    # volume.  Clipping every cell to ``Af`` before packing deleted that
    # volume on each pressure-wave cycle and produced a secular fall of the
    # tower level before gas arrival.  Preserve the complete non-negative
    # finite-volume inventory, then express it as an incompressible column
    # with at most one free-surface cut cell.
    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    q = np.asarray(discharge, dtype=float)
    if a.shape != q.shape:
        raise ValueError("riser area and discharge must have equal shape")
    volume = float(np.sum(a) * dz)
    momentum = float(np.sum(q) * dz)
    capacity = float(a.size * full_area * dz)
    volume = min(max(volume, 0.0), capacity)
    packed = np.zeros_like(a)
    cell_volume = full_area * dz
    complete = min(int(volume // cell_volume), a.size)
    if complete:
        packed[:complete] = full_area
    remainder = volume - complete * cell_volume
    if complete < a.size and remainder > 0.0:
        packed[complete] = remainder / dz
    velocity = momentum / max(volume, EPS)
    return packed, packed * velocity


def _sharpen_unswept_riser_liquid_slug(
    area,
    discharge,
    *,
    material_front_height: float,
    full_area: float,
    dz: float,
    slug_velocity: float | None = None,
):
    """Sharpen only the connected liquid slug ahead of a Taylor nose.

    The swept cells below the fitted nose are left bit-for-bit unchanged, so
    their film holdup remains a finite-volume result.  In the completely
    unswept cells above the nose, however, the liquid is one connected slug
    with one atmospheric free surface.  Packing only that subdomain removes
    advected droplets above the free surface without prescribing a film, a
    wave footprint, or a target height.  Liquid volume and axial momentum in
    the sharpened subdomain are conserved exactly when ``slug_velocity`` is
    omitted.  A fitted Taylor balance may instead supply the coherent slug
    velocity; this represents its resolved gas/interface momentum reaction.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("riser area and discharge arrays must match")
    if full_area <= 0.0 or dz <= 0.0:
        raise ValueError("positive riser geometry required")
    if a.size == 0:
        return a, q

    eps_index = 128.0 * np.finfo(float).eps
    first_unswept = int(
        np.clip(
            math.ceil(float(material_front_height) / float(dz) - eps_index),
            0,
            a.size,
        )
    )
    if first_unswept >= a.size:
        return a, q

    slug_area = a[first_unswept:]
    slug_discharge = q[first_unswept:]
    volume = float(np.sum(slug_area) * dz)
    momentum = float(np.sum(slug_discharge) * dz)
    packed = np.zeros_like(slug_area)
    if volume <= 1.0e-16:
        q[first_unswept:] = 0.0
        return a, q

    cell_volume = float(full_area) * float(dz)
    complete = min(int(volume // cell_volume), packed.size)
    if complete:
        packed[:complete] = float(full_area)
    remainder = volume - complete * cell_volume
    if complete < packed.size:
        if remainder > 0.0:
            packed[complete] = remainder / float(dz)
    elif remainder > 0.0:
        # Retain a legitimate elastic overfill instead of silently deleting it.
        packed[0] += remainder / float(dz)

    velocity = (
        momentum / max(volume, EPS)
        if slug_velocity is None
        else float(slug_velocity)
    )
    if not math.isfinite(velocity):
        raise ValueError("finite upper-slug velocity required")
    a[first_unswept:] = packed
    q[first_unswept:] = packed * velocity
    return a, q


def _collapse_upper_slug_at_taylor_breakthrough(
    area,
    discharge,
    *,
    material_front_height: float,
    full_area: float,
    dz: float,
    mixing_zone_height: float,
):
    """Return the vanishing upper slug below the nose at breakthrough.

    At the instant a confined Taylor nose meets the free surface, no coherent
    liquid slug can remain above that material contact.  Cell-centred advection
    may nevertheless leave one or two upper cells wet.  This one-time topology
    transition transfers their volume into the gravity-dominated junction
    mixing zone below the nose, starting at the bottom.  It conserves liquid
    volume exactly and dissipates the impact momentum locally; no target level
    or post-breakthrough time history is prescribed.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("riser area and discharge arrays must match")
    if full_area <= 0.0 or dz <= 0.0 or mixing_zone_height <= 0.0:
        raise ValueError("positive breakthrough geometry required")
    if a.size == 0:
        return a, q, 0.0

    eps_index = 128.0 * np.finfo(float).eps
    first_unswept = int(
        np.clip(
            math.ceil(float(material_front_height) / float(dz) - eps_index),
            0,
            a.size,
        )
    )
    if first_unswept >= a.size:
        return a, q, 0.0

    returned_volume = float(np.sum(a[first_unswept:]) * float(dz))
    if returned_volume <= 1.0e-16:
        return a, q, 0.0
    a[first_unswept:] = 0.0
    q[first_unswept:] = 0.0

    # The side-T churn zone scales with the measured riser diameter.  Restrict
    # this topology transition to that geometric neighbourhood; if it fills,
    # continue into the nearest lower swept cells rather than recreating an
    # upper slug.
    mixing_cells = max(
        1,
        min(
            first_unswept,
            int(math.ceil(float(mixing_zone_height) / float(dz))),
        ),
    )
    destination_order = list(range(mixing_cells)) + list(
        range(mixing_cells, first_unswept)
    )
    remaining = returned_volume
    for index in destination_order:
        if remaining <= 1.0e-16:
            break
        capacity = max((float(full_area) - float(a[index])) * float(dz), 0.0)
        add = min(capacity, remaining)
        if add <= 0.0:
            continue
        a[index] += add / float(dz)
        # The collapsing slug impacts a counter-current junction pool.  Its
        # coherent upward momentum is dissipated in that unresolved mixing;
        # the existing destination-cell momentum remains unchanged.
        remaining -= add
    if remaining > 1.0e-12 * max(returned_volume, 1.0):
        raise FloatingPointError("breakthrough upper slug exceeds lower capacity")
    return a, q, returned_volume


def _remap_vertical_gas_for_liquid_relocation(
    old_liquid_area,
    new_liquid_area,
    total_gas_mass,
    tracer_gas_mass,
    gas_momentum,
    *,
    full_area: float,
    dz: float,
):
    """Conservatively exchange gas when liquid is relocated between cells.

    A topology event that moves liquid downward opens the same gas volume in
    its source cells that it closes in the receiving cells.  Leaving gas mass
    behind while closing that volume turns an ordinary atmospheric/Taylor
    state into a spurious high-pressure capsule.  This local volume swap moves
    the gas parcel displaced from each liquid receiver into the liquid donor
    void, carrying total mass, tunnel tracer, and axial momentum together.

    The fraction removed from a closing cell equals its closed-void fraction,
    so that cell's gas density, tracer fraction, and velocity are unchanged.
    No gas is created, deleted, vented, or assigned a prescribed pressure.
    """

    old = np.asarray(old_liquid_area, dtype=float)
    new = np.asarray(new_liquid_area, dtype=float)
    mass = np.maximum(np.asarray(total_gas_mass, dtype=float), 0.0).copy()
    tracer = np.maximum(np.asarray(tracer_gas_mass, dtype=float), 0.0).copy()
    momentum = np.asarray(gas_momentum, dtype=float).copy()
    if not (
        old.shape == new.shape == mass.shape == tracer.shape == momentum.shape
    ) or old.ndim != 1:
        raise ValueError("liquid-relocation gas arrays must be equal 1-D fields")
    if full_area <= 0.0 or dz <= 0.0:
        raise ValueError("positive relocation geometry required")
    if np.any(old < -1.0e-14) or np.any(new < -1.0e-14):
        raise ValueError("liquid relocation received negative area")
    if np.any(tracer > mass + 1.0e-14):
        raise ValueError("vertical tracer mass exceeds total gas mass")

    opening = np.maximum(old - new, 0.0) * float(dz)
    closing = np.maximum(new - old, 0.0) * float(dz)
    volume_scale = max(
        float(np.sum(opening)),
        float(np.sum(closing)),
        float(full_area) * float(dz),
        1.0e-30,
    )
    volume_tolerance = 2.0e-12 * volume_scale
    if not math.isclose(
        float(np.sum(opening)),
        float(np.sum(closing)),
        rel_tol=0.0,
        abs_tol=volume_tolerance,
    ):
        raise FloatingPointError("liquid relocation does not conserve volume")
    if float(np.sum(closing)) <= 1.0e-18:
        return mass, tracer, momentum

    source_indices = list(np.flatnonzero(opening > volume_tolerance))
    source_remaining = opening.copy()
    source_cursor = 0
    for receiver in np.flatnonzero(closing > volume_tolerance):
        old_void = max(
            (float(full_area) - min(max(float(old[receiver]), 0.0), float(full_area)))
            * float(dz),
            0.0,
        )
        closed_void = float(closing[receiver])
        if old_void <= 0.0 or closed_void > old_void + 1.0e-14 * volume_scale:
            raise FloatingPointError("liquid receiver closes unavailable gas void")
        fraction = min(max(closed_void / old_void, 0.0), 1.0)
        parcel_mass = float(mass[receiver]) * fraction
        parcel_tracer = float(tracer[receiver]) * fraction
        parcel_momentum = float(momentum[receiver]) * fraction
        mass[receiver] -= parcel_mass
        tracer[receiver] -= parcel_tracer
        momentum[receiver] -= parcel_momentum

        remaining_volume = closed_void
        while remaining_volume > volume_tolerance:
            if source_cursor >= len(source_indices):
                raise FloatingPointError("gas relocation exhausted opened source void")
            source = source_indices[source_cursor]
            accepted_volume = min(
                remaining_volume,
                float(source_remaining[source]),
            )
            parcel_fraction = accepted_volume / closed_void
            mass[source] += parcel_mass * parcel_fraction
            tracer[source] += parcel_tracer * parcel_fraction
            momentum[source] += parcel_momentum * parcel_fraction
            source_remaining[source] -= accepted_volume
            remaining_volume -= accepted_volume
            if source_remaining[source] <= volume_tolerance:
                source_remaining[source] = 0.0
                source_cursor += 1

    if float(np.sum(source_remaining)) > volume_tolerance:
        raise FloatingPointError("gas relocation left unmatched opened void")
    state_scale = max(float(np.sum(mass)), 1.0)
    state_tolerance = 1.0e-13 * state_scale
    if np.any(mass < -state_tolerance) or np.any(tracer < -state_tolerance):
        raise FloatingPointError("gas relocation produced negative inventory")
    if np.any(tracer > mass + state_tolerance):
        raise FloatingPointError("gas relocation produced excess tracer")
    # Clamp round-off only.  In particular, do not zero momentum merely
    # because a parcel is small: that would silently violate the momentum
    # ledger this remap exists to preserve.
    mass = np.maximum(mass, 0.0)
    tracer = np.minimum(np.maximum(tracer, 0.0), mass)
    return mass, tracer, momentum


def _return_isolated_top_bulk_liquid(
    area,
    discharge,
    *,
    material_front_height: float,
    full_area: float,
    dz: float,
    mixing_zone_height: float,
    bulk_fraction: float = 0.50,
    separation_fraction: float = 0.10,
):
    """Return a disconnected post-breakthrough top liquid island downward.

    A genuine bulk outflow is connected through bulk liquid or to the tracked
    material front.  A nearly full top cell separated from both by a thin-film
    or dry gap is instead a cell-centred remnant.  In an open shaft it falls
    back; retaining it creates the stationary top plug and very large velocity
    reported by the old one-stream result.  This local topology test contains
    no case time or comparison target.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("riser area and discharge arrays must match")
    if (
        full_area <= 0.0
        or dz <= 0.0
        or mixing_zone_height <= 0.0
        or not 0.0 < separation_fraction < bulk_fraction < 1.0
    ):
        raise ValueError("invalid top-island return inputs")
    if a.size == 0 or a[-1] < float(bulk_fraction) * float(full_area):
        return a, q, 0.0

    alpha = a / float(full_area)
    island_start = a.size - 1
    while (
        island_start > 0
        and alpha[island_start - 1] > float(separation_fraction)
    ):
        island_start -= 1
    first_unswept = int(
        np.clip(
            math.ceil(
                float(material_front_height) / float(dz)
                - 128.0 * np.finfo(float).eps
            ),
            0,
            a.size,
        )
    )
    if island_start <= first_unswept:
        return a, q, 0.0

    returned_volume = float(np.sum(a[island_start:]) * float(dz))
    if returned_volume <= 1.0e-16:
        return a, q, 0.0
    a[island_start:] = 0.0
    q[island_start:] = 0.0

    mixing_cells = max(
        1,
        min(
            island_start,
            int(math.ceil(float(mixing_zone_height) / float(dz))),
        ),
    )
    destination_order = list(range(mixing_cells)) + list(
        range(mixing_cells, island_start)
    )
    remaining = returned_volume
    for index in destination_order:
        if remaining <= 1.0e-16:
            break
        capacity = max((float(full_area) - float(a[index])) * float(dz), 0.0)
        add = min(capacity, remaining)
        if add <= 0.0:
            continue
        a[index] += add / float(dz)
        remaining -= add
    if remaining > 1.0e-12 * max(returned_volume, 1.0):
        raise FloatingPointError("isolated top liquid exceeds lower capacity")
    return a, q, returned_volume


def _fit_riser_taylor_core(
    area,
    discharge,
    *,
    front_height,
    gas_core_area_fraction,
    full_area,
    dz,
):
    """Fit a Taylor-bubble core and conservatively displace its liquid.

    Cells passed by the fitted nose retain an annular liquid film.  Displaced
    water is packed into available liquid capacity ahead of the nose.  Any
    remainder must leave through the T junction and is returned separately to
    the caller for deposition in the horizontal junction control volume.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape:
        raise ValueError("riser Taylor-core arrays must have equal shape")
    if not 0.0 <= float(gas_core_area_fraction) < 1.0:
        raise ValueError("gas-core area fraction must lie in [0, 1)")
    if a.size == 0 or float(front_height) <= 0.0:
        return a, q, 0.0, 0.0

    cell_bottom = np.arange(a.size, dtype=float) * float(dz)
    axial_fraction = np.clip(
        (float(front_height) - cell_bottom) / float(dz), 0.0, 1.0
    )
    maximum_liquid_area = float(full_area) * (
        1.0 - float(gas_core_area_fraction) * axial_fraction
    )
    excess_area = np.maximum(a - maximum_liquid_area, 0.0)
    displaced_volume = float(np.sum(excess_area) * dz)
    if displaced_volume <= 1.0e-16:
        return a, q, 0.0, 0.0

    displaced_momentum = 0.0
    for index in np.flatnonzero(excess_area > 0.0):
        old_area = float(a[index])
        removed_area = float(excess_area[index])
        fraction = removed_area / max(old_area, EPS)
        displaced_momentum += float(q[index]) * dz * fraction
        a[index] = old_area - removed_area
        q[index] *= max(1.0 - fraction, 0.0)

    transfer_velocity = displaced_momentum / max(displaced_volume, EPS)
    remaining = displaced_volume
    for index in range(a.size):
        if remaining <= 1.0e-16:
            break
        capacity = max(
            (float(maximum_liquid_area[index]) - float(a[index])) * dz,
            0.0,
        )
        add = min(capacity, remaining)
        if add <= 0.0:
            continue
        area_add = add / dz
        a[index] += area_add
        q[index] += area_add * transfer_velocity
        remaining -= add

    returned_to_horizontal = max(remaining, 0.0)
    retained_in_riser = displaced_volume - returned_to_horizontal
    return a, q, retained_in_riser, returned_to_horizontal


def _project_riser_taylor_topology(
    area,
    discharge,
    *,
    front_height,
    gas_core_area_fraction,
    full_area,
    dz,
    film_velocity=None,
    slug_velocity=None,
):
    """Conservatively restore the unresolved Taylor-bubble topology.

    A one-dimensional cross-section cannot resolve the annular film and the
    coherent liquid slug above a Taylor nose at the same time.  Ordinary
    cell-centred advection can therefore scatter the upper slug into small
    liquid fractions all the way to the open tower rim.  At an outflow those
    numerical fragments are then counted as a geyser even though the physical
    free surface is still well below the outlet.

    The drift-flux closure supplies the missing sub-cell topology: cells swept
    by the nose contain the prescribed annular film, the displaced liquid is a
    connected slug immediately above the nose, and the atmospheric region
    above the slug is dry.  This projection preserves the resolved liquid
    volume and axial momentum.  Only volume exceeding the finite tower
    capacity is returned to the caller for conservative deposition at the T.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("riser Taylor-topology arrays must be equal and one-dimensional")
    if (
        not 0.0 <= float(gas_core_area_fraction) < 1.0
        or float(full_area) <= 0.0
        or float(dz) <= 0.0
    ):
        raise ValueError("invalid Taylor-topology geometry")
    if a.size == 0 or float(front_height) <= 0.0:
        return a, q, 0.0

    cell_bottom = np.arange(a.size, dtype=float) * float(dz)
    swept_fraction = np.clip(
        (float(front_height) - cell_bottom) / float(dz), 0.0, 1.0
    )
    swept = swept_fraction > 0.0
    film_area = float(full_area) * (
        1.0 - float(gas_core_area_fraction) * swept_fraction
    )

    resolved_volume = float(np.sum(a) * dz)
    tower_capacity = float(a.size * full_area * dz)
    retained_volume = min(resolved_volume, tower_capacity)
    returned_volume = max(resolved_volume - retained_volume, 0.0)

    target = np.zeros_like(a)
    required_film_volume = float(np.sum(film_area[swept]) * dz)
    if retained_volume <= required_film_volume + 1.0e-16:
        if required_film_volume > 0.0:
            target[swept] = (
                film_area[swept] * retained_volume / required_film_volume
            )
    else:
        target[swept] = film_area[swept]
        remaining = retained_volume - required_film_volume
        # Pack the coherent upper slug from the first unswept cell upward.
        for index in np.flatnonzero(~swept):
            if remaining <= 1.0e-16:
                break
            add = min(float(full_area) * dz, remaining)
            target[index] = add / dz
            remaining -= add
        if remaining > 1.0e-12 * max(retained_volume, 1.0):
            returned_volume += remaining

    if film_velocity is not None or slug_velocity is not None:
        # The shock fit represents an unresolved annular cross-section.  Its
        # phase momentum is closed by Taylor drift kinematics, not by preserving
        # the cell-centred liquid momentum that existed before the topology
        # change.  Gas pressure, interfacial shear and wall stress provide the
        # equal physical reaction, so liquid-phase momentum alone is not an
        # invariant of this projection.
        q_target = np.zeros_like(target)
        q_target[swept] = target[swept] * float(
            0.0 if film_velocity is None else film_velocity
        )
        q_target[~swept] = target[~swept] * float(
            0.0 if slug_velocity is None else slug_velocity
        )
        volume_error = float(
            np.sum(target) * dz + returned_volume - resolved_volume
        )
        if abs(volume_error) > 1.0e-11 * max(resolved_volume, 1.0):
            raise FloatingPointError("Taylor-topology projection lost liquid")
        return target, q_target, returned_volume

    # Legacy/default path: retain the momentum collocated with liquid that
    # already occupies its target volume.  This remains useful for isolated
    # conservative remap tests; the production Taylor closure supplies the
    # physically resolved film and slug velocities above.
    overlap = np.minimum(a, target)
    overlap_fraction = np.divide(
        overlap,
        a,
        out=np.zeros_like(a),
        where=a > EPS,
    )
    q_target = q * overlap_fraction
    removed_volume = float(np.sum(a - overlap) * dz)
    removed_momentum = float(np.sum(q * (1.0 - overlap_fraction)) * dz)
    deficit = np.maximum(target - overlap, 0.0)
    accepted_volume = float(np.sum(deficit) * dz)
    if accepted_volume > 1.0e-16 and removed_volume > 1.0e-16:
        transfer_velocity = removed_momentum / removed_volume
        q_target += deficit * transfer_velocity

    volume_error = float(np.sum(target) * dz + returned_volume - resolved_volume)
    if abs(volume_error) > 1.0e-11 * max(resolved_volume, 1.0):
        raise FloatingPointError("Taylor-topology projection lost liquid")
    return target, q_target, returned_volume


def _vertical_liquid_holdup_head(liquid_fraction, dz):
    """Hydrostatic head above each cell centre from resolved liquid holdup."""

    alpha = np.clip(np.asarray(liquid_fraction, dtype=float), 0.0, 1.0)
    if alpha.ndim != 1 or float(dz) <= 0.0:
        raise ValueError("one-dimensional liquid fraction and positive dz required")
    head_to_top_face = np.cumsum(alpha[::-1])[::-1] * float(dz)
    return np.maximum(head_to_top_face - 0.5 * alpha * float(dz), 0.0)


def _displace_newly_swept_taylor_slice(
    area,
    discharge,
    *,
    old_front_height: float,
    new_front_height: float,
    gas_core_area_fraction: float,
    full_area: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Open only the riser volume newly swept by a Taylor nose.

    A fitted material nose supplies a subcell topology for a one-dimensional
    two-fluid grid, but cells already behind it must not be projected to the
    same fixed gas fraction on every step.  This update removes liquid only
    from the axial slice swept during the current step and conservatively
    places that displaced liquid in the upper slug.  Hence a stationary front
    is exactly idempotent, while a moving front creates the Davies--Taylor core
    once rather than acting as a repeated volume source at the T junction.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("Taylor-slice arrays must be equal and one-dimensional")
    if (
        not 0.0 <= gas_core_area_fraction < 1.0
        or full_area <= 0.0
        or dz <= 0.0
    ):
        raise ValueError("invalid Taylor-slice geometry")
    old_front = max(float(old_front_height), 0.0)
    new_front = max(float(new_front_height), old_front)
    if a.size == 0 or new_front <= old_front + 1.0e-15:
        return a, q, 0.0

    cell_bottom = np.arange(a.size, dtype=float) * dz
    old_fraction = np.clip((old_front - cell_bottom) / dz, 0.0, 1.0)
    new_fraction = np.clip((new_front - cell_bottom) / dz, 0.0, 1.0)
    swept_fraction = np.maximum(new_fraction - old_fraction, 0.0)
    requested_area = gas_core_area_fraction * full_area * swept_fraction
    removable_area = np.minimum(requested_area, a)
    requested_volume = float(np.sum(removable_area) * dz)
    if requested_volume <= 1.0e-16:
        return a, q, 0.0

    # The displaced liquid belongs to the upper slug.  Do not refill any cell
    # already touched by the nose; use only unswept cells above it.
    receiver = new_fraction <= 1.0e-14
    capacity_area = np.where(
        receiver,
        np.maximum(full_area - a, 0.0),
        0.0,
    )
    capacity_volume = float(np.sum(capacity_area) * dz)
    accepted_volume = min(requested_volume, capacity_volume)
    if accepted_volume <= 1.0e-16:
        return a, q, 0.0
    removable_area *= accepted_volume / requested_volume

    removed_momentum = 0.0
    for index in np.flatnonzero(removable_area > 0.0):
        old_area = float(a[index])
        remove = float(removable_area[index])
        fraction = remove / max(old_area, EPS)
        removed_momentum += float(q[index]) * dz * fraction
        a[index] = old_area - remove
        q[index] *= max(1.0 - fraction, 0.0)

    transfer_velocity = removed_momentum / max(accepted_volume, EPS)
    remaining = accepted_volume
    for index in np.flatnonzero(receiver):
        if remaining <= 1.0e-16:
            break
        add = min(max((full_area - float(a[index])) * dz, 0.0), remaining)
        if add <= 0.0:
            continue
        area_add = add / dz
        a[index] += area_add
        q[index] += area_add * transfer_velocity
        remaining -= add
    if remaining > 1.0e-12 * max(accepted_volume, 1.0):
        raise FloatingPointError("new Taylor slice lost displaced liquid")
    return a, q, accepted_volume


def _return_new_taylor_sweep_to_side_t(
    area,
    discharge,
    *,
    old_front_height: float,
    new_front_height: float,
    gas_core_area_fraction: float,
    full_area: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return liquid displaced by a side-fed Taylor nose to the tee.

    The ventilation tower is fed through a side opening rather than from a
    piston below the complete liquid column.  As the fitted gas core sweeps a
    new axial slice, the liquid occupying that core turns downward through the
    annular return corridor and leaves through the same T opening.  This is the
    kinematic volume balance of a confined Taylor bubble.

    Only the newly swept slice is changed.  The function removes its liquid
    volume and collocated vertical momentum and returns both to the caller.
    The caller deposits the volume over the finite T footprint with zero
    *horizontal* momentum, as required by the 90-degree turn.  No wave shape,
    event time, liquid height, or axial impulse is prescribed.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("Taylor-return arrays must be equal and one-dimensional")
    if (
        not 0.0 <= float(gas_core_area_fraction) < 1.0
        or float(full_area) <= 0.0
        or float(dz) <= 0.0
    ):
        raise ValueError("invalid Taylor-return geometry")
    old_front = max(float(old_front_height), 0.0)
    new_front = max(float(new_front_height), old_front)
    if a.size == 0 or new_front <= old_front + 1.0e-15:
        return a, q, 0.0, 0.0

    cell_bottom = np.arange(a.size, dtype=float) * float(dz)
    old_fraction = np.clip(
        (old_front - cell_bottom) / float(dz), 0.0, 1.0
    )
    new_fraction = np.clip(
        (new_front - cell_bottom) / float(dz), 0.0, 1.0
    )
    swept_fraction = np.maximum(new_fraction - old_fraction, 0.0)
    removable_area = np.minimum(
        float(gas_core_area_fraction) * float(full_area) * swept_fraction,
        a,
    )
    returned_volume = float(np.sum(removable_area) * float(dz))
    if returned_volume <= 1.0e-16:
        return a, q, 0.0, 0.0

    returned_momentum = 0.0
    for index in np.flatnonzero(removable_area > 0.0):
        old_area = float(a[index])
        remove = float(removable_area[index])
        fraction = remove / max(old_area, EPS)
        returned_momentum += float(q[index]) * float(dz) * fraction
        a[index] = old_area - remove
        q[index] *= max(1.0 - fraction, 0.0)
    returned_velocity = returned_momentum / returned_volume
    return a, q, returned_volume, returned_velocity


def _return_refilled_taylor_core_to_side_t(
    area,
    discharge,
    *,
    front_height: float,
    gas_core_area_fraction: float,
    full_area: float,
    dz: float,
    maximum_return_volume: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Remove only liquid that numerically refilled an already swept gas core.

    The fitted Taylor coordinate is a moving internal boundary.  Once a slice
    has been swept, the one-stream FV stencil must not refill its gas-core
    portion on the following step.  Any excess above the annular/partial-cell
    capacity is therefore the liquid parcel crossing that fitted boundary and
    returns through the side T.  Removal is proportional across the excess
    field and carries collocated axial momentum.  The caller supplies the
    finite horizontal receiver limit and deposits exactly the returned volume.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("Taylor-refill arrays must be equal and one-dimensional")
    if (
        not 0.0 <= float(gas_core_area_fraction) < 1.0
        or float(full_area) <= 0.0
        or float(dz) <= 0.0
        or float(front_height) < 0.0
    ):
        raise ValueError("invalid Taylor-refill geometry")
    if maximum_return_volume is not None and (
        not math.isfinite(float(maximum_return_volume))
        or float(maximum_return_volume) < 0.0
    ):
        raise ValueError("Taylor-refill volume limit must be finite and non-negative")

    cell_bottom = np.arange(a.size, dtype=float) * float(dz)
    swept_fraction = np.clip(
        (float(front_height) - cell_bottom) / float(dz),
        0.0,
        1.0,
    )
    maximum_liquid_area = float(full_area) * (
        1.0 - float(gas_core_area_fraction) * swept_fraction
    )
    excess_area = np.maximum(a - maximum_liquid_area, 0.0)
    requested_volume = float(np.sum(excess_area) * float(dz))
    accepted_volume = requested_volume
    if maximum_return_volume is not None:
        accepted_volume = min(accepted_volume, float(maximum_return_volume))
    if accepted_volume <= 1.0e-16:
        return a, q, 0.0, 0.0
    excess_area *= accepted_volume / max(requested_volume, EPS)

    returned_momentum = 0.0
    for index in np.flatnonzero(excess_area > 0.0):
        old_area = float(a[index])
        remove = float(excess_area[index])
        fraction = remove / max(old_area, EPS)
        returned_momentum += float(q[index]) * float(dz) * fraction
        a[index] = old_area - remove
        q[index] *= max(1.0 - fraction, 0.0)
    returned_velocity = returned_momentum / accepted_volume
    return a, q, accepted_volume, returned_velocity


def _restore_refilled_taylor_core_to_unswept_slug(
    area,
    discharge,
    *,
    front_height: float,
    gas_core_area_fraction: float,
    full_area: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Undo cross-front numerical refill while the Taylor nose is confined.

    Before breakthrough the swept gas core and the unswept upper liquid slug
    are separated by a fitted material interface.  A cell-centred one-stream
    stencil can diffuse liquid backward across that interface on later steps.
    Move only this excess back into the immediately overlying unswept slug,
    carrying its axial momentum.  This is an idempotent conservative shock-fit
    remap: it neither changes the riser liquid inventory nor adds another T-mouth
    return on top of the already accepted Taylor displacement flow.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("Taylor-refill arrays must be equal and one-dimensional")
    if (
        not 0.0 <= float(gas_core_area_fraction) < 1.0
        or float(full_area) <= 0.0
        or float(dz) <= 0.0
        or float(front_height) < 0.0
    ):
        raise ValueError("invalid confined Taylor-refill geometry")

    cell_bottom = np.arange(a.size, dtype=float) * float(dz)
    swept_fraction = np.clip(
        (float(front_height) - cell_bottom) / float(dz),
        0.0,
        1.0,
    )
    maximum_liquid_area = float(full_area) * (
        1.0 - float(gas_core_area_fraction) * swept_fraction
    )
    excess_area = np.maximum(a - maximum_liquid_area, 0.0)
    moved_volume = float(np.sum(excess_area) * float(dz))
    if moved_volume <= 1.0e-16:
        return a, q, 0.0

    receiver = swept_fraction <= 1.0e-14
    capacity_area = np.where(
        receiver,
        np.maximum(float(full_area) - a, 0.0),
        0.0,
    )
    if float(np.sum(capacity_area) * float(dz)) + 1.0e-15 < moved_volume:
        raise FloatingPointError(
            "confined Taylor refill exceeds the unswept slug capacity"
        )

    carried_momentum = 0.0
    for index in np.flatnonzero(excess_area > 0.0):
        old_area = float(a[index])
        remove = float(excess_area[index])
        fraction = remove / max(old_area, EPS)
        carried_momentum += float(q[index]) * float(dz) * fraction
        a[index] = old_area - remove
        q[index] *= max(1.0 - fraction, 0.0)
    transfer_velocity = carried_momentum / moved_volume
    remaining = moved_volume
    for index in np.flatnonzero(receiver):
        if remaining <= 1.0e-16:
            break
        accepted = min(
            max((float(full_area) - float(a[index])) * float(dz), 0.0),
            remaining,
        )
        if accepted <= 0.0:
            continue
        area_add = accepted / float(dz)
        a[index] += area_add
        q[index] += area_add * transfer_velocity
        remaining -= accepted
    if remaining > 1.0e-12 * max(moved_volume, 1.0):
        raise FloatingPointError("confined Taylor refill remap lost liquid")
    return a, q, moved_volume


def _relax_elastic_riser_storage_for_twostream_handoff(
    area,
    discharge,
    *,
    full_area: float,
    dz: float,
    area_tolerance: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Release one-stream elastic storage at the incompressible handoff.

    The pre-breakthrough water-hammer branch stores compression as
    ``A_l > A_r``.  The directional post-breakthrough branch instead carries a
    true phase partition and cannot own that elastic coordinate.  At the single
    topology event, release the small excess into available volume of the same
    contiguous liquid body (or its immediately adjacent top free-surface cell).
    Each transferred parcel carries its donor velocity, so reference-volume
    and axial momentum are conserved.  No liquid crosses a dry separation and
    this operation is never repeated after the owner changes.
    """

    a = np.asarray(area, dtype=float).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("riser handoff arrays must be equal and one-dimensional")
    if full_area <= 0.0 or dz <= 0.0 or area_tolerance < 0.0:
        raise ValueError("valid riser handoff geometry and tolerance required")
    if np.any(a < -area_tolerance) or not (
        np.all(np.isfinite(a)) and np.all(np.isfinite(q))
    ):
        raise ValueError("riser handoff state must be finite and non-negative")
    a = np.maximum(a, 0.0)
    initial_area = float(np.sum(a))
    initial_discharge = float(np.sum(q))
    relaxed_volume = 0.0
    wet_tolerance = max(float(area_tolerance), 1.0e-12 * float(full_area))

    for start, stop in _regions(a > wet_tolerance):
        donors = [
            index
            for index in range(start, stop)
            if a[index] > float(full_area) + float(area_tolerance)
        ]
        if not donors:
            continue
        receivers = [
            index
            for index in range(start, stop)
            if a[index] < float(full_area) - float(area_tolerance)
        ]
        # A compressed continuous column may be full up to its free surface.
        # Admit only the immediately adjacent upper cell; never bridge a dry
        # gas gap to a detached liquid island.
        if stop < a.size:
            receivers.append(stop)
        component_excess = sum(
            max(float(a[index]) - float(full_area), 0.0)
            for index in donors
        )
        component_capacity = sum(
            max(float(full_area) - float(a[index]), 0.0)
            for index in receivers
        )
        if component_capacity + float(area_tolerance) < component_excess:
            raise FloatingPointError(
                "elastic riser storage has no local handoff receiving volume"
            )

        for donor in donors:
            excess_area = max(float(a[donor]) - float(full_area), 0.0)
            if excess_area <= float(area_tolerance):
                continue
            donor_velocity = float(q[donor]) / max(float(a[donor]), EPS)
            a[donor] -= excess_area
            q[donor] -= excess_area * donor_velocity
            remaining = excess_area
            for receiver in sorted(receivers, key=lambda index: abs(index - donor)):
                capacity = max(float(full_area) - float(a[receiver]), 0.0)
                accepted = min(capacity, remaining)
                if accepted <= 0.0:
                    continue
                a[receiver] += accepted
                q[receiver] += accepted * donor_velocity
                remaining -= accepted
                if remaining <= float(area_tolerance):
                    remaining = 0.0
                    break
            if remaining > float(area_tolerance):
                raise FloatingPointError(
                    "elastic riser storage was not fully released at handoff"
                )
            relaxed_volume += excess_area * float(dz)

    conservation_scale = max(initial_area, float(full_area), 1.0)
    if not math.isclose(
        float(np.sum(a)),
        initial_area,
        rel_tol=0.0,
        abs_tol=2.0e-13 * conservation_scale,
    ):
        raise FloatingPointError("riser handoff lost liquid reference volume")
    momentum_scale = max(abs(initial_discharge), float(full_area), 1.0)
    if not math.isclose(
        float(np.sum(q)),
        initial_discharge,
        rel_tol=0.0,
        abs_tol=2.0e-13 * momentum_scale,
    ):
        raise FloatingPointError("riser handoff lost axial liquid momentum")
    if np.max(a, initial=0.0) > float(full_area) + float(area_tolerance):
        raise FloatingPointError("riser handoff left unresolved elastic overfill")
    return a, q, relaxed_volume


def _advance_riser_taylor_front(
    front_height: float,
    *,
    free_surface_height: float,
    liquid_superficial_velocity: float,
    diameter: float,
    riser_height: float,
    dt: float,
    already_vented: bool = False,
) -> tuple[float, float, bool]:
    """Advance a confined Taylor nose only through the liquid column.

    A Taylor nose ceases to be a confined material front when it catches the
    bulk free surface.  The fitted coordinate then records the highest slice
    actually swept by the nose; it must not move downward with the equivalent
    column height because that would erase already opened gas-core cells in a
    single step.  Gas mass and momentum still vent through the conservative
    gas-network equations, while the remaining liquid drains as a wall film.
    """

    if dt <= 0.0 or diameter <= 0.0 or riser_height <= 0.0:
        raise ValueError("positive Taylor-front geometry and timestep required")
    old_front = float(np.clip(front_height, 0.0, riser_height))
    surface = float(np.clip(free_surface_height, 0.0, riser_height))
    if surface <= 0.0:
        # A falling/vanishing liquid inventory cannot "unsweep" cells already
        # traversed by the material Taylor nose.  Losing this height exactly
        # at breakthrough erased the geometric film corridor and handed the
        # persistent T node a fictitious full-bore descending liquid trace.
        return old_front, 0.0, bool(already_vented or old_front > 0.0)

    if already_vented:
        # The swept material domain is irreversible until the tunnel-origin
        # tracer has left the riser.  Once the nose has met the free surface,
        # neither a falling nor a gas-filled rising occupied height is a new
        # confined Taylor sweep.
        new_front = old_front
        return new_front, (new_front - old_front) / dt, True

    if surface <= old_front:
        # The free surface has descended onto the already swept material
        # domain.  This is breakthrough, not a retreat of the gas material
        # coordinate.
        return old_front, 0.0, True

    drift_velocity = 0.345 * math.sqrt(G * diameter)
    proposed = old_front + max(
        0.0, float(liquid_superficial_velocity) + drift_velocity
    ) * dt
    reached_surface = bool(proposed >= surface)
    new_front = min(proposed, surface, riser_height)
    return new_front, (new_front - old_front) / dt, reached_surface


def _restore_riser_annular_film(
    area,
    discharge,
    tracer_mass,
    *,
    minimum_film_fraction,
    full_area,
    dz,
    rho_reference,
    front_height=None,
):
    """Return liquid down the wall around resolved Taylor-bubble cells.

    A cross-sectionally averaged two-fluid cell cannot resolve the thin wall
    film geometrically.  When the gas core is present, this conservative
    closure maintains the prescribed minimum liquid-film area by moving water
    from the top of the overlying column down into deficient core cells.  The
    same parcel momentum is transferred with the volume.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    tracer = np.maximum(np.asarray(tracer_mass, dtype=float), 0.0)
    if not (a.shape == q.shape == tracer.shape):
        raise ValueError("riser film arrays must have equal shape")
    film_area = float(minimum_film_fraction) * float(full_area)
    if film_area <= 0.0 or a.size == 0:
        return a, q, 0.0
    tracer_threshold = (
        0.02 * float(rho_reference) * float(full_area) * float(dz)
    )
    gas_core = tracer > tracer_threshold
    if front_height is not None:
        # Atmospheric/tracer gas above the resolved bulk surface is not a
        # confined Taylor core and must not receive a prescribed wall film.
        cell_bottom = np.arange(a.size, dtype=float) * float(dz)
        gas_core &= cell_bottom < max(float(front_height), 0.0)
    deficient = gas_core & (a < film_area)
    if not np.any(deficient):
        return a, q, 0.0

    deficit_volume = float(
        np.sum(film_area - a[deficient]) * dz
    )
    donor_floor = np.where(gas_core, film_area, 0.0)
    available_volume = float(
        np.sum(np.maximum(a - donor_floor, 0.0)) * dz
    )
    accepted = min(deficit_volume, available_volume)
    if accepted <= 1.0e-16:
        return a, q, 0.0

    remaining = accepted
    carried_momentum = 0.0
    for index in range(a.size - 1, -1, -1):
        if remaining <= 1.0e-16:
            break
        available = max((float(a[index]) - donor_floor[index]) * dz, 0.0)
        take = min(available, remaining)
        if take <= 0.0:
            continue
        old_area = float(a[index])
        fraction = take / max(old_area * dz, EPS)
        carried_momentum += float(q[index]) * dz * fraction
        a[index] = old_area - take / dz
        q[index] *= max(1.0 - fraction, 0.0)
        remaining -= take

    transfer_velocity = carried_momentum / max(accepted, EPS)
    remaining = accepted
    for index in np.flatnonzero(deficient):
        if remaining <= 1.0e-16:
            break
        add = min((film_area - float(a[index])) * dz, remaining)
        if add <= 0.0:
            continue
        area_add = add / dz
        a[index] += area_add
        q[index] += area_add * transfer_velocity
        remaining -= add
    if remaining > 1.0e-13 * max(accepted, 1.0):
        raise FloatingPointError("annular-film return lost liquid")
    return a, q, accepted


def _orthogonal_junction_liquid_exchange(
    area: float,
    discharge: float,
    *,
    upward_flow: float,
    dt: float,
    cell_width: float,
) -> tuple[float, float]:
    """Apply one side-T liquid exchange to a horizontal control volume.

    Positive ``upward_flow`` removes liquid through the orthogonal riser.  The
    removed parcel carries the local horizontal velocity up to the turn, so
    its horizontal momentum is removed in the same fraction as its volume.
    Negative flow returns vertically descending water with zero horizontal
    inlet momentum; the T wall supplies the turning reaction and the axial
    pipe equations subsequently accelerate that water.  No response history
    or fitted junction velocity enters this finite-volume balance.
    """

    old_area = max(float(area), 0.0)
    old_discharge = float(discharge)
    volume_change = -float(upward_flow) * float(dt)
    new_area = old_area + volume_change / float(cell_width)
    if new_area < -1.0e-12:
        raise FloatingPointError(
            "side-T liquid exchange emptied its donor cell"
        )
    new_area = max(new_area, 0.0)
    if volume_change < 0.0 and old_area > EPS:
        new_discharge = old_discharge * new_area / old_area
    else:
        new_discharge = old_discharge
    return new_area, new_discharge


def _two_phase_mixing_activation(void_fraction: float) -> float:
    """Return the bounded interfacial-area weight ``4 alpha_g alpha_l``."""

    alpha_g = float(np.clip(void_fraction, 0.0, 1.0))
    return 4.0 * alpha_g * (1.0 - alpha_g)


def _countercurrent_flooding_liquid_flow(
    requested_flow: float,
    *,
    upward_gas_superficial_velocity: float,
    full_area: float,
    diameter: float,
    rho_l: float,
    rho_g: float,
    gravity: float,
    wallis_constant: float,
    wallis_slope: float = 1.0,
) -> float:
    """Apply the Wallis CCFL capacity to downward liquid flow.

    The dimensionless Wallis relation is
    ``sqrt(Jg*) + m sqrt(Jl*) = C`` with
    ``Jk* = Jk sqrt(rho_k/(g D (rho_l-rho_g)))``.  Positive/upward liquid
    flow is unchanged.  For counter-current operation the returned value is
    the requested downward flow unless it exceeds the flooding capacity.
    This is a local algebraic two-fluid closure, not a time- or result-based
    velocity limiter.
    """

    flow = float(requested_flow)
    if flow >= 0.0:
        return flow
    if float(upward_gas_superficial_velocity) <= 0.0:
        # CCFL is a counter-current two-phase limit.  With no upward gas phase
        # there is no flooding mechanism, so single-phase liquid return must
        # not be capped by the zero-gas intercept of the Wallis envelope.
        return flow
    if (
        full_area <= 0.0
        or diameter <= 0.0
        or rho_l <= 0.0
        or gravity <= 0.0
        or wallis_constant <= 0.0
        or wallis_slope <= 0.0
    ):
        raise ValueError("positive CCFL geometry, densities, and coefficients required")
    gas_density = min(max(float(rho_g), 1.0e-12), 0.999999 * float(rho_l))
    density_difference = max(float(rho_l) - gas_density, 1.0e-12)
    gas_velocity_scale = math.sqrt(
        gas_density / (float(gravity) * float(diameter) * density_difference)
    )
    jg_star = max(float(upward_gas_superficial_velocity), 0.0) * gas_velocity_scale
    remaining = max(float(wallis_constant) - math.sqrt(max(jg_star, 0.0)), 0.0)
    jl_star_capacity = (remaining / float(wallis_slope)) ** 2
    liquid_velocity_scale = math.sqrt(
        float(gravity)
        * float(diameter)
        * density_difference
        / float(rho_l)
    )
    downward_capacity = (
        jl_star_capacity * liquid_velocity_scale * float(full_area)
    )
    return max(flow, -downward_capacity)


def _side_t_opening_weights(
    cell_count: int,
    *,
    cell_width: float,
    junction_center: float,
    opening_width: float,
) -> np.ndarray:
    """Return conservative cell weights for the measured side-T footprint.

    The circular tower opening occupies ``opening_width`` along the tunnel
    axis.  Each cell receives the exact overlap of its finite-volume interval
    with that physical footprint.  The normalized weights therefore sum to
    one independently of grid alignment; no numerical spreading length is
    introduced.
    """

    count = int(cell_count)
    width = float(cell_width)
    centre = float(junction_center)
    mouth = float(opening_width)
    if count < 1 or width <= 0.0 or mouth <= 0.0:
        raise ValueError("positive side-T grid and opening geometry required")
    left = np.arange(count, dtype=float) * width
    right = left + width
    mouth_left = centre - 0.5 * mouth
    mouth_right = centre + 0.5 * mouth
    overlap = np.maximum(
        np.minimum(right, mouth_right) - np.maximum(left, mouth_left),
        0.0,
    )
    total = float(np.sum(overlap))
    if total <= 0.0:
        raise ValueError("side-T opening does not overlap the horizontal grid")
    return overlap / total


def _limit_side_t_upward_liquid_flow(
    area,
    *,
    requested_flow: float,
    opening_weights,
    dt: float,
    cell_width: float,
    retained_fraction: float = 0.10,
) -> float:
    """Apply the donor-positivity limit for a finite-width side outlet.

    Positive flow leaves the horizontal pipe and enters the riser.  Every
    footprint cell must retain the same prescribed fraction, so the most
    restrictive physical donor fixes the common branch flow.  Negative
    (returning) flow is not limited here because it adds liquid.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    weights = np.asarray(opening_weights, dtype=float)
    if a.shape != weights.shape or a.ndim != 1:
        raise ValueError("side-T area and opening weights must have equal shape")
    q = float(requested_flow)
    step = float(dt)
    width = float(cell_width)
    retain = float(retained_fraction)
    if step <= 0.0 or width <= 0.0 or not 0.0 <= retain < 1.0:
        raise ValueError("valid side-T timestep, width, and retention required")
    if q <= 0.0:
        return q
    active = weights > 0.0
    capacities = (
        (1.0 - retain) * a[active] * width
        / (step * weights[active])
    )
    return min(q, float(np.min(capacities)))


def _riser_acoustic_momentum_dissipation_flux(
    discharge,
    celerity,
    liquid_area,
    *,
    full_area: float,
    bulk_fraction: float = 0.50,
):
    """Return conservative Rusanov damping for the riser momentum faces.

    The sharp liquid-area contact is advected without an acoustic area flux so
    a stationary free surface cannot be numerically smeared up the shaft.  The
    momentum equation nevertheless carries water-hammer characteristics.  A
    centred pressure source without their face dissipation admits an odd-even
    velocity mode; in Case A that mode reached alternating multi-metre-per-
    second cells immediately before breakthrough.  This term supplies only
    the missing ``-0.5 c Delta Q`` momentum flux.  It is conservative, vanishes
    for uniform discharge, and does not transport liquid area.
    """

    q = np.asarray(discharge, dtype=float)
    c = np.asarray(celerity, dtype=float)
    area = np.asarray(liquid_area, dtype=float)
    if q.shape != c.shape or q.shape != area.shape or q.ndim != 1:
        raise ValueError(
            "riser acoustic arrays must be equal one-dimensional fields"
        )
    if (
        np.any(~np.isfinite(q))
        or np.any(~np.isfinite(c))
        or np.any(~np.isfinite(area))
        or np.any(c < 0.0)
        or full_area <= 0.0
        or not 0.0 < bulk_fraction <= 1.0
    ):
        raise ValueError("finite bulk-liquid acoustic inputs required")
    flux = np.zeros(q.size + 1, dtype=float)
    if q.size > 1:
        face_celerity = np.maximum(c[:-1], c[1:])
        bulk_face = (
            np.minimum(area[:-1], area[1:])
            >= float(bulk_fraction) * float(full_area)
        )
        flux[1:-1] = np.where(
            bulk_face,
            -0.5 * face_celerity * (q[1:] - q[:-1]),
            0.0,
        )
    return flux


def _limit_riser_bottom_inflow_by_receiving_capacity(
    area,
    face_flux,
    *,
    requested_flow: float,
    dt: float,
    cell_width: float,
    full_area: float,
) -> float:
    """Limit side-T inflow by the first riser control-volume capacity.

    Horizontal donor availability is necessary but not sufficient: a nearly
    full first riser cell can accept only its geometric void plus liquid that
    leaves simultaneously through face 1.  Enforcing this finite-volume
    packing inequality before the source is committed avoids converting an
    otherwise admissible gross inflow into a one-step elastic pressure needle.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    flux = np.asarray(face_flux, dtype=float)
    q = float(requested_flow)
    step = float(dt)
    width = float(cell_width)
    capacity_area = float(full_area)
    if a.ndim != 1 or a.size < 1 or flux.shape != (a.size + 1,):
        raise ValueError("riser receiving arrays have inconsistent shape")
    if step <= 0.0 or width <= 0.0 or capacity_area <= 0.0:
        raise ValueError("positive riser receiving geometry and timestep required")
    if q <= 0.0:
        return q
    storage_rate = max(capacity_area - float(a[0]), 0.0) * width / step
    throughflow_rate = float(flux[1])
    admissible = max(storage_rate + throughflow_rate, 0.0)
    return min(q, admissible)


def _limit_side_t_downward_liquid_flow(
    area,
    gas_mass,
    *,
    requested_flow: float,
    opening_weights,
    dt: float,
    cell_width: float,
    full_area: float,
    rho_reference: float,
    density_ceiling: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    topology_density_fraction: float,
) -> float:
    """Limit riser return so it cannot crush gas in the T footprint.

    ``requested_flow`` is negative for liquid descending from the riser.  The
    admissible magnitude is limited only by cells containing a geometrically
    open, mass-supported gas phase.  A liquid-full footprint cell may accept
    returning water as elastic TPA storage; treating such a cell as zero gas
    capacity intermittently shut the complete film flux and pumped the riser.
    For a genuinely gassy cell, the minimum void follows from conserved gas
    mass at ``density_ceiling``.  The shared riser-face flux is reduced before
    either branch is updated, so no liquid or gas inventory is discarded.
    """

    q = float(requested_flow)
    if q >= 0.0:
        return q
    a = np.asarray(area, dtype=float)
    m = np.maximum(np.asarray(gas_mass, dtype=float), 0.0)
    weights = np.asarray(opening_weights, dtype=float)
    if not (a.shape == m.shape == weights.shape) or a.ndim != 1:
        raise ValueError("side-T return limiter arrays must be equal 1-D fields")
    if (
        dt <= 0.0
        or cell_width <= 0.0
        or full_area <= 0.0
        or rho_reference <= 0.0
        or density_ceiling <= 1.0
        or not 0.0 < active_void_fraction < 1.0
        or not 0.0 < topology_density_fraction < 1.0
    ):
        raise ValueError("invalid side-T return limiter parameters")
    active = weights > 0.0
    if not np.any(active):
        return 0.0
    raw_void = np.maximum(full_area - np.minimum(np.maximum(a, 0.0), full_area), 0.0)
    gas_supported = (
        raw_void > active_void_fraction * full_area
    ) & (
        m
        > topology_density_fraction
        * rho_reference
        * raw_void
        * cell_width
    )
    constrained = active & gas_supported
    if not np.any(constrained):
        return q
    minimum_void = np.maximum(
        void_floor_fraction * full_area,
        m / (density_ceiling * rho_reference * cell_width),
    )
    maximum_liquid_area = full_area - minimum_void
    capacity = np.maximum(maximum_liquid_area - a, 0.0)
    allowable = np.min(
        capacity[constrained]
        * cell_width
        / np.maximum(dt * weights[constrained], EPS)
    )
    return -min(abs(q), max(float(allowable), 0.0))


def _side_t_return_exchange_weights(
    area,
    opening_weights,
    *,
    full_area: float,
    gas_supported=None,
) -> np.ndarray:
    """Return the actual footprint weights used by a descending side-T jet.

    Returning liquid first occupies the resolved void under the opening.  The
    same weights must be used by both the receiver-capacity limiter and the
    subsequent conservative exchange; otherwise a flux that is admissible for
    the geometric mouth weights can still overfill one gas-bearing receiver
    after the compliance redistribution.
    """

    a = np.asarray(area, dtype=float)
    weights = np.asarray(opening_weights, dtype=float)
    if a.shape != weights.shape or a.ndim != 1:
        raise ValueError("side-T area and opening weights must have equal shape")
    if full_area <= 0.0:
        raise ValueError("positive side-T full area required")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("nonnegative side-T opening weights required")

    exchange_weights = weights.copy()
    if gas_supported is not None:
        gas_mask = np.asarray(gas_supported, dtype=bool)
        if gas_mask.shape != a.shape:
            raise ValueError("side-T gas-support mask must match the area field")
        # The falling wall film reaches the liquid-filled lower part of the
        # finite side T.  When the measured footprint overlaps both a
        # gas-crown cell and a liquid-continuous cell, turn the return into the
        # latter instead of numerically crushing the crown gas.  This is the
        # cross-section-averaged representation of vertical liquid entering
        # below a horizontally stratified gas layer; it uses only local phase
        # topology and the measured opening overlap.
        liquid_path = (weights > 0.0) & ~gas_mask & (a > 0.0)
        if np.any(liquid_path):
            liquid_weights = np.where(
                liquid_path,
                weights * np.minimum(a / float(full_area), 1.0),
                0.0,
            )
            total_liquid_weight = float(np.sum(liquid_weights))
            if total_liquid_weight > 0.0:
                return liquid_weights / total_liquid_weight
    capacity = np.maximum(float(full_area) - a, 0.0)
    compliant = (weights > 0.0) & (capacity > 0.0)
    if np.any(compliant):
        compliance_weight = weights * capacity
        total_compliance = float(np.sum(compliance_weight))
        if total_compliance > 0.0:
            exchange_weights = compliance_weight / total_compliance
    return exchange_weights


def _limit_taylor_return_exchange_flow(
    riser_liquid_area,
    riser_volume_flux,
    horizontal_liquid_area,
    horizontal_gas_mass,
    *,
    requested_return_flow: float,
    opening_weights,
    dt: float,
    riser_cell_width: float,
    horizontal_cell_width: float,
    horizontal_full_area: float,
    rho_reference: float,
    density_ceiling: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    topology_density_fraction: float,
    retained_fraction: float = 0.10,
) -> float:
    """Limit one Taylor-return face transaction by both adjacent controls.

    ``requested_return_flow`` is a positive magnitude.  The Taylor closure
    replaces the already limited riser bottom face, so its new value must be
    limited again.  The bottom-cell donor budget reserves any simultaneous
    upward outflow through face 1, while the receiver budget uses the exact
    compliance weights that will deposit the liquid beneath the side opening.
    The returned positive magnitude can therefore be committed once as
    ``G1[0] = -return_flow`` without a later positivity clip creating water.
    """

    riser_area = np.maximum(np.asarray(riser_liquid_area, dtype=float), 0.0)
    riser_flux = np.asarray(riser_volume_flux, dtype=float)
    if (
        riser_area.ndim != 1
        or riser_area.size < 1
        or riser_flux.shape != (riser_area.size + 1,)
    ):
        raise ValueError("inconsistent riser area and face-flux arrays")
    step = float(dt)
    dz = float(riser_cell_width)
    retain = float(retained_fraction)
    requested = max(float(requested_return_flow), 0.0)
    if step <= 0.0 or dz <= 0.0 or not 0.0 <= retain < 1.0:
        raise ValueError("valid Taylor-return timestep, grid, and retention required")
    if requested <= 0.0:
        return 0.0

    # Face 0 and face 1 are the only possible outflows from the bottom riser
    # cell.  Incoming flow through face 1 is deliberately not borrowed by the
    # boundary limiter: every explicit step retains the same positive donor
    # reserve as the ordinary FV donor limiter.
    donor_rate = (1.0 - retain) * riser_area[0] * dz / step
    other_outflow = max(float(riser_flux[1]), 0.0)
    donor_limited = min(requested, max(donor_rate - other_outflow, 0.0))
    if donor_limited <= 0.0:
        return 0.0

    receiver_weights = _side_t_return_exchange_weights(
        horizontal_liquid_area,
        opening_weights,
        full_area=horizontal_full_area,
    )
    signed_limited = _limit_side_t_downward_liquid_flow(
        horizontal_liquid_area,
        horizontal_gas_mass,
        requested_flow=-donor_limited,
        opening_weights=receiver_weights,
        dt=step,
        cell_width=horizontal_cell_width,
        full_area=horizontal_full_area,
        rho_reference=rho_reference,
        density_ceiling=density_ceiling,
        void_floor_fraction=void_floor_fraction,
        active_void_fraction=active_void_fraction,
        topology_density_fraction=topology_density_fraction,
    )
    return max(-float(signed_limited), 0.0)


def _apply_finite_width_side_t_exchange(
    area,
    discharge,
    *,
    upward_flow: float,
    opening_weights,
    dt: float,
    cell_width: float,
    full_area: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange liquid through the real side-T footprint conservatively.

    Positive ``upward_flow`` removes liquid and its local axial momentum from
    the footprint.  A descending stream enters normal to the horizontal axis
    and therefore brings zero *axial* momentum into the one-dimensional
    horizontal equations.  The added volume changes the local pressure state;
    the ordinary horizontal finite-volume fluxes then generate and propagate
    the resulting waves.  No signed axial impulse or target wave shape is
    inserted at the junction.
    """

    a = np.asarray(area, dtype=float).copy()
    q = np.asarray(discharge, dtype=float).copy()
    weights = np.asarray(opening_weights, dtype=float)
    if a.shape != q.shape or a.shape != weights.shape or a.ndim != 1:
        raise ValueError("side-T exchange arrays must have equal one-dimensional shape")
    step = float(dt)
    width = float(cell_width)
    if step <= 0.0 or width <= 0.0:
        raise ValueError("positive side-T timestep and cell width required")
    volume_change = -float(upward_flow) * step
    old_area = a.copy()
    exchange_weights = weights.copy()
    if volume_change > 0.0 and full_area is not None:
        # A descending jet enters the finite T intersection and first occupies
        # the gas/open-channel volume available beneath the measured mouth.
        # Applying the same geometric fraction to a liquid-full east cell and
        # a gassy west cell compressed the former by O(10%) while leaving void
        # beside it, launching a nonphysical water-hammer jet.  The zero-volume
        # junction pressure instead distributes an incompressible addition by
        # local compliance: here that compliance is exactly the available void
        # within each geometrically overlapped cell.  No cell outside the real
        # mouth receives water and no axial momentum is prescribed.
        exchange_weights = _side_t_return_exchange_weights(
            a,
            weights,
            full_area=float(full_area),
        )
    a += volume_change * exchange_weights / width
    if np.any(a < -1.0e-12):
        raise FloatingPointError("finite-width side-T exchange emptied a donor cell")
    a = np.maximum(a, 0.0)
    removing = a < old_area
    wet = old_area > EPS
    scale = np.ones_like(a)
    scale[removing & wet] = a[removing & wet] / old_area[removing & wet]
    scale[removing & ~wet] = 0.0
    q *= scale
    actual_change = float(np.sum(a - old_area) * width)
    if not math.isclose(
        actual_change,
        volume_change,
        rel_tol=1.0e-10,
        abs_tol=1.0e-16,
    ):
        raise FloatingPointError("finite-width side-T exchange lost liquid volume")
    return a, q


def _apply_finite_width_side_t_gross_exchange(
    area,
    discharge,
    *,
    upward_flow: float,
    downward_flow: float,
    opening_weights,
    downward_weights=None,
    dt: float,
    cell_width: float,
    full_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Commit simultaneous gross liquid exchange over the side-T footprint.

    The upward horizontal donor and downward riser return are distinct parcels
    even when their net rate is small.  Apply the already limited upward
    withdrawal first, then place the orthogonal return into the newly available
    compliance.  The combined horizontal-volume change is exactly
    ``(Q_down-Q_up) dt``; neither branch is replaced by its signed net.
    """

    q_up = float(upward_flow)
    q_down = float(downward_flow)
    if (
        not math.isfinite(q_up)
        or not math.isfinite(q_down)
        or q_up < 0.0
        or q_down < 0.0
    ):
        raise ValueError("gross side-T rates must be finite and non-negative")
    before = float(np.sum(np.asarray(area, dtype=float)) * float(cell_width))
    a, q = _apply_finite_width_side_t_exchange(
        area,
        discharge,
        upward_flow=q_up,
        opening_weights=opening_weights,
        dt=dt,
        cell_width=cell_width,
        full_area=full_area,
    )
    return_weights = (
        opening_weights
        if downward_weights is None
        else downward_weights
    )
    a, q = _apply_finite_width_side_t_exchange(
        a,
        q,
        upward_flow=-q_down,
        opening_weights=return_weights,
        dt=dt,
        cell_width=cell_width,
        full_area=full_area,
    )
    after = float(np.sum(a) * float(cell_width))
    expected = (q_down - q_up) * float(dt)
    if not math.isclose(
        after - before,
        expected,
        rel_tol=1.0e-10,
        abs_tol=1.0e-16,
    ):
        raise FloatingPointError("gross side-T exchange lost liquid volume")
    return a, q


def _open_side_t_east_capillary_cutcell(
    liquid_area,
    liquid_discharge,
    gas_mass,
    gas_momentum,
    *,
    junction_face: int,
    cell_width: float,
    full_area: float,
    opening_width: float,
    target_void_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Activate the gas-wetted east half of a finite-width side-T opening.

    A side-T is not a zero-width point: its circular opening spans one tower
    diameter along the horizontal-pipe crown.  When the material gas front
    first reaches the opening, the part lying east of the discrete junction
    face is already connected.  Its initial crown void is the larger of the
    capillary-open segment and the resolved T-cell gas layer, enforcing
    geometric continuity rather than inserting a tiny high-pressure gas
    volume into an otherwise full downstream cell.

    This routine performs that one topology-changing cut-cell remap.  The
    opened volume follows from the measured opening width and resolved local
    gas-layer depth.  Gas mass and momentum are transferred from the
    west gas cell at its current density and velocity.  The displaced liquid
    is stored elastically in the remaining closed downstream column, with its
    axial momentum transferred with the parcel.  Both phase inventories are
    therefore conserved exactly; no later interface position is prescribed.
    """

    al = np.asarray(liquid_area, dtype=float).copy()
    ql = np.asarray(liquid_discharge, dtype=float).copy()
    mg = np.asarray(gas_mass, dtype=float).copy()
    jg = np.asarray(gas_momentum, dtype=float).copy()
    if not (al.shape == ql.shape == mg.shape == jg.shape) or al.ndim != 1:
        raise ValueError("side-T cut-cell arrays must be equal one-dimensional fields")
    face = int(junction_face)
    if not 1 <= face < al.size:
        raise ValueError("side-T cut-cell face lies outside the grid")
    if (
        cell_width <= 0.0
        or full_area <= 0.0
        or opening_width <= 0.0
        or not 0.0 < target_void_fraction < 1.0
    ):
        raise ValueError("invalid side-T cut-cell geometry")

    donor = face - 1
    donor_void = max(full_area - min(max(float(al[donor]), 0.0), full_area), 0.0)
    if donor_void <= 0.0 or mg[donor] <= 0.0:
        return al, ql, mg, jg, 0.0, 0.0
    donor_density = float(mg[donor] / max(donor_void * cell_width, EPS))
    donor_velocity = float(jg[donor] / max(mg[donor], EPS))

    face_x = face * cell_width
    opening_east = face_x + 0.5 * opening_width
    cut_indices: list[int] = []
    target_void_area: list[float] = []
    capillary_area = target_void_fraction * full_area
    for index in range(face, al.size):
        left = index * cell_width
        right = left + cell_width
        overlap = max(min(right, opening_east) - max(left, face_x), 0.0)
        if overlap <= 0.0:
            if left >= opening_east:
                break
            continue
        cut_indices.append(index)
        target_void_area.append(capillary_area * overlap / cell_width)
    if not cut_indices:
        return al, ql, mg, jg, 0.0, 0.0

    # A newly connected crown segment cannot be filled at the donor density
    # unless the material gas cell actually contains the required mass.  On a
    # refined grid the half-mouth may span several cells while the gas nose is
    # still only partly inside its donor.  Open the same geometric segment
    # fractionally in that case, rather than creating gas or aborting the
    # conservative update.  The scalar below is the unique monotone solution
    # for the largest admissible opening at the current donor state.
    current_void_area = np.array(
        [
            max(
                full_area - min(max(float(al[index]), 0.0), full_area),
                0.0,
            )
            for index in cut_indices
        ],
        dtype=float,
    )
    target_void_array = np.asarray(target_void_area, dtype=float)
    void_deficit = np.maximum(target_void_array - current_void_area, 0.0)

    def required_mass(open_fraction: float) -> float:
        opened_void = current_void_area + open_fraction * void_deficit
        target_mass = donor_density * opened_void * cell_width
        existing_mass = np.asarray(mg[cut_indices], dtype=float)
        return float(np.sum(np.maximum(target_mass - existing_mass, 0.0)))

    available_mass = 0.95 * float(mg[donor])
    open_fraction = 1.0
    if required_mass(open_fraction) > available_mass:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if required_mass(mid) <= available_mass:
                lo = mid
            else:
                hi = mid
        open_fraction = lo
    target_void_area = list(
        current_void_area + open_fraction * void_deficit
    )

    opened_volume = 0.0
    removed_momentum_integral = 0.0
    for index, target_void in zip(cut_indices, target_void_area):
        current_void = max(full_area - min(max(float(al[index]), 0.0), full_area), 0.0)
        deficit_area = max(target_void - current_void, 0.0)
        if deficit_area <= 0.0:
            continue
        old_area = max(float(al[index]), 0.0)
        old_q = float(ql[index])
        new_area = max(old_area - deficit_area, 0.0)
        new_q = old_q * new_area / max(old_area, EPS)
        al[index] = new_area
        ql[index] = new_q
        opened_volume += deficit_area * cell_width
        removed_momentum_integral += (old_q - new_q) * cell_width

    if opened_volume <= 0.0:
        return al, ql, mg, jg, 0.0, 0.0

    # The downstream end is closed.  On the water-hammer time scale the small
    # displaced cut-cell volume is elastic storage in the downstream column.
    storage = np.arange(cut_indices[-1] + 1, al.size, dtype=int)
    if storage.size == 0:
        raise FloatingPointError("side-T east cut-cell has no downstream liquid storage")
    area_increment = opened_volume / (storage.size * cell_width)
    transfer_velocity = removed_momentum_integral / max(opened_volume, EPS)
    al[storage] += area_increment
    ql[storage] += area_increment * transfer_velocity

    transferred_mass = 0.0
    for index, target_void in zip(cut_indices, target_void_area):
        target_mass = donor_density * target_void * cell_width
        add_mass = max(target_mass - float(mg[index]), 0.0)
        transferred_mass += add_mass
    if transferred_mass > available_mass * (1.0 + 1.0e-12) + 1.0e-18:
        raise FloatingPointError("side-T cut-cell donor limiter failed")
    donor_fraction = transferred_mass / max(float(mg[donor]), EPS)
    mg[donor] -= transferred_mass
    jg[donor] *= 1.0 - donor_fraction
    for index, target_void in zip(cut_indices, target_void_area):
        target_mass = donor_density * target_void * cell_width
        add_mass = max(target_mass - float(mg[index]), 0.0)
        mg[index] += add_mass
        jg[index] += add_mass * donor_velocity

    return al, ql, mg, jg, opened_volume, transferred_mass


def _limit_three_branch_junction_flows(
    area,
    *,
    junction_face: int,
    reference_flow: float,
    west_flow: float,
    east_flow: float,
    dt: float,
    cell_width: float,
    retained_fraction: float = 0.10,
) -> tuple[float, float, float]:
    """Positivity-limit a side-T flux triplet with one common face factor.

    Scaling both branch corrections about the uninterrupted reference flux by
    the same factor preserves

    ``q_w - q_e = q_vertical``.

    Hence the vertical boundary flux can use the returned factor and the three
    control-volume updates remain exactly conservative.  This is a donor-cell
    positivity condition, not a response or wave-amplitude limiter.
    """

    a = np.asarray(area, dtype=float)
    if a.ndim != 1:
        raise ValueError("horizontal junction area must be a 1-D field")
    face = int(junction_face)
    if not 1 <= face < a.size:
        raise ValueError("junction face lies outside the horizontal grid")
    step = float(dt)
    width = float(cell_width)
    retain = float(retained_fraction)
    if step <= 0.0 or width <= 0.0 or not 0.0 <= retain < 1.0:
        raise ValueError("valid junction timestep, width, and retention required")
    q_ref = float(reference_flow)
    q_w = float(west_flow)
    q_e = float(east_flow)
    if not np.all(np.isfinite([q_ref, q_w, q_e])):
        raise ValueError("finite side-T branch flows required")

    changes = np.asarray(
        [
            step / width * (q_ref - q_w),
            step / width * (q_e - q_ref),
        ],
        dtype=float,
    )
    donors = a[[face - 1, face]]
    factor = 1.0
    for old_area, change in zip(donors, changes):
        if change < 0.0:
            available = (1.0 - retain) * max(float(old_area), 0.0)
            factor = min(factor, available / max(-float(change), EPS))
    factor = float(np.clip(factor, 0.0, 1.0))
    return (
        q_ref + factor * (q_w - q_ref),
        q_ref + factor * (q_e - q_ref),
        factor,
    )


def _replace_horizontal_face_with_tjunction_fluxes(
    area,
    discharge,
    *,
    junction_face: int,
    reference_flow: float,
    west_flow: float,
    east_flow: float,
    dt: float,
    cell_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace one uninterrupted-pipe face by a conservative side-T node.

    ``west_flow`` and ``east_flow`` are positive toward increasing horizontal
    coordinate.  The liquid entering the vertical branch is therefore
    ``west_flow - east_flow``.  The provisional horizontal finite-volume step
    has already used ``reference_flow`` on both sides of the internal face;
    this correction substitutes the two characteristic branch flows on the
    respective west and east control volumes.  Thus descending riser liquid
    leaves the zero-volume junction immediately through the horizontal faces
    instead of being stored in, and sealing, an arbitrarily selected cell.

    The orthogonal inflow carries zero *axial* momentum: descending tower water
    changes the two adjacent control-volume areas, after which the horizontal
    pressure and momentum equations accelerate it.  Conversely, liquid leaving
    a horizontal cell for the riser carries that cell's axial velocity, so its
    discharge is reduced in the same proportion as its area.  This is the
    finite-volume side-source balance for a right-angle branch.  It replaces
    the former water-hammer-Courant relaxation of the two cell-centre
    discharges, which was not a momentum flux and injected a new grid-scale
    impulse at every step.
    """

    a = np.asarray(area, dtype=float).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("horizontal junction arrays must be equal 1-D fields")
    face = int(junction_face)
    if not 1 <= face < a.size:
        raise ValueError("junction face lies outside the horizontal grid")
    step = float(dt)
    width = float(cell_width)
    if not np.isfinite(step) or step <= 0.0 or width <= 0.0:
        raise ValueError("positive finite junction timestep and width required")
    q_ref = float(reference_flow)
    q_w = float(west_flow)
    q_e = float(east_flow)
    if not np.all(np.isfinite([q_ref, q_w, q_e])):
        raise ValueError("finite side-T branch flows required")

    west = face - 1
    east = face
    old_pair = a[[west, east]].copy()
    old_discharge_pair = q[[west, east]].copy()
    a[west] += step / width * (q_ref - q_w)
    a[east] += step / width * (q_e - q_ref)
    if np.any(a[[west, east]] < -1.0e-12):
        raise FloatingPointError("side-T branch flux emptied a horizontal cell")
    a[[west, east]] = np.maximum(a[[west, east]], 0.0)
    for local, index in enumerate((west, east)):
        # Negative area change is liquid leaving the horizontal branch.  The
        # parcel carries its pre-existing axial velocity into the fitting and
        # the tee wall supplies the turning reaction.  Positive area change is
        # vertical inflow and therefore brings no prescribed axial momentum.
        if a[index] < old_pair[local] and old_pair[local] > EPS:
            q[index] = (
                old_discharge_pair[local]
                * a[index]
                / old_pair[local]
            )

    expected_change = (q_e - q_w) * step
    actual_change = float(np.sum(a[[west, east]] - old_pair) * width)
    if not math.isclose(
        actual_change,
        expected_change,
        rel_tol=1.0e-10,
        abs_tol=1.0e-16,
    ):
        raise FloatingPointError("three-branch junction lost liquid volume")
    return a, q


def _limit_gas_void_closure_flux(
    area,
    gas_mass,
    volume_flux,
    momentum_flux,
    *,
    full_area: float,
    cell_width: float,
    dt: float,
    rho_reference: float,
    density_fraction: float,
    density_ceiling: float,
    void_floor_fraction: float,
    active_void_fraction: float,
    closure_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Locally limit liquid inflow that would collapse resolved gas void.

    The limiter acts on shared finite-volume faces, so a reduced inflow to one
    cell is the same reduced outflow from its neighbour.  Volume and momentum
    components receive the same face factor.  This replaces the previous
    *global* phase-volume timestep restriction with a standard conservative
    positivity correction confined to the closing gas cell.
    """

    a = np.asarray(area, dtype=float)
    m = np.asarray(gas_mass, dtype=float)
    q = np.asarray(volume_flux, dtype=float).copy()
    f2 = np.asarray(momentum_flux, dtype=float).copy()
    if a.ndim != 1 or m.shape != a.shape:
        raise ValueError("area and gas mass must be equal one-dimensional arrays")
    if q.shape != (a.size + 1,) or f2.shape != q.shape:
        raise ValueError("face flux arrays must have one more entry than cells")
    if (
        dt <= 0.0
        or cell_width <= 0.0
        or full_area <= 0.0
        or not 0.0 < void_floor_fraction < active_void_fraction < 1.0
    ):
        raise ValueError("positive geometry and timestep are required")
    theta = float(np.clip(closure_fraction, 0.0, 1.0))
    void = np.maximum(full_area - np.clip(a, 0.0, full_area), 0.0)
    void_effective = np.maximum(void, void_floor_fraction * full_area)
    # The upper density threshold is used by the gas momentum solver to
    # classify an unresolved compression front; it must not disable liquid
    # positivity.  A compressed gas-bearing cell needs *more*, not less,
    # protection against complete liquid closure.
    supported = (
        m > density_fraction * rho_reference * void_effective * cell_width
    )
    if not np.any(supported):
        return q, f2

    original_q = q.copy()
    factor = np.ones_like(q)
    # Repeated fractional closure must not geometrically erase a cell that
    # still contains conserved gas mass.  Retain the thermodynamic volume
    # required to keep that mass below the resolved-density ceiling.
    mass_supported_void = m / (
        density_ceiling * rho_reference * cell_width
    )
    # Once a material gas cell has opened past the active topology scale, do
    # not let the liquid update close it below that scale while its conserved
    # gas mass is still present.  Otherwise the next gas step closes both
    # faces around a compressed, non-empty cell and creates an artificial
    # liquid plug.  Isolated positivity-floor residue is not promoted because
    # it never satisfies ``topology_open``.
    topology_open = (
        void >= active_void_fraction * full_area
    ) & supported
    topology_void = np.where(
        topology_open,
        active_void_fraction * full_area,
        void_floor_fraction * full_area,
    )
    minimum_void = np.maximum.reduce(
        (
            topology_void,
            (1.0 - theta) * void,
            mass_supported_void,
        )
    )
    target_area = full_area - minimum_void

    # Face factors only decrease.  Iteration propagates a downstream limiter
    # upstream when reducing one cell's outflow increases its own net filling.
    for _ in range(8):
        changed = False
        q_limited = original_q * factor
        for i in np.flatnonzero(supported):
            left = float(q_limited[i])
            right = float(q_limited[i + 1])
            incoming = max(left, 0.0) + max(-right, 0.0)
            if incoming <= EPS:
                continue
            outgoing = max(-left, 0.0) + max(right, 0.0)
            capacity = max(
                (float(target_area[i]) - float(a[i]))
                * cell_width / dt,
                0.0,
            )
            allowed = outgoing + capacity
            if incoming <= allowed + 1.0e-15:
                continue
            local_factor = max(min(allowed / incoming, 1.0), 0.0)
            if left > 0.0 and local_factor < factor[i]:
                factor[i] = local_factor
                changed = True
            if right < 0.0 and local_factor < factor[i + 1]:
                factor[i + 1] = local_factor
                changed = True
        if not changed:
            break

    q = original_q * factor
    f2 *= factor
    return q, f2


def _limit_liquid_donor_flux(
    area,
    volume_flux,
    momentum_flux,
    *,
    cell_width: float,
    dt: float,
    retained_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively limit the *sum* of all outflows from each liquid cell."""

    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    q = np.asarray(volume_flux, dtype=float).copy()
    f2 = np.asarray(momentum_flux, dtype=float).copy()
    if a.ndim != 1 or q.shape != (a.size + 1,) or f2.shape != q.shape:
        raise ValueError("donor limiter received inconsistent cell/face arrays")
    if cell_width <= 0.0 or dt <= 0.0:
        raise ValueError("donor limiter requires positive dx and dt")
    keep = float(np.clip(retained_fraction, 0.0, 1.0))
    outgoing = np.maximum(-q[:-1], 0.0) + np.maximum(q[1:], 0.0)
    available_rate = (1.0 - keep) * a * cell_width / dt
    cell_factor = np.ones_like(a)
    draining = outgoing > available_rate
    cell_factor[draining] = np.divide(
        available_rate[draining],
        outgoing[draining],
        out=np.zeros_like(available_rate[draining]),
        where=outgoing[draining] > EPS,
    )
    face_factor = np.ones_like(q)
    # Internal positive flux is donated by the left cell; negative flux by the
    # right cell.  At domain boundaries only outward flow has a local donor.
    face_factor[1:-1] = np.where(
        q[1:-1] >= 0.0,
        cell_factor[:-1],
        cell_factor[1:],
    )
    if q[0] < 0.0:
        face_factor[0] = cell_factor[0]
    if q[-1] > 0.0:
        face_factor[-1] = cell_factor[-1]
    q *= face_factor
    f2 *= face_factor
    return q, f2


def _minmod(a, b):
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _advect_mass_muscl(mass, u_face, ds, dt):
    """Second-order TVD upwind advection for cell masses."""
    c = np.maximum(mass, 0.0) / ds
    slope = np.zeros_like(c)
    if len(c) > 2:
        slope[1:-1] = _minmod(c[1:-1] - c[:-2], c[2:] - c[1:-1])

    flux = np.zeros(len(c) + 1)
    left = c[:-1] + 0.5 * slope[:-1]
    right = c[1:] - 0.5 * slope[1:]
    uf = u_face[1:-1]
    flux[1:-1] = np.where(uf >= 0.0, uf * left, uf * right)
    flux[0] = 0.0
    flux[-1] = max(u_face[-1], 0.0) * (c[-1] + 0.5 * slope[-1])
    c_new = c - dt / ds * (flux[1:] - flux[:-1])
    return np.maximum(c_new * ds, 0.0)


def _advect_momentum_upwind(momentum, u_face, ds, dt):
    """Conservative first-order upwind advection for gas momentum."""
    q = momentum / ds
    q_ext = np.concatenate([[0.0], q, [0.0]])
    flux = np.where(u_face >= 0.0, u_face * q_ext[:-1], u_face * q_ext[1:])
    q_new = q - dt / ds * (flux[1:] - flux[:-1])
    return q_new * ds


def _phase_rusanov(mass, momentum, ds, dt, wave_speed, left="wall", right="wall"):
    """Finite-volume Rusanov update for one phase's mass and momentum.

    The conserved variables are cell-integrated mass [kg] and axial momentum
    [kg m/s]. Pressure, gravity, and drag are added separately as source terms.
    """
    m = np.maximum(mass, 0.0)
    mom = np.where(m > 1.0e-14, momentum, 0.0)
    mg = np.empty(len(m) + 2)
    jg = np.empty(len(m) + 2)
    mg[1:-1] = m
    jg[1:-1] = mom
    if left == "wall":
        mg[0] = m[0]; jg[0] = -mom[0]
    else:
        mg[0] = m[0]; jg[0] = mom[0]
    if right == "open":
        mg[-1] = m[-1]; jg[-1] = max(mom[-1], 0.0)
    elif right == "wall":
        mg[-1] = m[-1]; jg[-1] = -mom[-1]
    else:
        mg[-1] = m[-1]; jg[-1] = mom[-1]

    u = np.where(mg > 1.0e-14, jg / np.maximum(mg, 1.0e-14), 0.0)
    f_m = jg
    f_j = jg * u
    s = np.maximum(np.abs(u[:-1]) + wave_speed, np.abs(u[1:]) + wave_speed)
    fm = 0.5 * (f_m[:-1] + f_m[1:]) - 0.5 * s * (mg[1:] - mg[:-1])
    fj = 0.5 * (f_j[:-1] + f_j[1:]) - 0.5 * s * (jg[1:] - jg[:-1])
    m_new = m - dt / ds * (fm[1:] - fm[:-1])
    j_new = mom - dt / ds * (fj[1:] - fj[:-1])
    m_new = np.maximum(m_new, 0.0)
    j_new = np.where(m_new > 1.0e-14, j_new, 0.0)
    return m_new, j_new


def _cap_momentum(momentum, mass, velocity_cap):
    """Limit gas velocity while preserving the updated gas mass."""
    out = np.zeros_like(momentum)
    active = mass > 1.0e-14
    u = np.zeros_like(momentum)
    u[active] = np.clip(momentum[active] / mass[active], -velocity_cap, velocity_cap)
    out[active] = mass[active] * u[active]
    return out


def _map_external_horizontal_state(
    solver,
    state,
    *,
    x_target,
    full_area: float,
    dx: float,
):
    """Map a same-grid shock-fitting state into the network conserved fields.

    The hand-off is intentionally strict.  Interpolating a moving fitted shock
    onto a different grid would alter both the liquid and gas inventories, so a
    coupled production run must use the same cell centres in both solvers.
    Gas mass is distributed by resolved horizontal void volume; no gas is
    invented in the initially full downstream branch.
    """

    x_source = np.asarray(solver.x, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    if (
        x_source.shape != x_target.shape
        or not np.allclose(x_source, x_target, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "shock-fitting and network horizontal grids must be identical"
        )
    if not math.isclose(
        float(solver.section.full_area), full_area, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise ValueError(
            "shock-fitting and network pipe areas must be identical"
        )

    area = np.asarray(state.area, dtype=float).copy()
    discharge = np.asarray(state.discharge, dtype=float).copy()
    void_volume_geometric = (
        np.maximum(full_area - np.clip(area, 0.0, full_area), 0.0) * dx
    )
    # The fitted state carries one gas inventory bounded by the moving
    # interface.  Elastic rarefaction cells downstream may also have A_l<A,
    # but they are liquid storage, not disconnected receivers of that gas.
    connected_pocket = x_target < float(state.interface_x)
    void_volume = np.where(connected_pocket, void_volume_geometric, 0.0)
    total_void = float(np.sum(void_volume))
    if total_void <= 1.0e-14:
        raise FloatingPointError(
            "external horizontal state has no resolved gas volume"
        )
    gas_mass = np.asarray(
        float(state.gas.mass) * void_volume / total_void,
        dtype=float,
    )
    # The lumped shock-fitting gas has no resolved momentum field.  Reconstruct
    # the lowest-order kinematically compatible field for a closed left wall:
    # u_g(0)=0 and u_g(x_interface)=dx_interface/dt.  Assigning the interface
    # speed to the complete pocket injected an O(c_g) bulk impulse at handoff.
    interface_x = max(float(state.interface_x), 0.5 * dx)
    gas_velocity = np.where(
        void_volume > 0.0,
        float(state.interface_speed)
        * np.clip(x_target / interface_x, 0.0, 1.0),
        0.0,
    )
    gas_momentum = gas_mass * gas_velocity
    return area, discharge, gas_mass, gas_momentum


def run_network(
    case: NetworkCase,
    verbose: bool = True,
    *,
    external_horizontal_solver=None,
    external_horizontal_checkpoint: str | Path | None = None,
    diagnostic_wall_seconds: float | None = None,
    output_interval: float = 0.02,
) -> dict:
    if not np.isfinite(output_interval) or output_interval <= 0.0:
        raise ValueError("output_interval must be a positive finite number")
    A = case.A; Ar = case.Ar
    a2 = case.a_wh * case.a_wh
    rho_atm = P_ATM / (R_GAS * T_GAS)

    # ---- tunnel grid (horizontal) ----
    Nt = max(20, int(round(case.L_tunnel / case.ds)))
    dx = case.L_tunnel / Nt
    xt = (np.arange(Nt) + 0.5) * dx
    side_t_grid = face_aligned_t_indices(case.x_riser, dx, Nt)
    junction_face = side_t_grid.face
    junction_west_cell = side_t_grid.west_cell
    junction_east_cell = side_t_grid.east_cell
    junction_face_x = side_t_grid.face_x
    iv = int(np.clip(round(case.L_up / dx - 0.5), 1, Nt - 2))      # nearest butterfly-valve cell
    fv = int(np.clip(round(case.L_up / dx), 1, Nt - 1))             # butterfly-valve FACE
    (
        riser_film_thickness,
        riser_laminar_gas_core_fraction,
        riser_terminal_film_flow,
        riser_terminal_film_velocity,
    ) = _vw_laminar_film_closure(
        case.Dr,
        rho_l=RHO_L,
        rho_g=rho_atm,
        mu_l=MU_L,
        gravity=G,
    )
    if not 0.0 < case.vertical_taylor_core_area_fraction < 1.0:
        raise ValueError("vertical Taylor-core area fraction must lie in (0, 1)")
    if not 0.0 < case.vertical_taylor_return_efficiency <= 1.0:
        raise ValueError(
            "vertical Taylor return efficiency must lie in (0, 1]"
        )
    if not 0.0 < case.vertical_ccfl_constant <= 1.0:
        raise ValueError("vertical CCFL constant must lie in (0, 1]")
    # The laminar Nusselt balance is retained as a diagnostic scale, but for
    # this 57-mm side-fed tower it predicts Re_f > 6000 and an unrealistically
    # thin 1-mm film.  The network shock fit therefore uses the frozen
    # side-fed Taylor-core closure, which leaves a finite counter-current film
    # corridor and is also the area cap used by the conservative gas graph.
    riser_gas_core_fraction = min(
        float(riser_laminar_gas_core_fraction),
        float(case.vertical_taylor_core_area_fraction),
    )
    # Report the annular hydraulic scale for audit only.  The resolved falling
    # film uses the Nusselt wall-stress balance below, not Darcy pipe friction
    # based on this equivalent diameter.
    riser_film_hydraulic_diameter = (
        4.0
        * (
            0.25 * math.pi * case.Dr**2
            - 0.25
            * math.pi
            * (case.Dr - 2.0 * riser_film_thickness) ** 2
        )
        / (
            math.pi * case.Dr
            + math.pi * (case.Dr - 2.0 * riser_film_thickness)
        )
    )
    coupled_gas_parameters = CoupledGasParameters(
        horizontal_diameter=case.D,
        vertical_diameter=case.Dr,
        rho_l=RHO_L,
        gravity=G,
        gas_constant=R_GAS,
        gas_temperature=T_GAS,
        atmospheric_pressure=P_ATM,
        horizontal_holdup_drag_enhancement=(
            case.horizontal_holdup_drag_enhancement
        ),
        vertical_gas_core_area_fraction=riser_gas_core_fraction,
        # The fitted Taylor front owns material topology in both vertical
        # momentum formulations; this is independent of the drag owner below.
        vertical_fitted_front_receivers=True,
        # The persistent two-stream riser owns the equal-and-opposite gas/film
        # exchange, so its gas-transport stage must not apply the legacy drag a
        # second time.  The one-stream fallback retains the conservative
        # equal-and-opposite gas/liquid exchange.
        vertical_confined_interface_kinematics=case.enable_vertical_twostream,
        allow_horizontal_front_retreat=case.allow_horizontal_front_retreat,
    )
    # Before directional handoff the legacy liquid column has no separate
    # three-body drag owner, so the conservative gas network must return its
    # equal-and-opposite vertical interphase impulse to that one-stream state.
    # After handoff the persistent two-stream operator owns the same exchange
    # and the gas stage must suppress its legacy copy.  The previous global
    # ``case.enable_vertical_twostream`` switch suppressed drag in both
    # regimes, leaving the pre-handoff film in near free fall.
    coupled_gas_parameters_one_stream = replace(
        coupled_gas_parameters,
        vertical_confined_interface_kinematics=False,
    )
    if external_horizontal_solver is None:
        # Case A has one production horizontal solver.  The locally copied
        # shock-fitting model resolves the valve-release/wet-front stage; once
        # the fitted interface reaches the T, the finite-volume two-fluid
        # equations below take over without remapping or holding the pocket.
        from casea_shockfit_network import build_case_a_shockfit_solver

        external_horizontal_solver = build_case_a_shockfit_solver(
            dx=dx,
            wave_speed=case.a_wh,
        )

    # ---- riser grid (vertical) ----
    Nr = max(20, int(round(case.riser_height / case.dz)))
    dz = case.riser_height / Nr
    zr = (np.arange(Nr) + 0.5) * dz
    twostream_parameters = VerticalTwoStreamParameters(
        cell_count=Nr,
        cell_length=dz,
        diameter=case.Dr,
        liquid_density=RHO_L,
        gravity=G,
        # Wall stress is applied below with the existing Reynolds-aware
        # bulk/film closure separately to each directional stream.  Leaving
        # these constant Darcy slots at zero avoids applying it twice.
        wall_friction_up=0.0,
        wall_friction_down=0.0,
        interstream_drag=0.0,
    )
    twostream_mouth_losses = DirectionalMouthLosses(
        upward_turn=case.junction_loss_coeff,
        downward_turn=case.junction_loss_coeff,
        # The legacy one-stream ``glug_loss_coeff`` represents unresolved
        # gas--liquid churn before directional handoff.  After handoff the gas,
        # upward liquid and falling liquid are advanced by the distributed
        # three-body momentum exchange below.  Applying that empirical glug
        # loss again as a concentrated liquid--liquid mouth force double-counts
        # the churn and can store horizontal pressure until an artificial burst.
        countercurrent_mixing=0.0,
    )
    distributed_tnode_geometry = DistributedTNodeGeometry(
        horizontal_diameter=case.D,
        riser_diameter=case.Dr,
        opening_footprint_length=case.Dr,
        opening_footprint_volume=A * case.Dr,
        gravity=G,
    )
    # ---- initial state (V&W2011): upstream pipe = compressed air pocket; middle+downstream
    #      pipes water-filled; tower water to Y_fs0; both far ends closed ----
    Alt = A * np.ones(Nt)
    capsule = xt < case.L_up                      # air pocket occupies the upstream pipe
    # The upstream capsule is initially a true gas-filled dry reach.  A small
    # numerical liquid film here is visually and physically misleading: it makes
    # the whole closed-end reach appear wetted at valve opening, before the
    # finite-speed wetting front can arrive.
    Alt[capsule] = 0.0
    Qlt = np.zeros(Nt)
    Mgt = (P_ATM / (R_GAS * T_GAS)) * np.maximum(A - Alt, 1e-4 * A) * dx
    Pa0 = P_ATM + RHO_L * G * case.air_head       # initial absolute air-pocket pressure
    Mgt[capsule] = (Pa0 / (R_GAS * T_GAS)) * (A - Alt[capsule]) * dx
    Jgt = np.zeros(Nt)                            # horizontal gas momentum
    external_horizontal_state = None
    external_horizontal_active = False
    external_horizontal_handoff_time = None
    # Latched geometric topology event.  Before it opens, the shock-fitting
    # front owns the horizontal pipe.  At first contact the complete mapped
    # conservative fields are handed to the distributed two-fluid FV graph;
    # the old fitted/lumped pocket is never advanced in parallel afterwards.
    junction_topology_opened = False
    cfg = external_horizontal_solver.config
    if not math.isclose(
        float(cfg.length), case.L_tunnel, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("external horizontal length does not match the case")
    if cfg.vent_x is None or not math.isclose(
        float(cfg.vent_x), case.x_riser, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("external horizontal vent does not match the side-T")
    if int(external_horizontal_solver.junction_face_index) != junction_face:
        raise ValueError(
            "shock-fitting and network solvers disagree on the side-T face"
        )
    external_horizontal_state = (
        external_horizontal_solver.case_b_initial_state(
            initial_air_gauge_head=case.air_head,
            # The shock-fitting section head uses the pipe invert as datum,
            # whereas the experiment reports Y_fs from the pipe crown.
            initial_water_head=case.D + case.init_water_level,
        )
    )
    if external_horizontal_checkpoint is not None:
        checkpoint_path = Path(external_horizontal_checkpoint)
        with np.load(checkpoint_path) as checkpoint:
            checkpoint_area = np.asarray(
                checkpoint["area"], dtype=float
            ).copy()
            checkpoint_discharge = np.asarray(
                checkpoint["discharge"], dtype=float
            ).copy()
            checkpoint_x = np.asarray(checkpoint["x"], dtype=float)
            checkpoint_dx = float(np.ravel(checkpoint["dx"])[0])
            if (
                checkpoint_area.shape
                != external_horizontal_state.area.shape
                or checkpoint_discharge.shape != checkpoint_area.shape
            ):
                raise ValueError(
                    "external horizontal checkpoint grid does not match solver"
                )
            if not np.allclose(
                checkpoint_x,
                external_horizontal_solver.x,
                rtol=0.0,
                atol=2.0e-12,
            ) or not math.isclose(
                checkpoint_dx,
                external_horizontal_solver.dx,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            ):
                raise ValueError(
                    "external horizontal checkpoint coordinates do not match solver"
                )
            checkpoint_gas = replace(
                external_horizontal_state.gas,
                volume=float(np.ravel(checkpoint["gas_volume"])[0]),
                mass=float(np.ravel(checkpoint["gas_mass"])[0]),
            )
            external_horizontal_state = replace(
                external_horizontal_state,
                time=float(np.ravel(checkpoint["time"])[0]),
                area=checkpoint_area,
                discharge=checkpoint_discharge,
                gas=checkpoint_gas,
                air_pressure_abs=float(
                    np.ravel(checkpoint["air_pressure"])[0]
                ),
                interface_x=float(
                    np.ravel(checkpoint["interface_x"])[0]
                ),
                interface_speed=float(
                    np.ravel(checkpoint["interface_speed"])[0]
                ),
                interface_free_surface_depth=float(
                    np.ravel(
                        checkpoint["interface_free_surface_depth"]
                    )[0]
                ),
                interface_free_surface_velocity=float(
                    np.ravel(
                        checkpoint["interface_free_surface_velocity"]
                    )[0]
                ),
                interface_pressurised_head=float(
                    np.ravel(
                        checkpoint["interface_pressurised_head"]
                    )[0]
                ),
                interface_pressurised_velocity=float(
                    np.ravel(
                        checkpoint["interface_pressurised_velocity"]
                    )[0]
                ),
                interface_residual_linf=float(
                    np.ravel(checkpoint["interface_residual_linf"])[0]
                ),
                wetting_front_x=float(
                    np.ravel(checkpoint["wetting_front_x"])[0]
                ),
                vented=False,
                nonlinear_converged=True,
                liquid_volume_residual=float(
                    np.ravel(checkpoint["liquid_volume_error"])[0]
                ),
                cumulative_liquid_volume_residual=float(
                    np.ravel(checkpoint["liquid_volume_error"])[0]
                ),
            )
        if external_horizontal_state.time >= case.t_end:
            raise ValueError(
                "external horizontal checkpoint must precede case.t_end"
            )
    Alt, Qlt, Mgt, Jgt = _map_external_horizontal_state(
        external_horizontal_solver,
        external_horizontal_state,
        x_target=xt,
        full_area=A,
        dx=dx,
    )
    # The fitted horizontal liquid/interface solver remains the owner for the
    # whole run.  Replacing it by a distributed gas-cell FV state at the T was
    # the source of the nonphysical pocket fragmentation seen after 6.5 s.
    external_horizontal_active = True
    gas0 = float(np.sum(Mgt[capsule]))

    Yfs0 = case.init_water_level
    # sharp initial free surface: cells above Yfs0 are DRY (a 1% wall film here is
    # repacked into the column by the surface reconstruction and lifts the initial
    # level by ~1 cm -- a spurious first-frame jump against the paper's Yfs0)
    Alr = Ar * (zr <= Yfs0).astype(float)
    # The frozen 10-mm grid cannot represent Yfs0=0.356 m exactly.  Retain that
    # fixed initial-volume offset only in the reported level reconstruction so
    # the conservative column volume starts at the prescribed physical elevation.
    initial_riser_volume_offset = float(np.sum(Alr) * dz - Ar * Yfs0)
    Qlr = np.zeros(Nr)
    Mgr = (P_ATM / (R_GAS * T_GAS)) * np.maximum(Ar - Alr, 1e-4 * Ar) * dz
    Mgrs = np.zeros(Nr)                           # resolved gas injected from the tunnel
    Jgrs = np.zeros(Nr)                           # resolved gas momentum in the riser
    # Created exactly once when the horizontal gas pocket first opens the
    # measured T mouth.  From then on these four directional inventories are
    # authoritative; Alr/Qlr are only their exact compatibility totals.
    riser_twostream_state: VerticalTwoStreamState | None = None
    riser_twostream_next: VerticalTwoStreamState | None = None
    riser_liquid_provenance_state: (
        VerticalTwoStreamLiquidProvenanceState | None
    ) = None
    riser_liquid_provenance_next: (
        VerticalTwoStreamLiquidProvenanceState | None
    ) = None
    bidirectional_tnode_upward_speed: float | None = None
    taylor_swept_fraction = np.zeros(Nr, dtype=float)
    twostream_activated_time: float | None = None

    geyser_strength = 0.0
    t = float(external_horizontal_state.time)
    step = 0
    dbg_created = dict(t_floor=0.0, r_floor=0.0, r_repack=0.0, consol=0.0, crown=0.0)
    rec = dict(t=[], wtop=[], itop=[], jet_height=[], top_q=[],
               core_mass=[], pocket_head=[], up_head=[], pj_head=[], tr_head=[],
               base_q=[], base_head=[], junction_alpha=[], left_mean_alpha=[],
               right_mean_alpha=[], right_max_alpha=[], right_full_fraction=[],
               tun_gas_mass=[], tun_gas_vol=[], tot_liq=[], tot_liq_raw=[],
               escaped_gas_mass=[], total_resolved_gas_mass=[],
               escaped_liquid_volume=[], total_liquid_including_escape=[],
               frames_t=[], frames_alt=[], frames_alt_raw=[], frames_ult=[], frames_mgt=[], frames_alr=[], frames_alr_raw=[], frames_ulr=[], frames_agr=[], frames_itop=[],
               frames_core_mass=[],
               xt=xt, zr=zr, jx=junction_west_cell, iv=iv, fv=fv,
               junction_face=junction_face,
               junction_west_cell=junction_west_cell,
               junction_east_cell=junction_east_cell,
               junction_face_x=junction_face_x,
               dx=dx, dz=dz, Nt=Nt, Nr=Nr)
    itr = int(np.clip(round(case.x_transducer / dx - 0.5), 0, Nt - 1))   # transducer cell
    out_dt = float(output_interval)

    last_q_up = 0.0
    last_base_head = Yfs0
    last_top_q = 0.0
    jet_height_state = 0.0
    gas_escaped_mass = 0.0
    gas_atmospheric_exchange = 0.0
    vertical_open_headspace_mass_exchange = 0.0
    liquid_escaped_volume = 0.0
    junction_return_requested_volume = 0.0
    junction_return_deposited_volume = 0.0
    junction_return_unplaced_volume = 0.0
    junction_wave_max_source_cells = 0
    annular_film_return_volume = 0.0
    riser_gas_front = 0.0
    riser_gas_front_velocity = 0.0
    riser_entry_cut_front = 0.0
    riser_material_front = 0.0
    riser_breakthrough = False
    last_junction_east_flux = 0.0
    last_junction_west_pressure = P_ATM + RHO_L * G * Yfs0
    last_junction_east_pressure = last_junction_west_pressure
    last_junction_vertical_pressure = last_junction_west_pressure
    last_junction_east_flow = 0.0
    junction_liquid_balance_correction = 0.0
    horizontal_gas_substeps = 0
    horizontal_gas_active_cells = 0
    horizontal_gas_mass_error = 0.0
    horizontal_gas_kinetic_energy = 0.0
    horizontal_gas_center_of_mass = float(xt[0])
    horizontal_gas_maximum_velocity = 0.0
    coupled_gas_maximum_velocity = 0.0
    side_t_east_cut_opened = False
    side_t_east_cut_volume = 0.0
    side_t_east_cut_gas_mass = 0.0
    side_t_east_material_front = float(junction_face_x)
    side_t_east_topology_front = float(junction_face_x)
    side_t_east_material_front_velocity = 0.0
    side_t_east_retired_cell_count = 0
    last_junction_node_pressure = P_ATM + RHO_L * G * Yfs0
    last_junction_west_flow = 0.0
    last_junction_vertical_flow = 0.0
    last_junction_vertical_characteristic_flow = 0.0
    last_junction_taylor_return_flow = 0.0
    last_junction_gas_mouth_fraction = 0.0
    last_junction_gross_upward_flow = 0.0
    last_junction_gross_downward_flow = 0.0
    last_junction_circulation_flow = 0.0
    last_twostream_upward_volume_residual = 0.0
    last_twostream_downward_volume_residual = 0.0
    last_twostream_provenance_volume_residual = 0.0
    last_twostream_horizontal_source_volume = 0.0
    last_twostream_initial_source_volume = float(np.sum(Alr) * dz)
    last_twostream_drag_momentum_residual = 0.0
    last_twostream_bottom_inventory = 0.0
    last_tnode_pressure_residual = 0.0
    last_tnode_pressure_raw_residual = 0.0
    last_tnode_downward_pressure_residual = 0.0
    last_tnode_downward_pressure_raw_residual = 0.0
    last_tnode_capacity_pressure_impulse = 0.0
    last_tnode_capacity_pressure = 0.0
    last_tnode_capacity_upward_rate_correction = 0.0
    last_tnode_capacity_downward_rate_correction = 0.0
    last_tnode_capacity_kkt_residual = 0.0
    last_tnode_capacity_packing_residual = 0.0
    last_tnode_capacity_donor_residual = 0.0
    last_tnode_capacity_donor_multiplier = 0.0
    last_tnode_capacity_active_cells = 0
    last_tnode_capacity_topology_iterations = 0
    last_tnode_momentum_residual = 0.0
    last_tnode_physical_reaction_pressure = 0.0
    last_tnode_vertical_mouth_pressure = float(last_junction_vertical_pressure)
    last_twostream_bottom_pressure = float(last_junction_vertical_pressure)
    last_tnode_fv_mouth_pressure_residual = 0.0
    last_tnode_gas_reaction_requested = 0.0
    last_tnode_gas_reaction_applied = 0.0
    last_tnode_gas_reaction_application_residual = 0.0
    last_tnode_liquid_gas_action_residual = 0.0
    last_combined_interphase_momentum_residual = 0.0
    last_tnode_cell0_drag_length_fraction = 1.0
    last_tnode_horizontal_liquid_pressure = float(last_junction_node_pressure)
    last_tnode_horizontal_liquid_pressure_raw = float(
        last_junction_node_pressure
    )
    last_tnode_vertical_liquid_pressure = float(last_junction_vertical_pressure)
    last_tnode_upward_old_speed = 0.0
    last_tnode_upward_unconstrained_speed = 0.0
    last_tnode_upward_characteristic_speed = 0.0
    last_tnode_upward_characteristic_rate = 0.0
    last_tnode_first_cell_downward_rate = 0.0
    last_tnode_first_cell_downward_speed = 0.0
    last_tnode_outgoing_mouth_downward_rate = 0.0
    last_tnode_positive_net_receiving_capacity = 0.0
    last_tnode_node_liquid_volume = 0.0
    last_tnode_downward_donor_volume = 0.0
    last_tnode_mouth_upward_area = 0.0
    last_tnode_mouth_downward_area = 0.0
    last_tnode_mouth_gas_area = 0.0
    last_tnode_mouth_liquid_area = 0.0
    last_tnode_wallis_downward_reference = 0.0
    last_tnode_downward_constraint_reaction_flux = 0.0
    last_dt_outer = 0.0
    last_dt_phase = 0.0
    last_dt_junction = 0.0

    def append_record(
        sample_t,
        alt_state,
        alr_state,
        mgt_state,
        mgr_total_state,
        mgr_res_state,
        rho_g_state,
    ):
        alpha_g_raw = np.clip(
            mgr_res_state
            / np.maximum(rho_g_state * Ar * dz, 1.0e-12),
            0.0,
            0.90,
        )
        # The gas Riemann solve keeps one axial receiver cell ahead of the
        # fitted material front.  Reconstruct the physical subcell occupancy
        # rather than displaying that numerical buffer as a complete bubble
        # cell.  A fully passed cell keeps its resolved cross-sectional gas
        # fraction; the cut cell is weighted by its axial overlap.
        axial_front_fraction = np.clip(
            (
                riser_gas_front
                - (zr - 0.5 * dz)
            ) / dz,
            0.0,
            1.0,
        )
        alpha_g_raw *= axial_front_fraction
        wtop_now = _column_material_height(
            alr_state, alpha_g_raw, Ar, dz, initial_riser_volume_offset
        )
        # Do not hide resolved gas simply because it has displaced almost all
        # local liquid; the old wet-only mask made compact gas cells invisible.
        active_mask = (alr_state / Ar > 0.02) | (alpha_g_raw > 1.0e-4)
        alpha_g_r = np.where(active_mask, alpha_g_raw, 0.0)
        # The observable free surface is the top of the occupied (liquid +
        # resolved-gas) column.  The volume reconstruction above avoids the
        # 10-mm threshold quantisation of the legacy wet-cell diagnostic.
        gas_idx = np.where(alpha_g_r > 0.02)[0]
        itop_now = (
            min(float(riser_gas_front), case.riser_height)
            if gas_idx.size
            else 0.0
        )
        itop_now = min(itop_now, wtop_now)
        gas_mass = float(np.sum(mgr_res_state * (alpha_g_r > 0.02)))
        ph_up = 0.0
        m_main = 0.0
        # crown-exchange transport keeps gas mass and void collocated, so the pocket
        # regions are simply the connected runs of area-resolved void
        record_pocket = _pocket_mask(
            alt_state, A, mgt_state, dx, gas_min=0.002
        )
        # ``record_pocket`` is the pressure/inventory topology.  Its one-sided
        # tail cell remains active until a retreating fitted front crosses the
        # west face, but material gas occupies only the part west of the
        # continuously tracked front.  Keep these meanings separate: the
        # former closes the conservative gas ledger, while the latter controls
        # the phase field shown to the user and prevents an elastic tail cell
        # from appearing as a permanently pinned air depression.
        record_material_pocket = record_pocket & (
            (xt < float(junction_face_x))
            | (xt <= float(side_t_east_material_front) + 1.0e-12)
        )
        record_void = float(
            np.sum(np.maximum(A - alt_state[record_pocket], 0.0)) * dx
        )
        record_mass = float(np.sum(mgt_state))
        if (
            record_void > EPS
            and record_mass * R_GAS * T_GAS
            >= 0.2 * P_ATM * record_void
        ):
            m_main = record_mass
            ph_up = (
                record_mass * R_GAS * T_GAS / record_void - P_ATM
            ) / (RHO_L * G)
        # While the fitted solver owns the horizontal branch, report its
        # polytropic pocket pressure directly.  Reconstructing pressure from
        # the network's whole-cell void volume would make the t=0 capsule one
        # half cell too long and create a purely diagnostic pressure jump.
        if (
            external_horizontal_state is not None
            and abs(float(external_horizontal_state.time) - float(sample_t))
            <= 1.0e-8 * max(1.0, abs(float(sample_t)))
        ):
            ph_up = float(
                (
                    external_horizontal_state.air_pressure_abs
                    - P_ATM
                )
                / (RHO_L * G)
            )
        rec["t"].append(float(sample_t))
        measured_head = ph_up
        rec["up_head"].append(float(measured_head))
        rec["wtop"].append(float(wtop_now))
        rec["itop"].append(float(itop_now))
        rec["jet_height"].append(float(jet_height_state))
        rec["top_q"].append(float(last_top_q))
        rec["core_mass"].append(float(gas_mass))
        # Report the same main pocket used by the transducer/junction closure.
        # Taking the maximum over every tiny post-vent fragment produces enormous
        # EOS heads from negligible masses and is not a measurable apparatus state.
        rec["pocket_head"].append(float(measured_head))
        rec["base_q"].append(float(last_q_up))
        rec["base_head"].append(float(last_base_head))
        rec["junction_alpha"].append(
            float(alt_state[junction_west_cell] / A)
        )
        left_alpha = np.asarray(
            alt_state[:junction_face] / A, dtype=float
        )
        right_alpha = np.asarray(
            alt_state[junction_face:] / A, dtype=float
        )
        rec["left_mean_alpha"].append(float(np.mean(left_alpha)) if left_alpha.size else 0.0)
        rec["right_mean_alpha"].append(float(np.mean(right_alpha)) if right_alpha.size else 0.0)
        rec["right_max_alpha"].append(float(np.max(right_alpha)) if right_alpha.size else 0.0)
        rec["right_full_fraction"].append(
            float(np.mean(right_alpha >= 0.995)) if right_alpha.size else 0.0
        )
        rec.setdefault("right_branch_gas_mass", []).append(
            float(np.sum(mgt_state[junction_face:]))
        )
        rec["frames_t"].append(float(sample_t))
        # A sub-full TPA area without collocated gas mass is elastic storage in
        # a still liquid-full pipe, not a visible air layer.  Reconstruct the
        # phase plot from the mass-supported pocket topology so the animation
        # does not display pressure rarefactions as spurious water--air waves.
        horizontal_phase_alpha = np.where(
            record_material_pocket,
            np.clip(alt_state / A, 0.0, 1.0),
            1.0,
        )
        rec["frames_alt"].append(horizontal_phase_alpha.copy())
        rec["frames_alt_raw"].append(
            np.asarray(alt_state / A, dtype=float).copy()
        )
        rec["frames_ult"].append(
            np.where(alt_state > 1.0e-9,
                     Qlt / np.maximum(alt_state, EPS), 0.0).copy())
        rec.setdefault("frames_qlt", []).append(
            np.asarray(Qlt, dtype=float).copy()
        )
        rec["frames_mgt"].append(np.asarray(mgt_state, dtype=float).copy())
        rec.setdefault("frames_jgt", []).append(
            np.asarray(Jgt, dtype=float).copy()
        )
        rec["frames_alr"].append(np.clip(alr_state / Ar, 0, 1).copy())
        rec["frames_alr_raw"].append(
            np.asarray(alr_state / Ar, dtype=float).copy()
        )
        if riser_twostream_state is None:
            # Before the physical T mouth opens, embed the legacy one-stream
            # state only for output/audit.  This does not feed the solver.
            legacy_q = np.asarray(Qlr, dtype=float)
            upward_area_record = np.where(legacy_q >= 0.0, alr_state, 0.0)
            downward_area_record = np.where(legacy_q < 0.0, alr_state, 0.0)
            upward_discharge_record = np.maximum(legacy_q, 0.0)
            downward_discharge_record = np.minimum(legacy_q, 0.0)
        else:
            upward_area_record = np.asarray(
                riser_twostream_state.upward_area,
                dtype=float,
            )
            downward_area_record = np.asarray(
                riser_twostream_state.downward_area,
                dtype=float,
            )
            upward_discharge_record = np.asarray(
                riser_twostream_state.upward_discharge,
                dtype=float,
            )
            downward_discharge_record = np.asarray(
                riser_twostream_state.downward_discharge,
                dtype=float,
            )
        rec.setdefault("frames_riser_upward_area", []).append(
            upward_area_record.copy()
        )
        rec.setdefault("frames_riser_downward_area", []).append(
            downward_area_record.copy()
        )
        rec.setdefault("frames_riser_upward_discharge", []).append(
            upward_discharge_record.copy()
        )
        rec.setdefault("frames_riser_downward_discharge", []).append(
            downward_discharge_record.copy()
        )
        if riser_liquid_provenance_state is None:
            horizontal_source_upward_area = np.zeros_like(
                upward_area_record
            )
            horizontal_source_downward_area = np.zeros_like(
                downward_area_record
            )
        else:
            horizontal_source_upward_area = np.asarray(
                riser_liquid_provenance_state.upward_source1_area,
                dtype=float,
            )
            horizontal_source_downward_area = np.asarray(
                riser_liquid_provenance_state.downward_source1_area,
                dtype=float,
            )
        rec.setdefault(
            "frames_riser_horizontal_source_upward_area", []
        ).append(horizontal_source_upward_area.copy())
        rec.setdefault(
            "frames_riser_horizontal_source_downward_area", []
        ).append(horizontal_source_downward_area.copy())
        rec.setdefault("frames_riser_initial_source_area", []).append(
            np.maximum(
                upward_area_record
                + downward_area_record
                - horizontal_source_upward_area
                - horizontal_source_downward_area,
                0.0,
            )
        )
        rec["frames_ulr"].append(
            np.where(
                alr_state > 1.0e-9,
                Qlr / np.maximum(alr_state, 1.0e-2 * Ar),
                0.0,
            ).copy()
        )
        rec["frames_agr"].append(alpha_g_r.copy())
        rec.setdefault("frames_mgr", []).append(
            np.asarray(mgr_total_state, dtype=float).copy()
        )
        rec.setdefault("frames_mgrs", []).append(
            np.asarray(mgr_res_state, dtype=float).copy()
        )
        rec.setdefault("frames_jgrs", []).append(
            np.asarray(Jgrs, dtype=float).copy()
        )
        rec["frames_itop"].append(float(itop_now))
        rec["frames_core_mass"].append(float(gas_mass))
        physical_pocket = record_pocket
        rec["tun_gas_mass"].append(float(np.sum(mgt_state)))
        rec["tun_gas_vol"].append(float(
            np.sum(np.maximum(A - alt_state[physical_pocket], 0.0)) * dx
        ))
        liquid_raw = float(
            np.sum(alt_state) * dx + np.sum(alr_state) * dz
        )
        rec["tot_liq"].append(float(
            np.sum(np.clip(alt_state, 0.0, A)) * dx
            + np.sum(np.clip(alr_state, 0.0, Ar)) * dz
        ))
        rec["tot_liq_raw"].append(liquid_raw)
        rec["escaped_gas_mass"].append(float(gas_escaped_mass))
        rec["total_resolved_gas_mass"].append(
            float(np.sum(mgt_state) + np.sum(mgr_res_state) + gas_escaped_mass)
        )
        rec.setdefault("atmospheric_gas_mass_exchange", []).append(
            float(gas_atmospheric_exchange)
        )
        rec.setdefault("vertical_open_headspace_mass_exchange", []).append(
            float(vertical_open_headspace_mass_exchange)
        )
        rec.setdefault("total_gas_mass_including_atmosphere", []).append(
            float(
                np.sum(mgt_state)
                + np.sum(mgr_total_state)
                + gas_atmospheric_exchange
            )
        )
        rec["escaped_liquid_volume"].append(float(liquid_escaped_volume))
        rec["total_liquid_including_escape"].append(
            float(liquid_raw + liquid_escaped_volume)
        )
        rec.setdefault("junction_return_requested_volume", []).append(
            float(junction_return_requested_volume)
        )
        rec.setdefault("junction_return_deposited_volume", []).append(
            float(junction_return_deposited_volume)
        )
        rec.setdefault("junction_return_unplaced_volume", []).append(
            float(junction_return_unplaced_volume)
        )
        rec.setdefault("junction_wave_max_source_cells", []).append(
            float(junction_wave_max_source_cells)
        )
        rec.setdefault("side_t_east_cut_volume", []).append(
            float(side_t_east_cut_volume)
        )
        rec.setdefault("side_t_east_cut_gas_mass", []).append(
            float(side_t_east_cut_gas_mass)
        )
        rec.setdefault("side_t_east_material_front", []).append(
            float(side_t_east_material_front)
        )
        rec.setdefault("side_t_east_topology_front", []).append(
            float(side_t_east_topology_front)
        )
        rec.setdefault("side_t_east_material_front_velocity", []).append(
            float(side_t_east_material_front_velocity)
        )
        rec.setdefault("side_t_east_retired_cell_count", []).append(
            int(side_t_east_retired_cell_count)
        )
        rec.setdefault("annular_film_return_volume", []).append(
            float(annular_film_return_volume)
        )
        rec.setdefault("riser_gas_front", []).append(
            float(riser_gas_front)
        )
        rec.setdefault("riser_entry_cut_front", []).append(
            float(riser_entry_cut_front)
        )
        rec.setdefault("riser_material_front", []).append(
            float(riser_material_front)
        )
        rec.setdefault("riser_gas_front_velocity", []).append(
            float(riser_gas_front_velocity)
        )
        rec.setdefault("riser_breakthrough", []).append(
            bool(riser_breakthrough)
        )
        rec.setdefault("junction_east_liquid_flux", []).append(
            float(last_junction_east_flux)
        )
        rec.setdefault("junction_liquid_balance_correction", []).append(
            float(junction_liquid_balance_correction)
        )
        rec.setdefault("horizontal_gas_substeps", []).append(
            float(horizontal_gas_substeps)
        )
        rec.setdefault("horizontal_gas_active_cells", []).append(
            float(horizontal_gas_active_cells)
        )
        rec.setdefault("horizontal_gas_mass_error", []).append(
            float(horizontal_gas_mass_error)
        )
        rec.setdefault("horizontal_gas_kinetic_energy", []).append(
            float(horizontal_gas_kinetic_energy)
        )
        rec.setdefault("horizontal_gas_center_of_mass", []).append(
            float(horizontal_gas_center_of_mass)
        )
        rec.setdefault("horizontal_gas_maximum_velocity", []).append(
            float(horizontal_gas_maximum_velocity)
        )
        rec.setdefault("coupled_gas_maximum_velocity", []).append(
            float(coupled_gas_maximum_velocity)
        )
        rec.setdefault("junction_node_head", []).append(
            float((last_junction_node_pressure - P_ATM) / (RHO_L * G))
        )
        rec.setdefault("junction_west_liquid_flux", []).append(
            float(last_junction_west_flow)
        )
        rec.setdefault("junction_east_node_liquid_flux", []).append(
            float(last_junction_east_flow)
        )
        rec.setdefault("junction_vertical_liquid_flux", []).append(
            float(last_junction_vertical_flow)
        )
        rec.setdefault(
            "junction_vertical_characteristic_liquid_flux", []
        ).append(float(last_junction_vertical_characteristic_flow))
        rec.setdefault("junction_taylor_return_liquid_flux", []).append(
            float(last_junction_taylor_return_flow)
        )
        rec.setdefault("junction_gas_mouth_fraction", []).append(
            float(last_junction_gas_mouth_fraction)
        )
        rec.setdefault("junction_gross_upward_liquid_flux", []).append(
            float(last_junction_gross_upward_flow)
        )
        rec.setdefault("junction_gross_downward_liquid_flux", []).append(
            float(last_junction_gross_downward_flow)
        )
        rec.setdefault("junction_countercurrent_circulation_flux", []).append(
            float(last_junction_circulation_flow)
        )
        rec.setdefault("twostream_upward_volume_residual", []).append(
            float(last_twostream_upward_volume_residual)
        )
        rec.setdefault("twostream_downward_volume_residual", []).append(
            float(last_twostream_downward_volume_residual)
        )
        rec.setdefault("twostream_provenance_volume_residual", []).append(
            float(last_twostream_provenance_volume_residual)
        )
        rec.setdefault("twostream_horizontal_source_volume", []).append(
            float(last_twostream_horizontal_source_volume)
        )
        rec.setdefault("twostream_initial_source_volume", []).append(
            float(last_twostream_initial_source_volume)
        )
        rec.setdefault("twostream_drag_momentum_residual", []).append(
            float(last_twostream_drag_momentum_residual)
        )
        rec.setdefault("twostream_bottom_0p1m_inventory", []).append(
            float(last_twostream_bottom_inventory)
        )
        rec.setdefault("twostream_active", []).append(
            bool(riser_twostream_state is not None)
        )
        rec.setdefault("tnode_pressure_balance_residual", []).append(
            float(last_tnode_pressure_residual)
        )
        rec.setdefault("tnode_pressure_raw_residual", []).append(
            float(last_tnode_pressure_raw_residual)
        )
        rec.setdefault("tnode_downward_pressure_balance_residual", []).append(
            float(last_tnode_downward_pressure_residual)
        )
        rec.setdefault("tnode_downward_pressure_raw_residual", []).append(
            float(last_tnode_downward_pressure_raw_residual)
        )
        rec.setdefault("tnode_capacity_pressure_impulse", []).append(
            float(last_tnode_capacity_pressure_impulse)
        )
        rec.setdefault("tnode_capacity_pressure", []).append(
            float(last_tnode_capacity_pressure)
        )
        rec.setdefault("tnode_capacity_upward_rate_correction", []).append(
            float(last_tnode_capacity_upward_rate_correction)
        )
        rec.setdefault("tnode_capacity_downward_rate_correction", []).append(
            float(last_tnode_capacity_downward_rate_correction)
        )
        rec.setdefault("tnode_capacity_kkt_residual", []).append(
            float(last_tnode_capacity_kkt_residual)
        )
        rec.setdefault("tnode_capacity_packing_residual", []).append(
            float(last_tnode_capacity_packing_residual)
        )
        rec.setdefault("tnode_capacity_donor_residual", []).append(
            float(last_tnode_capacity_donor_residual)
        )
        rec.setdefault("tnode_capacity_donor_multiplier", []).append(
            float(last_tnode_capacity_donor_multiplier)
        )
        rec.setdefault("tnode_capacity_active_cells", []).append(
            int(last_tnode_capacity_active_cells)
        )
        rec.setdefault("tnode_capacity_topology_iterations", []).append(
            int(last_tnode_capacity_topology_iterations)
        )
        rec.setdefault("tnode_momentum_balance_residual", []).append(
            float(last_tnode_momentum_residual)
        )
        rec.setdefault("tnode_physical_reaction_pressure", []).append(
            float(last_tnode_physical_reaction_pressure)
        )
        rec.setdefault("tnode_vertical_mouth_pressure", []).append(
            float(last_tnode_vertical_mouth_pressure)
        )
        rec.setdefault("twostream_bottom_pressure", []).append(
            float(last_twostream_bottom_pressure)
        )
        rec.setdefault("tnode_fv_mouth_pressure_residual", []).append(
            float(last_tnode_fv_mouth_pressure_residual)
        )
        rec.setdefault("tnode_gas_reaction_requested", []).append(
            float(last_tnode_gas_reaction_requested)
        )
        rec.setdefault("tnode_gas_reaction_applied", []).append(
            float(last_tnode_gas_reaction_applied)
        )
        rec.setdefault("tnode_gas_reaction_application_residual", []).append(
            float(last_tnode_gas_reaction_application_residual)
        )
        rec.setdefault("tnode_liquid_gas_action_residual", []).append(
            float(last_tnode_liquid_gas_action_residual)
        )
        rec.setdefault("combined_interphase_momentum_residual", []).append(
            float(last_combined_interphase_momentum_residual)
        )
        rec.setdefault("tnode_cell0_drag_length_fraction", []).append(
            float(last_tnode_cell0_drag_length_fraction)
        )
        rec.setdefault("tnode_horizontal_liquid_pressure", []).append(
            float(last_tnode_horizontal_liquid_pressure)
        )
        rec.setdefault("tnode_horizontal_liquid_pressure_raw", []).append(
            float(last_tnode_horizontal_liquid_pressure_raw)
        )
        rec.setdefault("tnode_vertical_liquid_pressure", []).append(
            float(last_tnode_vertical_liquid_pressure)
        )
        rec.setdefault("tnode_upward_old_speed", []).append(
            float(last_tnode_upward_old_speed)
        )
        rec.setdefault("tnode_upward_unconstrained_speed", []).append(
            float(last_tnode_upward_unconstrained_speed)
        )
        rec.setdefault("tnode_upward_characteristic_speed", []).append(
            float(last_tnode_upward_characteristic_speed)
        )
        rec.setdefault("tnode_upward_characteristic_rate", []).append(
            float(last_tnode_upward_characteristic_rate)
        )
        rec.setdefault("tnode_first_cell_downward_rate", []).append(
            float(last_tnode_first_cell_downward_rate)
        )
        rec.setdefault("tnode_first_cell_downward_speed", []).append(
            float(last_tnode_first_cell_downward_speed)
        )
        rec.setdefault("tnode_outgoing_mouth_downward_rate", []).append(
            float(last_tnode_outgoing_mouth_downward_rate)
        )
        rec.setdefault("tnode_positive_net_receiving_capacity", []).append(
            float(last_tnode_positive_net_receiving_capacity)
        )
        rec.setdefault("tnode_node_liquid_volume", []).append(
            float(last_tnode_node_liquid_volume)
        )
        rec.setdefault("tnode_downward_donor_volume", []).append(
            float(last_tnode_downward_donor_volume)
        )
        rec.setdefault("tnode_mouth_upward_area", []).append(
            float(last_tnode_mouth_upward_area)
        )
        rec.setdefault("tnode_mouth_downward_area", []).append(
            float(last_tnode_mouth_downward_area)
        )
        rec.setdefault("tnode_mouth_gas_area", []).append(
            float(last_tnode_mouth_gas_area)
        )
        rec.setdefault("tnode_mouth_liquid_area", []).append(
            float(last_tnode_mouth_liquid_area)
        )
        rec.setdefault("tnode_wallis_downward_reference", []).append(
            float(last_tnode_wallis_downward_reference)
        )
        rec.setdefault("tnode_downward_constraint_reaction_flux", []).append(
            float(last_tnode_downward_constraint_reaction_flux)
        )
        rec.setdefault("junction_west_head", []).append(
            float((last_junction_west_pressure - P_ATM) / (RHO_L * G))
        )
        rec.setdefault("junction_east_head", []).append(
            float((last_junction_east_pressure - P_ATM) / (RHO_L * G))
        )
        rec.setdefault("junction_vertical_head", []).append(
            float((last_junction_vertical_pressure - P_ATM) / (RHO_L * G))
        )
        rec.setdefault("dt_outer", []).append(float(last_dt_outer))
        rec.setdefault("dt_phase_limit", []).append(float(last_dt_phase))
        rec.setdefault("dt_junction_limit", []).append(float(last_dt_junction))

    append_record(
        t,
        Alt,
        Alr,
        Mgt,
        Mgr,
        Mgrs,
        np.full(Nr, rho_atm),
    )
    next_out = t + out_dt

    dt_prev = 0.0
    wall_start = time.perf_counter()
    while t < case.t_end - 1e-12:
        resolved_horizontal_handoff_this_step = False
        twostream_active_step = riser_twostream_state is not None
        twostream_work_state: VerticalTwoStreamState | None = None
        twostream_provenance_work_state: (
            VerticalTwoStreamLiquidProvenanceState | None
        ) = None
        riser_liquid_provenance_next = riser_liquid_provenance_state
        twostream_mouth_plan = None
        twostream_top_boundary: DirectionalBoundaryFlux | None = None
        twostream_sweep_overflow_rate = 0.0
        twostream_handoff_requested = False
        confined_gross_exchange_active = False
        confined_gross_upward_flow = 0.0
        confined_gross_downward_flow = 0.0
        confined_gross_downward_weights = None
        external_gross_exchange_applied = False
        last_tnode_gas_reaction_requested = 0.0
        last_tnode_gas_reaction_applied = 0.0
        last_tnode_gas_reaction_application_residual = 0.0
        last_tnode_liquid_gas_action_residual = 0.0
        last_combined_interphase_momentum_residual = 0.0
        last_tnode_cell0_drag_length_fraction = 1.0
        last_junction_gross_upward_flow = 0.0
        last_junction_gross_downward_flow = 0.0
        last_junction_circulation_flow = 0.0
        bidirectional_tnode_upward_speed_work = (
            bidirectional_tnode_upward_speed
        )
        bidirectional_tnode_upward_speed_next = (
            bidirectional_tnode_upward_speed
        )
        taylor_swept_fraction_next = taylor_swept_fraction.copy()
        liquid_volume_before_step = float(
            np.sum(Alt) * dx + np.sum(Alr) * dz
        )
        junction_wave_active = bool(
            junction_topology_opened
            or (
                external_horizontal_state is not None
                and bool(external_horizontal_state.vented)
            )
        )
        # ---------- pressures (closed system: air-pocket compression + hydrostatic head from
        #            the OPEN tower free surface, which well-balances the resting water column) ----------
        # Gas mass and void MIGRATE TOGETHER (crown-exchange transport below), so a
        # plain area-based region detection suffices: every pocket cell carries both
        # its mass share and its volume.  gas_min=0.05 so the stratified crown layer
        # (a Benjamin cavity fills at most about half the bore) shares the pocket
        # pressure instead of being mistaken for a full-water acoustic cell.
        # NOTE: void-only pocket detection here is safe ONLY because the phantom
        # reclose in the consolidation pass (end of every step) guarantees the
        # state fed to _pressure has mass-consistent voids.  Testing mass per-cell
        # INSIDE _pressure instead fragments the capsule into flickering
        # sub-regions (Rusanov area flux and gas mass never agree cell-by-cell)
        # and the multi-EOS chaos diverges the run within seconds -- tried, reverted.
        if junction_wave_active:
            Pt, wett, _ = _pressure(
                Alt,
                np.full(Nt, A),
                Mgt,
                dx,
                a2,
                vent_top=False,
                p_floor=0.0,
                gas_min=0.002,
                tension_head=TENSION_HEAD,
                mass_consistent=True,
            )
        else:
            Pt, wett, _ = _pressure(
                Alt,
                np.full(Nt, A),
                Mgt,
                dx,
                a2,
                vent_top=False,
                p_floor=0.0,
                gas_min=0.05,
                tension_head=TENSION_HEAD,
                mass_consistent=False,
            )
        Pr, wetr, _ = _pressure(Alr, np.full(Nr, Ar), Mgr, dz, a2, vent_top=True, p_floor=0.0)
        # Use the same conservative material-height reconstruction in the
        # pressure equation and in the reported observable.  A sharp 0.356-m
        # column occupies 36 cells on the 10-mm grid; a wet-cell threshold
        # would therefore feed 0.360 m into the hydrostatic pressure while the
        # output subtracts the 4-mm cut-cell offset.  That inconsistent datum
        # drives a spurious base flow from the first step.  Tunnel-origin gas
        # contributes its resolved volume to the occupied-column height after
        # arrival; before arrival Mgrs is identically zero.
        alpha_g_height = np.clip(
            Mgrs / max(rho_atm * Ar * dz, EPS), 0.0, 0.98
        )
        alpha_g_height *= np.clip(
            (riser_gas_front - (zr - 0.5 * dz)) / dz,
            0.0,
            1.0,
        )
        Yfs = _column_material_height(
            Alr,
            alpha_g_height,
            Ar,
            dz,
            initial_riser_volume_offset,
        )
        # Only TRULY FULL tunnel cells carry the open-tower piezometric head; a cell
        # with ANY resolved crown void has a free surface there, and its section
        # pressure is pinned to the local gas ambient (pocket EOS in a gassy region,
        # ~atmospheric for an isolated micro-void), NOT the tower head.  The old
        # 0.95 threshold left a pressure-INVISIBLE band (alpha 0.95..1): micro-voids
        # opened by hammer transients read the same overlay as full water, felt no
        # closing gradient, and accumulated -- a one-way volume ratchet that drained
        # the tower ~4 cm and held the pocket over-compressed at twice its
        # equilibrium head.  With the pinch, full neighbours (at +rho*g*Yfs) drive
        # water into any micro-void immediately (cavity collapse), keeping the
        # column watertight.
        #
        # The overlay uses the LOW-PASS tower level (Yfs_slow) and a SMOOTH ramp in
        # alpha instead of a 0.999 step.  With the raw values, (a) every full cell's
        # pressure breathed with the instantaneous tower surface -- a global
        # parametric pump feeding the acoustic mode -- and (b) a micro-rarefied cell
        # (alpha 0.995..1) flipped between "atmospheric" and "full + tower head" in
        # alternate steps, a +-3.5 kPa square wave per 0.1% of area (the grid-scale
        # ringing that smeared the pocket).  The ramp makes the restoring force on a
        # micro-void grow continuously as it opens.
        # The shock-fitting/TPA full-pipe area already stores piezometric
        # pressure through H-D.  Add only the circular-section datum D to recover
        # H.  Adding rho*g*Yfs here counted the tower head twice; adding nothing
        # omitted D and made the tunnel under-pressure by one pipe diameter.
        w_full = np.clip((Alt / A - 0.99) / 0.01, 0.0, 1.0)
        Pt = np.where(
            Alt / A >= 0.99,
            Pt + RHO_L * G * case.D * w_full,
            Pt,
        )
        # The ONE partial cell touching each pocket edge (0.001 < alpha_g <= 0.05)
        # is the pocket's nose/tail transition: pressure is continuous across the
        # material contact, so it reads the POCKET pressure.  Leaving it at P_ATM
        # made the interface a pressure DIODE (both neighbours pushed into the band
        # cell whichever way the pocket was off equilibrium), which held the pocket
        # over-compressed ~0.1 m above the tower head indefinitely.  Painting is
        # strictly ONE cell deep: walking the whole partial band erased the collapse
        # gradient of every hammer-rarefied stretch, whose voids then persisted and
        # smeared the cavity across the entire tunnel within ~1.5 s (gas at the
        # tower at T*~1.5 instead of ~7).  Deeper micro-voids keep P_ATM and are
        # collapsed by their full-overlay neighbours (watertight column).
        # Local stratified-layer hydrostatics (the term the pure 1D section pressure
        # was missing): in a gassy cell the gas sits at the CROWN and the water layer
        # of depth h(alpha_l) lies under it, so the INVERT pressure -- the datum the
        # full-water cells, the T-junction (z=0 at invert) and the paper's transducer
        # all live on -- is P_gas + rho*g*h.  Without this term the full seal between
        # the nose and the T balanced the raw gas pressure against the whole tower
        # head and squeezed the pocket up to H* ~ Yfs (0.65L instead of the paper's
        # 0.537L plateau); with it the seal equilibrium is
        #   P_gas = P_atm + rho g (Yfs - h_nose)
        # i.e. the crown-ambient state Benjamin's cavity actually rides on.
        alpha_l_t = Alt / A
        gassy_cells_t = (1.0 - alpha_l_t) > 0.05
        h_layer = case.D * _depth_frac(np.clip(alpha_l_t, 0.0, 1.0))
        # Surface/crown pressure is retained for the tunnel momentum source.
        Pt_surface = Pt.copy()
        Pt = np.where(gassy_cells_t, Pt + RHO_L * G * h_layer, Pt)
        for (i0g, i1g) in _regions(gassy_cells_t):
            if i0g - 1 >= 0 and alpha_l_t[i0g - 1] < 0.999:
                Pt[i0g - 1] = Pt[i0g] + RHO_L * G * (h_layer[i0g - 1] - h_layer[i0g])
                Pt_surface[i0g - 1] = Pt_surface[i0g]
            if i1g < Nt and alpha_l_t[i1g] < 0.999:
                Pt[i1g] = Pt[i1g - 1] + RHO_L * G * (h_layer[i1g] - h_layer[i1g - 1])
                Pt_surface[i1g] = Pt_surface[i1g - 1]
        # Preserve the thermodynamic/interface trace before adding the
        # artificial bulk-viscosity stress.  The latter is an axial numerical
        # momentum flux and must not become a normal pressure pump at the
        # orthogonal riser mouth.
        Pt_surface_without_bulk = Pt_surface.copy()
        # Linear acoustic bulk viscosity in the (near-)full reach: q = -rho*nu_b*du/dx
        # with nu_b ~ a*dx.  This is the standard sub-cell damper for the stiff
        # elastic branch -- it acts ONLY on cell-scale compression/expansion rate
        # (zero in hydrostatic rest and in uniform translation), and kills the
        # grid-frequency ringing of the water column against the pocket spring that
        # no Riemann dissipation reaches in the contact-preserving flux (rarefied
        # band cells otherwise ring underdamped against the stiff overlay and the
        # rectified ringing pumps the tower / smears the pocket).
        # weight: full damping for alpha>=0.97, fading to zero at the pocket-region
        # threshold 0.95 -- pocket cells keep their thermodynamic EOS pressure.
        w_bulk_t = np.clip((Alt / A - 0.95) / 0.02, 0.0, 1.0)
        u_bulk_t = np.clip(Qlt / np.maximum(Alt, 1.0e-3 * A), -U_FLUX_MAX, U_FLUX_MAX)
        div_t = np.zeros(Nt)
        div_t[1:-1] = (u_bulk_t[2:] - u_bulk_t[:-2]) / (2.0 * dx)
        bulk_corr_t = w_bulk_t * RHO_L * (1.0 * case.a_wh * dx) * div_t
        Pt = Pt - bulk_corr_t
        Pt_surface = Pt_surface - bulk_corr_t
        # In the TPA update the circular-section hydrostatic term and the
        # pressurised elastic term are conservative momentum-flux components.
        # Retain only the imposed gas/tower pressure (plus bulk damping) as a
        # separate source, otherwise the elastic pressure is counted twice.
        elastic_dev_t = np.where(
            alpha_l_t >= 1.0,
            alpha_l_t - 1.0,
            -np.minimum(
                1.0 - alpha_l_t,
                TENSION_HEAD * G / max(a2, EPS),
            ),
        )
        horizontal_void_for_flux = np.maximum(
            A - np.clip(Alt, 0.0, A), 0.0
        )
        stratified_supported_t = _mass_backed_gas_topology(
            horizontal_void_for_flux,
            Mgt,
            full_area=A,
            cell_width=dx,
            rho_reference=rho_atm,
            void_floor_fraction=(
                coupled_gas_parameters.void_floor_fraction
            ),
            active_void_fraction=(
                coupled_gas_parameters.active_void_fraction
            ),
            topology_density_fraction=(
                coupled_gas_parameters.topology_density_fraction
            ),
            resolved_density_fraction=(
                coupled_gas_parameters.resolved_density_fraction
            ),
        )
        stratified_supported_t &= (
            horizontal_void_for_flux
            >= (
                coupled_gas_parameters.horizontal_capillary_void_fraction
                * A
            )
        )
        Pt_external = np.where(
            stratified_supported_t,
            Pt_surface,
            Pt_surface - RHO_L * a2 * elastic_dev_t,
        )
        Pt_external_tnode = np.where(
            stratified_supported_t,
            Pt_surface_without_bulk,
            Pt_surface_without_bulk - RHO_L * a2 * elastic_dev_t,
        )
        # The resolved free-surface flux now contains only the exact circular
        # Saint--Venant hydrostatic moment.  Its surface-gas pressure therefore
        # enters once through this regular pressure source.  Gas-free elastic
        # cells retain their external datum after subtracting the conservative
        # water-hammer pressure.
        # Hydrostatic pressure of the resolved vertical liquid column.
        head_r = (
            _vertical_liquid_holdup_head(Alr / Ar, dz)
            if riser_breakthrough
            else np.maximum(Yfs - zr, 0.0)
        )
        Pr = np.where(wetr, Pr + RHO_L * G * head_r, Pr)
        # acoustic bulk viscosity in the riser column (same damper as the tunnel)
        w_bulk_r = np.clip((Alr / Ar - 0.95) / 0.02, 0.0, 1.0)
        u_bulk_r = np.clip(Qlr / np.maximum(Alr, 1.0e-2 * Ar), -U_FLUX_MAX, U_FLUX_MAX)
        div_r = np.zeros(Nr)
        div_r[1:-1] = (u_bulk_r[2:] - u_bulk_r[:-2]) / (2.0 * dz)
        Pr = Pr - w_bulk_r * RHO_L * (1.0 * case.a_wh * dz) * div_r
        # Three-branch characteristic junction solve.  The physical side-T is
        # cut through the pipe crown.  ``Pt`` is the horizontal invert pressure,
        # while ``Pr`` uses z=0 at the crown; convert both horizontal traces to
        # the crown datum before enforcing a common node pressure.  Mixing the
        # two datums introduces a permanent rho*g*D pressure jump and drives the
        # tower even in hydrostatic rest.
        Pt_crown = Pt - RHO_L * G * h_layer

        # The T is attached to a horizontal finite-volume face.  Its west and
        # east traces are therefore the cells immediately adjacent to that
        # face, not a selected cell centre.  This distinction matters after the
        # fitted front arrives: the two horizontal face flows may differ by the
        # simultaneous vertical flow while the node itself stores no volume.
        iw = junction_west_cell
        ie = junction_east_cell
        p_w = float(Pt_crown[iw])
        p_e = float(Pt_crown[ie])
        p_v = float(Pr[0] + RHO_L * G * 0.5 * dz)
        last_junction_west_pressure = p_w
        last_junction_east_pressure = p_e
        last_junction_vertical_pressure = p_v
        u_w_out = -float(Qlt[iw] / max(Alt[iw], 1.0e-3 * A))
        u_e_out = float(Qlt[ie] / max(Alt[ie], 1.0e-3 * A))
        u_v_out = float(Qlr[0] / max(Alr[0], 1.0e-2 * Ar))
        alpha_g_j_pre = float(
            np.clip(1.0 - Alt[junction_west_cell] / A, 0.0, 0.98)
        )
        gas_void_j = max(
            A - Alt[junction_west_cell], 1.0e-4 * A
        )
        gas_density_j = max(
            Mgt[junction_west_cell] / max(gas_void_j * dx, EPS), 0.0
        )
        P_gas_j = gas_density_j * R_GAS * T_GAS
        P_connected_j = P_gas_j
        horizontal_void_raw = np.maximum(
            A - np.clip(Alt, 0.0, A),
            0.0,
        )
        # Use the same material-gas support criterion as the coupled gas graph.
        # The former node test required only 20% atmospheric density, while the
        # connected-pocket mask required 30%.  A newly remapped front cell in
        # that gap was declared to contain gas but its pressure was then read
        # from the isolated low-density receiver (often -3 to -10 m head)
        # instead of from the attached pocket.  That artificial suction drove
        # the riser downward and delayed gas entry by roughly half a second.
        junction_pocket_supported = _mass_backed_gas_topology(
            horizontal_void_raw,
            Mgt,
            full_area=A,
            cell_width=dx,
            rho_reference=rho_atm,
            void_floor_fraction=(
                coupled_gas_parameters.void_floor_fraction
            ),
            active_void_fraction=(
                coupled_gas_parameters.active_void_fraction
            ),
            topology_density_fraction=(
                coupled_gas_parameters.topology_density_fraction
            ),
            resolved_density_fraction=(
                coupled_gas_parameters.resolved_density_fraction
            ),
        )
        junction_pocket_receiver = np.zeros_like(
            junction_pocket_supported
        )
        if junction_pocket_receiver.size > 1:
            junction_pocket_receiver[1:] |= (
                junction_pocket_supported[:-1]
            )
            junction_pocket_receiver[:-1] |= (
                junction_pocket_supported[1:]
            )
        junction_pocket_receiver &= (
            horizontal_void_raw
            > coupled_gas_parameters.horizontal_capillary_void_fraction * A
        )
        junction_pocket_mask = (
            junction_pocket_supported | junction_pocket_receiver
        )
        # A cell can momentarily be packed to the liquid tolerance while still
        # carrying finite connected gas mass.  Declaring the crown mouth shut
        # from ``A_l`` alone then toggles the T pressure owner and creates the
        # nonphysical 8-s stop/restart.  Retain the Young--Laplace crown passage
        # when the local gas inventory can support that capillary volume at the
        # same topology-density criterion used by the gas graph.  Positivity
        # floor mass is below this threshold and cannot open the mouth.
        capillary_horizontal_void = (
            coupled_gas_parameters.horizontal_capillary_void_fraction * A
        )
        capillary_junction_mass = (
            coupled_gas_parameters.topology_density_fraction
            * rho_atm
            * capillary_horizontal_void
            * dx
        )
        junction_capillary_mass_supported = bool(
            Mgt[junction_west_cell] > capillary_junction_mass
        )
        effective_alpha_g_j = max(
            alpha_g_j_pre,
            (
                coupled_gas_parameters.horizontal_capillary_void_fraction
                if junction_capillary_mass_supported
                else 0.0
            ),
        )
        pocket_mass_j, pocket_volume_j = _connected_pocket_inventory(
            Alt,
            Mgt,
            junction_pocket_mask,
            index=junction_west_cell,
            full_area=A,
            cell_width=dx,
        )
        if pocket_volume_j > EPS and pocket_mass_j > 0.0:
            P_connected_j = min(
                pocket_mass_j * R_GAS * T_GAS / pocket_volume_j,
                12.0 * P_ATM,
            )
        horizontal_gas_at_junction = bool(
            junction_wave_active
            and (
                (
                    alpha_g_j_pre > case.tower_entry_alpha_min
                    and junction_pocket_mask[junction_west_cell]
                )
                or junction_capillary_mass_supported
            )
        )

        # The positivity-floor atmospheric mass stored in a nominally full
        # riser cell is not a material gas path to the side tee.  Open the
        # vertical mouth only when tunnel-origin tracer mass supports the base
        # void; once open, the collocated pressure below still uses the complete
        # gas mass ``Mgr``.
        vertical_gas_mouth = _mass_supported_vertical_gas_mouth(
            Alr[0],
            Mgrs[0],
            full_area=Ar,
            cell_width=dz,
            rho_reference=rho_atm,
            density_fraction=(
                coupled_gas_parameters.topology_density_fraction
            ),
            maximum_gas_area_fraction=(
                coupled_gas_parameters.vertical_gas_core_area_fraction
            ),
        )
        vertical_gas_at_junction = bool(
            junction_wave_active and vertical_gas_mouth > 0.0
        )
        gas_at_junction = bool(
            horizontal_gas_at_junction or vertical_gas_at_junction
        )
        if vertical_gas_at_junction:
            vertical_void_raw = np.maximum(
                Ar - np.clip(Alr, 0.0, Ar),
                0.0,
            )
            vertical_tracer_supported = _mass_backed_gas_topology(
                vertical_void_raw,
                Mgrs,
                full_area=Ar,
                cell_width=dz,
                rho_reference=rho_atm,
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                resolved_density_fraction=(
                    coupled_gas_parameters.resolved_density_fraction
                ),
            )
            vertical_component_end = 0
            while (
                vertical_component_end < Nr
                and vertical_tracer_supported[vertical_component_end]
            ):
                vertical_component_end += 1
            vertical_component_end = max(vertical_component_end, 1)
            vertical_component_mass = float(
                np.sum(Mgr[:vertical_component_end])
            )
            vertical_component_volume = float(
                np.sum(vertical_void_raw[:vertical_component_end]) * dz
            )
            # The gas acoustic crossing time of the T fitting is shorter than
            # one outer liquid step.  The normal-stress state seen by the liquid
            # characteristic is therefore the EOS equilibrium of the connected
            # horizontal pocket and the tunnel-origin vertical component, not
            # the pressure of an isolated, newly opened low-density base cell.
            # The latter produced another -10 m suction pulse immediately after
            # the first tracer mass entered the riser.
            common_mass = vertical_component_mass
            common_volume = vertical_component_volume
            if horizontal_gas_at_junction and pocket_volume_j > EPS:
                common_mass += pocket_mass_j
                common_volume += pocket_volume_j
            if common_volume > EPS and common_mass > 0.0:
                P_connected_j = min(
                    common_mass * R_GAS * T_GAS / common_volume,
                    12.0 * P_ATM,
                )
            # Before breakthrough the connected gas is a confined acoustic
            # pocket and the component EOS above is its correct short-time
            # normal stress.  After the material core reaches the open lip,
            # however, gas mass and pressure are already advanced by the
            # distributed vertical gas equation with its vent boundary.  A
            # second whole-component isothermal EOS treats that open column as
            # sealed and generated the spurious -40 to -70 kPa T-mouth suction
            # at 8 s.  Use the collocated resolved base-cell pressure once the
            # gas path is open.
            if (
                riser_breakthrough
                and vertical_void_raw[0] > EPS
                and Mgr[0] > 0.0
            ):
                P_connected_j = min(
                    float(Mgr[0])
                    * R_GAS
                    * T_GAS
                    / max(float(vertical_void_raw[0]) * dz, EPS),
                    12.0 * P_ATM,
                )

        # Gas and liquid share the physical T-mouth area.  Before gas arrival
        # the complete bore belongs to the liquid characteristic.  Once a
        # resolved crown cavity reaches the T, only the complementary area is
        # available to liquid; the gas solver is additionally limited by the
        # actually opened void in the first riser cell.
        horizontal_gas_mouth = (
            junction_mouth_area(
                effective_alpha_g_j, coupled_gas_parameters
            )
            if horizontal_gas_at_junction
            else 0.0
        )
        geometric_gas_mouth = max(
            horizontal_gas_mouth,
            vertical_gas_mouth,
        )
        # Incipient gas entry is a moving-boundary problem: before the first
        # riser cut cell contains finite gas volume, the pressure characteristic
        # must decide whether the side-T gas front points upward or into the
        # liquid-full east dead leg.  The Young--Laplace entry pressure is the
        # only additional threshold.  This local state flag selects a Riemann
        # branch; it prescribes neither a transfer fraction nor a front history.
        capillary_entry_pressure = (
            4.0
            * coupled_gas_parameters.surface_tension
            / max(case.Dr, EPS)
        )
        incipient_vertical_gas_receiving = bool(
            horizontal_gas_at_junction
            and horizontal_gas_mouth > 0.0
            and P_connected_j
            > p_v + capillary_entry_pressure
        )
        resolved_vertical_gas_receiving = bool(
            vertical_gas_at_junction and Jgrs[0] > 0.0
        )
        vertical_gas_receiving = bool(
            incipient_vertical_gas_receiving
            or resolved_vertical_gas_receiving
        )
        last_junction_gas_mouth_fraction = float(
            geometric_gas_mouth / max(Ar, EPS)
        )
        junction_liquid_area = max(Ar - geometric_gas_mouth, 0.0)
        if not gas_at_junction:
            # Before gas arrival the intersection is liquid-full and has
            # negligible storage.  The simultaneous three-branch acoustic
            # characteristics give one conservative pressure and three flows.
            y_h = A / max(RHO_L * case.a_wh, EPS)
            y_v = Ar / max(RHO_L * case.a_wh, EPS)
            Pj = float(
                (
                    y_h * p_w + y_h * p_e + y_v * p_v
                    - A * u_w_out - A * u_e_out - Ar * u_v_out
                )
                / max(2.0 * y_h + y_v, EPS)
            )
            horizontal_west_node_flow = -(
                A * u_w_out + y_h * (Pj - p_w)
            )
            horizontal_east_node_flow = (
                A * u_e_out + y_h * (Pj - p_e)
            )
            riser_wave_speed = case.a_wh
        else:
            # Once gas reaches the T, a zero-volume liquid-only junction can
            # demand pressure below the gas pressure.  The physical intersection
            # instead has finite gas/liquid storage, represented by the local
            # horizontal T control volume.  Its axial faces remain fully active.
            # Normal-stress continuity at the resolved gas/liquid interface.
            # Use the EOS pressure of the complete gas region connected to the
            # T cell.  This is independent of the per-cell free-surface
            # classification threshold and of liquid acoustic damping.
            Pj = float(P_connected_j)
            horizontal_throughflow = 0.5 * (
                float(Qlt[iw]) + float(Qlt[ie])
            )
            horizontal_west_node_flow = horizontal_throughflow
            horizontal_east_node_flow = horizontal_throughflow
            # While the Taylor nose remains confined, the upper liquid slug is
            # still a connected water column and uses the water-hammer
            # characteristic.  After material breakthrough, the mouth contains
            # an open gas core and a free liquid film/churn layer; the relevant
            # branch impedance is then the gravity-wave scale, not the acoustic
            # impedance of a fictitious full-bore water column.
            riser_wave_speed = (
                math.sqrt(G * case.Dr)
                if riser_breakthrough
                else case.a_wh
            )
        u_vertical_characteristic = (
            u_v_out
            + (Pj - p_v) / max(RHO_L * riser_wave_speed, EPS)
        )
        # Physical tee loss, solved analytically and implicitly in velocity:
        # |u| + K|u|^2/(2c) = |u_characteristic|.
        # Once gas and returning water share the mouth, the counter-current
        # churn loss belongs to the boundary characteristic itself.  The same
        # pressure loss is used below in the base momentum ghost, so the face
        # volume and cell momentum see one consistent T-junction condition.
        # Counter-current turn/mixing loss must appear continuously with the
        # local two-phase holdup.  Applying the complete ``glug_loss_coeff`` as
        # soon as a 2% crown void crossed the topology threshold introduced an
        # O(1) jump in K, arrested the riser return flow, and removed the very
        # T-generated gravity waves seen in the 2-D calculation.  The standard
        # symmetric interfacial-area weight 4*alpha_g*alpha_l vanishes in either
        # pure phase and reaches one only at equal holdup; it is a local closure,
        # not a time/window or target-wave prescription.
        if horizontal_gas_at_junction:
            mixing_gas_fraction = alpha_g_j_pre
        elif vertical_gas_at_junction:
            mixing_gas_fraction = geometric_gas_mouth / max(Ar, EPS)
        else:
            mixing_gas_fraction = 0.0
        glug_activation = _two_phase_mixing_activation(
            mixing_gas_fraction
        )
        vertical_gas_density_at_mouth = rho_atm
        vertical_gas_superficial_velocity = 0.0
        if gas_at_junction and (
            case.enable_vertical_twostream or riser_breakthrough
        ):
            vertical_void_at_mouth = max(
                Ar - float(Alr[0]), 1.0e-4 * Ar
            )
            vertical_gas_density_at_mouth = float(
                Mgr[0] / max(vertical_void_at_mouth * dz, EPS)
            )
            vertical_gas_velocity_at_mouth = float(
                Jgrs[0] / max(Mgr[0], EPS)
            )
            vertical_gas_superficial_velocity = max(
                vertical_gas_velocity_at_mouth, 0.0
            ) * vertical_void_at_mouth / max(Ar, EPS)
        # The bubbly/glug loss is a counter-current loss.  It must not be
        # applied when gas and liquid both move upward, because that suppresses
        # the physically resolved horizontal-to-vertical water tongue.  The
        # ordinary ninety-degree tee loss remains active in both directions.
        countercurrent_glug = bool(
            u_vertical_characteristic < 0.0
            and vertical_gas_superficial_velocity > 0.0
        )
        glug_direction_factor = (
            float(countercurrent_glug)
            if case.enable_vertical_twostream
            else 1.0
        )
        loss = max(
            float(case.junction_loss_coeff)
            + float(case.glug_loss_coeff)
            * glug_activation
            * glug_direction_factor,
            0.0,
        )
        speed_characteristic = abs(float(u_vertical_characteristic))
        if loss > 0.0 and speed_characteristic > 0.0:
            speed_node = (
                2.0 * speed_characteristic
                / (
                    1.0
                    + math.sqrt(
                        1.0
                        + 2.0 * loss * speed_characteristic
                        / max(riser_wave_speed, EPS)
                    )
                )
            )
        else:
            speed_node = speed_characteristic
        # The incoming riser characteristic already contains the inertia of
        # the resolved water column.  Solving the minor-loss relation on that
        # characteristic is the consistent hyperbolic boundary condition.
        # The former separate short-fitting ODE counted an additional
        # L=(D+Dr)/2 inertia and integrated the full gas-pressure jump as a
        # local acceleration; it produced 6--7 m/s branch jets and the
        # grid-wide saw-tooth wave train despite exact mass conservation.
        u_vertical_node = math.copysign(
            speed_node, u_vertical_characteristic
        )
        junction_vertical_node_flow = (
            junction_liquid_area * u_vertical_node
        )
        last_junction_vertical_characteristic_flow = float(
            junction_vertical_node_flow
        )
        if riser_breakthrough and gas_at_junction:
            junction_vertical_node_flow = _countercurrent_flooding_liquid_flow(
                junction_vertical_node_flow,
                upward_gas_superficial_velocity=(
                    vertical_gas_superficial_velocity
                ),
                full_area=Ar,
                diameter=case.Dr,
                rho_l=RHO_L,
                rho_g=vertical_gas_density_at_mouth,
                gravity=G,
                wallis_constant=case.vertical_ccfl_constant,
            )
        last_junction_taylor_return_flow = 0.0
        last_junction_node_pressure = float(Pj)
        last_junction_west_flow = float(horizontal_west_node_flow)
        last_junction_east_flow = float(horizontal_east_node_flow)
        last_junction_vertical_flow = float(junction_vertical_node_flow)
        P_base_liq = Pj
        last_base_head = (P_base_liq - P_ATM) / (RHO_L * G)

        # ---------- breakthrough detection (previous-step state) ----------
        # When a continuous resolved-gas core connects the riser base to the free
        # surface AND the junction carries gas, the trapped pocket is no longer sealed
        # by liquid: it is hydraulically open to the atmosphere through the tower.
        # The experiment shows exactly this as the sharp pressure-head collapse (Fig.5
        # at T*~8.3 for the wide tower; Fig.6 at T*~4.05 for the narrow tower, right
        # when the gas front reaches the top).
        rho_g_r_pre = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
        alpha_gr_pre = np.clip(Mgrs / np.maximum(rho_g_r_pre * Ar * dz, 1.0e-12), 0.0, 0.98)
        alpha_gr_pre *= np.clip(
            (riser_gas_front - (zr - 0.5 * dz)) / dz,
            0.0,
            1.0,
        )
        capv_pre = Ar * np.clip(1.0 - alpha_gr_pre, 0.0, 1.0) * dz
        liqv_pre = float(np.sum(np.clip(Alr, 0.0, Ar) * dz))
        ksurf_pre = min(int(np.searchsorted(np.cumsum(capv_pre), liqv_pre)), Nr - 1)
        # The T mouth "carries gas" whenever the cavity supplies the riser at all.
        # The old 20%-void test never fired for the wide tower: the volume-neutral
        # glug exchange keeps refilling the junction cell with descending water, so
        # its resolved void sits at 5-15% during the entire vent -- the breakthrough
        # blow-down below was dead code and the recorded H* held its 0.5 plateau to
        # the end of the run (experiment: collapse to zero at T*~8.3).
        junction_gassy = bool(alpha_g_j_pre > max(case.tower_entry_alpha_min, 0.05)
                              or (1.0 - Alt[junction_west_cell] / A) > 0.20)
        # Two hydraulically-open topologies (both are "the pocket sees atmosphere"):
        #  (a) annular/continuous core -- a resolved-gas column from base to surface
        #      (the narrow tower's channelised vent);
        #  (b) bubbly through-flow -- gas is bursting AT the free surface while the
        #      junction still feeds gas from below (the wide tower's glug train:
        #      discrete Taylor slugs, never a continuous core, yet the pocket drains
        #      to atmosphere through the standing column).  Requiring (a) alone kept
        #      the wide-tower pocket sealed forever: recorded H* stayed on its 0.5
        #      plateau to the end of the run while the experiment collapses to zero
        #      at T*~8.3 -- right when the gas front catches the free surface (the
        #      paper's own reading of Fig.5/Fig.7).
        surface_gassy_pre = bool(
            ksurf_pre >= 1
            and np.any(alpha_gr_pre[max(0, ksurf_pre - 1):min(ksurf_pre + 2, Nr)] > 0.10)
        )
        raw_breakthrough = bool(
            junction_gassy
            and ksurf_pre >= 1
            and (
                np.all(alpha_gr_pre[:ksurf_pre] > 0.20)
                or surface_gassy_pre
            )
        )
        # Once a fitted Taylor nose exists, acoustic receiver/tracer cells
        # ahead of it cannot declare pneumatic breakthrough.  The gas path
        # opens only when that material nose actually reaches the bulk liquid
        # surface; thereafter the state remains latched until its tracer has
        # vented.  This is a topology criterion, not a prescribed event time.
        if case.enable_vertical_twostream:
            # The directional branch is activated only by the fitted material
            # nose.  A diffuse acoustic tracer tail may satisfy the legacy
            # alpha_g test before that nose exists; latching on that tail sets
            # ``already_vented`` and prevents the material front from ever
            # starting, leaving no swept film corridor at the mouth.
            breakthrough = bool(
                riser_breakthrough and riser_material_front > 0.0
            )
        else:
            breakthrough = (
                bool(riser_breakthrough)
                if riser_material_front > 0.0
                else raw_breakthrough
            )
        # This is an instantaneous topology state, not a latched event or a
        # prescribed blow-down time.  It may close again if the resolved gas
        # connection disappears.
        riser_breakthrough = breakthrough
        _bt_dbg = dict(
            bt=breakthrough, jgassy=junction_gassy, ksurf=ksurf_pre,
            a_surf=float(np.max(alpha_gr_pre[max(0, ksurf_pre - 1):min(ksurf_pre + 2, Nr)]))
            if ksurf_pre >= 1 else 0.0,
            a_core_min=float(np.min(alpha_gr_pre[:ksurf_pre])) if ksurf_pre >= 1 else 0.0,
        )

        # ---------- timestep ----------
        # CFL velocities on FLUX-consistent areas (floored like uLf/uRf): raw Q/Al in
        # a nearly-emptied cell is a phantom velocity that only shrinks dt.
        ult = Qlt / np.maximum(Alt, 1.0e-3 * A)
        ulr = Qlr / np.maximum(Alr, 1.0e-2 * Ar)
        if riser_twostream_state is not None:
            twostream_up_velocity = np.divide(
                np.asarray(riser_twostream_state.upward_discharge),
                np.maximum(
                    np.asarray(riser_twostream_state.upward_area),
                    1.0e-3 * Ar,
                ),
            )
            twostream_down_velocity = np.divide(
                np.asarray(riser_twostream_state.downward_discharge),
                np.maximum(
                    np.asarray(riser_twostream_state.downward_area),
                    1.0e-3 * Ar,
                ),
            )
        else:
            twostream_up_velocity = np.zeros_like(ulr)
            twostream_down_velocity = np.zeros_like(ulr)
        ugt_now = np.where(Mgt > 1.0e-14, Jgt / np.maximum(Mgt, 1.0e-14), 0.0)
        ct = np.sqrt(a2 * np.clip((Alt / A - 0.6) / 0.35, 0.0, 1.0) + G * case.D + 1e-6)
        cr = np.sqrt(a2 * np.clip((Alr / Ar - 0.6) / 0.35, 0.0, 1.0) + G * case.Dr + 1e-6)
        # Jgrs is the momentum of the complete vertical gas phase, whereas
        # Mgrs is only the tunnel-origin tracer.  CFL must use total gas mass;
        # dividing by the initially tiny tracer produced fictitious 1e5 m/s
        # velocities as soon as the first trace of pocket gas entered the riser.
        vertical_gas_area_cfl = np.maximum(
            Ar - np.clip(Alr, 0.0, Ar),
            coupled_gas_parameters.void_floor_fraction * Ar,
        )
        vertical_gas_resolved = (
            Mgr > (
                coupled_gas_parameters.resolved_density_fraction
                * rho_atm
                * vertical_gas_area_cfl
                * dz
            )
        ) & (
            Mgr < (
                coupled_gas_parameters.resolved_density_ceiling
                * rho_atm
                * vertical_gas_area_cfl
                * dz
            )
        )
        ugr_now = np.where(
            vertical_gas_resolved,
            Jgrs / np.maximum(Mgr, 1.0e-14),
            0.0,
        )
        gas_wave = math.sqrt(G * case.Dr)
        # The post-arrival horizontal gas block performs its own acoustic CFL
        # subcycling.  Do not force the complete liquid/riser network onto the
        # velocity of a negligible-mass gas tail; doing so changes no gas step
        # but multiplies the expensive outer coupling iterations.
        horizontal_gas_outer_speed = (
            0.0
            if junction_wave_active
            else float(np.max(np.abs(ugt_now)) + gas_wave)
        )
        cfl_horizontal_liquid = float(np.max(np.abs(ult) + ct))
        cfl_vertical_liquid = float(np.max(np.abs(ulr) + cr))
        cfl_twostream_up = float(
            np.max(np.abs(twostream_up_velocity) + cr)
        )
        cfl_twostream_down = float(
            np.max(np.abs(twostream_down_velocity) + cr)
        )
        cfl_vertical_gas = float(np.max(np.abs(ugr_now)) + gas_wave)
        smax = max(
            cfl_horizontal_liquid,
            cfl_vertical_liquid,
            cfl_twostream_up,
            cfl_twostream_down,
            horizontal_gas_outer_speed,
            cfl_vertical_gas,
        )
        if junction_wave_active:
            horizontal_shallow_coefficient = _decoupled_restoring_coefficient(
                Alt,
                Qlt,
                Mgt,
                Jgt,
                area_full=A,
                diameter=case.D,
                cell_width=dx,
            )
            horizontal_shallow_celerity = np.sqrt(
                np.maximum(
                    horizontal_shallow_coefficient * np.maximum(Alt, 0.0),
                    0.0,
                )
                + 1.0e-8
            )
            smax = max(
                smax,
                float(np.max(np.abs(ult) + horizontal_shallow_celerity)),
            )
        # Gas-void positivity is enforced locally on the shared liquid face
        # fluxes below.  It must not impose a non-propagating global timestep.
        dt_phase = float("inf")
        phase_horizontal_flux = None

        # Positivity CFL for the three physical donor branches.  This is a
        # timestep condition, not a post-hoc flux cap: all three fluxes retain
        # the simultaneous characteristic solution and hence still sum to
        # zero.  Limiting only the vertical flux after the node solve was the
        # previous source of a horizontal/vertical mass mismatch.
        # The side-T donor condition is enforced locally on G1[0] after dt is
        # known, using the same limited flux in both connected control volumes.
        # It therefore remains conservative without collapsing the global step
        # when one donor cell becomes nearly empty.
        dt_junction = float("inf")

        dt = min(
            case.cfl * min(dx, dz) / max(smax, EPS),
            dt_phase,
            dt_junction,
            out_dt,
            case.t_end - t,
        )
        if (
            not math.isfinite(dt)
            or dt < 1.0e-9
            or not math.isfinite(smax)
        ):
            horizontal_speed = np.abs(
                Qlt / np.maximum(Alt, 1.0e-9 * A)
            )
            bad_index = int(np.argmax(horizontal_speed))
            raise FloatingPointError(
                "outer CFL collapsed: "
                f"t={t:.12g}, dt={dt:.12g}, smax={smax:.12g}, "
                f"cfl_hl={cfl_horizontal_liquid:.12g}, "
                f"cfl_vl={cfl_vertical_liquid:.12g}, "
                f"cfl_up={cfl_twostream_up:.12g}, "
                f"cfl_down={cfl_twostream_down:.12g}, "
                f"cfl_hg={horizontal_gas_outer_speed:.12g}, "
                f"cfl_vg={cfl_vertical_gas:.12g}, "
                f"cell={bad_index}, x={xt[bad_index]:.12g}, "
                f"alpha_l={Alt[bad_index]/A:.12g}, "
                f"Q_l={Qlt[bad_index]:.12g}, "
                f"u_l={Qlt[bad_index]/max(Alt[bad_index], 1.0e-9*A):.12g}, "
                f"M_g={Mgt[bad_index]:.12g}, "
                f"J_g={Jgt[bad_index]:.12g}"
            )
        last_dt_outer = float(dt)
        last_dt_phase = (
            float(dt_phase) if np.isfinite(dt_phase) else 0.0
        )
        last_dt_junction = (
            float(dt_junction) if np.isfinite(dt_junction) else 0.0
        )

        external_horizontal_next = None
        external_horizontal_commit_state = None
        if external_horizontal_active:
            # Advance the fitted interface over exactly the same physical time
            # increment as the network.  The shock solver performs its own CFL
            # subcycling.  Its conservative fields replace the provisional
            # tunnel update below until the interface reaches the side-T.
            external_horizontal_next = external_horizontal_solver.step(
                external_horizontal_state,
                dt,
                external_pressure_abs=None,
            )

        # ================= TUNNEL update =================
        # Before hand-off the distributed state is provisional, but its face
        # fluxes and donor capacities participate in the coupled T transaction.
        # Keep that stage until the T coupling is reformulated around a single
        # owner; replacing it by a mapped-state average changes the riser event.
        theta_v = min(max(t / max(case.valve_open_time, 1.0e-9), 0.02), 1.0)
        phi_v = theta_v * theta_v
        phi_flow = phi_v
        if junction_wave_active and phase_horizontal_flux is None:
            # SSP-RK2 evaluates its own two stage fluxes.  Do not compute and
            # discard a third identical Riemann pass before entering it.
            Alt_new, Qlt_new, F1, F2 = (
                _advance_horizontal_liquid_hyperbolic_ssprk2(
                    Alt,
                    Qlt,
                    Mgt,
                    Jgt,
                    area_full=A,
                    diameter=case.D,
                    wave_speed=case.a_wh,
                    cell_width=dx,
                    dt=dt,
                    valve_face=fv,
                    valve_transmissivity=phi_flow,
                    junction_wave_active=True,
                    rho_reference=rho_atm,
                    coupled_gas_parameters=coupled_gas_parameters,
                    phase_volume_cfl=case.phase_volume_cfl,
                )
            )
        else:
            if phase_horizontal_flux is None:
                F1, F2, _, _ = _decoupled_liquid_rusanov_flux(
                    Alt,
                    Qlt,
                    Mgt,
                    Jgt,
                    area_full=A,
                    diameter=case.D,
                    wave_speed=case.a_wh,
                    cell_width=dx,
                    minimum_stratified_void_fraction=(
                        coupled_gas_parameters.horizontal_capillary_void_fraction
                    ),
                )
            else:
                F1, F2, _, _ = (
                    component.copy() for component in phase_horizontal_flux
                )
            # Closed walls carry zero liquid volume flux.  During valve opening
            # the turning disc scales the single physical valve face.
            F1[0] = 0.0
            F1[-1] = 0.0
            if phi_flow < 1.0:
                F1[fv] *= phi_flow
                F2[fv] *= phi_flow
            F1, F2 = _limit_liquid_donor_flux(
                Alt,
                F1,
                F2,
                cell_width=dx,
                dt=dt,
                retained_fraction=0.10,
            )
            if junction_wave_active:
                F1, F2 = _limit_gas_void_closure_flux(
                    Alt,
                    Mgt,
                    F1,
                    F2,
                    full_area=A,
                    cell_width=dx,
                    dt=dt,
                    rho_reference=rho_atm,
                    density_fraction=(
                        coupled_gas_parameters.topology_density_fraction
                    ),
                    density_ceiling=(
                        coupled_gas_parameters.resolved_density_ceiling
                    ),
                    void_floor_fraction=(
                        coupled_gas_parameters.void_floor_fraction
                    ),
                    active_void_fraction=(
                        coupled_gas_parameters.active_void_fraction
                    ),
                    closure_fraction=case.phase_volume_cfl,
                )
            Alt_new = Alt - dt / dx * (F1[1:] - F1[:-1])
            Qlt_new = Qlt - dt / dx * (F2[1:] - F2[:-1])
        # The horizontal T control volume retains the ordinary west/east
        # finite-volume fluxes.  The vertical exchange is a conservative side
        # source applied below; therefore no horizontal face is frozen or
        # replaced when the gas pocket reaches the branch.
        last_junction_east_flux = (
            float(F1[junction_face]) if junction_wave_active else 0.0
        )
        Pt_momentum = Pt_external
        dPdx = _branch_consistent_external_pressure_gradient(
            Pt_momentum,
            stratified_supported_t,
            cell_width=dx,
        )
        if phi_flow < 1.0 and 1 <= fv <= Nt - 2:
            dPdx[fv - 1] = (
                phi_flow * (Pt_momentum[fv] - Pt_momentum[fv - 2])
                + (1.0 - phi_flow)
                * (Pt_momentum[fv - 1] - Pt_momentum[fv - 2])
            ) / (2.0 * dx)
            dPdx[fv] = (
                phi_flow * (Pt_momentum[fv + 1] - Pt_momentum[fv - 1])
                + (1.0 - phi_flow)
                * (Pt_momentum[fv + 1] - Pt_momentum[fv])
            ) / (2.0 * dx)
        Dh_t = case.D
        un = np.where(
            Alt_new > 1e-9,
            Qlt_new / np.maximum(Alt_new, EPS),
            0.0,
        )
        ag_t = np.clip(1.0 - Alt_new / A, 0.0, 1.0)
        churn_w = 4.0 * ag_t * (1.0 - ag_t)
        fric_t = (
            32.0 * MU_L / RHO_L / (Dh_t * Dh_t)
            + (
                0.025
                + case.horizontal_churn_friction * churn_w
            )
            / (2.0 * Dh_t)
            * np.abs(un)
        )
        valve_drag = (
            case.valve_loss_coeff / max(phi_flow, 1.0e-4)
        ) / (4.0 * dx)
        fric_t[fv - 1] += valve_drag * abs(un[fv - 1])
        fric_t[fv] += valve_drag * abs(un[fv])
        Qlt_new = (
            Qlt_new - dt * (Alt_new / RHO_L) * dPdx
        ) / (1.0 + dt * fric_t)
        # ================= RISER update (Rusanov) =================
        Arg = np.empty(Nr + 2); Qrg = np.empty(Nr + 2); crg = np.empty(Nr + 2)
        Arg[1:-1] = Alr; Qrg[1:-1] = Qlr; crg[1:-1] = cr
        # base (z=0): OPEN to the tunnel.  The riser column exchanges liquid with the
        # tunnel junction, driven by the junction pressure Pj (which carries the
        # air-pocket overpressure).  A zero-gradient velocity ghost allows through-flow;
        # the physical drive is the pressure source (Prh[0]=Pj below).  At rest
        # Pj = P_atm + rho g Yfs equals the riser-base hydrostatic pressure, so the
        # coupling is well-balanced (no spurious flow); only a genuine pocket
        # overpressure lifts the column, and liquid can also drain back to the tunnel.
        Arg[0] = Alr[0]
        Qrg[0] = Qlr[0]
        crg[0] = cr[0]
        # top open
        Arg[-1] = Alr[-1]; Qrg[-1] = Qlr[-1]; crg[-1] = cr[-1]
        uRf = np.clip(Qrg / np.maximum(Arg, 1.0e-2 * Ar), -U_FLUX_MAX, U_FLUX_MAX)
        # The axial water/air surface is a material contact.  A Rusanov area
        # flux diffuses that contact even at u=0, numerically painting water all
        # the way to the tower rim while leaving a near-vacuum at the base.
        # Advect liquid area and its momentum with the resolved face velocity:
        # a stationary free surface then has exactly zero flux.
        u_face_r = 0.5 * (uRf[:-1] + uRf[1:])
        donor_area_r = np.where(
            u_face_r >= 0.0,
            Arg[:-1],
            Arg[1:],
        )
        donor_velocity_r = np.where(
            u_face_r >= 0.0,
            uRf[:-1],
            uRf[1:],
        )
        G1 = u_face_r * donor_area_r
        G2 = (
            G1 * donor_velocity_r
            + _riser_acoustic_momentum_dissipation_flux(
                Qlr,
                cr,
                Alr,
                full_area=Ar,
            )
        )
        # The tower opens into air, not an exterior liquid reservoir.  A
        # negative reconstructed top flux would import water from the copied
        # ghost state and make the network liquid inventory grow.  Conversely,
        # a tiny cell-centred liquid remnant at the top is not a physical
        # overflow path while the bulk material surface remains below the lip.
        # Gate only the liquid boundary; gas remains open to atmosphere.
        bulk_material_at_outlet = _bulk_material_reaches_riser_outlet(
            Alr,
            alpha_gr_pre,
            full_area=Ar,
            cell_width=dz,
            riser_height=case.riser_height,
            initial_volume_offset=initial_riser_volume_offset,
        )
        if G1[-1] < 0.0 or not bulk_material_at_outlet:
            G1[-1] = 0.0
            G2[-1] = 0.0
        # Liquid characteristic boundary at the shared T node.  Extrapolate
        # the first riser-cell pressure down by half a cell and use the incoming
        # water-hammer characteristic to determine the single signed face
        # velocity.  This applies the pressure jump once (through rho*a
        # impedance) instead of both as a ghost-cell force and an unconstrained
        # Rusanov flux.
        p_riser_at_node = float(Pr[0] + RHO_L * G * 0.5 * dz)
        vertical_two_phase_mouth_pressure = _vertical_two_phase_mouth_pressure(
            liquid_trace_pressure=p_riser_at_node,
            connected_gas_pressure=float(P_connected_j),
            gas_mouth_area=float(geometric_gas_mouth),
            full_area=Ar,
        )
        u_riser_inner = float(
            Qlr[0] / max(Alr[0], 1.0e-2 * Ar)
        )
        u_t_liquid = (
            junction_vertical_node_flow
            / max(junction_liquid_area, EPS)
            if junction_liquid_area > EPS
            else 0.0
        )
        G1[0] = junction_liquid_area * u_t_liquid
        G2[0] = G1[0] * u_t_liquid
        vertical_tracer_presence_mass = 0.02 * rho_atm * Ar * dz
        vertical_tracer_present = bool(
            float(np.sum(Mgrs)) > vertical_tracer_presence_mass
        )
        # Do not impose ``Q_f = -A_core U_inf`` at the riser base.  That identity
        # is a zero-net-flow Taylor-bubble balance, whereas this side-connected
        # tower has a finite upper slug, a compressible gas pocket, and a
        # bidirectional liquid connection at the T.  For Case A it also implies
        # a 1-mm *laminar* film moving at about 3.4 m/s, outside the credible
        # regime of the Nusselt-film assumption.  Prescribing that value emptied
        # the riser before the 2-D junction oscillation developed.
        #
        # The fitted Taylor nose below is retained only as unresolved interface
        # geometry.  The signed base flow remains the characteristic value above
        # and is determined by resolved inertia, local pressure, T loss, wall
        # shear, and positivity constraints.  The Nusselt scale remains a wall-
        # friction/diagnostic scale, never a second boundary condition.
        G1, G2 = _limit_liquid_donor_flux(
            Alr,
            G1,
            G2,
            cell_width=dz,
            dt=dt,
            retained_fraction=0.10,
        )
        # Positive bottom flow is donated by the finite-width horizontal T
        # footprint, which is external to the vertical donor limiter.
        side_t_weights = _side_t_opening_weights(
            Nt,
            cell_width=dx,
            junction_center=case.x_riser,
            opening_width=case.Dr,
        )
        if G1[0] > 0.0:
            requested_bottom_flow = float(G1[0])
            limited_bottom_flow = _limit_side_t_upward_liquid_flow(
                Alt_new,
                requested_flow=requested_bottom_flow,
                opening_weights=side_t_weights,
                dt=dt,
                cell_width=dx,
                retained_fraction=0.10,
            )
            G1[0] = limited_bottom_flow
            if abs(requested_bottom_flow) > EPS:
                G2[0] *= limited_bottom_flow / requested_bottom_flow
        if junction_wave_active:
            G1, G2 = _limit_gas_void_closure_flux(
                Alr,
                Mgr,
                G1,
                G2,
                full_area=Ar,
                cell_width=dz,
                dt=dt,
                rho_reference=rho_atm,
                density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                density_ceiling=(
                    coupled_gas_parameters.resolved_density_ceiling
                ),
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                closure_fraction=case.phase_volume_cfl,
            )
        if G1[0] < 0.0 and junction_wave_active:
            requested_bottom_flow = float(G1[0])
            limited_bottom_flow = _limit_side_t_downward_liquid_flow(
                Alt_new,
                Mgt,
                requested_flow=requested_bottom_flow,
                opening_weights=side_t_weights,
                dt=dt,
                cell_width=dx,
                full_area=A,
                rho_reference=rho_atm,
                density_ceiling=(
                    coupled_gas_parameters.resolved_density_ceiling
                ),
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
            )
            G1[0] = limited_bottom_flow
            if abs(requested_bottom_flow) > EPS:
                G2[0] *= limited_bottom_flow / requested_bottom_flow
        if G1[0] > 0.0:
            requested_bottom_flow = float(G1[0])
            limited_bottom_flow = _limit_riser_bottom_inflow_by_receiving_capacity(
                Alr,
                G1,
                requested_flow=requested_bottom_flow,
                dt=dt,
                cell_width=dz,
                full_area=Ar,
            )
            G1[0] = limited_bottom_flow
            if abs(requested_bottom_flow) > EPS:
                G2[0] *= limited_bottom_flow / requested_bottom_flow
        # Outside a fitted Taylor sweep the actual bottom flux remains the
        # conservative characteristic flux computed above.  During a sweep,
        # the sub-grid Taylor film/entrainment relation below replaces that
        # one face flux; it is never added as a second drainage mechanism.

        junction_vertical_node_flow = float(G1[0])
        last_junction_vertical_flow = float(G1[0])
        Alr_new = Alr - dt / dz * (G1[1:] - G1[:-1])
        Qlr_new = Qlr - dt / dz * (G2[1:] - G2[:-1])
        # The vertical pressure field remains a distributed two-fluid finite-
        # volume solve, but its gas/liquid *material contact* is shock fitted.
        # Without this distinction the acoustic Roe flux transports a tiny
        # tracer tail through the complete tower and the liquid contact breaks
        # into spray-like fragments.  The fitted Taylor nose instead advances
        # with the standard drift relation.  Only the newly swept core volume
        # is opened; its liquid returns counter-currently through the side tee.
        material_return_velocity = 0.0
        material_front_velocity = 0.0
        active_taylor_core_fraction = riser_gas_core_fraction
        if (
            horizontal_gas_at_junction
            and (
                incipient_vertical_gas_receiving
                or riser_material_front > 0.0
            )
            and geometric_gas_mouth > 0.0
        ):
            old_material_front = float(riser_material_front)
            (
                proposed_material_front,
                material_front_velocity,
                material_front_reached_surface,
            ) = _advance_riser_taylor_front(
                old_material_front,
                free_surface_height=float(Yfs),
                # Before breakthrough the tower has no mixture outflow at the
                # open top.  Side-fed gas rises while the displaced annular
                # film returns through the same T, so the cross-section-
                # averaged mixture superficial velocity is zero.  Adding the
                # upward velocity of the upper liquid slug here double counts
                # displacement and makes the nose about twice too fast.
                liquid_superficial_velocity=0.0,
                diameter=case.Dr,
                riser_height=case.riser_height,
                dt=dt,
                already_vented=bool(riser_breakthrough),
            )
            active_taylor_core_fraction = min(
                riser_gas_core_fraction,
                geometric_gas_mouth / max(Ar, EPS),
            )
            swept_height = max(
                proposed_material_front - old_material_front, 0.0
            )
            if swept_height > 0.0 and not riser_breakthrough:
                # Gas-core displacement supplies the counter-current liquid
                # film before breakthrough.  ``return_efficiency`` is the
                # fraction that drains through the T; its complement remains
                # in the connected upper liquid slug as entrained holdup.
                # Represent the resulting drainage as the *single* liquid
                # face flux.  It is not added to the characteristic flux and
                # therefore cannot count the same returned volume twice.
                material_displacement_flow = (
                    active_taylor_core_fraction
                    * Ar
                    * swept_height
                    / dt
                )
                requested_material_return_flow = (
                    case.vertical_taylor_return_efficiency
                    * material_displacement_flow
                )
                # Before the material nose meets the free surface, the V&W
                # control-volume relation is the confined Taylor displacement
                # balance.  CCFL belongs to the open, post-breakthrough
                # topology and is not a second limiter on this same flux.
                material_return_flow = (
                    min(
                        requested_material_return_flow,
                        riser_terminal_film_flow,
                    )
                    if case.enable_vertical_twostream
                    else requested_material_return_flow
                )
                characteristic_upward_preview = max(float(G1[0]), 0.0)
                preview_horizontal_area, _ = (
                    _apply_finite_width_side_t_exchange(
                        Alt_new,
                        Qlt_new,
                        upward_flow=characteristic_upward_preview,
                        opening_weights=side_t_weights,
                        dt=dt,
                        cell_width=dx,
                        full_area=A,
                    )
                )
                return_exchange_weights = _side_t_return_exchange_weights(
                    preview_horizontal_area,
                    side_t_weights,
                    full_area=A,
                    gas_supported=_mass_backed_gas_topology(
                        np.maximum(
                            A - np.clip(preview_horizontal_area, 0.0, A),
                            0.0,
                        ),
                        Mgt,
                        full_area=A,
                        cell_width=dx,
                        rho_reference=rho_atm,
                        void_floor_fraction=(
                            coupled_gas_parameters.void_floor_fraction
                        ),
                        active_void_fraction=(
                            coupled_gas_parameters.active_void_fraction
                        ),
                        topology_density_fraction=(
                            coupled_gas_parameters.topology_density_fraction
                        ),
                        resolved_density_fraction=(
                            coupled_gas_parameters.resolved_density_fraction
                        ),
                    ),
                )
                # The return enters the finite horizontal T footprint, so its
                # first admissibility limit is the actual receiving capacity
                # there.  The liquid donor is not the bottom riser cell: it is
                # the axial slice newly swept by the Taylor nose this step.
                limited_signed_return = _limit_side_t_downward_liquid_flow(
                    preview_horizontal_area,
                    Mgt,
                    requested_flow=-material_return_flow,
                    opening_weights=return_exchange_weights,
                    dt=dt,
                    cell_width=dx,
                    full_area=A,
                    rho_reference=rho_atm,
                    density_ceiling=(
                        coupled_gas_parameters.resolved_density_ceiling
                    ),
                    void_floor_fraction=(
                        coupled_gas_parameters.void_floor_fraction
                    ),
                    active_void_fraction=(
                        coupled_gas_parameters.active_void_fraction
                    ),
                    topology_density_fraction=(
                        coupled_gas_parameters.topology_density_fraction
                    ),
                )
                material_return_flow = max(-limited_signed_return, 0.0)

                # The Taylor closure replaces the characteristic bottom face
                # during a confined sweep.  First undo that face's contribution
                # to the provisional FV update, then remove exactly the locally
                # swept return volume.  Depositing the same actual flow over the
                # horizontal mouth below closes the complete network balance.
                old_bottom_volume_flux = float(G1[0])
                old_bottom_momentum_flux = float(G2[0])
                # Build an interior-only provisional state before the remote
                # sweep transaction.  Both signs of the old face contribution
                # are first undone.  Once the newly swept original-water parcel
                # has been removed, the accepted horizontal inflow is injected
                # into cell 0.  This ordering prevents new horizontal water
                # entering during this step from being immediately selected as
                # Taylor-return donor liquid.
                characteristic_upward_flow = max(
                    old_bottom_volume_flux,
                    0.0,
                )
                characteristic_upward_momentum_flux = (
                    old_bottom_momentum_flux
                    if old_bottom_volume_flux > 0.0
                    else 0.0
                )
                Alr_new[0] += (
                    dt / dz * (0.0 - old_bottom_volume_flux)
                )
                Qlr_new[0] += (
                    dt / dz * (0.0 - old_bottom_momentum_flux)
                )
                effective_return_core_fraction = (
                    active_taylor_core_fraction
                    * material_return_flow
                    / max(material_displacement_flow, EPS)
                )
                (
                    Alr_new,
                    Qlr_new,
                    returned_volume,
                    _,
                ) = _return_new_taylor_sweep_to_side_t(
                    Alr_new,
                    Qlr_new,
                    old_front_height=old_material_front,
                    new_front_height=proposed_material_front,
                    gas_core_area_fraction=effective_return_core_fraction,
                    full_area=Ar,
                    dz=dz,
                )
                Alr_new[0] += (
                    dt / dz * characteristic_upward_flow
                )
                Qlr_new[0] += (
                    dt / dz * characteristic_upward_momentum_flux
                )
                material_return_flow = returned_volume / dt
                last_junction_taylor_return_flow = float(material_return_flow)
                film_velocity = -material_return_flow / max(
                    (1.0 - active_taylor_core_fraction) * Ar,
                    EPS,
                )
                confined_gross_exchange_active = True
                confined_gross_upward_flow = characteristic_upward_flow
                confined_gross_downward_flow = material_return_flow
                confined_gross_downward_weights = (
                    return_exchange_weights.copy()
                )
                G1[0] = (
                    confined_gross_upward_flow
                    - confined_gross_downward_flow
                )
                # The remote return already removed its collocated vertical
                # momentum from the swept slice.  Keep only the true bottom
                # inflow momentum on this compatibility face; adding a second
                # film term here would double count it.
                G2[0] = characteristic_upward_momentum_flux
                material_return_velocity = film_velocity
                junction_vertical_node_flow = float(G1[0])
                last_junction_vertical_flow = float(G1[0])
                last_junction_gross_upward_flow = float(
                    confined_gross_upward_flow
                )
                last_junction_gross_downward_flow = float(
                    confined_gross_downward_flow
                )
                last_junction_circulation_flow = float(
                    min(
                        confined_gross_upward_flow,
                        confined_gross_downward_flow,
                    )
                )
            if proposed_material_front > old_material_front + 1.0e-15:
                riser_material_front = proposed_material_front
                if not vertical_gas_at_junction:
                    riser_entry_cut_front = riser_material_front
        else:
            material_front_reached_surface = bool(riser_breakthrough)

        if (
            riser_twostream_state is None
            and riser_material_front > 0.0
            and not riser_breakthrough
            and not material_front_reached_surface
        ):
            # Keep the fitted material interface sharp without inventing a
            # second physical return flux.  Refill is moved back into the
            # connected upper slug with its momentum; the already accepted
            # side-T Taylor displacement remains the only mouth transaction.
            Alr_new, Qlr_new, _ = (
                _restore_refilled_taylor_core_to_unswept_slug(
                    Alr_new,
                    Qlr_new,
                    front_height=riser_material_front,
                    # A section-averaged grid resolves the churn/entrainment
                    # layer rather than a perfectly sharp 80/20 interface.
                    # Use the symmetric two-phase interfacial activation
                    # 4*alpha_g*alpha_l as the resolved sharpening fraction;
                    # the remaining excess is retained as prognostic holdup.
                    gas_core_area_fraction=(
                        active_taylor_core_fraction
                        * _two_phase_mixing_activation(
                            active_taylor_core_fraction
                        )
                    ),
                    full_area=Ar,
                    dz=dz,
                )
            )

        material_front_confined = bool(
            riser_material_front > 0.0
            and riser_material_front < float(Yfs) - 1.0e-12
            and not material_front_reached_surface
            and not riser_breakthrough
        )
        material_front_tracked = bool(
            riser_material_front > 0.0
            and (
                horizontal_gas_at_junction
                or vertical_tracer_present
                or riser_breakthrough
            )
        )
        if (
            not twostream_active_step
            and material_front_tracked
            and (
                material_front_confined
                or (
                    material_front_reached_surface
                    and not riser_breakthrough
                )
            )
        ):
            # The upper water body is one connected slug separated from the
            # fitted Taylor core.  Sharpen that contact before either liquid
            # pressure work or gas topology is evaluated, so both operators
            # see the same physical void.  The former post-gas ordering could
            # close a receiver after gas mass had already entered it.
            Alr_new, Qlr_new = _sharpen_unswept_riser_liquid_slug(
                Alr_new,
                Qlr_new,
                material_front_height=riser_material_front,
                full_area=Ar,
                dz=dz,
            )
        # The conservative gas Riemann solve below fills exactly this opened
        # volume; the fitted front restricts tracer topology but does not
        # replace gas pressure or momentum equations.
        Prh = np.empty(Nr + 2); Prh[1:-1] = Pr
        # Base ghost cell centre sits at z = -dz/2, so its well-balanced hydrostatic
        # value is P_base + rho*g*(dz/2).  Without the half-cell offset the bottom
        # cell feels a residual -0.25*rho*g body force that slowly and unphysically
        # DRAINS the tower into the tunnel (seen as the sagging free surface and the
        # over-compressed upstream pocket in earlier runs).
        #
        # The liquid bottom remains a well-balanced hydrostatic opening.  Pocket
        # overpressure acts on the gas-entry equation below, not as a piston under
        # the entire liquid column; applying Pj here made the sealed horizontal
        # capsule lift the tower to its rim before any gas reached the T-junction.
        # The tower surface therefore moves through conservative liquid exchange
        # and, after arrival, through the volume displaced by resolved rising gas.
        # T-junction minor loss on the base exchange (physical entry/exit loss,
        # K = junction_loss_coeff): opposes the base through-flow and damps the
        # residual column oscillation the quasi-steady coupling still admits.
        # While the junction carries gas the exchange is counter-current glug
        # (air up / water down through the same small mouth) -- bubbly churn is
        # far more dissipative than a clean liquid tee, so the loss coefficient
        # steps up to junction_loss_coeff + glug_loss_coeff.  This is what pins
        # the tower column onto the pocket spring after arrival (the experiment's
        # flat Yfs / flat H* plateau); without it the column rings at ~1.5 Hz
        # with +-0.25 L of head from arrival to the end of the run.
        u_base = float(np.clip(Qlr[0] / max(Alr[0], 1.0e-2 * Ar), -U_FLUX_MAX, U_FLUX_MAX))
        base_countercurrent_glug = bool(
            u_base < 0.0
            and vertical_gas_superficial_velocity > 0.0
        )
        base_glug_direction_factor = (
            float(base_countercurrent_glug)
            if case.enable_vertical_twostream
            else 1.0
        )
        K_base = (
            case.junction_loss_coeff
            + case.glug_loss_coeff
            * glug_activation
            * base_glug_direction_factor
        )
        liquid_base_pressure = P_base_liq
        Prh[0] = (liquid_base_pressure + RHO_L * G * (0.5 * dz)
                  - 0.5 * RHO_L * K_base * u_base * abs(u_base))
        Prh[-1] = P_ATM
        dPdz = (Prh[2:] - Prh[:-2]) / (2.0 * dz)
        # Gas momentum uses its own resolved thermodynamic pressure.  The bottom
        # ghost is the local horizontal gas pressure and the top is atmospheric;
        # no extra driving head is added.
        gas_void_r_raw = np.maximum(
            Ar - np.clip(Alr, 0.0, Ar),
            0.0,
        )
        gas_void_r = np.maximum(
            gas_void_r_raw,
            coupled_gas_parameters.void_floor_fraction * Ar,
        )
        material_pressure_cell = _riser_material_gas_mask(
            gas_void_r_raw,
            Mgrs,
            full_area=Ar,
            cell_length=dz,
            reference_density=rho_atm,
            void_floor_fraction=(
                coupled_gas_parameters.void_floor_fraction
            ),
            active_void_fraction=(
                coupled_gas_parameters.active_void_fraction
            ),
            topology_density_fraction=(
                coupled_gas_parameters.topology_density_fraction
            ),
            resolved_density_fraction=(
                coupled_gas_parameters.resolved_density_fraction
            ),
        )
        # Positivity mass stored in a liquid-full cell is not a pneumatic
        # control volume.  Only material-tracer-backed void receives an EOS
        # pressure; the open atmospheric component and unsupported microvoids
        # retain the common atmospheric trace.  This prevents a floor volume
        # from turning harmless background mass into a multi-megapascal source
        # in the neighbouring resolved film.
        P_gas_r = np.full(Nr, P_ATM, dtype=float)
        P_gas_r[material_pressure_cell] = (
            np.maximum(Mgr[material_pressure_cell], 0.0)
            * R_GAS
            * T_GAS
            / np.maximum(gas_void_r[material_pressure_cell] * dz, EPS)
        )
        Prh_gas = np.empty(Nr + 2)
        Prh_gas[1:-1] = P_gas_r
        Prh_gas[0] = P_gas_j if gas_at_junction else P_ATM
        Prh_gas[-1] = P_ATM
        dPdz_gas = (Prh_gas[2:] - Prh_gas[:-2]) / (2.0 * dz)
        # In an annular Taylor-film cell the liquid does not carry a
        # cross-section-filling hydrostatic pressure column.  Its normal
        # stress is the resolved gas/interface pressure, while gravity drives
        # the wall film downward.  Using ``dPdz`` there cancelled gravity and
        # left a motionless 5-cm-equivalent film suspended to t=13 s.  Bulk
        # liquid/slug cells retain the well-balanced liquid pressure gradient.
        pressure_film_cell = material_pressure_cell & (
            zr <= max(riser_gas_front, riser_material_front) + dz
        ) & (
            Alr_new <= 0.50 * Ar
        )
        # ``dPdz_gas`` contains the complementary liquid-buoyancy potential
        # used by the conservative gas momentum equation while the Taylor
        # core is confined.  Reusing that complete gradient as a second
        # liquid normal-stress gradient counts the same ``rho_l g`` coupling
        # twice: the confined gas settles to dP/dz ~= +rho_l g and the film is
        # then accelerated downward by both that numerical potential and its
        # own weight.  In the Case-A handoff this produced a 4--5 m/s falling
        # column before the directional owner even started.  Couple the
        # liquid film to the corresponding reduced interfacial pressure,
        # p_r = p_g - rho_l g z, while the fitted material core remains
        # confined.  Once the core is open, the gas solver already removes
        # this buoyancy-potential branch and no reduction is applied here.
        confined_material_core = bool(
            riser_material_front > 0.0
            and not riser_breakthrough
            and not material_front_reached_surface
        )
        reduced_gas_pressure_gradient = dPdz_gas.copy()
        if confined_material_core:
            reduced_gas_pressure_gradient[material_pressure_cell] -= (
                RHO_L * G
            )
        liquid_pressure_gradient = np.where(
            pressure_film_cell,
            reduced_gas_pressure_gradient,
            dPdz,
        )
        Qlr_new += dt * (
            -(Alr_new / RHO_L) * liquid_pressure_gradient
            - Alr_new * G
        )

        # ---------- persistent two-liquid T/riser topology ----------
        # The crossing step remains wholly owned by the one-stream operator.
        # Its final conservative state is mapped exactly once at commit below;
        # the directional FV operator starts on the following step.  This
        # prevents the same crossing interval from being advanced twice.
        if riser_twostream_state is not None:
            if bidirectional_tnode_upward_speed_work is None:
                raise FloatingPointError(
                    "two-stream riser has no persistent upward T-node speed"
                )
            if riser_liquid_provenance_state is None:
                raise FloatingPointError(
                    "two-stream riser has no persistent liquid provenance"
                )
            twostream_active_step = True
            twostream_work_state = riser_twostream_state
            # Reserve the thermodynamic volume occupied by conserved material
            # gas in every riser cell.  The former rule protected only cell 0
            # with a fixed Taylor fraction; cells 1+ could therefore be filled
            # to ``A_r`` in one falling-film step, deleting their gas corridor
            # and creating the six-cell numerical water plug seen after
            # 7.60 s.  This capacity is rebuilt from the current total gas
            # mass, tunnel tracer and liquid-side normal-stress prediction. It
            # is neither a fixed 80% core nor a saved target profile.
            dynamic_void_capacity = compute_dynamic_material_void_capacity(
                gas_mass=Mgr,
                tracer_mass=Mgrs,
                liquid_pressure_target=np.maximum(Pr, 0.10 * P_ATM),
                current_liquid_area=twostream_work_state.liquid_area,
                full_area=Ar,
                cell_length=dz,
                gas_constant=R_GAS,
                gas_temperature=T_GAS,
                tracer_mass_tolerance=0.0,
                area_tolerance=twostream_parameters.packing_tolerance,
                minimum_topology_void_area=(
                    coupled_gas_parameters.vertical_capillary_core_fraction
                    * Ar
                    * np.clip(taylor_swept_fraction_next, 0.0, 1.0)
                ),
            )
            twostream_liquid_capacity_area = (
                dynamic_void_capacity.liquid_capacity_area
            )
            twostream_provenance_work_state = (
                riser_liquid_provenance_state
            )
            mouth_swept_tolerance = max(
                twostream_parameters.packing_tolerance / max(Ar, EPS),
                128.0 * np.finfo(float).eps,
            )
            if taylor_swept_fraction_next[0] < 1.0 - mouth_swept_tolerance:
                raise FloatingPointError(
                    "active post-breakthrough T mouth lost its completed "
                    "Taylor sweep ledger"
                )
            new_swept_fraction = np.clip(
                (
                    float(riser_material_front)
                    - np.arange(Nr, dtype=float) * dz
                )
                / dz,
                0.0,
                1.0,
            )
            if not riser_breakthrough and np.any(
                new_swept_fraction
                > taylor_swept_fraction + 128.0 * np.finfo(float).eps
            ):
                sweep_geometry = advance_taylor_sweep_geometry(
                    twostream_work_state,
                    twostream_parameters,
                    previous_swept_fraction=taylor_swept_fraction,
                    new_swept_fraction=new_swept_fraction,
                    taylor_core_area_fraction=riser_gas_core_fraction,
                    taylor_rise_velocity=max(
                        float(material_front_velocity),
                        0.345 * math.sqrt(G * case.Dr),
                    ),
                )
                displacement = sweep_geometry.gas_core_displacement
                if displacement.source_shortfall_volume > 1.0e-13:
                    raise FloatingPointError(
                        "Taylor sweep could not open its resolved gas-core volume"
                    )
                twostream_work_state = sweep_geometry.state
                if displacement.overflow_liquid_volume > 1.0e-13:
                    raise FloatingPointError(
                        "Taylor sweep produced liquid overflow before the open-rim boundary solve"
                    )
                twostream_sweep_overflow_rate = 0.0
                taylor_swept_fraction_next = new_swept_fraction.copy()

            resolved_mouth_upward_area = float(
                twostream_work_state.upward_area[0]
            )
            mouth_upward_discharge = float(
                twostream_work_state.upward_discharge[0]
            )
            resolved_mouth_downward_area = float(
                twostream_work_state.downward_area[0]
            )
            mouth_downward_discharge = float(
                twostream_work_state.downward_discharge[0]
            )
            # The horizontal crown opening and the resolved vertical cut-cell
            # void are apertures in series.  Their shared gas face is their
            # overlap, not ``max(horizontal, vertical)``: the latter reserved
            # a Taylor-core-sized gas area while the first riser cell was still
            # mostly liquid and artificially squeezed both liquid traces.  The
            # vertical void is admitted only when backed by tracer gas mass or
            # by the currently swept Taylor cut cell.  The isolated closure
            # retains the resolved falling film, assigns remaining wet entrance
            # contact to the upward side-T stream, and closes
            # A_up+A_down+A_g=Ar without changing a cell inventory.
            mouth_phase_areas = resolve_tnode_mouth_phase_areas(
                resolved_upward_area=resolved_mouth_upward_area,
                resolved_downward_area=resolved_mouth_downward_area,
                horizontal_gas_opening_area=horizontal_gas_mouth,
                vertical_tracer_gas_mass=float(Mgrs[0]),
                full_area=Ar,
                vertical_cell_length=dz,
                reference_gas_density=rho_atm,
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                taylor_swept_fraction=float(
                    taylor_swept_fraction_next[0]
                ),
                taylor_core_area_fraction=riser_gas_core_fraction,
                area_tolerance=twostream_parameters.packing_tolerance,
            )
            mouth_gas_area = mouth_phase_areas.gas_area
            mouth_liquid_area = mouth_phase_areas.liquid_area
            mouth_upward_area = mouth_phase_areas.upward_area
            mouth_downward_area = mouth_phase_areas.downward_area
            # The pre-activation characteristic uses the union-like geometric
            # opening to detect incipient gas entry.  Once the persistent
            # two-stream state owns the tee, however, the horizontal and
            # vertical apertures are in series and ``mouth_gas_area`` above is
            # their actual shared opening.  Select the normal-stress owner from
            # that resolved partition.  Reusing ``geometric_gas_mouth`` here
            # applies connected-gas pressure even when the shared gas aperture
            # is closed; the T node then blocks gross exchange while the FV
            # bottom face continues to accelerate the trapped liquid with a
            # fictitious gas-pressure head.
            vertical_two_phase_mouth_pressure = (
                _vertical_two_phase_mouth_pressure(
                    liquid_trace_pressure=p_riser_at_node,
                    connected_gas_pressure=float(P_connected_j),
                    gas_mouth_area=float(mouth_gas_area),
                    full_area=Ar,
                )
            )
            last_junction_gas_mouth_fraction = float(
                mouth_gas_area / max(Ar, EPS)
            )
            node_liquid_volume = measured_footprint_liquid_inventory(
                np.maximum(Alt_new, 0.0),
                side_t_weights,
                geometry=distributed_tnode_geometry,
            )
            downward_donor_volume = max(
                float(twostream_work_state.downward_area[0]) * dz,
                0.0,
            )
            mouth_gas_density = max(
                float(P_connected_j / (R_GAS * T_GAS)),
                1.0e-9,
            )
            horizontal_liquid_pressure_raw = float(
                np.sum(side_t_weights * Pt_crown)
            )
            # The horizontal FV operator already carries its elastic
            # water-hammer pressure in the conservative momentum flux.  The
            # short orthogonal fitting therefore sees the regular external
            # crown-pressure trace, not a second copy of that elastic term.
            # Averaging ``Pt_crown`` here made the closed east stub's elastic
            # reflection (about 15 m of head in the coarse Case-A run) act as
            # a permanent pump into the riser even though the connected gas
            # and the 2-D mouth pressure remain near atmospheric.  The
            # orthogonal trace also excludes ``bulk_corr_t``: that term stays
            # in the horizontal axial momentum source, but it is a numerical
            # viscous stress rather than physical normal pressure on the riser.
            horizontal_liquid_pressure = float(
                np.sum(side_t_weights * Pt_external_tnode)
            )
            # Use the same liquid-inventory weighting as the footprint update:
            # the ratio of weighted discharge to weighted wet area is the
            # parcel velocity whose axial momentum is actually removed or
            # returned.  An arithmetic mean of cell velocities would overvalue
            # nearly dry cells and break the node/footprint ledger identity.
            horizontal_parcel_liquid_area = float(
                np.sum(side_t_weights * np.maximum(Alt_new, 0.0))
            )
            horizontal_axial_velocity = (
                float(np.sum(side_t_weights * Qlt_new))
                / horizontal_parcel_liquid_area
                if horizontal_parcel_liquid_area > EPS
                else 0.0
            )
            twostream_top_boundary = atmospheric_top_liquid_outflow(
                twostream_work_state,
                twostream_parameters,
                atmospheric_pressure=P_ATM,
            ).flux
            raw_downward_speed = (
                max(-mouth_downward_discharge, 0.0)
                / resolved_mouth_downward_area
                if resolved_mouth_downward_area
                > twostream_parameters.dry_area_tolerance
                else 0.0
            )
            coupled_mouth_characteristic = (
                solve_coupled_gross_mouth_characteristics(
                    old_upward_speed=bidirectional_tnode_upward_speed_work,
                    raw_downward_speed=raw_downward_speed,
                    upward_area=mouth_upward_area,
                    downward_area=mouth_downward_area,
                    horizontal_liquid_pressure_abs=horizontal_liquid_pressure,
                    # This is the same z=0 normal-stress trace used by the
                    # vertical FV bottom face below.  Once the gas aperture is
                    # open it is the connected-gas interface pressure; using
                    # the cell-centred liquid-column pressure here joined two
                    # different pressure traces at one geometric face.  The
                    # mismatch was grid dependent and could suppress the
                    # incoming horizontal-water tongue until the footprint
                    # overfilled and launched an artificial water-hammer pulse.
                    vertical_liquid_pressure_abs=(
                        vertical_two_phase_mouth_pressure
                    ),
                    liquid_density=RHO_L,
                    effective_inertance_length=(
                        distributed_tnode_geometry.effective_inertance_length
                    ),
                    time_step=dt,
                    # The falling film/churn trace is a free-surface outgoing
                    # characteristic.  Its impedance is the gravity-wave scale;
                    # using the donor material speed as a celerity makes a fast
                    # falling parcel artificially resistant to the resolved
                    # turn loss.
                    downward_characteristic_celerity=math.sqrt(G * case.Dr),
                    upward_turn_loss_coefficient=(
                        twostream_mouth_losses.upward_turn
                    ),
                    downward_turn_loss_coefficient=(
                        twostream_mouth_losses.downward_turn
                    ),
                    countercurrent_mixing_coefficient=(
                        twostream_mouth_losses.countercurrent_mixing
                    ),
                    dry_area_tolerance=(
                        twostream_parameters.dry_area_tolerance
                    ),
                )
            )
            last_tnode_horizontal_liquid_pressure = float(
                horizontal_liquid_pressure
            )
            last_tnode_horizontal_liquid_pressure_raw = float(
                horizontal_liquid_pressure_raw
            )
            last_tnode_vertical_liquid_pressure = float(
                vertical_two_phase_mouth_pressure
            )
            last_tnode_upward_old_speed = float(
                coupled_mouth_characteristic.old_upward_speed
            )
            last_tnode_upward_unconstrained_speed = float(
                coupled_mouth_characteristic.uncoupled_upward_speed
            )
            last_tnode_upward_characteristic_speed = float(
                coupled_mouth_characteristic.upward_speed
            )
            last_tnode_upward_characteristic_rate = float(
                coupled_mouth_characteristic.upward_speed
                * mouth_upward_area
            )
            mouth_gas_mass_flow = max(float(Jgrs[0] / dz), 0.0)
            wallis_active = bool(
                mouth_gas_area > twostream_parameters.dry_area_tolerance
                and mouth_gas_mass_flow > 0.0
            )
            wallis_gas_superficial_velocity = 0.0
            wallis_gas_parameter = 0.0
            wallis_downward_reference = math.inf
            if wallis_active:
                wallis_gas_superficial_velocity = (
                    mouth_gas_mass_flow / (mouth_gas_density * Ar)
                )
                density_difference = max(RHO_L - mouth_gas_density, EPS)
                wallis_gas_parameter = (
                    wallis_gas_superficial_velocity
                    * math.sqrt(
                        mouth_gas_density
                        / (G * case.Dr * density_difference)
                    )
                )
                wallis_remaining = max(
                    case.vertical_ccfl_constant
                    - math.sqrt(max(wallis_gas_parameter, 0.0)),
                    0.0,
                )
                wallis_downward_reference = (
                    wallis_remaining**2
                    * math.sqrt(
                        G
                        * case.Dr
                        * density_difference
                        / RHO_L
                    )
                    * Ar
                )
            # The first-cell falling stream is an outgoing characteristic;
            # prescribing a second, independent T-node velocity at that face
            # over-determines the FV boundary and reflects the falling film.
            # Form the shared gross trace from the independent upward inlet and
            # the resolved downward donor.  The same trace is committed to the
            # vertical FV owner and the horizontal footprint below.
            candidate_bottom_riemann = resolve_bottom_mouth_riemann(
                incoming_upward_characteristic_rate=(
                    coupled_mouth_characteristic.upward_speed
                    * mouth_upward_area
                ),
                liquid_area_capacity=float(mouth_liquid_area),
                incoming_upward_characteristic_speed=(
                    coupled_mouth_characteristic.upward_speed
                ),
                first_cell_downward_area=float(
                    resolved_mouth_downward_area
                ),
                first_cell_downward_discharge=float(
                    mouth_downward_discharge
                ),
                resolved_downward_mouth_area=float(mouth_downward_area),
                physical_downward_mouth_speed=(
                    coupled_mouth_characteristic.downward_speed
                ),
                downward_physical_reaction_flux=(
                    coupled_mouth_characteristic.downward_turn_reaction_flux
                    + coupled_mouth_characteristic.mixing_kinematic_reaction_flux
                ),
                finite_node_liquid_volume=node_liquid_volume,
                riser_downward_donor_volume=downward_donor_volume,
                time_step=dt,
                # Full-column capacity and its conjugate pressure are solved
                # below with the two gross mouth rates still variable.  A
                # scalar pre-clip here would destroy the pressure closure.
                positive_net_receiving_capacity=math.inf,
                wallis_downward_capacity=float(
                    wallis_downward_reference
                ),
                enforce_wallis_constraint=False,
                dry_area_tolerance=twostream_parameters.dry_area_tolerance,
            )
            preserve_capacity_partition = np.asarray(
                taylor_swept_fraction_next > mouth_swept_tolerance,
                dtype=bool,
            )
            preserve_capacity_partition[0] = True
            candidate_flux = candidate_bottom_riemann.flux
            candidate_upward_rate = float(candidate_flux.upward_rate)
            candidate_downward_rate = float(candidate_flux.downward_rate)
            dry_mouth_area = twostream_parameters.dry_area_tolerance
            upward_characteristic_area = float(mouth_upward_area)
            downward_characteristic_area = float(mouth_downward_area)
            downward_linear_coefficient = math.sqrt(G * case.Dr)
            upward_flux_inertance = flux_inertance_from_plug(
                liquid_density=RHO_L,
                effective_length=(
                    distributed_tnode_geometry.effective_inertance_length
                ),
                flow_area=max(upward_characteristic_area, dry_mouth_area),
            )
            downward_flux_inertance = flux_inertance_from_characteristic(
                liquid_density=RHO_L,
                celerity=downward_linear_coefficient,
                time_step=dt,
                flow_area=max(downward_characteristic_area, dry_mouth_area),
            )
            recoupled_capacity = project_state_mouth_and_capacity_pressure(
                twostream_work_state,
                preserve_stopped_partition=preserve_capacity_partition,
                candidate_bottom_upward_rate=candidate_upward_rate,
                candidate_bottom_downward_rate=candidate_downward_rate,
                bottom_upward_flux_inertance=upward_flux_inertance,
                bottom_downward_flux_inertance=downward_flux_inertance,
                bottom_upward_characteristic_area=upward_characteristic_area,
                bottom_downward_characteristic_area=downward_characteristic_area,
                bottom_downward_donor_rate_capacity=None,
                top_downward_rate=float(twostream_top_boundary.downward_rate),
                liquid_capacity_area=twostream_liquid_capacity_area,
                dt=dt,
                dz=dz,
                liquid_density=RHO_L,
                bottom_reaction_area=float(mouth_liquid_area),
                directional_area_tolerance=dry_mouth_area,
            )
            if twostream_provenance_work_state is None:
                raise FloatingPointError(
                    "capacity recoupling has no liquid-provenance owner"
                )
            for topology_transfer in recoupled_capacity.topology_transfers:
                provenance_topology = (
                    conservative_liquid_provenance_topology_transfer(
                        twostream_provenance_work_state,
                        topology_transfer,
                        area_tolerance=(
                            twostream_parameters.packing_tolerance
                        ),
                    )
                )
                twostream_provenance_work_state = provenance_topology.state
            twostream_work_state = recoupled_capacity.state
            capacity_projection = recoupled_capacity.projection
            last_tnode_capacity_pressure_impulse = float(
                capacity_projection.ledger.bottom_capacity_pressure_impulse
            )
            last_tnode_capacity_pressure = float(
                last_tnode_capacity_pressure_impulse / dt
            )
            last_tnode_capacity_upward_rate_correction = float(
                capacity_projection.final_bottom_upward_rate
                - capacity_projection.candidate_bottom_upward_rate
            )
            last_tnode_capacity_downward_rate_correction = float(
                capacity_projection.final_bottom_downward_rate
                - capacity_projection.candidate_bottom_downward_rate
            )
            last_tnode_capacity_kkt_residual = float(
                max(
                    capacity_projection.maximum_kkt_stationarity_residual,
                    capacity_projection.maximum_complementarity_residual,
                )
            )
            last_tnode_capacity_packing_residual = float(
                max(
                    capacity_projection.maximum_packing_residual,
                    capacity_projection.maximum_bound_residual,
                )
            )
            last_tnode_capacity_donor_residual = float(
                capacity_projection.maximum_downward_donor_residual
            )
            last_tnode_capacity_donor_multiplier = float(
                capacity_projection.downward_upper_bound_multiplier
            )
            last_tnode_capacity_active_cells = int(
                np.count_nonzero(capacity_projection.active_capacity_mask)
            )
            last_tnode_capacity_topology_iterations = int(
                recoupled_capacity.outer_iterations
            )
            accepted_upward_candidate = float(
                capacity_projection.final_bottom_upward_rate
            )
            accepted_downward_candidate = float(
                capacity_projection.final_bottom_downward_rate
            )
            accepted_upward_candidate_speed = (
                accepted_upward_candidate / mouth_upward_area
                if mouth_upward_area
                > twostream_parameters.dry_area_tolerance
                else 0.0
            )
            accepted_downward_candidate_speed = (
                accepted_downward_candidate / mouth_downward_area
                if mouth_downward_area
                > twostream_parameters.dry_area_tolerance
                else 0.0
            )
            final_mixing_reaction_flux = (
                0.5
                * twostream_mouth_losses.countercurrent_mixing
                * min(
                    accepted_upward_candidate,
                    accepted_downward_candidate,
                )
                * (
                    accepted_upward_candidate_speed
                    + accepted_downward_candidate_speed
                )
            )
            final_downward_reaction_flux = (
                0.5
                * twostream_mouth_losses.downward_turn
                * accepted_downward_candidate
                * accepted_downward_candidate_speed
                + final_mixing_reaction_flux
            )
            corrected_downward_donor_volume = max(
                float(twostream_work_state.downward_area[0]) * dz,
                0.0,
            )
            bottom_riemann = resolve_bottom_mouth_riemann(
                incoming_upward_characteristic_rate=(
                    accepted_upward_candidate
                ),
                liquid_area_capacity=float(mouth_liquid_area),
                incoming_upward_characteristic_speed=(
                    accepted_upward_candidate_speed
                ),
                first_cell_downward_area=float(
                    twostream_work_state.downward_area[0]
                ),
                first_cell_downward_discharge=float(
                    twostream_work_state.downward_discharge[0]
                ),
                resolved_downward_mouth_area=float(mouth_downward_area),
                physical_downward_mouth_speed=(
                    accepted_downward_candidate_speed
                ),
                downward_physical_reaction_flux=(
                    final_downward_reaction_flux
                ),
                finite_node_liquid_volume=node_liquid_volume,
                riser_downward_donor_volume=(
                    corrected_downward_donor_volume
                ),
                time_step=dt,
                positive_net_receiving_capacity=math.inf,
                wallis_downward_capacity=float(
                    wallis_downward_reference
                ),
                enforce_wallis_constraint=False,
                dry_area_tolerance=twostream_parameters.dry_area_tolerance,
            )
            downward_donor_volume = corrected_downward_donor_volume
            accepted_flux = bottom_riemann.flux
            last_tnode_first_cell_downward_rate = float(
                bottom_riemann.ledger.first_cell_downward_rate
            )
            last_tnode_first_cell_downward_speed = float(
                bottom_riemann.ledger.first_cell_downward_speed
            )
            last_tnode_outgoing_mouth_downward_rate = float(
                bottom_riemann.ledger.outgoing_mouth_downward_rate
            )
            last_tnode_positive_net_receiving_capacity = float(
                bottom_riemann.ledger.positive_net_receiving_capacity
            )
            last_tnode_node_liquid_volume = float(node_liquid_volume)
            last_tnode_downward_donor_volume = float(downward_donor_volume)
            last_tnode_mouth_upward_area = float(mouth_upward_area)
            last_tnode_mouth_downward_area = float(mouth_downward_area)
            last_tnode_mouth_gas_area = float(mouth_gas_area)
            last_tnode_mouth_liquid_area = float(mouth_liquid_area)
            last_tnode_wallis_downward_reference = float(
                wallis_downward_reference
                if math.isfinite(wallis_downward_reference)
                else 0.0
            )
            last_tnode_downward_constraint_reaction_flux = float(
                bottom_riemann.ledger.downward_constraint_reaction_flux
            )
            accepted_upward_flow = float(accepted_flux.upward_rate)
            accepted_downward_flow = float(accepted_flux.downward_rate)
            accepted_upward_velocity = float(accepted_flux.upward_speed)
            accepted_downward_speed = float(accepted_flux.downward_speed)
            accepted_circulation = min(
                accepted_upward_flow,
                accepted_downward_flow,
            )
            accepted_liquid_area = (
                bottom_riemann.upward_area
                + bottom_riemann.downward_area
            )
            gross_convective_momentum = RHO_L * (
                accepted_upward_flow * accepted_upward_velocity
                + accepted_downward_flow * accepted_downward_speed
            )
            bulk_convective_momentum = (
                RHO_L
                * bottom_riemann.q_net**2
                / max(accepted_liquid_area, EPS)
            )
            gross_kinetic_power = 0.5 * RHO_L * (
                accepted_upward_flow * accepted_upward_velocity**2
                + accepted_downward_flow * accepted_downward_speed**2
            )
            signed_kinetic_flux = 0.5 * RHO_L * (
                accepted_upward_flow * accepted_upward_velocity**2
                - accepted_downward_flow * accepted_downward_speed**2
            )
            upward_loss_power = (
                0.5
                * RHO_L
                * twostream_mouth_losses.upward_turn
                * accepted_upward_flow
                * accepted_upward_velocity**2
            )
            downward_loss_power = (
                0.5
                * RHO_L
                * twostream_mouth_losses.downward_turn
                * accepted_downward_flow
                * accepted_downward_speed**2
            )
            mixing_loss_power = (
                0.5
                * RHO_L
                * twostream_mouth_losses.countercurrent_mixing
                * accepted_circulation
                * (
                    accepted_upward_velocity
                    + accepted_downward_speed
                ) ** 2
            )
            mouth_radius = 0.5 * case.Dr
            film_inner_radius = math.sqrt(
                max(
                    mouth_radius * mouth_radius
                    - bottom_riemann.downward_area / math.pi,
                    0.0,
                )
            )
            mouth_film_thickness = mouth_radius - film_inner_radius
            mouth_nusselt_velocity = (
                max(RHO_L - mouth_gas_density, 0.0)
                * G
                * mouth_film_thickness**2
                / (3.0 * MU_L)
            )
            mouth_gravity_film_capacity = (
                bottom_riemann.downward_area * mouth_nusselt_velocity
            )
            accepted_exchange = TwoChannelMouthResult(
                q_net=float(bottom_riemann.q_net),
                upward_flow=accepted_upward_flow,
                downward_flow=accepted_downward_flow,
                circulation_flow=float(accepted_circulation),
                closure_residual=float(
                    accepted_upward_flow
                    - accepted_downward_flow
                    - bottom_riemann.q_net
                ),
                film_thickness=float(mouth_film_thickness),
                gravity_film_capacity=float(mouth_gravity_film_capacity),
                wallis_downward_capacity=float(wallis_downward_reference),
                downward_physical_capacity=float(
                    bottom_riemann.ledger.outgoing_mouth_downward_rate
                ),
                downward_physical_circulation_capacity=float(
                    max(
                        bottom_riemann.ledger.outgoing_mouth_downward_rate
                        - max(-bottom_riemann.q_net, 0.0),
                        0.0,
                    )
                ),
                finite_node_circulation_capacity=float(
                    max(
                        node_liquid_volume / dt
                        - max(bottom_riemann.q_net, 0.0),
                        0.0,
                    )
                ),
                riser_circulation_capacity=float(
                    max(
                        downward_donor_volume / dt
                        - max(-bottom_riemann.q_net, 0.0),
                        0.0,
                    )
                ),
                gas_superficial_velocity=float(
                    wallis_gas_superficial_velocity
                ),
                wallis_gas_parameter=float(wallis_gas_parameter),
                upward_channel_area=float(bottom_riemann.upward_area),
                downward_channel_area=float(bottom_riemann.downward_area),
                upward_channel_velocity=accepted_upward_velocity,
                downward_channel_velocity=-accepted_downward_speed,
                resolved_liquid_velocity=float(
                    bottom_riemann.q_net / max(accepted_liquid_area, EPS)
                ),
                resolved_net_flux_mismatch=float(
                    bottom_riemann.q_net
                    - (mouth_upward_discharge + mouth_downward_discharge)
                ),
                gross_convective_momentum_flux=float(
                    gross_convective_momentum
                ),
                bulk_convective_momentum_flux=float(
                    bulk_convective_momentum
                ),
                countercurrent_momentum_excess=float(
                    max(
                        gross_convective_momentum
                        - bulk_convective_momentum,
                        0.0,
                    )
                ),
                gross_kinetic_power=float(gross_kinetic_power),
                signed_kinetic_energy_flux=float(signed_kinetic_flux),
                upward_turn_loss_power=float(upward_loss_power),
                downward_turn_loss_power=float(downward_loss_power),
                countercurrent_mixing_loss_power=float(
                    mixing_loss_power
                ),
                total_dissipation_power=float(
                    upward_loss_power
                    + downward_loss_power
                    + mixing_loss_power
                ),
            )
            twostream_mouth_plan = TwoChannelMouthCouplingPlan(
                exchange=accepted_exchange,
                vertical_boundary=TwoLiquidMomentumBoundaryResidual(
                    upward_volume_rate=accepted_upward_flow,
                    downward_volume_rate=-accepted_downward_flow,
                    upward_convective_momentum_flux=(
                        accepted_upward_flow
                        * accepted_upward_velocity
                    ),
                    downward_convective_momentum_flux=(
                        accepted_downward_flow
                        * accepted_downward_speed
                    ),
                ),
                horizontal_liquid_volume_rate=float(
                    -bottom_riemann.q_net
                ),
                vertical_liquid_volume_rate=float(
                    bottom_riemann.q_net
                ),
                horizontal_axial_kinematic_momentum_rate=float(
                    -accepted_upward_flow * horizontal_axial_velocity
                ),
                horizontal_node_topology=(
                    HorizontalNodeTopology.DISTRIBUTED_FOOTPRINT
                ),
                legacy_paths_to_disable=(
                    "characteristic_bottom_flux_as_update",
                    "taylor_return_as_mass_flux",
                    "post_breakthrough_ccfl_on_q_net",
                    "net_only_horizontal_side_source",
                ),
            )
            bidirectional_tnode_upward_speed_next = float(
                accepted_upward_velocity
            )
            final_mixing_kinematic = (
                0.5
                * twostream_mouth_losses.countercurrent_mixing
                * min(accepted_upward_flow, accepted_downward_flow)
                * (accepted_upward_velocity + accepted_downward_speed)
            )
            final_drive_pressure = float(
                horizontal_liquid_pressure
                - vertical_two_phase_mouth_pressure
            )
            if (
                bottom_riemann.upward_area
                > twostream_parameters.dry_area_tolerance
            ):
                last_tnode_pressure_raw_residual = float(
                    RHO_L
                    * (
                        distributed_tnode_geometry.effective_inertance_length
                        / dt
                        * (
                            accepted_upward_velocity
                            - coupled_mouth_characteristic.old_upward_speed
                        )
                        + 0.5
                        * twostream_mouth_losses.upward_turn
                        * accepted_upward_velocity**2
                        + final_mixing_kinematic
                        / bottom_riemann.upward_area
                    )
                    - final_drive_pressure
                    + last_tnode_capacity_pressure
                )
                last_tnode_pressure_residual = float(
                    last_tnode_pressure_raw_residual
                    + (
                        capacity_projection.upward_upper_bound_multiplier
                        - capacity_projection.upward_lower_bound_multiplier
                    )
                    / dt
                )
            else:
                last_tnode_pressure_raw_residual = 0.0
                last_tnode_pressure_residual = 0.0
            if (
                bottom_riemann.downward_area
                > twostream_parameters.dry_area_tolerance
            ):
                last_tnode_downward_pressure_raw_residual = float(
                    RHO_L
                    * (
                        math.sqrt(G * case.Dr)
                        * (
                            accepted_downward_speed
                            - coupled_mouth_characteristic.raw_downward_speed
                        )
                        + 0.5
                        * twostream_mouth_losses.downward_turn
                        * accepted_downward_speed**2
                        + final_mixing_kinematic
                        / bottom_riemann.downward_area
                    )
                    - last_tnode_capacity_pressure
                )
                last_tnode_downward_pressure_residual = float(
                    last_tnode_downward_pressure_raw_residual
                    + (
                        capacity_projection.downward_upper_bound_multiplier
                        - capacity_projection.downward_lower_bound_multiplier
                    )
                    / dt
                )
            else:
                last_tnode_downward_pressure_raw_residual = 0.0
                last_tnode_downward_pressure_residual = 0.0
            last_tnode_momentum_residual = float(
                bottom_riemann.ledger.momentum_residual
            )
            last_tnode_physical_reaction_pressure = float(
                RHO_L
                * (
                    0.5
                    * twostream_mouth_losses.downward_turn
                    * accepted_downward_speed**2
                    + (
                        final_mixing_kinematic / bottom_riemann.downward_area
                        if bottom_riemann.downward_area
                        > twostream_parameters.dry_area_tolerance
                        else 0.0
                    )
                )
            )
            last_tnode_vertical_mouth_pressure = float(
                vertical_two_phase_mouth_pressure
            )
            last_junction_gross_upward_flow = float(
                twostream_mouth_plan.exchange.upward_flow
            )
            last_junction_gross_downward_flow = float(
                twostream_mouth_plan.exchange.downward_flow
            )
            last_junction_circulation_flow = float(
                twostream_mouth_plan.exchange.circulation_flow
            )
            # The legacy Taylor-return candidate is not a committed boundary
            # condition once the persistent two-stream T node owns the mouth.
            last_junction_taylor_return_flow = 0.0
            # These compatibility arrays are the only liquid fields seen by
            # the gas graph.  Their final post-transport values are committed
            # after the gas stage below.
            Alr_new = np.asarray(
                twostream_work_state.liquid_area,
                dtype=float,
            )
            Qlr_new = np.asarray(
                twostream_work_state.liquid_discharge,
                dtype=float,
            )
            G1[0] = float(bottom_riemann.q_net)
            G2[0] = float(
                accepted_flux.upward_momentum_flux
                + accepted_flux.downward_momentum_flux
            )

        # ---------- conservative finite-volume T exchange ----------
        # Accumulate the vertical exchange here and apply it to the physical
        # horizontal junction control volume below.  The axial west/east faces
        # remain ordinary finite-volume faces, so the horizontal equations—not
        # a prescribed branch split—determine how the returned liquid launches
        # pressure and free-surface waves away from the tee.
        # Net liquid exchange at the side tee.  ``G1[0]`` is already the one
        # selected bottom-face flux: the pressure characteristic outside a
        # fitted sweep, or the film/entrainment closure during the sweep.
        # Descending liquid enters the horizontal pipe normal to its axis, so
        # no axial impulse is manufactured here.
        q_up = float(G1[0])  # [m^3/s], upward positive
        junction_vertical_node_flow = q_up
        last_junction_vertical_flow = q_up
        old_riser_gas_front = float(riser_gas_front)
        riser_gas_front_velocity = 0.0

        # The measured side-T has finite intersection volume.  Once the local
        # finite-volume horizontal solver owns the field, apply the vertical
        # exchange to that junction control volume: upward liquid removes the
        # local parcel and its axial momentum.  Vertically returning liquid is
        # deposited over the measured mouth and its computed normal momentum is
        # redirected symmetrically by the finite-width impingement closure below;
        # the net axial impulse remains exactly zero.  Subsequent propagation is
        # determined by the horizontal pressure and momentum equations.  The one
        # hand-off step remains on the local shock-fitting state and uses its
        # conservative two-face adapter below.
        horizontal_west_node_flow = float(F1[junction_face])
        horizontal_east_node_flow = float(F1[junction_face])
        last_q_up = float(q_up)
        last_junction_vertical_flow = float(q_up)
        last_junction_west_flow = float(horizontal_west_node_flow)
        last_junction_east_flow = float(horizontal_east_node_flow)
        junction_volume_change = -q_up * dt
        if confined_gross_exchange_active:
            returned_volume_step = confined_gross_downward_flow * dt
            junction_return_requested_volume += returned_volume_step
            junction_return_deposited_volume += returned_volume_step
        elif junction_volume_change > 0.0:
            junction_return_requested_volume += junction_volume_change
            junction_return_deposited_volume += junction_volume_change
        if twostream_active_step and external_horizontal_next is not None:
            raise FloatingPointError(
                "two-stream T-node activated before the external horizontal handoff completed"
            )
        if junction_wave_active and external_horizontal_next is None:
            # The measured tower mouth is a finite side opening, not a
            # zero-volume node at one horizontal face.  Apply the shared riser
            # flux over the exact geometric footprint while the ordinary
            # horizontal west/east fluxes keep evolving without interruption.
            if twostream_active_step:
                if twostream_mouth_plan is None:
                    raise FloatingPointError(
                        "active two-stream stage has no gross mouth plan"
                    )
                horizontal_exchange = apply_twochannel_horizontal_footprint(
                    Alt_new,
                    Qlt_new,
                    side_t_weights,
                    cell_width=dx,
                    opening_length=case.Dr,
                    time_step=dt,
                    plan=twostream_mouth_plan,
                )
                Alt_new = horizontal_exchange.liquid_area
                Qlt_new = horizontal_exchange.liquid_discharge
            elif confined_gross_exchange_active:
                if confined_gross_downward_weights is None:
                    raise FloatingPointError(
                        "confined Taylor return has no horizontal phase path"
                    )
                Alt_new, Qlt_new = _apply_finite_width_side_t_gross_exchange(
                    Alt_new,
                    Qlt_new,
                    upward_flow=confined_gross_upward_flow,
                    downward_flow=confined_gross_downward_flow,
                    opening_weights=side_t_weights,
                    downward_weights=confined_gross_downward_weights,
                    dt=dt,
                    cell_width=dx,
                    full_area=A,
                )
            else:
                Alt_new, Qlt_new = _apply_finite_width_side_t_exchange(
                    Alt_new,
                    Qlt_new,
                    upward_flow=q_up,
                    opening_weights=side_t_weights,
                    dt=dt,
                    cell_width=dx,
                    full_area=A,
                )
            junction_wave_max_source_cells = max(
                junction_wave_max_source_cells,
                int(np.count_nonzero(side_t_weights)),
            )

        if external_horizontal_next is not None:
            # Apply the same three-branch liquid fluxes directly to the two
            # shock-fit cells adjacent to the physical T face.  The horizontal
            # solver stays active after gas arrival; no distributed gas-cell
            # hand-off or frozen interface is introduced.
            if not confined_gross_exchange_active:
                horizontal_mean_flow = 0.5 * (
                    horizontal_west_node_flow
                    + horizontal_east_node_flow
                )
                horizontal_west_node_flow = (
                    horizontal_mean_flow + 0.5 * q_up
                )
                horizontal_east_node_flow = (
                    horizontal_mean_flow - 0.5 * q_up
                )
                external_horizontal_next = (
                    external_horizontal_solver.apply_junction_liquid_fluxes(
                        external_horizontal_next,
                        west_flow=horizontal_west_node_flow,
                        east_flow=horizontal_east_node_flow,
                        dt=dt,
                    )
                )

            if (
                bool(external_horizontal_next.vented)
                and not junction_topology_opened
            ):
                # This is a change of graph ownership, not a remap to a
                # lumped reservoir.  The same-grid mapped liquid area,
                # discharge, gas mass and reconstructed gas momentum below
                # become the initial state of the distributed two-fluid FV
                # graph on the next step.  No second gas owner remains.
                junction_topology_opened = True
                resolved_horizontal_handoff_this_step = True
                external_horizontal_active = False
                external_horizontal_handoff_time = float(t + dt)

            Alt_new, Qlt_new, Mgt_new, Jgt_new = (
                _map_external_horizontal_state(
                    external_horizontal_solver,
                    external_horizontal_next,
                    x_target=xt,
                    full_area=A,
                    dx=dx,
                )
            )
            if not confined_gross_exchange_active:
                # The coupled vertical/gas stages below temporarily consume
                # this mapped owner state, then the shock-fit owner restores
                # it at commit.  Cache the pure mapping instead of evaluating
                # the same interpolation a second time later in this step.
                external_horizontal_commit_state = tuple(
                    np.asarray(component, dtype=float).copy()
                    for component in (Alt_new, Qlt_new, Mgt_new, Jgt_new)
                )
            if confined_gross_exchange_active:
                if not bool(external_horizontal_next.vented):
                    raise FloatingPointError(
                        "confined Taylor exchange preceded horizontal handoff"
                    )
                Alt_new, Qlt_new = _apply_finite_width_side_t_gross_exchange(
                    Alt_new,
                    Qlt_new,
                    upward_flow=confined_gross_upward_flow,
                    downward_flow=confined_gross_downward_flow,
                    opening_weights=side_t_weights,
                    downward_weights=confined_gross_downward_weights,
                    dt=dt,
                    cell_width=dx,
                    full_area=A,
                )
                external_gross_exchange_applied = True

        # Do not pre-open an artificial gas cut-cell in the liquid-full east
        # branch.  That former seed was tiny in volume but made the complete
        # downstream elastic rarefaction topologically gas-connected, so the
        # acoustic gas solve filled the dead leg long before the material
        # interface did.  The east branch now becomes a gas receiver only when
        # the conservative liquid solve itself creates a capillary-open void
        # adjacent to mass-supported gas.  Until then, the buoyant T-mouth path
        # into the riser is the only connected gas branch, as in the 2-D case.
        gas_mass_input = (
            Mgt_new if external_horizontal_next is not None else Mgt
        )
        gas_momentum_input = (
            Jgt_new if external_horizontal_next is not None else Jgt
        )
        vertical_gas_momentum_input = np.asarray(Jgrs, dtype=float).copy()
        if twostream_active_step:
            # The bottom owner is now gross-first and contains no independent
            # half-cell q_c gas/shear solve.  The complete first-cell gas/liquid
            # action--reaction is applied once by the distributed three-body
            # operator below.
            last_tnode_gas_reaction_requested = 0.0
            last_tnode_gas_reaction_applied = 0.0
            last_tnode_gas_reaction_application_residual = 0.0
            last_tnode_liquid_gas_action_residual = 0.0

        # ---------- one conservative horizontal--T--vertical gas network ----------
        # The complete horizontal branch, the 90-degree side opening, and the
        # complete riser are advanced in the same gas-acoustic SSP-RK substeps.
        # The T flux is subtracted from the horizontal junction cell and added
        # to the first vertical cell in the same residual.  There is no step-end
        # gas deletion, prescribed drift speed, transfer fraction, or surface
        # burst rule.  The vertical top is an atmospheric Riemann boundary.
        Mgt_new = np.maximum(gas_mass_input.copy(), 0.0)
        Jgt_new = gas_momentum_input.copy()
        Mgr_new = np.maximum(Mgr.copy(), 0.0)
        Mgrs_new = np.maximum(Mgrs.copy(), 0.0)
        Jgrs_new = vertical_gas_momentum_input.copy()
        if junction_wave_active:
            try:
                gas_advance = advance_coupled_gas_network(
                    gas_mass_input,
                    gas_momentum_input,
                    Mgr,
                    vertical_gas_momentum_input,
                    Mgrs,
                    Alt_new,
                    Qlt_new,
                    Alr_new,
                    Qlr_new,
                    dx=dx,
                    dz=dz,
                    dt=dt,
                    junction_index=junction_west_cell,
                    params=(
                        coupled_gas_parameters
                        if twostream_active_step
                        else coupled_gas_parameters_one_stream
                    ),
                    vertical_pocket_front_height=(
                        riser_material_front
                        if material_front_tracked and not riser_breakthrough
                        else None
                    ),
                    vertical_liquid_surface_height=float(wtop),
                    # The material Taylor core continues to make the riser
                    # the gas-receiving branch after pneumatic breakthrough.
                    # ``vertical_branch_confined`` is used by the gas graph
                    # only for side-T phase separation; it does not close the
                    # atmospheric top face.  Dropping this flag exactly when
                    # the nose met the free surface switched the tee to the
                    # east dead leg in one time step and launched a nonphysical
                    # rarefaction through the horizontal liquid film.
                    vertical_branch_confined=(
                        material_front_tracked and not riser_breakthrough
                    ),
                    vertical_branch_receiving_hint=(
                        vertical_gas_receiving
                    ),
                    horizontal_downstream_front_position=(
                        side_t_east_material_front
                    ),
                    horizontal_downstream_topology_front_position=(
                        side_t_east_topology_front
                    ),
                    # The upward branch is exclusive only while the Taylor
                    # core is confined.  After breakthrough the finite T can
                    # simultaneously vent upward and admit a conservative
                    # crown-gas flux into the right dead leg, as in the 2-D
                    # solution.  No gas mass or split fraction is prescribed.
                    prefer_vertical_branch=not riser_breakthrough,
                )
            except FloatingPointError as exc:
                horizontal_velocity = np.divide(
                    Jgt,
                    Mgt,
                    out=np.zeros_like(Jgt),
                    where=Mgt > 1.0e-14,
                )
                vertical_velocity = np.divide(
                    vertical_gas_momentum_input,
                    Mgr,
                    out=np.zeros_like(vertical_gas_momentum_input),
                    where=Mgr > 1.0e-14,
                )
                raise FloatingPointError(
                    "coupled gas failure "
                    f"at t={t:.12g}, dt={dt:.12g}, "
                    f"alpha_g_j={1.0-Alt_new[junction_west_cell]/A:.8g}, "
                    f"Mgt_j={Mgt[junction_west_cell]:.8g}, Mgr_0={Mgr[0]:.8g}, "
                    f"Mgrs_0={Mgrs[0]:.8g}, "
                    f"max_uh={np.max(np.abs(horizontal_velocity)):.8g}, "
                    f"max_uv={np.max(np.abs(vertical_velocity)):.8g}, "
                    f"Alr_0/Ar={Alr_new[0]/Ar:.8g}"
                ) from exc
            Mgt_new = gas_advance.horizontal_mass
            Jgt_new = gas_advance.horizontal_momentum
            Mgr_new = gas_advance.vertical_total_mass
            Jgrs_new = gas_advance.vertical_momentum
            Mgrs_new = gas_advance.vertical_tracer_mass
            vertical_void_new = np.maximum(
                Ar - np.clip(Alr_new, 0.0, Ar), 0.0
            )
            tracer_supported_new = _mass_backed_gas_topology(
                vertical_void_new,
                Mgrs_new,
                full_area=Ar,
                cell_width=dz,
                rho_reference=rho_atm,
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                resolved_density_fraction=(
                    coupled_gas_parameters.resolved_density_fraction
                ),
            )
            tracer_indices = np.flatnonzero(tracer_supported_new)
            riser_gas_front = (
                min((float(tracer_indices[-1]) + 1.0) * dz, case.riser_height)
                if tracer_indices.size
                else 0.0
            )
            if material_front_confined:
                riser_gas_front = min(
                    riser_gas_front,
                    float(riser_material_front),
                )
            riser_gas_front_velocity = (
                riser_gas_front - old_riser_gas_front
            ) / max(dt, EPS)
            if (
                gas_advance.downstream_front_position is not None
            ):
                side_t_east_material_front = float(
                    gas_advance.downstream_front_position
                )
            if gas_advance.downstream_topology_front_position is not None:
                side_t_east_topology_front = float(
                    gas_advance.downstream_topology_front_position
                )
            side_t_east_material_front_velocity = float(
                gas_advance.downstream_front_velocity
            )
            side_t_east_retired_cell_count += int(
                gas_advance.downstream_retired_cell_count
            )
            Qlt_new += gas_advance.horizontal_liquid_momentum_increment
            if twostream_active_step:
                if np.max(
                    np.abs(gas_advance.vertical_liquid_momentum_increment)
                ) > 1.0e-14:
                    raise FloatingPointError(
                        "legacy vertical gas drag remained active with two-stream coupling"
                    )
            else:
                Qlr_new += gas_advance.vertical_liquid_momentum_increment
            gas_escaped_mass += max(gas_advance.escaped_tracer_mass, 0.0)
            gas_atmospheric_exchange += gas_advance.atmospheric_mass_exchange
            horizontal_gas_substeps += gas_advance.substeps
            horizontal_gas_active_cells = int(np.count_nonzero(Mgt_new > 1.0e-14))
            horizontal_gas_mass_error += gas_advance.total_mass_error
            gas_velocity_h = np.divide(
                Jgt_new,
                Mgt_new,
                out=np.zeros_like(Jgt_new),
                where=Mgt_new > 1.0e-14,
            )
            horizontal_gas_kinetic_energy = float(
                np.sum(0.5 * Mgt_new * gas_velocity_h * gas_velocity_h)
            )
            gas_mass_h = float(np.sum(Mgt_new))
            horizontal_gas_center_of_mass = (
                float(np.sum(Mgt_new * xt) / gas_mass_h)
                if gas_mass_h > 1.0e-14
                else float(xt[0])
            )
            horizontal_void_h = np.maximum(
                A - np.clip(Alt_new, 0.0, A),
                coupled_gas_parameters.void_floor_fraction * A,
            )
            horizontal_density_h = Mgt_new / np.maximum(
                horizontal_void_h * dx, EPS
            )
            resolved_horizontal_h = (
                horizontal_density_h
                > coupled_gas_parameters.resolved_density_fraction * rho_atm
            ) & (
                horizontal_density_h
                < coupled_gas_parameters.resolved_density_ceiling * rho_atm
            )
            horizontal_gas_maximum_velocity = max(
                horizontal_gas_maximum_velocity,
                float(np.max(np.abs(gas_velocity_h[resolved_horizontal_h])))
                if np.any(resolved_horizontal_h)
                else 0.0,
            )
            coupled_gas_maximum_velocity = max(
                coupled_gas_maximum_velocity,
                gas_advance.maximum_velocity,
            )

        if twostream_active_step:
            if (
                twostream_work_state is None
                or twostream_mouth_plan is None
            ):
                raise FloatingPointError(
                    "incomplete persistent two-stream stage after gas advance"
                )
            mouth_exchange = twostream_mouth_plan.exchange
            bottom_boundary = DirectionalBoundaryFlux(
                upward_rate=float(mouth_exchange.upward_flow),
                upward_speed=float(mouth_exchange.upward_channel_velocity),
                downward_rate=float(mouth_exchange.downward_flow),
                downward_speed=abs(
                    float(mouth_exchange.downward_channel_velocity)
                ),
            )
            if twostream_top_boundary is None:
                raise FloatingPointError(
                    "active two-stream stage has no predicted top boundary"
                )
            pressure_geometry = coaxial_core_film_geometry(
                twostream_work_state,
                twostream_parameters,
            )
            pressure_gas_area = np.asarray(
                pressure_geometry.gas_area,
                dtype=float,
            )
            common_pressure_cells = np.asarray(Pr, dtype=float).copy()
            pressure_gas_active = _riser_material_gas_mask(
                pressure_gas_area,
                Mgrs_new,
                full_area=Ar,
                cell_length=dz,
                reference_density=rho_atm,
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                resolved_density_fraction=(
                    coupled_gas_parameters.resolved_density_fraction
                ),
            ) & (Mgr_new > 0.0)
            common_pressure_cells[pressure_gas_active] = (
                Mgr_new[pressure_gas_active]
                * R_GAS
                * T_GAS
                / np.maximum(
                    pressure_gas_area[pressure_gas_active] * dz,
                    EPS,
                )
            )
            open_ambient_pressure_mask = _top_connected_atmospheric_gas_mask(
                pressure_gas_area,
                Mgrs_new,
                cell_length=dz,
                reference_density=rho_atm,
                dry_area_tolerance=twostream_parameters.dry_area_tolerance,
            )
            common_pressure_cells[open_ambient_pressure_mask] = P_ATM
            common_pressure_faces = np.empty(Nr + 1, dtype=float)
            # This is the same z=0 normal-stress trace used as the upper end of
            # the finite T-node pressure-work segment.  The FV segment begins
            # here and extends to face 1, so no pressure drop is duplicated.
            common_pressure_faces[0] = float(
                vertical_two_phase_mouth_pressure
            )
            last_twostream_bottom_pressure = float(common_pressure_faces[0])
            last_tnode_fv_mouth_pressure_residual = float(
                last_twostream_bottom_pressure
                - last_tnode_vertical_mouth_pressure
            )
            common_pressure_faces[-1] = P_ATM
            if Nr > 1:
                common_pressure_faces[1:-1] = 0.5 * (
                    common_pressure_cells[:-1]
                    + common_pressure_cells[1:]
                )
            # A Taylor-gas core separates the rising liquid tongue from the
            # wall-connected falling film throughout the swept part of the
            # riser, not only in cell 0.  Keep a stopped directional corridor
            # there so an arbitrarily small sign crossing cannot relabel the
            # complete film inventory in one step.  The topology operator may
            # still transfer the minimum entropy-admissible area when a branch
            # genuinely reverses.  Unswept single-column cells retain the
            # ordinary one-stream merge rule.
            preserve_separated_partition = np.asarray(
                taylor_swept_fraction_next > mouth_swept_tolerance,
                dtype=bool,
            )
            preserve_separated_partition[0] = True
            twostream_transport = advance_vertical_two_stream_fv(
                twostream_work_state,
                twostream_parameters,
                dt=dt,
                pressure_faces=common_pressure_faces,
                boundaries=VerticalTwoStreamBoundaries(
                    bottom=bottom_boundary,
                    top=twostream_top_boundary,
                ),
                preserve_stopped_partition=preserve_separated_partition,
                liquid_capacity_area=twostream_liquid_capacity_area,
                bottom_downward_reaction_flux=(
                    bottom_riemann.ledger.downward_physical_reaction_flux
                ),
            )
            accepted_bottom_upward = float(
                twostream_transport.upward_area_flux[0]
            )
            accepted_bottom_downward = float(
                -twostream_transport.downward_area_flux[0]
            )
            transaction_tolerance = max(
                1.0e-14,
                2048.0
                * np.finfo(float).eps
                * max(
                    abs(float(mouth_exchange.upward_flow)),
                    abs(float(mouth_exchange.downward_flow)),
                    1.0e-12,
                ),
            )
            if (
                abs(
                    accepted_bottom_upward
                    - float(mouth_exchange.upward_flow)
                )
                > transaction_tolerance
                or abs(
                    accepted_bottom_downward
                    - float(mouth_exchange.downward_flow)
                )
                > transaction_tolerance
            ):
                raise FloatingPointError(
                    "horizontal and vertical T branches did not commit the "
                    "same capacity-limited gross liquid transaction: "
                    f"requested_up={mouth_exchange.upward_flow:.12e}, "
                    f"accepted_up={accepted_bottom_upward:.12e}, "
                    f"requested_down={mouth_exchange.downward_flow:.12e}, "
                    f"accepted_down={accepted_bottom_downward:.12e}"
                )
            if twostream_provenance_work_state is None:
                raise FloatingPointError(
                    "two-stream transport has no liquid-provenance owner"
                )
            try:
                provenance_transport = (
                    advance_vertical_two_stream_liquid_provenance(
                        twostream_provenance_work_state,
                        twostream_work_state,
                        twostream_transport,
                        twostream_parameters,
                        dt=dt,
                    )
                )
            except Exception as exc:
                raise type(exc)(
                    f"liquid provenance failed at t={t:.12g}, dt={dt:.12g}: {exc}"
                ) from exc
            transported = twostream_transport.state

            # Apply the existing Reynolds-aware wall law independently to the
            # two prognostic liquid momenta.  A descending channel is treated
            # as an annular film only where an actual gas void separates it
            # from the upward core; an unswept full-column downflow remains a
            # bulk pipe flow.
            transported_geometry = coaxial_core_film_geometry(
                transported,
                twostream_parameters,
            )
            transported_gas_area = np.asarray(
                transported_geometry.gas_area,
                dtype=float,
            )
            (
                Mgr_new,
                Jgrs_new,
                open_headspace_exchange,
                _,
            ) = _equilibrate_open_riser_headspace(
                Mgr_new,
                Jgrs_new,
                Mgrs_new,
                transported_gas_area,
                cell_length=dz,
                reference_density=rho_atm,
                dry_area_tolerance=twostream_parameters.dry_area_tolerance,
            )
            gas_atmospheric_exchange += open_headspace_exchange
            vertical_open_headspace_mass_exchange += open_headspace_exchange
            upward_area = np.asarray(transported.upward_area, dtype=float)
            upward_discharge = np.asarray(
                transported.upward_discharge,
                dtype=float,
            )
            downward_area = np.asarray(
                transported.downward_area,
                dtype=float,
            )
            downward_discharge = np.asarray(
                transported.downward_discharge,
                dtype=float,
            )
            upward_friction = _riser_liquid_friction_rate(
                upward_area,
                upward_discharge,
                np.zeros(Nr, dtype=bool),
                full_area=Ar,
                diameter=case.Dr,
                film_thickness=riser_film_thickness,
            )
            downward_friction = _riser_liquid_friction_rate(
                downward_area,
                downward_discharge,
                (
                    downward_area
                    > twostream_parameters.dry_area_tolerance
                )
                & (
                    transported_gas_area
                    > twostream_parameters.dry_area_tolerance
                ),
                full_area=Ar,
                diameter=case.Dr,
                film_thickness=np.where(
                    transported_gas_area
                    > twostream_parameters.dry_area_tolerance,
                    np.maximum(
                        0.5 * case.Dr
                        - np.sqrt(
                            np.maximum(
                                (0.5 * case.Dr) ** 2
                                - downward_area / math.pi,
                                0.0,
                            )
                        ),
                        1.0e-6 * case.Dr,
                    ),
                    riser_film_thickness,
                ),
            )
            friction_state = VerticalTwoStreamState.from_iterables(
                upward_area=upward_area,
                upward_discharge=(
                    upward_discharge / (1.0 + dt * upward_friction)
                ),
                downward_area=downward_area,
                downward_discharge=(
                    downward_discharge / (1.0 + dt * downward_friction)
                ),
            )

            # The gas transport stage owns gas mass and pressure.  This source
            # step only exchanges momentum with both liquid streams and writes
            # the equal-and-opposite gas reaction back once.
            drag_geometry = coaxial_core_film_geometry(
                friction_state,
                twostream_parameters,
            )
            drag_gas_area = np.asarray(drag_geometry.gas_area, dtype=float)
            drag_active = _riser_material_gas_mask(
                drag_gas_area,
                Mgrs_new,
                full_area=Ar,
                cell_length=dz,
                reference_density=rho_atm,
                void_floor_fraction=(
                    coupled_gas_parameters.void_floor_fraction
                ),
                active_void_fraction=(
                    coupled_gas_parameters.active_void_fraction
                ),
                topology_density_fraction=(
                    coupled_gas_parameters.topology_density_fraction
                ),
                resolved_density_fraction=(
                    coupled_gas_parameters.resolved_density_fraction
                ),
            ) & (Mgr_new > 0.0)
            drag_gas_mass = np.where(drag_active, Mgr_new, 0.0)
            drag_gas_momentum = np.where(drag_active, Jgrs_new, 0.0)
            # No independent half-cell T-node gas/shear solve remains.  The
            # distributed three-body operator therefore owns the complete
            # first cell, exactly as it owns every higher cell.
            distributed_drag_length_fraction = np.ones(Nr, dtype=float)
            last_tnode_cell0_drag_length_fraction = 1.0
            physical_drag_state = PhysicalGasInterphaseState.from_iterables(
                gas_mass=drag_gas_mass,
                gas_momentum=drag_gas_momentum,
                gas_area=np.where(drag_active, drag_gas_area, 0.0),
                upward_interface_perimeter=(
                    np.where(
                        drag_active,
                        drag_geometry.upward_gas_interface_perimeter
                        * distributed_drag_length_fraction,
                        0.0,
                    )
                ),
                downward_interface_perimeter=(
                    np.where(
                        drag_active,
                        drag_geometry.downward_gas_interface_perimeter
                        * distributed_drag_length_fraction,
                        0.0,
                    )
                ),
                upward_hydraulic_diameter=(
                    np.where(
                        drag_active,
                        drag_geometry.gas_hydraulic_diameter,
                        case.Dr,
                    )
                ),
                downward_hydraulic_diameter=(
                    np.where(
                        drag_active,
                        drag_geometry.gas_hydraulic_diameter,
                        case.Dr,
                    )
                ),
                gas_viscosity=coupled_gas_parameters.gas_viscosity,
            )
            drag_result = implicit_physical_three_body_drag_exchange(
                friction_state,
                twostream_parameters,
                physical_drag_state,
                dt=dt,
                preserve_stopped_partition=preserve_separated_partition,
            )
            provenance_after_drag = (
                conservative_liquid_provenance_topology_transfer(
                    provenance_transport.state,
                    drag_result.topology_transfer,
                    area_tolerance=max(
                        twostream_parameters.dry_area_tolerance,
                        twostream_parameters.packing_tolerance,
                    ),
                )
            )
            riser_liquid_provenance_next = provenance_after_drag.state
            riser_twostream_next = drag_result.state
            Jgrs_new = np.where(
                drag_active,
                np.asarray(drag_result.gas_momentum, dtype=float),
                Jgrs_new,
            )
            Alr_new = np.asarray(
                riser_twostream_next.liquid_area,
                dtype=float,
            )
            Qlr_new = np.asarray(
                riser_twostream_next.liquid_discharge,
                dtype=float,
            )
            last_twostream_upward_volume_residual = float(
                twostream_transport.ledger.upward_volume_residual
            )
            last_twostream_downward_volume_residual = float(
                twostream_transport.ledger.downward_volume_residual
            )
            last_twostream_provenance_volume_residual = float(
                provenance_transport.ledger.source1_volume_residual
                + dz * provenance_after_drag.source1_area_residual
            )
            last_twostream_horizontal_source_volume = float(
                dz
                * sum(
                    riser_liquid_provenance_next.source1_area
                )
            )
            last_twostream_initial_source_volume = float(
                dz * sum(riser_twostream_next.liquid_area)
                - last_twostream_horizontal_source_volume
            )
            last_twostream_drag_momentum_residual = float(
                drag_result.total_momentum_residual
            )
            last_combined_interphase_momentum_residual = float(
                drag_result.total_momentum_residual
            )
            bottom_remaining = 0.10
            bottom_inventory = 0.0
            for cell_area in riser_twostream_next.liquid_area:
                if bottom_remaining <= 0.0:
                    break
                segment = min(dz, bottom_remaining)
                bottom_inventory += float(cell_area) * segment
                bottom_remaining -= segment
            last_twostream_bottom_inventory = bottom_inventory
            resolved_top_rate = max(
                float(twostream_transport.upward_area_flux[-1])
                + float(twostream_transport.downward_area_flux[-1]),
                0.0,
            )
            G1[-1] = resolved_top_rate + twostream_sweep_overflow_rate
            G2[-1] = max(
                float(twostream_transport.upward_momentum_flux[-1])
                + float(twostream_transport.downward_momentum_flux[-1]),
                0.0,
            )

        # Complete the vertical momentum split after gas--liquid drag.  Bulk
        # liquid uses ordinary pipe friction; only a resolved thin annular film
        # receives the laminar--turbulent film stress.
        if not twostream_active_step:
            cell_top_r = (np.arange(Nr, dtype=float) + 1.0) * dz
            annular_film_cell = (
                (riser_gas_front > 0.0)
                & (cell_top_r <= riser_gas_front + 1.0e-12)
                & (
                    Alr_new
                    <= 1.25
                    * (
                        1.0
                        - coupled_gas_parameters.vertical_gas_core_area_fraction
                    )
                    * Ar
                )
            )
            fric_r = _riser_liquid_friction_rate(
                Alr_new,
                Qlr_new,
                annular_film_cell,
                full_area=Ar,
                diameter=case.Dr,
                film_thickness=riser_film_thickness,
            )
            Qlr_new /= 1.0 + dt * fric_r

        # Gas leaves the apparatus only through the resolved top Riemann flux.
        _bt_dbg["vented"] = 1.0 if riser_breakthrough else 0.0

        # Unresolved T-junction/film shear stress.  This local closure replaces
        # the former global Laplacian on Q (whose coefficient scaled with the
        # artificial water-hammer speed and therefore had no physical units).
        # It damps only resolved velocity gradients, is conservative at the
        # closed horizontal ends, and contains no prescribed wave footprint.
        if external_horizontal_next is None or external_gross_exchange_applied:
            Qlt_new = _implicit_smagorinsky_momentum_diffusion(
                Alt_new,
                Qlt_new,
                full_area=A,
                diameter=case.D,
                spacing=dx,
                dt=dt,
                coefficient=case.nu,
            )
        if not twostream_active_step:
            Qlr_new = _implicit_smagorinsky_momentum_diffusion(
                Alr_new,
                Qlr_new,
                full_area=Ar,
                diameter=case.Dr,
                spacing=dz,
                dt=dt,
                coefficient=case.nu_riser,
            )

        vertical_tracer_present_new = bool(
            float(np.sum(Mgrs_new)) > vertical_tracer_presence_mass
        )
        # Breakthrough changes the gas topology but does not teleport the
        # remaining upper liquid into bottom cells.  The former one-step
        # ``_collapse_upper_slug_at_taylor_breakthrough`` projection moved
        # liquid nonlocally, discarded its axial momentum, and filled the T
        # mouth just before the persistent two-stream handoff.  That artificial
        # repacking closed the resolved gas aperture and made the falling water
        # impossible to discharge.  Retain the cell inventories here: the
        # conservative FV faces, wall stress, gravity, and the open boundaries
        # now determine their subsequent fall or exit locally.

        if material_front_reached_surface:
            riser_breakthrough = True
            twostream_handoff_requested = bool(
                case.enable_vertical_twostream
                and riser_twostream_state is None
                and riser_material_front > 0.0
                and vertical_tracer_present_new
            )

        # A detached upper liquid island remains a resolved liquid body.  It is
        # allowed to fall through local faces after the directional handoff;
        # do not relocate it into the T mixing cells in one nonlocal operation.

        # Repack only a genuinely gas-free riser.  The former condition used
        # the instantaneous horizontal T-mouth state: as soon as a detached
        # Taylor bubble left the mouth, it collapsed the still-two-phase riser
        # into a bottom-packed liquid column and froze all later dynamics.
        if (
            not twostream_active_step
            and not material_front_tracked
            and not vertical_tracer_present_new
        ):
            Alr_new, Qlr_new = _project_single_liquid_column(
                Alr_new, Qlr_new, Ar, dz
            )
            riser_gas_front = 0.0
            riser_gas_front_velocity = 0.0
            riser_breakthrough = False
            riser_material_front = 0.0

        # Open cells are selected only by the positivity-scale liquid cutoff.
        if not twostream_active_step:
            Alr_new[Alr_new < 0] = 0.0
            Qlr_new = _regularize_near_dry_momentum(
                Alr_new,
                Qlr_new,
                full_area=Ar,
                dry_fraction=1.0e-2,
            )
            wetr_new = Alr_new / Ar > 1.0e-6
            open_top = np.zeros(Nr, dtype=bool)
            ii = Nr - 1
            while ii >= 0 and not wetr_new[ii]:
                open_top[ii] = True; ii -= 1
            Qlr_new = np.where(open_top, 0.0, Qlr_new)
        # Dry riser cells are part of the gas domain, not an instantaneous sink.
        # Their total gas mass, pocket tracer, and momentum have already been
        # advanced to the atmospheric top face by the coupled gas solver.
        Jgrs_new = np.where(Mgr_new > 1.0e-14, Jgrs_new, 0.0)
        # Positivity floor only.  The old 1.05*A CEILING silently DESTROYED the
        # elastic overfill volume of every hammer spike (~0.1 L over the release
        # transient): that missing water had to come from somewhere, so the tower
        # over-drained and the pocket over-compressed far beyond the Ha0->Yfs0
        # equilibrium.  Overfill is a legitimate elastic state (alpha=1.05 is a
        # 180 kPa water-hammer peak) and the stiff EOS pulls it back by itself;
        # the donor-availability flux limits already prevent runaway.
        dbg_created["t_floor"] += -float(np.sum(np.minimum(Alt_new, 0.0)) * dx)
        Alt_new = np.maximum(Alt_new, 0.0)
        if twostream_active_step:
            if (
                np.min(Alr_new) < -twostream_parameters.packing_tolerance
                or np.max(Alr_new)
                > Ar + twostream_parameters.packing_tolerance
            ):
                raise FloatingPointError(
                    "two-stream liquid area failed the riser packing audit"
                )
        else:
            Alr_new = np.maximum(Alr_new, 0.0)

        if twostream_handoff_requested:
            # Complete the crossing step with the old owner, then map its final
            # state once.  The former mid-step handoff mapped ``Alr/Qlr`` from
            # the beginning of the interval and immediately advanced another
            # full dt with the new owner, losing the accepted crossing flux and
            # creating a post-breakthrough refill gap.
            Alr_new, Qlr_new, _ = (
                _relax_elastic_riser_storage_for_twostream_handoff(
                    Alr_new,
                    Qlr_new,
                    full_area=Ar,
                    dz=dz,
                    area_tolerance=twostream_parameters.packing_tolerance,
                )
            )
            handoff_swept_fraction = np.clip(
                (
                    float(riser_material_front)
                    - np.arange(Nr, dtype=float) * dz
                )
                / dz,
                0.0,
                1.0,
            )
            # The first control volume contains the finite T intersection.  Its
            # Taylor corridor is complete when the material nose reaches the
            # atmospheric headspace, even if the equivalent bulk level has
            # already fallen into that same cell.
            handoff_swept_fraction[0] = 1.0
            breakthrough_mapping = map_taylor_breakthrough_to_twostream(
                Alr_new,
                Qlr_new,
                twostream_parameters,
                taylor_core_area_fraction=riser_gas_core_fraction,
                taylor_rise_velocity=0.345 * math.sqrt(G * case.Dr),
                swept_fraction=handoff_swept_fraction,
            )
            riser_twostream_state = breakthrough_mapping.state
            riser_liquid_provenance_state = (
                VerticalTwoStreamLiquidProvenanceState.initial_riser_water(
                    riser_twostream_state
                )
            )
            riser_liquid_provenance_next = riser_liquid_provenance_state
            taylor_swept_fraction = handoff_swept_fraction.copy()

            if confined_gross_exchange_active:
                handoff_upward_flow = float(confined_gross_upward_flow)
                handoff_downward_flow = float(confined_gross_downward_flow)
            else:
                handoff_upward_flow = max(float(G1[0]), 0.0)
                handoff_downward_flow = max(-float(G1[0]), 0.0)
            bidirectional_tnode_upward_speed = float(
                handoff_upward_flow
                / max(
                    riser_twostream_state.upward_area[0],
                    twostream_parameters.dry_area_tolerance,
                )
            )
            last_junction_gross_upward_flow = handoff_upward_flow
            last_junction_gross_downward_flow = handoff_downward_flow
            last_junction_circulation_flow = min(
                handoff_upward_flow,
                handoff_downward_flow,
            )

            activation_gas_area = np.asarray(
                coaxial_core_film_geometry(
                    riser_twostream_state,
                    twostream_parameters,
                ).gas_area,
                dtype=float,
            )
            (
                Mgr_new,
                Jgrs_new,
                activation_headspace_exchange,
                _,
            ) = _equilibrate_open_riser_headspace(
                Mgr_new,
                Jgrs_new,
                Mgrs_new,
                activation_gas_area,
                cell_length=dz,
                reference_density=rho_atm,
                dry_area_tolerance=twostream_parameters.dry_area_tolerance,
            )
            gas_atmospheric_exchange += activation_headspace_exchange
            vertical_open_headspace_mass_exchange += (
                activation_headspace_exchange
            )
            Alr_new = np.asarray(
                riser_twostream_state.liquid_area,
                dtype=float,
            )
            Qlr_new = np.asarray(
                riser_twostream_state.liquid_discharge,
                dtype=float,
            )
            twostream_activated_time = float(t + dt)

        if external_horizontal_next is not None:
            # The conservative T-face correction and open-gas update were
            # applied before the gas solve.  Remap only for the network record;
            # do not apply the branch flux a second time and do not hand the
            # horizontal state to the distributed gas-cell solver.
            if not external_gross_exchange_applied:
                if external_horizontal_commit_state is None:
                    raise FloatingPointError(
                        "external horizontal owner has no cached commit state"
                    )
                Alt_new, Qlt_new, Mgt_new, Jgt_new = (
                    external_horizontal_commit_state
                )
            external_horizontal_state = external_horizontal_next
            external_horizontal_active = bool(
                not resolved_horizontal_handoff_this_step
            )
        # The horizontal gas core can leave an almost dry liquid film under
        # the side-T.  As in the riser, finite-volume round-off momentum must
        # tend to zero faster than area in that wet/dry limit; otherwise a
        # finite Q divided by O(1e-6 A) creates a fictitious kilometre-per-
        # second liquid velocity and collapses the global CFL step.  This
        # standard desingularisation is smooth.  Its 1% transition scale is
        # well below the V&W annular film (about 7% of the Case-A tower area),
        # while suppressing momentum carried by sub-percent residual puddles.
        Qlt_new = _regularize_near_dry_momentum(
            Alt_new,
            Qlt_new,
            full_area=A,
        )
        if junction_wave_active:
            # Audit the complete network balance without correcting the state.
            # All three branch updates use the same signed junction fluxes, so
            # an internal T exchange must cancel algebraically.  A residual is
            # reported as a discretisation error; it is never injected into a
            # selected cell as an artificial source.
            target_liquid_volume = (
                liquid_volume_before_step - float(G1[-1]) * dt
            )
            balance_correction = target_liquid_volume - (
                float(np.sum(Alt_new)) * dx
                + float(np.sum(Alr_new)) * dz
            )
            junction_liquid_balance_correction += balance_correction

        if not (np.all(np.isfinite(Alt_new)) and np.all(np.isfinite(Alr_new))):
            print(f"  [DIVERGED] t={t:.4f} step={step}", flush=True)
            break

        # Commit the directional conserved state exactly once, only after all
        # coupled liquid/gas/node stages have completed successfully.  The
        # compatibility totals below are then an exact view of that persistent
        # state; no legacy single-column projection is allowed to overwrite it.
        if twostream_active_step:
            if riser_twostream_next is None:
                raise FloatingPointError(
                    "two-stream step completed without a persistent state"
            )
            riser_twostream_state = riser_twostream_next
            if riser_liquid_provenance_next is None:
                raise FloatingPointError(
                    "two-stream step completed without liquid provenance"
                )
            riser_liquid_provenance_state = (
                riser_liquid_provenance_next
            )
            bidirectional_tnode_upward_speed = (
                bidirectional_tnode_upward_speed_next
            )
            taylor_swept_fraction = taylor_swept_fraction_next
            Alr_new = np.asarray(
                riser_twostream_state.liquid_area,
                dtype=float,
            )
            Qlr_new = np.asarray(
                riser_twostream_state.liquid_discharge,
                dtype=float,
            )

        # Liquid leaving the open tower is removed by G1[-1], so the resolved
        # mass balance already accounts for ejection.  Continue only its
        # ballistic trajectory above the computational outlet for comparison
        # with the 2D plume.  No water is added to the riser state.
        last_top_q = max(float(G1[-1]), 0.0)
        liquid_escaped_volume += last_top_q * dt
        rho_step = np.maximum(
            Pr / (R_GAS * T_GAS), rho_atm
        )
        alpha_g_step = np.clip(
            Mgrs_new / np.maximum(
                rho_step * Ar * dz, 1.0e-12
            ),
            0.0,
            0.90,
        )
        alpha_g_step *= np.clip(
            (riser_gas_front - (zr - 0.5 * dz)) / dz,
            0.0,
            1.0,
        )
        step_wtop = _column_material_height(
            Alr_new,
            alpha_g_step,
            Ar,
            dz,
            initial_riser_volume_offset,
        )
        gas_step_idx = np.where(
            alpha_g_step > 0.02
        )[0]
        step_itop = (
            min(float(riser_gas_front), case.riser_height)
            if gas_step_idx.size else 0.0
        )
        Alt, Qlt, Mgt, Jgt = Alt_new, Qlt_new, Mgt_new, Jgt_new
        Alr, Qlr, Mgr, Mgrs, Jgrs = Alr_new, Qlr_new, Mgr_new, Mgrs_new, Jgrs_new
        t += dt; step += 1; dt_prev = dt

        wtop = step_wtop
        geyser_strength = max(geyser_strength, wtop)
        if t >= next_out - 1e-12 or t >= case.t_end - 1e-12:
            Pr_rec, _, _ = _pressure(Alr, np.full(Nr, Ar), Mgr, dz, a2, vent_top=True, p_floor=0.0)
            Pr_rec = np.where(Alr / Ar > 0.08, Pr_rec + RHO_L * G * np.maximum(wtop - zr, 0.0), Pr_rec)
            rho_g_rec = np.maximum(Pr_rec / (R_GAS * T_GAS), rho_atm)
            append_record(t, Alt, Alr, Mgt, Mgr, Mgrs, rho_g_rec)
            rec["pj_head"].append(float((Pj - P_ATM) / (RHO_L * G)))
            # Transducer reads at the pipe AXIS (paper Fig.5 t=0 value = Yfs0 - D/2):
            # subtract the water column between invert and axis -- min(h, D/2) --
            # so a shallow layer (axis in the gas) reads the gas pressure itself.
            h_itr = case.D * float(_depth_frac(min(max(Alt[itr] / A, 0.0), 1.0)))
            rec["tr_head"].append(float((Pt[itr] - P_ATM) / (RHO_L * G)
                                        - min(h_itr, 0.5 * case.D)))
            u_dbg = Qlt / np.maximum(Alt, 1.0e-3 * A)
            ke_cells = 0.5 * RHO_L * Alt * u_dbg * u_dbg * dx
            rec.setdefault("dbg_ke", []).append(float(np.sum(ke_cells)))
            ikm = int(np.argmax(ke_cells))
            rec.setdefault("dbg_ke_argmax", []).append(ikm)
            rec.setdefault("dbg_ke_max", []).append(float(ke_cells[ikm]))
            rec.setdefault("dbg_alpha_kemax", []).append(float(Alt[ikm] / A))
            rec.setdefault("dbg_u_kemax", []).append(float(u_dbg[ikm]))
            rec.setdefault("dbg_u_iv", []).append(float(u_dbg[iv]))
            rec.setdefault("dbg_u_jx", []).append(
                float(u_dbg[junction_west_cell])
            )
            rec.setdefault("dbg_pj", []).append(float((Pj - P_ATM) / (RHO_L * G)))
            rec.setdefault("dbg_riser_base_alpha", []).append(
                float(Alr[0] / Ar)
            )
            rec.setdefault("dbg_riser_base_u", []).append(
                float(Qlr[0] / max(Alr[0], 1.0e-2 * Ar))
            )
            rec.setdefault("dbg_riser_base_head", []).append(
                float((Pr[0] + RHO_L * G * 0.5 * dz - P_ATM) / (RHO_L * G))
            )
            rec.setdefault("dbg_external_interface_x", []).append(
                float(external_horizontal_state.interface_x)
                if external_horizontal_state is not None
                else float("nan")
            )
            h_void_dbg = np.maximum(
                A - np.clip(Alt, 0.0, A),
                coupled_gas_parameters.void_floor_fraction * A,
            )
            h_rho_dbg = Mgt / np.maximum(h_void_dbg * dx, EPS)
            h_resolved_dbg = (
                h_rho_dbg > (
                    coupled_gas_parameters.resolved_density_fraction * rho_atm
                )
            ) & (
                h_rho_dbg < (
                    coupled_gas_parameters.resolved_density_ceiling * rho_atm
                )
            )
            h_ug_dbg = np.where(
                h_resolved_dbg,
                Jgt / np.maximum(Mgt, 1.0e-14),
                0.0,
            )
            ihg = int(np.argmax(np.abs(h_ug_dbg)))
            rec.setdefault("dbg_hgas_u_max_current", []).append(
                float(abs(h_ug_dbg[ihg]))
            )
            rec.setdefault("dbg_hgas_u_max_x", []).append(float(xt[ihg]))
            rec.setdefault("dbg_hgas_u_max_rho_ratio", []).append(
                float(h_rho_dbg[ihg] / rho_atm)
            )
            v_void_dbg = np.maximum(
                Ar - np.clip(Alr, 0.0, Ar),
                coupled_gas_parameters.void_floor_fraction * Ar,
            )
            v_rho_dbg = Mgr / np.maximum(v_void_dbg * dz, EPS)
            v_resolved_dbg = (
                v_rho_dbg > (
                    coupled_gas_parameters.resolved_density_fraction * rho_atm
                )
            ) & (
                v_rho_dbg < (
                    coupled_gas_parameters.resolved_density_ceiling * rho_atm
                )
            )
            v_ug_dbg = np.where(
                v_resolved_dbg,
                Jgrs / np.maximum(Mgr, 1.0e-14),
                0.0,
            )
            ivg = int(np.argmax(np.abs(v_ug_dbg)))
            rec.setdefault("dbg_vgas_u_max_current", []).append(
                float(abs(v_ug_dbg[ivg]))
            )
            rec.setdefault("dbg_vgas_u_max_z", []).append(float(zr[ivg]))
            rec.setdefault("dbg_vgas_u_max_rho_ratio", []).append(
                float(v_rho_dbg[ivg] / rho_atm)
            )
            edge_dbg = 0.0
            for (i0d, i1d) in _regions((1.0 - Alt / A) > 0.05):
                if i0d == 0:
                    edge_dbg = float(1.0 - Alt[min(i1d, Nt - 1)] / A)
                    break
            rec.setdefault("dbg_edge", []).append(edge_dbg)
            rec.setdefault("dbg_bt", []).append(bool(_bt_dbg.get("bt", False)))
            rec.setdefault("dbg_bt_jgassy", []).append(bool(_bt_dbg.get("jgassy", False)))
            rec.setdefault("dbg_bt_ksurf", []).append(int(_bt_dbg.get("ksurf", 0)))
            rec.setdefault("dbg_bt_alpha_surf", []).append(float(_bt_dbg.get("a_surf", 0.0)))
            rec.setdefault("dbg_bt_alpha_core_min", []).append(float(_bt_dbg.get("a_core_min", 0.0)))
            rec.setdefault("dbg_bt_vented", []).append(float(_bt_dbg.get("vented", 0.0)))
            next_out += out_dt
        if verbose and step % 2000 == 0:
            print(
                f"  t={t:.3f} step={step} dt={dt:.1e} "
                f"wtop={wtop:.3f} gmax={geyser_strength:.3f} "
                f"xf={side_t_east_material_front:.3f} "
                f"xtop={side_t_east_topology_front:.3f} "
                f"uf={side_t_east_material_front_velocity:.2f} "
                f"nret={side_t_east_retired_cell_count} "
                f"ut={float(np.max(np.abs(ult))):.2f} "
                f"ur={float(np.max(np.abs(ulr))):.2f} "
                f"ugr={float(np.max(np.abs(ugr_now))):.2f}",
                flush=True,
            )
        if (
            diagnostic_wall_seconds is not None
            and step % 100 == 0
            and time.perf_counter() - wall_start
            >= float(diagnostic_wall_seconds)
        ):
            rec["diagnostic_wall_stop"] = {
                "time": float(t),
                "step": int(step),
                "dt": float(dt),
                "dt_phase": (
                    float(dt_phase) if np.isfinite(dt_phase) else None
                ),
                "dt_junction": (
                    float(dt_junction)
                    if np.isfinite(dt_junction)
                    else None
                ),
                "junction_vertical_flow": float(G1[0]),
                "junction_alpha": float(Alt[junction_west_cell] / A),
            }
            break
        if step >= case.max_steps:
            print("  [MAX_STEPS]", flush=True); break

    rec["geyser_strength"] = geyser_strength
    rec["riser_film_closure"] = {
        "thickness": float(riser_film_thickness),
        "gas_core_area_fraction": float(riser_gas_core_fraction),
        "return_efficiency": float(
            case.vertical_taylor_return_efficiency
        ),
        "laminar_gas_core_area_fraction": float(
            riser_laminar_gas_core_fraction
        ),
        "terminal_film_flow": float(riser_terminal_film_flow),
        "terminal_film_velocity": float(riser_terminal_film_velocity),
        "film_hydraulic_diameter": float(
            riser_film_hydraulic_diameter
        ),
    }
    rec["external_horizontal_handoff_time"] = (
        None
        if external_horizontal_handoff_time is None
        else float(external_horizontal_handoff_time)
    )
    rec["external_horizontal_used"] = bool(
        external_horizontal_solver is not None
    )
    rec["twostream_activated_time"] = (
        None
        if twostream_activated_time is None
        else float(twostream_activated_time)
    )
    rec["dbg_created"] = dbg_created
    return rec


def draw_case_snapshots(case: NetworkCase, rec: dict, out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    tw_w = 0.10 * (case.Dr / case.D) + 0.012
    pipe_h = 0.10
    nF = len(rec["frames_t"])
    if nF == 0:
        return None
    idx = np.linspace(0, nF - 1, min(6, nF)).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(2.3 * len(idx) + 0.6, 5.4), sharey=True)
    C_W, C_A = "#2b7fff", "#eef2f7"
    for ax, k in zip(np.atleast_1d(axes), idx):
        alt = rec["frames_alt"][k]; alr = rec["frames_alr"][k]
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A, edgecolor="0.5", lw=0.6))
        for xi, ai in zip(xt, alt):
            if ai >= 0.5:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, pipe_h, facecolor=C_W, edgecolor="none"))
        ax.add_patch(Rectangle((x_r, 0), tw_w, case.riser_height, facecolor=C_A, edgecolor="0.5", lw=0.6))
        for zi, ai in zip(zr, alr):
            f = float(min(max(ai, 0.0), 1.0))
            if f > 0.02:
                if f >= 0.98:
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), tw_w, dz, facecolor=C_W, edgecolor="none"))
                else:
                    film_w = 0.5 * f * tw_w
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), film_w, dz, facecolor=C_W, edgecolor="none"))
                    ax.add_patch(Rectangle((x_r + tw_w - film_w, zi - 0.5 * dz), film_w, dz, facecolor=C_W, edgecolor="none"))
        ax.set_title(f"t={rec['frames_t'][k]:.2f} s", fontsize=9)
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.05, case.riser_height + 0.1)
        ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].set_ylabel("riser height [m]")
    g = rec["geyser_strength"]
    fig.suptitle(f"V&W(2011): pipe D={case.D*1000:.0f}mm, tower Dt={case.Dr*1000:.1f}mm (Dt/D={case.Dr/case.D:.2f}), "
                 f"Ha0={case.air_head}m, Yfs0={case.init_water_level}m  ->  max tower rise {g:.3f} m (L={case.riser_height})", fontsize=9)
    fig.legend(handles=[Patch(facecolor=C_W, label="water"), Patch(facecolor=C_A, edgecolor="0.5", label="air")],
               loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = out_dir / f"case_snapshots_{tag}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def draw_case_history(case: NetworkCase, rec: dict, out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(rec["t"], rec["wtop"], color="#2b7fff", lw=2.0, label="riser water-surface height [m]")
    ax.plot(rec["t"], rec["pocket_head"], color="#b91c1c", lw=1.6, ls="--", label="trapped-air gauge head [m]")
    ax.axhline(case.riser_height, color="#16a34a", ls=":", lw=1.2, label=f"tower top L={case.riser_height} m")
    ax.set_xlabel("time [s]"); ax.set_ylabel("height / head [m]")
    ax.set_title("Riser water rise and trapped-air head vs time (self-evolving)")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out_dir / f"case_history_{tag}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def make_case_gif(case: NetworkCase, rec: dict, out_dir: Path, tag: str, fps: int = 15, max_frames: int = 90):
    """Animated GIF of the genuine two-fluid simulation: water(blue)/air(white) evolving
    in the horizontal tunnel and the vertical riser (riser bore exaggerated for clarity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    nF = len(rec["frames_t"])
    if nF == 0:
        return None
    sel = np.unique(np.linspace(0, nF - 1, min(max_frames, nF)).astype(int))
    pipe_h = 0.16
    riser_w = 0.16                       # tower bore drawn (widened for visibility)
    C_W, C_A = "#2b7fff", "#f2f4f8"

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    handles = [Patch(facecolor=C_W, label="water"), Patch(facecolor=C_A, edgecolor="0.5", label="air")]

    def draw(k):
        ax.clear()
        alt = rec["frames_alt"][k]; alr = rec["frames_alr"][k]
        # horizontal pipe: draw the cross-sectional liquid fraction as a WATER LAYER from the
        # bottom (height = alpha_l * bore), air above -> the advancing air pocket shows as
        # STRATIFIED flow (air on the crown, water underneath).
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A, edgecolor="0.5", lw=0.8))
        for xi, ai in zip(xt, alt):
            f = float(min(max(ai, 0.0), 1.0))
            wh = float(_depth_frac(f)) * pipe_h
            if f > 1.0e-4:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, wh, facecolor=C_W, edgecolor="none"))
        # vertical tower: centered gas core, symmetric water films.  Both
        # axial fronts are sub-cell positions; never paint a whole mixed cell
        # across the gas nose or the bulk free surface.
        ax.add_patch(Rectangle((x_r, 0), riser_w, case.riser_height, facecolor=C_A, edgecolor="0.5", lw=0.8))
        agr = rec["frames_agr"][k]
        wtop = rec["wtop"][k] if k < len(rec["wtop"]) else 0.0
        itop = (
            rec["frames_itop"][k]
            if k < len(rec.get("frames_itop", []))
            else (rec["itop"][k] if k < len(rec.get("itop", [])) else 0.0)
        )
        water_top = float(np.clip(wtop, 0.0, case.riser_height))
        gas_nose = float(np.clip(itop, 0.0, water_top))
        if water_top > 0.0:
            ax.add_patch(Rectangle(
                (x_r, 0.0), riser_w, water_top,
                facecolor=C_W, edgecolor="none",
            ))
        for zi, ag in zip(zr, agr):
            z0 = max(float(zi - 0.5 * dz), 0.0)
            z1 = min(float(zi + 0.5 * dz), gas_nose)
            if z1 <= z0:
                continue
            gfrac = float(min(max(ag, 0.0), 1.0))
            if gfrac > 0.01:
                gas_w = math.sqrt(gfrac) * riser_w
                ax.add_patch(Rectangle((x_r + 0.5 * (riser_w - gas_w), z0),
                                       gas_w, z1 - z0, facecolor=C_A, edgecolor="none"))
        if water_top - gas_nose > 1.0e-9:
            ax.plot([x_r, x_r + riser_w], [water_top, water_top],
                    color="#1d4ed8", lw=0.7)
        ax.text(0.02, 0.02, f"pipe D={case.D*1000:.0f} mm, tower Dt={case.Dr*1000:.1f} mm (Dt/D={case.Dr/case.D:.2f}); "
                            f"tower width drawn enlarged for visibility",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.45")
        ax.plot([x_r - 0.05, x_r + riser_w + 0.05], [case.riser_height, case.riser_height], color="#ef4444", ls="--", lw=1.0)
        ax.text(0.02, 0.96, f"t = {rec['frames_t'][k]:.2f} s    riser water height = {wtop:.2f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.05, case.riser_height + 0.15)
        ax.set_xlabel("horizontal distance [m]   (closed air-pipe at left, closed end at right; tower = vertical column)")
        ax.set_ylabel("height above tunnel invert [m]")
        ax.set_yticks(np.arange(0, case.riser_height + 0.01, 0.5))
        ax.set_title(f"V&W(2011) geyser — decoupled two-fluid  |  D={case.D*1000:.0f}mm, Dt={case.Dr*1000:.1f}mm "
                     f"(Dt/D={case.Dr/case.D:.2f}), Ha0={case.air_head}m, Yfs0={case.init_water_level}m, L={case.riser_height}m", fontsize=9)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=sel, interval=1000.0 / fps, blit=False)
    p = out_dir / f"case_{tag}.gif"
    anim.save(p, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return p


def make_case_frames(case: NetworkCase, rec: dict, out_dir: Path, tag: str, max_frames: int = 120):
    """Write a frame sequence for manual HTML browsing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    riser_frames_dir = out_dir / "riser_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    riser_frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    for old in riser_frames_dir.glob("riser_*.png"):
        old.unlink()

    xt = rec["xt"]; zr = rec["zr"]; dx = rec["dx"]; dz = rec["dz"]
    x_r = case.x_riser
    nF = len(rec["frames_t"])
    if nF == 0:
        return []

    sel = np.unique(np.linspace(0, nF - 1, min(max_frames, nF)).astype(int))
    # True x-y scale: pipe diameter and tower diameter are drawn with their actual
    # metric widths.  The 12.7 mm tower is therefore intentionally very thin.
    pipe_h = case.D
    pipe_bottom = -0.5 * case.D
    pipe_crown = 0.5 * case.D
    riser_w = case.Dr
    C_W, C_A = "#2b7fff", "#f2f4f8"
    handles = [Patch(facecolor=C_W, label="water"), Patch(facecolor=C_A, edgecolor="0.5", label="air")]
    frames = []

    def draw_riser_section(
        ax, x0, width, wtop, itop, alpha_g, jet_height=0.0,
        base_z=0.0,
    ):
        ax.add_patch(Rectangle((x0, base_z), width, case.riser_height, facecolor=C_A, edgecolor="0.5", lw=0.9))
        water_top = float(np.clip(wtop, 0.0, case.riser_height))
        gas_nose = float(np.clip(itop, 0.0, water_top))
        if water_top > 0.0:
            ax.add_patch(Rectangle(
                (x0, base_z), width, water_top,
                facecolor=C_W, edgecolor="none",
            ))
        for zi, ag in zip(zr, alpha_g):
            z0 = max(float(zi - 0.5 * dz), 0.0)
            z1 = min(float(zi + 0.5 * dz), gas_nose)
            if z1 <= z0:
                continue
            gas_fraction = float(np.clip(ag, 0.0, 1.0))
            gas_w = math.sqrt(gas_fraction) * width
            if gas_w > 0.001 * width:
                ax.add_patch(Rectangle(
                    (x0 + 0.5 * (width - gas_w), base_z + z0),
                    gas_w, z1 - z0,
                    facecolor=C_A, edgecolor="none",
                ))
        if water_top - gas_nose > 1.0e-9:
            ax.plot(
                [x0, x0 + width],
                [base_z + water_top, base_z + water_top],
                color="#1d4ed8", lw=0.7,
            )
        if jet_height > case.riser_height:
            jet_w = 0.55 * width
            ax.add_patch(Rectangle(
                (
                    x0 + 0.5 * (width - jet_w),
                    base_z + case.riser_height,
                ),
                jet_w,
                jet_height - case.riser_height,
                facecolor=C_W,
                edgecolor="none",
            ))

    for frame_no, k in enumerate(sel):
        fig, ax = plt.subplots(figsize=(12.0, 3.1))
        alt = rec["frames_alt"][k]
        ax.add_patch(Rectangle((0, pipe_bottom), case.L_tunnel, pipe_h, facecolor=C_A, edgecolor="0.5", lw=0.8))
        for xi, ai in zip(xt, alt):
            f = float(min(max(ai, 0.0), 1.0))
            # Area fraction is not depth fraction in a circular conduit.
            wh = float(_depth_frac(f)) * pipe_h
            if f > 1.0e-4:
                ax.add_patch(Rectangle((xi - 0.5 * dx, pipe_bottom), dx, wh, facecolor=C_W, edgecolor="none"))
        riser_x0 = x_r - 0.5 * riser_w
        wtop = rec["wtop"][k] if k < len(rec["wtop"]) else 0.0
        jet_height = (
            rec["jet_height"][k]
            if k < len(rec.get("jet_height", []))
            else 0.0
        )
        pocket_head = rec["pocket_head"][k] if k < len(rec["pocket_head"]) else 0.0
        itop = rec["frames_itop"][k] if k < len(rec.get("frames_itop", [])) else (rec["itop"][k] if k < len(rec["itop"]) else 0.0)
        core_mass = rec["frames_core_mass"][k] if k < len(rec.get("frames_core_mass", [])) else 0.0
        if k < len(rec.get("frames_agr", [])):
            agr = rec["frames_agr"][k]
        else:
            agr = np.clip(1.0 - rec["frames_alr"][k], 0.0, 1.0)
        draw_riser_section(
            ax, riser_x0, riser_w, wtop, itop, agr, jet_height,
            base_z=pipe_crown,
        )
        tower_top_y = pipe_crown + case.riser_height
        ax.plot([riser_x0 - 0.05, riser_x0 + riser_w + 0.05], [tower_top_y, tower_top_y],
                color="#ef4444", ls="--", lw=1.0)
        ax.plot([case.L_up, case.L_up], [pipe_bottom, pipe_crown + 0.008], color="#111827", ls=":", lw=0.9)
        ax.text(case.L_up, pipe_crown + 0.018, "valve x=0.546m", ha="center", va="bottom", fontsize=7)
        ax.text(x_r, tower_top_y + 0.045, "tower x=3.516m", ha="center", va="bottom", fontsize=7)
        ax.text(0.02, 0.96, f"t = {rec['frames_t'][k]:.2f} s    water surface = {wtop:.2f} m    gas front = {itop:.2f} m    jet top = {jet_height:.2f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.text(0.02, 0.02, f"true x-y scale: pipe D={case.D*1000:.0f} mm, tower Dt={case.Dr*1000:.1f} mm "
                            f"(Dt/D={case.Dr/case.D:.2f})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.45")
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(
            pipe_bottom - 0.04,
            max(tower_top_y + 0.35, 1.0),
        )
        ax.set_xlabel("horizontal distance [m]   (closed air-pipe at left, closed end at right; tower = vertical column)")
        ax.set_ylabel("vertical coordinate y [m]")
        ax.set_yticks(np.arange(0, case.riser_height + 0.01, 0.2))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"V&W(2011) selected test — decoupled two-fluid  |  D={case.D*1000:.0f}mm, "
                     f"Dt={case.Dr*1000:.1f}mm, Ha0={case.air_head}m, Yfs0={case.init_water_level}m",
                     fontsize=9)
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        name = f"frame_{frame_no:04d}.png"
        fig.savefig(frames_dir / name, dpi=130)
        plt.close(fig)

        zfig, zax = plt.subplots(figsize=(3.0, 6.0))
        zoom_w = 0.16
        draw_riser_section(
            zax, -0.5 * zoom_w, zoom_w, wtop, itop, agr,
            jet_height,
        )
        zax.plot([-0.58 * zoom_w, 0.58 * zoom_w], [case.riser_height, case.riser_height],
                 color="#ef4444", ls="--", lw=1.0)
        if itop > 0.0:
            zax.plot([-0.5 * zoom_w, 0.5 * zoom_w], [itop, itop], color="#f97316", ls=":", lw=1.2)
        zax.text(0.03, 0.97, f"t={rec['frames_t'][k]:.2f}s\nwater={wtop:.3f}m\ngas front={itop:.3f}m\njet top={jet_height:.3f}m\ngas={core_mass*1e6:.3f}mg",
                 transform=zax.transAxes, ha="left", va="top", fontsize=9)
        zax.set_xlim(-0.095, 0.095)
        zax.set_ylim(
            -0.015,
            max(case.riser_height + 0.40, 1.0),
        )
        zax.set_aspect("auto")
        zax.set_xlabel("enlarged riser section")
        zax.set_ylabel("height [m]")
        zax.set_title("vertical riser zoom\nsymmetric layout from 1D alpha_g", fontsize=9)
        zax.legend(handles=handles, loc="lower center", frameon=False, fontsize=8, ncol=2)
        zfig.tight_layout()
        riser_name = f"riser_{frame_no:04d}.png"
        zfig.savefig(riser_frames_dir / riser_name, dpi=140)
        plt.close(zfig)
        frames.append({
            "file": f"frames/{name}",
            "riserFile": f"riser_frames/{riser_name}",
            "time": float(rec["frames_t"][k]),
            "wtop": float(wtop),
            "jetHeight": float(jet_height),
            "topQ": float(
                rec["top_q"][k]
                if k < len(rec.get("top_q", []))
                else 0.0
            ),
            "itop": float(itop),
            "coreMassMg": float(core_mass * 1.0e6),
            "head": float(pocket_head),
        })
    return frames


def make_html_report(out_dir: Path, regen: bool = False):
    """Manual frame viewer for the single selected simulation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_manifest = out_dir / f"{SELECTED_TAG}_frames.json"
    if regen or not frames_manifest.exists():
        case = selected_case(t_end=12.0)
        rec = run_network(case, verbose=False)
        frames = make_case_frames(case, rec, out_dir, SELECTED_TAG, max_frames=90)
        frames_manifest.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        frames = json.loads(frames_manifest.read_text(encoding="utf-8"))

    frames_js = json.dumps(frames, ensure_ascii=False)

    # ---- paper-comparison section (built by compare_selected_case_vs_paper.py)
    dig = HERE / "paper_reference" / "digitized"
    for src_name, dst_name in (("fig6_center_panel.png", "paper_fig6_center.png"),
                               ("fig8_center_panel.png", "paper_fig8_center.png")):
        src = dig / src_name
        if src.exists():
            (out_dir / dst_name).write_bytes(src.read_bytes())
    mfile = out_dir / "comparison_metrics.json"
    cmp_html = ""
    if mfile.exists():
        mm = json.loads(mfile.read_text(encoding="utf-8"))
        mo, pa = mm["model"], mm["paper"]
        tsc = mm["Tstar_scale_s"]

        def _f(v, nd=2):
            return "—" if v is None else f"{v:.{nd}f}"

        def _sec(v):
            return "—" if v is None else f"{v * tsc:.1f} s"

        rows = [
            ("压头平台 H*（T*≈1–4 中位）", _f(pa.get("Hstar_plateau")), _f(mo.get("Hstar_plateau_tr")), ""),
            ("气水界面进入塔（Y*int 离底）", f"T*={_f(pa.get('int_liftoff_Tstar'))}（{_sec(pa.get('int_liftoff_Tstar'))}）",
             f"T*={_f(mo.get('int_liftoff_Tstar'))}（{_sec(mo.get('int_liftoff_Tstar'))}）", "模型偏早"),
            ("气水界面到达塔顶", f"T*={_f(pa.get('int_top_Tstar'))}（{_sec(pa.get('int_top_Tstar'))}）",
             f"T*={_f(mo.get('int_top_Tstar'))}（{_sec(mo.get('int_top_Tstar'))}）", "接近"),
            ("自由水面到顶（喷发/溢出）", f"T*={_f(pa.get('geyser_Tstar'))}（{_sec(pa.get('geyser_Tstar'))}）",
             f"T*={_f(mo.get('geyser_Tstar'))}（{_sec(mo.get('geyser_Tstar'))}）", "模型偏晚"),
            ("压头骤降（气囊排空）", f"T*={_f(pa.get('Hstar_drop_Tstar'))}（{_sec(pa.get('Hstar_drop_Tstar'))}）",
             f"T*≈{_f(mo.get('int_top_Tstar'))}（气体贯通塔顶）", ""),
        ]
        trs = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
        cmp_html = f"""
<div class="panel" style="margin-bottom:16px">
  <h2 style="margin-top:0">论文实验 vs 模型对比（本工况：Dt=12.7&nbsp;mm，Ha0=0.610&nbsp;m，WL=0.356&nbsp;m）</h2>
  <p>实验数据取自论文 <b>Fig.6</b>（变送器压头 H*，位于阀下游 1.07 m，即 x=1.616 m）与
  <b>Fig.8</b>（塔内自由水面 Y*<sub>fs</sub> 与气水界面 Y*<sub>int</sub>）的<b>中心面板</b>（正是本工况），
  用图像数字化提取后与模型输出画在同一坐标系里。时间与长度均按论文方式无量纲化：
  T*&nbsp;=&nbsp;t·√(gD<sub>t</sub>)/L，H*=H/L，Y*=Y/L（L=0.610&nbsp;m；T*=1 对应 {tsc:.2f}&nbsp;s）。</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">
    <div><h3 style="margin:4px 0">论文原图 Fig.6 中心面板（扫描）</h3><img src="paper_fig6_center.png" alt="paper fig6 center"></div>
    <div><h3 style="margin:4px 0">论文原图 Fig.8 中心面板（扫描）</h3><img src="paper_fig8_center.png" alt="paper fig8 center"></div>
  </div>
  <h3 style="margin:14px 0 4px 0">叠加对比 1：变送器压头 H*(T*)</h3>
  <img src="comparison_pressure.png" alt="pressure comparison">
  <h3 style="margin:14px 0 4px 0">叠加对比 2：塔内水面与气水界面 Y*(T*)</h3>
  <img src="comparison_levels.png" alt="levels comparison">
  <h3 style="margin:14px 0 4px 0">事件时刻与量值对照</h3>
  <table style="border-collapse:collapse;width:100%;font-size:13px">
    <tr style="background:#f3f4f6"><th style="text-align:left;padding:6px 8px">指标</th>
      <th style="text-align:left;padding:6px 8px">论文实验（数字化）</th>
      <th style="text-align:left;padding:6px 8px">模型</th>
      <th style="text-align:left;padding:6px 8px">备注</th></tr>
    {trs}
  </table>
  <p style="font-size:13px;color:#6b7280">吻合点：气水界面到顶时刻（T*≈4.1 vs 4.17）、界面爬升段的整体形态、
  最终水面到顶（本工况实验与模型都到达塔顶 Y*=1）。
  差异点：(1) 模型里气体<b>过早</b>进入塔底（T*≈2.3 vs 实验≈3.6）但爬升更慢，两者在塔顶汇合；
  (2) 实验压头在 0&lt;T*&lt;4 保持 ≈0.76L 平台后骤降归零，模型的变送器局部压头水平偏低（中位≈0.51L）
  且早期出现塔水位先降、气囊先增压的反向瞬态——这是简化节点耦合与水锤闭合在骤开阀初期的已知局限；
  (3) 实验压头骤降发生在界面尚未到顶前（T*≈4.05），模型要到气体贯通塔顶才泄压。
  数字化中间产物见 <code>paper_reference/digitized/</code>（含 debug 叠加图，可核对取点）。</p>
</div>"""

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>V&amp;W (2011) selected-test frame viewer</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}
.wrap{{max-width:1180px;margin:24px auto;padding:0 18px}}
.panel{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:16px}}
img{{width:100%;border:1px solid #ddd;border-radius:10px;background:#fff}}
.viewer{{display:grid;grid-template-columns:minmax(0,2.9fr) minmax(260px,1fr);gap:14px;align-items:start}}
.viewer h3{{margin:0 0 8px 0;font-size:15px}}
.viewer .hint{{font-size:12px;color:#6b7280;margin:6px 0 0 0}}
button{{padding:8px 14px;margin:6px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer}}
input{{width:68%;vertical-align:middle}}
.meta{{display:flex;gap:18px;margin:10px 0;font-weight:700;flex-wrap:wrap}}
p{{line-height:1.55;color:#374151}}
@media(max-width:900px){{.viewer{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>V&amp;W(2011) geyser 单工况复现 — 论文解耦两流体算法</h1>
<p>唯一工况：<code>Dt=12.7 mm</code>, <code>Ha0=0.610 m</code>, <code>Yfs0=0.356 m</code>。
管道布置和初始条件来自 V&amp;W2011 JHE；{FORMULATION_NOTE} 蓝色为水，浅灰为气。水平管按截面含水率显示分层气团推进；右侧放大图按一维体积分数把气相对称放在中间、液相分布在两侧。</p>
{cmp_html}
<div class="panel">
  <div class="meta"><span id="idx"></span><span id="time"></span><span id="wtop"></span><span id="itop"></span><span id="coremass"></span><span id="head"></span></div>
  <div class="viewer">
    <div><h3>全局 1:1 视图</h3><img id="frame" alt="simulation frame"><p class="hint">水平管和竖管按真实坐标比例显示，所以 12.7 mm 竖管会很细。</p></div>
    <div><h3>竖管放大同步视图</h3><img id="riserFrame" alt="riser zoom frame"><p class="hint">此图使用同一套 alpha_g(z,t)：气相按体积分数放在中间，液相对称分布在两侧；和左侧滑块/播放同步跳帧。</p></div>
  </div>
  <div><button id="prev">上一帧</button><button id="play">播放</button><input id="slider" type="range"><button id="next">下一帧</button></div>
</div>
</div><script>
const frames={frames_js};
let i=0,timer=null;
const img=document.getElementById('frame'),riserImg=document.getElementById('riserFrame'),slider=document.getElementById('slider'),play=document.getElementById('play');
slider.min=0;slider.max=Math.max(0,frames.length-1);slider.value=0;
function show(k){{
  i=Math.max(0,Math.min(frames.length-1,k));
  const f=frames[i];
  img.src=f.file; riserImg.src=f.riserFile; slider.value=i;
  idx.textContent=`帧 ${{i+1}}/${{frames.length}}`;
  time.textContent=`t=${{f.time.toFixed(2)}} s`;
  wtop.textContent=`塔内水位=${{f.wtop.toFixed(3)}} m`;
  itop.textContent=`竖管气相前沿=${{(f.itop||0).toFixed(3)}} m`;
  coremass.textContent=`竖管解析气体质量=${{(f.coreMassMg||0).toFixed(3)}} mg`;
  head.textContent=`气囊压力头=${{f.head.toFixed(3)}} m`;
}}
function stop(){{if(timer)clearInterval(timer);timer=null;play.textContent='播放';}}
prev.onclick=()=>{{stop();show(i-1)}};
next.onclick=()=>{{stop();show(i+1)}};
slider.oninput=e=>{{stop();show(Number(e.target.value))}};
play.onclick=()=>{{if(timer){{stop();return}};play.textContent='暂停';timer=setInterval(()=>show(i>=frames.length-1?0:i+1),260)}};
document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft'){{stop();show(i-1)}}if(e.key==='ArrowRight'){{stop();show(i+1)}}}});
show(0);
</script></body></html>"""
    p = out_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


def validate(out_dir: Path, t_end: float = 8.0):
    """V&W(2011) JHE: sweep tower diameter at the paper's reference condition
    (H_a0 = 0.305 m, Y_fs0 = 0.254 m) and compare the tower free-surface rise
    (normalized by tower length L) against the paper's reported values (Fig 7/8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    Dts = [0.0571, 0.0444, 0.0254, 0.0127]
    paper_rise_frac = {0.0571: 0.10, 0.0444: 0.20, 0.0254: 0.40, 0.0127: 1.00}  # >=1 => geyser/spill
    L = 0.610
    rows = []
    for Dt in Dts:
        case = NetworkCase(Dr=Dt, air_head=0.305, init_water_level=0.254, t_end=t_end)
        rec = run_network(case, verbose=False)
        ymax = rec["geyser_strength"]
        rise_model = (ymax - case.init_water_level) / L
        rows.append((Dt, Dt / 0.094, ymax, rise_model, paper_rise_frac[Dt]))
        print(f"Dt={Dt*1000:4.1f}mm Dt/D={Dt/0.094:.3f}  model rise/L={rise_model:.2f}  "
              f"paper rise/L~{paper_rise_frac[Dt]:.2f}", flush=True)
    with (out_dir / "validation_VW2011JHE.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dt_m", "Dt_over_D", "Yfs_max_m", "rise_frac_model", "rise_frac_paper"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = [r[1] for r in rows]
    ax.plot(x, [r[3] for r in rows], "o-", color="#d62728", lw=2, label="model rise / L")
    ax.plot(x, [min(r[4], 1.0) for r in rows], "s--", color="#1f77b4", lw=2, label="paper rise / L (Fig 7/8)")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2, label="spill = geyser")
    ax.set_xlabel("Dt / D"); ax.set_ylabel("tower free-surface rise / L")
    ax.set_title("V&W(2011) JHE: free-surface rise vs tower diameter\n(Ha0=0.305 m, Yfs0=0.254 m)")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "validation_VW2011JHE.png", dpi=150); plt.close(fig)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("schematic", "case", "validate", "report"), default="report")
    ap.add_argument("--Dr", type=float, default=0.0127, help="tower diameter Dt [m]")
    ap.add_argument("--air-head", type=float, default=0.610, help="air-phase initial pressure head Ha0 [m]")
    ap.add_argument("--water", type=float, default=0.356, help="initial tower water level Yfs0 [m]")
    ap.add_argument("--tend", type=float, default=8.0)
    ap.add_argument("--regen", action="store_true", help="regenerate GIFs when building the HTML report")
    args = ap.parse_args()
    out = HERE / "outputs" / "vw2011_network"
    if args.mode == "schematic":
        p = draw_apparatus_schematic(out)
        print(f"schematic -> {p}")
    elif args.mode == "case":
        case = NetworkCase(Dr=args.Dr, air_head=args.air_head, init_water_level=args.water, t_end=args.tend)
        rec = run_network(case)
        print(f"max tower free-surface rise = {rec['geyser_strength']:.3f} m  (tower L={case.riser_height} m)")
    elif args.mode == "validate":
        validate(out, t_end=args.tend)
    elif args.mode == "report":
        p = make_html_report(out, regen=args.regen)
        print(f"report -> {p}")
