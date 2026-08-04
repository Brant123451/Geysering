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

Both the horizontal pipe and vertical tower are advanced with the present paper's
decoupled two-fluid area formulation: liquid area/discharge evolution, conserved
    gas mass with an isothermal EOS pressure, gravity, and the pressurized-water
    water-hammer branch when the liquid reach is full.  The vertical tower is treated
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
from pathlib import Path

import numpy as np

from casea_coupled_gas_network import (
    CoupledGasParameters,
    OpenIsothermalGasInventory,
    _mass_backed_gas_topology,
    advance_lumped_pocket_vertical_network,
    advance_coupled_gas_network,
    junction_mouth_area,
)
from casea_horizontal_liquid_operator import (
    decoupled_lambda_and_derivative,
    HorizontalLiquidParameters,
    pressure_potential_state,
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


def _tpa_muscl_faces(area, discharge, area_full, diameter):
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
        area < 0.995 * area_full, free_surface_cap, U_FLUX_MAX
    )
    velocity = np.clip(velocity, -velocity_cap, velocity_cap)
    slope_a = np.zeros(n)
    slope_u = np.zeros(n)
    if n > 2:
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
    """Companion-model stratified restoring coefficient, Eq. (A31).

    The conserved tunnel gas arrays store cell mass and cell momentum, whereas
    Eq. (A31) uses phase density and velocity.  This adapter performs only that
    conversion; no fitted wave amplitude or propagation speed is imposed.
    """

    area_raw = np.asarray(area_l, dtype=float)
    area = np.clip(
        area_raw,
        1.0e-6 * area_full,
        0.995 * area_full,
    )
    discharge = np.asarray(discharge_l, dtype=float)
    mass = np.maximum(np.asarray(gas_mass, dtype=float), 0.0)
    momentum = np.asarray(gas_momentum, dtype=float)
    raw_void = np.maximum(
        area_full - np.clip(area_raw, 0.0, area_full), 0.0
    )
    area_g = np.maximum(raw_void, 1.0e-4 * area_full)
    rho_atm = P_ATM / (R_GAS * T_GAS)
    mass_consistent = _mass_backed_gas_topology(
        raw_void,
        mass,
        full_area=area_full,
        cell_width=cell_width,
        rho_reference=rho_atm,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    rho_g_raw = mass / np.maximum(area_g * cell_width, EPS)
    rho_g = np.where(
        mass_consistent,
        np.clip(rho_g_raw, 0.2 * rho_atm, 12.0 * rho_atm),
        rho_atm,
    )
    u_g_raw = np.where(
        mass > 1.0e-14,
        momentum / np.maximum(mass, EPS),
        0.0,
    )
    u_l = np.clip(
        discharge / np.maximum(area, EPS), -U_FLUX_MAX, U_FLUX_MAX
    )
    u_g = np.where(
        mass_consistent,
        np.clip(u_g_raw, -U_FLUX_MAX, U_FLUX_MAX),
        u_l,
    )
    depth = diameter * _depth_frac(np.clip(area / area_full, 0.0, 1.0))
    top_width = 2.0 * np.sqrt(
        np.maximum(depth * (diameter - depth), 0.0)
    )
    zeta = 1.0 / np.maximum(top_width, 1.0e-10 * diameter)
    p_g = rho_g * R_GAS * T_GAS
    h_g = (p_g - P_ATM) / (RHO_L * G)
    coefficient = (
        2.0 * G * h_g / area
        + (RHO_L - rho_g) / RHO_L * G * zeta
        - rho_g / RHO_L * (u_g - u_l) ** 2 / area_g
    )
    if not np.all(np.isfinite(coefficient)):
        raise FloatingPointError(
            "non-finite Eq. (A31) coefficient in T-junction wave branch"
        )
    return coefficient


def _connected_stratified_potential_offsets(
    area,
    discharge,
    gas_mass,
    gas_momentum,
    mass_supported,
    params,
):
    """Return one pressure-potential gauge per connected gas component.

    ``Psi`` is defined up to a spatial constant inside one barotropic branch.
    That constant must be common to the complete connected pocket; choosing it
    independently from each cell's frozen gas mass creates an O(100 kPa)
    numerical jump at the fitted gas/full-water interface.  Fix the component
    gauge by matching its boundary liquid pressure potential to the adjacent
    resolved elastic-liquid trace.  This is a traction-continuity condition,
    not a fitted wave amplitude or a presentation filter.
    """

    area_a = np.asarray(area, dtype=float)
    q_a = np.asarray(discharge, dtype=float)
    mass_a = np.asarray(gas_mass, dtype=float)
    momentum_a = np.asarray(gas_momentum, dtype=float)
    support = np.asarray(mass_supported, dtype=bool)
    if not (
        area_a.shape == q_a.shape == mass_a.shape
        == momentum_a.shape == support.shape
    ):
        raise ValueError("connected-potential fields must have equal shape")
    offsets = np.zeros_like(area_a)
    if not np.any(support):
        return offsets

    safe_area = np.maximum(area_a, 1.0e-9 * params.area_full)
    raw_lambda = np.zeros_like(area_a)
    lam, _ = decoupled_lambda_and_derivative(
        safe_area[support],
        q_a[support],
        mass_a[support],
        momentum_a[support],
        params,
    )
    raw_lambda[support] = lam
    raw_potential = 0.5 * raw_lambda * safe_area * safe_area

    for i0, i1 in _regions(support):
        candidates = []
        if i0 > 0 and not support[i0 - 1]:
            target = pressure_potential_state(
                max(float(area_a[i0 - 1]), 1.0e-9 * params.area_full),
                float(q_a[i0 - 1]),
                0.0,
                0.0,
                False,
                params,
            )
            candidates.append(
                float(target.potential) - float(raw_potential[i0])
            )
        if i1 < area_a.size and not support[i1]:
            target = pressure_potential_state(
                max(float(area_a[i1]), 1.0e-9 * params.area_full),
                float(q_a[i1]),
                0.0,
                0.0,
                False,
                params,
            )
            candidates.append(
                float(target.potential) - float(raw_potential[i1 - 1])
            )
        # A component touching only a physical open boundary has no adjacent
        # liquid traction to set its gauge; the natural companion-model
        # primitive is then 0.5*Lambda*A^2 (zero offset).
        component_offset = (
            math.fsum(candidates) / len(candidates)
            if candidates
            else 0.0
        )
        offsets[i0:i1] = component_offset
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
):
    """Topology-aware MUSCL-HLLC liquid flux.

    The conserved TPA area has two different meanings.  In a cell carrying
    resolved gas mass it is the *physical* wetted area and follows the circular
    free-surface pressure law.  In a gas-free cell it is elastic storage in a
    still liquid-full pipe and follows the water-hammer continuation through
    ``A=Af``.  Treating every ``A<Af`` state as a free surface nucleates gas in
    a rarefaction; using an unshifted elastic flux creates an O(a^2 Af) jump at
    the moving interface.  On a genuine stratified cell the liquid momentum
    flux is the companion model's

    ``Q_l**2/A_l + 0.5*Lambda_d*A_l**2``.

    Thus gas pressure, hydrostatic--buoyancy restoration, and slip enter the
    same conservative flux and wave speed.  Applying only ``A_l dp_g/dx`` as a
    separate source loses the finite gas-pressure restoring celerity whenever
    the connected pocket pressure is nearly uniform; in Case A that error lets
    the post-arrival holdup collapse into a deep trough immediately upstream of
    the side tee.
    """

    al, ql, ar, qr = _tpa_muscl_faces(
        area, discharge, area_full, diameter
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
    support_g = np.concatenate(
        ([gas_supported[0]], gas_supported, [gas_supported[-1]])
    )
    pressurised_l = ~support_g[:-1]
    pressurised_r = ~support_g[1:]

    # The pressure potential and its characteristic speed must come from the
    # same Jacobian.  The earlier block used ``sqrt(Lambda*A)`` while the flux
    # contained ``0.5*Lambda(A)*A**2``; omitting ``dLambda/dA`` under-estimated
    # the spectral radius by as much as an order of magnitude and made the
    # MUSCL spatial operator unstable under Forward Euler.  Reconstruct the
    # caller-owned gas conserved variables piecewise constantly at each face,
    # then evaluate the complete tangent modulus on both liquid traces.
    mass_cell = np.asarray(gas_mass, dtype=float)
    momentum_cell = np.asarray(gas_momentum, dtype=float)
    mass_ghost = np.concatenate(
        ([mass_cell[0]], mass_cell, [mass_cell[-1]])
    )
    momentum_ghost = np.concatenate(
        ([momentum_cell[0]], momentum_cell, [momentum_cell[-1]])
    )
    mass_l = mass_ghost[:-1]
    mass_r = mass_ghost[1:]
    momentum_l = momentum_ghost[:-1]
    momentum_r = momentum_ghost[1:]

    dry = 1.0e-9 * area_full
    al_eval = np.maximum(al, dry)
    ar_eval = np.maximum(ar, dry)
    ql_eval = np.where(al > dry, ql, 0.0)
    qr_eval = np.where(ar > dry, qr, 0.0)
    liquid_params = HorizontalLiquidParameters(
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
    component_offset_cell = _connected_stratified_potential_offsets(
        area_cell,
        np.asarray(discharge, dtype=float),
        mass_cell,
        momentum_cell,
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
    pressure_l = pressure_potential_state(
        al_eval,
        ql_eval,
        mass_l,
        momentum_l,
        ~pressurised_l,
        liquid_params,
        stratified_potential_offset=component_offset_ghost[:-1],
    )
    pressure_r = pressure_potential_state(
        ar_eval,
        qr_eval,
        mass_r,
        momentum_r,
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
    # The published stratified block uses scalar Rusanov dissipation with the
    # Lambda_d characteristic speed.  Retain HLLC only on an elastic or mixed-
    # topology face, where its contact resolution is needed by the fitted
    # pressurised front.  Using HLLC inside the complete stratified pocket
    # removes the model's prescribed block dissipation and lets the strong
    # gas-pressure restoring wave overshoot into an elastically overfilled T
    # cell.
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


def _liquid_surface_height(z, dz, Al, A, threshold=0.08):
    """Top of any resolved liquid in the riser, including thin side films."""
    idx = np.where(Al / A > threshold)[0]
    return float(z[idx[-1]] + 0.5 * dz) if idx.size else 0.0


def _column_material_height(Al, alpha_g, A, dz, initial_volume_offset=0.0):
    """Free-surface height from conservative liquid-plus-gas column volume."""
    occupied = np.clip(Al / A + alpha_g, 0.0, 1.0)
    volume = float(np.sum(occupied) * A * dz) - initial_volume_offset
    return float(max(volume / A, 0.0))


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
    if (
        full_area <= 0.0
        or diameter <= 0.0
        or film_thickness <= 0.0
        or kinematic_viscosity <= 0.0
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
    film_hydraulic_diameter = 2.0 * film_thickness
    reynolds = (
        np.abs(velocity) * film_hydraulic_diameter / kinematic_viscosity
    )
    laminar_rate = np.full_like(
        velocity,
        32.0 * kinematic_viscosity / (film_hydraulic_diameter**2),
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
    nu_face = molecular_viscosity + (coefficient * diameter) ** 2 * np.abs(gradient)
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

    # Retain the momentum collocated with liquid that already occupies its
    # target volume.  Move the excess parcels into target deficits with their
    # volume-weighted velocity; this is conservative whenever the tower has
    # sufficient capacity, which is the normal confined-bubble state.
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


def _sweep_vertical_material_slice_to_junction(
    area,
    discharge,
    *,
    old_front_height: float,
    new_front_height: float,
    gas_core_area_fraction: float,
    full_area: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Open a swept riser slice and return its liquid to the side tee.

    For a side-fed confined bubble, gas entering upward and the annular liquid
    film returning downward share the same T opening.  Removing the swept liquid
    and packing it above the nose gives the correct inventory but the wrong
    topology: it raises the complete upper column instead of producing the
    counter-current return seen in the two-dimensional solution.  This local
    cut-cell operation removes only the newly swept volume, preserves the
    remaining cell velocity, and returns the removed parcel volume and axial
    velocity to the caller for conservative deposition in the horizontal T
    footprint.
    """

    a = np.maximum(np.asarray(area, dtype=float), 0.0).copy()
    q = np.asarray(discharge, dtype=float).copy()
    if a.shape != q.shape or a.ndim != 1:
        raise ValueError("vertical sweep arrays must be equal and one-dimensional")
    if (
        not 0.0 <= gas_core_area_fraction < 1.0
        or full_area <= 0.0
        or dz <= 0.0
    ):
        raise ValueError("invalid vertical sweep geometry")
    old_front = max(float(old_front_height), 0.0)
    new_front = max(float(new_front_height), old_front)
    if a.size == 0 or new_front <= old_front + 1.0e-15:
        return a, q, 0.0, 0.0

    cell_bottom = np.arange(a.size, dtype=float) * dz
    old_fraction = np.clip((old_front - cell_bottom) / dz, 0.0, 1.0)
    new_fraction = np.clip((new_front - cell_bottom) / dz, 0.0, 1.0)
    swept_fraction = np.maximum(new_fraction - old_fraction, 0.0)
    removable_area = np.minimum(
        gas_core_area_fraction * full_area * swept_fraction,
        a,
    )
    returned_volume = float(np.sum(removable_area) * dz)
    if returned_volume <= 1.0e-16:
        return a, q, 0.0, 0.0

    returned_momentum = 0.0
    for index in np.flatnonzero(removable_area > 0.0):
        old_area = float(a[index])
        remove = float(removable_area[index])
        fraction = remove / max(old_area, EPS)
        returned_momentum += float(q[index]) * dz * fraction
        a[index] = old_area - remove
        q[index] *= max(1.0 - fraction, 0.0)
    returned_velocity = returned_momentum / returned_volume
    return a, q, returned_volume, returned_velocity


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
    bulk free surface.  Above that point the tunnel gas is connected to the
    atmospheric gas already occupying the dry part of the riser; advancing the
    fitted nose through that gas-only region would manufacture an annular film
    all the way to the outlet.  After breakthrough the reported core extent
    therefore follows the resolved bulk surface while gas mass and momentum
    continue to vent through the conservative gas-network equations.
    """

    if dt <= 0.0 or diameter <= 0.0 or riser_height <= 0.0:
        raise ValueError("positive Taylor-front geometry and timestep required")
    old_front = float(np.clip(front_height, 0.0, riser_height))
    surface = float(np.clip(free_surface_height, 0.0, riser_height))
    if surface <= 0.0:
        return 0.0, -old_front / dt, bool(already_vented)

    if already_vented:
        new_front = surface
        return new_front, (new_front - old_front) / dt, True

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


def _apply_finite_width_side_t_exchange(
    area,
    discharge,
    *,
    upward_flow: float,
    opening_weights,
    dt: float,
    cell_width: float,
    incoming_normal_velocity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange liquid through the real side-T footprint conservatively.

    Positive ``upward_flow`` removes liquid and its local axial momentum from
    the footprint.  A descending stream enters normal to the horizontal axis
    and impinges on the opposite wall.  Its resolved normal speed is redirected
    into equal-and-opposite axial parcel momenta on the two halves of the
    physical opening.  The weighted axial momentum is exactly zero, while the
    kinetic energy available to the horizontal equations comes only from the
    computed incoming stream.  No wave amplitude, frequency, or time history
    is prescribed.
    """

    a = np.asarray(area, dtype=float).copy()
    q = np.asarray(discharge, dtype=float).copy()
    weights = np.asarray(opening_weights, dtype=float)
    if a.shape != q.shape or a.shape != weights.shape or a.ndim != 1:
        raise ValueError("side-T exchange arrays must have equal one-dimensional shape")
    step = float(dt)
    width = float(cell_width)
    normal_velocity = float(incoming_normal_velocity)
    if (
        step <= 0.0
        or width <= 0.0
        or not math.isfinite(normal_velocity)
    ):
        raise ValueError("positive side-T timestep and cell width required")
    volume_change = -float(upward_flow) * step
    old_area = a.copy()
    a += volume_change * weights / width
    if np.any(a < -1.0e-12):
        raise FloatingPointError("finite-width side-T exchange emptied a donor cell")
    a = np.maximum(a, 0.0)
    removing = a < old_area
    wet = old_area > EPS
    scale = np.ones_like(a)
    scale[removing & wet] = a[removing & wet] / old_area[removing & wet]
    scale[removing & ~wet] = 0.0
    q *= scale
    if volume_change > 0.0 and abs(normal_velocity) > EPS:
        # Build a grid-independent split shape.  Subtracting its weighted mean
        # makes the total axial impulse vanish even when the opening is not
        # centred on a cell face; unit weighted RMS preserves the incoming
        # parcel kinetic-energy scale before finite-volume mixing.
        active = weights > 0.0
        coordinates = np.arange(weights.size, dtype=float)
        opening_centre = float(np.sum(weights * coordinates))
        turn_shape = np.zeros_like(weights)
        turn_shape[active] = np.sign(coordinates[active] - opening_centre)
        turn_shape[active] -= float(np.sum(weights * turn_shape))
        shape_norm = math.sqrt(float(np.sum(weights * turn_shape * turn_shape)))
        if shape_norm > EPS:
            turn_shape /= shape_norm
            added_area = np.maximum(a - old_area, 0.0)
            q += added_area * abs(normal_velocity) * turn_shape
    actual_change = float(np.sum(a - old_area) * width)
    if not math.isclose(
        actual_change,
        volume_change,
        rel_tol=1.0e-10,
        abs_tol=1.0e-16,
    ):
        raise FloatingPointError("finite-width side-T exchange lost liquid volume")
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
    jx = int(np.clip(round(case.x_riser / dx - 0.5), 1, Nt - 2))   # junction cell
    junction_face = int(np.clip(jx + 1, 1, Nt - 1))
    iv = int(np.clip(round(case.L_up / dx - 0.5), 1, Nt - 2))      # nearest butterfly-valve cell
    fv = int(np.clip(round(case.L_up / dx), 1, Nt - 1))             # butterfly-valve FACE
    (
        riser_film_thickness,
        riser_gas_core_fraction,
        riser_terminal_film_flow,
        riser_terminal_film_velocity,
    ) = _vw_laminar_film_closure(
        case.Dr,
        rho_l=RHO_L,
        rho_g=rho_atm,
        mu_l=MU_L,
        gravity=G,
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
    # Before the gas front reaches the side-T, the copied shock-fitting core
    # carries a closed polytropic inventory.  At first contact with the open
    # vertical branch it is rebased continuously onto one open, spatially
    # lumped ideal-gas reservoir.  From then on its mass can change only through
    # the conservative T-mouth Riemann flux; it is never split among horizontal
    # grid cells as independent material pockets.
    external_open_gas_inventory = None
    external_open_gas_parameters = None
    cfg = external_horizontal_solver.config
    if not math.isclose(
        float(cfg.length), case.L_tunnel, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("external horizontal length does not match the case")
    if cfg.vent_x is None or not math.isclose(
        float(cfg.vent_x), case.x_riser, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("external horizontal vent does not match the side-T")
    external_horizontal_state = (
        external_horizontal_solver.case_b_initial_state(
            initial_air_gauge_head=case.air_head,
            # The shock-fitting section head uses the pipe invert as datum,
            # whereas the experiment reports Y_fs from the pipe crown.
            initial_water_head=case.D + case.init_water_level,
        )
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

    geyser_strength = 0.0
    t = 0.0
    step = 0
    dbg_created = dict(t_floor=0.0, r_floor=0.0, r_repack=0.0, consol=0.0, crown=0.0)
    rec = dict(t=[], wtop=[], itop=[], jet_height=[], top_q=[],
               core_mass=[], pocket_head=[], up_head=[], pj_head=[], tr_head=[],
               base_q=[], base_head=[], junction_alpha=[], left_mean_alpha=[],
               right_mean_alpha=[], right_max_alpha=[], right_full_fraction=[],
               tun_gas_mass=[], tun_gas_vol=[], tot_liq=[], tot_liq_raw=[],
               escaped_gas_mass=[], total_resolved_gas_mass=[],
               escaped_liquid_volume=[], total_liquid_including_escape=[],
               frames_t=[], frames_alt=[], frames_alt_raw=[], frames_ult=[], frames_mgt=[], frames_alr=[], frames_ulr=[], frames_agr=[], frames_itop=[],
               frames_core_mass=[],
               xt=xt, zr=zr, jx=jx, iv=iv, fv=fv, dx=dx, dz=dz, Nt=Nt, Nr=Nr)
    itr = int(np.clip(round(case.x_transducer / dx - 0.5), 0, Nt - 1))   # transducer cell
    out_dt = float(output_interval)

    last_q_up = 0.0
    last_base_head = Yfs0
    last_top_q = 0.0
    jet_height_state = 0.0
    gas_escaped_mass = 0.0
    gas_atmospheric_exchange = 0.0
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
    side_t_east_material_front = float(case.x_riser)
    last_junction_node_pressure = P_ATM + RHO_L * G * Yfs0
    last_junction_west_flow = 0.0
    last_junction_vertical_flow = 0.0
    last_junction_gas_mouth_fraction = 0.0
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
        rec["junction_alpha"].append(float(alt_state[jx] / A))
        left_alpha = np.asarray(alt_state[:jx + 1] / A, dtype=float)
        right_alpha = np.asarray(alt_state[jx + 1:] / A, dtype=float)
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
            record_pocket,
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
        rec["frames_mgt"].append(np.asarray(mgt_state, dtype=float).copy())
        rec["frames_alr"].append(np.clip(alr_state / Ar, 0, 1).copy())
        rec["frames_ulr"].append(
            np.where(
                alr_state > 1.0e-9,
                Qlr / np.maximum(alr_state, 1.0e-2 * Ar),
                0.0,
            ).copy()
        )
        rec["frames_agr"].append(alpha_g_r.copy())
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
        rec.setdefault("junction_gas_mouth_fraction", []).append(
            float(last_junction_gas_mouth_fraction)
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
        0.0,
        Alt,
        Alr,
        Mgt,
        Mgr,
        Mgrs,
        np.full(Nr, rho_atm),
    )
    next_out = out_dt

    dt_prev = 0.0
    wall_start = time.perf_counter()
    while t < case.t_end - 1e-12:
        liquid_volume_before_step = float(
            np.sum(Alt) * dx + np.sum(Alr) * dz
        )
        junction_wave_active = bool(
            external_horizontal_state is not None
            and bool(external_horizontal_state.vented)
        )
        if junction_wave_active and external_open_gas_inventory is None:
            raise FloatingPointError(
                "vented shock-fit state has no open lumped gas inventory"
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
        Pt_external = np.where(
            stratified_supported_t,
            P_ATM - bulk_corr_t,
            Pt_surface - RHO_L * a2 * elastic_dev_t,
        )
        # In a resolved stratified cell the companion-model flux already
        # contains gas pressure through ``0.5*Lambda_d*A_l**2``.  Only the
        # acoustic bulk-viscosity correction remains as a regular pressure
        # source there.  Gas-free elastic cells retain their external datum
        # after subtracting the conservative water-hammer pressure.  This
        # branch-consistent split prevents counting gas pressure twice.
        # Hydrostatic pressure of the resolved vertical liquid column.
        head_r = np.maximum(Yfs - zr, 0.0)
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
        junction_face = int(np.clip(jx + 1, 1, Nt - 1))
        iw = junction_face - 1
        ie = junction_face
        p_w = float(Pt_crown[iw])
        p_e = float(Pt_crown[ie])
        p_v = float(Pr[0] + RHO_L * G * 0.5 * dz)
        last_junction_west_pressure = p_w
        last_junction_east_pressure = p_e
        last_junction_vertical_pressure = p_v
        u_w_out = -float(Qlt[iw] / max(Alt[iw], 1.0e-3 * A))
        u_e_out = float(Qlt[ie] / max(Alt[ie], 1.0e-3 * A))
        u_v_out = float(Qlr[0] / max(Alr[0], 1.0e-2 * Ar))
        alpha_g_j_pre = float(np.clip(1.0 - Alt[jx] / A, 0.0, 0.98))
        gas_void_j = max(A - Alt[jx], 1.0e-4 * A)
        gas_density_j = max(Mgt[jx] / max(gas_void_j * dx, EPS), 0.0)
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
        pocket_mass_j, pocket_volume_j = _connected_pocket_inventory(
            Alt,
            Mgt,
            junction_pocket_mask,
            index=jx,
            full_area=A,
            cell_width=dx,
        )
        if pocket_volume_j > EPS and pocket_mass_j > 0.0:
            P_connected_j = min(
                pocket_mass_j * R_GAS * T_GAS / pocket_volume_j,
                12.0 * P_ATM,
            )
        if external_open_gas_inventory is not None:
            # The post-contact horizontal gas phase is one acoustically
            # equilibrated reservoir.  Its pressure comes from its complete
            # mass and geometric volume, not from whichever mapped display
            # cell happens to contain the T centre.
            P_gas_j = external_open_gas_inventory.pressure_absolute
            P_connected_j = P_gas_j
        horizontal_gas_at_junction = bool(
            junction_wave_active
            and alpha_g_j_pre > case.tower_entry_alpha_min
            and junction_pocket_mask[jx]
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

        if external_open_gas_inventory is not None:
            P_connected_j = external_open_gas_inventory.pressure_absolute

        # Gas and liquid share the physical T-mouth area.  Before gas arrival
        # the complete bore belongs to the liquid characteristic.  Once a
        # resolved crown cavity reaches the T, only the complementary area is
        # available to liquid; the gas solver is additionally limited by the
        # actually opened void in the first riser cell.
        horizontal_gas_mouth = (
            junction_mouth_area(
                alpha_g_j_pre, coupled_gas_parameters
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
            # The remaining tower liquid is a connected water column.  Its
            # incoming pressure characteristic stays on the water-hammer
            # branch; the gas/free-surface wave speed belongs to the separately
            # resolved gas/interface equations and must not be used as liquid
            # impedance here.
            riser_wave_speed = case.a_wh
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
        loss = max(
            float(case.junction_loss_coeff)
            + float(case.glug_loss_coeff) * glug_activation,
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
                              or (1.0 - Alt[jx] / A) > 0.20)
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
        breakthrough = bool(junction_gassy and ksurf_pre >= 1
                            and (np.all(alpha_gr_pre[:ksurf_pre] > 0.20) or surface_gassy_pre))
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
        smax = max(
            float(np.max(np.abs(ult) + ct)),
            float(np.max(np.abs(ulr) + cr)),
            horizontal_gas_outer_speed,
            float(np.max(np.abs(ugr_now)) + gas_wave),
        )
        if junction_wave_active:
            horizontal_lambda_cfl = _decoupled_restoring_coefficient(
                Alt,
                Qlt,
                Mgt,
                Jgt,
                area_full=A,
                diameter=case.D,
                cell_width=dx,
            )
            horizontal_twofluid_celerity = np.sqrt(
                np.maximum(horizontal_lambda_cfl * np.maximum(Alt, 0.0), 0.0)
                + 1.0e-8
            )
            smax = max(
                smax,
                float(np.max(np.abs(ult) + horizontal_twofluid_celerity)),
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
        last_dt_outer = float(dt)
        last_dt_phase = (
            float(dt_phase) if np.isfinite(dt_phase) else 0.0
        )
        last_dt_junction = (
            float(dt_junction) if np.isfinite(dt_junction) else 0.0
        )

        external_horizontal_next = None
        if external_horizontal_active:
            # Advance the fitted interface over exactly the same physical time
            # increment as the network.  The shock solver performs its own CFL
            # subcycling.  Its conservative fields replace the provisional
            # tunnel update below until the interface reaches the side-T.
            external_horizontal_next = external_horizontal_solver.step(
                external_horizontal_state,
                dt,
                external_pressure_abs=(
                    None
                    if external_open_gas_inventory is None
                    else external_open_gas_inventory.pressure_absolute
                ),
            )

        # ================= TUNNEL update =================
        # One finite-volume liquid operator is used after hand-off.  Before
        # hand-off its provisional state is replaced by the conservative
        # shock-fitting solution over the same dt.
        if phase_horizontal_flux is None:
            F1, F2, Al_face, Ar_face = _decoupled_liquid_rusanov_flux(
                Alt,
                Qlt,
                Mgt,
                Jgt,
                area_full=A,
                diameter=case.D,
                wave_speed=case.a_wh,
                cell_width=dx,
            )
        else:
            F1, F2, Al_face, Ar_face = (
                component.copy() for component in phase_horizontal_flux
            )
        # CLOSED WALLS carry exactly zero volume flux.  The mirror ghost only
        # guarantees this for the central wet-branch flux (antisymmetric Q
        # cancels); the CONTACT branch is donor-cell upwind, and at a wall face
        # uf_t = 0.5*(-u0 + u0) = 0 selects the GHOST flux -Q0 -- i.e. the wall
        # face passed -Qlt[0] of volume whenever the end cell was gassy (the
        # capsule end ALWAYS is).  Every slosh cycle of the release transient
        # pumped water through the closed end: +0.48 L had appeared by t=4 s
        # (probe9), swelling the tower by 8 cm of level with the tunnel
        # inventory unchanged -- the "geysering" of the no-geyser case.
        F1[0] = 0.0
        F1[-1] = 0.0
        # Butterfly valve = a BLENDED WALL at its face during the opening stroke.
        # phi = theta^2 is the orifice transmissivity of the turning disc: at
        # phi<1 the face passes only phi of its open-valve flux.  Modelling the
        # stroke as cell FRICTION alone (the old closure) left the pressure
        # gradient free to slam the first full cell into the under-pressured
        # capsule at ~12 m/s^2 the instant t>0 -- a grid-sharp impact that rang
        # the whole slug on the pocket spring at +-0.4 m of head for the rest of
        # the run (the experiment's release is quasi-static: H* walks smoothly
        # from 0.50 to its 0.537 plateau with no overshoot, V&W2011 Fig.5).
        theta_v = min(max(t / max(case.valve_open_time, 1.0e-9), 0.02), 1.0)
        phi_v = theta_v * theta_v
        phi_flow = phi_v
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
        # The horizontal T control volume retains the ordinary west/east
        # finite-volume fluxes.  The vertical exchange is a conservative side
        # source applied below; therefore no horizontal face is frozen or
        # replaced when the gas pocket reaches the branch.
        last_junction_east_flux = (
            float(F1[junction_face]) if junction_wave_active else 0.0
        )
        Alt_new = Alt - dt / dx * (F1[1:] - F1[:-1])
        Qlt_new = Qlt - dt / dx * (F2[1:] - F2[:-1])
        # sources: pressure gradient (theta=0, no gravity) + friction
        Pt_momentum = Pt_external
        Pth = np.empty(Nt + 2); Pth[1:-1] = Pt_momentum
        Pth[0] = Pt_momentum[0]                   # closed upstream: zero-grad
        Pth[-1] = Pt_momentum[-1]                 # closed downstream: zero-grad
        dPdx = (Pth[2:] - Pth[:-2]) / (2.0 * dx)
        # While the disc still blocks the face, the pressure jump across it is
        # borne by the DISC (a wall force), not by the fluid: blend the two
        # adjacent cells' gradients toward their wall-reflected (one-sided)
        # values so the closed valve accelerates nobody.
        if phi_flow < 1.0 and 1 <= fv <= Nt - 2:
            dPdx[fv - 1] = (phi_flow * (Pt_momentum[fv] - Pt_momentum[fv - 2]) + (1.0 - phi_flow) * (Pt_momentum[fv - 1] - Pt_momentum[fv - 2])) / (2.0 * dx)
            dPdx[fv] = (phi_flow * (Pt_momentum[fv + 1] - Pt_momentum[fv - 1]) + (1.0 - phi_flow) * (Pt_momentum[fv + 1] - Pt_momentum[fv])) / (2.0 * dx)
        Dh_t = case.D
        un = np.where(Alt_new > 1e-9, Qlt_new / np.maximum(Alt_new, EPS), 0.0)
        # friction: laminar wall shear + turbulent Darcy term (f~0.025) + a mild
        # interface-churn increment in stratified cells (free-surface sloshing under
        # the cavity).  Churn must stay MILD: at 1.5 it froze the under-cavity
        # counter-current, which (a) blocked the pocket from pushing the column back
        # after the release overshoot (one-way valve: pocket stuck over-compressed at
        # ~0.45 m instead of relaxing to the ~0.35 m joint equilibrium) and (b)
        # halved the Benjamin nose speed (drain-back throttled).
        ag_t = np.clip(1.0 - Alt_new / A, 0.0, 1.0)
        churn_w = 4.0 * ag_t * (1.0 - ag_t)
        fric_t = (32.0 * MU_L / RHO_L / (Dh_t * Dh_t)
                  + (
                      0.025
                      + case.horizontal_churn_friction * churn_w
                  ) / (2.0 * Dh_t) * np.abs(un))
        # butterfly-valve local loss (disc remains in the bore when open, V&W Fig.2):
        # dP = 0.5*K*rho*u|u| across the valve cell -> friction-like sink over dx.
        # During the hand-opening stroke (~valve_open_time) the disc throttles the
        # release: K ~ K_open/theta^2 -> the column accelerates as a body over the
        # stroke instead of the contact cell alone taking a 0-ms burst.
        valve_drag = (case.valve_loss_coeff / max(phi_flow, 1.0e-4)) / (4.0 * dx)
        fric_t[fv - 1] += valve_drag * abs(un[fv - 1])
        fric_t[fv] += valve_drag * abs(un[fv])
        # SEMI-IMPLICIT friction: Q/(1+dt*fric) is unconditionally dissipative.
        # The explicit form -dt*fric*Q flips the sign of u whenever dt*fric > 2 --
        # which the throttled valve cell reaches for the whole opening stroke
        # (K/theta^2 ~ 5e3) -- so the "loss" term was a NEGATIVE damper there:
        # it injected ~11 J of kinetic energy into the slug within 0.1 s (probe12;
        # the physical release energy is ~0.01 J) and THAT was the hammer that rang
        # the pocket spring at +-0.4 L of head for the rest of every run.
        Qlt_new = (Qlt_new - dt * (Alt_new / RHO_L) * dPdx) / (1.0 + dt * fric_t)
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
        G2 = G1 * donor_velocity_r
        # The tower opens into air, not an exterior liquid reservoir.  A
        # negative reconstructed top flux would import water from the copied
        # ghost state and make the network liquid inventory grow.  Permit
        # ejection only; pressure/gas characteristics remain bidirectional.
        if G1[-1] < 0.0:
            G1[-1] = 0.0
            G2[-1] = 0.0
        # Liquid characteristic boundary at the shared T node.  Extrapolate
        # the first riser-cell pressure down by half a cell and use the incoming
        # water-hammer characteristic to determine the single signed face
        # velocity.  This applies the pressure jump once (through rho*a
        # impedance) instead of both as a ghost-cell force and an unconstrained
        # Rusanov flux.
        p_riser_at_node = float(Pr[0] + RHO_L * G * 0.5 * dz)
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
        # The V&W terminal-film relation is retained as a diagnostic scale,
        # not imposed as a second base boundary condition.  The resolved
        # annular film already receives gravity, wall friction, interphase
        # exchange, and the T-node pressure characteristic.  Overwriting G1[0]
        # here after that node solve double-counted drainage and emptied the
        # Case-A tower several seconds too early.  Its actual bottom flux now
        # remains the conservative characteristic flux computed above.

        junction_vertical_node_flow = float(G1[0])
        last_junction_vertical_flow = float(G1[0])
        Alr_new = Alr - dt / dz * (G1[1:] - G1[:-1])
        Qlr_new = Qlr - dt / dz * (G2[1:] - G2[:-1])
        # Side-T entry is a moving cut-cell problem.  A cell-centred two-fluid
        # topology has no vertical gas face before the first swept volume is
        # opened, while allowing every geometric void to join the acoustic gas
        # graph makes the material tracer jump to the top at sound speed.
        # Advance the contact by the standard confined-bubble drift relation
        # U_n = J_l + 0.345*sqrt(g*D_r), where J_l is the signed liquid
        # superficial velocity through the complete riser bore, and move the
        # swept liquid into the connected upper column conservatively.  Using
        # Q_l/A_l here would incorrectly attach the gas nose to the fast falling
        # annular film; the drift relation is explicitly counter-current.
        # Pressure and momentum remain finite-volume; the cut-cell update only
        # moves the liquid volume swept by the material contact into available
        # capacity in the connected upper column.  The operation conserves
        # liquid volume and axial momentum exactly.  A fitted front may open new
        # vertical gas volume only while a mass-supported horizontal gas path is
        # still connected to the T mouth.  Letting vertical gas alone sustain a
        # one-sided nose sweep creates void volume without a matching horizontal
        # gas-mass supply.  That defect caused the 7.10--7.15 s pressure collapse
        # and the near-dry pocket around x=3.5 m.  No Case-A time, transfer
        # fraction, or target result is imposed by this connectivity condition.
        if (
            horizontal_gas_at_junction
            and (incipient_vertical_gas_receiving or riser_material_front > 0.0)
            and geometric_gas_mouth > 0.0
        ):
            front_cell = min(
                int(riser_material_front / max(dz, EPS)),
                Nr - 1,
            )
            if front_cell == 0:
                front_liquid_superficial_velocity = (
                    junction_vertical_node_flow / max(Ar, EPS)
                )
            else:
                front_liquid_superficial_velocity = float(
                    Qlr_new[front_cell] / max(Ar, EPS)
                )
            material_front_velocity = max(
                front_liquid_superficial_velocity
                + 0.345 * math.sqrt(G * case.Dr),
                0.0,
            )
            old_material_front = float(riser_material_front)
            proposed_material_front = min(
                riser_material_front + material_front_velocity * dt,
                max(float(wtop), 0.0),
                case.riser_height,
            )
            (
                Alr_new,
                Qlr_new,
                material_swept_volume,
                material_return_velocity,
            ) = (
                _sweep_vertical_material_slice_to_junction(
                    Alr_new,
                    Qlr_new,
                    old_front_height=old_material_front,
                    new_front_height=proposed_material_front,
                    gas_core_area_fraction=(
                        geometric_gas_mouth / max(Ar, EPS)
                    ),
                    full_area=Ar,
                    dz=dz,
                )
            )
            if material_swept_volume > 0.0:
                riser_material_front = proposed_material_front
                if not vertical_gas_at_junction:
                    riser_entry_cut_front = riser_material_front
                Alt_new, Qlt_new = _apply_finite_width_side_t_exchange(
                    Alt_new,
                    Qlt_new,
                    upward_flow=-material_swept_volume / dt,
                    opening_weights=side_t_weights,
                    dt=dt,
                    cell_width=dx,
                    incoming_normal_velocity=abs(material_return_velocity),
                )
                junction_return_requested_volume += material_swept_volume
                junction_return_deposited_volume += material_swept_volume
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
        K_base = (
            case.junction_loss_coeff
            + case.glug_loss_coeff * glug_activation
        )
        liquid_base_pressure = P_base_liq
        Prh[0] = (liquid_base_pressure + RHO_L * G * (0.5 * dz)
                  - 0.5 * RHO_L * K_base * u_base * abs(u_base))
        Prh[-1] = P_ATM
        dPdz = (Prh[2:] - Prh[:-2]) / (2.0 * dz)
        # Gas momentum uses its own resolved thermodynamic pressure.  The bottom
        # ghost is the local horizontal gas pressure and the top is atmospheric;
        # no extra driving head is added.
        gas_void_r = np.maximum(Ar - Alr, 1.0e-4 * Ar)
        P_gas_r = (
            np.maximum(Mgr, 0.0) * R_GAS * T_GAS
            / np.maximum(gas_void_r * dz, EPS)
        )
        Prh_gas = np.empty(Nr + 2)
        Prh_gas[1:-1] = P_gas_r
        Prh_gas[0] = P_gas_j if gas_at_junction else P_ATM
        Prh_gas[-1] = P_ATM
        dPdz_gas = (Prh_gas[2:] - Prh_gas[:-2]) / (2.0 * dz)
        Qlr_new += dt * (
            -(Alr_new / RHO_L) * dPdz - Alr_new * G
        )

        # ---------- conservative finite-volume T exchange ----------
        # Accumulate the vertical exchange here and apply it to the physical
        # horizontal junction control volume below.  The axial west/east faces
        # remain ordinary finite-volume faces, so the horizontal equations—not
        # a prescribed branch split—determine how the returned liquid launches
        # pressure and free-surface waves away from the tee.
        q_up = float(G1[0])                       # [m^3/s] up the riser base face
        # Keep the vertical branch on the resolved two-fluid finite-volume
        # equations.  The former Davies--Taylor material-front projection
        # imposed an axisymmetric 93%-area gas core on the asymmetric tongue
        # entering this side T and converted front advance into a prescribed
        # return discharge.  The tracer extent is tracked only for diagnostics.
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
        if junction_volume_change > 0.0:
            junction_return_requested_volume += junction_volume_change
            junction_return_deposited_volume += junction_volume_change
        if junction_wave_active and external_horizontal_next is None:
            # The measured tower mouth is a finite side opening, not a
            # zero-volume node at one horizontal face.  Apply the shared riser
            # flux over the exact geometric footprint while the ordinary
            # horizontal west/east fluxes keep evolving without interruption.
            Alt_new, Qlt_new = _apply_finite_width_side_t_exchange(
                Alt_new,
                Qlt_new,
                upward_flow=q_up,
                opening_weights=side_t_weights,
                dt=dt,
                cell_width=dx,
                incoming_normal_velocity=(
                    float(G2[0] / G1[0]) if G1[0] < -EPS else 0.0
                ),
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
                and external_open_gas_inventory is None
            ):
                # Rebase at the instant the fitted front opens the T.  The
                # temperature is inferred from the closed state so pressure,
                # mass, and volume are all continuous across the topology
                # change.  It is subsequently held by the isothermal companion
                # gas model while mass changes only through resolved boundaries.
                gas_mass_open = float(external_horizontal_next.gas.mass)
                gas_volume_open = float(external_horizontal_next.gas.volume)
                gas_temperature_open = (
                    float(external_horizontal_next.air_pressure_abs)
                    * gas_volume_open
                    / max(gas_mass_open * R_GAS, EPS)
                )
                external_open_gas_inventory = OpenIsothermalGasInventory(
                    mass=gas_mass_open,
                    volume=gas_volume_open,
                    gas_constant=R_GAS,
                    temperature=gas_temperature_open,
                )
                external_open_gas_parameters = replace(
                    coupled_gas_parameters,
                    gas_temperature=gas_temperature_open,
                )
                external_horizontal_handoff_time = float(t + dt)

            if external_open_gas_inventory is not None:
                external_open_gas_inventory = (
                    external_open_gas_inventory.with_state(
                        volume=float(external_horizontal_next.gas.volume)
                    )
                )
                external_horizontal_next = replace(
                    external_horizontal_next,
                    gas=replace(
                        external_horizontal_next.gas,
                        mass=external_open_gas_inventory.mass,
                        volume=external_open_gas_inventory.volume,
                    ),
                    air_pressure_abs=(
                        external_open_gas_inventory.pressure_absolute
                    ),
                )

            Alt_new, Qlt_new, Mgt_new, Jgt_new = (
                _map_external_horizontal_state(
                    external_horizontal_solver,
                    external_horizontal_next,
                    x_target=xt,
                    full_area=A,
                    dx=dx,
                )
            )

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
        Jgrs_new = Jgrs.copy()
        if junction_wave_active:
            lumped_horizontal_gas = bool(
                external_horizontal_next is not None
                and external_open_gas_inventory is not None
                and external_open_gas_parameters is not None
            )
            try:
                if lumped_horizontal_gas:
                    horizontal_t_void_area = max(
                        A - float(np.clip(Alt_new[jx], 0.0, A)),
                        external_open_gas_parameters.void_floor_fraction * A,
                    )
                    gas_advance = advance_lumped_pocket_vertical_network(
                        external_open_gas_inventory,
                        horizontal_t_void_area,
                        Mgr,
                        Jgrs,
                        Mgrs,
                        Alr_new,
                        Qlr_new,
                        dz=dz,
                        dt=dt,
                        params=external_open_gas_parameters,
                        vertical_pocket_front_height=(
                            riser_material_front
                            if (
                                riser_material_front > 0.0
                                and riser_material_front
                                < max(float(wtop), 0.0) - 1.0e-12
                            )
                            else None
                        ),
                        vertical_liquid_surface_height=float(wtop),
                        vertical_branch_confined=bool(
                            riser_material_front > 0.0
                            and riser_material_front
                            < max(float(wtop), 0.0) - 1.0e-12
                        ),
                    )
                else:
                    gas_advance = advance_coupled_gas_network(
                        gas_mass_input,
                        gas_momentum_input,
                        Mgr,
                        Jgrs,
                        Mgrs,
                        Alt_new,
                        Qlt_new,
                        Alr_new,
                        Qlr_new,
                        dx=dx,
                        dz=dz,
                        dt=dt,
                        junction_index=jx,
                        params=coupled_gas_parameters,
                        vertical_pocket_front_height=(
                            riser_material_front
                            if (
                                riser_material_front > 0.0
                                and riser_material_front
                                < max(float(wtop), 0.0) - 1.0e-12
                            )
                            else None
                        ),
                        vertical_liquid_surface_height=float(wtop),
                        vertical_branch_confined=bool(
                            riser_material_front > 0.0
                            and riser_material_front
                            < max(float(wtop), 0.0) - 1.0e-12
                        ),
                        vertical_branch_receiving_hint=(
                            vertical_gas_receiving
                        ),
                        horizontal_downstream_front_position=(
                            side_t_east_material_front
                        ),
                    )
            except FloatingPointError as exc:
                horizontal_velocity = np.divide(
                    Jgt,
                    Mgt,
                    out=np.zeros_like(Jgt),
                    where=Mgt > 1.0e-14,
                )
                vertical_velocity = np.divide(
                    Jgrs,
                    Mgr,
                    out=np.zeros_like(Jgrs),
                    where=Mgr > 1.0e-14,
                )
                raise FloatingPointError(
                    "coupled gas failure "
                    f"at t={t:.12g}, dt={dt:.12g}, "
                    f"alpha_g_j={1.0-Alt_new[jx]/A:.8g}, "
                    f"Mgt_j={Mgt[jx]:.8g}, Mgr_0={Mgr[0]:.8g}, "
                    f"Mgrs_0={Mgrs[0]:.8g}, "
                    f"max_uh={np.max(np.abs(horizontal_velocity)):.8g}, "
                    f"max_uv={np.max(np.abs(vertical_velocity)):.8g}, "
                    f"Alr_0/Ar={Alr_new[0]/Ar:.8g}"
                ) from exc
            if lumped_horizontal_gas:
                external_open_gas_inventory = (
                    gas_advance.horizontal_inventory
                )
                external_horizontal_next = replace(
                    external_horizontal_next,
                    gas=replace(
                        external_horizontal_next.gas,
                        mass=external_open_gas_inventory.mass,
                        volume=external_open_gas_inventory.volume,
                    ),
                    air_pressure_abs=(
                        external_open_gas_inventory.pressure_absolute
                    ),
                )
                Mgt_new = _map_external_horizontal_state(
                    external_horizontal_solver,
                    external_horizontal_next,
                    x_target=xt,
                    full_area=A,
                    dx=dx,
                )[2]
                # The lumped reservoir has no resolved axial gas momentum.
                # Its T-normal trace is handled inside the Riemann solve.
                Jgt_new = np.zeros_like(Mgt_new)
                side_t_east_material_front = max(
                    float(side_t_east_material_front),
                    float(external_horizontal_next.interface_x),
                )
            else:
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
            riser_gas_front_velocity = (
                riser_gas_front - old_riser_gas_front
            ) / max(dt, EPS)
            if (
                not lumped_horizontal_gas
                and gas_advance.downstream_front_position is not None
            ):
                side_t_east_material_front = float(
                    gas_advance.downstream_front_position
                )
            if not lumped_horizontal_gas:
                Qlt_new += gas_advance.horizontal_liquid_momentum_increment
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

        # Complete the vertical momentum split after gas--liquid drag.  Bulk
        # liquid uses ordinary pipe friction; only a resolved thin annular film
        # receives the laminar--turbulent film stress.
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
        Qlt_new = _implicit_smagorinsky_momentum_diffusion(
            Alt_new,
            Qlt_new,
            full_area=A,
            diameter=case.D,
            spacing=dx,
            dt=dt,
            coefficient=case.nu,
        )
        Qlr_new = _implicit_smagorinsky_momentum_diffusion(
            Alr_new,
            Qlr_new,
            full_area=Ar,
            diameter=case.Dr,
            spacing=dz,
            dt=dt,
            coefficient=case.nu_riser,
        )

        # Repack only a genuinely gas-free riser.  The former condition used
        # the instantaneous horizontal T-mouth state: as soon as a detached
        # Taylor bubble left the mouth, it collapsed the still-two-phase riser
        # into a bottom-packed liquid column and froze all later dynamics.
        vertical_tracer_present_new = bool(
            float(np.sum(Mgrs_new)) > vertical_tracer_presence_mass
        )
        if (
            not gas_at_junction
            and not vertical_tracer_present_new
            and not riser_breakthrough
        ):
            Alr_new, Qlr_new = _project_single_liquid_column(
                Alr_new, Qlr_new, Ar, dz
            )
            riser_gas_front = 0.0
            riser_gas_front_velocity = 0.0
            riser_breakthrough = False

        # Open cells are selected only by the positivity-scale liquid cutoff.
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
        Alr_new = np.maximum(Alr_new, 0.0)

        if external_horizontal_next is not None:
            # The conservative T-face correction and open-gas update were
            # applied before the gas solve.  Remap only for the network record;
            # do not apply the branch flux a second time and do not hand the
            # horizontal state to the distributed gas-cell solver.
            Alt_new, Qlt_new, Mgt_new, Jgt_new = (
                _map_external_horizontal_state(
                    external_horizontal_solver,
                    external_horizontal_next,
                    x_target=xt,
                    full_area=A,
                    dx=dx,
                )
            )
            if external_open_gas_inventory is not None:
                Jgt_new = np.zeros_like(Mgt_new)
            external_horizontal_state = external_horizontal_next
            external_horizontal_active = True
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
            rec.setdefault("dbg_u_jx", []).append(float(u_dbg[jx]))
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
                "junction_alpha": float(Alt[jx] / A),
            }
            break
        if step >= case.max_steps:
            print("  [MAX_STEPS]", flush=True); break

    rec["geyser_strength"] = geyser_strength
    rec["riser_film_closure"] = {
        "thickness": float(riser_film_thickness),
        "gas_core_area_fraction": float(riser_gas_core_fraction),
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
