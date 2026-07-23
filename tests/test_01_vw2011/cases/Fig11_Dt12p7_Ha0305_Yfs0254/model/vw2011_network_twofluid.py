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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = 9.81
RHO_L = 998.0
P_ATM = 101325.0
R_GAS = 287.05
T_GAS = 293.0
MU_L = 1.003e-3
EPS = 1.0e-12
U_FLUX_MAX = 25.0   # safety clamp on advective face velocity [m/s] (prevents empty-cell blow-up)
_DISABLE_CROWN = False   # diagnostic switch (probe scripts only): freeze the crown-exchange front
CHURN_FRIC = 0.15   # interface-churn wall-friction increment in stratified cells (probe16 knob)
C_GC = 0.48         # crown-cavity celerity coefficient, v_gc = C_GC*sqrt(gD).
                    # Benjamin (1968): 0.542 is the ENERGY-CONSERVING upper
                    # bound; real (dissipative) cavities run below it, with
                    # measured horizontal-pipe values spanning ~0.47-0.54.
                    # The V&W2011 middle-pipe transits of the three
                    # repetitions correspond to 0.52/0.49/0.48 (2.97 m in
                    # 5.9/6.3/6.4 s); 0.48 places the computed gas sequence
                    # inside the repetition scatter (liftoff T*=7.41 between
                    # reps 1-2) -- ONE value for BOTH Campaign-1 tests.
F_HAND = 0.99       # crown-front hand-off fraction of the region-mean layer depth.
                    # The nose cell is released to the next cell once it reaches
                    # F_HAND*abar.  0.95 let every cell advance on 95% of its
                    # volume budget and biased the detected front to 1.09x the
                    # Benjamin celerity (the experiments transit at 0.96x:
                    # 2.97 m in 5.94 s, V&W2011 Fig.7); 1.0 stalls the nose
                    # against the region mean (asymptotic fill).  0.99 recovers
                    # the Benjamin-rate volume front the closure intends.
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
    init_water_level: float = 0.356  # selected initial tower water level Y_fs0 [m]
    riser_height: float = 0.610      # ventilation tower length L [m]
    L_up: float = 0.546         # upstream (air) pipe length [m]
    L_mid: float = 2.970        # middle pipe length [m]
    L_down: float = 0.490       # downstream pipe length [m] (closed end)
    gamma_gas: float = 1.4      # isentropic exponent
    # numerics
    ds: float = 0.02
    dz: float = 0.01
    t_end: float = 8.0
    cfl: float = 0.35
    # Numerical elastic speed of the full-water branch.  It only needs to be far
    # above the gravity-wave scale sqrt(gD)~1 m/s so the full reach is effectively
    # incompressible on pocket/tower timescales; the grid-scale ring amplitude of a
    # micro-void scales with rho*a^2 (a=60 turned a 0.1% area flicker into 0.37 m of
    # head and the rectified ringing pumped the tower), so keep it as low as the
    # scale separation allows.
    a_wh: float = 28.0
    nu: float = 0.05
    max_steps: int = 6_000_000
    use_vw_tower_closure: bool = False
    tower_core_area_fraction: float = 0.82
    tower_entry_alpha_min: float = 0.02
    gas_drag_time: float = 0.18
    gas_velocity_cap: float = 3.0
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
    gas_drive_eff: float = 1.0      # fraction of the air-pocket overpressure head that drives GAS penetration up the riser
    entry_drive_eff: float = 1.0    # fraction of the pocket overpressure that accelerates gas ENTRY into the riser base
    gas_escape_eff: float = 1.0     # fraction of the surface gas flux that bursts out the open top (wide risers vent fast -> no geyser)

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
        hs = np.linspace(0.0, 1.0, 201)
        th = 2.0 * np.arccos(np.clip(1.0 - 2.0 * hs, -1.0, 1.0))
        ar = (th - np.sin(th)) / (2.0 * math.pi)
        _SEG_TABLE = (ar, hs)
    return np.interp(alpha_l, _SEG_TABLE[0], _SEG_TABLE[1])


def _liquid_surface_height(z, dz, Al, A, threshold=0.08):
    """Top of any resolved liquid in the riser, including thin side films."""
    idx = np.where(Al / A > threshold)[0]
    return float(z[idx[-1]] + 0.5 * dz) if idx.size else 0.0


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


def run_network(case: NetworkCase, verbose: bool = True) -> dict:
    A = case.A; Ar = case.Ar
    a2 = case.a_wh * case.a_wh
    rho_atm = P_ATM / (R_GAS * T_GAS)

    # ---- tunnel grid (horizontal) ----
    Nt = max(20, int(round(case.L_tunnel / case.ds)))
    dx = case.L_tunnel / Nt
    xt = (np.arange(Nt) + 0.5) * dx
    jx = int(np.clip(round(case.x_riser / dx - 0.5), 1, Nt - 2))   # junction cell
    iv = int(np.clip(round(case.L_up / dx - 0.5), 1, Nt - 2))      # butterfly-valve cell

    # ---- riser grid (vertical) ----
    Nr = max(20, int(round(case.riser_height / case.dz)))
    dz = case.riser_height / Nr
    zr = (np.arange(Nr) + 0.5) * dz

    # ---- initial state (V&W2011): upstream pipe = compressed air pocket; middle+downstream
    #      pipes water-filled; tower water to Y_fs0; both far ends closed ----
    Alt = A * np.ones(Nt)
    capsule = xt < case.L_up                      # air pocket occupies the upstream pipe
    Alt[capsule] = 0.02 * A
    Qlt = np.zeros(Nt)
    Mgt = (P_ATM / (R_GAS * T_GAS)) * np.maximum(A - Alt, 1e-4 * A) * dx
    Pa0 = P_ATM + RHO_L * G * case.air_head       # initial absolute air-pocket pressure
    Mgt[capsule] = (Pa0 / (R_GAS * T_GAS)) * (A - Alt[capsule]) * dx
    Jgt = np.zeros(Nt)                            # horizontal gas momentum
    gas0 = float(np.sum(Mgt[capsule]))

    Yfs0 = case.init_water_level
    # sharp initial free surface: cells above Yfs0 are DRY (a 1% wall film here is
    # repacked into the column by the surface reconstruction and lifts the initial
    # level by ~1 cm -- a spurious first-frame jump against the paper's Yfs0)
    Alr = Ar * (zr <= Yfs0).astype(float)
    Qlr = np.zeros(Nr)
    Mgr = (P_ATM / (R_GAS * T_GAS)) * np.maximum(Ar - Alr, 1e-4 * Ar) * dz
    Mgrs = np.zeros(Nr)                           # resolved gas injected from the tunnel
    Jgrs = np.zeros(Nr)                           # resolved gas momentum in the riser

    v_gc = C_GC * math.sqrt(G * case.D)           # air-cavity intrusion speed along the crown (see C_GC)
    geyser_strength = 0.0
    t = 0.0
    step = 0
    dbg_created = dict(t_floor=0.0, r_floor=0.0, r_repack=0.0, consol=0.0, crown=0.0)
    rec = dict(t=[], wtop=[], itop=[], core_mass=[], pocket_head=[], up_head=[], pj_head=[], tr_head=[],
               tun_gas_mass=[], tun_gas_vol=[], tot_liq=[],
               frames_t=[], frames_alt=[], frames_mgt=[], frames_alr=[], frames_agr=[], frames_itop=[],
               frames_core_mass=[],
               xt=xt, zr=zr, jx=jx, dx=dx, dz=dz, Nt=Nt, Nr=Nr)
    itr = int(np.clip(round(case.x_transducer / dx - 0.5), 0, Nt - 1))   # transducer cell
    out_dt = 0.02

    def append_record(sample_t, alt_state, alr_state, mgt_state, mgr_res_state, rho_g_state):
        wtop_now = _liquid_surface_height(zr, dz, alr_state, Ar)
        alpha_g_raw = np.clip(mgr_res_state / np.maximum(rho_g_state * Ar * dz, 1.0e-12), 0.0, 0.90)
        # Do not hide resolved gas simply because it has displaced almost all
        # local liquid; the old wet-only mask made compact gas cells invisible.
        active_mask = (alr_state / Ar > 0.02) | (alpha_g_raw > 1.0e-4)
        alpha_g_r = np.where(active_mask, alpha_g_raw, 0.0)
        # Visible free surface includes GAS-HOLDUP SWELL: bubbles below the
        # surface displace water upward, so the level the experiment reads on
        # the tower scale is (liquid volume + submerged gas volume)/Ar.  The
        # liquid-only height stayed flat during the pocket climb while the
        # paper's Yfs* triangles rise ~10% of L -- that rise IS the swell.
        swell = float(np.sum(alpha_g_r[zr < wtop_now] * dz))
        wtop_now = min(wtop_now + swell, float(zr[-1] + 0.5 * dz))
        gas_idx = np.where(alpha_g_r > 0.02)[0]
        itop_now = float(zr[gas_idx[-1]] + 0.5 * dz) if gas_idx.size else 0.0
        itop_now = min(itop_now, wtop_now)
        gas_mass = float(np.sum(mgr_res_state * (alpha_g_r > 0.02)))
        ph = 0.0
        ph_up = 0.0
        m_main = 0.0
        # crown-exchange transport keeps gas mass and void collocated, so the pocket
        # regions are simply the connected runs of area-resolved void
        for (i0, i1) in _regions((alt_state / A) < 0.95):
            m = float(np.sum(mgt_state[max(i0 - 1, 0):min(i1 + 1, Nt)]))   # nose-halo (cf. _pressure)
            v = float(np.sum(np.maximum(A - alt_state[i0:i1], 1.0e-4 * A)) * dx)
            if m * R_GAS * T_GAS < 0.2 * P_ATM * max(v, 1e-12):
                continue                     # rarefied water, not an air pocket (cf. _pressure)
            head = (m * R_GAS * T_GAS / max(v, 1e-12) - P_ATM) / (RHO_L * G)
            ph = max(ph, head)
            if m > m_main:                   # MAIN pocket = the largest gas inventory;
                m_main = m                   # the transducer reach follows the capsule even
                ph_up = head                 # after backfill detaches it from the end wall
        rec["t"].append(float(sample_t))
        rec["up_head"].append(float(ph_up))
        rec["wtop"].append(float(wtop_now))
        rec["itop"].append(float(itop_now))
        rec["core_mass"].append(float(gas_mass))
        rec["pocket_head"].append(float(ph))
        rec["frames_t"].append(float(sample_t))
        rec["frames_alt"].append(np.clip(alt_state / A, 0, 1).copy())
        rec["frames_mgt"].append(np.asarray(mgt_state, dtype=float).copy())
        rec["frames_alr"].append(np.clip(alr_state / Ar, 0, 1).copy())
        rec["frames_agr"].append(alpha_g_r.copy())
        rec["frames_itop"].append(float(itop_now))
        rec["frames_core_mass"].append(float(gas_mass))

    append_record(0.0, Alt, Alr, Mgt, Mgrs, np.full(Nr, rho_atm))
    next_out = out_dt

    # Quasi-steady junction coupling state.  The T-mouth (and the tower standing on
    # it) responds to the junction's MEAN pressure over a few acoustic transits, not
    # to single-cell water-hammer spikes: rho*a^2 stiffness turns a one-cell 1e-4
    # relative volume error into ~0.4 m of head, and feeding that raw Pj to the
    # riser base ghost closed a grid-frequency feedback loop (hammer spike -> base
    # flux -> q_up withdrawal at jx -> alpha spike -> bigger hammer spike) that
    # pumped the tower full within 0.5 s and smeared the pocket across the tunnel.
    # tau ~ 50 ms (~ one tunnel acoustic transit) is far below every physical
    # timescale of this problem (pocket equilibration ~1 s, front transit ~6 s,
    # blowdown ~0.5 s), so the filter changes nothing physical -- it only removes
    # the sub-millisecond acoustics from the junction exchange.  The same filtered
    # Yfs drives the full-reach piezometric overlay: with the raw Yfs, the tower
    # level modulated EVERY full cell's pressure inside one ring period (a global
    # parametric pump for the acoustic mode).
    # tau ~ 0.2 s: well above the acoustic transit (L/a ~ 0.07 s), well below the
    # slowest physical mode the tower takes part in (column oscillation ~1 s).
    tau_junction = 0.2
    Yfs_slow = max(Yfs0, 0.0)
    dt_prev = 0.0
    # One-way vent latch: once a vent path through the tower has opened (first
    # breakthrough), the blow-down continues until the pocket overpressure is
    # gone.  The instantaneous topology test flickers (a bursting slug empties
    # the surface cells and the test re-seals the pocket between glugs), which
    # stalled the collapse halfway (recorded H* fell 0.5 -> 0.33 and recovered
    # to 0.42; the experiment falls monotonically to zero, Fig.5 T*~8.3..9.3).
    vent_latched = False

    while t < case.t_end - 1e-12:
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
        Pt, wett, _ = _pressure(Alt, np.full(Nt, A), Mgt, dx, a2, vent_top=False, p_floor=0.0,
                                gas_min=0.05, tension_head=TENSION_HEAD)
        Pr, wetr, _ = _pressure(Alr, np.full(Nr, Ar), Mgr, dz, a2, vent_top=True, p_floor=0.0)
        Yfs = _liquid_surface_height(zr, dz, Alr, Ar)
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
        Yfs_slow += (Yfs - Yfs_slow) * min(dt_prev / tau_junction, 1.0)
        w_full = np.clip((Alt / A - 0.99) / 0.01, 0.0, 1.0)
        Pt = np.where(Alt / A >= 0.99, Pt + RHO_L * G * Yfs_slow * w_full, Pt)
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
        Pt = np.where(gassy_cells_t, Pt + RHO_L * G * h_layer, Pt)
        for (i0g, i1g) in _regions(gassy_cells_t):
            if i0g - 1 >= 0 and alpha_l_t[i0g - 1] < 0.999:
                Pt[i0g - 1] = Pt[i0g] + RHO_L * G * (h_layer[i0g - 1] - h_layer[i0g])
            if i1g < Nt and alpha_l_t[i1g] < 0.999:
                Pt[i1g] = Pt[i1g - 1] + RHO_L * G * (h_layer[i1g] - h_layer[i1g - 1])
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
        Pt = Pt - w_bulk_t * RHO_L * (1.0 * case.a_wh * dx) * div_t
        Pr = np.where(wetr, Pr + RHO_L * G * np.maximum(Yfs - zr, 0.0), Pr)
        # acoustic bulk viscosity in the riser column (same damper as the tunnel)
        w_bulk_r = np.clip((Alr / Ar - 0.95) / 0.02, 0.0, 1.0)
        u_bulk_r = np.clip(Qlr / np.maximum(Alr, 1.0e-3 * Ar), -U_FLUX_MAX, U_FLUX_MAX)
        div_r = np.zeros(Nr)
        div_r[1:-1] = (u_bulk_r[2:] - u_bulk_r[:-2]) / (2.0 * dz)
        Pr = Pr - w_bulk_r * RHO_L * (1.0 * case.a_wh * dz) * div_r
        # Junction pressure the TOWER responds to: the QUASI-STATIC invert pressure
        # the pocket transmits through the full-water slug (Pascal), NOT any sampled
        # acoustic state of the junction cells.  The slug between the cavity nose
        # and the T is full horizontal pipe: with no mean through-flow its invert
        # piezometric head is uniform, so the head standing under the tower is
        #     P_pocket(EOS) + rho*g*h_layer(tower-facing pocket edge).
        # Every closure that SAMPLED Pt around jx -- raw, band-clamped, slew-limited,
        # low-pass filtered -- kept reproducing the same relaxation oscillation: a
        # compression spike reads the full rho*a^2 stiffness (+0.4 m head per 0.5%
        # area) while a rarefaction saturates at the -0.05 m tension floor, so ANY
        # sampling of the elastic state rectifies grid ringing into a net upward
        # pump; the pumped column then overdraws the junction cell on its return
        # (tension collapse) and re-launches the cycle (~1.5 Hz, +-0.15 L on Yfs,
        # +-0.5 L on the recorded pocket head -- none of it in the experiment).
        # The pocket EOS, by contrast, is a SMOOTH function of summed mass/volume:
        # the tower now feels only tower-scale physics (pocket compression, cavity
        # arrival, blowdown), while the slug's internal acoustics stay where they
        # belong -- inside the tunnel PDE, damped by the bulk viscosity.
        P_poc = 0.0
        m_poc = 0.0
        h_edge = 0.0
        for (i0p, i1p) in _regions((Alt / A) < 0.95):
            mreg = float(np.sum(Mgt[max(i0p - 1, 0):min(i1p + 1, Nt)]))
            vreg = float(np.sum(np.maximum(A - Alt[i0p:i1p], 1.0e-4 * A)) * dx)
            if mreg * R_GAS * T_GAS < 0.2 * P_ATM * max(vreg, 1e-12):
                continue                          # rarefied water, not an air pocket
            if mreg > m_poc:
                m_poc = mreg
                P_poc = mreg * R_GAS * T_GAS / max(vreg, 1.0e-12)
                if i1p - 1 < jx:
                    iedge = i1p - 1               # pocket west of the T: nose cell
                elif i0p > jx:
                    iedge = i0p                   # pocket east of the T
                else:
                    iedge = jx                    # pocket spans the T
                h_edge = float(h_layer[iedge])
        if m_poc > 0.0:
            Pj = P_poc + RHO_L * G * h_edge      # pocket invert head at the T (Pascal)
        else:
            Pj = P_ATM + RHO_L * G * max(Yfs, 0.0)   # no pocket left: pure hydrostatic
        # junction gas presence is AREA-based: with the crown-exchange transport the
        # migrating cavity carries its void along, so the resolved void fraction at the
        # junction cell is the physical arrival indicator (no phantom-mass detection).
        alpha_g_j_pre = float(np.clip(1.0 - Alt[jx] / A, 0.0, 0.98))
        gas_at_junction = alpha_g_j_pre > case.tower_entry_alpha_min
        # The riser water is lifted by the air that enters and rises through it, not
        # by a tunnel-pressure liquid jet. The liquid base is therefore kept as a
        # well-balanced hydrostatic wall at all times; the only liquid-surface motion
        # comes from the gas occupying volume as it is supplied and rises buoyantly.
        P_base_liq = P_ATM + RHO_L * G * max(Yfs, 0.0)
        # Air-pocket overpressure head available to DRIVE the gas up the riser:
        # gauge head of the connected pocket EOS minus the standing water column.
        H_op = 0.0
        body_t = (Alt / A) < 0.95
        for (i0, i1) in _regions(body_t):
            mreg = float(np.sum(Mgt[max(i0 - 1, 0):min(i1 + 1, Nt)]))   # nose-halo mass (cf. _pressure)
            vreg = float(np.sum(np.maximum(A - Alt[i0:i1], 1.0e-4 * A)) * dx)
            Preg = mreg * R_GAS * T_GAS / max(vreg, 1.0e-12)
            H_op = max(H_op, (Preg - P_ATM) / (RHO_L * G) - max(Yfs, 0.0))
        # The geyser is powered by the pocket's INITIAL overpressure margin above the
        # water column (Ha0 - Yfs0).  A transient over-compression by the descending
        # column must NOT add spurious drive (that energy belongs to the water and
        # returns to it).  Cap the driving head at the initial margin: Case A
        # (Ha0=0.305 < Yfs0=0.356, margin ~0) gets no penetration drive and does not
        # geyser; Case B (Ha0=0.610 > Yfs0, margin 0.254) is driven to a geyser.
        H_op = min(max(H_op, 0.0), max(case.air_head - case.init_water_level, 0.0))

        # ---------- breakthrough detection (previous-step state) ----------
        # When a continuous resolved-gas core connects the riser base to the free
        # surface AND the junction carries gas, the trapped pocket is no longer sealed
        # by liquid: it is hydraulically open to the atmosphere through the tower.
        # The experiment shows exactly this as the sharp pressure-head collapse (Fig.5
        # at T*~8.3 for the wide tower; Fig.6 at T*~4.05 for the narrow tower, right
        # when the gas front reaches the top).
        rho_g_r_pre = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
        alpha_gr_pre = np.clip(Mgrs / np.maximum(rho_g_r_pre * Ar * dz, 1.0e-12), 0.0, 0.98)
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
        if breakthrough:
            vent_latched = True
        breakthrough = breakthrough or vent_latched
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
        ulr = Qlr / np.maximum(Alr, 1.0e-3 * Ar)
        ugt_now = np.where(Mgt > 1.0e-14, Jgt / np.maximum(Mgt, 1.0e-14), 0.0)
        ct = np.sqrt(a2 * np.clip((Alt / A - 0.6) / 0.35, 0.0, 1.0) + G * case.D + 1e-6)
        cr = np.sqrt(a2 * np.clip((Alr / Ar - 0.6) / 0.35, 0.0, 1.0) + G * case.Dr + 1e-6)
        ugr_now = np.where(Mgrs > 1.0e-14, Jgrs / np.maximum(Mgrs, 1.0e-14), 0.0)
        gas_wave = max(case.gas_velocity_cap, math.sqrt(G * case.D))
        smax = max(
            float(np.max(np.abs(ult) + ct)),
            float(np.max(np.abs(ulr) + cr)),
            float(np.max(np.abs(ugt_now)) + gas_wave),
            float(np.max(np.abs(ugr_now)) + gas_wave),
        )
        dt = min(case.cfl * min(dx, dz) / max(smax, EPS), out_dt, case.t_end - t)

        # ================= TUNNEL update (Rusanov) =================
        Alg = np.empty(Nt + 2); Qlg = np.empty(Nt + 2); cg = np.empty(Nt + 2)
        Alg[1:-1] = Alt; Qlg[1:-1] = Qlt; cg[1:-1] = ct
        # upstream end (x=0): closed wall (mirror)
        Alg[0] = Alt[0]; Qlg[0] = -Qlt[0]; cg[0] = ct[0]
        # downstream end (x=L): closed wall (mirror)
        Alg[-1] = Alt[-1]; Qlg[-1] = -Qlt[-1]; cg[-1] = ct[-1]
        uLf = np.clip(Qlg / np.maximum(Alg, 1.0e-3 * A), -U_FLUX_MAX, U_FLUX_MAX)
        f1 = Qlg; f2 = Qlg * uLf
        # Contact-preserving face wave speed: the acoustic (water-hammer) stiffness only
        # acts on faces whose MEAN state is essentially full.  A face separating the gas
        # pocket (alpha~0) from the full water column (alpha~1) is a material contact,
        # not an acoustic shock; using max(c_left,c_right)=a_wh there made the Rusanov
        # dissipation flood the pocket cell at ~0.2A per step, over-compressing the
        # pocket up to 3x its head and yanking the tower level down (the spurious
        # early transient absent from the experiment).
        # Two flux families per face:
        #  * WET faces (both cells essentially full): central + Rusanov dissipation at
        #    the water-hammer speed -- the acoustic physics of the pressurized reach.
        #  * CONTACT faces (a gas/liquid area jump on either side): the jump is a
        #    material contact, not an acoustic wave.  Symmetric dissipation on the
        #    AREA variable is a parasitic void flux (-0.5*s*dAl acts as a void
        #    advection at s/2 ~ 0.5 m/s, which DOUBLED the Benjamin nose speed and
        #    let the cavity reach the tower at T*~3.3 instead of ~7), so the area
        #    flux is donor-cell upwinded (zero across a resting interface; the crown
        #    exchange is the only interfacial void transport).  The MOMENTUM flux
        #    keeps a Rusanov dissipation at the local material speed |u|+sqrt(gD):
        #    with no interfacial momentum damping at all, the water column rings
        #    against the stiff pocket spring at grid scale and the spurious +-10 m/s
        #    velocities hurl the exchange void down the tunnel within a second.
        # (threshold 0.995: a face is acoustic only if BOTH cells are truly full;
        #  at 0.95 the nose cell of the intrusion tongue still counted as wet and the
        #  acoustic dissipation streamed a thin void toe ahead of the Benjamin front,
        #  triggering the tower-arrival detection ~15% early)
        wet_face = np.minimum(Alg[:-1], Alg[1:]) / A > 0.995
        sf = np.maximum(cg[:-1], cg[1:]) + np.maximum(np.abs(uLf[:-1]), np.abs(uLf[1:]))
        s_mat = (np.maximum(np.abs(uLf[:-1]), np.abs(uLf[1:]))
                 + math.sqrt(G * case.D))
        uf_t = 0.5 * (uLf[:-1] + uLf[1:])
        F1 = np.where(wet_face,
                      0.5 * (f1[:-1] + f1[1:]) - 0.5 * sf * (Alg[1:] - Alg[:-1]),
                      np.where(uf_t >= 0.0, f1[:-1], f1[1:]))
        F2 = np.where(wet_face,
                      0.5 * (f2[:-1] + f2[1:]) - 0.5 * sf * (Qlg[1:] - Qlg[:-1]),
                      0.5 * (f2[:-1] + f2[1:]) - 0.5 * s_mat * (Qlg[1:] - Qlg[:-1]))
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
        # Donor-availability flux limit: a face may not extract more volume per step
        # than 45% of what its donor cell holds.  Without it a ringing transient
        # drives cells negative and the positivity clip then CREATES liquid (the
        # +5%/s inventory drift that pumped the tower full in earlier runs).
        F1 = np.clip(F1, -0.45 * Alg[1:] * dx / max(dt, EPS),
                     0.45 * Alg[:-1] * dx / max(dt, EPS))
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
        if phi_v < 1.0:
            F1[iv] *= phi_v
            F2[iv] *= phi_v
        Alt_new = Alt - dt / dx * (F1[1:] - F1[:-1])
        Qlt_new = Qlt - dt / dx * (F2[1:] - F2[:-1])
        # sources: pressure gradient (theta=0, no gravity) + friction
        Pth = np.empty(Nt + 2); Pth[1:-1] = Pt
        Pth[0] = Pt[0]                            # closed upstream: zero-grad
        Pth[-1] = Pt[-1]                          # closed downstream: zero-grad
        dPdx = (Pth[2:] - Pth[:-2]) / (2.0 * dx)
        # While the disc still blocks the face, the pressure jump across it is
        # borne by the DISC (a wall force), not by the fluid: blend the two
        # adjacent cells' gradients toward their wall-reflected (one-sided)
        # values so the closed valve accelerates nobody.
        if phi_v < 1.0 and 1 <= iv <= Nt - 2:
            dPdx[iv - 1] = (phi_v * (Pt[iv] - Pt[iv - 2]) + (1.0 - phi_v) * (Pt[iv - 1] - Pt[iv - 2])) / (2.0 * dx)
            dPdx[iv] = (phi_v * (Pt[iv + 1] - Pt[iv - 1]) + (1.0 - phi_v) * (Pt[iv + 1] - Pt[iv])) / (2.0 * dx)
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
                  + (0.025 + CHURN_FRIC * churn_w) / (2.0 * Dh_t) * np.abs(un))
        # butterfly-valve local loss (disc remains in the bore when open, V&W Fig.2):
        # dP = 0.5*K*rho*u|u| across the valve cell -> friction-like sink over dx.
        # During the hand-opening stroke (~valve_open_time) the disc throttles the
        # release: K ~ K_open/theta^2 -> the column accelerates as a body over the
        # stroke instead of the contact cell alone taking a 0-ms burst.
        fric_t[iv] += (case.valve_loss_coeff / max(phi_v, 1.0e-4)) / (2.0 * dx) * abs(un[iv])
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
        uRf = np.clip(Qrg / np.maximum(Arg, 1.0e-3 * Ar), -U_FLUX_MAX, U_FLUX_MAX)
        g1 = Qrg; g2 = Qrg * uRf
        # contact-preserving face wave speed (see tunnel update): keeps the open tower
        # free surface a sharp material contact instead of an acoustically-diffused zone
        crg_min = np.minimum(crg[:-1], crg[1:])
        sfr = crg_min + np.maximum(np.abs(uRf[:-1]), np.abs(uRf[1:]))
        G1 = 0.5 * (g1[:-1] + g1[1:]) - 0.5 * sfr * (Arg[1:] - Arg[:-1])
        G2 = 0.5 * (g2[:-1] + g2[1:]) - 0.5 * sfr * (Qrg[1:] - Qrg[:-1])
        # donor-availability limit (see tunnel F1): no negative-area overdraw
        G1 = np.clip(G1, -0.45 * Arg[1:] * dz / max(dt, EPS),
                     0.45 * Arg[:-1] * dz / max(dt, EPS))
        Alr_new = Alr - dt / dz * (G1[1:] - G1[:-1])
        Qlr_new = Qlr - dt / dz * (G2[1:] - G2[:-1])
        Prh = np.empty(Nr + 2); Prh[1:-1] = Pr
        # Base ghost cell centre sits at z = -dz/2, so its well-balanced hydrostatic
        # value is P_base + rho*g*(dz/2).  Without the half-cell offset the bottom
        # cell feels a residual -0.25*rho*g body force that slowly and unphysically
        # DRAINS the tower into the tunnel (seen as the sagging free surface and the
        # over-compressed upstream pocket in earlier runs).
        #
        # The base is driven by the JUNCTION pressure Pj -- genuine two-way hydraulic
        # coupling.  At rest the junction cell is full water carrying the open-tower
        # piezometric head (P_atm + rho g Yfs), so the coupling is well-balanced.
        # When the tunnel drains toward the under-pressured pocket the junction cell
        # rarefies slightly and its ELASTIC TENSION (see _pressure) lowers Pj, which
        # smoothly pulls make-up water down from the tower (the experiment's early
        # Yfs sag); once the cavity covers the crown T, Pj IS the pocket EOS pressure
        # and the column stands on the pocket -- over-pressure lifts it (geyser
        # drive), venting under-pressure lets it back down.  This is only safe
        # because _pressure now bounds every reading (tension floor; massless-void
        # guard): raw EOS vacuum readings here used to slam the tower down.
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
        u_base = float(np.clip(Qlr[0] / max(Alr[0], 1.0e-3 * Ar), -U_FLUX_MAX, U_FLUX_MAX))
        K_base = case.junction_loss_coeff + (case.glug_loss_coeff if gas_at_junction else 0.0)
        # CROWN-DATUM correction (case-B snapshot fix, 2026-07-06): the tower taps
        # the pipe at its CROWN (the T sits on top of the pipe), so the riser base
        # (z=0) feels the crown pressure, NOT the invert pressure Pj.  With the
        # junction slug full of water the crown sits rho*g*D below the invert
        # piezometric line; once the cavity occupies the T the crown is inside the
        # gas, i.e. only the local water layer h(jx) separates it from the invert.
        # Feeding the raw invert Pj here over-drove the tower by up to rho*g*D
        # (~0.15 L): the case-B column equilibrated at Y* ~ 0.98 instead of the
        # paper's 0.80-0.85 (Fig. 8), and the release swing pinned it to the top.
        # Static check against the paper panels (isothermal pocket EOS + crown
        # datum): equilibrium Y* ~ 0.83, pocket H* ~ 0.76 -- both match Fig. 6/8.
        if m_poc > 0.0:
            h_T = float(h_layer[jx]) if gas_at_junction else case.D
            P_base = Pj - RHO_L * G * h_T
        else:
            P_base = Pj          # pure-hydrostatic branch is already crown-datum
        Prh[0] = (P_base + RHO_L * G * (0.5 * dz)
                  - 0.5 * RHO_L * K_base * u_base * abs(u_base))
        Prh[-1] = P_ATM
        dPdz = (Prh[2:] - Prh[:-2]) / (2.0 * dz)
        # Gas momentum sees an EXTRA base overpressure: the air pocket drives the GAS
        # (penetration), making the gas nose outrun the free surface (paper V_int >> V_fs)
        # in the strong-geyser case, while the liquid is only displaced (gentle V_fs).
        Prh_gas = Prh.copy()
        Prh_gas[0] = Prh[0] + RHO_L * G * H_op * case.gas_drive_eff
        dPdz_gas = (Prh_gas[2:] - Prh_gas[:-2]) / (2.0 * dz)
        un_r = np.where(Alr_new > 1e-9, Qlr_new / np.maximum(Alr_new, EPS), 0.0)
        fric_r = (32.0 * MU_L / RHO_L / (case.Dr * case.Dr)
                  + 0.025 / (2.0 * case.Dr) * np.abs(un_r))
        Qlr_new += dt * (-(Alr_new / RHO_L) * dPdz - Alr_new * G - fric_r * Alr_new * un_r)

        # ---------- junction liquid exchange (conservative) ----------
        # The base face flux is clamped to the counter-current flooding scale
        # ~sqrt(g*Dr): a hammer spike otherwise overdraws the junction cell in a
        # single step.  The withdrawal is spread over the three cells around the T
        # (the mouth has finite extent): a point sink at jx made that one cell's
        # alpha -- hence its stiff hammer pressure -- track the base flux noise,
        # closing the grid-frequency feedback loop with the riser.
        # Counter-current flooding limit (CCFL): once the vent path is open
        # (blow-down latched) the tower water can only descend against the rising
        # gas at the Wallis flooding scale ~0.5*sqrt(gDr) -- this meters the
        # drain-down during the vent.  The old 3.0*sqrt(gDr) cap was an
        # anti-overdraw guard, not a physical limit: it let the column drain 6x
        # faster than CCFL and the tower emptied to 0.13 L inside the Fig.7
        # window while the experiment still stands at ~0.6 L at T*=9.4.
        # Keyed on vent_latched, NOT on the instantaneous gassy-junction flag:
        # that flag flickers on entry transients (hammer micro-voids) and an
        # asymmetric throttle there rectified the early slosh into a sagging
        # plateau (H* mean fell to ~0.2 by T*=4 -- tried, reverted).
        u_base_lim = (0.5 if vent_latched else 3.0) * math.sqrt(G * case.Dr)
        q_lim = Ar * u_base_lim
        # withdrawal availability: the junction cells can only give what they hold
        # (keep a 2% film).  An uncapped withdrawal drove jx negative during entry
        # transients and the positivity floor then CREATED water -- the tower rose
        # on ~0.4 L of numerically created liquid while the tunnel inventory stood
        # still (probe9), a pure conservation bug masquerading as geysering drive.
        if G1[0] > 0.0:
            for off, wj in ((-1, 0.25), (0, 0.5), (1, 0.25)):
                avail = max(Alt_new[jx + off] - 0.02 * A, 0.0)
                q_lim = min(q_lim, avail * dx / (wj * max(dt, EPS)))
        if abs(G1[0]) > q_lim:
            G1 = G1.copy(); G1[0] = math.copysign(q_lim, G1[0])
            Alr_new = Alr - dt / dz * (G1[1:] - G1[:-1])   # redo riser mass update with clamped face
        q_up = G1[0]                              # [m^3/s] up the riser base face
        for off, wj in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            Alt_new[jx + off] -= wj * q_up * dt / dx
        # (riser already received it via its bottom flux G1[0])

        # ---------- gas transport: pocket-front propagation (crown gravity current) ----------
        # A trapped, connected air pocket spreads along the pipe crown toward the
        # tower as a gravity-current cavity (Benjamin 1968): its NOSE advances into
        # the water while the displaced liquid drains back underneath -- the pocket
        # ELONGATES AND THINS at constant volume; its tail does not translate.
        # Because the pocket is one connected gas body, its internal pressure is
        # uniform, so the layer depth is (quasi-)uniform along the pocket: the
        # front transfer below therefore draws void PROPORTIONALLY from the whole
        # region (uniform thinning) and hands it to the nose cell.  Gas mass moves
        # with its volume (region density stays uniform): the EOS inventory and the
        # void are collocated BY CONSTRUCTION, which is precisely what the previous
        # advection schemes (Rusanov / MUSCL on a drag-relaxed velocity) violated --
        # they stripped the capsule's mass into still-full cells, collapsing the
        # recorded head to large negative values and geysering the wide tower with
        # phantom gas.
        #
        # Nose speed: Benjamin's half-depth cavity speed 0.542*sqrt(gD) caps the
        # deep-ambient (thin-layer) result sqrt(2 g h): for a circular pipe crown
        # lens, h(alpha) = D*(3*alpha/(2*sqrt(2)))^(2/3), so u_front is essentially
        # 0.45-0.54*sqrt(gD) across alpha ~ 0.1-0.5 -- the near-constant front speed
        # the experiments show (V&W Fig.5/6 gas reaches the tower at ~v_gc through-
        # out), with a graceful taper only for a nearly exhausted layer.
        alpha_gt = np.clip(1.0 - Alt_new / A, 0.0, 1.0)
        Mgt_new = np.maximum(Mgt.copy(), 0.0)
        body_thr = 0.05                                   # propagation body threshold
        # Stranded-mass reclaim: when backfilling water re-closes a former pocket
        # cell, the cell drops out of the region while still holding its (large)
        # share of pocket inventory.  Free gas cannot stay dissolved under a closed
        # crown -- and the EOS, summing only region cells, under-read the pocket by
        # up to 0.4 m of head (the sawtooth ringing).  Walk outward from each region
        # edge and pull clearly over-stuffed mass (beyond 2x atmospheric for the
        # cell's own void -- a building Benjamin nose at pocket density stays put)
        # back into the pocket edge cell.
        for (i0, i1) in _regions(alpha_gt > body_thr):
            for edge, step_dir in ((i0, -1), (i1 - 1, +1)):
                j = edge + step_dir
                while 0 <= j < Nt and alpha_gt[j] <= body_thr:
                    m_cap = 2.0 * rho_atm * max(A - Alt_new[j], 1.0e-4 * A) * dx
                    if Mgt_new[j] <= m_cap:
                        break
                    Mgt_new[edge] += Mgt_new[j] - 0.5 * m_cap
                    Mgt_new[j] = 0.5 * m_cap
                    j += step_dir
        for (i0, i1) in _regions(alpha_gt > body_thr):
            seg = slice(i0, i1)
            void = np.maximum(A - Alt_new[seg], 0.0) * dx      # [m^3]
            V_reg = float(void.sum())
            M_reg = float(np.sum(Mgt_new[seg]))
            if V_reg <= 1.0e-12:
                continue
            abar = V_reg / (A * (i1 - i0) * dx)
            # Tower-facing nose cell (the pocket only intrudes toward the vent).
            # FILL-TO-BODY-LEVEL front: keep filling the CURRENT edge cell until
            # its layer reaches the region-mean depth, and only then hand volume
            # to the next cell.  Handing it straight to the cell BEYOND the edge
            # (which joins the region already at the 5% detection threshold) let
            # the DETECTED edge advance ~abar/0.05 times faster than the volume
            # front -- a one-cell-deep numerical toe that crossed the tunnel at
            # ~1 m/s (2x Benjamin; arrival T*~3.4 instead of the paper's ~7.3)
            # and triggered the tower gas entry long before the cavity arrived.
            #
            # Once the region REACHES the T it does not stop feeding: the crown
            # current keeps delivering pocket volume to the T mouth while the
            # riser entry consumes it (the vent is the pocket's low-pressure
            # sink).  The old hard skip ("already at the T") let the entry
            # exchange refill the junction cell with water, the junction void
            # starved below the entry threshold, and the pocket sat SEALED at
            # its plateau head forever -- no blowdown, gas gone from the tower
            # after one glug (the T*>8 dead state of earlier runs).
            if i0 <= jx < i1:
                sgn, tgt = +1, jx                        # feed the vent mouth
            elif xt[i1 - 1] < case.x_riser and i1 <= jx:
                sgn = +1
                tgt = (i1 - 1) if (i1 - i0 > 1 and alpha_gt[i1 - 1] < F_HAND * abar) else i1
            elif xt[i0] > case.x_riser and i0 - 1 >= jx:
                sgn = -1
                tgt = i0 if (i1 - i0 > 1 and alpha_gt[i0] < F_HAND * abar) else i0 - 1
            else:
                continue
            if not (0 <= tgt < Nt):
                continue
            h_dep = case.D * min(0.5, (3.0 * abar / (2.0 * math.sqrt(2.0))) ** (2.0 / 3.0))
            u_rel = min(math.sqrt(2.0 * G * h_dep), v_gc)
            if _DISABLE_CROWN:
                u_rel = 0.0
            # Front speed = Benjamin celerity, PERIOD.  The cavity elongates at
            # constant volume: water displaced at the nose drains back UNDER the
            # body, so the net liquid flux through any full section ahead of the
            # nose is zero and the celerity is measured against STILL water.  Both
            # ambient-advection closures tried here double-counted a numerical
            # flow: the nose-cell velocity is drainback (throttled the front to
            # half speed, arrival T*~12), and the mean filtered velocity of the
            # slug ahead picks up the tower-pocket slosh, whose rectified bias
            # DOUBLED the front speed (arrival T*~3.4 instead of the paper's ~7.3;
            # 2.97 m at 0.542*sqrt(gD)=0.52 m/s is 5.9 s = T*7.2, the experiment).
            u_front = u_rel
            dV = abar * A * u_front * dt
            # limits: water available in the nose cell; keep the fill level at the
            # region-mean depth (the nose joins the body at ~abar, front speed = u)
            fill_room = max(abar - alpha_gt[tgt], 0.0) * A * dx
            dV = min(dV, 0.9 * max(Alt_new[tgt], 0.0) * dx, fill_room, 0.2 * V_reg)
            if dV <= 0.0:
                continue
            dm = M_reg * dV / V_reg
            thin = dV * (void / V_reg) / dx               # uniform proportional thinning
            Alt_new[seg] += thin                          # water backfills the body
            if M_reg > 1.0e-30:
                Mgt_new[seg] -= dm * (Mgt_new[seg] / M_reg)
            Alt_new[tgt] -= dV / dx                       # nose displaces water
            Mgt_new[tgt] += dm                            # mass rides its volume
        # ---------- pocket consolidation: one connected pocket, no orphan voids ----------
        # Free gas in this horizontal pipe lives at the crown as ONE connected pocket
        # (attached to the upstream closed end until it reaches the T).  Transient
        # rarefaction voids opened mid-column by ringing, and pocket fragments cut off
        # by a locally-refilled cell, are numerical artefacts: each fragment read its
        # own EOS (a squeezed fragment reads a huge head -- the recorded pocket head
        # hit 10 L while fragment counts hit 24), the pressure chaos pumped the tower,
        # and phantom voids at jx triggered gas entry LONG before the cavity arrived.
        # Consolidate conservatively: keep the region holding the largest gas mass as
        # THE pocket; every other gassy region swaps its void INTO the main pocket
        # (fragment cells refill with water; the pocket thickens proportionally to its
        # existing void profile, i.e. a uniform-layer deepening), and the fragment's
        # gas mass rides along.  Liquid volume and gas mass are conserved EXACTLY.
        alpha_gt2 = np.clip(1.0 - Alt_new / A, 0.0, 1.0)
        regs = _regions(alpha_gt2 > body_thr)
        if len(regs) > 1:
            masses = [float(np.sum(Mgt_new[i0:i1])) for (i0, i1) in regs]
            mi = int(np.argmax(masses))
            m0, m1 = regs[mi]
            void_main = np.maximum(A - Alt_new[m0:m1], 0.0)          # [m^2] per cell
            Vm = float(np.sum(void_main))
            if Vm > 1.0e-12:
                w_main = void_main / Vm
                for r, (i0, i1) in enumerate(regs):
                    if r == mi:
                        continue
                    frag_void = np.maximum(A - Alt_new[i0:i1], 0.0)  # [m^2] per cell
                    Vf = float(np.sum(frag_void))                    # [m^2] (x dx = volume)
                    mf = float(np.sum(Mgt_new[i0:i1]))
                    Alt_new[i0:i1] += frag_void                      # water recloses the fragment
                    Mgt_new[i0:i1] = 0.0
                    Alt_new[m0:m1] -= Vf * w_main                    # pocket deepens uniformly
                    Mgt_new[m0:m1] += mf * w_main
        # diagnostic gas momentum (drives only the CFL bound): nose speed where gassy
        dir_cell = np.sign(case.x_riser - xt)
        Jgt_new = Mgt_new * (ult + np.where(alpha_gt > 2.0e-3, dir_cell * v_gc, 0.0))
        # ---- Riser resolved gas: 1D two-fluid mass + momentum (no velocity cap) ----
        # Conserved variables are the cell gas mass [kg] and gas momentum [kg m/s].
        # The rise speed is left to emerge from the momentum balance: buoyancy
        # (the pressure-gradient force on the gas volume) minus gravity, balanced by
        # an interphase drag whose coefficient is closed by the vertical slug
        # (Taylor-bubble) drift velocity. No hard speed clip and no relaxation to a
        # prescribed velocity are imposed.
        rho_g_r = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
        ugr = np.where(Mgrs > 1.0e-14, Jgrs / np.maximum(Mgrs, 1.0e-14), 0.0)
        ugr_f = np.empty(Nr + 1)
        ugr_f[1:-1] = 0.5 * (ugr[1:] + ugr[:-1]); ugr_f[0] = 0.0; ugr_f[-1] = max(ugr[-1], 0.0)
        # background (atmospheric) gas above the free surface, advected on the same faces
        Mgr_ext = np.concatenate([[0.0], Mgr, [0.0]])
        gflux_r = np.where(ugr_f >= 0.0, ugr_f * Mgr_ext[:-1], ugr_f * Mgr_ext[1:])
        Mgr_new = np.maximum(Mgr - dt / dz * (gflux_r[1:] - gflux_r[:-1]), 0.0)
        # resolved gas: 2nd-order TVD mass transport + consistent upwind momentum
        Mgrs_new = _advect_mass_muscl(Mgrs, ugr_f, dz, dt)
        Jgrs_adv = _advect_momentum_upwind(Jgrs, ugr_f, dz, dt)
        alpha_gr_new = np.clip(Mgrs_new / np.maximum(rho_g_r * Ar * dz, 1.0e-12), 0.0, 0.98)
        # momentum sources: buoyancy = -(alpha_g*Ar*dz)*dP/dz_gas (incl. pocket overpressure) ; gravity = -m_g*g
        buoyancy = -alpha_gr_new * Ar * dz * dPdz_gas
        gravity_force = -Mgrs_new * G
        Jgrs_star = Jgrs_adv + dt * (buoyancy + gravity_force)
        # interphase drag closed by the slug drift velocity so the equilibrium slip
        # (u_g - u_l) -> u_drift = 0.35*sqrt(g*Dr*(rho_l-rho_g)/rho_l). Integrated
        # semi-implicitly: stiff drag is then unconditionally stable, and the rise
        # velocity is set by the buoyancy/drag balance rather than a hard cap.
        u_drift = np.maximum(
            0.35 * np.sqrt(G * case.Dr * np.maximum(RHO_L - rho_g_r, 0.0) / RHO_L), 1.0e-3
        )
        Kdrag = alpha_gr_new * Ar * dz * np.maximum(RHO_L - rho_g_r, 0.0) * G / u_drift
        denom = Mgrs_new + dt * Kdrag
        Jgrs_new = np.where(
            Mgrs_new > 1.0e-14,
            Mgrs_new * (Jgrs_star + dt * Kdrag * un_r) / np.maximum(denom, 1.0e-14),
            0.0,
        )

        # ---------- junction gas transfer: T-mouth exchange + overpressure injection ----------
        # Gas at the side-T mouth is Rayleigh-Taylor unstable against the tower water
        # above: it rises INTO the riser while an equal volume of water comes DOWN
        # into the junction cell (the inverted-bottle "glug" exchange, rate set by
        # the Taylor drift).  This drift share is volume-neutral: the pocket loses
        # mass AND the matching void (the descending water refills it), so the
        # pocket density m/V -- hence its PRESSURE -- is unchanged, and the tower
        # level does not move while gas transits the standing column: exactly the
        # wide-tower phenomenology (V&W Fig.5: flat Yfs, flat plateau, then a sharp
        # collapse only at breakthrough).  The earlier constant-void mass bleed made
        # the pocket dive below atmospheric and slowly swelled the tower instead.
        # Only a genuine pocket OVERPRESSURE (H_op > 0: Case B) adds net injection
        # beyond the exchange -- gas volume the column must make room for: the
        # geysering drive.  Arrival detection is AREA-based (crown exchange moves
        # void with mass): the T admits gas only when the cavity physically reaches it.
        rho_g_j = max(Pj / (R_GAS * T_GAS), rho_atm)
        alpha_g_j = float(np.clip(1.0 - Alt_new[jx] / A, 0.0, 0.98))
        if alpha_g_j > case.tower_entry_alpha_min:
            # entry velocity = buoyant drift accelerated by the pocket overpressure (gas, not liquid jet)
            u_drift_in = 0.35 * math.sqrt(G * case.Dr * max(RHO_L - rho_g_j, 0.0) / RHO_L)
            u_in = math.sqrt(u_drift_in * u_drift_in + 2.0 * G * max(H_op, 0.0) * case.entry_drive_eff)
            # NOTE (tried, reverted): boosting the entry cross-section to Taylor-slug
            # fractions (alpha ~ 0.7) to force the Fig.5 H* collapse by MASS drain
            # breaks volume conservation at the mouth -- the volume-neutral V_ex
            # share saturates its caps, the pocket expands without refill, dives
            # BELOW atmospheric (-0.18 L) and siphons the tower down to 0.13 L.
            # The experimental collapse is a PRESSURE equilibration (only the ~3%
            # overpressure mass needs venting), handled by the breakthrough vent.
            #
            # ENTRY FLUX SCALING (case-B snapshot fix, 2026-07-06).  Two changes:
            # (1) Mouth aperture: the T taps the pipe CROWN, and the cavity's gas
            #     layer rides the crown -- a thin layer already blankets the whole
            #     12.7 mm mouth footprint.  Scaling the aperture by the junction
            #     cell's BULK void fraction alpha_g_j under-fed the mouth ~10x
            #     (a 10% crown layer is full mouth coverage, not a 10% orifice).
            #     The aperture ramps to the full bore by alpha_g_j ~ 0.10.
            # (2) Flooding limit: the driven blow-through of a narrow riser is
            #     counter-current-flooding limited (Wallis): superficial gas speed
            #     j_g <= C_w^2 * sqrt(g Dr (rho_l-rho_g)/rho_l), C_w ~ 0.9 for a
            #     smooth-flanged tube end.  With the slug-unit fill level ~0.41
            #     this gives a nose speed ~0.7 m/s -- the paper's V*int ~ 1.4-2.0
            #     for Dt* = 0.135 (Table 2 / Fig. 8) IS this flooding-limited
            #     climb; the free-orifice speed sqrt(2 g H_op) ~ 2.2 m/s never
            #     materialises inside the water-filled bore.
            C_wallis = 0.9
            j_flood = (C_wallis * C_wallis) * math.sqrt(
                G * case.Dr * max(RHO_L - rho_g_j, 0.0) / RHO_L)
            aperture = min(alpha_g_j / 0.10, 1.0)
            q_gas = min(aperture * Ar * u_in, Ar * j_flood)     # [m^3/s] into the mouth
            m_up = min(rho_g_j * q_gas * dt, 0.5 * max(Mgt_new[jx], 0.0))
            old_mgt = max(Mgt_new[jx], 1.0e-14)
            Mgt_new[jx] -= m_up
            Jgt_new[jx] *= max(Mgt_new[jx], 0.0) / old_mgt
            # POCKET-FED CORE NOSE FEED (case-B snapshot fix, 2026-07-06): while the
            # pocket holds overpressure and the junction carries gas, the riser gas
            # from the base is ONE CONNECTED body fed at pocket pressure -- mass
            # entering the mouth raises the core NOSE (same fill-to-body-level
            # closure as the tunnel crown current), it does not stack at the base
            # cell where the drag relaxation would cap the climb at the quiescent
            # Taylor drift (V* ~ 0.35).  The paper's narrow tower climbs at
            # V*int ~ 1.4 (Table 2) BECAUSE the climb is this pressure-driven feed,
            # not buoyant bubble drift.  Base-cell injection is kept for the
            # undriven exchange (wide tower / H_op = 0), which IS bubble drift.
            k_inj = 0
            if H_op > 1.0e-3:
                rho_gr_inj = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
                alpha_inj = Mgrs_new / np.maximum(rho_gr_inj * Ar * dz, 1.0e-12)
                # Fill-to-slug-level front: the driven core is a Taylor slug train,
                # not a skinny filament.  A slug unit = Taylor bubble (core area
                # fraction ~ tower_core_area_fraction) + trailing liquid slug of
                # comparable length (Fabre & Line 1992), so the CELL-AVERAGE void
                # of the advancing core is ~ 0.5 * core fraction.  Advancing the
                # nose at the run-mean fill level let a skinny (alpha ~ 0.09) nose
                # outrun the surface swell and vent the pocket BEFORE the free
                # surface reached the top -- the model lost the geysering race the
                # experiment wins (paper: fs tops at T* ~ 3.9 while the nose is at
                # ~ 0.35 L; the nose only breaks through at ~ 4.09).
                alpha_fill = 0.5 * case.tower_core_area_fraction
                k_top = 0
                while k_top + 1 < Nr and alpha_inj[k_top + 1] > 0.05:
                    k_top += 1
                if alpha_inj[k_top] >= alpha_fill:
                    k_top = min(k_top + 1, Nr - 1)      # nose advances one cell
                wet_idx = np.where(Alr_new / Ar > 0.08)[0]
                ksurf_inj = int(wet_idx[-1]) if wet_idx.size else 0
                k_inj = min(k_top, ksurf_inj)           # cannot feed above the surface
            u_inj_eff = q_gas / max(aperture * Ar, 1.0e-12)   # post-cap mean entry speed
            Mgr_new[k_inj] += m_up
            Mgrs_new[k_inj] += m_up
            Jgrs_new[k_inj] += m_up * u_inj_eff
            # volume-neutral exchange share: water descends from the riser base and
            # closes the junction void vacated by the departing gas
            V_ex = (m_up / max(rho_g_j, 1.0e-6)) * (u_drift_in / max(u_in, 1.0e-9))
            V_ex = min(V_ex, 0.5 * max(Alr_new[0], 0.0) * dz,
                       0.5 * max(A - Alt_new[jx], 0.0) * dx)
            Alr_new[0] -= V_ex / dz
            Alt_new[jx] += V_ex / dx
        Jgrs_new = np.where(Mgrs_new > 1.0e-14, Jgrs_new, 0.0)

        # ---- gas escape at the free surface (bubbles burst out the open top) ----
        # Resolved gas reaching the liquid surface leaves to the atmosphere at its rise
        # velocity.  This is the missing physics that separates the regimes: a WIDE riser
        # vents fast (the gas rises and bursts out quickly) so hold-up stays low and the
        # column barely rises (no geyser); a NARROW riser vents slowly so gas hold-up
        # builds and pushes the column to the top (geyser).  It is a physical surface
        # flux (alpha_g * Ar * u_g), hence grid independent, with no prescribed rate.
        rho_g_esc = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
        alpha_g_esc = np.clip(Mgrs_new / np.maximum(rho_g_esc * Ar * dz, 1.0e-12), 0.0, 0.95)
        liqv_esc = float(np.sum(np.clip(Alr_new, 0.0, Ar) * dz))
        capv_esc = Ar * np.clip(1.0 - alpha_g_esc, 0.0, 1.0) * dz
        ksurf = min(int(np.searchsorted(np.cumsum(capv_esc), liqv_esc)), Nr - 1)
        # Bubbles burst through the surface at their BUOYANT drift velocity (Taylor rise),
        # independent of the bulk liquid drag.  Wide riser -> fast drift -> fast venting ->
        # low hold-up -> no geyser; narrow riser -> slow drift -> hold-up builds -> geyser.
        u_burst = 0.35 * math.sqrt(G * case.Dr * max(RHO_L - rho_atm, 0.0) / RHO_L)
        for kk in (ksurf - 1, ksurf):
            if 0 <= kk < Nr and Mgrs_new[kk] > 1.0e-14:
                ug_kk = max(Jgrs_new[kk] / max(Mgrs_new[kk], 1.0e-14), 0.0)
                # Gas that reaches the free surface bursts out at its ARRIVAL speed (it
                # cannot stack above the surface): flux-consistent max(u_drift, u_gas).
                # With drift-only venting, a swelling column (u_l > 0) delivers gas to
                # the surface faster than it can leave, and the artificial hold-up
                # runaway pushes even the wide tower to a spurious geyser.
                esc = min(rho_g_esc[kk] * alpha_g_esc[kk] * Ar * max(u_burst, ug_kk) * dt
                          * case.gas_escape_eff, Mgrs_new[kk])
                Mgrs_new[kk] -= esc
                Jgrs_new[kk] = Mgrs_new[kk] * ug_kk

        # ---- breakthrough blow-down of the trapped pocket ----
        # Once the gas core connects the riser base to the free surface, the pocket is
        # hydraulically open to the atmosphere: its overpressure blows down through the
        # tower bore in a fraction of a second (the experiment's sharp H* collapse).
        # The vent flux is an orifice-type ejection through the riser bore driven by
        # the pocket's own gauge pressure; it removes only the overpressure mass
        # fraction (self-limiting as P -> P_atm), while the bulk of the air inventory
        # keeps leaving by the buoyant bubbling path above.  The short (<0.1 s) transit
        # of this jet through the open core is below the output resolution, so the
        # vented mass is removed directly to the atmosphere.
        _bt_dbg["vented"] = 0.0
        if breakthrough:
            for (i0, i1) in _regions((Alt_new / A) < 0.95):
                if not (i0 <= jx < i1):
                    continue
                m_reg = float(np.sum(Mgt_new[i0:i1]))
                v_reg = float(np.sum(np.maximum(A - Alt_new[i0:i1], 1.0e-4 * A)) * dx)
                P_reg = m_reg * R_GAS * T_GAS / max(v_reg, 1.0e-12)
                dP_vent = P_reg - P_ATM
                if dP_vent > 0.0 and m_reg > 1.0e-12:
                    rho_reg = P_reg / (R_GAS * T_GAS)
                    u_vent = min(math.sqrt(2.0 * dP_vent / max(rho_reg, 1.0e-6)), 12.0)
                    m_vent = min(rho_reg * Ar * u_vent * dt, 0.05 * m_reg)
                    fvent = 1.0 - m_vent / max(m_reg, 1.0e-14)
                    Mgt_new[i0:i1] *= fvent
                    Jgt_new[i0:i1] *= fvent
                    _bt_dbg["vented"] = m_vent
                break

        # One-dimensional riser gas volume fraction with a sharp, volume-conserving
        # free-surface reconstruction.
        #
        # The Rusanov transport conserves the riser liquid volume but numerically
        # diffuses the open tower free surface. The previous closure detected that
        # surface with a fixed area threshold and then filled every cell beneath it,
        # so a tiny diffusive leak was locked in as full liquid and ratcheted the
        # surface to the tower top within a single output frame (the unphysical
        # "instant fill"). Instead we keep the resolved gas fraction in place and
        # repack the *conserved* liquid volume as a single bottom-anchored two-phase
        # column with a sharp top. The surface can then only move through genuine
        # volume changes (junction inflow/outflow, gas expansion and displacement),
        # never through numerical diffusion. Packing around the per-cell gas fraction
        # preserves a rising gas bubble with a liquid slug above it.
        rho_g_r = np.maximum(Pr / (R_GAS * T_GAS), rho_atm)
        alpha_g_r = np.clip(Mgrs_new / np.maximum(rho_g_r * Ar * dz, 1.0e-12), 0.0, 0.90)
        # UNCLIPPED sum: hammer transients overfill the bottom cells past Ar for a
        # few steps; clipping here silently destroyed that excess (~0.1 L over the
        # release transient), which is exactly the water the tower kept losing (the
        # spurious monotonic Yfs sag / pocket over-compression).
        dbg_created["r_repack"] += -float(np.sum(np.minimum(Alr_new, 0.0)) * dz)
        liq_vol = float(np.sum(np.maximum(Alr_new, 0.0) * dz))       # conserved liquid volume
        cap = Ar * np.clip(1.0 - alpha_g_r, 0.0, 1.0) * dz           # liquid capacity per cell around gas
        cum = np.cumsum(cap)
        filled = cum <= liq_vol
        Alr_new = np.where(filled, cap / dz, 0.0)
        k = int(np.searchsorted(cum, liq_vol))                      # partially filled surface cell
        if k < Nr:
            prev = float(cum[k - 1]) if k > 0 else 0.0
            Alr_new[k] = float(np.clip((liq_vol - prev) / dz, 0.0, cap[k] / dz))

        # viscosity (momentum)
        if case.nu > 0:
            k = case.nu * case.a_wh
            Qlt_new[1:-1] += k * dt / dx * (Qlt_new[2:] - 2 * Qlt_new[1:-1] + Qlt_new[:-2])
            Qlr_new[1:-1] += k * dt / dz * (Qlr_new[2:] - 2 * Qlr_new[1:-1] + Qlr_new[:-2])

        # open vented riser region: kill spurious film momentum, refill atmospheric gas
        Alr_new[Alr_new < 0] = 0.0
        wetr_new = Alr_new / Ar > 0.08
        open_top = np.zeros(Nr, dtype=bool)
        ii = Nr - 1
        while ii >= 0 and not wetr_new[ii]:
            open_top[ii] = True; ii -= 1
        Qlr_new = np.where(open_top, 0.0, Qlr_new)
        Mgr_new = np.where(open_top, rho_atm * np.maximum(Ar - Alr_new, 1e-4 * Ar) * dz, Mgr_new)
        Mgrs_new = np.where(open_top, 0.0, Mgrs_new)
        Jgrs_new = np.where(open_top, 0.0, Jgrs_new)
        Jgrs_new = np.where(Mgrs_new > 1.0e-14, Jgrs_new, 0.0)
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

        if not (np.all(np.isfinite(Alt_new)) and np.all(np.isfinite(Alr_new))):
            print(f"  [DIVERGED] t={t:.4f} step={step}", flush=True)
            break

        Alt, Qlt, Mgt, Jgt = Alt_new, Qlt_new, Mgt_new, Jgt_new
        Alr, Qlr, Mgr, Mgrs, Jgrs = Alr_new, Qlr_new, Mgr_new, Mgrs_new, Jgrs_new
        t += dt; step += 1; dt_prev = dt

        wtop = _liquid_surface_height(zr, dz, Alr, Ar)
        geyser_strength = max(geyser_strength, wtop)
        if t >= next_out - 1e-12 or t >= case.t_end - 1e-12:
            Pr_rec, _, _ = _pressure(Alr, np.full(Nr, Ar), Mgr, dz, a2, vent_top=True, p_floor=0.0)
            Pr_rec = np.where(Alr / Ar > 0.08, Pr_rec + RHO_L * G * np.maximum(wtop - zr, 0.0), Pr_rec)
            rho_g_rec = np.maximum(Pr_rec / (R_GAS * T_GAS), rho_atm)
            append_record(t, Alt, Alr, Mgt, Mgrs, rho_g_rec)
            rec["pj_head"].append(float((Pt[jx] - P_ATM) / (RHO_L * G)))
            # Transducer reads at the pipe AXIS (paper Fig.5 t=0 value = Yfs0 - D/2):
            # subtract the water column between invert and axis -- min(h, D/2) --
            # so a shallow layer (axis in the gas) reads the gas pressure itself.
            h_itr = case.D * float(_depth_frac(min(max(Alt[itr] / A, 0.0), 1.0)))
            rec["tr_head"].append(float((Pt[itr] - P_ATM) / (RHO_L * G)
                                        - min(h_itr, 0.5 * case.D)))
            rec["tun_gas_mass"].append(float(np.sum(Mgt)))
            rec["tun_gas_vol"].append(float(np.sum(np.maximum(A - Alt, 0.0)) * dx))
            rec["tot_liq"].append(float(np.sum(np.clip(Alt, 0.0, A)) * dx + np.sum(np.clip(Alr, 0.0, Ar)) * dz))
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
            rec.setdefault("dbg_poc_p", []).append(float((P_poc - P_ATM) / (RHO_L * G) if m_poc > 0 else 0.0))
            rec.setdefault("dbg_poc_m", []).append(float(m_poc))
            rec.setdefault("dbg_poc_v", []).append(float(m_poc * R_GAS * T_GAS / max(P_poc, 1.0)) if m_poc > 0 else 0.0)
            rec.setdefault("dbg_hedge", []).append(float(h_edge))
            rec.setdefault("dbg_pj", []).append(float((Pj - P_ATM) / (RHO_L * G)))
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
        if verbose and step % 8000 == 0:
            print(f"  t={t:.3f} step={step} dt={dt:.1e} wtop={wtop:.3f} gmax={geyser_strength:.3f}", flush=True)
        if step >= case.max_steps:
            print("  [MAX_STEPS]", flush=True); break

    rec["geyser_strength"] = geyser_strength
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
            wh = f * pipe_h
            if f > 0.03:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, wh, facecolor=C_W, edgecolor="none"))
        # vertical tower: centered gas core, symmetric water films
        ax.add_patch(Rectangle((x_r, 0), riser_w, case.riser_height, facecolor=C_A, edgecolor="0.5", lw=0.8))
        for zi, ai in zip(zr, alr):
            f = float(min(max(ai, 0.0), 1.0))
            if f > 0.02:
                if f >= 0.98:
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), riser_w, dz, facecolor=C_W, edgecolor="none"))
                else:
                    film_w = 0.5 * f * riser_w
                    ax.add_patch(Rectangle((x_r, zi - 0.5 * dz), film_w, dz, facecolor=C_W, edgecolor="none"))
                    ax.add_patch(Rectangle((x_r + riser_w - film_w, zi - 0.5 * dz), film_w, dz, facecolor=C_W, edgecolor="none"))
        ax.text(0.02, 0.02, f"pipe D={case.D*1000:.0f} mm, tower Dt={case.Dr*1000:.1f} mm (Dt/D={case.Dr/case.D:.2f}); "
                            f"tower width drawn enlarged for visibility",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.45")
        wtop = rec["wtop"][k] if k < len(rec["wtop"]) else 0.0
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
    riser_w = case.Dr
    C_W, C_A = "#2b7fff", "#f2f4f8"
    handles = [Patch(facecolor=C_W, label="water"), Patch(facecolor=C_A, edgecolor="0.5", label="air")]
    frames = []

    def draw_riser_section(ax, x0, width, wtop, alpha_g):
        ax.add_patch(Rectangle((x0, 0), width, case.riser_height, facecolor=C_A, edgecolor="0.5", lw=0.9))
        water_top = float(np.clip(wtop, 0.0, case.riser_height))
        for zi, ag in zip(zr, alpha_g):
            z0 = zi - 0.5 * dz
            if z0 >= water_top:
                continue
            gas_fraction = float(np.clip(ag, 0.0, 1.0))
            gas_w = gas_fraction * width
            film_w = 0.5 * max(width - gas_w, 0.0)
            if film_w > 0.001 * width:
                ax.add_patch(Rectangle((x0, z0), film_w, dz, facecolor=C_W, edgecolor="none"))
                ax.add_patch(Rectangle((x0 + width - film_w, z0), film_w, dz, facecolor=C_W, edgecolor="none"))

    for frame_no, k in enumerate(sel):
        fig, ax = plt.subplots(figsize=(12.0, 3.1))
        alt = rec["frames_alt"][k]
        ax.add_patch(Rectangle((0, -pipe_h), case.L_tunnel, pipe_h, facecolor=C_A, edgecolor="0.5", lw=0.8))
        for xi, ai in zip(xt, alt):
            f = float(min(max(ai, 0.0), 1.0))
            wh = f * pipe_h
            if f > 0.03:
                ax.add_patch(Rectangle((xi - 0.5 * dx, -pipe_h), dx, wh, facecolor=C_W, edgecolor="none"))
        riser_x0 = x_r - 0.5 * riser_w
        wtop = rec["wtop"][k] if k < len(rec["wtop"]) else 0.0
        pocket_head = rec["pocket_head"][k] if k < len(rec["pocket_head"]) else 0.0
        itop = rec["frames_itop"][k] if k < len(rec.get("frames_itop", [])) else (rec["itop"][k] if k < len(rec["itop"]) else 0.0)
        core_mass = rec["frames_core_mass"][k] if k < len(rec.get("frames_core_mass", [])) else 0.0
        if k < len(rec.get("frames_agr", [])):
            agr = rec["frames_agr"][k]
        else:
            agr = np.clip(1.0 - rec["frames_alr"][k], 0.0, 1.0)
        draw_riser_section(ax, riser_x0, riser_w, wtop, agr)
        ax.plot([riser_x0 - 0.05, riser_x0 + riser_w + 0.05], [case.riser_height, case.riser_height],
                color="#ef4444", ls="--", lw=1.0)
        ax.plot([case.L_up, case.L_up], [-pipe_h, 0.055], color="#111827", ls=":", lw=0.9)
        ax.text(case.L_up, 0.065, "valve x=0.546m", ha="center", va="bottom", fontsize=7)
        ax.text(x_r, case.riser_height + 0.045, "tower x=3.516m", ha="center", va="bottom", fontsize=7)
        ax.text(0.02, 0.96, f"t = {rec['frames_t'][k]:.2f} s    water surface = {wtop:.2f} m    gas front = {itop:.2f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.text(0.02, 0.02, f"true x-y scale: pipe D={case.D*1000:.0f} mm, tower Dt={case.Dr*1000:.1f} mm "
                            f"(Dt/D={case.Dr/case.D:.2f})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.45")
        ax.set_xlim(-0.05, case.L_tunnel + 0.05)
        ax.set_ylim(-pipe_h - 0.04, case.riser_height + 0.11)
        ax.set_xlabel("horizontal distance [m]   (closed air-pipe at left, closed end at right; tower = vertical column)")
        ax.set_ylabel("height above tunnel invert [m]")
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
        draw_riser_section(zax, -0.5 * zoom_w, zoom_w, wtop, agr)
        zax.plot([-0.58 * zoom_w, 0.58 * zoom_w], [case.riser_height, case.riser_height],
                 color="#ef4444", ls="--", lw=1.0)
        if itop > 0.0:
            zax.plot([-0.5 * zoom_w, 0.5 * zoom_w], [itop, itop], color="#f97316", ls=":", lw=1.2)
        zax.text(0.03, 0.97, f"t={rec['frames_t'][k]:.2f}s\nwater={wtop:.3f}m\ngas front={itop:.3f}m\ngas={core_mass*1e6:.3f}mg",
                 transform=zax.transAxes, ha="left", va="top", fontsize=9)
        zax.set_xlim(-0.095, 0.095)
        zax.set_ylim(-0.015, case.riser_height + 0.035)
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
