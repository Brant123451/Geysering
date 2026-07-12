# -*- coding: utf-8 -*-
"""Liu, Shao & Zhu (2020) JHE 146(2):04019055 -- junction-chamber geyser rig,
one-dimensional two-fluid network solver (Campaign 3 of the paper).

Apparatus (1:20 scale model of an Edmonton dropshaft):

    tank --Q(t)--> upstream pipe (L=5.80 m, D=0.20 m, slope 1:100)
                   --> junction chamber (0.30 x 0.30 x 0.45 m, invert drop 0.18 m)
                        |--> riser (dr=0.06 m, 1.22 m, open top) on the chamber lid
                   --> downstream pipe (Ld=5.95 m, Dd=0.28 m, horizontal)
                   --> overflow weir (open-channel tailwater, hd = Dd/4 in Series A)

One-dimensionalization (declared in the campaign README and in the paper):

* the two pipes are one-dimensional two-fluid PDE reaches (liquid area/flux
  cells, Rusanov transport, stiff full-pipe elastic overlay for the
  pressurized branch, region-EOS gas pockets under closed crowns);
* the chamber WITH the riser above it is a lumped STORAGE NODE (a surge
  tank): stage S below the lid follows the chamber plan area, above the lid
  it continues into the riser bore.  Geysering = stage reaching the riser
  top.  This is exactly the topology of the paper's own analytic model
  (their Eq. (7): a chamber--riser mass oscillator), so the lumping is the
  established reduced description of this rig;
* pipe--node coupling is by orifice-type face exchange with local-loss
  closures (quasi-steady over one cell transit).

Case A2: Q = 20 -> 100 L/s over ~0.4 s, downstream open channel (weir),
observed: NO geyser; PT3 0.99 kPa initially; strong PT1 fluctuation window
2--6 s; final steady PT2 = 2.15--2.22 kPa, PT3 = 4.94--4.99 kPa.

CASE-B3 COPY (this folder's frozen copy; two declared extensions over the
Case-A2 copy, both structural physics, no coefficient retuning):

1. downstream_full mode: the downstream pipe starts FULL (Series B initial
   condition) and its outlet is a full-bore free outfall (pressure head at
   the end section pinned to the crown when running full) instead of the
   Series-A tailwater weir.  This is an initial/boundary condition switch,
   not a model change.

2. the riser column is a RIGID-COLUMN ODE with inertia (pressure-driven
   acceleration, gravity, entrance/orifice loss on the moving column):
       (hr + L_eq) dwr/dt = g (H_b - hr) - K_riser wr|wr|/2,   dhr/dt = wr
   exchanging volume with the lid tap cell, and the chamber cells carry the
   same stiff elastic full-pipe overlay as the pipe cells (the A2 copy used
   a quasi-static column with a 0.25 s recording lag as an inertia
   surrogate and a column-height compliance instead of the elastic branch
   -- adequate for the non-geysering branch, but eruption velocity and the
   slam pressure peak ARE the observables of Case B3, so the surrogate is
   replaced by the actual momentum balance).  Geysering = the column top
   reaching the riser top with wr > 0; the ejection height is recorded as
   h_jet = Hr + wr^2/2g (ballistic height of the mixture leaving the top).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G = 9.81
RHO_L = 998.0
P_ATM = 101325.0
R_GAS = 287.0
T_GAS = 293.0
RHO_ATM = P_ATM / (R_GAS * T_GAS)
EPS = 1.0e-12
U_FACE_MAX = 5.0   # junction-face velocity cap [m/s]; the measured filling
                   # bore runs at 4.56 m/s, so a 3 m/s cap throttled the
                   # 100 L/s inflow (needs 3.18 m/s through the 0.20 m pipe)


@dataclass
class LiuCase:
    # inflow step
    Q0: float = 0.020            # initial steady inflow [m3/s]
    Q1: float = 0.100            # final inflow [m3/s]
    t_step: float = 0.0          # valve step start [s] (after warm-up)
    Tv: float = 0.4              # valve ramp time [s]
    t_warmup: float = 10.0       # settle time at Q0 BEFORE the ramp; records are
                                 # reported with t = 0 at the ramp start.  The
                                 # analytic initial condition (normal depth +
                                 # storage node) is not the exact discrete steady
                                 # state: without the warm-up the adjustment
                                 # transient (a spurious surge at the pipe end,
                                 # a draining chamber, and a short-lived sealed
                                 # crown pocket at t~1.1 s) contaminated the
                                 # first 2 s of the comparison window.
    # upstream pipe
    Lu: float = 5.80
    Du: float = 0.20
    slope_u: float = 0.01        # 1:100, falling toward the chamber
    # junction chamber (rectangular) + riser above it
    Lc: float = 0.30             # along-flow length
    Wc: float = 0.30             # width
    Hc: float = 0.45             # height above chamber bottom
    drop: float = 0.18           # upstream invert above chamber bottom
    dr: float = 0.06
    Hr: float = 1.22
    # downstream pipe
    Ld: float = 5.95
    Dd: float = 0.28
    hd0_frac: float = 0.25       # initial tailwater depth hd = Dd/4
    downstream_full: bool = False  # Series-B mode: downstream pipe starts FULL
                                   # and discharges as a full-bore outfall over
                                   # the hd/Dd=1 overflow-weir tailwater
                                   # (outlet piezometric head pinned at the
                                   # crown); the Series-A quarter-depth weir
                                   # rating is not used in this mode
    # numerics
    dx: float = 0.05
    cfl: float = 0.30
    a_wh: float = 40.0           # stiff pressurized-branch wave speed [m/s]
    t_end: float = 14.0
    # closures (constants; no per-case fitting)
    weir_Cd: float = 0.62        # standard sharp-crested value (NOT fitted); at
    weir_b: float = 0.28         # Q0 = 20 L/s it gives hd = 0.088 m against the
    weir_crest: float = 0.0      # reported Dd/4 = 0.07 m (+25%), acceptable for
                                 # a rating whose geometry the paper omits
    n_mann: float = 0.010        # smooth PVC/acrylic (friction uses the true
                                 # circular hydraulic radius; the earlier crude
                                 # rectangular Rh over-frictioned the pipe and
                                 # backed it up from the inlet at Q0)
    K_in: float = 1.0            # pipe->chamber expansion loss (Borda-Carnot)
    K_out: float = 1.0           # chamber->pipe: sharp-edged contraction + the
                                 # 90-degree path turn over the invert drop
    K_riser: float = 2.0         # riser-base orifice loss (sharp-edged entry
                                 # through the lid + the 90-degree turn); damps
                                 # the bore-arrival surge into the column
    L_eq_riser: float = 0.24     # entrance equivalent length ~ 4*dr [m]
    H_tail: float = 0.28         # Series-B downstream boundary: constant
                                 # tailwater level of the receiving tank above
                                 # the downstream invert [m].  This is the
                                 # hydrostatic equivalent of the paper's
                                 # Series-B overflow-weir setting hd/Dd=1.
                                 # The detailed weir rating is not reported;
                                 # the level is set at the downstream crown
                                 # (= Dd), NOT fitted to any transient or
                                 # final-state outcome.
    K_exit: float = 0.0          # EXTRA outfall drag beyond the velocity-head
                                 # conversion.  The momentum-face closure
                                 # enforces piez - H_tail = (1 + K_exit) u^2/2g
                                 # at steady state; the textbook submerged exit
                                 # (full velocity-head dissipation in the tank)
                                 # corresponds to piez = H_tail, so the face's
                                 # built-in "1" already over-counts by one
                                 # velocity head and K_exit adds nothing on top
                                 # (K_exit = 0 keeps the face stable and leaves
                                 # a declared +u^2/2g = +0.13 m conservative
                                 # bias at 100 L/s).
    Cp_jet: float = 0.5          # INSTRUMENT closure, not flow physics: PT3 sits
                                 # on the chamber front wall directly in the path
                                 # of the inflow jet, so its reading carries a
                                 # stagnation share of the jet dynamic head
                                 # (Cp * rho*u^2/2, active only while the pipe
                                 # end runs full so the jet spans the chamber).
                                 # Cp=0.5 set once against the reported final
                                 # steady PT3 (4.99 kPa) and held fixed.

    @property
    def Au(self) -> float:
        return 0.25 * math.pi * self.Du ** 2

    @property
    def Ad(self) -> float:
        return 0.25 * math.pi * self.Dd ** 2

    @property
    def Aplan(self) -> float:
        return self.Lc * self.Wc             # chamber plan area

    @property
    def Ar(self) -> float:
        return 0.25 * math.pi * self.dr ** 2


# --------------------------------------------------------------------------
def _circ_depth_from_area(al: float, D: float) -> float:
    A_full = 0.25 * math.pi * D * D
    frac = min(max(al / A_full, 0.0), 1.0)
    if frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return D
    th_lo, th_hi = 0.0, 2.0 * math.pi
    for _ in range(40):
        th = 0.5 * (th_lo + th_hi)
        if th - math.sin(th) < 2.0 * math.pi * frac:
            th_lo = th
        else:
            th_hi = th
    return 0.5 * D * (1.0 - math.cos(0.5 * th))


def _make_luts(D: float, n: int = 240):
    """area -> (depth, hydraulic radius) lookup for a circular section.
    The runtime friction MUST use the same circular geometry as the
    normal-depth initialization: a crude rectangular wetted perimeter made
    the friction ~15% high, the inlet cell backed up to full within 0.5 s at
    Q0 and a spurious pressurized front crept down the pipe."""
    a = np.linspace(1e-9, 0.25 * math.pi * D * D, n)
    h = np.array([_circ_depth_from_area(float(ai), D) for ai in a])
    th = 2.0 * np.arccos(np.clip(1.0 - 2.0 * h / D, -1.0, 1.0))
    pw = np.maximum(0.5 * th * D, 1e-6)
    rh = a / pw
    return a, h, rh


def _minmod(a, b):
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def _regions(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


class PipeReach:
    """one two-fluid PDE reach (liquid area + flux; gas mass for pocket EOS)"""

    def __init__(self, L, D, dx, zinv_fun, n_mann, a_wh):
        self.N = int(round(L / dx))
        self.dx = dx
        self.D = D
        self.A = 0.25 * math.pi * D * D
        self.x = (np.arange(self.N) + 0.5) * dx
        self.zinv = zinv_fun(self.x)
        self.n = n_mann
        self.a2 = a_wh ** 2
        self.a_wh = a_wh
        self.a_lut, self.h_lut, self.rh_lut = _make_luts(D)
        self.Al = np.zeros(self.N)
        self.Ql = np.zeros(self.N)
        self.Mg = np.zeros(self.N)
        self.gas_vented = 0.0     # net sealed-gas mass handed to vented zones [kg]

    def depth(self, Al=None):
        Al = self.Al if Al is None else Al
        return np.interp(Al, self.a_lut, self.h_lut)

    def pressures(self, vent_left: bool, vent_right: bool):
        """invert gauge pressure per cell (hydrostatic + pocket EOS + stiff overlay)"""
        h = self.depth()
        frac = self.Al / self.A
        P = RHO_L * G * h
        gassy = frac < 0.97
        for (i0, i1) in _regions(gassy):
            if (vent_left and i0 == 0) or (vent_right and i1 == self.N):
                continue
            m_reg = float(np.sum(self.Mg[i0:i1]))
            v_reg = float(np.sum(np.maximum(self.A - self.Al[i0:i1], 1e-6)) * self.dx)
            P_reg = m_reg * R_GAS * T_GAS / max(v_reg, 1e-12) - P_ATM
            P[i0:i1] = P[i0:i1] + max(P_reg, -0.5 * P_ATM)
        w_full = np.clip((frac - 0.97) / 0.03, 0.0, 1.0)
        P = P + w_full * RHO_L * self.a2 * np.clip(frac - 1.0, -0.02, 0.05)
        return P

    def gas_update(self, vent_left: bool, vent_right: bool):
        """Gas bookkeeping, CONSERVATIVE for sealed pockets.

        * cells that just became full strand their gas mass: it is swept to
          the nearest still-gassy cell (physically: the pocket is squeezed
          along the crown, not annihilated) -- the old code zeroed it, which
          made the t~1.1 s crown pocket vanish without a venting path;
        * sealed regions then redistribute their (augmented) mass over their
          void, keeping the region EOS meaningful;
        * zones connected to a vent (chamber, open outlet, inlet free surface)
          exchange freely with the atmosphere and track atmospheric density
          -- that exchange is physical (open air path), not a mass leak; the
          hand-off of sealed mass INTO a vented zone is logged.
        """
        void = np.maximum(self.A - self.Al, 0.0) * self.dx
        gassy = (self.Al / self.A) < 0.97
        Mg = self.Mg.copy()
        gassy_idx = np.where(gassy)[0]
        stranded = np.where(~gassy, Mg, 0.0)
        m_str = float(np.sum(stranded))
        if m_str > 1e-15:
            Mg = np.where(gassy, Mg, 0.0)
            if gassy_idx.size:
                for i in np.where(stranded > 1e-15)[0]:
                    j = int(gassy_idx[np.argmin(np.abs(gassy_idx - i))])
                    Mg[j] += stranded[i]
            else:
                self.gas_vented += m_str          # no gas zone left: vented
        for (i0, i1) in _regions(gassy):
            if (vent_left and i0 == 0) or (vent_right and i1 == self.N):
                self.gas_vented += float(np.sum(Mg[i0:i1]))  # hand-off logged
                Mg[i0:i1] = RHO_ATM * void[i0:i1]
                self.gas_vented -= float(np.sum(Mg[i0:i1]))
            else:
                m_reg = float(np.sum(Mg[i0:i1]))
                v_reg = float(np.sum(void[i0:i1]))
                Mg[i0:i1] = (m_reg * void[i0:i1] / max(v_reg, 1e-12)
                             if v_reg > 0 else 0.0)
        self.Mg = np.where(gassy, Mg, 0.0)

    def wave_speed(self):
        u = self.Ql / np.maximum(self.Al, 1e-4 * self.A)
        h = self.depth()
        frac = self.Al / self.A
        c = np.where(frac > 0.97, self.a_wh, np.sqrt(G * np.maximum(h, 1e-3)))
        return float(np.max(np.abs(u) + c))

    def step(self, dt, Q_in_face, Q_out_face, piez):
        """conservative Rusanov update with prescribed end-face volume fluxes.

        The mass flux carries the full Rusanov dissipation with the
        gravity-wave (or elastic, when full) celerity: without it the filling
        bore does not propagate (the first smoke run had the 100 L/s inflow
        piling at the inlet while the pipe end starved)."""
        N, dx = self.N, self.dx
        u = self.Ql / np.maximum(self.Al, 1e-4 * self.A)
        h = self.depth()
        frac = self.Al / self.A
        c = np.where(frac > 0.97, self.a_wh, np.sqrt(G * np.maximum(h, 1e-3)))

        F1 = np.empty(N + 1)
        F2f = np.empty(N + 1)
        sfc = np.maximum(np.abs(u[:-1]) + c[:-1], np.abs(u[1:]) + c[1:])
        F1[1:-1] = (0.5 * (self.Ql[:-1] + self.Ql[1:])
                    - 0.5 * sfc * (self.Al[1:] - self.Al[:-1]))
        F2f[1:-1] = (0.5 * (self.Ql[:-1] * u[:-1] + self.Ql[1:] * u[1:])
                     - 0.5 * sfc * (self.Ql[1:] - self.Ql[:-1]))
        F1[0] = Q_in_face
        F1[-1] = Q_out_face
        u_in = Q_in_face / max(self.Al[0], 1e-6)
        u_out = Q_out_face / max(self.Al[-1], 1e-6)
        F2f[0] = Q_in_face * u_in
        F2f[-1] = Q_out_face * u_out

        Al_new = self.Al - dt / dx * (F1[1:] - F1[:-1])
        Ql_star = self.Ql - dt / dx * (F2f[1:] - F2f[:-1])
        # MINMOD-limited pressure gradient.  The centered difference straddles
        # the filling-bore front: the cell just AHEAD of the front felt the
        # full pressurized head of the cell behind it, accelerated violently,
        # and evacuated the reach ahead of the bore (the chamber drained
        # before the bore arrived).  The limiter keeps the front a material
        # contact: no acceleration from a one-cell pressure wall.
        dpiez = np.zeros(N)
        dl = piez[1:-1] - piez[:-2]
        dr = piez[2:] - piez[1:-1]
        dpiez[1:-1] = _minmod(dl, dr) / dx
        dpiez[0] = min(max(piez[1] - piez[0], -0.5), 0.5) / dx
        dpiez[-1] = min(max(piez[-1] - piez[-2], -0.5), 0.5) / dx
        Ql_star += dt * (-G * np.maximum(Al_new, 0.0) * dpiez)
        # semi-implicit Manning friction on the TRUE circular hydraulic radius
        Rh = np.interp(np.maximum(Al_new, 1e-6), self.a_lut, self.rh_lut)
        un = Ql_star / np.maximum(Al_new, 1e-4 * self.A)
        lam = G * (self.n ** 2) * np.abs(un) / np.maximum(Rh ** (4.0 / 3.0), 1e-6)
        self.Ql = Ql_star / (1.0 + dt * lam)
        self.Al = np.maximum(Al_new, 0.0)


def run_case(case: LiuCase, verbose: bool = True) -> dict:
    """Composite-domain formulation (2026-07-08): the upstream pipe, the
    junction chamber, and the downstream pipe are ONE continuous PDE domain
    with per-cell section geometry; the riser is the volume continuation of
    the chamber above its lid (pure hydrostatic surge-tank compliance, no
    algebraic face laws anywhere).  Waves therefore transmit through the
    chamber in both directions and the chamber-pipe oscillation damps
    NATURALLY by radiation into the pipes + section-change losses -- the
    quasi-steady orifice faces of the lumped-node version reflected that
    energy and rang a ~9 s seiche whose troughs pulled the chamber nearly
    dry (three face-level damping hacks failed; see the note in the lumped
    implementation below, kept as run_case_lumped for A/B)."""
    return _run_case_composite(case, verbose)


def _run_case_composite(case: LiuCase, verbose: bool = True) -> dict:
    dx = case.dx
    Nu = int(round(case.Lu / dx))
    Nc = max(int(round(case.Lc / dx)), 3)
    Nd = int(round(case.Ld / dx))
    N = Nu + Nc + Nd
    ic0, ic1 = Nu, Nu + Nc
    x = (np.arange(N) + 0.5) * dx

    # ---- per-cell section geometry ----
    A_full = np.empty(N)
    zinv = np.zeros(N)
    A_full[:ic0] = case.Au
    A_full[ic0:ic1] = case.Aplan / case.Lc * case.Hc * case.Lc / case.Hc  # placeholder
    A_full[ic0:ic1] = case.Wc * case.Hc
    A_full[ic1:] = case.Ad
    zinv[:ic0] = case.drop + case.slope_u * (case.Lu - x[:ic0])
    a_u, h_u, rh_u = _make_luts(case.Du)
    a_d, h_d, rh_d = _make_luts(case.Dd)

    def depth_of(Al):
        h = np.empty(N)
        h[:ic0] = np.interp(Al[:ic0], a_u, h_u)
        h[ic0:ic1] = Al[ic0:ic1] / case.Wc
        h[ic1:] = np.interp(Al[ic1:], a_d, h_d)
        return h

    def rh_of(Al):
        r = np.empty(N)
        r[:ic0] = np.interp(Al[:ic0], a_u, rh_u)
        hc = Al[ic0:ic1] / case.Wc
        r[ic0:ic1] = Al[ic0:ic1] / np.maximum(case.Wc + 2.0 * hc, 1e-6)
        r[ic1:] = np.interp(Al[ic1:], a_d, rh_d)
        return r

    # ---- initial state: steady Q0 ----
    def manning_area(Q, D, slope, n):
        A_f = 0.25 * math.pi * D * D
        lo, hi = 1e-5, 0.95 * A_f
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            hh = _circ_depth_from_area(mid, D)
            th = 2.0 * math.acos(max(min(1.0 - 2.0 * hh / D, 1.0), -1.0))
            Rh = mid / max(0.5 * th * D, 1e-9)
            q = mid * Rh ** (2.0 / 3.0) * math.sqrt(slope) / n
            lo, hi = (mid, hi) if q < Q else (lo, mid)
        return mid

    Al = np.empty(N)
    Ql = np.full(N, case.Q0)
    Al[:ic0] = manning_area(case.Q0, case.Du, case.slope_u, case.n_mann)
    Al[ic0:ic1] = 0.10 * case.Wc
    hd0 = case.hd0_frac * case.Dd
    Al[ic1:] = np.interp(hd0, h_d, a_d)
    if case.downstream_full:
        # Series-B initial condition: downstream pipe full at Q0, chamber
        # stage standing near the downstream crown (the warm-up settles the
        # exact steady stage against the overflow-weir-equivalent tailwater)
        Al[ic1:] = A_full[ic1:]
        Al[ic0:ic1] = min(case.Dd + 0.02, 0.95 * case.Hc) * case.Wc
    Mg = RHO_ATM * np.maximum(A_full - Al, 0.0) * dx
    gas_vented = 0.0
    hr = 0.0
    wr = 0.0                 # riser column velocity [m/s] (rigid-column ODE)
    wr_eject = 0.0           # max column-top velocity while erupting
    geyser_hit = False
    a2 = case.a_wh ** 2

    rec = dict(t=[], PT1=[], PT2=[], PT3=[], hr=[], S=[], Qin=[], Qout=[],
               Qjin=[], Qjout=[],
               frames_t=[], frames_up_h=[], frames_dn_h=[], frames_S=[],
               frames_up_u=[], frames_dn_u=[], frames_ch_h=[],
               up_x=x[:ic0].copy(), dn_x=(x[ic1:] - x[ic1] + 0.5 * dx).copy(),
               ch_x=(x[ic0:ic1] - ic0 * dx).copy(),
               up_zinv=zinv[:ic0].copy(), dn_zinv=zinv[ic1:].copy())
    out_dt = 0.01
    frame_dt = 0.10
    next_out = 0.0
    next_frame = 0.0
    t = 0.0
    step = 0
    S = 0.10
    S_max = 0.0
    vol_in = 0.0
    vol_out = 0.0
    overflow_vol = 0.0
    mass0 = float(np.sum(Al)) * dx + hr * case.Ar
    t_total = case.t_warmup + case.t_end
    t_ramp0 = case.t_warmup + case.t_step
    Qface = [case.Q0, case.Q0]          # junction-face momentum states [m3/s]
    Qout_f = case.Q0                    # outlet-face momentum state [m3/s]

    # distributed local losses at the section changes (entry expansion with
    # the invert drop; exit contraction + turn) -- energy sinks the plunging
    # and mixing provide physically
    K_cell = np.zeros(N)
    K_cell[ic0] = case.K_in
    K_cell[ic1] = case.K_out if ic1 < N else 0.0
    icr_tap = (ic0 + ic1) // 2          # lid-tap cell (riser base)

    while t < t_total - 1e-12:
        h = depth_of(Al)
        frac = Al / A_full
        # ---- pressures ----
        # Chamber-connected gas regions are VENTED (same policy as the A2
        # copy): the chamber air escapes through the open 0.06 m lid tap.  A
        # sealed-pocket variant (chamber air as an EOS pocket venting at the
        # choked-tap rate) was tried and REVERTED: the pocket pressurized
        # the chamber while the bore was still filling the upstream pipe,
        # launched the riser column early, and the standing column then
        # blocked the tap's venting path -- a positive feedback that erupted
        # the case ~1 s before the bore even reached the chamber.  Resolving
        # the mixture compression the paper describes ("mixed with the air
        # initially in the chamber, filled and pressurized the chamber")
        # needs a dispersed-phase closure in the chamber, not a single
        # sealed pocket; with the vented policy the slam is carried by the
        # liquid inertia alone, which under-predicts the acoustic peak (a
        # declared limitation) but keeps the event sequence physical.
        P = RHO_L * G * h
        gassy = frac < 0.97
        for (i0, i1) in _regions(gassy):
            touches_chamber = not (i1 <= ic0 or i0 >= ic1)
            if touches_chamber or i0 == 0 or i1 == N:
                continue                                  # vented region
            m_reg = float(np.sum(Mg[i0:i1]))
            v_reg = float(np.sum(np.maximum(A_full[i0:i1] - Al[i0:i1], 1e-6)) * dx)
            P_reg = m_reg * R_GAS * T_GAS / max(v_reg, 1e-12) - P_ATM
            P[i0:i1] += max(P_reg, -0.5 * P_ATM)
        # CASE-B3 copy: the chamber cells carry the SAME stiff elastic overlay
        # as the pipe cells.  The slam peak at bore arrival (PT2/PT3 ~ 55 kPa,
        # the primary observable of this branch) is a compressibility
        # transient; the A2 copy replaced it by a column-height compliance
        # because its branch never pressurizes hard.  The standing-column
        # equilibrium is NOT prescribed: it emerges from the riser ODE below
        # (dwr/dt = 0 <=> lid head = hr).
        w_full = np.clip((frac - 0.97) / 0.03, 0.0, 1.0)
        P = P + w_full * RHO_L * a2 * np.clip(frac - 1.0, -0.02, 0.05)
        piez = P / (RHO_L * G) + zinv

        # ---- timestep ----
        u = Ql / np.maximum(Al, 1e-4 * A_full)
        c = np.sqrt(G * np.maximum(h, 1e-3))
        c = np.where((w_full > 0.5), case.a_wh, c)
        smax = float(np.max(np.abs(u) + c))
        dt = min(case.cfl * dx / max(smax, 1e-6), 1.0e-3, t_total - t)

        # ---- boundaries ----
        s = min(max((t - t_ramp0) / max(case.Tv, 1e-6), 0.0), 1.0)
        Qin = case.Q0 + (case.Q1 - case.Q0) * s
        if case.downstream_full:
            # Series-B outlet: SUBMERGED outfall against the fixed tank level
            # H_tail.  The face carries its own momentum state (same short-
            # connector closure as the internal junction faces): at steady
            # state it enforces piez[-1] - H_tail = (1 + K_exit) u^2/2g, and
            # in the transient it transmits (rather than rectifies) the slam
            # wave.  An instantaneous orifice face here would chatter against
            # the stiff elastic end cell exactly as the internal faces did.
            dpz_o = piez[-1] - case.H_tail
            a_o = max(Al[-1], 0.05 * A_full[-1])
            u_prev_o = Qout_f / a_o
            Qout_f = ((Qout_f + dt * G * a_o * dpz_o / dx)
                      / (1.0 + dt * (1.0 + case.K_exit) * abs(u_prev_o) / (2.0 * dx)))
            Qout_f = max(min(Qout_f, U_FACE_MAX * a_o), -U_FACE_MAX * a_o)
            Qout = min(Qout_f, 0.9 * Al[-1] * dx / dt)
        else:
            head_w = max(h[-1] - case.weir_crest, 0.0)
            Qout = case.weir_Cd * case.weir_b * math.sqrt(2.0 * G) * head_w ** 1.5
            Qout = min(Qout, 0.9 * Al[-1] * dx / dt)

        # ---- Rusanov fluxes over the whole composite domain ----
        F1 = np.empty(N + 1)
        F2f = np.empty(N + 1)
        sfc = np.maximum(np.abs(u[:-1]) + c[:-1], np.abs(u[1:]) + c[1:])
        F1[1:-1] = (0.5 * (Ql[:-1] + Ql[1:])
                    - 0.5 * sfc * (Al[1:] - Al[:-1]))
        F2f[1:-1] = (0.5 * (Ql[:-1] * u[:-1] + Ql[1:] * u[1:])
                     - 0.5 * sfc * (Ql[1:] - Ql[:-1]))
        # The two section-change faces are INTERNAL JUNCTION faces: the area
        # jump is geometry, not a wave state.  Each face carries its own
        # MOMENTUM STATE (a short connector of length dx with the local-loss
        # closure): dQf/dt = g a_f dpiez/dx - (1+K)|u_f| Q_f/(2 dx), advanced
        # semi-implicitly.  At steady state this enforces the energy balance
        # dpiez = (1+K) u^2/2g (velocity head + loss), it transmits waves in
        # both directions, and it cannot flip-flop (an INSTANTANEOUS orifice
        # face rectified its own chatter into ~100 L/s of forward flow against
        # an adverse 0.07 m head jump; a plain donor-cell face ignored the
        # head balance altogether).
        for f, K_f, qi in ((ic0, case.K_in, 0), (ic1, case.K_out, 1)):
            dpz_f = piez[f - 1] - piez[f]
            a_f = max(min(Al[f - 1], Al[f]), 1e-4 * min(A_full[f - 1], A_full[f]))
            u_prev = Qface[qi] / a_f
            Qface[qi] = ((Qface[qi] + dt * G * a_f * dpz_f / dx)
                         / (1.0 + dt * (1.0 + K_f) * abs(u_prev) / (2.0 * dx)))
            Qface[qi] = max(min(Qface[qi], U_FACE_MAX * a_f), -U_FACE_MAX * a_f)
            F1[f] = Qface[qi]
            F2f[f] = Qface[qi] * (u[f - 1] if Qface[qi] >= 0.0 else u[f])
        F1[0] = Qin
        F2f[0] = Qin * Qin / max(Al[0], 1e-6)
        F1[-1] = Qout
        F2f[-1] = Qout * max(u[-1], 0.0)

        Al_new = Al - dt / dx * (F1[1:] - F1[:-1])
        Ql_star = Ql - dt / dx * (F2f[1:] - F2f[:-1])
        # Pressure gradient: CENTERED everywhere except at moving FRONTS.
        # The minmod limiter is needed at a filling-bore front (the centered
        # difference straddles the front and catapults the shallow cell ahead)
        # -- but applied globally it also zeroes the gradient at the PERSISTENT
        # geometry interfaces (chamber<->pipe piez jumps have opposite one-
        # sided slopes, minmod = 0), so the chamber never felt the downstream
        # backwater and settled 0.2 m too low.  Limit only where a genuine
        # front exists (deep/shallow neighbor contrast).
        dpz = np.zeros(N)
        dl = piez[1:-1] - piez[:-2]
        dr = piez[2:] - piez[1:-1]
        cen = 0.5 * (dl + dr)
        lim = _minmod(dl, dr)
        hmid = np.maximum(h[1:-1], 1e-4)
        front = (h[2:] < 0.35 * hmid) | (h[:-2] < 0.35 * hmid)
        dpz[1:-1] = np.where(front, lim, cen) / dx
        dpz[0] = min(max(piez[1] - piez[0], -0.5), 0.5) / dx
        dpz[-1] = min(max(piez[-1] - piez[-2], -0.5), 0.5) / dx
        Ql_star += dt * (-G * np.maximum(Al_new, 0.0) * dpz)
        # semi-implicit friction + local section-change losses
        Rh = rh_of(np.maximum(Al_new, 1e-6))
        un = Ql_star / np.maximum(Al_new, 1e-4 * A_full)
        lam = (G * (case.n_mann ** 2) * np.abs(un)
               / np.maximum(Rh ** (4.0 / 3.0), 1e-6)
               + K_cell * np.abs(un) / (2.0 * dx))
        Ql = Ql_star / (1.0 + dt * lam)
        Al = np.maximum(Al_new, 0.0)

        # ---- riser: RIGID-COLUMN ODE at the lid tap (chamber center) ----
        # The eruption velocity and the slam rebound ARE the observables of
        # this branch, so the A2 copy's quasi-static column (Torricelli
        # drain + recording lag as an inertia surrogate) is replaced by the
        # actual momentum balance of the standing column:
        #   (hr + L_eq) dwr/dt = g (H_lid - hr) - K_riser wr|wr| / 2
        #   dhr/dt = wr
        # where H_lid is the piezometric head at the tap cell ABOVE THE LID
        # (from the same cell pressure the PDE uses, elastic overlay
        # included), and the column exchanges volume with the tap cell so
        # that chamber decompression by column launch is fed back to the
        # domain.  Equilibrium (dwr/dt = 0, wr = 0) is H_lid = hr: the
        # standing column, emergent, not prescribed.
        icr = icr_tap
        # head at the LID (the column base): when the tap cell is liquid-full
        # this is the invert-gauge head minus the lid height; when a sealed
        # gas pocket sits under the lid it is the pocket gauge head (the
        # invert pressure minus the local hydrostatic share) -- the pocket
        # pushing the mixture into the riser is the paper's own eruption
        # mechanism.  No positivity clamp: a sub-atmospheric rebound pulls
        # the column back down, which is the measured end of the geyser.
        h_tap = float(P[icr]) / (RHO_L * G)
        sealed = frac[icr] > 0.985
        H_lid = (h_tap - case.Hc) if sealed else (h_tap - float(h[icr]))
        if t < t_ramp0:
            # warm-up: the documented initial state has an EMPTY riser (the
            # chamber pocket breathes through the open tap).  Transient
            # pocket squeezes during the settling would otherwise launch a
            # spurious column that then blocks the tap venting path.
            hr = 0.0
            wr = 0.0
        elif sealed or hr > 0.0 or wr != 0.0 or H_lid > 1e-6:
            acc = (G * (H_lid - hr)
                   - case.K_riser * wr * abs(wr) / 2.0) / (hr + case.L_eq_riser)
            wr += dt * acc
            dvol = wr * case.Ar * dt              # column volume drawn from tap
            if dvol >= 0.0:
                avail = max(Al[icr] - 0.05 * A_full[icr], 0.0) * dx
                dvol = min(dvol, avail)
            else:
                # bounded by BOTH the cell's room and the column's holdings
                # (otherwise hr overshoots negative and its reset to zero
                # creates volume out of nothing)
                room = max(1.05 * A_full[icr] - Al[icr], 0.0) * dx
                dvol = -min(-dvol, room, max(hr, 0.0) * case.Ar)
            Al[icr] -= dvol / dx
            hr += dvol / case.Ar
            if hr <= 0.0:
                hr = 0.0
                wr = min(wr, 0.0) if H_lid <= 0.0 else wr
                if not sealed:
                    wr = 0.0
        if hr >= case.Hr:
            # column top at the riser top: mixture ejected while wr > 0
            if wr > 0.0:
                geyser_hit = True
                wr_eject = max(wr_eject, wr)
                spilled = (hr - case.Hr) * case.Ar
                overflow_vol += spilled
                hr = case.Hr
            else:
                hr = min(hr, case.Hr)

        # ---- conservative gas bookkeeping (same policy as the reaches) ----
        void = np.maximum(A_full - Al, 0.0) * dx
        gassy = (Al / A_full) < 0.97
        Mg2 = Mg.copy()
        g_idx = np.where(gassy)[0]
        stranded = np.where(~gassy, Mg2, 0.0)
        m_str = float(np.sum(stranded))
        if m_str > 1e-15:
            Mg2 = np.where(gassy, Mg2, 0.0)
            if g_idx.size:
                for i in np.where(stranded > 1e-15)[0]:
                    j = int(g_idx[np.argmin(np.abs(g_idx - i))])
                    Mg2[j] += stranded[i]
            else:
                gas_vented += m_str
        for (i0, i1) in _regions(gassy):
            touches_chamber = not (i1 <= ic0 or i0 >= ic1)
            if touches_chamber or i0 == 0 or i1 == N:
                gas_vented += float(np.sum(Mg2[i0:i1]))
                Mg2[i0:i1] = RHO_ATM * void[i0:i1]
                gas_vented -= float(np.sum(Mg2[i0:i1]))
            else:
                m_reg = float(np.sum(Mg2[i0:i1]))
                v_reg = float(np.sum(void[i0:i1]))
                Mg2[i0:i1] = (m_reg * void[i0:i1] / max(v_reg, 1e-12)
                              if v_reg > 0 else 0.0)
        Mg = np.where(gassy, Mg2, 0.0)

        # ---- stage bookkeeping ----
        h_now = depth_of(Al)
        frac_ch = float(np.mean(Al[ic0:ic1] / A_full[ic0:ic1]))
        S = float(np.mean(h_now[ic0:ic1])) if frac_ch < 0.995 else case.Hc + hr
        if t >= case.t_warmup:
            S_max = max(S_max, S)
        vol_in += Qin * dt
        vol_out += Qout * dt
        t += dt
        step += 1

        # ---- records ----
        t_rep = t - case.t_warmup
        if t_rep >= next_out - 1e-12:
            next_out += out_dt
            rec["t"].append(t_rep)
            # PT3 instrument closure: the tap sits on the chamber front wall in
            # the path of the inflow jet; once the pipe end runs full the jet
            # spans the chamber and the tap reads a stagnation share of its
            # dynamic head on top of the hydrostatic stage (the paper itself
            # attributes the PT3-PT2 anomaly to "the effect of dynamic
            # pressure").  Inactive at Q0 (open-channel jet plunges down the
            # drop below the tap's horizontal reach).
            u_end = float(Ql[ic0 - 1] / max(Al[ic0 - 1], 1e-4 * A_full[ic0 - 1]))
            w_end = min(max((Al[ic0 - 1] / A_full[ic0 - 1] - 0.90) / 0.08, 0.0), 1.0)
            p_dyn = w_end * case.Cp_jet * 0.5 * RHO_L * u_end * u_end
            # CASE-B3 copy: PT2/PT3 read the TAP-CELL PDE pressure directly
            # (elastic slam transient included), not a reconstructed static
            # stack -- the slam peak and the rebound trough are the
            # observables of this branch.  PT1 keeps the hydrostatic column
            # share above its elevation (the mist/aeration spikes the rig
            # transducer sees during ejection are not resolved).
            h_tap_r = float(P[icr_tap]) / (RHO_L * G)
            sealed_r = frac[icr_tap] > 0.985
            # PT2 reads the lid pressure: liquid-full tap cell -> invert head
            # minus lid height; sealed pocket under the lid -> the pocket
            # gauge head.  No positivity clamp: the rebound trough (PT2 to
            # -20 kPa in the experiment) is a genuine sub-atmospheric
            # transient of the sealed chamber and must be readable.
            pt2_head = (h_tap_r - case.Hc) if sealed_r \
                else (h_tap_r - float(h_now[icr_tap]))
            rec["PT1"].append(RHO_L * G * max(hr - 0.80, 0.0) / 1000.0)
            rec["PT2"].append(RHO_L * G * pt2_head / 1000.0)
            rec["PT3"].append((RHO_L * G * h_tap_r + p_dyn) / 1000.0)
            rec["hr"].append(hr)
            rec["S"].append(S)
            rec["Qin"].append(Qin)
            rec["Qout"].append(Qout)
            rec["Qjin"].append(float(F1[ic0]))
            rec["Qjout"].append(float(F1[ic1]))
        if t_rep >= next_frame - 1e-12:
            next_frame += frame_dt
            rec["frames_t"].append(t_rep)
            rec["frames_up_h"].append(h_now[:ic0].copy())
            rec["frames_dn_h"].append(h_now[ic1:].copy())
            rec["frames_S"].append(S)
            # per-cell chamber free surface (NOT the flat mean stage): during
            # the bore arrival the surface tilts and crests locally pin against
            # the lid; a flat-S rendering hid exactly the mechanism that feeds
            # the riser, which read as "riser wet while the chamber is not full"
            rec["frames_ch_h"].append(h_now[ic0:ic1].copy())
            uu = Ql / np.maximum(Al, 1e-4 * A_full)
            rec["frames_up_u"].append(uu[:ic0].copy())
            rec["frames_dn_u"].append(uu[ic1:].copy())
        if verbose and step % 20000 == 0:
            print(f"  t={t:6.2f}s  S={S:.3f}  hr={hr:.3f}  Qin={Qin*1e3:5.1f}"
                  f"  Qout={Qout*1e3:6.1f} L/s", flush=True)

    mass_final = float(np.sum(Al)) * dx + hr * case.Ar
    rec["S_max"] = S_max
    rec["hr_max"] = float(np.max(np.asarray(rec["hr"]))) if rec["hr"] else 0.0
    rec["geyser"] = bool(geyser_hit or S_max >= case.Hc + case.Hr - 1e-6)
    rec["wr_eject"] = float(wr_eject)
    # ballistic height of the mixture leaving the riser top, measured from
    # the chamber lid (the datum of the paper's jetting height h)
    rec["h_jet"] = float(case.Hr + wr_eject ** 2 / (2.0 * G)) if geyser_hit \
        else float(np.max(np.asarray(rec["hr"]))) if rec["hr"] else 0.0
    rec["overflow_vol"] = overflow_vol
    rec["mass_error"] = mass_final - mass0 - (vol_in - vol_out) + overflow_vol
    rec["gas_vented_kg"] = float(gas_vented)
    return rec


def run_case_lumped(case: LiuCase, verbose: bool = True) -> dict:
    dx = case.dx
    # upstream reach: invert falls toward the chamber, ends case.drop above it
    up = PipeReach(case.Lu, case.Du, dx,
                   lambda x: case.drop + case.slope_u * (case.Lu - x),
                   case.n_mann, case.a_wh)
    dn = PipeReach(case.Ld, case.Dd, dx, lambda x: np.zeros_like(x),
                   case.n_mann, case.a_wh)

    # ---------------- initial state: steady Q0, open channel ----------------
    def manning_area(Q, D, slope, n):
        A_full = 0.25 * math.pi * D * D
        lo, hi = 1e-5, 0.95 * A_full
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            h = _circ_depth_from_area(mid, D)
            th = 2.0 * math.acos(max(min(1.0 - 2.0 * h / D, 1.0), -1.0))
            Rh = mid / max(0.5 * th * D, 1e-9)
            q = mid * Rh ** (2.0 / 3.0) * math.sqrt(slope) / n
            if q < Q:
                lo = mid
            else:
                hi = mid
        return mid

    al_u0 = manning_area(case.Q0, case.Du, case.slope_u, case.n_mann)
    up.Al[:] = al_u0
    up.Ql[:] = case.Q0
    hd0 = case.hd0_frac * case.Dd
    dn.Al[:] = np.interp(hd0, dn.h_lut, dn.a_lut)
    dn.Ql[:] = case.Q0
    up.Mg = RHO_ATM * np.maximum(up.A - up.Al, 0.0) * dx
    dn.Mg = RHO_ATM * np.maximum(dn.A - dn.Al, 0.0) * dx

    # chamber + riser storage node (surge tank, volume-conservative):
    #   stage S <= Hc lives on the chamber plan area; the surplus above full
    #   lives IN THE RISER on its bore area (hr = E/Ar).  The RECORDED riser
    #   column lags the static column with a first-order time constant
    #   tau_riser (entrance inertia + orifice loss surrogate) -- the raw
    #   static column tracks the bore surge instantly and overshoots what the
    #   camera sees (mixture front).
    S = 0.10                       # reported initial chamber depth [m]
    V_ch = case.Aplan * S
    V_full = case.Aplan * case.Hc
    hr = 0.0                       # static (volume) riser column
    hr_rec = 0.0                   # recorded (lagged) riser column
    tau_riser = 0.25               # [s]

    rec = dict(t=[], PT1=[], PT2=[], PT3=[], hr=[], S=[], Qin=[], Qout=[],
               Qjin=[], Qjout=[],
               frames_t=[], frames_up_h=[], frames_dn_h=[], frames_S=[],
               up_x=up.x, dn_x=dn.x, up_zinv=up.zinv, dn_zinv=dn.zinv)
    out_dt = 0.01
    next_out = 0.0
    frame_dt = 0.10
    next_frame = 0.0
    t = 0.0
    step = 0
    S_max = 0.0
    vol_in = 0.0
    vol_out = 0.0
    V_node = V_ch
    mass0 = (float(np.sum(up.Al)) + float(np.sum(dn.Al))) * dx + V_node
    overflow_vol = 0.0
    Qj_in = case.Q0
    Qj_out = case.Q0
    tau_face = 0.05                     # junction-face relaxation time [s]
    S_prev = S
    dSdt_f = 0.0                        # filtered stage rate [m/s]

    t_total = case.t_warmup + case.t_end
    t_ramp0 = case.t_warmup + case.t_step
    while t < t_total - 1e-12:
        dt = case.cfl * dx / max(up.wave_speed(), dn.wave_speed(), 1e-6)
        dt = min(dt, 1.0e-3, t_total - t)

        # ---------------- boundary/interface fluxes ----------------
        s = min(max((t - t_ramp0) / max(case.Tv, 1e-6), 0.0), 1.0)
        Qin = case.Q0 + (case.Q1 - case.Q0) * s

        # NOTE (third damping attempt, tried and REVERTED 2026-07-08): scaling
        # both face loss coefficients with the filtered |dS/dt| ("aeration
        # mixing dissipation") trapped the arrival surge inside the node --
        # the amplified K throttled the OUTFLOW exactly during the fill phase
        # and the case false-geysered.  Together with the two reverted flux
        # hacks (slow outflow face; net-exchange brake) this establishes that
        # the ~9 s node-pipe seiche cannot be damped at the quasi-steady
        # orifice-face level: it needs the companion framework's two-way
        # Riemann junction coupling.  The seiche remains a documented artifact
        # (the measured chamber never nears dry; ours dips to ~0.06 m in the
        # troughs), and the converged steady state is unaffected.
        dSdt_f += ((S - S_prev) / max(dt, 1e-9) - dSdt_f) * min(dt / 0.3, 1.0)
        S_prev = S

        # upstream pipe -> node.  While the chamber stage is below the pipe
        # outlet the sloped pipe discharges as a free jet (it simply delivers
        # its own end-cell flux); once the stage submerges the outlet the face
        # follows the drowned-orifice head balance.  The blend weight w_sub
        # uses the stage over the pipe invert; the face state is RELAXED over
        # tau_face against step-to-step chatter, and the free-jet share is
        # bounded by the physical pipe conveyance (3 m/s) so a pressurized
        # end cell cannot pump the node unphysically.
        P_up = up.pressures(vent_left=True, vent_right=False)
        piez_up = P_up / (RHO_L * G) + up.zinv
        # face head = few-cell average: the single end cell's stiff elastic
        # overlay swings +-meters on one overfill step and the face chokes /
        # slams in a ~15 s limit cycle (node overfilled to geyser in A2)
        pz_up_face = float(np.mean(piez_up[-4:]))
        zu_in = case.drop                       # pipe invert at the chamber wall
        w_sub = min(max((S - zu_in) / case.Du, 0.0), 1.0)
        Q_free = float(np.clip(up.Ql[-1], 0.0, U_FACE_MAX * up.A))
        drive1 = float(np.clip(pz_up_face - S, -0.6, 0.6))
        u_eq1 = math.copysign(
            math.sqrt(2.0 * G * abs(drive1) / max(case.K_in, 0.3)), drive1)
        a_face1 = up.Al[-1] if drive1 >= 0 else min(up.A, up.Al[-1] + 0.2 * up.A)
        Q_drown = float(np.clip(u_eq1 * max(a_face1, 0.0),
                                -U_FACE_MAX * up.A, U_FACE_MAX * up.A))
        # PRESSURIZED-END override: once the bore fills the pipe end, the
        # outlet discharges as an orifice driven by its own pressure head no
        # matter how low the chamber stage is.  Without this the system
        # deadlocked after the ramp: the momentum-carried Q_free jammed to
        # ~2 L/s (one-sided overfill gradient decelerates the end cell), the
        # chamber drained below the invert, w_sub stayed 0, and the free-jet
        # branch kept taking only the jammed flux -- Qin 100 L/s piled into
        # the pipe while the chamber emptied.
        w_fend = min(max((up.Al[-1] / up.A - 0.90) / 0.08, 0.0), 1.0)
        drive_orif = float(np.clip(pz_up_face - max(S, zu_in), -0.6, 0.6))
        Q_orif = (up.A * math.sqrt(2.0 * G * max(drive_orif, 0.0)
                                   / max(case.K_in, 0.3))
                  if drive_orif > 0.0 else 0.0)
        Q_orif = min(Q_orif, U_FACE_MAX * up.A)
        Qj_eq1 = ((1.0 - w_fend) * ((1.0 - w_sub) * Q_free + w_sub * Q_drown)
                  + w_fend * max(Q_orif, Q_drown))
        Qj_in += (Qj_eq1 - Qj_in) * min(dt / tau_face, 1.0)
        Qj_in = max(min(Qj_in, 0.9 * up.Al[-1] * dx / dt), -0.9 * V_node / dt)

        # node -> downstream pipe (same relaxed face state).  In open-channel
        # through-flow (stage below the downstream crown) the chamber is just
        # a wide reach and the exit is nearly a pass-through (K ~ 0.2); the
        # full contraction + turn loss applies once the exit is submerged.
        P_dn = dn.pressures(vent_left=False, vent_right=True)
        piez_dn = P_dn / (RHO_L * G) + dn.zinv
        pz_dn_face = float(np.mean(piez_dn[:4]))
        drive2 = float(np.clip(S - pz_dn_face, -0.6, 0.6))
        # exit submergence: the pass-through (low-loss) branch applies only
        # while BOTH the chamber stage and the downstream head run as open
        # channel; once either side reaches the crown the exit is a submerged
        # contraction (keying on S alone let the post-surge drawdown flip back
        # to K=0.2 and the node rang a deep ~10 s cycle: S crashed to 0.07
        # and PT3 dipped to zero mid-window)
        sub_ref = max(S, pz_dn_face)
        w_sub2 = min(max((sub_ref - case.Dd) / (0.5 * case.Dd), 0.0), 1.0)
        K_out_eff = 0.2 + (case.K_out - 0.2) * w_sub2
        u_eq2 = math.copysign(
            math.sqrt(2.0 * G * abs(drive2) / max(K_out_eff, 0.15)), drive2)
        a_face2 = min(dn.A, max(dn.Al[0], 0.3 * dn.A)) if drive2 >= 0 else dn.Al[0]
        Qj_eq2 = float(np.clip(u_eq2 * a_face2, -U_FACE_MAX * dn.A, U_FACE_MAX * dn.A))
        Qj_out += (Qj_eq2 - Qj_out) * min(dt / tau_face, 1.0)
        Qj_out = min(max(Qj_out, -0.9 * dn.Al[0] * dx / dt), 0.9 * V_node / dt)

        # downstream weir outflow
        h_end = float(dn.depth()[-1])
        head_w = max(h_end - case.weir_crest, 0.0)
        Qout = case.weir_Cd * case.weir_b * math.sqrt(2.0 * G) * head_w ** 1.5
        Qout = min(Qout, 0.9 * dn.Al[-1] * dx / dt)

        # ---------------- advance reaches ----------------
        up.step(dt, Qin, Qj_in, piez_up)
        dn.step(dt, Qj_out, Qout, piez_dn)
        up.gas_update(vent_left=True, vent_right=False)
        dn.gas_update(vent_left=False, vent_right=True)

        # ---------------- node update (volume-conservative surge tank) ------
        V_ch += (Qj_in - Qj_out) * dt
        V_ch = max(V_ch, 0.0)
        E = max(V_ch - V_full, 0.0)               # surplus above the lid
        hr = E / case.Ar                          # static riser column
        if hr > case.Hr:
            # riser overflow: geyser discharge (recorded, leaves the system)
            spilled = (hr - case.Hr) * case.Ar
            overflow_vol += spilled
            V_ch -= spilled
            hr = case.Hr
        hr_rec += (hr - hr_rec) * min(dt / tau_riser, 1.0)
        # stage for pressure records / junction drives
        S = V_ch / case.Aplan if V_ch <= V_full else case.Hc + hr
        if t >= case.t_warmup:
            S_max = max(S_max, S)
        V_node = V_ch                             # for the face-flux caps

        vol_in += Qin * dt
        vol_out += Qout * dt
        t += dt
        step += 1

        # ---------------- records (t reported relative to the ramp start) ----
        t_rep = t - case.t_warmup
        if t_rep >= next_out - 1e-12:
            next_out += out_dt
            rec["t"].append(t_rep)
            # PT3 is quoted by the paper as the chamber water depth (their
            # initial 0.99 kPa "indicated a water depth of 0.10 m"), so the
            # recorded head is referenced to the chamber bottom
            rec["PT1"].append(RHO_L * G * max(hr_rec - 0.80, 0.0) / 1000.0)
            rec["PT2"].append(RHO_L * G * hr_rec / 1000.0)
            rec["PT3"].append(RHO_L * G * max(min(S, case.Hc) + hr_rec, 0.0) / 1000.0)
            rec["hr"].append(hr_rec)
            rec["S"].append(S)
            rec["Qin"].append(Qin)
            rec["Qout"].append(Qout)
            rec["Qjin"].append(Qj_in)
            rec["Qjout"].append(Qj_out)
        if t_rep >= next_frame - 1e-12:
            next_frame += frame_dt
            rec["frames_t"].append(t_rep)
            rec["frames_up_h"].append(up.depth().copy())
            rec["frames_dn_h"].append(dn.depth().copy())
            rec["frames_S"].append(S)
            rec.setdefault("frames_up_u", []).append(
                (up.Ql / np.maximum(up.Al, 1e-4 * up.A)).copy())
            rec.setdefault("frames_dn_u", []).append(
                (dn.Ql / np.maximum(dn.Al, 1e-4 * dn.A)).copy())
        if os.environ.get("LIU_DEBUG") and step % 200 == 0:
            print(f"[dbg] t={t:6.3f} Ql_end={up.Ql[-1]*1e3:7.2f} Qj_in={Qj_in*1e3:7.2f} "
                  f"Al_end={up.Al[-1]/up.A:.3f} S={S:.4f} Qj_out={Qj_out*1e3:7.2f} "
                  f"h_dn0={dn.depth()[0]:.4f} Qout={Qout*1e3:6.2f}", flush=True)
        if verbose and step % 20000 == 0:
            print(f"  t={t:6.2f}s  S={S:.3f}  hr={max(S-case.Hc,0):.3f} "
                  f" Qj_in={Qj_in*1e3:6.1f}  Qj_out={Qj_out*1e3:6.1f} "
                  f" Qout={Qout*1e3:6.1f} L/s", flush=True)

    mass_final = (float(np.sum(up.Al)) + float(np.sum(dn.Al))) * dx + V_node
    rec["S_max"] = S_max
    # camera-comparable column maximum: the RECORDED (inertia-lagged) column,
    # not the instantaneous static surge spike
    rec["hr_max"] = float(np.max(np.asarray(rec["hr"]))) if rec["hr"] else 0.0
    rec["geyser"] = bool(S_max >= case.Hc + case.Hr - 1e-6)
    rec["overflow_vol"] = overflow_vol
    rec["mass_error"] = mass_final - mass0 - (vol_in - vol_out) + overflow_vol
    rec["gas_vented_kg"] = float(up.gas_vented + dn.gas_vented)
    return rec


if __name__ == "__main__":
    case = LiuCase(t_end=7.0, downstream_full=True)
    rec = run_case(case)
    t = np.asarray(rec["t"])
    PT2 = np.asarray(rec["PT2"]); PT3 = np.asarray(rec["PT3"])
    ipk = int(np.argmax(PT2))
    print(json.dumps(dict(
        geyser=rec["geyser"], hr_max=rec["hr_max"],
        wr_eject=rec["wr_eject"], h_jet=rec["h_jet"],
        PT2_peak_kPa=float(PT2[ipk]), t_peak_s=float(t[ipk]),
        PT2_min_kPa=float(np.min(PT2)), PT3_min_kPa=float(np.min(PT3)),
        PT2_final=float(np.mean(PT2[-200:])),
        PT3_final=float(np.mean(PT3[-200:])),
        PT3_initial=float(np.mean(PT3[:50])),
        PT2_initial=float(np.mean(PT2[:50])),
        overflow_L=rec["overflow_vol"] * 1e3,
        mass_error_L=rec["mass_error"] * 1e3,
    ), indent=2))
