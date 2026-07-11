# Paper audit — V&W (2011) Test 1, Case B

This audit was completed before constructing the CFD case.  The primary
authority is Vasconcelos & Wright (2011), hereafter **VW**, in
`references/vasconcelos2011.pdf`.  PDF page numbers below count from the first
page of that file; the printed journal page is also given.  Wright, Lewis &
Vasconcelos (2011), `references/wright2011.pdf`, is a field-scale contextual
paper and does not define this laboratory case.

## Audited case identity

| Item | Adopted value | Source and status |
|---|---:|---|
| Main-pipe inside diameter | 0.094 m | VW PDF p.2, journal p.544; confirmed |
| Upstream chamber length | 0.546 m | VW p.2/544; confirmed |
| Middle pipe length | 2.970 m | VW p.2/544; confirmed |
| Downstream pipe length | 0.490 m | VW p.2/544; confirmed |
| Total horizontal length | 4.006 m | sum of the three reported lengths |
| Valve location | x = 0.546 m | VW p.2/544; confirmed |
| Tower centre | x = 3.516 m | 0.546 + 2.970 m |
| Downstream end | x = 4.006 m, closed | VW pp.2–3/544–545; confirmed |
| Tower inside diameter, \(D_t\) | 0.0127 m | VW Table 1, PDF p.3/journal p.545; confirmed |
| Tower height above pipe crown, \(L\) | 0.610 m | VW p.2/544; confirmed |
| Tower top | open, water may spill | VW pp.2–4/544–546; confirmed |
| Initial air gauge head, \(H_{a0}\) | 0.610 m water | VW Table 1 and centre column of Figs.6/8; confirmed |
| Initial tower water level, \(Y_{fs,0}\) | 0.356 m above crown | VW Table 1 and centre row of Figs.6/8; confirmed |
| Observed branch | geyser in every 12.7 mm-tower test | VW pp.4,8/546,550; confirmed |

The target is the centre panel of VW Figs.6 and 8: \(D_t/D=0.135\),
\(H_{a0}=0.610\) m and \(Y_{fs,0}=0.356\) m.  This agrees with
`config/case.json` and combination 5 in
`data/R227-05_experiment_matrix.csv`.

## 1. Main pipe and end conditions

VW PDF p.2 (journal p.544), apparatus description and Fig.2, reports a
horizontal 94 mm acrylic pipe made of 0.546, 2.970 and 0.490 m sections.  The
upstream 0.546 m section is the pressurised-air chamber.  The far downstream
end is closed.  The resulting model coordinates are x=0 at the upstream end,
valve x=0.546 m, tower x=3.516 m, and closed end x=4.006 m.

**Confirmed.** Coupling bore details, measured roughness and dimensional
tolerances are not reported.  The CFD model uses a smooth circular bore and a
Boolean circular tee.

## 2. Tower location, diameter, height and top

VW pp.2–3/544–545 and Table 1 give a 0.610 m ventilation tower connected
between the middle and downstream sections.  The Case-B bore is 12.7 mm.  The
top is open to atmosphere; water spilling from it is observable and is not a
closed or pressure-capped boundary.

**Confirmed.** The CFD tower is a true circular 12.7 mm bore.  Its physical
rim is 0.610 m above the main-pipe crown, and a separate exterior atmosphere
extends above the rim so the numerical outlet cannot clip the jet.

## 3. Butterfly valve and opening

VW p.2/544 reports a 102 mm butterfly valve at the end of the 0.546 m chamber.
It was initially closed and manually opened in **less than 1 s**.  No disc
angle trace, exact repeat opening times, loss curve, or start/mid/end trigger
definition for \(T_{ref}=0\) is supplied.  VW p.4/546 attributes part of the
repeat timing scatter to manual opening differences.

**Partly confirmed.** The CFD baseline uses a smooth 0.25 s opening of a
finite-resistance valve zone, with fully-open \(K=2\), and explicitly tests
other opening durations.  These are declared engineering assumptions, not
fitted paper inputs.  The valve model only removes resistance; it supplies no
pressure, velocity, or mass.

## 4. Initial air pocket: length, volume, pressure and temperature

VW pp.2–3/544–545 describes emptying the upstream section, injecting air, and
setting its pressure with a differential manometer.  The standard pocket
therefore has inferred length 0.546 m and inferred volume

\[
V_{a0}=\frac{\pi(0.094)^2}{4}(0.546)=3.789\times10^{-3}\ {\rm m^3}.
\]

The Case-B gauge head is 0.610 m, or approximately 5.973 kPa above the
101.325 kPa model atmosphere using \(\rho_w=998.2\) kg/m³ and
\(g=9.81\) m/s².  The manometer precision reported by VW is 0.031 m.
VW also tested smaller pockets down to 25% of this volume but did not retain
volume as a principal variable.

**Pressure and chamber geometry confirmed; temperature unresolved.** No
target-repeat gas temperature, leakage measurement, or pump isolation detail
is reported.  The baseline assumes sealed dry air at 293.15 K.  The paper's
simplified model (VW pp.7–9/549–551) uses isentropic air with
\(\gamma=1.4\); the CFD perfect-gas energy equation allows compression work
instead of imposing an isothermal pocket law.

## 5. Initial main-pipe water and interfaces

VW pp.2–3/544–545 says the apparatus downstream of the closed valve was
entirely filled with water, with care taken to eliminate crown air.  The
upstream section was emptied and pressurised.  Thus the initial horizontal
gas/water discontinuity is at the valve plane x=0.546 m; there is no
part-filled downstream main pipe.

**Confirmed.** The main pipe downstream of the valve is initialised as water,
the chamber as air, and both are at rest.  Initial pressures are gauge-head
based and hydrostatically consistent to the accuracy of `setFields`.

## 6. Initial tower water level

VW Table 1, p.3/545, gives 0.356 m for the centre-row condition.  The absolute
model elevation is pipe radius + level = 0.047 + 0.356 = 0.403 m.

**Confirmed.** \(Y_{fs,0}^*=0.356/0.610=0.58361\).  The approximately 0.82
level in Fig.8 is the later pre-arrival plateau, not the initial condition.

## 7. Pressure transducer and datum

VW p.2/544 identifies an Endevco 8510B-1 piezoresistive transducer, nominal
range 6.9 kPa, 1.07 m downstream of the valve and therefore 1.90 m upstream of
the tower (x=1.616 m).  Initial and final middle-pipe levels were used for
calibration.

**Longitudinal position confirmed; vertical tap datum unresolved.** The paper
does not state the circumferential tap elevation or define a crown/invert
correction.  This CFD case records absolute pressure near the lower pipe wall
at the Case-A reference probe coordinate (x=1.616, y=-0.043, z=0) and converts
that pressure directly to gauge head.  It does not apply the old frozen
one-dimensional model's additional \(-D/L\) correction.  A vertical-datum
uncertainty of up to \(D/L=0.154\) must accompany pressure interpretation.

## 8. Dimensionless definitions and target figures

VW notation, PDF p.12/journal p.554, defines

\[
D_t^*=D_t/D,\quad H^*=H/L,\quad
Y_{fs}^*=Y_{fs}/L,\quad Y_{int}^*=Y_{int}/L,
\]

\[
T_{ref}^*=\frac{T_{ref}}{L/\sqrt{gD_t}}
          =\frac{T_{ref}\sqrt{gD_t}}{L},\qquad
V_{fs,int}^*=\frac{V_{fs,int}}{\sqrt{gD_t}}.
\]

For Case B, \(\sqrt{gD_t}=0.35297\) m/s and
\(L/\sqrt{gD_t}=1.72820\) s.

* **Fig.6**, VW p.6/548, centre panel: pressure \(H^*(T^*)\), three repeats.
  The existing raster extraction gives a pseudo-median plateau 0.758 and
  first pseudo-median \(H^*<0.3\) at \(T^*=4.048\).  Its min/max columns are a
  dark-pixel envelope, not a statistical confidence interval.
* **Fig.8**, VW p.8/550, centre panel: free surface and lower gas-interface
  levels over \(3\leq T^*\leq5\).  Existing digitisation gives first retained
  gas markers at 3.648–3.742 and free-surface top markers at 3.900–3.981.
* **Table 2**, VW p.8/550: for \(D_t^*=0.135\),
  \(V_{fs}^*=0.44\) and \(V_{int}^*=1.43\).

**Important Table-2 qualification:** there is one row per tower diameter but
nine pressure/level combinations per diameter.  The paper does not state a
Case-B-only pooling rule or scatter.  The two velocities are therefore
diameter-level averages, not exact centre-panel measurements.

## 9. Repeat scatter

VW pp.3–4/545–546 states that every condition was repeated at least three
times; Figs.6 and 8 show three repetitions.  The paper provides no standard
deviation or raw tables.  Raster-derived, non-statistical spans are:

* gas first visible in tower: \(T^*=3.648\)–3.742;
* free surface near rim: \(T^*=3.900\)–3.981;
* credible high gas-interface markers: \(T^*=4.091\)–4.169;
* pressure collapse envelope: approximately \(T^*=4.02\)–4.18.

The current `fig8_caseB_levels.csv` row `(3.84687, 0.92000, int)` is isolated
from all three rising-interface tracks and overlaps the free-surface marker
region.  It is retained for provenance but flagged as a probable automated
marker misclassification.  Consequently the legacy “interface to 0.85L at
3.85” number has low confidence; the visible tracks support roughly
4.09–4.17.  No simulation parameter is tuned to either interpretation.

## 10. Experimental geyser criterion

VW p.4/546 defines the competing events: gas reaching the free surface
(\(Y_{int}^*=Y_{fs}^*\)) opens an atmospheric escape path, whereas the free
surface reaching the rim first (\(Y_{fs}^*=1\)) spills water and constitutes a
geyser.  Thus the criterion is

\[
Y_{fs}^*=1\quad\hbox{before}\quad Y_{int}^*=Y_{fs}^*.
\]

VW p.8/550 reports geysers for every \(D_t^*=0.135\) test.  In CFD, the
stronger observable is nonzero water volume above the physical rim; merely
filling the tower is not by itself labelled a resolved external jet.

## Source differences and decisions

1. The 2007 checkpoint lists 0.303/0.305 m as an unresolved low-head
   candidate and explicitly marks itself `simulation_ready: false`.  It does
   not override the primary-paper centre condition \(H_{a0}=0.610\) m.
2. The 2011 main paper prints 0.305 m for its low-head column; the 2007 matrix
   uses 0.303 m.  This difference does not affect Case B.
3. `wright2011.pdf` concerns a 3.7 m field tunnel and 2.4 m manhole.  It
   supports the trapped-air mechanism but contributes no Case-B input.
4. Both 2011 papers state that flooding instability was not observed/relevant
   at this laboratory scale.  The frozen 1-D model's Wallis cap is therefore a
   modelling hypothesis, not an experimental definition, and is not imposed
   in this CFD model.
5. The paper's simplified model uses isentropic air; the frozen 1-D snapshot
   uses an isothermal EOS.  The selected CFD solver uses perfect-gas
   compressibility with an energy equation and compressible liquid, as
   documented in `README.md`.

