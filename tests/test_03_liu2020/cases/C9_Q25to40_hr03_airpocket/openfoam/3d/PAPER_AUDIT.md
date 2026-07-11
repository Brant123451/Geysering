# C9 paper audit for the three-dimensional OpenFOAM model

Primary source: L. Liu, W. Shao, and D. Z. Zhu, “Experimental Study on
Stormwater Geyser in Vertical Shaft above Junction Chamber,” *Journal of
Hydraulic Engineering* 146(2), 04019055 (2020), DOI
10.1061/(ASCE)HY.1943-7900.0001660. Page references below are the numbered
journal/PDF pages in `references/liu2020.pdf`.

## Apparatus and measurements

The laboratory rig is a simplified 1:20 model. The dimensions used here are
reported on pp. 2–3 and in Fig. 2 (p. 3):

| component | paper value | model coordinate convention |
|---|---:|---|
| upstream acrylic pipe | \(L_u=5.80\) m, \(D_u=0.20\) m, slope 1:100 | \(x=-5.80\) to 0 m; invert falls toward the chamber |
| junction chamber | 0.30 × 0.30 × 0.45 m (length × width × height) | \(x=0\) to 0.30 m; downstream invert \(z=0\) |
| invert drop | 0.18 m | upstream invert at the chamber is \(z=0.18\) m |
| downstream acrylic pipe | \(L_d=5.95\) m, \(D_d=0.28\) m, horizontal | \(x=0.30\) to 6.25 m |
| vertical riser | \(d_r=0.06\) m, length 1.22 m | chamber-top centre; \(z=0.45\) to 1.67 m |
| downstream control | movable flat tailgate | equivalent resolved orifice; its unreported opening is uncertain |
| riser outlet | open laboratory atmosphere | resolved plume region with atmospheric open boundaries |

The four pressure transducers and their stated positions are on p. 2
(Fig. 2 is on p. 3):

* PT1: riser wall, 0.80 m above the chamber top.
* PT2: chamber top.
* PT3: chamber front wall, 0.02 m above the chamber bottom.
* PT4: upstream-pipe crown, 0.30 m upstream of the chamber.

The transducers had ±130 kPa range and ±0.2% stated accuracy; data were
sampled at 1,000 Hz (p. 2). The present point probes use those locations. A
point in a finite CFD cell cannot reproduce the precise wall-tap stagnation
response, so both absolute pressure and local phase fraction are retained.

## C9 operating condition

Table 1 (p. 4) identifies C9 as Series C with:

* \(Q_0=25\) L/s and \(Q_1=40\) L/s;
* downstream full-pipe flow controlled by the tailgate;
* an entrapped upstream air pocket; and
* initial riser water-column height \(h_{r0}=0.30\) m.

The manual valve opening took 0.2–0.4 s (p. 3). The paper uses
\(T_v=0.40\) s in Eq. (7)/Fig. 13 (pp. 11–12), so the CFD inlet ramps
linearly over 0.40 s. All tests were at about 20 °C (p. 3).

Series C was prepared by first establishing steady free-surface flow at
\(Q_0\), partly closing the tailgate so a surge filled the downstream pipe
and chamber, and then adjusting the gate until the required riser level was
obtained. Air formerly in the upper upstream pipe was thereby trapped
(p. 3). This preparation history is important: the final initial condition is
not an arbitrary spherical bubble in a full pipe.

## What the paper does and does not determine about the air pocket

The following points are directly supported:

* **Location:** upper part/crown of the upstream pipe (pp. 3 and 6; Fig. 2b).
* **Topology:** a main pocket plus a thin crown layer downstream of it. The
  thin layer initially connected the pocket to the chamber and allowed
  discrete bubbles into the chamber (p. 6).
* **Transport:** during phase 1 the upstream end moved downstream slowly
  while the downstream end stayed near its initial position; the main body
  reached the chamber at 6.46 s (pp. 7–8).
* **Pressure observation:** PT4 measures the pocket pressure once the main
  body reaches PT4/chamber (p. 7). The digitized pre-ramp PT4 trace is about
  3.6 kPa gauge, consistent with the reported hydrostatic initial state.
* **Atmospheric connectivity:** the entrapped upstream pocket is not directly
  vented to atmosphere. It releases into the chamber and then through a
  water/air mixture in the riser. An air slug becomes atmospheric only when
  it breaks at the riser top (p. 9).

The paper gives **no numerical pocket length, volume, upstream interface
coordinate, downstream interface coordinate, or separately tabulated initial
pocket pressure**. Fig. 2 is explicitly “not to scale.” Fig. 8 only images the
chamber and riser, not the complete upstream pocket. Those quantities
therefore cannot be labelled as paper parameters.

The CFD source exposes them as uncertain parameters. The three supplied
priors are geometric sensitivity cases, not claimed measurements:

| profile | body tail \(x\) | body nose \(x\) | body-interface \(z\) | thin layer | analytic volume |
|---|---:|---:|---:|---:|---:|
| `pocket_small` | −3.50 m | −1.20 m | 0.382 m | 0.006 m | 4.64 L |
| `base` | −4.80 m | −1.00 m | 0.378 m | 0.008 m | 12.64 L |
| `pocket_large` | −5.30 m | −0.70 m | 0.370 m | 0.010 m | 21.61 L |

The base gas pressure is obtained from hydrostatic compatibility with the
reported PT2 initial pressure and the selected interface, not by fitting the
eruption count:

\[
p_{g,0}-p_{atm}=\rho_w g[(0.45+h_{r0})-z_{interface}]
\approx 3.64\ {\rm kPa}.
\]

The mesh-integrated initial volume and mass, rather than the analytic cap
volume, are the values used for conservation reporting. The sensitivity range
must be retained when interpreting pocket-arrival time or phase-2 results.

## Experimental chronology and quantitative targets

The detailed C9 discussion is on pp. 6–10 and Figs. 8–11:

* Fig. 9 separates two phases. Phase 1 contains the first and second geysers,
  driven by the inflow transient; phase 2 contains six geysers (third through
  eighth), driven by release from the arriving pocket (pp. 6–7).
* PT2 first peaks at 10.69 kPa at 0.50 s (p. 7).
* The first free surface reaches the riser top at 0.73 s (p. 7).
* The phase-1 air cavity appears near 1.30 s, disappears at 1.97 s, and the
  second geyser ends at 3.99 s (p. 7).
* The main air pocket reaches the chamber at 6.46 s; this is the phase-2
  boundary (p. 7).
* The third geyser starts at 6.70 s and ends at 7.59 s (pp. 7 and 9).
  Air slug A1 expands from about 0.91 to 1.70 L between 6.70 and 7.00 s,
  with an ideal-gas pressure reduction of about 46% (p. 8).
* The fourth geyser starts at 8.12 s; PT4 and PT2 are then 9.33 and
  6.93 kPa (p. 9). Fifth through eighth geysers follow.
* The system is final-steady after about 19 s. Mean final PT2, PT3, and PT4
  pressures are 8.79, 12.76, and 9.25 kPa (p. 9).
* Fig. 11 (p. 10) reports positive approximately linear relationships between
  maximum PT2 pressure and both jetting height and final pressure. Across
  Series C, \(P_{Final}\) is about 0.56–0.76 \(P_{Max}\) (p. 10). The
  digitized case files give the regression coefficients used in plots; they
  are digitized values, not equations printed in the article text.

The paper defines a geyser as water slug or air–water mixture released out of
the riser top, and defines jetting height from the riser bottom (p. 3). The CFD
event detector follows that definition: merely filling the riser is not
counted unless water crosses the physical rim.

## Eq. (7), Eq. (8), and Fig. 13

The analytical model assumptions are stated on p. 11: initially water-filled
upstream/downstream pipes and chamber, incompressible water, negligible
losses/velocity head, and a sufficiently long nonspilling riser. It is
therefore a phase-1 water-column benchmark, not a phase-2 air-release model.

For a linear inflow increase, Eq. (7) gives the oscillating piezometric head

\[
H_s=H_{s0}+\frac{L_d}{gA_d}\frac{dQ_u}{dt}
\left[1-\cos\left(\frac{2\pi t}{T}\right)\right],
\]

with the paper’s typesetting interpreted as the coefficient multiplying the
bracketed term. Eq. (8) gives

\[
T=2\pi\sqrt{\frac{L_d A_r}{gA_d}+\frac{h_{r0}}{g}}.
\]

Fig. 13 (p. 12) compares Eq. (7) with measured Series-C PT2 head. For C9,
Eq. (8) gives 1.45 s (p. 12). The paper notes that Eq. (7) departs from the
measurement near minimum pressure when an air cavity forms, violating its
single-phase assumption (p. 12). It also estimates water-wave compressibility
as only 0.6% of the first period term for this acrylic downstream pipe
(pp. 11–12); that does not make gas compressibility negligible.

## Solver implication

Incompressible `interFoam` uses constant phase densities and cannot produce
the measured \(pV\) compression/expansion of a closed pocket. It is therefore
not the final C9 solver. The source case uses OpenFOAM v2512
`compressibleInterFoam`: VOF interface transport, perfect-gas air, and weakly
compressible liquid water. This retains an energy equation and supports
closed-pocket pressure/mass changes. `compressibleInterIsoFoam` is supplied as
an isothermal sensitivity variant. Both are interface-capturing continuum
models: they can advect and break a resolved gas region, but subcell bubbles,
coalescence, and entrainment are mesh/model dependent. Reproducing phase 1
does not by itself establish that phase 2 or eight eruptions were reproduced.
