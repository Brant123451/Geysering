# Cong (2017) Campaign 2: independent 2D result audit

## Bottom line

- **B-H1:** the completed 2D run supports the published **geyser outcome** and
  **front-arrival time**, but it does not quantitatively reproduce the
  eruption transient.  True liquid passage across the physical rim occurs at
  14.00 s instead of about 9.55 s; all three vertical speed measures are much
  too small; and the published compression/surge mechanism is absent.  The
  defensible claim is therefore *outcome and arrival only*, not "quantitative
  agreement."
- **B-H6:** the 20 s run supports the published **no-geyser outcome**, arrival
  time, qualitative stage order, and approximately hydrostatic vertical-pocket
  pressure.  The model rise and pocket-break process is nevertheless much too
  slow and reaches lower interface levels.  The defensible claim is
  *qualitative regime/mechanism agreement with major kinematic bias*.
- **Campaign-level status:** incomplete.  The refined B-H3 run is still active;
  this audit neither reads its changing fields nor changes/stops it.  A frozen
  B-H3 gate is provided below.

The result is stricter than a visual HTML comparison.  Classification comes
from finite liquid passage through the physical `z = 1.825 m` rim, and all
quantitative comparisons use the same formulas and valve-opening time origin.

## Evidence and terminology

The primary experimental source is Cong, Chan, and Lee (2017), *Geyser
Formation by Release of Entrapped Air from Horizontal Pipe into Vertical
Shaft*, Journal of Hydraulic Engineering 143(9), 04017039.  The local source
PDF is:

`tests/test_02_cong2017/_shared/reference/paper_source/cong2017_JHE2017_offprint.pdf`

Evidence types are kept separate:

1. **Published measurement:** Table 2 or an exact-run figure/prose statement.
2. **Figure digitization:** approximate values recovered from a rendered graph.
3. **Parameter-matched companion:** B-1 has the same physical parameters as
   B-H1, and B-32 has the same parameters as B-H6, but they are not the same
   experimental runs.
4. **2D-derived:** centreline `alpha.water = 0.5` crossings, gas-fraction-
   weighted pocket pressure, PT1 probe output, or the physical-rim surface
   integral.
5. **Not directly comparable:** raw planar per-extrusion discharge/volume is
   not compared with a circular experimental discharge/volume.

The paper prose associates the pressure discussion with B-H1/B-H6, but the
Fig. 10 caption explicitly identifies the plotted traces as **B-1 and B-32**.
This audit follows the figure caption and does not relabel those traces as
exact B-H1/B-H6 measurements.

## Common rules fixed before B-H3 completes

- Time zero: start of valve opening.
- `supported`: absolute relative error at most 10%, or an exact qualitative
  match.
- `partial`: relative error above 10% and at most 30%, or only part of a
  qualitative sequence matches.
- `failed`: relative error above 30%, or modeled behavior contradicts the
  published behavior.
- `missing`: no genuinely comparable quantity exists.
- Arrival: first resolved gas nose 5 mm above the pipe crown.
- Geyser: the common physical-rim gate, not the superseded 98%-of-rim level and
  not a rendered image.
- Catch-up/break proxy: first three consecutive stored samples satisfying
  `Yfs - Yint <= max(0.02 initial gap, 2 riser dz)`.
- Event end: earliest of physical-rim crossing, the catch-up proxy, or stored
  event end.
- Model `vfs` and `vint`: endpoint displacement divided by this event duration;
  `vnet = vint - vfs`.
- Vertical-pocket pressure: gas-fraction-weighted pressure of the uppermost
  enclosed centreline gas component, reported as `Ha/Lw`.  Samples with an
  inadequately resolved water column are excluded by one common resolution
  rule.

The paper states that its reported speeds are averages of measured
trajectories, but does not disclose the exact averaging implementation.  Thus
the Table-2/model speed comparison is physically matched but not algorithmically
identical; this comparability limitation does not explain errors of 43-100%.

## Parameter identity

Status: **partial** (`SUPPORTED_WITH_DECLARED_MODEL_CHOICES` in the independent
parameter audit).

Both completed cases use the published `D = 0.05 m`, case-specific `Dr`,
`H0 = 0.66 m`, `L0 = 0.61 m`, axial tee/valve/cap layout, atmospheric initial
pocket, Series-B constant-head inlet, quiescent start, 23 degC material basis,
and approximately 0.20 s valve opening.  The three cases also use a common
area-equivalent planar mapping `W2D = Dr^2/D`.

They are not exact digital replicas of the experiment.  The planar mapping,
sine-squared Forchheimer valve trajectory, `0.001` minimum valve aperture,
laminar closure, detailed thermophysical properties, and finite exterior air
box are declared modeling choices not specified by the paper.  The 0.20 s
duration is paper-based; the exact opening path is not.

## B-H1 audit

| Item | Published B-H1 evidence | Completed 2D evidence | Status |
|---|---:|---:|---|
| Geyser classification | yes | true liquid crosses physical rim | **supported** |
| Front arrival `Ta` | 8.07 s | 8.50 s (+5.3%) | **supported** |
| First true rim ejection | about 9.55 s | 14.00 s (+46.6%) | **failed** |
| `vfs` | 0.924 m/s | 0.2168 m/s (-76.5%) | **failed** |
| `vint` | 1.231 m/s | 0.2175 m/s (-82.3%) | **failed** |
| `vnet` | 0.301 m/s | 0.00079 m/s (-99.7%) | **failed** |
| `vnet/vTaylor` | 2.30 | 0.006 | **failed** |
| Exact-run numerical `H/H0` peak | not published for B-H1 | available only for model | **missing** |
| Compression/surge mechanism | strong late compression and rapid acceleration | `Ha/Lw` median 1.019, p05-p95 1.004-1.034; no surge | **failed** |
| Event order | arrival, partial entry, compression, surge, rapid ejection | arrival, slow near-plug rise, late ejection | **partial** |

The physical-rim conclusion is strong: at 14.00 s the stored-field audit finds
both a resolved upward liquid component and more than one adjacent-cell volume
of cumulative positive liquid passage.  The normal end at 14.8529203 s occurs
after this irreversible positive event, so failure to reach the originally
declared 16 s does not invalidate the positive classification.

The mechanism is not aligned.  From arrival to rim crossing, the model keeps
`Yfs - Yint` close to 0.60 m, so free surface and gas nose rise almost together.
In the experiment the gas nose accelerates relative to the surface, the pocket
is compressed, and the final jet follows a sharp pressure rise.  This explains
why a correct binary outcome is not enough to call H1 quantitatively qualified.

### H1 pressure normalization

For the model, PT1 gauge head is normalized by the same published
`H0 = 0.66 m`.  The global early oscillation peak is `1.777 H0`; the
post-arrival/event peak is only `0.951 H0`.

Fig. 10(a)'s parameter-matched **B-1**, digitized as `1.929 H0` globally and
`1.872 H0` after its arrival, gives two different conclusions:

- the initial oscillation amplitude is close (-7.9%, **supported companion
  evidence**);
- the post-arrival surge is absent (-49.2%, **failed companion evidence**).

These companion values must not be described as exact B-H1 pressure
measurements.

## B-H6 audit

| Item | Published B-H6 evidence | Completed 2D evidence | Status |
|---|---:|---:|---|
| Complete observation | experiment about 20 s | normal `End` at 20.0 s, no fatal | **supported** |
| Geyser classification | no | no; rim alpha, positive flow, and cumulative volume all exactly zero | **supported** |
| Front arrival `Ta` | 8.10 s | 8.04 s (-0.7%) | **supported** |
| Peak `Yfs` | 1.2008 m (Fig. 7a digitized) | 1.0212 m (-15.0%) | **partial** |
| Peak `Yint` before break | 1.1784 m (Fig. 7a digitized) | 0.9218 m (-21.8%) | **partial** |
| Break/catch-up | about 10.58-10.90 s | common proxy 12.47 s, 1.57-1.89 s late | **failed** |
| `vfs` | 0.246 m/s | 0.0721 m/s (-70.7%) | **failed** |
| `vint` | 0.476 m/s | 0.2066 m/s (-56.6%) | **failed** |
| `vnet` | 0.235 m/s | 0.1345 m/s (-42.8%) | **failed** |
| Hydrostatic pocket relation `Ha/Lw` | median about 0.964 | model median 1.004 | **supported** |
| Event order | arrival, Taylor-like rise, break in riser, no geyser | same broad order, but slower/lower and more oscillatory | **partial** |

The no-geyser result has complete-event support: all 401 stored rim planes from
0 to 20 s were sampled, the solver ended normally, and no finite liquid crossed
the rim.  This is stronger than merely failing to reach a display threshold.

The exact-run pressure mechanism is also credible.  A reproducible
digitization of the 29 red-square values in Fig. 7(d) gives `Ha/Lw` =
0.875-1.155, median 0.964.  The 2D uppermost enclosed gas component gives a
central 90% interval 0.850-1.173, median 1.004.  This supports the paper's
statement that the large-riser pocket is approximately hydrostatically
supported.  It does not rescue the too-slow interface kinematics.

Fig. 10(b) is parameter-matched **B-32**, not exact B-H6.  Relative to that
companion trace, the model early global PT1 peak is close (`1.820` versus
`1.929 H0`, -5.6%), while the post-arrival peak is low (`1.106` versus
`1.586 H0`, -30.2%).  The exact B-H6 `H/H0` peak remains **missing**.

## Event-stage comparison

### B-H1

1. Density-current propagation and arrival: **supported in timing**.
2. Entry and vertical rise: present, but the model behaves like a slow,
   nearly fixed-length plug rather than the rapidly evolving measured pocket.
3. Pocket compression and post-arrival pressure surge: **failed**.
4. Rapid top ejection: outcome present, but 4.45 s late and too slow.

### B-H6

1. Arrival: **supported**.
2. Hydrostatic/Taylor-like rise: **supported in pressure character**, failed in
   speed.
3. Pocket catches the free surface and breaks within the riser: broad topology
   present, but delayed.
4. No ejection through the complete 20 s event: **supported**.

## Frozen B-H3 gate

When the refined from-zero B-H3 run reaches a normal 20 s end, use this exact
workflow without changing thresholds or borrowing H1 timings:

| Quantity | Published B-H3 reference | Final treatment |
|---|---:|---|
| Classification | geyser | common physical-rim outlet gate |
| `Ta` | 8.18 s | same 5 mm arrival definition |
| `vfs` | 0.657 m/s | same event-endpoint formula and 10%/30% bands |
| `vint` | 0.916 m/s | same formula and bands |
| `vnet` | 0.267 m/s | `vint-vfs`, same bands |
| `vTaylor` | 0.174 m/s | report `vnet/vTaylor` |
| Physical-rim time | not published for exact B-H3 | **missing**, do not use H1's 9.55 s |
| Exact pressure peak/series | not published for exact B-H3 | **missing**, retain only qualitative mechanism evidence |
| Detailed exact-run chronology | not published | **missing/qualitative**, do not invent stages or times |

Required inputs are the normal 20 s solver log, complete stored trajectory and
PT1 series, final physical-rim report, actual case/paper-contract audit, and
centreline `alpha.water`, `p`, and `U`.  The analysis script in this directory
accepts those same inputs directly.

## What can safely be claimed now

- **Safe:** the selected H1 and H6 2D runs use the intended experimental scalar
  setup within declared 2D/modeling approximations; H1 ejects and H6 does not;
  both front-arrival times agree well; H6 exhibits approximately hydrostatic
  pocket pressure.
- **Not safe:** "the H1 and H6 2D transients quantitatively agree with the
  experiments," "H1 reproduces the compression-driven surge," or "H6
  reproduces the measured rise speed."
- **Manuscript use:** H1 may be used as qualitative classification/outlet
  evidence with explicit timing/mechanism limitations.  H6 may be used as
  qualitative no-geyser and hydrostatic-mechanism evidence, again disclosing
  the kinematic bias.  Neither is a strong quantitative trajectory validation.

## Reproducibility artifacts

- `audit_completed_case.py`: common completed-case extractor, reusable for H3.
- `h1_result_audit.json`, `h6_result_audit.json`: per-case metrics, source
  hashes, and definitions.
- `h1_bubble_pressure_series.csv`, `h6_bubble_pressure_series.csv`: direct
  centreline gas-pressure derivations.
- `digitize_red_square_series.py`: fixed-axis digitizer.
- `paper_fig7d_h6_Ha_over_Lw_digitized.csv` and its JSON manifest: approximate
  exact-run Fig. 7(d) pressure-ratio evidence.
- `campaign2_result_audit_summary.json`: machine-readable claim/evidence map.

No manuscript source, OpenFOAM field, solver, boundary condition, or active H3
process was modified by this audit.
