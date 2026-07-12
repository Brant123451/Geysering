# Liu, Shao & Zhu (2020) — Case B3 paper audit

This audit freezes the experimental inputs used by the three-dimensional
validation. It distinguishes reported quantities from numerical boundary
realisations; the latter are not fitted to the B3 pressure record.

## Source availability and traceability

- Article: L. Liu, W. Shao and D. Z. Zhu, *Experimental Study on Stormwater
  Geyser in Vertical Shaft above Junction Chamber*, Journal of Hydraulic
  Engineering 146(2), 04019055,
  DOI `10.1061/(ASCE)HY.1943-7900.0001660`.
- A user-supplied local copy is available at `references/liu2020.pdf`. Its
  SHA-256 is
  `32abd351252725009ebbef98c8d5c7794b957a448ee23bf8729fcb347ce5c9c3`,
  exactly matching the former file recorded by
  `docs/file-layout/pre-migration-files.csv`. The PDF is not part of the
  required base commit, but it was read directly for this audit.
- Pages 2--7 and 10--12 were checked directly. In particular, Fig. 2 fixes the
  apparatus and boundary-control hardware, Fig. 5 fixes the B3 chronology and
  pressure traces, Fig. 6 fixes the Series-B comparison, Fig. 7 fixes the
  pressure-height relation, Table 2 gives spilled volume, and pp. 11--12
  discuss the approximately 305 m/s acrylic-pipe wave speed and B3
  compressibility.
- Values were also cross-checked against
  `_shared/metadata/paper_reference/paper_parameters_Liu2020_JHE.md`,
  `scripts/caseB_digitize_and_compare.py`, `outputs/caseB_metrics.json`, and
  the independently digitised Fig. 5(b) traces in `data/digitized/`.
- No `PAPER_AUDIT.md` exists in A2 on the required base commit. The A2
  `openfoam/3d` source, A2 README, and shared parsed paper record were audited
  read-only. A2 was not modified.

## Experiment geometry (Fig. 2)

The B3 source case retains the same rig dimensions as A2:

| Component | Paper value | Model value |
|---|---:|---:|
| Upstream pipe | length 5.80 m, diameter 0.20 m, slope 1:100 | same |
| Junction chamber | 0.30 x 0.30 x 0.45 m | same |
| Upstream/downstream invert drop | 0.18 m | same |
| Downstream pipe | length 5.95 m, diameter 0.28 m, horizontal | same |
| Riser | diameter 0.06 m, length 1.22 m | same |
| Riser connection | centre of chamber lid, open to atmosphere | same |

The inlet headbox and the atmosphere volume above the riser are numerical
boundary plenums. They do not alter any reported pipe, chamber, or riser
dimension. In particular, the atmosphere volume is not an extension of the
physical riser: its lower annulus and side/top faces are open-pressure
boundaries, and the physical rim remains 1.22 m above the chamber lid.

## Flow programme and A2/B3 branch distinction

- B3 is Series B, `Q0 = 20 L/s` to `Q1 = 100 L/s`.
- The manual valve-opening duration varied from `0.2` to `0.4 s`; the paper's
  analytical comparisons use `0.4 s`. The CFD therefore uses a 0.4 s linear
  ramp as a declared numerical realisation, not a measured waveform.
- The paper states that the initial conditions of B3 differ from A2 only in
  the downstream condition:
  - A2: downstream open-channel flow, initially `hd/Dd = 1/4`, weir controlled;
  - B3: downstream pipe initially full, with `hd/Dd = 1`, controlled by the
    movable overflow weir in the downstream tank.
- The flat tailgate is used to create the pressurised Series-C cases. It is
  present in the rig but Table 1 does not identify it as the Series-B control.
- Series B has no deliberately trapped upstream air pocket. That intervention
  belongs to Series C.
- Pipe diameters, chamber volume, riser length, inflow, and initial gas volume
  are not changed to force geysering.

The detailed overflow-weir geometry/rating and receiving-tank transient are not
reported. The CFD therefore uses the hydrostatic equivalent of the reported
initial `hd/Dd=1`: a submerged full-bore outlet with tailwater free-surface
elevation at the downstream crown, `H_tail = Dd = 0.28 m`. This is a
reproducible *boundary realisation*, not a fitted transient. No unreported weir
or gate solid is fabricated.

## Pressure transducers

The paper reports OMEGA transducers with a 130 kPa range and 0.2% accuracy:

| Tap | Reported location | CFD sampling policy |
|---|---|---|
| PT1 | riser wall, 0.80 m above chamber lid | `z=1.25 m`, one cell inside the 0.03 m-radius riser |
| PT2 | chamber lid | one cell below the lid; in-plane location follows the A2 source |
| PT3 | chamber front wall, 0.02 m above its floor | `z=0.02 m`; in-plane location follows the A2 source |
| PT4 | upstream pipe crown, 0.30 m upstream of chamber | included as a diagnostic, not a requested metric |

The paper record does not provide PT2's exact in-plane coordinate or PT3's
horizontal coordinate. The A2 coordinates are retained for the branch
comparison and this uncertainty is not hidden. Pressure output is absolute
`p` minus 101.325 kPa, not `p_rgh`.

## B3 targets from Fig. 5, Fig. 6 and Fig. 7

Fig. 5 is on PDF page 5, Fig. 6 on page 6, and Fig. 7 on page 7; all three were
checked directly against the local PDF.

### Fig. 5(a): event chronology

| Event | Reported time after ramp start |
|---|---:|
| Filling bore reaches junction chamber | 1.20 s |
| Mist exits riser / main pressure peak (point C) | 1.47 s |
| Air-water mixture first exits physical riser | 1.51 s |
| Mixture column reaches riser top | 1.65 s |
| Column breakup frames | 1.70 and 1.89 s |

The experiment is classified as a single-shoot geyser.

### Fig. 5(b): pressure history

| Quantity | Paper text value |
|---|---:|
| PT2 maximum | 55.03 kPa at about 1.47 s |
| PT3 maximum | 51.76 kPa at the principal peak |
| PT1 minimum | -8.30 kPa |
| PT2 rebound minimum | -20.26 kPa |
| PT3 rebound minimum | -17.77 kPa |
| Subsequent periods | 0.51, 0.37, 0.37 s |
| Final time-averaged PT1/PT2/PT3 | 0.00 / 1.82 / 4.65 kPa |

Cross-checking the digitised median traces gives approximately PT2
`55.6 kPa at 1.47 s`, PT3 `54.2 kPa at 1.47 s`, PT2 minimum `-20.2 kPa`,
PT3 minimum `-17.8 kPa`, and PT1 minimum between `-7.4` and the quoted
`-8.3 kPa`. The digitised colour bands are used for time-history plots; quoted
paper extrema remain the scalar acceptance targets.

Fig. 6 contains the PT2 Series-B comparison B3/B6/B9/B12 at `Q1=100 L/s`.
There is no Fig. 6 curve digitisation in this commit, so no additional B3
numbers are inferred from it.

### Fig. 7(a): pressure-height relation

The reported Series-B regression is

`h = 0.6943 Pmax/(rho g) + 0.3086 m`, with `R^2 = 0.97`.

For `Pmax = 55.03 kPa`, `rho = 998.2 kg/m3`, and `g = 9.81 m/s2`, the
regression predicts `h ~= 4.21 m`. This is a regression estimate for B3, not a
separately reported exact B3 height. The riser-top threshold `h=1.22 m`
corresponds to a peak-pressure head of about `1.31 m`. Fig. 7(b)'s B3 critical
point has `Delta Q=80 L/s`; the reported geyser threshold lies between
`30` and `40 L/s`.

### Table 2: spilled water

The three B3 repeats spilled `0.65`, `0.78`, and `0.82 L`, with a reported mean
of `0.72 L`. The CFD comparison uses net integrated outward liquid flux over
all atmosphere faces; gross outward flux is retained separately because the
experimental value was water collected outside the riser.

## Compressibility decision

B3 contains a short 55 kPa positive pulse followed by about -20 kPa gauge
pressure. These are pressure-wave and gas-compression observables, not merely
quasi-static free-surface head. On p. 11 the paper gives an approximately
`305 m/s` wave speed for the nearly pure water in the clear acrylic downstream
pipe. The corresponding transit time over the two reported pipe lengths is
about `11.75/305 = 0.039 s`. An incompressible `interFoam` calculation cannot
establish the pressure amplitude and is therefore not used as the validation
solver.

The baseline uses OpenFOAM v2512 `compressibleInterIsoFoam`. It solves the
same two-compressible-phase pressure/energy system as
`compressibleInterFoam`, while using geometric isoAdvector reconstruction,
explicit phase-fraction bounds and clipping. Screening with
`compressibleInterFoam` produced local negative-temperature failures at mixed
tetrahedral free-surface cells, so its MULES interface transport was not used
for the production case:

- water/system compliance: `perfectFluid`, `R=93025 m2/s2`, giving
  `c=sqrt(R)=305 m/s`; `rho0=997.1107767 kg/m3` makes the EOS density
  `998.2 kg/m3` at 101325 Pa;
- air: `perfectGas` at 293.15 K;
- VOF surface tension: 0.072 N/m;
- absolute atmospheric pressure: 101325 Pa.

The 305 m/s value is a paper-sourced effective pipe-plus-water wave speed, not
a fitted reduced sound speed. The fluid-only mesh cannot deform its clear
acrylic pipes/riser or clear PVC chamber, so the effective EOS represents the
reported pipe compliance uniformly. Spatial differences in wall compliance,
dissolved/dispersed air below the mesh scale, cavitation, and phase change
remain unresolved uncertainty sources.

The production case bounds `|U| <= 50 m/s`, outside the expected experimental
state, only to stop isolated VOF/tetrahedral overshoots from corrupting the
thermodynamic update. Velocity and temperature extrema are stored so any
activation or thermal instability can be reported.

## Initialisation and non-paper assumptions

- Downstream pipe: water filled to the full circular section.
- Junction chamber: initial water level 0.30 m, 0.02 m above the downstream
  crown. The exact value is not tabulated, but it is consistent with the
  Fig. 5(a) `t=0` image and keeps the reported downstream pipe full while the
  0.45 m chamber remains unfilled. It is an image-based declared
  initialisation, not a pressure-fit parameter.
- Riser above that chamber level and the plume domain: atmospheric air.
- Upstream pipe: approximately 0.08 m initial depth, as reported for A2; B3
  retains A2's upstream condition. The numerical headbox is the inlet plenum.
- The experimental pre-ramp state carries `Q0=0.020 m3/s`. The CFD liquid
  velocity is therefore seeded with `Q0/A` in the 0.08 m-deep upstream pipe,
  chamber, and full downstream pipe; it is not started from stagnant water.
- A constant-Q0 settling stage precedes the 0.4 s ramp. Reported comparison
  time is shifted so `t=0` is the ramp start. The paper describes a manual
  0.2--0.4 s valve opening and also calls the fully-open indication `t=0`;
  therefore the exact sub-0.4 s time origin remains an experimental
  uncertainty.

No independent gas volume is tuned. Air in the chamber/riser follows directly
from the reported geometry and the declared Fig. 5-consistent initial water
surface, and remains connected to atmosphere.
