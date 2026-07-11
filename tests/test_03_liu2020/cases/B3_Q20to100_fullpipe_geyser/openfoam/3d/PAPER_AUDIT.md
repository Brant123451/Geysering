# Liu, Shao & Zhu (2020) — Case B3 paper audit

This audit freezes the experimental inputs used by the three-dimensional
validation. It distinguishes reported quantities from numerical boundary
realisations; the latter are not fitted to the B3 pressure record.

## Source availability and traceability

- Article: L. Liu, W. Shao and D. Z. Zhu, *Experimental Study on Stormwater
  Geyser in Vertical Shaft above Junction Chamber*, Journal of Hydraulic
  Engineering 146(2), 04019055,
  DOI `10.1061/(ASCE)HY.1943-7900.0001660`.
- The requested file `tests/test_03_liu2020/references/liu2020.pdf` is absent
  from required base commit `867b2fccd591a9f44325a13c2042bbce32405087`.
  `docs/file-layout/pre-migration-files.csv` records the former
  `papers/liu2020.pdf` as 3,024,304 bytes with SHA-256
  `32abd351252725009ebbef98c8d5c7794b957a448ee23bf8729fcb347ce5c9c3`,
  but the blob is not in Git history. The ASCE page exposes the abstract but
  the full text is access controlled.
- The requested B3 `reference/paper_scans/` directory is also empty. The
  migration manifest records page scans for Fig. 5, Fig. 6 and Fig. 7, but
  those image blobs are absent.
- Numerical values below were therefore cross-checked between the parsed paper
  record `_shared/metadata/paper_reference/paper_parameters_Liu2020_JHE.md`,
  the paper quotations frozen in `scripts/caseB_digitize_and_compare.py`,
  `outputs/caseB_metrics.json`, and the three independently digitised Fig. 5(b)
  traces in `data/digitized/`. Figure page references use the former scan names
  (`p05`, `p06`, `p07`) and are not a claim that the missing PDF was re-read.
- No `PAPER_AUDIT.md` exists in A2 on the required base commit. The A2
  `openfoam/3d` source, A2 README, and shared parsed paper record were audited
  read-only. A2 was not modified.

This missing-source limitation prevents a new page-image audit. It does not
justify inventing the unreported tail-gate opening, initial chamber level, air
volume, or sensor coordinates.

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
- The reported valve-opening duration is approximately `0.4 s`. The paper
  materials available in this commit do not resolve the valve trace within
  that interval. A linear ramp is a declared numerical realisation, not a
  measured waveform.
- The paper states that the initial conditions of B3 differ from A2 only in
  the downstream condition:
  - A2: downstream open-channel flow, initially `hd/Dd = 1/4`, weir controlled;
  - B3: downstream pipe initially full, tail-gate controlled.
- Series B has no deliberately trapped upstream air pocket. That intervention
  belongs to Series C.
- Pipe diameters, chamber volume, riser length, inflow, and initial gas volume
  are not changed to force geysering.

The tail-gate opening, gate loss coefficient, receiving-tank level, and exact
initial B3 chamber level are not reported in the available paper record.
Consequently the CFD boundary uses the least-head interpretation already
declared by the frozen B3 one-dimensional model: a submerged full-bore outlet
with tailwater free-surface elevation at the downstream crown
`H_tail = Dd = 0.28 m`. This is a reproducible *boundary realisation*, not a
paper datum and not a calibration. No gate solid or opening area is fabricated.

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

The former scan names place Fig. 5 on PDF page 5, Fig. 6 on page 6, and Fig. 7
on page 7. These pages are unavailable in the base commit.

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
flow increment is not available and is not reconstructed.

## Compressibility decision

B3 contains a short 55 kPa positive pulse followed by about -20 kPa gauge
pressure. These are pressure-wave and gas-compression observables, not merely
quasi-static free-surface head. The physical-water acoustic transit time over
the two pipes is about `11.75/1483 = 0.008 s`, far shorter than the
gravity/riser oscillation scale. An incompressible `interFoam` calculation
cannot establish the water-hammer amplitude and is therefore not used as the
validation solver.

The baseline uses OpenFOAM v2512 `compressibleInterFoam`:

- water: `perfectFluid`, `rho0=998.2 kg/m3`, `R=2.2e6 m2/s2`, giving
  `c=sqrt(R)~=1483 m/s` and a 2.2 GPa bulk modulus near atmospheric pressure;
- air: `perfectGas` at 293.15 K;
- VOF surface tension: 0.072 N/m;
- absolute atmospheric pressure: 101325 Pa.

No reduced numerical sound speed or fitted gas pocket is introduced. The PVC
wall is rigid because its thickness and elastic modulus are not reported.
Thus pipe-wall compliance, dissolved/dispersed air below the mesh scale,
cavitation, and phase change remain unresolved uncertainty sources.

## Initialisation and non-paper assumptions

- Downstream pipe: water filled to the full circular section.
- Junction chamber: initial water level 0.30 m, the minimal 0.02 m surcharge
  above the downstream crown used by the frozen B3 model. The paper does not
  report this level; sensitivity to this declared initialisation must not be
  confused with experimental uncertainty.
- Riser above that chamber level and the plume domain: atmospheric air.
- Upstream pipe/headbox: same free-surface initialisation used by A2.
- A constant-Q0 settling stage precedes the 0.4 s ramp. Reported comparison
  time is shifted so `t=0` is the ramp start.

No initial B3 air volume is prescribed because none is reported. Air present
in the chamber/riser follows directly from the measured geometry and declared
initial water surface, and remains connected to atmosphere.
