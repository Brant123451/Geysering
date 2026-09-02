# Campaign 2 B-H1/B-H3/B-H6 2D parameter audit

Audit date: 2026-08-10 (Asia/Shanghai)

Scope: independent, read-only comparison of the published Cong et al. (2017)
experiment against the actual OpenFOAM dictionaries and solver source used by:

- B-H1: `/tmp/bh1-2d-study/h1_refined_co015`
- B-H3: `/tmp/bh3-2d-qualification/h3_refined_iso_riser20`
- B-H6: `/tmp/bh6-2d-study/paper_tau0p2_areaeq`

No HTML text, old handoff conclusion, or postprocessed outcome label was used as
parameter evidence. No running process or case dictionary was changed.

## Bottom line

The three cases **support the same core experimental contract**: the published
case scalars, axial layout, initial head and pocket, atmospheric pressure,
quiescent start, constant-head inlet, sealed cap, open riser top, gravity and
0.20 s opening duration are present in the actual cases. The three cases also
use byte-identical material, initial-field, pressure-boundary, valve and valve-
zone dictionaries.

The stronger statement "all parameters and geometry are exactly the paper
experiment" is **not supported**. There are three material qualifications:

1. the circular apparatus is mapped to a planar area-equivalent model;
2. the paper gives only an approximately 0.2 s manual opening, while the model
   chooses a sine-squared-area/Forchheimer law with a finite 0.1% minimum area;
3. several transport/compressibility and laminar-closure choices are not
   reported by the paper.

Accordingly, the evidence status for objective item 1 is
`SUPPORTED_WITH_DECLARED_MODEL_CHOICES`, not exact geometric/constitutive
identity.

## Highest-priority published evidence

Primary source:
`tests/test_02_cong2017/_shared/reference/paper_source/cong2017_JHE2017_offprint.pdf`.

- p. 2, Fig. 1(b): Series B topology; `D=0.05 m`; tee/valve/cap layout;
  approximately 6 m pipe and 1.8 m riser; constant-head upstream tank and
  removable downstream cap.
- p. 3, Table 2: B-H1/B-H3/B-H6 values and observed classifications.
- p. 3, Experimental Procedure: downstream section emptied, cap closed,
  atmospheric pocket, selected ball valve initially closed, manual opening
  approximately 0.2 s.
- p. 4: laboratory temperature `23 degC`, water density `998 kg/m3`.
- p. 7: riser height `1.8 m`, explicitly measured from the horizontal-pipe
  soffit (pipe crown); B-H6 is no-geyser.
- pp. 8-9: B-H1 geyser and pressure/trajectory description; perspex-wall
  heuristic uses Darcy-Weisbach `f approximately 0.01`.
- p. 13: Series B `H0` is upstream head measured from the tunnel invert;
  `Vair=pi D^2 L0/4`, `Vw=pi Dr^2 H0/4`, and `sigma=0.072 N/m`.

The case `manifest.yaml` files confirm the three case identities but contain
no field-level geometry, material, IC or BC contract. The per-case
`config/case.json` files reproduce the Table 2 values; actual OpenFOAM files
were inspected independently.

## Published case scalars versus actual geometry

| Item | B-H1 | B-H3 | B-H6 | Evidence status |
|---|---:|---:|---:|---|
| `D` | 0.050 m | 0.050 m | 0.050 m | supported |
| physical `Dr` | 0.016 m | 0.026 m | 0.041 m | supported as source value |
| `Dr/D` | 0.32 | 0.52 | 0.82 | supported |
| `H0`, from invert | 0.66 m | 0.66 m | 0.66 m | supported |
| `L0` | 0.61 m | 0.61 m | 0.61 m | supported |
| `Vair/Vw` | 9.03 | 3.42 | 1.37 | preserved by the planar mapping |
| experiment geyser | yes | yes | no | Table 2 evidence only; not a parameter |
| planar riser width `W=Dr^2/D` | 0.00512 m | 0.01352 m | 0.03362 m | model choice, correctly calculated |

Actual `system/blockMeshDict` files give:

- pipe invert/crown `z=-0.025/+0.025 m`, hence 0.050 m channel height;
- inlet/Valve-1 reference plane `x=0`;
- tee axis `x=3.47 m`;
- selected release valve `x=5.98 m`;
- sealed cap `x=6.59 m`;
- initial pocket interval `x=5.98..6.59 m`;
- riser from crown `z=0.025 m` to physical rim `z=1.825 m`, exactly 1.80 m;
- gravity `(0 0 -9.81) m/s2`.

The selected-valve location is supported by Fig. 1's chain:
`3.47 + (3.12-L0) = 5.98 m`, and the cap is
`3.47 + 3.12 = 6.59 m`. The valve resistance zone is immediately upstream,
`x=5.954..5.980001 m`, in every case (`system/topoSetDict`). The initial
pocket is downstream of the valve and the observed gas front propagates
upstream, consistent with the published description.

### Area-equivalent 2D mapping: support range and impact

The actual riser is **not drawn with width `Dr`**. It uses
`W_2D=Dr^2/D`, so

`W_2D/D = (Dr/D)^2`,

which is the circular riser/main-pipe area ratio. It also preserves the
published air-to-riser-water volume ratio:

`D L0 / (W_2D H0) = D^2 L0 / (Dr^2 H0) = Vair/Vw`.

This is a clear, common and defensible planar mapping, not per-case fitting.
It does **not** preserve circular diameter, wetted perimeter, curvature,
three-dimensional stratification, or annular falling-film topology. Those
differences can affect quantitative damping, bubble/film kinematics and the
near-boundary B-H3 classification. The model also adds an atmospheric plume
domain `x=3.32..3.62 m`, `z=1.825..3.025 m`; the physical decision plane
remains the paper rim at `z=1.825 m`.

## Initial conditions and boundaries

The following files are byte-identical across H1/H3/H6:

- `0.orig/U`: uniform `(0 0 0)`;
- `0.orig/alpha.water`;
- `0.orig/p` and `0.orig/p_rgh`;
- `0.orig/T`;
- `system/setFieldsDict`;
- `system/topoSetDict`;
- `constant/g` and all thermophysical/valve dictionaries.

The actual fields implement:

- liquid everywhere by default;
- atmospheric gas in `x=5.98..6.591 m` (the 0.61 m downstream pocket);
- gas above `z=0.635 m` in the riser and numerical atmosphere;
- `H0=0.66 m` from invert `z=-0.025`, hence free surface
  `z=-0.025+0.66=0.635 m`, or 0.61 m above the crown;
- initial absolute pocket/free-surface pressure `101325 Pa`;
- quiescent velocity and uniform `296.15 K`;
- hydrostatic reduced pressure: water-side
  `101325 + 998*9.81*0.635 = 107541.89130 Pa`; atmospheric-air value
  `101332.42484 Pa` in `p_rgh`;
- upstream `fixedValue p_rgh=107541.89130 Pa`, liquid fraction 1 and
  `pressureInletOutletVelocity` (constant-head reservoir surrogate);
- downstream `downstreamCap` wall, no slip;
- atmosphere fixed-pressure/open-velocity with gas backflow;
- acrylic walls represented as no slip; no fitted contact angle
  (`alpha.water zeroGradient`).

These support the paper's physical IC/BC at the level available to a planar
CFD model. The upstream tank is a pressure boundary rather than resolved tank
geometry, and the room above the riser is a finite numerical air box; these
are disclosed boundary-model choices.

## Materials and closures

| Actual field/value | Paper support | Assessment |
|---|---|---|
| `T=296.15 K` | 23 degC on p. 4 | supported |
| `rho_w=998 kg/m3` at reference state | p. 4 | supported |
| `sigma=0.072 N/m` | p. 13 | supported |
| acrylic/perspex, no-slip wall | apparatus/p. 9 | partially supported; wetting not reported |
| `g=9.81 m/s2` | physical constant | supported |
| `mu_w=1.003e-3 Pa s` | paper gives only kinematic-viscosity symbol, no value | model choice |
| water bulk modulus `2.2 GPa` (`perfectFluid`) | not reported | model choice |
| air `M=28.965`, `mu=1.81e-5 Pa s`, perfect gas | not reported | model choice |
| `Cp`, `Pr`, pressure floor `pMin=50000 Pa` | not reported | numerical/constitutive choices |
| `simulationType laminar` | not reported; paper heuristic uses `f approximately 0.01` | unsupported closure choice |

Using Table 2's `Uf/sqrt(gD) approximately 0.44`, the model properties give a
reference `Re approximately 1.5e4`. Therefore the laminar planar-wall choice
can materially affect damping and pressure/arrival metrics and should not be
described as a published parameter. It is common to all three cases, so it is
not per-case outcome fitting.

## 0.20 s valve: paper support, model choice and measured early effect

### Paper support

The paper supports only these facts: a selected quarter-turn ball valve is
initially closed, then manually opened quickly, and the operation takes
approximately `0.2 s`. It does not publish valve angle versus time, effective
area versus angle, or a loss coefficient.

### Actual common model choice

All three actual `constant/valveProperties` files have the same SHA-256 and
specify:

- `model sineSquaredAreaForchheimer`;
- `openingDuration 0.2`;
- `minimumAreaFraction 0.001`;
- `resistanceLength 0.025`;
- `referenceFlowArea 0.00005` (0.05 m by 0.001 m extrusion).

The three `UEqn.H` files are also byte-identical. They apply

`A/A0 = max(0.001, sin^2(pi t/(2*0.2)))`,

`K = 1/(A/A0)^2 - 1`,

through a passive Picard-linearized Forchheimer loss, and remove the added
loss at `t>=0.2 s`. Thus `t=0` has `A/A0=0.001`, `K=999999`: an effective
near-closure regularization, **not a mathematically closed valve**. The 0.1%
plateau lasts until `t=0.0040270 s`.

### Read-only early-flow evidence

No archived valve-face `phi` or valve-zone `U` time series exists before the
first field write at 0.05 s, so exact cumulative through-valve leakage cannot
be reconstructed and is not claimed. Two available observables were audited:

1. the first `U` probe at the cap-side gas cell (`x=6.585 m`) at 0.005 s is
   only `3.90e-5`, `3.97e-5`, and `4.59e-5 m/s` for H1/H3/H6;
2. from the first solver step (`1.1999e-5 s`) to the last logged step before
   the 0.1% plateau ends, the global integrated alpha-water-volume change is
   approximately `0.349`, `0.371`, and `0.210 microlitre` for H1/H3/H6.

The second number is only a conservative **global phase-volume proxy**: it
includes inlet motion and interface-transport conservation error and is not a
valve-face flux integral. It is `0.00069-0.00122%` of the 2D pocket volume.
Together these outputs support "approximately no flow at model resolution",
but never "strictly closed". The common regularization is unlikely to create
a case-specific bias, although it can seed a very small common pressure/phase
response during the first 4 ms.

## Cross-case identity and numerical differences

Representative common SHA-256 values:

| File | SHA-256 |
|---|---|
| `constant/g` | `20062b7027fb1ebf12f829030e42c2248648cc7a0982c8657628829811b56c29` |
| `constant/thermophysicalProperties` | `d9c5d7be02670bec91d3bf62242d17902888be4680a9ab8da5e79a7c7ee8f7c7` |
| `constant/thermophysicalProperties.water` | `f8b457ab2367dba5ca83243723ab31ca0025949b884613b8b620eccb5424fcc6` |
| `constant/thermophysicalProperties.air` | `a555af6f9dd52341195acbcfe3b43d438a66ee035bcca1e9df74236e953c6c3f` |
| `constant/turbulenceProperties` | `178a6a812f439a52a2874885f8c19278526d63dc408b1fff186cc4c2c03b216f` |
| `constant/valveProperties` | `eda9ea641b1640a9b2a15ab7d9de80e18f845918a97a9d9494efd3693e6913f5` |
| `system/setFieldsDict` | `a438d4ac88238c2b5155fb8c70eb7660232cc29a61dee2a87706e05df441605b` |
| valve `UEqn.H` | `74de751b6febc90a500b642187f7aece3f5bf1c85e64273b4895bc8c4fbce83b` |

The physical contracts are common, but the numerical configurations are not
identical: H3 uses isoAdvector, linear-upwind momentum, 2400 riser-z cells and
strict `maxCo/maxAlphaCo=0.15/0.10`; H1 uses MULES with 2400 riser-z cells and
the same Courant limits; H6 uses MULES with 1200 riser-z cells and
`0.25/0.20`. This is a numerical-resolution/scheme difference, not a paper
input difference, and should be disclosed in any cross-case 2D claim.

## Evidence conflicts and stale metadata

- H3 `/tmp/.../paper_audit.json` still checks the old baseline strings
  `endTime 13` and `maxCo 0.25`; the actual refined
  `system/controlDict` is `endTime 20`, `maxCo 0.15`, `maxAlphaCo 0.10`.
- H6 `/tmp/.../paper_audit.json` retains the original 13 s check, while the
  completed continuation's actual `controlDict` targets 20 s.
- H1 registered `case_config.json` planned 16 s; its actual `controlDict` was
  reread for a normal effective end at 14.8529 s. This is an observation-window
  record issue, not a geometry/IC/BC change.

Those stale numerical checks must not be cited as current actual-case proof.

## Safe claim for the manuscript/workflow

Supported wording:

> The three 2D cases share the published Campaign 2 scalar inputs and axial
> layout, common hydrostatic initial and pressure boundary conditions, and a
> common passive 0.20 s valve surrogate. The planar riser width preserves the
> circular riser/main-pipe area and initial air/water-volume ratios.

Required limitation immediately after it:

> The 2D representation does not preserve circular perimeter or three-
> dimensional film/stratification, and the valve trajectory, laminar closure
> and unreported transport/compressibility properties are modeling choices;
> the valve is effectively but not mathematically closed during the first
> 4 ms.

