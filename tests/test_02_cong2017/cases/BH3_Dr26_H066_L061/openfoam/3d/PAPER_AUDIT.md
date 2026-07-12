# PAPER AUDIT — Cong, Chan & Lee (2017), Series B, Run B-H3

## Scope and audit gate

This audit applies only to `BH3_Dr26_H066_L061`. The primary source is
`references/cong2017.pdf`; repository notes and existing reduced-order outputs
are secondary candidates and cannot settle a disagreement with the paper.

- `VERIFIED`: unambiguously stated or dimensioned in the primary paper.
- `PARTIAL`: the paper constrains the item but omits a CFD-relevant detail.
- `UNRESOLVED`: no unambiguous model value can be obtained from the paper.

The first-pass gate was `STOP` because the primary experimental paper does not
report several CFD controls. It is now **RELEASED** using the directly paired
study by Chan, Cong & Lee (2018), *3D Numerical Modeling of Geyser Formation by
Release of Entrapped Air from Horizontal Pipe into Vertical Shaft*,
DOI `10.1061/(ASCE)HY.1943-7900.0001416`. That paper models the same apparatus
and supplies the absolute atmospheric pressure, numerical valve baseline,
computational height, wall roughness, pressure boundaries, gas law, and grid
strategy. These are identified below as paired-CFD inputs, not experimental
measurements. Values absent from both papers are declared numerical controls
and must be subjected to sensitivity checks.

## Required checks against the primary paper

Fig. 1 dimensions are referenced from the Valve #1/upstream test-pipe plane.
The CFD coordinate convention places its constant-head boundary on that plane
at `x = 0` and uses the horizontal-pipe invert as `z = 0`; absolute coordinates
below are therefore model translations of the paper's relative dimensions.

| Item | Primary-paper evidence | Audited B-H3 value | Status / consequence |
|---|---|---:|---|
| Horizontal-pipe physical total length | The prose says “approximately 6 m” (p. 2). Fig. 1 dimensions give `3.47 m` from Valve #1 to the T axis and `(3.12 m - L0) + L0 = 3.12 m` from the T axis to the cap. | `6.59 m` (nominal `6.6 m`) | `VERIFIED` from the dimension chain; this is consistent with, but more precise than, the rounded prose. |
| OpenFOAM effective modeled length | Fig. 1 gives the complete internal test-pipe dimension chain. Series B connects Valve #1 to a constant-head tank and closes the downstream cap. | `6.59 m` from a tank-as-pressure-boundary plane at `x=0` to the cap | `MODEL TRANSLATION`; the tank volume and short upstream plumbing are not modeled. |
| T-junction position | Fig. 1 explicitly labels Valve #1-to-riser-axis distance `3.47 m`. | `xT = 3.47 m` in the stated CFD coordinate convention | `VERIFIED` relative dimension; repository candidate `x=2.88 m` conflicts with the Fig. 1 dimension. |
| Ball-valve position | Fig. 1 labels T-to-selected-valve distance `3.12 m - L0` and selected-valve-to-cap distance `L0`. Table 2 gives B-H3 `L0=0.61 m`; the corresponding drawn release plane is the most-downstream valve (#4). | release plane `x = 3.47 + 3.12 - 0.61 = 5.98 m` | `VERIFIED` geometry; the experiment's opening is `~0.2 s`, while instantaneous opening is a paired-CFD numerical baseline. |
| Initial air-pocket position | The downstream section is emptied and capped, leaving atmospheric air (pp. 2–3); Table 2 gives `L0=0.61 m`. | `5.98 <= x <= 6.59 m`, full circular section, initially atmospheric | `VERIFIED`. |
| Closed-end position | Fig. 1 places the removable cap `L0` downstream of the selected valve; the procedure closes it with a plastic cap (pp. 2–3). | cap at `x=6.59 m` | `VERIFIED` as a no-flow wall. |
| Physical riser height | The apparatus has a `1.8 m` riser (p. 2), measured from the horizontal-pipe soffit/crown (p. 7). | `1.8 m` above the pipe crown; top at model `z=1.850 m` | `VERIFIED` relative height; `z=1.850 m` is the stated coordinate translation. The paired CFD study's `3.0 m` computational height is not a longer physical riser. |
| Upstream head | Table 2 gives B-H3 `H0=0.66 m`; notation defines Series-B `H0` from the tunnel/pipe invert (p. 13). | `0.66 m` above pipe invert | `VERIFIED`; `0.88 m` belongs to other runs. |
| Initial riser water level | Series B has the riser depth at the same level as the constant-head tank (p. 3); B-H3 has `H0=0.66 m`. | model free surface at `z=0.660 m`; non-overlap water-column height above the crown is `0.610 m` | `VERIFIED` relative level plus coordinate translation. |
| Riser-top condition | The paper defines a geyser as air-water mixture ejection through the riser top (p. 7), implying communication with the laboratory atmosphere, but does not prescribe a CFD pressure boundary. | physical rim open into an expanded external-air volume; pressure boundary is placed on the remote sides/top | `MODEL TRANSLATION`; external-domain dimensions and pressure treatment come from the paired CFD study and domain controls. |
| Valve opening time | Manual operation takes approximately `0.2 s` (p. 3); no aperture history is reported. | experiment `~0.2 s`; paired-CFD baseline instantaneous; `0.5 s` upper sensitivity from the paired paper | `PARTIAL`; all three are reported separately and no ramp is claimed to reconstruct the hand motion. |
| Pressure measurement positions | PT1 is at the pipe crown near the pipe end and measures air pressure; PT2 is at the pipe invert at the riser bottom (p. 4 and Fig. 1). | PT1: downstream-end crown near model `x=6.59 m`; PT2: invert beneath the riser at model `x=3.47 m` | `VERIFIED` physical locations; placing probes one local cell inward is a numerical sampling choice. |

## B-H3 row independently verified

| Quantity | Primary-paper value |
|---|---:|
| Run | `B-H3` |
| Horizontal-pipe ID, `D` | `0.050 m` |
| Riser ID, `Dr` | `0.026 m` |
| `Dr/D` | `0.52` |
| `H0` | `0.66 m` |
| `L0` | `0.61 m` |
| `Vair*` | `3.42` in Table 2 (`3.4181` from unrounded geometry) |
| Air-arrival time, `Ta` | `8.18 s` |
| `Uf/sqrt(gD)` | `0.438` |
| Mean free-surface speed, `vfs` | `0.657 m/s` |
| Mean vertical-interface speed, `vint` | `0.916 m/s` |
| Mean net interface speed, `vnet` | `0.267 m/s` |
| Taylor-bubble speed, `vTaylor` | `0.174 m/s` |
| Experimental classification | `GEYSER` |

The classification is a validation target, not a tuning input. B-H3 is on the
reported criterion boundary: `Dr/D=0.52` is on the geyser side, while its
unrounded `Vair*=3.418056` rounds to the paper threshold `3.42`. Any eventual
base/refined, time-step, interface-compression/diffusion, gas-mass, and
valve-opening sensitivities must report whether this classification changes.

## Conflict disposition

| Conflict | Primary-paper disposition |
|---|---|
| `6.6 m` versus `6.0 m` | Fig. 1’s dimension chain gives `6.59 m`; the prose's “approximately 6 m” is compatible with that chain, so nominal `6.6 m` is retained. |
| T at `x=3.47 m` versus `x=2.88 m` | Fig. 1 gives `x=3.47 m` from Valve #1; `2.88 m` is rejected. |
| Physical riser `1.8 m` versus CFD extension `3.0 m` | The experiment is `1.8 m`; the paired CFD study explicitly uses `3.0 m` computational height. This Case keeps the rim at `1.85 m` absolute elevation and expands into an external atmospheric volume up to `3.0 m`. |
| `H0=0.66 m` versus `0.88 m` | Table 2 fixes B-H3 at `0.66 m`; `0.88 m` is rejected. |

## Analytic initial-volume and air-mass targets

These are pre-mesh analytic targets, not mesh-integrated audit results.

| Quantity | Formula / assumption | Target |
|---|---|---:|
| Initial pocket volume | `π (0.050)² (0.61) / 4` | `0.0011977322 m³` = `1.1977322 L` |
| Paper-normalized riser water volume | `π (0.026)² (0.66) / 4` | `0.0003504132 m³` = `0.3504132 L` |
| `Vair/Vw` | ratio of the preceding paper definitions | `3.418056` |
| Non-overlap modeled riser water volume | `π (0.026)² (0.66-0.05) / 4`; excludes water already counted in the horizontal pipe | `0.0003238668 m³` = `0.3238668 L` |
| Laboratory temperature | measured in paper | `23 °C` = `296.15 K` |
| Water density | measured-temperature value in paper | `998 kg/m³` |
| Surface tension | stated in paper | `0.072 N/m` |
| Reference ideal-gas air mass | `pV/(Rair T)`, using `101325 Pa` and OpenFOAM `Rair=8314.46261815324/28.966 J/(kg K)` | `0.001427641 kg` |

The paired CFD paper defines the operating/atmospheric pressure as
`101.325 kPa`, so the reference air mass is released as the numerical target.

## Released numerical controls

| Required input | Traceable decision | Audit status |
|---|---|---|
| Gas compressibility | Ideal-gas air at `101325 Pa`; paired CFD Eqs. (2)–(4). | `RESOLVED`. |
| Valve process | Published paired-CFD baseline is instantaneous. Sensitivities use `0.2 s` (2017 experiment) and `0.5 s` (2018 paper statement) porous-baffle ramps; their smooth aperture law is explicitly numerical. | `RESOLVED` by sensitivity, with no claim that the ramp reconstructs the hand motion. |
| Wall contact angle | Neither paper enables or reports a wall-adhesion law. Reproduction therefore uses a neutral static `90 deg` condition (equivalent to no preferential wetting) and records it explicitly. | `NUMERICAL CONTROL`; non-calibrated. |
| Initial VOF transition | The measured free surface is at `H0`; neither paper defines a numerical interface thickness. A symmetric `15 mm` (three base-riser cells) linear transition preserves the analytic phase volume and passed the water-side startup-current gate. | `NUMERICAL CONTROL`; transported-interface compression is varied independently, and this width is identical in the BH3/BH4 contract. |
| External air domain | Physical rim stays at `z=1.850 m`; expanded atmosphere reaches the paired CFD total height `z=3.0 m`. Width is parameterized for domain-independence checks. | `RESOLVED` as a numerical-domain control. |
| Absolute ambient pressure | Paired CFD operating pressure `101.325 kPa`, applied at `z=H0`; the connected open-air column follows the isothermal ideal-gas hydrostatic profile implied by gravity and the selected EOS. | `PAIRED-CFD INPUT`; the vertical correction prevents an initially uniform-pressure gas column from entering gravitational free fall. |
| PT1/PT2 sampling coordinates | Main-paper physical locations are retained; numerical probes are one local cell inside the fluid. | `NUMERICAL TRANSLATION`. |
| Wall roughness | Paired CFD assumes smooth wall with roughness length `10^-3 mm`. | `RESOLVED`. |
| Initial temperature | Main experiment measured `296.15 K`; paired CFD used `300 K`. Experimental value is retained and the difference documented. | `RESOLVED`. |
| BH4 parity | `MODELING_CONTRACT.json` freezes every independent input and numerical control; `riser_diameter_m` is the sole permitted Case variable. | `RESOLVED` locally; the independent BH4 implementation must verify the contract hash. |

The gate release authorizes mesh and solver work. It does not pre-authorize a
“validated” label: every required check must still be run, and failures or
incomplete event windows must be reported without tuning to force a geyser.

## Implemented 3-D model cross-check

The source geometry and dictionaries were rechecked directly against Fig. 1,
Table 2, and the experimental procedure:

- `make_geometry.py` uses a `6.590 m` long, `0.050 m` ID circular main, a
  `0.026 m` ID circular riser on the `x=3.470 m` axis, a conformal valve plane
  at `x=5.980 m`, and a closed cap at `x=6.590 m`.
- The vertical construction cylinder overlaps the main to its centreline before
  the OpenCASCADE Boolean union. This buried overlap creates the full
  three-dimensional intersecting T passage; it is not exposed riser length.
  The exposed physical riser remains exactly `1.850-0.050=1.800 m`.
- The initial downstream full-bore pocket spans the last `0.610 m`; the main
  and riser below `z=0.660 m` are water-filled. The upstream boundary is a
  constant-head water boundary, the cap is a no-flow wall, and the remote
  external-air boundary is atmospheric.
- `compressibleInterFoam` uses ideal-gas air, weakly compressible water, no
  artificial pressure/velocity source, and explicit mass/air-mass audits.

This source-level check is a geometry/condition consistency result, not a
validation result. Static hold, full event-window, mesh/time-step/interface,
valve-opening, and conservation gates still control acceptance.
