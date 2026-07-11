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
and explicitly supplies the absolute atmospheric pressure, numerical valve
baseline, computational height, wall roughness, pressure boundaries, PT1
location, gas law, and grid strategy. Values absent from both papers remain
declared numerical controls and are subjected to sensitivity checks; they are
not presented as measurements.

## Required checks against the primary paper

Coordinates below use the Valve #1/upstream-pipe plane as `x = 0` and the
horizontal-pipe invert as `z = 0`.

| Item | Primary-paper evidence | Audited B-H3 value | Status / consequence |
|---|---|---:|---|
| Horizontal-pipe physical total length | The prose says “approximately 6 m” (p. 2). Fig. 1 gives `3.47 m` from Valve #1 to the T axis and `(3.12 m - L0) + L0 = 3.12 m` from the T axis to the cap. | `6.59 m` (nominal `6.6 m`) | `VERIFIED`; the prose’s `6 m` is rounded, not a second exact dimension. |
| OpenFOAM effective modeled length | Fig. 1 gives the complete internal-pipe dimension chain. Series B connects Valve #1 to a constant-head tank and closes the downstream cap. | `6.59 m` from an upstream constant-head boundary at `x=0` to the cap | `VERIFIED` for a tank-as-boundary representation; the tank volume is not included. |
| T-junction position | Fig. 1 explicitly labels Valve #1-to-riser-axis distance `3.47 m`. | `xT = 3.47 m` | `VERIFIED`; repository candidate `x=2.88 m` is rejected. |
| Ball-valve position | Fig. 1 labels T-to-selected-valve distance `3.12 m - L0` and selected-valve-to-cap distance `L0`. Table 2 gives B-H3 `L0=0.61 m`, corresponding to Valve #4. | release plane `x = 3.47 + 3.12 - 0.61 = 5.98 m` | `VERIFIED`; the paired CFD paper explicitly uses instantaneous numerical opening, with finite opening treated here only as sensitivity. |
| Initial air-pocket position | The downstream section is emptied and capped, leaving atmospheric air (pp. 2–3); Table 2 gives `L0=0.61 m`. | `5.98 <= x <= 6.59 m`, full circular section, initially atmospheric | `VERIFIED`. |
| Closed-end position | Fig. 1 places the removable cap `L0` downstream of the selected valve; the procedure closes it with a plastic cap (pp. 2–3). | cap at `x=6.59 m` | `VERIFIED` as a no-flow wall. |
| Physical riser height | The apparatus has a `1.8 m` riser (p. 2), measured from the horizontal-pipe soffit (p. 7). | `1.8 m` above the pipe crown; top at `z=1.850 m` | `VERIFIED`; the paired CFD study extends the computational height to `3.0 m`. This model preserves the physical rim and represents the remaining height as an expanded external-air domain. |
| Upstream head | Table 2 gives B-H3 `H0=0.66 m`; notation defines Series-B `H0` from the tunnel/pipe invert (p. 13). | `0.66 m` above pipe invert | `VERIFIED`; `0.88 m` belongs to other runs. |
| Initial riser water level | Series B has the riser depth at the same level as the constant-head tank (p. 3); B-H3 has `H0=0.66 m`. | free surface at `z=0.660 m`; non-overlap water-column height above the crown is `0.610 m` | `VERIFIED`. |
| Valve opening time | Manual operation takes approximately `0.2 s` (p. 3). | experiment `~0.2 s`; paired CFD baseline instantaneous; upper sensitivity `0.5 s` from the paired paper | `VERIFIED` as a three-level sensitivity, not as a reconstructed unreported aperture trace. |
| Pressure measurement positions | PT1 is at the pipe crown near the pipe end and measures air pressure; PT2 is at the pipe invert at the riser bottom (p. 4 and Fig. 1). The paired CFD paper calls these Point A “at the soffit of the downstream end” and Point B “at the invert level just beneath the vertical riser.” | PT1: downstream-end crown at `x=6.59 m`; PT2: invert at `x=3.47 m` | `VERIFIED`; probes are placed one local cell inward from the walls. |

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
| Mean free-surface speed, `vfs` | `0.657 m/s` |
| Mean vertical-interface speed, `vint` | `0.916 m/s` |
| Experimental classification | `GEYSER` |

The classification is a validation target, not a tuning input. B-H3 is on the
reported criterion boundary: `Dr/D=0.52` is on the geyser side, while its
unrounded `Vair*=3.418056` rounds to the paper threshold `3.42`. Any eventual
base/refined, time-step, interface-compression/diffusion, gas-mass, and
valve-opening sensitivities must report whether this classification changes.

## Conflict disposition

| Conflict | Primary-paper disposition |
|---|---|
| `6.6 m` versus `6.0 m` | Fig. 1’s dimension chain gives `6.59 m`; nominal `6.6 m` is retained. |
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
| Laboratory temperature | measured in paper | `23 °C` = `296.15 K` |
| Water density | measured-temperature value in paper | `998 kg/m³` |
| Surface tension | stated in paper | `0.072 N/m` |
| Reference ideal-gas air mass | `pV/(Rair T)`, using `101325 Pa` and `Rair=287.05 J/(kg K)` | `0.001427602 kg` |

The paired CFD paper defines the operating/atmospheric pressure as
`101.325 kPa`, so the reference air mass is released as the numerical target.

## Released numerical controls

| Required input | Traceable decision | Audit status |
|---|---|---|
| Gas compressibility | Ideal-gas air at `101325 Pa`; paired CFD Eqs. (2)–(4). | `RESOLVED`. |
| Valve process | Published paired-CFD baseline is instantaneous. Sensitivities use `0.2 s` (2017 experiment) and `0.5 s` (2018 paper statement) porous-baffle ramps; their smooth aperture law is explicitly numerical. | `RESOLVED` by sensitivity, with no claim that the ramp reconstructs the hand motion. |
| Wall contact angle | Neither paper enables or reports a wall-adhesion law. Reproduction therefore uses a neutral static `90 deg` condition (equivalent to no preferential wetting) and records it explicitly. | `RESOLVED` as a non-calibrated numerical control. |
| External air domain | Physical rim stays at `z=1.850 m`; expanded atmosphere reaches the paired CFD total height `z=3.0 m`. Width is parameterized for domain-independence checks. | `RESOLVED` as a numerical-domain control. |
| Absolute ambient pressure | Paired CFD operating pressure `101.325 kPa`. | `RESOLVED`. |
| PT1/PT2 | Paired CFD Point A at downstream-end soffit; Point B at invert beneath riser. | `RESOLVED`. |
| Wall roughness | Paired CFD assumes smooth wall with roughness length `10^-3 mm`. | `RESOLVED`. |
| Initial temperature | Main experiment measured `296.15 K`; paired CFD used `300 K`. Experimental value is retained and the difference documented. | `RESOLVED`. |
| BH4 parity | `MODELING_CONTRACT.json` freezes every independent input and numerical control; `riser_diameter_m` is the sole permitted Case variable. | `RESOLVED` locally; the independent BH4 implementation must verify the contract hash. |

The gate release authorizes mesh and solver work. It does not pre-authorize a
“validated” label: every required check must still be run, and failures or
incomplete event windows must be reported without tuning to force a geyser.
