# PAPER AUDIT — Cong, Chan & Lee (2017), Series B, Run B-H3

## Scope and hard stop

This audit applies only to `BH3_Dr26_H066_L061`. The primary source is
`references/cong2017.pdf`; repository notes and existing reduced-order outputs
are secondary candidates and cannot settle a disagreement with the paper.

- `VERIFIED`: unambiguously stated or dimensioned in the primary paper.
- `PARTIAL`: the paper constrains the item but omits a CFD-relevant detail.
- `UNRESOLVED`: no unambiguous model value can be obtained from the paper.

**Gate: STOP.** The original task forbids geometry generation or solution while
any model-affecting input remains `UNRESOLVED`. The primary paper does not
report the valve aperture history, acrylic contact angle, external-air-domain
extent, measured absolute atmospheric pressure, or exact PT1 coordinate.
Consequently no mesh or solver dictionaries are created after this audit.

## Required checks against the primary paper

Coordinates below use the Valve #1/upstream-pipe plane as `x = 0` and the
horizontal-pipe invert as `z = 0`.

| Item | Primary-paper evidence | Audited B-H3 value | Status / consequence |
|---|---|---:|---|
| Horizontal-pipe physical total length | The prose says “approximately 6 m” (p. 2). Fig. 1 gives `3.47 m` from Valve #1 to the T axis and `(3.12 m - L0) + L0 = 3.12 m` from the T axis to the cap. | `6.59 m` (nominal `6.6 m`) | `VERIFIED`; the prose’s `6 m` is rounded, not a second exact dimension. |
| OpenFOAM effective modeled length | Fig. 1 gives the complete internal-pipe dimension chain. Series B connects Valve #1 to a constant-head tank and closes the downstream cap. | `6.59 m` from an upstream constant-head boundary at `x=0` to the cap | `VERIFIED` for a tank-as-boundary representation; the tank volume is not included. |
| T-junction position | Fig. 1 explicitly labels Valve #1-to-riser-axis distance `3.47 m`. | `xT = 3.47 m` | `VERIFIED`; repository candidate `x=2.88 m` is rejected. |
| Ball-valve position | Fig. 1 labels T-to-selected-valve distance `3.12 m - L0` and selected-valve-to-cap distance `L0`. Table 2 gives B-H3 `L0=0.61 m`, corresponding to Valve #4. | release plane `x = 3.47 + 3.12 - 0.61 = 5.98 m` | `PARTIAL`; plane is verified, but ball/bore geometry and aperture history are not reported. |
| Initial air-pocket position | The downstream section is emptied and capped, leaving atmospheric air (pp. 2–3); Table 2 gives `L0=0.61 m`. | `5.98 <= x <= 6.59 m`, full circular section, initially atmospheric | `VERIFIED`. |
| Closed-end position | Fig. 1 places the removable cap `L0` downstream of the selected valve; the procedure closes it with a plastic cap (pp. 2–3). | cap at `x=6.59 m` | `VERIFIED` as a no-flow wall. |
| Physical riser height | The apparatus has a `1.8 m` riser (p. 2), measured from the horizontal-pipe soffit (p. 7). | `1.8 m` above the pipe crown; top at `z=1.850 m` | `VERIFIED`; `3.0 m` may only denote an external CFD extension and cannot replace the physical riser. |
| Upstream head | Table 2 gives B-H3 `H0=0.66 m`; notation defines Series-B `H0` from the tunnel/pipe invert (p. 13). | `0.66 m` above pipe invert | `VERIFIED`; `0.88 m` belongs to other runs. |
| Initial riser water level | Series B has the riser depth at the same level as the constant-head tank (p. 3); B-H3 has `H0=0.66 m`. | free surface at `z=0.660 m`; non-overlap water-column height above the crown is `0.610 m` | `VERIFIED`. |
| Valve opening time | Manual operation takes approximately `0.2 s` (p. 3). | nominal duration `0.2 s` | `PARTIAL`; aperture-versus-time law is not reported, so required opening-time sensitivity alone cannot identify the physical law. |
| Pressure measurement positions | PT1 is at the pipe crown near the pipe end and measures air pressure; PT2 is at the pipe invert at the riser bottom (p. 4 and Fig. 1). | PT1: crown in the final `L0` section near the cap; PT2: invert at `x=3.47 m` | PT2 `VERIFIED`; PT1 axial coordinate and tap geometry `UNRESOLVED`. |

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
| Physical riser `1.8 m` versus CFD extension `3.0 m` | Physical riser is `1.8 m` from the soffit. No external extension is paper-dimensioned. |
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

The pocket is stated to be atmospheric, but the experiment’s barometric
pressure is not reported. Therefore `101325 Pa` and the corresponding air mass
are explicitly a standard-atmosphere reference, not a verified experimental
absolute mass. Mesh-integrated phase volumes and gas-mass conservation cannot
be audited without a released model.

## Model-affecting unresolved inputs

| Required input | What the paper establishes | Status |
|---|---|---|
| Valve geometry and aperture law | Quarter-turn PVC ball valve; manual operation approximately `0.2 s`. | `UNRESOLVED`; neither bore geometry nor aperture versus time is given. |
| Wall contact angle | Pipe and riser are acrylic. | `UNRESOLVED`; static/dynamic contact angle and surface preparation are absent. |
| External air domain | Riser top is open to laboratory atmosphere. | `UNRESOLVED`; no far-field distance or enclosure geometry is given. |
| Absolute ambient pressure | Pocket initially at atmospheric pressure. | `UNRESOLVED` as a measured absolute value; gauge zero is verified. |
| PT1 coordinate | Crown near the pipe end. | `UNRESOLVED` for registered quantitative pressure comparison. |
| BH4 parity baseline | The requested paired model must differ only in `Dr`; no BH4 3-D source case exists at the specified base commit. | `UNRESOLVED` until one common modeling contract is supplied or reviewed across both independent branches. |

To release the gate, provide traceable values for the valve law, contact angle,
ambient absolute pressure, and PT1 coordinate, plus either a justified external
domain-independence protocol and a shared BH3/BH4 modeling contract or the
corresponding measured dimensions. Until then, `checkMesh`, closed-valve hold,
open-valve smoke, 13 s event, sensitivity runs, and 3-D comparisons must not be
represented as completed or paper-validated.
