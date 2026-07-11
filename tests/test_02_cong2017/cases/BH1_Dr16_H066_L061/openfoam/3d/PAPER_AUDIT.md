# Cong, Chan & Lee (2017) paper audit — Series B, run B-H1

## Scope and gate

This audit is for the single Case `BH1_Dr16_H066_L061`.  The primary source is
`references/cong2017.pdf` (Cong, Chan & Lee, 2017, JHE 143(9), 04017039).
Page references below are printed journal pages.  Figure dimensions take
precedence over rounded prose, and no value from another run is substituted.

**Formal-mesh gate: PASS.**  Every model-defining item requested below is
resolved.  The paper does not give a numerical offset for either pressure
transducer; their unambiguous physical locations are therefore used and the
mesh-dependent sampling offset must be reported rather than presented as an
experimental coordinate.

## Coordinate convention

- Main-pipe axis is the `x` axis; `x=0` is Valve #1 / the upstream-tank end of
  the acrylic test pipe and positive `x` points toward the downstream cap.
- Main-pipe centreline elevation is `z=0`; pipe invert and soffit (crown) are
  `z=-0.025 m` and `z=+0.025 m`, respectively.
- The riser axis intersects the main-pipe axis at `x=3.47 m`.
- Heads and vertical interface levels reported by the paper are measured from
  the horizontal-pipe invert unless explicitly stated otherwise.

## Item-by-item verification

| Required item | Paper evidence | Resolved value used by the 3-D model | Status |
|---|---|---|---|
| Horizontal-pipe physical total length | p.2 text says “approximately 6 m”; Fig. 1 gives `3.47 m` from Valve #1 to the tee and `3.12 m` from tee to cap | `3.47 + 3.12 = 6.59 m` (reported as about `6.6 m`) | RESOLVED |
| OpenFOAM effective modeled length | Fig. 1 contains the complete active test section from the upstream constant-head connection to the capped end | `x=0…6.59 m`; no truncation to `6.0 m` | RESOLVED |
| Tee position | Fig. 1 dimension from Valve #1 to vertical-riser axis | `x_T=3.47 m` | RESOLVED |
| Ball-valve position for `L0=0.61 m` | Fig. 1 partitions the tee-to-cap distance into `3.12 m - L0` and `L0`; procedure on p.3 says water is upstream and atmospheric air is between the selected valve and capped end | selected Valve #4 at `x=6.59-0.61=5.98 m` | RESOLVED |
| Initial air-pocket position | Fig. 1 and p.3 procedure | `x=5.98…6.59 m`, full circular pipe section, initially atmospheric | RESOLVED |
| Closed-end position | Fig. 1 cap at the end of the `3.12 m` tee-to-end dimension | downstream cap plane `x=6.59 m` | RESOLVED |
| Physical riser height | pp.2 and 8: `1.8 m`; p.8 explicitly says measured from the soffit of the horizontal pipe | physical rim is `1.8 m` above pipe soffit, hence `z=1.825 m` in the model coordinates | RESOLVED |
| Upstream head | Table 2, B-H1 row: `H0=0.66 m`; notation on p.12 defines Series-B `H0` from tunnel invert | constant free-surface elevation `z=-0.025+0.66=0.635 m`, equivalent to `0.66 m` water head above invert | RESOLVED |
| Initial riser water level | p.3: for Series B the initial riser water depth is at the same level as the constant-head tank | `Yfs(0)=H0=0.66 m` above invert (`z=0.635 m`) | RESOLVED |
| Valve opening time | p.3: manual opening takes approximately `0.2 s` | base opening duration `0.20 s`; sensitivity cases must bracket it and may not replace it with fitted timing | RESOLVED |
| Pressure measurement positions | p.4 and Fig. 1: PT1 at pipe crown near the capped pipe end; PT2 at pipe invert directly below the riser | PT1 sampled in the crown-adjacent cell nearest the cap; PT2 sampled in the invert-adjacent cell at `x=3.47 m`; exact cell centres are emitted with results | RESOLVED TO PAPER PRECISION |

## Conflict disposition

1. **`6.6 m` versus `6.0 m`:** Fig. 1 resolves the dimensioned active length as
   `6.59 m`; “approximately 6 m” is rounded prose.  The model uses `6.59 m`.
2. **Tee `x=3.47 m` versus `x=2.88 m`:** Fig. 1 directly dimensions
   Valve-#1-to-tee as `3.47 m`.  The `2.88 m` value results from forcing the
   rounded `6.0 m` total while retaining `3.12 m` downstream and is rejected.
3. **Physical riser `1.8 m` versus CFD extension `3.0 m`:** the acrylic riser
   remains `1.8 m` above the pipe soffit.  The numerical external-air domain
   may extend to `3.0 m` above the soffit (`z=3.025 m`), but that volume is
   outside the riser and must not be represented as a longer pipe.
4. **`H0=0.66 m` versus `0.88 m`:** Table 2 gives B-H1 as `0.66 m`.
   `0.88 m` belongs to other runs (for example B-11/B-42) and is rejected.

## Remaining paper-defined Case inputs

- Main-pipe inside diameter: `D=0.050 m` (p.2).
- B-H1 riser inside diameter: `Dr=0.016 m` (Table 2).
- Initial pocket length: `L0=0.61 m` (Table 2).
- Initial nominal pocket volume:
  `pi D^2 L0 / 4 = 0.00119773 m^3 = 1.19773 L`.
- Initial pocket pressure: atmospheric (p.3 procedure).
- Laboratory temperature: `23 degC`; paper water density `998 kg/m^3` (p.4).
- Riser top: open to the atmosphere; the external-air volume continues above
  the physical rim so expelled water is retained and measured.
- B-H1 measured comparison targets from Table 2 are `Ta=8.07 s`,
  `vfs=0.924 m/s`, `vint=1.231 m/s`, and geyser observed.  These are validation
  outputs, never forcing or calibration inputs.

## Explicit limitations that do not alter the apparatus definition

- The paper does not report a numerical distance from PT1 to the cap.  Any CFD
  probe coordinate claiming such a measured distance would be invented.
- The `3.0 m` external-domain top is a numerical containment boundary, not a
  physical dimension reported in the 2017 experiment.  Results must show that
  this far-field boundary does not control the event.
- The paper reports an approximately `0.2 s` manual valve motion but not its
  angle-versus-time trace.  The adopted monotone opening law and its timing
  sensitivity must therefore be documented.
