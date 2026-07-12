# Cong, Chan & Lee (2017) paper audit — Series B, run B-H6

## Scope and gate

This audit is for the single Case `BH6_Dr41_H066_L061`.  The primary source is
`references/cong2017.pdf` (Cong, Chan & Lee, 2017, JHE 143(9), 04017039).
Page references below are printed journal pages.  Figure dimensions take
precedence over rounded prose, and no value from another run is substituted.

The two tracked copies, `references/cong2017.pdf` and
`tests/test_02_cong2017/_shared/reference/paper_source/cong2017_JHE2017_offprint.pdf`,
are byte-identical (SHA-256
`6a2fd77ae65f6361ec5479780a3226e5f5cf69fde643866f692279588c16aa3e`).
This audit was repeated directly from that PDF, not inferred from repository
summaries.

**B-H6 experimental-input gate: PASS.**  Every paper-defined experimental
input required to build B-H6 is resolved.  The paper does not give a numerical
offset for either pressure transducer; their unambiguous physical locations
are therefore used and the mesh-dependent sampling offset must be reported
rather than presented as an experimental coordinate.

**Quantitative-production gate: CONDITIONAL.**  The 2017 paper is experimental
and does not prescribe an OpenFOAM discretization, contact angle, air
properties, or turbulence closure.  Its companion CFD paper (Chan, Cong &
Lee 2018, DOI `10.1061/(ASCE)HY.1943-7900.0001416`) supplies useful numerical
evidence but used ANSYS Fluent, standard `k-epsilon`, geometric VOF, and a
different numerical outlet extension.  The OpenFOAM mesh, turbulence and
interface choices must therefore be reported and tested rather than described
as paper inputs.

**Existing-code B-H1 pairing gate: NOT VERIFIABLE.**  The repository has no
`BH1_Dr16_H066_L061/openfoam/3d` case.  Its legacy 1-D model uses a rounded
`6.0 m` pipe and `x=2.88 m` tee, which conflict with the primary-paper
dimensions resolved below and therefore cannot serve as the requested 3-D
baseline.  This directory defines the common paper-audited 3-D baseline: a
future B-H1 3-D case must differ only in `Dr` and geometry/mesh entities
directly dependent on `Dr`.  Until that counterpart exists, “same as BH1”
is a design contract, not a completed source-to-source verification.

## Coordinate convention

- Main-pipe axis is the `x` axis; `x=0` is Valve #1 / the upstream-tank end of
  the acrylic test pipe and positive `x` points toward the downstream cap.
- Main-pipe centreline elevation is `z=0`; pipe invert and soffit (crown) are
  `z=-0.025 m` and `z=+0.025 m`, respectively.
- The riser axis intersects the main-pipe axis at `x=3.47 m`.
- `H0` is measured from the horizontal-pipe invert.  The image-derived
  `Yfs`/`Yint` trajectories are different: the measurement procedure on p.4
  explicitly measures their distance from the riser entrance, i.e. the pipe
  soffit.  The two vertical datums differ by `D=0.050 m`.

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
| Upstream head | Table 2, B-H6 row: `H0=0.66 m`; notation on p.12 defines Series-B `H0` from tunnel invert | constant free-surface elevation `z=-0.025+0.66=0.635 m`, equivalent to `0.66 m` water head above invert | RESOLVED |
| Initial riser water level | p.3: for Series B the initial riser water depth is at the same level as the constant-head tank; p.4 defines image trajectories from the riser entrance | physical level `z=0.635 m`, equal to `0.66 m` above invert and `0.61 m` above the riser entrance; digitised Figure-7 trajectories retain the latter datum | RESOLVED |
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
4. **`H0=0.66 m` versus `0.88 m`:** Table 2 gives B-H6 as `0.66 m`.
   `0.88 m` belongs to other runs (for example B-11/B-42) and is rejected.

## Remaining paper-defined Case inputs

- Main-pipe inside diameter: `D=0.050 m` (p.2).
- B-H6 riser inside diameter: `Dr=0.041 m` (Table 2).
- B-H6 diameter ratio: `Dr/D=0.82` (Table 2).
- Initial pocket length: `L0=0.61 m` (Table 2).
- Initial nominal pocket volume:
  `pi D^2 L0 / 4 = 0.00119773 m^3 = 1.19773 L`.
- Initial dimensionless pocket volume: `Vair*=1.37` in Table 2
  (`1.3745` using unrounded inputs).
- Initial pocket pressure: atmospheric (p.3 procedure).
- Laboratory temperature: `23 degC`; paper water density `998 kg/m^3` (p.4).
- Riser top: open to the atmosphere; the external-air volume continues above
  the physical rim so expelled water is retained and measured.
- B-H6 measured comparison targets from Table 2 are `Ta=8.10 s`,
  `Uf/sqrt(gD)=0.443`, `vfs=0.246 m/s`, `vint=0.476 m/s`,
  `vnet=0.235 m/s`, `vTaylor=0.219 m/s`, `Dr/D=0.82`,
  `Vair/Vw=1.37`, and no geyser observed.  Fig. 6 shows the bubble catching
  the free surface at about `10.5–10.9 s`; Fig. 7 shows the free surface rising
  from about `0.58 m` to `1.21 m`, both measured above the riser entrance.
  Table 2's quantitative `Ta=8.10 s` takes precedence over the prose estimate
  “approximately 8.6 s,” which describes visual arrival in the image sequence.
  Fig. 10(b) is Run B-32, not B-H6; it has the same nominal `Dr/H0/L0` but
  different measured velocities and may only be used as a labelled repeat-run
  pressure proxy.
  These are validation outputs, never forcing or calibration inputs.

## Direct as-built model cross-check

| Experimental feature | OpenFOAM representation | Disposition |
|---|---|---|
| Circular `D=0.050 m` pipe, `Dr=0.041 m` riser and true T-junction | OCC cylinders joined at `x=3.47 m`; no wedge or equivalent rectangle | MATCH |
| Dimension chain `3.47 m + 3.12 m` | Active pipe `x=0…6.59 m` | MATCH |
| Selected valve and `L0=0.61 m` atmospheric pocket | Valve plane `x=5.98 m`; gas to capped wall at `x=6.59 m` | MATCH |
| Constant-head upstream tank | Fixed-head pressure/water inlet at the test-pipe entrance | PHYSICALLY EQUIVALENT BOUNDARY; tank volume and entrance geometry omitted |
| Initially still water to `H0=0.66 m` above invert | `U=0`; free surface `z=0.635 m`; discrete hydrostatic `p/p_rgh` | MATCH |
| Initially atmospheric trapped air | Ideal-gas pocket referenced to `101325 Pa` at pipe centreline | MATCH |
| Physical `1.8 m` open riser | Circular wall to `z=1.825 m`, then an explicitly separate external-air domain | MATCH AT RIM; external-domain dimensions are numerical |
| Manual quarter-turn ball valve, approximately `0.2 s` | Zero-thickness variable-area `cyclicACMI`, smooth monotone `0.2 s` area law | DURATION MATCH; unmeasured opening law is a sensitivity |
| PT1 crown near cap; PT2 invert under riser | `(6.56,0,0.022)` and `(3.47,0,-0.022)` probes | TOPOLOGY MATCH; offsets are numerical proxies |
| Acrylic walls | Smooth no-slip walls; neutral `90 deg` contact angle and adiabatic heat flux | NO-SLIP MATCH; wetting/thermal assumptions unmeasured |

The actual tank, the three non-selected open valve bodies, and the selected
ball geometry are not resolved.  The resulting model is an experimentally
matched active test section with equivalent hydraulic boundaries, not a CAD
replica of every laboratory component.

## Companion 3-D CFD evidence and OpenFOAM differences

Chan, Cong & Lee (2018) used about `100,000` boundary-fitted cells for selected
fine simulations: about 25 cells across the main pipe, about 50 across the
riser, a minimum near-wall riser cell of `0.1 mm`, hexahedra away from a
tetrahedral T-junction, standard `k-epsilon`, second-order upwind transport,
geometric VOF reconstruction, and first-order implicit time marching.  Their
parametric mesh used about `40,000` cells.  They emphasized that resolving the
`0.6–1.2 mm` falling water film is important.

This case deliberately differs where the requested model differs:

- it uses open-source `compressibleInterFoam` with ideal-gas air;
- it retains the physical `1.8 m` riser and adds an external atmosphere,
  instead of treating the numerical `3.0 m` height as a longer closed riser;
- it resolves the measured `0.2 s` valve motion and brackets its duration,
  instead of opening instantaneously;
- it uses the measured `23 degC` rather than the companion model's assumed
  `300 K`.

Mesh and turbulence sensitivity are model-fidelity gates.  A coarse result may
screen stability, but it cannot by itself validate Taylor-bubble film dynamics.

## Explicit limitations that do not alter the apparatus definition

- The paper does not report a numerical distance from PT1 to the cap.  Any CFD
  probe coordinate claiming such a measured distance would be invented.
- The `3.0 m` external-domain top is a numerical containment boundary, not a
  physical dimension reported in the 2017 experiment.  Results must show that
  this far-field boundary does not control the event.
- The paper reports an approximately `0.2 s` manual valve motion but not its
  angle-versus-time trace.  The adopted monotone opening law and its timing
  sensitivity must therefore be documented.
