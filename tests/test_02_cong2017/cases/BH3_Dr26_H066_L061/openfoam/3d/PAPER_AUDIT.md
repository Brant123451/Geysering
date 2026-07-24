# PAPER AUDIT — Cong, Chan & Lee (2017), Series B, Run B-H3

## Scope and audit gate

This audit applies only to `BH3_Dr26_H066_L061`. The primary source is
`references/cong2017.pdf`; repository notes and existing reduced-order outputs
are secondary candidates and cannot settle a disagreement with the paper.

- `VERIFIED`: unambiguously stated in the primary-paper text or table.
- `FIGURE-DERIVED`: read from Fig. 1 or derived from its dimension chain.
- `PARTIAL`: the paper constrains the item but omits a CFD-relevant detail.
- `MODEL TRANSLATION`: an explicit mapping from the apparatus to the CFD domain.
- `NUMERICAL CONTROL`: not measured; declared and tested rather than calibrated.
- `UNRESOLVED`: no unambiguous model value can be obtained from the paper.

The primary-paper-only gate is `PARTIAL`: an experiment does not prescribe all
CFD controls. The model-input gate is **RELEASED** only by separately citing the
directly paired study by Chan, Cong & Lee (2018), *3D Numerical Modeling of
Geyser Formation by Release of Entrapped Air from Horizontal Pipe into Vertical
Shaft*, DOI `10.1061/(ASCE)HY.1943-7900.0001416`, and by declaring the remaining
choices as numerical controls. The paired paper uses FLUENT, an incompressible
water phase, ideal-gas air, a 3.0 m computational riser, and an instantaneous
valve opening; none of those choices is relabeled here as a 2017 measurement.
The present OpenFOAM translation and every sensitivity are reported separately.

## Required checks against the primary paper

Fig. 1 dimensions are referenced from the Valve #1/upstream test-pipe plane.
The CFD coordinate convention places its constant-head boundary on that plane
at `x = 0` and uses the horizontal-pipe invert as `z = 0`; absolute coordinates
below are therefore model translations of the paper's relative dimensions.
The embedded Fig. 1 image was also checked directly: it labels `3.47 m` from
Valve #1 to the riser axis, `3.12 m - L0` from that axis to the selected valve,
and `L0` from the selected valve to the cap. Thus B-H3's `L0=0.61 m` selects
Valve #4 and closes the dimension chain without relying on text extraction.

| Item | Primary-paper evidence | Audited B-H3 value | Status / consequence |
|---|---|---:|---|
| Horizontal-pipe physical total length | The prose says “approximately 6 m” (p. 2). Fig. 1 visibly dimensions `3.47 m` from Valve #1 to the T axis and `(3.12 m - L0) + L0 = 3.12 m` from the T axis to the cap. The paired CFD paper calls the pipe `6.6 m`. | `6.59 m` (nominal `6.6 m`) | `FIGURE-DERIVED`; consistent with the rounded prose and independently confirmed by the paired CFD geometry. |
| OpenFOAM effective modeled length | Fig. 1 gives the complete internal test-pipe dimension chain. Series B connects Valve #1 to a constant-head tank and closes the downstream cap. | `6.59 m` from a tank-as-pressure-boundary plane at `x=0` to the cap | `MODEL TRANSLATION`; the tank volume and short upstream plumbing are not modeled. |
| T-junction position | Fig. 1 visibly labels Valve #1-to-riser-axis distance `3.47 m`; the paired CFD paper also states `x=3.47 m` from upstream. | `xT = 3.47 m` in the stated CFD coordinate convention | `FIGURE-DERIVED` in the primary source and independently stated by the paired CFD paper. |
| Ball-valve position | Fig. 1 labels Valves #1–#4, T-to-selected-valve distance `3.12 m - L0`, and selected-valve-to-cap distance `L0`. Table 2 gives B-H3 `L0=0.61 m`, selecting the most-downstream Valve #4. | release plane `x = 3.47 + 3.12 - 0.61 = 5.98 m` | `FIGURE-DERIVED`; the experiment's opening is `~0.2 s`, while instantaneous opening is a paired-CFD numerical baseline. |
| Horizontal-pipe and riser IDs | The apparatus text gives `D=0.050 m` (p. 2); Table 2 gives B-H3 `Dr=0.026 m` and `Dr/D=0.52`. | `D=0.050 m`, `Dr=0.026 m` | `VERIFIED`. |
| Initial water/air partition | The selected valve is closed, its upstream side and riser are filled to the tank level, its downstream side is emptied and capped (p. 3). | water for `x<5.98 m` and in the riser below `z=0.66 m`; full-bore air pocket for `5.98<=x<=6.59 m` | `VERIFIED` partition plus coordinate translation. |
| Initial air-pocket pressure | The procedure explicitly leaves “an air pocket at atmospheric pressure” (p. 3); no absolute numerical value is stated. | hydrostatic ideal-gas pocket referenced to `101325 Pa` at pipe centreline | `VERIFIED` gauge condition; `101325 Pa` is a paired-CFD input. |
| Initial air-pocket position | Table 2 gives `L0=0.61 m`; Fig. 1 and the procedure place it between the selected valve and cap. | `5.98 <= x <= 6.59 m`, full circular section | `FIGURE-DERIVED`. |
| Closed-end position | Fig. 1 places the removable cap `L0` downstream of the selected valve; the procedure closes it with a plastic cap (pp. 2–3). | cap at `x=6.59 m` | `FIGURE-DERIVED`; translated as a no-flow wall. |
| Physical riser height | The apparatus has a `1.8 m` riser (p. 2), measured from the horizontal-pipe soffit/crown (p. 7). | `1.8 m` above the pipe crown; top at model `z=1.850 m` | `VERIFIED` relative height; `z=1.850 m` is the stated coordinate translation. The paired CFD study's `3.0 m` computational height is not a longer physical riser. |
| Upstream head | Table 2 gives B-H3 `H0=0.66 m`; notation defines Series-B `H0` from the tunnel/pipe invert (p. 13). | `0.66 m` above pipe invert | `VERIFIED`; `0.88 m` belongs to other runs. |
| Initial riser water level | Series B has the riser depth at the same level as the constant-head tank (p. 3); B-H3 has `H0=0.66 m`. | model free surface at `z=0.660 m`; non-overlap water-column height above the crown is `0.610 m` | `VERIFIED` relative level plus coordinate translation. |
| Riser-top condition | The paper defines a geyser as air-water mixture ejection through the riser top (p. 7), implying communication with the laboratory atmosphere, but does not prescribe a CFD pressure boundary. | physical rim open into an expanded external-air volume; pressure boundary is placed on the remote sides/top | `MODEL TRANSLATION`; external-domain dimensions and pressure treatment come from the paired CFD study and domain controls. |
| Valve opening time | Manual operation takes approximately `0.2 s` (p. 3); no aperture history is reported. The paired CFD paper instead describes the experiment as approximately `0.5 s` and uses instantaneous opening. | main experiment `~0.2 s`; paired-CFD baseline instantaneous; `0.5 s` discrepancy/sensitivity | `PARTIAL`; all three are reported separately and no ramp is claimed to reconstruct the hand motion. |
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
| `6.6 m` versus `6.0 m` | Fig. 1’s dimension chain gives `6.59 m`; the prose's “approximately 6 m” is rounded and the paired CFD paper states `6.6 m`. |
| T at `x=3.47 m` versus `x=2.88 m` | Fig. 1 gives `3.47 m` from Valve #1 and the paired CFD paper states `x=3.47 m` from its upstream boundary. `2.88 m` may use a different reduced-order origin, but it is not used as the T coordinate under this explicitly stated CFD convention. |
| Physical riser `1.8 m` versus CFD extension `3.0 m` | The experiment is `1.8 m`. The paired CFD study lengthens its confined computational riser to `3.0 m`; this Case does not reinterpret that as apparatus geometry. It keeps the physical rim at `z=1.85 m` and places a declared external-air domain above it to `z=3.0 m`. |
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
| Initial VOF transition | The measured free surface is at `H0`; neither paper defines a numerical interface thickness. A symmetric `15 mm` linear transition preserves the analytic phase volume but has **not yet passed** the 1 s closed-hold gate. A separately labeled zero-thickness diagnostic uses the conformal `z=0.66 m` partition to isolate this band. A symmetric 15 mm cosine diagnostic preserves the same volume and thickness while removing the linear profile's derivative jumps at the band edges. | `NUMERICAL CONTROL`; sharp/diffuse shape, thickness, and transported-interface compression are varied independently. The declared profile must remain identical in the BH3/BH4 contract. |
| Discrete initial pressure balance | Neither paper prescribes a finite-volume pressure initializer. The analytic phase-specific profiles are used as a first guess, then the closed-valve mesh projects `p_rgh` against the same gravity and CSF face operators used by the solver while holding `alpha.water` and `U=0` fixed. | `NUMERICAL CONTROL`; it preserves the isolated pocket reference and adds no runtime pressure or velocity source. Acceptance still requires the closed hold. |
| External air domain | Physical rim stays at `z=1.850 m`; a `0.30 m` wide external atmosphere reaches `z=3.0 m`. The paired CFD study instead uses a confined 3.0 m computational riser. | `MODEL TRANSLATION`; the physical-rim flux and remote-boundary influence must be reported. |
| Absolute ambient pressure | Paired CFD operating pressure `101.325 kPa`, applied at `z=H0`; the connected open-air column follows the isothermal ideal-gas hydrostatic profile implied by gravity and the selected EOS. | `PAIRED-CFD INPUT`; the vertical correction prevents an initially uniform-pressure gas column from entering gravitational free fall. |
| PT1/PT2 sampling coordinates | Main-paper physical locations are retained; numerical probes are one local cell inside the fluid. | `NUMERICAL TRANSLATION`. |
| Wall roughness | Paired CFD assumes smooth wall with roughness length `10^-3 mm`. | `RESOLVED`. |
| Initial temperature | Main experiment measured `296.15 K`; paired CFD used `300 K`. Experimental value is retained and the difference documented. | `RESOLVED`. |
| Surface tension | The primary paper gives `0.072 N/m`; the paired CFD momentum equation does not include a surface-tension term. | Primary-paper `0.072 N/m` remains the baseline; the separately labeled `closed_sigma_zero` diagnostic and `sigma_zero` event sensitivity cannot be relabeled as the experiment baseline. |
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

## Current execution gate

The strict `checkMesh -allGeometry -allTopology` base mesh passes. The completed
`0.02 s` open-valve smoke is numerically executable and conserves sampled mass,
but it is not a static test and cannot pass the closed-hold gate. The attempted
closed holds still generate excessive interface-adjacent velocity; therefore no
`13 s` result is currently accepted or labeled validated. Formal event and
sensitivity runs start only after the closed-hold result is written with
`closed_hold.applicable=true` and `closed_hold.pass=true`.
This is enforced by `run_study.py`: a failed hold is not cache-complete, and
core/sensitivity groups cannot start without a passing current-source hold.
The water and air pressure expressions are hydrostatic in their pure-phase
regions; their alpha-weighted blend through the declared 15 mm transition is
not claimed to be an exact discrete equilibrium. The previously tested
unweighted case-local pressure projection converged its EOS/pressure fixed
point, but its nonconservative reconstructed force residual remained too large
to substitute for the 1 s result. With `sigma=0`, the 15 mm profile reaches a
maximum water-weighted
speed of `0.02387 m/s` by `0.05 s`, above the declared `0.02 m/s` gate.
The conformal sharp-step, `sigma=0` isolation test is substantially worse:
its maximum water-weighted speed is `1.7366 m/s`, maximum domain speed is
`3.2103 m/s`, and the initial reconstructed force residual is
`2198.8` versus `475.5` for the diffuse profile. The sharp profile is
therefore rejected as a static-balance correction, not promoted to a
baseline. On the refined 456,068-cell mesh, the diffuse `sigma=0` diagnostic
reduces the reconstructed residual to `319.3` and the maximum water-weighted
speed over `0.05 s` to `0.01760 m/s`. The speed is still increasing at the
end of that short diagnostic, so it is **not** a 1 s static-hold pass.
Post-processing now requires the full declared `1.0 s` duration as well as
the drift and velocity thresholds before writing `closed_hold.pass=true`.
Restoring the primary-paper surface tension `0.072 N/m` on the same refined
mesh is worse: by `0.05 s` its maximum water-weighted speed is
`0.12539 m/s`, maximum domain speed is `0.93489 m/s`, and reconstructed
initial force residual is `1681.4`. Refinement therefore reduces the
gravity-only projection residual but does not control the nonconservative CSF
residual. The physical-sigma static gate remains failed, so no 13 s event run
is released. The next isolation check keeps the refined mesh, thickness,
surface tension, and all physical inputs fixed and changes only the 15 mm
initial band from linear to a symmetric volume-preserving cosine profile.
That check also fails: its maximum water-weighted speed is `0.12721 m/s`
and reconstructed initial force residual is `2171.2`, versus `0.12539 m/s`
and `1681.4` for the refined linear profile. Removing the band-edge derivative
jumps therefore does not cure the physical-sigma imbalance. Further event
work remains blocked pending a mesh/operator treatment that passes the static
gate without changing the measured physics.

Source inspection of OpenFOAM v2512 identified one formal mismatch in that
initializer: the solver pressure equation projects
`rAUf*(F_sigma + F_g)`, whereas the initializer projected the unweighted face
force. The revised diagnostic uses the startup leading-order approximation
`rAU=deltaT/rho`, reports fixed-point convergence separately from residual
acceptance, and records the maximum residual location and predicted first-step
water velocity. A paired refined-mesh A/B also changes only the curvature
normal gradient from `Gauss linear` to `leastSquares`; surface tension,
interface profile, mesh, contact angle, and all experimental inputs remain
fixed. These are declared operator diagnostics, not accepted results. Their
short-window and then 1 s outcomes must be recorded before this gate can be
released.

The paired operator runs are now complete. On the unchanged 456,068-cell
refined mesh, `rAU` weighting with the original `Gauss linear nHat` leaves
the physical-sigma result essentially unchanged (`0.12546 m/s` and
`1596.3 Pa/m`, versus the prior `0.12539 m/s` and `1681.4 Pa/m`).
Keeping that weighting and changing only `nHat` to `leastSquares` lowers the
`0.05 s` maximum water-weighted speed to `0.03173 m/s` and the reconstructed
initial residual to `958.5 Pa/m`. This isolates a substantial curvature-
gradient improvement but still fails the unchanged `0.02 m/s` gate. The
residual maximum is in a mixed interface cell
at `(3.4593,-0.00019,0.66644) m`, `alpha.water=0.0706`, near the tee and the
upper edge of the declared band. The next diagnostic therefore keeps the
improved operator and all physical inputs fixed, adds conformal planes at
`z=0.6525/0.6675 m`, and targets `2.5 mm` cells in the local interface band.
Its first implementation passes strict `checkMesh` but is rejected: the mesh
grows to 960,212 cells, maximum non-orthogonality rises to `67.79 deg`, the
initial residual rises to `2741.5 Pa/m`, and water-weighted speed reaches
`0.04948 m/s` by `0.028 s`. The run was deliberately ended with OpenFOAM
`stopAt=writeNow` once the rejection was unambiguous rather than spending the
remaining short window.

That result also exposed a mesh-control defect: lowering the global Gmsh
minimum from 4 to 2.5 mm allowed `MeshSizeFromCurvature=64` to refine much of
the full pipe, not only the declared interface box. A spatial size-floor
callback now restores the 4 mm floor outside `0.63<=z<=0.69 m`; the corrected
local profile must be rerun and must not reuse the 960,212-cell result. In
parallel, a declared `pointCellsLeastSquares` test checks whether a larger
point-neighbour stencil reduces curvature noise on the unchanged refined mesh.
If neither passes, the next mesh treatment is a 24-layer axial-prism slab with
the three transition heights on exact layer faces. No event run is released
by these diagnostic improvements alone.

The unchanged-refined-mesh `pointCellsLeastSquares` short diagnostic reaches
`0.01598 m/s` maximum water-weighted speed and `721.8 Pa/m` reconstructed
residual with physical surface tension. Its short-window thresholds pass, but
the speed rises again over the final samples and the run covers only `0.05 s`;
therefore `closed_hold.pass=false` remains correct. The prism mesh has now
passed strict mesh checks with 500,672 tetrahedra plus 2,136 prisms and all
25 required layer planes asserted, but it has no solver acceptance result yet.

The prism construction also exposed a small geometry inconsistency in the
older tetrahedral profiles: their atmosphere box began 1 mm below the physical
rim solely to overlap the riser Boolean, creating an annular external-air
sliver below `z=1.85 m`. The source now begins every external atmosphere
exactly at the paper-derived physical rim while retaining a 1 mm central-riser
overlap for robust Boolean union. Consequently all future base/refined/prism
comparisons use the same `0.11739557 m3` CAD fluid domain; prior diagnostic
fingerprints remain historical evidence and cannot release the current gate.

The physical-sigma prism short diagnostic now completes. Its `0.05 s`
maximum water-weighted speed is `2.36e-5 m/s`, maximum sampled riser speed is
`6.72e-4 m/s`, and water-volume and total-mass residual fractions are
`2.40e-9` and `3.00e-9`, respectively. The water-side speed falls to a
few micrometres per second after the initial projection instead of exhibiting
the growing tetrahedral CSF current. The reconstructed initial residual
(`693.4 Pa/m`) remains above its conservative diagnostic warning level, so
the initializer correctly reports `force_balance_accepted=false`; dynamic
static-hold acceptance remains governed by the independent 1 s drift,
velocity, and conservation gates. This result selects the prism profile with
`leastSquares nHat` as the formal `closed_base` candidate, not as an already
accepted event baseline.

The first 1 s candidate did not reproduce the short trajectory: at
`t=0.031 s`, while the water-side audit was still in the micrometre-per-second
range, the gas/energy coupling became unstable and
`compressibleInterFoam` aborted with `Negative initial temperature
T0: -2882.196026`. This is a failed formal hold, not an infrastructure
completion. Because the nominally identical four-process short run had reached
`0.05 s`, the next A/B set records process count and tests
`pointCellsLeastSquares` on the prism mesh in both four-process and serial
execution, plus a paired `sigma=0` run. These checks diagnose capillary
forcing versus parallel trajectory sensitivity; they do not replace the
physical-sigma 1 s requirement.

Those paired point-gradient runs now separate the trigger. The four-process
physical-sigma case aborts at `t=0.019 s` with
`T0=-3028.147 K`, while the otherwise identical serial case reaches
`0.05 s` without a thermodynamic failure; its maximum water-weighted speed is
`0.00784 m/s` and total-mass residual fraction is `5.45e-8`. The four-process
`sigma=0` case also reaches `0.05 s`, with `2.36e-5 m/s` maximum
water-weighted speed and `3.09e-9` total-mass residual fraction. These are
diagnostics, not grounds to omit the paper's measured surface tension.
Inspection also found that Scotch gave different cell allocations to
nominally identical four-process runs, so trajectory comparisons were
partition-confounded. All current parallel runs therefore use deterministic
axial `simple` decomposition with `n=(nProcs 1 1)` and record per-step extrema
and locations for `T`, pressure, velocity, `k`, `epsilon`, and `alpha.water`.
Two physical-sigma fixed-partition repeats now produce byte-identical time
series and identical metrics: `2.11e-5 m/s` maximum water-weighted speed and
`3.28e-9` total-mass residual fraction. Their temperature extrema are
`284.20--307.14 K`, both at the top of the external atmosphere, identifying
the remaining gas/energy oscillation spatially. The deterministic 1 s hold
subsequently failed at `t=0.086397 s`: the water-weighted speed remained only
`2.11e-5 m/s`, and water-volume and total-mass residual fractions remained
`2.39e-9` and `3.28e-9`, but the external-gas velocity rose abruptly and the
temperature inversion reached its 100-iteration limit with non-finite-scale
negative internal-energy temperatures. This localised gas/energy failure is
not a static-hold pass. A tetrahedral-atmosphere `waveTransmissive`
diagnostic crosses the former failure time, but its top-boundary gas speed
still rises to `1.04 m/s` at `0.12 s`; boundary treatment alone therefore
does not repair the poorly resolved acoustic outlet. The initializer audits
the exact atmosphere-patch acoustic Courant number
`(Un+sqrt(gamma/psi))*maxDeltaT*deltaCoeffs`, because the solver's ordinary
Courant control contains velocity but not sound speed. Net and absolute
atmosphere mass flux are both retained to detect cancelling local
inflow/outflow. On the current tetrahedral external-air mesh that audit is
`490.5` at `maxDeltaT=5e-4 s`, caused by a top circular-seam cell with
`deltaCoeffs=2843.7 1/m` (0.352 mm face-to-cell distance). Even `5e-6 s`
would leave this local acoustic number at `4.91`, so blindly extending that
expensive timestep diagnostic is not justified. A declared
`prism_atmosphere` mesh instead preserves the exact CAD domain while imposing
vertical prism layers in the external air. The first 46-layer, 25 mm
construction failed strict `checkMesh` with 8 low-determinant cells and 32
low-interpolation-weight faces at the physical-rim transition. Their exported
coordinates locate the defect at `z=1.85 m`, so the declared candidate now uses
92 exact 12.5 mm layers to reduce both the end-cell aspect ratio and the
tetrahedron/prism size jump. The revised 471,331-cell mesh and its closed-valve
baffle both pass `checkMesh -allGeometry -allTopology`: minimum determinant is
`0.002026`, minimum interpolation weight is `0.06023`, maximum aspect ratio is
`17.66`, and maximum non-orthogonality is `56.12 deg`.

A matched `0.12 s` A/B pair on this same layered mesh now separates the outlet
condition. Fixed hydrostatic pressure completes but develops `0.1487 m/s`
gas-weighted speed and a `273.46--318.71 K` temperature range.
`waveTransmissive` limits those values to `0.00457 m/s` and
`295.52--296.80 K`; its maximum water-weighted speed is `3.84e-5 m/s`, with
water-volume and total-mass residual fractions of `7.89e-9` and `7.99e-9`.
The acoustic audit falls from `490.5` to `31.64` at the unchanged
`maxDeltaT=5e-4 s`. This matched evidence selects the layered
`waveTransmissive` configuration for the formal 1 s hold.

That unchanged configuration now completes the full `1.0 s` hold and passes
the declared gate. Maximum water-weighted speed is `9.72e-5 m/s` against the
`0.02 m/s` limit; free-surface and isolated-pocket volume drifts are both zero
at the retained resolution. The all-domain/gas maximum is `0.00884 m/s`,
temperature remains within `295.33--297.05 K`, and maximum water-volume,
global-gas-mass, and total-mass residual fractions are `2.81e-8`, `9.21e-7`,
and `3.11e-8`, respectively. The dynamic pass does not relabel the initializer's
independent `force_balance_accepted=false` audit. It releases the open-valve
smoke and event calculations without changing any experimental target.

The first released instantaneous-opening smoke is negative evidence, not an
event result. By `t=0.003 s` the sharp removal of the valve loss drives local
maxima of `452.7 m/s`, `785.7 MPa`, and `k=1.02e4 m2/s2` around
`x=5.98 m`; the diagnostic was stopped rather than spending further compute on
that non-physical trajectory. This does not authorize deleting the
instantaneous sensitivity. The next smoke uses the primary paper's measured
approximately `0.2 s` opening time with the already declared time-varying
porous baffle, while retaining the same mesh, fluids, interface, and outlet.
The first finite-opening attempt exposed a singular sub-face endpoint:
`Amin=1e-4` implied an unresolved aperture and `I=2.0e9 1/m`, causing the
momentum and energy solves to fail on their first update. The corrected
regularization is mesh-derived rather than tuned: `Amin=1/Nface`, exactly one
generated valve-face area, with the smoothstep loss sampled at 101 times.

That inertial-only table still failed as diagnostic negative evidence: with
`D=0`, `porousBafflePressure` provides no jump at `U=0`, so the ~6.5 kPa
hydrostatic difference across Valve #4 could accelerate before `I|U|²`
resisted. The smoke conserved mass through about `0.003 s`
(`Ugas~4.4 m/s`, `Uwater~0.15 m/s`) and then diverged (`T`/`p`/`k`/`ε`
blow-up). A follow-up Darcy table scaled to the closed-hold water-speed gate
failed earlier (`~4e-6 s`, SIGFPE): the jump remains U-dependent, so a large
`D` stiffens the baffle without restoring a finite head at `U=0`. Finite
opening therefore keeps `porousBafflePressure` with open-area floor
`max(1/Nface, 0.05)` and a mild tutorial-ratio Darcy term `D=2*I`, after
first applying the closed-valve `balanceInitialPressure` projection, merging
the wall baffles, and stripping the empty patches before recreating the
opening cyclic. One-face `Amin` on that balanced start still diverged near
`0.003 s` with healthy pressures beforehand and is retained as
`open_smoke_valve_0p2_balanced_porous_*`. A `uniformJump` hydrostatic-head
decay on the balanced start produced a violent early compressible transient
and is retained only as diagnostic negative evidence. Instantaneous opening
remains a zero-loss cyclic on the same balanced start. The measured-opening
smoke also tightens Courant limits to `maxCo=0.1` / `maxAlphaCo=0.05`.

That measured-opening smoke completed the declared `0.25 s` window:
`solver_completed=true`, maximum water-weighted speed `1.436 m/s`, gas-weighted
speed `3.806 m/s`, temperature `291.8--303.1 K`, and water/gas/total mass
residual fractions `1.17e-6` / `9.24e-7` / `1.91e-7`. Core and sensitivity event variants inherit the smoke Courant limits
(`maxCo=0.1`, `maxAlphaCo=0.05`, `maxDeltaT=1e-4 s`) through the measured
opening; a first `base_nominal` attempt with the looser defaults diverged near
`0.003 s`. After the valve is fully open, the running 13 s event may restart
from a written checkpoint with the looser defaults (`maxCo=0.25`) as a
documented post-opening throughput control. Instantaneous opening remains a
diagnostic, not the 13 s baseline.

## PDF re-verification (2026-07-14)

The primary offprint
`tests/test_02_cong2017/_shared/reference/paper_source/cong2017_JHE2017_offprint.pdf`
was re-extracted and checked against the live OpenFOAM sources
(`make_geometry.py`, `Allrun`, `0.orig/*`, `setFieldsDict`,
`MODELING_CONTRACT.json`). No wrong-run selection was found.

Confirmed against Table 2 and the apparatus prose for **B-H3**:

| Paper item | Paper value | Live 3-D source |
|---|---:|---|
| Run | B-H3, geyser | `BH3_Dr26_H066_L061` / contract `B-H3` |
| `D` | 0.05 m | `PIPE_DIAMETER = 0.050` |
| `Dr` | 0.026 m | `--riser-diameter` default `0.026` |
| `H0` | 0.66 m from invert | `INITIAL_FREE_SURFACE_Z = 0.660`; inlet head `107786.65 Pa` |
| `L0` | 0.61 m | pocket `5.98–6.59 m` |
| Valve / T / cap | Fig. 1 chain → Valve #4 | `VALVE_X=5.980`, `TEE_X=3.470`, `PIPE_LENGTH=6.590` |
| Physical riser | 1.8 m above soffit | `PHYSICAL_RISER_HEIGHT=1.800`, rim `z=1.850` |
| Laboratory `T`, `ρw`, `σ` | 23 °C, 998 kg/m³, 0.072 N/m | `296.15 K`, EOS at 998, `surfaceTensionValue 0.072` |
| IC partition | water upstream + riser to tank level; atmospheric pocket to cap | `x<5.98` water / pocket air / headspace air at atm |
| Opening time | ~0.2 s (no aperture history) | event `valve=0.2` porous smoothstep |

Declared translations unchanged: external air box to `z=3.0 m`, tank replaced by
constant-head `inlet`, 15 mm VOF band, contact angle 90°, and
`waveTransmissive` atmosphere. These are not relabeled as 2017 measurements.

## Live 13 s `base_nominal` status

After the passed closed hold and measured-opening smoke, `base_nominal`
opened under smoke Courant limits and then continued from the written
`t=0.25` checkpoint with looser post-opening Courant
`maxCo=0.25` / `maxAlphaCo=0.15` for throughput. That looser segment ran
to about `t=3.075 s` and then aborted (`FOAM FATAL`: negative `T0`, with
`alpha.water` and Courant exploding in one step). The last healthy field
write was `t=3.05 s`.

On `2026-07-14T18:37Z` the event was restarted from `t=3.05` with the
smoke Courant limits restored (`maxCo=0.1`, `maxAlphaCo=0.05`,
`maxDeltaT=1e-4 s`). This is a stability control change only; geometry,
IC/BC, valve model, and the geyser classification target are unchanged.
Progress snapshots remain in `outputs/base_nominal_progress.json`.
A 20-minute watchdog (`/tmp/bh3-watchdog-20m.sh`) monitors stall/FATAL/α-blowup
and auto-restarts from `latestTime` with smoke Courant limits when unhealthy.
