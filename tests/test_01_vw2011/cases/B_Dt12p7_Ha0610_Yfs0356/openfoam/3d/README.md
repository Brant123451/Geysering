# Test 1 Case B — three-dimensional compressible VOF model

Status: **incomplete until the committed runtime evidence says otherwise**.
The source deck is designed for OpenFOAM.com v2512.  A smoke run or a mesh-only
run is not a reproduction of the experiment.  `outputs/openfoam_3d_metrics.json`
sets its own `completion_status` from the achieved time window and resolved
above-rim water.

Read `PAPER_AUDIT.md` before interpreting a comparison.  In particular, the
vertical pressure-tap datum is not reported, Table 2 velocities are
diameter-level averages, and one legacy Fig.8 marker at \(T^*=3.8469\) is
probably misclassified.

## Physical model

`make_mesh.py` builds one connected, genuinely three-dimensional volume:

* a 4.006 m long, 94 mm diameter circular main pipe;
* a 12.7 mm diameter circular tower connected by a Boolean circular tee at
  x=3.516 m;
* a physical rim 0.610 m above the main-pipe crown;
* a 0.24 m × 0.24 m exterior atmosphere extending 1.20 m above and 0.40 m
  below the rim.

The external volume is essential: the rim is not a numerical outlet, and a
water column can leave the physical tower before encountering an atmospheric
boundary.  An assumed 2 mm tower wall separates the interior from exterior
below the rim; this unmeasured thickness affects only the exterior casing.
The exterior bottom and sides are atmospheric/drain boundaries, so there is
no artificial shelf to retain spilled water.  Water detected above y=0.657 m
is an external ejection, not merely a full tower.  This is not a wedge, thin
layer, or unit-thickness conversion.

The initial state follows the paper:

* x=0–0.546 m is a sealed air pocket at 0.610 m water gauge head
  (107298.33 Pa absolute at the baseline);
* the main pipe downstream of the valve is entirely water filled;
* the tower free surface is 0.356 m above the crown (absolute y=0.403 m);
* the tower headspace and exterior start as 101325 Pa air;
* velocity is zero and temperature is 293.15 K.

`setFields` first assigns the bulk regions, then OpenFOAM `setAlphaField`
computes cut-cell volume fractions for the planar valve and tower interfaces.
This avoids cell-centre stair stepping on the tetrahedral mesh.  Finally,
two independent `setExprFields` processes first write `p` and then reload that
final field to make `p_rgh` hydrostatically consistent with the geometric
phase field.  The process split is required because OpenFOAM writes each
modified field through an unregistered object; a later expression in the same
process would otherwise read the pressure cached before those writes.  In
interface cells, `p_rgh` uses the same
alpha-weighted perfect-gas/perfect-fluid density that
`compressibleInterFlow` uses to reconstruct absolute pressure; a binary
alpha=0.5 density switch would create a nonphysical startup impulse.

The inferred initial chamber volume is 0.003789 m³.  The initial water volume
from ideal geometry is 0.0240116 m³ in the downstream main pipe plus
0.0000451 m³ in the tower extension.  `validate_case.py` recalculates these
values from the authoritative config before every run.

## Solver choice and represented physics

The application is DLR-RY TwoPhaseFlow `compressibleInterFlow` at pinned commit
`de9826f9ffb24f4b635ac97fd388ebd560cfc174`, compiled against OpenFOAM.com
**v2512**.  It is a single-momentum, two-phase VOF solver with phase
compressibility, an energy equation, gravity, viscosity and surface tension.
The baseline numerical candidate uses geometric `isoAdvection`, `plicRDF`
interface reconstruction and RDF curvature.  This migration addresses the
sustained free-surface parasitic currents found with stock
`compressibleInterFoam`.  A clean 0.006 s RDF screen exited normally, and its
repeat with the current TwoPhaseFlow-reference correctors reduced the
written-time free-surface velocity from stock's 3.839 to 1.285 m/s at
0.006 s; its peak over the screen was 1.682 m/s.  Alpha remained within
\([-5.77\times10^{-9},1+3.52\times10^{-12}]\), and gas/total balance errors
remained below \(4\times10^{-6}\%\).  The hotspot is still at the initial free
surface, so the candidate does not count as an accepted baseline until the
full 1.0 s hold passes.  A compressible solver remains necessary because the
forcing is expansion of a finite pressurised gas inventory.

The first RDF screen reached its requested end time but aborted during process
teardown: loading `fieldFunctionObjects` also loaded stock `libgeometricVoF`
beside TwoPhaseFlow `libVoF` and corrupted the duplicate runtime-selection
table at destruction.  Runtime extrema are now calculated by the existing
phase-accounting object, so the conflicting library is not loaded.  The clean
rerun exited with code zero; the failed teardown attempt is not treated as
screen evidence.

The first attempted full hold also exposed stock-deck corrector counts that
were not justified for TwoPhaseFlow: two outer loops, three pressure
correctors and two non-orthogonal passes caused about 18 pressure solves per
step.  TwoPhaseFlow's supplied surface-tension cases normally use one outer
loop, two pressure correctors and zero non-orthogonal passes, with one alpha
correction and two alpha subcycles.  Those values are now the numerical
candidate and are materialised in the run manifest.  Their repeat 0.006 s
screen exited normally in 587 s, 47.5% faster than the legacy-corrector RDF
screen.  Peak velocity increased by 22.6%, but the final velocity remained
66.5% below the stock-solver diagnostic, while alpha and mass bounds remained
tight.  The candidate is therefore cleared for the full hold; reduced cost
alone is not treated as hold evidence.  That full-hold attempt was rejected
after 0.06 s: raw pressure drift had already reached \(H^*=0.0570\), above the
0.02 acceptance limit, and one cell immediately upstream of the sharp
penalty-zone edge grew from 1.414 m/s at 0.04 s to 1.825 m/s at 0.06 s.
Global Courant control reduced the timestep below \(3\times10^{-5}\) s;
interface and capillary limits were inactive.  Alpha, mass and rim-water
checks remained clean, so this is a localized closed-valve pressure-support
failure rather than an RDF transport failure.

The conformal baffle then passed generated-mesh and 0.001 s startup checks,
but its extended RDF screen was rejected at 0.04 s.  It removed the delayed
valve-edge hotspot: the written velocity maximum remained at the initial
tower free surface and was 1.362 m/s at 0.04 s.  Nevertheless, the raw
transducer range was 337.96 Pa, or \(H^*=0.05658\), already above the immutable
0.02 hold limit.  Alpha stayed within approximately
\([-1.75\times10^{-10},1+2.44\times10^{-10}]\), phase/total balance errors
remained below \(6.5\times10^{-6}\%\), and there was no rim water or gas entry.
The baffle is therefore retained as the physically correct closed-valve
topology, but `plicRDF + RDF` is not an accepted hold configuration.  The next
screen used the library-supported `fitParaboloid` curvature sensitivity,
which has lower reported curvature error than RDF on unstructured
triangular/tetrahedral meshes.  Its 0.006 s written peak and final velocities
fell to 1.187 and 0.796 m/s, 29.4% and 38.1% below the reference-corrector RDF
screen, with clean alpha and mass bounds.  It nevertheless experienced a
one-step startup Courant spike of 1.957 while `plicRDF` was interpolating
interface normals.  This short run is not cleared for extension.  The next
screen sets `interpolateNormal false`, matching TwoPhaseFlow's static
surface-tension reconstruction benchmarks, and records that control in the
manifest.  That repeat exited normally at 0.006 s and reduced the global and
interface Courant maxima to 0.350 and 0.289.  It did not satisfy the joint
screening rule, however: peak velocity increased 9.7% to 1.302 m/s, while the
final value improved only 0.9% to 0.788 m/s.  Alpha and mass bounds remained
clean, with no rim water or gas entry.  The run had only one pressure sample,
so its zero sampled pressure range is not hold evidence.  Neither
`fitParaboloid` screen is cleared for extension.

The library's static curvature benchmarks use `interpolateNormal=false`, but
they reconstruct geometry without solving the transient momentum equation;
its dynamic capillary-wave cases use the source default `true`.  Since
OpenFOAM Courant control acts on the next timestep rather than as a strict
current-step cap, the next diagnostic pairs normal interpolation with a hard
startup timestep cap before choosing a pressure-drift candidate.  That
diagnostic reproduced global Co=1.992 at approximately
\(2.0\times10^{-5}\) s even though its preceding timestep was only
\(4.30\times10^{-6}\) s, below the imposed `maxDeltaT=1e-5`.  Interface Co was
only 0.0075, and the first written velocity was already 1.239 m/s.  The event
is therefore a highly local pressure-corrected flux excursion, not adaptive
timestep growth, and the interpolated-normal path is rejected.  The next
minimal sensitivity retains `interpolateNormal=false` while matching the
static benchmark's stricter plicRDF convergence controls
(`iterations=10`, `tol=1e-8`).  That sensitivity was worse: the first written
velocity rose to 1.905 m/s and global/interface Co reached 0.487, so it was
stopped at 0.0018 s.  The next screen uses `isoAlpha + fitParaboloid` to remove
the plicRDF reconstruction iteration from the candidate rather than tightening
it further.  That pairing was also rejected: global Co reached 1.206, a later
global/interface spike reached 0.705, and velocity rose to 1.667 m/s by
0.001 s.  It also removes the plicRDF contact-angle ghost geometry on which
the current fit implementation depends.  The subsequent
`constantCurvature=0` mechanism diagnostic retained the real surface-tension
coefficient while removing the variable-curvature force.  It exited normally
at 0.006 s and reduced the reference-RDF peak/final velocities to
1.437/0.770 m/s, but global/interface Courant maxima still reached
0.393/0.214 and the hotspot remained at the initial tower free surface.
Variable curvature is therefore not the sole source of the startup imbalance.
This nonphysical diagnostic is rejected for extension; the next physical
screen is `RDF + plicRDF + interpolateNormal=false`.  That physical screen
also exited normally at 0.006 s, but global/interface Courant maxima reached
0.365/0.306 and the end velocity remained 1.220 m/s at the initial free
surface.  A direct pre-solver consistency check then exposed the more
fundamental defect: `p_rgh - (p + rho*9.81*y)` ranged from -453.5 to
+3946.5 Pa, with the maximum at that same free surface.  In a single
`setExprFields` process, `p` was preloaded once; each absolute-pressure
expression wrote an independent unregistered object to disk, so the final
reduced-pressure expression still read the original cached `p`.  All startup
screens prepared by that path are consequently initialization-defective.
Commit `3a1777d` now runs absolute and reduced pressure initialization as
separate processes.  A direct check over all 1,903,549 cells reduced the
residual to exactly zero, and the corrected zero-curvature rerun lowered the
first written velocity to 0.170 m/s.  It did not remove a delayed hotspot:
velocity still peaked at 1.424 m/s near 0.00454 s and global Co reached 0.363.
The initialization repair is retained.  The corrected physical RDF rerun then
reached global/interface Co=0.366/0.303 and a 1.775 m/s written peak at the
initial free surface.  Its written `K_` field reached thousands of inverse
metres although the analytical plane has zero curvature, including
+1608.6 1/m in a free-surface cell at the velocity-peak time.  Since global K
extrema can lie outside the active alpha-CSF band, the next run must first
collocate alpha, density, curvature and force residuals with the actual
velocity hotspot; no hold extension is cleared.  That logger now shows the
velocity hotspot in nearly pure gas and 462/473 kPa/m pressure-gravity/surface
forces at the same internal face at y=0.184 m, far below the intended
y=0.403 m interface.  A nearby alpha probe changed from 1.0 to 0.437 by
0.00649 s.  This nonphysical wall-adjacent interface points to the
exact-radius tower cylinder used by geometric alpha initialization.  The next
screen expands that selector into the known solid wall gap so all tower-fluid
cells below the plane start fully wet without selecting exterior fluid.

Thermophysical choices are:

* air: `perfectGas`, molecular weight 28.965 kg/kmol, \(C_p=1005\) J/kg/K,
  \(\mu=1.81\times10^{-5}\) Pa s;
* water: `perfectFluid`,
  \(\rho=998.153943+p/(7504.690432\,T)\) kg/m³; at 293.15 K this gives
  \(\rho=998.2\) kg/m³ at atmosphere and \(c=\sqrt{RT}\approx1483\) m/s;
* both phases start at 293.15 K; walls are adiabatic;
* surface tension is 0.072 N/m;
* gravity is `(0 -9.81 0)`;
* laminar momentum transport and smooth no-slip walls;
* neutral 90° contact angle because no measured acrylic contact angle exists.

The former stock-solver control `CASEB_ALPHA_SMOOTH_CURVATURE` is intentionally
rejected by the new deck: it acts only on
`compressibleInterFoam::interfaceProperties` and has no meaning for RDF.
Historical zero/two-pass diagnostics remain recorded: two passes reduced the
first 0.001 s velocity maximum but produced larger subsequent free-surface
hotspots and higher runtime.  The active curvature sensitivity is now
`RDF|fitParaboloid|gradAlpha`; stock smoothing results remain labelled as
rejected diagnostics rather than silently carrying an ineffective setting
into the new solver.

A conformal Gmsh disk at the initial tower free surface was also tested and
rejected.  Although its 1,907,679-cell mesh had one connected region and
standard `Mesh OK.`, the velocity maximum at 0.001 s increased from 3.207 to
8.192 m/s and remained centred on the free surface; alpha undershoot also
increased to \(9.7\times10^{-6}\).  Stock `compressibleInterFoam` computes
curvature from cell-centred alpha rather than the geometric disk normal, so
face alignment did not cure the imbalance.  The reproducible source therefore
retains geometric cut-cell initialisation.  Compact values are preserved in
`../../outputs/openfoam_3d_numerical_diagnostics.json`.

Using a least-squares gradient only for OpenFOAM's `nHat` curvature normal was
likewise rejected.  It reduced the 0.001 s maximum from 3.207 to 2.018 m/s,
but the maximum then rose to 4.515 m/s at 0.006 s, 17.7% above the
Gauss-linear diagnostic; alpha undershoot also worsened to
\(1.6\times10^{-7}\).  The baseline therefore retains `Gauss linear`.
First-sample improvement alone is not accepted as stabilization.

The perfect-gas energy equation includes compression/expansion work; it does
not impose the frozen 1-D model's isothermal pocket law.  Liquid
compressibility is retained so acoustic propagation is not assigned an
infinite wave speed.

OpenFOAM `p` is absolute thermodynamic pressure.  `p_rgh = p-rho*g·h` is the
hydrostatically reduced solution variable.  The atmospheric absolute
reference is 101325 Pa.  Reported \(H^*\) is calculated directly as

\[
H^*=(p_{\rm probe}-101325)/(\rho_w gL)
\]

at x=1.616 m, y=-0.043 m.  No extra crown correction is silently applied.
Because the paper does not report the tap elevation, pressure comparisons
retain a possible datum uncertainty as large as \(D/L=0.154\).

### Valve

The mesh contains a 12 mm long `valveZone` centred on x=0.546 m.  A coded,
purely dissipative momentum resistance represents the butterfly valve.  It
starts effectively closed, follows a cubic smooth opening fraction, and ends
at a generic fully-open loss coefficient \(K=2\).  It never adds pressure,
velocity or mass.  The baseline opening time is 0.25 s; the experiment only
reports “less than 1 s,” so this is an explicit assumption and is varied in
sensitivity runs.  `CASEB_VALVE_MODE=instant` supplies the paper-model
instantaneous-connection limit.

The numerical closed-state cap \(K\le10^8\) is an impermeability device, not a
geyser calibration.  A quadratic loss has a zero Jacobian at zero velocity and
cannot numerically support the finite static pressure jump across a closed
valve.  The same dissipative loss is therefore linearised with a 1.0 m/s
seal-speed floor at fully closed state; the floor decreases smoothly to zero
as the valve opens and never prescribes pressure, velocity, or mass.
`CASEB_VALVE_SEAL_SPEED` exposes this numerical penalty for diagnostics.  Its
first extended static test failed because a connected porous zone cannot
support the finite closed-valve pressure jump at exactly zero velocity and
creates a sharp pressure-correction coefficient jump at its cell-zone edge.
The source now uses a conformal two-sided no-slip baffle at the valve plane
for closed mode, while opening/instant modes retain the purely dissipative
resistance.  The baffle removed the penalty-zone hotspot and showed no
leakage, but the first 0.04 s RDF screen still failed the pressure-drift
criterion because the sustained velocity hotspot remained at the tower free
surface.  The topology is therefore validated, while the numerical
free-surface candidate remains unaccepted.

## Mesh

Gmsh OpenCASCADE fuses the circular pipe/tower/exterior geometry, fragments it
with an exact-radius disk at x=0.546 m, and HXT creates unstructured
tetrahedra.  The disk is a conformal `valvePlane` face zone: it remains
ordinary owner-neighbour faces for opening runs and `createBaffles` converts
it to two no-slip wall patches for closed runs.  Refinement boxes target the
tower, circular tee, valve and initial pocket nose, initial free surface,
near-rim jet and plume corridor.  A 40 mm linear transition surrounds each
box, and a 4 mm corridor extends around the full exterior tower casing; this
prevents fine casing triangles from connecting directly to far-atmosphere
cells.  HXT is followed by explicit Gmsh tetrahedron optimization.  Netgen
remains an optional
`make_mesh.py --optimizer netgen` experiment when the installed Gmsh build
provides it; the packaged Gmsh 4.12.1 used for validation does not.  The
selected algorithm, optimizer, transition thickness and Gmsh version are
written to the mesh metadata.  The long main pipe and far atmosphere remain
coarse.

| preset | nominal tower edge | nominal edges across \(D_t\) | pipe edge | purpose |
|---|---:|---:|---:|---|
| `base` | 1.05 mm | 12.1 | 10 mm | baseline |
| `refined` | 0.70 mm | 18.1 | 8 mm | sensitivity |

These are nominal background sizes; `checkMesh` and the generated metadata are
the authority for realised cell count and quality.  Every run executes both

```bash
checkMesh
checkMesh -allTopology -allGeometry
```

before field initialisation.  The standard check must report `Mesh OK.`.  The
strict audit is also retained verbatim in `log.checkMesh.strict`; it normally
reports one determinant check for tetrahedra on sharp boundary edges.  In
OpenFOAM v2512 that determinant is assembled only from internal face normals,
so a valid positive-volume boundary tetrahedron with two internal and two wall
faces is rank deficient by definition.  `Allrun` accepts this narrow,
explicitly recorded exception only when:

* the determinant check is the sole strict failure;
* the low-determinant count exceeds the independently reported
  `twoInternalFacesCells` count by no more than five cells;
* cell volume, face-pyramid, face-tet, skewness, interpolation-weight and
  volume-ratio checks all pass.

`openfoam_3d_mesh_sensitivity.csv` records `standard_mesh_ok`,
`strict_mesh_ok`, `strict_tet_boundary_exception`, both cell counts and the
strict audit status.  Any negative volume, additional strict failure, failed
standard topology/geometry check, or missing patch stops `Allrun`; the strict
exception is not represented as a literal strict `Mesh OK.` result.

## Reproducible commands

Prerequisites are Gmsh with Python bindings, NumPy, Matplotlib, OpenFOAM.com
v2512, and the pinned TwoPhaseFlow build.  Run from this directory:

```bash
./build_twophaseflow.sh
```

`Allrun` verifies both the source commit and the installed
`compressibleInterFlow` binary before any solver stage.

Cheap preflight and mesh check:

```bash
python3 validate_case.py
CASEB_STAGE=mesh CASEB_MESH=base ./Allrun
```

Static closed-valve hold:

```bash
./Allclean
CASEB_STAGE=hold CASEB_MESH=base ./Allrun
```

Opened-valve smoke run:

```bash
./Allclean
CASEB_STAGE=smoke CASEB_MESH=base CASEB_END_TIME=0.5 ./Allrun
```

Full baseline through \(T^*\ge6\) (`10.5 s / 1.7282 s = 6.08`):

```bash
./Allclean
CASEB_STAGE=full CASEB_MESH=base CASEB_END_TIME=10.5 \
CASEB_VALVE_OPEN_TIME=0.25 CASEB_MAX_CO=0.30 \
CASEB_MAX_ALPHA_CO=0.20 OPENFOAM_NP=4 ./Allrun
```

Resume an interrupted decomposed run without remeshing:

```bash
./Allrun.resume
```

`Allrun.resume` restores every materialised run control from
`outputs/runtime/run_manifest.json`, ignores ambient `CASEB_*` overrides, and
refuses to run when the manifest is absent or incomplete.  A resume therefore
continues the same numerical experiment; change controls by preparing a new
run, not by mutating an interrupted one.  Stock `compressibleInterFoam`
processor state is incompatible and cannot be resumed.  Do not run it after
`Allclean`.

Refined-grid full run:

```bash
./Allclean
CASEB_STAGE=full CASEB_MESH=refined CASEB_END_TIME=10.5 \
CASEB_VALVE_OPEN_TIME=0.25 CASEB_MAX_CO=0.30 \
CASEB_MAX_ALPHA_CO=0.20 OPENFOAM_NP=4 ./Allrun
```

Sensitivity controls, each requiring a separate clean run, are:

```text
CASEB_MESH=base|refined
CASEB_MAX_CO=0.15|0.30
CASEB_MAX_CAPILLARY_NUM=0.5|1.0
CASEB_VALVE_OPEN_TIME=0|0.10|0.25|0.50|1.0
CASEB_ADVECTION_SCHEME=isoAdvection|MULESScheme
CASEB_RECONSTRUCTION_SCHEME=plicRDF|isoAlpha|gradAlpha
CASEB_RECONSTRUCTION_ITERATIONS=5|10
CASEB_RECONSTRUCTION_TOL=1e-6|1e-8
CASEB_INTERPOLATE_NORMAL=false|true
CASEB_CURVATURE_MODEL=RDF|fitParaboloid|gradAlpha
CASEB_C_ALPHA=0.5|1.0|1.5  # only with MULESScheme
CASEB_HA0=0.579|0.610|0.641
CASEB_GAS_EOS=perfectGas|rhoConst
```

Advanced corrector controls are recorded for reproducibility and targeted
numerical diagnostics:

```text
CASEB_N_ALPHA_CORR=1
CASEB_N_ALPHA_SUBCYCLES=2
CASEB_N_OUTER_CORRECTORS=1
CASEB_N_CORRECTORS=2
CASEB_N_NON_ORTHOGONAL_CORRECTORS=0
```

Changing any of these makes a non-baseline configuration.

The reconstruction iteration/tolerance controls are materialised separately
from the pressure and alpha corrector counts.  Their source defaults remain
5 and \(10^{-6}\); 10 and \(10^{-8}\) reproduce the stricter plicRDF
convergence settings used by TwoPhaseFlow's static curvature benchmark and are
being screened as a numerical sensitivity, not silently promoted to baseline.

`interpolateNormal=false` is the baseline plicRDF setting because it matches
TwoPhaseFlow's static surface-tension benchmarks.  The former `true` setting
is retained as a recorded sensitivity; its first fitParaboloid screen had a
one-step startup Courant spike of 1.957.  The corresponding `false` screen
reduced that spike but increased peak velocity, so this source default remains
an unaccepted numerical candidate until a full hold passes.

The \(H_{a0}\) endpoints are the reported ±0.031 m manometer precision.
`rhoConst` is a deliberately incompressible-gas limiting case at atmospheric
density; it is a sensitivity bound, not a plausible replacement baseline.
Gas compressibility is never emulated by a pressure source.

## Diagnostics and postprocessing

Runtime data are intentionally untracked.  Probes sample:

* absolute pressure and velocity at the transducer;
* five vertical lines across the tower every 5 mm;
* five vertical lines through the exterior plume every 10 mm.

Field, probe and accounting schedules use `runTime`, not
`adjustableRunTime`.  Samples may therefore lag their nominal interval by one
accepted timestep; this prevents output alignment from enlarging a
Courant-limited timestep after the solver has selected it.

The lower and upper transitions of the principal five-point arithmetic wet
profile define \(Y_{int}\) and \(Y_{fs}\).  A coded accounting object logs
liquid/gas mass, exact solver mixture mass and `rhoPhi`, signed atmospheric
phase-flux estimates, liquid inventory above the rim, and maximum cell-centre
height with \(\alpha_w\ge0.05\).  Mass errors include the time-integrated
signed atmospheric flux:

\[
\epsilon_m=(m(t)+\int_0^t\dot m_{\rm out}\,dt-m(0))/m(0).
\]

The total balance uses the solver's `rho` and `rhoPhi`.  Separate liquid/gas
balances use registered `thermo:rho.water`, `thermo:rho.air` and the geometric
`alphaPhi.water` boundary flux.  Cell inventories use transported alpha
without clipping, so an alpha-bound defect cannot be hidden by accounting.

`postprocess.py` creates the requested compact files under the Case-B
`outputs/` directory.  Pressure and level plots contain all three sources:
digitized experiment, frozen 1-D model, and 3-D CFD.  The “overflow volume” is
reported conservatively as the maximum liquid inventory above the rim; it is
not presented as an experimentally measured cumulative discharge.

Only the base-mesh full run with all baseline physical and numerical controls
writes the canonical plots and `openfoam_3d_metrics.json`.  A compatible
refined run writes its own preset metrics and updates the base result's mesh
comparison without replacing that canonical result.  Closed-valve evidence is
also monotonic: a shorter or failed diagnostic cannot overwrite an already
passed hold, and non-baseline hold controls remain runtime-only.
Every full run also upserts one configuration-keyed row in
`openfoam_3d_sensitivity.csv`, including all varied controls, event metrics,
geyser height, rim-water inventory, and phase/total conservation errors.

No `processor*`, `postProcessing`, `constant/polyMesh`, time directories,
mesh, logs, dynamic-code cache or frame sequence belongs in Git.

## Known physical and numerical limits

* The manual valve history, pressure-tap elevation, contact angle, roughness,
  coupling details, and experiment temperature are not reported.
* The valve is a distributed loss, not a resolved moving butterfly disc.
* Laminar single-momentum VOF does not resolve subgrid entrainment, droplets,
  or a separate gas/liquid turbulence closure.
* Five-point tower profiles are an arithmetic indicator, not an area-weighted
  circular-section integral; they and the \(\alpha_w=0.05\) plume threshold
  carry at least one local-cell uncertainty.
* The 2 mm exterior tower-wall thickness is assumed because wall thickness is
  not reported.
* The all-tetrahedral mesh retains the documented OpenFOAM v2512
  sharp-boundary determinant exception above.  It is a topology diagnostic,
  not a claim that `checkMesh -allTopology -allGeometry` prints `Mesh OK.`.
* The exterior is finite; a jet reaching its top invalidates the reported
  maximum height and requires a taller domain.
* Table 2 offers no Case-B-only velocity scatter, and Fig.6/8 inputs are
  raster digitizations rather than raw instrument data.

