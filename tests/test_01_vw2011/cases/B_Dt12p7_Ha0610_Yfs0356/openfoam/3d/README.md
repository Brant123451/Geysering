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

The inferred initial chamber volume is 0.003789 m³.  The initial water volume
from ideal geometry is 0.0240116 m³ in the downstream main pipe plus
0.0000451 m³ in the tower extension.  `validate_case.py` recalculates these
values from the authoritative config before every run.

## Solver choice and represented physics

The application is `compressibleInterFoam` from OpenFOAM.com **v2512**.  It is
a single-momentum, two-phase VOF solver with phase compressibility, an energy
equation, gravity, viscosity and surface tension.  It was selected instead of
incompressible `interFoam` because the forcing is the expansion of a finite
pressurised gas inventory.

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
instantaneous-connection limit, while `closed` is used for the hold test.

The numerical closed-state cap \(K\le10^8\) is an impermeability device, not a
geyser calibration.  Its adequacy must be judged from the static-hold leakage
and drift, which are not optional completion evidence.

## Mesh

Gmsh OpenCASCADE fuses the three circular/exterior volumes and HXT creates
unstructured tetrahedra.  Refinement boxes target the tower, circular tee,
valve and initial pocket nose, initial free surface, near-rim jet and plume
corridor.  A 40 mm linear transition surrounds each box, and a 4 mm corridor
extends around the full exterior tower casing; this prevents fine casing
triangles from connecting directly to far-atmosphere cells.  HXT is followed
by explicit Gmsh tetrahedron optimization.  Netgen remains an optional
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

Prerequisites are Gmsh with Python bindings, NumPy, Matplotlib and
OpenFOAM.com v2512.  Run from this directory.

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

`Allrun.resume` restores the valve settings from
`outputs/runtime/run_manifest.json`.  Do not run it after `Allclean`.

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
CASEB_VALVE_OPEN_TIME=0|0.10|0.25|0.50|1.0
CASEB_C_ALPHA=0.5|1.0|1.5
CASEB_HA0=0.579|0.610|0.641
CASEB_GAS_EOS=perfectGas|rhoConst
```

The \(H_{a0}\) endpoints are the reported ±0.031 m manometer precision.
`rhoConst` is a deliberately incompressible-gas limiting case at atmospheric
density; it is a sensitivity bound, not a plausible replacement baseline.
Gas compressibility is never emulated by a pressure source.

## Diagnostics and postprocessing

Runtime data are intentionally untracked.  Probes sample:

* absolute pressure and velocity at the transducer;
* five vertical lines across the tower every 5 mm;
* five vertical lines through the exterior plume every 10 mm.

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
balances split volumetric flux as \(\alpha\rho\phi\); they are diagnostic
approximations because `compressibleInterFoam` does not expose its subcycled
phase flux to the function object.

`postprocess.py` creates the requested compact files under the Case-B
`outputs/` directory.  Pressure and level plots contain all three sources:
digitized experiment, frozen 1-D model, and 3-D CFD.  The “overflow volume” is
reported conservatively as the maximum liquid inventory above the rim; it is
not presented as an experimentally measured cumulative discharge.

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

