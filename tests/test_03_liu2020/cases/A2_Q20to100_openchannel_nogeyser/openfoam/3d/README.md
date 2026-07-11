# Case A2 three-dimensional OpenFOAM validation

This directory contains the reproducible 3-D validation of Liu, Shao & Zhu
(2020), Case A2 (`Q=20→100 L/s`, downstream open channel). The geometry and
paper evidence are audited in [PAPER_AUDIT.md](PAPER_AUDIT.md). Generated
meshes, numerical time directories, logs, `processor*`, and `postProcessing`
are intentionally not versioned.

## Model

* Solver: OpenFOAM v2512 `interFoam`, VOF water–air, transient RANS
  `kOmegaSST`.
* Domain: full circular upstream/downstream pipes, full 0.30 m-wide chamber,
  and full circular riser; no symmetry, thin-layer, or 2-D approximation.
* Mesh: conformal first-order tetrahedra generated with Gmsh/OpenCASCADE.
  `base` and `refined` profiles change the chamber/riser and far-field sizes.
* Gravity: `(0 0 -9.81) m/s²`.
* Inlet: a positive liquid flow through the numerical headbox bottom,
  0.020 m³/s until `t=0`, linear to 0.100 m³/s by `t=0.4 s`.
* Initialization: approximate steady `Q0` velocities, 0.08 m upstream depth,
  chamber surface `z=0.12 m` inferred from PT3=0.99 kPa, and downstream
  `hd=Dd/4=0.070 m`. The simulation first stabilizes from `t=-4` to `0 s`,
  longer than one measured-depth Q0 pipe-flow transit.
* Downstream: fixed-stage equivalent at `hd=0.070 m`, split into hydrostatic
  water and atmospheric-air portions at the reported pipe end. This does not
  invent the unreported tank/weir dimensions or rating curve.
* Vents: numerical-headbox atmosphere and physical riser outlet are distinct
  patches. Water crossing the riser outlet is audited independently.
* Pressure comparison: sampled reconstructed gauge `p`, not `p_rgh`.

The required clock has `t=0` at ramp start. The paper defines zero at the
fully-open instant; experimental times are therefore shifted by +0.4 s in
the output comparison.

## Requirements

The validated environment uses:

* OpenFOAM.com v2512, launched by `/usr/bin/openfoam2512`;
* Python 3 with NumPy and Matplotlib;
* Gmsh Python API 4.15.2 and its `libGLU.so.1` runtime;
* four MPI ranks for the reported runs.

`OPENFOAM_LAUNCH`, `NP`, and `GMSH_THREADS` can be overridden. Mesh generation
defaults to one Gmsh thread so the tetrahedral mesh is deterministic.

## Reproduce

From `openfoam/3d/case`:

```bash
./Allrun base
```

This clean-clone entrypoint is equivalent to running `Allclean`,
`Allrun.mesh base`, `Allrun.solve smoke`, `Allrun.solve full`, and the
postprocessor in sequence.

`Allrun.mesh` requires both the independent combined STL to pass
`surfaceCheck` and the converted volume mesh to pass:

```bash
checkMesh -allGeometry -allTopology
```

The smoke run is a fresh 0.2 s `Q0` run. `Allrun.solve full` deliberately
starts fresh afterward, performs the complete `-4…14.4 s` run, and keeps only
three field checkpoints while retaining high-frequency compact function
outputs.

If a full decomposed solve is interrupted, do not run `Allrun.solve` again;
resume its latest processor checkpoint with:

```bash
NP=4 ./Allrun.resume
```

For the grid-sensitivity run:

```bash
./Allclean
./Allrun.mesh refined
./Allrun.solve full
python3 ../postprocess_compare.py --profile refined --no-primary
```

The postprocessor reruns the frozen Case A2 1-D model, reads all OpenFOAM
restart segments, applies the paper/simulation clock conversion, and writes
compact profile-specific series plus the required primary deliverables:

* `outputs/openfoam_3d_pressure_series.csv`
* `outputs/openfoam_3d_riser_series.csv`
* `outputs/openfoam_3d_metrics.json`
* `outputs/openfoam_3d_pressure_comparison.png`
* `outputs/openfoam_3d_riser_comparison.png`

## Numerical observables

`controlDict` records:

* PT1, PT2, and PT3 `p`, `p_rgh`, and phase fraction;
* 61 riser elevations with five radial samples at each elevation;
* chamber phase probes;
* water volume and phase-weighted water flux through every open boundary.

Riser results distinguish water-equivalent height, contiguous mixture-column
height, and highest mixture front. A geyser requires the mixture to reach the
1.22 m top and non-negligible water to leave the physical `riserOutlet`.
Liquid continuity is checked independently as

`V(t)-V(t0)+integral(sum(outward water fluxes) dt)`.

## Limitations

The strongest uncertainty is the downstream fixed-stage equivalent: Liu et
al. report only `hd/Dd=1/4`, not enough information to reconstruct the tank
and weir rating. The 20 mm ambiguity between the paper's stated chamber depth
and its PT3 pressure is also retained in the audit. Probe in-plane positions
are unreported. The mesh has no resolved viscous sublayer, so wall-function
friction is checked only through base/refined sensitivity. Finally,
incompressible single-velocity VOF does not resolve acoustic water hammer,
compressible trapped gas, or subgrid bubble slip/breakup; these restrictions
are material to pressure oscillations but not expected to control the vented
Series A no-geyser branch.
