# Cloud Agent handoff: geysering test 3 caseA2

Durable handoff target:

* PR: `https://github.com/brant123451/geysering/pull/9`
* branch: `cursor/test3-a2-openfoam-1850`
* base: `main`

Cloud-agent VMs, active solver processes, logs, meshes, and numerical time
directories do not transfer between Cursor accounts. Only committed and pushed
Git state is a reliable cross-account handoff.

## Evidence-based replacement model

The University of Alberta open-access MSc thesis for the same apparatus
(Liu 2018, DOI 10.7939/R30R9MK98) was recovered after the fixed-stage runs.
Unlike the condensed journal article, Sec. 3.1 reports a
0.57 × 0.61 × 0.89 m downstream tank and a 0.30 m-diameter, 0.40 m-high
movable circular overflow weir. Its A2 test table also gives the exact riser
inside diameter as 57 mm, which the journal rounds to 0.06 m. The source case
now includes that geometry, removes the unreported upstream headbox, applies
`Q(t)` at the reported pipe end, and extends the Q0 initialization to eight
seconds. Q0-only mesh pilots translated the movable crest from the preliminary
0.019 m estimate to 0.031 m so the numerical weir reproduces the reported
`Q0=20 L/s, hd=0.070 m` operating point; no transient pressure or no-geyser
result is used. The near-balanced pilot also exposed an unresolved source
inconsistency: PT3 relaxed to about 0.72 kPa, whereas the paper reports
0.99 kPa but describes that value as a 0.10 m chamber depth.

The old metrics below are therefore a superseded diagnostic baseline. New
base/refined strict meshes and full transients are required before completion.

## Completed and pushed

* Original Liu et al. paper, Case files, digitized traces, scans, frozen 1-D
  model, and old 3-D pilot independently audited.
* Full-dimensional geometry rebuilt with circular pipes, rectangular chamber,
  centered circular riser, and separately auditable atmospheric openings.
* Deterministic OpenCASCADE/Gmsh base and refined mesh profiles implemented.
* Base profile: 118,321 tetrahedra; full
  `checkMesh -allGeometry -allTopology` reports `Mesh OK`.
* The original fixed-stage downstream equivalent was implemented from
  `hd/Dd=1/4`, run, and shown quantitatively inadequate.
* `p_rgh` boundary handling corrected and pressure comparisons changed to
  reconstructed atmospheric-gauge `p`.
* The four-rank 0.2 s smoke run from the freshly initialized field completes:
  20 L/s inlet and outlet are initially consistent, PT3 is approximately
  0.99 kPa, probes and liquid-continuity outputs are written, and observed
  maximum Courant number remains below 0.5. This short smoke result is not a
  claim that the later four-second relaxation is converged.
* Clean-clone `Allrun`, `Allclean`, mesh/solve/resume scripts and compact
  experiment–1D–3D postprocessor implemented.
* Detailed evidence and modeling limitations are in `PAPER_AUDIT.md`; commands
  are in `README.md`.

## Completed after handoff

The new-account agent rebuilt both cases from committed source and completed:

1. base: 118,321 tetrahedra, strict `Mesh OK`, four-rank smoke and full
   `-4…14.4 s` transient;
2. refined: 187,195 tetrahedra, strict `Mesh OK`, four-rank smoke and full
   `-4…14.4 s` transient;
3. base/refined pressure, riser, Courant, liquid-continuity and grid-sensitivity
   postprocessing;
4. compact CSV/JSON/PNG output commits on this same PR branch.

The run result is an evidence-based **failed quantitative validation**, not a
completed match. Both grids predict no riser-top reach or water discharge, but
the four-second pre-ramp state is not converged and the pressure response is
far below the experiment:

| Metric | base | refined |
|---|---:|---:|
| Pre-ramp outlet (20 L/s inlet) | 22.05 L/s | 23.75 L/s |
| Pre-ramp water-volume slope | -1.93 L/s | -3.68 L/s |
| PT3 initial (paper 0.99 kPa) | 0.622 kPa | 0.591 kPa |
| Bore arrival, ramp clock (paper 1.60 s) | 2.805 s | 2.849 s |
| PT2 final-window mean (paper 2.15 kPa) | -0.034 kPa | -0.041 kPa |
| PT3 final-window mean (paper 4.99 kPa) | 1.643 kPa | 1.788 kPa |
| Maximum contiguous riser column | 0.020 m | 0.020 m |
| Maximum mixture front | 0.020 m | 0.080 m |

The unknown downstream tank/weir rating remains the leading physical
uncertainty. A longer initialization alone cannot reconstruct it. No
unreported geometry or fitted backpressure was introduced to force no-geyser.
The original acceptance requirement for a stable pre-ramp state therefore
remains failed even though both requested full calculations were completed.

## Reproduction

On the new agent:

```bash
git fetch origin cursor/test3-a2-openfoam-1850
git checkout cursor/test3-a2-openfoam-1850
cd tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/case
NP=4 ./Allrun base
NP=4 ./Allrun refined
```

The validated environment needs OpenFOAM.com v2512, Gmsh Python API 4.15.2
plus `libGLU.so.1`, NumPy, Matplotlib, and four MPI ranks. Do not commit
`processor*`, `postProcessing`, `constant/polyMesh`, numerical time
directories, logs, frames, or caches.

## Important interpretation constraints

* Required simulation clock: `t=0` at ramp start. Paper clock: `t=0` when the
  valve is fully open. Compare paper data at `t_sim=t_paper+0.4 s`.
* Compare transducer values to OpenFOAM `p` in kPa, not directly to `p_rgh`.
* The 0.13 m experimental riser value is a digitized first-column scalar, not
  a time series.
* `interFoam` is adequate for the vented filling-bore/no-geyser branch and mean
  pressures, but not acoustic water hammer, sealed-gas compression, or
  subgrid aerated-mixture physics.
* Modify no other Case, `paper/`, root README, `.gitignore`, bootstrap branch,
  or `main`.

## Durable final state

`NEW_AGENT_PROMPT.md` preserves the original acceptance requirements.
`PAPER_AUDIT.md`, this handoff, the 3-D `README.md`, and the Case root
`README.md` record which requirements passed and which runtime checks failed.
Generated meshes, decomposed fields, logs, and time directories remain
non-versioned; all compact evidence needed to audit the conclusions is under
the Case `outputs/` directory.
