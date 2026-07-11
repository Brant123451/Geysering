# Cloud Agent handoff: geysering test 3 caseA2

Durable handoff target:

* PR: `https://github.com/brant123451/geysering/pull/9`
* branch: `cursor/test3-a2-openfoam-1850`
* base: `main`

Cloud-agent VMs, active solver processes, logs, meshes, and numerical time
directories do not transfer between Cursor accounts. Only committed and pushed
Git state is a reliable cross-account handoff.

## Completed and pushed

* Original Liu et al. paper, Case files, digitized traces, scans, frozen 1-D
  model, and old 3-D pilot independently audited.
* Full-dimensional geometry rebuilt with circular pipes, rectangular chamber,
  centered circular riser, and separately auditable atmospheric openings.
* Deterministic OpenCASCADE/Gmsh base and refined mesh profiles implemented.
* Base profile: 118,321 tetrahedra; full
  `checkMesh -allGeometry -allTopology` reports `Mesh OK`.
* Phase-aware fixed-stage downstream equivalent implemented from the only
  reported datum, `hd/Dd=1/4`; unreported weir dimensions are not invented.
* `p_rgh` boundary handling corrected and pressure comparisons changed to
  reconstructed atmospheric-gauge `p`.
* Four-rank smoke run completes: 20 L/s inlet and outlet are consistent, PT3
  is approximately 0.99 kPa, probes and liquid-continuity outputs are written,
  and observed maximum Courant number remains below 0.5.
* Clean-clone `Allrun`, `Allclean`, mesh/solve/resume scripts and compact
  experiment–1D–3D postprocessor implemented.
* Detailed evidence and modeling limitations are in `PAPER_AUDIT.md`; commands
  are in `README.md`.

## Not yet complete at handoff

The old-account VM has a base full run in progress, but that runtime state is
not in Git and cannot be used by a new-account agent. The required compact
3-D outputs have not yet been accepted or committed.

Remaining acceptance work:

1. Run the complete base transient and postprocess it.
2. Run the refined mesh through full `checkMesh`, complete transient, and
   profile postprocessing.
3. Check base/refined pressure, bore timing, riser metrics, Courant number,
   pre-ramp convergence, and liquid mass residual.
4. Confirm the model predicts no water/mixture discharge from the 1.22 m riser
   top; report discrepancies without tuning to the desired branch.
5. Update the Case root `README.md` with the final 3-D status, then run
   `Allclean` and commit only source plus compact CSV/JSON/PNG outputs.

## Reproduction

On the new agent:

```bash
git fetch origin cursor/test3-a2-openfoam-1850
git checkout cursor/test3-a2-openfoam-1850
cd tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/case
./Allrun base
./Allrun refined
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

## Suggested new-agent prompt

The complete copy/paste task prompt, including every original acceptance
requirement and the current implementation state, is committed as
`NEW_AGENT_PROMPT.md`.

> Continue PR 9 on branch `cursor/test3-a2-openfoam-1850` for
> “geysering test 3 caseA2.” First read `openfoam/3d/HANDOFF.md`,
> `NEW_AGENT_PROMPT.md`, `PAPER_AUDIT.md`, and `README.md`, verify the PR head
> and existing diff, then execute every requirement in `NEW_AGENT_PROMPT.md`.
> Preserve the current physical/modeling decisions unless runtime evidence
> proves a defect. Commit and push to the same PR branch.
