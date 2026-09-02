# B-H3 baseline: physical outlet-crossing audit

## Decision

The completed baseline run `paper_bh3_tau0p2_areaeq` ended normally at
`t = 20 s`, but it did **not** resolve liquid crossing the physical riser rim.
The experimental classification is geysering, whereas this baseline 2D result
is non-geysering; it is therefore retained only as a diagnosed classification
miss and must not be used as the qualified B-H3 result.

This decision is based on the physical plane at absolute `z = 1.825 m`
(`1.800 m` above the pipe crown), not on the superseded 98%-of-rim plotting
criterion. Small nonzero VOF values above the rim are reported rather than
silently rounded to zero, but they are spatially dilute numerical traces and
not a resolved liquid column or droplet.

## Full-event evidence

- The run log contains `Time = 20` followed by `End`, with no true
  `FOAM FATAL`, NaN, floating-point exception, segmentation fault, or abnormal
  exit.
- The maximum reconstructed free-surface elevation over the full event is
  `1.50178404196 m` above the pipe crown at approximately `t = 13.27 s`.
  It remains `0.29821595804 m` below the physical rim.
- Maximum liquid volume fraction at the three centreline plume probes over
  the complete 20 s record:

  | absolute elevation | maximum alpha.water | time |
  |---:|---:|---:|
  | 1.824 m | 7.9014979e-6 | 17.660 s |
  | 1.900 m | 6.3170366e-6 | 18.705 s |
  | 2.100 m | 3.3773093e-10 | 14.170 s |

## Whole-width field audit

The cell and point fields were decoded directly from parallel `foamToVTK`
exports. The audit covers the whole external domain above the rim, rather than
only the centreline probes, and separately samples all 11 x-locations on both
extruded point planes across the area-equivalent riser width
`x = 3.46324...3.47676 m`.

At `t = 17.65 s`, adjacent to the largest near-rim probe trace:

- mesh: 240,816 points and 115,800 hexahedral cells;
- cells with centre at or above `z = 1.825 m`: 24,600;
- maximum cell `alpha.water` above the rim: `2.3335812784e-6`, at
  `(x,z) = (3.46797228, 1.83500004) m`;
- maximum positive cell `alpha.water * Uz` above the rim:
  `3.9791260065e-7 m/s`;
- counts above the rim: 39 cells with `alpha.water > 1e-8`, and zero cells
  with `alpha.water > 1e-5`, `1e-3`, or `0.5`;
- maximum rim-plane point `alpha.water`: `3.4624436012e-6`;
- trapezoidal whole-width integral of positive `alpha.water * Uz`:
  `2.0497174065e-9 m2/s` per unit depth, or `2.0497174065e-12 m3/s`
  for the model's 1 mm extrusion.

At `t = 13.25 s`, the stored frame nearest the free-surface maximum:

- maximum cell `alpha.water` above the rim: `9.0054957935e-10`;
- zero cells above the rim with `alpha.water > 1e-8`;
- maximum rim-plane point `alpha.water`: `6.9377154022e-10`;
- whole-width positive `alpha.water * Uz` integral:
  `3.1957859165e-13 m2/s` per unit depth, or `3.1957859165e-16 m3/s`
  for the 1 mm extrusion.

The threshold counts are supporting numerical-resolution diagnostics, not a
replacement geyser definition. The primary physical evidence is that no
coherent liquid phase reaches the rim, the reconstructed interface remains
almost 0.30 m below it, and only dilute VOF traces occur in the exterior.

## Provenance and reproduction

- Complete-event series:
  `results/openfoam_2d_riser_series.csv`
- Complete-event metrics:
  `results/openfoam_2d_metrics.json`
- Full solver record:
  `run_record/log.solve.complete20`
- Scratch plume probes:
  `/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq/postProcessing/plumeProbes/`
- 17.65 s VTU:
  `/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq/VTK_OUTLET_AUDIT/paper_bh3_tau0p2_areaeq_93000/internal.vtu`
  (`sha256 3ba1a906278bc00bf9565a519550f6351b3eb203ec41347888c4474cc6d2d762`)
- 13.25 s VTU:
  `/tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq/VTK_OUTLET_AUDIT/paper_bh3_tau0p2_areaeq_60242/internal.vtu`
  (`sha256 371bdc504d27457561208b341d3d1242f45710bfb53311d1da7869d16a41a1bb`)

The 13.25 s export was generated without modifying or restarting the solver:

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd /tmp/bh3-2d-study/paper_bh3_tau0p2_areaeq
mpirun -np 3 foamToVTK -parallel -time 13.25 -name VTK_OUTLET_AUDIT
```

No experimental outcome, artificial source, imposed jet, plotting rewrite,
or case-specific threshold was used to change the simulated classification.
