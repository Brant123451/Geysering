# Campaign 2 unified physical-rim outlet audit

## Uniform decision rule

All cases use the same physical and numerical rule. The sampled plane is
`z = 1.8250001 m`, only `1e-7 m` above the true riser rim at `z = 1.825 m`.
A resolved crossing requires both:

1. an upward `alpha.water >= 0.5` component whose contiguous area covers at
   least 95% of one local rim face; and
2. cumulative positive `integral(alpha.water * max(Uz,0) dA dt)` equal to at
   least one adjacent normal-direction finite-volume cell.

The rule does not read experimental classifications, does not use the old
98%-of-rim liquid-level threshold, and does not infer values from images.
A resolved crossing is irreversible classification evidence once that run
ends normally. In contrast, a `NO_GEYSER` decision is final only after the
declared observation window has actually been completed; an early normal
exit with no crossing remains incomplete.

## Completed-run results

| run | unified result | max rim alpha | peak Q+ for 1 mm extrusion (m3/s) | cumulative V+ (m3) | first full gate |
|---|---|---:|---:|---:|---:|
| B-H1 refined | GEYSER | 1.0 | 1.0387762e-5 | 2.5870669e-6 | 14.0 s |
| B-H3 baseline end20 | NO_GEYSER | 7.8864150e-6 | 1.9226087e-12 | 9.0458090e-13 | none |
| B-H6 end20 | NO_GEYSER | 0.0 | 0.0 | 0.0 | none |

Thus the already completed cases reproduce the expected **yes/no/no** audit
pattern: H1 resolves true outlet crossing, while the diagnosed H3 baseline
and completed H6 do not. This document does not use that expected pattern to
make the decisions. H3 baseline remains a classification miss and cannot be
promoted as the final H3 result. H1 passes only this outlet-classification
gate: its resolved crossing makes the positive classification irreversible,
although the planned 16 s tail was not completed. Its previously documented
quantitative timing and velocity errors remain.

The detailed JSON reports retain every source time-directory name, surface
file SHA-256, model-extrusion flow, physical-circular equivalent, cumulative
ledger, geometry checks, and solver-log status.

## Running H3 refined case

At the separate read-only snapshot documented in
`h3_refined_checkpoint_status.json`, the refined H3 solver was still running.
No unified rim surface series has been generated from that active case, so
max alpha, Q+, cumulative volume, first crossing, and final classification are
all explicitly missing. No result is guessed from the current checkpoint.

## Reproduction

The sampler and auditor are:

- `tests/test_02_cong2017/_shared/sample_physical_rim_readonly.sh`
- `tests/test_02_cong2017/_shared/physical_rim_surface_controlDict`
- `tests/test_02_cong2017/_shared/audit_physical_rim_outlet.py`

The sampler builds a symlink-only scratch case and writes all new VTK files
under that scratch tree. It never starts a solver and never writes
post-processing data into the source case.
