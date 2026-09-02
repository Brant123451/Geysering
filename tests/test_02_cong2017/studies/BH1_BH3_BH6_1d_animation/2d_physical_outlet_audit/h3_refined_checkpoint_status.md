# B-H3 refined checkpoint: read-only outlet-audit status

Snapshot: `2026-08-10T02:51:40.7053630Z`.

- One `mpirun` parent and six solver ranks were running normally.
- Latest log time: `3.008815299 s`.
- Latest complete stored checkpoint directory: `3.0 s`.
- Execution time: `30157.47 s`.
- No normal `End` yet; no true `FOAM FATAL`, NaN, segmentation fault, or
  floating-point exception was detected. The normal `trapFpe enabled` startup
  line is not an error.

The active case was not post-processed. Therefore max rim alpha,
alpha-weighted positive outlet flow, cumulative passed liquid volume, first
resolved crossing, and final classification are all **missing**, not inferred.
The same unified scratch-sampling audit must be run after the complete event.
