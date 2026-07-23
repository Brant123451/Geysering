# Test 01 — Vasconcelos & Wright (2011)

This Test reproduces the filling-shaft experiments used as Campaign 1 in the
Geysering paper.

## Cases

- `cases/A_Dt57p1_Ha0305_Yfs0356` — large tower, non-geyser branch, including
  the 2-D and 3-D OpenFOAM validation cases.
- `cases/B_Dt12p7_Ha0610_Yfs0356` — small tower, geyser branch.
- `cases/Fig10_Dt57p1_Ha0305_Yfs0254` — Figure 10 auxiliary water-level case.
- `cases/Fig11_Dt12p7_Ha0305_Yfs0254` — Figure 11 auxiliary comparison case.

Each Case contains its own configuration, experimental data, frozen model,
scripts, reference crops, and outputs. Original papers, parameter tables, and
cross-Case tools are under `_shared/`.

`_archive/` contains historical implementations retained for provenance; it is
not part of the current reproducibility path.
