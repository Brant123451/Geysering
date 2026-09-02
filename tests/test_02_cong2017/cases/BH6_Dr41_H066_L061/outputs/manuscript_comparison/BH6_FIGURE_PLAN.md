# B-H6 manuscript figure and table plan

## Location and dominant claim

- Manuscript location: `paper/sections/tests.tex`, Campaign 2, Series B (`sec:tests:cong2017:seriesB`).
- Focused claim: both calculations reproduce the observed non-geyser branch; the 2D run captures pocket arrival accurately, while both reduced descriptions underpredict the measured riser lift and gas-nose rise.
- Evidence status: **partial support**. Classification is matched, but the trajectory amplitudes are not quantitatively closed.

## Main-text assets

1. `cong2017_bh6_1d2d_full_domain_3frame.pdf`: complete pipe--tee--riser views at 0.0, 8.7 and 10.9 s, with 1D and 2D side by side on the same physical clock. Its single job is to show the system-level morphology/timing contrast.
2. `cong2017_bh6_1d2d_levels.pdf`: two panels redrawn from the digitized markers in Cong et al. (2017), Fig. 7(a), with unshifted 1D and OpenFOAM 2D curves. Its single job is quantitative trajectory validation.
3. Retain the existing Series-B table (`tab:cong_bh`) as the only main table; it already transcribes the essential B-H6 row from the paper's Table 2. Use `bh6_validation_metrics.csv` as the audit record instead of adding a redundant case table.

## Supporting/audit assets

- `source_crops/table2_seriesB.png`: source audit for the B-H6 row in Table 2.
- `source_crops/fig6_bh6_photos.png`: morphology audit at 8.7, 9.3, 9.9, 10.5 and 10.9 s. Do not reproduce in a submitted manuscript without checking publisher permissions.
- `source_crops/fig7_bh6_panels.png`: audit of the four original panels. Only panel (a) is used quantitatively.
- `cong2017_bh6_1d2d_pressure_supporting.pdf`: pressure context. The experiment is the same-condition companion run B-32 from Fig. 10(b), not the exact B-H6 high-speed record, so it must remain supporting evidence.

## Excluded from the main figure

- Fig. 7(b): velocities are differentiated from the same sparse positions and would duplicate/noisily amplify panel (a).
- Fig. 7(c): water/air column lengths are algebraic transforms of the interface positions and add no independent validation.
- Fig. 7(d): the published air-pocket pressure ratio has no directly equivalent, identically sampled 2D observable in the archived post-processing.
- Fig. 10(b): useful only as same-parameter pressure context, because its run number is B-32.

## Reproducible quantitative readout

- Arrival time: experiment 8.10 s, 1D 8.66 s, 2D 8.04 s.
- Free-surface marker RMSE: 1D 0.310 m, 2D 0.097 m.
- Gas-nose marker RMSE: 1D 0.317 m, 2D 0.245 m.
- No time shift is used. The dense 2D level traces use a centered 0.05-s running median only for plotting.

## Method-scope caveat

The geometry-matched H6 1D archive is a case-specific reduced application variant. Its horizontal gas-front closure is the Benjamin-celerity implementation, not an explicit full KH/IKH flux term. It should not be described as an independent verification of the manuscript's generic KH-containing horizontal formulation unless that solver-path discrepancy is resolved.
