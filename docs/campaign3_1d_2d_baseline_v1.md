# Campaign 3: case-local 1D--2D baseline v1

This report records the first zero-retuning comparison for Liu, Shao and Zhu
(2020) cases A2, B3, and C9.  It does not change the unfinished manuscript case
section.  The 2-D calculations remain exploratory spatial references, and the
three current 1-D sources are not yet a single campaign model.

## Common comparison rules

- A2 uses the native ramp-start clock and the paper comparison window
  `0 <= t <= 14 s`.
- B3 uses `t_match = t_solver - 2.0 s`.
- C9 uses `t_match = t_solver - 0.25 s`; only audited summary metrics are
  available locally.
- No fitted time shift, pressure offset, smoothing, or per-case coefficient
  adjustment is used.
- Pressure is gauge pressure at PT1--PT4.  Riser height uses an
  `alpha.water = 0.5` base-connected observer, with threshold sensitivity
  retained separately.

## Zero-retuning results

### A2: open-channel, no geyser

Both dimensions stay on the no-geyser branch.  The current 1-D model reaches a
maximum riser mixture-column height of 0.420 m during the 0--14 s comparison
window.  The 2-D probes give two different and physically useful views:

- base-connected liquid core: maximum 0.110 m;
- highest wet sample, including separated wet structures: maximum 0.810 m.

This observer spread shows that the present 1-D continuous mixture column is
not directly equivalent to either a clear-water core or a detached 2-D wet
front.  Pressure errors over 0--14 s are:

| Probe | RMSE (kPa) | Bias, 1D minus 2D (kPa) | 1D maximum (kPa) | 2D maximum (kPa) |
|---|---:|---:|---:|---:|
| PT1 | 0.011 | -0.005 | 0.000 | 0.116 |
| PT2 | 1.016 | 0.259 | 3.199 | 4.887 |
| PT3 | 4.884 | 4.617 | 5.716 | 5.558 |

PT3 has a similar absolute maximum but a large trajectory bias, so matching
only peak pressure would hide the principal discrepancy.

### B3: full-pipe, single-shoot geyser

Both dimensions reach the physical rim.  With the same 99% rim threshold:

| Quantity | 1D | 2D | 1D minus 2D |
|---|---:|---:|---:|
| first rim arrival (s) | 1.877 | 1.976 | -0.099 |
| maximum connected column (m) | 1.220 | 1.220 | 0.000 |

The connected-column RMSE is 0.347 m and the signed bias is -0.119 m.  Thus the
arrival time is already close, but the complete rise history is too low.

| Probe | RMSE (kPa) | Bias, 1D minus 2D (kPa) | 1D maximum (kPa) | 2D maximum (kPa) |
|---|---:|---:|---:|---:|
| PT1 | 1.656 | -0.196 | 4.112 | 5.898 |
| PT2 | 9.415 | -1.516 | 30.825 | 23.328 |
| PT3 | 7.770 | 1.110 | 37.139 | 24.154 |

The dominant B3 error is pressure amplitude and waveform, not rim-arrival
timing.

### C9: initial riser column and sealed upstream pocket

The retained 2-D audit gives one continuous `alpha.water >= 0.5` overtopping
episode beginning at paper time 0.75 s.  The two current 1-D variants give:

| Variant | first rim arrival (s) | rim episodes | PT1 max (kPa) | PT2 max (kPa) | PT3 max (kPa) | PT4 max (kPa) |
|---|---:|---:|---:|---:|---:|---:|
| 2-D audited summary | 0.750 | 1 | 7.385 | 16.469 | 22.330 | 19.967 |
| 1-D all-water | 0.911 | 1 | 4.112 | 11.493 | 16.387 | 12.847 |
| 1-D sealed pocket, 20 s | 1.078 | 1 | 4.112 | 16.485 | 3323.463 | 90.630 |

The sealed-pocket variant nearly reproduces the audited PT2 maximum
(`+0.016 kPa`) but develops a nonphysical long-horizon pressure runaway.  The
all-water variant stays bounded but systematically underpredicts all four
audited maxima and omits the documented pocket.  Neither is a complete C9
comparator.

## Model changes indicated by the baseline

1. Use the C9 source as the B3/C9 code base; with `series_c=False`, it reproduces
   the current B3 arrays exactly.
2. Merge A2 chamber entrainment and dispersed-gas storage into a common
   mixture-column equation that naturally reduces to the clear-water column
   when void fraction is zero.
3. Replace case-named switches with declared initial and boundary conditions:
   open weir, submerged outfall, throttled gate, free-surface, full-water, and
   sealed-pocket states.
4. Use one pressure observer at the physical probe coordinates and one
   state-dependent lid constraint.  The present A2 reconstructed pressure and
   B3/C9 PDE-tap pressure are not yet the same observation operator.
5. Repair sealed-pocket pressure relaxation before any closure fitting.  The
   20 s C9 runaway is structural and cannot be corrected by tuning a peak.
6. After the shared implementation passes migration tests, rerun A2, B3, and
   C9 together after every retained change.

## Reproducible artifacts

- Case-local 1-D export and source hashes:
  `output/one_d_two_d_matching/campaign3_case_local_baseline_v1/manifest.json`
- Standardized 2-D pressure and level observations:
  `output/one_d_two_d_matching/campaign3_2d_observations_v1/`
- Curve and summary metrics:
  `output/one_d_two_d_matching/campaign3_comparison_v1/`
- Extraction and comparison tools:
  `tools/one_d_two_d_matching/`

