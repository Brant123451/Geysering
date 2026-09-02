# S1 common 1-D/2-D comparison layer

This directory is an analysis layer for the **one Mahyawansi continuous-air
physical condition**. `coarse`, `medium_refine`, and `refined` are three meshes
of that one condition, not three experimental cases. The tools do not modify
the OpenFOAM campaign or the 1-D solver and cannot write `RESULT_ACCEPTED`.

## Frozen evidence contract

`OBSERVABLE_DEFINITIONS.yaml` is read together with, not instead of:

- `../openfoam/2d/mesh_levels/RESULT_ACCEPTANCE.yaml` for the frozen external
  eruption test and published/figure-read targets; and
- `../one_d/config/COMMON_OBSERVABLES.yaml` for canonical field names, the
  Stage-2 time origin, the 0.10 s grid, and the ban on time shifting.

The definition loader verifies both contracts and the SHA-256 of the locally
reviewed Mahyawansi et al. PDF before reading a result. Missing or changed
evidence fails closed.

The source paper was checked directly, including rendered journal pages 30 and
31. Figure 8's `0.8-0.9 m/s` is the **velocity magnitude of unmixed water in
the middle of the horizontal slug** at the states shown in Figures 4d and 7d.
The text explicitly states that air was not seeded, so PIV measured only water.
It is not gas-nose speed, gas-front speed, or slug-edge propagation speed. The
2-D field analogue is therefore an `alpha.water`-weighted cell-centred `|U|`
proxy in unmixed water (`alpha.water>=0.95`) inside the published
`x=-0.85...-0.75 m` window while a detected full-bore slug intersects it.
Gas-nose and slug-nose finite-difference speeds are separate proxy columns.

The Figure 2 caption directly publishes P1 at the origin, P2/P3 at vertical
coordinates `0.30/0.45 m`, and P4/P5/P6 at horizontal coordinates
`-0.80/-0.10/+0.10 m`. The model maps the paper's vertical `y` coordinate to
`z`. `H_upstream`, `riser_left`, and `riser_right` are project aliases for
P4/P5/P6, not names used by the paper.

## 2-D extraction

Export the formal Stage-2 times with the same ASCII cell-field form accepted by
the eruption classifier:

```bash
foamToVTK -case /path/to/formal/case -ascii -no-point-data \
  -fields '(alpha.water U)' -time 'stage2Start:' -output /path/to/vtk
```

Then run:

```powershell
python extract_2d_common.py `
  --case-dir E:\path\to\formal\case `
  --vtk-dir E:\path\to\ascii_vtk `
  --output-dir .\outputs\coarse
```

By default, `STAGE1_ACCEPTED_TIME` is the absolute solver time corresponding to
Stage-2 `t=0`. `--stage2-start` exists for synthetic/offline evidence only. A
nearest VTK frame is admitted only within `0.51*0.10 s`; the actual time error
is recorded. Pressure is linearly interpolated only across a probe gap no
larger than `0.10 s`. No signal is shifted in time.

The extractor writes:

- `2d_common_timeseries.csv`: all canonical columns plus explicit VTK/proxy
  evidence columns;
- `horizontal_phase_motion.csv`: compact gas-pocket and slug motion series;
- `2d_common_timeseries.metadata.json`: source hashes, published probe mapping,
  missing common times, and strict/proxy/unavailable classifications.

It reads `p` from all restart segments under `postProcessing/probesJHR`, checks
the six probe coordinates, lets a later restart supersede a duplicate time,
and converts absolute static pressure to gauge pressure using the declared
`101325 Pa` reference.

## What is strict, proxy, or unavailable

Operationally strict from `alpha.water` and cell geometry:

- the supply-connected horizontal gas component's threshold-cell nose, tail,
  gas-volume centroid and water-fraction-weighted gas-equivalent volume;
- arrival of that component in the riser-junction band;
- supply-branch gas-front elevation;
- the 99%-water-volume top of the largest wet component intersecting the riser;
- the frozen shared-face-connected external eruption state, including connected
  volume, launch attachment, q99 bulk top, and 0.10 s persistence; and
- the external maximum height used against Figure 11, taken only from the q99
  top of a persistent qualifying launch-attached component.

Declared cell-centred proxies:

- slug edges from an embedded full-bore-column run bracketed by gas-bearing
  columns;
- the Figure 8 middle-slug unmixed-water `|U|` quantity;
- finite differences of threshold-cell gas/slug noses;
- gross upward/downward liquid flow at `z=0.30 m`, positive mouth liquid
  outflow below the rim, gas volume outflow, and supply-branch liquid outflow.

Those flow estimates use cell-centred velocity times projected area. They are
not OpenFOAM face fluxes. `alpha.water` and `U` alone cannot provide gas mass
flow, phase/global conservation residuals, momentum residuals, or node reaction
impulse. Those canonical columns stay empty in the 2-D CSV and their reasons are
listed in metadata; they are never reconstructed or guessed.

The 2-D CSV also carries the exact `RESULT_ACCEPTANCE.yaml` event-evidence
columns (`launch_component_count`, `component_water_volume_m3`,
`bulk_top_q99_z_m`, `active_raw`, `active_persistent`, and companions). The
comparator recomputes the volume/q99 predicate and persistence from those
columns and rejects a standalone or forged eruption Boolean. Canonical columns
declared unavailable from `alpha.water` and `U` must remain empty.

## 1-D/2-D comparison

The future 1-D exporter must use the canonical names exactly. In particular,
`horizontal_slug_velocity_m_s` must carry the Figure 8 middle-slug **water
velocity magnitude**, not a moving-edge speed. The native riser profiles must
store `Aup/Qup/Adown/Qdown` independently.

```powershell
python compare_1d_2d.py `
  --one-d-csv E:\path\to\one_d_canonical.csv `
  --one-d-profile-npz E:\path\to\riser_twofluid_profiles.npz `
  --mesh coarse=E:\path\to\coarse\2d_common_timeseries.csv `
  --mesh medium_refine=E:\path\to\medium\2d_common_timeseries.csv `
  --mesh refined=E:\path\to\refined\2d_common_timeseries.csv `
  --output .\outputs\alignment_metrics.json
```

Both commands require their output to remain below this independent
`comparison/` directory. This is a mechanical guard against modifying the
frozen 2-D campaign or the concurrently developed `one_d/` tree.

The comparator requires exactly all three meshes, resamples every input to the
same Stage-2 `t=0,0.1,...` grid without time translation, and reports:

- the first hard gate: published eruption / each 2-D branch / 1-D branch;
- signed signature errors, zero-shift waveform RMSE and bias, and peak-time
  phase errors;
- source-target differences with no invented pass tolerance;
- 2-D grid spread separately from 1-D error; and
- profile field/shape/time/admissibility checks for the future 1-D NPZ.

The three mesh labels must also point to three distinct evidence files; one CSV
cannot be reused under coarse, medium, and refined aliases. This is an identity
guard, not proof that arbitrary externally supplied files have the claimed
mesh resolution.

The NPZ check also interpolates its native `Qup` and `Qdown` independently to
`z=0.30 m` and checks those values against the corresponding 1-D canonical
scalars at identical times. The shared-velocity VOF export does not contain
persistent labelled two-stream states, so the 2-D side is compared through the
explicit positive/negative cell-flux proxies only; no synthetic 2-D
`Aup/Qup/Adown/Qdown` profile is created.

For an incomplete trajectory, no eruption before 25 s is `inconclusive`. At a
complete 25 s, a stable no-eruption branch is a hard physics-alignment failure.
The 1-D model has (or is acquiring) a persistent reduced-order exterior-plume
inventory, but it has no resolved external free-surface domain. Consequently,
the measured maximum external height remains unavailable from the native 1-D
state. A separately named `derived_plume_proxy` may be reported, but neither it
nor the internal riser top may silently replace external height. The
post-eruption `P2` period is a declared detrended autocorrelation proxy, not a
published extraction method.

## Tests

```powershell
python -m pytest -q
python -m py_compile observables.py extract_2d_common.py compare_1d_2d.py
python extract_2d_common.py --help
python compare_1d_2d.py --help
```

The tests construct an ASCII hexahedral pipe/supply/riser/exterior VTK grid,
synthetic `probesJHR` segments, canonical CSVs, and a two-stream profile NPZ.
They cover the source velocity semantic, extraction, gauge pressure mapping,
unshifted time gate, eruption match/mismatch, waveform/phase metrics, mesh
spread, and the prohibition on result markers.
