# B-H1 OpenFOAM simulation data inventory (git-tracked)

GitHub hard limit is 100 MB/file, so the ~37 GB refined parallel field dumps
remain on the agent disk only (`openfoam/3d/runs/…`, gitignored). Everything
below is committed under `outputs/openfoam3d/`.

## Compact acceptance products (all runs)

For each completed run: `*-metrics.json`, `*-series.csv`, `*-comparison.png`
(where applicable), plus mesh/checkMesh/paper-audit JSON.

Full-event runs:

- `base-open-full-tau0`
- `base-open-full-tau0p2`
- `base-open-full-tau0p5`
- `refined-open-full-tau0`

Also: smoke / closed-static checks and `sensitivity-summary.*`.

## Function-object archives

`postProcessing-archives/`

| Archive | Approx. size |
|---|---:|
| `base-closed-static-postProcessing.tar.xz` | 0.2 MB |
| `base-open-smoke-postProcessing.tar.xz` | 27 KB |
| `base-open-smoke-tau0p2-postProcessing.tar.xz` | 27 KB |
| `base-open-full-tau0-postProcessing.tar.xz` | 6.8 MB |
| `base-open-full-tau0p2-postProcessing.tar.xz` | 6.9 MB |
| `base-open-full-tau0p5-postProcessing.tar.xz` | 6.9 MB |
| `refined-open-full-tau0-postProcessing.tar.xz` | 7.5 MB |

(Also mirrored as top-level `refined-open-full-tau0-postProcessing.tar.xz`.)

Contents include PT1/PT2 probes, riser centreline, phase mass/volume,
boundary fluxes, extrema, continuity.

## Solver logs

`solver-logs/`

| Log | Approx. size |
|---|---:|
| `base-open-full-tau0.log.xz` | 5.3 MB |
| `base-open-full-tau0p2.log.xz` | 5.3 MB |
| `base-open-full-tau0p5.log.xz` | 5.2 MB |
| `refined-open-full-tau0.log.xz` | 19 MB |

## ParaView VTK packages (`alpha.water` only)

| Directory | Times [s] |
|---|---|
| `refined-open-full-tau0-VTK/` | 0, 8, 8.2, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13 |
| `base-open-full-tau0-VTK/` | 0, 8.1, 9.5, 10, 11, 13 |
| `base-open-full-tau0p2-VTK/` | 0, 8.1, 9.5, 10, 11, 13 |
| `base-open-full-tau0p5-VTK/` | 0, 8.1, 9.5, 10, 11, 13 |

Each directory has `MANIFEST.csv`. Unpack any `t*.tar.xz` and open the `.vtm`
/ `internal.vtu` in ParaView.

## Local-only (not in git)

```text
openfoam/3d/runs/refined-open-full-tau0/   ~37–42 GB  (processor0..3 binary fields)
openfoam/3d/runs/base-open-full-tau0*/     ~3.6 GB each
```

See `LOCAL_RENDER_HANDOFF.md` for `reconstructPar` / multi-field `foamToVTK`.
