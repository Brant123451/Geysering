# Compact validated outputs

The `openfoam_2d_*` files come from the 9 s OpenFOAM v2512 planar run recorded
by repository commit `d29865b`; its executable model revision is `a808bef`.
The `openfoam_3d_*` files come from the 9 s circular run based on executable
model revision `caf5610` and are recorded by result commit `b0deae6`. Both
solvers used four MPI ranks.

- `openfoam_{2d,3d}_metrics.json`: scalar comparison metrics.
- `openfoam_{2d,3d}_series.csv`: pressure-head time histories.
- `openfoam_{2d,3d}_levels.csv`: tower interface and free-surface histories.
- `openfoam_{2d,3d}_*_comparison.png`: review-ready plots.
- `openfoam_{2d,3d}_*_comparison.pdf`: vector versions of the plots.

Run `../scripts/run_3d.sh` to regenerate the circular result or
`../scripts/run.sh` for the planar result. OpenFOAM time directories,
decomposed fields, raw probe histories, and logs are intentionally not stored
here.
