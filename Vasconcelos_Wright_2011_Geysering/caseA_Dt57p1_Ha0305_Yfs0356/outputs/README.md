# Compact validated outputs

These files come from the 9 s OpenFOAM v2512 run recorded by repository
commit `d29865b`. The executable model revision is `a808bef`; the solver used
four MPI ranks.

- `openfoam_2d_metrics.json`: scalar comparison metrics.
- `openfoam_2d_series.csv`: pressure-head time history.
- `openfoam_2d_levels.csv`: tower interface and free-surface histories.
- `openfoam_2d_*_comparison.png`: review-ready plots.
- `openfoam_2d_*_comparison.pdf`: vector versions of the plots.

Run `../scripts/run.sh` to regenerate this directory from the model and
experimental CSV files. OpenFOAM time directories, decomposed fields, raw
probe histories, and logs are intentionally not stored here.
