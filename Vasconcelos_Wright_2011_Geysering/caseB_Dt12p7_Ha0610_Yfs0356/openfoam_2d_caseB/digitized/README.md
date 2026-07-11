# Experimental reference data

The data in this directory are digitized from the centre panels of Figs. 6
and 8 in Vasconcelos and Wright (2011), DOI
[`10.1061/(ASCE)HY.1943-7900.0000332`](https://doi.org/10.1061/(ASCE)HY.1943-7900.0000332).
Those panels are labelled `Ha0=0.610 m`, `WLinit=0.356 m`; Fig. 6 and Fig. 8
are the `Dt*=0.135` (`Dt=12.7 mm`) groups.

Run from the Case B directory:

```bash
python3 digitize_reference.py
```

The script downloads the exact open-preview rasters, checks their SHA-256
hashes, and regenerates:

- `fig6_caseB_pressure_envelope.csv`: lower, median, and upper raster traces
  of the three pressure repetitions;
- `fig8_caseB_levels.csv`: dimensional-axis conversion of the retained marker
  centres in `fig8_caseB_level_pixels.csv`.

The Fig. 6 trace uses a five-pixel horizontal support, removes printed grid
lines, takes the 15/50/85% pixel quantiles, and applies a five-pixel median
filter. The Fig. 8 open interface markers were picked by their centres. The
filled free-surface symbols overlap at the available 714-by-515-pixel
resolution, so their shared centreline is retained instead of assigning
uncertain points to individual runs. A conservative +/-2-pixel uncertainty is
written to the level CSV (`T* +/-0.0189`, `Y* +/-0.0150`).

Axis calibration and source hashes are in `source_metadata.json`. No event-time
shift or fit to OpenFOAM output is used. The source images are fetched for
reproducibility but are not committed.
