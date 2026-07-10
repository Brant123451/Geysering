# Test 1 Case B 2-D research checkpoint

This directory is the safe-shutdown checkpoint for the planned two-dimensional
Test 1 Case B model. It records the evidence gathered before shutdown; it is
not a simulation-ready OpenFOAM case and contains no claimed numerical result.

## Preserved artifacts

- `case_definition.json` records the candidate Case B definition, provenance,
  and verification state in machine-readable form.
- `digitized/R227-05_experiment_matrix.csv` contains the experimental values
  transcribed from the cited source.

No OpenFOAM dictionaries or scripts were produced for Case B before shutdown,
so none are included or copied from Case A.

## Evidence available at shutdown

Wright, Lewis, and Vasconcelos (2007), *Mechanisms for Stormwater Surges in
Vertical Shafts*, DOI
[`10.14796/JWMM.R227-05`](https://doi.org/10.14796/JWMM.R227-05), describes the
head-of-air-pocket apparatus as:

- approximately `4 m` of horizontal acrylic pipe, `0.094 m` in diameter;
- butterfly valve `0.55 m` from the pressurized-air end;
- vertical shaft `0.49 m` from the far end and `0.61 m` high;
- pressure transducer `1.9 m` upstream of the shaft;
- shaft diameters `0.0127`, `0.0254`, `0.0444`, and `0.0571 m`;
- initial air pressure heads `0.303`, `0.610`, and `0.915 m`;
- initial shaft water levels `0.254`, `0.356`, and `0.457 m`.

The source labels Figure 5.7(b) as the `12.7 mm` tower and reports that every
test with that smallest shaft diameter spilled from the top. The intended
working hypothesis was therefore that Case B uses the same apparatus and
initial state as the existing Case A comparison, with the tower diameter
changed from `57.1 mm` to `12.7 mm`. That mapping still needs confirmation
against the primary Vasconcelos and Wright (2011) Case B definition. In
particular, the existing Case A uses `Ha0 = 0.305 m`, while the 2007 source
lists `0.303 m`; this checkpoint does not silently choose between them.

## Simulation status

- Primary 2011 Case B definition: not yet verified.
- Case B experimental curve digitization: not started.
- OpenFOAM geometry, fields, and solver dictionaries: not created.
- Mesh generation and solver run: not started.
- Recoverable OpenFOAM time/processor state: none.

The temporary full-text extraction and rendered source page remain outside the
repository as `/tmp/R227-05.txt` and `/tmp/R227-page-14.svg`. They are
third-party source extracts, not project results, and are intentionally not
committed.

## Resume

Restore this checkpoint with:

```bash
git switch cursor/test1-caseb-2d-4ac2
git pull origin cursor/test1-caseb-2d-4ac2
```

There is no valid solver-resume command yet because no Case B solver state
exists. After explicit authorization to continue, first verify the primary
Case B parameters, then create and review the Case B OpenFOAM case before
starting `Allrun`.
