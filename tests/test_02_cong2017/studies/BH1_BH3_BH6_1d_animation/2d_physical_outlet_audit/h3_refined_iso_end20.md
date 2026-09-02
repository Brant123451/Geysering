# B-H3 physical-rim outlet audit

## Decision

- Classification: **INDETERMINATE_EVIDENCE_GAP**
- Final evidence gate: no
- Scope of final gate: physical outlet classification only
- Resolved crossing gate: no
- Experimental outcome used in the decision: no
- 98%-of-rim liquid-level criterion used: no

## Physical outlet metrics

- Maximum `alpha.water` on the physical opening: `0.1492995918`
- Peak positive alpha-weighted flow: `5.76890428e-07 m3/s`
- Cumulative positive liquid volume: `7.495459147e-08 m3`
- Physical-circular equivalent peak flow: `2.265443413e-05 m3/s`
- Physical-circular equivalent cumulative volume: `2.943459924e-06 m3`
- First interface-supported upward crossing: not observed
- First one-cell-volume passage: `12.80000019 s`
- First full-gate time: not observed

## Uniform numerical-resolution gate

A resolved physical-rim crossing requires an upward alpha>=0.5 component covering at least 95% of one rim face and cumulative positive integral(alpha*Uz dA dt) of at least one adjacent normal-direction finite-volume cell.

- Interface value: `alpha.water >= 0.5`
- Minimum contiguous upward area: `1.2844e-06 m2`
- Minimum cumulative passage: `5.408e-09 m3`
- Flow definition: `integral_opening(alpha*max(Uz,0) dA)`
- Stored-field basis: cell-centred alpha.water and U sampled on the rim plane; this is an alpha-weighted advective-flow audit, not a recovered OpenFOAM face alphaPhi ledger

## Geometry and provenance

- True physical rim: `z = 1.825 m`
- Sample plane: `z = 1.8250001 m` (offset `1e-07 m`)
- Area-equivalent width: `0.01352 m`
- 2-D extrusion: `0.001 m`
- Model-to-physical circular area scale: `39.26990817`
- Mapping: `W_2D=Dr^2/D so W_2D/D=(Dr/D)^2`
- Source case: `\\wsl.localhost\Ubuntu\tmp\bh3-2d-qualification\h3_refined_iso_riser20`
- Source layout: `parallel_decomposed`
- Stored/source times sampled: `400/401`
- Sample interval: `0.05000000075` to `20 s`
- Declared observation end: `13 s`
- Declared observation end reached: `yes`
- Normal solver End: `yes`
- True fatal/NaN evidence: `no`

Each per-frame record, source time-directory name, sampled-surface SHA-256,
and cumulative flux ledger is stored in the companion JSON file.
