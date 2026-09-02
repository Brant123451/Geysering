# B-H1 physical-rim outlet audit

## Decision

- Classification: **GEYSER**
- Final evidence gate: yes
- Scope of final gate: physical outlet classification only
- Resolved crossing gate: yes
- Experimental outcome used in the decision: no
- 98%-of-rim liquid-level criterion used: no

## Physical outlet metrics

- Maximum `alpha.water` on the physical opening: `1`
- Peak positive alpha-weighted flow: `1.03877622e-05 m3/s`
- Cumulative positive liquid volume: `2.587066877e-06 m3`
- Physical-circular equivalent peak flow: `0.0004079264676 m3/s`
- Physical-circular equivalent cumulative volume: `0.0001015938787 m3`
- First interface-supported upward crossing: `14 s`
- First one-cell-volume passage: `14 s`
- First full-gate time: `14 s`

## Uniform numerical-resolution gate

A resolved physical-rim crossing requires an upward alpha>=0.5 component covering at least 95% of one rim face and cumulative positive integral(alpha*Uz dA dt) of at least one adjacent normal-direction finite-volume cell.

- Interface value: `alpha.water >= 0.5`
- Minimum contiguous upward area: `2.432e-07 m2`
- Minimum cumulative passage: `1.024e-09 m3`
- Flow definition: `integral_opening(alpha*max(Uz,0) dA)`
- Stored-field basis: cell-centred alpha.water and U sampled on the rim plane; this is an alpha-weighted advective-flow audit, not a recovered OpenFOAM face alphaPhi ledger

## Geometry and provenance

- True physical rim: `z = 1.825 m`
- Sample plane: `z = 1.8250001 m` (offset `1e-07 m`)
- Area-equivalent width: `0.00512 m`
- 2-D extrusion: `0.001 m`
- Model-to-physical circular area scale: `39.26990817`
- Mapping: `W_2D=Dr^2/D so W_2D/D=(Dr/D)^2`
- Source case: `/tmp/bh1-2d-study/h1_refined_co015`
- Source layout: `serial_reconstructed`
- Stored/source times sampled: `298/298`
- Sample interval: `0` to `14.85000038 s`
- Declared observation end: `16 s`
- Declared observation end reached: `no`
- Normal solver End: `yes`
- True fatal/NaN evidence: `no`

Each per-frame record, source time-directory name, sampled-surface SHA-256,
and cumulative flux ledger is stored in the companion JSON file.
