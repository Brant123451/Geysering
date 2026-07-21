# OpenFOAM 2-D Case B (VW2011 centre panel)

Planar pilot for Vasconcelos & Wright (2011) Test 1 Case B (Figs.6/8).

## Paper inputs retained

| Item | Value |
|---|---:|
| Pipe ID \(D\) | 0.094 m |
| Chamber / middle / down | 0.546 / 2.970 / 0.490 m |
| Valve / tower centre | \(x=0.546\) / \(3.516\) m |
| Paper tower ID \(D_t\) | 0.0127 m |
| Planar tower width \(W\) | \(D_t^2/D \approx 1.716\) mm (area-equivalent) |
| \(L\), \(H_{a0}\), \(Y_{fs,0}\) | 0.610 / 0.610 / 0.356 m |
| Air pressure | 107298.3 Pa |
| Surface tension | 0 (planar thin-bore pilot; physical \(\sigma\) capillary-locks) |
| Valve model | instantaneous open (paper: manual <1 s) |

## Why not draw \(D_t\) as width?

Planar extrusion with width \(D_t\) has area ratio \(D_t/D=0.135\), while the
circular geometry has \((D_t/D)^2\approx0.018\). The wider vent under-predicts
rise (`Yfs*~0.95`, no spill). Area-equivalent \(W\) restores the venting ratio
so a geyser (`Yfs*≥1`) can appear. Use `../3d` for geometry-exact \(D_t\).

```bash
./Allclean
./Allrun
```
