# Cong, Chan & Lee (2017) Test 2 / Series B paper audit

Audit decision: **PASS FOR B-H1 — DECLARED NUMERICAL CLOSURES REQUIRED**

The Case owner subsequently assigned Run B-H1 with `Dr=0.016 m`,
`H0=0.66 m`, and `L0=0.61 m`. The paper-defined geometry is now fixed. Items
that the papers do not measure are closed below as explicit, non-fitted
numerical treatments; none may be described as an experimental input.

## Sources and coordinate convention

Primary source:

- Cong, J., Chan, S. N., and Lee, J. H. W. (2017), *Geyser Formation
  by Release of Entrapped Air from Horizontal Pipe into Vertical Shaft*,
  J. Hydraul. Eng. 143(9), 04017039, DOI
  `10.1061/(ASCE)HY.1943-7900.0001332`;
  `references/cong2017.pdf`, SHA-256
  `6a2fd77ae65f6361ec5479780a3226e5f5cf69fde643866f692279588c16aa3e`.

The companion CFD paper is used only where the 2017 experimental paper cannot
define a computational extension:

- Chan, S. N., Cong, J., and Lee, J. H. W. (2018), *3D Numerical Modeling
  of Geyser Formation by Release of Entrapped Air from Horizontal Pipe into
  Vertical Shaft*, J. Hydraul. Eng. 144(3), 04017071, DOI
  `10.1061/(ASCE)HY.1943-7900.0001416`.

For the audit, `x = 0` is the upstream end of the horizontal test pipe and
positive `x` points toward the capped end. `z = 0` is the horizontal-pipe
invert. Thus the pipe crown/soffit is at `z = D = 0.050 m`.

## Paper verification

| Item | Primary-paper evidence | Verified value / interpretation | Status |
|---|---|---|---|
| Horizontal-pipe physical total length | PDF p.2, “approximately 6 m”; Fig. 1 dimension chain: `3.47 + (3.12 - L0) + L0 = 6.59 m` | `6.59 m`, reported to apparatus precision as `6.6 m`. The 2018 CFD paper explicitly says `6.6 m`. | RESOLVED |
| Effective OpenFOAM horizontal length | Derived from primary Fig. 1; corroborated by the 2018 paper’s boundaries at `x=0` and `x=6.6 m` | Model the full test pipe, `0 <= x <= 6.60 m`; do not truncate it to `6.0 m`. The Fig. 1 dimensions close at `6.59 m`, a 0.01 m rounding difference. | RESOLVED |
| Tee position | PDF p.2, Fig. 1: dimension `3.47 m` from the upstream end to the riser axis | `x_tee = 3.47 m`. | RESOLVED |
| Selected ball-valve position for `L0=0.61 m` | PDF p.2, Fig. 1: tee-to-valve distance `3.12-L0`, then `L0` from the selected valve to cap | Apparatus dimension chain gives `x_valve = 3.47 + 2.51 = 5.98 m` and `x_cap = 6.59 m`. In the rounded `6.60 m` CFD coordinate, preserving `L0` gives `x_valve = 5.99 m`. This is the rightmost valve (Valve 4). | RESOLVED to paper precision |
| Air-pocket position | PDF pp.2–3, Fig. 1(b) and procedure: water downstream of the selected valve is emptied, the end is capped, and the pocket is atmospheric | Full circular section over the final `L0=0.61 m`: nominal CFD interval `5.99 <= x <= 6.60 m`; initial gauge pressure `0 Pa`. | RESOLVED |
| Closed-end position | PDF pp.2–3, Fig. 1 and procedure: removable cap fitted after emptying downstream section | Capped wall at `x = 6.60 m` (`6.59 m` in the two-decimal Fig. 1 chain). | RESOLVED |
| Physical riser height | PDF p.2, Experiments; p.7, Observation of Geysers: `1.8 m`, measured from the horizontal-pipe soffit | Physical rim is `1.80 m` above the crown, i.e. `z_rim = 1.85 m` under the stated coordinate convention. | RESOLVED |
| Upstream head for the high-speed Series-B set | PDF p.3, Table 2, rows B-H1 through B-H7; p.12 notation says Series-B `H0` is measured from tunnel invert | `H0 = 0.66 m` above pipe invert for all B-H1…B-H7. Values `0.77/0.88 m` belong to other Series-B runs, not this fixed high-speed set. | RESOLVED |
| Initial riser water level | PDF p.3: pressurized-pipe Series B has the initial riser water depth at the same level; p.12 defines `H0` | Free surface initially at `z = H0 = 0.66 m`. `L0=0.61 m` is the horizontal air-pocket length, not a riser water level. | RESOLVED |
| Experimental valve opening time | PDF p.3, Experimental Procedure: manual operation “takes approximately `0.2 s`” | Experimental baseline duration is approximately `0.2 s`. The companion CFD paper says `0.5 s` and used instantaneous opening; that discrepancy belongs in the requested sensitivity study, not in the baseline geometry. | RESOLVED for duration; opening law unresolved below |
| Pressure taps | PDF p.4, Measurements; Fig. 1: PT1 at pipe crown near the capped end, PT2 at pipe invert directly beneath the riser | PT2/Point B: `x=3.47 m`, invert. PT1/Point A: crown/soffit near the downstream end. Because the exact PT1 offset is not reported, 3-D output must label a reproducible end-pocket proxy and a pocket-volume average rather than claim the experimental coordinate. | RESOLVED as a bounded measurement operator |

## Conflict adjudication

- **6.6 m versus 6.0 m:** `6.0 m` is not an apparatus dimension in the
  primary paper. Fig. 1 closes at `6.59 m`, and the companion CFD paper uses
  `x=0…6.6 m`. The repository’s `6.0 m` 1-D length is therefore not admissible
  for a paper-faithful 3-D geometry.
- **Tee `x=3.47 m` versus `x=2.88 m`:** Fig. 1 directly dimensions
  `3.47 m`. The `2.88 m` value is a repository-side kinematic back-calculation,
  not a measured apparatus dimension, and is rejected for this 3-D audit.
- **Physical 1.8 m versus computational 3.0 m:** the experiment has a
  `1.8 m` riser. The 2018 CFD paper deliberately extended the *confined riser*
  to `3.0 m` to keep the air pocket and water column inside it. It does not
  describe that extension as an exterior air domain.
- **`H0=0.66 m` versus `0.88 m`:** every B-H high-speed row uses
  `0.66 m`; `0.88 m` belongs to other Series-B runs and to the fine-mesh
  companion-CFD Run B1.

## Direct initial-pocket check

The primary paper defines
`Vair = pi D^2 L0 / 4`. For `D=0.050 m`, `L0=0.61 m`:

- `Vair = 0.0011977322 m3 = 1.197732 L` (the supplied `~1.20 L`);
- at the measured laboratory temperature `23 degC` (PDF p.4), atmospheric
  pressure `101325 Pa`, and `R_air=287.05 J/(kg K)`,
  `m_air = 0.00142760 kg = 1.42760 g`.

The isolated pocket values are the mandatory pre-mesh targets. The exact
mesh-integrated water and gas inventories must be generated and checked after
field initialization because the conformal circular T-junction has a Boolean
overlap volume.

## B-H1 assignment and numerical closures

These treatments close the audit gate without pretending that unpublished
measurements exist:

1. **Owned Case:** `tests/test_02_cong2017/cases/BH1_Dr16_H066_L061`,
   Run B-H1, `Dr=0.016 m` (`Dr/D=0.32`). The experimental targets are
   `Ta=8.07 s`, `vfs=0.924 m/s`, `vint=1.231 m/s`, and `GEYSER`. The
   classification is validation data, never an initial or source term.
2. **Exterior atmosphere:** retain the physical riser rim at `1.80 m` above
   the pipe crown. Above it, use an explicitly numerical
   `0.30 m x 0.30 m` exterior-air box extending `1.20 m`, so its top is
   `3.00 m` above the crown. The box sides and top are atmospheric; its floor
   outside the circular riser mouth is a wall. This separates the experimental
   `1.8 m` riser from the companion paper's `3.0 m` *confined* riser and lets
   expelled water remain in the domain. The lateral clearance is `>9 Dr` from
   the riser axis; boundary flux and pressure must be checked before accepting
   an event result.
3. **Pressure operator:** PT2 is sampled just inside the fluid at the invert
   below `x=3.47 m`. PT1 is reported as `PT1_proxy`, sampled one pipe radius
   upstream of the cap and `1 mm` below the crown, plus an independent
   end-pocket gas-pressure volume average. The repository's unsupported
   `x=5.85 m` is not used for the 3-D probe.
4. **Valve:** the companion CFD baseline is an instantaneous fully open
   internal connection at `t=0`; the static-hold check converts the
   `x=5.99 m` face zone to a zero-thickness impermeable wall baffle. This is a
   numerical closed-disk representation used only for that check. Opening-time
   sensitivity uses the same monotone,
   non-fitted normalized effective-area law for `0.2 s` and `0.5 s`. It is
   labelled an equivalent-valve uncertainty study, not a measured ball-angle
   history. No pressure or velocity source may be used.
5. **Wall wetting:** no contact angle is reported. The baseline therefore
   imposes no fitted contact-angle model (`alpha.water` uses `zeroGradient` on
   smooth no-slip acrylic walls). This is recorded as “contact angle not
   prescribed”, not as a measured `90 deg` angle.
6. **Thermal state:** initialize water and air at the measured laboratory
   temperature `296.15 K` (`23 degC`). Air uses an ideal-gas equation of state
   at `101325 Pa`; water uses the measured `998 kg/m3` density and physical
   bulk compressibility.

With these declarations, all model-affecting paper conflicts are either
resolved by primary evidence or isolated as named numerical sensitivities.
Production setup may proceed, but validation claims remain conditional on the
mesh, domain-boundary, conservation, and valve-sensitivity checks.

