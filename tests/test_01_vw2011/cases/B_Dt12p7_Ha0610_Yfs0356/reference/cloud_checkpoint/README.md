# Case B cloud research checkpoint

Source branch: `cursor/test1-caseb-2d-4ac2`
Source commit: `911d9c44f2110d02acf1c3a29432887b82783c00`

This checkpoint was created when the Case B cloud task was stopped. It contains
research evidence only; it did not produce a simulation-ready OpenFOAM case or
claim a numerical result.

The checkpoint used Wright, Lewis, and Vasconcelos (2007),
*Mechanisms for Stormwater Surges in Vertical Shafts*,
DOI `10.14796/JWMM.R227-05`, to preserve the 36-combination experiment matrix.
Its candidate mapping treated the 12.7 mm tower as the likely Case B diameter,
but left the `0.303 m` versus `0.305 m` initial pressure-head value unresolved.

The mature local Test 1 Case B already uses the primary 2011 definition
(`Dt=12.7 mm`, `Ha0=0.610 m`, `Yfs0=0.356 m`). Therefore this checkpoint is
retained as provenance and must not override the validated Case configuration.

Cloud status at shutdown:

- primary 2011 mapping was not verified;
- curve digitization was not started;
- no OpenFOAM dictionaries, mesh, solver state, or resume command existed.
