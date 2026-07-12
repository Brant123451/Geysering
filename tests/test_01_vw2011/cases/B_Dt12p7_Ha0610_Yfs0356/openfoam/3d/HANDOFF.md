# Cloud-agent handoff — Geysering Test 1 Case B 3D

## Repository state

* Pull request: `https://github.com/brant123451/geysering/pull/10`
* Branch: `cursor/test1-caseb-3d-0614`
* Base branch: `main`
* Case root:
  `tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356`
* CFD root: `openfoam/3d`
* Complete copy/paste prompt: `openfoam/3d/NEW_AGENT_PROMPT.md`

All source changes and compact base-mesh evidence are committed and pushed.
Large runtime artifacts are intentionally ignored, so a fresh cloud agent must
rerun solver stages.  Do not use or merge the old 2-D branch
`cursor/test1-caseb-2d-4ac2`, and do not modify the Case-A reference directory.

The stock `compressibleInterFoam` startup evidence below exposed a sustained
free-surface capillary hotspot (`|U|max=3.839 m/s` at 0.006 s).  Conformal
meshing, curvature-alpha smoothing and a least-squares `nHat` gradient were
screened and rejected.  The source deck now contains a numerical candidate
using DLR-RY TwoPhaseFlow `compressibleInterFlow` at pinned commit
`de9826f9ffb24f4b635ac97fd388ebd560cfc174`, with
`isoAdvection + plicRDF + RDF`.  Its clean 0.006 s screen now passes, but this
candidate is not accepted as the baseline until the full 1.0 s closed-valve
hold passes.

The first full-hold attempt then exposed stock-deck corrector counts that made
each step perform about 18 pressure solves.  The source now materialises the
TwoPhaseFlow surface-tension-case pattern (one alpha correction, two alpha
subcycles, one outer loop, two pressure correctors and no non-orthogonal
corrector).  This lower-cost candidate passed its repeat 0.006 s screen in
587 s wall time.  Its peak velocity rose from 1.371 to 1.682 m/s relative to
the legacy-corrector RDF screen, but the 0.006 s value remained 66.5% below
stock `compressibleInterFoam`; alpha and mass bounds remained tight.  It is
therefore a valid startup screen, not an accepted hold baseline.  The ensuing
full-hold attempt was rejected at 0.06 s: pressure drift had already exceeded
the immutable acceptance limit, and a growing velocity hotspot had moved to
the first cell upstream of the sharp penalty-valve zone.

The conformal two-sided baffle replacement passed its generated-mesh and
0.001 s startup checks, and it removed that valve-edge hotspot.  Its extended
RDF screen was nevertheless rejected at 0.04 s: the velocity maximum remained
at the initial tower free surface, while the transducer range reached
\(H^*=0.05658\), already above the 0.02 hold limit.  Alpha, phase/total
balances, rim-water inventory and gas-entry checks stayed clean.  This
separates the physically correct closed-valve topology from the remaining
free-surface curvature/pressure-balance defect.  The next numerical screen is
`plicRDF + fitParaboloid`, not another full RDF hold.  Its first 0.006 s run
reduced the written peak and final velocities to 1.187 and 0.796 m/s, but a
one-step startup Courant spike reached 1.957.  It must be repeated with
`interpolateNormal false`, as used by TwoPhaseFlow's static surface-tension
benchmarks, before any extension.  That repeat exited normally and reduced the
global/interface Courant maxima to 0.350/0.289, but its peak velocity increased
9.7% to 1.302 m/s; only the final velocity improved slightly to 0.788 m/s.
It therefore fails the required joint Courant-and-velocity improvement and
must not be extended.  A tightly capped `interpolateNormal=true` diagnostic
then reproduced the same Co=1.99 excursion at only \(2.0\times10^{-5}\) s,
while the actual timestep was already below the cap.  Its first written
velocity also rose to 1.239 m/s.  This rejects timestep growth as the cause and
rejects the interpolated-normal path.  The next minimal screen keeps
`interpolateNormal=false` and tests the static-benchmark plicRDF convergence
controls (`iterations=10`, `tol=1e-8`).  That screen also failed early: its
first written velocity rose to 1.905 m/s and global/interface Co reached
0.487.  The next candidate is `isoAlpha + fitParaboloid`, which changes the
reconstruction algorithm instead of further tightening rejected plicRDF
iterations.  It was also rejected: global Co reached 1.206, a later
global/interface event reached 0.705, and velocity rose to 1.667 m/s by
0.001 s.  The subsequent `constantCurvature=0` mechanism diagnostic exited
normally at 0.006 s.  It reduced the reference-RDF peak/final velocities to
1.437/0.770 m/s, but global/interface Courant maxima still reached
0.393/0.214, above their declared limits.  Removing variable curvature
therefore did not remove the startup free-surface imbalance, and this
nonphysical mechanism test must not be extended.  The subsequent physical
`RDF + plicRDF + interpolateNormal=false` screen also exited at 0.006 s, with
global/interface Courant maxima 0.365/0.306 and end velocity 1.220 m/s.
It exposed a more fundamental initialization error: a direct field check found
up to 3946.5 Pa inconsistency between `p` and `p_rgh` at the initial tower free
surface.  The single `setExprFields` process preloaded the original `p`, while
its independent unregistered writes did not update that cached field before
evaluating `p_rgh`.  All prior startup screens share this defect.  Commit
`3a1777d` now splits the two fields into separate processes, and a direct
1,903,549-cell check reduced the same residual from
[-453.5,+3946.5] Pa to exactly [0,0] Pa.  The corrected zero-curvature rerun
strongly reduced the first written velocity to 0.170 m/s, but a delayed
free-surface hotspot still reached 1.424 m/s at 0.00454 s and global Co reached
0.363.  The repair is valid, while the nonphysical mechanism candidate still
cannot be extended.  The corrected physical RDF rerun has now also finished:
global/interface Co reached 0.366/0.303, the first written velocity was
1.516 m/s, and the written peak was 1.775 m/s at the initial free surface.
The written RDF curvature field reached thousands of inverse metres although
the analytical planar curvature is zero; at the peak time a free-surface cell
reached +1608.6 1/m.  Global curvature extrema may lie outside the active
alpha-CSF band, so the next diagnostic must collocate alpha, density and K in
the actual maximum-velocity cell and separately report pressure-gravity,
surface-tension and total face-force residuals.  Do not extend physical RDF
until that mechanism evidence selects the next single-parameter screen.  The
new logger was compiled and executed during a 0.00049 s continuation.  It
found the velocity hotspot in almost pure gas
(\(\alpha_w=1.03\times10^{-5}\), \(K=337.5\ {\rm m^{-1}}\)) and found
pressure-gravity/surface-force magnitudes 462/473 kPa/m at the same internal
face deep in the tower, y=0.184 m.  A nearby alpha probe changed from 1.0 at
time zero to 0.437 by 0.00649 s, far below the intended y=0.403 m free surface.
This is not physically reachable advection and identifies a wall-adjacent
alpha/reconstruction layer.  The next minimal fix is to extend the tower-water
initialization cylinder into the known 2 mm solid wall gap, where there are no
fluid cells, so every tower-fluid cell below the plane starts fully wet.  That
repair is now implemented and its zero-curvature rerun removed the deep force
hotspot.  The written velocity peak fell from 1.424 to 1.031 m/s and the
maximum pressure-gravity residual fell from 462 to 10.5 kPa/m, now confined to
the intended free surface.  Interface Co stayed below 0.121, but a one-step
global Co event reached 0.419 after deltaT grew to 0.179 ms.  Physical RDF must
now be repeated with this fully wet initializer before choosing curvature or
timestep controls.

## Verified before handoff

1. The paper and Case-B definition were audited in `PAPER_AUDIT.md`.
2. Migrated Case-B Python paths pass their path checks.
3. OpenFOAM.com v2512 and Gmsh 4.12.1 generate a genuinely 3-D circular
   pipe/tower/exterior-atmosphere mesh.
4. The committed base mesh evidence contains:
   * 1,903,549 tetrahedra with a conformal 1,829-face valve plane;
   * 12.1 nominal edges across the 12.7 mm tower;
   * standard `checkMesh`: `Mesh OK.`;
   * maximum non-orthogonality 74.24 degrees (one face over 70 degrees);
   * maximum skewness 1.058;
   * minimum interpolation weight 0.0909;
   * minimum face-volume ratio 0.100.
5. The strict `checkMesh -allTopology -allGeometry` log has one documented
   OpenFOAM boundary-tetrahedron determinant diagnostic: 646 low-determinant
   cells versus 644 cells with only two internal faces.  All other strict
   checks pass.  The source records this as
   `accepted_boundary_tet_exception`, not as a strict `Mesh OK.` result.
6. `setFields` and hydrostatic `setExprFields` initialization complete.
7. A four-rank closed-valve startup reached and postprocessed 0.001 s without
   a solver crash.  Dynamic compilation of the phase-accounting function and
   valve source succeeded.  Commits `71d014d` and `2740253` make short-stage
   intervals and single-sample derivatives robust.  The committed smoke
   evidence is correctly marked incomplete: no overflow or gas entry, zero
   one-sample mass-balance error, and 0.00246 dimensionless free-surface drift.
8. The pinned RDF candidate completed a clean four-rank 0.006 s screen:
   * solver exit code 0 and wall time 1118 s;
   * maximum written-time velocity 1.371 m/s at the initial tower free surface,
     versus 3.839 m/s at 0.006 s for the stock Gauss-linear diagnostic;
   * alpha range over the run
     \([-2.66\times10^{-9}, 1+3.40\times10^{-9}]\);
   * maximum gas and total balance errors \(3.74\times10^{-6}\%\) and
     \(1.53\times10^{-8}\%\), with no water above the rim.
   The first attempt exposed a teardown-only conflict when
   `fieldFunctionObjects` loaded stock `libgeometricVoF` beside TwoPhaseFlow
   `libVoF`; commit `9a59974` moved extrema into the existing accounting object,
   and the clean rerun exited normally.
9. The repeat 0.006 s screen with the materialised TwoPhaseFlow-reference
   correctors also exited normally:
   * wall time 587 s, 47.5% below the legacy-corrector RDF screen;
   * peak velocity 1.682 m/s and final velocity 1.285 m/s, versus
     3.839 m/s for stock `compressibleInterFoam` at 0.006 s;
   * alpha range
     \([-5.77\times10^{-9}, 1+3.52\times10^{-12}]\);
   * maximum gas and total balance errors \(3.16\times10^{-6}\%\) and
     \(1.80\times10^{-8}\%\), with no water above the rim.
10. The connected-domain penalty-valve hold was intentionally stopped and
    postprocessed after it became impossible to pass:
    * duration 0.06 s and requested duration 1.0 s;
    * pressure peak-to-peak \(H^*=0.0570\), already above the 0.02 limit;
    * the same cell immediately upstream of `valveZone` grew from
      1.414 m/s at 0.04 s to 1.825 m/s at 0.06 s;
    * global Courant control reduced the timestep to \(2.79\times10^{-5}\) s,
      while interface and capillary controls were inactive;
    * phase/total balances remained below \(1.7\times10^{-5}\%\), alpha
      remained tightly bounded, and no rim water or gas entry occurred.
    This separates a local closed-valve pressure-support defect from the RDF
    free-surface screening result.
11. The conformal-baffle RDF screen was intentionally stopped and
    postprocessed once it also became impossible to pass:
    * requested duration 0.08 s and postprocessed duration 0.039996564 s;
    * pressure peak-to-peak \(H^*=0.05658\), above the 0.02 limit;
    * written velocity maxima remained at the initial tower free surface,
      decreasing from 1.757 m/s at 0.01 s to 1.362 m/s at 0.04 s;
    * no valve-edge hotspot or valve leakage was observed;
    * gas and total balance errors remained below
      \(6.5\times10^{-6}\%\), alpha remained tightly bounded, and no rim
      water or gas entry occurred.
    This validates the baffle topology but rejects `plicRDF + RDF` as the
    hold configuration.
12. The first conformal-baffle `fitParaboloid` screen exited normally at
    0.006 s:
    * written peak and final velocities were 1.187 and 0.796 m/s, 29.4% and
      38.1% below the reference-corrector RDF screen;
    * alpha stayed within
      \([-2.12\times10^{-22},1+4.20\times10^{-11}]\);
    * gas and total balance errors remained below
      \(3.9\times10^{-6}\%\), with no rim water or gas entry;
    * a one-step startup Courant spike reached 1.957 and interface Courant
      reached 0.676.
    The velocity reduction is promising, but the CFL regression prevents
    extension until the static-benchmark normal-interpolation setting is
    tested.
13. The repeat `fitParaboloid` screen with `interpolateNormal=false` also
    exited normally at 0.006 s:
    * global and interface Courant maxima fell from 1.957/0.676 to
      0.350/0.289;
    * the written velocity peak increased from 1.187 to 1.302 m/s, while the
      final value changed only from 0.796 to 0.788 m/s;
    * alpha stayed within
      \([-1.23\times10^{-11},1+5.97\times10^{-12}]\);
    * gas and total balance errors remained below
      \(3.4\times10^{-6}\%\), with no rim water or gas entry.
    This does not meet the required joint velocity-and-Courant improvement.
    The single pressure sample also gives no usable \(H^*\) stability evidence,
    so the candidate is not cleared for extension.
14. Reducing the interpolated-normal diagnostic to `maxCo=0.2` and
    `maxDeltaT=1e-5` did not suppress its startup event:
    * global Co reached 1.992 at approximately \(2.0\times10^{-5}\) s while
      the preceding timestep was only \(4.30\times10^{-6}\) s;
    * interface Co at the event was only 0.0075, so the hotspot was a highly
      local pressure-corrected flux outside the near-interface mask;
    * the first two written velocities were 1.239 and 1.207 m/s, already above
      the earlier 1.187 m/s peak.
    The run was stopped at 0.0014 s after both immutable screening limits had
    failed.  Further reduction of `maxDeltaT` is not a justified remedy.
15. The stricter static-benchmark plicRDF convergence controls were also
    rejected before the prior 0.003 s hotspot window:
    * `iterations=10`, `tol=1e-8`, `interpolateNormal=false`;
    * the first written velocity at 0.00047 s was 1.905 m/s, 46.3% above the
      default-false screen's overall peak;
    * global and interface Co both reached 0.487;
    * alpha stayed within roundoff-scale bounds, with no rim water or gas
      entry.
    The run was stopped at 0.0018 s because later decay cannot remove an
    already failed peak-to-peak numerical screen.
16. `isoAlpha + fitParaboloid` was rejected at approximately 0.001 s:
    * an early pure-cell global Co event reached 1.206;
    * a second global/interface Co event reached 0.705;
    * written velocity increased from 1.233 m/s at 0.00047 s to 1.667 m/s at
      0.00101 s;
    * alpha remained bounded, with no rim water or gas entry.
    `isoAlpha` also lacks the plicRDF wall ghost geometry on which
    `fitParaboloid` relies for contact-angle information, so this pairing is
    not retained.
17. The recorded `constantCurvature=0` mechanism diagnostic exited normally
    at 0.006 s:
    * global and interface Courant maxima were 0.393 and 0.214, above the
      declared 0.30 and 0.20 limits;
    * written peak and final velocities were 1.437 and 0.770 m/s, with the
      peak still at the initial tower free surface;
    * alpha stayed within
      \([-2.51\times10^{-11},1+5.18\times10^{-11}]\);
    * gas and total balance errors remained below
      \(3.9\times10^{-6}\%\), with no rim water or gas entry.
    The test shows that variable curvature is not the sole source of the
    startup imbalance.  It is not a physical hold candidate and is not
    cleared for extension.
18. The physical `RDF + plicRDF + interpolateNormal=false` screen exited
    normally at 0.006 s but was rejected:
    * global Co reached 0.365 at completed time 0.000854 s;
    * interface Co reached 0.306 at completed time 0.003903 s;
    * the end written velocity was 1.220 m/s at the initial tower free surface;
    * alpha remained within
      \([-1.58\times10^{-10},1+3.44\times10^{-13}]\), with no rim water or
      gas entry.
    The next-step reductions exactly matched the configured Courant ratios,
    confirming one-step controller lag rather than pressure-solver divergence.
    More importantly, a direct pre-solver field residual
    `p_rgh - (p + rho*9.81*y)` ranged from -453.5 to +3946.5 Pa, with the
    maximum at the initial free surface.  Source inspection confirmed that
    `setExprFields` wrote each new `p` through an unregistered object while the
    final expression still read the original preloaded `p`.  These screens
    cannot validate curvature behavior until this initialization defect is
    fixed.
19. The split-pressure initialization repair is verified:
    * the direct `p_rgh - (p + rho*9.81*y)` residual over all 1,903,549 cells
      changed from [-453.5,+3946.5] Pa to [0,0] Pa;
    * all 17 source/workflow tests pass;
    * a corrected `constantCurvature=0` rerun reached 0.006 s with exit code 0;
    * its first written velocity fell to 0.170 m/s at 0.00047 s, but a delayed
      free-surface peak still reached 1.424 m/s at 0.00454 s;
    * global/interface Co maxima were 0.363/0.186, alpha remained bounded, and
      gas/total balance errors stayed below \(3.9\times10^{-6}\%\).
    Thus the stale-pressure impulse was real but was not the sole source of
    the delayed hotspot.  Zero curvature remains diagnostic-only.
20. The corrected-initialization physical RDF screen exited normally at
    0.006 s but failed:
    * global/interface Co maxima were 0.366/0.303;
    * the first written velocity was 1.516 m/s at 0.00047 s;
    * the written peak was 1.775 m/s at 0.00349 s, at the initial free surface;
    * the final written velocity remained 1.344 m/s;
    * the RDF `K_` field ranged into thousands of inverse metres for an
      analytically planar interface, including +1608.6 1/m in a free-surface
      cell at the peak time;
    * alpha stayed bounded and gas/total balance errors remained below
      \(5.0\times10^{-6}\%\).
    The RDF-versus-zero-curvature first-frame difference proves a strong
    variable-curvature contribution, while the zero-curvature delayed peak
    independently proves a pressure-gravity contribution.
21. Collocated diagnostics are implemented, pass all 17 policy tests, compile,
    and execute.  At 0.006493 s they recorded:
    * Umax=1.178 m/s in a gas-dominant cell with
      \(\alpha_w=1.03\times10^{-5}\), \(\rho=1.215\ {\rm kg/m^3}\), and
      \(K=337.5\ {\rm m^{-1}}\);
    * maximum pressure-gravity and surface-tension face-force magnitudes of
      462 and 473 kPa/m at the same y=0.184 m tower-wall face;
    * maximum total residual 77.6 kPa/m near the intended free surface;
    * a probe beside the deep force hotspot changed from alpha=1 initially to
      0.437, despite being 0.219 m below the physical interface.
    The deep interfacial layer is numerical, not a geyser or gas breakthrough.
22. The tower-water selector now ends at radius 0.00735 m, halfway through the
    assumed solid wall and 1 mm short of exterior fluid.  Eighteen policy tests
    pass.  Its corrected zero-curvature rerun showed:
    * no deep wall-adjacent force hotspot;
    * peak/final written velocity 1.031/0.687 m/s, versus 1.424/0.760 m/s
      before the alpha repair;
    * maximum pressure-gravity residual 10.5 kPa/m at the intended free
      surface, versus 462 kPa/m in the deep artificial layer;
    * global/interface Co maxima 0.419/0.121;
    * alpha bounded and gas/total balance errors below
      \(1.6\times10^{-6}\%\).
    The remaining Umax cell is gas-dominant at the physical free surface.
23. The fully-wet physical RDF rerun exited normally at 0.006 s but was
    rejected:
    * the deep y=0.184 m interface and force hotspot did not recur;
    * all velocity and force hotspots were at the intended y=0.403 m free
      surface;
    * global/interface Courant maxima were 0.349/0.206;
    * first, peak and final written velocities were 1.338, 1.805 and
      1.056 m/s;
    * collocated pressure-gravity and surface-tension maxima were 82 and
      118 kPa/m at the same free-surface face;
    * alpha and phase/total balances remained clean, with no rim water or gas
      entry.
    Surface tension is the larger remaining startup-force term.  The next
    sensitivity must pair `curvFromTr=true|false` under the same hard
    `maxDeltaT` so curvature discretisation is the only model change.
24. The `curvFromTr=true` reference member of that pair completed 0.0035 s
    with `maxDeltaT=1e-5`:
    * global/interface Courant maxima were 0.115/0.021;
    * peak/final written velocities were 1.317/0.949 m/s;
    * the 119 kPa/m surface-force maximum remained at the physical interface;
    * a later 132 kPa/m pressure-gravity residual and transient velocity
      hotspot appeared in exterior gas at y=0.463 m;
    * alpha and mass balances remained clean.
    The cap itself materially changes the response, so this member cannot be
    compared directly with the adaptive-timestep result.  Complete the
    otherwise identical `curvFromTr=false` member before choosing a formula.
25. The otherwise identical `curvFromTr=false` member is complete and
    rejected:
    * global/interface Courant maxima stayed within limits at 0.125/0.024;
    * first written velocity fell from 1.317 to 1.121 m/s, but velocity then
      grew to 1.549 m/s at 0.0035 s versus 0.949 m/s for the trace formula;
    * the y=0.463 m exterior-gas pressure residual increased from 132 to
      198 kPa/m;
    * maximum surface force decreased only from 119 to 113 kPa/m.
    Retain source-default `curvFromTr=true`.  Repeat it to 0.006 s under the
    same hard cap to verify that the exterior-gas hotspot stays bounded before
    considering the 0.04 s pressure-drift window.
26. The retained trace formula completed that capped 0.006 s rerun, but the
    exterior-gas hotspot did not remain bounded:
    * global/interface Courant maxima passed at 0.152/0.024;
    * the final written velocity was the run maximum, 1.481 m/s, in pure gas
      at y=0.463 m;
    * curvature and surface force were zero at that velocity hotspot;
    * the collocated pressure-gravity/total residual reached 208 kPa/m, above
      the 119 kPa/m maximum free-surface force;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    Do not extend to 0.04 s.  Construct a discrete hydrostatic `p_rgh` field
    using the solver's gravity-force operator, verify the pre-solver face
    residual, and then repeat source-default RDF.
27. The discrete hydrostatic initializer now compiles and runs after explicitly
    linking its TwoPhaseFlow/OpenFOAM model libraries, applying
    `constrainPressure` before evaluating `fixedFluxPressure`, and registering
    the `hRef` object required by `prghPressure`.  Twenty policy tests pass.  Its
    ten-corrector physical RDF screen showed:
    * the pre-solver maximum gravity-pressure face residual fell from
      1.661 MPa/m to 1.018 kPa/m, while algebraic `p/p_rgh` consistency reached
      \(1.46\times10^{-11}\) Pa;
    * the remaining face residual is not roundoff, so the projection is an
      improvement rather than an exact facewise balance;
    * all written velocity and force hotspots stayed at the intended free
      surface; the y=0.463 m pure-gas hotspot did not occur;
    * global/interface Courant maxima were 0.365/0.204 and the first, peak and
      final written velocities were 1.279, 1.798 and 1.056 m/s;
    * dynamic pressure-gravity/surface-force maxima were 83/118 kPa/m;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    This is effectively unchanged from the otherwise identical analytic
    adaptive-timestep RDF run and is rejected.  Pair the discrete initializer
    with `maxDeltaT=1e-5` through 0.006 s to test directly whether it removes
    the analytic hard-cap run's growing exterior-gas hotspot.
28. That direct hard-cap pair is complete and the discrete member is worse:
    * global/interface Courant maxima passed at 0.214/0.024;
    * written velocity peaked at 2.173 m/s at 0.0045 s in pure exterior gas and
      remained 1.909 m/s at the end;
    * the collocated pressure-gravity residual reached 266 kPa/m with zero
      curvature, versus 208 kPa/m and 1.481 m/s for the analytic hard-cap run;
    * maximum surface force was unchanged at 119 kPa/m;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    The initial projection therefore does not cure the dynamically regenerated
    gas-pressure mode.  The final `p_rgh` correction frequently performs zero
    iterations when its initial residual is below the current \(10^{-7}\)
    absolute tolerance.  Materialise a pressure-solver tolerance control and
    repeat this exact hard-cap screen with tighter `p_rghFinal` convergence
    before changing any physical or curvature setting.
29. The otherwise identical hard-cap screen with
    `CASEB_PRESSURE_FINAL_TOLERANCE=1e-10` passed the short numerical gates:
    * global/interface Courant maxima were 0.041/0.021;
    * written velocity peaked in the first frame at 1.510 m/s and ended at
      1.187 m/s, with every velocity and force hotspot at the physical free
      surface;
    * maximum pressure-gravity residual fell from 266 to 93.4 kPa/m, while
      maximum surface force remained 119 kPa/m;
    * `p_rghFinal` now performs one or two iterations to approximately
      \(10^{-11}\) residual instead of accepting an order-\(10^{-8}\) initial
      residual with zero iterations;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    This is the first physical RDF candidate to pass the 0.006 s Courant and
    hotspot screen, not a passed drift window or hold.  Repeat the same
    \(10^{-10}\) configuration with adaptive `maxDeltaT=2.5e-4` before any
    extension to 0.04 s.
30. The paired tight-pressure adaptive-timestep screen completed normally but
    was rejected:
    * global/interface Courant maxima were 0.347/0.207, above the immutable
      0.30/0.20 observed limits;
    * written velocity peaked at 1.814 m/s at 0.00150 s and ended at
      1.051 m/s, versus 1.510/1.187 m/s under the hard cap;
    * every velocity and force hotspot stayed at the physical free surface,
      and maximum pressure-gravity/surface forces were 82/118 kPa/m;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    The tight pressure solve continues to suppress the exterior-gas mode, but
    default-target adaptive stepping still has one-step Courant overshoots.
    Screen the already-required `maxCo=0.15` timestep-sensitivity member with
    all other settings fixed.  It may enter the 0.04 s drift window only if
    observed global/interface Co stay at or below 0.30/0.20 and velocity
    remains bounded; otherwise retain `maxDeltaT=1e-5`.
31. The tight-pressure `maxCo=0.15` adaptive member passed the 0.006 s screen:
    * observed global/interface Courant maxima were 0.165/0.132;
    * written velocity peaked at 1.358 m/s at 0.00150 s and ended at
      1.015 m/s, 25.2% below the default-target adaptive peak and 10.1% below
      the hard-cap peak;
    * every velocity and force hotspot stayed at the physical free surface,
      and maximum pressure-gravity/surface forces were 83/118 kPa/m;
    * alpha and mass balances remained clean, with no rim water or gas entry.
    This selects the lower adaptive Courant target over both the rejected
    default target and the expensive hard cap for the next window.  Continue
    the same decomposed state to 0.04 s and require \(H^*\) peak-to-peak at or
    below 0.02 before starting a 1.0 s hold.

## Still required

The scientific reproduction is **not complete**.  Continue in this order:

1. Do not extend either existing `fitParaboloid` screen and do not retry
   `interpolateNormal=true` with a smaller timestep cap.  The source
   materialises plicRDF iteration and tolerance controls, and the
   static-benchmark values (`iterations=10`, `tol=1e-8`),
   `isoAlpha + fitParaboloid`, the nonphysical `constantCurvature=0`
   diagnostic, and the first `RDF + plicRDF + interpolateNormal=false`
   screen are also rejected.  The pressure split and corrected
   `constantCurvature=0` mechanism rerun and corrected physical RDF screen are
   complete, and the collocated logger has exposed an unintended deep
   wall-adjacent interface.  The alpha repair, zero-curvature verification and
   fully-wet physical RDF rerun and hard-timestep curvature-formula pair are
   complete.  `curvFromTr=false` is worse and rejected.  The retained trace
   formula's capped 0.006 s extension then exposed a growing pure-gas
   pressure-gravity hotspot.  Discrete hydrostatic initialization strongly
   reduces its pre-solver force residual, but the adaptive RDF rerun still
   fails both Courant gates and has essentially unchanged free-surface startup
   velocity.  Its directly paired hard-cap run amplifies, rather than removes,
   the dynamically regenerated exterior-gas pressure mode.  Tightening
   `p_rghFinal` to \(10^{-10}\) removes that mode and passes the hard-cap
   0.006 s screen.  The paired default-target adaptive run fails both observed
   Courant gates but does not regenerate the gas hotspot.  The required
   `maxCo=0.15` adaptive member now passes the 0.006 s gates and has the lowest
   peak velocity of the three timestep members.  Extend that exact state to
   the 0.04 s pressure-drift window next.
   Extend a candidate past the 0.04 s
   pressure-drift onset only if it stays within the declared Courant limits
   and reduces peak velocity, and only \(H^*\) peak-to-peak at or below 0.02
   may proceed to the full 1.0 s hold.  Opening runs retain the dissipative
   resistance and must be tested separately.
2. Run the opened-valve 0.5 s smoke case.
3. Run the 10.5 s base case through \(T^*\ge6\).
4. Run the refined grid and required timestep/valve/compressibility
   sensitivities.
5. Commit only compact CSV/JSON/plots; never commit meshes, time directories,
   `processor*`, `postProcessing`, logs, or dynamic-code output.
6. Update the PR without claiming a completed experiment until the acceptance
   fields in `outputs/openfoam_3d_metrics.json` pass.

The initial 0.001 s hold needed about 6.5 minutes on four ranks.  Its maximum
Courant number stayed near 0.3 and the timestep settled near \(1.4\times10^{-5}\)
s, so runtime cost and startup velocity should be reviewed before launching
the full window.

## Re-entry commands

```bash
git fetch origin cursor/test1-caseb-3d-0614
git switch cursor/test1-caseb-3d-0614
cd tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d

# Read these first.
less HANDOFF.md
less PAPER_AUDIT.md
less README.md

# Build the pinned geometric-VOF solver.
./build_twophaseflow.sh

# Recreate stages from source.
./Allclean
CASEB_STAGE=mesh CASEB_MESH=base ./Allrun

./Allclean
CASEB_STAGE=hold CASEB_MESH=base OPENFOAM_NP=4 ./Allrun

./Allclean
CASEB_STAGE=smoke CASEB_MESH=base CASEB_END_TIME=0.5 \
  OPENFOAM_NP=4 ./Allrun
```

The environment requires OpenFOAM.com v2512, Gmsh with Python bindings,
NumPy, and Matplotlib.  Runtime recovery is available through
`./Allrun.resume` only when `processor*` state exists in the same VM.

## Prompt for a fresh cloud agent

For the complete task specification, copy the entire fenced block from
`NEW_AGENT_PROMPT.md`.  The shorter prompt below is only a routing summary.

> Continue Geysering Test 1 Case B as a genuinely 3-D OpenFOAM validation from
> PR #10 and branch `cursor/test1-caseb-3d-0614`. Read
> `openfoam/3d/HANDOFF.md`, `PAPER_AUDIT.md`, and `README.md` first. Preserve
> the paper audit and authenticity rules, do not use the old 2-D branch, and
> do not modify Case A. Start by verifying the short closed-valve diagnostic
> fix, then complete hold, smoke, base/refined full runs, sensitivities, and
> compact experiment/1-D/3-D comparisons. Never claim completion without
> committed runtime evidence and passing acceptance fields.

