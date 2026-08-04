# Cloud Agent prompts

All Cloud Agents must start from:

`bootstrap/geysering-cases-20260711`

The repository contract and generated-file policy are documented in
`docs/cloud-agent-bootstrap.md`. One Agent owns one Case directory only.

## Common prompt

Copy this block and replace the three placeholders from the launch matrix.

```text
You are responsible for exactly one Geysering validation Case.

Base branch: bootstrap/geysering-cases-20260711
Create and push your own branch: <AGENT_BRANCH>
Owned directory: <CASE_ROOT>
Source paper: <PAPER_PATH>
Case objective: <OBJECTIVE>

Before changing anything:
1. Read README.md, manifest.yaml, config/case.json, the source paper, and all
   existing digitized data/reference scans in the owned Case.
2. Audit every geometry dimension, parameter, initial condition, boundary
   condition, probe location, and experimental target against the paper.
3. Treat an existing OpenFOAM folder as a pilot until the audit proves it is
   paper-faithful. Never invent missing dimensions or comparison data.

Work only inside <CASE_ROOT>. Shared papers and Test-level metadata are
read-only. Do not edit another Case, paper/, .gitignore, repository-root files,
or main. Do not merge or force-push.

Use OpenFOAM v2512 from .cursor/Dockerfile when CFD is part of the objective.
Generated time directories, processor*, postProcessing, meshes, logs, and frame
sequences must not be committed. Commit source cases plus compact CSV/JSON,
comparison figures, metrics, and an updated Case README.

Before finishing:
- run the Case's available checks/post-processing;
- run `python tools/validate_layout.py --root .`;
- clearly distinguish smoke-test, incomplete, and validated results;
- commit and push your branch;
- report branch name, commit SHA, exact commands run, quantitative comparison,
  remaining uncertainty, and any paper data that could not be recovered.
```

## Launch matrix

### Test 1 — Vasconcelos & Wright (2011)

Paper for all four Agents: `references/vasconcelos2011.pdf`

- Branch: `cursor/test1-case-a-openfoam`
  - Case: `tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356`
  - Objective: independently audit the existing 3-D OpenFOAM model against
    Case A, rerun it, and compare pressure and water-level histories with the
    digitized experiment. Keep the 2-D model diagnostic-only.
- Branch: `cursor/test1-case-b-openfoam`
  - Case: `tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356`
  - Objective: build a low-cost 2-D OpenFOAM validation of the geysering branch,
    preserve paper geometry/conditions, and compare with the digitized Case B
    experiment. State explicitly which circular-area or junction effects 2-D
    cannot reproduce.
- Branch: `cursor/test1-fig10-validation`
  - Case: `tests/test_01_vw2011/cases/Fig10_Dt57p1_Ha0305_Yfs0254`
  - Objective: reproduce Fig. 10 from the frozen model and digitized curves,
    verify axis calibration and metrics, and make the Case independently
    rerunnable.
- Branch: `cursor/test1-fig11-validation`
  - Case: `tests/test_01_vw2011/cases/Fig11_Dt12p7_Ha0305_Yfs0254`
  - Objective: reproduce Fig. 11 from the frozen model and digitized curves,
    verify axis calibration and metrics, and make the Case independently
    rerunnable.

### Test 2 — Cong, Chan & Lee (2017)

Paper for all seven Agents: `references/cong2017.pdf`

Case directories:

- `tests/test_02_cong2017/cases/BH1_Dr16_H066_L061`
- `tests/test_02_cong2017/cases/BH2_Dr21_H066_L061`
- `tests/test_02_cong2017/cases/BH3_Dr26_H066_L061`
- `tests/test_02_cong2017/cases/BH4_Dr31_H066_L061`
- `tests/test_02_cong2017/cases/BH5_Dr36_H066_L061`
- `tests/test_02_cong2017/cases/BH6_Dr41_H066_L061`
- `tests/test_02_cong2017/cases/BH7_Dr46_H066_L061`

Use branch names `cursor/test2-bh1-validation` through
`cursor/test2-bh7-validation`. For each Agent, the objective is:

```text
Validate this single Series-B run against its row in the Case-owned experiment
CSV and the Cong (2017) paper. Preserve the measured riser diameter, initial
water level, pocket length, flow conditions, and reported geyser/no-geyser
classification. Reproduce compact time-series/metrics with the frozen model.
If adding OpenFOAM, use 2-D only as a smoke test; the final T-junction/riser
validation must preserve the physical 3-D circular areas and junction volume.
```

### Test 3 — Liu, Shao & Zhu (2020)

Paper for all three Agents: `references/liu2020.pdf`

- Branch: `cursor/test3-a2-openfoam`
  - Case: `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser`
  - Objective: verify and complete the existing 3-D OpenFOAM pilot for A2.
    Match the 5.80 m upstream pipe, 0.30 x 0.30 x 0.45 m junction chamber,
    5.95 m downstream pipe, 0.06 m x 1.22 m riser, Q=20 to 100 L/s ramp,
    downstream open-channel condition, PT2/PT3 locations, and no-geyser result.
- Branch: `cursor/test3-b3-openfoam`
  - Case: `tests/test_03_liu2020/cases/B3_Q20to100_fullpipe_geyser`
  - Objective: build a paper-faithful 3-D OpenFOAM validation for B3. It should
    differ from A2 only where the paper specifies the downstream initial/
    boundary condition, then compare pressure and geyser height histories.
- Branch: `cursor/test3-c9-openfoam`
  - Case: `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket`
  - Objective: build a paper-faithful 3-D OpenFOAM validation for C9, including
    the specified trapped-air condition, flow transition, pressure probes, and
    oscillatory geyser response. Compare against the Case-owned digitized data.

Test 3 final validation is 3-D because the rectangular junction chamber,
circular pipes, central riser, side inflow/outflow, and chamber free surface
cannot be preserved by a planar 2-D model. A 2-D setup may be used only for a
clearly labeled smoke test.
