# Geysering manuscript project map

## Manuscript sources

- `paper/main.tex`: English submission entry point and abstract.
- `paper/sections/introduction.tex`: background, gap, contribution, and campaign plan.
- `paper/sections/model.tex`: compact description of the companion two-fluid framework.
- `paper/sections/tests.tex`: principal experimental campaigns, quantitative results, figures, and tables.
- `paper/sections/discussion.tex`: interpretation, limitations, and conclusions.
- `paper/sections/bibliography.tex`: manual `thebibliography` entries.
- `paper/figures/`: submission figure artifacts. Trace their numbers to upstream data before editing prose.
- `paper/zh/main_zh.tex`: Chinese scientific review manuscript.
- `paper/main.pdf` and `paper/zh/main_zh.pdf`: compiled outputs, not editable sources.

## Campaign 1: Vasconcelos and Wright (2011)

- Test A case: `tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/`
- Test A core metrics: `outputs/caseA_comparison_metrics.json`
- Test B case: `tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/`
- Test B core metrics: `outputs/caseB_comparison_metrics.json`
- Each case `manifest.yaml` records case identity and configuration.
- Fig. 10 and Fig. 11 reconstruction cases live under `cases/Fig10_*` and `cases/Fig11_*`.
- Treat 2D/3D OpenFOAM subdirectories as supporting or exploratory evidence unless the manuscript scope is explicitly expanded.

## Campaign 2: Cong, Chan, and Lee (2017)

- Series B cases: `tests/test_02_cong2017/cases/BH1_*` through `BH7_*`.
- Per-case summaries: `outputs/series_b_model_summary.csv`.
- Signature comparison metrics include `BH1_*/outputs/caseA_comparison_metrics.json` and `BH6_*/outputs/caseB_comparison_metrics.json`.
- Criterion-map workflow: `tests/test_02_cong2017/studies/criterion_map/`.
- Principal aggregated evidence:
  - `outputs/cong2017_BH_model_vs_measured.csv`
  - `outputs/cong2017_criterion_scan.csv`
  - `outputs/seriesB_fullsync.csv`
  - `outputs/criterion_scan_fullsync_*.csv`
  - `outputs/cong2017_reproduction_summary.txt`
- Inspect the generation scripts in the criterion-map directory before regenerating paper figures.

## Campaign 3: Liu, Shao, and Zhu (2020)

- A2 case: `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/`
- A2 metrics: `outputs/caseA_metrics.json`
- B3 case: `tests/test_03_liu2020/cases/B3_Q20to100_fullpipe_geyser/`
- B3 metrics: `outputs/caseB_metrics.json`
- C9 case: `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/`
- C9 metrics: `outputs/caseC_metrics.json`
- Read each `manifest.yaml` before interpreting boundary conditions or configuration variants.

## Evidence precedence

Use this order when sources disagree:

1. Original published experimental source for statements about measurements.
2. Frozen case manifest and the exact archived metrics used to generate the manuscript figure.
3. Figure-generation or comparison script and its digitized input.
4. Current manuscript table or caption.
5. README prose, handoff notes, logs, or filenames.

Do not silently choose among conflicting artifacts. Report the conflict, identify the likely stale artifact, and ask for scientific judgment when the resolution changes a conclusion.

## Current project constraints

- The LaTeX class is `elsarticle` in review/preprint mode.
- The target journal is not yet selected.
- The paper is an application/validation companion to a methods paper.
- The main scientific promise is one frozen one-dimensional solver configuration without prescribed release rates, scripted geyser heights, or per-case fitting.
- The bibliography is manual rather than BibTeX.
- The methods-paper citation remains provisional and must not be assigned fabricated publication metadata.
