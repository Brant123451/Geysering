# Geysering validation cases

This repository organizes the numerical validation work into three experimental
campaigns. Every modeled Case owns its configuration, experiment data, frozen
model, scripts, references, compact outputs, and optional OpenFOAM setup.

## Test matrix

- `tests/test_01_vw2011`: Vasconcelos & Wright (2011), four modeled/auxiliary
  Cases including the primary non-geysering and geysering branches.
- `tests/test_02_cong2017`: Cong, Chan & Lee (2017), Series B runs BH1–BH7 and
  the campaign-level criterion-map study.
- `tests/test_03_liu2020`: Liu, Shao & Zhu (2020), representative Cases A2,
  B3, and C9.

Original papers are under `references/`; the bilingual manuscript is under
`paper/`; migration and validation utilities are under `tools/`.

## Cloud Agents

Cloud Agents must use the bootstrap instructions in
`docs/cloud-agent-bootstrap.md`, work on exactly one Case, and push only to an
independent branch. Generated OpenFOAM time directories, `processor*`,
`postProcessing`, meshes, logs, and frame sequences are intentionally excluded
from Git; source cases and compact comparison artifacts are tracked normally.

Validate the repository layout with:

```bash
python tools/validate_layout.py --root .
python -m unittest discover -s tools/tests -v
```
