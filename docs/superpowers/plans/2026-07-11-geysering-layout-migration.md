# Geysering Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `E:\Geysering` into three Test trees with 14 self-contained Case directories and a parallel English/Chinese paper framework, while preserving reproducibility and removing verified duplicates and temporary files.

**Architecture:** A manifest-driven migration script records the complete pre-migration inventory, applies collision-safe moves, generates Case manifests, and writes move/deletion ledgers. A separate validator enforces the target directory contract, scans for stale path references, and verifies that the paper projects still build.

**Tech Stack:** Python 3 standard library, PowerShell, Git, LaTeX (`pdflatex`/`xelatex`), existing Case scripts and OpenFOAM configuration.

---

## Execution Constraints

- Work locally on the clean `main` working tree; do not commit or push without explicit user authorization.
- Do not run full OpenFOAM simulations during migration.
- Do not move active runtime state until process checks show no solver is writing under `E:\Geysering`.
- Never delete a file solely by name. Temporary files require a reference scan; duplicates require equal SHA-256.
- Do not use `git add -f .`.
- Keep `docs/file-layout/pre-migration-files.csv`, `move-manifest.csv`, and `deletion-manifest.csv` through final verification.

## Target Case IDs

```python
EXPECTED_CASES = {
    "test_01_vw2011": [
        "A_Dt57p1_Ha0305_Yfs0356",
        "B_Dt12p7_Ha0610_Yfs0356",
        "Fig10_Dt57p1_Ha0305_Yfs0254",
        "Fig11_Dt12p7_Ha0305_Yfs0254",
    ],
    "test_02_cong2017": [
        "BH1_Dr16_H066_L061",
        "BH2_Dr21_H066_L061",
        "BH3_Dr26_H066_L061",
        "BH4_Dr31_H066_L061",
        "BH5_Dr36_H066_L061",
        "BH6_Dr41_H066_L061",
        "BH7_Dr46_H066_L061",
    ],
    "test_03_liu2020": [
        "A2_Q20to100_openchannel_nogeyser",
        "B3_Q20to100_fullpipe_geyser",
        "C9_Q25to40_hr03_airpocket",
    ],
}
```

### Task 1: Establish Safety Baseline and Full Inventory

**Files:**
- Create: `tools/layout_inventory.py`
- Create: `docs/file-layout/pre-migration-files.csv`
- Create: `docs/file-layout/pre-migration-summary.json`

- [ ] **Step 1: Verify the working tree and active processes**

Run:

```powershell
git status --short --branch
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -match 'E:\\Geysering' -and
    $_.Name -match 'foam|interFoam|mpirun|python'
  } |
  Select-Object ProcessId, Name, CommandLine
```

Expected: Git reports a clean baseline except the approved design/plan files; no simulation process is writing to this repository. If a process is active, stop before migration and record its resume command.

- [ ] **Step 2: Implement the inventory utility**

The script must:

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HASH_LIMIT = 25 * 1024 * 1024
SKIP_DIRS = {".git"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path) if stat.st_size <= HASH_LIMIT else "",
            }
        )
    return rows
```

It must write a UTF-8 CSV and a JSON summary containing file count, total bytes, unhashed-large-file count, and counts by top-level directory.

- [ ] **Step 3: Generate and validate the baseline**

Run:

```powershell
python tools/layout_inventory.py `
  --root "E:\Geysering" `
  --csv "docs/file-layout/pre-migration-files.csv" `
  --summary "docs/file-layout/pre-migration-summary.json"
```

Expected: approximately 16,000 files are indexed; `.git` is excluded; every path is relative to the repository.

- [ ] **Step 4: Record intended commit boundary**

Intended commit: `docs: record geysering layout migration baseline`
Do not create the commit unless the user explicitly requests it.

### Task 2: Build the Layout Contract Validator First

**Files:**
- Create: `tools/validate_layout.py`
- Create: `tools/tests/test_validate_layout.py`
- Create: `tools/tests/__init__.py`

- [ ] **Step 1: Write failing validator tests**

```python
import tempfile
import unittest
from pathlib import Path

from tools.validate_layout import validate_case, validate_repository


class LayoutValidatorTests(unittest.TestCase):
    def test_case_requires_contract_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            errors = validate_case(case)
            self.assertIn("missing README.md", errors)
            self.assertIn("missing manifest.yaml", errors)
            self.assertIn("missing directory: data", errors)

    def test_repository_requires_all_expected_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            errors = validate_repository(Path(tmp))
            self.assertTrue(any("test_01_vw2011" in error for error in errors))

    def test_complete_minimal_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            (case / "README.md").write_text("# Case\n", encoding="utf-8")
            (case / "manifest.yaml").write_text("id: case\n", encoding="utf-8")
            for name in ("config", "data", "model", "scripts", "reference", "outputs"):
                (case / name).mkdir()
            self.assertEqual([], validate_case(case))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m unittest tools.tests.test_validate_layout -v
```

Expected: import or assertion failures because the validator is not implemented.

- [ ] **Step 3: Implement the minimal validator**

```python
REQUIRED_FILES = ("README.md", "manifest.yaml")
REQUIRED_DIRS = ("config", "data", "model", "scripts", "reference", "outputs")


def validate_case(case: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (case / name).is_file():
            errors.append(f"missing {name}")
    for name in REQUIRED_DIRS:
        if not (case / name).is_dir():
            errors.append(f"missing directory: {name}")
    return errors
```

`validate_repository()` must enforce all 14 Case IDs, reject root/Test-level `_tmp*`, `_dev*`, `_slugdbg*`, `.bak*`, and reject old top-level campaign paths after migration.

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
python -m unittest tools.tests.test_validate_layout -v
```

Expected: all tests pass.

- [ ] **Step 5: Record intended commit boundary**

Intended commit: `test: define geysering directory contract`
Do not create the commit unless explicitly requested.

### Task 3: Define the Declarative Migration Map

**Files:**
- Create: `docs/file-layout/migration-map.json`
- Create: `docs/file-layout/ownership-rules.md`

- [ ] **Step 1: Add top-level moves**

The map must include:

```json
{
  "campaigns": [
    {
      "source": "Vasconcelos_Wright_2011_Geysering",
      "target": "tests/test_01_vw2011"
    },
    {
      "source": "Cong_Chan_Lee_2017_Geyser_Horizontal_Pipe_Vertical_Shaft",
      "target": "tests/test_02_cong2017"
    },
    {
      "source": "Liu_Shao_Zhu_2020_Junction_Chamber_Geyser",
      "target": "tests/test_03_liu2020"
    }
  ]
}
```

Add exact source-to-target mappings for existing Case directories, shared paper/reference files, Test-level scripts, Test-level outputs, `papers`, `reproduction`, and root temporary files.

- [ ] **Step 2: Document ownership precedence**

Apply this order:

1. Exact Case parameter/name match;
2. README or script hard-coded Case path;
3. Output metadata/CSV parameter match;
4. Test-level shared asset;
5. Unique legacy archive;
6. Temporary deletion candidate.

No ambiguous file may be deleted. Ambiguous files go to `tests/<test>/_archive/unclassified` with their original path recorded.

- [ ] **Step 3: Validate every old top-level file has a disposition**

Run a script check comparing the inventory with the migration map. Expected: every source path is classified as move, retain, archive, generated-runtime exclusion, or delete candidate.

### Task 4: Implement Collision-Safe Migration Tooling

**Files:**
- Create: `tools/migrate_geysering_layout.py`
- Create: `tools/tests/test_migrate_geysering_layout.py`
- Create at runtime: `docs/file-layout/move-manifest.csv`
- Create at runtime: `docs/file-layout/deletion-manifest.csv`

- [ ] **Step 1: Test collision handling**

Tests must prove:

- missing source is reported;
- different-content target collision aborts;
- identical target collision records a dedupe candidate;
- moves cannot escape the repository root;
- `--dry-run` changes no files.

- [ ] **Step 2: Implement safe path operations**

Core behavior:

```python
def safe_move(root: Path, source_rel: str, target_rel: str, dry_run: bool) -> Move:
    source = (root / source_rel).resolve()
    target = (root / target_rel).resolve()
    source.relative_to(root.resolve())
    target.relative_to(root.resolve())
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() and source.is_file() and target.is_file():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"content collision: {target}")
        return Move(source_rel, target_rel, "duplicate")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return Move(source_rel, target_rel, "move")
```

Every operation must append to the move ledger before the next operation begins.

- [ ] **Step 3: Run unit tests**

Run:

```powershell
python -m unittest tools.tests.test_migrate_geysering_layout -v
```

Expected: all migration safety tests pass.

- [ ] **Step 4: Run the full dry run**

Run:

```powershell
python tools/migrate_geysering_layout.py `
  --root "E:\Geysering" `
  --map "docs/file-layout/migration-map.json" `
  --dry-run
```

Expected: zero unresolved collisions, zero root escapes, and a complete proposed move/delete summary.

### Task 5: Migrate Test 1 (VW2011)

**Files:**
- Move: `Vasconcelos_Wright_2011_Geysering/**`
- Create: `tests/test_01_vw2011/README.md`
- Create: four `tests/test_01_vw2011/cases/*/manifest.yaml`

- [ ] **Step 1: Move the two existing primary Case trees**

Map:

```text
caseA_Dt57p1_Ha0305_Yfs0356
  -> cases/A_Dt57p1_Ha0305_Yfs0356
caseB_Dt12p7_Ha0610_Yfs0356
  -> cases/B_Dt12p7_Ha0610_Yfs0356
```

Normalize `digitized` to `data/digitized`, `paper_scans` to `reference/paper_scans`, root Case scripts to `scripts`, and OpenFOAM directories under `openfoam`.

- [ ] **Step 2: Materialize Fig.10 and Fig.11 Cases**

Create:

```text
cases/Fig10_Dt57p1_Ha0305_Yfs0254
cases/Fig11_Dt12p7_Ha0305_Yfs0254
```

Move only parameter-specific CSV/JSON/figures and scripts from the original A/B output trees. Copy the correct frozen model into each new Case and write an explicit config containing the changed water level/header values.

- [ ] **Step 3: Classify shared and legacy assets**

- Source papers and parameter notes -> `_shared/reference` and `_shared/metadata`.
- Cross-Case production utilities -> `_shared/tools`.
- Unique old implementations -> `_archive/legacy`.
- `_caseA_unified_check` useful script -> `_archive/cross_case_checks`; disposable output -> deletion ledger.
- Group-level `outputs/vw2011_network` assets -> matching Case output or verified duplicate deletion.

- [ ] **Step 4: Generate README and manifests**

Each manifest must record Test ID, Case ID, parameters, source paper, run entrypoint, frozen model path, expected outputs, and validation status.

- [ ] **Step 5: Validate Test 1**

Run:

```powershell
python tools/validate_layout.py --root "E:\Geysering" --test test_01_vw2011
```

Expected: four Case contracts pass; no Test-root Case-specific output remains.

### Task 6: Migrate Test 2 and Materialize BH1–BH7

**Files:**
- Move: `Cong_Chan_Lee_2017_Geyser_Horizontal_Pipe_Vertical_Shaft/**`
- Create: seven `tests/test_02_cong2017/cases/*/manifest.yaml`
- Create: seven Case parameter/config files
- Move: scan package to `tests/test_02_cong2017/studies/criterion_map`

- [ ] **Step 1: Move detailed BH1 and BH6 trees**

Map:

```text
caseA_BH1_Dr16_H066_L061_geyser
  -> cases/BH1_Dr16_H066_L061
caseB_BH6_Dr41_H066_L061_nogeyser
  -> cases/BH6_Dr41_H066_L061
```

- [ ] **Step 2: Extract BH2–BH5 and BH7 from Series B**

For each run, create the Case contract and extract the matching row(s) from `seriesB_fullsync.csv` into `data/series_b_measurements.csv`. Create a Case-specific config with `Dr`, `H0`, `L0`, measured class, modeled class, and shared scan provenance.

- [ ] **Step 3: Provide frozen runnable code**

Copy the validated full-synchronous model and a Case-specific runner into each Case `model/` and `scripts/`. The runner must write only inside its own `outputs/`.

- [ ] **Step 4: Move the scan package**

Move `scan_seriesB_criterion_map` to `studies/criterion_map`; remove byte-identical duplicates from the old group-level `outputs` after recording hashes.

- [ ] **Step 5: Classify old solvers**

Move unique `real_solver` code to `_archive/legacy_real_solver`; move any still-required common source to `_shared/model_sources`. Old BH1-only runner scripts go to the BH1 Case or are deleted if superseded and unreferenced.

- [ ] **Step 6: Validate Test 2**

Expected: BH1–BH7 each have an independent manifest, config, data slice, model, runner, and outputs; the criterion map remains a clearly labeled Campaign study.

### Task 7: Migrate Test 3 (Liu2020)

**Files:**
- Move: `Liu_Shao_Zhu_2020_Junction_Chamber_Geyser/**`
- Create: three `tests/test_03_liu2020/cases/*/manifest.yaml`

- [ ] **Step 1: Move A2, B3, and C9**

Map each existing Case directory to its target ID, normalize its standard subdirectories, and place A2 OpenFOAM assets under `openfoam`.

- [ ] **Step 2: Move Test-level reference material**

Move paper parameters and shared metadata to `_shared/metadata` and the source paper to `_shared/reference`.

- [ ] **Step 3: Remove Case-root debug logs**

Scan each production script/README for references to `_run*.txt`, `_comp*.txt`, and similar logs. Delete unreferenced logs and record them; preserve meaningful unique diagnostic reports under `outputs/diagnostics`.

- [ ] **Step 4: Document C9 phase-2 scope**

The C9 README and manifest must state that phase-2 is not computed and is not represented as a separate Case.

- [ ] **Step 5: Validate Test 3**

Expected: three Case contracts pass and the Test root contains only README, `_shared`, `cases`, and optional `_archive`.

### Task 8: Rebuild the Parallel Paper Framework

**Files:**
- Move: `paper/main.tex` -> `paper/en/main.tex`
- Move: `paper/sections` -> `paper/en/sections`
- Move/rename: `paper/zh/main_zh.tex` -> `paper/zh/main.tex`
- Move: shared figures/bibliography/styles -> `paper/shared/*`
- Move final PDFs -> `paper/build/en` and `paper/build/zh`
- Move historical PDFs -> `paper/archive`
- Create: `paper/shared/figures/manifest.yaml`

- [ ] **Step 1: Separate English and Chinese sources**

Keep language-specific source under its language root. Shared bibliography and figure paths must be relative and platform-independent.

- [ ] **Step 2: Create curated figure provenance manifest**

Each entry must contain:

```yaml
- id: vw2011_case_a_pressure
  test: test_01_vw2011
  case: A_Dt57p1_Ha0305_Yfs0356
  source: tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/outputs/...
  published: paper/shared/figures/...
```

- [ ] **Step 3: Update LaTeX paths**

Update `\input`, `\includegraphics`, bibliography, and style paths in both language projects. Do not alter scientific text except where a path or filename changed.

- [ ] **Step 4: Configure build output**

Use:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory "..\build\en" main.tex

xelatex -interaction=nonstopmode -halt-on-error `
  -output-directory "..\build\zh" main.tex
```

Run each engine twice for references.

- [ ] **Step 5: Archive and clean**

Move `main_rev.pdf`, `main_zh_v3.pdf`, `main_zh_v4.pdf`, and similar historical releases to `paper/archive`; delete LaTeX cache files after successful builds.

### Task 9: Remove Verified Duplicates and Temporary Files

**Files:**
- Update: `docs/file-layout/deletion-manifest.csv`
- Remove after verification: root `_tmp_*`, `_slugdbg*`, Test `_dev_*`, `.bak*`, caches, and duplicate outputs

- [ ] **Step 1: Scan references before deletion**

Search all production Python, PowerShell, shell, Markdown, TeX, JSON, YAML, and HTML files for every deletion candidate basename and old path.

- [ ] **Step 2: Hash duplicate groups**

Only delete a duplicate when:

```python
source.stat().st_size == retained.stat().st_size
and sha256(source) == sha256(retained)
```

Record both paths and retained ownership.

- [ ] **Step 3: Delete temporary/backup files**

Delete only candidates that are unreferenced and classified as temporary, cache, backup, or pure log. Unique historical code goes to `_archive/legacy`.

- [ ] **Step 4: Verify forbidden patterns are gone**

Expected: no `_tmp_*`, `_dev_*`, `_slugdbg*`, `.bak*`, `__pycache__`, or Test-root ambiguous outputs remain outside `_archive`.

### Task 10: Replace Git Ignore Rules and Update Documentation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `tests/README.md`
- Create: `docs/file-layout/README.md`

- [ ] **Step 1: Replace the global deny-all ignore**

Use category exclusions:

```gitignore
__pycache__/
*.py[cod]
*.log
*.aux
*.out
*.spl

**/processor*/
**/postProcessing/
**/openfoam/**/[0-9]/
**/openfoam/**/[0-9].[0-9]*/
**/frames/
**/riser_frames/

docs/file-layout/*.tmp
```

Add explicit path exclusions for confirmed large runtime directories that do not match these generic patterns. Keep compact CSV/JSON/PNG/PDF/GIF/HTML outputs visible to Git.

- [ ] **Step 2: Update the root index**

Document the three Tests, 14 Case IDs, paper layout, source-paper library, validation command, and policy for large outputs.

- [ ] **Step 3: Check Git visibility**

Run:

```powershell
git status --short
git check-ignore -v `
  "tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/manifest.yaml" `
  "paper/en/main.tex" `
  "paper/build/en/main.pdf"
```

Expected: the three representative assets are not ignored; heavy runtime directories are ignored.

### Task 11: Update All Runtime and Documentation Paths

**Files:**
- Modify: Python/shell/PowerShell scripts under `tests/**`
- Modify: Case/Test/root READMEs
- Modify: paper sources under `paper/**`

- [ ] **Step 1: Search stale absolute paths**

Search for:

```text
E:\Geysering\Vasconcelos_Wright_2011_Geysering
E:\Geysering\Cong_Chan_Lee_2017_Geyser_Horizontal_Pipe_Vertical_Shaft
E:\Geysering\Liu_Shao_Zhu_2020_Junction_Chamber_Geyser
E:\Geysering\paper\main
E:\Geysering\reproduction
```

- [ ] **Step 2: Convert production scripts to path-relative resolution**

Use:

```python
CASE_ROOT = Path(__file__).resolve().parents[1]
DATA = CASE_ROOT / "data"
OUTPUTS = CASE_ROOT / "outputs"
MODEL = CASE_ROOT / "model"
REFERENCE = CASE_ROOT / "reference"
```

No production script may hard-code the old repository path.

- [ ] **Step 3: Compile/import smoke checks**

Run:

```powershell
python -m compileall -q tests tools
python tools/validate_layout.py --root "E:\Geysering"
```

Expected: no syntax errors and no layout errors.

### Task 12: Final Verification and Handoff

**Files:**
- Create: `docs/file-layout/post-migration-summary.json`
- Create: `docs/file-layout/verification-report.md`

- [ ] **Step 1: Run unit and structure tests**

```powershell
python -m unittest discover -s tools/tests -v
python tools/validate_layout.py --root "E:\Geysering"
```

Expected: all tests pass; 14/14 Case contracts valid.

- [ ] **Step 2: Run lightweight Case smoke checks**

Execute each Case runner in validation/dry-run mode. Do not launch full OpenFOAM. Confirm each runner resolves only files within its Case and writes only to its `outputs`.

- [ ] **Step 3: Build both papers**

Build English with `pdflatex` twice and Chinese with `xelatex` twice. Expected: both exit 0 and final PDFs exist in `paper/build/en` and `paper/build/zh`.

- [ ] **Step 4: Scan for stale paths and forbidden clutter**

Expected:

- no old top-level campaign directories;
- no old absolute path references in production files;
- no root/Test-root temp or backup files;
- no unknown duplicate compact outputs;
- no Case importing another Case.

- [ ] **Step 5: Generate post-migration inventory**

Compare pre/post counts, bytes, moved paths, deleted bytes, archived files, tracked/ignored counts, and final PDF metadata.

- [ ] **Step 6: Review Git diff without staging**

Run:

```powershell
git status --short
git diff --stat
git diff -- .gitignore README.md
```

Expected: changes group cleanly into infrastructure, Test 1, Test 2, Test 3, and paper. No commit or push occurs.

- [ ] **Step 7: Record intended commit sequence**

If the user later explicitly requests commits:

1. `chore: add layout migration tooling and manifests`
2. `refactor: organize geysering test cases`
3. `refactor: organize bilingual paper sources`
4. `chore: track compact reproducibility artifacts`

Do not create these commits without explicit authorization.
