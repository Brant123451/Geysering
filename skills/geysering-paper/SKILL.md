---
name: geysering-paper
description: Guide evidence-grounded writing, revision, audit, and bilingual synchronization of the Geysering SCI manuscript in E:\Geysering, including LaTeX sections, figures, tables, citations, quantitative claims, target-journal preparation, and reviewer responses. Use when working on paper/main.tex, paper/sections/*.tex, paper/zh/main_zh.tex, manuscript figures, paper claims derived from tests/, or submission and rebuttal tasks for this project.
---

# Geysering Paper

## Purpose

Develop the Geysering application paper without separating prose from its numerical and experimental evidence. Preserve the existing LaTeX project, distinguish verified results from exploratory outputs, and keep the English submission manuscript and Chinese review manuscript scientifically aligned.

## Start Every Task

1. Read `references/project-map.md`.
2. Inspect `git status --short` and preserve all user changes. Never clean, replace, or stage unrelated simulation outputs.
3. Read the affected manuscript section, its cited figures or tables, and the corresponding source artifacts before proposing or making edits.
4. State whether the task changes scientific content, presentation only, or both.
5. Keep the current `elsarticle` structure until the target journal is explicitly selected.

## Evidence Rules

- Never invent measurements, computed values, equations, settings, citations, DOI metadata, or mechanisms.
- Treat a manuscript figure as a presentation artifact, not the sole source of a number. Trace quantitative claims to a manifest, JSON, CSV, generation script, or cited experimental source.
- Label each major claim during revision as `supported`, `partial`, or `missing`. Weaken, source, or remove claims that are not supported.
- Distinguish published measurements, digitized literature data, one-dimensional model output, and exploratory 2D/3D OpenFOAM output.
- Do not promote files under directories named `attempt`, `FAIL`, `sensitivity`, `smoke`, `partial`, or similar into manuscript evidence without explicit user confirmation and a documented selection rationale.
- Preserve the frozen-solver premise. If a result uses a different configuration or per-case adjustment, disclose it rather than blending it into the frozen campaign.
- Use primary papers or publisher records for literature claims. Never create a reference from memory.
- Keep project paths and evidence notes outside final manuscript prose unless they belong in a reproducibility or data-availability statement.

## Manuscript Workflow

1. Build or refresh a compact claim-evidence map for the affected section.
2. Stabilize figures, tables, metrics, and captions before rewriting the Results narrative.
3. Revise in this order when the whole story changes: Results, Discussion, Introduction, Abstract, then Title.
4. Make each Results paragraph lead with one defensible finding and connect it to the exact figure, table, or metric that supports it.
5. Separate observation from interpretation. Put mechanistic explanation and limitations in Discussion unless brief interpretation is required for navigation.
6. Keep the three campaign roles distinct:
   - Campaign 1: branch selection and transient reproduction for Vasconcelos and Wright (2011).
   - Campaign 2: regime classification and parameter-space performance for Cong et al. (2017).
   - Campaign 3: prototype-motivated junction-chamber comparisons for Liu et al. (2020).
7. Maintain one visible contribution chain across title, abstract, introduction, Results, discussion, and conclusions.

## Bilingual Synchronization

- Treat `paper/main.tex` plus `paper/sections/*.tex` as the English submission source of truth unless the user explicitly says otherwise.
- Treat `paper/zh/main_zh.tex` as the Chinese scientific review version.
- After changing a result, number, limitation, conclusion, figure meaning, or citation in one language, update or flag the counterpart in the other language during the same task.
- Preserve equations, symbols, units, case identifiers, reference keys, and figure numbering across languages.
- Prefer scientific equivalence over sentence-by-sentence literal translation.

## Citations and References

- Preserve the existing manual `thebibliography` workflow in `paper/sections/bibliography.tex`. Do not convert it to BibTeX unless asked.
- Verify that every `\cite{key}` resolves to exactly one bibliography item and every major literature claim is supported by the cited source.
- Do not rely on `citation-verifier/scripts/scan_citations.py` for bibliography coverage in this project: its current scanner does not parse manual `\bibitem{key}` entries. Compare the `\cite{}` and `\bibitem{}` key sets directly and confirm the LaTeX log.
- Verify title, author order, year, journal, volume, pages or article number, and DOI against primary records before finalizing a reference.
- Treat the companion methods-paper entry as provisional until its submission or publication metadata is confirmed.
- Use `citation-verifier` for reference integrity and `submission-audit` for late-stage cross-checks.

## Companion Skills

Load only what the task needs:

- Use `scientific-writing` for section drafting and scientific prose.
- Use `manuscript-optimizer` for claim structure, terminology, and cross-section narrative repair.
- Use `results-section-revision` for Results flow after figures and evidence are stable.
- Use `figure-planner` before adding or redesigning manuscript figures.
- Use `citation-verifier` for citation and bibliography checks.
- Use `submission-audit` for pre-submission readiness.
- Use `rebuttal-response` for reviewer comments and response letters.
- Use the installed PDF skill to render and visually inspect the compiled manuscript when layout matters.

Project evidence rules override generic venue defaults. Do not impose Nature-specific style, biomedical reporting checklists, or arbitrary sentence-length rules unless the selected journal requires them.

## Verification Before Delivery

1. Re-read every edited paragraph against its source artifacts.
2. Search edited files for unresolved `TODO`, placeholder citations, stale figure references, and inconsistent terminology.
3. Compare LaTeX `\ref{fig:...}` keys with `\label{fig:...}` keys directly. The installed submission-audit helper does not currently recognize this project's LaTeX figure-reference form.
4. Compile `paper/main.tex` with the repository's documented LaTeX workflow when tooling is available.
5. Check the LaTeX log for undefined citations, undefined references, missing figures, and fatal errors.
6. Render and inspect the PDF when edits affect figures, tables, equations, floats, or pagination.
7. Report edited files, evidence used, checks run, unsupported claims remaining, bilingual-sync status, and the safest next step.
