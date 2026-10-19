# 04: README restructure

What to build: the README is the user guide, short enough for one Read and correct on PyPI. The backlog moves to `docs/planned-analyses.md`, the development notes move to `CONTRIBUTING.md`, every analysis has a table row, and the claims that disagree with the code are fixed.

Read these sections of `.scratch/agent-workflow/spec.md`: README, and test group 5. This ticket covers user stories 19, 20 and 22 to 24. Findings: DOC-3, DOC-7, DOC-18, SPEC-8, SPEC-16, SPEC-17, SPEC-20, REL-15, CORE-22, AN1-12, REL-4 (the links).

Blocked by: 02

Status: ready-for-agent

- [ ] The planned analyses live in `docs/planned-analyses.md`. The README links to it. AGENTS.md's Sources of truth and the new-analysis checklist name it. `tests/test_readme.py`'s checks on the backlog (each name once, no shipped analysis) read the new file.
- [ ] The backlog rows that conflict with the rules are rewritten:
  - `load_dataset` reads no remote source;
  - `run_analysis` and `run_standard_eda` take no run context (ADR 0004);
  - the artifact references are gone;
  - "config" and "split" are reworded per the glossary;
  - `test_functional_dependencies` gets a name without `test_`.
- [ ] `CONTRIBUTING.md` holds the development notes. Its commands come by pointer to AGENTS.md, and the README links to it.
- [ ] The Analyses section has a first-look table in `ae.guide()`'s order and an "Other analyses" table by module. The second table says it is empty today.
- [ ] A test fails unless every registered analysis appears in exactly one table. AGENTS.md and the checklist say which table a new analysis goes in.
- [ ] These claims match the code:
  - the signatures of `Project`, `DataPolicy` and `ColumnSpec`, with their keyword-only star;
  - the claim that `infer_column_semantics` flags leading-zero codes;
  - the example of a duplicate group's row IDs written as JSON text in a table cell;
  - "placeholder codes", the unqualified "tools", the ARM64 sentence and "every analysis that works column by column takes columns=";
  - the `mask_sensitive` row of `detect_missing_sentinels`;
  - "down to the byte", which becomes "in the same environment, BLAS thread count included".
- [ ] `tests/test_readme.py` checks the class signatures too.
- [ ] Every link to a repository file is an absolute `https://github.com/chrislemke/agent-eda/blob/main/...` URL, and a test fails on a relative link. Anchors within the README stay relative.
- [ ] An agent's Read shows the whole README. The Comments record its size before and after.
- [ ] The finish checks pass.

## Comments
