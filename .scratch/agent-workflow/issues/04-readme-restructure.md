# 04: README restructure

What to build: the README is the user guide, short enough for one Read and correct on PyPI. The backlog moves to `docs/planned-analyses.md`, the development notes move to `CONTRIBUTING.md`, every analysis has a table row, and the claims that disagree with the code are fixed.

Read these sections of `.scratch/agent-workflow/spec.md`: README, and test group 5. This ticket covers user stories 19, 20 and 22 to 24. Findings: DOC-3, DOC-7, DOC-18, SPEC-8, SPEC-16, SPEC-17, SPEC-20, REL-15, CORE-22, AN1-12, REL-4 (the links).

Blocked by: 02

Status: done

- [x] The planned analyses live in `docs/planned-analyses.md`. The README links to it. AGENTS.md's Sources of truth and the new-analysis checklist name it. `tests/test_readme.py`'s checks on the backlog (each name once, no shipped analysis) read the new file.
- [x] The backlog rows that conflict with the rules are rewritten:
  - `load_dataset` reads no remote source;
  - `run_analysis` and `run_standard_eda` take no run context (ADR 0004);
  - the artifact references are gone;
  - "config" and "split" are reworded per the glossary;
  - `test_functional_dependencies` gets a name without `test_`.
- [x] `CONTRIBUTING.md` holds the development notes. Its commands come by pointer to AGENTS.md, and the README links to it.
- [x] The Analyses section has a first-look table in `ae.guide()`'s order and an "Other analyses" table by module. The second table says it is empty today.
- [x] A test fails unless every registered analysis appears in exactly one table. AGENTS.md and the checklist say which table a new analysis goes in.
- [x] These claims match the code:
  - the signatures of `Project`, `DataPolicy` and `ColumnSpec`, with their keyword-only star;
  - the claim that `infer_column_semantics` flags leading-zero codes;
  - the example of a duplicate group's row IDs written as JSON text in a table cell;
  - "placeholder codes", the unqualified "tools", the ARM64 sentence and "every analysis that works column by column takes columns=";
  - the `mask_sensitive` row of `detect_missing_sentinels`;
  - "down to the byte", which becomes "in the same environment, BLAS thread count included".
- [x] `tests/test_readme.py` checks the class signatures too.
- [x] Every link to a repository file is an absolute `https://github.com/chrislemke/agent-eda/blob/main/...` URL, and a test fails on a relative link. Anchors within the README stay relative.
- [x] An agent's Read shows the whole README. The Comments record its size before and after.
- [x] The finish checks pass.

## Comments

- The size before and after, by `wc`: 1,039 lines and 97,999 characters (35.9K tokens by the audit) became 698 lines and 51,976 characters. An agent's Read of the whole file now shows all 698 lines; before, it stopped at line 631. `tests/test_readme.py` caps the README at 60,000 characters: Read's cap is about 25K tokens, which is about 68K characters at the audit's 2.7 characters per token, and 60K keeps a margin. The Reference is still hand-written; ticket 05 generates it and must not let it grow.
- The new files are `docs/planned-analyses.md` (345 lines, the same 268 rows, one per planned function) and `CONTRIBUTING.md`, which holds the development notes, with the CI sentence fixed to say ARM64 runs Python 3.12 only (REL-15). A test reads that from `ci.yml`'s matrix.
- Backlog renames settled here, since the ticket asked for "reworded per the glossary" without naming them:
  - `validate_project_config` (module `config`) became `validate_project` (module `project`);
  - `test_functional_dependencies` became `check_functional_dependencies`;
  - the three rows that said "split" became `assess_spatial_role_dependence`, `measure_role_overlap` and `audit_temporal_role_boundaries`, and say "dataset roles";
  - `verify_artifact_versions` became `verify_input_versions`. The ticket only asked for the artifact references to go; this row compared supplied artifacts, not an artifact store, but the test bans the word in the whole backlog so that no reference can come back. "configuration" and "configured" in other rows became "settings" and "declared", for the same reason.
  - `load_dataset` takes a local file (CSV, TSV or Parquet) and says it opens no network connection; `export_eda_json` writes a local file and returns its path and checksum; `run_analysis` and `run_standard_eda` call analyses with explicit keyword arguments and no run context, and the backlog's intro cites ADR 0004 and ADR 0006.
  - The other `test_*` names (`test_normality` and so on) stay: they run hypothesis tests, which AGENTS.md's Names rule allows.
- "Every analysis takes `columns=`" is true for 8 of 11. The README says "except `describe_dataset`, `profile_target` and `validate_column_types`", and a test compares that list with the analyses that lack the parameter.
- The README's example of a list cell in a `table_path` file is now the columns of a candidate key from `find_candidate_keys`; a test checks that its table has list cells. The leading-zero claim now says such numbers count as text, which the test checks against `infer_column_semantics`'s behaviour.
- The README keeps the Reference and its `#### name` headings unchanged except for the corrected claims and class signatures. A new first-look "Run all 11" paragraph stays in the Analyses section; the "Other analyses" table has its header and the line "No analysis outside the first look ships yet." The test requires that line only while every analysis is in the first-look table.
- `tests/test_commands.py` read the README's Development section; it now reads `CONTRIBUTING.md`, and the uv-caveat and review-skill-prefix checks cover `CONTRIBUTING.md` too.
- AGENTS.md: Sources of truth, the ticket paragraph and the Layout bullet now name `docs/planned-analyses.md` and the two tables. Not done here: `docs/agents/new-analysis.md`, which ticket 03 creates, does not exist in this branch, so "the new-analysis checklist names it, and says which table a new analysis goes in" is part of ticket 03's checklist. After merging, check that it says "move it from `docs/planned-analyses.md` to the README's Analyses section: the first-look table for a step of `ae.guide()`'s first look, otherwise 'Other analyses' under its module", and that its guard tests read `docs/planned-analyses.md`, not the README, for the backlog.
- Mutants, each failing the test named in brackets and then reverted: a first-look row removed [each analysis has one row; first-look table follows the guide]; an analysis listed in both tables [each analysis has one row]; a relative link added [no link is relative]; a link to a file that doesn't exist [no link is relative, each repository link leads to a file]; the old `Project` signature restored [project classes].
- Finish checks run and passed: ruff check, ruff format --check, ty check, `uv run pytest` (2202 passed, 2 skipped), both minimum-versions runs (1896 passed, 10 skipped without the extra; 2202 passed with it) and the MCP smoke test. The benchmark was not run: no analysis changed.
