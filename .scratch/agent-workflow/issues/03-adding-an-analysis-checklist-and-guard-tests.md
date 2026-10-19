# 03: Adding an analysis: checklist and guard tests

What to build: one checklist covers every place that adding or renaming an analysis touches, tests catch the steps that fail silently today, and AGENTS.md states the rules for what MCP callers read.

Read these sections of `.scratch/agent-workflow/spec.md`: Adding an analysis, and test groups 3 and 4. This ticket covers user stories 13 to 18. Findings: DOC-5, DOC-10, DOC-12, DOC-17, DES-19.

Blocked by: 02

Status: done

- [x] `docs/agents/new-analysis.md` lists every step of adding or renaming an analysis, and for each step what catches a miss. It starts from DOC-5's twelve steps, updated for this spec's README layout. AGENTS.md points to it in one line: "Before adding or renaming an analysis, follow docs/agents/new-analysis.md".
- [x] A test fails when a public function defined in a module under `agent_eda.data` or `agent_eda.statistics` isn't a registered analysis. Its message names the fix. Check it once with an exported `count_rows` that has no `@analysis`.
- [x] A test fails when `__all__`, `_EXPORTS` and the `TYPE_CHECKING` imports in `agent_eda/__init__.py` name different sets.
- [x] A test fails when an `AgentEDAError` subclass's `code` is missing from the `ae.errors` docstring or the README's errors table, and when either lists a code no class has.
- [x] AGENTS.md has an "MCP surface" rule:
  - the docstring sections become the MCP tool description;
  - a message an MCP tool can reach gives a fix an MCP caller can make;
  - every parameter type needs a JSON Schema, and a DataFrame passed by role needs a `<role>_path`;
  - nothing an MCP caller reads holds Python.
- [x] AGENTS.md's ADR trigger list names ADR 0006: the MCP server, data reading and anything an MCP caller reads.
- [x] AGENTS.md's Tests section points to `tests/recipes.py` and `tests/strategies.py`, saying what each gives a new analysis. It also says the repository's MCP server serves the code as it was when the Claude Code session started.
- [x] The finish checks pass.

## Comments

- Settled: the checklist has 15 steps, DOC-5's twelve plus the benchmark, the glossary and the finish checks, and a short section on renaming. Each row names what catches a miss and says "nothing" where only the implementer does.
- Settled: the registration guard also fails for a public function of an `agent_eda.data` or `agent_eda.statistics` module that is registered but not exported from `agent_eda` (DES-19: a module missing from `_EXPORTS` silently drops its analyses). Both guards skip functions a module only imports and names with a leading underscore.
- Settled: the `__init__.py` guard reads the file with `ast`. The `TYPE_CHECKING` imports minus `_SUBMODULES` (`errors`, `results`), `__all__` and the keys of `_EXPORTS` must be one set; the message names the lists a name is missing from.
- Settled: "a code no class has" counts a code as in use when a class has it as its `code`, or a raise site passes `code=` to an `AgentEDAError` subclass directly or through `_wording.worded`. Four documented codes (`invalid_call`, `unsupported_data_type`, `ambiguous_column_labels`, `unknown_project_key`) exist only that way. Finding and limitation codes are not error codes. The README side reads rows of the form ``| `code` | `ClassError` |`` anywhere in README.md, so it survives the README restructure of ticket 04 as long as the generated errors table keeps that row shape.
- Written against the final README layout (ticket 04, 05): the checklist names `docs/planned-analyses.md`, the "Other analyses" table and `scripts/generate_reference.py`, which don't exist yet. `tests/test_new_analysis_docs.py` skips those two paths when it checks that the files the checklist names exist; remove `CREATED_LATER` once ticket 05 lands. The README sentence I changed (Development, "New analyses") uses the current relative link; ticket 04's link rule will make it absolute.
- Mutant checks, each reverted afterwards:
  - an exported `count_rows` in `agent_eda/data/structure.py` without `@analysis`, in all three export lists: `test_every_public_function_of_an_analysis_module_is_a_registered_analysis` failed with "agent_eda.data.structure.count_rows is not a registered analysis: add @analysis(...) ...";
  - an error class whose `code` changed to `unknown_analysis_renamed`, plus a docstring line for a code nothing raises: both `tests/test_error_codes.py` guards failed, naming the missing code and the stale entry;
  - the export-list guard is checked in the suite against `__init__.py` with one name dropped from each of the three lists (`test_an_export_missing_from_one_list_is_reported_with_the_list_to_fix`).
- Finish checks run locally and passing: ruff check, ruff format --check, ty check; `uv run pytest` (2207 passed, 2 skipped); both minimum-versions runs (1901 passed and 10 MCP tests skipped without the extra; the run with the extra's floors passed); the MCP smoke test. The benchmark wasn't run: no analysis or code an analysis calls changed.
