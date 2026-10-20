# 05: Generated Reference

What to build: the README's Reference section is generated from the registry, the signatures, the docstrings and `ae.errors`, and a test fails when it is stale, so the docstring stays the only hand-written copy.

Read these sections of `.scratch/agent-workflow/spec.md`: Generated Reference, and test group 5. This ticket covers user story 21. Findings: DOC-6, SPEC-8.

Blocked by: 04

Status: done

- [x] `scripts/generate_reference.py` writes the Reference between `<!-- reference:start -->` and `<!-- reference:end -->`. Running it twice in a row changes nothing.
- [x] For each analysis, the generated Reference gives:
  - the signature, with its keyword-only star and defaults;
  - the summary;
  - each argument's type, default and meaning;
  - the fields to read;
  - each finding code with its assessment;
  - the statuses.
- [x] It also covers the discovery functions; Project, DataPolicy, ColumnSpec, LogicalType and AnalysisResult; the error codes; and the MCP server command.
- [x] A test regenerates the section and fails on any difference, naming the command to run.
- [x] The hand-written checks in `tests/test_readme.py` that the generation makes redundant are gone.
- [x] AGENTS.md says the docstring is the only hand-written copy and the README's Reference is generated from it. The sentence about generated agent docs coming in a later slice is gone.
- [x] The Comments record the Reference's size before and after; it doesn't grow.
- [x] The finish checks pass.

## Comments

- The Reference's size before and after, measured from `## Reference` to `## Documentation` in README.md: 30,872 characters (378 lines) before, 30,730 (398 lines) after, so it didn't grow. The whole README went from 51,976 to 51,833 characters.
- The generator shortens text to fit that budget: an argument's meaning, a field's description and a finding's claim are the first sentence of the docstring's text, an error's "Raised when" cell is its first clause, and a result's fields show one level of nesting. The full text stays in the docstring, `help()` and `ae.describe_analysis()`.
- The assessment of a finding code was hand-written in the old Reference and nothing structured declared it, so it comes from the docstring now: each `Returns:` bullet of a finding code starts with its assessment and a semicolon, as in `- \`constant_columns\`: suspected; a column holds one value`. About 33 bullets of seven analyses didn't, and now do (their wording otherwise unchanged; a trailing "Flag x." sentence replaced the leading "flag x" in `infer_column_semantics`). `tests/test_contract.py` enforces the format for every analysis, and the generator stops with the fix when a bullet lacks it. Nothing checks that the stated assessment matches what the analysis returns; that stays a review point. AGENTS.md's docstring list and the checklist (step 6) say so.
- The generated section keeps the `## Reference` heading outside the markers and holds only the generated text, with no hand-written prose beyond one line on how it is made, the lead-in to the analyses, and `print(res)` under `AnalysisResult`. The table of contents and the hand-written list of arguments shared by every analysis are gone: each analysis shows its own arguments.
- `uv run python scripts/generate_reference.py` rewrites the README; `--check` compares without writing and exits 1 with the command in its message. `--readme PATH` names another file, which the tests use. The tests load the script from its path with `importlib`, since it isn't shipped and isn't part of the package.
- Tests removed as redundant: the checks of every public name, each signature, each analysis's arguments, findings and statuses, the Project classes' signatures, and the errors table against the README text. Their content moved into tests of the generator's output, written against `ae.describe_analysis`, `inspect` and `ae.errors`. In the sentinel and leading-zeros tests, the assertions on hand-written Reference prose ("count as text", the `mask_sensitive` row) are gone. `tests/test_describe_analysis.py` expects the new bullet text. `CREATED_LATER` is gone from `tests/test_new_analysis_docs.py`.
- Mutants: removing "suspected; " from the `infinite_values` bullet fails `test_returns_gives_each_finding_code_its_assessment_before_a_semicolon[summarize_numeric]`, and changing a word in a docstring without regenerating fails `test_the_reference_is_the_one_the_generator_writes_from_the_docstrings` with the command to run.
- Ticket 06 changes docstrings and field descriptions, which changes the Reference: rerun the generator after merging it. Its text may also push the Reference past its old size; the size above is the number to compare.
- Finish checks run: ruff check and format, ty, `uv run pytest` (2,240 passed), both minimum-versions runs (1,934 and 2,240 passed, `uv.lock` unchanged), the MCP smoke test. The benchmark wasn't run: no analysis code changed, only docstrings.
- Review fix: the generator command and its `--check` form now sit only in AGENTS.md's Commands block; CONTRIBUTING.md, `docs/agents/new-analysis.md` and AGENTS.md's Docstrings section point there instead of repeating it.
