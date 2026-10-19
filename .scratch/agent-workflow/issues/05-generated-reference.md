# 05: Generated Reference

What to build: the README's Reference section is generated from the registry, the signatures, the docstrings and `ae.errors`, and a test fails when it is stale, so the docstring stays the only hand-written copy.

Read these sections of `.scratch/agent-workflow/spec.md`: Generated Reference, and test group 5. This ticket covers user story 21. Findings: DOC-6, SPEC-8.

Blocked by: 04

Status: ready-for-agent

- [ ] `scripts/generate_reference.py` writes the Reference between `<!-- reference:start -->` and `<!-- reference:end -->`. Running it twice in a row changes nothing.
- [ ] For each analysis, the generated Reference gives:
  - the signature, with its keyword-only star and defaults;
  - the summary;
  - each argument's type, default and meaning;
  - the fields to read;
  - each finding code with its assessment;
  - the statuses.
- [ ] It also covers the discovery functions; Project, DataPolicy, ColumnSpec, LogicalType and AnalysisResult; the error codes; and the MCP server command.
- [ ] A test regenerates the section and fails on any difference, naming the command to run.
- [ ] The hand-written checks in `tests/test_readme.py` that the generation makes redundant are gone.
- [ ] AGENTS.md says the docstring is the only hand-written copy and the README's Reference is generated from it. The sentence about generated agent docs coming in a later slice is gone.
- [ ] The Comments record the Reference's size before and after; it doesn't grow.
- [ ] The finish checks pass.

## Comments
