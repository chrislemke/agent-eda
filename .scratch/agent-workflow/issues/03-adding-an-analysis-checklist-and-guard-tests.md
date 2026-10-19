# 03: Adding an analysis: checklist and guard tests

What to build: one checklist covers every place that adding or renaming an analysis touches, tests catch the steps that fail silently today, and AGENTS.md states the rules for what MCP callers read.

Read these sections of `.scratch/agent-workflow/spec.md`: Adding an analysis, and test groups 3 and 4. This ticket covers user stories 13 to 18. Findings: DOC-5, DOC-10, DOC-12, DOC-17, DES-19.

Blocked by: 02

Status: ready-for-agent

- [ ] `docs/agents/new-analysis.md` lists every step of adding or renaming an analysis, and for each step what catches a miss. It starts from DOC-5's twelve steps, updated for this spec's README layout. AGENTS.md points to it in one line: "Before adding or renaming an analysis, follow docs/agents/new-analysis.md".
- [ ] A test fails when a public function defined in a module under `agent_eda.data` or `agent_eda.statistics` isn't a registered analysis. Its message names the fix. Check it once with an exported `count_rows` that has no `@analysis`.
- [ ] A test fails when `__all__`, `_EXPORTS` and the `TYPE_CHECKING` imports in `agent_eda/__init__.py` name different sets.
- [ ] A test fails when an `AgentEDAError` subclass's `code` is missing from the `ae.errors` docstring or the README's errors table, and when either lists a code no class has.
- [ ] AGENTS.md has an "MCP surface" rule:
  - the docstring sections become the MCP tool description;
  - a message an MCP tool can reach gives a fix an MCP caller can make;
  - every parameter type needs a JSON Schema, and a DataFrame passed by role needs a `<role>_path`;
  - nothing an MCP caller reads holds Python.
- [ ] AGENTS.md's ADR trigger list names ADR 0006: the MCP server, data reading and anything an MCP caller reads.
- [ ] AGENTS.md's Tests section points to `tests/recipes.py` and `tests/strategies.py`, saying what each gives a new analysis. It also says the repository's MCP server serves the code as it was when the Claude Code session started.
- [ ] The finish checks pass.

## Comments
