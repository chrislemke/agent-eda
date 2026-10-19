# 02: Commands, uv and finish checks

What to build: uv refuses to run when it is too old, AGENTS.md holds the only copy of the commands and an exact list of finish checks, and the hooks, coverage and plugins do what the documents say.

Read these sections of `.scratch/agent-workflow/spec.md`: Commands and checks, and test group 2. This ticket covers user stories 6 to 12. Findings: DOC-2, DOC-8, DOC-11 (the pytest hook), DOC-14, DOC-15, TEST-8, TEST-16.

Blocked by: 01

Status: ready-for-agent

- [ ] `pyproject.toml` has `[tool.uv] required-version = ">=0.13"`. The Comments record the message uv 0.10 prints when it refuses to run.
- [ ] AGENTS.md's Commands section lists each command once. Its "Finish checks" list gives the exact commands:
  - `ruff check`, `ruff format --check` and `ty check`;
  - `pytest`;
  - both minimum-versions runs, with the note that `--isolated` keeps `uv.lock` unchanged;
  - the MCP smoke test;
  - the benchmark, with when to run it.
- [ ] The uv 0.13 caveat sentences are gone from AGENTS.md and the README, since pyproject now states the requirement.
- [ ] `tests/test_ci_workflow.py` reads the minimum-versions commands from AGENTS.md. The README's development notes point to AGENTS.md instead of repeating the commands.
- [ ] The skip reason of the MCP test modules names `uv sync` or `uv run --extra mcp pytest` for this repository, and `pip install 'agent-eda[mcp]'` for an installed package.
- [ ] The pre-push pytest hook runs on every push, whatever the changed files, and fails when pytest collects no tests.
- [ ] `[tool.coverage.run]` sets `patch = ["subprocess"]`, and `uv run pytest --cov` reports `agent_eda/mcp.py`'s `main` as covered.
- [ ] `.claude/settings.json` no longer enables pyright-lsp. Every document that asks for a review names the skill with its plugin prefix.
- [ ] AGENTS.md's performance target says every analysis runs in under 5 seconds on 1,000,000 rows × 100 columns, as `tests/test_benchmark.py` checks.
- [ ] The finish checks pass, as AGENTS.md now lists them.

## Comments
