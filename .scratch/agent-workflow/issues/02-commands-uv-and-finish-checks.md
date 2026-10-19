# 02: Commands, uv and finish checks

What to build: uv refuses to run when it is too old, AGENTS.md holds the only copy of the commands and an exact list of finish checks, and the hooks, coverage and plugins do what the documents say.

Read these sections of `.scratch/agent-workflow/spec.md`: Commands and checks, and test group 2. This ticket covers user stories 6 to 12. Findings: DOC-2, DOC-8, DOC-11 (the pytest hook), DOC-14, DOC-15, TEST-8, TEST-16.

Blocked by: 01

Status: done

- [x] `pyproject.toml` has `[tool.uv] required-version = ">=0.13"`. The Comments record the message uv 0.10 prints when it refuses to run.
- [x] AGENTS.md's Commands section lists each command once. Its "Finish checks" list gives the exact commands:
  - `ruff check`, `ruff format --check` and `ty check`;
  - `pytest`;
  - both minimum-versions runs, with the note that `--isolated` keeps `uv.lock` unchanged;
  - the MCP smoke test;
  - the benchmark, with when to run it.
- [x] The uv 0.13 caveat sentences are gone from AGENTS.md and the README, since pyproject now states the requirement.
- [x] `tests/test_ci_workflow.py` reads the minimum-versions commands from AGENTS.md. The README's development notes point to AGENTS.md instead of repeating the commands.
- [x] The skip reason of the MCP test modules names `uv sync` or `uv run --extra mcp pytest` for this repository, and `pip install 'agent-eda[mcp]'` for an installed package.
- [x] The pre-push pytest hook runs on every push, whatever the changed files, and fails when pytest collects no tests.
- [x] `[tool.coverage.run]` sets `patch = ["subprocess"]`, and `uv run pytest --cov` reports `agent_eda/mcp.py`'s `main` as covered.
- [x] `.claude/settings.json` no longer enables pyright-lsp. Every document that asks for a review names the skill with its plugin prefix.
- [x] AGENTS.md's performance target says every analysis runs in under 5 seconds on 1,000,000 rows × 100 columns, as `tests/test_benchmark.py` checks.
- [x] The finish checks pass, as AGENTS.md now lists them.

## Comments

- uv 0.10.11 (`uvx --from uv==0.10.11 uv lock --check` in this repository) prints `error: Required uv version `>=0.13` does not match the running version `0.10.11`. Update `uv` by running `uv self update`.` and stops.
- AGENTS.md's Commands section keeps the setup and everyday commands in one block; "Finish checks" (a `###` heading inside Commands) holds the exact check commands. Each command appears once, which `tests/test_commands.py` checks. Commands that only the README's Development table had (hooks, coverage, the thorough profile, the MCP tests alone, the wheel build, all hooks) moved into AGENTS.md's block. The README's Development section now points to AGENTS.md and keeps its notes that are not commands. `CONTRIBUTING.md` does not exist yet, so the pointer is in the README; the README spec ticket moves it.
- `tests/test_ci_workflow.py` takes the two minimum-versions commands from the backticked spans of the Finish checks that contain `--resolution lowest-direct`, told apart by `--extra mcp`.
- `[tool.coverage.run] patch = ["subprocess"]` alone did not cover `main`: an MCP client starts the server with only a few environment variables, so the `COVERAGE_PROCESS_CONFIG` variable that coverage sets never reached the process. The tests that start `agent-eda-mcp` over stdio now pass the `COVERAGE_*` variables on through `tests/stdio.py`. `uv run pytest --cov` then reports `agent_eda/mcp.py` missing only lines 165-168 (outside `main`); `main` (lines 129-139) is covered. No `sigterm` setting was needed. With the patch on, `coverage.tomlconfig` appeared among the modules loaded in the subprocess of `test_save_and_load_use_no_toml_library_but_the_standard_one`, so that test ignores modules whose name starts with `coverage`.
- The pre-push hook is now `uv run pytest -q` with `always_run: true`; the old wrapper that turned exit code 5 into success is gone. The hook no longer has `types_or`, which `always_run` makes moot.
- No document in this repository asks for a review, so no name needed a prefix; the bare `code-review` that DOC-14 found sits in the `implement-spec` skill outside the repository. `tests/test_commands.py` fails if AGENTS.md, README.md or a file under `docs/` names `code-review` without a prefix. pyright-lsp is removed from `.claude/settings.json`.
- The MCP skip reason names `uv sync` and `uv run --extra mcp pytest` for this repository and `pip install 'agent-eda[mcp]'` for an installed package. The test runs the two modules with `mcp` blocked in `sys.modules` and reads the skip report.
- Guard tests were red before the change (14 failed against the old files) and green after. The benchmark was not run: no analysis changed.
- Finish checks run: ruff check, ruff format --check, ty check, pytest (2186 passed, 2 skipped), both minimum-versions runs (1880 passed, 10 skipped without the extra; 2186 passed with it), and the MCP smoke test all pass.
