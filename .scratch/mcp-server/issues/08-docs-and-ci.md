# 08: Docs, dogfooding and CI

**What to build:** A user finds in the README how to install the server, register it with Claude Code or any MCP client, pass data and use masking. A contributor gets the server in Claude Code from the repo's `.mcp.json`. CI runs the MCP tests on the latest and minimum versions and the stdio smoke test on Python 3.12 and 3.14.

Read these sections of `.scratch/mcp-server/spec.md`: Docs and records, CI. Covers user stories 5, 29 and 31.

**Blocked by:** 06-table-path, 07-discovery

**Status:** done

- [x] The README has an "Use over MCP" section: install, `claude mcp add agent-eda -- agent-eda-mcp` with and without `--project`, a generic `mcpServers` JSON snippet, the data path rules, `table_path`, and the masking caveat. The Development table and the CLAUDE.md commands cover running the MCP tests.
- [x] The repo's `.mcp.json` runs `uv run --extra mcp agent-eda-mcp`, and a manual session in Claude Code lists and calls the analyses (recorded in the comments).
- [x] `.github/workflows/ci.yml` installs the extra in the test jobs, keeps the minimum-versions job without the extra (pydantic 2.7, MCP tests skipped), adds a second minimum-versions job with the extra at its floors (mcp 2.3, pyarrow 14.0.1, pydantic 2.12), runs the smoke test on 3.12 and 3.14, and leaves the offline no-extras packaging job as it is.
- [x] Both minimum-versions commands in the README work locally.
- [x] GLOSSARY.md and ADR 0006 match what shipped; the glossary terms are used in the code, docstrings and messages.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

What this ticket settled that the spec left open:

- **README.** "Use over MCP" sits after "Projects and masking", with four parts: install and register (`pip`/`uv add` with the extra, the built wheel with `[mcp]` until 0.1 is on PyPI, `claude mcp add` with and without `--project`, a generic `mcpServers` snippet), passing data by path, reading the response (text, structured content, tool errors, `table_path`), and masking over MCP. It says that `table_path` never overwrites (`table_file_exists` suggests `<stem>-2<ext>`), that list and mapping cells are JSON text and mixed-type Parquet columns text (ticket 06), and that a URL is read as a local relative path (ticket 03). It repeats the guardrail sentence. The intro, the features list, the installation note and the CI paragraph mention the MCP server. `tests/test_readme.py` now checks that the README's JSON block parses and starts `agent-eda-mcp`.
- **Development table and CLAUDE.md.** New rows: "Test the MCP server alone" (`uv run --extra mcp pytest tests/test_mcp_*.py`) and "Smoke test the MCP server over stdio". The notes cover the smoke script and `.mcp.json`. CLAUDE.md (AGENTS.md) lists the smoke command, says `uv run` installs the extra with uv 0.13+ and that an older uv needs `--extra mcp`, and names both minimum-versions runs.
- **The smoke test** is a new script, `tests/smoke_mcp.py`, next to `tests/smoke.py`. It starts `agent-eda-mcp` over real stdio, lists the MCP tools and calls `describe_dataset`, and exits 1 unless the text and the record say `completed`. A script, rather than the stdio tests in `tests/test_mcp_server.py`, because the MCP test modules skip without the extra, so a CI job whose install dropped the extra would pass green. The script fails on the `mcp` import instead.
- **CI.** `tests-latest` runs `uv run --locked --extra mcp pytest`: the extra is explicit, though uv 0.13.1 installs it by default. The new job `tests-minimum-mcp` runs the README's "with the mcp extra" command. The new job `mcp-smoke` runs `uv run --locked --extra mcp python -I tests/smoke_mcp.py` on Python 3.12 and 3.14 (x86_64). `tests-minimum`, `packaging` and the benchmark are unchanged. `tests/test_ci_workflow.py` pins all five points: both minimum-versions jobs run the README's commands, the latest jobs pass `--extra mcp`, the smoke job's Pythons are 3.12 and 3.14, and no packaging step mentions `mcp`.
- **Glossary.** GLOSSARY.md and ADR 0006 already matched what shipped and are unchanged. The text a caller reads said "server" alone in four places, which the glossary asks to avoid: the `project_path`, `table_path` and `mask_sensitive` descriptions in the input schemas, and the `--project` help. They now say "MCP server", and so do the public `agent_eda.mcp` docstrings. New tests check that no MCP tool description, property description or the instructions use "server" without "MCP".

**Manual session in Claude Code (2.1.302), run headless from the worktree:**

```sh
claude -p "Using only the agent-eda MCP server: 1) list the names of all MCP tools it offers, 2) call its describe_dataset MCP tool with data_path <scratch>/shop.csv, 3) call detect_missing_sentinels on the same file. Reply with the tool names, then the exact text each call returned." --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__agent-eda"
```

`shop.csv` held `city,score,note` with an `N/A`, an empty field and a `-999`. Claude Code started the MCP server through `.mcp.json` (`uv run --extra mcp agent-eda-mcp`), listed all 13 MCP tools (the 11 analyses, `search_analyses` and `list_analyses`) and called both. `describe_dataset` completed on 4 rows x 3 columns. `detect_missing_sentinels` found `-999` in `score` and `N/A` in `note` as suspected sentinel values.

**Open point for the orchestrator (settled in ADR 0006: the server keeps both parts, text and record, and its instructions don't promise that the caller sees the text):** the model in that session saw only the structured content (the JSON record), not the text content (the printed result). The MCP server sends both: `tests/smoke_mcp.py` and the stdio tests read the text from the same response. So Claude Code 2.1.302 shows the model `structuredContent` alone when a result has both. Over Claude Code, an agent then gets the full record but not the at-most-40-line text, which goes against what the instructions and user story 19 promise. Possible responses: leave it, since the record is complete; say in the instructions that some clients show only the record; or drop the output schema and structured content for Claude Code, at the cost of user story 20. Not changed here.

Checks: `ruff check`, `ruff format`, `ty check` and `pytest` pass on the latest versions (with `--extra mcp`, local uv 0.10.11; 2083 passed), and in both minimum-versions runs (core: 1840 passed, MCP modules skipped; with the extra: 2083 passed). `tests/smoke_mcp.py` passes on Python 3.12 and 3.14. CI itself wasn't run: `actionlint` 1.7.12 (via `uvx --from actionlint-py`) reports no problems in `ci.yml`, and `tests/test_ci_workflow.py` passes.
