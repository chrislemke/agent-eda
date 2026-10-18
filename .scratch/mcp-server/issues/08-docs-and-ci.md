# 08: Docs, dogfooding and CI

**What to build:** A user finds in the README how to install the server, register it with Claude Code or any MCP client, pass data and use masking. A contributor gets the server in Claude Code from the repo's `.mcp.json`. CI runs the MCP tests on the latest and minimum versions and the stdio smoke test on Python 3.12 and 3.14.

Read these sections of `.scratch/mcp-server/spec.md`: Docs and records, CI. Covers user stories 5, 29 and 31.

**Blocked by:** 06-table-path, 07-discovery

**Status:** ready-for-agent

- [ ] The README has an "Use over MCP" section: install, `claude mcp add agent-eda -- agent-eda-mcp` with and without `--project`, a generic `mcpServers` JSON snippet, the data path rules, `table_path`, and the masking caveat. The Development table and the CLAUDE.md commands cover running the MCP tests.
- [ ] The repo's `.mcp.json` runs `uv run --extra mcp agent-eda-mcp`, and a manual session in Claude Code lists and calls the analyses (recorded in the comments).
- [ ] `.github/workflows/ci.yml` installs the extra in the test jobs, keeps the minimum-versions job without the extra (pydantic 2.7, MCP tests skipped), adds a second minimum-versions job with the extra at its floors (mcp 2.3, pyarrow 14, pydantic 2.12), runs the smoke test on 3.12 and 3.14, and leaves the offline no-extras packaging job as it is.
- [ ] Both minimum-versions commands in the README work locally.
- [ ] GLOSSARY.md and ADR 0006 match what shipped; the glossary terms are used in the code, docstrings and messages.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments
