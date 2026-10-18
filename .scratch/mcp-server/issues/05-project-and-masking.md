# 05: Project and masking

**What to build:** An MCP caller passes `project_path` to the analyses that support masking, or the user starts the server with `--project` so that every call uses the Project. Masked results over MCP equal masked results in Python, and the docs say plainly that masking is a guardrail.

Read these sections of `.scratch/mcp-server/spec.md`: Project and masking, and test group 5. Read ADR 0001. Covers user stories 26 to 29.

**Blocked by:** 04-errors-and-statuses

**Status:** ready-for-agent

- [ ] MCP tools of analyses that take `project` have `project_path` (string or null) in place of `project` and keep `mask_sensitive`; the other MCP tools have neither.
- [ ] `project_path` loads with `Project.load`; a missing or invalid file returns its `ProjectFileError` code and message as a tool error.
- [ ] `agent-eda-mcp --project project.toml` applies the Project to every call of a tool that takes one; an invalid file stops the server with its message on stderr and exit status 1; `--help` documents the option.
- [ ] Precedence holds: per-call `project_path` over `--project` over none, and a per-call `mask_sensitive` over both.
- [ ] With masking on, no raw value of a sensitive column appears in the text or the structured content of any masking analysis's response; the results equal the Python calls with the same Project.
- [ ] The descriptions of `project_path` and `mask_sensitive` say masking is a guardrail against accidental exposure, not anonymization or access control.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments
