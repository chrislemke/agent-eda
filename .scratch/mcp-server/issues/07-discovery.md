# 07: Discovery over MCP

**What to build:** An MCP caller connecting for the first time reads the server's instructions and knows what the package does, how to pass data, how to read results and which first-look order to follow. `search_analyses` finds an MCP tool from the caller's own wording, and `list_analyses` lists them all.

Read these sections of `.scratch/mcp-server/spec.md`: Discovery, and test group 7. Covers user stories 6 and 10.

**Blocked by:** 05-project-and-masking

**Status:** ready-for-agent

- [ ] The server's `instructions` cover what the package does, data paths (formats, the empty-field rule, relative paths, row IDs), results and statuses, masking with `project_path` and `--project`, and the first-look order from `discovery.FIRST_LOOK` limited to registered analyses.
- [ ] The instructions name every registered analysis, contain no Python call syntax, and keep within 60 lines of 120 characters.
- [ ] The `search_analyses` MCP tool takes `query` and `limit` (default 5) and returns the `SearchListing`'s text and record, equal to the Python call; the spec's search acceptance table holds over MCP.
- [ ] The `list_analyses` MCP tool returns the `AnalysisListing`'s text and record, equal to the Python call.
- [ ] There is no `describe_analysis` or `guide` MCP tool.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments
