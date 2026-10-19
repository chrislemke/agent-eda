# 07: Discovery over MCP

**What to build:** An MCP caller connecting for the first time reads the server's instructions and knows what the package does, how to pass data, how to read results and which first-look order to follow. `search_analyses` finds an MCP tool from the caller's own wording, and `list_analyses` lists them all.

Read these sections of `.scratch/mcp-server/spec.md`: Discovery, and test group 7. Covers user stories 6 and 10.

**Blocked by:** 05-project-and-masking

**Status:** done

- [x] The server's `instructions` cover what the package does, data paths (formats, the empty-field rule, relative paths, row IDs), results and statuses, masking with `project_path` and `--project`, and the first-look order from `discovery.FIRST_LOOK` limited to registered analyses.
- [x] The instructions name every registered analysis, contain no Python call syntax, and keep within 60 lines of 120 characters.
- [x] The `search_analyses` MCP tool takes `query` and `limit` (default 5) and returns the `SearchListing`'s text and record, equal to the Python call; the spec's search acceptance table holds over MCP.
- [x] The `list_analyses` MCP tool returns the `AnalysisListing`'s text and record, equal to the Python call.
- [x] There is no `describe_analysis` or `guide` MCP tool.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

- **Where the instructions live.** The private `agent_eda._mcp_instructions` (no `mcp` import) builds them from the guide's parts: `str(ae.list_analyses())` and the first-look lines, now one helper `discovery._first_look_lines` that `guide()` and the instructions share, plus MCP text for intro, data paths (extensions from the reader's `supported_extensions()`), the response (text, structured content with capped lists and `<field>_total`, `table_path`, statuses, findings, `isError`), masking (`project_path`, `agent-eda-mcp --project`, `mask_sensitive` precedence, the guardrail sentence) and a pointer to `search_analyses`/`list_analyses`. They come to 58 lines (max 119 characters) with 11 analyses; the guide is at 60, so both need trimming when the next analysis ships.
- **table_path** is described per the spec (a new `.csv`, `.parquet` or `.pq` file, never overwritten), written while ticket 06 ran in parallel.
- **"No Python call syntax"** is the regex the ticket-02 description test uses (a name followed by `(`), plus no `import ` and no `ae.`.
- **Discovery MCP tools.** Listed before the analysis MCP tools. Descriptions are fixed server text after the docstring's summary line (their docstrings' bodies carry Python such as `to_frame()`); `list_analyses` notes that `parameters` gives Python names (`data` is `data_path`, `project` is `project_path`). Input schemas: `search_analyses` has `query` (string, required) and `limit` (integer, minimum 1, default 5); `list_analyses` has none; both `additionalProperties: false`. Output schemas are `SearchListing`/`AnalysisListing.model_json_schema()`, which real records validate against (a made-up field fails). Annotations: read-only, not destructive, idempotent, closed world.
- **Errors.** An unknown or missing argument is `invalid_call` (as for analyses); a bad value, such as `limit: 0`, is the Python call's `InvalidArgumentError` with its message.
- **`create_server(analyses=...)`** limits only the analysis MCP tools; the discovery MCP tools and the instructions always cover the registry, as their Python counterparts do. Two ticket-02 tests now allow for the two discovery MCP tools.
- **Open point for the orchestrator.** The search text is equal to the Python call's, as the ticket asks, so over MCP it still says `More: ae.search_analyses('…', limit=11)`, `Next: print(ae.describe_analysis('…'))` and, with no match, `call ae.guide()`. An MCP caller can't make those calls. A later ticket could give the listing an MCP wording (for example a `for_mcp` text variant) if that matters more than equality.

**Code review follow-up (2026-10-18).**
- The user decided on 2026-10-18 to override "text equal to the Python call". Over MCP, printed results and the `search_analyses` text use MCP wording, and records stay equal to Python's. One rendering path takes a surface switch: `AnalysisResult._render(surface)` and `_DiscoveryRecord._render(surface)` with `"python"` or `"mcp"`, and `str()` stays Python. Over MCP the footer names the full table that `table_path` writes and the structured content. Tables only Python can ask for by name are left out, record fields lose `res.`, and masking settings read `mask_sensitive: true` and `data_policy.mask_sensitive = false`. The search text points to `search_analyses with limit N`, the MCP tool's description and input schema, and `list_analyses`. Error messages and the record's own text, such as status reasons, still carry Python fixes. Changing them would break the equal records.
- The instructions no longer promise that the caller sees the text. They say an MCP client may show the text, the structured content or both, and that the record's `table_file` gives the written table's path.
- Line limits, a follow-up: the instructions are at 59 of 60 lines and the guide at 60 of 60. The next registered analysis adds a line to each listing, so both need trimming then, for example by dropping the module headers from the listing or by shortening the response section. No code changed for this, by the user's decision.

Box 3 is superseded in part: commit ea01363 (code review follow-up, 2026-10-18) overrode "text equal to the Python call": the `search_analyses` text uses MCP wording over MCP, and only the record equals the Python call.
