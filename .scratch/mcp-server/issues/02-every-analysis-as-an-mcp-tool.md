# 02: Every analysis as an MCP tool

**What to build:** Every registered analysis becomes an MCP tool generated from the registry, with a description built from its docstring, a JSON Schema built from its `arguments` model and an output schema for its result. Registering and exporting a new analysis is enough to give it an MCP tool, and a contract test fails when that stops being true.

Read these sections of `.scratch/mcp-server/spec.md`: MCP tools for analyses, and test group 2. Covers user stories 7 to 9 and 30.

**Blocked by:** 01-server-skeleton-end-to-end

**Status:** ready-for-agent

- [ ] The server lists one MCP tool per entry in `ae.list_analyses()`, named like the analysis, built from the registry at start with no per-analysis server code.
- [ ] Each description holds the docstring's summary, `When to use:`, `Returns:` with the finding codes, `Statuses:` and `See Also:`, in that order, and no `Examples:` or Python call syntax.
- [ ] Each input schema is the `arguments` model's schema with `Args:` text as property descriptions, `data` replaced by the required `data_path`, and an optional `table_path` property (left unimplemented until ticket 06, documented as such in the ticket comments). For summarize_numeric, the schema lists the allowed `quantile_method` values and the range of `trim_fraction`.
- [ ] Each MCP tool carries the annotations `readOnlyHint: false`, `destructiveHint: false`, `idempotentHint: true`, `openWorldHint: false`.
- [ ] Each MCP tool has an output schema, and for every analysis the `to_dict()` of a real call validates against it with `jsonschema`.
- [ ] The contract test calls every analysis MCP tool with valid arguments on a small mixed-dtype CSV, gets a declared status, and gets structured content equal to the Python call's `to_dict()`.
- [ ] A parameter with no JSON Schema raises at server start, naming the analysis and the parameter (tested with a stand-in analysis registered in the test).
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments
