# 04: Errors and statuses

**What to build:** An MCP caller who makes a mistake gets a tool error carrying the stable code and the same fix-it message a Python caller gets, and can retry. A run that doesn't complete comes back as a normal result with its status and reason. Only bugs become protocol errors.

Read these sections of `.scratch/mcp-server/spec.md`: Errors, and test group 4. Covers user stories 24 and 25.

**Blocked by:** 02-every-analysis-as-an-mcp-tool

**Status:** ready-for-agent

- [ ] An `AgentEDAError` from an analysis, such as a misspelled column, returns `isError: true` with the exception's message as text and `{"code", "message"}` as structured content, equal to what the Python call raises.
- [ ] The errors from ticket 03 arrive the same way, with their codes.
- [ ] An error result's structured content doesn't trip the client's output-schema validation; the ticket comments record how the SDK handles `isError` with an output schema.
- [ ] Arguments that fail the input schema (wrong type, unknown property, value outside a Literal) come back as a tool error naming the property. The ticket comments record whether the SDK or our code produces it and the exact message shape.
- [ ] A non-completed status, such as `insufficient_data` on an empty CSV, is a normal result with `status` and `status_reason` in the structured content.
- [ ] An unexpected exception propagates as a JSON-RPC internal error (tested with a stand-in analysis registered in the test that raises).
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments
