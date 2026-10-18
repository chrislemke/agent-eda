# 04: Errors and statuses

**What to build:** An MCP caller who makes a mistake gets a tool error carrying the stable code and the same fix-it message a Python caller gets, and can retry. A run that doesn't complete comes back as a normal result with its status and reason. Only bugs become protocol errors.

Read these sections of `.scratch/mcp-server/spec.md`: Errors, and test group 4. Covers user stories 24 and 25.

**Blocked by:** 02-every-analysis-as-an-mcp-tool

**Status:** ready-for-agent

- [x] An `AgentEDAError` from an analysis, such as a misspelled column, returns `isError: true` with the exception's message as text and `{"code", "message"}` as structured content, equal to what the Python call raises.
- [ ] The errors from ticket 03 arrive the same way, with their codes.
- [x] An error result's structured content doesn't trip the client's output-schema validation; the ticket comments record how the SDK handles `isError` with an output schema.
- [x] Arguments that fail the input schema (wrong type, unknown property, value outside a Literal) come back as a tool error naming the property. The ticket comments record whether the SDK or our code produces it and the exact message shape.
- [x] A non-completed status, such as `insufficient_data` on an empty CSV, is a normal result with `status` and `status_reason` in the structured content.
- [x] An unexpected exception propagates as a JSON-RPC internal error (tested with a stand-in analysis registered in the test that raises).
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

What this ticket settled that the spec left open:

- **One place for every error.** The server's `call_tool` runs the call (argument checks, data reading, analysis) inside one `try`: an `AgentEDAError` becomes `CallToolResult(is_error=True, content=[TextContent(str(error))], structured_content={"code": error.code, "message": str(error)})`. Ticket 03's reader errors are `AgentEDAError`s raised inside that `try`, so they need nothing more.
- **`isError` with an output schema (SDK `mcp` 2.3.0).** The server sends whatever it returns, unvalidated. The SDK client (`ClientSession.call_tool` → `validate_tool_result`) validates `structuredContent` against the tool's output schema only when `isError` is false, so the `{"code", "message"}` record reaches the caller intact in both protocol eras (legacy handshake and the 2026-07-28 era); the test checks that the record does fail the output schema, so the check means something. Other clients may validate differently; the protocol allows this shape.
- **Input-schema errors come from our code, not the SDK.** The low-level SDK `Server` checks nothing against `inputSchema`, and neither does its client: every argument reaches `call_tool` as sent. The server then checks, in this order, before reading data:
  1. An argument that isn't one of the MCP tool's properties: code `invalid_call` (the code Python gives an unknown keyword), message from the registry's shared `unknown_name_message`, listing the MCP tool's properties, so `data` suggests `data_path`: `summarize_numeric has no argument 'colums'. Did you mean 'columns'? Allowed: data_path, columns, quantiles, quantile_method, trim_fraction.`
  2. A required property left out: code `invalid_call`, message from the registry's shared `missing_message`, with the Python opening and choices but an MCP fix: `profile_target is missing 2 required arguments: target_column and task. Pass them in the MCP tool's arguments, such as {"target_column": ..., "task": ...}, where task is one of 'binary_classification', 'multiclass_classification' or 'regression'. The MCP tool's input schema describes every argument.`
  3. A `data_path` that isn't a string: code `invalid_argument`, `summarize_numeric got 3 for data_path, but data_path must be a string: the path to the data file, absolute or relative to the MCP server's working directory.`
  4. Every other value (wrong type, outside a Literal, out of range) goes to the analysis, whose strict pydantic validation raises `InvalidArgumentError` (code `invalid_argument`) with exactly the Python message, which names the property: `summarize_numeric got 'lineer' for quantile_method, but quantile_method must be one of ... Did you mean 'linear'? See help(ae.summarize_numeric) for every argument.`
  So the spec's "code `invalid_argument`" holds for bad values; an unknown or missing property gets `invalid_call`, as the same mistake does in Python (`agent_eda.errors` documents `invalid_call` for an unknown keyword or a missing argument). The value messages end with the Python hint `See help(ae.<name>) ...`; they stay byte-equal to Python's, as user story 24 asks. Where the registry's normalisation is looser than the schema (a single string for a list option becomes a one-item list, as in Python), the server accepts it like Python does.
- **Bugs.** Any other exception is logged with its traceback on the `agent_eda.mcp` logger (stderr under stdio) and re-raised as `MCPError(code=INTERNAL_ERROR (-32603))`, message `<tool> failed with an unexpected error, which is a bug in agent-eda, not in the call: <Type>: <text>. The MCP server's log on stderr has the traceback.` Without this, SDK 2.3 answers a client that did the `initialize` handshake with code 0 (`jsonrpc_dispatcher.py`, "TODO(L58)"), not an internal error. With the in-memory client, catch the `MCPError` inside `async with Client(...)`: leaving the client wraps it in an `ExceptionGroup`.
- A non-completed status needed no code: a header-only CSV gives `summarize_numeric` `insufficient_data`, returned as a normal result equal to the Python call's.
- Tests are in `tests/test_mcp_errors.py`; the raising stand-in uses ticket 02's seam, `create_server(analyses=[stand_in])`.

For later tickets: put new MCP-only properties (`project_path`, `table_path`) in the input schema and they are covered by checks 1 and 2; add a type check like `data_path`'s if they need one, and raise `AgentEDAError`s inside the `try` so they arrive as tool errors.

Checks: ruff, ty and pytest pass on the latest versions (run with `--extra mcp`, since the local uv 0.10.11 skips the extra otherwise) and in both minimum-versions runs. CI wasn't run.
