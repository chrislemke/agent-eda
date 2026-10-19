# 02: Every analysis as an MCP tool

**What to build:** Every registered analysis becomes an MCP tool generated from the registry, with a description built from its docstring, a JSON Schema built from its `arguments` model and an output schema for its result. Registering and exporting a new analysis is enough to give it an MCP tool, and a contract test fails when that stops being true.

Read these sections of `.scratch/mcp-server/spec.md`: MCP tools for analyses, and test group 2. The contract test checks `project_path` and `table_path` once tickets 05 and 06 add them. Covers user stories 7 to 9 and 30.

**Blocked by:** 01-server-skeleton-end-to-end

**Status:** done

- [x] The server lists one MCP tool per entry in `ae.list_analyses()`, named like the analysis, built from the registry at start with no per-analysis server code.
- [x] Each description holds the docstring's summary, `When to use:`, `Returns:` with the finding codes, `Statuses:` and `See Also:`, in that order, and no `Examples:` or Python call syntax.
- [x] Each input schema is the `arguments` model's schema with `Args:` text as property descriptions, and `data` replaced by the required `data_path`. (`project_path` arrives with ticket 05 and `table_path` with ticket 06.) For summarize_numeric, the schema lists the allowed `quantile_method` values and the range of `trim_fraction`.
- [x] Each MCP tool carries the annotations `readOnlyHint: false`, `destructiveHint: false`, `idempotentHint: true`, `openWorldHint: false`.
- [x] Each MCP tool has an output schema, and for every analysis the `to_dict()` of a real call validates against it with `jsonschema`.
- [x] The contract test calls every analysis MCP tool with valid arguments on a small mixed-dtype CSV, gets a declared status, and gets structured content equal to the Python call's `to_dict()`.
- [x] A parameter with no JSON Schema raises at server start, naming the analysis and the parameter (tested with a stand-in analysis registered in the test).
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

What this ticket settled that the spec left open:

- **Stand-in seam.** Tests may import only public modules, and registering a stand-in with the private decorator would also leave it in the global registry for every later test. So the public `agent_eda.mcp.create_server(*, analyses=None)` takes the analysis functions to serve; `None` serves every registered analysis. A registered function uses its registry entry. Any other function, such as a stand-in a test defines, is read the way the decorator reads an analysis (`_registry.spec_of`: signature, docstring, result class; statuses and finding codes from its docstring), without registering it, and the server calls it as it is, without the registry's checks. Ticket 04's raising stand-in uses the same seam: `create_server(analyses=[stand_in])`. This is a Python seam for tests and embedding code, not the out-of-scope preset option: the CLI has no option for it.
- **Server start.** `create_server()` builds every MCP tool eagerly, so a broken analysis raises there, not on `tools/list`. An analysis parameter whose type has no JSON Schema raises `TypeError` (as the registry's own malformed-analysis errors do), naming the analysis, the parameter and its type. A role DataFrame parameter would raise too: the `<role>_path` rule is left for the first analysis that takes one.
- **Description.** Summary line, then `When to use:` (one paragraph), `Returns:` (the docstring's body, dedented, with its line breaks and bullets), `Statuses:` (one `status: reason` line per declared status) and `See Also:` (one `name: text` line per entry), separated by blank lines. The docstring parser now keeps `returns` and `see_also`.
- **"No Python call syntax"** means no `name(` anywhere in the description (the test's pattern is `[A-Za-z_][A-Za-z0-9_.]*\(`). The docstrings were edited at the source, so Python and MCP read the same text: `` `to_frame()` `` became `` `to_frame` ``, `to_frame(table="matrix")` became "`to_frame` ... with table="matrix"", `isna()` became `` `isna` `` and `duplicated()` became `DataFrame.duplicated`. Subscripts such as `res.columns["age"]` and `df.loc[row_ids]` stay; they aren't calls. Ticket 07's instructions should say that a description's `to_dict` record is the structured content and `to_frame` the complete table (which `table_path` writes, ticket 06).
- **Input schema.** `AnalysisSpec.parameters_schema()` now builds the schema with the `Args:` descriptions; `describe_analysis()` and the server both call it. The server puts `data_path` first in `properties` and `required`.
- **Output schema.** The new public classmethod `AnalysisResult.record_schema()` gives the JSON Schema of the `to_dict()` record: pydantic's validation-mode schema plus a required `<field>_total` (integer, or null when the list or the field its total comes from can be null) next to every capped field; `additionalProperties` stays false. The SDK client also validates each call's structured content against it, so the call test checks it a second time.
- The contract test reads the mixed-dtype CSV the way the server does and takes each analysis's required arguments from `tests/recipes.py`, so a new analysis needs only its recipe.
- `type_text` now shows a `Callable`'s parameters, as in `Callable[[float], bool]`.

Checks: ruff, ty and pytest pass on the latest versions (run with `--extra mcp`, since the local uv 0.10.11 skips the extra otherwise) and in both minimum-versions runs (core: MCP modules skip; with the extra: whole suite passes). CI wasn't run.

**Code review follow-up (2026-10-18).** The output schemas' field descriptions no longer say `to_frame()`: they name "the full table (to_frame, or table_path over MCP)", or "in Python, to_frame with table='...'" for a table only Python can ask for. This follows the user's decision that nothing an MCP caller reads holds `ae.` or call syntax. Formulas such as `sqrt(...)` stay. `tests/test_mcp_tools.py` checks every output schema for `to_frame(`, `to_dict(`, `to_json(`, `res.` and `ae.`. The output schema also gains the optional `table_file` property (ticket 06's follow-up). `create_server(analyses=...)` stays as the test seam; its docstring now says that serving a subset or an unregistered function is unsupported outside tests.

**Spec review follow-up (2026-10-18).** The input schemas of `infer_column_semantics` and `validate_column_types` held the whole ColumnSpec docstring under `$defs`, with its `Examples:` doctest, `ae.ColumnSpec(...)` and `ae.errors.InvalidArgumentError`. ColumnSpec's JSON Schema description now leaves out `Examples:` (`_docstring.without_examples`, its `json_schema_extra`), so Python's `parameters_schema` changes too, and its prose names no Python call. `test_the_input_schema_holds_no_python_example_or_call` checks every MCP tool's input schema; it allows formulas such as `int(trim_fraction * n)` but not a class built with call syntax.
