# 05: Project and masking

**What to build:** An MCP caller passes `project_path` to the analyses that support masking, or the user starts the server with `--project` so that every call uses the Project. Masked results over MCP equal masked results in Python, and the docs say plainly that masking is a guardrail.

Read these sections of `.scratch/mcp-server/spec.md`: Project and masking, and test group 5. Read ADR 0001. Covers user stories 26 to 29.

**Blocked by:** 04-errors-and-statuses

**Status:** ready-for-agent

- [x] MCP tools of analyses that take `project` have `project_path` (string or null) in place of `project` and keep `mask_sensitive`; the other MCP tools have neither.
- [x] `project_path` loads with `Project.load`; a missing or invalid file returns its `ProjectFileError` code and message as a tool error.
- [x] `agent-eda-mcp --project project.toml` applies the Project to every call of a tool that takes one; an invalid file stops the server with its message on stderr and exit status 1; `--help` documents the option.
- [x] Precedence holds: per-call `project_path` over `--project` over none, and a per-call `mask_sensitive` over both.
- [x] With masking on, no raw value of a sensitive column appears in the text or the structured content of any masking analysis's response; the results equal the Python calls with the same Project.
- [x] The descriptions of `project_path` and `mask_sensitive` say masking is a guardrail against accidental exposure, not anonymization or access control.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

What this ticket settled that the spec left open:

- **Schema.** On the 5 masking analyses, `project_path` (`anyOf` string/null, default null, fixed server text) takes `project`'s place in the property order, and the `Project` and `DataPolicy` `$defs` are dropped (`validate_column_types` keeps `ColumnSpec`). `mask_sensitive` keeps its docstring text, which already says masking is a guardrail, plus one fixed sentence: "Over MCP, the project is the file at project_path, or else the one the server was started with (agent-eda-mcp --project)." The `project_path` text describes the TOML file's keys and ends with the guardrail sentence and "whoever can call the MCP server can usually read the data file too" (ADR 0001, 0006). `tests/test_mcp_tools.py`'s schema test now expects this mapping.
- **Seam for the server's Project.** `agent_eda.mcp.create_server(*, analyses=None, project=None)` takes a `Project`; `main` loads `--project PATH` with `Project.load` and passes it. Tests use `create_server(project=...)` in memory, plus one stdio test of `agent-eda-mcp --project` with a relative path.
- **Start-up errors.** `main` checks the extra first, then loads the project. `AgentEDAError` and `OSError` (such as a project file without read permission) print the message alone to stderr and exit 1; stdout stays empty.
- **Precedence.** `project_path: null` and a left-out `project_path` both mean "the server's project". A call can't drop the server's Project, but `mask_sensitive: false` turns masking off for that call, as in Python. Without any project, `mask_sensitive: true` gives the Python `invalid_argument` error, whose text shows the Python fix (`project=ae.Project(...)`); it stays byte-equal to Python's, as ticket 04 settled for value messages.
- **Per-call errors.** `Project.load`'s errors (`project_file_not_found`, `invalid_project_file`, `unknown_project_key`) arrive as tool errors with the Python code and message. A `project_path` that isn't a string or null gives `invalid_argument` naming `project_path`, as for `data_path`. A project file that can't be read for another OS reason (permissions) isn't an `AgentEDAError`, so per call it becomes ticket 04's JSON-RPC internal error.
- **No raw sensitive value.** `tests/test_mcp_project.py` calls each of the 5 masking analyses with options that show the sensitive columns (levels, invalid values, classes, declared sentinel values, row IDs from `id_column`). It checks that the unmasked response does show them, so the check isn't vacuous, that the masked text and structured content show none, and that the masked response equals the Python call. A test fails when a new masking analysis has no entry there.

For later tickets: ticket 06 should extend the masking test to the written table (`table_path`), which spec test group 5 names. Ticket 07's `instructions` should explain `project_path` and `--project` and repeat the guardrail sentence. Ticket 08's README section can reuse the `--project` help text from `agent-eda-mcp --help`.

**pyarrow floor (security, CVE-2023-47248).** The mcp extra now requires `pyarrow>=14.0.1` (the dev group already did). `uv.lock` was relocked with uv 0.13.1, the hook's version; its only other change is uv 0.13.1's ordering of `resolution-markers`. The floor reads "pyarrow 14.0.1" in the README's minimum-versions note, the spec and ticket 08, and `tests/test_packaging.py` checks the wheel's `Requires-Dist`. The minimum run with the extra already resolved 14.0.1, so no installed version moved; ticket 03's Parquet DeprecationWarning filter is still needed with 14.0.1.

Checks: `ruff check`, `ruff format`, `ty check` and `pytest` pass on the latest versions (run with `--extra mcp`, local uv 0.10.11), and in both minimum-versions runs (core: 1835 passed, MCP modules skipped; with the extra at pyarrow 14.0.1, pydantic 2.12.0, pandas 2.2.0, numpy 1.26.0: 1987 passed). CI wasn't run.
