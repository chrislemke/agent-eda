# 01: Server skeleton end to end

**What to build:** A user installs `agent-eda[mcp]`, registers `agent-eda-mcp` with an MCP client and calls `describe_dataset` on a CSV file. The response carries the printed result as text and `to_dict()` as structured content. Without the extra, `agent-eda-mcp` says which install command fixes it. This slice builds the module, the extra, the console script, the in-memory test harness and the stdio smoke test that the later tickets extend.

Read these sections of `.scratch/mcp-server/spec.md`: Package and dependencies, Server, Reading data (CSV only), Responses, and test groups 1 and 8. Read ADR 0005 and ADR 0006. Covers user stories 1 to 4, 19, 20 and 23.

**Blocked by:** none

**Status:** done

- [x] `pyproject.toml` has the `mcp` extra (`mcp>=2.3`, `pyarrow>=14`) and the console script `agent-eda-mcp` pointing at `agent_eda.mcp:main`; `uv.lock` is updated, and `uv run pytest` installs the extra and runs the MCP tests.
- [x] `agent_eda.errors.MissingDependencyError` exists with code `missing_dependency`. Without `mcp` importable, `agent-eda-mcp` exits with status 1 and prints the `pip install 'agent-eda[mcp]'` and uv commands to stderr, without a traceback.
- [x] `import agent_eda` and each discovery function, in a fresh subprocess, import no `mcp` module.
- [x] Over the SDK's in-memory client, the server is named `agent-eda`, reports the package version and lists a `describe_dataset` MCP tool with a required `data_path`.
- [x] Calling `describe_dataset` with the path of a small CSV returns `str(res)` as text and `res.to_dict()` as structured content, equal to the Python call on the same file read by the same rule.
- [x] The CSV is read with only empty fields as missing: a column holding `"N/A"` keeps it as a string.
- [x] A subprocess smoke test starts `agent-eda-mcp` over stdio, initialises a session, calls `describe_dataset` and gets a completed result.
- [x] The existing minimum-versions run still resolves pydantic 2.7 and installs no `mcp`; its MCP test modules skip with a reason naming the extra. A second minimum-versions command runs the whole suite with the extra at its floors (mcp 2.3, pyarrow 14, pydantic 2.12), and the README's Development table lists it.
- [x] Two server processes with different `PYTHONHASHSEED` values return identical structured content for the same call.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments

What this ticket settled that the spec left open:

- **pyproject layout.** The extra `mcp = ["mcp>=2.3", "pyarrow>=14"]`, a dependency group `mcp = ["agent-eda[mcp]"]` in `default-groups` (so `uv run pytest` installs the extra), an empty group `core`, and `[tool.uv] conflicts` between `core` and both the extra and the `mcp` group. Without the conflicts uv resolves all extras together and the minimum run gets pydantic 2.12. Listing `mcp>=2.3` directly in the `mcp` group (instead of `agent-eda[mcp]`) lifted the core fork to pydantic 2.12 too (measured), so the group references the extra.
- **Minimum-versions commands.** Core: `uv run --isolated --resolution lowest-direct --no-group mcp --group core --python 3.12 pytest` (measured: pydantic 2.7.0, numpy 1.26.0, pandas 2.2.0, no mcp; `tests/test_mcp_server.py` skips with the extra's install command). With the extra: `uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 pytest` (measured: mcp 2.3.0, pydantic 2.12.0, pyarrow 14.0.1, numpy 1.26.0, pandas 2.2.0; whole suite passes). The core command changed, and `tests/test_ci_workflow.py` pins the README row to CI's `tests-minimum` step, so that step (and the benchmark's minimum step) now run the new command. The second CI job is left to ticket 08.
- **uv version.** `uv.lock` was written with uv 0.13.1 (CI and the uv-lock hook). The local uv 0.10.11 drops the extra on `uv sync` (it ignores the self-referencing group), so with it `uv run pytest` skips the MCP tests unless `--extra mcp` is passed; uv 0.13.1 installs it. The README's Development notes say so.
- **Module layout.** `agent_eda.mcp` is public and imports nothing from `mcp`: `create_server()` (the seam the in-memory tests connect to; ticket 05 can add the Project to it) and `main(argv=None)` (the console script; it prints an `AgentEDAError` that stops the start to stderr and exits 1). The protocol code lives in the private `agent_eda._mcp_server`, imported only after the extra is found: one data reader `read_data(data_path)` and one response builder `_response(result)`, which later tickets extend. `--help` works without the extra.
- **What 02 replaces.** The served analyses are the fixed tuple `_ANALYSES = ("describe_dataset",)`, the MCP tool description is only the summary line, and the input schema has only `data_path`. An unknown MCP tool name raises `MCPError(INVALID_PARAMS)`; a missing `data_path` is not handled yet (ticket 04).
- `MissingDependencyError` is also an `ImportError`. The server doesn't fill `describe_dataset`'s `source` from the data path, so the MCP call equals the Python call without `source`.
- Tests are sync functions calling `anyio.run`, with `mcp.Client(create_server(), mode="legacy")` in memory; the stdio tests start the `agent-eda-mcp` script next to `sys.executable`, since the SDK passes the subprocess only a few environment variables.

Checks: the four commands pass on the latest versions and in both minimum-versions runs; the MCP and packaging tests also pass on Python 3.14. CI itself wasn't run.

Box 5 is superseded in part: commit ea01363 (code review follow-up, 2026-10-18) gave the MCP text MCP wording, so the text is no longer `str(res)`. The structured content still equals `res.to_dict()` unless the call has `table_path`.
