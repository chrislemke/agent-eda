# 01: Server skeleton end to end

**What to build:** A user installs `agent-eda[mcp]`, registers `agent-eda-mcp` with an MCP client and calls `describe_dataset` on a CSV file. The response carries the printed result as text and `to_dict()` as structured content. Without the extra, `agent-eda-mcp` says which install command fixes it. This slice builds the module, the extra, the console script, the in-memory test harness and the stdio smoke test that the later tickets extend.

Read these sections of `.scratch/mcp-server/spec.md`: Package and dependencies, Server, Reading data (CSV only), Responses, and test groups 1 and 8. Read ADR 0005 and ADR 0006. Covers user stories 1 to 4, 19, 20 and 23.

**Blocked by:** none

**Status:** ready-for-agent

- [ ] `pyproject.toml` has the `mcp` extra (`mcp>=2.3`, `pyarrow>=14`) and the console script `agent-eda-mcp` pointing at `agent_eda.mcp:main`; `uv.lock` is updated, and `uv run pytest` installs the extra and runs the MCP tests.
- [ ] `agent_eda.errors.MissingDependencyError` exists with code `missing_dependency`. Without `mcp` importable, `agent-eda-mcp` exits with status 1 and prints the `pip install 'agent-eda[mcp]'` and uv commands to stderr, without a traceback.
- [ ] `import agent_eda` and each discovery function, in a fresh subprocess, import no `mcp` module.
- [ ] Over the SDK's in-memory client, the server is named `agent-eda`, reports the package version and lists a `describe_dataset` MCP tool with a required `data_path`.
- [ ] Calling `describe_dataset` with the path of a small CSV returns `str(res)` as text and `res.to_dict()` as structured content, equal to the Python call on the same file read by the same rule.
- [ ] The CSV is read with only empty fields as missing: a column holding `"N/A"` keeps it as a string.
- [ ] A subprocess smoke test starts `agent-eda-mcp` over stdio, initialises a session, calls `describe_dataset` and gets a completed result.
- [ ] The existing minimum-versions run still resolves pydantic 2.7 and installs no `mcp`; its MCP test modules skip with a reason naming the extra. A second minimum-versions command runs the whole suite with the extra at its floors (mcp 2.3, pyarrow 14, pydantic 2.12), and the README's Development table lists it.
- [ ] Two server processes with different `PYTHONHASHSEED` values return identical structured content for the same call.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest versions and in both minimum-versions runs.

## Comments
