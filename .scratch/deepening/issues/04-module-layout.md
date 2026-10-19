# 04: Module layout

What to build: modules sit where their scope is visible:
- the MCP adapter in `agent_eda/_mcp/`;
- single-analysis engines beside their analysis;
- argument handling in `_arguments`;
- `_run` split;
- one type renderer and one closest-name helper;
- immutable MCP schema fragments;
- no private imports across modules.

Read these sections of `.scratch/deepening/spec.md`: Module layout, and test groups 1 and 5. This ticket covers user stories 4 and 9. Findings: DES-9, STD-35, STD-36, STD-49, STD-50, STD-56.

Blocked by: 03

Status: ready-for-agent

- [ ] `_mcp_server`, `_mcp_instructions`, `_data_files`, `_local_files` and `_table_files` live in `agent_eda/_mcp/`. `agent_eda.mcp` and the `agent-eda-mcp` console script are unchanged.
- [ ] `_association`, `_keys`, `_conversions` and `_sentinel_rules` live beside their analysis as private siblings.
- [ ] Argument normalizing and the message helpers live in `_arguments`, with one type renderer and one closest-name helper. `_registry` holds only the registry.
- [ ] `_run` is split into status helpers and a warnings module that also holds `_absorb_warnings`. `_sentinels` has a name for what it holds.
- [ ] MCP schema fragments are immutable, and the file-format knowledge lives in one place.
- [ ] A test fails when a module imports an underscore name from another module that isn't one of the internal-interface modules it allows.
- [ ] `tests/test_packaging.py`'s module name strings follow the move.
- [ ] The baseline comparison passes unchanged.
- [ ] The finish checks pass.

## Comments
