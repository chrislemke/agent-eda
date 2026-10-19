# 09: Smaller MCP tool definitions

What to build: MCP tool definitions shrink by at least 30% through one-sentence descriptions of the MCP-only properties and no pydantic titles. The rules move to the instructions.

Read these sections of `.scratch/agent-experience/spec.md`: MCP tool definitions, and test group 7. This ticket covers user story 24. Findings: MCP-18, MCP-10 (the tool list), DOC-13.

Blocked by: 08

Status: ready-for-agent

- [x] The `data_path`, `table_path` and `project_path` descriptions are one sentence each. Their rules sit once in the MCP instructions, which keep within their line limit.
- [x] No input or output schema carries a pydantic `title`, at the top level or inside `$defs`.
- [ ] The descriptions plus input schemas of all MCP tools come to at least 30% fewer characters than the 100,159 measured in October 2026. The Comments record before and after, and the size of `tools/list`.
- [x] A test fails if the descriptions plus input schemas grow past the new size by more than 10% without the threshold being raised on purpose.
- [ ] The finish checks pass.

## Comments

Open: the third box stays unticked, so `Status:` stays `ready-for-agent`. The 100,159 baseline is stale. Descriptions plus input schemas were 120,678 characters at the start of this ticket (13 tools: the 11 analyses plus `search_analyses` and `list_analyses`; the correctness and agent-experience tickets since the audit added reading rules to `data_path`, longer `Returns:` sections and the two discovery tools). They are now 84,467. That is 30.0% fewer than 120,678 but 15.7% fewer than 100,159; the box's literal bar is at most 70,111. The remaining 14,356 characters are docstring prose, not repeated MCP text: `Returns:` is 31,304 of the 52,829 characters of descriptions, `When to use:` 10,602 and `Statuses:` 5,579, and the input schemas still hold the `Args:` texts (`columns` 3,834, `mask_sensitive` 3,349, `sentinel_values` 3,053, `$defs` 2,846). Cutting them means rewriting docstrings, which ticket 10 owns (DOC-13: describe each field once), and tickets 06 (the `failed` status lines) and 07 are changing them now. Decision for the maintainer: accept 84,467 (30% under what the tools were when this ticket started), or leave the box open until 10 lands and re-measure.

Measured with the same sum the test uses: `len(description) + len(json.dumps(input_schema))` over every tool of `tools/list`.

| Characters | Before (start of this ticket) | After |
| --- | --- | --- |
| Descriptions | 52,829 | 52,829 |
| Input schemas | 67,849 | 31,638 |
| Descriptions plus input schemas | 120,678 | 84,467 |
| Output schemas | 180,729 | 164,815 |
| Whole `tools/list` | 304,742 | 252,617 |

Audit baseline (October 2026, 11 tools): 100,159 for descriptions plus input schemas, 257,899 for `tools/list`. The input schemas dropped 36,211 characters: `table_path` 17,897 to 2,046 over the 11 analysis tools, `data_path` 15,455 to 1,694, `project_path` 5,130 to 1,332, `mask_sensitive` 4,717 to 3,349, and the 59 input titles are gone. The output schemas dropped 15,914 characters, the 682 titles.

What the ticket settled that the spec left open:
- **Where the rules went.** The MCP instructions now hold the CSV/TSV and Parquet reading rules (the shared `CSV_RULES` and `PARQUET_RULES`), the `table_path` rules (main table only, directory, no overwrite or symbolic link, the response line, `table_file`, JSON text cells, index column) and the project file format with its default. They are 57 of 60 lines; the overview below them doesn't grow with more analyses (the 60-stand-in test still passes).
- **`mask_sensitive`.** Its MCP description is the docstring's text without the guardrail sentence (the instructions and `--help` say it once) plus "Over MCP, the project is project_path." The precedence sentence is gone from it and from `project_path`.
- **Titles are dropped on the MCP surface only**, by `_wording.without_titles`, for the analysis tools' input and output schemas and the two discovery tools' output schemas. Python's `parameters_schema` and `record_schema()` keep theirs, so tickets 06, 07 and 10 don't collide with `results.py`.
- **The size guard** (`DEFINITION_SIZE` in `tests/test_mcp_tools.py`, 84,500) fails above 10% over it, so a new analysis or a longer docstring has to raise it on purpose.
- Checks: ruff, ty, pytest, both minimum-version runs and the MCP stdio smoke pass. The benchmark was not run: no analysis code changed.
