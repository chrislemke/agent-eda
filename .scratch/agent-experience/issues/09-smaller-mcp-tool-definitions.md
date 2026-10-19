# 09: Smaller MCP tool definitions

What to build: MCP tool definitions shrink by at least 30% through one-sentence descriptions of the MCP-only properties and no pydantic titles. The rules move to the instructions.

Read these sections of `.scratch/agent-experience/spec.md`: MCP tool definitions, and test group 7. This ticket covers user story 24. Findings: MCP-18, MCP-10 (the tool list), DOC-13.

Blocked by: 08

Status: ready-for-agent

- [ ] The `data_path`, `table_path` and `project_path` descriptions are one sentence each. Their rules sit once in the MCP instructions, which keep within their line limit.
- [ ] No input or output schema carries a pydantic `title`, at the top level or inside `$defs`.
- [ ] The descriptions plus input schemas of all MCP tools come to at least 30% fewer characters than the 100,159 measured in October 2026. The Comments record before and after, and the size of `tools/list`.
- [ ] A test fails if the descriptions plus input schemas grow past the new size by more than 10% without the threshold being raised on purpose.
- [ ] The finish checks pass.

## Comments
