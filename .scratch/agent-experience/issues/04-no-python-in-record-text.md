# 04: No Python in record text

What to build: status reasons, limitation reasons, finding claims and field descriptions name arguments and fixes without Python code, so that MCP callers, who read the record, never see Python. The guard test covers output schemas and records.

Read these sections of `.scratch/agent-experience/spec.md`: Records (No Python in record text), and test group 2. This ticket covers user story 11. Findings: STD-39, SPEC-10, AX-27, STD-9, and MCP's questionable decision on Python wording in records.

Blocked by: none

Status: ready-for-agent

- [ ] No status reason, limitation reason or finding claim holds code:
  - "make the index unique" replaces `df.reset_index(drop=True)`;
  - "an explicit type conversion" replaces `.astype('float64')`;
  - "the max_pairs limit of 1" replaces `max_pairs=1`;
  - "leaving the columns argument out" replaces `columns=None`.
- [ ] Field descriptions hold no `isna()`, `duplicated()`, `df.drop_duplicates(...)` or `scipy.stats.trim_mean(x, 0.1)`, and no mention of `table_path` on the Python side.
- [ ] The printed text and error messages keep their surface-specific fixes.
- [ ] The guard test from correctness ticket 10 scans every output schema and the records of every recipe.
- [ ] The Comments record that this reverses mcp-server ticket 07's choice.
- [ ] The finish checks pass.

## Comments
