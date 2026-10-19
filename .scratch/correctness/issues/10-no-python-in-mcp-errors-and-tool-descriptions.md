# 10: No Python in MCP errors and tool descriptions

What to build: nothing an MCP caller reads in an error message, a tool description or an input schema holds Python. A guard test keeps it that way.

Read these sections of `.scratch/correctness/spec.md`: No Python in what MCP callers read, and test group 8. This ticket covers user story 24. Findings: DES-7 (the three leaks), SPEC-3, SPEC-12, MCP-7, STD-8, DOC-10 (descriptions).

Blocked by: 08, 09

Status: ready-for-agent

- [ ] These MCP messages give MCP fixes:
  - sentinel examples, as `{"sentinel_values": {"age": [-999]}}`;
  - "did you mean" for datetimes and durations, as ISO strings;
  - `profile_target`'s task fix, as `{"task": "multiclass_classification"}`;
  - `project_path` problems, naming `project_path` and saying how to write the TOML file, with no `project.save` step;
  - echoed values, written as JSON (`null`, `true`).
- [ ] `tests/test_mcp_project.py` expects the MCP fix for a missing project file.
- [ ] The analysis docstrings' When to use, Returns, Statuses and See Also sections read correctly over MCP. MCP tool descriptions hold no `res.`, `df.`, `columns=None` or `table="matrix"`, and Python callers still get usable text.
- [ ] The MCP instructions say, in one line, that a description's `to_dict` record is the structured content and its `to_frame` table is what `table_path` writes (SPEC-12).
- [ ] A guard test scans every MCP tool description and input schema for Python syntax:
  - call syntax;
  - `name=value` keywords;
  - `pd.`, `np.`, `df.`, `res.` and `ae.`;
  - `None`, `True` and `False` used as values.
- [ ] A mistake table, extending `MISTAKES`, reaches every error code each MCP tool can raise, and the guard scans those messages too.
- [ ] The finish checks pass.

## Comments
