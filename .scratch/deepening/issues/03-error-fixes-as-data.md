# 03: Error fixes as data

What to build: errors carry what is wrong, what is allowed and their fixes as data, and one renderer per surface writes them. Raise sites stop branching on the surface. `ColumnRule` takes fixes and can be chosen per task.

Read these sections of `.scratch/deepening/spec.md`: Error fixes as data, and test groups 1 and 4. This ticket covers user stories 3 and 9. Findings: DES-7, STD-57, DES-15, STD-55 (the fix tables).

Blocked by: 01

Status: ready-for-agent

- [ ] `_wording` defines `Pass`, `LeaveOut`, `SeeArguments`, `PythonOnly` and `DeclareSensitive`, and one renderer per surface writes them.
- [ ] No raise site branches on the surface, and `worded` and its lambdas are gone.
- [ ] `ColumnRule` takes these fixes and can be chosen per task. `profile_target` selects its target with the task's rule, and `_check_target`, `_REGRESSION_FIXES` and `_MCP_REGRESSION_FIXES` are gone.
- [ ] A property over the renderer checks every fix type on both surfaces, including the no-Python guard for MCP. It replaces the curated `MISTAKES` entries it covers.
- [ ] The message tables give byte-identical messages to the baseline.
- [ ] The finish checks pass.

## Comments
