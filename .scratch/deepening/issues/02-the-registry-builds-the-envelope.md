# 02: The registry builds the envelope

What to build: the registration wrapper fills in `analysis`, `method`, `parameters` and `execution`. `ColumnSupport.of` derives the dependent counts, and a typed parameters view replaces `parameters.get(...)` in result text.

Read these sections of `.scratch/deepening/spec.md`: The registry builds the envelope, and test groups 1 and 3. This ticket covers user stories 2 and 9. Findings: DES-6, STD-53.

Blocked by: 01

Status: ready-for-agent

- [ ] No analysis builds `analysis=`, `method=`, `parameters=` or `execution=` itself. Analyses pass overrides only for resolved or masked values.
- [ ] For a masking analysis, the wrapper refuses to record data-valued parameters by default (`sentinel_values`, `expected_labels`). A test shows that a forgotten override raises instead of leaking.
- [ ] `ColumnSupport.of(column, eligible, excluded_by_reason)` builds every column support, and `check_contract` asserts that support adds up.
- [ ] Result text reads its settings through a typed parameters view. No `self.parameters.get(` remains in the result classes.
- [ ] The baseline comparison passes unchanged.
- [ ] The finish checks pass.

## Comments
