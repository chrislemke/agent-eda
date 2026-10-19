# 02: The registry builds the envelope

What to build: the registration wrapper fills in `analysis`, `method`, `parameters` and `execution`. `ColumnSupport.of` derives the dependent counts, and a typed parameters view replaces `parameters.get(...)` in result text.

Read these sections of `.scratch/deepening/spec.md`: The registry builds the envelope, and test groups 1 and 3. This ticket covers user stories 2 and 9. Findings: DES-6, STD-53.

Blocked by: 01

Status: done

- [x] No analysis builds `analysis=`, `method=`, `parameters=` or `execution=` itself. Analyses pass overrides only for resolved or masked values.
- [x] For a masking analysis, the wrapper refuses to record data-valued parameters by default (`sentinel_values`, `expected_labels`). A test shows that a forgotten override raises instead of leaking.
- [x] `ColumnSupport.of(column, eligible, excluded_by_reason)` builds every column support, and `check_contract` asserts that support adds up.
- [x] Result text reads its settings through a typed parameters view. No `self.parameters.get(` remains in the result classes.
- [x] The baseline comparison passes unchanged.
- [x] The finish checks pass.

## Comments

- The registry sets a call context (a `contextvars` variable, scoped to the call) that `AnalysisResult.for_call(parameters=..., libraries=..., **fields)` reads. An analysis returns a real, validated result; `for_call` takes `**fields: Any`, so ty no longer checks an analysis's own result fields at the call (pydantic still validates them at runtime). I chose this over a draft object because analyses set private attributes on the result after building it (`result._labels = ...`).
- `parameters` is the arguments model's JSON dump in signature order, minus the overridden names, plus the overrides. Overrides now left: `columns` (resolved), `categorical_columns`, `sentinel_values` and `expected_labels` (masked or resolved), `schema` (normalized to ColumnSpec dumps). `id_column`, `project`, `mask_sensitive` and the plain options need none. `method` is the call's `method` option when it has one, else null (no analysis has one yet). `libraries=("scipy",)` replaces the conditional execution of compute_association_matrix.
- The guard names `sentinel_values` and `expected_labels` in `_envelope.DATA_VALUED`; it applies to analyses registered with `masking=True`, and raises RuntimeError("agent_eda bug: ...") for a given, non-overridden one. `tests/registering.py` registers stand-in analyses for the tests, with an allowance in `tests/test_suite_rules.py`.
- `ColumnSupport.of(column, eligible, excluded_by_reason=None)` raises ValueError when exclusions outnumber the eligible values. `check_contract` gained `support_problems`.
- The typed view is `res.settings` (`_envelope.Settings`): each recorded parameter read by its argument's annotation in lax mode (a Project is a Project, a schema entry a ColumnSpec), raw when it can't be read, None when absent. `for_call` and `settings` are left out of dir()/help() of a result (help of ProfileTargetResult is near its 10,000-character test limit).
- `data_shape` and `execution` moved from `_run` to `_envelope`.
- Byte check: besides the baseline, a differential over 867 generated calls (record, text and parameters hashed) gave identical output before and after. The README needs no change: no public name, field or message changed (its Reference is unchanged, checked).
- The baseline test failed in this worktree only on the two messages that embed the working directory (the recorded path was another worktree's); with the path substituted, 0 differences on pandas 3. On pandas 2 the same two items differ. Deepening 03 fixes the harness.
- Checks run: ruff, format, ty, Reference check, full pytest, both minimum-version runs, MCP stdio smoke, benchmark (all within limits), properties on the thorough profile. Docs: docs/agents/new-analysis.md steps 3 and 4.
