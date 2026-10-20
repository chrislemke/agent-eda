# 06: Statuses and codes agents can branch on

What to build:
- only reachable statuses are declared and listed;
- status reason codes and limitation codes come from documented lists;
- exclusion reasons are typed;
- every null and every check that didn't run carries a limitation;
- data without rows gives one status everywhere.

Read these sections of `.scratch/agent-experience/spec.md`: Names, statuses and codes (from `failed` on), and test group 4. This ticket covers user stories 13 and 14. Findings: AX-20, AX-28, DES-11, STD-26, STD-27, STD-29, STD-31, TEST's questionable decision on `failed`, and STD's questionable decisions on statuses and nulls.

Blocked by: 05

Status: done

- [x] No analysis declares `failed`, and the 11 `Statuses:` sections drop it. `Status` still defines it.
- [x] `guide()`, the README and the MCP instructions list only statuses that some registered analysis can return, computed from the registry. A test checks this.
- [x] `StatusReason.code` is a `StatusReasonCode` Literal with one comment per code, and the six spellings of "nothing to work on" are merged.
- [x] Each analysis's `Statuses:` entries name their reason codes, and `describe_analysis` shows them. `_check_result` raises when a result's reason code isn't declared.
- [x] `describe_analysis` and the generated README Reference show each limitation code's meaning.
- [x] `excluded_by_reason` is typed, with one spelling per reason.
- [x] Checks that didn't run leave a limitation (STD-29). `infer_column_semantics`' null type has one (STD-31). `profile_target`'s task-dependent nulls carry one.
- [x] `find_duplicate_rows`' `duplicate_share` is null under `not_applicable`.
- [x] Data without rows gives `insufficient_data` in every analysis, `validate_column_types` included.
- [x] The CHANGELOG lists the changes.
- [x] The finish checks pass.

## Comments

- **Reason code family.** The six spellings of "nothing to work on" became `no_rows`, `no_columns` and `no_values`; the message is the qualifier (`no_numeric_columns` is `no_columns` with "The data has no numeric columns..."). `no_hashable_columns` joined `unhashable_values`. Kept as they were: `one_row`, `unsupported_values`, `incomplete_columns`, `too_many_pairs`, `too_many_combinations`. New: `numerical_failure`, the one code the `failed` status carries; no analysis returns it, and it is the only code `tests/test_status_codes.py` allows to stay undeclared.
- **Declaring the codes.** A `Statuses:` entry ends with `Reason code: ...` or `Reason codes: ... and ...`. The docstring parser splits it off the entry's text, `describe_analysis` shows each code with its meaning (`StatusDescription.reason_codes`), the MCP tool description puts the sentence back, and `_check_result` compares the returned (status, code) with the declared pairs. A code the Literal doesn't list raises `TypeError` when the analysis registers.
- **Meanings come from the Literal's comments.** `agent_eda/_codes.py` reads the comment above each code of `StatusReasonCode` and `LimitationCode` with `tokenize`, so the comments are the only copy. The first sentence of a limitation's comment is what the README prints, so each starts with a short meaning. Without the source file the meanings are empty.
- **Limitation codes are package-wide.** Analyses don't declare their limitation codes, so `describe_analysis` shows the whole list in `limitation_codes` (19 codes), and the README's Results section lists it once.
- **`excluded_by_reason`.** `ExclusionReason`: `missing`, `infinite`, `unexpected_label`, `unhashable_values`, `too_many_categories`. `compute_association_matrix` wrote `all_missing` and `no_finite_values` for a column it left out; those are now `missing` and `infinite` with the counts the column holds.
- **New limitation codes.** `schema_not_declared` (`infer_column_semantics`, field `columns`: it concerns every column, and a path per column would make the interest order of the table (ticket 01) put every column first), `nullable_not_declared` (`validate_column_types`, `columns[...].nullable`, raised whenever some declared column leaves `nullable` out), `type_not_inferred` (`columns[...].inferred_logical_type`), `not_applicable_to_task` (`profile_target`, the other task's fields). `detect_missing_sentinels` reuses `sentinel_values_not_declared`, pointing at `candidates`. `infer_column_semantics`' `too_few_values` keeps its container path `evidence`; only the null type needed a leaf.
- **Reason texts stay short.** Printed limitation lines are still cut at 120 characters until ticket 01 wraps them, so the new reasons fit on one line.
- **No rows.** `describe_dataset` is the one analysis that stays `completed` on data without rows: its `empty_data` finding is the report (first-look ticket 01 chose this, and its docstring and tests say so). Every other analysis, `validate_column_types` now included, returns `insufficient_data` with `no_rows`. `validate_column_types` still reports a declared column the data lacks.
- **README budget.** The new Statuses lines (reason codes) and the limitation list added about 3,100 characters; I offset them by tightening prose in the README (value equality, project file, MCP data, JSON and response sections, the ADR list, the Features line on errors) and by dropping `failed` and its siblings from the status lines. After merging tickets 01 to 03, 07 and 08, which had filled the README to its limit, I cut more prose (the ADR list shares one line, a few MCP and masking sentences). The README is 59,925 characters, 75 under the limit. Nothing a test checks was cut.
- **Checks.** ruff, ruff format, ty, the full suite (3,729 passed), both minimum-version runs (without the extra: 3,186 passed; with the extra: 3,729 passed), the MCP stdio smoke test and `pytest -m benchmark` (14 passed) all pass. Every check ran locally.
