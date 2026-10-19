# 05: One name per concept

What to build: the renames settled in round 2, Q6, plus any other clashes found against the glossary, made before 0.1.0 without shims. `describe_dataset` moves to `agent_eda.data.structure`.

Read these sections of `.scratch/agent-experience/spec.md`: Names, statuses and codes (the renames and the move), and test group 3. This ticket covers user stories 12 and 15. Findings: AX-17, STD-28, and DES's questionable decision on the module.

Blocked by: none

Status: ready-for-agent

- [ ] These renames are made in code, docstrings, README, tests and the MCP schemas:
  - `infer_column_semantics(declared_schema=)` becomes `schema=`;
  - `summarize_categorical`'s `cardinality` becomes `distinct_count`;
  - `summarize_numeric`'s `count` becomes `non_missing_count`;
  - `compute_association_matrix`'s `skipped_columns` becomes `left_out_columns`, its entries keeping their reasons;
  - `find_candidate_keys`' `duplicate_row_count` becomes `duplicate_group_row_count`;
  - the finding code `declared_columns_missing` becomes `missing_declared_columns`.
- [ ] Every other result field and argument was checked against the glossary. The Comments list each further rename, or why a name stays. `class_count` stays.
- [ ] `describe_dataset` lives in `agent_eda.data.structure`. `ae.list_analyses()`, the README, the planned analyses and the tests say so.
- [ ] A test fails if any old name appears in a signature, a result field or a finding code.
- [ ] The glossary and the CHANGELOG list the renames.
- [ ] The finish checks pass.

## Comments
