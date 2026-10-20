# 05: One name per concept

What to build: the renames settled in round 2, Q6, plus any other clashes found against the glossary, made before 0.1.0 without shims. `describe_dataset` moves to `agent_eda.data.structure`.

Read these sections of `.scratch/agent-experience/spec.md`: Names, statuses and codes (the renames and the move), and test group 3. This ticket covers user stories 12 and 15. Findings: AX-17, STD-28, and DES's questionable decision on the module.

Blocked by: none

Status: done

- [x] These renames are made in code, docstrings, README, tests and the MCP schemas:
  - `infer_column_semantics(declared_schema=)` becomes `schema=`;
  - `summarize_categorical`'s `cardinality` becomes `distinct_count`;
  - `summarize_numeric`'s `count` becomes `non_missing_count`;
  - `compute_association_matrix`'s `skipped_columns` becomes `left_out_columns`, its entries keeping their reasons;
  - `find_candidate_keys`' `duplicate_row_count` becomes `duplicate_group_row_count`;
  - the finding code `declared_columns_missing` becomes `missing_declared_columns`.
- [x] Every other result field and argument was checked against the glossary. The Comments list each further rename, or why a name stays. `class_count` stays.
- [x] `describe_dataset` lives in `agent_eda.data.structure`. `ae.list_analyses()`, the README, the planned analyses and the tests say so.
- [x] A test fails if any old name appears in a signature, a result field or a finding code.
- [x] The glossary and the CHANGELOG list the renames.
- [x] The finish checks pass.

## Comments

- **The six listed renames** are made everywhere without shims: signatures, result fields, finding codes, docstrings, the printed text and `to_frame()` columns, the README (the Reference is regenerated), the MCP schemas (they follow the signatures and result classes) and the tests. `compute_association_matrix`'s entry class `SkippedColumn` became `LeftOutColumn` and its reason type `SkipReason` became `LeftOutReason`; the printed line reads "Left out 1 column" where it read "Skipped 1 column".
- **Further renames, found against the glossary:**
  - `find_candidate_keys`' `distinct_count` became `distinct_row_count` (STD-28). It counts distinct combinations with missing values compared equal, not levels, so it clashed with `describe_dataset`'s `distinct_count`, which GLOSSARY.md's Level defines. STD-28 proposed `combination_count`, but `KeySearch.combination_count` already counts the sets the search covers. The internal `SetCounts` follows.
  - `compute_association_matrix`'s `level_count` on a column entry became `distinct_count`: it is the same count, a categorical column's distinct non-missing values, as `summarize_categorical`'s.
  - The `validate_column_types` finding code `valid_columns` became `passing_columns`: GLOSSARY.md avoids "valid" for compatible and convertible values together, which is what the finding groups.
- **Names that stay:**
  - `class_count` (a target's classes, which can include declared classes no row holds), as the ticket says.
  - `DescribeDatasetIndex.level_count` and `column_level_count`: pandas' own term for the levels of a MultiIndex, qualified by the index entry or the `column_` prefix; they count no values.
  - `count` on `SummarizeCategoricalLevel`, `ProfileTargetClass`, `SentinelCandidate`, `DeclaredSentinelValue`, `InvalidValue` and `ProfileTargetUnexpectedLabel`: the rows or values equal to one label or value, not a column's non-missing values.
  - `AssociationColumn.value_count`: the values the pairs can use, which leaves out infinite values, so it is not `non_missing_count`. `ProfileTargetResult.valid_label_count` and `ProfileTargetResult.non_finite_count` count something their neighbours don't.
  - `max_categories` and the left-out reason `too_many_categories`: the limit applies to a column of the categorical logical type, and the first-look spec and tickets name it; GLOSSARY.md's Level entry allows "category" for a pandas categorical, but this limit also applies to text columns, so a later pass could rename both to `max_levels`. I left it, since nothing in the audit flags it.
  - `duplicate_group_count` and `excess_copy_count` of `find_duplicate_rows`, `missing_row_count` of a key and `missing_count` elsewhere, `uniqueness_share`: each has one meaning.
- **The module move.** `describe_dataset` and its helpers moved into `agent_eda/data/structure.py`, and `agent_eda/data/provenance.py` is gone. `docs/planned-analyses.md` lists no shipped analysis, so it needs no change; its `data.provenance` rows are planned analyses that will create the module again. `ae.list_analyses()`, `ae.describe_analysis()` and the tests (`test_package.py`, `test_list_analyses.py`, `test_describe_analysis.py`, `test_warning_policy.py`) say `agent_eda.data.structure`. The README never named the module.
- **The guard** is `tests/test_names.py` (signatures, `list_analyses()` finding codes and docstrings, every result field reachable from a result class, the `to_frame()` column, the README) and one test in `tests/test_mcp_tools.py` for the MCP schemas. It also checks that CHANGELOG.md and GLOSSARY.md list each rename. GLOSSARY.md gets six entries with the old names on their Avoid lines; CHANGELOG.md is new, and the README's Documentation list links it.
- **Printed width.** The longer column names widen two printed tables. `find_candidate_keys`' table now shows its `kind` column cut at 120 characters in some results, and `summarize_categorical`'s level list is cut a few characters earlier; wrapping instead of cutting is ticket 01's. The `find_candidate_keys` example's data columns became `id` and `seq` so that the table fits the docstring line limit, and its docstring carries a `# noqa: E501` with that reason.
- **README budget.** It stays under 60,000 characters (59,964): the Documentation list's AGENTS.md line is shorter, which paid for the CHANGELOG line.
- **A test fixed on the way.** Editing `test_every_reported_set_counts_as_pandas_does` changed its derandomized examples, and one drew '' and '\x00' in one column, which pandas reads as equal text and the analysis (correctness ticket 01) does not. The test now skips column sets with NUL text, as `test_find_duplicate_rows.py` does.
