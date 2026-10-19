# Changelog

## Unreleased

### Renamed, with no old name kept

These names changed before 0.1.0, so the old ones don't work and have no warning. From 0.1.0, a renamed name keeps working for two minor releases (AGENTS.md, Names).

One name per concept:

| Where | Before | After |
| --- | --- | --- |
| `infer_column_semantics` argument | `declared_schema` | `schema`, as in `validate_column_types` |
| `summarize_categorical` field of each column | `cardinality` | `distinct_count` |
| `summarize_numeric` field of each column (and `profile_target`'s `statistics`) | `count` | `non_missing_count` |
| `compute_association_matrix` field, and its entry class | `skipped_columns`, `SkippedColumn` | `left_out_columns`, `LeftOutColumn`; entries keep their `reason` and `message` |
| `compute_association_matrix` column entry field | `level_count` | `distinct_count` |
| `find_candidate_keys` field of each column set | `duplicate_row_count` | `duplicate_group_row_count`, as in `find_duplicate_rows` |
| `find_candidate_keys` field of each column set | `distinct_count` | `distinct_row_count`, since missing values compare equal there |
| `infer_column_semantics` and `validate_column_types` finding code | `declared_columns_missing` | `missing_declared_columns` |
| `validate_column_types` finding code | `valid_columns` | `passing_columns`, since GLOSSARY.md avoids "valid" for compatible and convertible values together |

The printed text follows: the `count` column of `summarize_numeric` and of `profile_target`'s statistics is `non_missing_count`, and `compute_association_matrix` prints "Left out 1 column" where it printed "Skipped 1 column".

### Moved

- `describe_dataset` lives in `agent_eda.data.structure`, which `ae.list_analyses()` reports as its module. `agent_eda.data.provenance` no longer exists. `ae.describe_dataset` is unchanged.

### Statuses and codes

- No analysis declares or returns `failed` any more, and the `Statuses:` sections drop it. `Status` still defines it, with the reason code `numerical_failure`, for an analysis with an anticipated numerical failure. `ae.guide()`, the README and the MCP instructions list only the statuses some registered analysis can return: `completed`, `partial`, `insufficient_data`, `not_applicable` and `budget_exceeded`.
- `StatusReason.code` is one of the `StatusReasonCode` list, with a meaning for each code. Each analysis's `Statuses:` entries name the reason codes they return (`ae.describe_analysis` shows them in `StatusDescription.reason_codes`), and the registry raises when a result returns one that isn't declared. The six spellings of "nothing to work on" are merged:

  | Before | After |
  | --- | --- |
  | `no_columns_to_compare`, `no_columns_to_check`, `no_numeric_columns`, `no_categorical_columns`, `fewer_than_two_columns` | `no_columns` |
  | `no_finite_values`, `no_labels`, `no_valid_labels`, `no_defined_pairs` | `no_values` |
  | `no_hashable_columns` | `unhashable_values` |

- `ae.describe_analysis` and the README's Reference show the meaning of each limitation code (`AnalysisDescription.limitation_codes`).
- `excluded_by_reason` is typed (`ExclusionReason`) with one spelling per reason: `missing`, `infinite`, `unexpected_label`, `unhashable_values` and `too_many_categories`. `summarize_categorical`'s `unhashable` is `unhashable_values`, and `compute_association_matrix`'s `all_missing` and `no_finite_values` are `missing` and `infinite`.
- Checks that didn't run, and nulls, carry a limitation: `sentinel_values_not_declared` in `detect_missing_sentinels` without `sentinel_values`, `schema_not_declared` and `type_not_inferred` in `infer_column_semantics`, `nullable_not_declared` in `validate_column_types`, and `not_applicable_to_task` in `profile_target`.
- `find_duplicate_rows`' `duplicate_share` is null, with a `too_few_values` limitation, when nothing was compared. It was 0.0 when the data had no column to compare.
- `validate_column_types` returns `insufficient_data` with `no_rows` for data without rows. It returned `completed`. `describe_dataset` stays `completed`, with its `empty_data` finding, since it describes data of any shape.
