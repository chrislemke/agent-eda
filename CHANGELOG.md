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
