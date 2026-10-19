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

### Changed

- Every result's `to_frame` takes `table=`, a one-value `Literal` for a result with one table (`describe_dataset`, `summarize_numeric`: `'columns'`; `summarize_categorical`: `'levels'`; `find_duplicate_rows`: `'rows'`; `profile_target`: `'target'`; `detect_missing_sentinels` and `find_candidate_keys`: `'candidates'`). Any other name raises `invalid_argument` with the table names.
- `res.columns["Age"]` raises a `KeyError` that names the closest columns, on every result whose `columns` is keyed by name. `describe_dataset`'s `columns` stays a list, since labels can repeat, and also takes a name.
- pydantic's deprecated `json()`, `dict()`, `schema()`, `parse_obj()` and `copy()` on a result raise `invalid_argument`, naming `to_json()`, `to_dict()`, `record_schema()`, `model_validate()` or `model_copy()`.
- `help(type(res))` shows each field with its description and points to `ae.describe_analysis`, in under 10,000 characters. Rich prints the text template of a result, not pydantic's field dump.
- `ae.data`, `ae.statistics`, `ae.discovery`, `ae.project` and `ae.mcp` resolve on a fresh import. `ae.AgentEDAError` and other names that live in `ae.errors` or `ae.results` raise an `AttributeError` that points there.

### Added

- `InferColumnSemanticsResult.inferred_schema()`, a `schema` for `validate_column_types` that leaves out columns without an inferred type.
- `DetectMissingSentinelsResult.sentinel_values(columns=None)`, the `sentinel_values` for `profile_missing_values`. Masked candidates are left out and named in the returned mapping's `note`.
- `rich` is a dev dependency, for the test of the Rich display.
