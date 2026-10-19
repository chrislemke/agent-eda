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

### Printed text

- Printed text wraps long finding, limitation, fact and reason lines with a hanging indent instead of cutting them with "...". The 40-line cap and the 120-character width still hold, and the text counts what it leaves out.
- Printed findings are ordered violation, suspected, descriptive, passed, unknown, so a violation is never hidden behind suspected findings. The record keeps the analysis's own order.
- Printed tables of columns list the columns with a finding or limitation first, then the rest in DataFrame order. `profile_missing_values` and `validate_column_types` keep their own sorting.
- `summarize_categorical` prints each column's top levels on their own lines under the table, and its `case_or_space_variants` finding names the variants of each column (placeholders for masked columns).
- Printed text writes every share as a fraction under its field name, such as `missing_share 0.0598` and `trim_fraction 0.1`, and no longer prints percentages. The `missing_labels` finding of `profile_target` reads `(missing_share 0.125)`.
- The "Left out" lines of `summarize_numeric` and `summarize_categorical` say why each column was left out, and `summarize_numeric` no longer lists booleans as "not numeric".
- The `schema` line of `infer_column_semantics` accounts for every declared column: matching, not matching, not inferred and not in the data.
- The footer of an empty table says it has no rows.

### Removed from records

- `summarize_categorical`: `most_common_levels`, `share_denominator` and `level_order`. `levels` now holds the first `top_n` levels (it held 100), on the result and in the record, which adds `levels_total` (the `distinct_count`). The printed text calls its column `levels`. `to_frame()` and `table_path` still have every level.
- `profile_target`: `level_order`. `summarize_numeric`: `definitions`. `find_duplicate_rows`: `missing_values_compare_equal`. Their meaning moved into field descriptions, which `describe_analysis` and the MCP output schema carry.
- `support.columns` lists only columns with `excluded_count > 0`. A column not listed had every row eligible and evaluated.
- The `no_duplicate_rows` finding of `find_duplicate_rows`. A search that finds nothing gives no finding: `findings` is empty, `duplicate_group_count` is 0, and the printed fact line says which columns were compared and that no two rows are equal in them.

### Changed in records

- A capped record keeps the entries that findings and limitations point at, then the first of the others, up to 100. Field paths name a mapping entry by its key (`columns['c115'].min`) and an association pair by its two columns (`pairs['a', 'b'].value`); every path resolves in `to_dict()`.
- Lists and mappings inside a mapping argument of `parameters` are capped at 100 at any depth. A `<key>_total` appears beside them only when entries were dropped. `project` is recorded whole.
- `to_frame()` gives every row ID: the `row_ids` column of `detect_missing_sentinels`, the `invalid_row_ids` column of `validate_column_types`, and a new `row_ids` column in `profile_target`, with the rows that hold a label outside `expected_labels`. A result rebuilt from its record raises `result_table_unavailable` when it would lack them.

### Fixed

- `summarize_categorical` with a `top_n` below the level count could drop text labels that hold a NUL character.

### Moved

- `describe_dataset` lives in `agent_eda.data.structure`, which `ae.list_analyses()` reports as its module. `agent_eda.data.provenance` no longer exists. `ae.describe_dataset` is unchanged.
