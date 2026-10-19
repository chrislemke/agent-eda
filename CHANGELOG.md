# Changelog

All notable changes to agent-eda. Add a line under Unreleased for every change a caller can see.

## [Unreleased]

### Changed

- Printed text wraps long finding, limitation, fact and reason lines with a hanging indent instead of cutting them with "...". The 40-line cap and the 120-character width still hold, and the text counts what it leaves out.
- Printed findings are ordered violation, suspected, descriptive, passed, unknown, so a violation is never hidden behind suspected findings. The record keeps the analysis's own order.
- Printed tables of columns list the columns with a finding or limitation first, then the rest in DataFrame order. `profile_missing_values` and `validate_column_types` keep their own sorting.
- `summarize_categorical` prints each column's top levels on their own lines under the table, and its `case_or_space_variants` finding names the variants of each column (placeholders for masked columns).
- Printed text writes every share as a fraction under its field name, such as `missing_share 0.0598` and `trim_fraction 0.1`, and no longer prints percentages. The `missing_labels` finding of `profile_target` reads `(missing_share 0.125)`.
- The "Left out" lines of `summarize_numeric` and `summarize_categorical` say why each column was left out, and `summarize_numeric` no longer lists booleans as "not numeric".
- The `declared_schema` line of `infer_column_semantics` accounts for every declared column: matching, not matching, not inferred and not in the data.
- The footer of an empty table says it has no rows.

### Removed

- The `no_duplicate_rows` finding of `find_duplicate_rows`. A search that finds nothing gives no finding: `findings` is empty, `duplicate_group_count` is 0, and the printed fact line says which columns were compared and that no two rows are equal in them.
