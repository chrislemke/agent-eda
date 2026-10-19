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

- `infer_column_semantics`: `evidence.value_types` is capped at 100, with `value_types_total`, as in `summarize_categorical`.
- `rich.print([res])`, `rich.pretty.pretty_repr(res)` and devtools print the result's text, not a dump of every field. Results and discovery records share one base for their text and JSON methods.
- An integer beyond ±2^53 (9,007,199,254,740,991) is a string in every JSON record, in row IDs, labels, values and parameters, so that Node-based MCP clients, which read JSON numbers as float64, don't round IDs such as 1234567890123456789. Integers within ±9,007,199,254,740,991 stay numbers. `to_dict()`, `to_json()`, the MCP structured content and the discovery records follow the rule; the result objects keep the exact Python ints. The field descriptions and the output schemas say so.
- A capped record keeps the entries that findings and limitations point at, then the first of the others, up to 100. Field paths name a mapping entry by its key (`columns['c115'].min`) and an association pair by its two columns (`pairs['a', 'b'].value`); every path resolves in `to_dict()`.
- Lists and mappings inside a mapping argument of `parameters` are capped at 100 at any depth. A `<key>_total` appears beside them only when entries were dropped. `project` is recorded whole.
- `to_frame()` gives every row ID: the `row_ids` column of `detect_missing_sentinels`, the `invalid_row_ids` column of `validate_column_types`, and a new `row_ids` column in `profile_target`, with the rows that hold a label outside `expected_labels`. A result rebuilt from its record raises `result_table_unavailable` when it would lack them.

### Fixed

- `summarize_categorical` with a `top_n` below the level count could drop text labels that hold a NUL character.
- `infer_column_semantics` ranks candidates on exact fractions, so scores that are equal on paper tie by the documented order. A column whose categorical and text scores are both 3/5 now infers `categorical`; it inferred `text` because float arithmetic made the text score 0.6000000000000001.

### Texts that matched the code

- Corrected docstrings and field descriptions: `DetectMissingSentinelsResult.columns_without_rules` (a column without values is listed), the `insufficient_data` entries of `summarize_categorical` and `infer_column_semantics` (no selected column completed and one has no value), the `candidates` of `infer_column_semantics` (the types the values fit, leading-zero codes at score 0 included), `ValidateColumnTypesResult.to_frame` (the columns table counts the distinct invalid values; `table='invalid_values'` lists them), and the base class of records (frozen, without unknown fields or NaN and infinity; it is not strict). The findings of `detect_missing_sentinels` have their "The findings:" lead-in.
- The advice to convert a column is one wording in every analysis, from one table in `_columns`: "convert datetimes to numbers with (df['x'] - df['x'].min()).dt.total_seconds()", "convert durations to numbers with ...", "convert numbers stored as text with pd.to_numeric(...)", "convert datetimes to labels with df['x'].astype(str), or to months with ...". `summarize_numeric` now gives a fix for a datetime column; `summarize_categorical` and `profile_target` say "convert ... to labels" where they said "count its values as text" and "convert its values to labels first".
- The docs of `schema` and `ColumnSpec.datetime_format` say that locale-dependent directives such as `%b` read in the process's locale, so a process with another locale can count differently. The JSON Schema of `validate_column_types`' `schema` has `minProperties: 1`.

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
