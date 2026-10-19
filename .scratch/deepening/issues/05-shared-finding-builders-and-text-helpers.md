# 05: Shared finding builders and text helpers

What to build: one builder per shared finding code, the text helpers on the result base, a `TABLES` declaration per result, `profile_target`'s parallel structures collapsed, and the small repeated declarations in one place.

Read these sections of `.scratch/deepening/spec.md`: Shared builders, and test group 1. This ticket covers user stories 5 and 9. Findings: DES-16, DES-18, STD-47, STD-51, STD-54, STD-55.

Blocked by: 02, 04

Status: done

- [x] `_findings.py` builds `constant_columns`, `all_missing_columns`, `mixed_value_types` and `infinite_values` for every analysis that reports them. Each analysis's documented differences are explicit arguments.
- [x] The result base offers `shown(value, value_type, column)` and `row_id_source()`. The four copies of the quoting rule and the three copies of the row-ID fact are gone.
- [x] Each result class declares `TABLES`, which drives the names `to_frame` accepts, the table check and both surfaces' footers.
- [x] `profile_target` keeps one structure per finding code.
- [x] `validate_column_types` numbers its masked values through `_masking.mask_values`, with an offset or known-values option, instead of numbering them itself (STD-47).
- [x] `summarize_numeric`'s defaults appear once.
- [x] The small declarations STD-51 lists are shared.
- [x] The baseline comparison passes unchanged.
- [x] The finish checks pass.

## Comments

- Builders (`_findings.py`): `constant_columns`, `all_missing_columns`, `mixed_value_types`, `infinite_values`, with `ASSESSMENTS` as the one place the assessment of each shared code lives. The documented differences are arguments: `shown` (how the claim names the columns, for describe_dataset's positions), `verb` and `counting` (all_missing_columns: describe_dataset and infer_column_semantics say "contain", profile_missing_values says "hold" and adds ", counting declared sentinel values"), `wording` (mixed_value_types: summarize_categorical lists Python types, infer_column_semantics says "mix values of different types"). Byte equality kept those two mixed_value_types claims and the two verbs apart, so the builder takes them as arguments rather than unifying them. `infinite_values` takes an optional `claim` for profile_target, which counts one target's values instead of naming columns.
- The result base helpers are underscore names, like `_masked` beside them, so they stay off `dir(res)`: `MaskableResult._shown(value, value_type, column)`, `_row_id_source()` and `_ids_quoted()`. The quoting rule itself is `_text.shown_value`, which the public `level_label` and profile_target's claim text (`targets._shown`) use too. Row IDs quote through `label_text`, not `repr`, so they got `_ids_quoted()` instead of going through `_shown`.
- `TABLES` is a `ClassVar` mapping on `AnalysisResult`: table name to the footer clause (None for the first table, whose footer comes from its text table). The base builds `_text_other_tables()` from it, `_check_table(table)` is now a method, and a clause may use `{size}`-style fields from `_table_footer_fields()` (compute_association_matrix). The MCP text keeps leaving the other tables out, as `_text.render` already did. `tests/test_tables.py` checks TABLES against the `Literal` in each `to_frame` signature, the default, the error message and the footer.
- `profile_target`: `_FINDINGS` (code to assessment and default fields) replaces `_ASSESSMENTS`, `_FINDING_FIELDS` and the decorator's tuple, which is `tuple(_FINDINGS)`. The docstring bullets stay hand-written until ticket 07.
- `validate_column_types`: `mask_values(values, first=, known=)` takes the start number and the values that already have a placeholder. It numbers through `value_codes`, so row IDs that compare equal under pandas' factorize share a placeholder, as everywhere else. The baseline shows no change.
- summarize_numeric's defaults: profile_target's field descriptions and its text line build from `_numeric.DEFAULT_QUANTILES`, `DEFAULT_QUANTILE_METHOD` and `DEFAULT_TRIM_FRACTION`. The `summarize_numeric` docstring still states them in words, since the docstring is hand-written.
- STD-51: `_columns.any_column(default_text, default=)` replaces six rule declarations (its `accepted_text` is one wording; it only shows when a kind is rejected, which these rules never do), `_EVERY_KIND` is gone, `_result_base.entry_path` replaces four path helpers, `_status.NO_COLUMNS` replaces two copies, and `_result_base.ROW_IDS_KEPT` and `EXAMPLES_KEPT` replace the restated caps of 10 and 5 in missingness, targets and validation. The `nan_counted_as_missing` limitation was already built once, by `_numeric.stored_nan_limitations`, so nothing was left to merge. `_statistics_path` in profile_target stays a callback that ignores the name, now with `_name`, because the target is the only column. structure.py's `_NO_COLUMNS` stays: its message differs ("to search").
- Not done from STD-54: the status_reason-to-code mapping in three `to_frame` methods, and merging `SummarizeCategoricalLevel` with `ProfileTargetClass`. The ticket's boxes don't name them, and a merged level model would change a record's schema.
- Checks: ruff, ruff format, ty, `uv run pytest` (3,900 passed), both minimum-version runs (3,350 and 3,900 passed), the MCP stdio smoke and the benchmark (14 passed) all pass. The baseline comparison passes unchanged on pandas 3 and, in the minimum run, pandas 2. README.md needed no change; its Reference is current.
