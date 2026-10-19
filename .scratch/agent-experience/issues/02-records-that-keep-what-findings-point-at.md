# 02: Records that keep what findings point at

What to build: JSON records in which every field path resolves, levels bounded by `top_n`, compact support, no constant prose, capped nested parameters, and row IDs complete in `to_frame()`. ADR 0006's measured sizes are updated.

Read these sections of `.scratch/agent-experience/spec.md`: Records (up to the integers item), and test group 2. This ticket covers user stories 7 to 9. Findings: AX-2, AN2-8, AX-26, MCP-10, CORE-12, SPEC-15, STD-18, and STD's questionable decision on `to_frame()` row IDs.

Blocked by: none

Status: done

- [x] When a capped list drops entries, the record keeps those a finding or limitation points at. On a 120-column frame whose constant column sits beyond position 100, `summarize_numeric`'s finding path `columns['c115'].min` resolves in `json.loads(res.to_json())`.
- [x] Field paths into keyed lists name entries by key. For example, association pairs are named by their two columns.
- [x] `summarize_categorical`'s record keeps `top_n` levels per column, with `levels_total`. `to_frame()` and `table_path` keep every level.
- [x] `support.columns` lists only columns where something was excluded.
- [x] `definitions`, `share_denominator`, `level_order` and `missing_values_compare_equal` leave the record, and their meaning lives in field descriptions.
- [x] Nested lists in `parameters` are capped with totals, and `project` has no `project_total`.
- [x] `to_frame()` gives every row ID in `detect_missing_sentinels`, `validate_column_types` and `profile_target`.
- [x] `to_dict()`'s docstring matches what the record keeps.
- [x] ADR 0006 and the README carry before and after sizes for the first look on a 1,000 × 30 mixed CSV and for a wide `summarize_categorical`, which measured 383 KB on 1,000 × 100.
- [x] The finish checks pass.

## Comments

- **Path syntax.** A path into a mapping names the entry by its key (`columns['c115'].min`). A path into the association pairs names them by their two columns, `pairs['a', 'b'].value`. Other lists keep positions (`columns[2]` in `describe_dataset`, which has repeated labels; `candidates[3]`, `keys[0]`, `duplicate_groups[N]`). When the record drops entries before a pointed-at one, the record's findings and limitations carry the entry's position in the record's list, so every path resolves in `to_dict()`; the result object keeps the object positions. Tests resolve paths with `ast`, in `tests/test_record_paths.py`, on a 130-column frame for every analysis.
- **What a capped record keeps.** Entries that a finding or limitation points at (counting only the first 100 findings, limitations and paths per finding, which the record keeps), then the first of the others in order, 100 in all. A record can exceed 100 entries only when more than 100 are pointed at.
- **`summarize_categorical`.** `levels` now holds the first `top_n` levels, on the object and in the record (`levels_total` = `cardinality` in the record), so `most_common_levels` was a duplicate and is gone. The printed text calls its column and fact `levels`. `to_frame()` and `table_path` keep every level. Ticket 01 (printed text) touches the same lines of text: it needs the `levels` header, not `most_common_levels`.
- **Removed fields.** `definitions`, `share_denominator`, `level_order` (in `summarize_categorical` and `profile_target`) and `missing_values_compare_equal` are gone from the result, not only from the record. Their meaning moved into field descriptions (`describe_analysis`, the output schema): the statistics' definitions were already the descriptions of the entry fields.
- **Support.** `Support.columns` keeps only entries with `excluded_count > 0`, on the object and in the record. A column not listed had every row eligible and evaluated (every analysis sets `eligible_count` to `row_count`); the field description says so.
- **Parameters.** Lists and mappings inside a mapping argument are capped at 100 down to any depth. They get `<key>_total` beside them only when entries were dropped (a mapping of column specs gets no noise); the top-level `<name>_total` stays. A `<key>_total` that collides with another key gets underscores added. `project` is recorded whole.
- **Row IDs in `to_frame()`.** `detect_missing_sentinels` gives every candidate's full row-ID list in its `row_ids` column; `validate_column_types` does in `invalid_row_ids`; `profile_target` gained a `row_ids` column, the IDs of every row that holds a label outside `expected_labels` and null for classes. Results keep the IDs as pandas labels (compact) in a private attribute, so a result rebuilt from its record raises `result_table_unavailable` when it would lack IDs, like the other tables. With a sensitive `id_column` the shown IDs keep their placeholders and the rest continue the numbering. The `ambiguous_row_ids` check in these three now looks at every row, not only the shown ones.
- **Bug found and fixed on the way.** `ordered(levels, limit)` dropped text labels when a label held a NUL character (numpy coerced the comparison key), so `top_n` smaller than the level count could give no levels. The old fixed limit of 100 hid it. A regression test is in `tests/test_summarize_categorical.py`.
- **Sizes** (`len(res.to_json())`, seeded random frames, script in the session scratchpad; the nine first-look analyses on 1,000 x 30: numbers, integers, 8 categorical, 3 free-text, dates as text, a Yes/yes/No column): first look 162,231 -> 104,069 characters. `summarize_categorical` on 1,000 x 100 with 40 text columns: 285,373 -> 40,534 (263,979 -> 19,140 with `top_n=1`); with 100 text columns: 711,079 -> 99,403 (657,529 -> 45,853). The audit's 383 KB used other data. The first look is still dominated by `infer_column_semantics` (27 KB) and `compute_association_matrix` (27 KB); this ticket does not shrink them. ADR 0006 and the README carry these numbers.
- **CHANGELOG.** `CHANGELOG.md` doesn't exist on this branch (release ticket 04 creates it), so nothing was added. It should list: `most_common_levels`, `definitions`, `share_denominator`, `level_order` and `missing_values_compare_equal` removed; `levels` holds `top_n` levels; `support.columns` and `parameters` shape; pair paths named by columns; `row_ids` in `profile_target`'s `to_frame()`.
- **Checks.** Run locally: ruff, ruff format, ty, the generator `--check`, pytest (3,104 passed), both minimum-version runs (2,602 passed without the mcp extra; 3,104 with it), the MCP stdio smoke test and the benchmark (all under 2 s per analysis). The README is at 59,873 of 60,000 characters; later tickets that add README text must tighten other prose first.
