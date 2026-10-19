# 01: Printed text that shows what matters

What to build:
- long lines wrap instead of being cut;
- findings come in order of assessment, and table rows in order of interest;
- `summarize_categorical` shows its levels and the case variants it flags;
- shares are written one way, and "nothing found" is expressed one way.

Read these sections of `.scratch/agent-experience/spec.md`: Printed text, and test group 1. This ticket covers user stories 1 to 6. Findings: AX-1, AN2-11, AX-3, AN1-8, AX-9, AX-18, AX-19, AX-15, AN1-10, CORE-21.

Blocked by: none

Status: done

- [x] Finding, limitation, fact and reason lines wrap with a hanging indent, and no line ends in "…". On the audit's churn data, the `sentinel_values_not_declared` limitation of `profile_missing_values` prints its whole fix.
- [x] The 40-line cap still holds and counts what it leaves out. Every printed line keeps to 120 characters.
- [x] Findings are listed violation, suspected, descriptive, passed, then unknown. In `infer_column_semantics` with 7 columns and two declarations, `missing_declared_columns` (named `declared_columns_missing` until ticket 05) is shown.
- [x] Table rows put columns with a finding or limitation first. A 120-column `summarize_numeric` whose four -9999-wrecked columns sit beyond position 10 prints those four.
- [x] `summarize_categorical` prints each column's top levels on their own lines. A case-variant finding names its variants, with masking respected.
- [x] Printed shares are fractions under their field names, and no analysis prints percentages.
- [x] Each "nothing found" is expressed the same way: `find_duplicate_rows` drops `no_duplicate_rows`, and its fact line says what was compared. The docstring, the README and the CHANGELOG say so.
- [x] "Left out" lines say why each column was left out. `infer_column_semantics`' schema fact line accounts for every declared column. An empty table's footer says it has no rows.
- [x] One golden test per text template exists, and other text tests assert structure instead of whole texts.
- [x] The finish checks pass.

## Comments

- Wrapping: finding, limitation, fact and reason lines wrap with a hanging indent (4 spaces for facts and reasons, 2 for findings, limitations and the new level lines). Only a table cell too wide for its column still ends in "...". The churn data of the audit isn't in the repository, so the test uses a frame that triggers `sentinel_values_not_declared` and checks that the printed block holds the whole reason.
- The 40-line cap: when the lines run over, the tail is cut as before and `... N more lines; see res.to_json().` counts them. The new level lines of `summarize_categorical` are the one section that gives way first: they fill the lines the findings, limitations and footer leave free, and end with `and N more not shown.`.
- Order: the printed findings are sorted by assessment (violation, suspected, descriptive, passed, unknown), stable within an assessment. The record keeps the analysis's own order, which ticket 02 deals with. `infer_column_semantics`' own finding order was left alone for that reason.
- Interest order: columns named by a finding, or pointed at by a limitation's `columns[...]` path, come first in `describe_dataset`, `summarize_numeric`, `summarize_categorical` and `infer_column_semantics`. The spec's rule is all there is: no skew or mean-median ranking. On a 120-column `summarize_numeric`, the four -9999 columns are in the print because each is constant and so has a `constant_columns` finding; a column wrecked by -9999 that has no finding or limitation still sits in DataFrame order.
- `summarize_categorical` prints `Top levels per column:` with one line per shown column, `column: 'a': 3 (share 0.6), ...`, and its table lost the `most_common_levels` column. The `case_or_space_variants` claim now names each column's groups (`'answer' ('Yes', 'YES' and 'yes '; 'No' and 'no')`, three groups per column at most, placeholders for masked columns). This changes the claim in the record too.
- Shares: a share is written `missing_share 0.125` or `(complete_row_share 0.2)` or, for levels, `(share 0.6)`; `trimmed_mean cuts trim_fraction 0.1`. The `missing_labels` claim of `profile_target` carries `(missing_share 0.125)` instead of a percentage. The candidate `reasons` of `infer_column_semantics` still say "75% of values are or parse as whole numbers": they are record text, not printed, and the spec limits the change to printed text.
- "Nothing found": `no_duplicate_rows` is gone. I kept `no_unexpected_labels` (a passed finding against declared `expected_labels`) and `no_key_found` (it names the most unique set, so it carries more than "nothing").
- "Left out" lines: `summarize_numeric` and `summarize_categorical` group the left-out columns by kind (`Left out, boolean, which counts only when columns names it: 'member'.`, `Left out, not numeric (text): 'name'.`). The kind per column travels in a private attribute of the result, not in the record, since ticket 05 reshapes `left_out_columns` into entries with reasons; a result rebuilt from its record says `Left out, not numeric: ...` without kinds.
- `infer_column_semantics`' schema line reads `declared_schema: 4 columns declared: 1 matching (ties with the top count), 1 not, 1 not inferred, 1 not in the data.` and sums to the declared columns.
- An empty table's footer reads `res.to_frame() has no rows` (`describe_dataset` on a frame with no columns said `has all 0 columns`).
- Goldens: one per text template, in `tests/test_result_text.py`: the result template, the analysis listing, the search listing and the analysis description (the last three built from hand-made records, so they don't break when an analysis is added). The text tests of `detect_missing_sentinels`, `find_candidate_keys`, `find_duplicate_rows`, `validate_column_types`, `summarize_numeric`, `summarize_categorical` and `profile_missing_values` now assert structure.
- `CHANGELOG.md` didn't exist (release ticket 04 creates it), so this ticket creates it with an `## [Unreleased]` section holding this ticket's lines. Release ticket 04 should keep that section when it adds the rest.
- README: the usage examples print the new text, the Reference is regenerated, and one sentence on the printed text was added. The README sat at 59,990 characters, so I tightened three other sentences (the masking error-message bullet, the MCP "doesn't complete" sentence and the raw-values sentence) and dropped "A test compares every analysis across two hash seeds."; it is at 59,942 characters.
- Checks: ruff, ty, pytest (3,377 tests), the two minimum-version runs, the MCP stdio smoke and the benchmark pass locally. The benchmark doesn't print text, so the new `_by_interest` code (linear in the columns) isn't in it.
