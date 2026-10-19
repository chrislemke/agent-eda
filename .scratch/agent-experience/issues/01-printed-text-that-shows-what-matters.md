# 01: Printed text that shows what matters

What to build:
- long lines wrap instead of being cut;
- findings come in order of assessment, and table rows in order of interest;
- `summarize_categorical` shows its levels and the case variants it flags;
- shares are written one way, and "nothing found" is expressed one way.

Read these sections of `.scratch/agent-experience/spec.md`: Printed text, and test group 1. This ticket covers user stories 1 to 6. Findings: AX-1, AN2-11, AX-3, AN1-8, AX-9, AX-18, AX-19, AX-15, AN1-10, CORE-21.

Blocked by: none

Status: ready-for-agent

- [ ] Finding, limitation, fact and reason lines wrap with a hanging indent, and no line ends in "…". On the audit's churn data, the `sentinel_values_not_declared` limitation of `profile_missing_values` prints its whole fix.
- [ ] The 40-line cap still holds and counts what it leaves out. Every printed line keeps to 120 characters.
- [ ] Findings are listed violation, suspected, descriptive, passed, then unknown. In `infer_column_semantics` with 7 columns and two declarations, `missing_declared_columns` (named `declared_columns_missing` until ticket 05) is shown.
- [ ] Table rows put columns with a finding or limitation first. A 120-column `summarize_numeric` whose four -9999-wrecked columns sit beyond position 10 prints those four.
- [ ] `summarize_categorical` prints each column's top levels on their own lines. A case-variant finding names its variants, with masking respected.
- [ ] Printed shares are fractions under their field names, and no analysis prints percentages.
- [ ] Each "nothing found" is expressed the same way: `find_duplicate_rows` drops `no_duplicate_rows`, and its fact line says what was compared. The docstring, the README and the CHANGELOG say so.
- [ ] "Left out" lines say why each column was left out. `infer_column_semantics`' schema fact line accounts for every declared column. An empty table's footer says it has no rows.
- [ ] One golden test per text template exists, and other text tests assert structure instead of whole texts.
- [ ] The finish checks pass.

## Comments
