# 05: Shared finding builders and text helpers

What to build: one builder per shared finding code, the text helpers on the result base, a `TABLES` declaration per result, `profile_target`'s parallel structures collapsed, and the small repeated declarations in one place.

Read these sections of `.scratch/deepening/spec.md`: Shared builders, and test group 1. This ticket covers user stories 5 and 9. Findings: DES-16, DES-18, STD-47, STD-51, STD-54, STD-55.

Blocked by: 02, 04

Status: ready-for-agent

- [ ] `_findings.py` builds `constant_columns`, `all_missing_columns`, `mixed_value_types` and `infinite_values` for every analysis that reports them. Each analysis's documented differences are explicit arguments.
- [ ] The result base offers `shown(value, value_type, column)` and `row_id_source()`. The four copies of the quoting rule and the three copies of the row-ID fact are gone.
- [ ] Each result class declares `TABLES`, which drives the names `to_frame` accepts, the table check and both surfaces' footers.
- [ ] `profile_target` keeps one structure per finding code.
- [ ] `validate_column_types` numbers its masked values through `_masking.mask_values`, with an offset or known-values option, instead of numbering them itself (STD-47).
- [ ] `summarize_numeric`'s defaults appear once.
- [ ] The small declarations STD-51 lists are shared.
- [ ] The baseline comparison passes unchanged.
- [ ] The finish checks pass.

## Comments
