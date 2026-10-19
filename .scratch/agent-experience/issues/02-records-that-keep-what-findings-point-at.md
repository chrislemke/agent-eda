# 02: Records that keep what findings point at

What to build: JSON records in which every field path resolves, levels bounded by `top_n`, compact support, no constant prose, capped nested parameters, and row IDs complete in `to_frame()`. ADR 0006's measured sizes are updated.

Read these sections of `.scratch/agent-experience/spec.md`: Records (up to the integers item), and test group 2. This ticket covers user stories 7 to 9. Findings: AX-2, AN2-8, AX-26, MCP-10, CORE-12, SPEC-15, STD-18, and STD's questionable decision on `to_frame()` row IDs.

Blocked by: none

Status: ready-for-agent

- [ ] When a capped list drops entries, the record keeps those a finding or limitation points at. On a 120-column frame whose constant column sits beyond position 100, `summarize_numeric`'s finding path `columns['c115'].min` resolves in `json.loads(res.to_json())`.
- [ ] Field paths into keyed lists name entries by key. For example, association pairs are named by their two columns.
- [ ] `summarize_categorical`'s record keeps `top_n` levels per column, with `levels_total`. `to_frame()` and `table_path` keep every level.
- [ ] `support.columns` lists only columns where something was excluded.
- [ ] `definitions`, `share_denominator`, `level_order` and `missing_values_compare_equal` leave the record, and their meaning lives in field descriptions.
- [ ] Nested lists in `parameters` are capped with totals, and `project` has no `project_total`.
- [ ] `to_frame()` gives every row ID in `detect_missing_sentinels`, `validate_column_types` and `profile_target`.
- [ ] `to_dict()`'s docstring matches what the record keeps.
- [ ] ADR 0006 and the README carry before and after sizes for the first look on a 1,000 × 30 mixed CSV and for a wide `summarize_categorical`, which measured 383 KB on 1,000 × 100.
- [ ] The finish checks pass.

## Comments
