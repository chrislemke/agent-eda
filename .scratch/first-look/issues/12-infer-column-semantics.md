# 12: infer_column_semantics, with LogicalType and ColumnSpec

**What to build:** An agent calls `ae.infer_column_semantics(df)` and learns what kind of values each column holds. It gets ranked logical type candidates with the evidence behind them, plus flags for columns that need attention, such as identifier-like columns, numbers or dates stored as text, and mixed value types. That keeps the agent from treating IDs or postal codes as numbers, or dates stored as text as categories. A declared schema stays authoritative, and dates parse as ISO 8601 only, so day/month order is never guessed.

This ticket introduces `ae.LogicalType` and `ae.ColumnSpec`. validate_column_types reuses both, along with the parse rules written here. The analysis lives in `agent_eda.data.validation`.

Read these sections of `.scratch/first-look/spec.md`: the infer_column_semantics entry under The first-look analyses, and Public API layout. Covers user stories 82 to 85.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] `ae.LogicalType` is a literal with the values `boolean`, `integer`, `float`, `categorical`, `text`, `datetime` and `timedelta`. The spec left the values open, and this set was agreed when the tickets were cut. GLOSSARY.md's Logical type entry lists them.
- [ ] `ae.ColumnSpec` holds a logical type, `nullable` and an optional `datetime_format`. An invalid spec raises `InvalidArgumentError` in the package's message format, never a raw pydantic error.
- [ ] `declared_schema` is optional and maps column names to a logical type or a `ColumnSpec`.
- [ ] Per column, the result gives the candidate logical types ranked by a heuristic score between 0 and 1, with reasons; the flags `identifier_like`, `constant`, `all_missing`, `numbers_as_text`, `dates_as_text`, `non_iso_date_text`, `boolean_like` and `mixed_value_types`; and the evidence: non-missing and distinct counts, the uniqueness share, and the shares of values that parse as numbers, integers, ISO 8601 dates and booleans.
- [ ] Scores come from documented, deterministic rules, and the docstring says they are not probabilities.
- [ ] Parse checks run on distinct values weighted by their counts. Dates parse as ISO 8601 only. Other date-like text gets the `non_iso_date_text` flag and is never parsed.
- [ ] The analysis never overwrites a declared type. When the top candidate differs from it, the result carries a suspected finding.
- [ ] Tests cover every classification case in the spec:
  - postal codes with leading zeros stored as text rank a non-numeric type first
  - numeric-looking IDs are flagged `identifier_like`
  - ISO date strings rank `datetime` first
  - yes/no and 0/1 columns rank `boolean` first and carry the `boolean_like` flag
  - floats that are whole numbers apart from gaps rank `integer` first
  - free-text sentences rank `text` first
  - a few labels repeated over many rows rank `categorical` first
  - an all-distinct integer sequence is flagged `identifier_like`
- [ ] The result shows no raw data values.
- [ ] The declared statuses are `completed`, `partial`, `insufficient_data`, `not_applicable` and `failed`.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("what kind of column is this")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
