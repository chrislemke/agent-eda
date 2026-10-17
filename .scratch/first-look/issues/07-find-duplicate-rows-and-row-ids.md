# 07: find_duplicate_rows and row IDs

**What to build:** An agent calls `ae.find_duplicate_rows(df)` and learns whether exact duplicate rows exist, how many groups and excess copies there are, and which rows they are. Row IDs are index labels, so `df.loc[ids]` shows the rows straight away, or values of `id_column=` when the index is a plain row number. No rows are removed. The agent decides what to do with them.

This ticket also sets the row ID rules that validate_column_types, detect_missing_sentinels and profile_target reuse. The analysis lives in `agent_eda.data.duplicates`. Masking comes in ticket 08.

Read these sections of `.scratch/first-look/spec.md`: the find_duplicate_rows entry under The first-look analyses, Row IDs, and Analysis results. Covers user stories 50 to 52, 101 and 102.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] `columns` defaults to every column except `id_column`. An unknown `id_column` raises `ColumnNotFoundError` with suggestions, like any other column argument.
- [ ] The result gives the duplicate groups, each with its size and row IDs, ordered by first occurrence; the group count; the rows in duplicate groups; the excess copies, which are the rows in groups minus the groups; the duplicate share, which is the excess copies over all rows; and the compared columns.
- [ ] Missing values compare equal, as in pandas, and the result says so.
- [ ] `to_frame()` lists every row in a duplicate group with its group number. The record caps groups at 100 with the total.
- [ ] Row IDs are index labels by default and values of `id_column` when one is given. Each list of row IDs in the record holds at most 10 IDs plus the total count.
- [ ] Row IDs convert deterministically for JSON: timestamps to ISO 8601 strings, MultiIndex tuples to lists and NumPy scalars to Python numbers.
- [ ] When the index or the ID column has duplicates, the result carries an `ambiguous_row_ids` limitation that recommends a unique `id_column`.
- [ ] Duplicates give a suspected finding. When there are none, a descriptive finding says so.
- [ ] The declared statuses are `completed`, `insufficient_data`, `not_applicable` and `failed`.
- [ ] Group and row counts match pandas `duplicated(keep=False)` on the same columns.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("duplicate rows")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
