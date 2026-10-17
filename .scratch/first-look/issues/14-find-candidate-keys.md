# 14: find_candidate_keys

**What to build:** An agent calls `ae.find_candidate_keys(df)` and learns which columns, alone or combined, uniquely identify rows, with missing and duplicate counts for each candidate. When no exact key exists, the closest candidates expose a key broken by a few duplicates. The result says whether the search was exhaustive, so a cut-off search is never mistaken for proof that no key exists. The analysis lives in `agent_eda.data.structure`.

The 5-second target is the hard part. With 100 columns and the default `max_key_size` of 2, the search covers up to 5,050 column sets on 1M rows, far too many to check one by one on all rows. Rejecting candidates on an evenly spaced subset of rows, as the spec allows, has to do most of the work.

Read these sections of `.scratch/first-look/spec.md`: the find_candidate_keys entry under The first-look analyses, and Calling convention. Covers user stories 58, 103 to 105 and 116.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] `max_key_size` defaults to 2 and `max_combinations` to 10,000, and the result records both.
- [ ] The search covers single columns, then combinations in increasing size, and keeps only minimal keys. It skips supersets of any key it has found.
- [ ] Each candidate reports its columns, distinct key count, duplicate rows, rows with a missing key value, uniqueness share and whether it is minimal.
- [ ] Keys with missing values are listed separately and are not primary keys.
- [ ] Without an exact key, the result lists the 10 most unique column sets.
- [ ] The result says whether the search was exhaustive. Hitting `max_combinations` stops the search with `budget_exceeded`, keeps what was found, marks it non-exhaustive, and states the estimate, the limit and how to raise the limit or narrow `columns`.
- [ ] A deterministic, evenly spaced subset of rows may reject a candidate, because a duplicate in the subset proves it isn't unique. Uniqueness is only ever confirmed on all rows. A test uses a column set that is unique on the subset but not on all rows.
- [ ] Distinct and duplicate counts match pandas on the same columns.
- [ ] `ae.describe_analysis()` lists `max_combinations` as a work-limit parameter.
- [ ] The result shows no raw data values.
- [ ] The declared statuses are `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable` and `failed`.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("primary key")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted, among them `max_key_size` defaulting to 2.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
