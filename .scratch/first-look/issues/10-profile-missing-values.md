# 10: profile_missing_values

**What to build:** An agent calls `ae.profile_missing_values(df)`, optionally declaring sentinel values per column, and learns the data's true completeness. It sees native missing values and declared sentinel values per column, combined without double counting, with shares over all rows. It also sees how many values each row is missing, which shows whether gaps cluster in some rows.

The analysis lives in `agent_eda.data.missingness`. detect_missing_sentinels reuses its handling of declared sentinel values.

Read these sections of `.scratch/first-look/spec.md`: the profile_missing_values entry under The first-look analyses, and Analysis results. Covers user stories 39 and 90 to 92.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] `sentinel_values` maps column names to declared sentinel values and has no default.
- [ ] Per column, the result gives the native missing count, which covers None, NaN, NaT and pd.NA; the sentinel count; the combined missing count without double counting; and the missing share over all rows.
- [ ] For the dataset, the result gives the number of complete rows and the distribution of how many selected values each row is missing.
- [ ] Empty strings count as missing only when declared as sentinel values.
- [ ] Without `sentinel_values`, the result records that none were declared, so a sentinel count of zero can't be mistaken for a check that ran.
- [ ] A declared sentinel value that can never match its column's type raises `InvalidArgumentError` naming the likely intended value. For example, "-999" given as text for an integer column suggests -999.
- [ ] The result has two tables. `to_frame()` returns the per-column table and takes a `table` literal for the per-row distribution, and the printed summary names both tables.
- [ ] Findings cover all-missing columns and declared sentinel values that are present.
- [ ] The declared statuses are `completed`, `insufficient_data`, `not_applicable` and `failed`. With no rows, the status is `insufficient_data`, the counts are zero and the shares are null.
- [ ] Native missing counts match pandas `isna()` on every accepted dtype, including pyarrow-backed and pandas 3 string columns.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("missing values")` and `ae.search_analyses("nulls per column")` return it first. Both rows join the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
