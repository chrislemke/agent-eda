# 06: summarize_categorical

**What to build:** An agent calls `ae.summarize_categorical(df)` and sees, for each categorical column, which values occur and how often, with the denominator stated. It also learns about labels that differ only by case or surrounding spaces, categories that never occur and columns that mix value types, so it can clean them before modelling.

The analysis lives in `agent_eda.statistics.descriptive`. Masking comes in ticket 08, which adds `project` and `mask_sensitive` to this analysis.

Read these sections of `.scratch/first-look/spec.md`: the summarize_categorical entry under The first-look analyses, and Analysis results. Covers user stories 98 to 100.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] `columns=None` selects object, string, categorical and boolean columns. A numeric column requested by name is summarized by its values.
- [ ] Per column, the result gives the non-missing and missing counts, the cardinality, each level with its count and share, the most common levels, the number of levels left out of the display, the number of levels seen once, the unused categories of a categorical dtype, and whether values of different Python types are mixed.
- [ ] Shares use the non-missing count as their denominator, and the result states it.
- [ ] Levels are ordered by count, descending, then by label. Labels of mixed types follow a documented order that never depends on hash order. Labels are kept exactly as stored.
- [ ] `top_n`, default 10, limits only the display, and the result records it. The record caps levels at 100 with the total, and `to_frame()` returns every level.
- [ ] Labels that differ only by case or surrounding spaces, such as "Yes", "yes " and "YES", give a suspected finding. Unused categories give a descriptive finding and mixed value types a suspected one.
- [ ] The declared statuses are `completed`, `partial`, `insufficient_data`, `not_applicable` and `failed`, and they follow the per-column rules from ticket 02.
- [ ] Counts and shares match pandas `value_counts` for object, pandas 3 string, categorical, boolean and pyarrow-backed columns.
- [ ] summarize_numeric's `ColumnTypeError` for a text or categorical column now names summarize_categorical as the analysis that fits.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("value counts")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
