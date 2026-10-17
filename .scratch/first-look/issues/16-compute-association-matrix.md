# 16: compute_association_matrix

**What to build:** An agent calls `ae.compute_association_matrix(df)` and sees how strongly every suitable pair of numeric and categorical columns is associated, strongest first. The result records the method for each pair and computes each pair on its own complete rows. Perfect associations are flagged as possibly duplicated or derived columns, directional measures carry their direction, and skipped columns come with reasons. Numeric codes the agent lists as categorical count as categories, so codes such as 1, 2 and 3 aren't correlated as numbers. The analysis lives in `agent_eda.statistics.association`.

This is the largest analysis. With 100 columns there can be 4,950 pairs on 1M rows, so a pandas crosstab per pair can't meet the 5-second target. Factorize each column once and count on integer codes.

Read these sections of `.scratch/first-look/spec.md`: the compute_association_matrix entry under The first-look analyses, Analysis results, and test group 2 under Testing Decisions. Covers user stories 39, 58, 109 to 114 and 116.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [ ] The parameters, all recorded in the result, are:
  - `columns`
  - `categorical_columns`, numeric columns to treat as categorical
  - `numeric_method`, one of `pearson`, `spearman` and `kendall`, default `pearson`
  - `categorical_method`, one of `cramers_v` and `cramers_v_bias_corrected`, default `cramers_v_bias_corrected`
  - `max_categories`, default 50
  - `max_pairs`, default 5,000
- [ ] Numeric dtypes other than booleans count as numeric unless listed in `categorical_columns`. Booleans, strings, objects and categoricals count as categorical. Datetime and timedelta columns, all-missing columns and categorical columns with more levels than `max_categories` are skipped, each with a reason.
- [ ] Numeric pairs use the numeric method. Spearman and Kendall rank each pair's own complete rows, and Kendall is tau-b. Categorical pairs use Cramér's V, with Bergsma's 2013 bias correction by default. Categorical-numeric pairs use the correlation ratio, marked as directional from the categorical to the numeric column.
- [ ] Each pair uses its own complete rows and reports its pair count. A pair with a constant column, fewer than 3 complete rows or a single level is null with a reason. There are no p-values or intervals.
- [ ] The values match SciPy's `pearsonr`, `spearmanr`, `kendalltau` and contingency `association`, and published worked examples for bias-corrected Cramér's V and the correlation ratio, within a stated tolerance.
- [ ] The main table has one row per pair: both columns, their types, the method, the value, the pair count, the direction and any undefined reason. A second table, chosen with the `table` literal of `to_frame()`, gives the wide matrix, and the printed summary names both tables.
- [ ] The record keeps the 100 strongest pairs by absolute value, with the total.
- [ ] Findings flag perfect associations, with an absolute value of 1 within floating-point tolerance, as suspected duplicated or derived columns.
- [ ] More pairs than `max_pairs` gives `budget_exceeded` without computing anything, stating the pair count, the limit, and how to narrow `columns` or raise the limit. Fewer than two eligible columns gives `not_applicable`.
- [ ] `ae.describe_analysis()` lists `max_pairs` as a work-limit parameter.
- [ ] The result shows no raw data values.
- [ ] The declared statuses are `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable` and `failed`.
- [ ] The docstring has the seven sections in order and passes the contract tests.
- [ ] The property suite has a call recipe for it, and the properties hold.
- [ ] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [ ] `ae.search_analyses("correlation")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [ ] Its catalogue entry is marked implemented, with its differences from the catalogue noted, among them literal methods and defaults instead of required type and method mappings.
- [ ] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
