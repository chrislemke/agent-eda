# 02: summarize_numeric and the shared column rules

**What to build:** An agent calls `ae.summarize_numeric(df)` and gets, for each numeric column, the counts and statistics the spec lists, each with its definition stated and checked against SciPy and NumPy. A statistic that doesn't exist for a column comes back as null with a reason, never as NaN.

As the first analysis that takes `columns`, it also sets the rules every later multi-column analysis reuses: how `columns` is resolved, per-column statuses and the overall status, numerical warnings recorded as limitations, and caller-mistake errors that say what's wrong, what's allowed and how to fix it. The analysis lives in `agent_eda.statistics.descriptive` and is written NumPy-first, per ADR 0002. Ticket 03's benchmark checks the 5-second target.

Read these sections of `.scratch/first-look/spec.md`: Calling convention, Analysis results, Statuses, Errors, Warnings, the summarize_numeric entry under The first-look analyses, and test groups 1, 2 and 6 under Testing Decisions. Covers user stories 27 to 31, 41, 55 to 57, 60, 61, 64 to 66, 95 to 97 and 117.

**Blocked by:** 01-describe-dataset-end-to-end

**Status:** ready-for-agent

Column rules and errors:

- [ ] `columns=None` selects every numeric column except booleans, in DataFrame order. An explicit list keeps the caller's order. An empty list, a name listed twice or a column the analysis can't handle raises before any computation starts.
- [ ] Every parameter that takes column names ends in `column` or `columns`.
- [ ] An unknown column raises `ColumnNotFoundError`, a `LookupError`. The message lists the available columns and the closest matches, with case-insensitive matches first.
- [ ] A text, categorical or datetime column requested by name raises `ColumnTypeError`, a `TypeError`, that names the column's dtype. Until summarize_categorical exists, the message names the dtypes summarize_numeric accepts, and ticket 06 adds the pointer to summarize_categorical.
- [ ] A boolean column requested by name is read as 0 and 1, and a limitation says so.
- [ ] An invalid `quantile_method` raises `InvalidArgumentError` listing the allowed values and the closest one. A `trim_fraction` outside the range from 0 up to but excluding 0.5, or a quantile level outside 0 to 1, raises with the valid range. All argument checks run before any computation.
- [ ] summarize_numeric, like every later analysis except describe_dataset, raises `InvalidArgumentError` with a renaming hint when the DataFrame has duplicated column labels, labels whose string forms collide, or MultiIndex columns.
- [ ] No raw pydantic validation error reaches the caller. Argument validation re-raises it as `InvalidArgumentError` in the package's message format.
- [ ] Tests cover each error's code, its exception class and its message content: the available columns, the closest match and the allowed values.

Statistics:

- [ ] Per column, the result gives the count; the missing, infinite, zero and negative counts; the mean; the standard deviation with ddof 1; the minimum and maximum; the quantiles; the median; the trimmed mean; the raw MAD and the MAD scaled by 1.4826 for consistency with the normal distribution; the bias-corrected skewness; and the bias-corrected excess kurtosis. Moments use finite values only.
- [ ] `quantiles` defaults to 0.01, 0.05, 0.25, 0.5, 0.75, 0.95 and 0.99. `quantile_method` takes NumPy's quantile methods and defaults to linear. `trim_fraction` defaults to 0.1, cut from each end. The result records the values used and states each statistic's definition.
- [ ] The numbers match SciPy's `skew` and `kurtosis` with `bias=False`, `trim_mean` and `median_abs_deviation`, and NumPy's quantiles for every quantile method, within a stated tolerance. Counts match pandas.
- [ ] The standard deviation with fewer than 2 values, the skewness with fewer than 3 values or zero variance, and the kurtosis with fewer than 4 values or zero variance are null, and a limitation names the field path and the reason. NaN and infinity never appear in the record.
- [ ] Integers beyond 2^53 add a float-precision limitation.
- [ ] Nullable integer and float columns and pyarrow-backed numeric columns give the same numbers as their NumPy equivalents.
- [ ] Findings cover constant columns and infinite values.

Statuses and warnings:

- [ ] The result keeps one entry per column, keyed by column name, with its own status and, when it didn't complete, the reason. The overall status is `completed` when every selected column completed and `partial` when some did. When none did, it is the status the columns share, or, if they differ, the first of `failed`, `insufficient_data` and `not_applicable` that occurs.
- [ ] The declared statuses are `completed`, `partial`, `insufficient_data`, `not_applicable` and `failed`. An empty DataFrame, or columns without values, give `insufficient_data`. A DataFrame without numeric columns gives `not_applicable`.
- [ ] Numerical RuntimeWarnings raised inside the analysis, for division by zero, invalid values, overflow or precision loss, become limitations next to the numbers they affect. No warning reaches the caller.
- [ ] Callers never see a DeprecationWarning or FutureWarning from inside an analysis, yet the test suite still fails when one is raised.

Done:

- [ ] The docstring has the seven sections and passes the contract test. describe_dataset's `When to use:` and `See Also:` now point to summarize_numeric.
- [ ] The catalogue entry is marked implemented, with its differences noted, among them the optional `columns`, the added `quantiles` and `quantile_method`, and weights deferred.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.
