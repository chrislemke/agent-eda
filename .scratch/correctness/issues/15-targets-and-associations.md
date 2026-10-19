# 15: Targets and associations

What to build:
- **`compute_association_matrix`:** a bias-adjusted correlation ratio by default for mixed pairs; its own reason when each row is its own level; a pair budget that accounts for rank methods; pair values that don't depend on the other columns.
- **`profile_target`:** findings for missing declared classes, a regression task on a two-value target, and labels that differ only by case or spaces; `expected_labels` working with datetime and period categories.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (`compute_association_matrix` and `profile_target`), and test group 10. This ticket covers user stories 9 and 10. Findings: AN2-5, AN2-6, AX-14, AX-16, STD-13, STD-44, AN2-12, AN2 questionable decision 3 (rank-method cost), SPEC questionable decision (trivial perfect associations).

Blocked by: 01

Status: done

- [x] The default mixed-pair method is a bias-adjusted correlation ratio: the square root of epsilon squared, clipped at 0. The plain correlation ratio stays available as a method.
- [x] On 20 runs of 60 rows, a 40-level noise column against a noise column scores near 0 under the default, not 0.74. The README's example with `email` no longer prints perfect associations.
- [x] The bias-adjusted ratio matches a hand-computed epsilon squared within 1e-12.
- [x] A pair left null because each row is its own level gets its own reason and wording.
- [x] The pair budget accounts for the cost of rank methods. On 100 numeric columns of 1,000,000 rows with `numeric_method="kendall"`, the call returns `budget_exceeded` with a reason, instead of running for minutes.
- [x] A pair's value doesn't change with the other columns in the call, checked to the last digit. Or, if that can't hold, the docs say so and the Comments record why.
- [x] The `budget_exceeded` footer doesn't offer a matrix when no pair was computed.
- [x] `profile_target` with `expected_labels=["setosa", "versicolor", "virginica"]`, on data holding only the first two, reports a violation `missing_expected_classes`. "virginica" appears at count 0 in the class table and counts in the minority share and the imbalance ratio.
- [x] `profile_target`'s two new suspected findings:
  - `task="regression"` on a 0/1 column;
  - multiclass labels `"Yes", "yes", " yes", "NO", "No"`, flagged as labels that differ only by case or surrounding spaces.
- [x] `expected_labels` works with datetime and period category targets, and its error no longer speaks of sentinel values (STD-13).
- [x] The new finding codes are declared, documented and listed in the README.
- [x] The finish checks pass.

## Comments

**`compute_association_matrix`**

- The plain correlation ratio is a method of a new option, `mixed_method`: `'correlation_ratio_bias_corrected'` (default) or `'correlation_ratio'`, named like `categorical_method`. A pair's `method` says which one it used. Epsilon squared is Kelley's `(SSB - (k - 1) * SSW / (N - k)) / SST`, clipped at 0, then square-rooted; the batched and the pair-by-pair paths share `ratio_value`, so they can't disagree. The test reference computes it by hand with NumPy and agrees to 1e-12.
- On 20 runs of 60 rows with 40 levels the plain ratio averages above 0.7 and the default below 0.3 (half the runs give exactly 0, as epsilon squared of noise is negative half the time).
- The README example can't stop printing perfect associations through the formula alone: `email` was unique except for the exact duplicate, so every numeric column has a within-email sum of squares of 0, and epsilon squared is exactly 1 whatever the correction. I changed the sample data instead (customer 107 now shares `ann@example.com` with 101, a shared mailbox) and rewrote the sentences around the output. `email` with `age` still scores 0.9993, because the -999 dominates the total variance; nothing is perfect, so no finding.
- A pair nulled by `one_row_per_level` gets its own limitation code, `one_row_per_level` (new in `LimitationCode`), with its own wording. `too_few_values` now names only `too_few_rows` pairs.
- Rank-method cost: `max_pairs` counts a numeric pair under `spearman` or `kendall` once per 10,000 rows (rounded up, at least 1); `pearson` and every other pair count 1. The reason says so and offers `numeric_method='pearson'`. Measured: 100 numeric columns of 1,000,000 rows return `budget_exceeded` in 0.11 s for both methods (`tests/test_benchmark.py`, opt-in, since the frame needs about 1.2 GB). The check still runs after the columns are read, as before.
- Pair values and the other columns (STD-44): they do not hold to the last digit, and the docstring says so. The sums of all pairs run in batches of rows whose size depends on the number of columns, and BLAS adds in an order that depends on the matrix shape, so the same pair differs by up to about 2e-17 absolute (1e-15 relative) between a call with 2 columns and one with 30 more. Making them equal would mean one pair at a time, which costs the batching the 5-second target needs. The test bounds the difference by 1e-12 across column sets.
- The footer offers the matrix only when `pairs` isn't empty. The text still says `res.to_frame() has all 0 pairs` for `budget_exceeded`, as for any empty table.

**`profile_target`**

- Declared classes without rows are appended to the levels at count 0 (`Levels.with_empty`), so classes, `to_frame()`, masking placeholders, rare classes and the minority share follow from the level machinery. `class_count` counts them; `single_class` and `more_than_two_classes` count classes with rows only. With no valid label at all nothing is appended (there are no shares to divide), and the status stays `insufficient_data`.
- The imbalance ratio with an empty class would be infinite: it is null, with a `too_few_values` limitation on `imbalance_ratio`, since strict JSON can't hold infinity. The text says `imbalance ratio null, a class has no rows`.
- New finding codes: `missing_expected_classes` (violation), `two_value_target` (suspected; exactly two distinct finite values, not flagged for columns of integers beyond 2^53) and `case_or_space_variants` (suspected; the code `summarize_categorical` uses, found with the same `variant_groups`, over the labels the data holds, outside expected_labels too). Claims quote labels as the classes show them: placeholders when masked.
- `expected_labels` takes `datetime`, `date` and `pd.Period` besides text, numbers and booleans. A period has no JSON Schema, so `SkipJsonSchema` leaves it out of the MCP input schema. Over MCP the strings of `expected_labels` are read as ISO 8601 for a target of dates or datetimes, as `sentinel_values` are; periods can't be declared over MCP. A plain period or datetime column is still not a classification target; categories and object columns of them are. The label-type error is now `worded`, names the target's own types for periods (`Period values`) and no longer says sentinel. NaT is refused like NaN.
- Existing tests changed because of the new findings and classes: one test's data used `No` next to `no`, the masking test gains the `measles` class at count 0, the infinite-target test gets a third finite value, and `tests/test_mcp_arguments.py` fills the required arguments of `profile_target` for its format examples.

**Not run locally**: the GitHub-only checks (CI matrix on other platforms).
