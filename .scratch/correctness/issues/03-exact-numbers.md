# 03: Exact numbers

What to build: one rule for when a column is constant, decided on the stored values; an exact standard deviation for very small values; a precision limitation for large decimals; and NaN stored as a value counted as missing everywhere.

Read these sections of `.scratch/correctness/spec.md`: Exact numbers, and test group 3. This ticket covers user stories 2 to 5. Findings: STD-6, AN2-2, AN2-9, STD-14, DES-16 (constancy).

Blocked by: 01

Status: done

- [x] `describe_dataset`, `summarize_numeric`, `infer_column_semantics` and `profile_target` decide constancy through one shared rule.
- [x] On `pd.DataFrame({"id": [2**54, 2**54 + 1, 2**54 + 2]})`, `summarize_numeric` reports no `constant_columns`, and `profile_target(df, target_column="id", task="regression")` reports no `constant_target`. `describe_dataset` agrees.
- [x] `summarize_numeric(pd.DataFrame({"x": [1e-200, 2e-200, 4e-200, 8e-200]}))` gives a std within 1e-12 relative error of 3.0956959368344516e-200, the exact value computed with fractions.
- [x] The same tolerance holds for 1,000 normal values scaled by 1e-170 and for values near 1e-160. The fix also applies to `profile_target`'s regression summary.
- [x] A `decimal128(38, 1)` column holding 2**60+1, 2**60+3, 2**60+5 and 0.1 gets a `float_precision` limitation naming it, in `summarize_numeric`, `profile_target` and `compute_association_matrix`.
- [x] NaN stored as a value is missing in every analysis, such as NaN in a `double[pyarrow]` column. A limitation names the columns where this differs from pandas' `isna()`.
- [x] `[1.0, 1.0, NaN]` in a `double[pyarrow]` column gets the same constancy verdict in every analysis.
- [x] The docstrings and field descriptions that describe constancy and missing values say this.
- [x] The finish checks pass.

## Comments

- Constancy rule, settled here: `_values.is_constant(non_missing, distinct)` says two or more non-missing values and one distinct value. `describe_dataset` and `infer_column_semantics` pass the distinct count they already take from `value_codes`. `_numeric.read_numeric` decides it for `summarize_numeric` and `profile_target` on the stored values: integers and Arrow decimals compare their exact minimum and maximum, floats compare the stored float64 values with NaN dropped, and one infinity of one sign is constant. `summarize_values` now takes the `NumericColumn` instead of the bare array. The `zero_variance` null reason still follows the float64 values, so distinct integers beyond 2^53 that round to one float get no skewness and kurtosis, next to the `float_precision` limitation.
- Stored NaN, settled here: `_values.value_codes` reads NaN that a pyarrow-backed or nullable float column stores as a value as a missing value, so every analysis that counts levels, groups, classes or keys (describe_dataset, summarize_categorical, profile_target's classes, find_duplicate_rows, find_candidate_keys, infer_column_semantics, compute_association_matrix) agrees without per-analysis code. The analyses that count missing values from `isna()` (describe_dataset, profile_missing_values, detect_missing_sentinels) add the stored NaN there. Every analysis that reads a column where this differs from `isna()` returns the `nan_counted_as_missing` limitation, built by `_numeric.stored_nan_limitations`; the three existing builders use it, with their texts unchanged.
- `profile_missing_values` changed: `native_missing_count` now counts stored NaN (the field says None, NaN, NaT and pd.NA), and the limitation is `nan_counted_as_missing`. The code `nan_not_counted_as_missing` is gone from `LimitationCode`, since nothing returns it any more. That removes a public code; flag it if a deprecation period is wanted instead.
- `float_precision` now also covers Arrow decimal columns whose magnitude exceeds 2^53, and its reason reads "integers or decimals beyond ±2^53" (`test_summarize_numeric.py` changed to match).
- Small values: the moments are computed on deviations divided by a power of two near the largest deviation, which is exact, so std, skewness and excess kurtosis keep full precision for values near 1e-200 and for deviations whose fourth powers would underflow. Terms that still underflow are far below the largest square and don't change the sum, so no limitation is added for them. For ordinary data the results are bit-identical. The existing overflow retry stays.
- Tests: `tests/test_exact_numbers.py`. The reference std comes from `fractions` and `decimal`. `pytest.approx(..., rel=...)` keeps an absolute tolerance of 1e-12 unless `abs=0` is passed, which would have made the tiny-value tests pass on 0.0; the tests pass `abs=0`. Three older tests changed for the new rule (`test_profile_missing_values.py`, two in `test_summarize_categorical.py`, `test_describe_analysis.py` for the shortened finding meaning).
- Benchmark (`uv run pytest -m benchmark`, latest versions, Darwin arm64): every analysis under 2 s; summarize_numeric 1.38 s, describe_dataset 1.06 s.
- Finish checks run locally: ruff check and format, ty, full pytest (2,816 passed), both minimum-version runs, the MCP stdio smoke, the Reference check and the benchmark. Nothing was left for GitHub.
