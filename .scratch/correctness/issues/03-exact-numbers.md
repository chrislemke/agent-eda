# 03: Exact numbers

What to build: one rule for when a column is constant, decided on the stored values; an exact standard deviation for very small values; a precision limitation for large decimals; and NaN stored as a value counted as missing everywhere.

Read these sections of `.scratch/correctness/spec.md`: Exact numbers, and test group 3. This ticket covers user stories 2 to 5. Findings: STD-6, AN2-2, AN2-9, STD-14, DES-16 (constancy).

Blocked by: 01

Status: ready-for-agent

- [ ] `describe_dataset`, `summarize_numeric`, `infer_column_semantics` and `profile_target` decide constancy through one shared rule.
- [ ] On `pd.DataFrame({"id": [2**54, 2**54 + 1, 2**54 + 2]})`, `summarize_numeric` reports no `constant_columns`, and `profile_target(df, target_column="id", task="regression")` reports no `constant_target`. `describe_dataset` agrees.
- [ ] `summarize_numeric(pd.DataFrame({"x": [1e-200, 2e-200, 4e-200, 8e-200]}))` gives a std within 1e-12 relative error of 3.0956959368344516e-200, the exact value computed with fractions.
- [ ] The same tolerance holds for 1,000 normal values scaled by 1e-170 and for values near 1e-160. The fix also applies to `profile_target`'s regression summary.
- [ ] A `decimal128(38, 1)` column holding 2**60+1, 2**60+3, 2**60+5 and 0.1 gets a `float_precision` limitation naming it, in `summarize_numeric`, `profile_target` and `compute_association_matrix`.
- [ ] NaN stored as a value is missing in every analysis, such as NaN in a `double[pyarrow]` column. A limitation names the columns where this differs from pandas' `isna()`.
- [ ] `[1.0, 1.0, NaN]` in a `double[pyarrow]` column gets the same constancy verdict in every analysis.
- [ ] The docstrings and field descriptions that describe constancy and missing values say this.
- [ ] The finish checks pass.

## Comments
