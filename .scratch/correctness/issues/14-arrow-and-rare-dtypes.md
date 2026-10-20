# 14: Arrow and rare dtypes

What to build: pyarrow dictionary columns are classified by their value type, so an Arrow categorical read from Parquet is a categorical everywhere. Arrow time, binary and sparse columns get a defined kind, and the generated test data covers these dtypes and more kinds of categoricals.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (Arrow and rare dtypes), and test group 10. This ticket covers user story 6. Findings: AN1-5, AN2-7, TEST-6.

Blocked by: 01

Status: done

- [x] A categorical written to Parquet and read back with `dtype_backend="pyarrow"` (`dictionary<values=string, indices=int8>[pyarrow]`) is handled like the pandas categorical:
  - `summarize_categorical` summarizes it by default;
  - `detect_missing_sentinels` finds "N/A" and "unknown" in it;
  - `profile_missing_values` accepts `sentinel_values` for it;
  - `validate_column_types` calls its values compatible with categorical;
  - `compute_association_matrix` pairs it;
  - `profile_target` accepts it as classification labels.
- [x] Arrow `time64`, `binary` and sparse columns get a defined kind, or the kind "other" with a reason in the limitation. The error messages no longer claim that pyarrow-backed columns are accepted where they aren't.
- [x] `data_frames()` generates dictionary, time64 and binary columns, and categoricals of booleans, datetimes and mixed values. EDGE_FRAMES pins each of them.
- [x] The paths TEST-6 lists as never run are reached by a targeted test, or by the new generated kinds: the categorical sentinel rules, boolean and datetime categories in validation, key-code overflow, and the parse edges (second 60, offset +24:00).
- [x] The finish checks pass.

## Comments

Settled beyond the spec:
- Every Arrow dictionary dtype is a categorical, whatever its value type (strings, integers and so on), as a pandas categorical of integers is. `_values.categorical_array` gives such a column as a pandas Categorical with the dictionary's entries as categories (unused ones kept), so the levels, validation, sentinel and association code reads both forms alike. A dictionary with null or repeated entries falls back to the decoded values.
- Arrow `time64`, `binary` and nested columns, and sparse columns of anything but numbers and booleans (Sparse[object], sparse datetimes and durations), have the kind "other". Their reason sits where each analysis says why it left a column out: `left_out_columns`, the `unsupported_dtype` skip of `compute_association_matrix` (its message now names these dtypes), and the `ColumnTypeError` that shows the dtype. The README's "Reading a result" section says so.
- The column rules said "nullable and pyarrow-backed ones included"; they now say "of those kinds", since Arrow time and binary columns are not accepted.
- Found by the new generated kinds: `describe_dataset` crashed on Sparse[object] (pandas can't size the objects; now sized as Python objects), and `profile_target` raised `IndexError` on a categorical without categories (fixed in `_levels`).
- TEST-6 paths now reached: boolean, datetime, duration and mixed categories in `validate_column_types` (new RULES rows), categorical sentinel rules (`tests/test_unusual_values.py`), key-code overflow (`find_candidate_keys`, five columns of 7,200 levels), second 60 and offset +24:00 against `datetime.fromisoformat`, and datetime label order for a categorical past 100 levels. Not added: the sparse-table association paths and the `max_categories` probe over 20,000 rows (not in the ticket's list), and `OverflowError` from an integer beyond float range in `infer_column_semantics`: pandas 2.2's `factorize` raises it first on such object columns, a separate bug.
- `data_frames()` gained category of booleans, datetimes and mixed values, dictionary<string> and dictionary<int64>, time64, binary and Sparse[object]; every EDGE_FRAMES frame holds each, and `SECRETS` has a dictionary<string> column. `tests/recipes.py` counts dictionaries as pairable and countable, and Sparse[object] as neither.
- Checks: ruff, ruff format, ty, `uv run pytest` (2,485 tests), both minimum-version runs, the MCP stdio smoke and the benchmark all pass locally. Nothing was left to GitHub.
