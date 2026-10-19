# 14: Arrow and rare dtypes

What to build: pyarrow dictionary columns are classified by their value type, so an Arrow categorical read from Parquet is a categorical everywhere. Arrow time, binary and sparse columns get a defined kind, and the generated test data covers these dtypes and more kinds of categoricals.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (Arrow and rare dtypes), and test group 10. This ticket covers user story 6. Findings: AN1-5, AN2-7, TEST-6.

Blocked by: 01

Status: ready-for-agent

- [ ] A categorical written to Parquet and read back with `dtype_backend="pyarrow"` (`dictionary<values=string, indices=int8>[pyarrow]`) is handled like the pandas categorical:
  - `summarize_categorical` summarizes it by default;
  - `detect_missing_sentinels` finds "N/A" and "unknown" in it;
  - `profile_missing_values` accepts `sentinel_values` for it;
  - `validate_column_types` calls its values compatible with categorical;
  - `compute_association_matrix` pairs it;
  - `profile_target` accepts it as classification labels.
- [ ] Arrow `time64`, `binary` and sparse columns get a defined kind, or the kind "other" with a reason in the limitation. The error messages no longer claim that pyarrow-backed columns are accepted where they aren't.
- [ ] `data_frames()` generates dictionary, time64 and binary columns, and categoricals of booleans, datetimes and mixed values. EDGE_FRAMES pins each of them.
- [ ] The paths TEST-6 lists as never run are reached by a targeted test, or by the new generated kinds: the categorical sentinel rules, boolean and datetime categories in validation, key-code overflow, and the parse edges (second 60, offset +24:00).
- [ ] The finish checks pass.

## Comments
