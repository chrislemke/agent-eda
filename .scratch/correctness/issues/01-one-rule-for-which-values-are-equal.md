# 01: One rule for which values are equal

What to build: one private helper decides which values are equal, for every analysis that factorizes or counts distinct values. Text holding a NUL character is then counted correctly everywhere, and a column that mixes scalars with lists never crashes.

Read these sections of `.scratch/correctness/spec.md`: Value identity, and test group 1. This ticket covers user stories 1 and 13. Findings: AN1-1, DES-1, STD-2, STD-52, AN2-1, STD-16.

Blocked by: none

Status: ready-for-agent

- [ ] Every factorize or distinct count on raw column values goes through the helper. The Comments list each site converted, starting from DES-1's list.
- [ ] For the object column `pd.Series(["", "\x00a", "b"], dtype=object)`, every analysis counts 3 distinct values:
  - `describe_dataset`'s `distinct_count`;
  - `summarize_categorical`'s distinct count;
  - `infer_column_semantics`' distinct count;
  - `profile_target`'s classes;
  - `compute_association_matrix`'s levels.
- [ ] `find_duplicate_rows` on `pd.DataFrame({"t": pd.Series(["a", "a\x00b"], dtype=object)})` reports no duplicate rows, and `find_candidate_keys` finds `t` a key.
- [ ] `validate_column_types(pd.DataFrame({"qty": pd.Series(["1\x00x", "1", "2", "1\x00x"], dtype=object)}), schema={"qty": "integer"})` counts 2 convertible and 2 invalid values on pandas 3 and on pandas 2.2. The result equals the one for the same values with `dtype="string"`.
- [ ] `find_candidate_keys(pd.DataFrame({"payload": pd.Series([1, [2], 3], dtype=object), "id": [1, 2, 3]}))` completes and leaves `payload` out with `unhashable_values`. So do the int-plus-dict and int-plus-float-plus-list mixes.
- [ ] Each analysis keeps its documented policy for unhashable values, and one parametrized test checks all of them:
  - `find_duplicate_rows` compares by content;
  - `summarize_categorical`, `find_candidate_keys` and `compute_association_matrix` leave the column out;
  - `describe_dataset` and `infer_column_semantics` leave the distinct count null.
- [ ] Sets that hold missing values compare by content, as the docstring and field descriptions say (STD-16).
- [ ] `data_frames()` generates NUL text and int-plus-list columns. A property over `ae.list_analyses()` compares each analysis's level, group, class and key counts with a count by Python `==`.
- [ ] The benchmark doesn't get slower. The Comments record the times before and after, on the latest and the minimum versions.
- [ ] The finish checks pass.

## Comments
