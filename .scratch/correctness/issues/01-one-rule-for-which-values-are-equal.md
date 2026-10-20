# 01: One rule for which values are equal

What to build: one private helper decides which values are equal, for every analysis that factorizes or counts distinct values. Text holding a NUL character is then counted correctly everywhere, and a column that mixes scalars with lists never crashes.

Read these sections of `.scratch/correctness/spec.md`: Value identity, and test group 1. This ticket covers user stories 1 and 13. Findings: AN1-1, DES-1, STD-2, STD-52, AN2-1, STD-16.

Blocked by: none

Status: done

- [x] Every factorize or distinct count on raw column values goes through the helper. The Comments list each site converted, starting from DES-1's list.
- [x] For the object column `pd.Series(["", "\x00a", "b"], dtype=object)`, every analysis counts 3 distinct values:
  - `describe_dataset`'s `distinct_count`;
  - `summarize_categorical`'s distinct count;
  - `infer_column_semantics`' distinct count;
  - `profile_target`'s classes;
  - `compute_association_matrix`'s levels.
- [x] `find_duplicate_rows` on `pd.DataFrame({"t": pd.Series(["a", "a\x00b"], dtype=object)})` reports no duplicate rows, and `find_candidate_keys` finds `t` a key.
- [x] `validate_column_types(pd.DataFrame({"qty": pd.Series(["1\x00x", "1", "2", "1\x00x"], dtype=object)}), schema={"qty": "integer"})` counts 2 convertible and 2 invalid values on pandas 3 and on pandas 2.2. The result equals the one for the same values with `dtype="string"`.
- [x] `find_candidate_keys(pd.DataFrame({"payload": pd.Series([1, [2], 3], dtype=object), "id": [1, 2, 3]}))` completes and leaves `payload` out with `unhashable_values`. So do the int-plus-dict and int-plus-float-plus-list mixes.
- [x] Each analysis keeps its documented policy for unhashable values, and one parametrized test checks all of them:
  - `find_duplicate_rows` compares by content;
  - `summarize_categorical`, `find_candidate_keys` and `compute_association_matrix` leave the column out;
  - `describe_dataset` and `infer_column_semantics` leave the distinct count null.
- [x] Sets that hold missing values compare by content, as the docstring and field descriptions say (STD-16).
- [x] `data_frames()` generates NUL text and int-plus-list columns. A property over `ae.list_analyses()` compares each analysis's level, group, class and key counts with a count by Python `==`.
- [x] The benchmark doesn't get slower. The Comments record the times before and after, on the latest and the minimum versions.
- [x] The finish checks pass.

## Comments

`_values.value_codes(values, *, unhashable="by_content" | "unsupported")` is the helper. It returns `ValueCodes(codes, uniques, by_content)` with -1 for missing, or None for "unsupported" when a value can't be hashed. It takes a Series, Index, array or extension array.

Decisions the spec left open:

- The NUL guard acts only where pandas' string hash table is chosen: an object array (or the python-backed `string` dtype of pandas 2.2) whose values are all `str`, with no missing value. `infer_dtype(skipna=False) == "string"` finds those, and `"\x00" in "".join(chunk)`, 65,536 texts at a time, finds the NUL. Only then does the helper append a non-text marker to a copy, so pandas uses the object hash table, and drops the marker's code and unique. A column with None or NaN among the text was never wrong. Arrow-backed text and pandas 3's `str` dtype were never wrong either.
- The pandas 2.2 python-backed `string` dtype had the same bug, so `validate_column_types` on `dtype="string"` was wrong on 2.2 before. It is right now, which is what makes the object and `string` results equal on both versions.
- `find_candidate_keys`' `hashable()` no longer trusts `infer_dtype` to say a column holds lists. It accepts a column without hashing only when `infer_dtype` names a scalar-only kind (`_keys._SCALARS`, which excludes `mixed`, `mixed-integer` and `unknown-array`); any other object column is hashed by `value_codes`. That is what catches integers next to lists, dicts or lists with floats.
- `mask_values` now uses the helper too, replacing its own marker trick. Values it can't hash compare by content instead of raising.
- Sets inside a column compare by content with their items' stand-ins (`_values._content`), so `{1, nan}` and `{1, None}` are equal (STD-16). A `frozenset` is hashable, so it still compares as pandas compares it.
- `Series.duplicated` is not affected by the NUL bug (checked on pandas 3.0 and 2.2), so `_row_ids.duplicated_labels` stays as it was. Its hashability check for `id_column` goes through the helper.

Sites converted (every `pd.factorize` and `nunique` on raw column values):

- `_levels._count` (summarize_categorical, profile_target and infer_column_semantics through `count_levels`), and `_levels` case-and-space groups (the `strip().casefold()` keys).
- `_conversions._groups` (boolean, datetime, timedelta and text columns), `_numbers` (both factorizes) and `_objects` (validate_column_types).
- `_keys.codes_of` and `hashable` (find_candidate_keys).
- `data/duplicates._codes` (find_duplicate_rows).
- `data/provenance._object_column` and `_distinct_count` (describe_dataset; the latter was `Series.nunique`).
- `statistics/association._codes` (compute_association_matrix).
- `data/validation._read_unhashable` (infer_column_semantics).
- `_sentinel_rules._factorized`, `_object_suspects` and `_text_rows_suspects` (detect_missing_sentinels).
- `_row_ids.read_row_ids` (the hashability check of `id_column`).
- `_columns.Labels` (column labels).
- `_masking.mask_values`.

Left as they were, because they factorize derived integers or type objects, not column values: `_conversions` type codes and combined codes, `_keys` combined codes, `_association` cell codes, `data/duplicates._renumber`.

Tests: `tests/test_value_identity.py`. `tests/strategies.py` has three new kinds: NUL text, NUL text with missing values, and integers, floats and lists in one column. The property runs each analysis with a check against a Python `==` count; a test fails when a new analysis is in neither `CHECKS` nor `COUNTS_NOTHING`. With the guard switched off, five of the seven checks fail within 50 examples (the other two have their own targeted tests that fail).

Benchmark (`uv run pytest -m benchmark`, 1M rows x 100 columns, Darwin arm64, seconds, two runs each). Run-to-run noise on this machine is about 0.2 s, and more on some runs.

| analysis | latest before | latest after | minimum before | minimum after |
| --- | --- | --- | --- | --- |
| describe_dataset | 0.97 / 1.22 / 1.12 | 1.14 / 1.07 | 2.12 / 2.21 | 2.23 / 3.33 |
| summarize_categorical | 1.14 / 1.36 / 1.23 | 1.36 / 1.68 | 1.54 / 1.60 | 1.73 / 1.77 |
| infer_column_semantics | 1.69 / 1.81 / 2.15 | 1.73 / 2.18 | 2.03 / 2.55 | 2.35 / 2.72 |
| validate_column_types | 1.56 / 1.67 / 1.71 | 1.57 / 2.14 | 1.81 / 2.01 | 1.91 / 2.10 |
| find_candidate_keys | 0.42 / 0.54 / 0.73 | 0.48 / 0.47 | 0.54 / 0.57 | 0.62 / 0.73 |
| compute_association_matrix | 0.87 / 0.89 / 0.89 | 0.89 / 1.27 | 1.44 / 1.51 | 1.45 / 1.50 |
| others | within noise | within noise | within noise | within noise |

Latest is Python 3.12, numpy 2.5.4, pandas 3.0.6; minimum is numpy 1.26.0, pandas 2.2.0. On the latest versions text is the `str` dtype, which the guard never touches, so the differences there are noise. On the minimum versions text columns are object, and the guard's NUL check costs 10 ms for a 1M-row column of short texts (49 ms when every text is distinct), measured alone: about 0.1 to 0.2 s over the benchmark's 20 text columns. Every analysis stays under the 5 s target, and the slowest after (describe_dataset on pandas 2.2, 2.2 s in the median run) was already the slowest before.

Checks: ruff, ruff format, ty, `uv run pytest` (2,289 passed), both minimum-version runs, the MCP stdio smoke and the Reference check pass. The properties also pass under `--hypothesis-profile=thorough`. CI-only checks: none.
