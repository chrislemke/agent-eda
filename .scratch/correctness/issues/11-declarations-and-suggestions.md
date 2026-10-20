# 11: Declarations and suggestions

What to build: `Project`, `DataPolicy` and `ColumnSpec` share one validation base, so every way of building them raises the package's own error. One parser produces the "did you mean" suggestions, and every suggestion it offers is one the analyses accept.

Read these sections of `.scratch/correctness/spec.md`: Declarations and suggestions, and test group 9. This ticket covers user stories 25 and 26. Findings: DES-4, AN1-11, CORE-18, SPEC-9, STD-20, CORE-17, STD-37, DES-10, AN1-7, STD-19.

Blocked by: none

Status: done

- [x] Each of these raises `InvalidArgumentError` with the allowed values:
  - `ae.ColumnSpec.model_validate_json('{"logical_type": "int"}')`;
  - `ae.ColumnSpec.model_validate_strings(...)` with a bad value;
  - `ae.ColumnSpec.model_validate({1: "integer"})`;
  - `ae.validate_column_types(df, schema={"a": {1: "integer"}})`.
- [x] `model_copy(update=...)` with an invalid value raises, or returns a validated object, for every declaration model.
- [x] A contract test loops over every way of building `Project`, `DataPolicy` and `ColumnSpec`: the constructor, `model_validate`, `model_validate_json`, `model_validate_strings`, non-string keys and `model_copy(update=...)`.
- [x] `ae.DataPolicy(mask_sensitive=np.True_)` is accepted. NumPy scalars are accepted wherever Python scalars are, and `Project`'s types are written in one place (STD-37).
- [x] One parser produces every suggestion for `sentinel_values` and `expected_labels`. A property checks that every suggested value is accepted when declared. "1_000" and "١٢" get no suggestion that the analyses would reject.
- [x] `sentinel_values={"o": [True, 1]}` on an object column `[True, 1, 1.0, "1"]` is accepted. True matches 1 value, and 1 matches 2.
- [x] The finish checks pass.

## Comments

- The shared base is `src/agent_eda/_declarations.py` (`Declaration`). `Project`, `DataPolicy` and `ColumnSpec` subclass it. A subclass overrides `_rejected` (ColumnSpec for a missing `logical_type`) and `_unknown_field` (Project for a data policy field given to the Project); `ColumnSpec` keeps its own `__init__` only for the positional `logical_type`.
- `model_copy(update=...)` validates the merged fields through `model_validate`, so it raises `InvalidArgumentError` for an invalid value and returns a validated object otherwise. An update that names no field raises `invalid_call`. A copy without `update` stays pydantic's.
- NumPy scalars become Python scalars once, in the base's wrap validator, which every route passes through (constructor, `model_validate`, JSON, strings, copies).
- `Project`'s types are written once, in `_field_types`: help(), the error message and the docstring all say `DataPolicy | Mapping[str, Any] | None`. The error text now reads "must be DataPolicy | Mapping[str, Any] | None" instead of "must be DataPolicy". The messages of the declarations are built with `worded`, so a surface can carry its own wording.
- A `validate_column_types` schema with a non-string key inside a spec raises `invalid_argument` ("... at schema['a']: ColumnSpec got the key 1 ..."), because the analysis wraps the declaration's error; direct construction gives `invalid_call`.
- `_parse.read_as(text, family, *, whole=False)` is the one parser behind every suggestion (`likely_value` for `sentinel_values` and `expected_labels`). It reuses `_parse._read`, so ASCII-only digits, no leading zeros, no `1_000`, no inf or nan, and `yes`, `no`, `t`, `f`, `y`, `n`, `0` and `1` read as booleans. Dates and datetimes need the ISO 8601 form the analyses read (`20240105` no longer suggests anything, and nanoseconds are kept). Durations still read through pandas, ASCII text only: no analysis reads text as a duration.
- `check_distinct` takes an optional `key`; declared sentinel values use `(family, value)`. `1` and `1.0` are still one value, `True` and `1` are two. Declaring `[True, 1]` for the object column `[True, 1, 1.0, "1"]` counts 1 and 2 values. I did not change the `expected_labels` duplicate message: labels compare as Python `==`, so `[1, True]` is one label by design, and ticket 15 owns that wording.
- Not run locally: the benchmark (nothing here changes a cost that grows with rows or columns); GitHub-only checks.
