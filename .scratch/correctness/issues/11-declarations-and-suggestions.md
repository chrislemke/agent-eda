# 11: Declarations and suggestions

What to build: `Project`, `DataPolicy` and `ColumnSpec` share one validation base, so every way of building them raises the package's own error. One parser produces the "did you mean" suggestions, and every suggestion it offers is one the analyses accept.

Read these sections of `.scratch/correctness/spec.md`: Declarations and suggestions, and test group 9. This ticket covers user stories 25 and 26. Findings: DES-4, AN1-11, CORE-18, SPEC-9, STD-20, CORE-17, STD-37, DES-10, AN1-7, STD-19.

Blocked by: none

Status: ready-for-agent

- [ ] Each of these raises `InvalidArgumentError` with the allowed values:
  - `ae.ColumnSpec.model_validate_json('{"logical_type": "int"}')`;
  - `ae.ColumnSpec.model_validate_strings(...)` with a bad value;
  - `ae.ColumnSpec.model_validate({1: "integer"})`;
  - `ae.validate_column_types(df, schema={"a": {1: "integer"}})`.
- [ ] `model_copy(update=...)` with an invalid value raises, or returns a validated object, for every declaration model.
- [ ] A contract test loops over every way of building `Project`, `DataPolicy` and `ColumnSpec`: the constructor, `model_validate`, `model_validate_json`, `model_validate_strings`, non-string keys and `model_copy(update=...)`.
- [ ] `ae.DataPolicy(mask_sensitive=np.True_)` is accepted. NumPy scalars are accepted wherever Python scalars are, and `Project`'s types are written in one place (STD-37).
- [ ] One parser produces every suggestion for `sentinel_values` and `expected_labels`. A property checks that every suggested value is accepted when declared. "1_000" and "١٢" get no suggestion that the analyses would reject.
- [ ] `sentinel_values={"o": [True, 1]}` on an object column `[True, 1, 1.0, "1"]` is accepted. True matches 1 value, and 1 matches 2.
- [ ] The finish checks pass.

## Comments
