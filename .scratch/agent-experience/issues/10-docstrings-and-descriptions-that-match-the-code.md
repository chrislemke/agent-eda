# 10: Docstrings and descriptions that match the code

What to build: every docstring and field description that disagrees with the code is fixed, as are the small record inconsistencies and the differing conversion advice.

Read these sections of `.scratch/agent-experience/spec.md`: Docstrings and descriptions that disagree with the code. This ticket covers user story 25. Findings: STD-25, STD-32, STD-33, STD-45, STD-48 (details in `.scratch/audit/standards-review.md`).

Blocked by: 06

Status: done

- [x] Each case STD-25 lists is fixed, and the Comments note each one.
- [x] Advice on converting a column kind is worded the same way in every analysis (STD-45).
- [x] Each small inconsistency STD-48 lists is fixed, or the Comments record why it stays.
- [x] The docs say that `datetime_format` results depend on the process locale for locale-dependent directives (STD-32).
- [x] The `schema` argument's JSON Schema requires at least one entry (STD-33).
- [x] The finish checks pass.

## Comments

STD-25, case by case (tests in `tests/test_described_behavior.py` unless noted):
- `columns_without_rules`: the code is right and the text was wrong. An all-NA boolean column is listed, so the description now says "A column without values is listed too."
- `insufficient_data` of `summarize_categorical` and `infer_column_semantics`: the precedence rule stays. The Statuses entries now say that no selected column completed and at least one has no value, even if the others hold values that can't be hashed or that no logical type fits. `summarize_numeric` needed no change: every selected column is numeric.
- The `candidates` of `infer_column_semantics` are "the logical types the values fit", not "scoring above 0": integer and float appear at score 0 for leading-zero codes. The `completed` entry still says "a candidate scoring above 0", which is true of the top one.
- The tie order was a code bug, not only a text one: candidates are ranked on exact `Fraction`s (the record still holds floats), so categorical 3/5 against text 3/5 ties and `categorical` wins as documented. The docstring says scores equal as exact fractions tie.
- `ValidateColumnTypesResult.to_frame` says the columns table holds the number of distinct invalid values and points to `table='invalid_values'` for the values.
- `ResultModel` was documented as "frozen and strict" but is not strict (`count='3'` coerces). The docstring now says what it is (frozen, no unknown fields, no NaN or infinity). Making it strict would break rebuilding results from JSON, so the text changed.
- `detect_missing_sentinels` has its "The findings:" lead-in.
- "The 200 missed" in `tests/test_properties.py`: already correct on the integration branch (the comment says 500 and draws 500), so nothing to change.

STD-45: one table of conversions in `_columns` (`TO_NUMBERS`, `TO_LABELS`, and `sentence()`), used by `summarize_numeric`, `summarize_categorical`, `compute_association_matrix` and `profile_target`. `summarize_numeric` now has a fix for a datetime column. Wording changed where the analyses differed (see the CHANGELOG); a test asserts the same fragments in every analysis that rejects the kind. The skip messages of `compute_association_matrix` ("convert them to numbers or labels") carry no code, so they stay.

STD-48:
- `ColumnEvidence.value_types` is capped at 100 with `value_types_total`, as in `summarize_categorical`.
- `rich.print(res)` already used `__rich_console__` (an earlier ticket). `rich.print([res])`, `pretty_repr(res)` and devtools still dumped every field through `__rich_repr__` and `__pretty__`; they now give the text. Results and discovery records share one `_PrintedRecord` base for `__str__`, `__repr__`, the Rich and devtools hooks and `to_json`.
- `shows_raw_values` stays. It is a deliberate alias of `supports_masking`: agents ask "can this show my data?", the description says the two always agree, and `tests/test_contract.py` enforces it. Removing it would break a documented record field for no gain.

STD-32: the docs say it (`ColumnSpec.datetime_format`, the `schema` argument of both analyses, and one line in the README); the locale is not pinned, since ticket 13 chose it.

STD-33: the `schema` argument carries a private marker that adds `minProperties: 1` to its JSON Schema (`tests/test_mcp_tools.py`). pydantic's own `MinLen` would also enforce it with a generic message, and the existing "got an empty schema ... such as schema={'age': 'integer'}" message is better, so the marker affects the JSON Schema only. `tests/test_contract.py`'s `keyed_by_str` now unwraps `Annotated`.

README: the Reference is regenerated (only the `to_json` line changed) and stays under the 60,000-character limit (59,944; +16 after tightening two other sentences).

Checks run locally: ruff check and format, ty, `uv run pytest` (3757 passed), both minimum-version runs (3214 and 3758 passed), `smoke_mcp.py`, `pytest -m benchmark` (14 passed). All passed.
