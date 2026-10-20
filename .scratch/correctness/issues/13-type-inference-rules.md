# 13: Type inference rules

What to build: `infer_column_semantics` and `validate_column_types` judge ordinary data as settled in round 2, Q9 a to d and f:
- booleans only when the values really are booleans;
- leading-zero codes only for whole numbers;
- dates stored as text never called identifiers;
- ID columns recognised despite a few duplicate rows;
- ties with a declared type counted as matches;
- a declared datetime format honoured.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (Type inference), and test group 10. This ticket covers user story 7. Findings: AN1-2, AN1-3, AX-13, AN1-4, AX-12, AN1 questionable decisions 1, 2 and 5, AN1-6, STD-38 (overflowing text).

Blocked by: 01

Status: done

- [x] These columns rank categorical, not boolean, and get no `boolean_like` flag:
  - grades `A, B, C, F, F, F, F, F`;
  - `["F"] * 9 + ["M"]`;
  - `["N"] * 7 + ["E", "S", "W"]`.

  A real 0/1 or yes/no column still ranks boolean.
- [x] `[1, 1, 1, 1]` and an all-zero column are never boolean.
- [x] `["05.50", "12.00", "07.25", "10.10"]` ranks float, with no `identifier_like` flag and no leading-zero reason. "0123" is still a code, and validate_column_types counts "05.50" as convertible to float.
- [x] A 60-day series written as `%m/%d/%Y` text, and as `%d.%m.%Y` text, gets no `identifier_like` flag.
- [x] A customer ID column with 20 of 5,000 rows duplicated exactly is flagged `identifier_like`.
- [x] A 0/1 column declared integer, and the docstring's `amount` example declared float, get no `declared_type_mismatch`.
- [x] A declared `ColumnSpec("datetime", datetime_format="%d/%m/%Y")` matches text in that format.
- [x] Text that overflows float64, such as "1e400" and "-1e400", doesn't count as converting to float in `validate_column_types` or as a float share in `infer_column_semantics`. `Decimal('1E+400')` values in an object column give a result, not a pydantic `ValidationError` (STD-38).
- [x] With no type inferred, the printed table and `to_frame()` show no top type or score, and the docstring says what `candidates` holds (AN1-6).
- [x] The docstrings, the README and the Comments record the rules, and which first-look ticket 12 decisions they change.
- [x] The finish checks pass.

## Comments

What this ticket settled that the spec left open:

- **Boolean.** The boolean candidate exists only when every value reads as a boolean (a boolean, a 0 or 1, or a boolean word) and the column holds at least two distinct values. Its score is then 1; otherwise there is no boolean candidate, so `["yes", "no", "maybe"]` is not boolean either. A column stored as booleans (bool dtype or Python `True` and `False`) stays boolean when constant: the ticket names only `[1, 1, 1, 1]` and all-zero, and a dtype that states boolean is better answered with boolean plus the `constant` flag than with "categorical" or, for one row, no candidate at all. The spec's "a constant column is never boolean" is read as applying to values that only read as booleans.
- **Leading zeros.** Only digits alone are a code: an integer part of two or more digits starting with 0, with no decimal point and no exponent, sign allowed (`007`, `-007`). `05.50`, `00.5`, `0123.5` and `01e3` are numbers. This changes ticket 07's CSV reading, which shares the rule through `read_texts`: those four fields no longer keep a column as text. I changed three parameters of `test_each_way_to_write_a_leading_zero_code_keeps_the_column_text` into cases of `test_a_number_that_only_contains_a_zero_stays_a_number`, and the README says so.
- **Overflow.** Text whose value float64 can't hold (`1e400`, `-1e400`, a 400-digit integer) counts as neither a number nor a whole number, so it is plain text in `infer_column_semantics` and invalid for float and integer in `validate_column_types`. `1e-400` (underflow to 0) and integers above 2**53 are left as they were; the rest of AN1's "converts exactly" exceptions (questionable decision 5) stay open. `Decimal('1E+400')` already gave a result on the integration branch (no `ValidationError` in infer, validate or the other analyses), so I added a regression test only; decimals beyond float64 still count as convertible to float.
- **Date-like text.** `identifier_like` is off whenever `non_iso_date_text` would be on (at least half the values are dates, some not ISO 8601). A declared `datetime_format` that reads the text turns it into datetimes, so the column ranks datetime anyway.
- **Duplicate rows.** A column is judged again without exact duplicate rows only when its values are distinct in at least 90% of its non-missing rows, its top candidate is integer or text, and it isn't already all distinct. Duplicate rows are found over every column of `data`, not only the selected ones (a column alone would always look unique), by the code `find_duplicate_rows` uses, moved to `agent_eda/_row_groups.py` as `find_row_groups`; the candidate columns go first, so a near-unique column leaves few rows to compare. A frame with 1,000,000 rows x 100 columns, one id column and one text id column, with 50,000 exact duplicate rows, takes 1.1 s. The flag's other rules (two or more values, no multi-word text, the sequence rule for numbers) are unchanged and judged on the deduplicated count where it applies. As a consequence, `pd.Series([1, 1, 2, 3])` alone is `identifier_like` after dropping the repeated row.
- **Declared ties.** `matches_declared` is true when the declared type scores the same as the top candidate (exact equality, the one that orders candidates). `declared_type_match`'s claim and the printed "declared_schema:" line say a tie counts. The docstring's own `amount` example now prints a `declared_type_match` finding.
- **Declared datetime_format in inference.** Text that `datetime.strptime` reads in the declared format counts as a datetime (`datetime_share`, `dates_as_text`) and no longer as date-like text, as validate_column_types does through the shared `read_datetime_format`. The `dates_as_text` claim reads "stores dates as text" now, not "ISO 8601 dates", since the text may be in the declared format. The format is read once per distinct text that isn't ISO 8601.
- **No inferred type (AN1-6).** The printed table and `to_frame()`'s `score` use the top candidate only when `inferred_logical_type` isn't null; the `candidates` table still lists the 0-scored integer and float. The docstring and the field descriptions say that `candidates` also holds integer and float scored 0 when a leading zero rules them out.

First-look ticket 12 decisions this changes: the boolean score (was boolean_share, any column with a boolean-word top value ranked boolean), the identifier_like conditions (every value distinct, now after dropping duplicate rows and never for date-like text), what `matches_declared` and `declared_type_match` mean (equal top, now a tie counts), "inference parses ISO 8601 only, also for a declared datetime_format" (a declared format is now read), the leading-zero rule and the `number_share` of overflowing text. The `ColumnSpec.datetime_format` field description says both analyses read it.

Checks:

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check` and `uv run pytest` pass (2,488 passed). The two `--isolated --resolution lowest-direct` runs pass (without the extra: 2,124 passed, 12 MCP files skipped; with it: 2,488 passed). `tests/smoke_mcp.py` passes. `uv run python scripts/generate_reference.py --check` is clean. The `thorough` Hypothesis profile passes for the property, infer, validate and new rule tests.
- Benchmark (`uv run pytest -m benchmark`, Apple M3 Max, latest versions): infer_column_semantics 1.70 s, validate_column_types 1.55 s.
- Not run locally: Linux and the CI-only checks.
