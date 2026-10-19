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

Status: ready-for-agent

- [ ] These columns rank categorical, not boolean, and get no `boolean_like` flag:
  - grades `A, B, C, F, F, F, F, F`;
  - `["F"] * 9 + ["M"]`;
  - `["N"] * 7 + ["E", "S", "W"]`.

  A real 0/1 or yes/no column still ranks boolean.
- [ ] `[1, 1, 1, 1]` and an all-zero column are never boolean.
- [ ] `["05.50", "12.00", "07.25", "10.10"]` ranks float, with no `identifier_like` flag and no leading-zero reason. "0123" is still a code, and validate_column_types counts "05.50" as convertible to float.
- [ ] A 60-day series written as `%m/%d/%Y` text, and as `%d.%m.%Y` text, gets no `identifier_like` flag.
- [ ] A customer ID column with 20 of 5,000 rows duplicated exactly is flagged `identifier_like`.
- [ ] A 0/1 column declared integer, and the docstring's `amount` example declared float, get no `declared_type_mismatch`.
- [ ] A declared `ColumnSpec("datetime", datetime_format="%d/%m/%Y")` matches text in that format.
- [ ] Text that overflows float64, such as "1e400" and "-1e400", doesn't count as converting to float in `validate_column_types` or as a float share in `infer_column_semantics`. `Decimal('1E+400')` values in an object column give a result, not a pydantic `ValidationError` (STD-38).
- [ ] With no type inferred, the printed table and `to_frame()` show no top type or score, and the docstring says what `candidates` holds (AN1-6).
- [ ] The docstrings, the README and the Comments record the rules, and which first-look ticket 12 decisions they change.
- [ ] The finish checks pass.

## Comments
