# 16: Sentinels and missing values

What to build: the sentinel rules settled in round 3, Q6:
- numeric codes found in columns of number text;
- pandas' default NA strings on the built-in list;
- text stand-ins matched after stripping surrounding whitespace;
- one written form for datetimes and decimals in every record.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (Sentinels), and test group 10. This ticket covers user story 11. Findings: SPEC-5, AN1 questionable decisions 3 and 4, AN1-9, STD-17, STD-38 (decimal candidates).

Blocked by: 11, 14

Status: ready-for-agent

- [ ] In a text column `34, N/A, -999, -999, 51, 29`, as the MCP server reads it from a CSV, `detect_missing_sentinels` suggests both "N/A" and -999.
- [ ] The built-in text list includes `#N/A`, `#N/A N/A`, `#NA`, `<NA>`, `-NaN`, `-nan`, `1.#IND`, `1.#QNAN`, `-1.#IND` and `-1.#QNAN`.
- [ ] `" N/A"`, `"N/A "` and `"N/A\n"` match as the stand-in "N/A". The record says which raw form occurred.
- [ ] `validate_column_types`, `detect_missing_sentinels`, `profile_missing_values` and `summarize_categorical` write datetimes in one ISO 8601 form. They write decimals as numbers when float64 holds them exactly, and as exact text otherwise, the same rule as for integers above 2⁵³.
- [ ] Declared values that print the same can still be told apart, and `profile_missing_values`' `to_frame()` keeps every count.
- [ ] The decimal128 candidate `12345678901234567.25` is recorded exactly, and declaring the recorded value matches it (STD-38).
- [ ] The docstrings and the README state the rules, and the Comments record which first-look ticket 11 decisions they change.
- [ ] The finish checks pass.

## Comments
