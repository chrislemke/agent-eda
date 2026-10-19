# 16: Sentinels and missing values

What to build: the sentinel rules settled in round 3, Q6:
- numeric codes found in columns of number text;
- pandas' default NA strings on the built-in list;
- text stand-ins matched after stripping surrounding whitespace;
- one written form for datetimes and decimals in every record.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (Sentinels), and test group 10. This ticket covers user story 11. Findings: SPEC-5, AN1 questionable decisions 3 and 4, AN1-9, STD-17, STD-38 (decimal candidates).

Blocked by: 11, 14

Status: done

- [x] In a text column `34, N/A, -999, -999, 51, 29`, as the MCP server reads it from a CSV, `detect_missing_sentinels` suggests both "N/A" and -999.
- [x] The built-in text list includes `#N/A`, `#N/A N/A`, `#NA`, `<NA>`, `-NaN`, `-nan`, `1.#IND`, `1.#QNAN`, `-1.#IND` and `-1.#QNAN`.
- [x] `" N/A"`, `"N/A "` and `"N/A\n"` match as the stand-in "N/A". The record says which raw form occurred.
- [x] `validate_column_types`, `detect_missing_sentinels`, `profile_missing_values` and `summarize_categorical` write datetimes in one ISO 8601 form. They write decimals as numbers when float64 holds them exactly, and as exact text otherwise, the same rule as for integers above 2⁵³.
- [x] Declared values that print the same can still be told apart, and `profile_missing_values`' `to_frame()` keeps every count.
- [x] The decimal128 candidate `12345678901234567.25` is recorded exactly, and declaring the recorded value matches it (STD-38).
- [x] The docstrings and the README state the rules, and the Comments record which first-look ticket 11 decisions they change.
- [x] The finish checks pass.

## Comments

- Number text: in a text column (object, string, pyarrow, categorical categories, or the text values of an object column), the texts that parse under `_parse`'s number rule get the same two rules as a numeric column, judged among those texts only and by value (`"-999"` and `"-999.0"` are one number). Each distinct text is its own candidate and is recorded as the column holds it (`value_type` `str`), because that is the value a text column accepts when declared. So the CSV column `34, N/A, -999, -999, 51, 29` suggests `"-999"` (`builtin_list`, `isolated_extreme`) and `"N/A"`, and both can be declared as shown. Codes with a leading zero don't parse, as everywhere. A text column that is mostly numbers (zip codes without zeros) is judged like an integer column, so `"99999"` at its maximum is suspected.
- Text stand-ins: the built-in list matches after `str.strip()` (pyarrow's `utf8_trim_whitespace` agrees with it on every code point, checked), still in any mix of case. Each raw form that occurs is its own candidate (`" N/A"`, `"N/A\n"`), so the record's `value` says which form occurred. Declared values still match exactly, as before. The list gained `#N/A`, `#N/A N/A`, `#NA`, `<NA>`, `-nan` (so `-NaN` too), `1.#IND`, `1.#QNAN`, `-1.#IND` and `-1.#QNAN`.
- Written form: `record_scalar` in `_written_form.py` is the one rule behind `record_label` (levels, invalid values, class labels) and the sentinel records (candidates, declared values). Datetimes, dates and durations are ISO 8601; a decimal is a number when float64 holds it as written (the float's shortest text reads back as the same decimal: `2.5`, `19.99`, `3`) and exact text otherwise (`"12345678901234567.25"`). The spec's "holds them exactly" is read this way, not as `float(d) == d`, since that would turn every price such as `19.99` into text. Integer-valued decimals up to 2^53 are ints. Durations moved to ISO 8601 as well, in `validate_column_types`, `summarize_categorical` and `profile_target` labels, which the ticket doesn't list; they used `str()` before.
- Declared decimals: for a column of decimals (pyarrow decimal or an object column holding Decimal values and no text), text that writes an exact decimal is read as that decimal, in Python and over MCP alike, so the recorded `"12345678901234567.25"` can be declared and matches exactly. Such a column holding text too keeps declared text as text. `declared_values` returns `Decimal` for those values and both analyses record the declared values from its output.
- Told apart: `DeclaredSentinelValue` has a new `value_type` (`date` against `str`, `bool` against `int`). `profile_missing_values`' `to_frame()` cell for `sentinel_values` changed from `{value: count}`, which dropped a count when keys were equal (`True` and `1`), to a list of `{"value", "value_type", "count"}`.
- Ticket 11 decisions changed: (1) a suggestion for text declared for a column of decimals is no longer the float `likely_value` made of it (`1.2345678901234568e+16` for `'12345678901234567.25'`, which didn't match the stored decimal); the text itself is accepted. (2) "A suggestion is always a value the analyses accept" still holds: every `detect_missing_sentinels` value, now including whitespace forms, number text and exact decimal text, is accepted when declared. (3) `check_distinct` runs after decimal text is read, so `'2.5'` and `2.5` declared together for a decimal column are one value. (4) `json_value` and the candidates' `record_value` both call `record_scalar`; declared and suspected values are written alike.
- Benchmark: `detect_missing_sentinels` went from 0.63 s to 1.4 s on 1,000,000 rows x 100 columns (the 20 text columns now strip and read their numbers; arrow-backed text is filtered in pyarrow before any string becomes a Python object). Limit 5 s. Other analyses unchanged.
- README: it was 104 characters over the 60,000 limit on arrival, so the new paragraph (and the Reference regeneration) are offset by trimming prose in the project-file, MCP response, masking and determinism paragraphs, the pyarrow dictionary sentence and the planned-analyses sentence; now 59,924. The limit is untouched.
- Not run locally: GitHub-only checks.
