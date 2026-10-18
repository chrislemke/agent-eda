# 11: detect_missing_sentinels

**What to build:** An agent calls `ae.detect_missing_sentinels(df)` and gets the values that look like stand-ins for missing data, such as -999, "N/A" or 1900-01-01, each with its evidence and example row IDs. Suspected values stay apart from the sentinel values the agent declared, and the data never changes, so legitimate zeros, negatives and rare codes stay valid until the agent confirms otherwise.

The analysis lives in `agent_eda.data.missingness`.

Read these sections of `.scratch/first-look/spec.md`: the detect_missing_sentinels entry under The first-look analyses, Row IDs, and Project, data policy and masking. Covers user stories 93 and 94.

**Blocked by:** 08-project-and-masking, 10-profile-missing-values

**Status:** ready-for-agent

- [x] Candidates come only from fixed, documented rules with no tunable thresholds: values from the built-in list, and repeated numeric values at the column's minimum or maximum whose distance to the nearest other value exceeds the range of all other values.
- [x] The built-in list covers:
  - the numeric codes -9, -99, -999, -9999, 99, 999, 9999, 99999 and 999999, when they sit at the column's minimum or maximum
  - text, compared case-insensitively: "", whitespace only, "NA", "N/A", "NULL", "None", "nan", "-", "?", "unknown" and "missing"
  - the dates 1900-01-01, 1970-01-01, 2099-12-31 and 9999-12-31
- [x] Each candidate reports its column, value, count, share, evidence codes, up to 10 example row IDs with the total, and whether it is declared or suspected.
- [x] Declared values give descriptive findings, and matches from the built-in list or the gap rule give suspected ones. A declared value is never also reported as suspected.
- [x] Declared `sentinel_values` follow the same rules as in profile_missing_values, from one shared implementation, including the error for a value that can never match its column's type.
- [x] A value that matches no rule, such as a legitimate 0, a negative number or a rare code, is never a candidate.
- [x] The record caps candidates at 100 with the total, and `to_frame()` lists all of them.
- [x] `id_column`, `project` and `mask_sensitive` work as in find_duplicate_rows. With masking on, candidate values in sensitive columns and row IDs from a sensitive `id_column` become placeholders.
- [x] The declared statuses are `completed`, `insufficient_data`, `not_applicable` and `failed`.
- [x] The docstring has the seven sections in order and passes the contract tests.
- [x] The property suite has a call recipe for it, and the properties hold, including the no-leak property.
- [x] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [x] `ae.search_analyses("-999 placeholder codes")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [x] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [x] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **Result fields.** `DetectMissingSentinelsResult` has `candidates`, one `SentinelCandidate` per column and value, and `columns_without_rules`. A candidate has `column`, `value`, `value_type`, `count`, `share`, `evidence`, `confirmation` and `row_ids`. `value` is a declared value as declared and a suspected one as stored. Dates, datetimes and durations show as ISO 8601 text and decimals as floats, so `value_type` names the type an agent has to declare ('Timestamp', 'date', 'Decimal'). `share` divides by all rows, so declaring the value raises the column's `missing_share` in profile_missing_values by exactly that much. The result object keeps only the first 10 row IDs, through `_Cap(10, total="count")`, and the record adds `row_ids_total`. Candidates follow the checked columns; within a column the declared values come first, in the order given, then the suspected ones, most frequent first, ties by first row. The printed table sorts all candidates by count, ties in record order.
- **Default selection.** `columns=None` checks every column except `id_column`, as profile_missing_values does, so `sentinel_values` may name a column of any dtype in both analyses. `columns_without_rules` names the checked columns whose values no built-in rule reads (booleans, durations, lists, periods, and object or categorical columns holding only such values), and the text says only declared values were looked for there. A column without values isn't listed.
- **The number rules.** They read the finite values that are neither missing nor declared. Infinity is never a candidate, since a declared infinite value raises, and NaN stored as a value isn't either. For the gap rule, "all other values" means every value except the candidate, and they must hold two distinct values. Read literally, the rule would flag both values of any 0/1 column, since a range of 0 is always exceeded; the ticket's legitimate zeros have to stay valid, so I added that condition and documented it. Gaps compare as exact fractions, so integers beyond 2^53 and decimals don't round. Both extremes can be suspected; when they are equal, only one candidate shows.
- **The text rules.** "Compared case-insensitively" means any mix of upper and lower case of the ASCII words, looked up among their 306 spellings. Python's `casefold()` would also turn 'miſſing' into 'missing', which no agent means. Surrounding spaces count, so ' N/A' is no candidate while '' and '\t \n' are. Whitespace-only follows Python's `str.isspace()`. pyarrow's `utf8_is_space` agrees with it on all 1,112,064 code points with pyarrow 14.0.1 and 26.0.0, so pyarrow-backed text uses pandas' `str.isspace()` there.
- **The date rules.** A date column matches the four dates. A datetime matches at midnight on one of them in its own time zone, or in UTC when it has one, so the Unix epoch shown in Europe/Berlin (1970-01-01 01:00+01:00) counts. Where the clocks went back at midnight both midnights count, and where they skipped it none does. Text dates are text and never parsed, as in the other analyses. The datetime targets go through the shared `_sentinels.matches` and the new `_sentinels.can_match`, so a column that can't hold 9999-12-31, such as datetime64[ns], simply skips it.
- **Object columns.** They are read value by value, family by family, with `_sentinels.type_family`, so True never counts as the number 1 and -999 equals -999.0. Numbers of Python types compare as exact fractions; complex numbers and non-finite decimals are left out. Categoricals are judged by the categories their rows take, so an unused 'unknown' category is no candidate.
- **Declared values.** `_sentinels.declared_values` checks them and `_sentinels.matches` finds them, the same calls profile_missing_values makes, and a test compares the errors of both analyses message by message for 11 kinds of bad declarations. A declared value the column doesn't hold is no candidate; `parameters` still records it. The rules read a column as if its declared values were missing, so declaring -999 lets them find -998 behind it, and a declared value is never also suspected. A declared candidate's evidence is just `declared`.
- **Findings.** `sentinel_values_present` is descriptive and reuses profile_missing_values' code, since it states the same fact; `suspected_sentinel_values` is suspected. Both point at `candidates[i]`. A run without candidates has no finding.
- **Statuses.** Data without rows gives `insufficient_data` with code `no_rows`. With no column to check, the data having none or only `id_column`, the status is `not_applicable` with code `no_columns_to_check`, worded as find_duplicate_rows words its own. `failed` is declared, and no path returns it.
- **Masking.** One `mask_values` call per sensitive column numbers, in result order, its declared values in `parameters`, its candidate values and, when it is `id_column`, the row IDs. An agent can map its own declared values to placeholders by position. I mask the declared values because otherwise the result still shows values of the sensitive column, which the no-leak property caught as soon as the recipe declared them. `masking.columns` names the columns named in `sentinel_values`, the columns with candidates and `id_column`. profile_missing_values doesn't support masking, so it keeps showing declared values, as the spec has it.
- **pandas reads '\x00' as the end of text.** `pd.factorize` of strings uses a hash table that stops at a NUL, so it takes '' and '\x00a' for one value on both pandas versions, and pandas 2.2 even builds `pd.Series(['\x00', ''], dtype='category')` as two '\x00'. Random fuzzing found it. Text is therefore matched row by row and only the matching rows, which hold no NUL, are factorized. `_masking.mask_values` had the same bug, giving different row IDs one placeholder; it now adds a non-text marker that keeps pandas on its object hash table, which also fixes find_duplicate_rows. I didn't touch the factorize calls of summarize_categorical, describe_dataset and find_duplicate_rows, which still count such values as one; that is worth its own ticket.
- **Shared code.** The rules live in the new private module `agent_eda/_sentinel_rules.py`. `_sentinels.py` now shares `numpy_dtype` and `type_family`, formerly private, and gained `can_match`.
- **Tests.** `tests/test_detect_missing_sentinels.py` holds 99 tests. A property checks the number rules against a plain Python version written from the ticket's wording, and three more check that numbers, text and datetimes get the same suspects in every dtype that holds them (int64, Int64, float64, int64[pyarrow], object and category; five text dtypes and category; datetime64[ns], object, category and timestamp[pyarrow]). The recipe draws `id_column`, `columns` and declared values through `declared_sentinel_values`, which I split out of `missing_value_arguments` keeping its draw order, so profile_missing_values draws the same examples. `sensitive_options` declares values drawn from the sensitive columns and sometimes takes `id_column` from one: in 300 examples, 244 results showed sensitive values. Breaking masking three ways (no masking, unmasked parameters, unmasked row IDs) failed the no-leak property each time.
- **Search and docs.** Tags: sentinel, stand-ins, placeholder, -999, missing codes, magic number, plus the masking tags. "-999 placeholder codes" scores 9 against 3 for the next analysis, and "missing values" still finds profile_missing_values first, 8 to 6. profile_missing_values, describe_dataset, summarize_numeric and summarize_categorical point here in `When to use:` and `See Also:`. The guide's first-look order already named it, the README's status line names it, and the catalogue entry lists its differences.
- **Glossary.** No new term. The code says "candidate" as the spec does ("sentinel value candidates") and "stand-in" as GLOSSARY.md's definition of a sentinel value does.

Measurements, on the M3 Max with Python 3.12.11 and 2 threads:

| Versions | Benchmark, default arguments |
| --- | --- |
| pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0 | 0.58 and 0.58 s |
| pandas 2.2.0, numpy 1.26.0, pyarrow 14.0.1 | 1.41 and 1.42 s |

- The first version took 4.42 s with the latest versions and 5.67 s with the minimum ones. It judged every distinct value of text in a Python loop, 0.5 s for each all-distinct text column and 1.2 s for the categorical with a million categories. Judging distinct values with pandas' `isin` and `str.isspace()` instead brought the latest versions to 1.1 s, and matching text row by row to 0.8 s. On pandas 2.2 the text is object dtype, and `pd.factorize` of a column with 500,000 scattered strings alone took 0.17 s. A hash lookup of every row among the 306 spellings, combined with the whitespace check in one C-level pass over `str.strip`, needs no factorize and costs 75 ms per column, mostly `isna()` and that pass. The 20 text columns still take 1.4 of the 1.9 s that a run with per-column timings takes on pandas 2.2.

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass. `uv run --isolated --resolution lowest-direct --python 3.12 pytest` passes with numpy 1.26.0, pandas 2.2.0 and pyarrow 14.0.1 and leaves `uv.lock` unchanged. `uv run --isolated --python 3.13 pytest` and the same on 3.14 pass.
- The `thorough` profile passed for this analysis's tests and properties on both version sets. Random fuzzing with 1,000 examples per seed ran the generated-data contract, the no-leak property, the comparison with profile_missing_values and the four rule properties: seeds 1, 2, 6 and 7 on the latest versions and 3, 4 and 5 on the minimum ones. It found the NUL problem above.
- Not run locally: Linux and the 2-vCPU target machine, which ticket 05's CI covers.
