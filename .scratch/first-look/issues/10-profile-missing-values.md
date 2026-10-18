# 10: profile_missing_values

**What to build:** An agent calls `ae.profile_missing_values(df)`, optionally declaring sentinel values per column, and learns the data's true completeness. It sees native missing values and declared sentinel values per column, combined without double counting, with shares over all rows. It also sees how many values each row is missing, which shows whether gaps cluster in some rows.

The analysis lives in `agent_eda.data.missingness`. detect_missing_sentinels reuses its handling of declared sentinel values.

Read these sections of `.scratch/first-look/spec.md`: the profile_missing_values entry under The first-look analyses, and Analysis results. Covers user stories 39 and 90 to 92.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** ready-for-agent

- [x] `sentinel_values` maps column names to declared sentinel values and has no default.
- [x] Per column, the result gives the native missing count, which covers None, NaN, NaT and pd.NA; the sentinel count; the combined missing count without double counting; and the missing share over all rows.
- [x] For the dataset, the result gives the number of complete rows and the distribution of how many selected values each row is missing.
- [x] Empty strings count as missing only when declared as sentinel values.
- [x] Without `sentinel_values`, the result records that none were declared, so a sentinel count of zero can't be mistaken for a check that ran.
- [x] A declared sentinel value that can never match its column's type raises `InvalidArgumentError` naming the likely intended value. For example, "-999" given as text for an integer column suggests -999.
- [x] The result has two tables. `to_frame()` returns the per-column table and takes a `table` literal for the per-row distribution, and the printed summary names both tables.
- [x] Findings cover all-missing columns and declared sentinel values that are present.
- [x] The declared statuses are `completed`, `insufficient_data`, `not_applicable` and `failed`. With no rows, the status is `insufficient_data`, the counts are zero and the shares are null.
- [x] Native missing counts match pandas `isna()` on every accepted dtype, including pyarrow-backed and pandas 3 string columns.
- [x] The docstring has the seven sections in order and passes the contract tests.
- [x] The property suite has a call recipe for it, and the properties hold.
- [x] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [x] `ae.search_analyses("missing values")` and `ae.search_analyses("nulls per column")` return it first. Both rows join the search acceptance table, and every earlier row still passes.
- [x] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [x] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **"No default"** means no sentinel value is assumed: `sentinel_values` is optional and None by default. A column with nothing declared, or with an empty list, gets a null `sentinel_count`, and a `sentinel_values_not_declared` limitation names those fields. Its reason has two forms: one for a call that declares nothing at all and one that names the columns left without values. An empty mapping declares nothing.
- **Result fields.** Each column entry, keyed by name, has `native_missing_count` (what pandas `isna()` counts), `sentinel_count`, `missing_count` (both together), `missing_share` (over all rows) and `sentinel_values`, the declared values with a count each. For all selected columns together the result has `missing_count` and `missing_share` over every selected value, `complete_row_count`, `complete_row_share` and `row_distribution`: one entry per number of missing values that occurs, fewest first, each with `missing_count`, `row_count` and `row_share`. "Missing" always counts declared sentinel values too, and the field name says when it doesn't (`native_`). Without rows every count is 0 and every share null, with a `too_few_values` limitation naming the shares. Each column's status is then `insufficient_data` with code `no_rows`. Data with rows but no columns is `not_applicable`, code `no_columns`, and every row counts as complete.
- **No double counting by construction.** A declared value is matched only against values pandas doesn't count as missing, and a value counts once however many declared values equal it. A declared value that pandas would count as missing itself (NaN, NaT) raises, as does an infinite one. pydantic rejects None and pd.NA as types.
- **Matching rules**, in `agent_eda/_sentinels.py`. A value matches exactly. Text matches with its case and surrounding spaces, so "N/A" never matches "n/a" or " N/A". Numbers match by value whatever their type, so -999 matches -999.0. Datetimes match by the instant they name. In an object column, only values of the declared value's kind ("family": number, text, boolean, datetime, date, timedelta) match, so True never matches 1. A decimal, pyarrow-backed or a `Decimal` object, matches a declared number as written: pyarrow compares decimal128 with a float inexactly, which the random fuzzing below found, so 2379.95 is compared as `Decimal('2379.95')`. Categorical columns match through their categories. Object columns compare with NumPy on the non-missing values, which is 4 times faster than pandas' comparison; an element such as a NumPy array falls back to a per-item loop.
- **Values that can never match** raise `InvalidArgumentError` (`invalid_argument`) before any computation, all problems in one message with the first 5 listed. A value can fail by its kind: the text '-999' in an integer column suggests -999, -999 in a text column '-999', 'false' in a boolean column False, 1 in a boolean column True, '1900-01-01' or a date in a datetime column `pd.Timestamp('1900-01-01 00:00:00')`, a Timestamp in a date32 column `datetime.date(1900, 1, 1)`, and '1 day' in a duration column `pd.Timedelta('1 days 00:00:00')`. Dates parse as ISO 8601 only. It can also fail on the dtype's limits: an integer outside the dtype's range, a fraction in an integer column, a float beyond float32, a datetime beyond datetime64[ns] or finer than its unit, a time zone unlike the column's (with the aligned Timestamp as the suggestion), or a column whose values no sentinel can equal, such as pyarrow lists. An object column holds what its values are: one with only text takes text only, and one without values takes anything. A suggestion is checked against the column too, so 300 for a uint8 column gets none.
- **Keys** follow the column rules: an integer label can be given as itself or as its string form. 0 and '0' in one mapping raise. An unknown name raises `ColumnNotFoundError` with the closest names, and a column that `columns` leaves out raises `InvalidArgumentError`. One value instead of a list works, as do tuples, arrays, Series and NumPy scalars. NumPy datetimes and timedeltas become pandas ones, because `item()` of datetime64[ns] is an int. A value listed twice for one column raises.
- **Registry.** `_normalize` now reads a mapping option generically: keys as column names, each value like an option of the value's type. It also turns NumPy datetime64 and timedelta64 into pandas scalars everywhere. The message for a pydantic error inside a mapping names the path, as in `sentinel_values['age'][1]`, and leaves out pydantic's text for a union, which names only the first member. Tickets 12 and 13 can use both for `declared_schema` and `schema`.
- **Two tables.** `to_frame(table=...)` takes `Literal["columns", "row_distribution"]`, defaulting to the per-column table. An unknown name raises `InvalidArgumentError` from the shared `_check_table` in `results.py`. The text names the other table through `AnalysisResult._text_other_tables()`, which `_text.render` puts into a footer that now wraps. Ticket 16 builds the same in parallel; whoever merges second keeps one of the two.
- **Text.** The facts give the missing count and share over all selected values and the complete rows, then the distribution per row, naming 8 numbers and counting the rest. The table lists the columns with the most missing values first, ties in selected order, so the 10 rows shown are the gappiest. The record and `to_frame()` keep the selected order. The findings' claims are short enough to show whole.
- **Stored NaN.** A pyarrow-backed float column, or a nullable one on pandas 2, can store NaN as a value that `isna()` doesn't count. `native_missing_count` follows `isna()`, as the box says, and a `nan_not_counted_as_missing` limitation names such columns and how to count NaN (`.astype('float64')`). summarize_numeric counts it as missing, so the two analyses differ there; both now call `_numeric.stored_nan_count`.
- **Findings.** `all_missing_columns` is suspected, as in describe_dataset, but counts declared values as missing and says so. `sentinel_values_present` is descriptive and names the columns where declared values occur.
- **New limitation codes:** `sentinel_values_not_declared` and `nan_not_counted_as_missing`.
- **Record caps.** A mapping inside `parameters` now keeps its first 100 entries and gains `<name>_total`, as a list did. Each column's `sentinel_values` keeps 100 too.
- **Tests.** `test_contract.py` gained `KEYED_BY_COLUMN`: a parameter named after its setting, such as `sentinel_values`, may take a mapping keyed by column names (str). Ticket 12's `declared_schema` and ticket 13's `schema` belong there. `test_a_tie_goes_to_the_alphabetically_first_name` now looks for ties anywhere in the ranking for "missing", since an analysis named after missing values leads it alone. The recipe draws sentinel values from each column's own values plus a stand-in of the same type, for every dtype the strategy generates, and a property checks every count against a plain Python loop over the values.
- **For ticket 11.** `_sentinels.declared_values(analysis, data, selected, sentinel_values)` checks a call's declared values and returns them per column. `matches(series, value, missing)` gives the rows that hold one value, `json_value` gives its record form, and `literal` gives its Python form for messages. `holds` and `family` describe what a column and a value are. The messages name the analysis and stay neutral, so detect_missing_sentinels can raise the same ones.
- **Glossary gap:** "native missing value" (what pandas `isna()` counts: None, NaN, NaT, pd.NA) is used in the spec and here but isn't in GLOSSARY.md. Noted for /domain-modeling.

Measurements, on the M3 Max with Python 3.12.11 and 2 threads, while other agents loaded the machine:

| Versions | Benchmark, default arguments | Sentinel values declared for all 100 columns |
| --- | --- | --- |
| pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0 | 0.08 and 0.08 s (0.23 s on a busier run) | 0.32 s |
| pandas 2.2.0, numpy 1.26.0, pyarrow 14.0.1 | 0.34 and 0.35 s (0.81 s on a busier run) | 0.95 s |

With values declared for every column, pandas 2.2 took 3.5 s before object columns were compared with NumPy and checked with `infer_dtype`; a profile put most of it in the 20 object text columns. Each of them now takes 37 ms with two declared values and 14 ms with none.

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass. The minimum-versions run passed with numpy 1.26.0, pandas 2.2.0 and pyarrow 14.0.1 and left `uv.lock` unchanged. `uv run --isolated --python 3.13 pytest` and the same on 3.14 passed.
- The `thorough` profile passed for this analysis on both version sets. Random fuzzing with 3 seeds and 2,500 examples each ran the contract checks and the count property on both version sets. It found the decimal comparison above, which `test_a_decimal_matches_a_declared_number_as_written` now covers; after the fix all 7,500 examples passed on each version set.
- Linux and the 2-vCPU target weren't run locally; ticket 05's CI covers them.
