# Spec: correctness

Status: ready-for-agent

## Problem Statement

agent-eda promises correct, deterministic results that never leak masked values and never hang. The suite of 2,155 tests passes on every supported version. Yet the October 2026 audit (`.scratch/audit/findings.md`) reproduced wrong results, crashes, hangs, leaks and nondeterminism the suite doesn't catch. Every P1 below was re-run by the coordinator.

- **Wrong counts and verdicts.**
  - Text with a NUL character merges distinct values in 7 analyses: `''` counts as `'\x00a'`. `validate_column_types` can then call a column `passed` while some values don't convert (AN1-1, DES-1, STD-2).
  - Distinct integers beyond 2⁵³ are called constant, and `profile_target` reports a false `constant_target` violation (STD-6).
  - `std` is 0 for non-constant values below about 1e-154 (AN2-2).
  - pyarrow dictionary columns, the Arrow form of a categorical, count as "other" and are skipped or refused (AN1-5, AN2-7).
  - Ordinary data is misjudged:
    - `05.50` is read as a code (AN1-2);
    - grades A, B, C and F are called boolean (AN1-3, AX-13);
    - dates stored as text are flagged as identifiers (AN1-4);
    - a few duplicate rows hide the ID column (AX-12);
    - float measurements are offered as primary keys (AX-11);
    - the correlation ratio is inflated for many-level columns (AN2-5);
    - a declared class with no rows passes unnoticed (AN2-6).
- **Different output for the same call.** Labels that are sets or frozensets, and `np.str_` text, give different JSON in different processes (AN2-3, CORE-2, SPEC-1, STD-4, STD-5).
- **Crashes and hangs.**
  - `find_candidate_keys` raises `TypeError` on a column mixing ints and lists (AN2-1).
  - A huge `max_key_size` hangs it, and over MCP it freezes the whole server (AN2-4, CORE-3, MCP-3).
- **Threads.** Calls from several threads, which agent frameworks make, corrupt the caller's warning filters, swallow later warnings and move limitations between results (CORE-1).
- **Masking.**
  - `columns=df["email"]` puts raw emails into the error message (CORE-4).
  - `profile_missing_values` shows declared values of sensitive columns, while discovery says it shows no raw values (CORE-19, SPEC-7).
  - A sensitive name that differs from the data's only by case or Unicode form shows raw values, while the text claims none is shown (CORE-6).
  - The MCP docs imply that tools without masking show no raw values, while `summarize_numeric` shows a sensitive column's extremes (MCP-11).
- **Reading files over MCP.**
  - Files longer than one pandas read chunk get columns of ints *and* strings (STD-1, MCP-1).
  - A trailing delimiter shifts every column left and turns the first column into row IDs (MCP-2).
  - One-column files lose their empty rows (STD-3).
  - `01234` becomes `1234` (SPEC-2, MCP-12).
  - Repeated headers are silently renamed (SPEC-6).
  - The read cache ignores the file format (MCP-14).
  - Path problems arrive as "a bug in agent-eda" (MCP-4, MCP-5).
  - `~` works in one path argument out of three (SPEC-13, MCP-8).
- **MCP calls.**
  - Date sentinels that the input schema offers are refused (DES-3, SPEC-4, MCP-6, STD-10).
  - Error messages give Python fixes such as `pd.Timestamp(...)`, `task='...'` and `project.save(...)` (DES-7, SPEC-3, MCP-7, STD-8).
  - Tool descriptions show `res.columns[...]` and `df.loc[...]` (DOC-10).
- **Errors.**
  - Building a `ColumnSpec` from JSON or a non-string key leaks raw pydantic errors or a bare `TypeError` (AN1-11, DES-4, CORE-18, SPEC-9).
  - The fix offered for a path passed as data itself raises (AX-4, CORE-7).
  - An unknown `method` silently computes Kendall (STD-46).
  - Discovery functions raise bare `TypeError`s.
- **Result objects.**
  - `==` raises for three result classes (CORE-5).
  - `to_json()` fails on some row IDs (CORE-14).
  - `print(res)` can crash on lone surrogates (CORE-10).
  - `Project.save` can hang on a FIFO, and it writes files that `Project.load` refuses (CORE-8, CORE-9).

The suite misses these for structural reasons:
- shared rules were fixed in one module and stayed broken in others (DES-1);
- the MCP tests' reference read repeats the server's own read (STD-1);
- the masking property almost never exercises `find_duplicate_rows`, and a planted leak passed the suite (TEST-1);
- cross-process determinism is tested for 2 of 11 analyses (TEST-2);
- the generic contract checks cover less than the rules (TEST-3, TEST-4, TEST-7, TEST-9).

## Solution

Fix each problem once, in the shared code that causes it, and test it across every analysis:
- one rule for which values count as equal;
- one text form for set-like labels;
- one rule for when a column is constant;
- one validation base for declarations;
- one parser behind suggestions;
- one way to read MCP arguments as their schema describes them;
- one CSV reading rule that keeps values as written.

The contract checks grow to cover every result rule in AGENTS.md, and a cross-process test covers every analysis. Analyses that judge ordinary data follow the rules settled in the 2026-10-19 grilling session.

## User Stories

### Right answers

1. As an agent, I want distinct values counted as distinct whatever characters they hold, NUL included, so that counts, keys, duplicates and type verdicts are right.
2. As an agent, I want a column called constant only when its stored values are all equal, so that distinct IDs beyond 2⁵³ aren't called constant.
3. As an agent, I want the standard deviation of very small values computed exactly, so that a varying column never shows std 0.
4. As an agent, I want a limitation when decimals beyond 2⁵³ lose precision, as integers already get one.
5. As an agent, I want NaN stored as a value counted as missing in every analysis, with a limitation where that matters, so that one column never reads constant in one result and empty in another.
6. As an agent loading Parquet with the pyarrow backend, I want dictionary-encoded, time, binary and sparse columns handled like their pandas equivalents.
7. As an agent, I want type inference that doesn't call letter grades boolean, `05.50` a code, dates identifiers, or an ID column ordinary because a few rows repeat.
8. As an agent looking for keys, I want float measurements left out unless I name them.
9. As an agent, I want the association between a many-level column and a number adjusted for bias by default, so that noise doesn't rank first.
10. As an agent profiling a target, I want to hear when a declared class has no rows, when a two-value target is profiled as regression, and when class labels differ only by case or spaces.
11. As an agent declaring sentinels, I want `-999` found in a column that also holds `"N/A"`, `#N/A` and `<NA>` recognised, and `" N/A"` matched, so that the stand-ins pandas knows aren't missed.

### Same answer every time

12. As an agent, I want identical calls to give byte-identical JSON in different processes for every analysis and every kind of value, sets and objects included.

### Never crash, never hang

13. As an agent, I want columns mixing scalars and lists handled as documented in every analysis, never with a `TypeError`.
14. As an agent passing a huge limit to mean "no limit", I want the analysis to finish, and the MCP server to stay responsive.
15. As an agent framework that calls tools from threads, I want concurrent calls to give correct results and leave my process's warning settings alone.

### Masking

16. As an agent with masking on, I want `profile_missing_values` to mask the declared values of sensitive columns, as `detect_missing_sentinels` does.
17. As an agent, I want a warning when a sensitive name differs from a data column only by case or Unicode form, instead of a claim that nothing sensitive is shown.
18. As an agent that passed a column's values where its name belongs, I want an error that says so and shows no values.
19. As an MCP user, I want the docs to say that numeric statistics and index labels are never masked.

### Files and calls over MCP

20. As an MCP caller, I want my CSV read as written: one type per column whatever the file's length, no column turned into the index, empty lines of a one-column file kept as missing, codes like `01234` kept as text, and repeated header names kept and reported.
21. As an MCP caller, I want path problems to come back as tool errors that name the cause, never as an internal error.
22. As an MCP caller, I want `~` to work the same in every path argument.
23. As an MCP caller, I want to declare date and duration sentinels as the input schema offers them.
24. As an MCP caller, I want no Python in what I read: error messages, tool descriptions and input schemas.

### Errors and objects

25. As an agent, I want every way of building a declaration (`Project`, `DataPolicy`, `ColumnSpec`) to raise the package's own error with its fix.
26. As an agent, I want suggestions ("did you mean -999?") that the analyses then accept.
27. As an agent, I want each argument mistake to get a message that says what's wrong, what's allowed and a fix that works. That includes a path passed as data, an `id_column` that names the index, an unknown method, and wrong keywords to discovery functions.
28. As an agent, I want results to compare with `==`, survive a pydantic round trip, serialize every row ID, and print any text.
29. As an agent, I want `Project.save` to write only files `Project.load` accepts, atomically, never blocking on a special file, and to record a format version.

### Tests that catch regressions

30. As a maintainer, I want the contract checks to cover every result rule in AGENTS.md for every analysis, so that a new analysis can't break one silently.

## Implementation Decisions

### Value identity

- One private helper answers "which values are equal" for every analysis: for example, `_values.value_codes(values, *, unhashable=...)`. It returns codes with -1 for missing, the unique values and how they were compared.
- It guards against pandas' NUL problem by factorizing object text through the object hash table, with the marker `_masking.mask_values` already uses. The guard costs nothing when no value holds `'\x00'`, which one vectorized check establishes.
- It owns the policy for unhashable values. The parameter is one switch, `"by_content"` or `"unsupported"`, and each analysis keeps its documented behaviour:
  - `find_duplicate_rows` compares by content;
  - `summarize_categorical`, `find_candidate_keys` and `compute_association_matrix` leave the column out;
  - `describe_dataset` and `infer_column_semantics` leave the distinct count null.
- Every factorize or distinct count on raw column values (DES-1 lists about 18 sites in 12 modules) calls it. pandas' equality stays: 1, 1.0 and True are one level.
- Hashability is tested, not inferred from `infer_dtype`, so `mixed-integer` columns holding lists are caught (AN2-1).
- Sets holding missing values compare by content, as the docs say (STD-16).

### Deterministic records

- `_levels.label_key`, `_levels.record_label` and `_text.label_text` give set-like values one form: items sorted by their JSON text, recursively inside tuples, as `results.row_id_json` already does. An object whose repr carries a memory address is written as `<TypeName object>`. Ties keep the order of first appearance.
- `describe_dataset` sizes every `str` instance, subclasses such as `np.str_` included, without Python's cached UTF-8 copy.
- A cross-process test, parametrized by analysis, runs each recipe on `EDGE_FRAMES` and on generated frames in two subprocesses with different `PYTHONHASHSEED` values. It compares `to_json()`, `str(res)` and every `to_frame()` table. The frames include set, frozenset and `np.str_` values.
- The README says results are byte-identical in the same environment, BLAS thread count included (SPEC-16). The agent-workflow spec changes that sentence.

### Exact numbers

- One shared rule decides constancy on the stored values: exact for integer and decimal dtypes, on the stored floats for float dtypes, and with NaN stored as a value counted as missing. `describe_dataset`, `summarize_numeric`, `infer_column_semantics` and `profile_target` call it.
- The variance is computed on values scaled by a power of two whenever the mean square underflows, as `_association._scaled` does. Underflow is no longer ignored in silence.
- Decimal and Arrow decimal columns get the `float_precision` limitation when values beyond 2⁵³ lose precision.
- NaN stored as a value, such as NaN in a pyarrow double column or in object columns, is missing in every analysis. A limitation names the columns where the stored NaN differs from pandas' `isna()` (STD-14).

### Keys, limits and threads

- `find_candidate_keys` bounds its combination count by `min(max_key_size, number of columns)` before searching. The budget message suggests the largest `max_key_size` that fits `max_combinations`.
- `find_candidate_keys` leaves non-integral float columns out by default, with a limitation that names them. Naming them in `columns` includes them. This changes first-look ticket 14.
- Warning capture runs under one module-level lock, so concurrent calls run their captures one after another and never touch the caller's filters or `showwarning`. The docs say concurrent calls are safe. A test runs analyses in a thread pool and then checks `warnings.filters`, `warnings.showwarning` and each result's limitations.

### Masking boundary

- `profile_missing_values` takes `project=` and `mask_sensitive=`. It masks the declared sentinel values of sensitive columns in `parameters` and in its tables, as `detect_missing_sentinels` does, and becomes a masking analysis everywhere: describe_analysis, the MCP tool's `project_path`, the README and the contract.
- When a sensitive name is absent from the data but matches a column after `casefold()` and NFC normalization, the masking text names that column ("did you mean 'café_owner'?"). It doesn't claim that no sensitive value is shown. Matching stays exact, as documented.
- An option ending in `column` or `columns` refuses a Series, an Index or an array whose items aren't all column labels. The message says it looks like a column's values, shows none of them, and gives the fix (`columns=['email']`).
- Messages that list unknown names show at most 3, each cut to 40 characters, as `_value_text` cuts values.
- With masking on, no message quotes a value of a sensitive column.
- Over MCP, an internal error with masking on names only the exception type.
- The README's MCP section and the MCP instructions say that numeric statistics and index labels are never masked, and that analyses without masking don't read the project.
- The no-leak property gets `sensitive_options` for `find_duplicate_rows` that always repeat rows and set a sensitive `id_column`. A floor asserts that most examples would leak without masking (TEST-1).

### Reading CSV files over MCP

- `pd.read_csv` is called with these settings:
  - `low_memory=False`, so each column is typed once;
  - `index_col=False`;
  - `keep_default_na=False, na_values=[""]`, as today;
  - `skip_blank_lines=False` when the header has one column.
- A row with extra non-empty fields raises `DataFileError` naming the line. A trailing empty field is accepted.
- A column is read as numbers or booleans only when every non-empty field parses as one, and none is a whole number written with a leading zero. Otherwise the column stays text. The leading-zero rule is the one `_parse` uses: an integer part of two or more digits that starts with 0. Every other column reads as pandas reads it.
- Repeated header names stay repeated in the DataFrame. `describe_dataset` then reports them, and the other analyses raise their existing error with an MCP fix: rename the repeated columns in the file.
- The cache key adds the file format.
- The data_path description, the instructions and the README say how types are read, with the leading-zero rule and the treatment of booleans and `inf`. They correct "Parquet keeps its dtypes" for files without pandas metadata (STD-11).
- The tests write files longer than one read chunk and compare against an independent reference read, never the server's own call.

### Paths and files over MCP

- Resolving, `stat` and `open` for `data_path`, `table_path` and `project_path` turn `OSError`, `ValueError` and `RuntimeError` into coded tool errors that name the cause: `data_file_not_found`, `unreadable_data_file` or `invalid_argument`. The causes include permission, a name that is too long, a symlink loop, an embedded NUL and a read-only volume.
- `table_path` problems are found before the analysis runs wherever possible. A failed write removes the partial file.
- `~` expands in all three path arguments.
- `table_path` checks and opens the path as given, so a dangling symlink is refused (MCP-13).
- The reader's and writer's exception sets include the errors MCP-5 found: `KeyError`, `TypeError`, `OverflowError` and pyarrow's.
- Reader messages name the actual cause (MCP-16):
  - not a regular file;
  - permission denied;
  - duplicate Parquet fields;
  - the last suffix for an unknown extension;
  - a hint about the separator when the only column contains `;`.
- An unknown MCP tool name gets the closest names and the list of MCP tools (MCP-17, STD-40).
- `table_path` CSVs write the empty string as `""`, so it reads back as text, not missing (SPEC-14).
- The `table_path` line keeps to the text caps (STD-41).
- `create_server(analyses=...)` becomes private, for example `_create_server`; it was public only as a test hook (round 3, Q7c).

### MCP arguments

- The MCP server validates arguments in JSON mode, through the registry's arguments model, so an ISO 8601 string becomes a `datetime`, `date` or `timedelta` wherever the annotation allows one.
- For `sentinel_values`, whose items also allow text, a string declared for a datetime, date or timedelta column is read as ISO 8601 when it parses. This applies over MCP only, where JSON can't carry those types.
- A contract test sends every MCP tool an example of every `format` its input schema offers, and expects no `invalid_argument`.

### No Python in what MCP callers read

- **The known leaks are reworded for MCP:**
  - sentinel examples become JSON, such as `{"sentinel_values": {"age": [-999]}}`;
  - "did you mean" for a datetime gives the ISO string;
  - `task=` becomes `{"task": "multiclass_classification"}`;
  - `project_path` messages name `project_path` and say how to write the TOML file, with no `project.save` step;
  - echoed values are written as JSON (`null`, `true`).
- **Tool descriptions drop** `res.columns[...]`, `df.loc[...]`, `columns=None` and `table="matrix"`. Each analysis's docstring sections are written so that they read correctly for both surfaces.
- **A guard test** scans every MCP tool description and input schema for Python syntax. It also scans the messages of a table of mistakes that reaches every error code each MCP tool can raise.
- `tests/test_mcp_project.py` stops expecting the Python fix for a missing project file.
- Record text and output-schema field descriptions are the agent-experience spec's.

### Declarations and suggestions

- **Declarations:**
  - `Project`, `DataPolicy` and `ColumnSpec` share one validation base (for example, `_declarations.py`). Each way of building one raises `InvalidArgumentError` with the package's message: the constructor, `model_validate`, `model_validate_json`, `model_validate_strings`, non-string keys, and `model_copy(update=...)`.
  - NumPy scalars are accepted wherever Python scalars are.
  - The three ways `Project`'s types are written agree (STD-37).
- **Suggestions:**
  - One parser (for example `_parse.read_as(text, family)`, built on `_parse`'s rules) produces every suggestion for sentinel values and expected labels. A suggestion is always a value the analyses accept (DES-10).
  - Declared values are compared by type family and value, so `True` and `1` can both be declared (AN1-7, STD-19).

### Argument messages

- A `str` or `os.PathLike` passed as data suggests `pd.read_csv(path)` or `pd.read_parquet(path)`. A pyarrow RecordBatch or Table suggests `.to_pandas()`, and `None` says the variable holds no data.
- An `id_column` equal to the index's name says that index labels are already the row IDs.
- `project=` or `mask_sensitive=` passed to an analysis that shows no raw values says why it isn't taken.
- `columns=[None]` points to `columns=None`.
- A set inside a mapping gets a fix for that key.
- Messages say "an object".
- A NaN label is reported before duplicates are.
- Type annotations and pydantic phrases ("Input should be...") don't reach the message (AX-25).
- Closest-name suggestions find case-only typos and name the misspelled key (STD-35).
- An unknown `method` raises instead of computing Kendall, and dead defaults go (STD-46).
- `describe_analysis`, `search_analyses`, `Project.load` and `Project.save` raise `InvalidArgumentError` (`invalid_call`) for wrong keywords, with the closest name. This changes first-look ticket 09.
- Table-driven tests, one for Python and one for MCP, assert the code and key fragments of every message, including the paths TEST-5 lists.

### Judging ordinary data

These rules were settled in round 2, Q9, and round 3, Q6.

- **Type inference** (`infer_column_semantics`, and `validate_column_types` where it shares the rule):
  - "Boolean" only when the values that read as booleans cover every distinct value, and a constant column is never boolean.
  - The leading-zero rule applies to whole numbers only.
  - Date-like text is never `identifier_like`.
  - `identifier_like` is judged after dropping exact duplicate rows.
  - A declared type that ties with the top candidate is a match.
  - A declared `datetime_format` is honoured.
- **Arrow and rare dtypes.** `column_kind` classifies pyarrow dictionary dtypes by their value type, so a dictionary of strings is categorical. Arrow `time64`, binary and sparse columns get a defined kind or a documented "other" with a reason. The generated test data gains dictionary, time64 and binary columns, and categoricals of booleans, datetimes and mixed values (TEST-6).
- **`compute_association_matrix`:**
  - The default mixed-pair measure becomes a bias-adjusted correlation ratio: the square root of epsilon squared, clipped at 0. The plain correlation ratio stays available as a method.
  - A pair left null because each row is its own level gets its own reason, not "too few complete rows" (AX-14).
  - The pair budget accounts for the cost of rank methods.
  - A pair's value doesn't depend on which other columns are in the call; if it can't be made independent, the docs say so (STD-44).
  - The `budget_exceeded` footer doesn't offer an empty matrix (AN2-12).
- **`profile_target`:**
  - a violation `missing_expected_classes` for declared classes with no rows, with those classes in the class table at count 0;
  - a suspected finding for a regression target with two distinct values;
  - a suspected finding for class labels that differ only by case or surrounding spaces;
  - `expected_labels` working with datetime and period categories (STD-13).
- **Sentinels:**
  - Number text in a text column gets the numeric-code rule.
  - pandas' default NA strings join the built-in text list: `#N/A`, `#N/A N/A`, `#NA`, `<NA>`, `-NaN`, `-nan`, `1.#IND`, `1.#QNAN`, `-1.#IND` and `-1.#QNAN`.
  - Text stand-ins match after stripping surrounding whitespace.
  - Every record writes datetimes in one ISO 8601 form. Decimals are written as numbers when float64 holds them exactly and as exact text otherwise, the same rule the agent-experience spec applies to integers above 2⁵³. Declared values that print the same can still be told apart, and `profile_missing_values`' table keeps every count (AN1-9, STD-17, STD-38).
- **`find_candidate_keys`' docstring** stops claiming its counts are what `df.duplicated` gives for an object column holding None and NaN (STD-15).

### Result objects

- **Equality:** `==` on results compares the records and never raises (CORE-5).
- **pydantic round trips:** `model_validate(model_dump())` either keeps the full tables or makes `to_frame()` raise a clear error (CORE-13).
- **`row_id_json`** falls back to `str(value)` for values it can't write as ISO, such as timedeltas in years or months (CORE-14).
- **The text template** escapes lone surrogates (category Cs) and bidi controls (category Cf), and `to_json()` stays valid UTF-8 (CORE-10).
- **`Project.save`:**
  - writes through a temporary file and a rename;
  - refuses a target that isn't a regular file;
  - refuses to write more than `Project.load` accepts;
  - writes `format_version = 1`.

  `Project.load` reads files without the key as version 1, and refuses a newer version with a message naming the version that wrote it (round 3, Q7b).

### Contract checks for every analysis

`check_contract` gains these checks, so every recipe and every generated frame runs them:
- every list in the record holds no more than its cap, with its `_total`;
- every finding and limitation field path resolves;
- every finding's assessment equals the one its docstring documents;
- `to_dict()` equals `json.loads(to_json())`;
- the record carries no timestamps and no HTML.

The suite's own checks get stronger:
- a test drives the template to its 40-line cut (TEST-4);
- `test_contract.py` checks each Args entry's type, choices and meaning of None, and resolves See Also against `list_analyses()` (TEST-9);
- the one-seam guard also catches `importlib` and string module names (TEST-10);
- the meta-test covers ties and mixed types (TEST-11);
- the tests that can't fail are fixed (TEST-15, STD-60);
- `anyio` and `jsonschema` are declared in the dev group (STD-64);
- generated data no longer shares mutable objects or orders a set by hash (STD-63);
- every `# noqa` and `# ty: ignore` carries a reason, and a test checks it (STD-22);
- STD-61's coverage gaps and STD-62's implicit tolerances are fixed.

## Testing Decisions

### The seam

The public `agent_eda` API and the MCP protocol, through the SDK's in-memory client and real stdio, as today. Tests import only `agent_eda` (plus `mcp` and `jsonschema` in MCP tests). Shared rules are tested through every analysis that uses them, with properties that loop over `ae.list_analyses()`.

### What makes a good test here

- Write each fix test-first, starting with the reproduction from the audit, which every ticket quotes.
- Compare against an independent reference: Python's `==`, NumPy or SciPy with a stated tolerance, or an independent `pd.read_csv` call.
- Where a rule spans analyses, the test spans analyses too. One-analysis tests are what let these bugs through.

### Test groups

1. **Value identity:** the NUL and mixed-hashability kinds in `data_frames()`, and level, group, class and key counts against a count by Python `==`.
2. **Determinism:** subprocesses with different hash seeds for all 11 analyses.
3. **Numbers:** constancy, tiny-value variance and decimal precision against exact references.
4. **Limits and threads:** a huge `max_key_size` finishes quickly, and a thread pool leaves warnings untouched.
5. **Masking:** the no-leak property with a floor for every masking analysis; values passed as names; near-miss names; the new `profile_missing_values` masking.
6. **MCP reading:** files longer than a chunk, trailing delimiters, one-column files, leading zeros, repeated headers, the format in the cache key.
7. **MCP paths and arguments:** OS error causes, `~`, symlinks, partial files, ISO arguments for every schema format.
8. **MCP wording:** the guard over descriptions, input schemas and the mistake table.
9. **Declarations and messages:** every construction route of every declaration model, and the Python and MCP message tables.
10. **Judgement rules:** each rule with its audit example and its counter-example, such as a real boolean column, a real `0123` code and a real ID.
11. **Result objects and project files.**
12. **Contract:** the new generic checks, each run once against a mutant to show it fails.

## Out of Scope

- Reader options for MCP calls (`na_values`, separator, encoding, dtypes). The README documents how to prepare such files. Adding the options would grow every MCP tool's schema.
- DataFrames passed by role (`before=`, `after=`) and their `<role>_path`, until the first such analysis.
- Masking index labels or numeric statistics: documented as never masked.
- Matching sensitive names loosely: matching stays exact, and near misses are reported.
- Changing `max_categories` to return `budget_exceeded`: it keeps skipping columns, as documented.
- Rewriting the error wording as data, which the deepening spec covers. This spec fixes the leaks and adds the guard.

## Further Notes

- The plan was settled in a grilling session on 2026-10-19. The decisions behind this spec are round 1, Q7 to Q9, round 2, Q1 and Q9, and round 3, Q3, Q6 and Q7 b to d.
- Finding IDs refer to `.scratch/audit/findings.md`; the STD details are in `.scratch/audit/standards-review.md`.
- Changes to earlier decisions:
  - first-look ticket 14 (every dtype as a key candidate);
  - first-look ticket 12 (ties and `identifier_like`);
  - first-look ticket 09 (bare `TypeError` from discovery functions);
  - the first-look spec's list of non-masking analyses (`profile_missing_values` now masks);
  - the mcp-server spec's reading rules (pandas defaults).

  ADR 0006's intent that values reach the analyses as written stands, and the reading rules now serve it.
- Test fixes land with the code fixes they guard, ticket by ticket. Ticket 18 adds the generic checks that guard every analysis.
