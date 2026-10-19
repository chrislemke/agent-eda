# STD: Standards review of the whole agent-eda codebase

I covered every Python file under src/ (42 modules, 19,931 lines, plus `py.typed` and the two subpackage `__init__` docstrings) and tests/ (45 files, 19,522 lines). I first read the standards sources: AGENTS.md, GLOSSARY.md, the six ADRs, pyproject.toml, .pre-commit-config.yaml, both specs and the Comments of all 25 tickets.

I read the core infrastructure and 13 test files myself:
- `__init__`, `_registry`, `_run`, `_columns`, `_text`, `_annotations`, `_docstring`, `_values`, `_wording`, `_version`, `errors`, `mcp` and `_mcp_instructions`;
- conftest, smoke, smoke_mcp, test_suite_rules, test_ci_workflow, test_warning_policy, test_package, test_readme, test_contract, test_properties, test_analysis_result, test_mcp_server and test_mcp_errors.

Ten forks of this reviewer each read every line of one slice:
- results.py;
- the validation analyses with `_parse` and `_conversions`;
- missingness and targets with `_sentinels` and `_sentinel_rules`;
- the statistics analyses with `_association`, `_numeric` and `_levels`;
- describe_dataset, keys and duplicates with `_keys`, `_row_ids`, `_masking` and `_values`;
- discovery and project with `_search` and `_project_file`;
- the MCP modules;
- the remaining tests, in three slices.

I also ran these cross-cutting checks:
- every analysis's signature, statuses, finding codes and docstring sections, read from the registry (std/sigs.py, docs_check.py, seealso.py);
- greps for glossary Avoid-list words, NumPy-2-only APIs, `inplace`, `__file__`, randomness, module-level mutables, `warnings.*`, noqa and ty-ignore;
- an inventory of error codes;
- a walk of every record on edge frames for null values that no limitation names (std/nulls*.py);
- all 11 analyses run under PYTHONHASHSEED 1-7 (std/determinism.py);
- missing and distinct counts compared across analyses (std/distinct.py, storednan.py);
- scans of the MCP schemas, and MCP calls through the in-memory client.

All Python ran with `uv run --frozen`. My scripts are in std/ and the forks' in its f1–f10 subfolders. `git status` stayed clean. Writing report.md was refused ("Subagents should return findings as text"), so this message is the report.

There are 64 findings: 6 P1, 14 P2 and 44 P3.

### STD-1: The MCP reader parses CSVs in chunks, so in ordinary files one column holds ints in the first chunk and strings after it, and the analyses count each value twice
- Kind: bug
- Severity: P1. The numbers are wrong on plausible data. pandas sizes a chunk at about 2^20 divided by the column count: 16,384 rows at 50 columns, 262,144 at 2. Any file longer than one chunk is exposed if a code or ID column, or a numeric column holding 'N/A', has a non-numeric value after the first chunk.
- Hard violation or judgement call: Hard violation of AGENTS.md Warnings ("Data problems never become Python warnings"): pandas' DtypeWarning goes to the MCP server's stderr. It also defeats the mcp-server spec's reading rules and user story 13, which promise that stand-ins reach the analyses as values and that results agree with the same analysis on the same values. The spec chose "pandas' default C parser", not chunk-wise type inference.
- Where:
  - src/agent_eda/_data_files.py:105-112: `pd.read_csv(..., keep_default_na=False, na_values=[""])` leaves `low_memory` at its default, True.
  - Test gap: every CSV in tests/test_mcp_*.py has at most 6 rows, and the tests' Python reference reads repeat the server's call (test_mcp_server.py:55, test_mcp_errors.py:225, test_mcp_fixes.py:301, test_mcp_tools.py:325, test_mcp_tables.py:385, test_mcp_project.py:473). So the bug can't fail a test.
- Evidence: Confirmed.
  - std/widecsv.py: 20,000 rows by 50 columns, with `x7` at row 18,000 of `c0`. `DataReader().read` raises `['DtypeWarning']` and returns `{int: 16384, str: 3616}`. summarize_categorical then reports `cardinality 15 True`, against 8 true values.
  - std/probe_mcp_csv.py: 300,001 rows by 1 column, over MCP (`create_server()`, in-memory client). summarize_categorical reports `cardinality 15 mixed True ['int', 'str']` and findings `['mixed_value_types']`.
  - f7/chunks.py: a `score` column of `i % 97` with one 'N/A' at row 399,995 gives `cardinality: 195` against 98. infer_column_semantics flags `['numbers_as_text', 'mixed_value_types']`. With `low_memory=False` the column is all str, with 98 values and no warning.
- Why it matters: An MCP caller gets split levels (5 and '5'), spurious mixed-type and numbers-as-text findings, missed duplicates and split validate_column_types counts, all without a limitation. The results depend on where pandas cut the file, and the warning goes to a stderr the agent never sees.
- Fix direction: Read with `low_memory=False`. Add a test on a file wider and longer than one chunk, whose reference read uses `low_memory=False`.

### STD-2: Object text with an embedded NUL is merged with other text: '' and '\x00a' count as one value in seven analyses, and validate_column_types can call a column with invalid values "passed"
- Kind: bug
- Severity: P1. The counts and an assessment are wrong. The trigger is a NUL inside object text; under pandas 2.2 every CSV text column is object dtype.
- Hard violation or judgement call: Hard violation: the results claim counts, groups, keys and assessments the data doesn't support. First-look tickets 11 and 17 record the pandas factorize NUL problem as an open point for three analyses, not as a decision. The other four aren't mentioned anywhere.
- Where: every `pd.factorize` of object text.
  - src/agent_eda/data/provenance.py:281 (describe_dataset)
  - data/duplicates.py:392 (find_duplicate_rows)
  - _keys.py:93 (find_candidate_keys)
  - _levels.py:135 (summarize_categorical, profile_target, infer_column_semantics)
  - statistics/association.py:569
  - _conversions.py:282 (validate_column_types)
- Evidence: Confirmed by fork scripts f5/nul.py, f4/nul.py and f2/probe1.py.
  - Object column `t=["", "\x00a", "x"]` with `n=[1,1,2]`:
    - describe_dataset gives `distinct_count 2`; the true count is 3.
    - find_duplicate_rows gives the groups `[[0, 1]]`; there are none.
    - find_candidate_keys finds no key.
  - summarize_categorical gives `cardinality: 2 [('', 2), ('b', 1)]`, against `value_counts` `{'': 1, '\x00a': 1, 'b': 1}`.
  - `validate_column_types(pd.DataFrame({"x": pd.Series(["1", "1\x00x", "2"], dtype=object)}), schema={"x": "integer"})` gives `compatible 0 convertible 3 invalid 0 passed`. With `dtype="string"` it gives `convertible 2 invalid 1 violation`.
  - infer_column_semantics gives `distinct 2 integer_share 1.0` on object text, and 3 and 0.667 on string.
  - `_masking.mask_values` already avoids the bug with a non-text marker.
- Why it matters: The same values give different answers depending on dtype. An agent is told about duplicates that don't exist, misses a real key, or accepts a column of unconvertible values.
- Fix direction: Write one shared helper that factorizes object values with the non-text marker `mask_values` uses, or through `_values.hashable_values`, and call it from all six sites.

### STD-3: In a one-column CSV or TSV, the MCP reader drops empty fields as blank lines instead of counting them as missing
- Kind: bug
- Severity: P1. Rows and missing values vanish. The trigger is narrow: a one-column file, such as a label file.
- Hard violation or judgement call: Hard violation of the mcp-server spec's reading rules ("only an empty field is missing"; row IDs are "0-based data-row numbers").
- Where: src/agent_eda/_data_files.py:105-112, where `skip_blank_lines` keeps its default, True.
- Evidence: Confirmed. The file `score\n1\n\n3\n\n5\n` has 5 data rows, 2 of them empty. `DataReader().read` returns `3 [1, 3, 5]`, and profile_missing_values reports `row_count 3, missing_count 0`. f7/blank.py shows the same over MCP.
- Why it matters: Row counts, missing counts and shares are wrong. profile_target on a label file loses its missing labels, and every row ID after a gap points at the wrong data row.
- Fix direction: Read with `skip_blank_lines=False`, and add a one-column test.

### STD-4: Labels that are sets, frozensets or objects without a stable repr reach the record through str(), so identical calls give different JSON in different processes
- Kind: bug
- Severity: P1, for nondeterminism. The trigger is narrow: such values in an object column, from Python callers.
- Hard violation or judgement call: Hard violation of AGENTS.md Implementation ("output order never follows hash or set iteration, so identical calls give byte-identical JSON across processes"). It also contradicts `_levels.py`'s own docstring. The cross-process test covers one describe_dataset call only (tests/test_packaging.py:167-185, test_mcp_server.py:158-170).
- Where: src/agent_eda/_levels.py:393-404 (`record_label` returns `str(value)`) and :362-371 (`label_key` sorts other types by `str(value)`). Reached from summarize_categorical levels, profile_target classes and validate_column_types `invalid_values`.
- Evidence: Confirmed.
  - std/det_fs.py with PYTHONHASHSEED 1-4: summarize_categorical's first level is `"frozenset({'red', 'blue'})"` under seeds 1 and 4 and `"frozenset({'blue', 'red'})"` under seeds 2 and 3. profile_target's classes change with it.
  - std/determinism.py with seeds 1-7 on a mixed frame: only validate_column_types' record changes, with 4 distinct hashes. Its `invalid_values[0].value` is either `"{'a', 'b'}"` or `"{'b', 'a'}"`.
  - f4/fset2.py: an object without `__repr__` is recorded as `'<__main__.Tag object at 0x10349d550>'`.
  - `results.row_id_json` already sorts set items for row IDs (first-look ticket 07), so the two serializers disagree.
- Why it matters: An agent that diffs reruns or caches by record hash sees changes that aren't there, and tied levels can reorder.
- Fix direction: In `record_label` and `label_key`, give set-like values one canonical text with items sorted by their JSON, as `row_id_json` does, and give objects whose repr carries an address a type-name form. Extend the PYTHONHASHSEED test to every analysis.

### STD-5: describe_dataset gives different memory_bytes for two identical calls when an object column holds non-ASCII np.str_ values
- Kind: bug
- Severity: P1, for nondeterminism. It needs `np.str_` objects in an object column, which pandas 2 keeps when a frame is built from NumPy string scalars.
- Hard violation or judgement call: Hard violation of AGENTS.md Implementation (identical calls give byte-identical JSON). This is the bug first-look ticket 03 fixed for `str`, back for `np.str_`.
- Where: src/agent_eda/data/provenance.py:367-372. `_size` strips Python's cached UTF-8 copy only when `type(value) is str`.
- Evidence: Confirmed by f5/npstr2.py. Column `a = [np.str_('é0'), np.str_('é1'), np.str_('é2'), None]`, and column `b` holds the same objects. Output: `identical calls give identical JSON: False` and `memory_bytes call 1: [249, 316]  call 2: [261, 316]`. With plain `str` the check gives `True`.
- Why it matters: Reruns disagree, which breaks the reproducibility the package promises.
- Fix direction: Size every `str` instance, subclasses included, without the cached UTF-8 copy. Add `np.str_` to the shared-strings regression test.

### STD-6: summarize_numeric decides "constant" on float64-rounded values, so distinct large integers get a false constant_columns claim and profile_target a false violation
- Kind: bug
- Severity: P1, for a wrong claim. The triggers are integers beyond 2^53 that fall within one float64 ulp, such as close snowflake IDs, or NaN stored as a value.
- Hard violation or judgement call: Hard violation of AGENTS.md Implementation ("When two analyses compute the same thing ... they call one shared implementation, so they can't disagree"). summarize_numeric's docstring says it decides constancy "as describe_dataset decides it". The constant rule is written three times: provenance.py:512-519, _numeric.py:216-230 and validation.py:971.
- Where: src/agent_eda/_numeric.py:216 and :225-230 (`x.min() == x.max()` on float64); src/agent_eda/statistics/descriptive.py:169-170.
- Evidence: Confirmed with `df = pd.DataFrame({'id': [2**54, 2**54+1, 2**54+2]})`:
  - summarize_numeric reports `[('constant_columns', "1 column holds a single distinct non-missing value: 'id'.")]` with limitations `['float_precision', 'too_few_values', 'zero_variance']`.
  - describe_dataset reports no finding and `distinct_count 3`.
  - `profile_target(df, target_column='id', task='regression')` reports `[('constant_target', 'violation')]`.
  - f4/constant2.py: a `double[pyarrow]` column `[1.0, 1.0, NaN as a value]` is constant for summarize_numeric but not for describe_dataset.
- Why it matters: An agent is told that a column of distinct IDs is constant, and a regression target gets a violation it doesn't have.
- Fix direction: Decide constancy on the stored values with one shared helper, exact for integer dtypes. Keep float64 for the statistics only.

### STD-7: With mcp installed but pyarrow missing, agent-eda-mcp crashes with a bare ModuleNotFoundError instead of MissingDependencyError
- Kind: bug
- Severity: P2
- Hard violation or judgement call: Hard violation of AGENTS.md Implementation ("a missing extra raises MissingDependencyError with the install command") and of the mcp-server spec's Package rule (message on stderr, exit status 1, no traceback).
- Where:
  - src/agent_eda/mcp.py:54: `_EXTRA_MODULES = ("mcp", "mcp_types", "anyio")`.
  - src/agent_eda/_table_files.py:17: `import pyarrow as pa` at module top.
  - Test gap: tests/test_packaging.py:206-212 blocks only `mcp`.
- Evidence: Confirmed by f7/no_pyarrow.py, which blocks pyarrow: `create_server raised: builtins ModuleNotFoundError | code: None | No module named 'pyarrow'`. `main` fails the same way.
- Why it matters: `pip install agent-eda mcp`, or an image with mcp preinstalled and agent-eda installed with `--no-deps`, starts with a traceback instead of the install command the docstrings promise.
- Fix direction: Add "pyarrow" to `_EXTRA_MODULES`, or derive the list from the extra. Make the packaging test block pyarrow too.

### STD-8: Several caller mistakes that an MCP tool can reach still give MCP callers a Python fix
- Kind: inconsistency
- Severity: P2
- Hard violation or judgement call: Hard violation of the mcp-server spec, Errors ("Each raise site an MCP tool can reach builds both messages from one function of the surface (`agent_eda._wording`)"), and Discovery ("Nothing an MCP caller reads holds ... Python call syntax: ... the error messages").
- Where:
  - src/agent_eda/_sentinels.py:500-576 (`_message`, `_closing`, `literal`), shared by profile_missing_values and detect_missing_sentinels.
  - src/agent_eda/data/targets.py:425-430 (label count) and :458-479 (label types).
  - src/agent_eda/_project_file.py:142-147 (missing project file).
  - tests/test_mcp_fixes.py:36-226 (`MISTAKES`) has no case for any of these, and tests/test_mcp_project.py:157-185 asserts the Python wording for the missing-file case.
- Evidence: Confirmed over the in-memory MCP client (f3/e2.py, f7/mistakes*.py, f10/mcp_fix_sweep.py, f6/pf_check.py). `_mcp_message` is None for all of them (f3/e1.py).
  - `{"sentinel_values": {"age": ["-999"]}}` says "…such as sentinel_values={'age': [-999]}."
  - On a Parquet datetime column: "Did you mean pd.Timestamp('1900-01-01 00:00:00')? … sentinel_values={'when': [pd.Timestamp(...)]}". Other column types get `pd.Timedelta(...)`, `datetime.date(2024, 1, 1)` or `sentinel_values={'flag': [False]}`.
  - An infinite value gets `df.replace([np.inf, -np.inf], np.nan)`.
  - profile_target with a binary task and the wrong number of labels says "…or pass task='multiclass_classification'".
  - A missing `project_path` says "Save a project there first with project.save('…/missing.toml')".
- Why it matters: An MCP caller is told to make a Python call it can't make, which is the case the spec's fix-wording rule exists for. The `task=` message contradicts the spec's own JSON example, `{"task": "regression"}`.
- Fix direction: Build these messages with `worded(...)` and `as_written`. Point the missing-file case at the TOML format in the `project_path` description. Add every case to `MISTAKES`.

### STD-9: MCP output schemas still contain Python call syntax: isna(), duplicated(), df.drop_duplicates(...), df.duplicated(...) and scipy.stats.trim_mean(x, 0.1)
- Kind: inconsistency
- Severity: P2 by the rubric. The practical impact is small; agents still understand the meaning.
- Hard violation or judgement call: Hard violation of the mcp-server spec, Discovery ("Nothing an MCP caller reads holds `ae.` or Python call syntax: ... the output schemas' field descriptions"). Ticket 02 reworded `isna()` and `duplicated()` in the docstrings and exempted formulas such as `sqrt(...)`. These are pandas and SciPy calls in result-field descriptions, which `record_schema()` passes to MCP unchanged.
- Where: src/agent_eda/results.py:1176, 1299 (also inherited by ProfileTargetStatistics), 1642, 1777, 2124, 2462, 2466, 2888 and 3173. The guard at tests/test_mcp_tools.py:38 (`PYTHON_ONLY`) checks only `to_frame(`, `to_dict(`, `to_json(`, `res.` and `ae.`.
- Evidence: Confirmed. std/mcp_calls.py over `tools/list` finds:
  - `isna()` in the output schemas of 6 MCP tools (describe_dataset, infer_column_semantics, profile_missing_values and others);
  - `df.drop_duplicates(` and `df.duplicated(` in find_candidate_keys;
  - `duplicated()` in find_duplicate_rows.

  f1/schema_calls.py and f7/scan_texts.py also find `scipy.stats.trim_mean(x, 0.1)` in `trimmed_mean`.
- Why it matters: MCP callers read code they can't run, against the user's decision, and the test gives false assurance.
- Fix direction: Reword these as the docstrings do ("pandas' isna", "DataFrame.duplicated with keep=False"). Widen the schema test to catch `name(` and `df.` calls while still allowing formulas.

### STD-10: Over MCP, datetime, date and duration sentinel values can't be declared, though the input schema offers date-time strings
- Kind: agent-friction
- Severity: P2
- Hard violation or judgement call: Judgement call: the generated input schema and the behaviour disagree. mcp-server spec user story 9 says the schema lets the client validate the call.
- Where: src/agent_eda/_sentinels.py:42 (the `SentinelValue` union) and :301-311 (the family test in `_check`). The schema passes through src/agent_eda/_mcp_server.py:431-458.
- Evidence: Confirmed by f3/e3.py, f3/e2.py and f7.
  - The item schema of `sentinel_values` includes `{"format": "date-time", "type": "string"}`, plus `date` and `duration` members.
  - On a Parquet datetime column, `{"when": ["1900-01-01T00:00:00"]}` is a tool error: "…holds datetimes without a time zone (dtype datetime64[us]), so it can never match. Did you mean pd.Timestamp(...)".
  - The same happens with '0 days' on a timedelta column and '2024-01-01' on a date column.
  - JSON has no other way to send these values.
- Why it matters: An MCP agent with Parquet datetime or duration data can never declare stand-ins such as 1900-01-01, which are exactly what detect_missing_sentinels suspects.
- Fix direction: Parse ISO 8601 text for datetime, date and duration columns (the shared parse rules already validate it), or drop those members from the MCP schema and say so.

### STD-11: "Parquet keeps its dtypes" is false for Parquet files written without pandas metadata
- Kind: doc-drift
- Severity: P2. The docs contradict the code.
- Hard violation or judgement call: Hard violation. The mcp-server spec's reading rules promise "keeping the file's dtypes", and so do the `data_path` description and the MCP server's instructions.
- Where: src/agent_eda/_data_files.py:123-134 (`pd.read_parquet(handle)` with the default dtype backend); the texts at src/agent_eda/_mcp_server.py:43-52 (`_DATA_PATH`) and src/agent_eda/_mcp_instructions.py:35-42 (`_DATA`).
- Evidence: Confirmed by f7/parquet_dtypes.py. A file written by pyarrow directly, as polars, Spark and DuckDB write it, holds int64 with nulls, bool with nulls and date32. It reads back as `{'qty': 'float64', 'flag': 'object', 'when': 'datetime64[us]', 'd': 'object', 'name': 'str'}`.
- Why it matters: describe_dataset reports float64 for an int64 column, and validate_column_types counts integers as "convertible" instead of "compatible".
- Fix direction: Read with a nullable or Arrow dtype backend, or a `types_mapper` for ints, bools and dates. Otherwise, reword the promise.

### STD-12: A leading ~ expands in project_path but not in data_path or table_path
- Kind: inconsistency
- Severity: P2
- Hard violation or judgement call: Judgement call: an inconsistency that will make an agent err.
- Where: src/agent_eda/_local_files.py:45-48 (`resolve` returns `(Path.cwd() / Path(path)).resolve()`), against `os.path.expanduser` in src/agent_eda/_project_file.py:141, 157 and 211.
- Evidence: Confirmed by f7/paths.py. A data_path of `~/...` gives `data_file_not_found`, resolved under the working directory, while a project_path of `~/...` in the same call loads.
- Why it matters: The two path arguments of one call follow different rules, and a tool-only agent can't run `echo $HOME` to work around it.
- Fix direction: Expand `~` in `resolve`, and say so in the path descriptions.

### STD-13: profile_target's expected_labels can't be used with datetime or period category targets, and the error speaks of sentinel values
- Kind: bug
- Severity: P2
- Hard violation or judgement call: Hard violation of AGENTS.md Errors: a message must name what is allowed and a fix that works. Here the signature rejects the fix the message suggests.
- Where:
  - src/agent_eda/data/targets.py:439-479; :72 (`Label = str | bool | int | float`); :487 (`_LABEL_KINDS` accepts categorical and object columns).
  - src/agent_eda/_sentinels.py:131-135 and :154 ("values no sentinel value can equal").
- Evidence: Confirmed by f3/e4.py and f3/e8.py.
  - A category-of-Timestamps target with str labels gives "Did you mean pd.Timestamp('2024-01-01 00:00:00')".
  - Passing Timestamps instead gives "each value in expected_labels must be str | bool | int | float".
  - A period-category target gives "'month' holds categories of values no sentinel value can equal … Declare each label as the type the target holds."
  - Both targets complete when expected_labels is left out.
- Why it matters: The expected-label check can never run on such targets, and the agent is sent round in a loop.
- Fix direction: Match such labels by their string form, or accept the target's scalar types. Give the label check its own wording.

### STD-14: NaN stored as a value is missing in four analyses, present with a limitation in one, and a plain value without any limitation in three, so one column can be "constant" in one result and empty in another
- Kind: inconsistency
- Severity: P2
- Hard violation or judgement call: Judgement call against AGENTS.md Implementation (one shared implementation, so analyses can't disagree) and Results (limitations say why a number reads as it does). Each analysis's rule is a ticket decision (first-look tickets 01, 02, 06, 10, 15 and 16). No ticket chose that describe_dataset, summarize_categorical and profile_target say nothing about it.
- Where: src/agent_eda/data/provenance.py:252-253 and :512-533; summarize_categorical's levels (src/agent_eda/_levels.py); profile_target's classes (src/agent_eda/data/targets.py). Compare the analyses that call `_numeric.stored_nan_count`.
- Evidence: Confirmed by std/storednan.py on `pd.Series(pa.array([nan, nan, 1.0, None]), dtype=pd.ArrowDtype(pa.float64()))`:

  | Analysis | Result | Limitation |
  | --- | --- | --- |
  | describe_dataset | `missing 1, distinct 2` | none |
  | profile_missing_values | `native_missing_count 1` | `nan_not_counted_as_missing` |
  | summarize_numeric | `count 1, missing 3` | `nan_counted_as_missing` |
  | infer_column_semantics | `non_missing 1` | `nan_counted_as_missing` |
  | summarize_categorical | `missing 1, cardinality 2` | none |
  | profile_target (multiclass) | classes `[('nan', 2), (1.0, 1)]` | none |

  On `[nan, nan]`, describe_dataset reports `constant_columns` while summarize_numeric returns `insufficient_data`. f5/const.py shows the reverse on `[5.0, 5.0, NaN]`.
- Why it matters: An agent reading Parquet with pyarrow dtypes, or data from polars, gets a 'nan' class in its target profile and a "constant" column that another analysis calls empty. Neither result explains the gap.
- Fix direction: Add the shared `nan_not_counted_as_missing` limitation wherever a stored NaN stays a value: describe_dataset, summarize_categorical, and profile_target's classification.

### STD-15: find_candidate_keys says its counts are what df.drop_duplicates and df.duplicated give, which is false for a single object column holding None and NaN
- Kind: doc-drift
- Severity: P2. The docs contradict the code.
- Hard violation or judgement call: Hard violation: the field descriptions and `When to use:` name a pandas call that gives different numbers.
- Where: src/agent_eda/results.py:2460-2466 (CandidateKey `distinct_count` and `duplicate_row_count`); src/agent_eda/data/structure.py:84.
- Evidence: Confirmed by f5/defs.py on `a=["x", None, NaN, "y"]` (object):
  - find_candidate_keys reports `most unique: [(['a'], 3, 2)]`.
  - pandas' `drop_duplicates(subset=['a'])` keeps 4 rows, and `duplicated(subset=['a'], keep=False).sum()` is 0. For a single column pandas uses `Series.duplicated`, which tells None and NaN apart; ticket 14's tests work around this.
  - find_duplicate_rows' docstring states the caveat; these texts don't.
- Why it matters: An agent that checks a number with the call the description names gets a different number and distrusts one side.
- Fix direction: Word it as find_duplicate_rows does ("as pandas compares rows over two or more columns: every missing value equals every other").

### STD-16: Sets holding missing values don't compare by content, though the docstring and the field description say they do
- Kind: bug
- Severity: P2. The data is rare, and the docs contradict the code.
- Hard violation or judgement call: Hard violation. `_values._content`'s docstring says "Missing values inside them become None, so they compare equal", and FindDuplicateRowsResult says "Lists, dicts and sets compare by content".
- Where: src/agent_eda/_values.py:72-73 (`return frozenset(value)`, without `_content` on the items).
- Evidence: Confirmed by f5/sets.py, running find_duplicate_rows on a column `s` with `k=[1,1,2]`:
  - lists with None or NaN give `groups=[[0, 1]]`;
  - dicts give `[[0, 1]]`;
  - sets with None or NaN give `[]`;
  - two separately built `{1, nan}` give `[]`.
- Why it matters: Rows that the result's own rule calls duplicates go unreported, and lists and sets behave differently.
- Fix direction: Use `frozenset(map(_content, value))`.

### STD-17: Declared sentinel values that print as the same ISO text can't be told apart, and profile_missing_values' to_frame() drops a count
- Kind: bug
- Severity: P2. A count in `to_frame` is wrong. It needs an object column holding a date and its ISO text, both declared.
- Hard violation or judgement call: Hard violation of AGENTS.md Results ("`to_frame()` gives the complete tables"). Also Data Clumps: `DeclaredSentinelValue` lacks the `value_type` that `SentinelCandidate` and `InvalidValue` carry.
- Where: src/agent_eda/results.py:2107-2117 and 2272-2274.
- Evidence: Confirmed by f1/declared.py.
  - Declaring `[date(1900,1,1), "1900-01-01"]` gives the record `[{"value":"1900-01-01","count":1},{"value":"1900-01-01","count":2}]`, while `to_frame()` gives `[{'1900-01-01': 2}]`.
  - Declaring `[pd.Timedelta('1D'), 'P1DT0H0M0S']` gives `to_frame()` `[{'P1DT0H0M0S': 1}]` while `sentinel_count` is 2.
- Why it matters: The record shows two identical-looking entries, and the table loses one of their counts.
- Fix direction: Add `value_type` to `DeclaredSentinelValue`, and make the frame cell a list of `{value, value_type, count}`.

### STD-18: to_dict()'s docstring says the result object and to_frame() keep every entry, which eight capped fields contradict
- Kind: doc-drift
- Severity: P2. The docs contradict the code.
- Hard violation or judgement call: Hard violation. The docstring contradicts the code, and AGENTS.md promises complete tables from `to_frame()`.
- Where:
  - results.py:666-673 (the docstring).
  - The `_Cap(total=…)` fields that keep only their first entries on the object: results.py:1497, 1757, 1806, 1834, 1848, 2316, 3176 and 3187.
  - detect_missing_sentinels' `to_frame()`: results.py:2419-2427.
- Evidence: Confirmed by the STD-R fork's probes.
  - summarize_categorical with 150 levels has `len(levels)` 100, `cardinality` 150 and `len(to_frame())` 150.
  - A sentinel candidate found 20 times keeps 10 row IDs on the object and in its `to_frame()` cell.
- Why it matters: An agent that trusts `help(res.to_dict)` reads truncated levels or row IDs as complete.
- Fix direction: Name the fields that keep only their first 5, 10 or 100 entries, and say where the rest is.

### STD-19: Declaring both 1 and True as sentinel values of an object column raises "got 1 more than once", though the matching rules treat them as different values
- Kind: bug
- Severity: P2
- Hard violation or judgement call: Hard violation of AGENTS.md Errors: the message names a mistake the caller didn't make. It also contradicts first-look ticket 10 ("In an object column, only values of the declared value's kind ... match, so True never matches 1").
- Where: src/agent_eda/_sentinels.py:269 (`check_distinct(...)`); src/agent_eda/_run.py:135-144 (`Counter` merges 1, 1.0 and True). The same misleading wording appears for expected_labels at src/agent_eda/data/targets.py:412. There, merging the two is by design (ticket 15), but the message still names a duplicate the caller didn't write.
- Evidence: Confirmed with `df = pd.DataFrame({'x': pd.Series([1, True, 'a', 2], dtype=object)})`:
  - `ae.profile_missing_values(df, sentinel_values={'x': [1, True]})` raises `InvalidArgumentError invalid_argument | profile_missing_values got 1 more than once in sentinel_values['x']. List each value once.`
  - `[True]` alone matches 1 row, and `[1]` alone matches a different row.
  - `profile_target(..., expected_labels=[1, True])` says "got 1 more than once in expected_labels" (f3/e1.py).
- Why it matters: An agent can't declare both stand-ins in one call, and it hunts for a duplicate it never wrote.
- Fix direction: Compare declared sentinel values by (family, value), the key that matching uses. For expected_labels, name both values as given and say that they count as one label.

### STD-20: Public pydantic models let invalid values past their checks: ColumnSpec's JSON and string validators raise raw pydantic errors, and model_copy(update=…) makes objects that crash analyses
- Kind: bug
- Severity: P2
- Hard violation or judgement call: Hard violation of AGENTS.md Errors ("pydantic validation errors are re-raised as InvalidArgumentError") and of the first-look spec ("pydantic's raw errors never reach the caller"). Ticket 08 wrapped these entry points for Project and DataPolicy; ticket 12 wrapped only `ColumnSpec.model_validate`.
- Where: src/agent_eda/data/validation.py:171-184 (only `model_validate` is overridden); src/agent_eda/project.py:28-71 (`_Declaration` doesn't revalidate copies).
- Evidence: Confirmed.
  - `ae.ColumnSpec.model_validate_json('{"logical_type": "intger"}')` and `ae.ColumnSpec.model_validate_strings({'logical_type': 'intger'})` both raise `pydantic_core._pydantic_core ValidationError`, with no `code`.
  - f6/copy_check.py: `ae.Project().model_copy(update={"data_policy": {"sensitive_columns": ["email"], "mask_sensitive": True}})` followed by `ae.summarize_categorical(df, project=p)` raises a bare `AttributeError: 'dict' object has no attribute 'sensitive_columns'`.
  - f2/probe14.py: `ColumnSpec("integer").model_copy(update={"logical_type": "date"})` passes the registry, then crashes while the result is built, with a raw `ValidationError for ValidateColumnTypesColumn`.
- Why it matters: An agent that loads a schema from JSON, or changes a frozen Project the usual pydantic way, gets an uncoded crash instead of a fix.
- Fix direction: Wrap `model_validate_json` and `model_validate_strings` on ColumnSpec as Project does, and revalidate copies (`revalidate_instances="always"`) or override `model_copy`.

### STD-21: Every analysis's When to use and See Also name list_analyses, which is a discovery function, not an analysis
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Hard violation of the letter of AGENTS.md Docstrings ("`When to use:` and `See Also:` name only analyses that exist"). tests/test_contract.py:93-104 checks only that the name resolves in `agent_eda`.
- Where: all 11 docstrings, for example src/agent_eda/data/provenance.py:80-81 and :178; statistics/descriptive.py:232 and :554; statistics/association.py:358; data/validation.py:353-354, :513, :1311 and :1474.
- Evidence: Confirmed. For all 11 analyses, std/seealso.py prints `see_also list_analyses: ...  <-- NOT AN ANALYSIS` and `when_to_use names non-analysis exports: ['list_analyses']`. As a side effect, the words enter every analysis's search text, so `search_analyses('analyses')` matches all 11 with score 1 (f6).
- Why it matters: See Also is meant to name alternatives. find_candidate_keys' See Also is just describe_dataset plus list_analyses, and the MCP tool descriptions present list_analyses as a related analysis.
- Fix direction: Drop the entry and let the guide point to list_analyses, or amend the rule. Make the contract test check membership in `ae.list_analyses()`.

### STD-22: Some suppressions have no reason on the same line, and nothing in the suite enforces the rule
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Hard violation of AGENTS.md Tests ("A `# noqa` or `# ty: ignore` carries its reason on the same line").
- Where: src/agent_eda/_sentinels.py:180 (a bare `# noqa: B009`); tests/test_summarize_numeric.py:312-314 (`# ty: ignore[invalid-argument-type]  # see above`). tests/test_suite_rules.py checks only private imports and test names, and its name check also misses `async def`.
- Evidence: Confirmed. `grep -rn -e noqa -e "ty: ignore" src tests` finds `_sentinels.py:180:    return np.dtype(getattr(dtype, "numpy_dtype"))  # noqa: B009` and the three "see above" lines. Every other suppression states its reason, for example mcp.py:164.
- Why it matters: A reader can't tell why the suppression is there, which is what the rule guards against.
- Fix direction: State the reasons inline, and add a check for a trailing `# <reason>` to tests/test_suite_rules.py.

### STD-23: Glossary Avoid-list words appear in a field description, in names and in tests
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Hard violation of GLOSSARY.md: Assessment avoids "verdict", Status avoids "outcome", Analysis result avoids "output", and Analysis avoids "check" as its name.
- Where:
  - src/agent_eda/results.py:3197: "The verdict on the column…". This reaches MCP callers through the output schema.
  - src/agent_eda/_run.py:115: `budget_exceeded_reason(outcome=...)`.
  - statistics/association.py:373: `outcome = _read(...)`.
  - _keys.py:160: `SearchOutcome`.
  - data/missingness.py:566-574: "An anticipated failure stopped the check".
  - tests/test_find_candidate_keys.py:376-383: "in an output" and `outputs = [...]`.
- Evidence: Confirmed by `grep -rniI "verdict\|outcome" src/agent_eda` and the forks' greps. Only results.py:3197 is caller-facing. The "verdict" in `_conversions` names a different concept.
- Why it matters: The vocabulary drifts, and the first case reaches MCP callers.
- Fix direction: Write "The assessment of the column…", rename the variables (`result`, `read`), write "stopped the analysis", and use "rendering" in the test.

### STD-24: "server" appears without "MCP" in eight test names and in private docstrings, and the instructions say every MCP tool runs an analysis
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Hard violation of GLOSSARY.md ("MCP server … Avoid: server (unqualified)"). docs/agents/domain.md names test names explicitly. The text callers read is clean: no MCP schema holds a bare "server" (f10/bare_terms.py).
- Where:
  - Test names: tests/test_mcp_tools.py:62, 144, 277, 289; test_mcp_data.py:155; test_mcp_project.py:232, 245, 281.
  - Parameter ids at test_mcp_project.py:274-275, plus comments and docstrings across the MCP test modules.
  - Private docstrings: src/agent_eda/_mcp_server.py:5, 162, 217, 220-221, 287, 292, 313; _table_files.py:50.
  - `path_argument(tool: str, ...)` at src/agent_eda/_local_files.py:69 receives an analysis name.
  - src/agent_eda/_mcp_instructions.py:30 says "Each MCP tool runs one analysis", but two MCP tools are discovery tools.
- Evidence: Confirmed by grep, for example `test_the_server_lists_one_mcp_tool_per_analysis`.
- Why it matters: Small drift from the glossary, and one inaccurate sentence in text that agents read.
- Fix direction: Rename to `mcp_server` or "MCP server", rename the parameter to `analysis`, and reword the sentence.

### STD-25: Docstrings and field descriptions that disagree with the code
- Kind: doc-drift
- Severity: P3
- Hard violation or judgement call: Hard violations, each small: every text below states behaviour the code doesn't have.
- Where:
  - results.py:2329-2333: `columns_without_rules` says "A column without values isn't listed". An all-NA `boolean` column is listed.
  - descriptive.py:493-494 and validation.py:459-463 say `insufficient_data` means "no selected column has a non-missing value". An all-missing column next to a column of lists gives `insufficient_data` with code `no_values`, because of the precedence rule, although the lists column has values.
  - validation.py:383-384 says candidates are "the logical types scoring above 0". Integer and float appear at 0.
  - validation.py:409-411 says exact ties go to boolean, integer, …, categorical, text. `_candidates` sorts raw floats, so categorical 3/5 against text 3/5 gives `[('text', '0.6000000000000001'), ('categorical', '0.6')]` and the inferred type is text.
  - results.py:3307-3310 says validate_column_types' `to_frame()` columns table holds "the distinct invalid values". It holds only their count.
  - results.py:290-293 calls `ResultModel` "frozen and strict", but `strict` isn't set, so `count='3'` is coerced to 3.
  - missingness.py:536-574: detect_missing_sentinels' finding bullets follow its evidence-code bullets without the "The findings:" lead-in that the sibling docstrings use.
  - tests/test_properties.py:188-190 says "the 200 missed", but line 181 draws 500 frames.
- Evidence: Confirmed:
  - the `columns_without_rules` case by f3/e4.py;
  - the insufficient_data case by f4/mixstatus.py and f2/probe3.py;
  - the zero-score candidates by f2/probe7.py;
  - the tie order by f2/probe8.py;
  - the strictness by the STD-R probe.

  Read-only for the `to_frame` docstring, the detect_missing_sentinels lead-in and the test comment, by comparing the quoted text with the code.
- Why it matters: Agents and MCP descriptions read these texts as the contract.
- Fix direction: Correct each text. For the tie order, rank on exact fractions.

### STD-26: Limitation codes and status-reason codes have no discoverable meanings
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call. Finding codes get meanings in `Returns:`, which the contract test checks. Limitation codes are explained only in `#` comments, and status-reason codes such as `no_valid_labels`, `no_rows` and `no_columns_to_check` show up only at run time.
- Where: src/agent_eda/results.py:70-113 and :370-372 (`LimitationCode`); the `Statuses:` sections of every analysis. `StatusReason.code` is a free `str`.
- Evidence: Confirmed.
  - `describe_analysis("profile_target").to_json()` lacks "no_valid_labels", though a run returns it (f3/e9.py).
  - The output schema says only "Stable code from the package-wide list, LimitationCode" plus an enum (f1).
  - The comment on `numerical_warning` ("values it affects are null") is wrong for the registry's catch-all case, which has `fields=[]`.
- Why it matters: Agents branch on codes, but can't learn what a code means, or that it exists, before they meet it.
- Fix direction: Keep one code-to-meaning table per code family, feed it into the field descriptions and `describe_analysis`, and name each status's reason codes in `Statuses:`.

### STD-27: excluded_by_reason is a free dict, and one reason has two spellings
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call (Primitive Obsession).
- Where: src/agent_eda/results.py:395-397. statistics/descriptive.py:721 writes "unhashable"; data/structure.py:413 and statistics/association.py:101 and :519 write "unhashable_values".
- Evidence: Confirmed by f1/reasons.py.
  - summarize_categorical writes `['missing', 'unhashable']`.
  - find_candidate_keys writes `['unhashable_values']`.
  - compute_association_matrix writes `[..., 'unhashable_values']`.
  - summarize_categorical's own status code for the case is `unhashable_values` (descriptive.py:626).
- Why it matters: An agent branching on the reason misses one analysis's exclusions.
- Fix direction: Add an `ExclusionReason` Literal with one spelling, and list it in the field description.

### STD-28: One field name, two meanings: distinct_count and duplicate_row_count
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call (Mysterious Name). Ticket 07 rejected the name `duplicate_row_count` as ambiguous; ticket 14, written in parallel, used it anyway.
- Where:
  - `distinct_count` means distinct non-missing values at src/agent_eda/results.py:1179 and :2891, but combinations, with missing counted as one value, at :2460 (CandidateKey).
  - `duplicate_row_count` at :2464 is the same count as FindDuplicateRowsResult's `duplicate_group_row_count`.
- Evidence: Confirmed by std/distinct.py and f5/defs.py.
  - On the object column `[1, 1.0, True, "1", None, nan, "a"]`, describe_dataset and infer_column_semantics give `distinct=3`, while find_candidate_keys gives `(['c'], 4, 2)`.
  - On `sku=["s1","s2","s3",None]`, the counts are 3 against 4.
- Why it matters: An agent comparing a key candidate with describe_dataset reads two distinct counts for one column.
- Fix direction: Rename the CandidateKey fields (`combination_count`, `duplicate_group_row_count`), following the rule for renamed names.

### STD-29: Checks that didn't run leave no limitation in three analyses, unlike their siblings
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Defaults ("None meaning the check that needs them doesn't run and the result says so").
- Where: data/missingness.py:486-499 and :669-715 (detect_missing_sentinels without `sentinel_values`); data/validation.py:1633-1673 (infer_column_semantics without `declared_schema`, and validate_column_types with `nullable=None`).
- Evidence: Confirmed by f3/e4.py.
  - detect_missing_sentinels reports `limitations: []`, while profile_missing_values adds `sentinel_values_not_declared`.
  - profile_target adds `expected_labels_not_declared` and `rare_share_not_declared`.
  - The two validation analyses add none. The only trace is `nullable: null` or a fact line.
- Why it matters: An agent that reads limitations to learn what wasn't checked misses these.
- Fix direction: Add `*_not_declared` limitations, or document a different convention.

### STD-30: Analyses that are each other's next step don't point at each other
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Docstrings ("which other analysis to use instead"; neighbours point at each other).
- Where:
  - data/structure.py:78-86 and :180-183, and data/duplicates.py:60-77 and :180-185. find_candidate_keys reports a near-key's `duplicate_row_count` without row IDs; `find_duplicate_rows(columns=[...])` lists those rows, and neither docstring says so.
  - statistics/association.py:197-209 and :353-358 tell agents to list numeric codes in `categorical_columns` but don't name infer_column_semantics, which finds them.
  - summarize_categorical flags mixed types and numbers stored as text without naming infer_column_semantics.
- Evidence: Read-only: the quoted sections.
- Why it matters: After `no_key_found` the agent has to guess how to see the offending rows, and it misses the analysis that prepares `categorical_columns`.
- Fix direction: Add the pointers in `When to use:` and `See Also:`, in both directions.

### STD-31: infer_column_semantics leaves inferred_logical_type null without a limitation naming it, and names container paths
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Hard violation of the letter of AGENTS.md Results ("null, and a limitation names the field path and the reason"). The column's status_reason softens it.
- Where: src/agent_eda/data/validation.py:849-857 and :1213-1252.
- Evidence: Confirmed by f2/probe3.py.
  - Period and list columns get `inferred_logical_type None`.
  - The only limitations are `('too_few_values', ["columns['a'].evidence"])`, which names a container path where summarize_numeric names each leaf, and `unhashable_values`.
  - The printed table shows `p  null  null`.
- Why it matters: Each analysis explains its null values differently.
- Fix direction: Add a limitation naming `columns[...].inferred_logical_type`, and give `too_few_values` leaf paths.

### STD-32: datetime_format results depend on the process locale, and no public text says so
- Kind: doc-drift
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md's determinism rule (no global state). Ticket 13 chose the locale dependence but documented it only in the private `_parse.py:33`.
- Where: src/agent_eda/_parse.py:233-241; data/validation.py:1321-1325 (the `schema` description).
- Evidence: Confirmed by f2/probe9.py. With format `%d %b %Y`, the C locale gives `convertible 2 invalid 2`. After `setlocale(LC_TIME, "de_DE.UTF-8")`, the same call gives `convertible 1 invalid 3`.
- Why it matters: A rerun in another process can give different counts, with no hint why.
- Fix direction: Say so in the `datetime_format` and `schema` descriptions, or pin the C locale for parsing.

### STD-33: The schema argument's JSON Schema doesn't say it needs at least one column
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call (mcp-server spec user story 9: the schema lets the client validate the call).
- Where: src/agent_eda/data/validation.py:1288 and :1476-1484.
- Evidence: Confirmed by f2/probe13.py. The schema has `required: ['schema']` and no `minProperties`, so `{}` fails only at call time.
- Why it matters: An MCP client can't reject the call before sending it.
- Fix direction: Give the mapping a minimum length so the schema carries `minProperties: 1`.

### STD-34: Search and errors miss two common agent wordings
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call (the spec's curated synonym list; AGENTS.md Errors: the message gives the fix).
- Where: src/agent_eda/_search.py:43-48 (`SYNONYMS`); src/agent_eda/_registry.py:391-409 (the `project` option).
- Evidence: Confirmed (f6).
  - `search_analyses('PII')` and `search_analyses('privacy')` find nothing.
  - `'redact email addresses'` matches only the generic masking tag.
  - `summarize_categorical(df, project="project.toml")` says "…project must be Project | None. Input should be a valid dictionary or instance of Project. See help(...)", with no pointer to `ae.Project.load`.
- Why it matters: An agent asking how to keep personal data out of results finds nothing, and a likely slip right after `project.save(...)` gets no fix.
- Fix direction: Add synonym groups such as (sensitive, pii, personal, privacy) and (masking, redact, hide). For a str or path given as `project`, suggest `project=ae.Project.load('…')`.

### STD-35: Closest-name suggestions come from four implementations, and some miss case-only typos or never name the misspelled key
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code) against the first-look spec ("did you mean" suggestions "including differences only in case", stated for columns) and AGENTS.md Errors (name what is wrong).
- Where:
  - Case-sensitive difflib: src/agent_eda/_registry.py:413-419 (`unknown_name_message`, used for analysis keywords, MCP arguments and Project, DataPolicy and ColumnSpec fields), src/agent_eda/__init__.py:113 and results.py:265-266.
  - The case-folding version: _columns.py:277-289 (`closest_names`).
  - data/validation.py:255-271: `_spec_error` takes pydantic's first problem.
- Evidence: Confirmed.
  - `find_duplicate_rows(df, ID_COLUMN="a")` gives "has no argument 'ID_COLUMN'. Allowed: …" with no suggestion, while `columns=["A"]` gives "Did you mean 'a'?". `ae.Project(NAME="x")` and `ae.SUMMARIZE_NUMERIC` also get no suggestion (f6/kw_check.py).
  - `ae.ColumnSpec(logical_tpe="integer")` gives "ColumnSpec needs logical_type, …" and never names 'logical_tpe', because pydantic lists `missing` before `extra_forbidden` (f2/probe12.py).
  - ColumnSpec's docstring says these raise `invalid_argument`; they raise `invalid_call`.
- Why it matters: Whether an agent gets the hint it relies on depends on which code path raised.
- Fix direction: Route every suggestion through `closest_names`, and report `extra_forbidden` first, with its closest field.

### STD-36: Two type renderers, so error messages and describe_analysis state the same parameter's type differently
- Kind: doc-drift
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code). The `_annotations.py` docstring claims both use "the same functions, so they can't disagree".
- Where: src/agent_eda/_registry.py:548-563 (`_plain`) against src/agent_eda/_annotations.py:116-146 (`type_text`).
- Evidence: Confirmed (f6).
  - For `declared_schema`, the error says `Mapping[str, Literal['boolean', 'integer', …] | ColumnSpec] | None`, while describe_analysis says `Mapping[str, str | ColumnSpec] | None`.
  - `data` is `DataFrame` in one and `pd.DataFrame` in the other, for all 11 analyses.
  - Over MCP, the registry's message also carries pydantic's raw "Input should be a valid dictionary." (f2).
- Why it matters: Agents compare the description with the error text.
- Fix direction: Use one renderer with a flag for expanding Literals, or correct the docstring.

### STD-37: Project's types are written three times and disagree, and DataPolicy rejects a NumPy bool that every analysis accepts
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code). Accepting a dict is ticket 08's choice; see Questionable decisions.
- Where:
  - src/agent_eda/project.py:366-373: the TYPE_CHECKING stub, `data_policy: DataPolicy | Mapping[str, Any] | None`.
  - :354: `_field_types`.
  - :544-555: `__signature__`, `DataPolicy | None`.
  - :315: the docstring.
  - DataPolicy has its own copies at :209, :223-230 and :528-543.
  - :216: `mask_sensitive` without `python_scalar`.
- Evidence: Confirmed (f6).
  - `inspect.signature(ae.Project)` gives `data_policy: DataPolicy | None`.
  - `ae.Project(data_policy=5)` says the value "must be DataPolicy", though dicts work.
  - `ae.DataPolicy(sensitive_columns=['a'], mask_sensitive=np.bool_(True))` says "must be bool", while `summarize_categorical(..., mask_sensitive=np.bool_(True))` runs.
- Why it matters: `help()`, editors and error messages each state a different allowed type, and a bool computed with NumPy fails in one place only.
- Fix direction: Derive all three from one table, say "a DataPolicy, a dict of its fields, or None", and normalise NumPy scalars.

### STD-38: Smaller correctness gaps in the parse, sentinel and key-search rules
- Kind: bug
- Severity: P3. Edge cases.
- Hard violation or judgement call: Hard violations of the documented rules, on rare data.
- Where:
  - _parse.py:184-199 and _conversions.py:400-405: the text '1e400' counts as converting "exactly" to float, though float64 makes it inf.
  - _sentinel_rules.py:487-498: decimal candidates are recorded with `float(value)`.
  - _keys.py:212-216: the key search loops `range(1, max_key_size + 1)`, although `search()` caps at the column count.
  - _sentinels.py:261-263 and :283-293: with `id_column="email"` and `sentinel_values={"email": [...]}`, the error says the column is one "columns leaves out. Add it to columns", though the caller passed no columns.
- Evidence: Confirmed by fork probes.
  - Declaring float for `["1e400", "-1e400", "1"]` gives `3 0 passed` (f2/probe1.py).
  - `Decimal('1E+400')` twice in an object column raises an unexpected pydantic `ValidationError ... input_value=inf` (f3/e4.py).
  - decimal128 `12345678901234567.25` is recorded as `1.2345678901234568e+16`, and declaring that value then matches nothing (f3/e7.py).
  - `max_key_size=10_000_000` takes 0.55 s on 3 columns, so a huge value meant as "no limit" stalls the MCP server (f5/bigk.py).
  - The id_column message was reproduced by f3/e6.py.
- Why it matters: Each gives a wrong count, a crash or a misleading fix on unusual input.
- Fix direction:
  - Treat text that overflows float64 as not converting.
  - Record decimals as exact text.
  - Loop to `min(max_key_size, len(columns))`.
  - Say that `columns=None` leaves out the id_column.

### STD-39: Limitation reasons and skip messages carry Python call syntax into the MCP text, and fixing them wouldn't break record equality
- Kind: doc-drift
- Severity: P3
- Hard violation or judgement call: Judgement call. The mcp-server spec says nothing an MCP caller reads holds Python call syntax, "the printed results" included. Tickets 04 and 07 kept record text in Python so records stay equal; these edits would change both surfaces alike.
- Where: src/agent_eda/_numeric.py:171 ("which pandas isna() doesn't count as missing"); statistics/association.py:95-97 ("convert them with .dt.total_seconds() first").
- Evidence: Confirmed by f4/mcptext.py.
  - summarize_numeric's MCP text shows "- nan_counted_as_missing: 1 column stores NaN as a value, which pandas isna() doesn't count…".
  - A Parquet duration column puts ".dt.total_seconds()" into `skipped_columns[].message`.
- Why it matters: MCP callers read advice only Python can follow.
- Fix direction: Write "isna" without parentheses, and "convert them to seconds".

### STD-40: An unknown MCP tool name gets an error that lists no MCP tools and suggests none
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call against the spirit of AGENTS.md Errors. Ticket 01 chose the error type, `MCPError(INVALID_PARAMS)`, not the message.
- Where: src/agent_eda/_mcp_server.py:245-250.
- Evidence: Confirmed by f7/paths.py: `There is no MCP tool named 'describe_analysis'.` The same happens for `guide`.
- Why it matters: Agents that know the Python API will try `describe_analysis` and `guide`, and nothing points them to the MCP tool descriptions and instructions that replace these.
- Fix direction: Suggest the closest names, list the MCP tools, and say where the describe_analysis and guide content lives.

### STD-41: The table_path line is added after the text caps, so MCP text can pass 120 characters per line and 40 lines
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Results ("at most 40 lines of 120 characters") and the instructions' "in at most 40 lines".
- Where: src/agent_eda/_mcp_server.py:512-516; src/agent_eda/_table_files.py:72-75.
- Evidence: Confirmed by f7/caps.py: with a deep table_path, every analysis's text reaches a maximum width of 282 to 284 characters. Read-only: a text already capped at 40 lines gets a 41st.
- Why it matters: Clients and agents that rely on the stated caps get longer text.
- Fix direction: Wrap the line with `_text.wrap`, and render with a 39-line budget when a table is written.

### STD-42: Some operating-system refusals get a misleading fix or become "bug in agent-eda" internal errors
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call under AGENTS.md Errors. Project.load handles the same cases correctly.
- Where: src/agent_eda/_data_files.py:41 and 113-120 (an OSError is caught as a parse error); _local_files.py:45-48 (`resolve` raises on a NUL character); _table_files.py:53-61 (maps only FileExistsError, FileNotFoundError and PermissionError).
- Evidence:
  - Confirmed by f7/perm.py: an unreadable data file gets "[Errno 13] Permission denied ... Check that the file is UTF-8 text with a header row...".
  - Confirmed by f7/paths.py: a NUL character in data_path or table_path gives `-32603 ... a bug in agent-eda, not in the call: ValueError: lstat: embedded null character`.
  - Read-only: a full disk or a read-only filesystem on write raises an OSError that becomes an internal error, and may leave a partial file that later calls refuse as `table_file_exists`.
- Why it matters: Agents get wrong advice, or "bug" errors for problems with their call or their machine.
- Fix direction: Handle PermissionError and other OSErrors as `_project_file._unreadable` does, reject NUL characters in `path_argument`, and map other write errors to InvalidArgumentError after removing a partial file.

### STD-43: In a CSV that table_path writes, an empty string can't be told from a missing value
- Kind: agent-friction
- Severity: P3
- Hard violation or judgement call: Judgement call. This follows from the CSV convention that ADR 0006 and the spec chose, but the table_path text doesn't say so.
- Where: src/agent_eda/_table_files.py:186-190; the `_TABLE_PATH` description.
- Evidence: Confirmed by f7/tables.py. detect_missing_sentinels on Parquet text that holds '' writes `city,,str,2,...`, which the MCP reader reads back as `[nan, ' ']`. summarize_categorical's levels table behaves the same.
- Why it matters: The '' candidate, the most common text stand-in, looks missing in the "full table".
- Fix direction: Say in the table_path description that '' and missing both write as an empty CSV field and that Parquet keeps them apart, or quote empty strings.

### STD-44: A pair's association value changes in the last digits with the other columns in the call
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call; identical calls still agree.
- Where: src/agent_eda/_association.py:529-530. The rows per chunk come from `row_bytes`, which depends on the other columns' levels.
- Evidence: Confirmed by f4/colset2.py on 300k rows.
  - Pearson a–b is 0.3939336528284101 with `columns=["a","b"]` and 0.39393365282841003 with every column.
  - Eta c0–a ends in …637 in one call and …639 in the other.
- Why it matters: An agent that narrows the columns, or diffs records, sees an unchanged pair change.
- Fix direction: Fix the rows per chunk independently of width, or document the effect.

### STD-45: Advice for converting the same column kind differs by analysis
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Repeated Switches), touching AGENTS.md Errors (the message gives the fix).
- Where: statistics/descriptive.py:63-68 and :364 (`_NUMERIC.fixes`, which has no datetime entry); statistics/association.py:63-82.
- Evidence: Confirmed by f4/errs.py.
  - summarize_numeric on a datetime column says "...Leave 'd' out of columns."
  - compute_association_matrix on the same column says "Convert datetimes to numbers with (df['d'] - df['d'].min()).dt.total_seconds() ...".
- Why it matters: The same mistake gets a fix in one analysis and none in the next.
- Fix direction: Keep one per-kind conversion map in `_columns`, and have every ColumnRule pick from it.

### STD-46: Methods travel as plain strings, an unknown one silently computes Kendall, and dead defaults remain
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Primitive Obsession, Speculative Generality).
- Where:
  - src/agent_eda/_association.py:98 and :157 (`method: str`).
  - :180-181 (`else: value = kendall_tau_b(...)`).
  - :174-176 and :186: unreachable `rank_pair is None` and `buffer=None` paths.
  - statistics/association.py:537-544: a stand-in `NumericColumn(values=np.empty(0), ...)` built only for `read_limitations`.
  - statistics/association.py:459: an untyped parameter.
- Evidence: Read-only: the quoted code.
- Why it matters: A mistyped method computes the wrong measure without an error, and the stand-in hides what `read_limitations` needs.
- Fix direction: Pass the `NumericMethod` Literal through, give `read_limitations` a narrower input, and drop the dead defaults.

### STD-47: validate_column_types numbers masked row IDs itself instead of calling the shared mask_values
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Implementation (masking is one shared implementation); `_masking`'s docstring says to use `mask_values()` for row IDs.
- Where: src/agent_eda/data/validation.py:1579-1615 (`numbers.setdefault(value, placeholder(start + len(numbers)))`).
- Evidence: Confirmed by grep: duplicates.py, missingness.py and targets.py go through `mask_values`. f2/probe6.py found no leak today.
- Why it matters: Two numbering schemes, dict equality and pandas factorize, can drift apart, and factorize already needed a NUL workaround.
- Fix direction: Give `_masking` an offset or known-values option, and call it here.

### STD-48: Small inconsistencies in the result and discovery records
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement calls: the caps rule, Duplicated Code, and agent-first display.
- Where:
  - results.py:2933: `ColumnEvidence.value_types` is uncapped, while :1521 `SummarizeCategoricalColumn.value_types` is capped.
  - results.py:557 and :732: `__rich_repr__` and `__pretty__`, inherited from pydantic, yield every field and so bypass the 40-line template.
  - results.py:735-753 re-implements `__str__`, `__repr__`, `to_dict` and `to_json` of AnalysisResult for the discovery records.
  - discovery.py:281-282 sets `shows_raw_values=spec.supports_masking`, so two fields always hold one value.
- Evidence: Confirmed by the STD-R and STD-D forks.
  - f1/fields.py lists the two `value_types` caps.
  - f1/display.py shows `__rich_repr__` yielding all 300 row IDs of a duplicate group.

  Read-only for the duplicated plumbing and the redundant field.
- Why it matters: The same list has two record shapes, `rich.print(res)` dumps an uncapped result, and the duplicated plumbing can drift.
- Fix direction: Cap both `value_types`, define `__rich__` and `__pretty__` as `str(self)`, and share a small record mixin.

### STD-49: MCP schema fragments are shared mutable module dicts, and the call path reads the JSON Schema back as its source of truth
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (module-level mutable state, Feature Envy, Message Chains).
- Where: src/agent_eda/_mcp_server.py:441 (`properties = {"data_path": _DATA_PATH}`); :427 (`RECORD_SCHEMA` inserted as it is); :444 and :450 (`_PROJECT_PATH` and `_TABLE_PATH` copied with `dict(...)`); :298 and :384-395.
- Evidence: Confirmed by f7.
  - Across MCP tools, the `data_path` and `table_file` fragments are the same objects (`is` gives True).
  - `_call` decides whether an analysis takes a project with `"project_path" in served.tool.input_schema["properties"]` rather than from the registry spec.
- Why it matters: A mutation anywhere would leak into every MCP tool, and the logic depends on how the schema happens to be laid out.
- Fix direction: Copy or freeze every fragment, and decide project handling from `spec`.

### STD-50: File-format knowledge and the search rules are restated across MCP modules
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code, Repeated Switches, Shotgun Surgery).
- Where:
  - `_data_files._format_of` (89-97) duplicates `_table_files._format_of` (137-146).
  - Branches on `file_format.kind == "Parquet"` or `"TSV"` repeat in `_read`, `_expected` and `_encode`.
  - `_mcp_instructions._fill` repeats `_text.wrap`.
  - `supported_extensions().replace(" and ", " or ")` edits another function's output.
  - `_SEARCH_DESCRIPTION` in _mcp_server.py:124-135 restates `_search.STOPWORDS`, `_search.SYNONYMS` and the SearchListing fields by hand.
- Evidence: Read-only: the quoted sites. The search description is accurate today, but no test ties it to `_search`.
- Why it matters: A new format or synonym needs edits in several places that no test checks.
- Fix direction: Put the separator and label on FileFormat, add one `format_of_or_raise`, and generate the synonym sentence from `_search.SYNONYMS`.

### STD-51: Each analysis module declares the same small pieces again
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code, Shotgun Surgery).
- Where:
  - The "any column" ColumnRule is declared in data/duplicates.py:27, data/missingness.py:45 and :424, data/structure.py:41, data/validation.py:286 and _row_ids.py:24. Some say "of any dtype", others "of every dtype".
  - `_EVERY_KIND` (structure.py:40, validation.py:285) re-derives `_columns.ALL_KINDS`.
  - The `columns[{name!r}]` path helper exists in descriptive.py:307, missingness.py:344, association.py:550 and validation.py:1021. targets.py:1114 `_statistics_path(name)` ignores its argument.
  - missingness.py:301 `_NO_ROWS` equals `_run.NO_ROWS`, and `_NO_COLUMNS` is defined twice.
  - The row-ID cap of 10 is restated in missingness.py:719, targets.py:85 and validation.py:1256, and the example cap of 5 likewise, beside results' `_Cap`.
  - The `nan_counted_as_missing` limitation is built three times: validation.py:1241 and :1790, and `_numeric.read_limitations`.
- Evidence: Confirmed by greps of `_ANY_COLUMN`, `_EVERY_COLUMN`, `_EVERY_KIND`, the path helpers and the caps (STD-K, STD-M, STD-V and STD-D forks).
- Why it matters: Wording, paths and caps drift apart between analyses, and changing a cap takes edits in four files.
- Fix direction: Export one any-column rule factory, one entry-path helper and the caps from `_columns` and `results`, and import them.

### STD-52: Factorizing and distinct counting are implemented four times
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Implementation ("one shared implementation, so they can't disagree").
- Where:
  - find_duplicate_rows' `_find_groups`, `_codes` and `_renumber` (data/duplicates.py:330-403) and find_candidate_keys' `codes_of`, `_sharing` and `_combined` (_keys.py:86-94 and :366-401) both find the rows that `duplicated(keep=False)` marks.
  - describe_dataset uses its own `nunique` and `pd.factorize` (provenance.py:274-303) next to `_levels.count_levels`.
  - compute_association_matrix uses `_Levels` and `_codes` (association.py:555-582) next to `_levels._count`.
- Evidence: Confirmed.
  - f5/agree.py found 0 mismatches between the two duplicate searches on 1,067 frames, but they already differ on unhashable values: one compares by content, the other leaves the column out.
  - association's `_codes` turns stored NaN into missing, while `_levels` keeps it as a level, so `level_count` and `cardinality` can differ for one column.
  - STD-2 has to be fixed in all four places.
- Why it matters: Analyses report different counts for the same column, and every fix lands four times.
- Fix direction: Write one helper that gives per-row codes with missing as a single value, and one level counter, each with a flag for stored NaN.

### STD-53: Result text reads its settings back from the untyped parameters record, and None stands for "the index labels"
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Primitive Obsession, Message Chains).
- Where:
  - src/agent_eda/results.py has 24 `self.parameters.get(...)` calls, for example at 632, 1366, 1406, 1673, 1884-1889, 2015, 2076, 2557 and 3226. The deepest reads `parameters["project"]["data_policy"]["sensitive_columns"]`, with fallbacks `"?"` and `or 0`.
  - In _row_ids.py:37-38 and _masking.py:68-74, `None` means the index labels.
  - In data/duplicates.py:250, `[row_ids.id_column]` becomes `[None]`, meaning "index labels shown".
- Evidence: Confirmed by grep count; read-only for the None convention.
- Why it matters: Typed state round-trips through JSON, so a renamed parameter silently prints "cuts ?%", and `[None]` reads like "nothing shown".
- Fix direction: Keep the settings and the masking decision in typed private attributes, and add a small `RowIdSource` or `INDEX` sentinel.

### STD-54: The same snippets are written into several result classes, and summarize_numeric's defaults are restated as literals
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code, Data Clumps, Shotgun Surgery).
- Where: src/agent_eda/results.py:
  - the fact "Row IDs: values of id_column / index labels", at 1673-1679, 2350-2355 and 3226-3232;
  - the check that quotes id_column unless it is masked, at 1685, 1943 and 2391;
  - four helpers that turn a label into text, at 1461, 1930, 2430 and 3281;
  - the mapping from status_reason to code, at 1427, 2270 and 3065;
  - `SummarizeCategoricalLevel` (1440) and `ProfileTargetClass` (1724), which have the same four fields;
  - profile_target's text, which restates summarize_numeric's defaults as literals (1770-1777 and 1980-1981: "0.01, …, 0.99", "int(0.1 * n)", "cuts 10%"), though they live in `_numeric.py:70-72`.

  The three numeric options also travel together through four layers: _numeric.py:203-209, :362-369 and :396-404, and descriptive.py:242-250.
- Evidence: Read-only: the grep hits listed.
- Why it matters: The copies drift apart (STD-17 is one case). Changing a default takes four edits, or profile_target misstates how it computed its statistics.
- Fix direction: Write one helper per snippet and one level model, and build a small frozen options object from the `_numeric` constants.

### STD-55: profile_target keeps four parallel structures per finding code
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Data Clumps, Shotgun Surgery).
- Where: src/agent_eda/data/targets.py:88-100 (the decorator's codes), :1010-1019 (`_FINDING_FIELDS`), :1038-1048 (`_ASSESSMENTS`), plus the docstring bullets.
- Evidence: Read-only.
- Why it matters: Adding or renaming a code takes four edits.
- Fix direction: Keep one table of code → (assessment, fields), and derive the decorator tuple from it.

### STD-56: results.py and _registry.py each change for many unrelated reasons, and private names cross module boundaries
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Divergent Change, and Feature Envy for the private names). AGENTS.md makes `agent_eda.results` the public home of result classes; it doesn't require a single file.
- Where:
  - src/agent_eda/results.py (3,329 lines) holds the caps and serialization, the discovery records and their renderers, the masking text, and the text and frames of all 11 analyses.
  - src/agent_eda/_registry.py does registration, binding, normalisation, message building, warning absorption and result checks.
  - Private names are used across modules:
    - _project_file.py:35 imports `_registry._value_text`;
    - _mcp_instructions.py:16 imports `discovery._first_look_lines`;
    - _mcp_server.py:327 and :512 call `._render("mcp")`;
    - _wording.py:26 sets `error._mcp_message`.
  - data/validation.py mixes a public model, two analyses, the inference engine and the masking numbering.
- Evidence: Confirmed.
  - `git log --oneline -- src/agent_eda/results.py | wc -l` gives 25 commits, touching most first-look tickets and MCP ticket 02.
  - A grep of cross-module underscore imports shows ticket 09's "one import of an underscore name" is out of date.
- Why it matters: Merge friction, a large surface to navigate, and private methods that became an internal API without being named as one.
- Fix direction: Make `agent_eda.results` a package that re-exports the same names, move the message and value helpers out of the registry, and make `render(surface)` and `first_look_lines` package-public.

### STD-57: The python-or-MCP wording switch is repeated in about 30 places
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Repeated Switches, Shotgun Surgery). Ticket 07 chose one render path with a surface switch, not this scattering.
- Where: `surface == "python"`-style branches in _columns.py, _project_file.py, _row_ids.py, _text.py, _wording.py, data/validation.py, discovery.py, results.py (457-528 and 850-913) and data/targets.py. targets.py keeps `_REGRESSION_FIXES` at :500 and a parallel `_MCP_REGRESSION_FIXES` at :534.
- Evidence: Confirmed: a grep for `surface (==|!=) "(python|mcp)"|python = surface` finds 24 sites in 9 files, plus `if python` blocks.
- Why it matters: A missed branch is exactly how STD-8 happened.
- Fix direction: Pass one wording object per surface (call syntax, help pointer, setting spelling, field names) in place of the string.

### STD-58: Unclear names in core, analysis and test code
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Mysterious Name).
- Where:
  - `_text._RECORD` (footer phrases, _text.py:104) against `results._RECORD` (a serialization-context key, results.py:149).
  - `_project_file.invalid(...)` builds a ProjectFileError, and `_project_file._kind` returns TOML type names.
  - `_association.py:526` has a variable named `trash`.
  - `_levels.py:68`: `masked_types` switches `Levels.values` from labels to placeholder numbers.
  - _conversions.py:55, :178 and :220: `COMPATIBLE, CONVERTIBLE, INVALID = 0, 1, 2` index lists, and `_stored_verdict(kind, declared)` takes a ColumnKind where a LogicalType is meant.
  - Tests: a parametrized argument named `given` shadows Hypothesis' `given` (tests/test_profile_target.py:543-554). `codes()` means product codes in tests/test_validate_column_types.py:342 and finding codes in tests/test_describe_dataset.py:478.
- Evidence: Read-only: the quoted definitions.
- Why it matters: Readers have to know which mode or meaning a name has, and the overlapping vocabularies work only because the strings coincide.
- Fix direction: Rename them, for example `_FOOTER_RECORD`, `invalid_project_error`, `toml_type` and `missing_slot`. Use an IntEnum for the verdicts and a separate small type for masked levels.

### STD-59: Helpers and constants are copied across test modules instead of shared
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call (Duplicated Code). tests/recipes.py already exists as the shared helper module.
- Where:
  - MCP test helpers:
    - the in-memory `async def call` is in 8 test_mcp_* modules, and `list_tools` in 5;
    - "read the file as the MCP server does" is in test_mcp_errors.py:225, test_mcp_tables.py:385, test_mcp_project.py:500 and test_mcp_fixes.py:295;
    - the Parquet DeprecationWarning filter appears 3 times, plus once in src/agent_eda/_data_files.py:123.
  - Constants copied between test modules:
    - the mixed CSV in test_mcp_tools.py:46 and test_mcp_tables.py:33;
    - the eight statuses and FIRST_LOOK in both test_guide.py and test_mcp_discovery.py;
    - the guardrail sentence, 4 times;
    - the CALL and bare-server regexes, twice each.
  - Analysis test helpers: `flowing` (3 copies), `masking_project` (3, one with an extra parameter), `run_python` (3), `reject_constant` (5, with two different messages), `findings_by_code` (2), and docstring-section splitting (6 sites).
  - The `[info] = [... for ... in ae.list_analyses().analyses if ...]` lookup appears 37 times in 25 files, though test_list_analyses.py:11 defines `info_for`.
  - test_mcp_discovery.py:22 imports `SEARCH_ACCEPTANCE` from another test module.
- Evidence: Confirmed.
  - `grep -nE "^(async )?def " tests/test_mcp_*.py`: every `call` body is `async with Client(create_server(), mode="legacy") as client: return await client.call_tool(name, arguments)`, three of them with `server=` or `project=` variants.
  - The STD-T, STD-U and STD-I forks' greps confirm the rest.
- Why it matters: A change to the client mode, project construction or the strict-JSON check needs up to eight edits, and the copies already differ. The cross-module import fails under `--import-mode=importlib`.
- Fix direction: Add a `tests/mcp_helpers.py` imported after the `importorskip`, and move the analysis helpers and constants into recipes.py or a `tests/helpers.py`.

### STD-60: Some tests can't fail, or check less than their names say
- Kind: test-gap
- Severity: P3
- Hard violation or judgement call: Judgement call (vacuous assertions, misleading names), plus the mcp-server spec's test group 6 ("the rows equal to_frame()").
- Where:
  - tests/test_detect_missing_sentinels.py:782-787: `test_a_named_id_column_is_checked_too` asserts only the echoed `parameters["columns"]`, and its ID column holds no stand-in.
  - tests/test_find_duplicate_rows.py:303-306 and tests/test_find_candidate_keys.py:249-256 loop over `duplicate_groups` and `most_unique_column_sets` without asserting that there are any.
  - tests/test_mcp_tables.py:160-181 checks each analysis's written table only for column names and row count.
  - tests/test_benchmark.py:187-199 passes whatever status an analysis returns.
  - tests/test_masking.py:59-63: `test_both_analyses_that_show_raw_values_support_masking` checks only the two names from ticket 08. The spec's exact five are pinned only in tests/test_mcp_project.py:348, which is skipped in the core minimum-versions run.
  - tests/test_project.py:115-134 says DataPolicy takes names "as the analyses' columns does", but a set works there and raises in `columns`.
- Evidence: Read-only: the quoted asserts. Confirmed by f8/idcol.py, which shows that a stand-in placed in the ID column does become a candidate: the behaviour holds today, but nothing guards it. Also confirmed by f9/setcols.py: `DataPolicy(sensitive_columns={"b","a"})` gives `('a', 'b')`, while `columns={"b","a"}` raises.
- Why it matters: Regressions in these behaviours pass the suite.
- Fix direction:
  - Put a stand-in in the ID column, and assert the counts before the loops.
  - Compare the written cells after the documented conversions, and assert `status == "completed"` in the benchmark.
  - Assert `sorted(MASKING) ==` the five names in the core suite, and rename the set test.

### STD-61: Some of the spec's test groups have coverage gaps
- Kind: test-gap
- Severity: P3
- Hard violation or judgement call: Judgement call against the first-look spec's Testing Decisions (groups 1, 2 and 4) and AGENTS.md Tests ("pandas 2 and 3 string dtypes").
- Where:
  - tests/strategies.py:35-43: `_str_dtype()` always builds `pd.StringDtype("pyarrow", na_value=np.nan)`. The `str` dtype pandas 3 uses without pyarrow (python storage, NaN as missing) is never generated, though that is the offline AgentCore case.
  - No test builds a period or interval column, although several analyses document behaviour for them.
  - tests/test_find_candidate_keys.py:399-418 checks the counts of the reported sets, not that keys are unique, minimal and complete. Ticket 14's brute-force check ran only in a scratch script.
  - describe_dataset's counts and memory are never compared with pandas on generated frames; ticket 03's claim that they equal `memory_usage(deep=True)` isn't a test.
  - Cross-process determinism is tested for describe_dataset only, which is why STD-4 got through.
- Evidence: Confirmed by the STD-U fork's greps and probes:
  - f9/pystr.py: all 11 analyses pass on python-storage `str` today.
  - f9/periods.py and f9/intervals.py: no crash on period or interval columns.
  - f9/keys_brute.py: 600 frames, 0 problems.
  - `grep -nE "nunique|isna\(\)\.sum|memory_usage" tests/*.py` finds no comparison of describe_dataset with pandas on generated data.
- Why it matters: Regressions in these paths go unnoticed, and the property suite would catch several of them cheaply.
- Fix direction:
  - Add `str (python)`, `period[M]` and `interval[float64]` kinds.
  - Keep the brute-force check and the pandas count comparison as properties.
  - Run the PYTHONHASHSEED comparison for every analysis.

### STD-62: A few float comparisons leave the tolerance implicit, and several text-cap tests split on "\n"
- Kind: inconsistency
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Tests ("within a stated tolerance") and first-look ticket 03 (the text check splits with `splitlines()`).
- Where:
  - Bare `pytest.approx`: tests/test_profile_target.py:180, tests/test_summarize_numeric.py:294 and :305, and tests/test_masking.py:428.
  - tests/test_infer_column_semantics.py:120 and :206 compare shares with exact `==`, while :172 uses `abs=1e-15`.
  - `split("\n")`: tests/test_summarize_numeric.py:891, tests/test_summarize_categorical.py:795, tests/test_mcp_discovery.py:54, tests/test_describe_analysis.py:404 and tests/test_search_analyses.py:66.
- Evidence: Confirmed by `grep -n "approx(" ... | grep -v "rel=\|abs="`; the modules state `RTOL = 1e-12` elsewhere.
- Why it matters: A reader can't tell which comparisons are exact on purpose, and the split-based tests would miss a line-separator regression.
- Fix direction: Pass `rel=RTOL` or a commented `abs=`, and use `splitlines()`.

### STD-63: Generated test data shares mutable objects across examples and orders a set of types by hash
- Kind: smell
- Severity: P3
- Hard violation or judgement call: Judgement call against AGENTS.md Tests (independent of test order and of unseeded randomness).
- Where:
  - tests/strategies.py:247 and :271: the `ATTRS` lists and the `MIXED_VALUES` objects are reused in every example. The same happens at tests/test_find_duplicate_rows.py:768 and :782.
  - tests/recipes.py:230-232: `kinds = {type(value) for value in kept}` is iterated to build the candidates that `st.sampled_from` draws from. `label_pool`, at :403-404, sorts the same kind of set.
- Evidence: Confirmed.
  - After `df.attrs = a`, `df.attrs['tags'] is a['tags']` is `True`.
  - `list({type(v) for v in ['a',1,2.5,True]})` prints different orders in different processes (f9).
  - The candidate order happened to be stable in 12 runs, so the hazard is latent.
- Why it matters: A mutation would leak into later examples, and a hash-dependent draw order makes a CI failure hard to reproduce.
- Fix direction: Build `ATTRS` and `MIXED_VALUES` per example (or deepcopy them), and iterate `sorted(kinds, key=lambda k: k.__name__)`.

### STD-64: Tests import anyio and jsonschema directly, but no dependency group declares them
- Kind: ci-release
- Severity: P3
- Hard violation or judgement call: Judgement call (implicit dependency). The mcp-server spec allows jsonschema, and mcp-server ticket 01 chose anyio.
- Where: `import anyio` in 9 test modules and `import jsonschema` in 3; pyproject.toml `[dependency-groups]`. test_contract.py's `annotated_types` likewise comes only through pydantic.
- Evidence: Confirmed. The import grep counts 9 and 3 modules. `grep -n "anyio\|jsonschema" pyproject.toml` finds nothing (exit 1). In uv.lock both appear only as dependencies of mcp 2.3.0.
- Why it matters: If mcp drops either one, the MCP test modules fail at import after `importorskip` has already passed.
- Fix direction: Add anyio and jsonschema to the dev or mcp group.

## Questionable decisions

Each item below was chosen by the document named, but may cost agents something.

**Masking**
- profile_missing_values shows declared sentinel values unmasked, for example `{"value": "ann@x.com", "count": 2}` for a sensitive column, while detect_missing_sentinels masks the same values. describe_analysis also says its results show no raw values. Chosen by the first-look spec ("The other six never show raw values") and ticket 11.
- Placeholders are numbered per column, so profile_target can print "<masked 3> in row <masked 1>" with numbers from two columns (first-look spec).

**Statuses and nulls**
- Every analysis declares `failed` with no code path and says "None is known today", so agents may write branches that never run (first-look spec: "Every analysis can also return failed").
- profile_target leaves the other task's fields null without a limitation (ticket 15).
- find_duplicate_rows reports `duplicate_share` as 0.0 under not_applicable, when nothing was compared (ticket 07).
- validate_column_types returns completed on data without rows, while the other analyses return insufficient_data (ticket 13).
- `method` is null for all 11 analyses (first-look spec, ticket 16).

**Order of checks**

Some work runs before argument errors are raised, against AGENTS.md's "before any computation starts":
- `read_row_ids` factorizes `id_column` before a `columns` typo is reported (ticket 07).
- profile_target checks label types after counting them (ticket 15).
- The MCP server reads the whole data file before the analysis validates its options (mcp-server ticket 04).

**MCP wording and payloads**
- Status reasons, limitation reasons and record text keep Python syntax over MCP, such as `df.reset_index(drop=True)` and `.astype('float64')`. MCP errors echo JSON values as Python reprs, such as "got None" and "got True". Chosen by the mcp-server ticket 04 and 07 follow-ups; STD-39 lists the parts that could change.
- MCP tool descriptions and input schemas keep some Python: `res.columns["age"]`, `df.loc[row_ids]`, `columns=None`, `table="matrix"` and "None, the default" (mcp-server ticket 02).
- An MCP error result sends `{"code", "message"}` as structured content, which fails the MCP tool's output schema in strict clients (mcp-server ticket 04).
- An internal error forwards `str(error)` of an unexpected exception, which could carry a raw value past masking (mcp-server ticket 04).
- `create_server(analyses=...)` is a public parameter used only as a test seam, and it serves unregistered functions without the registry's checks (mcp-server ticket 02).
- ADR 0006 chose one MCP tool per analysis, with full descriptions and output schemas:
  - `tools/list` measured 264,126 characters for 13 MCP tools, and records 26 to 78 KB.
  - Constant prose fields add to every record (first-look tickets 02, 06, 07 and 15): `definitions` is 1,729 of 3,209 bytes in a one-column summarize_numeric record, and `level_order`, `share_denominator` and `missing_values_compare_equal` (always true) add more.
  - The instructions are at 59 of 60 lines and the guide at 60 of 60, so the next analysis breaks both (mcp-server ticket 07).
- The MCP record's `parameters` keeps the Python names `data` and `project` (mcp-server tickets 07 and 08).

**Results and tables**
- `to_frame()` isn't complete for row IDs: detect_missing_sentinels and validate_column_types keep 10, and profile_target keeps none (tickets 11, 13 and 15). Over MCP, `table_path` can't reach the rest.
- A stored NaN counts differently in each analysis: missing in summarize_numeric and compute_association_matrix, a level in summarize_categorical, a value in describe_dataset (first-look tickets 01, 02, 06, 10, 15 and 16). STD-14 covers the missing limitations.
- Fact lines are cut at 120 characters rather than wrapped, so a long `source` drops `source_version` (ticket 01).

**Arguments and defaults**
- `sentinel_values`, `declared_schema` and `schema` are keyed by column name, though their names don't end in column or columns (first-look spec; ticket 10's KEYED_BY_COLUMN).
- These have defaults although AGENTS.md lists only conventions and limits: `numeric_method='pearson'`, `categorical_method='cramers_v_bias_corrected'` and `trim_fraction=0.1`. Also, `trim_fraction` is a fraction not named `…_share`, unlike `rare_share` (first-look spec).
- `search_analyses(query, limit)` takes `limit` by position (ticket 04).

**Analysis limits and design**
- compute_association_matrix (first-look spec, ticket 16):
  - `max_categories` is a `max_*` parameter that skips columns instead of returning budget_exceeded.
  - `max_pairs` is checked only after every column is read and factorized.
  - The pair count doesn't bound the time of rank methods on wide frames, which can run for minutes.
  - The last digits depend on the BLAS thread count.
- find_candidate_keys' ranking cap has no `max_*` parameter (ticket 14).
- infer_column_semantics compares a declared type only with the top candidate, so an exact tie, such as a 0/1 column declared integer, becomes `declared_type_mismatch`. It also ignores a declared `datetime_format` (first-look spec, ticket 12).
- `all_labels_distinct` fires on two rows of 0 and 1 (ticket 15).

**Project and errors**
- `Project(data_policy=<dict>)` and `project=<dict>` accept a JSON mapping, against AGENTS.md's "no JSON mappings in public signatures". Assigning to a frozen field raises pydantic's raw ValidationError. DataPolicy accepts a set, sorted, while `columns` rejects one. All chosen by ticket 08.
- Wrong keywords to `describe_analysis`, `search_analyses`, `Project.load` and `save` raise bare TypeErrors with no code, and a misspelled `ae.<attr>` raises a plain AttributeError (tickets 09 and 01).
- `save(overwrite=True)` truncates the file in place. An interrupted save is caught on load, but the old file is gone (ticket 09).

**Packaging, globals and search**
- `_registry._REGISTRY` is module-global state, filled at import (first-look spec, registration decorator).
- scikit-learn and statsmodels are required dependencies that no module imports (first-look spec, ADR 0005).
- Search tags include words the glossary avoids: "semantic type", "data types", "placeholder", "missing codes" and "magic number" (tickets 04, 11 and 12). They translate agents' wording, which seems right.

**Test design**
- Skewness and kurtosis are checked against references computed on `x - median(x)` (ticket 02).
- The bias-corrected Cramér's V published example is checked only to ±0.005 (ticket 16).
- Quantiles are compared with exact equality.
- The brute-force check and the pandas memory check ran only in scratch scripts (tickets 03 and 14).
- Masking membership is pinned only in an MCP test (mcp-server ticket 05).
- The MCP tests' reference read repeats the server's own `read_csv` call (mcp-server spec), which is why STD-1 can't fail a test.

## Checked and fine

- **Call pattern:** All 11 analyses take data first and every option by keyword, and the registry enforces both at registration. Column parameters are `columns`, `id_column`, `target_column` and `categorical_columns`. `columns=None` selects the suitable columns in DataFrame order, and an explicit list keeps its order.
- **Typed arguments:** Every fixed choice is a Literal (task, the methods, quantile_method, `to_frame`'s table). Ranges are Annotated, and every range and choice appears in the Args text (std/docs_check.py). No analysis signature takes `dict[str, Any]`.
- **Defaults:** Only conventions and work or display limits have defaults. `_check_result` forces `parameters` to record every argument in signature order. Project-specific inputs (sentinel values, expected labels, rare_share, schema, nullable) have no defaults.
- **Read-only data:** check_contract's snapshot holds. The forks traced every `to_numpy()` view: no in-place writes, no `inplace=True`, no `.sort()` on caller data, and no lookup table built on the caller's index.
- **Results:** All 11 result classes are named after their analyses and subclass AnalysisResult. Every field of all 59 models has a description. Every 0–1 fraction ends in `_share`. The caps of 10, 5 and 100 come with totals. `to_json` uses `allow_nan=False` and the models set `allow_inf_nan=False`. There are no HTML display hooks. The null walk found only nulls that are designed or explained by a limitation, apart from the cases above.
- **Statuses:** The forks traced every return site of all 11 analyses. Each analysis returns only declared statuses, and every `Statuses:` section lists exactly the declared set, in order. `budget_exceeded` names its limit and gives the estimate.
- **Errors:** errors.py's docstring lists exactly the 16 codes raised in src. Unknown columns list the available columns and the closest names, ignoring case. Masking is resolved before any computation, and a per-call `mask_sensitive` wins over the Project's setting.
- **Warnings:** No `warnings.warn` in src. RuntimeWarnings become `numerical_warning` limitations. A probe of all 11 analyses on mixed data with `simplefilter("always")` let no warning escape. The warning filters in src and tests are scoped to specific messages, so pyproject's DeprecationWarning and FutureWarning errors still apply.
- **Names and exports:** Analysis names start with a verb and are unique, and `__test__ = False` is set by the decorator. `__all__`, `_EXPORTS` and the TYPE_CHECKING imports agree, and `dir(ae)` lists them.
- **Docstrings:** All 11 have the seven sections in order, summaries ending in "?", every parameter with its type and what None means, and every finding code defined. Their doctests print a result.
- **Determinism:** Apart from STD-4 and STD-5, all 11 analyses give identical records under PYTHONHASHSEED 1–7 on a mixed frame. Discovery JSON is identical across seeds. MCP structured content stays byte-identical after other calls use the cached DataFrame. There is no randomness, and module dicts are constants, apart from the registry.
- **Compatibility:** No NumPy 2-only functions (`unique_counts`, `trapezoid`, `np.strings`). The object, string, str, string[pyarrow_numpy] and Arrow text dtypes are classified. The MCP probes also ran on pandas 2.2.0, numpy 1.26.0 and pyarrow 14.0.1.
- **Lazy loading and files:** `import agent_eda` and the discovery functions load no scipy, statsmodels, sklearn or mcp. Nothing reads `__file__`, and there are no bundled files, so the importlib.resources rule doesn't apply.
- **Masking:** The forks probed the text, `to_dict`, `to_json`, `to_frame`, `repr` and pickles of all five masking analyses, over Python and over MCP, and found no leak.
- **MCP server:**
  - It talks over stdio only, opens no network connections and starts no threads.
  - The read cache belongs to each MCP server instance.
  - Data paths that look like URLs are read as local relative paths.
  - table_path is checked before the analysis runs and written with "xb".
  - Every AgentEDAError becomes `isError` with `{code, message}`, and any other exception becomes JSON-RPC error -32603.
- **Tests:**
  - No test imports a private module or uses a private attribute, and no test or helper function is named like an analysis.
  - Hypothesis runs derandomized through the conftest profile, with no overrides.
  - All randomness is seeded.
  - Files go only to tmp_path, and cwd and env changes go through monkeypatch or subprocess env.
  - The spec's test groups are present, apart from the gaps listed above.
- **Glossary:** No "insight", "issue", "alert", "severity", "PII", "redaction", "endpoint", "gateway", "config", "context" or "session" in caller-facing text in src. "Placeholder" appears only in its masking sense, and "anonymization" only where masking is said not to be anonymization.
- **Tool config:** pre-commit runs ruff and ty on commit and pytest on push, as AGENTS.md says. Ruff's 124-character limit fits the 120-character docstring examples. pyproject's `filterwarnings` turns DeprecationWarning and FutureWarning into errors.

## Not covered
- I didn't run the full suite or the minimum-versions suites; the coordinator's baseline says they are green. Fork probes ran on the minimum versions only for the MCP reader and writer and for `record_schema()`.
- Out of my scope: README.md and .github/workflows/ci.yml, except where tests read them.
