# 01: describe_dataset end to end, with the package, analysis result and registry

**What to build:** An agent or a person installs agent-eda, runs `import agent_eda as ae`, calls `ae.describe_dataset(df)` on any DataFrame and reads the analysis result in every form the spec defines: printed text within fixed caps, `to_frame()`, `to_dict()` and strict `to_json()`.

The repo has no code yet, so this first slice builds every shared part a later analysis needs and proves them end to end with one analysis. The shared parts are the analysis result base class with its findings, limitations, support and statuses, the text template, the registry with `ae.list_analyses()`, the error base class and the docstring contract test. The slice goes as far as packaging. The built wheel has to import straight from `sys.path`, the way a sandboxed AgentCore agent uses it.

Read these sections of `.scratch/first-look/spec.md`: Package and build, Public API layout, Calling convention, Analysis results, Statuses, Findings and limitations, Errors, Docstrings, the describe_dataset entry under The first-look analyses, and test groups 1, 3 and 8 under Testing Decisions. Covers user stories 1, 4, 5, 7, 8, 15, 19 to 26, 32 to 38, 40, 42 to 49, 53, 54, 59, 62, 63, 79 to 81, 119, 121 and 122.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

Package and imports:

- [x] The project uses a src layout and the uv build backend. `uv build` produces a pure-Python `py3-none-any` wheel that contains a `py.typed` marker and declares exactly the six required dependencies at their current minimums.
- [x] `ae.__version__` reports the installed version, also when the package is imported straight from the wheel file.
- [x] A test builds the wheel, puts the wheel file on `sys.path` in a fresh isolated interpreter, imports `agent_eda`, runs describe_dataset and prints the result. No code reads package files through `__file__`, and any bundled file loads through `importlib.resources`.
- [x] In a fresh subprocess, `import agent_eda` followed by `ae.list_analyses()` leaves SciPy, statsmodels and scikit-learn unimported.
- [x] The top level exports `describe_dataset`, `list_analyses`, `AnalysisResult` and `__version__` and loads them lazily. Type checkers and editors still see every export, and `dir(agent_eda)` lists them. `agent_eda.errors` and `agent_eda.results` are public modules, and describe_dataset lives in `agent_eda.data.provenance`.
- [x] pyarrow joins the development dependencies.
- [x] A documented command runs the test suite on Python 3.12 against the minimum dependency versions, numpy 1.26 and pandas 2.2 among them, without changing `uv.lock` or the default environment. The suite passes there, and the README's development table lists the command.

Analysis result:

- [x] `ae.AnalysisResult` is a frozen pydantic model with the shared fields `analysis`, `method`, `status`, `parameters`, `support`, `findings`, `limitations` and `execution`. describe_dataset returns `DescribeDatasetResult`, a subclass whose own typed fields sit next to the shared ones. Every field has a description.
- [x] `parameters` records the resolved arguments, defaults included, and shows the DataFrame only as its row and column counts. `support` records the rows given, the columns selected and, per column, the eligible, evaluated and excluded counts with reasons. `execution` records the versions of agent-eda, Python, numpy, pandas and any other library the analysis imported. Results hold no timestamps or durations.
- [x] `status` is one of the eight statuses in the spec. Each analysis declares the statuses it can return, and a result never carries an undeclared one. A `failed` result carries an error code and a message. Unexpected exceptions, which mean a bug, propagate.
- [x] A finding carries a stable code, a one-sentence claim from a fixed template, an assessment of descriptive, suspected, violation, passed or unknown, the columns it concerns and the result field paths behind it. A limitation carries a code from one package-wide list and the fields it affects.
- [x] `to_dict()` returns JSON-compatible Python objects, and `to_json()` returns strict JSON with non-finite numbers forbidden. Two identical calls give byte-identical JSON, also in two processes with different `PYTHONHASHSEED` values.
- [x] Every list in the record that grows with the data is capped and carries its total: 10 row IDs, 5 example values and 100 entries for anything else. The spec doesn't say whether per-column entries count, and this ticket settles that they do. The record keeps the first 100 columns plus the total, while the result object and `to_frame()` keep every column.
- [x] `str()` and `repr()` give the same plain text from a fixed template. The first line names the analysis, the status and the data size. The key table follows with at most 10 rows, then at most 5 findings, at most 5 limitations and a footer on how to get the rest. The text never exceeds 40 lines or 120 characters per line, and it counts whatever it leaves out, for example "10 of 40 columns shown". Numbers show 4 significant digits, and counts use thousands separators.
- [x] IPython and Jupyter display a result as the same plain text. Results have no HTML display.
- [x] `to_frame()` returns the complete per-column table.

describe_dataset:

- [x] `data` is the only positional parameter. `source` and `source_version` are optional keyword-only strings.
- [x] The result gives the row and column counts; per column, the dtype as pandas reports it, the non-missing, missing and distinct counts and the memory in bytes; the total memory; the index dtype, whether the index is unique and how many index labels are missing; and the provenance as supplied. Anything not supplied is recorded as not supplied, never inferred.
- [x] Findings cover duplicated or colliding column labels, MultiIndex columns, a non-unique index, empty data, all-missing columns and constant columns. describe_dataset reports bad column labels as findings, while every other analysis raises on them.
- [x] No field or finding shows a raw data value. The constant-column finding names the column, not its value.
- [x] The declared statuses are `completed` and `failed`. The status is `completed` also for zero rows, zero columns or both.
- [x] It handles NumPy numerics, bools, naive and time-zone-aware datetimes, timedeltas, the pandas nullable dtypes, pandas 3's default string dtype, categoricals, object columns and pyarrow-backed columns.
- [x] The input DataFrame keeps its values, dtypes, index, column order and `attrs`.
- [x] Passing a Series, a NumPy array or a polars DataFrame raises `InvalidArgumentError`, which subclasses `AgentEDAError` and `ValueError` and carries a stable `code`. The message shows the one-line conversion.
- [x] The catalogue entry for describe_dataset is marked implemented, with its differences from the catalogue noted. This ticket picks the marking convention the later tickets follow.

Registry and tests:

- [x] A registration decorator records each analysis's summary, tags, parameters, methods, declared statuses, declared finding codes, result class, masking support and work-limit parameters, and sets `__test__ = False` on the function.
- [x] `ae.list_analyses()` returns a typed listing with each analysis's name, module, one-line summary and masking support. It prints compactly and has `to_frame()`.
- [x] A contract test loops over `ae.list_analyses()` and checks each analysis's docstring:
  - the seven Google-style sections appear in order: a summary line phrased as a question ending in "?", `When to use:`, `Args:`, `Returns:`, `Statuses:`, `Examples:` and `See Also:`
  - `Args:` documents every parameter and nothing else
  - `Returns:` names every declared finding code, and `Statuses:` lists exactly the declared statuses
  - every name in `See Also:` resolves in `agent_eda`
  - the doctest example runs and prints a result
  - `__test__` is False
- [x] `When to use:` and `See Also:` name only analyses and functions that exist. Later tickets extend them as neighbouring analyses arrive.
- [x] Every test goes through the public `agent_eda` API, and none imports a private module.
- [x] A DeprecationWarning or FutureWarning raised while package code runs fails the test suite.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, and the suite also passes on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **Capped lists in the record.** A field that grows with the data is marked `Annotated[list[X], _ENTRIES]` (or `_ROW_IDS`, `_EXAMPLE_VALUES`) in `agent_eda/results.py`. `to_dict()` and `to_json()` keep its first 100 (10, 5) entries and always add a sibling key `<field>_total`, so `record["columns"][0]` still mirrors `res.columns[0]`. Dict fields are capped the same way. Per-column entries count, as the ticket says, and so do every finding's `columns` and `fields`, a limitation's `fields`, and the result's `findings` and `limitations`. Inside `parameters`, every list argument (such as a resolved column list) keeps 100 entries and gains `<name>_total`. The result object and `to_frame()` keep everything.
- **Shared fields.** Besides the eight listed, `AnalysisResult` has `status_reason` (`code`, `message`): the error code and message of `failed`, and the reason for every other status that isn't `completed`. A validator makes it null exactly when the status is `completed`. `parameters` is `dict[str, JsonValue]` in signature order, with `data` as `{"row_count", "column_count"}`. `execution` is `versions`, keyed by import name: agent_eda, python, numpy, pandas, then whatever the analysis passes to `_run.execution()`.
- **Findings** are one per code, aggregated over columns: `columns` lists every column concerned, and the claim names five and counts the rest ("'c0', 'c1', 'c2', 'c3', 'c4' and 3 more"). Field paths index the result object: `columns[2].missing_count`, `index.is_unique`. Limitation codes are the `LimitationCode` literal in `agent_eda.results`; so far only `unhashable_values`.
- **Text template.** First line `<analysis>: <status> on <n> rows x <m> columns`, plus `, method <m>` when there is one; `Reason: <code>: <message>` for a status other than completed; the analysis's fact lines; `<Title> (k of n shown):` and the table; `Findings (k of n shown):` with `- <assessment> <code>: <claim>`; `Limitations (k of n shown):` with `- <code>: <reason>`; and a `More: ...` footer. An empty section reads `Findings: none`. Longer lines are cut with `...`, and control characters and line separators in labels are escaped, so a label can't add lines. Byte counts from 1 KiB up add a binary unit at 4 significant digits ("18.84 MiB (19,753,072 bytes)").
- **Calls.** The registration decorator wraps each analysis: it checks the call's shape, that `data` is a DataFrame, and the keyword options against a strict pydantic model built from the signature, all before any computation. It then checks that the result has a declared status and declared finding codes and records exactly the signature's parameters; a mismatch is a bug and raises RuntimeError. Error codes so far: `invalid_argument`, `invalid_call` (an option passed by position, an unknown keyword with the closest name, missing data) and `unsupported_data_type`. A bad argument value appears in the message only if it is a scalar, shortened to 40 characters; a DataFrame, Series, array or container appears as its type and size, so a column passed by mistake can't put raw values into the message.
- **For ticket 02.** Validation is strict, so as it stands a `list[str]` parameter would reject a tuple or `columns=df.columns` (a pandas Index), and an `int` parameter would reject a NumPy integer. Convert those before validating, or relax strictness per field. The wrapper doesn't yet hide DeprecationWarning and FutureWarning from callers or turn RuntimeWarnings into limitations; ticket 02 has the boxes for both.
- **Discovery.** `ae.list_analyses()` entries carry everything the decorator records, so tests read declared statuses and finding codes through the public API. The listing is ordered by module, then name. The analyses are the decorated functions in the modules named in `agent_eda._EXPORTS`, the single list behind lazy loading. `ae.errors` and `ae.results` also resolve as attributes, and a misspelled attribute suggests the closest export.
- **Contract test.** Besides the ticket's checks, it requires every snake_case word in `When to use:` and `See Also:` to resolve in `agent_eda` unless it names the analysis's own parameters, result fields, finding codes, statuses or limitation codes, so an unquoted "use summarize_numeric" fails until that analysis exists. It also checks that every result field has a description and that the result class is named after the analysis.
- **describe_dataset.** Missing means what pandas `isna()` counts (NaN in a `double[pyarrow]` column is a value, not missing). Constant means at least two non-missing values and one distinct value; all-missing needs at least one row. Column labels appear as their string form (`column`) with `label_type`, and MultiIndex labels as tuple strings. Labels are duplicated exactly when pandas' `Index.duplicated()` says so: 1, 1.0 and True are one label, while None and NaN are two. Repeated index labels are counted on a Series over the index values, because `Index.duplicated()` first asks `is_unique`, which builds and keeps the caller's lookup table; a test checks that the caller's index and columns keep their memory size. Values that can't be hashed (lists, dicts, nested Arrow types) give a null distinct count and an `unhashable_values` limitation instead of a failure. `failed` is declared, but no code path returns it yet.
- **Memory** counts the values only. pandas' `memory_usage()` of an index or a categorical adds lookup tables that earlier lookups build, and on pandas 3, rebuilding a Series from object text re-infers it as `str`; tests cover both. One limit remains: `sys.getsizeof` of a non-ASCII `str` grows once Python caches its UTF-8 form (pandas 2.2's `nunique()` does this), so object-text memory can differ between processes whose earlier code touched the strings differently. Identical scripts give identical numbers.
- **Catalogue convention:** the function cell gets "**Implemented in agent-eda 0.1** as `module.name`" and the contract cell ends with "**Implemented differently:**" listing every departure. "How to use this document" describes it.
- **Tooling.** `lint.pycodestyle.max-line-length = 124` lets docstring examples show printed lines of up to 120 characters; the formatter still wraps code at 88. Dev dependencies: `pyarrow>=14.0.1` (pyarrow 26 refuses NumPy 1.x at import without declaring it), `polars>=1.0` and `ipython>=9.0`, so the polars error and the IPython display are tested against the real libraries. `ruff format .` also reformats Python blocks in Markdown, so it touched the README's usage block. The packaging tests build the wheel through the `uv_build` PEP 517 hook, from a `uv-build` dev dependency pinned to the `[build-system]` range, so they need no network and no particular uv version.

Measurements (M3 Max, Python 3.12.11; ticket 03 measures on the spec's benchmark and target machine):

- describe_dataset on 1,000,000 rows x 100 columns (25 float, 15 Int64, 21 text, 15 categorical, 10 bool, 14 datetime, about 5% missing) took 0.9 s with pandas 3.0.6 and 2.7 s with pandas 2.2.0. On pandas 2.x, object-dtype text dominates.
- A review run with 25 object-text columns, 5 of them all-distinct, took 5.95 s on pandas 2.2.0 and 1.9 s on pandas 3.0.6. Each all-distinct object column costs about 0.65 s: 0.34 s for the distinct count, 0.22 s for deep memory and 0.09 s for `isna()`. So describe_dataset will likely miss the 5-second target on pandas 2.x with high-cardinality text. Ticket 03's benchmark should settle this; one saving is to take the missing and distinct counts of object columns from a single `pd.factorize` pass.

Checks:

- Minimum versions: `uv run --isolated --resolution lowest-direct --python 3.12 pytest` installed numpy 1.26.0, pandas 2.2.0, pydantic 2.7.0, scipy 1.11.2, pyarrow 14.0.1, polars 1.0.0 and IPython 9.0.0, and the suite passed. The `uv.lock` checksum was the same before and after.
- The suite also passes on Python 3.13 and 3.14 with `uv run --isolated --python 3.1x pytest`. Linux x86_64 and ARM64 weren't run locally; ticket 05's CI covers them.
