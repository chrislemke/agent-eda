# 01: describe_dataset end to end, with the package, analysis result and registry

**What to build:** An agent or a person installs agent-eda, runs `import agent_eda as ae`, calls `ae.describe_dataset(df)` on any DataFrame and reads the analysis result in every form the spec defines: printed text within fixed caps, `to_frame()`, `to_dict()` and strict `to_json()`.

The repo has no code yet, so this first slice builds every shared part a later analysis needs and proves them end to end with one analysis. The shared parts are the analysis result base class with its findings, limitations, support and statuses, the text template, the registry with `ae.list_analyses()`, the error base class and the docstring contract test. The slice goes as far as packaging. The built wheel has to import straight from `sys.path`, the way a sandboxed AgentCore agent uses it.

Read these sections of `.scratch/first-look/spec.md`: Package and build, Public API layout, Calling convention, Analysis results, Statuses, Findings and limitations, Errors, Docstrings, the describe_dataset entry under The first-look analyses, and test groups 1, 3 and 8 under Testing Decisions. Covers user stories 1, 4, 5, 7, 8, 15, 19 to 26, 32 to 38, 40, 42 to 49, 53, 54, 59, 62, 63, 79 to 81, 119, 121 and 122.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

Package and imports:

- [ ] The project uses a src layout and the uv build backend. `uv build` produces a pure-Python `py3-none-any` wheel that contains a `py.typed` marker and declares exactly the six required dependencies at their current minimums.
- [ ] `ae.__version__` reports the installed version, also when the package is imported straight from the wheel file.
- [ ] A test builds the wheel, puts the wheel file on `sys.path` in a fresh isolated interpreter, imports `agent_eda`, runs describe_dataset and prints the result. No code reads package files through `__file__`, and any bundled file loads through `importlib.resources`.
- [ ] In a fresh subprocess, `import agent_eda` followed by `ae.list_analyses()` leaves SciPy, statsmodels and scikit-learn unimported.
- [ ] The top level exports `describe_dataset`, `list_analyses`, `AnalysisResult` and `__version__` and loads them lazily. Type checkers and editors still see every export, and `dir(agent_eda)` lists them. `agent_eda.errors` and `agent_eda.results` are public modules, and describe_dataset lives in `agent_eda.data.provenance`.
- [ ] pyarrow joins the development dependencies.
- [ ] A documented command runs the test suite on Python 3.12 against the minimum dependency versions, numpy 1.26 and pandas 2.2 among them, without changing `uv.lock` or the default environment. The suite passes there, and the README's development table lists the command.

Analysis result:

- [ ] `ae.AnalysisResult` is a frozen pydantic model with the shared fields `analysis`, `method`, `status`, `parameters`, `support`, `findings`, `limitations` and `execution`. describe_dataset returns `DescribeDatasetResult`, a subclass whose own typed fields sit next to the shared ones. Every field has a description.
- [ ] `parameters` records the resolved arguments, defaults included, and shows the DataFrame only as its row and column counts. `support` records the rows given, the columns selected and, per column, the eligible, evaluated and excluded counts with reasons. `execution` records the versions of agent-eda, Python, numpy, pandas and any other library the analysis imported. Results hold no timestamps or durations.
- [ ] `status` is one of the eight statuses in the spec. Each analysis declares the statuses it can return, and a result never carries an undeclared one. A `failed` result carries an error code and a message. Unexpected exceptions, which mean a bug, propagate.
- [ ] A finding carries a stable code, a one-sentence claim from a fixed template, an assessment of descriptive, suspected, violation, passed or unknown, the columns it concerns and the result field paths behind it. A limitation carries a code from one package-wide list and the fields it affects.
- [ ] `to_dict()` returns JSON-compatible Python objects, and `to_json()` returns strict JSON with non-finite numbers forbidden. Two identical calls give byte-identical JSON, also in two processes with different `PYTHONHASHSEED` values.
- [ ] Every list in the record that grows with the data is capped and carries its total: 10 row IDs, 5 example values and 100 entries for anything else. The spec doesn't say whether per-column entries count, and this ticket settles that they do. The record keeps the first 100 columns plus the total, while the result object and `to_frame()` keep every column.
- [ ] `str()` and `repr()` give the same plain text from a fixed template. The first line names the analysis, the status and the data size. The key table follows with at most 10 rows, then at most 5 findings, at most 5 limitations and a footer on how to get the rest. The text never exceeds 40 lines or 120 characters per line, and it counts whatever it leaves out, for example "10 of 40 columns shown". Numbers show 4 significant digits, and counts use thousands separators.
- [ ] IPython and Jupyter display a result as the same plain text. Results have no HTML display.
- [ ] `to_frame()` returns the complete per-column table.

describe_dataset:

- [ ] `data` is the only positional parameter. `source` and `source_version` are optional keyword-only strings.
- [ ] The result gives the row and column counts; per column, the dtype as pandas reports it, the non-missing, missing and distinct counts and the memory in bytes; the total memory; the index dtype, whether the index is unique and how many index labels are missing; and the provenance as supplied. Anything not supplied is recorded as not supplied, never inferred.
- [ ] Findings cover duplicated or colliding column labels, MultiIndex columns, a non-unique index, empty data, all-missing columns and constant columns. describe_dataset reports bad column labels as findings, while every other analysis raises on them.
- [ ] No field or finding shows a raw data value. The constant-column finding names the column, not its value.
- [ ] The declared statuses are `completed` and `failed`. The status is `completed` also for zero rows, zero columns or both.
- [ ] It handles NumPy numerics, bools, naive and time-zone-aware datetimes, timedeltas, the pandas nullable dtypes, pandas 3's default string dtype, categoricals, object columns and pyarrow-backed columns.
- [ ] The input DataFrame keeps its values, dtypes, index, column order and `attrs`.
- [ ] Passing a Series, a NumPy array or a polars DataFrame raises `InvalidArgumentError`, which subclasses `AgentEDAError` and `ValueError` and carries a stable `code`. The message shows the one-line conversion.
- [ ] The catalogue entry for describe_dataset is marked implemented, with its differences from the catalogue noted. This ticket picks the marking convention the later tickets follow.

Registry and tests:

- [ ] A registration decorator records each analysis's summary, tags, parameters, methods, declared statuses, declared finding codes, result class, masking support and work-limit parameters, and sets `__test__ = False` on the function.
- [ ] `ae.list_analyses()` returns a typed listing with each analysis's name, module, one-line summary and masking support. It prints compactly and has `to_frame()`.
- [ ] A contract test loops over `ae.list_analyses()` and checks each analysis's docstring:
  - the seven Google-style sections appear in order: a summary line phrased as a question ending in "?", `When to use:`, `Args:`, `Returns:`, `Statuses:`, `Examples:` and `See Also:`
  - `Args:` documents every parameter and nothing else
  - `Returns:` names every declared finding code, and `Statuses:` lists exactly the declared statuses
  - every name in `See Also:` resolves in `agent_eda`
  - the doctest example runs and prints a result
  - `__test__` is False
- [ ] `When to use:` and `See Also:` name only analyses and functions that exist. Later tickets extend them as neighbouring analyses arrive.
- [ ] Every test goes through the public `agent_eda` API, and none imports a private module.
- [ ] A DeprecationWarning or FutureWarning raised while package code runs fails the test suite.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, and the suite also passes on the minimum dependency versions.
