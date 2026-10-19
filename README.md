# agent-eda

Deterministic exploratory data analysis (EDA) for pandas DataFrames, built for LLM agents and usable by people.

Each analysis is a plain Python function. It takes a DataFrame and typed keyword arguments and returns a typed result that prints as a short summary. The library never calls an LLM or the network, so the same data and arguments give the same result in the same environment, BLAS thread count included.

It is written for three kinds of callers:

- agents in AWS Bedrock AgentCore, through its Code Interpreter
- coding agents such as Claude Code
- people, who use it like any other package

## Features

- **11 first-look analyses**: structure, column types, missing values, sentinel values such as -999, distributions, duplicates, keys, the target and associations.
- **Results you can read and parse**: `print(res)` gives at most 40 lines, `res.to_frame()` the full tables and `res.to_json()` the whole record as strict JSON.
- **Findings, not just numbers**: each result lists what it noticed, such as "1 value in 1 column may stand in for missing values".
- **Discovery for agents**: `ae.guide()`, `ae.search_analyses()` and `ae.describe_analysis()` explain the package.
- **Helpful errors**: a mistake in a call raises an error that says what is wrong, what is allowed and the fix: a misspelled column lists the available names and the closest match.
- **Opt-in masking** of sensitive columns, such as e-mail addresses.
- **An MCP server**: `agent-eda-mcp` serves every analysis as an MCP tool over stdio, for agents that don't write Python.
- **Pure Python** on numpy, pandas, pydantic, SciPy, scikit-learn and statsmodels, preinstalled in AgentCore Code Interpreter.

## Installation

agent-eda needs Python 3.12 or later, numpy 1.26 through 2.x and pandas 2.2 through 3.x.

Until 0.1 is on PyPI, build the wheel in a clone of this repository with [uv](https://docs.astral.sh/uv/):

```sh
uv build --wheel  # writes dist/agent_eda-0.1.0-py3-none-any.whl
pip install dist/agent_eda-0.1.0-py3-none-any.whl
```

Once released, `pip install agent-eda` is enough. The MCP server needs the `mcp` extra; see [Use over MCP](#use-over-mcp).

**AgentCore Code Interpreter.** In public network mode, run `pip install agent-eda` in the session. In sandbox mode, with no PyPI access, upload the wheel and run `pip install --no-deps <wheel>`, or put the wheel on `sys.path` and import from it. Installs don't carry over to a new session. Code Interpreter returns only printed output, so `print()` results.

**AgentCore Runtime.** Add `agent-eda` to the container image or code zip. AWS's example zip command uses the `manylinux2014` tag, which installs old numpy and pandas; use `--python-platform aarch64-manylinux_2_28` for current ones.

## Quick start

Pass the DataFrame first and every other argument by keyword. Analyses never change the DataFrame.

```python
import pandas as pd

import agent_eda as ae

names = ["ann", "bo", "cy", "di", "ed", "flo", "ann", "bo"]
df = pd.DataFrame(
    {
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "email": [f"{name}@example.com" for name in names],
        "age": [34, 51, -999, 29, 45, 38, 62, 51],
        "income": [42_000, 58_500, 61_000, None, 39_500, 75_000, 52_000, 58_500],
        "plan": ["basic", "pro", "basic", "Basic ", "pro", "basic", None, "pro"],
        "churned": [0, 1, 0, 0, 0, 0, 1, 1],
    }
)

res = ae.summarize_numeric(df, columns=["age", "income"])
print(res)
```

It prints:

```text
summarize_numeric: completed on 8 rows x 6 columns
Statistics use the finite values. Quantile method linear; trimmed_mean cuts trim_fraction 0.1 from each end.
res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis.
Columns (2 of 2 shown):
column  non_missing_count  missing_count       mean       std       min    median      max
age                     8              0     -86.12       369      -999      41.5       62
income                  7              1  5.521e+04  1.21e+04  3.95e+04  5.85e+04  7.5e+04
Findings: none
Limitations: none
More: res.to_frame() has all 2 columns; res.findings, res.limitations and res.to_json() have the rest.
```

The mean age of -86.12 is the first sign of trouble. The sample data hides three problems: a -999 that stands for a missing age, a plan written both `'basic'` and `'Basic '`, and one customer recorded twice, under the IDs 102 and 108.

## Analyses

The analyses are listed in the order `ae.guide()` suggests for a first look. Those that need project knowledge, such as your schema or target, ask for it; they never guess.

| Step | Analysis | What it answers | Required arguments |
| --- | --- | --- | --- |
| 1 | `describe_dataset` | What does this dataset contain? | none |
| 2 | `infer_column_semantics` | What kind of values does each column hold? | none |
| 2 | `validate_column_types` | Do the columns hold the logical types I declared? | `schema` |
| 3 | `profile_missing_values` | How much data is missing, and where? | none |
| 4 | `detect_missing_sentinels` | Which values look like stand-ins for missing data? | none |
| 5 | `summarize_numeric` | How are the values in each numeric column distributed? | none |
| 5 | `summarize_categorical` | Which values occur in each categorical column, and how often? | none |
| 6 | `find_duplicate_rows` | Are there exact duplicate rows? | none |
| 6 | `find_candidate_keys` | Which columns, alone or combined, uniquely identify rows? | none |
| 7 | `profile_target` | What does the target look like for this task? | `target_column`, `task` |
| 8 | `compute_association_matrix` | How strongly are pairs of columns associated? | none |

Every analysis takes `columns=` to limit it to some columns, except `describe_dataset`, `profile_target` and `validate_column_types`; the default uses every column it can. [Reference](#reference) lists every function with its arguments, result fields, findings and statuses.

Run all 11 first-look analyses on the sample data and print what they suspect:

```python
schema = {"customer_id": "integer", "age": "integer", "plan": "categorical"}
results = [
    ae.describe_dataset(df),
    ae.infer_column_semantics(df),
    ae.validate_column_types(df, schema=schema),
    ae.profile_missing_values(df),
    ae.detect_missing_sentinels(df, id_column="customer_id"),
    ae.summarize_numeric(df),
    ae.summarize_categorical(df),
    ae.find_duplicate_rows(df, id_column="customer_id"),
    ae.find_candidate_keys(df),
    ae.profile_target(df, target_column="churned", task="binary_classification"),
    ae.compute_association_matrix(df),
]
for res in results:
    for finding in res.findings:
        if finding.assessment == "suspected":
            print(f"{res.analysis}: {finding.claim}")
```

```text
infer_column_semantics: 1 column looks like an identifier, every value distinct: 'customer_id'.
detect_missing_sentinels: 1 value in 1 column may stand in for missing values: 'age'.
summarize_categorical: 1 column holds labels that differ only by case or surrounding spaces: 'plan' ('basic' and 'Basic ').
find_duplicate_rows: 2 rows form 1 duplicate group; 1 is an excess copy (duplicate_share 0.125).
```

Lines 2 to 4 are the planted problems. The first is expected: `customer_id` is an identifier.

### Other analyses

Analyses outside the first look, by module.

| Module | Analysis | What it answers | Required arguments |
| --- | --- | --- | --- |

No analysis outside the first look ships yet.

## Discovery

Four functions describe the package from inside Python:

| Function | What it gives you |
| --- | --- |
| `ae.guide()` | Prints one page: how to call an analysis and read its result, the areas with their analysis counts, and the first-look order. |
| `ae.search_analyses(query)` | The analyses that best match a question in your own words, such as "nulls per column"; weak matches are marked. |
| `ae.describe_analysis(name)` | An analysis's parameters, statuses, result fields, finding codes and an example. |
| `ae.list_analyses()` | Every analysis with its module, question and masking support; `.to_frame()` gives a table. |

```python
ae.guide()
print(ae.search_analyses("nulls per column"))
print(ae.describe_analysis("profile_missing_values"))
```

`pd.read_csv` turns "N/A" and "NULL" into missing values; `keep_default_na=False` keeps them as text for `detect_missing_sentinels`.

## Reading a result

Every result has the same shared fields, next to its own:

- `res.status` says how the run ended: `completed`, `partial`, `insufficient_data`, `not_applicable` or `budget_exceeded`. `res.status_reason` says why not, with a `code` from a documented list.
- `res.findings` are claims about the data, each with a `code`, a `claim` and an `assessment`: `descriptive`, `suspected`, `violation`, `passed` or `unknown`.
- `res.limitations` say what limits how far to read the numbers, such as too few values.
- `res.parameters` records every argument used, defaults included.

```python
res = ae.summarize_numeric(df, columns=["age", "income"])
table = res.to_frame()  # every statistic of every column, as a DataFrame
record = res.to_json()  # the whole result as strict JSON
age = res.columns["age"]  # typed fields, such as age.median and age.std
```

The printed text stays within 40 lines of 120 characters however large the data, wraps long lines instead of cutting them, and says what it leaves out. Findings come violation first, and tables list columns with a finding first. Lists that grow with the data are capped in the JSON record too, with totals, but it keeps the entries that findings and limitations point at, so every path resolves. The record writes an integer beyond ±2^53 as a string, so Node-based MCP clients don't round it; the result object keeps the `int`.

Every result takes `to_frame(table=...)`. `res.columns["Age"]` raises a `KeyError` that suggests `'age'`; `describe_dataset`'s `columns` is a list, and takes a name too. pydantic's deprecated `json()`, `dict()`, `schema()`, `parse_obj()` and `copy()` raise an error naming `to_json()`, `to_dict()`, `record_schema()`, `model_validate()` or `model_copy()`. `help(type(res))` lists each field with its description.

`infer_column_semantics(...).inferred_schema()` is a `schema` for `validate_column_types`, and `detect_missing_sentinels(...).sentinel_values()` the `sentinel_values` for `profile_missing_values`, leaving masked candidates out with a `.note`.

A mistake in a call raises a subclass of `ae.errors.AgentEDAError` before any work starts, with a stable `code` and the fix; data problems never raise.

Equal values are decided as pandas does: `1`, `1.0` and `True` are one value. NaN in a pyarrow-backed or nullable float column is missing, with a limitation. Values pandas can't hash, such as lists, are the exception: `find_duplicate_rows` compares them by content, `find_candidate_keys`, `summarize_categorical` and `compute_association_matrix` leave the column out, and `describe_dataset` and `infer_column_semantics` leave its distinct count null, each with an `unhashable_values` note.

`find_candidate_keys` leaves out float columns with a fraction such as `0.5` unless `columns` names them, and lists them in `non_integral_float_columns`; whole-number floats stay candidates.

Arrow time-of-day, binary and nested columns, and sparse columns of anything but numbers or booleans, are "other": the defaults leave them out, and naming one raises `ColumnTypeError`.

`infer_column_semantics` and `validate_column_types` read text by the same rules:

- A column ranks boolean only when every value is a boolean, a 0 or 1, or a word such as `yes` or `f`, and at least two values differ: grades `A, B, F` and only 1s are not.
- Digits alone with a leading zero, such as `01234`, are a code. `05.50` is a number; `1e400`, beyond float64, is not.
- Dates that aren't ISO 8601, such as `01/02/2020`, are never parsed and never `identifier_like`, unless a `ColumnSpec` declares a `datetime_format`, whose `%b`-style directives read in the process's locale.
- `identifier_like` is judged without rows that exactly repeat another row, for columns distinct in at least 90% of rows.

Labels and row IDs read the same in every process. A set or frozenset lists its items sorted by their JSON text, also inside tuples; an object whose repr holds a memory address reads `<TypeName object>`. Identical calls therefore give byte-identical `to_json()`, text and `to_frame()` tables whatever `PYTHONHASHSEED` is.

A record writes a datetime as ISO 8601 text, and a decimal as a number when float64 holds it as written, else as exact text. `detect_missing_sentinels` gives number text such as `"-999"` the numeric-code rule, lists pandas' default NA strings, and matches text stand-ins after stripping whitespace, showing the raw form that occurred (`" N/A"`).

Concurrent calls from several threads are safe. An analysis captures its warnings under one lock, one at a time, so each result gets only its own `numerical_warning` limitation and the caller's warning filters stay unchanged.

## Projects and masking

Six analyses can show raw values, such as labels, row IDs or declared sentinel values: `validate_column_types`, `detect_missing_sentinels`, `profile_missing_values`, `summarize_categorical`, `find_duplicate_rows` and `profile_target`. They take `project=` and `mask_sensitive=`, and `ae.list_analyses()` marks them `[supports masking]`.

A `Project` holds settings that stay the same across calls. Its data policy names the sensitive columns, and masking replaces their values with numbered placeholders; counts and shares stay:

```python
policy = ae.DataPolicy(sensitive_columns=["email"], mask_sensitive=True)
project = ae.Project(name="churn", data_policy=policy)
print(ae.summarize_categorical(df, columns=["email"], project=project))
```

```text
summarize_categorical: completed on 8 rows x 6 columns
Masked (data_policy.mask_sensitive=True): the values of column 'email' show as <masked 1>, <masked 2> and so on; counts
    and shares are unaffected.
Shares are fractions of non-missing values. Levels go by count, then label; levels has the top 10 of each column.
Columns (1 of 1 shown):
column  non_missing_count  missing_count  distinct_count
email                   8              0               6
Top levels per column:
  email: <masked 1>: 2 (share 0.25), <masked 2>: 2 (share 0.25), <masked 3>: 1 (share 0.125), <masked 4>: 1 (share
    0.125), <masked 5>: 1 (share 0.125), <masked 6>: 1 (share 0.125)
Findings: none
Limitations: none
More: res.to_frame() has all 6 levels; res.findings, res.limitations and res.to_json() have the rest.
```

How masking is decided:

- It is off by default, even for columns declared sensitive.
- `mask_sensitive=True` in the data policy switches it on for every call that gets the project.
- `mask_sensitive=True` or `False` in a call overrides the project.
- It covers the printed text, `to_dict()`, `to_json()` and `to_frame()` alike.
- It never touches numeric statistics, such as a minimum, or index labels. Pass a non-sensitive `id_column` if the index holds sensitive values. The other analyses, such as `summarize_numeric`, don't read a project.
- A sensitive column matches by exact name. A near miss in case or Unicode form, such as `'Email'` for `'email'`, masks nothing; the printed masking text and `masking.near_miss_columns` then say "did you mean 'email'?".
- With masking on, no error message quotes a value of a sensitive column: a list of unknown column names holds at most 3, cut to 40 characters, and a Series, Index or array given as a `column` or `columns` option is refused unseen.

Save the project once and load it in later scripts, which helps an agent that runs each script in a new process:

```python
project.save("project.toml")
project = ae.Project.load("project.toml")
```

`project.toml` is plain TOML that you can edit by hand:

```toml
# An agent-eda Project. Load it with ae.Project.load(path).
format_version = 1
name = "churn"

[data_policy]
sensitive_columns = [
    "email",
]
mask_sensitive = true
```

`format_version` is the file's layout: a file without it loads as version 1. `project.save` writes owner-readable only for a new file, and refuses a target that isn't a regular file, such as a named pipe, and a project over 1 MiB.

Masking is a guardrail against accidental exposure, not anonymization or access control. The package runs in the agent's own process, so an agent that can call it can read the data directly. [ADR 0001](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0001-data-policy-is-a-guardrail.md) records why.

## Use over MCP

Agents that call MCP tools instead of writing Python, such as Claude Desktop, Cursor or Claude Code without a shell, reach the analyses through a local MCP server. It speaks stdio only: the MCP client starts `agent-eda-mcp` as a subprocess, and the server opens no network connection.

### Install and register

The MCP server needs the optional `mcp` extra, adding the MCP Python SDK and pyarrow:

```sh
pip install 'agent-eda[mcp]'  # or, in a uv project: uv add 'agent-eda[mcp]'
```

Until 0.1 is on PyPI, install the wheel with it: `pip install 'dist/agent_eda-0.1.0-py3-none-any.whl[mcp]'`. Without the extra, `agent-eda-mcp` prints the install command on stderr and exits.

Register it with Claude Code, without or with a project:

```sh
claude mcp add agent-eda -- agent-eda-mcp
claude mcp add agent-eda -- agent-eda-mcp --project /path/to/project.toml
```

Claude Desktop, Cursor and most MCP clients read an `mcpServers` entry from their JSON configuration. Leave out `args` for no project:

```json
{
  "mcpServers": {
    "agent-eda": {
      "command": "agent-eda-mcp",
      "args": ["--project", "/path/to/project.toml"]
    }
  }
}
```

If the client doesn't start it in the environment with the extra, use the path `which agent-eda-mcp` prints as `command`.

Each analysis is an MCP tool of the same name. Its description is the docstring's `When to use:`, `Returns:` and `Statuses:` sections and the names under `See Also:`, and its input schema holds the `Args:`. Two more MCP tools find analyses: `search_analyses` takes a question in your own words, `list_analyses` lists them all. Its instructions tell a connecting agent how to pass data, read a response and follow the first-look order, and hold the rules of the path arguments.

### Pass data by path

An MCP tool takes `data_path`, a file path, where Python passes the DataFrame.

- The extension sets the format, in any case: `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz`, `.parquet` or `.pq`. Any other extension gives the error `unsupported_data_format`.
- CSV and TSV files are read as UTF-8, and only an empty field counts as missing. Values such as `N/A`, `NULL` or `-999` reach the analyses as values, for `detect_missing_sentinels` to find, `-999` too in a column that `N/A` made text.
- Each CSV or TSV column is typed once, over the whole file, and no column becomes the index:
  - A column is read as numbers or booleans (`true` or `false`, in any case) only when every non-empty field parses as one, else it stays text. `inf` and `-inf` are numbers.
  - A whole number written with a leading zero, such as the postal code `01234` or the ID `007`, is a code, so its column stays text and `find_candidate_keys` tells `"007"` and `"7"` apart. `-5`, `0`, `0.5`, `10` and a decimal such as `05.50` stay numbers.
  - Blank lines are skipped, except in a one-column file, where each is a row with a missing value. A quoted empty field (`""`) is an empty string, not a missing value, and its column stays text.
  - The server reads commas, or tabs for a TSV: a file with one column whose name holds a `;` gives `unreadable_data_file`, hinting at a semicolon separator. A row may end in an empty field, as in `A1,Oslo,10,`; a row with any other extra field gives `unreadable_data_file` and names the line.
  - A repeated header name stays repeated: `describe_dataset` reports it, and every other analysis asks you to rename the columns in the file.
- Parquet files keep the dtypes and index pandas stored; one without pandas metadata, such as from Spark or DuckDB, gets pandas' default dtypes and a 0-based index. A file with two columns of one name or bad pandas metadata gives `unreadable_data_file`.
- A relative path resolves against the server's working directory; pass absolute paths if unsure. A leading `~` is the home directory, in every path argument.
- The server reads only local files: it reads a URL such as `s3://bucket/data.csv` as a relative path. A path that leads nowhere gives `data_file_not_found`, with the path tried and the closest names.
- Row IDs in a result are 0-based data-row numbers for a CSV or TSV file and the stored index for a Parquet file; `id_column` names rows by a column's values instead.
- The MCP server keeps the last two files it read in memory, and rereads one when its modification time or size changes.

### Pass arguments as JSON

A string whose input schema format is `date-time`, `date` or `duration` is ISO 8601 text. `sentinel_values` takes datetimes, dates and durations that way, such as `{"sentinel_values": {"born": ["1900-01-01T00:00:00"], "wait": ["PT0S"]}}`, for a column that holds such values and no text; a string for any other column stays text. A time zone must match the column: `"1900-01-01T00:00:00Z"` for a column without one gives `invalid_argument` with the value to pass.

### Read the response

A call returns the printed result as text and the whole `to_dict()` record as structured content, with the same caps on long lists as in Python. The text is worded for MCP callers: where `print(res)` points to `res.to_frame()` and `res.to_json()`, it points to `table_path` and the structured content. A run that doesn't complete, such as `insufficient_data`, is a normal response. A mistake in the call comes back as an MCP tool error (`isError: true`) whose structured content holds the stable error `code` and `message`. The message says what is wrong, what is allowed and the fix in MCP terms, such as `{"task": "regression"}` for `task='regression'`. No message, MCP tool description, schema or record text holds Python syntax, and tests scan them all.

To get past the caps, pass `table_path`. The server writes the result's main table, what `res.to_frame()` gives, to a new `.csv`, `.parquet` or `.pq` file. The text ends with `Full table written to <absolute path> (<n> rows).`, and the structured content gains `table_file` with the same `path` and `row_count`.

- `table_path` never overwrites. An existing file gives the error `table_file_exists` before the analysis runs, and its message suggests a free name such as `table-2.csv`. A symbolic link at the path gives the same error. The directory must exist already.
- If the server can tell it may not write in the directory, the call gives `invalid_argument` before the analysis runs. A failed write removes its partial file.
- In a CSV, a missing value is an empty field and an empty string is the quoted field `""`. Parquet keeps them apart.
- The last line of the text keeps to 120 characters: a path that doesn't fit takes its own line, or is left to `table_file.path`.
- A cell that holds a list or a mapping, such as the columns of a candidate key from `find_candidate_keys`, is written as JSON text. In Parquet, a column of mixed types, such as numbers and text, is written as text, as in the CSV.
- An analysis with several tables writes only its main one. `compute_association_matrix` writes its pairs, for example, and its wide matrix stays a Python-only table.

### Masking over MCP

The six MCP tools that support masking take `project_path`, the path to a project file as `Project.save` writes it, in place of `project=`, and keep `mask_sensitive`. `agent-eda-mcp --project PATH` loads one project file at start and uses it for every call. A call's `project_path` replaces it for that call, and a call's `mask_sensitive` wins over both. Masking then covers the text, the structured content and the file `table_path` writes. As in Python, it never touches numeric statistics or index labels, and the MCP tools without masking don't read the project. With masking on, an internal error (a bug) names only the exception's type, as its message could quote a value; the stderr log keeps the traceback.

## Reference

<!-- reference:start -->

Generated from the docstrings by `scripts/generate_reference.py`: edit a docstring and rerun it, not this section. Every public name in `agent_eda`, with the first sentence of each argument and field. `help(ae.<name>)` shows the whole docstring and `ae.describe_analysis("<name>")` gives an analysis as data.

### Analysis functions

Every analysis takes the DataFrame first, as `data`, and never changes it. All other arguments are keyword-only. Each returns its own result class, which has the fields under [Results](#results) next to the ones listed here. A finding's assessment is in brackets: descriptive: a neutral fact. suspected: a possible problem to judge. violation: the data breaks something declared. passed: a declared check held. unknown: a declared check couldn't be evaluated.

#### `describe_dataset`

`ae.describe_dataset(data, *, source=None, source_version=None) -> DescribeDatasetResult`

What does this dataset contain?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `source` | `str \| None` | `None` | Where the data came from, such as a file path, URL or table name, recorded exactly as given. |
| `source_version` | `str \| None` | `None` | The version of the source, such as a snapshot date or a commit, recorded exactly as given. |

**Fields:** `row_count`, `column_count`, `column_level_count`, `memory_bytes`, `columns` (`position`, `column`, `label_type`, `dtype`, `non_missing_count`, `missing_count`, `distinct_count`, `memory_bytes`), `index` (`dtype`, `level_count`, `is_unique`, `duplicated_count`, `missing_count`, `memory_bytes`), `provenance` (`source`, `source_version`, `not_supplied`)

- `empty_data` (suspected): the data has no rows, no columns or neither.
- `duplicated_column_labels` (suspected): a column label occurs more than once.
- `colliding_column_labels` (suspected): different labels share a string form, such as 1 and '1'.
- `multiindex_columns` (suspected): the column labels form a MultiIndex.
- `non_unique_index` (suspected): index labels repeat, so they don't identify rows.
- `all_missing_columns` (suspected): a column holds only missing values.
- `constant_columns` (suspected): a column holds a single distinct non-missing value in at least two rows; NaN is missing.

**Statuses:** `completed`

#### `infer_column_semantics`

`ae.infer_column_semantics(data, *, columns=None, schema=None) -> InferColumnSemanticsResult`

What kind of values does each column hold?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to read, in the order to report them: a list or a single name. |
| `schema` | `Mapping[str, str \| ColumnSpec] \| None` | `None` | The logical types you declare, by column name: 'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or 'timedelta', or a ColumnSpec (a mapping of its fields works too), whose logical_type is the one compared. |

**Fields:** `columns` (`column`, `status`, `status_reason`, `dtype`, `inferred_logical_type`, `declared_logical_type`, `matches_declared`, `candidates`, `flags`, `evidence`), `missing_declared_columns`

- `identifier_like_columns` (suspected): every value is distinct, there are at least two, and the top candidate is integer or text without spaces.
- `constant_columns` (suspected): a single distinct value in at least two rows; NaN is missing.
- `all_missing_columns` (suspected): no value in at least one row.
- `numbers_as_text` (suspected): the top candidate is integer or float and some of the numbers are stored as text.
- `dates_as_text` (suspected): the top candidate is datetime and some of the datetimes are text, in ISO 8601 or the declared datetime_format.
- `non_iso_date_text` (suspected): at least half the values look like dates, and some of them aren't ISO 8601, such as 01/02/2020; they are never parsed.
- `boolean_like_columns` (descriptive): the top candidate is boolean and some values are numbers or text that read as booleans, such as 0/1 or yes/no.
- `mixed_value_types` (suspected): the values have more than one Python type, such as int and str, as `evidence.value_types` lists.
- `declared_type_mismatch` (suspected): a declared column's declared logical type scores below its top candidate, or isn't a candidate.
- `declared_type_match` (passed): a declared column's top candidate, or a candidate tied with it, is its declared logical type.
- `declared_type_unchecked` (unknown): a declared column has no candidate to compare, since it has no values or none fits.
- `missing_declared_columns` (violation): schema names columns the data doesn't have, listed in `missing_declared_columns`.

**Statuses:** `completed`, `partial` (`incomplete_columns`), `insufficient_data` (`no_rows`, `no_values`), `not_applicable` (`no_columns`, `unsupported_values`)

#### `validate_column_types`

`ae.validate_column_types(data, *, schema, id_column=None, project=None, mask_sensitive=None) -> ValidateColumnTypesResult`

Do the columns hold the logical types I declared?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema` | `Mapping[str, str \| ColumnSpec]` | required | The logical type each column should hold, by column name, in the order to report them: 'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or 'timedelta', or a ColumnSpec (a mapping of its fields works too). |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces the invalid values of sensitive columns, and row IDs taken from a sensitive id_column, with numbered placeholders, '<masked 1>' and so on, equal values alike, and needs a project that declares sensitive columns; False shows them as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `columns` (`column`, `declared_logical_type`, `nullable`, `datetime_format`, `present`, `dtype`, `compatible_count`, `convertible_count`, `invalid_count`, `missing_count`, `invalid_row_ids`, `invalid_distinct_count`, `invalid_values`, `assessment`), `undeclared_columns`

- `invalid_values` (violation): columns that hold invalid values.
- `missing_declared_columns` (violation): declared columns the data doesn't have, each with the closest names.
- `non_nullable_missing_values` (violation): columns declared non-nullable that have missing values.
- `unchecked_columns` (unknown): columns without a value to check, as when every value is missing.
- `convertible_values` (descriptive): columns with values stored as another type; convert them before you use them as the declared type.
- `passing_columns` (passed): columns that pass every check: only compatible and convertible values, and none missing when declared non-nullable.

**Statuses:** `completed`, `insufficient_data` (`no_rows`)

#### `profile_missing_values`

`ae.profile_missing_values(data, *, columns=None, sentinel_values=None, project=None, mask_sensitive=None) -> ProfileMissingValuesResult`

How much data is missing, and where?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to profile, in the order to report them: a list or a single name. |
| `sentinel_values` | `Mapping[str, Sequence[bool \| int \| float \| str \| datetime \| date \| timedelta]] \| None` | `None` | The sentinel values to count as missing, by column name, such as {'age': [-999], 'city': ['N/A', '']}: one value or a list of values per column, each a bool, int, float, str, datetime (pandas Timestamp too), date or timedelta (pandas Timedelta too). |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each declared sentinel value of a sensitive column, in parameters and in the columns table, with a numbered placeholder, '<masked 1>' for the first value declared and so on, and needs a project that declares sensitive columns; False shows the values as declared. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `columns` (`column`, `status`, `status_reason`, `native_missing_count`, `sentinel_count`, `missing_count`, `missing_share`, `sentinel_values`), `missing_count`, `missing_share`, `complete_row_count`, `complete_row_share`, `row_distribution` (`missing_count`, `row_count`, `row_share`)

- `all_missing_columns` (suspected): every value of a column is missing or a declared sentinel value.
- `sentinel_values_present` (descriptive): declared sentinel values occur in a column, where they count as missing.

**Statuses:** `completed`, `insufficient_data` (`no_rows`), `not_applicable` (`no_columns`)

#### `detect_missing_sentinels`

`ae.detect_missing_sentinels(data, *, columns=None, sentinel_values=None, id_column=None, project=None, mask_sensitive=None) -> DetectMissingSentinelsResult`

Which values look like stand-ins for missing data?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to check, in the order to report them: a list or a single name. |
| `sentinel_values` | `Mapping[str, Sequence[bool \| int \| float \| str \| datetime \| date \| timedelta]] \| None` | `None` | The sentinel values you declare, by column name, such as {'age': [-999], 'city': ['N/A', '']}, as profile_missing_values takes them: one value or a list of values per column, each a bool, int, float, str, datetime (pandas Timestamp too), date or timedelta (pandas Timedelta too). |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values, and the default columns leave it out of the check. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each value of a sensitive column that the result shows, its candidate values, its declared values in parameters and its values as row IDs when it is id_column, with a numbered placeholder, '<masked 1>' for the first value in result order and so on, equal values alike, and needs a project that declares sensitive columns; False shows the values as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `candidates` (`column`, `value`, `value_type`, `count`, `share`, `evidence`, `confirmation`, `row_ids`), `columns_without_rules`

- `sentinel_values_present` (descriptive): declared sentinel values occur in a column.
- `suspected_sentinel_values` (suspected): the built-in rules found values that may stand in for missing ones, which stay valid values until you declare them.

**Statuses:** `completed`, `insufficient_data` (`no_rows`), `not_applicable` (`no_columns`)

#### `summarize_numeric`

`ae.summarize_numeric(data, *, columns=None, quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), quantile_method='linear', trim_fraction=0.1) -> SummarizeNumericResult`

How are the values in each numeric column distributed?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to summarize, in the order to report them: a list or a single name. |
| `quantiles` | `Sequence[float]` | `(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)` | Quantile levels, each from 0 to 1 and listed once, in the order to report them; an empty list computes none. |
| `quantile_method` | `str` | `'linear'` | How numpy.quantile picks or interpolates between values: Hyndman and Fan's definitions 1 to 9 as 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased' and 'normal_unbiased', or 'lower', 'higher', 'midpoint' and 'nearest'. |
| `trim_fraction` | `float` | `0.1` | The share of the finite values cut from each end for the trimmed mean: trim_fraction times n, rounded down, values each, as SciPy's trim_mean cuts. |

**Fields:** `columns` (`column`, `status`, `status_reason`, `non_missing_count`, `missing_count`, `infinite_count`, `zero_count`, `negative_count`, `mean`, `std`, `min`, `max`, `quantiles`, `median`, `trimmed_mean`, `mad`, `mad_scaled`, `skewness`, `excess_kurtosis`), `left_out_columns`

- `constant_columns` (suspected): a column holds a single distinct non-missing value in at least two rows, on stored values; NaN is missing.
- `infinite_values` (suspected): a column holds inf or -inf, which every statistic leaves out.

**Statuses:** `completed`, `partial` (`incomplete_columns`), `insufficient_data` (`no_rows`, `no_values`), `not_applicable` (`no_columns`)

#### `summarize_categorical`

`ae.summarize_categorical(data, *, columns=None, top_n=10, project=None, mask_sensitive=None) -> SummarizeCategoricalResult`

Which values occur in each categorical column, and how often?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to summarize, in the order to report them: a list or a single name. |
| `top_n` | `int` | `10` | How many of each column's most common levels `levels` holds and the printed text shows, from 1 to 100; defaults to 10. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each level of a sensitive column with a numbered placeholder, '<masked 1>' for the most common and so on in level order, and needs a project that declares sensitive columns; False shows the levels as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `columns` (`column`, `status`, `status_reason`, `non_missing_count`, `missing_count`, `distinct_count`, `levels`, `left_out_level_count`, `seen_once_count`, `unused_categories`, `value_types`, `mixed_value_types`, `case_or_space_variants`), `left_out_columns`

- `case_or_space_variants` (suspected): a column holds labels that differ only by case or surrounding spaces, such as 'Yes' and 'yes '.
- `mixed_value_types` (suspected): a column holds values of more than one Python type, such as 1 and '1'.
- `unused_categories` (descriptive): a categorical column has categories that no value takes.

**Statuses:** `completed`, `partial` (`incomplete_columns`), `insufficient_data` (`no_rows`, `no_values`), `not_applicable` (`no_columns`, `unhashable_values`)

#### `find_duplicate_rows`

`ae.find_duplicate_rows(data, *, columns=None, id_column=None, project=None, mask_sensitive=None) -> FindDuplicateRowsResult`

Are there exact duplicate rows?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns in which two rows must be equal to count as duplicates, in the order to compare them: a list or a single name. |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values, and the default columns leave it out of the comparison. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each row ID taken from a sensitive id_column with a numbered placeholder, '<masked 1>' for the first ID the result shows and so on, equal IDs alike, and needs a project that declares sensitive columns; False shows the IDs as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `compared_columns`, `duplicate_groups` (`size`, `row_ids`), `duplicate_group_count`, `duplicate_group_row_count`, `excess_copy_count`, `duplicate_share`

- `duplicate_rows` (suspected): two or more rows are equal in every compared column.

**Statuses:** `completed`, `insufficient_data` (`no_rows`), `not_applicable` (`no_columns`)

#### `find_candidate_keys`

`ae.find_candidate_keys(data, *, columns=None, max_key_size=2, max_combinations=10000) -> FindCandidateKeysResult`

Which columns, alone or combined, uniquely identify rows?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to search, in the order to report them: a list, a tuple, a pandas Index or a single name. |
| `max_key_size` | `int` | `2` | The most columns a key may combine, at least 1; defaults to 2. |
| `max_combinations` | `int` | `10000` | The most column sets the search examines, at least 1; defaults to 10,000, which covers every set of up to 2 of 140 columns. |

**Fields:** `keys` (`columns`, `distinct_row_count`, `duplicate_group_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `keys_with_missing_values` (`columns`, `distinct_row_count`, `duplicate_group_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `most_unique_column_sets` (`columns`, `distinct_row_count`, `duplicate_group_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `exhaustive`, `search` (`column_count`, `combination_count`, `examined_count`, `rejected_on_subset_count`, `checked_on_all_rows_count`, `subset_row_count`), `left_out_columns`, `non_integral_float_columns`

- `keys_found` (descriptive): column sets identify every row and have no missing values.
- `keys_with_missing_values` (descriptive): column sets identify every row but have missing values, so they can't serve as primary keys.
- `no_key_found` (descriptive): no examined set identifies every row; the claim names the most unique set and its duplicate rows.

**Statuses:** `completed`, `budget_exceeded` (`too_many_combinations`), `insufficient_data` (`no_rows`, `one_row`), `not_applicable` (`no_columns`, `unhashable_values`)

#### `profile_target`

`ae.profile_target(data, *, target_column, task, expected_labels=None, rare_share=None, id_column=None, project=None, mask_sensitive=None) -> ProfileTargetResult`

What does the target look like for this task?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target_column` | `str` | required | Name of the column a model learns to predict. |
| `task` | `str` | required | What the model does with the target, never inferred from the dtype. |
| `expected_labels` | `Sequence[str \| bool \| int \| float \| datetime \| date \| pd.Period] \| None` | `None` | The labels the task allows, such as ['no', 'yes'] or [0, 1]: exactly 2 for binary_classification and at least 2 for multiclass_classification, each listed once; regression takes none. |
| `rare_share` | `float \| None` | `None` | For classification, the share of the valid labels below which a class counts as rare, such as 0.05; greater than 0 and less than 1. |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as a customer number; the row IDs of labels outside expected_labels are then its values. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces the labels of a sensitive target, the classes, the labels outside expected_labels and expected_labels as parameters records it, with numbered placeholders, '<masked 1>' for its first label in level order and so on, and the row IDs taken from a sensitive id_column too; it needs a project that declares sensitive columns. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`, `near_miss_columns`), `non_missing_count`, `missing_count`, `missing_share`, `valid_label_count`, `class_count`, `classes` (`label`, `label_type`, `count`, `share`), `majority_share`, `minority_share`, `imbalance_ratio`, `unexpected_label_count`, `unexpected_labels` (`label`, `label_type`, `count`, `row_ids`), `rare_class_count`, `rare_classes` (`label`, `label_type`, `count`, `share`), `statistics` (`column`, `status`, `status_reason`, `non_missing_count`, `missing_count`, `infinite_count`, `zero_count`, `negative_count`, `mean`, `std`, `min`, `max`, `quantiles`, `median`, `trimmed_mean`, `mad`, `mad_scaled`, `skewness`, `excess_kurtosis`), `non_finite_count`

- `single_class` (violation): every valid label is the same class, so a classifier has nothing to tell apart.
- `more_than_two_classes` (violation): a binary_classification target has more than two classes.
- `unexpected_labels` (violation): labels lie outside expected_labels.
- `no_unexpected_labels` (passed): every label that isn't missing is in expected_labels.
- `all_labels_distinct` (suspected): the values of a classification target that aren't missing are all distinct, as an identifier's or a measurement's are, not class labels'.
- `constant_target` (violation): at least two values of a regression target aren't missing and all of them are equal; NaN is missing.
- `infinite_values` (suspected): a regression target holds inf or -inf, which every statistic leaves out.
- `missing_labels` (descriptive): rows have no target value.
- `rare_classes` (descriptive): classes have a share below rare_share.
- `missing_expected_classes` (violation): declared classes have no rows.
- `case_or_space_variants` (suspected): labels differ only by case or surrounding spaces.
- `two_value_target` (suspected): a regression target has two distinct values, so it may be binary labels.

**Statuses:** `completed`, `insufficient_data` (`no_rows`, `no_values`)

#### `compute_association_matrix`

`ae.compute_association_matrix(data, *, columns=None, categorical_columns=None, numeric_method='pearson', categorical_method='cramers_v_bias_corrected', mixed_method='correlation_ratio_bias_corrected', max_categories=50, max_pairs=5000) -> ComputeAssociationMatrixResult`

How strongly are pairs of columns associated?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to pair, in the order to report them: a list or a single name. |
| `categorical_columns` | `Sequence[str] \| None` | `None` | Numeric columns to treat as categorical, so that codes such as 1, 2 and 3 are compared as categories instead of correlated as numbers. |
| `numeric_method` | `str` | `'pearson'` | The measure for two numeric columns: 'pearson', the default, Pearson's r for linear association; 'spearman', Pearson's r of the ranks, tied values sharing their mean rank, for monotonic association; or 'kendall', Kendall's tau-b, which adjusts for ties. |
| `categorical_method` | `str` | `'cramers_v_bias_corrected'` | The measure for two categorical columns: 'cramers_v_bias_corrected', the default, Cramér's V with Bergsma's (2013) correction, which removes most of the upward bias that few rows and many levels give V; or 'cramers_v', the uncorrected V of scipy.stats.contingency.association. |
| `mixed_method` | `str` | `'correlation_ratio_bias_corrected'` | The measure for a categorical and a numeric column, 'correlation_ratio_bias_corrected' or 'correlation_ratio'. |
| `max_categories` | `int` | `50` | The most distinct values a categorical column may have; a column with more is left out with reason too_many_categories. |
| `max_pairs` | `int` | `5000` | The most pairs the call computes, a work limit. |

**Fields:** `pairs` (`x_column`, `y_column`, `x_type`, `y_type`, `method`, `value`, `pair_count`, `direction`, `undefined_reason`), `columns` (`column`, `dtype`, `type`, `value_count`, `missing_count`, `infinite_count`, `distinct_count`), `left_out_columns` (`column`, `dtype`, `reason`, `message`)

- `perfect_associations` (suspected): a pair has an absolute value of 1, within 1e-9, so one column may duplicate or derive from the other.

**Statuses:** `completed`, `budget_exceeded` (`too_many_pairs`), `insufficient_data` (`no_rows`, `no_values`), `not_applicable` (`no_columns`)

### Discovery functions

#### `guide`

`ae.guide() -> None`

Print one page on agent-eda: what it does, how to call and read analyses.

**Returns:** The page is printed, since Code Interpreter returns only what is printed.

#### `search_analyses`

`ae.search_analyses(query, limit=5) -> SearchListing`

Find the analyses that best match a question in your own words.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `query` | `str` | required | What you want to know, such as 'nulls per column'. |
| `limit` | `int` | `5` | The most matches to return, at least 1; defaults to 5. |

**Returns:** `matches` holds the best matches, best first, each with its `name`, `summary`, `score`, `matched_terms` and `weak`; read the summary before you rely on a weak match.

#### `describe_analysis`

`ae.describe_analysis(name) -> AnalysisDescription`

Describe one analysis: everything needed for a correct first call.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | required | The analysis name, such as 'summarize_numeric', or its fully qualified name, such as 'agent_eda.statistics.descriptive.summarize_numeric'. |

**Returns:** `parameters` lists each parameter with `type`, `required`, `default`, `allowed_values`, `valid_range` and `description`.

#### `list_analyses`

`ae.list_analyses() -> AnalysisListing`

List every analysis with its module, one-line summary and masking support.

**Returns:** The analyses, ordered by module, then name.

### Projects and schemas

#### `Project`

`ae.Project(*, name=None, data_policy=None)`

A data science project's declarations, given to the analyses that use them.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str \| None` | `None` | The project's name; null, the default, leaves it unnamed. |
| `data_policy` | `DataPolicy \| Mapping[str, Any] \| None` | `None` | Which columns are sensitive and whether to mask them. |

#### `DataPolicy`

`ae.DataPolicy(*, sensitive_columns=(), mask_sensitive=False)`

Which columns hold sensitive values, and whether results mask them.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sensitive_columns` | `Sequence[str] \| None` | `()` | Names of the columns that hold sensitive values; empty, the default, declares none. |
| `mask_sensitive` | `bool` | `False` | Whether analyses mask the values of the sensitive columns when the call's mask_sensitive is None. |

#### `ColumnSpec`

`ae.ColumnSpec(logical_type, *, nullable=None, datetime_format=None)`

What a column is declared to hold: its logical type and more.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `logical_type` | `LogicalType` | required | What the values mean: 'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or 'timedelta'. |
| `nullable` | `bool \| None` | `None` | Whether the column may hold missing values; None, the default, declares nothing. |
| `datetime_format` | `str \| None` | `None` | For logical_type 'datetime' only: the datetime.strptime format of datetimes stored as text, such as '%d/%m/%Y'. |

- `project.save(path, *, overwrite=False) -> Path`: Save the project to a TOML file that Project.load() reads.
- `ae.Project.load(path) -> Project`: Load a project from a TOML file, such as one that project.save() wrote.

#### `LogicalType`

The logical type names as a `Literal`: `'boolean'`, `'integer'`, `'float'`, `'categorical'`, `'text'`, `'datetime'`, `'timedelta'`.

### Results

#### `AnalysisResult`

The record an analysis returns. Every analysis returns its own subclass, named after it, with its typed fields next to these shared ones. `print(res)` gives plain text from a fixed template. The shared fields:

| Field | What it holds |
| --- | --- |
| `analysis` | Name of the analysis, such as 'describe_dataset'. |
| `method` | The method used, or null when the analysis has no methods. |
| `status` | How the run ended. |
| `status_reason` | Why the run didn't complete; null when status is completed. |
| `parameters` | The resolved arguments by name, defaults included. data appears only as its row_count and column_count, and project whole. |
| `support` | The counts behind the result. |
| `findings` | Claims about the data. |
| `limitations` | What limits how far the result can be read. |
| `execution` | Library versions used for the run. |

The shared methods:

- `res.to_frame()`: The complete main table of the result as a DataFrame.
- `res.to_dict()`: The result record as JSON-compatible Python objects.
- `res.to_json()`: The record from to_dict() as strict JSON, without NaN or Infinity.
- `ResultClass.record_schema()`: The JSON Schema of the record that to_dict() and to_json() give.

Each limitation's `code` is one of these:

- `unhashable_values`: Values can't be hashed, so counts of distinct values are null.
- `non_integral_float_columns`: A key search left out float columns that hold fractions.
- `too_few_values`: A statistic needs more values than the column has, so it is null.
- `zero_variance`: Every value is equal, so a statistic such as skewness is null.
- `float_precision`: Numbers beyond ±2^53 were read as float64, which rounds them.
- `boolean_as_numeric`: Booleans were read as numbers: 0 for False and 1 for True.
- `nan_counted_as_missing`: A column stores NaN as a value, which every analysis counts as missing, unlike isna().
- `one_row_per_level`: A categorical column has one level per row of a pair, so its value is null.
- `numerical_warning`: A computation overflowed or failed, so the values it affects are null.
- `ambiguous_row_ids`: Index labels or id_column values repeat, so a row ID can match several rows.
- `sentinel_values_not_declared`: No sentinel values were declared, so counts are null or candidates stay suspected.
- `ranking_not_proven`: A ranking stopped at its work limit before it was proven.
- `expected_labels_not_declared`: No expected labels were declared, so labels outside them weren't checked.
- `rare_share_not_declared`: No rare_share was declared, so rare classes weren't checked.
- `column_not_in_data`: A declared column isn't in the data, so its dtype and counts are null.
- `schema_not_declared`: No schema was declared, so no column was compared with a declared type.
- `nullable_not_declared`: nullable wasn't declared for a column, so its missing values weren't checked and its nullable field is null.
- `type_not_inferred`: A column has no inferred logical type, so the field is null.
- `not_applicable_to_task`: A field belongs to the other task, so it is null.

### Errors

Exceptions that agent-eda raises when a call is wrong.

| Code | Exception | Raised when |
| --- | --- | --- |
| `invalid_argument` | `InvalidArgumentError` | An argument has the wrong type or value, such as a number outside its range, an unknown choice, a list that repeats a value, or a declared sentinel value that can never match its column. |
| `invalid_call` | `InvalidArgumentError` | The call itself is malformed, such as an option passed by position, an unknown keyword or a missing DataFrame. |
| `unsupported_data_type` | `InvalidArgumentError` | Data isn't a pandas DataFrame, such as a file path, a pyarrow Table or None. |
| `ambiguous_column_labels` | `InvalidArgumentError` | The data's column labels repeat, share a string form or form a MultiIndex, so a name can't pick out one column. |
| `column_not_found` | `ColumnNotFoundError` | A column name isn't in the data. |
| `unsupported_column_type` | `ColumnTypeError` | A named column has a dtype the analysis can't use, or an id_column holds values that can't be hashed, such as lists. |
| `unknown_analysis` | `UnknownAnalysisError` | describe_analysis got a name that isn't an analysis. |
| `invalid_project_file` | `ProjectFileError` | Project.load got a file that isn't UTF-8 TOML (the message gives the line at fault), lacks the [data_policy] table or one of its keys, holds a value a Project can't take, isn't a regular file, holds more than 1 MiB, can't be read or has a newer format_version than this agent-eda reads. |
| `unknown_project_key` | `ProjectFileError` | Project.load got a file with a key that is neither a field of the Project nor one of its data policy, such as a misspelling. |
| `project_file_not_found` | `ProjectFileNotFoundError` | Project.load found no file at the path. |
| `result_table_unavailable` | `ResultTableUnavailableError` | to_frame() was called on a result rebuilt from its record by model_validate(), which doesn't hold the tables that only a call of the analysis keeps. |
| `project_file_exists` | `ProjectFileExistsError` | project.save found a file at the path and overwrite was False. |
| `missing_dependency` | `MissingDependencyError` | A feature needs an optional extra that isn't installed, such as the MCP server without the mcp extra. |
| `unsupported_data_format` | `UnsupportedDataFormatError` | An MCP tool got a data path whose extension it can't read. |
| `data_file_not_found` | `DataFileNotFoundError` | An MCP tool found no file at its data path. |
| `unreadable_data_file` | `DataFileError` | An MCP tool's data path names a file the parser can't read, such as a CSV that isn't UTF-8. |
| `table_file_exists` | `TableFileExistsError` | An MCP tool's table_path leads to a file that already exists, which the MCP server never overwrites. |

### MCP server command

`agent-eda-mcp [--project PATH]`

Serve the agent-eda analyses as MCP tools over stdio. `--project PATH`: A project TOML file, as ae.Project.save writes it, whose data policy names the sensitive columns and says whether to mask them. [Use over MCP](#use-over-mcp) has the details.

<!-- reference:end -->

## Documentation

- [CONTRIBUTING.md](https://github.com/chrislemke/agent-eda/blob/main/CONTRIBUTING.md): how to set up, test and change the package.
- [AGENTS.md](https://github.com/chrislemke/agent-eda/blob/main/AGENTS.md): the rules for code here, and the commands.
- [GLOSSARY.md](https://github.com/chrislemke/agent-eda/blob/main/GLOSSARY.md): the terms used in code, docstrings and messages.
- [CHANGELOG.md](https://github.com/chrislemke/agent-eda/blob/main/CHANGELOG.md): what changed before 0.1.0.
- [Planned analyses](https://github.com/chrislemke/agent-eda/blob/main/docs/planned-analyses.md): the backlog, with proposed names and modules.
- Architecture decisions: [ADR 0001](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0001-data-policy-is-a-guardrail.md), [ADR 0002](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0002-pandas-in-numpy-inside.md), [ADR 0003](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0003-no-llm-calls-in-the-package.md), [ADR 0004](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0004-plain-functions-with-explicit-arguments.md), [ADR 0005](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0005-pure-python-within-code-interpreter-dependencies.md), [ADR 0006](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0006-local-mcp-server-over-stdio-data-by-path.md).

## License

MIT. See [LICENSE](https://github.com/chrislemke/agent-eda/blob/main/LICENSE).
