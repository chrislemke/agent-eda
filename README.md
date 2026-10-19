# agent-eda

Deterministic exploratory data analysis (EDA) for pandas DataFrames, built for LLM agents and usable by people.

Each analysis is a plain Python function. It takes a DataFrame and typed keyword arguments and returns a typed result that prints as a short text summary. The library never calls an LLM or the network, so the same data and arguments always give the same result, in the same environment, BLAS thread count included.

It is written for three kinds of callers:

- agents in AWS Bedrock AgentCore, through its Code Interpreter
- coding agents such as Claude Code
- people, who use it like any other package

## Features

- **11 first-look analyses**: structure, column types, missing values, sentinel values such as -999, distributions, duplicates, keys, the target and associations.
- **Results you can read and parse**: `print(res)` gives at most 40 lines of text, `res.to_frame()` the full tables and `res.to_json()` the whole record as strict JSON.
- **Findings, not just numbers**: each result lists what it noticed, such as "1 value in 1 column may stand in for missing values: 'age'".
- **Discovery for agents**: `ae.guide()`, `ae.search_analyses()` and `ae.describe_analysis()` explain the package from inside Python.
- **Helpful errors**: a misspelled column name raises an error that lists the available columns and the closest match.
- **Opt-in masking** of sensitive columns, such as e-mail addresses, in everything a result shows.
- **An MCP server**: `agent-eda-mcp` serves every analysis as an MCP tool over stdio, for agents that call MCP tools instead of writing Python.
- **Safe from threads**: concurrent calls from several threads are safe, and each result keeps its own limitations.
- **Pure Python** on numpy, pandas, pydantic, SciPy, scikit-learn and statsmodels, all preinstalled in AgentCore Code Interpreter.

## Installation

agent-eda needs Python 3.12 or later. It runs with numpy 1.26 through 2.x and pandas 2.2 through 3.x.

Until 0.1 is on PyPI, build the wheel in a clone of this repository with [uv](https://docs.astral.sh/uv/):

```sh
uv build --wheel  # writes dist/agent_eda-0.1.0-py3-none-any.whl
pip install dist/agent_eda-0.1.0-py3-none-any.whl
```

Once it is released, `pip install agent-eda` is enough. The MCP server needs the optional `mcp` extra; see [Use over MCP](#use-over-mcp).

**AgentCore Code Interpreter.** In public network mode, run `pip install agent-eda` in the session. In sandbox mode the session can't reach PyPI: upload the wheel and run `pip install --no-deps <wheel>`, or, if pip is blocked, put the wheel file on `sys.path` and import from it directly. Installs don't carry over to a new session. Code Interpreter returns only printed output, so always `print()` results.

**AgentCore Runtime.** Add `agent-eda` to the agent's container image or code zip. AWS's example zip command uses the `manylinux2014` platform tag, which installs old numpy and pandas releases; use `--python-platform aarch64-manylinux_2_28` to get current ones.

## Quick start

Pass the DataFrame first and every other argument by keyword. There is nothing to set up, and analyses never change the DataFrame.

```python
import pandas as pd

import agent_eda as ae

names = ["ann", "bo", "cy", "di", "ed", "flo", "gus", "bo"]
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
Statistics use the finite values. Quantile method linear; trimmed_mean cuts 10% from each end.
res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis.
Columns (2 of 2 shown):
column  count  missing_count       mean       std       min    median      max
age         8              0     -86.12       369      -999      41.5       62
income      7              1  5.521e+04  1.21e+04  3.95e+04  5.85e+04  7.5e+04
Findings: none
Limitations: none
More: res.to_frame() has all 2 columns; res.findings, res.limitations and res.to_json() have the rest.
```

The mean age of -86.12 is the first sign of trouble. The sample data hides three problems: a -999 that stands for a missing age, a plan written both `'basic'` and `'Basic '`, and one customer recorded twice, under the IDs 102 and 108.

## Analyses

The analyses are listed in the order `ae.guide()` suggests for a first look at a new dataset. Most need nothing but the DataFrame. The ones that need project knowledge, such as your schema or your target, ask for it; they never guess it.

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

Every analysis takes `columns=` to limit it to some columns, except `describe_dataset`, `profile_target` and `validate_column_types`; the default uses every column it can. [Reference](#reference) lists every function with its arguments, result fields, findings and statuses, and `help(ae.<name>)` and `ae.describe_analysis("<name>")` show all the options.

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
summarize_categorical: 1 column holds labels that differ only by case or surrounding spaces: 'plan'.
find_duplicate_rows: 2 rows form 1 duplicate group; 1 is an excess copy (duplicate_share 0.125).
compute_association_matrix: 3 pairs are perfectly associated, an absolute value of 1, so one column may duplicate or derive from the other: ('email', 'age'), ('email', 'income') and ('email', 'churned').
```

The second, third and fourth lines are the three planted problems. The other two are worth a glance but expected here: `customer_id` is meant to be an identifier, and `email` is unique for all but the duplicated customer, so it "predicts" every other column perfectly. A finding says what the data suggests; deciding what it means is your call.

### Other analyses

Analyses outside the first look, by module. A new analysis goes in the first-look table when `ae.guide()` makes it a step of the first look, and in this table otherwise. The [planned analyses](https://github.com/chrislemke/agent-eda/blob/main/docs/planned-analyses.md) are not listed here until they ship.

| Module | Analysis | What it answers | Required arguments |
| --- | --- | --- | --- |

No analysis outside the first look ships yet.

## Discovery

Four functions describe the package from inside Python, so an agent can start without reading the source:

| Function | What it gives you |
| --- | --- |
| `ae.guide()` | Prints one page: how to call an analysis and read its result, every analysis, and the first-look order. |
| `ae.search_analyses(query)` | The analyses that best match a question in your own words, such as "nulls per column". |
| `ae.describe_analysis(name)` | An analysis's parameters, statuses, result fields, finding codes and an example. |
| `ae.list_analyses()` | Every analysis with its module, question and masking support; `.to_frame()` gives a table. |

```python
ae.guide()
print(ae.search_analyses("nulls per column"))
print(ae.describe_analysis("profile_missing_values"))
```

## Reading a result

Every result has the same shared fields, next to its own:

- `res.status` says how the run ended: `completed`, `partial`, `insufficient_data`, `not_applicable`, `missing_prerequisites`, `budget_exceeded`, `blocked_by_policy` or `failed`. `res.status_reason` says why it didn't complete.
- `res.findings` are claims about the data, each with a `code`, a `claim` and an `assessment`: `descriptive`, `suspected`, `violation`, `passed` or `unknown`.
- `res.limitations` say what limits how far the numbers can be read, such as too few values for a statistic.
- `res.parameters` records every argument the analysis used, defaults included.

```python
res = ae.summarize_numeric(df, columns=["age", "income"])
table = res.to_frame()  # every statistic of every column, as a DataFrame
record = res.to_json()  # the whole result as strict JSON
age = res.columns["age"]  # typed fields, such as age.median and age.std
```

The printed text stays within 40 lines of 120 characters however large the data, and says what it leaves out. Lists that grow with the data are capped in the JSON record too, and carry their totals.

A mistake in a call, such as a misspelled column or an unknown option, raises a subclass of `ae.errors.AgentEDAError` before any work starts. Each error has a stable `code`, and its message says what is wrong and how to fix it. Problems in the data never raise; they show up in the status, the findings and the limitations.

Concurrent calls from several threads are safe. An analysis captures the warnings it raises under one lock, so captures run one at a time, each result gets only its own `numerical_warning` limitation, and the caller's warning filters and `warnings.showwarning` stay as they were.

## Projects and masking

Five analyses can show raw values, such as the labels `summarize_categorical` counts or the row IDs `find_duplicate_rows` lists: `validate_column_types`, `detect_missing_sentinels`, `summarize_categorical`, `find_duplicate_rows` and `profile_target`. They take `project=` and `mask_sensitive=`, and `ae.guide()` marks them `[supports masking]`.

A `Project` holds settings that stay the same across calls. Its data policy names the sensitive columns, and masking replaces their values with numbered placeholders. Counts and shares stay:

```python
policy = ae.DataPolicy(sensitive_columns=["email"], mask_sensitive=True)
project = ae.Project(name="churn", data_policy=policy)
print(ae.summarize_categorical(df, columns=["email"], project=project))
```

```text
summarize_categorical: completed on 8 rows x 6 columns
Masked (data_policy.mask_sensitive=True): the values of column 'email' show as <masked 1>, <masked 2> and so on; counts
    and shares are unaffected.
Shares are of each column's non-missing values. Levels go by count, then label; most_common_levels has the top 10.
Columns (1 of 1 shown):
column  non_missing_count  missing_count  cardinality  most_common_levels
email                   8              0            7  <masked 1>: 2 (25%), <masked 2>: 1 (12.5%), <masked 3>: 1 (12....
Findings: none
Limitations: none
More: res.to_frame() has all 7 levels; res.findings, res.limitations and res.to_json() have the rest.
```

How masking is decided:

- It is off by default, even for columns declared sensitive.
- `mask_sensitive=True` in the data policy switches it on for every call that gets the project.
- `mask_sensitive=True` or `False` in a call overrides the project for that call.
- It covers the printed text, `to_dict()`, `to_json()` and `to_frame()` alike.
- It never touches numeric statistics, such as a minimum, or index labels. Pass a non-sensitive `id_column` when the index holds sensitive values.

Save the project once and load it in every later script, which helps an agent that runs each script in a new process:

```python
project.save("project.toml")
project = ae.Project.load("project.toml")
```

`project.toml` is plain TOML that you can edit by hand:

```toml
# An agent-eda Project. Load it with ae.Project.load(path).
name = "churn"

[data_policy]
sensitive_columns = [
    "email",
]
mask_sensitive = true
```

Masking is a guardrail against accidental exposure, not anonymization or access control. The package runs in the agent's own Python process, so an agent that can call it can also read the data directly. [ADR 0001](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0001-data-policy-is-a-guardrail.md) records why.

## Use over MCP

Agents that call MCP tools instead of writing Python, such as Claude Desktop, Cursor or a Claude Code session without a shell, can reach the analyses through a local MCP server. It speaks stdio only. The MCP client starts `agent-eda-mcp` as a subprocess, and the MCP server opens no network connection. [ADR 0006](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0006-local-mcp-server-over-stdio-data-by-path.md) records the design.

### Install and register

The MCP server needs the optional `mcp` extra, which adds the MCP Python SDK and pyarrow:

```sh
pip install 'agent-eda[mcp]'  # or, in a uv project: uv add 'agent-eda[mcp]'
```

Until 0.1 is on PyPI, install the wheel you built with the extra: `pip install 'dist/agent_eda-0.1.0-py3-none-any.whl[mcp]'`. Without the extra, `agent-eda-mcp` exits with the install command on stderr.

Register it with Claude Code, without or with a project file:

```sh
claude mcp add agent-eda -- agent-eda-mcp
claude mcp add agent-eda -- agent-eda-mcp --project /path/to/project.toml
```

Claude Desktop, Cursor and most other MCP clients read an `mcpServers` entry from their JSON configuration. Leave out `args` to start without a project:

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

If the client doesn't start it in the environment where you installed the extra, give the full path that `which agent-eda-mcp` prints as `command`.

Each analysis is an MCP tool with the analysis's name. Its description and input schema come from the docstring that `help()` shows, with each argument's type, allowed values and default. Two more MCP tools find the analyses: `search_analyses` takes a question in your own words, and `list_analyses` lists them all. The MCP server's instructions tell a connecting agent how to pass data, how to read a response and which first-look order to follow.

### Pass data by path

An MCP tool takes `data_path`, the path to a file, where a Python caller passes the DataFrame.

- The extension sets the format, in any case: `.csv`, `.tsv`, `.csv.gz`, `.tsv.gz`, `.parquet` or `.pq`. Any other extension gives the error `unsupported_data_format`.
- The MCP server reads CSV and TSV files as UTF-8, and only an empty field counts as missing. Values such as `N/A`, `NULL` or `-999` reach the analyses as values, so `detect_missing_sentinels` can find them.
- Parquet files keep their dtypes and their stored index.
- A relative path resolves against the MCP server's working directory, the directory the MCP client starts it in. Pass absolute paths if you don't know which directory that is.
- The MCP server reads only local files. It reads a URL such as `s3://bucket/data.csv` as a relative file path, so the call fails with `data_file_not_found`.
- A path that leads nowhere gives `data_file_not_found`, with the absolute path tried and the closest file names in its directory.
- Row IDs in a result are 0-based data-row numbers for a CSV or TSV file, and follow the stored index for a Parquet file. `id_column` names rows by a column's values instead.
- The MCP server keeps the last two files it read in memory, so several analyses of one large file read it once. It reads a file again when its modification time or size changes.

### Pass arguments as JSON

The MCP server reads arguments the way an MCP tool's input schema describes them, so a string with the `date-time`, `date` or `duration` format is an ISO 8601 text.

- `sentinel_values` takes datetimes, dates and durations as ISO 8601 text, such as `{"sentinel_values": {"born": ["1900-01-01T00:00:00"], "wait": ["PT0S"]}}`. A string is read as one when its column holds datetimes, dates or durations and no text. A string for any other column stays text.
- A value that `detect_missing_sentinels` shows, such as `1900-01-01T00:00:00` or `P0DT0H0M0S`, can be declared exactly as shown.
- A time zone must match the column: `"1900-01-01T00:00:00Z"` for a column without one gives `invalid_argument`, with the value to pass instead.

### Read the response

A call returns the printed result as text and the whole `to_dict()` record as structured content, with the same caps on long lists as in Python. The record equals the Python one. The text is worded for an MCP caller: where `print(res)` points to `res.to_frame()` and `res.to_json()`, it points to `table_path` and the structured content. Some MCP clients show the model only one of the two. Claude Code, for one, shows only the structured content. A run that doesn't complete, such as `insufficient_data` from `summarize_numeric` on a file with no rows, is a normal response with its status and reason. A mistake in the call comes back as an MCP tool error (`isError: true`) whose structured content holds the stable error `code` and the `message`. The message says what is wrong and what is allowed as it does in Python, and gives the fix in MCP terms: `{"task": "regression"}` where Python shows `task='regression'`, the MCP tool's input schema where Python points to `help()`, and `project_path` where Python builds a `Project`.

To get past the caps, pass `table_path`. The MCP server writes the result's main table, the one `res.to_frame()` gives in Python without arguments, to a new `.csv`, `.parquet` or `.pq` file. The text ends with `Full table written to <absolute path> (<n> rows).`, and the structured content gains `table_file` with the same `path` and `row_count`.

- `table_path` never overwrites. An existing file gives the error `table_file_exists` before the analysis runs, and its message suggests a free name such as `table-2.csv`. The directory must exist already.
- A cell that holds a list or a mapping, such as the columns of a candidate key from `find_candidate_keys`, is written as JSON text. In Parquet, a column whose values mix types, such as numbers and text, is written as text, as in the CSV.
- An analysis with several tables writes only its main one. `compute_association_matrix` writes its pairs, for example, and its wide matrix stays a Python-only table.

### Masking over MCP

The five MCP tools that support masking take `project_path`, the path to a project file as `Project.save` writes it, in place of `project=`. They keep `mask_sensitive`. `agent-eda-mcp --project PATH` loads one project file at start and uses it for every call. A call's `project_path` replaces it for that call, and a call's `mask_sensitive` wins over both. Masking then covers the text, the structured content and the file `table_path` writes.

Masking over MCP is the same guardrail as in Python: it guards against accidental exposure and is not anonymization or access control. The MCP server enforces nothing, since whoever can call it can usually read the data file directly.

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
- `constant_columns` (suspected): a column holds a single distinct non-missing value in at least two rows.

**Statuses:** `completed`, `failed`

#### `infer_column_semantics`

`ae.infer_column_semantics(data, *, columns=None, declared_schema=None) -> InferColumnSemanticsResult`

What kind of values does each column hold?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to read, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `declared_schema` | `Mapping[str, str \| ColumnSpec] \| None` | `None` | The logical types you declare, by column name: 'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or 'timedelta', or a ColumnSpec (a dict of its fields works too), whose logical_type is the one compared. |

**Fields:** `columns` (`column`, `status`, `status_reason`, `dtype`, `inferred_logical_type`, `declared_logical_type`, `matches_declared`, `candidates`, `flags`, `evidence`), `missing_declared_columns`

- `identifier_like_columns` (suspected): every value is distinct, there are at least two, and the top candidate is integer or text without spaces, as for IDs.
- `constant_columns` (suspected): a single distinct value in at least two rows.
- `all_missing_columns` (suspected): no value in at least one row.
- `numbers_as_text` (suspected): the top candidate is integer or float and some of the numbers are stored as text.
- `dates_as_text` (suspected): the top candidate is datetime and some of the datetimes are ISO 8601 text.
- `non_iso_date_text` (suspected): at least half the values look like dates, and some of them aren't ISO 8601, such as 01/02/2020 or 5 Jan 2020; they are never parsed.
- `boolean_like_columns` (descriptive): the top candidate is boolean and some values are numbers or text that read as booleans, such as 0/1 or yes/no.
- `mixed_value_types` (suspected): the values have more than one Python type, such as int and str: evidence value_types lists them, as summarize_categorical does.
- `declared_type_mismatch` (suspected): a declared column's top candidate differs from its declared logical type.
- `declared_type_match` (passed): a declared column's top candidate is its declared logical type.
- `declared_type_unchecked` (unknown): a declared column has no candidate to compare, since it has no values or none fits.
- `declared_columns_missing` (violation): declared_schema names columns the data doesn't have, listed in `missing_declared_columns`.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `validate_column_types`

`ae.validate_column_types(data, *, schema, id_column=None, project=None, mask_sensitive=None) -> ValidateColumnTypesResult`

Do the columns hold the logical types I declared?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema` | `Mapping[str, str \| ColumnSpec]` | required | The logical type each column should hold, by column name, in the order to report them: 'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or 'timedelta', or a ColumnSpec (a dict of its fields works too). |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces the invalid values of sensitive columns, and row IDs taken from a sensitive id_column, with numbered placeholders, '<masked 1>' and so on, equal values alike, and needs a project that declares sensitive columns; False shows them as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`), `columns` (`column`, `declared_logical_type`, `nullable`, `datetime_format`, `present`, `dtype`, `compatible_count`, `convertible_count`, `invalid_count`, `missing_count`, `invalid_row_ids`, `invalid_distinct_count`, `invalid_values`, `assessment`), `undeclared_columns`

- `invalid_values` (violation): columns that hold invalid values.
- `declared_columns_missing` (violation): declared columns the data doesn't have, each with the closest names.
- `non_nullable_missing_values` (violation): columns declared with nullable=False that have missing values.
- `unchecked_columns` (unknown): columns without a value to check, as when every value is missing.
- `convertible_values` (descriptive): columns with values stored as another type; convert them before you use them as the declared type.
- `valid_columns` (passed): columns that pass every check: only compatible and convertible values, and none missing when declared non-nullable.

**Statuses:** `completed`, `failed`

#### `profile_missing_values`

`ae.profile_missing_values(data, *, columns=None, sentinel_values=None) -> ProfileMissingValuesResult`

How much data is missing, and where?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to profile, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `sentinel_values` | `Mapping[str, Sequence[bool \| int \| float \| str \| datetime \| date \| timedelta]] \| None` | `None` | The sentinel values to count as missing, by column name, such as {'age': [-999], 'city': ['N/A', '']}: one value or a list of values per column, each a bool, int, float, str, datetime (pd.Timestamp too), date or timedelta (pd.Timedelta too). |

**Fields:** `columns` (`column`, `status`, `status_reason`, `native_missing_count`, `sentinel_count`, `missing_count`, `missing_share`, `sentinel_values`), `missing_count`, `missing_share`, `complete_row_count`, `complete_row_share`, `row_distribution` (`missing_count`, `row_count`, `row_share`)

- `all_missing_columns` (suspected): every value of a column is missing or a declared sentinel value.
- `sentinel_values_present` (descriptive): declared sentinel values occur in a column, where they count as missing.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `detect_missing_sentinels`

`ae.detect_missing_sentinels(data, *, columns=None, sentinel_values=None, id_column=None, project=None, mask_sensitive=None) -> DetectMissingSentinelsResult`

Which values look like stand-ins for missing data?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to check, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `sentinel_values` | `Mapping[str, Sequence[bool \| int \| float \| str \| datetime \| date \| timedelta]] \| None` | `None` | The sentinel values you declare, by column name, such as {'age': [-999], 'city': ['N/A', '']}, as profile_missing_values takes them: one value or a list of values per column, each a bool, int, float, str, datetime (pd.Timestamp too), date or timedelta (pd.Timedelta too). |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values, and columns=None leaves it out of the check. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each value of a sensitive column that the result shows, its candidate values, its declared values in parameters and its values as row IDs when it is id_column, with a numbered placeholder, '<masked 1>' for the first value in result order and so on, equal values alike, and needs a project that declares sensitive columns; False shows the values as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`), `candidates` (`column`, `value`, `value_type`, `count`, `share`, `evidence`, `confirmation`, `row_ids`), `columns_without_rules`

- `sentinel_values_present` (descriptive): declared sentinel values occur in a column.
- `suspected_sentinel_values` (suspected): the built-in rules found values that may stand in for missing ones, which stay valid values until you declare them.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `summarize_numeric`

`ae.summarize_numeric(data, *, columns=None, quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), quantile_method='linear', trim_fraction=0.1) -> SummarizeNumericResult`

How are the values in each numeric column distributed?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to summarize, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `quantiles` | `Sequence[float]` | `(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)` | Quantile levels, each from 0 to 1 and listed once, in the order to report them; an empty list computes none. |
| `quantile_method` | `str` | `'linear'` | How numpy.quantile picks or interpolates between values: Hyndman and Fan's definitions 1 to 9 as 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased' and 'normal_unbiased', or 'lower', 'higher', 'midpoint' and 'nearest'. |
| `trim_fraction` | `float` | `0.1` | The share of the finite values cut from each end for the trimmed mean: int(trim_fraction * n) values each, as scipy.stats.trim_mean cuts. |

**Fields:** `columns` (`column`, `status`, `status_reason`, `count`, `missing_count`, `infinite_count`, `zero_count`, `negative_count`, `mean`, `std`, `min`, `max`, `quantiles`, `median`, `trimmed_mean`, `mad`, `mad_scaled`, `skewness`, `excess_kurtosis`), `left_out_columns`, `definitions`

- `constant_columns` (suspected): a column holds a single distinct non-missing value in at least two rows, as describe_dataset decides it.
- `infinite_values` (suspected): a column holds inf or -inf, which every statistic leaves out.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `summarize_categorical`

`ae.summarize_categorical(data, *, columns=None, top_n=10, project=None, mask_sensitive=None) -> SummarizeCategoricalResult`

Which values occur in each categorical column, and how often?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to summarize, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `top_n` | `int` | `10` | How many of each column's most common levels most_common_levels holds and the printed text shows, from 1 to 100; defaults to 10. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each level of a sensitive column with a numbered placeholder, '<masked 1>' for the most common and so on in level order, and needs a project that declares sensitive columns; False shows the levels as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`), `columns` (`column`, `status`, `status_reason`, `non_missing_count`, `missing_count`, `cardinality`, `levels`, `most_common_levels`, `left_out_level_count`, `seen_once_count`, `unused_categories`, `value_types`, `mixed_value_types`, `case_or_space_variants`), `left_out_columns`, `share_denominator`, `level_order`

- `case_or_space_variants` (suspected): a column holds labels that differ only by case or surrounding spaces, such as 'Yes' and 'yes '.
- `mixed_value_types` (suspected): a column holds values of more than one Python type, such as 1 and '1'.
- `unused_categories` (descriptive): a categorical column has categories that no value takes.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `find_duplicate_rows`

`ae.find_duplicate_rows(data, *, columns=None, id_column=None, project=None, mask_sensitive=None) -> FindDuplicateRowsResult`

Are there exact duplicate rows?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns in which two rows must be equal to count as duplicates, in the order to compare them: a list, a tuple, a pandas Index such as df.columns[:3], or a single name. |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as an order number; row IDs are then its values, and columns=None leaves it out of the comparison. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces each row ID taken from a sensitive id_column with a numbered placeholder, '<masked 1>' for the first ID the result shows and so on, equal IDs alike, and needs a project that declares sensitive columns; False shows the IDs as stored. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`), `compared_columns`, `missing_values_compare_equal`, `duplicate_groups` (`size`, `row_ids`), `duplicate_group_count`, `duplicate_group_row_count`, `excess_copy_count`, `duplicate_share`

- `duplicate_rows` (suspected): two or more rows are equal in every compared column.
- `no_duplicate_rows` (descriptive): no two rows are equal in every compared column.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `find_candidate_keys`

`ae.find_candidate_keys(data, *, columns=None, max_key_size=2, max_combinations=10000) -> FindCandidateKeysResult`

Which columns, alone or combined, uniquely identify rows?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to search, in the order to report them: a list, a tuple, a pandas Index or a single name. |
| `max_key_size` | `int` | `2` | The most columns a key may combine, at least 1; defaults to 2. |
| `max_combinations` | `int` | `10000` | The most column sets the search examines, at least 1; defaults to 10,000, which covers every set of up to 2 of 140 columns. |

**Fields:** `keys` (`columns`, `distinct_count`, `duplicate_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `keys_with_missing_values` (`columns`, `distinct_count`, `duplicate_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `most_unique_column_sets` (`columns`, `distinct_count`, `duplicate_row_count`, `missing_row_count`, `uniqueness_share`, `is_minimal`), `exhaustive`, `search` (`column_count`, `combination_count`, `examined_count`, `rejected_on_subset_count`, `checked_on_all_rows_count`, `subset_row_count`), `left_out_columns`

- `keys_found` (descriptive): column sets identify every row and have no missing values.
- `keys_with_missing_values` (descriptive): column sets identify every row but have missing values, so they can't serve as primary keys.
- `no_key_found` (descriptive): no examined set identifies every row; the claim names the most unique set and its duplicate rows.

**Statuses:** `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`, `failed`

#### `profile_target`

`ae.profile_target(data, *, target_column, task, expected_labels=None, rare_share=None, id_column=None, project=None, mask_sensitive=None) -> ProfileTargetResult`

What does the target look like for this task?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target_column` | `str` | required | Name of the column a model learns to predict. |
| `task` | `str` | required | What the model does with the target, never inferred from the dtype. |
| `expected_labels` | `Sequence[str \| bool \| int \| float] \| None` | `None` | The labels the task allows, such as ['no', 'yes'] or [0, 1]: exactly 2 for binary_classification and at least 2 for multiclass_classification, each listed once; regression takes none. |
| `rare_share` | `float \| None` | `None` | For classification, the share of the valid labels below which a class counts as rare, such as 0.05; greater than 0 and less than 1. |
| `id_column` | `str \| None` | `None` | Name of a column whose values identify rows, such as a customer number; the row IDs of labels outside expected_labels are then its values. |
| `project` | `Project \| None` | `None` | The project, whose data policy names the sensitive columns and says whether to mask them. |
| `mask_sensitive` | `bool \| None` | `None` | True replaces the labels of a sensitive target, the classes, the labels outside expected_labels and expected_labels as parameters records it, with numbered placeholders, '<masked 1>' for its first label in level order and so on, and the row IDs taken from a sensitive id_column too; it needs a project that declares sensitive columns. |

**Fields:** `masking` (`applied`, `decided_by`, `columns`, `index_columns`, `absent_columns`), `non_missing_count`, `missing_count`, `missing_share`, `valid_label_count`, `class_count`, `classes` (`label`, `label_type`, `count`, `share`), `majority_share`, `minority_share`, `imbalance_ratio`, `unexpected_label_count`, `unexpected_labels` (`label`, `label_type`, `count`, `row_ids`), `rare_class_count`, `rare_classes` (`label`, `label_type`, `count`, `share`), `statistics` (`column`, `status`, `status_reason`, `count`, `missing_count`, `infinite_count`, `zero_count`, `negative_count`, `mean`, `std`, `min`, `max`, `quantiles`, `median`, `trimmed_mean`, `mad`, `mad_scaled`, `skewness`, `excess_kurtosis`), `non_finite_count`, `level_order`

- `single_class` (violation): every valid label is the same class, so a classifier has nothing to tell apart.
- `more_than_two_classes` (violation): a binary_classification target has more than two classes.
- `unexpected_labels` (violation): labels lie outside expected_labels.
- `no_unexpected_labels` (passed): every label that isn't missing is in expected_labels.
- `all_labels_distinct` (suspected): the values of a classification target that aren't missing are all distinct, as an identifier's or a measurement's are, not class labels'.
- `constant_target` (violation): at least two values of a regression target aren't missing and all of them are equal, as describe_dataset decides it.
- `infinite_values` (suspected): a regression target holds inf or -inf, which every statistic leaves out.
- `missing_labels` (descriptive): rows have no target value.
- `rare_classes` (descriptive): classes have a share below rare_share.

**Statuses:** `completed`, `insufficient_data`, `failed`

#### `compute_association_matrix`

`ae.compute_association_matrix(data, *, columns=None, categorical_columns=None, numeric_method='pearson', categorical_method='cramers_v_bias_corrected', max_categories=50, max_pairs=5000) -> ComputeAssociationMatrixResult`

How strongly are pairs of columns associated?

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | `Sequence[str] \| None` | `None` | Names of the columns to pair, in the order to report them: a list, a tuple, a pandas Index such as df.columns[:10], or a single name. |
| `categorical_columns` | `Sequence[str] \| None` | `None` | Numeric columns to treat as categorical, so that codes such as 1, 2 and 3 are compared as categories instead of correlated as numbers. |
| `numeric_method` | `str` | `'pearson'` | The measure for two numeric columns: 'pearson', the default, Pearson's r for linear association; 'spearman', Pearson's r of the ranks, tied values sharing their mean rank, for monotonic association; or 'kendall', Kendall's tau-b, which adjusts for ties. |
| `categorical_method` | `str` | `'cramers_v_bias_corrected'` | The measure for two categorical columns: 'cramers_v_bias_corrected', the default, Cramér's V with Bergsma's (2013) correction, which removes most of the upward bias that few rows and many levels give V; or 'cramers_v', the uncorrected V of scipy.stats.contingency.association. |
| `max_categories` | `int` | `50` | The most distinct values a categorical column may have; a column with more is skipped with reason too_many_categories. |
| `max_pairs` | `int` | `5000` | The most pairs the call computes, a work limit. |

**Fields:** `pairs` (`x_column`, `y_column`, `x_type`, `y_type`, `method`, `value`, `pair_count`, `direction`, `undefined_reason`), `columns` (`column`, `dtype`, `type`, `value_count`, `missing_count`, `infinite_count`, `level_count`), `skipped_columns` (`column`, `dtype`, `reason`, `message`)

- `perfect_associations` (suspected): a pair has an absolute value of 1, within 1e-9, so one column may duplicate or derive from the other.

**Statuses:** `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`, `failed`

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

**Returns:** `matches` holds the best matches, best first, each with its `name`, `summary`, `score` and `matched_terms`.

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
| `data_policy` | `DataPolicy \| None` | `None` | Which columns are sensitive and whether to mask them. |

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
| `status` | How the run ended: completed, partial, insufficient_data, not_applicable, missing_prerequisites, budget_exceeded, blocked_by_policy or failed. |
| `status_reason` | Why the run didn't complete; null when status is completed. |
| `parameters` | The resolved arguments by name, defaults included. data appears only as its row_count and column_count. |
| `support` | The counts behind the result. |
| `findings` | Claims about the data. |
| `limitations` | What limits how far the result can be read. |
| `execution` | Library versions used for the run. |

The shared methods:

- `res.to_frame()`: The complete main table of the result as a DataFrame.
- `res.to_dict()`: The result record as JSON-compatible Python objects.
- `res.to_json()`: The result record from to_dict() as strict JSON, without NaN or Infinity.
- `ResultClass.record_schema()`: The JSON Schema of the record that to_dict() and to_json() give.

### Errors

Exceptions that agent-eda raises when a call is wrong.

| Code | Exception | Raised when |
| --- | --- | --- |
| `invalid_argument` | `InvalidArgumentError` | An argument has the wrong type or value, such as a number outside its range, an unknown choice, a list that repeats a value, or a declared sentinel value that can never match its column. |
| `invalid_call` | `InvalidArgumentError` | The call itself is malformed, such as an option passed by position, an unknown keyword or a missing DataFrame. |
| `unsupported_data_type` | `InvalidArgumentError` | Data isn't a pandas DataFrame. |
| `ambiguous_column_labels` | `InvalidArgumentError` | The data's column labels repeat, share a string form or form a MultiIndex, so a name can't pick out one column. |
| `column_not_found` | `ColumnNotFoundError` | A column name isn't in the data. |
| `unsupported_column_type` | `ColumnTypeError` | A named column has a dtype the analysis can't use, or an id_column holds values that can't be hashed, such as lists. |
| `unknown_analysis` | `UnknownAnalysisError` | describe_analysis got a name that isn't an analysis. |
| `invalid_project_file` | `ProjectFileError` | Project.load got a file that isn't UTF-8 TOML, and the message gives the line at fault, that lacks the [data_policy] table or one of its keys, or that holds a value a Project can't take. |
| `unknown_project_key` | `ProjectFileError` | Project.load got a file with a key that is neither a field of the Project nor one of its data policy, such as a misspelling. |
| `project_file_not_found` | `ProjectFileNotFoundError` | Project.load found no file at the path. |
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
- [AGENTS.md](https://github.com/chrislemke/agent-eda/blob/main/AGENTS.md): the rules for anyone writing code here, people or agents, and the commands.
- [GLOSSARY.md](https://github.com/chrislemke/agent-eda/blob/main/GLOSSARY.md): the terms used in code, docstrings and messages.
- [Planned analyses](https://github.com/chrislemke/agent-eda/blob/main/docs/planned-analyses.md): the backlog, with proposed names and modules.
- Architecture decisions:
  - [ADR 0001](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0001-data-policy-is-a-guardrail.md): the data policy is a guardrail.
  - [ADR 0002](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0002-pandas-in-numpy-inside.md): pandas at the boundary, NumPy inside.
  - [ADR 0003](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0003-no-llm-calls-in-the-package.md): no LLM calls in the package.
  - [ADR 0004](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0004-plain-functions-with-explicit-arguments.md): plain functions with explicit arguments.
  - [ADR 0005](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0005-pure-python-within-code-interpreter-dependencies.md): pure Python within Code Interpreter's dependencies.
  - [ADR 0006](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0006-local-mcp-server-over-stdio-data-by-path.md): a local MCP server over stdio, data by path.

## License

MIT. See [LICENSE](https://github.com/chrislemke/agent-eda/blob/main/LICENSE).
