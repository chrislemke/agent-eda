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

Every public name in `agent_eda`, with its signature and what it does. The docstrings hold the full detail: `help(ae.<name>)` shows one, and `ae.describe_analysis("<name>")` gives an analysis's parameters, result fields and finding codes as data.

- [Analysis functions](#analysis-functions): `describe_dataset`, `infer_column_semantics`, `validate_column_types`, `profile_missing_values`, `detect_missing_sentinels`, `summarize_numeric`, `summarize_categorical`, `find_duplicate_rows`, `find_candidate_keys`, `profile_target`, `compute_association_matrix`
- [Discovery functions](#discovery-functions): `guide`, `search_analyses`, `describe_analysis`, `list_analyses`
- [Projects and schemas](#projects-and-schemas): `Project`, `DataPolicy`, `ColumnSpec`, `LogicalType`
- [Results](#results): `AnalysisResult` and the methods every result has
- [Errors](#errors): every error code and the exception that carries it
- [MCP server command](#mcp-server-command): `agent-eda-mcp`

### Analysis functions

Every analysis takes the DataFrame first, as `data`, and never changes it. All other arguments are keyword-only. These mean the same in every analysis that takes them:

- `columns`: the columns to analyse, as a list, a tuple, a pandas Index or a single name, in the order to report them. `None` selects every column the analysis can use, and the result names the ones it left out.
- `id_column`: a column whose values name rows in the result, such as an order number. `None` uses the index labels.
- `project`: a `Project` whose data policy names the sensitive columns. `None` declares none.
- `mask_sensitive`: `True` replaces the values of sensitive columns that the result shows with placeholders such as `<masked 1>`, and `False` shows them. `None` follows the project, and masks nothing without one. See [Projects and masking](#projects-and-masking).

Each analysis returns its own result class, which has the shared fields and methods under [Results](#results) next to the fields listed here. A finding's assessment is in brackets: `descriptive` states a fact, `suspected` a possible problem, `violation` a broken declaration, `passed` a declared check that held and `unknown` one that couldn't run.

#### `describe_dataset`

`ae.describe_dataset(data, *, source=None, source_version=None) -> DescribeDatasetResult`

What does this dataset contain? Gives the shape, each column's dtype, missing and distinct counts and memory, the index, and the provenance you supply, and flags structural problems. It never shows a raw value. Call it first: the other analyses raise on the bad column labels it reports.

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | `None` | Where the data came from, such as a file path or table name, recorded as given. `None` records it as not supplied; it is never inferred. |
| `source_version` | `None` | The version of the source, such as a snapshot date or a commit, recorded as given. |

Read `row_count`, `column_count`, `memory_bytes`, `columns` (one entry per column), `index` and `provenance`. `to_frame()` gives one row per column.

- `empty_data` (suspected): the data has no rows, no columns or neither.
- `duplicated_column_labels` (suspected): a column label occurs more than once.
- `colliding_column_labels` (suspected): different labels share a string form, such as `1` and `'1'`.
- `multiindex_columns` (suspected): the column labels form a MultiIndex.
- `non_unique_index` (suspected): index labels repeat, so they don't identify rows.
- `all_missing_columns` (suspected): a column holds only missing values.
- `constant_columns` (suspected): a column holds one distinct value in at least two rows.

**Statuses:** `completed`, `failed`

#### `infer_column_semantics`

`ae.infer_column_semantics(data, *, columns=None, declared_schema=None) -> InferColumnSemanticsResult`

What kind of values does each column hold? Ranks the logical types `boolean`, `integer`, `float`, `categorical`, `text`, `datetime` and `timedelta` for each column by its values, not its dtype, with the evidence. It flags identifiers, numbers and ISO 8601 dates stored as text and booleans such as yes/no; whole numbers written with a leading zero, such as postal codes, count as text, not as numbers. Scores come from fixed rules and aren't probabilities.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to read. `None` reads every column. |
| `declared_schema` | `None` | The logical types you expect, by column: a type name or a `ColumnSpec`. Each is compared with the top candidate and never overwritten. `None` compares nothing. |

Read `columns["<name>"]`: `inferred_logical_type`, `candidates` (each with `score` and `reasons`), `flags` and `evidence`. `to_frame()` gives the evidence per column, and `to_frame(table="candidates")` every candidate.

- `identifier_like_columns` (suspected): every value is distinct and looks like an ID, so don't use the column as a number or a feature.
- `constant_columns` (suspected): a column holds one distinct value in at least two rows.
- `all_missing_columns` (suspected): a column holds no value.
- `numbers_as_text` (suspected): a column stores numbers as text.
- `dates_as_text` (suspected): a column stores ISO 8601 dates as text.
- `non_iso_date_text` (suspected): a column holds dates in another format, such as `01/02/2020`, which are never parsed.
- `boolean_like_columns` (descriptive): a column holds booleans stored as numbers or text, such as 0/1 or yes/no.
- `mixed_value_types` (suspected): a column holds values of more than one Python type, such as `1` and `'1'`.
- `declared_type_mismatch` (suspected): a declared column's top candidate differs from its declared type.
- `declared_type_match` (passed): a declared column's top candidate is its declared type.
- `declared_type_unchecked` (unknown): a declared column has no candidate to compare.
- `declared_columns_missing` (violation): `declared_schema` names columns the data lacks.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `validate_column_types`

`ae.validate_column_types(data, *, schema, id_column=None, project=None, mask_sensitive=None) -> ValidateColumnTypesResult`

Do the columns hold the logical types I declared? Checks every value of each declared column and counts the values already stored as the type, the ones that convert to it exactly, such as `'42'` for `integer`, the invalid ones and the missing ones. It lists example invalid values with their row IDs, converts nothing, and supports masking.

| Argument | Default | Meaning |
| --- | --- | --- |
| `schema` | required | The logical type of each column: `'boolean'`, `'integer'`, `'float'`, `'categorical'`, `'text'`, `'datetime'` or `'timedelta'`, or a `ColumnSpec` that also sets `nullable` and `datetime_format`. A declared column the data lacks is reported, not raised. |
| `id_column` | `None` | A column whose values name the rows that hold invalid values. |
| `project` | `None` | The project that names the sensitive columns. |
| `mask_sensitive` | `None` | Whether to mask the invalid values of sensitive columns and row IDs from a sensitive `id_column`. |

Read `columns["<name>"]`: `compatible_count`, `convertible_count`, `invalid_count` and `missing_count`, which add up to the rows, then `invalid_values`, `invalid_row_ids` and `assessment`. `to_frame()` gives one row per declared column, and `to_frame(table="invalid_values")` every distinct invalid value.

- `invalid_values` (violation): columns hold values that neither are of the declared type nor convert to it.
- `declared_columns_missing` (violation): declared columns the data lacks, each with the closest names.
- `non_nullable_missing_values` (violation): columns declared with `nullable=False` have missing values.
- `unchecked_columns` (unknown): columns have no value to check, such as when every value is missing.
- `convertible_values` (descriptive): columns hold values stored as another type that convert exactly; convert them before use.
- `valid_columns` (passed): columns pass every check.

**Statuses:** `completed`, `failed`

#### `profile_missing_values`

`ae.profile_missing_values(data, *, columns=None, sentinel_values=None) -> ProfileMissingValuesResult`

How much data is missing, and where? Counts missing values per column and per row: what pandas counts as missing (None, NaN, NaT and pd.NA) plus the sentinel values you declare, such as -999, never counting a value twice. It also gives the complete rows and how many rows miss each number of values. Nothing counts as a sentinel value unless you declare it; `detect_missing_sentinels` finds candidates.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to profile. `None` selects every column. |
| `sentinel_values` | `None` | Values that stand for missing ones, by column, such as `{"age": [-999], "city": ["N/A", ""]}`. They match exactly. `None` counts only the values pandas treats as missing. |

Read `missing_count`, `missing_share`, `complete_row_count`, `complete_row_share` and `row_distribution`, and per column `columns["<name>"]` with `native_missing_count`, `sentinel_count`, `missing_count` and `missing_share`. `to_frame()` gives one row per column, and `to_frame(table="row_distribution")` the rows per number of missing values.

- `all_missing_columns` (suspected): every value of a column is missing or a declared sentinel value.
- `sentinel_values_present` (descriptive): declared sentinel values occur in a column, where they count as missing.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `detect_missing_sentinels`

`ae.detect_missing_sentinels(data, *, columns=None, sentinel_values=None, id_column=None, project=None, mask_sensitive=None) -> DetectMissingSentinelsResult`

Which values look like stand-ins for missing data? Fixed rules suspect values such as -999, `'N/A'`, empty text or 1900-01-01: values on a built-in list, and a number that repeats at a column's minimum or maximum far from all the others. It only suspects. Declare the values you confirm in `sentinel_values`, and `profile_missing_values` counts them as missing. It supports masking.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to check. `None` checks every column except `id_column`. |
| `sentinel_values` | `None` | The sentinel values you have confirmed, by column, as `profile_missing_values` takes them. They come back as declared, apart from the suspected ones. |
| `id_column` | `None` | A column whose values name the rows that hold each candidate. |
| `project` | `None` | The project that names the sensitive columns. |
| `mask_sensitive` | `None` | Whether to mask the values of sensitive columns: the candidate values, the declared values in `parameters` and, for a sensitive `id_column`, the row IDs. |

Read `candidates`, each with `column`, `value`, `count`, `share`, `evidence`, `confirmation` (declared or suspected) and `row_ids`. `columns_without_rules` names the checked columns no rule reads, such as booleans. `to_frame()` gives every candidate.

- `sentinel_values_present` (descriptive): declared sentinel values occur in a column.
- `suspected_sentinel_values` (suspected): the rules found values that may stand in for missing ones; they stay valid until you declare them.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `summarize_numeric`

`ae.summarize_numeric(data, *, columns=None, quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), quantile_method='linear', trim_fraction=0.1) -> SummarizeNumericResult`

How are the values in each numeric column distributed? Gives the count and the missing, infinite, zero and negative counts, then the mean, standard deviation, minimum, maximum, quantiles, median, trimmed mean, median absolute deviation, skewness and excess kurtosis, all over the finite values. Booleans are left out unless named in `columns`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to summarize. `None` selects every numeric column except booleans. A text, categorical or datetime column named here raises `ColumnTypeError`. |
| `quantiles` | `(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)` | Quantile levels from 0 to 1, each listed once. An empty list computes none. |
| `quantile_method` | `'linear'` | How `numpy.quantile` picks or interpolates: Hyndman and Fan's definitions 1 to 9 (`'inverted_cdf'`, `'averaged_inverted_cdf'`, `'closest_observation'`, `'interpolated_inverted_cdf'`, `'hazen'`, `'weibull'`, `'linear'`, `'median_unbiased'`, `'normal_unbiased'`), or `'lower'`, `'higher'`, `'midpoint'` and `'nearest'`. |
| `trim_fraction` | `0.1` | The share of values cut from each end for the trimmed mean, from 0 up to but excluding 0.5. |

Read `columns["<name>"]`: `count`, `missing_count`, `infinite_count`, `zero_count`, `negative_count`, `mean`, `std`, `min`, `max`, `quantiles`, `median`, `trimmed_mean`, `mad`, `mad_scaled`, `skewness` and `excess_kurtosis`. `definitions` states how each statistic is defined. `to_frame()` gives one row per column, with one column per quantile level, such as `q0.25`.

- `constant_columns` (suspected): a column holds one distinct value in at least two rows.
- `infinite_values` (suspected): a column holds inf or -inf, which every statistic leaves out.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `summarize_categorical`

`ae.summarize_categorical(data, *, columns=None, top_n=10, project=None, mask_sensitive=None) -> SummarizeCategoricalResult`

Which values occur in each categorical column, and how often? For each text, object, categorical and boolean column it gives every level's count and share of the non-missing values, the cardinality and the most common levels. It flags labels that differ only by case or surrounding spaces, mixed value types and unused categories. Name a numeric column in `columns` to count its values, such as codes 1, 2 and 3. It supports masking.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to summarize. `None` selects every text, object, categorical and boolean column. A datetime or duration column named here raises. |
| `top_n` | `10` | How many of each column's most common levels `most_common_levels` holds and the text prints, from 1 to 100. |
| `project` | `None` | The project that names the sensitive columns. |
| `mask_sensitive` | `None` | Whether to mask the levels of sensitive columns. |

Read `columns["<name>"]`: `non_missing_count`, `missing_count`, `cardinality`, `levels`, `most_common_levels`, `case_or_space_variants`, `unused_categories` and `value_types`. `to_frame()` gives every level of every column.

- `case_or_space_variants` (suspected): a column holds labels that differ only by case or surrounding spaces, such as `'Yes'` and `'yes '`.
- `mixed_value_types` (suspected): a column holds values of more than one Python type, such as `1` and `'1'`.
- `unused_categories` (descriptive): a categorical column has categories that no value takes.

**Statuses:** `completed`, `partial`, `insufficient_data`, `not_applicable`, `failed`

#### `find_duplicate_rows`

`ae.find_duplicate_rows(data, *, columns=None, id_column=None, project=None, mask_sensitive=None) -> FindDuplicateRowsResult`

Are there exact duplicate rows? Groups the rows that are equal in every compared column and gives each group's row IDs, the excess copies and the duplicate share. Missing values compare equal, as in pandas. It removes no rows, and it supports masking of row IDs taken from a sensitive `id_column`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns in which two rows must be equal. `None` compares every column except `id_column`. |
| `id_column` | `None` | A column whose values name rows, such as an order number. `columns=None` leaves it out of the comparison, so unique IDs don't hide repeated records. |
| `project` | `None` | The project that names the sensitive columns. |
| `mask_sensitive` | `None` | Whether to mask row IDs taken from a sensitive `id_column`. |

Read `duplicate_groups` (each with `size` and `row_ids`), `duplicate_group_count`, `duplicate_group_row_count`, `excess_copy_count` and `duplicate_share`. `to_frame()` gives every row in a group with its group number.

- `duplicate_rows` (suspected): two or more rows are equal in every compared column.
- `no_duplicate_rows` (descriptive): no two rows are equal in every compared column.

**Statuses:** `completed`, `insufficient_data`, `not_applicable`, `failed`

#### `find_candidate_keys`

`ae.find_candidate_keys(data, *, columns=None, max_key_size=2, max_combinations=10000) -> FindCandidateKeysResult`

Which columns, alone or combined, uniquely identify rows? Searches single columns and combinations of up to `max_key_size` columns for minimal keys. It lists keys with missing values apart, since they can't serve as primary keys, and when no key exists, the column sets that come closest. Missing values compare equal.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to search. `None` searches every column. A column of values that can't be hashed, such as lists, is left out. |
| `max_key_size` | `2` | The most columns a key may combine, at least 1. |
| `max_combinations` | `10000` | The most column sets the search examines, at least 1. A search that needs more stops with `budget_exceeded` and keeps what it found. |

Read `keys`, `keys_with_missing_values` and `most_unique_column_sets`, each entry with `columns`, `distinct_count`, `duplicate_row_count`, `missing_row_count` and `uniqueness_share`, and `exhaustive`. `to_frame()` gives every entry with its kind.

- `keys_found` (descriptive): column sets identify every row and have no missing values.
- `keys_with_missing_values` (descriptive): column sets identify every row but have missing values.
- `no_key_found` (descriptive): no examined set identifies every row; the claim names the most unique set.

**Statuses:** `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`, `failed`

#### `profile_target`

`ae.profile_target(data, *, target_column, task, expected_labels=None, rare_share=None, id_column=None, project=None, mask_sensitive=None) -> ProfileTargetResult`

What does the target look like for this task? For classification it counts each class, gives the majority and minority shares and the imbalance ratio, and checks the labels against the ones you expect. For regression it gives the statistics of `summarize_numeric`. It flags a target that doesn't fit the task, such as a single class or a constant value, and supports masking.

| Argument | Default | Meaning |
| --- | --- | --- |
| `target_column` | required | The column a model learns to predict. |
| `task` | required | `'binary_classification'`, `'multiclass_classification'` or `'regression'`. It is never inferred from the dtype, and a regression target must hold numbers. |
| `expected_labels` | `None` | The labels the task allows, such as `['no', 'yes']`: exactly 2 for binary and at least 2 for multiclass classification, none for regression. `None` skips the check. |
| `rare_share` | `None` | For classification, the share below which a class counts as rare, such as 0.05, between 0 and 1. `None` skips the check. |
| `id_column` | `None` | A column whose values name the rows with labels outside `expected_labels`. |
| `project` | `None` | The project that names the sensitive columns. |
| `mask_sensitive` | `None` | Whether to mask the labels of a sensitive target and row IDs from a sensitive `id_column`. |

For classification, read `classes` (each with `label`, `count` and `share`), `class_count`, `majority_share`, `minority_share`, `imbalance_ratio`, `unexpected_labels` and `rare_classes`. For regression, read `statistics`, which holds the fields of `summarize_numeric`, and `non_finite_count`. Both have `missing_count` and `missing_share`. `to_frame()` gives every label, or for regression the statistics in one row.

- `single_class` (violation): every valid label is the same class.
- `more_than_two_classes` (violation): a `binary_classification` target has more than two classes.
- `unexpected_labels` (violation): labels lie outside `expected_labels`.
- `no_unexpected_labels` (passed): every label is in `expected_labels`.
- `all_labels_distinct` (suspected): every label of a classification target is distinct, as an identifier's are.
- `constant_target` (violation): every value of a regression target is equal.
- `infinite_values` (suspected): a regression target holds inf or -inf.
- `missing_labels` (descriptive): rows have no target value.
- `rare_classes` (descriptive): classes have a share below `rare_share`.

**Statuses:** `completed`, `insufficient_data`, `failed`

#### `compute_association_matrix`

`ae.compute_association_matrix(data, *, columns=None, categorical_columns=None, numeric_method='pearson', categorical_method='cramers_v_bias_corrected', max_categories=50, max_pairs=5000) -> ComputeAssociationMatrixResult`

How strongly are pairs of columns associated? Measures every pair of numeric and categorical columns, each on its own complete rows, and lists the strongest first: a correlation between two numeric columns, Cramér's V between two categorical ones and the correlation ratio from a categorical to a numeric column. It measures association, not causation, and gives no p-values.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | The columns to pair. `None` selects every numeric, boolean, text, object and categorical column, and lists the others in `skipped_columns`. A datetime column named here raises. |
| `categorical_columns` | `None` | Numeric columns to treat as categories, such as codes 1, 2 and 3. |
| `numeric_method` | `'pearson'` | The measure for two numeric columns: `'pearson'`, `'spearman'` or `'kendall'` (tau-b). |
| `categorical_method` | `'cramers_v_bias_corrected'` | The measure for two categorical columns: `'cramers_v_bias_corrected'` or `'cramers_v'`. |
| `max_categories` | `50` | The most distinct values a categorical column may have, at least 2. A column with more is skipped. |
| `max_pairs` | `5000` | The most pairs to compute, at least 1. With more, the call returns `budget_exceeded` without computing any. |

Read `pairs`, each with `x_column`, `y_column`, `method`, `value`, `pair_count`, `direction` and, when the value is null, `undefined_reason`. `columns` describes each paired column and `skipped_columns` says why the others were left out. `to_frame()` gives every pair, and `to_frame(table="matrix")` the wide matrix.

- `perfect_associations` (suspected): a pair has an absolute value of 1, so one column may duplicate or derive from the other.

**Statuses:** `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`, `failed`

### Discovery functions

#### `guide`

`ae.guide() -> None`

Prints one page on the package: how to call an analysis and read its result, every analysis by module with its question, how to mask sensitive values, and the order for a first look at a new dataset. Start here.

#### `search_analyses`

`ae.search_analyses(query, limit=5) -> SearchListing`

Finds the analyses that best match a question in your own words, such as `"nulls per column"`. It matches keywords against names, tags, summaries and the When to use text, with a few synonyms, without embeddings or network. `limit` caps the matches, at least 1. Read `matches`, best first, each with `name`, `summary`, `score` and `matched_terms`, and `match_count`.

#### `describe_analysis`

`ae.describe_analysis(name) -> AnalysisDescription`

Gives everything needed for a correct first call of one analysis: its question, signature and when to use it; each parameter with its type, default, allowed values and valid range; its statuses, finding codes and result fields; whether it supports masking; and an example. `name` is a name such as `'summarize_numeric'` or a qualified one. An unknown name raises `UnknownAnalysisError` with the closest names. `to_json()` adds `parameters_schema`, a JSON Schema of the parameters.

#### `list_analyses`

`ae.list_analyses() -> AnalysisListing`

Lists every analysis by module, with the question it answers and whether it supports masking. `analyses` holds each one's parameters, statuses, finding codes and result class, and `to_frame()` gives the same as a table.

### Projects and schemas

#### `Project`

`ae.Project(*, name=None, data_policy=None)`

Holds the settings that stay the same across calls: a name and the data policy. Pass it as `project=` to the analyses that support masking.

- `project.save(path, *, overwrite=False) -> Path` writes it to a TOML file and returns the absolute path. An existing file raises `ProjectFileExistsError` unless `overwrite=True`.
- `ae.Project.load(path) -> Project` reads a file that `save` wrote or you wrote by hand. A missing or misspelled key raises, so a broken file can't leave masking off.

#### `DataPolicy`

`ae.DataPolicy(*, sensitive_columns=(), mask_sensitive=False)`

Names the sensitive columns and says whether to mask their values. `mask_sensitive=True` needs at least one sensitive column, and a call's own `mask_sensitive` wins. Masking is a guardrail against accidental exposure, not anonymization or access control.

#### `ColumnSpec`

`ae.ColumnSpec(logical_type, *, nullable=None, datetime_format=None)`

Declares one column for `schema` in `validate_column_types` and `declared_schema` in `infer_column_semantics`. `logical_type` is a `LogicalType`. `nullable=False` makes missing values a violation, and `datetime_format`, a `strptime` format such as `'%d/%m/%Y'`, lets datetime text in that format convert. A dict of its fields works too, and a type name alone is short for a `ColumnSpec` with only `logical_type`.

#### `LogicalType`

The logical type names as a `Literal`: `'boolean'`, `'integer'`, `'float'`, `'categorical'`, `'text'`, `'datetime'` and `'timedelta'`.

### Results

#### `AnalysisResult`

The base class of every result. Each analysis returns its own subclass, such as `SummarizeNumericResult`, from `ae.results`. The shared fields:

| Field | What it holds |
| --- | --- |
| `analysis` | The analysis name. |
| `status` | How the run ended: `completed`, `partial`, `insufficient_data`, `not_applicable`, `missing_prerequisites`, `budget_exceeded`, `blocked_by_policy` or `failed`. |
| `status_reason` | Why the run didn't complete, with a `code` and a `message`; null when it completed. |
| `method` | The method used, or null when the analysis has none. |
| `parameters` | Every argument the analysis used, defaults included. |
| `support` | The rows and columns the result rests on. |
| `findings` | Claims about the data, each with a `code`, a `claim`, an `assessment`, its `columns` and the result `fields` behind it. |
| `limitations` | What limits how far the result can be read, each with a `code`, a `reason` and the `fields` it affects. |
| `execution` | The versions of agent-eda, Python and the libraries used. |

The shared methods:

| Method | What it gives |
| --- | --- |
| `print(res)` | Plain text from a fixed template, at most 40 lines of 120 characters, that says what it leaves out. |
| `res.to_frame()` | The complete main table as a DataFrame. Some results take `table=` for another, as their entries above say. |
| `res.to_dict()` | The whole record as JSON-compatible objects. Lists that grow with the data keep their first entries (10 row IDs, 5 example values, 100 of anything else) and gain a `<field>_total` with the full count. |
| `res.to_json()` | The record from `to_dict()` as strict JSON, without NaN or Infinity. |
| `ResultClass.record_schema()` | The JSON Schema of that record. |

### Errors

A mistake in a call raises before any work starts. Every exception subclasses `ae.errors.AgentEDAError` and the closest built-in exception, and carries a stable `code` to branch on. Its message says what is wrong, what is allowed and how to fix it. Problems in the data never raise; they come back in the status, the findings and the limitations.

| Code | Exception | Raised when |
| --- | --- | --- |
| `invalid_argument` | `InvalidArgumentError` | An argument has the wrong type or value, such as a number out of range, an unknown choice or a sentinel value that can never match its column. |
| `invalid_call` | `InvalidArgumentError` | The call itself is malformed, such as an option passed by position, an unknown keyword or no DataFrame. |
| `unsupported_data_type` | `InvalidArgumentError` | `data` isn't a pandas DataFrame; the message shows the conversion. |
| `ambiguous_column_labels` | `InvalidArgumentError` | The column labels repeat, share a string form or form a MultiIndex; `describe_dataset` reports them. |
| `column_not_found` | `ColumnNotFoundError` | A column name isn't in the data; the message lists the closest and the available names. |
| `unsupported_column_type` | `ColumnTypeError` | A named column has a dtype the analysis can't use, or an `id_column` holds values that can't be hashed. |
| `unknown_analysis` | `UnknownAnalysisError` | `describe_analysis` got a name that isn't an analysis. |
| `invalid_project_file` | `ProjectFileError` | `Project.load` got a file that isn't valid TOML or lacks a key, or `save` got text TOML can't hold. |
| `unknown_project_key` | `ProjectFileError` | `Project.load` got a file with an unknown key, such as a misspelling. |
| `project_file_not_found` | `ProjectFileNotFoundError` | `Project.load` found no file at the path. |
| `project_file_exists` | `ProjectFileExistsError` | `project.save` found a file at the path and `overwrite` was `False`. |
| `missing_dependency` | `MissingDependencyError` | A feature needs an optional extra that isn't installed; the message gives the install command. |
| `unsupported_data_format` | `UnsupportedDataFormatError` | An MCP tool got a data path with an extension it can't read. |
| `data_file_not_found` | `DataFileNotFoundError` | An MCP tool found no file at its data path. |
| `unreadable_data_file` | `DataFileError` | An MCP tool can't parse the file at its data path, such as a CSV that isn't UTF-8. |
| `table_file_exists` | `TableFileExistsError` | An MCP tool's `table_path` leads to a file that exists already. |

### MCP server command

`agent-eda-mcp [--project PATH]`

Serves every analysis as an MCP tool of the same name over stdio, plus `search_analyses` and `list_analyses`. The MCP tools take `data_path` in place of `data`, `table_path` to write the full main table, and `project_path` in place of `project`. `--project` loads a project file for every call. [Use over MCP](#use-over-mcp) has the details.

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
