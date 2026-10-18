# agent-eda

Deterministic exploratory data analysis (EDA) for pandas DataFrames, built for LLM agents and usable by people.

> Status: version 0.1, in development and not yet on PyPI. It ships 11 first-look analyses, four discovery functions, an optional `Project` for masking sensitive columns and an optional MCP server. The [catalogue](docs/eda-python-implementation-catalogue.md) lists the analyses planned next.

Each analysis is a plain Python function. It takes a DataFrame and typed keyword arguments and returns a typed result that prints as a short text summary. The library never calls an LLM or the network, so the same data and arguments always give the same result, down to the byte.

It is written for three kinds of callers:

- agents in AWS Bedrock AgentCore, through its Code Interpreter
- coding agents such as Claude Code
- people, who use it like any other package

## Features

- **11 first-look analyses**: structure, column types, missing values, placeholder codes such as -999, distributions, duplicates, keys, the target and associations.
- **Results you can read and parse**: `print(res)` gives at most 40 lines of text, `res.to_frame()` the full tables and `res.to_json()` the whole record as strict JSON.
- **Findings, not just numbers**: each result lists what it noticed, such as "1 value in 1 column may stand in for missing values: 'age'".
- **Discovery for agents**: `ae.guide()`, `ae.search_analyses()` and `ae.describe_analysis()` explain the package from inside Python.
- **Helpful errors**: a misspelled column name raises an error that lists the available columns and the closest match.
- **Opt-in masking** of sensitive columns, such as e-mail addresses, in everything a result shows.
- **An MCP server**: `agent-eda-mcp` serves every analysis as an MCP tool over stdio, for agents that call tools instead of writing Python.
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

Every analysis that works column by column takes `columns=` to limit it to some columns; the default uses every column it can. `help(ae.<name>)` and `ae.describe_analysis("<name>")` show all the options.

Run all 11 on the sample data and print what they suspect:

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

Masking is a guardrail against accidental exposure, not anonymization or access control. The package runs in the agent's own Python process, so an agent that can call it can also read the data directly. [ADR 0001](docs/adr/0001-data-policy-is-a-guardrail.md) records why.

## Use over MCP

Agents that call tools instead of writing Python, such as Claude Desktop, Cursor or a Claude Code session without a shell, can reach the analyses through a local MCP server. It speaks stdio only. The MCP client starts `agent-eda-mcp` as a subprocess, and the MCP server opens no network connection. [ADR 0006](docs/adr/0006-local-mcp-server-over-stdio-data-by-path.md) records the design.

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

A call returns the printed result as text and the whole `to_dict()` record as structured content, with the same caps on long lists as in Python. The record equals the Python one. The text is worded for an MCP caller: where `print(res)` points to `res.to_frame()` and `res.to_json()`, it points to `table_path` and the structured content. Some MCP clients show the model only one of the two. Claude Code, for one, shows only the structured content. A run that doesn't complete, such as `insufficient_data` from `summarize_numeric` on a file with no rows, is a normal response with its status and reason. A mistake in the call comes back as a tool error (`isError: true`) whose structured content holds the stable error `code` and the `message` a Python caller gets.

To get past the caps, pass `table_path`. The MCP server writes the result's main table, the one `res.to_frame()` gives in Python without arguments, to a new `.csv`, `.parquet` or `.pq` file. The text ends with `Full table written to <absolute path> (<n> rows).`, and the structured content gains `table_file` with the same `path` and `row_count`.

- `table_path` never overwrites. An existing file gives the error `table_file_exists` before the analysis runs, and its message suggests a free name such as `table-2.csv`. The directory must exist already.
- A cell that holds a list or a mapping, such as the row IDs of a duplicate group, is written as JSON text. In Parquet, a column whose values mix types, such as numbers and text, is written as text, as in the CSV.
- An analysis with several tables writes only its main one. `compute_association_matrix` writes its pairs, for example, and its wide matrix stays a Python-only table.

### Masking over MCP

The five MCP tools that support masking take `project_path`, the path to a project file as `Project.save` writes it, in place of `project=`. They keep `mask_sensitive`. `agent-eda-mcp --project PATH` loads one project file at start and uses it for every call. A call's `project_path` replaces it for that call, and a call's `mask_sensitive` wins over both. Masking then covers the text, the structured content and the file `table_path` writes.

Masking over MCP is the same guardrail as in Python: it guards against accidental exposure and is not anonymization or access control. The MCP server enforces nothing, since whoever can call it can usually read the data file directly.

## Development

The project uses [uv](https://docs.astral.sh/uv/). Clone the repository, then install the package with its dev tools and the git hooks:

```sh
uv sync
uv run pre-commit install
```

| Task | Command |
| --- | --- |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run ty check` |
| Test | `uv run pytest` |
| Test on the minimum versions | `uv run --isolated --resolution lowest-direct --no-group mcp --group core --python 3.12 pytest` |
| Test on the minimum versions with the mcp extra | `uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 pytest` |
| Test the MCP server alone | `uv run --extra mcp pytest tests/test_mcp_*.py` |
| Smoke test the MCP server over stdio | `uv run --extra mcp python -I tests/smoke_mcp.py` |
| Test with more generated examples | `uv run pytest --hypothesis-profile=thorough` |
| Test with coverage | `uv run pytest --cov` |
| Benchmark | `uv run pytest -m benchmark` |
| Build the wheel | `uv build --wheel` |
| Run all hooks | `uv run pre-commit run --all-files` |

Notes on the less obvious commands:

- **Minimum versions.** The run installs every dependency at the lowest version `pyproject.toml` allows (numpy 1.26, pandas 2.2, pydantic 2.7, scipy 1.11.2, scikit-learn 1.5, statsmodels 0.14.1) into a throwaway environment on Python 3.12, without the mcp extra, whose tests skip. The second run adds the extra at its floors (mcp 2.3 and pyarrow 14.0.1), and with them the pydantic 2.12 that mcp needs, so it runs the whole suite. Keep `--isolated`; without it, uv rewrites `uv.lock`.
- **The mcp extra.** `uv sync` and `uv run` install it, so `uv run pytest` runs the MCP server tests. This needs uv 0.13 or later, the version CI and the hooks use; uv 0.10 leaves the extra out unless you pass `--extra mcp`. Without the extra, the MCP test modules skip and say why.
- **MCP smoke test.** `tests/smoke_mcp.py` starts `agent-eda-mcp` over real stdio, lists the MCP tools and calls `describe_dataset`. Unlike the MCP test modules, it fails rather than skips when the extra is missing.
- **The MCP server in Claude Code.** The repository's `.mcp.json` runs `uv run --extra mcp agent-eda-mcp`, so Claude Code started in a clone offers the analyses of your checkout as MCP tools once you approve the MCP server. Restart the session to pick up a changed analysis.
- **Property tests.** Hypothesis calls every analysis on generated DataFrames full of edge cases. It runs derandomized: 50 examples per analysis, or 1,000 with the `thorough` profile.
- **Benchmark.** It times every analysis with default arguments on a deterministic DataFrame of 1,000,000 rows and 100 mixed columns, with BLAS and pyarrow limited to 2 threads, and fails any analysis that takes 5 seconds or more. The default test run skips it.
- **README.** `tests/test_readme.py` runs this README's python examples in order and fails when a `text` block right after one differs from what it prints. It also checks the analyses table against `ae.guide()`.
- **New analyses** need a recipe in `tests/recipes.py`, or the suite fails and says so. [AGENTS.md](AGENTS.md) has the rules every analysis follows.

The pre-commit hooks run ruff, ty and file checks on every commit, and the tests on `git push`. On GitHub, `.github/workflows/ci.yml` runs lint, type checks and the tests, the MCP server tests included, on Python 3.12 to 3.14 on Linux x86_64 and ARM64. It also runs both minimum-versions runs, the MCP smoke test on Python 3.12 and 3.14, and a packaging job that installs the wheel offline with only the required dependencies. The benchmark runs only when started by hand from the Actions tab.

## Documentation

- [GLOSSARY.md](GLOSSARY.md): the terms used in code, docstrings and messages.
- [docs/adr/](docs/adr/): the decisions that are hard to reverse, and why.
- [docs/eda-python-implementation-catalogue.md](docs/eda-python-implementation-catalogue.md): the planned analyses, with the shipped ones marked and their differences from the plan.
- [AGENTS.md](AGENTS.md): the rules for anyone writing code here, people or agents.

## License

MIT. See [LICENSE](LICENSE).
