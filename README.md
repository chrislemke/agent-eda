# agent-eda

Deterministic exploratory data analysis (EDA) functions for LLM agents.

> Status: version 0.1, in development and not yet on PyPI. It contains the 11 first-look analyses in the table under [A first look at a new dataset](#a-first-look-at-a-new-dataset) and four discovery functions: `guide`, `search_analyses`, `list_analyses` and `describe_analysis`. An optional `Project` declares sensitive columns whose values the analyses can mask, and saves to a TOML file. The [catalogue](docs/eda-python-implementation-catalogue.md) lists the analyses planned after these.

agent-eda is a Python library for the EDA phase of a data science project: data quality, missing values, duplicates, distributions, associations, leakage checks and more. Each analysis is a plain function. It takes a pandas DataFrame and typed arguments and returns a typed result. The library never calls an LLM or the network, so the same data, arguments and seed always give the same result.

Three kinds of callers matter equally:

- agents running in AWS Bedrock AgentCore that use its Code Interpreter
- coding agents such as Claude Code
- people, who install and import it like any other package

## Usage

Call an analysis with the DataFrame first and every other argument by keyword. There is nothing to set up, and the analysis leaves the DataFrame unchanged.

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
print(res)  # short text: status, key numbers, findings, limitations
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

The printed text stays within 40 lines of 120 characters however large the data, and counts what it leaves out. The result holds the rest:

```python
table = res.to_frame()  # every statistic of every column, as a DataFrame
record = res.to_json()  # the whole result as strict JSON
age = res.columns["age"]  # typed fields, such as age.median and age.std
```

A mistake in a call raises an error from `ae.errors` whose message says how to fix it, such as the closest name for a misspelled column.

### Finding an analysis

```python
ae.guide()  # one page: how to call and read analyses, every analysis, the first-look order
print(ae.search_analyses("nulls per column"))  # analyses for a question in your words
print(ae.describe_analysis("profile_missing_values"))  # its parameters and an example
```

### A first look at a new dataset

`ae.guide()` suggests this order. It covers structure, logical types, missing values and their stand-ins, distributions, duplicates and keys, the target and associations:

| Step | Analysis | The question it answers |
| --- | --- | --- |
| 1 | `describe_dataset` | What does this dataset contain? |
| 2 | `infer_column_semantics` | What kind of values does each column hold? |
| 2 | `validate_column_types` | Do the columns hold the logical types I declared? |
| 3 | `profile_missing_values` | How much data is missing, and where? |
| 4 | `detect_missing_sentinels` | Which values look like stand-ins for missing data? |
| 5 | `summarize_numeric` | How are the values in each numeric column distributed? |
| 5 | `summarize_categorical` | Which values occur in each categorical column, and how often? |
| 6 | `find_duplicate_rows` | Are there exact duplicate rows? |
| 6 | `find_candidate_keys` | Which columns, alone or combined, uniquely identify rows? |
| 7 | `profile_target` | What does the target look like for this task? |
| 8 | `compute_association_matrix` | How strongly are pairs of columns associated? |

```python
schema = {"customer_id": "integer", "age": "integer", "plan": "categorical"}
for res in [
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
]:
    print(res)
```

On this data they flag three problems. The -999 in `age` likely stands in for a missing age, and it pulled the mean above down to -86.12. `plan` holds `'Basic '` next to `'basic'`. Customers 102 and 108 are the same record twice. The analyses never guess a project-specific value. `validate_column_types` needs your schema, and `profile_target` needs the target column and its task.

### Masking sensitive columns

Some analyses can show raw values, such as the labels `summarize_categorical` counts or the row IDs `find_duplicate_rows` lists. These take `project=` and `mask_sensitive=`, and `ae.guide()` marks them `[supports masking]`. A `Project`'s data policy names the sensitive columns, and masking replaces their values with numbered placeholders. Counts and shares stay:

```python
policy = ae.DataPolicy(sensitive_columns=["email"], mask_sensitive=True)
project = ae.Project(data_policy=policy)
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

Masking covers the printed text, `to_dict()`, `to_json()` and `to_frame()` alike. It stays off until you switch it on, even for declared sensitive columns. `mask_sensitive=True` in the data policy switches it on for every call that gets the project, and `mask_sensitive=True` or `False` in a call decides for that call. Masking never touches numeric statistics, such as a minimum, or index labels, so pass a non-sensitive `id_column` when the index holds sensitive values.

An agent that runs each script in a new process can save the project once and load it in every later script:

```python
project.save("project.toml")
project = ae.Project.load("project.toml")
```

Masking is a guardrail against accidental exposure, not anonymization or access control. The package runs in the agent's own Python process, so an agent that can call it can also read the data directly. [ADR 0001](docs/adr/0001-data-policy-is-a-guardrail.md) records why.

## Installation

Nothing is on PyPI yet. Until then, build the wheel in a clone of this repository with `uv build --wheel`. It lands in `dist/`, ready for `pip install` or for uploading to a Code Interpreter session. Once it is released:

**People and coding agents**

```sh
pip install agent-eda
```

**AgentCore Runtime.** Add `agent-eda` to the agent's container image or code zip. AWS's example zip command uses the `manylinux2014` platform tag, which installs old numpy and pandas releases. Use `--python-platform aarch64-manylinux_2_28` to get current ones.

**AgentCore Code Interpreter.** In public network mode, run `pip install agent-eda` in the session. In sandbox mode the session can't reach PyPI. Upload the wheel and run `pip install --no-deps <wheel>`, or, if pip is blocked, put the wheel file on `sys.path` and import from it directly. Installs don't carry over to a new session. Code Interpreter only returns printed output to the agent, so `print()` results.

## Design documents

- [GLOSSARY.md](GLOSSARY.md) defines the terms used in code and docs.
- [docs/adr/](docs/adr/) records the decisions that are hard to reverse, and why.
- [docs/eda-python-implementation-catalogue.md](docs/eda-python-implementation-catalogue.md) lists the planned analyses and marks the ones that shipped, with how they differ from the plan.
- [AGENTS.md](AGENTS.md) holds the rules for anyone writing code here, people or agents.

## Development

Install the project with its dev tools:

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
| Test on the minimum versions | `uv run --isolated --resolution lowest-direct --python 3.12 pytest` |
| Test with more generated examples | `uv run pytest --hypothesis-profile=thorough` |
| Test with coverage | `uv run pytest --cov` |
| Benchmark | `uv run pytest -m benchmark` |
| Run all hooks | `uv run pre-commit run --all-files` |

The minimum-versions run installs every direct dependency at the lowest version `pyproject.toml` allows (numpy 1.26, pandas 2.2, pydantic 2.7, scipy 1.11.2, scikit-learn 1.5, statsmodels 0.14.1, and the dev tools' lower bounds) into a throwaway environment on Python 3.12, the newest Python with numpy 1.26 wheels. It leaves `uv.lock` and `.venv` unchanged. Keep `--isolated`: without it, uv rewrites `uv.lock` for the new resolution mode.

The packaging tests build the wheel offline with the `uv_build` backend from the dev dependencies, pinned to the range in `[build-system]`, so they need no network and work with any uv version. When `AGENT_EDA_WHEEL` names a wheel file, they check that file instead.

The property tests call every analysis in `ae.list_analyses()` on DataFrames that Hypothesis generates full of edge cases, and on fixed ones such as a column of every dtype. Hypothesis runs derandomized, so every run tests the same 50 examples per analysis, or 1,000 with the `thorough` profile. The benchmark times every analysis with its default arguments on a deterministic DataFrame of 1,000,000 rows and 100 mixed columns, fails one that takes 5 seconds or more, and prints the timings at the end. It limits NumPy's BLAS and pyarrow to 2 threads, like the 2-vCPU AgentCore session it targets. The default test run skips it. Both call each analysis through its recipe in `tests/recipes.py`, so a new analysis needs one there, or the suite fails and says so.

`tests/test_readme.py` runs the README's python examples in order and fails when a `text` block right after one differs from what it prints. It also checks that the first-look table follows `ae.guide()`, with each analysis's own question.

On every commit, pre-commit checks the files (whitespace, line endings, YAML/TOML/JSON syntax, merge conflict markers, private keys, files over 1 MB), keeps `uv.lock` in sync, and runs ruff and ty. The tests run on `git push`.

On GitHub, `.github/workflows/ci.yml` runs on every push and pull request. One job runs ruff and ty. The tests run with the versions in `uv.lock` on Python 3.12, 3.13 and 3.14 on Linux x86_64, and on Python 3.12 on Linux ARM64. Another job runs the minimum-versions command from the table, and `tests/test_ci_workflow.py` fails if the two differ. The packaging job builds the wheel and runs the packaging tests on it. It then installs the wheel with `pip install --no-deps` into an environment that holds only the six required dependencies at their minimum versions, as in an offline AgentCore sandbox, and runs `tests/smoke.py` there. That script prints a result and runs every analysis's docstring examples. The benchmark runs only when started by hand from the Actions tab. It runs on Linux ARM64, limited to 2 CPUs with `taskset`, once with the latest and once with the minimum versions.

## License

MIT. See [LICENSE](LICENSE).
