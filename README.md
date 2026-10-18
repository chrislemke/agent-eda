# agent-eda

Deterministic exploratory data analysis (EDA) for pandas DataFrames, built for LLM agents and usable by people.

> Status: version 0.1, in development and not yet on PyPI. It ships 11 first-look analyses, four discovery functions, an optional `Project` for masking sensitive columns and an optional MCP server. [Planned analyses](#planned-analyses) lists the ones that come next.

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

A call returns the printed result as text and the whole `to_dict()` record as structured content, with the same caps on long lists as in Python. The record equals the Python one. The text is worded for an MCP caller: where `print(res)` points to `res.to_frame()` and `res.to_json()`, it points to `table_path` and the structured content. Some MCP clients show the model only one of the two. Claude Code, for one, shows only the structured content. A run that doesn't complete, such as `insufficient_data` from `summarize_numeric` on a file with no rows, is a normal response with its status and reason. A mistake in the call comes back as a tool error (`isError: true`) whose structured content holds the stable error `code` and the `message`. The message says what is wrong and what is allowed as it does in Python, and gives the fix in MCP terms: `{"task": "regression"}` where Python shows `task='regression'`, the MCP tool's input schema where Python points to `help()`, and `project_path` where Python builds a `Project`.

To get past the caps, pass `table_path`. The MCP server writes the result's main table, the one `res.to_frame()` gives in Python without arguments, to a new `.csv`, `.parquet` or `.pq` file. The text ends with `Full table written to <absolute path> (<n> rows).`, and the structured content gains `table_file` with the same `path` and `row_count`.

- `table_path` never overwrites. An existing file gives the error `table_file_exists` before the analysis runs, and its message suggests a free name such as `table-2.csv`. The directory must exist already.
- A cell that holds a list or a mapping, such as the row IDs of a duplicate group, is written as JSON text. In Parquet, a column whose values mix types, such as numbers and text, is written as text, as in the CSV.
- An analysis with several tables writes only its main one. `compute_association_matrix` writes its pairs, for example, and its wide matrix stays a Python-only table.

### Masking over MCP

The five MCP tools that support masking take `project_path`, the path to a project file as `Project.save` writes it, in place of `project=`. They keep `mask_sensitive`. `agent-eda-mcp --project PATH` loads one project file at start and uses it for every call. A call's `project_path` replaces it for that call, and a call's `mask_sensitive` wins over both. Masking then covers the text, the structured content and the file `table_path` writes.

Masking over MCP is the same guardrail as in Python: it guards against accidental exposure and is not anonymization or access control. The MCP server enforces nothing, since whoever can call it can usually read the data file directly.

## Planned analyses

The 11 shipped analyses are the first of about 280 planned. The tables below list the rest by area, each with the `agent_eda` submodule it is meant for. Names and modules are proposals and can change while an analysis is built. Once one ships, its code and docstring are the source of truth, and it moves from these tables to [Analyses](#analyses).

Planned analyses follow the same rules as the shipped ones, which [AGENTS.md](AGENTS.md) lists. They take the DataFrame and typed keyword arguments, with no shared run context ([ADR 0004](docs/adr/0004-plain-functions-with-explicit-arguments.md)). They ask for project knowledge, such as units, keys, time cutoffs or dataset roles, rather than guess it. They report p-values and intervals only where the data design supports them, and say so otherwise.

<details>
<summary>Shared functions (8)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `load_dataset` | `io` | Load an approved local or S3 tabular source, select permitted columns and record the source and loading metadata. |
| `export_eda_json` | `io` | Write a validated UTF-8 JSON report and return its artifact reference and checksum. |
| `validate_project_config` | `config` | Validate and normalize project goals, target definition, metrics, restrictions, data roles and analysis settings; report missing or contradictory fields. |
| `check_analysis_requirements` | `discovery` | Check required data, metadata, permissions, dependencies, sample support and budget before running an analysis. |
| `run_standard_eda` | `execution` | Execute the configured standard set of analyses and record completed, skipped, inapplicable and failed analyses. |
| `run_analysis` | `execution` | Resolve named data bindings, validate prerequisites and execute one registered analysis with reproducibility metadata. |
| `validate_analysis_result` | `reporting` | Check the result schema, execution status, evidence, limitations and strict JSON serialization, including non-finite numbers. |
| `assemble_eda_report` | `reporting` | Combine results into a structured JSON report containing findings, uncertainty, limitations, coverage and unresolved questions. |

</details>

<details>
<summary>Data quality and structure (46)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `trace_data_lineage` | `data.provenance` | Trace supplied source, transformation and artifact relationships; identify missing links or inconsistent versions. |
| `describe_collection_process` | `data.provenance` | Summarize how, when and from whom data was collected, including instruments, inclusion rules and known collection changes. |
| `validate_data_contract` | `data.provenance` | Check required columns, schema, keys and declarative data rules against the declared contract. |
| `audit_data_permissions` | `data.provenance` | Check declared permissions for reading, analysis, model use, evaluation, retention and external processing. |
| `validate_measurement_units` | `data.validation` | Check supplied units, dimensional compatibility and configured conversion rules; flag missing unit evidence. |
| `validate_value_domains` | `data.validation` | Check numeric bounds, allowed categories, formats and inclusive or exclusive value constraints. |
| `validate_numeric_precision` | `data.validation` | Find unexpected decimal precision, quantization, overflow risks and precision loss relative to declared requirements. |
| `validate_datetime_values` | `data.validation` | Check date parsing, formats, time zones, daylight-saving ambiguities and allowed temporal bounds. |
| `validate_business_rules` | `data.validation` | Evaluate declared cross-field business rules and return violation counts, rates and permitted examples. |
| `profile_observation_units` | `data.structure` | Check the declared meaning and granularity of each row and summarize observation units and repeated measurements. |
| `profile_entities` | `data.structure` | Count entities, records per entity, repeated observations and optional observation spans. |
| `profile_table_relationships` | `data.structure` | Measure declared relationship cardinalities, referential coverage and unmatched keys between tables. |
| `test_functional_dependencies` | `data.structure` | Check declared determinant-to-dependent column relationships and quantify conflicting mappings. |
| `find_near_duplicate_records` | `data.duplicates` | Find similar tabular records using declared numeric or categorical matching rules; return scores and candidate groups. |
| `find_duplicate_entities` | `data.duplicates` | Find repeated entity records and compare them with declared rules for legitimate repeated observations. |
| `find_conflicting_records` | `data.duplicates` | Find records sharing a key but disagreeing on selected values beyond declared tolerances. |
| `assess_duplicate_impact` | `data.duplicates` | Quantify how duplicate groups affect effective row counts, target summaries and group representation. |
| `sample_analysis_data` | `data.sampling` | Create a reproducible random, stratified, group or time-based analysis sample and retain inclusion information and weights where justified. |
| `compare_sample_to_population` | `data.sampling` | Compare sample coverage and selected distributions against a reference population, using supplied weights. |
| `profile_sampling_design` | `data.sampling` | Summarize declared strata, clusters, stages, inclusion probabilities and sampling coverage. |
| `audit_sample_inclusion` | `data.sampling` | Audit which eligible records were included or excluded and quantify associated selection patterns. |
| `validate_sampling_weights` | `data.sampling` | Check missing, invalid or extreme weights and consistency with supplied inclusion probabilities and totals. |
| `estimate_effective_sample_size` | `data.sampling` | Estimate effective observations using an explicitly selected weighting, clustering or serial-dependence method. |
| `assess_group_representation` | `data.sampling` | Report group coverage and underrepresentation relative to supplied reference data or declared support thresholds. |
| `analyze_missingness_patterns` | `data.missingness` | Summarize joint missingness patterns, co-occurrence and concentrations of incomplete records. |
| `test_missingness_associations` | `data.missingness` | Test associations between missing-value indicators and target, group, time or other observed variables. |
| `fit_missingness_diagnostic_model` | `data.missingness` | Evaluate how predictably a field is missing using observed variables and valid diagnostic evaluation. |
| `analyze_multioutput_targets` | `data.targets` | Summarize joint target distributions, dependencies, co-occurrence and valid multi-output support. |
| `audit_label_consistency` | `data.targets` | Find contradictory labels for equivalent observations and violations of supplied label rules. |
| `audit_label_provenance` | `data.targets` | Check label sources, generation procedures, annotators, revisions and traceability to source evidence. |
| `analyze_label_latency` | `data.targets` | Measure delays between observation, outcome occurrence and the time a label became available. |
| `assess_outcome_maturity` | `data.targets` | Determine which observations have complete outcome follow-up at the supplied cutoff and summarize immature labels. |
| `estimate_repeatability` | `data.measurement` | Estimate consistency of repeated measurements on the same subjects under the declared measurement design. |
| `estimate_interrater_agreement` | `data.measurement` | Estimate categorical or ordinal agreement using the selected kappa or other supported agreement measure. |
| `estimate_intraclass_correlation` | `data.measurement` | Estimate the explicitly selected intraclass correlation form and its uncertainty. |
| `estimate_measurement_agreement` | `data.measurement` | Estimate measurement bias and agreement limits for paired instruments, accounting for declared repeated subjects. |
| `compare_measurement_sources` | `data.measurement` | Compare measurement distributions or bias across instruments, sources or collection processes. |
| `analyze_measurement_error_sensitivity` | `data.measurement` | Re-run a selected analysis under explicitly supplied measurement-error scenarios and compare the conclusions. |
| `audit_join_integrity` | `data.pipeline` | Check join cardinality, unmatched keys, duplicate expansion, changed row counts and introduced nulls. |
| `audit_row_conservation` | `data.pipeline` | Quantify retained, removed, added or multiplied observations and compare them with declared pipeline behavior. |
| `audit_introduced_missingness` | `data.pipeline` | Find newly missing values in aligned observations and distinguish expected changes from unexplained null introduction. |
| `compare_dataset_schemas` | `data.pipeline` | Report added, removed or changed columns and logical types against an approved reference schema. |
| `verify_artifact_versions` | `data.pipeline` | Compare supplied dataset, configuration, code, dependency and model artifact identities and checksums. |
| `verify_reproducibility` | `data.pipeline` | Compare recorded executions, seeds, environments, artifacts and numerical results under declared exact or tolerance-based criteria. |
| `audit_numeric_approximations` | `data.pipeline` | Record approximation settings and error guarantees and compare against exact references when available. |
| `assemble_execution_manifest` | `data.pipeline` | Assemble source, schema, configuration, code, dependency, seed, sampling, status and artifact metadata for the completed analysis run. |

</details>

<details>
<summary>Statistics (79)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `estimate_quantiles` | `statistics.descriptive` | Estimate requested quantiles with an explicit interpolation or approximation method and support counts. |
| `measure_concentration` | `statistics.descriptive` | Measure numerical or categorical concentration using a suitable declared measure such as Gini, HHI or entropy. |
| `estimate_empirical_distribution` | `statistics.distributions` | Return empirical cumulative probabilities and support as numerical values. |
| `compute_histogram` | `statistics.distributions` | Return bin edges, counts and optional weighted frequencies without rendering a plot. |
| `estimate_kernel_density` | `statistics.distributions` | Estimate density values on a supplied grid and report bandwidth and support limitations. |
| `detect_distribution_modes` | `statistics.distributions` | Identify evidence of one or multiple modes and report method and tuning sensitivity. |
| `fit_distribution_mixture` | `statistics.distributions` | Fit candidate mixture distributions and return component parameters, weights and fit diagnostics. |
| `analyze_distribution_boundaries` | `statistics.distributions` | Measure boundary mass, truncation indicators, floor or ceiling effects and values outside declared bounds. |
| `analyze_rounding_patterns` | `statistics.distributions` | Quantify rounding, heaping and quantization at specified numeric increments. |
| `fit_parametric_distribution` | `statistics.distributions` | Estimate parameters of a specified distribution and report support constraints and fitting diagnostics. |
| `compare_distribution_fits` | `statistics.distributions` | Compare compatible distribution fits using declared fit criteria and their uncertainty or limitations. |
| `test_normality` | `statistics.distributions` | Test normality using the selected procedure and report sample-size and assumption limitations. |
| `test_distribution_fit` | `statistics.distributions` | Run a goodness-of-fit test with suitable calibration, refitting estimated parameters within simulations when required. |
| `summarize_counts` | `statistics.counts` | Summarize count distributions, zeros and group support while checking count validity. |
| `estimate_event_rates` | `statistics.counts` | Estimate exposure-adjusted event rates and uncertainty by the requested groups. |
| `estimate_proportions` | `statistics.counts` | Estimate binomial or multinomial proportions with an explicitly selected interval method compatible with the declared outcome encoding. |
| `test_overdispersion` | `statistics.counts` | Test excess count variation relative to a specified reference model and its conditional mean structure. |
| `test_zero_inflation` | `statistics.counts` | Compare observed zeros with zeros expected under a specified count model. |
| `fit_diagnostic_count_model` | `statistics.counts` | Fit a diagnostic count model, including Poisson or negative binomial variants, and evaluate dispersion, zeros and residual behavior. |
| `extract_extreme_events` | `statistics.extremes` | Extract block maxima or peaks over threshold and report exceedance counts, exposure and declared declustering choices. |
| `fit_extreme_value_model` | `statistics.extremes` | Fit an extreme-value tail model such as GEV or generalized Pareto to extracted events. |
| `estimate_tail_risk` | `statistics.extremes` | Estimate exceedance probabilities or return levels with uncertainty and explicitly declared exposure units. |
| `analyze_extreme_threshold_sensitivity` | `statistics.extremes` | Compare tail parameters and risk estimates across candidate thresholds and report unstable or unsupported regions. |
| `compare_independent_means` | `statistics.comparisons` | Compare two independent-group means using conventional, Welch or robust trimmed-mean procedures. |
| `compare_independent_ranks` | `statistics.comparisons` | Compare two independent groups with Mann-Whitney or a specified rank procedure, recording ties and interpretation limits. |
| `compare_multiple_groups` | `statistics.comparisons` | Compare multiple independent groups using ANOVA, Welch ANOVA, Kruskal-Wallis or a declared permutation alternative. |
| `audit_repeated_measure_design` | `statistics.comparisons` | Check pairing, repeated-condition coverage, duplicate measurements and incomplete subjects before dependent comparisons. |
| `compare_paired_measurements` | `statistics.comparisons` | Compare two matched conditions using a paired t-test, Wilcoxon signed-rank or paired resampling. |
| `compare_repeated_measurements` | `statistics.comparisons` | Compare repeated conditions using repeated-measures ANOVA, Friedman or an appropriate subject-preserving alternative. |
| `build_contingency_table` | `statistics.comparisons` | Return observed and expected categorical cell counts, margins and sparse-cell warnings. |
| `test_categorical_independence` | `statistics.comparisons` | Test independence using chi-square, Fisher exact or a supported exact or resampling calibration for the table dimensions. |
| `compare_paired_categories` | `statistics.comparisons` | Compare paired categorical outcomes using McNemar or Cochran's Q with verified subject matching. |
| `test_stratified_categorical_association` | `statistics.comparisons` | Test categorical association across strata and assess effect homogeneity when supported by the selected procedure. |
| `compare_group_variances` | `statistics.comparisons` | Compare group variances using Levene, Fligner-Killeen or Bartlett and report assumptions. |
| `compare_covariance_matrices` | `statistics.comparisons` | Compare group covariance matrices using a declared parametric or calibrated resampling procedure. |
| `compare_multivariate_means` | `statistics.comparisons` | Compare multivariate group responses using Hotelling's T-squared, MANOVA or a specified compatible method. |
| `compare_multivariate_distances` | `statistics.comparisons` | Run a distance-based multivariate comparison with an appropriate exchangeability and resampling design. |
| `test_multivariate_dispersion` | `statistics.comparisons` | Test differences in within-group multivariate dispersion to support interpretation of distance-based comparisons. |
| `estimate_standardized_effect` | `statistics.effects` | Estimate a declared standardized effect such as Cohen's d or Hedges' g using the appropriate independent or paired design. |
| `estimate_rank_effect` | `statistics.effects` | Estimate a rank-based effect such as Cliff's delta or rank-biserial correlation with design-aware uncertainty. |
| `estimate_relative_risk` | `statistics.effects` | Estimate a risk ratio and uncertainty from explicitly oriented event and exposure counts. |
| `estimate_odds_ratio` | `statistics.effects` | Estimate an odds ratio, optionally across compatible strata, with declared cell orientation and small-count handling. |
| `test_equivalence` | `statistics.effects` | Test equivalence against explicit practical lower and upper margins using a compatible independent or paired procedure. |
| `estimate_correlation` | `statistics.association` | Estimate Pearson, Spearman, Kendall or point-biserial correlation with valid paired counts and supported uncertainty. |
| `estimate_categorical_association` | `statistics.association` | Estimate categorical association such as Cramer's V with contingency support and optional declared bias correction. |
| `estimate_correlation_ratio` | `statistics.association` | Measure numerical variation associated with category membership and report group support. |
| `estimate_ordinal_association` | `statistics.association` | Estimate ordinal association using declared category orders and tie handling. |
| `estimate_mutual_information` | `statistics.association` | Estimate mutual information for declared discrete or continuous variables and report estimator sensitivity and support. |
| `estimate_entropy` | `statistics.association` | Estimate discrete or differential entropy with declared units, assumptions and estimator settings. |
| `estimate_multivariate_information` | `statistics.association` | Estimate a specified multivariate information measure, such as total correlation, and record dimensionality limitations. |
| `estimate_nonlinear_dependence` | `statistics.association` | Estimate a nonlinear dependence score such as distance correlation or a kernel dependence statistic. |
| `test_nonlinear_independence` | `statistics.association` | Test nonlinear independence with a selected distance or kernel method and valid null calibration. |
| `estimate_partial_correlation` | `statistics.association` | Estimate correlation after accounting for explicitly selected conditioning variables. |
| `test_conditional_independence` | `statistics.association` | Test conditional independence using a method compatible with the declared data and conditioning structure. |
| `analyze_residualized_relationships` | `statistics.association` | Analyze relationships between residualized variables using an explicitly declared nuisance-fitting and evaluation design. |
| `fit_copula` | `statistics.association` | Fit a specified copula and marginals and return dependence parameters and fit diagnostics. |
| `estimate_tail_dependence` | `statistics.association` | Estimate upper or lower joint-tail dependence and report limited support at extreme quantiles. |
| `summarize_conditional_target` | `statistics.association` | Summarize target behavior within declared categories or numeric bins and return group support and uncertainty. |
| `estimate_interaction_effects` | `statistics.association` | Estimate specified feature interactions using a diagnostic model and valid evaluation. |
| `assess_subgroup_heterogeneity` | `statistics.association` | Compare a specified target statistic or association across subgroups with support and uncertainty. |
| `estimate_response_shape` | `statistics.association` | Estimate a numerical response shape using bins, smoothing or a valid model-based diagnostic; record extrapolation and confounding limits. |
| `detect_simpson_reversals` | `statistics.association` | Identify candidate reversals between aggregate and within-stratum associations with support and uncertainty. |
| `estimate_analytic_interval` | `statistics.inference` | Compute an analytic uncertainty interval using the declared reference distribution and any supported transformation. |
| `estimate_exact_interval` | `statistics.inference` | Compute a compatible exact interval for a specified count or proportion sampling model. |
| `bootstrap_statistic` | `statistics.inference` | Estimate a statistic's sampling distribution and interval using a bootstrap design that preserves declared dependence. |
| `jackknife_statistic` | `statistics.inference` | Estimate bias and uncertainty through declared leave-observation, leave-group or other supported jackknife deletions. |
| `run_permutation_test` | `statistics.inference` | Calibrate a test by exact or sampled permutations permitted under the declared null and exchangeability structure. |
| `run_monte_carlo_test` | `statistics.inference` | Calibrate a statistic against simulations from an explicitly specified null model and report simulation error. |
| `estimate_bayesian_interval` | `statistics.inference` | Estimate a credible interval from supplied posterior draws or a fully specified supported model, prior and observations. |
| `assess_resampling_precision` | `statistics.inference` | Assess Monte Carlo error and interval or p-value stability against declared precision targets. |
| `assess_rare_event_support` | `statistics.power` | Count rare events and non-events across requested groups and compare them with declared support requirements. |
| `assess_segment_sample_adequacy` | `statistics.power` | Assess segment-level observation, event and effective-sample support for explicitly selected analyses. |
| `estimate_statistical_power` | `statistics.power` | Estimate prospective power for an explicitly declared test, effect definition and independent or dependent study design. |
| `estimate_minimum_detectable_effect` | `statistics.power` | Estimate the smallest detectable effect for a declared design, significance level and desired power. |
| `estimate_required_sample_size` | `statistics.power` | Estimate required sample sizes for declared effect, allocation, variance and dependence assumptions. |
| `adjust_p_values` | `statistics.multiplicity` | Adjust p-values within explicitly supplied test families using a selected family-wise or false-discovery procedure. |
| `run_posthoc_comparisons` | `statistics.multiplicity` | Run compatible post-hoc group comparisons and retain raw or intrinsically adjusted results without applying duplicate corrections. |
| `annotate_adaptive_findings` | `statistics.multiplicity` | Label findings as prespecified, exploratory, adaptively selected or unknown using supplied analysis history. |
| `audit_multiple_testing` | `statistics.multiplicity` | Audit test-family membership, correction coverage, thresholds and adaptive-selection limitations. |

</details>

<details>
<summary>Features (15)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `find_duplicate_features` | `features.redundancy` | Find identical or tolerance-equivalent feature columns and return duplicate groups. |
| `analyze_matrix_conditioning` | `features.redundancy` | Compute matrix rank, singular values, eigenvalue summaries and condition numbers under declared scaling. |
| `estimate_variance_inflation` | `features.redundancy` | Estimate variance inflation factors and flag singular or unsupported feature combinations. |
| `estimate_covariance` | `features.redundancy` | Estimate ordinary, robust or regularized covariance with conditioning and sample-support diagnostics. |
| `find_redundant_feature_groups` | `features.redundancy` | Group strongly redundant features from a compatible association result using declared thresholds and grouping rules. |
| `decompose_numeric_structure` | `features.structure` | Fit PCA, kernel PCA, factor analysis, ICA or NMF and return component, loading and fit summaries. |
| `decompose_categorical_structure` | `features.structure` | Run correspondence or multiple-correspondence analysis and return category coordinates and inertia summaries. |
| `decompose_mixed_structure` | `features.structure` | Analyze mixed numerical and categorical factor structure using a suitable declared method. |
| `evaluate_structure_reconstruction` | `features.structure` | Measure reconstruction error, explained variation or a method-appropriate numerical fidelity measure. |
| `compute_pairwise_distances` | `features.structure` | Compute requested pairwise distances or their summaries under explicit scaling and output-size limits. |
| `analyze_neighborhood_structure` | `features.structure` | Summarize nearest-neighbor distances, connectivity, hubness and local support across selected neighborhood sizes. |
| `estimate_intrinsic_dimension` | `features.structure` | Estimate intrinsic dimensionality and report neighborhood, sampling and estimator sensitivity. |
| `measure_distance_concentration` | `features.structure` | Quantify how pairwise distances concentrate and how well nearest and farthest observations separate. |
| `compute_manifold_embedding` | `features.structure` | Compute a selected manifold embedding and return numerical coordinates or an artifact reference with fit settings. |
| `evaluate_embedding_quality` | `features.structure` | Measure neighborhood preservation, trustworthiness, continuity or stress without rendering a plot. |

</details>

<details>
<summary>Patterns (17)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `cluster_numeric_data` | `patterns.clustering` | Cluster numeric observations using k-means, hierarchical, Gaussian-mixture, DBSCAN or HDBSCAN methods. |
| `cluster_categorical_data` | `patterns.clustering` | Cluster categorical observations using a compatible matching distance or categorical clustering algorithm. |
| `cluster_mixed_data` | `patterns.clustering` | Cluster mixed-type observations using declared scaling, categorical handling and a compatible algorithm. |
| `summarize_clusters` | `patterns.clustering` | Return cluster sizes, feature profiles, mixture membership summaries and noise or unassigned counts. |
| `evaluate_cluster_separation` | `patterns.clustering` | Compute silhouette, Davies-Bouldin or other requested cluster-separation measures with valid support. |
| `estimate_gap_statistic` | `patterns.clustering` | Compare within-cluster dispersion with a declared reference distribution across candidate cluster counts. |
| `assess_cluster_stability` | `patterns.clustering` | Assess cluster membership stability under resampling and optional parameter changes, accounting for label permutations and noise. |
| `compare_clusterings` | `patterns.clustering` | Measure agreement between aligned clusterings or between clusters and known segments using a selected compatible metric. |
| `detect_univariate_anomalies` | `patterns.anomalies` | Score univariate extremes using IQR or robust-score rules and return declared threshold exceedances. |
| `detect_multivariate_anomalies` | `patterns.anomalies` | Score multivariate anomalies using robust Mahalanobis distance, Isolation Forest or a compatible selected method. |
| `detect_local_anomalies` | `patterns.anomalies` | Score local anomalies using local outlier factors or nearest-neighbor distances with explicit threshold choices. |
| `detect_contextual_anomalies` | `patterns.anomalies` | Find unusual values relative to declared groups, time windows or other contextual conditions. |
| `compare_anomaly_results` | `patterns.anomalies` | Compare anomaly-score rankings and flagged-observation overlap across methods on aligned records. |
| `summarize_anomalies` | `patterns.anomalies` | Summarize anomaly counts, scores, affected groups and permitted examples without equating rare cases with data errors. |
| `mine_frequent_itemsets` | `patterns.rules` | Find frequently co-occurring categorical conditions with support counts and search-limit metadata. |
| `mine_association_rules` | `patterns.rules` | Generate association rules and return antecedents, consequents, support, confidence and lift. |
| `discover_unusual_subgroups` | `patterns.rules` | Search bounded combinations of categorical conditions for unusual target behavior and label findings as exploratory. |

</details>

<details>
<summary>Diagnostics (9)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `fit_diagnostic_model` | `diagnostics.models` | Fit a budget-limited linear, GLM, additive or tree diagnostic under the declared evaluation design and return an artifact reference. |
| `evaluate_diagnostic_model` | `diagnostics.models` | Evaluate a diagnostic fit on permitted evaluation observations using declared metrics and dependence-aware uncertainty. |
| `analyze_model_residuals` | `diagnostics.models` | Analyze suitable out-of-sample residuals for bias, nonlinearity, variance patterns, dependence and influential observations. |
| `estimate_feature_importance` | `diagnostics.models` | Estimate model-specific or evaluation-based importance and document bias, correlation and evaluation limitations. |
| `estimate_conditional_feature_importance` | `diagnostics.models` | Estimate feature or feature-group importance while explicitly accounting for correlated predictors through a declared conditional method. |
| `assess_feature_selection_stability` | `diagnostics.models` | Measure selection frequencies and feature-set agreement across valid resamples without using a sealed holdout. |
| `run_sensitivity_analysis` | `diagnostics.sensitivity` | Re-run a selected analysis under declared sample, seed, threshold, time-window, missingness or transformation scenarios. |
| `analyze_influential_observations` | `diagnostics.sensitivity` | Measure how observations or declared groups influence a statistic, diagnostic model or conclusion. |
| `summarize_result_stability` | `diagnostics.sensitivity` | Summarize changes in estimates, uncertainty, rankings and conclusions across compatible sensitivity results. |

</details>

<details>
<summary>Time and sequences (32)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `validate_time_index` | `temporal.integrity` | Check ordering, valid timestamps, duplicate time keys and time-zone consistency within declared entities. |
| `analyze_observation_intervals` | `temporal.integrity` | Summarize inter-observation intervals, irregular spacing and interval changes by entity. |
| `find_time_gaps` | `temporal.integrity` | Find missing expected timestamps or extended gaps while respecting the supplied observation calendar. |
| `assess_temporal_coverage` | `temporal.integrity` | Measure observation coverage across the declared time range, periods and entities. |
| `assess_data_freshness` | `temporal.integrity` | Measure data age, ingestion lag and freshness violations relative to an explicit reference time. |
| `validate_observation_windows` | `temporal.integrity` | Check window ordering, duration, overlap and cross-field consistency against declared rules. |
| `estimate_time_trend` | `temporal.analysis` | Estimate a declared ordinary or robust temporal trend with uncertainty and sampling assumptions. |
| `decompose_time_series` | `temporal.analysis` | Return numerical trend, seasonal and residual components using STL or MSTL after validating sampling requirements. |
| `estimate_power_spectrum` | `temporal.analysis` | Estimate spectral power using an appropriate periodogram, spectral or Lomb-Scargle procedure for the observation schedule. |
| `measure_seasonality` | `temporal.analysis` | Quantify seasonal strength and repeatability at declared periods with coverage and detrending information. |
| `detect_periodicities` | `temporal.analysis` | Identify supported spectral peaks and candidate periods while recording resolution, aliasing and selection limitations. |
| `estimate_autocorrelation` | `temporal.analysis` | Estimate ACF or PACF at supported lags and report intervals and sampling assumptions. |
| `estimate_cross_correlation` | `temporal.analysis` | Estimate cross-correlation with an explicit lag-sign convention and temporal alignment. |
| `analyze_lagged_associations` | `temporal.analysis` | Measure lagged predictor-target associations without crossing declared entity boundaries. |
| `test_sequence_randomness` | `temporal.analysis` | Run a supported sequence-randomness or runs test with declared ordering and tie treatment. |
| `test_serial_correlation` | `temporal.analysis` | Test serial correlation, including Ljung-Box diagnostics with appropriate fitted-model degrees-of-freedom adjustments. |
| `test_stationarity` | `temporal.analysis` | Run ADF or KPSS with explicit deterministic terms and lag settings and state the selected test's null hypothesis. |
| `test_structural_breaks` | `temporal.analysis` | Test specified or unknown structural breaks using a suitable declared procedure, including CUSUM where applicable. |
| `detect_change_points` | `temporal.analysis` | Detect candidate changes in mean, variance or dependence under explicit segmentation and penalty rules. |
| `summarize_time_regimes` | `temporal.analysis` | Summarize observation support and distribution or dependence differences across detected time regimes. |
| `decompose_within_between_variation` | `temporal.longitudinal` | Separate within-group and between-group variation and report group sizes and imbalance. |
| `estimate_variance_components` | `temporal.longitudinal` | Estimate declared hierarchical variance components and record identifiability and boundary-fit limitations. |
| `fit_mixed_effects_diagnostic_model` | `temporal.longitudinal` | Fit a diagnostic mixed-effects model with declared group structure and valid evaluation. |
| `fit_gee_diagnostic_model` | `temporal.longitudinal` | Fit a diagnostic generalized estimating equation model with declared clusters, link family and working dependence. |
| `check_longitudinal_model_assumptions` | `temporal.longitudinal` | Check longitudinal residual patterns, covariance choices, convergence and group-level support for a fitted diagnostic model. |
| `analyze_interarrival_times` | `temporal.events` | Summarize time between successive events within entities and optional event types. |
| `analyze_state_dwell_times` | `temporal.events` | Measure durations spent in states and distinguish completed dwell periods from censored final states. |
| `estimate_state_transitions` | `temporal.events` | Return state-transition counts and probabilities at a declared event lag. |
| `validate_event_sequences` | `temporal.events` | Find invalid event ordering, missing prerequisites and prohibited transitions under supplied sequence rules. |
| `mine_event_sequences` | `temporal.events` | Find recurring bounded event sequences and return occurrence support and search constraints. |
| `analyze_cohort_progression` | `temporal.events` | Summarize cohort progression through declared stages and relative time periods. |
| `estimate_retention` | `temporal.events` | Estimate explicitly defined cohort retention while accounting for cohort maturity and observation coverage. |

</details>

<details>
<summary>Specialized data (39)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `profile_followup` | `specialized.survival` | Summarize follow-up duration, events, observations at risk and coverage by group. |
| `audit_censoring_and_truncation` | `specialized.survival` | Check censoring and entry-time encoding and quantify observed censoring or truncation patterns without proving non-informative censoring. |
| `estimate_survival_curve` | `specialized.survival` | Estimate supported survival probabilities, intervals and at-risk counts at requested times. |
| `estimate_cumulative_hazard` | `specialized.survival` | Estimate cumulative hazards and uncertainty using a method compatible with the supplied survival design. |
| `compare_survival_groups` | `specialized.survival` | Compare group survival using a declared log-rank or compatible stratified procedure. |
| `fit_survival_diagnostic_model` | `specialized.survival` | Fit and evaluate a suitable diagnostic survival model while preserving censoring and entry-time semantics. |
| `check_survival_model_assumptions` | `specialized.survival` | Check supported model assumptions such as proportional hazards and report residual, convergence and follow-up limitations. |
| `validate_spatial_coordinates` | `specialized.spatial` | Check coordinate ranges, missing values, declared reference systems and expected spatial bounds. |
| `summarize_spatial_coverage` | `specialized.spatial` | Summarize spatial extent, occupied regions, density and coverage gaps using declared region or grid definitions. |
| `compute_spatial_distances` | `specialized.spatial` | Compute projected or geodesic distances with explicit units and bounded numerical output. |
| `build_spatial_weights` | `specialized.spatial` | Construct a reproducible spatial-neighbor or distance-weight definition and report isolated observations. |
| `test_global_spatial_autocorrelation` | `specialized.spatial` | Estimate and test global Moran's I or Geary's C under the declared spatial weights and null model. |
| `test_local_spatial_autocorrelation` | `specialized.spatial` | Return local spatial-association statistics and raw test results for subsequent multiple-testing control. |
| `detect_spatial_hotspots` | `specialized.spatial` | Identify candidate high- or low-value spatial concentrations using a declared hotspot statistic and calibrated uncertainty. |
| `assess_spatial_split_dependence` | `specialized.spatial` | Measure cross-split spatial proximity and shared entities that may compromise spatially independent evaluation. |
| `validate_composition` | `specialized.compositional` | Check non-negativity, sum constraints, missing components and zero counts for compositional observations. |
| `analyze_log_ratios` | `specialized.compositional` | Return declared CLR, ALR or ILR log-ratio summaries with zero treatment and basis or reference metadata. |
| `compute_compositional_distances` | `specialized.compositional` | Compute Aitchison distances or their summaries after explicit compositional zero handling. |
| `analyze_compositional_zero_sensitivity` | `specialized.compositional` | Compare selected compositional findings across declared zero-replacement or zero-exclusion scenarios. |
| `validate_cyclic_measurements` | `specialized.directional` | Check declared cyclic units, periods, ranges and invalid or ambiguous angle encodings. |
| `summarize_circular_data` | `specialized.directional` | Estimate circular mean, dispersion and concentration and explain undefined summaries when directions cancel. |
| `summarize_directional_data` | `specialized.directional` | Summarize normalized multidimensional directions, mean direction and directional concentration under declared coordinates. |
| `test_directional_structure` | `specialized.directional` | Test a specified circular or directional structure, such as uniformity or concentration around a supplied direction. |
| `build_entity_graph` | `specialized.graphs` | Build a declared entity graph and record node identity, edge aggregation, direction, weights and isolated-node handling. |
| `summarize_graph_structure` | `specialized.graphs` | Summarize node and edge counts, degree distributions, density, self-loops and connectedness. |
| `find_connected_components` | `specialized.graphs` | Identify connected, weakly connected or strongly connected components as appropriate to the graph. |
| `estimate_graph_centrality` | `specialized.graphs` | Estimate selected centrality measures and document weighting, disconnectedness and approximation choices. |
| `detect_graph_communities` | `specialized.graphs` | Identify communities using a method compatible with the graph and return membership, support and quality summaries. |
| `estimate_graph_assortativity` | `specialized.graphs` | Measure degree or attribute assortativity under the declared graph and edge-weight semantics. |
| `analyze_relational_dependence` | `specialized.graphs` | Measure attribute dependence across related entities and optional graph connections between evaluation roles. |
| `validate_functional_observations` | `specialized.functional` | Check curve identities, domains, sampling grids, coverage, missing values and declared alignment requirements. |
| `summarize_functional_data` | `specialized.functional` | Return numerical mean or robust curves, variation summaries and support across a declared functional domain. |
| `align_functional_data` | `specialized.functional` | Align repeated curves under explicit landmark or registration constraints and record the transformations. |
| `decompose_functional_variation` | `specialized.functional` | Estimate functional principal components or another selected functional decomposition and summarize explained variation. |
| `compare_functional_shapes` | `specialized.functional` | Compare group curve shapes or trajectories with a compatible functional statistic and dependence-aware uncertainty. |
| `compute_persistent_homology` | `specialized.topology` | Compute persistent topological features from a point cloud, distance matrix or compatible supplied complex. |
| `summarize_topological_features` | `specialized.topology` | Return requested numerical persistence summaries, including connectivity, loops and higher-dimensional feature lifetimes. |
| `compare_topological_signatures` | `specialized.topology` | Measure differences between compatible persistence diagrams or topological summaries. |
| `assess_topological_stability` | `specialized.topology` | Assess how topological features change under declared sampling, perturbation or filtration scenarios. |

</details>

<details>
<summary>Evaluation (23)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `compare_univariate_distributions` | `evaluation.shift` | Compare selected marginal distributions using compatible numerical distances or two-sample tests and report support. |
| `compare_multivariate_distributions` | `evaluation.shift` | Measure joint distribution differences using a declared distance, energy or kernel two-sample procedure. |
| `fit_adversarial_validation_model` | `evaluation.shift` | Evaluate how well a diagnostic classifier distinguishes dataset membership under leakage-safe evaluation. |
| `localize_distribution_shift` | `evaluation.shift` | Locate observed shift by feature, group or time period without inferring a causal shift mechanism from marginal differences alone. |
| `profile_sensitive_groups` | `evaluation.fairness` | Summarize permitted sensitive-group definitions, support, missing membership and weighted representation. |
| `assess_intersectional_coverage` | `evaluation.fairness` | Measure intersectional group support and identify sparse or absent intersections within explicit search limits. |
| `compare_group_label_distributions` | `evaluation.fairness` | Compare observed label rates or target distributions across groups with uncertainty and small-group limitations. |
| `compare_group_data_quality` | `evaluation.fairness` | Compare missingness, validity and other specified data-quality indicators across groups. |
| `measure_proxy_associations` | `evaluation.fairness` | Measure associations between permitted sensitive attributes and potential proxy features without treating association as proof of unfairness. |
| `estimate_selection_propensity` | `evaluation.confounding` | Estimate observed inclusion propensity from measured covariates and document unavailable selection causes. |
| `assess_covariate_balance` | `evaluation.confounding` | Compare covariate balance between declared groups using standardized or distribution-based diagnostics. |
| `assess_population_overlap` | `evaluation.confounding` | Assess observed common support and near-positivity violations under a declared overlap diagnostic. |
| `evaluate_negative_controls` | `evaluation.confounding` | Evaluate supplied negative-control relationships and record assumptions needed to interpret their results. |
| `check_supplied_causal_assumptions` | `evaluation.confounding` | Evaluate testable implications of supplied causal assumptions and explicitly list assumptions that observed data cannot verify. |
| `analyze_confounding_sensitivity` | `evaluation.confounding` | Quantify sensitivity of a supplied estimand to explicitly parameterized unmeasured confounding assumptions. |
| `audit_prediction_time_availability` | `evaluation.leakage` | Check whether selected feature values were available at prediction time using supplied availability, update and lineage evidence. |
| `detect_target_derived_features` | `evaluation.leakage` | Identify direct or indirect target-derived features from explicit lineage and transformation evidence. |
| `detect_suspicious_target_proxies` | `evaluation.leakage` | Flag unusually predictive or target-like features for investigation while keeping statistical suspicion separate from confirmed leakage. |
| `measure_split_overlap` | `evaluation.leakage` | Measure exact-record, entity and configured near-duplicate overlap between permitted evaluation splits. |
| `audit_temporal_split_boundaries` | `evaluation.leakage` | Check chronological separation, purge or embargo rules and overlapping label or feature windows across splits. |
| `audit_preprocessing_fit_scope` | `evaluation.leakage` | Check which data roles fitted preprocessing, selection and other learned transformations using recorded execution evidence. |
| `audit_holdout_access` | `evaluation.leakage` | Audit recorded final-holdout reads and uses against the declared holdout policy and mark incomplete access evidence. |
| `evaluate_leakage_evidence` | `evaluation.leakage` | Combine leakage findings into confirmed violations, supported suspicions and unevaluated checks without converting missing evidence into a pass. |

</details>

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
- **README.** `tests/test_readme.py` runs this README's python examples in order and fails when a `text` block right after one differs from what it prints. It also checks the analyses table against `ae.guide()`, and that no shipped analysis is still listed under Planned analyses.
- **New analyses** need a recipe in `tests/recipes.py`, or the suite fails and says so. A shipped analysis moves from Planned analyses to the Analyses table. [AGENTS.md](AGENTS.md) has the rules every analysis follows.

The pre-commit hooks run ruff, ty and file checks on every commit, and the tests on `git push`. On GitHub, `.github/workflows/ci.yml` runs lint, type checks and the tests, the MCP server tests included, on Python 3.12 to 3.14 on Linux x86_64 and ARM64. It also runs both minimum-versions runs, the MCP smoke test on Python 3.12 and 3.14, and a packaging job that installs the wheel offline with only the required dependencies. The benchmark runs only when started by hand from the Actions tab.

## Documentation

- [GLOSSARY.md](GLOSSARY.md): the terms used in code, docstrings and messages.
- [docs/adr/](docs/adr/): the decisions that are hard to reverse, and why.
- [AGENTS.md](AGENTS.md): the rules for anyone writing code here, people or agents.

## License

MIT. See [LICENSE](LICENSE).
