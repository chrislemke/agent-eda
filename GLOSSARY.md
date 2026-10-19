# EDA for LLM agents

An open-source Python package of deterministic exploratory data analysis (EDA) functions, written for LLM agents to call during the EDA phase of a data science project.

## Language

### Callers

**Agent**:
An LLM-driven program that uses the package during a data science project, usually by writing and running Python that calls its functions.
_Avoid_: bot, assistant, LLM (when you mean the caller)

### Analyses and their results

**Analysis**:
A public function that answers one EDA question about data and returns an analysis result.
_Avoid_: capability, tool, check (as a name for the function)

**Method**:
A variant of an analysis that the caller chooses, such as Welch's versus Student's t-test.
_Avoid_: variant, kind, algorithm

**Analysis result**:
The record an analysis returns: its status, the computed values, findings, limitations and how it was run.
_Avoid_: output, report, response

**Status**:
How an analysis run ended, for example completed or not applicable. It says nothing about whether the data passed a check.
_Avoid_: outcome, result state

**Status reason**:
Why a run ended with a status other than completed: a stable code from the `StatusReasonCode` list and a one-sentence message. Nothing to work on is `no_rows`, `no_columns` or `no_values`, and the message says which. Each analysis names the codes it returns in its `Statuses:` section.
_Avoid_: error code (that names a raised error), failure reason

**Finding**:
A claim in an analysis result, backed by the result fields that support it and carrying an assessment.
_Avoid_: insight, issue, alert

**Assessment**:
The verdict on a finding: descriptive, suspected, violation, passed or unknown.
_Avoid_: severity, verdict, status (when you mean the finding's verdict)

**Support**:
The counts behind a result: how many rows or values were given, eligible, evaluated and excluded, with the reasons for exclusions.
_Avoid_: sample size, n (when you mean these counts)

**Row ID**:
The label a result uses to point at a row: the DataFrame index label, or the value in a declared ID column.
_Avoid_: observation ID, example ID, row number

**Limitation**:
Something that limits how far a result can be read, such as a check that didn't run or a field that is null. It has a stable code, a one-sentence reason and the paths of the result fields it affects. A finding makes a claim about the data; a limitation says what the result can't tell.
_Avoid_: warning, caveat, note, error (a limitation is part of a result, not a raised error)

**First look**:
The order in which a new dataset is examined, from its structure through its types, missing values, sentinel values, distributions, duplicates and keys, and target to the associations. `guide()` gives it; nothing runs it as one call.
_Avoid_: standard EDA, default pipeline, preset (a preset is a set of analyses run together, which the first look isn't)

**Preset**:
A named set of analyses meant to run together.
_Avoid_: profile (only a verb here, as in profiling missing values)

**Hypothesis family**:
A set of hypothesis tests whose p-values are adjusted together for multiple comparisons.
_Avoid_: test family, family (unqualified)

### Columns and values

**Logical type**:
What a column's values mean, as opposed to the dtype pandas stores them in: boolean, integer, float, categorical, text (free text), datetime or timedelta (a duration).
_Avoid_: semantic type, data type (when you mean the meaning)

**Sentinel value**:
A stand-in value that means "missing", such as -999, "N/A" or 1900-01-01.
_Avoid_: placeholder, missing code, magic number

**Level**:
A distinct non-missing value of a column, counted as pandas' `value_counts` and `factorize` tell values apart: 1, 1.0 and True are one level, while 'Yes' and 'yes ' are two, and so are '' and '\x00a', and None, NaN, NaT and pd.NA are no level.
_Avoid_: category (unless you mean a category of a pandas categorical dtype), unique value (a level can occur many times), factor

**Native missing value**:
A value that is missing without a declaration: None, NaN, NaT or pd.NA. pandas `isna()` counts all of them but NaN that a pyarrow-backed or nullable float column stores as a value; analyses count that NaN as missing too, and a `nan_counted_as_missing` limitation names such columns. A sentinel value is not one until the caller declares it, and then it is counted apart.
_Avoid_: null (in results, null means a field that has no value for the data), NA, empty value

**Compatible value**, **convertible value** and **invalid value**:
The three groups into which `validate_column_types` sorts the non-missing values of a column against its declared logical type. A compatible value is already stored as that type. A convertible value is stored as another type and converts to it exactly, without changing its meaning. An invalid value is neither. Nothing is converted.
_Avoid_: valid (for compatible and convertible together), mismatch, bad value, castable, coercible

**Key**, **primary key** and **candidate key**:
A key is a set of columns whose combination of values no two rows share, with missing values compared equal. A primary key is a key without missing values, so it is fit to identify rows. A candidate key is a minimal key: each of its columns adds distinct combinations, so no subset of it is a key.
_Avoid_: unique key, composite key (say a key of several columns), identifier (that is a row ID), index

**Target**, **class** and **label**:
The target is the column a model learns to predict. A label is a value of the target column. For classification, each distinct non-missing label is a class, counted with its share of the valid labels; a regression target has no classes.
_Avoid_: outcome, response, dependent variable, y; label (for a column's name: that is a column label, always qualified); category (for a class)

**Distinct count**:
The number of levels a column has, `distinct_count`: in `describe_dataset`, `infer_column_semantics`, `summarize_categorical` and, for a categorical column, `compute_association_matrix`. A key's own count, `distinct_row_count`, is the distinct rows of its columns, missing values compared equal. `class_count` counts the classes of a target, which `profile_target` may extend with declared classes no row holds.
_Avoid_: `cardinality`, `level_count` (for a column's distinct values; `level_count` stays for the levels of an index)

**Non-missing count**:
The values of a column that aren't missing, `non_missing_count`, in every analysis that counts them. `summarize_numeric` counts infinite values in it.
_Avoid_: `count` (for a column's non-missing values; `count` stays for the rows with one label or one value)

**Left-out column**:
A column an analysis didn't use, named in `left_out_columns`, with its reason where the entry has one.
_Avoid_: skipped column, `skipped_columns`

**Duplicate group row count**:
The rows that share their values with another row, the first of each group too: `duplicate_group_row_count`, in `find_duplicate_rows` and for each column set in `find_candidate_keys`.
_Avoid_: `duplicate_row_count`

**Schema**:
The logical types a caller declares, by column name: the `schema` argument of `infer_column_semantics` and `validate_column_types`. A declared column the data lacks is a `missing_declared_columns` finding in both.
_Avoid_: `declared_schema`, `declared_columns_missing`

**Passing column**:
A column that passes every check of `validate_column_types`: only compatible and convertible values, and none missing when declared non-nullable. The `passing_columns` finding lists them.
_Avoid_: valid column, `valid_columns`

**Sentinel candidate**:
A value of one column that `detect_missing_sentinels` reports as possibly standing in for missing values: one the caller declared, or one a built-in rule suspects. A suspected candidate stays a valid value until the caller declares it.
_Avoid_: possible sentinel, placeholder candidate, magic value, outlier

### Projects and guardrails

**Project**:
The optional declaration of a data science project's dataset roles and data policy, given to analyses where roles or the data policy matter.
_Avoid_: context, session, config

**Project file**:
A TOML file that holds a Project, so that scripts running in new processes can load the same Project instead of building it again.
_Avoid_: config file, settings file

**Dataset role**:
The part a dataset plays in a project: train, validation, test, holdout or reference.
_Avoid_: split (when you mean the role)

**Data policy**:
Declared rules about which datasets and columns may be used for which purpose. The package checks them to catch mistakes; it is not access control.
_Avoid_: access control, security policy

**Sensitive column**:
A column the data policy marks as holding values that should not appear in results when masking is on.
_Avoid_: PII column, private column

**Masking**:
Replacing the raw values of sensitive columns in a result with placeholders, while keeping counts and statistics. It is a guardrail, not anonymization.
_Avoid_: redaction, anonymization

### Calling over MCP

**MCP server**:
The local process that makes the analyses callable over the Model Context Protocol, for agents that call tools instead of writing Python.
_Avoid_: server (unqualified), gateway, API

**MCP tool**:
The protocol-level entry point for one analysis or discovery function on the MCP server. Always qualified: an analysis is never called a tool.
_Avoid_: tool (unqualified), endpoint, command

**Data path**:
The file an MCP tool reads the DataFrame from, in place of the DataFrame a Python caller passes.
_Avoid_: dataset (when you mean the file), source, input file
