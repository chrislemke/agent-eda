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

### Projects and guardrails

**Project**:
The optional declaration of a data science project's dataset roles and data policy, given to analyses where roles matter.
_Avoid_: context, session, config

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
