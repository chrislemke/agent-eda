"""Analyses that summarize the distribution of each column's values."""

from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from annotated_types import Ge, Le, Lt

from agent_eda._columns import (
    TO_LABELS,
    TO_NUMBERS,
    ColumnKind,
    ColumnRule,
    select_columns,
)
from agent_eda._levels import (
    Levels,
    count_levels,
    ordered,
    record_label,
    type_name,
    value_types,
    variant_groups,
)
from agent_eda._masking import (
    MaskedLevels,
    MaskingDecision,
    mask_levels,
    resolve_masking,
)
from agent_eda._numeric import (
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_QUANTILES,
    DEFAULT_TRIM_FRACTION,
    NumericSummary,
    QuantileMethod,
    null_limitations,
    numeric_entry,
    numeric_support,
    read_limitations,
    read_numeric,
    stored_nan_count,
    stored_nan_limitations,
    summarize_values,
)
from agent_eda._registry import analysis
from agent_eda._result_base import (
    ColumnSupport,
    ExclusionReason,
    Finding,
    Limitation,
    StatusReason,
    Support,
)
from agent_eda._run import check_distinct, data_shape, execution, overall_status
from agent_eda._text import join_items, join_labels, label_text, plural
from agent_eda.project import Project
from agent_eda.statistics._descriptive_results import (
    SummarizeCategoricalColumn,
    SummarizeCategoricalLevel,
    SummarizeCategoricalResult,
    SummarizeCategoricalVariants,
    SummarizeNumericColumn,
    SummarizeNumericResult,
    level_label,
)

_NUMERIC = ColumnRule(
    default=frozenset({"numeric"}),
    default_text="every numeric column except booleans",
    accepted=frozenset({"numeric", "boolean"}),
    accepted_text=(
        "integer, float and boolean columns, nullable and pyarrow-backed ones of "
        "those kinds included"
    ),
    fixes={
        kind: TO_NUMBERS[kind]
        for kind in ("text", "object", "categorical", "datetime", "timedelta")
    },
    alternatives=dict.fromkeys(
        ("text", "object", "categorical"), "summarize_categorical"
    ),
)


@analysis(
    statuses=("completed", "partial", "insufficient_data", "not_applicable"),
    finding_codes=("constant_columns", "infinite_values"),
    tags=(
        *("distribution", "mean", "standard deviation", "quantiles", "numeric"),
        *("percentiles", "median", "skewed", "skewness", "kurtosis", "outliers"),
        *("variance", "spread"),
    ),
)
def summarize_numeric(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    quantiles: Sequence[Annotated[float, Ge(0), Le(1)]] = DEFAULT_QUANTILES,
    quantile_method: QuantileMethod = DEFAULT_QUANTILE_METHOD,
    trim_fraction: Annotated[float, Ge(0), Lt(0.5)] = DEFAULT_TRIM_FRACTION,
) -> SummarizeNumericResult:
    """How are the values in each numeric column distributed?

    When to use:
        Call it for the distribution of every numeric column: missing, infinite, zero
        and negative counts; the mean and standard deviation; the extremes, quantiles
        and median; a trimmed mean and the median absolute deviation, which outliers
        barely move; and skewness and excess kurtosis. Booleans are left out unless named
        in columns. Statistics of identifiers or codes stored as numbers mean nothing,
        and numbers stored as text are left out: `infer_column_semantics` finds both. A
        text, categorical or datetime column named in columns is rejected. For which
        values occur in text, categorical and boolean columns and how often, call
        `summarize_categorical`. Stand-ins count here as values and distort every
        statistic: `detect_missing_sentinels` finds the likely ones.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to summarize, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself, such as 0, or as its string form. Boolean columns
            named here are read as 0 and 1. None, the default, selects every numeric
            column except booleans, in DataFrame order.
        quantiles (Sequence[float]): Quantile levels, each from 0 to 1 and
            listed once, in the order to report them; an empty list computes
            none. Defaults to 0.01, 0.05, 0.25, 0.5, 0.75, 0.95 and 0.99.
        quantile_method (str): How numpy.quantile picks or interpolates between
            values: Hyndman and Fan's definitions 1 to 9 as 'inverted_cdf',
            'averaged_inverted_cdf', 'closest_observation',
            'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear',
            'median_unbiased' and 'normal_unbiased', or 'lower', 'higher',
            'midpoint' and 'nearest'. Defaults to 'linear', definition 7, the
            default of NumPy, pandas and R. NumPy before 2.0.1 rounds
            'closest_observation' differently from definition 3.
        trim_fraction (float): The share of the finite values cut from each end
            for the trimmed mean: trim_fraction times n, rounded down, values each, as
            SciPy's trim_mean cuts. From 0 up to but excluding 0.5; defaults to 0.1.

    Returns:
        SummarizeNumericResult: `columns` maps each selected column name to its
        entry: the counts (`non_missing_count`, `missing_count`, `infinite_count`,
        `zero_count` and `negative_count`), `mean`, `std` with ddof 1, `min`, `max`,
        `quantiles` keyed by level, `median`, `trimmed_mean`, the median absolute
        deviation as `mad` and `mad_scaled`, `skewness` and `excess_kurtosis`
        (both bias-corrected, 0 for normal data), and the column's own `status`
        and `status_reason`. Every statistic uses the finite values only, and the
        description of each entry field, in `describe_analysis`, defines it.
        `left_out_columns` names the columns that the default columns skipped.

        A statistic that doesn't exist is null, and a limitation names its
        field and the reason. `too_few_values` means too few finite values:
        std needs 2, skewness 3, excess_kurtosis 4 and the rest 1.
        `zero_variance` means every finite value is equal, and
        `numerical_warning` that float64 can't hold the result. Three more
        limitations flag booleans read as 0 and 1, `boolean_as_numeric`;
        integers or decimals beyond ±2^53 that float64 rounds, `float_precision`;
        and NaN stored as a value and counted as missing, `nan_counted_as_missing`.

        Both findings are suspected:

        - `constant_columns`: suspected; a column holds a single distinct non-missing
          value in at least two rows, on stored values; NaN is
          missing.
        - `infinite_values`: suspected; a column holds inf or -inf, which every
          statistic leaves out.

    Statuses:
        completed: Every selected column has a finite value.
        partial: Some selected columns have no finite value; their entries have
            status insufficient_data. Reason code: `incomplete_columns`.
        insufficient_data: The data has no rows, or no selected column has a
            finite value. Reason codes: `no_rows` and `no_values`.
        not_applicable: columns is left out and the data has no numeric columns
            other than booleans. Reason code: `no_columns`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, 51, 29, 47, None, 38],
        ...         "income": [52.5, 61.0, 48.25, 1e6, 70.0, 55.0],
        ...         "member": [True, False, True, True, False, True],
        ...     }
        ... )
        >>> res = ae.summarize_numeric(df)
        >>> print(res)
        summarize_numeric: completed on 6 rows x 3 columns
        Statistics use the finite values. Quantile method linear; trimmed_mean cuts trim_fraction 0.1 from each end.
        Left out, boolean, which counts only when columns names it: 'member'.
        res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis.
        Columns (2 of 2 shown):
        column  non_missing_count  missing_count       mean        std    min  median    max
        age                     5              1       39.8      9.094     29      38     51
        income                  6              0  1.667e+05  4.082e+05  48.25      58  1e+06
        Findings: none
        Limitations: none
        More: res.to_frame() has all 2 columns; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["income"].median
        58.0
        >>> trimmed = ae.summarize_numeric(df, columns="income", trim_fraction=0.2)
        >>> trimmed.columns["income"].trimmed_mean
        59.625

    See Also:
        summarize_categorical: Each level's count and share and the most common levels
            of every text, categorical and boolean column.
        infer_column_semantics: Which columns hold identifiers, codes or numbers stored
            as text rather than measurements.
        profile_target: These statistics for a regression target, or the class counts
            and imbalance of a classification target, checked against the task.
    """
    levels = list(quantiles)
    check_distinct("summarize_numeric", "quantiles", levels, "level")
    selection = select_columns("summarize_numeric", data, columns, _NUMERIC)
    selected = selection.columns
    row_count = len(data)
    read = {
        column.name: read_numeric(data.iloc[:, column.position]) for column in selected
    }
    summaries = {
        name: summarize_values(
            column,
            quantiles=levels,
            quantile_method=quantile_method,
            trim_fraction=trim_fraction,
        )
        for name, column in read.items()
    }
    entries = {
        name: numeric_entry(SummarizeNumericColumn, name, summary)
        for name, summary in summaries.items()
    }
    status, status_reason = overall_status(
        row_count, list(entries.values()), no_columns=_NO_NUMERIC_COLUMNS
    )
    result = SummarizeNumericResult(
        analysis="summarize_numeric",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": [column.name for column in selected],
            "quantiles": list(levels),
            "quantile_method": quantile_method,
            "trim_fraction": trim_fraction,
        },
        support=Support(
            row_count=row_count,
            column_count=len(selected),
            columns=[
                numeric_support(name, summary) for name, summary in summaries.items()
            ],
        ),
        findings=[
            *_constant_columns(summaries),
            *_infinite_values(summaries),
        ],
        limitations=[
            *read_limitations(read, _entry_path),
            *null_limitations(summaries, _entry_path),
        ],
        execution=execution(("numpy", "pandas")),
        columns=entries,
        left_out_columns=selection.left_out,
    )
    result._left_out_kinds = selection.left_out_kinds
    return result


_NO_NUMERIC_COLUMNS = StatusReason(
    code="no_columns",
    message=(
        "The data has no numeric columns. Booleans count only when named in "
        "columns, which reads them as 0 and 1."
    ),
)


def _entry_path(name: str) -> str:
    """The result path of a column's entry, such as "columns['age']"."""
    return f"columns[{name!r}]"


def _constant_columns(summaries: dict[str, NumericSummary]) -> list[Finding]:
    names = [name for name, summary in summaries.items() if summary.constant]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    return [
        Finding(
            code="constant_columns",
            claim=f"{hold} a single distinct non-missing value: {join_labels(names)}.",
            assessment="suspected",
            columns=names,
            fields=[
                f"{_entry_path(name)}.{field}"
                for name in names
                for field in (
                    ("infinite_count",)
                    if summaries[name].constant_infinite
                    else ("min", "max")
                )
            ],
        )
    ]


def _infinite_values(summaries: dict[str, NumericSummary]) -> list[Finding]:
    names = [name for name, summary in summaries.items() if summary.infinite_count]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    return [
        Finding(
            code="infinite_values",
            claim=(
                f"{hold} infinite values, which every statistic leaves out: "
                f"{join_labels(names)}."
            ),
            assessment="suspected",
            columns=names,
            fields=[f"{_entry_path(name)}.infinite_count" for name in names],
        )
    ]


_CATEGORICAL = ColumnRule(
    default=frozenset({"text", "object", "categorical", "boolean"}),
    default_text="every text, object, categorical and boolean column",
    accepted=frozenset({"text", "object", "categorical", "boolean", "numeric"}),
    accepted_text=(
        "text, object, categorical, boolean and numeric columns, nullable and "
        "pyarrow-backed ones of those kinds included"
    ),
    fixes={kind: TO_LABELS[kind] for kind in ("datetime", "timedelta", "other")},
)


@analysis(
    statuses=("completed", "partial", "insufficient_data", "not_applicable"),
    finding_codes=("case_or_space_variants", "mixed_value_types", "unused_categories"),
    tags=(
        "value counts",
        "frequencies",
        "levels",
        "categories",
        "distinct_count",
        "most common values",
        "unused categories",
        "case variants",
        "capitalization",
        "whitespace",
        "typos",
        "mixed types",
    ),
    masking=True,
)
def summarize_categorical(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    top_n: Annotated[int, Ge(1), Le(100)] = 10,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> SummarizeCategoricalResult:
    """Which values occur in each categorical column, and how often?

    When to use:
        Call it for the values of every text, object, categorical and boolean column:
        each level with its count and its share of the non-missing values, the missing
        count, the distinct_count and the most common levels. It also flags labels that
        differ only by case or surrounding spaces, such as 'Yes', 'yes ' and 'YES',
        which a model treats as different categories; values of different Python types
        in one column, such as 1 and '1'; and categories of a categorical dtype that
        never occur. Name a numeric column in columns to count its values, such as codes
        1, 2 and 3. Levels are raw values: for columns that hold personal data, declare
        them sensitive in a project and switch masking on. Labels that stand in for
        missing values, such as 'N/A', count here as levels: `detect_missing_sentinels`
        finds them. For how numeric values are distributed,
        call `summarize_numeric` instead. For the classes of the column a model will
        predict, call `profile_target` instead: it checks them against the task and the
        labels you expect, and measures their imbalance.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to summarize, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself, such as 0, or as its string form. Numeric columns
            named here are summarized by their values; datetime and duration columns
            raise. None, the default, selects every text, object, categorical and
            boolean column, in DataFrame order.
        top_n (int): How many of each column's most common levels `levels`
            holds and the printed text shows, from 1 to 100; defaults to 10.
            It limits the record and the text: `to_frame` has every level.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces each level of a sensitive
            column with a numbered placeholder, '<masked 1>' for the most
            common and so on in level order, and needs a project that
            declares sensitive columns; False shows the levels as stored.
            None, the default, follows the project's
            data_policy.mask_sensitive, and masks nothing without a project.
            Masking is a guardrail against accidental exposure, not
            anonymization or access control.

    Returns:
        SummarizeCategoricalResult: `columns` maps each selected column name to
        its entry. `levels` holds the first top_n levels by count, descending,
        then by label: booleans and numbers by value, then text by Unicode code
        point, then other types by type name; the record adds `levels_total`,
        which equals `distinct_count`. Shares divide by the column's
        `non_missing_count`, so missing values never dilute them. Labels are kept
        as stored, so 'Yes' and 'yes ' are two levels; a label JSON can't hold,
        such as a Decimal or a timestamp, appears as its string form, with its
        type in `label_type`.
        `left_out_columns` names the columns that the default columns skipped.
        Values that can't be hashed, such as lists, leave a column's levels
        uncounted, with an `unhashable_values` limitation. NaN is missing, not a
        level, also NaN that a pyarrow-backed or nullable float column stores as
        a value: the `nan_counted_as_missing` limitation names such columns. With
        masking on, a sensitive column's labels and unused categories are
        placeholders in every output; counts, shares and label types stay.

        Two findings are suspected and one is descriptive:

        - `case_or_space_variants`: suspected; a column holds labels that
          differ only by case or surrounding spaces, such as 'Yes' and 'yes '.
        - `mixed_value_types`: suspected; a column holds values of more than
          one Python type, such as 1 and '1'.
        - `unused_categories`: descriptive; a categorical column has categories
          that no value takes.

    Statuses:
        completed: Every selected column has a non-missing value.
        partial: Some selected columns have no non-missing value or hold values
            that can't be hashed; their entries say which. Reason code:
            `incomplete_columns`.
        insufficient_data: The data has no rows, or no selected column
            completed and at least one of them has no non-missing value, even
            if the others hold values that can't be hashed. Reason codes:
            `no_rows` and `no_values`.
        not_applicable: columns is left out and the data has no text, object,
            categorical or boolean columns, or every selected column holds
            values that can't be hashed. Reason codes: `no_columns` and
            `unhashable_values`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "plan": ["basic", "pro", "basic", None, "team", "basic"],
        ...         "answer": ["Yes", "yes ", "no", "YES", "no", "Yes"],
        ...         "grade": pd.Categorical(
        ...             ["a", "b", "a", "a", None, "b"], categories=["a", "b", "c"]
        ...         ),
        ...         "age": [34, 51, 29, 47, 38, 41],
        ...     }
        ... )
        >>> res = ae.summarize_categorical(df, top_n=3)
        >>> print(res)
        summarize_categorical: completed on 6 rows x 4 columns
        Shares are fractions of non-missing values. Levels go by count, then label; levels has the top 3 of each column.
        Left out, not text, object, categorical or boolean (numeric): 'age'.
        Columns (3 of 3 shown):
        column  non_missing_count  missing_count  distinct_count
        answer                  6              0               4
        grade                   5              1               2
        plan                    5              1               3
        Top levels per column:
          answer: 'Yes': 2 (share 0.3333), 'no': 2 (share 0.3333), 'YES': 1 (share 0.1667)
          grade: 'a': 3 (share 0.6), 'b': 2 (share 0.4)
          plan: 'basic': 3 (share 0.6), 'pro': 1 (share 0.2), 'team': 1 (share 0.2)
        Findings (2 of 2 shown):
        - suspected case_or_space_variants: 1 column holds labels that differ only by case or surrounding spaces: 'answer'
          ('Yes', 'YES' and 'yes ').
        - descriptive unused_categories: 1 categorical column has categories that never occur: 'grade'.
        Limitations: none
        More: res.to_frame() has all 9 levels; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["plan"].levels[0].share
        0.6
        >>> [
        ...     level.label
        ...     for level in res.columns["answer"].case_or_space_variants[0].levels
        ... ]
        ['Yes', 'YES', 'yes ']
        >>> project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["plan"]))
        >>> masked = ae.summarize_categorical(df, project=project, mask_sensitive=True)
        >>> [level.label for level in masked.columns["plan"].levels]
        ['<masked 1>', '<masked 2>', '<masked 3>']

    See Also:
        find_duplicate_rows: Groups of rows that are equal in every column, with their
            row IDs.
        infer_column_semantics: Which columns hold mixed types, identifiers or numbers
            stored as text.
        detect_missing_sentinels: Values that look like stand-ins for missing ones, such
            as 'N/A', 'unknown' or empty text.
    """
    masking = resolve_masking("summarize_categorical", project, mask_sensitive)
    selection = select_columns("summarize_categorical", data, columns, _CATEGORICAL)
    selected = selection.columns
    tables: dict[str, Levels] = {}
    entries: dict[str, SummarizeCategoricalColumn] = {}
    stored_nan: list[str] = []
    for column in selected:
        series = data.iloc[:, column.position]
        if stored_nan_count(series):
            stored_nan.append(column.name)
        levels = count_levels(series, column.kind)
        masked = (
            mask_levels(levels)
            if levels is not None and masking.masks(column.name)
            else None
        )
        if levels is not None:
            # A masked column keeps only its placeholders, never its labels.
            tables[column.name] = levels if masked is None else masked.levels
        entries[column.name] = _categorical_entry(
            column.name, series, column.kind, levels, top_n, masked
        )
    status, status_reason = overall_status(
        len(data), list(entries.values()), no_columns=_NO_CATEGORICAL_COLUMNS
    )
    result = SummarizeCategoricalResult(
        analysis="summarize_categorical",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": [column.name for column in selected],
            "top_n": top_n,
            "project": project.model_dump(mode="json") if project else None,
            "mask_sensitive": mask_sensitive,
        },
        support=Support(
            row_count=len(data),
            column_count=len(selected),
            columns=[_categorical_support(e, len(data)) for e in entries.values()],
        ),
        findings=[
            *_case_or_space_variants(entries, masking),
            *_mixed_value_types(entries),
            *_unused_categories(entries),
        ],
        limitations=[
            *stored_nan_limitations(
                stored_nan,
                [f"{_entry_path(name)}.missing_count" for name in stored_nan],
                "missing_count counts it",
            ),
            *_unhashable_values(entries),
        ],
        execution=execution(("numpy", "pandas")),
        masking=masking.record([column.name for column in selected], data),
        columns=entries,
        left_out_columns=selection.left_out,
    )
    result._tables = tables
    result._left_out_kinds = selection.left_out_kinds
    return result


_NO_CATEGORICAL_COLUMNS = StatusReason(
    code="no_columns",
    message=(
        "The data has no text, object, categorical or boolean columns. Name a "
        "numeric column in columns to count its values."
    ),
)

_NO_VALUES = StatusReason(
    code="no_values", message="There are no non-missing values to count."
)
_UNHASHABLE = StatusReason(
    code="unhashable_values",
    message=(
        "The column holds values that can't be hashed, such as lists or dicts, so "
        "its levels can't be counted."
    ),
)
# The counts of levels that are null when a column's values can't be hashed.
_LEVEL_COUNTS = ("distinct_count", "left_out_level_count", "seen_once_count")


def _categorical_entry(
    name: str,
    series: pd.Series,
    kind: ColumnKind,
    levels: Levels | None,
    top_n: int,
    masked: MaskedLevels | None,
) -> SummarizeCategoricalColumn:
    """One column's entry; with `masked`, placeholders stand for its labels."""
    types = value_types(series, kind, levels)
    if levels is None:
        missing = int(series.isna().sum())
        return SummarizeCategoricalColumn(
            column=name,
            status="not_applicable",
            status_reason=_UNHASHABLE,
            non_missing_count=len(series) - missing,
            missing_count=missing,
            distinct_count=None,
            levels=[],
            left_out_level_count=None,
            seen_once_count=None,
            unused_categories=[],
            value_types=types,
            mixed_value_types=len(types) > 1,
            case_or_space_variants=[],
        )
    non_missing = len(series) - levels.missing_count
    shown = [
        _level(levels, position, non_missing, masked)
        for position in ordered(levels, top_n)
    ]
    distinct_count = len(levels.counts)
    unused = (
        [record_label(c) for c in levels.unused_categories]
        if masked is None
        else masked.unused_categories
    )
    return SummarizeCategoricalColumn(
        column=name,
        status="completed" if non_missing else "insufficient_data",
        status_reason=None if non_missing else _NO_VALUES,
        non_missing_count=non_missing,
        missing_count=levels.missing_count,
        distinct_count=distinct_count,
        levels=shown,
        left_out_level_count=distinct_count - len(shown),
        seen_once_count=int((levels.counts == 1).sum()),
        unused_categories=unused,
        value_types=types,
        mixed_value_types=len(types) > 1,
        case_or_space_variants=[
            SummarizeCategoricalVariants(
                levels=[
                    _level(levels, position, non_missing, masked) for position in group
                ]
            )
            for group in variant_groups(levels)
        ],
    )


def _level(
    levels: Levels, position: int, non_missing: int, masked: MaskedLevels | None
) -> SummarizeCategoricalLevel:
    label = levels.label(position)
    count = int(levels.counts[position])
    return SummarizeCategoricalLevel(
        label=record_label(label) if masked is None else masked.label(position),
        label_type=type_name(label),
        count=count,
        share=count / non_missing,
    )


def _categorical_support(
    entry: SummarizeCategoricalColumn, row_count: int
) -> ColumnSupport:
    excluded: dict[ExclusionReason, int] = {}
    if entry.missing_count:
        excluded["missing"] = entry.missing_count
    if entry.distinct_count is None and entry.non_missing_count:
        excluded["unhashable_values"] = entry.non_missing_count
    return ColumnSupport(
        column=entry.column,
        eligible_count=row_count,
        evaluated_count=row_count - sum(excluded.values()),
        excluded_count=sum(excluded.values()),
        excluded_by_reason=excluded,
    )


def _unhashable_values(
    entries: dict[str, SummarizeCategoricalColumn],
) -> list[Limitation]:
    names = [name for name, entry in entries.items() if entry.distinct_count is None]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    its = "its" if len(names) == 1 else "their"
    return [
        Limitation(
            code="unhashable_values",
            reason=(
                f"{hold} values that can't be hashed, such as lists or dicts, so "
                f"{its} levels weren't counted and {its} counts of levels are null: "
                f"{join_labels(names)}."
            ),
            fields=[
                f"{_entry_path(name)}.{field}"
                for name in names
                for field in _LEVEL_COUNTS
            ],
        )
    ]


# A column's variant groups, and the groups a claim names per column.
_NAMED_VARIANT_GROUPS = 3


def _variants_text(entry: SummarizeCategoricalColumn, *, masked: bool) -> str:
    """A column and its variant groups, such as "'answer' ('Yes' and 'yes ')";
    several groups are separated by semicolons."""
    groups = [
        join_items([level_label(level, masked=masked) for level in group.levels])
        for group in entry.case_or_space_variants
    ]
    named = "; ".join(groups[:_NAMED_VARIANT_GROUPS])
    if len(groups) > _NAMED_VARIANT_GROUPS:
        named += f"; and {len(groups) - _NAMED_VARIANT_GROUPS:,} more groups"
    return f"{label_text(entry.column)} ({named})"


def _case_or_space_variants(
    entries: dict[str, SummarizeCategoricalColumn], masking: MaskingDecision
) -> list[Finding]:
    names = [name for name, entry in entries.items() if entry.case_or_space_variants]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    described = [_variants_text(entries[n], masked=masking.masks(n)) for n in names]
    return [
        Finding(
            code="case_or_space_variants",
            claim=(
                f"{hold} labels that differ only by case or surrounding spaces: "
                f"{join_items(described)}."
            ),
            assessment="suspected",
            columns=names,
            fields=[f"{_entry_path(name)}.case_or_space_variants" for name in names],
        )
    ]


def _mixed_value_types(
    entries: dict[str, SummarizeCategoricalColumn],
) -> list[Finding]:
    names = [name for name, entry in entries.items() if entry.mixed_value_types]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    described = [
        f"{label_text(name)} ({', '.join(entries[name].value_types)})" for name in names
    ]
    return [
        Finding(
            code="mixed_value_types",
            claim=(
                f"{hold} values of more than one Python type: {join_items(described)}."
            ),
            assessment="suspected",
            columns=names,
            fields=[f"{_entry_path(name)}.value_types" for name in names],
        )
    ]


def _unused_categories(
    entries: dict[str, SummarizeCategoricalColumn],
) -> list[Finding]:
    names = [name for name, entry in entries.items() if entry.unused_categories]
    if not names:
        return []
    have = plural(len(names), "categorical column has", "categorical columns have")
    return [
        Finding(
            code="unused_categories",
            claim=f"{have} categories that never occur: {join_labels(names)}.",
            assessment="descriptive",
            columns=names,
            fields=[f"{_entry_path(name)}.unused_categories" for name in names],
        )
    ]
