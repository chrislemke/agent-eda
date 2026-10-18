"""Analyses that summarize the distribution of each column's values."""

from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from annotated_types import Ge, Le, Lt

from agent_eda._columns import ColumnRule, select_columns
from agent_eda._numeric import (
    NumericSummary,
    QuantileMethod,
    null_limitations,
    read_limitations,
    read_numeric,
    summarize_values,
)
from agent_eda._registry import analysis
from agent_eda._run import check_distinct, data_shape, execution, overall_status
from agent_eda._text import join_labels, plural
from agent_eda.results import (
    ColumnResult,
    ColumnSupport,
    Finding,
    StatusReason,
    SummarizeNumericColumn,
    SummarizeNumericResult,
    Support,
)

DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

_TEXT_FIX = (
    "convert numbers stored as text with pd.to_numeric({column}, errors='coerce')"
)
_NUMERIC = ColumnRule(
    default=frozenset({"numeric"}),
    default_text="every numeric column except booleans",
    accepted=frozenset({"numeric", "boolean"}),
    accepted_text=(
        "integer, float and boolean columns, nullable and pyarrow-backed ones included"
    ),
    fixes={
        "text": _TEXT_FIX,
        "object": _TEXT_FIX,
        "categorical": "convert numeric categories with {column}.astype(float)",
        "timedelta": "convert durations with {column}.dt.total_seconds()",
    },
)


@analysis(
    statuses=("completed", "partial", "insufficient_data", "not_applicable", "failed"),
    finding_codes=("constant_columns", "infinite_values"),
    tags=("distribution", "mean", "standard deviation", "quantiles", "numeric"),
)
def summarize_numeric(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    quantiles: Sequence[Annotated[float, Ge(0), Le(1)]] = DEFAULT_QUANTILES,
    quantile_method: QuantileMethod = "linear",
    trim_fraction: Annotated[float, Ge(0), Lt(0.5)] = 0.1,
) -> SummarizeNumericResult:
    """How are the values in each numeric column distributed?

    When to use:
        Call it after `describe_dataset` for the distribution of every numeric
        column: missing, infinite, zero and negative counts; the mean and
        standard deviation; the extremes, quantiles and median; a trimmed mean
        and the median absolute deviation, which outliers barely move; and
        skewness and excess kurtosis. The result states how each statistic is
        defined. Booleans are left out unless named in columns. A text,
        categorical or datetime column named in columns raises
        `errors.ColumnTypeError`; `describe_dataset` covers every dtype. For how
        strongly pairs of columns move together, call
        `compute_association_matrix`, and `list_analyses` lists the other
        analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to summarize, in the
            order to report them: a list, a tuple, a pandas Index such as
            df.columns[:3], or a single name. A label that isn't a string can
            be given as itself, such as 0, or as its string form. Boolean
            columns named here are read as 0 and 1. None, the default, selects
            every numeric column except booleans, in DataFrame order.
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
            for the trimmed mean: int(trim_fraction * n) values each, as
            scipy.stats.trim_mean cuts. From 0 up to but excluding 0.5;
            defaults to 0.1.

    Returns:
        SummarizeNumericResult: `columns` maps each selected column name to its
        entry, such as res.columns["age"]. An entry holds:

        - `count`, the values that aren't missing, infinite ones included, and
          `missing_count`, `infinite_count`, `zero_count` and `negative_count`.
        - `mean`, `std` with ddof 1, `min` and `max`.
        - `quantiles`, keyed by level, and `median`.
        - `trimmed_mean`, and the median absolute deviation as `mad`, unscaled,
          and `mad_scaled`, multiplied by 1.4826 to estimate the standard
          deviation of normal data.
        - `skewness` and `excess_kurtosis`, both bias-corrected as SciPy's with
          bias=False and both 0 for normal data.
        - `status` and `status_reason`, the column's own.

        Every statistic uses the finite values only, and `definitions` states
        each one. `left_out_columns` names the columns that columns=None
        skipped. `to_frame()` returns every statistic, one row per column and
        one column per quantile level, such as q0.25.

        A statistic that doesn't exist is null, and a limitation names its
        field and the reason. `too_few_values` means too few finite values:
        std needs 2, skewness 3, excess_kurtosis 4 and the rest 1.
        `zero_variance` means every finite value is equal, and
        `numerical_warning` that float64 can't hold the result. Three more
        limitations flag booleans read as 0 and 1, `boolean_as_numeric`;
        integers beyond ±2^53 that float64 rounds, `float_precision`; and NaN
        stored as a value and counted as missing, `nan_counted_as_missing`.

        Both findings are suspected:

        - `constant_columns`: a column holds a single distinct non-missing
          value in at least two rows, as describe_dataset decides it.
        - `infinite_values`: a column holds inf or -inf, which every statistic
          leaves out.

    Statuses:
        completed: Every selected column has a finite value.
        partial: Some selected columns have no finite value; their entries have
            status insufficient_data.
        insufficient_data: The data has no rows, or no selected column has a
            finite value.
        not_applicable: columns is None and the data has no numeric columns
            other than booleans.
        failed: An anticipated numerical failure stopped the analysis, and
            `status_reason` gives its code and message. None is known today: a
            statistic that can't be computed is null, with a limitation,
            instead.

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
        Statistics use the finite values. Quantile method linear; trimmed_mean cuts 10% from each end.
        Left out, not numeric or boolean: 'member'.
        res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis.
        Columns (2 of 2 shown):
        column  count  missing_count       mean        std    min  median    max
        age         5              1       39.8      9.094     29      38     51
        income      6              0  1.667e+05  4.082e+05  48.25      58  1e+06
        Findings: none
        Limitations: none
        More: res.to_frame() has all 2 columns; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["income"].median
        58.0
        >>> trimmed = ae.summarize_numeric(df, columns="income", trim_fraction=0.2)
        >>> trimmed.columns["income"].trimmed_mean
        59.625

    See Also:
        describe_dataset: Dtypes, missing and distinct counts and memory of every
            column, whatever its dtype.
        compute_association_matrix: How strongly every pair of numeric and
            categorical columns is associated, strongest first.
        list_analyses: Every analysis, with the question each one answers.
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
            column.values,
            quantiles=levels,
            quantile_method=quantile_method,
            trim_fraction=trim_fraction,
        )
        for name, column in read.items()
    }
    entries = {name: _entry(name, summary) for name, summary in summaries.items()}
    status, status_reason = overall_status(
        row_count, list(entries.values()), no_columns=_NO_NUMERIC_COLUMNS
    )
    return SummarizeNumericResult(
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
            columns=[_support(name, summary) for name, summary in summaries.items()],
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
        definitions=_DEFINITIONS,
    )


# Each count and statistic's definition: the description of its field.
_DEFINITIONS = {
    name: field.description or ""
    for name, field in SummarizeNumericColumn.model_fields.items()
    if name not in ColumnResult.model_fields
}
_NO_FINITE_VALUES = StatusReason(
    code="no_finite_values", message="There are no finite values to summarize."
)
_NO_NUMERIC_COLUMNS = StatusReason(
    code="no_numeric_columns",
    message=(
        "The data has no numeric columns. Booleans count only when named in "
        "columns, which reads them as 0 and 1."
    ),
)


def _entry(name: str, summary: NumericSummary) -> SummarizeNumericColumn:
    finite_count = summary.count - summary.infinite_count
    return SummarizeNumericColumn(
        column=name,
        status="completed" if finite_count else "insufficient_data",
        status_reason=None if finite_count else _NO_FINITE_VALUES,
        count=summary.count,
        missing_count=summary.missing_count,
        infinite_count=summary.infinite_count,
        zero_count=summary.zero_count,
        negative_count=summary.negative_count,
        quantiles=summary.quantiles,
        **summary.statistics,
    )


def _support(name: str, summary: NumericSummary) -> ColumnSupport:
    excluded = {
        reason: count
        for reason, count in (
            ("missing", summary.missing_count),
            ("infinite", summary.infinite_count),
        )
        if count
    }
    eligible = summary.count + summary.missing_count
    return ColumnSupport(
        column=name,
        eligible_count=eligible,
        evaluated_count=eligible - sum(excluded.values()),
        excluded_count=sum(excluded.values()),
        excluded_by_reason=excluded,
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
