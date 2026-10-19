"""The results of summarize_numeric and summarize_categorical."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import pandas as pd
from pydantic import (
    Field,
    PrivateAttr,
)

from agent_eda import _text
from agent_eda._levels import LEVEL_ORDER, Levels, levels_frame
from agent_eda._result_base import (
    _BIG_INTEGERS,
    _BY_NAME,
    _ENTRIES,
    _FRAME_DTYPES,
    _MORE_STATISTICS,
    AnalysisResult,
    ColumnResult,
    MaskableResult,
    ResultModel,
    _Cap,
    _check_table,
    _left_out_lines,
    _tables_unavailable,
)


class SummarizeNumericColumn(ColumnResult):
    """What summarize_numeric found in one column."""

    non_missing_count: int = Field(
        description="Values that aren't missing, infinite ones included."
    )
    missing_count: int = Field(
        description="Native missing values: NaN and the missing markers of "
        "object, nullable and pyarrow-backed columns, including NaN that a "
        "pyarrow-backed or nullable float column stores as a value, which pandas' "
        "missing-value check doesn't count."
    )
    infinite_count: int = Field(description="Values equal to inf or -inf.")
    zero_count: int = Field(description="Values equal to 0.")
    negative_count: int = Field(description="Values below 0, -inf included.")
    mean: float | None = Field(description="Arithmetic mean of the finite values.")
    std: float | None = Field(
        description="Sample standard deviation of the finite values with ddof 1: "
        "the square root of the sum of squared deviations from the mean divided by "
        "n - 1, where n counts finite values. Needs n >= 2."
    )
    min: float | None = Field(description="Smallest finite value.")
    max: float | None = Field(description="Largest finite value.")
    quantiles: dict[float, float | None] = Field(
        description="Quantiles of the finite values, keyed by level in the order "
        "of parameters.quantiles, as numpy.quantile computes them with "
        "parameters.quantile_method."
    )
    median: float | None = Field(
        description="Middle finite value, or the mean of the two middle values; "
        "the same whatever quantile_method is."
    )
    trimmed_mean: float | None = Field(
        description="Mean of the finite values left after cutting "
        "trim_fraction * n, rounded down, of them from each end, as SciPy's "
        "trimmed mean does."
    )
    mad: float | None = Field(
        description="Median absolute deviation of the finite values, unscaled: the median "
        "of the absolute differences from their median."
    )
    mad_scaled: float | None = Field(
        description="mad * 1.4826, which estimates the standard deviation when the "
        "values are normally distributed."
    )
    skewness: float | None = Field(
        description="Bias-corrected sample skewness G1 = (n * (n - 1))^0.5 / (n - 2) "
        "* m3 / m2^1.5 of the finite values, where m_k is the mean of the k-th "
        "powers of the deviations from the mean: SciPy's skewness with bias "
        "correction. Needs n >= 3 and two distinct values."
    )
    excess_kurtosis: float | None = Field(
        description="Bias-corrected sample excess kurtosis G2 = (n - 1) / ((n - 2) * "
        "(n - 3)) * ((n + 1) * m4 / m2^2 - 3 * (n - 1)) of the finite "
        "values, with m_k as for skewness; 0 for a normal distribution: SciPy's "
        "excess kurtosis with bias correction. Needs n >= 4 and two distinct "
        "values."
    )


class SummarizeNumericResult(AnalysisResult):
    """The result of summarize_numeric."""

    columns: Annotated[dict[str, SummarizeNumericColumn], _ENTRIES, _BY_NAME] = Field(
        description="One entry per selected column, keyed by column name, in the "
        "selected order. Every statistic uses the finite values only, and the "
        "description of each entry field defines it. The record keeps the first "
        "100 and adds columns_total; the complete table has every column."
    )
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns that the default column selection left out because "
        "they aren't numeric or are boolean, in DataFrame order; empty when the "
        "call names the columns."
    )
    # The kind of each left-out column, outside the record, so that the text
    # says why it was left out; None for a result rebuilt from its record.
    _left_out_kinds: Mapping[str, str] | None = PrivateAttr(default=None)

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        settings = self.settings
        facts = [
            "Statistics use the finite values. Quantile method "
            f"{settings.quantile_method}; trimmed_mean cuts "
            f"trim_fraction {_text.format_number(settings.trim_fraction)} from each end."
        ]
        facts.extend(
            _left_out_lines(
                self.left_out_columns,
                self._left_out_kinds,
                wanted="numeric",
                boolean="boolean, which counts only when columns names it",
            )
        )
        facts.append(_text.also_in_table(_MORE_STATISTICS, surface))
        return facts

    def _text_table(self) -> _text.Table:
        headers = (
            "column",
            "non_missing_count",
            "missing_count",
            "mean",
            "std",
            "min",
            "median",
        )
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=(*headers, "max"),
            rows=[
                (
                    e.column,
                    e.non_missing_count,
                    e.missing_count,
                    e.mean,
                    e.std,
                    e.min,
                    e.median,
                    e.max,
                )
                for e in self._by_interest(
                    list(self.columns.values()), list(self.columns)
                )
            ],
        )

    def to_frame(self, table: Literal["columns"] = "columns") -> pd.DataFrame:
        """Every statistic of every column, one row per column in selected order.

        The quantiles become one column per level, named "q" and the level,
        such as "q0.25"; status_reason holds the reason's code.

        Args:
            table: 'columns', the only table.
        """
        _check_table(table, ("columns",))
        levels = [float(level) for level in self.settings.quantiles]
        return numeric_frame(list(self.columns.values()), levels)


def numeric_frame(
    entries: Sequence[SummarizeNumericColumn], levels: Sequence[float]
) -> pd.DataFrame:
    """summarize_numeric's entries as a table, one row per entry.

    The quantiles become one column per level, named "q" and the level, such as
    "q0.25"; status_reason holds the reason's code.
    """
    columns: dict[str, pd.Series] = {}
    for name, field in SummarizeNumericColumn.model_fields.items():
        values = [getattr(entry, name) for entry in entries]
        if name == "status_reason":
            codes = [reason.code if reason else None for reason in values]
            columns[name] = pd.Series(codes, dtype=object)
        elif name == "quantiles":
            for level in levels:
                columns[f"q{level!r}"] = pd.Series(
                    [quantiles[level] for quantiles in values], dtype="float64"
                )
        else:
            dtype = _FRAME_DTYPES.get(field.annotation)
            columns[name] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(columns, columns=list(columns))


class SummarizeCategoricalLevel(ResultModel):
    """A level of a column: a distinct non-missing value and how often it occurs."""

    label: str | bool | int | float = Field(
        description="The value exactly as stored, so 'Yes' and 'yes ' are two "
        "levels. A datetime is ISO 8601 text and a Decimal a number or, when float64 "
        "can't hold it as written, its exact text; any other value JSON can't "
        "hold appears as its string form. label_type names its type. "
        "With masking on, a sensitive column's labels are numbered placeholders, "
        "'<masked 1>' for its first level in level order and so on. " + _BIG_INTEGERS
    )
    label_type: str = Field(
        description="The Python type of the stored value, such as 'str', 'int' or "
        "'bool'; a NumPy scalar counts as the Python type it holds."
    )
    count: int = Field(description="Values equal to the label.")
    share: float = Field(
        description="count / non_missing_count of the column: the share of its "
        "non-missing values, so missing values never dilute it."
    )


def level_label(level: SummarizeCategoricalLevel, *, masked: bool = False) -> str:
    """A level's label as the text shows it: text labels are quoted, so
    surrounding spaces stay visible; a masked label, a placeholder such as
    <masked 1>, is not."""
    quoted = level.label_type == "str" and not masked
    return repr(level.label) if quoted else str(level.label)


def _level_text(level: SummarizeCategoricalLevel, *, masked: bool = False) -> str:
    """A level as the text shows it, such as "'basic': 3 (share 0.6)"."""
    share = _text.format_number(level.share)
    return f"{level_label(level, masked=masked)}: {level.count:,} (share {share})"


class SummarizeCategoricalVariants(ResultModel):
    """Labels of one column that differ only by case or surrounding whitespace."""

    levels: Annotated[list[SummarizeCategoricalLevel], _ENTRIES] = Field(
        description="The labels of the group as levels, each with its count and "
        "share, in level order."
    )


class SummarizeCategoricalColumn(ColumnResult):
    """What summarize_categorical found in one column."""

    non_missing_count: int = Field(description="Values that aren't missing.")
    missing_count: int = Field(
        description="Native missing values: NaN, NaT and the missing markers of "
        "object, nullable and pyarrow-backed columns, the ones pandas' value "
        "counts leave out. NaN that a pyarrow-backed float column stores as a "
        "value is a level, as in those counts."
    )
    distinct_count: int | None = Field(
        description="Levels: the distinct non-missing values, as pandas "
        "value_counts tells them apart, so 1, 1.0 and the boolean true are one level. Null "
        "when the values can't be hashed."
    )
    levels: Annotated[
        list[SummarizeCategoricalLevel], _Cap(100, total="distinct_count")
    ] = Field(
        description="The first top_n levels, each with its label, label_type, "
        "count and share. Level order: " + LEVEL_ORDER + " distinct_count counts "
        "every level, and the record adds it as levels_total; the complete "
        "table has every level."
    )
    left_out_level_count: int | None = Field(
        description="Levels beyond the first top_n, which levels and the printed "
        "text leave out. Null when the values can't be hashed."
    )
    seen_once_count: int | None = Field(
        description="Levels that occur exactly once. Null when the values can't "
        "be hashed."
    )
    unused_categories: Annotated[list[str | bool | int | float], _ENTRIES] = Field(
        description="Categories of a categorical dtype that no value takes, in "
        "category order, each written as a level's label is; empty for other "
        "dtypes. With masking on, a sensitive column's are placeholders numbered "
        "on from its levels'. " + _BIG_INTEGERS
    )
    value_types: Annotated[list[str], _ENTRIES] = Field(
        description="The Python types of the non-missing values by name, sorted, "
        "such as ['int', 'str']; a NumPy scalar counts as the Python type it holds."
    )
    mixed_value_types: bool = Field(
        description="Whether the non-missing values have more than one Python "
        "type, such as 1 and '1'. Values pandas counts as one level, such as 1 "
        "and the boolean true, can still differ in type."
    )
    case_or_space_variants: Annotated[list[SummarizeCategoricalVariants], _ENTRIES] = (
        Field(
            description="Groups of text labels that become equal once str.strip "
            "removes surrounding whitespace and str.casefold folds case, such as "
            "'Yes', 'yes ' and 'YES'. Each group is in level order, and the groups "
            "follow the level order of their first label. Empty when there are none."
        )
    )


class SummarizeCategoricalResult(MaskableResult):
    """The result of summarize_categorical."""

    columns: Annotated[dict[str, SummarizeCategoricalColumn], _ENTRIES, _BY_NAME] = (
        Field(
            description="One entry per selected column, keyed by column name, in the "
            "selected order. Every share divides by the entry's non_missing_count, so "
            "missing values never dilute it and get no share of their own. The record "
            "keeps the first 100 and adds columns_total."
        )
    )
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns that the default column selection left out because "
        "they aren't text, object, categorical or boolean columns, in DataFrame "
        "order; empty when the call names the columns."
    )
    # Every level of every counted column, outside the record; to_frame() sorts
    # them when asked, since sorting a million labels is slow.
    # None for a result rebuilt from its record, which holds no levels.
    _tables: dict[str, Levels] | None = PrivateAttr(default=None)
    # The kind of each left-out column, outside the record, so that the text
    # says why it was left out; None for a result rebuilt from its record.
    _left_out_kinds: Mapping[str, str] | None = PrivateAttr(default=None)

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        facts = [
            "Shares are fractions of non-missing values. Levels go by count, then "
            f"label; levels has the top {self.settings.top_n} of each column."
        ]
        facts.extend(
            _left_out_lines(
                self.left_out_columns,
                self._left_out_kinds,
                wanted="text, object, categorical or boolean",
                boolean="boolean",
            )
        )
        return facts

    def _text_table(self) -> _text.Table:
        entries = self._by_interest(list(self.columns.values()), list(self.columns))
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=("column", "non_missing_count", "missing_count", "distinct_count"),
            rows=[
                (e.column, e.non_missing_count, e.missing_count, e.distinct_count)
                for e in entries
            ],
            frame_rows=sum(e.distinct_count or 0 for e in self.columns.values()),
            frame_unit=("level", "levels"),
            details=[
                f"{e.column}: "
                + ", ".join(
                    _level_text(level, masked=self._masked(e.column))
                    for level in e.levels
                )
                for e in entries[: _text.MAX_TABLE_ROWS]
                if e.levels
            ],
            details_title="Top levels per column",
        )

    def to_frame(self, table: Literal["levels"] = "levels") -> pd.DataFrame:
        """Every level of every column, one row per level.

        Rows follow the selected column order, then level order. The columns
        are column, label, label_type, count and share, and label holds each
        value exactly as stored, or with masking on, a sensitive column's
        placeholder, as in levels. Columns whose values can't be hashed have
        no rows.

        Args:
            table: 'levels', the only table.
        """
        _check_table(table, ("levels",))
        if self._tables is None:
            if any(column.distinct_count for column in self.columns.values()):
                raise _tables_unavailable(self)
            return levels_frame({})
        return levels_frame(self._tables)
