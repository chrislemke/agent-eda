"""Analyses that describe what a dataset is and where it came from."""

import sys
from typing import cast

import numpy as np
import pandas as pd

from agent_eda._columns import Labels, first_positions
from agent_eda._registry import analysis
from agent_eda._run import data_shape, execution
from agent_eda._text import plural
from agent_eda.results import (
    ColumnSupport,
    DescribeDatasetColumn,
    DescribeDatasetIndex,
    DescribeDatasetResult,
    Finding,
    Limitation,
    Provenance,
    Support,
)


@analysis(
    statuses=("completed", "failed"),
    finding_codes=(
        "empty_data",
        "duplicated_column_labels",
        "colliding_column_labels",
        "multiindex_columns",
        "non_unique_index",
        "all_missing_columns",
        "constant_columns",
    ),
    tags=(
        "overview",
        "shape",
        "size",
        "dtypes",
        "memory",
        "index",
        "provenance",
        "column labels",
        "duplicated column labels",
    ),
    any_column_labels=True,
)
def describe_dataset(
    data: pd.DataFrame,
    *,
    source: str | None = None,
    source_version: str | None = None,
) -> DescribeDatasetResult:
    """What does this dataset contain?

    When to use:
        Call it first on any new DataFrame. It reports the shape, each column's
        dtype, missing and distinct counts and memory, the index and the
        provenance you supply, and flags structural problems: empty data,
        duplicated or colliding column labels, MultiIndex columns, a non-unique
        index, all-missing columns and constant columns. Every other analysis
        raises on bad column labels, so run this one to find them. It never
        shows a raw data value. For how the values of each numeric column are
        distributed, call `summarize_numeric` next. `list_analyses` lists the
        other analyses.

    Args:
        data (pd.DataFrame): The data to describe. Any column labels are
            accepted, including duplicated ones and MultiIndex columns. It is
            never modified.
        source (str | None): Where the data came from, such as a file path, URL
            or table name, recorded exactly as given. None, the default, records
            the source as not supplied; it is never inferred.
        source_version (str | None): The version of the source, such as a
            snapshot date or a commit, recorded exactly as given. None, the
            default, records it as not supplied.

    Returns:
        DescribeDatasetResult: Read `row_count`, `column_count` and
        `memory_bytes` (columns plus index) for the size. `columns` has one
        entry per column in DataFrame order with its `dtype` as pandas reports
        it, `non_missing_count`, `missing_count` (what pandas isna() counts),
        `distinct_count` (distinct non-missing values; null, with an
        `unhashable_values` limitation, when values such as lists can't be
        hashed) and `memory_bytes`. `index` holds the index dtype, whether its
        labels are unique and how many are missing. `provenance` holds the
        source and version as supplied, and `not_supplied` names what wasn't.
        Memory counts the values only, not pandas' lookup tables or the UTF-8
        copies Python caches inside strings, so earlier work on the data
        doesn't change it. `to_frame()` returns the per-column table.

        Every finding is suspected:

        - `empty_data`: the data has no rows, no columns or neither.
        - `duplicated_column_labels`: a column label occurs more than once.
        - `colliding_column_labels`: different labels share a string form,
          such as 1 and '1'.
        - `multiindex_columns`: the column labels form a MultiIndex.
        - `non_unique_index`: index labels repeat, so they don't identify rows.
        - `all_missing_columns`: a column holds only missing values.
        - `constant_columns`: a column holds a single distinct non-missing value
          in at least two rows.

    Statuses:
        completed: The description finished, also for data with no rows, no
            columns or neither.
        failed: An anticipated computation failure stopped the description, and
            `status_reason` gives its code and message. None is known today: a
            count that can't be computed is null, with a limitation, instead.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, 51, 29, 51],
        ...         "score": [0.5, None, 0.75, 0.5],
        ...         "plan": [1] * 4,
        ...     }
        ... )
        >>> res = ae.describe_dataset(df, source="survey.csv")
        >>> print(res)
        describe_dataset: completed on 4 rows x 3 columns
        Memory: ... in total. Index: int64, unique, 0 missing labels.
        Provenance: source 'survey.csv'; source_version not supplied.
        Columns (3 of 3 shown):
        column  dtype    non_missing_count  missing_count  distinct_count  memory_bytes
        age     int64                    4              0               3            32
        score   float64                  3              1               2            32
        plan    int64                    4              0               1            32
        Findings (1 of 1 shown):
        - suspected constant_columns: 1 column holds a single distinct non-missing value: 'plan'.
        Limitations: none
        More: res.to_frame() has all 3 columns; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns[1].missing_count
        1

    See Also:
        summarize_numeric: Counts, mean, spread, quantiles and shape of every
            numeric column.
        list_analyses: Every analysis, with the question each one answers.
    """
    row_count = len(data)
    labels = Labels(data.columns)
    columns = [
        _describe_column(data.iloc[:, position], position, label, row_count)
        for position, label in enumerate(labels.labels)
    ]
    index = _describe_index(data.index)
    findings = [
        *_empty_data(row_count, len(columns)),
        *_duplicated_labels(labels),
        *_colliding_labels(labels),
        *_multiindex_columns(labels.level_count),
        *_non_unique_index(index),
        *_all_missing_columns(columns, labels, row_count),
        *_constant_columns(columns, labels),
    ]
    return DescribeDatasetResult(
        analysis="describe_dataset",
        method=None,
        status="completed",
        status_reason=None,
        parameters={
            "data": data_shape(data),
            "source": source,
            "source_version": source_version,
        },
        support=Support(
            row_count=row_count,
            column_count=len(columns),
            columns=[
                ColumnSupport(
                    column=c.column,
                    eligible_count=row_count,
                    evaluated_count=row_count,
                    excluded_count=0,
                    excluded_by_reason={},
                )
                for c in columns
            ],
        ),
        findings=findings,
        limitations=_unhashable_values(columns, labels),
        execution=execution(("numpy", "pandas")),
        row_count=row_count,
        column_count=len(columns),
        column_level_count=labels.level_count,
        memory_bytes=sum(c.memory_bytes for c in columns) + index.memory_bytes,
        columns=columns,
        index=index,
        provenance=Provenance(
            source=source,
            source_version=source_version,
            not_supplied=[
                name
                for name, value in (
                    ("source", source),
                    ("source_version", source_version),
                )
                if value is None
            ],
        ),
    )


def _describe_column(
    series: pd.Series, position: int, label: object, row_count: int
) -> DescribeDatasetColumn:
    if _holds_python_objects(series.dtype):
        missing, distinct, memory = _object_column(
            np.asarray(series.array, dtype=object)
        )
    else:
        missing = int(series.isna().sum())
        distinct = _distinct_count(series)
        memory = _memory_bytes(series)
    return DescribeDatasetColumn(
        position=position,
        column=str(label),
        label_type=type(label).__name__,
        dtype=str(series.dtype),
        non_missing_count=row_count - missing,
        missing_count=missing,
        distinct_count=distinct,
        memory_bytes=memory,
    )


def _holds_python_objects(dtype: object) -> bool:
    """Whether values are stored as Python objects: object dtype or python strings."""
    if isinstance(dtype, pd.StringDtype):
        return dtype.storage == "python"
    return isinstance(dtype, np.dtype) and dtype.kind == "O"


def _object_column(objects: np.ndarray) -> tuple[int, int | None, int]:
    """The missing count, distinct count and bytes of a column of Python objects.

    One hashing pass gives all three. The distinct count is None when values
    such as lists can't be hashed.
    """
    try:
        codes, found = pd.factorize(objects)
    except TypeError:
        return int(pd.isna(objects).sum()), None, _object_bytes(objects)
    uniques = np.asarray(found, dtype=object)
    missing = codes < 0
    if pd.api.types.infer_dtype(uniques, skipna=False) == "string":
        # Equal strings take equal space, so each distinct one is sized once.
        counts = np.bincount(codes[~missing], minlength=len(uniques))
        sizes = int(_sizes(uniques) @ counts) + int(_sizes(objects[missing]).sum())
        memory = objects.nbytes + sizes
    else:
        memory = _object_bytes(objects)
    return int(missing.sum()), len(uniques), memory


def _distinct_count(series: pd.Series) -> int | None:
    """Distinct non-missing values, or None when the values can't be hashed."""
    try:
        return int(series.nunique(dropna=True))
    # Lists and dicts raise TypeError; nested pyarrow types raise
    # ArrowNotImplementedError, a NotImplementedError.
    except (TypeError, NotImplementedError):
        return None


def _describe_index(index: pd.Index) -> DescribeDatasetIndex:
    if isinstance(index, pd.MultiIndex):
        # MultiIndex.duplicated() works on the codes alone.
        duplicated = int(index.duplicated().sum())
        missing = int(np.any([codes == -1 for codes in index.codes], axis=0).sum())
    else:
        # Index.duplicated() first asks is_unique, which builds and keeps the
        # caller's lookup table; a Series over the same values doesn't.
        values = pd.Series(index.array, copy=False, dtype=index.dtype)
        duplicated = int(values.duplicated().sum())
        missing = int(index.isna().sum())
    return DescribeDatasetIndex(
        dtype=str(index.dtype),
        level_count=index.nlevels,
        is_unique=duplicated == 0,
        duplicated_count=duplicated,
        missing_count=missing,
        memory_bytes=_memory_bytes(index),
    )


def _memory_bytes(values: pd.Series | pd.Index) -> int:
    """Deep memory of the values in bytes.

    pandas' memory_usage() of an index or a categorical adds the hash tables
    that lookups build and keep, so its answer depends on what the caller did
    before. This counts the values alone, so earlier lookups don't change it.
    """
    if isinstance(values, pd.MultiIndex):
        levels = sum(_memory_bytes(level) for level in values.levels)
        return levels + sum(int(codes.nbytes) for codes in values.codes)
    if isinstance(values, pd.RangeIndex):
        return int(values.nbytes)
    if isinstance(values.dtype, pd.CategoricalDtype):
        categorical = cast("pd.Categorical", values.array)
        return int(categorical.codes.nbytes) + _memory_bytes(categorical.categories)
    if _holds_python_objects(values.dtype):
        return _object_bytes(np.asarray(values.array, dtype=object))
    if isinstance(values, pd.Index):
        # The explicit dtype stops pandas 3 from turning object text into str.
        values = pd.Series(values.array, copy=False, dtype=values.dtype)
    return int(values.memory_usage(deep=True, index=False))


def _object_bytes(objects: np.ndarray) -> int:
    """Bytes of an array of Python objects: a reference per value and each object."""
    return objects.nbytes + int(_sizes(objects).sum())


def _sizes(objects: np.ndarray) -> np.ndarray:
    """Each object's size in bytes, sys.getsizeof as pandas' memory_usage counts it.

    A non-ASCII str counts without the UTF-8 copy Python keeps inside it once
    anything has encoded it, such as pyarrow, pandas' hashing or this analysis:
    otherwise the same strings would grow from one call to the next. ASCII
    text is its own UTF-8 form.
    """
    count = len(objects)
    if pd.api.types.infer_dtype(objects, skipna=False) == "string" and all(
        map(str.isascii, objects)
    ):
        return np.fromiter(map(sys.getsizeof, objects), dtype=np.int64, count=count)
    return np.fromiter(map(_size, objects), dtype=np.int64, count=count)


def _size(value: object) -> int:
    if type(value) is str and not value.isascii():
        widest = ord(max(value))
        width = 1 if widest < 0x100 else 2 if widest < 0x10000 else 4
        return _TEXT_HEADER + (len(value) + 1) * width
    return sys.getsizeof(value)


# The bytes a non-ASCII str takes besides its characters, measured on one made
# here, which nothing has encoded: 2 one-byte characters and a terminating one.
_TEXT_HEADER = "".join(map(chr, (0xE9, 0xE9))).__sizeof__() - 3


def _empty_data(row_count: int, column_count: int) -> list[Finding]:
    missing = [
        part
        for part, count in (("no rows", row_count), ("no columns", column_count))
        if count == 0
    ]
    if not missing:
        return []
    return [
        Finding(
            code="empty_data",
            claim=f"The data has {' and '.join(missing)}.",
            assessment="suspected",
            columns=[],
            fields=["row_count", "column_count"],
        )
    ]


def _duplicated_labels(labels: Labels) -> list[Finding]:
    positions = labels.repeated()
    if not positions:
        return []
    first = first_positions(labels, positions)
    occurs = plural(len(first), "column label occurs", "column labels occur")
    return [
        Finding(
            code="duplicated_column_labels",
            claim=f"{occurs} more than once: {labels.show(first)}.",
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].column" for p in positions],
        )
    ]


def _colliding_labels(labels: Labels) -> list[Finding]:
    positions = labels.colliding()
    if not positions:
        return []
    return [
        Finding(
            code="colliding_column_labels",
            claim=(
                f"{len(positions):,} different column labels share a string form "
                f"with another label: {labels.show(positions)}."
            ),
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[
                field
                for p in positions
                for field in (f"columns[{p}].column", f"columns[{p}].label_type")
            ],
        )
    ]


def _multiindex_columns(level_count: int) -> list[Finding]:
    if level_count == 1:
        return []
    return [
        Finding(
            code="multiindex_columns",
            claim=f"The column labels form a MultiIndex with {level_count} levels.",
            assessment="suspected",
            columns=[],
            fields=["column_level_count"],
        )
    ]


def _non_unique_index(index: DescribeDatasetIndex) -> list[Finding]:
    if index.is_unique:
        return []
    repeat = plural(
        index.duplicated_count, "index label repeats", "index labels repeat"
    )
    return [
        Finding(
            code="non_unique_index",
            claim=(
                f"{repeat} an earlier label, so index labels don't identify "
                "rows uniquely."
            ),
            assessment="suspected",
            columns=[],
            fields=["index.is_unique", "index.duplicated_count"],
        )
    ]


def _all_missing_columns(
    columns: list[DescribeDatasetColumn], labels: Labels, row_count: int
) -> list[Finding]:
    if row_count == 0:
        return []
    positions = [c.position for c in columns if c.non_missing_count == 0]
    if not positions:
        return []
    contain = plural(len(positions), "column contains", "columns contain")
    return [
        Finding(
            code="all_missing_columns",
            claim=f"{contain} only missing values: {labels.show(positions)}.",
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].missing_count" for p in positions],
        )
    ]


def _unhashable_values(
    columns: list[DescribeDatasetColumn], labels: Labels
) -> list[Limitation]:
    positions = [c.position for c in columns if c.distinct_count is None]
    if not positions:
        return []
    hold = plural(len(positions), "column holds", "columns hold")
    its = "its" if len(positions) == 1 else "their"
    return [
        Limitation(
            code="unhashable_values",
            reason=(
                f"{hold} values that can't be hashed, such as lists or dicts, so "
                f"{its} distinct values weren't counted: {labels.show(positions)}."
            ),
            fields=[f"columns[{p}].distinct_count" for p in positions],
        )
    ]


def _constant_columns(
    columns: list[DescribeDatasetColumn], labels: Labels
) -> list[Finding]:
    positions = [
        c.position
        for c in columns
        if c.non_missing_count >= 2 and c.distinct_count == 1
    ]
    if not positions:
        return []
    hold = plural(len(positions), "column holds", "columns hold")
    return [
        Finding(
            code="constant_columns",
            claim=(
                f"{hold} a single distinct non-missing value: {labels.show(positions)}."
            ),
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].distinct_count" for p in positions],
        )
    ]
