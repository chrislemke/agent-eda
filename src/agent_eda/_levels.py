"""The levels of a column: each distinct non-missing value and how often it occurs.

summarize_categorical reports the levels of each column, and profile_target can
count a target's classes with the same implementation, so the two can't
disagree. Values count as the same level exactly when pandas' factorize and
value_counts treat them as equal, so 1, 1.0 and True are one level, while 'Yes'
and 'yes ' are two. Missing values, None, NaN, NaT and pd.NA, are no level.

Levels are ordered by count, descending, then by label, as LEVEL_ORDER states.
The order never follows hash or set iteration order.
"""

import heapq
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Result classes build their tables here, so this module imports nothing
    # that imports them.
    from agent_eda._columns import ColumnKind

# How levels are ordered, as results state it.
LEVEL_ORDER = (
    "Count descending, then label. Booleans and numbers come first, by value, "
    "with False and True as 0 and 1 and NaN last. Text follows, by Unicode code "
    "point, then labels of any other type, by type name and string form. Labels "
    "that still tie keep their order in the data."
)


@dataclass(frozen=True)
class Levels:
    """The levels of one column.

    The labels stay in the form pandas holds them, which for pyarrow-backed
    text is far smaller than Python strings, since a result keeps its levels
    for to_frame(). objects() gives them as Python objects when a step needs
    them.
    """

    # The distinct non-missing values, in the order the data first holds them,
    # or in category order for a categorical.
    values: pd.Index
    # How often each label occurs.
    counts: np.ndarray
    missing_count: int
    # The labels as a NumPy array of numbers or booleans that sorts them by
    # value, or None when the labels need Python to sort them.
    sort_values: np.ndarray | None
    # Whether every label is a str.
    text: bool
    # The categories of a categorical dtype that never occur, in category order.
    unused_categories: list[object]

    def label(self, position: int) -> object:
        """One label as Python holds it."""
        return stored_label(self.values[position])

    def objects(self, positions: np.ndarray | None = None) -> np.ndarray:
        """The labels, or those at the given positions, as Python objects."""
        values = self.values if positions is None else self.values[positions]
        return np.asarray(values, dtype=object)


def count_levels(series: pd.Series, kind: "ColumnKind") -> Levels | None:
    """Count the levels of a column; None when its values can't be hashed."""
    if kind == "categorical":
        return _count_categories(series)
    try:
        codes, uniques = pd.factorize(series)
    # Lists and dicts raise TypeError; nested pyarrow types raise
    # ArrowNotImplementedError, a NotImplementedError.
    except (TypeError, NotImplementedError):
        return None
    # A Series gives an Index; an array would be wrapped without inferring a
    # dtype, which on pandas 3 would turn object text into str.
    values = (
        uniques if isinstance(uniques, pd.Index) else pd.Index(uniques, dtype=object)
    )
    # Code -1 marks a missing value, so the first count is the missing one.
    counts = np.bincount(codes + 1, minlength=len(values) + 1)
    return Levels(
        values=values,
        counts=counts[1:],
        missing_count=int(counts[0]),
        sort_values=_numpy_values(values) if kind in ("numeric", "boolean") else None,
        text=kind == "text" or _all_text(values),
        unused_categories=[],
    )


def _count_categories(series: pd.Series) -> Levels:
    categorical = cast("pd.Categorical", series.array)
    categories = categorical.categories
    codes = np.asarray(categorical.codes, dtype=np.intp)
    counts = np.bincount(codes + 1, minlength=len(categories) + 1)
    used = counts[1:] > 0
    values = categories[used]
    return Levels(
        values=values,
        counts=counts[1:][used],
        missing_count=int(counts[0]),
        sort_values=_numpy_values(values),
        text=_all_text(values),
        unused_categories=categories[~used].tolist(),
    )


def _numpy_values(values: pd.Index) -> np.ndarray | None:
    """The values as a NumPy array of numbers or booleans, if they are such."""
    dtype = values.dtype
    if getattr(dtype, "kind", "O") not in "biuf":
        return None
    # Nullable dtypes name the NumPy dtype of their values; the values hold no
    # missing value, since those are no level.
    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    array = values.to_numpy(dtype=numpy_dtype) if numpy_dtype else values.to_numpy()
    return array if array.dtype.kind in "biuf" else None


def _all_text(values: pd.Index) -> bool:
    return pd.api.types.infer_dtype(values, skipna=False) == "string"


def ordered(levels: Levels, limit: int | None = None) -> np.ndarray:
    """Positions of the levels in level order, only the first `limit` when given.

    Finding the first few levels never sorts every label, which is slow for
    text with a million distinct values.
    """
    counts = levels.counts
    size = len(counts)
    if limit is None or limit >= size:
        return _full_order(levels)
    # The limit-th largest count: fewer than `limit` levels have more.
    cut = np.partition(counts, size - limit)[size - limit]
    above = np.flatnonzero(counts > cut)
    head = sorted(above.tolist(), key=_level_key(levels))
    tied = np.flatnonzero(counts == cut)
    tail = _first_labels(levels, tied, limit - len(head))
    return np.asarray([*head, *tail], dtype=np.intp)


def _full_order(levels: Levels) -> np.ndarray:
    size = len(levels.counts)
    if levels.sort_values is not None:
        by_label = np.argsort(levels.sort_values, kind="stable")
    elif levels.text:
        by_label = np.argsort(levels.objects(), kind="stable")
    else:
        keys = [label_key(label) for label in levels.objects()]
        by_label = np.asarray(sorted(range(size), key=keys.__getitem__), dtype=np.intp)
    ranks = np.empty(size, dtype=np.intp)
    ranks[by_label] = np.arange(size)
    # lexsort sorts by the last key first: count descending, then label.
    return np.lexsort((ranks, -levels.counts))


def _level_key(levels: Levels) -> Callable[[int], tuple]:
    counts = levels.counts
    return lambda position: (
        -counts[position],
        label_key(levels.label(position)),
        position,
    )


def _first_labels(levels: Levels, positions: np.ndarray, count: int) -> list[int]:
    """The `count` positions whose labels come first in label order."""
    if levels.sort_values is not None:
        values = levels.sort_values[positions]
        first = np.arange(len(positions))
        if count < len(positions):
            first = np.argpartition(values, count - 1)[:count]
        first = first[np.argsort(values[first], kind="stable")]
        return positions[first].tolist()
    candidates = levels.objects(positions)
    if not levels.text:
        chosen = heapq.nsmallest(
            count, range(len(positions)), key=lambda i: label_key(candidates[i])
        )
        return positions[chosen].tolist()
    if count < len(positions):
        # A column's labels are distinct, so exactly `count` labels are at most
        # the count-th smallest. Comparing strings without a key is faster.
        last = heapq.nsmallest(count, candidates.tolist())[-1]
        keep = candidates <= last
        positions, candidates = positions[keep], candidates[keep]
    order = np.argsort(candidates, kind="stable")
    return positions[order].tolist()


def variant_groups(levels: Levels) -> list[list[int]]:
    """Positions of text labels that differ only by case or surrounding whitespace.

    Labels match once str.strip() and str.casefold() make them equal, so 'Yes',
    'yes ' and 'YES' form one group. Each group lists its labels in level order,
    and the groups follow the level order of their first label.
    """
    if levels.text:
        positions = np.arange(len(levels.values))
        texts = levels.objects()
    else:
        labels = levels.objects()
        positions = np.flatnonzero([isinstance(label, str) for label in labels])
        texts = labels[positions]
    if len(positions) < 2:
        return []
    keys = np.array([text.strip().casefold() for text in texts], dtype=object)
    # Labels already equal to their key can't match each other, so most text
    # needs no further work.
    if not (keys != texts).any():
        return []
    codes, _ = pd.factorize(keys)
    shared = np.bincount(codes)[codes] >= 2
    groups: dict[int, list[int]] = {}
    for code, position in zip(
        codes[shared].tolist(), positions[shared].tolist(), strict=True
    ):
        groups.setdefault(code, []).append(position)
    key = _level_key(levels)
    return sorted(
        (sorted(group, key=key) for group in groups.values()),
        key=lambda group: key(group[0]),
    )


def label_key(label: object) -> tuple:
    """The key that sorts labels of any type in the documented order."""
    value = stored_label(label)
    if isinstance(value, bool | int | float | Decimal | Fraction):
        if _is_nan(value):
            return (0, 1, 0)
        return (0, 0, value)
    if isinstance(value, str):
        return (1, value)
    return (2, type(value).__name__, str(value))


def _is_nan(value: object) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return isinstance(value, Decimal) and value.is_nan()


def stored_label(label: object) -> object:
    """A label as Python holds it: NumPy scalars become the Python scalars they hold.

    NumPy datetimes and durations stay as they are, since item() can turn them
    into integers.
    """
    if isinstance(label, np.timedelta64 | np.datetime64):
        return label
    if isinstance(label, np.number | np.bool_ | np.str_ | np.bytes_):
        return label.item()
    return label


def record_label(label: object) -> str | bool | int | float:
    """A label as the record holds it: a JSON scalar, or else its string form."""
    value = stored_label(label)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    return str(value)


def type_name(label: object) -> str:
    """The name of a label's Python type, such as 'int' for numpy.int64 too."""
    return _name_of(type(label))


def levels_frame(tables: Mapping[str, Levels]) -> pd.DataFrame:
    """Every level of every column: column order first, then level order.

    Labels stay as stored, NumPy scalars as the Python scalars they hold, and
    each share divides by the column's non-missing values.
    """
    columns: list[str] = []
    labels: list[object] = []
    types: list[str] = []
    counts: list[np.ndarray] = []
    shares: list[np.ndarray] = []
    for name, levels in tables.items():
        order = ordered(levels)
        ordered_counts = levels.counts[order].astype(np.int64)
        ordered_labels = [stored_label(label) for label in levels.objects(order)]
        columns.extend([name] * len(order))
        labels.extend(ordered_labels)
        types.extend(_type_names(levels, ordered_labels))
        counts.append(ordered_counts)
        total = int(ordered_counts.sum())
        shares.append(ordered_counts / total if total else ordered_counts * 0.0)
    return pd.DataFrame(
        {
            "column": pd.Series(columns, dtype=object if not columns else None),
            "label": pd.Series(labels, dtype=object),
            "label_type": pd.Series(types, dtype=object if not types else None),
            "count": np.concatenate([np.empty(0, np.int64), *counts]),
            "share": np.concatenate([np.empty(0, np.float64), *shares]),
        }
    )


def _type_names(levels: Levels, labels: list[object]) -> list[str]:
    """The type name of each label, without asking each label when one name fits all."""
    if levels.text:
        return ["str"] * len(labels)
    if levels.sort_values is not None:
        return [_name_of(levels.sort_values.dtype.type)] * len(labels)
    return [type_name(label) for label in labels]


def _name_of(kind: type) -> str:
    """A type's name, a NumPy scalar type named as the Python type it holds."""
    for numpy_type, name in _NUMPY_TYPE_NAMES:
        if issubclass(kind, numpy_type):
            return name
    return kind.__name__


# NumPy scalar types and the Python type their item() gives; durations first,
# since NumPy makes them a kind of integer. Datetimes and durations keep their
# names, since item() can turn them into integers.
_NUMPY_TYPE_NAMES: tuple[tuple[type, str], ...] = (
    (np.timedelta64, "timedelta64"),
    (np.datetime64, "datetime64"),
    (np.bool_, "bool"),
    (np.integer, "int"),
    (np.floating, "float"),
    (np.complexfloating, "complex"),
    (np.str_, "str"),
    (np.bytes_, "bytes"),
)


def value_types(
    series: pd.Series, kind: "ColumnKind", levels: Levels | None
) -> list[str]:
    """The names of the Python types of a column's non-missing values, sorted.

    Values that pandas counts as one level can still differ in type, such as 1,
    1.0 and True, so a column of objects that aren't all text is checked value
    by value. summarize_categorical and infer_column_semantics both call this,
    and more than one name means mixed value types to both.
    """
    if levels is not None:
        if not len(levels.values):
            return []
        if levels.text:
            return ["str"]
        if levels.sort_values is not None:
            return [_name_of(levels.sort_values.dtype.type)]
        if kind in ("datetime", "timedelta"):
            # A datetime or duration dtype gives every value one type, so the
            # first names them all without converting a million of them.
            return [type_name(levels.label(0))]
        if kind != "object":
            return sorted({type_name(label) for label in levels.objects()})
    values = np.asarray(series.array, dtype=object)
    present = values[~pd.isna(values)]
    return sorted({_name_of(value_type) for value_type in set(map(type, present))})
