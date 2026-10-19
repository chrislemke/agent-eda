"""How analyses read the values of a column or index and compare them.

Values compare as pandas compares them, so 1, 1.0 and True are one value, with
one exception: text. pandas' factorize reads '\\x00' in text as the end of the
text, so '' and '\\x00a' would be one value, which Python's == tells apart.
value_codes() is the one place that numbers the distinct values of a column,
and it keeps text apart. Every analysis that counts levels, groups, classes or
keys calls it, so they can't disagree.

Values that pandas can't hash, such as lists, dicts and sets, are the analysis's
choice: they compare by content through hashable stand-ins, or the column is
reported as unsupported.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray


def column_values(values: pd.Series | pd.Index) -> np.ndarray | ExtensionArray:
    """The values of a column or index as pd.factorize takes them, without a copy.

    That is a NumPy array for a NumPy dtype and pandas' extension array
    otherwise. pandas 2.2 warns when pd.factorize gets the wrapper around a
    NumPy array that Series.array returns for a NumPy dtype. The array can be a
    read-only view of the caller's data: never write to it.
    """
    if isinstance(values.dtype, np.dtype):
        return values.to_numpy()
    return values.array


@dataclass(frozen=True)
class ValueCodes:
    """The distinct values of a column and which of them each row holds."""

    # One code per row: the position of the row's value in `uniques`, -1 where
    # the value is missing (None, NaN, NaT or pd.NA).
    codes: np.ndarray
    # The distinct non-missing values, in order of first appearance. They are
    # the hashable stand-ins of the values when `by_content` is set.
    uniques: "pd.Index | np.ndarray | ExtensionArray"
    # Whether values that can't be hashed, such as lists, were compared by
    # their content.
    by_content: bool


# What value_codes does with values pandas can't hash: "by_content" compares
# them by content, "unsupported" gives no codes.
Unhashable = Literal["by_content", "unsupported"]


@overload
def value_codes(
    values: "pd.Series | pd.Index | np.ndarray | ExtensionArray",
    *,
    unhashable: Literal["by_content"],
) -> ValueCodes: ...


@overload
def value_codes(
    values: "pd.Series | pd.Index | np.ndarray | ExtensionArray",
    *,
    unhashable: Unhashable,
) -> ValueCodes | None: ...


def value_codes(
    values: "pd.Series | pd.Index | np.ndarray | ExtensionArray",
    *,
    unhashable: Unhashable,
) -> ValueCodes | None:
    """Number the distinct values of a column, index or array.

    Values are equal when pandas' factorize says so, except that text holding
    '\\x00' stays apart from text that is equal up to it. A value Python can't
    hash is tested by hashing, whatever else the column holds, so a column of
    integers and lists is caught. With unhashable="by_content" such values
    compare by content, as hashable_values() makes them; with "unsupported" the
    result is None. The caller's data isn't changed.
    """
    try:
        codes, uniques = _factorize(values)
    # Lists and dicts raise TypeError; nested pyarrow types raise
    # ArrowNotImplementedError, a NotImplementedError.
    except (TypeError, NotImplementedError):
        if unhashable == "unsupported":
            return None
        array = (
            column_values(values)
            if isinstance(values, pd.Series | pd.Index)
            else values
        )
        codes, uniques = _factorize(hashable_values(array))
        return ValueCodes(codes, uniques, by_content=True)
    return ValueCodes(codes, uniques, by_content=False)


def _factorize(
    values: "pd.Series | pd.Index | np.ndarray | ExtensionArray",
) -> "tuple[np.ndarray, pd.Index | np.ndarray | ExtensionArray]":
    objects = _all_text_objects(values)
    if objects is None or not _holds_nul(objects):
        return pd.factorize(values)
    # A value that isn't text keeps pandas off its hash table for text, which
    # reads '\x00' as the end of the text. The marker comes last, so no value
    # gets its code, and its place among the distinct values is the last.
    marked = np.empty(len(objects) + 1, dtype=object)
    marked[:-1] = objects
    marked[-1] = _NOT_TEXT
    codes, uniques = pd.factorize(marked)
    uniques = uniques[:-1]
    if isinstance(values, pd.Series | pd.Index):
        uniques = pd.Index(uniques, dtype=values.dtype)
    return codes[:-1], uniques


# value_codes appends this to text that holds '\x00', so the values are never
# all text.
_NOT_TEXT = object()
# Texts joined at a time when looking for '\x00'.
_CHUNK = 65_536


def _all_text_objects(
    values: "pd.Series | pd.Index | np.ndarray | ExtensionArray",
) -> np.ndarray | None:
    """The values as an array of Python objects if they are all text, else None.

    Only Python objects can hold such text: object dtype, and on pandas 2.2 the
    python-backed string dtype. Missing values aren't text, and keep pandas off
    its hash table for text.
    """
    dtype = values.dtype
    if isinstance(values, pd.Series | pd.Index):
        values = values.array
    if isinstance(dtype, pd.StringDtype):
        if dtype.storage != "python":
            return None
    elif dtype is not np.dtype(object):
        return None
    objects = np.asarray(values, dtype=object)
    if pd.api.types.infer_dtype(objects, skipna=False) != "string":
        return None
    return objects


def _holds_nul(objects: np.ndarray) -> bool:
    """Whether any of the texts holds '\\x00', found by joining them."""
    return any(
        "\x00" in "".join(objects[start : start + _CHUNK])
        for start in range(0, len(objects), _CHUNK)
    )


def hashable_values(values: np.ndarray | ExtensionArray) -> np.ndarray:
    """Each value in a form pandas can hash, as an object array.

    A value Python can hash stays as it is, so it compares exactly as pandas
    compares it, whatever else the array holds. A value it can't hash, such as
    a list, becomes a stand-in that equals another value's stand-in when the two
    have the same content.
    """
    return np.fromiter(map(_stand_in, values), dtype=object, count=len(values))


# Tags that keep the stand-ins of lists, dicts and other unhashable values apart
# from every value in the data.
_LIST = object()
_DICT = object()
_UNHASHABLE = object()


def _stand_in(value: object) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return _content(value)
    return value


def _content(value: object) -> Hashable:
    """A hashable stand-in for a value that compares by the value's content.

    Lists and NumPy arrays become tuples of their items' stand-ins, and dicts
    the set of their items, so the order of keys doesn't matter, as for ==.
    Sets become frozensets. Missing values inside them become None, so they
    compare equal. Any other value that can't be hashed compares by its type
    and repr.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
        if not isinstance(value, list):  # a 0-dimensional array holds one value
            return _content(value)
    if isinstance(value, list):
        return (_LIST, tuple(map(_content, value)))
    if isinstance(value, tuple):
        return tuple(map(_content, value))
    if isinstance(value, dict):
        return (_DICT, frozenset((key, _content(item)) for key, item in value.items()))
    if isinstance(value, set | frozenset):
        return frozenset(map(_content, value))
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    try:
        hash(value)
    except TypeError:
        return (_UNHASHABLE, type(value).__qualname__, repr(value))
    return value
