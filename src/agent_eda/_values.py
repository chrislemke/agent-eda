"""How analyses read the values of a column or index and compare them.

Values compare as pandas compares them. Values that pandas can't hash, such as
lists, dicts and sets, compare by content through hashable stand-ins.
"""

from collections.abc import Hashable

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
        return frozenset(value)
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    try:
        hash(value)
    except TypeError:
        return (_UNHASHABLE, type(value).__qualname__, repr(value))
    return value
