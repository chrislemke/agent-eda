"""How a row ID or a label is written: the same in every process.

Python lists a set's items by hash, and hashing text is seeded per process, so
str() of a set differs between runs. An object's default repr holds its memory
address, which differs too. The record writes a row ID as JSON with row_id_json
and a label as text with canonical_text, and both order a set's items by their
JSON text, so one value reads the same wherever it shows up.
"""

import datetime as dt
import json
import math
import re

import numpy as np
import pandas as pd
from pydantic import JsonValue

# The end of a repr that names a memory address, such as "<Foo object at 0x7f3a>".
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>$")


def row_id_list(ids: object) -> list[object]:
    """Row IDs held as pandas labels or as a list, as a list of Python objects.

    A result keeps the IDs of many rows as a pandas Index or Series, which holds
    them far more compactly, and lists them only when a table asks.
    """
    tolist = getattr(ids, "tolist", None)
    return tolist() if tolist is not None else list(ids)  # ty: ignore[invalid-argument-type] # ids is a pandas object or a list


def row_id_json(value: object) -> JsonValue:
    """A row ID as JSON, the same in every process and with every library version.

    Timestamps, dates and times become ISO 8601 strings, durations ISO 8601
    durations, MultiIndex tuples lists, and NumPy scalars Python numbers.
    Missing values become null, and infinite floats the strings 'inf' and
    '-inf'. A set becomes a list ordered by its items' JSON, not by Python's
    hash order, and a dict a JSON object. Anything else becomes its str(), such
    as a Period, a Decimal, or a duration in years or months, which has no ISO
    form pandas can write; an object whose repr holds a memory address becomes
    '<TypeName object>'.
    """
    if isinstance(value, tuple | list):
        return [row_id_json(item) for item in value]
    if isinstance(value, set | frozenset):
        return [row_id_json(item) for item in _in_set_order(value)]
    if isinstance(value, dict):
        return {str(key): row_id_json(item) for key, item in value.items()}
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    # Before the integers: NumPy's timedelta64 is one of its integer types.
    if isinstance(value, dt.timedelta | np.timedelta64):
        try:
            return pd.Timedelta(value).isoformat()
        except (ValueError, OverflowError):
            # A duration in years or months, or beyond what pandas holds.
            return str(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else repr(number)
    return canonical_text(value)


def canonical_text(value: object) -> str:
    """A value as text, the same in every process: str() with its sets in order.

    A set or frozenset lists its items by their JSON text, as row_id_json does,
    also inside tuples, lists and dicts, so frozenset({'b', 'a'}) reads
    "frozenset({'a', 'b'})" in any process. An object whose repr holds a memory
    address reads "<TypeName object>". A str stays as it is, and anything else
    is its str().
    """
    if isinstance(value, str):
        return str(value)
    if type(value) in _CONTAINERS:
        return _written(value)
    return _without_address(value, str(value))


def _written(value: object) -> str:
    """repr() of a value, with its sets in order and its addresses left out."""
    if isinstance(value, set | frozenset) and type(value) in _CONTAINERS:
        items = ", ".join(_written(item) for item in _in_set_order(value))
        if isinstance(value, frozenset):
            return f"frozenset({{{items}}})" if items else "frozenset()"
        return f"{{{items}}}" if items else "set()"
    if type(value) is tuple:
        items = ", ".join(_written(item) for item in value)
        return f"({items},)" if len(value) == 1 else f"({items})"
    if type(value) is list:
        return "[" + ", ".join(_written(item) for item in value) + "]"
    if type(value) is dict:
        pairs = (f"{_written(key)}: {_written(item)}" for key, item in value.items())
        return "{" + ", ".join(pairs) + "}"
    return _without_address(value, repr(value))


_CONTAINERS = (set, frozenset, tuple, list, dict)


def _without_address(value: object, text: str) -> str:
    if text.startswith("<") and _ADDRESS.search(text):
        return f"<{type(value).__name__} object>"
    return text


def _in_set_order(items: set | frozenset) -> list:
    """The items of a set in an order that doesn't depend on the process.

    By JSON text, then type name and written form for the items whose JSON
    reads the same, such as Decimal('1') and '1'.
    """
    return sorted(
        items,
        key=lambda item: (
            json.dumps(row_id_json(item), sort_keys=True),
            type(item).__name__,
            _written(item),
        ),
    )
