"""Generated DataFrames full of edge cases, for the property tests.

data_frames() draws up to MAX_ROWS rows and MAX_COLUMNS columns, from no rows
and no columns up. Each column has a dtype from KINDS and a shape: any values,
a few values with ties, one constant value, or only missing values. Column
labels are unique and distinct as strings, but can be integers, long, blank or
full of control characters; with any_column_labels=True they can also repeat,
collide as strings, be missing or form a MultiIndex. The row index varies too.

EDGE_FRAMES holds fixed DataFrames, such as one column of every kind, so that
every run covers each dtype and shape whatever Hypothesis draws;
LABEL_EDGE_FRAMES does the same for column labels that can't serve as names.
Everything here works on pandas 2.2 and 3.x.
"""

import datetime as dt
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from hypothesis import strategies as st
from pandas.api.extensions import ExtensionDtype

MAX_ROWS = 8
MAX_COLUMNS = 6

NAN = float("nan")
INF = float("inf")


def _str_dtype() -> pd.StringDtype:
    """pandas 3's default text dtype, str, or its closest twin on pandas 2.2."""
    try:  # pandas 2.3 and later
        return pd.StringDtype("pyarrow", na_value=np.nan)
    except TypeError:  # pandas 2.2 has no na_value
        return pd.StringDtype("pyarrow_numpy")


STR = _str_dtype()

# Text agents meet: empty and whitespace-only strings, case and space variants,
# stand-ins for missing values, characters of 1, 2 and 4 bytes, and any other
# short string. Each listed string is one object that columns and examples share.
TEXT = st.sampled_from(
    ["", " ", "  ", "a", "A", "a ", "Yes", "yes ", "N/A", "-999", "café", "日本", "🙂"]
) | st.text(max_size=5)


def floats(
    *, width: Literal[32, 64] = 64, allow_nan: bool = True
) -> st.SearchStrategy[float]:
    """Any float, with ±inf, ±0.0, NaN and the extremes coming up often."""
    special = [INF, -INF, 0.0, -0.0, 1.0, -1.0]
    if allow_nan:
        special.append(NAN)
    if width == 64:
        special.extend([1.7e308, -1.7e308, 5e-324])
    return st.sampled_from(special) | st.floats(width=width, allow_nan=allow_nan)


FLOATS = floats()
INTS = st.integers(-(2**63), 2**63 - 1)
DATETIMES = st.datetimes(dt.datetime(1900, 1, 1), dt.datetime(2200, 1, 1))
DATES = st.dates(dt.date(1900, 1, 1), dt.date(2200, 1, 1))
TIMEDELTAS = st.timedeltas(dt.timedelta(days=-1000), dt.timedelta(days=1000))
DECIMALS = st.decimals(-(10**6), 10**6, places=2, allow_nan=False)


def _series(dtype: str | type | ExtensionDtype) -> Callable[[list], pd.Series]:
    return lambda values: pd.Series(values, dtype=dtype)


def _arrow(kind: pa.DataType) -> Callable[[list], pd.Series]:
    return _series(pd.ArrowDtype(kind))


def _nan_as_value(values: list) -> pd.Series:
    """Float64 storing NaN as a value, as arithmetic leaves it on pandas 2."""
    mask = np.array([v is None for v in values], dtype=bool)
    data = np.array([0.0 if v is None else v for v in values], dtype="float64")
    return pd.Series(pd.arrays.FloatingArray(data, mask))


def _arrow_floats(values: list) -> pd.Series:
    """double[pyarrow] keeping NaN as a value apart from null, which is None."""
    stored = pa.array(values, type=pa.float64(), from_pandas=False)
    return pd.Series(pd.arrays.ArrowExtensionArray(stored))


def _sparse(values: list) -> pd.Series:
    floats = np.array([NAN if v is None else v for v in values], dtype="float64")
    return pd.Series(pd.arrays.SparseArray(floats))


def _datetimes(values: list) -> pd.Series:
    return pd.Series(values, dtype="datetime64[ns]")


def _categorical(
    categories: list, *, ordered: bool = False
) -> Callable[[list], pd.Series]:
    return lambda values: pd.Series(
        pd.Categorical(values, categories=categories, ordered=ordered)
    )


@dataclass(frozen=True)
class Kind:
    """A kind of column: its values, how to build it and four fixed values."""

    # One value that isn't missing.
    elements: st.SearchStrategy
    # A Series from a list of values in which None means missing.
    build: Callable[[list], pd.Series]
    # Four values for the fixed frames, None where the dtype can be missing.
    sample: list
    # Whether the dtype can hold a missing value.
    missing: bool = True


_DAY = dt.datetime(2020, 1, 1)
_OLD = dt.datetime(1900, 1, 1)
_TEXTS = ["a", None, "", "  "]
_TIMES = [_DAY, None, _OLD, _DAY]

# Every kind of column the strategy generates, simplest first: Hypothesis
# shrinks failures toward the first.
KINDS: dict[str, Kind] = {
    "float64": Kind(FLOATS, _series("float64"), [1.5, None, INF, 1.5]),
    "float32": Kind(floats(width=32), _series("float32"), [1.5, None, -INF, 0.0]),
    "Float64": Kind(floats(allow_nan=False), _series("Float64"), [1.5, None, INF, 1.5]),
    "Float64 with NaN as a value": Kind(FLOATS, _nan_as_value, [1.5, NAN, None, 1.5]),
    "double[pyarrow]": Kind(FLOATS, _arrow_floats, [1.5, NAN, None, INF]),
    "Sparse[float64]": Kind(FLOATS, _sparse, [0.0, None, 1.5, 0.0]),
    "int64": Kind(INTS, _series("int64"), [1, 3, -(2**63), 2**63 - 1], missing=False),
    "uint64": Kind(
        st.integers(0, 2**64 - 1),
        _series("uint64"),
        [0, 3, 2**64 - 1, 3],
        missing=False,
    ),
    "Int64": Kind(INTS, _series("Int64"), [1, None, 2**60, 1]),
    "UInt8": Kind(st.integers(0, 255), _series("UInt8"), [1, None, 255, 1]),
    "int64[pyarrow]": Kind(INTS, _series("int64[pyarrow]"), [1, None, -5, 1]),
    "bool": Kind(
        st.booleans(), _series("bool"), [True, False, True, True], missing=False
    ),
    "boolean": Kind(st.booleans(), _series("boolean"), [True, None, False, True]),
    "bool[pyarrow]": Kind(
        st.booleans(), _series("bool[pyarrow]"), [True, None, False, True]
    ),
    "object text": Kind(TEXT | st.just(NAN), _series(object), _TEXTS),
    # The dtype pandas infers for text: object on pandas 2, str on pandas 3.
    "inferred text": Kind(TEXT, pd.Series, ["a", None, " ", "a"]),
    "str": Kind(TEXT, _series(STR), _TEXTS),
    "string": Kind(TEXT, _series("string"), _TEXTS),
    "string[pyarrow]": Kind(TEXT, _series("string[pyarrow]"), _TEXTS),
    "arrow string": Kind(TEXT, _arrow(pa.string()), _TEXTS),
    "object of mixed types": Kind(
        TEXT
        | st.integers(-3, 3)
        | FLOATS
        | st.booleans()
        | st.sampled_from([NAN, pd.NA, pd.NaT, Decimal("1.5")]),
        _series(object),
        [1, "1", None, True],
    ),
    "object of lists and dicts": Kind(
        st.lists(st.integers(0, 2), max_size=2)
        | st.dictionaries(st.sampled_from("ab"), st.integers(0, 2), max_size=1)
        | TEXT,
        _series(object),
        [["a"], None, {"k": 1}, ["a"]],
    ),
    # "unused" and 4 are categories that never occur.
    "category": Kind(
        st.sampled_from(["a", "b", "B", "b "]),
        _categorical(["a", "b", "B", "b ", "unused"]),
        ["a", None, "b", "b"],
    ),
    "ordered category of integers": Kind(
        st.sampled_from([1, 2, 3]),
        _categorical([1, 2, 3, 4], ordered=True),
        [3, None, 1, 3],
    ),
    "datetime64[ns]": Kind(DATETIMES, _datetimes, _TIMES),
    # The unit pandas picks: ns on pandas 2, us on pandas 3, s when all missing.
    "inferred datetime": Kind(
        DATETIMES, lambda v: pd.to_datetime(pd.Series(v, dtype=object)), _TIMES
    ),
    "datetime64[ns, UTC]": Kind(
        DATETIMES, lambda v: _datetimes(v).dt.tz_localize("UTC"), _TIMES
    ),
    "datetime64[ns, Europe/Berlin]": Kind(
        DATETIMES,
        lambda v: _datetimes(v).dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin"),
        _TIMES,
    ),
    "timestamp[us][pyarrow]": Kind(DATETIMES, _arrow(pa.timestamp("us")), _TIMES),
    "date32[pyarrow]": Kind(
        DATES,
        _arrow(pa.date32()),
        [dt.date(2020, 1, 1), None, dt.date(1970, 1, 1), dt.date(2020, 1, 1)],
    ),
    "timedelta64[ns]": Kind(
        TIMEDELTAS,
        _series("timedelta64[ns]"),
        [dt.timedelta(days=1), None, dt.timedelta(0), dt.timedelta(days=1)],
    ),
    "duration[us][pyarrow]": Kind(
        TIMEDELTAS,
        _arrow(pa.duration("us")),
        [dt.timedelta(days=1), None, dt.timedelta(0), dt.timedelta(days=1)],
    ),
    "decimal128[pyarrow]": Kind(
        DECIMALS,
        _arrow(pa.decimal128(10, 2)),
        [Decimal("1.50"), None, Decimal("-2.25"), Decimal("1.50")],
    ),
    "list<int64>[pyarrow]": Kind(
        st.lists(st.integers(-3, 3), max_size=2),
        _arrow(pa.list_(pa.int64())),
        [[1, 2], None, [], [1, 2]],
    ),
}

SHAPES = ("any", "any", "ties", "constant", "missing")
INDEX_KINDS = (
    "range",
    "shuffled",
    "repeated",
    "text",
    "float",
    "datetime",
    "period",
    "categorical",
    "multi",
)
LABEL_STYLES = ("name", "name", "name", "integer", "long", "control", "blank")
LABEL_PROBLEMS = ("none", "repeated", "colliding", "missing", "multiindex")
ATTRS: list[dict[Hashable, Any]] = [{}, {"source": "generated.csv", "tags": ["raw"]}]


@st.composite
def data_frames(draw: st.DrawFn, *, any_column_labels: bool = False) -> pd.DataFrame:
    """A DataFrame of up to MAX_ROWS rows and MAX_COLUMNS columns of any kind.

    With any_column_labels, the column labels can also repeat, collide as
    strings, be missing or form a MultiIndex, which only analyses that take
    such data accept.
    """
    row_count = draw(st.integers(0, MAX_ROWS))
    column_count = draw(st.integers(0, MAX_COLUMNS))
    kinds = draw(
        st.lists(
            st.sampled_from(list(KINDS)), min_size=column_count, max_size=column_count
        )
    )
    columns = [draw(column_values(KINDS[kind], row_count)) for kind in kinds]
    df = _frame(columns, row_count)
    if columns:
        df.columns = draw(column_labels(column_count, any_column_labels))
    # Set afterwards: pd.DataFrame(columns, index=...) would align on the index.
    df.index = draw(indexes(row_count))
    df.attrs = draw(st.sampled_from(ATTRS))
    return df


def _frame(columns: list[pd.Series], row_count: int) -> pd.DataFrame:
    """A DataFrame of the columns, labelled by position."""
    if not columns:
        return pd.DataFrame(index=pd.RangeIndex(row_count))
    return pd.DataFrame(dict(enumerate(columns)))


@st.composite
def column_values(draw: st.DrawFn, kind: Kind, row_count: int) -> pd.Series:
    """A column of the kind: any values, ties, one constant value or all missing."""
    shape = draw(st.sampled_from(SHAPES))
    if shape == "missing" and kind.missing:
        return kind.build([None] * row_count)
    if shape == "constant":
        return kind.build([draw(kind.elements)] * row_count)
    # Missing values come at half the rate of the others.
    values = kind.elements
    if kind.missing:
        values = st.one_of(kind.elements, kind.elements, st.none())
    if shape == "ties":
        values = st.sampled_from(draw(st.lists(values, min_size=1, max_size=2)))
    return kind.build(draw(st.lists(values, min_size=row_count, max_size=row_count)))


@st.composite
def column_labels(draw: st.DrawFn, count: int, any_column_labels: bool) -> pd.Index:
    """Column labels, unique and distinct as strings unless any_column_labels."""
    labels: list[object] = [
        _label(draw(st.sampled_from(LABEL_STYLES)), i) for i in range(count)
    ]
    problem = draw(st.sampled_from(LABEL_PROBLEMS)) if any_column_labels else "none"
    if problem == "multiindex":
        return pd.MultiIndex.from_tuples(
            [(f"g{i % 2}", str(x)) for i, x in enumerate(labels)]
        )
    if problem == "repeated" and count >= 2:
        labels[-1] = labels[0]
    elif problem == "colliding" and count >= 2:
        labels[0], labels[-1] = 1, "1"
    elif problem == "missing":
        labels[0] = draw(st.sampled_from([None, NAN]))
    return pd.Index(labels, dtype=object) if problem != "none" else pd.Index(labels)


def _label(style: str, position: int) -> object:
    """A label whose string form no other position's label shares."""
    if style == "integer":
        return position
    if style == "long":
        return f"c{position}_{'a_long_column_label_' * 8}"
    if style == "control":
        return f"c{position}\n\t\r\x00\x1b[31m\u2028é 日本"
    if style == "blank":
        return " " * position  # "" at position 0
    return f"c{position}"


@st.composite
def indexes(draw: st.DrawFn, row_count: int) -> pd.Index:
    """A row index: unique or repeated labels, missing labels, or several levels."""
    kind = draw(st.sampled_from(INDEX_KINDS))
    order = draw(st.permutations(range(row_count)))
    if kind == "shuffled":
        return pd.Index(order, dtype="int64")
    if kind == "repeated":
        return pd.Index([i // 2 for i in range(row_count)], dtype="int64")
    if kind == "text":
        return pd.Index([f"r{i}" for i in order], name="row")
    if kind == "float":  # NaN is a missing label
        return pd.Index([float(i) if i else NAN for i in order], dtype="float64")
    if kind == "datetime":
        days = pd.to_timedelta(list(order), unit="D")
        return pd.DatetimeIndex(pd.Timestamp("2024-01-01") + days).where(
            [i != 1 for i in order]
        )
    if kind == "period":
        return pd.PeriodIndex([f"2024-{i % 12 + 1:02d}" for i in order], freq="M")
    if kind == "categorical":  # "z" never occurs; None is a missing label
        return pd.CategoricalIndex(
            [("a", "b", None)[i % 3] for i in order], categories=["a", "b", "z"]
        )
    if kind == "multi":
        return pd.MultiIndex.from_arrays(
            [[f"g{i % 2}" for i in order], [i if i != 1 else None for i in order]],
            names=["group", "item"],
        )
    return pd.RangeIndex(row_count)


def _shared_strings() -> pd.DataFrame:
    """Object columns sharing new non-ASCII string objects.

    Hashing a column of text alone can make Python keep a UTF-8 copy inside
    each string, which then grows the strings in every column that holds them.
    """
    texts = ["".join(["caf", "é"]), "".join(["日", "本"]), "".join(["a", "🙂"])]
    return pd.DataFrame(
        {
            "missing": pd.Series([texts[0], None, texts[1], texts[2]], dtype=object),
            "text": pd.Series([texts[0], texts[1], texts[2], texts[0]], dtype=object),
            "mixed": pd.Series([1, texts[2], None, texts[0]], dtype=object),
        }
    )


def _every_kind(values: Callable[[Kind], list], copies: int = 1) -> pd.DataFrame:
    """A column of every kind, `copies` times; later copies get a number."""
    return pd.DataFrame(
        {
            f"{name} {copy}" if copy else name: kind.build(values(kind))
            for copy in range(copies)
            for name, kind in KINDS.items()
        }
    )


# Fixed DataFrames every run checks, whatever Hypothesis draws.
EDGE_FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "nothing": pd.DataFrame,
    "no rows": lambda: _every_kind(lambda kind: []),
    "no columns": lambda: pd.DataFrame(index=pd.RangeIndex(3)),
    "one row": lambda: _every_kind(lambda kind: kind.sample[:1]),
    "every kind": lambda: _every_kind(lambda kind: kind.sample),
    # More columns than a printed table shows (10) and a record keeps (100).
    "many columns": lambda: _every_kind(lambda kind: kind.sample, copies=4),
    "constant": lambda: _every_kind(lambda kind: [kind.sample[0]] * 3),
    "all missing": lambda: pd.DataFrame(
        {name: kind.build([None] * 3) for name, kind in KINDS.items() if kind.missing}
    ),
    "shared strings": _shared_strings,
}
# Fixed DataFrames whose column labels can't serve as names.
LABEL_EDGE_FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "repeated labels": lambda: pd.DataFrame(
        [[1, None, 1]] * 2, columns=["a", "b", "a"]
    ),
    "colliding labels": lambda: pd.DataFrame([[1, 2, 3]], columns=[1, "1", "b"]),
    "missing labels": lambda: pd.DataFrame(
        [[1, 2]], columns=pd.Index([None, NAN], dtype=object)
    ),
    "multiindex columns": lambda: pd.DataFrame(
        [[1.0, 2.0]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    ),
}
