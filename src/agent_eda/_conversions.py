"""How the values of a column relate to a declared logical type.

validate_column_types sorts every value of a declared column into one of four
groups. Compatible values are already stored as the declared logical type.
Convertible values are stored as another type and convert to it exactly,
without changing their meaning. Invalid values do neither. Missing values are
None, NaN, NaT and pd.NA, and NaN that a float column stores as a value.
Nothing is converted.

A value is stored as a logical type by its Python type or its dtype: booleans
as boolean, ints as integer, floats as float, strings as text, datetimes and
dates as datetime, durations as timedelta, and the values of a categorical
dtype as categorical. A value converts:

- to boolean when it is a number equal to 0 or 1, or text that reads as a
  boolean;
- to integer when it is a whole number stored as a float or a decimal, or text
  that parses as a whole number;
- to float when it is any other number, or text that parses as a number;
- to datetime when it is text in ISO 8601 or in the declared datetime_format;
- to timedelta never: no text or number states its unit unambiguously;
- to categorical when it can be hashed, as category labels must;
- to text when it is a boolean, a number, a datetime or a duration.

A category of a categorical dtype converts to whatever its label is or converts
to. Text is read with the parse rules in _parse, which infer_column_semantics
shares, so the two analyses never disagree on what parses. Each distinct value
is read once and stands for every row that holds it, and values of different
Python types are read apart, though pandas counts 1, 1.0 and True as one.
"""

import typing
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray

from agent_eda._columns import ColumnKind
from agent_eda._levels import Levels, levels_of
from agent_eda._numeric import stored_nan_count
from agent_eda._parse import (
    ValueKind,
    is_whole,
    is_zero_or_one,
    read_datetime_format,
    read_texts,
    value_kind,
)
from agent_eda._values import categorical_array, value_codes
from agent_eda.results import LogicalType

# What a value is, measured against the declared logical type.
COMPATIBLE, CONVERTIBLE, INVALID = 0, 1, 2

type _Values = np.ndarray | pd.Index | ExtensionArray


@dataclass(frozen=True)
class Check:
    """How many of a column's values fall into each group, and which are invalid."""

    compatible: int
    convertible: int
    invalid: int
    missing: int
    # NaN that a float column stores as a value, counted as missing.
    stored_nan: int
    # Positions of the rows that hold an invalid value, in row order.
    invalid_rows: np.ndarray
    # The distinct invalid values with their counts, as levels: values of
    # different Python types are distinct here, though pandas counts 1 and True
    # as one value.
    invalid_values: Levels


@dataclass(frozen=True)
class _Groups:
    """A column's values grouped so that every value in a group is read alike.

    `codes` gives each row its group, -1 for a missing value; `values` holds
    one value per group and `verdicts` each group's verdict: COMPATIBLE,
    CONVERTIBLE or INVALID.
    """

    codes: np.ndarray
    values: _Values
    verdicts: np.ndarray


def check_column(
    series: pd.Series,
    kind: ColumnKind,
    declared: LogicalType,
    datetime_format: str | None,
) -> Check:
    """Sort a column's values by how they relate to the declared logical type.

    Keeps the positions of every row that holds an invalid value.
    """
    floats = (
        series.to_numpy(dtype="float64", na_value=np.nan)
        if kind == "numeric" and series.dtype.kind == "f"
        else None
    )
    stored_nan = stored_nan_count(series)
    uniform = _uniform_verdict(series, kind, declared)
    if uniform is not None:
        missing = (
            int(np.count_nonzero(np.isnan(floats)))
            if floats is not None
            else int(series.isna().sum())
        )
        by_verdict = [0, 0, 0]
        by_verdict[uniform] = len(series) - missing
        return Check(
            compatible=by_verdict[COMPATIBLE],
            convertible=by_verdict[CONVERTIBLE],
            invalid=0,
            missing=missing,
            stored_nan=stored_nan,
            invalid_rows=np.zeros(0, dtype=np.intp),
            invalid_values=levels_of(
                pd.Index([], dtype=object), np.zeros(0, dtype=np.intp)
            ),
        )
    groups = _groups(series, kind, declared, datetime_format, floats)
    present = groups.codes >= 0
    sizes = np.bincount(groups.codes[present], minlength=len(groups.verdicts))
    by_verdict = [int(sizes[groups.verdicts == v].sum()) for v in range(3)]
    # Categories that no value takes are groups without values.
    invalid = (groups.verdicts == INVALID) & (sizes > 0)
    # Code -1 picks the appended False: a missing value is never invalid.
    invalid_rows = np.flatnonzero(np.append(invalid, False)[groups.codes])
    positions = np.flatnonzero(invalid)
    return Check(
        compatible=by_verdict[COMPATIBLE],
        convertible=by_verdict[CONVERTIBLE],
        invalid=by_verdict[INVALID],
        missing=int(len(series) - np.count_nonzero(present)),
        stored_nan=stored_nan,
        invalid_rows=invalid_rows,
        invalid_values=levels_of(_index(groups.values)[positions], sizes[positions]),
    )


def _index(values: _Values) -> pd.Index:
    """Values as a pandas Index, object arrays kept as objects."""
    if isinstance(values, pd.Index):
        return values
    if isinstance(values, np.ndarray) and values.dtype == object:
        # pandas 3 would otherwise infer str for an array of strings.
        return pd.Index(values, dtype=object)
    return pd.Index(values)


def _numeric_storage(series: pd.Series) -> LogicalType | None:
    """The logical type a numeric dtype stores; None for decimals."""
    kind = series.dtype.kind
    if kind in "iu":
        return "integer"
    return "float" if kind == "f" else None


def _uniform_verdict(
    series: pd.Series, kind: ColumnKind, declared: LogicalType
) -> int | None:
    """The verdict on every value, when the dtype decides it and none is invalid.

    Then no value needs reading. None when the values decide.
    """
    if kind == "categorical":
        return COMPATIBLE if declared == "categorical" else None
    if kind in ("boolean", "datetime", "timedelta"):
        verdict = _stored_verdict(kind, declared)
        return None if verdict == INVALID else verdict
    if kind == "numeric":
        stored = _numeric_storage(series)
        if declared == stored:
            return COMPATIBLE
        if declared in ("float", "text", "categorical"):
            return CONVERTIBLE
        return None
    if kind == "text" or (
        # Objects that are all strings, judged without counting them.
        kind == "object"
        and declared in ("text", "categorical")
        and pd.api.types.infer_dtype(series, skipna=True) in ("string", "empty")
    ):
        if declared == "text":
            return COMPATIBLE
        return CONVERTIBLE if declared == "categorical" else None
    return None


def _stored_verdict(stored: LogicalType | None, declared: LogicalType) -> int:
    """The verdict on a value stored as a boolean, a datetime or a duration."""
    if stored == declared:
        return COMPATIBLE
    # Every such value converts to text, and can be hashed as a category.
    return CONVERTIBLE if declared in ("text", "categorical") else INVALID


def _groups(
    series: pd.Series,
    kind: ColumnKind,
    declared: LogicalType,
    datetime_format: str | None,
    floats: np.ndarray | None,
) -> _Groups:
    if kind == "numeric":
        return _numbers(series, declared, floats)
    if kind == "categorical":
        return _categories(series, declared, datetime_format)
    if kind in ("boolean", "datetime", "timedelta"):
        codes, uniques = _distinct(series)
        verdict = _stored_verdict(kind, declared)
        verdicts = np.full(len(uniques), verdict, dtype=np.uint8)
        return _Groups(codes=codes, values=uniques, verdicts=verdicts)
    if kind == "text":
        codes, uniques = _distinct(series)
        texts = np.asarray(uniques, dtype=object)
        verdicts = _text_verdicts(texts, declared, datetime_format)
        return _Groups(codes=codes, values=uniques, verdicts=verdicts)
    return _objects(series, declared, datetime_format)


def _distinct(values: "pd.Series | np.ndarray") -> tuple[np.ndarray, typing.Any]:
    """Each value's code and the distinct values, of values that can be hashed."""
    found = value_codes(values, unhashable="by_content")
    return found.codes, found.uniques


def _numbers(
    series: pd.Series, declared: LogicalType, floats: np.ndarray | None
) -> _Groups:
    """Integers, floats or decimals, by their dtype."""
    stored = _numeric_storage(series)
    if floats is not None:
        # NaN that a float column stores as a value is missing here, too.
        codes, uniques = _distinct(floats)
        numbers = uniques
    else:
        codes, uniques = _distinct(series)
        # A Series gives an Index of its distinct values.
        index = typing.cast("pd.Index", uniques)
        numbers = np.asarray(index.to_numpy(dtype="float64", na_value=np.nan))
    if declared == stored:
        verdicts = np.full(len(uniques), COMPATIBLE, dtype=np.uint8)
    elif declared == "integer":
        whole = np.isfinite(numbers) & (numbers == np.floor(numbers))
        verdicts = np.where(whole, CONVERTIBLE, INVALID).astype(np.uint8)
    elif declared == "boolean":
        zero_or_one = (numbers == 0) | (numbers == 1)
        verdicts = np.where(zero_or_one, CONVERTIBLE, INVALID).astype(np.uint8)
    elif declared in ("float", "text", "categorical"):
        verdicts = np.full(len(uniques), CONVERTIBLE, dtype=np.uint8)
    else:  # datetime and timedelta
        verdicts = np.full(len(uniques), INVALID, dtype=np.uint8)
    return _Groups(codes=codes, values=uniques, verdicts=verdicts)


def _categories(
    series: pd.Series, declared: LogicalType, datetime_format: str | None
) -> _Groups:
    """A categorical: each category is read once, by the label it holds."""
    categorical = categorical_array(series)
    categories = categorical.categories
    codes = np.asarray(categorical.codes, dtype=np.intp)
    verdicts = _object_verdicts(
        np.asarray(categories, dtype=object),
        declared,
        datetime_format,
        in_categorical=True,
    )
    return _Groups(codes=codes, values=categories, verdicts=verdicts)


def _objects(
    series: pd.Series, declared: LogicalType, datetime_format: str | None
) -> _Groups:
    """Python objects of any type, read by their own type."""
    values = np.asarray(series.array, dtype=object)
    found = value_codes(values, unhashable="by_content")
    codes, uniques = found.codes, found.uniques
    if not found.by_content and pd.api.types.infer_dtype(uniques, skipna=False) in (
        "string",
        "empty",
    ):
        texts = np.asarray(uniques, dtype=object)
        verdicts = _text_verdicts(texts, declared, datetime_format)
        return _Groups(codes=codes, values=texts, verdicts=verdicts)
    # pandas counts 1, 1.0 and True as one value: tell values apart by type too.
    types = np.fromiter(map(type, values), dtype=object, count=len(values))
    type_codes, type_uniques = pd.factorize(types)
    present = codes >= 0
    if len(type_uniques) > 1:
        combined = codes[present] * len(type_uniques) + type_codes[present]
        codes = codes.copy()
        codes[present] = pd.factorize(combined)[0]
    representatives = values[_first_positions(codes)]
    verdicts = _object_verdicts(representatives, declared, datetime_format)
    return _Groups(codes=codes, values=representatives, verdicts=verdicts)


def _first_positions(codes: np.ndarray) -> np.ndarray:
    """Where each code first occurs, for codes numbered in order of appearance.

    Each code is one more than the largest before it where it first occurs.
    """
    running = np.maximum.accumulate(codes) if len(codes) else codes
    return np.flatnonzero(np.diff(running, prepend=-1) > 0)


def _object_verdicts(
    values: np.ndarray,
    declared: LogicalType,
    datetime_format: str | None,
    *,
    in_categorical: bool = False,
) -> np.ndarray:
    """The verdict on each of some distinct values of any type."""
    kinds = [value_kind(value) for value in values]
    is_text = np.array([kind == "text" for kind in kinds], dtype=bool)
    verdicts = np.empty(len(values), dtype=np.uint8)
    verdicts[is_text] = _text_verdicts(
        values[is_text], declared, datetime_format, in_categorical=in_categorical
    )
    for position in np.flatnonzero(~is_text).tolist():
        verdicts[position] = _value_verdict(
            values[position], kinds[position], declared, in_categorical=in_categorical
        )
    return verdicts


def _value_verdict(
    value: object, kind: ValueKind, declared: LogicalType, *, in_categorical: bool
) -> int:
    """The verdict on one value that isn't text."""
    if declared == "categorical":
        if in_categorical:
            return COMPATIBLE
        return (
            CONVERTIBLE if isinstance(value, Hashable) and _hashes(value) else INVALID
        )
    stored = _stored_type(value, kind)
    if stored == declared:
        return CONVERTIBLE if in_categorical else COMPATIBLE
    if declared == "text":
        return INVALID if kind == "other" else CONVERTIBLE
    if kind != "number":
        return INVALID
    if declared == "float":
        return CONVERTIBLE
    if declared == "integer":
        return CONVERTIBLE if is_whole(value) else INVALID
    if declared == "boolean":
        return CONVERTIBLE if is_zero_or_one(value) else INVALID
    return INVALID


def _stored_type(value: object, kind: ValueKind) -> LogicalType | None:
    """The logical type a value is stored as, by its Python type."""
    if kind == "number":
        if isinstance(value, int | np.integer):
            return "integer"
        return "float" if isinstance(value, float | np.floating) else None
    return None if kind == "other" else kind


def _hashes(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _text_verdicts(
    texts: np.ndarray,
    declared: LogicalType,
    datetime_format: str | None,
    *,
    in_categorical: bool = False,
) -> np.ndarray:
    """The verdict on each of some distinct strings, by the shared parse rules."""
    if declared == "text":
        verdict = CONVERTIBLE if in_categorical else COMPATIBLE
        return np.full(len(texts), verdict, dtype=np.uint8)
    if declared == "categorical":
        verdict = COMPATIBLE if in_categorical else CONVERTIBLE
        return np.full(len(texts), verdict, dtype=np.uint8)
    if declared == "timedelta" or not len(texts):
        return np.full(len(texts), INVALID, dtype=np.uint8)
    read = read_texts(texts)
    if declared == "boolean":
        converts = read.boolean
    elif declared == "integer":
        converts = read.whole
    elif declared == "float":
        converts = read.number
    else:  # datetime
        converts = read.iso_date.copy()
        if datetime_format is not None:
            rest = np.flatnonzero(~converts)
            converts[rest] = read_datetime_format(texts[rest], datetime_format)
    return np.where(converts, CONVERTIBLE, INVALID).astype(np.uint8)
