"""Declared sentinel values: which ones a column can hold, and where they occur.

A sentinel value is a stand-in that means missing, such as -999 or 'N/A'.
profile_missing_values counts the declared ones as missing and
detect_missing_sentinels reports them. Both check and match them here, so they
can't disagree about which values a call may declare or what a value matches.

A declared value matches exactly, among the values pandas doesn't count as
missing: text with its case and surrounding spaces, numbers by value whatever
their type (-999 matches -999.0, but no boolean), and datetimes by the instant
they name. A value that can never match its column raises before any
computation, naming the value the caller likely meant: one of another type,
such as the text '-999' for an integer column, or one the dtype can't hold,
such as 300 for a uint8 column. So does a value that pandas counts as missing
itself, such as NaN, and an infinite one.
"""

import dataclasses
import datetime as dt
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeGuard, cast

import numpy as np
import pandas as pd

from agent_eda._columns import (
    ALL_KINDS,
    Column,
    ColumnRule,
    column_kind,
    select_columns,
)
from agent_eda._parse import ReadFamily, read_as
from agent_eda._run import check_distinct
from agent_eda._text import Surface, join_items, join_labels
from agent_eda._wording import worded
from agent_eda.errors import InvalidArgumentError

# The types a declared sentinel value can have.
SentinelValue = bool | int | float | str | dt.datetime | dt.date | dt.timedelta

# How a declared value appears in a result record.
type JsonSentinel = bool | int | float | str

# What kind of thing a value is, whatever its exact type: 3, 3.0 and
# Decimal('3') are all numbers.
Family = Literal["number", "text", "boolean", "datetime", "date", "timedelta", "other"]

# Each family as messages name its values, in the order they list them.
_WORDS: dict[Family, str] = {
    "number": "numbers",
    "text": "text",
    "boolean": "booleans",
    "datetime": "datetimes",
    "date": "dates",
    "timedelta": "durations",
    "other": "other values",
}
# The units pandas stores datetimes and timedeltas in.
Unit = Literal["s", "ms", "us", "ns"]
_UNIT_WORDS: dict[Unit, str] = {
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
}
# A message lists at most this many values it can't use.
_MAX_LISTED = 5

# Sentinel values can name a column of any dtype; whether a value can match it
# is checked value by value.
_ANY_COLUMN = ColumnRule(
    default=frozenset(),
    default_text="no column",
    accepted=ALL_KINDS,
    accepted_text="columns of every dtype",
    fixes={},
)


def family(value: object) -> Family:
    """The family of a value, such as "number" for 3, 3.0 and Decimal('3')."""
    return type_family(type(value))


def type_family(kind: type) -> Family:
    """The family of the values of a type, NumPy scalar types included."""
    if issubclass(kind, bool | np.bool_):
        return "boolean"
    # Before numbers: NumPy counts timedelta64 as an integer.
    if issubclass(kind, dt.timedelta | np.timedelta64):
        return "timedelta"
    if issubclass(kind, dt.datetime | np.datetime64):
        return "datetime"
    if issubclass(kind, dt.date):
        return "date"
    if issubclass(kind, numbers.Number):
        return "number"
    if issubclass(kind, str):
        return "text"
    return "other"


@dataclass(frozen=True)
class Holds:
    """What the values of a column can be, as far as declared values go."""

    # The dtype as pandas shows it, such as 'Int64'.
    dtype: str
    # The families its values belong to; None when any value can occur, as in
    # an object column without values.
    families: frozenset[Family] | None
    # How messages name its values, such as "integers".
    what: str
    # The smallest and largest integer an integer dtype holds.
    integers: tuple[int, int] | None = None
    # The largest magnitude a float dtype holds.
    float_max: float | None = None
    # The time zone of a datetime dtype; None when it has none.
    tz: dt.tzinfo | str | None = None
    # The unit a datetime or timedelta dtype stores, such as "ns".
    unit: Unit | None = None


def holds(series: pd.Series) -> Holds:
    """What the column's values can be: from its dtype, or its values if objects."""
    dtype = series.dtype
    shown = str(dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        inner = holds(pd.Series(dtype.categories))
        return dataclasses.replace(
            inner, dtype=shown, what=f"categories of {inner.what}"
        )
    kind = column_kind(dtype)
    if kind == "numeric":
        return _numbers(dtype, shown)
    if kind == "boolean":
        return Holds(shown, frozenset({"boolean"}), "booleans")
    if kind == "text":
        return Holds(shown, frozenset({"text"}), "text")
    if kind == "datetime":
        return _datetimes(dtype, shown)
    if kind == "timedelta":
        durations: frozenset[Family] = frozenset({"timedelta"})
        return Holds(shown, durations, "durations", unit=_unit(dtype))
    if kind == "object":
        present = _families_present(series)
        if not present:
            return Holds(shown, None, "no values")
        words = [word for name, word in _WORDS.items() if name in present]
        return Holds(shown, frozenset(present), join_items(words))
    return Holds(shown, frozenset({"other"}), "values no sentinel value can equal")


def _numbers(dtype: object, shown: str) -> Holds:
    kind = getattr(dtype, "kind", "O")
    if kind in "iu":
        limits = np.iinfo(numpy_dtype(dtype))
        return Holds(
            shown,
            frozenset({"number"}),
            "integers",
            integers=(int(limits.min), int(limits.max)),
        )
    if kind == "f":
        largest = float(np.finfo(numpy_dtype(dtype)).max)
        return Holds(shown, frozenset({"number"}), "numbers", float_max=largest)
    return Holds(shown, frozenset({"number"}), "numbers")  # pyarrow decimals


def numpy_dtype(dtype: object) -> np.dtype:
    """The NumPy dtype behind a NumPy, nullable, pyarrow-backed or sparse dtype."""
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, pd.SparseDtype):
        return np.dtype(dtype.subtype)
    # Nullable and pyarrow-backed dtypes name their NumPy counterpart.
    return np.dtype(getattr(dtype, "numpy_dtype"))  # noqa: B009


def _datetimes(dtype: object, shown: str) -> Holds:
    if isinstance(dtype, pd.ArrowDtype) and str(dtype.pyarrow_dtype).startswith("date"):
        return Holds(shown, frozenset({"date"}), "dates")
    tz = _time_zone(dtype)
    what = f"datetimes in time zone {tz}" if tz else "datetimes without a time zone"
    return Holds(shown, frozenset({"datetime"}), what, tz=tz, unit=_unit(dtype))


def _time_zone(dtype: object) -> dt.tzinfo | str | None:
    """The time zone of a datetime dtype, or None when it has none."""
    if isinstance(dtype, pd.DatetimeTZDtype):
        return dtype.tz
    if isinstance(dtype, pd.ArrowDtype):
        zone = getattr(dtype.pyarrow_dtype, "tz", None)
        return str(zone) if zone else None
    return None


def _unit(dtype: object) -> Unit | None:
    """The unit a datetime or timedelta dtype stores, such as "ns"."""
    if isinstance(dtype, pd.ArrowDtype):
        unit = str(getattr(dtype.pyarrow_dtype, "unit", ""))
    elif isinstance(dtype, pd.DatetimeTZDtype):
        unit = str(dtype.unit)
    else:
        unit = np.datetime_data(str(dtype))[0]
    return next((known for known in _UNIT_WORDS if known == unit), None)


def _families_present(series: pd.Series) -> set[Family]:
    """The families of the values that pandas doesn't count as missing."""
    objects = np.asarray(series.array, dtype=object)
    # A quick answer for the common cases: only text, or nothing at all.
    inferred = pd.api.types.infer_dtype(objects, skipna=True)
    if inferred in ("string", "empty"):
        return {"text"} if inferred == "string" else set()
    present = objects[~np.asarray(series.isna(), dtype=bool)]
    return {type_family(kind) for kind in set(map(type, present))}


@dataclass(frozen=True)
class _Problem:
    """A declared value a call can't use, and why."""

    column: str
    value: object
    # "type": the column holds another type; "limit": the column's dtype can't
    # hold the value; "missing": it is a missing value itself; "infinite": it
    # is infinite.
    problem: Literal["type", "limit", "missing", "infinite"]
    # What the column holds, for a type or limit problem, such as "integers
    # from 0 to 255", and its dtype.
    holds: str = ""
    dtype: str = ""
    # The value the caller likely meant, if any.
    likely: object = None


def declared_values(
    analysis: str,
    data: pd.DataFrame,
    selected: Sequence[Column],
    sentinel_values: Mapping[str, Sequence[SentinelValue]] | None,
) -> dict[str, list[SentinelValue]]:
    """The declared values of each selected column, checked against the data.

    A column with an empty list has none declared, as one the mapping leaves
    out. Raises before any computation: ColumnNotFoundError for a name that
    isn't a column, and InvalidArgumentError for a column the call leaves out,
    a value listed twice for one column, or a value that can never match its
    column.
    """
    if not sentinel_values:
        return {}
    names = list(sentinel_values)
    # Raises ColumnNotFoundError for a name that isn't a column of the data.
    select_columns(analysis, data, names, _ANY_COLUMN, parameter="sentinel_values")
    by_name = {column.name: column for column in selected}
    left_out = [name for name in names if name not in by_name]
    if left_out:
        raise InvalidArgumentError(_left_out_message(analysis, left_out))
    problems: list[_Problem] = []
    declared: dict[str, list[SentinelValue]] = {}
    for name, values in sentinel_values.items():
        column = by_name[name]
        values = list(values)
        # 1 and True are two values, as the matching keeps them apart: values are
        # the same when they are equal and of one family.
        check_distinct(
            analysis,
            f"sentinel_values[{name!r}]",
            values,
            "value",
            key=lambda value: (family(value), value),
        )
        column_holds = holds(data.iloc[:, column.position])
        problems.extend(
            problem
            for value in values
            if (problem := _check(name, value, column_holds)) is not None
        )
        if values:
            declared[name] = values
    if problems:
        raise worded(
            InvalidArgumentError, lambda surface: _message(analysis, problems, surface)
        )
    return declared


def _left_out_message(analysis: str, names: list[str]) -> str:
    if len(names) == 1:
        return (
            f"{analysis} got sentinel values for {names[0]!r}, which columns leaves "
            "out. Add it to columns, or leave it out of sentinel_values."
        )
    return (
        f"{analysis} got sentinel values for {len(names):,} columns that columns "
        f"leaves out: {join_labels(names)}. Add them to columns, or leave them out "
        "of sentinel_values."
    )


def can_match(value: SentinelValue, column: Holds) -> bool:
    """Whether the value can match a value of the column, as declared_values decides."""
    return _check("", value, column) is None


def _check(name: str, value: SentinelValue, column: Holds) -> _Problem | None:
    """Why a declared value can never match the column, or None if it can."""
    if value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return _Problem(name, value, "missing")
    if isinstance(value, float) and math.isinf(value):
        return _Problem(name, value, "infinite")
    if column.families == {"other"}:
        return _Problem(name, value, "limit", column.what, column.dtype)
    if column.families is not None and family(value) not in column.families:
        likely = likely_value(value, column)
        return _Problem(name, value, "type", column.what, column.dtype, likely)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _number_limits(name, value, column)
    if isinstance(value, dt.datetime):
        return _datetime_limits(name, value, column)
    if isinstance(value, dt.timedelta):
        return _timedelta_limits(name, value, column)
    return None


def _number_limits(name: str, value: float, column: Holds) -> _Problem | None:
    """Why the column's integer or float dtype can't hold the number, if it can't."""
    if column.integers is not None:
        if isinstance(value, float) and not value.is_integer():
            return _Problem(name, value, "limit", "integers", column.dtype)
        low, high = column.integers
        if not low <= value <= high:
            holds = f"integers from {low:,} to {high:,}"
            return _Problem(name, value, "limit", holds, column.dtype)
    largest = column.float_max
    if largest is not None and abs(value) > largest:
        holds = f"numbers from {-largest:.4g} to {largest:.4g}"
        return _Problem(name, value, "limit", holds, column.dtype)
    return None


def _datetime_limits(name: str, value: dt.datetime, column: Holds) -> _Problem | None:
    """Why the column can't hold the datetime: its time zone, range or unit.

    A column of Python objects has no unit, so only its values' family counts.
    """
    unit = column.unit
    if unit is None:
        return None
    stamp = _timestamp(value)
    if (stamp.tz is None) != (column.tz is None):
        likely = _in_zone(stamp, column.tz)
        return _Problem(name, value, "type", column.what, column.dtype, likely)
    try:
        stamp.as_unit(unit, round_ok=False)
    except pd.errors.OutOfBoundsDatetime:
        low, high = pd.Timestamp.min, pd.Timestamp.max
        holds = f"datetimes from {low:%Y-%m-%d} to {high:%Y-%m-%d}"
        return _Problem(name, value, "limit", holds, column.dtype)
    except ValueError:  # it would lose precision
        holds = f"datetimes in whole {_UNIT_WORDS[unit]}"
        return _Problem(name, value, "limit", holds, column.dtype)
    return None


def _timedelta_limits(name: str, value: dt.timedelta, column: Holds) -> _Problem | None:
    """Why the column can't hold the duration: its range or unit."""
    unit = column.unit
    if unit is None:
        return None
    try:
        pd.Timedelta(value).as_unit(unit, round_ok=False)
    except pd.errors.OutOfBoundsTimedelta:
        holds = f"durations within ±{pd.Timedelta.max.days:,} days"
        return _Problem(name, value, "limit", holds, column.dtype)
    except ValueError:  # it would lose precision
        holds = f"durations in whole {_UNIT_WORDS[unit]}"
        return _Problem(name, value, "limit", holds, column.dtype)
    return None


def _timestamp(value: dt.date) -> pd.Timestamp:
    """A date or datetime as a pandas Timestamp.

    pandas types the result as possibly NaT, which a declared value never is:
    declared_values rejects NaT first.
    """
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp):
        raise TypeError(f"{value!r} names no time")
    return stamp


def likely_value(value: SentinelValue, column: Holds) -> SentinelValue | None:
    """The value of a type the column holds that the caller likely meant, or None.

    Text converts to a number, a boolean, an ISO 8601 date or datetime, or a
    duration; a number or boolean to text, 1 and 0 to True and False and back;
    a datetime at midnight to a date and a date to a datetime. A guess the
    column couldn't hold either is no guess. profile_target guesses the labels
    its expected_labels meant with it too, so the two parse text alike.
    """
    for target in _WORDS:
        if column.families is None or target not in column.families:
            continue
        likely = _convert(value, target, column)
        if likely is not None and _check("", likely, column) is None:
            return likely
    return None


def _convert(
    value: SentinelValue, target: Family, column: Holds
) -> SentinelValue | None:
    """The value as one of the target family, or None when it has no such form."""
    if target == "text":
        return _as_text(value)
    if isinstance(value, str):
        return _parse(value.strip(), target, column)
    if isinstance(value, bool):
        return int(value) if target == "number" else None
    if isinstance(value, int | float):
        return bool(value) if target == "boolean" and value in (0, 1) else None
    if isinstance(value, dt.datetime):
        stamp = _timestamp(value)
        if target == "date" and stamp == stamp.normalize():
            return stamp.date()
        return None
    if isinstance(value, dt.date) and target == "datetime":
        return _in_zone(_timestamp(value), column.tz)
    return None


def _as_text(value: SentinelValue) -> str | None:
    """A value as the text a column most likely holds for it.

    Durations are written in too many ways to guess one.
    """
    if isinstance(value, str | dt.timedelta):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dt.datetime):
        stamp = _timestamp(value)
        if stamp.tz is None and stamp == stamp.normalize():
            return stamp.date().isoformat()
        return stamp.isoformat(sep=" ")
    return value.isoformat()  # a date


def read_iso(
    data: pd.DataFrame, sentinel_values: Mapping[str, Sequence[SentinelValue]]
) -> dict[str, list[SentinelValue]]:
    """The declared values, with each string that an ISO 8601 value of the
    column's family spells read as that value.

    Over MCP, JSON can't carry a datetime, a date or a duration, so a caller
    declares one as its ISO 8601 text, which the input schema offers next to
    plain text. The text names such a value when the column holds datetimes,
    dates or durations and no text, and the text is in ISO 8601: '1900-01-01',
    '1900-01-01T00:00:00+01:00' or 'PT0S'. Every other string stays text, so
    declared_values reports it, as it does for a Python caller. A time zone is
    kept as written: a column without one rejects a time zone, as it does a
    pd.Timestamp that has one.
    """
    positions: dict[str, list[int]] = {}
    for position, label in enumerate(data.columns):
        positions.setdefault(str(label), []).append(position)
    read: dict[str, list[SentinelValue]] = {}
    for name, values in sentinel_values.items():
        values = list(values)
        where = positions.get(name, [])
        if len(where) == 1 and any(isinstance(value, str) for value in values):
            column = holds(data.iloc[:, where[0]])
            values = [_read_iso(value, column) for value in values]
        read[name] = values
    return read


def _read_iso(value: SentinelValue, column: Holds) -> SentinelValue:
    """The value as an ISO 8601 datetime, date or duration that its column can
    hold, or as it is when it is no such text."""
    if not isinstance(value, str) or column.families is None:
        return value
    if "text" in column.families:
        return value
    for target in _ISO_FAMILIES:
        if target in column.families:
            read = _iso(value.strip(), target)
            if read is not None:
                return read
    return value


# The families whose values an ISO 8601 string spells, in the order they are tried.
_ISO_FAMILIES: tuple[Family, ...] = ("datetime", "date", "timedelta")


def _iso(text: str, target: Family) -> SentinelValue | None:
    """Text as an ISO 8601 datetime, date or duration, or None when it isn't one.

    The rules are the analyses' own (_parse.read_as), except that a duration
    must be written in ISO 8601, as the input schema's format says: pandas also
    reads '5' as five nanoseconds and '1 day' as a day.
    """
    if target == "timedelta" and not text.lstrip("+-").startswith("P"):
        return None
    if target not in _ISO_FAMILIES:
        return None
    value = read_as(text, cast("ReadFamily", target))
    return None if isinstance(value, bool | int | float) else value


def _parse(text: str, target: Family, column: Holds) -> SentinelValue | None:
    """Text as a value of the target family, or None when it doesn't parse.

    The rules are the analyses' own (_parse.read_as), so a suggestion is a
    value that infer_column_semantics and validate_column_types read the same
    text as. Dates and datetimes read as ISO 8601 only, so no day and month
    order is guessed and words such as 'now' stay text.
    """
    if target in ("text", "other"):
        return None
    value = read_as(text, target, whole=column.integers is not None)
    if isinstance(value, pd.Timestamp):
        return _in_zone(value, column.tz)
    return value


def _in_zone(stamp: pd.Timestamp, tz: dt.tzinfo | str | None) -> pd.Timestamp:
    """The timestamp with the column's time zone, keeping its wall time."""
    if tz is None:
        return stamp.tz_localize(None) if stamp.tz is not None else stamp
    return stamp.tz_localize(tz) if stamp.tz is None else stamp


def literal(value: object, surface: Surface = "python") -> str:
    """A value as the caller writes it: the Python that gives it, such as
    pd.Timestamp('1900-01-01'), or over MCP its JSON, such as "1900-01-01T00:00:00".
    """
    if surface == "mcp" and _is_sentinel_value(value):
        return json.dumps(json_value(value))
    text = repr(value)
    if isinstance(value, pd.Timestamp | pd.Timedelta):
        return f"pd.{text}"
    return text


def _is_sentinel_value(value: object) -> TypeGuard[SentinelValue]:
    return isinstance(value, bool | int | float | str | dt.date | dt.timedelta)


def _message(analysis: str, problems: list[_Problem], surface: Surface) -> str:
    """Say which declared values can't be used, why, and how to fix the call."""
    if len(problems) == 1:
        [problem] = problems
        value = literal(problem.value, surface)
        head = (
            f"{analysis} got {value} for column {problem.column!r} in sentinel_values"
        )
        if problem.problem == "missing":
            why = f"{value} is a missing value itself, not a stand-in for one."
        elif problem.problem == "infinite":
            why = f"{value} is an infinite value, not a stand-in for a missing one."
        else:
            why = (
                f"{problem.column!r} holds {problem.holds} (dtype {problem.dtype}), "
                "so it can never match."
            )
            if problem.likely is not None:
                why += f" Did you mean {literal(problem.likely, surface)}?"
        return f"{head}, but {why} {_closing(problems, surface)}"
    fragments = [_fragment(problem, surface) for problem in problems[:_MAX_LISTED]]
    if len(problems) > _MAX_LISTED:
        fragments.append(f"and {len(problems) - _MAX_LISTED:,} more")
    return (
        f"{analysis} can't use {len(problems):,} values in sentinel_values: "
        f"{'; '.join(fragments)}. {_closing(problems, surface)}"
    )


def _fragment(problem: _Problem, surface: Surface) -> str:
    """A value the call can't use and why, as one item of a list."""
    named = f"{literal(problem.value, surface)} for {problem.column!r}"
    if problem.problem == "missing":
        return f"{named}, a missing value itself"
    if problem.problem == "infinite":
        return f"{named}, an infinite value"
    likely = (
        f"; did you mean {literal(problem.likely, surface)}?"
        if problem.likely is not None
        else ""
    )
    return f"{named}, which holds {problem.holds} (dtype {problem.dtype}{likely})"


def _closing(problems: list[_Problem], surface: Surface) -> str:
    """How to fix each kind of problem, in a fixed order."""
    kinds = {problem.problem for problem in problems}
    sentences = []
    if "type" in kinds:
        sentence = "Declare each value as the type its column holds"
        [first] = problems if len(problems) == 1 else [None]
        if first is not None and first.likely is not None:
            likely = literal(first.likely, surface)
            sentence += (
                f', such as {{"sentinel_values": {{{json.dumps(first.column)}: '
                f"[{likely}]}}}}"
                if surface == "mcp"
                else f", such as sentinel_values={{{first.column!r}: [{likely}]}}"
            )
        sentences.append(f"{sentence}.")
    if "limit" in kinds:
        sentences.append(
            "Leave it out of sentinel_values, or declare a value the column can hold."
            if len(problems) == 1
            else "Leave out the values their columns can't hold."
        )
    if "missing" in kinds:
        sentences.append(
            "Declare only stand-ins for missing values, such as -999 or 'N/A'."
        )
    if "infinite" in kinds:
        sentences.append(
            "Replace infinite values with NaN first if they mean missing, such as "
            "df.replace([np.inf, -np.inf], np.nan)."
        )
    return " ".join(sentences)


def json_value(value: SentinelValue) -> JsonSentinel:
    """A declared value as the record shows it: dates and times as ISO 8601 text."""
    if isinstance(value, dt.timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, dt.date):  # datetimes and pandas Timestamps too
        return value.isoformat()
    return value


def matches(series: pd.Series, value: SentinelValue, missing: np.ndarray) -> np.ndarray:
    """Where the column holds the declared value, as a NumPy bool array.

    The value must be one declared_values accepted for the column, and
    `missing` marks the values pandas isna() counts as missing, which never
    match. In a column of Python objects, only values of the declared value's
    family match: Python's True == 1 doesn't count.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        # Match the categories, then find the rows that hold a matching one.
        categories = pd.Series(dtype.categories)
        hits = np.flatnonzero(
            matches(categories, value, np.zeros(len(categories), dtype=bool))
        )
        codes = np.asarray(series.cat.codes)
        return np.isin(codes, hits) if hits.size else np.zeros(len(codes), bool)
    if isinstance(dtype, np.dtype) and dtype.kind == "O":
        return _object_matches(series, value, missing)
    # pyarrow decimals compare with a float inexactly, as floats themselves.
    decimals = column_kind(dtype) == "numeric" and dtype.kind == "O"
    equal = series == (_decimal(value) if decimals else value)
    return np.asarray(equal.to_numpy(dtype=bool, na_value=False), dtype=bool) & ~missing


def _decimal(value: SentinelValue) -> SentinelValue | Decimal:
    """A number as the decimal it reads as, such as Decimal('0.1') for 0.1.

    A decimal equals a declared number when it equals the number as written:
    0.1 as a float is 0.1000000000000000055..., which no decimal 0.10 equals.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(value)


def _object_matches(
    series: pd.Series, value: SentinelValue, missing: np.ndarray
) -> np.ndarray:
    """Where a column of Python objects holds the value, comparing with NumPy."""
    objects = np.asarray(series.array, dtype=object)
    present = np.flatnonzero(~missing)
    candidates = objects[present]
    kind = family(value)
    try:
        # NumPy compares objects with Python's ==, faster than pandas does.
        equal = np.asarray(candidates == value, dtype=bool)
        if isinstance(value, float) and not value.is_integer():
            # Decimals equal to the number as written, which Python's
            # Decimal('0.1') == 0.1 says aren't.
            equal |= np.asarray(candidates == _decimal(value), dtype=bool)
    except (TypeError, ValueError):
        # A value such as a NumPy array compares to an array, not a bool.
        equal = np.fromiter(
            (family(item) == kind and _equal(item, value) for item in candidates),
            dtype=bool,
            count=len(candidates),
        )
    hits = present[equal]
    same = np.fromiter(
        (family(objects[p]) == kind for p in hits), dtype=bool, count=len(hits)
    )
    found = np.zeros(len(objects), dtype=bool)
    found[hits[same]] = True
    return found


def _equal(item: object, value: SentinelValue) -> bool:
    """Whether a Python object equals a declared value of its family."""
    if isinstance(item, Decimal):
        return bool(item == _decimal(value))
    return bool(item == value)
