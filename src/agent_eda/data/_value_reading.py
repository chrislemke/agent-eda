"""How the values of a column read: one engine for the two type analyses.

read_values(series, kind) gives a ValueReading, which infer_column_semantics
and validate_column_types both take their counts from, so neither walks a
column itself and the two can't disagree on what a value is or what parses. A
reading holds the column's distinct values and their counts, the kind of each
value and the logical type the dtype stores, and a verdict on every distinct
value for every logical type. Each part is found when something first asks for
it, so a column whose dtype decides the answer is never walked.

A verdict says how a value relates to a logical type. Compatible values are
already stored as it. Convertible values are stored as another type and
convert to it exactly, without changing their meaning. Invalid values do
neither. Missing values are None, NaN, NaT and pd.NA, and NaN that a float
column stores as a value. Nothing is converted.

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
to. Text is read with the parse rules in _parse, so the analyses never disagree
on what parses. Each distinct value is read once and stands for every row that
holds it, and values of different Python types are read apart, though pandas
counts 1, 1.0 and True as one: `distinct` counts them as pandas does, the
groups that the verdicts cover keep them apart.
"""

import math
import typing
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from agent_eda._columns import ColumnKind, column_kind
from agent_eda._levels import Levels, count_levels_by_row, levels_of, value_types
from agent_eda._numeric import stored_nan_count
from agent_eda._parse import (
    VALUE_KINDS,
    TextReading,
    ValueKind,
    is_whole,
    is_zero_or_one,
    read_datetime_format,
    read_texts,
    value_kind,
)
from agent_eda._result_base import LogicalType
from agent_eda._values import value_codes

# What a value is, measured against a logical type.
COMPATIBLE, CONVERTIBLE, INVALID = 0, 1, 2


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

    `codes` gives each row its group, -1 for a missing value; `labels` holds
    one value per group and `counts` the rows of each. `objects` holds the
    values as Python objects when the groups were split by Python type, and is
    None when they are the column's levels.
    """

    codes: np.ndarray
    labels: "pd.Index | np.ndarray"
    counts: np.ndarray
    objects: np.ndarray | None


@dataclass(frozen=True)
class _TextFacts:
    """How the text values among the groups read, one entry per text group."""

    # The positions of the text groups; None when every group is text.
    at: np.ndarray | None
    reading: TextReading
    # Parses as ISO 8601 or, when declared, in the datetime_format.
    iso_date: np.ndarray
    # Looks like a date without parsing as either.
    date_like: np.ndarray
    counts: np.ndarray


def read_values(
    series: pd.Series,
    kind: ColumnKind | None = None,
    datetime_format: str | None = None,
) -> "ValueReading":
    """Read a column's values: nothing is walked until a part of it is asked for.

    The kind is the column's own, from its dtype, unless the caller has it.
    Text in the datetime_format, when there is one, reads as datetimes.
    """
    return ValueReading(
        series, column_kind(series.dtype) if kind is None else kind, datetime_format
    )


class ValueReading:
    """What a column's values are, and how each reads as every logical type.

    A column is read through its distinct values: `levels` counts them as
    pandas does, `values`, `counts` and `kinds` describe the groups of values
    that read alike, and `verdict(logical_type)` gives each group's verdict:
    COMPATIBLE, CONVERTIBLE or INVALID. The counts the analyses report are sums
    of the groups' counts: `boolean` is the number of values that are or read
    as booleans, `number` of those that are or read as numbers, and so on.
    """

    def __init__(
        self, series: pd.Series, kind: ColumnKind, datetime_format: str | None
    ) -> None:
        self.series = series
        self.kind = kind
        self.datetime_format = datetime_format
        self._verdicts: dict[LogicalType, np.ndarray] = {}

    # What the dtype says.

    @cached_property
    def storage(self) -> LogicalType | None:
        """The logical type the dtype stores; None for decimals and objects."""
        if self.kind == "numeric":
            dtype_kind = self.series.dtype.kind
            if dtype_kind in "iu":
                return "integer"
            return "float" if dtype_kind == "f" else None
        if self.kind in ("boolean", "datetime", "timedelta", "text", "categorical"):
            return self.kind
        return None

    @cached_property
    def stored_nan(self) -> int:
        """NaN that a float column stores as a value, counted as missing."""
        return stored_nan_count(self.series)

    @cached_property
    def _floats(self) -> np.ndarray | None:
        """A float column's values as floats, NaN where missing."""
        if self.kind == "numeric" and self.series.dtype.kind == "f":
            return self.series.to_numpy(dtype="float64", na_value=np.nan)
        return None

    @cached_property
    def missing_in_the_dtype(self) -> int:
        """The missing values, counted without reading the values."""
        floats = self._floats
        if floats is not None:
            return int(np.count_nonzero(np.isnan(floats)))
        return int(self.series.isna().sum())

    # The distinct values and their counts.

    @cached_property
    def _counted(self) -> tuple[Levels, np.ndarray] | None:
        return count_levels_by_row(self.series, self.kind)

    @cached_property
    def levels(self) -> Levels | None:
        """The distinct values and their counts as pandas counts them.

        None when the values can't be hashed, such as lists.
        """
        counted = self._counted
        return None if counted is None else counted[0]

    @cached_property
    def distinct(self) -> int | None:
        """The number of distinct values that aren't missing; None when unhashable."""
        levels = self.levels
        return None if levels is None else len(levels.values)

    @cached_property
    def non_missing(self) -> int:
        """The number of values that aren't missing."""
        # The levels know it without grouping the values by their type.
        levels = self.levels
        return int((self.counts if levels is None else levels.counts).sum())

    @cached_property
    def _groups(self) -> _Groups:
        counted = self._counted
        series = self.series
        floats = self._floats
        if floats is not None and not isinstance(series.dtype, np.dtype):
            # pyarrow's floats keep -0.0 and 0.0 apart as levels, but they are
            # one value to every check of the values.
            found = value_codes(floats, unhashable="by_content")
            sizes = np.bincount(found.codes + 1, minlength=len(found.uniques) + 1)
            return _Groups(found.codes, np.asarray(found.uniques), sizes[1:], None)
        if counted is not None:
            levels, codes = counted
            if (
                self.kind in ("object", "other")
                and len(levels.values)
                and not levels.text
            ):
                values = np.asarray(series.array, dtype=object)
                return _split_by_type(values, codes)
            return _Groups(codes, levels.values, levels.counts, None)
        values = np.asarray(series.array, dtype=object)
        found = value_codes(values, unhashable="by_content")
        return _split_by_type(values, found.codes)

    @property
    def distinct_values(self) -> "pd.Index | np.ndarray":
        """One value per group, in order of first appearance."""
        return self._groups.labels

    @property
    def counts(self) -> np.ndarray:
        """The rows each group's value holds."""
        return self._groups.counts

    @property
    def codes(self) -> np.ndarray:
        """Each row's group, -1 where the value is missing."""
        return self._groups.codes

    @cached_property
    def _objects(self) -> np.ndarray:
        """The groups' values as Python objects."""
        objects = self._groups.objects
        if objects is not None:
            return objects
        levels = self.levels
        assert levels is not None  # unhashable values are always split
        return levels.objects()

    # What each value is.

    @cached_property
    def _all_text(self) -> bool:
        """Whether every group is a string, so none needs reading as a Python object."""
        levels = self.levels
        if self.kind == "text":
            return True
        return (
            levels is not None
            and self._groups.objects is None
            and self.kind not in ("numeric", "boolean", "datetime", "timedelta")
            and levels.text
        )

    @cached_property
    def kinds(self) -> list[ValueKind] | ValueKind:
        """The kind of each group's value, or the one kind they all have."""
        if self.kind == "numeric":
            return "number"
        if self.kind in ("boolean", "datetime", "timedelta", "text"):
            return self.kind
        if self._all_text:
            return "text"
        return [value_kind(value) for value in self._objects]

    @cached_property
    def kind_counts(self) -> dict[ValueKind, int]:
        """How many values are of each kind."""
        totals = dict.fromkeys(VALUE_KINDS, 0)
        kinds = self.kinds
        if isinstance(kinds, str):
            totals[kinds] = self.non_missing
            return totals
        for kind, count in zip(kinds, self.counts.tolist(), strict=True):
            totals[kind] += count
        return totals

    @cached_property
    def types(self) -> list[str]:
        """The names of the values' Python types, as summarize_categorical gives them."""
        return value_types(self.series, self.kind, self.levels)

    @cached_property
    def _is_text(self) -> np.ndarray | None:
        """Which groups are text; None when all of them are."""
        kinds = self.kinds
        if isinstance(kinds, str):
            return None if kinds == "text" else np.zeros(len(self.counts), dtype=bool)
        return np.array([kind == "text" for kind in kinds], dtype=bool)

    @cached_property
    def _numbers(self) -> np.ndarray:
        """A numeric column's distinct values as floats."""
        labels = self._groups.labels
        if isinstance(labels, pd.Index):
            return np.asarray(labels.to_numpy(dtype="float64", na_value=np.nan))
        return np.asarray(labels, dtype="float64")

    # How the text values read.

    @cached_property
    def _text(self) -> _TextFacts:
        is_text = self._is_text
        at = None if is_text is None else np.flatnonzero(is_text)
        objects = self._objects
        texts = objects if at is None else objects[at]
        reading = read_texts(texts)
        iso_date, date_like = reading.iso_date, reading.date_like
        if self.datetime_format is not None and len(texts):
            iso_date = iso_date.copy()
            rest = np.flatnonzero(~iso_date)
            iso_date[rest] = read_datetime_format(texts[rest], self.datetime_format)
            date_like = date_like & ~iso_date
        counts = self.counts if at is None else self.counts[at]
        return _TextFacts(at, reading, iso_date, date_like, counts)

    def _weight(self, mask: np.ndarray) -> int:
        return int(self._text.counts[mask].sum())

    # Verdicts.

    def verdict(self, logical_type: LogicalType) -> np.ndarray:
        """Each group's verdict on the logical type: COMPATIBLE, CONVERTIBLE or INVALID."""
        found = self._verdicts.get(logical_type)
        if found is None:
            found = self._verdicts[logical_type] = self._verdict(logical_type)
        return found

    def _verdict(self, declared: LogicalType) -> np.ndarray:
        kind, count = self.kind, len(self.counts)
        if kind == "numeric":
            return self._number_verdicts(declared)
        if kind in ("boolean", "datetime", "timedelta"):
            return np.full(count, _stored_verdict(kind, declared), dtype=np.uint8)
        is_text = self._is_text
        if is_text is None:
            return self._text_verdicts(declared)
        verdicts = np.empty(count, dtype=np.uint8)
        verdicts[is_text] = self._text_verdicts(declared)
        kinds = typing.cast("list[ValueKind]", self.kinds)
        objects = self._objects
        in_categorical = kind == "categorical"
        for position in np.flatnonzero(~is_text).tolist():
            verdicts[position] = _value_verdict(
                objects[position],
                kinds[position],
                declared,
                in_categorical=in_categorical,
            )
        return verdicts

    def _number_verdicts(self, declared: LogicalType) -> np.ndarray:
        """Integers, floats or decimals, by their dtype."""
        numbers = self._numbers
        if declared == self.storage:
            verdicts = np.full(len(numbers), COMPATIBLE, dtype=np.uint8)
        elif declared == "integer":
            whole = np.isfinite(numbers) & (numbers == np.floor(numbers))
            verdicts = np.where(whole, CONVERTIBLE, INVALID).astype(np.uint8)
        elif declared == "boolean":
            zero_or_one = (numbers == 0) | (numbers == 1)
            verdicts = np.where(zero_or_one, CONVERTIBLE, INVALID).astype(np.uint8)
        elif declared in ("float", "text", "categorical"):
            verdicts = np.full(len(numbers), CONVERTIBLE, dtype=np.uint8)
        else:  # datetime and timedelta
            verdicts = np.full(len(numbers), INVALID, dtype=np.uint8)
        return verdicts

    def _text_verdicts(self, declared: LogicalType) -> np.ndarray:
        """The verdict on each text group, by the shared parse rules."""
        in_categorical = self.kind == "categorical"
        size = len(self.counts) if self._is_text is None else len(self._text.counts)
        if declared == "text":
            verdict = CONVERTIBLE if in_categorical else COMPATIBLE
            return np.full(size, verdict, dtype=np.uint8)
        if declared == "categorical":
            verdict = COMPATIBLE if in_categorical else CONVERTIBLE
            return np.full(size, verdict, dtype=np.uint8)
        if declared == "timedelta" or not size:
            return np.full(size, INVALID, dtype=np.uint8)
        facts = self._text
        if declared == "boolean":
            converts = facts.reading.boolean
        elif declared == "integer":
            converts = facts.reading.whole
        elif declared == "float":
            converts = facts.reading.number
        else:  # datetime
            converts = facts.iso_date
        return np.where(converts, CONVERTIBLE, INVALID).astype(np.uint8)

    # The counts infer_column_semantics reports.

    def _reads_as(self, logical_type: LogicalType) -> int:
        """How many values are or read as the logical type."""
        return int(self.counts[self.verdict(logical_type) != INVALID].sum())

    @cached_property
    def boolean(self) -> int:
        """Values that are booleans or read as one."""
        return self._reads_as("boolean")

    @cached_property
    def number(self) -> int:
        """Values that are numbers or parse as one."""
        return self._reads_as("float")

    @cached_property
    def whole(self) -> int:
        """Values that are whole numbers or parse as one."""
        return self._reads_as("integer")

    @cached_property
    def datetime(self) -> int:
        """Values that are datetimes or ISO 8601 text, or text in the format."""
        return self._reads_as("datetime")

    @cached_property
    def timedelta(self) -> int:
        """Values that are durations."""
        return self._reads_as("timedelta")

    @cached_property
    def leading_zero(self) -> int:
        """Text values that are whole numbers written with a leading zero."""
        return self._text_count(lambda facts: facts.reading.leading_zero)

    @cached_property
    def date_like(self) -> int:
        """Text values that look like dates without parsing as one."""
        return self._text_count(lambda facts: facts.date_like)

    @cached_property
    def text_numbers(self) -> int:
        """Text values that parse as numbers."""
        return self._text_count(lambda facts: facts.reading.number)

    @cached_property
    def text_dates(self) -> int:
        """Text values that parse as dates, in ISO 8601 or the format."""
        return self._text_count(lambda facts: facts.iso_date)

    @cached_property
    def _plain_and_words(self) -> tuple[int, int]:
        """The text values that read as nothing else, and those with several words."""
        if not self.kind_counts["text"]:
            return 0, 0
        facts = self._text
        # With leading zeros, a column's numbers are codes: they count as text.
        readable = facts.iso_date | facts.reading.boolean
        if not self.leading_zero:
            readable = readable | facts.reading.number
        return (
            self._weight(~readable),
            self._weight(~readable & facts.reading.several_words),
        )

    @property
    def plain_text(self) -> int:
        """Text values that read as nothing else: no number, boolean or date."""
        return self._plain_and_words[0]

    @property
    def several_words(self) -> int:
        """Plain text values that hold several words."""
        return self._plain_and_words[1]

    def _text_count(self, flag: Callable[[_TextFacts], np.ndarray]) -> int:
        if not self.kind_counts["text"]:
            return 0
        return self._weight(flag(self._text))

    @cached_property
    def span(self) -> float | None:
        """max - min + 1 of the values stored as numbers; None without any, or
        when one isn't finite."""
        kind, levels = self.kind, self.levels
        if kind == "numeric":
            if not self.non_missing:
                return None
            if self.series.dtype.kind in "iu":
                found = None if levels is None else levels.sort_values
                assert found is not None  # integers sort as NumPy numbers
                return int(found.max()) - int(found.min()) + 1
            numbers = self._numbers
            if not np.isfinite(numbers).all():
                return None
            return float(numbers.max()) - float(numbers.min()) + 1
        if kind in ("boolean", "datetime", "timedelta") or self._all_text:
            return None
        if levels is None:
            return number_span(self._objects)
        # The values pandas counts as one level, such as 1 and True, give one number.
        return number_span(levels.objects())

    # The counts validate_column_types reports.

    def _uniform_verdict(self, declared: LogicalType) -> int | None:
        """The verdict on every value, when the dtype decides it and none is invalid.

        Then no value needs reading. None when the values decide.
        """
        kind = self.kind
        if kind == "categorical":
            return COMPATIBLE if declared == "categorical" else None
        if kind in ("boolean", "datetime", "timedelta"):
            verdict = _stored_verdict(kind, declared)
            return None if verdict == INVALID else verdict
        if kind == "numeric":
            if declared == self.storage:
                return COMPATIBLE
            if declared in ("float", "text", "categorical"):
                return CONVERTIBLE
            return None
        if kind == "text" or (
            # Objects that are all strings, judged without counting them.
            kind == "object"
            and declared in ("text", "categorical")
            and pd.api.types.infer_dtype(self.series, skipna=True)
            in ("string", "empty")
        ):
            if declared == "text":
                return COMPATIBLE
            return CONVERTIBLE if declared == "categorical" else None
        return None

    def check(self, declared: LogicalType) -> Check:
        """Sort the values by how they relate to the declared logical type.

        Keeps the positions of every row that holds an invalid value.
        """
        uniform = self._uniform_verdict(declared)
        if uniform is not None:
            missing = self.missing_in_the_dtype
            by_verdict = [0, 0, 0]
            by_verdict[uniform] = len(self.series) - missing
            return Check(
                compatible=by_verdict[COMPATIBLE],
                convertible=by_verdict[CONVERTIBLE],
                invalid=0,
                missing=missing,
                stored_nan=self.stored_nan,
                invalid_rows=np.zeros(0, dtype=np.intp),
                invalid_values=levels_of(
                    pd.Index([], dtype=object), np.zeros(0, dtype=np.intp)
                ),
            )
        verdicts = self.verdict(declared)
        sizes = self.counts
        by_verdict = [int(sizes[verdicts == v].sum()) for v in range(3)]
        invalid = verdicts == INVALID
        # Code -1 picks the appended False: a missing value is never invalid.
        invalid_rows = np.flatnonzero(np.append(invalid, False)[self.codes])
        positions = np.flatnonzero(invalid)
        return Check(
            compatible=by_verdict[COMPATIBLE],
            convertible=by_verdict[CONVERTIBLE],
            invalid=by_verdict[INVALID],
            missing=len(self.series) - self.non_missing,
            stored_nan=self.stored_nan,
            invalid_rows=invalid_rows,
            invalid_values=levels_of(
                _index(self.distinct_values)[positions], sizes[positions]
            ),
        )


def _index(values: "pd.Index | np.ndarray") -> pd.Index:
    """Values as a pandas Index, object arrays kept as objects."""
    if isinstance(values, pd.Index):
        return values
    if values.dtype == object:
        # pandas 3 would otherwise infer str for an array of strings.
        return pd.Index(values, dtype=object)
    return pd.Index(values)


def _split_by_type(values: np.ndarray, codes: np.ndarray) -> _Groups:
    """Groups of Python objects, told apart by their own type.

    pandas counts 1, 1.0 and True as one value, so the rows of each value are
    split by the type they hold.
    """
    types = np.fromiter(map(type, values), dtype=object, count=len(values))
    type_codes, type_uniques = pd.factorize(types)
    present = codes >= 0
    if len(type_uniques) > 1:
        combined = codes[present] * len(type_uniques) + type_codes[present]
        codes = codes.copy()
        codes[present] = pd.factorize(combined)[0]
    representatives = values[_first_positions(codes)]
    counts = np.bincount(codes[present], minlength=len(representatives))
    return _Groups(codes, representatives, counts, representatives)


def _first_positions(codes: np.ndarray) -> np.ndarray:
    """Where each code first occurs, for codes numbered in order of appearance.

    Each code is one more than the largest before it where it first occurs.
    """
    running = np.maximum.accumulate(codes) if len(codes) else codes
    return np.flatnonzero(np.diff(running, prepend=-1) > 0)


def _stored_verdict(kind: ColumnKind, declared: LogicalType) -> int:
    """The verdict on a value stored as a boolean, a datetime or a duration."""
    # These column kinds are named as the logical types they store.
    if kind == declared:
        return COMPATIBLE
    # Every such value converts to text, and can be hashed as a category.
    return CONVERTIBLE if declared in ("text", "categorical") else INVALID


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


def number_span(values: Iterable[object]) -> float | None:
    """max - min + 1 of the numbers among the values; None without one or with a
    number that isn't finite."""
    try:
        numbers = [
            float(typing.cast("Any", v)) for v in values if value_kind(v) == "number"
        ]
    except OverflowError:  # an int too large for a float
        return None
    if not numbers or not all(map(math.isfinite, numbers)):
        return None
    return max(numbers) - min(numbers) + 1
