"""The built-in rules that find values that look like sentinel values.

detect_missing_sentinels suspects a value when one of these fixed rules holds
for it, and never otherwise. No rule has a threshold to tune:

- builtin_list: a value from the built-in lists below. Numbers count at the
  column's minimum or maximum, text in any mix of upper and lower case, and
  dates at midnight.
- isolated_extreme: a number that occurs more than once at the column's
  minimum or maximum, and lies farther from the nearest other value than the
  other values span. It needs at least two other distinct values, so neither
  value of a column holding two, such as 0 and 1, is suspected.

The rules read the finite values that are neither missing nor declared, and a
column of Python objects value by value: its numbers as numbers, its text as
text and its dates and datetimes as such. Numbers and datetimes are compared
with NumPy and pandas over the whole column, text with one hash lookup per
row, and categories and decimals once per distinct value, so no Python loop
runs over a million values.
"""

import datetime as dt
import itertools
import math
import numbers
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal, cast

import numpy as np
import pandas as pd

from agent_eda._columns import Column, column_kind
from agent_eda._levels import stored_label
from agent_eda._sentinels import (
    Family,
    Holds,
    JsonSentinel,
    SentinelValue,
    can_match,
    holds,
    json_value,
    matches,
    numpy_dtype,
    type_family,
)

# Numbers often used for missing values; they count at the column's minimum
# or maximum.
LISTED_NUMBERS = (-9, -99, -999, -9999, 99, 999, 9999, 99999, 999999)
# Text often used for missing values, in any mix of upper and lower case;
# empty and whitespace-only text count too.
LISTED_TEXT = ("NA", "N/A", "NULL", "None", "nan", "-", "?", "unknown", "missing")
# Dates often used for missing values. A datetime counts at midnight on one of
# them, in its own time zone or, when it has one, in UTC.
LISTED_DATES = (
    dt.date(1900, 1, 1),
    dt.date(1970, 1, 1),
    dt.date(2099, 12, 31),
    dt.date(9999, 12, 31),
)


def _cases(word: str) -> list[str]:
    """Every way to write a word in upper and lower case, such as 'na' and 'Na'."""
    letters = [sorted({letter.lower(), letter.upper()}) for letter in word]
    return ["".join(spelling) for spelling in itertools.product(*letters)]


# The listed text in every mix of case, and empty text: 306 strings, which a
# hash lookup finds among any number of values in one pass.
_TEXT_SPELLINGS = frozenset(
    {"", *(text for word in LISTED_TEXT for text in _cases(word))}
)
_SORTED_SPELLINGS = sorted(_TEXT_SPELLINGS)

Evidence = Literal["builtin_list", "isolated_extreme"]

# The families the rules read; values of others, such as booleans, are never
# candidates.
_READ: tuple[Family, ...] = ("number", "text", "date", "datetime")


@dataclass(frozen=True)
class Suspect:
    """A value the rules suspect, where it occurs and why."""

    value: object
    # The rows that hold it, in row order.
    positions: np.ndarray
    evidence: tuple[Evidence, ...]


def read_by_rules(column: Holds) -> bool:
    """Whether a rule reads any of the column's values: numbers, text, dates or
    datetimes, or any value at all when a column of objects has none."""
    return column.families is None or not column.families.isdisjoint(_READ)


def suspects(
    series: pd.Series, column: Column, holding: Holds, remaining: np.ndarray
) -> list[Suspect]:
    """The values of a column the rules suspect, among the rows in `remaining`.

    `holding` is what the column holds, as _sentinels.holds() tells. The most
    frequent values come first, and values as frequent as each other in the
    order of their first row.
    """
    kind = column.kind
    if kind == "numeric" and getattr(series.dtype, "kind", "O") in "iuf":
        found = _number_suspects(series, remaining)
    elif kind == "numeric":  # pyarrow decimals
        found = _distinct_suspects(*_factorized(series, remaining))
    elif kind == "text" or (kind == "object" and holding.families == {"text"}):
        found = _text_suspects(series, remaining)
    elif kind == "categorical":
        found = _distinct_suspects(*_categories(series, remaining))
    elif kind == "datetime":
        found = _date_suspects(series, holding, remaining)
    elif kind == "object" and holding.families is not None:
        found = _object_suspects(series, remaining)
    else:
        found = []
    return sorted(found, key=lambda s: (-len(s.positions), s.positions[0]))


def _number_suspects(series: pd.Series, remaining: np.ndarray) -> list[Suspect]:
    """The suspected numbers of a numeric column, read as a NumPy array."""
    values, finite = _numbers(series)
    checked = remaining & finite
    kept = values[checked]
    if not kept.size:
        return []
    return [
        Suspect(stored_label(value), np.flatnonzero(checked & (values == value)), why)
        for value, why in _number_hits(kept).items()
    ]


def _numbers(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """A numeric column as a NumPy array of its own type, and where it is finite.

    Missing values read as 0 in an integer array and NaN in a float one, so
    only the rows the caller keeps count. The array can be a read-only view of
    the caller's data: never write to it.
    """
    dtype = series.dtype
    if isinstance(dtype, np.dtype):
        values = series.to_numpy()
    else:
        own = numpy_dtype(dtype)
        values = series.to_numpy(dtype=own, na_value=0 if own.kind in "iu" else np.nan)
    return values, _finite(values)


def _finite(values: np.ndarray) -> np.ndarray:
    """Which numbers of a NumPy numeric array are finite."""
    if values.dtype.kind == "f":
        return np.isfinite(values)
    return np.ones(len(values), dtype=bool)


def _exact(value: object) -> Fraction | None:
    """A finite real number as an exact fraction, so that numbers of any Python
    type compare by value; None for any other value, such as a complex number."""
    value = stored_label(value)
    if isinstance(value, Decimal):
        return Fraction(value) if value.is_finite() else None
    if isinstance(value, numbers.Rational):
        return Fraction(value)
    if isinstance(value, numbers.Real):  # floats, NumPy's ones as Python's
        number = float(value)
        return Fraction(number) if math.isfinite(number) else None
    return None


def _number_hits(
    values: np.ndarray, counts: np.ndarray | None = None
) -> dict[object, tuple[Evidence, ...]]:
    """The suspected numbers among finite ones, each with its evidence.

    `values` holds one number per row, or with `counts`, distinct numbers and
    how often each occurs. An object array of Python numbers, such as
    decimals, compares exactly too.
    """
    low, high = values.min(), values.max()
    if counts is None:
        low_count = int(np.count_nonzero(values == low))
        high_count = int(np.count_nonzero(values == high))
    else:
        low_count = int(counts[values == low].sum())
        high_count = int(counts[values == high].sum())
    found: dict[object, list[Evidence]] = {}
    for end in dict.fromkeys([low, high]):
        if end in LISTED_NUMBERS:
            found.setdefault(end, []).append("builtin_list")
    # The other values must span a range, so they hold two distinct values.
    if low_count >= 2:
        above = values[values > low]
        if above.size and above.min() != high:
            nearest = above.min()
            if _gap(low, nearest) > _gap(nearest, high):
                found.setdefault(low, []).append("isolated_extreme")
    if high_count >= 2:
        below = values[values < high]
        if below.size and below.max() != low:
            nearest = below.max()
            if _gap(nearest, high) > _gap(low, nearest):
                found.setdefault(high, []).append("isolated_extreme")
    return {value: tuple(evidence) for value, evidence in found.items()}


def _gap(lower: object, upper: object) -> Fraction:
    """The exact distance between two finite numbers, without rounding."""
    return Fraction(cast("int", stored_label(upper))) - Fraction(
        cast("int", stored_label(lower))
    )


def _factorized(
    series: pd.Series, remaining: np.ndarray
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """A column's distinct values, how often the read rows hold each, and per
    row the position of its value, -1 for a row not read."""
    codes, uniques = pd.factorize(series)
    codes = np.where(remaining, np.asarray(codes, dtype=np.int64), -1)
    values = uniques if isinstance(uniques, pd.Index) else pd.Index(uniques)
    return values, np.bincount(codes[codes >= 0], minlength=len(values)), codes


def _categories(
    series: pd.Series, remaining: np.ndarray
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """A categorical column's categories, how often the read rows take each,
    and per row the position of its category, -1 for a row not read."""
    categorical = cast("pd.Categorical", series.array)
    codes = np.where(remaining, np.asarray(categorical.codes, dtype=np.int64), -1)
    categories = categorical.categories
    return categories, np.bincount(codes[codes >= 0], minlength=len(categories)), codes


def _object_suspects(series: pd.Series, remaining: np.ndarray) -> list[Suspect]:
    """The suspected values of a column of Python objects of several families.

    Each family is counted on its own, so True and 1, which Python counts as
    equal, stay apart.
    """
    objects = np.asarray(series.array, dtype=object)
    present = np.flatnonzero(remaining)
    values = objects[present]
    families = _families(values)
    found = []
    for family in _READ:
        where = np.flatnonzero(families == family)
        if not where.size:
            continue
        if family == "text":
            # Row by row: factorizing text takes '' and text that starts with
            # '\x00' for one value.
            found.extend(_text_rows_suspects(series, present[where]))
            continue
        family_codes, uniques = pd.factorize(values[where])
        codes = np.full(len(objects), -1, dtype=np.int64)
        codes[present[where]] = family_codes
        counts = np.bincount(family_codes, minlength=len(uniques))
        found.extend(
            _family_suspects(pd.Index(uniques, dtype=object), counts, codes, family)
        )
    return found


def _distinct_suspects(
    values: pd.Index, counts: np.ndarray, codes: np.ndarray
) -> list[Suspect]:
    """The suspected values among a column's distinct values.

    `counts` says how often the read rows hold each value, and `codes` which
    value each row holds, -1 for a row not read. Distinct values of Python
    objects, such as categories of mixed types, are judged by family.
    """
    if column_kind(values.dtype) != "object":
        family = _family_of(values)
        if family is None:
            return []
        return _family_suspects(values, counts, codes, family)
    if pd.api.types.infer_dtype(values, skipna=False) == "string":
        return _family_suspects(values, counts, codes, "text")
    families = _families(values)
    found = []
    for family in _READ:
        positions = np.flatnonzero(families == family)
        if positions.size:
            found.extend(
                _family_suspects(values, counts, codes, family, positions=positions)
            )
    return found


def _families(values: np.ndarray | pd.Index) -> np.ndarray:
    """The family of each Python object, looked up once per type."""
    kinds = {kind: type_family(kind) for kind in set(map(type, values))}
    return np.array([kinds[type(value)] for value in values], dtype=object)


def _family_of(values: pd.Index) -> Family | None:
    """The family the rules read the values of a dtype as, or None."""
    families = holds(pd.Series(values)).families
    if families is None or len(families) != 1:
        return None
    [family] = families
    return family if family in _READ else None


def _family_suspects(
    values: pd.Index,
    counts: np.ndarray,
    codes: np.ndarray,
    family: Family,
    positions: np.ndarray | None = None,
) -> list[Suspect]:
    """The suspected values among distinct values of one family.

    `positions` picks the values of the family, all of them by default.
    """
    if positions is None:
        positions = np.arange(len(values))
    positions = positions[counts[positions] > 0]
    found = []
    for index, why in _hits(values[positions], counts[positions], family):
        position = positions[index]
        found.append(
            Suspect(
                stored_label(values[position]), np.flatnonzero(codes == position), why
            )
        )
    return found


def _hits(
    values: pd.Index, counts: np.ndarray, family: Family
) -> list[tuple[int, tuple[Evidence, ...]]]:
    """The positions of the suspected values among distinct values of one
    family, each with its evidence."""
    if not len(values):
        return []
    if family == "number":
        stored = values.to_numpy()
        if stored.dtype.kind in "iuf":
            kept = np.flatnonzero(_finite(stored))
            numbers = stored[kept]
        else:  # Python objects, such as decimals, compared as exact fractions
            exact = [_exact(value) for value in stored]
            kept = np.flatnonzero([number is not None for number in exact])
            numbers = np.array([exact[i] for i in kept], dtype=object)
        if not kept.size:
            return []
        hits = _number_hits(numbers, counts[kept])
        return [
            (int(kept[np.flatnonzero(numbers == value)[0]]), why)
            for value, why in hits.items()
        ]
    if family == "text":
        listed = _listed_texts(values, np.arange(len(values)))
    elif family == "date":
        listed = [i for i, day in enumerate(values) if day in LISTED_DATES]
    else:
        listed = [i for i, stamp in enumerate(values) if _listed_datetime(stamp)]
    return [(int(i), ("builtin_list",)) for i in listed]


def _text_suspects(series: pd.Series, remaining: np.ndarray) -> list[Suspect]:
    """The listed texts of a column that holds only text, found row by row."""
    return _text_rows_suspects(series, np.flatnonzero(remaining))


def _text_rows_suspects(series: pd.Series, rows: np.ndarray) -> list[Suspect]:
    """The listed texts among the given rows of text, found row by row.

    A hash lookup of every value among the listed spellings is several times
    faster than counting each distinct value of text with many, and only the
    few rows that match are grouped by value. None of them holds '\\x00',
    which pandas' factorize would take for the end of the text.
    """
    hits = _listed_texts(series, rows)
    if not hits.size:
        return []
    codes, uniques = pd.factorize(np.asarray(series.iloc[hits], dtype=object))
    return [
        Suspect(stored_label(text), hits[codes == code], ("builtin_list",))
        for code, text in enumerate(uniques)
    ]


def _listed_texts(texts: pd.Series | pd.Index, rows: np.ndarray) -> np.ndarray:
    """The rows whose text is on the built-in list, in any mix of case, or is
    empty or whitespace only, as Python's str.isspace() decides.

    `rows` picks the texts to check, which must not be missing.
    """
    dtype = texts.dtype
    if isinstance(dtype, pd.ArrowDtype) or getattr(dtype, "storage", "") == "pyarrow":
        listed = _as_bools(texts.isin(_SORTED_SPELLINGS))[rows]
        # pyarrow agrees with Python's str.isspace() on every character.
        spaces = _as_bools(pd.Series(texts).str.isspace())[rows]
        return rows[listed | spaces]
    # Python strings, which C-level maps read faster than pandas' own loops.
    objects = np.asarray(texts, dtype=object)[rows]
    # One pass finds a superset: text that is listed once stripped of
    # surrounding whitespace, as whitespace-only text is, as ''.
    maybe = np.fromiter(
        map(_TEXT_SPELLINGS.__contains__, map(str.strip, objects)),
        dtype=bool,
        count=len(rows),
    )
    return np.asarray(
        [
            row
            for row, text in zip(rows[maybe], objects[maybe], strict=True)
            if text in _TEXT_SPELLINGS or text.isspace()
        ],
        dtype=np.intp,
    )


def _as_bools(values: pd.Series | pd.Index | np.ndarray) -> np.ndarray:
    """Booleans as a NumPy array, False where one is missing."""
    return np.asarray(pd.Series(values).to_numpy(dtype=bool, na_value=False))


def _date_suspects(
    series: pd.Series, holding: Holds, remaining: np.ndarray
) -> list[Suspect]:
    """The listed dates in a column of datetimes or dates, matched as instants."""
    found = []
    for target in _date_targets(holding):
        positions = np.flatnonzero(matches(series, target, ~remaining))
        if positions.size:
            found.append(
                Suspect(series.iloc[positions[0]], positions, ("builtin_list",))
            )
    return found


def _date_targets(column: Holds) -> list[dt.date]:
    """The instants of the listed dates that a column of datetimes or dates can hold.

    Dates are themselves; datetimes are midnight on each date in the column's
    time zone, and also in UTC when the column has one. Where the clocks went
    back at midnight, both midnights count; where they skipped it, none does.
    """
    if column.families == {"date"}:
        return [day for day in LISTED_DATES if can_match(day, column)]
    targets: list[pd.Timestamp] = []
    for day in LISTED_DATES:
        midnight = pd.Timestamp(day)
        if not isinstance(midnight, pd.Timestamp):  # pandas types it as maybe NaT
            continue
        if column.tz is None:
            targets.append(midnight)
            continue
        for earlier in (True, False):
            local = midnight.tz_localize(
                column.tz, ambiguous=earlier, nonexistent="NaT"
            )
            if isinstance(local, pd.Timestamp) and local not in targets:
                targets.append(local)
        utc = midnight.tz_localize("UTC")
        if isinstance(utc, pd.Timestamp) and utc not in targets:
            targets.append(utc)
    return [target for target in targets if can_match(target, column)]


def _listed_datetime(value: object) -> bool:
    """Whether a datetime is midnight on a listed date, in its time zone or UTC."""
    stamp = pd.Timestamp(cast("dt.datetime", value))
    if not isinstance(stamp, pd.Timestamp):  # pandas types it as maybe NaT
        return False
    if stamp == stamp.normalize() and stamp.date() in LISTED_DATES:
        return True
    if stamp.tz is None:
        return False
    utc = stamp.tz_convert("UTC")
    return utc == utc.normalize() and utc.date() in LISTED_DATES


def record_value(value: object) -> JsonSentinel:
    """A stored value as the record shows it, as declared values are shown.

    NumPy scalars become Python's, decimals floats, and dates and times ISO
    8601 text.
    """
    value = stored_label(value)
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if isinstance(value, Decimal):
        return float(value)
    return json_value(cast("SentinelValue", value))
