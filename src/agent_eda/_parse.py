"""The rules for reading a value as a boolean, a number, a date or a duration.

infer_column_semantics measures how many values each rule accepts, and
validate_column_types counts the values that convert under the same rules, so
the two analyses never disagree on what parses.

A value that isn't text is read by its Python type: booleans, numbers (int,
float, Decimal, Fraction and NumPy numbers), datetimes (datetime, date,
pd.Timestamp and np.datetime64), durations (timedelta, pd.Timedelta and
np.timedelta64), and everything else, such as lists. Text is read after
stripping surrounding whitespace, and only ASCII text parses:

- A number is a decimal number such as 42, -3.5, .5, 1e-3 or +4.2E1. Thousands
  separators (1,000), decimal commas (3,5), hexadecimal, inf and nan don't
  parse, and neither do whole numbers written with a leading zero, such as 007
  or 01234: they are codes, which a number would change by dropping the zeros.
- A whole number is a number without a fractional part, such as 42, 42.0 or
  4.2e1, decided on the digits, not on a float.
- An ISO 8601 date is a calendar date YYYY-MM-DD, alone or followed by T or a
  space and hh:mm, hh:mm:ss or hh:mm:ss.fraction (1 to 9 digits), and then
  optionally Z or an offset +hh, +hh:mm or +hhmm. The date and time must exist:
  years 0001 to 9999, months 01 to 12, days that exist in that month, hours 00
  to 23, minutes and seconds 00 to 59. No other format parses, so day and month
  order is never guessed.
- Date-like text looks like a date without being ISO 8601, such as 01/02/2020,
  2.1.2020, 2020/01/02, 2020-1-2, 5 Jan 2020, January 5, 2020 or 2020-13-45.
  It is recognised, never parsed.
- A boolean is one of the words true, false, t, f, yes, no, y and n in any
  case, or a number equal to 0 or 1, such as 0, 1 or 1.0.
"""

import calendar
import datetime as dt
import math
import numbers
import re
import typing
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal

import numpy as np

# What a value is by its Python type.
ValueKind = Literal["boolean", "number", "text", "datetime", "timedelta", "other"]
VALUE_KINDS: tuple[ValueKind, ...] = typing.get_args(ValueKind)

# The words that read as booleans, in any case; 0 and 1 read as numbers do.
BOOLEAN_WORDS = frozenset(("true", "false", "t", "f", "yes", "no", "y", "n"))

_NUMBER = re.compile(
    r"(?P<sign>[+-])?"
    r"(?:(?P<integer>[0-9]+)(?:\.(?P<fraction>[0-9]*))?|\.(?P<point_fraction>[0-9]+))"
    r"(?:[eE](?P<exponent>[+-]?[0-9]+))?"
)
_ISO_DATE = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"(?:[T ](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:\.[0-9]{1,9})?)?"
    r"(?:Z|(?P<offset_hours>[+-][0-9]{2})(?::?(?P<offset_minutes>[0-9]{2}))?)?)?"
)
_TIME = (
    r"(?:[ T][0-9]{1,2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?"
    r"(?: ?[ap]\.?m\.?)?(?: ?(?:z|utc|gmt|[+-][0-9]{2}:?[0-9]{2}))?)?"
)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_DAY = r"[0-9]{1,2}(?:st|nd|rd|th)?"
_YEAR = r"(?:[0-9]{4}|[0-9]{2})"
_DATE_LIKE = re.compile(
    r"(?:"
    # 2020/01/02, 2020-1-2 or 2020.1.2
    r"[0-9]{4}(?P<first>[-/.])[0-9]{1,2}(?P=first)[0-9]{1,2}"
    # 01/02/2020, 1.2.20 or 01-02-2020
    rf"|[0-9]{{1,2}}(?P<second>[-/.])[0-9]{{1,2}}(?P=second){_YEAR}"
    # 5 Jan 2020, 05-Jan-20 or 5th January, 2020
    rf"|{_DAY}[ -]{_MONTH},?[ -]{_YEAR}"
    # Jan 5, 2020 or January 5th 2020
    rf"|{_MONTH}[ -]{_DAY},?[ -]{_YEAR}"
    # Jan 2020
    rf"|{_MONTH}[ -][0-9]{{4}}"
    rf"){_TIME}",
    re.IGNORECASE,
)

# A string can only parse, or look like a date, when its first character is
# one of these: a digit, a sign, a point, whitespace, or the first letter of a
# boolean word or a month name.
_STARTS = np.array(
    sorted(
        {ord(ch) for ch in "0123456789+-.tfynjmasodTFYNJMASOD"}
        | {code for code in range(0x3001) if chr(code).isspace()}
    ),
    dtype=np.uint32,
)

# What _read finds in a string, as bits.
_NUMBER_BIT = 1
_WHOLE_BIT = 2
_LEADING_ZERO_BIT = 4
_BOOLEAN_BIT = 8
_ISO_DATE_BIT = 16
_DATE_LIKE_BIT = 32


@dataclass(frozen=True)
class TextReading:
    """How each of some strings reads, one boolean array entry per string."""

    # Parses as a number.
    number: np.ndarray
    # Parses as a whole number.
    whole: np.ndarray
    # A whole number written with a leading zero, such as 007: a code that
    # doesn't parse as a number.
    leading_zero: np.ndarray
    # Reads as a boolean.
    boolean: np.ndarray
    # Parses as an ISO 8601 date, with or without a time.
    iso_date: np.ndarray
    # Looks like a date without parsing as ISO 8601.
    date_like: np.ndarray
    # Holds a space between other characters: several words.
    several_words: np.ndarray


def read_texts(texts: np.ndarray) -> TextReading:
    """Read each string of an object array under the parse rules.

    Strings that can't parse, judged by their first character, cost one
    vectorized check; the rest are read one by one.
    """
    count = len(texts)
    several_words = np.array(
        [" " in text and " " in text.strip() for text in texts], dtype=bool
    )
    bits = np.zeros(count, dtype=np.uint8)
    if count:
        # Each string's first character as a code point, 0 for an empty one.
        firsts = texts.astype("U1").view(np.uint32)
        for position in np.flatnonzero(np.isin(firsts, _STARTS)).tolist():
            bits[position] = _read(texts[position])
    return TextReading(
        number=(bits & _NUMBER_BIT) != 0,
        whole=(bits & _WHOLE_BIT) != 0,
        leading_zero=(bits & _LEADING_ZERO_BIT) != 0,
        boolean=(bits & _BOOLEAN_BIT) != 0,
        iso_date=(bits & _ISO_DATE_BIT) != 0,
        date_like=(bits & _DATE_LIKE_BIT) != 0,
        several_words=several_words,
    )


def _read(text: str) -> int:
    """What one string parses as, as bits."""
    stripped = text.strip()
    if not stripped.isascii():
        return 0  # every rule reads ASCII text only; "" is ASCII
    if stripped.isdigit():
        if len(stripped) > 1 and stripped[0] == "0":
            return _LEADING_ZERO_BIT
        bits = _NUMBER_BIT | _WHOLE_BIT
        return bits | _BOOLEAN_BIT if stripped in ("0", "1") else bits
    if stripped.lower() in BOOLEAN_WORDS:
        return _BOOLEAN_BIT
    match = _NUMBER.fullmatch(stripped)
    if match:
        return _number_bits(match)
    match = _ISO_DATE.fullmatch(stripped)
    if match:
        return _ISO_DATE_BIT if _exists(match) else _DATE_LIKE_BIT
    if _DATE_LIKE.fullmatch(stripped):
        return _DATE_LIKE_BIT
    return 0


def _number_bits(match: re.Match[str]) -> int:
    """Whether a matched number is whole and equal to 0 or 1, from its digits."""
    integer = match["integer"] or ""
    if len(integer) > 1 and integer[0] == "0":
        return _LEADING_ZERO_BIT
    fraction = match["fraction"] or match["point_fraction"] or ""
    digits = integer + fraction
    significant = digits.rstrip("0")
    if not significant.strip("0"):  # the value is zero
        return _NUMBER_BIT | _WHOLE_BIT | _BOOLEAN_BIT
    # The power of ten of the last significant digit.
    scale = int(match["exponent"] or 0) - len(fraction) + len(digits) - len(significant)
    if scale < 0:
        return _NUMBER_BIT
    one = scale == 0 and significant.lstrip("0") == "1" and match["sign"] != "-"
    return _NUMBER_BIT | _WHOLE_BIT | (_BOOLEAN_BIT if one else 0)


def _exists(match: re.Match[str]) -> bool:
    """Whether a date and time in ISO 8601 form exists on the calendar and clock."""
    year, month, day = int(match["year"]), int(match["month"]), int(match["day"])
    if not (1 <= year <= 9999 and 1 <= month <= 12):
        return False
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        return False
    if match["hour"] is not None and (
        int(match["hour"]) > 23 or int(match["minute"]) > 59
    ):
        return False
    if match["second"] is not None and int(match["second"]) > 59:
        return False
    if match["offset_hours"] is not None and abs(int(match["offset_hours"])) > 23:
        return False
    return match["offset_minutes"] is None or int(match["offset_minutes"]) <= 59


def value_kind(value: object) -> ValueKind:
    """What a value that isn't missing is, by its Python type."""
    if isinstance(value, str):
        return "text"
    if isinstance(value, bool | np.bool_):
        return "boolean"
    if isinstance(value, numbers.Real | Decimal):
        return "number"
    # datetime.datetime and pd.Timestamp are dates too.
    if isinstance(value, dt.date | np.datetime64):
        return "datetime"
    if isinstance(value, dt.timedelta | np.timedelta64):
        return "timedelta"
    return "other"


def is_whole(number: object) -> bool:
    """Whether a number (by value_kind) has no fractional part."""
    if isinstance(number, int | np.integer):
        return True
    if isinstance(number, Fraction):
        return number.denominator == 1
    if isinstance(number, Decimal):
        return number.is_finite() and number == number.to_integral_value()
    value = float(typing.cast("float", number))
    return math.isfinite(value) and value.is_integer()


def is_zero_or_one(number: object) -> bool:
    """Whether a number (by value_kind) equals 0 or 1, so it reads as a boolean."""
    if isinstance(number, Decimal) and not number.is_finite():
        return False
    return bool(number == 0 or number == 1)
