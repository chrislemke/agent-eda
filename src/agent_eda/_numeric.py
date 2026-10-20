"""The numeric summary of one column, shared by every analysis that reports one.

summarize_numeric reports it for each numeric column and profile_target for a
regression target, both with numeric_entry(), so the two can't disagree. A
column is read as float64 with NaN for missing values, and every statistic uses
the finite values only.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._result_base import (
    ColumnSupport,
    ExclusionReason,
    Limitation,
    StatusReason,
)
from agent_eda._text import join_labels, plural
from agent_eda._values import is_constant, stored_nan_mask
from agent_eda._warnings import watch_warnings
from agent_eda.statistics._descriptive_results import SummarizeNumericColumn

# The statistics of the finite values, in the order results list them.
STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "trimmed_mean",
    "mad",
    "mad_scaled",
    "skewness",
    "excess_kurtosis",
)
# Finite values each statistic needs; the others need 1.
MINIMUM_COUNTS = {"std": 2, "skewness": 3, "excess_kurtosis": 4}
# Statistics that don't exist when every finite value is equal.
SHAPE_STATISTICS = ("skewness", "excess_kurtosis")

# Makes the MAD estimate the standard deviation of normally distributed data; the
# rounded 1 / Phi^-1(3/4) = 1.482602..., as R's mad() uses.
MAD_SCALE = 1.4826

# Integers up to this size, in absolute value, convert to float64 exactly.
EXACT_INTEGER_LIMIT = 2**53

# numpy.quantile's methods: Hyndman and Fan's nine definitions, then four older
# ones. numpy 1.26 and 2.x offer the same names.
QuantileMethod = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]

# summarize_numeric's defaults, which profile_target uses for a regression target.
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
DEFAULT_QUANTILE_METHOD: QuantileMethod = "linear"
DEFAULT_TRIM_FRACTION = 0.1

# Why a statistic is null: a limitation code.
NullReason = Literal["too_few_values", "zero_variance", "numerical_warning"]


@dataclass(frozen=True)
class ReadEffects:
    """What reading a column as float64 changed, which limitations report."""

    # Whether it held booleans, read as 0 and 1.
    boolean: bool
    # Whether it holds integers that float64 can't represent exactly.
    beyond_exact_integers: bool
    # NaN values that pandas' isna() doesn't count as missing, as pyarrow-backed
    # floats (and nullable floats on pandas 2) can store; read as missing.
    stored_nan_count: int


@dataclass(frozen=True)
class NumericColumn(ReadEffects):
    """A column read as float64, and what reading it that way changed."""

    # NaN where a value is missing. It can be a read-only view of the caller's
    # data: never write to it.
    values: np.ndarray
    # Whether at least two values aren't missing and all of them are equal,
    # decided on the stored values, not on their float64 copies.
    constant: bool
    # Whether that single value is inf or -inf rather than a finite value.
    constant_infinite: bool


def read_numeric(series: pd.Series) -> NumericColumn:
    """Read a numeric or boolean column as float64."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    exact = _exact_range(series)
    constant, infinite = _constancy(values, exact)
    return NumericColumn(
        values=values,
        boolean=series.dtype.kind == "b",
        beyond_exact_integers=_beyond_exact_integers(exact),
        stored_nan_count=stored_nan_count(series),
        constant=constant,
        constant_infinite=infinite,
    )


def stored_nan_count(series: pd.Series) -> int:
    """NaN values the column stores apart from missing ones; 0 for most dtypes.

    pyarrow-backed floats, and nullable floats on pandas 2, can store NaN as a
    value that pandas' isna() doesn't count.
    """
    found = stored_nan_mask(series)
    return 0 if found is None else int(found.sum())


@dataclass(frozen=True)
class _ExactRange:
    """The count, smallest and largest of the values of an integer or decimal column."""

    count: int
    low: int | Decimal
    high: int | Decimal


def _exact_range(series: pd.Series) -> _ExactRange | None:
    """The range of an integer or decimal column as stored, or None for others.

    Integers and decimals compare exactly here, where their float64 copies
    round values beyond 2^53 to the same number.
    """
    dtype = series.dtype
    is_integer = dtype.kind in "iu"
    # Arrow decimals, the only numeric columns of kind "O".
    is_decimal = isinstance(dtype, pd.ArrowDtype) and dtype.kind == "O"
    if not (is_integer or is_decimal):
        return None
    count = int(series.count())
    if count == 0:
        return _ExactRange(0, 0, 0)
    return _ExactRange(
        count,
        int(series.min()) if is_integer else series.min(),
        int(series.max()) if is_integer else series.max(),
    )


def _constancy(values: np.ndarray, exact: _ExactRange | None) -> tuple[bool, bool]:
    """Whether the column is constant, and whether its one value is infinite.

    The shared rule, on stored values: integers and decimals compare exactly, and
    floats as stored. NaN, stored or not, is missing.
    """
    if exact is not None:
        return is_constant(exact.count, 1 if exact.low == exact.high else 2), False
    present = values[~np.isnan(values)]
    if present.size < 2:
        return False, False
    low, high = present.min(), present.max()
    constant = bool(low == high)
    return constant, constant and bool(np.isinf(low))


def stored_nan_limitations(
    names: Sequence[object], fields: list[str], effect: str
) -> list[Limitation]:
    """The limitation for columns that store NaN as a value; none without columns.

    Every analysis counts that NaN as missing, which pandas' missing-value check
    doesn't. `effect` says what the analysis does with it, such as
    "missing_count counts it", and `fields` name the fields where it differs.
    """
    if not names:
        return []
    store = plural(len(names), "column stores", "columns store")
    return [
        Limitation(
            code="nan_counted_as_missing",
            reason=(
                f"{store} NaN as a value, which pandas doesn't count as missing; "
                f"{effect}: {join_labels(names)}."
            ),
            fields=fields,
        )
    ]


def _beyond_exact_integers(exact: _ExactRange | None) -> bool:
    if exact is None or exact.count == 0:
        return False
    return exact.high > EXACT_INTEGER_LIMIT or exact.low < -EXACT_INTEGER_LIMIT


def read_limitations(
    columns: Mapping[str, ReadEffects], path: Callable[[str], str]
) -> list[Limitation]:
    """The limitations that reading the columns as float64 brings.

    `path` gives the result path of a column's entry, such as "columns['x']".
    """
    limitations = []
    booleans = [name for name, column in columns.items() if column.boolean]
    if booleans:
        were = plural(len(booleans), "boolean column was", "boolean columns were")
        limitations.append(
            Limitation(
                code="boolean_as_numeric",
                reason=(
                    f"{were} read as 0 for false and 1 for true: {join_labels(booleans)}."
                ),
                fields=[path(name) for name in booleans],
            )
        )
    beyond = [name for name, column in columns.items() if column.beyond_exact_integers]
    if beyond:
        holds = plural(len(beyond), "column holds", "columns hold")
        its = "its" if len(beyond) == 1 else "their"
        limitations.append(
            Limitation(
                code="float_precision",
                reason=(
                    f"{holds} integers or decimals beyond ±2^53 = {EXACT_INTEGER_LIMIT:,}, which "
                    f"float64 can't represent exactly, so {its} statistics may be "
                    f"rounded: {join_labels(beyond)}."
                ),
                fields=[path(name) for name in beyond],
            )
        )
    stored = [name for name, column in columns.items() if column.stored_nan_count]
    limitations += stored_nan_limitations(
        stored,
        [f"{path(name)}.missing_count" for name in stored],
        "missing_count counts it",
    )
    return limitations


@dataclass(frozen=True)
class NumericSummary:
    """Counts and statistics of one column's values."""

    non_missing_count: int
    missing_count: int
    infinite_count: int
    zero_count: int
    negative_count: int
    statistics: dict[str, float | None]
    quantiles: dict[float, float | None]
    # Why each null field is null, by field name: a statistic, "quantiles" for
    # every level, or one level such as "quantiles[0.5]".
    nulls: dict[str, NullReason]
    # The numerical warnings behind each field null for numerical_warning.
    warnings: dict[str, tuple[str, ...]]
    # Whether at least two values aren't missing and all of them are equal, as
    # NumericColumn.constant decides it on the stored values.
    constant: bool
    # Whether that single value is inf or -inf rather than a finite value.
    constant_infinite: bool


def summarize_values(
    column: NumericColumn,
    *,
    quantiles: list[float],
    quantile_method: QuantileMethod,
    trim_fraction: float,
) -> NumericSummary:
    """Summarize a column read as float64, in which NaN marks a missing value."""
    values = column.values  # noqa: PD011 - a NumPy array, not a pandas attribute
    missing = int(np.isnan(values).sum())
    finite_mask = np.isfinite(values)
    x = values if finite_mask.all() else values[finite_mask]
    non_missing_count = values.size - missing
    infinite = non_missing_count - x.size
    equal = x.size > 0 and bool(x.min() == x.max())
    nulls = _null_reasons(x.size, equal, quantiles)
    computed: dict[str, float] = {}
    warned: dict[str, tuple[str, ...]] = {}
    if x.size:
        computed, warned = _finite_statistics(
            x, quantiles, quantile_method, trim_fraction, skip=set(nulls)
        )
        nulls.update(dict.fromkeys(warned, "numerical_warning"))
    keys = [*STATISTICS, "quantiles", *(_quantile_key(level) for level in quantiles)]
    return NumericSummary(
        non_missing_count=non_missing_count,
        missing_count=missing,
        infinite_count=infinite,
        zero_count=int((values == 0).sum()),
        negative_count=int((values < 0).sum()),
        statistics={name: computed.get(name) for name in STATISTICS},
        quantiles={level: computed.get(_quantile_key(level)) for level in quantiles},
        nulls=dict(sorted(nulls.items(), key=lambda item: keys.index(item[0]))),
        warnings=warned,
        constant=column.constant,
        constant_infinite=column.constant_infinite,
    )


# Why a column without a finite value has status insufficient_data.
NO_FINITE_VALUES = StatusReason(
    code="no_values", message="There are no finite values to summarize."
)


def numeric_entry[Entry: SummarizeNumericColumn](
    model: type[Entry], name: str, summary: NumericSummary
) -> Entry:
    """A column's entry with summarize_numeric's fields; `model` is its class."""
    finite_count = summary.non_missing_count - summary.infinite_count
    return model(
        column=name,
        status="completed" if finite_count else "insufficient_data",
        status_reason=None if finite_count else NO_FINITE_VALUES,
        non_missing_count=summary.non_missing_count,
        missing_count=summary.missing_count,
        infinite_count=summary.infinite_count,
        zero_count=summary.zero_count,
        negative_count=summary.negative_count,
        quantiles=summary.quantiles,
        **summary.statistics,
    )


def numeric_support(name: str, summary: NumericSummary) -> ColumnSupport:
    """The support of a column's summary: missing and infinite values are excluded."""
    excluded: dict[ExclusionReason, int] = {}
    if summary.missing_count:
        excluded["missing"] = summary.missing_count
    if summary.infinite_count:
        excluded["infinite"] = summary.infinite_count
    eligible = summary.non_missing_count + summary.missing_count
    return ColumnSupport.of(name, eligible, excluded)


def null_limitations(
    summaries: Mapping[str, NumericSummary], path: Callable[[str], str]
) -> list[Limitation]:
    """One limitation per reason, naming every null field it explains.

    `path` gives the result path of a column's entry, such as "columns['x']".
    """
    found: dict[NullReason, tuple[list[str], list[str]]] = {}
    messages: list[str] = []
    for name, summary in summaries.items():
        for field, reason in summary.nulls.items():
            names, fields = found.setdefault(reason, ([], []))
            if name not in names:
                names.append(name)
            fields.append(f"{path(name)}.{field}")
        for warned in summary.warnings.values():
            messages.extend(m for m in warned if m not in messages)
    return [
        Limitation(
            code=reason,
            reason=_NULL_REASONS[reason].format(
                columns=plural(len(found[reason][0]), "column", "columns"),
                names=join_labels(found[reason][0]),
                warnings="; ".join(messages) or "none, but the results weren't finite",
            ),
            fields=found[reason][1],
        )
        for reason in _NULL_REASONS
        if reason in found
    ]


_NULL_REASONS: dict[NullReason, str] = {
    "too_few_values": (
        "Too few finite values left some statistics null in {columns}: {names}. "
        "Every statistic needs 1 finite value, std 2, skewness 3 and "
        "excess_kurtosis 4."
    ),
    "zero_variance": (
        "Every finite value is equal as a float64 number in {columns}, so skewness "
        "and excess_kurtosis are undefined and null: {names}."
    ),
    "numerical_warning": (
        "Some statistics overflowed float64 or failed numerically, so they are "
        "null in {columns}: {names}. Warnings: {warnings}."
    ),
}


def _null_reasons(
    finite_count: int, equal: bool, levels: list[float]
) -> dict[str, NullReason]:
    """Which statistics don't exist for the finite values, and why."""
    reasons: dict[str, NullReason] = {
        name: "too_few_values"
        for name in STATISTICS
        if finite_count < MINIMUM_COUNTS.get(name, 1)
    }
    if finite_count == 0 and levels:
        reasons["quantiles"] = "too_few_values"
    if equal:
        for name in SHAPE_STATISTICS:
            reasons.setdefault(name, "zero_variance")
    return reasons


def _quantile_key(level: float) -> str:
    return f"quantiles[{level!r}]"


def _finite_statistics(
    x: np.ndarray,
    levels: list[float],
    method: QuantileMethod,
    trim_fraction: float,
    *,
    skip: set[str],
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """The statistics of at least one finite value x, by field name.

    A statistic in `skip` isn't computed. A result that overflows float64 or
    fails numerically is computed again on the values divided by a power of
    two, which is exact, so only results that float64 can't hold stay null:
    they are left out of the first dict, and the second gives the numerical
    warnings behind each of them.
    """
    values, notes = _compute(x, levels, method, trim_fraction, skip=skip, scale=1.0)
    failed = [key for key, value in values.items() if not math.isfinite(value)]
    largest = max(abs(float(x.min())), abs(float(x.max())))
    scale = math.ldexp(1.0, math.frexp(largest)[1] - 1) if largest else 1.0
    if failed and scale != 1.0:
        again, again_notes = _compute(
            x, levels, method, trim_fraction, skip=skip, scale=scale
        )
        for key in failed:
            if math.isfinite(again[key]):
                values[key] = again[key]
            else:
                notes[key] = list(dict.fromkeys(notes[key] + again_notes[key]))
    finite = {key: value for key, value in values.items() if math.isfinite(value)}
    warned = {key: tuple(notes[key]) for key in values if key not in finite}
    return finite, warned


def _compute(
    x: np.ndarray,
    levels: list[float],
    method: QuantileMethod,
    trim_fraction: float,
    *,
    skip: set[str],
    scale: float,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Every statistic not in `skip`, computed on x / scale and scaled back.

    A value can be inf or NaN. The second dict gives each statistic the
    RuntimeWarnings raised while computing it.
    """
    y = x if scale == 1.0 else x / scale
    n = y.size
    values: dict[str, float] = {}
    notes: dict[str, list[str]] = {}

    # Order statistics and the trimmed mean, which cuts int(trim_fraction * n)
    # values from each end, as scipy.stats.trim_mean does. One partition puts
    # the middle values and the cut points in place and leaves the copy nearly
    # sorted, which makes numpy.quantile on it fast; the values numpy.quantile
    # returns don't depend on the order of its input.
    cut = int(trim_fraction * n)
    quantile_keys = [_quantile_key(level) for level in levels]
    with watch_warnings() as warned:
        part = np.partition(y, sorted({(n - 1) // 2, n // 2, cut, n - cut - 1}))
        median = (part[(n - 1) // 2] + part[n // 2]) / 2
        values["min"] = float(y.min() * scale)
        values["max"] = float(y.max() * scale)
        values["median"] = float(median * scale)
        values["trimmed_mean"] = float(np.mean(part[cut : n - cut]) * scale)
        quantiles = np.quantile(part, levels, method=method) * scale
        for key, quantile in zip(quantile_keys, quantiles.tolist(), strict=True):
            values[key] = quantile
    for key in ["min", "max", "median", "trimmed_mean", *quantile_keys]:
        notes[key] = warned

    # The median absolute deviation, as scipy.stats.median_abs_deviation.
    with watch_warnings() as warned:
        mad = np.median(np.abs(y - median))
        values["mad"] = float(mad * scale)
        values["mad_scaled"] = float(mad * MAD_SCALE * scale)
    notes["mad"] = notes["mad_scaled"] = warned

    # Moments about the median: subtracting a value close to the data first
    # keeps them exact when the values are large and close together, where
    # subtracting the computed mean alone loses digits to its rounding. The
    # deviations are then divided by a power of two near the largest, which is
    # exact, so their squares and fourth powers neither underflow, as those of
    # values near 1e-160 do, nor overflow. Terms that still underflow are too
    # small to change a sum that holds the largest square.
    with watch_warnings() as warned:
        shifted = y - median
        center = np.mean(shifted)
        values["mean"] = float((median + center) * scale)
        deviations = shifted - center
        spread = float(np.max(np.abs(deviations)))
        exponent = int(np.frexp(spread)[1]) if spread and math.isfinite(spread) else 0
        unit = np.ldexp(deviations, -exponent) if exponent else deviations
        squared = unit * unit
        total = np.sum(squared)
        m2 = total / n
        if "std" not in skip:
            values["std"] = float(np.ldexp(np.sqrt(total / (n - 1)), exponent) * scale)
        # The bias-corrected G1 and G2 of scipy.stats.skew and kurtosis with
        # bias=False, in SciPy's arrangement of the formulas; both are the same
        # on deviations divided by a constant.
        if "skewness" not in skip:
            m3 = np.mean(squared * unit)
            g1 = np.sqrt((n - 1.0) * n) / (n - 2.0) * m3 / m2**1.5
            values["skewness"] = float(g1)
        if "excess_kurtosis" not in skip:
            m4 = np.mean(squared * squared)
            g2 = (
                1.0 / (n - 2) / (n - 3) * ((n**2 - 1.0) * m4 / m2**2 - 3 * (n - 1) ** 2)
            )
            values["excess_kurtosis"] = float(g2)
    for key in ["mean", "std", "skewness", "excess_kurtosis"]:
        notes[key] = warned
    return values, {key: notes[key] for key in values}
