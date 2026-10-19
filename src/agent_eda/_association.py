"""The association measures of compute_association_matrix.

measure_pairs computes every pair of columns at once. The counts and sums behind
every Pearson correlation, correlation ratio and contingency table come from a
few matrix products over chunks of rows, so the work grows with the rows times
the columns and levels instead of the rows times the pairs. Each column is
divided by a power of two near its largest absolute value, which is exact, and
centred once, near its mean.

A pair is computed again on its own complete rows, in two passes, whenever its
batched sums could lose precision: when a numeric column in the pair's complete
rows lies far from the centre its whole column got, compared with its spread,
which includes a column that is constant there. Rank correlations are computed
one pair at a time, ranking the pair's own complete rows; spearman reuses a
column's ranks for every pair whose complete rows hold all the column's values.

Numeric columns are float64 with NaN where a value is missing; a pair's complete
rows hold finite values only. Categorical columns are integer codes from 0 with
-1 where a value is missing.
"""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda.statistics._association_results import UndefinedReason

# The measures of a pair of numeric columns.
NumericMethod = Literal["pearson", "spearman", "kendall"]

# Every measure needs at least this many complete rows.
MINIMUM_ROWS = 3
# The batched sums of a pair count only when, in the pair's complete rows, the
# sum of squares about the pair's own mean keeps at least this share of the sum
# of squares about the column's centre. Then the one-pass formulas' rounding
# error stays within a small factor of what two passes would make.
WELL_CONDITIONED = 0.25
# Below this mean square, squares lose precision to subnormal numbers.
TINY_MEAN_SQUARE = 2.0**-900
# Categorical columns with more levels are paired one pair at a time.
BATCH_LEVELS = 64
# One product gives the contingency tables among the categorical columns with up
# to this many levels; a pair with a column of more levels is counted on its
# own. On a million rows counting one pair takes about 1.6 ms, and a product
# costs about 1 ps per pair of levels per row with Accelerate and 9 ps with
# OpenBLAS, so the product wins up to about 13 x 13 levels even on OpenBLAS.
TABLE_LEVELS = 12
# A contingency table with more cells is built from the cells that occur only.
MAX_TABLE_CELLS = 2**22
# Relative rounding error below which the bias-corrected V's numerator is 0.
ROUNDING = 64 * np.finfo(np.float64).eps
# Rows per chunk: as many as fit in about 32 MiB of buffers, at most 65,536,
# which keeps counts in float32 exact.
CHUNK_BYTES = 2**25
MAX_CHUNK_ROWS = 2**16


@dataclass(frozen=True)
class Measure:
    """A pair's value, the complete rows behind it and why it may not exist."""

    value: float | None
    pair_count: int
    undefined: UndefinedReason | None = None


@dataclass(frozen=True)
class Codes:
    """A categorical column: codes from 0, -1 where missing, and the level count."""

    codes: np.ndarray
    levels: int

    @functools.cached_property
    def shifted(self) -> np.ndarray:
        """The codes plus 1, so that 0 stands for a missing value."""
        return self.codes + 1


@dataclass(frozen=True)
class Measures:
    """Every pair's measure, keyed by the columns' indices in their own lists."""

    # (i, j) for numeric columns i < j.
    numeric: dict[tuple[int, int], Measure]
    # (a, b) for categorical columns a < b.
    categorical: dict[tuple[int, int], Measure]
    # (a, j) for categorical column a and numeric column j.
    mixed: dict[tuple[int, int], Measure]


def measure_pairs(
    numeric: Sequence[np.ndarray],
    categorical: Sequence[Codes],
    *,
    numeric_method: NumericMethod,
    bias_corrected: bool,
    ratio_bias_corrected: bool,
) -> Measures:
    """Every pair's measure: numeric pairs by numeric_method, categorical pairs by
    Cramér's V and mixed ones by the correlation ratio, each plain or
    bias-corrected."""
    # The batched categorical columns, those with tables first.
    batch = sorted(
        (a for a, c in enumerate(categorical) if c.levels <= BATCH_LEVELS),
        key=lambda a: categorical[a].levels > TABLE_LEVELS,
    )
    tabled = [a for a in batch if categorical[a].levels <= TABLE_LEVELS]
    if len(tabled) < 2:
        tabled = []
    pearson_batch = numeric_method == "pearson" and len(numeric) >= 2
    sums = None
    if pearson_batch or (batch and numeric) or tabled:
        sums = _batch_sums(
            numeric,
            [categorical[a] for a in batch],
            table_columns=len(tabled),
            products=pearson_batch,
        )
    numeric_measures = _numeric_measures(
        numeric, sums if pearson_batch else None, numeric_method
    )
    position = {a: i for i, a in enumerate(batch)}
    in_tables = set(tabled)
    rows = len(categorical[0].codes) if categorical else 0
    # Only pairs of categorical columns use it.
    buffer = np.empty(rows if len(categorical) >= 2 else 0, dtype=np.int64)
    categorical_measures: dict[tuple[int, int], Measure] = {}
    for a, b in combinations(range(len(categorical)), 2):
        if sums is not None and a in in_tables and b in in_tables:
            table = sums.table(position[a], position[b])
            categorical_measures[a, b] = table_measure(table, bias_corrected)
        else:
            categorical_measures[a, b] = categorical_pair(
                categorical[a],
                categorical[b],
                bias_corrected=bias_corrected,
                buffer=buffer,
            )
    mixed_measures: dict[tuple[int, int], Measure] = {}
    for a, codes in enumerate(categorical):
        batched: list[Measure | None] = [None] * len(numeric)
        if sums is not None and a in position:
            batched = sums.correlation_ratios(position[a], ratio_bias_corrected)
        for j, values in enumerate(numeric):
            measure = batched[j]
            mixed_measures[a, j] = (
                mixed_pair(codes.codes, values, ratio_bias_corrected)
                if measure is None
                else measure
            )
    return Measures(numeric_measures, categorical_measures, mixed_measures)


# Given a pair's complete rows and their count, both columns' ranks within them.
type RankPair = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def numeric_pair(
    x: np.ndarray, y: np.ndarray, method: NumericMethod, *, rank_pair: RankPair
) -> Measure:
    """Pearson's r, Spearman's rho or Kendall's tau-b on the complete rows.

    For spearman, rank_pair supplies the ranks of the pair's complete rows.
    """
    complete = np.isfinite(x) & np.isfinite(y)
    count = int(complete.sum())
    if count < MINIMUM_ROWS:
        return Measure(None, count, "too_few_rows")
    a, b = x[complete], y[complete]
    if _constant(a) or _constant(b):
        return Measure(None, count, "constant_column")
    if method == "pearson":
        value = pearson(a, b)
    elif method == "spearman":
        value = pearson(*rank_pair(complete, count))
    elif method == "kendall":
        value = kendall_tau_b(a, b)
    else:
        raise ValueError(
            f"agent_eda bug: unknown numeric method {method!r}; the choices are "
            "'pearson', 'spearman' and 'kendall'"
        )
    return Measure(_clip(value, -1.0), count)


def categorical_pair(
    x: Codes, y: Codes, *, bias_corrected: bool, buffer: np.ndarray
) -> Measure:
    """Cramér's V, plain or bias-corrected, on the complete rows.

    buffer, an int64 array as long as the columns, holds the joint codes.
    """
    cells = (x.levels + 1) * (y.levels + 1)
    if cells > MAX_TABLE_CELLS:
        return _sparse_table_measure(x, y, bias_corrected)
    # Code 0 of the shifted codes stands for a missing value, so they need no mask.
    joint = np.multiply(x.shifted, y.levels + 1, out=buffer)
    joint += y.shifted
    counts = np.bincount(joint, minlength=cells)
    table = counts.reshape(x.levels + 1, y.levels + 1)[1:, 1:]
    return table_measure(table, bias_corrected)


def table_measure(table: np.ndarray, bias_corrected: bool) -> Measure:
    """Cramér's V of a contingency table of the complete rows, empty levels kept."""
    count = int(table.sum())
    if count < MINIMUM_ROWS:
        return Measure(None, count, "too_few_rows")
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    undefined = _undefined_levels(min(table.shape), max(table.shape), count)
    if undefined is not None:
        return Measure(None, count, undefined)
    return Measure(_clip(cramers_v(table, bias_corrected=bias_corrected)), count)


def _undefined_levels(fewest: int, most: int, count: int) -> UndefinedReason | None:
    """Why a measure of categorical columns doesn't exist in the complete rows.

    fewest and most are the smaller and larger level counts of the columns. A
    column with a level for every complete row explains any other column
    perfectly and trivially, and the bias-corrected V divides by zero there.
    """
    if fewest < 2:
        return "single_level"
    if most == count:
        return "one_row_per_level"
    return None


def _sparse_table_measure(x: Codes, y: Codes, bias_corrected: bool) -> Measure:
    """Cramér's V from the cells that occur, for tables too large to hold whole.

    Every cell that doesn't occur adds its expected count to chi2, and those
    add up to the rows less the expected counts of the cells that do.
    """
    complete = (x.codes >= 0) & (y.codes >= 0)
    count = int(complete.sum())
    if count < MINIMUM_ROWS:
        return Measure(None, count, "too_few_rows")
    rows, columns = x.codes[complete], y.codes[complete]
    found, cells = pd.factorize(rows * y.levels + columns)
    observed = np.bincount(found).astype(np.float64)
    cell_rows, cell_columns = np.divmod(np.asarray(cells), y.levels)
    row_totals = np.bincount(rows, minlength=x.levels).astype(np.float64)
    column_totals = np.bincount(columns, minlength=y.levels).astype(np.float64)
    r, k = int((row_totals > 0).sum()), int((column_totals > 0).sum())
    undefined = _undefined_levels(min(r, k), max(r, k), count)
    if undefined is not None:
        return Measure(None, count, undefined)
    expected = row_totals[cell_rows] * column_totals[cell_columns] / count
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    chi2 += max(0.0, count - float(expected.sum()))
    return Measure(_clip(_cramers_v(chi2 / count, r, k, count, bias_corrected)), count)


def mixed_pair(codes: np.ndarray, y: np.ndarray, bias_corrected: bool) -> Measure:
    """The correlation ratio from the categorical codes to the numeric y, plain or
    bias-corrected."""
    complete = (codes >= 0) & np.isfinite(y)
    count = int(complete.sum())
    if count < MINIMUM_ROWS:
        return Measure(None, count, "too_few_rows")
    levels = _compact(codes[complete])
    present = int(levels.max()) + 1
    undefined = _undefined_levels(present, present, count)
    if undefined is not None:
        return Measure(None, count, undefined)
    values = y[complete]
    if _constant(values):
        return Measure(None, count, "constant_column")
    ratio = correlation_ratio(levels, values, bias_corrected=bias_corrected)
    return Measure(_clip(ratio), count)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson's r of two non-constant arrays, in two passes.

    Each array is first divided by a power of two near its largest absolute
    value, which is exact and keeps squares and sums far from overflow.
    """
    da = _centred(_scaled(a))
    db = _centred(_scaled(b))
    return float(np.sum(da * db) / np.sqrt(np.sum(da * da) * np.sum(db * db)))


def _centred(values: np.ndarray) -> np.ndarray:
    """The values less their mean, which is then subtracted again.

    For values large and close together, such as 1e15 plus small integers, the
    differences are exact but the mean is rounded by up to half an ulp of the
    values, a visible share of their spread; the second pass removes that.
    """
    deviations = values - np.mean(values)
    return deviations - np.mean(deviations)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, tied values sharing the mean of their ranks."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], ordered.size]
    ranks = np.empty(ordered.size)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-b, which adjusts for ties, as scipy.stats.kendalltau."""
    from scipy import stats

    return float(stats.kendalltau(a, b, variant="b").statistic)


def cramers_v(table: np.ndarray, *, bias_corrected: bool) -> float:
    """Cramér's V of a table without empty rows or columns, at least 2 x 2.

    chi2 is Pearson's statistic without continuity correction, as in
    scipy.stats.contingency.association. The bias correction follows Bergsma
    (2013) and needs fewer levels than rows in both columns.
    """
    counts = table.astype(np.float64)
    n = int(table.sum())
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
    phi2 = float(np.sum((counts - expected) ** 2 / expected) / n)
    return _cramers_v(phi2, *table.shape, n, bias_corrected)


def _cramers_v(phi2: float, r: int, k: int, n: int, bias_corrected: bool) -> float:
    """V from phi^2 = chi2 / n of an r x k table of n rows."""
    if not bias_corrected:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    corrected = phi2 - (k - 1) * (r - 1) / (n - 1)
    # A difference within rounding error of 0 is 0; its square root would
    # otherwise show the rounding at about 1e-8.
    if corrected <= ROUNDING * phi2:
        corrected = 0.0
    # r - (r - 1)^2 / (n - 1) - 1, written without the cancellation.
    r_corrected = (r - 1) * (n - r) / (n - 1)
    k_corrected = (k - 1) * (n - k) / (n - 1)
    return float(np.sqrt(corrected / min(r_corrected, k_corrected)))


def correlation_ratio(
    levels: np.ndarray, values: np.ndarray, *, bias_corrected: bool
) -> float:
    """The correlation ratio eta, or its bias-corrected form, of numeric values
    by level.

    levels are codes from 0, each present; values aren't constant.
    """
    deviations = _scaled(values)
    deviations = deviations - np.mean(deviations)
    counts = np.bincount(levels).astype(np.float64)
    sums = np.bincount(levels, weights=deviations)
    centre = sums.sum() / counts.sum()
    between = np.sum(counts * (sums / counts - centre) ** 2)
    total = np.sum((deviations - centre) ** 2)
    return ratio_value(
        between, total, levels=len(counts), count=len(values), corrected=bias_corrected
    )


def ratio_value(
    between: float, total: float, *, levels: int, count: int, corrected: bool
) -> float:
    """eta = sqrt(between / total), or sqrt(epsilon squared) when corrected.

    between and total are the between-level and total sums of squares; levels
    counts the levels present, at least 2 and fewer than the count rows.
    Epsilon squared, Kelley's (1935) bias-adjusted share of the variance that
    the levels explain, takes the within-level variance that chance alone
    would leave between the levels off the between-level sum of squares:
    (between - (levels - 1) * within / (count - levels)) / total, where within
    is total less between. Below 0, which it is for about half the pairs of
    unrelated columns, it is clipped at 0.
    """
    if not corrected:
        return float(np.sqrt(between / total))
    within = total - between
    epsilon_squared = (between - (levels - 1) * within / (count - levels)) / total
    return float(np.sqrt(max(epsilon_squared, 0.0)))


def _numeric_measures(
    numeric: Sequence[np.ndarray], sums: "_Sums | None", method: NumericMethod
) -> dict[tuple[int, int], Measure]:
    """Every numeric pair: from the batched sums where they are precise enough."""
    correlations = sums.correlations() if sums is not None else None
    column_ranks = _ColumnRanks(numeric)
    measures = {}
    for i, j in combinations(range(len(numeric)), 2):
        found = None if correlations is None else correlations(i, j)
        if found is None:
            rank_pair = functools.partial(column_ranks.pair, i, j)
            found = numeric_pair(numeric[i], numeric[j], method, rank_pair=rank_pair)
        measures[i, j] = found
    return measures


class _ColumnRanks:
    """Each numeric column's ranks over all its finite values, made once.

    A pair whose complete rows hold every finite value of a column ranks that
    column as the whole column does, so the column is ranked only once for all
    such pairs, as in data without missing values. Other pairs rank their own
    complete rows.
    """

    def __init__(self, numeric: Sequence[np.ndarray]) -> None:
        self._numeric = numeric
        self._finite_counts: dict[int, int] = {}
        self._ranks: dict[int, np.ndarray] = {}

    def pair(
        self, i: int, j: int, complete: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Columns i's and j's ranks within the complete rows, in row order."""
        return self._within(i, complete, count), self._within(j, complete, count)

    def _within(self, i: int, complete: np.ndarray, count: int) -> np.ndarray:
        values = self._numeric[i]
        if i not in self._finite_counts:
            self._finite_counts[i] = int(np.isfinite(values).sum())
        if self._finite_counts[i] != count:
            return average_ranks(values[complete])
        if i not in self._ranks:
            ranks = np.full(values.size, np.nan)
            finite = np.isfinite(values)
            ranks[finite] = average_ranks(values[finite])
            self._ranks[i] = ranks
        return self._ranks[i][complete]


@dataclass(frozen=True)
class _Sums:
    """Counts and sums over chunks of rows, from which pairs are read off.

    z is a numeric column scaled and centred, 0 where the value isn't finite; m
    is 1 where it is finite. A categorical level's indicator is 1 in the rows
    holding it.
    """

    # Numeric column i and j: rows where both are finite, the sum of z_i, of
    # z_i^2 and of z_i z_j over those rows; empty without products.
    counts: np.ndarray
    first: np.ndarray
    first_squares: np.ndarray
    products: np.ndarray
    # Sum of z_j^2 over the rows where numeric column j is finite.
    squares: np.ndarray
    # Level by level over the categorical columns with tables, which come
    # first: rows holding both levels.
    level_pairs: np.ndarray
    # Level l and numeric column j: rows holding l where j is finite, and the
    # sum of z_j over them.
    level_counts: np.ndarray
    level_sums: np.ndarray
    # Categorical column a and numeric column j: sum of z_j^2 over the rows
    # where a is missing and j is finite.
    missing_squares: np.ndarray
    # Where each batched categorical column's levels start, and their counts.
    offsets: np.ndarray
    levels: np.ndarray

    def table(self, a: int, b: int) -> np.ndarray:
        """The contingency table of batched categorical columns a and b."""
        rows = slice(self.offsets[a], self.offsets[a] + self.levels[a])
        columns = slice(self.offsets[b], self.offsets[b] + self.levels[b])
        return np.rint(self.level_pairs[rows, columns]).astype(np.int64)

    def correlations(self):
        """A function giving the Measure of numeric pair (i, j), or None when the
        batched sums aren't precise enough for it."""
        n = self.counts
        with np.errstate(all="ignore"):  # pairs without rows are never read
            means = self.first / n
            spread = self.first_squares - self.first * means
            covariance = self.products - self.first * self.first.T / n
            r = covariance / np.sqrt(spread * spread.T)
            conditioned = (self.first_squares > n * TINY_MEAN_SQUARE) & (
                spread >= WELL_CONDITIONED * self.first_squares
            )
        conditioned = conditioned & conditioned.T

        def measure(i: int, j: int) -> Measure | None:
            count = round(float(n[i, j]))
            if count < MINIMUM_ROWS:
                return Measure(None, count, "too_few_rows")
            if not conditioned[i, j]:
                return None
            return Measure(_clip(float(r[i, j]), -1.0), count)

        return measure

    def correlation_ratios(self, a: int, bias_corrected: bool) -> list[Measure | None]:
        """Batched categorical column a with every numeric column: its Measure,
        or None where the batched sums aren't precise enough."""
        rows = slice(self.offsets[a], self.offsets[a] + self.levels[a])
        counts = self.level_counts[rows]
        sums = self.level_sums[rows]
        total = counts.sum(axis=0)
        total_sum = sums.sum(axis=0)
        squares = self.squares - self.missing_squares[a]
        present = counts > 0
        with np.errstate(all="ignore"):  # pairs without rows are never read
            mean = total_sum / total
            spread = squares - total_sum * mean
            level_means = np.where(present, sums / np.where(present, counts, 1), 0.0)
            between = np.sum(counts * (level_means - mean) ** 2, axis=0)
            conditioned = (
                (self.missing_squares[a] <= (1 - WELL_CONDITIONED) * self.squares)
                & (squares > total * TINY_MEAN_SQUARE)
                & (spread >= WELL_CONDITIONED * squares)
            )
        measures: list[Measure | None] = []
        for j in range(counts.shape[1]):
            count = round(float(total[j]))
            levels = int(present[:, j].sum())
            undefined = _undefined_levels(levels, levels, count)
            if count < MINIMUM_ROWS:
                measures.append(Measure(None, count, "too_few_rows"))
            elif undefined is not None:
                measures.append(Measure(None, count, undefined))
            elif not conditioned[j]:
                measures.append(None)
            else:
                ratio = ratio_value(
                    float(between[j]),
                    float(spread[j]),
                    levels=levels,
                    count=count,
                    corrected=bias_corrected,
                )
                measures.append(Measure(_clip(ratio), count))
        return measures


def _batch_sums(
    numeric: Sequence[np.ndarray],
    categorical: Sequence[Codes],
    *,
    table_columns: int,
    products: bool,
) -> _Sums:
    """Accumulate every count and sum the pairs need, chunk of rows by chunk.

    The first table_columns categorical columns get their contingency tables;
    products asks for the numeric pairs' sums.
    """
    rows = len(numeric[0]) if numeric else len(categorical[0].codes)
    p, q = len(numeric), len(categorical)
    levels = np.array([c.levels for c in categorical], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(levels)[:-1]]).astype(np.int64)
    level_total = int(levels.sum())
    table_levels = int(levels[:table_columns].sum())
    trash = level_total  # the one-hot column missing values go to
    width = level_total + 1 + p
    exponents, centres = _exponents_and_centres(numeric)
    row_bytes = 4 * width + 8 * (level_total + 4 * p + q)
    chunk = max(1, min(MAX_CHUNK_ROWS, rows, CHUNK_BYTES // row_bytes))

    one_hot = np.zeros((chunk, width), dtype=np.float32)
    flat = one_hot.reshape(-1)
    base = np.arange(chunk, dtype=np.int64) * width
    z_and_squares = np.empty((chunk, 2 * p), order="F")
    z, squared = z_and_squares[:, :p], z_and_squares[:, p:]
    finite = np.empty((chunk, p), order="F")
    missing = np.empty((chunk, q), order="F")

    level_pairs = np.zeros((table_levels, table_levels))
    with_finite = np.zeros((width, p))
    level_sums = np.zeros((level_total, p))
    numeric_sums = np.zeros((2 * p, p))
    numeric_products = np.zeros((p, p))
    missing_squares = np.zeros((q, p))
    squares = np.zeros(p)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        m = stop - start
        for j, values in enumerate(numeric):
            block = values[start:stop]
            ok = np.isfinite(block)
            finite[:m, j] = ok
            column = z[:m, j]
            if -1022 <= -exponents[j] <= 1023:
                # Multiplying by a normal power of two is exact, and faster.
                np.multiply(block, math.ldexp(1.0, -exponents[j]), out=column)
            else:
                np.ldexp(block, -exponents[j], out=column)
            column -= centres[j]
            np.copyto(column, 0.0, where=~ok)
        np.multiply(z[:m], z[:m], out=squared[:m])
        one_hot[:m, : trash + 1] = 0.0
        one_hot[:m, trash + 1 :] = finite[:m]
        for a, codes in enumerate(categorical):
            block = codes.codes[start:stop]
            missing[:m, a] = block < 0
            flat[base[:m] + np.where(block >= 0, block + offsets[a], trash)] = 1.0
        hot = one_hot[:m]
        if table_levels:
            tabled = hot[:, :table_levels]
            level_pairs += tabled.T @ tabled
        with_finite += hot.T @ hot[:, trash + 1 :]
        level_sums += hot[:, :level_total].astype(np.float64).T @ z[:m]
        missing_squares += missing[:m].T @ squared[:m]
        squares += squared[:m].sum(axis=0)
        if products:
            numeric_sums += z_and_squares[:m].T @ finite[:m]
            numeric_products += z[:m].T @ z[:m]
    return _Sums(
        counts=with_finite[trash + 1 :],
        first=numeric_sums[:p],
        first_squares=numeric_sums[p:],
        products=numeric_products,
        squares=squares,
        level_pairs=level_pairs,
        level_counts=with_finite[:level_total],
        level_sums=level_sums,
        missing_squares=missing_squares,
        offsets=offsets,
        levels=levels,
    )


def _exponents_and_centres(
    numeric: Sequence[np.ndarray],
) -> tuple[list[int], list[float]]:
    """For each numeric column, the power of two that brings its largest absolute
    value into [0.5, 1), and a centre near its mean after scaling.

    Scaling goes through np.ldexp with the negated exponent, as 2^-e itself can
    overflow for subnormal values. The centre is the mean of the finite values
    among evenly spaced rows: any value near the mean keeps the sums precise.
    """
    exponents, centres = [], []
    for values in numeric:
        high = float(np.fmax.reduce(values))
        low = float(np.fmin.reduce(values))
        if not (np.isfinite(high) and np.isfinite(low)):
            finite = values[np.isfinite(values)]
            high, low = float(finite.max()), float(finite.min())
        largest = max(abs(high), abs(low))
        exponent = int(np.frexp(largest)[1]) if largest else 0
        sample = values[:: max(1, len(values) // 65_536)]
        sample = sample[np.isfinite(sample)]
        if not sample.size:
            sample = values[np.isfinite(values)]
        exponents.append(exponent)
        centres.append(float(np.mean(np.ldexp(sample, -exponent))))
    return exponents, centres


def _compact(codes: np.ndarray) -> np.ndarray:
    """Codes from 0 renumbered in order, keeping only the levels present."""
    present = np.bincount(codes) > 0
    return (np.cumsum(present) - 1)[codes]


def _constant(values: np.ndarray) -> bool:
    return bool(values.min() == values.max())


def _scaled(values: np.ndarray) -> np.ndarray:
    """The values divided by a power of two near their largest absolute value."""
    largest = max(abs(float(values.min())), abs(float(values.max())))
    if largest == 0:
        return values
    return np.ldexp(values, -np.frexp(largest)[1])


def _clip(value: float, low: float = 0.0) -> float:
    """The value within its range, which rounding can overshoot by a few ulps."""
    return min(max(value, low), 1.0)
