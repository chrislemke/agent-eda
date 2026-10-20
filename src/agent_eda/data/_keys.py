"""The search for column sets that identify rows: candidate keys.

A column set is a key when no two rows share its combination of values, with
missing values of every kind (None, NaN, NaT, NA) compared equal. A
superset of a key is a key too, but not a minimal one, so the search examines
single columns first, then sets of increasing size built only from smaller
sets that aren't keys.

Checking thousands of column sets one by one on a million rows takes minutes,
so each set is first checked on evenly spaced subsets of the rows. A repeated
combination there proves the set isn't a key. A set that shows none is checked
on all rows, the only way the search confirms a key.

When no key turns up, the search ranks the sets it examined by their distinct
combinations, counted on all rows. Counting every set would take minutes too,
so sets are counted highest upper bound first: once no set left can beat the
10th best count, the ranking is proven. Counting stops at a work limit, and the
ranking then says it isn't proven.
"""

import heapq
import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agent_eda._values import value_codes

# Rows in the evenly spaced subset; smaller data is checked whole.
SUBSET_ROWS = 16_384
# A set is checked on every 32nd row of the subset, then every 4th, then all of
# them: most sets repeat a combination within the first few hundred rows.
STAGE_STEPS = (32, 4, 1)
# In data with more rows than this, a set that shows no repeat on the subset is
# also checked on every QUARTER_STEP-th row before all rows. That subset sees a
# sixteenth of the pairs of rows that repeat a combination, at a quarter of the
# cost of all rows.
QUARTER_STEP = 4
QUARTER_FROM_ROWS = QUARTER_STEP * SUBSET_ROWS
# How many of the most unique column sets a search without keys lists.
MOST_UNIQUE_SETS = 10
# The rows the ranking may hash: RANKING_PASSES times the rows of the data, or
# RANKING_ROWS when that is more.
RANKING_PASSES = 100
RANKING_ROWS = 100_000_000
# Combined codes up to this many times the rows are counted with np.bincount,
# which is faster than hashing them.
_BINCOUNT_FACTOR = 4
# What pandas' infer_dtype calls object columns whose every value is a scalar,
# which can be hashed. infer_dtype reads every value to say so. Any other
# answer, such as "mixed", "mixed-integer" for integers next to lists, or
# "unknown-array", is settled by hashing the values.
_SCALARS = frozenset(
    {
        "string",
        "bytes",
        "floating",
        "integer",
        "mixed-integer-float",
        "decimal",
        "complex",
        "boolean",
        "datetime64",
        "datetime",
        "date",
        "timedelta64",
        "timedelta",
        "time",
        "period",
        "interval",
        "empty",
    }
)


def subset_positions(row_count: int) -> np.ndarray:
    """The rows of the subset: row i * n // 16,384 for i below 16,384, or all rows."""
    if row_count <= SUBSET_ROWS:
        return np.arange(row_count, dtype=np.int64)
    return np.arange(SUBSET_ROWS, dtype=np.int64) * row_count // SUBSET_ROWS


def hashable(values: pd.Series) -> bool:
    """Whether every value can be hashed, which grouping rows by value needs.

    Object columns that infer_dtype finds hold only scalars are hashable;
    any other, such as integers next to lists, is hashed whole. Nested Arrow
    types, such as lists and structs, can't be hashed.
    """
    dtype = values.dtype
    if isinstance(dtype, pd.ArrowDtype):
        return getattr(dtype.pyarrow_dtype, "num_fields", 0) == 0
    if isinstance(dtype, np.dtype) and dtype.kind == "O":
        if pd.api.types.infer_dtype(values, skipna=True) in _SCALARS:
            return True
    elif isinstance(dtype, np.dtype | pd.CategoricalDtype | pd.StringDtype):
        return True
    return value_codes(values, unhashable="unsupported") is not None


def codes_of(values: pd.Series) -> tuple[np.ndarray, int]:
    """Integer codes, 0 for a missing value and 1 and up for each distinct value.

    Every kind of missing value (None, NaN, NaT, pd.NA) gets code 0, so missing
    values compare equal. Returns the codes and their range: one more than the
    number of distinct non-missing values.
    """
    found = value_codes(values, unhashable="by_content")
    return found.codes.astype(np.int64) + 1, len(found.uniques) + 1


@dataclass(frozen=True)
class SetCounts:
    """How a column set identifies rows, counted on all rows."""

    positions: tuple[int, ...]
    distinct_row_count: int
    duplicate_group_row_count: int
    missing_row_count: int


class _Column:
    """A searched column's codes on all rows, and the groups of rows they form."""

    def __init__(self, values: pd.Series) -> None:
        codes, self.size = codes_of(values)
        # Rows per code; code 0 holds the missing values.
        group_sizes = np.bincount(codes, minlength=self.size)
        row_count = len(codes)
        self.missing_count = int(group_sizes[0])
        self.distinct_row_count = int(np.count_nonzero(group_sizes))
        singles = int(np.count_nonzero(group_sizes == 1))
        self.duplicate_group_row_count = row_count - singles
        # The chance that two rows drawn at random share a value.
        squares = float(np.dot(group_sizes, group_sizes.astype(np.float64)))
        self.collision = squares / row_count**2 if row_count else 0.0
        self.codes = codes.astype(_smallest_unsigned(self.size))
        # Rows whose value another row shares; only they can share a combination.
        self.shared = group_sizes[codes] > 1
        self.missing = codes == 0 if self.missing_count else None
        # The sizes of the groups of non-missing values, smallest first, with how
        # many groups have each size.
        histogram = np.bincount(group_sizes[1:])
        self._sizes = np.flatnonzero(histogram)
        counts = histogram[self._sizes]
        # Groups and rows in groups of the i-th size and every larger one.
        self._groups_from = np.cumsum(counts[::-1])[::-1]
        self._rows_from = np.cumsum((counts * self._sizes)[::-1])[::-1]

    def repeats_at_least(self, capacity: int, missing_everywhere: int) -> int:
        """The fewest rows that must repeat a combination of this and other columns.

        Rows sharing this column's value can take at most `capacity` combinations
        of the other columns, so a group of g rows repeats at least g - capacity
        of them. `missing_everywhere` rows are missing in this and every other
        column, so they share one combination.
        """
        first = int(np.searchsorted(self._sizes, capacity, side="right"))
        above = 0
        if first < len(self._sizes):
            groups = int(self._groups_from[first])
            above = int(self._rows_from[first]) - capacity * groups
        missing = max(0, self.missing_count - capacity, missing_everywhere - 1)
        return above + missing


def _smallest_unsigned(size: int) -> type[np.unsignedinteger]:
    for dtype in (np.uint8, np.uint16, np.uint32):
        if size <= np.iinfo(dtype).max + 1:
            return dtype
    return np.uint64


@dataclass
class KeySearchTally:
    """What the search examined and found."""

    keys: list[SetCounts] = field(default_factory=list)
    examined_count: int = 0
    rejected_on_subset_count: int = 0
    checked_on_all_rows_count: int = 0
    # Whether max_combinations stopped the search before every set was examined.
    stopped: bool = False
    # Every examined set that isn't a key, with the rows that repeat a
    # combination on the rows it was checked on: a lower bound on the rows that
    # repeat one on all rows, and the exact count for a set checked on all rows.
    non_keys: list[tuple[tuple[int, ...], int]] = field(default_factory=list)


@dataclass(frozen=True)
class Ranking:
    """The most unique column sets the search examined, counted on all rows."""

    sets: list[SetCounts]
    # Sets of two or more columns among those examined whose distinct
    # combinations were counted on all rows.
    counted_count: int
    # Whether bounds prove that no other examined set has more combinations.
    proven: bool


# The steps that tighten a set's bound in the ranking, cheapest first; after
# them, the set is counted on all rows.
_MISSING_EVERYWHERE, _QUARTER = 1, 2


class Searcher:
    """Searches the given columns for minimal keys of up to max_key_size columns."""

    def __init__(self, columns: Sequence[pd.Series]) -> None:
        self.columns = list(columns)
        self.row_count = len(self.columns[0]) if self.columns else 0
        self.positions = subset_positions(self.row_count)
        subset = [codes_of(column.iloc[self.positions]) for column in self.columns]
        # One row of subset codes per column, and each column's code range there.
        self.subset_codes = np.vstack(
            [codes for codes, _ in subset] or [np.empty(0, np.int64)]
        )
        self.subset_sizes = np.array([size for _, size in subset], dtype=np.int64)
        self._full: dict[int, _Column] = {}
        # Rows repeating a combination of a set among every step-th row, by set
        # and step.
        self._repeats: dict[tuple[tuple[int, ...], int], int] = {}
        # Rows hashed to count repeats, which the ranking's work limit counts.
        self.rows_hashed = 0

    def combination_count(self, max_key_size: int) -> int:
        """Column sets of 1 to max_key_size columns: the most the search examines."""
        largest = min(max_key_size, len(self.columns))
        return sum(math.comb(len(self.columns), size) for size in range(1, largest + 1))

    def largest_key_size_within(self, max_combinations: int, max_key_size: int) -> int:
        """The most columns a key may combine for the sets to fit max_combinations.

        At most max_key_size; 0 when even the single columns don't fit.
        """
        count, fitting = 0, 0
        for size in range(1, min(max_key_size, len(self.columns)) + 1):
            count += math.comb(len(self.columns), size)
            if count > max_combinations:
                break
            fitting = size
        return fitting

    def search(self, *, max_key_size: int, max_combinations: int) -> KeySearchTally:
        """Examine sets in increasing size until none is left or the limit is hit."""
        tally = KeySearchTally()
        previous: list[tuple[int, ...]] = []
        for size in range(1, min(max_key_size, len(self.columns)) + 1):
            non_keys: list[tuple[int, ...]] = []
            for prefix, all_partners in self._batches(size, previous):
                room = max_combinations - tally.examined_count
                partners = all_partners[:room]
                if len(partners) < len(all_partners):
                    tally.stopped = True
                if not partners:
                    break
                tally.examined_count += len(partners)
                subset_repeats = self._subset_repeats(prefix, partners)
                for partner, repeats in zip(partners, subset_repeats, strict=True):
                    candidate = (*prefix, partner)
                    if not repeats and self._uses_quarter(candidate):
                        repeats = self.repeats(candidate, step=QUARTER_STEP)
                    if repeats:
                        tally.rejected_on_subset_count += 1
                    else:
                        tally.checked_on_all_rows_count += 1
                        repeats = self.repeats(candidate)
                        if not repeats:
                            tally.keys.append(self.count(candidate))
                            continue
                    non_keys.append(candidate)
                    tally.non_keys.append((candidate, int(repeats)))
                if tally.stopped:
                    break
            if tally.stopped or not non_keys:
                break
            previous = non_keys
        return tally

    def _uses_quarter(self, positions: tuple[int, ...]) -> bool:
        """Whether a set is checked on every 4th row before all rows.

        A single column is counted from its own codes, which cost the same.
        """
        return len(positions) > 1 and self.row_count > QUARTER_FROM_ROWS

    def _batches(
        self, size: int, previous: list[tuple[int, ...]]
    ) -> Iterator[tuple[tuple[int, ...], list[int]]]:
        """The sets of this size to examine, as a prefix and the columns to add.

        A set is examined only if every subset one column smaller isn't a key,
        so supersets of keys are never generated. Sets come in lexicographic
        order of their column positions.
        """
        if size == 1:
            yield (), list(range(len(self.columns)))
            return
        known = set(previous)
        for _, group in itertools.groupby(previous, key=lambda s: s[:-1]):
            members = list(group)
            lasts = [member[-1] for member in members]
            for i, prefix in enumerate(members):
                partners = [
                    last
                    for last in lasts[i + 1 :]
                    if all(
                        (*prefix[:j], *prefix[j + 1 :], last) in known
                        for j in range(len(prefix) - 1)
                    )
                ]
                if partners:
                    yield prefix, partners

    def _subset_repeats(
        self, prefix: tuple[int, ...], partners: list[int]
    ) -> np.ndarray:
        """Rows repeating a combination of the prefix and each partner on the subset.

        Each set is checked on ever larger evenly spaced parts of the subset and
        stops at the first that repeats a combination; 0 means none repeated.
        """
        base = self._subset_prefix_codes(prefix)
        repeats = np.zeros(len(partners), dtype=np.int64)
        alive = np.arange(len(partners))
        columns = np.asarray(partners, dtype=np.int64)
        for step in STAGE_STEPS:
            rows = base[::step]
            chosen = columns[alive]
            combined = (
                rows[None, :] * self.subset_sizes[chosen, None]
                + self.subset_codes[chosen, ::step]
            )
            ordered = np.sort(combined, axis=1)
            found = np.count_nonzero(ordered[:, 1:] == ordered[:, :-1], axis=1)
            repeats[alive] = found
            alive = alive[found == 0]
            if alive.size == 0:
                break
        return repeats

    def _subset_prefix_codes(self, prefix: tuple[int, ...]) -> np.ndarray:
        """The prefix's combination on each subset row, as small integer codes."""
        codes = np.zeros(len(self.positions), dtype=np.int64)
        for position in prefix:
            codes = codes * self.subset_sizes[position] + self.subset_codes[position]
            codes = pd.factorize(codes)[0].astype(np.int64)
        return codes

    def column(self, position: int) -> _Column:
        if position not in self._full:
            self._full[position] = _Column(self.columns[position])
        return self._full[position]

    def count(self, positions: tuple[int, ...]) -> SetCounts:
        """Distinct combinations, rows sharing one and rows with a missing value."""
        missing = np.zeros(self.row_count, dtype=bool)
        for position in positions:
            column_missing = self.column(position).missing
            if column_missing is not None:
                missing |= column_missing
        repeats = self.repeats(positions)
        return SetCounts(
            positions,
            self.row_count - repeats,
            self._sharing(positions) if repeats else 0,
            int(np.count_nonzero(missing)),
        )

    def repeats(self, positions: tuple[int, ...], *, step: int = 1) -> int:
        """Rows repeating an earlier row's combination, among every step-th row.

        With step 1, the rows minus the distinct combinations.
        """
        key = (positions, step)
        if key not in self._repeats:
            if len(positions) == 1 and step == 1:
                column = self.column(positions[0])
                self._repeats[key] = self.row_count - column.distinct_row_count
            else:
                rows = self._shared_rows(positions, step)
                self.rows_hashed += len(rows)
                combined, span = self._combined(positions, rows)
                if span <= _BINCOUNT_FACTOR * len(rows):
                    distinct = np.count_nonzero(np.bincount(combined))
                    self._repeats[key] = len(rows) - int(distinct)
                else:
                    duplicated = pd.Series(combined).duplicated()
                    self._repeats[key] = int(np.count_nonzero(duplicated))
        return self._repeats[key]

    def _sharing(self, positions: tuple[int, ...]) -> int:
        """Rows whose combination another row shares too."""
        if len(positions) == 1:
            return self.column(positions[0]).duplicate_group_row_count
        rows = self._shared_rows(positions, 1)
        if not len(rows):
            return 0
        combined, _ = self._combined(positions, rows)
        group_sizes = np.bincount(pd.factorize(combined)[0])
        return len(rows) - int(np.count_nonzero(group_sizes == 1))

    def _shared_rows(self, positions: tuple[int, ...], step: int) -> np.ndarray:
        """Every step-th row whose value another row shares in each column.

        Only those rows can share a combination; any other row's is unique.
        """
        shared = self.column(positions[0]).shared.copy()
        for position in positions[1:]:
            shared &= self.column(position).shared
        rows = np.flatnonzero(shared)
        return rows[rows % step == 0] if step > 1 else rows

    def _combined(
        self, positions: tuple[int, ...], rows: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """One integer per row for the set's combination, and their range."""
        combined = np.zeros(len(rows), dtype=np.int64)
        span = 1
        for position in positions:
            column = self.column(position)
            if span * column.size >= 2**62:
                refactorized, uniques = pd.factorize(combined)
                combined, span = refactorized.astype(np.int64), len(uniques)
            combined = combined * column.size + column.codes[rows].astype(np.int64)
            span *= column.size
        return combined, span

    def rank(self, non_keys: Sequence[tuple[tuple[int, ...], int]]) -> Ranking:
        """The most unique minimal sets among those examined, best first.

        Sets are ordered by distinct combinations, then by fewer columns, then
        by column position. A set is minimal when each column adds combinations:
        no smaller set of its columns has as many.
        """
        n = self.row_count
        budget = self.rows_hashed + max(RANKING_PASSES * n, RANKING_ROWS)
        # Each entry: (-distinct, size, positions), so that sorting puts the
        # most unique first.
        best: list[tuple[int, int, tuple[int, ...]]] = []
        queue = []
        for positions, repeats in non_keys:
            if not self._needs_hashing(positions):
                # Counted already, single columns by their own codes.
                best = self._entered(best, positions)
                continue
            bound = self._bound(positions, repeats, missing_everywhere=0)
            if self._cannot_be_minimal(positions, bound):
                continue
            expected = self._expected_repeats(positions)
            queue.append((-bound, expected, positions, 0))
        heapq.heapify(queue)
        proven = True
        while queue:
            negative_bound, expected, positions, level = heapq.heappop(queue)
            if len(best) == MOST_UNIQUE_SETS:
                worst = best[-1]
                if negative_bound > worst[0]:
                    break  # nothing left can beat the 10th best
                if (negative_bound, len(positions), positions) > worst:
                    continue  # it could only tie, and loses the tie
            if level < _QUARTER:
                if level + 1 == _QUARTER and self.rows_hashed >= budget:
                    proven = False
                    break
                bound = min(-negative_bound, self._tightened(positions, level + 1))
                heapq.heappush(queue, (-bound, expected, positions, level + 1))
                continue
            if self._needs_hashing(positions) and self.rows_hashed >= budget:
                proven = False
                break
            best = self._entered(best, positions)
        if not proven:
            # Sets counted while the queue held them still enter the list.
            for _, _, positions, _ in queue:
                if not self._needs_hashing(positions):
                    best = self._entered(best, positions)
        sets = [self.count(positions) for _, _, positions in best]
        counted = sum(
            len(positions) > 1 and (positions, 1) in self._repeats
            for positions, _ in non_keys
        )
        return Ranking(sets=sets, counted_count=counted, proven=proven)

    def _entered(
        self,
        best: list[tuple[int, int, tuple[int, ...]]],
        positions: tuple[int, ...],
    ) -> list[tuple[int, int, tuple[int, ...]]]:
        """The 10 best with the set counted on all rows and entered if minimal."""
        n = self.row_count
        distinct = n - self.repeats(positions)
        subsets = self._subsets(positions) if len(positions) > 1 else []
        if any(n - self.repeats(subset) == distinct for subset in subsets):
            return best  # a smaller set does as well: not minimal
        entry = (-distinct, len(positions), positions)
        return sorted([*best, entry])[:MOST_UNIQUE_SETS]

    def _needs_hashing(self, positions: tuple[int, ...]) -> bool:
        """Whether counting the set and the subsets it must beat hashes rows."""
        subsets = self._subsets(positions) if len(positions) > 1 else []
        return any(
            len(each) > 1 and (each, 1) not in self._repeats
            for each in [positions, *subsets]
        )

    def _tightened(self, positions: tuple[int, ...], level: int) -> int:
        """A tighter upper bound on the set's distinct combinations."""
        n = self.row_count
        if level == _MISSING_EVERYWHERE:
            missing = self._missing_everywhere(positions)
            return self._bound(positions, 0, missing_everywhere=missing)
        if level == _QUARTER and self._uses_quarter(positions):
            return n - self.repeats(positions, step=QUARTER_STEP)
        return n

    def _subsets(self, positions: tuple[int, ...]) -> list[tuple[int, ...]]:
        """The subsets one column smaller, which a minimal set must beat."""
        return [(*positions[:i], *positions[i + 1 :]) for i in range(len(positions))]

    def _bound(
        self,
        positions: tuple[int, ...],
        repeats: int,
        *,
        missing_everywhere: int,
    ) -> int:
        """An upper bound on the set's distinct combinations.

        Within each group of rows that share one column's value, the other
        columns take at most the product of their distinct counts. A repeat on
        a subset of rows is a repeat on all rows.
        """
        columns = [self.column(position) for position in positions]
        for i, column in enumerate(columns):
            capacity = 1
            for j, other in enumerate(columns):
                if j != i:
                    capacity = min(self.row_count, capacity * other.distinct_row_count)
            repeats = max(
                repeats, column.repeats_at_least(capacity, missing_everywhere)
            )
        return self.row_count - repeats

    def _cannot_be_minimal(self, positions: tuple[int, ...], bound: int) -> bool:
        """Whether a single column of the set already has as many as the bound."""
        return bound <= max(self.column(p).distinct_row_count for p in positions)

    def _missing_everywhere(self, positions: tuple[int, ...]) -> int:
        """Rows missing in every column of the set."""
        masks = []
        for position in positions:
            mask = self.column(position).missing
            if mask is None:
                return 0
            masks.append(mask)
        return int(np.count_nonzero(np.logical_and.reduce(masks)))

    def _expected_repeats(self, positions: tuple[int, ...]) -> float:
        """Rows expected to repeat a combination if the columns were independent.

        Only orders sets with equal bounds, so that likely winners are counted
        first.
        """
        chance = math.prod(self.column(position).collision for position in positions)
        return chance * self.row_count**2 / 2
