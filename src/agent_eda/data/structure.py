"""Analyses of how a dataset's rows are identified."""

from collections.abc import Sequence
from typing import Annotated, get_args

import numpy as np
import pandas as pd
from annotated_types import Ge

from agent_eda._columns import ColumnKind, ColumnRule, select_columns
from agent_eda._keys import (
    SUBSET_ROWS,
    KeySearchTally,
    Ranking,
    Searcher,
    SetCounts,
    hashable,
)
from agent_eda._registry import analysis
from agent_eda._run import (
    budget_exceeded_reason,
    data_shape,
    execution,
    overall_status,
)
from agent_eda._text import MAX_NAMED, join_labels, label_text, plural
from agent_eda.results import (
    CandidateKey,
    ColumnSupport,
    FindCandidateKeysResult,
    Finding,
    KeySearch,
    Limitation,
    Status,
    StatusReason,
    Support,
)

# A budget message suggests raising max_combinations only up to this: beyond it
# the search takes too long to be worth suggesting.
MAX_SUGGESTED_COMBINATIONS = 1_000_000

# Any column can be part of a key; one whose values can't be hashed is left out
# when the search starts, since only its values tell.
_EVERY_KIND: frozenset[ColumnKind] = frozenset(get_args(ColumnKind))
_ANY_COLUMN = ColumnRule(
    default=_EVERY_KIND,
    default_text="every column",
    accepted=_EVERY_KIND,
    accepted_text="columns of any dtype",
    fixes={},
)


@analysis(
    statuses=(
        "completed",
        "budget_exceeded",
        "insufficient_data",
        "not_applicable",
        "failed",
    ),
    finding_codes=("keys_found", "keys_with_missing_values", "no_key_found"),
    tags=(
        "primary key",
        "unique key",
        "composite key",
        "unique id",
        "identifier",
        "uniqueness",
    ),
    work_limits=("max_combinations",),
)
def find_candidate_keys(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    max_key_size: Annotated[int, Ge(1)] = 2,
    max_combinations: Annotated[int, Ge(1)] = 10_000,
) -> FindCandidateKeysResult:
    """Which columns, alone or combined, uniquely identify rows?

    When to use:
        Call it to learn how rows are identified: which single columns or
        combinations of up to max_key_size columns are keys, meaning no two rows
        share their values. It lists minimal keys, keys with missing values,
        which can't serve as primary keys, and, when no key exists, the column
        sets closest to one, such as an ID broken by a few duplicates. Missing
        values compare equal, whichever kind they are: None, NaN, NaT and pd.NA
        are one value, though pandas' duplicated tells None from NaN in an object
        column. With columns=None, float columns holding a fraction are left out:
        a measurement such as 0.5 is no identifier, and listing them buries the
        real keys. For each column's distinct and missing counts, call
        `describe_dataset`; `list_analyses` lists the other analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to search, in the
            order to report them: a list, a tuple, a pandas Index or a single
            name. Columns of every dtype are accepted; one whose values can't be
            hashed, such as lists, is left out and named in left_out_columns.
            None, the default, searches every column in DataFrame order, except
            the non-integral float columns: floats of which some finite value
            isn't a whole number, such as 0.5. They are named in
            non_integral_float_columns with a limitation. Name one in columns to
            search it. A float column holding only whole numbers, with or without
            missing values, stays a candidate.
        max_key_size (int): The most columns a key may combine, at least 1;
            defaults to 2. Each column more multiplies the sets to examine. A value
            above the number of searched columns acts as that number.
        max_combinations (int): The most column sets the search examines, at
            least 1; defaults to 10,000, which covers every set of up to 2 of
            140 columns. Supersets of keys aren't examined and don't count. A
            search that needs more stops with budget_exceeded.

    Returns:
        FindCandidateKeysResult: `keys` lists the minimal keys without missing
        values, each fit to serve as a primary key, fewest columns first.
        `keys_with_missing_values` lists the minimal keys with a missing value
        in some rows. When the search found neither, `most_unique_column_sets`
        lists the 10 minimal sets with the most distinct combinations, most
        first. A set is minimal when each of its columns adds combinations, so
        supersets of a key are never listed. Each entry has `columns`,
        `distinct_count`, `duplicate_row_count` (rows whose combination another
        row shares), `missing_row_count`, `uniqueness_share` and `is_minimal`,
        all counted on all rows. `to_frame` lists every entry with its kind.

        `exhaustive` says whether the search examined every set of up to
        max_key_size columns; a key with more columns can exist either way.
        Each set is checked on 16,384 evenly spaced rows first and, when it has
        several columns and the data more than 65,536 rows, on every 4th row: a
        repeated combination there proves it isn't a key. Only a check on all
        rows confirms a key. `search`
        counts the sets examined, rejected on those subsets and checked on all
        rows. The ranking counts sets on all rows, highest upper bound first,
        until the bounds prove that no other set ranks higher. If that would
        hash more rows than 100 passes over the data, or 100,000,000, it stops
        with a `ranking_not_proven` limitation. `left_out_columns` names the
        columns left out for unhashable values, with an `unhashable_values`
        limitation, and `non_integral_float_columns` the float columns left out
        for their fractions, with a `non_integral_float_columns` limitation.

        Every finding is descriptive:

        - `keys_found`: descriptive; column sets identify every row and have no missing
          values.
        - `keys_with_missing_values`: descriptive; column sets identify every row but
          have missing values, so they can't serve as primary keys.
        - `no_key_found`: descriptive; no examined set identifies every row; the claim
          names the most unique set and its duplicate rows.

    Statuses:
        completed: The search examined every set of up to max_key_size columns,
            apart from supersets of keys.
        budget_exceeded: The complete search covers more column sets than
            max_combinations allows. It stopped at the limit and kept what it
            found, and `status_reason` says how many sets the complete search
            covers and the largest max_key_size that fits max_combinations.
        insufficient_data: The data has fewer than 2 rows, which any column
            identifies.
        not_applicable: No column can be searched: the data has no columns,
            every selected column holds values that can't be hashed, or
            columns=None left out every column for holding fractions.
        failed: An anticipated computation failure stopped the search, and
            `status_reason` gives its code and message. None is known today.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "order": [1, 1, 2, 2, 3, 3],
        ...         "line": [1, 2, 1, 2, 1, 2],
        ...         "store": ["a", "a", "b", "b", "a", "a"],
        ...         "sku": ["s1", "s2", "s3", "s4", "s5", None],
        ...     }
        ... )
        >>> res = ae.find_candidate_keys(df)
        >>> print(res)
        find_candidate_keys: completed on 6 rows x 4 columns
        Exhaustive search for keys of up to 2 columns among 4 columns: examined 7 sets, skipped 3 supersets of keys.
        Missing values compare equal; a key with missing values can't serve as a primary key.
        Keys (2 of 2 shown):
        columns          kind                     distinct_count  duplicate_row_count  missing_row_count  uniqueness_share
        'order', 'line'  key                                   6                    0                  0                 1
        'sku'            key_with_missing_values               6                    0                  1                 1
        Findings (2 of 2 shown):
        - descriptive keys_found: 1 column set identifies every row and has no missing values: 'order', 'line'.
        - descriptive keys_with_missing_values: 1 column set identifies every row but has missing values: 'sku'.
        Limitations: none
        More: res.to_frame() has all 2 column sets; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.keys[0].columns
        ['order', 'line']

    See Also:
        describe_dataset: Distinct and missing counts of every column, whatever
            its dtype.
        list_analyses: Every analysis, with the question each one answers.
    """
    selection = select_columns("find_candidate_keys", data, columns, _ANY_COLUMN)
    selected = selection.columns
    measurements: list[str] = []
    if columns is None:
        measurements = [
            column.name
            for column in selected
            if _holds_fractions(data.iloc[:, column.position])
        ]
        selected = [column for column in selected if column.name not in measurements]
    row_count = len(data)
    values = {column.name: data.iloc[:, column.position] for column in selected}
    left_out = [name for name, series in values.items() if not hashable(series)]
    names = [name for name in values if name not in left_out]
    status, status_reason = _status_before_search(
        row_count, values, names, measurements
    )
    ranking = None
    combination_count = 0
    if status is None:
        searcher = Searcher([values[name] for name in names])
        combination_count = searcher.combination_count(max_key_size)
        tally = searcher.search(
            max_key_size=max_key_size, max_combinations=max_combinations
        )
        status = "completed"
        if tally.stopped:
            status = "budget_exceeded"
            fitting = searcher.largest_key_size_within(max_combinations, max_key_size)
            status_reason = budget_exceeded_reason(
                code="too_many_combinations",
                needs="The complete search covers up to",
                required=combination_count,
                unit=("column set", "column sets"),
                parameter="max_combinations",
                limit=max_combinations,
                consequence=(
                    f"it stopped after examining {tally.examined_count:,} and kept "
                    "what it found"
                ),
                narrow="pass fewer columns",
                lower=("max_key_size", fitting) if fitting else None,
                raise_up_to=MAX_SUGGESTED_COMBINATIONS,
            )
        if not tally.keys:
            ranking = searcher.rank(tally.non_keys)
    else:
        tally = KeySearchTally()
    keys = [
        _candidate(key, names, row_count)
        for key in tally.keys
        if not key.missing_row_count
    ]
    with_missing = [
        _candidate(key, names, row_count) for key in tally.keys if key.missing_row_count
    ]
    most_unique = [
        _candidate(counts, names, row_count)
        for counts in (ranking.sets if ranking else [])
    ]
    exhaustive = status == "completed"
    return FindCandidateKeysResult(
        analysis="find_candidate_keys",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": list(values),
            "max_key_size": max_key_size,
            "max_combinations": max_combinations,
        },
        support=Support(
            row_count=row_count,
            column_count=len(selected),
            columns=[_support(name, row_count, name in left_out) for name in values],
        ),
        findings=[
            *_keys_found(keys),
            *_keys_with_missing_values(with_missing),
            *_no_key_found(
                most_unique,
                largest=min(max_key_size, len(names)),
                examined=tally.examined_count,
                exhaustive=exhaustive,
            ),
        ],
        limitations=[
            *_non_integral_float_columns(measurements),
            *_unhashable_values(left_out),
            *_ranking_not_proven(ranking, tally),
        ],
        execution=execution(("numpy", "pandas")),
        keys=keys,
        keys_with_missing_values=with_missing,
        most_unique_column_sets=most_unique,
        exhaustive=exhaustive,
        search=KeySearch(
            column_count=len(names) if tally.examined_count else 0,
            combination_count=combination_count,
            examined_count=tally.examined_count,
            rejected_on_subset_count=tally.rejected_on_subset_count,
            checked_on_all_rows_count=tally.checked_on_all_rows_count,
            subset_row_count=min(row_count, SUBSET_ROWS),
        ),
        left_out_columns=left_out,
        non_integral_float_columns=measurements,
    )


def _sets_text(candidates: list[CandidateKey]) -> str:
    """Name column sets as "'a'; 'b', 'c'", at most 5 of them, counting the rest."""
    named = [
        ", ".join(label_text(name) for name in candidate.columns)
        for candidate in candidates
    ]
    shown = "; ".join(named[:MAX_NAMED])
    if len(named) > MAX_NAMED:
        shown += f" and {len(named) - MAX_NAMED:,} more"
    return shown


def _columns_of(candidates: list[CandidateKey]) -> list[str]:
    """Every column the candidates hold, each once, in order of first mention."""
    return list(dict.fromkeys(name for c in candidates for name in c.columns))


def _keys_found(keys: list[CandidateKey]) -> list[Finding]:
    if not keys:
        return []
    identify = plural(len(keys), "column set identifies", "column sets identify")
    has = "has" if len(keys) == 1 else "have"
    return [
        Finding(
            code="keys_found",
            claim=(
                f"{identify} every row and {has} no missing values: {_sets_text(keys)}."
            ),
            assessment="descriptive",
            columns=_columns_of(keys),
            fields=[f"keys[{i}]" for i in range(len(keys))],
        )
    ]


def _keys_with_missing_values(keys: list[CandidateKey]) -> list[Finding]:
    if not keys:
        return []
    identify = plural(len(keys), "column set identifies", "column sets identify")
    has = "has" if len(keys) == 1 else "have"
    return [
        Finding(
            code="keys_with_missing_values",
            claim=(
                f"{identify} every row but {has} missing values: {_sets_text(keys)}."
            ),
            assessment="descriptive",
            columns=_columns_of(keys),
            fields=[
                f"keys_with_missing_values[{i}].missing_row_count"
                for i in range(len(keys))
            ],
        )
    ]


def _no_key_found(
    most_unique: list[CandidateKey], *, largest: int, examined: int, exhaustive: bool
) -> list[Finding]:
    if not most_unique:
        return []
    if not exhaustive:
        searched = f"No key in the {plural(examined, 'set', 'sets')} examined"
    elif largest == 1:
        searched = "No single column is a key"
    else:
        searched = f"No key of up to {largest:,} columns"
    best = most_unique[0]
    duplicates = plural(best.duplicate_row_count, "duplicate row", "duplicate rows")
    return [
        Finding(
            code="no_key_found",
            claim=(
                f"{searched}; the most unique, {_sets_text([best])}, has {duplicates}."
            ),
            assessment="descriptive",
            columns=list(best.columns),
            fields=[
                "keys",
                "keys_with_missing_values",
                "most_unique_column_sets[0].duplicate_row_count",
            ],
        )
    ]


_ONE_ROW = StatusReason(
    code="one_row",
    message=(
        "The data has 1 row, which every column identifies; finding keys needs at "
        "least 2 rows."
    ),
)
_NO_COLUMNS = StatusReason(
    code="no_columns", message="The data has no columns to search."
)
_ONLY_MEASUREMENTS = StatusReason(
    code="no_columns",
    message=(
        "Every column holds non-integral floats, which are measurements rather than "
        "IDs and are left out unless columns names them."
    ),
)
_NO_HASHABLE_COLUMNS = StatusReason(
    code="no_hashable_columns",
    message=(
        "No selected column can be searched: their values can't be hashed, such as "
        "lists or dicts."
    ),
)


def _status_before_search(
    row_count: int,
    values: dict[str, pd.Series],
    names: list[str],
    measurements: list[str],
) -> tuple[Status | None, StatusReason | None]:
    """The status when the data allows no search; None for both when it does."""
    if row_count == 0 or not values:
        return overall_status(
            row_count,
            [],
            no_columns=_ONLY_MEASUREMENTS if measurements else _NO_COLUMNS,
        )
    if row_count == 1:
        return "insufficient_data", _ONE_ROW
    if not names:
        return "not_applicable", _NO_HASHABLE_COLUMNS
    return None, None


def _candidate(counts: SetCounts, names: list[str], row_count: int) -> CandidateKey:
    return CandidateKey(
        columns=[names[position] for position in counts.positions],
        distinct_count=counts.distinct_count,
        duplicate_row_count=counts.duplicate_row_count,
        missing_row_count=counts.missing_row_count,
        uniqueness_share=counts.distinct_count / row_count,
        is_minimal=True,
    )


def _support(name: str, row_count: int, left_out: bool) -> ColumnSupport:
    """Every row of a searched column is evaluated; a column left out has none."""
    excluded = {"unhashable_values": row_count} if left_out and row_count else {}
    return ColumnSupport(
        column=name,
        eligible_count=row_count,
        evaluated_count=row_count - sum(excluded.values()),
        excluded_count=sum(excluded.values()),
        excluded_by_reason=excluded,
    )


def _ranking_not_proven(
    ranking: Ranking | None, tally: KeySearchTally
) -> list[Limitation]:
    if ranking is None or ranking.proven:
        return []
    # Sets of 2 or more columns: single columns cost nothing to count.
    examined = sum(len(positions) > 1 for positions, _ in tally.non_keys)
    return [
        Limitation(
            code="ranking_not_proven",
            reason=(
                f"Counted {ranking.counted_count:,} of {examined:,} sets on all rows "
                "before the work limit; an uncounted set could rank higher."
            ),
            fields=["most_unique_column_sets"],
        )
    ]


def _holds_fractions(values: pd.Series) -> bool:
    """Whether a float column holds a finite value that isn't a whole number."""
    if not pd.api.types.is_float_dtype(values.dtype):
        return False
    numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
    return bool(np.any(np.isfinite(numbers) & (numbers != np.floor(numbers))))


def _non_integral_float_columns(measurements: list[str]) -> list[Limitation]:
    if not measurements:
        return []
    columns = plural(len(measurements), "column", "columns")
    them = "it" if len(measurements) == 1 else "them"
    return [
        Limitation(
            code="non_integral_float_columns",
            reason=(
                f"The search left out {columns} holding non-integral floats, which "
                f"are measurements rather than IDs: {join_labels(measurements)}. "
                f"Name {them} in columns to search {them}."
            ),
            fields=["non_integral_float_columns"],
        )
    ]


def _unhashable_values(left_out: list[str]) -> list[Limitation]:
    if not left_out:
        return []
    columns = plural(len(left_out), "column", "columns")
    return [
        Limitation(
            code="unhashable_values",
            reason=(
                f"The search left out {columns} whose values can't be hashed, such as "
                f"lists: {join_labels(left_out)}."
            ),
            fields=["left_out_columns"],
        )
    ]
