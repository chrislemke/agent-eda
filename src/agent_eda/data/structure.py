"""Analyses of what a dataset contains and how its rows are identified."""

import sys
from collections.abc import Sequence
from typing import Annotated, cast, get_args

import numpy as np
import pandas as pd
from annotated_types import Ge

from agent_eda._columns import (
    ColumnKind,
    ColumnRule,
    Labels,
    first_positions,
    select_columns,
)
from agent_eda._keys import (
    SUBSET_ROWS,
    KeySearchTally,
    Ranking,
    Searcher,
    SetCounts,
    hashable,
)
from agent_eda._numeric import stored_nan_count, stored_nan_limitations
from agent_eda._registry import analysis
from agent_eda._result_base import (
    ColumnSupport,
    ExclusionReason,
    Finding,
    Limitation,
    Status,
    StatusReason,
    Support,
)
from agent_eda._row_ids import duplicated_labels
from agent_eda._run import (
    budget_exceeded_reason,
    overall_status,
)
from agent_eda._text import MAX_NAMED, join_labels, label_text, plural
from agent_eda._values import is_constant, nan_as_missing, value_codes
from agent_eda.data._structure_results import (
    CandidateKey,
    DescribeDatasetColumn,
    DescribeDatasetIndex,
    DescribeDatasetResult,
    FindCandidateKeysResult,
    KeySearch,
    Provenance,
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
        Call it to learn how rows are identified: which single columns or combinations
        of up to max_key_size columns are keys, meaning no two rows share their values.
        It lists minimal keys, keys with missing values, which can't serve as primary
        keys, and, when no key exists, the column sets closest to one, such as an ID
        broken by a few duplicates. Missing values compare equal, whichever kind they
        are: None, NaN, NaT and NA are one value. With the default columns, float
        columns holding a fraction are left out: a measurement such as 0.5 is no
        identifier. For the rows that repeat on a near-key's columns, call
        `find_duplicate_rows`; for each column's distinct and missing counts,
        `describe_dataset`.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to search, in the
            order to report them: a list, a tuple, a pandas Index or a single
            name. Columns of every dtype are accepted; one whose values can't be
            hashed, such as lists, is left out and named in left_out_columns.
            None, the default, searches every column in DataFrame order, except
            the non-integral float columns, floats of which some finite value
            isn't a whole number, such as 0.5: they are named in
            non_integral_float_columns with a limitation, and naming one in
            columns searches it. A float column of whole numbers, missing
            values or not, stays a candidate.
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
        `uniqueness_share` and the counts behind it, all on all rows.

        `exhaustive` says whether the search examined every set of up to
        max_key_size columns; a key with more columns can exist either way.
        Each set is checked on evenly spaced subsets of rows first: a repeated
        combination there proves it isn't a key, and only a check on all rows
        confirms one. `search` counts the sets examined, rejected on those
        subsets and checked on all rows. The ranking stops with a
        `ranking_not_proven` limitation when proving it would hash more rows
        than 100 passes over the data, or 100,000,000. `left_out_columns` names
        the columns left out for unhashable values, with an `unhashable_values`
        limitation, and `non_integral_float_columns` the float columns left out
        for their fractions, with a limitation of the same name.
        `nan_counted_as_missing` names the columns that store NaN as a value,
        which counts as missing here.

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
            Reason code: `too_many_combinations`.
        insufficient_data: The data has no rows, or just 1 row, which any
            column identifies. Reason codes: `no_rows` and `one_row`.
        not_applicable: No column can be searched: the data has no columns,
            every selected column holds values that can't be hashed, or
            the default columns left out every column for holding fractions.
            Reason codes: `no_columns` and `unhashable_values`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 1, 2, 2, 3, 3],
        ...         "seq": [1, 2, 1, 2, 1, 2],
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
        columns      kind                     distinct_row_count  duplicate_group_row_count  missing_row_count  uniqueness_share
        'id', 'seq'  key                                       6                          0                  0                 1
        'sku'        key_with_missing_values                   6                          0                  1                 1
        Findings (2 of 2 shown):
        - descriptive keys_found: 1 column set identifies every row and has no missing values: 'id', 'seq'.
        - descriptive keys_with_missing_values: 1 column set identifies every row but has missing values: 'sku'.
        Limitations: none
        More: res.to_frame() has all 2 column sets; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.keys[0].columns
        ['id', 'seq']

    See Also:
        find_duplicate_rows: Groups of rows that are equal in every column, or in the
            columns of a near-key, with their row IDs.
        profile_target: The classes and imbalance, or the statistics, of the column a
            model will predict, checked against the task.
        describe_dataset: Distinct and missing counts of every column, whatever its
            dtype.
    """  # noqa: E501 - the example prints a table 120 characters wide
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
    return FindCandidateKeysResult.for_call(
        parameters={"columns": list(values)},
        status=status,
        status_reason=status_reason,
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
            *stored_nan_limitations(
                [name for name in names if stored_nan_count(values[name])],
                ["keys_with_missing_values", "most_unique_column_sets"],
                "missing_row_count counts it",
            ),
            *_non_integral_float_columns(measurements),
            *_unhashable_values(left_out),
            *_ranking_not_proven(ranking, tally),
        ],
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
    duplicates = plural(
        best.duplicate_group_row_count, "duplicate row", "duplicate rows"
    )
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
                "most_unique_column_sets[0].duplicate_group_row_count",
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
    code="unhashable_values",
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
        distinct_row_count=counts.distinct_row_count,
        duplicate_group_row_count=counts.duplicate_group_row_count,
        missing_row_count=counts.missing_row_count,
        uniqueness_share=counts.distinct_row_count / row_count,
        is_minimal=True,
    )


def _support(name: str, row_count: int, left_out: bool) -> ColumnSupport:
    """Every row of a searched column is evaluated; a column left out has none."""
    excluded: dict[ExclusionReason, int] = (
        {"unhashable_values": row_count} if left_out and row_count else {}
    )
    return ColumnSupport.of(name, row_count, excluded)


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


@analysis(
    statuses=("completed",),
    finding_codes=(
        "empty_data",
        "duplicated_column_labels",
        "colliding_column_labels",
        "multiindex_columns",
        "non_unique_index",
        "all_missing_columns",
        "constant_columns",
    ),
    tags=(
        "overview",
        "shape",
        "size",
        "dtypes",
        "memory",
        "index",
        "provenance",
        "column labels",
        "duplicated column labels",
        "rows",
        "row count",
        "zero variance",
        "constant",
        "constant columns",
    ),
    any_column_labels=True,
)
def describe_dataset(
    data: pd.DataFrame,
    *,
    source: str | None = None,
    source_version: str | None = None,
) -> DescribeDatasetResult:
    """What does this dataset contain?

    When to use:
        Call it first on any new DataFrame. It reports the shape, each column's dtype,
        missing and distinct counts and memory, the index and the provenance you supply,
        and flags structural problems: empty data, duplicated or colliding column
        labels, MultiIndex columns, a non-unique index, all-missing columns and constant
        columns. Every other analysis raises on bad column labels, so run this one to
        find them. It never shows a raw data value. A dtype only tells how pandas stores
        a column: to learn each column's logical type from its values, call
        `infer_column_semantics` next. For missing values per column and per row, call
        `profile_missing_values`; for how numeric values are distributed,
        `summarize_numeric`.

    Args:
        data (pd.DataFrame): The data to describe. Any column labels are
            accepted, including duplicated ones and MultiIndex columns. It is
            never modified.
        source (str | None): Where the data came from, such as a file path, URL
            or table name, recorded exactly as given. None, the default, records
            the source as not supplied; it is never inferred.
        source_version (str | None): The version of the source, such as a
            snapshot date or a commit, recorded exactly as given. None, the
            default, records it as not supplied.

    Returns:
        DescribeDatasetResult: Read `row_count`, `column_count` and
        `memory_bytes` (columns plus index) for the size. `columns` has one
        entry per column in DataFrame order with its `dtype` as pandas reports
        it, `non_missing_count`, `missing_count` (also NaN that a pyarrow-backed
        or nullable float column stores as a value, with a `nan_counted_as_missing`
        limitation naming such columns), `distinct_count` (null, with an
        `unhashable_values` limitation, when values such as lists can't be
        hashed) and `memory_bytes`. `index` holds the index dtype, whether its
        labels are unique and how many are missing. `provenance` holds the
        source and version as supplied, and `not_supplied` names what wasn't.
        Memory counts the values only, not pandas' lookup tables or the UTF-8
        copies Python caches inside strings.

        Every finding is suspected:

        - `empty_data`: suspected; the data has no rows, no columns or neither.
        - `duplicated_column_labels`: suspected; a column label occurs more than once.
        - `colliding_column_labels`: suspected; different labels share a string form,
          such as 1 and '1'.
        - `multiindex_columns`: suspected; the column labels form a MultiIndex.
        - `non_unique_index`: suspected; index labels repeat, so they don't identify
          rows.
        - `all_missing_columns`: suspected; a column holds only missing values.
        - `constant_columns`: suspected; a column holds a single distinct non-missing
          value in at least two rows; NaN is missing.

    Statuses:
        completed: The description finished, also for data with no rows, no
            columns or neither; a count that can't be computed is null, with a
            limitation.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, 51, 29, 51],
        ...         "score": [0.5, None, 0.75, 0.5],
        ...         "plan": [1] * 4,
        ...     }
        ... )
        >>> res = ae.describe_dataset(df, source="survey.csv")
        >>> print(res)
        describe_dataset: completed on 4 rows x 3 columns
        Memory: ... in total. Index: int64, unique, 0 missing labels.
        Provenance: source 'survey.csv'; source_version not supplied.
        Columns (3 of 3 shown):
        column  dtype    non_missing_count  missing_count  distinct_count  memory_bytes
        plan    int64                    4              0               1            32
        age     int64                    4              0               3            32
        score   float64                  3              1               2            32
        Findings (1 of 1 shown):
        - suspected constant_columns: 1 column holds a single distinct non-missing value: 'plan'.
        Limitations: none
        More: res.to_frame() has all 3 columns; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns[1].missing_count
        1

    See Also:
        infer_column_semantics: The logical type of each column, from its values, with
            flags for identifiers and numbers or dates stored as text.
        profile_missing_values: Missing values per column and per row, counting declared
            sentinel values.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every numeric
            column.
    """
    row_count = len(data)
    labels = Labels(data.columns)
    columns = [
        _describe_column(data.iloc[:, position], position, label, row_count)
        for position, label in enumerate(labels.labels)
    ]
    index = _describe_index(data.index)
    findings = [
        *_empty_data(row_count, len(columns)),
        *_duplicated_labels(labels),
        *_colliding_labels(labels),
        *_multiindex_columns(labels.level_count),
        *_non_unique_index(index),
        *_all_missing_columns(columns, labels, row_count),
        *_constant_columns(columns, labels),
    ]
    return DescribeDatasetResult.for_call(
        status="completed",
        status_reason=None,
        support=Support(
            row_count=row_count,
            column_count=len(columns),
            columns=[ColumnSupport.of(c.column, row_count) for c in columns],
        ),
        findings=findings,
        limitations=[
            *_stored_nan(data, labels),
            *_unhashable_columns(columns, labels),
        ],
        row_count=row_count,
        column_count=len(columns),
        column_level_count=labels.level_count,
        memory_bytes=sum(c.memory_bytes for c in columns) + index.memory_bytes,
        columns=columns,
        index=index,
        provenance=Provenance(
            source=source,
            source_version=source_version,
            not_supplied=[
                name
                for name, value in (
                    ("source", source),
                    ("source_version", source_version),
                )
                if value is None
            ],
        ),
    )


def _describe_column(
    series: pd.Series, position: int, label: object, row_count: int
) -> DescribeDatasetColumn:
    if _holds_python_objects(series.dtype):
        missing, distinct, memory = _object_column(
            np.asarray(series.array, dtype=object)
        )
    else:
        # NaN that the column stores as a value is missing, as in every analysis.
        missing = int(nan_as_missing(series).isna().sum())
        distinct = _distinct_count(series)
        memory = _memory_bytes(series)
    return DescribeDatasetColumn(
        position=position,
        column=str(label),
        label_type=type(label).__name__,
        dtype=str(series.dtype),
        non_missing_count=row_count - missing,
        missing_count=missing,
        distinct_count=distinct,
        memory_bytes=memory,
    )


def _holds_python_objects(dtype: object) -> bool:
    """Whether values are stored as Python objects: object dtype or python strings."""
    if isinstance(dtype, pd.StringDtype):
        return dtype.storage == "python"
    return isinstance(dtype, np.dtype) and dtype.kind == "O"


def _object_column(objects: np.ndarray) -> tuple[int, int | None, int]:
    """The missing count, distinct count and bytes of a column of Python objects.

    One hashing pass gives all three. The distinct count is None when values
    such as lists can't be hashed.
    """
    distinct = value_codes(objects, unhashable="unsupported")
    if distinct is None:
        return int(pd.isna(objects).sum()), None, _object_bytes(objects)
    codes = distinct.codes
    uniques = np.asarray(distinct.uniques, dtype=object)
    missing = codes < 0
    if pd.api.types.infer_dtype(uniques, skipna=False) == "string":
        # Equal strings take equal space, so each distinct one is sized once.
        counts = np.bincount(codes[~missing], minlength=len(uniques))
        sizes = int(_sizes(uniques) @ counts) + int(_sizes(objects[missing]).sum())
        memory = objects.nbytes + sizes
    else:
        memory = _object_bytes(objects)
    return int(missing.sum()), len(uniques), memory


def _distinct_count(series: pd.Series) -> int | None:
    """Distinct non-missing values, or None when the values can't be hashed."""
    found = value_codes(series, unhashable="unsupported")
    return None if found is None else len(found.uniques)


def _describe_index(index: pd.Index) -> DescribeDatasetIndex:
    # The rule every analysis uses for repeated row IDs.
    duplicated = int(duplicated_labels(index, keep="first").sum())
    if isinstance(index, pd.MultiIndex):
        missing = int(np.any([codes == -1 for codes in index.codes], axis=0).sum())
    else:
        missing = int(index.isna().sum())
    return DescribeDatasetIndex(
        dtype=str(index.dtype),
        level_count=index.nlevels,
        is_unique=duplicated == 0,
        duplicated_count=duplicated,
        missing_count=missing,
        memory_bytes=_memory_bytes(index),
    )


def _memory_bytes(values: pd.Series | pd.Index) -> int:
    """Deep memory of the values in bytes.

    pandas' memory_usage() of an index or a categorical adds the hash tables
    that lookups build and keep, so its answer depends on what the caller did
    before. This counts the values alone, so earlier lookups don't change it.
    """
    if isinstance(values, pd.MultiIndex):
        levels = sum(_memory_bytes(level) for level in values.levels)
        return levels + sum(int(codes.nbytes) for codes in values.codes)
    if isinstance(values, pd.RangeIndex):
        return int(values.nbytes)
    if isinstance(values.dtype, pd.CategoricalDtype):
        categorical = cast("pd.Categorical", values.array)
        return int(categorical.codes.nbytes) + _memory_bytes(categorical.categories)
    if _holds_python_objects(values.dtype):
        return _object_bytes(np.asarray(values.array, dtype=object))
    if isinstance(values.dtype, pd.SparseDtype) and values.dtype.subtype.kind == "O":
        # pandas can't size the Python objects behind a sparse array.
        sparse = cast("pd.arrays.SparseArray", values.array)
        stored = np.asarray(sparse.sp_values, dtype=object)
        return int(sparse.nbytes) + _object_bytes(stored) - int(stored.nbytes)
    if isinstance(values, pd.Index):
        # The explicit dtype stops pandas 3 from turning object text into str.
        values = pd.Series(values.array, copy=False, dtype=values.dtype)
    return int(values.memory_usage(deep=True, index=False))


def _object_bytes(objects: np.ndarray) -> int:
    """Bytes of an array of Python objects: a reference per value and each object."""
    return objects.nbytes + int(_sizes(objects).sum())


def _sizes(objects: np.ndarray) -> np.ndarray:
    """Each object's size in bytes, sys.getsizeof as pandas' memory_usage counts it.

    A non-ASCII str counts without the UTF-8 copy Python keeps inside it once
    anything has encoded it, such as pyarrow, pandas' hashing or this analysis:
    otherwise the same strings would grow from one call to the next. ASCII
    text is its own UTF-8 form.
    """
    count = len(objects)
    if pd.api.types.infer_dtype(objects, skipna=False) == "string" and all(
        map(str.isascii, objects)
    ):
        return np.fromiter(map(sys.getsizeof, objects), dtype=np.int64, count=count)
    return np.fromiter(map(_size, objects), dtype=np.int64, count=count)


def _size(value: object) -> int:
    if isinstance(value, str) and not value.isascii():
        widest = ord(max(value))
        width = 1 if widest < 0x100 else 2 if widest < 0x10000 else 4
        header = _text_header(type(value))
        if header is not None:
            return header + (len(value) + 1) * width
    return sys.getsizeof(value)


def _text_header(kind: type[str]) -> int | None:
    """The bytes a non-ASCII str of this type takes besides its characters.

    A str keeps its characters inside the object. A subclass such as np.str_
    keeps them in a separate buffer, so its header is that of an empty one made
    here, which holds no UTF-8 copy, less its terminating character. None for
    a subclass that can't be made from "".
    """
    if kind is str:
        return _TEXT_HEADER
    try:
        return sys.getsizeof(kind.__new__(kind, "")) - 1
    except TypeError:
        return None


# The bytes a non-ASCII str takes besides its characters, measured on one made
# here, which nothing has encoded: 2 one-byte characters and a terminating one.
_TEXT_HEADER = "".join(map(chr, (0xE9, 0xE9))).__sizeof__() - 3


def _empty_data(row_count: int, column_count: int) -> list[Finding]:
    missing = [
        part
        for part, count in (("no rows", row_count), ("no columns", column_count))
        if count == 0
    ]
    if not missing:
        return []
    return [
        Finding(
            code="empty_data",
            claim=f"The data has {' and '.join(missing)}.",
            assessment="suspected",
            columns=[],
            fields=["row_count", "column_count"],
        )
    ]


def _duplicated_labels(labels: Labels) -> list[Finding]:
    positions = labels.repeated()
    if not positions:
        return []
    first = first_positions(labels, positions)
    occurs = plural(len(first), "column label occurs", "column labels occur")
    return [
        Finding(
            code="duplicated_column_labels",
            claim=f"{occurs} more than once: {labels.show(first)}.",
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].column" for p in positions],
        )
    ]


def _colliding_labels(labels: Labels) -> list[Finding]:
    positions = labels.colliding()
    if not positions:
        return []
    return [
        Finding(
            code="colliding_column_labels",
            claim=(
                f"{len(positions):,} different column labels share a string form "
                f"with another label: {labels.show(positions)}."
            ),
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[
                field
                for p in positions
                for field in (f"columns[{p}].column", f"columns[{p}].label_type")
            ],
        )
    ]


def _multiindex_columns(level_count: int) -> list[Finding]:
    if level_count == 1:
        return []
    return [
        Finding(
            code="multiindex_columns",
            claim=f"The column labels form a MultiIndex with {level_count} levels.",
            assessment="suspected",
            columns=[],
            fields=["column_level_count"],
        )
    ]


def _non_unique_index(index: DescribeDatasetIndex) -> list[Finding]:
    if index.is_unique:
        return []
    repeat = plural(
        index.duplicated_count, "index label repeats", "index labels repeat"
    )
    return [
        Finding(
            code="non_unique_index",
            claim=(
                f"{repeat} an earlier label, so index labels don't identify "
                "rows uniquely."
            ),
            assessment="suspected",
            columns=[],
            fields=["index.is_unique", "index.duplicated_count"],
        )
    ]


def _all_missing_columns(
    columns: list[DescribeDatasetColumn], labels: Labels, row_count: int
) -> list[Finding]:
    if row_count == 0:
        return []
    positions = [c.position for c in columns if c.non_missing_count == 0]
    if not positions:
        return []
    contain = plural(len(positions), "column contains", "columns contain")
    return [
        Finding(
            code="all_missing_columns",
            claim=f"{contain} only missing values: {labels.show(positions)}.",
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].missing_count" for p in positions],
        )
    ]


def _stored_nan(data: pd.DataFrame, labels: Labels) -> list[Limitation]:
    positions = [
        position
        for position in range(data.shape[1])
        if stored_nan_count(data.iloc[:, position])
    ]
    return stored_nan_limitations(
        [labels.labels[p] for p in positions],
        [f"columns[{p}].missing_count" for p in positions],
        "missing_count counts it",
    )


def _unhashable_columns(
    columns: list[DescribeDatasetColumn], labels: Labels
) -> list[Limitation]:
    positions = [c.position for c in columns if c.distinct_count is None]
    if not positions:
        return []
    hold = plural(len(positions), "column holds", "columns hold")
    its = "its" if len(positions) == 1 else "their"
    return [
        Limitation(
            code="unhashable_values",
            reason=(
                f"{hold} values that can't be hashed, such as lists or dicts, so "
                f"{its} distinct values weren't counted: {labels.show(positions)}."
            ),
            fields=[f"columns[{p}].distinct_count" for p in positions],
        )
    ]


def _constant_columns(
    columns: list[DescribeDatasetColumn], labels: Labels
) -> list[Finding]:
    positions = [
        c.position
        for c in columns
        if is_constant(c.non_missing_count, c.distinct_count)
    ]
    if not positions:
        return []
    hold = plural(len(positions), "column holds", "columns hold")
    return [
        Finding(
            code="constant_columns",
            claim=(
                f"{hold} a single distinct non-missing value: {labels.show(positions)}."
            ),
            assessment="suspected",
            columns=[labels.texts[p] for p in positions],
            fields=[f"columns[{p}].distinct_count" for p in positions],
        )
    ]
