"""Analyses that measure how strongly columns are associated with each other."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Literal, get_args

import numpy as np
import pandas as pd
from annotated_types import Ge

from agent_eda._association import Codes, Measure, NumericMethod, measure_pairs
from agent_eda._columns import (
    TO_LABELS,
    TO_NUMBERS,
    Column,
    ColumnKind,
    ColumnRule,
    Selection,
    column_kind,
    select_columns,
)
from agent_eda._numeric import (
    ReadEffects,
    read_limitations,
    read_numeric,
    stored_nan_count,
)
from agent_eda._registry import analysis
from agent_eda._result_base import (
    ColumnSupport,
    ExclusionReason,
    Finding,
    Limitation,
    LimitationCode,
    Status,
    StatusReason,
    Support,
)
from agent_eda._run import NO_ROWS, budget_exceeded_reason
from agent_eda._text import join_items, join_labels, plural
from agent_eda._values import value_codes
from agent_eda.errors import InvalidArgumentError
from agent_eda.statistics._association_results import (
    AssociationColumn,
    AssociationMethod,
    AssociationPair,
    AssociationType,
    ComputeAssociationMatrixResult,
    LeftOutColumn,
    LeftOutReason,
    UndefinedReason,
)

CategoricalMethod = Literal["cramers_v", "cramers_v_bias_corrected"]
MixedMethod = Literal["correlation_ratio_bias_corrected", "correlation_ratio"]

_ANALYSIS = "compute_association_matrix"
_MIXED_METHODS = frozenset(get_args(MixedMethod))
_PAIRED_KINDS: frozenset[ColumnKind] = frozenset(
    {"numeric", "boolean", "categorical", "text", "object"}
)
_ASSOCIATION = ColumnRule(
    default=_PAIRED_KINDS,
    default_text="every numeric, boolean, text, object and categorical column",
    accepted=_PAIRED_KINDS,
    accepted_text=(
        "numeric, boolean, text, object and categorical columns, nullable and "
        "pyarrow-backed ones of those kinds included"
    ),
    fixes={
        "datetime": f"{TO_NUMBERS['datetime']}; {TO_LABELS['datetime']}",
        "timedelta": TO_NUMBERS["timedelta"],
    },
)
_CATEGORICAL = ColumnRule(
    default=frozenset(),
    default_text="no numeric column read as categorical",
    accepted=_PAIRED_KINDS,
    accepted_text=(
        "numeric columns to read as categorical; boolean, text, object and "
        "categorical columns are categorical anyway"
    ),
    fixes={kind: TO_LABELS[kind] for kind in ("datetime", "timedelta")},
)
# Why columns=None leaves out a column of each other kind.
_KIND_SKIPS: dict[str, LeftOutReason] = {
    "datetime": "datetime",
    "timedelta": "timedelta",
    "other": "unsupported_dtype",
}
_SKIP_MESSAGES: dict[LeftOutReason, str] = {
    "datetime": (
        "Datetimes have no measure here; convert them to numbers or labels and "
        "name the column in columns."
    ),
    "timedelta": (
        "Durations have no measure here; convert them to numbers of seconds first."
    ),
    "unsupported_dtype": (
        "Its dtype holds neither numbers nor categories: Arrow times of day, "
        "binary and nested types and sparse arrays of text or objects are none "
        "of them."
    ),
    "all_missing": "Every value is missing.",
    "no_finite_values": "Every value is missing, inf or -inf.",
    "unhashable_values": (
        "Its values, such as lists or dicts, can't be hashed, so they can't be "
        "compared as categories."
    ),
    "too_many_categories": (
        "It has more distinct values than the max_categories limit of "
        "{max_categories}; raise max_categories to pair it."
    ),
}
# A column with more rows than this first counts the distinct values in this
# many rows, so a column with far too many is left out without reading it all.
PROBE_ROWS = 20_000
# A rank method (spearman, kendall) spends about 0.3 seconds per million rows on
# each numeric pair, and pearson a few milliseconds, so max_pairs counts such a
# pair once per this many rows.
RANK_PAIR_ROWS = 10_000

_FEWER_THAN_TWO_COLUMNS = StatusReason(
    code="no_columns",
    message=(
        "Fewer than two columns can be paired: numeric, boolean, text, object "
        "and categorical columns can, and left_out_columns says why the others "
        "can't."
    ),
)
_NO_DEFINED_PAIRS = StatusReason(
    code="no_values",
    message=(
        "No pair has a value: each pair's undefined_reason says why, such as "
        "too_few_rows."
    ),
)


def _rank_weight(numeric_method: NumericMethod, row_count: int) -> int:
    """How many pairs a numeric pair counts toward max_pairs: 1 for pearson, which
    a batch of matrix products computes for every pair at once, and for a rank
    method one per RANK_PAIR_ROWS rows, rounded up, since it ranks or sorts
    each pair's own complete rows."""
    if numeric_method == "pearson":
        return 1
    return max(1, -(-row_count // RANK_PAIR_ROWS))


@dataclass(frozen=True)
class _Paired:
    """A column the pairs use: float64 values or integer codes."""

    entry: AssociationColumn
    # A numeric column as read_numeric reads it, and its float64 values with
    # NaN where missing; both None for a categorical column.
    numbers: ReadEffects | None
    floats: np.ndarray | None
    # Codes from 0, -1 where missing, for a categorical column.
    codes: np.ndarray | None

    def support(self, row_count: int) -> ColumnSupport:
        excluded: dict[ExclusionReason, int] = {}
        if self.entry.missing_count:
            excluded["missing"] = self.entry.missing_count
        if self.entry.infinite_count:
            excluded["infinite"] = self.entry.infinite_count
        return ColumnSupport.of(self.entry.column, row_count, excluded)


@analysis(
    statuses=(
        "completed",
        "budget_exceeded",
        "insufficient_data",
        "not_applicable",
    ),
    finding_codes=("perfect_associations",),
    tags=(
        "correlation",
        "association",
        "cramers v",
        "pearson",
        "spearman",
        "kendall",
        "correlation ratio",
        "related",
        "relationships",
        "dependence",
        "redundant",
        "multicollinearity",
        "collinearity",
    ),
    work_limits=("max_pairs",),
)
def compute_association_matrix(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    numeric_method: NumericMethod = "pearson",
    categorical_method: CategoricalMethod = "cramers_v_bias_corrected",
    mixed_method: MixedMethod = "correlation_ratio_bias_corrected",
    max_categories: Annotated[int, Ge(2)] = 50,
    max_pairs: Annotated[int, Ge(1)] = 5_000,
) -> ComputeAssociationMatrixResult:
    """How strongly are pairs of columns associated?

    When to use:
        Call it after the per-column summaries to see which columns move together: one
        call measures every pair of numeric and categorical columns and lists the
        strongest first. Each pair gets the measure that fits its types and uses its own
        complete rows, so gaps in one column don't shrink every pair. Perfect
        associations are flagged, since one column may duplicate or derive from the
        other. List numeric codes, such as postal codes or class labels 1, 2 and 3, in
        categorical_columns so they count as categories: `infer_column_semantics` finds
        them. It measures association, not causation, without p-values. For how one
        numeric column is distributed, call `summarize_numeric`; for how the column a
        model will predict goes with the others, `profile_target`.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to pair, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself or as its string form. Numeric, boolean, text, object
            and categorical columns are accepted; any other column named here, such as a
            datetime, timedelta, period or list column, is rejected with the code
            `unsupported_column_type`. None, the default, selects every
            numeric, boolean, text, object and categorical column in DataFrame order and
            lists the others in left_out_columns.
        categorical_columns (Sequence[str] | None): Numeric columns to treat as
            categorical, so that codes such as 1, 2 and 3 are compared as
            categories instead of correlated as numbers. Each must be a column
            the call pairs. Listing a boolean, text or categorical column
            changes nothing. None, the default, treats every numeric column
            other than booleans as numeric.
        numeric_method (str): The measure for two numeric columns: 'pearson',
            the default, Pearson's r for linear association; 'spearman',
            Pearson's r of the ranks, tied values sharing their mean rank, for
            monotonic association; or 'kendall', Kendall's tau-b, which adjusts
            for ties. The rank methods rank each pair's own complete rows, one
            pair at a time: up to about 0.3 seconds per numeric pair per
            million rows, against a few milliseconds for 'pearson', so narrow
            columns before using them on large data.
        categorical_method (str): The measure for two categorical columns:
            'cramers_v_bias_corrected', the default, Cramér's V with Bergsma's
            (2013) correction, which removes most of the upward bias that few
            rows and many levels give V; or 'cramers_v', the uncorrected V of
            scipy.stats.contingency.association.
        mixed_method (str): The measure for a categorical and a numeric
            column, 'correlation_ratio_bias_corrected' or 'correlation_ratio'.
            The default is the square root of epsilon
            squared, Kelley's (1935) bias-adjusted share of the numeric column's
            variance that the categories explain, clipped at 0. The plain ratio
            eta rises with the number of levels even when the columns are
            unrelated: 40 levels in 60 rows give about 0.8 for pure noise.
        max_categories (int): The most distinct values a categorical column may
            have; a column with more is left out with reason
            too_many_categories. At least 2; defaults to 50.
        max_pairs (int): The most pairs the call computes, a work limit. With
            more, it returns budget_exceeded without computing any. A numeric
            pair under 'spearman' or 'kendall' counts once per 10,000 rows,
            rounded up, since each is ranked or sorted on its own: 100 columns
            of 1,000,000 rows give 4,950 pairs that count 100 each. At least 1;
            defaults to 5,000.

    Returns:
        ComputeAssociationMatrixResult: `pairs` holds one entry per pair,
        strongest first by absolute value, with null values last. An entry
        gives the two columns and their types, the `method`, the `value`, the
        `pair_count` of complete rows it uses, the `direction` and, when the
        value is null, the `undefined_reason`. pearson, spearman and kendall
        run from -1 to 1, Cramér's V and the correlation ratio from 0, no
        association, to 1. Two numeric columns get numeric_method and two
        categorical ones categorical_method. A categorical and a numeric column
        get mixed_method, a correlation ratio, with direction x_to_y: it says
        how much the categorical x_column explains the numeric y_column, not the
        reverse. Complete rows have both values present, and finite in a numeric
        column. A pair's value can differ in its last digits, about 1e-15
        relative, with the other columns in the call; no pair differs by more
        than 1e-12.

        `columns` describes each paired column, keyed by name. `left_out_columns`
        lists the columns paired with no other, each with its `reason`: datetime,
        timedelta, unsupported_dtype, all_missing, no_finite_values,
        unhashable_values or too_many_categories. The record keeps the 100
        strongest pairs and adds pairs_total.

        A value that doesn't exist is null with its `undefined_reason`:
        too_few_rows, fewer than 3 complete rows; single_level or
        constant_column, a categorical column with one level or a numeric
        column with one value in them; or one_row_per_level, a categorical
        column with a level for every complete row, such as an ID, which
        would explain the other column perfectly and trivially. The limitations
        `too_few_values`, `zero_variance` and `one_row_per_level` name those
        pairs, each with its own reason.
        `float_precision` flags integers or decimals beyond ±2^53 read as float64, and
        `nan_counted_as_missing` NaN that a column stores as a value. There are
        no p-values or intervals.

        The finding is suspected:

        - `perfect_associations`: suspected; a pair has an absolute value of 1, within
          1e-9, so one column may duplicate or derive from the other.

    Statuses:
        completed: At least one pair has a value.
        budget_exceeded: The paired columns make more pairs than max_pairs, so
            none was computed; `status_reason` gives the count and the limit.
            Reason code: `too_many_pairs`.
        insufficient_data: The data has no rows, or no pair has a value, such
            as when every pair has fewer than 3 complete rows. Reason codes:
            `no_rows` and `no_values`.
        not_applicable: Fewer than two columns can be paired; left_out_columns
            says why the others can't. Reason code: `no_columns`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, 51, 29, 47, 38, 62, 45, 27],
        ...         "income": [52.5, 61.0, 48.25, 70.0, 55.0, 80.5, 58.0, 45.0],
        ...         "region": [
        ...             "north",
        ...             "south",
        ...             "north",
        ...             "east",
        ...             "south",
        ...             "east",
        ...             "east",
        ...             "north",
        ...         ],
        ...         "member": [True, False, True, True, False, True, False, False],
        ...         "joined": pd.to_datetime(["2024-01-05"] * 8),
        ...     }
        ... )
        >>> res = ae.compute_association_matrix(df)
        >>> print(res)
        compute_association_matrix: completed on 8 rows x 5 columns
        Pairs, each on its own complete rows: 1 numeric (pearson), 1 categorical (cramers_v_bias_corrected),
            4 mixed (correlation_ratio_bias_corrected, from the categorical x_column to the numeric y_column).
        Left out 1 column, see res.left_out_columns: 'joined' (datetime).
        Pairs (6 of 6 shown):
        x_column  y_column  method                             value  pair_count  direction
        age       income    pearson                           0.9429           8  symmetric
        region    age       correlation_ratio_bias_corrected  0.7738           8  x_to_y
        region    income    correlation_ratio_bias_corrected  0.7495           8  x_to_y
        region    member    cramers_v_bias_corrected          0.2357           8  symmetric
        member    age       correlation_ratio_bias_corrected       0           8  x_to_y
        member    income    correlation_ratio_bias_corrected       0           8  x_to_y
        Findings: none
        Limitations: none
        More: res.to_frame() has all 6 pairs; res.to_frame(table='matrix') has the wide 4 x 4 matrix; res.findings,
            res.limitations and res.to_json() have the rest.
        >>> top = res.pairs[0]
        >>> top.x_column, top.y_column, round(top.value, 4)
        ('age', 'income', 0.9429)
        >>> matrix = res.to_frame(table="matrix")
        >>> round(float(matrix.loc["region", "income"]), 4)
        0.7495

    See Also:
        profile_target: The classes and imbalance, or the statistics, of the column a
            model will predict, checked against the task.
        summarize_numeric: How the values of each numeric column are distributed.
        infer_column_semantics: Which columns hold numeric codes, identifiers or numbers
            stored as text.
    """
    selection = select_columns(_ANALYSIS, data, columns, _ASSOCIATION)
    categorical = _categorical_names(data, columns, categorical_columns, selection)
    row_count = len(data)
    left_out: list[LeftOutColumn] = []
    paired: list[_Paired] = []
    support: list[ColumnSupport] = []
    selected_names = {column.name for column in selection.columns}
    for label, dtype in data.dtypes.items():
        name = str(label)
        if columns is None and name not in selected_names:
            reason = _KIND_SKIPS[column_kind(dtype)]
            left_out.append(_left_out(name, str(dtype), reason, max_categories))
    for column in selection.columns:
        read = _read(data, column, column.name in categorical, max_categories)
        if isinstance(read, _Paired):
            paired.append(read)
            support.append(read.support(row_count))
        else:
            left_out.append(_left_out(column.name, column.dtype, read, max_categories))
            support.append(
                ColumnSupport.of(
                    column.name,
                    row_count,
                    _left_out_exclusions(data.iloc[:, column.position], read),
                )
            )
    if columns is None:
        order = {str(label): i for i, label in enumerate(data.columns)}
        left_out.sort(key=lambda s: order[s.column])
    pair_count = len(paired) * (len(paired) - 1) // 2
    weight = _rank_weight(numeric_method, row_count)
    numeric_pairs = sum(p.floats is not None for p in paired)
    numeric_pairs = numeric_pairs * (numeric_pairs - 1) // 2
    work = pair_count + numeric_pairs * (weight - 1)
    pairs: list[AssociationPair] = []
    status: Status = "completed"
    reason: StatusReason | None = None
    if row_count == 0:
        status, reason = "insufficient_data", NO_ROWS
    elif len(paired) < 2:
        status, reason = "not_applicable", _FEWER_THAN_TWO_COLUMNS
    elif work > max_pairs:
        status = "budget_exceeded"
        if weight == 1:
            needs = f"The {len(paired):,} paired columns make"
            narrow = "name fewer columns in columns"
        else:
            needs = (
                f"The {len(paired):,} paired columns make {pair_count:,} pairs, and "
                f"{numeric_method} ranks each of the {numeric_pairs:,} numeric "
                f"ones on {row_count:,} rows, which counts as {weight:,} pairs, so "
                "the call counts"
            )
            narrow = (
                "name fewer columns in columns, or set numeric_method to pearson, "
                "which counts each pair once"
            )
        reason = budget_exceeded_reason(
            code="too_many_pairs",
            needs=needs,
            required=work,
            unit=("pair", "pairs"),
            parameter="max_pairs",
            limit=max_pairs,
            consequence="no pair was computed",
            narrow=narrow,
        )
    else:
        pairs = _strongest_first(
            _pairs(paired, numeric_method, categorical_method, mixed_method)
        )
        if all(p.value is None for p in pairs):
            status, reason = "insufficient_data", _NO_DEFINED_PAIRS
    return ComputeAssociationMatrixResult.for_call(
        parameters={
            "columns": [column.name for column in selection.columns],
            "categorical_columns": list(categorical),
        },
        libraries=(
            ("scipy",)
            if any(p.method == "kendall" and p.value is not None for p in pairs)
            else ()
        ),
        status=status,
        status_reason=reason,
        support=Support(
            row_count=row_count,
            column_count=len(selection.columns),
            columns=support,
        ),
        findings=_perfect_associations(
            pairs, {p.entry.column: i for i, p in enumerate(paired)}
        ),
        limitations=[
            *read_limitations(
                {p.entry.column: p.numbers for p in paired if p.numbers is not None},
                _entry_path,
            ),
            *_null_limitations(pairs),
        ],
        pairs=pairs,
        columns={p.entry.column: p.entry for p in paired},
        left_out_columns=left_out,
    )


def _categorical_names(
    data: pd.DataFrame,
    columns: Sequence[str] | None,
    categorical_columns: Sequence[str] | None,
    selection: Selection,
) -> list[str]:
    """The checked names in categorical_columns, in the given order."""
    if categorical_columns is None:
        return []
    listed = select_columns(
        _ANALYSIS,
        data,
        categorical_columns,
        _CATEGORICAL,
        parameter="categorical_columns",
    )
    selected = {column.name for column in selection.columns}
    outside = [c.name for c in listed.columns if c.name not in selected]
    if columns is not None and outside:
        them = "it" if len(outside) == 1 else "them"
        raise InvalidArgumentError(
            f"{_ANALYSIS} got {join_labels(outside)} in categorical_columns, but "
            f"columns leaves {them} out. Add {them} to columns, or leave {them} out "
            "of categorical_columns."
        )
    return [column.name for column in listed.columns]


def _read(
    data: pd.DataFrame, column: Column, as_category: bool, max_categories: int
) -> "_Paired | LeftOutReason":
    """The column's values or codes, or why the pairs can't use it."""
    series = data.iloc[:, column.position]
    rows = len(series)
    if column.kind == "numeric" and not as_category:
        numbers = read_numeric(series)
        missing = int(np.isnan(numbers.values).sum())
        finite = int(np.isfinite(numbers.values).sum())
        if rows and missing == rows:
            return "all_missing"
        if rows and not finite:
            return "no_finite_values"
        return _Paired(
            entry=AssociationColumn(
                column=column.name,
                dtype=column.dtype,
                type="numeric",
                value_count=finite,
                missing_count=missing,
                infinite_count=rows - missing - finite,
                distinct_count=None,
            ),
            numbers=numbers,
            floats=numbers.values,
            codes=None,
        )
    is_number = column.kind == "numeric"
    if rows > PROBE_ROWS:
        # More levels in the first rows than allowed means more in all of them.
        probe = _codes(series.iloc[:PROBE_ROWS], numbers=is_number)
        if probe is not None and probe.levels > max_categories:
            return "too_many_categories"
    found = _codes(series, numbers=is_number)
    if found is None:
        return "all_missing" if bool(series.isna().all()) else "unhashable_values"
    if rows and not found.levels:
        return "all_missing"
    if found.levels > max_categories:
        return "too_many_categories"
    missing = int((found.codes < 0).sum())
    return _Paired(
        entry=AssociationColumn(
            column=column.name,
            dtype=column.dtype,
            type="categorical",
            value_count=rows - missing,
            missing_count=missing,
            infinite_count=0,
            distinct_count=found.levels,
        ),
        # Only the NaN stored as a value matters to the limitations here:
        # the values are compared as categories, not read as float64.
        numbers=ReadEffects(
            boolean=False,
            beyond_exact_integers=False,
            stored_nan_count=stored_nan_count(series),
        )
        if is_number
        else None,
        floats=None,
        codes=found.codes,
    )


def _entry_path(name: str) -> str:
    """The result path of a paired column's entry, such as "columns['age']"."""
    return f"columns[{name!r}]"


@dataclass(frozen=True)
class _Levels:
    """A column's codes per distinct value, -1 where missing, and the count."""

    codes: np.ndarray
    levels: int


def _codes(series: pd.Series, *, numbers: bool) -> _Levels | None:
    """Integer codes per distinct value, or None when the values can't be
    hashed. In a numeric column NaN is missing even where the dtype stores it
    as a value, as summarize_numeric counts it.
    """
    found = value_codes(series, unhashable="unsupported")
    if found is None:
        return None
    uniques = found.uniques
    codes = np.asarray(found.codes, dtype=np.int64)
    if numbers and len(uniques):
        levels = pd.Index(uniques).to_numpy(dtype="float64", na_value=np.nan)
        is_nan = np.isnan(levels)
        if is_nan.any():
            renumbered = np.where(is_nan, -1, np.cumsum(~is_nan) - 1)
            codes = np.where(codes >= 0, renumbered[codes], -1)
            return _Levels(codes, int((~is_nan).sum()))
    return _Levels(codes, len(uniques))


def _left_out_exclusions(
    series: pd.Series, reason: LeftOutReason
) -> dict[ExclusionReason, int]:
    """Why every value of a column left out of the pairs is excluded, and how many.

    A column with no finite value holds only missing and infinite values; the
    other reasons exclude every row for one reason.
    """
    rows = len(series)
    if not rows:
        return {}
    if reason == "all_missing":
        return {"missing": rows}
    if reason == "no_finite_values":
        missing = int(np.isnan(read_numeric(series).values).sum())
        return {
            key: count
            for key, count in (("missing", missing), ("infinite", rows - missing))
            if count
        }
    if reason == "unhashable_values":
        return {"unhashable_values": rows}
    return {"too_many_categories": rows}


def _left_out(
    name: str, dtype: str, reason: LeftOutReason, max_categories: int
) -> LeftOutColumn:
    return LeftOutColumn(
        column=name,
        dtype=dtype,
        reason=reason,
        message=_SKIP_MESSAGES[reason].format(max_categories=max_categories),
    )


def _pairs(
    paired: list[_Paired],
    numeric_method: NumericMethod,
    categorical_method: CategoricalMethod,
    mixed_method: MixedMethod,
) -> list[AssociationPair]:
    """Every pair of paired columns in the selected order, the categorical column
    first in a mixed pair."""
    numeric: list[np.ndarray] = []
    categorical: list[Codes] = []
    # Each paired column's index among the numeric or the categorical ones.
    index: list[int] = []
    for column in paired:
        if column.floats is not None:
            index.append(len(numeric))
            numeric.append(column.floats)
        else:
            assert column.codes is not None
            assert column.entry.distinct_count is not None
            index.append(len(categorical))
            categorical.append(Codes(column.codes, column.entry.distinct_count))
    measures = measure_pairs(
        numeric,
        categorical,
        numeric_method=numeric_method,
        bias_corrected=categorical_method == "cramers_v_bias_corrected",
        ratio_bias_corrected=mixed_method == "correlation_ratio_bias_corrected",
    )
    pairs = []
    for s, t in combinations(range(len(paired)), 2):
        first, second = paired[s], paired[t]
        kinds: tuple[AssociationType, AssociationType] = (
            first.entry.type,
            second.entry.type,
        )
        if kinds == ("numeric", "numeric"):
            measure = measures.numeric[index[s], index[t]]
            pairs.append(_association_pair(first, second, numeric_method, measure))
        elif kinds == ("categorical", "categorical"):
            measure = measures.categorical[index[s], index[t]]
            pairs.append(_association_pair(first, second, categorical_method, measure))
        elif kinds[0] == "categorical":
            measure = measures.mixed[index[s], index[t]]
            pairs.append(_association_pair(first, second, mixed_method, measure))
        else:
            measure = measures.mixed[index[t], index[s]]
            pairs.append(_association_pair(second, first, mixed_method, measure))
    return pairs


def _strongest_first(pairs: list[AssociationPair]) -> list[AssociationPair]:
    """By absolute value, descending, null values last; stable, so ties keep the
    selected column order."""
    return sorted(pairs, key=lambda p: (p.value is None, -abs(p.value or 0.0)))


def _pair_label(pair: AssociationPair) -> str:
    return f"({pair.x_column!r}, {pair.y_column!r})"


def _pair_path(pair: AssociationPair) -> str:
    """The result path of a pair's value, which names the pair by its columns."""
    return f"pairs[{pair.x_column!r}, {pair.y_column!r}].value"


# How far from 1 an absolute value may lie and still count as a perfect
# association: rounding leaves an exact relation a few ulps short of 1.
PERFECT_TOLERANCE = 1e-9


def _perfect_associations(
    pairs: list[AssociationPair], order: dict[str, int]
) -> list[Finding]:
    """The finding for pairs with an absolute value of 1, in column order.

    `order` gives each paired column's position in the selected order.
    """

    def place(i: int) -> tuple[int, int]:
        x, y = order[pairs[i].x_column], order[pairs[i].y_column]
        return min(x, y), max(x, y)

    positions = sorted(
        (
            i
            for i, p in enumerate(pairs)
            if p.value is not None and abs(p.value) >= 1 - PERFECT_TOLERANCE
        ),
        key=place,
    )
    if not positions:
        return []
    columns = sorted(
        {name for i in positions for name in (pairs[i].x_column, pairs[i].y_column)},
        key=order.__getitem__,
    )
    are = "pair is" if len(positions) == 1 else "pairs are"
    return [
        Finding(
            code="perfect_associations",
            claim=(
                f"{len(positions):,} {are} perfectly associated, an absolute value "
                "of 1, so one column may duplicate or derive from the other: "
                f"{join_items([_pair_label(pairs[i]) for i in positions])}."
            ),
            assessment="suspected",
            columns=columns,
            fields=[_pair_path(pairs[i]) for i in positions],
        )
    ]


# The limitation behind each reason a pair's value is null, and what it says.
_NULL_LIMITATIONS: dict[LimitationCode, tuple[tuple[UndefinedReason, ...], str]] = {
    "too_few_values": (
        ("too_few_rows",),
        "{pairs_have} too few complete rows for a value, so it is null: "
        "{names}. Every measure needs at least 3.",
    ),
    "one_row_per_level": (
        ("one_row_per_level",),
        "{pairs_have} a categorical column with a level for every complete row, "
        "such as an ID, which would explain the other column perfectly and "
        "trivially, so the value is null: {names}.",
    ),
    "zero_variance": (
        ("single_level", "constant_column"),
        "{pairs_have} a categorical column with a single level or a numeric "
        "column with a single value in their complete rows, so the association "
        "is undefined and null: {names}.",
    ),
}


def _null_limitations(pairs: list[AssociationPair]) -> list[Limitation]:
    """One limitation per reason, naming every pair whose value it makes null."""
    limitations = []
    for code, (reasons, template) in _NULL_LIMITATIONS.items():
        positions = [i for i, p in enumerate(pairs) if p.undefined_reason in reasons]
        if not positions:
            continue
        limitations.append(
            Limitation(
                code=code,
                reason=template.format(
                    pairs_have=plural(len(positions), "pair has", "pairs have"),
                    names=join_items([_pair_label(pairs[i]) for i in positions]),
                ),
                fields=[_pair_path(pairs[i]) for i in positions],
            )
        )
    return limitations


def _association_pair(
    x: _Paired, y: _Paired, method: AssociationMethod, measure: Measure
) -> AssociationPair:
    return AssociationPair(
        x_column=x.entry.column,
        y_column=y.entry.column,
        x_type=x.entry.type,
        y_type=y.entry.type,
        method=method,
        value=measure.value,
        pair_count=measure.pair_count,
        direction="x_to_y" if method in _MIXED_METHODS else "symmetric",
        undefined_reason=measure.undefined,
    )
