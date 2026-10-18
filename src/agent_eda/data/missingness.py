"""Analyses of missing values and the stand-ins that mean missing."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from agent_eda._columns import ALL_KINDS, Column, ColumnRule, select_columns
from agent_eda._numeric import stored_nan_count
from agent_eda._registry import analysis
from agent_eda._run import data_shape, execution, overall_status
from agent_eda._sentinels import (
    SentinelValue,
    declared_values,
    json_value,
    matches,
)
from agent_eda._text import join_labels, plural
from agent_eda.results import (
    ColumnSupport,
    DeclaredSentinelValue,
    Finding,
    Limitation,
    ProfileMissingValuesColumn,
    ProfileMissingValuesResult,
    ProfileMissingValuesRowCount,
    StatusReason,
    Support,
)

# Every dtype can have missing values.
_EVERY_COLUMN = ColumnRule(
    default=ALL_KINDS,
    default_text="every column",
    accepted=ALL_KINDS,
    accepted_text="columns of every dtype",
    fixes={},
)


@analysis(
    statuses=("completed", "insufficient_data", "not_applicable", "failed"),
    finding_codes=("all_missing_columns", "sentinel_values_present"),
    tags=("missing values", "completeness", "per column", "per row", "gaps"),
)
def profile_missing_values(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sentinel_values: Mapping[str, Sequence[SentinelValue]] | None = None,
) -> ProfileMissingValuesResult:
    """How much data is missing, and where?

    When to use:
        Call it after `describe_dataset` to learn how complete the data really
        is, per column and per row. It counts the values pandas treats as
        missing (None, NaN, NaT and pd.NA) together with the sentinel values you
        declare, stand-ins for missing values such as -999, 'N/A' or an empty
        string, and counts no value twice. Shares are over all rows. Per row,
        it counts how many selected values each row misses, which shows whether
        gaps cluster in some rows, and how many rows are complete. Nothing
        counts as a sentinel value unless you declare it. Missing values in
        columns you declare non-nullable are violations in
        `validate_column_types`. For how the values of each numeric column are
        distributed, call `summarize_numeric`; `list_analyses` lists the other
        analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to profile, in the
            order to report them: a list, a tuple, a pandas Index such as
            df.columns[:3], or a single name. A label that isn't a string can
            be given as itself, such as 0, or as its string form. Columns of
            every dtype are accepted. None, the default, selects every column,
            in DataFrame order.
        sentinel_values (Mapping[str, Sequence[bool | int | float | str | datetime | date | timedelta]] | None):
            The sentinel values to count as missing, by column name, such as
            {'age': [-999], 'city': ['N/A', '']}: one value or a list of values
            per column, each a bool, int, float, str, datetime (pd.Timestamp
            too), date or timedelta (pd.Timedelta too). A value matches exactly:
            text with its case and surrounding spaces, numbers by value (-999
            matches -999.0 but never True), datetimes by the instant they name.
            Every column named must be selected. A value that can never match
            its column raises errors.InvalidArgumentError naming the value you
            likely meant, such as -999 for the text '-999' in an integer
            column, and so do NaN, NaT and infinite values. None, the default,
            declares none: then sentinel_count is null and only native missing
            values count, which the result records as a limitation.

    Returns:
        ProfileMissingValuesResult: `columns` maps each selected column name to
        its entry, such as res.columns["age"], which holds:

        - `native_missing_count`, the values pandas isna() counts as missing:
          None, NaN, NaT and pd.NA.
        - `sentinel_count`, the values equal to a declared sentinel value;
          null when the column has none declared.
        - `missing_count`, both together, and `missing_share`, that count over
          all rows.
        - `sentinel_values`, each declared value with its count.

        For all selected columns together, `missing_count` and `missing_share`
        count every selected value, `complete_row_count` and
        `complete_row_share` the rows that miss none, and `row_distribution`
        how many rows miss each number of values, fewest first. Shares lie
        between 0 and 1, and a share without rows is null. The printed table
        shows the columns with the most missing values first. to_frame()
        returns the per-column table in the selected order, and
        to_frame(table='row_distribution') the rows per number of missing
        values.

        Limitations: `sentinel_values_not_declared` names the columns whose
        sentinel_count is null; `nan_not_counted_as_missing` the columns that
        store NaN as a value, which isna() doesn't count; `too_few_values` the
        shares that are null for want of rows or columns.

        The findings:

        - `all_missing_columns`: suspected; every value of a column is missing
          or a declared sentinel value. Data without rows has none.
        - `sentinel_values_present`: descriptive; declared sentinel values
          occur in a column, where they count as missing.

    Statuses:
        completed: The data has rows and columns, and every selected column was
            profiled.
        insufficient_data: The data has no rows; every count is 0 and every
            share is null.
        not_applicable: The data has rows but no columns.
        failed: An anticipated numerical failure stopped the analysis, and
            `status_reason` gives its code and message. None is known today.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, -999, 29, None, 51],
        ...         "city": ["Lyon", "N/A", "", "Oslo", None],
        ...         "score": [0.5, 0.75, None, None, 0.25],
        ...     }
        ... )
        >>> res = ae.profile_missing_values(
        ...     df, sentinel_values={"age": [-999], "city": ["N/A", ""]}
        ... )
        >>> print(res)
        profile_missing_values: completed on 5 rows x 3 columns
        Missing: 7 of 15 selected values (46.67%), counting declared sentinel values. Complete rows: 1 of 5 (20%).
        Missing values per row: 0 in 1 row, 1 in 1 row and 2 in 3 rows.
        Columns, most missing first (3 of 3 shown):
        column  native_missing_count  sentinel_count  missing_count  missing_share
        city                       1               2              3            0.6
        age                        1               1              2            0.4
        score                      2            null              2            0.4
        Findings (1 of 1 shown):
        - descriptive sentinel_values_present: Declared sentinel values occur in 2 columns: 'age' and 'city'.
        Limitations (1 of 1 shown):
        - sentinel_values_not_declared: No sentinel values were declared for 1 column, so its sentinel_count is ...
        More: res.to_frame() has all 3 columns; res.to_frame(table='row_distribution') has the rows per number of missing
            values; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["city"].missing_count
        3
        >>> res.complete_row_count
        1

    See Also:
        describe_dataset: Dtypes, missing and distinct counts and memory of
            every column, whatever its dtype.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every
            numeric column.
        validate_column_types: The values of each declared column measured
            against its logical type, with missing values in a column declared
            non-nullable as violations.
        list_analyses: Every analysis, with the question each one answers.
    """
    selection = select_columns("profile_missing_values", data, columns, _EVERY_COLUMN)
    declared = declared_values(
        "profile_missing_values", data, selection.columns, sentinel_values
    )
    row_count = len(data)
    entries = {}
    # Columns that store NaN as a value, which isna() doesn't count.
    stored_nan: list[Column] = []
    # How many selected values each row misses.
    per_row = np.zeros(row_count, dtype=np.int64)
    for column in selection.columns:
        series = data.iloc[:, column.position]
        native = np.asarray(series.isna(), dtype=bool)
        native_count = int(native.sum())
        if stored_nan_count(series):
            stored_nan.append(column)
        # A value counts once, however many declared values it equals.
        missing_mask, sentinel, counted = native, None, []
        values = declared.get(column.name)
        if values is not None:
            found = np.zeros(row_count, dtype=bool)
            for value in values:
                match = matches(series, value, native)
                found |= match
                counted.append(
                    DeclaredSentinelValue(
                        value=json_value(value), count=int(match.sum())
                    )
                )
            sentinel = int(found.sum())
            missing_mask = native | found
        per_row += missing_mask
        missing = native_count + (sentinel or 0)
        entries[column.name] = ProfileMissingValuesColumn(
            column=column.name,
            status="completed" if row_count else "insufficient_data",
            status_reason=None if row_count else _NO_ROWS,
            native_missing_count=native_count,
            sentinel_count=sentinel,
            missing_count=missing,
            missing_share=missing / row_count if row_count else None,
            sentinel_values=counted,
        )
    rows_by_count = np.bincount(per_row)
    distribution = [
        ProfileMissingValuesRowCount(
            missing_count=count,
            row_count=int(rows),
            row_share=int(rows) / row_count,
        )
        for count, rows in enumerate(rows_by_count.tolist())
        if rows
    ]
    complete = int(rows_by_count[0]) if rows_by_count.size else 0
    missing_total = sum(entry.missing_count for entry in entries.values())
    value_count = row_count * len(entries)
    status, status_reason = overall_status(
        row_count, list(entries.values()), no_columns=_NO_COLUMNS
    )
    return ProfileMissingValuesResult(
        analysis="profile_missing_values",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": [column.name for column in selection.columns],
            "sentinel_values": None
            if sentinel_values is None
            else {
                name: [json_value(value) for value in values]
                for name, values in sentinel_values.items()
            },
        },
        support=Support(
            row_count=row_count,
            column_count=len(selection.columns),
            columns=[
                ColumnSupport(
                    column=name,
                    eligible_count=row_count,
                    evaluated_count=row_count,
                    excluded_count=0,
                    excluded_by_reason={},
                )
                for name in entries
            ],
        ),
        findings=[
            *_all_missing_columns(entries, row_count, any_declared=bool(declared)),
            *_sentinel_values_present(entries),
        ],
        limitations=[
            *_nan_not_counted(stored_nan),
            *_not_declared(entries, any_declared=bool(declared)),
            *_null_shares(entries, row_count),
        ],
        execution=execution(("numpy", "pandas")),
        columns=entries,
        missing_count=missing_total,
        missing_share=missing_total / value_count if value_count else None,
        complete_row_count=complete,
        complete_row_share=complete / row_count if row_count else None,
        row_distribution=distribution,
    )


_NO_COLUMNS = StatusReason(code="no_columns", message="The data has no columns.")
_NO_ROWS = StatusReason(code="no_rows", message="The data has no rows.")


def _nan_not_counted(columns: list[Column]) -> list[Limitation]:
    if not columns:
        return []
    store = plural(len(columns), "column stores", "columns store")
    names = join_labels([column.name for column in columns])
    return [
        Limitation(
            code="nan_not_counted_as_missing",
            reason=(
                f"{store} NaN as a value, which pandas isna() doesn't count as "
                f"missing, so native_missing_count leaves it out: {names}. To count "
                "it, read the column as float64 first, such as "
                f"df[{columns[0].label!r}].astype('float64')."
            ),
            fields=[
                f"{_entry_path(column.name)}.native_missing_count" for column in columns
            ],
        )
    ]


def _null_shares(
    entries: dict[str, ProfileMissingValuesColumn], row_count: int
) -> list[Limitation]:
    """Why shares are null: no rows to divide by, or no selected values."""
    if row_count == 0:
        fields = [
            "missing_share",
            "complete_row_share",
            *(f"{_entry_path(name)}.missing_share" for name in entries),
        ]
        reason = "The data has no rows, so every share is null."
    elif not entries:
        fields = ["missing_share"]
        reason = "The data has no columns, so missing_share is null."
    else:
        return []
    return [Limitation(code="too_few_values", reason=reason, fields=fields)]


def _entry_path(name: str) -> str:
    """The result path of a column's entry, such as "columns['age']"."""
    return f"columns[{name!r}]"


def _all_missing_columns(
    entries: dict[str, ProfileMissingValuesColumn],
    row_count: int,
    *,
    any_declared: bool,
) -> list[Finding]:
    names = [
        name
        for name, entry in entries.items()
        if row_count and entry.missing_count == row_count
    ]
    if not names:
        return []
    hold = plural(len(names), "column holds", "columns hold")
    counting = ", counting declared sentinel values" if any_declared else ""
    return [
        Finding(
            code="all_missing_columns",
            claim=f"{hold} only missing values{counting}: {join_labels(names)}.",
            assessment="suspected",
            columns=names,
            fields=[f"{_entry_path(name)}.missing_count" for name in names],
        )
    ]


def _sentinel_values_present(
    entries: dict[str, ProfileMissingValuesColumn],
) -> list[Finding]:
    names = [name for name, entry in entries.items() if entry.sentinel_count]
    if not names:
        return []
    columns = plural(len(names), "column", "columns")
    return [
        Finding(
            code="sentinel_values_present",
            claim=f"Declared sentinel values occur in {columns}: {join_labels(names)}.",
            assessment="descriptive",
            columns=names,
            fields=[f"{_entry_path(name)}.sentinel_count" for name in names],
        )
    ]


def _not_declared(
    entries: dict[str, ProfileMissingValuesColumn], *, any_declared: bool
) -> list[Limitation]:
    names = [name for name, entry in entries.items() if entry.sentinel_count is None]
    if not names:
        return []
    if any_declared:
        columns = plural(len(names), "column", "columns")
        its = "its" if len(names) == 1 else "their"
        reason = (
            f"No sentinel values were declared for {columns}, so {its} "
            f"sentinel_count is null and {its} missing_count counts only native "
            f"missing values: {join_labels(names)}."
        )
    else:
        reason = (
            "No sentinel values were declared, so sentinel_count is null and "
            "missing_count counts only native missing values (None, NaN, NaT and "
            "pd.NA). Declare stand-ins for missing values, such as -999 or 'N/A', "
            "in sentinel_values."
        )
    return [
        Limitation(
            code="sentinel_values_not_declared",
            reason=reason,
            fields=[f"{_entry_path(name)}.sentinel_count" for name in names],
        )
    ]
