"""Analyses of missing values and the stand-ins that mean missing."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from agent_eda._columns import ALL_KINDS, Column, ColumnRule, select_columns
from agent_eda._declared_values import (
    DeclaredValue,
    JsonSentinel,
    SentinelValue,
    declared_values,
    holds,
    json_value,
    matches,
)
from agent_eda._levels import stored_label, type_name
from agent_eda._masking import mask_values, resolve_masking
from agent_eda._numeric import stored_nan_limitations
from agent_eda._registry import analysis
from agent_eda._result_base import (
    ColumnSupport,
    Finding,
    Limitation,
    Status,
    StatusReason,
    Support,
)
from agent_eda._row_ids import ambiguous_row_ids, read_row_ids
from agent_eda._run import NO_ROWS, overall_status
from agent_eda._text import join_labels, plural
from agent_eda._values import stored_nan_mask
from agent_eda._written_form import record_scalar, row_id_list
from agent_eda.data._missingness_results import (
    DeclaredSentinelValue,
    DetectMissingSentinelsResult,
    ProfileMissingValuesColumn,
    ProfileMissingValuesResult,
    ProfileMissingValuesRowCount,
    SentinelCandidate,
    SentinelEvidence,
)
from agent_eda.data._sentinel_rules import read_by_rules, suspects
from agent_eda.project import Project

# Every dtype can have missing values.
_EVERY_COLUMN = ColumnRule(
    default=ALL_KINDS,
    default_text="every column",
    accepted=ALL_KINDS,
    accepted_text="columns of every dtype",
    fixes={},
)


@analysis(
    statuses=("completed", "insufficient_data", "not_applicable"),
    finding_codes=("all_missing_columns", "sentinel_values_present"),
    tags=(
        *("missing values", "completeness", "per column", "per row", "gaps"),
        *("nulls", "null counts", "na counts", "nan", "missingness", "missing data"),
    ),
    masking=True,
)
def profile_missing_values(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sentinel_values: Mapping[str, Sequence[SentinelValue]] | None = None,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> ProfileMissingValuesResult:
    """How much data is missing, and where?

    When to use:
        Call it after `describe_dataset` to learn how complete the data really is, per
        column and per row. It counts the values pandas treats as missing (None, NaN,
        NaT and NA) together with the sentinel values you declare, stand-ins for missing
        values such as -999, 'N/A' or an empty string, and counts no value twice. Shares
        are over all rows. Per row, it counts how many selected values each row misses,
        which shows whether gaps cluster in some rows, and how many rows are complete.
        Nothing counts as a sentinel value unless you declare it: to find the values
        that look like stand-ins, call `detect_missing_sentinels` first. Missing values
        in columns you declare non-nullable are violations in `validate_column_types`.
        Declared sentinel values are raw values: for personal data, declare the column
        sensitive in a project and switch masking on.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to profile, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself, such as 0, or as its string form. Columns of every
            dtype are accepted. None, the default, selects every column, in DataFrame
            order.
        sentinel_values (Mapping[str, Sequence[bool | int | float | str | datetime | date | timedelta]] | None):
            The sentinel values to count as missing, by column name, such as {'age':
            [-999], 'city': ['N/A', '']}: one value or a list of values per column, each
            a bool, int, float, str, datetime (pandas Timestamp too), date or timedelta
            (pandas Timedelta too). A value matches exactly: text with its case and
            surrounding spaces, numbers by value (-999 matches -999.0 but never true),
            datetimes by the instant they name. Over MCP, write a datetime, date or
            timedelta as its ISO 8601 text, such as '1900-01-01T00:00:00', '1900-01-01'
            or 'PT0S': it is read as one for a column that holds such values and no
            text. List each value once: -999 and -999.0 are one value, while true and 1
            are two. For a column of decimals, text such as '12345678901234567.25'
            declares an exact decimal. Every column named must be selected. A value that
            can never match its column is rejected, naming the value you likely meant,
            such as -999 for the text '-999' in an integer column; so are NaN, NaT and
            infinite values. None, the default, declares none: then sentinel_count is
            null and only native missing values count, with a limitation.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces each declared sentinel value
            of a sensitive column, in parameters and in the columns table, with
            a numbered placeholder, '<masked 1>' for the first value declared
            and so on, and needs a project that declares sensitive columns;
            False shows the values as declared. None, the default, follows the
            project's data_policy.mask_sensitive, and masks nothing without a
            project. Counts and shares are never masked. Masking is a guardrail
            against accidental exposure, not anonymization or access control.

    Returns:
        ProfileMissingValuesResult: `columns` maps each selected column name to
        its entry: `native_missing_count` (None, NaN, NaT and NA, NaN that a
        pyarrow-backed or nullable float column stores as a value included),
        `sentinel_count` (null when the column has no declared sentinel value),
        `missing_count` and `missing_share` of both together, and
        `sentinel_values`, each declared value with its `value_type` and count,
        which tell apart values that print the same, such as the date 2020-01-01
        and the text '2020-01-01'.

        For all selected columns together, `missing_count` and `missing_share`
        count every selected value, `complete_row_count` and `complete_row_share`
        the rows that miss none, and `row_distribution` how many rows miss each
        number of values, fewest first. A share without rows is null.

        Limitations: `sentinel_values_not_declared` names the columns whose
        sentinel_count is null; `nan_counted_as_missing` the columns that
        store NaN as a value, which `isna` doesn't count and this analysis does;
        `too_few_values` the shares that are null for want of rows or columns.

        The findings:

        - `all_missing_columns`: suspected; every value of a column is missing
          or a declared sentinel value. Data without rows has none.
        - `sentinel_values_present`: descriptive; declared sentinel values
          occur in a column, where they count as missing.

    Statuses:
        completed: The data has rows and columns, and every selected column was
            profiled.
        insufficient_data: The data has no rows; every count is 0 and every
            share is null. Reason code: `no_rows`.
        not_applicable: The data has rows but no columns. Reason code:
            `no_columns`.

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
        Missing: 7 of 15 selected values (missing_share 0.4667), counting declared sentinel values. Complete rows: 1 of 5
            (complete_row_share 0.2).
        Missing values per row: 0 in 1 row, 1 in 1 row and 2 in 3 rows.
        Columns, most missing first (3 of 3 shown):
        column  native_missing_count  sentinel_count  missing_count  missing_share
        city                       1               2              3            0.6
        age                        1               1              2            0.4
        score                      2            null              2            0.4
        Findings (1 of 1 shown):
        - descriptive sentinel_values_present: Declared sentinel values occur in 2 columns: 'age' and 'city'.
        Limitations (1 of 1 shown):
        - sentinel_values_not_declared: No sentinel values were declared for 1 column, so its sentinel_count is null and its
          missing_count counts only native missing values: 'score'.
        More: res.to_frame() has all 3 columns; res.to_frame(table='row_distribution') has the rows per number of missing
            values; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["city"].missing_count
        3
        >>> res.complete_row_count
        1

    See Also:
        detect_missing_sentinels: Values that look like stand-ins for missing ones, such
            as -999 or 'N/A', to declare as sentinel values.
        describe_dataset: Dtypes, missing and distinct counts and memory of every
            column, whatever its dtype.
        validate_column_types: The values of each declared column measured against its
            logical type, with missing values in a column declared non-nullable as
            violations.
    """
    masking = resolve_masking("profile_missing_values", project, mask_sensitive)
    selection = select_columns("profile_missing_values", data, columns, _EVERY_COLUMN)
    declared = declared_values(
        "profile_missing_values",
        data,
        selection.columns,
        sentinel_values,
        hidden=masking.hidden(sentinel_values),
    )
    row_count = len(data)
    # The declared values as the result shows them: a sensitive column's as
    # placeholders when masking is on, numbered in the order declared.
    shown = {
        name: mask_values(values)
        if masking.masks(name)
        else [json_value(v) for v in values]
        for name, values in declared.items()
    }
    entries = {}
    # Columns that store NaN as a value, which isna() doesn't count.
    stored_nan: list[str] = []
    # How many selected values each row misses.
    per_row = np.zeros(row_count, dtype=np.int64)
    for column in selection.columns:
        series = data.iloc[:, column.position]
        native = np.asarray(series.isna(), dtype=bool)
        stored = stored_nan_mask(series)
        if stored is not None:
            # NaN stored as a value is missing here, though isna() says otherwise.
            native = native | stored
            stored_nan.append(column.name)
        native_count = int(native.sum())
        # A value counts once, however many declared values it equals.
        missing_mask, sentinel, counted = native, None, []
        values = declared.get(column.name)
        if values is not None:
            found = np.zeros(row_count, dtype=bool)
            for value, shown_value in zip(values, shown[column.name], strict=True):
                match = matches(series, value, native)
                found |= match
                counted.append(
                    DeclaredSentinelValue(
                        value=cast("JsonSentinel", shown_value),
                        value_type=type_name(value),
                        count=int(match.sum()),
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
    return ProfileMissingValuesResult.for_call(
        parameters={
            "columns": [column.name for column in selection.columns],
            "sentinel_values": None
            if sentinel_values is None
            else {name: shown.get(name, []) for name in sentinel_values},
        },
        status=status,
        status_reason=status_reason,
        support=Support(
            row_count=row_count,
            column_count=len(selection.columns),
            columns=[ColumnSupport.of(name, row_count) for name in entries],
        ),
        findings=[
            *_all_missing_columns(entries, row_count, any_declared=bool(declared)),
            *_sentinel_values_present(entries),
        ],
        limitations=[
            *stored_nan_limitations(
                stored_nan,
                [f"{_entry_path(name)}.native_missing_count" for name in stored_nan],
                "native_missing_count counts it",
            ),
            *_not_declared(entries, any_declared=bool(declared)),
            *_null_shares(entries, row_count),
        ],
        masking=masking.record(declared, data),
        columns=entries,
        missing_count=missing_total,
        missing_share=missing_total / value_count if value_count else None,
        complete_row_count=complete,
        complete_row_share=complete / row_count if row_count else None,
        row_distribution=distribution,
    )


_NO_COLUMNS = StatusReason(code="no_columns", message="The data has no columns.")
_NO_ROWS = StatusReason(code="no_rows", message="The data has no rows.")


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
            "missing_count counts only native missing values. Declare stand-ins for "
            "missing values, such as -999 or 'N/A', in sentinel_values."
        )
    return [
        Limitation(
            code="sentinel_values_not_declared",
            reason=reason,
            fields=[f"{_entry_path(name)}.sentinel_count" for name in names],
        )
    ]


# Sentinel values can hide in a column of any dtype.
_ANY_COLUMN = ColumnRule(
    default=ALL_KINDS,
    default_text="every column except id_column",
    accepted=ALL_KINDS,
    accepted_text="columns of every dtype",
    fixes={},
)


@analysis(
    statuses=("completed", "insufficient_data", "not_applicable"),
    finding_codes=("sentinel_values_present", "suspected_sentinel_values"),
    tags=(
        "sentinel",
        "stand-ins",
        "placeholder",
        "-999",
        "missing codes",
        "magic number",
    ),
    masking=True,
)
def detect_missing_sentinels(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    sentinel_values: Mapping[str, Sequence[SentinelValue]] | None = None,
    id_column: str | None = None,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> DetectMissingSentinelsResult:
    """Which values look like stand-ins for missing data?

    When to use:
        Call it after `profile_missing_values` to find values that may stand in for
        missing ones, such as -999, 'N/A', empty text or 1900-01-01, before they distort
        a summary or a model. Fixed rules pick the candidates: values from a built-in
        list, and a number that repeats at a column's minimum or maximum, far from all
        the others. It only suspects values and never changes the data. Declare the
        values you confirm in sentinel_values: they come back as declared, apart from
        the suspected ones, and `profile_missing_values` counts them as missing.
        Candidate values and row IDs are raw values, so when a column holds personal
        data, declare it sensitive in a project and switch masking on. Index labels are
        never masked: when the index holds sensitive values, pass a non-sensitive
        id_column. For which values occur in a column and how often, call
        `summarize_categorical`; for the extremes and spread of numeric columns,
        `summarize_numeric`.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to check, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself, such as 0, or as its string form. Columns of every
            dtype are accepted. None, the default, checks every column except id_column,
            in DataFrame order.
        sentinel_values (Mapping[str, Sequence[bool | int | float | str | datetime | date | timedelta]] | None):
            The sentinel values you declare, by column name, such as {'age': [-999],
            'city': ['N/A', '']}, as profile_missing_values takes them: one value or a
            list of values per column, each a bool, int, float, str, datetime (pandas
            Timestamp too), date or timedelta (pandas Timedelta too). A value matches
            exactly: text with its case and surrounding spaces, numbers by value (-999
            matches -999.0 but never true), datetimes by the instant they name. Over
            MCP, write a datetime, date or timedelta as its ISO 8601 text, such as
            '1900-01-01T00:00:00', '1900-01-01' or 'PT0S': it is read as one for a
            column that holds such values and no text. List each value once: -999 and
            -999.0 are one value, while true and 1 are two. For a column of decimals,
            text such as '12345678901234567.25' declares an exact decimal. Every column
            named must be checked. A value that can never match its column is rejected,
            naming the value you likely meant, such as -999 for the text '-999' in an
            integer column; so are NaN, NaT and infinite values. The declared values a
            column holds come back with confirmation 'declared'. None, the default,
            declares none.
        id_column (str | None): Name of a column whose values identify rows,
            such as an order number; row IDs are then its values, and the default
            columns leave it out of the check. Its values must be hashable, such as
            numbers or strings; a column of lists raises. A label that isn't a string
            can be given as itself or as its string form. None, the default, uses the
            index labels as row IDs.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces each value of a sensitive
            column that the result shows, its candidate values, its declared
            values in parameters and its values as row IDs when it is
            id_column, with a numbered placeholder, '<masked 1>' for the first
            value in result order and so on, equal values alike, and needs a
            project that declares sensitive columns; False shows the values as
            stored. None, the default, follows the project's
            data_policy.mask_sensitive, and masks nothing without a project.
            Index labels are never masked. Masking is a guardrail against
            accidental exposure, not anonymization or access control.

    Returns:
        DetectMissingSentinelsResult: `candidates` lists the values that may
        stand in for missing ones, column by column in the order checked: first
        the declared values a column holds, in the order given, then the
        suspected ones, most frequent first. Each has its `column`, `value`,
        `count` and `share` of all rows, its `evidence` codes, its
        `confirmation` (declared or suspected) and the `row_ids` of its first
        10 rows. Row IDs are index labels, or id_column values when given. The
        record keeps the first 100 candidates and adds candidates_total.
        `columns_without_rules` names the checked columns whose values
        no rule reads, such as booleans and durations, where only declared values
        were looked for. NaN that a pyarrow-backed or nullable float column
        stores as a value is missing, as in every analysis, and the
        `nan_counted_as_missing` limitation names such columns.

        The evidence codes, and the rules behind them:

        - `declared`: the value is declared in sentinel_values.
        - `builtin_list`: the value is on the built-in list: the numbers -9,
          -99, -999, -9999, 99, 999, 9999, 99999 and 999999 when they are the
          column's minimum or maximum, in a text column among the texts that
          parse as numbers, such as '-999'; the text '', whitespace only, 'NA',
          'N/A', 'NULL', 'None', 'nan', '-', '?', 'unknown', 'missing' and
          pandas' other default NA strings '#N/A', '#N/A N/A', '#NA', '<NA>',
          '-NaN', '-nan', '1.#IND', '1.#QNAN', '-1.#IND' and '-1.#QNAN',
          compared case-insensitively and after stripping surrounding
          whitespace, so ' N/A', 'N/A ' and 'N/A\\n' match; each raw form
          that occurs is its own candidate; and the dates 1900-01-01,
          1970-01-01, 2099-12-31 and 9999-12-31, as dates or as datetimes at
          midnight in their own time zone or in UTC.
        - `isolated_extreme`: a number, or number text in a text column, that
          occurs more than once as the column's minimum or maximum and lies
          farther from the nearest other value than the other values span;
          they must hold two distinct values, so a column of 0 and 1 has none.
          Number text is read by value: '-999' and '-999.0' are one number and
          two candidates.

        The rules read the finite values that are neither missing nor
        declared, and a column of Python objects value by value: numbers as
        numbers, text as text, dates as dates. A value that no rule matches,
        such as a legitimate 0, a negative number or a rare code, is never a
        candidate. When index labels or id_column values repeat, the
        `ambiguous_row_ids` limitation names the candidates whose row IDs
        repeat.

        The findings:

        - `sentinel_values_present`: descriptive; declared sentinel values
          occur in a column.
        - `suspected_sentinel_values`: suspected; the built-in rules found
          values that may stand in for missing ones, which stay valid values
          until you declare them.

    Statuses:
        completed: The data has rows and a column to check, whether or not a
            candidate turns up.
        insufficient_data: The data has no rows, so there is no value to
            check. Reason code: `no_rows`.
        not_applicable: There is no column to check: the data has no columns,
            or only id_column, which the default columns leave out. Reason
            code: `no_columns`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [34, -999, 29, -999, 51, 0],
        ...         "city": ["Lyon", "N/A", "", "Oslo", "n/a", "Rome"],
        ...         "visits": [3, 0, 5, 0, 4, 2],
        ...     }
        ... )
        >>> res = ae.detect_missing_sentinels(df)
        >>> print(res)
        detect_missing_sentinels: completed on 6 rows x 3 columns
        Checked 3 columns: 4 suspected candidates. Shares are of all rows; row IDs are index labels.
        Suspected values stay valid until declared in sentinel_values, which profile_missing_values counts as missing.
        Candidates, most frequent first (4 of 4 shown):
        column  value  count   share  confirmation  evidence                        row_ids
        age     -999       2  0.3333  suspected     builtin_list, isolated_extreme  1 and 3
        city    'N/A'      1  0.1667  suspected     builtin_list                    1
        city    ''         1  0.1667  suspected     builtin_list                    2
        city    'n/a'      1  0.1667  suspected     builtin_list                    4
        Findings (1 of 1 shown):
        - suspected suspected_sentinel_values: 4 values in 2 columns may stand in for missing values: 'age' and 'city'.
        Limitations (1 of 1 shown):
        - sentinel_values_not_declared: No sentinel values were declared, so every candidate is suspected, not confirmed.
        More: res.to_frame() has all 4 candidates; res.findings, res.limitations and res.to_json() have the rest.
        >>> declared = ae.detect_missing_sentinels(df, sentinel_values={"age": [-999]})
        >>> [
        ...     (c.value, c.confirmation)
        ...     for c in declared.candidates
        ...     if c.column == "age"
        ... ]
        [(-999, 'declared')]
        >>> sentinels = res.sentinel_values()
        >>> sentinels
        {'age': [-999], 'city': ['N/A', '', 'n/a']}
        >>> ae.profile_missing_values(df, sentinel_values=sentinels).columns[
        ...     "age"
        ... ].sentinel_count
        2

    See Also:
        profile_missing_values: Missing values per column and per row, counting declared
            sentinel values.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every numeric
            column, which stand-ins such as -999 distort.
        summarize_categorical: Each level's count and share of every text, categorical
            and boolean column, where stand-ins show up as levels.
    """
    masking = resolve_masking("detect_missing_sentinels", project, mask_sensitive)
    row_ids = read_row_ids("detect_missing_sentinels", data, id_column)
    selection = select_columns(
        "detect_missing_sentinels",
        data,
        columns,
        _ANY_COLUMN,
        id_column=row_ids.id_column,
    )
    declared = declared_values(
        "detect_missing_sentinels",
        data,
        selection.columns,
        sentinel_values,
        hidden=masking.hidden(sentinel_values),
        # Only a call without columns leaves the id_column out by itself.
        id_column=row_ids.id_column if columns is None else None,
    )
    row_count = len(data)
    checked = {
        column.name: _check_column(
            data.iloc[:, column.position], column, declared.get(column.name, [])
        )
        for column in selection.columns
    }
    found = [candidate for done in checked.values() for candidate in done.candidates]
    shown = [candidate.positions[:_ROW_IDS] for candidate in found]
    view = _View(
        declared={
            name: [json_value(value) for value in declared.get(name, [])]
            for name in sentinel_values or {}
        },
        values=[candidate.record_value() for candidate in found],
        row_ids=[row_ids.at(positions) for positions in shown],
        all_row_ids=[row_ids.labels.take(c.positions) for c in found],
    )
    for name in _shown_columns(found, sentinel_values, row_ids.id_column):
        if masking.masks(name):
            view = _masked(view, name, found, declared, row_ids.id_column)
    candidates = [
        SentinelCandidate(
            column=candidate.column,
            value=value,
            value_type=type_name(stored_label(candidate.value)),
            count=len(candidate.positions),
            share=len(candidate.positions) / row_count,
            evidence=list(candidate.evidence),
            confirmation="declared" if candidate.declared else "suspected",
            row_ids=ids,
        )
        for candidate, value, ids in zip(found, view.values, view.row_ids, strict=True)
    ]
    # The candidates showing a row ID that another row shares.
    ambiguous = [
        f"candidates[{number}].row_ids"
        for number, candidate in enumerate(found)
        if row_ids.repeated[candidate.positions].any()
    ]
    status, status_reason = _check_status(
        row_count, selection.columns, row_ids.id_column
    )
    result = DetectMissingSentinelsResult.for_call(
        parameters={
            "columns": [column.name for column in selection.columns],
            "sentinel_values": None if sentinel_values is None else view.declared,
        },
        status=status,
        status_reason=status_reason,
        support=Support(
            row_count=row_count,
            column_count=len(checked),
            columns=[
                ColumnSupport.of(
                    name,
                    row_count,
                    {"missing": done.missing_count} if done.missing_count else {},
                )
                for name, done in checked.items()
            ],
        ),
        findings=_candidate_findings(candidates),
        limitations=[
            *ambiguous_row_ids(row_ids, ambiguous),
            *stored_nan_limitations(
                [name for name, done in checked.items() if done.stored_nan],
                ["support.columns"],
                "support counts it as missing",
            ),
            *([_NO_DECLARED_SENTINELS] if sentinel_values is None else []),
        ],
        masking=masking.record(
            [
                *_shown_columns(found, sentinel_values, None),
                *([row_ids.id_column] if found else []),
            ],
            data,
        ),
        candidates=candidates,
        columns_without_rules=[
            name for name, done in checked.items() if not done.ruled
        ],
    )
    result._row_ids = view.all_row_ids
    return result


# A candidate keeps the IDs of this many rows that hold it.
_ROW_IDS = 10


@dataclass(frozen=True)
class _Candidate:
    """A value of a column that is declared or suspected, and where it occurs."""

    column: str
    # A declared value as declared, a suspected one as the column stores it.
    value: object
    # Every row that holds it, in row order.
    positions: np.ndarray
    evidence: tuple[SentinelEvidence, ...]
    declared: bool

    def record_value(self) -> JsonSentinel:
        """The value as the record shows it, as every record writes a value."""
        return record_scalar(self.value)


@dataclass(frozen=True)
class _View:
    """What the result shows of the data's values: declared ones by column, each
    candidate's value and its row IDs, raw or masked."""

    declared: dict[str, list[JsonSentinel]]
    values: list[JsonSentinel]
    # The first row IDs of each candidate, as the record shows them.
    row_ids: list[list[object]]
    # Every row ID of each candidate: pandas labels, or a list once masked.
    all_row_ids: list[Any]


def _shown_columns(
    found: list[_Candidate],
    sentinel_values: Mapping[str, Sequence[SentinelValue]] | None,
    id_column: str | None,
) -> list[str]:
    """The columns whose values the result shows, in result order: those named
    in sentinel_values, those with candidates and id_column when it gives the
    candidates' row IDs."""
    names = [
        *(sentinel_values or {}),
        *(candidate.column for candidate in found),
        *([id_column] if id_column is not None and found else []),
    ]
    return list(dict.fromkeys(names))


def _masked(
    view: _View,
    name: str,
    found: list[_Candidate],
    declared: Mapping[str, Sequence[DeclaredValue]],
    id_column: str | None,
) -> _View:
    """The view with placeholders in place of the values of column `name`.

    One call numbers every value of the column in result order: its declared
    values, then each candidate's value and, when the column is id_column, its
    row IDs, so a value shows as the same placeholder everywhere.
    """
    own = declared.get(name, [])
    raw: list[object] = list(own)
    for candidate, ids in zip(found, view.row_ids, strict=True):
        if candidate.column == name:
            raw.append(candidate.value)
        if id_column == name:
            raw.extend(ids)
    # The row IDs beyond the shown ones come last, so the shown ones keep their
    # numbers and equal IDs share a placeholder.
    rest = (
        [
            row_id_list(full)[len(ids) :]
            for full, ids in zip(view.all_row_ids, view.row_ids, strict=True)
        ]
        if id_column == name
        else []
    )
    raw.extend(i for ids in rest for i in ids)
    placeholders = iter(mask_values(raw))

    def placeholder() -> str:
        # Declared and candidate values are never missing, so never None.
        return cast("str", next(placeholders))

    masked_declared = dict(view.declared)
    if name in masked_declared:
        masked_declared[name] = [placeholder() for _ in own]
    values: list[JsonSentinel] = []
    row_ids: list[list[object]] = []
    for candidate, value, ids in zip(found, view.values, view.row_ids, strict=True):
        values.append(placeholder() if candidate.column == name else value)
        # A missing row ID stays None.
        row_ids.append([next(placeholders) for _ in ids] if id_column == name else ids)
    if id_column != name:
        return _View(masked_declared, values, row_ids, view.all_row_ids)
    all_row_ids = [
        [*shown, *(next(placeholders) for _ in later)]
        for shown, later in zip(row_ids, rest, strict=True)
    ]
    return _View(masked_declared, values, row_ids, all_row_ids)


@dataclass(frozen=True)
class _Checked:
    """What checking one column found."""

    candidates: list[_Candidate]
    # Values pandas counts as missing, which no rule reads.
    missing_count: int
    # Whether a built-in rule reads any of its values.
    ruled: bool
    # Whether the column stores NaN as a value, which counts as missing here.
    stored_nan: bool


def _check_column(
    series: pd.Series, column: Column, declared: list[DeclaredValue]
) -> _Checked:
    """The declared values a column holds, in the order given, then the values
    the rules suspect among the rest.

    The rules read only values that are neither missing nor declared, so no
    declared value is also suspected, and the rules look past declared values
    as if they were missing.
    """
    native = np.asarray(series.isna(), dtype=bool)
    stored = stored_nan_mask(series)
    if stored is not None:
        native = native | stored
    remaining = ~native
    found = []
    for value in declared:
        match = matches(series, value, native)
        remaining &= ~match
        positions = np.flatnonzero(match)
        if positions.size:
            found.append(_Candidate(column.name, value, positions, ("declared",), True))
    holding = holds(series)
    found.extend(
        _Candidate(
            column.name, suspect.value, suspect.positions, suspect.evidence, False
        )
        for suspect in suspects(series, column, holding, remaining)
    )
    return _Checked(
        found, int(native.sum()), read_by_rules(holding), stored is not None
    )


def _check_status(
    row_count: int, checked: list[Column], id_column: str | None
) -> tuple[Status, StatusReason | None]:
    """insufficient_data without rows, not_applicable without a column to check,
    completed otherwise."""
    if not row_count:
        return "insufficient_data", NO_ROWS
    if checked:
        return "completed", None
    # Only columns=None can select nothing; it leaves out id_column.
    if id_column is None:
        message = "The data has no columns to check."
    else:
        message = (
            f"The data has no column to check besides id_column {id_column!r}, "
            "which the default columns leave out. Name the columns to check in columns."
        )
    return "not_applicable", StatusReason(code="no_columns", message=message)


_NO_DECLARED_SENTINELS = Limitation(
    code="sentinel_values_not_declared",
    reason="No sentinel values were declared, so every candidate is suspected, not confirmed.",
    fields=["candidates"],
)


def _candidate_findings(candidates: list[SentinelCandidate]) -> list[Finding]:
    """sentinel_values_present for declared candidates, suspected_sentinel_values
    for the others."""
    findings = []
    for confirmation in ("declared", "suspected"):
        paths = [
            (f"candidates[{position}]", candidate.column)
            for position, candidate in enumerate(candidates)
            if candidate.confirmation == confirmation
        ]
        if not paths:
            continue
        names = list(dict.fromkeys(column for _, column in paths))
        columns = plural(len(names), "column", "columns")
        if confirmation == "declared":
            code, assessment = "sentinel_values_present", "descriptive"
            claim = (
                f"Declared sentinel values occur in {columns}: {join_labels(names)}."
            )
        else:
            code, assessment = "suspected_sentinel_values", "suspected"
            claim = (
                f"{plural(len(paths), 'value', 'values')} in {columns} may stand in "
                f"for missing values: {join_labels(names)}."
            )
        findings.append(
            Finding(
                code=code,
                claim=claim,
                assessment=assessment,
                columns=names,
                fields=[path for path, _ in paths],
            )
        )
    return findings
