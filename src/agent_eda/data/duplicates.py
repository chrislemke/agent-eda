"""Analyses that find repeated records."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from agent_eda._columns import ALL_KINDS, ColumnRule, select_columns
from agent_eda._masking import mask_values, resolve_masking
from agent_eda._numeric import stored_nan_count, stored_nan_limitations
from agent_eda._registry import analysis
from agent_eda._result_base import (
    ColumnSupport,
    Finding,
    Limitation,
    Status,
    StatusReason,
    Support,
)
from agent_eda._row_groups import find_row_groups
from agent_eda._row_ids import ambiguous_row_ids, read_row_ids
from agent_eda._status import NO_ROWS
from agent_eda._text import format_number, plural
from agent_eda.data._duplicates_results import FindDuplicateRowsResult
from agent_eda.project import Project

# Rows can be compared in columns of any dtype.
_ANY_COLUMN = ColumnRule(
    default=ALL_KINDS,
    default_text="every column except id_column",
    accepted=ALL_KINDS,
    accepted_text="columns of any dtype",
    fixes={},
)


@analysis(
    statuses=("completed", "insufficient_data", "not_applicable"),
    finding_codes=("duplicate_rows",),
    tags=(
        "duplicates",
        "duplicate rows",
        "repeated rows",
        "identical rows",
        "dedupe",
        "deduplicate",
        "row ids",
    ),
    masking=True,
)
def find_duplicate_rows(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    id_column: str | None = None,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> FindDuplicateRowsResult:
    """Are there exact duplicate rows?

    When to use:
        Call it after `describe_dataset` to find rows that are equal in every compared
        column, such as records loaded twice. It groups the equal rows and reports each
        group's row IDs, the excess copies and the duplicate share; it removes no rows.
        Missing values compare equal, as in pandas' DataFrame.duplicated over two or more
        columns, even when only one column is compared. Pass id_column when a column
        identifies rows, such as an order number: row IDs then come from it, and the
        default columns leave it out of the comparison, so unique IDs don't hide
        duplicated records. Row IDs are raw values: when id_column holds personal data,
        declare it sensitive in a project and switch masking on; index labels are never
        masked, so pass a non-sensitive id_column. `describe_dataset` reports duplicated column labels and a non-unique
        index, `find_candidate_keys` the columns that identify rows, and
        `compute_association_matrix` the columns whose values duplicate or derive from
        each other.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns in which two rows
            must be equal to count as duplicates, in the order to compare them: a list
            or a single name. A label that isn't a string can be given as itself, such
            as 0, or as its string form. Columns of any dtype work. None, the default,
            compares every column except id_column, in DataFrame order.
        id_column (str | None): Name of a column whose values identify rows,
            such as an order number; row IDs are then its values, and the default
            columns leave it out of the comparison. Its values must be hashable, such as
            numbers or strings; a column of lists raises. A label that isn't a string
            can be given as itself or as its string form. None, the default, uses the
            index labels as row IDs.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces each row ID taken from a
            sensitive id_column with a numbered placeholder, '<masked 1>' for
            the first ID the result shows and so on, equal IDs alike, and
            needs a project that declares sensitive columns; False shows the
            IDs as stored. None, the default, follows the project's
            data_policy.mask_sensitive, and masks nothing without a project.
            Index labels are never masked. Masking is a guardrail against
            accidental exposure, not anonymization or access control.

    Returns:
        FindDuplicateRowsResult: `duplicate_groups` lists each group of rows that are
        equal in every compared column, ordered by first occurrence, with its `size` and
        its `row_ids` in row order (index labels, or id_column values when given).
        `duplicate_group_count` counts the groups, `duplicate_group_row_count` the rows
        in them, and `excess_copy_count` the rows beyond the first of each group, the
        ones that removing duplicates would drop. `duplicate_share` is the excess
        copies over all rows; it is null, with a `too_few_values` limitation, when
        nothing was compared. `compared_columns` names the columns compared. Missing
        values of every kind (None, NaN, NaT and NA) equal each other, and lists, dicts
        and sets compare by content.

        When index labels or id_column values repeat, a row ID can match more
        than one row: the `ambiguous_row_ids` limitation names the groups whose
        row IDs repeat and recommends a unique id_column. NaN that a pyarrow-backed
        or nullable float column stores as a value, which pandas' `isna` doesn't
        count, is a missing value here and equals every other: the
        `nan_counted_as_missing` limitation names such columns.

        With masking on, the row IDs from a sensitive id_column are placeholders in
        every output; the groups and counts stay.

        When no two rows are equal in every compared column, `findings` is empty
        and `duplicate_group_count` is 0: a search that finds nothing gives no
        finding.

        - `duplicate_rows`: suspected; two or more rows are equal in every compared
          column. Repeated records can be real, such as two identical purchases.

    Statuses:
        completed: The rows were compared, whether or not any repeat.
        insufficient_data: The data has no rows. Reason code: `no_rows`.
        not_applicable: There is no column to compare: the data has no
            columns, or only id_column, which the default columns leave out.
            Reason code: `no_columns`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "order": [101, 102, 103, 104, 105, 106],
        ...         "customer": ["ann", "bob", "ann", "cy", "bob", "ann"],
        ...         "amount": [9.5, 20.0, 9.5, None, 20.0, 9.5],
        ...     }
        ... )
        >>> res = ae.find_duplicate_rows(df, id_column="order")
        >>> print(res)
        find_duplicate_rows: completed on 6 rows x 3 columns
        Compared 2 columns: 'customer' and 'amount'; missing values compare equal, as in pandas.
        Row IDs: values of id_column 'order'.
        Duplicate groups (2 of 2 shown):
        group  size  row_ids
            0     3  101, 103 and 106
            1     2  102 and 105
        Findings (1 of 1 shown):
        - suspected duplicate_rows: 5 rows form 2 duplicate groups; 3 are excess copies (duplicate_share 0.5).
        Limitations: none
        More: res.to_frame() has all 5 rows in groups; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.duplicate_groups[0].row_ids
        [101, 103, 106]
        >>> project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["order"]))
        >>> masked = ae.find_duplicate_rows(
        ...     df, id_column="order", project=project, mask_sensitive=True
        ... )
        >>> masked.duplicate_groups[0].row_ids
        ['<masked 1>', '<masked 2>', '<masked 3>']

    See Also:
        find_candidate_keys: Which columns, alone or combined, identify rows; the rows
            behind a near-key's duplicate count are listed here.
        describe_dataset: Dtypes, missing and distinct counts and memory of every
            column, a non-unique index and duplicated column labels.
        compute_association_matrix: How strongly pairs of columns are associated, with
            columns that duplicate each other flagged.
    """
    masking = resolve_masking("find_duplicate_rows", project, mask_sensitive)
    row_ids = read_row_ids("find_duplicate_rows", data, id_column)
    selection = select_columns(
        "find_duplicate_rows", data, columns, _ANY_COLUMN, id_column=row_ids.id_column
    )
    compared = [column.name for column in selection.columns]
    stored_nan = [
        column.name
        for column in selection.columns
        if stored_nan_count(data.iloc[:, column.position])
    ]
    row_count = len(data)
    status, status_reason = _status(row_count, compared, row_ids.id_column)
    if status == "completed":
        positions, numbers = find_row_groups(
            data, [column.position for column in selection.columns]
        )
    else:
        positions = numbers = np.zeros(0, dtype=np.int64)
    sizes = np.bincount(numbers).tolist()
    # By group, then by position: positions ascend, and a stable sort keeps that.
    ids: list[object] = row_ids.at(positions[np.argsort(numbers, kind="stable")])
    if masking.masks(row_ids.id_column):
        ids = list(mask_values(ids))
    ends = np.cumsum(sizes).tolist()
    excess = len(positions) - len(sizes)
    # The groups holding a row ID that another row shares.
    shared = np.bincount(numbers[row_ids.repeated[positions]], minlength=len(sizes))
    ambiguous = [
        f"duplicate_groups[{number}].row_ids" for number in np.flatnonzero(shared)
    ]
    return FindDuplicateRowsResult.for_call(
        parameters={"columns": compared},
        status=status,
        status_reason=status_reason,
        support=Support(
            row_count=row_count,
            column_count=len(compared),
            columns=[ColumnSupport.of(name, row_count) for name in compared],
        ),
        findings=(
            _findings(compared, len(sizes), len(positions), row_count)
            if status == "completed"
            else []
        ),
        limitations=[
            *ambiguous_row_ids(row_ids, ambiguous),
            *stored_nan_limitations(
                stored_nan,
                ["duplicate_groups"],
                "it equals every missing value",
            ),
            *([] if status == "completed" else [_no_share(row_count)]),
        ],
        # The row IDs are the only values the result shows, if it has groups.
        masking=masking.record([row_ids.id_column] if sizes else [], data),
        compared_columns=compared,
        duplicate_groups=[
            {"size": size, "row_ids": ids[end - size : end]}
            for size, end in zip(sizes, ends, strict=True)
        ],
        duplicate_group_count=len(sizes),
        duplicate_group_row_count=len(positions),
        excess_copy_count=excess,
        duplicate_share=excess / row_count if status == "completed" else None,
    )


def _no_share(row_count: int) -> Limitation:
    """Why duplicate_share is null: nothing was compared."""
    why = (
        "The data has no rows"
        if not row_count
        else "No column was compared, so no row can repeat another"
    )
    return Limitation(
        code="too_few_values",
        reason=f"{why}, so duplicate_share, the excess copies over all rows, doesn't exist.",
        fields=["duplicate_share"],
    )


def _status(
    row_count: int, compared: list[str], id_column: str | None
) -> tuple[Status, StatusReason | None]:
    """How the run ends: insufficient_data without rows, not_applicable without
    columns to compare, completed otherwise."""
    if not row_count:
        return "insufficient_data", NO_ROWS
    if compared:
        return "completed", None
    # Only columns=None can select nothing; it leaves out id_column.
    if id_column is None:
        message = "The data has no columns to compare."
    else:
        message = (
            f"The data has no column to compare besides id_column {id_column!r}, "
            "which the default columns leave out. Name the columns to compare in columns."
        )
    return "not_applicable", StatusReason(code="no_columns", message=message)


def _findings(
    compared: list[str], groups: int, rows: int, row_count: int
) -> list[Finding]:
    """duplicate_rows when rows repeat, and no finding when none do."""
    if not groups:
        return []
    excess = rows - groups
    form = plural(groups, "duplicate group", "duplicate groups")
    copies = "1 is an excess copy" if excess == 1 else f"{excess:,} are excess copies"
    share = format_number(excess / row_count)
    return [
        Finding(
            code="duplicate_rows",
            claim=f"{rows:,} rows form {form}; {copies} (duplicate_share {share}).",
            assessment="suspected",
            columns=compared,
            fields=[
                "duplicate_group_count",
                "duplicate_group_row_count",
                "excess_copy_count",
                "duplicate_share",
            ],
        )
    ]
