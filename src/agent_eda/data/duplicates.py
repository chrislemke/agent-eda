"""Analyses that find repeated records."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from agent_eda._columns import ALL_KINDS, ColumnRule, select_columns
from agent_eda._masking import mask_values, resolve_masking
from agent_eda._registry import analysis
from agent_eda._row_ids import ambiguous_row_ids, read_row_ids
from agent_eda._run import NO_ROWS, data_shape, execution
from agent_eda._text import format_number, plural
from agent_eda._values import column_values, hashable_values
from agent_eda.project import Project
from agent_eda.results import (
    ColumnSupport,
    FindDuplicateRowsResult,
    Finding,
    Limitation,
    Status,
    StatusReason,
    Support,
)

# Rows can be compared in columns of any dtype.
_ANY_COLUMN = ColumnRule(
    default=ALL_KINDS,
    default_text="every column except id_column",
    accepted=ALL_KINDS,
    accepted_text="columns of any dtype",
    fixes={},
)


@analysis(
    statuses=("completed", "insufficient_data", "not_applicable", "failed"),
    finding_codes=("duplicate_rows", "no_duplicate_rows"),
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
        Call it after `describe_dataset` to find rows that are equal in every
        compared column, such as records loaded twice. It groups the equal rows
        and reports each group's row IDs, the rows in groups, the excess copies
        and the duplicate share. It removes no rows, so you decide what to do
        with them. Missing values compare equal, as in pandas'
        DataFrame.duplicated over two or more columns, even when only one
        column is compared.
        Pass id_column when a column identifies rows, such as an order number:
        row IDs then come from it, and columns=None leaves it out of the
        comparison, so unique IDs don't hide duplicated records. Row IDs are
        raw values, so when id_column holds personal data, such as email
        addresses, declare it sensitive in a project and switch masking on.
        Index labels are never masked: when the index holds sensitive values,
        pass a non-sensitive id_column. `describe_dataset` reports duplicated
        column labels and a non-unique index, and columns whose values
        duplicate or derive from each other show up as perfect associations in
        `compute_association_matrix`; `list_analyses` lists the other analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns in which two rows
            must be equal to count as duplicates, in the order to compare them:
            a list, a tuple, a pandas Index such as df.columns[:3], or a single
            name. A label that isn't a string can be given as itself, such as 0,
            or as its string form. Columns of any dtype work. None, the default,
            compares every column except id_column, in DataFrame order.
        id_column (str | None): Name of a column whose values identify rows,
            such as an order number; row IDs are then its values, and
            columns=None leaves it out of the comparison. Its values must be
            hashable, such as numbers or strings; a column of lists raises. A
            label that isn't a string can be given as itself or as its string
            form. None, the default, uses the index labels as row IDs.
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
        FindDuplicateRowsResult: `duplicate_groups` lists each group of rows
        that are equal in every compared column, ordered by first occurrence,
        with its `size` and its `row_ids` in row order. With index labels as
        row IDs, df.loc[row_ids] shows the rows. `duplicate_group_count`
        counts the groups and `duplicate_group_row_count` the rows in them.
        `excess_copy_count` counts the rows beyond the first of each group,
        the ones that removing duplicates would drop, and `duplicate_share` is
        the excess copies over all rows; it is null, with a `too_few_values`
        limitation, when the data has no rows. `compared_columns` names the
        columns compared, and `missing_values_compare_equal` records that
        missing values (None, NaN, NaT, pd.NA) equal each other. Lists, dicts
        and sets compare by content. `to_frame` returns every row in a
        duplicate group with its group number. The record keeps the first 100
        groups and the first 10 row IDs of each, with their totals.

        When index labels or id_column values repeat, a row ID can match more
        than one row: the `ambiguous_row_ids` limitation names the groups whose
        row IDs repeat and recommends a unique id_column.

        `masking` says whether masking was on and names id_column when it is
        sensitive. With masking on, its row IDs are placeholders, numbered in
        the order the groups show them, in the printed text, `to_dict`,
        `to_json` and `to_frame` alike; the groups and counts stay.

        - `duplicate_rows`: suspected; two or more rows are equal in every compared
          column. Repeated records can be real, such as two identical purchases.
        - `no_duplicate_rows`: descriptive; no two rows are equal in every compared
          column.

    Statuses:
        completed: The rows were compared, whether or not any repeat.
        insufficient_data: The data has no rows.
        not_applicable: There is no column to compare: the data has no
            columns, or only id_column, which columns=None leaves out.
        failed: An anticipated failure stopped the comparison, and
            `status_reason` gives its code and message. None is known today:
            values that can't be hashed, such as lists, compare by content
            instead.

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
        describe_dataset: Dtypes, missing and distinct counts and memory of
            every column, a non-unique index and duplicated column labels.
        compute_association_matrix: How strongly pairs of columns are
            associated, with columns that duplicate each other flagged.
        list_analyses: Every analysis, with the question each one answers.
    """
    masking = resolve_masking("find_duplicate_rows", project, mask_sensitive)
    row_ids = read_row_ids("find_duplicate_rows", data, id_column)
    selection = select_columns(
        "find_duplicate_rows", data, columns, _ANY_COLUMN, id_column=row_ids.id_column
    )
    compared = [column.name for column in selection.columns]
    row_count = len(data)
    status, status_reason = _status(row_count, compared, row_ids.id_column)
    if status == "completed":
        positions, numbers = _find_groups(
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
    return FindDuplicateRowsResult(
        analysis="find_duplicate_rows",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": compared,
            "id_column": row_ids.id_column,
            "project": project.model_dump(mode="json") if project else None,
            "mask_sensitive": mask_sensitive,
        },
        support=Support(
            row_count=row_count,
            column_count=len(compared),
            columns=[
                ColumnSupport(
                    column=name,
                    eligible_count=row_count,
                    evaluated_count=row_count,
                    excluded_count=0,
                    excluded_by_reason={},
                )
                for name in compared
            ],
        ),
        findings=(
            _findings(compared, len(sizes), len(positions), row_count)
            if status == "completed"
            else []
        ),
        limitations=[
            *ambiguous_row_ids(row_ids, ambiguous),
            *([] if row_count else [_NO_SHARE]),
        ],
        execution=execution(("numpy", "pandas")),
        # The row IDs are the only values the result shows, if it has groups.
        masking=masking.record([row_ids.id_column] if sizes else [], data),
        compared_columns=compared,
        missing_values_compare_equal=True,
        duplicate_groups=[
            {"size": size, "row_ids": ids[end - size : end]}
            for size, end in zip(sizes, ends, strict=True)
        ],
        duplicate_group_count=len(sizes),
        duplicate_group_row_count=len(positions),
        excess_copy_count=excess,
        duplicate_share=excess / row_count if row_count else None,
    )


_NO_SHARE = Limitation(
    code="too_few_values",
    reason=(
        "The data has no rows, so duplicate_share, the excess copies over all "
        "rows, doesn't exist."
    ),
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
            "which columns=None leaves out. Name the columns to compare in columns."
        )
    return "not_applicable", StatusReason(code="no_columns_to_compare", message=message)


def _findings(
    compared: list[str], groups: int, rows: int, row_count: int
) -> list[Finding]:
    """duplicate_rows when rows repeat, no_duplicate_rows otherwise."""
    if not groups:
        return [
            Finding(
                code="no_duplicate_rows",
                claim="No two rows are equal in every compared column.",
                assessment="descriptive",
                columns=compared,
                fields=["duplicate_group_count"],
            )
        ]
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


# Keys combine the codes of the columns read so far. Before a key could grow past
# this, keys are renumbered, so int64 never overflows.
_KEY_LIMIT = 2**62
# Dropping rows whose key is unique keeps paying off while it drops at least
# this share of the candidate rows.
_DROP_SHARE = 0.1


def _find_groups(
    data: pd.DataFrame, positions: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The rows equal to another row in the columns at `positions`, and their groups.

    Returns the rows' positions in ascending order and each one's group number.
    Groups are numbered from 0 in the order of their first row.

    Each candidate row carries a key that rows equal in the columns read so far
    share, extended column by column. While the keys can tell rows apart, rows
    whose key no other row has stop being candidates, so the remaining columns
    are read only for the rows that can still have duplicates: with few
    duplicates, a few columns settle every row. Once that stops dropping rows,
    as when most rows have duplicates, the keys keep growing and are only
    renumbered before they could overflow.
    """
    row_count = len(data)
    candidates = np.arange(row_count)
    keys = np.zeros(row_count, dtype=np.int64)
    span = 1  # every key lies in range(span)
    dropping = True
    for position in positions:
        codes, code_count = _codes(data.iloc[:, position], candidates, row_count)
        if span * code_count > _KEY_LIMIT:
            keys, span = _renumber(keys)
        keys = keys * code_count + codes
        span *= code_count
        if dropping and span >= candidates.size:
            keys, span = _renumber(keys)
            keep = np.bincount(keys)[keys] >= 2
            dropped = candidates.size - int(np.count_nonzero(keep))
            dropping = dropped >= _DROP_SHARE * candidates.size
            if dropped:
                candidates, keys = candidates[keep], keys[keep]
            if not candidates.size:
                break
    keys, _ = _renumber(keys)
    keep = np.bincount(keys)[keys] >= 2
    numbers, _ = _renumber(keys[keep])
    return candidates[keep], numbers


def _codes(
    column: pd.Series, rows: np.ndarray, row_count: int
) -> tuple[np.ndarray, int]:
    """A code per row that equal values share, and how many codes there can be.

    Codes start at 1, and every missing value gets 0, so missing values compare
    equal. Values that can't be hashed, such as lists and dicts, compare by
    their content, while every other value in the column still compares as
    pandas compares it, so which rows are left doesn't change the result.
    """
    values = column_values(column)
    if rows.size < row_count:
        values = values.take(rows)
    try:
        codes, uniques = pd.factorize(values)
    except (TypeError, NotImplementedError):
        # Lists and dicts raise TypeError; nested pyarrow types raise
        # ArrowNotImplementedError, a NotImplementedError.
        codes, uniques = pd.factorize(hashable_values(values))
    return codes.astype(np.int64, copy=False) + 1, len(uniques) + 1


def _renumber(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Number the distinct keys from 0 in order of appearance, and count them."""
    numbers, uniques = pd.factorize(keys)
    return numbers.astype(np.int64, copy=False), len(uniques)
