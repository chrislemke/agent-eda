"""How an analysis's run ends: its status, and why.

An analysis that works column by column ends with one status for the run, which
`overall_status` picks from the columns', and one that stops at a work limit says
why with `budget_exceeded_reason`. `NO_ROWS` is the reason shared by every
analysis of data without rows.
"""

from collections.abc import Sequence

from agent_eda._result_base import (
    ColumnResult,
    Status,
    StatusReason,
    StatusReasonCode,
)
from agent_eda._text import join_labels, plural

# When no column completed and their statuses differ, the first of these that
# occurs is the result's status.
_PRECEDENCE: tuple[Status, ...] = ("failed", "insufficient_data", "not_applicable")


# Why an analysis of data without rows ends with insufficient_data.
NO_ROWS = StatusReason(code="no_rows", message="The data has no rows.")


def overall_status(
    row_count: int, entries: Sequence[ColumnResult], *, no_columns: StatusReason
) -> tuple[Status, StatusReason | None]:
    """The status of an analysis over several columns, and why.

    Data without rows gives insufficient_data and no selected column gives
    not_applicable, with `no_columns` as the reason. Otherwise the result is
    completed when every column completed and partial when some did. When none
    did, it takes the status the columns share or, if they differ, the first
    of failed, insufficient_data and not_applicable that occurs.
    """
    if row_count == 0:
        return "insufficient_data", NO_ROWS
    if not entries:
        return "not_applicable", no_columns
    incomplete = [entry for entry in entries if entry.status != "completed"]
    if not incomplete:
        return "completed", None
    names = join_labels([entry.column for entry in incomplete])
    # A column that didn't complete always has a reason; its model checks that.
    reasons = [entry.status_reason for entry in incomplete if entry.status_reason]
    messages = list(dict.fromkeys(reason.message for reason in reasons))
    why = messages[0] if len(messages) == 1 else "Each column's status_reason says why."
    if len(incomplete) < len(entries):
        return "partial", StatusReason(
            code="incomplete_columns",
            message=(
                f"{len(incomplete):,} of {len(entries):,} columns didn't complete: "
                f"{names}. {why}"
            ),
        )
    statuses = [entry.status for entry in incomplete]
    status = statuses[0]
    if len(set(statuses)) > 1:
        status = next((s for s in _PRECEDENCE if s in statuses), status)
    # The reason of the first column with that status gives the code.
    code = next(
        reason.code
        for entry, reason in zip(incomplete, reasons, strict=True)
        if entry.status == status
    )
    return status, StatusReason(
        code=code, message=f"No column completed: {names}. {why}"
    )


def budget_exceeded_reason(
    *,
    code: StatusReasonCode,
    needs: str,
    required: int,
    unit: tuple[str, str],
    parameter: str,
    limit: int,
    consequence: str,
    narrow: str,
    lower: tuple[str, int] | None = None,
    raise_up_to: int | None = None,
) -> StatusReason:
    """Why a run ended at its work limit: the work it needs, the limit and the fixes.

    The message reads "<needs> <required> <unit>, more than the <parameter> limit
    of <limit> allows; <consequence>. Raise <parameter> to <required>, or
    <narrow>.", such as "The complete search covers up to 10 column sets, more
    than the max_combinations limit of 5 allows; it stopped after examining 5
    and kept what it found. Raise max_combinations to 10, or name fewer
    columns."

    `lower` is another argument that shrinks the work, with the value that fits
    the limit, and comes first: "Lower max_key_size to 2, raise ...". Raising the
    limit is suggested only when `required` is at most `raise_up_to`, so that no
    message asks for a limit no run could use; None means any value will do.
    """
    steps = []
    if lower is not None:
        steps.append(f"lower {lower[0]} to {lower[1]:,}")
    if raise_up_to is None or required <= raise_up_to:
        steps.append(f"raise {parameter} to {required:,}")
    steps.append(narrow)
    fix = ", ".join(steps[:-1]) + (", or " if len(steps) > 1 else "") + steps[-1]
    return StatusReason(
        code=code,
        message=(
            f"{needs} {plural(required, *unit)}, more than the {parameter} limit of {limit:,} "
            f"allows; {consequence}. {fix[0].upper()}{fix[1:]}."
        ),
    )
