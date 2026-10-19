"""Pieces every analysis puts into its result, and checks every analysis runs."""

import contextlib
import importlib
import json
import platform
import threading
import warnings
from collections import Counter
from collections.abc import Callable, Generator, Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue

from agent_eda._text import Surface, join_items, join_labels, plural
from agent_eda._version import __version__
from agent_eda._wording import echoed, worded
from agent_eda.errors import InvalidArgumentError
from agent_eda.results import (
    ColumnResult,
    Execution,
    Status,
    StatusReason,
    StatusReasonCode,
)

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


# warnings.catch_warnings swaps the process's warning filters and showwarning
# hook, so two captures that overlap in time restore each other's state and
# leave the caller's changed. Every capture runs under this one lock. It is
# reentrant because a capture can start inside another, such as a numeric
# summary inside an analysis.
_CAPTURE_LOCK = threading.RLock()


@contextlib.contextmanager
def capture_warnings() -> Generator[list[warnings.WarningMessage], None, None]:
    """Record the warnings raised in the block, one capture at a time process-wide."""
    with _CAPTURE_LOCK, warnings.catch_warnings(record=True) as caught:
        yield caught


@contextlib.contextmanager
def watch_all_warnings() -> Generator[list[warnings.WarningMessage], None, None]:
    """Record every warning raised in the block instead of showing it.

    NumPy's floating-point errors warn inside the block, whatever numpy.seterr
    says outside it, so they are recorded too.
    """
    with capture_warnings() as caught:
        warnings.simplefilter("always")
        with np.errstate(all="warn", under="ignore"):
            yield caught


@contextlib.contextmanager
def watch_warnings() -> Generator[list[str], None, None]:
    """Collect the messages of the RuntimeWarnings raised in the block.

    NumPy's floating-point errors (division by zero, overflow, invalid values)
    warn inside the block, whatever numpy.seterr says outside it. Once the block
    ends, the list holds each message once, in order; other warnings pass on.
    """
    messages: list[str] = []
    with watch_all_warnings() as caught:
        yield messages
    for warning in caught:
        if issubclass(warning.category, RuntimeWarning):
            if str(warning.message) not in messages:
                messages.append(str(warning.message))
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
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


def check_distinct(
    analysis: str,
    parameter: str,
    values: Sequence[Hashable],
    noun: str,
    *,
    key: Callable[[Any], Hashable] | None = None,
    hide: bool = False,
    within: str | None = None,
) -> None:
    """Raise when a list argument holds the same value more than once.

    Values are the same when Python's == says so, or when their `key` is. With
    `hide`, the message counts the repeated values and doesn't quote them: they
    are values of a sensitive column. `within` names the key of a mapping
    argument whose list this is, such as a column of sentinel_values.
    """
    keyed = values if key is None else [key(value) for value in values]
    repeated = [
        values[keyed.index(item)] for item, count in Counter(keyed).items() if count > 1
    ]
    if not repeated:
        return

    def text(surface: Surface) -> str:
        if hide:
            named = (
                f"{plural(len(repeated), noun, noun + 's')} (not shown: masking "
                "hides the values of a sensitive column)"
            )
        else:
            named = join_items([echoed(value, surface) for value in repeated])
        where = parameter
        if within is not None:
            where = (
                f"{parameter}[{within!r}]"
                if surface == "python"
                else f"the {parameter} list for {json.dumps(within)}"
            )
        return (
            f"{analysis} got {named} more than once in {where}. List each {noun} once."
        )

    raise worded(InvalidArgumentError, text)


def data_shape(data: pd.DataFrame) -> dict[str, JsonValue]:
    """How a DataFrame argument appears in a result's parameters."""
    return {"row_count": data.shape[0], "column_count": data.shape[1]}


def execution(libraries: Sequence[str]) -> Execution:
    """The versions of agent_eda, Python and the given libraries, by import name."""
    versions = {"agent_eda": __version__, "python": platform.python_version()}
    for name in libraries:
        versions[name] = str(importlib.import_module(name).__version__)
    return Execution(versions=versions)
