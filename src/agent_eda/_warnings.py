"""Warnings raised inside an analysis: recorded, never shown to the caller.

Numerical RuntimeWarnings become limitations; every other warning is raised again
under the caller's own filters, so only a caller whose filters turn it into an
error, as the test suite's do, gets it. Recording changes the process's warning
filters, so one capture runs at a time, under a lock.
"""

import contextlib
import threading
import warnings
from collections.abc import Generator

import numpy as np

from agent_eda._result_base import AnalysisResult, Limitation

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


def absorb_warnings(result: object, caught: list[warnings.WarningMessage]) -> object:
    """Keep the warnings an analysis raised away from its caller.

    Numerical RuntimeWarnings that the analysis didn't tie to the values they
    affect become one limitation. Every other warning, DeprecationWarning and
    FutureWarning among them, is raised again under the caller's own filters
    but shown to no one: only a caller whose filters turn it into an error, as
    the test suite's do, gets it, as that error. A recorded warning keeps its
    file but not its module, so a filter that names a module matches against
    the file's path.
    """
    numerical: list[str] = []
    others: dict[tuple[type[Warning], str, str, int], warnings.WarningMessage] = {}
    for warning in caught:
        if issubclass(warning.category, RuntimeWarning):
            if str(warning.message) not in numerical:
                numerical.append(str(warning.message))
        else:
            key = (
                warning.category,
                str(warning.message),
                warning.filename,
                warning.lineno,
            )
            others.setdefault(key, warning)
    for warning in others.values():
        with capture_warnings():
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    if not numerical or not isinstance(result, AnalysisResult):
        return result
    limitation = Limitation(
        code="numerical_warning",
        reason=(
            "Numerical warnings were raised that can't be tied to particular "
            f"values: {'; '.join(numerical)}."
        ),
        fields=[],
    )
    return result.model_copy(update={"limitations": [*result.limitations, limitation]})
