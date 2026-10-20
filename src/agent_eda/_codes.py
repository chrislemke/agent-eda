"""What each code of the package-wide lists means.

The lists are `Literal`s in `agent_eda._result_base`, each with a mapping from its
codes to their meanings beside it. The mapping is the only copy of a code's
meaning, so that the list and its meanings can't disagree: describe_analysis and
the README's Reference read the meanings from here. A meaning is data, not a
comment, so it is there however the package is installed, such as from a zip of
bytecode with no source to read.
"""

from agent_eda._result_base import (
    LIMITATION_MEANINGS,
    STATUS_REASON_MEANINGS,
    LimitationCode,
    StatusReasonCode,
)


def status_reason_meanings() -> dict[StatusReasonCode, str]:
    """Each status reason code with its meaning, in the order the list has them."""
    return dict(STATUS_REASON_MEANINGS)


def limitation_meanings() -> dict[LimitationCode, str]:
    """Each limitation code with its meaning, in the order the list has them."""
    return dict(LIMITATION_MEANINGS)
