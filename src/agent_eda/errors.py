"""Exceptions that agent-eda raises when a call is wrong.

Every exception subclasses AgentEDAError and carries a stable `code` to branch
on. Each also subclasses the closest built-in exception, so generic handlers keep
working. Every message says what is wrong, what is allowed and how to fix it.

Mistakes in a call raise before any computation starts. Problems in the data
never raise: they come back in the result's status, findings and limitations.

Codes:
    invalid_argument: an argument has the wrong type or value.
    invalid_call: the call itself is malformed, such as an option passed by
        position, an unknown keyword or a missing DataFrame.
    unsupported_data_type: data isn't a pandas DataFrame; the message shows the
        one-line conversion.
"""


class AgentEDAError(Exception):
    """Base class of every exception agent-eda raises.

    Attributes:
        code: A stable code naming the kind of mistake, such as
            'invalid_argument'. Codes are public API.
    """

    code: str = "agent_eda_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidArgumentError(AgentEDAError, ValueError):
    """An argument, or the call itself, is invalid."""

    code = "invalid_argument"
