"""Exceptions that agent-eda raises when a call is wrong.

Every exception subclasses AgentEDAError and carries a stable `code` to branch
on. Each also subclasses the closest built-in exception, so generic handlers keep
working. Every message says what is wrong, what is allowed and how to fix it.

Mistakes in a call raise before any computation starts. Problems in the data
never raise: they come back in the result's status, findings and limitations.

Codes, with the class that raises each:
    invalid_argument (InvalidArgumentError): an argument has the wrong type or
        value, such as a number outside its range, an unknown choice, a list
        that repeats a value, or a declared sentinel value that can never match
        its column; that message names the value the caller likely meant.
    invalid_call (InvalidArgumentError): the call itself is malformed, such as
        an option passed by position, an unknown keyword or a missing DataFrame.
    unsupported_data_type (InvalidArgumentError): data isn't a pandas
        DataFrame; the message shows the one-line conversion.
    ambiguous_column_labels (InvalidArgumentError): the data's column labels
        repeat, share a string form or form a MultiIndex, so a name can't pick
        out one column; the message shows how to rename them. describe_dataset
        takes such data and reports the labels.
    column_not_found (ColumnNotFoundError): a column name isn't in the data;
        the message lists the closest names and the available ones.
    unsupported_column_type (ColumnTypeError): a named column has a dtype the
        analysis can't use, or an id_column holds values that can't be hashed,
        such as lists; the message names the dtype, what is accepted and a fix.
    unknown_analysis (UnknownAnalysisError): describe_analysis got a name that
        isn't an analysis; the message suggests the closest names and lists
        every analysis.
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


class ColumnNotFoundError(AgentEDAError, LookupError):
    """A column name in the call isn't a column of the data."""

    code = "column_not_found"


class ColumnTypeError(AgentEDAError, TypeError):
    """A column named in the call has a dtype the analysis can't use."""

    code = "unsupported_column_type"


class UnknownAnalysisError(AgentEDAError, LookupError):
    """A name passed to describe_analysis isn't the name of an analysis."""

    code = "unknown_analysis"
