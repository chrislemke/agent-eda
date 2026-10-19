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
        an option passed by position, an unknown keyword or a missing DataFrame;
        discovery functions and Project.load and Project.save raise it for an
        unknown keyword too, naming the closest one.
    unsupported_data_type (InvalidArgumentError): data isn't a pandas
        DataFrame, such as a file path, a pyarrow Table or None; the message
        shows how to get one.
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
    invalid_project_file (ProjectFileError): Project.load got a file that
        isn't UTF-8 TOML, and the message gives the line at fault, that lacks
        the [data_policy] table or one of its keys, or that holds a value a
        Project can't take, that isn't a regular file, that holds more than
        1 MiB, that the operating system refuses to read, or that has a newer
        format_version than this agent-eda reads; or project.save got a
        project with text that TOML can't hold, a lone surrogate, or one that
        would fill more than the 1 MiB that Project.load reads.
    unknown_project_key (ProjectFileError): Project.load got a file with a key
        that is neither a field of the Project nor one of its data policy,
        such as a misspelling; the message names the closest key and every
        allowed one.
    project_file_not_found (ProjectFileNotFoundError, also a
        FileNotFoundError): Project.load found no file at the path; the
        message shows where the path leads and how to save a project there.
    result_table_unavailable (ResultTableUnavailableError): to_frame() was
        called on a result that was rebuilt from its record, such as by
        model_validate(res.model_dump()), which doesn't hold the tables that
        only a call of the analysis keeps; the message says to call the
        analysis again or to copy the result with copy.deepcopy or model_copy.
    project_file_exists (ProjectFileExistsError, also a FileExistsError):
        project.save found a file at the path and overwrite was False; pass
        overwrite=True to replace it.
    missing_dependency (MissingDependencyError, also an ImportError): a
        feature needs an optional extra that isn't installed, such as the MCP
        server without the mcp extra; the message gives the pip and uv install
        commands.
    unsupported_data_format (UnsupportedDataFormatError): an MCP tool got a
        data path whose extension it can't read; the message lists the
        supported extensions.
    data_file_not_found (DataFileNotFoundError, also a FileNotFoundError): an
        MCP tool found no file at its data path; the message gives the
        absolute path tried, whether its directory exists and the closest file
        names in it.
    unreadable_data_file (DataFileError): an MCP tool's data path names a file
        the parser can't read, such as a CSV that isn't UTF-8; the message
        quotes the parser.
    table_file_exists (TableFileExistsError, also a FileExistsError): an MCP
        tool's table_path leads to a file that already exists, which the MCP
        server never overwrites; the message suggests a free path. A table_path
        with an extension other than .csv, .parquet or .pq, or in a directory
        that doesn't exist, raises invalid_argument. Both come before the
        analysis runs.
"""


class AgentEDAError(Exception):
    """Base class of every exception agent-eda raises.

    Attributes:
        code: A stable code naming the kind of mistake, such as
            'invalid_argument'. Codes are public API.
    """

    code: str = "agent_eda_error"
    # The message an MCP caller reads when its fix differs from the Python
    # one; agent_eda._wording sets it, and the MCP server sends it.
    _mcp_message: str | None = None

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


class ProjectFileError(AgentEDAError, ValueError):
    """A project file can't become a Project, or a Project can't become one."""

    code = "invalid_project_file"


class ProjectFileNotFoundError(ProjectFileError, FileNotFoundError):
    """Project.load found no file at its path.

    It is also a FileNotFoundError, so handlers for a missing file catch it.
    """

    code = "project_file_not_found"


class ProjectFileExistsError(ProjectFileError, FileExistsError):
    """project.save found a file at its path and wasn't told to replace it.

    It is also a FileExistsError, so handlers for an existing file catch it.
    """

    code = "project_file_exists"


class ResultTableUnavailableError(AgentEDAError, ValueError):
    """A result rebuilt from its record can't give a table it doesn't hold.

    model_dump() gives the record, which holds no level tables, so a result
    built from it again by model_validate() raises this from to_frame() instead
    of returning a table without them. copy.deepcopy and model_copy keep them.
    """

    code = "result_table_unavailable"


class MissingDependencyError(AgentEDAError, ImportError):
    """A feature needs an optional extra that isn't installed.

    It is also an ImportError, so handlers for a missing module catch it.
    """

    code = "missing_dependency"


class UnsupportedDataFormatError(InvalidArgumentError):
    """An MCP tool got a data path whose extension it can't read."""

    code = "unsupported_data_format"


class DataFileError(InvalidArgumentError):
    """An MCP tool can't read the file its data path names."""

    code = "unreadable_data_file"


class DataFileNotFoundError(DataFileError, FileNotFoundError):
    """An MCP tool found no file at its data path.

    It is also a FileNotFoundError, so handlers for a missing file catch it.
    """

    code = "data_file_not_found"


class TableFileExistsError(AgentEDAError, FileExistsError):
    """An MCP tool's table_path leads to a file that already exists.

    The MCP server never overwrites a file. It is also a FileExistsError, so
    handlers for an existing file catch it.
    """

    code = "table_file_exists"
