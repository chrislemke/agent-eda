"""The local files an MCP tool reads and writes: their formats and their paths.

A data path, a table path and a project path each name a local file, absolute
or relative to the MCP server's working directory. One table of formats serves
the reader (data_files) and the writer (table_files), and one check says what
a path argument must be. Nothing here imports mcp.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_eda._text import ELLIPSIS, join_alternatives, join_items
from agent_eda.errors import InvalidArgumentError


@dataclass(frozen=True)
class FileFormat:
    """How to read, and maybe write, the files with one extension."""

    extension: str
    kind: Literal["CSV", "TSV", "Parquet"]
    gzip: bool = False
    # Whether table_path writes this format too.
    writable: bool = False

    @property
    def is_parquet(self) -> bool:
        return self.kind == "Parquet"

    @property
    def separator(self) -> str:
        """The character that separates the fields of a CSV or TSV file."""
        return "\t" if self.kind == "TSV" else ","

    @property
    def separator_word(self) -> str:
        """The separator as a message says it."""
        return "tabs" if self.kind == "TSV" else "commas"

    @property
    def label(self) -> str:
        """The format as a message names it: "CSV", "gzipped TSV", "Parquet"."""
        return f"{'gzipped ' if self.gzip else ''}{self.kind}"


# Every format the MCP server reads, matched by extension in any case.
READABLE = (
    FileFormat(".csv", "CSV", writable=True),
    FileFormat(".tsv", "TSV"),
    FileFormat(".csv.gz", "CSV", gzip=True),
    FileFormat(".tsv.gz", "TSV", gzip=True),
    FileFormat(".parquet", "Parquet", writable=True),
    FileFormat(".pq", "Parquet", writable=True),
)
# The formats table_path writes.
WRITABLE = tuple(f for f in READABLE if f.writable)

# What every path argument accepts, after "absolute or relative".
_RELATIVE = "absolute or relative to the MCP server's working directory"


# What the operating system, or Python on its behalf, raises for a path it can't
# follow: an OSError for most causes, a ValueError for a NUL character and, in
# Python 3.12, a RuntimeError for a symbolic link loop or a ~user that has no
# home directory.
PATH_ERRORS = (OSError, ValueError, RuntimeError)

# The longest a path argument is shown in a message.
_SHOWN = 80


def resolve(path: str) -> Path:
    """The absolute local path a path argument names: a leading ~ is the home
    directory, and relative paths start at the working directory.

    Raises:
        PATH_ERRORS: the path can't be followed, as os_cause says.
    """
    return (Path.cwd() / Path(path).expanduser()).resolve()


def shown(path: str) -> str:
    """The path as text for a message, cut in the middle when it is long."""
    text = repr(path)
    if len(text) <= _SHOWN:
        return text
    half = (_SHOWN - len(ELLIPSIS)) // 2
    return f"{text[:half]}{ELLIPSIS}{text[-half:]}"


def extensions(formats: Sequence[FileFormat], *, either: bool = False) -> str:
    """The extensions of formats, as text for a message: ".csv, .tsv and .pq", or
    with `either` ".csv, .tsv or .pq"."""
    names = [f.extension for f in formats]
    return join_alternatives(names) if either else join_items(names, limit=len(names))


def format_names(formats: Sequence[FileFormat]) -> str:
    """The kinds of formats, each once, as text for a message: "CSV, TSV or
    Parquet"."""
    return join_alternatives(list(dict.fromkeys(f.kind for f in formats)))


def format_of(path: str, formats: Sequence[FileFormat]) -> FileFormat | None:
    """The format whose extension ends the path's file name, in any case."""
    name = Path(path).name.lower()
    return next((f for f in formats if name.endswith(f.extension)), None)


def extension_text(path: str) -> str:
    """The path's extension as text for a message, such as "the extension
    '.json'" or, under a .gz, "the extension '.json.gz'", or "no extension"."""
    found = Path(path).suffixes
    suffixes = "".join(found[-2:] if found[-1:] == [".gz"] else found[-1:])
    return f"the extension {suffixes!r}" if suffixes else "no extension"


def path_argument(
    analysis: str, argument: str, value: object, *, what: str, nullable: bool
) -> str:
    """The value of a path argument, such as data_path; raise unless it is text.

    `what` says what the file is, such as "the path to the data file".

    Raises:
        InvalidArgumentError: value isn't a string.
    """
    if isinstance(value, str):
        if "\x00" in value:
            raise InvalidArgumentError(
                f"{analysis} got {shown(value)} for {argument}, which holds a NUL "
                f"character that no file name can. Pass {what}, {_RELATIVE}, "
                "without it."
            )
        return value
    allowed = "a string or null" if nullable else "a string"
    raise InvalidArgumentError(
        f"{analysis} got {value!r} for {argument}, but {argument} must be {allowed}: "
        f"{what}, {_RELATIVE}."
    )
