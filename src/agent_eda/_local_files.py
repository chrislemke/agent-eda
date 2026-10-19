"""The local files an MCP tool reads and writes: their formats and their paths.

A data path, a table path and a project path each name a local file, absolute
or relative to the MCP server's working directory. One table of formats serves
the reader (_data_files) and the writer (_table_files), and one check says what
a path argument must be. Nothing here imports mcp.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_eda._text import join_items
from agent_eda.errors import InvalidArgumentError


@dataclass(frozen=True)
class FileFormat:
    """How to read, and maybe write, the files with one extension."""

    extension: str
    kind: Literal["CSV", "TSV", "Parquet"]
    gzip: bool = False
    # Whether table_path writes this format too.
    writable: bool = False


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


def resolve(path: str) -> Path:
    """The absolute local path a path argument names, relative ones resolved
    against the working directory."""
    return (Path.cwd() / Path(path)).resolve()


def extensions(formats: Sequence[FileFormat]) -> str:
    """The extensions of formats, as text for a message: ".csv, .tsv and .pq"."""
    return join_items([f.extension for f in formats], limit=len(formats))


def format_of(path: str, formats: Sequence[FileFormat]) -> FileFormat | None:
    """The format whose extension ends the path's file name, in any case."""
    name = Path(path).name.lower()
    return next((f for f in formats if name.endswith(f.extension)), None)


def extension_text(path: str) -> str:
    """The path's last two suffixes as text for a message, such as
    "the extension '.json.gz'", or "no extension"."""
    suffixes = "".join(Path(path).suffixes[-2:])
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
        return value
    allowed = "a string or null" if nullable else "a string"
    raise InvalidArgumentError(
        f"{analysis} got {value!r} for {argument}, but {argument} must be {allowed}: "
        f"{what}, {_RELATIVE}."
    )
