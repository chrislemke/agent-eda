"""Writing a result's full table to the file a table path names.

An MCP tool given a table_path writes the table to_frame gives, as CSV or
Parquet. The path is checked before the analysis runs, and the file is created
exclusively, so an existing file is never overwritten, not even one that
appears while the analysis runs. Like data paths, a table path is a local
path: this module opens the file itself, so nothing reaches pandas' URL
handling.
"""

import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from agent_eda._local_files import (
    PATH_ERRORS,
    WRITABLE,
    FileFormat,
    extension_text,
    extensions,
    format_of,
    os_cause,
    path_argument,
    shown,
)
from agent_eda._text import MAX_WIDTH, plural
from agent_eda._written_form import row_id_json
from agent_eda.errors import InvalidArgumentError, TableFileExistsError

# What pyarrow raises for a column it can't store as one Arrow type, such as
# numbers mixed with text, or numbers that don't fit one (a Python int past 64
# bits is an OverflowError).
_ARROW_ERRORS = (
    pa.ArrowInvalid,
    pa.ArrowTypeError,
    pa.ArrowNotImplementedError,
    OverflowError,
)

# What stands in for an empty string while pandas writes a CSV, repeated until
# no other cell holds it, up to this many times.
_EMPTY = "\x01"
_MARKER_TRIES = 4

# What encoding a table as a file can still raise after the columns Arrow can't
# store are written as text: a column name or an index that has no place in the
# file.
_ENCODE_ERRORS = (*_ARROW_ERRORS, KeyError, TypeError, ValueError, pa.ArrowException)


@dataclass(frozen=True)
class TableFile:
    """A checked table path: the absolute path and the format to write."""

    path: Path
    file_format: FileFormat

    def write(self, table: pd.DataFrame) -> "WrittenTable":
        """Write the table and say where and how many rows.

        A write that fails partway removes the partial file, so the same call
        can be made again.

        Raises:
            TableFileExistsError: a file appeared at the path after the check.
            InvalidArgumentError: the directory went away after the check, the
                MCP server may not write there, the write failed (such as on a
                full disk) or the table can't be stored in the file's format.
        """
        try:
            content = _encode(_cells(table), self.file_format)
        except _ENCODE_ERRORS as error:
            raise InvalidArgumentError(_not_storable(self, error)) from None
        try:
            # O_EXCL: never overwrite, and never write through a symbolic link.
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o666,
            )
        except FileExistsError:
            raise TableFileExistsError(_exists(self.path)) from None
        except FileNotFoundError:
            raise InvalidArgumentError(_no_directory(self.path)) from None
        except PATH_ERRORS as error:
            raise InvalidArgumentError(_cannot_write(self.path, error)) from None
        try:
            try:
                remaining = memoryview(content)
                while remaining:
                    remaining = remaining[os.write(descriptor, remaining) :]
            finally:
                os.close(descriptor)
        except BaseException as error:
            # Nothing but this call could have made the file.
            self.path.unlink(missing_ok=True)
            if isinstance(error, OSError):
                raise InvalidArgumentError(_cannot_write(self.path, error)) from None
            raise
        return WrittenTable(self.path, len(table))


@dataclass(frozen=True)
class WrittenTable:
    """A table written to a file: what the response's text and record gain."""

    path: Path
    row_count: int

    def lines(self) -> list[str]:
        """The lines the text gains, each within the text's width.

        One line with the path when it fits. Otherwise the path takes a line of
        its own, and a path that is too long even for that is left to the record.
        """
        rows = plural(self.row_count, "row", "rows")
        line = f"Full table written to {self.path} ({rows})."
        if len(line) <= MAX_WIDTH:
            return [line]
        if len(str(self.path)) <= MAX_WIDTH:
            return [f"Full table written ({rows}) to:", str(self.path)]
        return [
            f"Full table written ({rows}); its path is table_file.path in the "
            "structured content."
        ]

    def record(self) -> dict[str, object]:
        """What the record gains under RECORD_KEY, as RECORD_SCHEMA describes."""
        return {"path": str(self.path), "row_count": self.row_count}


# The key the record of a call with table_path gains, and its JSON Schema. A
# client that shows the model only the structured content, such as Claude
# Code, shows it this way where the table went.
RECORD_KEY = "table_file"
RECORD_SCHEMA = {
    "type": "object",
    "description": "Only in the response of a call with table_path: the file the "
    "full table was written to and its row count. The record a Python caller gets "
    "has no such field.",
    "properties": {
        "path": {
            "type": "string",
            "description": "The absolute path of the file written.",
        },
        "row_count": {
            "type": "integer",
            "minimum": 0,
            "description": "The rows written, every row of the full table.",
        },
    },
    "required": ["path", "row_count"],
    "additionalProperties": False,
}


def table_file(name: str, table_path: object) -> TableFile:
    """The file table_path names, checked before the analysis runs.

    A leading ~ is the home directory. The path is checked as given, not where
    a symbolic link at its end leads, so a link, dangling or not, is refused.

    Raises:
        InvalidArgumentError: table_path isn't a string, has an extension
            other than .csv, .parquet or .pq, leads into a directory that
            doesn't exist or that the MCP server may not write in (also a
            read-only volume), to a directory, or somewhere the operating
            system refuses, such as a name over 255 characters.
        TableFileExistsError: a file, or a symbolic link, is already there.
    """
    table_path = path_argument(
        name,
        "table_path",
        table_path,
        what=f"the path of a new {extensions(WRITABLE)} file",
        nullable=True,
    )
    file_format = _format_of(table_path)
    try:
        path = _unresolved(table_path)
        _check_free(path, file_format)
        if not path.parent.is_dir():
            raise InvalidArgumentError(_no_directory(path))
        _check_writable(path)
    except PATH_ERRORS as error:
        if isinstance(error, InvalidArgumentError | TableFileExistsError):
            raise
        raise InvalidArgumentError(_refused(table_path, error)) from None
    return TableFile(path, file_format)


def _unresolved(table_path: str) -> Path:
    """The absolute path with its directory resolved and its last name as
    given, since resolving a link there would hide it."""
    given = Path.cwd() / Path(table_path).expanduser()
    return given.parent.resolve() / given.name


def _check_free(path: Path, file_format: FileFormat) -> None:
    """Raise unless nothing is at the path, a link included."""
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(status.st_mode):
        raise TableFileExistsError(_link(path))
    if stat.S_ISDIR(status.st_mode):
        raise InvalidArgumentError(
            f"table_path leads to {path}, which is a directory. Pass the path of "
            f"a new file, such as {path / ('table' + file_format.extension)}."
        )
    raise TableFileExistsError(_exists(path))


def _check_writable(path: Path) -> None:
    """Raise when the directory is on a read-only volume or the MCP server may
    not write in it, wherever the operating system says so before a write."""
    directory = path.parent
    if hasattr(os, "statvfs") and os.statvfs(directory).f_flag & os.ST_RDONLY:
        raise InvalidArgumentError(
            f"table_path leads to {path}, but {directory} is on a read-only volume, "
            "so no file can be written there. Pass a path on a volume the MCP "
            "server can write to."
        )
    if not os.access(directory, os.W_OK | os.X_OK):
        raise InvalidArgumentError(_not_allowed(path))


def _format_of(table_path: str) -> FileFormat:
    file_format = format_of(table_path, WRITABLE)
    if file_format is not None:
        return file_format
    raise InvalidArgumentError(
        f"table_path {table_path!r} has {extension_text(table_path)}, which the MCP "
        f"server can't write. It writes {extensions(WRITABLE)} files, in upper or "
        "lower case: CSV for any reader, Parquet to keep the dtypes. Pass a path "
        "that ends in one of them."
    )


def _cells(table: pd.DataFrame) -> pd.DataFrame:
    """The table as written: the index as the first columns when it holds more
    than the row numbers 0, 1, 2, ..., and lists and mappings as JSON text."""
    index = table.index
    if (
        index.nlevels > 1
        or index.name is not None
        or not index.equals(pd.RangeIndex(len(table)))
    ):
        table = table.reset_index()
    else:
        table = table.reset_index(drop=True)
    nested = [
        column
        for column in table.columns
        if table[column].dtype == object and table[column].map(_is_nested).any()
    ]
    if not nested:
        return table
    table = table.copy()
    for column in nested:
        table[column] = table[column].map(_json_text)
    return table


def _is_nested(value: object) -> bool:
    return isinstance(value, list | tuple | dict | set | frozenset)


def _json_text(value: object) -> object:
    """A list or a mapping as JSON text, as the record shows it; any other value
    as it is."""
    if not _is_nested(value):
        return value
    return json.dumps(row_id_json(value), ensure_ascii=False, allow_nan=False)


def _encode(table: pd.DataFrame, file_format: FileFormat) -> bytes:
    """The file's content. A missing value is an empty CSV field, which the MCP
    server reads back as missing, and a Parquet null. An empty string is the
    quoted CSV field "", which it reads back as an empty string."""
    if file_format.kind == "CSV":
        return _csv_text(table).encode("utf-8")
    buffer = io.BytesIO()
    _as_arrow_types(table).to_parquet(buffer, index=False)
    return buffer.getvalue()


def _csv_text(table: pd.DataFrame) -> str:
    """The table as CSV text, with each empty string written as "", which
    pandas leaves as an empty field, the way a missing value is written.

    The empty strings are replaced by a marker that no other cell holds, which
    is then replaced by "" in the text. A table whose cells hold every marker
    tried is written with empty fields.
    """
    empties: list[tuple[int, np.ndarray]] = []
    for position in range(table.shape[1]):
        column = table.iloc[:, position]
        if column.dtype == object or pd.api.types.is_string_dtype(column.dtype):
            empty = (column == "").to_numpy(dtype=bool, na_value=False)
            if empty.any():
                empties.append((position, empty))
    if not empties:
        return table.to_csv(index=False, lineterminator="\n")
    count = sum(int(empty.sum()) for _, empty in empties)
    marked = table.copy()
    for length in range(1, _MARKER_TRIES + 1):
        marker = _EMPTY * length
        for position, empty in empties:
            marked.isetitem(position, marked.iloc[:, position].mask(empty, marker))
        text = marked.to_csv(index=False, lineterminator="\n")
        if text.count(marker) == count:
            return text.replace(marker, '""')
    return table.to_csv(index=False, lineterminator="\n")


def _as_arrow_types(table: pd.DataFrame) -> pd.DataFrame:
    """The table with each column Arrow can't store as one type, such as numbers
    mixed with text, as text, the way the CSV shows it."""
    mixed = []
    for column in table.columns:
        if table[column].dtype != object:
            continue
        try:
            pa.array(table[column], from_pandas=True)
        except _ARROW_ERRORS:
            mixed.append(column)
    if not mixed:
        return table
    table = table.copy()
    for column in mixed:
        values = table[column]
        table[column] = values.map(str).where(values.notna(), None)
    return table


def _exists(path: Path) -> str:
    return (
        f"table_path leads to {path}, where a file already exists, and the MCP "
        f"server never overwrites a file. Pass a path where no file exists yet, "
        f"such as {_free_path(path)}, or read the file that is there."
    )


def _link(path: Path) -> str:
    return (
        f"table_path leads to {path}, which is a symbolic link, and the MCP server "
        "never writes through one, even to a file that doesn't exist yet. Pass the "
        f"path of a new file, such as {_free_path(path)}."
    )


def _refused(table_path: str, error: BaseException) -> str:
    return (
        f"table_path {shown(table_path)} can't be used: {os_cause(error)}. Pass "
        "the path of a new file in a directory the MCP server can write to, "
        f"absolute or relative to the MCP server's working directory, {Path.cwd()}."
    )


def _cannot_write(path: Path, error: BaseException) -> str:
    return (
        f"The MCP server couldn't write {path}: {os_cause(error)}. The partial file "
        "was removed. Free up space or pass a path in a directory it can write "
        "to, then call again."
    )


def _not_storable(file: "TableFile", error: BaseException) -> str:
    kind = file.file_format.kind
    other = ".csv" if kind == "Parquet" else ".parquet"
    return (
        f"The full table can't be stored as {kind} ({type(error).__name__}: "
        f"{str(error).strip()[:200]}). Nothing was written. Pass a table_path that "
        f"ends in {other}, or read the structured content."
    )


def _free_path(path: Path) -> Path:
    """The first of name-2, name-3, ... with the same extension that is free."""
    name = path.name
    lower = name.lower()
    extension = next(f.extension for f in WRITABLE if lower.endswith(f.extension))
    stem, suffix = name[: -len(extension)], name[-len(extension) :]
    number = 2
    while (path.parent / f"{stem}-{number}{suffix}").exists():
        number += 1
    return path.parent / f"{stem}-{number}{suffix}"


def _no_directory(path: Path) -> str:
    return (
        f"table_path leads to {path}, but its directory, {path.parent}, doesn't "
        "exist, and the MCP server doesn't create directories. Create it first, "
        "or pass a path in an existing directory, absolute or relative to the MCP "
        f"server's working directory, {Path.cwd()}."
    )


def _not_allowed(path: Path) -> str:
    return (
        f"table_path leads to {path}, but the MCP server may not write files in "
        f"{path.parent}. Pass a path in a directory it can write to."
    )
