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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from agent_eda._local_files import (
    WRITABLE,
    FileFormat,
    extension_text,
    extensions,
    format_of,
    path_argument,
    resolve,
)
from agent_eda._text import plural
from agent_eda.errors import InvalidArgumentError, TableFileExistsError
from agent_eda.results import row_id_json

# What pyarrow raises for a column it can't store as one Arrow type, such as
# numbers mixed with text.
_ARROW_ERRORS = (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError)


@dataclass(frozen=True)
class TableFile:
    """A checked table path: the absolute path and the format to write."""

    path: Path
    file_format: FileFormat

    def write(self, table: pd.DataFrame) -> str:
        """Write the table and return the line the response text gains.

        Raises:
            TableFileExistsError: a file appeared at the path after the check.
            InvalidArgumentError: the directory went away after the check, or
                the server may not write there.
        """
        content = _encode(_cells(table), self.file_format)
        try:
            with self.path.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            raise TableFileExistsError(_exists(self.path)) from None
        except FileNotFoundError:
            raise InvalidArgumentError(_no_directory(self.path)) from None
        except PermissionError:
            raise InvalidArgumentError(_not_allowed(self.path)) from None
        return (
            f"Full table written to {self.path} ({plural(len(table), 'row', 'rows')})."
        )


def table_file(name: str, table_path: object) -> TableFile:
    """The file table_path names, checked before the analysis runs.

    Raises:
        InvalidArgumentError: table_path isn't a string, has an extension
            other than .csv, .parquet or .pq, or leads into a directory that
            doesn't exist, or to a directory.
        TableFileExistsError: a file already exists at the path.
    """
    table_path = path_argument(
        name,
        "table_path",
        table_path,
        what=f"the path of a new {extensions(WRITABLE)} file",
        nullable=True,
    )
    file_format = _format_of(table_path)
    path = resolve(table_path)
    if path.is_dir():
        raise InvalidArgumentError(
            f"table_path leads to {path}, which is a directory. Pass the path of "
            f"a new file, such as {path / ('table' + file_format.extension)}."
        )
    if path.exists() or path.is_symlink():
        raise TableFileExistsError(_exists(path))
    if not path.parent.is_dir():
        raise InvalidArgumentError(_no_directory(path))
    return TableFile(path, file_format)


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
    server reads back as missing, and a Parquet null."""
    if file_format.kind == "CSV":
        return table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    _as_arrow_types(table).to_parquet(buffer, index=False)
    return buffer.getvalue()


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
