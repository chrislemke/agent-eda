"""Reading the file a data path names into a DataFrame.

The MCP server reads every data path through a DataReader. It reads only local
files: a path is resolved against the working directory and opened by this
module, so a path that looks like a URL never reaches pandas' URL handling and
can't start a network read.
"""

import warnings
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from agent_eda._columns import closest_names
from agent_eda._local_files import (
    READABLE,
    FileFormat,
    extension_text,
    extensions,
    format_of,
    resolve,
)
from agent_eda._text import join_alternatives
from agent_eda.errors import (
    DataFileError,
    DataFileNotFoundError,
    UnsupportedDataFormatError,
)

# The DataFrames a DataReader keeps, the most recently read ones.
CACHED_FILES = 2


# What the parsers raise on a file they can't read: pandas' ParserError,
# EmptyDataError and UnicodeDecodeError, pyarrow's ArrowInvalid (all
# ValueErrors), gzip's BadGzipFile and a permission error (OSErrors), a
# truncated gzip stream (EOFError) and a corrupt one (zlib.error).
_PARSE_ERRORS = (ValueError, OSError, EOFError, zlib.error)

type _Key = tuple[Path, int, int]


class DataReader:
    """Reads data paths, keeping the most recently read DataFrames.

    A file is read again when its modification time or size changes. The
    analyses treat their data as read-only, so a kept DataFrame stays valid
    across calls.
    """

    def __init__(self) -> None:
        self._frames: OrderedDict[_Key, pd.DataFrame] = OrderedDict()

    def read(self, data_path: str) -> pd.DataFrame:
        """The DataFrame in the file data_path names.

        Raises:
            UnsupportedDataFormatError: the extension isn't a supported one.
            DataFileNotFoundError: there is no file at the path.
            DataFileError: the parser can't read the file.
        """
        file_format = _format_of(data_path)
        path = resolve(data_path)
        if not path.is_file():
            raise DataFileNotFoundError(_not_found(path))
        status = path.stat()
        key = (path, status.st_mtime_ns, status.st_size)
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        data = _read(path, file_format)
        # Drop an earlier version of the file, whose key can't match again.
        for old in [old for old in self._frames if old[0] == path]:
            del self._frames[old]
        self._frames[key] = data
        while len(self._frames) > CACHED_FILES:
            self._frames.popitem(last=False)
        return data


def supported_extensions() -> str:
    """The extensions the reader reads, as text for a message."""
    return extensions(READABLE)


def _format_of(data_path: str) -> FileFormat:
    file_format = format_of(data_path, READABLE)
    if file_format is not None:
        return file_format
    raise UnsupportedDataFormatError(
        f"data_path {data_path!r} has {extension_text(data_path)}, which the MCP "
        f"server can't read. It reads {supported_extensions()} files, in upper or "
        "lower case. Save the data as CSV, TSV or Parquet and pass that file's path."
    )


def _read(path: Path, file_format: FileFormat) -> pd.DataFrame:
    try:
        with path.open("rb") as handle:
            if file_format.kind == "Parquet":
                return _read_parquet(handle)
            return pd.read_csv(
                handle,
                sep="\t" if file_format.kind == "TSV" else ",",
                compression="gzip" if file_format.gzip else None,
                encoding="utf-8",
                keep_default_na=False,
                na_values=[""],
            )
    except _PARSE_ERRORS as error:
        detail = str(error).strip().rstrip(".") or type(error).__name__
        gzipped = "gzipped " if file_format.gzip else ""
        raise DataFileError(
            f"Can't read {path} as {gzipped}{file_format.kind}: {detail}. Check "
            f"that the file is {_expected(file_format)}, or pass the path of a "
            "file that is."
        ) from None


def _read_parquet(handle: BinaryIO) -> pd.DataFrame:
    with warnings.catch_warnings():
        # pyarrow 14 builds the DataFrame from pandas internals that pandas 2.2
        # deprecates, so every read warns, and a read with a time zone warns
        # twice. The warnings say nothing about the data or the call.
        warnings.filterwarnings(
            "ignore",
            message="(Passing a BlockManager to DataFrame|DatetimeTZBlock) is "
            "deprecated",
            category=DeprecationWarning,
        )
        return pd.read_parquet(handle)


def _expected(file_format: FileFormat) -> str:
    if file_format.kind == "Parquet":
        return "a Parquet file"
    separator = "tabs" if file_format.kind == "TSV" else "commas"
    compressed = ", gzip-compressed," if file_format.gzip else ""
    return (
        f"UTF-8 text{compressed} with a header row and fields separated by {separator}"
    )


def _not_found(path: Path) -> str:
    """Say where the path led, whether its directory exists and which file
    names in it come closest."""
    directory = path.parent
    fix = (
        "Pass the path of an existing file, absolute or relative to the MCP "
        f"server's working directory, {Path.cwd()}."
    )
    if path.is_dir():
        return f"data_path leads to {path}, which is a directory, not a file. {fix}"
    tried = f"There is no file at {path}."
    if not directory.is_dir():
        return f"{tried} Its directory, {directory}, doesn't exist either. {fix}"
    close = closest_names(path.name, _file_names(directory))
    if close:
        names = join_alternatives([repr(name) for name in close])
        return f"{tried} Its directory exists. Did you mean {names}? {fix}"
    return f"{tried} Its directory exists but holds no file with a similar name. {fix}"


def _file_names(directory: Path) -> list[str]:
    """The names of the files in directory, or none if it can't be listed."""
    try:
        return sorted(entry.name for entry in directory.iterdir() if entry.is_file())
    except OSError:
        return []
