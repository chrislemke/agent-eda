"""Reading the file a data path names into a DataFrame.

The MCP server reads every data path through a DataReader. It reads only local
files: a path is resolved against the working directory and opened by this
module, so a path that looks like a URL never reaches pandas' URL handling and
can't start a network read.
"""

import codecs
import csv
import gzip
import io
import stat
import warnings
import zlib
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agent_eda._mcp.local_files import (
    PATH_ERRORS,
    READABLE,
    FileFormat,
    extension_text,
    extensions,
    format_names,
    format_of,
    resolve,
    shown,
)
from agent_eda._parse import read_texts
from agent_eda._text import closest_names, join_alternatives, quoted_path
from agent_eda._warnings import scoped_filters
from agent_eda._wording import names_nothing, os_cause
from agent_eda.errors import (
    DataFileError,
    DataFileNotFoundError,
    InvalidArgumentError,
    UnsupportedDataFormatError,
)

# How the reader types a CSV or TSV file, and a Parquet one, as the data_path
# description and the MCP instructions say it.
CSV_RULES = (
    "A CSV or TSV column is read as numbers or booleans (true or false, in any "
    "case) only when every non-empty field parses as one, and as text otherwise. "
    "A whole number written with a leading zero, such as 01234 or 007, is a code, "
    "so its column stays text; inf and -inf read as numbers. Blank lines are "
    "skipped, except in a file with one column, where each is a missing value. A "
    'quoted empty field ("") is an empty string, not a missing value, and its '
    "column stays text. A row may end in an empty field but has no other field "
    "past the header's. A header name that repeats stays repeated, and most "
    "analyses then ask you to rename it in the file."
)
PARQUET_RULES = (
    "A Parquet file keeps the dtypes and the index that pandas stored in it. One "
    "written without pandas metadata, such as by Spark or DuckDB, gets pandas' "
    "default dtypes for its Parquet types (an integer column with nulls becomes "
    "float) and a 0-based index."
)

# The DataFrames a DataReader keeps, the most recently read ones.
CACHED_FILES = 2


# What the parsers raise on a file they can't read: pandas' ParserError,
# EmptyDataError and UnicodeDecodeError, pyarrow's ArrowInvalid (all
# ValueErrors) and the rest of its exceptions, gzip's BadGzipFile and the
# operating system's refusals (OSErrors), a truncated gzip stream (EOFError), a
# corrupt one (zlib.error), the csv module's error on a field it can't hold, and
# what pandas' reader of a Parquet file raises for pandas metadata that doesn't
# fit the file (KeyError, TypeError, IndexError, AttributeError).
_PARSE_ERRORS = (
    ValueError,
    OSError,
    EOFError,
    zlib.error,
    csv.Error,
    pa.ArrowException,
    KeyError,
    TypeError,
    IndexError,
    AttributeError,
    OverflowError,
)

# Two control bytes that stand in for a quoted empty field while pandas reads
# the file, since pandas can't tell it from an unquoted one.
_EMPTY_BYTES = tuple(range(1, 32))

# A file and its format: the same file read under another extension, such as a
# symlink with another name, is a different frame.
type _Key = tuple[Path, int, int, FileFormat]

# The rows pandas types at a time when it looks for leading-zero codes.
_CHUNK_ROWS = 100_000

# A number that starts with 0 and another digit: what _parse calls a code when
# it is a whole number, such as 007 or 01234.
_ZERO_PREFIX = r"\s*[+-]?0[0-9]"

# How much of the file is searched at a time, and how much of the previous part
# is searched again, so a field at the edge between two parts is found.
_SEARCH_BYTES = 1 << 24
_OVERLAP_BYTES = 64


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
            DataFileError: the path leads somewhere that isn't a regular file,
                the operating system refuses it, or the parser can't read the
                file.
            InvalidArgumentError: the path holds a NUL character.
        """
        file_format = _format_of(data_path)
        path, status = _stat(data_path)
        key = (path, status.st_mtime_ns, status.st_size, file_format)
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


def _format_of(data_path: str) -> FileFormat:
    file_format = format_of(data_path, READABLE)
    if file_format is not None:
        return file_format
    raise UnsupportedDataFormatError(
        f"data_path {quoted_path(data_path)} has {extension_text(data_path)}, which the MCP "
        f"server can't read. It reads {extensions(READABLE)} files, in upper or "
        f"lower case. Save the data as {format_names(READABLE)} and pass that "
        "file's path."
    )


def _stat(data_path: str) -> tuple[Path, Any]:
    """The resolved path of a data path and its status, once it is a regular
    file.

    Raises:
        DataFileNotFoundError: there is nothing at the path, or a directory.
        DataFileError: the path isn't a regular file, or the operating system
            refuses it.
        InvalidArgumentError: the path holds a NUL character.
    """
    try:
        path = resolve(data_path)
        status = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        raise DataFileNotFoundError(_not_found(resolve(data_path))) from None
    except ValueError:
        raise InvalidArgumentError(_refused(data_path, "NUL")) from None
    except PATH_ERRORS as error:
        if names_nothing(error):
            raise DataFileNotFoundError(_not_found(resolve(data_path))) from None
        raise DataFileError(_refused(data_path, os_cause(error))) from None
    if stat.S_ISDIR(status.st_mode):
        raise DataFileNotFoundError(_not_found(path))
    if not stat.S_ISREG(status.st_mode):
        raise DataFileError(
            f"data_path leads to {path}, which isn't a regular file (it is a named "
            "pipe, a device or a socket), so the MCP server doesn't open it. Pass "
            f"the path of a {format_names(READABLE)} file."
        )
    return path, status


def _refused(data_path: str, cause: str) -> str:
    if cause == "NUL":
        cause = "it holds a NUL character, which no file name can"
    return (
        f"data_path {shown(data_path)} can't be read: {cause}. Pass the path of a "
        "file the MCP server can read, absolute or relative to the MCP server's "
        f"working directory, {Path.cwd()}."
    )


def _read(path: Path, file_format: FileFormat) -> pd.DataFrame:
    try:
        if file_format.is_parquet:
            with path.open("rb") as handle:
                return _read_parquet(handle, path)
        return _read_csv(path, file_format)
    except DataFileError:
        raise
    except _PARSE_ERRORS as error:
        if isinstance(error, OSError) and error.errno is not None:
            # The operating system refused the file; it isn't the file's content.
            raise DataFileError(
                f"Can't read {path} as {file_format.label}: "
                f"{os_cause(error)}. Give the user that runs the MCP server read "
                "permission on the file, or pass the path of a file it can read."
            ) from None
        detail = _detail(error, file_format)
        raise DataFileError(
            f"Can't read {path} as {file_format.label}: {detail}. Check "
            f"that the file is {_expected(file_format)}, or pass the path of a "
            "file that is."
        ) from None


def _detail(error: Exception, file_format: FileFormat) -> str:
    """What the parser said, or for an error from broken Parquet metadata what
    that means."""
    text = str(error).strip().rstrip(".") or type(error).__name__
    if file_format.is_parquet and isinstance(
        error, KeyError | TypeError | IndexError | AttributeError
    ):
        return (
            "the pandas metadata stored in the file doesn't match its columns "
            f"({type(error).__name__}: {text})"
        )
    return text


def _read_csv(path: Path, file_format: FileFormat) -> pd.DataFrame:
    """A CSV or TSV file with the values it holds.

    Every column is typed once, over the whole file. No column becomes the
    index, so a trailing delimiter adds no column and shifts no value. Only an
    unquoted empty field is missing; a quoted one, "", is an empty string, and
    its column is text. The empty lines of a one-column file are rows with a
    missing value. A column whose numbers include a code with a leading zero
    stays text, and repeated header names stay repeated.
    """
    sep = file_format.separator
    source = _Source(path, file_format.gzip)
    token = _mark_quoted_empties(source, sep)
    if token is not None:
        source = _Source(path, file_format.gzip, token.content)
    header = _header(source, sep)
    options: dict[str, Any] = {
        "sep": sep,
        "encoding": "utf-8",
        "index_col": False,
        "low_memory": False,
        "keep_default_na": False,
        "na_values": [""],
        # An empty line is a row of a one-column file, and no row of any other.
        "skip_blank_lines": len(header) > 1,
    }
    try:
        frame = _parse(source, options)
        labels = _labels(header, list(frame.columns))
    except (pd.errors.ParserWarning, pd.errors.ParserError) as error:
        if not _has_extra_fields(error):
            raise
        # Some row has more fields than the header, which pandas either drops
        # or rejects according to the first row. A trailing empty field is no
        # data, so read the file again with room for the widest row.
        width, header_lines = _check_extra_fields(source, file_format, sep, len(header))
        options["header"] = None
        options["skiprows"] = header_lines
        options["names"] = list(range(width))
        frame = _parse(source, options).iloc[:, : len(header)]
        labels = _labels(header, None)
    _keep_codes_as_text(source, options, frame)
    if token is not None:
        _restore_empties(frame, token.text)
        labels = [
            f"Unnamed: {place}" if label == token.text else label
            for place, label in enumerate(labels)
        ]
    if len(labels) == 1 and ";" in str(labels[0]):
        raise DataFileError(_semicolons(path, file_format, str(labels[0])))
    frame.columns = pd.Index(labels)
    return frame


def _semicolons(path: Path, file_format: FileFormat, name: str) -> str:
    separator = file_format.separator_word
    return (
        f"Can't read {path} as {file_format.label}: the only column is named "
        f"{name!r}, which looks like a header whose fields are separated by "
        f"semicolons, but the MCP server separates fields by {separator}. Save the "
        f"file with {separator}, or, if the column really has that name, rename it "
        "in the file."
    )


@dataclass(frozen=True)
class _Source:
    """The text of a CSV or TSV file: read from the path, decompressed if
    gzipped, or the bytes held in memory once quoted empty fields were marked."""

    path: Path
    gzipped: bool
    content: bytes | bytearray | None = None

    def open(self) -> BinaryIO:
        if self.content is not None:
            return io.BytesIO(self.content)
        if self.gzipped:
            # A GzipFile reads like a binary file; typeshed doesn't say so.
            return cast(BinaryIO, gzip.open(self.path, "rb"))
        return self.path.open("rb")


@dataclass(frozen=True)
class _Marked:
    """The file's bytes with each quoted empty field replaced by `text`."""

    content: bytearray
    text: str


def _mark_quoted_empties(source: _Source, sep: str) -> _Marked | None:
    """The file's bytes with each field that is just "" replaced by two control
    bytes, or None when there is no such field.

    pandas reads a quoted empty field as missing, as it does an unquoted one,
    but the first is an empty string. After the marker is read as text, the
    reader sets those cells to "". The two bytes are ones the file doesn't hold
    in a row, so a marker is never a value of the file.

    A field is just "" when the quote that opens it comes after an even number
    of quotes, so it opens a field instead of closing one or escaping a quote,
    and a delimiter, line break or the file's edge is on both sides. The file is
    held in memory once, and only when some "" is in it.
    """
    if not _holds_quote_pair(source):
        return None
    with source.open() as handle:
        raw = bytearray(handle.read())
    marker = next(
        (bytes([byte, byte]) for byte in _EMPTY_BYTES if bytes([byte] * 2) not in raw),
        None,
    )
    if marker is None:
        return None
    data = np.frombuffer(raw, dtype=np.uint8)
    quotes = np.flatnonzero(data == ord('"'))
    opens = quotes[:-1:2]
    opens = opens[quotes[1::2] == opens + 1]
    breaks = np.array([ord(sep), ord("\n"), ord("\r")], dtype=np.uint8)
    size = len(data)
    before = np.where(opens > 0, data[np.maximum(opens - 1, 0)], ord("\n"))
    after = np.where(opens + 2 < size, data[np.minimum(opens + 2, size - 1)], ord("\n"))
    after_bom = (opens == len(codecs.BOM_UTF8)) & (
        bytes(raw[: len(codecs.BOM_UTF8)]) == codecs.BOM_UTF8
    )
    opens = opens[(np.isin(before, breaks) | after_bom) & np.isin(after, breaks)]
    if not len(opens):
        return None
    data[opens] = marker[0]
    data[opens + 1] = marker[1]
    return _Marked(raw, marker.decode("ascii"))


def _holds_quote_pair(source: _Source) -> bool:
    """Whether the file has two quotes in a row anywhere."""
    with source.open() as handle:
        kept = b""
        while block := handle.read(_SEARCH_BYTES):
            if b'""' in kept + block:
                return True
            kept = block[-1:]
    return False


def _restore_empties(frame: pd.DataFrame, marker: str) -> None:
    """Set each cell that holds the marker to the empty string."""
    for position, dtype in enumerate(frame.dtypes):
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            continue
        column = frame.iloc[:, position]
        marked = column == marker
        if marked.any():
            frame.isetitem(position, column.mask(marked, ""))


def _parse(source: _Source, options: dict[str, Any], **more: Any) -> pd.DataFrame:
    """read_csv on the file, with a header row that has fewer names than a row
    has fields raised as an error, not warned about."""
    with scoped_filters(), source.open() as handle:
        warnings.filterwarnings(
            "error",
            message="Length of header or names does not match length of data",
            category=pd.errors.ParserWarning,
        )
        return pd.read_csv(handle, **options, **more)


def _has_extra_fields(error: Exception) -> bool:
    """Whether pandas stopped because a row has more fields than the header."""
    text = str(error)
    return "loss of data" in text or ("Expected" in text and "fields in line" in text)


@contextmanager
def _text(source: _Source) -> Generator[TextIO]:
    """The file as UTF-8 text for the csv module, which leaves line ends alone."""
    with io.TextIOWrapper(source.open(), encoding="utf-8-sig", newline="") as handle:
        yield handle


def _header(source: _Source, sep: str) -> list[str]:
    """The names in the header row, which is the first line that isn't blank.
    There are none in an empty file."""
    with _text(source) as handle:
        return next((row for row in csv.reader(handle, delimiter=sep) if row), [])


def _labels(header: list[str], parsed: list[Any] | None) -> list[Any]:
    """The column labels: what pandas made of each name, except that a name
    repeated in the header keeps its text. Without pandas' labels, an empty
    name is "Unnamed: " and its position, as pandas writes it."""
    labels: list[Any] = []
    for position, name in enumerate(header):
        if name and name in header[:position]:
            labels.append(name)
        elif parsed is None:
            labels.append(name or f"Unnamed: {position}")
        else:
            labels.append(parsed[position])
    return labels


def _check_extra_fields(
    source: _Source, file_format: FileFormat, sep: str, width: int
) -> tuple[int, int]:
    """The most fields a row has, and the lines up to the end of the header row,
    once no row has an extra field that isn't empty.

    Raises:
        DataFileError: a row has a non-empty field beyond the header's.
    """
    widest = width
    with _text(source) as handle:
        reader = csv.reader(handle, delimiter=sep)
        header_lines = 0
        for row in reader:
            if not header_lines:
                header_lines = reader.line_num if row else 0
                continue
            if len(row) > width and any(row[width:]):
                extra = sum(1 for field in row[width:] if field)
                # A row ends on the line the reader is at; the row's text may
                # start earlier when a quoted field holds line breaks.
                raise DataFileError(
                    f"Can't read {source.path} as {file_format.label}: line "
                    f"{reader.line_num} has {len(row)} fields, and {extra} of "
                    f"those past the header's {width} names aren't empty. Check "
                    f"that every row has at most {width} fields, or add the "
                    "missing names to the header row, and pass the path again."
                )
            widest = max(widest, len(row))
    return widest, header_lines


def _keep_codes_as_text(
    source: _Source, options: dict[str, Any], frame: pd.DataFrame
) -> None:
    """Read again as text each numeric column that holds a code with a leading
    zero, such as 01234 or 007, which parsing as a number would change."""
    numeric = [
        position
        for position, dtype in enumerate(frame.dtypes)
        if pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    ]
    codes: list[int] = []
    if numeric and _may_hold_codes(source, options):
        with scoped_filters(), source.open() as handle:
            reader = pd.read_csv(
                handle, **options, usecols=numeric, dtype=str, chunksize=_CHUNK_ROWS
            )
            for chunk in reader:
                for place, position in enumerate(numeric):
                    if position not in codes and _holds_code(chunk.iloc[:, place]):
                        codes.append(position)
                if len(codes) == len(numeric):
                    break
    if codes:
        codes.sort()
        text = _parse(source, options, usecols=codes, dtype=str)
        for place, position in enumerate(codes):
            frame.isetitem(position, text.iloc[:, place])


def _may_hold_codes(source: _Source, options: dict[str, Any]) -> bool:
    """Whether any field of the file starts like a code with a leading zero:
    a 0 and a digit after a line break, the delimiter, a quote, or one of those
    and a sign, quote or space.

    A quick search of the bytes, so that a file of ordinary numbers is read once.
    It may say yes for a field that isn't a number, and never says no for a code.
    """
    delimiter = ord(options["sep"])
    with source.open() as handle:
        kept = b""
        while block := handle.read(_SEARCH_BYTES):
            # A line break in front, so that the first field has a predecessor.
            text = np.frombuffer(b"\n\n" + kept + block + b"\n", dtype=np.uint8)
            if _has_zero_field(text, delimiter):
                return True
            kept = (kept + block)[-_OVERLAP_BYTES:]
    return False


def _has_zero_field(text: np.ndarray, delimiter: int) -> bool:
    """Whether the bytes hold a field that starts with a 0 and a digit."""
    zeros = np.flatnonzero(text[2:-1] == ord("0")) + 2
    zeros = zeros[(text[zeros + 1] >= ord("0")) & (text[zeros + 1] <= ord("9"))]
    starts = np.zeros(256, dtype=bool)
    starts[[ord("\n"), ord("\r"), delimiter]] = True
    leaders = np.zeros(256, dtype=bool)
    leaders[[ord("+"), ord("-"), ord('"'), ord(" ")]] = True
    after_start = starts[text[zeros - 1]]
    after_leader = leaders[text[zeros - 1]] & starts[text[zeros - 2]]
    return bool((after_start | after_leader).any())


def _holds_code(values: pd.Series) -> bool:
    """Whether some text in a numeric column is a whole number written with a
    leading zero, by the rule the parse rules share."""
    present = values.dropna()
    if present.empty:
        return False
    likely = present[present.str.match(_ZERO_PREFIX)]
    return bool(
        len(likely) and read_texts(likely.to_numpy(dtype=object)).leading_zero.any()
    )


def _read_parquet(handle: BinaryIO, path: Path) -> pd.DataFrame:
    names = pq.read_schema(handle).names
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        listed = join_alternatives([repr(name) for name in repeated])
        raise DataFileError(
            f"Can't read {path} as Parquet: it has more than one column named "
            f"{listed}, which a DataFrame can't hold apart. Write the file again "
            "with distinct column names."
        )
    handle.seek(0)
    with scoped_filters():
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
    if file_format.is_parquet:
        return "a Parquet file"
    separator = file_format.separator_word
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
