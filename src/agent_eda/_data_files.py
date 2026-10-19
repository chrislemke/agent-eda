"""Reading the file a data path names into a DataFrame.

The MCP server reads every data path through a DataReader. It reads only local
files: a path is resolved against the working directory and opened by this
module, so a path that looks like a URL never reaches pandas' URL handling and
can't start a network read.
"""

import csv
import gzip
import warnings
import zlib
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import pandas as pd

from agent_eda._local_files import (
    READABLE,
    FileFormat,
    extension_text,
    extensions,
    format_of,
    resolve,
)
from agent_eda._parse import read_texts
from agent_eda._text import closest_names, join_alternatives
from agent_eda.errors import (
    DataFileError,
    DataFileNotFoundError,
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
    "row may end in an empty field but has no other field past the header's. A "
    "header name that repeats stays repeated, and most analyses then ask you to "
    "rename it in the file."
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
# ValueErrors), gzip's BadGzipFile and a permission error (OSErrors), a
# truncated gzip stream (EOFError), a corrupt one (zlib.error) and the csv
# module's error on a field it can't hold.
_PARSE_ERRORS = (ValueError, OSError, EOFError, zlib.error, csv.Error)

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
            DataFileError: the parser can't read the file.
        """
        file_format = _format_of(data_path)
        path = resolve(data_path)
        if not path.is_file():
            raise DataFileNotFoundError(_not_found(path))
        status = path.stat()
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
        if file_format.kind == "Parquet":
            with path.open("rb") as handle:
                return _read_parquet(handle)
        return _read_csv(path, file_format)
    except DataFileError:
        raise
    except _PARSE_ERRORS as error:
        detail = str(error).strip().rstrip(".") or type(error).__name__
        gzipped = "gzipped " if file_format.gzip else ""
        raise DataFileError(
            f"Can't read {path} as {gzipped}{file_format.kind}: {detail}. Check "
            f"that the file is {_expected(file_format)}, or pass the path of a "
            "file that is."
        ) from None


def _read_csv(path: Path, file_format: FileFormat) -> pd.DataFrame:
    """A CSV or TSV file with the values it holds.

    Every column is typed once, over the whole file. No column becomes the
    index, so a trailing delimiter adds no column and shifts no value. Only an
    empty field is missing. The empty lines of a one-column file are rows with
    a missing value. A column whose numbers include a code with a leading zero
    stays text, and repeated header names stay repeated.
    """
    header = _header(path, file_format)
    options: dict[str, Any] = {
        "sep": "\t" if file_format.kind == "TSV" else ",",
        "compression": "gzip" if file_format.gzip else None,
        "encoding": "utf-8",
        "index_col": False,
        "low_memory": False,
        "keep_default_na": False,
        "na_values": [""],
        # An empty line is a row of a one-column file, and no row of any other.
        "skip_blank_lines": len(header) > 1,
    }
    try:
        frame = _parse(path, options)
        labels = _labels(header, list(frame.columns))
    except (pd.errors.ParserWarning, pd.errors.ParserError) as error:
        if not _has_extra_fields(error):
            raise
        # Some row has more fields than the header, which pandas either drops
        # or rejects according to the first row. A trailing empty field is no
        # data, so read the file again with room for the widest row.
        width, header_lines = _check_extra_fields(
            path, file_format, options["sep"], len(header)
        )
        options["header"] = None
        options["skiprows"] = header_lines
        options["names"] = list(range(width))
        frame = _parse(path, options).iloc[:, : len(header)]
        labels = _labels(header, None)
    _keep_codes_as_text(path, options, frame)
    frame.columns = pd.Index(labels)
    return frame


def _parse(path: Path, options: dict[str, Any], **more: Any) -> pd.DataFrame:
    """read_csv on the file, with a header row that has fewer names than a row
    has fields raised as an error, not warned about."""
    with warnings.catch_warnings(), path.open("rb") as handle:
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
def _text(path: Path, file_format: FileFormat) -> Generator[TextIO]:
    """The file as UTF-8 text for the csv module, which leaves line ends alone."""
    if file_format.gzip:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield handle
    else:
        with path.open("rt", encoding="utf-8-sig", newline="") as handle:
            yield handle


def _header(path: Path, file_format: FileFormat) -> list[str]:
    """The names in the header row, which is the first line that isn't blank.
    There are none in an empty file."""
    sep = "\t" if file_format.kind == "TSV" else ","
    with _text(path, file_format) as handle:
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
    path: Path, file_format: FileFormat, sep: str, width: int
) -> tuple[int, int]:
    """The most fields a row has, and the lines up to the end of the header row,
    once no row has an extra field that isn't empty.

    Raises:
        DataFileError: a row has a non-empty field beyond the header's.
    """
    widest = width
    with _text(path, file_format) as handle:
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
                    f"Can't read {path} as {_kind(file_format)}: line "
                    f"{reader.line_num} has {len(row)} fields, and {extra} of "
                    f"those past the header's {width} names aren't empty. Check "
                    f"that every row has at most {width} fields, or add the "
                    "missing names to the header row, and pass the path again."
                )
            widest = max(widest, len(row))
    return widest, header_lines


def _kind(file_format: FileFormat) -> str:
    gzipped = "gzipped " if file_format.gzip else ""
    return f"{gzipped}{file_format.kind}"


def _keep_codes_as_text(
    path: Path, options: dict[str, Any], frame: pd.DataFrame
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
    if numeric and _may_hold_codes(path, options):
        with warnings.catch_warnings(), path.open("rb") as handle:
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
        text = _parse(path, options, usecols=codes, dtype=str)
        for place, position in enumerate(codes):
            frame.isetitem(position, text.iloc[:, place])


def _may_hold_codes(path: Path, options: dict[str, Any]) -> bool:
    """Whether any field of the file starts like a code with a leading zero:
    a 0 and a digit after a line break, the delimiter, a quote, or one of those
    and a sign, quote or space.

    A quick search of the bytes, so that a file of ordinary numbers is read once.
    It may say yes for a field that isn't a number, and never says no for a code.
    """
    delimiter = ord(options["sep"])
    opener = gzip.open if options["compression"] == "gzip" else open
    with opener(path, "rb") as handle:
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
