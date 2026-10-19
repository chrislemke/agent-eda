"""Project files: a Project saved as TOML once and loaded in later scripts.

project.save() calls write() and Project.load() calls read(). A file holds the
project's `name` at the top level, left out when there is none, since TOML has
no null, and a [data_policy] table with every field of the data policy.
write() lays it out the same way every time, one array item per line, so
equal projects give identical files on the same Python version. read() parses
with the standard library's tomllib, and write() needs no TOML library: the
values are only text, booleans and lists of text, which it writes itself.

read() requires the [data_policy] table and each of its keys, so that a file
cut short, such as by a full disk, raises instead of loading with masking off;
only `name` may be left out. A caller's mistake raises before any file
changes: InvalidArgumentError for a path or overwrite that can't work, a
target that isn't a regular file included,
ProjectFileNotFoundError for a missing file, ProjectFileExistsError for a file
that save would replace unasked, and ProjectFileError for a file that isn't a
project file, which names the line or the key at fault, or that read() can't
read: not a regular file (a named pipe or a device would block or never end),
refused by the operating system, such as without read permission, or larger
than MAX_BYTES. A file may
hold `format_version`, the version of the layout it was written in: write()
records 1, a file without the key is read as version 1, and a higher version
raises, naming it. write() replaces a file whole, through a temporary file and
a rename, and refuses a project that read() would refuse for its size. Messages
never quote the file's content, since a data file passed by mistake would put its values
in front of the agent.
"""

import codecs
import contextlib
import os
import re
import stat
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_eda._columns import closest_names
from agent_eda._registry import _value_text, python_scalar
from agent_eda._text import Surface, join_items
from agent_eda._wording import message, worded
from agent_eda.errors import (
    AgentEDAError,
    InvalidArgumentError,
    ProjectFileError,
    ProjectFileExistsError,
    ProjectFileNotFoundError,
)

# The most a project file may hold. A name and a data policy take a few KiB;
# a larger file is a data file passed by mistake.
MAX_BYTES = 1024 * 1024

# The layout version that write() records as `format_version`. A file without the
# key is version 1, and read() refuses a version above this one.
FORMAT_VERSION = 1

# The first line of every project file, for whoever opens one.
_HEADER = "# An agent-eda Project. Load it with ae.Project.load(path)."

# Sensitive columns as a project file declares them, for messages that show how;
# an MCP caller declares a data policy only in a project file.
TOML_EXAMPLE = 'sensitive_columns = ["email"]'


def _format_source(surface: Surface, *, start: bool = False) -> str:
    """Where the caller reads how a project file writes each key."""
    if surface == "python":
        return "help(ae.Project.load)"
    where = "project_path description in the MCP tool's input schema"
    return f"{'The' if start else 'the'} {where}"


def _rewrite(surface: Surface) -> str:
    """How to recover from a file that doesn't parse."""
    if surface == "python":
        return (
            "Fix the file, or write it again with project.save(path, overwrite=True) "
            "from a project built in code; help(ae.Project.load) shows the format."
        )
    return f"Fix the file; {_format_source(surface)} shows the format."


# Why read() requires the whole data policy.
_WHOLE_POLICY = (
    "A project file states its whole data policy, so that a file cut short can't "
    "leave masking off quietly"
)
# tomllib's messages end with where it stopped: "(at line 3, column 18)", or
# "(at end of document)" when the file ended first.
_AT_LINE = re.compile(
    r"(?P<reason>.*) \(at line (?P<line>\d+), column (?P<column>\d+)\)"
)
_AT_END = re.compile(r"(?P<reason>.*) \(at end of document\)")

# How a project file writes the value of each field, and the check that a
# value has that form, for messages that say how to write it. The Project's
# own validation still decides, so a field without an entry is checked there.
_FORMS: dict[str, tuple[str, Callable[[object], bool]]] = {
    "name": ('a string, such as name = "churn"', lambda value: isinstance(value, str)),
    "sensitive_columns": (
        'an array of strings, such as sensitive_columns = ["email", "name"]',
        lambda value: isinstance(value, str),
    ),
    "mask_sensitive": (
        "true or false, without quotes",
        lambda value: isinstance(value, bool),
    ),
}
# The fields whose value is an array of what their check accepts.
_ARRAYS = {"sensitive_columns"}

# The characters a TOML basic string writes with a short escape.
_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def read(
    path: object, *, keys: Sequence[str], table: str, table_keys: Sequence[str]
) -> dict[str, Any]:
    """The keys and values of a project file, laid out as a Project's fields.

    `keys` are the keys the file may hold at the top level, among them the
    required table named `table`, and `table_keys` the keys that table must
    hold.
    """
    given = _file_path("Project.load", path)
    fields = _parse(given, _read_bytes(given))
    _check_format_version(fields, given)
    _check_keys(fields, given, keys=keys, table=table, table_keys=table_keys)
    _check_table(fields, given, table=table, table_keys=table_keys)
    _check_forms(fields, given, table=table)
    return fields


def _check_format_version(fields: dict[str, Any], given: str) -> None:
    """Take `format_version` out of the fields; raise unless this version reads it.

    A file without the key is version 1, the first layout.
    """
    if "format_version" not in fields:
        return
    version = fields.pop("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProjectFileError(
            f"Project.load found {_kind(version)} for format_version at the top "
            f"level of {given!r}, but format_version takes a positive integer, "
            f"such as format_version = {FORMAT_VERSION}, or can be left out."
        )
    if version > FORMAT_VERSION:
        raise worded(
            ProjectFileError,
            lambda surface: (
                f"Project.load found format_version = {version} in {given!r}, "
                "written by a newer agent-eda, but this agent-eda reads "
                f"format_version {FORMAT_VERSION} and below. Upgrade agent-eda "
                "(pip install --upgrade agent-eda), or "
                + (
                    "save the project again with this version, "
                    "project.save(path, overwrite=True), from a project built in "
                    "code."
                    if surface == "python"
                    else "write the file again with this version's layout."
                )
            ),
        )


def _read_bytes(given: str) -> bytes:
    """The content of the regular file at the path, at most MAX_BYTES of it.

    A named pipe or a device, such as /dev/zero, would block or never end, so
    only a regular file is opened; one larger than MAX_BYTES isn't read past it.
    """
    try:
        status = os.stat(os.path.expanduser(given))
    except (FileNotFoundError, NotADirectoryError):
        raise ProjectFileNotFoundError(
            f"Project.load found no file at {_where(given)}. Save a project there "
            f"first with project.save({given!r}), or pass the path of an existing "
            "project file."
        ) from None
    except OSError as error:
        raise ProjectFileError(_unreadable(given, error)) from None
    if not stat.S_ISREG(status.st_mode):
        raise ProjectFileError(
            f"Project.load found {_where(given)}, but it isn't a regular file, such "
            "as a named pipe or a device, so it can't be a project file. Pass the "
            "path of a project TOML file, such as one project.save wrote."
        )
    try:
        with open(os.path.expanduser(given), "rb") as file:
            data = file.read(MAX_BYTES + 1)
    except OSError as error:
        raise ProjectFileError(_unreadable(given, error)) from None
    if len(data) > MAX_BYTES:
        raise ProjectFileError(
            f"Project.load found more than {MAX_BYTES // 1024 // 1024} MiB in "
            f"{_where(given)}, but a project file holds only a name and a data "
            "policy, which take a few KiB at most. Check that the path names a "
            "project file, not a data file."
        )
    return data


def _unreadable(given: str, error: OSError) -> str:
    """Say why the operating system refused the file, and how to fix it."""
    reason = error.strerror or type(error).__name__
    return (
        f"Project.load can't read {_where(given)}: {reason}. Give the user that "
        "runs this process read permission on the file, or pass the path of a "
        "project file it can read."
    )


def invalid(path: str | os.PathLike[str], error: AgentEDAError) -> ProjectFileError:
    """The error for a file that holds values a Project can't take together.

    `error` is the one building the Project raised, which names the field, the
    value and what the field allows.
    """
    return worded(
        ProjectFileError,
        lambda surface: (
            f"Project.load can't build a project from "
            f"{os.fspath(path)!r}: {message(error, surface)} "
            f"{_format_source(surface, start=True)} shows how a project file writes "
            "each key."
        ),
    )


def write(path: object, fields: Mapping[str, Any], *, overwrite: object) -> Path:
    """Write a project's fields, as model_dump(mode="json") gives them, to a file.

    Returns the file's absolute path.
    """
    given = _file_path("Project.save", path)
    overwrite = python_scalar(overwrite)
    if not isinstance(overwrite, bool):
        raise InvalidArgumentError(
            f"Project.save got {_value_text(overwrite)} for overwrite, but overwrite "
            "must be bool: True replaces an existing file, False leaves it. See "
            "help(ae.Project.save) for every argument."
        )
    absolute = Path(os.path.abspath(os.path.expanduser(given)))
    if absolute.parent.exists() and not absolute.parent.is_dir():
        raise InvalidArgumentError(
            f"Project.save got {given!r} for path, but {absolute.parent} is a file, "
            "so it can't hold one. Pass a path in a directory."
        )
    if not absolute.parent.is_dir():
        raise InvalidArgumentError(
            f"Project.save got {given!r} for path, but its directory "
            f"{absolute.parent} doesn't exist. Create it first, such as with "
            f"os.makedirs({str(absolute.parent)!r}), or pass a path in an existing "
            "directory."
        )
    content = _toml({"format_version": FORMAT_VERSION, **fields}).encode("utf-8")
    if len(content) > MAX_BYTES:
        raise ProjectFileError(
            f"Project.save can't write the project to {_where(given)}: it takes "
            f"{len(content):,} bytes, but Project.load reads at most "
            f"{MAX_BYTES // 1024 // 1024} MiB ({MAX_BYTES:,} bytes) of a project "
            "file and would refuse it. Declare fewer sensitive columns, or build "
            "the project in code in each script instead of saving it."
        )
    if not overwrite:
        # lexists sees a dangling link too, which must not lead the new file
        # somewhere else; the exclusive creation below refuses it as well.
        if os.path.lexists(absolute):
            raise _exists(given)
        _replace(absolute, content, overwrite=False, given=given)
        return absolute
    # A link to a file is followed, so that the file it points to is replaced and
    # the link stays. A dangling link is replaced itself, not written through.
    target = Path(os.path.realpath(absolute)) if absolute.exists() else absolute
    if target.exists() and not stat.S_ISREG(target.stat().st_mode):
        raise InvalidArgumentError(
            f"Project.save found {_where(given)}, but it isn't a regular file, "
            "such as a named pipe or a device, so it can't hold a project "
            "file, and writing to it could wait for a reader without end. Pass "
            "the path of a regular file, such as 'project.toml'."
        )
    _replace(target, content, overwrite=True, given=given)
    return absolute


def _exists(given: str) -> ProjectFileExistsError:
    return ProjectFileExistsError(
        f"Project.save found a file at {_where(given)} already, and replaces a "
        "file only when told to. Pass overwrite=True to replace it, or pass "
        "another path."
    )


def _replace(target: Path, content: bytes, *, overwrite: bool, given: str) -> None:
    """Put the content at the target in one step, so that a save cut short, such
    as by a full disk or an interrupt, leaves the file that was there.

    The content goes to a temporary file in the same directory first. It is
    readable by its owner only, since a project names sensitive columns, unless
    a file is replaced: that file's permissions carry over. Without overwrite,
    a hard link puts the file in place only if nothing is there, not even a
    dangling link, so a file that appears meanwhile is never replaced.
    """
    mode = 0o600
    if overwrite and target.exists():
        mode = stat.S_IMODE(target.stat().st_mode)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        _create(temporary, content, mode)
        if overwrite:
            os.replace(temporary, target)
            return
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise _exists(given) from None
        except OSError:
            # A file system without hard links: create the file exclusively.
            _create(target, content, mode)
    except FileExistsError:
        raise _exists(given) from None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _create(path: Path, content: bytes, mode: int) -> None:
    """Create a new file with exactly these permissions, whatever the umask."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as file:
        file.write(content)
        if hasattr(os, "fchmod"):
            os.fchmod(file.fileno(), mode)


def _file_path(owner: str, path: object) -> str:
    """The path a caller gave, as text; raise unless it can name a file.

    A leading ~ means the home directory, as pandas reads it.
    """
    given = os.fspath(path) if isinstance(path, str | os.PathLike) else None
    if not isinstance(given, str):
        raise InvalidArgumentError(
            f"{owner} got {_value_text(path)} for path, but path must be a str or a "
            f"pathlib.Path, such as 'project.toml'. See help(ae.{owner}) for every "
            "argument."
        )
    if not given:
        raise InvalidArgumentError(
            f"{owner} got an empty path, but path must name a file, such as "
            "'project.toml'."
        )
    if "\x00" in given:
        raise InvalidArgumentError(
            f"{owner} got {given!r} for path, but a path can't hold a null "
            "character. Pass the path of a file, such as 'project.toml'."
        )
    separators = tuple(sep for sep in (os.sep, os.altsep) if sep)
    example = os.path.join(given, "project.toml")
    if given.endswith(separators):
        raise InvalidArgumentError(
            f"{owner} got {given!r} for path, whose last {given[-1]!r} makes it name "
            f"a directory, but path must name a file, such as {example!r}."
        )
    if os.path.isdir(os.path.abspath(os.path.expanduser(given))):
        raise InvalidArgumentError(
            f"{owner} got the directory {_where(given)} for path, but path must name "
            f"a file in it, such as {example!r}."
        )
    return given


def _where(given: str) -> str:
    """A path as the caller gave it, and where it leads if that differs."""
    absolute = os.path.abspath(os.path.expanduser(given))
    return repr(given) if absolute == given else f"{given!r} ({absolute})"


def _parse(given: str, data: bytes) -> dict[str, Any]:
    """The keys and values of TOML text, given as UTF-8 bytes."""
    # Editors on Windows may start UTF-8 text with a byte order mark.
    data = data.removeprefix(codecs.BOM_UTF8)
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as error:
        line = data[: error.start].count(b"\n") + 1
        raise worded(
            ProjectFileError,
            lambda surface: (
                f"Project.load found bytes that aren't UTF-8 text in "
                f"{given!r} at line {line}. A project file is TOML, which is UTF-8 text. "
                f"{_rewrite(surface)}"
            ),
        ) from None
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as error:
        reason = str(error)
        raise worded(
            ProjectFileError,
            lambda surface: _syntax_message(given, content, reason, surface),
        ) from None


def _syntax_message(given: str, content: str, error: str, surface: Surface) -> str:
    """Say where a file isn't TOML and how to fix it, without quoting the file."""
    hint = ""
    if match := _AT_LINE.fullmatch(error):
        reason = match["reason"]
        where = f" at line {match['line']}, column {match['column']}"
        if reason == "Invalid value":
            hint = (
                " TOML puts text in quotes, writes true and false in lower case and "
                "has no null, so leave out a key that has no value."
            )
    elif match := _AT_END.fullmatch(error):
        reason = match["reason"]
        lines = content.split("\n")
        last = max((n for n, line in enumerate(lines, 1) if line.strip()), default=1)
        where = f" at its end, after line {last}"
        hint = (
            " The file stops in the middle of its last key, value or table line, such "
            "as an array without its closing ]."
        )
    else:
        reason, where = error, ""
    return (
        f"Project.load found invalid TOML in {given!r}{where}: {reason}.{hint} "
        f"{_rewrite(surface)}"
    )


def _check_keys(
    fields: Mapping[str, Any],
    given: str,
    *,
    keys: Sequence[str],
    table: str,
    table_keys: Sequence[str],
) -> None:
    """Raise unless every key of the file is a field where it stands.

    A misspelled key would otherwise be dropped quietly, so a typo could switch
    masking off. Each unknown key is named with the closest field, which may
    stand at the other level: in TOML, every key below a table's line belongs
    to that table.
    """
    top, inside = "at the top level", f"in [{table}]"
    unknown = [(key, top) for key in fields if key not in keys]
    if isinstance(fields.get(table), Mapping):
        unknown += [(key, inside) for key in fields[table] if key not in table_keys]
    if not unknown:
        return
    fields_at = {top: keys, inside: table_keys}
    hints = [_key_hint(key, place, fields_at) for key, place in unknown]
    if len(unknown) == 1:
        [(key, place)], [(hint, _)] = unknown, hints
        where = f"in the [{table}] table" if place == inside else place
        found = f"an unknown key {key!r} {where} of {given!r}."
        if hint:
            found += f" {hint[0].upper()}{hint[1:]}{'' if hint.endswith('?') else '.'}"
    else:
        described = [
            f"{key!r} {place}" + (f" ({hint})" if hint else "")
            for (key, place), (hint, _) in zip(unknown, hints, strict=True)
        ]
        found = f"{len(unknown):,} unknown keys in {given!r}: {join_items(described)}."
    rule = (
        f" Keys below a [{table}] line belong to that table, and keys above it to "
        "the top level."
        if any(moves for _, moves in hints)
        else ""
    )
    raise ProjectFileError(
        f"Project.load found {found}{rule} Allowed: {join_items(keys)} {top}; "
        f"{join_items(table_keys)} {inside}.",
        code="unknown_project_key",
    )


def _key_hint(
    key: str, place: str, fields_at: Mapping[str, Sequence[str]]
) -> tuple[str, bool]:
    """How to fix an unknown key, and whether the fix moves it to the other level.

    `fields_at` maps each level, such as "at the top level", to its fields.
    """
    [other] = [level for level in fields_at if level != place]
    same = closest_names(key, fields_at[place])[:1]
    if same:
        return f"did you mean {same[0]!r}?", False
    if key in fields_at[other]:
        return f"it belongs {other}", True
    close = closest_names(key, fields_at[other])[:1]
    if close:
        return f"did you mean {close[0]!r} {other}?", True
    return "", False


def _check_table(
    fields: Mapping[str, Any], given: str, *, table: str, table_keys: Sequence[str]
) -> None:
    """Raise unless the file has the table, as a table, with each of its keys."""

    def form(surface: Surface) -> str:
        return (
            f"a [{table}] line, with single brackets, followed by "
            f"{join_items(table_keys)}, as {_format_source(surface)} shows"
        )

    if table not in fields:
        raise worded(
            ProjectFileError,
            lambda surface: (
                f"Project.load found no [{table}] table in {given!r}. "
                f"{_WHOLE_POLICY}: add {form(surface)}."
            ),
        )
    if not isinstance(fields[table], Mapping):
        raise worded(
            ProjectFileError,
            lambda surface: (
                f"Project.load found {_kind(fields[table])} for {table} "
                f"in {given!r}, but {table} is one table: {form(surface)}."
            ),
        )
    missing = [key for key in table_keys if key not in fields[table]]
    if missing:
        raise worded(
            ProjectFileError,
            lambda surface: (
                f"Project.load found no {join_items(missing)} in the "
                f"[{table}] table of {given!r}. {_WHOLE_POLICY}: add "
                f"{join_items(missing)} below the [{table}] line, as "
                f"{_format_source(surface)} shows."
            ),
        )


def _check_forms(fields: Mapping[str, Any], given: str, *, table: str) -> None:
    """Raise unless each value has the form the file writes for its field."""
    values = [
        *((key, value, "at the top level") for key, value in fields.items()),
        *(
            (key, value, f"in the [{table}] table")
            for key, value in fields[table].items()
        ),
    ]
    for key, value, place in values:
        if key not in _FORMS:
            continue
        form, fits = _FORMS[key]
        items = value if key in _ARRAYS and isinstance(value, list) else None
        if items is None and not fits(value):
            raise ProjectFileError(
                f"Project.load found {_kind(value)} for {key} {place} of {given!r}, "
                f"but {key} takes {form}."
            )
        for index, item in enumerate(items or []):
            if not fits(item):
                raise ProjectFileError(
                    f"Project.load found {_kind(item)} at {key}[{index}] {place} of "
                    f"{given!r}, but {key} takes {form}."
                )


def _kind(value: object) -> str:
    """What TOML calls the type of a value, with an article."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a float"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, dict):
        return "a table"
    if isinstance(value, list):
        tables = value and all(isinstance(item, dict) for item in value)
        return "an array of tables" if tables else "an array"
    return "a date or time"


def _toml(fields: Mapping[str, Any]) -> str:
    """A project's fields as TOML: its values first, then a table per mapping.

    TOML has no null, so a field that is None is left out.
    """
    lines = [_HEADER]
    tables = []
    for key, value in fields.items():
        if isinstance(value, Mapping):
            tables.append((key, value))
        elif value is not None:
            lines.append(f"{key} = {_value(value, key)}")
    for name, table in tables:
        lines.extend(["", f"[{name}]"])
        lines.extend(
            f"{key} = {_value(value, f'{name}.{key}')}"
            for key, value in table.items()
            if value is not None
        )
    return "\n".join(lines) + "\n"


def _value(value: object, field: str) -> str:
    """A value as TOML writes it: a string, a boolean, an integer or an array.

    `field` is where the value sits, such as "data_policy.sensitive_columns[1]".
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _string(value, field)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = (
            f"    {_value(item, f'{field}[{i}]')},\n" for i, item in enumerate(value)
        )
        return "[\n" + "".join(items) + "]"
    raise TypeError(f"a project file has no form for a {type(value).__name__}")


def _string(text: str, field: str) -> str:
    """Text as a TOML basic string, on one line.

    Quotes, backslashes and control characters are escaped, as TOML requires,
    and so is every other character that prints as nothing, such as a
    zero-width space, so that whoever reads the file sees it. A lone
    surrogate, which isn't Unicode text, raises.
    """
    surrogates = [ch for ch in text if 0xD800 <= ord(ch) <= 0xDFFF]
    if surrogates:
        raise ProjectFileError(
            f"Project.save can't write {field}, {text!r}, to TOML, which holds only "
            f"Unicode text: {surrogates[0]!r} is a lone surrogate, as left by "
            "decoding bytes with errors='surrogateescape'. Use text without it, or "
            "build the project in code in each script instead of saving it."
        )
    return '"' + "".join(_character(ch) for ch in text) + '"'


def _character(ch: str) -> str:
    if ch in _ESCAPES:
        return _ESCAPES[ch]
    if ch.isprintable():
        return ch
    code = ord(ch)
    return f"\\u{code:04X}" if code <= 0xFFFF else f"\\U{code:08X}"
