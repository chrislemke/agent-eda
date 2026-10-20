"""A caller mistake's message, worded for the surface the call came through.

A message says what is wrong, what is allowed and how to fix the call. What is
wrong and allowed read the same for a Python caller and an MCP caller; the fix
doesn't, since an MCP caller passes JSON arguments and files, not Python.

A raise site states a message as data: texts and fixes, such as
Pass(task='regression'), LeaveOut('columns'), SeeArguments('summarize_numeric'),
PythonOnly("df.reset_index()") and DeclareSensitive(['email']). It never asks
which surface it is on. One renderer writes the same wording for each surface,
so a fix can't be left in Python for an MCP caller: str(error) is the Python
message, and message(error, "mcp") the MCP one, which the MCP server sends.
"""

import errno
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from agent_eda._text import MAX_NAMED, Surface, join_items
from agent_eda.errors import AgentEDAError


class Fix:
    """A piece of a message that reads differently for each surface."""

    def render(self, surface: Surface) -> str:
        """The piece as a caller on the surface reads it."""
        raise NotImplementedError

    def bind(self, **names: str) -> "Part":
        """The piece with `{name}` in its code replaced; most pieces have none."""
        return self


type Part = str | Fix
# What a message is made of, in order: texts and fixes, nested as needed.
type Wording = Part | Iterable[Wording]


def parts_of(wording: Wording) -> tuple[Part, ...]:
    """The wording as a flat tuple of texts and fixes."""
    if isinstance(wording, str | Fix):
        return (wording,)
    return tuple(part for item in wording for part in parts_of(item))


def render(wording: Wording, surface: Surface) -> str:
    """The wording as a caller on the surface reads it."""
    return "".join(
        part if isinstance(part, str) else part.render(surface)
        for part in parts_of(wording)
    )


def _bound(wording: Wording, names: Mapping[str, str]) -> tuple[Part, ...]:
    return tuple(
        part if isinstance(part, str) else part.bind(**names)
        for part in parts_of(wording)
    )


class Words(Fix):
    """Words that name something differently on each surface, such as who reads
    a project file: Project.load in Python, and the MCP server over MCP."""

    def __init__(self, python: Wording, mcp: Wording) -> None:
        self.python = python
        self.mcp = mcp

    def render(self, surface: Surface) -> str:
        return render(self.python if surface == "python" else self.mcp, surface)

    def bind(self, **names: str) -> Part:
        return Words(_bound(self.python, names), _bound(self.mcp, names))


# How a caller writes None: Python's, and JSON's null over MCP.
NONE = Words("None", "null")


class Say(Fix):
    """Parts that read as one, such as a step of a fix."""

    def __init__(self, *parts: Wording) -> None:
        self.parts = parts

    def render(self, surface: Surface) -> str:
        return render(self.parts, surface)

    def bind(self, **names: str) -> Part:
        return Say(*_bound(self.parts, names))


class Pass(Fix):
    """Arguments as the caller writes them: task='regression' in Python, and
    {"task": "regression"}, a JSON object, over MCP."""

    def __init__(self, **arguments: object) -> None:
        self.arguments = arguments

    def render(self, surface: Surface) -> str:
        if surface == "python":
            return ", ".join(
                f"{name}={value!r}" for name, value in self.arguments.items()
            )
        return json.dumps(self.arguments)


class LeaveOut(Fix):
    """How a caller passes None for an option: pass columns=None in Python, and
    leave columns out over MCP, where null is the default."""

    def __init__(self, name: str) -> None:
        self.name = name

    def render(self, surface: Surface) -> str:
        if surface == "python":
            return f"pass {self.name}=None"
        return f"leave {self.name} out"


class SeeArguments(Fix):
    """Where to read about every argument of an analysis, or field of a model."""

    def __init__(self, owner: str, noun: str = "argument") -> None:
        self.owner = owner
        self.noun = noun

    def render(self, surface: Surface) -> str:
        if surface == "python":
            return f"See help(ae.{self.owner}) for every {self.noun}."
        if self.noun == "argument":
            return "The MCP tool's input schema describes every argument."
        return (
            f"The {self.owner} definition in the MCP tool's input schema describes "
            f"every {self.noun}."
        )


class PythonOnly(Fix):
    """Code only a Python caller can run, such as a pandas conversion. An MCP
    caller reads `otherwise` in its place, which is nothing by default. `{column}`
    in the code becomes the code for a column when the fix is bound to one."""

    def __init__(self, code: str, otherwise: str = "") -> None:
        self.code = code
        self.otherwise = otherwise

    def render(self, surface: Surface) -> str:
        return self.code if surface == "python" else self.otherwise

    def bind(self, **names: str) -> Part:
        return PythonOnly(self.code.format(**names), self.otherwise)


class DeclareSensitive(Fix):
    """An example of declaring sensitive columns: in code for a Python caller,
    as a project or as a data policy, and over MCP the TOML a project file holds,
    since an MCP caller declares a data policy only in a project file."""

    def __init__(self, columns: Iterable[str], *, of: str = "project") -> None:
        if of not in ("project", "policy"):
            raise ValueError(f"of must be 'project' or 'policy', not {of!r}")
        self.columns = list(columns)
        self.of = of

    def render(self, surface: Surface) -> str:
        if surface == "mcp":
            return f"sensitive_columns = {json.dumps(self.columns)}"
        if self.of == "project":
            return (
                "project=ae.Project(data_policy=ae.DataPolicy("
                f"sensitive_columns={self.columns!r}))"
            )
        return f"ae.DataPolicy(sensitive_columns={self.columns!r}, mask_sensitive=True)"


class Echo(Fix):
    """A value a message quotes, as the caller wrote it: Python's repr, and over
    MCP the JSON words for null and the booleans, such as null and true.
    Everything else shows as its repr."""

    def __init__(self, value: object) -> None:
        self.value = value

    def render(self, surface: Surface) -> str:
        return echoed(self.value, surface)


class TypeOf(Fix):
    """What a message calls the type of a value: Python's type name, and over MCP
    the JSON type, with its article."""

    def __init__(self, value: object) -> None:
        self.value = value

    def render(self, surface: Surface) -> str:
        return type_of(self.value, surface)


class PathInto(Fix):
    """The steps into an option as the caller writes them: ['age'][1] in Python,
    and over MCP /age/1, a JSON pointer, since JSON has no subscripts."""

    def __init__(self, steps: Iterable[object]) -> None:
        self.steps = list(steps)

    def render(self, surface: Surface) -> str:
        if surface == "python":
            return "".join(f"[{step!r}]" for step in self.steps)
        return "".join(f"/{step}" for step in self.steps)


class Listed(Fix):
    """Items written as "a, b and c", each as the caller on the surface reads it,
    naming at most `limit` of them and counting the rest."""

    def __init__(
        self,
        items: Iterable[Wording],
        *,
        limit: int = MAX_NAMED,
        total: int | None = None,
    ) -> None:
        self.items = list(items)
        self.limit = limit
        self.total = total

    def render(self, surface: Surface) -> str:
        return join_items(
            [render(item, surface) for item in self.items],
            limit=self.limit,
            total=self.total,
        )


class Of(Fix):
    """The message of another error, as the caller on the surface reads it."""

    def __init__(self, error: AgentEDAError) -> None:
        self.error = error

    def render(self, surface: Surface) -> str:
        return message(self.error, surface)


class OneOf(Fix):
    """Steps that fix a call, any one of which does, written as a list.

    A step an MCP caller can't take, such as a PythonOnly one, renders as nothing
    and drops out. One step reads as it is; a step and a last resort read
    "a, or b"; more read "a; b; or c", since steps hold commas of their own.
    `last` is the step that is always there, such as leaving a column out. With
    `capital`, the list starts with a capital.
    """

    def __init__(
        self,
        steps: Iterable[Wording],
        last: Wording | None = None,
        *,
        capital: bool = False,
    ) -> None:
        self.steps = list(steps)
        self.last = last
        self.capital = capital

    def render(self, surface: Surface) -> str:
        steps = [text for step in self.steps if (text := render(step, surface))]
        last = None if self.last is None else render(self.last, surface)
        text = self._list(steps, last)
        return f"{text[:1].upper()}{text[1:]}" if self.capital else text

    @staticmethod
    def _list(steps: list[str], last: str | None) -> str:
        if last:
            if not steps:
                return last
            if len(steps) == 1:
                return f"{steps[0]}, or {last}"
            steps.append(last)
        if len(steps) <= 1:
            return "".join(steps)
        return "; ".join([*steps[:-1], f"or {steps[-1]}"])


def mistake[E: AgentEDAError](
    error_class: type[E], *wording: Wording, code: str | None = None
) -> E:
    """The error for a caller's mistake, with its message written for each surface.

    The wording holds what is wrong, what is allowed and the fixes, as texts and
    Fix objects. str(error) is the message a Python caller reads, and
    message(error, "mcp") the one an MCP caller reads.
    """
    parts = parts_of(wording)
    error = error_class(render(parts, "python"), code=code)
    error._wording = parts
    return error


def message(error: AgentEDAError, surface: Surface) -> str:
    """The error's message as a caller on the surface reads it."""
    if error._wording is None:
        return str(error)
    return render(error._wording, surface)


def echoed(value: object, surface: Surface) -> str:
    """A value a message quotes, as the caller wrote it: Python's repr, and over MCP
    the JSON words for null and the booleans, such as null and true. Everything
    else shows as its repr."""
    if surface == "mcp":
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, bool):
            return json.dumps(value)
    return repr(value)


def type_of(value: object, surface: Surface) -> str:
    """What a message calls the type of a value: Python's type name, and over MCP
    the JSON type, with its article."""
    if surface == "python":
        return type(value).__name__
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list | tuple):
        return "an array"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


# None, True or False as a word of the docs; a quoted or backticked one is a
# value of the data, such as the text 'None' among the stand-ins for missing
# values.
_CONSTANT = re.compile(r"(?<![\w'\"`])(None|True|False)(?![\w'\"`])")
_JSON_WORDS = {"None": "null", "True": "true", "False": "false"}


# The sentence on labels that aren't strings, which no file an MCP tool reads
# has: a CSV header, and the column names of a Parquet file, are text.
_NON_STRING_LABELS = re.compile(
    r"\s?A label that isn't a string can be given as itself(?:, such as 0,)? "
    r"or as its string form\."
)


def json_words(text: str) -> str:
    """The text as an MCP caller reads it: Python's None, True and False written as
    JSON's null, true and false, the words an MCP caller passes, and without the
    sentence on labels that aren't strings."""
    return _CONSTANT.sub(
        lambda match: _JSON_WORDS[match[0]], _NON_STRING_LABELS.sub("", text)
    )


def without_titles(schema: Any) -> Any:
    """The JSON Schema without pydantic's titles, which only repeat the names
    they sit under. A property that is itself named title is kept."""
    if isinstance(schema, dict):
        return {
            key: without_titles(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [without_titles(item) for item in schema]
    return schema


def json_words_in(schema: Any) -> Any:
    """The JSON Schema with json_words applied to each description in it."""
    if isinstance(schema, dict):
        return {
            key: json_words(value)
            if key == "description" and isinstance(value, str)
            else json_words_in(value)
            for key, value in schema.items()
        }
    if isinstance(schema, list):
        return [json_words_in(item) for item in schema]
    return schema


def os_cause(error: BaseException) -> str:
    """Why the operating system refused a path, in words that say what to
    change: for an OSError, ValueError or RuntimeError raised by resolving a path."""
    code = error.errno if isinstance(error, OSError) else None
    if isinstance(error, ValueError):
        return "it holds a NUL character, which no file name can"
    if isinstance(error, RuntimeError):
        if "home directory" in str(error):
            return "its ~ can't be expanded because no home directory is known for it"
        return "it leads through symbolic links that loop back on themselves"
    if code == errno.ENAMETOOLONG:
        return (
            "a name in it is longer than 255 characters, or the whole path is "
            "longer than the operating system allows"
        )
    if code == errno.ELOOP:
        return "it leads through symbolic links that loop back on themselves"
    if code in (errno.EACCES, errno.EPERM):
        return (
            "permission denied: the process that runs agent-eda may not search a "
            "directory on the path or read the file"
        )
    if code == errno.EROFS:
        return "the volume is read-only"
    if code == errno.ENOSPC:
        return "No space left on device"
    if code == errno.ENOTDIR:
        return "a part of the path that should be a directory is a file"
    reason = getattr(error, "strerror", None) or str(error)
    return reason.strip().rstrip(".") or type(error).__name__
