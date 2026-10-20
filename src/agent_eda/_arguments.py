"""Reading the arguments of a call, and saying what is wrong with one.

Analyses, discovery functions and the Project classes all take their arguments
through this module. It turns what agents commonly pass into the plain Python an
option declares, checks that the keywords and the data are the right kind, and
words each mistake: an unknown name with the closest one, a missing argument, a
value that doesn't fit its annotation, a value passed twice. The wording of a
fix, for a Python or an MCP caller, is _wording's; the type of an option as
text is _annotations'; the closest names are _text's. This module only decides
which of them a message needs.

It knows nothing of the registry: what it needs from an analysis, its name and
annotations, is passed in.
"""

import functools
import inspect
import json
import os
import typing
from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any

import numpy as np
import pandas as pd

from agent_eda._annotations import (
    LIST_TYPES,
    MAPPING_TYPES,
    allowed_text,
    item_annotation,
    members,
    strip_optional,
)
from agent_eda._text import (
    MAX_NAME_LENGTH,
    Surface,
    a_or_an,
    closest_names,
    join_items,
    plural,
    short_repr,
)
from agent_eda._wording import (
    Echo,
    Fix,
    Listed,
    Of,
    PathInto,
    SeeArguments,
    Wording,
    Words,
    echoed,
    mistake,
)
from agent_eda.errors import AgentEDAError, InvalidArgumentError

# The arguments of an analysis that shows raw values, and of its MCP tool.
_MASKING_ARGUMENTS = frozenset({"project", "project_path", "mask_sensitive"})


# What a DataFrame replacement is called and the conversion to a DataFrame,
# by library and class name.
_CONVERSIONS = {
    ("pandas", "Series"): ("a pandas Series", "series.to_frame()"),
    ("numpy", "ndarray"): ("a NumPy array", "pd.DataFrame(array)"),
    ("polars", "DataFrame"): ("a polars DataFrame", "df.to_pandas()"),
    ("polars", "LazyFrame"): ("a polars LazyFrame", "lf.collect().to_pandas()"),
    ("polars", "Series"): ("a polars Series", "s.to_frame().to_pandas()"),
    ("pyarrow", "Table"): ("a pyarrow Table", "table.to_pandas()"),
    ("pyarrow", "RecordBatch"): ("a pyarrow RecordBatch", "batch.to_pandas()"),
}


def check_data(name: str, data: object) -> None:
    """Raise unless data is a pandas DataFrame, showing the conversion."""
    if isinstance(data, pd.DataFrame):
        return
    raise InvalidArgumentError(_not_a_frame(name, data), code="unsupported_data_type")


def _not_a_frame(name: str, data: object) -> str:
    """Say what data is, and the one-line conversion to a DataFrame if there is one."""
    start = f"data must be a pandas DataFrame, but got {_describe_data(data)}."
    call = f"ae.{name}"
    if data is None:
        return (
            f"{start} The variable holds no data, so whatever should have loaded "
            f"or built it didn't. Check that step, then pass its DataFrame: "
            f"{call}(df)."
        )
    if isinstance(data, str | os.PathLike):
        return (
            f"{start} A path isn't data; read the file first: "
            f"{call}(pd.read_csv(path)) for a CSV file, or "
            f"{call}(pd.read_parquet(path)) for a Parquet file."
        )
    return f"{start} Convert it first: {call}({_conversion(data)})"


def _describe_data(data: object) -> str:
    """What data is, for the message: 'a pandas Series', 'None', 'a str'."""
    if data is None:
        return "None"
    kind = type(data)
    known = _CONVERSIONS.get((kind.__module__.split(".")[0], kind.__name__))
    return known[0] if known is not None else a_or_an(kind.__name__)


def _conversion(data: object) -> str:
    """The one-line conversion of data to a DataFrame."""
    kind = type(data)
    known = _CONVERSIONS.get((kind.__module__.split(".")[0], kind.__name__))
    return known[1] if known is not None else "pd.DataFrame(data)"


def checked_keywords[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Make a call with a keyword the function doesn't take raise InvalidArgumentError.

    Python raises a bare TypeError for it. The error has the code invalid_call and
    names the closest keyword, as an analysis's does. The check reads the
    function's signature, and the function keeps its name, docstring and
    signature. A method's own `self` or `cls` isn't a keyword.
    """
    owner = getattr(function, "__qualname__", "the function")
    names = [
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.name not in ("self", "cls")
    ]

    @functools.wraps(function)
    def checked(*args: P.args, **kwargs: P.kwargs) -> R:
        for key in kwargs:
            if key not in names:
                raise InvalidArgumentError(
                    unknown_name_message(owner, "argument", key, names)
                    if names
                    else f"{owner} takes no arguments, but got {key!r}. Call it as "
                    f"{owner}().",
                    code="invalid_call",
                )
        return function(*args, **kwargs)

    return checked


def check_distinct(
    analysis: str,
    parameter: str,
    values: Sequence[Hashable],
    noun: str,
    *,
    key: Callable[[Any], Hashable] | None = None,
    hide: bool = False,
    within: str | None = None,
) -> None:
    """Raise when a list argument holds the same value more than once.

    Values are the same when Python's == says so, or when their `key` is. With
    `hide`, the message counts the repeated values and doesn't quote them: they
    are values of a sensitive column. `within` names the key of a mapping
    argument whose list this is, such as a column of sentinel_values.
    """
    keyed = values if key is None else [key(value) for value in values]
    repeated = [
        values[keyed.index(item)] for item, count in Counter(keyed).items() if count > 1
    ]
    if not repeated:
        return

    if hide:
        named: Wording = (
            f"{plural(len(repeated), noun, noun + 's')} (not shown: masking "
            "hides the values of a sensitive column)"
        )
    else:
        named = Listed(Echo(value) for value in repeated)
    where: Wording = parameter
    if within is not None:
        where = Words(
            f"{parameter}[{within!r}]",
            f"the {parameter} list for {json.dumps(within)}",
        )
    raise mistake(
        InvalidArgumentError,
        f"{analysis} got ",
        named,
        " more than once in ",
        where,
        f". List each {noun} once.",
    )


def unknown_name_message(
    owner: str,
    noun: str,
    name: str,
    allowed: Sequence[str],
    *,
    of_analysis: bool = False,
) -> str:
    """Say that a keyword isn't one of `owner`'s, naming the closest and the rest.

    A project or mask_sensitive passed to an analysis that has none says why: it
    shows no raw values, so there is nothing to mask. `of_analysis` says that
    `owner` is an analysis, whose arguments those are.
    """
    listed = f"Allowed: {', '.join(allowed)}."
    takes_masking = bool(_MASKING_ARGUMENTS & set(allowed))
    if (
        noun == "argument"
        and name in _MASKING_ARGUMENTS
        and of_analysis
        and not takes_masking
    ):
        return (
            f"{owner} takes no {name}: it shows no raw values, so there is nothing "
            f"to mask. Leave {name} out. {listed}"
        )
    closest = closest_names(name, allowed)[:1]
    hint = f" Did you mean {closest[0]!r}?" if closest else ""
    return f"{owner} has no {noun} {name!r}.{hint} {listed}"


def missing_message(
    owner: str,
    annotations: Mapping[str, object],
    missing: list[str],
    *,
    fix: str,
    see: str,
) -> str:
    """Name the required arguments a call left out, how to pass them, and where
    to read about every argument.

    The choices of an option that takes fixed values are listed too, and so
    are those of the values inside a mapping or list option, such as a schema.
    A name that isn't an option, such as an MCP tool's data_path, has none.
    """
    arguments = plural(len(missing), "required argument", "required arguments")
    them = "it" if len(missing) == 1 else "them"
    choices = []
    for name in missing:
        if name not in annotations:
            continue
        annotation = strip_optional(annotations[name])
        item = item_annotation(annotation)
        if typing.get_origin(annotation) is typing.Literal:
            choices.append(f"{name} is {allowed_text(annotation, name)}")
        elif item is not annotation and any(
            typing.get_origin(member) is typing.Literal for member in members(item)
        ):
            choices.append(f"each value in {name} is {allowed_text(item, 'value')}")
    where = f", where {join_items(choices)}" if choices else ""
    return (
        f"{owner} is missing {arguments}: {join_items(missing)}. Pass {them} "
        f"{fix}{where}. {see}"
    )


def invalid_message(
    owner: str,
    annotations: Mapping[str, object],
    options: Mapping[str, Any],
    problems: Sequence[Any],
    *,
    noun: str = "argument",
    single: Collection[str] = (),
) -> Wording:
    """Say what is wrong with an option, what it allows and how to fix it.

    `problems` are pydantic's errors for the options; the first one is shown,
    unless one comes from a check of the package's own, such as ColumnSpec's,
    whose message says it best. The message says what the option allows in
    words, never in pydantic's or Python's typing syntax. The options in
    `single` were given one value, which became a list of one: its message is
    about the option, not about the list's first item.
    """
    problem = next((p for p in problems if own_error(p) is not None), problems[0])
    location = problem["loc"]
    name = str(location[0]) if location else "?"
    annotation = annotations.get(name)
    value, steps, inner = locate(options.get(name), annotation, location[1:])
    # Steps into the option; none for a single value that became a list of one.
    inside = [] if name in single and steps == [0] else steps
    path = PathInto(inside)
    own = own_error(problem)
    if own is not None:
        # The package's own check, such as ColumnSpec's, says it best.
        if not inside:
            return Of(own)
        return f"{owner} got an invalid value at {name}", path, ": ", Of(own)
    where: Wording
    rule: Wording
    if inside:
        # A value inside a list option, or inside a mapping or its lists.
        where = f"at {name}", path
        rule = f"each value in {name} must be ", allowed_words(inner, "value")
    else:
        where = f"for {name}"
        rule = f"{name} must be ", allowed_words(annotation, name)
    detail: Wording = ""
    if problem["type"] == "literal_error":
        # The choices of the Literal, alone or in a union such as a logical type
        # or a ColumnSpec.
        choices = [
            str(choice)
            for member in members(strip_optional(inner))
            if typing.get_origin(member) is typing.Literal
            for choice in typing.get_args(member)
        ]
        closest = closest_names(value, choices)[:1] if isinstance(value, str) else []
        detail = f" Did you mean {closest[0]!r}?" if closest else ""
    if name == "project" and isinstance(value, str | os.PathLike):
        rule, detail = "project must be a Project, not a path", load_project_fix(value)
    return (
        f"{owner} got ",
        ShownValue(value),
        " ",
        where,
        ", but ",
        rule,
        ".",
        detail,
        " ",
        SeeArguments(owner, noun),
    )


def allowed_words(annotation: object, subject: str) -> Fix:
    """What an option allows, in words, with None as the caller writes it."""
    return Words(
        allowed_text(annotation, subject, none="None"),
        allowed_text(annotation, subject, none="null"),
    )


def load_project_fix(path: str | os.PathLike[str]) -> Fix:
    """The fix for a path given where a Project belongs: load the file."""
    text = os.fspath(path)
    shown = repr(text) if len(text) <= 80 else "path"
    return Words(
        f" Load the file first, as project=ae.Project.load({shown}).",
        " Pass the path of the project file as project_path instead.",
    )


def own_error(problem: Any) -> AgentEDAError | None:
    """The package's own error behind a pydantic error, if there is one."""
    error = problem.get("ctx", {}).get("error")
    return error if isinstance(error, AgentEDAError) else None


def locate(
    value: object, annotation: object, steps: tuple[object, ...]
) -> tuple[object, list[object], object]:
    """The value at a validation error's location inside an option.

    Follows list positions and mapping keys as far as they lead and returns the
    value found, the steps taken (such as ["age", 1], empty for the option
    itself) and its annotation.
    """
    taken: list[object] = []
    for step in steps:
        if isinstance(value, list) and isinstance(step, int) and step < len(value):
            value = value[step]
            annotation = item_annotation(annotation)
        elif isinstance(value, dict) and step in value:
            value = value[step]
            annotation = _mapping_values(annotation) or annotation
        else:
            break  # such as the name of a union's member
        taken.append(step)
    return value, taken, annotation


def is_single(value: object) -> bool:
    """Whether _as_list makes a list of one of the value."""
    return value is not None and (
        isinstance(value, str | bytes | np.generic) or not isinstance(value, Iterable)
    )


def normalize_option(
    owner: str,
    annotations: Mapping[str, object],
    data: pd.DataFrame,
    name: str,
    value: object,
) -> object:
    """Turn what agents commonly pass into the plain Python an option declares.

    A list option takes any ordered list-like, such as a tuple, a pandas Index
    or Series, a 1-D NumPy array or a generator, and a single value as a
    one-item list. NumPy scalars become Python scalars, and NumPy datetimes and
    timedeltas pandas ones. A column option, such as columns or id_column, also
    takes labels that aren't strings and names them by their string form, the
    way results show them. A mapping option is keyed by column names, which
    follow the same rule, and each of its values is read like an option of the
    value's type. A column option refuses a Series, an Index or an array that
    holds a column's values in place of names. Validation then stays strict
    about everything else.
    """
    annotation = annotations[name]
    values = _mapping_values(annotation)
    if values is not None:
        return _normalize_mapping(owner, name, value, values)
    if not _takes_list(annotation):
        value = python_scalar(value)
        if name.endswith("column") and value is not None:
            if isinstance(value, pd.Series | pd.Index | np.ndarray):
                raise _values_as_names(owner, data, name, value)
            return _column_name(value)
        return value
    items = _as_list(owner, name, value)
    if (
        isinstance(value, pd.Series | pd.Index | np.ndarray)
        and isinstance(items, list)
        and name.endswith("columns")
        and not _are_labels(data, items)
    ):
        raise _values_as_names(owner, data, name, value)
    if isinstance(items, list) and name.endswith("columns"):
        return [_column_name(item) for item in items]
    return items


def _are_labels(data: pd.DataFrame, items: list[object]) -> bool:
    """Whether every item names a column of the data, by label or string form."""
    labels = {str(label) for label in data.columns}
    for item in items:
        try:
            if str(_column_name(item)) not in labels:
                return False
        except TypeError:  # an unhashable item, such as a list
            return False
    return True


def _values_as_names(
    owner: str, data: pd.DataFrame, name: str, value: object
) -> InvalidArgumentError:
    """The error for a column's values passed where its name or names belong.

    It describes the value by its type and size, never by its items: they may
    be values of a sensitive column.
    """
    # The fix shows a real column name: the first one that is short enough.
    example = next(
        (c for c in map(str, data.columns) if len(c) <= MAX_NAME_LENGTH), "email"
    )
    if name.endswith("columns"):
        fix = f"Pass the names of columns, such as {name}=[{example!r}]."
    else:
        fix = f"Pass the name of a column, such as {name}={example!r}."
    return InvalidArgumentError(
        f"{owner} got {value_text(value)} for {name}, which looks like a "
        f"column's values, not column names. {fix}"
    )


def _takes_list(annotation: object) -> bool:
    return any(
        typing.get_origin(member) in LIST_TYPES for member in members(annotation)
    )


def _mapping_values(annotation: object) -> object:
    """The annotation of a mapping option's values, or None if it takes no mapping."""
    for member in members(annotation):
        if typing.get_origin(member) in MAPPING_TYPES:
            return typing.get_args(member)[1]
    return None


def _normalize_mapping(owner: str, name: str, value: object, values: object) -> object:
    """A mapping keyed by column names, each value read as `values` declares it."""
    if not isinstance(value, Mapping):
        return value  # None, or something validation rejects, naming its type
    normalized: dict[object, object] = {}
    given: dict[object, object] = {}
    for key, item in value.items():
        column = _column_name(key)
        if column in normalized:
            raise InvalidArgumentError(
                f"{owner} got column {column!r} more than once in {name}, as "
                f"{given[column]!r} and {key!r}. Name each column once."
            )
        given[column] = key
        normalized[column] = (
            _as_list(owner, name, item, key=column)
            if _takes_list(values)
            else python_scalar(item)
        )
    return normalized


def _as_list(owner: str, name: str, value: object, *, key: object = None) -> object:
    """The value as a list, or as it is when validation should reject it.

    `key` is the key of a mapping option whose value this is, if it is one.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        return [python_scalar(value)]
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return [value]
    if isinstance(value, set | frozenset):
        where = name if key is None else f"{name}[{key!r}]"
        instead = "a list or tuple" if key is None else "a list or tuple for that key"
        fix = f"{name}=sorted(...)" if key is None else f"{{{key!r}: sorted(...)}}"
        raise InvalidArgumentError(
            f"{owner} got a set for {where}, and a set has no fixed order. "
            f"Pass {instead} instead, such as {fix}."
        )
    if isinstance(value, pd.DataFrame | Mapping) or (
        isinstance(value, np.ndarray) and value.ndim != 1
    ):
        return value  # validation rejects it, naming its type
    if isinstance(value, pd.Index | pd.Series):
        return value.tolist()  # pandas gives Timestamps for datetimes
    return [python_scalar(item) for item in value]


def python_scalar(value: object) -> object:
    """A NumPy scalar as Python's or pandas' own; anything else as it is.

    NumPy datetimes and timedeltas become pandas ones: their item() is an int
    for nanoseconds.
    """
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value)
    return value.item() if isinstance(value, np.generic) else value


def _column_name(item: object) -> object:
    """A column label as the string form results show, if it is a label at all."""
    if isinstance(item, str) or not isinstance(item, Hashable):
        return item
    return str(item)


class ShownValue(Fix):
    """A scalar argument as a message shows it, and a larger one by its type."""

    def __init__(self, value: object) -> None:
        self.value = value

    def render(self, surface: Surface) -> str:
        return value_text(self.value, surface)


def value_text(value: object, surface: Surface = "python") -> str:
    """Show a scalar argument; describe anything larger by its type and size.

    A container passed by mistake, such as a column of a DataFrame, would
    otherwise put raw data values into the message. Over MCP null and the
    booleans show as JSON writes them, and a mapping is an object.
    """
    if value is None or isinstance(value, bool | int | float | str | np.generic):
        if surface == "mcp" and (value is None or isinstance(value, bool | np.bool_)):
            return echoed(value, surface)
        return short_repr(value)
    if isinstance(value, pd.DataFrame):
        rows = plural(value.shape[0], "row", "rows")
        return f"a pandas DataFrame with {rows} and {plural(value.shape[1], 'column', 'columns')}"
    if isinstance(value, pd.Series | pd.Index):
        return f"a pandas {type(value).__name__} of {plural(len(value), 'value', 'values')}"
    if isinstance(value, np.ndarray):
        return f"a NumPy array of shape {value.shape}"
    if isinstance(value, list | tuple | set | frozenset | dict):
        kind = (
            "an object"
            if surface == "mcp" and isinstance(value, dict)
            else a_or_an(type(value).__name__)
        )
        return f"{kind} of {plural(len(value), 'item', 'items')}"
    if isinstance(value, type):
        return f"the type {value.__name__}"
    return a_or_an(type(value).__name__)
