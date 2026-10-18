"""How analyses read their parameters' annotations and show them to an agent.

Error messages and describe_analysis state types, choices and ranges with the
same functions, so they can't disagree.
"""

import collections.abc
import types
import typing
from dataclasses import dataclass

import annotated_types
from pydantic import JsonValue

# Short names for the libraries an annotation can come from.
_LIBRARY_ALIASES = {"pandas": "pd", "numpy": "np"}

# The annotations of options that take a list.
LIST_TYPES = (list, tuple, collections.abc.Sequence)


@dataclass(frozen=True)
class TypeName:
    """Annotated metadata naming a type the way agents should read it.

    Such as 'RowId' for a field that holds any index label; type_text() shows
    the name instead of the type underneath.
    """

    name: str


def members(annotation: object) -> tuple[object, ...]:
    """The members of a union, the annotation alone otherwise; Annotated unwrapped."""
    if typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return typing.get_args(annotation)
    return (annotation,)


def strip_optional(annotation: object) -> object:
    """The annotation without `| None` when it is a union with None."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        kept = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(kept) == 1:
            return kept[0]
    return annotation


def item_annotation(annotation: object) -> object:
    """The annotation of the values inside a list option, else the annotation."""
    for member in members(annotation):
        if typing.get_origin(member) in LIST_TYPES:
            return typing.get_args(member)[0]
    return annotation


def bounds(metadata: typing.Iterable[object], subject: str) -> str:
    """A declared range as "0 <= trim_fraction < 0.5", or "" when there is none."""
    lower = upper = ""
    for item in metadata:
        if isinstance(item, annotated_types.Ge):
            lower = f"{item.ge:g} <= "
        elif isinstance(item, annotated_types.Gt):
            lower = f"{item.gt:g} < "
        elif isinstance(item, annotated_types.Le):
            upper = f" <= {item.le:g}"
        elif isinstance(item, annotated_types.Lt):
            upper = f" < {item.lt:g}"
    return f"{lower}{subject}{upper}" if lower or upper else ""


def valid_range(annotation: object, name: str) -> str | None:
    """The range an option or each of its values must lie in, if it declares one.

    Such as "0 <= trim_fraction < 0.5", or "each value: 0 <= value <= 1" for a
    list option.
    """
    own = _range_of(strip_optional(annotation), name)
    if own:
        return own
    item = item_annotation(annotation)
    if item is annotation:
        return None
    each = _range_of(strip_optional(item), "value")
    return f"each value: {each}" if each else None


def _range_of(annotation: object, subject: str) -> str:
    if typing.get_origin(annotation) is not typing.Annotated:
        return ""
    return bounds(typing.get_args(annotation)[1:], subject)


def choices(annotation: object) -> list[JsonValue] | None:
    """The fixed values an option, or each of its values, may take, if any."""
    for candidate in (annotation, item_annotation(annotation)):
        literals = [
            member
            for member in members(candidate)
            if typing.get_origin(member) is typing.Literal
        ]
        if literals:
            return [value for literal in literals for value in typing.get_args(literal)]
    return None


def type_text(annotation: object) -> str:
    """A type as an agent reads it, such as 'Sequence[str] | None'.

    Ranges declared with Annotated are left out, and a Literal shows the type of
    its values, such as 'str'; the allowed values are listed on their own. A
    TypeName in Annotated gives the name to show.
    """
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is typing.Annotated:
        names = [item.name for item in arguments[1:] if isinstance(item, TypeName)]
        return names[0] if names else type_text(arguments[0])
    if origin is typing.Literal:
        names = [type(argument).__name__ for argument in arguments]
        return " | ".join(dict.fromkeys(names))
    if origin in (typing.Union, types.UnionType):
        texts = [type_text(argument) for argument in arguments]
        return " | ".join(dict.fromkeys(texts))
    if origin is not None:
        inner = ", ".join(type_text(argument) for argument in arguments)
        return f"{getattr(origin, '__name__', str(origin))}[{inner}]"
    if annotation is type(None):
        return "None"
    if annotation is Ellipsis:  # as in tuple[float, ...]
        return "..."
    name = getattr(annotation, "__name__", str(annotation))
    library = getattr(annotation, "__module__", "").split(".")[0]
    alias = _LIBRARY_ALIASES.get(library)
    return f"{alias}.{name}" if alias else name


def python_literal(value: JsonValue) -> str:
    """A JSON value as the Python literal that gives it, such as "['a', 1]"."""
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (f"{key!r}: {python_literal(item)}" for key, item in value.items())
        return "{" + ", ".join(pairs) + "}"
    return repr(value)
