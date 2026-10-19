"""A caller mistake's message, worded for the surface the call came through.

A message says what is wrong, what is allowed and how to fix the call. What is
wrong and allowed read the same for a Python caller and an MCP caller; the fix
doesn't, since an MCP caller passes JSON arguments and files, not Python. A
raise site an MCP tool can reach builds its message for both surfaces with one
function of the surface, so the two can't drift apart, and the MCP server sends
the MCP one.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from agent_eda._text import Surface
from agent_eda.errors import AgentEDAError


def worded[E: AgentEDAError](
    error_class: Callable[..., E], text: Callable[[Surface], str], **kwargs: Any
) -> E:
    """The error with the Python message, carrying the MCP one where it differs."""
    error = error_class(text("python"), **kwargs)
    mcp = text("mcp")
    if mcp != str(error):
        error._mcp_message = mcp
    return error


def message(error: AgentEDAError, surface: Surface) -> str:
    """The error's message as a caller on the surface reads it."""
    if surface == "mcp" and error._mcp_message is not None:
        return error._mcp_message
    return str(error)


def see_every(owner: str, noun: str, surface: Surface) -> str:
    """Where to read about every argument of an analysis, or field of a model."""
    if surface == "python":
        return f"See help(ae.{owner}) for every {noun}."
    if noun == "argument":
        return "The MCP tool's input schema describes every argument."
    return (
        f"The {owner} definition in the MCP tool's input schema describes every {noun}."
    )


def as_written(surface: Surface, **arguments: object) -> str:
    """Arguments as the caller writes them: task='regression' in Python, and
    {"task": "regression"}, a JSON object, over MCP."""
    if surface == "python":
        return ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    return json.dumps(arguments)


def left_out(name: str, surface: Surface) -> str:
    """How a caller passes None for an option: pass columns=None in Python, and
    leave columns out over MCP, where null is the default."""
    return f"pass {name}=None" if surface == "python" else f"leave {name} out"


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


def json_words(text: str) -> str:
    """The text with Python's None, True and False written as JSON's null, true and
    false, the words an MCP caller passes."""
    return _CONSTANT.sub(lambda match: _JSON_WORDS[match[0]], text)


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
