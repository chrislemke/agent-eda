"""A caller mistake's message, worded for the surface the call came through.

A message says what is wrong, what is allowed and how to fix the call. What is
wrong and allowed read the same for a Python caller and an MCP caller; the fix
doesn't, since an MCP caller passes JSON arguments and files, not Python. A
raise site an MCP tool can reach builds its message for both surfaces with one
function of the surface, so the two can't drift apart, and the MCP server sends
the MCP one.
"""

import json
from collections.abc import Callable
from typing import Any

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
