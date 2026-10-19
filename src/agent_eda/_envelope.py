"""The envelope of a result: the part that follows from the call, not the data.

The registration wrapper of an analysis (agent_eda._registry) records what the
call was: the analysis's name, its method, its arguments and the libraries it
used. An analysis builds only the rest of its result, through
`AnalysisResult.for_call`, which reads the record from here. It passes an
override for each argument whose recorded value differs from the one given: a
column selection it resolved, or declared values it masked.
"""

import contextlib
import contextvars
import functools
import importlib
import platform
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pydantic
from pydantic import JsonValue

from agent_eda._result_base import Execution
from agent_eda._version import __version__

# The libraries every analysis uses; `for_call` names the others.
LIBRARIES = ("numpy", "pandas")

# The arguments that hold values of the data, such as the values an analysis
# is told to treat as missing. An analysis that supports masking records them
# only as it shows them, masked or not, so it overrides them in `for_call`.
DATA_VALUED = ("sentinel_values", "expected_labels")


@dataclass(frozen=True)
class Call:
    """What one call of an analysis recorded."""

    analysis: str
    method: str | None
    data: dict[str, JsonValue]
    # The validated arguments, defaults included, without `data`.
    arguments: pydantic.BaseModel
    # The arguments of this analysis that hold data values and must be overridden.
    data_valued: tuple[str, ...]

    def parameters(self, overrides: Mapping[str, Any]) -> dict[str, JsonValue]:
        """The arguments as the record shows them: `data` as its shape, each
        argument as the JSON dump of its value, in signature order, except for
        the overridden ones, which show the override."""
        unknown = [
            name for name in overrides if name not in type(self.arguments).model_fields
        ]
        if unknown:
            raise RuntimeError(
                f"agent_eda bug: {self.analysis} overrides {unknown}, which aren't "
                f"its arguments: {list(type(self.arguments).model_fields)}"
            )
        forgotten = [
            name
            for name in self.data_valued
            if name not in overrides and getattr(self.arguments, name) is not None
        ]
        if forgotten:
            raise RuntimeError(
                f"agent_eda bug: {self.analysis} records {forgotten}, which hold "
                "data values, as given: pass each one's masked or resolved value "
                "to for_call(parameters=...), so that masking can't be bypassed"
            )
        dumped = self.arguments.model_dump(mode="json", exclude=set(overrides))
        return {
            "data": self.data,
            **{
                name: overrides[name] if name in overrides else dumped[name]
                for name in type(self.arguments).model_fields
            },
        }


_CURRENT: contextvars.ContextVar[Call | None] = contextvars.ContextVar(
    "agent_eda_call", default=None
)


@contextlib.contextmanager
def recording(call: Call) -> Generator[None]:
    """Make `call` the one that `for_call` records while the block runs."""
    token = _CURRENT.set(call)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def envelope(
    overrides: Mapping[str, Any] | None, libraries: Sequence[str]
) -> dict[str, Any]:
    """The analysis, method, parameters and execution of the call being run."""
    call = _CURRENT.get()
    if call is None:
        raise RuntimeError(
            "agent_eda bug: a result is built for a call only while a registered "
            "analysis runs"
        )
    return {
        "analysis": call.analysis,
        "method": call.method,
        "parameters": call.parameters(overrides or {}),
        "execution": execution((*LIBRARIES, *libraries)),
    }


def data_shape(data: pd.DataFrame) -> dict[str, JsonValue]:
    """How a DataFrame argument appears in a result's parameters."""
    return {"row_count": data.shape[0], "column_count": data.shape[1]}


def execution(libraries: Sequence[str]) -> Execution:
    """The versions of agent_eda, Python and the given libraries, by import name."""
    versions = {"agent_eda": __version__, "python": platform.python_version()}
    for name in libraries:
        versions[name] = str(importlib.import_module(name).__version__)
    return Execution(versions=versions)


class Settings:
    """The recorded parameters of an analysis, each read as its argument is typed.

    The view has one attribute per argument of the analysis. A value that the
    argument's type reads, in lax mode, has that type: a Project is a Project
    and a schema's declarations are ColumnSpec objects. A value it can't read,
    such as a parameter that a hand-edited record holds, stays as recorded, and
    an argument the record lacks is None.
    """

    def __init__(self, analysis: str, parameters: Mapping[str, JsonValue]) -> None:
        self._analysis = analysis
        self._parameters = parameters

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        annotations = _annotations(self._analysis)
        if name not in annotations:
            raise AttributeError(
                f"{self._analysis} has no argument {name!r}; its arguments are "
                f"{list(annotations)}"
            )
        if name not in self._parameters:
            return None
        try:
            return _adapter(self._analysis, name).validate_python(
                self._parameters[name], strict=False
            )
        except pydantic.ValidationError:
            return self._parameters[name]


@functools.cache
def _annotations(analysis: str) -> dict[str, Any]:
    """The annotation of each argument, from the arguments model of the analysis."""
    from agent_eda._registry import arguments_model

    return {
        name: field.annotation
        for name, field in arguments_model(analysis).model_fields.items()
    }


@functools.cache
def _adapter(analysis: str, name: str) -> pydantic.TypeAdapter[Any]:
    annotation = _annotations(analysis)[name]
    try:
        return pydantic.TypeAdapter(annotation)
    except pydantic.errors.PydanticSchemaGenerationError:
        # A value of a pandas type, such as a Period among sentinel values.
        return pydantic.TypeAdapter(
            annotation, config=pydantic.ConfigDict(arbitrary_types_allowed=True)
        )
