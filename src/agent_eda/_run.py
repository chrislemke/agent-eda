"""Pieces every analysis puts into its result."""

import importlib
import platform
from collections.abc import Sequence

import pandas as pd
from pydantic import JsonValue

from agent_eda._version import __version__
from agent_eda.results import Execution


def data_shape(data: pd.DataFrame) -> dict[str, JsonValue]:
    """How a DataFrame argument appears in a result's parameters."""
    return {"row_count": data.shape[0], "column_count": data.shape[1]}


def execution(libraries: Sequence[str]) -> Execution:
    """The versions of agent_eda, Python and the given libraries, by import name."""
    versions = {"agent_eda": __version__, "python": platform.python_version()}
    for name in libraries:
        versions[name] = str(importlib.import_module(name).__version__)
    return Execution(versions=versions)
