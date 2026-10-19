"""The one place a test serves analyses other than the registered ones."""

from collections.abc import Callable

import pandas as pd

import agent_eda.mcp
from agent_eda.project import Project
from agent_eda.results import AnalysisResult


def serving(*analyses: Callable[..., AnalysisResult], project: Project | None = None):
    """An MCP server that serves only these analysis functions, such as a
    stand-in the test defines, through the hook that create_server keeps
    private."""
    return agent_eda.mcp._create_server(analyses=list(analyses), project=project)


def stand_in_analyses(count: int) -> list[Callable[..., AnalysisResult]]:
    """`count` stand-in analyses, spread over the data and statistics areas."""
    modules = ["agent_eda.data.stand_ins", "agent_eda.statistics.stand_ins"]
    functions = []
    for number in range(count):

        def stand_in(data: pd.DataFrame) -> AnalysisResult:
            raise AssertionError("a stand-in is listed, never called")

        stand_in.__name__ = f"stand_in_{number:02d}"
        stand_in.__module__ = modules[number % len(modules)]
        stand_in.__doc__ = (
            f"What does stand-in analysis {number} answer?\n\n"
            "Statuses:\n    completed: Never; a stand-in only fills the listing.\n"
        )
        functions.append(stand_in)
    return functions


def with_stand_ins(count: int):
    """A context manager that lists `count` stand-in analyses as registered ones,
    through the hook that the registry keeps private."""
    return agent_eda.mcp._stand_ins(stand_in_analyses(count))
