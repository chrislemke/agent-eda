"""The one place a test serves analyses other than the registered ones."""

from collections.abc import Callable

import agent_eda.mcp
from agent_eda.project import Project
from agent_eda.results import AnalysisResult


def serving(*analyses: Callable[..., AnalysisResult], project: Project | None = None):
    """An MCP server that serves only these analysis functions, such as a
    stand-in the test defines, through the hook that create_server keeps
    private."""
    return agent_eda.mcp._create_server(analyses=list(analyses), project=project)
