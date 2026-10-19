"""The environment of an agent-eda-mcp process that a test starts over stdio."""

import os


def stdio_environment(**extra: str) -> dict[str, str]:
    """The variables to pass a StdioServerParameters, on top of the client's defaults.

    An MCP client gives the server process only a few environment variables. Under
    `uv run pytest --cov`, coverage tells subprocesses to measure themselves through
    COVERAGE_* variables, so these are passed on, and `agent_eda.mcp.main` shows as
    covered. Without coverage the result holds only the extra variables.
    """
    coverage = {k: v for k, v in os.environ.items() if k.startswith("COVERAGE_")}
    return {**coverage, **extra}
