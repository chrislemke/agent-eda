"""The MCP server, which makes the analyses callable over the Model Context Protocol.

Agents that call tools instead of writing Python reach the analyses through it
(ADR 0006). Install the mcp extra and register the console script with an MCP
client, which starts it as a subprocess and talks to it over stdio:

    pip install 'agent-eda[mcp]'
    claude mcp add agent-eda -- agent-eda-mcp

Each MCP tool takes a data path in place of the DataFrame. The server reads a
CSV file counting only an empty field as missing, so stand-ins such as 'N/A'
reach the analyses as values. A call returns the printed result as text and the
result's to_dict() record as structured content.

The package never imports this module, so `import agent_eda` stays free of the
mcp extra. Without the extra, create_server() raises MissingDependencyError and
agent-eda-mcp prints its install commands.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from agent_eda.errors import AgentEDAError, MissingDependencyError

if TYPE_CHECKING:
    from mcp.server import Server

    from agent_eda.results import AnalysisResult

__all__ = ["create_server", "main"]

# The packages of the mcp extra that the server imports.
_EXTRA_MODULES = ("mcp", "mcp_types", "anyio")


def create_server(
    *, analyses: "Sequence[Callable[..., AnalysisResult]] | None" = None
) -> "Server":
    """Build the MCP server, ready to run over any MCP transport.

    agent-eda-mcp runs it over stdio. Tests and embedding code can connect an
    in-memory MCP client to it instead. Each MCP tool is built when the server
    starts, from the analysis's registry entry and docstring: its name, a
    description, an input schema with data_path in place of data, and an
    output schema of the analysis's to_dict() record.

    Args:
        analyses: The analysis functions to serve, such as
            [ae.summarize_numeric]. None, the default, serves every analysis
            that ae.list_analyses() lists. A function that agent_eda doesn't
            register, such as a stand-in a test defines, is read the way the
            registry reads an analysis (data first, options by keyword, a
            docstring and an AnalysisResult subclass as the return type) and
            is called as it is, without the registry's checks.

    Returns:
        mcp.server.Server: The low-level MCP server named 'agent-eda', which
        reports the package version and serves one MCP tool per analysis.

    Raises:
        MissingDependencyError: The mcp extra isn't installed. The message gives
            the pip and uv install commands.
        TypeError: An analysis can't be an MCP tool, such as one with a
            parameter whose type has no JSON Schema. The message names the
            analysis and the parameter.
    """
    _require_extra()
    from agent_eda._mcp_server import create_server as create

    return create(analyses)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MCP server over stdio until the client closes the connection.

    This is the agent-eda-mcp console script. A mistake that stops the server
    from starting, such as a missing mcp extra, prints its message to stderr
    and exits with status 1, without a traceback.

    Args:
        argv: The command-line arguments after the program name; None reads
            them from sys.argv.
    """
    _parser().parse_args(argv)
    try:
        server = create_server()
    except AgentEDAError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None
    from agent_eda._mcp_server import run_stdio

    run_stdio(server)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="agent-eda-mcp",
        description="Serve the agent-eda analyses as MCP tools over stdio. An MCP "
        "client starts this command as a subprocess; register it with, for "
        "example, claude mcp add agent-eda -- agent-eda-mcp.",
    )


def _require_extra() -> None:
    """Import the server's implementation, which needs the mcp extra."""
    try:
        import agent_eda._mcp_server  # noqa: F401  # imported for the check
    except ImportError as error:
        if (error.name or "").split(".")[0] not in _EXTRA_MODULES:
            raise
        raise MissingDependencyError(
            "The agent-eda MCP server needs the optional mcp extra, which isn't "
            f"installed ({error}). Install it with pip install 'agent-eda[mcp]', "
            "or in a uv project with uv add 'agent-eda[mcp]', then start "
            "agent-eda-mcp again.",
        ) from None
