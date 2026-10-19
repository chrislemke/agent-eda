"""The MCP server, which makes the analyses callable over the Model Context Protocol.

Agents that call tools instead of writing Python reach the analyses through it
(ADR 0006). Install the mcp extra and register the console script with an MCP
client, which starts it as a subprocess and talks to it over stdio:

    pip install 'agent-eda[mcp]'
    claude mcp add agent-eda -- agent-eda-mcp

Each MCP tool takes a data path in place of the DataFrame. The MCP server reads
a CSV file counting only an empty field as missing, so stand-ins such as 'N/A'
reach the analyses as values. A call returns the printed result as text and the
result's to_dict() record as structured content, also when the run doesn't
complete, such as with status insufficient_data. The text is worded for an MCP
caller: where print(res) names res.to_frame() and res.to_json(), it names
table_path and the structured content. A call with table_path writes the
result's main table to that file and adds its path and row count to both: a
last line of text and a table_file field in the structured content, since some
clients show the model only one of the two. A caller mistake, an
AgentEDAError, comes back as a tool error (isError true) whose text is the
exception's message and whose structured content is {"code", "message"}. Any
other exception is a bug and becomes a JSON-RPC internal error.

The MCP tools of analyses that take a project have project_path in its place:
the path to a project TOML file, as ae.Project.save writes it. Started as
agent-eda-mcp --project project.toml, the MCP server uses that Project for
every such call. A call's project_path replaces the MCP server's project
(agent-eda-mcp --project) for that call, and a call's mask_sensitive wins over
both. Masking is a guardrail against accidental exposure, not anonymization or
access control (ADR 0001).

The package never imports this module, so `import agent_eda` stays free of the
mcp extra. Without the extra, create_server() raises MissingDependencyError and
agent-eda-mcp prints its install commands.
"""

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from agent_eda._mcp_instructions import GUARDRAIL, PRECEDENCE
from agent_eda.errors import AgentEDAError, MissingDependencyError
from agent_eda.project import Project

if TYPE_CHECKING:
    from mcp.server import Server

    from agent_eda.results import AnalysisResult

__all__ = ["create_server", "main"]

# The packages of the mcp extra that the MCP server imports.
_EXTRA_MODULES = ("mcp", "mcp_types", "anyio")


def create_server(
    *,
    project: Project | None = None,
) -> "Server":
    """Build the MCP server, ready to run over any MCP transport.

    agent-eda-mcp runs it over stdio. Tests and embedding code can connect an
    in-memory MCP client to it instead. Each MCP tool is built when the MCP
    server starts, from the analysis's registry entry and docstring: its name, a
    description, an input schema with data_path in place of data, and an
    output schema of the analysis's to_dict() record, with the table_file a
    call with table_path adds. The search_analyses and list_analyses MCP tools
    return the Python listings, worded for an MCP caller, and the MCP server's
    instructions are the guide for an MCP caller: what the package does, data
    paths, results and statuses, masking, every analysis and the first-look
    order. describe_analysis and guide are not MCP tools.

    Args:
        project: The MCP server's Project, such as
            ae.Project.load('project.toml'), which agent-eda-mcp --project
            loads. Every call of an MCP tool that has project_path uses it. A
            call's project_path replaces it for that call, and a call's
            mask_sensitive wins over both. MCP tools of analyses that don't
            take a project ignore it. None, the default, gives no Project.
            Masking is a guardrail against accidental exposure, not
            anonymization or access control.

    Returns:
        mcp.server.Server: The low-level MCP server named 'agent-eda', which
        reports the package version and its instructions, and serves the
        search_analyses and list_analyses MCP tools and one MCP tool per
        analysis.

    Raises:
        MissingDependencyError: The mcp extra isn't installed. The message gives
            the pip and uv install commands.
        TypeError: An analysis can't be an MCP tool, such as one with a
            parameter whose type has no JSON Schema. The message names the
            analysis and the parameter.
    """
    return _create_server(project=project)


def _create_server(
    *,
    analyses: "Sequence[Callable[..., AnalysisResult]] | None" = None,
    project: Project | None = None,
) -> "Server":
    """create_server, with the test hook that serves other analyses.

    `analyses` are the analysis functions to serve in place of every registered
    one, such as [ae.summarize_numeric] or a stand-in a test defines. An
    unregistered function is read the way the registry reads an analysis (data
    first, options by keyword, a docstring and an AnalysisResult subclass as the
    return type) and is called as it is, without the registry's checks. The
    discovery MCP tools and the instructions still cover every registered
    analysis. Only tests use it.
    """
    _require_extra()
    from agent_eda._mcp_server import create_server as create

    return create(analyses, project)


def _stand_ins(
    analyses: "Iterable[Callable[..., AnalysisResult]]",
) -> "AbstractContextManager[None]":
    """The test hook that lists stand-in analyses as registered while it runs.

    The guide and the instructions read the registry, so a test enters it to
    see them with many more analyses. Only tests use it.
    """
    from agent_eda._registry import listed_as_registered

    return listed_as_registered(analyses)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MCP server over stdio until the client closes the connection.

    This is the agent-eda-mcp console script. --project PATH loads the MCP
    server's Project once, at start. A mistake that stops the MCP server from
    starting, such as a missing mcp extra or a project file that
    ae.Project.load refuses or can't read, prints its message to stderr and
    exits with status 1, without a traceback.

    Args:
        argv: The command-line arguments after the program name; None reads
            them from sys.argv.
    """
    options = _parser().parse_args(argv)
    try:
        _require_extra()
        project = None if options.project is None else Project.load(options.project)
        server = create_server(project=project)
    except AgentEDAError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None
    from agent_eda._mcp_server import run_stdio

    run_stdio(server)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eda-mcp",
        description="Serve the agent-eda analyses as MCP tools over stdio. An MCP "
        "client starts this command as a subprocess; register it with, for "
        "example, claude mcp add agent-eda -- agent-eda-mcp.",
    )
    parser.add_argument(
        "--project",
        metavar="PATH",
        help="A project TOML file, as ae.Project.save writes it, whose data policy "
        "names the sensitive columns and says whether to mask them. The MCP "
        "server loads it once, at start, for every call of an MCP tool that "
        f"takes project_path. {PRECEDENCE} An invalid file stops the MCP server "
        f"with its message. {GUARDRAIL}",
    )
    return parser


def _require_extra() -> None:
    """Import the MCP server's implementation, which needs the mcp extra."""
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
