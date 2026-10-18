"""The MCP server behind agent_eda.mcp.

It imports the mcp extra at the top, so only agent_eda.mcp imports it, after
checking that the extra is installed. Every analysis call goes through one data
reader, read_data, and one response builder, _response.
"""

from typing import Any

import anyio
import pandas as pd
from mcp import MCPError, types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

import agent_eda
from agent_eda._version import __version__
from agent_eda.discovery import describe_analysis
from agent_eda.results import AnalysisResult

# The analyses served as MCP tools.
_ANALYSES = ("describe_dataset",)

_DATA_PATH = {
    "type": "string",
    "description": "Path to the CSV file that holds the data, absolute or relative "
    "to the MCP server's working directory. Only an empty field counts as "
    "missing, so values such as 'N/A' or -999 reach the analysis as values. "
    "Row IDs in the result are 0-based data-row numbers.",
}


def create_server() -> Server:
    """The MCP server named 'agent-eda', with one MCP tool per served analysis."""
    return Server(
        "agent-eda",
        version=__version__,
        on_list_tools=_list_tools,
        on_call_tool=_call_tool,
    )


def run_stdio(server: Server) -> None:
    """Serve over stdin and stdout until the client closes the connection."""

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)


def read_data(data_path: str) -> pd.DataFrame:
    """Read the DataFrame an MCP tool's data path names.

    Only an empty field is missing, so stand-ins such as 'N/A' stay values. The
    index is a RangeIndex, so row IDs are 0-based data-row numbers.
    """
    return pd.read_csv(data_path, keep_default_na=False, na_values=[""])


def _tool(name: str) -> types.Tool:
    return types.Tool(
        name=name,
        description=describe_analysis(name).summary,
        input_schema={
            "type": "object",
            "properties": {"data_path": _DATA_PATH},
            "required": ["data_path"],
            "additionalProperties": False,
        },
    )


async def _list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(tools=[_tool(name) for name in _ANALYSES])


async def _call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    if params.name not in _ANALYSES:
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"There is no MCP tool named {params.name!r}.",
        )
    arguments: dict[str, Any] = dict(params.arguments or {})
    data = read_data(arguments.pop("data_path"))
    result = getattr(agent_eda, params.name)(data, **arguments)
    return _response(result)


def _response(result: AnalysisResult) -> types.CallToolResult:
    """The printed result as text, and its to_dict() record as structured content."""
    return types.CallToolResult(
        content=[types.TextContent(text=str(result))],
        structured_content=result.to_dict(),
    )
