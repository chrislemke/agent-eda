"""The MCP server behind agent_eda.mcp.

It imports the mcp extra at the top, so only agent_eda.mcp imports it, after
checking that the extra is installed. Every analysis call goes through one data
reader, read_data, and one response builder, _response.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import anyio
import pandas as pd
import pydantic
from mcp import MCPError, types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

import agent_eda
from agent_eda._annotations import type_text
from agent_eda._registry import (
    AnalysisSpec,
    missing_message,
    registered,
    spec_of,
    unknown_name_message,
)
from agent_eda._version import __version__
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.results import AnalysisResult

_log = logging.getLogger("agent_eda.mcp")

_DATA_PATH = {
    "type": "string",
    "description": "Path to the CSV file that holds the data, absolute or relative "
    "to the MCP server's working directory. Only an empty field counts as "
    "missing, so values such as 'N/A' or -999 reach the analysis as values. "
    "Row IDs in the result are 0-based data-row numbers.",
}

# Every analysis MCP tool may write a file (table_path), never overwrites one,
# gives the same result for the same call and reaches nothing outside the
# machine.
_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


@dataclass(frozen=True)
class _AnalysisTool:
    """An analysis served as an MCP tool."""

    spec: AnalysisSpec
    function: Callable[..., AnalysisResult]
    tool: types.Tool


def create_server(
    analyses: Sequence[Callable[..., AnalysisResult]] | None = None,
) -> Server:
    """The MCP server named 'agent-eda', with one MCP tool per analysis.

    The MCP tools are built here, when the server starts, from the registry
    entry of each analysis: every registered one when analyses is None.
    """
    if analyses is None:
        analyses = [getattr(agent_eda, spec.name) for spec in registered()]
    tools: dict[str, _AnalysisTool] = {}
    for function in analyses:
        served = _analysis_tool(spec_of(function), function)
        tools[served.tool.name] = served

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[served.tool for served in tools.values()])

    async def call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        served = tools.get(params.name)
        if served is None:
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"There is no MCP tool named {params.name!r}.",
            )
        try:
            return _call(served, dict(params.arguments or {}))
        except AgentEDAError as error:
            return _error_response(error)
        except Exception as error:
            # A bug. Without this, the SDK answers a client that did the
            # initialize handshake with error code 0, not an internal error.
            _log.exception("%s failed with an unexpected error", params.name)
            raise MCPError(
                code=types.INTERNAL_ERROR,
                message=f"{params.name} failed with an unexpected error, which is "
                f"a bug in agent-eda, not in the call: {type(error).__name__}: "
                f"{error}. The MCP server's log on stderr has the traceback.",
            ) from error

    return Server(
        "agent-eda",
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _call(served: _AnalysisTool, arguments: dict[str, Any]) -> types.CallToolResult:
    """Check the arguments, read the data, run the analysis and build the response.

    The SDK passes the arguments on without checking them against the input
    schema. The server checks what the MCP tool adds to the analysis's own
    arguments (unknown and missing names among the MCP tool's properties, and
    data_path); the analysis checks its options, with the message a Python
    caller gets.
    """
    _check_arguments(served, arguments)
    data = read_data(arguments.pop("data_path"))
    return _response(served.function(data, **arguments))


def _check_arguments(served: _AnalysisTool, arguments: dict[str, Any]) -> None:
    """Raise InvalidArgumentError for an argument the MCP tool doesn't take, a
    required one left out, or a data path that isn't a string."""
    name = served.spec.name
    schema = served.tool.input_schema
    properties = list(schema["properties"])
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise InvalidArgumentError(
            unknown_name_message(name, "argument", unknown[0], properties),
            code="invalid_call",
        )
    missing = [key for key in schema["required"] if key not in arguments]
    if missing:
        example = ", ".join(f'"{key}": ...' for key in missing)
        raise InvalidArgumentError(
            missing_message(
                served.spec,
                missing,
                fix=f"in the MCP tool's arguments, such as {{{example}}}",
                see="The MCP tool's input schema describes every argument.",
            ),
            code="invalid_call",
        )
    data_path = arguments["data_path"]
    if not isinstance(data_path, str):
        raise InvalidArgumentError(
            f"{name} got {data_path!r} for data_path, but data_path must be a "
            "string: the path to the data file, absolute or relative to the MCP "
            "server's working directory."
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


def _analysis_tool(
    spec: AnalysisSpec, function: Callable[..., AnalysisResult]
) -> _AnalysisTool:
    tool = types.Tool(
        name=spec.name,
        description=_description(spec),
        input_schema=_input_schema(spec),
        output_schema=spec.result_class.record_schema(),
        annotations=_ANNOTATIONS,
    )
    return _AnalysisTool(spec=spec, function=function, tool=tool)


def _input_schema(spec: AnalysisSpec) -> dict[str, Any]:
    """The options' JSON Schema, with the required data_path in place of data.

    Raises TypeError, naming the analysis and the parameter, when a parameter's
    type has no JSON Schema, since no MCP client could pass a value for it.
    """
    try:
        schema = spec.parameters_schema()
    except pydantic.errors.PydanticInvalidForJsonSchema:
        raise TypeError(_no_json_schema(spec)) from None
    schema["properties"] = {"data_path": _DATA_PATH, **schema.get("properties", {})}
    schema["required"] = ["data_path", *schema.get("required", [])]
    return schema


def _no_json_schema(spec: AnalysisSpec) -> str:
    """Say which parameter has a type without a JSON Schema."""
    for name in spec.parameters[1:]:
        annotation = spec.annotations[name]
        field: dict[str, Any] = {"value": (annotation, ...)}
        probe = pydantic.create_model(
            "Probe",
            __config__=pydantic.ConfigDict(arbitrary_types_allowed=True),
            **field,
        )
        try:
            probe.model_json_schema()
        except pydantic.errors.PydanticInvalidForJsonSchema:
            return (
                f"{spec.name} can't be an MCP tool: its parameter {name} has the "
                f"type {type_text(annotation)}, which has no JSON Schema, so an MCP "
                f"client can't pass a value for it. Give {name} a type that JSON "
                "can hold, such as str, a number, a list or a Literal of choices."
            )
    raise AssertionError(f"{spec.name} has a parameter without a JSON Schema")


def _description(spec: AnalysisSpec) -> str:
    """The docstring an MCP caller needs: summary, When to use:, Returns:,
    Statuses: and See Also:, in that order. Args: goes into the input schema,
    and Examples: is left out because it is Python."""
    doc = spec.docstring
    statuses = [f"{status}: {doc.statuses.get(status, '')}" for status in spec.statuses]
    see_also = [f"{name}: {text}" for name, text in doc.see_also.items()]
    return "\n\n".join(
        [
            doc.summary,
            f"When to use:\n{doc.when_to_use}",
            f"Returns:\n{doc.returns}",
            "Statuses:\n" + "\n".join(statuses),
            "See Also:\n" + "\n".join(see_also),
        ]
    )


def _response(result: AnalysisResult) -> types.CallToolResult:
    """The printed result as text, and its to_dict() record as structured content."""
    return types.CallToolResult(
        content=[types.TextContent(text=str(result))],
        structured_content=result.to_dict(),
    )


def _error_response(error: AgentEDAError) -> types.CallToolResult:
    """A caller mistake as a tool error: the message as text, and the stable
    code and the message as structured content."""
    message = str(error)
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        structured_content={"code": error.code, "message": message},
        is_error=True,
    )
