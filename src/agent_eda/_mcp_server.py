"""The MCP server behind agent_eda.mcp.

It imports the mcp extra at the top, so only agent_eda.mcp imports it, after
checking that the extra is installed. Every analysis call goes through one data
reader, the server's DataReader, and one response builder, _response.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import anyio
import pydantic
from mcp import MCPError, types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

import agent_eda
from agent_eda._annotations import type_text
from agent_eda._data_files import DataReader, supported_extensions
from agent_eda._registry import AnalysisSpec, registered, spec_of
from agent_eda._version import __version__
from agent_eda.results import AnalysisResult

_DATA_PATH = {
    "type": "string",
    "description": "Path to the CSV, TSV or Parquet file that holds the data, "
    "absolute or relative to the MCP server's working directory. The extension "
    f"names the format: {supported_extensions()}, in any case. In a CSV or TSV "
    "only an empty field counts as missing, so values such as 'N/A' or -999 "
    "reach the analysis as values, and row IDs in the result are 0-based "
    "data-row numbers. Parquet keeps its dtypes, and row IDs follow the index "
    "stored in the file.",
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
    entry of each analysis: every registered one when analyses is None. The
    server reads data paths through one DataReader, which keeps the most
    recently read files.
    """
    reader = DataReader()
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
        arguments: dict[str, Any] = dict(params.arguments or {})
        data = reader.read(arguments.pop("data_path"))
        return _response(served.function(data, **arguments))

    return Server(
        "agent-eda",
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def run_stdio(server: Server) -> None:
    """Serve over stdin and stdout until the client closes the connection."""

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)


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
