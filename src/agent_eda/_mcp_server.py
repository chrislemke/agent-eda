"""The MCP server behind agent_eda.mcp.

It imports the mcp extra at the top, so only agent_eda.mcp imports it, after
checking that the extra is installed. Every analysis call goes through one data
reader, the server's DataReader, and one response builder, _response.
"""

import inspect
import logging
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
from agent_eda._mcp_instructions import instructions
from agent_eda._registry import (
    AnalysisSpec,
    missing_message,
    registered,
    spec_of,
    unknown_name_message,
)
from agent_eda._table_files import TableFile, table_file
from agent_eda._version import __version__
from agent_eda.discovery import list_analyses, search_analyses
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.project import Project
from agent_eda.results import AnalysisListing, AnalysisResult, SearchListing

_log = logging.getLogger("agent_eda.mcp")

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

_GUARDRAIL = (
    "Masking is a guardrail against accidental exposure, not anonymization or "
    "access control: whoever can call the MCP server can usually read the data "
    "file too."
)

_PROJECT_PATH = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
    "default": None,
    "description": "Path to a project TOML file, absolute or relative to the MCP "
    "server's working directory, whose data policy names the sensitive columns "
    "and says whether to mask them. The file has a [data_policy] table with "
    'sensitive_columns = ["email"] and mask_sensitive = true or false, and an '
    "optional top-level name. null, the default, uses the project the MCP "
    "server was started with (agent-eda-mcp --project), or no project, which "
    "masks nothing. A project_path replaces the MCP server's project for this "
    "call. "
    f"{_GUARDRAIL}",
}

_TABLE_PATH = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
    "default": None,
    "description": "Path of a new file to write the result's complete table to, "
    "the table a Python caller gets from to_frame, for when the capped lists of "
    "the structured content aren't enough. The extension names the format: .csv "
    "for CSV, or .parquet or .pq for Parquet, in any case. The path is absolute "
    "or relative to the MCP server's working directory, and its directory must "
    "exist. The MCP server never overwrites: a path where a file already exists "
    "is an error, raised before the analysis runs. The text of the response then ends "
    "with the line 'Full table written to <absolute path> (<n> rows).' The index "
    "becomes the first column only when it is more than the row numbers 0, 1, 2, "
    "... A cell that holds a list or a mapping, such as row IDs, is written as "
    "JSON text, and in Parquet a column whose values mix types, such as numbers "
    "and text, is written as text, as in the CSV. null, the default, writes no "
    "file.",
}

# Follows the docstring's mask_sensitive text, which speaks of the project
# argument a Python caller passes.
_MCP_PROJECT = (
    "Over MCP, the project is the file at project_path, or else the one the "
    "MCP server was started with (agent-eda-mcp --project)."
)

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


# search_analyses and list_analyses only read the registry.
_DISCOVERY_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

_SEARCH_DESCRIPTION = """\
{summary}

The query's words, without common ones such as 'the' and 'of', are matched against each analysis's name, tags, \
summary and When to use: text. A few synonyms help: nulls and NaN find missing values, dupes find duplicates, \
correlation finds association and label finds target. The search is keyword based, so the same query always gives \
the same ranking.

Returns:
The text lists the best matches, best first, as name: summary with the words each matched. The structured content \
holds matches, each with its name, module, summary, score and matched_terms; match_count, which counts every match, \
also those beyond limit; and terms, the words searched for with their synonyms. Each name is an MCP tool name."""

_LIST_DESCRIPTION = """\
{summary}

Returns:
The text lists one line per analysis, name: summary, grouped under its module; the summary is the question the \
analysis answers, and [supports masking] marks the MCP tools that take project_path. The structured content's \
analyses hold what each analysis declares: parameters, methods, statuses, finding_codes, result_class and \
work_limit_parameters. Each name is an MCP tool name; parameters gives the Python names, in which data is data_path \
and project is project_path over MCP."""


@dataclass(frozen=True)
class _DiscoveryTool:
    """A discovery function served as an MCP tool: it takes JSON arguments and
    returns a listing whose text and record make the response."""

    function: Callable[..., AnalysisListing | SearchListing]
    tool: types.Tool


def _discovery_tools() -> dict[str, _DiscoveryTool]:
    """The search_analyses and list_analyses MCP tools, by name.

    describe_analysis and guide are not MCP tools: each analysis MCP tool's
    description and input schema carry what describe_analysis says, and the
    server's instructions carry the guide.
    """
    search = types.Tool(
        name="search_analyses",
        description=_SEARCH_DESCRIPTION.format(summary=_summary(search_analyses)),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you want to know, in your own words, "
                    "such as 'nulls per column'.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "The most matches to return, at least 1; "
                    "defaults to 5.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema=SearchListing.model_json_schema(),
        annotations=_DISCOVERY_ANNOTATIONS,
    )
    listing = types.Tool(
        name="list_analyses",
        description=_LIST_DESCRIPTION.format(summary=_summary(list_analyses)),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_schema=AnalysisListing.model_json_schema(),
        annotations=_DISCOVERY_ANNOTATIONS,
    )
    return {
        "search_analyses": _DiscoveryTool(function=search_analyses, tool=search),
        "list_analyses": _DiscoveryTool(function=list_analyses, tool=listing),
    }


def _summary(function: Callable[..., object]) -> str:
    """The first line of a function's docstring."""
    return (inspect.getdoc(function) or "").split("\n")[0]


def create_server(
    analyses: Sequence[Callable[..., AnalysisResult]] | None = None,
    project: Project | None = None,
) -> Server:
    """The MCP server named 'agent-eda', with one MCP tool per analysis.

    The MCP tools are built here, when the server starts, from the registry
    entry of each analysis: every registered one when analyses is None. The
    server reads data paths through one DataReader, which keeps the most
    recently read files. project is the server's Project, which every call of
    an MCP tool with project_path uses unless the call names its own. The
    discovery MCP tools and the instructions read the registry, as their
    Python counterparts do.
    """
    reader = DataReader()
    if analyses is None:
        analyses = [getattr(agent_eda, spec.name) for spec in registered()]
    tools: dict[str, _AnalysisTool] = {}
    for function in analyses:
        served = _analysis_tool(spec_of(function), function)
        tools[served.tool.name] = served
    discovery = _discovery_tools()

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[served.tool for served in [*discovery.values(), *tools.values()]]
        )

    async def call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        served = tools.get(params.name) or discovery.get(params.name)
        if served is None:
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"There is no MCP tool named {params.name!r}.",
            )
        arguments = dict(params.arguments or {})
        try:
            if isinstance(served, _DiscoveryTool):
                return _discover(served, arguments)
            return _call(served, reader, project, arguments)
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
        instructions=instructions(),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _call(
    served: _AnalysisTool,
    reader: DataReader,
    project: Project | None,
    arguments: dict[str, Any],
) -> types.CallToolResult:
    """Check the arguments, read the data, run the analysis and build the response.

    The SDK passes the arguments on without checking them against the input
    schema. The server checks what the MCP tool adds to the analysis's own
    arguments (unknown and missing names among the MCP tool's properties, and
    data_path); the analysis checks its options, with the message a Python
    caller gets. A table_path is checked before the data is read, so an
    existing file or a missing directory stops the call before the analysis
    runs. A project_path replaces the server's project for the call; a
    mask_sensitive goes to the analysis, where it wins over either.
    """
    _check_arguments(served, arguments)
    table_path = arguments.pop("table_path", None)
    table = None if table_path is None else table_file(served.spec.name, table_path)
    if "project_path" in served.tool.input_schema["properties"]:
        project_path = arguments.pop("project_path", None)
        if project_path is not None:
            project = _load_project(served.spec.name, project_path)
        arguments["project"] = project
    data = reader.read(arguments.pop("data_path"))
    return _response(served.function(data, **arguments), table)


def _discover(
    served: _DiscoveryTool, arguments: dict[str, Any]
) -> types.CallToolResult:
    """Run search_analyses or list_analyses: the listing's text and record.

    The server checks the argument names; the function checks the values, with
    the message a Python caller gets.
    """
    name = served.tool.name
    properties = list(served.tool.input_schema["properties"])
    unknown = [key for key in arguments if key not in properties]
    if unknown and not properties:
        raise InvalidArgumentError(
            f"{name} takes no arguments, but got {unknown[0]!r}. Call it with {{}}.",
            code="invalid_call",
        )
    if unknown:
        raise InvalidArgumentError(
            unknown_name_message(name, "argument", unknown[0], properties),
            code="invalid_call",
        )
    missing = [
        key
        for key in served.tool.input_schema.get("required", [])
        if key not in arguments
    ]
    if missing:
        raise InvalidArgumentError(
            f"{name} is missing the required argument {missing[0]}. Pass what you "
            'want to know in your own words, such as {"query": "nulls per column"}.',
            code="invalid_call",
        )
    listing = served.function(**arguments)
    return types.CallToolResult(
        content=[types.TextContent(text=str(listing))],
        structured_content=listing.to_dict(),
    )


def _load_project(name: str, project_path: object) -> Project:
    """The Project in the file at project_path.

    Project.load raises the errors a Python caller gets, such as
    ProjectFileNotFoundError, so they reach the MCP caller as tool errors.
    """
    if not isinstance(project_path, str):
        raise InvalidArgumentError(
            f"{name} got {project_path!r} for project_path, but project_path must "
            "be a string or null: the path to a project TOML file, absolute or "
            "relative to the MCP server's working directory."
        )
    return Project.load(project_path)


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
    properties = {"data_path": _DATA_PATH}
    for name, value in schema.get("properties", {}).items():
        if name == "project":
            properties["project_path"] = dict(_PROJECT_PATH)
        elif name == "mask_sensitive":
            description = f"{value['description']} {_MCP_PROJECT}"
            properties[name] = {**value, "description": description}
        else:
            properties[name] = value
    properties["table_path"] = dict(_TABLE_PATH)
    schema["properties"] = properties
    schema["required"] = ["data_path", *schema.get("required", [])]
    definitions = schema.get("$defs", {})
    for name in ("Project", "DataPolicy"):
        definitions.pop(name, None)
    if "$defs" in schema and not definitions:
        del schema["$defs"]
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


def _response(
    result: AnalysisResult, table: TableFile | None = None
) -> types.CallToolResult:
    """The printed result as text, and its to_dict() record as structured content.

    Given a table file, it writes the result's full table there first, and the
    text gains a line that says where.
    """
    text = str(result)
    if table is not None:
        text = f"{text}\n{table.write(result.to_frame())}"
    return types.CallToolResult(
        content=[types.TextContent(text=text)],
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
