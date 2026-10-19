"""The MCP server behind agent_eda.mcp.

It imports the mcp extra at the top, so only agent_eda.mcp imports it, after
checking that the extra is installed. Every analysis call goes through one data
reader, the MCP server's DataReader, and one response builder, _response.
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
from agent_eda._data_files import (
    CSV_RULES,
    PARQUET_RULES,
    DataReader,
    supported_extensions,
)
from agent_eda._local_files import path_argument
from agent_eda._mcp_instructions import GUARDRAIL, PRECEDENCE, instructions
from agent_eda._registry import (
    AnalysisSpec,
    missing_message,
    registered,
    spec_of,
    unknown_name_message,
    validate_json,
)
from agent_eda._sentinels import read_iso
from agent_eda._table_files import (
    RECORD_KEY,
    RECORD_SCHEMA,
    TableFile,
    WrittenTable,
    table_file,
)
from agent_eda._text import closest_names, join_alternatives
from agent_eda._version import __version__
from agent_eda._wording import message as worded_message
from agent_eda._wording import see_every
from agent_eda.discovery import list_analyses, search_analyses
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.project import Project
from agent_eda.results import AnalysisListing, AnalysisResult, SearchListing

_log = logging.getLogger("agent_eda.mcp")

_DATA_PATH = {
    "type": "string",
    "description": "Path to the CSV, TSV or Parquet file that holds the data, "
    "absolute or relative to the MCP server's working directory; a leading ~ is "
    "the home directory. The extension "
    f"names the format: {supported_extensions()}, in any case. In a CSV or TSV "
    "only an empty field counts as missing, so values such as 'N/A' or -999 "
    "reach the analysis as values, and row IDs in the result are 0-based "
    "data-row numbers; in a Parquet file they follow the index stored in it. "
    f"{CSV_RULES} {PARQUET_RULES}",
}

# Why the MCP server makes no enforcement claim (ADR 0006).
_NO_ENFORCEMENT = "Whoever can call the MCP server can usually read the data file too."

_PROJECT_PATH = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
    "default": None,
    "description": "Path to a project TOML file, absolute or relative to the MCP "
    "server's working directory (a leading ~ is the home directory), whose data "
    "policy names the sensitive columns "
    "and says whether to mask them. The file has a [data_policy] table with "
    'sensitive_columns = ["email"] and mask_sensitive = true or false, and an '
    "optional top-level name. null, the default, uses the project the MCP "
    "server was started with (agent-eda-mcp --project), or no project, which "
    f"masks nothing. {PRECEDENCE} {GUARDRAIL} {_NO_ENFORCEMENT}",
}

_TABLE_PATH = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
    "default": None,
    "description": "Path of a new file to write the result's full table to, for "
    "when the capped lists of the structured content aren't enough. It writes "
    "only the main table, the one a Python caller gets from to_frame without "
    "arguments; the other tables some results have, such as "
    "compute_association_matrix's matrix, are for Python callers. The extension "
    "names the format: .csv "
    "for CSV, or .parquet or .pq for Parquet, in any case. The path is absolute "
    "or relative to the MCP server's working directory (a leading ~ is the home "
    "directory), and its directory must exist and be one the MCP server may "
    "write in. The MCP server never overwrites and never writes through a "
    "symbolic link: a path where a file or a link already exists is an error, "
    "raised before the analysis runs, like a directory it may not write in. A "
    "write that fails partway leaves no file. The text of the response then "
    "ends with the line 'Full table written to <absolute path> (<n> rows).' (the "
    "path on a line of its own when the line would pass 120 characters), and "
    "the structured content gains table_file with the same path and row_count. "
    "The index "
    "becomes the first column only when it is more than the row numbers 0, 1, 2, "
    "... A cell that holds a list or a mapping, such as row IDs, is written as "
    "JSON text, and in Parquet a column whose values mix types, such as numbers "
    "and text, is written as text, as in the CSV. In a CSV a missing value is an "
    'empty field and an empty string is the quoted field "", which the MCP '
    "server reads back as an empty string. null, the default, writes no file.",
}

# Follows the docstring's mask_sensitive text, which speaks of the project
# argument a Python caller passes.
_MCP_PROJECT = f"Over MCP: {PRECEDENCE}"

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
    MCP server's instructions carry the guide.
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

    The MCP tools are built here, when the MCP server starts, from the registry
    entry of each analysis: every registered one when analyses is None, and
    otherwise the ones a test passes (a test seam, unsupported for serving). The
    MCP server reads data paths through one DataReader, which keeps the most
    recently read files. project is the MCP server's Project, which every call of
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
                message=_unknown_tool(params.name, [*discovery, *tools]),
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
            # The message of a bug can quote a value of the data, so with
            # masking on only the exception's type reaches the caller.
            reason = type(error).__name__
            if not _masking_on(served, project, arguments):
                reason = f"{reason}: {error}"
            raise MCPError(
                code=types.INTERNAL_ERROR,
                message=f"{params.name} failed with an unexpected error, which is "
                f"a bug in agent-eda, not in the call: {reason}. The MCP server's "
                "log on stderr has the traceback.",
            ) from error

    return Server(
        "agent-eda",
        version=__version__,
        instructions=instructions(),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


# The Python functions that have no MCP tool.
_PYTHON_ONLY = ("guide", "describe_analysis")


def _unknown_tool(name: str, names: list[str]) -> str:
    """Say that no MCP tool has this name, naming the closest and every MCP tool."""
    parts = [f"There is no MCP tool named {name!r}."]
    close = closest_names(name, names)
    if close:
        parts.append(f"Did you mean {join_alternatives([repr(n) for n in close])}?")
    if name in _PYTHON_ONLY:
        parts.append(
            f"{name} is a Python function, not an MCP tool: the MCP server's "
            "instructions hold the guide, and each MCP tool's description and "
            "input schema hold what describe_analysis says."
        )
    parts.append(f"MCP tools: {', '.join(names)}.")
    return " ".join(parts)


def _masking_on(
    served: _AnalysisTool | _DiscoveryTool,
    project: Project | None,
    arguments: dict[str, Any],
) -> bool:
    """Whether the call that failed was masking, or may have been.

    `arguments` are what the call left of them: it pops project_path and puts
    the project it used under "project". A project that was never loaded counts
    as masking, so a bug before that point can't show a value either.
    """
    if not isinstance(served, _AnalysisTool) or not served.spec.supports_masking:
        return False
    if (call := arguments.get("mask_sensitive")) is not None:
        return call is not False
    if "project_path" in arguments:
        return True
    used = arguments.get("project", project)
    return used is not None and used.data_policy.mask_sensitive


def _call(
    served: _AnalysisTool,
    reader: DataReader,
    project: Project | None,
    arguments: dict[str, Any],
) -> types.CallToolResult:
    """Check the arguments, read the data, run the analysis and build the response.

    The SDK passes the arguments on without checking them against the input
    schema. The MCP server checks what the MCP tool adds to the analysis's own
    arguments (unknown and missing names among the MCP tool's properties, and
    data_path) and validates the analysis's options in JSON mode, so each
    reads as its schema says: an ISO 8601 string is a datetime, date or
    duration wherever the schema offers one, and a string in sentinel_values
    that names such a value for its column is read as one. The message of a
    bad option is the one a Python caller gets, its fix worded for an MCP
    caller. A table_path is checked before the data is read, so an
    existing file or a missing directory stops the call before the analysis
    runs. A project_path replaces the MCP server's project for the call; a
    mask_sensitive goes to the analysis, where it wins over either.
    """
    _check_arguments(served, arguments)
    table_path = arguments.pop("table_path", None)
    table = None if table_path is None else table_file(served.spec.name, table_path)
    takes_project = "project_path" in served.tool.input_schema["properties"]
    if takes_project:
        # project_path stays in `arguments` until the project is loaded, and the
        # project replaces it at once, so _masking_on sees the call's masking
        # whichever step fails.
        project_path = arguments.get("project_path")
        if project_path is not None:
            project = _load_project(served.spec.name, project_path)
        arguments.pop("project_path", None)
        arguments["project"] = project
    data = reader.read(arguments.pop("data_path"))
    given = {name: value for name, value in arguments.items() if name != "project"}
    options = validate_json(served.spec, data, given)
    if options.get("sentinel_values") is not None:
        options["sentinel_values"] = read_iso(data, options["sentinel_values"])
    if takes_project:
        options["project"] = project
    return _response(served.function(data, **options), table)


def _discover(
    served: _DiscoveryTool, arguments: dict[str, Any]
) -> types.CallToolResult:
    """Run search_analyses or list_analyses: the listing's text, in MCP wording,
    and its record.

    The MCP server checks the argument names; the function checks the values, with
    the message a Python caller gets, its fix worded for an MCP caller.
    """
    _check_names(
        served.tool,
        arguments,
        missing=lambda names: (
            f"{served.tool.name} is missing the required "
            f"argument {names[0]}. Pass what you want to know in your own words, such "
            'as {"query": "nulls per column"}.'
        ),
    )
    listing = served.function(**arguments)
    return types.CallToolResult(
        content=[types.TextContent(text=listing._render("mcp"))],
        structured_content=listing.to_dict(),
    )


def _load_project(name: str, project_path: object) -> Project:
    """The Project in the file at project_path.

    Project.load raises the errors a Python caller gets, such as
    ProjectFileNotFoundError, so they reach the MCP caller as tool errors.
    """
    path = path_argument(
        name,
        "project_path",
        project_path,
        what="the path to a project TOML file",
        nullable=True,
    )
    return Project.load(path)


def _check_arguments(served: _AnalysisTool, arguments: dict[str, Any]) -> None:
    """Raise InvalidArgumentError for an argument the MCP tool doesn't take, a
    required one left out, or a data path that isn't a string."""

    def missing(names: list[str]) -> str:
        example = ", ".join(f'"{key}": ...' for key in names)
        return missing_message(
            served.spec,
            names,
            fix=f"in the MCP tool's arguments, such as {{{example}}}",
            see=see_every(served.spec.name, "argument", "mcp"),
        )

    _check_names(served.tool, arguments, missing=missing)
    path_argument(
        served.spec.name,
        "data_path",
        arguments["data_path"],
        what="the path to the data file",
        nullable=False,
    )


def _check_names(
    tool: types.Tool,
    arguments: dict[str, Any],
    *,
    missing: Callable[[list[str]], str],
) -> None:
    """Raise invalid_call for an argument the MCP tool doesn't take, naming the
    closest property, or for required ones left out, with the message
    `missing` gives for their names.

    The SDK passes the arguments on without checking them against the input
    schema, so every MCP tool checks the names here.
    """
    properties = list(tool.input_schema["properties"])
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise InvalidArgumentError(
            unknown_name_message(tool.name, "argument", unknown[0], properties)
            if properties
            else f"{tool.name} takes no arguments, but got {unknown[0]!r}. Call it "
            "with {}.",
            code="invalid_call",
        )
    required = tool.input_schema.get("required", [])
    absent = [key for key in required if key not in arguments]
    if absent:
        raise InvalidArgumentError(missing(absent), code="invalid_call")


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
        output_schema=_output_schema(spec),
        annotations=_ANNOTATIONS,
    )
    return _AnalysisTool(spec=spec, function=function, tool=tool)


def _output_schema(spec: AnalysisSpec) -> dict[str, Any]:
    """The JSON Schema of the record, with the optional table_file that a call
    with table_path adds."""
    schema = spec.result_class.record_schema()
    schema["properties"][RECORD_KEY] = RECORD_SCHEMA
    return schema


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
    """The printed result in MCP wording as text, and its to_dict() record as
    structured content.

    Given a table file, it writes the result's full table there first. The
    text gains a line that says where, and the record gains table_file with
    the same facts, since some clients show the model only the structured
    content.
    """
    record = result.to_dict()
    if table is None:
        text = result._render("mcp")
    else:
        frame = result.to_frame()
        # The lines the text gains count toward its 40, so the result leaves room.
        lines = WrittenTable(table.path, len(frame)).lines()
        text = "\n".join([result._render("mcp", reserve=len(lines)), *lines])
        written = table.write(frame)
        record[RECORD_KEY] = written.record()
    return types.CallToolResult(
        content=[types.TextContent(text=text)],
        structured_content=record,
    )


def _error_response(error: AgentEDAError) -> types.CallToolResult:
    """A caller mistake as a tool error: the message, with its fix worded for an
    MCP caller, as text, and the stable code and the message as structured
    content."""
    message = worded_message(error, "mcp")
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        structured_content={"code": error.code, "message": message},
        is_error=True,
    )
