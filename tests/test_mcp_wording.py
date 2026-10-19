"""No Python in what an MCP caller reads: tool descriptions, input schemas, messages.

An MCP caller passes JSON arguments and file paths, so nothing it reads may hold
Python syntax. These guards scan every MCP tool's description and input schema,
the MCP instructions, and the message of a mistake for every error code an MCP
tool can raise (tests/test_mcp_argument_messages.py and tests/test_mcp_fixes.py
hold the mistakes).
"""

import re

import pytest

pytest.importorskip(
    "mcp",
    reason=(
        "the MCP server tests need the mcp extra: run `uv sync` (uv 0.13 or later) or "
        "uv run --extra mcp pytest in this repository, or "
        "pip install 'agent-eda[mcp]' for an installed package"
    ),
)

import anyio
from mcp import Client
from python_syntax import python_in, strings_in
from test_mcp_argument_messages import MISTAKES, arguments_of
from test_mcp_argument_messages import call as call_mcp

import agent_eda as ae
from agent_eda.mcp import create_server

# The codes a Python caller can meet and an MCP tool never raises, and why.
NOT_OVER_MCP = {
    "unsupported_data_type": "an MCP tool always reads a DataFrame from the file",
    "unknown_analysis": "only describe_analysis raises it; search_analyses doesn't",
    "project_file_exists": "only project.save raises it, and no MCP tool saves",
    "result_table_unavailable": "only to_frame on a rebuilt result raises it",
    "missing_dependency": "an MCP tool runs only where the mcp extra is installed",
}

# The unit of an error code's documentation: "    code (Class): text".
CODE = re.compile(r"^    (\w+) \(", flags=re.MULTILINE)


async def listed():
    async with Client(create_server(), mode="legacy") as client:
        return client.instructions, (await client.list_tools()).tools


@pytest.fixture(scope="module")
def served():
    return anyio.run(listed)


def test_no_mcp_tool_description_or_input_schema_holds_python_syntax(served):
    _, tools = served

    leaks = [
        (tool.name, path, found)
        for tool in tools
        for path, text in [
            ("description", tool.description),
            *strings_in(tool.input_schema, "input_schema"),
        ]
        for found in python_in(text)
    ]

    assert leaks == []


def test_the_mcp_instructions_hold_no_python_syntax(served):
    instructions, _ = served

    assert python_in(instructions) == []


def test_no_mcp_tool_description_holds_what_it_held_before(served):
    _, tools = served
    descriptions = {tool.name: tool.description for tool in tools}
    leaks = [
        (name, phrase)
        for name, description in descriptions.items()
        for phrase in (r"\bres\.", r"\bdf\.", "columns=None", "table=[\"']")
        if re.search(phrase, description)
    ]

    assert leaks == []


def test_python_callers_keep_the_text_that_mcp_callers_read_in_json_words(served):
    _, tools = served
    schema = next(t for t in tools if t.name == "summarize_numeric").input_schema
    over_mcp = schema["properties"]["columns"]["description"]
    in_python = ae.describe_analysis("summarize_numeric").to_dict()["parameters_schema"]
    python = in_python["properties"]["columns"]["description"]

    # The same sentence, with None for the default in Python and null over MCP.
    assert "None, the default, selects every numeric column" in python
    assert over_mcp == python.replace("None,", "null,")
    assert "null, the default, selects every numeric column" in over_mcp
    # The docstring a Python caller reads says nothing a Python caller can't use.
    assert "`columns` maps each selected column name to its entry" in " ".join(
        (ae.summarize_numeric.__doc__ or "").split()
    )


def test_the_mcp_instructions_map_a_descriptions_record_and_table_in_one_line(served):
    instructions, _ = served

    [line] = [
        line
        for line in instructions.splitlines()
        if "to_dict" in line and "to_frame" in line
    ]
    assert "structured content" in line
    assert "table_path" in line


def test_the_guard_finds_each_kind_of_python_syntax():
    for text in (
        "see res.columns for it",
        "Pass columns=None to skip it",
        "such as df.loc[row_ids]",
        "missing values (None, NaN)",
        "got True for it",
        "call ae.guide() first",
        "such as pd.Timestamp('2024-01-01')",
        "np.nan stands for it",
        "such as task='regression'",
    ):
        assert python_in(text), text
    for text in (
        "the text 'None' is a stand-in",
        "a TOML file with mask_sensitive = true",
        'such as {"task": "regression"}',
        "null, true and false",
        "unknown column 'None' in columns",
    ):
        assert python_in(text) == [], text


def test_every_error_code_an_mcp_tool_can_raise_has_a_mistake_that_reaches_it():
    documented = set(CODE.findall(ae.errors.__doc__ or ""))
    reached = {code for _, _, _, code, _ in MISTAKES.values()}

    assert documented - set(NOT_OVER_MCP) - reached == set()
    assert reached - documented == set()


@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_the_message_of_every_mistake_holds_no_python_syntax(tmp_path, mistake):
    name, arguments, kind, _, _ = MISTAKES[mistake]

    response = anyio.run(call_mcp, name, arguments_of(tmp_path, arguments, kind))

    assert response.is_error
    assert python_in(response.structured_content["message"]) == []
