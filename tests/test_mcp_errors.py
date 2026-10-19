"""Mistakes and statuses over MCP, tested as an MCP client meets them.

A caller mistake comes back as a tool error, isError true, with the stable code
and the message a Python caller gets, its fix worded for an MCP caller
(tests/test_mcp_fixes.py), so the caller can fix the call and retry.
A run that doesn't complete is a normal result with its status and reason. Only
a bug becomes a JSON-RPC error.
"""

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
import jsonschema
import pandas as pd
from mcp import Client, MCPError
from mcp.types import INTERNAL_ERROR
from serving import serving

import agent_eda as ae
from agent_eda.errors import AgentEDAError
from agent_eda.mcp import create_server
from agent_eda.results import AnalysisResult

CSV = "city,score\nOslo,1.5\nLyon,2.5\nOslo,4.0\n"


def test_a_misspelled_column_is_a_tool_error_with_the_python_code_and_message(
    tmp_path,
):
    path = shop_csv(tmp_path)
    with pytest.raises(AgentEDAError) as raised:
        ae.summarize_numeric(read(path), columns=["scor"])

    response = anyio.run(
        call, "summarize_numeric", {"data_path": str(path), "columns": ["scor"]}
    )

    assert response.is_error
    assert [block.text for block in response.content] == [str(raised.value)]
    assert response.structured_content == {
        "code": "column_not_found",
        "message": str(raised.value),
    }


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_an_error_record_passes_the_client_in_either_protocol_era(tmp_path, mode):
    path = shop_csv(tmp_path)
    arguments = {"data_path": str(path), "columns": ["scor"]}

    async def scenario():
        async with Client(create_server(), mode=mode) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            response = await client.call_tool("summarize_numeric", arguments)
            return tools["summarize_numeric"].output_schema, response

    output_schema, response = anyio.run(scenario)

    # The error record isn't the analysis's record, and the SDK client
    # validates structured content against the output schema only when
    # isError is false, so it reaches the caller intact.
    assert not jsonschema.Draft202012Validator(output_schema).is_valid(
        response.structured_content
    )
    assert response.is_error
    assert response.structured_content["code"] == "column_not_found"


@pytest.mark.parametrize(
    "options",
    [
        {"trim_fraction": "a tenth"},  # a string for a number
        {"quantiles": ["median"]},  # a string in a list of numbers
        {"quantile_method": "lineer"},  # a value outside the Literal
        {"trim_fraction": 0.7},  # a number outside the range
    ],
    ids=["string-for-number", "string-in-list", "outside-literal", "out-of-range"],
)
def test_an_argument_against_the_schema_gets_the_python_message_with_an_mcp_fix(
    tmp_path, options
):
    path = shop_csv(tmp_path)
    with pytest.raises(AgentEDAError) as raised:
        ae.summarize_numeric(read(path), **options)
    (name,) = options

    response = anyio.run(call, "summarize_numeric", {"data_path": str(path), **options})

    assert response.is_error
    assert response.structured_content == {
        "code": "invalid_argument",
        "message": str(raised.value).replace(
            "See help(ae.summarize_numeric) for every argument.",
            "The MCP tool's input schema describes every argument.",
        ),
    }
    assert f" {name}" in response.structured_content["message"]


def test_an_unknown_argument_names_the_closest_and_every_mcp_property(tmp_path):
    path = shop_csv(tmp_path)

    response = anyio.run(
        call, "summarize_numeric", {"data_path": str(path), "colums": ["score"]}
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_call"
    assert response.content[0].text.startswith(
        "summarize_numeric has no argument 'colums'. Did you mean 'columns'? "
        "Allowed: data_path, columns, "
    )


def test_data_in_place_of_data_path_names_data_path(tmp_path):
    path = shop_csv(tmp_path)

    response = anyio.run(call, "summarize_numeric", {"data": str(path)})

    assert response.is_error
    assert response.structured_content["code"] == "invalid_call"
    assert response.content[0].text.startswith(
        "summarize_numeric has no argument 'data'. Did you mean 'data_path'? "
    )


def test_a_missing_data_path_is_named_with_the_fix():
    response = anyio.run(call, "summarize_numeric", {})

    assert response.is_error
    assert response.structured_content == {
        "code": "invalid_call",
        "message": "summarize_numeric is missing 1 required argument: data_path. "
        'Pass it in the MCP tool\'s arguments, such as {"data_path": ...}. The '
        "MCP tool's input schema describes every argument.",
    }


def test_missing_required_arguments_are_named_together(tmp_path):
    path = shop_csv(tmp_path)

    response = anyio.run(call, "profile_target", {"data_path": str(path)})

    assert response.is_error
    assert response.structured_content["code"] == "invalid_call"
    assert response.content[0].text == (
        "profile_target is missing 2 required arguments: target_column and task. "
        'Pass them in the MCP tool\'s arguments, such as {"target_column": ..., '
        "\"task\": ...}, where task is one of 'binary_classification', "
        "'multiclass_classification' or 'regression'. The MCP tool's input schema "
        "describes every argument."
    )


def test_a_data_path_that_is_not_a_string_is_an_invalid_argument():
    response = anyio.run(call, "summarize_numeric", {"data_path": 3})

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert response.content[0].text.startswith(
        "summarize_numeric got 3 for data_path, but data_path must be a string: "
    )


def test_a_run_that_does_not_complete_is_a_normal_result_with_its_status(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("city,score\n", encoding="utf-8")  # a header and no rows
    expected = ae.summarize_numeric(read(path))

    response = anyio.run(call, "summarize_numeric", {"data_path": str(path)})

    assert not response.is_error
    assert response.structured_content["status"] == "insufficient_data"
    assert response.structured_content["status_reason"]
    assert response.structured_content == expected.to_dict()
    # The heading and the reason, as Python prints them.
    lines = response.content[0].text.splitlines()
    assert lines[:2] == str(expected).splitlines()[:2]


def test_a_bug_in_an_analysis_is_a_json_rpc_internal_error(tmp_path, caplog):
    path = shop_csv(tmp_path)

    async def scenario():
        async with Client(serving(count_rows), mode="legacy") as c:
            # Caught here, since leaving the client wraps it in an
            # ExceptionGroup.
            with pytest.raises(MCPError) as raised:
                await c.call_tool("count_rows", {"data_path": str(path)})
            return raised.value

    error = anyio.run(scenario)

    assert error.code == INTERNAL_ERROR
    assert error.message == (
        "count_rows failed with an unexpected error, which is a bug in "
        "agent-eda, not in the call: ZeroDivisionError: the stand-in's bug. "
        "The MCP server's log on stderr has the traceback."
    )
    # The server logs the traceback, which the MCP client shows on stderr.
    assert "ZeroDivisionError: the stand-in's bug" in caplog.text


def count_rows(data: pd.DataFrame) -> AnalysisResult:
    """How many rows does the data have?

    A stand-in analysis that only this test serves, with a bug.

    Statuses:
        completed: Never; the stand-in always raises.
    """
    raise ZeroDivisionError("the stand-in's bug")


def shop_csv(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def read(path):
    """The DataFrame the server reads from the file, built by hand."""
    text = path.read_text(encoding="utf-8")
    if text == CSV:
        return pd.DataFrame(
            {"city": ["Oslo", "Lyon", "Oslo"], "score": [1.5, 2.5, 4.0]}
        )
    assert text == "city,score\n"  # a header and no rows: every column is object
    return pd.DataFrame(
        {"city": pd.Series(dtype=object), "score": pd.Series(dtype=object)}
    )


async def call(name, arguments):
    # The SDK client validates a result's structured content against the MCP
    # tool's output schema unless isError is true.
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
