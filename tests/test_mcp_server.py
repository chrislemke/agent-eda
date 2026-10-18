"""The MCP server, tested as an MCP client uses it.

Each test connects the SDK's client to the server and lists or calls MCP tools.
The in-memory client runs the initialize handshake over JSON-RPC streams, as a
stdio client does; the smoke tests start agent-eda-mcp as a subprocess.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "mcp",
    reason="the MCP server tests need the mcp extra: pip install 'agent-eda[mcp]'",
)

import anyio
import pandas as pd
from mcp import Client, StdioServerParameters

import agent_eda as ae
from agent_eda.mcp import create_server

# A small CSV with the values an agent meets: a stand-in "N/A", an empty field,
# numbers and text.
CSV = "city,score,note\nOslo,1.5,N/A\nLyon,,ok\nOslo,2.5,\n"


def test_the_mcp_server_is_named_agent_eda_and_reports_the_package_version():
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            return client.server_info

    info = anyio.run(scenario)

    assert info.name == "agent-eda"
    assert info.version == ae.__version__


def test_describe_dataset_is_an_mcp_tool_that_requires_a_data_path():
    tools = anyio.run(list_tools)

    (tool,) = [tool for tool in tools if tool.name == "describe_dataset"]
    assert tool.input_schema["required"] == ["data_path"]
    assert tool.input_schema["properties"]["data_path"]["type"] == "string"


def test_a_call_returns_the_printed_result_and_the_record(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV)
    # The server reads a CSV counting only an empty field as missing.
    expected = ae.describe_dataset(
        pd.read_csv(path, keep_default_na=False, na_values=[""])
    )

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert not response.is_error
    assert [block.text for block in response.content] == [str(expected)]
    assert response.structured_content == expected.to_dict()


def test_a_stand_in_such_as_n_a_reaches_the_analysis_as_a_value(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    columns = {c["column"]: c for c in response.structured_content["columns"]}
    # "N/A" counts as a value, the empty field as missing.
    assert columns["note"]["non_missing_count"] == 2
    assert columns["note"]["missing_count"] == 1
    assert columns["score"]["missing_count"] == 1
    assert columns["note"]["dtype"] in ("object", "str")


def test_agent_eda_mcp_answers_a_call_over_stdio(tmp_path):
    (tmp_path / "shop.csv").write_text(CSV)

    # A relative path, which the server resolves against its working directory.
    response = anyio.run(call_over_stdio, tmp_path, "1", "shop.csv")

    assert not response.is_error
    assert response.structured_content["status"] == "completed"
    assert response.content[0].text.startswith("describe_dataset: completed on 3 rows")


def test_two_server_processes_with_different_hash_seeds_return_the_same_record(
    tmp_path,
):
    (tmp_path / "shop.csv").write_text(CSV)

    records = [
        json.dumps(
            anyio.run(call_over_stdio, tmp_path, seed, "shop.csv").structured_content
        )
        for seed in ("1", "2")
    ]

    assert records[0] == records[1]


async def list_tools():
    async with Client(create_server(), mode="legacy") as client:
        return (await client.list_tools()).tools


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)


async def call_over_stdio(cwd, hash_seed, data_path):
    """Start agent-eda-mcp as an MCP client does and call describe_dataset."""
    server = StdioServerParameters(
        # The console script installed next to this interpreter, since the
        # client passes the subprocess only a few environment variables.
        command=str(Path(sys.executable).parent / "agent-eda-mcp"),
        env={"PYTHONHASHSEED": hash_seed},
        cwd=str(cwd),
    )
    async with Client(server, mode="legacy") as client:
        return await client.call_tool("describe_dataset", {"data_path": data_path})
