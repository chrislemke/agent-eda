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
    reason=(
        "the MCP server tests need the mcp extra: run `uv sync` (uv 0.13 or later) or "
        "uv run --extra mcp pytest in this repository, or "
        "pip install 'agent-eda[mcp]' for an installed package"
    ),
)

import anyio
import pandas as pd
from mcp import Client, StdioServerParameters
from stdio import stdio_environment

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


def test_a_call_returns_the_printed_result_in_mcp_wording_and_the_record(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV, encoding="utf-8")
    # The server reads a CSV counting only an empty field as missing.
    expected = ae.describe_dataset(
        pd.DataFrame(
            {
                "city": ["Oslo", "Lyon", "Oslo"],
                "score": [1.5, None, 2.5],
                "note": ["N/A", "ok", float("nan")],
            }
        )
    )

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert not response.is_error
    assert response.structured_content == expected.to_dict()
    (text,) = [block.text for block in response.content]
    # The Python text, whose dtypes and memory differ between pandas 2 and 3,
    # with a footer that names what an MCP caller can reach.
    assert text.splitlines()[:-1] == str(expected).splitlines()[:-1]
    assert text.splitlines()[-1] == (
        "More: the full table, which table_path writes to a file, has all 3 "
        "columns; the structured content has the rest."
    )


def test_the_printed_result_names_what_an_mcp_caller_can_reach(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV, encoding="utf-8")

    response = anyio.run(call, "summarize_numeric", {"data_path": str(path)})

    lines = response.content[0].text.splitlines()
    assert lines[3] == (
        "The full table (table_path) also has the quantiles, "
        "trimmed_mean, mad, mad_scaled, skewness and excess_kurtosis."
    )
    assert lines[-1] == (
        "More: the full table, which table_path writes to a file, has the 1 "
        "column; the structured content has the rest."
    )


def test_the_printed_result_names_record_fields_without_python(tmp_path):
    path = tmp_path / "shop.parquet"
    # Parquet keeps the dates' dtype, which the association matrix skips.
    pd.DataFrame(
        {
            "plan": ["pro", "team", "pro", "team"],
            "spend": [10, 20, 30, 25],
            "joined": pd.to_datetime(["2024-01-01"] * 4),
        }
    ).to_parquet(path)

    response = anyio.run(call, "compute_association_matrix", {"data_path": str(path)})

    text = response.content[0].text
    assert "Left out 1 column, see left_out_columns: 'joined' (" in text
    # Over MCP, table_path writes only the default table, so the footer names no
    # other one.
    assert text.splitlines()[-1] == (
        "More: the full table, which table_path writes to a file, has the 1 pair; "
        "the structured content has the rest."
    )


def test_the_masking_line_says_how_to_mask_in_json(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(
        "customer,plan\nann@example.org,pro\nbo@example.org,team\n", encoding="utf-8"
    )
    project_path = tmp_path / "project.toml"
    project_path.write_text(
        '[data_policy]\nsensitive_columns = ["customer"]\nmask_sensitive = false\n',
        encoding="utf-8",
    )

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(path), "project_path": str(project_path)},
    )

    text = " ".join(response.content[0].text.split())
    assert (
        "Not masked (data_policy.mask_sensitive = false): the values of sensitive "
        "column 'customer' show as stored. Pass mask_sensitive: true to mask them."
    ) in text


def test_a_stand_in_such_as_n_a_reaches_the_analysis_as_a_value(tmp_path):
    path = tmp_path / "shop.csv"
    path.write_text(CSV, encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    columns = {c["column"]: c for c in response.structured_content["columns"]}
    # "N/A" counts as a value, the empty field as missing.
    assert columns["note"]["non_missing_count"] == 2
    assert columns["note"]["missing_count"] == 1
    assert columns["score"]["missing_count"] == 1
    assert columns["note"]["dtype"] in ("object", "str")


def test_agent_eda_mcp_answers_a_call_over_stdio(tmp_path):
    (tmp_path / "shop.csv").write_text(CSV, encoding="utf-8")

    # A relative path, which the server resolves against its working directory.
    response = anyio.run(call_over_stdio, tmp_path, "1", "shop.csv")

    assert not response.is_error
    assert response.structured_content["status"] == "completed"
    assert response.content[0].text.startswith("describe_dataset: completed on 3 rows")


def test_two_mcp_server_processes_with_different_hash_seeds_return_the_same_record(
    tmp_path,
):
    (tmp_path / "shop.csv").write_text(CSV, encoding="utf-8")

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
        env=stdio_environment(PYTHONHASHSEED=hash_seed),
        cwd=str(cwd),
    )
    async with Client(server, mode="legacy") as client:
        return await client.call_tool("describe_dataset", {"data_path": data_path})


# send_ping is the one way the client checks that the server still answers, and
# legacy mode, which these tests use, supports it.
@pytest.mark.filterwarnings("ignore:ping is removed")
def test_a_huge_max_key_size_returns_and_the_mcp_server_still_answers_a_ping(tmp_path):
    (tmp_path / "keys.csv").write_text("a,b\n1,1\n2,1\n3,2\n", encoding="utf-8")

    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            with anyio.fail_after(10):
                response = await client.call_tool(
                    "find_candidate_keys",
                    {"data_path": str(tmp_path / "keys.csv"), "max_key_size": 10**20},
                )
                await client.send_ping()  # ty: ignore[deprecated]  # legacy mode, which this test runs, answers pings
            return response

    response = anyio.run(scenario)

    assert not response.is_error
    assert (response.structured_content or {})["status"] == "completed"
