"""The message of every argument mistake an MCP caller can make.

Each row of MISTAKES makes one mistake through an MCP tool and says the code of
the tool error and the fragments of its message. A message says what is wrong,
what is allowed and a fix an MCP caller can make: JSON arguments and files, not
Python. None shows a type annotation or a pydantic phrase.
tests/test_argument_messages.py runs the mistakes a Python caller can make.
"""

import os
import re
import sys

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
from mcp import Client, MCPError
from mcp.types import INVALID_PARAMS

from agent_eda.mcp import create_server

CSV = "order,city,score\nA1,Oslo,1.5\nA2,Lyon,2.5\nA3,Oslo,\nA4,Oslo,4.0\n"

# What only a Python caller can write or run.
PYTHON_FIX = re.compile(
    r"\bae\.|\bhelp\(|\b\w+=(None|True|False|'|\{|\[)|\bdf\b|\bpd\.|\.astype\(|"
    r"\bres\.|to_frame\(|reset_index\(|sorted\("
)

# What no message may show: a typing annotation, or what pydantic says.
LEAKS = re.compile(
    r"Input should|valid dictionary|valid number|valid string|Sequence\[|Mapping\[|"
    r"Literal\[|\bUnion\[|pydantic|Field required|Extra inputs"
)

# The data a mistake reads, by kind. The default is a CSV file.
INDEXED = pd.DataFrame(
    {"order": ["A1", "A2", "A3"], "score": [1.5, 2.5, 4.0]}
).set_index("order")
NO_COLUMNS = pd.DataFrame(index=range(3))


# Each mistake: the MCP tool, its arguments besides data_path, the kind of data
# it reads, the code of the tool error and fragments the message holds.
MISTAKES = {
    "project-for-an-analysis-that-shows-no-raw-values": (
        "summarize_numeric",
        {"project_path": "project.toml"},
        "csv",
        "invalid_call",
        [
            "summarize_numeric takes no project_path",
            "shows no raw values",
            "Leave project_path out",
        ],
    ),
    "mask-sensitive-for-an-analysis-that-shows-no-raw-values": (
        "find_candidate_keys",
        {"mask_sensitive": True},
        "csv",
        "invalid_call",
        [
            "find_candidate_keys takes no mask_sensitive",
            "shows no raw values",
            "Leave mask_sensitive out",
        ],
    ),
    "argument-misspelled-in-case-only": (
        "summarize_numeric",
        {"COLUMNS": ["score"]},
        "csv",
        "invalid_call",
        ["has no argument 'COLUMNS'", "Did you mean 'columns'?", "Allowed: data_path"],
    ),
    "argument-of-the-search-tool-misspelled": (
        "search_analyses",
        {"quey": "nulls"},
        None,
        "invalid_call",
        ["search_analyses has no argument 'quey'", "Did you mean 'query'?"],
    ),
    "argument-for-the-tool-that-takes-none": (
        "list_analyses",
        {"limit": 3},
        None,
        "invalid_call",
        ["list_analyses takes no arguments, but got 'limit'"],
    ),
    "columns-holding-null": (
        "summarize_numeric",
        {"columns": [None]},
        "csv",
        "column_not_found",
        [
            "unknown column 'None' in columns",
            "leave columns out rather than a list holding null",
        ],
    ),
    "id-column-that-names-the-index": (
        "find_duplicate_rows",
        {"id_column": "order"},
        "indexed",
        "column_not_found",
        [
            "id_column 'order', which names the index, not a column",
            "already the row IDs",
            "leave id_column out",
            "write it as a column in the data file",
        ],
    ),
    "sentinel-values-of-the-id-column": (
        "detect_missing_sentinels",
        {"sentinel_values": {"order": ["n/a"]}, "id_column": "order"},
        "csv",
        "invalid_argument",
        [
            "sentinel values for 'order'",
            "leaving the columns argument out leaves out the id_column",
            "Name it in columns",
        ],
    ),
    "task-that-is-a-prefix": (
        "profile_target",
        {"target_column": "score", "task": "binary"},
        "csv",
        "invalid_argument",
        ["got 'binary' for task", "Did you mean 'binary_classification'?"],
    ),
    "method-misspelled": (
        "compute_association_matrix",
        {"numeric_method": "spearmen"},
        "csv",
        "invalid_argument",
        ["got 'spearmen' for numeric_method", "Did you mean 'spearman'?"],
    ),
    "schema-that-is-a-list": (
        "validate_column_types",
        {"schema": ["score"]},
        "csv",
        "invalid_argument",
        [
            "a list of 1 item for schema",
            "a mapping from column names",
            "one of 'boolean', 'integer'",
        ],
    ),
    "quantiles-that-is-a-string": (
        "summarize_numeric",
        {"quantiles": "0.5"},
        "csv",
        "invalid_argument",
        ["got '0.5' for quantiles", "quantiles must be a list of numbers"],
    ),
    "data-without-columns": (
        "summarize_numeric",
        {"columns": ["score"]},
        "no-columns",
        "column_not_found",
        ["unknown column 'score' in columns", "The data has no columns."],
    ),
    # Paths.
    "data-path-that-is-a-directory": (
        "summarize_numeric",
        {"data_path": "{folder}"},
        "csv",
        "data_file_not_found",
        ["data_path leads to", "which is a directory, not a file"],
    ),
    "table-path-that-is-a-directory": (
        "summarize_numeric",
        {"table_path": "{folder}"},
        "csv",
        "invalid_argument",
        ["table_path leads to", "which is a directory"],
    ),
    "table-path-where-a-file-exists": (
        "summarize_numeric",
        {"table_path": "{data}"},
        "csv",
        "table_file_exists",
        ["table_path leads to", "where a file already exists"],
    ),
    "table-path-in-a-missing-directory": (
        "summarize_numeric",
        {"table_path": "{directory}/missing/table.csv"},
        "csv",
        "invalid_argument",
        ["table_path leads to", "its directory", "doesn't exist"],
    ),
    "table-path-of-another-format": (
        "summarize_numeric",
        {"table_path": "{directory}/table.xlsx"},
        "csv",
        "invalid_argument",
        ["table_path", "'.xlsx'", ".csv"],
    ),
    "table-path-that-is-a-number": (
        "summarize_numeric",
        {"table_path": 3},
        "csv",
        "invalid_argument",
        ["table_path", "must be a string"],
    ),
    "data-path-that-is-a-number": (
        "summarize_numeric",
        {"data_path": 3},
        None,
        "invalid_argument",
        ["got 3 for data_path", "data_path must be a string"],
    ),
}


@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_an_argument_mistake_gets_its_code_and_an_mcp_fix(tmp_path, mistake):
    name, arguments, kind, code, fragments = MISTAKES[mistake]
    data = data_file(tmp_path, kind)
    (tmp_path / "folder.csv").mkdir()
    arguments = {key: fill(value, tmp_path, data) for key, value in arguments.items()}
    if kind is not None and "data_path" not in arguments:
        arguments["data_path"] = str(data)

    response = anyio.run(call, name, arguments)

    assert response.is_error
    assert response.structured_content["code"] == code
    message = response.structured_content["message"]
    assert [block.text for block in response.content] == [message]
    for fragment in fragments:
        assert fragment in message
    assert PYTHON_FIX.findall(message) == []
    assert LEAKS.findall(message) == []


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a read-only directory only stops a user who isn't root, on POSIX",
)
def test_a_table_path_in_a_directory_that_cannot_be_written_is_a_coded_error(tmp_path):
    data = data_file(tmp_path, "csv")
    directory = tmp_path / "read_only"
    directory.mkdir()
    directory.chmod(0o555)
    try:
        response = anyio.run(
            call,
            "summarize_numeric",
            {"data_path": str(data), "table_path": str(directory / "table.csv")},
        )
    finally:
        directory.chmod(0o755)

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert "table_path" in response.structured_content["message"]
    assert not (directory / "table.csv").exists()


def test_an_unknown_mcp_tool_is_a_json_rpc_invalid_params_error():
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            with pytest.raises(MCPError) as raised:
                await client.call_tool("summarise_numeric", {})
            return raised.value

    error = anyio.run(scenario)

    assert error.error.code == INVALID_PARAMS
    assert "There is no MCP tool named 'summarise_numeric'" in error.error.message


def fill(value, tmp_path, data):
    """A path argument written with {directory}, {folder} (a directory named like a
    CSV file) or {data} for the test's own."""
    if isinstance(value, str):
        return value.format(
            directory=tmp_path, folder=tmp_path / "folder.csv", data=data
        )
    return value


def data_file(tmp_path, kind):
    """The file a mistake reads: CSV, or a Parquet file that keeps an index or
    holds no column, which a CSV file can't."""
    if kind == "csv":
        path = tmp_path / "shop.csv"
        path.write_text(CSV, encoding="utf-8")
    elif kind == "indexed":
        path = tmp_path / "shop.parquet"
        INDEXED.to_parquet(path)
    elif kind == "no-columns":
        path = tmp_path / "empty.parquet"
        NO_COLUMNS.to_parquet(path)
    else:
        path = tmp_path / "unused"
    return path


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
