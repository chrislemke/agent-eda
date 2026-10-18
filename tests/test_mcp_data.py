"""Reading data: the file a data path names becomes the DataFrame an analysis gets.

Each test writes a small file to tmp_path, calls an MCP tool on it through the
SDK's in-memory client and compares the response with the same analysis called
in Python on the DataFrame the file was written from.
"""

import os
from dataclasses import dataclass

import pytest

pytest.importorskip(
    "mcp",
    reason="the MCP server tests need the mcp extra: pip install 'agent-eda[mcp]'",
)

import anyio
import pandas as pd
from mcp import Client
from mcp.types import TextContent

import agent_eda as ae
from agent_eda.errors import (
    DataFileError,
    DataFileNotFoundError,
    InvalidArgumentError,
    UnsupportedDataFormatError,
)
from agent_eda.mcp import create_server

# Integers, text with stand-ins that only look missing, and floats with an
# empty field: exactly what a CSV holds after the round trip.
ORDERS = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 104],
        "city": ["Oslo", "N/A", "Lyon", "Oslo"],
        "score": [1.5, None, -999.0, 4.0],
    }
)


@pytest.mark.parametrize(
    "file_name",
    [
        "orders.csv",
        "orders.tsv",
        "orders.csv.gz",
        "orders.tsv.gz",
        "orders.parquet",
        "orders.pq",
        "ORDERS.CSV",
        "Orders.Tsv.Gz",
        "orders.PQ",
    ],
)
def test_each_supported_format_reaches_the_analysis_unchanged(tmp_path, file_name):
    path = tmp_path / file_name
    write(ORDERS, path)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert not response.is_error
    assert response.structured_content == ae.describe_dataset(ORDERS).to_dict()


def test_parquet_keeps_nullable_and_pyarrow_backed_dtypes(tmp_path):
    data = pd.DataFrame(
        {
            "count": pd.array([1, None, 3], dtype="Int64"),
            "flag": pd.array([True, None, False], dtype="boolean"),
            "label": pd.array(["a", None, "c"], dtype="string"),
            "code": pd.array(["x", "y", None], dtype="string[pyarrow]"),
            "amount": pd.array([1, 2, None], dtype="int64[pyarrow]"),
            "when": pd.to_datetime(["2024-01-01", None, "2024-03-01"], utc=True),
            "plan": pd.Categorical(["basic", "pro", "basic"]),
        }
    )
    path = tmp_path / "typed.parquet"
    data.to_parquet(path)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    dtypes = [column["dtype"] for column in response.structured_content["columns"]]
    assert dtypes == [str(dtype) for dtype in data.dtypes]
    missing = [c["missing_count"] for c in response.structured_content["columns"]]
    assert missing == [1, 1, 1, 1, 1, 1, 0]


def test_only_an_empty_field_is_missing_so_stand_ins_reach_the_analysis(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text("answer,age\nNA,31\nN/A,-999\nNULL,40\n,52\nyes,\n")

    response = anyio.run(call, "detect_missing_sentinels", {"data_path": str(path)})

    found = [
        (candidate["column"], candidate["value"])
        for candidate in response.structured_content["candidates"]
    ]
    assert sorted(found, key=str) == sorted(
        [("answer", "NA"), ("answer", "N/A"), ("answer", "NULL"), ("age", -999)],
        key=str,
    )
    described = anyio.run(call, "describe_dataset", {"data_path": str(path)})
    # The empty fields, one in each column, are the only missing values.
    columns = described.structured_content["columns"]
    assert [(c["column"], c["missing_count"]) for c in columns] == [
        ("answer", 1),
        ("age", 1),
    ]


def test_row_ids_of_a_csv_are_0_based_data_row_numbers(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("order,city\nA1,Oslo\nA2,Lyon\nA3,Oslo\nA4,Oslo\n")

    response = anyio.run(
        call, "find_duplicate_rows", {"data_path": str(path), "columns": ["city"]}
    )

    groups = response.structured_content["duplicate_groups"]
    assert [group["row_ids"] for group in groups] == [[0, 2, 3]]


def test_row_ids_of_a_parquet_file_follow_the_index_stored_in_it(tmp_path):
    path = tmp_path / "orders.parquet"
    data = pd.DataFrame(
        {"city": ["Oslo", "Lyon", "Oslo", "Oslo"]},
        index=pd.Index(["A1", "A2", "A3", "A4"], name="order"),
    )
    data.to_parquet(path)

    response = anyio.run(
        call, "find_duplicate_rows", {"data_path": str(path), "columns": ["city"]}
    )

    groups = response.structured_content["duplicate_groups"]
    assert [group["row_ids"] for group in groups] == [["A1", "A3", "A4"]]


def test_id_column_replaces_the_row_numbers_of_a_csv(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("order,city\nA1,Oslo\nA2,Lyon\nA3,Oslo\nA4,Oslo\n")

    response = anyio.run(
        call,
        "find_duplicate_rows",
        {"data_path": str(path), "columns": ["city"], "id_column": "order"},
    )

    groups = response.structured_content["duplicate_groups"]
    assert [group["row_ids"] for group in groups] == [["A1", "A3", "A4"]]


def test_a_relative_path_starts_at_the_server_working_directory(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    write(ORDERS, tmp_path / "data" / "orders.csv")
    monkeypatch.chdir(tmp_path)

    response = anyio.run(call, "describe_dataset", {"data_path": "data/orders.csv"})

    assert not response.is_error
    assert response.structured_content == ae.describe_dataset(ORDERS).to_dict()


def test_data_file_errors_are_invalid_arguments_and_a_missing_file_is_not_found():
    # Python callers of the reader's errors can catch them by these classes.
    assert issubclass(UnsupportedDataFormatError, InvalidArgumentError)
    assert issubclass(DataFileError, InvalidArgumentError)
    assert issubclass(DataFileNotFoundError, DataFileError)
    assert issubclass(DataFileNotFoundError, FileNotFoundError)


@pytest.mark.parametrize("file_name", ["orders.xlsx", "orders.json.gz", "orders"])
def test_an_unknown_extension_lists_the_supported_ones(tmp_path, file_name):
    path = tmp_path / file_name
    path.write_text("order_id\n1\n")

    error = anyio.run(data_error, {"data_path": str(path)})

    assert error.code == "unsupported_data_format"
    assert ".csv, .tsv, .csv.gz, .tsv.gz, .parquet and .pq" in error.message
    assert error.message.startswith(f"data_path {str(path)!r} has ")


def test_a_missing_file_names_the_path_tried_and_the_closest_files(tmp_path):
    for name in ["orders.csv", "orders_2024.csv", "notes.txt"]:
        (tmp_path / name).write_text("order_id\n1\n")
    path = tmp_path / "order.csv"

    error = anyio.run(data_error, {"data_path": str(path)})

    assert error.code == "data_file_not_found"
    assert error.message.startswith(
        f"There is no file at {path.resolve()}. Its directory exists. "
        "Did you mean 'orders.csv' or 'orders_2024.csv'? Pass the path of an "
        "existing file, absolute or relative to the MCP server's working directory"
    )


def test_a_missing_file_says_when_its_directory_is_missing_too(tmp_path):
    path = tmp_path / "exports" / "orders.csv"

    error = anyio.run(data_error, {"data_path": str(path)})

    assert error.code == "data_file_not_found"
    assert error.message.startswith(
        f"There is no file at {path.resolve()}. Its directory, "
        f"{path.parent.resolve()}, doesn't exist either."
    )


def test_a_relative_path_that_misses_shows_the_absolute_path_tried(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    error = anyio.run(data_error, {"data_path": "orders.csv"})

    assert error.code == "data_file_not_found"
    assert f"There is no file at {tmp_path.resolve() / 'orders.csv'}." in error.message
    assert f"working directory, {tmp_path.resolve()}." in error.message


@pytest.mark.parametrize(
    "data_path",
    ["http://agent-eda.invalid/orders.csv", "s3://agent-eda-invalid/orders.parquet"],
)
def test_a_path_that_looks_like_a_url_is_a_local_path(tmp_path, monkeypatch, data_path):
    # A network read would fail with a URL or connection error instead.
    monkeypatch.chdir(tmp_path)

    error = anyio.run(data_error, {"data_path": data_path})

    assert error.code == "data_file_not_found"
    assert f"There is no file at {tmp_path.resolve()}/" in error.message


@pytest.mark.parametrize(
    ("file_name", "content", "quoted"),
    [
        ("latin1.csv", "city\nZürich\n".encode("latin-1"), "can't decode byte"),
        ("empty.csv", b"", "No columns to parse from file"),
        ("ragged.tsv", b"a\tb\n1\t2\n3\t4\t5\t6\n", "Expected 2 fields in line 3"),
        ("plain.csv.gz", b"a,b\n1,2\n", "Not a gzipped file"),
        ("text.parquet", b"a,b\n1,2\n", "Parquet"),
    ],
)
def test_a_file_the_parser_cannot_read_quotes_the_parser(
    tmp_path, file_name, content, quoted
):
    path = tmp_path / file_name
    path.write_bytes(content)

    error = anyio.run(data_error, {"data_path": str(path)})

    assert error.code == "unreadable_data_file"
    assert error.message.startswith(f"Can't read {path.resolve()} as ")
    assert quoted in error.message


def test_a_second_call_on_the_same_file_does_not_read_it_again(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n1\n")

    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            first = await distinct_count(client, path)
            # New values, but the same modification time and size.
            rewrite_unnoticed(path, "x\n1\n2\n")
            return first, await distinct_count(client, path)

    assert anyio.run(scenario) == (1, 1)


def test_a_file_whose_modification_time_changed_is_read_again(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n1\n")

    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            first = await distinct_count(client, path)
            rewrite_unnoticed(path, "x\n1\n2\n")
            status = path.stat()
            os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
            return first, await distinct_count(client, path)

    assert anyio.run(scenario) == (1, 2)


def test_a_file_whose_size_changed_is_read_again(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n1\n")

    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            first = await distinct_count(client, path)
            status = path.stat()
            path.write_text("x\n1\n2\n3\n")
            os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
            return first, await distinct_count(client, path)

    assert anyio.run(scenario) == (1, 3)


def test_a_third_file_evicts_the_least_recently_used_one(tmp_path):
    paths = {name: tmp_path / f"{name}.csv" for name in "abc"}
    for path in paths.values():
        path.write_text("x\n1\n1\n")

    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            for name in "abac":
                await distinct_count(client, paths[name])
            # a was used after b, so reading c evicted b and kept a.
            rewrite_unnoticed(paths["a"], "x\n1\n2\n")
            rewrite_unnoticed(paths["b"], "x\n1\n2\n")
            return [await distinct_count(client, paths[name]) for name in "ab"]

    assert anyio.run(scenario) == [1, 2]


async def distinct_count(client, path):
    """The distinct count of column x, as describe_dataset reports it."""
    response = await client.call_tool("describe_dataset", {"data_path": str(path)})
    (column,) = response.structured_content["columns"]
    return column["distinct_count"]


def rewrite_unnoticed(path, text):
    """Write text, of the file's size, and keep the modification time."""
    status = path.stat()
    assert len(text.encode()) == status.st_size
    path.write_text(text)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))


def write(data, path):
    """Write data to path in the format its extension names."""
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        data.to_parquet(path)
        return
    sep = "\t" if ".tsv" in name else ","
    compression = "gzip" if name.endswith(".gz") else None
    data.to_csv(path, sep=sep, index=False, compression=compression)


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)


@dataclass(frozen=True)
class ToolError:
    """The code and message of an MCP tool error."""

    code: str
    message: str


async def data_error(arguments):
    """The tool error a describe_dataset call with these arguments returns."""
    async with Client(create_server(), mode="legacy") as client:
        response = await client.call_tool("describe_dataset", arguments)
    assert response.is_error
    error = ToolError(**response.structured_content)
    assert response.content == [TextContent(text=error.message)]
    return error
