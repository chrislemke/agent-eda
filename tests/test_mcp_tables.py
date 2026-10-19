"""The full table over MCP, tested as an MCP client meets it.

The structured content caps its lists. An MCP caller who needs every row passes
table_path, and the server writes the table a Python caller gets from to_frame
to that file, as CSV or Parquet, and says where. It never overwrites a file.
"""

import pytest

pytest.importorskip(
    "mcp",
    reason=(
        "the MCP server tests need the mcp extra: run uv sync or "
        "uv run --extra mcp pytest in this repository, or "
        "pip install 'agent-eda[mcp]' for an installed package"
    ),
)

import json
import warnings

import anyio
import pandas as pd
from mcp import Client
from recipes import recipe_for

import agent_eda as ae
from agent_eda.errors import AgentEDAError, TableFileExistsError
from agent_eda.mcp import create_server
from agent_eda.results import SummarizeNumericResult

ANALYSES = [info.name for info in ae.list_analyses().analyses]

CSV = "city,score,visits\nOslo,1.5,3\nLyon,,7\nOslo,4.0,2\nRome,2.5,5\n"

# Mixed dtypes, with the stand-ins "N/A" and -999 for missing values.
MIXED_CSV = (
    "order_id,city,score,joined,member,plan\n"
    "1,Oslo,1.5,2024-01-03,true,basic\n"
    "2,Lyon,,2024-02-11,false,pro\n"
    "3,Oslo,2.5,2024-03-07,true,basic\n"
    "4,N/A,-999,2024-04-21,false,pro\n"
    "5,Lyon,4.0,,true,basic\n"
)


def test_a_csv_table_path_gets_every_row_of_the_full_table(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    table_path = tmp_path / "summary.csv"
    expected = ae.summarize_numeric(read(data_path)).to_frame()

    anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(data_path), "table_path": str(table_path)},
    )

    written = pd.read_csv(table_path, keep_default_na=False, na_values=[""])
    assert_same_table(written, expected)


@pytest.mark.parametrize("file_name", ["summary.parquet", "summary.pq", "Summary.PQ"])
def test_a_parquet_table_path_keeps_the_full_table_and_its_dtypes(tmp_path, file_name):
    data_path = write(tmp_path / "shop.csv", CSV)
    table_path = tmp_path / file_name
    expected = ae.summarize_numeric(read(data_path)).to_frame()

    anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(data_path), "table_path": str(table_path)},
    )

    written = read_parquet(table_path)
    assert_same_table(written, expected)
    numbers = expected.select_dtypes("number").columns
    assert written[numbers].dtypes.to_dict() == expected[numbers].dtypes.to_dict()


def test_the_text_gains_one_line_with_the_absolute_path_and_nothing_else_changes(
    tmp_path, monkeypatch
):
    write(tmp_path / "shop.csv", CSV)
    monkeypatch.chdir(tmp_path)
    without = anyio.run(call, "summarize_categorical", {"data_path": "shop.csv"})

    # A relative path, resolved against the server's working directory.
    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": "shop.csv", "table_path": "levels.csv"},
    )

    assert not response.is_error
    assert [block.text for block in response.content] == [
        f"{without.content[0].text}\nFull table written to "
        f"{tmp_path.resolve() / 'levels.csv'} (3 rows)."
    ]


def test_the_record_gains_the_written_tables_path_and_row_count(tmp_path, monkeypatch):
    data_path = write(tmp_path / "shop.csv", CSV)
    expected = ae.summarize_categorical(read(data_path))
    monkeypatch.chdir(tmp_path)

    # The client validates the structured content against the output schema,
    # so the schema covers table_file too.
    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": "shop.csv", "table_path": "levels.csv"},
    )

    assert not response.is_error
    assert response.structured_content == {
        **expected.to_dict(),
        "table_file": {
            "path": str(tmp_path.resolve() / "levels.csv"),
            "row_count": 3,
        },
    }


def test_the_output_schema_allows_table_file_without_requiring_it():
    tools = anyio.run(list_tools)

    for tool in tools:
        if tool.name in ANALYSES:
            schema = tool.output_schema
            assert "table_file" in schema["properties"], tool.name
            assert "table_file" not in schema["required"], tool.name


def test_a_table_of_one_row_says_1_row(tmp_path):
    data_path = write(tmp_path / "shop.csv", "score\n1.5\n2.5\n")
    table_path = tmp_path / "summary.csv"

    response = anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(data_path), "table_path": str(table_path)},
    )

    assert response.content[0].text.endswith(
        f"\nFull table written to {table_path.resolve()} (1 row)."
    )


def test_a_null_table_path_writes_nothing(tmp_path, monkeypatch):
    data_path = write(tmp_path / "shop.csv", CSV)
    expected = ae.summarize_numeric(read(data_path))
    monkeypatch.chdir(tmp_path)

    without = anyio.run(call, "summarize_numeric", {"data_path": "shop.csv"})
    response = anyio.run(
        call, "summarize_numeric", {"data_path": "shop.csv", "table_path": None}
    )

    assert response.content == without.content
    assert response.structured_content == expected.to_dict()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shop.csv"]


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
@pytest.mark.parametrize("name", ANALYSES)
def test_every_analysis_writes_its_full_table_in_either_format(
    tmp_path, name, extension
):
    data_path = write(tmp_path / "orders.csv", MIXED_CSV)
    data = read(data_path)
    arguments = recipe_for(name).required(data)
    assert arguments is not None, f"the mixed CSV allows no call of {name}"
    expected = getattr(ae, name)(data, **arguments).to_frame()
    table_path = tmp_path / f"table{extension}"

    response = anyio.run(
        call,
        name,
        {"data_path": str(data_path), "table_path": str(table_path), **arguments},
    )

    assert not response.is_error, response.content[0].text
    written = read_back(table_path)
    assert list(written.columns) == list(expected.columns)
    assert len(written) == len(expected)


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_lists_become_json_text_and_parquet_writes_mixed_values_as_text(
    tmp_path, extension
):
    data_path = write(tmp_path / "orders.csv", MIXED_CSV)
    table_path = tmp_path / f"stand-ins{extension}"

    anyio.run(
        call,
        "detect_missing_sentinels",
        {"data_path": str(data_path), "table_path": str(table_path)},
    )

    written = read_back(table_path)
    # The text "N/A" and the number -999 share the value column, which
    # value_type tells apart.
    shown = written[["column", "value", "value_type"]].astype(str)
    assert shown.to_numpy().tolist() == [
        ["city", "N/A", "str"],
        ["score", "-999.0", "float"],
    ]
    assert written["row_ids"].tolist() == ["[3]", "[3]"]
    assert [json.loads(text) for text in written["evidence"]] == [
        ["builtin_list"],
        ["builtin_list"],
    ]


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_an_index_that_holds_more_than_row_numbers_is_written_first(
    tmp_path, extension
):
    data_path = write(tmp_path / "shop.csv", CSV)
    table_path = tmp_path / f"by-column{extension}"
    server = create_server(analyses=[summarize_by_column])

    anyio.run(
        call,
        "summarize_by_column",
        {"data_path": str(data_path), "table_path": str(table_path)},
        server,
    )

    written = read_back(table_path)
    assert list(written.columns[:2]) == ["column", "status"]
    assert written["column"].tolist() == ["score", "visits"]
    assert written["count"].tolist() == [3, 4]


def test_an_existing_file_is_refused_before_the_analysis_runs_and_left_alone(
    tmp_path,
):
    data_path = write(tmp_path / "shop.csv", CSV)
    table_path = write(tmp_path / "summary.csv", "keep me\n")
    write(tmp_path / "summary-2.csv", "taken too\n")

    error = anyio.run(table_error, data_path, str(table_path))

    assert error == {
        "code": "table_file_exists",
        "message": f"table_path leads to {table_path}, where a file already "
        "exists, and the MCP server never overwrites a file. Pass a path where no "
        f"file exists yet, such as {tmp_path / 'summary-3.csv'}, or read the file "
        "that is there.",
    }
    assert table_path.read_text() == "keep me\n"


def test_table_file_exists_error_is_a_file_exists_error():
    assert issubclass(TableFileExistsError, FileExistsError)
    assert issubclass(TableFileExistsError, AgentEDAError)
    assert TableFileExistsError.code == "table_file_exists"


@pytest.mark.parametrize(
    ("file_name", "got"),
    [
        ("summary.json", "the extension '.json'"),
        ("summary.csv.gz", "the extension '.csv.gz'"),
        ("summary.tsv", "the extension '.tsv'"),
        ("summary", "no extension"),
    ],
)
def test_an_unsupported_extension_is_refused_before_the_analysis_runs(
    tmp_path, file_name, got
):
    data_path = write(tmp_path / "shop.csv", CSV)

    error = anyio.run(table_error, data_path, file_name)

    assert error == {
        "code": "invalid_argument",
        "message": f"table_path {file_name!r} has {got}, which the MCP server "
        "can't write. It writes .csv, .parquet and .pq files, in upper or lower "
        "case: CSV for any reader, Parquet to keep the dtypes. Pass a path that "
        "ends in one of them.",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shop.csv"]


def test_a_missing_directory_is_refused_before_the_analysis_runs(tmp_path, monkeypatch):
    data_path = write(tmp_path / "shop.csv", CSV)
    monkeypatch.chdir(tmp_path)

    error = anyio.run(table_error, data_path, "tables/summary.csv")

    tables = tmp_path.resolve() / "tables"
    assert error == {
        "code": "invalid_argument",
        "message": f"table_path leads to {tables / 'summary.csv'}, but its "
        f"directory, {tables}, doesn't exist, and the MCP server doesn't create "
        "directories. Create it first, or pass a path in an existing directory, "
        "absolute or relative to the MCP server's working directory, "
        f"{tmp_path.resolve()}.",
    }
    assert not tables.exists()


def test_a_directory_is_refused_with_a_file_path_in_it(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    (tmp_path / "tables.csv").mkdir()

    error = anyio.run(table_error, data_path, str(tmp_path / "tables.csv"))

    assert error["code"] == "invalid_argument"
    assert str(tmp_path / "tables.csv" / "table.csv") in error["message"]


@pytest.mark.parametrize("table_path", [3, ["summary.csv"], True])
def test_a_table_path_that_is_not_a_string_is_an_invalid_argument(tmp_path, table_path):
    data_path = write(tmp_path / "shop.csv", CSV)

    error = anyio.run(table_error, data_path, table_path)

    assert error["code"] == "invalid_argument"
    assert error["message"].startswith(
        f"break_on_run got {table_path!r} for table_path"
    )


async def table_error(data_path, table_path):
    """The tool error a table_path gets from a stand-in analysis that fails if
    it runs, so a tool error shows the check came first."""
    server = create_server(analyses=[break_on_run])
    response = await call(
        "break_on_run",
        {"data_path": str(data_path), "table_path": table_path},
        server,
    )
    assert response.is_error
    return response.structured_content


def break_on_run(data: pd.DataFrame) -> SummarizeNumericResult:
    """Does the analysis run?

    A stand-in analysis that only this test serves. It raises if it runs, which
    the MCP server reports as a bug, not as a tool error.

    Statuses:
        completed: Never; the stand-in always raises.
    """
    raise ZeroDivisionError("the analysis ran")


class IndexedSummary(SummarizeNumericResult):
    """A summary whose table has the column names as its index."""

    def to_frame(self) -> pd.DataFrame:
        return super().to_frame().set_index("column")


def summarize_by_column(data: pd.DataFrame) -> IndexedSummary:
    """What are the numeric summaries, by column?

    A stand-in analysis that only this test serves, whose table has an index
    that holds information.

    Statuses:
        completed: Always.
    """
    result = ae.summarize_numeric(data)
    return IndexedSummary.model_construct(**dict(result))


def assert_same_table(written, expected):
    """The same columns and rows, with any missing value as None, whatever the
    dtype a file format gives a column."""
    pd.testing.assert_frame_equal(cells(written), cells(expected))


def cells(frame):
    values = frame.astype(object)
    return values.where(values.notna(), None)


def write(path, text):
    path.write_text(text)
    return path


def read(path):
    """The DataFrame as the server reads the file: only an empty field is missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def read_back(path):
    """A written table, read as CSV or Parquet by its extension."""
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return read_parquet(path)


def read_parquet(path):
    with warnings.catch_warnings():
        # pandas 2.2 deprecates the internals pyarrow 14 builds DataFrames from.
        warnings.filterwarnings(
            "ignore",
            message="(Passing a BlockManager to DataFrame|DatetimeTZBlock) is "
            "deprecated",
            category=DeprecationWarning,
        )
        return pd.read_parquet(path)


async def list_tools():
    async with Client(create_server(), mode="legacy") as client:
        return (await client.list_tools()).tools


async def call(name, arguments, server=None):
    async with Client(server or create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
