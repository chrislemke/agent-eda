"""Reading CSV files over MCP: every analysis sees the values the file holds.

Each test writes a file to tmp_path, calls an MCP tool on it through the SDK's
in-memory client and compares the response with the same analysis called in
Python on a DataFrame built by hand, or with an independent read of the file.
"""

import gzip
import warnings

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
import numpy as np
import pandas as pd
from mcp import Client
from mcp.types import TextContent

import agent_eda as ae
from agent_eda.mcp import create_server


def test_a_column_is_typed_once_however_long_the_file_is(tmp_path):
    # pandas types a long file in chunks of about 2^20 / columns rows, so a text
    # value after the first chunk would leave the first chunk's numbers numbers.
    rows = 600_000
    codes = np.arange(rows) % 10
    labels = codes.astype(str).astype(object)
    labels[550_000] = "N/A"
    path = tmp_path / "codes.csv"
    pd.DataFrame({"code": labels}).to_csv(path, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = anyio.run(call, "summarize_categorical", {"data_path": str(path)})

    assert not response.is_error
    column = response.structured_content["columns"]["code"]
    assert column["cardinality"] == 11
    assert column["value_types"] == ["str"]
    assert not column["mixed_value_types"]


def test_a_wide_file_with_a_late_text_value_reads_that_column_as_text(tmp_path):
    rng = np.random.default_rng(0)
    rows, width = 20_000, 50
    data = pd.DataFrame(
        rng.integers(0, 100, size=(rows, width)),
        columns=[f"c{i}" for i in range(width)],
    )
    data["c7"] = data["c7"].astype(str)
    data.loc[18_000, "c7"] = "x7"
    path = tmp_path / "wide.csv"
    data.to_csv(path, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = anyio.run(call, "summarize_categorical", {"data_path": str(path)})

    assert not response.is_error
    column = response.structured_content["columns"]["c7"]
    assert column["cardinality"] == data["c7"].nunique()
    assert not column["mixed_value_types"]
    assert column["value_types"] == ["str"]


def test_a_trailing_delimiter_adds_no_column_and_no_index(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,city,amount\nA1,Oslo,10,\nA2,Lyon,20,\nA3,Oslo,10,\nA1,Oslo,10,\n",
        encoding="utf-8",
    )
    expected = pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A1"],
            "city": ["Oslo", "Lyon", "Oslo", "Oslo"],
            "amount": [10, 20, 10, 10],
        }
    )

    described = anyio.run(call, "describe_dataset", {"data_path": str(path)})
    duplicates = anyio.run(call, "find_duplicate_rows", {"data_path": str(path)})

    assert described.structured_content == ae.describe_dataset(expected).to_dict()
    assert duplicates.structured_content == ae.find_duplicate_rows(expected).to_dict()
    groups = duplicates.structured_content["duplicate_groups"]
    assert [group["row_ids"] for group in groups] == [[0, 3]]


def test_a_trailing_delimiter_on_some_rows_only_is_accepted_too(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("a,b\n1,2\n3,4,\n5,6\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    expected = pd.DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    assert response.structured_content == ae.describe_dataset(expected).to_dict()


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("a,b\n1,2\n3,4,5\n", 3),
        ("a,b\n1,2,9\n3,4\n", 2),
        ("a,b\n1,2,\n3,4\n5,6,7,\n", 4),
        ("a,b\n1,2\n\n3,4\n5,6,x\n", 5),
    ],
)
def test_a_row_with_a_non_empty_extra_field_names_its_line(tmp_path, text, line):
    path = tmp_path / "orders.csv"
    path.write_text(text, encoding="utf-8")

    error = anyio.run(data_error, str(path))

    assert error.code == "unreadable_data_file"
    assert f"line {line}" in error.message
    assert "the header's 2 names" in error.message


def test_an_empty_line_of_a_one_column_file_is_a_missing_value(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("score\n1\n\n3\n\n5\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content["row_count"] == 5
    column = response.structured_content["columns"][0]
    assert column["missing_count"] == 2
    assert response.structured_content == (
        ae.describe_dataset(pd.DataFrame({"score": [1, None, 3, None, 5]})).to_dict()
    )


def test_a_file_with_several_columns_skips_its_blank_lines(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("id,score\n1,10\n\n2,20\n\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content["row_count"] == 2


def test_codes_with_leading_zeros_stay_text_so_each_is_a_distinct_value(tmp_path):
    path = tmp_path / "customers.csv"
    path.write_text(
        "zip,customer_id,visits\n01234,007,1\n1234,7,2\n02139,0042,3\n90210,42,4\n",
        encoding="utf-8",
    )
    expected = pd.DataFrame(
        {
            "zip": ["01234", "1234", "02139", "90210"],
            "customer_id": ["007", "7", "0042", "42"],
            "visits": [1, 2, 3, 4],
        }
    )

    keys = anyio.run(
        call,
        "find_candidate_keys",
        {"data_path": str(path), "columns": ["customer_id"]},
    )
    levels = anyio.run(
        call, "summarize_categorical", {"data_path": str(path), "columns": ["zip"]}
    )

    assert keys.structured_content == (
        ae.find_candidate_keys(expected, columns=["customer_id"]).to_dict()
    )
    assert [k["columns"] for k in keys.structured_content["keys"]] == [["customer_id"]]
    assert levels.structured_content == (
        ae.summarize_categorical(expected, columns=["zip"]).to_dict()
    )
    labels = [
        level["label"]
        for level in levels.structured_content["columns"]["zip"]["levels"]
    ]
    assert sorted(labels) == ["01234", "02139", "1234", "90210"]


def test_a_leading_zero_after_the_first_chunk_still_makes_the_column_text(tmp_path):
    rows = 300_000
    values = (np.arange(rows) % 1000 + 1).astype(str).astype(object)
    values[-1] = "007"
    path = tmp_path / "late.csv"
    pd.DataFrame({"code": values, "n": np.arange(rows)}).to_csv(path, index=False)

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(path), "columns": ["code"]}
    )

    column = response.structured_content["columns"]["code"]
    assert column["value_types"] == ["str"]
    assert column["cardinality"] == 1001


def test_numbers_without_a_leading_zero_code_stay_numbers(tmp_path):
    path = tmp_path / "numbers.csv"
    path.write_text(
        "n,flag,ratio,label\n-5,true,0.5,a\n0,False,-0.25,b\n0.5,TRUE,1e3,c\n10,,inf,d\n",
        encoding="utf-8",
    )

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert (
        response.structured_content
        == ae.describe_dataset(
            pd.DataFrame(
                {
                    "n": [-5.0, 0.0, 0.5, 10.0],
                    "flag": pd.Series([True, False, True, np.nan], dtype=object),
                    "ratio": [0.5, -0.25, 1000.0, np.inf],
                    "label": ["a", "b", "c", "d"],
                }
            )
        ).to_dict()
    )


def test_every_other_column_reads_as_pandas_reads_it(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(
        "id,zip,when,amount,note\n"
        "1,01234,2024-01-05,1.5,N/A\n"
        "2,90210,2024-02-06,,NA\n"
        "3,02139,2024-03-07,-3,\n",
        encoding="utf-8",
    )
    reference = pd.read_csv(
        path, keep_default_na=False, na_values=[""], dtype={"zip": str}
    )

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content == ae.describe_dataset(reference).to_dict()


def test_repeated_header_names_stay_repeated_and_are_reported(tmp_path):
    path = tmp_path / "repeated.csv"
    path.write_text("id,score,score,id\n1,2,3,4\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert not response.is_error
    assert [c["column"] for c in response.structured_content["columns"]] == [
        "id",
        "score",
        "score",
        "id",
    ]
    codes = [f["code"] for f in response.structured_content["findings"]]
    assert "duplicated_column_labels" in codes


def test_another_analysis_asks_to_rename_the_repeated_columns_in_the_file(tmp_path):
    path = tmp_path / "repeated.csv"
    path.write_text("id,score,score,id\n1,2,3,4\n", encoding="utf-8")

    response = anyio.run(call, "summarize_numeric", {"data_path": str(path)})

    assert response.is_error
    message = response.content[0].text
    assert "'id' and 'score' occur more than once" in message
    assert "Rename the repeated columns in the file" in message
    assert "df." not in message


def test_a_file_read_under_two_names_is_read_in_each_names_format(tmp_path):
    tsv = tmp_path / "both.tsv"
    tsv.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")
    csv = tmp_path / "both.csv"
    csv.symlink_to(tsv)

    async def both():
        async with Client(create_server(), mode="legacy") as client:
            first = await client.call_tool("describe_dataset", {"data_path": str(tsv)})
            second = await client.call_tool("describe_dataset", {"data_path": str(csv)})
            return first, second

    first, second = anyio.run(both)

    assert [c["column"] for c in first.structured_content["columns"]] == ["a", "b"]
    assert [c["column"] for c in second.structured_content["columns"]] == ["a\tb"]


def test_the_data_path_description_states_the_reading_rules():
    async def description():
        async with Client(create_server(), mode="legacy") as client:
            tools = await client.list_tools()
        tool = next(t for t in tools.tools if t.name == "describe_dataset")
        return tool.input_schema["properties"]["data_path"]["description"]

    text = anyio.run(description).lower()

    for rule in ("leading zero", "true", "inf", "blank line"):
        assert rule in text
    assert "Parquet keeps its dtypes" not in text


@pytest.mark.parametrize("file_name", ["orders.tsv", "orders.tsv.gz", "orders.csv.gz"])
def test_every_csv_format_reads_trailing_delimiters_and_codes_alike(
    tmp_path, file_name
):
    sep = "\t" if ".tsv" in file_name else ","
    rows = [["code", "n"], ["007", "1", ""], ["8", "2", ""], ["9", "3", ""]]
    text = "\n".join(sep.join(row) for row in rows) + "\n"
    path = tmp_path / file_name
    if file_name.endswith(".gz"):
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    expected = pd.DataFrame({"code": ["007", "8", "9"], "n": [1, 2, 3]})

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content == ae.describe_dataset(expected).to_dict()


@pytest.mark.parametrize(
    ("field", "kept"),
    [
        ("007", "007"),
        ('"007"', "007"),
        ("-007", "-007"),
        ("+007", "+007"),
        ("00", "00"),
    ],
)
def test_each_way_to_write_a_leading_zero_code_keeps_the_column_text(
    tmp_path, field, kept
):
    path = tmp_path / "codes.csv"
    path.write_text(f"code,n\n5,1\n{field},2\n12,3\n", encoding="utf-8")

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(path), "columns": ["code"]}
    )

    levels = response.structured_content["columns"]["code"]["levels"]
    assert sorted(level["label"] for level in levels) == sorted(["5", kept, "12"])


@pytest.mark.parametrize(
    "field", ["0", "0.5", "-0.5", "10", "100", "2024", "00.5", "0123.5", "01e3"]
)
def test_a_number_that_only_contains_a_zero_stays_a_number(tmp_path, field):
    path = tmp_path / "numbers.csv"
    path.write_text(f"n\n5\n{field}\n12\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content["columns"][0]["dtype"] in ("int64", "float64")


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)


class ToolError:
    """The code and message of an MCP tool error."""

    def __init__(self, code, message):
        self.code = code
        self.message = message


async def _data_error(path):
    async with Client(create_server(), mode="legacy") as client:
        response = await client.call_tool("describe_dataset", {"data_path": path})
    assert response.is_error
    error = ToolError(**response.structured_content)
    assert response.content == [TextContent(text=error.message)]
    return error


def data_error(path):
    return _data_error(path)
