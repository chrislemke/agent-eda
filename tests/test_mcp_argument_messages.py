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
from python_syntax import python_in

from agent_eda.mcp import create_server

CSV = "order,city,score\nA1,Oslo,1.5\nA2,Lyon,2.5\nA3,Oslo,\nA4,Oslo,4.0\n"

# What only a Python caller can write or run.
PYTHON_FIX = re.compile(r"\bdf\b|\.astype\(|reset_index\(|sorted\(")

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
ZONED = pd.DataFrame({"when": pd.to_datetime(["2024-01-05", "2024-02-05"], utc=True)})
MULTI_INDEX = pd.DataFrame(
    [[1.5, 2.0]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
)

# The project files a mistake can pass, as {name} in an argument.
PROJECT_FILES = {
    "project_not_toml": "[data_policy\n",
    "project_with_a_typo": (
        '[data_policy]\nsensitive_columns = ["city"]\nmask_sensitve = true\n'
    ),
}


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
    # Fixes an MCP caller makes in JSON, never in Python.
    "sentinel-value-written-as-text-for-a-number": (
        "detect_missing_sentinels",
        {"sentinel_values": {"score": ["-999"]}},
        "csv",
        "invalid_argument",
        [
            "got \"-999\" for column 'score'",
            "Did you mean -999?",
            'such as {"sentinel_values": {"score": [-999]}}',
        ],
    ),
    "sentinel-value-of-another-kind-written-as-a-boolean": (
        "detect_missing_sentinels",
        {"sentinel_values": {"score": [True]}},
        "csv",
        "invalid_argument",
        ["got true for column 'score'", "Did you mean 1?"],
    ),
    "sentinel-value-listed-twice": (
        "detect_missing_sentinels",
        {"sentinel_values": {"score": [-999, -999.0]}},
        "csv",
        "invalid_argument",
        ['got -999 more than once in the sentinel_values list for "score"'],
    ),
    "sentinel-values-that-are-true-twice": (
        "detect_missing_sentinels",
        {"sentinel_values": {"score": [True, True]}},
        "csv",
        "invalid_argument",
        ['got true more than once in the sentinel_values list for "score"'],
    ),
    "datetime-sentinel-written-as-a-date-for-a-column-with-a-time-zone": (
        "detect_missing_sentinels",
        {"sentinel_values": {"when": ["1900-01-01"]}},
        "zoned",
        "invalid_argument",
        ['"1900-01-01T00:00:00+00:00"', "Did you mean"],
    ),
    "task-that-needs-more-expected-labels": (
        "profile_target",
        {
            "target_column": "city",
            "task": "binary_classification",
            "expected_labels": ["Oslo"],
        },
        "csv",
        "invalid_argument",
        [
            "got 1 label in expected_labels for a binary_classification task",
            'pass {"task": "multiclass_classification"}.',
        ],
    ),
    "expected-label-of-a-type-the-target-cannot-hold": (
        "profile_target",
        {
            "target_column": "city",
            "task": "multiclass_classification",
            "expected_labels": ["Oslo", True],
        },
        "csv",
        "invalid_argument",
        ["got true in expected_labels", "Declare each label as the type"],
    ),
    "project-path-that-is-empty": (
        "summarize_categorical",
        {"project_path": ""},
        "csv",
        "invalid_argument",
        [
            "project_path got an empty path",
            'project_path must name a file, such as "project.toml"',
        ],
    ),
    "project-path-that-is-a-directory": (
        "summarize_categorical",
        {"project_path": "{directory}"},
        "csv",
        "invalid_argument",
        ["project_path got the directory", "for project_path"],
    ),
    "option-written-as-a-boolean": (
        "summarize_numeric",
        {"trim_fraction": True},
        "csv",
        "invalid_argument",
        ["got true for trim_fraction"],
    ),
    "option-written-as-null": (
        "summarize_categorical",
        {"top_n": None},
        "csv",
        "invalid_argument",
        ["got null for top_n"],
    ),
    "value-inside-a-mapping-of-the-wrong-type": (
        "validate_column_types",
        {"schema": {"score": 3}},
        "csv",
        "invalid_argument",
        ["at schema/score", "got 3"],
    ),
    "search-query-that-is-null": (
        "search_analyses",
        {"query": None},
        None,
        "invalid_argument",
        ['such as {"query": "missing values"}', "but got null"],
    ),
    "search-limit-that-is-zero": (
        "search_analyses",
        {"query": "nulls", "limit": 0},
        None,
        "invalid_argument",
        ["got 0 for limit", "must be an integer of at least 1"],
    ),
    "column-of-a-type-the-analysis-cannot-use": (
        "summarize_numeric",
        {"columns": ["city"]},
        "csv",
        "unsupported_column_type",
        ["column 'city'", "Use summarize_categorical"],
    ),
    "column-labels-that-form-a-multiindex": (
        "summarize_numeric",
        {},
        "multi-index",
        "ambiguous_column_labels",
        ["MultiIndex", "Flatten them in the data file first"],
    ),
    # Paths.
    "data-path-of-another-format": (
        "summarize_numeric",
        {"data_path": "{directory}/shop.xlsx"},
        None,
        "unsupported_data_format",
        ["'.xlsx'", ".csv"],
    ),
    "data-file-that-is-not-utf-8": (
        "summarize_numeric",
        {},
        "latin1",
        "unreadable_data_file",
        ["Can't read", "UTF-8"],
    ),
    "project-path-without-a-file": (
        "summarize_categorical",
        {"project_path": "{directory}/missing.toml"},
        "csv",
        "project_file_not_found",
        ["project_path"],
    ),
    "project-file-that-is-not-toml": (
        "summarize_categorical",
        {"project_path": "{project_not_toml}"},
        "csv",
        "invalid_project_file",
        ["project_path"],
    ),
    "project-file-with-a-misspelled-key": (
        "summarize_categorical",
        {"project_path": "{project_with_a_typo}"},
        "csv",
        "unknown_project_key",
        ["mask_sensitve", "Did you mean 'mask_sensitive'?"],
    ),
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

    response = anyio.run(call, name, arguments_of(tmp_path, arguments, kind))

    assert response.is_error
    assert response.structured_content["code"] == code
    message = response.structured_content["message"]
    assert [block.text for block in response.content] == [message]
    for fragment in fragments:
        assert fragment in message
    assert PYTHON_FIX.findall(message) == []
    assert python_in(message) == []
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


def arguments_of(tmp_path, arguments, kind):
    """A mistake's arguments with the data file as data_path, unless it gives
    one, and each path written as {directory}, {folder} (a directory named like a
    CSV file), {data} or the name of a project file filled in for the test's
    own."""
    data = data_file(tmp_path, kind)
    (tmp_path / "folder.csv").mkdir()
    names = {
        "directory": tmp_path,
        "folder": tmp_path / "folder.csv",
        "data": data,
    }
    for name, text in PROJECT_FILES.items():
        names[name] = tmp_path / f"{name}.toml"
        names[name].write_text(text, encoding="utf-8")
    filled = {
        key: value.format(**names) if isinstance(value, str) else value
        for key, value in arguments.items()
    }
    if kind is not None and "data_path" not in filled:
        filled["data_path"] = str(data)
    return filled


def data_file(tmp_path, kind):
    """The file a mistake reads: CSV, or a Parquet file that keeps an index,
    holds no column or has a MultiIndex of columns, or a CSV file that isn't
    UTF-8, which a plain CSV file can't be."""
    if kind == "csv":
        path = tmp_path / "shop.csv"
        path.write_text(CSV, encoding="utf-8")
    elif kind == "latin1":
        path = tmp_path / "shop.csv"
        path.write_bytes("order,city\nA1,Zürich\n".encode("latin-1"))
    elif kind == "zoned":
        path = tmp_path / "zoned.parquet"
        ZONED.to_parquet(path)
    elif kind == "multi-index":
        path = tmp_path / "shop.parquet"
        MULTI_INDEX.to_parquet(path)
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
