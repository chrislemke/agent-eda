"""Paths and files over MCP: what a path argument can do wrong and what comes back.

Every problem with data_path, table_path or project_path is a tool error that
names its cause, never a JSON-RPC internal error, and a table_path never leaves
a partial file or writes through a symbolic link.
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

import errno
import inspect
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mcp import Client, MCPError
from mcp_helpers import call, read_parquet
from serving import serving

from agent_eda.mcp import create_server
from agent_eda.results import SummarizeNumericResult

ROOT = Path(__file__).resolve().parent.parent

CSV = "city,score\nOslo,1.5\nLyon,2.5\nOslo,4.0\n"
MASKED = '[data_policy]\nsensitive_columns = ["city"]\nmask_sensitive = true\n'

POSIX_USER = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions and symbolic links that bind the user",
)

# What no path problem may turn into.
CODED = {"data_file_not_found", "unreadable_data_file", "invalid_argument"}


def test_the_mcp_server_builder_has_no_test_hook_and_no_document_mentions_one():
    assert list(inspect.signature(create_server).parameters) == ["project"]
    doc = inspect.getdoc(create_server) or ""
    assert "analyses:" not in doc
    assert "test seam" not in doc
    for path in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        text = path.read_text(encoding="utf-8")
        assert "create_server(analyses" not in text, path
        assert not re.search(r"\banalyses:\s+A test seam", text), path


# --- data_path and table_path problems ---------------------------------------


def loop_path(tmp_path):
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    return str(tmp_path / "a" / "data.csv")


def no_search_path(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "data.csv").write_text(CSV, encoding="utf-8")
    locked.chmod(0)
    return str(locked / "data.csv")


@pytest.mark.parametrize("argument", ["data_path", "table_path"])
@pytest.mark.parametrize(
    ("kind", "cause"),
    [
        ("long", "longer than 255 characters"),
        ("nul", "NUL character"),
        pytest.param("loop", "symbolic links", marks=POSIX_USER),
        pytest.param("locked", "search", marks=POSIX_USER),
    ],
)
def test_a_path_the_os_refuses_is_a_coded_tool_error_naming_the_cause(
    tmp_path, argument, kind, cause
):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    bad = {
        "long": lambda: "x" * 300 + ".csv",
        "nul": lambda: "a\0b.csv",
        "loop": lambda: loop_path(tmp_path),
        "locked": lambda: no_search_path(tmp_path),
    }[kind]()
    arguments = {"data_path": str(shop)}
    if argument == "data_path":
        arguments["data_path"] = bad
    else:
        arguments["table_path"] = bad.replace("data.csv", "out.csv")
    try:
        response = anyio.run(call, "describe_dataset", arguments)
    finally:
        for locked in tmp_path.glob("locked"):
            locked.chmod(0o700)

    assert response.is_error
    assert response.structured_content["code"] in CODED
    assert cause in response.structured_content["message"]


def test_a_table_path_on_a_read_only_volume_is_refused_before_the_analysis_runs(
    tmp_path, monkeypatch
):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")

    def read_only(path):
        return SimpleNamespace(f_flag=os.ST_RDONLY)

    monkeypatch.setattr(os, "statvfs", read_only)
    monkeypatch.setattr(os, "access", lambda *a, **k: False)

    response = anyio.run(
        call,
        "break_on_run",
        {"data_path": str(shop), "table_path": str(tmp_path / "out.csv")},
        serving(break_on_run),
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert "read-only" in response.structured_content["message"]
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_a_write_that_fails_partway_removes_its_partial_file(
    tmp_path, monkeypatch, extension
):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    table_path = tmp_path / f"summary{extension}"
    real_write = os.write

    def full_disk(fd, data):
        if len(data) < 20:  # the event loop's own wake-up bytes
            return real_write(fd, data)
        real_write(fd, bytes(data)[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", full_disk)
        response = anyio.run(
            call,
            "summarize_numeric",
            {"data_path": str(shop), "table_path": str(table_path)},
        )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert "No space left on device" in response.structured_content["message"]
    assert not table_path.exists()
    # The same call works once the disk has room, not table_file_exists.
    retry = anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(shop), "table_path": str(table_path)},
    )
    assert not retry.is_error
    assert table_path.exists()


# --- "~" and symbolic links --------------------------------------------------


def test_a_tilde_expands_in_data_path_table_path_and_project_path_alike(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "shop.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "project.toml").write_text(MASKED, encoding="utf-8")
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    response = anyio.run(
        call,
        "summarize_categorical",
        {
            "data_path": "~/shop.csv",
            "table_path": "~/levels.csv",
            "project_path": "~/project.toml",
        },
    )

    assert not response.is_error, response.content
    assert response.structured_content["masking"]["applied"] is True
    assert response.structured_content["table_file"]["path"] == str(
        tmp_path.resolve() / "levels.csv"
    )
    assert (tmp_path / "levels.csv").exists()
    assert not (elsewhere / "~").exists()


@POSIX_USER
def test_a_dangling_symlink_as_table_path_is_refused_and_nothing_is_written(tmp_path):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    target = tmp_path / "target.csv"
    link = tmp_path / "link.csv"
    link.symlink_to(target)

    response = anyio.run(
        call,
        "break_on_run",
        {"data_path": str(shop), "table_path": str(link)},
        serving(break_on_run),
    )

    assert response.is_error
    assert response.structured_content["code"] == "table_file_exists"
    assert "symbolic link" in response.structured_content["message"]
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.csv", "shop.csv"]


@POSIX_USER
def test_a_project_path_the_os_refuses_is_a_tool_error_naming_the_cause(tmp_path):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    for project_path, cause in [
        ("x" * 300 + ".toml", "longer than 255 characters"),
        (loop_path(tmp_path), "symbolic links"),
    ]:
        response = anyio.run(
            call,
            "summarize_categorical",
            {"data_path": str(shop), "project_path": project_path},
        )

        assert response.is_error
        assert response.structured_content["code"] == "invalid_project_file"
        assert cause in response.structured_content["message"]
        assert "read permission" not in response.structured_content["message"]

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(shop), "project_path": "a\0b.toml"},
    )
    assert response.structured_content["code"] == "invalid_argument"
    assert "NUL character" in response.structured_content["message"]


# --- file contents the reader and writer can't handle ------------------------


def test_a_parquet_file_with_broken_pandas_metadata_is_unreadable_not_a_bug(tmp_path):
    table = pa.table({"a": [1, 2]}).replace_schema_metadata(
        {
            b"pandas": b'{"columns": [{"name": "a"}], "index_columns": ["nope"], '
            b'"column_indexes": []}'
        }
    )
    path = tmp_path / "bad.parquet"
    pq.write_table(table, path)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.is_error
    assert response.structured_content["code"] == "unreadable_data_file"


@pytest.mark.parametrize("extension", [".parquet", ".csv"])
def test_a_uint64_sentinel_in_a_table_path_succeeds_or_gives_a_coded_error(
    tmp_path, extension
):
    data = pd.DataFrame({"x": np.array([1, 2, 2**64 - 1], dtype="uint64")})
    data.to_parquet(tmp_path / "shop.parquet")
    table_path = tmp_path / f"sentinels{extension}"

    response = anyio.run(
        call,
        "detect_missing_sentinels",
        {
            "data_path": str(tmp_path / "shop.parquet"),
            "sentinel_values": {"x": [2**64 - 1]},
            "table_path": str(table_path),
        },
    )

    if response.is_error:
        assert response.structured_content["code"] in CODED
    else:
        assert table_path.exists()
        written = (
            read_parquet(table_path)
            if extension == ".parquet"
            else pd.read_csv(table_path, dtype=str)
        )
        assert str(2**64 - 1) in written["value"].astype(str).tolist()


# --- reader messages name the cause ------------------------------------------


@POSIX_USER
def test_a_named_pipe_is_not_a_regular_file(tmp_path):
    path = tmp_path / "pipe.csv"
    os.mkfifo(path)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.is_error
    message = response.structured_content["message"]
    assert "isn't a regular file" in message
    assert "Did you mean" not in message


@POSIX_USER
def test_a_file_without_read_permission_says_permission_denied(tmp_path):
    path = tmp_path / "secret.csv"
    path.write_text(CSV, encoding="utf-8")
    path.chmod(0)
    try:
        response = anyio.run(call, "describe_dataset", {"data_path": str(path)})
    finally:
        path.chmod(0o600)

    assert response.is_error
    assert response.structured_content["code"] == "unreadable_data_file"
    message = response.structured_content["message"]
    assert "permission denied" in message.lower()
    assert "UTF-8" not in message


def test_duplicate_parquet_field_names_are_named_not_dumped(tmp_path):
    path = tmp_path / "dup.parquet"
    pq.write_table(pa.table([pa.array([1]), pa.array([2])], names=["a", "a"]), path)

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.is_error
    assert response.structured_content["code"] == "unreadable_data_file"
    message = response.structured_content["message"]
    assert "more than one column named 'a'" in message
    assert "__fragment_index" not in message


def test_an_unknown_extension_names_the_last_suffix(tmp_path):
    path = tmp_path / "sales.2024.json"
    path.write_text("[]", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.structured_content["code"] == "unsupported_data_format"
    assert "has the extension '.json'," in response.structured_content["message"]


def test_a_gzipped_unknown_extension_names_both_suffixes(tmp_path):
    path = tmp_path / "sales.json.gz"
    path.write_text("[]", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert "has the extension '.json.gz'," in response.structured_content["message"]


def test_a_single_column_named_with_a_semicolon_hints_at_the_separator(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("city;score\nOslo;1.5\nLyon;2.5\n", encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    assert response.is_error
    assert response.structured_content["code"] == "unreadable_data_file"
    message = response.structured_content["message"]
    assert "'city;score'" in message
    assert "semicolon" in message
    assert "commas" in message


# --- unknown MCP tool names --------------------------------------------------


def unknown_tool(name):
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            with pytest.raises(MCPError) as raised:
                await client.call_tool(name, {})
            return raised.value, [t.name for t in (await client.list_tools()).tools]

    return anyio.run(scenario)


def test_an_unknown_tool_name_gets_the_closest_names_and_every_tool():
    error, tools = unknown_tool("Describe_Dataset")

    assert "There is no MCP tool named 'Describe_Dataset'." in error.message
    assert "Did you mean 'describe_dataset'?" in error.message
    assert "MCP tools: " + ", ".join(tools) in error.message


@pytest.mark.parametrize("name", ["guide", "describe_analysis"])
def test_the_python_discovery_names_point_to_the_instructions_and_descriptions(name):
    error, tools = unknown_tool(name)

    assert f"no MCP tool named {name!r}" in error.message
    assert "instructions" in error.message
    assert "description" in error.message
    assert "MCP tools: " + ", ".join(tools) in error.message


# --- the table_path CSV and text ---------------------------------------------


def test_an_empty_string_in_a_table_path_csv_is_quoted_and_reads_back_as_text(
    tmp_path,
):
    data_path = tmp_path / "shop.parquet"
    pd.DataFrame({"t": ["", "a", None, "a"]}).to_parquet(data_path)
    table_path = tmp_path / "levels.csv"

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "table_path": str(table_path)},
    )

    assert not response.is_error
    raw = table_path.read_text(encoding="utf-8")
    assert re.search(r'^t,"",', raw, flags=re.MULTILINE)
    # The empty string is a label; the missing value is no row of the table.
    back = anyio.run(call, "describe_dataset", {"data_path": str(table_path)})
    labels = next(
        c for c in back.structured_content["columns"] if c["column"] == "label"
    )
    assert labels["missing_count"] == 0


def test_a_quoted_empty_field_reads_as_text_and_an_unquoted_one_as_missing(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text('a,b\n"",1\n,2\nx,3\n"say ""hi""",4\n"",5\n', encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    column = next(
        c for c in response.structured_content["columns"] if c["column"] == "a"
    )
    assert column["missing_count"] == 1
    assert column["distinct_count"] == 3
    other = next(
        c for c in response.structured_content["columns"] if c["column"] == "b"
    )
    assert other["dtype"] != "str"


def test_the_table_path_line_and_the_text_keep_to_the_caps(tmp_path):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    deep = tmp_path
    for _ in range(14):
        deep = deep / "a-directory-with-a-long-name"
    deep.mkdir(parents=True)
    table_path = deep / "table.csv"

    response = anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(shop), "table_path": str(table_path)},
    )

    assert not response.is_error
    lines = response.content[0].text.splitlines()
    assert len(lines) <= 40
    assert max(len(line) for line in lines) <= 120
    assert lines[-1].startswith("Full table written")
    assert response.structured_content["table_file"]["path"] == str(
        table_path.resolve()
    )


def test_a_text_that_is_already_full_loses_a_line_to_the_table_path_line(tmp_path):
    shop = tmp_path / "shop.csv"
    columns = {f"c{n}": [1.0, 2.0, 3.0] for n in range(12)}
    pd.DataFrame(columns).to_csv(shop, index=False)
    plain = anyio.run(call, "describe_dataset", {"data_path": str(shop)})
    with_table = anyio.run(
        call,
        "describe_dataset",
        {"data_path": str(shop), "table_path": str(tmp_path / "t.csv")},
    )

    assert len(plain.content[0].text.splitlines()) <= 40
    assert len(with_table.content[0].text.splitlines()) <= 40
    assert with_table.content[0].text.splitlines()[-1].startswith("Full table written")


def break_on_run(data: pd.DataFrame) -> SummarizeNumericResult:
    """Does the analysis run?

    A stand-in analysis that only these tests serve. It raises if it runs,
    which the MCP server reports as a bug, not as a tool error.

    Statuses:
        completed: Never; the stand-in always raises.
    """
    raise ZeroDivisionError("the analysis ran")


def test_a_quoted_empty_field_makes_a_numeric_column_text_and_a_header_unnamed(
    tmp_path,
):
    path = tmp_path / "numbers.csv"
    path.write_text('"",b\n1,2\n"",3\n,4\n', encoding="utf-8")

    response = anyio.run(call, "describe_dataset", {"data_path": str(path)})

    first, second = response.structured_content["columns"]
    assert first["column"] == "Unnamed: 0"
    assert first["dtype"] in {"str", "object"}
    # The unquoted empty field is missing, the quoted one a value.
    assert (first["missing_count"], first["distinct_count"]) == (1, 2)
    assert second["dtype"] == "int64"


def test_a_table_path_line_that_only_the_path_overflows_takes_a_second_line(
    tmp_path,
):
    shop = tmp_path / "shop.csv"
    shop.write_text(CSV, encoding="utf-8")
    base = len(str(tmp_path.resolve() / "d" / "table.csv"))
    if base > 105:
        pytest.skip("the temporary directory's path is already too long to pad")
    pad = tmp_path / ("d" * (112 - base + 1))
    pad.mkdir()
    table_path = pad / "table.csv"
    assert 100 < len(str(table_path.resolve())) <= 120

    response = anyio.run(
        call,
        "summarize_numeric",
        {"data_path": str(shop), "table_path": str(table_path)},
    )

    *_, intro, path_line = response.content[0].text.splitlines()
    assert intro == "Full table written (1 row) to:"
    assert path_line == str(table_path.resolve())
