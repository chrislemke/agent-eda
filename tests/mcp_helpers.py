"""What the MCP tests share: the client call, the tool listing, file writing, the
Python reference run and the independent reference read."""

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from mcp import Client

import agent_eda as ae
from agent_eda.mcp import create_server


async def call(name: str, arguments: dict[str, Any], server=None):
    """Call one MCP tool as an MCP client does, on a fresh server unless one is given.

    The SDK client validates a result's structured content against the MCP tool's
    output schema unless isError is true.
    """
    async with Client(server or create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)


async def list_tools(server=None):
    """The MCP tools a server lists, on a fresh server unless one is given."""
    async with Client(server or create_server(), mode="legacy") as client:
        return (await client.list_tools()).tools


def listed_tools(server=None) -> dict[str, Any]:
    """The listed MCP tools by name."""
    return {tool.name: tool for tool in anyio.run(list_tools, server)}


def write(path: Path, text: str) -> Path:
    """Write a text file the server will read, and give its path back."""
    path.write_text(text, encoding="utf-8")
    return path


def write_frame(data: pd.DataFrame, path: Path) -> None:
    """Write data to path in the format its extension names."""
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        data.to_parquet(path)
        return
    sep = "\t" if ".tsv" in name else ","
    compression = "gzip" if name.endswith(".gz") else None
    data.to_csv(path, sep=sep, index=False, compression=compression)


def read_parquet(path: Path) -> pd.DataFrame:
    """A Parquet file, read without the deprecation warnings of pandas 2.2."""
    with warnings.catch_warnings():
        # pandas 2.2 deprecates the internals pyarrow 14 builds DataFrames from.
        warnings.filterwarnings(
            "ignore",
            message="(Passing a BlockManager to DataFrame|DatetimeTZBlock) is "
            "deprecated",
            category=DeprecationWarning,
        )
        return pd.read_parquet(path)


def read_table(path: Path) -> pd.DataFrame:
    """A written table, read as CSV or Parquet by its extension. Only an empty field
    is missing."""
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return read_parquet(path)


def reference_read(path: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The DataFrame the server reads from a file, by the file's CSV text.

    The test builds each frame by hand, so the reference never runs the server's
    reader. A text the test did not describe is a KeyError.
    """
    return frames[path.read_text(encoding="utf-8")]


def call_in_python(name: str, data: pd.DataFrame, arguments: dict[str, Any]):
    """Make the Python call that matches the MCP call: the data read from
    data_path, and the project loaded from project_path."""
    options = {k: v for k, v in arguments.items() if not k.endswith("_path")}
    if "project_path" in arguments:
        options["project"] = ae.Project.load(arguments["project_path"])
    return getattr(ae, name)(data, **options)
