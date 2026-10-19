"""Smoke test of the MCP server over real stdio: start agent-eda-mcp, call one MCP tool.

CI's mcp-smoke job runs this script on Python 3.12 and 3.14 with the mcp extra
installed. Unlike the MCP test modules, which skip without the extra, it fails
when the extra or the console script is missing. Run it with the environment's
Python: uv run --extra mcp python -I tests/smoke_mcp.py
"""

import sys
import tempfile
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

CSV = "city,score,note\nOslo,1.5,N/A\nLyon,,ok\nOslo,2.5,\n"


async def smoke(directory: Path) -> int:
    (directory / "shop.csv").write_text(CSV, encoding="utf-8")
    server = StdioServerParameters(
        # The console script installed next to this interpreter.
        command=str(Path(sys.executable).parent / "agent-eda-mcp"),
        cwd=str(directory),
    )
    async with Client(server, mode="legacy") as client:
        names = [tool.name for tool in (await client.list_tools()).tools]
        print(f"{len(names)} MCP tools: {', '.join(names)}")
        response = await client.call_tool("describe_dataset", {"data_path": "shop.csv"})
    text = "\n".join(c.text for c in response.content if isinstance(c, TextContent))
    print(text)
    status = (response.structured_content or {}).get("status")
    completed = text.startswith("describe_dataset: completed") and status == "completed"
    return 0 if completed and not response.is_error else 1


with tempfile.TemporaryDirectory() as directory:
    sys.exit(anyio.run(smoke, Path(directory)))
