"""The Project and masking over MCP, tested as an MCP client meets them.

An MCP caller names a project file with project_path, or the user starts the
server with --project so that every call uses it. Masked results over MCP
equal masked results in Python. Masking is a guardrail against accidental
exposure, not anonymization or access control (ADR 0001).
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

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import INTERNAL_ERROR
from serving import serving
from stdio import stdio_environment

import agent_eda as ae
from agent_eda.errors import AgentEDAError
from agent_eda.mcp import create_server

MASKING = sorted(a.name for a in ae.list_analyses().analyses if a.supports_masking)

CSV = (
    "customer,plan,spend\n"
    "ann@example.org,pro,10\n"
    "bo@example.org,pro,20\n"
    "ann@example.org,team,30\n"
)

# Which project a call uses, and what masking is, as every MCP text says it.
PRECEDENCE = (
    "A call's project_path replaces the MCP server's project (agent-eda-mcp "
    "--project) for that call, and a call's mask_sensitive wins over both."
)
GUARDRAIL = (
    "Masking is a guardrail against accidental exposure, not anonymization or "
    "access control."
)

MASKED = '[data_policy]\nsensitive_columns = ["customer"]\nmask_sensitive = true\n'
UNMASKED = '[data_policy]\nsensitive_columns = ["customer"]\nmask_sensitive = false\n'


def test_masking_mcp_tools_take_project_path_in_place_of_project():
    tools = anyio.run(list_tools)

    for tool in tools:
        properties = tool.input_schema["properties"]
        assert "project" not in properties, tool.name
        if tool.name in MASKING:
            assert properties["project_path"]["anyOf"] == [
                {"type": "string"},
                {"type": "null"},
            ]
            assert properties["project_path"]["default"] is None
            assert "mask_sensitive" in properties
        else:
            assert "project_path" not in properties, tool.name
            assert "mask_sensitive" not in properties, tool.name
    assert {tool.name for tool in tools} >= set(MASKING)


def test_project_path_and_mask_sensitive_say_masking_is_only_a_guardrail():
    tools = [tool for tool in anyio.run(list_tools) if tool.name in MASKING]

    for tool in tools:
        properties = tool.input_schema["properties"]
        for name in ("project_path", "mask_sensitive"):
            description = properties[name]["description"]
            assert (
                "Masking is a guardrail against accidental exposure, not "
                "anonymization or access control" in description
            ), (tool.name, name)
        # mask_sensitive says where an MCP caller's project comes from.
        assert "project_path" in properties["mask_sensitive"]["description"]
        assert "--project" in properties["mask_sensitive"]["description"]


def test_the_precedence_and_the_guardrail_read_the_same_wherever_they_appear():
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            return client.instructions, (await client.list_tools()).tools

    instructions, tools = anyio.run(scenario)
    properties = [
        tool.input_schema["properties"][name]["description"]
        for tool in tools
        if tool.name in MASKING
        for name in ("project_path", "mask_sensitive")
    ]
    finished = run_agent_eda_mcp("--help")

    for text in [*properties, instructions, finished.stdout]:
        # argparse breaks lines after a hyphen, as in agent-eda-|mcp.
        unwrapped = " ".join(re.sub(r"(?<=\w)-\n\s*", "-", text).split())
        assert PRECEDENCE in unwrapped, text
        assert GUARDRAIL in unwrapped, text


def test_no_masking_mcp_tool_schema_keeps_the_project_definitions():
    tools = anyio.run(list_tools)

    for tool in tools:
        definitions = tool.input_schema.get("$defs", {})
        assert "Project" not in definitions, tool.name
        assert "DataPolicy" not in definitions, tool.name


def test_a_call_with_project_path_equals_the_python_call_with_that_project(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = write(tmp_path / "project.toml", MASKED)
    expected = ae.summarize_categorical(
        read(data_path), project=ae.Project.load(project_path)
    )

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "project_path": str(project_path)},
    )

    assert not response.is_error
    assert response.structured_content == expected.to_dict()
    assert response.content[0].text.splitlines()[0] == str(expected).splitlines()[0]
    assert response.structured_content["masking"]["applied"] is True
    assert "ann@example.org" not in response.content[0].text


def test_a_relative_project_path_resolves_against_the_working_directory(
    tmp_path, monkeypatch
):
    write(tmp_path / "shop.csv", CSV)
    write(tmp_path / "project.toml", MASKED)
    monkeypatch.chdir(tmp_path)

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": "shop.csv", "project_path": "project.toml"},
    )

    assert response.structured_content["masking"]["applied"] is True


# A file that isn't TOML or leaves out a key gets an MCP fix in place of the
# Python one: tests/test_mcp_fixes.py.
def test_a_missing_project_file_says_how_to_write_one_and_names_project_path(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = tmp_path / "project.toml"
    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load(project_path)

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "project_path": str(project_path)},
    )

    assert response.is_error
    message = response.structured_content["message"]
    assert response.structured_content["code"] == raised.value.code
    assert message.startswith(f"project_path leads to '{project_path}', but no file")
    assert 'a [data_policy] table that has sensitive_columns = ["email"]' in message
    assert "mask_sensitive = true or false" in message
    # The Python message tells to save a project, which no MCP tool can.
    assert "project.save" in str(raised.value)
    assert "save" not in message
    assert "Project.load" not in message
    assert [block.text for block in response.content] == [message]


def test_a_project_file_with_a_misspelled_key_gets_the_python_code_and_names_project_path(
    tmp_path,
):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = write(
        tmp_path / "project.toml",
        "[data_policy]\nsensitive_columns = []\nmask_sensitve = true\n",
    )
    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load(project_path)

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "project_path": str(project_path)},
    )

    assert response.is_error
    message = response.structured_content["message"]
    assert response.structured_content["code"] == raised.value.code
    assert message == str(raised.value).replace(
        "Project.load found an unknown key 'mask_sensitve' in the [data_policy] "
        f"table of '{project_path}'.",
        "The MCP server found an unknown key 'mask_sensitve' in the [data_policy] "
        f"table of the project_path file '{project_path}'.",
    )
    assert "Did you mean 'mask_sensitive'?" in message


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs named pipes and POSIX permissions that bind the user",
)
@pytest.mark.parametrize("kind", ["named-pipe", "unreadable"])
def test_a_project_file_that_cannot_be_read_is_a_tool_error_not_a_bug(tmp_path, kind):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = tmp_path / "project.toml"
    if kind == "named-pipe":
        os.mkfifo(project_path)
    else:
        write(project_path, MASKED).chmod(0)
    try:
        response = anyio.run(
            call,
            "summarize_categorical",
            {"data_path": str(data_path), "project_path": str(project_path)},
        )
    finally:
        project_path.chmod(0o600)

    assert response.is_error
    assert response.structured_content["code"] == "invalid_project_file"
    message = response.structured_content["message"]
    assert message.startswith("The MCP server ")
    assert f"the project_path file '{project_path}'" in message
    assert "Project.load" not in message


def test_a_project_path_that_is_not_a_string_is_a_tool_error(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(data_path), "project_path": 3}
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert "project_path" in response.structured_content["message"]


def test_the_mcp_server_project_applies_to_every_call_that_takes_one(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project = ae.Project.load(write(tmp_path / "project.toml", MASKED))
    expected = ae.summarize_categorical(read(data_path), project=project)

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(data_path)}, project
    )

    assert response.structured_content == expected.to_dict()
    assert response.structured_content["masking"]["decided_by"] == "project"


def test_the_mcp_server_project_leaves_analyses_without_one_alone(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project = ae.Project.load(write(tmp_path / "project.toml", MASKED))
    expected = ae.summarize_numeric(read(data_path))

    response = anyio.run(
        call, "summarize_numeric", {"data_path": str(data_path)}, project
    )

    assert response.structured_content == expected.to_dict()


@pytest.mark.parametrize(
    ("mcp_server_project", "per_call", "mask_sensitive", "applied", "decided_by"),
    [
        # A per-call project_path replaces the MCP server's project.
        (MASKED, UNMASKED, None, False, "project"),
        (UNMASKED, MASKED, None, True, "project"),
        # A per-call mask_sensitive wins over either project.
        (MASKED, None, False, False, "mask_sensitive"),
        (UNMASKED, None, True, True, "mask_sensitive"),
        (UNMASKED, UNMASKED, True, True, "mask_sensitive"),
        (MASKED, MASKED, False, False, "mask_sensitive"),
        # Without a project, nothing is masked.
        (None, None, None, False, "default"),
    ],
    ids=[
        "call-project-unmasks",
        "call-project-masks",
        "call-false-over-mcp-server-project",
        "call-true-over-mcp-server-project",
        "call-true-over-both",
        "call-false-over-both",
        "no-project",
    ],
)
def test_project_path_wins_over_the_mcp_server_project_and_mask_sensitive_over_both(
    tmp_path, mcp_server_project, per_call, mask_sensitive, applied, decided_by
):
    data_path = write(tmp_path / "shop.csv", CSV)
    server_project = None
    if mcp_server_project is not None:
        server_project = ae.Project.load(
            write(tmp_path / "mcp_server.toml", mcp_server_project)
        )
    arguments = {"data_path": str(data_path), "mask_sensitive": mask_sensitive}
    call_project = server_project
    if per_call is not None:
        arguments["project_path"] = str(write(tmp_path / "call.toml", per_call))
        call_project = ae.Project.load(arguments["project_path"])
    expected = ae.summarize_categorical(
        read(data_path), project=call_project, mask_sensitive=mask_sensitive
    )

    response = anyio.run(call, "summarize_categorical", arguments, server_project)

    assert response.structured_content == expected.to_dict()
    masking = response.structured_content["masking"]
    assert (masking["applied"], masking["decided_by"]) == (applied, decided_by)


# Data whose sensitive columns, email and segment, every masking analysis shows
# when masking is off: as levels, invalid values, classes, declared sentinel
# values or row IDs from id_column.
PEOPLE = (
    "email,segment,plan,spend\n"
    "ann@example.org,gold-tier,pro,10\n"
    "bo@example.org,silver-tier,pro,-999\n"
    "cy@example.org,no-tier,team,30\n"
    "ann@example.org,gold-tier,pro,10\n"
    "dee@example.org,silver-tier,team,-999\n"
)
PEOPLE_POLICY = (
    '[data_policy]\nsensitive_columns = ["email", "segment"]\nmask_sensitive = true\n'
)
SENSITIVE = [
    "ann@example.org",
    "bo@example.org",
    "cy@example.org",
    "dee@example.org",
    "gold-tier",
    "silver-tier",
    "no-tier",
]
# The options of each masking analysis that make it show sensitive values.
MASKING_CALLS = {
    "detect_missing_sentinels": {
        "id_column": "email",
        "sentinel_values": {"segment": ["no-tier"]},
    },
    "find_duplicate_rows": {"columns": ["plan", "spend"], "id_column": "email"},
    "profile_target": {
        "target_column": "segment",
        "task": "multiclass_classification",
        "id_column": "email",
    },
    "profile_missing_values": {
        "sentinel_values": {"segment": ["no-tier"], "email": ["cy@example.org"]}
    },
    "summarize_categorical": {},
    "validate_column_types": {
        "schema": {"email": "integer", "segment": "integer"},
        "id_column": "email",
    },
}


def test_the_masking_check_covers_every_masking_analysis():
    assert sorted(MASKING_CALLS) == MASKING


@pytest.mark.parametrize("name", sorted(MASKING_CALLS))
def test_masking_keeps_every_sensitive_value_out_of_the_response(tmp_path, name):
    data_path = write(tmp_path / "people.csv", PEOPLE)
    project_path = write(tmp_path / "project.toml", PEOPLE_POLICY)
    options = MASKING_CALLS[name]
    expected = getattr(ae, name)(
        read(data_path), project=ae.Project.load(project_path), **options
    )
    arguments = {"data_path": str(data_path), "project_path": str(project_path)}

    masked = anyio.run(call, name, {**arguments, **options})
    unmasked = anyio.run(call, name, {**arguments, **options, "mask_sensitive": False})

    assert masked.structured_content == expected.to_dict()
    assert masked.content[0].text.splitlines()[0] == str(expected).splitlines()[0]
    assert shown(masked) == []
    # The values the response would show without masking.
    assert shown(unmasked) != []


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
@pytest.mark.parametrize("name", sorted(MASKING_CALLS))
def test_masking_keeps_every_sensitive_value_out_of_the_written_table(
    tmp_path, name, extension
):
    data_path = write(tmp_path / "people.csv", PEOPLE)
    project_path = write(tmp_path / "project.toml", PEOPLE_POLICY)
    options = MASKING_CALLS[name]
    arguments = {"data_path": str(data_path), "project_path": str(project_path)}
    masked_path = tmp_path / f"masked{extension}"
    unmasked_path = tmp_path / f"unmasked{extension}"
    expected = getattr(ae, name)(
        read(data_path), project=ae.Project.load(project_path), **options
    ).to_frame()

    anyio.run(call, name, {**arguments, **options, "table_path": str(masked_path)})
    anyio.run(
        call,
        name,
        {
            **arguments,
            **options,
            "mask_sensitive": False,
            "table_path": str(unmasked_path),
        },
    )

    masked = table_text(masked_path)
    assert len(read_table(masked_path)) == len(expected)
    assert [value for value in SENSITIVE if value in masked] == []
    # The values the table would hold without masking.
    unmasked = table_text(unmasked_path)
    assert [value for value in SENSITIVE if value in unmasked] != []


def test_agent_eda_mcp_with_project_masks_every_call_over_stdio(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = write(tmp_path / "project.toml", MASKED)
    expected = ae.summarize_categorical(
        read(data_path), project=ae.Project.load(project_path)
    )

    # A relative path, resolved against the server's working directory.
    response = anyio.run(
        call_over_stdio,
        tmp_path,
        ["--project", "project.toml"],
        "summarize_categorical",
        {"data_path": "shop.csv"},
    )

    assert response.structured_content == expected.to_dict()
    assert response.structured_content["masking"]["applied"] is True


@pytest.mark.parametrize(
    "content",
    [None, '[data_policy]\nsensitive_columns = ["customer"]\n'],
    ids=["missing", "key-left-out"],
)
def test_agent_eda_mcp_with_a_bad_project_file_stops_with_its_message(
    tmp_path, content
):
    project_path = tmp_path / "project.toml"
    if content is not None:
        write(project_path, content)
    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load(project_path)

    finished = run_agent_eda_mcp("--project", str(project_path))

    assert finished.returncode == 1
    assert finished.stderr == f"{raised.value}\n"
    assert finished.stdout == ""


def test_agent_eda_mcp_help_documents_the_project_option():
    finished = run_agent_eda_mcp("--help")

    assert finished.returncode == 0
    text = " ".join(finished.stdout.split())
    assert "--project PATH" in text
    assert "project_path" in text
    assert "guardrail against accidental exposure, not anonymization or access" in text


async def list_tools():
    async with Client(create_server(), mode="legacy") as client:
        return (await client.list_tools()).tools


async def call(name, arguments, project=None):
    """Call an MCP tool on a server started with this Project, as --project does."""
    async with Client(create_server(project=project), mode="legacy") as client:
        return await client.call_tool(name, arguments)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def read(path):
    """The DataFrame the server reads from this file's CSV text, built by hand."""
    frames = {
        CSV: pd.DataFrame(
            {
                "customer": ["ann@example.org", "bo@example.org", "ann@example.org"],
                "plan": ["pro", "pro", "team"],
                "spend": [10, 20, 30],
            }
        ),
        PEOPLE: pd.DataFrame(
            {
                "email": [
                    "ann@example.org",
                    "bo@example.org",
                    "cy@example.org",
                    "ann@example.org",
                    "dee@example.org",
                ],
                "segment": [
                    "gold-tier",
                    "silver-tier",
                    "no-tier",
                    "gold-tier",
                    "silver-tier",
                ],
                "plan": ["pro", "pro", "team", "pro", "team"],
                "spend": [10, -999, 30, 10, -999],
            }
        ),
    }
    return frames[path.read_text(encoding="utf-8")]


# The console script installed next to this interpreter.
AGENT_EDA_MCP = str(Path(sys.executable).parent / "agent-eda-mcp")


def run_agent_eda_mcp(*args):
    return subprocess.run(
        [AGENT_EDA_MCP, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


async def call_over_stdio(cwd, args, name, arguments):
    """Start agent-eda-mcp with these arguments, as an MCP client does."""
    server = StdioServerParameters(
        command=AGENT_EDA_MCP, args=args, env=stdio_environment(), cwd=str(cwd)
    )
    async with Client(server, mode="legacy") as client:
        return await client.call_tool(name, arguments)


def read_table(path):
    """A written table, read as CSV or Parquet by its extension."""
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    with warnings.catch_warnings():
        # pandas 2.2 deprecates the internals pyarrow 14 builds DataFrames from.
        warnings.filterwarnings(
            "ignore",
            message="Passing a BlockManager to DataFrame is deprecated",
            category=DeprecationWarning,
        )
        return pd.read_parquet(path)


def table_text(path):
    """Every column name and cell of a written table, as CSV text."""
    return read_table(path).to_csv()


def shown(response):
    """The sensitive values in a response's text or structured content."""
    text = " ".join(block.text for block in response.content)
    record = json.dumps(response.structured_content, ensure_ascii=False)
    return [value for value in SENSITIVE if value in text or value in record]


def failing_summarize_categorical():
    """summarize_categorical as the server serves it, with a bug whose message
    holds a value of the data."""

    def failing(data, **options):
        raise ZeroDivisionError(f"bug at {data['customer'].iloc[0]}")

    # The registry knows it by its name and module, so it keeps its parameters
    # and its masking.
    failing.__name__ = "summarize_categorical"
    failing.__module__ = ae.summarize_categorical.__module__
    return failing


async def call_failing(arguments, project=None):
    served = serving(failing_summarize_categorical(), project=project)
    async with Client(served, mode="legacy") as client:
        with pytest.raises(MCPError) as raised:
            await client.call_tool("summarize_categorical", arguments)
        return raised.value


@pytest.mark.parametrize(
    ("how", "masks"),
    [
        ("project_path", True),
        ("mask_sensitive", True),
        ("server_project", True),
        ("project_path_masking_off", False),
        ("call_off", False),
        ("no_project", False),
    ],
)
def test_an_internal_error_with_masking_on_names_only_the_exception_type(
    tmp_path, how, masks
):
    data_path = write(tmp_path / "data.csv", CSV)
    on = write(tmp_path / "on.toml", MASKED)
    off = write(tmp_path / "off.toml", UNMASKED)
    arguments: dict[str, Any] = {"data_path": str(data_path)}
    project = None
    if how == "project_path":
        arguments["project_path"] = str(on)
    elif how == "mask_sensitive":
        arguments |= {"project_path": str(off), "mask_sensitive": True}
    elif how == "server_project":
        project = ae.Project.load(on)
    elif how == "project_path_masking_off":
        arguments["project_path"] = str(off)
    elif how == "call_off":
        arguments |= {"project_path": str(on), "mask_sensitive": False}

    error = anyio.run(call_failing, arguments, project)

    assert error.code == INTERNAL_ERROR
    assert "ZeroDivisionError" in error.message
    assert ("ann@example.org" not in error.message) is masks
    if masks:
        assert error.message == (
            "summarize_categorical failed with an unexpected error, which is a bug "
            "in agent-eda, not in the call: ZeroDivisionError. The MCP server's log "
            "on stderr has the traceback."
        )


def test_an_internal_error_of_an_analysis_without_masking_keeps_its_message(tmp_path):
    def failing(data, **options):
        raise ZeroDivisionError("bug at row 3")

    failing.__name__ = "summarize_numeric"
    failing.__module__ = ae.summarize_numeric.__module__
    data_path = write(tmp_path / "data.csv", CSV)

    async def scenario():
        served = serving(failing)
        async with Client(served, mode="legacy") as client:
            with pytest.raises(MCPError) as raised:
                await client.call_tool(
                    "summarize_numeric", {"data_path": str(data_path)}
                )
            return raised.value

    error = anyio.run(scenario)

    assert "ZeroDivisionError: bug at row 3" in error.message


def test_the_instructions_and_the_readme_say_what_masking_never_covers():
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            return client.instructions

    instructions = " ".join(anyio.run(scenario).split())
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    section = readme.split("### Masking over MCP")[1].split("\n## ")[0]
    masking = " ".join(section.split())

    assert "Numeric statistics and index labels are never masked;" in instructions
    assert "other tools ignore the project." in instructions
    assert "never touches numeric statistics" in masking
    assert "or index labels" in masking
    assert "the MCP tools without masking don't read the project" in masking
    assert "six MCP tools that support masking" in masking
    assert len(MASKING) == 6


def test_the_readme_counts_the_analyses_that_can_show_raw_values():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    section = readme.split("## Projects and masking")[1].split("\n## ")[0]

    [paragraph] = [p for p in section.split("\n\n") if p.startswith("Six analyses")]
    for name in MASKING:
        assert f"`{name}`" in paragraph
