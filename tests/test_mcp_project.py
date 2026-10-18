"""The Project and masking over MCP, tested as an MCP client meets them.

An MCP caller names a project file with project_path, or the user starts the
server with --project so that every call uses it. Masked results over MCP
equal masked results in Python. Masking is a guardrail against accidental
exposure, not anonymization or access control (ADR 0001).
"""

import pytest

pytest.importorskip(
    "mcp",
    reason="the MCP server tests need the mcp extra: pip install 'agent-eda[mcp]'",
)

import json
import subprocess
import sys
import warnings
from pathlib import Path

import anyio
import pandas as pd
from mcp import Client, StdioServerParameters

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
    assert [block.text for block in response.content] == [str(expected)]
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


@pytest.mark.parametrize(
    "content",
    [
        None,  # no file
        "[data_policy\n",  # not TOML
        '[data_policy]\nsensitive_columns = ["customer"]\n',  # a key left out
        "[data_policy]\nsensitive_columns = []\nmask_sensitve = true\n",  # typo
    ],
    ids=["missing", "not-toml", "key-left-out", "unknown-key"],
)
def test_a_bad_project_file_is_a_tool_error_with_the_python_code_and_message(
    tmp_path, content
):
    data_path = write(tmp_path / "shop.csv", CSV)
    project_path = tmp_path / "project.toml"
    if content is not None:
        write(project_path, content)
    with pytest.raises(AgentEDAError) as raised:
        ae.Project.load(project_path)

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "project_path": str(project_path)},
    )

    assert response.is_error
    assert response.structured_content == {
        "code": raised.value.code,
        "message": str(raised.value),
    }
    assert [block.text for block in response.content] == [str(raised.value)]


def test_a_project_path_that_is_not_a_string_is_a_tool_error(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(data_path), "project_path": 3}
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    assert "project_path" in response.structured_content["message"]


def test_the_server_project_applies_to_every_call_that_takes_one(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project = ae.Project.load(write(tmp_path / "project.toml", MASKED))
    expected = ae.summarize_categorical(read(data_path), project=project)

    response = anyio.run(
        call, "summarize_categorical", {"data_path": str(data_path)}, project
    )

    assert response.structured_content == expected.to_dict()
    assert response.structured_content["masking"]["decided_by"] == "project"


def test_the_server_project_leaves_analyses_without_one_alone(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    project = ae.Project.load(write(tmp_path / "project.toml", MASKED))
    expected = ae.summarize_numeric(read(data_path))

    response = anyio.run(
        call, "summarize_numeric", {"data_path": str(data_path)}, project
    )

    assert response.structured_content == expected.to_dict()


@pytest.mark.parametrize(
    ("server", "per_call", "mask_sensitive", "applied", "decided_by"),
    [
        # A per-call project_path replaces the server's project.
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
        "call-false-over-server",
        "call-true-over-server",
        "call-true-over-both",
        "call-false-over-both",
        "no-project",
    ],
)
def test_project_path_wins_over_the_server_project_and_mask_sensitive_over_both(
    tmp_path, server, per_call, mask_sensitive, applied, decided_by
):
    data_path = write(tmp_path / "shop.csv", CSV)
    server_project = None
    if server is not None:
        server_project = ae.Project.load(write(tmp_path / "server.toml", server))
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


def test_mask_sensitive_without_any_project_is_the_python_error(tmp_path):
    data_path = write(tmp_path / "shop.csv", CSV)
    with pytest.raises(AgentEDAError) as raised:
        ae.summarize_categorical(read(data_path), mask_sensitive=True)

    response = anyio.run(
        call,
        "summarize_categorical",
        {"data_path": str(data_path), "mask_sensitive": True},
    )

    assert response.structured_content == {
        "code": raised.value.code,
        "message": str(raised.value),
    }


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
    assert [block.text for block in masked.content] == [str(expected)]
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
    path.write_text(text)
    return path


def read(path):
    """The DataFrame the server reads from a CSV file."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


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
    server = StdioServerParameters(command=AGENT_EDA_MCP, args=args, cwd=str(cwd))
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
