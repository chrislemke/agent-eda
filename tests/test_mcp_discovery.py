"""Discovery over MCP, tested as an MCP client meets it.

A caller connecting for the first time reads the server's instructions: what
the package does, how to pass data, how to read results and statuses, how to
mask, and which first-look order to follow. The search_analyses and
list_analyses MCP tools return what the Python calls return.
"""

import inspect
import re

import pytest

pytest.importorskip(
    "mcp",
    reason="the MCP server tests need the mcp extra: pip install 'agent-eda[mcp]'",
)

import anyio
import jsonschema
from mcp import Client
from test_search_analyses import SEARCH_ACCEPTANCE

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError
from agent_eda.mcp import create_server

ANALYSES = [info.name for info in ae.list_analyses().analyses]
# Python call syntax, such as to_frame() or ae.guide(): a name and a bracket.
CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\(")
# The order for a first look at a new dataset, as ae.guide() shows it.
FIRST_LOOK = [
    "describe_dataset",
    "infer_column_semantics and validate_column_types",
    "profile_missing_values",
    "detect_missing_sentinels",
    "summarize_numeric and summarize_categorical",
    "find_duplicate_rows and find_candidate_keys",
    "profile_target",
    "compute_association_matrix",
]


@pytest.fixture(scope="module")
def instructions():
    async def scenario():
        async with Client(create_server(), mode="legacy") as client:
            return client.instructions

    return anyio.run(scenario)


def test_the_instructions_keep_within_one_page_of_the_guides_limits(instructions):
    lines = instructions.split("\n")

    assert 20 <= len(lines) <= 60
    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]


def test_the_instructions_hold_no_python_call_syntax(instructions):
    assert CALL.findall(instructions) == []
    assert "import " not in instructions
    assert "ae." not in instructions


def test_the_instructions_name_every_registered_analysis_with_its_summary(
    instructions,
):
    lines = instructions.split("\n")

    for info in ae.list_analyses().analyses:
        masking = " [supports masking]" if info.supports_masking else ""
        assert f"  {info.name}: {info.summary}{masking}" in lines


def test_the_instructions_say_what_the_package_does(instructions):
    first = instructions.split("\n")[0]

    assert first.startswith(f"agent-eda {ae.__version__}: deterministic ")
    assert "exploratory data analysis" in first


def test_the_instructions_explain_data_paths(instructions):
    text = unwrapped(instructions)

    assert "data_path" in text
    for extension in [".csv", ".tsv", ".parquet", ".pq", ".csv.gz", ".tsv.gz"]:
        assert extension in text
    assert "only an empty field counts as missing" in text
    assert "relative to the MCP server's working directory" in text
    assert "0-based data-row numbers" in text
    assert "id_column" in text


def test_the_instructions_explain_results_and_statuses(instructions):
    text = unwrapped(instructions)

    for status in [
        "completed",
        "partial",
        "insufficient_data",
        "not_applicable",
        "missing_prerequisites",
        "budget_exceeded",
        "blocked_by_policy",
        "failed",
    ]:
        assert status in text
    for assessment in ["descriptive", "suspected", "violation", "passed", "unknown"]:
        assert assessment in text
    # The text is the printed result, the structured content the record, and
    # table_path the complete table past the record's caps.
    assert "structured content" in text
    assert "_total" in text
    assert "table_path" in text
    assert "isError" in text


def test_the_instructions_explain_masking_as_only_a_guardrail(instructions):
    text = unwrapped(instructions)

    assert "project_path" in text
    assert "agent-eda-mcp --project" in text
    assert "mask_sensitive" in text
    assert (
        "Masking is a guardrail against accidental exposure, not anonymization or "
        "access control" in text
    )


def test_the_instructions_give_the_first_look_order(instructions):
    lines = instructions.split("\n")
    start = lines.index("A first look at a new dataset, in this order:") + 1

    steps = [line.split(": ")[0] for line in lines[start : start + len(FIRST_LOOK)]]

    assert steps == [
        f"  {number}. {names}" for number, names in enumerate(FIRST_LOOK, start=1)
    ]


def test_the_instructions_point_to_the_discovery_mcp_tools(instructions):
    assert "search_analyses" in instructions
    assert "list_analyses" in instructions


def test_list_analyses_returns_the_python_listing():
    response = anyio.run(call, "list_analyses", {})
    expected = ae.list_analyses()

    assert not response.is_error
    assert [block.text for block in response.content] == [str(expected)]
    assert response.structured_content == expected.to_dict()


def test_list_analyses_names_every_analysis_mcp_tool():
    response = anyio.run(call, "list_analyses", {})

    names = [info["name"] for info in response.structured_content["analyses"]]
    assert names == ANALYSES
    assert set(names) <= {tool.name for tool in anyio.run(list_tools)}


@pytest.mark.parametrize(
    ("query", "limit"),
    [("nulls per column", 5), ("nulls per column", 2), ("xylophone", 5), ("", 1)],
)
def test_search_analyses_returns_the_python_listing(query, limit):
    response = anyio.run(call, "search_analyses", {"query": query, "limit": limit})
    expected = ae.search_analyses(query, limit=limit)

    assert not response.is_error
    assert [block.text for block in response.content] == [str(expected)]
    assert response.structured_content == expected.to_dict()


def test_search_analyses_returns_five_matches_without_a_limit():
    response = anyio.run(call, "search_analyses", {"query": "column"})

    assert response.structured_content == ae.search_analyses("column").to_dict()
    assert response.structured_content["limit"] == 5
    assert len(response.structured_content["matches"]) == 5


@pytest.mark.parametrize(
    ("query", "expected"),
    SEARCH_ACCEPTANCE,
    ids=[query for query, _ in SEARCH_ACCEPTANCE],
)
def test_an_agents_wording_finds_the_right_mcp_tool_first(query, expected):
    response = anyio.run(call, "search_analyses", {"query": query})

    assert response.structured_content["matches"][0]["name"] == expected


def test_the_discovery_mcp_tools_declare_their_arguments_and_records():
    tools = {tool.name: tool for tool in anyio.run(list_tools)}
    search = tools["search_analyses"]
    listing = tools["list_analyses"]

    assert search.input_schema["required"] == ["query"]
    assert search.input_schema["properties"]["query"]["type"] == "string"
    limit = search.input_schema["properties"]["limit"]
    assert (limit["type"], limit["default"], limit["minimum"]) == ("integer", 5, 1)
    assert listing.input_schema["properties"] == {}
    for tool, record in [
        (search, ae.search_analyses("nulls per column").to_dict()),
        (listing, ae.list_analyses().to_dict()),
    ]:
        summary = (inspect.getdoc(getattr(ae, tool.name)) or "").split("\n")[0]
        assert tool.description.startswith(summary)
        assert CALL.findall(tool.description) == []
        validator = jsonschema.Draft202012Validator(tool.output_schema)
        validator.validate(record)
        assert not validator.is_valid({**record, "made_up_field": 1})
        # They read the registry and write nothing.
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False


def test_describe_analysis_and_guide_are_not_mcp_tools():
    names = {tool.name for tool in anyio.run(list_tools)}

    assert "describe_analysis" not in names
    assert "guide" not in names
    assert names == {*ANALYSES, "search_analyses", "list_analyses"}


def test_a_search_limit_below_one_is_the_python_error():
    with pytest.raises(InvalidArgumentError) as raised:
        ae.search_analyses("nulls", limit=0)

    response = anyio.run(call, "search_analyses", {"query": "nulls", "limit": 0})

    assert response.is_error
    assert response.structured_content == {
        "code": raised.value.code,
        "message": str(raised.value),
    }
    assert [block.text for block in response.content] == [str(raised.value)]


@pytest.mark.parametrize(
    ("name", "arguments", "named"),
    [
        ("search_analyses", {"query": "nulls", "limt": 2}, "'limt'"),
        ("search_analyses", {"limit": 2}, "query"),
        ("list_analyses", {"query": "nulls"}, "'query'"),
    ],
)
def test_an_unknown_or_missing_discovery_argument_is_an_invalid_call(
    name, arguments, named
):
    response = anyio.run(call, name, arguments)

    assert response.is_error
    assert response.structured_content["code"] == "invalid_call"
    assert named in response.structured_content["message"]


def unwrapped(text):
    """The text with its line breaks and indents as single spaces."""
    return " ".join(text.split())


async def list_tools():
    async with Client(create_server(), mode="legacy") as client:
        return (await client.list_tools()).tools


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
