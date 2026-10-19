"""Discovery over MCP, tested as an MCP client meets it.

A caller connecting for the first time reads the server's instructions: what
the package does, how to pass data, how to read results and statuses, how to
mask, and which first-look order to follow. The search_analyses and
list_analyses MCP tools return what the Python calls return.
"""

import importlib
import inspect
import re
import typing

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
import jsonschema
from mcp import Client
from serving import serving, with_stand_ins
from test_search_analyses import SEARCH_ACCEPTANCE, accepted

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError
from agent_eda.mcp import create_server
from agent_eda.results import Status

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
    lines = instructions.splitlines()

    assert 20 <= len(lines) <= 60
    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]


def test_the_instructions_say_mcp_server_as_the_glossary_does(
    instructions,
):
    # GLOSSARY.md: "MCP server", never "server" alone.
    bare = re.compile(r"(?<!MCP )\bserver\b", flags=re.IGNORECASE)

    assert bare.findall(" ".join(instructions.split())) == []


def test_the_instructions_hold_no_python_call_syntax(instructions):
    assert CALL.findall(instructions) == []
    assert "import " not in instructions
    assert "ae." not in instructions


def test_the_instructions_give_each_area_with_its_count_and_no_module_headings(
    instructions,
):
    text = unwrapped(instructions)

    assert [line for line in instructions.splitlines() if "agent_eda" in line] == []
    for area, count in {"data": 8, "statistics": 3}.items():
        covers = inspect.getdoc(importlib.import_module(f"agent_eda.{area}"))
        assert covers
        noun = "analysis" if count == 1 else "analyses"
        assert f" {area} ({count} {noun}): {covers.splitlines()[0]}" in text
    # list_analyses lists each analysis with its summary.
    lines = instructions.splitlines()
    assert [name for name in ANALYSES if f"  {name}: " in lines] == []


def test_the_instructions_stay_the_same_length_with_sixty_analyses():
    def served_instructions():
        async def scenario():
            # The instructions cover every registered analysis, however few the
            # MCP server serves.
            async with Client(serving(ae.describe_dataset), mode="legacy") as client:
                return client.instructions

        return anyio.run(scenario)

    before = served_instructions().splitlines()

    with with_stand_ins(60 - len(ANALYSES)):
        after = served_instructions().splitlines()

    assert len(before) <= 60
    assert len(after) == len(before)
    assert max(len(line) for line in after) <= 120
    changed = [line for line in after if line not in before]
    assert changed[0] == "60 analyses in 2 areas:"
    assert all(" analyses): " in line for line in changed[1:])


def test_the_instructions_say_what_the_package_does(instructions):
    first = instructions.splitlines()[0]

    assert first.startswith(f"agent-eda {ae.__version__}: deterministic ")
    assert "exploratory data analysis" in first


def test_the_instructions_do_not_say_that_every_mcp_tool_runs_an_analysis(instructions):
    # list_analyses and search_analyses are MCP tools too: they find analyses.
    text = unwrapped(instructions)

    assert "Each MCP tool runs one analysis" not in text
    assert "Every MCP tool runs" not in text
    assert "Each analysis is an MCP tool" in text
    assert "search_analyses finds analyses" in text
    assert "list_analyses lists them" in text


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

    # Only the statuses some analysis can return, computed from the registry.
    reachable = {s for info in ae.list_analyses().analyses for s in info.statuses}
    assert reachable == {
        "completed",
        "partial",
        "insufficient_data",
        "not_applicable",
        "budget_exceeded",
    }
    for status in typing.get_args(Status):
        assert (status in text) == (status in reachable), status
    for assessment in ["descriptive", "suspected", "violation", "passed", "unknown"]:
        assert assessment in text
    # The text is the printed result, the structured content the record, and
    # table_path the complete table past the record's caps, never overwriting.
    assert "structured content" in text
    assert "_total" in text
    assert "table_path" in text
    assert "never overwrites" in text
    assert "table_file_exists" in text
    assert "isError" in text


def test_the_instructions_promise_no_part_of_the_response_over_the_other(
    instructions,
):
    # Some clients show the model only the structured content, others the text.
    text = unwrapped(instructions)

    assert "may show you its text, its structured content or both" in text
    assert "table_file" in text


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
    lines = instructions.splitlines()
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
def test_search_analyses_returns_the_python_listing_in_mcp_wording(query, limit):
    response = anyio.run(call, "search_analyses", {"query": query, "limit": limit})
    expected = ae.search_analyses(query, limit=limit)

    assert not response.is_error
    assert response.structured_content == expected.to_dict()
    (text,) = [block.text for block in response.content]
    assert text.splitlines()[0] == str(expected).splitlines()[0]
    assert CALL.findall(text) == []
    assert "ae." not in text


def test_search_analyses_points_to_mcp_tools_for_more_and_next():
    response = anyio.run(call, "search_analyses", {"query": "column", "limit": 2})

    assert response.content[0].text.splitlines()[-2:] == [
        "More: search_analyses with limit 11 shows all 11 matches.",
        "Next: the compute_association_matrix MCP tool's description and input schema "
        "say how to call it.",
    ]


def test_a_search_without_matches_points_to_list_analyses():
    response = anyio.run(call, "search_analyses", {"query": "xylophone"})

    assert response.content[0].text.splitlines()[-1] == (
        "Try other words, or call list_analyses for every analysis; the MCP "
        "server's instructions say where to start."
    )


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

    matches = response.structured_content["matches"]
    if expected is None:
        assert matches == []
    else:
        assert matches[0]["name"] in accepted(expected)


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


@pytest.mark.parametrize(
    ("arguments", "fixes"),
    [
        (
            {"query": 3},
            {
                "such as ae.search_analyses('missing values'),": (
                    'such as {"query": "missing values"},'
                ),
                "a value of type int": "a number",
            },
        ),
        (
            {"query": "nulls", "limit": 0},
            {
                "such as ae.search_analyses('missing values', limit=3).": (
                    'such as {"query": "missing values", "limit": 3}.'
                ),
                "an int of at least": "an integer of at least",
            },
        ),
    ],
    ids=["query-not-text", "limit-below-one"],
)
def test_a_bad_search_value_is_the_python_error_with_an_mcp_fix(arguments, fixes):
    with pytest.raises(InvalidArgumentError) as raised:
        ae.search_analyses(**arguments)

    response = anyio.run(call, "search_analyses", arguments)

    expected = str(raised.value)
    for python_fix, mcp_fix in fixes.items():
        assert python_fix in expected
        expected = expected.replace(python_fix, mcp_fix)
    assert response.is_error
    assert response.structured_content == {
        "code": raised.value.code,
        "message": expected,
    }
    assert [block.text for block in response.content] == [
        response.structured_content["message"]
    ]


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
