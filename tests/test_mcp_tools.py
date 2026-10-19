"""Every analysis is an MCP tool, built from the registry and the docstring.

The contract loops over ae.list_analyses(): each analysis has an MCP tool with
its name, a description from its docstring, an input schema from its arguments
and an output schema its record validates against, and a call with valid
arguments returns what the Python call returns.
"""

import inspect
import json
import re
from collections.abc import Callable

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
import pandas as pd
from mcp_helpers import call, list_tools
from python_syntax import json_words, json_words_in, titles_in, without_titles
from recipes import recipe_for
from serving import serving

import agent_eda as ae
from agent_eda.results import AnalysisResult

ANALYSES = [info.name for info in ae.list_analyses().analyses]
# The MCP tools every server lists before the analyses.
DISCOVERY = ["search_analyses", "list_analyses"]
# The sections of an MCP tool's description, in order, after the summary line.
DESCRIBED = ["When to use:", "Returns:", "Statuses:", "See Also:"]
# Python call syntax, such as to_frame() or ae.guide(): a name and a bracket.
CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\(")
# What only a Python caller can call: the result's methods, res and ae.
PYTHON_ONLY = re.compile(r"\bto_(frame|dict|json)\(|\bres\.|\bae\.")
# A class built with Python call syntax, such as ColumnSpec(logical_type=...).
CLASS = re.compile(r"\b[A-Z]\w*\(")
# "server" without "MCP " before it, which GLOSSARY.md asks to avoid.
BARE_SERVER = re.compile(r"(?<!MCP )\bserver\b", flags=re.IGNORECASE)

# A small CSV of mixed dtypes: integers, text with a stand-in "N/A", floats
# with an empty field and -999, dates as text, booleans and class labels.
CSV = (
    "order_id,city,score,joined,member,plan\n"
    "1,Oslo,1.5,2024-01-03,true,basic\n"
    "2,Lyon,,2024-02-11,false,pro\n"
    "3,Oslo,2.5,2024-03-07,true,basic\n"
    "4,N/A,-999,2024-04-21,false,pro\n"
    "5,Lyon,4.0,,true,basic\n"
)


@pytest.fixture(scope="module")
def tools():
    """The MCP tools the server lists, by name."""
    return {tool.name: tool for tool in anyio.run(list_tools)}


def test_the_mcp_server_lists_one_mcp_tool_per_analysis(tools):
    # Next to the discovery MCP tools (tests/test_mcp_discovery.py).
    assert sorted(set(tools) - set(DISCOVERY)) == sorted(ANALYSES)


def test_the_schema_argument_of_validate_column_types_needs_at_least_one_column(tools):
    schema = tools["validate_column_types"].input_schema

    assert "schema" in schema["required"]
    assert schema["properties"]["schema"]["minProperties"] == 1


@pytest.fixture(params=ANALYSES)
def name(request):
    return request.param


def test_the_description_holds_the_docstring_sections_in_order(tools, name):
    description = tools[name].description
    described = ae.describe_analysis(name)

    assert description.startswith(described.summary + "\n")
    lines = description.splitlines()
    assert [line for line in lines if line in DESCRIBED] == DESCRIBED
    assert json_words(described.when_to_use) in " ".join(
        section(description, "When to use:")
    )
    returns = " ".join(section(description, "Returns:"))
    assert [
        c.code for c in described.finding_codes if f"`{c.code}`" not in returns
    ] == []
    statuses = section(description, "Statuses:")
    assert [
        s.status
        for s in described.statuses
        if not any(line.startswith(f"{s.status}: ") for line in statuses)
    ] == []
    # The names only: list_analyses and each tool's own description say what
    # each answers.
    see_also = section(description, "See Also:")
    assert see_also == [", ".join(docstring_entries(name, "See Also:"))]


def test_the_description_leaves_out_examples_and_python_calls(tools, name):
    description = tools[name].description

    assert "Examples:" not in description
    assert ">>>" not in description
    assert CALL.findall(description) == []


def test_the_input_schema_is_the_arguments_schema_with_a_required_data_path(
    tools, name
):
    schema = dict(tools[name].input_schema)
    properties = dict(schema.pop("properties"))
    data_path = properties.pop("data_path")
    # The server's own property, checked in the test below.
    properties.pop("table_path")
    # The docstrings' None, True and False are null, true and false over MCP.
    expected = without_titles(
        json_words_in(ae.describe_analysis(name).to_dict()["parameters_schema"])
    )
    # project becomes project_path (tests/test_mcp_project.py), and the
    # Project's definitions go with it.
    expected_properties = dict(expected["properties"])
    if expected_properties.pop("project", None) is not None:
        assert list(properties).index("project_path") == list(
            expected["properties"]
        ).index("project")
        properties.pop("project_path")
        mask_sensitive = dict(properties.pop("mask_sensitive"))
        expected_mask = dict(expected_properties.pop("mask_sensitive"))
        # The docstring's sentence on the guardrail sits in the instructions,
        # and the description says where an MCP caller's project comes from.
        guardrail = (
            "Masking is a guardrail against accidental exposure, not "
            "anonymization or access control."
        )
        docstring = expected_mask.pop("description").replace(guardrail, "").strip()
        assert mask_sensitive.pop("description") == (
            f"{docstring} Over MCP, the project is project_path."
        )
        assert mask_sensitive == expected_mask
    definitions = {
        key: value
        for key, value in expected.get("$defs", {}).items()
        if key not in ("Project", "DataPolicy")
    }

    assert data_path["type"] == "string"
    assert schema.pop("required")[0] == "data_path"
    assert schema.pop("$defs", {}) == definitions
    assert schema == {
        key: value
        for key, value in expected.items()
        if key not in ("required", "properties", "$defs")
    }
    assert properties == expected_properties
    assert [p for p in properties if not properties[p]["description"]] == []
    assert tools[name].input_schema["required"][1:] == expected.get("required", [])


def test_every_mcp_tool_takes_an_optional_table_path_with_the_mcp_server_text(tools):
    # The analysis MCP tools; the discovery ones write no table.
    analysis_tools = {name: tools[name] for name in ANALYSES}
    table_paths = [
        tool.input_schema["properties"]["table_path"]
        for tool in analysis_tools.values()
    ]

    assert [
        n
        for n, t in analysis_tools.items()
        if "table_path" in t.input_schema["required"]
    ] == []
    # The same server text on every MCP tool, not text from a docstring.
    assert all(table_path == table_paths[0] for table_path in table_paths)
    table_path = table_paths[0]
    assert table_path["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert table_path["default"] is None
    description = table_path["description"]
    # The rules on overwriting, the response line and the table_file record sit
    # in the instructions (tests/test_mcp_discovery.py).
    for fact in (".csv", ".parquet", ".pq", "main table"):
        assert fact in description, fact


def sentences(text):
    return re.split(r"(?<=\.) (?=[A-Z])", text)


def test_the_path_descriptions_are_one_sentence_each(tools):
    found = {
        (tool.name, name): prop["description"]
        for tool in tools.values()
        for name, prop in tool.input_schema["properties"].items()
        if name in ("data_path", "table_path", "project_path")
    }

    # data_path and table_path on the 11 analysis MCP tools, project_path on the
    # six that support masking.
    assert len(found) == 2 * len(ANALYSES) + len(
        [a for a in ae.list_analyses().analyses if a.supports_masking]
    )
    assert {k: v for k, v in found.items() if len(sentences(v)) != 1} == {}


def test_no_schema_of_an_mcp_tool_carries_a_pydantic_title(tools):
    schemas = [
        (tool.name, kind, schema)
        for tool in tools.values()
        for kind, schema in (
            ("input", tool.input_schema),
            ("output", tool.output_schema),
        )
    ]

    assert [(n, k) for n, k, s in schemas if s is None] == []
    assert {(n, k): titles_in(s) for n, k, s in schemas if titles_in(s)} == {}
    # Inside $defs too.
    assert any("$defs" in schema for _, _, schema in schemas)


def definition_size(tools):
    """The characters of the descriptions plus the input schemas of every MCP
    tool, the size the audit measured in October 2026 (100,159)."""
    return sum(
        len(tool.description or "") + len(json.dumps(tool.input_schema))
        for tool in tools.values()
    )


# The size of the descriptions plus input schemas once ticket 09 cut the
# repeated path rules, the pydantic titles and the docstring prose that
# repeated the field descriptions (see .scratch/agent-experience/issues/
# 09-smaller-mcp-tool-definitions.md): 30% under the 100,159 characters the
# audit measured in October 2026 is 70,111. Raise it on purpose, in the same
# change as the text that grows, never to make this test pass.
DEFINITION_SIZE = 69_730


def test_the_tool_definitions_are_30_percent_smaller_than_the_audit_measured(tools):
    assert definition_size(tools) <= 100_159 * 0.7


def test_the_tool_definitions_stay_within_ten_percent_of_their_recorded_size(tools):
    size = definition_size(tools)

    assert size <= DEFINITION_SIZE * 1.1, (
        f"the descriptions and input schemas of the MCP tools come to {size} "
        f"characters, more than 10% over the {DEFINITION_SIZE} recorded in "
        "DEFINITION_SIZE; shorten them, or raise DEFINITION_SIZE on purpose"
    )


def test_the_mcp_tools_say_mcp_server_as_the_glossary_does(tools):
    # GLOSSARY.md: "MCP server", never "server" alone, in what a caller reads.
    texts = [tool.description for tool in tools.values()] + [
        prop.get("description", "")
        for tool in tools.values()
        for prop in tool.input_schema["properties"].values()
    ]
    sentences = {s for text in texts for s in re.split(r"(?<=\.) ", text)}

    assert sorted(s for s in sentences if BARE_SERVER.search(s)) == []


def test_the_summarize_numeric_schema_lists_methods_and_the_trim_range(tools):
    properties = tools["summarize_numeric"].input_schema["properties"]

    # Hyndman and Fan's nine definitions, and numpy's four others.
    assert properties["quantile_method"]["enum"] == [
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]
    assert properties["trim_fraction"]["minimum"] == 0
    assert properties["trim_fraction"]["exclusiveMaximum"] == 0.5


def test_each_mcp_tool_may_write_a_file_but_never_destroys_or_reaches_out(tools, name):
    annotations = tools[name].annotations

    # table_path writes a file, never over an existing one, and nothing is
    # read from or sent to the network.
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is True
    assert annotations.open_world_hint is False


def test_each_mcp_tool_has_an_output_schema_its_record_validates_against(
    tmp_path, tools, name
):
    _, data = mixed_csv(tmp_path)
    record = getattr(ae, name)(data, **valid_arguments(name, data)).to_dict()

    validator = jsonschema.Draft202012Validator(tools[name].output_schema)
    validator.validate(record)
    # The schema holds the record's exact fields, not any object.
    assert not validator.is_valid({**record, "made_up_field": 1})


def test_no_mcp_schema_holds_a_name_that_a_rename_retired(tools, name):
    retired = {
        "declared_schema",
        "cardinality",
        "skipped_columns",
        "duplicate_row_count",
    }

    schemas = (tools[name].input_schema, tools[name].output_schema)

    assert {
        key for schema in schemas for key in property_names(schema)
    } & retired == set()


def test_a_call_with_valid_arguments_returns_the_python_record(tmp_path, name):
    data_path, data = mixed_csv(tmp_path)
    arguments = valid_arguments(name, data)
    expected = getattr(ae, name)(data, **arguments)

    response = anyio.run(call, name, {"data_path": str(data_path), **arguments})

    assert not response.is_error
    statuses = [s.status for s in ae.describe_analysis(name).statuses]
    assert response.structured_content["status"] in statuses
    assert response.structured_content == expected.to_dict()
    (text,) = [block.text for block in response.content]
    assert text.splitlines()[0] == str(expected).splitlines()[0]
    assert len(text.splitlines()) <= 40


def test_the_printed_result_over_mcp_holds_no_python(tmp_path, name):
    data_path, data = mixed_csv(tmp_path)
    arguments = valid_arguments(name, data)

    response = anyio.run(call, name, {"data_path": str(data_path), **arguments})

    (text,) = [block.text for block in response.content]
    assert CALL.findall(text) == []
    assert "ae." not in text
    assert "res." not in text


def test_the_input_schema_holds_no_python_example_or_call(tools, name):
    descriptions = list(schema_descriptions(tools[name].input_schema))

    assert [d for d in descriptions if ">>>" in d or "Examples:" in d] == []
    # Formulas such as int(trim_fraction * n) stay; building a class doesn't.
    assert [d for d in descriptions if PYTHON_ONLY.search(d) or CLASS.search(d)] == []


def test_the_output_schema_names_no_python_method_of_the_result(tools, name):
    descriptions = list(schema_descriptions(tools[name].output_schema))

    # Formulas such as sqrt(n) stay; calls only a Python caller can make go.
    assert [d for d in descriptions if PYTHON_ONLY.search(d)] == []


def test_a_parameter_without_a_json_schema_stops_the_mcp_server_from_starting():
    with pytest.raises(TypeError) as raised:
        serving(ae.describe_dataset, check_rule)

    assert str(raised.value) == (
        "check_rule can't be an MCP tool: its parameter rule has the type "
        "Callable[[float], bool], which has no JSON Schema, so an MCP client "
        "can't pass a value for it. Give rule a type that JSON can hold, such "
        "as str, a number, a list or a Literal of choices."
    )


def test_the_mcp_server_serves_the_analyses_it_is_given():
    tools = anyio.run(list_tools, serving(ae.describe_dataset))

    assert [tool.name for tool in tools] == [*DISCOVERY, "describe_dataset"]


def property_names(schema):
    """Every property name in a JSON Schema, however deep."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                yield from value
            yield from property_names(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from property_names(value)


def schema_descriptions(schema):
    """Every description in a JSON Schema, however deep."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "description" and isinstance(value, str):
                yield value
            else:
                yield from schema_descriptions(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from schema_descriptions(value)


def check_rule(data: pd.DataFrame, *, rule: Callable[[float], bool]) -> AnalysisResult:
    """Does every value pass the rule?

    A stand-in analysis that only this test serves, with a parameter that
    takes a Python function, which no MCP client can send.

    Args:
        rule (Callable[[float], bool]): The rule each value must pass.
    """
    raise AssertionError("the server must not start, so nothing calls it")


def mixed_csv(tmp_path):
    """The mixed-dtype CSV's path, and its DataFrame as the server reads it."""
    path = tmp_path / "orders.csv"
    path.write_text(CSV, encoding="utf-8")
    # Only an empty field counts as missing, so "N/A" and -999 are values.
    return path, pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "city": ["Oslo", "Lyon", "Oslo", "N/A", "Lyon"],
            "score": [1.5, None, 2.5, -999.0, 4.0],
            "joined": [
                "2024-01-03",
                "2024-02-11",
                "2024-03-07",
                "2024-04-21",
                float("nan"),
            ],
            "member": [True, False, True, False, True],
            "plan": ["basic", "pro", "basic", "pro", "basic"],
        }
    )


def valid_arguments(name, data):
    """The arguments a call needs, as JSON values an MCP client can send."""
    arguments = recipe_for(name).required(data)
    assert arguments is not None, f"the mixed CSV allows no call of {name}"
    return arguments


def section(description, header):
    """The lines of one section of a description, without the header."""
    lines = description.splitlines()
    start = lines.index(header) + 1
    following = [i for i, line in enumerate(lines) if line in DESCRIBED and i >= start]
    return [line for line in lines[start : following[0] if following else None] if line]


def docstring_entries(name, header):
    """The names a section of the analysis's docstring lists, in order."""
    doc = inspect.getdoc(getattr(ae, name)) or ""
    body = doc.split(f"\n{header}\n")[1]
    return re.findall(r"^    ([a-z_]+):", body, flags=re.MULTILINE)
