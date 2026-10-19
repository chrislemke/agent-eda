"""MCP arguments read as their input schema says.

The input schema of an MCP tool offers `date-time`, `date` and `duration`
strings for the values of sentinel_values, so an MCP caller can pass them. Each
test writes a Parquet file with datetime, duration and date columns, calls an MCP
tool through the SDK's in-memory client and compares the response with the same
analysis called in Python with pd.Timestamp, pd.Timedelta and datetime.date.
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

import datetime as dt
from typing import Any

import anyio
import pandas as pd
from mcp import Client

import agent_eda as ae
from agent_eda.mcp import create_server

DAY = dt.date(1900, 1, 1)

TIMES = pd.DataFrame(
    {
        "born": pd.to_datetime(["1900-01-01", "2000-01-02", "1900-01-01"]).as_unit(
            "us"
        ),
        "wait": pd.to_timedelta(["0s", "3h", "0s"]).as_unit("us"),
        "day": [DAY, dt.date(2020, 2, 2), DAY],
        "name": ["1900-01-01", "ann", "P0D"],
    }
)

# What a Python caller declares for each column, and what an MCP caller writes
# for the same value.
PYTHON: dict[str, list[Any]] = {
    "born": [pd.Timestamp("1900-01-01")],
    "wait": [pd.Timedelta(0)],
    "day": [DAY],
}
JSON_VALUES = [
    ("born", "1900-01-01"),
    ("born", "1900-01-01T00:00:00"),
    ("born", "1900-01-01T00:00:00.000000"),
    ("wait", "PT0S"),
    ("wait", "P0DT0H0M0S"),
    ("wait", "P0D"),
    ("day", "1900-01-01"),
]


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    path = tmp_path_factory.mktemp("times") / "times.parquet"
    TIMES.to_parquet(path)
    return path


@pytest.mark.parametrize("tool", ["profile_missing_values", "detect_missing_sentinels"])
@pytest.mark.parametrize(("column", "text"), JSON_VALUES)
def test_an_iso_string_declares_a_datetime_duration_or_date_sentinel(
    path, tool, column, text
):
    response = anyio.run(
        call, tool, {"data_path": str(path), "sentinel_values": {column: [text]}}
    )

    assert not response.is_error, response.structured_content
    expected = getattr(ae, tool)(TIMES, sentinel_values={column: PYTHON[column]})
    assert response.structured_content == expected.to_dict()


def test_iso_strings_are_the_expected_labels_of_a_target_of_dates(path):
    response = anyio.run(
        call,
        "profile_target",
        {
            "data_path": str(path),
            "target_column": "day",
            "task": "multiclass_classification",
            "expected_labels": ["1900-01-01", "2020-02-02", "2021-03-03"],
        },
    )

    assert not response.is_error, response.structured_content
    expected = ae.profile_target(
        TIMES,
        target_column="day",
        task="multiclass_classification",
        expected_labels=[DAY, dt.date(2020, 2, 2), dt.date(2021, 3, 3)],
    )
    assert response.structured_content == expected.to_dict()
    counts = [(c.label, c.count) for c in expected.classes or []]
    assert counts == [("1900-01-01", 2), ("2020-02-02", 1), ("2021-03-03", 0)]


def test_each_iso_string_gives_the_sentinel_count_of_the_python_call(path):
    declared = {column: [text] for column, text in JSON_VALUES[::3]}

    response = anyio.run(
        call,
        "profile_missing_values",
        {"data_path": str(path), "sentinel_values": declared},
    )

    expected = ae.profile_missing_values(TIMES, sentinel_values=PYTHON)
    counts = {
        name: column["sentinel_count"]
        for name, column in response.structured_content["columns"].items()
    }
    assert counts == {
        "born": 2,
        "wait": 2,
        "day": 2,
        "name": None,
    }
    assert counts == {
        name: column["sentinel_count"]
        for name, column in expected.to_dict()["columns"].items()
    }


def test_a_string_for_a_text_column_stays_text(path):
    response = anyio.run(
        call,
        "profile_missing_values",
        {"data_path": str(path), "sentinel_values": {"name": ["1900-01-01", "P0D"]}},
    )

    assert not response.is_error
    expected = ae.profile_missing_values(
        TIMES, sentinel_values={"name": ["1900-01-01", "P0D"]}
    )
    assert response.structured_content == expected.to_dict()
    assert response.structured_content["columns"]["name"]["sentinel_count"] == 2


def test_a_string_that_is_not_iso_gets_the_type_error_with_a_json_fix(path):
    response = anyio.run(
        call,
        "profile_missing_values",
        {"data_path": str(path), "sentinel_values": {"born": ["01/01/1900"]}},
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    message = response.structured_content["message"]
    assert "01/01/1900" in message
    assert "datetimes without a time zone" in message
    assert "pd." not in message
    assert "sentinel_values=" not in message


def test_a_time_zone_the_column_does_not_have_gets_an_error_with_a_json_fix(path):
    response = anyio.run(
        call,
        "profile_missing_values",
        {
            "data_path": str(path),
            "sentinel_values": {"born": ["1900-01-01T00:00:00Z"]},
        },
    )

    assert response.is_error
    assert response.structured_content["code"] == "invalid_argument"
    message = response.structured_content["message"]
    described = anyio.run(call, "describe_dataset", {"data_path": str(path)})
    dtype = described.structured_content["columns"][0]["dtype"]  # its unit varies
    assert message == (
        'profile_missing_values got "1900-01-01T00:00:00+00:00" for column '
        f"'born' in sentinel_values, but 'born' holds datetimes without a time "
        f"zone (dtype {dtype}), so it can never match. Did you mean "
        '"1900-01-01T00:00:00"? Declare each value as the type its column holds, '
        'such as {"sentinel_values": {"born": ["1900-01-01T00:00:00"]}}.'
    )
    assert "pd." not in message


def test_a_python_caller_declaring_a_string_for_a_datetime_column_gets_the_python_fix():
    with pytest.raises(ae.errors.InvalidArgumentError) as raised:
        ae.profile_missing_values(TIMES, sentinel_values={"born": ["1900-01-01"]})

    assert str(raised.value) == (
        "profile_missing_values got '1900-01-01' for column 'born' in "
        "sentinel_values, but 'born' holds datetimes without a time zone (dtype "
        "datetime64[us]), so it can never match. Did you mean "
        "pd.Timestamp('1900-01-01 00:00:00')? Declare each value as the type its "
        "column holds, such as sentinel_values={'born': "
        "[pd.Timestamp('1900-01-01 00:00:00')]}."
    )


ZONED = pd.DataFrame(
    {
        "utc": pd.to_datetime(["1900-01-01", "2000-01-02", "2099-12-31"], utc=True)
        .as_unit("us")
        .tz_convert("UTC"),
        "berlin": pd.to_datetime(["1900-01-01", "2000-01-02", "1970-01-01"])
        .as_unit("us")
        .tz_localize("Europe/Berlin"),
        "naive": pd.to_datetime(["1900-01-01", "2000-01-02", "2099-12-31"]).as_unit(
            "us"
        ),
        "wait": pd.to_timedelta(["0s", "3h", "0s"]).as_unit("us"),
        "day": [DAY, dt.date(2020, 2, 2), dt.date(1970, 1, 1)],
        "words": ["1900-01-01", "1900-01-01", "ann"],
    }
)


def test_a_value_detect_missing_sentinels_suggests_can_be_declared_as_shown(tmp_path):
    path = tmp_path / "zoned.parquet"
    ZONED.to_parquet(path)
    suggested = anyio.run(call, "detect_missing_sentinels", {"data_path": str(path)})
    candidates = suggested.structured_content["candidates"]
    assert {c["column"] for c in candidates} >= {"utc", "berlin", "naive", "day"}

    declared: dict[str, list] = {}
    for candidate in candidates:
        declared.setdefault(candidate["column"], []).append(candidate["value"])
    response = anyio.run(
        call,
        "detect_missing_sentinels",
        {"data_path": str(path), "sentinel_values": declared},
    )

    assert not response.is_error, response.structured_content
    found = [
        (c["column"], c["value"], c["count"])
        for c in response.structured_content["candidates"]
        if c["confirmation"] == "declared"
    ]
    assert sorted(found) == sorted(
        (c["column"], c["value"], c["count"]) for c in candidates
    )


# An example of each `format` a JSON Schema string can name, and the column of
# TIMES that holds such a value. A format without an example here fails the
# contract test below: add one that every MCP tool offering the format accepts.
EXAMPLES = {
    "date-time": ("born", "1900-01-01T00:00:00"),
    "date": ("day", "1900-01-01"),
    "duration": ("wait", "PT0S"),
}
NOTHING = object()
# A second example of a format, for the list of labels that needs at least two.
SECOND = {"date-time": "1900-01-02T00:00:00", "date": "1900-01-02"}
# The arguments a tool requires besides data_path, for the column of an example.
REQUIRED = {
    "profile_target": lambda column: {
        "target_column": column,
        "task": "multiclass_classification",
    }
}


def formats(schema, root):
    """Every `format` a JSON Schema offers, references followed."""
    if isinstance(schema, list):
        for item in schema:
            yield from formats(item, root)
    elif isinstance(schema, dict):
        if "format" in schema:
            yield schema["format"]
        if "$ref" in schema:
            yield from formats(root["$defs"][schema["$ref"].split("/")[-1]], root)
        for key, value in schema.items():
            if key not in ("$defs", "$ref"):
                yield from formats(value, root)


def example_of(form, schema, root):
    """A value the schema accepts that holds the format's example, or NOTHING
    when the schema offers no such format."""
    column, example = EXAMPLES[form]
    if "$ref" in schema:
        return example_of(form, root["$defs"][schema["$ref"].split("/")[-1]], root)
    if schema.get("format") == form:
        return example
    for branch in schema.get("anyOf", []):
        value = example_of(form, branch, root)
        if value is not NOTHING:
            return value
    inner = schema.get("additionalProperties")
    if isinstance(inner, dict):
        value = example_of(form, inner, root)
        return NOTHING if value is NOTHING else {column: value}
    if "items" in schema:
        value = example_of(form, schema["items"], root)
        return NOTHING if value is NOTHING else [value]
    for name, prop in schema.get("properties", {}).items():
        value = example_of(form, prop, root)
        if value is not NOTHING:
            return {name: value}
    return NOTHING


@pytest.fixture(scope="module")
def tools():
    async def listing():
        async with Client(create_server(), mode="legacy") as client:
            return (await client.list_tools()).tools

    return {tool.name: tool for tool in anyio.run(listing)}


def test_every_format_an_input_schema_offers_has_an_example(tools):
    offered = {
        form
        for tool in tools.values()
        for form in formats(tool.input_schema, tool.input_schema)
    }

    assert offered
    assert offered <= set(EXAMPLES)


def test_every_mcp_tool_accepts_an_example_of_each_format_its_input_schema_offers(
    tools, path
):
    sent = 0
    for name, tool in tools.items():
        schema = tool.input_schema
        for form in sorted(set(formats(schema, schema))):
            arguments = {"data_path": str(path)}
            if name in REQUIRED:
                arguments |= REQUIRED[name](EXAMPLES[form][0])
            for argument, prop in schema["properties"].items():
                value = example_of(form, prop, schema)
                if value is not NOTHING:
                    if argument == "expected_labels" and form in SECOND:
                        value = [*value, SECOND[form]]
                    arguments[argument] = value

            response = anyio.run(call, name, arguments)

            sent += 1
            problem = response.structured_content if response.is_error else None
            assert problem is None or problem["code"] not in (
                "invalid_argument",
                "invalid_call",
            ), f"{name} refuses an example of {form}: {arguments}: {problem}"
    assert sent


async def call(name, arguments):
    async with Client(create_server(), mode="legacy") as client:
        return await client.call_tool(name, arguments)
