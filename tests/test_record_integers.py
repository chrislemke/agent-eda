"""Integers beyond +-2**53 are strings in every JSON record, so that Node-based
MCP clients, which read JSON numbers as float64, don't round them. The result
objects keep the exact Python ints."""

import json
import shutil
import subprocess

import pandas as pd
import pytest

import agent_eda as ae
from agent_eda.results import FindDuplicateRowsResult

BIG = 1234567890123456789
LARGEST_EXACT = 9_007_199_254_740_991
NODE_READS_FIRST_ID = (
    "const r = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
    "console.log(r.duplicate_groups[0].row_ids[0]);"
)


def duplicates_by_order_id(order_ids):
    data = pd.DataFrame({"order_id": order_ids, "city": ["Oslo"] * len(order_ids)})
    return ae.find_duplicate_rows(data, columns=["city"], id_column="order_id")


def test_a_row_id_beyond_two_to_the_53_is_a_string_in_the_record():
    res = duplicates_by_order_id([BIG, BIG + 1])

    assert res.to_dict()["duplicate_groups"][0]["row_ids"] == [str(BIG), str(BIG + 1)]
    assert json.loads(res.to_json())["duplicate_groups"][0]["row_ids"] == [
        str(BIG),
        str(BIG + 1),
    ]
    assert f'"{BIG}"' in res.to_json()


def test_the_result_object_keeps_the_exact_python_int():
    res = duplicates_by_order_id([BIG, BIG + 1])

    assert res.duplicate_groups[0].row_ids == [BIG, BIG + 1]
    assert all(type(i) is int for i in res.duplicate_groups[0].row_ids)


def test_integers_within_two_to_the_53_stay_numbers_and_the_next_ones_do_not():
    ids = [-LARGEST_EXACT, LARGEST_EXACT, -LARGEST_EXACT - 1, LARGEST_EXACT + 1]

    record = duplicates_by_order_id(ids).to_dict()

    assert record["duplicate_groups"][0]["row_ids"] == [
        -LARGEST_EXACT,
        LARGEST_EXACT,
        str(-LARGEST_EXACT - 1),
        str(LARGEST_EXACT + 1),
    ]


def test_a_label_beyond_two_to_the_53_is_a_string_and_the_level_keeps_the_int():
    data = pd.DataFrame({"code": [BIG, BIG, 7]})

    res = ae.summarize_categorical(data, columns=["code"])

    labels = [level["label"] for level in res.to_dict()["columns"]["code"]["levels"]]
    assert labels == [str(BIG), 7]
    assert res.columns["code"].levels[0].label == BIG


def test_a_declared_value_beyond_two_to_the_53_is_a_string_in_the_parameters():
    data = pd.DataFrame({"x": [BIG, 1, 2, BIG]})

    res = ae.detect_missing_sentinels(data, sentinel_values={"x": [BIG]})

    assert res.to_dict()["parameters"]["sentinel_values"] == {"x": [str(BIG)]}
    assert f'"x":["{BIG}"]' in res.to_json()
    assert res.parameters["sentinel_values"] == {"x": [BIG]}


def test_a_boolean_is_not_taken_for_an_integer():
    data = pd.DataFrame({"flag": [True, True, False]})

    record = ae.summarize_categorical(data).to_dict()

    assert record["columns"]["flag"]["levels"][0]["label"] is True


def test_a_discovery_record_follows_the_rule_too():
    listing = ae.list_analyses()

    assert json.loads(listing.to_json()) == listing.to_dict()


def test_the_field_descriptions_say_big_integers_are_strings():
    description = ae.describe_analysis("find_duplicate_rows")
    shared = {f.name: f for f in description.shared_result_fields}
    groups = {f.name: f for f in description.result_fields}["duplicate_groups"]
    row_ids = {f.name: f for f in groups.fields}["row_ids"]
    levels = {
        f.name: f for f in ae.describe_analysis("summarize_categorical").result_fields
    }
    label = {
        f.name: f
        for f in {f.name: f for f in levels["columns"].fields}["levels"].fields
    }["label"]

    for field in (shared["parameters"], row_ids, label):
        assert "2^53" in field.description, field.name
        assert "string" in field.description, field.name


def test_the_output_schema_says_big_integers_are_strings():
    schema = FindDuplicateRowsResult.record_schema()

    assert "2^53" in schema["description"]
    assert "string" in schema["description"]


def test_a_record_with_large_integers_validates_against_the_output_schema():
    jsonschema = pytest.importorskip("jsonschema")
    data = pd.DataFrame({"code": [BIG, BIG, 7], "x": [1, 2, 3]})
    results = [
        duplicates_by_order_id([BIG, BIG + 1]),
        ae.summarize_categorical(data, columns=["code"]),
        ae.detect_missing_sentinels(data, sentinel_values={"code": [BIG]}),
    ]

    for res in results:
        jsonschema.validate(res.to_dict(), type(res).record_schema())


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js isn't installed")
def test_node_reads_the_exact_id_from_the_record():
    res = duplicates_by_order_id([BIG, BIG + 1])

    read = subprocess.run(
        ["node", "-e", NODE_READS_FIRST_ID],
        input=res.to_json(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert read == str(BIG)


def test_an_mcp_response_holds_the_large_id_as_a_string_in_its_bytes(tmp_path):
    pytest.importorskip("mcp", reason="needs the mcp extra")
    import anyio
    from mcp import Client

    from agent_eda.mcp import create_server

    path = tmp_path / "orders.csv"
    path.write_text(f"order_id,city\n{BIG},Oslo\n{BIG + 1},Oslo\n", encoding="utf-8")

    async def call():
        async with Client(create_server(), mode="legacy") as client:
            return await client.call_tool(
                "find_duplicate_rows",
                {"data_path": str(path), "columns": ["city"], "id_column": "order_id"},
            )

    response = anyio.run(call)

    assert not response.is_error
    sent = json.dumps(response.structured_content, separators=(",", ":"))
    assert f'"row_ids":["{BIG}","{BIG + 1}"]' in sent
