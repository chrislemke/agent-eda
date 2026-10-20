"""A restructuring changes no byte of any record, text, table, tool or message.

tests/baseline/pandas<major>.json holds what scripts/record_baseline.py recorded
before the restructuring: the SHA-256 of every analysis's record, printed text
and tables on the fixed and generated DataFrames, the discovery records, the MCP
tool definitions, and every argument-mistake message in full. This test records
them again and compares. Refresh the baseline only where a change is meant to
alter one of them: see the script's docstring.
"""

import importlib.util
import json
import os
import platform
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_baseline.py"
MCP_KEYS = ("tool:", "call:", "mcp:", "mcp fixes:")


def load_script():
    spec = importlib.util.spec_from_file_location("record_baseline", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_schemas(records):
    return {
        key: {part: text for part, text in value.items() if part != "json"}
        if key.startswith("describe_analysis(")
        else value
        for key, value in records.items()
    }


def test_every_record_text_table_tool_and_message_matches_the_baseline():
    script = load_script()
    path = script.baseline_path()
    if not path.exists():
        pytest.skip(f"no baseline for pandas {pd.__version__.split('.')[0]}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    fresh = script.collect()
    if not any(key.startswith(MCP_KEYS) for key in fresh):
        # Without the mcp extra the tool records and the MCP messages don't run,
        # and the pydantic of the oldest supported version words the JSON Schema
        # that describe_analysis records for a parameter differently.
        recorded = {k: v for k, v in recorded.items() if not k.startswith(MCP_KEYS)}
        recorded = without_schemas(recorded)
        fresh = without_schemas(fresh)

    found = script.differences(recorded, fresh)

    assert found == [], (
        "The restructuring changed a record, text, table, tool or message. If the "
        "change is meant, refresh the baseline with scripts/record_baseline.py. "
        f"{len(found)} differ. Every record and part that differs:\n"
        + "\n".join(line.split(": ")[0] for line in found)
        + "\nThe first of them in full:\n"
        + "\n".join(found[:10])
    )


def recorded_versions():
    res = ae.describe_dataset(pd.DataFrame({"a": [1]}))
    return json.loads(res.to_json())["execution"]["versions"]


def digests_of_a_record(script):
    data = pd.DataFrame({"a": [1.0, 2.5, None], "b": ["x", "y", "x"]})
    res = ae.describe_dataset(data)
    return {part: script._digest(text) for part, text in script._record(res).items()}


def test_a_digest_does_not_depend_on_the_versions_of_python_numpy_and_pandas(
    monkeypatch,
):
    script = load_script()
    here = digests_of_a_record(script)
    assert set(recorded_versions()) == {"agent_eda", "python", "numpy", "pandas"}

    monkeypatch.setattr(platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(np, "__version__", "0.0.1")
    monkeypatch.setattr(pd, "__version__", "0.0.2")
    elsewhere = digests_of_a_record(script)

    # The patches reach the record, and still no digest differs.
    assert recorded_versions() == {
        "agent_eda": recorded_versions()["agent_eda"],
        "python": "3.12.3",
        "numpy": "0.0.1",
        "pandas": "0.0.2",
    }
    assert elsewhere == here


def test_a_digest_changes_when_a_library_leaves_the_versions_of_a_record():
    script = load_script()
    record = '{"execution":{"versions":{"agent_eda":"0.1.0","python":"3.12.3"}}}'
    fewer = '{"execution":{"versions":{"agent_eda":"0.1.0"}}}'

    assert script._digest(record) != script._digest(fewer)


def test_only_the_versions_of_a_record_are_blanked():
    script = load_script()
    record = (
        '{"execution":{"versions":{"agent_eda":"0.1.0","python":"3.12.3"}},'
        '"note":"python 3.12.3","versions":{"type":{"a":"1"}}}'
    )

    assert script.without_environment(record) == (
        '{"execution":{"versions":{"agent_eda":"<version>","python":"<version>"}},'
        '"note":"python 3.12.3","versions":{"type":{"a":"1"}}}'
    )


def test_a_table_ends_its_lines_with_a_line_feed_on_every_operating_system(
    monkeypatch,
):
    script = load_script()
    monkeypatch.setattr(os, "linesep", "\r\n")

    text = script._table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    assert "\r" not in text
    assert text.startswith(",a,b\n0,1,x\n1,2,y\n")


@pytest.mark.parametrize(
    "message",
    [
        r"no file at C:\Users\run\Temp\t1\shop.csv, so",
        r"no file at C:\\Users\\run\\Temp\\t1\\shop.csv, so",
        "no file at C:/Users/run/Temp/t1/shop.csv, so",
    ],
)
def test_a_message_names_a_temporary_windows_directory_the_way_it_names_any(message):
    script = load_script()

    stable = script._stable(message, PureWindowsPath(r"C:\Users\run\Temp\t1"))

    assert stable == "no file at <tmp>/shop.csv, so"


def test_a_message_keeps_what_follows_the_temporary_directory_on_a_unix_system():
    script = load_script()
    message = r"Write <tmp>/project.toml\"." + " In /x/y, not in <tmp>/a/b.csv"

    assert script._stable(message, Path("/nothing/here")) == message


def test_the_numbers_of_an_association_record_are_compared_to_ten_digits_only():
    script = load_script()
    key = "compute_association_matrix | many rows | default"
    one = '{"value":1.0,"other":0.15461082092709807}'
    other = '{"value":0.9999999999999996,"other":0.15461082092709805}'
    moved = '{"value":0.999999,"other":0.15461082092709805}'

    assert script._rounded(key, one) == script._rounded(key, other)
    assert script._rounded(key, one) != script._rounded(key, moved)
    # Any other record is compared whole.
    assert script._rounded("describe_dataset | many rows | default", one) == one
