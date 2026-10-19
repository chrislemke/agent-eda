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
from pathlib import Path

import pandas as pd
import pytest

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
        "change is meant, refresh the baseline with scripts/record_baseline.py."
    )
