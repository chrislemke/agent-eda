"""Identical calls give byte-identical records, text and tables in any process.

Python orders sets by hash, and hashing text is seeded per process, so anything
that writes a set's items in iteration order changes between runs. Memory
addresses and Python's cached UTF-8 copies change too.
"""

import json
import os
import pickle
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from strategies import data_frames, frames_with_sets

import agent_eda as ae

HASH_SEEDS = ("1", "2", "3", "4")
HERE = Path(__file__).parent


def run_python(code, *, env=None):
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def printed_under_each_hash_seed(code):
    """What the code prints in a process per PYTHONHASHSEED, by seed."""
    return {
        seed: run_python(code, env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in HASH_SEEDS
    }


LABEL_FRAME = """
import pandas as pd
import agent_eda as ae
colours = frozenset({"red", "blue", "green"})
shapes = frozenset({"circle", "square", "star", "oval"})
labels = [colours, shapes, (colours, 1), colours, shapes, colours, (shapes, 2)]
df = pd.DataFrame(
    {"label": pd.Series(labels, dtype=object), "x": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]}
)
"""

LABEL_CALLS = {
    "summarize_categorical": "ae.summarize_categorical(df, columns=['label'])",
    "profile_target": (
        "ae.profile_target(df, target_column='label', task='multiclass_classification')"
    ),
    "validate_column_types": (
        "ae.validate_column_types(df, schema={'label': 'integer'})"
    ),
}


@pytest.mark.parametrize("call", LABEL_CALLS)
def test_set_labels_are_written_the_same_in_every_process_with_items_sorted(call):
    code = (
        LABEL_FRAME
        + f"res = {LABEL_CALLS[call]}\n"
        + "print(res.to_json())\nprint(res)\n"
    )

    printed = printed_under_each_hash_seed(code)

    assert len(set(printed.values())) == 1
    text = printed["1"]
    assert "frozenset({'blue', 'green', 'red'})" in text
    assert "frozenset({'circle', 'oval', 'square', 'star'})" in text
    assert "(frozenset({'blue', 'green', 'red'}), 1)" in text


# frozenset({0, 8}) and frozenset({8, 0}) are equal, but Python iterates them in
# different orders, since both items want the same slot of its hash table.
FIRST = frozenset([0, 8, 16])
SECOND = frozenset([16, 8, 0])


def test_the_two_orders_of_one_set_are_equal_in_python_but_differ_in_iteration():
    assert FIRST == SECOND
    assert list(FIRST) != list(SECOND)


def test_equal_sets_in_different_iteration_orders_are_one_level_with_one_label():
    df = pd.DataFrame({"x": pd.Series([FIRST, SECOND, FIRST], dtype=object)})

    entry = ae.summarize_categorical(df).columns["x"]

    assert [(level.label, level.count) for level in entry.levels] == [
        ("frozenset({0, 16, 8})", 3)
    ]


def test_set_row_ids_are_one_sorted_list_in_the_record_and_one_form_in_the_text():
    df = pd.DataFrame(
        {"x": [1, 1, 2]},
        index=pd.Index([FIRST, SECOND, frozenset(["b", "a"])], dtype=object),
    )

    res = ae.find_duplicate_rows(df)

    assert json.loads(res.to_json())["duplicate_groups"][0]["row_ids"] == [
        [0, 16, 8],
        [0, 16, 8],
    ]
    assert "frozenset({0, 16, 8})" in str(res)


def test_sets_inside_tuples_are_sorted_too_in_labels_and_row_ids():
    df = pd.DataFrame({"x": pd.Series([(FIRST, "a"), (SECOND, "a")], dtype=object)})
    # Set afterwards: a Series would align on the new index.
    df.index = pd.Index([(FIRST, 1), (SECOND, 1)], tupleize_cols=False)

    res = ae.summarize_categorical(df)
    dup = ae.find_duplicate_rows(df, id_column=None)

    assert res.columns["x"].levels[0].label == "(frozenset({0, 16, 8}), 'a')"
    assert json.loads(dup.to_json())["duplicate_groups"][0]["row_ids"] == [
        [[0, 16, 8], 1],
        [[0, 16, 8], 1],
    ]


def test_items_of_a_set_that_share_their_json_are_still_ordered_the_same_way():
    # 1 and 'x' as text and as a number differ, but Decimal('1') and '1' write
    # the same JSON, so the type name decides.
    one = frozenset([Decimal(1), "1"])
    other = frozenset(["1", Decimal(1)])
    df = pd.DataFrame({"x": pd.Series([one, other], dtype=object)})

    entry = ae.summarize_categorical(df).columns["x"]

    assert [level.label for level in entry.levels] == ["frozenset({Decimal('1'), '1'})"]


class Plain:
    pass


@pytest.mark.parametrize("make", [object, Plain])
def test_an_object_whose_repr_has_an_address_is_written_by_its_type_name(make):
    one = make()
    df = pd.DataFrame({"x": pd.Series([one, one], dtype=object)})
    name = make.__name__

    res = ae.summarize_categorical(df)
    invalid = ae.validate_column_types(df, schema={"x": "integer"})

    labels = [level.label for level in res.columns["x"].levels]
    assert labels == [f"<{name} object>"]
    shown = [item.value for item in invalid.columns["x"].invalid_values]
    assert shown == [f"<{name} object>"]
    assert "0x" not in res.to_json() + str(res) + invalid.to_json() + str(invalid)


def test_an_object_inside_a_set_is_written_by_its_type_name_too():
    df = pd.DataFrame({"x": pd.Series([frozenset([Plain()])], dtype=object)})

    entry = ae.summarize_categorical(df).columns["x"]

    assert entry.levels[0].label == "frozenset({<Plain object>})"


@pytest.mark.parametrize(
    "second",
    [
        [0, 1, 2, 3],
        [0, 1, 2, 0],
        [2, 1, 0, 2],
    ],
    ids=["same values", "text only, one repeated", "text only, reordered"],
)
def test_memory_bytes_stays_the_same_when_a_second_column_holds_the_same_np_str_objects(
    second,
):
    values = [np.str_("é0"), np.str_("é1"), np.str_("é2"), None]
    df = pd.DataFrame({"a": pd.Series(values, dtype=object)})
    df["b"] = pd.Series([values[i] for i in second], dtype=object)

    first = ae.describe_dataset(df)
    again = ae.describe_dataset(df)

    assert first.memory_bytes == again.memory_bytes
    assert first.to_json() == again.to_json()


# The cross-process test. A worker process per hash seed runs every analysis on
# every frame and prints what it returned; the cases compare the prints.
WORKER = HERE / "determinism_worker.py"


@pytest.fixture(scope="module")
def worker_records(tmp_path_factory):
    """What every analysis returned on every frame, by hash seed."""
    generated = []

    @settings(max_examples=30, derandomize=True, database=None, deadline=None)
    @given(frames_with_sets() | data_frames())
    def collect(df):
        generated.append((f"generated {len(generated)}", df))

    collect()
    path = tmp_path_factory.mktemp("frames") / "frames.pickle"
    path.write_bytes(pickle.dumps(generated))
    records = {}
    for seed in ("1", "2"):
        completed = subprocess.run(
            [sys.executable, str(WORKER), str(path)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, completed.stderr
        records[seed] = json.loads(completed.stdout)
    return records


ANALYSES = [info.name for info in ae.list_analyses().analyses]


@pytest.mark.parametrize("name", ANALYSES)
def test_every_analysis_gives_identical_records_text_and_tables_across_processes(
    name, worker_records
):
    first, second = (worker_records[seed] for seed in ("1", "2"))
    cases = [case for case in first if case.split("|")[0] == name]

    assert cases, f"the worker ran no case of {name}"
    assert [c for c in cases if first[c] != second[c]] == []
    assert any(first[c].get("to_json") for c in cases), "every case was skipped"
