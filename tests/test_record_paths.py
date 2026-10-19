"""Records keep what findings and limitations point at: every field path they
name resolves in `to_dict()` and `to_json()`, and in the result object."""

import ast
import json

import numpy as np
import pandas as pd
import pytest
from recipes import recipe_for

import agent_eda as ae

ANALYSES = [info.name for info in ae.list_analyses().analyses]


def resolve(root, path):
    """The value a field path names, in a record or in a result object.

    Raises LookupError when the path doesn't resolve. A bracket holds a position,
    a key of a mapping, or for a list the values of the entry's key fields.
    """
    node = ast.parse(path, mode="eval").body
    steps = []
    while not isinstance(node, ast.Name):
        if isinstance(node, ast.Attribute):
            steps.append(("attribute", node.attr))
            node = node.value
        elif isinstance(node, ast.Subscript):
            steps.append(("item", ast.literal_eval(node.slice)))
            node = node.value
        else:
            raise LookupError(path)
    steps.append(("attribute", node.id))
    value = root
    for kind, name in reversed(steps):
        value = step(value, kind, name, path)
    return value


def step(value, kind, name, path):
    if kind == "attribute":
        if isinstance(value, dict):
            if name in value:
                return value[name]
        elif hasattr(value, name):
            return getattr(value, name)
        raise LookupError(path)
    if isinstance(value, dict):
        if name in value:
            return value[name]
        raise LookupError(path)
    if isinstance(name, int):
        if 0 <= name < len(value):
            return value[name]
        raise LookupError(path)
    names = name if isinstance(name, tuple) else (name,)
    for entry in value:
        if keys_of(entry)[: len(names)] == names:
            return entry
    raise LookupError(path)


def keys_of(entry):
    """The columns that name a list entry: a pair's two columns, or its column."""
    get = entry.get if isinstance(entry, dict) else lambda n: getattr(entry, n, None)
    return tuple(
        get(n) for n in ("x_column", "y_column", "column", "columns") if get(n)
    )


def paths(record):
    """Every field path the findings and limitations of a record or result name."""
    get = (
        (lambda item: item["fields"])
        if isinstance(record, dict)
        else (lambda item: item.fields)
    )
    return [
        (f"{kind}[{i}]", path)
        for kind in ("findings", "limitations")
        for i, item in enumerate(
            record[kind] if isinstance(record, dict) else getattr(record, kind)
        )
        for path in get(item)
    ]


def wide_with_constant(width=120, constant="c115"):
    """A frame whose constant column sits beyond the 100 entries a record keeps."""
    rng = np.random.default_rng(0)
    data = {f"c{i}": rng.normal(size=20) for i in range(width)}
    data[constant] = np.full(20, 7.0)
    return pd.DataFrame(data)


def test_a_finding_beyond_the_hundredth_column_resolves_in_the_record():
    res = ae.summarize_numeric(wide_with_constant())

    record = json.loads(res.to_json())

    [finding] = [f for f in record["findings"] if f["code"] == "constant_columns"]
    assert "columns['c115'].min" in finding["fields"]
    assert resolve(record, "columns['c115'].min") == 7.0
    assert len(record["columns"]) == 100
    assert record["columns_total"] == 120
    # The kept entries are the pointed one and the first others, in order.
    assert list(record["columns"])[:99] == [f"c{i}" for i in range(99)]
    assert list(record["columns"])[99] == "c115"


@pytest.mark.parametrize("name", ANALYSES)
def test_every_field_path_of_a_wide_frame_resolves_in_the_object_and_the_record(name):
    rng = np.random.default_rng(1)
    width = 130
    data = {}
    for i in range(width):
        if i % 3 == 0:
            data[f"c{i}"] = rng.normal(size=30)
        elif i % 3 == 1:
            data[f"c{i}"] = rng.choice(["a", "b", "c"], 30)
        else:
            data[f"c{i}"] = rng.integers(0, 4, 30)
    # Late columns that every analysis finds something in.
    data["c125"] = np.full(30, 1.0)
    data["c126"] = [None] * 30
    data["c127"] = [-999] * 10 + list(range(20))
    data["c128"] = ["Yes", "yes "] * 15
    data["c129"] = ["1", "x"] * 15
    df = pd.DataFrame(data)
    arguments = recipe_for(name).required(df)
    if arguments is None:
        pytest.skip(f"{name} takes no call on this frame")

    res = getattr(ae, name)(df, **arguments)
    record = json.loads(res.to_json())

    for _owner, path in paths(res):
        resolve(res, path)
    for owner, path in paths(record):
        try:
            resolve(record, path)
        except LookupError:
            raise AssertionError(
                f"{owner} names {path}, which the record lacks"
            ) from None


def test_association_pairs_are_named_by_their_two_columns_and_kept_when_pointed_at():
    rng = np.random.default_rng(2)
    data = {f"n{i:02d}": rng.normal(size=30) for i in range(25)}
    data["n24"] = np.full(30, 3.0)
    res = ae.compute_association_matrix(pd.DataFrame(data))

    record = json.loads(res.to_json())

    [limitation] = [m for m in record["limitations"] if m["code"] == "zero_variance"]
    assert "pairs['n00', 'n24'].value" in limitation["fields"]
    assert all(path.startswith("pairs['") for path in limitation["fields"])
    assert len(record["pairs"]) == 100
    assert record["pairs_total"] == 300
    assert resolve(record, "pairs['n00', 'n24'].value") is None
    assert resolve(res, "pairs['n00', 'n24'].value") is None
    assert len(limitation["fields"]) == 24


def test_a_positional_path_is_renumbered_to_the_entrys_place_in_the_record():
    df = pd.DataFrame({f"c{i}": [1.0, 2.0] for i in range(130)})
    df["c120"] = [None, None]

    res = ae.describe_dataset(df)
    record = json.loads(res.to_json())

    [finding] = [f for f in record["findings"] if f["code"] == "all_missing_columns"]
    [path] = finding["fields"]
    assert path == "columns[99].missing_count"
    assert resolve(record, "columns[99].column") == "c120"
    assert resolve(record, path) == 2
    assert resolve(res, "columns[120].column") == "c120"
    assert len(record["columns"]) == 100
