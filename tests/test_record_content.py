"""What the JSON record holds: only values that vary with the data or the call,
bounded levels, compact support and capped parameters."""

import json
import typing

import numpy as np
import pandas as pd
import pytest

import agent_eda as ae


def frame_with_one_gap(width=30):
    data: dict[str, object] = {f"c{i}": np.arange(10.0) for i in range(width)}
    data["c7"] = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return pd.DataFrame(data)


def test_support_lists_only_the_columns_where_something_was_excluded():
    res = ae.summarize_numeric(frame_with_one_gap())

    record = json.loads(res.to_json())

    assert record["support"]["row_count"] == 10
    assert record["support"]["column_count"] == 30
    assert [c["column"] for c in record["support"]["columns"]] == ["c7"]
    assert record["support"]["columns_total"] == 1
    assert [c.column for c in res.support.columns] == ["c7"]


def test_support_of_a_clean_frame_lists_no_column():
    res = ae.summarize_numeric(frame_with_one_gap().drop(columns="c7"))

    assert res.support.columns == []
    assert res.support.column_count == 29


def test_the_support_field_description_states_the_rule_for_unlisted_columns():
    shared = {
        f.name: f
        for f in ae.describe_analysis("summarize_numeric").shared_result_fields
    }

    columns = {f.name: f.description for f in shared["support"].fields}["columns"]

    assert "not listed" in columns
    assert "every row" in columns


def test_nested_lists_in_parameters_are_capped_with_their_totals():
    df = pd.DataFrame({"c0": range(400), "c1": range(400)})

    res = ae.detect_missing_sentinels(
        df, sentinel_values={"c0": list(range(-300, 0)), "c1": [-1]}
    )
    parameters = json.loads(res.to_json())["parameters"]

    declared = parameters["sentinel_values"]
    assert declared["c0"] == list(range(-300, -200))
    assert declared["c0_total"] == 300
    assert declared["c1"] == [-1]
    assert "c1_total" not in declared
    assert parameters["sentinel_values_total"] == 2
    assert (
        len(
            typing.cast("dict[str, list[int]]", res.parameters["sentinel_values"])["c0"]
        )
        == 300
    )


def test_the_project_is_recorded_whole_without_a_total():
    project = ae.Project(
        data_policy=ae.DataPolicy(sensitive_columns=[f"s{i}" for i in range(150)])
    )

    res = ae.summarize_categorical(pd.DataFrame({"s0": ["a", "b"]}), project=project)
    parameters = json.loads(res.to_json())["parameters"]

    assert "project_total" not in parameters
    assert len(parameters["project"]["data_policy"]["sensitive_columns"]) == 150
    assert parameters["project"] == res.parameters["project"]


def sentinel_frame():
    values = [-999] * 25 + list(range(1, 76))
    return pd.DataFrame({"x": values, "k": [f"r{i}" for i in range(100)]})


def test_to_frame_gives_every_row_id_of_a_sentinel_candidate_the_record_caps():
    res = ae.detect_missing_sentinels(sentinel_frame())

    [candidate] = res.candidates
    row_ids = res.to_frame()["row_ids"].tolist()

    assert candidate.count == 25
    assert len(candidate.row_ids) == 10
    assert row_ids == [list(range(25))]
    assert json.loads(res.to_json())["candidates"][0]["row_ids_total"] == 25


def test_to_frame_row_ids_of_a_sentinel_candidate_use_id_column_and_masking():
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["k"]))

    res = ae.detect_missing_sentinels(
        sentinel_frame(), id_column="k", project=project, mask_sensitive=True
    )
    [candidate] = res.candidates
    [row_ids] = res.to_frame()["row_ids"].tolist()

    assert len(row_ids) == 25
    assert row_ids[:10] == candidate.row_ids
    assert row_ids[10:] == [f"<masked {i}>" for i in range(11, 26)]


def test_to_frame_gives_every_invalid_row_id_of_a_declared_column():
    df = pd.DataFrame(
        {"x": ["1"] * 5 + ["n/a"] * 25, "k": [f"r{i}" for i in range(30)]}
    )

    res = ae.validate_column_types(df, schema={"x": "integer"}, id_column="k")
    [ids] = res.to_frame()["invalid_row_ids"].tolist()

    assert len(res.columns["x"].invalid_row_ids) == 10
    assert ids == [f"r{i}" for i in range(5, 30)]


def test_to_frame_gives_every_row_id_of_an_unexpected_target_label():
    df = pd.DataFrame({"y": ["a"] * 10 + ["b"] * 5 + ["zz"] * 25, "k": range(40)})

    res = ae.profile_target(
        df,
        target_column="y",
        task="multiclass_classification",
        expected_labels=["a", "b"],
        id_column="k",
    )
    frame = res.to_frame()
    by_label = dict(zip(frame["label"], frame["row_ids"], strict=True))

    assert by_label["zz"] == list(range(15, 40))
    assert res.unexpected_labels is not None
    assert len(res.unexpected_labels[0].row_ids) == 10
    assert by_label["a"] is None
    assert by_label["b"] is None


@pytest.mark.parametrize(
    ("name", "arguments", "gone"),
    [
        ("summarize_numeric", {}, ["definitions"]),
        ("summarize_categorical", {}, ["share_denominator", "level_order"]),
        ("find_duplicate_rows", {}, ["missing_values_compare_equal"]),
        (
            "profile_target",
            {"target_column": "a", "task": "multiclass_classification"},
            ["level_order"],
        ),
    ],
)
def test_constant_prose_is_not_in_the_record(name, arguments, gone):
    df = pd.DataFrame({"a": ["x", "y", "x", "z"], "b": [1.0, 2.0, 3.0, 4.0]})

    res = getattr(ae, name)(df, **arguments)

    for field in gone:
        assert field not in res.to_dict()
        assert field not in res.to_json()
        assert not hasattr(res, field)


def test_the_meaning_of_the_removed_fields_lives_in_field_descriptions():
    duplicates = {
        f.name: f.description
        for f in ae.describe_analysis("find_duplicate_rows").result_fields
    }
    target = {
        f.name: f.description
        for f in ae.describe_analysis("profile_target").result_fields
    }

    assert (
        "Missing values of every kind equal each other"
        in duplicates["compared_columns"]
    )
    assert "Count descending, then label" in target["classes"]


def test_a_rebuilt_result_cannot_give_the_row_ids_it_never_held():
    df = sentinel_frame()
    res = ae.detect_missing_sentinels(df)

    rebuilt = type(res).model_validate(res.model_dump())

    with pytest.raises(ae.errors.AgentEDAError) as caught:
        rebuilt.to_frame()
    assert caught.value.code == "result_table_unavailable"
    assert res.to_frame()["row_ids"].tolist() == [list(range(25))]
    copied = res.model_copy()
    assert copied.to_frame()["row_ids"].tolist() == [list(range(25))]


def test_the_to_dict_docstring_says_what_the_record_keeps():
    text = ae.AnalysisResult.to_dict.__doc__ or ""

    for phrase in ("top_n levels", "point at", "position", "to_frame() has every one"):
        assert phrase in " ".join(text.split()), phrase
