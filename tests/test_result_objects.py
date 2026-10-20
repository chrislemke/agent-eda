"""Result objects: they compare, round-trip through pydantic, serialize and print."""

import inspect
import typing

import numpy as np
import pandas as pd
import pytest
from recipes import recipe_for
from strategies import EDGE_FRAMES

import agent_eda as ae
from agent_eda.errors import AgentEDAError

ANALYSES = [info.name for info in ae.list_analyses().analyses]
RESULT_CLASSES = 11


def call(name, df=None):
    """One call of the analysis on a frame with a column of every kind."""
    df = EDGE_FRAMES["every kind"]() if df is None else df
    arguments = recipe_for(name).required(df)
    if arguments is None:
        pytest.skip(f"{name} takes no call on this frame")
    return getattr(ae, name)(df, **arguments)


def table_names(res):
    """The values of the `table` argument of to_frame, or None without one."""
    parameter = inspect.signature(res.to_frame).parameters.get("table")
    return None if parameter is None else typing.get_args(parameter.annotation)


def test_there_are_eleven_result_classes():
    assert len(ANALYSES) == RESULT_CLASSES


@pytest.mark.parametrize("name", ANALYSES)
def test_two_identical_calls_give_results_that_compare_equal(name):
    first, second = call(name), call(name)

    assert first == second
    assert (first != second) is False
    assert second in [first]
    assert [first].count(second) == 1


@pytest.mark.parametrize("name", ANALYSES)
def test_results_of_different_data_compare_unequal_without_raising(name):
    df = EDGE_FRAMES["every kind"]()
    first = call(name, df)
    second = call(name, df.iloc[:1])

    assert first != second
    assert (first == second) is False
    assert second not in [first]
    assert first != name
    assert first != None  # noqa: E711 # the comparison with None is what is tested


def test_results_of_different_analyses_compare_unequal():
    df = EDGE_FRAMES["every kind"]()

    assert ae.summarize_numeric(df) != ae.summarize_categorical(df)


def test_results_that_differ_in_one_row_id_compare_unequal():
    df = pd.DataFrame({"x": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
    other = df.set_axis([1, 2, 3, 5])

    assert ae.find_duplicate_rows(df) != ae.find_duplicate_rows(other)
    assert ae.find_duplicate_rows(df) == ae.find_duplicate_rows(df.copy())


@pytest.mark.parametrize("name", ANALYSES)
def test_a_round_trip_through_pydantic_keeps_every_table_or_says_it_cannot(name):
    res = call(name)
    rebuilt = type(res).model_validate(res.model_dump())

    for table in table_names(res) or [None]:
        kwargs = {} if table is None else {"table": table}
        expected = res.to_frame(**kwargs)
        try:
            frame = rebuilt.to_frame(**kwargs)
        except AgentEDAError as error:
            raised = error
        else:
            pd.testing.assert_frame_equal(frame, expected)
            continue
        assert raised.code == "result_table_unavailable"
        assert type(res).__name__ in str(raised)
        assert "model_validate" in str(raised)
        assert f"ae.{name}(" in str(raised)
    assert rebuilt == res


@pytest.mark.parametrize(
    "name", ["summarize_categorical", "profile_target", "validate_column_types"]
)
def test_a_copy_keeps_the_tables_a_rebuilt_result_loses(name):
    import copy

    res = call(name)

    for kept in (copy.deepcopy(res), res.model_copy()):
        for table in table_names(res) or [None]:
            kwargs = {} if table is None else {"table": table}
            pd.testing.assert_frame_equal(
                kept.to_frame(**kwargs), res.to_frame(**kwargs)
            )


def test_a_rebuilt_result_with_no_levels_gives_the_same_empty_table():
    res = ae.summarize_categorical(pd.DataFrame({"x": [1.5, 2.5]}).iloc[:0])
    rebuilt = type(res).model_validate(res.model_dump())

    pd.testing.assert_frame_equal(rebuilt.to_frame(), res.to_frame())


def test_a_rebuilt_result_that_lost_its_levels_says_how_to_get_them_back():
    res = ae.summarize_categorical(pd.DataFrame({"x": ["a", "b", "a"]}))
    rebuilt = type(res).model_validate(res.model_dump())

    with pytest.raises(AgentEDAError) as caught:
        rebuilt.to_frame()

    assert caught.value.code == "result_table_unavailable"
    assert str(caught.value) == (
        "SummarizeCategoricalResult.to_frame() needs the full tables that only a "
        "call of the analysis keeps, but this result was rebuilt from its record, "
        "such as by model_validate(res.model_dump()), which doesn't hold them. "
        "Call ae.summarize_categorical(...) again, or copy the result with "
        "copy.deepcopy(res) or res.model_copy(), which keep the tables."
    )


def test_a_timedelta_in_years_as_a_row_id_gives_json():
    import json

    odd = np.timedelta64(5000, "Y")
    index = pd.Index([odd, np.timedelta64(1, "Y"), odd], dtype=object)
    by_index = pd.DataFrame({"x": [1, 1, 2]}, index=index)
    by_column = pd.DataFrame({"id": index, "x": [1, 1, 2]})

    for res in (
        ae.find_duplicate_rows(by_index),
        ae.find_duplicate_rows(by_column, id_column="id"),
    ):
        record = json.loads(res.to_json())

        assert record["duplicate_groups"][0]["row_ids"] == ["5000 years", "1 years"]


@pytest.mark.parametrize("name", ["describe_dataset", "summarize_numeric"])
def test_a_label_with_a_lone_surrogate_prints_and_encodes_as_utf8(name):
    label = "caf\udce9"
    # pandas 3 can't hold a lone surrogate in a text label, an object one can.
    df = pd.DataFrame(
        {0: [1.0, 2.0, 3.0], 1: [1.0, 2.0, 2.0]},
    ).set_axis(pd.Index([label, "ok"], dtype=object), axis=1)

    res = getattr(ae, name)(df)
    text = str(res)
    record = res.to_json()

    text.encode("utf-8")
    record.encode("utf-8")
    assert "\udce9" not in text
    assert "caf\\udce9" in text
    assert "\udce9" not in record
    assert "caf\\udce9" in record


def test_a_lone_surrogate_in_a_record_reads_back_as_the_label():
    import json

    label = "caf\udce9"
    df = pd.DataFrame({0: [1.0, 2.0]}).set_axis(pd.Index([label], dtype=object), axis=1)
    res = ae.describe_dataset(df)

    record = json.loads(res.to_json())

    assert record["columns"][0]["column"] == label


def test_a_right_to_left_override_in_a_label_is_escaped_in_the_text():
    label = "total‮12"
    df = pd.DataFrame({label: [1.0, 2.0, 3.0], "ok": [1.0, 2.0, 2.0]})

    for res in (ae.describe_dataset(df), ae.summarize_numeric(df)):
        text = str(res)

        assert "‮" not in text
        assert "total\\u202e12" in text


def test_a_label_with_a_newline_still_prints_on_one_line():
    df = pd.DataFrame({"a\nb": [1.0, 2.0, 3.0]})

    assert "a\\nb" in str(ae.describe_dataset(df))


def test_an_emoji_label_prints_as_it_is():
    df = pd.DataFrame({"café ✓": [1.0, 2.0, 3.0]})

    assert "café ✓" in str(ae.describe_dataset(df))
