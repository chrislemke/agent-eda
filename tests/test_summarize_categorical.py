import os
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from recipes import has_countable_values, names
from strategies import STR, data_frames

import agent_eda as ae
from agent_eda.errors import ColumnTypeError, InvalidArgumentError


def mixed_frame():
    return pd.DataFrame(
        {
            "flag": [True, False, True],
            "n": [1.5, 2.5, 3.5],
            "city": pd.Series(["Lyon", "Oslo", "Lyon"], dtype=object),
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "grade": pd.Categorical(["a", "b", "a"]),
            "name": pd.Series(["x", "y", None], dtype="string"),
            "code": pd.array([1, None, 3], dtype="Int64"),
            "answer": pd.Series(["yes", "no", None], dtype="string[pyarrow]"),
            "member": pd.array([True, None, False], dtype="boolean"),
        }
    )


def test_default_selects_text_object_categorical_and_boolean_columns_in_order():
    res = ae.summarize_categorical(mixed_frame())

    selected = ["flag", "city", "grade", "name", "answer", "member"]
    assert list(res.columns) == selected
    assert res.parameters["columns"] == selected
    assert res.left_out_columns == ["n", "when", "code"]


ACCEPTED = (
    "It accepts text, object, categorical, boolean and numeric columns, nullable "
    "and pyarrow-backed ones included."
)


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
        pd.to_timedelta(["1D", None, "3D"]),
        pd.Series([[1], None, [2]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))),
    ],
    ids=["datetime", "timedelta", "arrow-list"],
)
def test_a_column_it_cannot_count_raises_with_its_dtype_and_a_fix(values):
    df = pd.DataFrame({"x": values, "plan": ["a", "b", "a"]})

    with pytest.raises(ColumnTypeError) as caught:
        ae.summarize_categorical(df, columns=["plan", "x"])

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        f"summarize_categorical can't use column 'x' with dtype {df['x'].dtype}. "
        f"{ACCEPTED} Count its values as text with df['x'].astype(str), or leave "
        "'x' out of columns."
    )


@pytest.mark.parametrize("top_n", [0, 101, -1])
def test_a_top_n_outside_1_to_100_raises_with_the_range(top_n):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_categorical(letters(), top_n=top_n)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"summarize_categorical got {top_n} for top_n, but top_n must be an integer with "
        "1 <= top_n <= 100. See help(ae.summarize_categorical) for every argument."
    )


def levels_of(entry):
    return [(level.label, level.count) for level in entry.levels]


def test_a_numeric_column_named_in_columns_is_summarized_by_its_values():
    df = pd.DataFrame(
        {
            "rating": [3, 1, 3, 2, 3],
            "score": [0.5, None, 0.5, 1.0, 0.5],
            "code": pd.array([7, 7, None, 8, 7], dtype="Int64"),
        }
    )

    res = ae.summarize_categorical(df, columns=["score", "rating", "code"])

    assert list(res.columns) == ["score", "rating", "code"]
    assert levels_of(res.columns["score"]) == [(0.5, 3), (1.0, 1)]
    assert levels_of(res.columns["rating"]) == [(3, 3), (1, 1), (2, 1)]
    assert levels_of(res.columns["code"]) == [(7, 3), (8, 1)]
    assert res.left_out_columns == []


def test_counts_missing_and_non_missing_values_and_the_distinct_levels():
    df = pd.DataFrame(
        {
            "plan": pd.Series(
                ["basic", "pro", None, "basic", np.nan, "team"], dtype=object
            )
        }
    )

    entry = ae.summarize_categorical(df).columns["plan"]

    assert (entry.non_missing_count, entry.missing_count) == (4, 2)
    assert entry.cardinality == 3


def test_shares_divide_by_the_non_missing_count_which_the_result_states():
    plans = ["basic", "pro", None, "basic", None, "basic", "team", None]
    df = pd.DataFrame({"plan": pd.Series(plans, dtype=object)})

    res = ae.summarize_categorical(df)

    assert [
        (lvl.label, lvl.count, lvl.share) for lvl in res.columns["plan"].levels
    ] == [
        ("basic", 3, 0.6),
        ("pro", 1, 0.2),
        ("team", 1, 0.2),
    ]
    assert res.share_denominator == "non_missing_count"


def letters():
    """13 levels: 'a' 3 times, 'b' twice and 'c' to 'm' once each."""
    return pd.DataFrame({"letter": list("aaabbcdefghijklm")})


def test_most_common_levels_show_the_top_10_and_count_the_levels_left_out():
    res = ae.summarize_categorical(letters())

    entry = res.columns["letter"]
    assert [level.label for level in entry.most_common_levels] == list("abcdefghij")
    assert entry.left_out_level_count == 3
    assert res.parameters["top_n"] == 10


def test_top_n_limits_only_the_most_common_levels():
    res = ae.summarize_categorical(letters(), top_n=3)

    entry = res.columns["letter"]
    assert [level.label for level in entry.most_common_levels] == ["a", "b", "c"]
    assert entry.left_out_level_count == 10
    assert len(entry.levels) == entry.cardinality == 13
    assert res.parameters["top_n"] == 3


def test_counts_the_levels_seen_once():
    entry = ae.summarize_categorical(letters()).columns["letter"]

    # 'c' to 'm' occur once each; 'a' and 'b' more often.
    assert entry.seen_once_count == 11


def test_lists_the_categories_that_never_occur_in_category_order():
    grade = pd.Categorical(["b", None, "a", "b"], categories=["c", "a", "b", "z"])
    df = pd.DataFrame({"grade": grade, "plan": ["x", "y", "x", "y"]})

    res = ae.summarize_categorical(df)

    assert res.columns["grade"].unused_categories == ["c", "z"]
    assert levels_of(res.columns["grade"]) == [("b", 2), ("a", 1)]
    assert res.columns["grade"].cardinality == 2
    assert res.columns["plan"].unused_categories == []


def test_tells_whether_a_column_mixes_values_of_different_python_types():
    df = pd.DataFrame(
        {
            "code": pd.Series([1, "1", 2.5, None, "a"], dtype=object),
            # pandas counts these as one level, 1, but they are three types.
            "merged": pd.Series([1, True, 1.0, None, 1], dtype=object),
            "numpy": pd.Series([np.int64(1), 2, None, 3, np.int32(4)], dtype=object),
            "plain": pd.Series(["a", "b", None, "a", "c"], dtype=object),
            "flag": [True, False, True, True, False],
            "grade": pd.Categorical([1, 2, 1, None, 2]),
        }
    )

    res = ae.summarize_categorical(df, columns=list(df.columns))

    found = {
        name: (entry.value_types, entry.mixed_value_types)
        for name, entry in res.columns.items()
    }
    assert found == {
        "code": (["float", "int", "str"], True),
        "merged": (["bool", "float", "int"], True),
        # A NumPy scalar counts as the Python type it holds.
        "numpy": (["int"], False),
        "plain": (["str"], False),
        "flag": (["bool"], False),
        "grade": (["int"], False),
    }
    assert levels_of(res.columns["merged"]) == [(1, 4)]


def test_orders_levels_by_count_descending_then_by_label():
    # pandas' value_counts would put 'b' before 'a', in order of appearance.
    df = pd.DataFrame({"plan": ["d", "b", "a", "c", "a", "b", "b", "c", "a"]})

    entry = ae.summarize_categorical(df).columns["plan"]

    assert levels_of(entry) == [("a", 3), ("b", 3), ("c", 2), ("d", 1)]


def test_keeps_labels_exactly_as_stored():
    answers = ["Yes", "yes ", "YES", " Yes", "Yes", "yes"]

    entry = ae.summarize_categorical(pd.DataFrame({"answer": answers})).columns[
        "answer"
    ]

    assert levels_of(entry) == [
        ("Yes", 2),
        (" Yes", 1),
        ("YES", 1),
        ("yes", 1),
        ("yes ", 1),
    ]


def test_orders_labels_of_mixed_types_numbers_first_then_text_then_the_rest():
    values = ["b", 2, True, b"x", "a", 1.5, (1, 2), Decimal("0.5"), float("nan")]
    df = pd.DataFrame({"mixed": pd.Series(values, dtype=object)})

    entry = ae.summarize_categorical(df).columns["mixed"]

    # Equal counts: numbers and booleans by value, then text by code point, then
    # other types by type name and string form. NaN in an object column is
    # missing, so it is no level.
    assert [(level.label, level.label_type) for level in entry.levels] == [
        ("0.5", "Decimal"),
        (True, "bool"),
        (1.5, "float"),
        (2, "int"),
        ("a", "str"),
        ("b", "str"),
        ("b'x'", "bytes"),
        ("(1, 2)", "tuple"),
    ]


def many_levels():
    """250 levels occurring 1, 2 or 3 times, so ties cross the 100th level."""
    codes = [f"v{i:03d}" for i in range(250) for _ in range(1 + i % 3)]
    shuffled = np.random.default_rng(0).permutation(len(codes))
    return pd.DataFrame({"code": [codes[i] for i in shuffled]})


def test_the_record_keeps_the_first_100_levels_and_their_total():
    res = ae.summarize_categorical(many_levels())

    entry = res.to_dict()["columns"]["code"]
    assert len(entry["levels"]) == 100
    assert entry["levels_total"] == 250
    assert entry["cardinality"] == 250
    assert entry["levels"] == [
        level.model_dump(mode="json") for level in res.columns["code"].levels
    ]


def test_to_frame_has_every_level_of_every_column_in_level_order():
    df = many_levels()
    df["flag"] = [index % 2 == 0 for index in range(len(df))]

    res = ae.summarize_categorical(df)
    frame = res.to_frame()

    assert list(frame.columns) == ["column", "label", "label_type", "count", "share"]
    assert frame["column"].tolist() == ["code"] * 250 + ["flag"] * 2
    codes = frame[frame["column"] == "code"]
    expected = (
        df["code"]
        .value_counts()
        .rename_axis("label")
        .reset_index()
        .sort_values(["count", "label"], ascending=[False, True])
    )
    assert codes["label"].tolist() == expected["label"].tolist()
    assert codes["count"].tolist() == expected["count"].tolist()
    assert codes["share"].tolist() == (expected["count"] / len(df)).tolist()
    # Its first 100 rows are the levels the entry keeps.
    assert [(lvl.label, lvl.count) for lvl in res.columns["code"].levels] == list(
        zip(codes["label"][:100], codes["count"][:100], strict=True)
    )
    assert frame["count"].dtype == "int64"
    assert frame["share"].dtype == "float64"


def tied_values(make):
    """250 values made from 0 to 249, occurring 1, 2 or 3 times, shuffled."""
    values = [make(i) for i in range(250) for _ in range(1 + i % 3)]
    return [values[i] for i in np.random.default_rng(1).permutation(len(values))]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(tied_values(lambda i: (i * 37 % 250) / 4)),
        pd.Series(tied_values(lambda i: i * 37 % 250), dtype="Int64"),
        pd.Series(pd.Categorical(tied_values(lambda i: f"k{i * 37 % 250}"))),
        pd.Series(tied_values(lambda i: i if i % 2 else f"s{i}"), dtype=object),
    ],
    ids=["floats", "nullable-integers", "categorical", "mixed-objects"],
)
def test_the_first_100_levels_are_the_first_100_rows_of_to_frame(values):
    res = ae.summarize_categorical(pd.DataFrame({"x": values}), columns=["x"])

    frame = res.to_frame()
    entry = res.columns["x"]
    assert len(frame) == entry.cardinality == 250
    assert [(level.label, level.count) for level in entry.levels] == list(
        zip(frame["label"][:100], frame["count"][:100], strict=True)
    )
    # Count descending, then label: within each count, labels ascend.
    for count, labels in frame.groupby("count", sort=False)["label"]:
        keys = [(isinstance(label, str), label) for label in labels]
        assert keys == sorted(keys), count


def test_to_frame_keeps_each_label_as_stored():
    values = [Decimal("0.5"), True, 2, "a", b"x", (1, 2), None]
    df = pd.DataFrame({"mixed": pd.Series(values, dtype=object)})

    frame = ae.summarize_categorical(df).to_frame()

    assert frame["label"].tolist() == [Decimal("0.5"), True, 2, "a", b"x", (1, 2)]
    assert frame["label_type"].tolist() == [
        "Decimal",
        "bool",
        "int",
        "str",
        "bytes",
        "tuple",
    ]


NO_VALUES = "There are no non-missing values to count."
UNHASHABLE = (
    "The column holds values that can't be hashed, such as lists or dicts, so its "
    "levels can't be counted."
)


def test_declares_its_statuses_and_parameters():
    [info] = [
        a for a in ae.list_analyses().analyses if a.name == "summarize_categorical"
    ]

    assert info.module == "agent_eda.statistics.descriptive"
    assert info.statuses == [
        "completed",
        "partial",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]
    assert info.parameters == [
        "data",
        "columns",
        "top_n",
        "project",
        "mask_sensitive",
    ]


def test_every_column_completes_when_each_has_a_value():
    res = ae.summarize_categorical(pd.DataFrame({"a": ["x", None], "b": [None, "y"]}))

    assert res.status == "completed"
    assert res.status_reason is None
    assert [(e.status, e.status_reason) for e in res.columns.values()] == [
        ("completed", None),
        ("completed", None),
    ]


def test_a_column_without_values_makes_the_result_partial():
    df = pd.DataFrame(
        {
            "plan": ["a", "b", "a"],
            "empty": pd.Series([None, np.nan, None], dtype=object),
        }
    )

    res = ae.summarize_categorical(df)

    empty = res.columns["empty"]
    assert empty.status == "insufficient_data"
    assert empty.status_reason is not None
    assert (empty.status_reason.code, empty.status_reason.message) == (
        "no_values",
        NO_VALUES,
    )
    assert (empty.non_missing_count, empty.missing_count, empty.cardinality) == (
        0,
        3,
        0,
    )
    assert (empty.levels, empty.most_common_levels, empty.value_types) == ([], [], [])
    assert res.status == "partial"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "incomplete_columns",
        f"1 of 2 columns didn't complete: 'empty'. {NO_VALUES}",
    )


def test_no_column_with_values_gives_insufficient_data():
    df = pd.DataFrame({"a": pd.Series([None, None], dtype="string"), "b": [1.5, 2.5]})

    res = ae.summarize_categorical(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_values",
        f"No column completed: 'a'. {NO_VALUES}",
    )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"t": pd.Series([], dtype=object)}),
        pd.DataFrame(
            {"t": pd.Series([], dtype="string"), "n": pd.Series([], dtype=float)}
        ),
        pd.DataFrame({"n": pd.Series([], dtype=float)}),
    ],
    ids=["nothing", "text", "text-and-numbers", "numbers"],
)
def test_data_without_rows_gives_insufficient_data(df):
    res = ae.summarize_categorical(df)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_rows",
        "The data has no rows.",
    )
    assert all(e.status == "insufficient_data" for e in res.columns.values())


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"age": [34, 51], "when": pd.to_datetime(["2024-01-01", None])}),
        pd.DataFrame(index=range(3)),
    ],
    ids=["numbers-and-datetimes", "no-columns"],
)
def test_data_without_text_categorical_or_boolean_columns_is_not_applicable(df):
    res = ae.summarize_categorical(df)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_categorical_columns",
        "The data has no text, object, categorical or boolean columns. Name a "
        "numeric column in columns to count its values.",
    )
    assert res.columns == {}
    assert res.parameters["columns"] == []


def test_values_that_cannot_be_hashed_make_a_column_not_applicable():
    tags = pd.Series([["a"], None, {"k": 1}, "x"], dtype=object)
    df = pd.DataFrame({"tags": tags, "plan": ["a", "b", "a", "b"]})

    res = ae.summarize_categorical(df)

    entry = res.columns["tags"]
    assert entry.status == "not_applicable"
    assert entry.status_reason is not None
    assert (entry.status_reason.code, entry.status_reason.message) == (
        "unhashable_values",
        UNHASHABLE,
    )
    assert (entry.non_missing_count, entry.missing_count) == (3, 1)
    assert (entry.cardinality, entry.left_out_level_count, entry.seen_once_count) == (
        None,
        None,
        None,
    )
    assert (entry.levels, entry.most_common_levels) == ([], [])
    # The types of the values are still known.
    assert (entry.value_types, entry.mixed_value_types) == (
        ["dict", "list", "str"],
        True,
    )
    assert res.status == "partial"
    [limitation] = [lim for lim in res.limitations if lim.code == "unhashable_values"]
    assert limitation.reason == (
        "1 column holds values that can't be hashed, such as lists or dicts, so its "
        "levels weren't counted and its counts of levels are null: 'tags'."
    )
    assert limitation.fields == [
        "columns['tags'].cardinality",
        "columns['tags'].left_out_level_count",
        "columns['tags'].seen_once_count",
    ]
    assert res.to_frame()["column"].tolist() == ["plan", "plan"]


def test_only_columns_that_cannot_be_hashed_give_not_applicable():
    df = pd.DataFrame({"tags": pd.Series([["a"], ["b"]], dtype=object)})

    res = ae.summarize_categorical(df)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "unhashable_values",
        f"No column completed: 'tags'. {UNHASHABLE}",
    )


def test_support_counts_the_values_each_column_left_out():
    df = pd.DataFrame(
        {
            "plan": pd.Series(["a", None, "b", None], dtype=object),
            "flag": [True, False, True, True],
            "tags": pd.Series([["a"], None, ["b"], "c"], dtype=object),
        }
    )

    support = ae.summarize_categorical(df).support

    assert (support.row_count, support.column_count) == (4, 3)
    assert [
        (
            c.column,
            c.eligible_count,
            c.evaluated_count,
            c.excluded_count,
            c.excluded_by_reason,
        )
        for c in support.columns
    ] == [
        ("plan", 4, 2, 2, {"missing": 2}),
        ("flag", 4, 4, 0, {}),
        ("tags", 4, 0, 4, {"missing": 1, "unhashable": 3}),
    ]


def findings_by_code(res):
    return {finding.code: finding for finding in res.findings}


def test_labels_that_differ_only_by_case_or_surrounding_spaces_are_suspected():
    df = pd.DataFrame(
        {
            "answer": ["Yes", "yes ", "YES", "no", "No", "maybe", "Yes"],
            "city": ["Lyon", "Oslo", "Rome", "Lyon", "Oslo", "Rome", "Lyon"],
        }
    )

    res = ae.summarize_categorical(df)

    groups = [
        [(level.label, level.count) for level in group.levels]
        for group in res.columns["answer"].case_or_space_variants
    ]
    # In level order: 'Yes' leads, then the labels seen once by code point.
    assert groups == [[("Yes", 2), ("YES", 1), ("yes ", 1)], [("No", 1), ("no", 1)]]
    assert res.columns["city"].case_or_space_variants == []
    finding = findings_by_code(res)["case_or_space_variants"]
    assert finding.assessment == "suspected"
    assert finding.claim == (
        "1 column holds labels that differ only by case or surrounding spaces: "
        "'answer'."
    )
    assert finding.columns == ["answer"]
    assert finding.fields == ["columns['answer'].case_or_space_variants"]


def test_case_follows_unicode_case_folding_and_whitespace_includes_tabs():
    df = pd.DataFrame({"street": ["Straße", "STRASSE", "\tstrasse\n", "Ring"]})

    entry = ae.summarize_categorical(df).columns["street"]

    [group] = entry.case_or_space_variants
    assert [level.label for level in group.levels] == [
        "\tstrasse\n",
        "STRASSE",
        "Straße",
    ]


def test_values_of_different_python_types_are_suspected():
    df = pd.DataFrame(
        {
            "code": pd.Series([1, "1", 2.5], dtype=object),
            "plain": ["a", "b", "c"],
            "flag": [True, False, True],
        }
    )

    res = ae.summarize_categorical(df)

    finding = findings_by_code(res)["mixed_value_types"]
    assert finding.assessment == "suspected"
    assert finding.claim == (
        "1 column holds values of more than one Python type: 'code' (float, int, str)."
    )
    assert finding.columns == ["code"]
    assert finding.fields == ["columns['code'].value_types"]


def test_unused_categories_are_a_descriptive_finding():
    df = pd.DataFrame(
        {
            "grade": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "size": pd.Categorical(["s", "m", "s"]),
        }
    )

    res = ae.summarize_categorical(df)

    finding = findings_by_code(res)["unused_categories"]
    assert finding.assessment == "descriptive"
    assert finding.claim == (
        "1 categorical column has categories that never occur: 'grade'."
    )
    assert finding.columns == ["grade"]
    assert finding.fields == ["columns['grade'].unused_categories"]


def test_declares_its_finding_codes_and_reports_them_in_that_order():
    df = pd.DataFrame(
        {
            "grade": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "code": pd.Series([1, "1", 2.5], dtype=object),
            "answer": ["Yes", "yes", "no"],
        }
    )
    [info] = [
        a for a in ae.list_analyses().analyses if a.name == "summarize_categorical"
    ]

    res = ae.summarize_categorical(df)

    codes = ["case_or_space_variants", "mixed_value_types", "unused_categories"]
    assert info.finding_codes == codes
    assert [finding.code for finding in res.findings] == codes


def test_clean_columns_give_no_findings():
    df = pd.DataFrame({"plan": ["basic", "pro", "basic"], "flag": [True, False, True]})

    assert ae.summarize_categorical(df).findings == []


TEXT = ["b", "a", None, "b", "c", None, "a", "b", "A"]
FLAGS = [True, None, False, True, True, None, False, True, True]
NUMBERS = [3, 1, None, 3, 2, None, 1, 3, 7]

# The same values in every dtype family the ticket names; None becomes missing.
VALUE_COUNT_CASES = {
    "object": pd.Series(TEXT, dtype=object),
    "pandas 3 str": pd.Series(TEXT, dtype=STR),
    "string": pd.Series(TEXT, dtype="string"),
    "string[pyarrow]": pd.Series(TEXT, dtype="string[pyarrow]"),
    "arrow string": pd.Series(TEXT, dtype=pd.ArrowDtype(pa.string())),
    "category": pd.Series(pd.Categorical(TEXT, categories=["c", "b", "a", "A", "z"])),
    "bool": pd.Series([bool(flag) for flag in FLAGS]),
    "boolean": pd.Series(FLAGS, dtype="boolean"),
    "bool[pyarrow]": pd.Series(FLAGS, dtype="bool[pyarrow]"),
    "int64[pyarrow]": pd.Series(NUMBERS, dtype="int64[pyarrow]"),
    "Int64": pd.Series(NUMBERS, dtype="Int64"),
    "float64": pd.Series(NUMBERS, dtype="float64"),
}


@pytest.mark.parametrize("dtype", list(VALUE_COUNT_CASES))
def test_counts_and_shares_match_pandas_value_counts(dtype):
    values = VALUE_COUNT_CASES[dtype]
    df = pd.DataFrame({"x": values})

    res = ae.summarize_categorical(df, columns=["x"])

    frame = res.to_frame()
    counts = values.value_counts(dropna=True)
    shares = values.value_counts(dropna=True, normalize=True)
    # A categorical lists its unused categories with a count of 0.
    assert dict(zip(frame["label"], frame["count"], strict=True)) == {
        label: count for label, count in counts.items() if count
    }
    assert dict(zip(frame["label"], frame["share"], strict=True)) == pytest.approx(
        {label: share for label, share in shares.items() if share}, rel=1e-12
    )
    entry = res.columns["x"]
    assert entry.non_missing_count == values.count()
    assert entry.missing_count == values.isna().sum()
    assert entry.cardinality == values.nunique()


def survey():
    return pd.DataFrame(
        {
            "plan": pd.Series(
                ["basic", "pro", "basic", None, "team", "basic"], dtype=object
            ),
            "answer": ["Yes", "yes ", "no", "YES", "no", "Yes"],
            "grade": pd.Categorical(
                ["a", "b", "a", "a", None, "b"], categories=["a", "b", "c"]
            ),
            "age": [34, 51, 29, 47, 38, 41],
        }
    )


def test_text_shows_each_columns_most_common_levels_with_count_and_share():
    res = ae.summarize_categorical(survey())

    assert str(res) == "\n".join(
        [
            "summarize_categorical: completed on 6 rows x 4 columns",
            "Shares are of each column's non-missing values. Levels go by count, "
            "then label; most_common_levels has the top 10.",
            "Left out, not text, object, categorical or boolean: 'age'.",
            "Columns (3 of 3 shown):",
            "column  non_missing_count  missing_count  cardinality  most_common_levels",
            "plan                    5              1            3  'basic': 3 (60%), "
            "'pro': 1 (20%), 'team': 1 (20%)",
            # Cut to 120 characters, with "..." for the rest.
            "answer                  6              0            4  'Yes': 2 (33.33%), "
            "'no': 2 (33.33%), 'YES': 1 (16.67%), 'yes '...",
            "grade                   5              1            2  'a': 3 (60%), "
            "'b': 2 (40%)",
            "Findings (2 of 2 shown):",
            "- suspected case_or_space_variants: 1 column holds labels that differ "
            "only by case or surrounding spaces: 'answer'.",
            "- descriptive unused_categories: 1 categorical column has categories "
            "that never occur: 'grade'.",
            "Limitations: none",
            "More: res.to_frame() has all 9 levels; res.findings, res.limitations "
            "and res.to_json() have the rest.",
        ]
    )


def test_text_shows_labels_of_other_types_without_quotes():
    df = pd.DataFrame({"flag": [True, False, True, True], "n": [2.5, 2.5, None, 1.0]})

    lines = str(
        ae.summarize_categorical(df, columns=["flag", "n"], top_n=1)
    ).splitlines()

    assert lines[1].endswith("most_common_levels has the top 1.")
    assert lines[4].endswith("  True: 3 (75%)")
    assert lines[5].endswith("  2.5: 2 (66.67%)")


def test_text_of_a_partial_result_names_the_incomplete_columns():
    df = pd.DataFrame(
        {"plan": ["a", "b"], "empty": pd.Series([None, None], dtype=object)}
    )

    lines = str(ae.summarize_categorical(df)).splitlines()

    assert lines[:2] == [
        "summarize_categorical: partial on 2 rows x 2 columns",
        f"Reason: incomplete_columns: 1 of 2 columns didn't complete: 'empty'. {NO_VALUES}",
    ]
    assert "empty                   0              2            0" in lines


def test_text_stays_within_40_lines_of_120_characters():
    long = "a_long_label_" * 12
    df = pd.DataFrame(
        {
            f"{'a_long_column_name_' * 8}{i}": [f"{long}{j}" for j in range(5)]
            for i in range(60)
        }
    )

    lines = str(ae.summarize_categorical(df)).split("\n")

    assert len(lines) <= 40
    assert max(len(line) for line in lines) <= 120


def stored_nan_as_missing(values: pd.Series) -> pd.Series:
    """The column with NaN that it stores as a value made missing, as pandas won't."""
    dtype = values.dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and dtype.kind == "f":
        floats = values.to_numpy(dtype="float64", na_value=np.nan)
        return values.mask(np.isnan(floats))
    return values


@given(df=data_frames(), top_n=st.integers(1, 100))
def test_any_columns_levels_match_value_counts_and_each_other(df, top_n):
    countable = names(df, has_countable_values)
    assume(countable)

    res = ae.summarize_categorical(df, columns=countable, top_n=top_n)

    frame = res.to_frame()
    labels = [str(label) for label in df.columns]
    for name in countable:
        entry = res.columns[name]
        if entry.status == "not_applicable":  # values that can't be hashed
            continue
        values = df.iloc[:, labels.index(name)]
        values = stored_nan_as_missing(values)
        counts = values.value_counts(dropna=True)
        rows = frame[frame["column"] == name]
        assert sorted(rows["count"]) == sorted(c for c in counts if c)
        assert entry.cardinality == len(rows)
        assert entry.non_missing_count == rows["count"].sum() == values.count()
        assert list(rows["count"]) == sorted(rows["count"], reverse=True)
        assert list(rows["share"]) == list(
            rows["count"] / max(entry.non_missing_count, 1)
        )
        kept = rows.head(100)
        assert [(lvl.count, lvl.label_type) for lvl in entry.levels] == list(
            zip(kept["count"], kept["label_type"], strict=True)
        )
        assert entry.most_common_levels == entry.levels[:top_n]
        assert entry.left_out_level_count == entry.cardinality - len(
            entry.most_common_levels
        )


def test_the_result_states_the_level_order():
    res = ae.summarize_categorical(letters())

    assert res.level_order == (
        "Count descending, then label. Booleans and numbers come first, by value, "
        "with False and True as 0 and 1 and NaN last. Text follows, by Unicode code "
        "point, then labels of any other type, by type name and string form. Labels "
        "that still tie keep their order in the data."
    )


def test_orders_numbers_by_value_with_infinities_at_the_ends_and_nan_last():
    stored = pa.array(
        [1.5, float("nan"), float("inf"), -float("inf"), None],
        type=pa.float64(),
        from_pandas=False,
    )
    df = pd.DataFrame({"x": pd.Series(pd.arrays.ArrowExtensionArray(stored))})

    entry = ae.summarize_categorical(df, columns=["x"]).columns["x"]

    # pyarrow keeps NaN as a value, apart from the missing value; the analysis
    # counts both as missing.
    assert [(level.label, level.label_type) for level in entry.levels] == [
        ("-inf", "float"),
        (1.5, "float"),
        ("inf", "float"),
    ]
    assert entry.missing_count == 2


def test_identical_calls_in_processes_with_different_hash_seeds_give_the_same_json():
    code = (
        "import pandas as pd\n"
        "import agent_eda as ae\n"
        "values = ['b', 2, True, b'x', 'a', 1.5, (1, 2), 'A', 'a ', 'c', 'b', 2]\n"
        "df = pd.DataFrame({'m': pd.Series(values, dtype=object), 't': [str(v) for v in values]})\n"
        "print(ae.summarize_categorical(df).to_json())\n"
    )

    printed = [
        run_python(code, env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ("1", "2")
    ]

    assert printed[0] == printed[1]
    assert '"label":"a"' in printed[0]


def run_python(code, *, env):
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout
