import datetime as dt
import inspect
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import data_frames

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError


def orders():
    """Six orders; 101, 103 and 106 repeat one purchase, 102 and 105 another."""
    return pd.DataFrame(
        {
            "order": [101, 102, 103, 104, 105, 106],
            "customer": ["ann", "bob", "ann", "cy", "bob", "ann"],
            "amount": [9.5, 20.0, 9.5, 9.5, 20.0, 9.5],
        }
    )


def test_by_default_every_column_is_compared():
    res = ae.find_duplicate_rows(orders())

    assert res.compared_columns == ["order", "customer", "amount"]
    assert res.parameters["columns"] == ["order", "customer", "amount"]


def test_by_default_the_id_column_is_left_out_of_the_comparison():
    res = ae.find_duplicate_rows(orders(), id_column="order")

    assert res.compared_columns == ["customer", "amount"]
    assert res.parameters["id_column"] == "order"


def test_named_columns_are_compared_in_the_callers_order_id_column_included():
    res = ae.find_duplicate_rows(
        orders(), columns=["amount", "order"], id_column="order"
    )

    assert res.compared_columns == ["amount", "order"]


def test_an_unknown_id_column_raises_with_the_closest_names():
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.find_duplicate_rows(orders(), id_column="Order ")

    assert caught.value.code == "column_not_found"
    assert str(caught.value) == (
        "find_duplicate_rows got an unknown column 'Order ' in id_column. Did you "
        "mean 'order'? The data has 3 columns: 'order', 'customer' and 'amount'."
    )


@pytest.mark.parametrize(
    ("tags", "dtype"),
    [
        (pd.Series([["a"], ["b"]], dtype=object), "object"),
        (
            pd.Series([[1], [2]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))),
            "list<item: int64>[pyarrow]",
        ),
    ],
    ids=["object", "pyarrow"],
)
def test_an_id_column_whose_values_cant_be_hashed_raises(tags, dtype):
    df = pd.DataFrame({"tags": tags, "x": [1, 2]})

    with pytest.raises(ColumnTypeError) as caught:
        ae.find_duplicate_rows(df, id_column="tags")

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        f"find_duplicate_rows can't use column 'tags' with dtype {dtype} as "
        "id_column: it holds values that can't be hashed, such as lists or dicts, "
        "and so can't identify rows. Pass a column of IDs such as numbers or "
        "strings, or leave id_column=None to use the index labels."
    )


def test_an_id_column_with_a_label_that_isnt_a_string_can_be_named_by_the_label():
    df = pd.DataFrame({0: [7, 8, 9], 1: ["a", "a", "b"]})

    res = ae.find_duplicate_rows(df, id_column=0)  # ty: ignore[invalid-argument-type]  # a label that isn't a str is the test

    assert res.parameters["id_column"] == "0"
    assert res.compared_columns == ["1"]


def test_groups_hold_their_size_and_row_ids_and_the_counts_follow():
    res = ae.find_duplicate_rows(orders(), id_column="order")

    assert [(g.size, g.row_ids) for g in res.duplicate_groups] == [
        (3, [101, 103, 106]),
        (2, [102, 105]),
    ]
    assert res.duplicate_group_count == 2
    assert res.duplicate_group_row_count == 5
    # Excess copies are the rows in groups minus the groups: 5 - 2.
    assert res.excess_copy_count == 3
    # The excess copies over all rows: 3 / 6.
    assert res.duplicate_share == 0.5


def test_groups_are_ordered_by_first_occurrence_and_rows_by_position():
    df = pd.DataFrame(
        {"x": ["b", "a", "a", "b", "c", "a"]}, index=[50, 40, 30, 20, 10, 0]
    )

    res = ae.find_duplicate_rows(df)

    assert [g.row_ids for g in res.duplicate_groups] == [[50, 20], [40, 30, 0]]


def test_rows_must_be_equal_in_every_compared_column():
    df = pd.DataFrame({"x": [1, 1, 1, 2], "y": ["a", "a", "b", "a"]})

    both = ae.find_duplicate_rows(df)
    only_x = ae.find_duplicate_rows(df, columns=["x"])

    assert [g.row_ids for g in both.duplicate_groups] == [[0, 1]]
    assert [g.row_ids for g in only_x.duplicate_groups] == [[0, 1, 2]]
    assert (only_x.excess_copy_count, only_x.duplicate_share) == (2, 0.5)


def test_without_duplicates_every_count_is_zero():
    res = ae.find_duplicate_rows(orders())

    assert res.duplicate_groups == []
    assert (
        res.duplicate_group_count,
        res.duplicate_group_row_count,
        res.excess_copy_count,
        res.duplicate_share,
    ) == (0, 0, 0, 0.0)


def test_missing_values_compare_equal_in_every_dtype():
    df = pd.DataFrame(
        {
            "float": [1.5, None, 1.5, None],
            "text": pd.Series(["a", None, "a", None], dtype=object),
            "integer": pd.array([7, None, 7, None], dtype="Int64"),
            "when": pd.to_datetime(["2024-01-01", None, "2024-01-01", None]),
            "kind": pd.Categorical(["x", None, "x", None]),
            "arrow": pd.array([1.0, None, 1.0, None], dtype="double[pyarrow]"),
        }
    )

    res = ae.find_duplicate_rows(df)

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 2], [1, 3]]
    assert res.missing_values_compare_equal is True


def test_every_kind_of_missing_value_equals_every_other():
    # As pandas' duplicated() compares two or more columns; for one column, its
    # Series.duplicated() tells None, NaN, NaT and pd.NA apart in object columns.
    missing = pd.Series([None, float("nan"), pd.NaT, pd.NA], dtype=object)

    res = ae.find_duplicate_rows(pd.DataFrame({"x": missing}))

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 1, 2, 3]]


def test_lists_and_dicts_compare_by_content():
    tags = pd.Series(
        [["a", "b"], {"k": 1, "j": 2}, ["a", "b"], {"j": 2, "k": 1}, ["b", "a"], "a"],
        dtype=object,
    )

    res = ae.find_duplicate_rows(pd.DataFrame({"tags": tags}))

    # Dicts are equal whatever the order of their keys, as Python's == says.
    assert [g.row_ids for g in res.duplicate_groups] == [[0, 2], [1, 3]]


def test_missing_values_inside_lists_compare_equal():
    tags = pd.Series([[1, None], [1, float("nan")], [1, 2]], dtype=object)

    res = ae.find_duplicate_rows(pd.DataFrame({"tags": tags}))

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 1]]


def test_arrays_sets_and_other_unhashable_values_compare_by_content():
    values = [
        np.array([1.0, 2.0]),
        {1, 2},
        (1, [2]),
        bytearray(b"ab"),
        np.array(5),
        np.array([1.0, 2.0]),
        {2, 1},
        (1, [2]),
        bytearray(b"ab"),
        5,
        np.array([[1.0, 2.0]]),
    ]
    column = np.empty(len(values), dtype=object)
    column[:] = values  # an element-wise copy keeps arrays as single values

    res = ae.find_duplicate_rows(pd.DataFrame({"v": column}))

    # A 0-dimensional array equals the value it holds; shapes tell arrays apart.
    assert [g.row_ids for g in res.duplicate_groups] == [
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
    ]


def test_values_pandas_can_hash_compare_as_pandas_whatever_else_the_column_holds():
    # The list makes find_duplicate_rows compare this column by content. The
    # tuples it can hash still compare as pandas compares them, (1, nan) and
    # (1, None) as different values, so the order of the columns doesn't matter.
    nan = float("nan")
    tuples = pd.Series([(1, nan), (1, None), [5]], dtype=object)
    df = pd.DataFrame({"x": [0, 0, 1], "t": tuples})

    by_x_first = ae.find_duplicate_rows(df, columns=["x", "t"])
    by_t_first = ae.find_duplicate_rows(df, columns=["t", "x"])

    assert by_x_first.duplicate_groups == by_t_first.duplicate_groups == []


def test_pyarrow_lists_compare_by_content():
    lists = pd.Series(
        [[1, 2], None, [], [1, 2], None, [2, 1]],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )

    res = ae.find_duplicate_rows(pd.DataFrame({"x": [0, 1, 1, 0, 1, 0], "l": lists}))

    assert [g.row_ids for g in res.duplicate_groups] == [[0, 3], [1, 4]]


def test_to_frame_lists_every_row_in_a_group_with_its_group_number():
    frame = ae.find_duplicate_rows(orders(), id_column="order").to_frame()

    assert list(frame.columns) == ["group", "row_id"]
    assert frame["group"].dtype == "int64"
    assert frame.to_dict("list") == {
        "group": [0, 0, 0, 1, 1],
        "row_id": [101, 103, 106, 102, 105],
    }


def test_to_frame_without_duplicates_is_empty_with_the_same_columns():
    frame = ae.find_duplicate_rows(orders()).to_frame()

    assert frame.empty
    assert list(frame.columns) == ["group", "row_id"]


def test_the_record_keeps_the_first_100_groups_and_their_total():
    df = pd.DataFrame({"x": list(range(150)) * 2})

    res = ae.find_duplicate_rows(df)
    record = res.to_dict()

    assert len(record["duplicate_groups"]) == 100
    assert record["duplicate_groups_total"] == 150
    assert record["duplicate_groups"][99]["row_ids"] == [99, 249]
    assert len(res.duplicate_groups) == 150
    assert len(res.to_frame()) == 300


def test_row_ids_are_index_labels_so_loc_shows_the_rows():
    df = orders().set_index(pd.Index(["a", "b", "c", "d", "e", "f"]))

    res = ae.find_duplicate_rows(df, columns=["customer", "amount"])

    ids = res.duplicate_groups[0].row_ids
    assert ids == ["a", "c", "f"]
    assert df.loc[ids, "order"].tolist() == [101, 103, 106]


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2024-01-01", periods=4, freq="h", tz="Europe/Berlin"),
        pd.MultiIndex.from_tuples([("g0", 1), ("g0", 2), ("g1", 1), ("g1", 2)]),
        pd.CategoricalIndex(["w", "x", "y", "z"]),
        pd.Index([0.5, 1.5, 2.5, 3.5]),
    ],
    ids=["datetime", "multiindex", "categorical", "float"],
)
def test_index_labels_of_any_dtype_find_the_rows_with_loc(index):
    df = pd.DataFrame({"x": [1, 2, 1, 2]}, index=index)

    res = ae.find_duplicate_rows(df)

    for group in res.duplicate_groups:
        values = df.loc[group.row_ids, "x"]
        assert (values == values.iloc[0]).all()
        assert len(values) == group.size


def test_row_ids_are_the_id_columns_values_when_it_is_given():
    df = orders().assign(code=["o1", "o2", "o3", "o4", "o5", "o6"])

    res = ae.find_duplicate_rows(df, columns=["customer", "amount"], id_column="code")

    assert [g.row_ids for g in res.duplicate_groups] == [
        ["o1", "o3", "o6"],
        ["o2", "o5"],
    ]


def test_the_record_keeps_10_row_ids_per_group_and_their_total():
    res = ae.find_duplicate_rows(pd.DataFrame({"x": [1] * 12}))

    [group] = res.to_dict()["duplicate_groups"]

    assert group == {"size": 12, "row_ids": list(range(10)), "row_ids_total": 12}
    assert res.duplicate_groups[0].row_ids == list(range(12))


def record_row_ids(labels):
    """The record's row IDs when every row, labelled by `labels`, is a duplicate."""
    df = pd.DataFrame({"x": [1] * len(labels)}, index=labels)
    res = ae.find_duplicate_rows(df)
    record = json.loads(res.to_json(), parse_constant=reject_constant)
    return record["duplicate_groups"][0]["row_ids"]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (
            pd.DatetimeIndex(["2024-01-01", "2024-03-01 12:30:00.000001"]),
            ["2024-01-01T00:00:00", "2024-03-01T12:30:00.000001"],
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-07-01"], tz="Europe/Berlin"),
            ["2024-01-01T00:00:00+01:00", "2024-07-01T00:00:00+02:00"],
        ),
        (
            pd.MultiIndex.from_tuples([("g0", 1), ("g1", None)]),
            [["g0", 1], ["g1", None]],
        ),
        (
            pd.Index([np.int64(5), np.float64(2.5), np.bool_(True)], dtype=object),
            [5, 2.5, True],
        ),
        (pd.Index([1.5, np.nan, np.inf, -np.inf]), [1.5, None, "inf", "-inf"]),
        (pd.Index([None, pd.NaT, "a"], dtype=object), [None, None, "a"]),
        (pd.TimedeltaIndex(["1 day", "90 min"]), ["P1DT0H0M0S", "P0DT1H30M0S"]),
        (pd.PeriodIndex(["2024-01", "2024-02"], freq="M"), ["2024-01", "2024-02"]),
        (
            pd.Index([dt.date(2024, 1, 1), Decimal("1.50")], dtype=object),
            ["2024-01-01", "1.50"],
        ),
        (
            pd.Index([np.timedelta64(1, "D"), np.timedelta64(90, "m")], dtype=object),
            ["P1DT0H0M0S", "P0DT1H30M0S"],
        ),
        (
            # Sets list their items in a fixed order, not Python's hash order.
            pd.Index(
                [frozenset({"b", "c", "a"}), ("t", frozenset({2, 1}))], dtype=object
            ),
            [["a", "b", "c"], ["t", [1, 2]]],
        ),
        (
            pd.Index([{"b": 1, "a": {2, 1}}, {"c": None}], dtype=object),
            [{"b": 1, "a": [1, 2]}, {"c": None}],
        ),
    ],
    ids=[
        "naive",
        "aware",
        "multiindex",
        "numpy",
        "float",
        "missing",
        "timedelta",
        "period",
        "date-decimal",
        "numpy-timedelta",
        "sets",
        "dicts",
    ],
)
def test_row_ids_convert_to_json_deterministically(labels, expected):
    assert record_row_ids(labels) == expected


def test_the_result_object_keeps_the_labels_as_pandas_gives_them():
    labels = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC")

    res = ae.find_duplicate_rows(pd.DataFrame({"x": [1, 1]}, index=labels))

    assert res.duplicate_groups[0].row_ids == list(labels)


def reject_constant(name):
    raise AssertionError(f"to_json() isn't strict JSON: it holds {name}")


def test_the_record_converts_only_the_groups_and_row_ids_it_keeps():
    converted = []

    class Label:
        """An index label that notes each time it is turned into text."""

        def __init__(self, number):
            self.number = number

        def __hash__(self):
            return hash(self.number)

        def __eq__(self, other):
            return isinstance(other, Label) and other.number == self.number

        def __str__(self):
            converted.append(self.number)
            return f"L{self.number}"

    # 150 groups of 12 rows each.
    index = pd.Index([Label(i) for i in range(1_800)], dtype=object)
    res = ae.find_duplicate_rows(
        pd.DataFrame({"x": list(range(150)) * 12}, index=index)
    )
    converted.clear()

    record = res.to_dict()

    assert record["duplicate_groups"][1]["row_ids"][:2] == ["L1", "L151"]
    assert len(converted) == 100 * 10


def test_repeated_index_labels_make_row_ids_ambiguous():
    # Concatenating data without ignore_index repeats the index labels.
    df = pd.DataFrame({"x": [1, 2, 1, 2, 3]}, index=[0, 1, 0, 1, 2])

    res = ae.find_duplicate_rows(df)

    [limitation] = res.limitations
    assert limitation.code == "ambiguous_row_ids"
    assert limitation.fields == [
        "duplicate_groups[0].row_ids",
        "duplicate_groups[1].row_ids",
    ]
    assert limitation.reason == (
        "4 rows share an index label, so a row ID can match several rows: use a "
        "unique id_column. Or make the index unique, such as with "
        "df.reset_index(drop=True)."
    )
    # The printed line wraps, so the fix that follows is there too.
    assert (
        "- ambiguous_row_ids: 4 rows share an index label, so a row ID can match "
        "several rows: use a unique id_column. Or make the index unique, such as "
        "with df.reset_index(drop=True)."
    ) in " ".join(line.strip() for line in str(res).splitlines())


def test_repeated_id_column_values_make_row_ids_ambiguous():
    df = orders().assign(code=["o1", "o2", "o1", "o4", "o5", "o6"])

    res = ae.find_duplicate_rows(df, columns=["customer", "amount"], id_column="code")

    [limitation] = res.limitations
    assert limitation.code == "ambiguous_row_ids"
    assert limitation.fields == ["duplicate_groups[0].row_ids"]
    assert limitation.reason == (
        "2 rows share an id_column value, so a row ID can match several rows: use "
        "a unique id_column."
    )


def test_repeated_row_ids_are_ambiguous_even_when_no_group_holds_one():
    df = pd.DataFrame({"x": [5, 6, 7, 7]}, index=["a", "a", "b", "c"])

    res = ae.find_duplicate_rows(df)

    [limitation] = res.limitations
    assert limitation.code == "ambiguous_row_ids"
    assert limitation.fields == []
    assert limitation.reason.endswith(" None of the row IDs in this result repeats.")


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([f"row_{i}" for i in (3, 1, 2, 0, 1)]),
        pd.Index([30, 10, 20, 0, 40]),
        pd.MultiIndex.from_tuples([("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 3)]),
    ],
    ids=["text", "integers", "multiindex"],
)
def test_checking_row_ids_builds_no_lookup_table_on_the_callers_index(index):
    df = pd.DataFrame({"b": [1, 2, 1, 2, 3], "a": [0] * 5}, index=index)
    index_memory = df.index.memory_usage(deep=True)
    columns_memory = df.columns.memory_usage(deep=True)

    res = ae.find_duplicate_rows(df)

    assert res.duplicate_group_count == 2
    assert df.index.memory_usage(deep=True) == index_memory
    assert df.columns.memory_usage(deep=True) == columns_memory


@pytest.mark.parametrize(
    ("labels", "repeat"),
    [
        (pd.Index([np.nan, np.nan, 3.0, 4.0]), True),
        # df.loc tells None, NaN and NaT apart, and so does pandas' duplicated().
        (pd.Index([None, np.nan, pd.NaT, "a"], dtype=object), False),
        (pd.Index([[1], [2], [1], [3]], dtype=object), True),
        (
            pd.Index(
                pd.array(
                    [[1], [2], [1], [3]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))
                )
            ),
            True,
        ),
    ],
    ids=["nan", "kinds-of-missing", "lists", "pyarrow-lists"],
)
def test_row_ids_repeat_where_describe_dataset_finds_the_index_not_unique(
    labels, repeat
):
    df = pd.DataFrame({"x": [1, 1, 2, 3]}, index=labels)

    res = ae.find_duplicate_rows(df)
    described = ae.describe_dataset(df)

    assert res.status == "completed"
    assert res.duplicate_group_count == 1
    assert ([lim.code for lim in res.limitations] == ["ambiguous_row_ids"]) is repeat
    assert described.index.is_unique is not repeat
    json.loads(res.to_json(), parse_constant=reject_constant)


def test_unique_row_ids_are_not_ambiguous():
    df = pd.DataFrame(
        {"x": [1, 1, 2]},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]),
    )

    assert ae.find_duplicate_rows(df).limitations == []
    assert ae.find_duplicate_rows(orders(), id_column="order").limitations == []


def test_duplicates_give_a_suspected_finding():
    res = ae.find_duplicate_rows(orders(), id_column="order")

    [finding] = res.findings
    assert finding.code == "duplicate_rows"
    assert finding.assessment == "suspected"
    assert finding.claim == (
        "5 rows form 2 duplicate groups; 3 are excess copies (duplicate_share 0.5)."
    )
    assert finding.columns == ["customer", "amount"]
    assert finding.fields == [
        "duplicate_group_count",
        "duplicate_group_row_count",
        "excess_copy_count",
        "duplicate_share",
    ]


def test_a_single_group_and_copy_read_in_the_singular():
    [finding] = ae.find_duplicate_rows(pd.DataFrame({"x": [1, 1, 2]})).findings

    assert finding.claim == (
        "2 rows form 1 duplicate group; 1 is an excess copy (duplicate_share 0.3333)."
    )


def test_no_duplicates_give_no_finding_and_the_text_says_what_was_compared():
    res = ae.find_duplicate_rows(orders())

    assert res.findings == []
    assert res.duplicate_group_count == 0


def test_the_declared_statuses_are_the_four_the_ticket_names():
    [info] = [i for i in ae.list_analyses().analyses if i.name == "find_duplicate_rows"]

    assert info.statuses == [
        "completed",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]


def test_data_without_rows_gives_insufficient_data_and_a_null_share():
    res = ae.find_duplicate_rows(orders().iloc[:0])

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_rows",
        "The data has no rows.",
    )
    assert res.findings == []
    assert (res.duplicate_group_count, res.duplicate_share) == (0, None)
    [limitation] = res.limitations
    assert limitation.code == "too_few_values"
    assert limitation.fields == ["duplicate_share"]
    assert limitation.reason == (
        "The data has no rows, so duplicate_share, the excess copies over all "
        "rows, doesn't exist."
    )


@pytest.mark.parametrize(
    ("df", "id_column", "message"),
    [
        (
            pd.DataFrame(index=range(3)),
            None,
            "The data has no columns to compare.",
        ),
        (
            pd.DataFrame({"order": [1, 2, 3]}),
            "order",
            "The data has no column to compare besides id_column 'order', which "
            "the default columns leave out. Name the columns to compare in columns.",
        ),
    ],
    ids=["no-columns", "only-the-id-column"],
)
def test_no_column_to_compare_gives_not_applicable(df, id_column, message):
    res = ae.find_duplicate_rows(df, id_column=id_column)

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_columns_to_compare"
    assert res.status_reason.message == message
    assert res.findings == []
    assert res.duplicate_groups == []


def test_a_single_row_completes_without_duplicates():
    res = ae.find_duplicate_rows(orders().iloc[:1])

    assert res.status == "completed"
    assert res.findings == []
    assert res.duplicate_group_count == 0


def test_the_printed_result_shows_the_groups_and_how_rows_were_compared():
    res = ae.find_duplicate_rows(orders(), id_column="order")

    lines = str(res).splitlines()

    assert lines[0] == "find_duplicate_rows: completed on 6 rows x 3 columns"
    assert lines[1].startswith("Compared 2 columns: 'customer' and 'amount';")
    assert "Row IDs: values of id_column 'order'." in lines
    start = lines.index("Duplicate groups (2 of 2 shown):")
    assert [line.split() for line in lines[start + 1 : start + 4]] == [
        ["group", "size", "row_ids"],
        ["0", "3", "101,", "103", "and", "106"],
        ["1", "2", "102", "and", "105"],
    ]
    assert (
        "- suspected duplicate_rows: 5 rows form 2 duplicate groups; 3 are excess "
        "copies (duplicate_share 0.5)."
    ) in lines
    assert lines[-1].startswith("More: res.to_frame() has all 5 rows in groups;")


def test_the_printed_result_without_duplicates_says_so():
    lines = str(ae.find_duplicate_rows(orders())).splitlines()

    assert lines[1:5] == [
        "Compared 3 columns: 'order', 'customer' and 'amount'; no two rows are "
        "equal in them; missing values compare equal, as in",
        "    pandas.",
        "Row IDs: index labels.",
        "Duplicate groups: none",
    ]
    assert lines[-1] == (
        "More: res.to_frame() has no rows; res.findings, res.limitations and "
        "res.to_json() have the rest."
    )


def test_the_printed_result_shows_10_groups_and_5_row_ids_of_each():
    df = pd.DataFrame({"x": list(range(12)) * 7})

    lines = str(ae.find_duplicate_rows(df)).splitlines()

    assert lines[3] == "Duplicate groups (10 of 12 shown):"
    assert lines[5] == "    0     7  0, 12, 24, 36, 48 and 2 more"
    assert lines[-1].startswith("More: res.to_frame() has all 84 rows in groups;")


def pandas_duplicates(subset):
    """pandas' duplicated(keep=False) and duplicated() on the columns of `subset`.

    pandas compares one column with Series.duplicated(), which tells None, NaN,
    NaT and pd.NA apart in an object column. With two or more columns it
    factorizes each, so every missing value equals every other, as
    find_duplicate_rows compares. A constant column leads pandas down that path
    without changing which rows are equal.
    """
    reference = subset.copy()
    reference.columns = range(subset.shape[1])
    reference[subset.shape[1]] = 0
    return reference.duplicated(keep=False), reference.duplicated(keep="first")


def check_against_pandas(df, columns):
    """The groups and counts agree with pandas on a frame with a RangeIndex."""
    res = ae.find_duplicate_rows(df, columns=columns)
    every, later = pandas_duplicates(df[columns])

    assert res.duplicate_group_row_count == int(every.sum())
    assert res.excess_copy_count == int(later.sum())
    assert res.duplicate_group_count == int(every.sum()) - int(later.sum())
    grouped = sorted(row for group in res.duplicate_groups for row in group.row_ids)
    assert grouped == np.flatnonzero(every.to_numpy()).tolist()
    for group in res.duplicate_groups:
        # Rows in a group equal each other: pandas keeps only the first.
        assert int((~pandas_duplicates(df[columns].iloc[group.row_ids])[1]).sum()) == 1
    firsts = [group.row_ids[0] for group in res.duplicate_groups]
    assert firsts == sorted(firsts)


@given(df=data_frames(), data=st.data())
def test_groups_and_counts_match_pandas_duplicated(df, data):
    df = df.reset_index(drop=True)
    names = [str(label) for label in df.columns]
    assume(len(df) and names)
    picked = data.draw(st.lists(st.sampled_from(names), min_size=1, unique=True))
    columns = [label for label in df.columns if str(label) in picked]
    try:
        pandas_duplicates(df[columns])
    except (TypeError, NotImplementedError):
        assume(False)  # pandas can't hash lists, dicts and nested Arrow types
    # pandas reads '' and '\x00a' as one text, which Python's == and
    # find_duplicate_rows tell apart: test_value_identity covers such text.
    assume(not any(holds_nul(df[column]) for column in columns))

    check_against_pandas(df, columns)


def holds_nul(column):
    """Whether a column holds text with a NUL character."""
    return any(isinstance(value, str) and "\x00" in value for value in column)


@given(df=data_frames(), data=st.data())
def test_the_order_of_the_compared_columns_doesnt_change_the_groups(df, data):
    # Generated columns of lists and dicts also hold tuples, which can be hashed.
    df = df.reset_index(drop=True)
    names = [str(label) for label in df.columns]
    assume(len(df) and names)
    shuffled = data.draw(st.permutations(names))

    in_order = ae.find_duplicate_rows(df, columns=names)
    reordered = ae.find_duplicate_rows(df, columns=shuffled)

    assert in_order.duplicate_groups == reordered.duplicate_groups


# Values pandas can hash, some differing only in the kind of missing value they
# hold, next to values it can't hash.
MIXED_VALUES = [(0, float("nan")), (0, None), (0, 1), [1], {"a": 1}, None, "x"]


@given(
    picks=st.lists(
        st.tuples(st.integers(0, len(MIXED_VALUES) - 1), st.booleans()),
        min_size=2,
        max_size=8,
    )
)
def test_rows_compare_alike_whatever_other_rows_the_data_holds(picks):
    # The duplicates among some rows are the same with or without the other
    # rows, which may hold the values that can't be hashed.
    values = np.empty(len(picks), dtype=object)
    values[:] = [MIXED_VALUES[i] for i, _ in picks]
    kept = [position for position, (_, keep) in enumerate(picks) if keep]

    whole = ae.find_duplicate_rows(pd.DataFrame({"v": values}))
    part = ae.find_duplicate_rows(pd.DataFrame({"v": values[kept]}, index=kept))

    within = [[row for row in g.row_ids if row in kept] for g in whole.duplicate_groups]
    expected = sorted(group for group in within if len(group) >= 2)
    assert [g.row_ids for g in part.duplicate_groups] == expected


def test_groups_and_counts_match_pandas_when_every_row_appears_twice():
    rng = np.random.default_rng(7)
    half = pd.DataFrame(
        {f"c{i}": rng.integers(0, 400, 500) for i in range(12)}
        | {"text": rng.choice(["a", "b", None], 500), "flag": rng.random(500) < 0.5}
    )
    df = pd.concat([half, half.iloc[::-1], half.iloc[:50]], ignore_index=True)

    check_against_pandas(df, list(df.columns))
    check_against_pandas(df, ["text", "flag"])


def test_groups_and_counts_match_pandas_when_few_rows_repeat():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=5_000).round(1),
            "y": rng.integers(0, 3, 5_000),
            "when": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 50, 5_000), unit="D"),
        }
    )

    check_against_pandas(df, ["x", "y", "when"])


def test_describe_dataset_points_to_it_for_repeated_rows():
    doc = inspect.getdoc(ae.describe_dataset) or ""
    when_to_use = doc.split("When to use:")[1].split("Args:")[0]
    see_also = doc.split("See Also:")[1]

    assert "`find_duplicate_rows`" in when_to_use
    assert "\n    find_duplicate_rows: " in see_also


def test_describe_analysis_names_the_type_of_row_ids():
    description = ae.describe_analysis("find_duplicate_rows")

    [groups] = [f for f in description.result_fields if f.name == "duplicate_groups"]
    [row_ids] = [f for f in groups.fields if f.name == "row_ids"]
    assert row_ids.type == "list[RowId]"
