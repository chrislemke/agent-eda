import datetime as dt
import inspect
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from recipes import missing_value_arguments
from strategies import EDGE_FRAMES, STR, data_frames

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, InvalidArgumentError


def arrow(values, kind):
    """A pyarrow-backed column of the values as the Arrow type."""
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(values, type=kind)))


NAIVE = pd.to_datetime(pd.Series(["1900-01-01", None])).astype("datetime64[ns]")


def test_counts_the_declared_sentinel_values_of_a_column():
    df = pd.DataFrame({"age": [34, -999, 29, -999, 41]})

    res = ae.profile_missing_values(df, sentinel_values={"age": [-999]})

    assert res.columns["age"].sentinel_count == 2


def test_sentinel_values_are_keyword_only_and_default_to_none_declared():
    parameter = inspect.signature(ae.profile_missing_values).parameters[
        "sentinel_values"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    # Without a declaration, nothing counts as a sentinel value, not even -999.
    res = ae.profile_missing_values(pd.DataFrame({"age": [34, -999]}))
    assert res.columns["age"].missing_count == 0


def test_counts_none_nan_nat_and_na_as_native_missing_values():
    df = pd.DataFrame(
        {
            "score": [0.5, np.nan, 0.25, np.nan],
            "city": pd.Series(["Lyon", None, "Oslo", "Rome"], dtype=object),
            "seen": pd.to_datetime(["2024-01-01", None, None, "2024-01-04"]),
            "visits": pd.array([1, None, 3, None], dtype="Int64"),
        }
    )

    res = ae.profile_missing_values(df)

    native = {name: entry.native_missing_count for name, entry in res.columns.items()}
    assert native == {"score": 2, "city": 1, "seen": 2, "visits": 2}


def test_missing_count_adds_declared_sentinel_values_to_native_missing_values():
    df = pd.DataFrame({"age": [34.0, -999.0, None, -999.0, 41.0]})

    age = ae.profile_missing_values(df, sentinel_values={"age": [-999]}).columns["age"]

    # The share is over all 5 rows.
    assert age.native_missing_count == 1
    assert age.sentinel_count == 2
    assert age.missing_count == 3
    assert age.missing_share == 0.6


def test_declared_values_never_match_a_native_missing_value():
    # Each value counts once: a missing one as native, a declared one as a
    # sentinel value, however many declared values there are.
    df = pd.DataFrame(
        {
            "code": pd.Series(["N/A", None, "?", "N/A", "ok"], dtype="string"),
            "when": pd.to_datetime(["1900-01-01", None, "2024-05-01", None, None]),
        }
    )

    res = ae.profile_missing_values(
        df,
        sentinel_values={"code": ["N/A", "?"], "when": [dt.datetime(1900, 1, 1)]},
    )

    code, when = res.columns["code"], res.columns["when"]
    assert (code.native_missing_count, code.sentinel_count, code.missing_count) == (
        1,
        3,
        4,
    )
    assert code.missing_share == 0.8
    assert (when.native_missing_count, when.sentinel_count, when.missing_count) == (
        3,
        1,
        4,
    )


def gappy_frame():
    """Row 0 is complete, row 1 misses 2 values, row 2 misses 3 counting -999."""
    return pd.DataFrame(
        {
            "a": [1.0, None, None, 4.0, 5.0],
            "b": pd.Series(["x", None, "N/A", "y", "z"], dtype=object),
            "c": [1, 2, -999, 4, 5],
        }
    )


GAPPY_SENTINELS = {"b": ["N/A"], "c": [-999]}


def test_counts_complete_rows_and_how_many_values_each_row_misses():
    res = ae.profile_missing_values(gappy_frame(), sentinel_values=GAPPY_SENTINELS)

    assert res.complete_row_count == 3
    assert res.complete_row_share == 0.6
    assert [
        (entry.missing_count, entry.row_count, entry.row_share)
        for entry in res.row_distribution
    ] == [(0, 3, 0.6), (2, 1, 0.2), (3, 1, 0.2)]


def test_counts_missing_values_over_all_selected_values():
    res = ae.profile_missing_values(gappy_frame(), sentinel_values=GAPPY_SENTINELS)

    # 5 of the 15 values are missing or declared sentinel values.
    assert res.missing_count == 5
    assert res.missing_share == 5 / 15


def test_rows_count_only_the_selected_columns():
    res = ae.profile_missing_values(
        gappy_frame(), columns=["c", "b"], sentinel_values=GAPPY_SENTINELS
    )

    assert list(res.columns) == ["c", "b"]
    assert res.complete_row_count == 3
    assert [(e.missing_count, e.row_count) for e in res.row_distribution] == [
        (0, 3),
        (1, 1),
        (2, 1),
    ]
    assert res.missing_count == 3
    assert res.missing_share == 3 / 10


def test_to_frame_has_one_row_per_column_in_the_selected_order():
    res = ae.profile_missing_values(gappy_frame(), sentinel_values=GAPPY_SENTINELS)

    frame = res.to_frame()

    expected = pd.DataFrame(
        {
            "column": ["a", "b", "c"],
            "status": ["completed"] * 3,
            "status_reason": pd.Series([None] * 3, dtype=object),
            "native_missing_count": [2, 1, 0],
            "sentinel_count": pd.array([None, 1, 1], dtype="Int64"),
            "missing_count": [2, 2, 1],
            "missing_share": [0.4, 0.4, 0.2],
            "sentinel_values": [{}, {"N/A": 1}, {-999: 1}],
        }
    )
    pd.testing.assert_frame_equal(frame, expected)
    pd.testing.assert_frame_equal(res.to_frame(table="columns"), expected)


def test_to_frame_of_the_row_distribution_has_each_number_of_missing_values():
    res = ae.profile_missing_values(gappy_frame(), sentinel_values=GAPPY_SENTINELS)

    frame = res.to_frame(table="row_distribution")

    expected = pd.DataFrame(
        {
            "missing_count": [0, 2, 3],
            "row_count": [3, 1, 1],
            "row_share": [0.6, 0.2, 0.2],
        }
    )
    pd.testing.assert_frame_equal(frame, expected)


def test_to_frame_of_an_unknown_table_raises_with_the_tables_there_are():
    res = ae.profile_missing_values(gappy_frame())

    with pytest.raises(InvalidArgumentError) as caught:
        res.to_frame(table="rows")  # ty: ignore[invalid-argument-type]  # the bad name is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "to_frame got 'rows' for table, but table must be one of 'columns' or "
        "'row_distribution'. res.to_frame() returns the columns table."
    )


def test_flags_columns_with_only_missing_values():
    df = pd.DataFrame({"empty": [None, None, None], "ok": [1.0, None, 3.0]})

    res = ae.profile_missing_values(df)

    [finding] = res.findings
    assert finding.code == "all_missing_columns"
    assert finding.assessment == "suspected"
    assert finding.claim == "1 column holds only missing values: 'empty'."
    assert finding.columns == ["empty"]
    assert finding.fields == ["columns['empty'].missing_count"]


def test_all_missing_columns_count_declared_sentinel_values_as_missing():
    df = pd.DataFrame(
        {"empty": [None, None, None], "coded": [-999, -999, None], "ok": [1, 2, 3]}
    )

    res = ae.profile_missing_values(df, sentinel_values={"coded": [-999]})

    [finding] = [f for f in res.findings if f.code == "all_missing_columns"]
    assert finding.claim == (
        "2 columns hold only missing values, counting declared sentinel values: "
        "'empty' and 'coded'."
    )
    assert finding.columns == ["empty", "coded"]
    assert finding.fields == [
        "columns['empty'].missing_count",
        "columns['coded'].missing_count",
    ]


def test_flags_the_columns_where_declared_sentinel_values_occur():
    res = ae.profile_missing_values(
        gappy_frame(), sentinel_values={"a": [-1], "b": ["N/A"], "c": [-999]}
    )

    [finding] = [f for f in res.findings if f.code == "sentinel_values_present"]
    assert finding.assessment == "descriptive"
    assert finding.claim == "Declared sentinel values occur in 2 columns: 'b' and 'c'."
    assert finding.columns == ["b", "c"]
    assert finding.fields == [
        "columns['b'].sentinel_count",
        "columns['c'].sentinel_count",
    ]


def test_complete_data_has_no_findings():
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    res = ae.profile_missing_values(df, sentinel_values={"x": [-999]})

    assert res.findings == []


def test_text_shows_the_counts_most_missing_first_and_names_both_tables():
    res = ae.profile_missing_values(gappy_frame(), sentinel_values=GAPPY_SENTINELS)

    assert str(res).splitlines() == [
        "profile_missing_values: completed on 5 rows x 3 columns",
        "Missing: 5 of 15 selected values (33.33%), counting declared sentinel values. "
        "Complete rows: 3 of 5 (60%).",
        "Missing values per row: 0 in 3 rows, 2 in 1 row and 3 in 1 row.",
        "Columns, most missing first (3 of 3 shown):",
        "column  native_missing_count  sentinel_count  missing_count  missing_share",
        "a                          2            null              2            0.4",
        "b                          1               1              2            0.4",
        "c                          0               1              1            0.2",
        "Findings (1 of 1 shown):",
        "- descriptive sentinel_values_present: Declared sentinel values occur in 2 "
        "columns: 'b' and 'c'.",
        "Limitations (1 of 1 shown):",
        "- sentinel_values_not_declared: No sentinel values were declared for 1 "
        "column, so its sentinel_count is null and its ...",
        "More: res.to_frame() has all 3 columns; res.to_frame(table='row_distribution') "
        "has the rows per number of missing",
        "    values; res.findings, res.limitations and res.to_json() have the rest.",
    ]


def test_text_orders_columns_by_missing_count_and_keeps_ties_in_order():
    df = pd.DataFrame(
        {
            "full": [1, 2, 3, 4],
            "half": [1, None, 3, None],
            "most": [None, None, None, 4.0],
            "also_half": [None, 2, None, 4],
        }
    )

    lines = str(ae.profile_missing_values(df)).splitlines()

    start = lines.index("Columns, most missing first (4 of 4 shown):")
    shown = [line.split()[0] for line in lines[start + 2 : start + 6]]
    assert shown == ["most", "half", "also_half", "full"]


def test_text_says_when_no_value_is_missing():
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    lines = str(ae.profile_missing_values(df)).splitlines()

    assert lines[1] == (
        "Missing: 0 of 4 selected values (0%). Complete rows: 2 of 2 (100%)."
    )
    assert lines[2] == "Missing values per row: 0 in 2 rows."


def test_text_names_the_first_8_numbers_of_missing_values_per_row_and_counts_the_rest():
    # Row i misses i of the 12 values.
    df = pd.DataFrame(
        {f"c{j}": [None if j < i else 1.0 for i in range(13)] for j in range(12)}
    )

    lines = str(ae.profile_missing_values(df)).splitlines()

    # The line wraps, with its continuation indented.
    assert lines[3].startswith("    ")
    assert " ".join(line.strip() for line in lines[2:4]) == (
        "Missing values per row: 0 in 1 row, 1 in 1 row, 2 in 1 row, 3 in 1 row, 4 in "
        "1 row, 5 in 1 row, 6 in 1 row, 7 in 1 row and 5 more."
    )


@pytest.mark.parametrize("neighbour", ["describe_dataset", "summarize_numeric"])
def test_nearby_analyses_point_to_it_for_missing_values(neighbour):
    doc = inspect.getdoc(getattr(ae, neighbour))
    assert doc

    assert "`profile_missing_values`" in ae.describe_analysis(neighbour).when_to_use
    assert "    profile_missing_values: " in doc.split("See Also:")[1]


def test_declares_its_statuses_and_finding_codes():
    [info] = [
        info
        for info in ae.list_analyses().analyses
        if info.name == "profile_missing_values"
    ]

    assert info.module == "agent_eda.data.missingness"
    assert info.statuses == [
        "completed",
        "insufficient_data",
        "not_applicable",
        "failed",
    ]
    assert info.finding_codes == ["all_missing_columns", "sentinel_values_present"]


def test_every_column_completes_when_the_data_has_rows():
    res = ae.profile_missing_values(gappy_frame())

    assert res.status == "completed"
    assert res.status_reason is None
    assert {entry.status for entry in res.columns.values()} == {"completed"}


def test_data_without_rows_gives_insufficient_data_zero_counts_and_null_shares():
    df = pd.DataFrame(
        {"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype=object)}
    )

    res = ae.profile_missing_values(df, sentinel_values={"b": ["N/A"]})

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_rows"
    entries = list(res.columns.values())
    assert [
        (e.native_missing_count, e.sentinel_count, e.missing_count, e.missing_share)
        for e in entries
    ] == [(0, None, 0, None), (0, 0, 0, None)]
    assert [e.status for e in entries] == ["insufficient_data"] * 2
    assert (res.missing_count, res.missing_share) == (0, None)
    assert (res.complete_row_count, res.complete_row_share) == (0, None)
    assert res.row_distribution == []
    [limitation] = [lim for lim in res.limitations if lim.code == "too_few_values"]
    assert limitation.reason == "The data has no rows, so every share is null."
    assert limitation.fields == [
        "missing_share",
        "complete_row_share",
        "columns['a'].missing_share",
        "columns['b'].missing_share",
    ]
    record = json.loads(res.to_json())
    assert record["missing_share"] is None
    assert record["columns"]["a"]["missing_share"] is None


def test_data_without_columns_is_not_applicable():
    res = ae.profile_missing_values(pd.DataFrame(index=range(3)))

    assert res.status == "not_applicable"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_columns"
    assert res.columns == {}
    # Every row misses none of no values.
    assert (res.complete_row_count, res.complete_row_share) == (3, 1.0)
    assert (res.missing_count, res.missing_share) == (0, None)
    [limitation] = res.limitations
    assert limitation.code == "too_few_values"
    assert limitation.reason == "The data has no columns, so missing_share is null."
    assert limitation.fields == ["missing_share"]


def pandas_missing(column: pd.Series) -> int:
    """What isna() counts, plus the NaN that an extension float column stores."""
    stored = 0
    if isinstance(column.dtype, pd.api.extensions.ExtensionDtype) and (
        column.dtype.kind == "f"
    ):
        floats = column.to_numpy(dtype="float64", na_value=np.nan)
        stored = int(np.isnan(floats).sum()) - int(column.isna().sum())
    return int(column.isna().sum()) + stored


@pytest.mark.parametrize("frame", ["every kind", "one row", "all missing", "constant"])
def test_native_missing_counts_match_pandas_isna_for_every_dtype(frame):
    df = EDGE_FRAMES[frame]()

    res = ae.profile_missing_values(df)

    counted = {name: entry.native_missing_count for name, entry in res.columns.items()}
    assert counted == {str(name): pandas_missing(df[name]) for name in df.columns}


@given(df=data_frames())
def test_counts_per_column_and_per_row_agree_with_pandas(df):
    res = ae.profile_missing_values(df)

    missing = df.isna().to_numpy()
    assert [e.native_missing_count for e in res.columns.values()] == [
        int(n) for n in missing.sum(axis=0)
    ]
    per_row = missing.sum(axis=1)
    counts, rows = np.unique(per_row, return_counts=True)
    assert [(e.missing_count, e.row_count) for e in res.row_distribution] == [
        (int(k), int(n)) for k, n in zip(counts, rows, strict=True)
    ]
    assert res.complete_row_count == int((per_row == 0).sum())
    assert res.missing_count == int(missing.sum())


@given(data=st.data())
def test_each_declared_value_counts_the_values_equal_to_it(data):
    df = data.draw(data_frames(), "data")
    arguments = data.draw(missing_value_arguments(df), "arguments")

    res = ae.profile_missing_values(df, **arguments)

    labels = {str(label): label for label in df.columns}
    for name, values in (arguments.get("sentinel_values") or {}).items():
        column = df[labels[name]].tolist()
        counted = res.columns[name].sentinel_values
        assert [entry.count for entry in counted] == [
            sum(1 for item in column if equal_as_declared(item, value))
            for value in values
        ]


def equal_as_declared(item, value):
    """Whether a column's value matches a declared one: Python's ==, but a
    boolean only ever equals a boolean, a decimal compares with a float as the
    decimal it reads as, and nothing missing matches."""
    if item is None or item is pd.NA or item is pd.NaT:
        return False
    if isinstance(item, float) and np.isnan(item):
        return False
    if isinstance(item, bool | np.bool_) != isinstance(value, bool):
        return False
    if isinstance(item, Decimal):
        if isinstance(value, float):
            return item == Decimal(repr(value))
        return isinstance(value, int) and item == value
    numbers = isinstance(item, int | float) and isinstance(value, int | float)
    # Numbers of different types still compare by value, as -999 == -999.0.
    if not (numbers or isinstance(item, type(value)) or isinstance(value, type(item))):
        return False
    return bool(item == value)


def test_nan_stored_as_a_value_is_native_missing_and_a_limitation_says_so():
    stored = pa.array([1.5, float("nan"), None, float("nan")], from_pandas=False)
    df = pd.DataFrame(
        {
            "x": pd.Series(pd.arrays.ArrowExtensionArray(stored)),
            "y": [1.5, np.nan, None, 2.0],
        }
    )

    res = ae.profile_missing_values(df)

    # pandas' isna() counts only the null; the analysis counts the NaN too.
    assert df["x"].isna().sum() == 1
    assert res.columns["x"].native_missing_count == 3
    assert res.columns["x"].missing_count == 3
    [limitation] = [
        lim for lim in res.limitations if lim.code == "nan_counted_as_missing"
    ]
    assert limitation.fields == ["columns['x'].native_missing_count"]
    assert limitation.reason == (
        "1 column stores NaN as a value, which pandas doesn't count as "
        "missing; native_missing_count counts it: 'x'."
    )


@pytest.mark.parametrize(
    "values",
    [
        arrow([Decimal("2379.95"), Decimal("0.10"), None], pa.decimal128(10, 2)),
        pd.Series([Decimal("2379.95"), Decimal("0.10"), None], dtype=object),
    ],
    ids=["pyarrow decimal", "object decimal"],
)
def test_a_decimal_matches_a_declared_number_as_written(values):
    # Read as a float, Decimal('2379.95') is 2379.9500000000003 to pyarrow and
    # isn't equal to the float 2379.95 to Python; compared as decimals, it is.
    df = pd.DataFrame({"amount": values})

    res = ae.profile_missing_values(df, sentinel_values={"amount": [2379.95, 0.1]})

    counted = res.columns["amount"].sentinel_values
    assert [(v.value, v.count) for v in counted] == [(2379.95, 1), (0.1, 1)]


def test_a_decimal_column_matches_declared_numbers():
    decimals = pa.array(
        [Decimal("-999.00"), Decimal("1.50"), None], pa.decimal128(10, 2)
    )
    df = pd.DataFrame({"amount": pd.Series(pd.arrays.ArrowExtensionArray(decimals))})

    res = ae.profile_missing_values(df, sentinel_values={"amount": [-999, 1.5]})

    amount = res.columns["amount"]
    assert [(v.value, v.count) for v in amount.sentinel_values] == [(-999, 1), (1.5, 1)]
    assert (amount.native_missing_count, amount.missing_count) == (1, 3)


@pytest.mark.parametrize(
    ("values", "declared", "count"),
    [
        (pd.Series([1.5, -999.0, None], dtype="float32"), -999, 1),
        (pd.Series([0.1, -999.0, None], dtype="float32"), 0.1, 1),
        (pd.Series([1, -999, None], dtype="Int16"), -999.0, 1),
        (pd.Series([1, 255, None], dtype="uint8[pyarrow]"), 255, 1),
        (pd.Series(pd.arrays.SparseArray([0.0, -999.0, np.nan])), -999, 1),
        (pd.Series([True, None, False], dtype="bool[pyarrow]"), False, 1),
        (pd.Series(["N/A", None, "n/a"], dtype=STR), "N/A", 1),
        (arrow(["N/A", None, "x"], pa.large_string()), "N/A", 1),
        (pd.Series(pd.Categorical([1, 2, None], categories=[1, 2, 3])), 2.0, 1),
        (
            pd.Series(pd.Categorical(["N/A", "a", None], categories=["a", "N/A", "b"])),
            "N/A",
            1,
        ),
        (
            NAIVE.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin"),
            pd.Timestamp("1900-01-01", tz="UTC"),
            1,
        ),
        (
            arrow([dt.datetime(1900, 1, 1), None], pa.timestamp("us")),
            dt.datetime(1900, 1, 1),
            1,
        ),
        (arrow([dt.date(1900, 1, 1), None], pa.date32()), dt.date(1900, 1, 1), 1),
        (
            arrow([dt.timedelta(days=-1), None], pa.duration("us")),
            pd.Timedelta(days=-1),
            1,
        ),
    ],
    ids=lambda case: str(getattr(case, "dtype", case)),
)
def test_declared_values_match_in_every_dtype(values, declared, count):
    df = pd.DataFrame({"x": values})

    res = ae.profile_missing_values(df, sentinel_values={"x": [declared]})

    assert res.columns["x"].sentinel_count == count
    assert res.columns["x"].native_missing_count == values.isna().sum()


# Text dtypes: object, python and pyarrow strings, and what pandas infers for
# text (object on pandas 2, str on pandas 3).
TEXT_DTYPES = [object, "string", "string[pyarrow]", None]


@pytest.mark.parametrize("dtype", TEXT_DTYPES)
def test_empty_strings_count_as_missing_only_when_declared(dtype):
    df = pd.DataFrame({"note": pd.Series(["ok", "", "  ", None, ""], dtype=dtype)})

    undeclared = ae.profile_missing_values(df).columns["note"]
    declared = ae.profile_missing_values(df, sentinel_values={"note": [""]})

    assert (undeclared.native_missing_count, undeclared.missing_count) == (1, 1)
    note = declared.columns["note"]
    # Only the empty strings: whitespace counts only when declared itself.
    assert (note.native_missing_count, note.sentinel_count, note.missing_count) == (
        1,
        2,
        3,
    )


def test_without_declared_values_sentinel_counts_are_null_and_a_limitation_says_so():
    df = pd.DataFrame({"age": [34, -999, None], "city": ["Lyon", None, "N/A"]})

    res = ae.profile_missing_values(df)

    assert [entry.sentinel_count for entry in res.columns.values()] == [None, None]
    assert res.parameters["sentinel_values"] is None
    [limitation] = [
        lim for lim in res.limitations if lim.code == "sentinel_values_not_declared"
    ]
    assert limitation.fields == [
        "columns['age'].sentinel_count",
        "columns['city'].sentinel_count",
    ]
    assert limitation.reason == (
        "No sentinel values were declared, so sentinel_count is null and "
        "missing_count counts only native missing values. Declare stand-ins for "
        "missing values, such as -999 or 'N/A', in sentinel_values."
    )


def test_a_declared_value_that_never_occurs_counts_zero():
    df = pd.DataFrame({"age": [34, -999, None], "city": ["Lyon", None, "N/A"]})

    res = ae.profile_missing_values(df, sentinel_values={"age": [-1]})

    assert res.columns["age"].sentinel_count == 0
    assert res.columns["city"].sentinel_count is None
    assert res.parameters["sentinel_values"] == {"age": [-1]}
    [limitation] = [
        lim for lim in res.limitations if lim.code == "sentinel_values_not_declared"
    ]
    assert limitation.fields == ["columns['city'].sentinel_count"]
    assert limitation.reason == (
        "No sentinel values were declared for 1 column, so its sentinel_count is "
        "null and its missing_count counts only native missing values: 'city'."
    )


def test_each_column_lists_its_declared_values_and_how_often_each_occurs():
    df = pd.DataFrame(
        {
            "code": pd.Series(["N/A", "", "N/A", "ok", None], dtype=object),
            "born": pd.to_datetime(
                ["1900-01-01", "1984-03-02", None, "1900-01-01", "1971-07-30"]
            ),
            "other": [1, 2, 3, 4, 5],
        }
    )

    res = ae.profile_missing_values(
        df,
        sentinel_values={"code": ["N/A", "", "?"], "born": [dt.datetime(1900, 1, 1)]},
    )

    code, born, other = res.columns.values()
    assert [(v.value, v.count) for v in code.sentinel_values] == [
        ("N/A", 2),
        ("", 1),
        ("?", 0),
    ]
    assert [(v.value, v.count) for v in born.sentinel_values] == [
        ("1900-01-01T00:00:00", 2)
    ]
    assert other.sentinel_values == []
    # The record shows dates and times as ISO 8601 text.
    record = res.to_dict()
    assert record["columns"]["born"]["sentinel_values"] == [
        {"value": "1900-01-01T00:00:00", "count": 2}
    ]
    assert record["parameters"]["sentinel_values"] == {
        "code": ["N/A", "", "?"],
        "born": ["1900-01-01T00:00:00"],
    }


def test_text_declared_for_an_integer_column_raises_naming_the_number():
    df = pd.DataFrame({"age": [34, -999, 29]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"age": ["-999"]})

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "profile_missing_values got '-999' for column 'age' in sentinel_values, but "
        "'age' holds integers (dtype int64), so it can never match. Did you mean "
        "-999? Declare each value as the type its column holds, such as "
        "sentinel_values={'age': [-999]}."
    )


# A column, a declared value of the wrong type, how the error describes the
# column, and the value it suggests instead.
WRONG_TYPES = [
    (pd.Series(["a", "-999"], dtype="string"), -999, "text (dtype string)", "'-999'"),
    (pd.Series(["a", "-999"], dtype=object), -999.0, "text (dtype object)", "'-999'"),
    (pd.Series(["a"], dtype=object), 2.5, "text (dtype object)", "'2.5'"),
    (pd.Series(["Yes"], dtype=object), True, "text (dtype object)", "'True'"),
    (
        pd.Series(["1900-01-01"], dtype=object),
        pd.Timestamp("1900-01-01"),
        "text (dtype object)",
        "'1900-01-01'",
    ),
    (pd.Series([1.5, None]), "-999.5", "numbers (dtype float64)", "-999.5"),
    (pd.Series([1, None], dtype="Int64"), " -999 ", "integers (dtype Int64)", "-999"),
    (pd.Series([1, 2], dtype="int8"), "-99.0", "integers (dtype int8)", "-99"),
    (pd.Series([1.5]), True, "numbers (dtype float64)", "1"),
    (
        pd.Series([True, None], dtype="boolean"),
        "false",
        "booleans (dtype boolean)",
        "False",
    ),
    (pd.Series([True, False]), 1, "booleans (dtype bool)", "True"),
    (
        NAIVE,
        "1900-01-01",
        "datetimes without a time zone (dtype datetime64[ns])",
        "pd.Timestamp('1900-01-01 00:00:00')",
    ),
    (
        NAIVE,
        dt.date(1900, 1, 1),
        "datetimes without a time zone (dtype datetime64[ns])",
        "pd.Timestamp('1900-01-01 00:00:00')",
    ),
    (
        NAIVE.dt.tz_localize("UTC"),
        "1900-01-01",
        "datetimes in time zone UTC (dtype datetime64[ns, UTC])",
        "pd.Timestamp('1900-01-01 00:00:00+0000', tz='UTC')",
    ),
    (
        arrow([dt.date(1900, 1, 1), None], pa.date32()),
        pd.Timestamp("1900-01-01"),
        "dates (dtype date32[day][pyarrow])",
        "datetime.date(1900, 1, 1)",
    ),
    (
        arrow([dt.date(1900, 1, 1), None], pa.date32()),
        "1900-01-01",
        "dates (dtype date32[day][pyarrow])",
        "datetime.date(1900, 1, 1)",
    ),
    (
        pd.Series(pd.to_timedelta(["1 day", None])).astype("timedelta64[ns]"),
        "1 day",
        "durations (dtype timedelta64[ns])",
        "pd.Timedelta('1 days 00:00:00')",
    ),
    (
        pd.Series(pd.Categorical(["a", "-1", None])),
        -1,
        "categories of text (dtype category)",
        "'-1'",
    ),
    (
        pd.Series(pd.Categorical([3, 1, None])),
        "3",
        "categories of integers (dtype category)",
        "3",
    ),
]


@pytest.mark.parametrize(
    ("values", "declared", "holds", "likely"),
    WRONG_TYPES,
    ids=[f"{case[2]}-{case[1]!r}" for case in WRONG_TYPES],
)
def test_a_value_of_another_type_raises_naming_the_value_of_the_columns_type(
    values, declared, holds, likely
):
    df = pd.DataFrame({"x": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [declared]})

    message = str(caught.value)
    assert f"but 'x' holds {holds}, so it can never match. " in message
    assert f"Did you mean {likely}? " in message
    assert message.endswith(f"such as sentinel_values={{'x': [{likely}]}}.")


@pytest.mark.parametrize(
    ("values", "declared", "holds"),
    [
        (pd.Series([1.5, None]), "N/A", "numbers (dtype float64)"),
        (pd.Series([1, 2]), dt.date(1900, 1, 1), "integers (dtype int64)"),
        (pd.Series(["a"], dtype="string"), dt.timedelta(days=1), "text (dtype string)"),
    ],
)
def test_a_value_of_another_type_without_a_likely_meaning_raises_without_a_guess(
    values, declared, holds
):
    df = pd.DataFrame({"x": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [declared]})

    assert str(caught.value) == (
        f"profile_missing_values got {declared!r} for column 'x' in sentinel_values, "
        f"but 'x' holds {holds}, so it can never match. Declare each value as the "
        "type its column holds."
    )


# A column, a declared value its dtype can't hold, and how the error says why.
OUT_OF_REACH = [
    (pd.Series([1, 2], dtype="uint8"), 300, "integers from 0 to 255 (dtype uint8)"),
    (pd.Series([1, None], dtype="UInt8"), -1, "integers from 0 to 255 (dtype UInt8)"),
    (
        pd.Series([1, None], dtype="int8[pyarrow]"),
        -999,
        "integers from -128 to 127 (dtype int8[pyarrow])",
    ),
    (
        pd.Series([1, 2]),
        2**63,
        "integers from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 "
        "(dtype int64)",
    ),
    (pd.Series([1, 2]), -999.5, "integers (dtype int64)"),
    (
        pd.Series([0.5], dtype="float32"),
        1e300,
        "numbers from -3.403e+38 to 3.403e+38 (dtype float32)",
    ),
    (
        NAIVE,
        dt.datetime(9999, 12, 31),
        "datetimes from 1677-09-21 to 2262-04-11 (dtype datetime64[ns])",
    ),
    (
        NAIVE.astype("datetime64[s]"),
        pd.Timestamp("2020-01-01 00:00:00.5"),
        "datetimes in whole seconds (dtype datetime64[s])",
    ),
    (
        pd.Series(pd.to_timedelta(["1 day", None])).astype("timedelta64[ns]"),
        dt.timedelta(days=999_999),
        "durations within ±106,751 days (dtype timedelta64[ns])",
    ),
]


@pytest.mark.parametrize(
    ("values", "declared", "holds"),
    OUT_OF_REACH,
    ids=[f"{case[2]}-{case[1]!r}" for case in OUT_OF_REACH],
)
def test_a_value_the_columns_dtype_cannot_hold_raises(values, declared, holds):
    df = pd.DataFrame({"x": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [declared]})

    assert str(caught.value) == (
        f"profile_missing_values got {literal(declared)} for column 'x' in "
        f"sentinel_values, but 'x' holds {holds}, so it can never match. Leave it "
        "out of sentinel_values, or declare a value the column can hold."
    )


@pytest.mark.parametrize(
    ("values", "declared", "holds", "likely"),
    [
        (
            NAIVE.dt.tz_localize("UTC"),
            pd.Timestamp("1900-01-01"),
            "datetimes in time zone UTC (dtype datetime64[ns, UTC])",
            "pd.Timestamp('1900-01-01 00:00:00+0000', tz='UTC')",
        ),
        (
            NAIVE,
            pd.Timestamp("1900-01-01", tz="Europe/Berlin"),
            "datetimes without a time zone (dtype datetime64[ns])",
            "pd.Timestamp('1900-01-01 00:00:00')",
        ),
        (
            arrow([dt.datetime(1900, 1, 1), None], pa.timestamp("us", tz="UTC")),
            dt.datetime(1900, 1, 1),
            "datetimes in time zone UTC (dtype timestamp[us, tz=UTC][pyarrow])",
            "pd.Timestamp('1900-01-01 00:00:00+0000', tz='UTC')",
        ),
    ],
)
def test_a_datetime_with_or_without_a_time_zone_unlike_its_column_raises(
    values, declared, holds, likely
):
    df = pd.DataFrame({"x": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [declared]})

    assert str(caught.value) == (
        f"profile_missing_values got {literal(declared)} for column 'x' in "
        f"sentinel_values, but 'x' holds {holds}, so it can never match. Did you mean "
        f"{likely}? Declare each value as the type its column holds, such as "
        f"sentinel_values={{'x': [{likely}]}}."
    )


@pytest.mark.parametrize(
    ("values", "valid", "declared"),
    [
        (pd.Series([1.5, None]), -1, float("nan")),
        (pd.Series([1.5, None], dtype="double[pyarrow]"), -1, float("nan")),
        (NAIVE, pd.Timestamp("1900-01-01"), pd.NaT),
    ],
)
def test_a_missing_value_declared_as_a_sentinel_value_raises(values, valid, declared):
    df = pd.DataFrame({"x": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [valid, declared]})

    assert str(caught.value) == (
        f"profile_missing_values got {declared!r} for column 'x' in sentinel_values, "
        f"but {declared!r} is a missing value itself, not a stand-in for one. "
        "Declare only stand-ins for missing values, such as -999 or 'N/A'."
    )


def test_an_infinite_value_declared_as_a_sentinel_value_raises():
    df = pd.DataFrame({"x": [1.5, np.inf]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"x": [-np.inf]})

    assert str(caught.value) == (
        "profile_missing_values got -inf for column 'x' in sentinel_values, but -inf "
        "is an infinite value, not a stand-in for a missing one. Replace infinite "
        "values with NaN first if they mean missing, such as "
        "df.replace([np.inf, -np.inf], np.nan)."
    )


def test_a_column_whose_values_no_sentinel_value_can_equal_raises():
    df = pd.DataFrame({"tags": arrow([[1, 2], None], pa.list_(pa.int64()))})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"tags": ["[]"]})

    assert str(caught.value) == (
        "profile_missing_values got '[]' for column 'tags' in sentinel_values, but "
        "'tags' holds values no sentinel value can equal (dtype "
        "list<item: int64>[pyarrow]), so it can never match. Leave it out of "
        "sentinel_values, or declare a value the column can hold."
    )


def test_every_value_that_can_never_match_is_named_in_one_error():
    df = pd.DataFrame(
        {
            "age": [34, 51, -999],
            "city": pd.Series(["Lyon", "-1", None], dtype=object),
            "n": pd.Series([1, 2, 3], dtype="uint8"),
            "score": [0.5, None, 0.25],
        }
    )

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(
            df,
            sentinel_values={
                "age": [-999, "-999"],
                "city": ["N/A", -1],
                "n": [999],
                "score": [np.nan],
            },
        )

    assert str(caught.value) == (
        "profile_missing_values can't use 4 values in sentinel_values: '-999' for "
        "'age', which holds integers (dtype int64; did you mean -999?); -1 for "
        "'city', which holds text (dtype object; did you mean '-1'?); 999 for 'n', "
        "which holds integers from 0 to 255 (dtype uint8); nan for 'score', a missing "
        "value itself. Declare each value as the type its column holds. Leave out the "
        "values their columns can't hold. Declare only stand-ins for missing values, "
        "such as -999 or 'N/A'."
    )


def test_an_error_names_the_first_5_values_it_cannot_use_and_counts_the_rest():
    df = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(
            df, sentinel_values={"x": ["a", "b", "c", "d", "e", "f", "g"]}
        )

    message = str(caught.value)
    assert message.startswith(
        "profile_missing_values can't use 7 values in sentinel_values: 'a' for 'x', "
    )
    assert "'e' for 'x', which holds integers (dtype int64); and 2 more. " in message
    assert "'f'" not in message


def test_a_declared_value_matches_only_values_of_its_kind_in_an_object_column():
    # Python's == says True == 1 and 1 == 1.0; a number never matches a boolean.
    df = pd.DataFrame(
        {"x": pd.Series([-999, -999.0, "-999", True, 1, None, False], dtype=object)}
    )

    res = ae.profile_missing_values(df, sentinel_values={"x": [-999, "-999", True]})

    x = res.columns["x"]
    assert [(v.value, v.count) for v in x.sentinel_values] == [
        (-999, 2),
        ("-999", 1),
        (True, 1),
    ]
    assert (x.native_missing_count, x.sentinel_count, x.missing_count) == (1, 4, 5)


def test_an_object_column_holds_what_its_values_are():
    numbers = pd.DataFrame({"x": pd.Series([1, 2.5, None], dtype=object)})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(numbers, sentinel_values={"x": ["-999"]})

    assert "but 'x' holds numbers (dtype object), so it can never match. Did you " in (
        str(caught.value)
    )
    assert "Did you mean -999? " in str(caught.value)


def test_an_object_column_without_values_takes_any_declared_value():
    df = pd.DataFrame({"x": pd.Series([None, np.nan], dtype=object)})

    res = ae.profile_missing_values(df, sentinel_values={"x": [-999, "N/A"]})

    assert res.columns["x"].sentinel_count == 0
    assert res.columns["x"].missing_count == 2


def test_an_object_column_holding_unhashable_or_odd_values_is_matched_safely():
    df = pd.DataFrame(
        {
            "x": pd.Series(
                [[1, 2], {"k": 1}, np.array([1, 2]), "N/A", pd.NA, "N/A"],
                dtype=object,
            )
        }
    )

    res = ae.profile_missing_values(df, sentinel_values={"x": ["N/A"]})

    assert res.columns["x"].sentinel_count == 2
    assert res.columns["x"].native_missing_count == 1


def test_an_unknown_column_in_sentinel_values_raises_with_the_closest_names():
    df = pd.DataFrame({"age": [34, -999], "city": ["Lyon", "N/A"]})

    with pytest.raises(ColumnNotFoundError) as caught:
        ae.profile_missing_values(df, sentinel_values={"Age": [-999]})

    assert str(caught.value) == (
        "profile_missing_values got an unknown column 'Age' in sentinel_values. Did "
        "you mean 'age'? The data has 2 columns: 'age' and 'city'."
    )


def test_sentinel_values_for_columns_the_call_leaves_out_raise():
    df = pd.DataFrame({"age": [34, -999], "city": ["Lyon", "N/A"], "n": [1, 2]})

    with pytest.raises(InvalidArgumentError) as one:
        ae.profile_missing_values(df, columns=["city"], sentinel_values={"age": [-999]})
    with pytest.raises(InvalidArgumentError) as two:
        ae.profile_missing_values(
            df, columns=["city"], sentinel_values={"age": [-999], "n": [0]}
        )

    assert str(one.value) == (
        "profile_missing_values got sentinel values for 'age', which columns leaves "
        "out. Add it to columns, or leave it out of sentinel_values."
    )
    assert str(two.value) == (
        "profile_missing_values got sentinel values for 2 columns that columns leaves "
        "out: 'age' and 'n'. Add them to columns, or leave them out of "
        "sentinel_values."
    )


def test_a_value_declared_twice_for_a_column_raises():
    df = pd.DataFrame({"age": [34, -999]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"age": [-999, -1, -999.0]})

    assert str(caught.value) == (
        "profile_missing_values got -999 more than once in sentinel_values['age']. "
        "List each value once."
    )


@pytest.mark.parametrize(
    "declared",
    [-999, np.int64(-999), (-999,), np.array([-999]), pd.Series([-999]), [-999.0]],
    ids=["int", "numpy int", "tuple", "numpy array", "series", "float"],
)
def test_sentinel_values_take_one_value_or_any_list_of_values(declared):
    df = pd.DataFrame({"age": [34, -999, 51]})

    res = ae.profile_missing_values(df, sentinel_values={"age": declared})

    assert res.columns["age"].sentinel_count == 1


def test_numpy_datetimes_and_timedeltas_are_read_as_pandas_ones():
    df = pd.DataFrame(
        {
            "when": NAIVE,
            "took": pd.Series(pd.to_timedelta(["1 day", None])),
        }
    )

    res = ae.profile_missing_values(
        df,
        sentinel_values={  # ty: ignore[invalid-argument-type]  # NumPy scalars are accepted at runtime
            "when": np.datetime64("1900-01-01"),
            "took": [np.timedelta64(1, "D")],
        },
    )

    assert res.columns["when"].sentinel_count == 1
    assert res.columns["took"].sentinel_count == 1
    assert res.parameters["sentinel_values"] == {
        "when": ["1900-01-01T00:00:00"],
        "took": ["P1DT0H0M0S"],
    }


def test_sentinel_values_name_columns_by_label_or_its_string_form():
    df = pd.DataFrame({0: [34, -999], 1: ["Lyon", "N/A"]})

    res = ae.profile_missing_values(df, sentinel_values={0: [-999], "1": ["N/A"]})  # ty: ignore[invalid-argument-type]  # labels that aren't strings are accepted at runtime

    assert [entry.sentinel_count for entry in res.columns.values()] == [1, 1]
    assert res.parameters["sentinel_values"] == {"0": [-999], "1": ["N/A"]}


def test_a_column_named_twice_in_sentinel_values_raises():
    df = pd.DataFrame({0: [34, -999]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={0: [-999], "0": [-1]})  # ty: ignore[invalid-argument-type]  # the clashing labels are the test

    assert str(caught.value) == (
        "profile_missing_values got column '0' more than once in sentinel_values, as "
        "0 and '0'. Name each column once."
    )


@pytest.mark.parametrize(
    ("declared", "shown"),
    [
        (None, "None"),
        ([1, 2], "a list of 2 items"),
        (dt.time(0, 0), "a time"),
        (pd.NA, "a NAType"),
    ],
)
def test_a_value_of_another_python_type_raises_with_the_types_allowed(declared, shown):
    df = pd.DataFrame({"age": [34, -999]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values={"age": [-1, declared]})

    assert str(caught.value) == (
        f"profile_missing_values got {shown} at sentinel_values['age'][1], but each "
        "value in sentinel_values must be a boolean, an integer, a number, a "
        "string, a datetime, a date or a timedelta. See "
        "help(ae.profile_missing_values) for every argument."
    )


def test_sentinel_values_that_are_no_mapping_raise_with_the_type_allowed():
    df = pd.DataFrame({"age": [34, -999]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_missing_values(df, sentinel_values=[-999])  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert str(caught.value) == (
        "profile_missing_values got a list of 1 item for sentinel_values, but "
        "sentinel_values must be a mapping from column names to a list of "
        "booleans, integers, numbers, strings, datetimes, dates or timedeltas, or "
        "None. See help(ae.profile_missing_values) for every argument."
    )


def literal(value):
    """A value as Python code, the way messages show it."""
    text = repr(value)
    return f"pd.{text}" if isinstance(value, pd.Timestamp | pd.Timedelta) else text


def test_the_record_keeps_100_columns_of_declared_values_and_their_total():
    df = pd.DataFrame({f"c{i}": [i, -999] for i in range(120)})

    res = ae.profile_missing_values(
        df, sentinel_values={f"c{i}": [-999] for i in range(120)}
    )

    record = res.to_dict()
    declared = record["parameters"]["sentinel_values"]
    assert list(declared) == [f"c{i}" for i in range(100)]
    assert record["parameters"]["sentinel_values_total"] == 120
    kept = res.parameters["sentinel_values"]
    assert isinstance(kept, dict)
    assert len(kept) == 120
    assert record["columns_total"] == 120


def test_an_empty_list_declares_nothing_for_its_column():
    df = pd.DataFrame({"age": [34, -999, None]})

    res = ae.profile_missing_values(df, sentinel_values={"age": []})

    assert res.columns["age"].sentinel_count is None
    assert [lim.code for lim in res.limitations] == ["sentinel_values_not_declared"]


def test_every_column_with_declared_values_drops_the_limitation():
    df = pd.DataFrame({"age": [34, -999, None]})

    res = ae.profile_missing_values(df, sentinel_values={"age": [-999]})

    assert [lim.code for lim in res.limitations] == []


def contacts():
    """Five contacts; email is sensitive and stands in 'N/A' and '' for unknown."""
    return pd.DataFrame(
        {
            "email": ["ann@x.org", "N/A", "", "N/A", "bob@x.org"],
            "age": [34, -999, 29, -999, 51],
        }
    )


def masking_project(*columns):
    return ae.Project(
        data_policy=ae.DataPolicy(sensitive_columns=list(columns), mask_sensitive=True)
    )


def everything_shown(res):
    """The printed text, the record and both tables, as one string."""
    return (
        str(res)
        + res.to_json()
        + res.to_frame().to_csv()
        + res.to_frame(table="row_distribution").to_csv()
    )


def test_masking_puts_placeholders_in_place_of_declared_values_of_sensitive_columns():
    res = ae.profile_missing_values(
        contacts(),
        sentinel_values={"email": ["N/A", ""], "age": [-999]},
        project=masking_project("email"),
    )

    assert res.parameters["sentinel_values"] == {
        "email": ["<masked 1>", "<masked 2>"],
        "age": [-999],
    }
    assert [(v.value, v.count) for v in res.columns["email"].sentinel_values] == [
        ("<masked 1>", 2),
        ("<masked 2>", 1),
    ]
    assert res.columns["age"].sentinel_values[0].value == -999
    assert res.to_frame()["sentinel_values"].tolist()[0] == {
        "<masked 1>": 2,
        "<masked 2>": 1,
    }
    assert "N/A" not in everything_shown(res)
    # Counts and shares stay.
    assert res.columns["email"].sentinel_count == 3
    assert res.masking.applied is True
    assert res.masking.columns == ["email"]


def test_masking_off_shows_declared_values_as_declared_and_says_so():
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["email"]))

    res = ae.profile_missing_values(
        contacts(), sentinel_values={"email": ["N/A"]}, project=project
    )

    assert res.parameters["sentinel_values"] == {"email": ["N/A"]}
    assert res.masking.applied is False
    assert res.masking.columns == ["email"]
    assert "Not masked (data_policy.mask_sensitive=False)" in str(res)


def test_a_call_asks_for_masking_without_a_project_that_declares_columns():
    with pytest.raises(InvalidArgumentError, match="mask_sensitive=True"):
        ae.profile_missing_values(contacts(), mask_sensitive=True)


def test_masking_is_recorded_and_discovery_lists_the_analysis_as_masking():
    description = ae.describe_analysis("profile_missing_values")
    [info] = [
        i for i in ae.list_analyses().analyses if i.name == "profile_missing_values"
    ]

    res = ae.profile_missing_values(contacts(), project=masking_project("age"))

    assert res.masking.decided_by == "project"
    assert res.to_dict()["parameters"]["project"] == {
        "name": None,
        "data_policy": {"sensitive_columns": ["age"], "mask_sensitive": True},
    }
    assert description.supports_masking is True
    assert description.shows_raw_values is True
    assert info.supports_masking is True
