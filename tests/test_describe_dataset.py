import inspect

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given
from strategies import data_frames, nan_as_missing

import agent_eda as ae


def test_reports_row_and_column_counts():
    df = pd.DataFrame({"age": [34, 51, 29], "city": ["Lyon", "Oslo", None]})

    res = ae.describe_dataset(df)

    assert res.row_count == 3
    assert res.column_count == 2


def test_reports_dtype_counts_and_memory_per_column():
    df = pd.DataFrame({"age": [34, 51, 29, 51], "score": [0.5, None, 0.75, 0.5]})

    res = ae.describe_dataset(df)

    rows = [
        (
            c.position,
            c.column,
            c.dtype,
            c.non_missing_count,
            c.missing_count,
            c.distinct_count,
            c.memory_bytes,
        )
        for c in res.columns
    ]
    assert rows == [
        (0, "age", "int64", 4, 0, 3, 32),
        (1, "score", "float64", 3, 1, 2, 32),
    ]


MIXED_COLUMNS = {
    # name: (values, expected non-missing, missing and distinct counts)
    "int64": ([1, 2, 2, 3], (4, 0, 3)),
    "int8": (np.array([1, 1, 2, 3], dtype=np.int8), (4, 0, 3)),
    "float32": (np.array([1.5, np.nan, 1.5, 2.5], dtype=np.float32), (3, 1, 2)),
    "bool": ([True, False, True, True], (4, 0, 2)),
    "datetime": (
        pd.to_datetime(["2024-01-01", None, "2024-01-01", "2024-01-02"]),
        (3, 1, 2),
    ),
    "datetime_tz": (
        pd.to_datetime(["2024-01-01", None, "2024-01-01", "2024-01-02"]).tz_localize(
            "Europe/Paris"
        ),
        (3, 1, 2),
    ),
    "timedelta": (pd.to_timedelta(["1D", None, "2D", "1D"]), (3, 1, 2)),
    "Int64": (pd.array([1, None, 1, 2], dtype="Int64"), (3, 1, 2)),
    "Float64": (pd.array([0.5, None, 0.5, 1.5], dtype="Float64"), (3, 1, 2)),
    "boolean": (pd.array([True, None, False, True], dtype="boolean"), (3, 1, 2)),
    "string": (pd.array(["a", None, "a", "b"], dtype="string"), (3, 1, 2)),
    "default_text": (pd.Series(["a", None, "a", "b"]), (3, 1, 2)),
    "category": (
        pd.Categorical(["x", None, "x", "y"], categories=["x", "y", "unused"]),
        (3, 1, 2),
    ),
    "object": (pd.Series([1, "a", None, 1], dtype=object), (3, 1, 2)),
    "int64_pyarrow": (pd.array([1, None, 1, 2], dtype="int64[pyarrow]"), (3, 1, 2)),
    "string_pyarrow": (
        pd.array(["a", None, "a", "b"], dtype="string[pyarrow]"),
        (3, 1, 2),
    ),
    "bool_pyarrow": (
        pd.array([True, None, True, False], dtype="bool[pyarrow]"),
        (3, 1, 2),
    ),
    "timestamp_pyarrow": (
        pd.array(
            pd.to_datetime(["2024-01-01", None, "2024-01-01", "2024-01-02"]),
            dtype=pd.ArrowDtype(pa.timestamp("us", tz="UTC")),
        ),
        (3, 1, 2),
    ),
}


def test_handles_every_accepted_column_dtype():
    df = pd.DataFrame({name: values for name, (values, _) in MIXED_COLUMNS.items()})

    res = ae.describe_dataset(df)

    assert {
        c.column: (c.non_missing_count, c.missing_count, c.distinct_count)
        for c in res.columns
    } == {name: counts for name, (_, counts) in MIXED_COLUMNS.items()}
    assert [c.dtype for c in res.columns] == [str(t) for t in df.dtypes]
    assert all(c.memory_bytes > 0 for c in res.columns)
    assert res.findings == []
    assert res.limitations == []


def test_leaves_the_input_dataframe_unchanged():
    df = pd.DataFrame(
        {name: values for name, (values, _) in MIXED_COLUMNS.items()},
        index=pd.Index(["b", "a", "b", None], name="key"),
    )
    df.attrs = {"source": "sales.csv", "tags": ["raw"]}
    before = df.copy(deep=True)

    ae.describe_dataset(df, source="sales.csv")

    pd.testing.assert_frame_equal(df, before, check_exact=True)
    assert list(df.columns) == list(before.columns)
    assert df.index.equals(before.index)
    assert df.attrs == {"source": "sales.csv", "tags": ["raw"]}


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([f"row_{i}" for i in (3, 1, 2, 0, 1)]),
        pd.Index([30, 10, 20, 0, 40]),
        pd.MultiIndex.from_tuples([("b", 2), ("a", 1), ("b", 2), ("c", 0), ("a", 3)]),
    ],
    ids=["text", "integers", "multiindex"],
)
def test_builds_no_lookup_table_on_the_callers_index_or_columns(index):
    df = pd.DataFrame({"b": range(5), "a": range(5)}, index=index)
    index_memory = df.index.memory_usage(deep=True)
    columns_memory = df.columns.memory_usage(deep=True)

    ae.describe_dataset(df)

    assert df.index.memory_usage(deep=True) == index_memory
    assert df.columns.memory_usage(deep=True) == columns_memory


def test_reports_index_facts_and_total_memory():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]}, index=pd.Index([10.0, 10.0, None], dtype="float64")
    )

    res = ae.describe_dataset(df)

    assert res.index.dtype == "float64"
    assert res.index.level_count == 1
    assert res.index.is_unique is False
    assert res.index.duplicated_count == 1
    assert res.index.missing_count == 1
    assert res.index.memory_bytes == 24
    assert res.memory_bytes == 48


def test_memory_of_object_text_counts_the_objects_as_stored():
    text = pd.Series(["café", "b", None] * 100, dtype=object)
    df = pd.DataFrame({"t": text}, index=pd.Index(["x", "y", "z"] * 100, dtype=object))
    index_memory = df.index.memory_usage(deep=True)

    res = ae.describe_dataset(df)

    assert res.columns[0].dtype == "object"
    assert res.columns[0].memory_bytes == df["t"].memory_usage(deep=True, index=False)
    assert res.index.memory_bytes == index_memory


def test_memory_ignores_lookup_tables_built_by_earlier_lookups():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "c": pd.Categorical(["a", "b", "a"])},
        index=pd.Index([30, 10, 20]),
    )
    before = ae.describe_dataset(df)

    df.loc[10]
    df["c"].cat.categories.get_loc("b")
    after = ae.describe_dataset(df)

    assert after.memory_bytes == before.memory_bytes
    assert [c.memory_bytes for c in after.columns] == [
        c.memory_bytes for c in before.columns
    ]


def fresh(text):
    """A new str object, so that nothing has encoded it yet."""
    return text.encode().decode()


def test_identical_calls_agree_when_columns_share_a_non_ascii_string():
    # Found by the property tests: counting the distinct values of the second
    # column made Python cache the string's UTF-8 form, which grew the first
    # column's memory on the next call.
    shared = fresh("|û¤q[")
    df = pd.DataFrame(
        {
            "mixed": pd.Series([None, shared, "a"], dtype=object),
            "same": pd.Series([shared] * 3, dtype=object),
        }
    )

    first = ae.describe_dataset(df).to_json()
    second = ae.describe_dataset(df).to_json()

    assert second == first


@pytest.mark.parametrize(
    "texts",
    [["été", "café"], ["日本", "ŝtono"], ["🙂", "a🙂b"]],
    ids=["1-byte", "2-byte", "4-byte"],
)
def test_memory_of_text_ignores_the_utf8_copy_python_caches(texts):
    # Python keeps a UTF-8 copy inside a non-ASCII str once something encodes
    # it, as pyarrow does here, and sys.getsizeof counts that copy.
    values = [fresh(text) for text in texts] + [None]
    df = pd.DataFrame({"t": pd.Series(values, dtype=object)})
    stored = df["t"].memory_usage(deep=True, index=False)
    before = ae.describe_dataset(df).columns[0].memory_bytes

    pa.array(values)
    after = ae.describe_dataset(df).columns[0].memory_bytes

    assert df["t"].memory_usage(deep=True, index=False) > stored
    assert before == after == stored


def test_records_supplied_provenance():
    df = pd.DataFrame({"x": [1, 2]})

    res = ae.describe_dataset(df, source="s3://shop/sales.csv", source_version="v7")

    assert res.provenance.source == "s3://shop/sales.csv"
    assert res.provenance.source_version == "v7"
    assert res.provenance.not_supplied == []


def test_records_missing_provenance_as_not_supplied_without_inferring_it():
    df = pd.DataFrame({"x": [1, 2]})
    df.attrs["source"] = "sales.csv"

    res = ae.describe_dataset(df, source_version="2026-10-01")

    assert res.provenance.source is None
    assert res.provenance.source_version == "2026-10-01"
    assert res.provenance.not_supplied == ["source"]


@pytest.mark.parametrize(
    ("df", "claim"),
    [
        (pd.DataFrame({"x": pd.Series([], dtype="float64")}), "The data has no rows."),
        (pd.DataFrame(index=range(3)), "The data has no columns."),
        (pd.DataFrame(), "The data has no rows and no columns."),
    ],
    ids=["no-rows", "no-columns", "neither"],
)
def test_empty_data_completes_with_a_finding(df, claim):
    res = ae.describe_dataset(df)

    assert res.status == "completed"
    finding = only_finding(res, "empty_data")
    assert finding.claim == claim
    assert finding.assessment == "suspected"
    assert finding.fields == ["row_count", "column_count"]


def test_flags_columns_that_are_entirely_missing():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "notes": [None, None, None],
            "score": [np.nan, np.nan, np.nan],
        }
    )

    res = ae.describe_dataset(df)

    finding = only_finding(res, "all_missing_columns")
    assert finding.assessment == "suspected"
    assert finding.columns == ["notes", "score"]
    assert finding.fields == ["columns[1].missing_count", "columns[2].missing_count"]
    assert finding.claim == (
        "2 columns contain only missing values: 'notes' and 'score'."
    )


def test_claims_quote_string_labels_and_show_other_labels_as_they_are():
    df = pd.DataFrame({0: [None, None], "0": [None, None], 2.5: [1, 2]})

    res = ae.describe_dataset(df)

    assert only_finding(res, "all_missing_columns").claim == (
        "2 columns contain only missing values: 0 and '0'."
    )


def test_claims_name_five_columns_and_count_the_rest():
    df = pd.DataFrame({f"c{i}": [None, None] for i in range(8)})

    res = ae.describe_dataset(df)

    finding = only_finding(res, "all_missing_columns")
    assert finding.claim == (
        "8 columns contain only missing values: "
        "'c0', 'c1', 'c2', 'c3', 'c4' and 3 more."
    )
    assert len(finding.columns) == 8


def test_flags_constant_columns_without_showing_the_value():
    df = pd.DataFrame(
        {
            "country": ["Secretland", "Secretland", None, "Secretland"],
            "id": [1, 2, 3, 4],
            "single": [7.0, None, None, None],
        }
    )

    res = ae.describe_dataset(df)

    finding = only_finding(res, "constant_columns")
    assert finding.columns == ["country"]
    assert finding.fields == ["columns[0].distinct_count"]
    assert finding.claim == (
        "1 column holds a single distinct non-missing value: 'country'."
    )
    assert "Secretland" not in res.to_json()
    assert "Secretland" not in str(res)


def test_flags_duplicated_column_labels_instead_of_raising():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])

    res = ae.describe_dataset(df)

    finding = only_finding(res, "duplicated_column_labels")
    assert finding.assessment == "suspected"
    assert finding.columns == ["a", "a"]
    assert finding.fields == ["columns[0].column", "columns[2].column"]
    assert finding.claim == "1 column label occurs more than once: 'a'."
    assert [c.column for c in res.columns] == ["a", "b", "a"]


@pytest.mark.parametrize(
    ("labels", "duplicated"),
    [
        ([None, np.nan], False),
        ([None, None], True),
        ([np.nan, np.nan], True),
        ([1, 1.0], True),
    ],
    ids=["none-and-nan", "two-nones", "two-nans", "int-and-float"],
)
def test_column_labels_are_duplicated_exactly_when_pandas_says_so(labels, duplicated):
    columns = pd.Index(labels, dtype=object)
    df = pd.DataFrame([[1, 2]], columns=columns)

    res = ae.describe_dataset(df)

    assert ("duplicated_column_labels" in codes(res)) is duplicated
    assert columns.duplicated().any() == duplicated


def test_flags_labels_whose_string_forms_collide():
    df = pd.DataFrame([[1, 2, 3]], columns=[1, "1", "b"])

    res = ae.describe_dataset(df)

    finding = only_finding(res, "colliding_column_labels")
    assert finding.columns == ["1", "1"]
    assert finding.fields == [
        "columns[0].column",
        "columns[0].label_type",
        "columns[1].column",
        "columns[1].label_type",
    ]
    assert finding.claim == (
        "2 different column labels share a string form with another label: 1 and '1'."
    )
    assert [c.label_type for c in res.columns] == ["int", "str", "str"]
    assert "duplicated_column_labels" not in codes(res)


def test_flags_multiindex_columns():
    columns = pd.MultiIndex.from_tuples([("price", "min"), ("price", "max")])
    df = pd.DataFrame([[1.0, 2.0]], columns=columns)

    res = ae.describe_dataset(df)

    finding = only_finding(res, "multiindex_columns")
    assert finding.claim == "The column labels form a MultiIndex with 2 levels."
    assert finding.fields == ["column_level_count"]
    assert res.column_level_count == 2
    assert [c.column for c in res.columns] == [
        "('price', 'min')",
        "('price', 'max')",
    ]


def test_flags_a_non_unique_index():
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=["a", "b", "a", "a"])

    res = ae.describe_dataset(df)

    finding = only_finding(res, "non_unique_index")
    assert finding.claim == (
        "2 index labels repeat an earlier label, so index labels don't identify "
        "rows uniquely."
    )
    assert finding.fields == ["index.is_unique", "index.duplicated_count"]


def test_clean_data_has_no_findings():
    df = pd.DataFrame({"id": [1, 2, 3], "city": ["Lyon", "Oslo", "Rome"]})

    res = ae.describe_dataset(df)

    assert res.findings == []


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([["a"], ["b"], None, ["a"]]),
        pd.Series([{"k": 1}, {"k": 2}, None, {"k": 1}]),
        pd.Series([[1], [2], None, [1]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))),
    ],
    ids=["lists", "dicts", "arrow-lists"],
)
def test_unhashable_values_give_a_null_distinct_count_and_a_limitation(values):
    df = pd.DataFrame({"n": [1, 2, 3, 4], "tags": values})

    res = ae.describe_dataset(df)

    assert res.status == "completed"
    assert res.columns[1].distinct_count is None
    assert res.columns[1].missing_count == 1
    [limitation] = res.limitations
    assert limitation.code == "unhashable_values"
    assert limitation.fields == ["columns[1].distinct_count"]
    assert limitation.reason == (
        "1 column holds values that can't be hashed, such as lists or dicts, so "
        "its distinct values weren't counted: 'tags'."
    )
    assert "constant_columns" not in codes(res)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"x": pd.Series([], dtype="float64")}),
        pd.DataFrame([[1, None, 1]] * 3, columns=["a", "b", "a"], index=[0, 0, 1]),
        pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])),
        pd.DataFrame([[1, 2, [3]]], columns=[1, "1", "nested"]),
    ],
    ids=["empty", "no-rows", "duplicates", "multiindex", "collisions-and-lists"],
)
def test_results_use_only_declared_statuses_and_finding_codes(df):
    [info] = [a for a in ae.list_analyses().analyses if a.name == "describe_dataset"]

    res = ae.describe_dataset(df)

    assert res.status in info.statuses
    assert set(codes(res)) <= set(info.finding_codes)


def test_docstring_points_to_the_numeric_summary_for_distributions():
    doc = inspect.getdoc(ae.describe_dataset) or ""
    when_to_use = doc.split("When to use:")[1].split("Args:")[0]
    see_also = doc.split("See Also:")[1]

    assert "`summarize_numeric`" in when_to_use
    assert "\n    summarize_numeric: " in see_also


def codes(res):
    return [f.code for f in res.findings]


def only_finding(res, code):
    matches = [f for f in res.findings if f.code == code]
    assert len(matches) == 1, f"expected one {code!r} finding, got {codes(res)}"
    return matches[0]


def compares_as_pandas_does(dtype):
    """Numbers, booleans, dates and durations, whose values pandas compares as
    describe_dataset does; text and objects hold values that differ only here,
    such as True and 1, or a text with a NUL."""
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
    )


@given(df=data_frames())
def test_counts_and_memory_agree_with_pandas_on_any_frame(df):
    reference = nan_as_missing(df)
    # What pandas measures for NumPy numbers, booleans, dates and durations,
    # taken first so that nothing describe_dataset does can change it. Other
    # dtypes differ on purpose, such as text, whose cached UTF-8 copies it
    # ignores, and categoricals, whose lookup tables it ignores.
    numpy_memory = {
        position: reference.iloc[:, position].memory_usage(deep=True, index=False)
        for position in range(df.shape[1])
        if isinstance(df.dtypes.iloc[position], np.dtype)
        and df.dtypes.iloc[position].kind in "biufmM"
    }

    res = ae.describe_dataset(df)

    assert (res.row_count, res.column_count) == df.shape
    assert len(res.columns) == df.shape[1]
    for position, entry in enumerate(res.columns):
        column = reference.iloc[:, position]
        assert entry.missing_count == int(column.isna().sum())
        assert entry.non_missing_count == int(column.count())
        if entry.distinct_count is not None and compares_as_pandas_does(column.dtype):
            assert entry.distinct_count == column.nunique(dropna=True)
        if position in numpy_memory:
            assert entry.memory_bytes == numpy_memory[position]
