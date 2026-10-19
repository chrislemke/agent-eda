"""Arrow dictionary, time, binary and sparse columns, judged like their pandas kin."""

import io
import warnings

import pandas as pd
import pyarrow as pa
import pytest

import agent_eda as ae
from agent_eda.errors import ColumnTypeError

WORDS = ["a", "N/A", "unknown", "b", "a", "b"] * 5


def parquet_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """The frame as Parquet gives it back with the pyarrow backend."""
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    with warnings.catch_warnings():
        # pyarrow 14 builds the DataFrame through a pandas 2.2 internal.
        warnings.simplefilter("ignore", DeprecationWarning)
        return pd.read_parquet(buffer, dtype_backend="pyarrow")


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same data with a pandas categorical and as an Arrow dictionary."""
    df = pd.DataFrame(
        {
            "c": pd.Categorical(WORDS),
            "n": list(range(30)),
            "t": ["x", "y", "z"] * 10,
        }
    )
    arrow = parquet_roundtrip(df)
    assert str(arrow["c"].dtype).startswith("dictionary<values=string")
    return df, arrow


def tables_without_dtypes(res) -> pd.DataFrame:
    """The result's table, minus the columns that name a dtype."""
    table = res.to_frame()
    return table[[c for c in table.columns if "dtype" not in c]]


def test_a_parquet_categorical_read_with_the_pyarrow_backend_is_a_categorical():
    pandas_frame, arrow_frame = frames()

    res = ae.summarize_categorical(arrow_frame)

    assert res.status == "completed"
    assert [c.column for c in res.columns.values()] == ["c", "t"]
    expected = ae.summarize_categorical(pandas_frame)
    pd.testing.assert_frame_equal(
        tables_without_dtypes(res), tables_without_dtypes(expected)
    )
    assert res.left_out_columns == ["n"]


def test_the_sentinels_in_an_arrow_categorical_are_found():
    pandas_frame, arrow_frame = frames()

    res = ae.detect_missing_sentinels(arrow_frame)

    assert [(c.column, c.value) for c in res.candidates] == [
        ("c", "N/A"),
        ("c", "unknown"),
    ]
    pd.testing.assert_frame_equal(
        tables_without_dtypes(res),
        tables_without_dtypes(ae.detect_missing_sentinels(pandas_frame)),
    )


def test_declared_values_count_in_an_arrow_categorical():
    pandas_frame, arrow_frame = frames()

    res = ae.profile_missing_values(arrow_frame, sentinel_values={"c": ["N/A"]})

    assert res.columns["c"].sentinel_count == 5
    expected = ae.profile_missing_values(pandas_frame, sentinel_values={"c": ["N/A"]})
    assert res.columns["c"].sentinel_count == expected.columns["c"].sentinel_count


def test_an_arrow_categorical_declared_categorical_is_compatible():
    pandas_frame, arrow_frame = frames()

    res = ae.validate_column_types(arrow_frame, schema={"c": "categorical"})

    entry = res.columns["c"]
    assert (entry.compatible_count, entry.convertible_count, entry.invalid_count) == (
        30,
        0,
        0,
    )
    expected = ae.validate_column_types(pandas_frame, schema={"c": "categorical"})
    assert entry.compatible_count == expected.columns["c"].compatible_count


def test_an_arrow_categorical_declared_text_converts_and_declared_integer_is_invalid():
    _, arrow_frame = frames()

    text = ae.validate_column_types(arrow_frame, schema={"c": "text"})
    integer = ae.validate_column_types(arrow_frame, schema={"c": "integer"})

    assert text.columns["c"].convertible_count == 30
    assert integer.columns["c"].invalid_count == 30


def test_an_arrow_categorical_pairs_with_the_other_columns():
    pandas_frame, arrow_frame = frames()

    res = ae.compute_association_matrix(arrow_frame)

    assert res.skipped_columns == []
    assert {(p.x_column, p.y_column) for p in res.pairs} == {
        ("c", "t"),
        ("c", "n"),
        ("t", "n"),
    }
    pd.testing.assert_frame_equal(
        tables_without_dtypes(res),
        tables_without_dtypes(ae.compute_association_matrix(pandas_frame)),
    )


def test_an_arrow_categorical_is_a_classification_target():
    pandas_frame, arrow_frame = frames()

    res = ae.profile_target(
        arrow_frame, target_column="c", task="multiclass_classification"
    )

    assert res.status == "completed"
    assert res.class_count == 4
    expected = ae.profile_target(
        pandas_frame, target_column="c", task="multiclass_classification"
    )
    pd.testing.assert_frame_equal(res.to_frame(), expected.to_frame())


def test_an_arrow_categorical_with_unused_entries_reports_them_like_pandas():
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0, None], type=pa.int8()), pa.array(["a", "b", "unused"])
    )
    arrow = pd.DataFrame({"c": pd.arrays.ArrowExtensionArray(dictionary)})
    pandas_frame = pd.DataFrame(
        {"c": pd.Categorical(["a", "b", "a", None], ["a", "b", "unused"])}
    )

    res = ae.summarize_categorical(arrow)
    expected = ae.summarize_categorical(pandas_frame)

    assert [f.code for f in res.findings] == [f.code for f in expected.findings]
    assert "unused_categories" in [f.code for f in res.findings]
    assert res.columns["c"].missing_count == 1


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(pd.Categorical([None, None, None]), id="pandas categorical"),
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.DictionaryArray.from_arrays(
                    pa.array([None, None, None], type=pa.int8()),
                    pa.array([], type=pa.string()),
                )
            ),
            id="arrow dictionary",
        ),
    ],
)
def test_a_categorical_without_categories_has_no_classes(column):
    df = pd.DataFrame({"c": column})

    res = ae.profile_target(df, target_column="c", task="binary_classification")

    assert res.status == "insufficient_data"
    assert res.class_count == 0


def test_a_dictionary_of_numbers_is_a_categorical_too():
    values = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0, 1], type=pa.int8()), pa.array([-999, 5])
    )
    arrow = pd.DataFrame({"c": pd.arrays.ArrowExtensionArray(values)})

    res = ae.summarize_categorical(arrow)
    sentinels = ae.detect_missing_sentinels(arrow)

    assert [lv.label for lv in res.columns["c"].most_common_levels] == [-999, 5]
    assert sentinels.status == "completed"


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(
            pd.Series(
                pd.arrays.ArrowExtensionArray(pa.array([1, None], pa.time64("us")))
            ),
            id="time64",
        ),
        pytest.param(
            pd.Series(pd.arrays.ArrowExtensionArray(pa.array([b"a", None]))),
            id="binary",
        ),
        pytest.param(
            pd.Series(pd.arrays.SparseArray(["a", None], dtype=object)),
            id="sparse of objects",
        ),
    ],
)
def test_time_binary_and_sparse_object_columns_are_other_and_say_why(column):
    df = pd.DataFrame({"x": column, "y": ["u", "v"]})
    dtype = str(df["x"].dtype)

    described = ae.describe_dataset(df)
    summary = ae.summarize_categorical(df)
    pairs = ae.compute_association_matrix(df)

    assert described.columns[0].dtype == dtype
    # The default leaves the column out and names it.
    assert summary.left_out_columns == ["x"]
    assert [(s.column, s.reason) for s in pairs.skipped_columns] == [
        ("x", "unsupported_dtype")
    ]
    assert "neither numbers nor categories" in pairs.skipped_columns[0].message
    with pytest.raises(ColumnTypeError) as caught:
        ae.summarize_categorical(df, columns=["x"])
    message = str(caught.value)
    assert dtype in message
    # The message no longer claims that every pyarrow-backed column is accepted.
    assert "pyarrow-backed ones included" not in message
    assert "pyarrow-backed ones of those kinds included" in message


def test_sparse_numbers_keep_their_numeric_kind():
    df = pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.5, 0.0])})

    res = ae.summarize_numeric(df)

    assert [c.column for c in res.columns.values()] == ["x"]
