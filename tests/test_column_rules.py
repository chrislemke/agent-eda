"""The rules every analysis that takes columns follows, shown with summarize_numeric."""

import numpy as np
import pandas as pd
import pytest

import agent_eda as ae
from agent_eda.errors import (
    AgentEDAError,
    ColumnNotFoundError,
    ColumnTypeError,
    InvalidArgumentError,
)

ACCEPTED = (
    "It accepts integer, float and boolean columns, nullable and pyarrow-backed "
    "ones of those kinds included."
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "age": [34, 51, 29],
            "income": [52.5, 61.0, 48.25],
            "city": ["Lyon", "Oslo", "Rome"],
        }
    )


@pytest.mark.parametrize(
    "columns",
    [
        ["income", "age"],
        ("income", "age"),
        pd.Index(["income", "age"]),
        pd.Series(["income", "age"]),
        np.array(["income", "age"]),
        (name for name in ["income", "age"]),
    ],
    ids=["list", "tuple", "index", "series", "numpy", "generator"],
)
def test_columns_take_any_ordered_list_of_names(df, columns):
    res = ae.summarize_numeric(df, columns=columns)

    assert list(res.columns) == ["income", "age"]
    assert res.parameters["columns"] == ["income", "age"]


def test_columns_take_a_slice_of_the_dataframes_own_columns(df):
    res = ae.summarize_numeric(df, columns=df.columns[:2])

    assert list(res.columns) == ["age", "income"]


def test_columns_take_a_single_name(df):
    res = ae.summarize_numeric(df, columns="income")

    assert list(res.columns) == ["income"]
    assert res.parameters["columns"] == ["income"]


def test_columns_whose_labels_are_not_strings_are_named_by_label_or_string_form():
    df = pd.DataFrame(np.arange(6.0).reshape(2, 3))

    by_label = ae.summarize_numeric(df, columns=[2, np.int64(0)])  # ty: ignore[invalid-argument-type]  # labels that aren't strings are accepted at runtime
    by_name = ae.summarize_numeric(df, columns=["2", "0"])

    assert list(by_label.columns) == ["2", "0"]
    assert by_label.parameters["columns"] == ["2", "0"]
    assert by_name.to_json() == by_label.to_json()


def test_a_set_of_columns_raises_because_it_has_no_order(df):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df, columns={"age", "income"})  # ty: ignore[invalid-argument-type]  # the unordered set is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "summarize_numeric got a set for columns, and a set has no fixed order. "
        "Pass a list or tuple instead, such as columns=sorted(...)."
    )


def test_duplicated_column_labels_raise_with_a_renaming_hint():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "a", "a"])

    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df)

    assert caught.value.code == "ambiguous_column_labels"
    assert str(caught.value) == (
        "summarize_numeric refers to columns by name, the string form of each "
        "label, so labels must be unique and distinct as strings, but 'a' occurs "
        "more than once. Rename them first, such as df.columns = [f'{c}_{i}' if "
        "dup else c for i, (c, dup) in enumerate(zip(df.columns, "
        "df.columns.duplicated()))]. describe_dataset describes data with any "
        "labels."
    )


def test_labels_whose_string_forms_collide_raise_with_a_renaming_hint():
    df = pd.DataFrame([[1, 2, 3]], columns=["b", 1, "1"])

    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df, columns=["b"])

    assert caught.value.code == "ambiguous_column_labels"
    assert str(caught.value) == (
        "summarize_numeric refers to columns by name, the string form of each "
        "label, so labels must be unique and distinct as strings, but 1 and '1' "
        "share a string form. Rename them first, such as "
        "df = df.rename(columns={1: '1_int'}). describe_dataset describes data "
        "with any labels."
    )


def test_duplicated_and_colliding_labels_are_named_together():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", 1, "a", "1"])

    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df)

    message = str(caught.value)
    assert "but 'a' occurs more than once, and 1 and '1' share a string form." in (
        message
    )
    assert "df.columns.duplicated()" in message
    assert "df = df.rename(columns={1: '1_int'})" in message


def test_multiindex_columns_raise_with_a_flattening_hint():
    columns = pd.MultiIndex.from_tuples([("price", "min"), ("price", "max")])
    df = pd.DataFrame([[1.0, 2.0]], columns=columns)

    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df)

    assert caught.value.code == "ambiguous_column_labels"
    assert str(caught.value) == (
        "summarize_numeric refers to columns by name, but the column labels form "
        "a MultiIndex with 2 levels. Flatten them first, such as df.columns = "
        "['_'.join(map(str, c)) for c in df.columns]. describe_dataset describes "
        "data with any labels."
    )


def test_an_empty_column_list_raises_with_the_none_alternative(df):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df, columns=[])

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "summarize_numeric got an empty list for columns. Name at least one "
        "column, or pass columns=None for every numeric column except booleans."
    )


def test_a_column_listed_twice_raises():
    df = pd.DataFrame({"age": [1, 2], "income": [3.0, 4.0], "score": [5, 6]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.summarize_numeric(df, columns=["age", "income", "age", "score", "income"])

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "summarize_numeric got 'age' and 'income' more than once in columns. "
        "List each column once."
    )


def test_an_unknown_column_raises_with_the_closest_names_and_the_available_ones():
    df = pd.DataFrame({"income": [1.0], "INCOMES": [2.0], "age": [3], "city": ["x"]})

    with pytest.raises(ColumnNotFoundError) as caught:
        ae.summarize_numeric(df, columns=["age", "INCOME"])

    error = caught.value
    assert isinstance(error, LookupError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "column_not_found"
    # 'income' differs only in case, so it comes before the closer spelling.
    assert str(error) == (
        "summarize_numeric got an unknown column 'INCOME' in columns. Did you "
        "mean 'income' or 'INCOMES'? The data has 4 columns: 'income', "
        "'INCOMES', 'age' and 'city'."
    )


def test_several_unknown_columns_are_named_together_each_with_its_matches(df):
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.summarize_numeric(df, columns=["Age ", "income", "zzz", "incme"])

    assert str(caught.value) == (
        "summarize_numeric got 3 unknown columns in columns: 'Age ' (did you mean "
        "'age'?), 'zzz' and 'incme' (did you mean 'income'?). The data has 3 "
        "columns: 'age', 'income' and 'city'."
    )


def test_the_available_columns_are_capped_at_20_names():
    df = pd.DataFrame({f"col{i}": [i] for i in range(25)})

    with pytest.raises(ColumnNotFoundError) as caught:
        ae.summarize_numeric(df, columns=["nothing_like_it"])

    names = ", ".join(f"'col{i}'" for i in range(20))
    assert str(caught.value) == (
        "summarize_numeric got an unknown column 'nothing_like_it' in columns. "
        f"The data has 25 columns: {names} and 5 more."
    )


TEXT_FIX = (
    "Use summarize_categorical for 'x' instead; convert numbers stored as text "
    "with pd.to_numeric(df['x'], errors='coerce'); or leave 'x' out of columns."
)


@pytest.mark.parametrize(
    ("values", "fix"),
    [
        (pd.Series(["1", "2", None]), TEXT_FIX),
        (pd.Series(["1", "2", None], dtype="string"), TEXT_FIX),
        (pd.Series([1, "a", None], dtype=object), TEXT_FIX),
        (
            pd.Categorical([1, 2, 1]),
            "Use summarize_categorical for 'x' instead; convert numeric categories "
            "with df['x'].astype(float); or leave 'x' out of columns.",
        ),
        (
            pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "Leave 'x' out of columns.",
        ),
        (
            pd.to_timedelta(["1D", None, "3D"]),
            "Convert durations with df['x'].dt.total_seconds(), or leave 'x' out "
            "of columns.",
        ),
    ],
    ids=["default-text", "string", "object", "categorical", "datetime", "timedelta"],
)
def test_a_column_of_an_unsupported_dtype_raises_with_its_dtype_and_a_fix(values, fix):
    df = pd.DataFrame({"x": values, "y": [1.0, 2.0, 3.0]})

    with pytest.raises(ColumnTypeError) as caught:
        ae.summarize_numeric(df, columns=["y", "x"])

    error = caught.value
    assert isinstance(error, TypeError)
    assert isinstance(error, AgentEDAError)
    assert error.code == "unsupported_column_type"
    assert str(error) == (
        f"summarize_numeric can't use column 'x' with dtype {df['x'].dtype}. "
        f"{ACCEPTED} {fix}"
    )


def test_several_columns_of_unsupported_dtypes_are_named_together(df):
    df["grade"] = pd.Categorical(["a", "b", "a"])

    with pytest.raises(ColumnTypeError) as caught:
        ae.summarize_numeric(df, columns=["grade", "age", "city"])

    assert str(caught.value) == (
        "summarize_numeric can't use 2 columns: 'grade' with dtype category and "
        f"'city' with dtype {df['city'].dtype}. {ACCEPTED} Use "
        "summarize_categorical for 'grade' and 'city' instead; convert numeric "
        "categories with df['grade'].astype(float); convert numbers stored as "
        "text with pd.to_numeric(df['city'], errors='coerce'); or leave them out "
        "of columns."
    )


def test_the_analysis_that_fits_is_named_only_for_the_columns_it_fits(df):
    df["when"] = pd.to_datetime(["2024-01-01", None, "2024-03-01"])

    with pytest.raises(ColumnTypeError) as caught:
        ae.summarize_numeric(df, columns=["when", "city"])

    assert str(caught.value).endswith(
        f"{ACCEPTED} Use summarize_categorical for 'city' instead; convert numbers "
        "stored as text with pd.to_numeric(df['city'], errors='coerce'); or leave "
        "them out of columns."
    )
