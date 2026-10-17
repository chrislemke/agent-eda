import numpy as np
import pandas as pd
import polars as pl
import pytest

import agent_eda as ae
from agent_eda.errors import AgentEDAError, InvalidArgumentError


@pytest.mark.parametrize(
    ("data", "got", "conversion"),
    [
        (pd.Series([1, 2], name="x"), "a pandas Series", "series.to_frame()"),
        (np.array([[1, 2], [3, 4]]), "a NumPy array", "pd.DataFrame(array)"),
        (pl.DataFrame({"x": [1, 2]}), "a polars DataFrame", "df.to_pandas()"),
    ],
    ids=["series", "numpy", "polars"],
)
def test_input_that_is_not_a_dataframe_raises_with_the_conversion(
    data, got, conversion
):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset(data)

    error = caught.value
    assert error.code == "unsupported_data_type"
    assert str(error) == (
        f"data must be a pandas DataFrame, but got {got}. Convert it first: "
        f"ae.describe_dataset({conversion})"
    )


def test_invalid_argument_error_is_an_agent_eda_error_and_a_value_error():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset([1, 2, 3])  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert isinstance(caught.value, AgentEDAError)
    assert isinstance(caught.value, ValueError)
    assert "pd.DataFrame(data)" in str(caught.value)


def test_an_argument_of_the_wrong_type_raises_without_a_pydantic_error():
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset(df, source=5)  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    error = caught.value
    assert error.code == "invalid_argument"
    assert str(error).startswith(
        "describe_dataset got 5 for source, but source must be str | None."
    )
    assert error.__cause__ is None
    assert error.__suppress_context__


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (
            pd.DataFrame({"email": [f"p{i}@example.com" for i in range(100)]}),
            "a pandas DataFrame with 100 rows and 1 column",
        ),
        (
            pd.Series([f"p{i}@example.com" for i in range(60)]),
            "a pandas Series of 60 values",
        ),
        ([f"p{i}@example.com" for i in range(3)], "a list of 3 items"),
    ],
    ids=["dataframe", "series", "list"],
)
def test_a_bad_value_is_shown_by_type_and_size_not_by_its_contents(value, shown):
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset(df, source=value)

    message = str(caught.value)
    assert message.startswith(f"describe_dataset got {shown} for source, but ")
    assert "example.com" not in message
    assert "\n" not in message


def test_options_passed_by_position_raise_with_the_keyword_form():
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset(df, "sales.csv")  # ty: ignore[too-many-positional-arguments]  # the bad call is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "describe_dataset takes data as its only positional argument; pass "
        "everything else by keyword, such as describe_dataset(data, source=...)."
    )


def test_an_unknown_keyword_raises_with_the_closest_name_and_the_allowed_ones():
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset(df, sourc="sales.csv")  # ty: ignore[unknown-argument]  # the typo is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "describe_dataset has no argument 'sourc'. Did you mean 'source'? "
        "Allowed: data, source, source_version."
    )


def test_missing_data_raises_with_the_call_pattern():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.describe_dataset()  # ty: ignore[missing-argument]  # the bad call is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "describe_dataset needs data, a pandas DataFrame: describe_dataset(df)."
    )
