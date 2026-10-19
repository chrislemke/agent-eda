"""The suggestions in a declared value's error read text by the parse rules.

A sentinel value or an expected label of the wrong type gets "Did you mean ...?"
with the value the caller likely meant. The suggestion reads the text the way
infer_column_semantics and validate_column_types do, so it is always a value
those analyses accept, and text they reject gets none.
"""

import datetime as dt
import re

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

COLUMNS = {
    "integer": pd.Series([1, 2, 3], dtype="int64"),
    "float": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
    "boolean": pd.Series([True, False, True], dtype="bool"),
    "datetime": pd.Series(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    ).astype("datetime64[ns]"),
}


def suggestion(text: str, logical_type: str) -> object | None:
    """The value a declared text is suggested as for a column, or None."""
    df = pd.DataFrame({"c": COLUMNS[logical_type]})
    with pytest.raises(InvalidArgumentError) as caught:
        ae.detect_missing_sentinels(df, sentinel_values={"c": [text]})
    found = re.search(r"Did you mean (.+)\?", str(caught.value))
    if found is None:
        return None
    return eval(  # the message shows the value as Python, in a controlled namespace
        found.group(1), {"__builtins__": {}}, {"pd": pd, "datetime": dt}
    )


@pytest.mark.parametrize(
    ("text", "logical_type", "expected"),
    [
        ("1000", "integer", 1000),
        (" 12 ", "integer", 12),
        ("+5", "integer", 5),
        ("4.2e1", "integer", 42),
        ("12.0", "integer", 12),
        ("1.5", "float", 1.5),
        (".5", "float", 0.5),
        ("1e-3", "float", 0.001),
        ("yes", "boolean", True),
        ("No", "boolean", False),
        ("T", "boolean", True),
        ("y", "boolean", True),
        ("1", "boolean", True),
        ("0", "boolean", False),
        ("2024-01-05", "datetime", pd.Timestamp("2024-01-05")),
        ("2024-01-05 10:30", "datetime", pd.Timestamp("2024-01-05 10:30")),
    ],
)
def test_text_the_parse_rules_read_is_suggested_as_what_it_reads_as(
    text, logical_type, expected
):
    found = suggestion(text, logical_type)

    assert found == expected
    assert type(found) is type(expected)


@pytest.mark.parametrize(
    ("text", "logical_type"),
    [
        ("1_000", "integer"),  # Python's int() reads it; the parse rules don't
        ("١٢", "integer"),  # Arabic-Indic digits, read by int() only
        ("١٢", "float"),
        ("007", "integer"),  # a code: a number would drop its zeros
        ("1,000", "integer"),
        ("1.5", "integer"),
        ("inf", "float"),
        ("nan", "float"),
        ("1e400", "float"),
        ("0x10", "integer"),
        ("maybe", "boolean"),
        ("2", "boolean"),
        ("20240105", "datetime"),  # datetime.fromisoformat reads it; ISO 8601 text
        ("2024-13-01", "datetime"),
        ("01/02/2024", "datetime"),
        ("now", "datetime"),
    ],
)
def test_text_the_parse_rules_reject_gets_no_suggestion(text, logical_type):
    assert suggestion(text, logical_type) is None


def test_a_datetime_suggestion_keeps_the_nanoseconds_written():
    found = suggestion("2024-01-05 10:30:00.123456789", "datetime")

    assert found == pd.Timestamp("2024-01-05 10:30:00.123456789")


def test_the_expected_labels_of_a_target_read_text_by_the_same_rules():
    df = pd.DataFrame({"y": [1, 0, 1, 0, 1, 1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            df,
            target_column="y",
            task="binary_classification",
            expected_labels=["1_000", "1"],
        )

    assert "Did you mean" not in str(caught.value)

    df = pd.DataFrame({"y": [True, False, True, False]})
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            df,
            target_column="y",
            task="binary_classification",
            expected_labels=["yes", "no"],
        )

    assert "Did you mean True and False?" in str(caught.value)


texts = st.one_of(
    st.text(alphabet="0123456789+-.eE_, ١٢", max_size=8),
    st.text(alphabet="truefalsynoTRUEFALSYNO ", max_size=6),
    st.text(alphabet="0123456789-:T .Zz+", max_size=30),
    st.sampled_from(["1_000", "١٢", "007", "1e400", "20240105", "inf"]),
)


@settings(max_examples=300, derandomize=True, deadline=None)
@given(text=texts, logical_type=st.sampled_from(sorted(COLUMNS)))
def test_every_suggestion_is_a_value_the_analyses_accept(text, logical_type):
    found = suggestion(text, logical_type)
    if found is None:
        return

    # Declared, the suggestion is accepted.
    df = pd.DataFrame({"c": COLUMNS[logical_type]})
    ae.detect_missing_sentinels(df, sentinel_values={"c": [found]})  # ty: ignore[invalid-argument-type]  # the suggestion is a sentinel value
    # And the analyses read the text it came from as the same logical type.
    texts_column = pd.DataFrame({"c": pd.Series([text], dtype=object)})
    res = ae.validate_column_types(texts_column, schema={"c": logical_type})
    assert res.columns["c"].invalid_count == 0
