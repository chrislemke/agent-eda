"""Sentinel rules of round 3 (Q6): number text, pandas' NA strings, stripped
stand-ins, and one written form for datetimes and decimals in every record."""

import datetime as dt
import json
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

TEXT_DTYPES = [object, "string", "string[pyarrow]", pd.ArrowDtype(pa.string())]


def suspected(res):
    return [
        (c.column, c.value, c.evidence)
        for c in res.candidates
        if c.confirmation == "suspected"
    ]


# Number text in a text column


@pytest.mark.parametrize("dtype", TEXT_DTYPES, ids=str)
def test_number_text_in_a_text_column_gets_the_numeric_code_rule(dtype):
    # What the MCP server reads from a CSV column that holds "N/A".
    df = pd.DataFrame(
        {"x": pd.Series(["34", "N/A", "-999", "-999", "51", "29"], dtype=dtype)}
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("x", "-999", ["builtin_list", "isolated_extreme"]),
        ("x", "N/A", ["builtin_list"]),
    ]
    assert [c.value_type for c in res.candidates] == ["str", "str"]
    assert [c.row_ids for c in res.candidates] == [[2, 3], [1]]


def test_the_number_text_a_candidate_shows_is_declarable_and_matches_exactly():
    df = pd.DataFrame({"x": ["34", "N/A", "-999", "-999", "51", "29"]})

    declared = ae.detect_missing_sentinels(df, sentinel_values={"x": ["-999"]})
    profile = ae.profile_missing_values(df, sentinel_values={"x": ["-999", "N/A"]})

    assert [(c.value, c.confirmation) for c in declared.candidates] == [
        ("-999", "declared"),
        ("N/A", "suspected"),
    ]
    assert profile.columns["x"].sentinel_count == 3


def test_number_text_at_the_extremes_is_judged_by_value_not_spelling():
    # -999 and -999.0 are one number, two texts: each is its own candidate.
    df = pd.DataFrame({"x": ["-999", "-999.0", "10", "20", "30", "x"]})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("x", "-999", ["builtin_list", "isolated_extreme"]),
        ("x", "-999.0", ["builtin_list", "isolated_extreme"]),
    ]


def test_number_text_in_the_middle_or_unlisted_is_no_candidate():
    df = pd.DataFrame({"x": ["-9999", "-999", "5", "7", "99", "120", "007", "abc"]})

    assert suspected(ae.detect_missing_sentinels(df)) == [
        ("x", "-9999", ["builtin_list"])
    ]


def test_number_text_in_an_object_column_is_read_as_text():
    # The number -999 is a number, the text '-999' is text: each family has its
    # own minimum and maximum.
    values = [1, 2, 3, "10", "20", "-999", "-999"]
    df = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [("x", "-999", ["builtin_list", "isolated_extreme"])]


def test_number_text_in_a_categorical_is_judged_by_its_categories():
    df = pd.DataFrame({"x": pd.Categorical(["34", "-999", "-999", "51", "29", "N/A"])})

    assert suspected(ae.detect_missing_sentinels(df)) == [
        ("x", "-999", ["builtin_list", "isolated_extreme"]),
        ("x", "N/A", ["builtin_list"]),
    ]


# pandas' default NA strings

PANDAS_NA_STRINGS = [
    "#N/A",
    "#N/A N/A",
    "#NA",
    "<NA>",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "-1.#IND",
    "-1.#QNAN",
]


@pytest.mark.parametrize("dtype", TEXT_DTYPES, ids=str)
def test_the_built_in_text_list_has_pandas_default_na_strings(dtype):
    df = pd.DataFrame(
        {"x": pd.Series(["Lyon", *PANDAS_NA_STRINGS, "Oslo"], dtype=dtype)}
    )

    res = ae.detect_missing_sentinels(df)

    assert suspected(res) == [
        ("x", text, ["builtin_list"]) for text in PANDAS_NA_STRINGS
    ]


def test_pandas_na_strings_match_in_any_mix_of_case():
    df = pd.DataFrame({"x": ["1.#qnan", "<na>", "#n/a n/a"]})

    assert [c.value for c in ae.detect_missing_sentinels(df).candidates] == [
        "1.#qnan",
        "<na>",
        "#n/a n/a",
    ]


def test_the_docstring_lists_pandas_na_strings():
    doc = ae.detect_missing_sentinels.__doc__ or ""
    for text in PANDAS_NA_STRINGS:
        assert text in doc


# Stand-ins match after stripping surrounding whitespace


@pytest.mark.parametrize("dtype", [*TEXT_DTYPES, "category"], ids=str)
def test_text_stand_ins_match_after_stripping_and_the_record_keeps_the_raw_form(dtype):
    df = pd.DataFrame(
        {
            "x": pd.Series(
                ["ok", " N/A", "N/A ", "N/A\n", "N/A", "\tnull ", "fine"], dtype=dtype
            )
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert {(c.value, c.count) for c in res.candidates} == {
        ("\tnull ", 1),
        (" N/A", 1),
        ("N/A", 1),
        ("N/A ", 1),
        ("N/A\n", 1),
    }
    assert all(c.evidence == ["builtin_list"] for c in res.candidates)


def test_the_raw_form_that_occurred_can_be_declared_and_counted():
    df = pd.DataFrame({"x": ["ok", " N/A", "N/A ", "N/A "]})

    res = ae.detect_missing_sentinels(df, sentinel_values={"x": ["N/A "]})

    assert [(c.value, c.count, c.confirmation) for c in res.candidates] == [
        ("N/A ", 2, "declared"),
        (" N/A", 1, "suspected"),
    ]


def test_text_that_only_resembles_a_stand_in_is_still_no_candidate():
    df = pd.DataFrame({"x": ["N / A", "n.a.", " none yet ", "NAN-", "NA NA"]})

    assert ae.detect_missing_sentinels(df).candidates == []


# One written form for datetimes and decimals

DAY = dt.datetime(1900, 1, 1)


def test_datetimes_are_written_in_iso_8601_by_every_analysis():
    stamps = pd.Series([DAY, DAY, dt.datetime(2020, 1, 1, 5, 30, 15)])
    df = pd.DataFrame({"when": stamps})
    iso = "1900-01-01T00:00:00"

    detected = ae.detect_missing_sentinels(df)
    profile = ae.profile_missing_values(df, sentinel_values={"when": [DAY]})
    validated = ae.validate_column_types(df, schema={"when": "integer"})

    assert [c.value for c in detected.candidates] == [iso]
    assert [v.value for v in profile.columns["when"].sentinel_values] == [iso]
    values = {v.value for v in validated.columns["when"].invalid_values}
    assert values == {iso, "2020-01-01T05:30:15"}


def test_datetime_categories_are_written_in_iso_8601():
    df = pd.DataFrame({"when": pd.Categorical([DAY, DAY, dt.datetime(2020, 1, 1, 5)])})

    res = ae.summarize_categorical(df, columns=["when"])

    assert [level.label for level in res.columns["when"].levels] == [
        "1900-01-01T00:00:00",
        "2020-01-01T05:00:00",
    ]


def test_zoned_dates_and_nanoseconds_have_one_iso_form_too():
    zoned = pd.Timestamp("2020-01-01 05:00:00", tz="Europe/Berlin")
    fine = pd.Timestamp("2020-01-01 05:00:00.123456789")
    day = dt.date(2020, 1, 1)
    df = pd.DataFrame({"x": pd.Series([zoned, fine, day, day], dtype=object)})

    res = ae.validate_column_types(df, schema={"x": "integer"})

    assert {v.value for v in res.columns["x"].invalid_values} == {
        "2020-01-01T05:00:00+01:00",
        "2020-01-01T05:00:00.123456789",
        "2020-01-01",
    }


def test_a_decimal_float64_holds_is_a_number_and_any_other_is_exact_text():
    values = [
        Decimal("2.5"),
        Decimal("19.99"),
        Decimal("3"),
        Decimal("12345678901234567.25"),
        Decimal("0.1000000000000000055511151231257827"),
    ]
    df = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    res = ae.summarize_categorical(df, columns=["x"])

    labels = {level.label_type: [] for level in res.columns["x"].levels}
    for level in res.columns["x"].levels:
        labels[level.label_type].append(level.label)
    assert sorted(labels["Decimal"], key=str) == sorted(
        [2.5, 19.99, 3, "12345678901234567.25", "0.1000000000000000055511151231257827"],
        key=str,
    )
    numbers = [label for label in labels["Decimal"] if not isinstance(label, str)]
    assert {type(label) for label in numbers} <= {int, float}


def test_a_decimal_candidate_float64_cannot_hold_is_recorded_exactly():
    exact = Decimal("12345678901234567.25")
    values = [exact, exact, Decimal("1.50"), Decimal("2.50"), Decimal("3.50")]
    df = pd.DataFrame(
        {
            "x": pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(25, 2))),
            "y": pd.Series(values, dtype=object),
        }
    )

    res = ae.detect_missing_sentinels(df)

    assert [(c.column, c.value, c.value_type) for c in res.candidates] == [
        ("x", "12345678901234567.25", "Decimal"),
        ("y", "12345678901234567.25", "Decimal"),
    ]
    assert res.to_dict()["candidates"][0]["value"] == "12345678901234567.25"


@pytest.mark.parametrize(
    "make",
    [
        lambda values: pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(25, 2))),
        lambda values: pd.Series(values, dtype=object),
    ],
    ids=["arrow decimal", "decimal objects"],
)
def test_declaring_the_recorded_decimal_matches_it(make):
    exact = Decimal("12345678901234567.25")
    other = [Decimal("1.50"), Decimal("2.50"), Decimal("3.50")]
    df = pd.DataFrame({"x": make([exact, exact, *other])})
    recorded = ae.detect_missing_sentinels(df).candidates[0].value

    detected = ae.detect_missing_sentinels(df, sentinel_values={"x": [recorded]})
    profile = ae.profile_missing_values(df, sentinel_values={"x": [recorded]})

    [candidate] = detected.candidates
    assert (candidate.value, candidate.count, candidate.confirmation) == (
        recorded,
        2,
        "declared",
    )
    assert profile.columns["x"].sentinel_count == 2
    # The declared value is recorded as it is.
    assert detected.parameters["sentinel_values"] == {"x": [recorded]}
    assert profile.parameters["sentinel_values"] == {"x": [recorded]}


def test_a_decimal_column_does_not_take_text_that_is_no_exact_decimal():
    df = pd.DataFrame({"x": pd.Series([Decimal("1.5"), Decimal("2.5")], dtype=object)})

    with pytest.raises(InvalidArgumentError, match="never match"):
        ae.profile_missing_values(df, sentinel_values={"x": ["n/a"]})


def test_a_column_of_decimals_and_text_keeps_declared_text_as_text():
    df = pd.DataFrame({"x": pd.Series([Decimal("1.5"), "1.5", "1.5"], dtype=object)})

    res = ae.profile_missing_values(df, sentinel_values={"x": ["1.5"]})

    assert res.columns["x"].sentinel_count == 2


# Declared values that print the same can still be told apart


def test_declared_values_that_print_the_same_are_told_apart_by_type():
    day = dt.date(2020, 1, 1)
    df = pd.DataFrame(
        {"x": pd.Series([day, "2020-01-01", day, True, 1, 1], dtype=object)}
    )

    res = ae.profile_missing_values(
        df, sentinel_values={"x": [day, "2020-01-01", True, 1]}
    )

    entries = res.columns["x"].sentinel_values
    assert [(v.value, v.value_type, v.count) for v in entries] == [
        ("2020-01-01", "date", 2),
        ("2020-01-01", "str", 1),
        (True, "bool", 1),
        (1, "int", 2),
    ]
    assert res.columns["x"].sentinel_count == 6


def test_the_columns_table_keeps_every_count_of_declared_values():
    day = dt.date(2020, 1, 1)
    df = pd.DataFrame(
        {"x": pd.Series([day, "2020-01-01", day, True, 1, 1], dtype=object)}
    )
    res = ae.profile_missing_values(
        df, sentinel_values={"x": [day, "2020-01-01", True, 1]}
    )

    [cell] = res.to_frame()["sentinel_values"].tolist()

    assert [(e["value"], e["value_type"], e["count"]) for e in cell] == [
        ("2020-01-01", "date", 2),
        ("2020-01-01", "str", 1),
        (True, "bool", 1),
        (1, "int", 2),
    ]
    assert sum(e["count"] for e in cell) == 6


def test_masked_declared_values_keep_their_types_and_counts():
    df = pd.DataFrame({"email": ["a@x.org", "n/a", "n/a", "b@x.org"]})
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["email"]))

    res = ae.profile_missing_values(
        df, sentinel_values={"email": ["n/a"]}, project=project, mask_sensitive=True
    )

    [cell] = res.to_frame()["sentinel_values"].tolist()
    assert [(e["value"], e["count"]) for e in cell] == [("<masked 1>", 2)]


def test_every_written_value_is_strict_json():
    df = pd.DataFrame(
        {
            "d": pd.Series(
                [Decimal("12345678901234567.25")] * 2 + [Decimal("1.5")] * 3
            ),
            "t": pd.Categorical([DAY, DAY, *[dt.datetime(2020, 1, 1)] * 3]),
        }
    )

    for res in (
        ae.detect_missing_sentinels(df),
        ae.summarize_categorical(df, columns=["t"]),
        ae.validate_column_types(df, schema={"d": "boolean", "t": "boolean"}),
    ):
        assert res.to_dict() == json.loads(res.to_json())


@pytest.mark.parametrize("dtype", TEXT_DTYPES, ids=str)
def test_whitespace_around_a_stand_in_is_whatever_python_strips(dtype):
    # Python strips these; the zero-width space is no whitespace at all.
    spaces = ["\x1c", "\x85", "\u00a0", "\u2003", "\u3000"]
    texts = [f"{space}NA{space}" for space in spaces] + ["\u200bNA", "ok"]
    df = pd.DataFrame({"x": pd.Series(texts, dtype=dtype)})

    res = ae.detect_missing_sentinels(df)

    assert [c.value for c in res.candidates] == texts[:5]
