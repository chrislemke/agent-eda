import datetime as dt
import inspect
import json
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, InvalidArgumentError

STATUSES = ["completed", "partial", "insufficient_data", "not_applicable", "failed"]


def infer(values, **arguments):
    """The entry for a single column named "x" holding the values."""
    data = values if isinstance(values, pd.Series) else pd.Series(values)
    return ae.infer_column_semantics(pd.DataFrame({"x": data}), **arguments).columns[
        "x"
    ]


def ranking(entry):
    return [candidate.logical_type for candidate in entry.candidates]


# The spec's classification cases.


def test_postal_codes_with_leading_zeros_rank_a_type_that_isnt_numeric_first():
    zips = ["02134", "10115", "01234", "80331", "02134", "60311", "01234", "10115"]

    entry = infer(zips)

    assert entry.inferred_logical_type in ("categorical", "text")
    assert ranking(entry)[:2] == ["categorical", "text"]
    scores = {c.logical_type: c.score for c in entry.candidates}
    # A number would drop the zeros, so numbers score 0.
    assert scores["integer"] == scores["float"] == 0
    assert "numbers_as_text" not in entry.flags
    assert entry.evidence.leading_zero_share == 0.5


def test_postal_codes_that_are_all_distinct_still_rank_text_first():
    entry = infer(["02134", "10115", "01234", "80331"])

    assert entry.inferred_logical_type == "text"
    assert entry.flags == ["identifier_like"]


def test_numeric_looking_ids_are_flagged_identifier_like():
    entry = infer(["100234", "583921", "100587", "772001", "310245"])

    assert "identifier_like" in entry.flags
    assert entry.inferred_logical_type == "integer"


def test_iso_date_strings_rank_datetime_first():
    entry = infer(["2024-01-05", "2024-02-29", "2023-12-31", "2024-01-05", None])

    assert entry.inferred_logical_type == "datetime"
    assert entry.candidates[0].score == 1
    assert entry.flags == ["dates_as_text"]


@pytest.mark.parametrize(
    "values",
    [
        ["yes", "no", "no", "yes", "Yes", "NO"],
        pd.Series([0, 1, 1, 0, 1, 1]),
        pd.Series([1.0, 0.0, None, 1.0], dtype="Float64"),
        ["1", "0", "0", "1"],
        ["true", "False", "TRUE", "t", "F", "y", "n"],
    ],
    ids=["yes-no", "zero-one", "nullable-floats", "zero-one-text", "words"],
)
def test_yes_no_and_zero_one_columns_rank_boolean_first_and_are_boolean_like(values):
    entry = infer(values)

    assert entry.inferred_logical_type == "boolean"
    assert "boolean_like" in entry.flags


def test_a_column_stored_as_booleans_ranks_boolean_first_without_the_flag():
    entry = infer(pd.Series([True, False, None, True], dtype="boolean"))

    assert entry.inferred_logical_type == "boolean"
    assert entry.flags == []


def test_floats_that_are_whole_apart_from_gaps_rank_integer_first():
    entry = infer([3.0, None, 7.0, 3.0, None, 12.0, 7.0])

    assert ranking(entry)[:2] == ["integer", "float"]
    assert entry.evidence.integer_share == 1


def test_free_text_sentences_rank_text_first():
    entry = infer(
        [
            "The parcel arrived late and damaged.",
            "Great service, will order again.",
            "Too expensive for what you get.",
            "Delivery was quick.",
            "The parcel arrived late and damaged.",
        ]
    )

    assert entry.inferred_logical_type == "text"
    assert entry.flags == []


def test_a_few_labels_repeated_over_many_rows_rank_categorical_first():
    entry = infer(["red", "green", "blue", "green"] * 25)

    assert entry.inferred_logical_type == "categorical"
    # 1 - 3 / 100, within floating-point rounding.
    assert entry.candidates[0].score == pytest.approx(0.97, abs=1e-15)


def test_an_all_distinct_integer_sequence_is_flagged_identifier_like():
    entry = infer(np.arange(1, 51))

    assert entry.inferred_logical_type == "integer"
    assert entry.flags == ["identifier_like"]


def test_a_sequence_with_gaps_stored_as_floats_is_flagged_identifier_like():
    # Half the numbers from 101 to 108 occur, as after deleting some rows.
    entry = infer([101.0, None, 103.0, 104.0, 108.0])

    assert entry.flags == ["identifier_like"]


def test_distinct_whole_numbers_that_arent_a_sequence_are_no_identifiers():
    # Ages: distinct in a small sample, but spread over more than twice their count.
    assert infer([34, 51, 29, 47, 38]).flags == []


# Candidates and their scores.


def test_ranks_candidates_by_score_with_reasons_from_fixed_templates():
    entry = infer(["1", "2", "2", "x"])

    # "x" isn't a boolean, so the column has no boolean candidate.
    assert [(c.logical_type, c.score) for c in entry.candidates] == [
        ("integer", 0.75),
        ("float", 0.75),
        ("categorical", 0.25),
        ("text", 0.09375),
    ]
    reasons = {c.logical_type: c.reasons for c in entry.candidates}
    assert reasons["integer"] == ["75% of values are or parse as whole numbers."]
    assert reasons["categorical"] == [
        "1 - uniqueness share: 3 distinct values over 4 values."
    ]
    assert reasons["text"] == [
        "(plain text share 0.25 x uniqueness share 0.75 + multi-word share 0) / 2."
    ]


def test_scores_follow_the_documented_rules():
    # 6 values: 4 distinct, 2 plain text of several words, 1 date, 1 boolean word,
    # which doesn't make a boolean candidate while other values aren't booleans.
    entry = infer(["a b", "a b", "x", "2024-01-05", "yes", "x"])

    scores = {c.logical_type: c.score for c in entry.candidates}
    evidence = entry.evidence
    assert evidence.uniqueness_share == pytest.approx(4 / 6, abs=1e-15)
    assert evidence.plain_text_share == pytest.approx(4 / 6, abs=1e-15)
    assert evidence.multi_word_share == pytest.approx(2 / 6, abs=1e-15)
    assert scores == pytest.approx(
        {
            "categorical": 1 - 4 / 6,
            "text": (4 / 6 * 4 / 6 + 2 / 6) / 2,
            "datetime": 1 / 6,
        },
        abs=1e-15,
    )


def test_equal_scores_rank_the_more_specific_logical_type_first():
    entry = infer([0, 1])

    # boolean, integer and float all score 1.
    assert ranking(entry) == ["boolean", "integer", "float"]


def test_the_docstring_states_that_scores_are_not_probabilities():
    doc = inspect.getdoc(ae.infer_column_semantics)

    assert doc
    assert "heuristic, not probabilities" in " ".join(doc.split())


def test_lists_the_evidence_behind_the_candidates():
    entry = infer(["1", "2", "2", None, "1.5", "0", "x", "01"])

    evidence = entry.evidence
    # A share is one integer over another, correctly rounded, so each equals
    # Python's quotient exactly.
    assert evidence.non_missing_count == 7
    assert evidence.distinct_count == 6
    assert evidence.uniqueness_share == 6 / 7
    assert evidence.number_share == 5 / 7
    assert evidence.integer_share == 4 / 7
    assert evidence.boolean_share == 2 / 7
    assert evidence.datetime_share == 0
    assert evidence.leading_zero_share == 1 / 7
    assert evidence.value_type_counts.text == 7


def test_parses_each_distinct_value_once_and_weights_it_by_its_count():
    entry = infer(["7"] * 99 + ["seven"])

    assert entry.evidence.number_share == 0.99
    assert entry.evidence.distinct_count == 2


# Parse rules.


@pytest.mark.parametrize(
    ("text", "number", "whole"),
    [
        ("42", True, True),
        (" -3.5 ", True, False),
        ("+4.2E1", True, True),
        ("1e-3", True, False),
        (".5", True, False),
        ("42.0", True, True),
        ("100e-2", True, True),
        ("12345678901234567890123", True, True),
        ("1,000", False, False),
        ("3,5", False, False),
        ("0x1F", False, False),
        ("inf", False, False),
        ("nan", False, False),
        ("1_000", False, False),
        ("٤٢", False, False),
        ("007", False, False),
        ("0", True, True),
        ("0.25", True, False),
    ],
)
def test_parses_decimal_numbers_and_tells_whole_ones_apart(text, number, whole):
    evidence = infer([text]).evidence

    assert (evidence.number_share, evidence.integer_share) == (number, whole)


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("2024-02-29", True),
        ("2024-02-29T13:45", True),
        ("2024-02-29 13:45:59", True),
        ("2024-02-29T13:45:59.123456789Z", True),
        ("2024-02-29T13:45:59+01:00", True),
        ("2024-02-29T13:45:59-0530", True),
        ("0001-01-01", True),
        ("9999-12-31T23:59:59", True),
        ("2023-02-29", False),
        ("2024-13-01", False),
        ("2024-04-31", False),
        ("2024-01-01T24:00", False),
        ("2024-01-01T12:60", False),
        ("0000-01-01", False),
        ("2024-1-5", False),
        ("20240105", False),
        ("2024-W01-1", False),
        ("2024-01", False),
        ("01/02/2024", False),
    ],
)
def test_parses_iso_8601_dates_only_and_checks_the_calendar(text, iso):
    assert infer([text]).evidence.datetime_share == iso


@pytest.mark.parametrize(
    "text",
    [
        "01/02/2024",
        "1/2/24",
        "02.01.2024",
        "2024/01/02",
        "2024-1-2",
        "2024-13-45",
        "5 Jan 2024",
        "05-Jan-24",
        "January 5, 2024",
        "Jan 2024",
        "01/02/2024 10:30 PM",
    ],
)
def test_date_like_text_that_isnt_iso_8601_is_flagged_and_never_parsed(text):
    entry = infer([text, text])

    assert entry.evidence.non_iso_date_share == 1
    assert entry.evidence.datetime_share == 0
    assert "datetime" not in ranking(entry)
    assert "non_iso_date_text" in entry.flags


@pytest.mark.parametrize("text", ["1.2.3", "Marketing 2024", "12/2024", "10:30"])
def test_text_with_numbers_isnt_mistaken_for_a_date(text):
    assert infer([text]).evidence.non_iso_date_share == 0


def test_a_few_date_like_values_among_other_text_raise_no_flag():
    entry = infer(["01/02/2024", "call back", "no answer", "left a message"])

    assert entry.evidence.non_iso_date_share == 0.25
    assert "non_iso_date_text" not in entry.flags


@pytest.mark.parametrize(
    ("values", "share"),
    [
        (["true", "False", "YES", "no", "t", "F", "y", "N"], 1.0),
        (["1", "0", "1.0", "-0", "+1"], 1.0),
        (["on", "off", "2", "-1", "ja"], 0.0),
        ([0, 1, 2, 3], 0.5),
        ([True, False], 1.0),
    ],
)
def test_reads_booleans_from_words_and_from_zero_and_one(values, share):
    assert infer(values).evidence.boolean_share == share


# Flags.


def test_flags_a_constant_column():
    assert infer(["a", "a", None, "a"]).flags == ["constant"]


def test_flags_an_all_missing_column_and_gives_it_no_candidates():
    entry = infer([None, None], declared_schema={"x": "integer"})

    assert entry.flags == ["all_missing"]
    assert entry.candidates == []
    assert entry.inferred_logical_type is None
    assert entry.status == "insufficient_data"
    assert entry.evidence.uniqueness_share is None


def test_constant_and_all_missing_columns_are_findings_that_point_at_their_evidence():
    df = pd.DataFrame(
        {
            "same": ["a", "a", None, "a"],
            "empty": [None, None, None, None],
            "varied": ["a", "b", "c", "a"],
        }
    )

    res = ae.infer_column_semantics(df)

    findings = {finding.code: finding for finding in res.findings}
    constant = findings["constant_columns"]
    assert constant.assessment == "suspected"
    assert constant.columns == ["same"]
    assert constant.claim == (
        "1 column holds a single distinct non-missing value: 'same'."
    )
    assert constant.fields == ["columns['same'].evidence.distinct_count"]
    assert res.columns["same"].evidence.distinct_count == 1
    all_missing = findings["all_missing_columns"]
    assert all_missing.assessment == "suspected"
    assert all_missing.columns == ["empty"]
    assert all_missing.claim == "1 column contains only missing values: 'empty'."
    assert all_missing.fields == ["columns['empty'].evidence.non_missing_count"]
    assert res.columns["empty"].evidence.non_missing_count == 0
    # A column that varies is in neither finding.
    assert "varied" not in constant.columns + all_missing.columns


def test_flags_numbers_stored_as_text():
    entry = infer(["3.5", "2.25", "3.5", "4", "N/A"])

    assert entry.inferred_logical_type == "float"
    assert entry.flags == ["numbers_as_text"]


def test_numbers_stored_as_numbers_are_not_numbers_as_text():
    assert infer([3.5, 2.25, 3.5, 4.0]).flags == []


def test_flags_values_of_mixed_types():
    entry = infer(pd.Series([1, "a", 2.5, True, None, "a"], dtype=object))

    counts = entry.evidence.value_type_counts
    assert (counts.number, counts.text, counts.boolean) == (2, 2, 1)
    assert "mixed_value_types" in entry.flags


@pytest.mark.parametrize(
    "values",
    [[1, 2.5, None], [1, "a", True, 1.0], ["a", "b"], [1.5, None], [[1], "a"]],
    ids=["int-float", "int-str-bool", "text", "floats", "lists"],
)
def test_mixed_value_types_agree_with_summarize_categorical(values):
    df = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    inferred = ae.infer_column_semantics(df).columns["x"]
    summarized = ae.summarize_categorical(df).columns["x"]

    assert inferred.evidence.value_types == summarized.value_types
    assert ("mixed_value_types" in inferred.flags) == summarized.mixed_value_types


def test_tells_true_from_1_although_pandas_counts_them_as_one_value():
    entry = infer(pd.Series([1, True, 1.0, True, 0], dtype=object))

    counts = entry.evidence.value_type_counts
    assert (counts.number, counts.boolean) == (3, 2)
    assert entry.evidence.number_share == 0.6
    assert entry.evidence.boolean_share == 1.0
    assert entry.evidence.distinct_count == 2
    assert "mixed_value_types" in entry.flags


# Column dtypes.


@pytest.mark.parametrize(
    ("values", "inferred"),
    [
        (pd.Series([1.5, 2.25, None, 7.125]), "float"),
        (pd.Series([1, 2, None, 2], dtype="Int64"), "integer"),
        (pd.Series([1, 2, None, 2], dtype="int64[pyarrow]"), "integer"),
        (pd.Series([1.5, None, 2.5, 1.5], dtype="double[pyarrow]"), "float"),
        (
            pd.Series(
                [Decimal("1.50"), None, Decimal("2.25")],
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            "float",
        ),
        (pd.Series([True, False, True], dtype="bool[pyarrow]"), "boolean"),
        (pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-01"])), "datetime"),
        (
            pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"])).dt.tz_localize(
                "UTC"
            ),
            "datetime",
        ),
        (
            pd.Series([dt.date(2024, 1, 1), None], dtype=pd.ArrowDtype(pa.date32())),
            "datetime",
        ),
        (pd.Series(pd.to_timedelta(["1 day", "2 days", None])), "timedelta"),
        (pd.Series(["a", "b", "a", "a"], dtype="category"), "categorical"),
        (pd.Series([1, 2, 1, 1], dtype="category"), "integer"),
        (pd.Series(["2024-01-05", "2024-01-06"], dtype="string[pyarrow]"), "datetime"),
        (pd.Series(["x", "y", "x"], dtype=pd.ArrowDtype(pa.string())), "categorical"),
        (pd.Series(["x", "y", "x"], dtype="string[python]"), "categorical"),
    ],
)
def test_reads_every_dtype_by_its_values(values, inferred):
    assert infer(values).inferred_logical_type == inferred


def test_a_categorical_counts_only_the_categories_that_occur():
    values = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b", "1"]))

    evidence = infer(values).evidence

    assert evidence.distinct_count == 2
    assert evidence.number_share == 0


def test_nan_a_float_column_stores_as_a_value_counts_as_missing():
    stored = pa.array([1.5, float("nan"), None, 2.5], from_pandas=False)
    res = ae.infer_column_semantics(
        pd.DataFrame({"x": pd.Series(pd.arrays.ArrowExtensionArray(stored))})
    )

    assert res.columns["x"].evidence.non_missing_count == 2
    [limitation] = res.limitations
    assert limitation.code == "nan_counted_as_missing"
    assert limitation.fields == ["columns['x'].evidence.non_missing_count"]


def test_values_that_cant_be_hashed_leave_the_distinct_count_null():
    res = ae.infer_column_semantics(
        pd.DataFrame({"x": pd.Series([[1], "2", [1], {"a": 1}, "2"], dtype=object)})
    )

    entry = res.columns["x"]
    assert entry.evidence.distinct_count is None
    assert entry.evidence.uniqueness_share is None
    assert entry.evidence.value_type_counts.other == 3
    assert entry.evidence.number_share == 0.4
    assert "mixed_value_types" in entry.flags
    assert [lim.code for lim in res.limitations] == ["unhashable_values"]


def test_a_column_whose_values_fit_no_logical_type_is_not_applicable():
    res = ae.infer_column_semantics(
        pd.DataFrame({"x": pd.Series([[1], [2, 3], None], dtype=object)})
    )

    assert res.status == "not_applicable"
    assert res.columns["x"].status_reason
    assert res.columns["x"].status_reason.code == "unsupported_values"
    assert res.columns["x"].candidates == []


# Columns, declared types and statuses.


def frame():
    return pd.DataFrame(
        {
            "zip": ["02134", "10115", "02134", "10115"],
            "amount": [12.5, 3.25, None, 7.0],
            "when": ["2024-01-05", "2024-01-06", "2024-01-05", None],
            "member": ["yes", "no", "no", "yes"],
        }
    )


def test_selects_every_column_by_default_and_keeps_the_callers_order():
    assert list(ae.infer_column_semantics(frame()).columns) == [
        "zip",
        "amount",
        "when",
        "member",
    ]
    picked = ae.infer_column_semantics(frame(), columns=["when", "zip"])
    assert list(picked.columns) == ["when", "zip"]
    assert picked.parameters["columns"] == ["when", "zip"]


def test_an_unknown_column_raises_with_the_closest_name():
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.infer_column_semantics(frame(), columns=["Zip"])

    assert "Did you mean 'zip'?" in str(caught.value)


def test_declared_schema_takes_logical_types_and_column_specs():
    # A dict of a ColumnSpec's fields works as a ColumnSpec, as JSON gives it.
    schema: dict[str, Any] = {
        "zip": "text",
        "amount": ae.ColumnSpec(logical_type="float", nullable=True),
        "when": {"logical_type": "datetime", "nullable": False},
    }

    res = ae.infer_column_semantics(frame(), declared_schema=schema)

    declared = {
        name: entry.declared_logical_type for name, entry in res.columns.items()
    }
    assert declared == {
        "zip": "text",
        "amount": "float",
        "when": "datetime",
        "member": None,
    }
    assert res.parameters["declared_schema"] == {
        "zip": {"logical_type": "text", "nullable": None, "datetime_format": None},
        "amount": {"logical_type": "float", "nullable": True, "datetime_format": None},
        "when": {
            "logical_type": "datetime",
            "nullable": False,
            "datetime_format": None,
        },
    }


def test_never_overwrites_a_declared_type_and_says_where_it_differs():
    res = ae.infer_column_semantics(
        frame(), declared_schema={"zip": "integer", "when": "datetime"}
    )

    zip_code = res.columns["zip"]
    assert zip_code.declared_logical_type == "integer"
    assert zip_code.inferred_logical_type == "categorical"
    assert zip_code.matches_declared is False
    assert res.columns["when"].matches_declared is True
    findings = {finding.code: finding for finding in res.findings}
    mismatch = findings["declared_type_mismatch"]
    assert mismatch.assessment == "suspected"
    assert mismatch.columns == ["zip"]
    assert mismatch.claim == (
        "1 column has a top candidate other than the declared logical type: 'zip'."
    )
    assert mismatch.fields == [
        "columns['zip'].declared_logical_type",
        "columns['zip'].inferred_logical_type",
    ]
    assert findings["declared_type_match"].assessment == "passed"
    assert findings["declared_type_match"].columns == ["when"]


def test_a_declared_column_without_values_cant_be_compared():
    df = pd.DataFrame({"x": [None, None], "y": [1, 2]})

    res = ae.infer_column_semantics(df, declared_schema={"x": "integer"})

    [unchecked] = [f for f in res.findings if f.code == "declared_type_unchecked"]
    assert unchecked.assessment == "unknown"
    assert unchecked.columns == ["x"]
    assert res.columns["x"].matches_declared is None


def test_a_declared_column_the_data_lacks_is_a_violation_with_the_closest_name():
    res = ae.infer_column_semantics(
        frame(), declared_schema={"amout": "float", "zip": "text"}
    )

    assert res.missing_declared_columns == ["amout"]
    [missing] = [f for f in res.findings if f.code == "declared_columns_missing"]
    assert missing.assessment == "violation"
    assert missing.claim == (
        "1 declared column is not in the data: 'amout' (did you mean 'amount'?)."
    )
    assert missing.fields == ["missing_declared_columns"]


def test_without_declared_schema_no_comparison_runs():
    res = ae.infer_column_semantics(frame())

    assert all(e.declared_logical_type is None for e in res.columns.values())
    assert all(e.matches_declared is None for e in res.columns.values())
    assert not [f for f in res.findings if f.code.startswith("declared_")]
    assert (
        "No declared_schema was given, so no column was compared with a declared "
        "logical type."
    ) in str(res).splitlines()


def test_declared_schema_takes_labels_that_are_not_strings():
    df = pd.DataFrame({0: ["a", "b", "a"], 1: [1.5, 2.5, 3.5]})
    # A label that isn't a string is accepted at runtime.
    schema: dict[Any, Any] = {0: "categorical", "1": "float"}

    res = ae.infer_column_semantics(df, declared_schema=schema)

    assert res.columns["0"].declared_logical_type == "categorical"
    assert res.columns["1"].declared_logical_type == "float"
    recorded = res.parameters["declared_schema"]
    assert isinstance(recorded, dict)
    assert list(recorded) == ["0", "1"]


def test_an_unknown_declared_logical_type_raises_with_the_closest():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.infer_column_semantics(frame(), declared_schema={"zip": "int"})  # ty: ignore[invalid-argument-type]  # the typo is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "infer_column_semantics got 'int' at declared_schema['zip'], but each "
        "value in declared_schema must be one of 'boolean', 'integer', 'float', "
        "'categorical', 'text', 'datetime' or 'timedelta', or a ColumnSpec. Did "
        "you mean 'integer'? See help(ae.infer_column_semantics) for every "
        "argument."
    )


def test_an_invalid_column_spec_in_declared_schema_raises_in_the_package_format():
    schema: dict[str, Any] = {"zip": {"logical_type": "txt"}}

    with pytest.raises(InvalidArgumentError) as caught:
        ae.infer_column_semantics(frame(), declared_schema=schema)

    assert str(caught.value) == (
        "infer_column_semantics got an invalid value at declared_schema['zip']: "
        "ColumnSpec got 'txt' for logical_type, but logical_type must be one of "
        "'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or "
        "'timedelta'. Did you mean 'text'? See help(ae.ColumnSpec) for every field."
    )
    assert caught.value.__cause__ is None


def test_a_column_named_twice_in_declared_schema_raises():
    df = pd.DataFrame({1: [1, 2]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.infer_column_semantics(df, declared_schema={1: "integer", "1": "float"})  # ty: ignore[invalid-argument-type]  # labels that aren't strings are accepted at runtime

    assert str(caught.value) == (
        "infer_column_semantics got column '1' more than once in "
        "declared_schema, as 1 and '1'. Name each column once."
    )


def test_declares_exactly_its_statuses():
    [info] = [
        i for i in ae.list_analyses().analyses if i.name == "infer_column_semantics"
    ]

    assert info.statuses == STATUSES


@pytest.mark.parametrize(
    ("df", "status", "code"),
    [
        (frame(), "completed", None),
        (frame().assign(empty=None), "partial", "incomplete_columns"),
        (frame().iloc[:0], "insufficient_data", "no_rows"),
        (pd.DataFrame({"x": [None, np.nan]}), "insufficient_data", "no_values"),
        (pd.DataFrame(index=range(3)), "not_applicable", "no_columns"),
    ],
    ids=["completed", "partial", "no-rows", "no-values", "no-columns"],
)
def test_returns_the_status_the_data_allows(df, status, code):
    res = ae.infer_column_semantics(df)

    assert res.status == status
    assert (res.status_reason.code if res.status_reason else None) == code


def test_the_result_shows_no_raw_data_values():
    secret = "zz-secret-zz"
    df = pd.DataFrame(
        {
            "name": [f"{secret} {i}" for i in range(6)],
            "code": [f"{secret}{i:03d}" for i in range(6)],
            "flag": [secret, "yes", "no", "yes", "no", "yes"],
            "when": [f"0{i}/02/2024" for i in range(1, 7)],
            "amount": [1234.5678, 1234.5678, 1.0, 2.0, 3.0, 4.0],
        }
    )

    res = ae.infer_column_semantics(df, declared_schema={"code": "text"})

    shown = [
        str(res),
        res.to_json(),
        res.to_frame().to_csv(),
        res.to_frame("candidates").to_csv(),
    ]
    for text in shown:
        assert secret not in text
        assert "1234.5678" not in text
        assert "02/2024" not in text


def test_prints_each_column_with_its_top_candidates_and_flags():
    res = ae.infer_column_semantics(frame(), declared_schema={"zip": "integer"})

    lines = str(res).splitlines()
    assert lines[0] == "infer_column_semantics: completed on 4 rows x 4 columns"
    assert lines[1] == (
        "Scores rank logical types by fixed rules and aren't probabilities. Dates "
        "parse as ISO 8601 only."
    )
    assert lines[2] == (
        "declared_schema: 1 column declared: 0 matching (ties with the top "
        "count), 1 not, 0 not inferred, 0 not in the data."
    )
    assert lines[3] == "Columns (4 of 4 shown):"
    header = lines[4].split()
    assert header == [
        "column",
        "inferred",
        "score",
        "next",
        "next_score",
        "declared",
        "flags",
    ]
    assert lines[5].split()[:6] == [
        "zip",
        "categorical",
        "0.5",
        "text",
        "0.25",
        "integer",
    ]
    # The footer names the second table.
    assert "res.to_frame(table='candidates') has every candidate with its reasons" in (
        " ".join(lines[-2:])
    )
    assert max(len(line) for line in lines) <= 120


def test_to_frame_has_the_evidence_and_the_candidates():
    res = ae.infer_column_semantics(frame())

    columns = res.to_frame()
    candidates = res.to_frame("candidates")

    assert list(columns["column"]) == ["zip", "amount", "when", "member"]
    assert list(columns["inferred_logical_type"]) == [
        "categorical",
        "float",
        "datetime",
        "boolean",
    ]
    assert list(columns["dates_as_text"]) == [False, False, True, False]
    assert columns.loc[1, "non_missing_count"] == 3
    assert columns.loc[1, "number_values"] == 3
    assert list(candidates.columns) == [
        "column",
        "rank",
        "logical_type",
        "score",
        "reasons",
    ]
    first = candidates[candidates["column"] == "zip"].iloc[0]
    assert (first["rank"], first["logical_type"]) == (1, "categorical")


def test_to_frame_names_its_tables_when_given_another():
    res = ae.infer_column_semantics(frame())

    with pytest.raises(InvalidArgumentError) as caught:
        res.to_frame("evidence")  # ty: ignore[invalid-argument-type]  # the wrong name is the test

    assert str(caught.value) == (
        "to_frame got 'evidence' for table, but table must be one of 'columns' or "
        "'candidates'. res.to_frame() returns the columns table."
    )


def test_the_record_caps_a_long_declared_schema_and_gives_its_total():
    df = pd.DataFrame({f"c{i}": [i, i + 1] for i in range(150)})
    schema: dict[str, Any] = {f"c{i}": "integer" for i in range(150)}

    res = ae.infer_column_semantics(df, declared_schema=schema)

    record = res.to_dict()
    assert len(record["parameters"]["declared_schema"]) == 100
    assert record["parameters"]["declared_schema_total"] == 150
    assert record["parameters"]["data"] == {"row_count": 2, "column_count": 150}
    assert "data_total" not in record["parameters"]
    recorded = res.parameters["declared_schema"]
    assert isinstance(recorded, dict)
    assert len(recorded) == 150


def test_identical_calls_give_identical_json():
    first = ae.infer_column_semantics(frame(), declared_schema={"zip": "text"})
    second = ae.infer_column_semantics(frame(), declared_schema={"zip": "text"})

    assert first.to_json() == second.to_json()
    json.loads(first.to_json())
