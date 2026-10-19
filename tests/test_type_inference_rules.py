"""The judgement rules that infer_column_semantics and validate_column_types share.

Each rule has its audit example and a counter-example: a real boolean column, a
real "0123" code and a real ID still read as before.
"""

from decimal import Decimal

import pandas as pd
import pytest

import agent_eda as ae


def infer(values, **arguments):
    data = values if isinstance(values, pd.Series) else pd.Series(values)
    return ae.infer_column_semantics(pd.DataFrame({"x": data}), **arguments).columns[
        "x"
    ]


def validate(values, declared, **arguments):
    data = values if isinstance(values, pd.Series) else pd.Series(values)
    return ae.validate_column_types(
        pd.DataFrame({"x": data}), schema={"x": declared}, **arguments
    ).columns["x"]


def scores(entry):
    return {c.logical_type: c.score for c in entry.candidates}


# Booleans only when the values really are booleans.


@pytest.mark.parametrize(
    "values",
    [
        ["A", "B", "C", "F", "F", "F", "F", "F"],
        ["F"] * 9 + ["M"],
        ["N"] * 7 + ["E", "S", "W"],
    ],
    ids=["grades", "sex", "compass"],
)
def test_letter_codes_with_a_boolean_letter_rank_categorical_not_boolean(values):
    entry = infer(values)

    assert entry.inferred_logical_type == "categorical"
    assert "boolean" not in scores(entry)
    assert "boolean_like" not in entry.flags


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 1, 0, 1],
        ["yes", "no", "no", "yes", "YES"],
        ["Y", "N", "N", "Y"],
        ["t", "f", "t", None],
        [True, False, True],
        ["0", "1", "yes", "no"],
    ],
)
def test_a_real_boolean_column_still_ranks_boolean_first(values):
    entry = infer(values)

    assert entry.inferred_logical_type == "boolean"
    assert scores(entry)["boolean"] == 1


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 1, 1],
        [0, 0, 0],
        [0.0, 0.0],
        ["yes", "yes", "yes"],
        pd.Series([1, 1, 1], dtype="Int64"),
    ],
)
def test_a_constant_column_is_never_boolean(values):
    entry = infer(values)

    assert "boolean" not in scores(entry)
    assert entry.inferred_logical_type != "boolean"
    assert "boolean_like" not in entry.flags
    assert "constant" in entry.flags


def test_a_constant_column_stored_as_booleans_still_ranks_boolean():
    entry = infer(pd.Series([True, True, True]))

    assert entry.inferred_logical_type == "boolean"
    assert "constant" in entry.flags
    assert "boolean_like" not in entry.flags


def test_a_constant_column_does_not_convert_to_boolean_by_inference_alone():
    # validate_column_types judges by value, so declaring boolean still works.
    entry = validate([1, 1, 1, 1], "boolean")

    assert (entry.convertible_count, entry.invalid_count) == (4, 0)


def test_a_stray_value_that_isnt_a_boolean_keeps_a_column_from_ranking_boolean():
    entry = infer(["yes", "no", "yes", "no", "maybe"])

    assert "boolean" not in scores(entry)


# Leading-zero codes only for whole numbers.


def test_decimal_text_with_leading_zeros_ranks_float_not_a_code():
    values = ["05.50", "12.00", "07.25", "10.10"]

    entry = infer(values)

    assert entry.inferred_logical_type == "float"
    assert "identifier_like" not in entry.flags
    assert entry.evidence.leading_zero_share == 0
    assert all("leading zero" not in r for c in entry.candidates for r in c.reasons)
    assert "numbers_as_text" in entry.flags


@pytest.mark.parametrize("text", ["05.50", "01.5", "00.5", "-007.25", "00.5e1"])
def test_validate_counts_decimal_text_with_leading_zeros_as_float(text):
    entry = validate([text, "1.5"], "float")

    assert (entry.convertible_count, entry.invalid_count) == (2, 0)


@pytest.mark.parametrize("text", ["0123", "007", "01234", "-007"])
def test_a_whole_number_with_a_leading_zero_is_still_a_code(text):
    entry = infer([text, "10115", "80331", "10115"])

    assert entry.evidence.leading_zero_share == 0.25
    assert scores(entry)["integer"] == scores(entry)["float"] == 0
    assert validate([text], "integer").invalid_count == 1
    assert validate([text], "float").invalid_count == 1


# Date-like text is never identifier_like.


@pytest.mark.parametrize("pattern", ["%m/%d/%Y", "%d.%m.%Y"])
def test_distinct_non_iso_date_text_is_not_identifier_like(pattern):
    days = pd.date_range("2024-01-13", periods=60, freq="D")
    entry = infer(list(days.strftime(pattern)))

    assert "non_iso_date_text" in entry.flags
    assert "identifier_like" not in entry.flags


def test_a_declared_datetime_format_does_not_make_date_text_identifier_like():
    days = pd.date_range("2024-01-13", periods=60, freq="D")
    spec = ae.ColumnSpec("datetime", datetime_format="%m/%d/%Y")

    entry = infer(list(days.strftime("%m/%d/%Y")), declared_schema={"x": spec})

    assert "identifier_like" not in entry.flags
    assert entry.matches_declared is True


# identifier_like is judged after dropping exact duplicate rows.


def customers(rows=5000, repeated=20):
    ids = [f"C{i:06d}" for i in range(rows - repeated)]
    ids += ids[:repeated]
    return pd.DataFrame(
        {
            "customer": ids,
            "segment": [("a", "b", "c")[i % 3] for i in range(rows)],
            "score": [i % 7 for i in range(rows)],
        }
    )


def test_an_id_column_with_a_few_exactly_duplicated_rows_is_identifier_like():
    data = customers()
    # The repeated rows are copies of whole rows.
    data.iloc[-20:, 1:] = data.iloc[:20, 1:].to_numpy()
    assert data.duplicated().sum() == 20

    res = ae.infer_column_semantics(data)

    assert "identifier_like" in res.columns["customer"].flags
    assert "identifier_like" not in res.columns["segment"].flags
    assert "identifier_like" not in res.columns["score"].flags


def test_an_id_that_repeats_in_rows_that_differ_is_not_identifier_like():
    data = customers()  # same ID, different segment or score

    assert data.duplicated().sum() == 0
    res = ae.infer_column_semantics(data)

    assert "identifier_like" not in res.columns["customer"].flags


def test_the_duplicate_rows_are_read_over_every_column_not_only_the_selected_ones():
    data = customers()
    data.iloc[-20:, 1:] = data.iloc[:20, 1:].to_numpy()

    res = ae.infer_column_semantics(data, columns=["customer"])

    assert "identifier_like" in res.columns["customer"].flags


def test_a_real_id_column_is_still_identifier_like():
    entry = infer([f"U{i}" for i in range(50)])

    assert entry.flags == ["identifier_like"]


# A declared type that ties with the top candidate is a match.


def test_a_zero_one_column_declared_integer_matches():
    res = ae.infer_column_semantics(
        pd.DataFrame({"flag": [0, 1, 1, 0, 1]}), declared_schema={"flag": "integer"}
    )

    entry = res.columns["flag"]
    assert entry.inferred_logical_type == "boolean"
    assert entry.matches_declared is True
    assert [f.code for f in res.findings if f.code.startswith("declared_")] == [
        "declared_type_match"
    ]


def test_the_docstring_amount_example_declared_float_matches():
    df = pd.DataFrame({"amount": [12.0, 30.0, None, 7.0, 12.0, 45.0]})

    res = ae.infer_column_semantics(df, declared_schema={"amount": "float"})

    assert res.columns["amount"].matches_declared is True
    assert "declared_type_mismatch" not in [f.code for f in res.findings]


def test_a_declared_type_below_the_top_score_still_mismatches():
    res = ae.infer_column_semantics(
        pd.DataFrame({"x": [0, 1, 1, 0, 1]}), declared_schema={"x": "text"}
    )

    assert res.columns["x"].matches_declared is False
    assert "declared_type_mismatch" in [f.code for f in res.findings]


# A declared datetime_format is honoured.


def test_a_declared_datetime_format_matches_text_in_that_format():
    days = pd.date_range("2024-01-13", periods=6, freq="D")
    text = list(days.strftime("%d/%m/%Y")) * 2
    spec = ae.ColumnSpec("datetime", datetime_format="%d/%m/%Y")

    res = ae.infer_column_semantics(
        pd.DataFrame({"x": text}), declared_schema={"x": spec}
    )

    entry = res.columns["x"]
    assert entry.inferred_logical_type == "datetime"
    assert entry.matches_declared is True
    assert entry.evidence.datetime_share == 1
    assert entry.evidence.non_iso_date_share == 0
    assert "non_iso_date_text" not in entry.flags
    assert "declared_type_mismatch" not in [f.code for f in res.findings]
    # validate_column_types agrees.
    assert validate(text, spec).convertible_count == 12


def test_without_a_declared_format_the_same_text_is_non_iso_date_text():
    days = pd.date_range("2024-01-13", periods=6, freq="D")
    text = list(days.strftime("%d/%m/%Y")) * 2

    entry = infer(text)

    assert entry.inferred_logical_type != "datetime"
    assert "non_iso_date_text" in entry.flags


def test_a_datetime_format_other_than_the_texts_changes_nothing():
    spec = ae.ColumnSpec("datetime", datetime_format="%Y%m%d")
    text = ["13/01/2024", "14/01/2024", "13/01/2024"]

    entry = infer(text, declared_schema={"x": spec})

    assert entry.evidence.datetime_share == 0
    assert entry.matches_declared is False
    assert "non_iso_date_text" in entry.flags


# Text that overflows float64 doesn't count as a number.


@pytest.mark.parametrize("text", ["1e400", "-1e400", "9e999", "1" + "0" * 400])
def test_text_that_overflows_float64_doesnt_convert_to_float(text):
    entry = validate([text, "1"], "float")

    assert (entry.convertible_count, entry.invalid_count) == (1, 1)
    assert validate([text, "1"], "integer").invalid_count == 1


def test_text_that_overflows_float64_is_no_float_share_in_inference():
    entry = infer(["1e400", "-1e400", "2.5", "3.5"])

    assert entry.evidence.number_share == 0.5
    assert entry.evidence.integer_share == 0
    assert entry.evidence.plain_text_share == 0.5


def test_the_largest_float64_text_still_converts():
    assert validate(["1.7976931348623157e308", "-1e308"], "float").invalid_count == 0


def test_decimals_beyond_float64_in_an_object_column_give_a_result():
    values = pd.Series(
        [Decimal("1E+400"), Decimal("1E+400"), Decimal("2")], dtype=object
    )

    res = ae.infer_column_semantics(pd.DataFrame({"x": values}))

    assert res.status == "completed"
    res.to_json()
    ae.validate_column_types(
        pd.DataFrame({"x": values}), schema={"x": "float"}
    ).to_json()


# With no type inferred, the printed table and to_frame show no top type or score.


def test_no_top_type_or_score_is_shown_when_no_candidate_scores_above_zero():
    # Values that can't be hashed are never scored categorical or text, and the
    # leading zero rules integer and float out at a score of 0.
    data = pd.DataFrame({"x": pd.Series([["a"], "007", ["b"]], dtype=object)})

    res = ae.infer_column_semantics(data)

    entry = res.columns["x"]
    assert entry.inferred_logical_type is None
    assert [c.score for c in entry.candidates] == [0, 0]
    frame = res.to_frame()
    assert pd.isna(frame.loc[0, "score"])
    assert pd.isna(frame.loc[0, "inferred_logical_type"])
    table = [line for line in str(res).splitlines() if line.startswith("x ")]
    assert table
    assert "integer" not in table[0]
    assert "float" not in table[0]
    # The candidates table still lists them, scored 0.
    assert len(res.to_frame(table="candidates")) == 2


def test_the_docstring_says_what_candidates_holds():
    flat = " ".join((ae.infer_column_semantics.__doc__ or "").split())

    assert "integer and float are listed too, scored 0" in flat
    assert "A column where every candidate scores 0 has no inferred type" in flat
