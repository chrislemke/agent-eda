"""A finding code that several analyses report means the same in each.

constant_columns, all_missing_columns, mixed_value_types and infinite_values come
from one builder each, so their code, assessment and claim wording agree across
analyses; what differs is what each analysis documents: a verb, a clause or the
fields that back the claim.
"""

import numpy as np
import pandas as pd

import agent_eda as ae


def findings(res, code):
    return [f for f in res.findings if f.code == code]


def only(res, code):
    (found,) = findings(res, code)
    return found


def test_constant_columns_reads_the_same_in_every_analysis_that_reports_it():
    data = pd.DataFrame({"flag": [1, 1, 1, 1], "n": [1, 2, 3, 4]})
    claim = "1 column holds a single distinct non-missing value: 'flag'."

    for res in (
        ae.describe_dataset(data),
        ae.summarize_numeric(data),
        ae.infer_column_semantics(data),
    ):
        found = only(res, "constant_columns")
        assert (found.claim, found.assessment, found.columns) == (
            claim,
            "suspected",
            ["flag"],
        ), res.analysis


def test_constant_columns_names_the_fields_each_analysis_documents():
    data = pd.DataFrame({"flag": [1, 1, 1, 1], "n": [1, 2, 3, 4]})

    assert only(ae.describe_dataset(data), "constant_columns").fields == [
        "columns[0].distinct_count"
    ]
    assert only(ae.summarize_numeric(data), "constant_columns").fields == [
        "columns['flag'].min",
        "columns['flag'].max",
    ]
    assert only(ae.infer_column_semantics(data), "constant_columns").fields == [
        "columns['flag'].evidence.distinct_count"
    ]


def test_all_missing_columns_takes_the_verb_and_clause_each_analysis_documents():
    data = pd.DataFrame({"gone": [None, None, None], "n": [1, 2, 3]})

    assert (
        only(ae.describe_dataset(data), "all_missing_columns").claim
        == "1 column contains only missing values: 'gone'."
    )
    assert (
        only(ae.infer_column_semantics(data), "all_missing_columns").claim
        == "1 column contains only missing values: 'gone'."
    )
    assert (
        only(ae.profile_missing_values(data), "all_missing_columns").claim
        == "1 column holds only missing values: 'gone'."
    )
    counted = ae.profile_missing_values(data, sentinel_values={"n": [3]})
    assert only(counted, "all_missing_columns").claim == (
        "1 column holds only missing values, counting declared sentinel values: 'gone'."
    )


def test_infinite_values_reads_the_same_for_one_column_or_many():
    data = pd.DataFrame({"a": [1.0, np.inf], "b": [-np.inf, 2.0], "c": [1.0, 2.0]})

    both = only(ae.summarize_numeric(data), "infinite_values")
    one = only(ae.summarize_numeric(data, columns=["a"]), "infinite_values")

    assert both.claim == (
        "2 columns hold infinite values, which every statistic leaves out: 'a' and 'b'."
    )
    assert one.claim == (
        "1 column holds infinite values, which every statistic leaves out: 'a'."
    )
    assert both.assessment == one.assessment == "suspected"


def test_profile_target_reports_infinite_values_with_the_shared_assessment():
    data = pd.DataFrame({"y": [1.0, 2.0, np.inf, 4.0]})

    res = ae.profile_target(data, target_column="y", task="regression")

    found = only(res, "infinite_values")
    assert found.assessment == "suspected"
    assert found.columns == ["y"]
    assert (
        found.claim == "1 target value is infinite, which every statistic leaves out."
    )


def test_mixed_value_types_is_worded_by_what_each_analysis_can_say():
    data = pd.DataFrame({"m": pd.Series([1, "a", 2], dtype=object)})

    listed = only(ae.summarize_categorical(data), "mixed_value_types")
    flagged = only(ae.infer_column_semantics(data), "mixed_value_types")

    assert listed.claim == (
        "1 column holds values of more than one Python type: 'm' (int, str)."
    )
    assert flagged.claim == "1 column mixes values of different types: 'm'."
    assert listed.assessment == flagged.assessment == "suspected"


def test_printed_text_says_where_row_ids_come_from_in_one_wording():
    data = pd.DataFrame({"k": ["a", "a", "b"], "n": ["1", "x", "x"]})

    def printed(id_column):
        return [
            str(ae.find_duplicate_rows(data, id_column=id_column)),
            str(
                ae.validate_column_types(
                    data, schema={"n": "integer"}, id_column=id_column
                )
            ),
            str(ae.detect_missing_sentinels(data, id_column=id_column)),
        ]

    for text in printed(None):
        assert "index labels" in text
    for text in printed("k"):
        assert "values of id_column 'k'" in text
