"""Texts that describe behaviour: the behaviour matches what the texts say."""

import inspect

import pandas as pd
import pytest

import agent_eda as ae


def words(text):
    return " ".join(text.split())


def doc(analysis):
    return words(inspect.getdoc(analysis) or "")


def field_description(analysis, path):
    schema = getattr(
        ae.results, ae.describe_analysis(analysis).result_class
    ).record_schema()
    node = schema
    for part in path:
        if "$ref" in node:
            node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        node = node["properties"][part]
    return words(node["description"])


def test_a_tie_between_categorical_and_text_goes_to_categorical_on_exact_fractions():
    # 5 values, 2 distinct, 4 of 5 of several words: categorical scores 3/5 and
    # text (5/5 * 2/5 + 4/5) / 2 = 3/5, which float arithmetic makes 0.6000000000000001.
    data = pd.DataFrame({"x": ["a b", "a b", "a b", "a b", "c"]})

    entry = ae.infer_column_semantics(data).columns["x"]

    assert [(c.logical_type, c.score) for c in entry.candidates] == [
        ("categorical", 0.6),
        ("text", 0.6),
    ]
    assert entry.inferred_logical_type == "categorical"


def test_a_column_without_values_is_listed_among_the_columns_without_rules():
    data = pd.DataFrame({"flag": pd.array([None, None, None], dtype="boolean")})

    res = ae.detect_missing_sentinels(data)

    assert res.columns_without_rules == ["flag"]
    description = field_description(
        "detect_missing_sentinels", ["columns_without_rules"]
    )
    assert "A column without values is listed too." in description
    assert "isn't listed" not in description


def test_insufficient_data_is_documented_for_an_all_missing_column_next_to_lists():
    data = pd.DataFrame({"gone": [None, None, None], "lists": [[1], [2], [3]]})

    for analysis in (ae.summarize_categorical, ae.infer_column_semantics):
        res = analysis(data)
        assert res.status == "insufficient_data"
        assert res.status_reason is not None
        assert res.status_reason.code == "no_values"
        assert "no selected column completed and at least one of them has no" in doc(
            analysis
        )


def test_the_candidates_of_a_leading_zero_column_include_types_that_score_zero():
    data = pd.DataFrame({"zip": ["01234", "10115", "01234"]})

    entry = ae.infer_column_semantics(data).columns["zip"]

    assert [c.score for c in entry.candidates if c.logical_type == "integer"] == [0.0]
    assert "the logical types scoring above 0" not in doc(ae.infer_column_semantics)


def test_the_invalid_values_of_a_column_are_a_count_in_the_columns_table():
    data = pd.DataFrame({"n": ["1", "x", "y"]})

    res = ae.validate_column_types(
        data, schema={"n": ae.ColumnSpec(logical_type="integer")}
    )

    frame = res.to_frame()
    assert frame["invalid_distinct_count"].tolist() == [2]
    assert "invalid_values" not in frame.columns
    description = words(res.to_frame.__doc__)
    assert "the distinct invalid values," not in description
    assert "the number of distinct invalid values" in description


def test_the_value_types_of_both_records_are_capped_at_100_with_their_total():
    classes = [type(f"T{i:03d}", (), {}) for i in range(150)]
    data = pd.DataFrame({"x": pd.Series([c() for c in classes], dtype=object)})

    inferred = ae.infer_column_semantics(data).to_dict()["columns"]["x"]["evidence"]
    counted = ae.summarize_categorical(data).to_dict()["columns"]["x"]

    for record in (inferred, counted):
        assert len(record["value_types"]) == 100
        assert record["value_types_total"] == 150


@pytest.mark.parametrize("name", ["infer_column_semantics", "validate_column_types"])
def test_the_schema_argument_says_that_locale_dependent_directives_follow_the_locale(
    name,
):
    parameters = {p.name: p for p in ae.describe_analysis(name).parameters}

    assert "process's locale" in words(parameters["schema"].description)


def test_the_datetime_format_of_a_column_spec_says_it_depends_on_the_locale():
    schema = ae.ColumnSpec.model_json_schema()

    description = words(schema["properties"]["datetime_format"]["description"])
    assert "locale" in description
    assert "%b" in description


def error_text(call):
    with pytest.raises(ae.errors.ColumnTypeError) as caught:
        call()
    return str(caught.value)


def number_messages(values, analyses):
    data = pd.DataFrame({"x": values, "y": [1.0, 2.0, 3.0]})
    calls = {
        "summarize_numeric": lambda: ae.summarize_numeric(data, columns=["x"]),
        "compute_association_matrix": lambda: ae.compute_association_matrix(
            data, columns=["x", "y"]
        ),
        "profile_target": lambda: ae.profile_target(
            data, target_column="x", task="regression"
        ),
    }
    return {name: error_text(calls[name]) for name in analyses}


@pytest.mark.parametrize(
    ("values", "advice", "analyses"),
    [
        (
            pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "onvert datetimes to numbers with "
            "(df['x'] - df['x'].min()).dt.total_seconds()",
            ["summarize_numeric", "compute_association_matrix", "profile_target"],
        ),
        (
            pd.to_timedelta(["1D", None, "3D"]),
            "onvert durations to numbers with df['x'].dt.total_seconds()",
            ["summarize_numeric", "compute_association_matrix", "profile_target"],
        ),
        (
            pd.Series(["1", "2", None]),
            "onvert numbers stored as text with "
            "pd.to_numeric(df['x'], errors='coerce')",
            ["summarize_numeric", "profile_target"],
        ),
        (
            pd.Categorical([1, 2, 1]),
            "onvert numeric categories with df['x'].astype(float)",
            ["summarize_numeric", "profile_target"],
        ),
    ],
    ids=["datetime", "timedelta", "text", "categorical"],
)
def test_advice_to_convert_a_column_to_numbers_is_worded_the_same_in_every_analysis(
    values, advice, analyses
):
    messages = number_messages(values, analyses)

    for name in analyses:
        assert advice in messages[name], name


@pytest.mark.parametrize(
    ("values", "advice"),
    [
        (
            pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "onvert datetimes to labels with df['x'].astype(str), or to months with "
            "df['x'].dt.strftime('%Y-%m')",
        ),
        (
            pd.to_timedelta(["1D", None, "3D"]),
            "onvert durations to labels with df['x'].astype(str)",
        ),
    ],
    ids=["datetime", "timedelta"],
)
def test_advice_to_convert_a_column_to_labels_is_worded_the_same_in_every_analysis(
    values, advice
):
    data = pd.DataFrame({"x": values, "y": ["a", "b", "a"]})
    messages = {
        "summarize_categorical": error_text(
            lambda: ae.summarize_categorical(data, columns=["x"])
        ),
        "compute_association_matrix": error_text(
            lambda: ae.compute_association_matrix(data, categorical_columns=["x"])
        ),
        "profile_target": error_text(
            lambda: ae.profile_target(
                data, target_column="x", task="multiclass_classification"
            )
        ),
    }

    for name, text in messages.items():
        assert advice in text, name


def test_a_datetime_named_in_summarize_numeric_gets_a_conversion():
    data = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-03-01"])})

    text = error_text(lambda: ae.summarize_numeric(data, columns=["when"]))

    assert "(df['when'] - df['when'].min()).dt.total_seconds()" in text
