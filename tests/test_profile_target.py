import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from recipes import column_named, is_number, label_columns, names
from strategies import data_frames

import agent_eda as ae
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError, InvalidArgumentError

INF = np.inf


def filled[T](values: list[T] | None) -> list[T]:
    """A list field that the task and options fill, such as classes."""
    assert values is not None
    return values


def customers():
    """Ten customers: churned is 'no' for seven and 'yes' for three."""
    return pd.DataFrame(
        {
            "customer": [f"c{i}" for i in range(10)],
            "churned": ["no", "no", "yes", "no", "no", "yes", "no", "no", "yes", "no"],
            "plan": ["basic", "pro", "basic", "team", "pro"] * 2,
            "spend": [12.5, 40.0, 7.25, 99.0, 35.0, 0.0, 18.0, 52.5, 3.0, 27.0],
        }
    )


def test_target_column_and_task_are_required():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(customers())  # ty: ignore[missing-argument]  # leaving them out is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "profile_target is missing 2 required arguments: target_column and task. "
        "Pass them by keyword, such as ae.profile_target(df, target_column=..., "
        "task=...), where task is one of 'binary_classification', "
        "'multiclass_classification' or 'regression'. See help(ae.profile_target) "
        "for every argument."
    )


def test_the_call_takes_the_target_the_task_and_optional_checks_without_defaults():
    described = {p.name: p for p in ae.describe_analysis("profile_target").parameters}

    assert list(described) == [
        "data",
        "target_column",
        "task",
        "expected_labels",
        "rare_share",
        "id_column",
        "project",
        "mask_sensitive",
    ]
    assert [name for name, p in described.items() if p.required] == [
        "data",
        "target_column",
        "task",
    ]
    assert described["task"].allowed_values == [
        "binary_classification",
        "multiclass_classification",
        "regression",
    ]
    assert described["expected_labels"].default is None
    assert described["rare_share"].default is None
    assert described["rare_share"].valid_range == "0 < rare_share < 1"


def test_an_unknown_task_raises_with_the_allowed_tasks():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(customers(), target_column="churned", task="classification")  # ty: ignore[invalid-argument-type]  # a wrong task is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "profile_target got 'classification' for task, but task must be one of "
        "'binary_classification', 'multiclass_classification' or 'regression'. "
        "Did you mean 'binary_classification'? See help(ae.profile_target) for "
        "every argument."
    )


def test_classes_come_with_counts_and_shares_of_the_valid_labels():
    churned = ["no", "yes", "no", None, "no", "yes", "no", "no", None, "no"]
    df = pd.DataFrame({"churned": churned})

    res = ae.profile_target(df, target_column="churned", task="binary_classification")

    assert res.status == "completed"
    assert [(c.label, c.label_type, c.count, c.share) for c in filled(res.classes)] == [
        ("no", "str", 6, 0.75),
        ("yes", "str", 2, 0.25),
    ]
    assert res.class_count == 2
    # Without expected_labels, every label that isn't missing is valid.
    assert res.valid_label_count == 8
    assert res.non_missing_count == 8
    assert res.missing_count == 2
    assert res.missing_share == 0.2
    assert res.majority_share == 0.75
    assert res.minority_share == 0.25
    # The majority count over the minority count: 6 / 2.
    assert res.imbalance_ratio == 3.0
    expected = df["churned"].value_counts(normalize=True)
    assert {c.label: c.share for c in filled(res.classes)} == expected.to_dict()


def test_labels_outside_expected_labels_come_with_counts_and_example_row_ids():
    df = pd.DataFrame(
        {"churned": ["no", "yes", "No", "no", "unknown", "No", "yes", "no"]},
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )

    res = ae.profile_target(
        df,
        target_column="churned",
        task="binary_classification",
        expected_labels=["no", "yes"],
    )

    assert [
        (u.label, u.label_type, u.count, u.row_ids)
        for u in filled(res.unexpected_labels)
    ] == [("No", "str", 2, [12, 15]), ("unknown", "str", 1, [14])]
    assert res.unexpected_label_count == 2
    # The shares divide by the 5 labels that expected_labels holds.
    assert [(c.label, c.count, c.share) for c in filled(res.classes)] == [
        ("no", 3, 0.6),
        ("yes", 2, 0.4),
    ]
    assert res.valid_label_count == 5
    assert res.non_missing_count == 8
    assert res.class_count == 2


def test_each_unexpected_label_keeps_its_first_10_row_ids_and_counts_them_all():
    df = pd.DataFrame({"y": ["a", "b", *["?"] * 12]})

    res = ae.profile_target(
        df, target_column="y", task="binary_classification", expected_labels=["a", "b"]
    )

    [unexpected] = filled(res.unexpected_labels)
    assert unexpected.count == 12
    assert unexpected.row_ids == list(range(2, 12))
    record = res.to_dict()["unexpected_labels"][0]
    assert record["row_ids"] == list(range(2, 12))
    assert record["row_ids_total"] == 12


def test_rare_classes_are_the_classes_whose_share_lies_below_rare_share():
    df = pd.DataFrame({"plan": ["basic"] * 6 + ["pro"] * 3 + ["team"]})

    res = ae.profile_target(
        df, target_column="plan", task="multiclass_classification", rare_share=0.3
    )

    # team has a share of 0.1; pro's 0.3 isn't below 0.3.
    assert [(c.label, c.count, c.share) for c in filled(res.rare_classes)] == [
        ("team", 1, 0.1)
    ]
    assert res.rare_class_count == 1


def test_a_regression_target_gets_the_fields_and_numbers_summarize_numeric_gives():
    price = [10.0, 12.5, None, INF, 7.0, 30.0, 12.5, -INF, 8.0, 11.0]
    df = pd.DataFrame({"price": price, "plan": ["basic", "pro"] * 5})

    res = ae.profile_target(df, target_column="price", task="regression")
    numeric = ae.summarize_numeric(df, columns=["price"]).columns["price"]

    assert res.status == "completed"
    assert res.statistics is not None
    assert res.statistics.model_dump() == numeric.model_dump()
    assert res.statistics.mean == pytest.approx(13.0)
    # One missing value and two infinite ones.
    assert res.non_finite_count == 3
    assert res.missing_count == 1
    assert res.non_missing_count == 9
    assert res.missing_share == 0.1
    assert res.classes is None
    assert res.class_count is None
    assert res.imbalance_ratio is None


def test_without_expected_labels_or_rare_share_the_result_says_they_werent_checked():
    res = ae.profile_target(
        customers(), target_column="churned", task="binary_classification"
    )

    assert res.unexpected_labels is None
    assert res.unexpected_label_count is None
    assert res.rare_classes is None
    assert res.rare_class_count is None
    assert [(lim.code, lim.fields) for lim in res.limitations] == [
        (
            "expected_labels_not_declared",
            ["unexpected_label_count", "unexpected_labels"],
        ),
        ("rare_share_not_declared", ["rare_class_count", "rare_classes"]),
    ]
    assert [lim.reason for lim in res.limitations] == [
        "Labels outside an expected set weren't checked: pass expected_labels to "
        "check them.",
        "Rare classes weren't checked: pass rare_share, the share below which a "
        "class is rare.",
    ]
    # Each printed limitation keeps its fix within the 120-character line.
    assert all(not line.endswith("...") for line in str(res).splitlines())
    record = res.to_dict()
    assert record["unexpected_labels"] is None
    assert record["unexpected_labels_total"] is None
    assert record["rare_classes"] is None
    assert record["rare_classes_total"] is None


def test_with_expected_labels_and_rare_share_both_checks_run():
    res = ae.profile_target(
        customers(),
        target_column="churned",
        task="binary_classification",
        expected_labels=["no", "yes"],
        rare_share=0.05,
    )

    assert res.unexpected_labels == []
    assert res.unexpected_label_count == 0
    assert res.rare_classes == []
    assert res.rare_class_count == 0
    assert res.limitations == []


def findings_of(res):
    return [(f.code, f.assessment, f.claim, f.columns, f.fields) for f in res.findings]


@pytest.mark.parametrize("task", ["binary_classification", "multiclass_classification"])
def test_a_single_class_is_a_violation(task):
    df = pd.DataFrame({"churned": ["no", "no", None, "no"]})

    res = ae.profile_target(df, target_column="churned", task=task)

    assert ("single_class", "violation") in [
        (f.code, f.assessment) for f in res.findings
    ]
    [single] = [f for f in res.findings if f.code == "single_class"]
    assert single.claim == (
        "Every valid label is the same class, so a classifier has nothing to tell "
        "apart."
    )
    assert single.columns == ["churned"]
    assert single.fields == ["class_count", "classes"]


def test_a_binary_task_with_more_than_two_classes_is_a_violation():
    df = pd.DataFrame({"churned": ["no", "yes", "maybe", "no", "yes"]})

    binary = ae.profile_target(
        df, target_column="churned", task="binary_classification"
    )
    multiclass = ae.profile_target(
        df, target_column="churned", task="multiclass_classification"
    )

    assert findings_of(binary) == [
        (
            "more_than_two_classes",
            "violation",
            "The target has 3 classes, more than the 2 of binary_classification.",
            ["churned"],
            ["class_count", "classes"],
        )
    ]
    assert findings_of(multiclass) == []


def test_labels_outside_expected_labels_are_a_violation():
    df = pd.DataFrame({"churned": ["no", "yes", "No", "no", "unknown", "No"]})

    res = ae.profile_target(
        df,
        target_column="churned",
        task="binary_classification",
        expected_labels=["no", "yes"],
    )

    assert findings_of(res) == [
        (
            "unexpected_labels",
            "violation",
            "2 labels in 3 rows are outside expected_labels.",
            ["churned"],
            ["unexpected_label_count", "unexpected_labels"],
        )
    ]


def test_a_declared_label_set_that_every_label_keeps_passes():
    res = ae.profile_target(
        customers(),
        target_column="churned",
        task="binary_classification",
        expected_labels=["yes", "no"],
    )

    assert findings_of(res) == [
        (
            "no_unexpected_labels",
            "passed",
            "Every label that isn't missing is in expected_labels.",
            ["churned"],
            ["unexpected_label_count"],
        )
    ]


def test_a_classification_target_whose_values_are_all_distinct_is_suspected():
    df = pd.DataFrame({"spend": [12.5, 40.0, None, 7.25, 99.0]})

    res = ae.profile_target(df, target_column="spend", task="multiclass_classification")

    assert findings_of(res) == [
        (
            "all_labels_distinct",
            "suspected",
            "All 4 target values that aren't missing are distinct, as in an "
            "identifier or a measurement rather than class labels.",
            ["spend"],
            ["class_count", "non_missing_count"],
        ),
        (
            "missing_labels",
            "descriptive",
            "1 of 5 rows (20%) has no label: its target value is missing.",
            ["spend"],
            ["missing_count", "missing_share"],
        ),
    ]


def test_a_repeated_label_is_enough_for_values_not_to_be_all_distinct():
    df = pd.DataFrame({"plan": ["basic", "pro", "basic"]})

    res = ae.profile_target(df, target_column="plan", task="binary_classification")

    assert res.findings == []


def test_rare_classes_are_described():
    df = pd.DataFrame({"plan": ["basic"] * 6 + ["pro"] * 3 + ["team"]})

    res = ae.profile_target(
        df, target_column="plan", task="multiclass_classification", rare_share=0.3
    )

    assert findings_of(res) == [
        (
            "rare_classes",
            "descriptive",
            "1 of 3 classes has a share below rare_share 0.3.",
            ["plan"],
            ["rare_class_count", "rare_classes"],
        )
    ]


def test_a_constant_regression_target_is_a_violation():
    df = pd.DataFrame({"price": [9.5, 9.5, None, 9.5]})

    res = ae.profile_target(df, target_column="price", task="regression")

    assert findings_of(res) == [
        (
            "constant_target",
            "violation",
            "Every target value that isn't missing is the same, so a regression has "
            "nothing to predict.",
            ["price"],
            ["statistics.min", "statistics.max"],
        ),
        (
            "missing_labels",
            "descriptive",
            "1 of 4 rows (25%) has no label: its target value is missing.",
            ["price"],
            ["missing_count", "missing_share"],
        ),
    ]


def test_a_regression_target_of_one_infinite_value_is_constant_too():
    df = pd.DataFrame({"price": [INF, INF]})

    res = ae.profile_target(df, target_column="price", task="regression")

    assert [(f.code, f.fields) for f in res.findings] == [
        ("constant_target", ["statistics.infinite_count"]),
        ("infinite_values", ["statistics.infinite_count", "non_finite_count"]),
    ]


def test_infinite_regression_target_values_are_suspected():
    df = pd.DataFrame({"price": [9.5, INF, 3.0, -INF]})

    res = ae.profile_target(df, target_column="price", task="regression")

    assert findings_of(res) == [
        (
            "infinite_values",
            "suspected",
            "2 target values are infinite, which every statistic leaves out.",
            ["price"],
            ["statistics.infinite_count", "non_finite_count"],
        )
    ]


def test_two_classes_fit_a_binary_task_without_a_finding():
    res = ae.profile_target(
        customers(), target_column="churned", task="binary_classification"
    )

    assert res.findings == []


def test_an_unknown_target_column_raises_with_the_closest_names():
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.profile_target(
            customers(), target_column="Churned ", task="binary_classification"
        )

    assert caught.value.code == "column_not_found"
    assert str(caught.value) == (
        "profile_target got an unknown column 'Churned ' in target_column. Did you "
        "mean 'churned'? The data has 4 columns: 'customer', 'churned', 'plan' and "
        "'spend'."
    )


def not_numeric():
    """Columns that a regression can't take: text, categories, booleans and times."""
    return pd.DataFrame(
        {
            "plan": ["basic", "pro", "basic"],
            "grade": pd.Categorical(["1", "2", "1"]),
            "churned": [True, False, True],
            "wait": pd.to_timedelta([1, 2, 3], unit="h"),
            "signed_up": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        }
    )


CLASSIFY = (
    "If its values are class labels, pass task='binary_classification' or "
    "task='multiclass_classification'"
)


@pytest.mark.parametrize(
    ("column", "fix"),
    [
        (
            "plan",
            f"{CLASSIFY}; if they are numbers stored as text, convert them with "
            "pd.to_numeric(df['plan'], errors='coerce').",
        ),
        (
            "grade",
            f"{CLASSIFY}; if its categories are numbers, convert them with "
            "df['grade'].astype(float).",
        ),
        (
            "churned",
            "If its values are class labels, pass task='binary_classification'; to "
            "read False and True as 0 and 1, convert them with "
            "df['churned'].astype(float).",
        ),
        (
            "wait",
            "Convert durations to numbers first, such as "
            "df['wait'].dt.total_seconds().",
        ),
        (
            "signed_up",
            "Convert datetimes to numbers first, such as the seconds since the "
            "earliest: (df['signed_up'] - df['signed_up'].min()).dt.total_seconds().",
        ),
    ],
)
def test_a_regression_task_on_a_column_that_isnt_numeric_raises_with_the_fix(
    column, fix
):
    df = not_numeric()

    with pytest.raises(ColumnTypeError) as caught:
        ae.profile_target(df, target_column=column, task="regression")

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        f"profile_target can't use column {column!r} with dtype {df[column].dtype} "
        "as a regression target, which must hold numbers: integer or float "
        "columns, nullable and pyarrow-backed ones of those kinds included. " + fix
    )


@pytest.mark.parametrize("column", ["wait", "signed_up"])
def test_a_classification_task_on_times_raises_with_the_fix(column):
    df = not_numeric()

    with pytest.raises(ColumnTypeError) as caught:
        ae.profile_target(df, target_column=column, task="multiclass_classification")

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        f"profile_target can't use column {column!r} with dtype {df[column].dtype} "
        "as a classification target, whose labels must be text, numbers, booleans "
        "or categories, nullable and pyarrow-backed ones of those kinds included. "
        "Convert its "
        f"values to labels first, such as df[{column!r}].astype(str)."
    )


def test_a_classification_target_whose_values_cant_be_hashed_raises():
    df = pd.DataFrame({"tags": [["a"], ["b"], ["a"]]})

    with pytest.raises(ColumnTypeError) as caught:
        ae.profile_target(df, target_column="tags", task="multiclass_classification")

    assert caught.value.code == "unsupported_column_type"
    assert str(caught.value) == (
        "profile_target can't use column 'tags' with dtype object as a "
        "classification target: it holds values that can't be hashed, such as "
        "lists or dicts, and so can't be class labels. Pass a column of labels "
        "such as strings or numbers."
    )


@pytest.mark.parametrize(
    ("options", "given", "them"),
    [
        ({"expected_labels": [0, 1]}, "expected_labels", "it"),
        ({"rare_share": 0.05}, "rare_share", "it"),
        (
            {"expected_labels": [0, 1], "rare_share": 0.05},
            "expected_labels and rare_share",
            "them",
        ),
    ],
)
def test_class_checks_on_a_regression_task_raise(options, given, them):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            customers(), target_column="spend", task="regression", **options
        )

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"profile_target got {given} for a regression task, which has no class "
        f"labels to check. Leave {them} out, or pass "
        "task='binary_classification' or task='multiclass_classification' if the "
        "target holds class labels."
    )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (
            [],
            "profile_target got an empty list for expected_labels. List the labels "
            "the task allows, or pass expected_labels=None to leave the check out.",
        ),
        (
            ["no", "yes", "no"],
            "profile_target got 'no' more than once in expected_labels. List each "
            "label once.",
        ),
        (
            ["no", float("nan")],
            "profile_target got nan in expected_labels, but NaN and infinity can't "
            "be class labels; missing labels are counted apart, in missing_count. "
            "Leave them out of expected_labels.",
        ),
        (
            ["no", "yes", "maybe"],
            "profile_target got 3 labels in expected_labels for a "
            "binary_classification task, which has 2 classes. List its 2 labels, or "
            "pass task='multiclass_classification'.",
        ),
        (
            ["no"],
            "profile_target got 1 label in expected_labels for a "
            "binary_classification task, which has 2 classes. List its 2 labels, or "
            "pass task='multiclass_classification'.",
        ),
    ],
    ids=["empty", "repeated", "nan", "three-for-binary", "one-for-binary"],
)
def test_expected_labels_the_task_cant_have_raise(labels, message):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            customers(),
            target_column="churned",
            task="binary_classification",
            expected_labels=labels,
        )

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == message


def test_a_multiclass_task_needs_at_least_two_expected_labels():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            customers(),
            target_column="plan",
            task="multiclass_classification",
            expected_labels=["basic"],
        )

    assert str(caught.value) == (
        "profile_target got 1 label in expected_labels for a "
        "multiclass_classification task, which has at least 2 classes. List every "
        "label the task allows."
    )


@pytest.mark.parametrize(
    ("values", "labels", "message"),
    [
        (
            [0, 1, 1],
            ["0", "1"],
            "profile_target can't use 2 labels in expected_labels: '0' and '1', "
            "since 'y' holds integers (dtype int64), so they can never equal a "
            "label. Did you mean 0 and 1? Declare each label as the type the "
            "target holds.",
        ),
        (
            pd.Series(["0", "1", "1"], dtype="string"),
            [0, "1"],
            "profile_target got 0 in expected_labels, but 'y' holds text (dtype "
            "string), so it can never equal a label. Did you mean '0'? Declare "
            "each label as the type the target holds.",
        ),
        (
            pd.Categorical(["low", "high", "low"]),
            ["low", 2],
            "profile_target got 2 in expected_labels, but 'y' holds categories of "
            "text (dtype category), so it can never equal a label. Did you mean "
            "'2'? Declare each label as the type the target holds.",
        ),
    ],
    ids=["text-for-integers", "number-for-text", "number-for-categories"],
)
def test_expected_labels_of_a_type_the_target_never_holds_raise(
    values, labels, message
):
    df = pd.DataFrame({"y": values})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            df,
            target_column="y",
            task="binary_classification",
            expected_labels=labels,
        )

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == message


def test_numbers_and_booleans_match_as_python_compares_them():
    df = pd.DataFrame({"churned": [True, False, True, True]})

    res = ae.profile_target(
        df,
        target_column="churned",
        task="binary_classification",
        expected_labels=[0, 1],
    )

    # True == 1 and False == 0, as in pandas' value_counts.
    assert res.unexpected_labels == []
    assert [(c.label, c.count) for c in filled(res.classes)] == [(True, 3), (False, 1)]


def test_row_ids_come_from_id_column_when_it_is_given():
    df = pd.DataFrame(
        {"order": [101, 102, 103, 104], "status": ["paid", "paid", "lost", "?"]}
    )

    res = ae.profile_target(
        df,
        target_column="status",
        task="multiclass_classification",
        expected_labels=["paid", "refunded"],
        id_column="order",
    )

    assert [(u.label, u.row_ids) for u in filled(res.unexpected_labels)] == [
        ("?", [104]),
        ("lost", [103]),
    ]
    assert res.parameters["id_column"] == "order"


def test_an_unknown_id_column_raises_with_the_closest_names():
    with pytest.raises(ColumnNotFoundError) as caught:
        ae.profile_target(
            customers(),
            target_column="churned",
            task="binary_classification",
            id_column="Customer",
        )

    assert str(caught.value) == (
        "profile_target got an unknown column 'Customer' in id_column. Did you mean "
        "'customer'? The data has 4 columns: 'customer', 'churned', 'plan' and "
        "'spend'."
    )


def test_repeated_row_ids_are_a_limitation_when_the_result_can_show_row_ids():
    df = pd.DataFrame({"status": ["paid", "?", "paid", "?"]}, index=[1, 1, 2, 3])

    checked = ae.profile_target(
        df,
        target_column="status",
        task="binary_classification",
        expected_labels=["paid", "refunded"],
    )
    unchecked = ae.profile_target(
        df, target_column="status", task="binary_classification"
    )

    [ambiguous] = [
        lim for lim in checked.limitations if lim.code == "ambiguous_row_ids"
    ]
    assert ambiguous.fields == ["unexpected_labels[0].row_ids"]
    assert ambiguous.reason == (
        "2 rows share an index label, so a row ID can match several rows: use a "
        "unique id_column. Or make the index unique, such as with "
        "df.reset_index(drop=True)."
    )
    # Without expected_labels, the result shows no row ID.
    assert "ambiguous_row_ids" not in [lim.code for lim in unchecked.limitations]


def test_the_text_shows_the_classes_and_their_balance():
    res = ae.profile_target(
        customers(), target_column="churned", task="binary_classification"
    )

    assert str(res).splitlines()[:7] == [
        "profile_target: completed on 10 rows x 4 columns",
        "Target 'churned' for binary_classification: 2 classes in 10 valid labels; "
        "0 missing.",
        "Majority share 0.7, minority share 0.3; imbalance ratio 2.333, the majority "
        "count over the minority count.",
        "Classes (2 of 2 shown):",
        "label  count  share",
        "'no'       7    0.7",
        "'yes'      3    0.3",
    ]
    assert str(res).splitlines()[-1] == (
        "More: res.to_frame() has all 2 labels; res.findings, res.limitations and "
        "res.to_json() have the rest."
    )


def test_the_text_shows_unexpected_labels_with_row_ids_and_marks_rare_classes():
    df = pd.DataFrame(
        {"churned": ["no", "yes", "No", "no", "unknown", "No", "no", None]},
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )

    res = ae.profile_target(
        df,
        target_column="churned",
        task="binary_classification",
        expected_labels=["no", "yes"],
        rare_share=0.3,
    )

    assert str(res).splitlines()[1:9] == [
        "Target 'churned' for binary_classification: 2 classes in 4 valid labels; 1 "
        "missing (12.5%).",
        "Majority share 0.75, minority share 0.25; imbalance ratio 3, the majority "
        "count over the minority count.",
        "Outside expected_labels: 2 labels in 3 rows: 'No' in rows 12 and 15; "
        "'unknown' in row 14.",
        "Rare classes, with a share below rare_share 0.3: 1 of 2.",
        "Classes (2 of 2 shown):",
        "label  count  share  rare",
        "'no'       3   0.75    no",
        "'yes'      1   0.25   yes",
    ]


def test_the_text_names_the_first_unexpected_labels_and_counts_the_rest():
    df = pd.DataFrame({"y": ["a", "b", *[f"x{i % 5}" for i in range(20)]]})

    res = ae.profile_target(
        df, target_column="y", task="binary_classification", expected_labels=["a", "b"]
    )

    lines = str(res).splitlines()
    assert lines[3].startswith("Outside expected_labels:")
    assert lines[4].startswith("    ")
    assert flowing("\n".join(lines[3:5])) == (
        "Outside expected_labels: 5 labels in 20 rows: 'x0' in rows 2, 7, 12 and 1 "
        "more; 'x1' in rows 3, 8, 13 and 1 more; 'x2' in rows 4, 9, 14 and 1 more; "
        "and 2 more labels."
    )


def test_the_text_shows_the_statistics_of_a_regression_target():
    price = [10.0, 12.5, None, INF, 7.0, 30.0, 12.5, -INF, 8.0, 11.0]

    res = ae.profile_target(
        pd.DataFrame({"price": price}), target_column="price", task="regression"
    )

    lines = str(res).splitlines()
    assert lines[1:7] == [
        "Target 'price' for regression: 7 finite values; 3 aren't: 1 missing (10%) "
        "and 2 infinite.",
        "Statistics use the finite values, as in summarize_numeric: quantile method "
        "linear; trimmed_mean cuts 10% from each end.",
        "res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, "
        "skewness and excess_kurtosis.",
        "Statistics (1 of 1 shown):",
        "column  count  missing_count  mean    std  min  median  max",
        # The finite values 7, 8, 10, 11, 12.5, 12.5 and 30: mean 91 / 7 = 13 and
        # std sqrt(363.5 / 6) = 7.784.
        "price       9              1    13  7.784    7      11   30",
    ]


def test_to_frame_gives_every_label_with_its_share_and_checks():
    df = pd.DataFrame({"churned": ["no", "yes", "No", "no", "unknown", "No", "no"]})

    res = ae.profile_target(
        df,
        target_column="churned",
        task="binary_classification",
        expected_labels=["no", "yes"],
        rare_share=0.3,
    )
    frame = res.to_frame()

    assert frame["label"].tolist() == ["no", "No", "unknown", "yes"]
    assert frame["label_type"].tolist() == ["str"] * 4
    assert frame["count"].tolist() == [3, 2, 1, 1]
    assert frame["share"].tolist()[0] == 0.75
    assert frame["share"].isna().tolist() == [False, True, True, False]
    assert frame["expected"].tolist() == [True, False, False, True]
    assert frame["rare"].tolist() == [False, pd.NA, pd.NA, True]


def test_to_frame_says_when_a_check_didnt_run():
    res = ae.profile_target(
        customers(), target_column="churned", task="binary_classification"
    )
    frame = res.to_frame()

    assert frame.columns.tolist() == [
        "label",
        "label_type",
        "count",
        "share",
        "expected",
        "rare",
    ]
    assert frame["expected"].isna().all()
    assert frame["rare"].isna().all()


def test_to_frame_gives_the_statistics_of_a_regression_target():
    price = [10.0, 12.5, None, INF, 7.0, 30.0]
    df = pd.DataFrame({"price": price})

    res = ae.profile_target(df, target_column="price", task="regression")
    frame = res.to_frame()
    numeric = ae.summarize_numeric(df, columns=["price"]).to_frame()

    pd.testing.assert_frame_equal(frame.drop(columns="non_finite_count"), numeric)
    assert frame["non_finite_count"].tolist() == [2]


def patients():
    """Six patients: the diagnosis and the email are sensitive."""
    return pd.DataFrame(
        {
            "email": [
                f"{name}@x.org" for name in ("ann", "bob", "cy", "dan", "eve", "fay")
            ],
            "diagnosis": ["flu", "cold", "flu", "covid", "flu", "?"],
        }
    )


def masking_project(*columns):
    return ae.Project(
        data_policy=ae.DataPolicy(sensitive_columns=list(columns), mask_sensitive=True)
    )


def test_masking_replaces_labels_and_row_ids_with_placeholders_but_keeps_counts():
    df = patients()

    res = ae.profile_target(
        df,
        target_column="diagnosis",
        task="multiclass_classification",
        expected_labels=["flu", "cold", "measles"],
        rare_share=0.3,
        id_column="email",
        project=masking_project("diagnosis", "email"),
    )

    # In level order: flu (3 rows), then '?', cold and covid (1 each) by code point.
    assert [(c.label, c.label_type, c.count, c.share) for c in filled(res.classes)] == [
        ("<masked 1>", "str", 3, 0.75),
        ("<masked 3>", "str", 1, 0.25),
    ]
    assert [(u.label, u.count, u.row_ids) for u in filled(res.unexpected_labels)] == [
        ("<masked 2>", 1, ["<masked 1>"]),
        ("<masked 4>", 1, ["<masked 2>"]),
    ]
    assert [c.label for c in filled(res.rare_classes)] == ["<masked 3>"]
    # measles never occurs, so its placeholder follows the 4 labels.
    assert res.parameters["expected_labels"] == [
        "<masked 1>",
        "<masked 3>",
        "<masked 5>",
    ]
    assert res.masking.applied is True
    assert res.masking.columns == ["diagnosis", "email"]
    frame = res.to_frame()
    assert frame["label"].tolist() == [f"<masked {n}>" for n in (1, 2, 3, 4)]
    assert frame["expected"].tolist() == [True, False, True, False]
    shown = str(res) + res.to_json() + frame.to_csv()
    raw = [*df["email"], "flu", "cold", "covid", "measles"]
    assert [value for value in raw if value in shown] == []


def test_a_target_that_is_also_id_column_shows_each_label_as_one_placeholder():
    df = pd.DataFrame({"code": ["a", "b", "a", "x"]})

    res = ae.profile_target(
        df,
        target_column="code",
        task="binary_classification",
        expected_labels=["a", "b"],
        id_column="code",
        project=masking_project("code"),
    )

    [unexpected] = filled(res.unexpected_labels)
    assert (unexpected.label, unexpected.row_ids) == ("<masked 3>", ["<masked 3>"])


def test_masking_leaves_a_regression_targets_statistics_and_says_so():
    df = pd.DataFrame({"salary": [52000.0, 61000.0, 48000.0]})

    res = ae.profile_target(
        df,
        target_column="salary",
        task="regression",
        project=masking_project("salary"),
    )

    assert res.statistics is not None
    assert res.statistics.max == 61000.0
    assert res.masking.applied is True
    assert res.masking.columns == []
    assert flowing(str(res)).startswith(
        "profile_target: completed on 3 rows x 1 column Masking on "
        "(data_policy.mask_sensitive=True), but numeric statistics are never "
        "masked: those of sensitive column 'salary', such as its minimum and "
        "maximum, show as computed."
    )


def flowing(text):
    """The text with its wrapped lines joined again."""
    return " ".join(line.strip() for line in text.splitlines())


TASKS = ["binary_classification", "multiclass_classification", "regression"]


def test_the_declared_statuses_are_completed_insufficient_data_and_failed():
    [info] = [i for i in ae.list_analyses().analyses if i.name == "profile_target"]

    assert info.statuses == ["completed", "insufficient_data", "failed"]


def too_few_values(res):
    """The fields the too_few_values limitation names, and its reason."""
    [limitation] = [lim for lim in res.limitations if lim.code == "too_few_values"]
    return limitation.fields, limitation.reason


@pytest.mark.parametrize("task", TASKS)
def test_data_without_rows_gives_insufficient_data(task):
    df = pd.DataFrame({"y": pd.Series([], dtype="float64")})

    res = ae.profile_target(df, target_column="y", task=task)

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_rows"
    assert res.missing_count == 0
    assert res.missing_share is None
    fields, reason = too_few_values(res)
    assert fields[0] == "missing_share"
    assert reason.startswith("The data has no rows, so missing_share")
    assert all(not line.endswith("...") for line in str(res).splitlines())
    assert res.findings == []


def test_a_classification_target_without_labels_gives_insufficient_data():
    df = pd.DataFrame({"y": pd.Series([None, None, None], dtype="string")})

    res = ae.profile_target(df, target_column="y", task="binary_classification")

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_labels",
        "Every target value is missing, so there are no labels to count.",
    )
    assert (res.classes, res.class_count, res.valid_label_count) == ([], 0, 0)
    assert (res.majority_share, res.minority_share, res.imbalance_ratio) == (
        None,
        None,
        None,
    )
    assert too_few_values(res) == (
        ["majority_share", "minority_share", "imbalance_ratio"],
        "No label is valid, so majority_share, minority_share and imbalance_ratio "
        "are null.",
    )
    assert [f.code for f in res.findings] == ["missing_labels"]


def test_a_classification_target_without_an_expected_label_gives_insufficient_data():
    df = pd.DataFrame({"y": ["a", "b", "a"]})

    res = ae.profile_target(
        df,
        target_column="y",
        task="binary_classification",
        expected_labels=["yes", "no"],
    )

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert (res.status_reason.code, res.status_reason.message) == (
        "no_valid_labels",
        "No target value is in expected_labels; unexpected_labels lists the labels "
        "found.",
    )
    assert res.unexpected_label_count == 2
    assert [f.code for f in res.findings] == ["unexpected_labels"]


def test_a_regression_target_without_a_finite_value_gives_insufficient_data():
    df = pd.DataFrame({"price": [None, INF, None]})

    res = ae.profile_target(df, target_column="price", task="regression")

    assert res.status == "insufficient_data"
    assert res.status_reason is not None
    assert res.status_reason.code == "no_finite_values"
    assert res.statistics is not None
    assert res.statistics.mean is None
    assert res.non_finite_count == 3
    fields, _ = too_few_values(res)
    assert "statistics.mean" in fields


@pytest.mark.parametrize("rare_share", [0, 1, 1.5, -0.1])
def test_rare_share_lies_between_0_and_1(rare_share):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.profile_target(
            customers(),
            target_column="churned",
            task="binary_classification",
            rare_share=rare_share,
        )

    assert str(caught.value) == (
        f"profile_target got {rare_share!r} for rare_share, but rare_share must be "
        "a number with 0 < rare_share < 1, or None. See help(ae.profile_target) for "
        "every argument."
    )


@given(df=data_frames(), data=st.data())
def test_class_counts_and_shares_match_value_counts_on_any_data(df, data):
    labels = label_columns(df)
    assume(labels)
    target = data.draw(st.sampled_from(labels), "target")
    column = column_named(df, target)

    res = ae.profile_target(df, target_column=target, task="multiclass_classification")

    frame = res.to_frame()
    counts = column.value_counts(dropna=True)
    assert sorted(frame["count"]) == sorted(count for count in counts if count)
    assert list(frame["count"]) == sorted(frame["count"], reverse=True)
    assert res.class_count == len(frame)
    assert res.valid_label_count == res.non_missing_count == column.count()
    assert res.missing_count == len(column) - column.count()
    kept = frame.head(100)
    assert [(c.count, c.share, c.label_type) for c in filled(res.classes)] == list(
        zip(kept["count"], kept["share"], kept["label_type"], strict=True)
    )
    if res.class_count:
        assert res.majority_share == frame["count"].max() / column.count()
        assert res.imbalance_ratio == frame["count"].max() / frame["count"].min()


@given(df=data_frames(), data=st.data())
def test_a_regression_target_gets_summarize_numerics_numbers_on_any_data(df, data):
    numbers = names(df, is_number)
    assume(numbers)
    target = data.draw(st.sampled_from(numbers), "target")

    res = ae.profile_target(df, target_column=target, task="regression")
    numeric = ae.summarize_numeric(df, columns=[target]).columns[target]

    assert res.statistics is not None
    assert res.statistics.model_dump() == numeric.model_dump()
    assert res.non_finite_count == numeric.missing_count + numeric.infinite_count
    assert res.missing_count == numeric.missing_count
