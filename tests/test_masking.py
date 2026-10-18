import pandas as pd
import pytest
from recipes import recipe_for

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

# Every analysis whose results can show raw values, and so supports masking.
MASKING = [info.name for info in ae.list_analyses().analyses if info.supports_masking]


CUSTOMERS = ["Ann Lee", "Bob Ray", "Ann Lee", "Cy Fox", "Ann Lee", "Bob Ray"]


def tickets():
    """Six support tickets; customer holds names, which are sensitive."""
    return pd.DataFrame(
        {
            "ticket": [101, 102, 103, 104, 105, 106],
            "customer": CUSTOMERS,
            "plan": ["basic", "pro", "basic", "basic", "team", "pro"],
        }
    )


def orders():
    """Six orders; mug for 9.5 repeats three times and pen for 2.0 twice."""
    return pd.DataFrame(
        {
            "email": [
                "ann@x.org",
                "bob@x.org",
                "ann@x.org",
                "cy@x.org",
                "bob@x.org",
                "dan@x.org",
            ],
            "item": ["mug", "pen", "mug", "cap", "pen", "mug"],
            "amount": [9.5, 2.0, 9.5, 5.0, 2.0, 9.5],
        }
    )


def masking_project(*columns, mask_sensitive=True):
    return ae.Project(
        data_policy=ae.DataPolicy(
            sensitive_columns=list(columns), mask_sensitive=mask_sensitive
        )
    )


def call(name, df, **arguments):
    """Call an analysis with the arguments its recipe requires, and these."""
    required = recipe_for(name).required(df)
    assert required is not None
    return getattr(ae, name)(df, **required, **arguments)


def test_both_analyses_that_show_raw_values_support_masking():
    assert {"summarize_categorical", "find_duplicate_rows"} <= set(MASKING)


@pytest.mark.parametrize("name", ["summarize_categorical", "find_duplicate_rows"])
def test_discovery_says_they_can_show_raw_values_and_support_masking(name):
    [info] = [i for i in ae.list_analyses().analyses if i.name == name]
    description = ae.describe_analysis(name)

    assert info.supports_masking is True
    assert description.supports_masking is True
    assert description.shows_raw_values is True
    assert (
        "Masking: supported (project=, mask_sensitive=). Its results can show raw "
        "values, such as labels or row IDs, and masking replaces those of sensitive "
        "columns with numbered placeholders."
    ) in flowing(str(description))


def test_discovery_says_the_other_analyses_show_no_raw_values():
    description = ae.describe_analysis("summarize_numeric")

    assert description.supports_masking is False
    assert description.shows_raw_values is False
    assert (
        "Masking: not needed. Its results show no raw values such as labels, "
        "example values or row IDs; numeric statistics, such as a minimum, are "
        "never masked."
    ) in flowing(str(description))


def test_searching_for_masking_finds_every_analysis_that_supports_it():
    found = ae.search_analyses("mask sensitive columns", limit=len(MASKING))

    assert sorted(match.name for match in found.matches) == sorted(MASKING)


NO_SENSITIVE_COLUMNS = ae.DataPolicy()
SENSITIVE = ae.DataPolicy(sensitive_columns=["customer"])
MASKED = ae.DataPolicy(sensitive_columns=["customer"], mask_sensitive=True)


@pytest.mark.parametrize("name", MASKING)
@pytest.mark.parametrize(
    ("policy", "mask_sensitive", "applied", "decided_by"),
    [
        (None, None, False, "default"),
        (None, False, False, "mask_sensitive"),
        (NO_SENSITIVE_COLUMNS, None, False, "project"),
        (NO_SENSITIVE_COLUMNS, False, False, "mask_sensitive"),
        (SENSITIVE, None, False, "project"),
        (SENSITIVE, True, True, "mask_sensitive"),
        (SENSITIVE, False, False, "mask_sensitive"),
        (MASKED, None, True, "project"),
        (MASKED, True, True, "mask_sensitive"),
        (MASKED, False, False, "mask_sensitive"),
    ],
    ids=[
        "no-project",
        "no-project-call-off",
        "nothing-sensitive",
        "nothing-sensitive-call-off",
        "project-off",
        "project-off-call-on",
        "project-off-call-off",
        "project-on",
        "project-on-call-on",
        "project-on-call-off",
    ],
)
def test_the_call_decides_masking_then_the_project_and_without_one_nothing_is_masked(
    name, policy, mask_sensitive, applied, decided_by
):
    project = None if policy is None else ae.Project(data_policy=policy)

    res = call(name, tickets(), project=project, mask_sensitive=mask_sensitive)

    assert res.masking.applied is applied
    assert res.masking.decided_by == decided_by


@pytest.mark.parametrize("name", MASKING)
def test_a_project_can_be_given_as_a_dict_of_its_fields(name):
    project = {"data_policy": {"sensitive_columns": ["customer"]}}

    res = call(name, tickets(), project=project, mask_sensitive=True)

    assert res.masking.applied is True
    assert res.parameters["project"] == {
        "name": None,
        "data_policy": {"sensitive_columns": ["customer"], "mask_sensitive": False},
    }


@pytest.mark.parametrize("name", MASKING)
@pytest.mark.parametrize(
    ("project", "code", "message"),
    [
        (
            {"nmae": "churn"},
            "invalid_call",
            "Project has no field 'nmae'. Did you mean 'name'? Allowed: name, "
            "data_policy.",
        ),
        (
            {"data_policy": {"mask_sensitive": True}},
            "invalid_argument",
            "DataPolicy got mask_sensitive=True but no sensitive_columns, so there "
            "would be nothing to mask. Name the sensitive columns, such as "
            "ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True).",
        ),
        (
            ae.DataPolicy(sensitive_columns=["customer"]),
            "invalid_argument",
            "project takes a Project, but got a DataPolicy. Put the policy in a "
            "project: project=ae.Project(data_policy=policy).",
        ),
        (
            {1: "churn"},
            "invalid_call",
            "Project got the key 1, but its fields are named by strings: name, "
            "data_policy.",
        ),
    ],
    ids=["unknown-field", "nothing-to-mask", "data-policy", "key-not-a-string"],
)
def test_an_invalid_project_raises_the_projects_own_error(name, project, code, message):
    with pytest.raises(InvalidArgumentError) as caught:
        call(name, tickets(), project=project)

    assert caught.value.code == code
    assert str(caught.value) == message


@pytest.mark.parametrize("name", MASKING)
def test_masking_in_the_call_without_a_project_raises(name):
    with pytest.raises(InvalidArgumentError) as caught:
        call(name, tickets(), mask_sensitive=True)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"{name} got mask_sensitive=True without a project, so no column is "
        "declared sensitive and there would be nothing to mask. Pass a project "
        "that names the sensitive columns, such as project=ae.Project("
        "data_policy=ae.DataPolicy(sensitive_columns=['email']))."
    )


@pytest.mark.parametrize("name", MASKING)
def test_masking_in_the_call_without_declared_sensitive_columns_raises(name):
    with pytest.raises(InvalidArgumentError) as caught:
        call(name, tickets(), project=ae.Project(name="churn"), mask_sensitive=True)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"{name} got mask_sensitive=True, but the project declares no sensitive "
        "columns, so there would be nothing to mask. Name them in its data policy, "
        "such as project=ae.Project(data_policy=ae.DataPolicy("
        "sensitive_columns=['email']))."
    )


def test_masking_stays_off_by_default_even_with_sensitive_columns_declared():
    project = ae.Project(data_policy=ae.DataPolicy(sensitive_columns=["customer"]))

    res = ae.summarize_categorical(tickets(), project=project)

    assert [level.label for level in res.columns["customer"].levels] == [
        "Ann Lee",
        "Bob Ray",
        "Cy Fox",
    ]
    assert res.masking.applied is False
    assert res.masking.columns == ["customer"]


def test_masking_covers_the_levels_in_the_text_the_record_and_to_frame():
    res = ae.summarize_categorical(tickets(), project=masking_project("customer"))

    text = str(res)
    record = res.to_json()
    frame = res.to_frame()

    assert [name for name in set(CUSTOMERS) if name in text or name in record] == []
    lines = text.splitlines()
    assert lines[1:3] == [
        "Masked (data_policy.mask_sensitive=True): the values of column 'customer' "
        "show as <masked 1>, <masked 2> and so on;",
        "    counts and shares are unaffected.",
    ]
    assert (
        "customer                  6              0            3  <masked 1>: 3 (50%), "
        "<masked 2>: 2 (33.33%), <masked 3>: 1 (..."
    ) in lines
    customer = frame[frame["column"] == "customer"]
    assert customer["label"].tolist() == ["<masked 1>", "<masked 2>", "<masked 3>"]
    assert customer["label_type"].tolist() == ["str", "str", "str"]
    assert customer["count"].tolist() == [3, 2, 1]
    assert res.to_dict()["masking"] == {
        "applied": True,
        "decided_by": "project",
        "columns": ["customer"],
        "columns_total": 1,
        "index_columns": [],
        "index_columns_total": 0,
        "absent_columns": [],
        "absent_columns_total": 0,
    }


def test_masked_row_ids_of_numbers_or_dates_become_placeholders_missing_ones_null():
    df = pd.DataFrame(
        {
            "customer_number": pd.array([70001, 70002, None, 70004], dtype="Int64"),
            "signed_up": pd.to_datetime(
                ["2021-03-04", "2021-03-04", "2022-05-06", None]
            ),
            "plan": ["pro", "pro", "pro", "pro"],
        }
    )
    project = masking_project("customer_number", "signed_up")

    by_number = ae.find_duplicate_rows(
        df, columns=["plan"], id_column="customer_number", project=project
    )
    by_date = ae.find_duplicate_rows(
        df, columns=["plan"], id_column="signed_up", project=project
    )

    assert by_number.duplicate_groups[0].row_ids == [
        "<masked 1>",
        "<masked 2>",
        None,
        "<masked 3>",
    ]
    assert by_date.duplicate_groups[0].row_ids == [
        "<masked 1>",
        "<masked 1>",
        "<masked 2>",
        None,
    ]
    assert "7000" not in str(by_number) + by_number.to_json()
    assert "2021" not in str(by_date) + by_date.to_json()


def test_masking_covers_the_row_ids_in_the_text_the_record_and_to_frame():
    res = ae.find_duplicate_rows(
        orders(), id_column="email", project=masking_project("email")
    )

    text = str(res)
    record = res.to_json()
    frame = res.to_frame()

    emails = set(orders()["email"])
    assert [email for email in emails if email in text or email in record] == []
    lines = text.splitlines()
    assert "    0     3  <masked 1>, <masked 1> and <masked 2>" in lines
    assert "    1     2  <masked 3> and <masked 3>" in lines
    assert frame["row_id"].tolist() == [
        "<masked 1>",
        "<masked 1>",
        "<masked 2>",
        "<masked 3>",
        "<masked 3>",
    ]
    assert res.to_dict()["duplicate_groups"][1]["row_ids"] == [
        "<masked 3>",
        "<masked 3>",
    ]
    assert res.masking.columns == ["email"]


def test_index_labels_are_never_masked():
    df = orders().set_index("item")

    res = ae.find_duplicate_rows(df, project=masking_project("email", "amount"))

    assert res.duplicate_groups[0].row_ids == ["mug", "mug"]
    assert res.masking.applied is True
    assert res.masking.columns == []
    assert res.masking.index_columns == []


def test_a_sensitive_column_moved_into_the_index_shows_unmasked_and_says_so():
    df = orders().set_index("email")

    res = ae.find_duplicate_rows(df, project=masking_project("email"))

    assert res.duplicate_groups[0].row_ids == ["ann@x.org", "ann@x.org", "dan@x.org"]
    assert res.masking.index_columns == ["email"]
    assert res.masking.absent_columns == []
    assert str(res).splitlines()[1:3] == [
        "Masking on (data_policy.mask_sensitive=True), but index labels are never "
        "masked, and the index holds sensitive column",
        "    'email', whose values show as row IDs. Pass a non-sensitive id_column, "
        "such as after df.reset_index().",
    ]


@pytest.mark.parametrize("name", MASKING)
def test_the_docs_call_masking_a_guardrail_not_anonymization_or_access_control(name):
    described = {p.name: p for p in ae.describe_analysis(name).parameters}

    assert (
        "Masking is a guardrail against accidental exposure, not anonymization or "
        "access control."
    ) in described["mask_sensitive"].description


@pytest.mark.parametrize("name", MASKING)
def test_the_docs_say_index_labels_are_never_masked(name):
    docs = ae.describe_analysis(name)
    mask_sensitive = next(p for p in docs.parameters if p.name == "mask_sensitive")

    if "id_column" in [p.name for p in docs.parameters]:
        assert "Index labels are never masked" in mask_sensitive.description
        assert "pass a non-sensitive id_column" in docs.when_to_use


def flowing(text):
    """The text with its wrapped lines joined again."""
    return " ".join(line.strip() for line in text.splitlines())


def test_sensitive_columns_the_data_lacks_are_named_as_a_typo_would_be():
    res = ae.summarize_categorical(
        tickets(), project=masking_project("customer", "emial")
    )

    assert res.masking.absent_columns == ["emial"]
    assert "Sensitive columns the data lacks: 'emial'." in flowing(str(res))


def test_the_text_says_when_sensitive_values_show_as_stored():
    res = ae.summarize_categorical(tickets(), project=ae.Project(data_policy=SENSITIVE))

    assert (
        "Not masked (data_policy.mask_sensitive=False): the values of sensitive "
        "column 'customer' show as stored. Pass mask_sensitive=True to mask them."
    ) in flowing(str(res))


def test_without_a_project_the_text_says_nothing_about_masking():
    res = ae.summarize_categorical(tickets())

    assert "mask" not in str(res)


def test_values_of_sensitive_columns_the_result_doesnt_show_need_no_masking():
    res = ae.find_duplicate_rows(orders(), project=masking_project("email", "item"))

    assert res.masking.applied is True
    assert res.masking.columns == []
    assert str(res).splitlines()[1] == (
        "Masking on (data_policy.mask_sensitive=True), but the result shows no value "
        "of a sensitive column."
    )


def test_masked_levels_become_numbered_placeholders_in_level_order():
    res = ae.summarize_categorical(tickets(), project=masking_project("customer"))

    levels = res.columns["customer"].levels
    assert [(level.label, level.label_type, level.count) for level in levels] == [
        ("<masked 1>", "str", 3),
        ("<masked 2>", "str", 2),
        ("<masked 3>", "str", 1),
    ]
    assert [level.share for level in levels] == pytest.approx([1 / 2, 1 / 3, 1 / 6])
    assert [level.label for level in res.columns["plan"].levels] == [
        "basic",
        "pro",
        "team",
    ]


def test_every_label_of_a_masked_column_is_a_placeholder_unused_categories_too():
    names = pd.Categorical(
        ["Ann", "ann ", "Bob", "Ann"], categories=["Ann", "ann ", "Bob", "Cy"]
    )

    res = ae.summarize_categorical(
        pd.DataFrame({"name": names}), project=masking_project("name"), top_n=2
    )

    entry = res.columns["name"]
    # Ann occurs twice; Bob and 'ann ' once each, and 'B' sorts before 'a'.
    assert [level.label for level in entry.levels] == [
        "<masked 1>",
        "<masked 2>",
        "<masked 3>",
    ]
    assert [level.label for level in entry.most_common_levels] == [
        "<masked 1>",
        "<masked 2>",
    ]
    assert entry.unused_categories == ["<masked 4>"]
    assert [
        [level.label for level in group.levels]
        for group in entry.case_or_space_variants
    ] == [["<masked 1>", "<masked 3>"]]
    assert [finding.code for finding in res.findings] == [
        "case_or_space_variants",
        "unused_categories",
    ]


def test_a_masked_numeric_column_keeps_its_label_types_counts_and_shares():
    df = pd.DataFrame({"zip": [10115, 10115, 80331]})

    res = ae.summarize_categorical(df, columns=["zip"], project=masking_project("zip"))

    levels = res.columns["zip"].levels
    assert [(level.label, level.label_type, level.count) for level in levels] == [
        ("<masked 1>", "int", 2),
        ("<masked 2>", "int", 1),
    ]
    assert res.to_frame()["label"].tolist() == ["<masked 1>", "<masked 2>"]
    assert res.to_frame()["label_type"].tolist() == ["int", "int"]
    assert "<masked 1>: 2 (66.67%), <masked 2>: 1 (33.33%)" in str(res)
    assert "10115" not in str(res) + res.to_json()


def test_masked_row_ids_become_numbered_placeholders_that_equal_ids_share():
    res = ae.find_duplicate_rows(
        orders(), id_column="email", project=masking_project("email")
    )

    assert [group.row_ids for group in res.duplicate_groups] == [
        ["<masked 1>", "<masked 1>", "<masked 2>"],
        ["<masked 3>", "<masked 3>"],
    ]
    assert [group.size for group in res.duplicate_groups] == [3, 2]
    assert res.excess_copy_count == 3
