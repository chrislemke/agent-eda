import doctest
import inspect

import numpy as np
import pandas as pd
import pydantic
import pytest

import agent_eda as ae
from agent_eda.errors import AgentEDAError, InvalidArgumentError

NOTHING_TO_MASK = (
    "DataPolicy got mask_sensitive=True but no sensitive_columns, so there would "
    "be nothing to mask. Name the sensitive columns, such as "
    "ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True)."
)


def test_a_project_built_in_code_holds_its_name_and_data_policy():
    policy = ae.DataPolicy(sensitive_columns=["email", "name"], mask_sensitive=True)

    project = ae.Project(name="churn", data_policy=policy)

    assert project.name == "churn"
    assert project.data_policy.sensitive_columns == ("email", "name")
    assert project.data_policy.mask_sensitive is True
    assert isinstance(project, pydantic.BaseModel)
    assert isinstance(policy, pydantic.BaseModel)


def test_a_project_needs_no_arguments_and_declares_nothing_sensitive():
    project = ae.Project()

    assert project.name is None
    assert project.data_policy == ae.DataPolicy()
    assert project.data_policy.sensitive_columns == ()
    assert project.data_policy.mask_sensitive is False


def test_masking_stays_off_when_sensitive_columns_are_declared():
    policy = ae.DataPolicy(sensitive_columns=["email"])

    assert policy.mask_sensitive is False


def test_a_project_has_only_a_name_and_a_data_policy():
    assert list(ae.Project.model_fields) == ["name", "data_policy"]
    assert list(ae.DataPolicy.model_fields) == ["sensitive_columns", "mask_sensitive"]


@pytest.mark.parametrize("model", ["Project", "DataPolicy"])
def test_the_docs_call_masking_a_guardrail_not_anonymization_or_access_control(model):
    doc = inspect.getdoc(getattr(ae, model))
    assert doc
    text = " ".join(doc.split())

    assert "Masking is a guardrail against accidental exposure" in text
    assert "not anonymization" in text
    assert "not access control" in text


@pytest.mark.parametrize("model", ["Project", "DataPolicy"])
def test_the_docstring_examples_run(model):
    runner = doctest.DocTestRunner()
    report = []
    finder = doctest.DocTestFinder(recurse=False)
    for test in finder.find(getattr(ae, model), name=model, globs={}):
        runner.run(test, out=report.append)

    assert runner.tries > 0
    assert runner.failures == 0, "".join(report)


def test_help_shows_the_fields_as_keyword_arguments():
    assert str(inspect.signature(ae.Project)) == (
        "(*, name: str | None = None, data_policy: agent_eda.project.DataPolicy | "
        "collections.abc.Mapping[str, typing.Any] | None = None)"
    )
    assert str(inspect.signature(ae.DataPolicy)) == (
        "(*, sensitive_columns: collections.abc.Sequence[str] | None = (), "
        "mask_sensitive: bool = False)"
    )


@pytest.mark.parametrize("sensitive_columns", [None, [], ()])
def test_masking_without_sensitive_columns_raises(sensitive_columns):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.DataPolicy(sensitive_columns=sensitive_columns, mask_sensitive=True)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == NOTHING_TO_MASK


def test_masking_without_sensitive_columns_raises_from_a_policy_given_as_a_dict():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project(data_policy={"mask_sensitive": True})

    assert str(caught.value) == NOTHING_TO_MASK


def test_a_data_policy_can_be_given_as_a_dict_of_its_fields():
    project = ae.Project(
        data_policy={"sensitive_columns": ["email"], "mask_sensitive": True}
    )

    assert project.data_policy == ae.DataPolicy(
        sensitive_columns=["email"], mask_sensitive=True
    )


def test_none_for_the_data_policy_means_the_default_one():
    assert ae.Project(data_policy=None).data_policy == ae.DataPolicy()


@pytest.mark.parametrize(
    ("given", "names"),
    [
        ("email", ("email",)),
        (["email", 0, 1.5], ("email", "0", "1.5")),
        (("b", "a"), ("b", "a")),
        ({"b", "a"}, ("a", "b")),
        (pd.Index(["a", "b"]), ("a", "b")),
        (np.array(["a", "b"]), ("a", "b")),
        ((name for name in ["a"]), ("a",)),
        (np.int64(3), ("3",)),
    ],
    ids=["one-name", "labels", "tuple", "set", "index", "array", "generator", "numpy"],
)
def test_sensitive_columns_take_names_as_the_analyses_columns_argument_does(
    given, names
):
    policy = ae.DataPolicy(sensitive_columns=given)

    assert policy.sensitive_columns == names


def test_a_sensitive_column_listed_twice_raises():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.DataPolicy(sensitive_columns=["email", "name", "email"])

    assert str(caught.value) == (
        "DataPolicy got 'email' more than once in sensitive_columns. List each "
        "column once."
    )


def test_a_mapping_for_sensitive_columns_raises_with_the_form_to_use():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.DataPolicy(sensitive_columns={"email": True})  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert str(caught.value) == (
        "DataPolicy got a dict for sensitive_columns, but "
        "sensitive_columns must be a list of column names, such as "
        "ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True)."
    )


def test_a_project_and_its_data_policy_are_frozen_and_hashable():
    project = ae.Project(name="churn", data_policy={"sensitive_columns": ["email"]})

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        project.name = "other"  # ty: ignore[invalid-assignment]  # the change is the test
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        project.data_policy.mask_sensitive = True  # ty: ignore[invalid-assignment]  # the change is the test
    assert hash(project) == hash(
        ae.Project(name="churn", data_policy={"sensitive_columns": ("email",)})
    )


def test_model_validate_builds_a_project_from_a_dict_of_its_fields():
    project = ae.Project.model_validate(
        {"name": "churn", "data_policy": {"sensitive_columns": ["email"]}}
    )

    assert project == ae.Project(
        name="churn", data_policy=ae.DataPolicy(sensitive_columns=["email"])
    )


@pytest.mark.parametrize(
    ("given", "message"),
    [
        (
            {"name": "churn", "polcy": {}},
            "Project has no field 'polcy'. Did you mean 'data_policy'?",
        ),
        (
            ["churn"],
            "Project.model_validate takes a dict of the fields, but got a list. "
            "ae.Project(name='churn', data_policy=ae.DataPolicy("
            "sensitive_columns=['email'])) builds one in code.",
        ),
        ({"data_policy": {"mask_sensitive": True}}, NOTHING_TO_MASK),
    ],
    ids=["unknown-field", "not-a-dict", "nothing-to-mask"],
)
def test_model_validate_raises_in_the_package_format(given, message):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project.model_validate(given)

    assert str(caught.value).startswith(message)


def test_a_project_saved_as_json_loads_again():
    project = ae.Project(
        name="churn",
        data_policy={"sensitive_columns": ["email"], "mask_sensitive": True},
    )

    assert ae.Project.model_validate_json(project.model_dump_json()) == project


@pytest.mark.parametrize(
    ("load", "code", "message"),
    [
        (
            lambda: ae.Project.model_validate_json('{"nme": "churn"}'),
            "invalid_call",
            "Project has no field 'nme'. Did you mean 'name'? Allowed: name, "
            "data_policy.",
        ),
        (
            lambda: ae.DataPolicy.model_validate_json('{"mask_sensitive": true}'),
            "invalid_argument",
            NOTHING_TO_MASK,
        ),
        (
            lambda: ae.Project.model_validate_json('{"name": "churn"'),
            "invalid_argument",
            "Project.model_validate_json got text that isn't valid JSON",
        ),
        (
            lambda: ae.Project.model_validate_strings({"name": 1}),
            "invalid_argument",
            "Project got 1 for name, but name must be str | None.",
        ),
        (
            lambda: ae.Project.model_validate({1: "churn"}),
            "invalid_call",
            "Project got the key 1, but its fields are named by strings: name, "
            "data_policy.",
        ),
        (
            lambda: ae.Project(data_policy={1: "email"}),  # ty: ignore[invalid-argument-type]  # the wrong key type is the test
            "invalid_call",
            "DataPolicy got the key 1, but its fields are named by strings: "
            "sensitive_columns, mask_sensitive.",
        ),
    ],
    ids=[
        "json-unknown-field",
        "json-nothing-to-mask",
        "json-syntax",
        "strings",
        "key-not-a-string",
        "nested-key-not-a-string",
    ],
)
def test_every_way_to_build_one_raises_in_the_package_format(load, code, message):
    with pytest.raises(InvalidArgumentError) as caught:
        load()

    assert caught.value.code == code
    assert str(caught.value).startswith(message)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["ann@x.org", "bob@x.org"], name="email"),
        pd.DataFrame({"email": ["ann@x.org"]}),
    ],
    ids=["series", "dataframe"],
)
def test_data_given_as_sensitive_columns_raises_without_showing_its_values(values):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.DataPolicy(sensitive_columns=values)

    assert "@x.org" not in str(caught.value)
    assert "sensitive_columns takes column names, not data" in str(caught.value)


def test_bytes_name_one_column_as_a_string_does():
    policy = ae.DataPolicy(sensitive_columns=b"key")  # ty: ignore[invalid-argument-type]  # a bytes label is the test

    assert policy.sensitive_columns == ("b'key'",)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: ae.Project(nmae="churn"),  # ty: ignore[unknown-argument]  # the typo is the test
            "Project has no field 'nmae'. Did you mean 'name'? "
            "Allowed: name, data_policy.",
        ),
        (
            lambda: ae.Project(roles={"train": "train.csv"}),  # ty: ignore[unknown-argument]  # the unknown field is the test
            "Project has no field 'roles'. Allowed: name, data_policy.",
        ),
        (
            lambda: ae.DataPolicy(sensitive_column=["email"]),  # ty: ignore[unknown-argument]  # the typo is the test
            "DataPolicy has no field 'sensitive_column'. Did you mean "
            "'sensitive_columns'? Allowed: sensitive_columns, mask_sensitive.",
        ),
    ],
    ids=["misspelled", "dataset-roles", "data-policy"],
)
def test_an_unknown_field_raises_with_the_closest_name_and_the_allowed_ones(
    build, message
):
    with pytest.raises(InvalidArgumentError) as caught:
        build()

    error = caught.value
    assert isinstance(error, AgentEDAError)
    assert isinstance(error, ValueError)
    assert error.code == "invalid_call"
    assert str(error) == message
    assert not isinstance(error, pydantic.ValidationError)
    assert error.__cause__ is None
    assert error.__suppress_context__


@pytest.mark.parametrize("field", ["sensitive_columns", "mask_sensitive"])
def test_a_data_policy_field_given_to_the_project_shows_where_it_belongs(field):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project(**{field: ["email"]})  # ty: ignore[invalid-argument-type]  # the misplaced field is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        f"Project has no field {field!r}. It is a field of the data policy: "
        f"ae.Project(data_policy=ae.DataPolicy({field}=...)). "
        "Allowed: name, data_policy."
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: ae.Project(name=3),  # ty: ignore[invalid-argument-type]  # the wrong type is the test
            "Project got 3 for name, but name must be str | None. Input should be "
            "a valid string. See help(ae.Project) for every field.",
        ),
        (
            lambda: ae.Project(data_policy=["email"]),  # ty: ignore[invalid-argument-type]  # the wrong type is the test
            "Project got a list of 1 item for data_policy, but data_policy must be "
            "DataPolicy | Mapping[str, Any] | None. See help(ae.Project) for every "
            "field.",
        ),
        (
            lambda: ae.DataPolicy(mask_sensitive="yes"),  # ty: ignore[invalid-argument-type]  # the wrong type is the test
            "DataPolicy got 'yes' for mask_sensitive, but mask_sensitive must be "
            "bool. Input should be a valid boolean. See help(ae.DataPolicy) for "
            "every field.",
        ),
        (
            lambda: ae.DataPolicy(sensitive_columns=["email", ["name"]]),  # ty: ignore[invalid-argument-type]  # the wrong type is the test
            "DataPolicy got a list of 1 item at sensitive_columns[1], but each "
            "value in sensitive_columns must be str. Input should be a valid "
            "string. See help(ae.DataPolicy) for every field.",
        ),
    ],
    ids=["name", "data-policy", "mask-sensitive", "sensitive-column"],
)
def test_a_field_of_the_wrong_type_raises_with_what_it_allows(build, message):
    with pytest.raises(InvalidArgumentError) as caught:
        build()

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: ae.Project("churn"),  # ty: ignore[too-many-positional-arguments]  # the call is the test
            "Project takes its fields by keyword, such as ae.Project(name='churn', "
            "data_policy=ae.DataPolicy(sensitive_columns=['email'])).",
        ),
        (
            lambda: ae.DataPolicy(["email"]),  # ty: ignore[too-many-positional-arguments]  # the call is the test
            "DataPolicy takes its fields by keyword, such as "
            "ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True).",
        ),
    ],
    ids=["project", "data-policy"],
)
def test_fields_passed_by_position_raise_with_the_keyword_form(build, message):
    with pytest.raises(InvalidArgumentError) as caught:
        build()

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == message
