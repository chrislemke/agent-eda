"""The declaration models: Project, DataPolicy and ColumnSpec build alike.

Every way of building one raises the package's InvalidArgumentError for a
mistake, and none lets an invalid value through.
"""

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

# The fields a model needs besides the one under test.
REQUIRED: dict[str, dict[str, Any]] = {"ColumnSpec": {"logical_type": "integer"}}

# For each model: a valid one, and fields with one bad value that is text, so
# that every route, the string route included, reads it the same way.
MODELS: dict[str, tuple[Callable[[], Any], dict[str, Any]]] = {
    "Project": (ae.Project, {"data_policy": "all columns"}),
    "DataPolicy": (ae.DataPolicy, {"mask_sensitive": "maybe"}),
    "ColumnSpec": (lambda: ae.ColumnSpec("integer"), {"logical_type": "int"}),
}


def model_class(name: str) -> Any:
    return getattr(ae, name)


def build_by_call(name: str, fields: dict[str, Any]) -> Any:
    return model_class(name)(**fields)


def build_by_validate(name: str, fields: dict[str, Any]) -> Any:
    return model_class(name).model_validate(fields)


def build_by_json(name: str, fields: dict[str, Any]) -> Any:
    return model_class(name).model_validate_json(json.dumps(fields))


def build_by_strings(name: str, fields: dict[str, Any]) -> Any:
    return model_class(name).model_validate_strings(fields)


def build_by_copy(name: str, fields: dict[str, Any]) -> Any:
    valid, _ = MODELS[name]
    return valid().model_copy(update=fields)


ROUTES = [
    build_by_call,
    build_by_validate,
    build_by_json,
    build_by_strings,
    build_by_copy,
]


def raised(build: Callable[..., Any], name: str, fields: dict[Any, Any]) -> Any:
    """The error a route raises for the fields, which is the package's own."""
    with pytest.raises(InvalidArgumentError) as caught:
        build(name, fields)
    error = caught.value
    assert not isinstance(error, pydantic.ValidationError)
    assert error.__cause__ is None
    assert error.code in {"invalid_argument", "invalid_call"}
    return error


@pytest.mark.parametrize("build", ROUTES, ids=lambda build: build.__name__)
@pytest.mark.parametrize("name", MODELS)
def test_a_bad_value_raises_the_package_error_by_every_route(name, build):
    _, bad = MODELS[name]

    error = raised(build, name, bad)

    assert str(error).startswith(f"{name} got ")


@pytest.mark.parametrize("build", ROUTES, ids=lambda build: build.__name__)
@pytest.mark.parametrize("name", MODELS)
def test_an_unknown_field_raises_the_package_error_by_every_route(name, build):
    error = raised(build, name, {**REQUIRED.get(name, {}), "nulable": "x"})

    assert error.code == "invalid_call"
    assert str(error).startswith(f"{name} has no field 'nulable'")


@pytest.mark.parametrize("build", [build_by_validate, build_by_copy])
@pytest.mark.parametrize("name", MODELS)
def test_a_key_that_is_not_a_string_raises_the_package_error(name, build):
    fields: dict[Any, Any] = {1: "integer"}
    error = raised(build, name, fields)

    assert error.code == "invalid_call"
    assert str(error).startswith(f"{name} got the key 1")


@pytest.mark.parametrize("name", MODELS)
def test_a_copy_with_a_valid_update_is_a_validated_declaration(name):
    valid, _ = MODELS[name]
    updates = {
        "Project": {"name": "churn"},
        "DataPolicy": {"sensitive_columns": ["a"], "mask_sensitive": True},
        "ColumnSpec": {"logical_type": "float", "nullable": False},
    }

    copied = valid().model_copy(update=updates[name])

    assert type(copied) is type(valid())
    assert copied == type(valid()).model_validate(
        {**valid().model_dump(), **updates[name]}
    )


def test_a_copy_of_a_project_with_a_data_policy_dict_holds_a_data_policy():
    project = ae.Project().model_copy(
        update={"data_policy": {"sensitive_columns": ["email"], "mask_sensitive": True}}
    )

    assert project.data_policy == ae.DataPolicy(
        sensitive_columns=["email"], mask_sensitive=True
    )
    df = pd.DataFrame({"email": ["a@b.c", "d@e.f", "a@b.c"]})
    res = ae.summarize_categorical(df, project=project)
    assert res.masking.columns == ["email"]


def test_a_copy_that_would_mask_nothing_raises_like_the_call_does():
    with pytest.raises(InvalidArgumentError, match="nothing to mask"):
        ae.DataPolicy().model_copy(update={"mask_sensitive": True})


def test_a_copy_without_an_update_is_equal():
    project = ae.Project(
        name="churn",
        data_policy=ae.DataPolicy(sensitive_columns=["email"], mask_sensitive=True),
    )

    assert project.model_copy() == project


def test_the_column_spec_routes_in_the_audit_raise_with_the_allowed_values():
    for build in (
        lambda: ae.ColumnSpec.model_validate_json('{"logical_type": "int"}'),
        lambda: ae.ColumnSpec.model_validate_strings({"logical_type": "int"}),
    ):
        with pytest.raises(InvalidArgumentError) as caught:
            build()
        assert "must be one of 'boolean', 'integer'" in str(caught.value)
        assert "Did you mean 'integer'?" in str(caught.value)


def test_a_schema_with_a_key_that_is_not_a_string_raises_the_package_error():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(InvalidArgumentError) as caught:
        ae.validate_column_types(df, schema={"a": {1: "integer"}})  # ty: ignore[invalid-argument-type]  # the call is the test

    assert caught.value.code == "invalid_argument"
    assert "at schema['a']: ColumnSpec got the key 1" in str(caught.value)


@pytest.mark.parametrize("flag", [np.True_, np.False_])
def test_a_data_policy_takes_a_numpy_bool_as_the_analyses_do(flag):
    policy = ae.DataPolicy(sensitive_columns=["a"], mask_sensitive=flag)

    assert policy.mask_sensitive is bool(flag)
    assert type(policy.mask_sensitive) is bool
    again = ae.DataPolicy.model_validate(
        {"sensitive_columns": ["a"], "mask_sensitive": flag}
    )
    assert again == policy


def test_a_project_takes_a_numpy_string_as_its_name():
    assert ae.Project(name=np.str_("churn")).name == "churn"


def test_the_project_types_read_the_same_in_the_signature_and_the_message():
    annotation = inspect.signature(ae.Project).parameters["data_policy"].annotation
    assert str(annotation) == str(ae.DataPolicy | Mapping[str, Any] | None)

    with pytest.raises(InvalidArgumentError) as caught:
        ae.Project(data_policy=5)  # ty: ignore[invalid-argument-type]  # the call is the test

    # The message says the same in words, without typing syntax.
    assert "must be a DataPolicy, a dict or None" in str(caught.value)
    # A dict of the fields, which the signature allows, builds one.
    project = ae.Project(data_policy={"sensitive_columns": ["a"]})
    assert project.data_policy == ae.DataPolicy(sensitive_columns=["a"])
