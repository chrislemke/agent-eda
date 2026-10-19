"""The declared logical types: ae.LogicalType and ae.ColumnSpec."""

import typing

import numpy as np
import pydantic
import pytest

import agent_eda as ae
from agent_eda.errors import AgentEDAError, InvalidArgumentError

LOGICAL_TYPES = [
    "boolean",
    "integer",
    "float",
    "categorical",
    "text",
    "datetime",
    "timedelta",
]


def test_logical_type_is_a_literal_of_the_seven_logical_types():
    assert typing.get_origin(ae.LogicalType) is typing.Literal
    assert list(typing.get_args(ae.LogicalType)) == LOGICAL_TYPES


def test_holds_a_logical_type_whether_values_may_be_missing_and_a_datetime_format():
    spec = ae.ColumnSpec(
        logical_type="datetime", nullable=False, datetime_format="%d/%m/%Y"
    )

    assert spec.logical_type == "datetime"
    assert spec.nullable is False
    assert spec.datetime_format == "%d/%m/%Y"


def test_declares_only_the_logical_type_unless_told_more():
    spec = ae.ColumnSpec(logical_type="integer")

    # None declares nothing, so no check depends on it.
    assert spec.nullable is None
    assert spec.datetime_format is None


def test_takes_the_logical_type_by_position_too():
    assert ae.ColumnSpec("text", nullable=True) == ae.ColumnSpec(
        logical_type="text", nullable=True
    )


def test_takes_numpy_scalars_as_python_values():
    spec = ae.ColumnSpec(logical_type=np.str_("boolean"), nullable=np.True_)  # ty: ignore[invalid-argument-type]  # NumPy scalars are accepted at runtime

    assert spec == ae.ColumnSpec(logical_type="boolean", nullable=True)


def invalid_spec(**fields):
    """The error a ColumnSpec with these fields raises."""
    with pytest.raises(InvalidArgumentError) as caught:
        ae.ColumnSpec(**fields)
    error = caught.value
    assert isinstance(error, AgentEDAError)
    assert not isinstance(error, pydantic.ValidationError)
    assert error.__cause__ is None
    return error


def test_an_unknown_logical_type_raises_with_the_allowed_ones_and_the_closest():
    error = invalid_spec(logical_type="int")

    assert error.code == "invalid_argument"
    assert str(error) == (
        "ColumnSpec got 'int' for logical_type, but logical_type must be one of "
        "'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or "
        "'timedelta'. Did you mean 'integer'? See help(ae.ColumnSpec) for every "
        "field."
    )


def test_a_field_of_the_wrong_type_raises_naming_the_type_it_takes():
    error = invalid_spec(logical_type="integer", nullable="no")

    assert error.code == "invalid_argument"
    assert str(error).startswith(
        "ColumnSpec got 'no' for nullable, but nullable must be a boolean, or None."
    )


def test_an_unknown_field_raises_with_the_closest_and_the_allowed_ones():
    error = invalid_spec(logical_type="integer", nulable=False)

    assert error.code == "invalid_call"
    assert str(error) == (
        "ColumnSpec has no field 'nulable'. Did you mean 'nullable'? Allowed: "
        "logical_type, nullable, datetime_format."
    )


def test_a_spec_without_a_logical_type_raises_with_an_example():
    error = invalid_spec(nullable=False)

    assert error.code == "invalid_call"
    assert str(error) == (
        "ColumnSpec needs logical_type, one of 'boolean', 'integer', 'float', "
        "'categorical', 'text', 'datetime' or 'timedelta', such as "
        "ae.ColumnSpec(logical_type='integer', nullable=False)."
    )


def test_more_than_one_positional_field_raises():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.ColumnSpec("integer", False)  # ty: ignore[too-many-positional-arguments]  # the call is the test

    assert caught.value.code == "invalid_call"
    assert str(caught.value) == (
        "ColumnSpec takes only logical_type by position; pass the other fields by "
        "keyword, such as ae.ColumnSpec('integer', nullable=False)."
    )


def test_a_datetime_format_for_another_logical_type_raises():
    error = invalid_spec(logical_type="integer", datetime_format="%Y")

    assert error.code == "invalid_argument"
    assert str(error) == (
        "ColumnSpec got datetime_format '%Y' with logical_type 'integer', but a "
        "datetime_format applies only to logical_type 'datetime'. Leave "
        "datetime_format out, or declare logical_type='datetime'."
    )


def test_validating_a_dict_of_fields_raises_in_the_package_format_too():
    assert ae.ColumnSpec.model_validate({"logical_type": "float"}) == ae.ColumnSpec(
        "float"
    )
    with pytest.raises(InvalidArgumentError) as caught:
        ae.ColumnSpec.model_validate({"logical_type": "decimal"})

    assert not isinstance(caught.value, pydantic.ValidationError)
    assert str(caught.value).startswith(
        "ColumnSpec got 'decimal' for logical_type, but logical_type must be one of"
    )


def test_validating_something_other_than_a_dict_raises_with_an_example():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.ColumnSpec.model_validate("integer")

    assert str(caught.value) == (
        "ColumnSpec.model_validate takes a dict of the fields, such as "
        "{'logical_type': 'integer', 'nullable': False}, but got a str. "
        "ae.ColumnSpec('integer') builds one from a logical type."
    )


@pytest.mark.parametrize("datetime_format", ["", "dd/mm/yyyy", "%d/%m/%Q"])
def test_a_datetime_format_without_known_directives_raises(datetime_format):
    error = invalid_spec(logical_type="datetime", datetime_format=datetime_format)

    assert error.code == "invalid_argument"
    assert str(error).startswith(
        f"ColumnSpec got datetime_format {datetime_format!r}, but a datetime_format "
        "is a format for datetime.strptime, with directives such as %Y, %m and %d, "
        "like '%d/%m/%Y'."
    )


def test_the_schema_describes_a_column_spec_without_its_python_example():
    record = ae.describe_analysis("validate_column_types").to_dict()
    description = record["parameters_schema"]["$defs"]["ColumnSpec"]["description"]

    assert description.startswith("What a column is declared to hold")
    assert "Examples:" not in description
    assert ">>>" not in description
