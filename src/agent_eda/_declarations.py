"""The base of the models callers build to declare something.

Project, DataPolicy and ColumnSpec (which lives here) share it, so every way of building one
fails the same way: with an InvalidArgumentError in the package's format, never
pydantic's ValidationError and never a bare TypeError. That holds for the
constructor, model_validate, model_validate_json, model_validate_strings, a dict
with a key that isn't a string, and model_copy(update=...), which validates its
result instead of building an object that no check has seen. NumPy scalars are
accepted wherever Python scalars are.
"""

import inspect
import json
import re
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import pydantic
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from agent_eda._arguments import invalid_message, python_scalar, unknown_name_message
from agent_eda._docstring import without_examples
from agent_eda._result_base import LogicalType
from agent_eda._text import a_or_an, join_alternatives
from agent_eda._wording import Pass, Words, mistake
from agent_eda.errors import AgentEDAError, InvalidArgumentError


class Declaration(BaseModel):
    """Base of the models callers build to declare something, such as a Project.

    Building one raises InvalidArgumentError in the package's format, never
    pydantic's ValidationError: by calling it, with model_validate,
    model_validate_json or model_validate_strings, from a dict of its fields
    inside another declaration or given to an analysis, or by model_copy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    # A call that builds one, for messages that show how.
    _example: ClassVar[str]
    # Each field's type as callers pass it, for messages and help().
    _field_types: ClassVar[dict[str, object]]

    if not TYPE_CHECKING:
        # Type checkers read each subclass's own signature instead.

        def __init__(self, *args: Any, **fields: Any) -> None:
            if args:
                raise InvalidArgumentError(
                    f"{type(self).__name__} takes its fields by keyword, such as "
                    f"{self._example}.",
                    code="invalid_call",
                )
            fields = {name: python_scalar(value) for name, value in fields.items()}
            try:
                super().__init__(**fields)
            except pydantic.ValidationError as error:
                raise type(self)._rejected(fields, error.errors()) from None

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        """Build one from a dict of its fields, as JSON gives it."""
        if not isinstance(obj, (Mapping, cls)):
            raise InvalidArgumentError(
                f"{cls.__name__}.model_validate takes a dict of the fields, but got "
                f"{a_or_an(type(obj).__name__)}. {cls._example} builds one in code."
            )
        try:
            return super().model_validate(obj, *args, **kwargs)
        except pydantic.ValidationError as error:
            fields = dict(obj) if isinstance(obj, Mapping) else {}
            raise cls._rejected(fields, error.errors()) from None

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, *args: Any, **kwargs: Any
    ) -> Self:
        """Build one from JSON text of its fields, as model_dump_json() writes it."""
        try:
            fields = json.loads(json_data)
        except (TypeError, ValueError) as error:
            raise InvalidArgumentError(
                f"{cls.__name__}.model_validate_json got text that isn't valid JSON: "
                f"{error}. Pass the text model_dump_json() writes, a JSON object of "
                "the fields."
            ) from None
        return cls.model_validate(fields, *args, **kwargs)

    @classmethod
    def model_validate_strings(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        """Build one from a dict of its fields whose values are strings."""
        if not isinstance(obj, Mapping):
            raise InvalidArgumentError(
                f"{cls.__name__}.model_validate_strings takes a dict of the fields, "
                f"but got a {type(obj).__name__}. {cls._example} builds one in code."
            )
        try:
            return super().model_validate_strings(obj, *args, **kwargs)
        except pydantic.ValidationError as error:
            raise cls._rejected(dict(obj), error.errors()) from None

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """A copy with some fields replaced, checked as building one is.

        pydantic's own copy trusts the update, so it would make a declaration
        that no check has seen; this one validates the result.
        """
        if not update:
            return super().model_copy(deep=deep)
        fields = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self).model_validate({**fields, **update})

    @model_validator(mode="wrap")
    @classmethod
    def _fields_as_python_values(cls, value: Any, handler: Any) -> Any:
        """A dict of fields whose keys are all strings, with NumPy scalars as Python's.

        This runs before pydantic hands a dict of fields to __init__, where a
        key that isn't a string would raise a bare TypeError.
        """
        if isinstance(value, Mapping):
            keys = [key for key in value if not isinstance(key, str)]
            if keys:
                raise InvalidArgumentError(
                    f"{cls.__name__} got the key {keys[0]!r}, but its fields are "
                    f"named by strings: {', '.join(cls.model_fields)}.",
                    code="invalid_call",
                )
            value = {name: python_scalar(item) for name, item in value.items()}
        return handler(value)

    @classmethod
    def _unknown_field(cls, name: str) -> str:
        """Say that there is no such field, naming the closest and the rest."""
        return unknown_name_message(cls.__name__, "field", name, list(cls.model_fields))

    @classmethod
    def _rejected(cls, fields: Mapping[str, Any], problems: list[Any]) -> AgentEDAError:
        """The error to raise for fields that pydantic rejected.

        A check of the package's own, such as a nested declaration's, raised an
        error that says it best. Otherwise the first problem is shown in the
        format of the analyses' argument errors.
        """
        for problem in problems:
            error = problem.get("ctx", {}).get("error")
            if isinstance(error, AgentEDAError):
                return error
        # A field that doesn't exist comes first: pydantic lists a missing field
        # before it, but the missing one is often the misspelled one.
        extra = next((p for p in problems if p["type"] == "extra_forbidden"), None)
        if extra is not None:
            name = str(extra["loc"][0])
            return InvalidArgumentError(cls._unknown_field(name), code="invalid_call")
        # As lists, which the message follows to the value at fault.
        given = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in fields.items()
        }
        return mistake(
            InvalidArgumentError,
            invalid_message(
                cls.__name__, cls._field_types, given, problems, noun="field"
            ),
        )


def keyword_signature(
    model: type[Declaration], defaults: Mapping[str, object]
) -> inspect.Signature:
    """The call of a declaration as help() shows it: keywords with their types.

    The types are the model's _field_types, so help(), the TYPE_CHECKING stub's
    twin and the messages state each type once.
    """
    return inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=model._field_types[name],
            )
            for name, default in defaults.items()
        ]
    )


_LOGICAL_TYPES = typing.get_args(LogicalType)

# The directives datetime.strptime knows, as %Y or %:z; %% is a literal %.
_DIRECTIVE = re.compile(r"%(:z|.?)", re.DOTALL)
_KNOWN_DIRECTIVES = frozenset("aAwdbBmyYHIpMSfzZjUWcxXGuV%") | {":z"}


class ColumnSpec(Declaration):
    """What a column is declared to hold: its logical type and more.

    A declared schema maps column names to a ColumnSpec or to a logical type
    alone, which is short for a ColumnSpec with only logical_type. Fields other
    than logical_type are optional, and None declares nothing, so no check
    depends on them. An invalid spec is rejected with the code invalid_argument.

    Examples:
        >>> import agent_eda as ae
        >>> ae.ColumnSpec("datetime", nullable=False, datetime_format="%d/%m/%Y")
        ColumnSpec(logical_type='datetime', nullable=False, datetime_format='%d/%m/%Y')
    """

    _example = "ae.ColumnSpec('integer', nullable=False)"
    _field_types: ClassVar[dict[str, object]] = {
        "logical_type": LogicalType,
        "nullable": bool | None,
        "datetime_format": str | None,
    }

    logical_type: LogicalType = Field(
        description="What the values mean: 'boolean', 'integer', 'float', "
        "'categorical', 'text', 'datetime' or 'timedelta'."
    )
    nullable: bool | None = Field(
        default=None,
        description="Whether the column may hold missing values; None, the "
        "default, declares nothing.",
    )
    datetime_format: str | None = Field(
        default=None,
        description="For logical_type 'datetime' only: the datetime.strptime "
        "format of datetimes stored as text, such as '%d/%m/%Y'. None, the "
        "default, declares none, so only ISO 8601 text reads as a datetime. "
        "infer_column_semantics and validate_column_types both read text in "
        "this format as datetimes. Directives that depend on the locale, such "
        "as %b and %p, read in the process's locale, so a process with another "
        "locale can count differently.",
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            logical_type: LogicalType,
            *,
            nullable: bool | None = None,
            datetime_format: str | None = None,
        ) -> None: ...

    else:

        def __init__(self, *args: Any, **fields: Any) -> None:
            if len(args) > 1:
                raise InvalidArgumentError(
                    "ColumnSpec takes only logical_type by position; pass the "
                    "other fields by keyword, such as "
                    "ae.ColumnSpec('integer', nullable=False).",
                    code="invalid_call",
                )
            if args:
                if "logical_type" in fields:
                    raise InvalidArgumentError(
                        "ColumnSpec got logical_type both by position and by "
                        "keyword; pass it once.",
                        code="invalid_call",
                    )
                fields = {"logical_type": args[0], **fields}
            super().__init__(**fields)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """The JSON Schema, described without Examples:, which is Python.

        pydantic 2.7 adds the docstring after json_schema_extra runs, so the
        description is cut here, where every version has it.
        """
        schema = handler(core_schema)
        without_examples(handler.resolve_ref_schema(schema))
        return schema

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        """Build a ColumnSpec from a dict of its fields, as JSON gives one."""
        if not isinstance(obj, Mapping | ColumnSpec):
            raise InvalidArgumentError(
                "ColumnSpec.model_validate takes a dict of the fields, such as "
                "{'logical_type': 'integer', 'nullable': False}, but got a "
                f"{type(obj).__name__}. ae.ColumnSpec('integer') builds one from "
                "a logical type."
            )
        return super().model_validate(obj, *args, **kwargs)

    @classmethod
    def _rejected(cls, fields: Mapping[str, Any], problems: list[Any]) -> AgentEDAError:
        """A spec without a logical_type says how to write one for the surface."""
        first = problems[0]
        if (
            first["type"] == "missing"
            and not any(
                isinstance(problem.get("ctx", {}).get("error"), AgentEDAError)
                for problem in problems
            )
            and not any(problem["type"] == "extra_forbidden" for problem in problems)
        ):
            allowed = join_alternatives([repr(t) for t in _LOGICAL_TYPES])
            return mistake(
                InvalidArgumentError,
                f"ColumnSpec needs logical_type, one of {allowed}, such as ",
                _SPEC_EXAMPLE,
                ".",
                code="invalid_call",
            )
        return super()._rejected(fields, problems)

    @model_validator(mode="after")
    def _check_datetime_format(self) -> Self:
        text = self.datetime_format
        if text is None:
            return self
        if self.logical_type != "datetime":
            raise mistake(
                InvalidArgumentError,
                f"ColumnSpec got datetime_format {text!r} with "
                f"logical_type {self.logical_type!r}, but a datetime_format applies "
                "only to logical_type 'datetime'. Leave datetime_format out, or "
                "declare ",
                Pass(logical_type="datetime"),
                ".",
            )
        directives = _DIRECTIVE.findall(text)
        unknown = [d for d in directives if d not in _KNOWN_DIRECTIVES]
        if unknown or not [d for d in directives if d != "%"]:
            problem = (
                f" It has an unknown directive: %{unknown[0]}."
                if unknown
                else " It has no directive."
            )
            raise mistake(
                InvalidArgumentError,
                f"ColumnSpec got datetime_format {text!r}, but a datetime_format is ",
                Words("a format for datetime.strptime", "a date format"),
                f", with directives such as %Y, %m and %d, like '%d/%m/%Y'.{problem}",
            )
        return self


# help() and editors show the call as it reads, not as pydantic builds it.
ColumnSpec.__signature__ = inspect.Signature(
    [
        inspect.Parameter(
            "logical_type",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=LogicalType,
        ),
        inspect.Parameter(
            "nullable",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=bool | None,
        ),
        inspect.Parameter(
            "datetime_format",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str | None,
        ),
    ]
)


# A ColumnSpec as the caller writes one.
_SPEC_FIELDS = Pass(logical_type="integer", nullable=False)
_SPEC_EXAMPLE = Words(("ae.ColumnSpec(", _SPEC_FIELDS, ")"), _SPEC_FIELDS)
