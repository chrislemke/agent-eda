"""The Project: an optional declaration of a data science project's data policy.

Every analysis runs without a Project. The analyses whose results can show raw
values take one as project=, and its data policy says which columns are
sensitive and whether to mask them. The data policy is declared intent, not
access control: the package runs in the caller's own process, which can read
the data directly (ADR 0001).
"""

import inspect
import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self

import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_eda._registry import invalid_message, python_scalar, unknown_name_message
from agent_eda._run import check_distinct
from agent_eda.errors import AgentEDAError, InvalidArgumentError


class _Declaration(BaseModel):
    """Base of the models callers build to declare something, such as a Project.

    Building one raises InvalidArgumentError in the package's format, never
    pydantic's ValidationError: by calling it, with model_validate,
    model_validate_json or model_validate_strings, or from a dict of its
    fields inside another declaration or given to an analysis.
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
            try:
                super().__init__(**fields)
            except pydantic.ValidationError as error:
                raise _declaration_error(type(self), fields, error.errors()) from None

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        """Build one from a dict of its fields, as JSON gives it."""
        if not isinstance(obj, (Mapping, cls)):
            raise InvalidArgumentError(
                f"{cls.__name__}.model_validate takes a dict of the fields, but got "
                f"a {type(obj).__name__}. {cls._example} builds one in code."
            )
        try:
            return super().model_validate(obj, *args, **kwargs)
        except pydantic.ValidationError as error:
            fields = dict(obj) if isinstance(obj, Mapping) else {}
            raise _declaration_error(cls, fields, error.errors()) from None

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
            raise _declaration_error(cls, dict(obj), error.errors()) from None

    @model_validator(mode="wrap")
    @classmethod
    def _fields_named_by_strings(cls, value: Any, handler: Any) -> Any:
        """A dict whose keys aren't all strings can't name fields; say so.

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
        return handler(value)


def _declaration_error(
    model: type[_Declaration], fields: Mapping[str, Any], problems: list[Any]
) -> AgentEDAError:
    """The error to raise for a declaration that pydantic rejected.

    A check of the package's own, such as a nested declaration's, raised an
    error that says it best. Otherwise the first problem is shown in the
    format of the analyses' argument errors, as ColumnSpec shows its own.
    """
    for problem in problems:
        error = problem.get("ctx", {}).get("error")
        if isinstance(error, AgentEDAError):
            return error
    first = problems[0]
    if first["type"] == "extra_forbidden":
        name = str(first["loc"][0])
        return InvalidArgumentError(_unknown_field(model, name), code="invalid_call")
    # As lists, which the message follows to the value at fault.
    given = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in fields.items()
    }
    return InvalidArgumentError(
        invalid_message(
            model.__name__, model._field_types, given, problems, noun="field"
        )
    )


def _unknown_field(model: type[_Declaration], name: str) -> str:
    """Say that a declaration has no such field, naming the closest and the rest."""
    allowed = list(model.model_fields)
    if model is Project and name in DataPolicy.model_fields:
        return (
            f"Project has no field {name!r}. It is a field of the data policy: "
            f"ae.Project(data_policy=ae.DataPolicy({name}=...)). "
            f"Allowed: {', '.join(allowed)}."
        )
    return unknown_name_message(model.__name__, "field", name, allowed)


class DataPolicy(_Declaration):
    """Which columns hold sensitive values, and whether results mask them.

    A sensitive column holds values that should not appear in results when
    masking is on, such as names or email addresses. Masking replaces the
    values of sensitive columns that a result would show, such as category
    labels and row IDs taken from a sensitive id_column, with numbered
    placeholders ('<masked 1>', '<masked 2>' and so on) that reveal neither
    content nor length. Counts, shares and statistics stay. Masking stays off,
    even with sensitive columns declared, until mask_sensitive is True here
    or in the call; a call's mask_sensitive argument wins.

    Masking is a guardrail against accidental exposure, such as raw values
    reaching your model provider through a printed result. It is not
    anonymization: counts and statistics stay, and the data itself never
    changes. It is not access control either: the package runs in your own
    process, which can read the data directly. Index labels are never
    masked, so when the index holds sensitive values, pass a non-sensitive
    id_column to the analyses that show row IDs.

    Args:
        sensitive_columns (Sequence[str] | None): Names of the columns that
            hold sensitive values: a list, a tuple, a set or a single name. A
            label that isn't a string can be given as itself, such as 0, or as
            its string form. Names the data lacks are allowed, since a project
            can span several datasets. None or an empty list, the default,
            declares none.
        mask_sensitive (bool): Whether analyses mask the values of the
            sensitive columns when the call's own mask_sensitive is None.
            Defaults to False. True needs sensitive_columns.

    Examples:
        >>> import agent_eda as ae
        >>> ae.DataPolicy(sensitive_columns=["email", "name"], mask_sensitive=True)
        DataPolicy(sensitive_columns=('email', 'name'), mask_sensitive=True)
    """

    # The docstring is long for a JSON Schema, which tool frameworks show.
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Which columns hold sensitive values, and whether "
            "results mask them with numbered placeholders. Masking is a "
            "guardrail against accidental exposure, not anonymization or access "
            "control."
        }
    )

    _example = "ae.DataPolicy(sensitive_columns=['email'], mask_sensitive=True)"
    _field_types = {"sensitive_columns": Sequence[str] | None, "mask_sensitive": bool}

    sensitive_columns: tuple[str, ...] = Field(
        default=(),
        description="Names of the columns that hold sensitive values; empty, the "
        "default, declares none.",
    )
    mask_sensitive: bool = Field(
        default=False,
        description="Whether analyses mask the values of the sensitive columns "
        "when the call's mask_sensitive is None. Defaults to false; true needs "
        "sensitive_columns.",
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            sensitive_columns: Sequence[str] | None = (),
            mask_sensitive: bool = False,
        ) -> None: ...

    @field_validator("sensitive_columns", mode="before")
    @classmethod
    def _names(cls, value: Any) -> Any:
        """Column names as the analyses' columns argument takes them."""
        if value is None:
            return ()
        if isinstance(value, pd.Series | pd.DataFrame):
            # Its values would become names, and messages would show them.
            names = (
                "[series.name] for the column it came from"
                if isinstance(value, pd.Series)
                else "list(df.columns) for its columns"
            )
            raise InvalidArgumentError(
                f"DataPolicy got a pandas {type(value).__name__} for "
                "sensitive_columns, but sensitive_columns takes column names, not "
                f"data. Pass the names, such as sensitive_columns=['email'], or "
                f"{names}."
            )
        if isinstance(value, Mapping):
            raise InvalidArgumentError(
                f"DataPolicy got a {type(value).__name__} for sensitive_columns, but "
                "sensitive_columns must be a list of column names, such as "
                f"{cls._example}."
            )
        if isinstance(value, str | bytes) or not isinstance(value, Iterable):
            items = [value]
        else:
            items = value.tolist() if hasattr(value, "tolist") else list(value)
        names = [_name(item) for item in items]
        if isinstance(value, set | frozenset):
            names = sorted(names, key=str)
        # Validation rejects anything that isn't a name, naming its position.
        texts = [name for name in names if isinstance(name, str)]
        check_distinct("DataPolicy", "sensitive_columns", texts, "column")
        return tuple(names)

    @model_validator(mode="after")
    def _something_to_mask(self) -> Self:
        if self.mask_sensitive and not self.sensitive_columns:
            raise InvalidArgumentError(
                "DataPolicy got mask_sensitive=True but no sensitive_columns, so "
                "there would be nothing to mask. Name the sensitive columns, such "
                f"as {self._example}."
            )
        return self


def _name(item: object) -> object:
    """A column label as the string form results show, if it is a label at all."""
    item = python_scalar(item)
    if isinstance(item, Hashable):
        return str(item)
    return item  # validation rejects it, naming its type


class Project(_Declaration):
    """A data science project's declarations, given to the analyses that use them.

    A Project is optional, and every analysis runs without one. It holds a
    name and the data policy, which declares the sensitive columns and whether
    to mask them. Pass it as project= to the analyses whose results can show
    raw values; ae.list_analyses() marks them [supports masking]. Build it
    once and pass the same Project to every call.

    The data policy is declared intent that the package checks to catch
    honest mistakes. Masking is a guardrail against accidental exposure, not
    anonymization and not access control: see help(ae.DataPolicy).

    Args:
        name (str | None): The project's name, such as 'churn'. None, the
            default, leaves it unnamed.
        data_policy (DataPolicy): Which columns are sensitive and whether to
            mask them: a DataPolicy, or a dict of its fields. Defaults to
            DataPolicy(), which declares no sensitive column and leaves masking
            off.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> project = ae.Project(
        ...     name="support tickets",
        ...     data_policy=ae.DataPolicy(
        ...         sensitive_columns=["customer"], mask_sensitive=True
        ...     ),
        ... )
        >>> df = pd.DataFrame(
        ...     {
        ...         "customer": ["Ann Lee", "Bo Chu", "Ann Lee"],
        ...         "plan": ["pro", "pro", "team"],
        ...     }
        ... )
        >>> res = ae.summarize_categorical(df, project=project)
        >>> [(level.label, level.count) for level in res.columns["customer"].levels]
        [('<masked 1>', 2), ('<masked 2>', 1)]
        >>> res.masking.columns
        ['customer']
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "A data science project's declarations: its name and "
            "its data policy, which names the sensitive columns and whether to "
            "mask them."
        }
    )

    _example = (
        "ae.Project(name='churn', "
        "data_policy=ae.DataPolicy(sensitive_columns=['email']))"
    )
    _field_types = {"name": str | None, "data_policy": DataPolicy}

    name: str | None = Field(
        default=None,
        description="The project's name; null, the default, leaves it unnamed.",
    )
    data_policy: DataPolicy = Field(
        default_factory=DataPolicy,
        description="Which columns are sensitive and whether to mask them. "
        "Defaults to a policy that declares no sensitive column.",
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            name: str | None = None,
            data_policy: DataPolicy | Mapping[str, Any] | None = None,
        ) -> None: ...

    @field_validator("data_policy", mode="before")
    @classmethod
    def _default_policy(cls, value: Any) -> Any:
        """None means the default policy, as the signature shows."""
        return DataPolicy() if value is None else value

    @model_validator(mode="before")
    @classmethod
    def _not_a_policy(cls, value: Any) -> Any:
        """A data policy where a project belongs, as in project=policy, raises."""
        if isinstance(value, DataPolicy):
            raise InvalidArgumentError(
                "project takes a Project, but got a DataPolicy. Put the policy in a "
                "project: project=ae.Project(data_policy=policy)."
            )
        return value


# help() and editors show the calls as they read, not as pydantic builds them.
DataPolicy.__signature__ = inspect.Signature(
    [
        inspect.Parameter(
            "sensitive_columns",
            inspect.Parameter.KEYWORD_ONLY,
            default=(),
            annotation=Sequence[str] | None,
        ),
        inspect.Parameter(
            "mask_sensitive",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool,
        ),
    ]
)
Project.__signature__ = inspect.Signature(
    [
        inspect.Parameter(
            "name", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=str | None
        ),
        inspect.Parameter(
            "data_policy",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=DataPolicy | None,
        ),
    ]
)
