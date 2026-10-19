"""The registry of analyses and the decorator that registers each one.

The decorator records an analysis's metadata and wraps it with the checks every
analysis shares. The call's shape, the type of `data`, its column labels and the
keyword arguments are checked before any computation starts. Warnings raised
while the analysis runs never reach the caller, and the result is checked
against what the analysis declared.
"""

import contextlib
import functools
import importlib
import inspect
import json
import os
import re
import typing
import warnings
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from types import FunctionType
from typing import Any, cast

import numpy as np
import pandas as pd
import pydantic

from agent_eda._annotations import (
    LIST_TYPES,
    MAPPING_TYPES,
    allowed_text,
    item_annotation,
    members,
    strip_optional,
)
from agent_eda._columns import check_column_labels
from agent_eda._docstring import Docstring
from agent_eda._docstring import parse as parse_docstring
from agent_eda._run import capture_warnings, watch_all_warnings
from agent_eda._text import (
    MAX_NAME_LENGTH,
    Surface,
    a_or_an,
    closest_names,
    join_items,
    plural,
    short_repr,
)
from agent_eda._wording import echoed, message, see_every, worded
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.results import AnalysisResult, Limitation, Status, StatusReasonCode

_CODE = re.compile(r"[a-z][a-z0-9_]*")

# Search tags of every analysis that supports masking. They avoid words such as
# "values" and "columns", which questions about the data use too.
MASKING_TAGS = ("masking", "mask sensitive")


@dataclass(frozen=True)
class AnalysisSpec:
    """What the registration decorator records about one analysis."""

    name: str
    module: str
    summary: str
    docstring: Docstring
    tags: tuple[str, ...]
    signature: inspect.Signature
    methods: tuple[str, ...]
    statuses: tuple[Status, ...]
    finding_codes: tuple[str, ...]
    result_class: type[AnalysisResult]
    supports_masking: bool
    work_limits: tuple[str, ...]
    arguments: type[pydantic.BaseModel]
    # Each parameter's annotation, data's included, resolved from the type hints
    # with Annotated ranges kept.
    annotations: dict[str, object]
    any_column_labels: bool

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.signature.parameters)

    def reason_codes(self, status: str) -> tuple[StatusReasonCode, ...]:
        """The status reason codes the Statuses: section gives a status."""
        return cast(
            "tuple[StatusReasonCode, ...]", self.docstring.status_codes.get(status, ())
        )

    def parameters_schema(self) -> dict[str, Any]:
        """The JSON Schema of the options, with each one's Args: description.

        Raises pydantic's PydanticInvalidForJsonSchema when an option's type
        has no JSON Schema.
        """
        schema = self.arguments.model_json_schema()
        for name, prop in schema.get("properties", {}).items():
            prop["description"] = self.docstring.args.get(name, "")
        return schema


_REGISTRY: dict[str, AnalysisSpec] = {}

# The arguments of an analysis that shows raw values, and of its MCP tool.
_MASKING_ARGUMENTS = frozenset({"project", "project_path", "mask_sensitive"})


def analysis[F: FunctionType](
    *,
    statuses: tuple[Status, ...],
    finding_codes: tuple[str, ...],
    tags: tuple[str, ...] = (),
    methods: tuple[str, ...] = (),
    masking: bool = False,
    work_limits: tuple[str, ...] = (),
    any_column_labels: bool = False,
) -> Callable[[F], F]:
    """Register an analysis and wrap it with the shared checks.

    Args:
        statuses: Every status the analysis can return.
        finding_codes: Every finding code the analysis can report.
        tags: Words that help search find the analysis.
        methods: The values its `method` argument accepts, if it has one.
        masking: Whether it supports masking sensitive columns. Such an
            analysis also gets the tags in MASKING_TAGS, so a search for
            masking finds every one of them.
        work_limits: Its `max_*` parameters that bound the work it does.
        any_column_labels: Whether it takes data whose column labels can't
            serve as names (duplicated, colliding or a MultiIndex), as
            describe_dataset does to report them. Every other analysis raises.
    """

    def register(func: F) -> F:
        spec = _spec(
            func,
            statuses=statuses,
            finding_codes=finding_codes,
            tags=tags,
            methods=methods,
            masking=masking,
            work_limits=work_limits,
            any_column_labels=any_column_labels,
        )
        # Reloading a module (as Jupyter's autoreload does) registers it again.
        known = _REGISTRY.get(spec.name)
        if known is not None and known.module != spec.module:
            raise TypeError(
                f"analysis {spec.name!r} is defined in both {known.module} and "
                f"{spec.module}; analysis names must be unique"
            )
        _REGISTRY[spec.name] = spec

        @functools.wraps(func)
        def checked(*args: Any, **kwargs: Any) -> Any:
            data, options = _bind(spec, args, kwargs)
            _check_data(spec.name, data)
            if not spec.any_column_labels:
                check_column_labels(spec.name, data.columns)
            validated = _validate(spec, data, options)
            with watch_all_warnings() as caught:
                result = func(data, **validated)
            result = _absorb_warnings(result, caught)
            _check_result(spec, result)
            return result

        # pytest would otherwise collect analyses named test_*.
        checked.__dict__["__test__"] = False
        return cast("F", checked)

    return register


def registered() -> list[AnalysisSpec]:
    """Every analysis exported by agent_eda, ordered by module, then name."""
    import agent_eda

    for module in sorted(set(agent_eda._EXPORTS.values())):
        importlib.import_module(module)
    return sorted(_REGISTRY.values(), key=lambda spec: (spec.module, spec.name))


@contextlib.contextmanager
def listed_as_registered(functions: Iterable[Callable[..., Any]]) -> Generator[None]:
    """List stand-in analysis functions as registered ones while the block runs.

    Only tests use it, to see how the guide and the MCP server's instructions
    grow with the number of analyses. Each function is read the way spec_of
    reads an unregistered one. Its name must not belong to a registered
    analysis, and the registry is restored on leaving the block.
    """
    specs = [spec_of(function) for function in functions]
    taken = [spec.name for spec in specs if spec.name in _REGISTRY]
    if taken:
        raise ValueError(f"{taken} are registered analyses, not stand-ins")
    _REGISTRY.update({spec.name: spec for spec in specs})
    try:
        yield
    finally:
        for spec in specs:
            _REGISTRY.pop(spec.name, None)


def spec_of(function: Callable[..., Any]) -> AnalysisSpec:
    """The registry entry of an analysis function.

    A function the registry doesn't hold, such as a stand-in a test defines, is
    read the way the decorator reads an analysis, without registering it: its
    statuses and finding codes are the ones its docstring lists.
    """
    name = getattr(function, "__name__", None)
    spec = _REGISTRY.get(name) if isinstance(name, str) else None
    if spec is not None and getattr(function, "__module__", None) == spec.module:
        return spec
    if not isinstance(function, FunctionType):
        raise TypeError(f"{function!r} is not an analysis function")
    docstring = parse_docstring(inspect.getdoc(function) or "")
    return _spec(
        function,
        statuses=cast("tuple[Status, ...]", tuple(docstring.statuses)),
        finding_codes=tuple(docstring.codes),
        tags=(),
        methods=(),
        masking=False,
        work_limits=(),
        any_column_labels=False,
    )


def _spec(
    func: FunctionType,
    *,
    statuses: tuple[Status, ...],
    finding_codes: tuple[str, ...],
    tags: tuple[str, ...],
    methods: tuple[str, ...],
    masking: bool,
    work_limits: tuple[str, ...],
    any_column_labels: bool,
) -> AnalysisSpec:
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        raise TypeError(f"analysis {name!r} has no docstring")
    signature = inspect.signature(func)
    # include_extras keeps the ranges declared with Annotated.
    hints = typing.get_type_hints(func, include_extras=True)
    names = list(signature.parameters)
    if not names or names[0] != "data" or "data" not in hints:
        raise TypeError(
            f"analysis {name!r} must take data: pd.DataFrame as its first parameter"
        )
    if any(
        p.kind is not inspect.Parameter.KEYWORD_ONLY
        for p in list(signature.parameters.values())[1:]
    ):
        raise TypeError(f"analysis {name!r} must take every option by keyword")
    result_class = hints.get("return")
    if not (
        isinstance(result_class, type) and issubclass(result_class, AnalysisResult)
    ):
        raise TypeError(f"analysis {name!r} must return an AnalysisResult subclass")
    bad_codes = [code for code in finding_codes if not _CODE.fullmatch(code)]
    if bad_codes:
        raise TypeError(f"analysis {name!r} has malformed finding codes {bad_codes}")
    bad_limits = [p for p in work_limits if p not in names or not p.startswith("max_")]
    if bad_limits:
        raise TypeError(f"analysis {name!r} has unknown work limits {bad_limits}")
    # pydantic marks a field required with ..., where inspect marks a parameter
    # without a default with Parameter.empty.
    options = {
        p.name: (hints[p.name], ... if p.default is p.empty else p.default)
        for p in list(signature.parameters.values())[1:]
    }
    with warnings.catch_warnings():
        # An option may share a name with a BaseModel attribute, such as
        # validate_column_types' schema; the arguments model never uses it.
        warnings.filterwarnings(
            "ignore",
            message=r"Field name .* shadows an attribute",
            category=UserWarning,
        )
        arguments = pydantic.create_model(
            f"{result_class.__name__.removesuffix('Result')}Arguments",
            __config__=pydantic.ConfigDict(
                strict=True, extra="forbid", arbitrary_types_allowed=True
            ),
            **cast("dict[str, Any]", options),
        )
    docstring = parse_docstring(doc)
    known_codes = typing.get_args(StatusReasonCode)
    bad_reasons = [
        code
        for codes in docstring.status_codes.values()
        for code in codes
        if code not in known_codes
    ]
    if bad_reasons:
        raise TypeError(
            f"analysis {name!r} names status reason codes {bad_reasons} that "
            f"StatusReasonCode doesn't list: {list(known_codes)}"
        )
    return AnalysisSpec(
        name=name,
        module=func.__module__,
        summary=docstring.summary,
        docstring=docstring,
        tags=(*tags, *MASKING_TAGS) if masking else tags,
        signature=signature,
        methods=methods,
        statuses=statuses,
        finding_codes=finding_codes,
        result_class=result_class,
        supports_masking=masking,
        work_limits=work_limits,
        arguments=arguments,
        annotations={
            "data": hints["data"],
            **{option: annotation for option, (annotation, _) in options.items()},
        },
        any_column_labels=any_column_labels,
    )


def _bind(
    spec: AnalysisSpec, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Split a call into data and the keyword options, or raise on a bad call."""
    options = spec.parameters[1:]
    if len(args) > 1:
        example = f", {options[0]}=..." if options else ""
        raise InvalidArgumentError(
            f"{spec.name} takes data as its only positional argument; pass "
            f"everything else by keyword, such as {spec.name}(data{example}).",
            code="invalid_call",
        )
    unknown = [key for key in kwargs if key not in spec.parameters]
    if unknown:
        raise InvalidArgumentError(
            unknown_name_message(spec.name, "argument", unknown[0], spec.parameters),
            code="invalid_call",
        )
    if args and "data" in kwargs:
        raise InvalidArgumentError(
            f"{spec.name} got data both by position and by keyword; pass it once.",
            code="invalid_call",
        )
    if not args and "data" not in kwargs:
        raise InvalidArgumentError(
            f"{spec.name} needs data, a pandas DataFrame: {spec.name}(df).",
            code="invalid_call",
        )
    missing = [
        name
        for name, parameter in spec.signature.parameters.items()
        if name != "data"
        and parameter.default is parameter.empty
        and name not in kwargs
    ]
    if missing:
        raise InvalidArgumentError(_missing_message(spec, missing), code="invalid_call")
    data = args[0] if args else kwargs["data"]
    return data, {key: value for key, value in kwargs.items() if key != "data"}


def _missing_message(spec: AnalysisSpec, missing: list[str]) -> str:
    """Name the options without a default that a call left out, and the fix."""
    example = ", ".join(f"{name}=..." for name in missing)
    return missing_message(
        spec,
        missing,
        fix=f"by keyword, such as ae.{spec.name}(df, {example})",
        see=see_every(spec.name, "argument", "python"),
    )


def missing_message(
    spec: AnalysisSpec, missing: list[str], *, fix: str, see: str
) -> str:
    """Name the required arguments a call left out, how to pass them, and where
    to read about every argument.

    The choices of an option that takes fixed values are listed too, and so
    are those of the values inside a mapping or list option, such as a schema.
    A name that isn't an option, such as an MCP tool's data_path, has none.
    """
    arguments = plural(len(missing), "required argument", "required arguments")
    them = "it" if len(missing) == 1 else "them"
    choices = []
    for name in missing:
        if name not in spec.annotations:
            continue
        annotation = strip_optional(spec.annotations[name])
        item = item_annotation(annotation)
        if typing.get_origin(annotation) is typing.Literal:
            choices.append(f"{name} is {allowed_text(annotation, name)}")
        elif item is not annotation and any(
            typing.get_origin(member) is typing.Literal for member in members(item)
        ):
            choices.append(f"each value in {name} is {allowed_text(item, 'value')}")
    where = f", where {join_items(choices)}" if choices else ""
    return (
        f"{spec.name} is missing {arguments}: {join_items(missing)}. Pass {them} "
        f"{fix}{where}. {see}"
    )


def _check_data(name: str, data: object) -> None:
    """Raise unless data is a pandas DataFrame, showing the conversion."""
    if isinstance(data, pd.DataFrame):
        return
    raise InvalidArgumentError(_not_a_frame(name, data), code="unsupported_data_type")


def _not_a_frame(name: str, data: object) -> str:
    """Say what data is, and the one-line conversion to a DataFrame if there is one."""
    start = f"data must be a pandas DataFrame, but got {_describe_data(data)}."
    call = f"ae.{name}"
    if data is None:
        return (
            f"{start} The variable holds no data, so whatever should have loaded "
            f"or built it didn't. Check that step, then pass its DataFrame: "
            f"{call}(df)."
        )
    if isinstance(data, str | os.PathLike):
        return (
            f"{start} A path isn't data; read the file first: "
            f"{call}(pd.read_csv(path)) for a CSV file, or "
            f"{call}(pd.read_parquet(path)) for a Parquet file."
        )
    return f"{start} Convert it first: {call}({_conversion(data)})"


def _describe_data(data: object) -> str:
    """What data is, for the message: 'a pandas Series', 'None', 'a str'."""
    if data is None:
        return "None"
    kind = type(data)
    known = _CONVERSIONS.get((kind.__module__.split(".")[0], kind.__name__))
    return known[0] if known is not None else a_or_an(kind.__name__)


# What a DataFrame replacement is called and the conversion to a DataFrame,
# by library and class name.
_CONVERSIONS = {
    ("pandas", "Series"): ("a pandas Series", "series.to_frame()"),
    ("numpy", "ndarray"): ("a NumPy array", "pd.DataFrame(array)"),
    ("polars", "DataFrame"): ("a polars DataFrame", "df.to_pandas()"),
    ("polars", "LazyFrame"): ("a polars LazyFrame", "lf.collect().to_pandas()"),
    ("polars", "Series"): ("a polars Series", "s.to_frame().to_pandas()"),
    ("pyarrow", "Table"): ("a pyarrow Table", "table.to_pandas()"),
    ("pyarrow", "RecordBatch"): ("a pyarrow RecordBatch", "batch.to_pandas()"),
}


def _conversion(data: object) -> str:
    """The one-line conversion of data to a DataFrame."""
    kind = type(data)
    known = _CONVERSIONS.get((kind.__module__.split(".")[0], kind.__name__))
    return known[1] if known is not None else "pd.DataFrame(data)"


def validate_json(
    spec: AnalysisSpec, data: pd.DataFrame, options: dict[str, Any]
) -> dict[str, Any]:
    """Check the arguments of an MCP call the way their input schema reads them.

    The options are JSON values, so they are validated in pydantic's JSON mode:
    an ISO 8601 string becomes a datetime, date or timedelta wherever the
    annotation allows one and no other type matches the string better. Raises
    the same InvalidArgumentError a Python call with a bad option does. Returns
    the validated value of each option passed, not the defaults of the others.
    """
    validated = _validate(spec, data, options, json_mode=True)
    return {name: validated[name] for name in options}


def _validate(
    spec: AnalysisSpec,
    data: pd.DataFrame,
    options: dict[str, Any],
    *,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Check the keyword options against their annotations, defaults included."""
    given = options
    options = {
        name: _normalize(spec, data, name, value) for name, value in options.items()
    }
    # Options given one value that became a list of one, as quantiles=0.5 does.
    single = frozenset(
        name
        for name, value in given.items()
        if isinstance(options[name], list) and _is_single(value)
    )
    try:
        if json_mode:
            validated = spec.arguments.model_validate_json(json.dumps(options))
        else:
            validated = spec.arguments.model_validate(options)
    except pydantic.ValidationError as error:
        problems = error.errors()
        # The package's own error about a whole option, such as a Project
        # given as a dict, keeps its code: invalid_call for an unknown field.
        for problem in problems:
            own = _own_error(problem)
            if own is not None and len(problem["loc"]) == 1:
                raise own from None
        raise worded(
            InvalidArgumentError,
            lambda surface: invalid_message(
                spec.name,
                spec.annotations,
                options,
                problems,
                single=single,
                surface=surface,
            ),
        ) from None
    return {name: getattr(validated, name) for name in spec.parameters[1:]}


def checked_keywords[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Make a call with a keyword the function doesn't take raise InvalidArgumentError.

    Python raises a bare TypeError for it. The error has the code invalid_call and
    names the closest keyword, as an analysis's does. The check reads the
    function's signature, and the function keeps its name, docstring and
    signature. A method's own `self` or `cls` isn't a keyword.
    """
    owner = getattr(function, "__qualname__", "the function")
    names = [
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.name not in ("self", "cls")
    ]

    @functools.wraps(function)
    def checked(*args: P.args, **kwargs: P.kwargs) -> R:
        for key in kwargs:
            if key not in names:
                raise InvalidArgumentError(
                    unknown_name_message(owner, "argument", key, names)
                    if names
                    else f"{owner} takes no arguments, but got {key!r}. Call it as "
                    f"{owner}().",
                    code="invalid_call",
                )
        return function(*args, **kwargs)

    return checked


def _is_single(value: object) -> bool:
    """Whether _as_list makes a list of one of the value."""
    return value is not None and (
        isinstance(value, str | bytes | np.generic) or not isinstance(value, Iterable)
    )


def unknown_name_message(
    owner: str, noun: str, name: str, allowed: Sequence[str]
) -> str:
    """Say that a keyword isn't one of `owner`'s, naming the closest and the rest.

    A project or mask_sensitive passed to an analysis that has none says why: it
    shows no raw values, so there is nothing to mask.
    """
    listed = f"Allowed: {', '.join(allowed)}."
    takes_masking = bool(_MASKING_ARGUMENTS & set(allowed))
    if (
        noun == "argument"
        and name in _MASKING_ARGUMENTS
        and owner in _REGISTRY
        and not takes_masking
    ):
        return (
            f"{owner} takes no {name}: it shows no raw values, so there is nothing "
            f"to mask. Leave {name} out. {listed}"
        )
    closest = closest_names(name, allowed)[:1]
    hint = f" Did you mean {closest[0]!r}?" if closest else ""
    return f"{owner} has no {noun} {name!r}.{hint} {listed}"


def invalid_message(
    owner: str,
    annotations: Mapping[str, object],
    options: Mapping[str, Any],
    problems: Sequence[Any],
    *,
    noun: str = "argument",
    single: Collection[str] = (),
    surface: Surface = "python",
) -> str:
    """Say what is wrong with an option, what it allows and how to fix it.

    `problems` are pydantic's errors for the options; the first one is shown,
    unless one comes from a check of the package's own, such as ColumnSpec's,
    whose message says it best. The message says what the option allows in
    words, never in pydantic's or Python's typing syntax. The options in
    `single` were given one value, which became a list of one: its message is
    about the option, not about the list's first item.
    """
    problem = next((p for p in problems if _own_error(p) is not None), problems[0])
    location = problem["loc"]
    name = str(location[0]) if location else "?"
    annotation = annotations.get(name)
    none = "None" if surface == "python" else "null"
    value, steps, inner = _locate(options.get(name), annotation, location[1:])
    path = _path_text(steps, surface)
    if name in single and steps == [0]:
        path = ""
    own = _own_error(problem)
    if own is not None:
        # The package's own check, such as ColumnSpec's, says it best.
        own_message = message(own, surface)
        return (
            f"{owner} got an invalid value at {name}{path}: {own_message}"
            if path
            else own_message
        )
    if path:
        # A value inside a list option, or inside a mapping or its lists.
        where = f"at {name}{path}"
        rule = f"each value in {name} must be {allowed_text(inner, 'value', none=none)}"
    else:
        where = f"for {name}"
        rule = f"{name} must be {allowed_text(annotation, name, none=none)}"
    detail = ""
    if problem["type"] == "literal_error":
        # The choices of the Literal, alone or in a union such as a logical type
        # or a ColumnSpec.
        choices = [
            str(choice)
            for member in members(strip_optional(inner))
            if typing.get_origin(member) is typing.Literal
            for choice in typing.get_args(member)
        ]
        closest = closest_names(value, choices)[:1] if isinstance(value, str) else []
        detail = f" Did you mean {closest[0]!r}?" if closest else ""
    if name == "project" and isinstance(value, str | os.PathLike):
        rule, detail = (
            "project must be a Project, not a path",
            _load_fix(value, surface),
        )
    return (
        f"{owner} got {_value_text(value, surface)} {where}, but {rule}.{detail} "
        f"{see_every(owner, noun, surface)}"
    )


def _load_fix(path: str | os.PathLike[str], surface: Surface) -> str:
    """The fix for a path given where a Project belongs: load the file."""
    if surface == "mcp":
        return " Pass the path of the project file as project_path instead."
    text = os.fspath(path)
    shown = repr(text) if len(text) <= 80 else "path"
    return f" Load the file first, as project=ae.Project.load({shown})."


def _own_error(problem: Any) -> AgentEDAError | None:
    """The package's own error behind a pydantic error, if there is one."""
    error = problem.get("ctx", {}).get("error")
    return error if isinstance(error, AgentEDAError) else None


def _locate(
    value: object, annotation: object, steps: tuple[object, ...]
) -> tuple[object, list[object], object]:
    """The value at a validation error's location inside an option.

    Follows list positions and mapping keys as far as they lead and returns the
    value found, the steps taken (such as ["age", 1], empty for the option
    itself) and its annotation.
    """
    taken: list[object] = []
    for step in steps:
        if isinstance(value, list) and isinstance(step, int) and step < len(value):
            value = value[step]
            annotation = item_annotation(annotation)
        elif isinstance(value, dict) and step in value:
            value = value[step]
            annotation = _mapping_values(annotation) or annotation
        else:
            break  # such as the name of a union's member
        taken.append(step)
    return value, taken, annotation


def _path_text(steps: Sequence[object], surface: Surface) -> str:
    """The steps into an option as the caller writes them: ['age'][1] in Python,
    and over MCP /age/1, a JSON pointer, since JSON has no subscripts."""
    if surface == "python":
        return "".join(f"[{step!r}]" for step in steps)
    return "".join(f"/{step}" for step in steps)


def _normalize(
    spec: AnalysisSpec, data: pd.DataFrame, name: str, value: object
) -> object:
    """Turn what agents commonly pass into the plain Python an option declares.

    A list option takes any ordered list-like, such as a tuple, a pandas Index
    or Series, a 1-D NumPy array or a generator, and a single value as a
    one-item list. NumPy scalars become Python scalars, and NumPy datetimes and
    timedeltas pandas ones. A column option, such as columns or id_column, also
    takes labels that aren't strings and names them by their string form, the
    way results show them. A mapping option is keyed by column names, which
    follow the same rule, and each of its values is read like an option of the
    value's type. A column option refuses a Series, an Index or an array that
    holds a column's values in place of names. Validation then stays strict
    about everything else.
    """
    annotation = spec.annotations[name]
    values = _mapping_values(annotation)
    if values is not None:
        return _normalize_mapping(spec, name, value, values)
    if not _takes_list(annotation):
        value = python_scalar(value)
        if name.endswith("column") and value is not None:
            if isinstance(value, pd.Series | pd.Index | np.ndarray):
                raise _values_as_names(spec, data, name, value)
            return _column_name(value)
        return value
    items = _as_list(spec, name, value)
    if (
        isinstance(value, pd.Series | pd.Index | np.ndarray)
        and isinstance(items, list)
        and name.endswith("columns")
        and not _are_labels(data, items)
    ):
        raise _values_as_names(spec, data, name, value)
    if isinstance(items, list) and name.endswith("columns"):
        return [_column_name(item) for item in items]
    return items


def _are_labels(data: pd.DataFrame, items: list[object]) -> bool:
    """Whether every item names a column of the data, by label or string form."""
    labels = {str(label) for label in data.columns}
    for item in items:
        try:
            if str(_column_name(item)) not in labels:
                return False
        except TypeError:  # an unhashable item, such as a list
            return False
    return True


def _values_as_names(
    spec: AnalysisSpec, data: pd.DataFrame, name: str, value: object
) -> InvalidArgumentError:
    """The error for a column's values passed where its name or names belong.

    It describes the value by its type and size, never by its items: they may
    be values of a sensitive column.
    """
    # The fix shows a real column name: the first one that is short enough.
    example = next(
        (c for c in map(str, data.columns) if len(c) <= MAX_NAME_LENGTH), "email"
    )
    if name.endswith("columns"):
        fix = f"Pass the names of columns, such as {name}=[{example!r}]."
    else:
        fix = f"Pass the name of a column, such as {name}={example!r}."
    return InvalidArgumentError(
        f"{spec.name} got {_value_text(value)} for {name}, which looks like a "
        f"column's values, not column names. {fix}"
    )


def _takes_list(annotation: object) -> bool:
    return any(
        typing.get_origin(member) in LIST_TYPES for member in members(annotation)
    )


def _mapping_values(annotation: object) -> object:
    """The annotation of a mapping option's values, or None if it takes no mapping."""
    for member in members(annotation):
        if typing.get_origin(member) in MAPPING_TYPES:
            return typing.get_args(member)[1]
    return None


def _normalize_mapping(
    spec: AnalysisSpec, name: str, value: object, values: object
) -> object:
    """A mapping keyed by column names, each value read as `values` declares it."""
    if not isinstance(value, Mapping):
        return value  # None, or something validation rejects, naming its type
    normalized: dict[object, object] = {}
    given: dict[object, object] = {}
    for key, item in value.items():
        column = _column_name(key)
        if column in normalized:
            raise InvalidArgumentError(
                f"{spec.name} got column {column!r} more than once in {name}, as "
                f"{given[column]!r} and {key!r}. Name each column once."
            )
        given[column] = key
        normalized[column] = (
            _as_list(spec, name, item, key=column)
            if _takes_list(values)
            else python_scalar(item)
        )
    return normalized


def _as_list(
    spec: AnalysisSpec, name: str, value: object, *, key: object = None
) -> object:
    """The value as a list, or as it is when validation should reject it.

    `key` is the key of a mapping option whose value this is, if it is one.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        return [python_scalar(value)]
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return [value]
    if isinstance(value, set | frozenset):
        where = name if key is None else f"{name}[{key!r}]"
        instead = "a list or tuple" if key is None else "a list or tuple for that key"
        fix = f"{name}=sorted(...)" if key is None else f"{{{key!r}: sorted(...)}}"
        raise InvalidArgumentError(
            f"{spec.name} got a set for {where}, and a set has no fixed order. "
            f"Pass {instead} instead, such as {fix}."
        )
    if isinstance(value, pd.DataFrame | Mapping) or (
        isinstance(value, np.ndarray) and value.ndim != 1
    ):
        return value  # validation rejects it, naming its type
    if isinstance(value, pd.Index | pd.Series):
        return value.tolist()  # pandas gives Timestamps for datetimes
    return [python_scalar(item) for item in value]


def python_scalar(value: object) -> object:
    """A NumPy scalar as Python's or pandas' own; anything else as it is.

    NumPy datetimes and timedeltas become pandas ones: their item() is an int
    for nanoseconds.
    """
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value)
    return value.item() if isinstance(value, np.generic) else value


def _column_name(item: object) -> object:
    """A column label as the string form results show, if it is a label at all."""
    if isinstance(item, str) or not isinstance(item, Hashable):
        return item
    return str(item)


def _value_text(value: object, surface: Surface = "python") -> str:
    """Show a scalar argument; describe anything larger by its type and size.

    A container passed by mistake, such as a column of a DataFrame, would
    otherwise put raw data values into the message. Over MCP null and the
    booleans show as JSON writes them, and a mapping is an object.
    """
    if value is None or isinstance(value, bool | int | float | str | np.generic):
        if surface == "mcp" and (value is None or isinstance(value, bool | np.bool_)):
            return echoed(value, surface)
        return short_repr(value)
    if isinstance(value, pd.DataFrame):
        rows = plural(value.shape[0], "row", "rows")
        return f"a pandas DataFrame with {rows} and {plural(value.shape[1], 'column', 'columns')}"
    if isinstance(value, pd.Series | pd.Index):
        return f"a pandas {type(value).__name__} of {plural(len(value), 'value', 'values')}"
    if isinstance(value, np.ndarray):
        return f"a NumPy array of shape {value.shape}"
    if isinstance(value, list | tuple | set | frozenset | dict):
        kind = (
            "an object"
            if surface == "mcp" and isinstance(value, dict)
            else a_or_an(type(value).__name__)
        )
        return f"{kind} of {plural(len(value), 'item', 'items')}"
    if isinstance(value, type):
        return f"the type {value.__name__}"
    return a_or_an(type(value).__name__)


def _absorb_warnings(result: object, caught: list[warnings.WarningMessage]) -> object:
    """Keep the warnings an analysis raised away from its caller.

    Numerical RuntimeWarnings that the analysis didn't tie to the values they
    affect become one limitation. Every other warning, DeprecationWarning and
    FutureWarning among them, is raised again under the caller's own filters
    but shown to no one: only a caller whose filters turn it into an error, as
    the test suite's do, gets it, as that error. A recorded warning keeps its
    file but not its module, so a filter that names a module matches against
    the file's path.
    """
    numerical: list[str] = []
    others: dict[tuple[type[Warning], str, str, int], warnings.WarningMessage] = {}
    for warning in caught:
        if issubclass(warning.category, RuntimeWarning):
            if str(warning.message) not in numerical:
                numerical.append(str(warning.message))
        else:
            key = (
                warning.category,
                str(warning.message),
                warning.filename,
                warning.lineno,
            )
            others.setdefault(key, warning)
    for warning in others.values():
        with capture_warnings():
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    if not numerical or not isinstance(result, AnalysisResult):
        return result
    limitation = Limitation(
        code="numerical_warning",
        reason=(
            "Numerical warnings were raised that can't be tied to particular "
            f"values: {'; '.join(numerical)}."
        ),
        fields=[],
    )
    return result.model_copy(update={"limitations": [*result.limitations, limitation]})


def _check_result(spec: AnalysisSpec, result: object) -> None:
    """Raise if an analysis returned something it didn't declare: a bug."""
    problems = []
    if not isinstance(result, spec.result_class):
        problems.append(f"returned {type(result).__name__}")
    else:
        if result.analysis != spec.name:
            problems.append(f"named itself {result.analysis!r}")
        if result.status not in spec.statuses:
            problems.append(f"returned undeclared status {result.status!r}")
        reason = result.status_reason
        if reason is not None and reason.code not in spec.reason_codes(result.status):
            problems.append(
                f"returned undeclared reason code {reason.code!r} with status "
                f"{result.status!r}"
            )
        undeclared = [
            f.code for f in result.findings if f.code not in spec.finding_codes
        ]
        if undeclared:
            problems.append(f"reported undeclared finding codes {undeclared}")
        if list(result.parameters) != list(spec.parameters):
            problems.append(f"recorded parameters {list(result.parameters)}")
    if problems:
        raise RuntimeError(f"agent_eda bug: {spec.name} " + "; ".join(problems))
