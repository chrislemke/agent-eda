"""The registry of analyses and the decorator that registers each one.

The decorator records an analysis's metadata and wraps it with the checks every
analysis shares. The call's shape, the type of `data`, its column labels and the
keyword arguments are checked before any computation starts. Warnings raised
while the analysis runs never reach the caller, and the result is checked
against what the analysis declared.
"""

import difflib
import functools
import importlib
import inspect
import re
import reprlib
import types
import typing
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from types import FunctionType
from typing import Any, cast

import numpy as np
import pandas as pd
import pydantic

from agent_eda._annotations import (
    LIST_TYPES,
    bounds,
    item_annotation,
    members,
    strip_optional,
)
from agent_eda._columns import check_column_labels, closest_names
from agent_eda._docstring import Docstring
from agent_eda._docstring import parse as parse_docstring
from agent_eda._run import watch_all_warnings
from agent_eda._text import join_alternatives, plural
from agent_eda.errors import InvalidArgumentError
from agent_eda.results import AnalysisResult, Limitation, Status

_CODE = re.compile(r"[a-z][a-z0-9_]*")


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


_REGISTRY: dict[str, AnalysisSpec] = {}


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
        masking: Whether it supports masking sensitive columns.
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
            validated = _validate(spec, options)
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
    options = {
        p.name: (hints[p.name], p.default)
        for p in list(signature.parameters.values())[1:]
    }
    arguments = pydantic.create_model(
        f"{result_class.__name__.removesuffix('Result')}Arguments",
        __config__=pydantic.ConfigDict(
            strict=True, extra="forbid", arbitrary_types_allowed=True
        ),
        **cast("dict[str, Any]", options),
    )
    docstring = parse_docstring(doc)
    return AnalysisSpec(
        name=name,
        module=func.__module__,
        summary=docstring.summary,
        docstring=docstring,
        tags=tags,
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
        closest = difflib.get_close_matches(unknown[0], spec.parameters, n=1)
        hint = f" Did you mean {closest[0]!r}?" if closest else ""
        raise InvalidArgumentError(
            f"{spec.name} has no argument {unknown[0]!r}.{hint} "
            f"Allowed: {', '.join(spec.parameters)}.",
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
    data = args[0] if args else kwargs["data"]
    return data, {key: value for key, value in kwargs.items() if key != "data"}


def _check_data(name: str, data: object) -> None:
    """Raise unless data is a pandas DataFrame, showing the conversion."""
    if isinstance(data, pd.DataFrame):
        return
    got, conversion = _conversion(data)
    raise InvalidArgumentError(
        f"data must be a pandas DataFrame, but got {got}. Convert it first: "
        f"ae.{name}({conversion})",
        code="unsupported_data_type",
    )


def _conversion(data: object) -> tuple[str, str]:
    """Describe what data is and the one-line conversion to a DataFrame."""
    kind = type(data)
    library = kind.__module__.split(".")[0]
    known = {
        ("pandas", "Series"): ("a pandas Series", "series.to_frame()"),
        ("numpy", "ndarray"): ("a NumPy array", "pd.DataFrame(array)"),
        ("polars", "DataFrame"): ("a polars DataFrame", "df.to_pandas()"),
        ("polars", "LazyFrame"): ("a polars LazyFrame", "lf.collect().to_pandas()"),
        ("polars", "Series"): ("a polars Series", "s.to_frame().to_pandas()"),
        ("pyarrow", "Table"): ("a pyarrow Table", "table.to_pandas()"),
    }
    if (library, kind.__name__) in known:
        return known[(library, kind.__name__)]
    if data is None:
        return "None", "pd.DataFrame(data)"
    return f"a {kind.__name__}", "pd.DataFrame(data)"


def _validate(spec: AnalysisSpec, options: dict[str, Any]) -> dict[str, Any]:
    """Check the keyword options against their annotations, defaults included."""
    options = {name: _normalize(spec, name, value) for name, value in options.items()}
    try:
        validated = spec.arguments.model_validate(options)
    except pydantic.ValidationError as error:
        raise InvalidArgumentError(
            _invalid_message(spec, options, error.errors()[0])
        ) from None
    return {name: getattr(validated, name) for name in spec.parameters[1:]}


def _invalid_message(spec: AnalysisSpec, options: dict[str, Any], problem: Any) -> str:
    """Say what is wrong with an option, what it allows and how to fix it."""
    location = problem["loc"]
    name = str(location[0]) if location else "?"
    annotation = spec.annotations.get(name)
    value = options.get(name)
    if len(location) > 1 and isinstance(location[1], int) and isinstance(value, list):
        # A value inside a list option.
        value = value[location[1]]
        where = f"at {name}[{location[1]}]"
        each = _allowed(item_annotation(annotation), "value")
        rule = f"each value in {name} must be {each}"
    else:
        where = f"for {name}"
        rule = f"{name} must be {_allowed(annotation, name)}"
    if problem["type"] == "literal_error":
        choices = [str(c) for c in typing.get_args(strip_optional(annotation))]
        closest = closest_names(value, choices)[:1] if isinstance(value, str) else []
        detail = f" Did you mean {closest[0]!r}?" if closest else ""
    elif problem["type"] in _OUT_OF_RANGE:
        detail = ""
    else:
        detail = f" {problem['msg']}."
    return (
        f"{spec.name} got {_value_text(value)} {where}, but {rule}.{detail} "
        f"See help(ae.{spec.name}) for every argument."
    )


# pydantic's error types for a value outside a declared range; the message
# states the range itself.
_OUT_OF_RANGE = {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}


def _allowed(annotation: object, subject: str) -> str:
    """What an option allows, such as "float with 0 <= trim_fraction < 0.5"."""
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        choices = [repr(choice) for choice in typing.get_args(annotation)]
        return f"one of {join_alternatives(choices)}"
    if origin is typing.Annotated:
        inner, *metadata = typing.get_args(annotation)
        limits = bounds(metadata, subject)
        return f"{_allowed(inner, subject)} with {limits}" if limits else _plain(inner)
    return _plain(annotation)


def _plain(annotation: object) -> str:
    """A type as an agent reads it: 'Sequence[str] | None', without Annotated."""
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is typing.Annotated:
        return _plain(arguments[0])
    if origin in (typing.Union, types.UnionType):
        return " | ".join(_plain(argument) for argument in arguments)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(argument) for argument in arguments)}]"
    if origin is not None:
        inner = ", ".join(_plain(argument) for argument in arguments)
        return f"{getattr(origin, '__name__', str(origin))}[{inner}]"
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", str(annotation))


def _normalize(spec: AnalysisSpec, name: str, value: object) -> object:
    """Turn what agents commonly pass into the plain Python an option declares.

    A list option takes any ordered list-like, such as a tuple, a pandas Index
    or Series, a 1-D NumPy array or a generator, and a single value as a
    one-item list. NumPy scalars become Python scalars. A column option, such
    as columns or id_column, also takes labels that aren't strings and names
    them by their string form, the way results show them. Validation then stays
    strict about everything else.
    """
    if not any(
        typing.get_origin(member) in LIST_TYPES
        for member in members(spec.annotations[name])
    ):
        value = _python_scalar(value)
        if name.endswith("column") and value is not None:
            return _column_name(value)
        return value
    items = _as_list(spec, name, value)
    if isinstance(items, list) and name.endswith("columns"):
        return [_column_name(item) for item in items]
    return items


def _as_list(spec: AnalysisSpec, name: str, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return [value.item()]
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return [value]
    if isinstance(value, set | frozenset):
        raise InvalidArgumentError(
            f"{spec.name} got a set for {name}, and a set has no fixed order. "
            f"Pass a list or tuple instead, such as {name}=sorted(...)."
        )
    if isinstance(value, pd.DataFrame | Mapping) or (
        isinstance(value, np.ndarray) and value.ndim != 1
    ):
        return value  # validation rejects it, naming its type
    if isinstance(value, pd.Index | pd.Series | np.ndarray):
        return value.tolist()
    return [_python_scalar(item) for item in value]


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _column_name(item: object) -> object:
    """A column label as the string form results show, if it is a label at all."""
    if isinstance(item, str) or not isinstance(item, Hashable):
        return item
    return str(item)


_SCALAR_REPR = reprlib.Repr(maxstring=40, maxother=40)


def _value_text(value: object) -> str:
    """Show a scalar argument; describe anything larger by its type and size.

    A container passed by mistake, such as a column of a DataFrame, would
    otherwise put raw data values into the message.
    """
    if value is None or isinstance(value, bool | int | float | str | np.generic):
        return _SCALAR_REPR.repr(value)
    if isinstance(value, pd.DataFrame):
        rows = plural(value.shape[0], "row", "rows")
        return f"a pandas DataFrame with {rows} and {plural(value.shape[1], 'column', 'columns')}"
    if isinstance(value, pd.Series | pd.Index):
        return f"a pandas {type(value).__name__} of {plural(len(value), 'value', 'values')}"
    if isinstance(value, np.ndarray):
        return f"a NumPy array of shape {value.shape}"
    if isinstance(value, list | tuple | set | frozenset | dict):
        return f"a {type(value).__name__} of {plural(len(value), 'item', 'items')}"
    return f"a {type(value).__name__}"


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
        with warnings.catch_warnings(record=True):
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
        undeclared = [
            f.code for f in result.findings if f.code not in spec.finding_codes
        ]
        if undeclared:
            problems.append(f"reported undeclared finding codes {undeclared}")
        if list(result.parameters) != list(spec.parameters):
            problems.append(f"recorded parameters {list(result.parameters)}")
    if problems:
        raise RuntimeError(f"agent_eda bug: {spec.name} " + "; ".join(problems))
