"""The registry of analyses and the decorator that registers each one.

The decorator records an analysis's metadata and wraps it with the checks every
analysis shares: the call's shape, the type of `data` and the keyword arguments
are checked before any computation starts, and the result is checked against
what the analysis declared.
"""

import difflib
import functools
import importlib
import inspect
import re
import reprlib
import typing
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any, cast

import numpy as np
import pandas as pd
import pydantic

from agent_eda._text import plural
from agent_eda.errors import InvalidArgumentError
from agent_eda.results import AnalysisResult, Status

_CODE = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class AnalysisSpec:
    """What the registration decorator records about one analysis."""

    name: str
    module: str
    summary: str
    tags: tuple[str, ...]
    signature: inspect.Signature
    methods: tuple[str, ...]
    statuses: tuple[Status, ...]
    finding_codes: tuple[str, ...]
    result_class: type[AnalysisResult]
    supports_masking: bool
    work_limits: tuple[str, ...]
    arguments: type[pydantic.BaseModel]

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
) -> Callable[[F], F]:
    """Register an analysis and wrap it with the shared checks.

    Args:
        statuses: Every status the analysis can return.
        finding_codes: Every finding code the analysis can report.
        tags: Words that help search find the analysis.
        methods: The values its `method` argument accepts, if it has one.
        masking: Whether it supports masking sensitive columns.
        work_limits: Its `max_*` parameters that bound the work it does.
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
            result = func(data, **_validate(spec, options))
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
) -> AnalysisSpec:
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        raise TypeError(f"analysis {name!r} has no docstring")
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    names = list(signature.parameters)
    if not names or names[0] != "data":
        raise TypeError(f"analysis {name!r} must take data as its first parameter")
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
    return AnalysisSpec(
        name=name,
        module=func.__module__,
        summary=doc.splitlines()[0].strip(),
        tags=tags,
        signature=signature,
        methods=methods,
        statuses=statuses,
        finding_codes=finding_codes,
        result_class=result_class,
        supports_masking=masking,
        work_limits=work_limits,
        arguments=arguments,
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
    try:
        validated = spec.arguments.model_validate(options)
    except pydantic.ValidationError as error:
        problem = error.errors()[0]
        name = str(problem["loc"][0]) if problem["loc"] else "?"
        allowed = _type_text(spec.arguments.model_fields[name].annotation)
        raise InvalidArgumentError(
            f"{spec.name} got {_value_text(options.get(name))} for {name}, but "
            f"{name} must be {allowed}. {problem['msg']}. See "
            f"help(ae.{spec.name}) for every argument.",
        ) from None
    return {name: getattr(validated, name) for name in spec.parameters[1:]}


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


def _type_text(annotation: object) -> str:
    return inspect.formatannotation(annotation).replace("typing.", "")
