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
import re
import typing
import warnings
from collections.abc import (
    Callable,
    Generator,
    Iterable,
)
from dataclasses import dataclass
from types import FunctionType
from typing import Any, cast

import pandas as pd
import pydantic

from agent_eda._arguments import (
    check_data,
    invalid_message,
    is_single,
    missing_message,
    normalize_option,
    own_error,
    unknown_name_message,
)
from agent_eda._columns import check_column_labels
from agent_eda._docstring import Docstring
from agent_eda._docstring import parse as parse_docstring
from agent_eda._envelope import DATA_VALUED, Call, data_shape, recording
from agent_eda._result_base import AnalysisResult, Status, StatusReasonCode
from agent_eda._warnings import absorb_warnings, watch_all_warnings
from agent_eda._wording import (
    SeeArguments,
    mistake,
)
from agent_eda.errors import InvalidArgumentError

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
            check_data(spec.name, data)
            if not spec.any_column_labels:
                check_column_labels(spec.name, data.columns)
            model = _validated(spec, data, options)
            call = Call(
                analysis=spec.name,
                method=_method_of(model),
                data=data_shape(data),
                arguments=model,
                data_valued=data_valued(spec.parameters)
                if spec.supports_masking
                else (),
            )
            with recording(call), watch_all_warnings() as caught:
                result = func(
                    data, **{name: getattr(model, name) for name in spec.parameters[1:]}
                )
            result = absorb_warnings(result, caught)
            _check_result(spec, result)
            return result

        # pytest would otherwise collect analyses named test_*.
        checked.__dict__["__test__"] = False
        return cast("F", checked)

    return register


def data_valued(parameters: Iterable[str]) -> tuple[str, ...]:
    """The given parameters that hold data values, in the order given."""
    return tuple(name for name in parameters if name in DATA_VALUED)


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
            unknown_name_message(
                spec.name, "argument", unknown[0], spec.parameters, of_analysis=True
            ),
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
        spec.name,
        spec.annotations,
        missing,
        fix=f"by keyword, such as ae.{spec.name}(df, {example})",
        see=SeeArguments(spec.name).render("python"),
    )


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
    validated = _validated(spec, data, options, json_mode=json_mode)
    return {name: getattr(validated, name) for name in spec.parameters[1:]}


def _validated(
    spec: AnalysisSpec,
    data: pd.DataFrame,
    options: dict[str, Any],
    *,
    json_mode: bool = False,
) -> pydantic.BaseModel:
    """The arguments model holding the options, checked, defaults included."""
    given = options
    options = {
        name: normalize_option(spec.name, spec.annotations, data, name, value)
        for name, value in options.items()
    }
    # Options given one value that became a list of one, as quantiles=0.5 does.
    single = frozenset(
        name
        for name, value in given.items()
        if isinstance(options[name], list) and is_single(value)
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
            own = own_error(problem)
            if own is not None and len(problem["loc"]) == 1:
                raise own from None
        raise mistake(
            InvalidArgumentError,
            invalid_message(
                spec.name, spec.annotations, options, problems, single=single
            ),
        ) from None
    return validated


def _method_of(model: pydantic.BaseModel) -> str | None:
    """The method the call chose, for an analysis whose `method` option picks one."""
    method = getattr(model, "method", None)
    return method if isinstance(method, str) else None


def arguments_model(analysis: str) -> type[pydantic.BaseModel]:
    """The arguments model of a registered analysis, importing its module if need be."""
    import agent_eda

    module = agent_eda._EXPORTS.get(analysis)
    if module is not None:
        importlib.import_module(module)
    if analysis not in _REGISTRY:
        raise KeyError(f"{analysis!r} is not a registered analysis")
    return _REGISTRY[analysis].arguments


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
