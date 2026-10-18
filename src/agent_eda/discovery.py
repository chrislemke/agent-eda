"""Functions that help an agent find the right analysis and call it correctly.

guide() prints where to start, search_analyses() finds analyses for a question,
describe_analysis() tells how to call one and list_analyses() lists them all.
They read the registry and each analysis's docstring, so there is no second copy
to keep in sync, and none of them imports SciPy, statsmodels or scikit-learn.
"""

import inspect
import numbers
import typing

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from agent_eda._annotations import choices, python_literal, type_text, valid_range
from agent_eda._columns import closest_names
from agent_eda._registry import AnalysisSpec, registered
from agent_eda._search import rank, terms
from agent_eda._text import join_alternatives, join_items, plural
from agent_eda._version import __version__
from agent_eda.errors import InvalidArgumentError, UnknownAnalysisError
from agent_eda.results import (
    AnalysisDescription,
    AnalysisInfo,
    AnalysisListing,
    AnalysisResult,
    FindingCodeDescription,
    ParameterDescription,
    ResultFieldDescription,
    SearchListing,
    SearchMatch,
    SearchTerm,
    StatusDescription,
)

# The order for a first look at a new dataset: what each step covers and the
# analyses that cover it. The guide shows the analyses registered so far.
FIRST_LOOK: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("structure", ("describe_dataset",)),
    ("logical types", ("infer_column_semantics",)),
    ("missing values", ("profile_missing_values",)),
    ("stand-ins for missing values", ("detect_missing_sentinels",)),
    ("distributions", ("summarize_numeric", "summarize_categorical")),
    ("duplicates and keys", ("find_duplicate_rows", "find_candidate_keys")),
    ("the target", ("profile_target",)),
    ("associations", ("compute_association_matrix",)),
)

# The guide's fixed text; the analyses and the first-look order come from the
# registry. Lines stay within 120 characters, the whole guide within 60 lines.
_GUIDE_INTRO = """\
agent-eda {version}: deterministic exploratory data analysis (EDA) of pandas DataFrames. Each analysis is a plain
function that answers one EDA question and returns a typed result; the same call always gives the same result."""

_GUIDE_CALL = """\
Call an analysis with the DataFrame first and every other argument by keyword:
    import agent_eda as ae
    res = ae.summarize_numeric(df, columns=["age", "income"])
    print(res)
columns=None, the default, selects every column the analysis can use. A mistake in a call raises an
ae.errors.AgentEDAError whose message says how to fix it."""

_GUIDE_RESULTS = """\
Read the result:
- print(res) shows a short summary: status, key numbers, findings and limitations. Always print it: Code
  Interpreter returns only what you print.
- res.status says how the run ended: completed, partial, insufficient_data, not_applicable, missing_prerequisites,
  budget_exceeded, blocked_by_policy or failed; res.status_reason says why it didn't complete.
- res.findings are claims about the data, each with a code and an assessment: descriptive, suspected, violation,
  passed or unknown. res.limitations say what limits how far the numbers can be read.
- res.to_frame() returns the complete table; res.to_dict() and res.to_json() return the whole record."""

_GUIDE_MORE = """\
Find more:
  ae.search_analyses("nulls per column")  finds analyses for a question in your own words
  ae.describe_analysis("summarize_numeric")  its parameters, statuses, result fields and an example
  ae.list_analyses()  every analysis with what it declares; .to_frame() gives it as a table
  help(ae.summarize_numeric)  the full docstring"""


def guide() -> None:
    """Print one page on agent-eda: what it does, how to call and read analyses.

    The page covers what the package is, the call pattern, how to read a result
    and its status, every analysis grouped by module with the question it
    answers, and an order for a first look at a new dataset. It stays within
    about 60 lines of at most 120 characters. Start here when you are new to the
    package; then find an analysis with search_analyses() and learn how to call
    it with describe_analysis().

    Returns:
        None: The page is printed, since Code Interpreter returns only what is
        printed.

    Examples:
        >>> import agent_eda as ae
        >>> ae.guide()  # doctest: +ELLIPSIS
        agent-eda ...: deterministic exploratory data analysis (EDA) of pandas DataFrames. ...
        ...
        A first look at a new dataset, in this order:
          1. describe_dataset: structure
        ...
    """
    print(_guide_text())


def _guide_text() -> str:
    listing = list_analyses()
    registered = {info.name for info in listing.analyses}
    steps = [
        (covers, [name for name in names if name in registered])
        for covers, names in FIRST_LOOK
    ]
    first_look = [
        f"  {number}. {' and '.join(names)}: {covers}"
        for number, (covers, names) in enumerate(
            [(covers, names) for covers, names in steps if names], start=1
        )
    ]
    return "\n".join(
        [
            _GUIDE_INTRO.format(version=__version__),
            "",
            _GUIDE_CALL,
            "",
            _GUIDE_RESULTS,
            "",
            str(listing),
            "",
            "A first look at a new dataset, in this order:",
            *first_look,
            "",
            _GUIDE_MORE,
        ]
    )


def list_analyses() -> AnalysisListing:
    """List every analysis with its module, one-line summary and masking support.

    Printing the listing shows one line per analysis, "name: summary", grouped
    under its module; the summary is the question the analysis answers. The
    `analyses` field holds what each analysis declares (parameters, methods,
    statuses, finding codes, result class and work limits), and `to_frame()`
    returns the same as a DataFrame. Listing never imports SciPy, statsmodels or
    scikit-learn.

    Returns:
        AnalysisListing: The analyses, ordered by module, then name.

    Examples:
        >>> import agent_eda as ae
        >>> listing = ae.list_analyses()
        >>> [info] = [i for i in listing.analyses if i.name == "describe_dataset"]
        >>> print(info.module, info.summary)
        agent_eda.data.provenance What does this dataset contain?
    """
    return AnalysisListing(
        analyses=[
            AnalysisInfo(
                name=spec.name,
                module=spec.module,
                summary=spec.summary,
                supports_masking=spec.supports_masking,
                tags=list(spec.tags),
                parameters=list(spec.parameters),
                methods=list(spec.methods),
                statuses=list(spec.statuses),
                finding_codes=list(spec.finding_codes),
                work_limit_parameters=list(spec.work_limits),
                result_class=spec.result_class.__name__,
            )
            for spec in registered()
        ]
    )


def describe_analysis(name: str) -> AnalysisDescription:
    """Describe one analysis: everything needed for a correct first call.

    Printing the description shows the question the analysis answers, its
    signature, when to use it, each parameter with its type, default, allowed
    values, valid range and meaning, the statuses it can return and why, its
    finding codes, the fields of its result, whether it supports masking, its
    work-limit parameters and an example call. Everything comes from the
    registry and the analysis's docstring; help() shows the docstring itself.

    Args:
        name (str): The analysis name, such as 'summarize_numeric', or its fully
            qualified name, such as
            'agent_eda.statistics.descriptive.summarize_numeric'.
            'agent_eda.summarize_numeric' and 'ae.summarize_numeric' work too.

    Returns:
        AnalysisDescription: `parameters` lists each parameter with `type`,
        `required`, `default`, `allowed_values`, `valid_range` and
        `description`. `statuses`, `finding_codes`, `result_fields` and
        `shared_result_fields` describe the result, and `example` holds the
        docstring's doctest. `to_dict()` and `to_json()` give the whole record,
        with `parameters_schema`, a JSON Schema of the parameters other than
        data.

    Raises:
        UnknownAnalysisError: The name isn't an analysis. The message suggests
            the closest names and lists every analysis.
        InvalidArgumentError: The name isn't a string.

    Examples:
        >>> import agent_eda as ae
        >>> description = ae.describe_analysis("summarize_numeric")
        >>> print(description)  # doctest: +ELLIPSIS
        summarize_numeric: How are the values in each numeric column distributed?
        Signature: summarize_numeric(data: pd.DataFrame, *, columns: Sequence[str] | None = None, ...
        ...
        >>> [p.name for p in description.parameters]
        ['data', 'columns', 'quantiles', 'quantile_method', 'trim_fraction']
        >>> description.parameters[4].valid_range
        '0 <= trim_fraction < 0.5'
    """
    spec = _find(name)
    parameters = [
        _parameter(spec, parameter) for parameter in spec.signature.parameters.values()
    ]
    return AnalysisDescription(
        name=spec.name,
        module=spec.module,
        summary=spec.summary,
        when_to_use=spec.docstring.when_to_use,
        signature=_signature(spec, parameters),
        parameters=parameters,
        parameters_schema=_parameters_schema(spec, parameters),
        statuses=[
            StatusDescription(
                status=status, reason=spec.docstring.statuses.get(status, "")
            )
            for status in spec.statuses
        ],
        finding_codes=[
            FindingCodeDescription(
                code=code, description=spec.docstring.codes.get(code)
            )
            for code in spec.finding_codes
        ],
        result_class=spec.result_class.__name__,
        result_fields=[
            _field(field_name, field)
            for field_name, field in spec.result_class.model_fields.items()
            if field_name not in AnalysisResult.model_fields
        ],
        shared_result_fields=[
            _field(field_name, field)
            for field_name, field in AnalysisResult.model_fields.items()
        ],
        methods=list(spec.methods),
        supports_masking=spec.supports_masking,
        work_limit_parameters=list(spec.work_limits),
        example=spec.docstring.examples,
    )


def search_analyses(query: str, limit: int = 5) -> SearchListing:
    """Find the analyses that best match a question in your own words.

    The query's words, without common ones such as 'the' and 'of', are matched
    against each analysis's name, tags, summary and When to use: text, after
    lowercasing and dropping plural endings. A small curated list adds
    synonyms: nulls and NaN find missing values, dupes find duplicates,
    correlation finds association and label finds target. A word scores 4 in a
    name, 3 in a tag, 2 in the summary and 1 in the When to use: text, and an
    analysis's score is the sum. An analysis whose name is the query comes
    first; ties go to the alphabetically first name. The search is keyword
    based, without embeddings or network, so the same query always gives the
    same ranking.

    Args:
        query (str): What you want to know, such as 'nulls per column'.
        limit (int): The most matches to return, at least 1; defaults to 5.

    Returns:
        SearchListing: `matches` holds the best matches, best first, each with
        its `name`, `summary`, `score` and `matched_terms`. `match_count`
        counts every match, also those beyond the limit, and `terms` shows the
        words searched for with their synonyms. With no match, `matches` is
        empty and the printed listing points to ae.guide().

    Raises:
        InvalidArgumentError: The query isn't a string, or limit isn't an int of
            at least 1.

    Examples:
        >>> import agent_eda as ae
        >>> found = ae.search_analyses("mean and standard deviation")
        >>> found.matches[0].name
        'summarize_numeric'
        >>> found.matches[0].matched_terms
        ['mean', 'standard', 'deviation']
        >>> print(ae.search_analyses("xylophone"))
        No analysis matches 'xylophone'. Searched for: xylophone.
        Try other words, or call ae.guide() for every analysis and where to start.
    """
    if not isinstance(query, str):
        raise InvalidArgumentError(
            "search_analyses takes the query as a string, such as "
            "ae.search_analyses('missing values'), but got a value of type "
            f"{type(query).__name__}."
        )
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit < 1:
        raise InvalidArgumentError(
            f"search_analyses got {limit!r} for limit, but limit must be an int of "
            "at least 1, such as ae.search_analyses('missing values', limit=3)."
        )
    limit = int(limit)
    searched = terms(query)
    ranked = rank(registered(), searched, query)
    return SearchListing(
        query=query,
        terms=[
            SearchTerm(term=term.word, synonyms=list(term.synonyms))
            for term in searched
        ],
        limit=limit,
        match_count=len(ranked),
        matches=[
            SearchMatch(
                name=match.spec.name,
                module=match.spec.module,
                summary=match.spec.summary,
                score=match.score,
                matched_terms=match.matched_terms,
            )
            for match in ranked[:limit]
        ],
    )


def _field(
    name: str, field: FieldInfo, seen: tuple[type, ...] = ()
) -> ResultFieldDescription:
    """A result field with its type, description and its entries' fields."""
    model = _record_type(field.annotation)
    nested = (
        [
            _field(inner, inner_field, (*seen, model))
            for inner, inner_field in model.model_fields.items()
        ]
        if model is not None and model not in seen
        else []
    )
    return ResultFieldDescription(
        name=name,
        type=type_text(field.annotation),
        description=field.description or "",
        fields=nested,
    )


def _record_type(annotation: object) -> type[BaseModel] | None:
    """The record type a field holds, alone or in a list, dict or union."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in typing.get_args(annotation):
        found = _record_type(argument)
        if found is not None:
            return found
    return None


def _parameter(
    spec: AnalysisSpec, parameter: inspect.Parameter
) -> ParameterDescription:
    # The annotation the registry validates with, resolved even when the module
    # writes annotations as strings.
    annotation = spec.annotations[parameter.name]
    required = parameter.default is inspect.Parameter.empty
    return ParameterDescription(
        name=parameter.name,
        type=type_text(annotation),
        required=required,
        default=None if required else to_jsonable_python(parameter.default),
        allowed_values=choices(annotation),
        valid_range=valid_range(annotation, parameter.name),
        description=spec.docstring.args.get(parameter.name, ""),
    )


def _parameters_schema(
    spec: AnalysisSpec, parameters: list[ParameterDescription]
) -> dict[str, JsonValue]:
    """The JSON Schema of the options, with each one's docstring description."""
    schema = spec.arguments.model_json_schema()
    properties = schema.get("properties", {})
    for parameter in parameters:
        if parameter.name in properties:
            properties[parameter.name]["description"] = parameter.description
    return schema


def _signature(spec: AnalysisSpec, parameters: list[ParameterDescription]) -> str:
    """The call signature with readable types and JSON-like defaults."""
    parts = []
    kinds = [p.kind for p in spec.signature.parameters.values()]
    for parameter, kind in zip(parameters, kinds, strict=True):
        if kind is inspect.Parameter.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        part = f"{parameter.name}: {parameter.type}"
        if not parameter.required:
            part += f" = {python_literal(parameter.default)}"
        parts.append(part)
    return f"{spec.name}({', '.join(parts)}) -> {spec.result_class.__name__}"


# describe_analysis names at most this many analyses in an error message.
_MAX_LISTED_ANALYSES = 20


def _not_a_name(value: object, names: list[str]) -> str:
    """Say that describe_analysis takes a name, showing the call to make."""
    function = getattr(value, "__name__", None)
    if callable(value) and function in names:
        return (
            "describe_analysis takes the name of an analysis as a string, but got "
            f"the function {function}. Pass its name instead: "
            f"ae.describe_analysis({function!r})."
        )
    return (
        "describe_analysis takes the name of an analysis as a string, such as "
        "ae.describe_analysis('describe_dataset'), but got a value of type "
        f"{type(value).__name__}. ae.list_analyses() lists every name."
    )


def _find(name: object) -> AnalysisSpec:
    """The analysis with this short or fully qualified name, or raise."""
    specs = registered()
    names = [spec.name for spec in specs]
    if not isinstance(name, str):
        raise InvalidArgumentError(_not_a_name(name, names))
    for spec in specs:
        if name in (
            spec.name,
            f"{spec.module}.{spec.name}",
            f"agent_eda.{spec.name}",
            f"ae.{spec.name}",
        ):
            return spec
    closest = closest_names(name.rpartition(".")[2], names)
    hint = (
        f" Did you mean {join_alternatives([repr(n) for n in closest])}?"
        if closest
        else ""
    )
    listed = join_items([repr(n) for n in names], limit=_MAX_LISTED_ANALYSES)
    raise UnknownAnalysisError(
        f"describe_analysis got an unknown analysis {name!r}.{hint} agent-eda has "
        f"{plural(len(names), 'analysis', 'analyses')}: {listed}. Pass one of these "
        "names, or find one from a question with ae.search_analyses('your question')."
    )
