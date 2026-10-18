"""Result classes returned by agent-eda analyses."""

import difflib
import itertools
import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from agent_eda import _text
from agent_eda._annotations import python_literal
from agent_eda.errors import InvalidArgumentError

Status = Literal[
    "completed",
    "partial",
    "insufficient_data",
    "not_applicable",
    "missing_prerequisites",
    "budget_exceeded",
    "blocked_by_policy",
    "failed",
]

Assessment = Literal["descriptive", "suspected", "violation", "passed", "unknown"]

# The package-wide list of limitation codes. Codes are public API: renaming one
# follows the rule for renamed names.
LimitationCode = Literal[
    # Values such as lists or dicts can't be hashed, so distinct values weren't
    # counted and the affected counts are null.
    "unhashable_values",
    # A statistic needs more values than a column has, such as a standard
    # deviation of one value, so it is null.
    "too_few_values",
    # Every value is equal, so a statistic such as skewness is undefined and null.
    "zero_variance",
    # Integers beyond ±2^53 were read as float64, which rounds them.
    "float_precision",
    # Booleans were read as numbers: 0 for False and 1 for True.
    "boolean_as_numeric",
    # A column stores NaN as a value, which pandas' isna() doesn't count as
    # missing; a numeric summary counts it as missing all the same.
    "nan_counted_as_missing",
    # A computation overflowed float64 or failed numerically (division by zero,
    # an invalid value, precision loss), so the values it affects are null.
    "numerical_warning",
]


class _Cap:
    """Marks a list or dict field that grows with the data.

    The record (to_dict and to_json) keeps the first `limit` entries and adds
    `<field>_total` with the full count. The result object keeps every entry.
    With `each_list`, a dict field stays whole and every list inside it is
    capped instead, with `<key>_total` next to it.
    """

    def __init__(self, limit: int, *, each_list: bool = False) -> None:
        self.limit = limit
        self.each_list = each_list


# The caps the spec sets for every list in the record that grows with the data.
_ROW_IDS = _Cap(10)
_EXAMPLE_VALUES = _Cap(5)
_ENTRIES = _Cap(100)
_LISTS_INSIDE = _Cap(100, each_list=True)

_RECORD = "agent_eda_record"


def _capped(values: dict[str, Any], limits: dict[str, int]) -> dict[str, Any]:
    """Keep the first entries of each named list or dict and add its total."""
    capped: dict[str, Any] = {}
    for name, value in values.items():
        limit = limits.get(name)
        if limit is None:
            capped[name] = value
            continue
        if isinstance(value, dict):
            capped[name] = dict(itertools.islice(value.items(), limit))
        else:
            capped[name] = value[:limit]
        capped[f"{name}_total"] = len(value)
    return capped


# Nullable integers keep their type in to_frame() and null floats become NaN;
# other fields use pandas' default.
_FRAME_DTYPES: dict[Any, str] = {
    int: "int64",
    int | None: "Int64",
    float | None: "float64",
    bool: "bool",
}


def _entries_frame(
    entries: Sequence["ResultModel"], model: type["ResultModel"]
) -> pd.DataFrame:
    """One row per entry and one column per field of the entry model."""
    columns = {
        name: pd.Series(
            [getattr(entry, name) for entry in entries],
            dtype=_FRAME_DTYPES.get(field.annotation),
        )
        for name, field in model.model_fields.items()
    }
    return pd.DataFrame(columns, columns=list(model.model_fields))


class ResultModel(BaseModel):
    """Base for every record inside an analysis result: frozen and strict."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @model_serializer(mode="wrap")
    def _cap_growing_fields(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        data = handler(self)
        if not (info.context and info.context.get(_RECORD)):
            return data
        caps = {
            name: marker
            for name, field in type(self).model_fields.items()
            for marker in field.metadata
            if isinstance(marker, _Cap)
        }
        if not caps:
            return data
        record = _capped(
            data,
            {name: cap.limit for name, cap in caps.items() if not cap.each_list},
        )
        for name, cap in caps.items():
            inner = record.get(name)
            if cap.each_list and isinstance(inner, dict):
                lists = {
                    key: cap.limit for key, v in inner.items() if isinstance(v, list)
                }
                record[name] = _capped(inner, lists)
        return record


class Finding(ResultModel):
    """A claim about the data, backed by result fields and carrying an assessment."""

    code: str = Field(description="Stable code naming the kind of finding.")
    claim: str = Field(description="One sentence from a fixed template.")
    assessment: Assessment = Field(
        description=(
            "descriptive: a neutral fact. suspected: a possible problem to judge. "
            "violation: the data breaks something declared. passed: a declared "
            "check held. unknown: a declared check couldn't be evaluated."
        )
    )
    columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns the finding concerns."
    )
    fields: Annotated[list[str], _ENTRIES] = Field(
        description="Paths of the result fields behind the claim, such as "
        "'columns[2].missing_count'."
    )


class Limitation(ResultModel):
    """Something that limits how far a result can be read."""

    code: LimitationCode = Field(
        description="Stable code from the package-wide list, LimitationCode."
    )
    reason: str = Field(description="One sentence on what is limited and why.")
    fields: Annotated[list[str], _ENTRIES] = Field(
        description="Paths of the result fields it affects."
    )


class StatusReason(ResultModel):
    """Why a run ended with a status other than completed."""

    code: str = Field(
        description="Stable code for the reason; for failed, the error code."
    )
    message: str = Field(description="One sentence on what happened.")


class ColumnSupport(ResultModel):
    """The counts behind one column's results."""

    column: str = Field(description="The column label as a string.")
    eligible_count: int = Field(description="Values eligible for the analysis.")
    evaluated_count: int = Field(description="Values the analysis evaluated.")
    excluded_count: int = Field(description="Eligible values left out.")
    excluded_by_reason: dict[str, int] = Field(
        description="Excluded values per reason code, such as 'missing'."
    )


class Support(ResultModel):
    """The counts behind a result: rows given and per-column counts."""

    row_count: int = Field(description="Rows in the data given to the analysis.")
    column_count: int = Field(description="Columns the analysis selected.")
    columns: Annotated[list[ColumnSupport], _ENTRIES] = Field(
        description="One entry per selected column, in the selected order."
    )


class Execution(ResultModel):
    """How the run was executed. Holds no timestamps or durations."""

    versions: dict[str, str] = Field(
        description="Versions of agent_eda, Python and every library the analysis "
        "used, keyed by import name."
    )


def _check_status_reason(status: Status, reason: StatusReason | None) -> None:
    if (status == "completed") != (reason is None):
        raise ValueError(
            "status_reason must be null when status is completed and set otherwise"
        )


class ColumnResult(ResultModel):
    """One column's entry in the result of an analysis over several columns."""

    column: str = Field(description="The column name: the label's string form.")
    status: Status = Field(
        description="How the run ended for this column, in the same terms as the "
        "result's status."
    )
    status_reason: StatusReason | None = Field(
        description="Why this column didn't complete; null when its status is "
        "completed."
    )

    @model_validator(mode="after")
    def _reason_matches_status(self) -> Self:
        _check_status_reason(self.status, self.status_reason)
        return self


class AnalysisResult(ResultModel):
    """The record an analysis returns.

    Every analysis returns its own subclass, named after it, with its typed fields
    next to these shared ones.
    """

    analysis: str = Field(
        description="Name of the analysis, such as 'describe_dataset'."
    )
    method: str | None = Field(
        description="The method used, or null when the analysis has no methods."
    )
    status: Status = Field(
        description="How the run ended: completed, partial, insufficient_data, "
        "not_applicable, missing_prerequisites, budget_exceeded, blocked_by_policy "
        "or failed. It says nothing about whether the data passed a check."
    )
    status_reason: StatusReason | None = Field(
        description="Why the run didn't complete; null when status is completed."
    )
    parameters: Annotated[dict[str, JsonValue], _LISTS_INSIDE] = Field(
        description="The resolved arguments by name, defaults included. data "
        "appears only as its row_count and column_count. In the record, a list "
        "argument keeps its first 100 entries and gains <name>_total."
    )
    support: Support = Field(description="The counts behind the result.")
    findings: Annotated[list[Finding], _ENTRIES] = Field(
        description="Claims about the data."
    )
    limitations: Annotated[list[Limitation], _ENTRIES] = Field(
        description="What limits how far the result can be read."
    )
    execution: Execution = Field(description="Library versions used for the run.")

    @model_validator(mode="after")
    def _reason_matches_status(self) -> Self:
        _check_status_reason(self.status, self.status_reason)
        return self

    def __str__(self) -> str:
        reason = self.status_reason
        facts = [f"Reason: {reason.code}: {reason.message}"] if reason else []
        return _text.render(
            heading=self._heading(),
            facts=[*facts, *self._text_facts()],
            table=self._text_table(),
            findings=[
                _text.Item(f"{f.assessment} {f.code}", f.claim) for f in self.findings
            ],
            limitations=[_text.Item(lim.code, lim.reason) for lim in self.limitations],
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _heading(self) -> str:
        shape = self.parameters.get("data")
        rows = shape.get("row_count") if isinstance(shape, dict) else None
        columns = shape.get("column_count") if isinstance(shape, dict) else None
        heading = f"{self.analysis}: {self.status}"
        if isinstance(rows, int) and isinstance(columns, int):
            heading += (
                f" on {_text.plural(rows, 'row', 'rows')} x "
                f"{_text.plural(columns, 'column', 'columns')}"
            )
        if self.method is not None:
            heading += f", method {self.method}"
        return heading

    def to_frame(self) -> pd.DataFrame:
        """The complete main table of the result as a DataFrame."""
        raise NotImplementedError(f"{type(self).__name__} has no table")

    def _text_facts(self) -> list[str]:
        """Lines of key facts under the first line; analyses override this."""
        return []

    def _text_table(self) -> _text.Table | None:
        """The key table; analyses override this."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """The result record as JSON-compatible Python objects.

        Every list that grows with the data keeps its first entries (10 row IDs,
        5 example values, 100 entries of anything else) and gains a
        `<field>_total` key with the full count. The result object and
        to_frame() keep every entry.
        """
        return self.model_dump(mode="json", context={_RECORD: True})

    def to_json(self) -> str:
        """The result record from to_dict() as strict JSON, without NaN or Infinity."""
        return json.dumps(
            self.to_dict(), allow_nan=False, ensure_ascii=False, separators=(",", ":")
        )


class _DiscoveryRecord(ResultModel):
    """What a discovery function returns: plain text to print and a JSON record."""

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-compatible Python objects."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """The record from to_dict() as strict JSON, without NaN or Infinity."""
        return json.dumps(
            self.to_dict(), allow_nan=False, ensure_ascii=False, separators=(",", ":")
        )


class AnalysisInfo(ResultModel):
    """What the registry records about one analysis."""

    name: str = Field(description="The analysis name, such as 'describe_dataset'.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    supports_masking: bool = Field(
        description="Whether it can mask the values of sensitive columns."
    )
    tags: list[str] = Field(description="Words that help search find it.")
    parameters: list[str] = Field(description="Its parameter names, data first.")
    methods: list[str] = Field(
        description="The methods it offers; empty when it has none."
    )
    statuses: list[Status] = Field(description="Every status it can return.")
    finding_codes: list[str] = Field(description="Every finding code it can report.")
    work_limit_parameters: list[str] = Field(
        description="Its max_* parameters that bound the work it does."
    )
    result_class: str = Field(
        description="The name of its result class in agent_eda.results."
    )


class AnalysisListing(_DiscoveryRecord):
    """Every analysis agent-eda offers, ordered by module, then name."""

    analyses: list[AnalysisInfo] = Field(description="One entry per analysis.")

    def __str__(self) -> str:
        count = len(self.analyses)
        lines = [f"{_text.plural(count, 'analysis', 'analyses')}, by module:"]
        module = None
        for info in self.analyses:
            if info.module != module:
                module = info.module
                lines.append(module)
            masking = " [supports masking]" if info.supports_masking else ""
            lines.append(f"  {info.name}: {info.summary}{masking}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per analysis, with every recorded field as a column."""
        return _entries_frame(self.analyses, AnalysisInfo)


class SearchMatch(ResultModel):
    """An analysis that matched a search, with what matched."""

    name: str = Field(description="The analysis name.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    score: int = Field(
        description="How well it matched: the sum over the matched query terms of "
        "the weight of the best field each occurs in (name 4, tags 3, summary 2, "
        "When to use: text 1)."
    )
    matched_terms: list[str] = Field(
        description="The query terms that matched, in query order."
    )


class SearchTerm(ResultModel):
    """A word of a search query and the other words it matched."""

    term: str = Field(description="The query word, lowercased.")
    synonyms: list[str] = Field(
        description="The package's word it matched as well, from a small curated "
        "list, such as missing for nulls; empty for any other word."
    )


class SearchListing(_DiscoveryRecord):
    """The analyses that match a search query, best first."""

    query: str = Field(description="The query as given.")
    terms: list[SearchTerm] = Field(
        description="The words searched for, in query order, without common words "
        "such as 'the' and 'of', each with its synonyms."
    )
    limit: int = Field(description="The most matches the listing keeps.")
    match_count: int = Field(
        description="How many analyses matched, including any beyond the limit."
    )
    matches: list[SearchMatch] = Field(
        description="The best matches, highest score first, ties by name; empty "
        "when nothing matched."
    )

    def __str__(self) -> str:
        searched = (
            ", ".join(
                f"{term.term} (also {', '.join(term.synonyms)})"
                if term.synonyms
                else term.term
                for term in self.terms
            )
            if self.terms
            else "no words, since common ones such as 'the' and 'of' don't count"
        )
        if not self.matches:
            lines = [
                *_text.wrap(
                    f"No analysis matches {self.query!r}. Searched for: {searched}.",
                    first="",
                    indent="    ",
                ),
                "Try other words, or call ae.guide() for every analysis and where "
                "to start.",
            ]
            return "\n".join(lines)
        lines = _text.wrap(
            f"Analyses matching {self.query!r} ({len(self.matches):,} of "
            f"{self.match_count:,} shown, best first):",
            first="",
            indent="    ",
        )
        for match in self.matches:
            lines.extend(
                _text.wrap(
                    f"{match.name}: {match.summary} Matched: "
                    f"{', '.join(match.matched_terms)}.",
                    first="  ",
                    indent="    ",
                )
            )
        lines.extend(_text.wrap(f"Searched for: {searched}.", first="", indent="    "))
        if self.match_count > len(self.matches):
            lines.extend(
                _text.wrap(
                    f"More: ae.search_analyses({self.query!r}, "
                    f"limit={self.match_count}) shows all {self.match_count:,} "
                    "matches.",
                    first="",
                    indent="    ",
                )
            )
        lines.append(
            f"Next: print(ae.describe_analysis({self.matches[0].name!r})) shows how "
            "to call it."
        )
        return "\n".join(lines)


class ParameterDescription(ResultModel):
    """One parameter of an analysis, as describe_analysis shows it."""

    name: str = Field(description="The parameter name.")
    type: str = Field(
        description="Its type as an agent reads it, such as 'Sequence[str] | None'."
    )
    required: bool = Field(description="Whether a call must pass it.")
    default: JsonValue = Field(
        description="Its default as a JSON value; null when it is required or "
        "defaults to None."
    )
    allowed_values: list[JsonValue] | None = Field(
        description="The fixed values it takes, or that each of its values takes "
        "when it is a list; null when any value of its type is allowed."
    )
    valid_range: str | None = Field(
        description="The range it must lie in, such as '0 <= trim_fraction < 0.5', "
        "or 'each value: 0 <= value <= 1' for a list; null when it declares none."
    )
    description: str = Field(
        description="What it means and takes, from the docstring's Args: section."
    )


class StatusDescription(ResultModel):
    """A status an analysis can return, and why it returns it."""

    status: Status = Field(description="The status.")
    reason: str = Field(
        description="When the analysis returns it, from the docstring's Statuses: "
        "section."
    )


class FindingCodeDescription(ResultModel):
    """A finding code an analysis can report, and what it means."""

    code: str = Field(description="The finding code.")
    description: str | None = Field(
        description="What it means, from the docstring's Returns: section; null "
        "when the docstring doesn't define it in a bullet."
    )


class ResultFieldDescription(ResultModel):
    """A field of an analysis result, and what it holds."""

    name: str = Field(description="The field name.")
    type: str = Field(
        description="Its type as an agent reads it, such as 'list[str]' or "
        "'dict[str, SummarizeNumericColumn]'."
    )
    description: str = Field(description="What it holds and how to read it.")
    fields: list["ResultFieldDescription"] = Field(
        description="The fields of each entry when it holds records, such as "
        "one per column; empty otherwise."
    )


class AnalysisDescription(_DiscoveryRecord):
    """Everything an agent needs for a correct first call of one analysis."""

    name: str = Field(description="The analysis name, such as 'summarize_numeric'.")
    module: str = Field(description="The module that defines it.")
    summary: str = Field(
        description="The question it answers: its docstring's first line."
    )
    when_to_use: str = Field(
        description="When to call it and what to call instead, from the "
        "docstring's When to use: section."
    )
    signature: str = Field(
        description="How to call it, with each parameter's type and default, such "
        "as 'describe_dataset(data: pd.DataFrame, *, source: str | None = None, "
        "...)'. A parameter with fixed choices shows the type of the choices."
    )
    parameters: list[ParameterDescription] = Field(
        description="Every parameter in signature order, data first."
    )
    parameters_schema: dict[str, JsonValue] = Field(
        description="A JSON Schema of the parameters other than data, with their "
        "types, defaults, allowed values, ranges and descriptions, as tool "
        "frameworks take it."
    )
    statuses: list[StatusDescription] = Field(
        description="Every status it can return, in the order it declares them."
    )
    finding_codes: list[FindingCodeDescription] = Field(
        description="Every finding code it can report, in the order it declares them."
    )
    result_class: str = Field(
        description="The name of its result class in agent_eda.results."
    )
    result_fields: list[ResultFieldDescription] = Field(
        description="The fields its result adds to the ones every result has."
    )
    shared_result_fields: list[ResultFieldDescription] = Field(
        description="The fields every analysis result has, such as status and findings."
    )
    methods: list[str] = Field(
        description="The methods it offers; empty when it has none."
    )
    supports_masking: bool = Field(
        description="Whether it can mask the values of sensitive columns, through "
        "its project and mask_sensitive parameters."
    )
    work_limit_parameters: list[str] = Field(
        description="Its max_* parameters that bound the work it does; it returns "
        "budget_exceeded when one is hit."
    )
    example: str = Field(
        description="The doctest from its docstring's Examples: section: Python "
        "lines after '>>> ' and '... ' prompts, each followed by what it prints."
    )

    def __str__(self) -> str:
        masking = (
            "supported (project=, mask_sensitive=)"
            if self.supports_masking
            else "not supported"
        )
        limits = ", ".join(self.work_limit_parameters) or "none"
        lines = [
            f"{self.name}: {self.summary}",
            *_text.wrap(self.signature, first="Signature: ", indent="    "),
            *_text.wrap(
                f"Module: {self.module}. Masking: {masking}. Work limits: {limits}.",
                first="",
                indent="    ",
            ),
        ]
        if self.methods:
            lines.extend(_text.wrap(f"Methods: {', '.join(self.methods)}."))
        lines.append("When to use:")
        lines.extend(_text.wrap(self.when_to_use, indent="  "))
        lines.append("Parameters:")
        for parameter in self.parameters:
            lines.extend(_parameter_lines(parameter))
        lines.append("Statuses:")
        for status in self.statuses:
            lines.extend(_entry_lines(status.status, status.reason))
        lines.append("Finding codes:" if self.finding_codes else "Finding codes: none")
        for finding in self.finding_codes:
            lines.extend(_entry_lines(finding.code, finding.description))
        shared = _text.join_items(
            [field.name for field in self.shared_result_fields],
            limit=len(self.shared_result_fields),
        )
        lines.extend(
            _text.wrap(
                f"Result: {self.result_class}. Besides the fields every result has "
                f"({shared}), it has:",
                first="",
                indent="    ",
            )
        )
        for field in self.result_fields:
            lines.extend(
                _text.wrap(f"{field.name}: {field.type}", first="  ", indent="      ")
            )
            lines.extend(_text.wrap(field.description, indent="    "))
            if field.fields:
                names = ", ".join(inner.name for inner in field.fields)
                lines.extend(_text.wrap(f"Fields: {names}.", indent="    "))
        lines.append("Example:")
        lines.extend(f"  {code}".rstrip() for code in _example_code(self.example))
        lines.extend(
            _text.wrap(
                f"More: help(ae.{self.name}) has the full docstring; to_json() adds "
                "the parameters' JSON Schema.",
                first="",
                indent="    ",
            )
        )
        return "\n".join(lines)


def _parameter_lines(parameter: ParameterDescription) -> list[str]:
    head = f"{parameter.name}: {parameter.type}"
    if parameter.required:
        head += " (required)"
    else:
        head += f" = {python_literal(parameter.default)}"
    if parameter.valid_range:
        head += f"; {parameter.valid_range}"
    lines = _text.wrap(head, first="  ", indent="      ")
    if parameter.allowed_values is not None:
        allowed = ", ".join(python_literal(value) for value in parameter.allowed_values)
        lines.extend(_text.wrap(allowed, first="    Allowed: ", indent="    "))
    lines.extend(_text.wrap(parameter.description, indent="    "))
    return lines


def _entry_lines(name: str, text: str | None) -> list[str]:
    """A "name: text" entry under a heading, its text wrapped below it."""
    return _text.wrap(f"{name}: {text}" if text else name, first="  ", indent="    ")


def _example_code(doctest_text: str) -> list[str]:
    """The Python lines of a doctest, without prompts.

    A statement's output follows it as a comment when it is one line, such as
    'res.columns["age"].median  # 38.0', and is left out when it is longer, such
    as a printed result. As doctest reads it, a line starting with '...' right
    after a '>>>' line continues that statement.
    """
    code: list[str] = []
    output: list[str] = []

    def add_output() -> None:
        if len(output) == 1 and code:
            commented = f"{code[-1]}  # {output[0].strip()}"
            if len(commented) + 2 <= _text.MAX_WIDTH:
                code[-1] = commented
        output.clear()

    in_statement = False
    for line in doctest_text.splitlines():
        if line.startswith(">>>"):
            add_output()
            code.append(line[4:])
            in_statement = True
        elif in_statement and line.startswith("..."):
            code.append(line[4:])
        else:
            in_statement = False
            if line.strip():
                output.append(line)
    add_output()
    return code


class DescribeDatasetColumn(ResultModel):
    """What describe_dataset found in one column."""

    position: int = Field(description="0-based position of the column in the data.")
    column: str = Field(description="The column label as a string.")
    label_type: str = Field(
        description="Python type of the column label, such as 'str', 'int' or "
        "'tuple'. Select a column whose label isn't a str with its original label."
    )
    dtype: str = Field(description="The dtype as pandas reports it, such as 'int64'.")
    non_missing_count: int = Field(description="Values that are not missing.")
    missing_count: int = Field(
        description="Missing values as pandas isna() counts them: None, NaN, NaT "
        "and pd.NA."
    )
    distinct_count: int | None = Field(
        description="Distinct non-missing values; null when the values can't be "
        "hashed, with an unhashable_values limitation."
    )
    memory_bytes: int = Field(
        description="Bytes the values use, counting each object a text or object "
        "column holds. Excludes pandas' lookup tables and the UTF-8 copy Python "
        "caches inside a string once something has encoded it, so earlier work on "
        "the data doesn't change it."
    )


class DescribeDatasetIndex(ResultModel):
    """What describe_dataset found in the row index."""

    dtype: str = Field(description="The index dtype as pandas reports it.")
    level_count: int = Field(description="Index levels: 1, or more for a MultiIndex.")
    is_unique: bool = Field(description="Whether every index label occurs once.")
    duplicated_count: int = Field(description="Labels that repeat an earlier label.")
    missing_count: int = Field(
        description="Missing labels; in a MultiIndex, labels with any level missing."
    )
    memory_bytes: int = Field(
        description="Bytes the index labels use. Excludes pandas' lookup tables and "
        "the UTF-8 copies Python caches inside strings."
    )


class Provenance(ResultModel):
    """Where the data came from, exactly as the caller supplied it."""

    source: str | None = Field(
        description="The source as supplied, or null when not supplied. Never inferred."
    )
    source_version: str | None = Field(
        description="The source version as supplied, or null when not supplied."
    )
    not_supplied: list[str] = Field(
        description="Provenance fields the caller did not supply."
    )


class DescribeDatasetResult(AnalysisResult):
    """The result of describe_dataset."""

    row_count: int = Field(description="Rows in the data.")
    column_count: int = Field(description="Columns in the data.")
    column_level_count: int = Field(
        description="Levels in the column labels: 1, or more for MultiIndex columns."
    )
    memory_bytes: int = Field(
        description="Bytes the columns and the index use together."
    )
    columns: Annotated[list[DescribeDatasetColumn], _ENTRIES] = Field(
        description="One entry per column, in DataFrame order. The record keeps "
        "the first 100 and adds columns_total; to_frame() has every column."
    )
    index: DescribeDatasetIndex = Field(description="Facts about the row index.")
    provenance: Provenance = Field(description="The provenance as supplied.")

    def _text_facts(self) -> list[str]:
        index = self.index
        levels = f" with {index.level_count} levels" if index.level_count > 1 else ""
        repeats = _text.plural(
            index.duplicated_count, "repeated label", "repeated labels"
        )
        unique = "unique" if index.is_unique else f"not unique ({repeats})"
        missing = _text.plural(index.missing_count, "missing label", "missing labels")
        provenance = "; ".join(
            f"{name} {value!r}" if value is not None else f"{name} not supplied"
            for name, value in (
                ("source", self.provenance.source),
                ("source_version", self.provenance.source_version),
            )
        )
        return [
            f"Memory: {_text.format_bytes(self.memory_bytes)} in total. "
            f"Index: {index.dtype}{levels}, {unique}, {missing}.",
            f"Provenance: {provenance}.",
        ]

    def to_frame(self) -> pd.DataFrame:
        """Every column's entry as one row, in DataFrame order."""
        return _entries_frame(self.columns, DescribeDatasetColumn)

    def _text_table(self) -> _text.Table:
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=(
                "column",
                "dtype",
                "non_missing_count",
                "missing_count",
                "distinct_count",
                "memory_bytes",
            ),
            rows=[
                (
                    c.column,
                    c.dtype,
                    c.non_missing_count,
                    c.missing_count,
                    c.distinct_count,
                    c.memory_bytes,
                )
                for c in self.columns
            ],
        )


class SummarizeNumericColumn(ColumnResult):
    """What summarize_numeric found in one column."""

    count: int = Field(
        description="Values that aren't missing, infinite ones included."
    )
    missing_count: int = Field(
        description="Missing values: None, NaN and pd.NA, including NaN that a "
        "pyarrow-backed or nullable float column stores as a value, which pandas "
        "isna() doesn't count."
    )
    infinite_count: int = Field(description="Values equal to inf or -inf.")
    zero_count: int = Field(description="Values equal to 0.")
    negative_count: int = Field(description="Values below 0, -inf included.")
    mean: float | None = Field(description="Arithmetic mean of the finite values.")
    std: float | None = Field(
        description="Sample standard deviation of the finite values with ddof 1: "
        "sqrt(sum((x - mean)^2) / (n - 1)), where n counts finite values. Needs "
        "n >= 2."
    )
    min: float | None = Field(description="Smallest finite value.")
    max: float | None = Field(description="Largest finite value.")
    quantiles: dict[float, float | None] = Field(
        description="Quantiles of the finite values, keyed by level in the order "
        "of parameters.quantiles, as numpy.quantile computes them with "
        "parameters.quantile_method."
    )
    median: float | None = Field(
        description="Middle finite value, or the mean of the two middle values; "
        "the same whatever quantile_method is."
    )
    trimmed_mean: float | None = Field(
        description="Mean of the finite values left after cutting "
        "int(trim_fraction * n) of them from each end, as scipy.stats.trim_mean "
        "does."
    )
    mad: float | None = Field(
        description="Median absolute deviation of the finite values, unscaled: "
        "median(|x - median(x)|)."
    )
    mad_scaled: float | None = Field(
        description="mad * 1.4826, which estimates the standard deviation when the "
        "values are normally distributed."
    )
    skewness: float | None = Field(
        description="Bias-corrected sample skewness G1 = sqrt(n(n - 1)) / (n - 2) "
        "* m3 / m2^1.5 of the finite values, where m_k = mean((x - mean)^k): "
        "scipy.stats.skew with bias=False. Needs n >= 3 and two distinct values."
    )
    excess_kurtosis: float | None = Field(
        description="Bias-corrected sample excess kurtosis G2 = (n - 1) / ((n - 2)"
        "(n - 3)) * ((n + 1) * m4 / m2^2 - 3(n - 1)) of the finite values; 0 for "
        "a normal distribution: scipy.stats.kurtosis with bias=False. Needs n >= 4 "
        "and two distinct values."
    )


class SummarizeNumericResult(AnalysisResult):
    """The result of summarize_numeric."""

    columns: Annotated[dict[str, SummarizeNumericColumn], _ENTRIES] = Field(
        description="One entry per selected column, keyed by column name, in the "
        "selected order. The record keeps the first 100 and adds columns_total; "
        "to_frame() has every column."
    )
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns that columns=None left out because they aren't "
        "numeric or are boolean, in DataFrame order; empty when columns names the "
        "columns."
    )
    definitions: dict[str, str] = Field(
        description="How each count and statistic in a column entry is defined, by "
        "field name."
    )

    def _text_facts(self) -> list[str]:
        trim = self.parameters.get("trim_fraction")
        percent = f"{trim * 100:.4g}%" if isinstance(trim, int | float) else "?"
        facts = [
            "Statistics use the finite values. Quantile method "
            f"{self.parameters.get('quantile_method')}; trimmed_mean cuts {percent} "
            "from each end."
        ]
        if self.left_out_columns:
            names = _text.join_labels(self.left_out_columns)
            facts.append(f"Left out, not numeric or boolean: {names}.")
        facts.append(
            "res.to_frame() also has the quantiles, trimmed_mean, mad, mad_scaled, "
            "skewness and excess_kurtosis."
        )
        return facts

    def _text_table(self) -> _text.Table:
        headers = ("column", "count", "missing_count", "mean", "std", "min", "median")
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=(*headers, "max"),
            rows=[
                (
                    e.column,
                    e.count,
                    e.missing_count,
                    e.mean,
                    e.std,
                    e.min,
                    e.median,
                    e.max,
                )
                for e in self.columns.values()
            ],
        )

    def to_frame(self) -> pd.DataFrame:
        """Every statistic of every column, one row per column in selected order.

        The quantiles become one column per level, named "q" and the level,
        such as "q0.25"; status_reason holds the reason's code.
        """
        recorded = self.parameters.get("quantiles")
        levels = [
            float(level)
            for level in (recorded if isinstance(recorded, list) else [])
            if isinstance(level, int | float)
        ]
        columns: dict[str, pd.Series] = {}
        for name, field in SummarizeNumericColumn.model_fields.items():
            values = [getattr(entry, name) for entry in self.columns.values()]
            if name == "status_reason":
                codes = [reason.code if reason else None for reason in values]
                columns[name] = pd.Series(codes, dtype=object)
            elif name == "quantiles":
                for level in levels:
                    columns[f"q{level!r}"] = pd.Series(
                        [quantiles[level] for quantiles in values], dtype="float64"
                    )
            else:
                dtype = _FRAME_DTYPES.get(field.annotation)
                columns[name] = pd.Series(values, dtype=dtype)
        return pd.DataFrame(columns, columns=list(columns))


# How compute_association_matrix treats a column it pairs.
AssociationType = Literal["numeric", "categorical"]

# The measures compute_association_matrix computes, one per pair.
AssociationMethod = Literal[
    "pearson",
    "spearman",
    "kendall",
    "cramers_v",
    "cramers_v_bias_corrected",
    "correlation_ratio",
]

# Why compute_association_matrix pairs a column with no other.
SkipReason = Literal[
    "datetime",
    "timedelta",
    "unsupported_dtype",
    "all_missing",
    "no_finite_values",
    "unhashable_values",
    "too_many_categories",
]

# Why a pair's value doesn't exist.
UndefinedReason = Literal[
    "too_few_rows", "single_level", "constant_column", "one_row_per_level"
]


class AssociationColumn(ResultModel):
    """A column compute_association_matrix paired with the others."""

    column: str = Field(description="The column name: the label's string form.")
    dtype: str = Field(description="The dtype as pandas reports it, such as 'Int64'.")
    type: AssociationType = Field(
        description="How the pairs treat the column: numeric (numbers other than "
        "booleans, unless listed in categorical_columns) or categorical (booleans, "
        "text, objects, categoricals and listed numbers)."
    )
    value_count: int = Field(
        description="Values the pairs can use: present, and finite in a numeric column."
    )
    missing_count: int = Field(
        description="Missing values: None, NaN, NaT and pd.NA, including NaN that a "
        "pyarrow-backed or nullable float column stores as a value."
    )
    infinite_count: int = Field(
        description="Values equal to inf or -inf in a numeric column, which the "
        "pairs leave out; 0 in a categorical column, where they are levels."
    )
    level_count: int | None = Field(
        description="Distinct values of a categorical column, missing values not "
        "counted; null for a numeric column."
    )


class SkippedColumn(ResultModel):
    """A column compute_association_matrix paired with no other, and why."""

    column: str = Field(description="The column name: the label's string form.")
    dtype: str = Field(description="The dtype as pandas reports it.")
    reason: SkipReason = Field(
        description="Why it was skipped: datetime, timedelta or unsupported_dtype "
        "for dtypes without a measure (only when columns is None); all_missing; "
        "no_finite_values for a numeric column holding only inf and -inf; "
        "unhashable_values for values such as lists; too_many_categories for a "
        "categorical column with more distinct values than max_categories."
    )
    message: str = Field(description="One sentence on why it was skipped.")


class AssociationPair(ResultModel):
    """How strongly two columns are associated, as compute_association_matrix found."""

    x_column: str = Field(
        description="The first column: the earlier one in the selected order, or "
        "the categorical column of a correlation ratio."
    )
    y_column: str = Field(
        description="The second column: the later one, or the numeric column of a "
        "correlation ratio."
    )
    x_type: AssociationType = Field(description="How the pair treats x_column.")
    y_type: AssociationType = Field(description="How the pair treats y_column.")
    method: AssociationMethod = Field(
        description="The measure: numeric_method for two numeric columns, "
        "categorical_method for two categorical ones, correlation_ratio from the "
        "categorical to the numeric column otherwise."
    )
    value: float | None = Field(
        description="The measure on the pair's complete rows: from -1 to 1 for "
        "pearson, spearman and kendall, from 0 to 1 for the others; null when it "
        "doesn't exist, and undefined_reason says why."
    )
    pair_count: int = Field(
        description="The pair's complete rows, which the value uses: both values "
        "present, and finite in a numeric column."
    )
    direction: Literal["symmetric", "x_to_y"] = Field(
        description="x_to_y for the correlation ratio, which measures how much the "
        "categorical x_column explains the numeric y_column and not the reverse; "
        "symmetric for the other measures."
    )
    undefined_reason: UndefinedReason | None = Field(
        description="Why value is null: too_few_rows (fewer than 3 complete rows), "
        "single_level (a categorical column has one level in them), "
        "constant_column (a numeric column has one value in them) or "
        "one_row_per_level (a categorical column has a level for every complete "
        "row, such as an ID, so it would explain the other trivially); null when "
        "value exists."
    )


class ComputeAssociationMatrixResult(AnalysisResult):
    """The result of compute_association_matrix."""

    pairs: Annotated[list[AssociationPair], _ENTRIES] = Field(
        description="One entry per pair of paired columns, strongest first: by "
        "absolute value, descending, then in the selected column order, with null "
        "values last. The record keeps the 100 strongest and adds pairs_total; "
        "to_frame() has every pair."
    )
    columns: Annotated[dict[str, AssociationColumn], _ENTRIES] = Field(
        description="The paired columns, keyed by column name, in the selected "
        "order. The record keeps the first 100 and adds columns_total."
    )
    skipped_columns: Annotated[list[SkippedColumn], _ENTRIES] = Field(
        description="The columns paired with no other, with the reason, in "
        "DataFrame order when columns is None and in the given order otherwise."
    )

    def to_frame(self, table: Literal["pairs", "matrix"] = "pairs") -> pd.DataFrame:
        """A complete table of the result, chosen by name.

        Args:
            table: "pairs", the default, gives one row per pair with every field
                of res.pairs, strongest first. "matrix" gives the paired columns
                as rows and columns in the selected order: the cell in row x and
                column y holds the pair's value, NaN where it is null. A
                correlation ratio fills only the cell from its categorical row
                to its numeric column, and the diagonal is empty.
        """
        _check_table(table, ("pairs", "matrix"))
        if table == "pairs":
            return _entries_frame(self.pairs, AssociationPair)
        names = list(self.columns)
        position = {name: i for i, name in enumerate(names)}
        cells = np.full((len(names), len(names)), np.nan)
        for pair in self.pairs:
            value = np.nan if pair.value is None else pair.value
            x, y = position[pair.x_column], position[pair.y_column]
            cells[x, y] = value
            if pair.direction == "symmetric":
                cells[y, x] = value
        return pd.DataFrame(
            cells,
            index=pd.Index(names, name="x_column"),
            columns=pd.Index(names, name="y_column"),
        )

    def _text_facts(self) -> list[str]:
        facts = []
        if self.pairs:
            counts = dict.fromkeys(("numeric", "categorical", "mixed"), 0)
            for pair in self.pairs:
                kind = pair.x_type if pair.x_type == pair.y_type else "mixed"
                counts[kind] += 1
            facts.extend(
                [
                    "Pairs, each on its own complete rows: "
                    f"{counts['numeric']:,} numeric "
                    f"({self.parameters.get('numeric_method')}), "
                    f"{counts['categorical']:,} categorical "
                    f"({self.parameters.get('categorical_method')}),",
                    f"    {counts['mixed']:,} mixed (correlation_ratio, from the "
                    "categorical x_column to the numeric y_column).",
                ]
            )
        if self.skipped_columns:
            skipped = [f"{s.column!r} ({s.reason})" for s in self.skipped_columns]
            facts.extend(
                _text.wrap(
                    "Skipped "
                    f"{_text.plural(len(skipped), 'column', 'columns')}, see "
                    f"res.skipped_columns: {_text.join_items(skipped)}.",
                    first="",
                    indent="    ",
                )
            )
        if self.pairs:
            size = len(self.columns)
            facts.append(
                "Tables: res.to_frame() has every pair, strongest first; "
                f"res.to_frame(table='matrix') has the {size:,} x {size:,} matrix."
            )
        return facts

    def _text_table(self) -> _text.Table:
        return _text.Table(
            title="Pairs",
            unit=("pair", "pairs"),
            headers=(
                "x_column",
                "y_column",
                "method",
                "value",
                "pair_count",
                "direction",
            ),
            rows=[
                (
                    p.x_column,
                    p.y_column,
                    p.method,
                    p.value,
                    p.pair_count,
                    p.direction,
                )
                for p in self.pairs
            ],
        )


def _check_table(table: object, allowed: tuple[str, ...]) -> None:
    """Raise unless `table` names one of a result's tables."""
    if isinstance(table, str) and table in allowed:
        return
    choices = _text.join_alternatives([repr(name) for name in allowed])
    closest = (
        difflib.get_close_matches(table, allowed, n=1) if isinstance(table, str) else []
    )
    hint = f" Did you mean {closest[0]!r}?" if closest else ""
    raise InvalidArgumentError(
        f"to_frame got {_text.value_text(table)} for table, but table must be "
        f"{choices}.{hint}"
    )
