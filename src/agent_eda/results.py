"""Result classes returned by agent-eda analyses."""

import datetime as dt
import difflib
import itertools
import json
import math
import typing
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from agent_eda import _text
from agent_eda._annotations import TypeName, python_literal
from agent_eda._levels import Levels, levels_frame
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

# What a column's values mean, as opposed to the dtype pandas stores them in.
# infer_column_semantics ranks these for each column, and a declared schema
# names one per column.
LogicalType = Literal[
    "boolean",
    "integer",
    "float",
    "categorical",
    "text",
    "datetime",
    "timedelta",
]

# What infer_column_semantics flags in a column that needs attention.
ColumnFlag = Literal[
    "identifier_like",
    "constant",
    "all_missing",
    "numbers_as_text",
    "dates_as_text",
    "non_iso_date_text",
    "boolean_like",
    "mixed_value_types",
]

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
    # A column stores NaN as a value, which pandas' isna() doesn't count as
    # missing, and neither does a count of native missing values.
    "nan_not_counted_as_missing",
    # A computation overflowed float64 or failed numerically (division by zero,
    # an invalid value, precision loss), so the values it affects are null.
    "numerical_warning",
    # Index labels or id_column values repeat, so a row ID can match more than
    # one row.
    "ambiguous_row_ids",
    # No sentinel values were declared for a column, so its sentinel count is
    # null and stand-ins such as -999 don't count as missing.
    "sentinel_values_not_declared",
    # A ranking stopped at its work limit before bounds proved it, so a set it
    # didn't count could rank higher than one it lists.
    "ranking_not_proven",
]


class _Cap:
    """Marks a list or dict field that grows with the data.

    The record (to_dict and to_json) keeps the first `limit` entries and adds
    `<field>_total` with the full count. The result object keeps every entry.
    With `each_list`, a dict field stays whole and every list or dict inside it
    is capped instead, with `<key>_total` next to it, except the keys named in
    `whole`. With `total`, the field holds only the first entries even on the
    result object, and `<field>_total` takes the full count from the field that
    `total` names.
    """

    def __init__(
        self,
        limit: int,
        *,
        each_list: bool = False,
        total: str | None = None,
        whole: tuple[str, ...] = (),
    ) -> None:
        self.limit = limit
        self.each_list = each_list
        self.total = total
        self.whole = whole


# The caps the spec sets for every list in the record that grows with the data.
_ROW_IDS = _Cap(10)
_EXAMPLE_VALUES = _Cap(5)
_ENTRIES = _Cap(100)
# For parameters: list and mapping arguments, but not data's shape.
_LISTS_INSIDE = _Cap(100, each_list=True, whole=("data",))

_RECORD = "agent_eda_record"


def row_id_json(value: object) -> JsonValue:
    """A row ID as JSON, the same in every process and with every library version.

    Timestamps, dates and times become ISO 8601 strings, durations ISO 8601
    durations, MultiIndex tuples lists, and NumPy scalars Python numbers.
    Missing values become null, and infinite floats the strings 'inf' and
    '-inf'. A set becomes a list ordered by its items' JSON, not by Python's
    hash order, and a dict a JSON object. Anything else, such as a Period or a
    Decimal, becomes its str().
    """
    if isinstance(value, tuple | list):
        return [row_id_json(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [row_id_json(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, dict):
        return {str(key): row_id_json(item) for key, item in value.items()}
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    # Before the integers: NumPy's timedelta64 is one of its integer types.
    if isinstance(value, dt.timedelta | np.timedelta64):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else repr(number)
    return str(value)


# A row ID: an index label, or a value of the column id_column names. The result
# object holds it as pandas gives it, so df.loc[row_ids] finds the rows; the
# record converts it with row_id_json.
RowId = Annotated[
    Any,
    PlainSerializer(row_id_json, return_type=Any, when_used="json"),
    TypeName("RowId"),
]


def _head(values: list[Any] | dict[Any, Any], limit: int) -> list[Any] | dict[Any, Any]:
    """The first `limit` entries of a list or dict."""
    if isinstance(values, dict):
        return dict(itertools.islice(values.items(), limit))
    return values[:limit]


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
    float: "float64",
    float | None: "float64",
    bool: "bool",
}


def _check_table(table: object, tables: Sequence[str]) -> None:
    """Raise unless `table` names one of a result's tables; the first is the default.

    Results with more than one table take the name in to_frame(table=...).
    """
    if isinstance(table, str) and table in tables:
        return
    shown = repr(table) if isinstance(table, str) else f"a {type(table).__name__}"
    closest = (
        difflib.get_close_matches(table, tables, n=1) if isinstance(table, str) else []
    )
    hint = f" Did you mean {closest[0]!r}?" if closest else ""
    allowed = _text.join_alternatives([repr(name) for name in tables])
    raise InvalidArgumentError(
        f"to_frame got {shown} for table, but table must be one of {allowed}.{hint} "
        f"res.to_frame() returns the {tables[0]} table."
    )


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
        if not (info.context and info.context.get(_RECORD)):
            return handler(self)
        caps = {
            name: marker
            for name, field in type(self).model_fields.items()
            for marker in field.metadata
            if isinstance(marker, _Cap)
        }
        if not caps:
            return handler(self)
        limits = {name: cap.limit for name, cap in caps.items() if not cap.each_list}
        lengths = {name: len(getattr(self, name)) for name in limits}
        # A cap with `total` names the field that holds the full count, since
        # the result object keeps only the first entries of that list.
        totals = {
            name: getattr(self, total) if (total := caps[name].total) else length
            for name, length in lengths.items()
        }
        # Serialize only the entries the record keeps, which saves converting
        # what it would drop, such as the row IDs of 500,000 duplicate groups.
        heads = {
            name: _head(getattr(self, name), limit)
            for name, limit in limits.items()
            if lengths[name] > limit
        }
        data = handler(self.model_copy(update=heads) if heads else self)
        record: dict[str, Any] = {}
        for name, value in data.items():
            record[name] = value
            if name in totals:
                record[f"{name}_total"] = totals[name]
        for name, cap in caps.items():
            inner = record.get(name)
            if cap.each_list and isinstance(inner, dict):
                lists = {
                    key: cap.limit
                    for key, v in inner.items()
                    if isinstance(v, list | dict) and key not in cap.whole
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
        "or mapping argument keeps its first 100 entries and gains <name>_total."
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
        # Wrapped, not cut: a reason ends with how to fix the call.
        facts = (
            _text.wrap(
                f"Reason: {reason.code}: {reason.message}", indent="    ", first=""
            )
            if reason
            else []
        )
        return _text.render(
            heading=self._heading(),
            facts=[*facts, *self._text_facts()],
            table=self._text_table(),
            findings=[
                _text.Item(f"{f.assessment} {f.code}", f.claim) for f in self.findings
            ],
            limitations=[_text.Item(lim.code, lim.reason) for lim in self.limitations],
            other_tables=self._text_other_tables(),
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

    def _text_other_tables(self) -> list[str]:
        """What to_frame(table=...) returns besides the key table, as clauses for
        the footer; analyses with more than one table override this."""
        return []

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


class SummarizeCategoricalLevel(ResultModel):
    """A level of a column: a distinct non-missing value and how often it occurs."""

    label: str | bool | int | float = Field(
        description="The value exactly as stored, so 'Yes' and 'yes ' are two "
        "levels. A value JSON can't hold, such as a Decimal, a timestamp or a "
        "non-finite float, appears as its string form; label_type names its type."
    )
    label_type: str = Field(
        description="The Python type of the stored value, such as 'str', 'int' or "
        "'bool'; a NumPy scalar counts as the Python type it holds."
    )
    count: int = Field(description="Values equal to the label.")
    share: float = Field(
        description="count / non_missing_count of the column: the share of its "
        "non-missing values, so missing values never dilute it."
    )


def _level_text(level: SummarizeCategoricalLevel) -> str:
    """A level as the text shows it, such as "'basic': 3 (60%)".

    Text labels are quoted, so surrounding spaces stay visible.
    """
    label = repr(level.label) if level.label_type == "str" else str(level.label)
    share = _text.format_number(level.share * 100)
    return f"{label}: {level.count:,} ({share}%)"


class SummarizeCategoricalVariants(ResultModel):
    """Labels of one column that differ only by case or surrounding whitespace."""

    levels: Annotated[list[SummarizeCategoricalLevel], _ENTRIES] = Field(
        description="The labels of the group as levels, each with its count and "
        "share, in level order."
    )


class SummarizeCategoricalColumn(ColumnResult):
    """What summarize_categorical found in one column."""

    non_missing_count: int = Field(description="Values that aren't missing.")
    missing_count: int = Field(
        description="Missing values: None, NaN, NaT and pd.NA, the ones pandas "
        "value_counts leaves out. NaN that a pyarrow-backed float column stores "
        "as a value is a level, as in value_counts."
    )
    cardinality: int | None = Field(
        description="Levels: the distinct non-missing values, as pandas "
        "value_counts tells them apart, so 1, 1.0 and True are one level. Null "
        "when the values can't be hashed."
    )
    levels: Annotated[
        list[SummarizeCategoricalLevel], _Cap(100, total="cardinality")
    ] = Field(
        description="The first 100 levels, ordered by count, descending, then by "
        "label, as level_order states. cardinality counts every level, the "
        "record adds it as levels_total, and to_frame() has every level."
    )
    most_common_levels: list[SummarizeCategoricalLevel] = Field(
        description="The first top_n levels: the ones the printed text shows."
    )
    left_out_level_count: int | None = Field(
        description="Levels beyond the first top_n, which most_common_levels and "
        "the printed text leave out. Null when the values can't be hashed."
    )
    seen_once_count: int | None = Field(
        description="Levels that occur exactly once. Null when the values can't "
        "be hashed."
    )
    unused_categories: Annotated[list[str | bool | int | float], _ENTRIES] = Field(
        description="Categories of a categorical dtype that no value takes, in "
        "category order, each written as a level's label is; empty for other "
        "dtypes."
    )
    value_types: Annotated[list[str], _ENTRIES] = Field(
        description="The Python types of the non-missing values by name, sorted, "
        "such as ['int', 'str']; a NumPy scalar counts as the Python type it holds."
    )
    mixed_value_types: bool = Field(
        description="Whether the non-missing values have more than one Python "
        "type, such as 1 and '1'. Values pandas counts as one level, such as 1 "
        "and True, can still differ in type."
    )
    case_or_space_variants: Annotated[list[SummarizeCategoricalVariants], _ENTRIES] = (
        Field(
            description="Groups of text labels that become equal once str.strip "
            "removes surrounding whitespace and str.casefold folds case, such as "
            "'Yes', 'yes ' and 'YES'. Each group is in level order, and the groups "
            "follow the level order of their first label. Empty when there are none."
        )
    )


class SummarizeCategoricalResult(AnalysisResult):
    """The result of summarize_categorical."""

    columns: Annotated[dict[str, SummarizeCategoricalColumn], _ENTRIES] = Field(
        description="One entry per selected column, keyed by column name, in the "
        "selected order. The record keeps the first 100 and adds columns_total."
    )
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns that columns=None left out because they aren't text, "
        "object, categorical or boolean columns, in DataFrame order; empty when "
        "columns names the columns."
    )
    share_denominator: Literal["non_missing_count"] = Field(
        description="The entry field every share divides by: non_missing_count, "
        "the column's values that aren't missing. Missing values get no share; "
        "missing_count counts them."
    )
    level_order: str = Field(
        description="How levels are ordered. They go by count, descending, then by "
        "label, and labels of different types follow a fixed order that never "
        "depends on hashing."
    )
    # Every level of every counted column, outside the record; to_frame() sorts
    # them when asked, since sorting a million labels is slow.
    _tables: dict[str, Levels] = PrivateAttr(default_factory=dict)

    def _text_facts(self) -> list[str]:
        facts = [
            "Shares are of each column's non-missing values. Levels go by count, "
            f"then label; most_common_levels has the top {self.parameters.get('top_n')}."
        ]
        if self.left_out_columns:
            names = _text.join_labels(self.left_out_columns)
            facts.append(
                f"Left out, not text, object, categorical or boolean: {names}."
            )
        return facts

    def _text_table(self) -> _text.Table:
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=(
                "column",
                "non_missing_count",
                "missing_count",
                "cardinality",
                "most_common_levels",
            ),
            rows=[
                (
                    e.column,
                    e.non_missing_count,
                    e.missing_count,
                    e.cardinality,
                    ", ".join(_level_text(level) for level in e.most_common_levels),
                )
                for e in self.columns.values()
            ],
            frame_rows=sum(e.cardinality or 0 for e in self.columns.values()),
            frame_unit=("level", "levels"),
        )

    def to_frame(self) -> pd.DataFrame:
        """Every level of every column, one row per level.

        Rows follow the selected column order, then level order. The columns
        are column, label, label_type, count and share, and label holds each
        value exactly as stored. Columns whose values can't be hashed have no
        rows.
        """
        return levels_frame(self._tables)


class DuplicateGroup(ResultModel):
    """Rows that are equal in every compared column."""

    size: int = Field(description="Rows in the group, at least 2.")
    row_ids: Annotated[list[RowId], _ROW_IDS] = Field(
        description="The rows' IDs in row order: index labels, or values of "
        "id_column when it is given. The record keeps the first 10 and adds "
        "row_ids_total."
    )


class FindDuplicateRowsResult(AnalysisResult):
    """The result of find_duplicate_rows."""

    compared_columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns in which rows must be equal to count as duplicates, "
        "in the order compared. The record keeps the first 100 and adds "
        "compared_columns_total."
    )
    missing_values_compare_equal: bool = Field(
        description="Whether missing values count as equal when rows are compared: "
        "always true. None, NaN, NaT and pd.NA all equal each other, as in pandas' "
        "duplicated() over two or more columns. Lists, dicts and sets compare by "
        "content."
    )
    duplicate_groups: Annotated[list[DuplicateGroup], _ENTRIES] = Field(
        description="Each group of rows equal in every compared column, ordered "
        "by first occurrence; a group's number is its position here. The record "
        "keeps the first 100 and adds duplicate_groups_total; to_frame() has "
        "every row of every group."
    )
    duplicate_group_count: int = Field(description="Groups of duplicate rows.")
    duplicate_group_row_count: int = Field(
        description="Rows in duplicate groups, every copy counted, the first too."
    )
    excess_copy_count: int = Field(
        description="Rows in duplicate groups beyond the first of each group: "
        "duplicate_group_row_count - duplicate_group_count, the rows that "
        "removing duplicates would drop."
    )
    duplicate_share: float | None = Field(
        description="excess_copy_count over all rows, from 0 up to but excluding 1; "
        "null when the data has no rows."
    )

    def _text_facts(self) -> list[str]:
        facts = []
        if self.compared_columns:
            compared = _text.plural(len(self.compared_columns), "column", "columns")
            facts.append(
                f"Compared {compared}: {_text.join_labels(self.compared_columns)}; "
                "missing values compare equal, as in pandas."
            )
        id_column = self.parameters.get("id_column")
        source = (
            f"values of id_column {id_column!r}"
            if isinstance(id_column, str)
            else "index labels"
        )
        facts.append(f"Row IDs: {source}.")
        return facts

    def _text_table(self) -> _text.Table:
        rows = self.duplicate_group_row_count
        return _text.Table(
            title="Duplicate groups",
            unit=("group", "groups"),
            headers=("group", "size", "row_ids"),
            rows=[
                (number, group.size, _text.join_labels(group.row_ids))
                for number, group in enumerate(
                    self.duplicate_groups[: _text.MAX_TABLE_ROWS]
                )
            ],
            total=len(self.duplicate_groups),
            frame=f"all {_text.plural(rows, 'row', 'rows')} in groups"
            if rows
            else "no rows",
        )

    def to_frame(self) -> pd.DataFrame:
        """Every row in a duplicate group, by group, then in row order.

        The column group holds the group's number, its position in
        duplicate_groups, and row_id the row's ID.
        """
        numbers = [
            number
            for number, group in enumerate(self.duplicate_groups)
            for _ in group.row_ids
        ]
        row_ids = [
            row_id for group in self.duplicate_groups for row_id in group.row_ids
        ]
        return pd.DataFrame(
            {
                "group": pd.Series(numbers, dtype="int64"),
                "row_id": pd.Series(row_ids, dtype=object),
            }
        )


class DeclaredSentinelValue(ResultModel):
    """A sentinel value declared for a column, and how often it occurs there."""

    value: bool | int | float | str = Field(
        description="The value as declared; dates, datetimes and timedeltas as ISO "
        "8601 text."
    )
    count: int = Field(
        description="Values in the column equal to it, among those pandas doesn't "
        "count as missing."
    )


class ProfileMissingValuesColumn(ColumnResult):
    """What profile_missing_values found in one column."""

    native_missing_count: int = Field(
        description="Values pandas isna() counts as missing: None, NaN, NaT and pd.NA."
    )
    sentinel_count: int | None = Field(
        description="Values equal to a sentinel value declared for the column; null "
        "when none was declared."
    )
    missing_count: int = Field(
        description="native_missing_count plus sentinel_count: every value that is "
        "missing or a declared sentinel value, each counted once."
    )
    missing_share: float | None = Field(
        description="missing_count over all rows, from 0 to 1; null when the data "
        "has no rows."
    )
    sentinel_values: Annotated[list[DeclaredSentinelValue], _ENTRIES] = Field(
        description="The sentinel values declared for the column, in the order "
        "given, each with its count; empty when none was declared. The record "
        "keeps the first 100 and adds sentinel_values_total."
    )


class ProfileMissingValuesRowCount(ResultModel):
    """How many rows miss a given number of the selected values."""

    missing_count: int = Field(
        description="Selected values a row misses, counting declared sentinel values."
    )
    row_count: int = Field(description="Rows that miss exactly that many values.")
    row_share: float = Field(description="row_count over all rows, from 0 to 1.")


def _percent(share: float | None) -> str:
    """A share as " (33.33%)" with 4 significant digits; "" when it is null."""
    return "" if share is None else f" ({_text.format_number(share * 100)}%)"


class ProfileMissingValuesResult(AnalysisResult):
    """The result of profile_missing_values."""

    columns: Annotated[dict[str, ProfileMissingValuesColumn], _ENTRIES] = Field(
        description="One entry per selected column, keyed by column name, in the "
        "selected order. The record keeps the first 100 and adds columns_total; "
        "to_frame() has every column."
    )
    missing_count: int = Field(
        description="Selected values that are missing or a declared sentinel value, "
        "over every selected column."
    )
    missing_share: float | None = Field(
        description="missing_count over all selected values (rows times selected "
        "columns), from 0 to 1; null when there are none."
    )
    complete_row_count: int = Field(
        description="Rows that miss none of the selected values, counting declared "
        "sentinel values as missing."
    )
    complete_row_share: float | None = Field(
        description="complete_row_count over all rows, from 0 to 1; null when the "
        "data has no rows."
    )
    row_distribution: Annotated[list[ProfileMissingValuesRowCount], _ENTRIES] = Field(
        description="How many rows miss each number of selected values, one entry "
        "per number that occurs, fewest missing first. The record keeps the first "
        "100 and adds row_distribution_total; to_frame(table='row_distribution') "
        "has every entry."
    )

    def _text_facts(self) -> list[str]:
        entries = self.columns.values()
        rows = self.support.row_count
        declared = any(entry.sentinel_count is not None for entry in entries)
        missing = (
            f"Missing: {self.missing_count:,} of {rows * len(self.columns):,} "
            f"selected values{_percent(self.missing_share)}"
            f"{', counting declared sentinel values' if declared else ''}. "
            f"Complete rows: {self.complete_row_count:,} of {rows:,}"
            f"{_percent(self.complete_row_share)}."
        )
        facts = _text.wrap(missing, first="", indent="    ")
        if self.row_distribution:
            per_row = [
                f"{entry.missing_count:,} in "
                f"{_text.plural(entry.row_count, 'row', 'rows')}"
                for entry in self.row_distribution
            ]
            facts.extend(
                _text.wrap(
                    f"Missing values per row: {_text.join_items(per_row, limit=8)}.",
                    first="",
                    indent="    ",
                )
            )
        return facts

    def _text_table(self) -> _text.Table:
        # Most missing first, so a long table shows where data is missing.
        entries = sorted(self.columns.values(), key=lambda e: -e.missing_count)
        return _text.Table(
            title="Columns, most missing first",
            unit=("column", "columns"),
            headers=(
                "column",
                "native_missing_count",
                "sentinel_count",
                "missing_count",
                "missing_share",
            ),
            rows=[
                (
                    e.column,
                    e.native_missing_count,
                    e.sentinel_count,
                    e.missing_count,
                    e.missing_share,
                )
                for e in entries
            ],
        )

    def _text_other_tables(self) -> list[str]:
        return [
            "res.to_frame(table='row_distribution') has the rows per number of "
            "missing values"
        ]

    def to_frame(
        self, table: Literal["columns", "row_distribution"] = "columns"
    ) -> pd.DataFrame:
        """One of the result's two tables as a DataFrame.

        Args:
            table: 'columns', the default, gives one row per selected column in
                the selected order, with its counts and share; status_reason
                holds the reason's code, and sentinel_values maps each declared
                value to its count. 'row_distribution' gives one row per number
                of missing values that occurs: missing_count, row_count and
                row_share.
        """
        _check_table(table, ("columns", "row_distribution"))
        if table == "row_distribution":
            return _entries_frame(self.row_distribution, ProfileMissingValuesRowCount)
        columns: dict[str, pd.Series] = {}
        entries = list(self.columns.values())
        for name, field in ProfileMissingValuesColumn.model_fields.items():
            values = [getattr(entry, name) for entry in entries]
            if name == "status_reason":
                codes = [reason.code if reason else None for reason in values]
                columns[name] = pd.Series(codes, dtype=object)
            elif name == "sentinel_values":
                counts = [{v.value: v.count for v in declared} for declared in values]
                columns[name] = pd.Series(counts, dtype=object)
            else:
                dtype = _FRAME_DTYPES.get(field.annotation)
                columns[name] = pd.Series(values, dtype=dtype)
        return pd.DataFrame(columns, columns=list(columns))


class CandidateKey(ResultModel):
    """A column set the key search reports, with how well it identifies rows."""

    columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns of the set, in the order of the searched columns."
    )
    distinct_count: int = Field(
        description="Distinct combinations of values in these columns, with missing "
        "values compared equal: the rows df.drop_duplicates(subset=columns) keeps."
    )
    duplicate_row_count: int = Field(
        description="Rows whose combination another row shares, the rows "
        "df.duplicated(subset=columns, keep=False) marks; 0 for a key."
    )
    missing_row_count: int = Field(
        description="Rows with a missing value in at least one of these columns."
    )
    uniqueness_share: float = Field(
        description="distinct_count divided by the rows of the data; 1 for a key."
    )
    is_minimal: bool = Field(
        description="Whether no smaller set of these columns identifies rows as "
        "well, so every column is needed."
    )


class KeySearch(ResultModel):
    """How far the key search went."""

    column_count: int = Field(
        description="Columns searched: the selected ones whose values can be hashed."
    )
    combination_count: int = Field(
        description="Column sets of 1 to max_key_size searched columns: the most "
        "the complete search examines, and the max_combinations that always "
        "suffices."
    )
    examined_count: int = Field(
        description="Column sets the search examined. Supersets of keys aren't "
        "examined."
    )
    rejected_on_subset_count: int = Field(
        description="Examined sets that repeated a combination within an evenly "
        "spaced subset of rows, which proves they aren't keys."
    )
    checked_on_all_rows_count: int = Field(
        description="Examined sets checked on all rows because no subset showed "
        "a repeat. Only a check on all rows confirms a key."
    )
    subset_row_count: int = Field(
        description="Rows in the evenly spaced subset each set is checked on "
        "first: row i * rows // 16,384 for each i below 16,384, or every row of "
        "smaller data. In data of more than 65,536 rows, a set without a repeat "
        "there is checked on every 4th row next."
    )


class FindCandidateKeysResult(AnalysisResult):
    """The result of find_candidate_keys."""

    keys: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="Minimal keys without missing values, each fit to serve as a "
        "primary key: column sets whose combination of values no two rows share. "
        "Fewest columns first, then in column order. The record keeps the first "
        "100 and adds keys_total; to_frame() has every one."
    )
    keys_with_missing_values: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="Minimal keys with a missing value in some rows, which can't "
        "serve as primary keys, in the same order as keys."
    )
    most_unique_column_sets: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="When the search found no key: the 10 minimal column sets "
        "with the most distinct combinations, most first, then fewest columns, "
        "then in column order. Empty when it found a key."
    )
    exhaustive: bool = Field(
        description="Whether the search examined every set of up to max_key_size "
        "columns, apart from supersets of keys. False when max_combinations "
        "stopped it, or when no search ran."
    )
    search: KeySearch = Field(description="How far the search went.")
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Selected columns the search left out because their values "
        "can't be hashed, such as lists or dicts, in the selected order; empty "
        "when it searched every selected column."
    )

    def _candidates(self) -> list[tuple[str, CandidateKey]]:
        """Every reported column set with its kind, in the order of the lists."""
        return [
            *(("key", key) for key in self.keys),
            *(
                ("key_with_missing_values", key)
                for key in self.keys_with_missing_values
            ),
            *(("not_a_key", key) for key in self.most_unique_column_sets),
        ]

    def _text_facts(self) -> list[str]:
        search = self.search
        if not search.examined_count:
            return []
        size = self.parameters.get("max_key_size")
        largest = min(size, search.column_count) if isinstance(size, int) else 0
        keys = f"keys of up to {_text.plural(largest, 'column', 'columns')}"
        if self.exhaustive:
            columns = _text.plural(search.column_count, "column", "columns")
            examined = _text.plural(search.examined_count, "set", "sets")
            skipped = search.combination_count - search.examined_count
            searched = (
                f"Exhaustive search for {keys} among {columns}: examined {examined}"
            )
            searched += f", skipped {skipped:,} supersets of keys." if skipped else "."
        else:
            searched = (
                f"Not exhaustive: max_combinations stopped the search for {keys} after "
                f"{search.examined_count:,} of up to {search.combination_count:,} sets."
            )
        facts = [
            searched,
            "Missing values compare equal; a key with missing values can't serve as a "
            "primary key.",
        ]
        if self.left_out_columns:
            names = _text.join_labels(self.left_out_columns)
            facts.append(f"Left out, values can't be hashed: {names}.")
        return facts

    def _text_table(self) -> _text.Table | None:
        if not self.search.examined_count:
            return None
        found = self.keys or self.keys_with_missing_values
        return _text.Table(
            title="Keys" if found else "Most unique column sets",
            unit=("column set", "column sets"),
            headers=(
                "columns",
                "kind",
                "distinct_count",
                "duplicate_row_count",
                "missing_row_count",
                "uniqueness_share",
            ),
            rows=[
                (
                    ", ".join(_text.label_text(name) for name in candidate.columns),
                    kind,
                    candidate.distinct_count,
                    candidate.duplicate_row_count,
                    candidate.missing_row_count,
                    candidate.uniqueness_share,
                )
                for kind, candidate in self._candidates()
            ],
        )

    def to_frame(self) -> pd.DataFrame:
        """Every reported column set, one row each: keys, keys with missing values,
        then the most unique sets, with `kind` naming the list each comes from.
        """
        candidates = self._candidates()
        frame = _entries_frame([candidate for _, candidate in candidates], CandidateKey)
        frame.insert(
            1, "kind", pd.Series([kind for kind, _ in candidates], dtype=object)
        )
        return frame


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
        return facts

    def _text_other_tables(self) -> list[str]:
        size = len(self.columns)
        return [f"res.to_frame(table='matrix') has the wide {size:,} x {size:,} matrix"]

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
                for p in self.pairs[: _text.MAX_TABLE_ROWS]
            ],
            total=len(self.pairs),
        )


class LogicalTypeCandidate(ResultModel):
    """A logical type a column may hold, with its score and the reasons."""

    logical_type: LogicalType = Field(description="The logical type.")
    score: float = Field(
        description="Heuristic score from 0 to 1 by fixed, documented rules; "
        "higher fits better. It isn't a probability."
    )
    reasons: list[str] = Field(
        description="Why the type scored this, one sentence per rule that "
        "applied, from fixed templates that show shares and counts, never values."
    )


class ValueTypeCounts(ResultModel):
    """How many of a column's non-missing values have each Python type."""

    boolean: int = Field(description="Values that are booleans.")
    number: int = Field(
        description="Values that are numbers: int, float, Decimal, Fraction or "
        "NumPy numbers, not booleans."
    )
    text: int = Field(description="Values that are strings, whatever they hold.")
    datetime: int = Field(
        description="Values that are datetimes or dates, pd.Timestamp included."
    )
    timedelta: int = Field(description="Values that are durations.")
    other: int = Field(
        description="Values of any other type, such as lists, dicts or periods."
    )


class ColumnEvidence(ResultModel):
    """The counts and shares a column's candidates and flags rest on.

    Every share is of the non-missing values and is null when there are none.
    Text is parsed once per distinct value, weighted by how often it occurs.
    """

    non_missing_count: int = Field(
        description="Values that aren't missing: None, NaN, NaT and pd.NA are "
        "missing, as pandas isna() counts them, and so is NaN that a float "
        "column stores as a value."
    )
    distinct_count: int | None = Field(
        description="Distinct non-missing values, as pandas counts them, so 1, "
        "1.0 and True are one value; null when values such as lists can't be "
        "hashed."
    )
    uniqueness_share: float | None = Field(
        description="distinct_count / non_missing_count: 1 when every value is "
        "distinct."
    )
    boolean_share: float | None = Field(
        description="Values that are booleans or read as one: numbers equal to 0 "
        "or 1, and the words true, false, t, f, yes, no, y and n in any case."
    )
    number_share: float | None = Field(
        description="Values that are numbers or text that parses as a decimal "
        "number, such as '42', '-3.5' or '1e-3'."
    )
    integer_share: float | None = Field(
        description="Values that are or parse as whole numbers, such as 42, 42.0 "
        "or '4.2e1'."
    )
    datetime_share: float | None = Field(
        description="Values that are datetimes or dates, or text that parses as "
        "an ISO 8601 date, with or without a time."
    )
    timedelta_share: float | None = Field(description="Values that are durations.")
    leading_zero_share: float | None = Field(
        description="Text values that are whole numbers written with a leading "
        "zero, such as '007': codes, which don't parse as numbers."
    )
    non_iso_date_share: float | None = Field(
        description="Text values that look like dates without being ISO 8601, "
        "such as '01/02/2020' or '5 Jan 2020'; they are never parsed."
    )
    plain_text_share: float | None = Field(
        description="Text values that parse as no number, ISO 8601 date or "
        "boolean; leading zeros and date-like text included. With leading zeros "
        "in a column, its numbers count as plain text too."
    )
    multi_word_share: float | None = Field(
        description="Plain text values with a space between other characters."
    )
    value_types: list[str] = Field(
        description="The names of the Python types of the non-missing values, "
        "sorted, as summarize_categorical reports them, such as ['int', 'str']; "
        "more than one flags mixed_value_types."
    )
    value_type_counts: ValueTypeCounts = Field(
        description="The non-missing values by the kind of value the scores read: "
        "booleans, numbers, text, datetimes, durations and others."
    )


class InferColumnSemanticsColumn(ColumnResult):
    """What infer_column_semantics found in one column."""

    dtype: str = Field(description="The dtype as pandas reports it, such as 'str'.")
    inferred_logical_type: LogicalType | None = Field(
        description="The top candidate's logical type; null when no candidate "
        "scores above 0. A guess from the values, never a declaration."
    )
    declared_logical_type: LogicalType | None = Field(
        description="The logical type declared_schema gives the column; null when "
        "it declares none. Inference never overwrites it."
    )
    matches_declared: bool | None = Field(
        description="Whether inferred_logical_type equals declared_logical_type; "
        "null when either is null."
    )
    candidates: list[LogicalTypeCandidate] = Field(
        description="The logical types with a score above 0, best first, ties in "
        "the order boolean, integer, float, datetime, timedelta, categorical, "
        "text. Integer and float also appear, scored 0, when leading zeros rule "
        "them out."
    )
    flags: list[ColumnFlag] = Field(
        description="What needs attention, in a fixed order: identifier_like, "
        "constant, all_missing, numbers_as_text, dates_as_text, "
        "non_iso_date_text, boolean_like and mixed_value_types."
    )
    evidence: ColumnEvidence = Field(
        description="The counts and shares the candidates and flags rest on."
    )


_FLAGS: tuple[ColumnFlag, ...] = typing.get_args(ColumnFlag)


class InferColumnSemanticsResult(AnalysisResult):
    """The result of infer_column_semantics."""

    columns: Annotated[dict[str, InferColumnSemanticsColumn], _ENTRIES] = Field(
        description="One entry per selected column, keyed by column name, in the "
        "selected order. The record keeps the first 100 and adds columns_total; "
        "to_frame() has every column."
    )
    missing_declared_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns declared_schema names that the data doesn't have, in "
        "declared order; empty without declared_schema."
    )

    def _text_facts(self) -> list[str]:
        facts = [
            "Scores rank logical types by fixed rules and aren't probabilities. "
            "Dates parse as ISO 8601 only."
        ]
        if self.parameters.get("declared_schema") is None:
            facts.append(
                "No declared_schema was given, so no column was compared with a "
                "declared logical type."
            )
        else:
            entries = self.columns.values()
            declared = sum(e.declared_logical_type is not None for e in entries)
            matching = sum(e.matches_declared is True for e in entries)
            differing = sum(e.matches_declared is False for e in entries)
            missing = len(self.missing_declared_columns)
            facts.append(
                f"declared_schema: {_text.plural(declared, 'column', 'columns')} "
                f"declared in the data, {matching:,} with that top candidate, "
                f"{differing:,} with another; {missing:,} not in the data."
            )
        return facts

    def _text_other_tables(self) -> list[str]:
        return ["res.to_frame(table='candidates') has every candidate with its reasons"]

    def _text_table(self) -> _text.Table:
        declared = self.parameters.get("declared_schema") is not None
        headers = ["column", "inferred", "score", "next", "next_score"]
        if declared:
            headers.append("declared")
        headers.append("flags")
        rows = []
        for entry in self.columns.values():
            top = entry.candidates[0] if entry.candidates else None
            runner_up = entry.candidates[1] if len(entry.candidates) > 1 else None
            row: list[_text.Cell] = [
                entry.column,
                top.logical_type if top else None,
                top.score if top else None,
                runner_up.logical_type if runner_up else None,
                runner_up.score if runner_up else None,
            ]
            if declared:
                row.append(entry.declared_logical_type)
            row.append(", ".join(entry.flags))
            rows.append(row)
        return _text.Table(
            title="Columns", unit=("column", "columns"), headers=headers, rows=rows
        )

    def to_frame(
        self, table: Literal["columns", "candidates"] = "columns"
    ) -> pd.DataFrame:
        """One of the result's two tables as a DataFrame.

        Args:
            table: 'columns', the default, gives one row per column in selected
                order: its status, dtype, inferred and declared logical types,
                the top score, one boolean column per flag and the evidence,
                with value_types joined by ", " and the value type counts as
                boolean_values, number_values and so on. 'candidates' gives one
                row per candidate: the column, its rank from 1, the logical
                type, the score and the reasons.
        """
        _check_table(table, ("columns", "candidates"))
        if table == "candidates":
            return self._candidates_frame()
        entries = list(self.columns.values())
        frame: dict[str, pd.Series] = {
            "column": pd.Series([e.column for e in entries], dtype=object),
            "status": pd.Series([e.status for e in entries], dtype=object),
            "status_reason": pd.Series(
                [e.status_reason.code if e.status_reason else None for e in entries],
                dtype=object,
            ),
            "dtype": pd.Series([e.dtype for e in entries], dtype=object),
            "inferred_logical_type": pd.Series(
                [e.inferred_logical_type for e in entries], dtype=object
            ),
            "score": pd.Series(
                [e.candidates[0].score if e.candidates else None for e in entries],
                dtype="float64",
            ),
            "declared_logical_type": pd.Series(
                [e.declared_logical_type for e in entries], dtype=object
            ),
            "matches_declared": pd.Series(
                [e.matches_declared for e in entries], dtype=object
            ),
        }
        for flag in _FLAGS:
            frame[flag] = pd.Series([flag in e.flags for e in entries], dtype="bool")
        for name, field in ColumnEvidence.model_fields.items():
            if name == "value_types":
                frame[name] = pd.Series(
                    [", ".join(e.evidence.value_types) for e in entries], dtype=object
                )
            elif name != "value_type_counts":
                frame[name] = pd.Series(
                    [getattr(e.evidence, name) for e in entries],
                    dtype=_FRAME_DTYPES.get(field.annotation),
                )
        for kind in ValueTypeCounts.model_fields:
            frame[f"{kind}_values"] = pd.Series(
                [getattr(e.evidence.value_type_counts, kind) for e in entries],
                dtype="int64",
            )
        return pd.DataFrame(frame, columns=list(frame))

    def _candidates_frame(self) -> pd.DataFrame:
        rows = [
            (entry.column, rank, c.logical_type, c.score, " ".join(c.reasons))
            for entry in self.columns.values()
            for rank, c in enumerate(entry.candidates, start=1)
        ]
        names = ["column", "rank", "logical_type", "score", "reasons"]
        dtypes = [object, "int64", object, "float64", object]
        return pd.DataFrame(
            {
                name: pd.Series([row[i] for row in rows], dtype=dtype)
                for i, (name, dtype) in enumerate(zip(names, dtypes, strict=True))
            },
            columns=names,
        )
