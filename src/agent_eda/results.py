"""Result classes returned by agent-eda analyses."""

import itertools
import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

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


# Nullable integers keep their type in to_frame(); other fields use pandas' default.
_FRAME_DTYPES: dict[Any, str] = {int: "int64", int | None: "Int64", bool: "bool"}


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
        if (self.status == "completed") != (self.status_reason is None):
            raise ValueError(
                "status_reason must be null when status is completed and set otherwise"
            )
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


class AnalysisListing(ResultModel):
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

    def __repr__(self) -> str:
        return self.__str__()

    def to_frame(self) -> pd.DataFrame:
        """One row per analysis, with every recorded field as a column."""
        return _entries_frame(self.analyses, AnalysisInfo)


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
        "column holds. Excludes pandas' lookup tables."
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
        description="Bytes the index labels use. Excludes pandas' lookup tables."
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
