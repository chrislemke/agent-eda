"""Analyses of what kind of values each column holds, and declarations of it."""

import inspect
import math
import re
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_eda._columns import ColumnKind, ColumnRule, closest_names, select_columns
from agent_eda._parse import (
    VALUE_KINDS,
    ValueKind,
    is_whole,
    is_zero_or_one,
    read_texts,
    value_kind,
)
from agent_eda._registry import (
    analysis,
    invalid_message,
    python_scalar,
    unknown_name_message,
)
from agent_eda._run import data_shape, execution, overall_status
from agent_eda._text import join_alternatives, join_items, join_labels, plural
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.results import (
    Assessment,
    ColumnEvidence,
    ColumnFlag,
    ColumnSupport,
    Finding,
    InferColumnSemanticsColumn,
    InferColumnSemanticsResult,
    Limitation,
    LogicalType,
    LogicalTypeCandidate,
    StatusReason,
    Support,
    ValueTypeCounts,
)

_LOGICAL_TYPES = typing.get_args(LogicalType)

# The directives datetime.strptime knows, as %Y or %:z; %% is a literal %.
_DIRECTIVE = re.compile(r"%(:z|.?)", re.DOTALL)
_KNOWN_DIRECTIVES = frozenset("aAwdbBmyYHIpMSfzZjUWcxXGuV%") | {":z"}


class ColumnSpec(BaseModel):
    """What a column is declared to hold: its logical type and more.

    A declared schema maps column names to a ColumnSpec or to a logical type
    alone, which is short for ColumnSpec(logical_type=...). infer_column_semantics
    compares what it infers with the declared logical type, and never
    overwrites it. Fields other than logical_type are optional, and None
    declares nothing, so no check depends on them. An invalid spec raises
    ae.errors.InvalidArgumentError.

    Examples:
        >>> import agent_eda as ae
        >>> ae.ColumnSpec("datetime", nullable=False, datetime_format="%d/%m/%Y")
        ColumnSpec(logical_type='datetime', nullable=False, datetime_format='%d/%m/%Y')
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

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
        "default, declares none, so only ISO 8601 text reads as a datetime.",
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
            fields = {name: python_scalar(value) for name, value in fields.items()}
            try:
                super().__init__(**fields)
            except pydantic.ValidationError as error:
                raise _spec_error(fields, error.errors()) from None

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
        try:
            return super().model_validate(obj, *args, **kwargs)
        except pydantic.ValidationError as error:
            raise _spec_error(dict(obj), error.errors()) from None

    @model_validator(mode="after")
    def _check_datetime_format(self) -> Self:
        text = self.datetime_format
        if text is None:
            return self
        if self.logical_type != "datetime":
            raise InvalidArgumentError(
                f"ColumnSpec got datetime_format {text!r} with logical_type "
                f"{self.logical_type!r}, but a datetime_format applies only to "
                "logical_type 'datetime'. Leave datetime_format out, or declare "
                "logical_type='datetime'."
            )
        directives = _DIRECTIVE.findall(text)
        unknown = [d for d in directives if d not in _KNOWN_DIRECTIVES]
        if unknown or not [d for d in directives if d != "%"]:
            problem = (
                f" It has an unknown directive: %{unknown[0]}."
                if unknown
                else " It has no directive."
            )
            raise InvalidArgumentError(
                f"ColumnSpec got datetime_format {text!r}, but a datetime_format "
                "is a format for datetime.strptime, with directives such as %Y, "
                f"%m and %d, like '%d/%m/%Y'.{problem}"
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

_SPEC_FIELDS = tuple(ColumnSpec.model_fields)


def _spec_error(fields: dict[str, Any], problems: list[Any]) -> AgentEDAError:
    """The error to raise for a ColumnSpec that pydantic rejected."""
    for problem in problems:
        error = problem.get("ctx", {}).get("error")
        if isinstance(error, AgentEDAError):
            return error
    first = problems[0]
    if first["type"] == "extra_forbidden":
        name = str(first["loc"][0])
        return InvalidArgumentError(
            unknown_name_message("ColumnSpec", "field", name, _SPEC_FIELDS),
            code="invalid_call",
        )
    if first["type"] == "missing":
        allowed = join_alternatives([repr(t) for t in _LOGICAL_TYPES])
        return InvalidArgumentError(
            f"ColumnSpec needs logical_type, one of {allowed}, such as "
            "ae.ColumnSpec(logical_type='integer', nullable=False).",
            code="invalid_call",
        )
    annotations = {
        name: spec_field.annotation
        for name, spec_field in ColumnSpec.model_fields.items()
    }
    return InvalidArgumentError(
        invalid_message("ColumnSpec", annotations, fields, problems, noun="field")
    )


# Every column can be read for what it holds.
_EVERY_KIND: frozenset[ColumnKind] = frozenset(typing.get_args(ColumnKind))
_EVERY_COLUMN = ColumnRule(
    default=_EVERY_KIND,
    default_text="every column",
    accepted=_EVERY_KIND,
    accepted_text="columns of every dtype",
    fixes={},
)

# Equal scores rank the more specific logical type first.
_TIE_ORDER: tuple[LogicalType, ...] = (
    "boolean",
    "integer",
    "float",
    "datetime",
    "timedelta",
    "categorical",
    "text",
)


@analysis(
    statuses=("completed", "partial", "insufficient_data", "not_applicable", "failed"),
    finding_codes=(
        "identifier_like_columns",
        "constant_columns",
        "all_missing_columns",
        "numbers_as_text",
        "dates_as_text",
        "non_iso_date_text",
        "boolean_like_columns",
        "mixed_value_types",
        "declared_type_mismatch",
        "declared_type_match",
        "declared_type_unchecked",
        "declared_columns_missing",
    ),
    tags=(
        "logical type",
        "semantic type",
        "kind",
        "identifier",
        "numbers as text",
        "dates as text",
        "postal code",
    ),
)
def infer_column_semantics(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    declared_schema: Mapping[str, LogicalType | ColumnSpec] | None = None,
) -> InferColumnSemanticsResult:
    """What kind of values does each column hold?

    When to use:
        Call it right after `describe_dataset`, before you summarize or model,
        to learn each column's logical type from its values rather than its
        dtype: numbers stored as text, identifiers that look numeric, postal
        codes and other codes with leading zeros, ISO 8601 dates stored as
        text, booleans such as yes/no or 0/1, a few labels repeated over many
        rows, and free text. Each column gets candidate logical types ranked by
        a heuristic score, the evidence behind them and flags for what needs
        attention. Pass the logical types you expect as declared_schema to
        compare each with what the values show; a declaration is never
        overwritten. For how the values of each numeric column are
        distributed, call `summarize_numeric` next; `list_analyses` lists the
        other analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to read, in the
            order to report them: a list, a tuple, a pandas Index such as
            df.columns[:3], or a single name. A label that isn't a string can
            be given as itself, such as 0, or as its string form. Columns of
            every dtype are accepted. None, the default, selects every column,
            in DataFrame order.
        declared_schema (Mapping[str, str | ColumnSpec] | None): The logical
            types you declare, by column name: 'boolean', 'integer', 'float',
            'categorical', 'text', 'datetime' or 'timedelta', or a
            ColumnSpec (a dict of its fields works too), whose logical_type is
            the one compared. Each declared column's top candidate is compared
            with its declaration, which is never overwritten. A declared column
            the data lacks is reported as a violation, and declarations for
            columns that columns leaves out are ignored. A label that isn't a
            string can be given as itself or as its string form. None, the
            default, declares nothing, so no comparison runs.

    Returns:
        InferColumnSemanticsResult: `columns` maps each selected column name to
        its entry, such as res.columns["zip"]. An entry holds:

        - `inferred_logical_type`, the top candidate: a guess from the values,
          null when no candidate scores above 0.
        - `candidates`, the logical types scoring above 0, best first, each
          with its `score` and `reasons`.
        - `flags`, what needs attention, each explained by a finding below.
        - `declared_logical_type` and `matches_declared`, when declared_schema
          declares the column.
        - `evidence`: `non_missing_count`, `distinct_count`,
          `uniqueness_share`, the shares of values that read as booleans,
          numbers, whole numbers, datetimes and durations, and the shares of
          leading zeros, date-like text, plain text and text of several words.
        - `status` and `status_reason`, the column's own.

        Scores lie between 0 and 1 and come from fixed rules over the evidence;
        they are heuristic, not probabilities. Every share is of the
        non-missing values. boolean scores boolean_share: booleans, numbers
        equal to 0 or 1, and true, false, t, f, yes, no, y and n in any case.
        integer scores integer_share, the values that are or parse as whole
        numbers, and float number_share, the values that are or parse as
        decimal numbers. datetime scores datetime_share: datetimes, and text in
        ISO 8601, YYYY-MM-DD with an optional time and offset. No other date
        format parses, so day and month order is never guessed. timedelta
        scores timedelta_share. categorical scores 1 - uniqueness_share, high
        when a few values repeat over many rows. text scores (plain_text_share
        * uniqueness_share + multi_word_share) / 2, high for mostly distinct
        text with several words. When a text value is a whole number written
        with a leading zero, such as 01234, the column holds codes: integer and
        float score 0, and its numbers count as plain text. Ties go to
        boolean, then integer, float, datetime, timedelta, categorical and
        text. Text is parsed once per distinct value, weighted by how often it
        occurs. `to_frame()` returns every column's evidence and
        to_frame("candidates") every candidate with its reasons.

        A share is null for a column without values, with a `too_few_values`
        limitation. `unhashable_values` means values such as lists left the
        distinct count null, and `nan_counted_as_missing` that a float column
        stores NaN as a value, counted as missing.

        Findings name the columns with each flag, and compare declarations:

        - `identifier_like_columns`: suspected, flag identifier_like. Every
          value is distinct, there are at least two, and the top candidate is
          integer or text without spaces, as for IDs. Numbers stored as
          numbers must also run like a sequence: at least half the whole
          numbers from the smallest to the largest occur. Don't use these
          columns as numbers or features.
        - `constant_columns`: suspected, flag constant. A single distinct
          value in at least two rows.
        - `all_missing_columns`: suspected, flag all_missing. No value in at
          least one row.
        - `numbers_as_text`: suspected, flag numbers_as_text. The top candidate
          is integer or float and some of the numbers are stored as text.
        - `dates_as_text`: suspected, flag dates_as_text. The top candidate is
          datetime and some of the datetimes are ISO 8601 text.
        - `non_iso_date_text`: suspected, flag non_iso_date_text. At least half
          the values look like dates, and some of them aren't ISO 8601, such as
          01/02/2020 or 5 Jan 2020; they are never parsed.
        - `boolean_like_columns`: descriptive, flag boolean_like. The top
          candidate is boolean and some values are numbers or text that read
          as booleans, such as 0/1 or yes/no.
        - `mixed_value_types`: suspected, flag mixed_value_types. The values
          have more than one Python type, such as numbers and strings.
        - `declared_type_mismatch`: suspected. A declared column's top
          candidate differs from its declared logical type.
        - `declared_type_match`: passed. A declared column's top candidate is
          its declared logical type.
        - `declared_type_unchecked`: unknown. A declared column has no
          candidate to compare, since it has no values or none fits.
        - `declared_columns_missing`: violation. declared_schema names columns
          the data doesn't have, listed in `missing_declared_columns`.

    Statuses:
        completed: Every selected column has values and a candidate scoring
            above 0.
        partial: Some selected columns have no values or no candidate; their
            entries have status insufficient_data or not_applicable.
        insufficient_data: The data has no rows, or no selected column has a
            value that isn't missing.
        not_applicable: The data has no columns, or no logical type fits any
            selected column: the values are of other types, such as lists or
            intervals, and aren't category labels repeated over rows.
        failed: An anticipated computation failure stopped the analysis, and
            `status_reason` gives its code and message. None is known today.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "zip": ["01234", "10115", "02134", "10115", "80331", "01234"],
        ...         "order_id": ["1001", "1002", "1003", "1004", "1005", "1006"],
        ...         "ordered": ["2024-01-05", "2024-01-06", "2024-02-01"] * 2,
        ...         "paid": ["yes", "no", "yes", "yes", "no", "yes"],
        ...         "amount": [12.0, 30.0, None, 7.0, 12.0, 45.0],
        ...     }
        ... )
        >>> res = ae.infer_column_semantics(df, declared_schema={"amount": "float"})
        >>> print(res)
        infer_column_semantics: completed on 6 rows x 5 columns
        Scores rank logical types by fixed rules and aren't probabilities. Dates parse as ISO 8601 only.
        declared_schema: 1 column declared in the data, 0 with that top candidate, 1 with another; 0 not in the data.
        res.to_frame() has each column's evidence; res.to_frame('candidates') has every candidate with its reasons.
        Columns (5 of 5 shown):
        column    inferred      score  next         next_score  declared  flags
        zip       categorical  0.3333  text             0.3333  null
        order_id  integer           1  float                 1  null      identifier_like, numbers_as_text
        ordered   datetime          1  categorical         0.5  null      dates_as_text
        paid      boolean           1  categorical      0.6667  null      boolean_like
        amount    integer           1  float                 1  float
        Findings (5 of 5 shown):
        - suspected identifier_like_columns: 1 column looks like an identifier, every value distinct: 'order_id'.
        - suspected numbers_as_text: 1 column stores numbers as text: 'order_id'.
        - suspected dates_as_text: 1 column stores ISO 8601 dates as text: 'ordered'.
        - descriptive boolean_like_columns: 1 column holds booleans stored as numbers or text: 'paid'.
        - suspected declared_type_mismatch: 1 column has a top candidate other than the declared logical type: 'amount'.
        Limitations: none
        More: res.to_frame() has all 5 columns; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["zip"].inferred_logical_type
        'categorical'
        >>> res.columns["zip"].evidence.leading_zero_share
        0.5

    See Also:
        describe_dataset: Dtypes, missing and distinct counts and memory of every
            column, whatever its dtype.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every
            numeric column.
        list_analyses: Every analysis, with the question each one answers.
    """
    selected = select_columns(
        "infer_column_semantics", data, columns, _EVERY_COLUMN
    ).columns
    row_count = len(data)
    declared = {
        name: spec if isinstance(spec, ColumnSpec) else ColumnSpec(logical_type=spec)
        for name, spec in (declared_schema or {}).items()
    }
    names = [str(label) for label in data.columns]
    known = set(names)
    missing_declared = [name for name in declared if name not in known]
    readings = {
        column.name: _read_column(data.iloc[:, column.position], column.kind)
        for column in selected
    }
    entries = {
        column.name: _entry(
            column.name,
            column.dtype,
            readings[column.name],
            row_count,
            declared.get(column.name),
        )
        for column in selected
    }
    status, status_reason = overall_status(
        row_count, list(entries.values()), no_columns=_NO_COLUMNS
    )
    return InferColumnSemanticsResult(
        analysis="infer_column_semantics",
        method=None,
        status=status,
        status_reason=status_reason,
        parameters={
            "data": data_shape(data),
            "columns": [column.name for column in selected],
            "declared_schema": None
            if declared_schema is None
            else {
                name: spec.model_dump(mode="json") for name, spec in declared.items()
            },
        },
        support=Support(
            row_count=row_count,
            column_count=len(selected),
            columns=[
                _support(name, reading, row_count) for name, reading in readings.items()
            ],
        ),
        findings=[
            *_flag_findings(entries),
            *_declared_findings(entries, missing_declared, names),
        ],
        limitations=_limitations(readings),
        execution=execution(("numpy", "pandas")),
        columns=entries,
        missing_declared_columns=missing_declared,
    )


_NO_COLUMNS = StatusReason(code="no_columns", message="The data has no columns.")
_NO_VALUES = StatusReason(
    code="no_values", message="The column has no values that aren't missing."
)
_UNSUPPORTED_VALUES = StatusReason(
    code="unsupported_values",
    message=(
        "No logical type fits: the values are of other types, such as lists or "
        "intervals, and aren't category labels repeated over rows."
    ),
)


@dataclass
class _Reading:
    """What a column's non-missing values are, and how many read as each type."""

    non_missing: int = 0
    # Distinct non-missing values; None when values can't be hashed.
    distinct: int | None = 0
    # NaN stored as a value in a float column, counted as missing.
    stored_nan: int = 0
    kinds: dict[ValueKind, int] = field(
        default_factory=lambda: dict.fromkeys(VALUE_KINDS, 0)
    )
    boolean: int = 0
    number: int = 0
    whole: int = 0
    datetime: int = 0
    timedelta: int = 0
    leading_zero: int = 0
    date_like: int = 0
    plain_text: int = 0
    several_words: int = 0
    # Text values that parse as numbers and as ISO 8601 dates.
    text_numbers: int = 0
    text_dates: int = 0
    # max - min + 1 of the values stored as numbers; None without any, or when
    # one isn't finite.
    span: float | None = None


def _read_column(series: pd.Series, kind: ColumnKind) -> _Reading:
    """Read what a column's values are, the way its dtype allows."""
    if kind == "numeric":
        return _read_numbers(series)
    if kind in ("boolean", "datetime", "timedelta"):
        # Every value is of the kind its dtype stores.
        non_missing = len(series) - int(series.isna().sum())
        reading = _Reading(non_missing=non_missing, distinct=int(series.nunique()))
        reading.kinds[kind] = non_missing
        setattr(reading, kind, non_missing)
        return reading
    if kind == "categorical":
        codes = series.cat.codes.to_numpy()
        present = codes >= 0
        categories = series.cat.categories
        counts = np.bincount(codes[present], minlength=len(categories))
        used = np.flatnonzero(counts)
        values = np.asarray(categories, dtype=object)[used]
        reading = _read_distinct(values, counts[used])
        reading.distinct = len(used)
        return reading
    if kind == "text":
        codes, uniques = pd.factorize(series)
        present = codes >= 0
        counts = np.bincount(codes[present], minlength=len(uniques))
        reading = _read_texts(np.asarray(uniques, dtype=object), counts)
        reading.distinct = len(uniques)
        return reading
    return _read_objects(np.asarray(series.array, dtype=object))


def _read_numbers(series: pd.Series) -> _Reading:
    """Read a column of numbers: integers, floats or decimals."""
    if series.dtype.kind in "iu":
        non_missing = len(series) - int(series.isna().sum())
        reading = _Reading(non_missing=non_missing, distinct=int(series.nunique()))
        reading.number = reading.whole = reading.kinds["number"] = non_missing
        reading.boolean = int(((series == 0) | (series == 1)).sum())
        if non_missing:
            reading.span = int(series.max()) - int(series.min()) + 1
        return reading
    floats = series.to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(floats)
    values = floats[~missing]
    finite = values[np.isfinite(values)]
    reading = _Reading(
        non_missing=len(values),
        distinct=len(pd.unique(values)),
        stored_nan=int(missing.sum()) - int(series.isna().sum()),
    )
    reading.number = reading.kinds["number"] = len(values)
    reading.whole = int(np.count_nonzero(finite == np.floor(finite)))
    reading.boolean = int(np.count_nonzero((values == 0) | (values == 1)))
    if len(values) and len(finite) == len(values):
        reading.span = float(finite.max()) - float(finite.min()) + 1
    return reading


def _read_objects(objects: np.ndarray) -> _Reading:
    """Read a column of Python objects, whatever their types."""
    try:
        codes, found = pd.factorize(objects)
    except TypeError:  # values such as lists can't be hashed
        return _read_unhashable(objects)
    uniques = np.asarray(found, dtype=object)
    present = codes >= 0
    counts = np.bincount(codes[present], minlength=len(uniques))
    if _all_text(uniques):
        reading = _read_texts(uniques, counts)
        reading.distinct = len(uniques)
        return reading
    # pandas counts 1, 1.0 and True as one value, so the values behind a
    # boolean or a number equal to 0 or 1 are read one by one.
    ambiguous = np.array(
        [_reads_as_number_or_boolean(value) for value in uniques], dtype=bool
    )
    reading = _read_distinct(uniques[~ambiguous], counts[~ambiguous])
    reading.distinct = len(uniques)
    reading.span = _number_span(uniques)
    if ambiguous.any():
        positions = np.flatnonzero(np.isin(codes, np.flatnonzero(ambiguous)))
        booleans = sum(
            isinstance(objects[position], bool | np.bool_)
            for position in positions.tolist()
        )
        numbers = len(positions) - booleans
        reading.non_missing += len(positions)
        reading.kinds["boolean"] += booleans
        reading.kinds["number"] += numbers
        reading.boolean += len(positions)
        reading.number += numbers
        reading.whole += numbers
    return reading


def _reads_as_number_or_boolean(value: object) -> bool:
    """Whether a value is a boolean, or a number equal to 0 or 1."""
    kind = value_kind(value)
    return kind == "boolean" or (kind == "number" and is_zero_or_one(value))


def _read_unhashable(objects: np.ndarray) -> _Reading:
    """Read Python objects some of which can't be hashed, so can't be counted."""
    present = objects[~pd.isna(objects)]
    texts = _objects([value for value in present if isinstance(value, str)])
    others = _objects([value for value in present if not isinstance(value, str)])
    codes, uniques = pd.factorize(texts)
    counts = np.bincount(codes, minlength=len(uniques))
    reading = _read_texts(np.asarray(uniques, dtype=object), counts)
    rest = _read_distinct(others, np.ones(len(others), dtype=np.int64))
    reading = _combined(reading, rest)
    reading.distinct = None
    reading.span = rest.span
    return reading


def _objects(values: list[object]) -> np.ndarray:
    """A 1-D object array of the values, even when they are lists themselves."""
    return np.fromiter(values, dtype=object, count=len(values))


def _all_text(values: np.ndarray) -> bool:
    """Whether every value of an object array is a string; True when it's empty."""
    return pd.api.types.infer_dtype(values, skipna=False) in ("string", "empty")


def _read_distinct(values: np.ndarray, counts: np.ndarray) -> _Reading:
    """Read distinct values of any type, each standing for `counts` values."""
    if _all_text(values):
        return _read_texts(values, counts)
    kinds = [value_kind(value) for value in values]
    is_text = np.array([kind == "text" for kind in kinds], dtype=bool)
    reading = _read_texts(values[is_text], counts[is_text])
    for value, kind, count in zip(values, kinds, counts.tolist(), strict=True):
        if kind == "text":
            continue
        reading.non_missing += count
        reading.kinds[kind] += count
        if kind == "boolean":
            reading.boolean += count
        elif kind == "number":
            reading.number += count
            reading.whole += count if is_whole(value) else 0
            reading.boolean += count if is_zero_or_one(value) else 0
        elif kind == "datetime":
            reading.datetime += count
        elif kind == "timedelta":
            reading.timedelta += count
    reading.span = _number_span(values)
    return reading


def _number_span(values: Iterable[object]) -> float | None:
    """max - min + 1 of the numbers among the values; None without one or with a
    number that isn't finite."""
    try:
        numbers = [
            float(typing.cast("Any", v)) for v in values if value_kind(v) == "number"
        ]
    except OverflowError:  # an int too large for a float
        return None
    if not numbers or not all(map(math.isfinite, numbers)):
        return None
    return max(numbers) - min(numbers) + 1


def _read_texts(texts: np.ndarray, counts: np.ndarray) -> _Reading:
    """Read distinct strings, each standing for `counts` values, by the parse rules."""
    parsed = read_texts(texts)

    def weight(mask: np.ndarray) -> int:
        return int(counts[mask].sum())

    total = int(counts.sum())
    leading_zero = weight(parsed.leading_zero)
    # With leading zeros, a column's numbers are codes: they count as text.
    readable = parsed.iso_date | parsed.boolean
    if not leading_zero:
        readable = readable | parsed.number
    reading = _Reading(
        non_missing=total,
        boolean=weight(parsed.boolean),
        number=weight(parsed.number),
        whole=weight(parsed.whole),
        datetime=weight(parsed.iso_date),
        leading_zero=leading_zero,
        date_like=weight(parsed.date_like),
        plain_text=weight(~readable),
        several_words=weight(~readable & parsed.several_words),
        text_numbers=weight(parsed.number),
        text_dates=weight(parsed.iso_date),
    )
    reading.kinds["text"] = total
    return reading


def _combined(first: _Reading, second: _Reading) -> _Reading:
    """The reading of two disjoint sets of values together."""
    combined = _Reading()
    for name in _COUNTS:
        setattr(combined, name, getattr(first, name) + getattr(second, name))
    combined.kinds = {
        kind: first.kinds[kind] + second.kinds[kind] for kind in VALUE_KINDS
    }
    return combined


_COUNTS = (
    "non_missing",
    "boolean",
    "number",
    "whole",
    "datetime",
    "timedelta",
    "leading_zero",
    "date_like",
    "plain_text",
    "several_words",
    "text_numbers",
    "text_dates",
)


def _entry(
    name: str,
    dtype: str,
    reading: _Reading,
    row_count: int,
    declared: ColumnSpec | None,
) -> InferColumnSemanticsColumn:
    candidates = _candidates(reading) if reading.non_missing else []
    top = candidates[0] if candidates and candidates[0].score > 0 else None
    inferred = top.logical_type if top else None
    if not reading.non_missing:
        status, reason = "insufficient_data", _NO_VALUES
    elif top is None:
        status, reason = "not_applicable", _UNSUPPORTED_VALUES
    else:
        status, reason = "completed", None
    declared_type = declared.logical_type if declared else None
    return InferColumnSemanticsColumn(
        column=name,
        status=status,
        status_reason=reason,
        dtype=dtype,
        inferred_logical_type=inferred,
        declared_logical_type=declared_type,
        matches_declared=None
        if declared_type is None or inferred is None
        else inferred == declared_type,
        candidates=candidates,
        flags=_flags(reading, inferred, row_count),
        evidence=_evidence(reading),
    )


def _candidates(reading: _Reading) -> list[LogicalTypeCandidate]:
    """The logical types the values fit, best first, by the documented rules."""
    count = reading.non_missing

    def share(part: int) -> float:
        return part / count

    found: dict[LogicalType, tuple[float, str]] = {}
    if reading.boolean:
        found["boolean"] = (
            share(reading.boolean),
            f"{_percent(share(reading.boolean))} of values are booleans or read "
            "as one.",
        )
    if reading.leading_zero:
        codes = (
            f"{plural(reading.leading_zero, 'value is a whole number', 'values are whole numbers')} "
            "written with a leading zero, which a number would drop, so the "
            "column holds codes."
        )
        found["integer"] = found["float"] = (0.0, codes)
    else:
        if reading.whole:
            found["integer"] = (
                share(reading.whole),
                f"{_percent(share(reading.whole))} of values are or parse as "
                "whole numbers.",
            )
        if reading.number:
            found["float"] = (
                share(reading.number),
                f"{_percent(share(reading.number))} of values are or parse as numbers.",
            )
    if reading.datetime:
        found["datetime"] = (
            share(reading.datetime),
            f"{_percent(share(reading.datetime))} of values are datetimes or "
            "ISO 8601 dates.",
        )
    if reading.timedelta:
        found["timedelta"] = (
            share(reading.timedelta),
            f"{_percent(share(reading.timedelta))} of values are durations.",
        )
    if reading.distinct is not None:
        uniqueness = share(reading.distinct)
        if uniqueness < 1:
            found["categorical"] = (
                1 - uniqueness,
                f"1 - uniqueness share: {plural(reading.distinct, 'distinct value', 'distinct values')} "
                f"over {count:,} values.",
            )
        text = (
            share(reading.plain_text) * uniqueness + share(reading.several_words)
        ) / 2
        if text > 0:
            found["text"] = (
                text,
                f"(plain text share {_share(share(reading.plain_text))} x "
                f"uniqueness share {_share(uniqueness)} + multi-word share "
                f"{_share(share(reading.several_words))}) / 2.",
            )
    ranked = sorted(
        found.items(), key=lambda item: (-item[1][0], _TIE_ORDER.index(item[0]))
    )
    return [
        LogicalTypeCandidate(logical_type=logical_type, score=score, reasons=[reason])
        for logical_type, (score, reason) in ranked
    ]


def _percent(share: float) -> str:
    return f"{share * 100:.4g}%"


def _share(share: float) -> str:
    return f"{share:.4g}"


def _flags(
    reading: _Reading, inferred: LogicalType | None, row_count: int
) -> list[ColumnFlag]:
    """The flags a column's reading and top candidate raise, in fixed order."""
    count = reading.non_missing
    # Numbers stored as numbers look like identifiers when they run like a
    # sequence: at least half the whole numbers between the smallest and the
    # largest occur. Text needs no such shape.
    sequence = reading.kinds["text"] > 0 or (
        reading.span is not None and reading.span <= 2 * count
    )
    raised: dict[ColumnFlag, bool] = {
        "identifier_like": count >= 2
        and reading.distinct == count
        and inferred in ("integer", "text")
        and not reading.several_words
        and sequence,
        "constant": count >= 2 and reading.distinct == 1,
        "all_missing": row_count >= 1 and count == 0,
        "numbers_as_text": inferred in ("integer", "float")
        and reading.text_numbers > 0,
        "dates_as_text": inferred == "datetime" and reading.text_dates > 0,
        "non_iso_date_text": reading.date_like > 0
        and 2 * (reading.datetime + reading.date_like) >= count,
        "boolean_like": inferred == "boolean"
        and reading.boolean > reading.kinds["boolean"],
        "mixed_value_types": sum(1 for n in reading.kinds.values() if n) > 1,
    }
    return [flag for flag, on in raised.items() if on]


def _evidence(reading: _Reading) -> ColumnEvidence:
    count = reading.non_missing

    def share(part: int | None) -> float | None:
        return None if part is None or not count else part / count

    return ColumnEvidence(
        non_missing_count=count,
        distinct_count=reading.distinct,
        uniqueness_share=share(reading.distinct),
        boolean_share=share(reading.boolean),
        number_share=share(reading.number),
        integer_share=share(reading.whole),
        datetime_share=share(reading.datetime),
        timedelta_share=share(reading.timedelta),
        leading_zero_share=share(reading.leading_zero),
        non_iso_date_share=share(reading.date_like),
        plain_text_share=share(reading.plain_text),
        multi_word_share=share(reading.several_words),
        value_type_counts=ValueTypeCounts(**reading.kinds),
    )


def _support(name: str, reading: _Reading, row_count: int) -> ColumnSupport:
    missing = row_count - reading.non_missing
    return ColumnSupport(
        column=name,
        eligible_count=row_count,
        evaluated_count=reading.non_missing,
        excluded_count=missing,
        excluded_by_reason={"missing": missing} if missing else {},
    )


def _path(name: str) -> str:
    """The result path of a column's entry, such as "columns['zip']"."""
    return f"columns[{name!r}]"


@dataclass(frozen=True)
class _FlagFinding:
    """The finding that names the columns with a flag."""

    flag: ColumnFlag
    code: str
    assessment: Assessment
    # What the columns do, after "1 column" and after "2 columns".
    singular: str
    plural: str
    # The evidence field behind the flag.
    evidence: str


_FLAG_FINDINGS = (
    _FlagFinding(
        "identifier_like",
        "identifier_like_columns",
        "suspected",
        "looks like an identifier, every value distinct",
        "look like identifiers, every value distinct",
        "uniqueness_share",
    ),
    _FlagFinding(
        "constant",
        "constant_columns",
        "suspected",
        "holds a single distinct non-missing value",
        "hold a single distinct non-missing value",
        "distinct_count",
    ),
    _FlagFinding(
        "all_missing",
        "all_missing_columns",
        "suspected",
        "contains only missing values",
        "contain only missing values",
        "non_missing_count",
    ),
    _FlagFinding(
        "numbers_as_text",
        "numbers_as_text",
        "suspected",
        "stores numbers as text",
        "store numbers as text",
        "number_share",
    ),
    _FlagFinding(
        "dates_as_text",
        "dates_as_text",
        "suspected",
        "stores ISO 8601 dates as text",
        "store ISO 8601 dates as text",
        "datetime_share",
    ),
    _FlagFinding(
        "non_iso_date_text",
        "non_iso_date_text",
        "suspected",
        "holds dates that aren't ISO 8601, which are never parsed",
        "hold dates that aren't ISO 8601, which are never parsed",
        "non_iso_date_share",
    ),
    _FlagFinding(
        "boolean_like",
        "boolean_like_columns",
        "descriptive",
        "holds booleans stored as numbers or text",
        "hold booleans stored as numbers or text",
        "boolean_share",
    ),
    _FlagFinding(
        "mixed_value_types",
        "mixed_value_types",
        "suspected",
        "mixes values of different types",
        "mix values of different types",
        "value_type_counts",
    ),
)


def _flag_findings(entries: dict[str, InferColumnSemanticsColumn]) -> list[Finding]:
    findings = []
    for finding in _FLAG_FINDINGS:
        names = [name for name, entry in entries.items() if finding.flag in entry.flags]
        if not names:
            continue
        does = plural(
            len(names), f"column {finding.singular}", f"columns {finding.plural}"
        )
        findings.append(
            Finding(
                code=finding.code,
                claim=f"{does}: {join_labels(names)}.",
                assessment=finding.assessment,
                columns=names,
                fields=[f"{_path(name)}.evidence.{finding.evidence}" for name in names],
            )
        )
    return findings


def _declared_findings(
    entries: dict[str, InferColumnSemanticsColumn],
    missing: list[str],
    names: list[str],
) -> list[Finding]:
    findings = []
    declared = {
        name: entry
        for name, entry in entries.items()
        if entry.declared_logical_type is not None
    }
    mismatched = [n for n, e in declared.items() if e.matches_declared is False]
    if mismatched:
        findings.append(
            Finding(
                code="declared_type_mismatch",
                claim=f"{plural(len(mismatched), 'column has', 'columns have')} a "
                "top candidate other than the declared logical type: "
                f"{join_labels(mismatched)}.",
                assessment="suspected",
                columns=mismatched,
                fields=[
                    f"{_path(n)}.{part}"
                    for n in mismatched
                    for part in ("declared_logical_type", "inferred_logical_type")
                ],
            )
        )
    matched = [n for n, e in declared.items() if e.matches_declared is True]
    if matched:
        findings.append(
            Finding(
                code="declared_type_match",
                claim=f"{plural(len(matched), 'column has', 'columns have')} the "
                f"declared logical type as top candidate: {join_labels(matched)}.",
                assessment="passed",
                columns=matched,
                fields=[f"{_path(n)}.matches_declared" for n in matched],
            )
        )
    unchecked = [n for n, e in declared.items() if e.inferred_logical_type is None]
    if unchecked:
        findings.append(
            Finding(
                code="declared_type_unchecked",
                claim=f"{plural(len(unchecked), 'declared column has', 'declared columns have')} "
                f"no candidate to compare: {join_labels(unchecked)}.",
                assessment="unknown",
                columns=unchecked,
                fields=[f"{_path(n)}.inferred_logical_type" for n in unchecked],
            )
        )
    if missing:
        described = join_items(
            [
                f"{name!r} (did you mean {join_alternatives([repr(c) for c in close])}?)"
                if (close := closest_names(name, names))
                else repr(name)
                for name in missing
            ]
        )
        findings.append(
            Finding(
                code="declared_columns_missing",
                claim=f"{plural(len(missing), 'declared column is', 'declared columns are')} "
                f"not in the data: {described}.",
                assessment="violation",
                columns=missing,
                fields=["missing_declared_columns"],
            )
        )
    return findings


def _limitations(readings: dict[str, _Reading]) -> list[Limitation]:
    limitations = []
    empty = [name for name, reading in readings.items() if not reading.non_missing]
    if empty:
        limitations.append(
            Limitation(
                code="too_few_values",
                reason=f"{plural(len(empty), 'column has', 'columns have')} no values "
                f"that aren't missing, so the shares are null: {join_labels(empty)}.",
                fields=[f"{_path(name)}.evidence" for name in empty],
            )
        )
    unhashable = [name for name, r in readings.items() if r.distinct is None]
    if unhashable:
        hold = plural(len(unhashable), "column holds", "columns hold")
        limitations.append(
            Limitation(
                code="unhashable_values",
                reason=f"{hold} values that can't be hashed, such as lists or "
                "dicts, so distinct values weren't counted and categorical and "
                f"text weren't scored: {join_labels(unhashable)}.",
                fields=[
                    f"{_path(name)}.evidence.{part}"
                    for name in unhashable
                    for part in ("distinct_count", "uniqueness_share")
                ],
            )
        )
    stored = [name for name, r in readings.items() if r.stored_nan]
    if stored:
        store = plural(len(stored), "column stores", "columns store")
        limitations.append(
            Limitation(
                code="nan_counted_as_missing",
                reason=f"{store} NaN as a value, which pandas isna() doesn't count "
                f"as missing; non_missing_count leaves it out: {join_labels(stored)}.",
                fields=[f"{_path(name)}.evidence.non_missing_count" for name in stored],
            )
        )
    return limitations
