"""Analyses of what kind of values each column holds, and declarations of it."""

import inspect
import math
import re
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from agent_eda._columns import (
    Column,
    ColumnKind,
    ColumnRule,
    column_kind,
    select_columns,
)
from agent_eda._conversions import Check, check_column
from agent_eda._declarations import Declaration
from agent_eda._docstring import without_examples
from agent_eda._levels import (
    Levels,
    count_levels,
    ordered,
    record_label,
    type_name,
    value_types,
)
from agent_eda._masking import MaskedLevels, mask_levels, resolve_masking
from agent_eda._numeric import stored_nan_count, stored_nan_limitations
from agent_eda._parse import (
    VALUE_KINDS,
    ValueKind,
    is_whole,
    is_zero_or_one,
    read_datetime_format,
    read_texts,
    value_kind,
)
from agent_eda._registry import (
    analysis,
)
from agent_eda._result_base import (
    Assessment,
    ColumnFlag,
    ColumnSupport,
    Finding,
    Limitation,
    LogicalType,
    StatusReason,
    Support,
)
from agent_eda._row_groups import find_row_groups
from agent_eda._row_ids import RowIds, ambiguous_row_ids, read_row_ids
from agent_eda._run import NO_ROWS, data_shape, execution, overall_status
from agent_eda._text import (
    Surface,
    closest_names,
    join_alternatives,
    join_items,
    join_labels,
    placeholder,
    plural,
)
from agent_eda._values import is_constant, value_codes
from agent_eda._wording import as_written, worded
from agent_eda._written_form import row_id_list
from agent_eda.data._validation_results import (
    ColumnEvidence,
    InferColumnSemanticsColumn,
    InferColumnSemanticsResult,
    InvalidValue,
    LogicalTypeCandidate,
    ValidateColumnTypesColumn,
    ValidateColumnTypesResult,
    ValueTypeCounts,
)
from agent_eda.errors import AgentEDAError, InvalidArgumentError
from agent_eda.project import Project

_LOGICAL_TYPES = typing.get_args(LogicalType)

# The directives datetime.strptime knows, as %Y or %:z; %% is a literal %.
_DIRECTIVE = re.compile(r"%(:z|.?)", re.DOTALL)
_KNOWN_DIRECTIVES = frozenset("aAwdbBmyYHIpMSfzZjUWcxXGuV%") | {":z"}


class ColumnSpec(Declaration):
    """What a column is declared to hold: its logical type and more.

    A declared schema maps column names to a ColumnSpec or to a logical type
    alone, which is short for a ColumnSpec with only logical_type. Fields other
    than logical_type are optional, and None declares nothing, so no check
    depends on them. An invalid spec is rejected with the code invalid_argument.

    Examples:
        >>> import agent_eda as ae
        >>> ae.ColumnSpec("datetime", nullable=False, datetime_format="%d/%m/%Y")
        ColumnSpec(logical_type='datetime', nullable=False, datetime_format='%d/%m/%Y')
    """

    _example = "ae.ColumnSpec('integer', nullable=False)"
    _field_types: ClassVar[dict[str, object]] = {
        "logical_type": LogicalType,
        "nullable": bool | None,
        "datetime_format": str | None,
    }

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
        "default, declares none, so only ISO 8601 text reads as a datetime. "
        "infer_column_semantics and validate_column_types both read text in "
        "this format as datetimes. Directives that depend on the locale, such "
        "as %b and %p, read in the process's locale, so a process with another "
        "locale can count differently.",
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
            super().__init__(**fields)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """The JSON Schema, described without Examples:, which is Python.

        pydantic 2.7 adds the docstring after json_schema_extra runs, so the
        description is cut here, where every version has it.
        """
        schema = handler(core_schema)
        without_examples(handler.resolve_ref_schema(schema))
        return schema

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
        return super().model_validate(obj, *args, **kwargs)

    @classmethod
    def _rejected(cls, fields: Mapping[str, Any], problems: list[Any]) -> AgentEDAError:
        """A spec without a logical_type says how to write one for the surface."""
        first = problems[0]
        if (
            first["type"] == "missing"
            and not any(
                isinstance(problem.get("ctx", {}).get("error"), AgentEDAError)
                for problem in problems
            )
            and not any(problem["type"] == "extra_forbidden" for problem in problems)
        ):
            allowed = join_alternatives([repr(t) for t in _LOGICAL_TYPES])
            return worded(
                InvalidArgumentError,
                lambda surface: (
                    f"ColumnSpec needs logical_type, one of {allowed}, such "
                    f"as {_spec_example(surface)}."
                ),
                code="invalid_call",
            )
        return super()._rejected(fields, problems)

    @model_validator(mode="after")
    def _check_datetime_format(self) -> Self:
        text = self.datetime_format
        if text is None:
            return self
        if self.logical_type != "datetime":
            raise worded(
                InvalidArgumentError,
                lambda surface: (
                    f"ColumnSpec got datetime_format {text!r} with "
                    f"logical_type {self.logical_type!r}, but a datetime_format applies "
                    "only to logical_type 'datetime'. Leave datetime_format out, or "
                    f"declare {as_written(surface, logical_type='datetime')}."
                ),
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


def _spec_example(surface: Surface) -> str:
    """A ColumnSpec as the caller writes one."""
    fields = as_written(surface, logical_type="integer", nullable=False)
    return f"ae.ColumnSpec({fields})" if surface == "python" else fields


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
    statuses=("completed", "partial", "insufficient_data", "not_applicable"),
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
        "missing_declared_columns",
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
    schema: Mapping[str, LogicalType | ColumnSpec] | None = None,
) -> InferColumnSemanticsResult:
    """What kind of values does each column hold?

    When to use:
        Call it right after `describe_dataset`, before you summarize or model, to learn
        each column's logical type from its values rather than its dtype: numbers stored
        as text, identifiers that look numeric, postal codes and other codes with
        leading zeros, ISO 8601 dates stored as text, booleans such as yes/no or 0/1, a
        few labels repeated over many rows, and free text. Each column gets candidate
        logical types ranked by a heuristic score, the evidence behind them and flags
        for what needs attention. Pass the logical types you expect as schema to compare
        each with what the values show; a declaration is never overwritten. To test
        every value against the logical types you declare, call `validate_column_types`.
        For how numeric values are distributed, call `summarize_numeric` next.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        columns (Sequence[str] | None): Names of the columns to read, in the
            order to report them: a list or a single name. A label that isn't a string
            can be given as itself, such as 0, or as its string form. Columns of every
            dtype are accepted. None, the default, selects every column, in DataFrame
            order.
        schema (Mapping[str, str | ColumnSpec] | None): The logical
            types you declare, by column name: 'boolean', 'integer', 'float',
            'categorical', 'text', 'datetime' or 'timedelta', or a
            ColumnSpec (a mapping of its fields works too), whose logical_type is
            the one compared. Each declared column's top candidate is compared
            with its declaration, which is never overwritten: a declaration that
            ties with the top candidate matches. A ColumnSpec with a
            datetime_format, such as '%d/%m/%Y', also reads text in that format
            as datetimes, as validate_column_types does; its locale-dependent
            directives, such as %b, read in the process's locale. A declared
            column the data lacks is reported as a violation, and declarations
            for columns that columns leaves out are ignored. A label that isn't a
            string can be given as itself or as its string form. None, the
            default, declares nothing, so no comparison runs.

    Returns:
        InferColumnSemanticsResult: `columns` maps each selected column name to
        its entry: `inferred_logical_type` (null when no candidate scores above 0),
        the `candidates` best first with their `score` and `reasons`, `flags`,
        the `evidence` behind them and, for a column schema declares,
        `declared_logical_type` and `matches_declared` (true when it is the top
        candidate or ties with it). Scores run from 0 to 1 by fixed rules over
        the evidence and are heuristic, not probabilities. boolean scores 1 when
        every value reads as a boolean and the column holds at least two distinct
        values, otherwise it isn't a candidate; integer scores integer_share,
        float number_share, datetime datetime_share (ISO 8601 text or the
        ColumnSpec's datetime_format only, so day and month order is never
        guessed), timedelta timedelta_share, categorical 1 - uniqueness_share
        and text (plain_text_share * uniqueness_share + multi_word_share) / 2.
        Text that is a whole number with a leading zero, such as 01234, is a
        code: integer and float are listed too, scored 0, and it counts as plain
        text. A column where every candidate scores 0 has no inferred type. Equal
        scores tie, in the order boolean, integer, float, datetime, timedelta,
        categorical, text.

        A share is null for a column without values, with a `too_few_values`
        limitation; `unhashable_values` leaves the distinct count null, and
        `nan_counted_as_missing` counts a float column's NaN values as missing.

        Findings name the columns with each flag, and compare declarations:

        - `identifier_like_columns`: suspected; every value is distinct, there are at
          least two, and the top candidate is integer or text without spaces. Exact
          duplicate rows are set aside first for a column distinct in at least 90% of
          its rows. Date-like text never counts; numbers stored as numbers must also
          run like a sequence: at least half the whole numbers from the smallest to
          the largest occur. Don't use these columns as numbers or features.
        - `constant_columns`: suspected; a single distinct value in at least two rows;
          NaN is missing.
        - `all_missing_columns`: suspected; no value in at least one row.
        - `numbers_as_text`: suspected; the top candidate is integer or float and some
          of the numbers are stored as text.
        - `dates_as_text`: suspected; the top candidate is datetime and some of the
          datetimes are text, in ISO 8601 or the declared datetime_format.
        - `non_iso_date_text`: suspected; at least half the values look like dates, and
          some of them aren't ISO 8601, such as 01/02/2020; they are never parsed.
        - `boolean_like_columns`: descriptive; the top candidate is boolean and some
          values are numbers or text that read as booleans, such as 0/1 or yes/no.
        - `mixed_value_types`: suspected; the values have more than one Python type,
          such as int and str, as `evidence.value_types` lists.
        - `declared_type_mismatch`: suspected; a declared column's declared logical type
          scores below its top candidate, or isn't a candidate.
        - `declared_type_match`: passed; a declared column's top candidate, or a
          candidate tied with it, is its declared logical type.
        - `declared_type_unchecked`: unknown; a declared column has no candidate to
          compare, since it has no values or none fits.
        - `missing_declared_columns`: violation; schema names columns the data
          doesn't have, listed in `missing_declared_columns`.

    Statuses:
        completed: Every selected column has values and a candidate scoring
            above 0.
        partial: Some selected columns have no values or no candidate; their
            entries have status insufficient_data or not_applicable. Reason
            code: `incomplete_columns`.
        insufficient_data: The data has no rows, or no selected column
            completed and at least one of them has no value that isn't missing,
            even if the others hold values no logical type fits. Reason codes:
            `no_rows` and `no_values`.
        not_applicable: The data has no columns, or no logical type fits any
            selected column: the values are of other types, such as lists or
            intervals, and aren't category labels repeated over rows. Reason
            codes: `no_columns` and `unsupported_values`.

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
        >>> res = ae.infer_column_semantics(df, schema={"amount": "float"})
        >>> print(res)
        infer_column_semantics: completed on 6 rows x 5 columns
        Scores rank logical types by fixed rules and aren't probabilities. Dates parse as ISO 8601 only.
        schema: 1 column declared: 1 matching (ties with the top count), 0 not, 0 not inferred, 0 not in the data.
        Columns (5 of 5 shown):
        column    inferred      score  next         next_score  declared  flags
        order_id  integer           1  float                 1  null      identifier_like, numbers_as_text
        ordered   datetime          1  categorical         0.5  null      dates_as_text
        paid      boolean           1  categorical      0.6667  null      boolean_like
        amount    integer           1  float                 1  float
        zip       categorical  0.3333  text             0.3333  null
        Findings (5 of 5 shown):
        - suspected identifier_like_columns: 1 column looks like an identifier, every value distinct: 'order_id'.
        - suspected numbers_as_text: 1 column stores numbers as text: 'order_id'.
        - suspected dates_as_text: 1 column stores dates as text: 'ordered'.
        - descriptive boolean_like_columns: 1 column holds booleans stored as numbers or text: 'paid'.
        - passed declared_type_match: 1 column has the declared logical type as top candidate: 'amount'.
        Limitations: none
        More: res.to_frame() has all 5 columns; res.to_frame(table='candidates') has every candidate with its reasons;
            res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["zip"].inferred_logical_type
        'categorical'
        >>> res.columns["zip"].evidence.leading_zero_share
        0.5
        >>> schema = res.inferred_schema()
        >>> schema["order_id"], schema["ordered"]
        ('integer', 'datetime')
        >>> ae.validate_column_types(df, schema=schema).status
        'completed'

    See Also:
        validate_column_types: How many values of each declared column have, convert to
            or fail the logical type you declare, with the invalid ones.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every numeric
            column.
        compute_association_matrix: How strongly pairs of columns are associated,
            counting the numeric codes found here as categories.
    """
    selected = select_columns(
        "infer_column_semantics", data, columns, _EVERY_COLUMN
    ).columns
    row_count = len(data)
    declared = {
        name: spec if isinstance(spec, ColumnSpec) else ColumnSpec(logical_type=spec)
        for name, spec in (schema or {}).items()
    }
    names = [str(label) for label in data.columns]
    known = set(names)
    missing_declared = [name for name in declared if name not in known]
    readings = {
        column.name: _read_column(
            data.iloc[:, column.position],
            column.kind,
            declared[column.name].datetime_format if column.name in declared else None,
        )
        for column in selected
    }
    _judge_without_duplicate_rows(data, selected, readings)
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
            "schema": None
            if schema is None
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
        limitations=_limitations(readings, entries, schema is None),
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
    # The names of the values' Python types, as summarize_categorical reports
    # them; more than one means mixed value types.
    types: list[str] = field(default_factory=list)
    # When the values repeat only in rows that duplicate a whole other row: the
    # non-missing values left once those rows are dropped, all distinct.
    without_duplicate_rows: int | None = None


def _read_column(
    series: pd.Series, kind: ColumnKind, datetime_format: str | None = None
) -> _Reading:
    """Read what a column's values are, the way its dtype allows.

    Text in the declared datetime_format, when there is one, counts as datetimes.

    The distinct values and their counts are the column's levels, and the names
    of the values' types come from the same module, both as
    summarize_categorical counts them, so the two analyses can't disagree.
    """
    levels = count_levels(series, kind)
    if levels is None:  # values such as lists can't be hashed
        reading = _read_unhashable(
            np.asarray(series.array, dtype=object), datetime_format
        )
    elif kind == "numeric":
        reading = _read_numbers(series, levels)
    elif kind in ("boolean", "datetime", "timedelta"):
        # Every value is of the kind its dtype stores.
        non_missing = int(levels.counts.sum())
        reading = _Reading(non_missing=non_missing, distinct=len(levels.values))
        reading.kinds[kind] = non_missing
        setattr(reading, kind, non_missing)
    elif kind == "object":
        reading = _read_objects(series, levels, datetime_format)
    else:  # categorical, text and other dtypes, such as periods
        reading = _read_distinct(levels.objects(), levels.counts, datetime_format)
        reading.distinct = len(levels.values)
    reading.types = value_types(series, kind, levels)
    return reading


def _read_numbers(series: pd.Series, levels: Levels) -> _Reading:
    """Read a column of numbers: integers, floats or decimals."""
    if series.dtype.kind in "iu":
        non_missing = int(levels.counts.sum())
        reading = _Reading(non_missing=non_missing, distinct=len(levels.values))
        reading.number = reading.whole = reading.kinds["number"] = non_missing
        reading.boolean = int(((series == 0) | (series == 1)).sum())
        if non_missing:
            reading.span = int(series.max()) - int(series.min()) + 1
        return reading
    floats = series.to_numpy(dtype="float64", na_value=np.nan)
    values = floats[~np.isnan(floats)]
    finite = values[np.isfinite(values)]
    # NaN that the column stores as a value counts as missing here, but pandas
    # counts it as a level.
    nan_levels = (
        int(np.isnan(levels.sort_values).sum())
        if levels.sort_values is not None and levels.sort_values.dtype.kind == "f"
        else 0
    )
    reading = _Reading(
        non_missing=len(values),
        distinct=len(levels.values) - nan_levels,
        stored_nan=stored_nan_count(series),
    )
    reading.number = reading.kinds["number"] = len(values)
    reading.whole = int(np.count_nonzero(finite == np.floor(finite)))
    reading.boolean = int(np.count_nonzero((values == 0) | (values == 1)))
    if len(values) and len(finite) == len(values):
        reading.span = float(finite.max()) - float(finite.min()) + 1
    return reading


def _read_objects(
    series: pd.Series, levels: Levels, datetime_format: str | None
) -> _Reading:
    """Read a column of Python objects, whatever their types."""
    uniques = levels.objects()
    counts = levels.counts
    if levels.text:
        reading = _read_texts(uniques, counts, datetime_format)
        reading.distinct = len(uniques)
        return reading
    # pandas counts 1, 1.0 and True as one value, so the values behind a
    # boolean or a number equal to 0 or 1 are read one by one.
    ambiguous = np.array(
        [_reads_as_number_or_boolean(value) for value in uniques], dtype=bool
    )
    reading = _read_distinct(uniques[~ambiguous], counts[~ambiguous], datetime_format)
    reading.distinct = len(uniques)
    reading.span = _number_span(uniques)
    if ambiguous.any():
        # Every boolean stands behind such a level, and missing values aren't
        # booleans, so the booleans among all values are the ones it holds.
        behind = int(counts[ambiguous].sum())
        booleans = sum(
            isinstance(value, bool | np.bool_)
            for value in np.asarray(series.array, dtype=object)
        )
        numbers = behind - booleans
        reading.non_missing += behind
        reading.kinds["boolean"] += booleans
        reading.kinds["number"] += numbers
        reading.boolean += behind
        reading.number += numbers
        reading.whole += numbers
    return reading


def _reads_as_number_or_boolean(value: object) -> bool:
    """Whether a value is a boolean, or a number equal to 0 or 1."""
    kind = value_kind(value)
    return kind == "boolean" or (kind == "number" and is_zero_or_one(value))


def _read_unhashable(objects: np.ndarray, datetime_format: str | None) -> _Reading:
    """Read Python objects some of which can't be hashed, so can't be counted."""
    present = objects[~pd.isna(objects)]
    texts = _objects([value for value in present if isinstance(value, str)])
    others = _objects([value for value in present if not isinstance(value, str)])
    found = value_codes(texts, unhashable="by_content")
    codes, uniques = found.codes, found.uniques
    counts = np.bincount(codes, minlength=len(uniques))
    reading = _read_texts(np.asarray(uniques, dtype=object), counts, datetime_format)
    rest = _read_distinct(others, np.ones(len(others), dtype=np.int64), datetime_format)
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


def _read_distinct(
    values: np.ndarray, counts: np.ndarray, datetime_format: str | None
) -> _Reading:
    """Read distinct values of any type, each standing for `counts` values."""
    if _all_text(values):
        return _read_texts(values, counts, datetime_format)
    kinds = [value_kind(value) for value in values]
    is_text = np.array([kind == "text" for kind in kinds], dtype=bool)
    reading = _read_texts(values[is_text], counts[is_text], datetime_format)
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


def _read_texts(
    texts: np.ndarray, counts: np.ndarray, datetime_format: str | None
) -> _Reading:
    """Read distinct strings, each standing for `counts` values, by the parse rules.

    Text that datetime.strptime reads in the declared datetime_format, when
    there is one, is a datetime, and no longer date-like text.
    """
    parsed = read_texts(texts)
    iso_date, date_like = parsed.iso_date, parsed.date_like
    if datetime_format is not None and len(texts):
        iso_date = iso_date.copy()
        rest = np.flatnonzero(~iso_date)
        iso_date[rest] = read_datetime_format(texts[rest], datetime_format)
        date_like = date_like & ~iso_date

    def weight(mask: np.ndarray) -> int:
        return int(counts[mask].sum())

    total = int(counts.sum())
    leading_zero = weight(parsed.leading_zero)
    # With leading zeros, a column's numbers are codes: they count as text.
    readable = iso_date | parsed.boolean
    if not leading_zero:
        readable = readable | parsed.number
    reading = _Reading(
        non_missing=total,
        boolean=weight(parsed.boolean),
        number=weight(parsed.number),
        whole=weight(parsed.whole),
        datetime=weight(iso_date),
        leading_zero=leading_zero,
        date_like=weight(date_like),
        plain_text=weight(~readable),
        several_words=weight(~readable & parsed.several_words),
        text_numbers=weight(parsed.number),
        text_dates=weight(iso_date),
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


# A column whose values are distinct in at least this share of the rows can be
# an identifier that a few duplicate rows repeat; the rest is never read.
_NEAR_UNIQUE_SHARE = 0.9


def _judge_without_duplicate_rows(
    data: pd.DataFrame, selected: Sequence[Column], readings: dict[str, _Reading]
) -> None:
    """Judge the near-unique columns again once duplicate rows are dropped.

    A row that equals another in every column of the data adds no new value, so
    an identifier repeated by such rows is still an identifier. The columns are
    read only when some column could change its answer.
    """
    near = []
    for column in selected:
        reading = readings[column.name]
        if (
            reading.distinct is not None
            and reading.non_missing >= 2
            and reading.distinct < reading.non_missing
            and reading.distinct >= _NEAR_UNIQUE_SHARE * reading.non_missing
            and _candidates(reading)[0].logical_type in ("integer", "text")
        ):
            near.append(column)
    if not near:
        return
    first = [column.position for column in near]
    positions = first + [p for p in range(data.shape[1]) if p not in first]
    rows, groups = find_row_groups(data, positions)
    if not rows.size:
        return
    # Rows are in ascending order, so the first row of each group is its first.
    repeats = np.ones(len(rows), dtype=bool)
    repeats[np.unique(groups, return_index=True)[1]] = False
    keep = np.ones(len(data), dtype=bool)
    keep[rows[repeats]] = False
    for column in near:
        kept = _read_column(data.iloc[:, column.position].iloc[keep], column.kind)
        if kept.distinct == kept.non_missing:
            readings[column.name].without_duplicate_rows = kept.non_missing


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
        if declared_type is None or top is None
        else any(
            c.logical_type == declared_type and c.score == top.score for c in candidates
        ),
        candidates=candidates,
        flags=_flags(reading, inferred, row_count),
        evidence=_evidence(reading),
    )


def _candidates(reading: _Reading) -> list[LogicalTypeCandidate]:
    """The logical types the values fit, best first, by the documented rules."""
    count = reading.non_missing

    # Scores are exact fractions, so that scores equal on paper tie: 1 - 2/5 and
    # (1 * 2/5 + 4/5) / 2 are both 3/5, but not as floats.
    def share(part: int) -> Fraction:
        return Fraction(part, count)

    found: dict[LogicalType, tuple[Fraction, str]] = {}
    # Booleans need every value to read as one, so that letter codes such as
    # F and M aren't, and two distinct values, so that a constant column of
    # ones isn't, unless it is stored as booleans.
    if (
        reading.boolean == count
        and reading.distinct is not None
        and (reading.distinct >= 2 or reading.kinds["boolean"] == count)
    ):
        found["boolean"] = (
            Fraction(1),
            "All values are booleans or read as one.",
        )
    if reading.leading_zero:
        codes = (
            f"{plural(reading.leading_zero, 'value is a whole number', 'values are whole numbers')} "
            "written with a leading zero, which a number would drop, so the "
            "column holds codes."
        )
        found["integer"] = found["float"] = (Fraction(0), codes)
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
        LogicalTypeCandidate(
            logical_type=logical_type, score=float(score), reasons=[reason]
        )
        for logical_type, (score, reason) in ranked
    ]


def _percent(share: Fraction) -> str:
    return f"{float(share) * 100:.4g}%"


def _share(share: Fraction) -> str:
    return f"{float(share):.4g}"


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
    # Values that are all distinct, once the rows that duplicate a whole other
    # row are dropped.
    identifiers = count if reading.distinct == count else reading.without_duplicate_rows
    non_iso_dates = (
        reading.date_like > 0 and 2 * (reading.datetime + reading.date_like) >= count
    )
    raised: dict[ColumnFlag, bool] = {
        "identifier_like": identifiers is not None
        and identifiers >= 2
        and inferred in ("integer", "text")
        and not reading.several_words
        and not non_iso_dates
        and sequence,
        "constant": is_constant(count, reading.distinct),
        "all_missing": row_count >= 1 and count == 0,
        "numbers_as_text": inferred in ("integer", "float")
        and reading.text_numbers > 0,
        "dates_as_text": inferred == "datetime" and reading.text_dates > 0,
        "non_iso_date_text": non_iso_dates,
        "boolean_like": inferred == "boolean"
        and reading.boolean > reading.kinds["boolean"],
        # As summarize_categorical decides it: more than one Python type.
        "mixed_value_types": len(reading.types) > 1,
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
        value_types=reading.types,
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
        "stores dates as text",
        "store dates as text",
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
        "value_types",
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
        findings.append(
            Finding(
                code="missing_declared_columns",
                claim=_missing_columns_claim(missing, names),
                assessment="violation",
                columns=missing,
                fields=["missing_declared_columns"],
            )
        )
    return findings


def _missing_columns_claim(missing: list[str], names: list[str]) -> str:
    """The claim that declared columns aren't in the data, with the closest names.

    infer_column_semantics and validate_column_types both make it.
    """
    described = join_items(
        [
            f"{name!r} (did you mean {join_alternatives([repr(c) for c in close])}?)"
            if (close := closest_names(name, names))
            else repr(name)
            for name in missing
        ]
    )
    return (
        f"{plural(len(missing), 'declared column is', 'declared columns are')} "
        f"not in the data: {described}."
    )


def _limitations(
    readings: dict[str, _Reading],
    entries: dict[str, InferColumnSemanticsColumn],
    no_schema: bool,
) -> list[Limitation]:
    limitations = []
    if no_schema and entries:
        limitations.append(
            Limitation(
                code="schema_not_declared",
                reason=(
                    "No schema was declared, so no column was compared with a "
                    "declared logical type and matches_declared is null."
                ),
                # Every column, so none is named: a path per column would make
                # each look like the one with a limitation.
                fields=["columns"],
            )
        )
    untyped = [
        name for name, entry in entries.items() if entry.inferred_logical_type is None
    ]
    if untyped:
        has = plural(len(untyped), "column has", "columns have")
        limitations.append(
            Limitation(
                code="type_not_inferred",
                reason=(
                    f"{has} no inferred logical type, since no value is left after "
                    "the missing ones or no logical type fits the values: "
                    f"{join_labels(untyped)}."
                ),
                fields=[f"{_path(name)}.inferred_logical_type" for name in untyped],
            )
        )
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
    limitations += stored_nan_limitations(
        stored,
        [f"{_path(name)}.evidence.non_missing_count" for name in stored],
        "non_missing_count leaves it out",
    )
    return limitations


# The record keeps this many row IDs and example values, as the spec caps them.
_ROW_ID_LIMIT = 10
_EXAMPLE_LIMIT = 5

# validate_column_types' findings in the order it reports them: violations
# first, then what couldn't be checked, what converts and what held.
_FINDING_CODES = (
    "invalid_values",
    "missing_declared_columns",
    "non_nullable_missing_values",
    "unchecked_columns",
    "convertible_values",
    "passing_columns",
)


class _AtLeastOneColumn:
    """Marks a mapping that must declare a column, in its JSON Schema only.

    pydantic would enforce the minimum with its own message; the analysis checks
    it itself, so that the message gives the fix, and the JSON Schema says it
    for the callers that validate before they call.
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {**handler(core_schema), "minProperties": 1}


@analysis(
    statuses=("completed", "insufficient_data"),
    finding_codes=_FINDING_CODES,
    tags=(
        "schema",
        "data contract",
        "validation",
        "declared types",
        "data types",
        "expected types",
        "invalid values",
    ),
    masking=True,
)
def validate_column_types(
    data: pd.DataFrame,
    *,
    schema: Annotated[Mapping[str, LogicalType | ColumnSpec], _AtLeastOneColumn],
    id_column: str | None = None,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> ValidateColumnTypesResult:
    """Do the columns hold the logical types I declared?

    When to use:
        Call it once you know the logical type each column should hold, such as from a
        data contract, to check every value against its declared type without changing
        the data. Declare each type in schema, alone or in a ColumnSpec that also says
        whether values may be missing and which datetime format text uses. Each declared
        column gets counts of the values already stored as its type, the values that
        convert to it exactly, such as numbers or ISO 8601 dates stored as text, and the
        invalid ones, with example row IDs and values. A declared column the data lacks
        is reported, not raised. Example values and row IDs are raw values: when a column
        holds personal data, declare it sensitive in a project and switch masking on.
        Index labels are never masked, so pass a non-sensitive id_column when the index
        holds sensitive values. To guess each column's logical type from its values
        instead, call `infer_column_semantics`; `describe_dataset` gives every dtype.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified, and no value is converted.
        schema (Mapping[str, str | ColumnSpec]): The logical type each column
            should hold, by column name, in the order to report them:
            'boolean', 'integer', 'float', 'categorical', 'text', 'datetime' or
            'timedelta', or a ColumnSpec (a mapping of its fields works too).
            A ColumnSpec declared non-nullable makes missing values a violation,
            and one with a datetime_format, a datetime.strptime format such as
            '%d/%m/%Y', lets datetime text in that format convert as ISO 8601
            text does; strptime reads each distinct text, which takes about 4
            seconds per million. Locale-dependent directives, such as %b, read
            in the process's locale. A declared column the data lacks is
            reported as a violation. A label that isn't a string can be given
            as itself or as its string form. It must declare at least one column; there
            is no default.
        id_column (str | None): Name of a column whose values identify rows,
            such as an order number; row IDs are then its values. Its values
            must be hashable, such as numbers or strings; a column of lists
            raises. A label that isn't a string can be given as itself or as
            its string form. None, the default, uses the index labels as row
            IDs.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces the invalid values of
            sensitive columns, and row IDs taken from a sensitive id_column,
            with numbered placeholders, '<masked 1>' and so on, equal values
            alike, and needs a project that declares sensitive columns; False
            shows them as stored. None, the default, follows the project's
            data_policy.mask_sensitive, and masks nothing without a project.
            Index labels are never masked. Masking is a guardrail against
            accidental exposure, not anonymization or access control.

    Returns:
        ValidateColumnTypesResult: `columns` maps each declared column, in declared
        order, to its entry: the declaration, `present`, `dtype` and four counts that
        add up to the rows (`compatible_count`, `convertible_count`, `invalid_count`
        and `missing_count`), with `invalid_row_ids` (the first 10 rows that hold an
        invalid value) and `invalid_values` (the first 5 distinct ones with their
        counts).
        `undeclared_columns` lists the columns schema leaves out, which aren't
        checked. A declared column the data lacks has a null dtype and null counts,
        with a `column_not_in_data` limitation. Each entry's `assessment` is
        violation, unknown or passed, as its findings give it.

        What converts: to boolean, numbers equal to 0 or 1 and text such as yes, no,
        true, false, t, f, y, n, 0 or 1 in any case; to integer, whole floats and
        decimals and text that parses as a whole number; to float, any other number and
        text that parses as a number; to datetime, text in ISO 8601 (YYYY-MM-DD with an
        optional time and offset) or in the declared datetime_format; to timedelta,
        nothing; to categorical, any value that can be hashed; to text, booleans,
        numbers, datetimes and durations. The values of a categorical dtype are
        compatible with 'categorical' and convert to the types their categories convert
        to. A whole number written with a leading zero, such as 01234, is a code that
        doesn't convert to a number, and no date format other than ISO 8601 and the
        declared one parses, so day and month order is never guessed. Text is read after
        stripping surrounding whitespace, once per distinct value, by the rules
        infer_column_semantics reads it with. Values of different types count apart, so
        1 and true are two values.

        With masking on, the invalid values of sensitive columns and row IDs from a
        sensitive id_column are placeholders in every output.

        - `invalid_values`: violation; columns that hold invalid values.
        - `missing_declared_columns`: violation; declared columns the data doesn't have,
          each with the closest names.
        - `non_nullable_missing_values`: violation; columns declared non-nullable
          that have missing values.
        - `unchecked_columns`: unknown; columns without a value to check, as when every
          value is missing.
        - `convertible_values`: descriptive; columns with values stored as another type;
          convert them before you use them as the declared type.
        - `passing_columns`: passed; columns that pass every check: only compatible and
          convertible values, and none missing when declared non-nullable.

    Statuses:
        completed: Every declared column was checked, also when the data lacks
            some of them; the findings say what the checks found.
        insufficient_data: The data has no rows, so there is no value to check;
            the counts are 0, and a declared column the data lacks is still
            reported. Reason code: `no_rows`.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "order_id": ["A1", "A2", "A3", "A4", "A5"],
        ...         "qty": ["2", "1", "n/a", "3", "n/a"],
        ...         "ordered": [
        ...             "2024-01-05",
        ...             "2024-01-06",
        ...             None,
        ...             "05/01/2024",
        ...             "2024-01-09",
        ...         ],
        ...         "paid": ["yes", "no", "yes", "yes", None],
        ...     },
        ...     dtype=object,
        ... )
        >>> res = ae.validate_column_types(
        ...     df,
        ...     schema={
        ...         "qty": "integer",
        ...         "ordered": ae.ColumnSpec("datetime", nullable=False),
        ...         "paid": "boolean",
        ...         "customer": "text",
        ...     },
        ...     id_column="order_id",
        ... )
        >>> print(res)
        validate_column_types: completed on 5 rows x 4 columns
        compatible: stored as the declared type; convertible: converts exactly; invalid: neither. Nothing was converted.
        Row IDs: values of id_column 'order_id'.
        Not declared in schema, so not checked: 'order_id'.
        Declared columns, violations first (4 of 4 shown):
        column    declared            dtype            compatible  convertible  invalid  missing  invalid_values
        qty       integer             object                    0            3        2        0  'n/a': 2
        ordered   datetime, not null  object                    0            3        1        1  '05/01/2024': 1
        customer  text                not in the data        null         null     null     null
        paid      boolean             object                    0            4        0        1
        Findings (5 of 5 shown):
        - violation invalid_values: 2 columns hold invalid values: 'qty' and 'ordered'.
        - violation missing_declared_columns: 1 declared column is not in the data: 'customer'.
        - violation non_nullable_missing_values: 1 column declared non-nullable has missing values: 'ordered'.
        - descriptive convertible_values: 3 columns hold convertible values: 'qty', 'ordered' and 'paid'.
        - passed passing_columns: 1 column passes every check: 'paid'.
        Limitations (2 of 2 shown):
        - nullable_not_declared: nullable wasn't declared, so missing values weren't checked: 'qty', 'paid' and 'customer'.
        - column_not_in_data: 1 declared column isn't in the data, so its dtype and counts are null: 'customer'.
        More: res.to_frame() has all 4 declared columns; res.to_frame(table='invalid_values') counts every distinct invalid
            value; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.columns["qty"].invalid_row_ids
        ['A3', 'A5']

    See Also:
        profile_missing_values: How many values are missing per column and per row,
            declared sentinel values included.
        infer_column_semantics: What kind of values each column holds, ranked by
            heuristic scores, compared with the logical types you declare.
        describe_dataset: Dtypes, missing and distinct counts and memory of every
            column.
    """
    if not schema:
        raise worded(
            InvalidArgumentError,
            lambda surface: (
                "validate_column_types got an empty schema, which "
                "declares no column to check. Declare the logical type of at least one "
                f"column, such as {as_written(surface, schema={'age': 'integer'})}."
            ),
        )
    masking = resolve_masking("validate_column_types", project, mask_sensitive)
    row_ids = read_row_ids("validate_column_types", data, id_column)
    declared = {
        name: spec if isinstance(spec, ColumnSpec) else ColumnSpec(logical_type=spec)
        for name, spec in schema.items()
    }
    names = [str(label) for label in data.columns]
    positions = {name: position for position, name in enumerate(names)}
    checks = {
        name: check_column(
            data.iloc[:, positions[name]],
            column_kind(data.dtypes.iloc[positions[name]]),
            spec.logical_type,
            spec.datetime_format,
        )
        for name, spec in declared.items()
        if name in positions
    }
    masked = {
        name: mask_levels(check.invalid_values)
        for name, check in checks.items()
        if masking.masks(name) and len(check.invalid_values.values)
    }
    ids, every_id = _row_ids_of(
        checks, row_ids, masked, masked=masking.masks(row_ids.id_column)
    )
    entries = {
        name: _checked_entry(
            name,
            spec,
            str(data.dtypes.iloc[positions[name]]),
            checks[name],
            ids[name],
            masked.get(name),
        )
        if name in checks
        else _absent_entry(name, spec)
        for name, spec in declared.items()
    }
    absent = [name for name, entry in entries.items() if not entry.present]
    # The paths of the fields that hold a row ID another row shares.
    ambiguous = [
        f"{_path(name)}.invalid_row_ids"
        for name, check in checks.items()
        if row_ids.repeated[check.invalid_rows].any()
    ]
    result = ValidateColumnTypesResult(
        analysis="validate_column_types",
        method=None,
        status="completed" if len(data) else "insufficient_data",
        status_reason=None if len(data) else NO_ROWS,
        parameters={
            "data": data_shape(data),
            "schema": {
                name: spec.model_dump(mode="json") for name, spec in declared.items()
            },
            "id_column": row_ids.id_column,
            "project": project.model_dump(mode="json") if project else None,
            "mask_sensitive": mask_sensitive,
        },
        support=Support(
            row_count=len(data),
            column_count=len(checks),
            columns=[
                ColumnSupport(
                    column=name,
                    eligible_count=len(data),
                    evaluated_count=len(data) - check.missing,
                    excluded_count=check.missing,
                    excluded_by_reason={"missing": check.missing}
                    if check.missing
                    else {},
                )
                for name, check in checks.items()
            ],
        ),
        findings=_validation_findings(entries, absent, names),
        limitations=[
            *ambiguous_row_ids(row_ids, ambiguous),
            *_validation_limitations(checks, absent, entries),
        ],
        execution=execution(("numpy", "pandas")),
        masking=masking.record(_shown_columns(entries, row_ids.id_column), data),
        columns=entries,
        undeclared_columns=[name for name in names if name not in declared],
    )
    result._tables = {
        name: masked[name].levels if name in masked else check.invalid_values
        for name, check in checks.items()
    }
    result._row_ids = every_id
    return result


def _row_ids_of(
    checks: dict[str, Check],
    row_ids: RowIds,
    masked_levels: dict[str, MaskedLevels],
    *,
    masked: bool,
) -> tuple[dict[str, list[object]], dict[str, Any]]:
    """The row IDs each entry shows and the row IDs of every invalid row.

    An entry shows the IDs of its first rows, and the second mapping has every
    one: pandas labels, or lists once id_column is masked.

    Masked IDs are placeholders. A row ID equal to an invalid value of id_column
    that the result shows takes that value's placeholder. The other IDs are
    numbered on from there, the shown ones first, in result order, and equal
    IDs share a placeholder.
    """
    everyone = {
        name: row_ids.labels.take(check.invalid_rows) for name, check in checks.items()
    }
    shown = {
        name: row_ids.at(check.invalid_rows[:_ROW_ID_LIMIT])
        for name, check in checks.items()
    }
    if not masked:
        return shown, everyone
    id_column = typing.cast("str", row_ids.id_column)
    listed = {name: row_id_list(ids) for name, ids in everyone.items()}
    known: dict[object, str] = {}
    start = 1
    if id_column in masked_levels:
        levels = checks[id_column].invalid_values
        flat = pd.Series([i for ids in listed.values() for i in ids], dtype=object)
        for position in levels.positions_of(flat).tolist():
            known[levels.label(position)] = masked_levels[id_column].label(position)
        start += len(levels.values)
    numbers: dict[object, str] = {}

    def number(ids: list[object], begin: int, end: int) -> None:
        for index in range(begin, end):
            value = ids[index]
            if pd.api.types.is_scalar(value) and pd.isna(value):
                ids[index] = None
            elif value in known:
                ids[index] = known[value]
            else:
                ids[index] = numbers.setdefault(
                    value, placeholder(start + len(numbers))
                )

    for ids in listed.values():
        number(ids, 0, min(len(ids), _ROW_ID_LIMIT))
    for ids in listed.values():
        number(ids, _ROW_ID_LIMIT, len(ids))
    return {name: ids[:_ROW_ID_LIMIT] for name, ids in listed.items()}, listed


def _shown_columns(
    entries: dict[str, ValidateColumnTypesColumn], id_column: str | None
) -> list[str | None]:
    """The columns whose values the result shows, in result order.

    An entry shows its invalid values and their row IDs, which are values of
    id_column, or index labels, None here, without one.
    """
    shown: list[str | None] = []
    for name, entry in entries.items():
        if entry.invalid_row_ids:
            shown.extend([id_column, name])
    return shown


def _checked_entry(
    name: str,
    spec: ColumnSpec,
    dtype: str,
    check: Check,
    row_ids: list[object],
    masked: MaskedLevels | None,
) -> ValidateColumnTypesColumn:
    """A declared column of the data, its values checked against its spec."""
    levels = check.invalid_values
    if check.invalid or (spec.nullable is False and check.missing):
        assessment: Literal["passed", "violation", "unknown"] = "violation"
    elif not (check.compatible or check.convertible or check.invalid):
        assessment = "unknown"
    else:
        assessment = "passed"
    return ValidateColumnTypesColumn(
        column=name,
        declared_logical_type=spec.logical_type,
        nullable=spec.nullable,
        datetime_format=spec.datetime_format,
        present=True,
        dtype=dtype,
        compatible_count=check.compatible,
        convertible_count=check.convertible,
        invalid_count=check.invalid,
        missing_count=check.missing,
        invalid_row_ids=row_ids,
        invalid_distinct_count=len(levels.values),
        invalid_values=[
            InvalidValue(
                value=masked.label(position)
                if masked
                else record_label(levels.label(position)),
                value_type=type_name(levels.label(position)),
                count=int(levels.counts[position]),
            )
            for position in ordered(levels, _EXAMPLE_LIMIT).tolist()
        ],
        assessment=assessment,
    )


def _absent_entry(name: str, spec: ColumnSpec) -> ValidateColumnTypesColumn:
    """A declared column the data lacks: nothing to count."""
    return ValidateColumnTypesColumn(
        column=name,
        declared_logical_type=spec.logical_type,
        nullable=spec.nullable,
        datetime_format=spec.datetime_format,
        present=False,
        dtype=None,
        compatible_count=None,
        convertible_count=None,
        invalid_count=None,
        missing_count=None,
        invalid_row_ids=[],
        invalid_distinct_count=None,
        invalid_values=[],
        assessment="violation",
    )


@dataclass(frozen=True)
class _ColumnFinding:
    """The finding that names the declared columns a check picks out."""

    code: str
    assessment: Assessment
    # What the columns do, after "1 column" and after "2 columns".
    singular: str
    plural: str
    # The entry field behind the claim.
    field: str
    picks: Callable[[ValidateColumnTypesColumn], bool]


_COLUMN_FINDINGS = (
    _ColumnFinding(
        "invalid_values",
        "violation",
        "holds invalid values",
        "hold invalid values",
        "invalid_count",
        lambda e: bool(e.invalid_count),
    ),
    _ColumnFinding(
        "non_nullable_missing_values",
        "violation",
        "declared non-nullable has missing values",
        "declared non-nullable have missing values",
        "missing_count",
        lambda e: e.nullable is False and bool(e.missing_count),
    ),
    _ColumnFinding(
        "unchecked_columns",
        "unknown",
        "has no value to check",
        "have no value to check",
        "missing_count",
        lambda e: (
            e.present
            and not (e.compatible_count or e.convertible_count or e.invalid_count)
        ),
    ),
    _ColumnFinding(
        "convertible_values",
        "descriptive",
        "holds convertible values",
        "hold convertible values",
        "convertible_count",
        lambda e: bool(e.convertible_count),
    ),
    _ColumnFinding(
        "passing_columns",
        "passed",
        "passes every check",
        "pass every check",
        "invalid_count",
        lambda e: e.assessment == "passed",
    ),
)


def _validation_findings(
    entries: dict[str, ValidateColumnTypesColumn], absent: list[str], names: list[str]
) -> list[Finding]:
    """Violations first, then what couldn't be checked, what converts and what held."""
    findings = {}
    for finding in _COLUMN_FINDINGS:
        picked = [name for name, entry in entries.items() if finding.picks(entry)]
        if picked:
            does = plural(
                len(picked), f"column {finding.singular}", f"columns {finding.plural}"
            )
            findings[finding.code] = Finding(
                code=finding.code,
                claim=f"{does}: {join_labels(picked)}.",
                assessment=finding.assessment,
                columns=picked,
                fields=[f"{_path(name)}.{finding.field}" for name in picked],
            )
    if absent:
        findings["missing_declared_columns"] = Finding(
            code="missing_declared_columns",
            claim=_missing_columns_claim(absent, names),
            assessment="violation",
            columns=absent,
            fields=[f"{_path(name)}.present" for name in absent],
        )
    return [findings[code] for code in _FINDING_CODES if code in findings]


def _validation_limitations(
    checks: dict[str, Check],
    absent: list[str],
    entries: dict[str, ValidateColumnTypesColumn],
) -> list[Limitation]:
    limitations = []
    open_nullable = [name for name, entry in entries.items() if entry.nullable is None]
    if open_nullable:
        limitations.append(
            Limitation(
                code="nullable_not_declared",
                reason=(
                    "nullable wasn't declared, so missing values weren't checked: "
                    f"{join_labels(open_nullable)}."
                ),
                fields=[f"{_path(name)}.nullable" for name in open_nullable],
            )
        )
    stored = [name for name, check in checks.items() if check.stored_nan]
    limitations += stored_nan_limitations(
        stored,
        [f"{_path(name)}.missing_count" for name in stored],
        "missing_count counts it",
    )
    if absent:
        if len(absent) == 1:
            what = (
                "1 declared column isn't in the data, so its dtype and counts are null"
            )
        else:
            what = (
                f"{len(absent):,} declared columns aren't in the data, so their "
                "dtypes and counts are null"
            )
        limitations.append(
            Limitation(
                code="column_not_in_data",
                reason=f"{what}: {join_labels(absent)}.",
                fields=[_path(name) for name in absent],
            )
        )
    return limitations
