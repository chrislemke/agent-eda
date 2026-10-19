"""The results of infer_column_semantics and validate_column_types."""

import typing
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    Field,
    PrivateAttr,
)

from agent_eda import _text
from agent_eda._levels import Levels, levels_frame
from agent_eda._result_base import (
    _BIG_INTEGERS,
    _BY_NAME,
    _ENTRIES,
    _FRAME_DTYPES,
    AnalysisResult,
    ColumnFlag,
    ColumnResult,
    LogicalType,
    MaskableResult,
    ResultModel,
    RowId,
    _Cap,
    _check_table,
    _tables_unavailable,
)
from agent_eda._written_form import row_id_list


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
        description="Values that are datetimes or dates, pandas Timestamps included."
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
        description="Values that aren't missing: native missing values, and NaN "
        "that a float column stores as a value, are missing."
    )
    distinct_count: int | None = Field(
        description="Distinct non-missing values, as pandas counts them, so 1, "
        "1.0 and the boolean true are one value; null when values such as lists can't be "
        "hashed."
    )
    uniqueness_share: float | None = Field(
        description="distinct_count / non_missing_count: 1 when every value is "
        "distinct."
    )
    boolean_share: float | None = Field(
        description="Values that are booleans or read as one: numbers equal to 0 "
        "or 1, and the words true, false, t, f, yes, no, y and n in any case. "
        "The boolean candidate needs a share of 1."
    )
    number_share: float | None = Field(
        description="Values that are numbers or text that parses as a decimal "
        "number that float64 can hold, such as '42', '-3.5' or '1e-3'; '1e400' "
        "doesn't count."
    )
    integer_share: float | None = Field(
        description="Values that are or parse as whole numbers, such as 42, 42.0 "
        "or '4.2e1'."
    )
    datetime_share: float | None = Field(
        description="Values that are datetimes or dates, or text that parses as "
        "an ISO 8601 date, with or without a time, or as a ColumnSpec's "
        "datetime_format for that column."
    )
    timedelta_share: float | None = Field(description="Values that are durations.")
    leading_zero_share: float | None = Field(
        description="Text values that are whole numbers written as digits alone "
        "with a leading zero, such as '007': codes, which don't parse as "
        "numbers. A decimal such as '05.50' is a number."
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
    value_types: Annotated[list[str], _ENTRIES] = Field(
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
        description="The logical type schema gives the column; null when "
        "it declares none. Inference never overwrites it."
    )
    matches_declared: bool | None = Field(
        description="Whether declared_logical_type is the top candidate or ties "
        "with it in score; null when it is null or no type was inferred."
    )
    candidates: list[LogicalTypeCandidate] = Field(
        description="The logical types with a score above 0, best first, ties in "
        "the order boolean, integer, float, datetime, timedelta, categorical, "
        "text. Integer and float also appear, scored 0, when leading zeros rule "
        "them out. A column where every candidate scores 0 has no inferred type."
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

    columns: Annotated[dict[str, InferColumnSemanticsColumn], _ENTRIES, _BY_NAME] = (
        Field(
            description="One entry per selected column, keyed by column name, in the "
            "selected order. The record keeps the first 100 and adds columns_total; "
            "the complete table has every column."
        )
    )
    missing_declared_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns schema names that the data doesn't have, in "
        "declared order; empty without schema."
    )

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        facts = [
            "Scores rank logical types by fixed rules and aren't probabilities. "
            "Dates parse as ISO 8601 only."
        ]
        if self.parameters.get("schema") is None:
            facts.append(
                "No schema was given, so no column was compared with a "
                "declared logical type."
            )
        else:
            entries = self.columns.values()
            present = sum(e.declared_logical_type is not None for e in entries)
            matching = sum(e.matches_declared is True for e in entries)
            differing = sum(e.matches_declared is False for e in entries)
            unchecked = present - matching - differing
            missing = len(self.missing_declared_columns)
            declared = _text.plural(present + missing, "column", "columns")
            # Every declared column is in exactly one of the counts.
            facts.append(
                f"schema: {declared} declared: {matching:,} matching "
                f"(ties with the top count), {differing:,} not, {unchecked:,} not "
                f"inferred, {missing:,} not in the data."
            )
        return facts

    def inferred_schema(self) -> dict[str, Any]:
        """The inferred logical types as a schema for validate_column_types.

        A column without an inferred type is left out, since nothing could be
        declared for it. A datetime column keeps the datetime_format that this
        call's schema declared for it, so that the text that was read as dates
        is read the same way.

        Returns:
            A dict from column name to a logical type, or to a ColumnSpec for a
            datetime column with a declared format.
        """
        from agent_eda.data.validation import ColumnSpec  # circular at import time

        declared = self.parameters.get("schema")
        formats = {
            name: spec.get("datetime_format")
            for name, spec in (declared if isinstance(declared, dict) else {}).items()
            if isinstance(spec, dict)
        }
        schema: dict[str, Any] = {}
        for name, entry in self.columns.items():
            logical_type = entry.inferred_logical_type
            if logical_type is None:
                continue
            datetime_format = formats.get(name) if logical_type == "datetime" else None
            schema[name] = (
                ColumnSpec(logical_type, datetime_format=datetime_format)
                if isinstance(datetime_format, str)
                else logical_type
            )
        return schema

    def _text_other_tables(self) -> list[str]:
        return ["res.to_frame(table='candidates') has every candidate with its reasons"]

    def _text_table(self) -> _text.Table:
        declared = self.parameters.get("schema") is not None
        headers = ["column", "inferred", "score", "next", "next_score"]
        if declared:
            headers.append("declared")
        headers.append("flags")
        rows = []
        for entry in self._by_interest(list(self.columns.values()), list(self.columns)):
            # Candidates scored 0 rank no type: a column with none has no top.
            top = entry.candidates[0] if entry.inferred_logical_type else None
            runner_up = (
                entry.candidates[1] if top and len(entry.candidates) > 1 else None
            )
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
                [
                    e.candidates[0].score if e.inferred_logical_type else None
                    for e in entries
                ],
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


class InvalidValue(ResultModel):
    """A distinct value that neither has nor converts to the declared logical type."""

    value: str | bool | int | float = Field(
        description="The value as stored. A datetime is ISO 8601 text and a Decimal a number "
        "or, when float64 can't hold it as written, its exact text; any other "
        "value JSON can't hold appears as its string form. value_type names "
        "its type. With masking on, a sensitive column's values "
        "are numbered placeholders, '<masked 1>' for its first invalid value in "
        "level order and so on. " + _BIG_INTEGERS
    )
    value_type: str = Field(
        description="The Python type of the stored value, such as 'str' or 'float'; "
        "a NumPy scalar counts as the Python type it holds."
    )
    count: int = Field(
        description="Values in the column equal to it and of the same type."
    )


class ValidateColumnTypesColumn(ResultModel):
    """How the values of one declared column relate to its declared logical type."""

    column: str = Field(description="The declared column name.")
    declared_logical_type: LogicalType = Field(
        description="The logical type schema declares for the column: the one its "
        "values are checked against."
    )
    nullable: bool | None = Field(
        description="Whether schema allows missing values in the column: false "
        "makes them a violation; null declares nothing, so they aren't checked."
    )
    datetime_format: str | None = Field(
        description="The datetime.strptime format declared for datetimes stored as "
        "text, besides ISO 8601; null when none is declared."
    )
    present: bool = Field(description="Whether the data has the column.")
    dtype: str | None = Field(
        description="The dtype as pandas reports it, such as 'str'; null when the "
        "column isn't in the data."
    )
    compatible_count: int | None = Field(
        description="Values already stored as the declared logical type, such as "
        "integers in an int64 or Int64 column for 'integer'."
    )
    convertible_count: int | None = Field(
        description="Values stored as another type that convert to the declared "
        "logical type exactly, without changing their meaning, such as '42' or "
        "42.0 for 'integer'. Nothing is converted."
    )
    invalid_count: int | None = Field(
        description="Values that neither have the declared logical type nor convert "
        "to it exactly."
    )
    missing_count: int | None = Field(
        description="Native missing values, and NaN that a float column stores "
        "as a value."
    )
    invalid_row_ids: Annotated[list[RowId], _Cap(10, total="invalid_count")] = Field(
        description="The IDs of the first 10 rows that hold an invalid value, in row "
        "order: index labels, or values of id_column when it is given. "
        "invalid_count counts them all, and the record adds it as "
        "invalid_row_ids_total. With masking on and a sensitive id_column, "
        "numbered placeholders, equal IDs alike. " + _BIG_INTEGERS
    )
    invalid_distinct_count: int | None = Field(
        description="Distinct invalid values; values of different types count "
        "apart, so 1 and the boolean true are two."
    )
    invalid_values: Annotated[
        list[InvalidValue], _Cap(5, total="invalid_distinct_count")
    ] = Field(
        description="The first 5 distinct invalid values with their counts, in level "
        "order as summarize_categorical orders levels: most frequent first, then "
        "by value. invalid_distinct_count counts them all, the record adds it as "
        "invalid_values_total, and the Python-only table invalid_values has "
        "every one."
    )
    assessment: Literal["passed", "violation", "unknown"] = Field(
        description="The assessment of the column, as its findings give it: violation "
        "when the data lacks it, it holds invalid values, or it is declared "
        "non-nullable and has missing values; unknown when it has no value to "
        "check; passed otherwise."
    )


class ValidateColumnTypesResult(MaskableResult):
    """The result of validate_column_types."""

    columns: Annotated[dict[str, ValidateColumnTypesColumn], _ENTRIES, _BY_NAME] = (
        Field(
            description="One entry per declared column, keyed by column name, in "
            "declared order. The record keeps the first 100 and adds columns_total; "
            "the complete table has every column."
        )
    )
    undeclared_columns: Annotated[list[str], _ENTRIES] = Field(
        description="The data's columns that schema doesn't declare, in DataFrame "
        "order; they weren't checked. The record keeps the first 100 and adds "
        "undeclared_columns_total."
    )
    # Every distinct invalid value of every declared column in the data,
    # outside the record; masked columns hold placeholders' numbers instead.
    # None for a result rebuilt from its record, which holds no values.
    _tables: dict[str, Levels] | None = PrivateAttr(default=None)
    # The ID of every row that holds an invalid value, by declared column, outside
    # the record: pandas labels, or lists of placeholders when masked. None for a
    # result rebuilt from its record, which holds the first 10 of each.
    _row_ids: dict[str, Any] | None = PrivateAttr(default=None)

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        facts = [
            "compatible: stored as the declared type; convertible: converts "
            "exactly; invalid: neither. Nothing was converted."
        ]
        id_column = self.parameters.get("id_column")
        source = (
            f"values of id_column {id_column!r}"
            if isinstance(id_column, str)
            else "index labels"
        )
        facts.append(f"Row IDs: {source}.")
        if self.undeclared_columns:
            names = _text.join_labels(self.undeclared_columns)
            facts.extend(
                _text.wrap(
                    f"Not declared in schema, so not checked: {names}.",
                    first="",
                    indent="    ",
                )
            )
        else:
            facts.append("schema declares every column of the data.")
        return facts

    def _text_table(self) -> _text.Table:
        # Violations first, then columns without a value to check; a stable
        # sort keeps the declared order within each.
        rank = {"violation": 0, "unknown": 1, "passed": 2}
        entries = sorted(self.columns.values(), key=lambda e: rank[e.assessment])
        return _text.Table(
            title="Declared columns, violations first",
            unit=("declared column", "declared columns"),
            headers=(
                "column",
                "declared",
                "dtype",
                "compatible",
                "convertible",
                "invalid",
                "missing",
                "invalid_values",
            ),
            rows=[
                (
                    e.column,
                    e.declared_logical_type
                    + (", not null" if e.nullable is False else ""),
                    e.dtype if e.present else "not in the data",
                    e.compatible_count,
                    e.convertible_count,
                    e.invalid_count,
                    e.missing_count,
                    self._examples_text(e),
                )
                for e in entries[: _text.MAX_TABLE_ROWS]
            ],
            total=len(entries),
        )

    def _examples_text(self, entry: ValidateColumnTypesColumn) -> str:
        """The invalid example values with their counts, such as "'n/a': 2".

        Text is quoted, so surrounding spaces stay visible; placeholders aren't.
        """
        masked = self._masked(entry.column)
        shown = [
            f"{value.value!r}: {value.count:,}"
            if value.value_type == "str" and not masked
            else f"{value.value}: {value.count:,}"
            for value in entry.invalid_values
        ]
        more = (entry.invalid_distinct_count or 0) - len(shown)
        return ", ".join(shown) + (f" and {more:,} more" if more > 0 else "")

    def _text_other_tables(self) -> list[str]:
        return [
            "res.to_frame(table='invalid_values') counts every distinct invalid value"
        ]

    def to_frame(
        self, table: Literal["columns", "invalid_values"] = "columns"
    ) -> pd.DataFrame:
        """One of the result's two tables as a DataFrame.

        Args:
            table: 'columns', the default, gives one row per declared column in
                declared order: its declaration, whether it is present, its
                dtype, the four counts, the number of distinct invalid values,
                the IDs of every row with an invalid value, and the assessment.
                The values themselves are in table='invalid_values'.
                'invalid_values' gives one row per distinct invalid value of
                each column, by column in declared order, then by count,
                descending, then by value: column, value as stored (or its
                placeholder, with masking on), value_type and count.
        """
        _check_table(table, ("columns", "invalid_values"))
        if table == "invalid_values":
            if self._tables is None and any(
                column.invalid_distinct_count for column in self.columns.values()
            ):
                raise _tables_unavailable(self)
            frame = levels_frame(self._tables or {}).drop(columns="share")
            return frame.rename(columns={"label": "value", "label_type": "value_type"})
        entries = list(self.columns.values())
        columns: dict[str, pd.Series] = {}
        for name, field in ValidateColumnTypesColumn.model_fields.items():
            if name == "invalid_values":
                continue  # table='invalid_values' has every one
            if name == "invalid_row_ids":
                columns[name] = pd.Series(self._every_row_id(entries), dtype=object)
                continue
            columns[name] = pd.Series(
                [getattr(entry, name) for entry in entries],
                dtype=_FRAME_DTYPES.get(field.annotation, object),
            )
        return pd.DataFrame(columns, columns=list(columns))

    def _every_row_id(
        self, entries: Sequence[ValidateColumnTypesColumn]
    ) -> list[list[object]]:
        """The IDs of every row with an invalid value, per entry."""
        if self._row_ids is None:
            if any(
                (entry.invalid_count or 0) > len(entry.invalid_row_ids)
                for entry in entries
            ):
                raise _tables_unavailable(self)
            return [entry.invalid_row_ids for entry in entries]
        return [
            row_id_list(self._row_ids[entry.column]) if entry.present else []
            for entry in entries
        ]
