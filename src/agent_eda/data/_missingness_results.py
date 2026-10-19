"""The results of profile_missing_values and detect_missing_sentinels."""

import datetime as dt
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    Field,
    PrivateAttr,
)

from agent_eda import _text
from agent_eda._result_base import (
    _BIG_INTEGERS,
    _BY_NAME,
    _ENTRIES,
    _FRAME_DTYPES,
    ColumnResult,
    MaskableResult,
    ResultModel,
    RowId,
    _Cap,
    _check_table,
    _entries_frame,
    _hint,
    _tables_unavailable,
    share_text,
)
from agent_eda._written_form import row_id_list
from agent_eda.errors import (
    ColumnNotFoundError,
)


class DeclaredSentinelValue(ResultModel):
    """A sentinel value declared for a column, and how often it occurs there."""

    value: bool | int | float | str = Field(
        description="The value as declared; dates, datetimes and timedeltas as ISO "
        "8601 text, and a decimal as a number or, when float64 can't hold it as "
        "written, its exact text. With masking on, a sensitive column's values "
        "are numbered placeholders, '<masked 1>', '<masked 2>' and so on, in the "
        "order declared. " + _BIG_INTEGERS
    )
    value_type: str = Field(
        description="The Python type of the declared value, such as 'int', 'bool', "
        "'str', 'date' or 'Timestamp'. It tells apart values that print the same: "
        "the date 2020-01-01 and the text '2020-01-01', or the boolean true and 1."
    )
    count: int = Field(
        description="Values in the column equal to it, among those pandas doesn't "
        "count as missing."
    )


class ProfileMissingValuesColumn(ColumnResult):
    """What profile_missing_values found in one column."""

    native_missing_count: int = Field(
        description="Values that are missing: native missing values, including "
        "NaN that a pyarrow-backed or nullable float column stores as a value, "
        "which pandas' missing-value check doesn't count."
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


class ProfileMissingValuesResult(MaskableResult):
    """The result of profile_missing_values."""

    columns: Annotated[dict[str, ProfileMissingValuesColumn], _ENTRIES, _BY_NAME] = (
        Field(
            description="One entry per selected column, keyed by column name, in the "
            "selected order. The record keeps the first 100 and adds columns_total; "
            "the complete table has every column."
        )
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
        "100 and adds row_distribution_total; the Python-only table "
        "row_distribution has every entry."
    )

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        entries = self.columns.values()
        rows = self.support.row_count
        declared = any(entry.sentinel_count is not None for entry in entries)
        missing = (
            f"Missing: {self.missing_count:,} of {rows * len(self.columns):,} "
            f"selected values{share_text('missing_share', self.missing_share)}"
            f"{', counting declared sentinel values' if declared else ''}. "
            f"Complete rows: {self.complete_row_count:,} of {rows:,}"
            f"{share_text('complete_row_share', self.complete_row_share)}."
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
                holds the reason's code, and sentinel_values lists each declared
                value with its type and count, so values that print the same
                stay apart. 'row_distribution' gives one row per number
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
                counts = [
                    [
                        {"value": v.value, "value_type": v.value_type, "count": v.count}
                        for v in declared
                    ]
                    for declared in values
                ]
                columns[name] = pd.Series(counts, dtype=object)
            else:
                dtype = _FRAME_DTYPES.get(field.annotation)
                columns[name] = pd.Series(values, dtype=dtype)
        return pd.DataFrame(columns, columns=list(columns))


# Why a value is a sentinel value candidate.
SentinelEvidence = Literal["declared", "builtin_list", "isolated_extreme"]


class SentinelCandidate(ResultModel):
    """A value of one column that looks like a stand-in for missing values."""

    column: str = Field(description="The column that holds the value.")
    value: bool | int | float | str = Field(
        description="The value: a declared one as declared, a suspected one as "
        "stored. Dates, datetimes and durations appear as ISO 8601 text, and a "
        "decimal as a number or, when float64 can't hold it as written, its exact "
        "text; value_type names the type. A text stand-in keeps the surrounding "
        "whitespace it occurs with. " + _BIG_INTEGERS
    )
    value_type: str = Field(
        description="The Python type of the value, such as 'int', 'float', 'str', "
        "'Timestamp' or 'date'; a NumPy scalar counts as the Python type it holds."
    )
    count: int = Field(
        description="Values in the column equal to the value, among those pandas "
        "doesn't count as missing."
    )
    share: float = Field(
        description="count over all rows, from 0 to 1: declaring the value raises "
        "the column's missing_share in profile_missing_values by this much."
    )
    evidence: list[SentinelEvidence] = Field(
        description="Why the value is a candidate, in this order: declared, it is "
        "declared in sentinel_values; builtin_list, it is on the built-in list of "
        "stand-ins; isolated_extreme, it repeats at the column's minimum or "
        "maximum, farther from the nearest other value than the others span."
    )
    confirmation: Literal["declared", "suspected"] = Field(
        description="declared: you declared it in sentinel_values. suspected: a "
        "built-in rule found it, and it stays a valid value until you declare it."
    )
    row_ids: Annotated[list[RowId], _Cap(10, total="count")] = Field(
        description="The IDs of the first 10 rows that hold the value, in row "
        "order: index labels, or values of id_column when it is given. The "
        "record adds row_ids_total, which equals count. " + _BIG_INTEGERS
    )


class SentinelValues(dict[str, list[Any]]):
    """The sentinel_values that a detect_missing_sentinels result gives: column
    names mapped to lists of values, as profile_missing_values takes them.

    `note` says which columns were left out because masking replaced their
    values, or is None when none was.
    """

    note: str | None = None


# How a candidate of each type that the record writes as text comes back.
_PARSE_VALUE: dict[str, Callable[[str], Any]] = {
    "Timestamp": pd.Timestamp,
    "datetime": dt.datetime.fromisoformat,
    "date": dt.date.fromisoformat,
    "time": dt.time.fromisoformat,
    "Timedelta": pd.Timedelta,
    "timedelta": pd.Timedelta,
    "Decimal": lambda text: Decimal(str(text)),
}


def _declared_value(candidate: "SentinelCandidate") -> Any:
    """The candidate's value as the type its column holds."""
    parse = _PARSE_VALUE.get(candidate.value_type)
    return candidate.value if parse is None else parse(str(candidate.value))


class DetectMissingSentinelsResult(MaskableResult):
    """The result of detect_missing_sentinels."""

    candidates: Annotated[list[SentinelCandidate], _ENTRIES] = Field(
        description="The values that look like stand-ins for missing values."
    )
    columns_without_rules: Annotated[list[str], _ENTRIES] = Field(
        description="Checked columns whose values no built-in rule reads, such as "
        "booleans, durations or lists, in the order checked: only declared values "
        "were looked for there. A column without values is listed too."
    )

    # Every row ID of each candidate, in the order of candidates, outside the
    # record: pandas labels, or lists of placeholders when masked. None for a
    # result rebuilt from its record, which holds the first 10 of each.
    _row_ids: list[Any] | None = PrivateAttr(default=None)

    def sentinel_values(self, columns: Sequence[str] | None = None) -> "SentinelValues":
        """The candidates as sentinel_values for profile_missing_values.

        Every candidate counts, suspected ones too, so declare only after you
        have read them. A value that the result writes as text comes back as the
        type its column holds: a Timestamp, a date, a time, a Timedelta or a
        Decimal. A candidate of a masked column is left out, since the result
        holds a placeholder instead of its value, and the returned mapping's
        `note` names those columns.

        Args:
            columns: The columns to take candidates from; None, the default,
                takes every column with a candidate.

        Returns:
            A dict from column name to its list of values, in candidate order,
            with a `note` attribute that is None when nothing was left out.

        Raises:
            ColumnNotFoundError: A name in columns isn't a column of the result.
        """
        names = list(dict.fromkeys(c.column for c in self.candidates))
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            for name in columns:
                if name not in names:
                    raise ColumnNotFoundError(
                        f"sentinel_values got column {name!r}, which has no "
                        "candidate in this result."
                        + _hint(name, names)
                        + f" The columns with candidates are {_text.join_labels(names)}."
                    )
        wanted = names if columns is None else list(columns)
        masked = [name for name in wanted if self._masked(name)]
        declared = SentinelValues()
        for candidate in self.candidates:
            if candidate.column in wanted and candidate.column not in masked:
                declared.setdefault(candidate.column, []).append(
                    _declared_value(candidate)
                )
        if masked:
            declared.note = (
                f"Left out the candidates of {_text.join_labels(masked)}: masking is "
                "on, so the result holds placeholders, not their values. Declare "
                "those values yourself."
            )
        return declared

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        declared = sum(c.confirmation == "declared" for c in self.candidates)
        counts = [
            f"{count:,} {word}"
            for count, word in (
                (declared, "declared"),
                (len(self.candidates) - declared, "suspected"),
            )
            if count
        ]
        found = (
            f"{' and '.join(counts)} candidate{'' if len(self.candidates) == 1 else 's'}"
            if counts
            else "no candidates"
        )
        id_column = self.parameters.get("id_column")
        source = (
            f"values of id_column {id_column!r}"
            if isinstance(id_column, str)
            else "index labels"
        )
        checked = _text.plural(self.support.column_count, "column", "columns")
        facts = _text.wrap(
            f"Checked {checked}: {found}. Shares are of all rows; row IDs are "
            f"{source}.",
            first="",
            indent="    ",
        )
        if self.columns_without_rules:
            names = self.columns_without_rules
            facts.extend(
                _text.wrap(
                    "No built-in rule reads "
                    f"{_text.plural(len(names), 'column', 'columns')}, so only "
                    f"declared values were looked for: {_text.join_labels(names)}.",
                    first="",
                    indent="    ",
                )
            )
        if declared < len(self.candidates):
            facts.extend(
                _text.wrap(
                    "Suspected values stay valid until declared in sentinel_values, "
                    "which profile_missing_values counts as missing.",
                    first="",
                    indent="    ",
                )
            )
        return facts

    def _text_table(self) -> _text.Table:
        # Most frequent first, ties in the order of candidates.
        order = sorted(
            range(len(self.candidates)), key=lambda i: -self.candidates[i].count
        )
        id_column = self.parameters.get("id_column")
        quote_ids = not (isinstance(id_column, str) and self._masked(id_column))
        return _text.Table(
            title="Candidates, most frequent first",
            unit=("candidate", "candidates"),
            headers=(
                "column",
                "value",
                "count",
                "share",
                "confirmation",
                "evidence",
                "row_ids",
            ),
            rows=[
                (
                    c.column,
                    _candidate_text(c, masked=self._masked(c.column)),
                    c.count,
                    c.share,
                    c.confirmation,
                    ", ".join(c.evidence),
                    _row_ids_text(c, quote=quote_ids),
                )
                for c in (self.candidates[i] for i in order[: _text.MAX_TABLE_ROWS])
            ],
            total=len(self.candidates),
        )

    def to_frame(self, table: Literal["candidates"] = "candidates") -> pd.DataFrame:
        """Every candidate, one row each, in the order of candidates.

        The columns are the candidate's fields. value holds each value as the
        result does, and evidence and row_ids hold lists; row_ids has the ID of
        every row that holds the value, where the record keeps the first 10.

        Args:
            table: 'candidates', the only table.
        """
        _check_table(table, ("candidates",))
        frame = _entries_frame(self.candidates, SentinelCandidate)
        frame["value"] = pd.Series([c.value for c in self.candidates], dtype=object)
        if self._row_ids is None:
            if any(c.count > len(c.row_ids) for c in self.candidates):
                raise _tables_unavailable(self)
            every: list[list[object]] = [c.row_ids for c in self.candidates]
        else:
            every = [row_id_list(ids) for ids in self._row_ids]
        frame["row_ids"] = pd.Series(every, dtype=object)
        return frame


def _candidate_text(candidate: SentinelCandidate, *, masked: bool) -> str:
    """A candidate's value as the text shows it: text quoted, so that empty and
    whitespace-only text stay visible, and placeholders, numbers, dates and
    times as they are."""
    value = candidate.value
    if candidate.value_type == "str" and not masked:
        return repr(value)
    return repr(value) if isinstance(value, float) else str(value)


# The text names this many row IDs of a candidate and counts the rest.
_ROW_IDS_NAMED = 3


def _row_ids_text(candidate: SentinelCandidate, *, quote: bool) -> str:
    """The first row IDs of a candidate, such as "'a', 'b', 'c' and 9 more"."""
    show = _text.label_text if quote else str
    named = [show(row_id) for row_id in candidate.row_ids[:_ROW_IDS_NAMED]]
    rest = candidate.count - len(named)
    if rest > 0:
        return f"{', '.join(named)} and {rest:,} more"
    return _text.join_items(named)
