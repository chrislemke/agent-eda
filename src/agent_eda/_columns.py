"""How analyses classify columns and resolve the columns they are asked for.

Every analysis except describe_dataset refers to columns by name: the string
form of each label, which is also how results show columns.
"""

import typing
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._run import check_distinct
from agent_eda._text import (
    MAX_UNKNOWN_NAMED,
    closest_names,
    join_alternatives,
    join_items,
    join_labels,
    plural,
    short_repr,
)
from agent_eda._values import is_arrow_dictionary, value_codes
from agent_eda._wording import (
    NONE,
    Fix,
    LeaveOut,
    OneOf,
    Part,
    PythonOnly,
    Wording,
    Words,
    mistake,
)
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError, InvalidArgumentError

# Error messages name at most this many of the data's columns.
MAX_LISTED_COLUMNS = 20

ColumnKind = Literal[
    "numeric",
    "boolean",
    "datetime",
    "timedelta",
    "categorical",
    "text",
    "object",
    "other",
]
# Every kind of column, for analyses that take any column.
ALL_KINDS: frozenset[ColumnKind] = frozenset(typing.get_args(ColumnKind))

# How to convert a column of a kind that an analysis rejects, in one wording for
# every analysis: pandas code, which only a Python caller can run. {column}
# becomes the code for the column. An analysis picks the kinds it rejects from
# these, and builds its other advice around them.
_STRFTIME_MONTHS = "{column}.dt.strftime('%Y-%m')"
TO_NUMBERS: Mapping[ColumnKind, PythonOnly] = {
    "text": PythonOnly(
        "convert numbers stored as text with pd.to_numeric({column}, errors='coerce')"
    ),
    "object": PythonOnly(
        "convert numbers stored as text with pd.to_numeric({column}, errors='coerce')"
    ),
    "categorical": PythonOnly("convert numeric categories with {column}.astype(float)"),
    "boolean": PythonOnly(
        "convert False and True to 0 and 1 with {column}.astype(float)"
    ),
    "datetime": PythonOnly(
        "convert datetimes to numbers with ({column} - {column}.min()).dt.total_seconds()"
    ),
    "timedelta": PythonOnly(
        "convert durations to numbers with {column}.dt.total_seconds()"
    ),
}
TO_LABELS: Mapping[ColumnKind, PythonOnly] = {
    "datetime": PythonOnly(
        "convert datetimes to labels with {column}.astype(str), or to months with "
        + _STRFTIME_MONTHS
    ),
    "timedelta": PythonOnly("convert durations to labels with {column}.astype(str)"),
    "other": PythonOnly("convert its values to labels with {column}.astype(str)"),
}


class Labels:
    """The column labels, with a code per label that equal labels share."""

    def __init__(self, columns: pd.Index) -> None:
        flat = columns.to_flat_index()
        self.level_count = columns.nlevels
        # Iterating gives Python scalars, so label types read 'int', not 'int64'.
        self.labels: list[object] = list(flat)
        self.texts = [str(label) for label in self.labels]
        found = value_codes(flat, unhashable="by_content")
        codes, uniques = found.codes, found.uniques
        # factorize gives every missing label -1, but pandas tells None, NaN,
        # NaT and NA apart as labels: give each kind of missing label its code.
        missing_kinds: dict[str, int] = {}
        self.codes = [
            int(code)
            if code >= 0
            else missing_kinds.setdefault(
                "nan" if isinstance(label, float) else type(label).__name__,
                len(uniques) + len(missing_kinds),
            )
            for code, label in zip(codes, self.labels, strict=True)
        ]

    def show(self, positions: list[int]) -> str:
        return join_labels([self.labels[p] for p in positions])

    def repeated(self) -> list[int]:
        """Positions of every label that occurs more than once, as pandas decides."""
        occurrences = Counter(self.codes)
        return [p for p, code in enumerate(self.codes) if occurrences[code] > 1]

    def colliding(self) -> list[int]:
        """First positions of different labels that share a string form."""
        # For each string form, the first position of every different label with it.
        by_text: dict[str, dict[int, int]] = {}
        for position, (text, code) in enumerate(
            zip(self.texts, self.codes, strict=True)
        ):
            by_text.setdefault(text, {}).setdefault(code, position)
        return sorted(
            position
            for first_by_code in by_text.values()
            if len(first_by_code) > 1
            for position in first_by_code.values()
        )


def first_positions(labels: Labels, positions: list[int]) -> list[int]:
    """The first position of each label among the given positions."""
    first: dict[int, int] = {}
    for position in positions:
        first.setdefault(labels.codes[position], position)
    return sorted(first.values())


def check_column_labels(analysis: str, columns: pd.Index) -> None:
    """Raise unless every column can be named by the string form of its label."""
    fix_elsewhere = " describe_dataset describes data with any labels."
    if isinstance(columns, pd.MultiIndex):
        raise mistake(
            InvalidArgumentError,
            f"{analysis} refers to columns by name, but the column "
            f"labels form a MultiIndex with {columns.nlevels} levels. ",
            Words(
                "Flatten them first, such as df.columns = ['_'.join(map(str, c)) "
                "for c in df.columns]. ",
                "Flatten them in the data file first. ",
            ),
            fix_elsewhere.lstrip(),
            code="ambiguous_column_labels",
        )
    labels = Labels(columns)
    problems, fixes, in_file = [], [], []
    repeated = first_positions(labels, labels.repeated())
    if repeated:
        occur = "occurs" if len(repeated) == 1 else "occur"
        problems.append(f"{labels.show(repeated)} {occur} more than once")
        fixes.append(
            "df.columns = [f'{c}_{i}' if dup else c for i, (c, dup) in "
            "enumerate(zip(df.columns, df.columns.duplicated()))]"
        )
        in_file.append("the repeated columns")
    colliding = labels.colliding()
    if colliding:
        problems.append(f"{labels.show(colliding)} share a string form")
        # Rename the first label that isn't a string, else the second one.
        label = next(
            (
                labels.labels[p]
                for p in colliding
                if not isinstance(labels.labels[p], str)
            ),
            labels.labels[colliding[1]],
        )
        renamed = f"{label}_{type(label).__name__}"
        fixes.append(f"df = df.rename(columns={{{label!r}: {renamed!r}}})")
        in_file.append("the columns that share a string form")
    if problems:
        raise mistake(
            InvalidArgumentError,
            f"{analysis} refers to columns by name, the string form of each "
            "label, so labels must be unique and distinct as strings, but "
            f"{', and '.join(problems)}. ",
            Words(
                f"Rename them first, such as {' and '.join(fixes)}.",
                f"Rename {' and '.join(in_file)} in the file, then pass its path "
                "again.",
            ),
            fix_elsewhere,
            code="ambiguous_column_labels",
        )


def column_kind(dtype: object) -> ColumnKind:
    """What a column of this dtype holds, the same for NumPy, nullable and Arrow.

    The dtype's kind code is uniform across NumPy, pandas' nullable and
    pyarrow-backed dtypes; categoricals, text and Arrow decimals need checks of
    their own. An Arrow dictionary is a categorical, as the form Parquet gives
    pandas' categoricals. A sparse dtype is read by the values it stores, but
    only numbers and booleans are: sparse text, objects, datetimes and durations
    are "other", as are Arrow times of day, binary and nested types, which no
    analysis reads.
    """
    if isinstance(dtype, pd.CategoricalDtype) or is_arrow_dictionary(dtype):
        return "categorical"
    kind = getattr(dtype, "kind", "O")
    if isinstance(dtype, pd.SparseDtype) and kind not in "biuf":
        return "other"
    if kind == "b":
        return "boolean"
    if kind in "iuf":
        return "numeric"
    if kind == "M":
        return "datetime"
    if kind == "m":
        return "timedelta"
    if isinstance(dtype, pd.ArrowDtype) and pd.api.types.is_numeric_dtype(dtype):
        return "numeric"  # decimals, whose kind is "O"
    if isinstance(dtype, pd.StringDtype) or (
        isinstance(dtype, pd.ArrowDtype) and kind == "U"
    ):
        return "text"
    if isinstance(dtype, np.dtype) and kind == "O":
        return "object"
    return "other"


@dataclass(frozen=True)
class Column:
    """A column of the data, as an analysis selects it."""

    position: int
    label: Hashable
    name: str
    # The dtype as pandas shows it, such as 'Int64'.
    dtype: str
    kind: ColumnKind


@dataclass(frozen=True)
class ColumnRule:
    """The columns an analysis selects by default and the ones it accepts by name."""

    # What columns=None selects, and how to say it: "pass columns=None for ...".
    default: frozenset[ColumnKind]
    default_text: str
    # What a column named in the call may hold, and how to say it: "It accepts ...".
    accepted: frozenset[ColumnKind]
    accepted_text: str
    # For a rejected kind, how to fix the call, such as PythonOnly("convert durations
    # to numbers with {column}.dt.total_seconds()"): a fix, or the fixes in order;
    # {column} becomes the code for the column.
    fixes: Mapping[ColumnKind, Fix | tuple[Fix, ...]]
    # For a rejected kind, the analysis that fits it, such as
    # "summarize_categorical" for text; the message names it first.
    alternatives: Mapping[ColumnKind, str] = field(default_factory=dict)
    # What the message says of the rejected column, with {analysis}, {what} and
    # {accepted} filled in.
    rejection: str = "{analysis} can't use {what}. It accepts {accepted}."
    # Whether the fix can also be to leave the column out of the call.
    leave_out: bool = True


@dataclass(frozen=True)
class Selection:
    """The columns an analysis works on, and the ones its default left out."""

    columns: list[Column]
    # Names of the columns that columns=None left out for their kind, in
    # DataFrame order, without id_column; empty when the call named the columns.
    left_out: list[str]
    # The kind of each left-out column, so that a text can say why it was left out.
    left_out_kinds: dict[str, ColumnKind] = field(default_factory=dict)


def select_columns(
    analysis: str,
    data: pd.DataFrame,
    requested: Sequence[str] | None,
    rule: ColumnRule,
    *,
    parameter: str = "columns",
    id_column: str | None = None,
) -> Selection:
    """The columns to analyse: the requested ones in their order, or the defaults.

    The defaults also leave out `id_column`, the column that identifies rows,
    without listing it in `left_out`, while a request may name it. Raises before
    any computation when the request is empty, names a column twice, names an
    unknown column or names one the analysis can't handle.
    """
    columns = [
        Column(position, label, str(label), str(dtype), column_kind(dtype))
        for position, (label, dtype) in enumerate(data.dtypes.items())
    ]
    if requested is None:
        return Selection(
            columns=[
                c for c in columns if c.kind in rule.default and c.name != id_column
            ],
            left_out=[c.name for c in columns if c.kind not in rule.default],
            left_out_kinds={
                c.name: c.kind for c in columns if c.kind not in rule.default
            },
        )
    if not requested:
        raise mistake(
            InvalidArgumentError,
            f"{analysis} got an empty list for {parameter}. Name at least one "
            "column, or ",
            LeaveOut(parameter),
            f" for {rule.default_text}.",
        )
    check_distinct(analysis, parameter, requested, "column")
    by_name = {column.name: column for column in columns}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise mistake(
            ColumnNotFoundError,
            _not_found(
                analysis,
                parameter,
                unknown,
                list(by_name),
                index_names=[str(n) for n in data.index.names if n is not None],
            ),
        )
    selected = [by_name[name] for name in requested]
    rejected = [column for column in selected if column.kind not in rule.accepted]
    if rejected:
        raise mistake(ColumnTypeError, _wrong_type(analysis, parameter, rejected, rule))
    return Selection(columns=selected, left_out=[])


def _not_found(
    analysis: str,
    parameter: str,
    unknown: list[str],
    names: list[str],
    *,
    index_names: Sequence[str] = (),
) -> Wording:
    if names:
        listed = join_items([repr(n) for n in names], limit=MAX_LISTED_COLUMNS)
        available = f"The data has {plural(len(names), 'column', 'columns')}: {listed}."
    else:
        available = "The data has no columns."
    # Each unknown name with the names it most likely means, if any are close.
    # Only the first few are named, each cut short: an option given a column's
    # values in place of names would otherwise put the values into the message.
    shown = unknown[:MAX_UNKNOWN_NAMED]
    matches = {
        name: join_alternatives([repr(m) for m in closest_names(name, names)])
        for name in shown
    }
    if len(unknown) == 1:
        [name] = unknown
        if parameter == "id_column" and name in index_names:
            return (
                f"{analysis} got id_column {name!r}, which names the index, not a "
                "column. The index labels are already the row IDs: ",
                LeaveOut(parameter),
                " to use them. To use the index as a column instead, ",
                PythonOnly(
                    "call df.reset_index() first",
                    otherwise="write it as a column in the data file",
                ),
                f". {available}",
            )
        hint: tuple[Wording, ...] = (
            (f" Did you mean {matches[name]}?",) if matches[name] else ()
        )
        if name == "None" and parameter.endswith("columns"):
            hint += (
                " To select every suitable column, ",
                LeaveOut(parameter),
                " rather than a list holding ",
                NONE,
                ".",
            )
        return (
            f"{analysis} got an unknown column {short_repr(name)} in {parameter}.",
            *hint,
            f" {available}",
        )
    described = [
        f"{short_repr(name)} (did you mean {match}?)" if match else short_repr(name)
        for name, match in matches.items()
    ]
    return (
        f"{analysis} got {len(unknown):,} unknown columns in {parameter}: "
        f"{join_items(described, limit=MAX_UNKNOWN_NAMED, total=len(unknown))}. "
        f"{available}"
    )


def _wrong_type(
    analysis: str, parameter: str, rejected: list[Column], rule: ColumnRule
) -> Wording:
    # Each analysis that fits some of the columns, with the columns it fits.
    fitting: dict[str, list[str]] = {}
    for column in rejected:
        other = rule.alternatives.get(column.kind)
        if other is not None:
            fitting.setdefault(other, []).append(repr(column.name))
    # Each fix once, shown with the first column it applies to. A fix is
    # pandas code, which only a Python caller can run.
    fixes: dict[str, Part] = {}
    for column in rejected:
        given = rule.fixes.get(column.kind, ())
        for template in given if isinstance(given, tuple) else (given,):
            fixes.setdefault(
                template.render("python"),
                template.bind(column=f"df[{column.label!r}]"),
            )
    if len(rejected) == 1:
        [column] = rejected
        what = f"column {column.name!r} with dtype {column.dtype}"
        leave = f"leave {column.name!r} out of {parameter}"
    else:
        named = [f"{c.name!r} with dtype {c.dtype}" for c in rejected]
        listed = join_items(named, limit=MAX_LISTED_COLUMNS)
        what = f"{len(rejected):,} columns: {listed}"
        leave = f"leave them out of {parameter}"
    steps: list[Wording] = [
        f"use {other} for {join_items(names, limit=MAX_LISTED_COLUMNS)} instead"
        for other, names in fitting.items()
    ]
    steps.extend(fixes.values())
    return (
        rule.rejection.format(
            analysis=analysis, what=what, accepted=rule.accepted_text
        ),
        " ",
        OneOf(steps, last=leave if rule.leave_out else None, capital=True),
        ".",
    )
