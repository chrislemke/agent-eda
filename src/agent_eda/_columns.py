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
    Surface,
    closest_names,
    join_alternatives,
    join_items,
    join_labels,
    plural,
    short_repr,
)
from agent_eda._values import value_codes
from agent_eda._wording import left_out, worded
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
    fix_elsewhere = "describe_dataset describes data with any labels."
    if isinstance(columns, pd.MultiIndex):
        raise worded(
            InvalidArgumentError,
            lambda surface: (
                f"{analysis} refers to columns by name, but the column "
                f"labels form a MultiIndex with {columns.nlevels} levels. "
                + (
                    "Flatten them first, such as df.columns = ['_'.join(map(str, c)) "
                    "for c in df.columns]. "
                    if surface == "python"
                    else "Flatten them in the data file first. "
                )
                + fix_elsewhere
            ),
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
        raise worded(
            InvalidArgumentError,
            lambda surface: (
                f"{analysis} refers to columns by name, the string form of each "
                "label, so labels must be unique and distinct as strings, but "
                f"{', and '.join(problems)}. "
                + (
                    f"Rename them first, such as {' and '.join(fixes)}."
                    if surface == "python"
                    else f"Rename {' and '.join(in_file)} in the file, then pass "
                    "its path again."
                )
                + f" {fix_elsewhere}"
            ),
            code="ambiguous_column_labels",
        )


def column_kind(dtype: object) -> ColumnKind:
    """What a column of this dtype holds, the same for NumPy, nullable and Arrow.

    The dtype's kind code is uniform across NumPy, pandas' nullable and
    pyarrow-backed dtypes; categoricals, text and Arrow decimals need checks of
    their own.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    kind = getattr(dtype, "kind", "O")
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
    # For a rejected kind, how to fix the call, such as "convert durations with
    # {column}.dt.total_seconds()"; {column} becomes the code for the column.
    fixes: Mapping[ColumnKind, str]
    # For a rejected kind, the analysis that fits it, such as
    # "summarize_categorical" for text; the message names it first.
    alternatives: Mapping[ColumnKind, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """The columns an analysis works on, and the ones its default left out."""

    columns: list[Column]
    # Names of the columns that columns=None left out for their kind, in
    # DataFrame order, without id_column; empty when the call named the columns.
    left_out: list[str]


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
        )
    if not requested:
        raise worded(
            InvalidArgumentError,
            lambda surface: (
                f"{analysis} got an empty list for {parameter}. Name at "
                f"least one column, or {left_out(parameter, surface)} for "
                f"{rule.default_text}."
            ),
        )
    check_distinct(analysis, parameter, requested, "column")
    by_name = {column.name: column for column in columns}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise worded(
            ColumnNotFoundError,
            lambda surface: _not_found(
                analysis,
                parameter,
                unknown,
                list(by_name),
                index_names=[str(n) for n in data.index.names if n is not None],
                surface=surface,
            ),
        )
    selected = [by_name[name] for name in requested]
    rejected = [column for column in selected if column.kind not in rule.accepted]
    if rejected:
        raise worded(
            ColumnTypeError,
            lambda surface: _wrong_type(analysis, parameter, rejected, rule, surface),
        )
    return Selection(columns=selected, left_out=[])


def _not_found(
    analysis: str,
    parameter: str,
    unknown: list[str],
    names: list[str],
    *,
    index_names: Sequence[str] = (),
    surface: Surface = "python",
) -> str:
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
                f"column. The index labels are already the row IDs: "
                f"{left_out(parameter, surface)} to use them. To use the index as a "
                f"column instead, {_as_column(surface)}. {available}"
            )
        hint = f" Did you mean {matches[name]}?" if matches[name] else ""
        if name == "None" and parameter.endswith("columns"):
            none = "None" if surface == "python" else "null"
            hint += (
                f" To select every suitable column, {left_out(parameter, surface)} "
                f"rather than a list holding {none}."
            )
        return (
            f"{analysis} got an unknown column {short_repr(name)} in {parameter}."
            f"{hint} {available}"
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


def _as_column(surface: Surface) -> str:
    """How to turn the index into a column, for the surface."""
    if surface == "python":
        return "call df.reset_index() first"
    return "write it as a column in the data file"


def _wrong_type(
    analysis: str,
    parameter: str,
    rejected: list[Column],
    rule: ColumnRule,
    surface: Surface,
) -> str:
    # Each analysis that fits some of the columns, with the columns it fits.
    fitting: dict[str, list[str]] = {}
    for column in rejected:
        other = rule.alternatives.get(column.kind)
        if other is not None:
            fitting.setdefault(other, []).append(repr(column.name))
    # Each fix once, shown with the first column it applies to. A fix is
    # pandas code, which only a Python caller can run.
    fixes: dict[str, str] = {}
    for column in rejected if surface == "python" else []:
        template = rule.fixes.get(column.kind)
        if template is not None and template not in fixes:
            fixes[template] = template.format(column=f"df[{column.label!r}]")
    if len(rejected) == 1:
        [column] = rejected
        what = f"column {column.name!r} with dtype {column.dtype}"
        leave = f"leave {column.name!r} out of {parameter}"
    else:
        named = [f"{c.name!r} with dtype {c.dtype}" for c in rejected]
        listed = join_items(named, limit=MAX_LISTED_COLUMNS)
        what = f"{len(rejected):,} columns: {listed}"
        leave = f"leave them out of {parameter}"
    steps = [
        f"use {other} for {join_items(names, limit=MAX_LISTED_COLUMNS)} instead"
        for other, names in fitting.items()
    ]
    steps.extend(fixes.values())
    if not steps:
        fix = leave
    elif len(steps) == 1:
        fix = f"{steps[0]}, or {leave}"
    else:
        fix = "; ".join([*steps, f"or {leave}"])
    return (
        f"{analysis} can't use {what}. It accepts {rule.accepted_text}. "
        f"{fix[0].upper()}{fix[1:]}."
    )
