"""How analyses classify columns and resolve the columns they are asked for.

Every analysis except describe_dataset refers to columns by name: the string
form of each label, which is also how results show columns.
"""

import difflib
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._run import check_distinct
from agent_eda._text import join_alternatives, join_items, join_labels, plural
from agent_eda.errors import ColumnNotFoundError, ColumnTypeError, InvalidArgumentError

# Error messages name at most this many of the data's columns.
MAX_LISTED_COLUMNS = 20
# Suggestions for a misspelled column name.
MAX_SUGGESTIONS = 3

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


class Labels:
    """The column labels, with a code per label that equal labels share."""

    def __init__(self, columns: pd.Index) -> None:
        flat = columns.to_flat_index()
        self.level_count = columns.nlevels
        # Iterating gives Python scalars, so label types read 'int', not 'int64'.
        self.labels: list[object] = list(flat)
        self.texts = [str(label) for label in self.labels]
        codes, uniques = flat.factorize()
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
        raise InvalidArgumentError(
            f"{analysis} refers to columns by name, but the column labels form a "
            f"MultiIndex with {columns.nlevels} levels. Flatten them first, such "
            "as df.columns = ['_'.join(map(str, c)) for c in df.columns]. "
            f"{fix_elsewhere}",
            code="ambiguous_column_labels",
        )
    labels = Labels(columns)
    problems, fixes = [], []
    repeated = first_positions(labels, labels.repeated())
    if repeated:
        occur = "occurs" if len(repeated) == 1 else "occur"
        problems.append(f"{labels.show(repeated)} {occur} more than once")
        fixes.append(
            "df.columns = [f'{c}_{i}' if dup else c for i, (c, dup) in "
            "enumerate(zip(df.columns, df.columns.duplicated()))]"
        )
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
    if problems:
        raise InvalidArgumentError(
            f"{analysis} refers to columns by name, the string form of each label, "
            "so labels must be unique and distinct as strings, but "
            f"{', and '.join(problems)}. Rename them first, such as "
            f"{' and '.join(fixes)}. {fix_elsewhere}",
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
    # Names of the columns that columns=None left out, in DataFrame order;
    # empty when the call named the columns.
    left_out: list[str]


def select_columns(
    analysis: str,
    data: pd.DataFrame,
    requested: Sequence[str] | None,
    rule: ColumnRule,
    *,
    parameter: str = "columns",
) -> Selection:
    """The columns to analyse: the requested ones in their order, or the defaults.

    Raises before any computation when the request is empty, names a column
    twice, names an unknown column or names one the analysis can't handle.
    """
    columns = [
        Column(position, label, str(label), str(dtype), column_kind(dtype))
        for position, (label, dtype) in enumerate(data.dtypes.items())
    ]
    if requested is None:
        return Selection(
            columns=[c for c in columns if c.kind in rule.default],
            left_out=[c.name for c in columns if c.kind not in rule.default],
        )
    if not requested:
        raise InvalidArgumentError(
            f"{analysis} got an empty list for {parameter}. Name at least one "
            f"column, or pass {parameter}=None for {rule.default_text}."
        )
    check_distinct(analysis, parameter, requested, "column")
    by_name = {column.name: column for column in columns}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ColumnNotFoundError(
            _not_found(analysis, parameter, unknown, list(by_name))
        )
    selected = [by_name[name] for name in requested]
    rejected = [column for column in selected if column.kind not in rule.accepted]
    if rejected:
        raise ColumnTypeError(_wrong_type(analysis, parameter, rejected, rule))
    return Selection(columns=selected, left_out=[])


def closest_names(name: str, names: Sequence[str]) -> list[str]:
    """The names closest to `name`, at most MAX_SUGGESTIONS of them.

    Names equal to it apart from case and surrounding spaces come first, then
    close spellings, ranked by difflib's similarity.
    """
    by_key: dict[str, list[str]] = {}
    for candidate in names:
        by_key.setdefault(_fold(candidate), []).append(candidate)
    key = _fold(name)
    similar = difflib.get_close_matches(key, list(by_key), n=MAX_SUGGESTIONS)
    found = [*by_key.get(key, []), *(c for k in similar for c in by_key[k])]
    return list(dict.fromkeys(found))[:MAX_SUGGESTIONS]


def _fold(name: str) -> str:
    return name.strip().casefold()


def _not_found(
    analysis: str, parameter: str, unknown: list[str], names: list[str]
) -> str:
    if names:
        listed = join_items([repr(n) for n in names], limit=MAX_LISTED_COLUMNS)
        available = f"The data has {plural(len(names), 'column', 'columns')}: {listed}."
    else:
        available = "The data has no columns."
    # Each unknown name with the names it most likely means, if any are close.
    matches = {
        name: join_alternatives([repr(m) for m in closest_names(name, names)])
        for name in unknown
    }
    if len(unknown) == 1:
        [name] = unknown
        hint = f" Did you mean {matches[name]}?" if matches[name] else ""
        return f"{analysis} got an unknown column {name!r} in {parameter}.{hint} {available}"
    described = [
        f"{name!r} (did you mean {match}?)" if match else repr(name)
        for name, match in matches.items()
    ]
    return (
        f"{analysis} got {len(unknown):,} unknown columns in {parameter}: "
        f"{join_items(described, limit=MAX_LISTED_COLUMNS)}. {available}"
    )


def _wrong_type(
    analysis: str, parameter: str, rejected: list[Column], rule: ColumnRule
) -> str:
    # Each analysis that fits some of the columns, with the columns it fits.
    fitting: dict[str, list[str]] = {}
    for column in rejected:
        other = rule.alternatives.get(column.kind)
        if other is not None:
            fitting.setdefault(other, []).append(repr(column.name))
    # Each fix once, shown with the first column it applies to.
    fixes: dict[str, str] = {}
    for column in rejected:
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
