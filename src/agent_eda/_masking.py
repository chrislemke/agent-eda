"""Masking: numbered placeholders in place of the values of sensitive columns.

Every analysis whose result can show raw values (category labels, example
values, sentinel candidates, class labels, row IDs from id_column) supports
masking. It registers with masking=True, takes `project: Project | None = None`
and `mask_sensitive: bool | None = None`, returns a results.MaskableResult and:

1. calls resolve_masking() before any computation, which raises for masking
   without sensitive columns;
2. for each column whose values the result shows and decision.masks(column)
   holds, puts placeholders in place of the values: mask_levels() for levels
   or classes, mask_values() for row IDs or example values, one call per
   column with the values in result order;
3. records decision.record(shown, data) as the result's `masking` field.

The raw values never enter the result, so the printed text, to_dict(),
to_json() and to_frame() all read the same placeholders. Placeholders are
numbered per column in result order, '<masked 1>', '<masked 2>' and so on, and
reveal neither content nor length. Counts, shares and statistics stay. Index
labels are never masked.

Masking is a guardrail against accidental exposure, not anonymization and not
access control (ADR 0001).
"""

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._levels import Levels, label_types, ordered
from agent_eda._result_base import Masking
from agent_eda._text import placeholder
from agent_eda._values import value_codes
from agent_eda._wording import DeclareSensitive, Pass, Words, mistake
from agent_eda.errors import InvalidArgumentError
from agent_eda.project import Project

__all__ = [
    "MaskedLevels",
    "MaskingDecision",
    "mask_levels",
    "mask_values",
    "placeholder",
    "resolve_masking",
]

# A call that declares sensitive columns, for messages that show how.
_EXAMPLE = DeclareSensitive(["email"])


@dataclass(frozen=True)
class MaskingDecision:
    """Whether a call masks, what decided it, and the columns it concerns."""

    applied: bool
    decided_by: Literal["mask_sensitive", "project", "default"]
    # The columns the project declares sensitive, in its order.
    sensitive_columns: tuple[str, ...]

    def masks(self, column: str | None) -> bool:
        """Whether the result shows this column's values as placeholders."""
        return self.applied and column is not None and column in self.sensitive_columns

    def hidden(self, names: Iterable[str] | None) -> set[str]:
        """The columns among `names` whose values no message may quote.

        Error messages that would quote a value the caller declared for a
        column, such as a sentinel value or an expected label, leave it out
        when the call masks that column.
        """
        return {name for name in names or () if self.masks(name)}

    def record(self, shown: Iterable[str | None], data: pd.DataFrame) -> Masking:
        """The result's masking field.

        `shown` are the columns whose values the result shows, in result
        order, such as the selected columns or id_column; None stands for the
        index labels as row IDs, which are never masked.
        """
        shown = list(shown)
        sensitive = set(self.sensitive_columns)
        columns = [c for c in dict.fromkeys(shown) if c is not None and c in sensitive]
        # The index's level names; an unnamed level has none.
        index = [str(name) for name in data.index.names if name is not None]
        names = [str(label) for label in data.columns] + index
        present = set(names)
        absent = [c for c in self.sensitive_columns if c not in present]
        # A name the data spells with other case or Unicode form is no match, as
        # masking matches exactly, but it is likely meant: report it.
        spelled: dict[str, str] = {}
        for name in names:
            spelled.setdefault(_loosely(name), name)
        near_misses = {
            c: spelled[_loosely(c)]
            for c in absent
            if _loosely(c) in spelled and spelled[_loosely(c)] not in sensitive
        }
        return Masking(
            applied=self.applied,
            decided_by=self.decided_by,
            columns=columns,
            index_columns=(
                [c for c in self.sensitive_columns if c in index]
                if None in shown
                else []
            ),
            absent_columns=absent,
            near_miss_columns=near_misses,
        )


def _loosely(name: str) -> str:
    """The name with case and Unicode form ignored, to find a near miss."""
    return unicodedata.normalize("NFC", name.casefold())


def resolve_masking(
    analysis: str, project: Project | None, mask_sensitive: bool | None
) -> MaskingDecision:
    """Decide whether a call masks: a per-call True or False wins, then the project.

    Without a project nothing is masked. Masking with no sensitive column
    declared raises InvalidArgumentError, so nobody believes data is masked
    when it isn't.
    """
    sensitive = project.data_policy.sensitive_columns if project is not None else ()
    if mask_sensitive is not None:
        decision = MaskingDecision(mask_sensitive, "mask_sensitive", sensitive)
    elif project is not None:
        decision = MaskingDecision(
            project.data_policy.mask_sensitive, "project", sensitive
        )
    else:
        decision = MaskingDecision(False, "default", sensitive)
    if decision.applied and not sensitive:
        if project is None:
            raise mistake(
                InvalidArgumentError,
                f"{analysis} got ",
                Pass(mask_sensitive=True),
                " without a project, so no column is declared sensitive and there "
                "would be nothing to mask. ",
                Words(
                    "Pass a project that names the sensitive columns",
                    "Pass a project_path whose [data_policy] table names the "
                    "sensitive columns",
                ),
                ", such as ",
                _EXAMPLE,
                ".",
            )
        raise mistake(
            InvalidArgumentError,
            f"{analysis} got ",
            Pass(mask_sensitive=True),
            ", but the project declares no sensitive columns, so there would be "
            "nothing to mask. ",
            Words(
                "Name them in its data policy",
                "Name them in the [data_policy] table of its project file",
            ),
            ", such as ",
            _EXAMPLE,
            ".",
        )
    return decision


@dataclass(frozen=True)
class MaskedLevels:
    """A column's levels with numbered placeholders in place of the labels."""

    # The levels as a result keeps them for to_frame(): the placeholders'
    # numbers in level order, with the type name of each value they stand
    # for, and no label.
    levels: Levels
    # For each position of the original levels, its placeholder's number.
    numbers: np.ndarray

    def label(self, position: int) -> str:
        """The placeholder of the original level at this position."""
        return placeholder(int(self.numbers[position]))

    @property
    def unused_categories(self) -> list[str]:
        """The placeholders of a categorical dtype's unused categories."""
        return [str(category) for category in self.levels.unused_categories]


def mask_levels(levels: Levels) -> MaskedLevels:
    """Put numbered placeholders in place of a column's labels, in level order.

    The k-th level in level order becomes '<masked k>', and a categorical
    dtype's unused categories follow the levels in category order. Counts and
    the labels' types stay. The masked levels hold no label, so a result that
    keeps them for to_frame() keeps no raw value.
    """
    order = ordered(levels)
    size = len(order)
    numbers = np.empty(size, dtype=np.intp)
    numbers[order] = np.arange(1, size + 1)
    unused = len(levels.unused_categories)
    masked = replace(
        levels,
        values=pd.RangeIndex(1, size + 1),
        counts=levels.counts[order],
        sort_values=None,
        text=False,
        unused_categories=[placeholder(size + n) for n in range(1, unused + 1)],
        placeholder_types=label_types(levels, order),
    )
    return MaskedLevels(levels=masked, numbers=numbers)


def mask_values(
    values: Sequence[object],
    *,
    first: int = 1,
    known: Sequence[tuple[object, str]] = (),
) -> list[str | None]:
    """Numbered placeholders for values in the order given, such as row IDs.

    Equal values share a placeholder, numbered in order of first appearance,
    so values that repeat still show as repeating. Missing values (None, NaN,
    NaT, pd.NA) stay None. Pass the values of one column in result order.

    `first` is the number of the first new placeholder, for values that
    continue the numbering of another call. `known` pairs values that already
    have a placeholder with it: a value equal to one of them takes that
    placeholder and uses no number.
    """
    # fromiter keeps tuples whole, where np.array would make them a dimension.
    objects = np.fromiter(
        [*(value for value, _ in known), *values],
        dtype=object,
        count=len(known) + len(values),
    )
    codes = value_codes(objects, unhashable="by_content").codes
    fixed = {int(code): text for code, (_, text) in zip(codes, known, strict=False)}
    numbered: dict[int, str] = {}
    placeholders: list[str | None] = []
    for code in codes[len(known) :].tolist():
        if code < 0:
            placeholders.append(None)
        elif code in fixed:
            placeholders.append(fixed[code])
        else:
            if code not in numbered:
                numbered[code] = placeholder(first + len(numbered))
            placeholders.append(numbered[code])
    return placeholders
