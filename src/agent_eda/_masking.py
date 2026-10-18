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

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._levels import Levels, label_types, ordered
from agent_eda._text import placeholder
from agent_eda.errors import InvalidArgumentError
from agent_eda.project import Project
from agent_eda.results import Masking

__all__ = [
    "MaskedLevels",
    "MaskingDecision",
    "mask_levels",
    "mask_values",
    "placeholder",
    "resolve_masking",
]

# A call that declares sensitive columns, for messages that show how.
_EXAMPLE = "project=ae.Project(data_policy=ae.DataPolicy(sensitive_columns=['email']))"


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
        present = {str(label) for label in data.columns} | set(index)
        return Masking(
            applied=self.applied,
            decided_by=self.decided_by,
            columns=columns,
            index_columns=(
                [c for c in self.sensitive_columns if c in index]
                if None in shown
                else []
            ),
            absent_columns=[c for c in self.sensitive_columns if c not in present],
        )


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
            raise InvalidArgumentError(
                f"{analysis} got mask_sensitive=True without a project, so no "
                "column is declared sensitive and there would be nothing to mask. "
                f"Pass a project that names the sensitive columns, such as {_EXAMPLE}."
            )
        raise InvalidArgumentError(
            f"{analysis} got mask_sensitive=True, but the project declares no "
            "sensitive columns, so there would be nothing to mask. Name them in its "
            f"data policy, such as {_EXAMPLE}."
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
        masked_types=label_types(levels, order),
    )
    return MaskedLevels(levels=masked, numbers=numbers)


def mask_values(values: Sequence[object]) -> list[str | None]:
    """Numbered placeholders for values in the order given, such as row IDs.

    Equal values share a placeholder, numbered in order of first appearance,
    so values that repeat still show as repeating. Missing values (None, NaN,
    NaT, pd.NA) stay None. Pass the values of one column in result order.
    """
    # A value that isn't text keeps pandas off its hash table for text, which
    # reads '\x00' as the end of the text, so '' and '\x00a' would be equal.
    # fromiter keeps tuples whole, where np.array would make them a dimension.
    marked = np.fromiter(
        itertools.chain(values, [_NOT_TEXT]), dtype=object, count=len(values) + 1
    )
    codes, _ = pd.factorize(marked)
    return [placeholder(int(code) + 1) if code >= 0 else None for code in codes[:-1]]


# mask_values appends this to the values it numbers, so they are never all
# text. It comes last, so no value gets its code.
_NOT_TEXT = object()
