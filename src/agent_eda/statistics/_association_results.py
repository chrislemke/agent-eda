"""The result of compute_association_matrix."""

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import (
    Field,
)

from agent_eda import _text
from agent_eda._result_base import (
    _BY_NAME,
    _ENTRIES,
    AnalysisResult,
    ResultModel,
    _entries_frame,
)

# How compute_association_matrix treats a column it pairs.
AssociationType = Literal["numeric", "categorical"]

# The measures compute_association_matrix computes, one per pair.
AssociationMethod = Literal[
    "pearson",
    "spearman",
    "kendall",
    "cramers_v",
    "cramers_v_bias_corrected",
    "correlation_ratio_bias_corrected",
    "correlation_ratio",
]

# Why compute_association_matrix pairs a column with no other.
LeftOutReason = Literal[
    "datetime",
    "timedelta",
    "unsupported_dtype",
    "all_missing",
    "no_finite_values",
    "unhashable_values",
    "too_many_categories",
]

# Why a pair's value doesn't exist.
UndefinedReason = Literal[
    "too_few_rows", "single_level", "constant_column", "one_row_per_level"
]


class AssociationColumn(ResultModel):
    """A column compute_association_matrix paired with the others."""

    column: str = Field(description="The column name: the label's string form.")
    dtype: str = Field(description="The dtype as pandas reports it, such as 'Int64'.")
    type: AssociationType = Field(
        description="How the pairs treat the column: numeric (numbers other than "
        "booleans, unless listed in categorical_columns) or categorical (booleans, "
        "text, objects, categoricals and listed numbers)."
    )
    value_count: int = Field(
        description="Values the pairs can use: present, and finite in a numeric column."
    )
    missing_count: int = Field(
        description="Native missing values, including NaN that a pyarrow-backed "
        "or nullable float column stores as a value."
    )
    infinite_count: int = Field(
        description="Values equal to inf or -inf in a numeric column, which the "
        "pairs leave out; 0 in a categorical column, where they are levels."
    )
    distinct_count: int | None = Field(
        description="Distinct values of a categorical column, missing values not "
        "counted; null for a numeric column."
    )


class LeftOutColumn(ResultModel):
    """A column compute_association_matrix paired with no other, and why."""

    column: str = Field(description="The column name: the label's string form.")
    dtype: str = Field(description="The dtype as pandas reports it.")
    reason: LeftOutReason = Field(
        description="Why it was left out: datetime, timedelta or unsupported_dtype "
        "for dtypes without a measure (only when the call leaves columns out); all_missing; "
        "no_finite_values for a numeric column holding only inf and -inf; "
        "unhashable_values for values such as lists; too_many_categories for a "
        "categorical column with more distinct values than max_categories."
    )
    message: str = Field(description="One sentence on why it was left out.")


class AssociationPair(ResultModel):
    """How strongly two columns are associated, as compute_association_matrix found."""

    x_column: str = Field(
        description="The first column: the earlier one in the selected order, or "
        "the categorical column of a correlation ratio."
    )
    y_column: str = Field(
        description="The second column: the later one, or the numeric column of a "
        "correlation ratio."
    )
    x_type: AssociationType = Field(description="How the pair treats x_column.")
    y_type: AssociationType = Field(description="How the pair treats y_column.")
    method: AssociationMethod = Field(
        description="The measure: numeric_method for two numeric columns, "
        "categorical_method for two categorical ones, mixed_method from the "
        "categorical to the numeric column otherwise."
    )
    value: float | None = Field(
        description="The measure on the pair's complete rows: from -1 to 1 for "
        "pearson, spearman and kendall, from 0 to 1 for the others; null when it "
        "doesn't exist, and undefined_reason says why."
    )
    pair_count: int = Field(
        description="The pair's complete rows, which the value uses: both values "
        "present, and finite in a numeric column."
    )
    direction: Literal["symmetric", "x_to_y"] = Field(
        description="x_to_y for the correlation ratio, plain or bias-corrected, "
        "which measures how much the categorical x_column explains the numeric "
        "y_column and not the reverse; symmetric for the other measures."
    )
    undefined_reason: UndefinedReason | None = Field(
        description="Why value is null: too_few_rows (fewer than 3 complete rows), "
        "single_level (a categorical column has one level in them), "
        "constant_column (a numeric column has one value in them) or "
        "one_row_per_level (a categorical column has a level for every complete "
        "row, such as an ID, so it would explain the other trivially); null when "
        "value exists."
    )


def _pair_key(pair: "AssociationPair") -> tuple[str, str]:
    """What a field path calls a pair: its two columns."""
    return pair.x_column, pair.y_column


class ComputeAssociationMatrixResult(AnalysisResult):
    """The result of compute_association_matrix."""

    pairs: Annotated[list[AssociationPair], _ENTRIES.derive(key=_pair_key)] = Field(
        description="One entry per pair of paired columns, strongest first: by "
        "absolute value, descending, then in the selected column order, with null "
        "values last. The record keeps the {limit} strongest and adds pairs_total; "
        "the complete table has every pair."
    )
    columns: Annotated[dict[str, AssociationColumn], _ENTRIES, _BY_NAME] = Field(
        description="The paired columns, keyed by column name, in the selected "
        "order. {cap}."
    )
    left_out_columns: Annotated[list[LeftOutColumn], _ENTRIES] = Field(
        description="The columns paired with no other, with the reason, in "
        "DataFrame order when the call leaves columns out and in the given order "
        "otherwise."
    )

    TABLES: ClassVar[Mapping[str, str | None]] = {
        "pairs": None,
        "matrix": "has the wide {size} x {size} matrix",
    }

    def _table_footer_fields(self) -> dict[str, str]:
        return {"size": f"{len(self.columns):,}"}

    def _text_other_tables(self) -> list[str]:
        # Nothing was computed without pairs, so the matrix would hold no value.
        return super()._text_other_tables() if self.pairs else []

    def to_frame(self, table: Literal["pairs", "matrix"] = "pairs") -> pd.DataFrame:
        """A complete table of the result, chosen by name.

        Args:
            table: "pairs", the default, gives one row per pair with every field
                of res.pairs, strongest first. "matrix" gives the paired columns
                as rows and columns in the selected order: the cell in row x and
                column y holds the pair's value, NaN where it is null. A
                correlation ratio fills only the cell from its categorical row
                to its numeric column, and the diagonal is empty.
        """
        self._check_table(table)
        if table == "pairs":
            return _entries_frame(self.pairs, AssociationPair)
        names = list(self.columns)
        position = {name: i for i, name in enumerate(names)}
        cells = np.full((len(names), len(names)), np.nan)
        for pair in self.pairs:
            value = np.nan if pair.value is None else pair.value
            x, y = position[pair.x_column], position[pair.y_column]
            cells[x, y] = value
            if pair.direction == "symmetric":
                cells[y, x] = value
        return pd.DataFrame(
            cells,
            index=pd.Index(names, name="x_column"),
            columns=pd.Index(names, name="y_column"),
        )

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        facts = []
        if self.pairs:
            counts = dict.fromkeys(("numeric", "categorical", "mixed"), 0)
            for pair in self.pairs:
                kind = pair.x_type if pair.x_type == pair.y_type else "mixed"
                counts[kind] += 1
            facts.extend(
                [
                    "Pairs, each on its own complete rows: "
                    f"{counts['numeric']:,} numeric "
                    f"({self.settings.numeric_method}), "
                    f"{counts['categorical']:,} categorical "
                    f"({self.settings.categorical_method}),",
                    f"    {counts['mixed']:,} mixed ({self.settings.mixed_method}, "
                    "from the categorical x_column to the numeric y_column).",
                ]
            )
        if self.left_out_columns:
            left_out = [f"{s.column!r} ({s.reason})" for s in self.left_out_columns]
            facts.extend(
                _text.wrap(
                    "Left out "
                    f"{_text.plural(len(left_out), 'column', 'columns')}, see "
                    f"{_text.field_name('left_out_columns', surface)}: "
                    f"{_text.join_items(left_out)}.",
                    first="",
                    indent="    ",
                )
            )
        return facts

    def _text_table(self) -> _text.Table:
        return _text.Table(
            title="Pairs",
            unit=("pair", "pairs"),
            headers=(
                "x_column",
                "y_column",
                "method",
                "value",
                "pair_count",
                "direction",
            ),
            rows=[
                (
                    p.x_column,
                    p.y_column,
                    p.method,
                    p.value,
                    p.pair_count,
                    p.direction,
                )
                for p in self.pairs[: _text.MAX_TABLE_ROWS]
            ],
            total=len(self.pairs),
        )
