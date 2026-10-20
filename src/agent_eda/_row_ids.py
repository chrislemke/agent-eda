"""How analyses point at rows: by index label, or by the value of id_column.

validate_column_types, detect_missing_sentinels, find_duplicate_rows and
profile_target point at rows, and all of them read row IDs here. A result holds
each row ID as pandas gives it, so df.loc[row_ids] finds the rows, and its
record converts the IDs to JSON with results.row_id_json. When IDs repeat, the
result carries an ambiguous_row_ids limitation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from agent_eda._columns import any_column, select_columns
from agent_eda._result_base import Limitation
from agent_eda._values import hashable_values, value_codes
from agent_eda._wording import Words, mistake
from agent_eda.errors import ColumnTypeError

# id_column may name a column of any dtype.
_ID_COLUMN = any_column("the index labels")


@dataclass(frozen=True)
class RowIds:
    """The data's row IDs: one per row, from the index or from id_column."""

    # The column the IDs come from, or None for the index labels.
    id_column: str | None
    # One ID per row.
    labels: pd.Index | pd.Series
    # Per row, whether another row has the same ID, as duplicated_labels decides.
    repeated: np.ndarray

    def at(self, positions: np.ndarray) -> list[object]:
        """The IDs of the rows at these positions, in that order.

        Each ID is the label as pandas gives it, such as a Timestamp or, for a
        MultiIndex, a tuple, so df.loc[ids] finds the rows.
        """
        return self.labels.take(positions).tolist()


def read_row_ids(analysis: str, data: pd.DataFrame, id_column: str | None) -> RowIds:
    """The data's row IDs: its index labels, or the values of id_column.

    An unknown id_column raises ColumnNotFoundError with the closest names, as
    any other column argument does, and one whose values can't be hashed, such
    as lists, raises ColumnTypeError.
    """
    if id_column is None:
        repeated = duplicated_labels(data.index, keep=False)
        return RowIds(id_column=None, labels=data.index, repeated=repeated)
    [column] = select_columns(
        analysis, data, [id_column], _ID_COLUMN, parameter="id_column"
    ).columns
    labels = data.iloc[:, column.position]
    if value_codes(labels, unhashable="unsupported") is None:
        raise mistake(
            ColumnTypeError,
            f"{analysis} can't use column {column.name!r} with dtype "
            f"{column.dtype} as id_column: it holds values that can't be hashed, "
            "such as lists or dicts, and so can't identify rows. Pass a column of "
            "IDs such as numbers or strings, or ",
            Words(
                "leave id_column=None to use the index labels.",
                "leave id_column out to use the default row IDs.",
            ),
        ) from None
    repeated = duplicated_labels(labels, keep=False)
    return RowIds(id_column=column.name, labels=labels, repeated=repeated)


def duplicated_labels(
    labels: pd.Index | pd.Series, *, keep: Literal["first", False]
) -> np.ndarray:
    """Which labels equal another label, as pandas' duplicated() decides.

    keep="first" marks each label that repeats an earlier one, and keep=False
    every label that another one equals. As df.loc does, pandas tells None, NaN
    and NaT apart in an object index, while equal values, missing ones of the
    same kind included, repeat. Labels pandas can't hash, such as pyarrow
    lists, compare by content. describe_dataset decides whether the index is
    unique with this too, so the analyses can't disagree. The caller's index
    builds no lookup table.
    """
    if isinstance(labels, pd.RangeIndex):
        return np.zeros(len(labels), dtype=bool)
    if isinstance(labels, pd.MultiIndex):
        # Works on the level codes alone.
        return labels.duplicated(keep=keep)
    # Index.duplicated() first asks is_unique, which builds and keeps the
    # caller's lookup table; a Series over the same values doesn't.
    values = (
        labels
        if isinstance(labels, pd.Series)
        else pd.Series(labels.array, copy=False, dtype=labels.dtype)
    )
    try:
        return values.duplicated(keep=keep).to_numpy()
    except (TypeError, NotImplementedError):  # nested pyarrow types
        stand_ins = pd.Series(hashable_values(values.array), dtype=object)
        return stand_ins.duplicated(keep=keep).to_numpy()


def ambiguous_row_ids(row_ids: RowIds, fields: Sequence[str]) -> list[Limitation]:
    """The ambiguous_row_ids limitation when row IDs repeat; none otherwise.

    `fields` are the paths of the result fields that hold a repeated row ID.
    The reason states the problem and the fix first, so that the printed line,
    cut at 120 characters, keeps both.
    """
    shared = int(np.count_nonzero(row_ids.repeated))
    if not shared:
        return []
    if row_ids.id_column is None:
        what = "an index label"
        more = " Or make the index unique."
    else:
        what = "an id_column value"
        more = ""
    if not fields:
        more += " No row ID in this result repeats."
    return [
        Limitation(
            code="ambiguous_row_ids",
            reason=(
                f"{shared:,} rows share {what}, so a row ID can match several "
                f"rows: use a unique id_column.{more}"
            ),
            fields=list(fields),
        )
    ]
