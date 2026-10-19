"""Which rows of a DataFrame equal another row, and in which group."""

import numpy as np
import pandas as pd

from agent_eda._values import column_values, value_codes

# Keys combine the codes of the columns read so far. Before a key could grow past
# this, keys are renumbered, so int64 never overflows.
_KEY_LIMIT = 2**62
# Dropping rows whose key is unique keeps paying off while it drops at least
# this share of the candidate rows.
_DROP_SHARE = 0.1


def find_row_groups(
    data: pd.DataFrame, positions: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The rows equal to another row in the columns at `positions`, and their groups.

    Returns the rows' positions in ascending order and each one's group number.
    Groups are numbered from 0 in the order of their first row.

    Each candidate row carries a key that rows equal in the columns read so far
    share, extended column by column. While the keys can tell rows apart, rows
    whose key no other row has stop being candidates, so the remaining columns
    are read only for the rows that can still have duplicates: with few
    duplicates, a few columns settle every row. Once that stops dropping rows,
    as when most rows have duplicates, the keys keep growing and are only
    renumbered before they could overflow.
    """
    row_count = len(data)
    candidates = np.arange(row_count)
    keys = np.zeros(row_count, dtype=np.int64)
    span = 1  # every key lies in range(span)
    dropping = True
    for position in positions:
        codes, code_count = _codes(data.iloc[:, position], candidates, row_count)
        if span * code_count > _KEY_LIMIT:
            keys, span = _renumber(keys)
        keys = keys * code_count + codes
        span *= code_count
        if dropping and span >= candidates.size:
            keys, span = _renumber(keys)
            keep = np.bincount(keys)[keys] >= 2
            dropped = candidates.size - int(np.count_nonzero(keep))
            dropping = dropped >= _DROP_SHARE * candidates.size
            if dropped:
                candidates, keys = candidates[keep], keys[keep]
            if not candidates.size:
                break
    keys, _ = _renumber(keys)
    keep = np.bincount(keys)[keys] >= 2
    numbers, _ = _renumber(keys[keep])
    return candidates[keep], numbers


def _codes(
    column: pd.Series, rows: np.ndarray, row_count: int
) -> tuple[np.ndarray, int]:
    """A code per row that equal values share, and how many codes there can be.

    Codes start at 1, and every missing value gets 0, so missing values compare
    equal. Values that can't be hashed, such as lists and dicts, compare by
    their content, while every other value in the column still compares as
    pandas compares it, so which rows are left doesn't change the result.
    """
    values = column_values(column)
    if rows.size < row_count:
        values = values.take(rows)
    found = value_codes(values, unhashable="by_content")
    return found.codes.astype(np.int64, copy=False) + 1, len(found.uniques) + 1


def _renumber(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Number the distinct keys from 0 in order of appearance, and count them."""
    numbers, uniques = pd.factorize(keys)
    return numbers.astype(np.int64, copy=False), len(uniques)
