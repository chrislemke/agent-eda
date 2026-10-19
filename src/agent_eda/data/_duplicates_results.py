"""The result of find_duplicate_rows."""

from typing import Annotated, Literal

import pandas as pd
from pydantic import (
    Field,
)

from agent_eda import _text
from agent_eda._result_base import (
    _BIG_INTEGERS,
    _ENTRIES,
    _ROW_IDS,
    MaskableResult,
    ResultModel,
    RowId,
    _check_table,
)


class DuplicateGroup(ResultModel):
    """Rows that are equal in every compared column."""

    size: int = Field(description="Rows in the group, at least 2.")
    row_ids: Annotated[list[RowId], _ROW_IDS] = Field(
        description="The rows' IDs in row order: index labels, or values of "
        "id_column when it is given. With masking on and a sensitive id_column, "
        "numbered placeholders, '<masked 1>' for the first ID the groups show "
        "and so on, equal IDs alike. The record keeps the first 10 and adds "
        "row_ids_total. " + _BIG_INTEGERS
    )


class FindDuplicateRowsResult(MaskableResult):
    """The result of find_duplicate_rows."""

    compared_columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns in which rows must be equal to count as duplicates, "
        "in the order compared. Missing values of every kind equal each other, as "
        "in pandas' duplicate check over two or more columns, and lists, dicts "
        "and sets compare by content. The record keeps the first 100 and adds "
        "compared_columns_total."
    )
    duplicate_groups: Annotated[list[DuplicateGroup], _ENTRIES] = Field(
        description="Each group of rows equal in every compared column, ordered "
        "by first occurrence; a group's number is its position here. The record "
        "keeps the first 100 and adds duplicate_groups_total; the complete table "
        "has every row of every group."
    )
    duplicate_group_count: int = Field(description="Groups of duplicate rows.")
    duplicate_group_row_count: int = Field(
        description="Rows in duplicate groups, every copy counted, the first too."
    )
    excess_copy_count: int = Field(
        description="Rows in duplicate groups beyond the first of each group: "
        "duplicate_group_row_count - duplicate_group_count, the rows that "
        "removing duplicates would drop."
    )
    duplicate_share: float | None = Field(
        description="excess_copy_count over all rows, from 0 up to but excluding 1; "
        "null when nothing was compared: no rows, or no column to compare."
    )

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        facts = []
        if self.compared_columns:
            compared = _text.plural(len(self.compared_columns), "column", "columns")
            none = (
                "no two rows are equal in them; "
                if self.status == "completed" and not self.duplicate_groups
                else ""
            )
            facts.extend(
                _text.wrap(
                    f"Compared {compared}: "
                    f"{_text.join_labels(self.compared_columns)}; {none}missing "
                    "values compare equal, as in pandas.",
                    first="",
                    indent="    ",
                )
            )
        id_column = self.parameters.get("id_column")
        source = (
            f"values of id_column {id_column!r}"
            if isinstance(id_column, str)
            else "index labels"
        )
        facts.append(f"Row IDs: {source}.")
        return facts

    def _text_table(self) -> _text.Table:
        rows = self.duplicate_group_row_count
        id_column = self.parameters.get("id_column")
        quote = not (isinstance(id_column, str) and self._masked(id_column))
        return _text.Table(
            title="Duplicate groups",
            unit=("group", "groups"),
            headers=("group", "size", "row_ids"),
            rows=[
                (number, group.size, _text.join_labels(group.row_ids, quote=quote))
                for number, group in enumerate(
                    self.duplicate_groups[: _text.MAX_TABLE_ROWS]
                )
            ],
            total=len(self.duplicate_groups),
            frame=f"all {_text.plural(rows, 'row', 'rows')} in groups"
            if rows
            else "no rows",
        )

    def to_frame(self, table: Literal["rows"] = "rows") -> pd.DataFrame:
        """Every row in a duplicate group, by group, then in row order.

        The column group holds the group's number, its position in
        duplicate_groups, and row_id the row's ID.

        Args:
            table: 'rows', the only table.
        """
        _check_table(table, ("rows",))
        numbers = [
            number
            for number, group in enumerate(self.duplicate_groups)
            for _ in group.row_ids
        ]
        row_ids = [
            row_id for group in self.duplicate_groups for row_id in group.row_ids
        ]
        return pd.DataFrame(
            {
                "group": pd.Series(numbers, dtype="int64"),
                "row_id": pd.Series(row_ids, dtype=object),
            }
        )
