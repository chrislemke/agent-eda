"""The results of describe_dataset and find_candidate_keys."""

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal

import pandas as pd
from pydantic import (
    AfterValidator,
    Field,
)

from agent_eda import _text
from agent_eda._result_base import (
    _ENTRIES,
    AnalysisResult,
    ResultModel,
    _entries_frame,
    _list_by_name,
)


class DescribeDatasetColumn(ResultModel):
    """What describe_dataset found in one column."""

    position: int = Field(description="0-based position of the column in the data.")
    column: str = Field(description="The column label as a string.")
    label_type: str = Field(
        description="Python type of the column label, such as 'str', 'int' or "
        "'tuple'. Select a column whose label isn't a str with its original label."
    )
    dtype: str = Field(description="The dtype as pandas reports it, such as 'int64'.")
    non_missing_count: int = Field(description="Values that are not missing.")
    missing_count: int = Field(
        description="Native missing values: NaN, NaT and the missing markers of "
        "object, nullable and pyarrow-backed columns, including NaN that a "
        "pyarrow-backed or nullable float column stores as a value, which pandas' "
        "missing-value check doesn't count."
    )
    distinct_count: int | None = Field(
        description="Distinct non-missing values; null when the values can't be "
        "hashed, with an unhashable_values limitation."
    )
    memory_bytes: int = Field(
        description="Bytes the values use, counting each object a text or object "
        "column holds. Excludes pandas' lookup tables and the UTF-8 copy Python "
        "caches inside a string once something has encoded it, so earlier work on "
        "the data doesn't change it."
    )


class DescribeDatasetIndex(ResultModel):
    """What describe_dataset found in the row index."""

    dtype: str = Field(description="The index dtype as pandas reports it.")
    level_count: int = Field(description="Index levels: 1, or more for a MultiIndex.")
    is_unique: bool = Field(description="Whether every index label occurs once.")
    duplicated_count: int = Field(description="Labels that repeat an earlier label.")
    missing_count: int = Field(
        description="Missing labels; in a MultiIndex, labels with any level missing."
    )
    memory_bytes: int = Field(
        description="Bytes the index labels use. Excludes pandas' lookup tables and "
        "the UTF-8 copies Python caches inside strings."
    )


class Provenance(ResultModel):
    """Where the data came from, exactly as the caller supplied it."""

    source: str | None = Field(
        description="The source as supplied, or null when not supplied. Never inferred."
    )
    source_version: str | None = Field(
        description="The source version as supplied, or null when not supplied."
    )
    not_supplied: list[str] = Field(
        description="Provenance fields the caller did not supply."
    )


class DescribeDatasetResult(AnalysisResult):
    """The result of describe_dataset."""

    __help_note__ = (
        "res.columns is a list, since column labels can repeat; res.columns['age'] "
        "finds an entry by name, and res.columns[0] by position."
    )

    row_count: int = Field(description="Rows in the data.")
    column_count: int = Field(description="Columns in the data.")
    column_level_count: int = Field(
        description="Levels in the column labels: 1, or more for MultiIndex columns."
    )
    memory_bytes: int = Field(
        description="Bytes the columns and the index use together."
    )
    columns: Annotated[
        list[DescribeDatasetColumn], _ENTRIES, AfterValidator(_list_by_name)
    ] = Field(
        description="One entry per column, in DataFrame order: a list, not a mapping "
        "by name, because column labels can repeat. {cap}; the complete table has "
        "every column."
    )
    index: DescribeDatasetIndex = Field(description="Facts about the row index.")
    provenance: Provenance = Field(description="The provenance as supplied.")

    TABLES: ClassVar[Mapping[str, str | None]] = {
        "columns": None,
    }

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        index = self.index
        levels = f" with {index.level_count} levels" if index.level_count > 1 else ""
        repeats = _text.plural(
            index.duplicated_count, "repeated label", "repeated labels"
        )
        unique = "unique" if index.is_unique else f"not unique ({repeats})"
        missing = _text.plural(index.missing_count, "missing label", "missing labels")
        provenance = "; ".join(
            f"{name} {value!r}" if value is not None else f"{name} not supplied"
            for name, value in (
                ("source", self.provenance.source),
                ("source_version", self.provenance.source_version),
            )
        )
        return [
            f"Memory: {_text.format_bytes(self.memory_bytes)} in total. "
            f"Index: {index.dtype}{levels}, {unique}, {missing}.",
            f"Provenance: {provenance}.",
        ]

    def to_frame(self, table: Literal["columns"] = "columns") -> pd.DataFrame:
        """Every column's entry as one row, in DataFrame order.

        Args:
            table: 'columns', the only table.
        """
        self._check_table(table)
        return _entries_frame(self.columns, DescribeDatasetColumn)

    def _text_table(self) -> _text.Table:
        return _text.Table(
            title="Columns",
            unit=("column", "columns"),
            headers=(
                "column",
                "dtype",
                "non_missing_count",
                "missing_count",
                "distinct_count",
                "memory_bytes",
            ),
            rows=[
                (
                    c.column,
                    c.dtype,
                    c.non_missing_count,
                    c.missing_count,
                    c.distinct_count,
                    c.memory_bytes,
                )
                for c in self._by_interest(
                    self.columns, [c.column for c in self.columns]
                )
            ],
        )


class CandidateKey(ResultModel):
    """A column set the key search reports, with how well it identifies rows."""

    columns: Annotated[list[str], _ENTRIES] = Field(
        description="The columns of the set, in the order of the searched columns."
    )
    distinct_row_count: int = Field(
        description="Distinct combinations of values in these columns, with missing "
        "values of every native kind compared equal: the rows of the "
        "set that no earlier row repeats."
    )
    duplicate_group_row_count: int = Field(
        description="Rows whose combination another row shares, counting every row "
        "of a repeated combination, the first one too; 0 for a key."
    )
    missing_row_count: int = Field(
        description="Rows with a missing value in at least one of these columns."
    )
    uniqueness_share: float = Field(
        description="distinct_row_count divided by the rows of the data; 1 for a key."
    )
    is_minimal: bool = Field(
        description="Whether no smaller set of these columns identifies rows as "
        "well, so every column is needed."
    )


class KeySearch(ResultModel):
    """How far the key search went."""

    column_count: int = Field(
        description="Columns searched: the selected ones whose values can be hashed."
    )
    combination_count: int = Field(
        description="Column sets of 1 to max_key_size searched columns: the most "
        "the complete search examines, and the max_combinations that always "
        "suffices."
    )
    examined_count: int = Field(
        description="Column sets the search examined. Supersets of keys aren't "
        "examined."
    )
    rejected_on_subset_count: int = Field(
        description="Examined sets that repeated a combination within an evenly "
        "spaced subset of rows, which proves they aren't keys."
    )
    checked_on_all_rows_count: int = Field(
        description="Examined sets checked on all rows because no subset showed "
        "a repeat. Only a check on all rows confirms a key."
    )
    subset_row_count: int = Field(
        description="Rows in the evenly spaced subset each set is checked on "
        "first: row i * rows // 16,384 for each i below 16,384, or every row of "
        "smaller data. In data of more than 65,536 rows, a set without a repeat "
        "there is checked on every 4th row next."
    )


class FindCandidateKeysResult(AnalysisResult):
    """The result of find_candidate_keys."""

    keys: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="Minimal keys without missing values, each fit to serve as a "
        "primary key: column sets whose combination of values no two rows share. "
        "Fewest columns first, then in column order. {cap}; the complete table has every one."
    )
    keys_with_missing_values: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="Minimal keys with a missing value in some rows, which can't "
        "serve as primary keys, in the same order as keys."
    )
    most_unique_column_sets: Annotated[list[CandidateKey], _ENTRIES] = Field(
        description="When the search found no key: the 10 minimal column sets "
        "with the most distinct combinations, most first, then fewest columns, "
        "then in column order. Empty when it found a key."
    )
    exhaustive: bool = Field(
        description="Whether the search examined every set of up to max_key_size "
        "columns, apart from supersets of keys. It is false when max_combinations "
        "stopped the search, or when no search ran."
    )
    search: KeySearch = Field(description="How far the search went.")
    left_out_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Selected columns the search left out because their values "
        "can't be hashed, such as lists or dicts, in the selected order; empty "
        "when it searched every selected column."
    )
    non_integral_float_columns: Annotated[list[str], _ENTRIES] = Field(
        description="Columns of floats with at least one fractional value, such "
        "as 0.5, which the default column selection leaves out because they are "
        "measurements, not IDs, in DataFrame order. Empty when the call named the "
        "columns searched, "
        "or when none holds a fraction."
    )

    TABLES: ClassVar[Mapping[str, str | None]] = {
        "candidates": None,
    }

    def _candidates(self) -> list[tuple[str, CandidateKey]]:
        """Every reported column set with its kind, in the order of the lists."""
        return [
            *(("key", key) for key in self.keys),
            *(
                ("key_with_missing_values", key)
                for key in self.keys_with_missing_values
            ),
            *(("not_a_key", key) for key in self.most_unique_column_sets),
        ]

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        search = self.search
        if not search.examined_count:
            return []
        largest = min(self.settings.max_key_size, search.column_count)
        keys = f"keys of up to {_text.plural(largest, 'column', 'columns')}"
        if self.exhaustive:
            columns = _text.plural(search.column_count, "column", "columns")
            examined = _text.plural(search.examined_count, "set", "sets")
            skipped = search.combination_count - search.examined_count
            searched = (
                f"Exhaustive search for {keys} among {columns}: examined {examined}"
            )
            searched += f", skipped {skipped:,} supersets of keys." if skipped else "."
        else:
            searched = (
                f"Not exhaustive: max_combinations stopped the search for {keys} after "
                f"{search.examined_count:,} of up to {search.combination_count:,} sets."
            )
        facts = [
            searched,
            "Missing values compare equal; a key with missing values can't serve as a "
            "primary key.",
        ]
        if self.left_out_columns:
            names = _text.join_labels(self.left_out_columns)
            facts.append(f"Left out, values can't be hashed: {names}.")
        if self.non_integral_float_columns:
            names = _text.join_labels(self.non_integral_float_columns)
            facts.append(f"Left out, floats with fractions: {names}.")
        return facts

    def _text_table(self) -> _text.Table | None:
        if not self.search.examined_count:
            return None
        found = self.keys or self.keys_with_missing_values
        return _text.Table(
            title="Keys" if found else "Most unique column sets",
            unit=("column set", "column sets"),
            headers=(
                "columns",
                "kind",
                "distinct_row_count",
                "duplicate_group_row_count",
                "missing_row_count",
                "uniqueness_share",
            ),
            rows=[
                (
                    ", ".join(_text.label_text(name) for name in candidate.columns),
                    kind,
                    candidate.distinct_row_count,
                    candidate.duplicate_group_row_count,
                    candidate.missing_row_count,
                    candidate.uniqueness_share,
                )
                for kind, candidate in self._candidates()
            ],
        )

    def to_frame(self, table: Literal["candidates"] = "candidates") -> pd.DataFrame:
        """Every reported column set, one row each: keys, keys with missing values,
        then the most unique sets, with `kind` naming the list each comes from.

        Args:
            table: 'candidates', the only table.
        """
        self._check_table(table)
        candidates = self._candidates()
        frame = _entries_frame([candidate for _, candidate in candidates], CandidateKey)
        frame.insert(
            1, "kind", pd.Series([kind for kind, _ in candidates], dtype=object)
        )
        return frame
