"""The result of profile_target."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import (
    Field,
    PrivateAttr,
)

from agent_eda import _text
from agent_eda._levels import LEVEL_ORDER, Levels, level_rows
from agent_eda._numeric import (
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_QUANTILES,
    DEFAULT_TRIM_FRACTION,
)
from agent_eda._result_base import (
    _BIG_INTEGERS,
    _MORE_STATISTICS,
    MaskableResult,
    ResultModel,
    RowId,
    _Cap,
    _masking_text,
    _tables_unavailable,
    share_text,
)
from agent_eda._written_form import row_id_list
from agent_eda.statistics._descriptive_results import (
    SummarizeNumericColumn,
    numeric_frame,
)


class ProfileTargetClass(ResultModel):
    """A class of a classification target: a valid label and how often it occurs."""

    label: str | bool | int | float = Field(
        description="The label exactly as stored, so 'Yes' and 'yes ' are two "
        "classes. A datetime is ISO 8601 text and a Decimal a number or, when float64 "
        "can't hold it as written, its exact text; any other label JSON can't "
        "hold appears as its string form. label_type names its type. With "
        "masking on and a sensitive target, a numbered placeholder, '<masked 1>' "
        "for the target's first label in level order and so on. " + _BIG_INTEGERS
    )
    label_type: str = Field(
        description="The Python type of the stored label, such as 'str', 'int' or "
        "'bool'; a NumPy scalar counts as the Python type it holds."
    )
    count: int = Field(description="Rows with this label.")
    share: float = Field(
        description="count / valid_label_count: the class's share of the valid "
        "labels, from 0 to 1."
    )


class ProfileTargetUnexpectedLabel(ResultModel):
    """A label of a classification target that expected_labels doesn't hold."""

    label: str | bool | int | float = Field(
        description="The label exactly as stored, written as a class's label is. "
        "With masking on and a sensitive target, a numbered placeholder, numbered "
        "together with the classes in level order. " + _BIG_INTEGERS
    )
    label_type: str = Field(
        description="The Python type of the stored label, such as 'str' or 'int'."
    )
    count: int = Field(description="Rows with this label.")
    row_ids: Annotated[list[RowId], _Cap(10, total="count")] = Field(
        description="The IDs of the first 10 rows with this label, in row order: "
        "index labels, or values of id_column when it is given. count counts "
        "every row, and the record adds it as row_ids_total. With masking on and "
        "a sensitive id_column, numbered placeholders, '<masked 1>' for the first "
        "ID the labels show and so on, equal IDs alike. " + _BIG_INTEGERS
    )


# summarize_numeric's default quantile levels as the descriptions write them.
_DEFAULT_LEVELS = [str(level) for level in DEFAULT_QUANTILES]


class ProfileTargetStatistics(SummarizeNumericColumn):
    """summarize_numeric's entry for a regression target, with its default options."""

    quantiles: dict[float, float | None] = Field(
        description="Quantiles of the finite values at summarize_numeric's default "
        f"levels {_text.join_items(_DEFAULT_LEVELS, limit=len(_DEFAULT_LEVELS))}, "
        "keyed by level, as numpy.quantile computes them with its default method, "
        f"{DEFAULT_QUANTILE_METHOD}."
    )
    trimmed_mean: float | None = Field(
        description="Mean of the finite values left after cutting "
        f"{DEFAULT_TRIM_FRACTION:.0%} of them, "
        "rounded down, from each end, summarize_numeric's default trim_fraction, "
        "as SciPy's trimmed mean does."
    )


class ProfileTargetResult(MaskableResult):
    """The result of profile_target."""

    non_missing_count: int = Field(description="Target values that aren't missing.")
    missing_count: int = Field(
        description="Target values that are missing, the rows without a label. "
        "For classification: the native missing values, as pandas' value counts "
        "leave them out. For regression: as statistics.missing_count counts "
        "them, NaN that a pyarrow-backed float column stores as a value included."
    )
    missing_share: float | None = Field(
        description="missing_count over all rows, from 0 to 1; null when the data "
        "has no rows."
    )
    valid_label_count: int | None = Field(
        description="Classification: the labels the classes count, which every "
        "share divides by: each label that isn't missing, or with "
        "expected_labels, each one in it. Null for regression."
    )
    class_count: int | None = Field(
        description="Classification: the classes, the distinct valid labels, as "
        "pandas value_counts tells them apart, so 1, 1.0 and the boolean true are one class. "
        "With expected_labels, a declared label that no row holds is a class of "
        "count 0. Null for regression."
    )
    classes: Annotated[
        list[ProfileTargetClass] | None, _Cap(100, total="class_count")
    ] = Field(
        description="Classification: the first 100 classes, each with its count "
        "and share. Level order: " + LEVEL_ORDER + " A declared class without "
        "rows comes last, at count 0. "
        "class_count counts every class, the record adds it as classes_total, and "
        "the complete table has every label. Null "
        "for regression."
    )
    majority_share: float | None = Field(
        description="Classification: the share of the most common class; null for "
        "regression or without a valid label."
    )
    minority_share: float | None = Field(
        description="Classification: the share of the least common class, 0 when "
        "a declared class has no rows; null for regression or without a valid "
        "label."
    )
    imbalance_ratio: float | None = Field(
        description="Classification: the count of the most common class over the "
        "count of the least common one, 1 for balanced classes and higher the "
        "more imbalanced they are; null for regression, without a valid label, "
        "and when a declared class has no rows, which would divide by 0, with a "
        "`too_few_values` limitation."
    )
    unexpected_label_count: int | None = Field(
        description="Classification with expected_labels: the distinct labels "
        "outside expected_labels, which no class counts. Null without "
        "expected_labels, which leaves the check out, and for regression."
    )
    unexpected_labels: Annotated[
        list[ProfileTargetUnexpectedLabel] | None,
        _Cap(100, total="unexpected_label_count"),
    ] = Field(
        description="Classification with expected_labels: the first 100 labels "
        "outside expected_labels, in level order, each with its count and the IDs "
        "of its first rows. unexpected_label_count counts every one, and the "
        "record adds it as unexpected_labels_total. Null when the check didn't "
        "run."
    )
    rare_class_count: int | None = Field(
        description="Classification with rare_share: the classes whose share lies "
        "below rare_share. Null without rare_share, which leaves the check out, "
        "and for regression."
    )
    rare_classes: Annotated[
        list[ProfileTargetClass] | None, _Cap(100, total="rare_class_count")
    ] = Field(
        description="Classification with rare_share: the first 100 classes whose "
        "share lies below rare_share, in level order, as in classes. "
        "rare_class_count counts every one, and the record adds it as "
        "rare_classes_total. Null when the check didn't run."
    )
    statistics: ProfileTargetStatistics | None = Field(
        description="Regression: summarize_numeric's entry for the target, from the "
        "same implementation with summarize_numeric's defaults: counts, mean, std, "
        "min, max, quantiles, median, trimmed_mean, mad, mad_scaled, skewness and "
        "excess_kurtosis of the finite values. Null for classification."
    )
    non_finite_count: int | None = Field(
        description="Regression: target values that aren't finite numbers, the "
        "missing ones and inf or -inf, which a model can't learn from: "
        "statistics.missing_count + statistics.infinite_count. Null for "
        "classification."
    )
    # Every label of a classification target, outside the record, for
    # to_frame(); with masking on, the placeholders' numbers instead.
    _labels: Levels | None = PrivateAttr(default=None)
    # Per label of _labels, whether expected_labels holds it; None without
    # expected_labels.
    _expected: np.ndarray | None = PrivateAttr(default=None)
    # The ID of every row that holds a label outside expected_labels, by the
    # label's position in _labels: pandas labels, or lists when masked.
    _row_ids: dict[int, Any] | None = PrivateAttr(default=None)

    TABLES: ClassVar[Mapping[str, str | None]] = {
        "target": None,
    }

    def _target_text(self) -> str:
        return _text.label_text(self.settings.target_column)

    def _masking_facts(self, surface: _text.Surface) -> list[str]:
        # A regression target's statistics show as computed, sensitive or not.
        target = self.settings.target_column
        project = self.settings.project
        sensitive = project.data_policy.sensitive_columns if project else ()
        statistics = (
            [target] if self.statistics is not None and target in sensitive else []
        )
        return _masking_text(self.masking, surface, statistics)

    def _text_facts(self, surface: _text.Surface) -> list[str]:
        if self.statistics is not None:
            return self._regression_facts(self.statistics, surface)
        task = self.settings.task
        missing = f"{self.missing_count:,} missing" + (
            share_text("missing_share", self.missing_share)
            if self.missing_count
            else ""
        )
        classes = _text.plural(self.class_count or 0, "class", "classes")
        facts = [
            f"Target {self._target_text()} for {task}: {classes} in "
            f"{self.valid_label_count or 0:,} valid labels; {missing}."
        ]
        if self.majority_share is not None and self.minority_share is not None:
            ratio = (
                "null, a class has no rows"
                if self.imbalance_ratio is None
                else _text.format_number(self.imbalance_ratio)
            )
            facts.append(
                f"Majority share {_text.format_number(self.majority_share)}, minority "
                f"share {_text.format_number(self.minority_share)}; imbalance ratio "
                f"{ratio}, the majority count over the minority count."
            )
        if self.unexpected_labels:
            facts.extend(self._unexpected_text(self.unexpected_labels))
        rare_share = self.settings.rare_share
        if self.rare_class_count is not None and rare_share is not None:
            facts.append(
                "Rare classes, with a share below rare_share "
                f"{_text.format_number(rare_share)}: {self.rare_class_count:,} of "
                f"{self.class_count or 0:,}."
            )
        return facts

    def _label_text(self, label: object, label_type: str) -> str:
        """A label as the text shows it: text quoted, a placeholder as it is."""
        return self._shown(label, label_type, self.settings.target_column)

    def _unexpected_text(
        self, unexpected: Sequence["ProfileTargetUnexpectedLabel"]
    ) -> list[str]:
        """The first unexpected labels, each with its first row IDs."""
        show = _text.label_text if self._ids_quoted() else str
        items = []
        for entry in unexpected[:_SHOWN_UNEXPECTED]:
            ids = [show(row_id) for row_id in entry.row_ids[:_SHOWN_ROW_IDS]]
            if entry.count > len(ids):
                ids.append(f"{entry.count - len(ids):,} more")
            rows = "row" if entry.count == 1 else "rows"
            label = self._label_text(entry.label, entry.label_type)
            items.append(f"{label} in {rows} {_text.join_items(ids)}")
        labels = self.unexpected_label_count or len(unexpected)
        if labels > len(items):
            items.append(f"and {labels - len(items):,} more labels")
        rows = self.non_missing_count - (self.valid_label_count or 0)
        text = (
            f"Outside expected_labels: {_text.plural(labels, 'label', 'labels')} in "
            f"{_text.plural(rows, 'row', 'rows')}: {'; '.join(items)}."
        )
        return _text.wrap(text, first="", indent="    ")

    def _regression_facts(
        self, statistics: "ProfileTargetStatistics", surface: _text.Surface
    ) -> list[str]:
        finite = statistics.non_missing_count - statistics.infinite_count
        non_finite = self.non_finite_count or 0
        if non_finite:
            others = (
                f"{non_finite:,} aren't: {self.missing_count:,} missing"
                f"{share_text('missing_share', self.missing_share) if self.missing_count else ''} and "
                f"{statistics.infinite_count:,} infinite"
            )
        else:
            others = "none missing or infinite"
        return [
            f"Target {self._target_text()} for regression: "
            f"{_text.plural(finite, 'finite value', 'finite values')}; {others}.",
            "Statistics use the finite values, as in summarize_numeric: quantile "
            f"method {DEFAULT_QUANTILE_METHOD}, trim_fraction "
            f"{_text.format_number(DEFAULT_TRIM_FRACTION)}.",
            _text.also_in_table(_MORE_STATISTICS, surface),
        ]

    def _text_table(self) -> _text.Table:
        if self.statistics is not None:
            entry = self.statistics
            return _text.Table(
                title="Statistics",
                unit=("column", "columns"),
                headers=(
                    "column",
                    "non_missing_count",
                    "missing_count",
                    "mean",
                    "std",
                    "min",
                    "median",
                    "max",
                ),
                rows=[
                    (
                        entry.column,
                        entry.non_missing_count,
                        entry.missing_count,
                        entry.mean,
                        entry.std,
                        entry.min,
                        entry.median,
                        entry.max,
                    )
                ],
                frame="every statistic in one row",
            )
        # The share below which a class is rare, when the check ran.
        rare_share = self.settings.rare_share if self.rare_classes is not None else None
        classes = self.classes or []
        return _text.Table(
            title="Classes",
            unit=("class", "classes"),
            headers=(
                "label",
                "count",
                "share",
                *(() if rare_share is None else ("rare",)),
            ),
            rows=[
                (
                    self._label_text(c.label, c.label_type),
                    c.count,
                    c.share,
                    *(() if rare_share is None else (c.share < rare_share,)),
                )
                for c in classes[: _text.MAX_TABLE_ROWS]
            ],
            total=self.class_count or 0,
            frame_rows=(self.class_count or 0) + (self.unexpected_label_count or 0),
            frame_unit=("label", "labels"),
        )

    def to_frame(self, table: Literal["target"] = "target") -> pd.DataFrame:
        """The target's table as a DataFrame.

        Classification: every label, classes and labels outside expected_labels
        alike, in level order, with label, label_type, count, share (null for a
        label outside expected_labels), expected (null when expected_labels is
        None), rare (null when rare_share is None, and for labels outside
        expected_labels) and row_ids (the ID of every row that holds a label
        outside expected_labels; null for the others). label holds each value
        exactly as stored, or with masking on, a sensitive target's
        placeholder. Regression: one row of
        statistics, as summarize_numeric's to_frame() gives it, with
        non_finite_count.

        Args:
            table: 'target', the only table.
        """
        self._check_table(table)
        if self.statistics is not None:
            frame = numeric_frame([self.statistics], list(self.statistics.quantiles))
            frame["non_finite_count"] = pd.Series(
                [self.non_finite_count], dtype="int64"
            )
            return frame
        levels = self._labels
        if levels is None:
            if (self.class_count or 0) + (self.unexpected_label_count or 0) > 0:
                raise _tables_unavailable(self)
            order = np.zeros(0, dtype=np.intp)
            labels: list[object] = []
            types: list[str] = []
            counts = np.zeros(0, dtype=np.int64)
        else:
            order, labels, types = level_rows(levels)
            counts = levels.counts[order].astype(np.int64)
        found = self._row_ids or {}
        row_ids = [
            row_id_list(found[position]) if position in found else None
            for position in order.tolist()
        ]
        expected = None if self._expected is None else self._expected[order]
        valid = np.ones(len(order), dtype=bool) if expected is None else expected
        shares = counts / max(self.valid_label_count or 0, 1)
        rare_share = self.settings.rare_share
        return pd.DataFrame(
            {
                "label": pd.Series(labels, dtype=object),
                "label_type": pd.Series(types, dtype=object),
                "count": counts,
                "share": np.where(valid, shares, np.nan),
                "expected": pd.array(
                    [pd.NA] * len(order) if expected is None else expected,
                    dtype="boolean",
                ),
                "rare": pd.array(
                    [
                        bool(share < rare_share) if is_valid else pd.NA
                        for share, is_valid in zip(
                            shares.tolist(), valid.tolist(), strict=True
                        )
                    ]
                    if isinstance(rare_share, float)
                    else [pd.NA] * len(order),
                    dtype="boolean",
                ),
                "row_ids": pd.Series(row_ids, dtype=object),
            }
        )


# The labels outside expected_labels, and the row IDs of each, that the text shows.
_SHOWN_UNEXPECTED = 3
_SHOWN_ROW_IDS = 3
