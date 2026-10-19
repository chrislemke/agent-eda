"""Analyses of the target: the column a model learns to predict."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, cast

import numpy as np
import pandas as pd
from annotated_types import Gt, Lt

from agent_eda._columns import (
    ALL_KINDS,
    Column,
    ColumnKind,
    ColumnRule,
    select_columns,
)
from agent_eda._levels import (
    LEVEL_ORDER,
    Levels,
    count_levels_by_row,
    ordered_among,
    record_label,
    stored_label,
)
from agent_eda._levels import type_name as label_type_name
from agent_eda._masking import (
    MaskedLevels,
    MaskingDecision,
    mask_levels,
    mask_values,
    resolve_masking,
)
from agent_eda._numeric import (
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_QUANTILES,
    DEFAULT_TRIM_FRACTION,
    NumericSummary,
    null_limitations,
    numeric_entry,
    numeric_support,
    read_limitations,
    read_numeric,
    stored_nan_count,
    stored_nan_limitations,
    summarize_values,
)
from agent_eda._registry import analysis
from agent_eda._row_ids import RowIds, ambiguous_row_ids, read_row_ids
from agent_eda._run import NO_ROWS, check_distinct, data_shape, execution
from agent_eda._sentinels import Family, SentinelValue, holds, likely_value, literal
from agent_eda._text import Surface, format_number, join_items, placeholder, plural
from agent_eda._wording import as_written, worded
from agent_eda.errors import ColumnTypeError, InvalidArgumentError
from agent_eda.project import Project
from agent_eda.results import (
    Assessment,
    ColumnSupport,
    Finding,
    Limitation,
    ProfileTargetClass,
    ProfileTargetResult,
    ProfileTargetStatistics,
    ProfileTargetUnexpectedLabel,
    Status,
    StatusReason,
    Support,
)

# The tasks a target can be profiled for.
Task = Literal["binary_classification", "multiclass_classification", "regression"]
# A class label as expected_labels declares it.
Label = str | bool | int | float

# target_column may name a column of any dtype; the task decides which it takes.
_ANY_COLUMN = ColumnRule(
    default=frozenset(),
    default_text="no column",
    accepted=ALL_KINDS,
    accepted_text="a column of any dtype",
    fixes={},
)
# The classes and unexpected labels each result keeps; to_frame() has every label.
_KEPT_LABELS = 100
# The row IDs each unexpected label keeps.
_KEPT_ROW_IDS = 10


@analysis(
    statuses=("completed", "insufficient_data", "failed"),
    finding_codes=(
        "single_class",
        "more_than_two_classes",
        "unexpected_labels",
        "no_unexpected_labels",
        "all_labels_distinct",
        "constant_target",
        "infinite_values",
        "missing_labels",
        "rare_classes",
    ),
    tags=(
        "target",
        "label",
        "class balance",
        "class imbalance",
        "classification",
        "regression target",
        "rare classes",
        "majority class",
    ),
    masking=True,
)
def profile_target(
    data: pd.DataFrame,
    *,
    target_column: str,
    task: Task,
    expected_labels: Sequence[Label] | None = None,
    rare_share: Annotated[float, Gt(0), Lt(1)] | None = None,
    id_column: str | None = None,
    project: Project | None = None,
    mask_sensitive: bool | None = None,
) -> ProfileTargetResult:
    """What does the target look like for this task?

    When to use:
        Call it once you know the column a model should learn to predict and
        the task: binary or multiclass classification, or regression. The task
        is never inferred from the dtype, so numbers can be class labels and a
        numeric label isn't mistaken for a regression target. For
        classification it counts each class and its share of the valid labels,
        gives the majority and minority shares and the imbalance ratio, and
        counts missing labels; with expected_labels it lists the labels outside
        that set with the IDs of their first rows, and with rare_share the
        classes whose share lies below it. For regression it gives the
        statistics `summarize_numeric` gives, from the same implementation,
        and counts the values that aren't finite. It flags a target that
        doesn't fit the task, such as a single class or a constant regression
        target, before you train anything. Class labels and row IDs are raw
        values, so when the target holds personal data, such as a diagnosis,
        declare it sensitive in a project and switch masking on; index labels
        are never masked, so pass a non-sensitive id_column when the index
        holds sensitive values. For the values of other categorical columns,
        call `summarize_categorical`; for other numeric columns,
        `summarize_numeric`; for how the target goes with other columns,
        `compute_association_matrix`. `list_analyses` lists the other
        analyses.

    Args:
        data (pd.DataFrame): The data. Column labels must be unique, distinct as
            strings and a single level; describe_dataset reports labels that
            aren't. It is never modified.
        target_column (str): Name of the column a model learns to predict.
            Required. A label that isn't a string can be given as itself, such
            as 0, or as its string form.
        task (str): What the model does with the target, never inferred from
            the dtype. Required: 'binary_classification' for two classes,
            'multiclass_classification' for two or more, or 'regression' for a
            number. Class labels can be text, numbers, booleans or categories;
            a regression target must hold integers or floats, and another dtype
            raises errors.ColumnTypeError that suggests a classification task or
            a conversion.
        expected_labels (Sequence[str | bool | int | float] | None): The labels
            the task allows, such as ['no', 'yes'] or [0, 1]: exactly 2 for
            binary_classification and at least 2 for
            multiclass_classification, each listed once; regression takes
            none. A value matches a label equal to it, as Python's == decides,
            so 1 matches 1.0 and True, and text matches with its case and
            surrounding spaces. A label of a type the target never holds, such
            as '1' for an integer column, raises errors.InvalidArgumentError
            naming the label you likely meant. None, the default, leaves the
            check out: unexpected_labels is null, and a limitation says so.
        rare_share (float | None): For classification, the share of the valid
            labels below which a class counts as rare, such as 0.05; greater
            than 0 and less than 1. None, the default, leaves the check out:
            rare_classes is null, and a limitation says so.
        id_column (str | None): Name of a column whose values identify rows,
            such as a customer number; the row IDs of labels outside
            expected_labels are then its values. Its values must be hashable;
            a column of lists raises. None, the default, uses the index labels.
        project (Project | None): The project, whose data policy names the
            sensitive columns and says whether to mask them. None, the
            default, declares no sensitive column, so nothing is masked.
        mask_sensitive (bool | None): True replaces the labels of a sensitive
            target, the classes, the labels outside expected_labels and
            expected_labels as parameters records it, with numbered
            placeholders, '<masked 1>' for its first label in level order and
            so on, and the row IDs taken from a sensitive id_column too; it
            needs a project that declares sensitive columns. False shows them
            as stored. None, the default, follows the project's
            data_policy.mask_sensitive, and masks nothing without a project.
            Index labels are never masked, and neither are the statistics of a
            regression target, such as its minimum. Masking is a guardrail
            against accidental exposure, not anonymization or access control.

    Returns:
        ProfileTargetResult: For classification, `classes` lists the first 100
        classes, the distinct valid labels, by count, descending, then by
        label, as `level_order` states, each with its `label`, `label_type`,
        `count` and `share`. `class_count` counts every class, and
        `valid_label_count` counts the labels the shares divide by: those that
        aren't missing, or with expected_labels, those in it. `majority_share`
        and `minority_share` are the shares of the most and least common
        classes, and `imbalance_ratio` is the majority count over the minority
        count: 1 for balanced classes, higher the more imbalanced. With
        expected_labels, `unexpected_labels` lists the first 100 labels outside
        it, each with its `count` and the `row_ids` of its first 10 rows, and
        `unexpected_label_count` counts them. With rare_share, `rare_classes`
        lists the classes whose share lies below it, and `rare_class_count`
        counts them. A check that didn't run leaves its fields null, with an
        `expected_labels_not_declared` or `rare_share_not_declared`
        limitation. `to_frame` returns every label with its count, its
        share, whether expected_labels holds it and whether it is rare.

        For regression, `statistics` holds summarize_numeric's fields for the
        target, from the same implementation with summarize_numeric's default
        quantiles, quantile method and trimmed mean, and `non_finite_count`
        counts the values that are missing, inf or -inf. `to_frame` returns
        the statistics in one row.

        For both tasks, `missing_count` and `missing_share` count the rows
        without a label, and a share without rows is null, with a
        `too_few_values` limitation. Labels are kept exactly as stored. With
        index labels as row IDs, df.loc[row_ids] shows the rows, and an
        `ambiguous_row_ids` limitation flags row IDs that repeat. NaN is a
        missing label, also NaN that a pyarrow-backed or nullable float column
        stores as a value, which pandas' `isna` doesn't count; the
        `nan_counted_as_missing` limitation names such a target. `masking`
        says whether masking was on and names the sensitive columns whose
        values the result shows; with masking on, their labels and row IDs
        are placeholders in the printed text, `to_dict`, `to_json` and
        `to_frame` alike, while counts and shares stay.

        The findings:

        - `single_class`: violation; every valid label is the same class, so
          a classifier has nothing to tell apart.
        - `more_than_two_classes`: violation; a binary_classification target
          has more than two classes. One class gives single_class instead.
        - `unexpected_labels`: violation; labels lie outside expected_labels.
        - `no_unexpected_labels`: passed; every label that isn't missing is in
          expected_labels.
        - `all_labels_distinct`: suspected; the values of a classification
          target that aren't missing are all distinct, as an identifier's or a
          measurement's are, not class labels'.
        - `constant_target`: violation; at least two values of a regression
          target aren't missing and all of them are equal, decided on the stored
          values as every analysis does: distinct integers beyond 2^53 aren't
          equal, and NaN is missing.
        - `infinite_values`: suspected; a regression target holds inf or -inf,
          which every statistic leaves out.
        - `missing_labels`: descriptive; rows have no target value.
        - `rare_classes`: descriptive; classes have a share below rare_share.

    Statuses:
        completed: The target has a valid label, or for regression a finite
            value, whether or not it fits the task.
        insufficient_data: The data has no rows, every target value is
            missing, no label is in expected_labels, or a regression target has
            no finite value.
        failed: An anticipated numerical failure stopped the analysis, and
            `status_reason` gives its code and message. None is known today: a
            statistic that can't be computed is null, with a limitation,
            instead.

    Examples:
        >>> import pandas as pd
        >>> import agent_eda as ae
        >>> df = pd.DataFrame(
        ...     {
        ...         "customer": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        ...         "churned": ["no", "no", "yes", "no", "No", "no", None, "yes"],
        ...         "spend": [12.5, 40.0, 7.25, 99.0, 35.0, 0.0, 18.0, 52.5],
        ...     }
        ... )
        >>> res = ae.profile_target(
        ...     df,
        ...     target_column="churned",
        ...     task="binary_classification",
        ...     expected_labels=["no", "yes"],
        ...     rare_share=0.05,
        ...     id_column="customer",
        ... )
        >>> print(res)
        profile_target: completed on 8 rows x 3 columns
        Target 'churned' for binary_classification: 2 classes in 6 valid labels; 1 missing (12.5%).
        Majority share 0.6667, minority share 0.3333; imbalance ratio 2, the majority count over the minority count.
        Outside expected_labels: 1 label in 1 row: 'No' in row 'c5'.
        Rare classes, with a share below rare_share 0.05: 0 of 2.
        Classes (2 of 2 shown):
        label  count   share  rare
        'no'       4  0.6667    no
        'yes'      2  0.3333    no
        Findings (2 of 2 shown):
        - violation unexpected_labels: 1 label in 1 row is outside expected_labels.
        - descriptive missing_labels: 1 of 8 rows (12.5%) has no label: its target value is missing.
        Limitations: none
        More: res.to_frame() has all 3 labels; res.findings, res.limitations and res.to_json() have the rest.
        >>> res.imbalance_ratio
        2.0
        >>> res.unexpected_labels[0].row_ids
        ['c5']
        >>> spend = ae.profile_target(df, target_column="spend", task="regression")
        >>> spend.statistics.median
        26.5

    See Also:
        summarize_categorical: Each level's count and share of every text,
            categorical and boolean column, for columns besides the target.
        summarize_numeric: Counts, mean, spread, quantiles and shape of every
            numeric column, the statistics a regression target gets.
        infer_column_semantics: Which columns hold identifiers, codes or
            numbers stored as text rather than measurements or labels.
        compute_association_matrix: How strongly pairs of columns are
            associated, the target with the others included.
        list_analyses: Every analysis, with the question each one answers.
    """
    masking = resolve_masking("profile_target", project, mask_sensitive)
    labels = None if expected_labels is None else list(expected_labels)
    _check_options(task, labels, rare_share, hide=masking.masks(target_column))
    [target] = select_columns(
        "profile_target", data, [target_column], _ANY_COLUMN, parameter="target_column"
    ).columns
    _check_target(target, task)
    row_ids = read_row_ids("profile_target", data, id_column)
    if task == "regression":
        profile = _regression(data, target)
    else:
        profile = _classification(
            data, target, task, labels, rare_share, row_ids, masking
        )
    # The other task's fields are null.
    fields: dict[str, Any] = {**dict.fromkeys(_TASK_FIELDS), **profile.fields}
    result = ProfileTargetResult(
        analysis="profile_target",
        method=None,
        status=profile.status,
        status_reason=profile.status_reason,
        parameters={
            "data": data_shape(data),
            "target_column": target.name,
            "task": task,
            "expected_labels": profile.expected_labels,
            "rare_share": rare_share,
            "id_column": row_ids.id_column,
            "project": project.model_dump(mode="json") if project else None,
            "mask_sensitive": mask_sensitive,
        },
        support=Support(row_count=len(data), column_count=1, columns=[profile.support]),
        findings=profile.findings,
        limitations=profile.limitations,
        execution=execution(("numpy", "pandas")),
        masking=masking.record(profile.shown, data),
        level_order=LEVEL_ORDER,
        **fields,
    )
    result._labels = profile.labels
    result._expected = profile.expected
    return result


# The result fields that only one of the tasks fills.
_TASK_FIELDS = (
    "valid_label_count",
    "class_count",
    "classes",
    "majority_share",
    "minority_share",
    "imbalance_ratio",
    "unexpected_label_count",
    "unexpected_labels",
    "rare_class_count",
    "rare_classes",
    "statistics",
    "non_finite_count",
)


def _check_options(
    task: Task,
    labels: Sequence[Label] | None,
    rare_share: float | None,
    *,
    hide: bool,
) -> None:
    """Raise when expected_labels or rare_share can't serve the task.

    With `hide`, the target is a masked sensitive column and no message quotes
    its labels.
    """
    if task == "regression":
        given = [
            name
            for name, value in (("expected_labels", labels), ("rare_share", rare_share))
            if value is not None
        ]
        if given:
            them = "it" if len(given) == 1 else "them"
            raise worded(
                InvalidArgumentError,
                lambda surface: (
                    f"profile_target got {join_items(given)} for a "
                    f"regression task, which has no class labels to check. Leave {them} "
                    f"out, or pass {_classification_tasks(surface)} if the target holds "
                    "class labels."
                ),
            )
        return
    if labels is None:
        return
    if not labels:
        raise worded(
            InvalidArgumentError,
            lambda surface: (
                "profile_target got an empty list for expected_labels. "
                "List the labels the task allows, or "
                + (
                    "pass expected_labels=None to leave the check out."
                    if surface == "python"
                    else "leave expected_labels out to skip the check."
                )
            ),
        )
    not_finite = [
        repr(label)
        for label in labels
        if isinstance(label, float) and not math.isfinite(label)
    ]
    if not_finite:
        raise InvalidArgumentError(
            f"profile_target got {join_items(not_finite)} in expected_labels, but "
            "NaN and infinity can't be class labels; missing labels are counted "
            "apart, in missing_count. Leave them out of expected_labels."
        )
    check_distinct("profile_target", "expected_labels", labels, "label", hide=hide)
    given = plural(len(labels), "label", "labels")
    if task == "binary_classification" and len(labels) != 2:
        raise InvalidArgumentError(
            f"profile_target got {given} in expected_labels for a "
            "binary_classification task, which has 2 classes. List its 2 labels, or "
            "pass task='multiclass_classification'."
        )
    if len(labels) < 2:
        raise InvalidArgumentError(
            f"profile_target got {given} in expected_labels for a "
            "multiclass_classification task, which has at least 2 classes. List "
            "every label the task allows."
        )


def _check_label_types(
    target: Column, series: pd.Series, labels: Sequence[Label], *, hide: bool
) -> None:
    """Raise when an expected label can never equal a value of the target.

    With `hide`, the target is a masked sensitive column: the message counts the
    labels without quoting them or the labels they likely meant.

    Labels compare as Python's == does, so numbers and booleans can equal each
    other, while text only equals text. An object column without values can
    hold anything.
    """
    column = holds(series)
    if column.families is None:
        return
    unusable = [
        label
        for label in labels
        if not (({"text"} if isinstance(label, str) else _NUMBERS) & column.families)
    ]
    if not unusable:
        return
    likely = [likely_value(cast("SentinelValue", label), column) for label in unusable]
    hint = (
        f" Did you mean {join_items([literal(value) for value in likely])}?"
        if all(value is not None for value in likely) and not hide
        else ""
    )
    holding = f"{target.name!r} holds {column.what} (dtype {column.dtype})"
    if len(unusable) == 1:
        shown = "a label" if hide else literal(unusable[0])
        problem = (
            f"got {shown} in expected_labels, but {holding}, so it can never equal "
            "a label."
        )
    else:
        listed = "" if hide else f": {join_items([literal(x) for x in unusable])}"
        problem = (
            f"can't use {len(unusable):,} labels in expected_labels{listed}, since "
            f"{holding}, so they can never equal a label."
        )
    raise InvalidArgumentError(
        f"profile_target {problem}{hint} Declare each label as the type the target "
        "holds."
    )


# The value families a number or boolean label can equal, as Python's == decides.
_NUMBERS: frozenset[Family] = frozenset({"number", "boolean"})


# The kinds of column whose values can be class labels.
_LABEL_KINDS: frozenset[ColumnKind] = frozenset(
    {"numeric", "boolean", "text", "object", "categorical"}
)
_CLASSIFY = (
    "If its values are class labels, pass task='binary_classification' or "
    "task='multiclass_classification'"
)
_TEXT_FIX = (
    f"{_CLASSIFY}; if they are numbers stored as text, convert them with "
    "pd.to_numeric({column}, errors='coerce')"
)
# How to fix a regression task on a column of each kind that isn't numeric;
# {column} becomes the code for the column.
_REGRESSION_FIXES: dict[ColumnKind, str] = {
    "text": _TEXT_FIX,
    "object": _TEXT_FIX,
    "categorical": (
        f"{_CLASSIFY}; if its categories are numbers, convert them with "
        "{column}.astype(float)"
    ),
    "boolean": (
        "If its values are class labels, pass task='binary_classification'; to read "
        "False and True as 0 and 1, convert them with {column}.astype(float)"
    ),
    "timedelta": (
        "Convert durations to numbers first, such as {column}.dt.total_seconds()"
    ),
    "datetime": (
        "Convert datetimes to numbers first, such as the seconds since the "
        "earliest: ({column} - {column}.min()).dt.total_seconds()"
    ),
    "other": "Pass a column of numbers instead",
}


def _classification_tasks(surface: Surface) -> str:
    """The two classification tasks as the caller passes them."""
    return " or ".join(
        as_written(surface, task=task)
        for task in ("binary_classification", "multiclass_classification")
    )


# How an MCP caller fixes a regression task on a column of each kind that
# isn't numeric: converting the column is Python, so it picks another task or
# another column.
_MCP_CLASSIFY = f"If its values are class labels, pass {_classification_tasks('mcp')}"
_MCP_REGRESSION_FIXES: dict[ColumnKind, str] = {
    "text": _MCP_CLASSIFY,
    "object": _MCP_CLASSIFY,
    "categorical": _MCP_CLASSIFY,
    "boolean": (
        "If its values are class labels, pass "
        f"{as_written('mcp', task='binary_classification')}"
    ),
    "timedelta": "Pass a column of numbers instead",
    "datetime": "Pass a column of numbers instead",
    "other": "Pass a column of numbers instead",
}


def _check_target(target: Column, task: Task) -> None:
    """Raise unless the task can take the target column's dtype.

    A regression target holds numbers; class labels can be text, numbers,
    booleans or categories. The task is never inferred from the dtype, so
    numbers can be class labels.
    """
    code = f"df[{target.label!r}]"
    what = f"column {target.name!r} with dtype {target.dtype}"
    if task == "regression" and target.kind != "numeric":
        fixes = {
            "python": _REGRESSION_FIXES[target.kind].format(column=code),
            "mcp": _MCP_REGRESSION_FIXES[target.kind],
        }
        raise worded(
            ColumnTypeError,
            lambda surface: (
                f"profile_target can't use {what} as a regression "
                "target, which must hold numbers: integer or float columns, nullable "
                f"and pyarrow-backed ones included. {fixes[surface]}."
            ),
        )
    if task != "regression" and target.kind not in _LABEL_KINDS:
        fixes = {
            "python": f"Convert its values to labels first, such as {code}.astype(str)",
            "mcp": "Pass a column of labels instead",
        }
        raise worded(
            ColumnTypeError,
            lambda surface: (
                f"profile_target can't use {what} as a classification "
                "target, whose labels must be text, numbers, booleans or categories, "
                f"nullable and pyarrow-backed ones included. {fixes[surface]}."
            ),
        )


@dataclass
class _Profile:
    """What one task's profile puts into the result."""

    status: Status
    status_reason: StatusReason | None
    support: ColumnSupport
    # The result's own fields; those of the other task stay null.
    fields: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    limitations: list[Limitation] = field(default_factory=list)
    # The columns whose values the result shows, None for index labels.
    shown: list[str | None] = field(default_factory=list)
    # A classification target's labels for to_frame(), and per label whether
    # expected_labels holds it; None without expected_labels.
    labels: Levels | None = None
    expected: np.ndarray | None = None
    # expected_labels as the parameters record them, placeholders when masked.
    expected_labels: list[Label] | list[str] | None = None


@dataclass(frozen=True)
class _Labels:
    """A classification target's labels as the result shows them.

    With masking on, numbered placeholders, in level order, stand for them.
    """

    levels: Levels
    masked: MaskedLevels | None

    def entry(self, position: int) -> tuple[str | bool | int | float, str]:
        """The label at a level position as the result shows it, and its type."""
        label = self.levels.label(position)
        shown = (
            record_label(label) if self.masked is None else self.masked.label(position)
        )
        return shown, label_type_name(label)


def _classification(
    data: pd.DataFrame,
    target: Column,
    task: Task,
    labels: Sequence[Label] | None,
    rare_share: float | None,
    row_ids: RowIds,
    masking: MaskingDecision,
) -> _Profile:
    series = data.iloc[:, target.position]
    row_count = len(data)
    levels, row_levels = _count_labels(target, series)
    if labels is not None:
        _check_label_types(target, series, labels, hide=masking.masks(target.name))
    valid = _valid_levels(levels, labels)
    view = _Labels(levels, mask_levels(levels) if masking.masks(target.name) else None)
    counts = levels.counts[valid]
    valid_count = int(counts.sum())
    classes = [
        _class(view, position, valid_count)
        for position in ordered_among(levels, valid, _KEPT_LABELS)
    ]
    rare = below = None
    if rare_share is not None:
        # The share as each class reports it, so the two can't disagree.
        below = valid & (levels.counts / max(valid_count, 1) < rare_share)
        rare = [
            _class(view, position, valid_count)
            for position in ordered_among(levels, below, _KEPT_LABELS)
        ]
    unexpected = None
    # Row IDs show only when labels are checked; then repeated ones limit them.
    ambiguous: list[Limitation] = []
    if labels is not None:
        unexpected, ambiguous = _unexpected_labels(
            view, ~valid, row_levels, row_ids, masking, target
        )
    missing = levels.missing_count
    non_missing = row_count - missing
    unexpected_count = None if labels is None else int((~valid).sum())
    status, status_reason = _classification_status(row_count, non_missing, valid_count)
    shown: list[str | None] = [target.name] if len(levels.counts) else []
    if unexpected:
        shown.append(row_ids.id_column)
    return _Profile(
        status=status,
        status_reason=status_reason,
        findings=[
            *_classification_findings(
                target,
                task,
                class_count=len(counts),
                unexpected_count=unexpected_count,
                unexpected_rows=non_missing - valid_count,
                # Each label that isn't missing occurs once.
                distinct=non_missing if len(levels.counts) == non_missing else 0,
            ),
            *_missing_labels(target, missing, row_count),
            *_rare_classes(
                target,
                rare_share,
                class_count=len(counts),
                rare_count=0 if below is None else int(below.sum()),
            ),
        ],
        limitations=[
            *ambiguous,
            *stored_nan_limitations(
                [target.name] if stored_nan_count(series) else [],
                ["missing_count"],
                "missing_count counts it",
            ),
            *([_NO_EXPECTED_LABELS] if labels is None else []),
            *([_NO_RARE_SHARE] if rare_share is None else []),
            *_null_shares(row_count, valid_count),
        ],
        support=ColumnSupport(
            column=target.name,
            eligible_count=row_count,
            evaluated_count=valid_count,
            excluded_count=row_count - valid_count,
            excluded_by_reason={
                reason: count
                for reason, count in (
                    ("missing", missing),
                    ("unexpected_label", non_missing - valid_count),
                )
                if count
            },
        ),
        fields={
            "non_missing_count": non_missing,
            "missing_count": missing,
            "missing_share": missing / row_count if row_count else None,
            "valid_label_count": valid_count,
            "class_count": len(counts),
            "classes": classes,
            **_balance(counts, valid_count),
            "unexpected_label_count": unexpected_count,
            "unexpected_labels": unexpected,
            "rare_class_count": None if below is None else int(below.sum()),
            "rare_classes": rare,
        },
        shown=shown,
        **_table(view, valid if labels is not None else None),
        expected_labels=_recorded_labels(view, labels, valid),
    )


def _count_labels(target: Column, series: pd.Series) -> tuple[Levels, np.ndarray]:
    """The target's levels and each row's level, or raise when they can't be hashed."""
    counted = count_levels_by_row(series, target.kind)
    if counted is None:
        raise ColumnTypeError(
            f"profile_target can't use column {target.name!r} with dtype "
            f"{target.dtype} as a classification target: it holds values that can't "
            "be hashed, such as lists or dicts, and so can't be class labels. Pass a "
            "column of labels such as strings or numbers."
        )
    return counted


def _unexpected_labels(
    view: _Labels,
    outside: np.ndarray,
    row_levels: np.ndarray,
    row_ids: RowIds,
    masking: MaskingDecision,
    target: Column,
) -> tuple[list[ProfileTargetUnexpectedLabel], list[Limitation]]:
    """The first labels outside expected_labels with the IDs of their first rows.

    Also gives the ambiguous_row_ids limitation when row IDs repeat. Row IDs
    from a sensitive id_column become placeholders, numbered in the order the
    labels show them; when id_column is the target itself, a row ID is its
    label, so it shows the label's placeholder.
    """
    levels = view.levels
    kept = ordered_among(levels, outside, _KEPT_LABELS).tolist()
    rows = _first_rows(row_levels, np.asarray(kept, dtype=np.intp), len(levels.counts))
    ids: list[list[object]] = [row_ids.at(positions) for positions in rows]
    if masking.masks(row_ids.id_column):
        if row_ids.id_column == target.name and view.masked is not None:
            masked = view.masked
            ids = [
                [masked.label(position)] * len(entry)
                for position, entry in zip(kept, ids, strict=True)
            ]
        else:
            placeholders = iter(mask_values([i for entry in ids for i in entry]))
            ids = [[next(placeholders) for _ in entry] for entry in ids]
    entries = []
    for position, entry_ids in zip(kept, ids, strict=True):
        label, label_type = view.entry(position)
        entries.append(
            ProfileTargetUnexpectedLabel(
                label=label,
                label_type=label_type,
                count=int(levels.counts[position]),
                row_ids=entry_ids,
            )
        )
    ambiguous = ambiguous_row_ids(
        row_ids,
        [
            f"unexpected_labels[{number}].row_ids"
            for number, positions in enumerate(rows)
            if row_ids.repeated[positions].any()
        ],
    )
    return entries, ambiguous


def _table(view: _Labels, expected: np.ndarray | None) -> dict[str, Any]:
    """The labels a result keeps for to_frame(), and which expected_labels holds.

    A masked table keeps only the placeholders' numbers, in level order, so
    the flags follow that order.
    """
    if view.masked is None:
        return {"labels": view.levels, "expected": expected}
    masked_expected = None
    if expected is not None:
        masked_expected = np.empty_like(expected)
        masked_expected[view.masked.numbers - 1] = expected
    return {"labels": view.masked.levels, "expected": masked_expected}


def _recorded_labels(
    view: _Labels, labels: Sequence[Label] | None, valid: np.ndarray
) -> list[Label] | list[str] | None:
    """expected_labels as the result's parameters record them.

    With masking on, each label shows as the placeholder of the class it
    matches, and a label that never occurs as a new placeholder, numbered on
    from the target's labels in the order given.
    """
    if labels is None or view.masked is None:
        return None if labels is None else list(labels)
    # Python's == decides which label a class matches, as in _valid_levels.
    index: dict[object, int] = {label: number for number, label in enumerate(labels)}
    found: dict[int, int] = {}
    for position in np.flatnonzero(valid).tolist():
        number = index.get(stored_label(view.levels.label(position)))
        if number is not None:
            found.setdefault(number, position)
    after = len(view.levels.counts)
    recorded = []
    for number in range(len(labels)):
        if number in found:
            recorded.append(view.masked.label(found[number]))
        else:
            after += 1
            recorded.append(placeholder(after))
    return recorded


def _classification_status(
    row_count: int, non_missing: int, valid_count: int
) -> tuple[Status, StatusReason | None]:
    """insufficient_data without rows, labels or valid labels; completed otherwise."""
    if not row_count:
        return "insufficient_data", NO_ROWS
    if not non_missing:
        return "insufficient_data", _NO_LABELS
    if not valid_count:
        return "insufficient_data", _NO_VALID_LABELS
    return "completed", None


_NO_LABELS = StatusReason(
    code="no_labels",
    message="Every target value is missing, so there are no labels to count.",
)
_NO_VALID_LABELS = StatusReason(
    code="no_valid_labels",
    message=(
        "No target value is in expected_labels; unexpected_labels lists the labels "
        "found."
    ),
)
# The shares that need at least one valid label.
_BALANCE_FIELDS = ("majority_share", "minority_share", "imbalance_ratio")


def _balance(counts: np.ndarray, valid_count: int) -> dict[str, float | None]:
    """The majority and minority shares and the imbalance ratio of the classes."""
    if not valid_count:
        return dict.fromkeys(_BALANCE_FIELDS)
    majority, minority = int(counts.max()), int(counts.min())
    return {
        "majority_share": majority / valid_count,
        "minority_share": minority / valid_count,
        "imbalance_ratio": majority / minority,
    }


def _null_shares(row_count: int, valid_count: int) -> list[Limitation]:
    """The too_few_values limitation for the shares that don't exist, if any."""
    if not row_count:
        return [
            Limitation(
                code="too_few_values",
                reason=(
                    "The data has no rows, so missing_share, majority_share, "
                    "minority_share and imbalance_ratio are null."
                ),
                fields=["missing_share", *_BALANCE_FIELDS],
            )
        ]
    if not valid_count:
        return [
            Limitation(
                code="too_few_values",
                reason=(
                    "No label is valid, so majority_share, minority_share and "
                    "imbalance_ratio are null."
                ),
                fields=list(_BALANCE_FIELDS),
            )
        ]
    return []


def _classification_findings(
    target: Column,
    task: Task,
    *,
    class_count: int,
    unexpected_count: int | None,
    unexpected_rows: int,
    distinct: int,
) -> list[Finding]:
    """The findings about a classification target, violations first.

    `distinct` counts the labels that aren't missing when each occurs once, and
    is 0 otherwise.
    """
    findings = []
    if class_count == 1:
        findings.append(_finding("single_class", target, _SINGLE_CLASS))
    if task == "binary_classification" and class_count > 2:
        claim = (
            f"The target has {class_count:,} classes, more than the 2 of "
            "binary_classification."
        )
        findings.append(_finding("more_than_two_classes", target, claim))
    if unexpected_count:
        are = "is" if unexpected_count == 1 else "are"
        claim = (
            f"{plural(unexpected_count, 'label', 'labels')} in "
            f"{plural(unexpected_rows, 'row', 'rows')} {are} outside "
            "expected_labels."
        )
        findings.append(_finding("unexpected_labels", target, claim))
    elif unexpected_count == 0 and class_count:
        claim = "Every label that isn't missing is in expected_labels."
        findings.append(_finding("no_unexpected_labels", target, claim))
    if distinct >= 2:
        claim = (
            f"All {distinct:,} target values that aren't missing are distinct, as in "
            "an identifier or a measurement rather than class labels."
        )
        findings.append(_finding("all_labels_distinct", target, claim))
    return findings


def _missing_labels(target: Column, missing: int, row_count: int) -> list[Finding]:
    """The missing_labels finding, when target values are missing."""
    if not missing:
        return []
    percent = format_number(missing / row_count * 100)
    has = (
        "has no label: its target value is missing"
        if missing == 1
        else "have no label: their target values are missing"
    )
    claim = f"{missing:,} of {plural(row_count, 'row', 'rows')} ({percent}%) {has}."
    return [_finding("missing_labels", target, claim)]


def _rare_classes(
    target: Column, rare_share: float | None, *, class_count: int, rare_count: int
) -> list[Finding]:
    """The rare_classes finding, when classes have a share below rare_share."""
    if rare_share is None or not rare_count:
        return []
    have = "has" if rare_count == 1 else "have"
    claim = (
        f"{rare_count:,} of {plural(class_count, 'class', 'classes')} {have} a share "
        f"below rare_share {format_number(rare_share)}."
    )
    return [_finding("rare_classes", target, claim)]


def _regression_findings(
    target: Column, summary: NumericSummary, row_count: int
) -> list[Finding]:
    """The findings about a regression target, as summarize_numeric decides them."""
    findings = []
    if summary.constant:
        claim = (
            "Every target value that isn't missing is the same, so a regression has "
            "nothing to predict."
        )
        findings.append(
            _finding(
                "constant_target",
                target,
                claim,
                ["statistics.infinite_count"]
                if summary.constant_infinite
                else ["statistics.min", "statistics.max"],
            )
        )
    if summary.infinite_count:
        are = "is" if summary.infinite_count == 1 else "are"
        claim = (
            f"{plural(summary.infinite_count, 'target value', 'target values')} {are} "
            "infinite, which every statistic leaves out."
        )
        findings.append(_finding("infinite_values", target, claim))
    findings.extend(_missing_labels(target, summary.missing_count, row_count))
    return findings


_SINGLE_CLASS = (
    "Every valid label is the same class, so a classifier has nothing to tell apart."
)
# The fields behind each finding.
_FINDING_FIELDS = {
    "single_class": ["class_count", "classes"],
    "more_than_two_classes": ["class_count", "classes"],
    "unexpected_labels": ["unexpected_label_count", "unexpected_labels"],
    "no_unexpected_labels": ["unexpected_label_count"],
    "all_labels_distinct": ["class_count", "non_missing_count"],
    "missing_labels": ["missing_count", "missing_share"],
    "rare_classes": ["rare_class_count", "rare_classes"],
    "infinite_values": ["statistics.infinite_count", "non_finite_count"],
}


def _finding(
    code: str, target: Column, claim: str, fields: list[str] | None = None
) -> Finding:
    """A finding about the target, with the assessment its code always has.

    `fields` replaces the fields the code is usually backed by.
    """
    return Finding(
        code=code,
        claim=claim,
        assessment=_ASSESSMENTS[code],
        columns=[target.name],
        fields=_FINDING_FIELDS[code] if fields is None else fields,
    )


_ASSESSMENTS: dict[str, Assessment] = {
    "single_class": "violation",
    "more_than_two_classes": "violation",
    "unexpected_labels": "violation",
    "no_unexpected_labels": "passed",
    "all_labels_distinct": "suspected",
    "constant_target": "violation",
    "infinite_values": "suspected",
    "missing_labels": "descriptive",
    "rare_classes": "descriptive",
}


_NO_EXPECTED_LABELS = Limitation(
    code="expected_labels_not_declared",
    reason=(
        "Labels outside an expected set weren't checked: pass expected_labels to "
        "check them."
    ),
    fields=["unexpected_label_count", "unexpected_labels"],
)
_NO_RARE_SHARE = Limitation(
    code="rare_share_not_declared",
    reason=(
        "Rare classes weren't checked: pass rare_share, the share below which a "
        "class is rare."
    ),
    fields=["rare_class_count", "rare_classes"],
)


def _regression(data: pd.DataFrame, target: Column) -> _Profile:
    """summarize_numeric's entry for the target, with its defaults."""
    column = read_numeric(data.iloc[:, target.position])
    summary = summarize_values(
        column,
        quantiles=list(DEFAULT_QUANTILES),
        quantile_method=DEFAULT_QUANTILE_METHOD,
        trim_fraction=DEFAULT_TRIM_FRACTION,
    )
    statistics = numeric_entry(ProfileTargetStatistics, target.name, summary)
    row_count = len(data)
    missing = summary.missing_count
    limitations = [
        *read_limitations({target.name: column}, _statistics_path),
        *null_limitations({target.name: summary}, _statistics_path),
    ]
    if not row_count:
        # One limitation per code: missing_share is null for the same reason.
        [nulls] = [lim for lim in limitations if lim.code == "too_few_values"]
        limitations = [
            Limitation(
                code="too_few_values",
                reason=(
                    "The data has no rows, so missing_share and the target's "
                    "statistics are null."
                ),
                fields=["missing_share", *nulls.fields],
            )
        ]
    return _Profile(
        status="insufficient_data" if not row_count else statistics.status,
        status_reason=NO_ROWS if not row_count else statistics.status_reason,
        findings=_regression_findings(target, summary, row_count),
        support=numeric_support(target.name, summary),
        fields={
            "non_missing_count": summary.count,
            "missing_count": missing,
            "missing_share": missing / row_count if row_count else None,
            "statistics": statistics,
            "non_finite_count": missing + summary.infinite_count,
        },
        limitations=limitations,
    )


def _statistics_path(name: str) -> str:
    """The result path of a regression target's statistics."""
    return "statistics"


def _valid_levels(levels: Levels, labels: Sequence[Label] | None) -> np.ndarray:
    """Which levels are valid labels: those in labels, or all of them without labels.

    A level matches a label when Python's == says they are equal, as pandas'
    value_counts merges values, so 1 matches 1.0 and True.
    """
    size = len(levels.counts)
    if labels is None:
        return np.ones(size, dtype=bool)
    expected = set(labels)
    return np.fromiter(
        (stored_label(label) in expected for label in levels.objects()),
        dtype=bool,
        count=size,
    )


def _first_rows(
    row_levels: np.ndarray, kept: np.ndarray, level_count: int
) -> list[np.ndarray]:
    """For each kept level, the positions of the first rows holding it."""
    slots = np.full(level_count + 1, -1, dtype=np.intp)
    slots[kept] = np.arange(len(kept))
    # Missing values have level -1, which picks the extra slot at the end.
    row_slots = slots[row_levels]
    rows = np.flatnonzero(row_slots >= 0)
    rows = rows[np.argsort(row_slots[rows], kind="stable")]
    starts = np.searchsorted(row_slots[rows], np.arange(len(kept) + 1))
    return [
        rows[start : min(start + _KEPT_ROW_IDS, end)]
        for start, end in zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True)
    ]


def _class(view: _Labels, position: int, valid: int) -> ProfileTargetClass:
    """The class at a level position, with its share of the valid labels."""
    label, label_type = view.entry(position)
    count = int(view.levels.counts[position])
    return ProfileTargetClass(
        label=label, label_type=label_type, count=count, share=count / valid
    )
