"""How the property tests and the benchmark call each analysis.

Every analysis in ae.list_analyses() needs a recipe here; one without fails the
suite. A recipe builds valid arguments for a given DataFrame: `required` gives
the arguments a call can't do without, and the benchmark passes those alone, so
every option keeps its default; `options` draws further arguments, which the
property tests add.
"""

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import strategies as st

import agent_eda as ae

type Arguments = dict[str, Any]


def _no_arguments(data: pd.DataFrame) -> Arguments | None:
    return {}


def _no_options(data: pd.DataFrame) -> st.SearchStrategy[Arguments]:
    return st.just({})


def _no_sensitive_options(
    data: pd.DataFrame, sensitive: list[str]
) -> st.SearchStrategy[Arguments]:
    return st.just({})


@dataclass(frozen=True)
class Recipe:
    """Valid arguments for one analysis, built for a given DataFrame."""

    # The arguments without a default, or None when the DataFrame allows no
    # valid call, such as a target column in data without columns.
    required: Callable[[pd.DataFrame], Arguments | None] = field(default=_no_arguments)
    # Arguments to add, drawn by Hypothesis; they may replace required ones.
    options: Callable[[pd.DataFrame], st.SearchStrategy[Arguments]] = field(
        default=_no_options
    )
    # Whether the analysis takes column labels that can't serve as names:
    # repeated, colliding as strings, missing or a MultiIndex.
    any_column_labels: bool = False
    # For an analysis that supports masking: arguments that make its result
    # show values of the given sensitive columns, such as id_column for row
    # IDs. The no-leak property draws them, and they may replace the options.
    sensitive_options: Callable[
        [pd.DataFrame, list[str]], st.SearchStrategy[Arguments]
    ] = field(default=_no_sensitive_options)


def recipe_for(name: str) -> Recipe:
    """The recipe for an analysis; fails the test when it has none."""
    if name not in RECIPES:
        pytest.fail(
            f"{name} has no call recipe. Add one to RECIPES in tests/recipes.py: "
            "it builds valid arguments for a DataFrame, which the property tests "
            "and the benchmark call the analysis with."
        )
    return RECIPES[name]


def optional(**options: st.SearchStrategy[Any]) -> st.SearchStrategy[Arguments]:
    """Arguments that each keep their default or take a drawn value."""
    return st.fixed_dictionaries({}, optional=options)


def choices(analysis: str, parameter: str) -> list[Any]:
    """The values a parameter allows, as ae.describe_analysis() tells an agent."""
    [described] = [
        p for p in ae.describe_analysis(analysis).parameters if p.name == parameter
    ]
    assert described.allowed_values, f"{analysis}'s {parameter} has no fixed values"
    return list(described.allowed_values)


def names(data: pd.DataFrame, keep: Callable[[Any], bool]) -> list[str]:
    """The names of the columns whose dtype `keep` accepts, in DataFrame order."""
    return [str(label) for label, dtype in data.dtypes.items() if keep(dtype)]


def column_lists(candidates: list[str]) -> st.SearchStrategy[list[str] | None]:
    """None, or some of the candidates, each once, in any order."""
    if not candidates:
        return st.none()
    return st.none() | st.lists(st.sampled_from(candidates), min_size=1, unique=True)


# Formats a ColumnSpec can declare for datetimes stored as text.
DATETIME_FORMATS = ["%d/%m/%Y", "%Y%m%d", "%d.%m.%Y %H:%M", "%b %d, %Y"]


def declared_schemas(
    data: pd.DataFrame, analysis: str, parameter: str, *, min_size: int = 0
) -> st.SearchStrategy[dict[str, Any]]:
    """Declarations for some columns, and for one the data lacks.

    Each declares a logical type, a ColumnSpec or a ColumnSpec as a dict.
    """
    logical_types = st.sampled_from(choices(analysis, parameter))
    nullable = st.none() | st.booleans()
    specs = st.builds(ae.ColumnSpec, logical_type=logical_types, nullable=nullable)
    specs |= st.builds(
        lambda datetime_format, nullable: ae.ColumnSpec(
            "datetime", nullable=nullable, datetime_format=datetime_format
        ),
        st.sampled_from(DATETIME_FORMATS),
        nullable,
    )
    declarations = logical_types | specs | specs.map(lambda spec: spec.model_dump())
    columns = st.sampled_from([*names(data, any_dtype), "not_a_column"])
    return st.dictionaries(columns, declarations, min_size=min_size, max_size=4)


def every_column_declared(data: pd.DataFrame) -> Arguments:
    """A schema declaring every column, the logical types in turn by position.

    Data without columns gets a declaration for a column it lacks.
    """
    logical_types = choices("validate_column_types", "schema")
    labels = names(data, any_dtype)
    if not labels:
        return {"schema": {"not_a_column": "integer"}}
    return {
        "schema": {
            name: logical_types[position % len(logical_types)]
            for position, name in enumerate(labels)
        }
    }


def column_names(candidates: list[str]) -> st.SearchStrategy[str | None]:
    """None, or one of the candidates."""
    if not candidates:
        return st.none()
    return st.none() | st.sampled_from(candidates)


def hashable_names(data: pd.DataFrame) -> list[str]:
    """The names of the columns whose values can all be hashed, as row IDs must."""
    found = []
    for label, column in data.items():
        try:
            pd.factorize(column)
        except (TypeError, NotImplementedError):  # lists, dicts, nested Arrow types
            continue
        found.append(str(label))
    return found


def is_dictionary(dtype: Any) -> bool:
    """pyarrow's dictionary type: the Arrow form of a categorical."""
    return isinstance(dtype, pd.ArrowDtype) and pa.types.is_dictionary(
        dtype.pyarrow_dtype
    )


def is_number_or_boolean(dtype: Any) -> bool:
    """Integers, floats and booleans, nullable and pyarrow-backed ones included."""
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if dtype.kind in "iufb":
        return True
    # pyarrow decimals are numbers whose kind is "O".
    return isinstance(dtype, pd.ArrowDtype) and pd.api.types.is_numeric_dtype(dtype)


def is_pairable(dtype: Any) -> bool:
    """Numbers, booleans, text, objects and categoricals, whatever their storage."""
    if isinstance(dtype, pd.CategoricalDtype | pd.StringDtype) or is_dictionary(dtype):
        return True
    if is_number_or_boolean(dtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.kind == "U"  # strings; lists, dates and durations aren't
    # Object columns; periods and intervals, also of kind "O", aren't.
    return isinstance(dtype, np.dtype) and dtype.kind == "O"


def association_options(data: pd.DataFrame) -> st.SearchStrategy[Arguments]:
    """Columns to pair, some of them read as categorical, and any other option."""
    pairable = names(data, is_pairable)
    return column_lists(pairable).flatmap(
        lambda chosen: optional(
            columns=st.just(chosen),
            categorical_columns=column_lists(pairable if chosen is None else chosen),
            numeric_method=st.sampled_from(
                choices("compute_association_matrix", "numeric_method")
            ),
            categorical_method=st.sampled_from(
                choices("compute_association_matrix", "categorical_method")
            ),
            mixed_method=st.sampled_from(
                choices("compute_association_matrix", "mixed_method")
            ),
            max_categories=st.integers(2, 60),
            max_pairs=st.integers(1, 6_000),
        )
    )


def has_countable_values(dtype: Any) -> bool:
    """Text, object, categorical, boolean and numeric columns, pyarrow ones too."""
    if isinstance(dtype, pd.CategoricalDtype | pd.StringDtype) or is_dictionary(dtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.kind in "biufU" or pd.api.types.is_numeric_dtype(dtype)
    if isinstance(dtype, pd.SparseDtype):  # only sparse numbers and booleans
        return dtype.kind in "biuf"
    return dtype.kind in "biufO"


def any_dtype(dtype: Any) -> bool:
    return True


def sentinel_candidates(column: pd.Series) -> list[Any]:
    """Values an agent could declare as sentinel values for the column.

    They are values the column holds, plus a stand-in it may not hold, all of a
    type the column can hold, so declaring any of them is a valid call.
    """
    values = (
        column.cat.categories
        if isinstance(column.dtype, pd.CategoricalDtype)
        else column
    )
    kept = [
        value for value in (_sentinel(v) for v in values.tolist()) if value is not None
    ]
    kinds = {type(value) for value in kept}
    stand_ins = {str: "N/A", int: 0, float: -999.0}
    kept.extend(stand_ins[kind] for kind in kinds if kind in stand_ins)
    return list(dict.fromkeys(kept))


def _sentinel(value: Any) -> Any:
    """A value as a declared sentinel value, or None when it can't be one."""
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None  # NaN means missing, and infinities can't be declared
    if value is pd.NaT or value is pd.NA or isinstance(value, list | dict):
        return None
    if isinstance(value, bool | int | float | str | dt.date | dt.timedelta):
        return value
    return None


@st.composite
def missing_value_arguments(draw: st.DrawFn, data: pd.DataFrame) -> Arguments:
    """Columns, and sentinel values for some of the selected ones."""
    arguments = draw(optional(columns=column_lists(names(data, any_dtype))))
    selected = arguments.get("columns") or [str(label) for label in data.columns]
    return arguments | draw(declared_sentinel_values(data, selected))


@st.composite
def declared_sentinel_values(
    draw: st.DrawFn, data: pd.DataFrame, selected: list[str]
) -> Arguments:
    """sentinel_values for some of the selected columns, None, or left out."""
    labels = {str(label): label for label in data.columns}
    candidates = {
        name: values
        for name in selected
        if (values := sentinel_candidates(data[labels[name]]))
    }
    if candidates and draw(st.booleans()):
        chosen = draw(st.lists(st.sampled_from(list(candidates)), unique=True))
        return {
            "sentinel_values": {
                name: draw(
                    st.lists(
                        st.sampled_from(candidates[name]),
                        min_size=1,
                        max_size=3,
                        unique=True,
                    )
                )
                for name in chosen
            }
        }
    if draw(st.booleans()):
        return {"sentinel_values": None}
    return {}


@st.composite
def sentinel_arguments(draw: st.DrawFn, data: pd.DataFrame) -> Arguments:
    """id_column, columns, and sentinel values for some of the selected columns.

    columns=None leaves id_column out, so no value is declared for it then.
    """
    arguments = draw(optional(id_column=column_names(hashable_names(data))))
    arguments |= draw(optional(columns=column_lists(names(data, any_dtype))))
    selected = arguments.get("columns") or [
        name for name in names(data, any_dtype) if name != arguments.get("id_column")
    ]
    return arguments | draw(declared_sentinel_values(data, selected))


@st.composite
def sentinels_of_sensitive_columns(
    draw: st.DrawFn, data: pd.DataFrame, sensitive: list[str]
) -> Arguments:
    """Arguments that make detect_missing_sentinels show sensitive values.

    It checks the sensitive columns and declares some of their own values,
    which become candidates, or takes row IDs from one of them, or both.
    """
    id_column = draw(st.none() | st.sampled_from(sensitive))
    columns = draw(st.none() | st.permutations(sensitive).map(list))
    selected = columns or [name for name in names(data, any_dtype) if name != id_column]
    labels = {str(label): label for label in data.columns}
    declared = {
        name: draw(
            st.lists(st.sampled_from(values), min_size=1, max_size=3, unique=True)
        )
        for name in sensitive
        if name in selected and (values := sentinel_candidates(data[labels[name]]))
    }
    return {
        "id_column": id_column,
        "columns": columns,
        "sentinel_values": declared or None,
    }


@st.composite
def declared_values_of_sensitive_columns(
    draw: st.DrawFn, data: pd.DataFrame, sensitive: list[str]
) -> Arguments:
    """Arguments that make profile_missing_values show sensitive values: every
    column selected, and some of each sensitive column's own values declared."""
    labels = {str(label): label for label in data.columns}
    declared = {
        name: draw(
            st.lists(st.sampled_from(values), min_size=1, max_size=3, unique=True)
        )
        for name in sensitive
        if (values := sentinel_candidates(data[labels[name]]))
    }
    return {"columns": None, "sentinel_values": declared or None}


@st.composite
def masking_arguments(draw: st.DrawFn, data: pd.DataFrame) -> Arguments:
    """project= and mask_sensitive= in any valid combination, or left out.

    The project declares some of the data's columns sensitive, maybe with a
    name the data lacks, or none. mask_sensitive=True comes only with sensitive
    columns, since without them it raises.
    """
    candidates = [*(str(label) for label in data.columns), "not in the data"]
    sensitive = draw(st.lists(st.sampled_from(candidates), unique=True, max_size=3))
    arguments: Arguments = {}
    if sensitive or draw(st.booleans()):
        arguments["project"] = ae.Project(
            name=draw(st.none() | st.sampled_from(["churn", "support tickets"])),
            data_policy=ae.DataPolicy(
                sensitive_columns=sensitive,
                mask_sensitive=bool(sensitive) and draw(st.booleans()),
            ),
        )
    mask_sensitive = draw(
        st.sampled_from([None, False, *([True] if sensitive else [])])
    )
    if mask_sensitive is not None or draw(st.booleans()):
        arguments["mask_sensitive"] = mask_sensitive
    return arguments


def with_masking(
    options: Callable[[pd.DataFrame], st.SearchStrategy[Arguments]],
) -> Callable[[pd.DataFrame], st.SearchStrategy[Arguments]]:
    """An analysis's options plus project= and mask_sensitive=, for the analyses
    that support masking."""
    return lambda data: st.tuples(options(data), masking_arguments(data)).map(
        lambda parts: {**parts[0], **parts[1]}
    )


def is_number(dtype: Any) -> bool:
    """Integers and floats, nullable and pyarrow-backed ones included, not booleans."""
    return is_number_or_boolean(dtype) and dtype.kind != "b"


def label_columns(data: pd.DataFrame) -> list[str]:
    """The columns whose values can be class labels: numbers, booleans, text,
    objects and categories that can all be hashed."""
    hashable = set(hashable_names(data))
    return [name for name in names(data, is_pairable) if name in hashable]


def column_named(data: pd.DataFrame, name: str) -> pd.Series:
    """The column whose label's string form is `name`."""
    [position] = [i for i, label in enumerate(data.columns) if str(label) == name]
    return data.iloc[:, position]


def label_pool(column: pd.Series, stand_ins: dict[type, list[Any]]) -> list[Any]:
    """Labels an agent could expect in the column: its own and stand-ins it lacks.

    Each is a str, bool, int or finite float the column's values could equal,
    and no two are equal, as Python's == decides, so 1 and True are one label.
    """
    values = (
        column.cat.categories
        if isinstance(column.dtype, pd.CategoricalDtype)
        else column
    )
    pool: dict[Any, Any] = {}
    for value in values.tolist():
        if isinstance(value, Decimal):
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, bool | int | float | str):
            pool.setdefault(value, value)
    kinds = {bool if isinstance(v, bool) else type(v) for v in pool.values()}
    for kind in sorted(kinds, key=lambda kind: kind.__name__):
        for stand_in in stand_ins.get(kind, []):
            pool.setdefault(stand_in, stand_in)
    return list(pool.values())


# Labels that may not occur in a column, by the type of the labels it holds.
STAND_INS: dict[type, list[Any]] = {
    str: ["stand-in a", "stand-in b"],
    bool: [True, False],
    int: [123_456_789, 987_654_321],
    float: [0.125, 0.625],
}


@st.composite
def expected_labels(draw: st.DrawFn, pool: list[Any], task: str) -> list[Any] | None:
    """Labels for expected_labels: 2 for a binary task, 2 or more otherwise."""
    if len(pool) < 2:
        return None
    size = 2 if task == "binary_classification" else draw(st.integers(2, len(pool)))
    return draw(
        st.lists(st.sampled_from(pool), min_size=size, max_size=size, unique=True)
    )


@st.composite
def target_arguments(draw: st.DrawFn, data: pd.DataFrame) -> Arguments:
    """A target and a task that fits it, the checks that need declared values,
    and id_column."""
    candidates = {"regression": names(data, is_number)}
    labels = label_columns(data)
    for task in ("binary_classification", "multiclass_classification"):
        candidates[task] = labels
    tasks = [task for task in choices("profile_target", "task") if candidates[task]]
    if not tasks:
        return {}
    task = draw(st.sampled_from(tasks))
    target = draw(st.sampled_from(candidates[task]))
    arguments: Arguments = {"target_column": target, "task": task}
    if task != "regression":
        pool = label_pool(column_named(data, target), STAND_INS)
        arguments |= draw(
            optional(
                expected_labels=expected_labels(pool, task),
                rare_share=st.floats(0, 1, exclude_min=True, exclude_max=True),
            )
        )
    return arguments | draw(optional(id_column=column_names(hashable_names(data))))


# Stand-in labels that hold the markers of tests/strategies.py's secrets, so a
# leaked expected label shows too.
SECRET_STAND_INS: dict[type, list[Any]] = {
    str: ["secret stand-in"],
    int: [918_273_645_99],
}


@st.composite
def sensitive_target_arguments(
    draw: st.DrawFn, data: pd.DataFrame, sensitive: list[str]
) -> Arguments:
    """A sensitive column as the target, with labels outside expected_labels, and
    a sensitive id_column for their row IDs."""
    arguments: Arguments = {}
    targets = [name for name in label_columns(data) if name in sensitive]
    if targets:
        target = draw(st.sampled_from(targets))
        own = label_pool(column_named(data, target), {})
        pool = label_pool(column_named(data, target), SECRET_STAND_INS)
        # Some of its own labels and a stand-in, so others fall outside.
        kept = (
            draw(st.lists(st.sampled_from(own), max_size=len(own) - 1, unique=True))
            if len(own) > 1
            else []
        )
        stand_ins = [label for label in pool if label not in own]
        expected = [*kept, *stand_ins][:5] if len(kept) + len(stand_ins) >= 2 else None
        arguments = {
            "target_column": target,
            "task": "multiclass_classification",
            "expected_labels": expected,
            "rare_share": draw(
                st.none() | st.floats(0, 1, exclude_min=True, exclude_max=True)
            ),
        }
    ids = [name for name in hashable_names(data) if name in sensitive]
    return arguments | draw(optional(id_column=st.sampled_from(ids or [None])))


def first_label_target(data: pd.DataFrame) -> Arguments | None:
    """The first column whose values can be class labels, for a multiclass task."""
    labels = label_columns(data)
    if not labels:
        return None
    return {"target_column": labels[0], "task": "multiclass_classification"}


# Any text, sometimes longer than a printed line.
TEXT = st.text() | st.text(min_size=121, max_size=130)

RECIPES: dict[str, Recipe] = {
    "compute_association_matrix": Recipe(options=association_options),
    "describe_dataset": Recipe(
        options=lambda data: optional(
            source=st.none() | TEXT, source_version=st.none() | TEXT
        ),
        any_column_labels=True,
    ),
    "detect_missing_sentinels": Recipe(
        options=with_masking(sentinel_arguments),
        # Candidates declared among the sensitive columns' own values, and row
        # IDs from a sensitive id_column.
        sensitive_options=sentinels_of_sensitive_columns,
    ),
    "find_candidate_keys": Recipe(
        options=lambda data: optional(
            columns=column_lists(names(data, lambda dtype: True)),
            max_key_size=st.integers(1, 4),
            max_combinations=st.integers(1, 64),
        ),
    ),
    "find_duplicate_rows": Recipe(
        options=with_masking(
            lambda data: optional(
                columns=column_lists(names(data, lambda dtype: True)),
                id_column=column_names(hashable_names(data)),
            )
        ),
        # Every column compared, so the rows that frames_with_secrets repeats
        # always form groups, and row IDs from a sensitive id_column.
        sensitive_options=lambda data, sensitive: st.fixed_dictionaries(
            {"columns": st.none(), "id_column": st.sampled_from(sensitive)}
        ),
    ),
    "profile_missing_values": Recipe(
        options=with_masking(missing_value_arguments),
        # Declared values among the sensitive columns' own values.
        sensitive_options=declared_values_of_sensitive_columns,
    ),
    "profile_target": Recipe(
        required=first_label_target,
        options=with_masking(target_arguments),
        # The labels of a sensitive target and row IDs from a sensitive column.
        sensitive_options=sensitive_target_arguments,
    ),
    "summarize_numeric": Recipe(
        options=lambda data: optional(
            columns=column_lists(names(data, is_number_or_boolean)),
            quantiles=st.lists(st.floats(0, 1), max_size=5, unique=True),
            quantile_method=st.sampled_from(
                choices("summarize_numeric", "quantile_method")
            ),
            trim_fraction=st.floats(0, 0.5, exclude_max=True),
        ),
    ),
    "summarize_categorical": Recipe(
        options=with_masking(
            lambda data: optional(
                columns=column_lists(names(data, has_countable_values)),
                top_n=st.integers(1, 100),
            )
        ),
        # The levels of the sensitive columns it can count.
        sensitive_options=lambda data, sensitive: optional(
            columns=column_lists(
                [n for n in names(data, has_countable_values) if n in sensitive]
            )
        ),
    ),
    "infer_column_semantics": Recipe(
        options=lambda data: optional(
            columns=column_lists(names(data, any_dtype)),
            declared_schema=st.none()
            | declared_schemas(data, "infer_column_semantics", "declared_schema"),
        ),
    ),
    "validate_column_types": Recipe(
        required=every_column_declared,
        options=with_masking(
            lambda data: optional(
                schema=declared_schemas(
                    data, "validate_column_types", "schema", min_size=1
                ),
                id_column=column_names(hashable_names(data)),
            )
        ),
        # Types that every sensitive value fails, text, number or datetime, so
        # the result shows them as invalid values; and row IDs from them.
        sensitive_options=lambda data, sensitive: st.fixed_dictionaries(
            {
                "schema": st.fixed_dictionaries(
                    {
                        name: st.sampled_from(["boolean", "timedelta"])
                        for name in sensitive
                    }
                )
            },
            optional={"id_column": st.sampled_from(sensitive)},
        ),
    ),
}
