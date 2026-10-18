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


def declared_schemas(
    data: pd.DataFrame, analysis: str, parameter: str
) -> st.SearchStrategy[dict[str, Any]]:
    """Declarations for some columns, and for one the data lacks.

    Each declares a logical type, a ColumnSpec or a ColumnSpec as a dict.
    """
    logical_types = st.sampled_from(choices(analysis, parameter))
    specs = st.builds(
        ae.ColumnSpec, logical_type=logical_types, nullable=st.none() | st.booleans()
    )
    declarations = logical_types | specs | specs.map(lambda spec: spec.model_dump())
    columns = st.sampled_from([*names(data, any_dtype), "not_a_column"])
    return st.dictionaries(columns, declarations, max_size=4)


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
    if isinstance(dtype, pd.CategoricalDtype | pd.StringDtype):
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
            max_categories=st.integers(2, 60),
            max_pairs=st.integers(1, 6_000),
        )
    )


def has_countable_values(dtype: Any) -> bool:
    """Text, object, categorical, boolean and numeric columns, pyarrow ones too."""
    if isinstance(dtype, pd.CategoricalDtype | pd.StringDtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.kind in "biufU" or pd.api.types.is_numeric_dtype(dtype)
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
    labels = {str(label): label for label in data.columns}
    selected = arguments.get("columns") or list(labels)
    candidates = {
        name: values
        for name in selected
        if (values := sentinel_candidates(data[labels[name]]))
    }
    if candidates and draw(st.booleans()):
        chosen = draw(st.lists(st.sampled_from(list(candidates)), unique=True))
        arguments["sentinel_values"] = {
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
    elif draw(st.booleans()):
        arguments["sentinel_values"] = None
    return arguments


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
        # Row IDs from a sensitive id_column.
        sensitive_options=lambda data, sensitive: optional(
            id_column=st.sampled_from(sensitive)
        ),
    ),
    "profile_missing_values": Recipe(options=missing_value_arguments),
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
}
