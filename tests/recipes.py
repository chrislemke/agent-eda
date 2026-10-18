"""How the property tests and the benchmark call each analysis.

Every analysis in ae.list_analyses() needs a recipe here; one without fails the
suite. A recipe builds valid arguments for a given DataFrame: `required` gives
the arguments a call can't do without, and the benchmark passes those alone, so
every option keeps its default; `options` draws further arguments, which the
property tests add.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from hypothesis import strategies as st

import agent_eda as ae

type Arguments = dict[str, Any]


def _no_arguments(data: pd.DataFrame) -> Arguments | None:
    return {}


def _no_options(data: pd.DataFrame) -> st.SearchStrategy[Arguments]:
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


# Any text, sometimes longer than a printed line.
TEXT = st.text() | st.text(min_size=121, max_size=130)

RECIPES: dict[str, Recipe] = {
    "describe_dataset": Recipe(
        options=lambda data: optional(
            source=st.none() | TEXT, source_version=st.none() | TEXT
        ),
        any_column_labels=True,
    ),
    "find_duplicate_rows": Recipe(
        options=lambda data: optional(
            columns=column_lists(names(data, lambda dtype: True)),
            id_column=column_names(hashable_names(data)),
        ),
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
}
