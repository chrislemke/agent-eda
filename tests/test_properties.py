"""Every registered analysis keeps the result contract on generated DataFrames.

The checks loop over ae.list_analyses() and call each analysis with its recipe
from tests/recipes.py, so a new analysis is held to them without new tests.
"""

import copy
import json
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from recipes import RECIPES, Arguments, recipe_for
from strategies import EDGE_FRAMES, LABEL_EDGE_FRAMES, STR, data_frames

import agent_eda as ae

ANALYSES = [info.name for info in ae.list_analyses().analyses]
INFO = {info.name: info for info in ae.list_analyses().analyses}


@pytest.mark.parametrize("name", ANALYSES)
def test_every_analysis_has_a_call_recipe(name):
    recipe_for(name)


@pytest.mark.parametrize("name", ANALYSES)
@given(data=st.data())
def test_every_analysis_keeps_the_contract_on_generated_data(name, data):
    recipe = recipe_for(name)
    df = data.draw(data_frames(any_column_labels=recipe.any_column_labels), "data")
    arguments = recipe.required(df)
    assume(arguments is not None)  # the data allows no valid call
    arguments |= data.draw(recipe.options(df), "arguments")

    check_contract(name, df, arguments)


def edge_cases():
    """Each analysis with each fixed frame it takes."""
    cases = []
    for name in ANALYSES:
        recipe = RECIPES.get(name)
        any_labels = recipe is not None and recipe.any_column_labels
        for frame in [*EDGE_FRAMES, *(LABEL_EDGE_FRAMES if any_labels else [])]:
            cases.append(pytest.param(name, frame, id=f"{name}-{frame}"))
    return cases


@pytest.mark.parametrize(("name", "frame"), edge_cases())
def test_every_analysis_keeps_the_contract_on_edge_cases(name, frame):
    recipe = recipe_for(name)
    df = (EDGE_FRAMES | LABEL_EDGE_FRAMES)[frame]()
    arguments = recipe.required(df)
    if arguments is None:
        pytest.skip(f"{frame!r} allows no valid call of {name}")

    check_contract(name, df, arguments)


# What the generated DataFrames must cover, read off a column or a whole frame.
COLUMN_FEATURES: dict[str, Callable[[pd.Series], bool]] = {
    "floats with NaN": lambda s: str(s.dtype) == "float64" and bool(s.isna().any()),
    "floats with infinity": lambda s: (
        str(s.dtype) == "float64" and bool(np.isinf(s.to_numpy()).any())
    ),
    "nullable integers": lambda s: str(s.dtype) in {"Int64", "UInt8"},
    "bool": lambda s: str(s.dtype) == "bool",
    "boolean": lambda s: str(s.dtype) == "boolean",
    "object strings, empty": lambda s: (
        str(s.dtype) == "object" and any(v == "" for v in s if isinstance(v, str))
    ),
    "object strings, whitespace only": lambda s: (
        str(s.dtype) == "object" and any(v.isspace() for v in s if isinstance(v, str))
    ),
    "pandas 3 strings": lambda s: (
        isinstance(s.dtype, pd.StringDtype) and s.dtype == STR
    ),
    "categoricals with unused categories": lambda s: (
        isinstance(s.dtype, pd.CategoricalDtype) and s.nunique() < len(s.cat.categories)
    ),
    "naive datetimes": lambda s: isinstance(s.dtype, np.dtype) and s.dtype.kind == "M",
    "time-zone-aware datetimes": lambda s: isinstance(s.dtype, pd.DatetimeTZDtype),
    "pyarrow-backed columns": lambda s: (
        isinstance(s.dtype, pd.ArrowDtype)
        or getattr(s.dtype, "storage", "") == "pyarrow"
    ),
    "constant columns": lambda s: (
        len(s) >= 2 and not s.isna().any() and len({repr(v) for v in s}) == 1
    ),
    "all-missing columns": lambda s: len(s) >= 1 and bool(s.isna().all()),
}
FRAME_FEATURES: dict[str, Callable[[pd.DataFrame], bool]] = {
    "zero rows": lambda df: len(df) == 0,
    "one row": lambda df: len(df) == 1,
    "zero columns": lambda df: df.shape[1] == 0,
}


def test_generated_frames_cover_every_dtype_family_and_shape():
    seen = set()

    @settings(max_examples=200)
    @given(data_frames())
    def collect(df):
        seen.update(name for name, has in FRAME_FEATURES.items() if has(df))
        for position in range(df.shape[1]):
            column = df.iloc[:, position]
            seen.update(name for name, has in COLUMN_FEATURES.items() if has(column))

    collect()

    assert [
        name for name in [*FRAME_FEATURES, *COLUMN_FEATURES] if name not in seen
    ] == []


def check_contract(name: str, df: pd.DataFrame, arguments: Arguments) -> None:
    """Call the analysis twice and check what every result promises."""
    before = Snapshot(df)
    analysis = getattr(ae, name)
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        # As everywhere in the suite, these fail the test instead.
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        res = analysis(df, **arguments)
        text = str(res)
        record = res.to_json()
        res.to_frame()
        again = analysis(df, **arguments).to_json()

    assert res.status in INFO[name].statuses
    json.loads(record, parse_constant=reject_constant)
    assert again == record, "two identical calls gave different JSON"
    before.assert_unchanged(df)
    assert [str(w.message) for w in escaped] == [], "warnings reached the caller"
    lines = text.splitlines()
    assert len(lines) <= 40, f"the text has {len(lines)} lines"
    assert max(len(line) for line in lines) <= 120, "a line of the text is too wide"


class Snapshot:
    """A DataFrame's values, dtypes, index, columns and attrs, to compare later."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.frame = df.copy(deep=True)
        # A deep copy of a DataFrame shares the index and columns' values.
        self.index = df.index.copy(deep=True)
        self.columns = df.columns.copy(deep=True)
        self.attrs = copy.deepcopy(df.attrs)

    def assert_unchanged(self, df: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(
            df,
            self.frame,
            check_exact=True,
            check_index_type=True,
            check_column_type=True,
            check_flags=True,
            obj="the input DataFrame",
        )
        pd.testing.assert_index_equal(df.index, self.index, exact=True)
        pd.testing.assert_index_equal(df.columns, self.columns, exact=True)
        assert df.attrs == self.attrs, "the analysis changed the input's attrs"


def reject_constant(name):
    raise AssertionError(f"to_json() isn't strict JSON: it holds {name}")
