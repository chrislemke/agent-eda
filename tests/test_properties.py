"""Every registered analysis keeps the result contract on generated DataFrames.

The checks loop over ae.list_analyses() and call each analysis with its recipe
from tests/recipes.py, so a new analysis is held to them without new tests.
"""

import copy
import json
import typing
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import Phase, assume, find, given, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from recipes import RECIPES, Arguments, recipe_for
from strategies import (
    EDGE_FRAMES,
    LABEL_EDGE_FRAMES,
    STR,
    data_frames,
    frames_with_secrets,
)

import agent_eda as ae

ANALYSES = [info.name for info in ae.list_analyses().analyses]
INFO = {info.name: info for info in ae.list_analyses().analyses}
# The analyses whose results can show raw values, and so support masking.
MASKING = [info.name for info in ae.list_analyses().analyses if info.supports_masking]


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


@pytest.mark.parametrize("name", MASKING)
@given(data=st.data())
def test_with_masking_on_no_value_of_a_sensitive_column_reaches_the_result(name, data):
    recipe = recipe_for(name)
    df, markers = data.draw(frames_with_secrets(), "data")
    arguments = recipe.required(df)
    assume(arguments is not None)  # the data allows no valid call
    arguments |= data.draw(recipe.options(df), "arguments")
    arguments |= data.draw(recipe.sensitive_options(df, list(markers)), "showing")
    arguments |= data.draw(masking_on(list(markers)), "masking")

    check_contract(name, df, arguments)
    res = getattr(ae, name)(df, **arguments)

    assert res.masking.applied
    outputs = {"the text": str(res), "to_json()": res.to_json(), **frames_as_text(res)}
    leaks = [
        f"{marker!r} in {where}"
        for where, text in outputs.items()
        for marker in sorted(set(markers.values()))
        if marker in text.lower()
    ]
    assert leaks == [], "raw values of sensitive columns reached the result"


def masking_on(columns: list[str]) -> st.SearchStrategy[Arguments]:
    """project= and mask_sensitive= that switch masking on for these columns.

    Either the project's data policy switches it on, and the call leaves it or
    says True too, or the call switches it on against the project's setting.
    The project may also declare a column the data lacks.
    """

    def arguments(project_masks: bool, call: bool | None, absent: bool) -> Arguments:
        sensitive = [*columns, *(["not in the data"] if absent else [])]
        policy = ae.DataPolicy(
            sensitive_columns=sensitive, mask_sensitive=project_masks
        )
        return {"project": ae.Project(data_policy=policy), "mask_sensitive": call}

    return st.builds(
        arguments, st.just(True), st.sampled_from([None, True]), st.booleans()
    ) | st.builds(arguments, st.just(False), st.just(True), st.booleans())


def frames_as_text(res: Any) -> dict[str, str]:
    """Every table to_frame() returns, as CSV text, by how to get it.

    `res` is any analysis result; some take the table to return by name.
    """
    hints = typing.get_type_hints(res.to_frame)
    tables = typing.get_args(hints["table"]) if "table" in hints else (None,)
    return {
        f"to_frame({'' if table is None else f'table={table!r}'})": (
            res.to_frame() if table is None else res.to_frame(table=table)
        ).to_csv()
        for table in tables
    }


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

    # Hypothesis also draws constants it finds in the package's own modules, so
    # any new module changes which frames come up, derandomized or not. In 500
    # frames the rarest feature comes up about a dozen times.
    @settings(max_examples=500)
    @given(data_frames())
    def collect(df):
        seen.update(features_of(df))

    collect()

    # Hypothesis also draws literals from the package's own code, so the frames
    # it draws shift whenever the code gains a constant. Search on for a frame
    # with each feature the 200 missed before calling it uncovered.
    uncovered = []
    for name in [*FRAME_FEATURES, *COLUMN_FEATURES]:
        if name in seen:
            continue
        try:
            find(
                data_frames(),
                lambda df, name=name: name in features_of(df),
                settings=settings(max_examples=1_000, phases=[Phase.generate]),
            )
        except NoSuchExample:
            uncovered.append(name)
    assert uncovered == []


def features_of(df: pd.DataFrame) -> set[str]:
    found = {name for name, has in FRAME_FEATURES.items() if has(df)}
    for position in range(df.shape[1]):
        column = df.iloc[:, position]
        found.update(name for name, has in COLUMN_FEATURES.items() if has(column))
    return found


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
    if res.status == "budget_exceeded":
        # The reason names the limit to raise.
        assert res.status_reason is not None
        limits = INFO[name].work_limit_parameters
        assert any(limit in res.status_reason.message for limit in limits)
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
