"""Every registered analysis keeps the result contract on generated DataFrames.

The checks loop over ae.list_analyses() and call each analysis with its recipe
from tests/recipes.py, so a new analysis is held to them without new tests.
"""

import ast
import copy
import datetime as dt
import json
import re
import typing
import warnings
from collections.abc import Callable
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import pydantic
import pytest
from hypothesis import Phase, assume, find, given, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from IPython.core.formatters import DisplayFormatter
from recipes import RECIPES, Arguments, recipe_for
from strategies import (
    EDGE_FRAMES,
    LABEL_EDGE_FRAMES,
    PYTHON_STR,
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
    df, markers, arguments = masking_example(name, data)

    check_contract(name, df, arguments)
    res = getattr(ae, name)(df, **arguments)

    assert res.masking.applied
    assert leaks_in(res, markers) == [], "raw values of sensitive columns reached it"


@pytest.mark.parametrize("name", MASKING)
def test_most_examples_of_the_no_leak_property_would_leak_with_masking_off(name):
    """The property can only fail where the result shows sensitive values."""
    leaked = []

    @settings(max_examples=60)
    @given(data=st.data())
    def collect(data):
        df, markers, arguments = masking_example(name, data)
        res = getattr(ae, name)(df, **{**arguments, "mask_sensitive": False})
        leaked.append(bool(leaks_in(res, markers)))

    collect()

    assert len(leaked) >= 20, "too few examples allowed a valid call"
    assert sum(leaked) * 2 >= len(leaked), (
        f"only {sum(leaked)} of {len(leaked)} examples of {name} show a sensitive "
        "value with masking off, so most of them can't catch a leak: fix the "
        "recipe's sensitive_options"
    )


def masking_example(name: str, data: st.DataObject):
    """A frame with sensitive columns, its markers and the arguments of a call
    that masks them and shows their values."""
    recipe = recipe_for(name)
    df, markers = data.draw(frames_with_secrets(), "data")
    required = recipe.required(df)
    assume(required is not None)  # the data allows no valid call
    arguments = dict(required or {})
    arguments |= data.draw(recipe.options(df), "arguments")
    arguments |= data.draw(recipe.sensitive_options(df, list(markers)), "showing")
    arguments |= data.draw(masking_on(list(markers)), "masking")
    return df, markers, arguments


def leaks_in(res: Any, markers: dict[str, str]) -> list[str]:
    """Where a marker of a sensitive value shows in a result's text, record or
    tables."""
    renderings = {
        "the text": str(res),
        "to_json()": res.to_json(),
        **frames_as_text(res),
    }
    return [
        f"{marker!r} in {where}"
        for where, text in renderings.items()
        for marker in sorted(set(markers.values()))
        if marker in text.lower()
    ]


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
    "strings stored as Python objects": lambda s: (
        isinstance(s.dtype, pd.StringDtype) and s.dtype.storage == "python"
    ),
    "period columns": lambda s: isinstance(s.dtype, pd.PeriodDtype),
    "interval columns": lambda s: isinstance(s.dtype, pd.IntervalDtype),
    "ties": lambda s: 2 <= distinct_count(s) < s.count(),
    "mixed types": lambda s: (
        str(s.dtype) == "object"
        and len({type(v) for v in s.tolist() if not is_missing(v)}) >= 2
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


def distinct_count(column: pd.Series) -> int:
    """How many different values a column holds, told apart by repr so that
    values that can't be hashed count too."""
    return len({repr(v) for v in column.tolist() if not is_missing(v)})


def is_missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and value != value)  # NaN
    )


FRAME_FEATURES: dict[str, Callable[[pd.DataFrame], bool]] = {
    "zero rows": lambda df: len(df) == 0,
    "one row": lambda df: len(df) == 1,
    "zero columns": lambda df: df.shape[1] == 0,
}


def test_generated_frames_cover_every_dtype_family_and_shape():
    seen = set()

    # 500 frames cover most features: the rarest comes up about a dozen times.
    @settings(max_examples=500)
    @given(data_frames())
    def collect(df):
        seen.update(features_of(df))

    collect()

    # Hypothesis also draws literals it finds in the package's own code, so the
    # frames it draws shift whenever the code gains a constant. A feature the 500
    # frames missed gets a search for a frame with it, of up to 1,000 frames,
    # before it counts as uncovered.
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


@pytest.mark.parametrize(
    ("feature", "column"),
    [
        ("ties", pd.Series(["a", "b", "a"])),
        ("mixed types", pd.Series([1, "a", None], dtype=object)),
        ("strings stored as Python objects", pd.Series(["a"], dtype=PYTHON_STR)),
        ("period columns", pd.Series(pd.period_range("2020-01", periods=2, freq="M"))),
        ("interval columns", pd.Series(pd.interval_range(0.0, 2.0, periods=2))),
    ],
)
def test_the_feature_checks_find_what_their_names_say(feature, column):
    assert COLUMN_FEATURES[feature](column)


@pytest.mark.parametrize(
    ("feature", "column"),
    [
        ("ties", pd.Series(["a", "a", "a"])),  # one value, constant
        ("ties", pd.Series(["a", "b", "c"])),  # no value repeats
        ("ties", pd.Series(["a", None, "b"])),  # missing values don't tie
        ("mixed types", pd.Series([1, 2, None], dtype=object)),
        ("mixed types", pd.Series([1, "a"], dtype=object)[:1]),
        ("period columns", pd.Series(["2020-01"])),
    ],
)
def test_the_feature_checks_leave_out_what_their_names_dont_say(feature, column):
    assert not COLUMN_FEATURES[feature](column)


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
    check_record(name, res, record)


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


# The most entries the record keeps of a list that grows with the data: 10 row
# IDs, 5 example values and 100 entries of anything else.
RECORD_CAP = 100
ROW_ID_CAP = 10
EXAMPLE_CAP = 5


def cap_of(field: str) -> int:
    """How many entries the record keeps of a list or mapping with this name."""
    if field == "row_ids" or field.endswith("_row_ids"):
        return ROW_ID_CAP
    if field.startswith("example") or field.endswith("_examples"):
        return EXAMPLE_CAP
    # invalid_values holds the invalid values that validate_column_types shows.
    if field == "invalid_values":
        return EXAMPLE_CAP
    return RECORD_CAP


def check_record(name: str, res: Any, text: str) -> None:
    """What every record promises, beyond being strict JSON."""
    record = res.to_dict()
    assert record == json.loads(text), "to_dict() differs from json.loads(to_json())"
    jsonschema.validate(record, type(res).record_schema())
    problems = [
        *capping_problems(record, "record"),
        *path_problems(res),
        *assessment_problems(name, res),
        *time_and_html_problems(res, record),
    ]
    assert problems == []


def pointed_at(record: dict[str, Any]) -> set[str]:
    """The entries, such as "columns['age']", that the record's paths point at."""
    entry = re.compile(rf"[A-Za-z_]\w*\[(?:{QUOTED}|[^\]'\"]*)(?:, (?:{QUOTED}))*\]")
    return {
        found.group(0)
        for kind in ("findings", "limitations")
        for item in record.get(kind, [])
        for path in item["fields"]
        if (found := entry.match(path))
    }


def capping_problems(
    node: object, path: str, pointed: set[str] | None = None
) -> list[str]:
    """Each list or mapping that exceeds its cap, or whose total doesn't fit.

    A capped list or mapping sits next to `<name>_total` with the full count.
    Whatever is at its cap is a cut or a coincidence, so it needs its total.
    """
    problems = []
    if pointed is None and isinstance(node, dict):
        pointed = pointed_at(node)
    if isinstance(node, list):
        for position, item in enumerate(node):
            problems += capping_problems(item, f"{path}[{position}]", set())
    elif isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}"
            if isinstance(value, list | dict):
                # The record keeps every entry that a finding or limitation points
                # at, so a top-level list or mapping may hold more than its cap.
                cap = max(
                    cap_of(key),
                    sum(1 for entry in pointed or () if entry.startswith(f"{key}[")),
                )
                total = node.get(f"{key}_total")
                if len(value) > cap:
                    problems.append(f"{where} has {len(value)} entries, over {cap}")
                if len(value) == cap and total is None:
                    problems.append(f"{where} is at its cap of {cap} without a total")
                if total is not None and total > len(value) and len(value) < cap:
                    problems.append(
                        f"{where} holds {len(value)} of {total} entries, short of {cap}"
                    )
            if key.endswith("_total") and isinstance(value, int):
                shown = node.get(key.removesuffix("_total"))
                if isinstance(shown, list | dict) and value < len(shown):
                    problems.append(f"{where} is {value}, below the {len(shown)} shown")
            problems += capping_problems(value, where, set())
    return problems


# One step of a path: a field after a dot or at the start, or an index, a
# column's name or a mapping key in brackets, written as repr() writes it.
QUOTED = r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*\""""
PATH_STEP = re.compile(
    r"""\.?(?P<field>[A-Za-z_][A-Za-z0-9_]*)"""
    rf"""|\[(?P<key>(?:{QUOTED})(?:, (?:{QUOTED}))*|-?[0-9][0-9.e+-]*)\]"""
)


def path_problems(res: Any) -> list[str]:
    """The field paths of findings and limitations that don't resolve."""
    problems = []
    for kind in ("findings", "limitations"):
        for item in getattr(res, kind):
            problems += [
                f"{kind} {item.code}: {path}"
                for path in item.fields
                if not resolves_on(res, path)
            ]
    return problems


def resolves_on(res: Any, path: str) -> bool:
    """Whether a path such as "columns['age'].missing_count" finds a field of res."""
    current = res
    position = 0
    while position < len(path):
        step = PATH_STEP.match(path, position)
        if step is None:
            return False
        position = step.end()
        found = step_into(current, step["field"], step["key"])
        if found is MISSING:
            return False
        current = found
    return position > 0


MISSING = object()


def step_into(current: Any, field: str | None, key: str | None) -> Any:
    """What a field or a bracketed key of a path finds in `current`, or MISSING."""
    if field is not None:
        if isinstance(current, dict):
            return current.get(field, MISSING)
        return (
            getattr(current, field) if field in type(current).model_fields else MISSING
        )
    value = ast.literal_eval(key or "")
    if isinstance(current, dict):
        return current.get(value, MISSING)
    if isinstance(value, tuple):
        # An association pair is read by its two columns.
        pairs = [
            entry
            for entry in current
            if (getattr(entry, "x_column", None), getattr(entry, "y_column", None))
            == value
        ]
        return pairs[0] if len(pairs) == 1 else MISSING
    if isinstance(value, int) and not isinstance(value, bool):
        return current[value] if 0 <= value < len(current) else MISSING
    # A list of one entry per column is read by the column's name.
    named = [entry for entry in current if getattr(entry, "column", None) == value]
    return named[0] if len(named) == 1 else MISSING


def assessment_problems(name: str, res: Any) -> list[str]:
    """Each finding whose code or assessment isn't the one the docstring gives."""
    documented = {
        code.code: (code.description or "").partition("; ")[0]
        for code in ae.describe_analysis(name).finding_codes
    }
    problems = []
    for finding in res.findings:
        if finding.code not in documented:
            problems.append(f"{finding.code} isn't a declared finding code")
        elif finding.assessment != documented[finding.code]:
            problems.append(
                f"{finding.code} is {finding.assessment}, the docstring says "
                f"{documented[finding.code]}"
            )
    return problems


# Words in a field's name that say it holds a moment or how long a run took.
TIME_WORDS = {
    "timestamp",
    "time",
    "created",
    "generated",
    "started",
    "finished",
    "duration",
    "elapsed",
    "runtime",
    "seconds",
    "at",
}
VERSION = re.compile(r"[0-9][0-9A-Za-z.+!_-]*")


def time_and_html_problems(res: Any, record: dict[str, Any]) -> list[str]:
    """Timestamps, durations and HTML, which a result never holds."""
    problems = []
    for model in models_of(type(res)):
        for field_name, field in model.model_fields.items():
            if TIME_WORDS & set(field_name.lower().split("_")):
                problems.append(f"{model.__name__}.{field_name} reads as a time")
            if any(
                isinstance(part, type) and issubclass(part, datetime_types)
                for part in annotation_parts(field.annotation)
            ):
                problems.append(f"{model.__name__}.{field_name} is a time type")
    problems += [
        f"versions of {library!r} is {version!r}"
        for library, version in record["execution"]["versions"].items()
        if not VERSION.fullmatch(version)
    ]
    shown, _ = DisplayFormatter().format(res)
    if shown != {"text/plain": str(res)}:
        problems.append(f"it displays as {sorted(shown)}, not as plain text alone")
    for display in ("_repr_html_", "_repr_mimebundle_", "to_html"):
        if hasattr(res, display):
            problems.append(f"it has {display}")
    return problems


datetime_types = (dt.date, dt.time, dt.timedelta, pd.Timestamp, pd.Timedelta)


def models_of(model: type[pydantic.BaseModel]) -> list[type[pydantic.BaseModel]]:
    """Every pydantic model class reachable from the fields of `model`."""
    found = [model]
    for candidate in found:
        for field in candidate.model_fields.values():
            for part in annotation_parts(field.annotation):
                if (
                    isinstance(part, type)
                    and issubclass(part, pydantic.BaseModel)
                    and part not in found
                ):
                    found.append(part)
    return found


def annotation_parts(annotation: object) -> list[object]:
    """An annotation and every type inside it."""
    parts = [annotation]
    for argument in typing.get_args(annotation):
        parts += annotation_parts(argument)
    return parts
