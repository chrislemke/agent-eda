"""No Python in what an MCP caller reads in a record: field descriptions, reasons, claims.

An MCP caller reads the record (the structured content) and its output schema, and
passes JSON arguments, so a status reason, a limitation reason, a finding claim or a
field description names an argument and its fix in words, with no code. The printed
text and the error messages keep their surface-specific fixes.

The guards scan the output schema of every analysis and of the two discovery
listings an MCP tool serves, and the status reason, limitations and findings of the
record of every analysis (see tests/recipes.py) on the fixed and the generated
DataFrames the property tests use.
"""

import re

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from python_syntax import CALL, KEYWORD, NAME, python_in, strings_in
from recipes import RECIPES, recipe_for
from strategies import EDGE_FRAMES, LABEL_EDGE_FRAMES, data_frames

import agent_eda as ae
from agent_eda.errors import AgentEDAError
from agent_eda.results import AnalysisListing, SearchListing

ANALYSES = [info.name for info in ae.list_analyses().analyses]
# A quoted text is a label or value of the data, such as a column named 'f(x)'.
QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")


# Without None, True and False, which a claim may quote as labels of the data.
WITHOUT_CONSTANTS = re.compile("|".join(p.pattern for p in (CALL, KEYWORD, NAME)))


def python_outside_quotes(text: str, constants: bool) -> list[str]:
    text = QUOTED.sub("''", text)
    if constants:
        return python_in(text)
    return [m[0] for m in WITHOUT_CONSTANTS.finditer(text)]


def python_in_prose(record: dict, constants: bool = True) -> list[tuple[str, str]]:
    """The Python in the texts of a record that name a fix, make a claim or define
    a field: the status reason, limitations, findings, definitions and any other
    text at the top level, but not what is derived from the data's values.

    Over generated data a claim may quote a label such as True or None, which looks
    like a constant, so those calls pass constants=False.
    """
    texts = [
        text
        for key in ("status_reason", "limitations", "findings", "definitions")
        for text in strings_in(record.get(key), key)
    ]
    texts += [
        (key, value)
        for key, value in record.items()
        if isinstance(value, str) and key not in ("analysis", "status")
    ]
    return [
        (path, found)
        for path, text in texts
        for found in python_outside_quotes(text, constants)
    ]


@pytest.mark.parametrize("name", ANALYSES)
def test_no_field_description_in_an_output_schema_holds_python_syntax(name):
    result_class = getattr(ae.results, ae.describe_analysis(name).result_class)
    schema = result_class.record_schema()

    leaks = [(p, f) for p, text in strings_in(schema) for f in python_in(text)]

    assert leaks == []


@pytest.mark.parametrize("listing", [SearchListing, AnalysisListing])
def test_no_field_description_in_a_discovery_output_schema_holds_python_syntax(
    listing,
):
    schema = listing.model_json_schema()

    leaks = [(p, f) for p, text in strings_in(schema) for f in python_in(text)]

    assert leaks == []


def edge_cases():
    cases = []
    for name in ANALYSES:
        any_labels = RECIPES[name].any_column_labels
        for frame in [*EDGE_FRAMES, *(LABEL_EDGE_FRAMES if any_labels else [])]:
            cases.append(pytest.param(name, frame, id=f"{name}-{frame}"))
    return cases


@pytest.mark.parametrize(("name", "frame"), edge_cases())
def test_no_status_reason_limitation_or_claim_holds_python_syntax_on_edge_cases(
    name, frame
):
    df = (EDGE_FRAMES | LABEL_EDGE_FRAMES)[frame]()
    arguments = RECIPES[name].required(df)
    if arguments is None:
        pytest.skip(f"{frame!r} allows no valid call of {name}")

    record = getattr(ae, name)(df, **arguments).to_dict()

    assert python_in_prose(record) == []


@pytest.mark.parametrize("name", ANALYSES)
@given(data=st.data())
def test_no_status_reason_limitation_or_claim_holds_python_syntax_on_generated_data(
    name, data
):
    recipe = recipe_for(name)
    df = data.draw(data_frames(any_column_labels=recipe.any_column_labels), "data")
    arguments = recipe.required(df)
    assume(arguments is not None)  # the data allows no valid call
    arguments |= data.draw(recipe.options(df), "arguments")

    try:
        record = getattr(ae, name)(df, **arguments).to_dict()
    except AgentEDAError:
        assume(False)  # the drawn arguments don't suit the drawn data
    assert python_in_prose(record, constants=False) == []


def test_the_guard_finds_python_in_a_record_and_not_in_a_quoted_label():
    leaking = {"limitations": [{"reason": "Convert it with df['f'].astype(str)."}]}
    quoted = {"limitations": [{"reason": "Column 'f(x)' holds 'None' and \"a=b\"."}]}

    assert python_in_prose(leaking)
    assert python_in_prose(quoted) == []
