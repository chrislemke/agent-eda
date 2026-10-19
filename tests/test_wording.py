"""The renderer that words every caller mistake for the Python and the MCP surface.

A raise site states a mistake's message as data: texts and fixes. These
properties run the renderer itself over every fix type on both surfaces, so that
a fix can't be written in Python for an MCP caller. The mistakes each analysis
raises, and the messages they give, are held by test_argument_messages.py,
test_mcp_argument_messages.py and test_mcp_fixes.py.
"""

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_syntax import python_in

import agent_eda as ae
import agent_eda._wording as wording
from agent_eda._declared_values import Declared
from agent_eda._registry import ShownValue
from agent_eda._wording import (
    NONE,
    DeclareSensitive,
    Echo,
    Fix,
    LeaveOut,
    Listed,
    Of,
    OneOf,
    Pass,
    PathInto,
    PythonOnly,
    Say,
    SeeArguments,
    TypeOf,
    Words,
    message,
    mistake,
    render,
)
from agent_eda.errors import InvalidArgumentError

PACKAGE = Path(ae.__file__).parent

# Names, owners and text a message holds as words: no brackets, equals signs or
# dots, which python_in reads as Python.
WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)
SENTENCES = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ,;", min_size=0, max_size=30)
SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-10_000, 10_000),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    WORDS,
)
ARGUMENTS = st.dictionaries(WORDS, SCALARS, min_size=1, max_size=3)
COLUMNS = st.lists(WORDS, min_size=1, max_size=3, unique=True)


def pass_fix():
    return ARGUMENTS.map(lambda arguments: Pass(**arguments))


# One strategy per fix type that reads the same in the words of both surfaces.
FIXES = st.one_of(
    pass_fix(),
    WORDS.map(LeaveOut),
    st.builds(SeeArguments, WORDS, st.sampled_from(["argument", "field"])),
    st.builds(PythonOnly, SENTENCES),
    st.builds(DeclareSensitive, COLUMNS, of=st.sampled_from(["project", "policy"])),
    SCALARS.map(Echo),
    SCALARS.map(TypeOf),
    st.lists(st.one_of(WORDS, st.integers(0, 9)), max_size=3).map(PathInto),
    SCALARS.map(ShownValue),
    # A declared value is never None: that is a missing value, which no one declares.
    SCALARS.filter(lambda value: value is not None).map(Declared),
    st.just(NONE),
)


@given(arguments=ARGUMENTS)
def test_pass_writes_the_arguments_as_python_keywords_and_as_a_json_object(arguments):
    fix = Pass(**arguments)

    call = ast.parse(f"dict({render(fix, 'python')})", mode="eval").body
    assert isinstance(call, ast.Call)
    assert {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
    } == arguments
    assert json.loads(render(fix, "mcp")) == arguments


@given(name=WORDS)
def test_leave_out_passes_none_in_python_and_leaves_the_argument_out_over_mcp(name):
    fix = LeaveOut(name)

    assert render(fix, "python") == f"pass {name}=None"
    assert render(fix, "mcp") == f"leave {name} out"


@given(owner=WORDS, noun=st.sampled_from(["argument", "field"]))
def test_see_arguments_points_python_to_help_and_mcp_to_the_input_schema(owner, noun):
    fix = SeeArguments(owner, noun)

    assert f"help(ae.{owner})" in render(fix, "python")
    assert "help(" not in render(fix, "mcp")
    assert "input schema" in render(fix, "mcp")


@given(code=SENTENCES, otherwise=SENTENCES)
def test_python_only_is_dropped_over_mcp_or_replaced_by_its_alternative(
    code, otherwise
):
    assert render(PythonOnly(code), "python") == code
    assert render(PythonOnly(code), "mcp") == ""
    assert render(PythonOnly(code, otherwise=otherwise), "mcp") == otherwise


@given(columns=COLUMNS, of=st.sampled_from(["project", "policy"]))
def test_declare_sensitive_shows_code_in_python_and_toml_over_mcp(columns, of):
    fix = DeclareSensitive(columns, of=of)

    assert repr(columns) in render(fix, "python")
    assert "ae.DataPolicy" in render(fix, "python")
    assert tomllib.loads(render(fix, "mcp")) == {"sensitive_columns": columns}


@given(value=st.one_of(st.none(), st.booleans()))
def test_echo_writes_null_and_booleans_as_json_over_mcp(value):
    assert render(Echo(value), "python") == repr(value)
    assert json.loads(render(Echo(value), "mcp")) is value


@given(steps=st.lists(st.one_of(WORDS, st.integers(0, 9)), min_size=1, max_size=3))
def test_path_into_writes_subscripts_in_python_and_a_json_pointer_over_mcp(steps):
    python = render(PathInto(steps), "python")
    mcp = render(PathInto(steps), "mcp")

    subscripts = ast.parse(f"x{python}", mode="eval").body
    written = []
    while isinstance(subscripts, ast.Subscript):
        written.insert(0, ast.literal_eval(subscripts.slice))
        subscripts = subscripts.value
    assert written == steps
    assert "[" not in mcp
    assert mcp.split("/")[1:] == [str(step) for step in steps]


@given(fix=FIXES)
def test_no_fix_leaves_python_syntax_in_what_an_mcp_caller_reads(fix):
    assert python_in(render(fix, "mcp")) == []


@given(
    steps=st.lists(
        st.one_of(WORDS, st.builds(PythonOnly, WORDS)), min_size=0, max_size=4
    ),
    last=st.one_of(st.none(), WORDS),
    capital=st.booleans(),
)
def test_one_of_lists_the_steps_each_surface_can_take(steps, last, capital):
    fix = OneOf(steps, last, capital=capital)

    for surface in ("python", "mcp"):
        kept = [text for step in steps if (text := render(step, surface))]
        items = [*kept, *([last] if last else [])]
        written = render(fix, surface)
        assert written.lower() == _listed(items, has_last=bool(last)).lower()
        assert written[:1] == (written[:1].upper() if capital else written[:1])


def test_one_of_drops_a_step_that_only_python_can_take_over_mcp():
    fix = OneOf(
        [PythonOnly("df.reset_index()"), "use summarize_categorical"],
        last="leave the column out",
    )

    assert render(fix, "python") == (
        "df.reset_index(); use summarize_categorical; or leave the column out"
    )
    assert render(fix, "mcp") == "use summarize_categorical, or leave the column out"
    assert render(
        OneOf([PythonOnly("df.reset_index()")], last="leave it out"), "mcp"
    ) == ("leave it out")


def _listed(items, *, has_last):
    if len(items) <= 1:
        return "".join(items)
    if has_last and len(items) == 2:
        return f"{items[0]}, or {items[1]}"
    return "; ".join([*items[:-1], f"or {items[-1]}"])


@given(parts=st.lists(st.one_of(SENTENCES, FIXES), max_size=6))
def test_a_wording_reads_as_its_parts_in_order_on_each_surface(parts):
    for surface in ("python", "mcp"):
        expected = "".join(
            part if isinstance(part, str) else part.render(surface) for part in parts
        )
        assert render(parts, surface) == expected
        assert render(Say(*parts), surface) == expected
        assert render([[part] for part in parts], surface) == expected
        assert render(Words(parts if surface == "python" else "", parts), surface) == (
            expected
        )


@given(items=st.lists(SCALARS, max_size=8), limit=st.integers(1, 5))
def test_listed_names_at_most_the_limit_and_counts_the_rest(items, limit):
    for surface in ("python", "mcp"):
        written = render(Listed((Echo(item) for item in items), limit=limit), surface)
        if len(items) > limit:
            assert written.endswith(f" and {len(items) - limit:,} more")


@given(parts=st.lists(st.one_of(SENTENCES, FIXES), min_size=1, max_size=6))
def test_a_mistake_reads_in_python_as_its_message_and_over_mcp_as_the_fixes_for_mcp(
    parts,
):
    error = mistake(InvalidArgumentError, *parts, code="invalid_call")

    assert isinstance(error, InvalidArgumentError)
    assert error.code == "invalid_call"
    assert str(error) == render(parts, "python") == message(error, "python")
    assert message(error, "mcp") == render(parts, "mcp")
    assert render(Of(error), "mcp") == message(error, "mcp")
    assert render(Of(error), "python") == str(error)


def test_an_error_without_wording_reads_the_same_on_both_surfaces():
    error = InvalidArgumentError("a plain message")

    assert message(error, "python") == message(error, "mcp") == "a plain message"


def test_the_vocabulary_of_fixes_is_the_one_these_properties_cover():
    fixes = {
        name: value
        for name, value in vars(wording).items()
        if isinstance(value, type) and issubclass(value, Fix) and value is not Fix
    }

    # Every fix type of the vocabulary is exercised by FIXES above or by its own
    # property; a new one must join them.
    covered = {
        "Pass", "LeaveOut", "SeeArguments", "PythonOnly", "DeclareSensitive", "Echo",
        "TypeOf", "PathInto", "Words", "Say", "OneOf", "Listed", "Of",
    }  # fmt: skip
    assert set(fixes) == covered


SOURCES = {path: path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")}


@pytest.mark.parametrize(
    "gone",
    [
        r"\bworded\(",
        r"lambda surface",
        r"_check_target",
        r"_REGRESSION_FIXES",
        r"_MCP_REGRESSION_FIXES",
        r"_mcp_message",
    ],
)
def test_no_raise_site_words_a_message_for_the_surface_itself(gone):
    found = [
        str(path.relative_to(PACKAGE))
        for path, text in SOURCES.items()
        if re.search(gone, text)
    ]

    assert found == []
