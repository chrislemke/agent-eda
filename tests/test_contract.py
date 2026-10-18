"""Every registered analysis meets the docstring and registration contract."""

import doctest
import inspect
import re
import typing

import annotated_types
import pydantic
import pytest
from pydantic.fields import FieldInfo

import agent_eda as ae

ANALYSES = [info.name for info in ae.list_analyses().analyses]
INFO = {info.name: info for info in ae.list_analyses().analyses}
SECTIONS = ["When to use:", "Args:", "Returns:", "Statuses:", "Examples:", "See Also:"]
# Parameters that take text other than column names. A new one joins this list
# only if it takes no column names; otherwise its name ends in column(s).
NOT_COLUMN_NAMES = {"source", "source_version"}
ENTRY = re.compile(r"^    ([A-Za-z_][A-Za-z0-9_]*)(?: \([^)]*\))?:(?: |$)")
REFERENCE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`")
SNAKE_CASE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


@pytest.fixture(params=ANALYSES)
def name(request):
    return request.param


def docstring_lines(name):
    doc = inspect.getdoc(getattr(ae, name))
    assert doc, f"{name} has no docstring"
    return doc.splitlines()


def sections(name):
    """The docstring split into its summary line and its sections' bodies."""
    lines = docstring_lines(name)
    found = {"summary": lines[0]}
    current = None
    for line in lines[1:]:
        if line in SECTIONS:
            current = line
            found[current] = ""
        elif current is not None:
            found[current] += line + "\n"
    return found


def entries(body):
    return [m.group(1) for line in body.splitlines() if (m := ENTRY.match(line))]


def test_summary_line_is_the_question_the_analysis_answers(name):
    summary = sections(name)["summary"]

    assert summary.endswith("?")
    assert summary == INFO[name].summary


def test_the_six_sections_follow_the_summary_in_order(name):
    lines = docstring_lines(name)

    headers = [line for line in lines if line in SECTIONS]
    assert headers == SECTIONS
    assert lines[1] == ""


def test_args_document_every_parameter_and_nothing_else(name):
    documented = entries(sections(name)["Args:"])

    assert documented == list(inspect.signature(getattr(ae, name)).parameters)


def test_returns_names_every_declared_finding_code(name):
    body = sections(name)["Returns:"]

    missing = [code for code in INFO[name].finding_codes if code not in body]
    assert missing == []


def test_statuses_list_exactly_the_declared_statuses(name):
    assert entries(sections(name)["Statuses:"]) == INFO[name].statuses


def test_see_also_and_when_to_use_name_only_what_exists(name):
    found = sections(name)
    text = found["When to use:"] + found["See Also:"]
    see_also = entries(found["See Also:"])
    references = REFERENCE.findall(text)
    # Unquoted names count too, unless they name the analysis's own parameters,
    # result fields, finding codes, statuses or limitation codes.
    words = [word for word in SNAKE_CASE.findall(text) if word not in own_names(name)]

    assert see_also
    unknown = [ref for ref in see_also + references + words if not resolves(ref)]
    assert unknown == []


def test_every_result_field_has_a_description(name):
    result_class = getattr(ae.results, INFO[name].result_class)

    undocumented = [
        f"{model.__name__}.{field_name}"
        for model in models_in(result_class)
        for field_name, field in model.model_fields.items()
        if not (field.description or "").strip()
    ]
    assert undocumented == []


def test_describe_analysis_describes_every_parameter_status_and_finding(name):
    description = ae.describe_analysis(name)

    assert description.name == name
    assert [p.name for p in description.parameters] == INFO[name].parameters
    assert [p.name for p in description.parameters if not p.description] == []
    assert [s.status for s in description.statuses if not s.reason] == []
    # Returns: defines each finding code in a bullet "- `code`: what it means".
    assert [f.code for f in description.finding_codes if not f.description] == []
    assert description.when_to_use
    assert description.example


def test_describe_analysis_shows_every_declared_choice_and_range(name):
    hints = typing.get_type_hints(getattr(ae, name), include_extras=True)
    described = {p.name: p for p in ae.describe_analysis(name).parameters}

    missing = []
    for parameter, annotation in hints.items():
        if parameter == "return":
            continue
        has_choices, has_range = declares(annotation)
        if has_choices and not described[parameter].allowed_values:
            missing.append(f"{parameter}: allowed_values")
        if has_range and not described[parameter].valid_range:
            missing.append(f"{parameter}: valid_range")
    assert missing == []


def test_describe_analysis_gives_every_result_field_a_description(name):
    description = ae.describe_analysis(name)

    fields = [*description.result_fields, *description.shared_result_fields]
    assert description.result_fields
    assert [path for path, text in field_texts(fields) if not text.strip()] == []


def test_searching_for_its_name_finds_it_first(name):
    assert ae.search_analyses(name).matches[0].name == name


def test_examples_run_and_print_a_result(name):
    func = getattr(ae, name)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    report = []
    tests = doctest.DocTestFinder(recurse=False).find(func, name=name, globs={})
    for test in tests:
        runner.run(test, out=report.append)

    examples = [example for test in tests for example in test.examples]
    assert runner.failures == 0, "".join(report)
    assert any(
        "print(" in example.source and example.want.startswith(f"{name}: ")
        for example in examples
    ), "the examples must print a result"


def test_parameters_that_take_names_end_in_column_or_columns(name):
    parameters = list(inspect.signature(getattr(ae, name)).parameters.values())[1:]

    textual = [p.name for p in parameters if takes_text(p.annotation)]
    misnamed = [
        p
        for p in textual
        if not p.endswith(("column", "columns")) and p not in NOT_COLUMN_NAMES
    ]
    assert misnamed == []


def test_work_limits_and_budget_exceeded_come_together(name):
    info = INFO[name]
    described = {p.name: p for p in ae.describe_analysis(name).parameters}

    assert ("budget_exceeded" in info.statuses) == bool(info.work_limit_parameters)
    for parameter in info.work_limit_parameters:
        # An int with a default and a lower bound, such as 1 <= max_pairs.
        assert described[parameter].type == "int"
        assert not described[parameter].required
        assert described[parameter].valid_range


def test_analysis_is_hidden_from_pytest(name):
    assert getattr(ae, name).__test__ is False


def test_result_class_is_named_after_the_analysis(name):
    camel = "".join(part.title() for part in name.split("_"))

    assert INFO[name].result_class == f"{camel}Result"
    assert issubclass(getattr(ae.results, INFO[name].result_class), ae.AnalysisResult)


RANGES = (
    annotated_types.Ge,
    annotated_types.Gt,
    annotated_types.Le,
    annotated_types.Lt,
    annotated_types.Interval,
)


def declares(annotation):
    """Whether an annotation, anywhere inside it, has fixed choices and a range."""
    has_choices = has_range = False
    pending = [annotation]
    while pending:
        current = pending.pop()
        if isinstance(current, typing.TypeAliasType):
            pending.append(current.__value__)
        elif typing.get_origin(current) is typing.Literal:
            has_choices = True
        elif typing.get_origin(current) is typing.Annotated:
            inner, *metadata = typing.get_args(current)
            pending.append(inner)
            for item in metadata:
                found = item.metadata if isinstance(item, FieldInfo) else [item]
                has_range = has_range or any(isinstance(m, RANGES) for m in found)
        else:
            pending.extend(typing.get_args(current))
    return has_choices, has_range


def field_texts(fields, prefix=""):
    """Each described result field's path and description, entries' fields too."""
    for field in fields:
        path = f"{prefix}{field.name}"
        yield path, field.description
        yield from field_texts(field.fields, f"{path}.")


def resolves(reference):
    target = ae
    for part in reference.removeprefix("ae.").split("."):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def own_names(name):
    """The words an analysis's docstring may use without them existing in ae."""
    info = INFO[name]
    result_class = getattr(ae.results, info.result_class)
    fields = {
        field for model in models_in(result_class) for field in model.model_fields
    }
    limitations = set(typing.get_args(ae.results.LimitationCode))
    return (
        set(info.parameters)
        | fields
        | set(info.finding_codes)
        | set(info.statuses)
        | limitations
    )


def models_in(model, seen=None):
    """Every pydantic model class reachable from a result class's fields."""
    seen = [] if seen is None else seen
    if model in seen:
        return seen
    seen.append(model)
    for field in model.model_fields.values():
        for candidate in nested_types(field.annotation):
            if isinstance(candidate, type) and issubclass(
                candidate, pydantic.BaseModel
            ):
                models_in(candidate, seen)
    return seen


def nested_types(annotation):
    yield annotation
    for arg in typing.get_args(annotation):
        yield from nested_types(arg)


def takes_text(annotation):
    """Whether an annotation accepts str, alone or in a container, not a Literal."""
    if typing.get_origin(annotation) is typing.Literal:
        return False
    return annotation is str or any(
        takes_text(arg) for arg in typing.get_args(annotation)
    )
