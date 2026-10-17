"""Every registered analysis meets the docstring and registration contract."""

import doctest
import inspect
import re
import typing

import pydantic
import pytest

import agent_eda as ae

ANALYSES = [info.name for info in ae.list_analyses().analyses]
INFO = {info.name: info for info in ae.list_analyses().analyses}
SECTIONS = ["When to use:", "Args:", "Returns:", "Statuses:", "Examples:", "See Also:"]
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


def test_analysis_is_hidden_from_pytest(name):
    assert getattr(ae, name).__test__ is False


def test_result_class_is_named_after_the_analysis(name):
    camel = "".join(part.title() for part in name.split("_"))

    assert INFO[name].result_class == f"{camel}Result"
    assert issubclass(getattr(ae.results, INFO[name].result_class), ae.AnalysisResult)


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
