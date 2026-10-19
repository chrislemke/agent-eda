"""Guard: the error codes in use, the ae.errors docstring and the README's errors table agree."""

import ast
import re
from pathlib import Path

import agent_eda as ae
from agent_eda.errors import AgentEDAError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(ae.__file__).parent
README = ROOT / "README.md"

# Lines such as: "    column_not_found (ColumnNotFoundError): a column ..."
DOCSTRING_ENTRY = re.compile(r"^    (\w+) \((\w+)", flags=re.MULTILINE)
# Rows such as: | `column_not_found` | `ColumnNotFoundError` | ... |
README_ROW = re.compile(r"^\| `(\w+)` \| `(\w+Error)` \|", flags=re.MULTILINE)


def test_every_error_code_is_documented_and_every_documented_code_is_in_use():
    in_use = codes_in_use()
    documented = DOCSTRING_ENTRY.findall(ae.errors.__doc__ or "")

    assert in_use
    assert documented
    assert (
        code_problems(
            in_use, {code for code, _ in documented}, "the ae.errors docstring"
        )
        == []
    )


def test_every_error_code_is_in_the_readme_errors_table_and_every_listed_code_is_in_use():
    in_use = codes_in_use()
    listed = README_ROW.findall(README.read_text(encoding="utf-8"))

    assert listed, (
        "README.md has no errors table with rows like | `code` | `Class` | ... |"
    )
    assert (
        code_problems(in_use, {code for code, _ in listed}, "the README's errors table")
        == []
    )


def test_a_documented_class_is_an_error_class_of_ae_errors():
    documented = DOCSTRING_ENTRY.findall(ae.errors.__doc__ or "")

    unknown = [
        (code, name)
        for code, name in documented
        if not issubclass(getattr(ae.errors, name, object), AgentEDAError)
    ]

    assert unknown == []


def test_a_class_code_missing_from_the_docs_is_reported_with_the_fix():
    in_use = {"column_not_found": "ColumnNotFoundError", "new_code": "NewError"}

    [problem] = code_problems(in_use, {"column_not_found"}, "the ae.errors docstring")

    assert "new_code" in problem
    assert "NewError" in problem
    assert "the ae.errors docstring" in problem
    assert "add" in problem


def test_a_documented_code_that_no_class_has_is_reported_with_the_fix():
    in_use = {"column_not_found": "ColumnNotFoundError"}

    [problem] = code_problems(
        in_use, {"column_not_found", "old_code"}, "the README's errors table"
    )

    assert "old_code" in problem
    assert "the README's errors table" in problem
    assert "remove" in problem


def test_codes_passed_when_raising_count_as_in_use():
    in_use = codes_in_use()

    # A code that InvalidArgumentError carries only when a raise site passes code=.
    assert in_use["invalid_call"] == "InvalidArgumentError"
    assert in_use["unknown_project_key"] == "ProjectFileError"
    # A class's own code.
    assert in_use["table_file_exists"] == "TableFileExistsError"
    # Finding and limitation codes are not error codes.
    assert "ambiguous_row_ids" not in in_use


def code_problems(in_use, listed, where):
    """Messages for the codes in use but not listed, and listed but not in use."""
    return [
        f"error code {code!r} ({in_use[code]}) is missing from {where}: add it"
        for code in sorted(set(in_use) - listed)
    ] + [
        f"{where} lists the error code {code!r}, which no AgentEDAError subclass has "
        "and no raise site passes as code=: remove it, or add the class"
        for code in sorted(listed - set(in_use))
    ]


def codes_in_use():
    """Each error code mapped to the class that carries it.

    A class carries its own `code`; a raise site can also pass code= to a
    class, as InvalidArgumentError(..., code="invalid_call") does, directly or
    through _wording.mistake.
    """
    classes = error_classes()
    found = {cls.code: name for name, cls in classes.items() if name != "AgentEDAError"}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            code = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "code"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ),
                None,
            )
            raised = raised_class(node, classes)
            if code is not None and raised is not None:
                found.setdefault(code, raised)
    return found


def raised_class(call, classes):
    """The error class a call constructs, directly or as mistake's first argument."""
    for candidate in [call.func, *call.args[:1]]:
        name = getattr(candidate, "id", None) or getattr(candidate, "attr", None)
        if name in classes:
            return name
    return None


def error_classes():
    """Every AgentEDAError subclass of agent_eda.errors, the base class included.

    Subclasses a test defines elsewhere don't count.
    """
    found = {}
    pending: list[type[AgentEDAError]] = [AgentEDAError]
    while pending:
        cls = pending.pop()
        if cls.__module__ == "agent_eda.errors":
            found[cls.__name__] = cls
        pending.extend(cls.__subclasses__())
    return found
