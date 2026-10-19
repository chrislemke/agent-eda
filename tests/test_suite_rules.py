"""The rules every test in this suite follows, checked over the test files."""

import ast
import re
from pathlib import Path

import pydantic

import agent_eda as ae
import agent_eda.results

# Test modules and the helpers they share, such as the recipes.
TESTS = sorted(Path(__file__).parent.glob("*.py"))


def test_tests_import_only_public_agent_eda_modules():
    private = [
        f"{path.name}: {module}"
        for path in TESTS
        for module in imported_modules(path)
        if module.split(".")[0] == "agent_eda"
        and any(part.startswith("_") for part in module.split("."))
    ]

    assert private == []


def test_test_function_names_never_reuse_an_analysis_name():
    analyses = {info.name for info in ae.list_analyses().analyses}

    clashes = [
        f"{path.name}: {node.name}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name in analyses
    ]

    assert clashes == []


def test_text_files_are_read_and_written_as_utf_8():
    # Without encoding=, Python uses the locale's: cp1252 on Windows, not UTF-8.
    found = [
        f"{path.name}:{node.lineno}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and is_text_file_call(node)
        and not any(keyword.arg == "encoding" for keyword in node.keywords)
    ]

    assert found == [], 'Pass encoding="utf-8" to these calls.'


def is_text_file_call(call):
    """Whether a call reads or writes a file as text: read_text, write_text or open."""
    name = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    if name in ("read_text", "write_text"):
        return True
    if name != "open" and getattr(call.func, "id", "") != "open":
        return False
    # open(file, mode) takes the mode second, Path.open(mode) first.
    mode_args = call.args[:1] if name == "open" else call.args[1:2]
    modes = [
        arg.value
        for arg in [*mode_args, *(k.value for k in call.keywords if k.arg == "mode")]
        if isinstance(arg, ast.Constant)
    ]
    return not any("b" in mode for mode in modes)


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


# Words GLOSSARY.md avoids: Assessment avoids "verdict", Status avoids "outcome"
# and Analysis result avoids "output". The "verdict" in _conversions names the group
# a value falls into (compatible, convertible or invalid), not an assessment.
AVOIDED_WORDS = re.compile(r"\b(verdicts?|outcomes?|outputs?)\b", flags=re.IGNORECASE)
PACKAGE = Path(ae.__file__).parent
# "server" without "MCP " before it, which GLOSSARY.md asks to avoid.
# A dotted name such as mcp.server.Server is code, not wording.
BARE_SERVER = re.compile(r"(?<!MCP )(?<!\.)\bserver\b", flags=re.IGNORECASE)
# Names that other libraries give: an MCP tool's output schema is JSON Schema's
# word, and subprocess.run takes capture_output.
EXTERNAL_NAMES = ("output_schema", "capture_output")


def test_no_name_uses_a_word_the_glossary_avoids():
    found = [
        f"{path.name}: {name}"
        for path in [*sorted(PACKAGE.rglob("*.py")), *TESTS]
        for name in identifiers(path)
        if AVOIDED_WORDS.search(name.replace("_", " "))
        and not any(external in name for external in EXTERNAL_NAMES)
        and not (path.name == "_conversions.py" and "verdict" in name)
    ]

    assert found == [], "Rename these: GLOSSARY.md says which word to use."


def test_no_result_field_description_uses_a_word_the_glossary_avoids():
    found = [
        f"{cls.__name__}.{name}: {field.description}"
        for cls in vars(ae.results).values()
        if isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)
        for name, field in cls.model_fields.items()
        if AVOIDED_WORDS.search((field.description or "").replace("output schema", ""))
    ]

    assert found == []


def test_test_names_say_the_mcp_server():
    found = [
        f"{path.name}: {node.name}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and BARE_SERVER.search(node.name.replace("_", " "))
    ]

    assert found == [], "Write mcp_server, as GLOSSARY.md does."


def test_package_texts_say_the_mcp_server():
    found = [
        f"{path.name}: ...{flat[max(match.start() - 30, 0) : match.end() + 30]}..."
        for path in sorted(PACKAGE.rglob("*.py"))
        for text in strings(path)
        for flat in [" ".join(text.split())]
        # A one-word string is an annotation such as "Server", a class name.
        for match in BARE_SERVER.finditer(flat if " " in flat else "")
    ]

    assert found == [], "Write 'MCP server', as GLOSSARY.md does."


def identifiers(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg


def strings(path):
    """The text constants of a module: docstrings, messages and the like."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
