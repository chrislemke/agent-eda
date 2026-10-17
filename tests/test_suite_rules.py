"""The rules every test in this suite follows, checked over the test files."""

import ast
from pathlib import Path

import agent_eda as ae

TESTS = sorted(Path(__file__).parent.glob("test_*.py"))


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
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in analyses
    ]

    assert clashes == []


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)
