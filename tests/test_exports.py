"""Guards for the steps of adding an analysis that fail silently: registering and exporting it."""

import ast
import importlib
import inspect
import pkgutil
import types
from pathlib import Path

import pytest

import agent_eda as ae

INIT = Path(ae.__file__).read_text()
# The packages whose public functions are analyses.
ANALYSIS_PACKAGES = ["agent_eda.data", "agent_eda.statistics"]


def test_every_public_function_of_an_analysis_module_is_a_registered_analysis():
    registered = {info.name for info in ae.list_analyses().analyses}

    assert unregistered(analysis_modules(), registered) == []


def test_every_public_function_of_an_analysis_module_is_exported():
    assert unexported(analysis_modules(), set(ae.__all__)) == []


def test_the_export_lists_in_init_name_the_same_set():
    assert export_problems(INIT) == []


def test_an_unregistered_public_function_is_reported_with_the_fix():
    module = standin_module("count_rows")

    problems = unregistered([module], registered={"describe_dataset"})

    assert len(problems) == 1
    assert "count_rows" in problems[0]
    assert "agent_eda.data.counting" in problems[0]
    assert "@analysis(" in problems[0]
    assert "leading underscore" in problems[0]
    assert unregistered([module], registered={"count_rows"}) == []


def test_a_private_function_is_not_reported():
    module = standin_module("_count_rows")

    assert unregistered([module], registered=set()) == []
    assert unexported([module], exported=set()) == []


def test_a_function_a_module_only_imports_is_not_reported():
    module = types.ModuleType("agent_eda.data.counting")
    module.guide = ae.guide  # ty: ignore[unresolved-attribute]  # a stand-in module

    assert unregistered([module], registered=set()) == []


def test_an_analysis_missing_from_the_top_level_exports_is_reported_with_the_fix():
    [problem] = unexported([standin_module("count_rows")], exported=set())

    assert "count_rows" in problem
    assert "__all__" in problem
    assert "_EXPORTS" in problem
    assert "TYPE_CHECKING" in problem


@pytest.mark.parametrize(
    ("old", "list_name"),
    [
        ('    "infer_column_semantics",\n    "list_analyses",\n', "__all__"),
        ('    "infer_column_semantics": "agent_eda.data.validation",\n', "_EXPORTS"),
        ("        infer_column_semantics,\n", "TYPE_CHECKING"),
    ],
    ids=["__all__", "_EXPORTS", "TYPE_CHECKING"],
)
def test_an_export_missing_from_one_list_is_reported_with_the_list_to_fix(
    old, list_name
):
    assert old in INIT
    new = '    "list_analyses",\n' if list_name == "__all__" else ""

    problems = export_problems(INIT.replace(old, new, 1))

    assert len(problems) == 1
    assert "infer_column_semantics" in problems[0]
    assert f"missing from {list_name}" in problems[0]


def standin_module(function_name):
    """A module under agent_eda.data that defines one function."""
    module = types.ModuleType("agent_eda.data.counting")

    def function(data):
        return len(data)

    function.__name__ = function_name
    function.__module__ = module.__name__
    setattr(module, function_name, function)
    return module


def analysis_modules():
    """Every public module of the analysis packages, imported."""
    modules = []
    for package_name in ANALYSIS_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            if not info.name.startswith("_"):
                modules.append(importlib.import_module(f"{package_name}.{info.name}"))
    assert modules, f"found no modules under {ANALYSIS_PACKAGES}"
    return modules


def public_functions(modules):
    """(module name, function name) of each public function a module defines."""
    return [
        (module.__name__, name)
        for module in modules
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        and value.__module__ == module.__name__
        and not name.startswith("_")
    ]


def unregistered(modules, registered):
    return [
        f"{module}.{name} is not a registered analysis: add @analysis(...) to it, "
        "or rename it with a leading underscore if it is a helper"
        for module, name in public_functions(modules)
        if name not in registered
    ]


def unexported(modules, exported):
    return [
        f"{module}.{name} is not exported from agent_eda: add {name} to __all__, "
        "_EXPORTS and the TYPE_CHECKING imports in src/agent_eda/__init__.py, "
        "or rename it with a leading underscore if it is a helper"
        for module, name in public_functions(modules)
        if name not in exported
    ]


def export_problems(source):
    """What differs between the three lists in agent_eda/__init__.py, as messages."""
    tree = ast.parse(source)
    lists = {
        "__all__": literal_names(tree, "__all__"),
        "_EXPORTS": literal_names(tree, "_EXPORTS"),
        "TYPE_CHECKING": type_checking_names(tree) - literal_names(tree, "_SUBMODULES"),
    }
    everything = set().union(*lists.values())
    return [
        f"{name} is missing from {', '.join(missing)} in src/agent_eda/__init__.py: "
        "each export is listed in __all__, _EXPORTS and the TYPE_CHECKING imports"
        for name in sorted(everything)
        if (missing := [label for label, names in lists.items() if name not in names])
    ]


def literal_names(tree, target):
    """The strings of a module-level list or tuple, or the keys of a dict."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == target for t in targets):
            value = node.value
            if isinstance(value, ast.Dict):
                return {k.value for k in value.keys if isinstance(k, ast.Constant)}
            if isinstance(value, ast.List | ast.Tuple):
                return {e.value for e in value.elts if isinstance(e, ast.Constant)}
    raise AssertionError(f"src/agent_eda/__init__.py has no literal {target}")


def type_checking_names(tree):
    """The names that `if TYPE_CHECKING:` imports, as the module binds them."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    names |= {a.asname or a.name for a in statement.names}
    assert names, "src/agent_eda/__init__.py imports nothing under TYPE_CHECKING"
    return names
