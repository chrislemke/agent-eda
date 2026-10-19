"""Where each module of the package sits, and which modules may reach into which.

A module's location says its scope: a shared helper at the package root, a
single-analysis engine beside its analysis, the MCP adapter in `_mcp`. These tests
read the package's source, as test_packaging does, so they follow the files.
"""

import ast
from pathlib import Path

import pytest

import agent_eda as ae

PACKAGE = Path(ae.__file__).parent


def pkg(text: str) -> str:
    """The text with PKG, the package's name, written out: a test that names
    private modules in strings spells the package PKG, so the rule that tests
    reach only public names doesn't take the strings for uses."""
    return text.replace("PKG", "agent_eda")


# The modules that are an internal interface: other modules may import their
# underscore names, but only the importers listed for them.
# - The result base holds the shared parts of every result class (the table
#   check, the caps, the frame dtypes), which the result module beside each
#   analysis builds on.
# - The package root's export table is how the registry finds each analysis's
#   module.
INTERNAL_INTERFACES = {
    pkg("PKG._result_base"): lambda importer: importer.endswith("_results"),
    "agent_eda": lambda importer: importer == pkg("PKG._registry"),
}


def module_names(root: Path = PACKAGE) -> dict[str, Path]:
    """Every module of the package by dotted name, a package by its own name."""
    names = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names[".".join(parts)] = path
    return names


def is_underscore(name: str) -> bool:
    """A name that starts with one underscore: private, and not a dunder name."""
    return name.startswith("_") and not name.startswith("__")


def underscore_uses(source: str, importer: str, modules: set[str]) -> list[str]:
    """The underscore names a module takes from another module, as "module.name".

    That is a name imported with `from module import _name`, or read as an
    attribute of an imported module, `module._name`. Importing a private module
    itself, such as `from the package import _text`, is no such use.
    """
    tree = ast.parse(source)
    found = set()
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] != "agent_eda":
                continue
            for alias in node.names:
                if f"{node.module}.{alias.name}" in modules:
                    imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                elif is_underscore(alias.name) and node.module != importer:
                    found.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "agent_eda":
                    root = alias.name if alias.asname else "agent_eda"
                    imported[alias.asname or "agent_eda"] = root
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in imported
            and is_underscore(node.attr)
            and imported[node.value.id] != importer
        ):
            found.add(f"{imported[node.value.id]}.{node.attr}")
    return sorted(found)


def forbidden_uses(source: str, importer: str, modules: set[str]) -> list[str]:
    """The underscore names a module takes from another that isn't an internal
    interface open to it."""
    return [
        use
        for use in underscore_uses(source, importer, modules)
        if not any(
            use.rpartition(".")[0] == interface and allowed(importer)
            for interface, allowed in INTERNAL_INTERFACES.items()
        )
    ]


def test_no_module_imports_an_underscore_name_from_another_module():
    modules = module_names()
    found = [
        f"{importer} uses {use}"
        for importer, path in modules.items()
        for use in forbidden_uses(
            path.read_text(encoding="utf-8"), importer, set(modules)
        )
    ]

    assert found == [], (
        "A name that starts with an underscore belongs to its module: make it "
        "public, or move the code that needs it into that module."
    )


@pytest.mark.parametrize(
    ("source", "importer", "use"),
    [
        ("from PKG._registry import _REGISTRY", "PKG.x", "_registry._REGISTRY"),
        ("from PKG import _text\n_text._wrap()", "PKG.x", "_text._wrap"),
        ("import PKG\nPKG._EXPORTS", "PKG.data.x", "PKG._EXPORTS"),
        (
            "from PKG._result_base import _Cap",
            "PKG.data.structure",
            "_result_base._Cap",
        ),
        (
            "from PKG import _result_base\n_result_base._Cap",
            "PKG._registry",
            "_result_base._Cap",
        ),
    ],
)
def test_the_guard_catches_every_way_to_take_an_underscore_name(source, importer, use):
    found = forbidden_uses(pkg(source), pkg(importer), set(module_names()))

    assert any(item.endswith(pkg(use)) for item in found), found


@pytest.mark.parametrize(
    ("source", "importer"),
    [
        ("from PKG import _text", "PKG.x"),
        ("from PKG._text import wrap", "PKG.x"),
        ("from PKG._registry import _REGISTRY", "PKG._registry"),
        ("from PKG._result_base import _Cap", "PKG.data._structure_results"),
        ("import PKG\nPKG._EXPORTS", "PKG._registry"),
        ("from other import _name", "PKG.x"),
        ("from PKG._text import __doc__", "PKG.x"),
    ],
)
def test_the_guard_leaves_public_names_modules_and_internal_interfaces_alone(
    source, importer
):
    assert forbidden_uses(pkg(source), pkg(importer), set(module_names())) == []


def test_the_mcp_adapter_sits_in_its_own_package():
    root = {path.name for path in PACKAGE.glob("*.py")}
    adapter = {path.name for path in (PACKAGE / "_mcp").glob("*.py")}

    assert {"server.py", "instructions.py", "data_files.py"} <= adapter
    assert {"local_files.py", "table_files.py"} <= adapter
    assert not {name for name in root if name.startswith("_mcp")}
    assert not root & {"_data_files.py", "_local_files.py", "_table_files.py"}


def test_only_the_public_mcp_module_imports_the_mcp_adapter():
    modules = module_names()
    importers = [
        name
        for name, path in modules.items()
        if not name.startswith(pkg("PKG._mcp"))
        and pkg("PKG._mcp") in path.read_text(encoding="utf-8")
    ]

    assert importers == [pkg("PKG.mcp")]


def test_the_mcp_adapter_holds_no_module_level_dict_list_or_set():
    # Each MCP tool's schema is built anew: a fragment shared by every tool could
    # be changed through any of them.
    found = []
    for path in sorted((PACKAGE / "_mcp").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            value = getattr(node, "value", None)
            if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(
                value,
                ast.Dict
                | ast.List
                | ast.Set
                | ast.DictComp
                | ast.ListComp
                | ast.SetComp,
            ):
                found.append(f"{path.name}:{node.lineno}")

    assert found == []


def test_file_formats_are_told_apart_in_one_module_only():
    # The reader, the writer and the instructions ask the format what it is
    # (`is_parquet`, `separator`, `label`) rather than compare its kind.
    found = [
        path.name
        for path in sorted((PACKAGE / "_mcp").glob("*.py"))
        if path.name != "local_files.py" and ".kind" in path.read_text(encoding="utf-8")
    ]

    assert found == []


def test_single_analysis_engines_sit_beside_their_analysis():
    root = {path.name for path in PACKAGE.glob("*.py")}

    assert {"_keys.py", "_value_reading.py", "_sentinel_rules.py"} <= {
        path.name for path in (PACKAGE / "data").glob("*.py")
    }
    assert "_association.py" in {
        path.name for path in (PACKAGE / "statistics").glob("*.py")
    }
    assert not root & {
        "_association.py",
        "_keys.py",
        "_conversions.py",
        "_value_reading.py",
        "_sentinel_rules.py",
    }
    assert "_conversions.py" not in {
        path.name for path in (PACKAGE / "data").glob("*.py")
    }


def top_level_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }


def test_argument_handling_and_its_messages_sit_in_arguments_not_in_the_registry():
    arguments = top_level_names(PACKAGE / "_arguments.py")
    registry = top_level_names(PACKAGE / "_registry.py")

    helpers = {
        "invalid_message",
        "unknown_name_message",
        "missing_message",
        "value_text",
        "python_scalar",
        "check_distinct",
        "normalize_option",
        "checked_keywords",
    }
    assert helpers <= arguments
    assert not helpers & registry


def test_the_run_helpers_are_split_into_status_and_warnings_modules():
    assert not (PACKAGE / "_run.py").exists()
    assert {"overall_status", "budget_exceeded_reason"} <= top_level_names(
        PACKAGE / "_status.py"
    )
    assert {
        "capture_warnings",
        "watch_all_warnings",
        "watch_warnings",
        "absorb_warnings",
    } <= top_level_names(PACKAGE / "_warnings.py")


def test_declared_sentinel_values_have_a_module_that_says_so():
    assert (PACKAGE / "_declared_values.py").exists()
    assert not (PACKAGE / "_sentinels.py").exists()


def test_one_closest_name_helper_and_one_type_renderer():
    # difflib ranks the suggestions in _text only, and _annotations alone turns an
    # annotation into the text of its type.
    difflib_users = [
        name
        for name, path in module_names().items()
        if "difflib" in path.read_text(encoding="utf-8")
    ]
    renderers = [
        name
        for name, path in module_names().items()
        if "type_text" in top_level_names(path)
    ]

    assert difflib_users == [pkg("PKG._text")]
    assert renderers == [pkg("PKG._annotations")]


def test_one_engine_reads_the_values_of_a_column_for_both_type_analyses():
    engine = PACKAGE / "data" / "_value_reading.py"
    validation = (PACKAGE / "data" / "validation.py").read_text(encoding="utf-8")

    assert {"read_values", "ValueReading"} <= top_level_names(engine)
    # Neither analysis numbers, counts or parses a column's values itself.
    walkers = ("value_codes", "count_levels", "read_texts", "value_kind", "factorize")
    assert [name for name in walkers if name in validation] == []


def test_column_spec_sits_with_the_other_declarations():
    assert "ColumnSpec" in top_level_names(PACKAGE / "_declarations.py")
    assert "ColumnSpec" not in top_level_names(PACKAGE / "data" / "validation.py")


def test_column_spec_is_still_importable_from_its_old_places():
    from agent_eda.data.validation import ColumnSpec

    assert ae.ColumnSpec is ColumnSpec
