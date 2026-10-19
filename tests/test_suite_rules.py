"""The rules every test in this suite follows, checked over the test files."""

import ast
import io
import re
import tokenize
from pathlib import Path

import pydantic
import pytest

import agent_eda as ae
import agent_eda.results

# Test modules and the helpers they share, such as the recipes.
TESTS = sorted(Path(__file__).parent.glob("*.py"))


# A private module or file of the package written as text: a dotted module name
# or a path, such as agent_eda, a dot or slash and a name that starts with one
# underscore. Dunder names such as __init__ are public.
PRIVATE_TEXT = re.compile(r"agent_eda(?:[./]\w+)*?[./]_[A-Za-z]")
# Where a test may reach a private name of the package, and which ones: the one
# helper that serves stand-in analyses through the hook create_server keeps
# private, and lists them as registered ones through the hook of the registry,
# and the one that registers stand-in analyses with the registration decorator.
ALLOWED_PRIVATE = {
    "serving.py": {"_create_server", "_stand_ins"},
    "registering.py": {"_registry", "_REGISTRY"},
    # The renderer that words every mistake for both surfaces, and the two fixes
    # that other modules define, are tested as the private parts they are.
    "test_wording.py": {"_wording", "_arguments", "_declared_values"},
}


def private_uses(source, *, allowed=frozenset()):
    """How a test module reaches a private name of agent_eda: by an import, a
    private attribute of the package, its modules or anything imported from
    them, getattr with a private name, or a private module's name in a string,
    such as the argument of importlib.import_module."""
    tree = ast.parse(source)
    bound = package_names(tree)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.update(
                f"imports {module}"
                for module in imported_names(node)
                if module.split(".")[0] == "agent_eda"
                and any(is_private(part) for part in module.split("."))
            )
        elif isinstance(node, ast.Attribute):
            if is_private(node.attr) and root_of(node.value) in bound:
                found.add(f"reads the private attribute {node.attr}")
        elif isinstance(node, ast.Call) and (name := private_getattr(node, bound)):
            found.add(f"calls getattr with {name!r}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.update(
                f"writes {match.group(0)}... in a string"
                for match in [PRIVATE_TEXT.search(node.value)]
                if match
            )
    return sorted(use for use in found if not any(name in use for name in allowed))


def is_private(name):
    """A name that starts with one underscore: not a dunder name, not public."""
    return name.startswith("_") and not name.startswith("__")


def package_names(tree):
    """The names a module binds to agent_eda, its modules or what they define."""
    bound = {"agent_eda"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
                if alias.name.split(".")[0] == "agent_eda"
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "agent_eda"
        ):
            bound.update(alias.asname or alias.name for alias in node.names)
    return bound


def root_of(node):
    """The name at the start of a chain of attributes and calls, or None."""
    while isinstance(node, ast.Attribute | ast.Call | ast.Subscript):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def private_getattr(call, bound):
    """The private name that `getattr(<something of agent_eda>, name)` reads, or None."""
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "getattr"
        and len(call.args) >= 2
        and root_of(call.args[0]) in bound
    ):
        return None
    name = call.args[1]
    if isinstance(name, ast.Constant) and isinstance(name.value, str):
        return name.value if is_private(name.value) else None
    return None


def test_tests_reach_only_public_names_of_agent_eda():
    private = [
        f"{path.name} {use}"
        for path in TESTS
        for use in private_uses(
            path.read_text(encoding="utf-8"),
            allowed=ALLOWED_PRIVATE.get(path.name, frozenset()),
        )
    ]

    assert private == [], "Tests use the public API only: import public modules."


@pytest.mark.parametrize(
    ("snippet", "use"),
    [
        ("import PKG._registry", "imports PKG._registry"),
        ("from PKG import _registry", "imports PKG._registry"),
        ("from PKG._text import wrap", "imports PKG._text"),
        ("import importlib\nimportlib.import_module('PKG._x')", "in a string"),
        ("__import__('PKG.data._x')", "in a string"),
        ("import PKG as ae\nae._registry", "private attribute _registry"),
        ("import PKG as ae\nae.results._Cap", "private attribute _Cap"),
        ("import PKG as ae\ngetattr(ae, '_registry')", "getattr with '_registry'"),
        ("from PKG import results\ngetattr(results, '_Cap')", "getattr with '_Cap'"),
        (
            "import subprocess\nsubprocess.run(['python', '-c', 'import PKG._x'])",
            "in a string",
        ),
        ("PATH = 'src/PKG/_registry.py'", "in a string"),
    ],
)
def test_the_guard_catches_every_way_to_reach_a_private_name(snippet, use):
    found = private_uses(snippet.replace("PKG", "agent_eda"))

    assert any(use.replace("PKG", "agent_eda") in found_use for found_use in found)


@pytest.mark.parametrize(
    "snippet",
    [
        "import PKG as ae\nae.__version__",
        "import PKG as ae\ngetattr(ae, 'summarize_numeric')",
        "from PKG import results\nresults.__name__",
        "PATH = 'src/PKG/__init__.py'",
        "text = 'a_private_name _x'\nvalue = other._x",
    ],
)
def test_the_guard_leaves_public_and_dunder_names_alone(snippet):
    assert private_uses(snippet.replace("PKG", "agent_eda")) == []


def test_test_function_names_never_reuse_an_analysis_name():
    analyses = {info.name for info in ae.list_analyses().analyses}

    clashes = [
        f"{path.name}: {node.name}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in analyses
    ]

    assert clashes == []


def test_every_approximate_comparison_states_its_tolerance():
    bare = [
        f"{path.name}:{node.lineno}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "approx"
        and not any(keyword.arg in ("rel", "abs") for keyword in node.keywords)
    ]

    assert bare == [], "Pass rel= or abs= to pytest.approx, or compare exactly."


# What a suppression comment names: ruff's `noqa` with its codes, or a type
# checker's `ignore` with its rules.
SUPPRESSION = re.compile(
    r"^(?:noqa(?::\s*[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)?"
    r"|(?:ty|type|pyright):\s*ignore(?:\[[^\]]*\])?)"
)
# A reason has at least this many words, so "see above" is none.
MIN_REASON_WORDS = 3
ROOT = Path(__file__).parent.parent


def unexplained_suppressions(source):
    """The line numbers of suppression comments without a reason on the same
    line: `# noqa: B009  # why`, `# noqa: B009 - why` or `# ty: ignore[rule]  # why`."""
    flagged = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        suppressed = False
        reason = []
        for segment in token.string.lstrip("#").split("#"):
            directive = SUPPRESSION.match(segment.strip())
            suppressed = suppressed or directive is not None
            rest = segment.strip()[directive.end() :] if directive else segment
            reason.extend(rest.strip(" -:").split())
        if suppressed and (
            len(reason) < MIN_REASON_WORDS or " ".join(reason[:2]) == "see above"
        ):
            flagged.append(token.start[0])
    return flagged


def test_every_noqa_and_ty_ignore_carries_its_reason_on_the_same_line():
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *TESTS,
    ]

    found = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in files
        for line in unexplained_suppressions(path.read_text(encoding="utf-8"))
    ]

    assert found == [], "Write why after the suppression, as `# noqa: B009  # why`."


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("x = f()  # noqa: B009", True),
        ("x = f()  # noqa", True),
        ("x = f()  # ty: ignore[invalid-argument-type]", True),
        ("x = f()  # ty: ignore[invalid-argument-type]  # see above", True),
        ("x = f()  # noqa: B009  # dtype is typed object", False),
        ("x = f()  # noqa: PD011 - a NumPy array, not a pandas attribute", False),
        ("x = f()  # noqa: B018  # ty: ignore[unresolved-attribute]  # the typo", True),
        ("x = f()  # ty: ignore[unknown-argument]  # the typo is the test", False),
        ("x = f()  # a note that mentions noqa in passing", False),
        ("x = 'no qa'  # an ordinary comment", False),
        ("x = '# noqa: B009'", False),
    ],
)
def test_the_suppression_check_flags_exactly_the_ones_without_a_reason(line, flagged):
    assert bool(unexplained_suppressions(line)) is flagged


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


def imported_names(node):
    """The modules an import statement names, and for `from m import n`, m.n."""
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif node.module:
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
# word, subprocess.run takes capture_output, and doctest.OutputChecker has
# check_output.
EXTERNAL_NAMES = ("output_schema", "capture_output", "check_output")


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
        for cls in (getattr(ae.results, public) for public in dir(ae.results))
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
