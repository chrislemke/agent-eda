"""Packaging checks, each in a fresh interpreter, as a sandboxed agent would run."""

import ast
import email
import gzip
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]

# ADR 0005 allows six required libraries. The package requires the ones its code
# imports, at the minimums Code Interpreter runs; one of the other two is added
# back when an analysis first imports it.
REQUIRED_DEPENDENCIES = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pydantic>=2.7",
    "scipy>=1.11.2",
]
MCP_EXTRA = ["mcp>=2.3 ; extra == 'mcp'", "pyarrow>=14.0.1 ; extra == 'mcp'"]
# What the MCP server imports, which only the mcp extra installs (anyio comes with mcp).
MCP_EXTRA_MODULES = {"mcp", "mcp_types", "anyio", "pyarrow"}
# Imported without being required: pydantic always installs the first two, and
# Rich, the third, is the caller of the one function that imports it.
COMES_WITH_THE_REQUIRED = {"pydantic_core", "annotated_types", "rich"}
REPOSITORY = "https://github.com/chrislemke/agent-eda"

# Fails the subprocess on a DeprecationWarning or FutureWarning, like the suite.
STRICT_WARNINGS = ("-W", "error::DeprecationWarning", "-W", "error::FutureWarning")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """The wheel file that AGENT_EDA_WHEEL names, or else one built here.

    CI's packaging job builds the wheel with `uv build` and names it, so these
    tests check the file the job then installs. A wheel built here comes from
    the backend that [build-system] declares: uv_build is a dev dependency
    pinned to that range, so the build needs no network and works whatever uv
    version runs the tests.
    """
    if os.environ.get("AGENT_EDA_WHEEL"):
        return Path(os.environ["AGENT_EDA_WHEEL"]).resolve()
    return build("build_wheel", ROOT, tmp_path_factory.mktemp("dist"))


@pytest.fixture(scope="module")
def stray_files_build(tmp_path_factory):
    """A wheel and a source distribution built in a working copy that holds the
    files Finder leaves in every folder it opens, which .gitignore hides from git
    but not from a build."""
    copy = tmp_path_factory.mktemp("working-copy")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, copy / name)
    shutil.copytree(
        ROOT / "src", copy / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    package = copy / "src" / "agent_eda"
    for directory in (package, package / "data"):
        (directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    out = tmp_path_factory.mktemp("stray-dist")
    return build("build_wheel", copy, out), build("build_sdist", copy, out)


def build(function, source, out):
    """Build with the uv_build of this environment, as the PEP 517 hook does."""
    bin_dir = str(Path(sys.executable).parent)  # the hook runs uv-build from PATH
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, uv_build; print(uv_build.{function}(sys.argv[1]))",
            str(out),
        ],
        cwd=source,
        env={**os.environ, "PATH": os.pathsep.join([bin_dir, os.environ["PATH"]])},
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    return out / built.stdout.strip().splitlines()[-1]


def wheel_metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return email.message_from_string(archive.read(name).decode())


def test_wheel_is_pure_python_typed_and_declares_the_four_dependencies(wheel):
    info = f"agent_eda-{ae.__version__}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        tags = archive.read(f"{info}/WHEEL").decode()
        metadata = archive.read(f"{info}/METADATA").decode()
        scripts = archive.read(f"{info}/entry_points.txt").decode()

    assert wheel.name == f"agent_eda-{ae.__version__}-py3-none-any.whl"
    assert "Root-Is-Purelib: true" in tags.splitlines()
    assert "Tag: py3-none-any" in tags.splitlines()
    assert "agent_eda/py.typed" in names
    assert not [n for n in names if n.endswith((".so", ".pyd", ".dylib", ".c"))]
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    assert sorted(r for r in requires if "extra ==" not in r) == REQUIRED_DEPENDENCIES
    # The MCP server's dependencies come only with the mcp extra (ADR 0006).
    assert sorted(r for r in requires if "extra ==" in r) == MCP_EXTRA
    assert "Provides-Extra: mcp" in metadata.splitlines()
    assert "agent-eda-mcp = agent_eda.mcp:main" in scripts.splitlines()


def test_the_wheel_metadata_gives_pypi_an_author_keywords_classifiers_and_urls(wheel):
    metadata = wheel_metadata(wheel)
    classifiers = metadata.get_all("Classifier")
    urls = dict(value.split(", ", 1) for value in metadata.get_all("Project-URL") or [])

    assert "Chris Lemke" in (metadata["Author"] or metadata["Author-email"])
    assert {"eda", "mcp"} <= set(metadata["Keywords"].split(","))
    assert classifiers == [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
    ]
    # The license is the PEP 639 expression, which a License classifier would repeat.
    assert metadata["License-Expression"] == "MIT"
    assert urls == {
        "Homepage": REPOSITORY,
        "Source": REPOSITORY,
        "Issues": f"{REPOSITORY}/issues",
        "Changelog": f"{REPOSITORY}/blob/main/CHANGELOG.md",
    }


def test_the_wheel_packs_only_python_files_and_the_py_typed_marker_in_the_package(
    wheel,
):
    with zipfile.ZipFile(wheel) as archive:
        packed = [
            n
            for n in archive.namelist()
            if n.startswith("agent_eda/") and not n.endswith("/")
        ]

    assert "agent_eda/__init__.py" in packed
    assert [
        n for n in packed if not n.endswith(".py") and n != "agent_eda/py.typed"
    ] == []


def test_files_that_a_folder_view_leaves_in_the_working_copy_stay_out_of_both_files(
    stray_files_build,
):
    wheel, sdist = stray_files_build
    with zipfile.ZipFile(wheel) as archive:
        in_wheel = archive.namelist()
    # GzipFile and TarFile read bytes; tarfile.open reads as well, but looks like
    # a text open to the suite's rule on encodings.
    with gzip.GzipFile(sdist) as packed, tarfile.TarFile(fileobj=packed) as archive:
        in_sdist = archive.getnames()

    # Each file holds the package, so an empty build can't pass for a clean one.
    assert "agent_eda/__init__.py" in in_wheel
    assert any(name.endswith("src/agent_eda/__init__.py") for name in in_sdist)
    assert [n for n in in_wheel + in_sdist if n.endswith(".DS_Store")] == []


def test_the_package_requires_the_libraries_its_code_imports_and_no_others():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = {
        re.split(r"[\s<>=!~;\[]", spec, maxsplit=1)[0]
        for spec in project["project"]["dependencies"]
    }
    adr = ROOT / "docs/adr/0005-pure-python-within-code-interpreter-dependencies.md"

    # numpy, pandas, pydantic and scipy are imported under their own names.
    assert (
        third_party_imports() - MCP_EXTRA_MODULES - COMES_WITH_THE_REQUIRED == required
    )
    # A new required dependency needs its ADR first, which the ADR's text says.
    assert [
        name for name in required if name not in adr.read_text(encoding="utf-8")
    ] == []
    assert sorted(project["project"]["dependencies"]) == REQUIRED_DEPENDENCIES


def test_wheel_on_sys_path_imports_and_runs_in_an_isolated_interpreter(wheel, tmp_path):
    code = f"""
import importlib.metadata
import sys
sys.path.insert(0, {str(wheel)!r})
import pandas as pd
import agent_eda as ae
assert ae.__file__.startswith({str(wheel)!r}), ae.__file__
located = str(importlib.metadata.distribution("agent-eda").locate_file(""))
assert located.startswith({str(wheel)!r}), located
print(ae.__version__)
print(ae.describe_dataset(pd.DataFrame({{"a": [1, 2]}})))
print(ae.list_analyses())
ae.guide()
print(ae.search_analyses("dataset overview"))
print(ae.describe_analysis("summarize_numeric"))
"""

    out = run_python(code, args=("-I", *STRICT_WARNINGS), cwd=tmp_path)

    lines = out.splitlines()
    assert lines[0] == ae.__version__
    assert lines[1] == "describe_dataset: completed on 2 rows x 1 column"
    assert "  describe_dataset: What does this dataset contain?" in lines
    assert "A first look at a new dataset, in this order:" in lines
    assert any(line.startswith("  describe_dataset: What does") for line in lines)
    assert "  trim_fraction: float = 0.1; 0 <= trim_fraction < 0.5" in lines


def test_a_zip_without_source_files_still_gives_every_code_its_meaning(tmp_path):
    # Only bytecode, so nothing can read the source of a module back.
    package = Path(ae.__file__).parent
    archive = tmp_path / "agent_eda_bytecode.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(package.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            name = path.relative_to(package.parent)
            if path.suffix == ".py":
                compiled = tmp_path / "compiled.pyc"
                py_compile.compile(str(path), cfile=str(compiled), doraise=True)
                bundle.write(compiled, str(name.with_suffix(".pyc")))
            else:
                bundle.write(path, str(name))
    code = f"""
import sys
sys.path.insert(0, {str(archive)!r})
import agent_eda as ae
assert ae.__file__.startswith({str(archive)!r}), ae.__file__
description = ae.describe_analysis("describe_dataset")
codes = [*description.limitation_codes]
for status in description.statuses:
    codes += status.reason_codes
assert codes and all(entry.meaning for entry in codes), codes
print(len(codes))
"""

    out = run_python(code, args=("-I",), cwd=tmp_path)

    assert int(out) > 10


# The steps an agent takes first, in order: import the package, then call each
# discovery function and print its text.
IMPORT_STEPS = {
    "import": "pass",
    "list": "print(ae.list_analyses())",
    "guide": "ae.guide()",
    "search": "print(ae.search_analyses('nulls per column').to_json())",
    "describe": (
        "[print(ae.describe_analysis(i.name).to_json()) "
        "for i in ae.list_analyses().analyses]"
    ),
}


@pytest.fixture(scope="module")
def imports():
    """What each step of IMPORT_STEPS newly loads or reaches, from one interpreter.

    Starting an interpreter and importing pandas costs most of a second, so the
    steps run in one, in order, and each step reports what appeared since the
    step before it.
    """
    code = f"""
import json
import sys
import agent_eda as ae
network = []
def record(event, args):
    # An audit hook records every attempt to open a socket or a URL. Reading this
    # machine's own name reaches no network: pandas' import does it on Windows,
    # where platform.machine() asks socket.gethostname for the node name.
    if event.split(".")[0] in ("socket", "urllib", "http"):
        if event != "socket.gethostname":
            network.append(event)
sys.addaudithook(record)
def is_mcp(name):
    # mcp, its submodules, mcp_types, and the package's own MCP modules.
    return any(part.lstrip("_").startswith("mcp") for part in name.split("."))
heavy = {{"scipy", "statsmodels", "sklearn"}}
seen = {{"heavy": set(), "mcp": set(), "network": set()}}
report = {{}}
for step, call in {IMPORT_STEPS!r}.items():
    exec(call)
    now = {{
        "heavy": {{n for n in sys.modules if n.split(".")[0] in heavy}},
        "mcp": {{n for n in sys.modules if is_mcp(n)}},
        "network": set(network),
    }}
    report[step] = {{kind: sorted(now[kind] - seen[kind]) for kind in now}}
    for kind in now:
        seen[kind] |= now[kind]
print("report:", json.dumps(report))
"""
    last = run_python(code, args=("-I", *STRICT_WARNINGS)).splitlines()[-1]
    return json.loads(last.removeprefix("report: "))


@pytest.mark.parametrize("step", list(IMPORT_STEPS))
def test_discovery_leaves_scipy_statsmodels_sklearn_and_the_network_alone(
    imports, step
):
    assert imports[step]["heavy"] == []
    assert imports[step]["network"] == []


@pytest.mark.parametrize("step", list(IMPORT_STEPS))
def test_the_package_and_discovery_import_no_mcp_module(imports, step):
    assert imports[step]["mcp"] == []


def test_identical_calls_give_identical_json_under_different_hash_seeds():
    code = """
import pandas as pd
import agent_eda as ae
df = pd.DataFrame(
    [["Oslo", "b", 1, None, "x", 1.5]] * 3 + [["Lyon", "a", 2, None, "y", 1.5]],
    columns=["city", "kind", 1, "notes", "1", "kind"],
    index=["r1", "r2", "r1", "r3"],
)
df["tier"] = pd.Categorical(["gold", "silver", "gold", "bronze"])
print(ae.describe_dataset(df, source="shop.csv").to_json())
"""

    printed = {
        run_python(code, env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ("1", "2", "3")
    }

    assert len(printed) == 1


def test_search_and_descriptions_are_identical_under_different_hash_seeds():
    code = """
import agent_eda as ae
for query in ["nulls per column, dupes and correlation", "label", "missing"]:
    print(ae.search_analyses(query, limit=100).to_json())
for info in ae.list_analyses().analyses:
    print(ae.describe_analysis(info.name).to_json())
"""

    printed = {
        run_python(code, env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ("1", "2", "3")
    }

    assert len(printed) == 1


# What a user may lack: all of the extra (a plain `pip install agent-eda`), or one
# part of it, such as mcp without pyarrow, which writes the table files.
MISSING_FROM_THE_EXTRA = [
    ("mcp", "anyio", "pyarrow"),
    ("mcp",),
    ("pyarrow",),
]


def run_the_mcp_command_without(missing, *arguments):
    """Start agent-eda-mcp as its generated script does, with these modules missing.

    Setting sys.modules[name] to None makes `import name` fail, as it does where
    the module isn't installed. The console script is loaded from the installed
    entry point.
    """
    code = f"""
import sys
import warnings
from importlib.metadata import entry_points
# pandas 2.2 warns that it found no pyarrow, which is what this run takes away.
warnings.filterwarnings(
    "ignore", message=r"\\s*Pyarrow will become a required", category=DeprecationWarning
)
for name in {missing!r}:
    sys.modules[name] = None
(script,) = entry_points(group="console_scripts", name="agent-eda-mcp")
assert script.value == "agent_eda.mcp:main", script.value
sys.argv = ["agent-eda-mcp", *{list(arguments)!r}]
sys.exit(script.load()())
"""
    return subprocess.run(
        [sys.executable, *STRICT_WARNINGS, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("missing", MISSING_FROM_THE_EXTRA, ids="+".join)
def test_the_mcp_server_without_a_module_of_the_extra_names_the_install_commands(
    missing,
):
    completed = run_the_mcp_command_without(missing)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "pip install 'agent-eda[mcp]'" in completed.stderr
    assert "uv add 'agent-eda[mcp]'" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_the_mcp_command_prints_its_help_without_the_extra():
    completed = run_the_mcp_command_without(MISSING_FROM_THE_EXTRA[0], "--help")

    assert completed.returncode == 0, completed.stderr
    assert "usage: agent-eda-mcp" in completed.stdout
    assert "--project" in completed.stdout
    assert completed.stderr == ""


def test_missing_dependency_error_has_a_stable_code_and_is_an_import_error():
    error = ae.errors.MissingDependencyError("pip install 'agent-eda[mcp]'")

    assert error.code == "missing_dependency"
    assert isinstance(error, ae.errors.AgentEDAError)
    assert isinstance(error, ImportError)


def test_package_code_never_reads_files_through_dunder_file():
    sources = sorted((ROOT / "src" / "agent_eda").rglob("*.py"))

    assert sources
    assert [
        p.name for p in sources if "__file__" in p.read_text(encoding="utf-8")
    ] == []


def third_party_imports():
    """The top-level names that the package's modules import and no standard
    library module or agent_eda module provides, whether at the top of a module,
    inside a function or under TYPE_CHECKING."""
    names = set()
    for source in (ROOT / "src" / "agent_eda").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names - sys.stdlib_module_names - {"agent_eda"}


def run_python(code, *, args=(), env=None, cwd=None):
    completed = subprocess.run(
        [sys.executable, *args, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=cwd,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout
