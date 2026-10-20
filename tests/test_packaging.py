"""Packaging checks, each in a fresh interpreter, as a sandboxed agent would run."""

import json
import os
import py_compile
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import agent_eda as ae

ROOT = Path(__file__).resolve().parents[1]

# ADR 0005: the six required dependencies, at the minimums Code Interpreter runs.
REQUIRED_DEPENDENCIES = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pydantic>=2.7",
    "scikit-learn>=1.5",
    "scipy>=1.11.2",
    "statsmodels>=0.14.1",
]
MCP_EXTRA = ["mcp>=2.3 ; extra == 'mcp'", "pyarrow>=14.0.1 ; extra == 'mcp'"]

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
    out = tmp_path_factory.mktemp("dist")
    bin_dir = str(Path(sys.executable).parent)  # the hook runs uv-build from PATH
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, uv_build; print(uv_build.build_wheel(sys.argv[1]))",
            str(out),
        ],
        cwd=ROOT,
        env={**os.environ, "PATH": os.pathsep.join([bin_dir, os.environ["PATH"]])},
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    return out / built.stdout.strip().splitlines()[-1]


def test_wheel_is_pure_python_typed_and_declares_the_six_dependencies(wheel):
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


def test_the_mcp_server_without_the_extra_names_the_install_commands():
    # Setting sys.modules["mcp"] to None makes `import mcp` fail, as it does
    # where the extra isn't installed. The console script is loaded from the
    # installed entry point and run the way its generated script runs it.
    code = """
import sys
from importlib.metadata import entry_points
sys.modules["mcp"] = None
(script,) = entry_points(group="console_scripts", name="agent-eda-mcp")
assert script.value == "agent_eda.mcp:main", script.value
sys.argv = ["agent-eda-mcp"]
sys.exit(script.load()())
"""

    completed = subprocess.run(
        [sys.executable, *STRICT_WARNINGS, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "pip install 'agent-eda[mcp]'" in completed.stderr
    assert "uv add 'agent-eda[mcp]'" in completed.stderr
    assert "Traceback" not in completed.stderr


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
