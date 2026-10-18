"""Packaging checks, each in a fresh interpreter, as a sandboxed agent would run."""

import os
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
    assert sorted(requires) == REQUIRED_DEPENDENCIES


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
"""

    out = run_python(code, args=("-I", *STRICT_WARNINGS), cwd=tmp_path)

    lines = out.splitlines()
    assert lines[0] == ae.__version__
    assert lines[1] == "describe_dataset: completed on 2 rows x 1 column"
    assert "  describe_dataset: What does this dataset contain?" in lines


def test_import_and_list_analyses_leave_scipy_statsmodels_and_sklearn_unimported():
    code = """
import sys
import agent_eda as ae
ae.list_analyses()
heavy = {"scipy", "statsmodels", "sklearn"}
print(sorted(name for name in sys.modules if name.split(".")[0] in heavy))
"""

    assert run_python(code, args=("-I", *STRICT_WARNINGS)).strip() == "[]"


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

    outputs = {
        run_python(code, env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ("1", "2", "3")
    }

    assert len(outputs) == 1


def test_package_code_never_reads_files_through_dunder_file():
    sources = sorted((ROOT / "src" / "agent_eda").rglob("*.py"))

    assert sources
    assert [p.name for p in sources if "__file__" in p.read_text()] == []


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
