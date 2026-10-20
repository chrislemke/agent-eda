import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

import agent_eda as ae

REQUIRED_EXPORTS = [
    "AnalysisResult",
    "__version__",
    "describe_analysis",
    "describe_dataset",
    "guide",
    "list_analyses",
    "search_analyses",
]


def test_version_is_the_installed_version():
    assert ae.__version__ == importlib.metadata.version("agent-eda")


def test_top_level_exports_are_listed_and_resolve():
    assert set(REQUIRED_EXPORTS) <= set(ae.__all__)
    assert set(ae.__all__) <= set(dir(ae))
    for name in ae.__all__:
        assert getattr(ae, name) is not None


def test_errors_and_results_are_public_modules():
    errors = importlib.import_module("agent_eda.errors")
    results = importlib.import_module("agent_eda.results")

    assert ae.errors is errors
    assert ae.results is results
    assert errors.InvalidArgumentError.__module__ == "agent_eda.errors"
    # A result class is defined in a private module next to its analysis;
    # agent_eda.results only gives it by name.
    package, _, module = results.DescribeDatasetResult.__module__.rpartition(".")
    assert package == ae.describe_dataset.__module__.rpartition(".")[0]
    assert module.startswith("_")
    assert ae.AnalysisResult is results.AnalysisResult


def test_describe_dataset_lives_in_data_structure():
    structure = importlib.import_module("agent_eda.data.structure")

    assert ae.describe_dataset is structure.describe_dataset
    assert ae.describe_dataset.__module__ == "agent_eda.data.structure"


def test_a_misspelled_attribute_suggests_the_closest_export():
    with pytest.raises(AttributeError, match="Did you mean 'describe_dataset'"):
        ae.describe_datset  # noqa: B018  # ty: ignore[unresolved-attribute]  # the typo is the test


def test_importing_the_package_loads_analyses_only_when_first_used():
    code = (
        "import sys, agent_eda as ae\n"
        "print('agent_eda.data.structure' in sys.modules)\n"
        "ae.describe_dataset\n"
        "print('agent_eda.data.structure' in sys.modules)\n"
    )

    out = run_python(code)

    assert out.split() == ["False", "True"]


def test_type_checkers_see_every_export_and_flag_typos(tmp_path):
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import pandas as pd\n"
        "import agent_eda as ae\n"
        "\n"
        "res = ae.describe_dataset(pd.DataFrame({'x': [1]}), source='a.csv')\n"
        "rows: int = res.row_count\n"
        "version: str = ae.__version__\n"
        "listing = ae.list_analyses()\n"
        "base: type[ae.AnalysisResult] = type(res)\n"
        "error: type[Exception] = ae.errors.InvalidArgumentError\n"
        "unknown: type[LookupError] = ae.errors.UnknownAnalysisError\n"
        "first: str = ae.search_analyses('nulls', limit=3).matches[0].name\n"
        "signature: str = ae.describe_analysis('describe_dataset').signature\n"
        "nothing: None = ae.guide()\n"
        "ae.describe_datset\n",
        encoding="utf-8",
    )
    ty = Path(sys.executable).with_name("ty")

    checked = subprocess.run(
        [
            ty,
            "check",
            "--python",
            sys.executable,
            "--output-format",
            "concise",
            consumer.name,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    diagnostics = [
        line for line in checked.stdout.splitlines() if line.startswith("consumer.py")
    ]
    assert len(diagnostics) == 1, checked.stdout
    assert diagnostics[0].startswith("consumer.py:14:")
    assert "unresolved-attribute" in diagnostics[0]


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
