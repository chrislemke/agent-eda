"""The commands and checks that AGENTS.md documents, and the files that run them."""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text()
README = (ROOT / "README.md").read_text()
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text()


def section(text, heading):
    """The body of a `## heading` or `### heading` section, up to the next heading."""
    body = re.split(rf"^#{{2,3}} {re.escape(heading)}\n", text, flags=re.MULTILINE)[1]
    return re.split(r"^#{2,3} ", body, flags=re.MULTILINE)[0]


def test_uv_refuses_to_run_below_the_version_the_commands_need():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["tool"]["uv"]["required-version"] == ">=0.13"


@pytest.mark.parametrize(
    "text",
    [AGENTS, README, CONTRIBUTING],
    ids=["AGENTS.md", "README.md", "CONTRIBUTING.md"],
)
def test_no_document_repeats_the_uv_version_caveat(text):
    assert "uv 0.13 or later" not in text
    assert "older uv" not in text
    assert "uv 0.10" not in text


def test_each_command_in_agents_md_appears_once():
    commands = re.findall(r"^(uv [^#\n]*?)\s*(?:#.*)?$", AGENTS, flags=re.MULTILINE)
    commands += re.findall(r"`(uv [^`]+)`", AGENTS)

    repeated = sorted({command for command in commands if commands.count(command) > 1})

    assert repeated == [], f"AGENTS.md lists these commands more than once: {repeated}"


GENERATOR_COMMAND = "uv run python scripts/generate_reference.py"


def test_agents_md_lists_the_reference_generator_and_its_check_form():
    commands = section(AGENTS, "Commands")

    assert f"\n{GENERATOR_COMMAND} " in commands
    assert f"\n{GENERATOR_COMMAND} --check " in commands


@pytest.mark.parametrize(
    "command",
    [GENERATOR_COMMAND, "uv run pytest -m benchmark"],
    ids=["generator", "benchmark"],
)
def test_the_other_guides_point_to_agents_md_instead_of_spelling_out_these_commands(
    command,
):
    guides = [
        ROOT / "CONTRIBUTING.md",
        *sorted((ROOT / "docs" / "agents").glob("*.md")),
    ]

    spelled = [guide.name for guide in guides if command in guide.read_text()]

    assert spelled == [], f"{spelled} repeat `{command}`: point to AGENTS.md instead."


def test_the_finish_checks_list_the_exact_commands():
    checks = section(AGENTS, "Finish checks")
    spans = re.findall(r"`([^`]+)`", checks)

    for command in (
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run ty check",
        "uv run pytest",
        "uv run --extra mcp python -I tests/smoke_mcp.py",
        "uv run pytest -m benchmark",
    ):
        assert command in spans, f"The Finish checks do not give `{command}`."
    minimum = [span for span in spans if "lowest-direct" in span]
    assert len(minimum) == 2
    assert any("--no-group mcp" in span for span in minimum)
    assert any("--extra mcp" in span for span in minimum)


def test_the_finish_checks_say_why_isolated_is_there_and_when_to_run_the_benchmark():
    checks = section(AGENTS, "Finish checks")

    assert "--isolated" in checks
    assert "uv.lock" in checks
    assert re.search(r"benchmark.*(cost|rows|columns)", checks, flags=re.IGNORECASE)


@pytest.mark.parametrize(
    "text", [README, CONTRIBUTING], ids=["README.md", "CONTRIBUTING.md"]
)
def test_the_guides_point_to_agents_md_instead_of_repeating_the_commands(text):
    assert "AGENTS.md" in text
    assert "lowest-direct" not in text
    assert "| Task | Command |" not in text


def test_agents_md_states_the_performance_target_the_benchmark_checks():
    assert (
        "under 5 seconds on 1,000,000 rows \N{MULTIPLICATION SIGN} 100 columns"
        in AGENTS
    )
    assert "1M rows" not in AGENTS


def test_the_pre_push_pytest_hook_runs_on_every_push_and_fails_on_no_tests():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    hook = hooks["pytest"]

    assert hook["stages"] == ["pre-push"]
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
    assert "exit 0" not in hook["entry"]
    assert "-eq 5" not in hook["entry"]
    assert hook["entry"].split()[:3] == ["uv", "run", "pytest"]


def test_coverage_follows_subprocesses():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["tool"]["coverage"]["run"]["patch"] == ["subprocess"]


def test_claude_code_does_not_enable_pyright_lsp():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())

    assert not [name for name in settings["enabledPlugins"] if "pyright" in name]


@pytest.mark.parametrize("path", ["AGENTS.md", "README.md", "CONTRIBUTING.md", "docs"])
def test_documents_name_review_skills_with_their_plugin_prefix(path):
    target = ROOT / path
    files = [target] if target.is_file() else sorted(target.rglob("*.md"))

    bare = [
        file.name
        for file in files
        if re.search(r"(?<![\w:-])code-review\b", file.read_text())
    ]

    assert bare == [], f"Write mattpocock-skills:code-review, not code-review: {bare}"


def test_the_mcp_test_modules_say_how_to_get_the_extra_when_it_is_missing():
    # sys.modules["mcp"] = None makes `import mcp` fail as it does without the extra.
    code = (
        "import sys; sys.modules['mcp'] = None; import pytest; "
        "sys.exit(pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', "
        "'tests/test_mcp_server.py', 'tests/test_mcp_data.py']))"
    )

    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert "skipped" in run.stdout, run.stdout + run.stderr
    assert "uv sync` (uv 0.13 or later)" in run.stdout or (
        "uv sync (uv 0.13 or later)" in run.stdout
    )
    assert "uv run --extra mcp pytest" in run.stdout
    assert "pip install 'agent-eda[mcp]'" in run.stdout
