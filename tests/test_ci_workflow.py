"""The CI workflow runs the commands that the README documents for local runs."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_the_minimum_versions_job_runs_the_documented_command():
    command = documented_command("Test on the minimum versions")

    assert command in commands("tests-minimum")
    assert "--no-group mcp" in command


def test_a_second_minimum_versions_job_runs_the_documented_command_with_the_extra():
    command = documented_command("Test on the minimum versions with the mcp extra")

    assert command in commands("tests-minimum-mcp")
    assert "--extra mcp" in command


def test_the_latest_versions_jobs_install_the_mcp_extra():
    assert "uv run --locked --extra mcp pytest" in commands("tests-latest")


def test_the_stdio_smoke_test_runs_on_python_3_12_and_3_14():
    job = workflow()["jobs"]["mcp-smoke"]
    pythons = [entry["python"] for entry in job["strategy"]["matrix"]["include"]]

    assert sorted(pythons) == ["3.12", "3.14"]
    assert "uv run --locked --extra mcp python -I tests/smoke_mcp.py" in commands(
        "mcp-smoke"
    )


def test_the_offline_packaging_job_installs_no_extra():
    assert not [run for run in commands("packaging") if "mcp" in run]


def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def commands(job):
    """The shell commands of a CI job's steps, stripped."""
    return [
        step["run"].strip()
        for step in workflow()["jobs"][job]["steps"]
        if "run" in step
    ]


def documented_command(task):
    """The command that the README's Development table gives for a task."""
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == task:
            return cells[1].strip("`")
    pytest.fail(f"The README's Development table has no row for {task!r}.")
