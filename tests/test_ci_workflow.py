"""The CI workflow runs the commands that the README documents for local runs."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_the_minimum_versions_job_runs_the_documented_command():
    command = documented_command("Test on the minimum versions")
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["tests-minimum"]["steps"]

    assert command in [step.get("run", "").strip() for step in steps]


def documented_command(task):
    """The command that the README's Development table gives for a task."""
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == task:
            return cells[1].strip("`")
    pytest.fail(f"The README's Development table has no row for {task!r}.")
