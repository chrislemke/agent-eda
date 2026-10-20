"""The CI workflow runs the commands that AGENTS.md lists as finish checks, and a
version tag runs the release workflow that publishes to PyPI and then to GitHub."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE_JOBS = ["ci", "build", "provenance", "publish", "github-release"]
needs_posix_shell = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the release steps are bash scripts that run on Ubuntu",
)


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


def test_the_latest_versions_job_also_runs_on_macos_and_windows_with_bash():
    job = workflow()["jobs"]["tests-latest"]
    cells = job["strategy"]["matrix"]["include"]

    for runner_prefix in ("macos-", "windows-"):
        pythons = {
            cell["python"] for cell in cells if cell["runner"].startswith(runner_prefix)
        }
        assert {"3.12", "3.14"} <= pythons, runner_prefix
    # The commands are bash, whichever the runner: Windows would run PowerShell.
    assert job["defaults"]["run"]["shell"] == "bash"


def test_the_stdio_smoke_test_runs_on_python_3_12_and_3_14():
    job = workflow()["jobs"]["mcp-smoke"]
    pythons = [entry["python"] for entry in job["strategy"]["matrix"]["include"]]

    assert sorted(pythons) == ["3.12", "3.14"]
    assert "uv run --locked --extra mcp python -I tests/smoke_mcp.py" in commands(
        "mcp-smoke"
    )


def test_the_offline_packaging_job_installs_no_extra():
    assert not [run for run in commands("packaging") if "mcp" in run]


def test_a_tag_push_does_not_start_the_ci_workflow_a_second_time():
    push = triggers("ci.yml")["push"]

    # The release workflow calls ci.yml for the tag, so ci.yml's own push trigger
    # stays on branches.
    assert push["branches"] == ["**"]
    assert "tags" not in push


def test_another_workflow_can_call_ci_and_ask_for_the_full_run():
    inputs = triggers("ci.yml")["workflow_call"]["inputs"]

    assert inputs["full"]["type"] == "boolean"
    assert inputs["full"]["default"] is False


def test_the_benchmark_runs_when_started_by_hand_or_when_a_caller_asks_for_full():
    condition = workflow()["jobs"]["benchmark"]["if"]

    assert "workflow_dispatch" in condition
    # A called workflow sees the caller's event, so the event name alone says nothing.
    assert "inputs.full" in condition


def test_the_release_workflow_is_the_file_the_trusted_publishers_name():
    assert (WORKFLOWS / "release.yml").is_file()


def test_only_a_version_tag_starts_a_release():
    assert triggers("release.yml") == {"push": {"tags": ["v*"]}}


def test_a_release_calls_the_full_ci_before_anything_else():
    ci = workflow("release.yml")["jobs"]["ci"]

    assert ci["uses"] == "./.github/workflows/ci.yml"
    assert ci["with"] == {"full": True}
    for job in RELEASE_JOBS[1:]:
        assert "ci" in ancestors(job), job


def test_a_release_builds_attests_publishes_and_only_then_releases_on_github():
    assert list(workflow("release.yml")["jobs"]) == RELEASE_JOBS
    assert needs("build") == {"ci"}
    assert needs("provenance") == {"build"}
    assert {"build", "provenance"} <= needs("publish")
    assert {"build", "publish"} <= needs("github-release")


def test_each_release_job_holds_only_the_permissions_it_uses():
    jobs = workflow("release.yml")["jobs"]

    assert workflow("release.yml")["permissions"] == {}
    assert jobs["ci"]["permissions"] == {"contents": "read"}
    assert jobs["build"]["permissions"] == {"contents": "read"}
    assert jobs["provenance"]["permissions"] == {
        "id-token": "write",
        "attestations": "write",
        "artifact-metadata": "write",
    }
    assert jobs["publish"]["permissions"] == {"id-token": "write"}
    assert jobs["github-release"]["permissions"] == {"contents": "write"}


def test_the_publish_job_runs_no_code_but_the_upload_and_the_download_of_the_build():
    steps = workflow("release.yml")["jobs"]["publish"]["steps"]
    actions = [step["uses"].split("@")[0] for step in steps]

    assert actions == [
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ]


def test_the_environments_are_pypi_and_testpypi_and_only_publishing_uses_one():
    jobs = workflow("release.yml")["jobs"]
    route = step_script("build", "route")

    # The build job names the index, and the publish job takes it as its environment.
    assert set(re.findall(r"index=(\w+)", route)) == {"pypi", "testpypi"}
    assert jobs["publish"]["environment"] == "${{ needs.build.outputs.index }}"
    assert [name for name, job in jobs.items() if "environment" in job] == ["publish"]


def test_a_release_holds_no_long_lived_token():
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")

    assert set(re.findall(r"secrets\.(\w+)", text)) <= {"GITHUB_TOKEN"}
    assert "password:" not in text


def test_every_action_is_pinned_to_a_commit_sha():
    for name in workflow_files():
        for job_name, job in workflow(name)["jobs"].items():
            for step in job.get("steps", []):
                if "uses" in step:
                    assert re.fullmatch(r".+@[0-9a-f]{40}", step["uses"]), (
                        name,
                        job_name,
                        step["uses"],
                    )
            # A reusable workflow of this repository has no commit to pin.
            if "uses" in job:
                assert job["uses"].startswith("./.github/workflows/"), (name, job_name)


def test_every_workflow_builds_with_the_backend_the_project_declares():
    builds = [
        command
        for name in workflow_files()
        for job in workflow(name)["jobs"].values()
        for step in job.get("steps", [])
        for command in [step.get("run", "")]
        if "uv build" in command
    ]

    assert builds
    assert all("--force-pep517" in command for command in builds)


def test_the_github_release_marks_a_pre_release_and_checks_that_the_tag_exists():
    command = step_script("github-release", "create")

    assert "gh release create" in command
    assert "--verify-tag" in command
    assert "--prerelease" in command
    assert "--latest=false" in command


@needs_posix_shell
@pytest.mark.parametrize(
    ("version", "index", "prerelease"),
    [
        ("0.1.0rc1", "testpypi", "true"),
        ("0.1.0a1", "pypi", "true"),
        ("0.1.0b2", "pypi", "true"),
        ("0.1.0", "pypi", "false"),
        ("0.1.0.post1", "pypi", "false"),
    ],
)
def test_a_tag_goes_to_the_index_that_its_kind_of_version_belongs_to(
    tmp_path, version, index, prerelease
):
    run = run_route(tmp_path, version=version, tag=f"v{version}")

    assert run.returncode == 0, run.stderr
    assert run.values["version"] == version
    assert run.values["index"] == index
    assert run.values["prerelease"] == prerelease
    expected = {
        "pypi": "https://upload.pypi.org/legacy/",
        "testpypi": "https://test.pypi.org/legacy/",
    }[index]
    assert run.values["repository_url"] == expected


@needs_posix_shell
def test_a_tag_that_differs_from_the_package_version_stops_the_release(tmp_path):
    run = run_route(tmp_path, version="0.1.0", tag="v0.2.0")

    assert run.returncode == 1
    assert "v0.2.0" in run.stdout
    assert "0.1.0" in run.stdout
    assert not run.values


@needs_posix_shell
def test_a_version_that_no_index_route_covers_stops_the_release(tmp_path):
    run = run_route(tmp_path, version="0.1.0.dev1", tag="v0.1.0.dev1")

    assert run.returncode == 1
    assert "rc" in run.stdout
    assert not run.values


CHANGELOG = """# Changelog

## [Unreleased]

- Unreleased line.

## [0.2.0] - 2026-12-01

- Line of 0.2.0.

## [0.1.0] - 2026-11-01

### Changed

- Line of 0.1.0.
"""


@needs_posix_shell
def test_a_final_release_takes_the_notes_of_its_own_changelog_section(tmp_path):
    run = run_notes(tmp_path, CHANGELOG, version="0.1.0", prerelease="false")

    assert run.returncode == 0, run.stdout
    assert run.notes.strip() == "### Changed\n\n- Line of 0.1.0."


@needs_posix_shell
def test_a_pre_release_without_its_own_section_takes_the_unreleased_notes(tmp_path):
    run = run_notes(tmp_path, CHANGELOG, version="0.3.0rc1", prerelease="true")

    assert run.returncode == 0, run.stdout
    assert "- Unreleased line." in run.notes
    assert "Line of 0.2.0" not in run.notes


@needs_posix_shell
def test_a_final_release_without_its_own_section_stops_the_release(tmp_path):
    run = run_notes(tmp_path, CHANGELOG, version="0.3.0", prerelease="false")

    assert run.returncode == 1
    assert "## [0.3.0]" in run.stdout


@needs_posix_shell
def test_the_changelog_of_the_repository_gives_a_release_candidate_its_notes(tmp_path):
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    run = run_notes(tmp_path, changelog, version="0.1.0rc1", prerelease="true")

    assert run.returncode == 0, run.stdout
    assert run.notes.strip()
    assert "## [" not in run.notes


@needs_posix_shell
def test_a_changelog_section_with_no_lines_stops_the_release(tmp_path):
    changelog = CHANGELOG.replace("- Line of 0.2.0.\n", "")
    run = run_notes(tmp_path, changelog, version="0.2.0", prerelease="false")

    assert run.returncode == 1
    assert "empty" in run.stdout


def workflow(name="ci.yml"):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def workflow_files():
    return sorted(path.name for path in WORKFLOWS.glob("*.yml"))


def triggers(name):
    """The events that start a workflow. YAML 1.1 reads the key `on` as True."""
    parsed = workflow(name)
    return parsed["on"] if "on" in parsed else parsed[True]


def commands(job, name="ci.yml"):
    """The shell commands of a workflow job's steps, stripped."""
    return [
        step["run"].strip()
        for step in workflow(name)["jobs"][job]["steps"]
        if "run" in step
    ]


def needs(job):
    declared = workflow("release.yml")["jobs"][job].get("needs", [])
    return {declared} if isinstance(declared, str) else set(declared)


def ancestors(job):
    """Every release job that must finish before this one starts."""
    found = set()
    for before in needs(job):
        found |= {before} | ancestors(before)
    return found


def step_script(job, step_id):
    """The bash script of one step of a release job, found by the step's id."""
    steps = workflow("release.yml")["jobs"][job]["steps"]
    return next(step["run"] for step in steps if step.get("id") == step_id)


class StepRun:
    def __init__(self, completed, values, notes):
        self.returncode = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        self.values = values
        self.notes = notes


def run_step(tmp_path, script, env, *, files=None):
    """Run a step's script as GitHub runs bash: in a working directory with the
    files a checkout would hold, a `uv` that prints FAKE_VERSION, and a file for
    the step's values."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    for name, content in (files or {}).items():
        (work / name).write_text(content, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\necho "$FAKE_VERSION"\n', encoding="utf-8")
    uv.chmod(0o755)
    values_file = tmp_path / "github_output"
    values_file.touch()
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=work,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(values_file),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    values = dict(
        line.split("=", 1)
        for line in values_file.read_text(encoding="utf-8").splitlines()
    )
    notes_file = tmp_path / "release-notes" / "RELEASE_NOTES.md"
    notes = notes_file.read_text(encoding="utf-8") if notes_file.exists() else ""
    return StepRun(completed, values, notes)


def run_route(tmp_path, *, version, tag):
    return run_step(
        tmp_path,
        step_script("build", "route"),
        {"TAG": tag, "FAKE_VERSION": version},
    )


def run_notes(tmp_path, changelog, *, version, prerelease):
    return run_step(
        tmp_path,
        step_script("build", "notes"),
        {"VERSION": version, "PRERELEASE": prerelease},
        files={"CHANGELOG.md": changelog},
    )


def documented_command(task):
    """The minimum-versions command that AGENTS.md's Finish checks give for a task.

    The task is "Test on the minimum versions" for the run without the mcp extra
    and "Test on the minimum versions with the mcp extra" for the run with it.
    """
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    checks = text.split("\n### Finish checks\n")[1].split("\n## ")[0]
    minimum = [
        span
        for span in re.findall(r"`([^`]+)`", checks)
        if "--resolution lowest-direct" in span
    ]
    with_extra = task.endswith("with the mcp extra")
    matching = [span for span in minimum if ("--extra mcp" in span) == with_extra]
    if len(matching) != 1:
        pytest.fail(f"AGENTS.md's Finish checks need one command for {task!r}.")
    return matching[0]
