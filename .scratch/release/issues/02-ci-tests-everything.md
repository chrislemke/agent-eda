# 02: CI tests everything else

What to build: CI covers:
- the newest dependency releases, weekly;
- the AgentCore versions on ARM64;
- the pre-commit file checks;
- Python installed by uv;
- triggers and concurrency fit for a calling release workflow;
- one real benchmark run on GitHub.

Read these sections of `.scratch/release/spec.md`: ci.yml, and test group 1. This ticket covers user stories 2, 3, 7, 8 and 9. Findings: REL-5, REL-7, REL-8, REL-9, REL-10, REL-13, REL-14, TEST-12.

Blocked by: 01

Status: ready-for-agent

- [ ] **Triggers:** `push` on branches only, `pull_request`, `workflow_dispatch`, a weekly `schedule`, and `workflow_call` with a boolean input `full`. Superseded pull-request runs are cancelled.
- [ ] `UV_PYTHON_PREFERENCE: only-managed` is set for the whole workflow. The Comments quote the Python version a 3.12 job now uses.
- [ ] `tests-newest` runs on the schedule, on dispatch and when `full` is true. It uses `--upgrade-package` for every runtime dependency and the mcp extra.
- [ ] `tests-agentcore` runs on ARM64 with numpy 1.26.4 and pandas 2.3.1 installed over the locked environment.
- [ ] The lint job runs the pre-commit file checks, skipping ruff, ty, pytest and uv-lock, which have their own steps. `.pre-commit-config.yaml` pins its hooks by commit SHA.
- [ ] The benchmark job runs on dispatch or when `inputs.full` is true.
- [ ] One dispatched run of ci.yml is green, the benchmark included. The Comments record its URL and the slowest analysis's time.
- [ ] The Comments record whether floating transitive dependencies in the minimum-versions runs caused trouble (REL-13), and the fix, if there was one.
- [ ] `tests/test_ci_workflow.py` checks the parts of each command that matter (`--extra mcp`, `--isolated`, `pytest`) instead of exact strings. It also checks the new jobs and triggers.
- [ ] `.github/dependabot.yml` updates the `github-actions` ecosystem weekly.
- [ ] The finish checks pass.

## Comments
