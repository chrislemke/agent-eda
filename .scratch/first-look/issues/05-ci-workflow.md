# 05: CI workflow

**What to build:** Once the repo is on GitHub, every push runs the checks AgentCore users depend on. That covers lint and types, the tests on every supported Python with the latest and the minimum dependency versions, Linux ARM64, and the offline install routes for sandboxed agents. The benchmark runs only when triggered by hand.

The repo has no remote yet, so this ticket delivers the workflow and runs each job's commands locally where the platform allows.

Read these sections of `.scratch/first-look/spec.md`: CI and test group 8 under Testing Decisions, and Compatibility. Covers user stories 2, 3, 6, 120 and 121.

**Blocked by:** 03-property-tests-and-benchmark

**Status:** ready-for-agent

- [ ] A GitHub Actions workflow has these jobs:
  - lint: `ruff check`, `ruff format --check` and `ty`
  - tests with the latest dependencies on Python 3.12, 3.13 and 3.14, Linux x86_64
  - tests with the lowest direct dependencies on Python 3.12, since numpy 1.26 has no 3.13 wheels
  - tests on Linux ARM64, Python 3.12, latest dependencies
  - packaging: build the wheel, check that it is pure Python, run the `sys.path` import test, install the wheel with `pip install --no-deps` into an environment holding only the minimum required versions, and run a smoke test that prints an analysis result
  - the benchmark, on manual trigger only, limited to 2 CPUs to match the target machine
- [ ] The lowest-versions job runs the same command as the documented local one from ticket 01, so the two can't drift apart.
- [ ] The lint, latest-versions, lowest-versions and packaging commands each pass when run locally. Any job that couldn't be checked locally, such as ARM64, is listed under `## Comments` in this ticket.
- [ ] The workflow file passes the repo's pre-commit checks.
- [ ] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass.
