# 05: CI workflow

**What to build:** Once the repo is on GitHub, every push runs the checks AgentCore users depend on. That covers lint and types, the tests on every supported Python with the latest and the minimum dependency versions, Linux ARM64, and the offline install routes for sandboxed agents. The benchmark runs only when triggered by hand.

The repo has no remote yet, so this ticket delivers the workflow and runs each job's commands locally where the platform allows.

Read these sections of `.scratch/first-look/spec.md`: CI and test group 8 under Testing Decisions, and Compatibility. Covers user stories 2, 3, 6, 120 and 121.

**Blocked by:** 03-property-tests-and-benchmark

**Status:** ready-for-agent

- [x] A GitHub Actions workflow has these jobs:
  - lint: `ruff check`, `ruff format --check` and `ty`
  - tests with the latest dependencies on Python 3.12, 3.13 and 3.14, Linux x86_64
  - tests with the lowest direct dependencies on Python 3.12, since numpy 1.26 has no 3.13 wheels
  - tests on Linux ARM64, Python 3.12, latest dependencies
  - packaging: build the wheel, check that it is pure Python, run the `sys.path` import test, install the wheel with `pip install --no-deps` into an environment holding only the minimum required versions, and run a smoke test that prints an analysis result
  - the benchmark, on manual trigger only, limited to 2 CPUs to match the target machine
- [x] The lowest-versions job runs the same command as the documented local one from ticket 01, so the two can't drift apart.
- [x] The lint, latest-versions, lowest-versions and packaging commands each pass when run locally. Any job that couldn't be checked locally, such as ARM64, is listed under `## Comments` in this ticket.
- [x] The workflow file passes the repo's pre-commit checks.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass.

## Comments

What this ticket settled that the spec left open:

- **Triggers.** The workflow `.github/workflows/ci.yml` runs on every push, on every pull request, so pull requests from forks get checked too, and on `workflow_dispatch`. Only the benchmark job waits for `workflow_dispatch`; a manual run also runs every other job. A pull request from a branch of this repo runs the checks twice, once for the push and once for the pull request's merge commit.
- **Runners.** `ubuntu-24.04` for x86_64 and `ubuntu-24.04-arm` for ARM64, so both run the same Ubuntu. GitHub gives public repositories 4 vCPUs and 16 GB on both, and private ones 2 vCPUs and 8 GB, the target's size.
- **Latest versions** are the versions in `uv.lock`, the ones `uv run` uses locally. Every job that uses the lock runs `uv run --locked`, which fails when `uv.lock` is out of date. `UV_LOCKED=1` can't be set for the whole workflow, because the minimum-versions command then stops with "The lockfile at `uv.lock` needs to be updated". The ARM64 run is the fourth entry of the latest-versions matrix. CI sees a new numpy or pandas release only after someone upgrades the lock; a scheduled run against fresh releases would catch breakage sooner, and isn't part of this ticket.
- **Drift.** `tests/test_ci_workflow.py` reads the "Test on the minimum versions" row of the README's Development table and fails unless the `tests-minimum` job has a step that runs exactly that command. Changing either side alone fails it; I checked both. It parses the workflow with PyYAML, which joins the dev dependencies as `pyyaml>=6.0.3`.
- **Packaging job.** It builds the wheel with `uv build --wheel --force-pep517`. Without the flag, uv 0.13.1 builds with the backend bundled in it and warns that `uv_build>=0.10.11,<0.11.0` in `[build-system]` doesn't contain its version. With it, uv installs uv_build 0.10.12 from that range and calls its PEP 517 hook, as pip would. Today the two wheels differ only in the `Generator` line of `WHEEL` and its hash in `RECORD`. The job then runs `tests/test_packaging.py` on that file: the `wheel` fixture returns the file `AGENT_EDA_WHEEL` names and builds one as before when the variable is unset. Pointed at a copy renamed to `agent_eda-0.1.0-cp312-cp312-linux_x86_64.whl`, the purity test fails. Next, `uv venv --seed --python 3.12` and `uv pip install --resolution lowest-direct -r pyproject.toml` make an environment with the six dependencies at their minimums (numpy 1.26.0, pandas 2.2.0, pydantic 2.7.0, scikit-learn 1.5.0, scipy 1.11.2, statsmodels 0.14.1), their own dependencies and nothing else, so no pyarrow. pip installs the wheel there with `--no-deps --no-index`, which keeps it off the network, and `pip check` confirms that the minimums satisfy the wheel's metadata. Last, `tests/smoke.py` prints the version, the import path and a describe_dataset result, then runs every analysis's docstring examples with the contract test's doctest options and exits with 1 if one fails. Every other job has the dev dependencies installed, so this is the one place where an analysis that imports something beyond the six dependencies fails. pytest doesn't collect the script. In that environment pandas 2.2.0 prints its DeprecationWarning about pyarrow on import, as it does for anyone without pyarrow.
- **Benchmark job.** It runs on `ubuntu-24.04-arm` under `taskset --cpu-list 0,1`, as a matrix of two runs: `uv run --locked pytest -m benchmark`, and the README's minimum-versions command with `-m benchmark`. ARM64, because AgentCore Runtime runs on ARM64 and those runners' vCPUs are whole cores, like Graviton's; ticket 03 left the 2-vCPU target unmeasured and handed the 2-CPU run to this ticket. The minimum versions matter most there, since Code Interpreter reportedly runs pandas 2.3.1 with numpy 1.26.4 (ADR 0005). The benchmark's report line now gives the CPUs the process may use, `len(os.sched_getaffinity(0))` on Linux and `os.cpu_count()` elsewhere, so a run under taskset on a 4-vCPU runner reads "2 of 4 CPUs".
- **uv and the actions.** Every job installs uv 0.13.1, the version the uv-lock pre-commit hook runs; uv 0.10.11 and 0.13.1 both pass `uv lock --check` on the current lock. The actions are pinned to commit SHAs with the tag in a comment: actions/checkout v7.0.1 and astral-sh/setup-uv v10.3.0, which publishes no floating major tag. Checkout keeps no credentials, the token can only read contents, and the jobs skip the GitHub Actions cache, because setup-uv would give the minimum-versions runs the same cache key as the latest ones. uv 0.13.1 still prints the `[build-system]` warning once per job when it builds the editable install. Moving `[build-system]` and the `uv-build` dev dependency to the uv version in use would silence it, but that changes the build for every contributor, so I left it for a separate change.
- **Timeouts.** 10 minutes for lint, 15 for packaging, 20 for each test job and 30 for each benchmark run, instead of GitHub's 6 hours.

Checks:

- actionlint 1.7.12 with shellcheck 0.11.0 reports no errors, and zizmor 1.31.1 reports no findings; its pedantic persona only suggests a `concurrency` block. Neither runs as a pre-commit hook. To repeat them: `uvx --from actionlint-py actionlint` and `uvx zizmor .github/workflows/ci.yml`.
- I ran each job's `run:` steps on macOS arm64, read from `ci.yml` and run with `bash -eo pipefail`, uv 0.13.1, `RUNNER_TEMP` and, for the matrix, `UV_PYTHON`, as the runner and setup-uv set them. Lint passed. The tests passed with the latest versions on Python 3.12, 3.13 and 3.14, and with the minimum versions (numpy 1.26.0, pandas 2.2.0, scipy 1.11.2, pyarrow 14.0.1, PyYAML 6.0.3), which left `uv.lock` unchanged: 393 tests each on this ticket's branch, and 483 once ticket 04 was merged in. Packaging passed, and its smoke test ran 14 examples of 2 analyses.
- The benchmark passed without taskset, which macOS lacks, on an M3 Max that other agents were using at the same time. With the latest versions, describe_dataset took 1.14 s and summarize_numeric 1.70 s; with the minimum versions, 2.90 s and 1.92 s.
- `uv pip install --dry-run --only-binary :all:` found wheels for every locked package on Linux x86_64 for Python 3.12, 3.13 and 3.14 and on Linux ARM64 for 3.12, and for every package of the lowest-direct resolution on both architectures for 3.12. No job should have to build anything from source.

Not checked locally, for lack of Linux and of a GitHub remote:

- Every job on Linux: lint with ty's Linux platform rules (`uv run ty check --python-platform linux` passes on macOS), the latest-versions runs on x86_64 and ARM64, the minimum-versions run, packaging with Ubuntu's Python 3.12, and both benchmark runs on ARM64.
- `taskset` itself, and the benchmark's CPU count under it.
- Everything GitHub provides: the triggers, the runner labels, the pinned actions, setup-uv's `python-version` input, and the manual run from Actions > CI > Run workflow.
