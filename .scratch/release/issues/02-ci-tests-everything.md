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

Status: done

- [x] **Triggers:** `push` on branches only, `pull_request`, `workflow_dispatch`, a weekly `schedule`, and `workflow_call` with a boolean input `full`. Superseded pull-request runs are cancelled.
- [x] `UV_PYTHON_PREFERENCE: only-managed` is set for the whole workflow. The Comments quote the Python version a 3.12 job now uses.
- [x] `tests-newest` runs on the schedule, on dispatch and when `full` is true. It uses `--upgrade-package` for every runtime dependency and the mcp extra.
- [x] `tests-agentcore` runs on ARM64 with numpy 1.26.4 and pandas 2.3.1 installed over the locked environment.
- [x] The lint job runs the pre-commit file checks, skipping ruff, ty, pytest and uv-lock, which have their own steps. `.pre-commit-config.yaml` pins its hooks by commit SHA.
- [x] The benchmark job runs on dispatch or when `inputs.full` is true.
- [x] One dispatched run of ci.yml is green, the benchmark included. The Comments record its URL and the slowest analysis's time.
- [x] The Comments record whether floating transitive dependencies in the minimum-versions runs caused trouble (REL-13), and the fix, if there was one.
- [x] `tests/test_ci_workflow.py` checks the parts of each command that matter (`--extra mcp`, `--isolated`, `pytest`) instead of exact strings. It also checks the new jobs and triggers.
- [x] `.github/dependabot.yml` updates the `github-actions` ecosystem weekly.
- [x] The finish checks pass.

## Comments

- Green runs on commit `433d730`: the dispatched run https://github.com/chrislemke/agent-eda/actions/runs/39416163642 (`gh workflow run ci.yml --ref release/prepare-0.1.0`; all 19 jobs, the benchmark and `tests-newest` included) and the push run https://github.com/chrislemke/agent-eda/actions/runs/39416132423 (every job but the benchmark and `tests-newest`, which only dispatch, the schedule and releases start).
- Benchmark on GitHub's 2-CPU ARM64 runner (1,000,000 rows x 100 columns, every analysis under the 5-second limit):
  - Latest versions (Python 3.12.15, numpy 2.5.4, pandas 3.0.6): the slowest is `infer_column_semantics` at 3.52 s, then `validate_column_types` 3.18 s and `summarize_categorical` 2.88 s. The 14 tests take 38 s.
  - Minimum versions (numpy 1.26.0, pandas 2.2.0): the slowest is `describe_dataset` at 4.45 s, then `infer_column_semantics` 4.37 s and `detect_missing_sentinels` 3.87 s. That is the closest the suite comes to the limit, 0.55 s of margin on pandas 2.2.0.
- Python: with `UV_PYTHON_PREFERENCE: only-managed`, the Linux 3.12 jobs download and run `cpython-3.12.15` (`Using CPython 3.12.15`) where they ran Ubuntu's own CPython 3.12.3.
- REL-13, floating transitive dependencies in the minimum-versions runs: no trouble, no fix. The direct dependencies sit at their floors (numpy 1.26.0, pandas 2.2.0, pydantic 2.7.0, scipy 1.11.2, scikit-learn 1.5.0 and statsmodels 0.14.1 at the time), the transitive ones resolve to the newest (pytz and tzdata 2026.5, packaging 26.3, joblib 1.6.0, patsy 1.0.3), and both minimum-versions runs passed locally and on GitHub in every run above. `--exclude-newer` wasn't needed.

Settled here, where the spec left it open:
- `tests-newest` runs on Python 3.12 and 3.14. Its `--upgrade-package` list is checked against `pyproject.toml` (every runtime dependency and the extra's), so a new dependency can't be left out. Today the newest releases (numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, pyarrow 26.0.0, mcp 2.3.0) are the versions in `uv.lock`, so the job finds nothing the locked runs don't; it will once they drift.
- The schedule is Mondays at 04:23 UTC. It first fires on `main` after the merge, so the schedule itself hasn't run yet.
- The lint job runs `uv run --locked pre-commit run --all-files`, not `uvx pre-commit`: pre-commit is already in `uv.lock`, so its version is pinned and no unpinned tool is fetched. The two hook repositories keep the versions they had (v6.0.0 and 0.13.1), written as commit SHAs the way `pre-commit autoupdate --freeze` writes them.
- `.github/dependabot.yml` has a 7-day cooldown: zizmor's `dependabot-cooldown` audit flags the file without one (checked by removing it).
- Superseded pull-request runs are cancelled by the `concurrency` group; the test checks the expression, and I didn't watch a cancellation happen.

The AgentCore job differs from the spec, which swapped numpy and pandas into the locked environment. I tried that first and it didn't give a usable run: 42 test modules failed to import.
- `uv.lock` holds scipy 1.18.1, whose metadata needs a newer numpy than 1.26.4 (`uv pip install numpy==1.26.4 'scipy==1.18.*'` finds no solution), and which touches `np.long`, so numpy 1.26 raises a `FutureWarning` that the suite turns into an error.
- The locked pyarrow 26 refuses to import under numpy 1.x (`pyarrow requires NumPy 2.0 or newer, found 1.26.4`) and says nothing about it in its metadata.
- So the job resolves the project next to numpy 1.26.4 and pandas 2.3.1, as pip would (constraints file, `pyarrow<26`). On the ARM64 runner that gave scipy 1.17.1, pyarrow 25.0.1, pydantic 2.14.1 and mcp 2.3.0, and `uv pip check` passes. A step asserts that the interpreter imports the two versions: my first local attempt ran the latest versions without my noticing, because `uv pip install` ignores `UV_PROJECT_ENVIRONMENT`.
- Two tests can't pass on pandas 2.3.1 as written. Neither is a defect of the package:
  - `tests/test_baseline.py`: the byte baseline for pandas 2 holds the digests of pandas 2.2.0, the minimum. pandas 2.3 gives the generated frames the dtype `str` where 2.2 has `string` (`tests/strategies.py` builds the closest twin of pandas 3's `str` per version), and it counts the memory of a frame differently (3.609 KiB against 3.633 KiB in the same `describe_dataset` text). The AgentCore job deselects the file; both minimum-versions runs and the latest-versions runs still check it. `scripts/record_baseline.py` says the digests hold across every release of a major version, which is not true from pandas 2.2 to 2.3. A baseline per minor version would fix that, and is the maintainer's to choose.
  - `tests/test_result_pickles.py`: a NumPy 1 fixture written on pandas 2.2 holds a string dtype under the storage name that pandas 2.3 deprecated. Pickling it again warns, and the suite turns `FutureWarning` into an error. The test lets exactly that warning by.
- Everything else passes on the AgentCore versions: 3,964 tests on GitHub's ARM64 runner.

For the maintainer:
- pyarrow 26 needs NumPy 2 without declaring it. `pip install 'agent-eda[mcp]'` into an environment that already has numpy 1.26 therefore installs a pyarrow that can't import, and `agent-eda-mcp` exits 1 with a traceback that ends in pyarrow's message (checked with mcp 2.3.0, pyarrow 26.0.0 and numpy 1.26.4). Not changed: the extra's floor is right for every other environment, and a requirement can't say "below 26 when numpy is 1.x".
- The ruleset `main` has an empty list of required status checks today (2026-10-20), so the check names that ticket 04 listed as unsatisfiable are gone. The AgentCore job's name, `tests, AgentCore versions (Python 3.12, ARM64)`, is the one that ticket 04 expected, should the list name it again.
- The skipped jobs show their names unexpanded (`benchmark, ${{ matrix.versions }} versions ...`), since GitHub doesn't expand a matrix expression in the name of a job it skips.
