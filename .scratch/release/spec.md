# Spec: CI that tests everything, and releasing to GitHub and PyPI

Status: ready-for-agent

## Problem Statement

agent-eda has never been released. There are no tags, no GitHub releases and no PyPI project, and the name `agent-eda` was still free on PyPI on 2026-10-19. The maintainer wants three things:
- a GitHub Actions run that tests everything and is shown to run;
- an action that releases to GitHub;
- an action that publishes a version to PyPI.

`.github/workflows/ci.yml` exists, and its run on 2026-10-18 passed all ten jobs. The October 2026 audit (`.scratch/audit/findings.md`) found what it doesn't test, and what a release would ship wrong.

- **What CI misses.**
  - It never builds or checks the source distribution, and the artifacts depend on which uv builds them (REL-1).
  - No job runs on macOS or Windows, where the MCP clients the README names run. The Windows-only code paths have never executed (REL-6).
  - No job tests the dependency releases a fresh install gets (REL-7).
  - The AgentCore version set from ADR 0005 isn't a CI cell (REL-8).
  - The 3.12 jobs use Ubuntu's CPython 3.12.3 (REL-9).
  - There is no concurrency control, and tag pushes would run CI twice once a release workflow calls it (REL-10).
  - The built wheel's MCP route is never exercised (REL-11).
  - The benchmark job has never run on GitHub (REL-5).
  - The pre-commit file checks run only locally (REL-14).
  - With `mcp` installed but pyarrow missing, `agent-eda-mcp` dies with a traceback (STD-7, MCP-15).
- **What a release would ship wrong.**
  - A wheel built in a working copy contains `agent_eda/.DS_Store` (REL-2).
  - The metadata has no project URLs, authors, classifiers or keywords (REL-3).
  - The README, which becomes the PyPI description and can't be edited after upload, says "Until 0.1 is on PyPI…" and has relative links that break there (REL-4). The agent-workflow spec fixes the links.
  - scikit-learn and statsmodels are required, though nothing imports them (SPEC questionable decisions).
- **What constrains the design.** These facts come from primary sources, collected in REL's report:
  - PyPI's trusted publishing can't run from a reusable (called) workflow (docs.pypi.org/trusted-publishers/troubleshooting; pypi/warehouse#11096 is open).
  - A release created with `GITHUB_TOKEN` doesn't start another workflow (docs.github.com, "Triggering a workflow").

  So a GitHub-release workflow can't start a separate PyPI workflow without a long-lived token.

## Solution

- `ci.yml` tests everything the project claims to support:
  - Linux x86_64 and ARM64, macOS and Windows;
  - Python 3.12 to 3.14;
  - the minimum, AgentCore and newest dependency versions;
  - the wheel and the source distribution, with and without the mcp extra;
  - the benchmark.

  It can be called by another workflow.
- The package metadata is complete, nothing stray is packed, and only numpy, pandas, pydantic and scipy are required.
- One `release.yml`, started by a version tag, runs these jobs in order:
  1. the full CI;
  2. the build;
  3. build provenance;
  4. publishing to PyPI, after the maintainer's approval (TestPyPI for release candidates);
  5. the GitHub release with the files and the changelog section.
- `CHANGELOG.md` records what changed for callers.
- The maintainer does a one-time setup with a generated wizard script, rehearses with `0.1.0rc1` on TestPyPI, and publishes 0.1.0 once the correctness and agent-experience specs are done.

## User Stories

### Tests that cover what we claim

1. As a maintainer, I want every push and pull request tested on Linux, macOS and Windows, so that the MCP server works where its clients run.
2. As a maintainer, I want CI to test the newest dependency releases every week, so that a fresh install never meets a break CI hasn't seen.
3. As an AgentCore agent, I want the versions Code Interpreter ships tested, so that the package works in my sandbox.
4. As a maintainer, I want the source distribution built, checked and installed in CI, so that the file PyPI keeps forever is known good.
5. As an MCP user, I want the built wheel with the mcp extra started in CI, so that a packaging slip can't break `agent-eda-mcp`.
6. As a user who installed `mcp` without pyarrow, I want the install command instead of a traceback.
7. As a maintainer, I want the benchmark run on GitHub's 2-CPU ARM64 runners once now and in every release, so that the 5-second target is checked where it is meant to hold.
8. As a contributor who skips the local hooks, I want CI to run the file checks the hooks run.
9. As a maintainer, I want superseded pull-request runs cancelled and tag pushes not tested twice.

### A package fit for PyPI

10. As a visitor on PyPI, I want the project page to link to the repository, the issues and the changelog, with verified URLs, so that I can trust and follow it.
11. As a user, I want only files the package needs inside the wheel and the source distribution.
12. As a user outside AgentCore, I want to install only the dependencies the package uses.

### Releasing

13. As a maintainer, I want to release by pushing a version tag, so that a release is one command plus one approval.
14. As a maintainer, I want nothing published unless the full CI, benchmark included, passes on the tagged commit.
15. As a maintainer, I want the tag and the package version checked against each other before anything is built for upload.
16. As a maintainer, I want PyPI uploads to need my approval in a protected environment, and no long-lived token anywhere.
17. As a user, I want each release's files to carry build provenance and PEP 740 attestations, so that I can verify where they came from.
18. As a maintainer, I want release candidates to go to TestPyPI and a GitHub pre-release, so that I can rehearse without touching PyPI.
19. As a user, I want a GitHub release with the wheel, the source distribution and the release notes for every version on PyPI.
20. As an agent upgrading agent-eda, I want a changelog that says what changed for callers, so that I can adjust my code.
21. As a maintainer, I want a script that walks me through the one-time PyPI, TestPyPI and GitHub setup, so that I get every field right.

## Implementation Decisions

### ci.yml

- **Triggers:** `push` on branches only (`branches: ["**"]`), `pull_request`, `workflow_dispatch`, a weekly `schedule`, and `workflow_call` with a boolean input `full`.
- **Concurrency:** group `ci-${{ github.workflow }}-${{ github.ref }}`; superseded `pull_request` runs are cancelled.
- `UV_PYTHON_PREFERENCE: only-managed` for the whole workflow.
- **tests-latest** adds `macos-26` and `windows-2025` cells with `defaults.run.shell: bash`, for at least the oldest and newest supported Python on each. Tests that write text files pass `encoding="utf-8"`, and Windows-only branches get exercised. Whatever else fails on Windows is fixed in the code or the tests, and recorded in the ticket.
- **tests-newest** (schedule, dispatch and `full`) runs the suite with `--upgrade-package` for every runtime dependency and the mcp extra.
- **tests-agentcore** (ARM64) installs the AgentCore versions (numpy 1.26.4, pandas 2.3.1) over the locked environment and runs the suite.
- **lint** adds the pre-commit file checks: `SKIP=ruff-check,ruff-format,ty,pytest,uv-lock uvx pre-commit run --all-files`. The hooks in `.pre-commit-config.yaml` are pinned by commit SHA (`pre-commit autoupdate --freeze`).
- **packaging** builds the wheel and the source distribution with `uv build --force-pep517`. It runs `twine check --strict` and `check-wheel-contents`, installs both into a minimum-versions environment and runs `tests/smoke.py`.
- **packaging-mcp** installs the built wheel with `[mcp]` into a fresh environment and runs `tests/smoke_mcp.py`. The existing packaging job stays free of the extra.
- **benchmark** runs on `workflow_dispatch` or when `inputs.full` is true.
- **tests/test_ci_workflow.py** checks the parts of each command that matter, not exact strings (TEST-12). It also checks that every `uses:` in every workflow is pinned to a 40-character commit SHA.
- `.github/dependabot.yml` keeps the action pins current, weekly.

### Packaging

- **Metadata in `[project]`:**
  - `authors = [{ name = "Chris Lemke" }]`;
  - keywords;
  - classifiers: Development Status :: 3 - Alpha; Intended Audience :: Developers and Science/Research; Programming Language :: Python :: 3.12, 3.13 and 3.14; Topic :: Scientific/Engineering; Typing :: Typed. No `License ::` classifier, since the PEP 639 expression covers the license.
  - `[project.urls]` Homepage, Source, Issues and Changelog on `github.com/chrislemke/agent-eda`.
- **Build:** `[tool.uv.build-backend]` excludes `.DS_Store` from the source distribution and the wheel. A packaging test fails unless every file under `agent_eda/` in the wheel is `.py` or `py.typed`. `[build-system]` and the `uv-build` dev dependency move together to `uv_build>=0.13.1,<0.14`.
- **Dependencies:** they become `numpy>=1.26`, `pandas>=2.2`, `pydantic>=2.7` and `scipy>=1.11.2`. ADR 0005 gets a sentence: the required set is limited to those six libraries, and one of them is added back when an analysis first imports it.
- `agent_eda.mcp` checks for pyarrow along with the other modules of the extra. The packaging test blocks pyarrow as well as `mcp`.

### release.yml

- **Trigger and permissions:** pushing a tag `v*`; `permissions: {}` at the top, and each job gets only what it needs; `concurrency: release-${{ github.ref }}` without cancelling.
- **Tag routing:**

  | Tag | Index | GitHub release |
  | --- | --- | --- |
  | contains `rc` (`v0.1.0rc1`) | TestPyPI (`testpypi` environment) | pre-release |
  | `a` or `b` (`v0.1.0a1`) | PyPI (`pypi` environment) | pre-release |
  | final (`v0.1.0`) | PyPI (`pypi` environment) | latest release |

- **Jobs, in `needs:` order:**
  1. **ci**: `uses: ./.github/workflows/ci.yml` with `full: true`; `contents: read`.
  2. **build** (`contents: read`):
     - fails unless the tag equals `v` followed by `uv version --short`;
     - builds with `uv build --force-pep517` and uploads the `dist` artifact;
     - runs `twine check --strict` and `check-wheel-contents`;
     - smoke-tests installs from the wheel and from the source distribution;
     - outputs the version, the index and whether it is a pre-release.
  3. **provenance**: `actions/attest` on `dist/*`, with `id-token`, `attestations` and `artifact-metadata` all `write`.
  4. **publish**: the environment comes from the build job; only `id-token: write`; `pypa/gh-action-pypi-publish` alone in its job, with `repository-url` for TestPyPI. PEP 740 attestations are on by default.
  5. **github-release** (`contents: write`): `gh release create "$GITHUB_REF_NAME" dist/* --verify-tag` with the notes from `CHANGELOG.md`, adding `--prerelease --latest=false` for pre-releases. It runs after publishing, so a GitHub release exists only for files that are on an index.
- **Action pins** (looked up 2026-10-19; Dependabot keeps them current):

  | Action | Version | Commit SHA |
  | --- | --- | --- |
  | actions/checkout | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
  | astral-sh/setup-uv | v10.3.0 | `1c37ad07a6a961277cf70c0d37d6f313000f5884` |
  | actions/upload-artifact | v7.0.2 | `cf430e030ddbb5b0abf93d22962f4752f3646cd9` |
  | actions/download-artifact | v8.0.2 | `9000827ccba6bdab643e8b6fd33ac0654aef8333` |
  | actions/attest | v4.2.2 | `1e69f48acb82d1966a394da916b4c1698aa569d6` |
  | pypa/gh-action-pypi-publish | v1.14.2 | `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` |

- **Benchmark:** the `ci` job's `full` input makes the called CI run it. ci.yml's benchmark condition reads `inputs.full` as well as `github.event_name`, because a called workflow sees the caller's event.
- **Tests:** `tests/test_ci_workflow.py` checks that:
  - the file is named `release.yml`, since the trusted publisher names it;
  - the publish job's permissions are exactly `id-token: write`;
  - the environments are `pypi` and `testpypi`;
  - `ci` is called with `full: true`;
  - both workflows build with `--force-pep517`.
- **Lint:** actionlint and zizmor (auditor persona) are clean on both workflows. A finding that is deliberately accepted is recorded with its reason, for example zizmor's same-repository `uses:` note, because actionlint rejects the suggested syntax.
- **ADR 0007** records the shape: one workflow, why not two, trusted publishing, the tag routing and the approval gate.

### CHANGELOG.md

- Sections in the form `## [Unreleased]` and `## [0.1.0] - YYYY-MM-DD`. Each line says what changed for a caller.
- The release job uses the section whose heading is the tag's version. For pre-release tags it falls back to `## [Unreleased]`. A final tag without its own section fails the build job.
- AGENTS.md: a ticket whose change a caller can see adds a line under `## [Unreleased]`. Until 0.1.0, lines collect there and become the 0.1.0 section.

### The maintainer's one-time setup

- `scripts/release-setup-wizard.sh`, generated with the `mattpocock-skills:wizard` skill, walks the maintainer through the setup in this order:
  1. **PyPI pending publisher:** project `agent-eda`, owner `chrislemke`, repository `agent-eda`, workflow `release.yml`, environment `pypi`. A pending publisher doesn't reserve the name.
  2. **TestPyPI:** a separate account, and the same pending publisher with environment `testpypi`.
  3. **GitHub environment `pypi`:** required reviewer `chrislemke`, with "prevent self-review" off, and deployment tags `v*`.
  4. **GitHub environment `testpypi`:** deployment tags `v*`, no reviewers.
  5. **A tag ruleset for `v*`:** restrict creation, update and deletion, block force pushes, and let repository admins bypass.
  6. **Release immutability:** enabled.
  7. **Optionally:** allow only GitHub's own actions plus `astral-sh/setup-uv` and `pypa/gh-action-pypi-publish`, and require full SHA pins.

  Each step checks what it can with read-only `gh api` calls.
- The CONTRIBUTING.md "Releasing" section gives the release steps:
  1. `uv version <new>`;
  2. move the CHANGELOG lines;
  3. commit;
  4. `git tag -a vX.Y.Z -m "agent-eda X.Y.Z"` and push;
  5. approve the `pypi` deployment;
  6. check the results.

### Verifying that it all runs

- Each workflow change is pushed on a branch and opened as a pull request, then watched with `gh pr checks --watch`. The first macOS and Windows runs happen there.
- `gh workflow run ci.yml --ref main` runs the benchmark and the newest-dependencies job once, and `gh run watch` follows it.
- **Rehearsal** (`0.1.0rc1`):
  - TestPyPI shows the project;
  - `gh release view v0.1.0rc1` shows a pre-release with both files;
  - `gh attestation verify` passes on the wheel;
  - `pip install --no-deps --index-url https://test.pypi.org/simple/ agent-eda==0.1.0rc1` installs in an environment that already has the dependencies.
- **0.1.0:**
  - `pip install agent-eda==0.1.0` and `pip install 'agent-eda[mcp]'` work, and `agent-eda-mcp --help` runs;
  - the PyPI page shows the GitHub URLs as verified;
  - the provenance endpoint answers;
  - `gh release verify v0.1.0` passes.

## Testing Decisions

### The seam

For the package: the public API and the built artifacts, installed in throwaway environments as `tests/test_packaging.py` does. For the workflows: the YAML files read by `tests/test_ci_workflow.py`, and real runs on GitHub. A workflow is only done when it has run green on GitHub, and the ticket records the run's URL.

### What makes a good test here

A workflow test checks the property that matters, such as a permission, a pin or a flag that must be present. It doesn't check a whole command string, so harmless edits stay green.

### Test groups

1. **Workflow structure:** the triggers, the jobs, permissions, environments, SHA pins and `--force-pep517`.
2. **Packaging:** only `.py` and `py.typed` files are packed; the metadata has its URLs and classifiers; the mcp extra check covers pyarrow; the dependency list matches ADR 0005.
3. **Real runs:** CI on a pull request, the dispatched full CI, the rehearsal on TestPyPI, and the 0.1.0 release.

## Out of Scope

- Publishing to conda-forge or an MCP registry, and MCPB bundles.
- Versioning from git tags: uv_build has no VCS versioning, and a static version plus the tag check is simpler.
- release-please and semantic-release: they need conventional commits, and this repository's commits are named "NN: <ticket title>".
- Signing with long-lived keys: trusted publishing and attestations cover provenance.
- Docker images.
- Making every pull request wait for the benchmark: it runs in releases and on demand.

## Further Notes

- The plan was settled in a grilling session on 2026-10-19. The decisions behind this spec are round 1, Q2 to Q4 and Q10, round 2, Q8 and Q11, and round 3, Q2.
- Rejected release designs, with their reasons, are in REL's report in `.scratch/audit/findings.md`:
  - a separate `publish.yml` started by a published release;
  - publishing from a reusable workflow;
  - `workflow_run`;
  - checking for a green CI run through the checks API;
  - `uv publish` with `astral-sh/attest-action`;
  - API tokens;
  - building inside the publish job.
- Once `0.1.0a1` is the only version on PyPI, `pip install agent-eda` and uv will install it, because installers fall back to pre-releases when no final release exists. Ticket 07 waits for the maintainer to confirm the name claim with that in mind.
