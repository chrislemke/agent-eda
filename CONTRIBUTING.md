# Contributing to agent-eda

The project uses [uv](https://docs.astral.sh/uv/), 0.13 or later: `pyproject.toml` makes older versions refuse to run. [AGENTS.md](AGENTS.md) is the only place that lists the commands: the setup and everyday ones under Commands, and the checks to run before finishing a change under Finish checks. They include the two minimum-versions runs, the MCP smoke test, which starts `agent-eda-mcp` over real stdio and fails rather than skips when the mcp extra is missing, and the benchmark. Without the extra, the MCP test modules skip and say how to install it.

[AGENTS.md](AGENTS.md) also has the rules every analysis follows, and [GLOSSARY.md](GLOSSARY.md) the terms to use in code, docstrings and messages. The analyses still to build are in [docs/planned-analyses.md](docs/planned-analyses.md).

## Development notes

- **The MCP server in Claude Code.** The repository's `.mcp.json` runs `uv run --extra mcp agent-eda-mcp`, so Claude Code started in a clone offers the analyses of your checkout as MCP tools once you approve the MCP server. Restart the session to pick up a changed analysis.
- **Property tests.** Hypothesis calls every analysis on generated DataFrames full of edge cases. It runs derandomized: 50 examples per analysis, or 1,000 with the `thorough` profile.
- **Benchmark.** It times every analysis with default arguments on a deterministic DataFrame of 1,000,000 rows and 100 mixed columns, with BLAS and pyarrow limited to 2 threads, and fails any analysis that takes 5 seconds or more. The default test run skips it.
- **README.** `tests/test_readme.py` runs the README's python examples in order and fails when a `text` block right after one differs from what it prints. It also checks that every analysis has one row in the Analyses tables, in the first-look table or in "Other analyses"; the first-look table against `ae.guide()`; that the README has no relative link; that the Reference is current; and that no shipped analysis is still listed in `docs/planned-analyses.md`.
- **The Reference.** The README's Reference, between `<!-- reference:start -->` and `<!-- reference:end -->`, is generated from the registry, the signatures, the docstrings and `ae.errors` by `scripts/generate_reference.py`, so never edit it by hand. After changing a docstring, a signature, a result field description or an error code, run the generator and its `--check` form from AGENTS.md's Commands and commit the README; the test fails when it is stale and names the command.
- **New analyses** follow the checklist in [docs/agents/new-analysis.md](https://github.com/chrislemke/agent-eda/blob/main/docs/agents/new-analysis.md). Tests fail for a public function in `agent_eda.data` or `agent_eda.statistics` that isn't registered or exported, for export lists in `agent_eda/__init__.py` that disagree, for an error code missing from the `ae.errors` docstring or the errors table, and for an analysis without a recipe in `tests/recipes.py`; each says what to fix. A shipped analysis moves from `docs/planned-analyses.md` to the README's Analyses section: into the first-look table when `ae.guide()` makes it a step of the first look, otherwise into the "Other analyses" table under its module.

## Continuous integration

The pre-commit hooks run ruff, ty and file checks on every commit, and the tests on `git push`. On GitHub, `.github/workflows/ci.yml` runs lint, type checks and the pre-commit file checks, so a contributor without the local hooks gets them too. It runs the tests, the MCP server tests included, on Python 3.12 to 3.14 on Linux x86_64, on Python 3.12 on Linux ARM64, and on Python 3.12 and 3.14 on macOS and Windows. It also runs both minimum-versions runs, the tests on the AgentCore versions of numpy and pandas (ADR 0005) on Python 3.12 on Linux ARM64, the MCP smoke test on Python 3.12 and 3.14, and two packaging jobs: one builds the wheel and the source distribution, checks them with `twine check --strict` and `check-wheel-contents`, and installs each into an environment of only the required dependencies at their minimum versions, and the other installs the built wheel with the `mcp` extra into a fresh environment and starts the MCP server over stdio. Every job uses the Python builds that uv installs, not the runner's own.

Every Monday, and when started by hand, CI also tests the newest release of every dependency, which is what a fresh install gets. The benchmark runs only when started by hand from the Actions tab, and in every release. A superseded pull-request run is cancelled. A push of a tag doesn't start this workflow: the release workflow calls it. The actions are pinned by commit SHA, and Dependabot proposes newer ones every week.

## Releasing

A version tag starts `.github/workflows/release.yml`: the full CI with the benchmark, the build, build provenance, the upload to PyPI, and then the GitHub release with the wheel, the source distribution and the notes from `CHANGELOG.md`. The upload waits for your approval. [docs/adr/0007](https://github.com/chrislemke/agent-eda/blob/main/docs/adr/0007-one-tag-started-workflow-publishes-then-releases.md) says why it is one workflow and how it is secured.

Once, before the first release: run `bash scripts/release-setup-wizard.sh`. It walks through the trusted publishers on PyPI and TestPyPI, the GitHub environments `pypi` and `testpypi`, the tag ruleset and release immutability, and checks what it can.

For each release:

1. Set the version: `uv version <new>`, such as `uv version 0.1.0rc1`.
2. Move the lines under `## [Unreleased]` in `CHANGELOG.md` into a section headed `## [<new>] - YYYY-MM-DD`. A final version needs its own section, a pre-release takes Unreleased when it has none.
3. Commit both files, and merge them to `main`.
4. Tag that commit and push the tag: `git tag -a v<new> -m "agent-eda <new>"` and `git push origin v<new>`. The tag must be `v` plus the version in `pyproject.toml`, or the build job stops.
5. When the CI and the build have passed, approve the `pypi` deployment (the Actions run asks for it). A release candidate, such as `0.1.0rc1`, goes to TestPyPI and a GitHub pre-release, and `a` and `b` versions go to PyPI as pre-releases.
6. Check the results: the project page on PyPI, `gh release view v<new>`, and `gh attestation verify <wheel> --repo chrislemke/agent-eda` on the downloaded wheel.

A job that fails before the upload leaves nothing published: fix the cause, move the tag with `git tag -f` and `git push -f origin v<new>` (the tag ruleset lets an admin do so), or delete the tag and make a new one. PyPI never takes a file name twice, so once the upload has succeeded, a mistake needs a new version.
