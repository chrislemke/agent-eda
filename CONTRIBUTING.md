# Contributing to agent-eda

The project uses [uv](https://docs.astral.sh/uv/), 0.13 or later: `pyproject.toml` makes older versions refuse to run. [AGENTS.md](AGENTS.md) is the only place that lists the commands: the setup and everyday ones under Commands, and the checks to run before finishing a change under Finish checks. They include the two minimum-versions runs, the MCP smoke test, which starts `agent-eda-mcp` over real stdio and fails rather than skips when the mcp extra is missing, and the benchmark. Without the extra, the MCP test modules skip and say how to install it.

[AGENTS.md](AGENTS.md) also has the rules every analysis follows, and [GLOSSARY.md](GLOSSARY.md) the terms to use in code, docstrings and messages. The analyses still to build are in [docs/planned-analyses.md](docs/planned-analyses.md).

## Development notes

- **The MCP server in Claude Code.** The repository's `.mcp.json` runs `uv run --extra mcp agent-eda-mcp`, so Claude Code started in a clone offers the analyses of your checkout as MCP tools once you approve the MCP server. Restart the session to pick up a changed analysis.
- **Property tests.** Hypothesis calls every analysis on generated DataFrames full of edge cases. It runs derandomized: 50 examples per analysis, or 1,000 with the `thorough` profile.
- **Benchmark.** It times every analysis with default arguments on a deterministic DataFrame of 1,000,000 rows and 100 mixed columns, with BLAS and pyarrow limited to 2 threads, and fails any analysis that takes 5 seconds or more. The default test run skips it.
- **README.** `tests/test_readme.py` runs the README's python examples in order and fails when a `text` block right after one differs from what it prints. It also checks that every analysis has one row in the Analyses tables, in the first-look table or in "Other analyses"; the first-look table against `ae.guide()`; that the README has no relative link; the Reference against the code: every public name, each signature, each analysis's arguments, finding codes and statuses, and the error codes; and that no shipped analysis is still listed in `docs/planned-analyses.md`.
- **New analyses** need a recipe in `tests/recipes.py`, or the suite fails and says so. A shipped analysis moves from `docs/planned-analyses.md` to the README's Analyses section: into the first-look table when `ae.guide()` makes it a step of the first look, otherwise into the "Other analyses" table under its module.

## Continuous integration

The pre-commit hooks run ruff, ty and file checks on every commit, and the tests on `git push`. On GitHub, `.github/workflows/ci.yml` runs lint, type checks and the tests, the MCP server tests included, on Python 3.12 to 3.14 on Linux x86_64, and on Python 3.12 on Linux ARM64. It also runs both minimum-versions runs, the MCP smoke test on Python 3.12 and 3.14, and a packaging job that installs the wheel offline with only the required dependencies. The benchmark runs only when started by hand from the Actions tab.
