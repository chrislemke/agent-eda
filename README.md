# agent-eda

Deterministic exploratory data analysis (EDA) functions for LLM agents.

> Status: in development. `describe_dataset` and `list_analyses` work; the rest of the code below shows the planned interface.

agent-eda is a Python library for the EDA phase of a data science project: data quality, missing values, duplicates, distributions, associations, leakage checks and more. Each analysis is a plain function. It takes a pandas DataFrame and typed arguments and returns a typed result. The library never calls an LLM or the network, so the same data, arguments and seed always give the same result.

Three kinds of callers matter equally:

- agents running in AWS Bedrock AgentCore that use its Code Interpreter
- coding agents such as Claude Code
- people, who install and import it like any other package

## Planned usage

```python
import agent_eda as ae

res = ae.summarize_numeric(df, columns=["age", "income"])
print(res)  # short text: status, key numbers, findings, limitations
res.to_frame()  # the full table as a DataFrame

ae.guide()  # where to start
ae.search_analyses("compare two groups")  # find an analysis by keyword
```

You don't need any setup to call an analysis. An optional `Project` declares dataset roles (train, test, holdout and so on) and a data policy for the analyses that check them.

## Installation

Nothing is on PyPI yet. Once it is released:

**People and coding agents**

```sh
pip install agent-eda
```

**AgentCore Runtime.** Add `agent-eda` to the agent's container image or code zip. AWS's example zip command uses the `manylinux2014` platform tag, which installs old numpy and pandas releases. Use `--python-platform aarch64-manylinux_2_28` to get current ones.

**AgentCore Code Interpreter.** In public network mode, run `pip install agent-eda` in the session. In sandbox mode the session can't reach PyPI. Upload the wheel and run `pip install --no-deps <wheel>`, or, if pip is blocked, put the wheel file on `sys.path` and import from it directly. Installs don't carry over to a new session. Code Interpreter only returns printed output to the agent, so `print()` results.

## Design documents

- [GLOSSARY.md](GLOSSARY.md) defines the terms used in code and docs.
- [docs/adr/](docs/adr/) records the decisions that are hard to reverse, and why.
- [docs/eda-python-implementation-catalogue.md](docs/eda-python-implementation-catalogue.md) lists the planned analyses.
- [AGENTS.md](AGENTS.md) holds the rules for anyone writing code here, people or agents.

## Development

Install the project with its dev tools:

```sh
uv sync
uv run pre-commit install
```

| Task | Command |
| --- | --- |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run ty check` |
| Test | `uv run pytest` |
| Test on the minimum versions | `uv run --isolated --resolution lowest-direct --python 3.12 pytest` |
| Test with coverage | `uv run pytest --cov` |
| Run all hooks | `uv run pre-commit run --all-files` |

The minimum-versions run installs every direct dependency at the lowest version `pyproject.toml` allows (numpy 1.26, pandas 2.2, pydantic 2.7, scipy 1.11.2, scikit-learn 1.5, statsmodels 0.14.1, and the dev tools' lower bounds) into a throwaway environment on Python 3.12, the newest Python with numpy 1.26 wheels. It leaves `uv.lock` and `.venv` unchanged. Keep `--isolated`: without it, uv rewrites `uv.lock` for the new resolution mode.

The packaging tests build the wheel offline with the `uv_build` backend from the dev dependencies, pinned to the range in `[build-system]`, so they need no network and work with any uv version.

On every commit, pre-commit checks the files (whitespace, line endings, YAML/TOML/JSON syntax, merge conflict markers, private keys, files over 1 MB), keeps `uv.lock` in sync, and runs ruff and ty. The tests run on `git push`.

## License

MIT. See [LICENSE](LICENSE).
