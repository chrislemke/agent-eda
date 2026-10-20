# 03: Packaging fit for PyPI

What to build: complete metadata, nothing stray in the files, only the dependencies the package uses, a checked source distribution, a checked wheel with the mcp extra, and the install command instead of a traceback when pyarrow is missing.

Read these sections of `.scratch/release/spec.md`: Packaging, the packaging jobs under ci.yml, and test group 2. This ticket covers user stories 4 to 6 and 10 to 12. Findings: REL-1, REL-2, REL-3, REL-11, STD-7, MCP-15, and SPEC's questionable decision on unused dependencies.

Blocked by: 02

Status: done

- [x] `[project]` has authors, keywords, the classifiers the spec lists, and `[project.urls]` Homepage, Source, Issues and Changelog.
- [x] `.DS_Store` files are excluded from both artifacts. A packaging test fails when any file under `agent_eda/` in the wheel is neither `.py` nor `py.typed`. Check that test once against a wheel built from a working copy that contains a `.DS_Store`.
- [x] `[build-system]` and the `uv-build` dev dependency both say `uv_build>=0.13.1,<0.14`. The build no longer warns about a backend version mismatch.
- [x] The required dependencies are numpy, pandas, pydantic and scipy.
- [x] ADR 0005 has a sentence saying that one of the six libraries is added back when an analysis first imports it.
- [x] The README's install text names the four required libraries.
- [x] Nothing imports scikit-learn or statsmodels, and a test checks the dependency list.
- [x] The packaging job builds both artifacts with `uv build --force-pep517`, runs `twine check --strict` and `check-wheel-contents`, and installs both into a minimum-versions environment before running `tests/smoke.py`.
- [x] The `packaging-mcp` job installs the wheel with `[mcp]` into a fresh environment and runs `tests/smoke_mcp.py`.
- [x] `agent_eda.mcp` checks for pyarrow along with the other modules of the extra. With pyarrow blocked, `agent-eda-mcp` exits 1 with the install command on stderr, and the packaging test checks this.
- [x] A pull-request run shows the packaging jobs green. The Comments record its URL.
- [x] The finish checks pass.

## Comments

- Pull-request run, all jobs green, both packaging jobs among them: https://github.com/chrislemke/agent-eda/actions/runs/39418566364 (draft PR #4, commit `45ca71f`).
  - `packaging, wheel and source distribution` built both files with `uv build --force-pep517`, passed `twine check --strict` on both and `check-wheel-contents` on the wheel, ran the 25 packaging tests against the built wheel, and installed the wheel (`pip install --no-deps --no-index`) and the source distribution (`pip install --no-deps`, which builds it) into environments with numpy 1.26.0 and pandas 2.2.0: `pip check` clean, `tests/smoke.py` printed "11 analyses ran 90 examples, 0 failed" for each.
  - `packaging, wheel with the mcp extra` installed `agent_eda-0.1.0rc1-py3-none-any.whl[mcp]` into a fresh environment (mcp 2.3.0, numpy 2.5.4, pandas 3.0.6, pyarrow 26.0.0), and the installed server listed 13 MCP tools and ran `describe_dataset` over stdio.
- The `.DS_Store` check, twice. A packaging test builds a wheel and a source distribution from a copy of the project that has a `.DS_Store` in `agent_eda/` and in `agent_eda/data/`: it failed with four stray entries before the exclusion and passes with it, whatever the maintainer's Finder has left in the clone. By hand, with the two files in the real working tree, plain `uv build` and `uv build --force-pep517` each gave zero `.DS_Store` entries in both files (I removed the files afterwards). The globs are `**/.DS_Store`, to reach the nested folders. A second test fails unless every file under `agent_eda/` in the wheel is `.py` or `py.typed`.
- `uv build` with the bundled backend (0.13.1) and with `--force-pep517` (uv_build 0.13.3) print no backend-version warning, and the `[build-system]` range, the dev dependency and `uv.lock` agree (`uv-build` 0.10.12 became 0.13.3).
- Dependencies: `uv.lock` drops scikit-learn, statsmodels and the packages only they needed (cloudpickle, formulaic, interface-meta, joblib, narwhals, patsy, wrapt); no other locked version changed. A test walks every module of the package and requires that the third-party names it imports are exactly the required libraries, the mcp extra's modules, and what comes with them: `pydantic_core` and `annotated_types` (pydantic installs both) and `rich` (imported only by the Rich display hook, when Rich is already running). It also requires every required library to appear in ADR 0005, which says a new one needs its own ADR first.
- The STD-7 bug was larger than the ticket says, and `_EXTRA_MODULES` was not the cause. `agent_eda.mcp` imported `agent_eda/_mcp/instructions.py`, which imported `agent_eda/_mcp/data_files.py` for two rule texts, and that module imports pyarrow at the top. So `import agent_eda.mcp` failed, before `main` ran, in every environment without pyarrow: a plain `pip install agent-eda` ended in a traceback instead of the install command, and `agent-eda-mcp --help` failed too. The tests didn't show it because the development environment always has pyarrow, and the old test blocked only `mcp`.
  - The two rule texts (`CSV_RULES`, `PARQUET_RULES`) moved to `agent_eda/_mcp/local_files.py`, which needs nothing from the extra, and `pyarrow` joined `_EXTRA_MODULES`, so a half-installed extra (mcp without pyarrow) is named too. Text, tool definitions and the byte baseline are unchanged.
  - The test now runs the console script with all of the extra blocked (`mcp`, `anyio`, `pyarrow`), with `mcp` alone and with `pyarrow` alone: each exits 1 with both install commands and no traceback. A fourth test checks that `agent-eda-mcp --help` prints its help without the extra. Without the source change, the all-blocked, `pyarrow` and `--help` cases fail.
  - In the subprocess, the DeprecationWarning of pandas 2.2 ("Pyarrow will become a required dependency") is let by: pandas raises it whenever pyarrow can't be imported, which is what these runs arrange, and both minimum-versions runs failed on it before.
- Test code: `tarfile.open` looks like a text open to the suite's encoding rule, so the test reads the source distribution through `gzip.GzipFile` and `tarfile.TarFile`.
- CHANGELOG.md has a "Packaging" section and a line under "Fixed" for the three changes a caller can see (the dependencies and metadata, the `.DS_Store` files, and the `agent-eda-mcp` message).
- Finish checks on the tree of the last commit: ruff, ruff format and ty clean; 3,986 passed on the latest versions; 3,430 (without the extra) and 3,986 (with it) on the minimum versions; the stdio MCP smoke test exits 0; `generate_reference.py --check` and `record_baseline.py --check` pass. The benchmark wasn't run locally for this ticket: no analysis changed, and it ran green on GitHub (ticket 02).
