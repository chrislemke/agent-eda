# 03: Packaging fit for PyPI

What to build: complete metadata, nothing stray in the files, only the dependencies the package uses, a checked source distribution, a checked wheel with the mcp extra, and the install command instead of a traceback when pyarrow is missing.

Read these sections of `.scratch/release/spec.md`: Packaging, the packaging jobs under ci.yml, and test group 2. This ticket covers user stories 4 to 6 and 10 to 12. Findings: REL-1, REL-2, REL-3, REL-11, STD-7, MCP-15, and SPEC's questionable decision on unused dependencies.

Blocked by: 02

Status: ready-for-agent

- [ ] `[project]` has authors, keywords, the classifiers the spec lists, and `[project.urls]` Homepage, Source, Issues and Changelog.
- [ ] `.DS_Store` files are excluded from both artifacts. A packaging test fails when any file under `agent_eda/` in the wheel is neither `.py` nor `py.typed`. Check that test once against a wheel built from a working copy that contains a `.DS_Store`.
- [ ] `[build-system]` and the `uv-build` dev dependency both say `uv_build>=0.13.1,<0.14`. The build no longer warns about a backend version mismatch.
- [ ] The required dependencies are numpy, pandas, pydantic and scipy.
- [ ] ADR 0005 has a sentence saying that one of the six libraries is added back when an analysis first imports it.
- [ ] The README's install text names the four required libraries.
- [ ] Nothing imports scikit-learn or statsmodels, and a test checks the dependency list.
- [ ] The packaging job builds both artifacts with `uv build --force-pep517`, runs `twine check --strict` and `check-wheel-contents`, and installs both into a minimum-versions environment before running `tests/smoke.py`.
- [ ] The `packaging-mcp` job installs the wheel with `[mcp]` into a fresh environment and runs `tests/smoke_mcp.py`.
- [ ] `agent_eda.mcp` checks for pyarrow along with the other modules of the extra. With pyarrow blocked, `agent-eda-mcp` exits 1 with the install command on stderr, and the packaging test checks this.
- [ ] A pull-request run shows the packaging jobs green. The Comments record its URL.
- [ ] The finish checks pass.

## Comments
