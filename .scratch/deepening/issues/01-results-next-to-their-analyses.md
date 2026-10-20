# 01: Results next to their analyses

What to build:
- `agent_eda.results` becomes a lazy facade;
- the shared result base moves to `_result_base.py`, and the discovery records move next to discovery;
- each analysis's result types move into a private sibling of the analysis;
- a byte baseline proves nothing changed.

Read these sections of `.scratch/deepening/spec.md`: Results next to their analyses, the Testing Decisions (the baseline), and test groups 1 and 2. This ticket covers user stories 1 and 9. Findings: DES-5, STD-56.

Blocked by: none

Status: done

- [x] Before any move, a script under `scripts/` records the baseline:
  - `to_json()`, `str(res)` and every `to_frame()` table for every recipe on `EDGE_FRAMES` and a fixed set of generated frames;
  - every message in the message tables.

  A test compares against the baseline, and the Comments record how to refresh it.
- [x] `agent_eda.results` is a lazy facade with an export table. `ae.results.<Name>` and `from agent_eda.results import <Name>` work for every public result name.
- [x] A sync test checks the export table against every `AnalysisResult` subclass.
- [x] The shared base lives in `_result_base.py`, the discovery records next to discovery, and each analysis's result types in a private sibling of its analysis module.
- [x] A pickle of each result class, written with 0.1.0 and kept as a test fixture, loads and gives an equal record.
- [x] `tests/test_package.py`'s `__module__` check follows the new layout.
- [x] The baseline comparison passes unchanged.
- [x] The finish checks pass.

## Comments

- The baseline is `scripts/record_baseline.py`, recorded before any move, one file per pandas major version (`tests/baseline/pandas2.json`, `pandas3.json`). Each holds the SHA-256 of `to_json()`, `str(res)` and every `to_frame()` table of every analysis on `EDGE_FRAMES`, `LABEL_EDGE_FRAMES` and four seeded generated frames (default call, and for a masking analysis a call that masks every column), the discovery records, the MCP tool definitions and one call of each tool, and the code and full message of every row of the three `MISTAKES` tables. `tests/test_baseline.py` records again and compares. To refresh, only where a ticket says why: `uv run python scripts/record_baseline.py` (pandas 3), and `uv run --isolated --resolution lowest-direct --extra mcp --python 3.12 python scripts/record_baseline.py` (pandas 2). `--check` compares without writing; `--dump DIR` writes every full text, to diff before and after.
- The generated frames use a seeded NumPy generator, not Hypothesis, so a Hypothesis version never changes the baseline. Without the mcp extra (pydantic 2.7) the test skips the MCP records and the `json` part of `describe_analysis`, which holds pydantic's JSON Schema; the other two runs cover them.
- Layout: `_result_base.py` (base, caps, shared types and `share_text`, which was `_share` in the profile_missing_values result and is used by two texts), `_discovery_results.py` (the discovery records, next to `discovery.py`), and `data/_{structure,duplicates,missingness,targets,validation}_results.py` and `statistics/_{descriptive,association}_results.py`. `_numeric_frame` became `numeric_frame`, since `profile_target`'s result imports it from the descriptive result module. Internal code imports from the defining modules, not the facade.
- `ae.results` is a lazy facade with `_EXPORTS` and a `TYPE_CHECKING` block (type checkers see every name; a typo stays an error) and `__all__`. It also resolves `_ColumnsByName` and `_ColumnList`, which a 0.1.0 pickle names, through a separate `_PICKLED` table that `dir()` leaves out. `row_id_json` stays importable from it. `tests/test_results_facade.py` is the sync test (every `AnalysisResult` and public `ResultModel` subclass of the package is given by name) and checks the facade stays lazy.
- Pickles: `tests/fixtures/pickles/numpy1` and `numpy2` hold a pickle and its `to_json()` per result class, 14 classes each, written with the pre-move code (the 0.1.0 layout, `agent_eda.results.<Name>`). They are per NumPy major since a result holds arrays and DataFrames that a NumPy 2 pickle can't give back on NumPy 1. The NumPy 1 ones came from a built copy of the integration tip on the minimum versions. `scripts/record_pickles.py` adds a class; never rerun it to refresh a pickle.
- `__module__` of a result class is now its private module; `tests/test_package.py` asserts that it is a private module in the package of its analysis. `agent_eda/_codes.py` reads the code comments from `_result_base`. `tests/test_suite_rules.py` read `vars(ae.results)`, which the facade leaves empty, and would have passed without checking anything; it now reads `dir(ae.results)`.
- Updated docs: AGENTS.md (Layout, status reason codes), `docs/agents/new-analysis.md` step 4 (the sibling module and the facade table) and the CHANGELOG. The README needed no change and stays at its size.
- Ran locally: ruff, ty, `uv run pytest`, both minimum-version runs, the MCP stdio smoke, the benchmark (unchanged: no analysis code moved). Nothing is GitHub-only.
