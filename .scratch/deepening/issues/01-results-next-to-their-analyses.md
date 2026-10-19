# 01: Results next to their analyses

What to build:
- `agent_eda.results` becomes a lazy facade;
- the shared result base moves to `_result_base.py`, and the discovery records move next to discovery;
- each analysis's result types move into a private sibling of the analysis;
- a byte baseline proves nothing changed.

Read these sections of `.scratch/deepening/spec.md`: Results next to their analyses, the Testing Decisions (the baseline), and test groups 1 and 2. This ticket covers user stories 1 and 9. Findings: DES-5, STD-56.

Blocked by: none

Status: ready-for-agent

- [ ] Before any move, a script under `scripts/` records the baseline:
  - `to_json()`, `str(res)` and every `to_frame()` table for every recipe on `EDGE_FRAMES` and a fixed set of generated frames;
  - every message in the message tables.

  A test compares against the baseline, and the Comments record how to refresh it.
- [ ] `agent_eda.results` is a lazy facade with an export table. `ae.results.<Name>` and `from agent_eda.results import <Name>` work for every public result name.
- [ ] A sync test checks the export table against every `AnalysisResult` subclass.
- [ ] The shared base lives in `_result_base.py`, the discovery records next to discovery, and each analysis's result types in a private sibling of its analysis module.
- [ ] A pickle of each result class, written with 0.1.0 and kept as a test fixture, loads and gives an equal record.
- [ ] `tests/test_package.py`'s `__module__` check follows the new layout.
- [ ] The baseline comparison passes unchanged.
- [ ] The finish checks pass.

## Comments
