# 18: Contract checks for every analysis

What to build: `check_contract` enforces every result rule in AGENTS.md for every recipe and generated frame, and the suite's own safety nets get stronger, so that a new analysis can't break a rule silently.

Read these sections of `.scratch/correctness/spec.md`: Contract checks for every analysis, and test group 12. This ticket covers user story 30. Findings: TEST-3, TEST-4, TEST-7, TEST-9, TEST-10, TEST-11, TEST-15, STD-22, STD-60, STD-61, STD-62, STD-63, STD-64.

Blocked by: none

Status: ready-for-agent

- [ ] `check_contract` asserts that:
  - every list in the record keeps to its cap and carries its `_total`;
  - every finding and limitation field path resolves on the result object;
  - every finding's assessment equals the one its docstring documents;
  - `to_dict()` equals `json.loads(to_json())`;
  - the record holds no timestamp or HTML.
- [ ] Each new assertion fails against a mutant once. The Comments record which mutant.
- [ ] `infer_column_semantics`' `constant_columns` and `all_missing_columns` findings are asserted directly (TEST-3).
- [ ] A result that fills every section prints exactly 40 lines, with the right count of what it leaves out (TEST-4).
- [ ] `tests/test_contract.py` checks that each Args entry gives a type, its Literal choices and what None means. It resolves See Also entries against `ae.list_analyses()`, and runs each doctest once (TEST-9).
- [ ] The one-seam guard catches `importlib.import_module("agent_eda._x")`, `getattr(ae, "_x")` and module names in strings (TEST-10).
- [ ] The meta-test for generated frames covers ties and mixed types, and its comments match the code (TEST-11).
- [ ] The tests that can't fail are fixed, such as the masking case of `summarize_categorical` and an empty first-look table (TEST-15, STD-60). STD-61's coverage gaps and STD-62's implicit tolerances are fixed.
- [ ] Generated data shares no mutable objects across examples and orders no set by hash (STD-63).
- [ ] `anyio` and `jsonschema` are declared in a dependency group (STD-64).
- [ ] Every `# noqa` and `# ty: ignore` carries its reason on the same line, and a test checks this (STD-22).
- [ ] The finish checks pass.

## Comments
