# 18: Contract checks for every analysis

What to build: `check_contract` enforces every result rule in AGENTS.md for every recipe and generated frame, and the suite's own safety nets get stronger, so that a new analysis can't break a rule silently.

Read these sections of `.scratch/correctness/spec.md`: Contract checks for every analysis, and test group 12. This ticket covers user story 30. Findings: TEST-3, TEST-4, TEST-7, TEST-9, TEST-10, TEST-11, TEST-15, STD-22, STD-60, STD-61, STD-62, STD-63, STD-64.

Blocked by: none

Status: done

- [x] `check_contract` asserts that:
  - every list in the record keeps to its cap and carries its `_total`;
  - every finding and limitation field path resolves on the result object;
  - every finding's assessment equals the one its docstring documents;
  - `to_dict()` equals `json.loads(to_json())`;
  - the record holds no timestamp or HTML.
- [x] Each new assertion fails against a mutant once. The Comments record which mutant.
- [x] `infer_column_semantics`' `constant_columns` and `all_missing_columns` findings are asserted directly (TEST-3).
- [x] A result that fills every section prints exactly 40 lines, with the right count of what it leaves out (TEST-4).
- [x] `tests/test_contract.py` checks that each Args entry gives a type, its Literal choices and what None means. It resolves See Also entries against `ae.list_analyses()`, and runs each doctest once (TEST-9).
- [x] The one-seam guard catches `importlib.import_module("agent_eda._x")`, `getattr(ae, "_x")` and module names in strings (TEST-10).
- [x] The meta-test for generated frames covers ties and mixed types, and its comments match the code (TEST-11).
- [x] The tests that can't fail are fixed, such as the masking case of `summarize_categorical` and an empty first-look table (TEST-15, STD-60). STD-61's coverage gaps and STD-62's implicit tolerances are fixed.
- [x] Generated data shares no mutable objects across examples and orders no set by hash (STD-63).
- [x] `anyio` and `jsonschema` are declared in a dependency group (STD-64).
- [x] Every `# noqa` and `# ty: ignore` carries its reason on the same line, and a test checks this (STD-22).
- [x] The finish checks pass.

## Comments

- `check_contract` (tests/test_properties.py) now ends in `check_record`: the record validates against `record_schema()` with `jsonschema` (so every capped field has its `_total`, and no field is undeclared), every list or mapping keeps to its cap by name (row IDs 10, example values 5, anything else 100) and a list at its cap, or cut short of its total, must carry the total; every finding and limitation path resolves on the result object (`columns['age'].missing_count`, `columns[2]`, `quantiles[0.5]` and plain fields); each finding's code is declared and its assessment equals the word the docstring's Returns bullet starts with (`describe_analysis(name).finding_codes`); `to_dict() == json.loads(to_json())`; no result field is named like a time (timestamp, created, started, duration, ...), none has a date or time type, every `execution.versions` value reads as a version, and IPython's display formatter gives only `text/plain`.
- Mutants, each killed by the new assertion and then reverted: `_ROW_IDS = _Cap(11)` (over its cap); `<name>_total` one too high (total doesn't fit); a misspelt path in describe_dataset's findings; `constant_columns` given the assessment `descriptive` (the contract check and the direct test of TEST-3); `all_missing_columns` pointing at `missing_count` (the direct test); `to_json` writing `method: "x"`; a field `started_at` on `Execution`; a fixed ISO timestamp in `execution.versions`; `_repr_html_` on the result; `hidden = len(lines) - limit` in the text cut; `_json_text` writing `str(value)` in a table file; an Args entry without its type or with another type; a dropped `Literal` choice; a dropped sentence on what `None` means; a non-analysis in See Also.
- The edge frame `many rows` (40 rows) is new, since frames of 8 rows never reach the cap of 10 row IDs. The kinds `str (python storage)`, `period[M]` and `interval[float64]` are new too (STD-61). They shifted which examples Hypothesis draws, which surfaced two reference problems in old tests, both fixed in the tests: the pandas reference for duplicates and key counts treated a NaN stored as a value as different from a missing value, where every analysis counts it as missing (`strategies.nan_as_missing`), and `has_countable_values` took period and interval columns for ones summarize_categorical counts.
- TEST-4: a result with a 12-line table, 8 findings, 7 limitations and a reason that wraps onto 10 to 30 lines prints 39 or 40 lines uncut, and from 41 lines on exactly 40 with "... N more lines; see res.to_json()." where N is the uncut length minus 39.
- TEST-9: Args entries must give "(type)", which equals the type `describe_analysis` shows (apart from the `pd.` prefix); every choice of a `Literal` appears in the entry; an entry whose type has `None` says what it means. See Also entries are checked against `ae.list_analyses()`, except `list_analyses` itself, which all 11 docstrings still list (STD-21). The set `SEE_ALSO_UNTIL_08` in tests/test_contract.py lets that one name through until agent-experience ticket 08 drops it from the docstrings; that ticket deletes the set. The doctest of each analysis now runs once (test_contract), and test_describe_analysis only checks that `example` is the docstring's section.
- TEST-10: the guard (`private_uses` in tests/test_suite_rules.py) reads imports, private attributes of anything imported from `agent_eda`, `getattr(..., "_x")` on such names and private module names inside strings or paths; its own tests feed it snippets it must flag and snippets it must leave alone. `tests/serving.py` may reach `_create_server`, the one hook for stand-in analyses. test_packaging no longer writes the private MCP module's name: it recognises MCP modules by the last part of their name.
- STD-60 and TEST-15: a stand-in in the ID column of detect_missing_sentinels, counts asserted before the loops over duplicate groups and key sets, the written cells of every MCP table compared with `to_frame()` after the documented conversions (JSON text, the index first), `status == "completed"` in the benchmark, the exact masking set (six analyses, matched against `shows_raw_values`), the test of DataPolicy names no longer claims to follow `columns` (a set works there), the masking docs test now checks the analyses without `id_column` (summarize_categorical, profile_missing_values) instead of passing, and the README's first-look test asserts that it found the guide's steps. The spec's "five" masking analyses are six now.
- STD-61 also adds: `find_candidate_keys` against a brute force over every column set of up to 3 columns (text with NUL characters excluded, since pandas' factorize misreads them), and describe_dataset's missing, non-missing and distinct counts and the memory of NumPy columns against pandas on generated frames. Cross-process determinism of every analysis was already tested by ticket 02.
- STD-62: every `pytest.approx` states `rel=` or `abs=` (a test checks this), exact comparisons of shares say why they are exact, and the text-cap tests use `splitlines()`.
- STD-63: `df.attrs`, the sample values of each kind and `MIXED_VALUES` are copied per example, and the stand-in candidates of the sentinel recipe iterate types sorted by name.
- STD-64: `annotated-types`, `anyio` and `jsonschema` are in the `dev` group with the floors that pydantic and mcp 2.3 require, so the core run, which keeps `dev`, has `jsonschema` for `check_contract`. `uv.lock` was refreshed with `uv lock`.
- STD-22: the four reasonless suppressions got a reason (`_sentinels.py` and tests/test_summarize_numeric.py, tests/test_concurrent_calls.py), and tests/test_suite_rules.py reads every comment with `tokenize` and requires at least 3 words of reason on the same line. Its name check now covers `async def` too.
- docs/agents/new-analysis.md says what the contract tests now catch. The README needs no change: nothing a caller sees changed, and it stays at 59,997 characters.
- Not fixed, found on the way: `uv run pytest --hypothesis-profile=thorough` fails for `detect_missing_sentinels` and `profile_missing_values` when the recipe declares `''` for a `Sparse[object]` column, which both analyses refuse (`InvalidArgumentError`: "holds values no sentinel value can equal"). It does so without this ticket's changes too, since the kind and the recipe are older; ticket 16 (sentinels and missing values) or the recipe should settle it.
- Checks: ruff, ty, `uv run pytest` (3096 passed, 3 skipped), both minimum-version runs (2664 passed with 14 MCP skips on pydantic 2.7; the run with the mcp extra), the MCP stdio smoke and the benchmark (all 11 completed, the slowest 1.8 s) pass locally. Nothing is GitHub-only.
