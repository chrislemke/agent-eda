# 06: One value-reading engine

What to build: `infer_column_semantics` and `validate_column_types` read values through one engine, `read_values(series) -> ValueReading`, and `ColumnSpec` moves to the declarations module.

Read these sections of `.scratch/deepening/spec.md`: One value-reading engine, and test group 1. This ticket covers user stories 6 and 9. Findings: DES-17.

Blocked by: 04

Status: done

- [x] `read_values` returns a `ValueReading` with:
  - the distinct values and their counts;
  - the value kinds and storage type;
  - a verdict per logical type.
- [x] `infer_column_semantics` computes its shares from `ValueReading`, and `validate_column_types` its compatible, convertible and invalid counts. Neither walks a column on its own.
- [x] `ColumnSpec` lives in the declarations module. `ae.ColumnSpec` and `agent_eda.data.validation.ColumnSpec` still work.
- [x] The benchmark times for both analyses stay within 10% of the times before the change, on the latest and the minimum versions. The Comments record them.
- [x] The baseline comparison passes unchanged.
- [x] The finish checks pass.

## Comments

- `read_values(series, kind=None, datetime_format=None)` is in `agent_eda/data/_value_reading.py`; it replaces `_conversions.py` (the layout and glossary tests name the new file). A `ValueReading` finds each part when first asked, so a column whose dtype decides the answer is never walked, which keeps the fast paths of tickets 12 and 13: `levels` (the distinct values and counts as pandas counts them), `distinct_values`, `counts` and `codes` (the groups that the verdicts cover, told apart by Python type), `kinds`, `storage`, and `verdict(logical_type)`. `check(declared)` gives `validate_column_types` its counts, invalid rows and invalid values; the reading's `boolean`, `number`, `whole`, `datetime`, `timedelta`, text counts, `span` and `types` give `infer_column_semantics` its shares. infer's `_Reading` and all its walkers are gone; `without_duplicate_rows` stays in infer as a plain dict, since it is not a reading of the values.
- `ColumnSpec` moved to `agent_eda/_declarations.py`; `_EXPORTS` points there. `ColumnSpec.__module__` is now `agent_eda._declarations`.
- Benchmark, seconds on 1,000,000 rows x 100 columns, before then after (separate runs back to back on one machine, old code run through PYTHONPATH). Latest versions (Python 3.12, uv.lock), infer: 2.14, 2.15 then 2.10, 2.12, 2.16; validate: 2.07, 2.06 then 1.90, 1.94, 1.92. Minimum versions (3.12, numpy 1.26, pandas 2.2, mcp extra at its floors), infer: 3.34, 3.19 then 3.13, 3.20; validate: 2.83, 2.82 then 2.68, 2.73. Both are within 10% of before, validate about 5% faster. (A first run of the old code, taken while the machine was busy, read 2.9 and 2.6 seconds; the back-to-back runs above are the comparison.)
- Equality beyond the baseline: 4,510 frames (3,000 Hypothesis frames, the EDGE_FRAMES and 1,500 object columns mixing booleans, 0/1, floats, Decimals, Fractions, dates, lists, dicts, numpy scalars and text) gave the same JSON, printed text and `invalid_values` tables from the old and the new code for `infer_column_semantics` and `validate_column_types`, with every logical type declared and with a datetime_format, except 33 frames described next.
- One difference, a fix: `_values._all_text_objects` tested `dtype is not np.dtype(object)`, which is false for the object dtype of unpickled data (`pickle.loads`, `pd.read_pickle`). `value_codes` then fell back to pandas' factorize, which reads `'\x00'` as the end of the text, so `infer_column_semantics` counted `''` and `'\x00'` as one value while `validate_column_types` (which factorized a NumPy array) kept them apart. The test is now `!=`, so the two agree and the documented rule holds; `test_text_that_differs_by_a_nul_character_stays_apart_in_unpickled_data` covers it. CHANGELOG has an entry.
- Kept deliberately: for pyarrow-backed floats the levels keep -0.0 and 0.0 apart (so infer's `distinct_count` does too), while every check of the values treats them as one, as before; the reading builds the groups of such a column from its floats. That disagreement was already there and is not in this ticket's scope.
- README unchanged: it is at its 60,000-character limit and says nothing about these internals. `scripts/generate_reference.py --check` passes.
- Not run locally: nothing; all finish checks ran (ruff, format, ty, pytest, both minimum-version runs, MCP stdio smoke, benchmark, and the thorough Hypothesis profile on the two analyses' tests).
