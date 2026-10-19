# 02: Byte-identical records across processes

What to build: identical calls give byte-identical JSON in different processes, for every analysis and every kind of value. A cross-process test guards all 11 analyses.

Read these sections of `.scratch/correctness/spec.md`: Deterministic records, and test group 2. This ticket covers user story 12. Findings: AN2-3, CORE-2, SPEC-1, STD-4, STD-5, TEST-2.

Blocked by: 01

Status: done

- [x] Labels that are sets or frozensets, such as `frozenset({"red", "blue", "green"})`, give byte-identical `to_json()` under `PYTHONHASHSEED` 1 to 4. This holds in `summarize_categorical`, `profile_target` and `validate_column_types`, and the label is written with its items sorted.
- [x] Set-like row IDs and labels share one written form, including inside tuples, and the printed text uses the same form.
- [x] An object whose repr carries a memory address is written as `<TypeName object>`.
- [x] `describe_dataset` gives the same `memory_bytes` in two calls on an object column of `[np.str_("é0"), np.str_("é1"), np.str_("é2"), None]`, when the same objects also fill a second column.
- [x] A test parametrized by analysis runs each recipe on `EDGE_FRAMES` and on generated frames that include set, frozenset and `np.str_` values. It runs them in two subprocesses with different `PYTHONHASHSEED` values and compares `to_json()`, `str(res)` and every `to_frame()` table.
- [x] Run once against a mutant that iterates a set in `detect_missing_sentinels`, the test fails. The Comments record that.
- [x] The finish checks pass.

## Comments

- Written form, settled here: `canonical_text` (new `_written_form.py`, which also holds `row_id_json`, still importable from `agent_eda.results`) writes a set as `{...}` or `frozenset({...})` and keeps `str()`'s form for everything else. Items are ordered by their JSON text, as `row_id_json` orders them, so `{10, 2}` lists `10` first. Items whose JSON reads the same (`Decimal('1')` and `'1'`) are ordered by type name, then written form. Containers are tuples, lists, dicts, sets and frozensets of exactly those types, recursively. `_levels.label_key`, `_levels.record_label`, `_text.label_text` and the fall-through of `row_id_json` use it. The record holds a set row ID as a sorted JSON list and a set label as this text, and the printed text uses the text form for both.
- `to_frame()` keeps sets as the objects they are, so `to_csv()` of a table with a frozenset cell still lists it in hash order. The cross-process test therefore writes each table's cells with `row_id_json`, as the MCP table files do. The README says so.
- `describe_dataset` sizes any non-ASCII `str` instance without its cached UTF-8 copy. The header of a subclass such as `np.str_` is measured on an empty instance of its type; a subclass that can't be made from `""` falls back to `sys.getsizeof`. A second column holding the same objects with a repeat (`[0, 1, 2, 0]`) showed the drift before the fix (697 then 709 bytes); the box's own example (`[0, 1, 2, 3]`, no repeat) already gave equal sizes in this environment, and it stays in the test.
- The cross-process test (`tests/test_determinism.py`, worker `tests/determinism_worker.py`) starts one worker per `PYTHONHASHSEED` (1 and 2) that runs every recipe's required arguments on `EDGE_FRAMES`, `LABEL_EDGE_FRAMES` where the recipe takes them, and 30 frames drawn from `data_frames()` and the new `frames_with_sets()` (set, frozenset, tuples of frozensets, `np.str_`, objects with a default repr, frozenset row indexes), pickled by the test. It compares `to_json()`, `str(res)` and every table, or the error raised. The label tests use seeds 1 to 4.
- Mutant run: changing `names = list(dict.fromkeys(column for _, column in paths))` to `names = list({column for _, column in paths})` in `detect_missing_sentinels` (`data/missingness.py`, the sentinel finding) made `test_every_analysis_gives_identical_records_text_and_tables_across_processes[detect_missing_sentinels]` fail; reverted afterwards.
- Finish checks run locally: ruff check and format, ty, full pytest, both minimum-version runs, MCP stdio smoke, and the benchmark (all analyses under 2 s). Nothing was left for GitHub.
