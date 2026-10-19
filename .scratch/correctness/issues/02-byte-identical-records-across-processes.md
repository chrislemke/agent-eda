# 02: Byte-identical records across processes

What to build: identical calls give byte-identical JSON in different processes, for every analysis and every kind of value. A cross-process test guards all 11 analyses.

Read these sections of `.scratch/correctness/spec.md`: Deterministic records, and test group 2. This ticket covers user story 12. Findings: AN2-3, CORE-2, SPEC-1, STD-4, STD-5, TEST-2.

Blocked by: 01

Status: ready-for-agent

- [ ] Labels that are sets or frozensets, such as `frozenset({"red", "blue", "green"})`, give byte-identical `to_json()` under `PYTHONHASHSEED` 1 to 4. This holds in `summarize_categorical`, `profile_target` and `validate_column_types`, and the label is written with its items sorted.
- [ ] Set-like row IDs and labels share one written form, including inside tuples, and the printed text uses the same form.
- [ ] An object whose repr carries a memory address is written as `<TypeName object>`.
- [ ] `describe_dataset` gives the same `memory_bytes` in two calls on an object column of `[np.str_("é0"), np.str_("é1"), np.str_("é2"), None]`, when the same objects also fill a second column.
- [ ] A test parametrized by analysis runs each recipe on `EDGE_FRAMES` and on generated frames that include set, frozenset and `np.str_` values. It runs them in two subprocesses with different `PYTHONHASHSEED` values and compares `to_json()`, `str(res)` and every `to_frame()` table.
- [ ] Run once against a mutant that iterates a set in `detect_missing_sentinels`, the test fails. The Comments record that.
- [ ] The finish checks pass.

## Comments
