# 06: Masking boundary

What to build:
- `profile_missing_values` masks the declared values of sensitive columns;
- sensitive names that miss a column only by case or Unicode form are reported instead of glossed over;
- a column's values passed where its name belongs are refused without being shown;
- messages never quote sensitive values;
- the docs say what is never masked;
- the no-leak property really exercises `find_duplicate_rows`.

Read these sections of `.scratch/correctness/spec.md`: Masking boundary, and test group 5. This ticket covers user stories 16 to 19. Findings: CORE-4, CORE-6, CORE-19, SPEC-7, MCP-11, TEST-1, and STD's questionable decision on internal errors.

Blocked by: none

Status: done

- [x] `profile_missing_values` takes `project=` and `mask_sensitive=`. The declared sentinel values of sensitive columns are masked in `parameters` and in the tables, as in `detect_missing_sentinels`. These all treat it as a masking analysis:
  - `describe_analysis`;
  - the MCP tool, which gains `project_path`;
  - the contract;
  - the README, where five masking analyses become six;
  - `tests/test_mcp_project.py`.
- [x] A sensitive column declared as NFC "café_owner", when the data's column uses the NFD form, makes the masking text name the near match. The same holds for "Café_Owner" against "café_owner". In both cases the text no longer claims that no sensitive value is shown.
- [x] `columns=df["email"]`, an Index or an ndarray of values raises a message that says it looks like a column's values. The message shows none of the values and gives the fix. The same holds for `id_column`, `target_column` and `categorical_columns`, with masking on and off.
- [x] Messages that list unknown names show at most 3 names, each at most 40 characters long.
- [x] With masking on, no error message quotes a value of a sensitive column. Test it with duplicate `expected_labels` for a sensitive target, and with `sentinel_values` for a sensitive column.
- [x] Over MCP, an internal error with masking on names only the exception type.
- [x] The MCP instructions and the README's MCP section say that numeric statistics and index labels are never masked, and that analyses without masking don't read the project.
- [x] The no-leak property gives `find_duplicate_rows` options that always repeat rows and set a sensitive `id_column`. It asserts that at least half of the examples for each masking analysis would leak with masking off.
- [x] A mutant that skips masking in `find_duplicate_rows` when there are 3 or more groups fails the suite. The Comments record that.
- [x] The finish checks pass.

## Comments

- Near misses are a new field on `Masking`: `near_miss_columns` maps each absent sensitive name to the data column or index level it matches after `casefold()` and NFC. A name whose near match is itself declared sensitive is not reported. The text says "did you mean ...?" and that the near match is not masked, and it no longer says "the result shows no value of a sensitive column". Matching stays exact. Masking off prints no near-miss text, as it prints nothing about absent columns today.
- Values passed as names: the refusal sits in the registry's normalisation step, so it covers every option ending in `column` or `columns`. A Series, Index or ndarray is refused for `*_column`. For `*_columns` it is refused when any item is not a column label of the data, so `df.columns[1:]` and `pd.Series(["age", "income"])` still work, and a misspelled name in a list keeps its did-you-mean message. A misspelled name inside an Index or array is refused as "looks like a column's values" instead. The fix names the first column of the data that is at most 40 characters long.
- Messages that list unknown names (`ColumnNotFoundError`) show 3 names and count the rest, each cut to 40 characters by the same `reprlib` rule `_value_text` uses (now `_text.short_repr`).
- Hidden values: `MaskingDecision.hidden(names)` gives the masked columns whose declared values no message may quote. `check_distinct`, the sentinel-value messages (including the value the caller likely meant) and `profile_target`'s `expected_labels` messages count the values instead of quoting them for such a column. With masking off the messages are unchanged.
- Over MCP, `_masking_on` decides from what the call left of its arguments. A `project_path` that was never loaded counts as masking, so a bug before the project is read cannot show a value either. The traceback stays in the server log on stderr.
- The instructions stay within 60 lines by saying "Numeric statistics, such as a minimum, and index labels are never masked. Tools without [supports masking] don't read the project." The README says the same in "Masking over MCP", and adds the near-miss, hidden-value and internal-error behaviour to "Projects and masking".
- Test data: `frames_with_secrets` now always repeats 1 to 4 rows at least twice, and gives an all-missing sensitive column a value, since a column with no value has nothing to leak. The recipe for `find_duplicate_rows` compares every column and takes a sensitive `id_column`; the one for `profile_missing_values` declares some of each sensitive column's own values. `test_most_examples_of_the_no_leak_property_would_leak_with_masking_off` is the floor: 60 examples per masking analysis, at least half must show a sensitive value with masking off. Measured over 200 examples each: find_duplicate_rows 166, detect_missing_sentinels 153, profile_missing_values 172, profile_target 159, validate_column_types 190, summarize_categorical 155. Before the data change, find_duplicate_rows leaked in about 105 of 200 and profile_missing_values in 25 of 60.
- Mutant: with `if masking.masks(row_ids.id_column) and len(sizes) < 3:` in `find_duplicate_rows` (no masking from 3 duplicate groups), the previous suite passed (its `test_properties.py` and `test_masking.py` ran against the mutant with only the unrelated record-shape failure of the new `near_miss_columns` field). With this ticket the mutant fails `test_with_masking_on_no_value_of_a_sensitive_column_reaches_the_result[find_duplicate_rows]` and `test_masking_covers_the_row_ids_however_many_duplicate_groups_there_are[3, 4, 10]`. The mutant was reverted.
- Finish checks: ruff, ruff format, ty, `uv run pytest` (2,435 passed), both minimum-version runs (2,116 passed without the extra, 2,435 with it), `tests/smoke_mcp.py` and `pytest -m benchmark` pass locally. The thorough Hypothesis profile also passes the masking properties. GitHub-only checks were not run.
