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

Status: ready-for-agent

- [ ] `profile_missing_values` takes `project=` and `mask_sensitive=`. The declared sentinel values of sensitive columns are masked in `parameters` and in the tables, as in `detect_missing_sentinels`. These all treat it as a masking analysis:
  - `describe_analysis`;
  - the MCP tool, which gains `project_path`;
  - the contract;
  - the README, where five masking analyses become six;
  - `tests/test_mcp_project.py`.
- [ ] A sensitive column declared as NFC "café_owner", when the data's column uses the NFD form, makes the masking text name the near match. The same holds for "Café_Owner" against "café_owner". In both cases the text no longer claims that no sensitive value is shown.
- [ ] `columns=df["email"]`, an Index or an ndarray of values raises a message that says it looks like a column's values. The message shows none of the values and gives the fix. The same holds for `id_column`, `target_column` and `categorical_columns`, with masking on and off.
- [ ] Messages that list unknown names show at most 3 names, each at most 40 characters long.
- [ ] With masking on, no error message quotes a value of a sensitive column. Test it with duplicate `expected_labels` for a sensitive target, and with `sentinel_values` for a sensitive column.
- [ ] Over MCP, an internal error with masking on names only the exception type.
- [ ] The MCP instructions and the README's MCP section say that numeric statistics and index labels are never masked, and that analyses without masking don't read the project.
- [ ] The no-leak property gives `find_duplicate_rows` options that always repeat rows and set a sensitive `id_column`. It asserts that at least half of the examples for each masking analysis would leak with masking off.
- [ ] A mutant that skips masking in `find_duplicate_rows` when there are 3 or more groups fails the suite. The Comments record that.
- [ ] The finish checks pass.

## Comments
