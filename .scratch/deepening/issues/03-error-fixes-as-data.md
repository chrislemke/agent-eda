# 03: Error fixes as data

What to build: errors carry what is wrong, what is allowed and their fixes as data, and one renderer per surface writes them. Raise sites stop branching on the surface. `ColumnRule` takes fixes and can be chosen per task.

Read these sections of `.scratch/deepening/spec.md`: Error fixes as data, and test groups 1 and 4. This ticket covers user stories 3 and 9. Findings: DES-7, STD-57, DES-15, STD-55 (the fix tables).

Blocked by: 01

Status: done

- [x] `_wording` defines `Pass`, `LeaveOut`, `SeeArguments`, `PythonOnly` and `DeclareSensitive`, and one renderer per surface writes them.
- [x] No raise site branches on the surface, and `worded` and its lambdas are gone.
- [x] `ColumnRule` takes these fixes and can be chosen per task. `profile_target` selects its target with the task's rule, and `_check_target`, `_REGRESSION_FIXES` and `_MCP_REGRESSION_FIXES` are gone.
- [x] A property over the renderer checks every fix type on both surfaces, including the no-Python guard for MCP. It replaces the curated `MISTAKES` entries it covers.
- [x] The message tables give byte-identical messages to the baseline.
- [x] The finish checks pass.

## Comments

- **Vocabulary.** `_wording` holds the five fixes the ticket names and a few more that raise sites needed: `Words(python, mcp)` for a name or phrase that differs wholesale (who reads a project file: `Project.load` or the MCP server), `Echo`, `TypeOf`, `PathInto`, `Listed`, `Of` (another error's message), `Say`, `OneOf` (the list of steps of a fix, dropping the ones a surface can't take) and the constant `NONE`. `Declared` (a declared value as the caller writes it) lives in `_sentinels` and `ShownValue` in `_registry`, next to the code they wrap. `PythonOnly(code, otherwise="")` also carries the MCP alternative: `profile_target` gives an MCP caller "Pass a column of numbers instead" where a Python caller gets a conversion.
- **Raising.** `mistake(ErrorClass, *wording, code=None)` builds the error. `str(error)` is the Python message and `message(error, "mcp")` the MCP one; the error keeps its wording in `_wording`, which replaces `_mcp_message`. The remaining `surface` parameters sit inside the renderers (`_value_text`, `literal`, `echoed`, `type_of`) and in result and discovery text, which are not error messages.
- **ColumnRule.** It takes `fixes` as fix objects (one, or several in order), a `rejection` sentence template and `leave_out`. `TO_NUMBERS` and `TO_LABELS` are `PythonOnly` objects. `profile_target` has one rule per task (regression, classification) and `select_columns` does the type check; its message is byte-identical.
- **MISTAKES.** The tables stay as they are, since the byte baseline records their code and full message and the ticket requires it to pass unchanged. `tests/test_wording.py` is the property over the renderer (every fix type on both surfaces, JSON and Python round trips, no Python syntax over MCP, dropped steps) and is the one test that imports private modules (`tests/test_suite_rules.py` lists it).
- **Byte comparison.** Beyond the baseline, I compared the Python and MCP message of 173 more mistakes (every column kind against every task and rejecting analysis, column-label, masking, sentinel, target label, project file and argument mistakes) between the old and the new code: identical.
- **Baseline harness fix.** Two baseline messages named the working directory of the worktree the baseline was recorded in (`data_path` that is a directory, `table_path` in a missing directory), so `tests/test_baseline.py` failed in any other directory. `scripts/record_baseline.py` now writes the working directory as `<cwd>` like it does the temporary one, and the two entries of both baseline files were rewritten the same way; no other byte of the baseline changed.
- **Not run locally.** Nothing: the finish checks, both minimum-version runs, the MCP stdio smoke test and the benchmark all passed.
