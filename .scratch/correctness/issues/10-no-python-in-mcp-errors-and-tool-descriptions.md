# 10: No Python in MCP errors and tool descriptions

What to build: nothing an MCP caller reads in an error message, a tool description or an input schema holds Python. A guard test keeps it that way.

Read these sections of `.scratch/correctness/spec.md`: No Python in what MCP callers read, and test group 8. This ticket covers user story 24. Findings: DES-7 (the three leaks), SPEC-3, SPEC-12, MCP-7, STD-8, DOC-10 (descriptions).

Blocked by: 08, 09

Status: done

- [x] These MCP messages give MCP fixes:
  - sentinel examples, as `{"sentinel_values": {"age": [-999]}}`;
  - "did you mean" for datetimes and durations, as ISO strings;
  - `profile_target`'s task fix, as `{"task": "multiclass_classification"}`;
  - `project_path` problems, naming `project_path` and saying how to write the TOML file, with no `project.save` step;
  - echoed values, written as JSON (`null`, `true`).
- [x] `tests/test_mcp_project.py` expects the MCP fix for a missing project file.
- [x] The analysis docstrings' When to use, Returns, Statuses and See Also sections read correctly over MCP. MCP tool descriptions hold no `res.`, `df.`, `columns=None` or `table="matrix"`, and Python callers still get usable text.
- [x] The MCP instructions say, in one line, that a description's `to_dict` record is the structured content and its `to_frame` table is what `table_path` writes (SPEC-12).
- [x] A guard test scans every MCP tool description and input schema for Python syntax:
  - call syntax;
  - `name=value` keywords;
  - `pd.`, `np.`, `df.`, `res.` and `ae.`;
  - `None`, `True` and `False` used as values.
- [x] A mistake table, extending `MISTAKES`, reaches every error code each MCP tool can raise, and the guard scans those messages too.
- [x] The finish checks pass.

## Comments

- Where the spec left it open:
  - Most of the five known leaks were already worded by earlier tickets: the sentinel examples, the datetime "did you mean" and `mask_sensitive` echoes. What remained was `profile_target`'s task fix (also for a label of a type the target can't hold), the infinite-sentinel fix, `project_path` messages, echoed `None` and `True`, `NoneType`, `schema['score']`-style locations, `sentinel_values['age']` and `Project.load` as the subject of every project-file message. All are worded for both surfaces now.
  - Over MCP, the subject of a project-file message is "The MCP server", the file is "the project_path file '…'", and a missing file says how to write the TOML file, with no `project.save` step.
  - Over MCP, an echoed `None`/`True`/`False` is `null`/`true`/`false`, a location inside an option is a JSON pointer (`schema/score`, `quantiles/0`), and strings keep their quotes.
  - The docstrings stay the one hand-written copy. Where text only made sense in Python it was rewritten to read on both surfaces (`res.columns["age"]`, `df.loc[row_ids]`, `columns=None`, `table="matrix"`, `pd.NA`, `bias=False` and the like). The tables other than the main one now read "its Python-only table `matrix`".
  - The MCP server turns the docstrings' `None`, `True` and `False` into `null`, `true` and `false` in descriptions and input schemas (`_wording.json_words`), so an `Args:` entry still says "None, the default" to a Python caller and "null, the default" over MCP.
  - The one-line mapping of `to_dict` and `to_frame` replaced the data-reading detail in the instructions, which the `data_path` description already holds, to stay within 60 lines.
- Mistake table: `MISTAKES` in `tests/test_mcp_argument_messages.py` now reaches every code in the `ae.errors` docstring except the six that no MCP tool can raise (`NOT_OVER_MCP` in `tests/test_mcp_wording.py` says why for each). `tests/test_mcp_wording.py` fails when a new code has no row.
- The README was over its limit on the integration tip (60,862 characters, limit 60,000). It is 59,935 now: the docstring changes cut about 300, and prose in Features, Projects and masking and Use over MCP, and in the `errors` docstring, was tightened to pay for the new sentence on MCP wording.
- Not run locally: nothing. The benchmark ran (12 passed) although no analysis's cost changed.
