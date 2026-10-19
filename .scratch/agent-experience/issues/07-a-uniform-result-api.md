# 07: A uniform result API

What to build:
- `to_frame(table=...)` on every result;
- helpful errors for mistyped column keys;
- pydantic's deprecated methods pointing to the right ones;
- a useful `help(res)` and a Rich display that keeps to the text template;
- methods that carry one analysis's result into the next;
- public submodules that always resolve.

Read these sections of `.scratch/agent-experience/spec.md`: Result API, and test group 5. This ticket covers user stories 16 to 20. Findings: AX-6, AX-7, AX-8, AX-23, AX-24, CORE-20, and SPEC's questionable decision on Rich display.

Blocked by: 05

Status: done

- [x] Every result accepts `to_frame(table=...)`. A result with one table accepts its name, and any other name gets the error that `compute_association_matrix` gives today.
- [x] `res.columns["Age"]` raises a `KeyError` that names the closest names, such as 'age'. `describe_dataset` results support lookup by name with the same message, and the docs say why `columns` is a list there.
- [x] `res.json()`, `res.dict()`, `res.schema()`, `res.parse_obj()` and `res.copy()` raise an error naming `to_json()`, `to_dict()`, `record_schema()` or `model_copy()`.
- [x] `help(type(res))` shows each field's description and points to `ae.describe_analysis`. `pydoc.render_doc` of a result is under 10,000 characters.
- [x] Rich renders the capped text template.
- [x] `InferColumnSemanticsResult.inferred_schema()` returns a mapping that `validate_column_types` accepts, leaving out columns without a type.
- [x] `DetectMissingSentinelsResult.sentinel_values(columns=None)` returns `sentinel_values` that `profile_missing_values` accepts. Masked candidates are left out, with a note.
- [x] Both docstrings show the flow in their Examples.
- [x] `ae.data`, `ae.statistics`, `ae.discovery`, `ae.project` and `ae.mcp` resolve on a fresh import. `ae.mcp` gives the install command when the extra is missing. `ae.AgentEDAError` raises an AttributeError that points to `ae.errors.AgentEDAError`.
- [x] The finish checks pass.

## Comments

Tests are in `tests/test_result_api.py`.

- **Table names.** A result with one table names it: `describe_dataset` and `summarize_numeric` `'columns'`, `summarize_categorical` `'levels'`, `find_duplicate_rows` `'rows'`, `profile_target` `'target'`, `detect_missing_sentinels` and `find_candidate_keys` `'candidates'`. A wrong name gets `_check_table`'s message; with one table it reads "table must be 'columns'", not "one of".
- **Column keys.** The six `columns` mappings are a `dict` subclass (an after-validator) whose `__missing__` raises a `KeyError` with the closest names and the first 10 columns. `describe_dataset`'s `columns` stays a list, whose `__getitem__` also takes a name; a label that repeats names its positions. The field description says why it is a list. The lookup in Python (`res.columns['age']`) is in help's note only, since field descriptions reach MCP callers.
- **Deprecated methods.** They raise `InvalidArgumentError` (code `invalid_argument`), since the ticket didn't name a class. They sit last in `ResultModel`, so a method named `dict` doesn't shadow the type in later annotations of that body.
- **help.** A metaclass on `ResultModel` writes each analysis result's docstring with its fields and the `ae.describe_analysis('<name>')` pointer, replaces the constructor signature (pydantic's held `<_Cap object at 0x...>`) and trims `dir(cls)` to the package's own members. The largest class, `ProfileTargetResult`, renders in 8,543 characters; the others in 4,300 to 5,500. `record_schema()` puts the plain docstring back as the schema's description, so the output schema keeps no Python. `generate_reference.py` reads the plain docstring.
- **Rich.** `__rich_console__` on analysis results and discovery records yields one `Text` of `str(res)`, so square brackets in labels aren't markup. `rich` is not imported by the package; it is a dev dependency (`rich>=13.0`) for the test.
- **Bridges.** `inferred_schema()` gives logical type names, and a `ColumnSpec` for a datetime column whose call declared a `datetime_format`. `sentinel_values()` converts the ISO text of a Timestamp, date, time, Timedelta or Decimal candidate back to that type, since `profile_missing_values` refuses the text for such a column (`ae.mcp` callers have no such method). "Masked candidates are left out, with a note": the return value is a `dict` subclass (`SentinelValues`) with a `note` attribute, because the package raises no warnings for data; the note is None when nothing was left out. Suspected and declared candidates are both included. An unknown name in `columns` raises `ColumnNotFoundError` with the closest names.
- **Submodules.** `ae.data`, `ae.discovery`, `ae.project`, `ae.statistics` and `ae.mcp` join `ae.errors` and `ae.results` in `_SUBMODULES`. `agent_eda.mcp` imports without the extra, so `ae.mcp` always resolves and `ae.mcp.create_server()` gives the install command when the extra is missing (tested with `mcp` blocked); I didn't make the attribute access itself raise, since `hasattr` and `dir()`-based tools would then fail with an ImportError. A capitalised name that lives in `ae.errors` or `ae.results` raises an `AttributeError` that names `ae.errors.<Name>` or `ae.results.<Name>`.
- **README.** It had 61 characters of room, so the additions (about 1,000 characters) were paid for by tightening other prose; after merging the integration branch (which had 59,985) it is at 59,953 of 60,000, so the next README addition must be offset again. The README's `AnalysisResult` text comes from `AnalysisResult.__plain_doc__`.
- **Finish checks.** All pass locally: ruff, ty, `pytest`, both minimum-version runs, the MCP stdio smoke test and the benchmark (no regression), rerun after the merge.
