# 04: Module layout

What to build: modules sit where their scope is visible:
- the MCP adapter in `agent_eda/_mcp/`;
- single-analysis engines beside their analysis;
- argument handling in `_arguments`;
- `_run` split;
- one type renderer and one closest-name helper;
- immutable MCP schema fragments;
- no private imports across modules.

Read these sections of `.scratch/deepening/spec.md`: Module layout, and test groups 1 and 5. This ticket covers user stories 4 and 9. Findings: DES-9, STD-35, STD-36, STD-49, STD-50, STD-56.

Blocked by: 03

Status: done

- [x] `_mcp_server`, `_mcp_instructions`, `_data_files`, `_local_files` and `_table_files` live in `agent_eda/_mcp/`. `agent_eda.mcp` and the `agent-eda-mcp` console script are unchanged.
- [x] `_association`, `_keys`, `_conversions` and `_sentinel_rules` live beside their analysis as private siblings.
- [x] Argument normalizing and the message helpers live in `_arguments`, with one type renderer and one closest-name helper. `_registry` holds only the registry.
- [x] `_run` is split into status helpers and a warnings module that also holds `_absorb_warnings`. `_sentinels` has a name for what it holds.
- [x] MCP schema fragments are immutable, and the file-format knowledge lives in one place.
- [x] A test fails when a module imports an underscore name from another module that isn't one of the internal-interface modules it allows.
- [x] `tests/test_packaging.py`'s module name strings follow the move.
- [x] The baseline comparison passes unchanged.
- [x] The finish checks pass.

## Comments

- **Names.** `_mcp/` holds `server.py`, `instructions.py`, `data_files.py`, `local_files.py` and `table_files.py`. The engines are `data/_keys.py`, `data/_conversions.py`, `data/_sentinel_rules.py` and `statistics/_association.py`. `_sentinels` became `_declared_values`. `_run` became `_status` (`overall_status`, `budget_exceeded_reason`, `NO_ROWS`) and `_warnings` (the capture helpers and `absorb_warnings`, which lost its underscore). `check_distinct` went to `_arguments`.
- **`_arguments`.** It takes the argument normalizing (`normalize_option`), `checked_keywords`, the data type check (`check_data`) and the message helpers. It knows nothing of the registry: `missing_message` and `normalize_option` take the analysis name and its annotations, and `unknown_name_message` takes `of_analysis=` where it used to ask `owner in _REGISTRY` (the registry and the MCP server pass `True`; the masking-argument sentence it guards is unchanged). `_registry` keeps the spec, the decorator, the call shape, validation with the arguments model and the result check. `_value_text` is public as `value_text`, which `_project_file` used to import by its underscore name.
- **One type renderer, one closest-name helper.** Earlier tickets had already merged both (`_annotations.type_text`, `_text.closest_names`), so this ticket moved nothing: a test now pins that `difflib` is imported in `_text` only and that `type_text` is defined once. `closest_names` stays in `_text` with the other message text helpers.
- **MCP schema fragments.** The data, project and table path schemas and the record schema are functions that build a new dict per tool, instead of module-level dicts that tools shared or copied shallowly. A test asserts the `_mcp` modules hold no module-level dict, list or set. `_call` decides whether a tool takes a project from the spec (`"project" in spec.parameters`), not from the input schema's properties.
- **File formats.** `FileFormat` answers what the reader, writer and instructions asked by comparing `kind` (`is_parquet`, `separator`, `separator_word`, `label`), and `local_files` builds the format lists in prose (`extensions(..., either=True)`, `format_names`). `supported_extensions().replace(" and ", " or ")` is gone. `os_cause` moved to `_wording` because `_project_file` (not an MCP module) uses it.
- **The import rule.** `tests/test_layout.py` finds every underscore name a module takes from another, by `from m import _name` or `m._name`, and allows two internal interfaces: `_result_base` for the `_*_results` modules (they build on its table check, caps and dtypes) and the package root's `_EXPORTS` for `_registry`. Importing a private module itself (`from agent_eda import _text`) is not a use. The same test file checks the layout (the five MCP modules in `_mcp`, only `agent_eda.mcp` imports it, the engines beside their analyses, `_run` and `_sentinels` gone).
- **`test_packaging`.** The one name check, `is_mcp` in the import report, now looks at every part of a dotted name, so `agent_eda._mcp.*` counts as an MCP module as `agent_eda._mcp_server` did.
- **Baseline.** Unchanged: `tests/baseline/*.json` and `tests/test_baseline.py` are not touched, and the comparison passes.
- **Merge.** Deepening 02 reached integration while this ticket was in flight, so I merged integration first (conflicts only in the import blocks of `missingness.py` and `targets.py`) and did the registry split on top of 02's `_envelope`.
- **Docs.** AGENTS.md's Layout says what each location means and names the import rule, and its `_run.capture_warnings` reference is `_warnings.capture_warnings`. The CHANGELOG has one line under Moved. The README names no private module, so it is unchanged (it is at its size limit).
- **Not run locally.** Nothing: ruff, ty, `uv run pytest` (3,859 passed), both minimum-version runs, the MCP stdio smoke test and the benchmark all passed.
