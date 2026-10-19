# 07: Statuses, codes and caps from one source

What to build: registration reads statuses and finding codes from the docstring, and each cap's number lives only in its marker, from which the field descriptions get their cap clauses.

Read these sections of `.scratch/deepening/spec.md`: Single sources, and test groups 1 and 6. This ticket covers user stories 7 and 9. Findings: DES-13, DES-14.

Blocked by: 05

Status: done

- [x] `@analysis` no longer takes `statuses=` or `finding_codes=`. Registration reads both from the parsed docstring, and raises for an unknown status or a malformed code. Test both with a broken stand-in.
- [x] The cap numbers 10, 5 and 100 appear once each, in the markers. Truncating analyses ask the result class for the limit.
- [x] Each capped field's description gets its cap clause from its marker. `check_contract` asserts that a capped field holds `min(cap, total)` entries.
- [x] The baseline comparison passes unchanged; descriptions may differ only where the generated cap clause rewords them, and the Comments list those.
- [x] The finish checks pass.

## Comments

- Finding codes are the Returns: bullets whose text starts with an assessment and a semicolon, in the order written. Other bullets, such as the evidence codes of `detect_missing_sentinels` (`declared`, `builtin_list`, `isolated_extreme`), are no finding codes. The docstring bullet regex now reads any text between the backticks as a code, so registration can raise `TypeError` for a malformed one ("malformed finding codes"). An unknown `Statuses:` entry raises `TypeError` too. `spec_of` reads a stand-in the same way.
- `profile_target`'s bullets were in a different order from its old `finding_codes`. The bullets now follow the old declared order, so `list_analyses()` and `describe_analysis()` give the same code order as before. This reorders the bullets in its docstring, so its MCP tool definition hash changed: `tool: profile_target [definition]` is the only baseline entry refreshed (pandas 2 and 3 files). Every other record, text, table, message, description schema and tool is byte-identical.
- Caps: `_ROW_IDS`, `_EXAMPLE_VALUES` and `_ENTRIES` in `_result_base` are the only places the numbers 10, 5 and 100 are written. Other markers come from `.derive(...)`. `ResultModel.limit_of(field)` gives a field's limit (ValueError for an uncapped field); `ROW_IDS_KEPT` and `EXAMPLES_KEPT` are gone, and the analyses read their limits from the result classes (`profile_target`'s `_KEPT_LABELS` too). summarize_categorical's `top_n` upper bound is `SummarizeCategoricalColumn.limit_of("levels")`.
- Field descriptions write `{cap}` ("The record keeps the first N and adds <field>_total") or `{limit}`, which `ResultModel.__pydantic_init_subclass__` fills from the marker and rebuilds the class for. Docstrings and the MCP instructions write `{row_ids_kept}`, `{examples_kept}` or `{entries_kept}`, which `@analysis`, `AnalysisResult.to_dict`'s docstring and the instructions fill. The generated text equals the old hand-written text, so no description changed. Only fields whose description already spoke of a cap got a clause; adding one to every capped field would reword about 40 descriptions and grow the README Reference, which is at its limit.
- `check_contract` takes each field's cap from the marker (a map built by walking the `ResultModel` subclasses, which must give one cap per field name) and asserts `len >= min(cap, total)`. A test scans the package for a cap number written outside `_result_base`.
- `docs/agents/new-analysis.md` steps 3, 4 and 6 describe the new sources. The README needed no change: its Reference is unchanged (`generate_reference.py --check` passes).
- Finish checks run locally: ruff, ruff format, ty, `uv run pytest`, both minimum-version runs, the MCP stdio smoke test and the benchmark all pass. Nothing was left to GitHub.
