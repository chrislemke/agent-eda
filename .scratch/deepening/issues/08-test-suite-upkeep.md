# 08: Test suite upkeep

What to build:
- one helper module for the MCP tests;
- one golden test per text template, with structural assertions elsewhere;
- a README test that finds its JSON block by content;
- one interpreter for the packaging import checks;
- clearer names.

Read these sections of `.scratch/deepening/spec.md`: Test suite upkeep. This ticket covers user story 8. Findings: TEST-12 (README part), TEST-13, TEST-14, TEST-17, STD-58, STD-59.

Blocked by: none

Status: done

- [x] One helper module serves the MCP tests: the client call, the tool listing, file writing, the Python reference run and an independent reference read. No copies remain across test modules (TEST-14).
- [x] Each text template has one golden test, and other text tests assert structure, codes, fields and key fragments.
- [x] Rewording the footer from "have the rest" to "hold the rest" fails at most 3 tests. The Comments record the count.
- [x] `tests/test_readme.py` selects the MCP JSON block by its content.
- [x] The packaging import checks run in one interpreter, and the Comments record the time saved.
- [x] Names that STD-58 found unclear are renamed.
- [x] The finish checks pass.

## Comments

- MCP helper module: `tests/mcp_helpers.py` holds `call`, `list_tools`, `listed_tools`, `write`, `write_frame`, `read_parquet`, `read_table`, `reference_read` and `call_in_python`. The twelve MCP test modules and `test_record_integers.py` import them; `test_mcp_wording.py` no longer imports `call` from another test module. Each module keeps only the hand-built frames its own CSV texts stand for, and passes them to `reference_read`. The scenario-style `async with Client(...)` blocks that make several calls in one session stay where they are. A call on a server started with a Project is `call(name, arguments, create_server(project=project))`.
- Footer reword: before this ticket, "have the rest" to "hold the rest" in `_text.py` failed 16 tests (11 docstring examples in `test_contract.py`, the association, duplicates and profile_target text tests, the README example test and the template test). After it, it fails 2: `test_result_text.py::test_text_follows_the_fixed_template` (the golden test of the analysis result template) and `test_readme.py::test_the_usage_examples_run_as_written_and_print_what_the_readme_shows` (the README pins its examples' output). The docstring examples are compared by a doctest output checker that treats a `More:` footer as `More: ...` on both sides, so the golden test is the only place the footer's wording is pinned in the example run. The listing and description templates keep their own golden tests, which don't print the footer. `tests/smoke.py` is a CI script, not a test; it still compares the footer, as it runs in an environment without the test suite.
- README test: `test_the_mcp_client_configuration_is_valid_json_that_starts_agent_eda_mcp` keeps the json blocks that hold `mcpServers`, so another json block can join the README.
- Packaging: one interpreter runs the five import steps (import, list, guide, search, describe) and reports what each newly loads (scipy, statsmodels, sklearn, mcp modules, network events). The 9 interpreter tests took 3.8 s of call time (0.4 to 0.5 s each); the 10 tests now share one fixture that runs once, 0.5 s in all, so about 3.3 s are saved.
- STD-58 renames: `_text._RECORD` to `_FOOTER_RECORD`; `_project_file.invalid` to `invalid_project_error`; `_project_file._kind` to `toml_type`; `trash` in `_association` to `missing_slot`; `Levels.masked_types` to `placeholder_types`; `_conversions._stored_verdict` takes `kind: ColumnKind`; in tests, the parametrized `given` in `test_profile_target.py` is `passed`, and `codes()` is `product_codes()` in `test_validate_column_types.py` and `finding_codes()` in `test_describe_dataset.py` and `test_exact_numbers.py`. Not done: the IntEnum for the conversion verdicts and a separate type for masked levels, which the finding offers as a direction; they change types, not names.
- `check_output`, the method name `doctest.OutputChecker` requires, joined `EXTERNAL_NAMES` in `tests/test_suite_rules.py`, since the glossary rule avoids "output".
- Nothing user-facing changed, so the README is untouched.
