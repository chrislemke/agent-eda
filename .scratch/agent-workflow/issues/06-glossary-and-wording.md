# 06: Glossary and wording

What to build: the glossary defines every term the analyses rely on, and code, tests and docs stop using the words it avoids.

Read these sections of `.scratch/agent-workflow/spec.md`: Glossary and wording. This ticket covers user stories 25 and 26. Findings: DOC-16, SPEC-19, STD-23, STD-24.

Blocked by: 04

Status: done

- [x] `GLOSSARY.md` defines each of these, with an Avoid line in the existing format:
  - Limitation
  - First look
  - Level
  - Native missing value
  - Compatible, convertible and invalid value
  - Key, primary key and candidate key
  - Target, class and label
  - Sentinel candidate
- [x] Each definition matches what the code does. For example, a level is a distinct non-missing value as pandas `value_counts` tells them apart.
- [x] The words STD-23 lists are gone from field descriptions, names and tests.
- [x] Unqualified "server" is gone from the 8 test names and the private docstrings STD-24 lists, and the MCP instructions no longer say that every MCP tool runs an analysis.
- [x] The finish checks pass.

## Comments

- Entries: each bullet of the ticket is one entry with its terms bold in the heading, such as `**Key**, **primary key** and **candidate key**`. `tests/test_glossary.py` finds an entry by any of those terms and requires a definition and an `_Avoid_` line.
- Definitions match the code, each checked by a test through the public API on a small DataFrame: a level is what `summarize_categorical` counts as `cardinality` and pandas `value_counts` tells apart (1, 1.0 and True are one level); a native missing value is `profile_missing_values`' `native_missing_count`, apart from `sentinel_count`; compatible, convertible and invalid are `validate_column_types`' three counts; a key, a primary key (no missing value) and a candidate key (minimal) are `find_candidate_keys`' `keys` and `keys_with_missing_values`; target, class and label follow `profile_target`; a sentinel candidate is a `declared` or `suspected` candidate of `detect_missing_sentinels`; a limitation has the `code`, `reason` and `fields` of `Limitation`.
- "label" is a value of the target column. A column's name stays "column label" where pandas' term is meant (34 uses); the Avoid line says to qualify it.
- STD-23: "verdict" became "assessment" in the `ValidateColumnTypesColumn.assessment` description; `outcome` became `consequence` (the `budget_exceeded_reason` parameter), `tally` (the key search's `KeySearchTally`) and `read` (association); "stopped the check" became "stopped the analysis"; test variables `outputs` and `output` became `printed` or `renderings`, and `outcome` in `test_describe_analysis` became `described`. `_conversions.py` keeps `verdict`: it names the group a value falls into, not an assessment (STD-23's own evidence). `output_schema` and `capture_output` are other libraries' words and stay. Not touched: backlog names such as `assess_outcome_maturity` in `docs/planned-analyses.md`, which are proposals.
- STD-24: the 8 test names and 5 more that the audit's grep missed say `mcp_server`; the parametrize ids and the `server` parameter in `test_mcp_project.py` say `mcp_server_project`; the private docstrings in `_mcp_server.py` and `_table_files.py` and one in `mcp.py` say "MCP server"; `path_argument`'s `tool` parameter is `analysis`. The instructions now open with "Each analysis is an MCP tool that answers one EDA question", and the closing paragraph says `search_analyses` finds analyses and `list_analyses` lists them, so neither claims that every MCP tool runs an analysis. The instructions are still 59 lines.
- Guard tests in `tests/test_suite_rules.py`: no identifier in the package or the tests holds verdict, outcome or output (with the exceptions above); no result field description does; no test name and no text constant in the package says "server" without "MCP". They failed on the unchanged code (90 identifier hits, one field description, 13 test names, 10 texts), which is the mutant check, and pass after the renames.
- README: no change needed; it holds none of the changed wording ("placeholder codes" went with ticket 04).
- Finish checks run and passed: ruff check, ruff format --check, ty check, `uv run pytest` (2237 passed, 2 skipped), both minimum-versions runs (1930 passed, 10 skipped without the extra; 2237 passed with it), the MCP smoke test and the benchmark (`_keys.py` was renamed in).
