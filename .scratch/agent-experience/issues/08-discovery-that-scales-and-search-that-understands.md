# 08: Discovery that scales, and search that understands

What to build:
- `guide()` and the MCP instructions give an overview that doesn't grow with every analysis;
- docstrings name at most three alternatives and point to the next step;
- search understands agents' words, marks weak matches and keeps its text bounded.

Read these sections of `.scratch/agent-experience/spec.md`: Discovery and search, and test group 6. This ticket covers user stories 21 to 23. Findings: DES-2, DOC-4, DES-12, DOC-13, STD-21, STD-30, AX-10, AX-22, CORE-15, CORE-16, SPEC-11, STD-34.

Blocked by: 05

Status: done

- [x] One overview builder serves `guide()` and the MCP instructions. It covers the areas with their counts and a line each from the subpackage docstrings, the first-look order, and pointers to `list_analyses` and `search_analyses`.
- [x] The MCP instructions hold no Python module headings.
- [x] The guide and the MCP instructions each keep within 60 lines with 11 analyses, and with 60 stand-in analyses registered through the test hook.
- [x] Every `When to use:` names its trigger and at most three alternatives. `describe_dataset` no longer lists every analysis.
- [x] See Also names only analyses and includes each analysis's natural next step. A test checks the "at most three" rule.
- [x] AGENTS.md's rule on updating neighbours says the same.
- [x] Search weighs generic words (column(s), values, rows, data, count) low and drops one-letter tokens. It gains the synonyms and tags from the spec's search table.
- [x] Weak matches are marked as weak, and the query echo and the listed terms are bounded. An 8,000-character query prints within the 40-line cap.
- [x] The search acceptance table covers every query in AX-10, CORE-16, SPEC-11 and STD-34, and each gives the right analysis first, or no match where none fits.
- [x] The guide says how `pd.read_csv`'s NA defaults turn "N/A" and "NULL" into missing values, and how to keep them.
- [x] The Comments record which Discovery decisions of the first-look and mcp-server specs this changes.
- [x] The finish checks pass.

## Comments

What the ticket settled that the spec left open:
- **Overview.** `discovery.overview(surface)` builds the areas (an area is the second part of a module path: `data`, `statistics`; its line is the first paragraph of the subpackage docstring), the first-look order and the pointers to `search_analyses` and `list_analyses`. `guide()` and `_mcp_instructions` add their own text around it. With 11 analyses the guide is 46 lines and the MCP instructions 45; with 60 stand-ins (listed through the registry's test hook, `agent_eda.mcp._stand_ins`, reached by `tests/serving.py`) both keep their length, and only the counts change. The guide's masking sentence now says `list_analyses` marks the analyses that support masking, since the guide no longer lists them. GLOSSARY.md gains "Area".
- **"At most three".** `When to use:` names at most three other analyses, and `See Also:` lists one to three. A contract test checks both, that no discovery function is named, and that each first-look analysis lists the next first-look step in See Also (STD-30). `SEE_ALSO_UNTIL_08` is deleted. The neighbour tests that required `describe_dataset` to point to six analyses, or `summarize_numeric` to point to `profile_missing_values`, now check the pointers that remain. AGENTS.md and `docs/agents/new-analysis.md` say the same.
- **Search scoring.** A tag that the query holds whole (its non-generic words, or all of them when all are generic) scores 5, above a name (4). Generic words (column, value, row, data, count) score at most 1 elsewhere, and one-letter words are dropped. Only the first 20 distinct words are searched and listed, with `term_count` giving the total. New synonym groups: `missingness` for missing, and `mask, PII, privacy, anonymize, redact, hide, sensitive, personal` for masking. New tags on `describe_dataset`, `summarize_numeric`, `summarize_categorical`, `profile_missing_values` and `compute_association_matrix`.
- **Weak matches.** `SearchMatch.weak` is true when only generic words matched or the words matched only in the When to use: text. The text marks the line "(weak match)" and adds a line when every match shown is weak.
- **"first look" and "eda".** They give no matches and `SearchListing.first_look` true, since no single analysis answers them; the text points to `ae.guide()` (the MCP instructions over MCP). "leakage" and "n" find nothing: no analysis covers leakage, and one-letter words are dropped.
- **Bounded text.** The query echo is cut to 40 characters, each word to 30, matched words to 6 per line, and matches are dropped from the text until it fits 40 lines (the record keeps all of them). A cut query's "More:" line drops the Python example, which would have to repeat the whole query.
- **Acceptance table.** `SEARCH_ACCEPTANCE` covers every query of AX-10, CORE-16, SPEC-11 and STD-34 and the spec's table. An expected value may be a tuple (the six masking analyses for PII and privacy) or None (no match).
- **NA defaults (AX-22).** The guide's "Reading a CSV" paragraph and the README say `pd.read_csv` turns "N/A" and "NULL" into missing values and `keep_default_na=False` keeps them as text.

Discovery decisions of the earlier specs that this changes:
- first-look spec, Discovery: `guide()` no longer prints "every analysis grouped by module with its one-line summary"; it prints the areas with counts. Its test 5 ("the guide names every analysis") becomes "names the 11 first-look analyses in order, and keeps its length with 60 analyses". The first-look order stays. Search keeps "no match points to the guide" and gains weak matches and the first-look pointer.
- mcp-server spec, Discovery: the instructions no longer "name every registered analysis" (its test 7), and hold no module headings; they carry the same overview as the guide.

Not done: STD-34's second half (a `project=` string should suggest `ae.Project.load(...)`) is an argument message, not discovery, and neither the spec nor a ticket covers it; only its search half is in this ticket's boxes. It still needs an owner.

Checks: ruff check, ruff format --check, ty, `uv run pytest` (3,463 passed), both minimum-version runs (2,922 and 3,463 passed), and the MCP stdio smoke test pass. The benchmark was not run: nothing here has a cost that grows with rows or columns.
