# 08: Discovery that scales, and search that understands

What to build:
- `guide()` and the MCP instructions give an overview that doesn't grow with every analysis;
- docstrings name at most three alternatives and point to the next step;
- search understands agents' words, marks weak matches and keeps its text bounded.

Read these sections of `.scratch/agent-experience/spec.md`: Discovery and search, and test group 6. This ticket covers user stories 21 to 23. Findings: DES-2, DOC-4, DES-12, DOC-13, STD-21, STD-30, AX-10, AX-22, CORE-15, CORE-16, SPEC-11, STD-34.

Blocked by: 05

Status: ready-for-agent

- [ ] One overview builder serves `guide()` and the MCP instructions. It covers the areas with their counts and a line each from the subpackage docstrings, the first-look order, and pointers to `list_analyses` and `search_analyses`.
- [ ] The MCP instructions hold no Python module headings.
- [ ] The guide and the MCP instructions each keep within 60 lines with 11 analyses, and with 60 stand-in analyses registered through the test hook.
- [ ] Every `When to use:` names its trigger and at most three alternatives. `describe_dataset` no longer lists every analysis.
- [ ] See Also names only analyses and includes each analysis's natural next step. A test checks the "at most three" rule.
- [ ] AGENTS.md's rule on updating neighbours says the same.
- [ ] Search weighs generic words (column(s), values, rows, data, count) low and drops one-letter tokens. It gains the synonyms and tags from the spec's search table.
- [ ] Weak matches are marked as weak, and the query echo and the listed terms are bounded. An 8,000-character query prints within the 40-line cap.
- [ ] The search acceptance table covers every query in AX-10, CORE-16, SPEC-11 and STD-34, and each gives the right analysis first, or no match where none fits.
- [ ] The guide says how `pd.read_csv`'s NA defaults turn "N/A" and "NULL" into missing values, and how to keep them.
- [ ] The Comments record which Discovery decisions of the first-look and mcp-server specs this changes.
- [ ] The finish checks pass.

## Comments
