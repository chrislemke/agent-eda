# 08: Test suite upkeep

What to build:
- one helper module for the MCP tests;
- one golden test per text template, with structural assertions elsewhere;
- a README test that finds its JSON block by content;
- one interpreter for the packaging import checks;
- clearer names.

Read these sections of `.scratch/deepening/spec.md`: Test suite upkeep. This ticket covers user story 8. Findings: TEST-12 (README part), TEST-13, TEST-14, TEST-17, STD-58, STD-59.

Blocked by: none

Status: ready-for-agent

- [ ] One helper module serves the MCP tests: the client call, the tool listing, file writing, the Python reference run and an independent reference read. No copies remain across test modules (TEST-14).
- [ ] Each text template has one golden test, and other text tests assert structure, codes, fields and key fragments.
- [ ] Rewording the footer from "have the rest" to "hold the rest" fails at most 3 tests. The Comments record the count.
- [ ] `tests/test_readme.py` selects the MCP JSON block by its content.
- [ ] The packaging import checks run in one interpreter, and the Comments record the time saved.
- [ ] Names that STD-58 found unclear are renamed.
- [ ] The finish checks pass.

## Comments
