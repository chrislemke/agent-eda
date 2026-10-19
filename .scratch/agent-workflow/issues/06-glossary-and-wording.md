# 06: Glossary and wording

What to build: the glossary defines every term the analyses rely on, and code, tests and docs stop using the words it avoids.

Read these sections of `.scratch/agent-workflow/spec.md`: Glossary and wording. This ticket covers user stories 25 and 26. Findings: DOC-16, SPEC-19, STD-23, STD-24.

Blocked by: 04

Status: ready-for-agent

- [ ] `GLOSSARY.md` defines each of these, with an Avoid line in the existing format:
  - Limitation
  - First look
  - Level
  - Native missing value
  - Compatible, convertible and invalid value
  - Key, primary key and candidate key
  - Target, class and label
  - Sentinel candidate
- [ ] Each definition matches what the code does. For example, a level is a distinct non-missing value as pandas `value_counts` tells them apart.
- [ ] The words STD-23 lists are gone from field descriptions, names and tests.
- [ ] Unqualified "server" is gone from the 8 test names and the private docstrings STD-24 lists, and the MCP instructions no longer say that every MCP tool runs an analysis.
- [ ] The finish checks pass.

## Comments
