# 10: Docstrings and descriptions that match the code

What to build: every docstring and field description that disagrees with the code is fixed, as are the small record inconsistencies and the differing conversion advice.

Read these sections of `.scratch/agent-experience/spec.md`: Docstrings and descriptions that disagree with the code. This ticket covers user story 25. Findings: STD-25, STD-32, STD-33, STD-45, STD-48 (details in `.scratch/audit/standards-review.md`).

Blocked by: 06

Status: ready-for-agent

- [ ] Each case STD-25 lists is fixed, and the Comments note each one.
- [ ] Advice on converting a column kind is worded the same way in every analysis (STD-45).
- [ ] Each small inconsistency STD-48 lists is fixed, or the Comments record why it stays.
- [ ] The docs say that `datetime_format` results depend on the process locale for locale-dependent directives (STD-32).
- [ ] The `schema` argument's JSON Schema requires at least one entry (STD-33).
- [ ] The finish checks pass.

## Comments
