# 07: Statuses, codes and caps from one source

What to build: registration reads statuses and finding codes from the docstring, and each cap's number lives only in its marker, from which the field descriptions get their cap clauses.

Read these sections of `.scratch/deepening/spec.md`: Single sources, and test groups 1 and 6. This ticket covers user stories 7 and 9. Findings: DES-13, DES-14.

Blocked by: 05

Status: ready-for-agent

- [ ] `@analysis` no longer takes `statuses=` or `finding_codes=`. Registration reads both from the parsed docstring, and raises for an unknown status or a malformed code. Test both with a broken stand-in.
- [ ] The cap numbers 10, 5 and 100 appear once each, in the markers. Truncating analyses ask the result class for the limit.
- [ ] Each capped field's description gets its cap clause from its marker. `check_contract` asserts that a capped field holds `min(cap, total)` entries.
- [ ] The baseline comparison passes unchanged; descriptions may differ only where the generated cap clause rewords them, and the Comments list those.
- [ ] The finish checks pass.

## Comments
