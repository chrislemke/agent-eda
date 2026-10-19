# 06: Statuses and codes agents can branch on

What to build:
- only reachable statuses are declared and listed;
- status reason codes and limitation codes come from documented lists;
- exclusion reasons are typed;
- every null and every check that didn't run carries a limitation;
- data without rows gives one status everywhere.

Read these sections of `.scratch/agent-experience/spec.md`: Names, statuses and codes (from `failed` on), and test group 4. This ticket covers user stories 13 and 14. Findings: AX-20, AX-28, DES-11, STD-26, STD-27, STD-29, STD-31, TEST's questionable decision on `failed`, and STD's questionable decisions on statuses and nulls.

Blocked by: 05

Status: ready-for-agent

- [ ] No analysis declares `failed`, and the 11 `Statuses:` sections drop it. `Status` still defines it.
- [ ] `guide()`, the README and the MCP instructions list only statuses that some registered analysis can return, computed from the registry. A test checks this.
- [ ] `StatusReason.code` is a `StatusReasonCode` Literal with one comment per code, and the six spellings of "nothing to work on" are merged.
- [ ] Each analysis's `Statuses:` entries name their reason codes, and `describe_analysis` shows them. `_check_result` raises when a result's reason code isn't declared.
- [ ] `describe_analysis` and the generated README Reference show each limitation code's meaning.
- [ ] `excluded_by_reason` is typed, with one spelling per reason.
- [ ] Checks that didn't run leave a limitation (STD-29). `infer_column_semantics`' null type has one (STD-31). `profile_target`'s task-dependent nulls carry one.
- [ ] `find_duplicate_rows`' `duplicate_share` is null under `not_applicable`.
- [ ] Data without rows gives `insufficient_data` in every analysis, `validate_column_types` included.
- [ ] The CHANGELOG lists the changes.
- [ ] The finish checks pass.

## Comments
