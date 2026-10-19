# 04: find_candidate_keys: limits and float columns

What to build: `find_candidate_keys` finishes for any `max_key_size`, its budget message suggests a fix that works, and float measurements are no longer offered as primary keys unless named.

Read these sections of `.scratch/correctness/spec.md`: Keys, limits and threads, and test group 4. This ticket covers user stories 8 and 14. Findings: AN2-4, CORE-3, MCP-3, STD-38 (the key-search loop), AN2-10, AX-11, STD-15.

Blocked by: 01

Status: ready-for-agent

- [ ] `find_candidate_keys(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]}), max_key_size=sys.maxsize)` finishes in under one second.
- [ ] Over MCP, a call with `"max_key_size": 100000000000000000000` returns, and a `ping` sent after it is answered.
- [ ] On 40 columns with `max_key_size=40`, the `budget_exceeded` reason suggests the largest `max_key_size` that fits `max_combinations`. It never suggests an impossible `max_combinations`.
- [ ] Non-integral float columns are left out by default, with a limitation that names them. Naming them in `columns` includes them. Float columns holding only whole numbers stay candidates.
- [ ] The docstring and the README describe the float rule. The Comments record that this changes first-look ticket 14.
- [ ] The docstring no longer claims its counts are what `df.duplicated` gives for an object column holding None and NaN (STD-15).
- [ ] The finish checks pass.

## Comments
